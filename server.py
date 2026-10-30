"""
Assistant GM for AI assistants: an MCP server (https://mcp.tbscouting.com/mcp) that lets Claude,
ChatGPT and other MCP clients read Assistant GM data as the person who connected them.

  Sign-in     scout-api is the OAuth authorization server. An assistant finds it through this
              server's protected resource metadata, the person approves it on the portal's
              consent page, and the assistant sends us an access token issued for this server.
  Data        everything comes from the public /v1 API, never the database. We trade the
              assistant's token for a /v1 token on the same connection (RFC 8693 token exchange,
              authenticated with MCP_EXCHANGE_SECRET) and call /v1 with that. The assistant's own
              token is never forwarded, and /v1 refuses tokens issued for this server.
  Limits      /v1's: scopes, the account's API access, 60 requests a minute per connection.
              Removing the connection in the portal stops both tokens at once.

Run: uvicorn server:app --host 127.0.0.1 --port 8100 (behind Caddy).
"""
import hashlib
import logging
import os
import time
from typing import Literal

import httpx
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

log = logging.getLogger("scout-mcp")
logging.getLogger("httpx").setLevel(logging.WARNING)   # not a line per /v1 call; the API logs those

API_URL = os.environ.get("SCOUT_API_URL", "http://127.0.0.1:8000").rstrip("/")      # where we call /v1 and /oauth/token
ISSUER = os.environ.get("SCOUT_API_PUBLIC_URL", "https://api.tbscouting.com").rstrip("/")
PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "https://mcp.tbscouting.com").rstrip("/")
EXCHANGE_SECRET = os.environ.get("MCP_EXCHANGE_SECRET", "")
CLIENT_ID = "tbs_mcp_server"
EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"

if not EXCHANGE_SECRET:
    raise SystemExit("MCP_EXCHANGE_SECRET is not set")

_http = httpx.AsyncClient(base_url=API_URL, timeout=20)


# ── Tokens ─────────────────────────────────────────────────────────────────────

class _Exchanged:
    def __init__(self, api_token: str, scopes: list[str], expires_at: float):
        self.api_token, self.scopes, self.expires_at = api_token, scopes, expires_at


# sha256(assistant's token) → its /v1 token. Tokens last 15 minutes, so this stays small.
_exchanged: dict[str, _Exchanged] = {}


def _key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class ExchangeVerifier:
    """A token is good if scout-api will exchange it; the exchange checks it was issued for us."""

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token.startswith("tbs_at_"):
            return None
        now = time.time()
        hit = _exchanged.get(_key(token))
        if not hit or hit.expires_at <= now:
            hit = await self._exchange(token)
            if not hit:
                return None
        return AccessToken(token=token, client_id=CLIENT_ID, scopes=hit.scopes, expires_at=int(hit.expires_at),
                           resource=PUBLIC_URL + "/mcp")

    async def _exchange(self, token: str) -> _Exchanged | None:
        try:
            r = await _http.post("/oauth/token", auth=(CLIENT_ID, EXCHANGE_SECRET), data={
                "grant_type": EXCHANGE_GRANT, "subject_token": token, "subject_token_type": ACCESS_TOKEN_TYPE,
                "resource": ISSUER + "/v1",
            })
        except httpx.HTTPError:
            log.exception("token exchange failed")
            return None
        if r.status_code != 200:
            if r.status_code != 400:      # 400 is just a bad or expired token
                log.warning("token exchange: HTTP %s %s", r.status_code, r.text[:200])
            return None
        d = r.json()
        now = time.time()
        for k in [k for k, v in _exchanged.items() if v.expires_at <= now]:
            del _exchanged[k]
        hit = _Exchanged(d["access_token"], d.get("scope", "").split(), now + d["expires_in"] - 5)
        _exchanged[_key(token)] = hit
        return hit


# ── Calling /v1 ────────────────────────────────────────────────────────────────

class ApiError(ToolError):
    """Shown to the assistant as the tool's error message."""


async def _get(path: str, **params) -> dict:
    """GET /v1{path} as the connected person. Errors come back as messages the assistant can relay."""
    return await _call("GET", path, params=params)


async def _post(path: str, body: dict) -> dict:
    """POST /v1{path} (NIL contributions) as the connected person."""
    return await _call("POST", path, json={k: v for k, v in body.items() if v is not None})


async def _call(method: str, path: str, params: dict | None = None, json: dict | None = None) -> dict:
    token = get_access_token()
    hit = _exchanged.get(_key(token.token)) if token else None
    if not hit:
        raise ApiError("Not signed in. Reconnect Assistant GM in your assistant's settings.")
    query = {k: v for k, v in (params or {}).items() if v is not None}
    try:
        r = await _http.request(method, "/v1" + path, params=query, json=json, headers={"Authorization": "Bearer " + hit.api_token})
    except httpx.HTTPError:
        log.exception("%s /v1%s failed", method, path)
        raise ApiError("Assistant GM is unavailable right now. Try again shortly.")
    if r.status_code == 200:
        return r.json()
    try:
        body = r.json()
    except ValueError:
        body = {}
    message = body.get("error_description") or body.get("detail") or body.get("message") or f"HTTP {r.status_code}"
    if r.status_code == 401:
        _exchanged.pop(_key(token.token), None)
        raise ApiError(f"{message}. Reconnect Assistant GM in your assistant's settings.")
    if r.status_code == 403 and body.get("error") == "insufficient_scope":
        raise ApiError(f"{message}; this connection wasn't given it. Your account's API owner controls what's available.")
    if r.status_code == 429:
        raise ApiError(f"Rate limited ({message}). Wait {r.headers.get('retry-after', '60')} seconds.")
    raise ApiError(str(message))


# ── Server ─────────────────────────────────────────────────────────────────────

mcp = MCPServer(
    name="assistant-gm",
    title="Assistant GM",
    website_url="https://portal.tbscouting.com",
    instructions="""\
Assistant GM is Travis Branham Scouting Service's college basketball scouting portal. These tools read it
as the person who connected it, limited to what their account allows. Everything is read-only except
the two NIL contribution tools.

- To find players by stats, eligibility or NIL ("a guard for next year who shoots 40% from three, top 5 by
  NIL"), use screen_players: one call, not a search followed by per-player lookups. "Next year" is
  for_season="next" (the server applies the recruiting calendar). With a percentage filter always add a
  volume minimum (e.g. min three_pa 3, or games 15), and say which stats season the numbers are from.
  For a future season the list is players projected to have eligibility, not confirmed returners: NBA and
  transfer decisions come later (from around January). Pass on the response's eligibility_note.
- College players: search_players, then get_player / get_player_stats / get_player_scores / get_player_nil
  with a search result's id as player_id. transfer_portal lists the transfer board.
- The account's own team: my_team, my_roster, my_team_history.
- High school prospects (classes 2026-2030): search_hs_prospects, get_hs_prospect, get_hs_performances.
  For prospects, the scouts' performance grades and notes matter far more than box-score stats.
- GM score is the portal's headline rating. Levels: HM = high major, MM = mid major, LM = low major.
- NIL figures are estimates, each labelled expert or model (with the model version), the NIL season and
  when it was estimated. Present them as estimates, never as facts, and say when they're from: they
  change as new information comes in.
- NIL access: the team's own roster (get_team_nil) with the basic NIL permission; any player, history and
  the whole-market export (list_nil_estimates, 200 per page) with the nil:market upgrade. For a long list,
  page through list_nil_estimates rather than calling get_player_nil per player.
- submit_nil_estimate and report_nil_deal send the person's own information to TB Scouting for review.
  Only use them when the person asks you to, with figures they gave you; confirm the player and amount
  first. Never submit figures you inferred or made up.
- If a tool says a scope is missing or the connection is gone, tell the person; don't retry in a loop.""",
    token_verifier=ExchangeVerifier(),
    auth=AuthSettings(issuer_url=ISSUER, resource_server_url=PUBLIC_URL + "/mcp",
                      service_documentation_url=ISSUER + "/v1/docs",
                      validate_token_resource=False),   # the exchange refuses tokens not issued for us
)

READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
Position = Literal["G", "F", "C"]


@mcp.tool(annotations=READ)
async def whoami() -> dict:
    """The connected person, their account, the account's linked team, and what this connection can read."""
    return await _get("/me")


@mcp.tool(annotations=READ)
async def search_players(
    q: str | None = None,
    team: str | None = None,
    conference: str | None = None,
    position: Position | None = None,
    class_year: Literal["FR", "SO", "JR", "SR", "GR"] | None = None,
    status: Literal["active", "portal", "committed", "prospect", "withdrawn"] | None = None,
    sort: Literal["gm_score", "scout_score", "name", "ppg", "rpg", "apg"] = "gm_score",
    page: int = 1,
    per_page: int = 25,
) -> dict:
    """Search college players. q = part of a name (2+ letters); team = exact current team (e.g. "Kentucky");
    conference = exact conference (e.g. "SEC"). Sorted by GM score unless sort says otherwise.
    Each result's id is the player_id the other player tools take."""
    return await _get("/players", q=q, team=team, conference=conference, position=position, class_year=class_year,
                      status=status, sort=sort, page=page, per_page=per_page)


@mcp.tool(annotations=READ)
async def screen_players(
    position: Position | None = None,
    for_season: str | None = None,
    status: Literal["active", "portal", "committed", "prospect", "withdrawn", "available"] | None = None,
    conference: str | None = None,
    team: str | None = None,
    class_year: Literal["FR", "SO", "JR", "SR", "GR"] | None = None,
    min: dict[str, float] | None = None,
    max: dict[str, float] | None = None,
    sort: str = "gm_score",
    order: Literal["desc", "asc"] | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict:
    """Find college players by stats, eligibility and NIL in one call, with each player's latest-season
    stats, projected class and NIL estimate (as the account allows).
    for_season: "next" (players projected to have eligibility then, still in college), "current", or "2027-28".
    status "available" = in the transfer portal, uncommitted. conference: "HM", "MM", "LM" or a name.
    min / max: stat bounds, e.g. min={"three_pct": 40, "three_pa": 3}. Stats: games, mpg, ppg, rpg, apg,
    spg, bpg, tov, fg_pct, three_pct, three_pa, ft_pct, efg_pct, ts_pct, usg_pct, ast_pct, trb_pct, bpm, per.
    Shooting percentages take 40 or 0.40. sort: "nil" (needs the nil:market upgrade), "gm_score",
    "scout_score", "name", or any stat; order defaults to desc (asc for name). per_page up to 100."""
    bounds = {f"min_{k}": v for k, v in (min or {}).items()} | {f"max_{k}": v for k, v in (max or {}).items()}
    return await _get("/screen", position=position, for_season=for_season, status=status, conference=conference,
                      team=team, class_year=class_year, sort=sort, order=order, page=page, per_page=per_page, **bounds)


@mcp.tool(annotations=READ)
async def get_player(player_id: int) -> dict:
    """A college player's profile: bio, team, status, transfer portal details, scores when available."""
    return await _get(f"/players/{player_id}")


@mcp.tool(annotations=READ)
async def get_player_stats(player_id: int) -> dict:
    """A college player's season stats (box score and advanced), one entry per season."""
    return await _get(f"/players/{player_id}/stats")


@mcp.tool(annotations=READ)
async def get_player_scores(player_id: int) -> dict:
    """A college player's scout scores and grades, including the GM score and projected level."""
    return await _get(f"/players/{player_id}/scores")


@mcp.tool(annotations=READ)
async def get_player_nil(player_id: int) -> dict:
    """A college player's current NIL estimate: kind (expert or model, with model version), NIL season,
    low/mid/high US dollars per year, and when it was first estimated. Players on the account's own
    roster need the basic NIL permission; anyone else needs the nil:market upgrade."""
    return await _get(f"/players/{player_id}/nil")


@mcp.tool(annotations=READ)
async def get_player_nil_history(player_id: int) -> dict:
    """How a college player's NIL estimate has changed over time, oldest first, with when each figure
    applied (estimated_since → replaced_at)."""
    return await _get(f"/players/{player_id}/nil/history")


@mcp.tool(annotations=READ)
async def get_team_nil() -> dict:
    """The current NIL estimate for every player on the account's own team roster."""
    return await _get("/team/nil")


@mcp.tool(annotations=READ)
async def list_nil_estimates(
    season: str | None = None,
    kind: Literal["expert", "model"] | None = None,
    team: str | None = None,
    conference: str | None = None,
    position: Position | None = None,
    status: Literal["active", "portal", "committed", "prospect", "withdrawn"] | None = None,
    min_mid: int | None = None,
    max_mid: int | None = None,
    sort: Literal["mid", "name", "changed"] = "mid",
    as_of: str | None = None,
    changed_since: str | None = None,
    page: int = 1,
    per_page: int = 200,
) -> dict:
    """NIL estimates for every college player (needs the nil:market upgrade), up to 200 per page, highest
    first. season = NIL season like "2026-27"; conference = "HM", "MM", "LM" or a conference name;
    min_mid/max_mid in US dollars. as_of (ISO 8601 UTC, e.g. "2026-10-01T00:00:00Z") exports a past moment;
    changed_since returns only figures that changed since then. The response's as_of says when the data is from."""
    return await _get("/nil", season=season, kind=kind, team=team, conference=conference, position=position,
                      status=status, min_mid=min_mid, max_mid=max_mid, sort=sort, as_of=as_of,
                      changed_since=changed_since, page=page, per_page=per_page)


WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)


@mcp.tool(annotations=WRITE)
async def submit_nil_estimate(player_id: int, amount: int) -> dict:
    """Submit the person's own estimate of a college player's NIL value, in US dollars per year, as a
    Coach's Market vote for the current NIL season. One per player per season; submitting again replaces
    it. TB Scouting reviews it before it counts. Only when the person asks, with their figure."""
    return await _post(f"/players/{player_id}/nil/votes", {"amount": amount})


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
async def report_nil_deal(
    player_id: int,
    amount: int,
    kind: Literal["deal", "agent_ask"] = "deal",
    basis: Literal["per_year", "total"] = "per_year",
    years: int | None = None,
    amount_high: int | None = None,
    season: str | None = None,
    source: str | None = None,
    source_url: str | None = None,
    note: str | None = None,
) -> dict:
    """Report a NIL deal (or an agent's ask) the person knows about for a college player, in US dollars.
    basis "total" needs years. amount_high for a stated range. source / source_url: where it comes from.
    TB Scouting reviews it before it counts. Only when the person asks, with their information."""
    return await _post(f"/players/{player_id}/nil/reports", {
        "amount": amount, "kind": kind, "basis": basis, "years": years, "amount_high": amount_high,
        "season": season, "source": source, "source_url": source_url, "note": note})


@mcp.tool(annotations=READ)
async def my_nil_contributions() -> dict:
    """The person's NIL estimates (Coach's Market votes) and reported deals, with their review status."""
    return await _get("/nil/contributions")


@mcp.tool(annotations=READ)
async def transfer_portal(
    status: Literal["available", "committed"] | None = None,
    position: Position | None = None,
    conference: str | None = None,
    q: str | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict:
    """The transfer board: players in or through the transfer portal, best GM score first.
    status "available" = in the portal and uncommitted. conference = "HM", "MM" or "LM" for a level, or a conference name."""
    return await _get("/portal", status=status, position=position, conference=conference, q=q, page=page, per_page=per_page)


@mcp.tool(annotations=READ)
async def my_team() -> dict:
    """The account's linked college team."""
    return await _get("/team")


@mcp.tool(annotations=READ)
async def my_roster(season: str | None = None) -> dict:
    """The linked team's roster. season like "2026-27"; defaults to the current season."""
    return await _get("/team/roster", season=season)


@mcp.tool(annotations=READ)
async def my_team_history() -> dict:
    """The linked team's history by season."""
    return await _get("/team/history")


@mcp.tool(annotations=READ)
async def search_hs_prospects(
    class_year: int | None = None,
    grade: Literal["HM+", "HM", "HM-", "MM+", "MM", "MM-", "LM+", "LM"] | None = None,
    position: str | None = None,
    q: str | None = None,
    page: int = 1,
    per_page: int = 25,
) -> dict:
    """Search high school prospects (classes 2026-2030). class_year = graduating class, e.g. 2027;
    grade = projected level; position like "PG", "Wing", "Big"; q = part of a name.
    Each result's id is the player_id the other prospect tools take."""
    return await _get("/hs/players", **{"class": class_year}, grade=grade, position=position, q=q, page=page, per_page=per_page)


@mcp.tool(annotations=READ)
async def get_hs_prospect(player_id: int) -> dict:
    """A high school prospect's profile and projected level."""
    return await _get(f"/hs/players/{player_id}")


@mcp.tool(annotations=READ)
async def get_hs_performances(player_id: int) -> dict:
    """A high school prospect's evaluated performances: the scouts' grades and notes by event.
    These expert evaluations are the most important information on a prospect."""
    return await _get(f"/hs/players/{player_id}/performances")


@mcp.custom_route("/", methods=["GET"])
async def home(request: Request) -> PlainTextResponse:
    return PlainTextResponse("Assistant GM MCP server. Add https://mcp.tbscouting.com/mcp to your assistant as a connector.\n")


@mcp.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
async def resource_metadata_root(request: Request) -> JSONResponse:
    """The SDK serves this at …/oauth-protected-resource/mcp (RFC 9728); some clients only look at the root."""
    return JSONResponse({"resource": PUBLIC_URL + "/mcp", "authorization_servers": [ISSUER],
                         "bearer_methods_supported": ["header"], "resource_name": "Assistant GM",
                         "resource_documentation": ISSUER + "/v1/docs"})


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok\n")


app = mcp.streamable_http_app(
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=[PUBLIC_URL.split("://", 1)[1], "127.0.0.1:*", "localhost:*"],
        allowed_origins=[PUBLIC_URL, "https://claude.ai", "https://chatgpt.com"],
    ),
)
