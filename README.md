# scout-mcp

Assistant GM for AI assistants: an [MCP](https://modelcontextprotocol.io) server that lets Claude,
ChatGPT and other MCP clients read Assistant GM data as the person who connected them.

Connector URL: **https://mcp.tbscouting.com/mcp**

## How it works

- **Sign-in.** scout-api (`api.tbscouting.com`) is the OAuth server. An assistant finds it from this
  server's protected resource metadata (`/.well-known/oauth-protected-resource/mcp`). The person
  approves the connection on the portal's consent page, and the assistant gets a token issued for
  this server.
- **Data.** Everything comes from the public `/v1` API, never the database. The server trades the
  assistant's token for a `/v1` token on the same connection (RFC 8693 token exchange, client
  `tbs_mcp_server`, authenticated with `MCP_EXCHANGE_SECRET`). The assistant's own token is never
  forwarded, and `/v1` refuses tokens issued for this server.
- **Limits.** The same as `/v1`: the connection's scopes, the account's API access, and 60 requests a
  minute per connection. Removing the connection under Connected apps stops both tokens.

Tools are read-only: `whoami`, `search_players`, `get_player`, `get_player_stats`,
`get_player_scores`, `get_player_nil`, `transfer_portal`, `my_team`, `my_roster`,
`my_team_history`, `search_hs_prospects`, `get_hs_prospect`, `get_hs_performances`.

## Running

```
pip install -r requirements.txt
MCP_EXCHANGE_SECRET=... uvicorn server:app --host 127.0.0.1 --port 8100
```

| Variable | Default | |
|---|---|---|
| `MCP_EXCHANGE_SECRET` | (required) | Same value as scout-api's `MCP_EXCHANGE_SECRET` |
| `SCOUT_API_URL` | `http://127.0.0.1:8000` | Where to call `/oauth/token` and `/v1` (the token exchange only works from this machine) |
| `SCOUT_API_PUBLIC_URL` | `https://api.tbscouting.com` | The OAuth issuer |
| `MCP_PUBLIC_URL` | `https://mcp.tbscouting.com` | This server's public URL |

In production it runs under pm2 (`scout-mcp`, `~/apps/ecosystem.config.js`) behind Caddy, and is
deployed with `bash ~/apps/deploy.sh mcp`.
