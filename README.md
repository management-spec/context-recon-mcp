# Context_Recon_MCP

> **A lightweight, local MCP server for deterministic context retrieval**  
> Built to stay simple, predictable, and inexpensive to run.

We like putting our dollars where they actually count.

This tool grew out of that mindset, and we’re sharing it so others can do more with less.

---

## 🚦 Legal & Usage

Please read this section before using the project:

- **No affiliation** with Google, Anthropic, OpenAI, or any other vendor  
- You bring and manage your **own Gemini CLI credentials**
- You are responsible for **usage, billing, and compliance**
- This repository is **source code only** (not hosted, not commercial)
- Provided as-is, with no dedicated support or roadmap commitments. 
  If you find ways to improve it, feel free to share them.

---

## 🧭 What This Server Does

Context_Recon_MCP is focused on **retrieval only**.

It helps tools and agents:

- Inspect a local codebase
- Search deterministically
- Extract stable, citeable excerpts

It intentionally does **not** modify files or mutate your project.

### Scope at a glance

- 🖥️ Runs locally only  
- 🧠 Uses Gemini CLI as context retrieval engine  
- 🧾 Deterministic search and excerpting for stability and auditability  
- 🔒 Retrieval-only surface (no code editing)  
- 🚫 Dot-prefixed paths excluded by default  
- 🗂️ Skips common system/cache directories during broad scans

---

## 🧰 Tool Surface

A small, focused set of MCP tools:

| Tool | Description |
|----|----|
| `context.project_overview` | Directory orientation and key file hints |
| `context.index_inspection` | Index status, limits, Gemini status, dashboard URL |
| `context.code_search` | Lexical discovery across the project |
| `context.relevant_code` | Ranked excerpts for a query |
| `context.context_recon` | Single-tool semantic retrieval wrapper |
| `context.file_slice` | Authoritative excerpts with hash and mtime |
| `context.tool_update` | Pulls latest server changes from git (`pull --ff-only`) |
| `context.tool_cleanup` | Terminates sibling/orphaned server processes |

---

## 🖥️ Local Dashboard

**Default URL:**  
```
http://127.0.0.1:8765
```

Served from `ui/index.html`.

### Dashboard features

- Connection and status checks
- Request counts and token totals
- Usage and quota info from local Gemini CLI auth
- File explorer, query console, snippet inspector
- Click-to-load recent MCP activity
- Frequently used query chips

### Runtime behavior

- UI refresh runs **only when open**
- No model connectivity checks on refresh
- Visibility-aware background polling
- Gemini auto-poll disabled by default
- Orphan auto-exit when parent process is gone (`ppid == 1`)
- Idle auto-exit after 15 minutes
- Optional startup auto-update (`git pull --ff-only`)
- Optional periodic auto-update poll (default every 5 minutes)
- Conservative token usage defaults

If the dashboard is closed, no UI refresh runs.  
For terminal-only inspection, use `context.index_inspection`.

---

## 📦 Install

```bash
./scripts/install_system.sh
```

### Default paths

- `${XDG_DATA_HOME:-$HOME/.local/share}/context_recon_mcp`
- `~/.local/bin/context-recon-mcp`

### Required environment (OAuth mode)

```bash
GEMINI_DEFAULT_AUTH_TYPE=oauth-personal
NO_BROWSER=true
OAUTH_CALLBACK_HOST=127.0.0.1
```

### Optional environment (required for some Workspace/Code Assist accounts)

```bash
GOOGLE_CLOUD_PROJECT=<your-gcp-project-id>
GOOGLE_CLOUD_PROJECT_ID=<your-gcp-project-id>
```

Optional process label:

```bash
CONTEXT_RECON_PROCESS_NAME=Context_Recon_MCP
```

Optional process lifecycle controls:

```bash
# Environment variable (launcher default is true). Set false to disable orphan self-shutdown.
CONTEXT_RECON_ORPHAN_SHUTDOWN=true
```

```yaml
# config.yaml lifecycle defaults
orphan_shutdown_enabled: true
idle_shutdown_seconds: 900
```

Optional update controls:

```bash
# Optional environment overrides
CONTEXT_RECON_AUTO_UPDATE_ON_START=true
CONTEXT_RECON_UPDATE_REMOTE=origin
CONTEXT_RECON_UPDATE_BRANCH=main
CONTEXT_RECON_UPDATE_ALLOW_DIRTY=false
CONTEXT_RECON_UPDATE_TIMEOUT_SECONDS=45
CONTEXT_RECON_UPDATE_POLL_SECONDS=300
```

```yaml
# config.yaml update defaults
auto_update_on_start: true
update_remote: "origin"
update_branch: ""   # empty means current branch
update_allow_dirty: false
update_timeout_seconds: 45
update_poll_seconds: 300
```

Notes:

- Updates use `git pull --ff-only` and will not create merge commits.
- If local changes exist and `update_allow_dirty=false`, update is skipped.
- After a successful code update, restart MCP to load the new server code.

### Process Cleanup (Stale Servers)

Preferred cleanup path is the MCP tool `context.tool_cleanup`.

Terminal fallback (keeps the newest server PID and stops older duplicates):

```bash
pids="$(pgrep -f 'context-recon-mcp/src/server.py' || true)"; keep="$(printf '%s\n' "$pids" | sort -n | tail -n 1)"; printf '%s\n' "$pids" | while IFS= read -r pid; do [ -n "$pid" ] && [ "$pid" != "$keep" ] && kill "$pid"; done
```

---

## 🔌 MCP Configuration Examples

### Command

```bash
context-recon-mcp
```

### Codex (`~/.codex/config.toml`)

```toml
[mcp_servers.Context_Recon_MCP]
command = "context-recon-mcp"
args = []
enabled = true

[mcp_servers.Context_Recon_MCP.env]
GEMINI_DEFAULT_AUTH_TYPE = "oauth-personal"
NO_BROWSER = "true"
OAUTH_CALLBACK_HOST = "127.0.0.1"
CONTEXT_RECON_ORPHAN_SHUTDOWN = "true"
CONTEXT_RECON_AUTO_UPDATE_ON_START = "true"
CONTEXT_RECON_UPDATE_POLL_SECONDS = "300"
GOOGLE_CLOUD_PROJECT = "<your-gcp-project-id>"
GOOGLE_CLOUD_PROJECT_ID = "<your-gcp-project-id>"
```

### Claude Code (`~/.claude.json`)

```json
{
  "Context_Recon_MCP": {
    "type": "stdio",
    "command": "context-recon-mcp",
    "args": [],
    "env": {
      "GEMINI_DEFAULT_AUTH_TYPE": "oauth-personal",
      "NO_BROWSER": "true",
      "OAUTH_CALLBACK_HOST": "127.0.0.1",
      "CONTEXT_RECON_ORPHAN_SHUTDOWN": "true",
      "CONTEXT_RECON_AUTO_UPDATE_ON_START": "true",
      "CONTEXT_RECON_UPDATE_POLL_SECONDS": "300",
      "GOOGLE_CLOUD_PROJECT": "<your-gcp-project-id>",
      "GOOGLE_CLOUD_PROJECT_ID": "<your-gcp-project-id>"
    }
  }
}
```

### Cursor / VS Code

```json
{
  "mcpServers": {
    "Context_Recon_MCP": {
      "command": "context-recon-mcp",
      "args": [],
      "env": {
        "GEMINI_DEFAULT_AUTH_TYPE": "oauth-personal",
        "NO_BROWSER": "true",
        "OAUTH_CALLBACK_HOST": "127.0.0.1",
        "CONTEXT_RECON_ORPHAN_SHUTDOWN": "true",
        "CONTEXT_RECON_AUTO_UPDATE_ON_START": "true",
        "CONTEXT_RECON_UPDATE_POLL_SECONDS": "300",
        "GOOGLE_CLOUD_PROJECT": "<your-gcp-project-id>",
        "GOOGLE_CLOUD_PROJECT_ID": "<your-gcp-project-id>"
      }
    }
  }
}
```

### Optional CLI Provider Overrides (Any Host)

- `CONTEXT_RECON_RERANK_PROVIDER="auto"` (or `gemini`, `claude`, `codex`)
- `CONTEXT_RECON_GEMINI_COMMAND="gemini"`
- `CONTEXT_RECON_GEMINI_ARGS="--model gemini-2.5-pro"`
- `CONTEXT_RECON_CLAUDE_COMMAND="claude"`
- `CONTEXT_RECON_CLAUDE_ARGS="--model claude-3-7-sonnet-20250219 --max-turns 1 --no-session-persistence --disable-slash-commands --tools \"\""`
- `CONTEXT_RECON_CLAUDE_PROMPT_MODE="arg"` (or `stdin`)
- `CONTEXT_RECON_CLAUDE_USAGE_SOURCE="auto"` (or `web`, `oauth`, `off`)
- `CONTEXT_RECON_CLAUDE_COOKIE_SOURCE="auto"` (or `manual`, `off`)
- `CONTEXT_RECON_CLAUDE_COOKIE_HEADER="sessionKey=..."`
- `CONTEXT_RECON_CODEX_COMMAND="codex"`
- `CONTEXT_RECON_CODEX_ARGS="-m codex-mini-latest"`
- `CONTEXT_RECON_CODEX_PROMPT_MODE="arg"` (or `stdin`)
- `CONTEXT_RECON_CODEX_USAGE_SOURCE="auto"` (or `web`, `oauth`, `off`)
- `CONTEXT_RECON_CODEX_COOKIE_SOURCE="auto"` (or `manual`, `off`)
- `CONTEXT_RECON_CODEX_COOKIE_HEADER="__Secure-next-auth.session-token=..."`
- `CONTEXT_RECON_BROWSER_COOKIE_ORDER="safari,chrome,brave,edge,chromium,firefox,opera"`

`*_ARGS` accepts either a JSON array (e.g. `["--model","gemini-2.5-pro"]`) or a shell-style string.

### Defaults + config.yaml Examples

Defaults (when you do nothing):
- Provider: `auto` (first available: `gemini` → `claude` → `codex`)
- Gemini model: whatever your Gemini CLI defaults to
- Claude model: `claude-3-7-sonnet-20250219` (via args below)
- Codex model: `codex-mini-latest`

Edit `config.yaml` to pin a provider/model without env vars:

```yaml
reranker_provider: "claude"
claude_args:
  - "--model"
  - "claude-3-7-sonnet-20250219"
  - "--max-turns"
  - "1"
  - "--no-session-persistence"
  - "--disable-slash-commands"
  - "--tools"
  - ""
```

```yaml
reranker_provider: "codex"
codex_args:
  - "-m"
  - "codex-mini-latest"
```

```yaml
reranker_provider: "gemini"
gemini_args:
  - "--model"
  - "gemini-2.5-pro"
```

If the chosen provider CLI is missing, the server auto-falls back to the next available CLI in the order above.

Env vars always override `config.yaml`.

Claude adapter notes:
- Adds `--output-format json` and `--json-schema` automatically for structured output, and default args limit turns and tools for token efficiency.
- Usage sync follows the CodexBar-style web-cookie strategy; use `*_COOKIE_HEADER` for manual cookies.

Codex adapter notes:
- Uses `codex exec` under the hood.
- Adds `--output-schema` and `--output-last-message` automatically for structured JSON output.
- Defaults to `--sandbox read-only` and `--ask-for-approval never` unless you override those flags.
- Default model is set to `codex-mini-latest` since it is optimized for Codex CLI and the Codex launch default.
- Usage sync follows the CodexBar-style web dashboard cookie path; if cookies are unavailable, the dashboard will show usage as unavailable.

Token usage tracking:
- Gemini uses CLI stats.
- Claude/Codex attempt to read usage fields from CLI JSON when available; otherwise we estimate from prompt/output size.

---

## 🧪 Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

---

## 📄 License

MIT (see `LICENSE`).

---

<sub>
Maintained by the team behind <strong>AI Stage Coach</strong>,  
a training-focused shooting analysis app built with the same cost-conscious, judgment-first approach.
Train Smarter. Improve Faster.
</sub>
