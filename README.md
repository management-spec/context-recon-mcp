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
- Idle auto-exit after 15 minutes
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

Optional process label:

```bash
CONTEXT_RECON_PROCESS_NAME=Context_Recon_MCP
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
      "OAUTH_CALLBACK_HOST": "127.0.0.1"
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
        "OAUTH_CALLBACK_HOST": "127.0.0.1"
      }
    }
  }
}
```

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
