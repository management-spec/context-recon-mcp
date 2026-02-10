#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_ROOT="${1:-${XDG_DATA_HOME:-$HOME/.local/share}/context_recon_mcp}"
BIN_DIR="${HOME}/.local/bin"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing Python runtime: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Missing required tool: rsync" >&2
  exit 1
fi

mkdir -p "${INSTALL_ROOT}" "${BIN_DIR}"

rsync -a --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".pytest_cache/" \
  --exclude ".uv-cache/" \
  --exclude ".context_engine/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  "${SOURCE_ROOT}/" "${INSTALL_ROOT}/"

"${PYTHON_BIN}" -m venv "${INSTALL_ROOT}/.venv"
"${INSTALL_ROOT}/.venv/bin/pip" install --upgrade pip >/dev/null
"${INSTALL_ROOT}/.venv/bin/pip" install -e "${INSTALL_ROOT}" >/dev/null

cat > "${BIN_DIR}/context-recon-mcp" <<EOF
#!/usr/bin/env bash
set -euo pipefail
INSTALL_ROOT="${INSTALL_ROOT}"
PYTHON_BIN="\${INSTALL_ROOT}/.venv/bin/python"
if [[ ! -x "\${PYTHON_BIN}" ]]; then
  echo "Context_Recon_MCP error: missing venv python at \${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -z "\${GEMINI_CONTEXT_WORKSPACE_ROOT:-}" ]]; then
  for candidate in "\${MCP_WORKSPACE_ROOT:-}" "\${CODEX_WORKSPACE_ROOT:-}" "\${CLAUDE_PROJECT_DIR:-}" "\${PWD}"; do
    if [[ -n "\${candidate}" && "\${candidate}" != "/" ]]; then
      export GEMINI_CONTEXT_WORKSPACE_ROOT="\${candidate}"
      break
    fi
  done
fi
export GEMINI_CONTEXT_SERVER_HOME="\${INSTALL_ROOT}"
export CONTEXT_RECON_ORPHAN_SHUTDOWN="\${CONTEXT_RECON_ORPHAN_SHUTDOWN:-false}"
export CONTEXT_RECON_PROCESS_NAME="\${CONTEXT_RECON_PROCESS_NAME:-Context_Recon_MCP}"
cd "\${INSTALL_ROOT}"
exec -a "\${CONTEXT_RECON_PROCESS_NAME}" "\${PYTHON_BIN}" "\${INSTALL_ROOT}/src/server.py"
EOF

chmod +x "${BIN_DIR}/context-recon-mcp"

cat <<EOF
Installed Context_Recon_MCP.

Install root: ${INSTALL_ROOT}
Launcher: ${BIN_DIR}/context-recon-mcp
Dashboard: http://127.0.0.1:8765

Next:
1) Ensure ${BIN_DIR} is on PATH.
2) Configure your MCP client command as: context-recon-mcp
3) Set env for OAuth mode:
   GEMINI_DEFAULT_AUTH_TYPE=oauth-personal
   NO_BROWSER=true
   OAUTH_CALLBACK_HOST=127.0.0.1
EOF
