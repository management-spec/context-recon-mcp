#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BLOCKLIST_REGEX='auggie|serena|codebase-retrieval'

if grep -R --line-number -i "${BLOCKLIST_REGEX}" \
  "${ROOT}/src" "${ROOT}/tests" "${ROOT}/README.md" "${ROOT}/config.yaml"; then
  echo "Release audit failed: blocked terms found." >&2
  exit 1
fi

echo "Release audit passed: no blocked terms in source/docs/config."
