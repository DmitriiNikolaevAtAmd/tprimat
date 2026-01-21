#!/bin/bash
# Update Primus to latest version (run inside Docker container)

set -e

PRIMUS_PATH="${PRIMUS_PATH:-/workspace/Primus}"

echo "[*] Updating Primus at $PRIMUS_PATH"

if [ ! -d "$PRIMUS_PATH" ]; then
    echo "[x] Primus not found at $PRIMUS_PATH"
    exit 1
fi

cd "$PRIMUS_PATH"

if [ ! -d ".git" ]; then
    echo "[x] Not a git repository: $PRIMUS_PATH"
    exit 1
fi

echo "[*] Current branch: $(git branch --show-current)"
echo "[*] Pulling latest changes..."

git pull

echo "[+] Primus updated successfully"
echo "[*] Latest commit: $(git log -1 --oneline)"
