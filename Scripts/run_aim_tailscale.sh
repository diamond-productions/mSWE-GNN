#!/usr/bin/env bash
set -euo pipefail

REPO="${AIM_REPO:-results/aim}"
PORT="${AIM_PORT:-43800}"
HOST="${AIM_HOST:-}"

if [[ -z "$HOST" ]]; then
    if ! command -v tailscale >/dev/null 2>&1; then
        echo "tailscale command not found. Set AIM_HOST=<tailscale-ipv4> manually." >&2
        exit 1
    fi

    HOST="$(tailscale ip -4 | head -n 1)"
fi

if [[ -z "$HOST" ]]; then
    echo "Could not determine Tailscale IPv4. Set AIM_HOST=<tailscale-ipv4> manually." >&2
    exit 1
fi

echo "Starting Aim UI at http://${HOST}:${PORT}"
pixi run aim up --repo "$REPO" --host "$HOST" --port "$PORT"
