#!/usr/bin/env bash
# Run NeuroFlow TelegramIngestor as a background daemon
# Usage: bash scripts/run-ingestor.sh
set -euo pipefail

cd "$(dirname "$0")/.."
LOG="/opt/data/neuroflow-ingestor.log"
PIDFILE="/opt/data/neuroflow-ingestor.pid"

export NEUROFLOW_DB_PATH="${NEUROFLOW_DB_PATH:-/opt/data/neuroflow.db}"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Ingestor already running (PID $(cat "$PIDFILE"))"
    exit 0
fi

nohup python -m neuroflow_core > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "Ingestor started (PID $!)"
