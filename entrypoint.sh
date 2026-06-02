#!/bin/sh
# entrypoint.sh — graceful shutdown via SIGTERM → SIGINT chain
set -e

# Trap SIGTERM (docker stop) and forward as SIGINT to the app
trap 'kill -INT "$PID" 2>/dev/null; wait "$PID"' TERM

exec "$@" &
PID=$!
wait "$PID"
