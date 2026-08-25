#!/bin/bash
# Watchdog for the full MARL run.
#
# The first attempt at Task A died silently with its parent session and the
# loss was not noticed for eleven hours, because nothing was watching. This
# checks liveness every 60s and writes one line per check, so a death is
# visible in minutes. It also records per-cell episode progress, which catches
# the other failure mode a PID check misses: a process that is alive but stuck.
#
# Usage: scripts/watchdog_marl.sh <log-dir> <expected-count> [interval-seconds]
# Writes to <log-dir>/../watchdog.log
set -u
LOG_DIR="${1:-outputs/marl_full_run2/logs}"
EXPECTED="${2:-6}"
INTERVAL="${3:-60}"
OUT="$(dirname "$LOG_DIR")/watchdog.log"

echo "[$(date '+%F %T')] watchdog started: expecting $EXPECTED cells, every ${INTERVAL}s" >> "$OUT"
prev_total=-1
while true; do
    alive=$(pgrep -f "run_marl_training" 2>/dev/null | wc -l | tr -d " ")
    total=0
    for f in "$LOG_DIR"/*.log; do
        [ -e "$f" ] || continue
        n=$(grep -c "^episode" "$f" 2>/dev/null || echo 0)
        total=$((total + n))
    done

    if [ "$alive" -eq 0 ]; then
        echo "[$(date '+%F %T')] ALERT: no training processes alive (expected $EXPECTED); episodes=$total" >> "$OUT"
        echo "[$(date '+%F %T')] watchdog exiting: run is over or dead" >> "$OUT"
        exit 1
    elif [ "$alive" -lt "$EXPECTED" ]; then
        echo "[$(date '+%F %T')] ALERT: only $alive/$EXPECTED cells alive; episodes=$total" >> "$OUT"
    elif [ "$total" -eq "$prev_total" ]; then
        echo "[$(date '+%F %T')] ALERT: $alive alive but no episode progress since last check (episodes=$total)" >> "$OUT"
    else
        echo "[$(date '+%F %T')] ok: $alive/$EXPECTED alive, episodes=$total" >> "$OUT"
    fi
    prev_total=$total
    sleep "$INTERVAL"
done
