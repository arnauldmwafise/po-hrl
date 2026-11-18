#!/bin/bash
# Runs job2.py; if no training step finishes for 25 minutes, kills it and restarts (resumes from last step).
cd /workspace/pohrl
OUT=/workspace/pohrl/m3_Qwen3-1.7B
for attempt in $(seq 1 12); do
  [ -f DONE2 ] && { pkill -9 -f EngineCore; echo "$(date +%T) finished"; exit 0; }
  echo "$(date +%T) start attempt $attempt"
  setsid python3 job2.py >> job2.log 2>&1 &
  PID=$!; START=$(date +%s)
  while kill -0 $PID 2>/dev/null; do
    sleep 60
    LAST=$(stat -c %Y $OUT/*_s0/state.json $OUT/reeval.jsonl 2>/dev/null | sort -n | tail -1)
    [ -z "$LAST" ] || [ "$LAST" -lt "$START" ] && LAST=$START
    if [ $(( $(date +%s) - LAST )) -gt 1500 ]; then
      echo "$(date +%T) no progress for 25 min: restarting"
      kill -9 -- -$PID 2>/dev/null; pkill -9 -f EngineCore; sleep 30; break
    fi
  done
done
echo "$(date +%T) gave up after 12 attempts"
