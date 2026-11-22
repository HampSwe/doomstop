#!/bin/bash
# The presenter side of the demo (DEMO.md) on this Mac alone, without the camera computer:
# call/score_sender.py --simulate sends a made-up attention score over UDP on localhost to the LED eye
# (attention/live_insights.py), which records this Mac's microphone. Close the eye (its × box, or Esc) to end
# the talk: the feedback card shows the attention graph with the topic at each moment, named by Ollama.
#
#   ./demo_local.sh             starts `ollama serve` if it isn't running, and stops it again at the end
#   ./demo_local.sh --no-llm    extra arguments go to live_insights.py
cd "$(dirname "$0")" || exit 1
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
if command -v ollama > /dev/null && ! curl -s -o /dev/null localhost:11434/api/version; then
    ollama serve > /dev/null 2>&1 &
    pids+=($!)
    for _ in $(seq 50); do curl -s -o /dev/null localhost:11434/api/version && break; sleep 0.2; done
fi
.venv/bin/python call/score_sender.py --to 127.0.0.1 --simulate > /dev/null &
pids+=($!)
.venv/bin/python attention/live_insights.py --udp 8767 --record --no-highlights "$@"
