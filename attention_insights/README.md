# attention_insights

Find which topics of a talk got particularly high or low audience attention.

```
audio ──transcribe.py──▶ words.json ─┐
                                     ├─insights.py─▶ report.md, timeline.png, insights.json
attention.csv (t, attention∈[0,1]) ──┘
```

1. **transcribe.py**: faster-whisper with word-level timestamps.
2. **insights.py**:
   - splits the words into chunks at sentence ends or pauses, sized to the talk (about 4 s for a 30 s talk, up to 12–30 s for long ones);
   - smooths and linearly detrends the attention signal to remove fatigue drift;
   - averages attention per chunk, shifted by the reaction `--lag`, and z-scores it;
   - picks and merges the high and low episodes, dropping anything under 10 s;
   - asks a local open-source LLM (Ollama, default `qwen2.5:3b`) for topic labels and insights, or falls back to keywords;
   - plots the timeline.

For a demo without real attention data:

- **manuscript.md**: a ~30-second script whose sections are engaging or dull on purpose.
- **record.py**: records the microphone to a WAV file. It stops on Ctrl+C or after 4 s of silence.
- **simulate_attention.py**: builds a plausible attention curve for the recording. It knows how engaging each manuscript section is, and adds reaction delay, sluggish response, fatigue, noise and one spurious spike.

```bash
pip install faster-whisper soundfile sounddevice numpy matplotlib
brew install ollama && ollama serve &   # local LLM for the labels
ollama pull qwen2.5:3b
python attention_insights/record.py                       # -> data/talk.wav
python attention_insights/transcribe.py attention_insights/data/talk.wav attention_insights/data/words.json
python attention_insights/simulate_attention.py attention_insights/data/words.json
python attention_insights/insights.py attention_insights/data/words.json attention_insights/data/attention.csv
```

Without a running Ollama you get keyword labels (or pass `--no-llm`).

## Live

`attention/live_insights.py` runs this analysis while the talk is going. It reads the attention stream and
the transcript words as they arrive, shows the graph with what was said at each high and low moment, and
writes this folder's report when the stream ends (into `data/live/`):

```bash
python attention/simulate.py --demo-talk | python attention/live_insights.py   # replays data/ live
```

`insights.py` can also be used from code: `analyse(words, t, s)` (works on a talk that is still going) and
`write_report(out_dir, words, t, s)`.
