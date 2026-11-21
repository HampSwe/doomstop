# attention

`attention_profile.py` takes an attention score over time and finds the most interesting parts of a
presentation:

- **Most interesting part**: the 2-minute window with the highest average attention (and the least
  interesting one, for contrast).
- **Highest-attention events**: the moments that stand out most, with the time of the peak and how long it
  lasted.
- **Lowest-attention events**: the same for the dips, i.e. when attention dropped, how low and for how
  long. These are the moments to cut or rework.
- **Topics ranked**: every topic, slide or section ranked by its average attention, with its own peak time
  and trend (rising, falling or flat).
- **Profile**: the smoothed attention curve relative to the talk's own typical level, in the JSON output, so
  talks with different score scales can be compared.

Any score scale works (0-100, focus levels 1-4, 0-1). "+1.0 spread" means one typical fluctuation above
the talk's usual (median) level.

## Run

```bash
python attention/attention_profile.py --demo --report demo.png           # synthetic talk, to see the output
python attention/attention_profile.py scores.csv --topics topics.csv --report talk.png --json talk.json
my_detector | python attention/attention_profile.py -                    # live: "time,score" lines on stdin
```

**Live window:** `live_view.py` shows the graph while the scores come in, and redraws it twice a second
with peaks, dips and the most interesting part so far. `simulate.py` generates test data (a synthetic
25-minute talk, 10x speed by default):

```bash
python attention/simulate.py | python attention/live_view.py             # test: watch the generated talk
my_detector | python attention/live_view.py                              # real data: "time,score[,topic]" lines
python attention/live_view.py --udp 8767                                 # or receive UDP datagrams instead
python attention/simulate.py --udp 8767                                  #   ... e.g. from the simulator
python attention/simulate.py --speed 0 > talk.csv                        # test data as a file
```

**Input CSV:** a time column (seconds, `mm:ss`, `h:mm:ss` or ISO timestamps) and a score column. The names
are detected automatically (`time`/`t`/`t_s`/`timestamp` and `attention`/`score`/`engagement`/`level`/`focus`)
or set with `--time-col` and `--score-col`. Rows with the same time, e.g. one per face cam, are averaged. A
`topic`/`marker`/`slide` column starts a new topic wherever it has a value.

**Topics file (optional):** one `start,name` per line. Each topic lasts until the next one starts:

```
0:00,Intro
3:20,Pricing
7:45,Live demo
```

Without topics, the talk is cut into equal parts (`--segment` to choose the length).

**Useful options:** `--smooth 30` (seconds of smoothing), `--window 120` (length of the best/worst part),
`--peaks 5`, `--dips 5`, `--min-gap 60` (seconds between listed events), `--min-level 0.5` / `--min-prominence 0.3`
(how much an event must stand out).

## From Python

```python
from attention_profile import AttentionProfile, print_report

p = AttentionProfile()
p.add(t_seconds, score)            # for every incoming sample; marker="slide 3" starts a topic
result = p.analyse()               # or analyse(topics=[(0, "Intro"), (200, "Demo")])
print_report(result)
result["best_part"], result["peaks"], result["dips"], result["topics"]
```
