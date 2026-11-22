#!/usr/bin/env python3
"""
attention_profile.py - find the most interesting parts of a presentation from an attention score over time.

Input: attention samples (time, score) from a CSV file, from stdin while the talk is running, or added one by
one from Python. Any scale works (0-100, 1-4 focus levels, 0-1 probabilities ...): everything is judged
relative to the presentation's own typical level, so talks with different scales can be compared.

Output
  profile      the attention curve on an even time grid, smoothed, and as "relative attention": how far above
               or below the talk's typical level (median) it is, in units of its typical spread (robust std)
  best part    the --window (default 2 min) with the highest average attention, and the lowest for contrast
  peaks        the highest-attention events: peak time, height and how long the peak lasted
  dips         the lowest-attention events, the same way: when attention dropped, how low, for how long
  topics       every topic (slide / section) ranked by average relative attention, with its peak and trend.
               Topics come from --topics (start times), from a marker/topic column in the data, or else
               the talk is cut into equal parts (--segment)

Usage
  python attention/attention_profile.py scores.csv
  python attention/attention_profile.py scores.csv --topics topics.csv --report talk.png --json talk.json
  my_detector | python attention/attention_profile.py -           # live "time,score" lines; Ctrl+C = analyse
  python attention/attention_profile.py --demo --report demo.png   # synthetic talk, to see what you get

From Python:
  from attention_profile import AttentionProfile
  p = AttentionProfile()
  p.add(t_seconds, score)                     # call for every incoming sample (marker="slide 3" optional)
  result = p.analyse(topics=[(0, "Intro"), (200, "Demo")])
  print_report(result)

Input CSV: a time column (seconds, mm:ss, h:mm:ss or ISO timestamps) and a score column. Names are
auto-detected (time / t / t_s / timestamp and attention / score / engagement / level / focus) or set with
--time-col / --score-col. Rows that share a time (e.g. one per face cam) are averaged; rows without a
number (e.g. "calibrating") are skipped. A header-less file is read as time,score[,topic].

Topics file: one "start,name" per line (start in seconds, mm:ss or h:mm:ss); each topic lasts until the
next one starts. Example:
  0:00,Intro
  3:20,Pricing
  7:45,Live demo
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime

import numpy as np

TIME_COLS = ["t_s", "time_s", "t", "time", "seconds", "timestamp", "elapsed"]
SCORE_COLS = ["attention", "attention_score", "score", "engagement", "level", "focus", "value"]
TOPIC_COLS = ["topic", "marker", "slide", "section"]

# Light-surface chart tokens (same reference palette as the rest of the project)
SURFACE, TEXT, TEXT2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, RED = "#2a78d6", "#e34948"


# ------------------------------------------------------------------------------------------------ helpers
def hms(t):
    t = int(round(t))
    h, r = divmod(abs(t), 3600)
    m, s = divmod(r, 60)
    return ("-" if t < 0 else "") + (f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}")


def parse_time(s):
    """'95' / '95.5' -> seconds, '1:35' / '0:01:35' -> seconds, ISO timestamp -> datetime, else None."""
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if ":" in s and "-" not in s and "T" not in s:
        try:
            parts = [float(p) for p in s.split(":")]
            return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def parse_float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pick(names, wanted, given=None):
    """Column name to use: the given one (case-insensitive), else the first known name present."""
    low = {n.lower().strip(): n for n in names}
    if given:
        if given.lower() not in low:
            raise SystemExit(f"column '{given}' not found; columns are: {', '.join(names)}")
        return low[given.lower()]
    return next((low[w] for w in wanted if w in low), None)


def load_topics(path):
    topics = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "," not in line:
                continue
            start, name = (p.strip() for p in line.split(",", 1))
            t = parse_time(start)
            if isinstance(t, float):
                topics.append((t, name.strip('"') or hms(t)))
    if not topics:
        raise SystemExit(f"no 'start,name' lines found in {path}")
    return sorted(topics)


def nice_segment(duration, minimum=0.0):
    """Round segment length that cuts the talk into at most ~12 parts, and is at least `minimum` s long."""
    for s in (30, 60, 120, 180, 300, 600, 900, 1800):
        if duration / s <= 12 and s >= minimum:
            return float(s)
    return 3600.0


# ------------------------------------------------------------------------------------------------ analysis
def gaussian_smooth(y, step, seconds):
    """Centred Gaussian smoothing over ~`seconds` (no lag), ignoring gaps (NaN stays NaN)."""
    sigma = seconds / 4.0 / step  # in samples; +-2 sigma ~ the window
    half = min(int(math.ceil(3 * sigma)), (len(y) - 1) // 2)
    if seconds <= 0 or half < 1:
        return y.copy()
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    ok = ~np.isnan(y)
    num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    out = num / np.where(den > 1e-9, den, np.nan)
    out[~ok] = np.nan
    return out


def window_extreme(grid, y, w, highest=True):
    """Start index and mean of the w-sample window with the highest (or lowest) mean; needs 70% coverage."""
    w = max(1, min(w, len(y)))
    ok = ~np.isnan(y)
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, y, 0.0))])
    n = np.concatenate([[0], np.cumsum(ok)])
    sums, counts = c[w:] - c[:-w], n[w:] - n[:-w]
    means = np.where(counts >= 0.7 * w, sums / np.maximum(counts, 1), np.nan)
    if np.all(np.isnan(means)):
        return None, None
    i = int(np.nanargmax(means) if highest else np.nanargmin(means))
    return i, float(means[i])


def find_peaks(r, min_prominence, wlen):
    """Local maxima of r with their prominence (height above the higher of the two surrounding valleys,
    looking at most `wlen` samples to each side, so an event is measured against its own surroundings and
    not against a dull stretch far away) and the half-prominence extent [lo, hi] (sample indices)."""
    n, out = len(r), []
    for i in range(n):
        if (i > 0 and r[i - 1] > r[i]) or (i < n - 1 and r[i + 1] >= r[i]):
            continue
        bases = []
        for direction in (-1, 1):
            j, low = i + direction, r[i]
            while 0 <= j < n and abs(j - i) <= wlen and r[j] <= r[i]:
                low = min(low, r[j])
                j += direction
            bases.append(low)
        prom = r[i] - max(bases)
        if prom < min_prominence:
            continue
        level, lo, hi = r[i] - prom / 2, i, i
        while lo > 0 and r[lo - 1] > level:
            lo -= 1
        while hi < n - 1 and r[hi + 1] > level:
            hi += 1
        out.append((i, prom, lo, hi))
    return out


class AttentionProfile:
    """Collects (time, score) samples as they come in; analyse() turns them into the profile + findings."""

    def __init__(self):
        self.t, self.s, self.markers = [], [], []
        self.clock_start = None  # set when the input used real timestamps

    def add(self, t, score, marker=None):
        self.t.append(float(t))
        self.s.append(float(score))
        if marker:
            self.markers.append((float(t), str(marker)))

    def __len__(self):
        return len(self.t)

    def analyse(self, topics=None, smooth=30.0, window=120.0, step=None, segment=None, peaks=5, dips=5,
                min_gap=60.0, min_prominence=0.3, min_level=0.5):
        if len(self.t) < 5:
            raise ValueError("need at least 5 samples")
        t, s = np.asarray(self.t), np.asarray(self.s)
        ut, inv = np.unique(t, return_inverse=True)  # same time (several faces) -> average
        us = np.bincount(inv, weights=s) / np.bincount(inv)
        if step is None:
            step = max(float(np.median(np.diff(ut))), 0.5) if len(ut) > 1 else 1.0
        grid = np.arange(ut[0], ut[-1] + step / 2, step)
        y = np.interp(grid, ut, us)
        idx = np.clip(np.searchsorted(ut, grid, side="right"), 1, len(ut) - 1)  # blank long gaps without data
        y[(ut[idx] - ut[idx - 1] > max(3 * step, 20.0)) & (grid - ut[idx - 1] > 1e-9)] = np.nan
        sm = gaussian_smooth(y, step, smooth)

        typical = float(np.nanmedian(y))
        # robust std (MAD), but never below half the plain std: a signal that mostly sits on one exact value
        # (e.g. focus level 3) would otherwise get a near-zero spread and absurd relative values
        spread = max(1.4826 * float(np.nanmedian(np.abs(y - typical))), 0.5 * float(np.nanstd(y)))
        if spread < 1e-9:
            spread = 1.0
        rel = (sm - typical) / spread
        duration = float(grid[-1] - grid[0])

        def part(i, mean, w):
            if i is None:
                return None
            a, b = float(grid[i]), float(grid[min(i + w - 1, len(grid) - 1)] + step)
            return {"start": a, "end": min(b, float(grid[-1])), "mean": mean, "relative": (mean - typical) / spread}

        w = max(1, int(round(window / step)))
        best = part(*window_extreme(grid, y, w, True), w)
        worst = part(*window_extreme(grid, y, w, False), w)

        r_fill = np.interp(grid, grid[~np.isnan(rel)], rel[~np.isnan(rel)])  # gaps bridged for the event search

        def events(sign, n):
            """sign +1: highest-attention peaks; sign -1: lowest-attention dips (peaks of the flipped curve)."""
            r, found = sign * r_fill, []
            if n <= 0:
                return []
            for i, prom, lo, hi in sorted(find_peaks(r, min_prominence, w), key=lambda p: -r[p[0]]):
                if (np.isnan(rel[i]) or sign * rel[i] < min_level
                        or any(abs(grid[i] - grid[j]) < min_gap for j, *_ in found)):
                    continue
                found.append((i, prom, lo, hi))
                if len(found) == n:
                    break
            return [{"time": float(grid[i]), "score": float(sm[i]), "relative": float(rel[i]),
                     "prominence": float(prom), "start": float(grid[lo]), "end": float(grid[hi])}
                    for i, prom, lo, hi in found]

        peak_list, dip_list = events(+1, peaks), events(-1, dips)

        source = "given"
        if not topics and self.markers:
            topics, source = sorted(self.markers), "markers"
            if topics[0][0] > grid[0] + step:
                topics.insert(0, (float(grid[0]), "Start"))
        if not topics:
            seg = segment or nice_segment(duration, 2 * smooth)
            topics = [(float(a), f"{hms(a)}-{hms(min(a + seg, grid[-1]))}") for a in np.arange(grid[0], grid[-1], seg)]
            source = f"equal parts of {hms(seg)}"
        topic_list = []
        for k, (start, name) in enumerate(topics):
            end = topics[k + 1][0] if k + 1 < len(topics) else float(grid[-1]) + step
            sel = (grid >= start) & (grid < end) & ~np.isnan(y)
            if sel.sum() < 2:
                continue
            vals, ts = y[sel], grid[sel]
            mean = float(vals.mean())
            slope = float(np.polyfit(ts, vals, 1)[0]) if np.ptp(ts) > 0 else 0.0
            change = slope * float(np.ptp(ts)) / spread  # start -> end of topic, in spreads
            j = int(np.nanargmax(np.where(sel, sm, np.nan)))
            topic_list.append({"topic": name, "start": float(start), "end": float(min(end, grid[-1])),
                               "mean": mean, "relative": (mean - typical) / spread, "peak_time": float(grid[j]),
                               "above_typical": float(np.mean(vals > typical)), "samples": int(sel.sum()),
                               "trend": "rising" if change >= 0.5 else "falling" if change <= -0.5 else "flat",
                               "short": bool(sel.sum() < 5 or end - start < smooth)})
        ranked = sorted(topic_list, key=lambda d: -d["relative"])
        for rank, d in enumerate(ranked, 1):
            d["rank"] = rank

        def topic_at(x):
            hit = [d["topic"] for d in topic_list if d["start"] <= x < d["end"] + step]
            return hit[-1] if hit else None

        for item in [best, worst] + peak_list + dip_list:
            if item:
                item["topic"] = topic_at(item.get("time", (item["start"] + item["end"]) / 2))
        return {
            "summary": {"start": float(grid[0]), "duration_s": duration, "samples": len(self.t), "step_s": step,
                        "typical": typical, "spread": spread, "smooth_s": smooth, "window_s": window,
                        "topics_from": source, "clock_start": self.clock_start},
            "best_part": best, "worst_part": worst, "peaks": peak_list, "dips": dip_list, "topics": ranked,
            "profile": {"t": grid.tolist(), "score": y.tolist(), "smoothed": sm.tolist(), "relative": rel.tolist()},
        }


# ------------------------------------------------------------------------------------------------ input
def read_samples(f, prof, time_col=None, score_col=None, topic_col=None):
    """Feed CSV rows from an open file (or stdin, line by line as they arrive) into prof."""
    reader = csv.reader(f)
    first = next(reader, None)
    if first is None:
        return
    numeric = len(first) >= 2 and parse_time(first[0]) is not None and parse_float(first[1]) is not None
    if numeric and not (time_col or score_col):
        ti, si, mi = 0, 1, 2 if len(first) > 2 else None
        rows = [first]
    else:
        tc, sc = pick(first, TIME_COLS, time_col), pick(first, SCORE_COLS, score_col)
        if tc is None or sc is None:
            raise SystemExit(f"could not find the time / score columns in {first}; use --time-col / --score-col")
        mc = pick(first, TOPIC_COLS, topic_col)
        ti, si = first.index(tc), first.index(sc)
        mi = first.index(mc) if mc else None
        print(f"reading time from '{tc}', attention from '{sc}'" + (f", topics from '{mc}'" if mc else ""),
              file=sys.stderr)
        rows = []
    t0 = None
    for row in _chain(rows, reader):
        if len(row) <= max(ti, si):
            continue
        t, v = parse_time(row[ti]), parse_float(row[si])
        if t is None or v is None:
            continue
        if isinstance(t, datetime):
            t0 = t0 or t
            prof.clock_start = prof.clock_start or t0.isoformat(timespec="seconds")
            t = (t - t0).total_seconds()
        prof.add(t, v, row[mi].strip() if mi is not None and mi < len(row) and row[mi].strip() else None)


def _chain(first_rows, rest):
    yield from first_rows
    yield from rest


DEMO_TOPICS = [(0, "Intro"), (180, "The problem"), (420, "Live demo"), (720, "Pricing"), (960, "Roadmap"),
               (1200, "Q&A")]


def demo_stream(seed=7):
    """A synthetic 25-minute talk sampled every 5 s on a 1-4 scale, with a known best topic, peaks and a dip.
    Yields (t, score, topic) - topic is the topic's name on its first sample, else None."""
    rng = np.random.default_rng(seed)
    level = {"Intro": 3.0, "The problem": 2.9, "Live demo": 3.5, "Pricing": 2.6, "Roadmap": 2.8, "Q&A": 3.1}
    # (time, height, width s): joke, demo wow, technical problem (a dip), audience question
    events = [(310, 0.9, 25), (575, 1.0, 40), (1080, -0.9, 25), (1290, 0.8, 30)]
    current = None
    for t in np.arange(0, 1500, 5.0):
        name = [n for s, n in DEMO_TOPICS if s <= t][-1]
        base = level[name] - (0.5 * (t - 720) / 240 if name == "Pricing" else 0)  # pricing loses the room
        bump = sum(h * math.exp(-0.5 * ((t - c) / wd) ** 2) for c, h, wd in events)
        yield float(t), float(np.clip(base + bump + rng.normal(0, 0.25), 1, 4)), (name if name != current else None)
        current = name


def demo_samples(prof, seed=7):
    for t, score, _ in demo_stream(seed):
        prof.add(t, score)
    return list(DEMO_TOPICS)


# ------------------------------------------------------------------------------------------------ output
def print_report(res):
    s = res["summary"]

    def tag(item):
        return f'  - "{item["topic"]}"' if item.get("topic") else ""

    def rel(x):
        return f"{x:+.1f} spread"

    print(f"\n=== Attention profile ({hms(s['duration_s'])}, {s['samples']} samples"
          + (f", started {s['clock_start']}" if s["clock_start"] else "") + ") ===")
    print(f"Typical level {s['typical']:.3g} (median), typical spread {s['spread']:.3g}; "
          "relative attention below = how many spreads above / below typical")
    b, w = res["best_part"], res["worst_part"]
    if b:
        print(f"Most interesting part : {hms(b['start'])}-{hms(b['end'])}  avg {b['mean']:.3g}  "
              f"({rel(b['relative'])}){tag(b)}")
    if w:
        print(f"Least interesting part: {hms(w['start'])}-{hms(w['end'])}  avg {w['mean']:.3g}  "
              f"({rel(w['relative'])}){tag(w)}")
    for title, key, at in (("Highest-attention events", "peaks", "peak at"),
                           ("Lowest-attention events", "dips", "low at")):
        print(f"\n{title}")
        if not res[key]:
            print("  none stand out (try a lower --min-level / --min-prominence, or less --smooth)")
            continue
        print(f"  {'#':<3}{at:<10}{'level':<8}{'vs typical':<13}{'lasted':<16}topic")
        for k, p in enumerate(res[key], 1):
            print(f"  {k:<3}{hms(p['time']):<10}{p['score']:<8.3g}{rel(p['relative']):<13}"
                  f"{hms(p['start']) + '-' + hms(p['end']):<16}{p.get('topic') or ''}")
    print(f"\nTopics ranked by attention (topics from {s['topics_from']})")
    width = max([len(d["topic"]) for d in res["topics"]] + [5]) + 2
    print(f"  {'#':<3}{'topic':<{width}}{'time':<14}{'avg':<7}{'vs typical':<13}{'peak at':<9}trend")
    for d in res["topics"]:
        note = "  (short - rough)" if d["short"] else ""
        print(f"  {d['rank']:<3}{d['topic']:<{width}}{hms(d['start']) + '-' + hms(d['end']):<14}{d['mean']:<7.3g}"
              f"{rel(d['relative']):<13}{hms(d['peak_time']):<9}{d['trend']}{note}")


def to_json(res):
    def clean(x):
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    return json.dumps(clean(res), indent=1)


def chart_style():
    import matplotlib
    matplotlib.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "xtick.color": MUTED,
                                "ytick.color": MUTED, "axes.labelcolor": TEXT2, "text.color": TEXT})


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def draw_timeline(ax1, res, title="Attention over the presentation", min_span=0.0):
    """Attention over time on ax1: raw samples, smoothed line, typical level, topic boundaries, the most
    interesting part and the ▲ peak / ▼ dip events. Used by the PNG report and by live_view.py.
    min_span: show at least this many seconds on the x axis (live view: room to grow)."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    s, prof, topics = res["summary"], res["profile"], res["topics"]
    t = np.array(prof["t"])  # the data's own clock, same times as the printed report
    y, sm = np.array(prof["score"], float), np.array(prof["smoothed"], float)
    style_axis(ax1)
    ax1.grid(axis="y", color=GRID, linewidth=0.6)
    lo, hi = np.nanmin(y), np.nanmax(y)
    pad = 0.08 * (hi - lo or 1)
    b = res["best_part"]
    if b:  # band + label in its own row under the topic names, so it never meets a peak label
        ax1.axvspan(b["start"], b["end"], color=BLUE, alpha=0.10, linewidth=0)
        ax1.text((b["start"] + b["end"]) / 2, lo - 2.5 * pad, "most interesting", ha="center", va="center",
                 fontsize=8, color=TEXT2)
    for d in sorted(topics, key=lambda d: d["start"]):  # topic boundaries + names along the bottom
        ax1.axvline(d["start"], color=BASE, linewidth=0.7)
        ax1.text(max(d["start"], t[0]), lo - 0.6 * pad, f" {d['topic']}", fontsize=7.5, color=MUTED, va="top",
                 clip_on=True)
    ax1.axhline(s["typical"], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax1.text(t[-1], s["typical"], " typical", fontsize=7.5, color=MUTED, va="center")
    ax1.plot(t, y, color=BASE, linewidth=0.8)
    ax1.plot(t, sm, color=BLUE, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    for key, color, sign, mark in (("peaks", BLUE, 1, "▲"), ("dips", RED, -1, "▼")):
        for k, p in enumerate(res[key], 1):  # peaks labelled above the point, dips below
            x = p["time"]
            ax1.plot(x, p["score"], "o", ms=7, color=color, mec=SURFACE, mew=1.5, zorder=3)
            ax1.annotate(f"{mark}{k} {hms(x)}", (x, p["score"]), xytext=(0, 9 * sign), textcoords="offset points",
                         ha="center", va="bottom" if sign > 0 else "top", fontsize=8, color=TEXT2)
    ax1.set_ylim(lo - 3.2 * pad, hi + 2.0 * pad)
    x1 = max(t[-1], t[0] + min_span)
    ax1.set_xlim(t[0], x1)
    ax1.set_ylabel("attention")
    tick = next((v for v in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600) if (x1 - t[0]) / v <= 10), 7200)
    ax1.xaxis.set_major_locator(MultipleLocator(tick))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: hms(max(v, 0))))
    ax1.set_title(title, loc="left", fontsize=12, color=TEXT, weight="bold", pad=10)


def save_report(res, path):
    """Top: attention over time (smoothed line over the raw samples) with topics, best part and peaks.
    Bottom: topics ranked by average attention relative to typical (diverging bars)."""
    try:
        from matplotlib.figure import Figure  # no pyplot: drawn off-screen, never touches open windows
    except ImportError:
        print("matplotlib not installed - skipping the PNG report", file=sys.stderr)
        return False
    topics = res["topics"]
    chart_style()
    n_top = len(topics)
    fig = Figure(figsize=(9.5, 4.4 + 0.3 * n_top), facecolor=SURFACE)
    ax1, ax2 = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3.2, 0.9 + 0.3 * n_top], "hspace": 0.45})
    draw_timeline(ax1, res)
    style_axis(ax2)
    ranked = topics[::-1]  # best at the top
    vals = [d["relative"] for d in ranked]
    ax2.barh(range(n_top), vals, height=0.62, color=[BLUE if v >= 0 else RED for v in vals],
             edgecolor=SURFACE, linewidth=1)
    ax2.axvline(0, color=MUTED, linewidth=0.8)
    span = max(max(abs(v) for v in vals), 0.5)
    for k, v in enumerate(vals):
        ax2.text(v + (0.03 if v >= 0 else -0.03) * span, k, f"{v:+.1f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=8, color=TEXT2)
    ax2.set_yticks(range(n_top), [f"{d['rank']}. {d['topic']}" for d in ranked], color=TEXT2)
    ax2.set_xlim(-span * 1.25, span * 1.25)
    ax2.grid(axis="x", color=GRID, linewidth=0.6)
    ax2.set_xlabel("average attention vs typical (in typical spreads)")
    ax2.set_title("Topics ranked", loc="left", fontsize=10, color=TEXT, weight="bold")
    fig.subplots_adjust(left=0.2, right=0.93, top=0.93, bottom=0.08)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    return True


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="CSV with time + attention columns, or - for stdin (live)")
    p.add_argument("--demo", action="store_true", help="analyse a synthetic 25-minute talk instead")
    p.add_argument("--topics", help="file with 'start,name' lines (see above)")
    p.add_argument("--time-col", help="name of the time column (default: auto-detect)")
    p.add_argument("--score-col", help="name of the attention column (default: auto-detect)")
    p.add_argument("--topic-col", help="name of a column whose non-empty values start a new topic")
    p.add_argument("--smooth", type=float, default=30, help="smoothing window in seconds (default 30)")
    p.add_argument("--window", type=float, default=120, help="length of the best / worst part in s (default 120)")
    p.add_argument("--segment", type=float, help="without topics: cut the talk into parts this long (s)")
    p.add_argument("--peaks", type=int, default=5, help="how many highest-attention events to list (default 5)")
    p.add_argument("--dips", type=int, default=5, help="how many lowest-attention events to list (default 5)")
    p.add_argument("--min-gap", type=float, default=60, help="minimum seconds between two listed events")
    p.add_argument("--min-prominence", type=float, default=0.3,
                   help="how much an event must stand out from its surroundings, in typical spreads")
    p.add_argument("--min-level", type=float, default=0.5,
                   help="how far above the typical level an event must reach, in typical spreads")
    p.add_argument("--step", type=float, help="resampling step in s (default: median sample interval)")
    p.add_argument("--report", help="save a PNG chart, e.g. talk.png")
    p.add_argument("--json", help="save the full profile + findings as JSON, e.g. talk.json")
    a = p.parse_args(argv)

    prof, topics = AttentionProfile(), load_topics(a.topics) if a.topics else None
    if a.demo:
        topics = topics or demo_samples(prof)
    elif a.input == "-":
        print("reading 'time,score' lines from stdin - Ctrl+C or end of input to analyse", file=sys.stderr)
        try:
            read_samples(sys.stdin, prof, a.time_col, a.score_col, a.topic_col)
        except KeyboardInterrupt:
            pass
    elif a.input:
        with open(a.input, newline="") as f:
            read_samples(f, prof, a.time_col, a.score_col, a.topic_col)
    else:
        p.error("give an input CSV, - for stdin, or --demo")
    if len(prof) < 5:
        raise SystemExit(f"only {len(prof)} usable samples - need at least 5")
    res = prof.analyse(topics=topics, smooth=a.smooth, window=a.window, step=a.step, segment=a.segment,
                       peaks=a.peaks, dips=a.dips, min_gap=a.min_gap, min_prominence=a.min_prominence, min_level=a.min_level)
    print_report(res)
    saved = []
    if a.report and save_report(res, a.report):
        saved.append(a.report)
    if a.json:
        with open(a.json, "w") as f:
            f.write(to_json(res))
        saved.append(a.json)
    if saved:
        print("\nSaved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
