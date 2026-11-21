"""Find what was being said when audience attention was particularly high or low.

    python attention_insights/insights.py words.json attention.csv [--out-dir DIR]

Steps
  1. Group words into ~20-45 s chunks, breaking at sentence ends / pauses.
  2. Smooth + linearly detrend the attention signal, then give each chunk the mean
     attention over [start+lag, end+lag] and a z-score relative to the session.
  3. Pick high / low episodes (|z| above threshold, merged when adjacent, minimum length).
  4. Summarize with Claude (if credentials are available), else a keyword fallback.
  5. Write report.md + timeline.png (+ insights.json).
"""

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MODEL = "claude-opus-5-5"


@dataclass
class Chunk:
    start: float
    end: float
    text: str
    att: float = float("nan")
    z: float = 0.0


@dataclass
class Episode:
    kind: str  # "high" | "low"
    start: float
    end: float
    att: float
    z: float
    text: str
    label: str = ""
    why: str = ""
    keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 1. chunking
def make_chunks(words: list[dict], min_dur=12.0, max_dur=30.0, pause=0.6) -> list[Chunk]:
    chunks, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        dur = cur[-1]["end"] - cur[0]["start"]
        sentence_end = bool(re.search(r"[.!?]$", w["word"]))
        gap = words[i + 1]["start"] - w["end"] if i + 1 < len(words) else 99
        if (dur >= min_dur and (sentence_end or gap > pause)) or dur >= max_dur:
            chunks.append(Chunk(cur[0]["start"], cur[-1]["end"], " ".join(x["word"] for x in cur)))
            cur = []
    if cur:
        if chunks and cur[-1]["end"] - cur[0]["start"] < min_dur / 2:
            last = chunks[-1]  # tiny tail: append to previous chunk
            chunks[-1] = Chunk(last.start, cur[-1]["end"], last.text + " " + " ".join(x["word"] for x in cur))
        else:
            chunks.append(Chunk(cur[0]["start"], cur[-1]["end"], " ".join(x["word"] for x in cur)))
    return chunks


# ---------------------------------------------------------------- 2. alignment
def prepare_signal(t: np.ndarray, s: np.ndarray, smooth_s=4.0) -> np.ndarray:
    fs = 1 / np.median(np.diff(t))
    k = max(1, int(round(smooth_s * fs)))
    padded = np.pad(s, (k // 2, k - 1 - k // 2), mode="edge")  # no dips at the edges
    sm = np.convolve(padded, np.ones(k) / k, mode="valid")
    slope, icpt = np.polyfit(t, sm, 1)  # remove fatigue / drift
    return sm - (slope * t + icpt) + sm.mean()


def score_chunks(chunks: list[Chunk], t, s_detr, lag=3.0) -> None:
    for c in chunks:
        m = (t >= c.start + lag) & (t < c.end + lag)
        c.att = float(s_detr[m].mean()) if m.any() else float("nan")
    a = np.array([c.att for c in chunks])
    mu, sd = np.nanmean(a), np.nanstd(a) or 1.0
    for c in chunks:
        c.z = (c.att - mu) / sd


# ---------------------------------------------------------------- 3. selection
def select_episodes(chunks: list[Chunk], z_thr=0.6, k=3, min_dur=10.0) -> list[Episode]:
    eps: list[Episode] = []
    for kind, sign in (("high", 1), ("low", -1)):
        cur: list[Chunk] = []
        groups = []
        for c in chunks + [None]:
            if c is not None and sign * c.z >= z_thr:
                cur.append(c)
                continue
            if cur:
                groups.append(cur)
            cur = []
        for g in groups:
            if g[-1].end - g[0].start < min_dur:
                continue
            w = np.array([x.end - x.start for x in g])
            eps.append(Episode(kind, g[0].start, g[-1].end,
                               float(np.average([x.att for x in g], weights=w)),
                               float(np.average([x.z for x in g], weights=w)),
                               " ".join(x.text for x in g)))
    highs = sorted([e for e in eps if e.kind == "high"], key=lambda e: -e.z)[:k]
    lows = sorted([e for e in eps if e.kind == "low"], key=lambda e: e.z)[:k]
    return sorted(highs + lows, key=lambda e: e.start)


# ---------------------------------------------------------------- 4. summarize
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being
it its this that these those there here i you we they he she me my your our their them us
do does did have has had not no so than then too very can could will would should may might just
about into over after before while what which who whom when where why how all any some more most
other such only own same also like up out one two let now get got go going even still""".split())


def keywords(text: str, background: Counter, n=6) -> list[str]:
    toks = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2]
    c = Counter(toks)
    tot_bg = sum(background.values())
    score = {w: f * np.log(tot_bg / (background[w] + 1)) for w, f in c.items()}
    return [w for w, _ in sorted(score.items(), key=lambda x: -x[1])[:n]]


def summarize_with_claude(episodes: list[Episode], full_text: str) -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    blocks = []
    for i, e in enumerate(episodes):
        blocks.append(f'<episode id="{i}" kind="{e.kind}" time="{fmt(e.start)}-{fmt(e.end)}" '
                      f'attention="{e.att:.2f}" z="{e.z:+.2f}">\n{e.text}\n</episode>')
    prompt = f"""Below is a transcript of a talk, followed by passages where the audience's
measured attention (in [0,1], averaged over the audience, detrended for fatigue) was
notably HIGH or LOW relative to the rest of the session.

<transcript>
{full_text}
</transcript>

{chr(10).join(blocks)}

Reply with JSON only, no code fences, in this shape:
{{"episodes": [{{"id": 0, "label": "<3-6 word topic label>", "why": "<one sentence: what in the content/delivery plausibly explains the attention level>"}}],
 "insights": ["<3-5 short bullets contrasting what high-attention parts have in common vs. the low ones, with concrete advice for the speaker>"]}}

Describe correlations ("attention was lower during X"), not proven causes."""
    try:
        client = anthropic.Anthropic()
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        print("Claude: no valid credentials (set ANTHROPIC_API_KEY) - using keyword fallback.")
        return None
    except anthropic.APIConnectionError:
        print("Claude: network error - using keyword fallback.")
        return None
    except anthropic.APIStatusError as e:
        print(f"Claude: API error {e.status_code} - using keyword fallback.")
        return None
    except TypeError as e:  # e.g. credentials missing entirely
        print(f"Claude unavailable ({e}) - using keyword fallback.")
        return None
    if resp.stop_reason == "refusal":
        return None
    return "".join(b.text for b in resp.content if b.type == "text")


def apply_llm(episodes: list[Episode], raw: str) -> list[str]:
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0) if m else raw)
    for item in data.get("episodes", []):
        i = int(item["id"])
        if 0 <= i < len(episodes):
            episodes[i].label = item.get("label", "")
            episodes[i].why = item.get("why", "")
    return data.get("insights", [])


def fallback_insights(episodes: list[Episode]) -> list[str]:
    out = []
    for kind, word in (("high", "highest"), ("low", "lowest")):
        es = [e for e in episodes if e.kind == kind]
        if es:
            labs = "; ".join(f"{e.label} ({fmt(e.start)})" for e in es)
            out.append(f"Attention was {word} during: {labs}.")
    out.append("Set ANTHROPIC_API_KEY to get LLM-written topic labels and explanations.")
    return out


# ---------------------------------------------------------------- 5. output
def fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def plot(path: Path, t, s_raw, s_detr, chunks, episodes, lag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(t, s_raw, color="#9aa4b2", lw=0.8, label="attention (raw)")
    ax.plot(t, s_detr, color="#2563eb", lw=1.8, label="smoothed + detrended")
    for c in chunks:
        ax.hlines(c.att, c.start + lag, c.end + lag, color="#111827", lw=2.5, alpha=0.6)
    for e in episodes:
        col = "#16a34a" if e.kind == "high" else "#dc2626"
        ax.axvspan(e.start + lag, e.end + lag, color=col, alpha=0.12)
        y = 0.97 if e.kind == "high" else 0.05
        ax.text((e.start + e.end) / 2 + lag, y, e.label, ha="center", va="top" if y > 0.5 else "bottom",
                fontsize=9, color=col, wrap=True, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1])
    ticks = np.arange(0, t[-1] + 1, 30)
    ax.set_xticks(ticks, [fmt(x) for x in ticks])
    ax.set_xlabel("time (shaded spans shifted by reaction lag)")
    ax.set_ylabel("audience attention")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, bbox_to_anchor=(1, 0.88))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("words_json")
    ap.add_argument("attention_csv")
    ap.add_argument("--out-dir")
    ap.add_argument("--lag", type=float, default=5.0, help="audience reaction lag in seconds")
    ap.add_argument("--z", type=float, default=0.6, help="z-score threshold for high/low")
    ap.add_argument("--k", type=int, default=3, help="max episodes per kind")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir or Path(a.words_json).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(a.words_json).read_text())
    words = [w for s in data["segments"] for w in s["words"]]
    with open(a.attention_csv) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    s = np.array([float(r["attention"]) for r in rows])

    chunks = make_chunks(words)
    s_detr = prepare_signal(t, s)
    score_chunks(chunks, t, s_detr, a.lag)
    episodes = select_episodes(chunks, a.z, a.k)

    full_text = " ".join(w["word"] for w in words)
    bg = Counter(w for w in re.findall(r"[a-z]+", full_text.lower()) if w not in STOP)
    for e in episodes:
        e.keywords = keywords(e.text, bg)
        e.label = ", ".join(e.keywords[:3])

    raw = None if a.no_llm else summarize_with_claude(episodes, full_text)
    insights = apply_llm(episodes, raw) if raw else fallback_insights(episodes)
    source = f"Claude ({MODEL})" if raw else "keyword fallback"

    plot(out_dir / "timeline.png", t, s, s_detr, chunks, episodes, a.lag)

    lines = ["# Attention insights", "",
             f"Talk length {fmt(t[-1])} · mean attention {s.mean():.2f} · "
             f"{len(chunks)} chunks · lag {a.lag:.0f} s · summaries by {source}", "",
             "![timeline](timeline.png)", "", "## Key insights", ""]
    lines += [f"- {x}" for x in insights]
    for kind, title in (("high", "High-attention moments"), ("low", "Low-attention moments")):
        lines += ["", f"## {title}", ""]
        for e in [e for e in episodes if e.kind == kind]:
            lines += [f"### {fmt(e.start)}–{fmt(e.end)} · {e.label}",
                      f"attention {e.att:.2f} (z {e.z:+.2f}) · keywords: {', '.join(e.keywords)}", ""]
            if e.why:
                lines += [e.why, ""]
            lines += [f"> {e.text[:400]}{'…' if len(e.text) > 400 else ''}", ""]
    lines += ["## All chunks", "", "| time | attention | z | text |", "|---|---|---|---|"]
    lines += [f"| {fmt(c.start)}–{fmt(c.end)} | {c.att:.2f} | {c.z:+.2f} | {c.text[:80]}… |" for c in chunks]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    (out_dir / "insights.json").write_text(json.dumps(
        {"insights": insights, "episodes": [asdict(e) for e in episodes],
         "chunks": [asdict(c) for c in chunks]}, indent=1))

    print("\n".join(lines[:lines.index("## All chunks")]))
    print(f"\nWrote {out_dir/'report.md'}, {out_dir/'timeline.png'}, {out_dir/'insights.json'}")


if __name__ == "__main__":
    main()
