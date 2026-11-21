"""Simulate a plausible audience attention score for a recorded manuscript reading.

    python attention_insights/simulate_attention.py words.json [attention.csv]

The simulator knows how engaging each manuscript section is meant to be (stories,
questions and surprises engage; definitions, methodology and admin don't). It finds
where each section starts in the transcript, then builds a 2 Hz signal with:
  - a sluggish response (first-order lag, tau ~6 s) plus a reaction delay of ~3 s,
  - slow decay of attention inside long sections,
  - fatigue drift over the whole talk,
  - AR(1) noise and one short spurious spike (to test the minimum-duration filter).

The insights pipeline never sees these section levels; it only gets the CSV.
"""

import argparse
import csv
import difflib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# Target engagement per manuscript section (order matters).
LEVELS = {
    "Opening story": 0.80,
    "Definitions and numbers": 0.35,
    "Audience question": 0.85,
    "Study methodology": 0.28,
    "Surprising result": 0.88,
    "Logistics and announcements": 0.30,
    "Closing": 0.62,
}

FS = 2.0          # samples per second
DELAY = 3.0       # audience reaction delay (s)
TAU = 6.0         # response time constant (s)
FATIGUE = 0.10    # total drop over the talk


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def manuscript_sections(path: Path) -> list[tuple[str, list[str]]]:
    text = path.read_text()
    parts = re.split(r"^## \d+\. (.+)$", text, flags=re.M)
    out = []
    for title, body in zip(parts[1::2], parts[2::2]):
        out.append((title.strip(), norm(body).split()))
    return out


def locate_sections(words: list[dict], sections) -> list[tuple[str, float]]:
    """Return (title, start_time) for each section by fuzzy-matching its first words."""
    tw = [norm(w["word"]) for w in words]
    starts, pos = [], 0
    for title, sw in sections:
        probe = " ".join(sw[:8])
        best, best_i = -1.0, pos
        for i in range(pos, max(pos + 1, len(tw) - 3)):
            cand = " ".join(tw[i:i + 8])
            r = difflib.SequenceMatcher(None, probe, cand).ratio()
            if r > best:
                best, best_i = r, i
        starts.append((title, words[best_i]["start"] if words else 0.0))
        pos = best_i + 1
        print(f"  {title:<30} starts at {starts[-1][1]:6.1f} s  (match {best:.2f})")
    return starts


def simulate(words_json: Path, manuscript: Path, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(words_json.read_text())
    words = [w for s in data["segments"] for w in s["words"]]
    starts = locate_sections(words, manuscript_sections(manuscript))
    dur = max(data["duration"], words[-1]["end"] if words else 0) + 5
    t = np.arange(0, dur, 1 / FS)
    rng = np.random.default_rng(seed)

    # Piecewise target with in-section decay, shifted by the reaction delay.
    target = np.full_like(t, 0.55)  # before the talk starts: neutral
    bounds = [s for _, s in starts] + [dur]
    for (title, s0), s1 in zip(starts, bounds[1:]):
        m = (t >= s0 + DELAY) & (t < s1 + DELAY)
        lvl = LEVELS.get(title, 0.5)
        since = t[m] - (s0 + DELAY)
        # high sections fade slowly, low sections sag faster
        decay = 0.0015 if lvl > 0.6 else 0.004
        target[m] = lvl * np.exp(-decay * since) + (1 - np.exp(-decay * since)) * 0.2

    # First-order lag (sluggish group response).
    a = 1 - np.exp(-1 / (FS * TAU))
    score = np.empty_like(target)
    score[0] = target[0]
    for i in range(1, len(t)):
        score[i] = score[i - 1] + a * (target[i] - score[i - 1])

    score -= FATIGUE * t / dur
    noise = np.zeros_like(t)
    for i in range(1, len(t)):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.015)
    score += noise
    # One spurious 2-second spike (e.g. a door slamming) during the methodology section.
    meth = dict(starts).get("Study methodology")
    if meth is not None:
        k = (t > meth + 12) & (t < meth + 14)
        score[k] += 0.25
    return t, np.clip(score, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("words_json")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--manuscript", default=str(HERE / "manuscript.md"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out or Path(a.words_json).parent / "attention.csv")
    t, s = simulate(Path(a.words_json), Path(a.manuscript), a.seed)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "attention"])
        w.writerows(zip(np.round(t, 2), np.round(s, 4)))
    print(f"Saved {out} ({len(t)} samples, mean {s.mean():.2f})")


if __name__ == "__main__":
    main()
