"""Fit the attention model on the sessions recorded with collect.py.

    python train.py                       # every data/attention_*.csv -> models/attention_model.json
    python train.py --windows 1,4,10      # other averaging windows (seconds) than the default 2,6
    python train.py --add face_down_score --out models/attention_model_face_down.json
                                          # a second model that also uses the head tilt
    python train.py --decay 6 --recover 2 # another ground-truth rule
    python train.py --no-save ...         # just show the scores, to compare settings

The recordings hold only the plain values of every moment, so everything
below is decided here and can be changed without recording again:

  --signals   which recorded columns the model uses (default: attention.SIGNALS)
  --add       columns to use on top of those, for example face_down_score
  --windows   the model's inputs are each signal averaged over these last seconds
  --decay     seconds of scripted distraction for the target attention to fall from 1 to 0
  --recover   seconds back on task for it to return from 0 to 1
  --settle    rows this soon after a change of action are left out of the fit: people need
              a moment to react to the beep, so the camera did not yet see what the script says
  --l2        how strongly the weights are shrunk

It prints how well the model does on a person it has not seen: it is fitted on
all but one person and scored on that person, for each person in turn (with
three people: three times, each fitted on the other two). With only one
person's data that is impossible, and the scores shown are on the training
data itself, which flatters the model.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from attention import DECAY_S, MODEL_PATH, RECOVER_S, SIGNALS, WINDOWS_S, History, feature_names, fit, next_target

ROOT = Path(__file__).resolve().parent
SETTLE_S = 1.0
SCRIPT_COLUMNS = ("person", "scenario", "t", "phase", "action", "distracted", "file")   # not measurements


def number(row, name):
    return float(row[name]) if row.get(name) not in ("", None) else 0.0


def load(folder):
    """Every row of every session file, as dicts, with the file name added.

    The first recordings were made when the product signal was gaze_down * phone
    (their gaze_down_x_phone column). gaze_down_x_phone_in_hand is rebuilt for
    them here from the two recorded columns it is the product of.
    """
    rows = []
    for path in sorted(Path(folder).glob("attention_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["file"] = path.name
                if "gaze_down_x_phone_in_hand" not in row:
                    row["gaze_down_x_phone_in_hand"] = number(row, "gaze_down") * number(row, "phone_in_hand")
                if "gaze_down_x_phone" not in row:          # and the old product for the newer recordings
                    row["gaze_down_x_phone"] = number(row, "gaze_down") * number(row, "phone")
                rows.append(row)
    return rows


def build(rows, windows=WINDOWS_S, decay=DECAY_S, recover=RECOVER_S, settle=SETTLE_S, signals=SIGNALS):
    """Features and targets from the plain recorded rows.

    For every recorded scenario, in time order: the signal history starts
    afresh (as the windows did live), the target follows the scripted
    distraction with the decay / recover rule, and rows within `settle`
    seconds of a change of action are skipped.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["file"], row["scenario"])].append(row)
    X, y, people, scenarios = [], [], [], []
    for (_, scenario), group in groups.items():
        group.sort(key=lambda r: float(r["t"]))
        history = History(windows, signals)
        attention = 0.0 if group[0]["distracted"] == "1" else 1.0   # they began during the countdown
        last_t, phase, phase_start = None, group[0]["phase"], 0.0
        for row in group:
            t = float(row["t"])
            distracted = row["distracted"] == "1"
            attention = next_target(attention, distracted, 0.0 if last_t is None else t - last_t, decay, recover)
            last_t = t
            if row["phase"] != phase:
                phase, phase_start = row["phase"], t
            history.add(t, {name: number(row, name) for name in signals})
            if t - phase_start < settle:
                continue
            X.append(history.features(t))
            y.append(attention)
            people.append(row["person"])
            scenarios.append(scenario)
    return np.array(X), np.array(y), np.array(people), np.array(scenarios)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=str(ROOT / "data"), help="folder with the attention_*.csv files")
    p.add_argument("--out", default=str(MODEL_PATH), help="where to write the model")
    p.add_argument("--signals", default=",".join(SIGNALS),
                   help="recorded columns the model uses, comma-separated (default %(default)s)")
    p.add_argument("--add", default="", help="columns to use on top of --signals, for example face_down_score")
    p.add_argument("--windows", default=",".join(f"{w:g}" for w in WINDOWS_S),
                   help="averaging windows in seconds, comma-separated (default %(default)s)")
    p.add_argument("--decay", type=float, default=DECAY_S, help="default %(default)s s")
    p.add_argument("--recover", type=float, default=RECOVER_S, help="default %(default)s s")
    p.add_argument("--settle", type=float, default=SETTLE_S, help="default %(default)s s")
    p.add_argument("--l2", type=float, default=1.0, help="how strongly to shrink the weights (default 1.0)")
    p.add_argument("--no-save", action="store_true", help="show the scores but do not write the model")
    args = p.parse_args()
    try:
        windows = tuple(float(w) for w in args.windows.split(","))
        assert windows and all(w > 0 for w in windows)
    except (ValueError, AssertionError):
        sys.exit("ERROR: --windows takes positive numbers of seconds, for example 2,6")

    rows = load(args.data)
    if not rows:
        sys.exit(f"ERROR: no attention_*.csv in {args.data}. Record some with collect.py first.")
    signals = tuple(dict.fromkeys(name.strip() for name in f"{args.signals},{args.add}".split(",") if name.strip()))
    missing = [name for name in signals if name not in rows[0]]
    if missing or not signals:
        sys.exit(f"ERROR: not a recorded column: {', '.join(missing) or '(none given)'}. "
                 f"Recorded: {', '.join(c for c in rows[0] if c not in SCRIPT_COLUMNS)}")
    X, y, people, scenarios = build(rows, windows, args.decay, args.recover, args.settle, signals)
    names = sorted(set(people))
    print(f"{len(rows)} rows from {len(names)} people ({', '.join(names)}); {len(y)} used after dropping "
          f"the first {args.settle:g} s after each change of action")
    unused = [c for c in rows[0] if c not in signals and c not in SCRIPT_COLUMNS]
    print(f"Model inputs: {', '.join(signals)}")
    print(f"Recorded but not used by the model: {', '.join(unused)}")
    print(f"Settings: windows {', '.join(f'{w:g} s' for w in windows)}; target falls over {args.decay:g} s "
          f"and recovers over {args.recover:g} s; l2 {args.l2:g}")

    # How well does it do on a person it has not seen?
    predicted = np.zeros(len(y))
    held_out = len(names) >= 2
    if held_out:
        for name in names:
            test = people == name
            predicted[test] = fit(X[~test], y[~test], args.l2, windows, signals).predict(X[test])
    else:
        predicted = fit(X, y, args.l2, windows, signals).predict(X)
    kind = "on people left out of the fit" if held_out else "ON THE TRAINING DATA (one person only: optimistic)"
    error = np.abs(predicted - y)
    agree = (predicted >= 0.5) == (y >= 0.5)
    print(f"\nScores {kind}:")
    print(f"  average error of the attention score: {error.mean():.2f}   (0 = perfect, 0.5 = no better than guessing 0.5)")
    print(f"  attentive / distracted called correctly: {100 * agree.mean():.0f} % of moments")
    if held_out:
        print("  per person:  " + "   ".join(
            f"{name} {error[people == name].mean():.2f} / {100 * agree[people == name].mean():.0f} %" for name in names))
    print(f"\n  {'scenario':28s} {'rows':>5s}  {'target':>6s}  {'model':>6s}  {'error':>5s}")
    for scenario in dict.fromkeys(scenarios):
        m = scenarios == scenario
        print(f"  {scenario:28s} {m.sum():5d}  {y[m].mean():6.2f}  {predicted[m].mean():6.2f}  {error[m].mean():5.2f}")

    model = fit(X, y, args.l2, windows, signals)
    model.info = {"fitted": time.strftime("%Y-%m-%d %H:%M"), "people": names, "rows": int(len(y)),
                  "decay_s": args.decay, "recover_s": args.recover, "settle_s": args.settle, "l2": args.l2,
                  "held_out_error": round(float(error.mean()), 3) if held_out else None,
                  "held_out_agreement": round(float(agree.mean()), 3) if held_out else None}
    print("\nWeights (per standard deviation of each input; negative lowers attention):")
    for name, weight in sorted(zip(feature_names(windows, signals), model.weights), key=lambda item: item[1]):
        print(f"  {name:32s} {weight:+.2f}")
    print(f"  {'(bias)':32s} {model.bias:+.2f}")
    if args.no_save:
        print("\nNot saved (--no-save).")
    else:
        model.save(args.out)
        default = Path(args.out).resolve() == MODEL_PATH.resolve()
        print(f"\nSaved to {args.out}. " + ("dashboard.py now shows the attention score." if default
                                            else f"Show it live with: dashboard.py --model {args.out}"))


if __name__ == "__main__":
    main()
