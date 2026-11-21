#!/usr/bin/env python3
"""Fit logistic regression on recorded sessions: is the user doomscrolling?

    python train.py                       # all data/session_*.csv
    python train.py data/session_A.csv data/session_B.csv
    python train.py --include-fake        # also use *_fake sessions

Validation never trains and tests on neighbouring rows (those are nearly
identical). With 3+ sessions it holds out one whole session at a time;
otherwise it holds out whole label segments (one stretch between two SPACE
presses).

Outputs: metrics in the terminal, reports/coefficients.png,
reports/timeline.png, models/doomscroll_lr.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = ["session", "wall_time", "t", "label", "label_name", "secs_since_switch", "cam_fps",
        "segment"]

# palette: diverging pair (orange = pushes toward doomscrolling, blue = toward attentive)
ORANGE, BLUE = "#eb6834", "#2a78d6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load(paths, include_fake):
    if not paths:
        paths = sorted(Path("data").glob("session_*.csv"))
        if not include_fake:
            paths = [p for p in paths if not p.stem.endswith("_fake")]
    if not paths:
        sys.exit("no sessions found in data/ (record some with record.py, or pass --include-fake)")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["session"] = df["session"].astype(str)
        # segment id = which stretch between label switches this row belongs to
        df["segment"] = df["session"] + ":" + (df["label"].diff().fillna(0) != 0).cumsum().astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True), paths


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def plot_coefficients(coefs: pd.Series, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = coefs.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(c) + 1.4), facecolor=SURFACE)
    style(ax)
    ax.barh(c.index, c.values, height=0.6, color=[ORANGE if v > 0 else BLUE for v in c.values],
            edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(c.values):
        ax.text(v + (0.03 if v >= 0 else -0.03) * max(1, abs(c).max()), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.set_xlabel("change in log-odds of doomscrolling per +1 standard deviation", color=INK2, fontsize=9)
    ax.set_title("What gives a doomscroller away", loc="left", color=INK, fontsize=12, weight="bold",
                 pad=20)
    ax.text(0, 1.01, "orange pushes toward doomscrolling, blue toward attentive",
            transform=ax.transAxes, color=INK2, fontsize=8, va="bottom")
    lim = abs(c).max() * 1.3 or 1
    ax.set_xlim(-lim, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_timeline(df: pd.DataFrame, p: np.ndarray, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session = df["session"].iloc[-1]  # most recent session
    m = (df["session"] == session).to_numpy()
    d, pp = df[m], p[m]
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor=SURFACE)
    style(ax)
    ax.fill_between(d["t"], 0, 1, where=d["label"] == 1, step="post", color=ORANGE, alpha=0.15,
                    linewidth=0, label="labelled doomscrolling (SPACE)")
    # break the line where rows were dropped (just after a switch) instead of bridging the gap
    pp = np.where(d["t"].diff().fillna(0).to_numpy() > 0.6, np.nan, pp)
    ax.plot(d["t"], pp, color=INK, linewidth=2, label="model P(doomscrolling), held-out")
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylim(0, 1)
    ax.set_xlabel("seconds into session", color=INK2, fontsize=9)
    ax.set_title(f"Session {session}", loc="left", color=INK, fontsize=12, weight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", type=Path)
    ap.add_argument("--include-fake", action="store_true")
    ap.add_argument("--guard", type=float, default=1.0,
                    help="drop rows less than N s after a label switch (default 1.0)")
    ap.add_argument("--warmup", type=float, default=5.0,
                    help="drop the first N s of each session, while 5 s windows fill (default 5)")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 regularisation strength")
    a = ap.parse_args(argv)

    df, paths = load(a.csv, a.include_fake)
    n_all = len(df)
    df = df[(df["secs_since_switch"] >= a.guard) & (df["t"] >= a.warmup)].reset_index(drop=True)
    features = [c for c in df.columns if c not in META and df[c].notna().any()]
    dropped = [c for c in df.columns if c not in META and c not in features]
    X, y = df[features].astype(float), df["label"].to_numpy()

    print(f"{len(paths)} session(s), {n_all} rows -> {len(df)} after dropping warm-up and "
          f"rows <{a.guard}s after a switch")
    print(f"doomscrolling share: {y.mean():.0%}")
    if dropped:
        print(f"no data at all (not plugged in yet?): {', '.join(dropped)}")
    if len(np.unique(y)) < 2:
        sys.exit("need rows of both labels; press SPACE during recording")

    n_sessions = df["session"].nunique()
    if n_sessions >= 3:
        groups, cv, how = df["session"], LeaveOneGroupOut(), "leave-one-session-out"
    else:
        groups = df["segment"]
        k = min(5, groups.nunique())
        if k < 2:
            sys.exit("need at least 2 label segments to validate")
        cv, how = GroupKFold(n_splits=k), f"{k}-fold over label segments"

    def model():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=a.C, max_iter=1000))

    p = np.full(len(df), np.nan)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue  # this fold's training set has only one label
        p[te] = model().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    ok = ~np.isnan(p)
    pred = (p[ok] >= 0.5).astype(int)
    print(f"\nvalidation ({how}, {ok.sum()} rows):")
    print(f"  balanced accuracy {balanced_accuracy_score(y[ok], pred):.3f}")
    if len(np.unique(y[ok])) == 2:
        print(f"  ROC AUC           {roc_auc_score(y[ok], p[ok]):.3f}")
    tn, fp, fn, tp = confusion_matrix(y[ok], pred, labels=[0, 1]).ravel()
    print(f"  confusion: attentive called attentive {tn}, called doomscrolling {fp}")
    print(f"             doomscrolling called doomscrolling {tp}, called attentive {fn}")

    final = model().fit(X, y)
    lr = final[-1]
    coefs = pd.Series(lr.coef_[0], index=features)
    print(f"\ncoefficients (log-odds per +1 sd), intercept {lr.intercept_[0]:+.2f}:")
    for name, v in coefs.reindex(coefs.abs().sort_values(ascending=False).index).items():
        print(f"  {name:22s} {v:+.3f}  {'doomscrolling' if v > 0 else 'attentive'}")

    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    plot_coefficients(coefs, Path("reports/coefficients.png"))
    plot_timeline(df[ok].reset_index(drop=True), p[ok], Path("reports/timeline.png"))
    joblib.dump({"model": final, "features": features}, "models/doomscroll_lr.joblib")
    print("\nwrote reports/coefficients.png, reports/timeline.png, models/doomscroll_lr.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
