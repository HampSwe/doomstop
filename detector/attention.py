"""The attention score: its inputs over time, its ground truth during data collection, and the model.

Inputs. By default the five signals in SIGNALS, each averaged over the last 2
and the last 6 seconds (WINDOWS_S). The recordings hold more columns than that
(face_down, face_down_score, the yes/no gaze_down and phone signals, raw
measurements); those are not inputs unless a model is fitted with them
(train.py --add). A fitted model remembers its own signal list. The averages
are what makes the score depend on time: a two-second glance at the phone
barely moves the 6-second average, half a minute of scrolling fills it.

The windows, and the two times of the ground-truth rule below, are not baked
into the recorded data: collect.py stores the plain values of every moment,
and train.py applies whatever windows and rule it is given (the defaults are
the constants here). The fitted model remembers its windows, so the live
score uses the same ones.

Ground truth. During collection every moment is scripted as "on task" or
"distracted". The target attention follows that with a delay, because a quick
glance is not lost attention: it falls from 1 to 0 over DECAY_S seconds of
distraction and climbs back over RECOVER_S seconds on task.

Model. Logistic regression on the standardised features, fitted to that target
(which can be any value from 0 to 1, not just 0 or 1). Written out here, as
scikit-learn is not installed and ten inputs do not need it.
"""

import json
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # the repository folder
MODEL_PATH = ROOT / "models" / "attention_model.json"

SIGNALS = ("gaze_down_score", "color_phone", "phone", "phone_in_hand", "gaze_down_x_phone_in_hand")
WINDOWS_S = (2.0, 6.0)

DECAY_S = 4.0       # seconds of distraction for the target to fall from 1 to 0
RECOVER_S = 1.5     # seconds on task for it to climb back from 0 to 1


def feature_names(windows=WINDOWS_S, signals=SIGNALS):
    return tuple(f"{name}_{window:g}s" for name in signals for window in windows)


def next_target(attention, distracted, dt, decay=DECAY_S, recover=RECOVER_S):
    """The target attention `dt` seconds later."""
    change = -dt / decay if distracted else dt / recover
    return min(max(attention + change, 0.0), 1.0)


class History:
    """The recent signal values, turned into the model's features."""

    def __init__(self, windows=WINDOWS_S, signals=SIGNALS):
        self.windows, self.signals = tuple(windows), tuple(signals)
        self._rows = deque()            # (time, [value per signal])

    def add(self, now, values):
        """`values`: a signals dict. A missing or switched-off signal counts as 0."""
        self._rows.append((now, [float(values.get(name) or 0.0) for name in self.signals]))
        while now - self._rows[0][0] > max(self.windows):
            self._rows.popleft()

    def features(self, now):
        """Each signal's average over each window, in the order of feature_names(windows, signals)."""
        times = np.array([t for t, _ in self._rows])
        data = np.array([row for _, row in self._rows])
        means = [data[times > now - window].mean(axis=0) if (times > now - window).any() else data[-1]
                 for window in self.windows]
        return np.array([means[w][s] for s in range(len(self.signals)) for w in range(len(self.windows))])


class Model:
    """attention = sigmoid(bias + weights . (features - mean) / scale)"""

    def __init__(self, weights, bias, mean, scale, windows=WINDOWS_S, info=None, signals=SIGNALS):
        self.weights, self.bias = np.asarray(weights, float), float(bias)
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.windows = tuple(float(w) for w in windows)   # the averaging windows it was fitted with
        self.signals = tuple(signals)                     # the signals it was fitted on
        self.info = info or {}

    def predict(self, features):
        """Attention, 0..1, for one feature vector or one per row."""
        z = self.bias + ((np.asarray(features, float) - self.mean) / self.scale) @ self.weights
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def save(self, path=MODEL_PATH):
        Path(path).parent.mkdir(exist_ok=True)
        Path(path).write_text(json.dumps({
            "signals": list(self.signals), "windows": list(self.windows),
            "features": list(feature_names(self.windows, self.signals)),
            "weights": self.weights.tolist(), "bias": self.bias,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(), "info": self.info}, indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path=MODEL_PATH):
        """The saved model, with the signals and windows it was fitted on, or None if there is no such file."""
        path = Path(path)
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["weights"], d["bias"], d["mean"], d["scale"], d["windows"], d.get("info"),
                   d.get("signals", SIGNALS))


def fit(X, y, l2=1.0, windows=WINDOWS_S, signals=SIGNALS):
    """Logistic regression of targets y (0..1) on feature rows X, by Newton's method.

    `windows` and `signals` are what X was built with; the model remembers them.

    `l2` shrinks the weights (not the bias); larger means smaller weights.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-6] = 1.0                       # a signal that never moved: leave it unscaled
    Z = np.hstack([np.ones((len(X), 1)), (X - mean) / scale])
    penalty = np.eye(Z.shape[1]) * l2 / len(Z)
    penalty[0, 0] = 0.0
    w = np.zeros(Z.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        gradient = Z.T @ (p - y) / len(Z) + penalty @ w
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z / len(Z) + penalty + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return Model(w[1:], w[0], mean, scale, windows, signals=signals)
