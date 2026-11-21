"""Guided data collection for the attention model: one person, nine short scenarios, about four minutes.

    python collect.py --person anna
    python collect.py --person anna --device 1     # name the colour camera index

The window tells the person exactly what to do. Each scenario has a "get
ready" page (what to set up, then SPACE), a 3-second countdown, and the
recording, during which the current action is shown in large letters with the
seconds left. Everything is also said in sound, because during the phone
scenarios the person is not looking at the laptop: three short low beeps count
down 3, 2, 1, a long beep starts the recording, a beep marks every change of
action, and a double beep ends the scenario. Turn the sound on.

What is recorded into data/attention_<person>_<time>.csv, one row for every
camera frame of every recording (roughly 10 to 30 a second): the time, the
scripted action and whether it is a distraction, every signal (see
look_down.signals), and the yes/no phone signals and the measurements behind
the signals (engine.RAW). Only these numbers are saved: no video and no
pictures.

Every signal is saved, including the ones left out of the dashboard list and
of the model: face_down, face_down_score, the yes/no gaze_down, and the yes/no
phone ones. The model uses only attention.SIGNALS.

Nothing derived is stored. The target attention, the averages over the last
seconds and the "just changed action" margin are all worked out by train.py
from these rows, so their settings can be changed afterwards without
recording again.

The scenarios are chosen so that each signal is seen both with and without
real distraction:

  on task      phone away / phone lying on the desk / looking down at keyboard
               or notes beside the phone / holding the phone but reading the screen
  distracted   scrolling above the desk / in the lap / with the phone flat on the desk
  changing     two 2-second glances at the phone (attention should hardly dip),
               and work -> 20 s of scrolling -> work

Keys: SPACE continue   r redo the scenario just recorded   q or Esc quit (finished scenarios are kept)

Then: python train.py   and   python dashboard.py
"""

import argparse
import csv
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from engine import RAW

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WINDOW = "attention data collection"
WIDTH, HEIGHT = 1280, 720
SPACE = 32
COUNTDOWN_S = 3.0     # "start now" time before a recording begins

SIGNAL_COLUMNS = ("face_down", "face_down_score", "gaze_down", "gaze_down_score", "color_phone", "phone",
                  "phone_in_hand", "gaze_down_x_phone")
COLUMNS = ("person", "scenario", "t", "phase", "action", "distracted") + SIGNAL_COLUMNS + RAW

Phase = namedtuple("Phase", "seconds action distracted")
Scenario = namedtuple("Scenario", "name title setup phases")

WORK = "WORK ON THE LAPTOP"
SCENARIOS = (
    Scenario("work_phone_away", "Working, phone out of sight", (
        "Put your phone away: pocket or bag, out of sight.",
        "Hands on the keyboard and mouse.",
        "Read and type the way you normally work."),
        (Phase(15, WORK, False),)),
    Scenario("work_phone_on_desk", "Working, phone lying on the desk", (
        "Lay your phone flat on the desk beside the keyboard, screen up.",
        "In the small grey picture the phone should get a red box.",
        "Do not touch the phone. Read and type on the laptop."),
        (Phase(15, WORK, False),)),
    Scenario("look_down_phone_on_desk", "Looking down at the keyboard or notes", (
        "Leave the phone on the desk and do not touch it.",
        "Look down at the keyboard while you type, or write on paper.",
        "Do not look at the phone."),
        (Phase(15, "LOOK DOWN AT THE KEYBOARD OR YOUR NOTES", False),)),
    Scenario("hold_phone_eyes_on_screen", "Holding the phone, reading the screen", (
        "Pick up your phone and hold it in one hand above the desk.",
        "Keep your eyes on the laptop screen the whole time.",
        "Do not look at the phone."),
        (Phase(15, "HOLD THE PHONE, READ THE LAPTOP SCREEN", False),)),
    Scenario("scroll_at_desk", "Scrolling, phone held above the desk", (
        "Hold your phone above the desk, the way you usually scroll.",
        "Open any app and scroll. Look at the phone, not the laptop.",
        "Keep scrolling until you hear the double beep."),
        (Phase(20, "SCROLL ON YOUR PHONE", True),)),
    Scenario("scroll_in_lap", "Scrolling, phone in your lap", (
        "Hold your phone in your lap, below the edge of the desk.",
        "Scroll, and look at the phone.",
        "Keep scrolling until you hear the double beep."),
        (Phase(20, "SCROLL ON YOUR PHONE IN YOUR LAP", True),)),
    Scenario("scroll_flat_on_desk", "Scrolling, phone flat on the desk", (
        "Lay the phone flat on the desk in front of you.",
        "Scroll with one finger, and look at the phone.",
        "Keep scrolling until you hear the double beep."),
        (Phase(15, "SCROLL WITH THE PHONE FLAT ON THE DESK", True),)),
    Scenario("glances", "Working, with two quick glances at the phone", (
        "Lay the phone on the desk beside the keyboard.",
        "Work on the laptop. At a beep: pick up the phone and look at it.",
        "At the next beep, two seconds later: put it down and work again.",
        "This happens twice."),
        (Phase(8, WORK, False), Phase(2, "GLANCE AT YOUR PHONE", True), Phase(9, WORK, False),
         Phase(2, "GLANCE AT YOUR PHONE", True), Phase(11, WORK, False))),
    Scenario("work_scroll_work", "Working, then scrolling, then working again", (
        "Lay the phone on the desk beside the keyboard.",
        "Work on the laptop. At the beep: pick up the phone and scroll.",
        "At the next beep, 20 seconds later: put it down and work again."),
        (Phase(10, WORK, False), Phase(20, "SCROLL ON YOUR PHONE", True), Phase(10, WORK, False))),
)

WHITE, GREY, GREEN, ORANGE, YELLOW, RED, DARK = ((255, 255, 255), (150, 150, 150), (80, 200, 80), (0, 140, 255),
                                                 (0, 255, 255), (0, 0, 255), (30, 30, 30))


def beep(kind):
    """The sounds of a scenario. Never blocks the camera.

    "tick"    short and low: one for each second of the countdown (3, 2, 1)
    "start"   long: recording starts now
    "change"  change of action
    "end"     double beep: the scenario is over
    """
    def play():
        try:
            import winsound
        except ImportError:        # not Windows: silent
            return
        if kind == "end":
            winsound.Beep(1175, 160)
            time.sleep(0.09)
            winsound.Beep(1175, 160)
        elif kind == "tick":
            winsound.Beep(587, 120)
        elif kind == "start":
            winsound.Beep(880, 450)
        else:
            winsound.Beep(880, 200)
    threading.Thread(target=play, daemon=True).start()


class Session:
    """One person's run through the scenarios: what is on screen now, and the rows recorded."""

    def __init__(self, person, engine, path, scenarios=SCENARIOS):
        self.person, self.engine, self.path, self.scenarios = person, engine, Path(path), scenarios
        self.mode = "intro"          # intro, calibrate, ready, countdown, run, saved, finished
        self.index = 0               # which scenario
        self.done = {}               # scenario name -> its rows
        self.quit = False
        self._mark = 0.0             # when the current countdown or recording started
        self._rows, self._phase_i = [], 0
        self._ticks = 0              # countdown beeps played so far

    # --- what the drawing needs -------------------------------------------------
    @property
    def scenario(self):
        return self.scenarios[self.index]

    def elapsed(self, now):
        return now - self._mark

    def phase_at(self, t):
        """(phase index, seconds left in it) at `t` seconds into the scenario."""
        for i, phase in enumerate(self.scenario.phases):
            if t < phase.seconds or i == len(self.scenario.phases) - 1:
                return i, max(phase.seconds - t, 0.0)
            t -= phase.seconds

    @property
    def total_s(self):
        return sum(p.seconds for p in self.scenario.phases)

    # --- the state machine --------------------------------------------------------
    def update(self, step, key):
        now, det = step.now, self.engine.det
        if key in (ord("q"), 27):
            self.quit = True
        elif self.mode == "intro" and key == SPACE:
            beep("change")                              # lets the person check the sound is on
            det.calibrate("screen")
            self.mode = "calibrate"
        elif self.mode == "calibrate" and not det.calibrating and det.base is not None:
            self.mode = "ready"
        elif self.mode == "ready" and key == SPACE:
            self.mode, self._mark, self._ticks = "countdown", now, 1
            beep("tick")                                # "3"
        elif self.mode == "countdown" and self.elapsed(now) >= COUNTDOWN_S:
            self.mode, self._mark = "run", now
            self._rows, self._phase_i = [], 0
            beep("start")
        elif self.mode == "countdown":
            due = int(self.elapsed(now)) + 1            # a beep as each of "2" and "1" appears
            if due > self._ticks:
                self._ticks = due
                beep("tick")
        elif self.mode == "run":
            self._record(step)
        elif self.mode == "saved" and key == ord("r"):
            del self.done[self.scenario.name]
            self._save()
            self.mode = "ready"
        elif self.mode == "saved" and key == SPACE:
            if self.index + 1 < len(self.scenarios):
                self.index += 1
                self.mode = "ready"
            else:
                self.mode = "finished"

    def _record(self, step):
        t = self.elapsed(step.now)
        if t >= self.total_s:
            self.done[self.scenario.name] = self._rows
            self._save()
            beep("end")
            self.mode = "saved"
            return
        i, _ = self.phase_at(t)
        if i != self._phase_i:
            self._phase_i = i
            beep("change")
        phase = self.scenario.phases[i]
        row = {"person": self.person, "scenario": self.scenario.name, "t": round(t, 3), "phase": i,
               "action": phase.action, "distracted": int(phase.distracted)}
        for name in SIGNAL_COLUMNS:
            value = step.values.get(name)
            row[name] = "" if value is None else round(float(value), 4)
        for name in RAW:
            value = step.raw.get(name)
            row[name] = "" if value is None else round(float(value), 4)
        self._rows.append(row)

    def _save(self):
        """Rewrite the file with every finished scenario (so a redo replaces the old rows)."""
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            for scenario in self.scenarios:
                writer.writerows(self.done.get(scenario.name, []))


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

def text(canvas, s, x, y, scale, colour=WHITE, thick=2):
    cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def centred(canvas, s, y, scale, colour=WHITE, thick=2, max_width=WIDTH - 80):
    """Centred text, shrunk if needed so it fits `max_width`."""
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if w > max_width:
        scale *= max_width / w
        thick = max(1, int(round(thick * max_width / w)))
        (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    text(canvas, s, (canvas.shape[1] - w) // 2, y, scale, colour, thick)


def previews(canvas, session, step):
    """Small pictures of both streams, so the person can see the camera has them and the phone in view."""
    from dashboard import colour_view, infrared_view
    engine = session.engine
    colour = colour_view(step.frame.copy(), engine.face(step.now), engine.worker.phones["rgb"], engine.det, True)
    infrared = infrared_view(step.ir_frame, engine.worker, True)
    h = 220
    colour = cv2.resize(colour, (round(colour.shape[1] * h / colour.shape[0]), h), interpolation=cv2.INTER_AREA)
    infrared = cv2.resize(infrared, (round(infrared.shape[1] * h / infrared.shape[0]), h), interpolation=cv2.INTER_AREA)
    x = WIDTH - 30 - colour.shape[1] - 10 - infrared.shape[1]
    canvas[80:80 + h, x:x + colour.shape[1]] = colour
    canvas[80:80 + h, x + colour.shape[1] + 10:x + colour.shape[1] + 10 + infrared.shape[1]] = infrared
    return x


def bullets(canvas, lines, y, max_x):
    for line in lines:
        scale = 0.85
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if 76 + w > max_x:                          # keep clear of the previews
            scale *= (max_x - 76) / w
        cv2.circle(canvas, (52, y - 9), 6, YELLOW, -1)
        text(canvas, line, 76, y, scale)
        y += 52
    return y


def draw(session, step):
    """The whole window for the current moment."""
    canvas = np.full((HEIGHT, WIDTH, 3), DARK, np.uint8)
    now, mode = step.now, session.mode
    n = len(session.scenarios)
    if mode in ("ready", "countdown", "run", "saved"):
        text(canvas, f"{session.person}   scenario {session.index + 1} of {n}:   {session.scenario.title}",
             30, 46, 0.8, GREY)

    if mode == "intro":
        centred(canvas, "ATTENTION DATA COLLECTION", 60, 1.2)
        left = previews(canvas, session, step) - 20
        y = bullets(canvas, ("Turn your sound on.",
                             "Three short beeps count down 3, 2, 1. The long beep means: start.",
                             "A beep after that means: change what you are doing.",
                             f"There are {n} short scenarios, about 4 minutes."), 130, left)
        bullets(canvas, ("Each one first shows what to set up. Take your time there.",
                         "Then do exactly what the large text says, until the beep.",
                         "A double beep means: stop, and look at the laptop again.",
                         "Sit the way you normally work, facing the laptop."), max(y, 350), WIDTH - 40)
        centred(canvas, "Press SPACE to begin (you should hear a beep)", 660, 1.1, YELLOW)

    elif mode == "calibrate":
        centred(canvas, "LOOK AT THIS TEXT", 330, 2.4, YELLOW, 5)
        centred(canvas, "Sit normally and keep looking here for a second ...", 420, 1.0)

    elif mode == "ready":
        left = previews(canvas, session, step) - 20
        text(canvas, "GET READY", 30, 120, 1.5, YELLOW, 3)
        y = bullets(canvas, session.scenario.setup, 185, left)
        y = max(y, 350) + 20
        text(canvas, "What will happen:", 30, y, 0.8, GREY)
        for phase in session.scenario.phases:
            y += 42
            text(canvas, f"{phase.seconds:>3} s", 52, y, 0.8, GREY)
            text(canvas, phase.action, 140, y, 0.8, ORANGE if phase.distracted else GREEN)
        centred(canvas, "Press SPACE when you are ready", 680, 1.2, YELLOW, 3)

    elif mode == "countdown":
        left = COUNTDOWN_S - session.elapsed(now)
        first = session.scenario.phases[0]
        centred(canvas, "START NOW:", 200, 1.4, YELLOW, 3)
        centred(canvas, first.action, 300, 2.2, ORANGE if first.distracted else GREEN, 5)
        centred(canvas, str(max(int(np.ceil(left)), 1)), 520, 6.0, WHITE, 12)
        centred(canvas, "recording starts at the long beep", 640, 0.9, GREY)

    elif mode == "run":
        t = session.elapsed(now)
        i, left = session.phase_at(t)
        phase = session.scenario.phases[i]
        colour = ORANGE if phase.distracted else GREEN
        cv2.circle(canvas, (WIDTH - 150, 38), 10, RED, -1)
        text(canvas, "REC", WIDTH - 130, 46, 0.8, RED)
        centred(canvas, phase.action, 290, 2.6, colour, 6)
        centred(canvas, f"{int(np.ceil(left))} s", 420, 2.6, WHITE, 5)
        if i + 1 < len(session.scenario.phases):
            centred(canvas, f"at the beep:  {session.scenario.phases[i + 1].action}", 520, 1.1, GREY)
        else:
            centred(canvas, "keep going until the double beep", 520, 1.1, GREY)
        # progress through the scenario, with a tick at every change of action
        x0, x1, y0, y1 = 60, WIDTH - 60, 600, 630
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + int((x1 - x0) * min(t / session.total_s, 1.0)), y1), colour, -1)
        at = 0
        for p in session.scenario.phases[:-1]:
            at += p.seconds
            tick = x0 + int((x1 - x0) * at / session.total_s)
            cv2.line(canvas, (tick, y0 - 8), (tick, y1 + 8), WHITE, 2)

    elif mode == "saved":
        left = previews(canvas, session, step) - 20
        text(canvas, "DONE. Look at the laptop again.", 30, 130, 1.2, GREEN, 3)
        rows = len(session.done.get(session.scenario.name, []))
        bullets(canvas, (f"{rows} rows recorded for this scenario.",
                         "If something went wrong, press R to do it again."), 200, left)
        last = session.index + 1 == n
        centred(canvas, "Press SPACE to finish" if last else "Press SPACE for the next scenario", 680, 1.2, YELLOW, 3)

    else:   # finished
        centred(canvas, "ALL DONE. THANK YOU!", 300, 2.2, GREEN, 5)
        centred(canvas, f"Saved to {session.path.parent.name}/{session.path.name}", 400, 0.9)
        centred(canvas, "Press Q to close", 660, 1.1, YELLOW)
    return canvas


def run(session, show=None, wait_key=None):
    """Drive a session until the person quits. `show` and `wait_key` default to the OpenCV window."""
    show = show or (lambda canvas: cv2.imshow(WINDOW, canvas))
    wait_key = wait_key or (lambda: cv2.waitKey(1) & 0xFF)
    while not session.quit:
        step = session.engine.step()
        if step is None:
            if wait_key() in (ord("q"), 27):
                break
            continue
        show(draw(session, step))
        session.update(step, wait_key())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--person", required=True, help="a short name for whoever is sitting at the laptop")
    p.add_argument("--device", type=int, help="colour camera index (default: find the LogiInsight camera)")
    p.add_argument("--ir-device", type=int, help="infrared camera index (default: auto-detect)")
    args = p.parse_args()

    from engine import Engine, EngineError
    try:
        engine = Engine(args.device, args.ir_device)
    except EngineError as e:
        sys.exit(f"ERROR: {e}")
    person = "".join(c for c in args.person if c.isalnum() or c in "-_") or "person"
    path = DATA_DIR / f"attention_{person}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    session = Session(person, engine, path)
    try:
        run(session)
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        cv2.destroyAllWindows()
    rows = sum(len(r) for r in session.done.values())
    print(f"{len(session.done)} of {len(SCENARIOS)} scenarios, {rows} rows"
          + (f", saved to {path}" if session.done else "; nothing saved"))


if __name__ == "__main__":
    main()
