"""One window: the colour and infrared streams side by side, the attention score, and every signal.

The attention score (0 = distracted, 1 = attentive) is the model fitted by
train.py on data recorded with collect.py. It uses the signals averaged over
the last few seconds, so it follows what you do with a short delay. Until a
model has been fitted the row says so.

    python dashboard.py               # finds the LogiInsight colour and infrared cameras
    python dashboard.py --device 1    # or name the colour camera index
    python dashboard.py --face        # also show face_down and face_down_score
    python dashboard.py --binary-gaze # also show gaze_down (0 or 1)
    python dashboard.py --model models/attention_model_face_down.json   # another fitted model
    python dashboard.py --popup       # also pop up a friendly card when you doomscroll (this is focus.py)

Signals (scores are 0..1; the white tick on a bar is where it counts as "on"):

    gaze_down_score      how far the eyelids have dropped           colour stream, camera's face data
                         (0.5 = looking below the screen)
    color_phone          phone detector confidence                  colour stream
    phone                phone detector confidence                  infrared stream
    phone_in_hand        how steadily a hand has been at that phone infrared stream
    gaze_down_x_phone_in_hand   gaze_down * phone_in_hand: the phone-in-hand score while looking down, else 0

Left out of the list for now, but still computed (HIDDEN below): the yes/no
gaze_down, which --binary-gaze brings back, and face_down / face_down_score
(head tilt), which --face brings back. gaze_down_x_phone_in_hand keeps using
gaze_down either way.

Needs the LogiInsight camera and its login (LOGIWEBCAM_USER / LOGIWEBCAM_PASSWORD),
like look_down.py, which this builds on. Look at the screen for the first
second: that is when your normal pose is measured.

Keys: c  re-measure the "looking at the screen" pose
      d  measure a "looking down" pose and set the thresholds from it
      r  reset the thresholds
      q  or Esc: quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from attention import MODEL_PATH, History, Model
from engine import Engine, EngineError
from hand_detect import IN_HAND_ON, draw_hands
from nudge import AFTER_S, BELOW, COOLDOWN_S, Nudger
from look_down import AI_WARN_S, GREEN, GREY, RED, WHITE, YELLOW, draw_face, draw_phones, put, put_centred
from phone_detect import enhance_grey

WINDOW = "doomscroll dashboard (q / Esc to quit)"
PAD, GAP = 16, 12
VIEW_H = 495
COLOUR_W, IR_W = 880, 396          # 16:9 and 4:5 at that height
WIDTH = PAD + COLOUR_W + GAP + IR_W + PAD
ROW_H = 38
BACKGROUND = (30, 30, 30)
MESSAGE_S = 4.0                    # how long a note such as "screen pose saved" stays in the bottom line
# Signals that are computed but not in the list, and the option that brings each group back.
HIDDEN = {"face": ("face_down", "face_down_score"), "binary_gaze": ("gaze_down",)}


def colour_view(frame, face, phones, det, mirror):
    """The colour picture with the face and phone marked, at panel size. Draws on `frame`."""
    draw_face(frame, face, False)
    draw_phones(frame, phones)
    if mirror:
        frame = cv2.flip(frame, 1)
    view = cv2.resize(frame, (COLOUR_W, VIEW_H), interpolation=cv2.INTER_AREA)
    put(view, "colour", (12, 30))
    if det.calibrating:
        target = "LOOK AT THE SCREEN" if det.calibrating == "screen" else "LOOK DOWN (at your phone)"
        put_centred(view, target, VIEW_H // 2, 1.1, YELLOW, 2)
        put_centred(view, f"measuring {int(det.progress * 100)} %", VIEW_H // 2 + 40, 0.7, YELLOW, 2)
    elif det.base is None:
        put_centred(view, "waiting for the face mesh", VIEW_H // 2, 0.9, GREY, 2)
    return view


def infrared_view(frame, worker, mirror):
    """The infrared picture as the detectors see it, with the phone and hands marked, at panel size."""
    if frame is None:
        view = np.full((VIEW_H, IR_W, 3), 60, np.uint8)
        put_centred(view, "waiting for infrared", VIEW_H // 2, 0.7, GREY, 2)
        return view
    view = enhance_grey(frame)
    draw_phones(view, worker.phones["ir"])
    draw_hands(view, worker.hand_points)
    if mirror:
        view = cv2.flip(view, 1)
    view = cv2.resize(view, (IR_W, VIEW_H), interpolation=cv2.INTER_LINEAR)
    put(view, "infrared" + (f": {worker.use.state}" if worker.use else ""), (12, 30))
    return view


def draw_signal(canvas, x, y, name, value, threshold):
    """One signal: name, value, and a bar that is red at or above `threshold` (marked by a white tick)."""
    put(canvas, name, (x, y + 17), 0.6, WHITE, 1, backing=False)
    text = "-" if value is None else str(value) if isinstance(value, int) else f"{value:.2f}"
    put(canvas, text, (x + 268, y + 17), 0.6, WHITE, 1, backing=False)
    x0, x1 = x + 325, x + 630
    cv2.rectangle(canvas, (x0, y), (x1, y + 22), (55, 55, 55), -1)
    if value is not None:
        cv2.rectangle(canvas, (x0, y), (x0 + int((x1 - x0) * value), y + 22),
                      RED if value >= threshold else GREEN, -1)
    if not isinstance(value, int):
        tick = x0 + int((x1 - x0) * threshold)
        cv2.line(canvas, (tick, y - 3), (tick, y + 25), WHITE, 2)
    cv2.rectangle(canvas, (x0, y), (x1, y + 22), GREY, 1)


def draw_attention(canvas, y, attention):
    """The model's attention score as one wide bar: green when attentive, red below 0.5."""
    put(canvas, "ATTENTION", (PAD, y + 24), 0.8, WHITE, 2, backing=False)
    x0, x1 = PAD + 325, WIDTH - PAD
    cv2.rectangle(canvas, (x0, y), (x1, y + 32), (55, 55, 55), -1)
    if attention is None:
        put(canvas, "no model yet: record people with collect.py, then run train.py", (x0 + 12, y + 22), 0.55,
            GREY, 1, backing=False)
    else:
        put(canvas, f"{attention:.2f}", (PAD + 190, y + 24), 0.8, WHITE, 2, backing=False)
        cv2.rectangle(canvas, (x0, y), (x0 + int((x1 - x0) * attention), y + 32),
                      GREEN if attention >= 0.5 else RED, -1)
        tick = x0 + (x1 - x0) // 2
        cv2.line(canvas, (tick, y - 3), (tick, y + 35), WHITE, 2)
    cv2.rectangle(canvas, (x0, y), (x1, y + 32), GREY, 1)


def compose(colour, infrared, values, thresholds, footer, footer_colour, attention=None):
    """The whole window: the two views, the attention bar, the signals in two columns, a line of text."""
    rows = (len(values) + 1) // 2
    attention_top = PAD + VIEW_H + 16
    panel_top = attention_top + 32 + 22
    canvas = np.full((panel_top + rows * ROW_H + 44, WIDTH, 3), BACKGROUND, np.uint8)
    canvas[PAD:PAD + VIEW_H, PAD:PAD + COLOUR_W] = colour
    canvas[PAD:PAD + VIEW_H, PAD + COLOUR_W + GAP:PAD + COLOUR_W + GAP + IR_W] = infrared
    draw_attention(canvas, attention_top, attention)
    column_w = (WIDTH - 2 * PAD - GAP) // 2
    for i, (name, value) in enumerate(values.items()):
        draw_signal(canvas, PAD + (i % 2) * (column_w + GAP), panel_top + (i // 2) * ROW_H,
                    name, value, thresholds.get(name, 1))
    put(canvas, footer, (PAD, canvas.shape[0] - 14), 0.55, footer_colour, 1, backing=False)
    return canvas


def signal_thresholds(phone_score=0.35):
    """Where each score counts as "on": the white tick on its bar."""
    return {"face_down_score": 0.5, "gaze_down_score": 0.5, "color_phone": phone_score, "phone": phone_score,
            "phone_in_hand": IN_HAND_ON, "gaze_down_x_phone_in_hand": IN_HAND_ON}


def render(engine, step, attention, footer, footer_colour=GREY, hidden=None, phone_score=0.35, mirror=True):
    """The whole window for one camera step (also used by call/score_sender.py --dashboard).

    `hidden`: signal names to leave out of the list; None leaves out the HIDDEN ones.
    """
    if hidden is None:
        hidden = {name for names in HIDDEN.values() for name in names}
    values = {name: v for name, v in step.values.items() if name not in hidden}
    return compose(
        colour_view(step.frame, engine.face(step.now), engine.worker.phones["rgb"], engine.det, mirror),
        infrared_view(step.ir_frame, engine.worker, mirror),
        values, signal_thresholds(phone_score), footer, footer_colour, attention)


def main(doc=None):
    p = argparse.ArgumentParser(description=(doc or __doc__).splitlines()[0])
    p.add_argument("--device", type=int, help="colour camera index (default: find the LogiInsight camera)")
    p.add_argument("--ir-device", type=int, help="infrared camera index (default: auto-detect)")
    p.add_argument("--face", action="store_true", help="also show face_down and face_down_score")
    p.add_argument("--binary-gaze", action="store_true", help="also show the yes/no gaze_down")
    p.add_argument("--model", default=str(MODEL_PATH), help="the fitted attention model to show (default %(default)s)")
    p.add_argument("--phone-score", type=float, default=0.35,
                   help="confidence at which a phone counts as seen (default 0.35)")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the pictures")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until quit)")
    p.add_argument("--popup", action="store_true", help="pop up a friendly card when the attention score stays low")
    p.add_argument("--popup-below", type=float, default=BELOW,
                   help="attention below this counts as scrolling (default %(default)s)")
    p.add_argument("--popup-after", type=float, default=AFTER_S,
                   help="seconds it has to stay below before the card appears (default %(default)s)")
    p.add_argument("--popup-cooldown", type=float, default=COOLDOWN_S,
                   help="seconds from one card to the next (default %(default)s)")
    p.add_argument("--popup-position", choices=["top-right", "center"], default="top-right",
                   help="where the card appears (default %(default)s)")
    p.add_argument("--no-popup-sound", action="store_true", help="no sound with the card")
    args = p.parse_args()

    try:
        engine = Engine(args.device, args.ir_device, args.phone_score)
    except EngineError as e:
        sys.exit(f"ERROR: {e}")
    print("Look at the screen for a second while it measures your normal pose.")

    det = engine.det
    model = Model.load(args.model)
    history = History(model.windows, model.signals) if model else History()
    if model is None:
        print(f"No attention model at {args.model}: record people with collect.py, then run train.py.")
    else:
        print(f"Attention model {Path(args.model).name}: {', '.join(model.signals)}")
    nudger = None
    if args.popup and model is None:
        print("No pop-ups: they need the attention score, and there is no model.")
    elif args.popup:
        nudger = Nudger(args.popup_below, args.popup_after, args.popup_cooldown,
                        position=args.popup_position, sound=not args.no_popup_sound)
        print(f"Pop-ups on: a card when attention stays below {args.popup_below:g} for {args.popup_after:g} s, "
              f"at most one every {args.popup_cooldown:g} s.")
    hidden = {name for option, names in HIDDEN.items() if not getattr(args, option) for name in names}
    n_frames = 0
    message, message_t = "", 0.0
    try:
        while not args.frames or n_frames < args.frames:
            step = engine.step()
            if step is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            now = step.now
            n_frames += 1
            history.add(now, step.values)
            attention = float(model.predict(history.features(now))) if model else None
            if nudger and nudger.update(now, attention):
                print(f"  pop-up {nudger.shown}: attention {attention:.2f}", flush=True)

            if det.message != message:                # a new note from the detector: show it for a while
                message, message_t = det.message, now
            if now - engine.started > AI_WARN_S:
                footer, colour = (f"camera AI on for {int((now - engine.started) // 60)} min: "
                                  "quit soon, it can overheat"), YELLOW
            elif det.message and now - message_t < MESSAGE_S:
                footer, colour = det.message, YELLOW
            else:
                footer, colour = "c: screen pose   d: down pose   r: reset   q: quit", GREY
                if nudger:
                    footer += f"      pop-ups on ({nudger.shown} shown)"
            cv2.imshow(WINDOW, render(engine, step, attention, footer, colour, hidden, args.phone_score,
                                      not args.no_mirror))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                det.calibrate("screen")
            elif key == ord("d"):
                det.calibrate("down")
            elif key == ord("r"):
                det.reset_thresholds()
    except KeyboardInterrupt:
        pass
    finally:
        if nudger:
            nudger.close()
        engine.close()
        cv2.destroyAllWindows()
    print(f"{n_frames} frames, {engine.feed.n_mesh} face-mesh results"
          + (f", {nudger.shown} pop-ups" if nudger else ""))


if __name__ == "__main__":
    main()
