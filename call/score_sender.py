"""Send the live attention score from the camera machine to the presenter's live plot.

Camera machine (the audience, with the LogiInsight):
    python call/score_sender.py --to 172.20.10.3                         # Windows / Linux: colour + infrared
    sudo .venv/bin/python call/score_sender.py --to 172.20.10.3 --mac    # Mac: colour only (see mac/README.md)
    python call/score_sender.py --to 172.20.10.3 --simulate              # no camera: a made-up score, for tests
    python call/score_sender.py --to 172.20.10.3 --dashboard --popup     # also show the dashboard here, and
                                                                         # pop up the card when you doomscroll

Presenter Mac (start it whenever, before or after the sender; see DEMO.md):
    .venv/bin/python attention/live_insights.py --udp 8767 --bind 0.0.0.0 --record   # the LED eye + feedback
    .venv/bin/python attention/live_view.py --udp 8767 --bind 0.0.0.0 --short       # or: a plain live graph

Sending starts by itself as soon as the camera is open and the normal pose is
measured. With --wait it prints

    DONE. PRESS ENTER TO START

instead, and nothing is sent or logged until Enter is pressed. The presenter's
feedback counts the talk from the first score that arrives, so with --wait,
Enter is the start of the talk.

The score is computed exactly as dashboard.py computes it: the engine's
signals go into attention.History, and the model (models/attention_model.json,
or --model) turns their averages into attention (0 = distracted, 1 =
attentive). Each model brings its own list of signals and windows.

Every score goes to the presenter as one UDP datagram, "ISO time,score", which
is the line format attention/live_view.py reads, at most --rate a second. UDP
has no connection, so nothing has to reconnect: if the network or the plot
goes away, the sender keeps going, and the plot carries on as soon as datagrams
arrive again. If the camera stops, the sender reopens it every few seconds.
Every score is also written to call/scores/scores_<time>.csv here, which
attention/attention_profile.py can read afterwards.

--dashboard shows dashboard.py's window on this computer while it sends: both
streams, the attention score and the signals, with the sending status in the
bottom line. --popup adds focus.py's card ("DOOMSCROLL DETECTED", with its
voice) whenever the attention stays low. One program does all of it, because
only one program can hold the camera.

Keys in the --dashboard window: c measures the normal pose again, q or Esc quits.
Keys in the --preview window: q or Esc quits.
"""

import argparse
import csv
import random
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "logiinsight_api")]

PORT = 8767
REOPEN_S = 3.0


def camera_scores(args, on_step=None):
    """(time, attention, colour frame) for every camera frame, as dashboard.py computes it.
    Reopens the camera whenever it stops, instead of ending.

    `on_step(engine, step, attention, ready)` is called for every camera step, also while the
    normal pose is still being measured (ready False): the --dashboard window is drawn there."""
    import engine
    from attention import History, Model

    if args.mac:
        from mac.engine_mac import MacEngine as Engine
    else:
        Engine = engine.Engine
    model = Model.load(args.model) if args.model else Model.load()
    if model is None:
        sys.exit(f"No attention model at {args.model or 'models/attention_model.json'}: "
                 "record with collect.py, then run train.py.")
    print(f"model: {args.model or 'models/attention_model.json'}, signals {', '.join(model.signals)}", flush=True)
    while True:
        try:
            eng = Engine(args.device, args.ir_device, args.phone_score)
        except engine.EngineError as e:
            print(f"camera did not open ({e}); trying again in {REOPEN_S:.0f} s", flush=True)
            time.sleep(REOPEN_S)
            continue
        print("Camera open. Look at the screen for a second while it measures your normal pose.", flush=True)
        history = History(model.windows, model.signals)   # the signals this model was fitted on
        try:
            while True:
                step = eng.step()
                if step is None:
                    yield None
                    continue
                history.add(step.now, step.values)
                attention = float(model.predict(history.features(step.now)))
                det = getattr(eng, "det", None)
                ready = det is None or not (det.calibrating or det.base is None)
                if on_step:
                    on_step(eng, step, attention, ready)
                if not ready:
                    yield None                            # still measuring the normal pose: not ready yet
                    continue
                yield step.now, attention, step.frame
        except engine.EngineError as e:
            print(f"camera stopped ({e}); reopening in {REOPEN_S:.0f} s", flush=True)
        finally:
            eng.close()
        time.sleep(REOPEN_S)


def simulated_scores(args):
    """A made-up score shaped like the demo: attentive, with a phone break every half minute or so.
    For testing the link and the plot without a camera; --speed runs it faster than real time."""
    rng = random.Random(args.seed)
    start, sim_t, level, dt = time.time(), 0.0, 0.9, 1 / args.rate
    phases = []
    while True:
        if not phases:
            phases = [(rng.uniform(18, 32), 0.88), (rng.uniform(7, 14), 0.12)]
        length, target = phases[0]
        tau = 4.0 if target < level else 1.5            # falls over ~4 s of distraction, recovers in ~1.5 s
        level += (target - level) * min(1.0, dt / tau)
        score = min(1.0, max(0.0, level + rng.gauss(0, 0.03)))
        phases[0] = (length - dt, target)
        if phases[0][0] <= 0:
            phases.pop(0)
        sim_t += dt
        time.sleep(dt / args.speed)
        yield start + sim_t, score, None


def resolve(host):
    while True:
        try:
            return socket.gethostbyname(host)
        except OSError as e:
            print(f"cannot find {host} ({e}); trying again in 2 s", flush=True)
            time.sleep(2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--to", required=True, help="the presenter Mac's address, e.g. 172.20.10.3")
    p.add_argument("--port", type=int, default=PORT, help=f"live_view.py's --udp port (default {PORT})")
    p.add_argument("--rate", type=float, default=10, help="scores sent per second, at most (default 10)")
    p.add_argument("--model", help="another fitted model, e.g. models/attention_model_face_down.json "
                                   "(default: models/attention_model.json, Hampus's latest)")
    p.add_argument("--mac", action="store_true", help="camera machine is a Mac: colour only, run with sudo")
    p.add_argument("--simulate", action="store_true", help="no camera: send a made-up score")
    p.add_argument("--speed", type=float, default=1, help="with --simulate: times faster than real time")
    p.add_argument("--seed", type=int, help="with --simulate: the random seed")
    p.add_argument("--device", type=int, help="colour camera index (default: find the LogiInsight camera)")
    p.add_argument("--ir-device", type=int, help="infrared camera index (default: auto-detect)")
    p.add_argument("--phone-score", type=float, default=0.35, help="phone confidence threshold (default 0.35)")
    p.add_argument("--preview", action="store_true", help="show a small window with the score being sent")
    p.add_argument("--wait", action="store_true",
                   help="when everything is ready, wait for Enter before sending (default: start at once)")
    p.add_argument("--no-wait", action="store_true", help=argparse.SUPPRESS)   # the default now; still accepted
    p.add_argument("--dashboard", action="store_true",
                   help="show dashboard.py's window here too: both streams, the attention score, the signals")
    p.add_argument("--popup", action="store_true",
                   help="pop up focus.py's DOOMSCROLL DETECTED card, with its voice, when the attention stays low")
    p.add_argument("--no-popup-sound", action="store_true", help="with --popup: no voice")
    args = p.parse_args()

    import cv2
    import numpy as np

    dest = (resolve(args.to), args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    log_dir = ROOT / "call" / "scores"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"scores_{datetime.now():%Y%m%d_%H%M%S}.csv"
    quit_asked = []                            # filled when q or Esc is pressed in the dashboard window
    nudger = None
    if args.popup:
        from nudge import Nudger
        nudger = Nudger(sound=not args.no_popup_sound)
        print(f"pop-ups on: a card when attention stays below {nudger.below:g} for {nudger.after_s:g} s, "
              f"at most one every {nudger.cooldown_s:g} s", flush=True)

    def show_dashboard(eng, step, attention, ready):
        """dashboard.py's window for this step, with what the sender is doing in the bottom line."""
        if not ready:
            footer, colour = "measuring your normal pose: look at the screen", dashboard.YELLOW
        elif not go.is_set():
            footer, colour = "READY: press ENTER in the terminal to start sending", dashboard.YELLOW
        else:
            footer, colour = f"sending to {dest[0]}: {sent} scores sent", dashboard.GREEN
        footer += "      c: screen pose   q: quit" + (f"      pop-ups on ({nudger.shown} shown)" if nudger else "")
        cv2.imshow("score_sender: dashboard", dashboard.render(eng, step, attention, footer, colour,
                                                               phone_score=args.phone_score))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            quit_asked.append(True)
        elif key == ord("c"):
            eng.det.calibrate("screen")

    on_step = None
    if args.dashboard and not args.simulate:
        import dashboard
        on_step = show_dashboard
    source = simulated_scores(args) if args.simulate else camera_scores(args, on_step)
    kind = "SIMULATED score" if args.simulate else "attention score"
    print(f"the {kind} goes to {dest[0]}:{dest[1]}, {args.rate:g} a second; also logged to {log_path}",
          flush=True)

    interval, last_sent, last_status, sent, failing, score = 1 / args.rate, 0.0, time.time(), 0, False, None
    go, asked = threading.Event(), False       # with --wait, nothing is sent until Enter is pressed

    def wait_for_enter():
        sys.stdin.readline()                   # returns on Enter, and at once when there is no keyboard
        print("started", flush=True)
        go.set()
    with open(log_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["time", "attention"])
        try:
            for item in source:
                if item is not None and not asked:            # the first score: everything is initialised
                    asked = True
                    if args.wait:
                        print("\nDONE. PRESS ENTER TO START", flush=True)
                        threading.Thread(target=wait_for_enter, daemon=True).start()
                    else:
                        print("ready: sending", flush=True)
                        go.set()
                if item is not None:
                    now, score, frame = item
                    if nudger and nudger.update(now, score):
                        print(f"  pop-up {nudger.shown}: attention {score:.2f}", flush=True)
                    if go.is_set() and now - last_sent >= 0.9 * interval:   # 0.9: a score due exactly on time still goes
                        last_sent = now
                        stamp = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
                        try:
                            sock.sendto(f"{stamp},{score:.4f}".encode(), dest)
                            sent += 1
                            if failing:
                                print("sending again", flush=True)
                                failing = False
                        except OSError as e:                  # network gone: keep scoring, keep trying
                            if not failing:
                                print(f"cannot send to {dest[0]} ({e}); still trying", flush=True)
                                failing = True
                        log.writerow([stamp, f"{score:.4f}"])
                        f.flush()
                    if args.preview and not on_step:
                        view = frame.copy() if frame is not None else np.full((120, 420, 3), 30, np.uint8)
                        h, w = view.shape[:2]
                        cv2.rectangle(view, (0, h - 36), (int(w * score), h), (80, 200, 80) if score >= 0.5
                                      else (60, 60, 230), -1)
                        cv2.putText(view, f"{'sending' if go.is_set() else 'ready:'} {score:.2f}", (10, h - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (255, 255, 255), 2)
                        cv2.imshow("score_sender (q: quit)", view)
                if quit_asked or (args.preview and not on_step and cv2.waitKey(1) & 0xFF in (ord("q"), 27)):
                    break
                if time.time() - last_status >= 5 and (go.is_set() or not asked):
                    last_status = time.time()
                    print(f"  sent {sent} scores, now {score:.2f}" if score is not None else "  waiting for the camera",
                          flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            if nudger:
                nudger.close()
            cv2.destroyAllWindows()
    print(f"sent {sent} scores; log: {log_path}")


if __name__ == "__main__":
    main()
