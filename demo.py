#!/usr/bin/env python3
"""The demo, presenter side, in one command (on the presenter Mac, from the repo folder):

    python3 demo.py

It checks and fixes what it can, tells you the one command to run on the camera PC, and starts the LED eye.
Close the eye when the talk is over: the feedback card comes up a few seconds later, with the attention graph
and a headline of what was said at each moment (by Ollama's qwen2.5:3b; keywords without Ollama). Then press
Enter for another talk, or q to quit. Every talk is saved in its own folder, attention_insights/data/runs/<time>/
(recording, transcript, report, graph, and the output printed here).

What it checks and fixes before each talk:
  - runs itself with the repo's .venv Python, and installs the packages the presenter side needs if missing
  - this Mac's address, and whether it is on the iPhone hotspot (172.20.10.x)
  - an eye left running from before (it holds the port the scores come in on): it is closed
  - the microphone: a short test recording, so a missing macOS permission shows up now and not after the talk
  - Ollama, for the headlines: started if it is installed and not running (and stopped again at the end)

    python3 demo.py --check      only the checks, no talk
    python3 demo.py --moments    the card also lists the moments that got / lost attention (~20 s more with Ollama)
    python3 demo.py --no-llm     keyword topics, don't use Ollama
    python3 demo.py --keep       skip the "another talk?" question: one talk, then quit
    python3 demo.py --redo       the card again for the latest saved talk (or --redo <run folder>)
"""
import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
PORT = 8767
LLM = "qwen2.5:3b"                     # attention_insights/insights.py's default model
OLLAMA = "http://localhost:11434"
RUNS = ROOT / "attention_insights" / "data" / "runs"
NEEDED = {"numpy": "numpy", "matplotlib": "matplotlib", "sounddevice": "sounddevice", "soundfile": "soundfile",
          "faster_whisper": "faster-whisper"}


def say(msg=""):
    print(msg, flush=True)


def ok(msg):
    say(f"  ok    {msg}")


def warn(msg):
    say(f"  !!    {msg}")


def use_venv():
    """Run again with the repo's .venv Python if this is another one."""
    if VENV_PY.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        os.execv(str(VENV_PY), [str(VENV_PY), __file__, *sys.argv[1:]])
    if not VENV_PY.exists():
        warn(f"no .venv in {ROOT}: using {sys.executable}. To make one: python3 -m venv .venv")


def packages():
    import importlib.util
    missing = [pip for mod, pip in NEEDED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        ok("Python packages")
        return True
    say(f"  ..    installing {', '.join(missing)}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing])
    if r.returncode != 0:
        warn(f"could not install them; try: {sys.executable} -m pip install {' '.join(missing)}")
        return False
    ok("Python packages (installed)")
    return True


def address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent: this only picks the network interface
            ip = s.getsockname()[0]
    except OSError:
        ip = None
    if not ip or ip.startswith("127."):
        warn("no network: join the iPhone hotspot (Settings > Personal Hotspot on the phone)")
        return None
    if ip.startswith("172.20.10."):
        ok(f"on the iPhone hotspot, this Mac is {ip}")
    else:
        warn(f"this Mac is {ip}, not on an iPhone hotspot (172.20.10.x). Fine if the PC is on this same network.")
    return ip


def free_port():
    """Close an eye left running from before; it holds the port the scores arrive on."""
    def taken():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            try:
                s.bind(("0.0.0.0", PORT))
                return False
            except OSError:
                return True
    if not taken():
        ok(f"port {PORT} free for the scores")
        return True
    pids = subprocess.run(["lsof", "-ti", f"udp:{PORT}"], capture_output=True, text=True).stdout.split()
    for pid in pids:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if "live_insights.py" in cmd or "live_view.py" in cmd:
            say(f"  ..    closing an eye left running from before (pid {pid})")
            subprocess.run(["kill", pid])
        else:
            warn(f"port {PORT} is used by another program (pid {pid}: {cmd[:80]}); close it and run this again")
            return False
    for _ in range(20):
        if not taken():
            ok(f"port {PORT} free for the scores")
            return True
        time.sleep(0.2)
    warn(f"port {PORT} is still in use; close the old eye window and run this again")
    return False


def microphone():
    """A 0.6 s test recording: macOS gives silence (not an error) when Terminal may not use the microphone."""
    try:
        import numpy as np
        import sounddevice as sd
        rec = sd.rec(int(0.6 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        name = sd.query_devices(kind="input")["name"]
    except Exception as e:  # noqa: BLE001
        warn(f"no microphone ({e}): the eye still works, but the feedback will have no topics")
        return False
    if float(np.abs(rec).max()) == 0.0:  # exact zeros: macOS blocks it (a quiet room still has noise)
        app = terminal_app()
        warn(f"the microphone ({name}) is BLOCKED for {app}: macOS records exact silence, so the talk would")
        warn("have no transcript and the feedback no topics.")
        say(f"        Fix: System Settings > Privacy & Security > Microphone, switch on {app}.")
        say(f"        (No {app} in that list? macOS asks the first time: answer Allow, then run this again.)")
        say(f"        Then QUIT {app} completely (Cmd+Q), open it again and run this again.")
        subprocess.run(["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"])
        return False
    ok(f"microphone: {name}")
    return True


def terminal_app():
    """The app this runs in: it is the one that needs the microphone permission."""
    return {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "Visual Studio Code (or Cursor)",
            "WarpTerminal": "Warp", "ghostty": "Ghostty"}.get(os.environ.get("TERM_PROGRAM", ""),
                                                             "the app you run this in")


def ollama_up():
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=1) as r:
            import json
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except OSError:
        return None


def ollama(started):
    """True if Ollama can write the headlines. Starts `ollama serve` if it is installed but not running; the
    process goes into `started`, so it is stopped again at the end."""
    exe = shutil.which("ollama") or next((p for p in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama",
                                                      "/Applications/Ollama.app/Contents/Resources/ollama")
                                          if Path(p).exists()), None)
    models = ollama_up()
    if models is None and exe:
        say("  ..    starting Ollama")
        started.append(subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for _ in range(50):
            time.sleep(0.2)
            if (models := ollama_up()) is not None:
                break
    if models is None:
        warn("no Ollama: the topics will be keywords, not headlines. To get headlines (about 2 GB, on normal "
             f"Wi-Fi): brew install ollama, then: ollama serve & ollama pull {LLM}")
        return False
    if not any(m == LLM or m.startswith(LLM + "-") for m in models):
        warn(f"Ollama runs, but has no {LLM}: the topics will be keywords. Get it (about 2 GB, on normal Wi-Fi): "
             f"ollama pull {LLM}")
        return False
    ok(f"Ollama with {LLM}: headlines for the topics")
    return True


def checks(args, started):
    say("Checking the presenter Mac:")
    good = packages()
    ip = address()
    good = free_port() and good
    if good and not microphone():
        answer = input("\n  Fix it and run this again (Enter), or c to go on WITHOUT topics: ").strip().lower()
        good = answer.startswith("c")
    args.llm = False if args.no_llm else ollama(started)  # a warning only: keyword topics work without it
    return good and ip is not None, ip


def pc_instructions(ip):
    say()
    say("On the camera PC (PowerShell, in the kgnugen folder), if the sender is not running yet:")
    say(f"    .venv\\Scripts\\python.exe call\\score_sender.py --to {ip}")
    say("  Leave it running for all the talks: it sends all the time, and each talk starts when you open the")
    say("  eye here. Its window prints 'sent N scores' every 5 s; if N stops growing, nothing reaches this Mac.")
    say("  Test without the camera: add --simulate")


def run_eye(cmd, out, args, log_name):
    """live_insights.py with the card options, its output shown here and saved in out/log_name."""
    if not args.llm:
        cmd.append("--no-llm")
    if not args.moments:  # the card: the graph with a headline at each moment; quick with Ollama
        cmd.append("--no-highlights")
    with open(out / log_name, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                if not line.startswith("[20"):  # ctranslate2's tips
                    sys.stdout.write(line)
                    sys.stdout.flush()
        except KeyboardInterrupt:
            proc.terminate()
        proc.wait()
    say()
    say(f"Saved in {out.relative_to(ROOT)}/  ({', '.join(sorted(f.name for f in out.iterdir()))})")


def talk(args):
    out = RUNS / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    say()
    say("Starting the eye. It turns green/red once scores arrive, and says 'no signal' when they stop.")
    say("The talk counts from the first score. CLOSE THE EYE (× or Esc) when the talk is over.")
    say()
    run_eye([sys.executable, "-u", str(ROOT / "attention" / "live_insights.py"), "--udp", str(PORT),
             "--bind", "0.0.0.0", "--record", "--out-dir", str(out)], out, args, "output.txt")


def redo(args, started):
    """The feedback card again for a saved talk (its words.json and scores.csv), e.g. now with Ollama."""
    runs = sorted(d for d in RUNS.glob("*/") if (d / "scores.csv").exists() and (d / "words.json").exists())
    out = Path(args.redo).resolve() if args.redo != "latest" else (runs[-1] if runs else None)
    if out is None or not (out / "scores.csv").exists():
        sys.exit(f"no saved talk with scores.csv{f' in {args.redo}' if args.redo != 'latest' else ''} "
                 "(talks run before this change did not save their scores)")
    say(f"The feedback again for the talk in {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    args.llm = False if args.no_llm else ollama(started)
    run_eye([sys.executable, "-u", str(ROOT / "attention" / "live_insights.py"), "--redo", str(out)], out, args,
            "redo_output.txt")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="only run the checks")
    p.add_argument("--moments", action="store_true",
                   help="the card also lists the moments that got / lost attention (~20 s more with Ollama)")
    p.add_argument("--no-llm", action="store_true", help="keyword topics, don't use Ollama")
    p.add_argument("--keep", action="store_true", help="one talk, then quit")
    p.add_argument("--redo", nargs="?", const="latest", metavar="RUN_DIR",
                   help="the feedback card again for a saved talk (default: the latest), e.g. now with Ollama")
    args = p.parse_args()
    use_venv()
    os.chdir(ROOT)
    started = []  # an Ollama this started: stopped again at the end
    try:
        if args.redo:
            return redo(args, started)
        while True:
            good, ip = checks(args, started)
            if not good:
                say("\nFix the !! lines above, then run this again.")
                sys.exit(1)
            pc_instructions(ip)
            if args.check:
                return
            say()
            input("Press Enter to start the eye ... ")
            talk(args)
            if args.keep:
                return
            if input("\nAnother talk? Enter = yes, q = quit: ").strip().lower().startswith("q"):
                return
            say()
    finally:
        for proc in started:
            proc.terminate()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
