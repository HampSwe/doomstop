#!/usr/bin/env python3
"""
live_insights.py - presenter feedback: a small attention meter during the talk, the full picture right after.

Joins the live attention stream (attention_profile.py) with attention_insights (transcript + LLM summary):
  - while attention samples and transcript words come in, only a small meter is shown (presenter_ui.py): an
    LED eye that lights up and turns from red to green with the audience's attention, so the presenter isn't
    distracted;
  - when the stream ends, a feedback card pops up with the attention over the whole talk; hovering over the
    graph shows the topic at that moment (named by a local LLM via Ollama, else keywords).
    attention_insights also writes report.md, timeline.png and insights.json into --out-dir.

    python attention/simulate.py --demo-talk | python attention/live_insights.py    # test: recorded 30 s talk
    my_feed | python attention/live_insights.py                    # attention + words from your own pipeline
    my_feed | python attention/live_insights.py --words words.json # transcript known in advance (e.g. a script)
    python attention/live_insights.py --udp 8767                   # receive UDP datagrams instead of stdin
    ... --no-window                                                # terminal only: print moments as found

    python attention/live_insights.py --udp 8767 --bind 0.0.0.0 --record
        # the demo: attention from call/score_sender.py on the camera computer, the talk recorded from this
        # computer's microphone and transcribed when it ends

Closing the meter early (its × box) ends the talk there and shows the feedback on what came in. With UDP
input the stream never ends by itself, so that is how the talk ends.

--record: this computer's microphone is recorded from the start (to --out-dir/talk.wav). When the talk ends
it is transcribed with faster-whisper (--whisper, loaded at the start so it is ready), and the words feed the
topics. Attention samples are then timed when they arrive here, on this computer's clock, so they line up
with the recording whatever the sender's clock says (the network adds only some 25 ms).

Input lines (stdin or UDP, one per line / datagram):
    <time>,<attention>[,<topic>]     an attention sample (seconds or mm:ss, any scale), e.g. 12.5,0.73
    word,<start>,<end>,<word>        a transcript word once it has been recognised, e.g. word,12.1,12.4,surprise
"""
import argparse
import json
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from attention_profile import AttentionProfile, hms, parse_float, parse_time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attention_insights"))
import insights  # noqa: E402  attention_insights/insights.py

OUT_DIR = Path(__file__).resolve().parent.parent / "attention_insights" / "data" / "live"


class Feed:
    """Thread-safe collector: reader threads add lines, the main thread reads them."""

    def __init__(self):
        self.prof, self.words, self.lock = AttentionProfile(), [], threading.Lock()
        self.version, self.done, self.t0 = 0, False, None
        self.arrival_start = None  # time.time() of t = 0 when samples are timed on arrival (--record)
        self.last_rx = None        # time.monotonic() of the last attention sample

    def add_line(self, line):
        line = line.strip()
        if line.lower().startswith("word,"):
            parts = line.split(",", 3)
            if len(parts) == 4:
                a, b, word = parse_time(parts[1]), parse_time(parts[2]), parts[3].strip()
                if isinstance(a, float) and isinstance(b, float) and word:
                    with self.lock:
                        self.words.append({"start": a, "end": b, "word": word})
                        self.version += 1
            return
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            return
        t, v = parse_time(parts[0]), parse_float(parts[1])
        if t is None or v is None:  # header line or junk
            return
        if self.arrival_start is not None:  # this computer's clock, the same one as the recording
            t = time.time() - self.arrival_start
        elif isinstance(t, datetime):
            self.t0 = self.t0 or t
            t = (t - self.t0).total_seconds()
        with self.lock:
            self.prof.add(t, v, parts[2] if len(parts) > 2 and parts[2] else None)
            self.version += 1
            self.last_rx = time.monotonic()


def read_stdin(feed):
    try:
        for line in sys.stdin:
            feed.add_line(line)
    except (OSError, ValueError):
        pass
    feed.done = True


def read_udp(feed, port, bind="127.0.0.1"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind, port))  # 127.0.0.1: this computer only; 0.0.0.0: also from other computers
    while True:
        data, _ = sock.recvfrom(65536)
        for line in data.decode("utf-8", errors="replace").splitlines():
            feed.add_line(line)


RATE = 16000     # Whisper works at 16 kHz
LIVE_STEP_S = 8  # during the talk, transcribe each time this much new audio has come in
TAIL_S = 2.0     # ... but leave the last bit for the next pass: a word may be cut off there


class Talk:
    """--record: this computer's microphone from the start of the talk, transcribed WHILE it is recorded.
    Every LIVE_STEP_S seconds a background thread transcribes the new audio and keeps the sentences that have
    ended, so when the talk ends only the last few seconds are left to do (base.en: ~1 s instead of ~8 s per
    minute of talk). The Whisper model loads in the background at the start."""

    def __init__(self, wav, whisper):
        import sounddevice as sd
        import soundfile as sf

        self.wav, self.whisper_name = Path(wav), whisper
        self.wav.parent.mkdir(parents=True, exist_ok=True)
        self.model, self.model_error = None, None
        threading.Thread(target=self._load, daemon=True).start()
        self.blocks, self.frames, self.peak = queue.Queue(), 0, 0.0
        self.audio, self.audio_lock = [], threading.Lock()   # every block so far, for the live transcription
        self.segments, self.done_upto = [], 0                # transcribed segments; samples transcribed so far
        self.stopping, self.stopped = threading.Event(), threading.Event()  # the talk is over; the mic has stopped
        self.file = sf.SoundFile(self.wav, mode="w", samplerate=RATE, channels=1, subtype="PCM_16")

        def record(data, frames, t, status):
            if self.stopping.is_set():  # the stream stops itself: stream.stop() (and abort() / close()) from
                raise sd.CallbackStop   # another thread hang on macOS while Tk 9 runs on the main thread
            self.blocks.put(data[:, 0].copy())

        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", callback=record,
                                     finished_callback=self.stopped.set)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()
        self.live = threading.Thread(target=self._transcribe_live, daemon=True)
        self.live.start()
        self.stream.start()
        self.started = time.time()  # t = 0 of the recording, and of the attention samples
        print(f"recording the talk from '{sd.query_devices(kind='input')['name']}' into {self.wav}", file=sys.stderr)

    def _load(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.whisper_name, device="cpu", compute_type="int8")
            print(f"speech model {self.whisper_name} ready", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - the eye must keep working without it
            self.model_error = e
            print(f"WARNING: speech model {self.whisper_name} did not load ({e}); "
                  "the feedback will have no topics", file=sys.stderr)

    def _write(self):
        while (block := self.blocks.get()) is not None:
            self.file.write(block)
            with self.audio_lock:
                self.audio.append(block)
                self.frames += len(block)
            self.peak = max(self.peak, float(np.abs(block).max()))

    def _pass(self, final):
        """Transcribe the audio not transcribed yet. Unless final, keep only the segments that end more than
        TAIL_S before the end of the audio; the rest is done again next time, with what follows it."""
        with self.audio_lock:
            if self.audio:
                self.audio = [np.concatenate(self.audio)]
            audio = self.audio[0][self.done_upto:] if self.audio else np.zeros(0, np.float32)
        offset, length = self.done_upto / RATE, len(audio) / RATE
        if length < 0.5:
            return
        segments, _ = self.model.transcribe(audio, word_timestamps=True, vad_filter=True,
                                            condition_on_previous_text=False)
        keep_until = length if final else length - TAIL_S
        kept_end, heard = None, False
        for seg in segments:
            heard = True
            if seg.end > keep_until:
                if kept_end is not None:  # resume in the pause before this sentence: word end stamps run early,
                    kept_end = min(max(kept_end, (kept_end + seg.start) / 2), kept_end + 0.5)  # a cut at one
                break                                                     # would repeat the word's tail next time
            ws = [{"start": round(offset + w.start, 2), "end": round(offset + w.end, 2), "word": w.word.strip()}
                  for w in (seg.words or []) if w.word.strip()]
            self.segments.append({"start": round(offset + seg.start, 2), "end": round(offset + seg.end, 2),
                                  "text": seg.text.strip(), "words": ws})
            kept_end = seg.end
        if final:
            kept_end = length
        elif kept_end is None and length > 4 * LIVE_STEP_S:  # one long run-on sentence: take it as it is
            return self._pass(final=True)
        elif kept_end is None:  # no speech: skip it (not the tail); a sentence still going: wait for its end
            kept_end = 0.0 if heard else max(0.0, length - TAIL_S)
        self.done_upto += int(kept_end * RATE)

    def _transcribe_live(self):
        while not self.stopping.is_set():
            if self.model is not None and self.frames - self.done_upto >= LIVE_STEP_S * RATE:
                try:
                    self._pass(final=False)
                except Exception as e:  # noqa: BLE001 - whatever is left gets transcribed at the end
                    print(f"WARNING: live transcription failed ({e}); transcribing at the end", file=sys.stderr)
                    return
            self.stopping.wait(0.5)

    def words(self, out_dir):
        """Stop recording; return the transcript words [{start, end, word}] in seconds from the start, and
        save them as out_dir/words.json (the format of attention_insights/transcribe.py)."""
        self.stopping.set()      # the audio callback stops the stream, and the live transcription stops
        self.stopped.wait(1.0)
        self.blocks.put(None)
        self.writer.join()
        self.file.close()
        print(f"recorded {self.frames / RATE:.0f} s of talk", file=sys.stderr)
        if self.peak < 1e-3:
            print("WARNING: the microphone recorded only silence. On a Mac: System Settings > Privacy & "
                  "Security > Microphone, allow Terminal.", file=sys.stderr)
            return []
        while self.model is None and self.model_error is None:  # still loading
            time.sleep(0.2)
        if self.model is None:
            return []
        self.live.join()  # let a pass that is running finish
        t, left = time.time(), (self.frames - self.done_upto) / RATE
        self._pass(final=True)
        words = [w for seg in self.segments for w in seg["words"]]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "words.json").write_text(json.dumps({"duration": round(self.frames / RATE, 2),
                                                              "segments": self.segments}, indent=1))
        print(f"transcribed: {len(words)} words; the last {left:.0f} s took {time.time() - t:.1f} s",
              file=sys.stderr)
        return words


def scaled(duration):
    """Settings that fit the talk so far: a 30 s talk and an hour-long one need different windows."""
    return {"smooth": float(np.clip(duration / 12, 2, 30)), "window": float(np.clip(duration / 6, 4, 120)),
            "min_gap": float(np.clip(duration / 10, 2, 60))}


def said_around(words, t, half=4.0, max_words=18):
    """The words spoken within `half` seconds of time t, as a short quote ('' if nothing was said)."""
    near = [w for w in words if w["end"] > t - half and w["start"] < t + half]
    sel = [w["word"] for w in near]
    if len(sel) > max_words:  # keep the words closest to t
        i = int(np.argmin([abs((w["start"] + w["end"]) / 2 - t) for w in near]))
        lo = max(0, min(i - max_words // 2, len(sel) - max_words))
        sel = (["…"] if lo > 0 else []) + sel[lo:lo + max_words] + (["…"] if lo + max_words < len(sel) else [])
    return " ".join(sel)


def snapshot(feed):
    with feed.lock:
        prof = AttentionProfile()
        prof.t, prof.s, prof.markers = list(feed.prof.t), list(feed.prof.s), list(feed.prof.markers)
        words = sorted(feed.words, key=lambda w: w["start"])
    return prof, words


def talk_on_clock(prof, words, ended, lag):
    """With --record the recording starts with the eye, often before the camera computer sends anything.
    Keep what lies between the first score and `ended` (closing the eye), and count time from that first
    score, so the feedback starts at 0:00 and the setup talk before it is not part of it. In place."""
    t0 = prof.t[0]
    keep = [i for i, t in enumerate(prof.t) if t <= ended]
    prof.t = [prof.t[i] - t0 for i in keep]
    prof.s = [prof.s[i] for i in keep]
    prof.markers = [(m - t0, name) for m, name in prof.markers if t0 <= m <= ended]
    words[:] = [dict(w, start=round(w["start"] - t0, 2), end=round(w["end"] - t0, 2)) for w in words
                if t0 <= w["start"] <= ended]
    if t0 > 1:
        print(f"the talk: from the first attention score, {t0:.0f} s into the recording", file=sys.stderr)


def analyse(prof, words, lag):
    """Attention profile of the data so far, with what was said at each ▲ / ▼ moment.
    Returns (profile result, (t, attention) on an even grid without gaps)."""
    t = np.asarray(prof.t)
    duration = float(t.max() - t.min())
    res = prof.analyse(**scaled(duration))
    half = float(np.clip(duration / 14, 1.5, 6))  # quote window: ~3 s around the moment in a 40 s talk
    for ev in res["peaks"] + res["dips"]:
        ev["said"] = said_around(words, ev["time"] - lag, half)
    grid = np.asarray(res["profile"]["t"])
    y = np.asarray(res["profile"]["score"], float)
    ok = ~np.isnan(y)
    return res, (grid[ok], y[ok])


def finish(feed, lag, out_dir, model, talk=None, explain=True):
    """End of the talk: print the ▲ / ▼ moments, let attention_insights write its report, and return what the
    feedback card shows (None if there is too little attention data). With --record, transcribe first.
    explain=False: the LLM only names the topics (~2 s), it doesn't label and explain the high / low moments
    (~20 s more), which only the card's list of moments shows."""
    ended = time.time() - talk.started if talk is not None else None
    if talk is not None:
        words = talk.words(out_dir)
        with feed.lock:
            feed.words.extend(words)
    prof, words = snapshot(feed)
    if len(prof) < 5:
        print("not enough attention data for feedback", file=sys.stderr)
        return None
    if talk is not None:  # the talk is from the first score to closing the eye: drop the setup before it
        talk_on_clock(prof, words, ended, lag)
    res, (t, y) = analyse(prof, words, lag)
    print(f"\n=== Live insights ({hms(t[-1] - t[0])} of attention, {len(words)} words) ===")
    for key, title in (("peaks", "Highest attention"), ("dips", "Lowest attention")):
        print(f"\n{title}")
        for k, ev in enumerate(res[key], 1):
            print(f"  {k}. {hms(ev['time'])}  ({ev['relative']:+.1f} spread)  said: “{ev['said'] or '-'}”")
        if not res[key]:
            print("  none stand out")
    result = {"res": res, "episodes": [], "chunks": [], "tips": [], "report": None, "n_words": len(words),
              "lag": lag}
    if len(words) < 6:
        print("\nToo few transcript words for subjects (send 'word,<start>,<end>,<word>' lines or use --words).")
        return result
    if y.min() < 0 or y.max() > 1:  # attention_insights works on 0..1: rescale other score scales
        y = (y - y.min()) / ((y.max() - y.min()) or 1.0)
        print("(attention rescaled to 0..1 for the insights report)")
    print()
    out_dir = Path(out_dir)
    lines = insights.write_report(out_dir, words, t, y, lag, model=model, explain=explain)
    print("\n".join(lines[:lines.index("## All chunks")]))
    print(f"\nWrote {out_dir / 'report.md'}, {out_dir / 'timeline.png'}, {out_dir / 'insights.json'}")
    data = json.loads((out_dir / "insights.json").read_text())
    llm = any(e.get("why") for e in data["episodes"])  # the keyword fallback writes no explanations or tips
    result.update(episodes=data["episodes"], chunks=data["chunks"], tips=data["insights"] if llm else [],
                  report=out_dir / "report.md")
    return result


def run_terminal(feed, lag):
    """Print every new ▲ / ▼ moment with its quote once it has settled, until the input ends."""
    seen, version = set(), -1
    while True:
        done = feed.done
        if feed.version != version:
            version = feed.version
            prof, words = snapshot(feed)
            duration = max(prof.t) - min(prof.t) if len(prof) else 0.0
            if len(prof) >= 5 and duration >= 8:
                res, _ = analyse(prof, words, lag)
                settled = max(prof.t) - max(2.0, scaled(duration)["min_gap"])  # newer moments may still move
                for key, mark in (("peaks", "▲ high"), ("dips", "▼ low")):
                    for ev in [ev for ev in res[key] if ev["time"] <= settled]:
                        k = (key, round(ev["time"]))
                        if k not in seen and not any((key, k[1] + d) in seen for d in (-2, -1, 1, 2)):
                            seen.add(k)
                            print(f"[{hms(prof.t[-1])}] {mark} attention at {hms(ev['time'])}, said: "
                                  f"“{ev['said'] or '-'}”", flush=True)
        if done:
            return
        time.sleep(0.5)


def main(argv=None):
    p = argparse.ArgumentParser(description="Presenter feedback: live attention meter, feedback card at the end.")
    p.add_argument("--udp", type=int, metavar="PORT", help="receive UDP datagrams on 127.0.0.1:PORT instead of stdin")
    p.add_argument("--bind", default="127.0.0.1",
                   help="with --udp: the address to listen on; 0.0.0.0 also takes scores from other computers")
    p.add_argument("--record", action="store_true",
                   help="record this computer's microphone and transcribe the talk when it ends")
    p.add_argument("--whisper", default="base.en",
                   help="faster-whisper model for --record (default base.en; small.en is better but slower)")
    p.add_argument("--no-highlights", action="store_true",
                   help="feedback card without the highest / lowest attention moments under the graph")
    p.add_argument("--words", help="transcript known in advance (words.json from attention_insights/transcribe.py)")
    p.add_argument("--lag", type=float, default=2.0, help="audience reaction lag in seconds (default 2)")
    p.add_argument("--out-dir", default=str(OUT_DIR), help="where the attention_insights report goes")
    p.add_argument("--model", default=insights.MODEL, help="Ollama model for the subject labels and tips")
    p.add_argument("--no-llm", action="store_true", help="keyword labels only, don't ask a local LLM")
    p.add_argument("--no-window", action="store_true", help="terminal only: print moments as they are found")
    p.add_argument("--no-sound", action="store_true", help="no chime when the feedback card appears")
    a = p.parse_args(argv)
    model = None if a.no_llm else a.model
    if model:  # load the LLM now, like the speech model, so the topics take seconds when the talk ends
        threading.Thread(target=insights.load_llm, args=(model,), daemon=True).start()
    explain = not a.no_highlights  # the LLM's labels of the high / low moments only show in their list

    feed = Feed()
    if a.words:
        data = json.loads(Path(a.words).read_text())
        feed.words = [dict(w) for s in data["segments"] for w in s["words"]]
    talk = None
    if a.record:  # before the readers start, so every sample is timed on this computer's clock
        try:
            talk = Talk(Path(a.out_dir) / "talk.wav", a.whisper)
            feed.arrival_start = talk.started
        except Exception as e:  # noqa: BLE001 - no microphone: the eye still works, the card has no topics
            print(f"WARNING: cannot record the microphone ({e}); no transcript, so no topics", file=sys.stderr)
    if a.udp:
        threading.Thread(target=read_udp, args=(feed, a.udp, a.bind), daemon=True).start()
        source = f"udp://{a.bind}:{a.udp}"
        if a.bind == "0.0.0.0":
            source += f" (this computer's address on the network: {own_address()})"
    else:
        threading.Thread(target=read_stdin, args=(feed,), daemon=True).start()
        source = "stdin"
    print(f"live insights: reading attention + words from {source}", file=sys.stderr)
    if a.udp:
        print("close the eye (its × box, or Esc) to end the talk and see the feedback", file=sys.stderr)

    if a.no_window:
        try:
            run_terminal(feed, a.lag)
        except KeyboardInterrupt:  # UDP never ends by itself: Ctrl+C ends the talk
            pass
        finish(feed, a.lag, a.out_dir, model, talk, explain)
    else:
        import presenter_ui
        presenter_ui.run(feed, lambda: finish(feed, a.lag, a.out_dir, model, talk, explain), sound=not a.no_sound,
                         highlights=not a.no_highlights)


def own_address():
    """This computer's address on the network that has the default route (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "unknown (macOS: ipconfig getifaddr en0)"


if __name__ == "__main__":
    main()
