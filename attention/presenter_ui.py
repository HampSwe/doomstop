"""
presenter_ui.py - what the presenter sees: a small attention meter during the talk, a feedback card after it.

Same warm retro look as message/popup.py (cream paper, ink outlines, hard shadows, Futura + American Typewriter),
drawn with tkinter only.

  Meter      a small always-on-top card in the bottom-right corner (drag it anywhere): a column of blocks that
             goes up and down with the live attention, the current level and whether it is rising or falling.
  Feedback   when the stream has ended: the attention profile of the whole talk, the most and least engaging
             subjects with when they came up, what was said and the related topics, and tips when a local LLM
             wrote them. "OPEN REPORT" opens the full attention_insights report.

live_insights.py runs this; run(feed, finish) is the entry point.
"""
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from attention_profile import hms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "message"))
from popup import (EMPTY, INK, MUSTARD, ORANGE, ORANGE_HOVER, PAPER, RUST, TEAL, TEXT2,  # noqa: E402
                   ease_out_back, round_rect)

PANEL, FAINT = "#efe3c9", "#cbbd9d"   # chart background, raw-sample line / grid
DISPLAY, TYPEWRITER = ("Futura", "Arial Rounded MT Bold"), ("American Typewriter", "Courier New")
SOUND = "/System/Library/Sounds/Hero.aiff"
BLOCKS = 12                           # meter blocks, bottom to top: rust, orange, mustard, teal
_fonts = {}


def mix(a, b, t):
    """Blend two #rrggbb colours: t = 0 -> a, t = 1 -> b."""
    ca, cb = ([int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def font(kind, size, weight="normal"):
    from tkinter import font as tkfont
    key = (kind, size, weight)
    if key not in _fonts:
        families = set(tkfont.families())
        family = next((f for f in (DISPLAY if kind == "display" else TYPEWRITER) if f in families), None)
        if family:
            _fonts[key] = tkfont.Font(family=family, size=size, weight=weight)
        else:
            f = tkfont.nametofont("TkDefaultFont").copy()
            f.configure(size=size, weight=weight)
            _fonts[key] = f
    return _fonts[key]


def chime(sound):
    if sound and sys.platform == "darwin" and os.path.exists(SOUND):
        subprocess.Popen(["afplay", SOUND], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class RetroWindow:
    """A borderless card: hard shadow, rounded corners, ink title bar (label, optional stripes, close box)."""

    def __init__(self, win, width, label, header, shadow, on_close, stripes=True):
        import tkinter as tk
        self.win, self.W, self.S, self.header = win, width, shadow, header
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.0)
        if sys.platform == "darwin":  # transparent window -> real rounded corners and a free-standing shadow
            win.attributes("-transparent", True)
            clear = "systemTransparent"
        elif sys.platform == "win32":
            clear = "#010203"
            win.attributes("-transparentcolor", clear)
        else:
            clear = PAPER
        win.configure(bg=clear)
        c = self.c = tk.Canvas(win, width=width + shadow + 2, height=100, bg=clear, highlightthickness=0, bd=0)
        c.pack()
        e = 1.25
        self.shadow = c.create_polygon(0, 0, 1, 0, 1, 1, fill=INK, outline=INK, width=2.5)
        self.card = c.create_polygon(0, 0, 1, 0, 1, 1, fill=PAPER, outline=INK, width=2.5)
        c.create_polygon(round_rect(e, e, width - e, header, 12, bottom=False), fill=INK, outline=INK, width=2.5,
                         tags=("drag",))
        mid = (header + e) / 2
        self.label = c.create_text(12 if width < 200 else 24, mid, anchor="w", text=label, fill=PAPER,
                                   font=font("typewriter", 10 if width < 200 else 11, "bold"), tags=("drag",))
        bx1, by1, bx2, by2 = width - (24 if width < 200 else 32), mid - 7, width - (10 if width < 200 else 16), mid + 7
        box = c.create_rectangle(bx1, by1, bx2, by2, fill=INK, outline=PAPER, width=1.5, tags=("close",))
        x = c.create_text((bx1 + bx2) / 2, (by1 + by2) / 2, text="×", fill=PAPER, font=font("display", 11, "bold"),
                          tags=("close",))
        if stripes:
            sx0, sx1 = c.bbox(self.label)[2] + 14, bx1 - 12
            for i, color in enumerate((RUST, ORANGE, MUSTARD, TEAL)):
                c.create_rectangle(sx0, mid - 11 + 6 * i, sx1, mid - 7 + 6 * i, fill=color, outline="", tags=("drag",))
        hand = "pointinghand" if sys.platform == "darwin" else "hand2"
        c.tag_bind("close", "<Enter>", lambda _e: (c.itemconfigure(box, fill=ORANGE), c.itemconfigure(x, fill=INK),
                                                   c.configure(cursor=hand)))
        c.tag_bind("close", "<Leave>", lambda _e: (c.itemconfigure(box, fill=INK), c.itemconfigure(x, fill=PAPER),
                                                   c.configure(cursor="")))
        c.tag_bind("close", "<Button-1>", lambda _e: on_close())
        c.tag_bind("drag", "<ButtonPress-1>", self._grab)  # move the card by its title bar
        c.tag_bind("drag", "<B1-Motion>", self._drag)
        win.bind("<Escape>", lambda _e: on_close())

    def _grab(self, ev):
        self._offset = (ev.x_root - self.win.winfo_x(), ev.y_root - self.win.winfo_y())

    def _drag(self, ev):
        self.win.geometry(f"+{ev.x_root - self._offset[0]}+{ev.y_root - self._offset[1]}")

    def finish(self, height):
        """Give the card its height; returns the window size."""
        e, W, S = 1.25, self.W, self.S
        self.c.coords(self.shadow, *round_rect(e + S, e + S, W - e + S, height - e + S, 12))
        self.c.coords(self.card, *round_rect(e, e, W - e, height - e, 12))
        self.size = (W + S + 2, int(height) + S + 2)
        self.c.configure(height=self.size[1])
        return self.size

    def button(self, x, y, text, primary, command):
        """A chunky retro button (pushes into its shadow when clicked). Returns its right edge."""
        c = self.c
        f = font("display", 12, "bold")
        w, h = f.measure(text) + 36, 32
        fill = ORANGE if primary else PAPER
        shadow = c.create_polygon(round_rect(x + 3, y + 3, x + w + 3, y + h + 3, 7), fill=INK, outline=INK, width=2.5)
        tag = f"btn{x:.0f}"
        face = c.create_polygon(round_rect(x, y, x + w, y + h, 7), fill=fill, outline=INK, width=2.5, tags=(tag,))
        c.create_text(x + w / 2, y + h / 2 + 1, text=text, font=f, fill=INK, tags=(tag,))
        hover = ORANGE_HOVER if primary else mix(PAPER, MUSTARD, 0.35)
        hand = "pointinghand" if sys.platform == "darwin" else "hand2"
        c.tag_bind(tag, "<Enter>", lambda _e: (c.itemconfigure(face, fill=hover), c.configure(cursor=hand)))
        c.tag_bind(tag, "<Leave>", lambda _e: (c.itemconfigure(face, fill=fill), c.configure(cursor="")))

        def press(_e):
            c.move(tag, 3, 3)
            c.itemconfigure(shadow, state="hidden")
            self.win.after(120, lambda: (c.move(tag, -3, -3), c.itemconfigure(shadow, state="normal"), command()))

        c.tag_bind(tag, "<ButtonPress-1>", press)
        return x + w


class Meter:
    """The small live meter: blocks up to the current attention, a tick at the talk's typical level."""
    W, HEADER = 132, 26

    def __init__(self, win, on_close):
        self.rw = rw = RetroWindow(win, self.W, "ATTENTION", self.HEADER, 5, on_close, stripes=False)
        c = rw.c
        top, bh, gap = self.HEADER + 16, 9, 3
        self.x0, self.x1 = 34, self.W - 26
        self.blocks = []
        for i in range(BLOCKS):  # bottom to top
            y = top + (BLOCKS - 1 - i) * (bh + gap)
            color = RUST if i < 3 else ORANGE if i < 6 else MUSTARD if i < 9 else TEAL
            self.blocks.append((c.create_rectangle(self.x0, y, self.x1, y + bh, fill=EMPTY, outline=""), color))
        self.col = (top, top + BLOCKS * (bh + gap) - gap)
        self.typical = c.create_polygon(0, 0, 0, 0, 0, 0, fill=INK, outline="", state="hidden")  # ▶ tick
        y = self.col[1] + 26
        self.value = c.create_text(self.W / 2 - 6, y, text="--", font=font("display", 20, "bold"), fill=INK)
        self.trend = c.create_text(self.W / 2 + 34, y, text="", font=font("display", 12, "bold"), fill=TEAL)
        self.status = c.create_text(self.W / 2, y + 24, text="waiting for data", font=font("typewriter", 10),
                                    fill=TEXT2)
        cw, ch = rw.finish(y + 40)
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{cw}x{ch}+{sw - cw - 24}+{sh - ch - 90}")
        self.shown, self.lit = 0.0, -1

    def show(self, level, typical, trend, status):
        """level / typical in 0..1 (None = nothing yet); trend > 0 rising, < 0 falling."""
        c = self.rw.c
        if level is not None:
            self.shown += (level - self.shown) * 0.25  # the column glides instead of jumping
            lit = round(self.shown * BLOCKS)
            if lit != self.lit:
                for i, (block, color) in enumerate(self.blocks):
                    c.itemconfigure(block, fill=color if i < lit else EMPTY)
                self.lit = lit
            c.itemconfigure(self.value, text=f"{round(level * 100)}%")
            arrow, color = ("▲", TEAL) if trend > 0.03 else ("▼", RUST) if trend < -0.03 else ("▬", TEXT2)
            c.itemconfigure(self.trend, text=arrow, fill=color)
        if typical is not None:
            y = self.col[1] - typical * (self.col[1] - self.col[0])
            c.coords(self.typical, self.x0 - 14, y - 5, self.x0 - 14, y + 5, self.x0 - 5, y)
            c.itemconfigure(self.typical, state="normal")
        c.itemconfigure(self.status, text=status)


def draw_profile(c, x0, y0, x1, y1, res, highs, lows, lag):
    """The attention profile in a panel: raw samples, smoothed curve, typical level, numbered high (H) and low
    (L) passages, time axis underneath. Returns the y below the axis labels."""
    prof = res["profile"]
    t = prof["t"]
    raw = [v if v == v else None for v in prof["score"]]  # NaN -> None (gap)
    sm = [v if v == v else None for v in prof["smoothed"]]
    vals = [v for v in raw if v is not None]
    unit = 0 <= min(vals) and max(vals) <= 1  # a 0..1 score: show the axis as percent
    lo, hi = (0.0, 1.0) if unit else (min(vals), max(vals))
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    t0, t1 = t[0], max(t[-1], t[0] + 1)
    c.create_polygon(round_rect(x0, y0, x1, y1, 8), fill=PANEL, outline=INK, width=2)
    px0, px1, py0, py1 = x0 + 42, x1 - 14, y0 + 14, y1 - 12

    def X(v):
        return px0 + (min(max(v, t0), t1) - t0) / (t1 - t0) * (px1 - px0)

    def Y(v):
        return py1 - (v - lo) / (hi - lo) * (py1 - py0)

    small = font("typewriter", 9)
    for k in (0.25, 0.5, 0.75):
        v = lo + k * (hi - lo)
        c.create_line(px0, Y(v), px1, Y(v), fill=FAINT, dash=(2, 4))
        c.create_text(px0 - 8, Y(v), text=f"{round(v * 100)}%" if unit else f"{v:.3g}", anchor="e", font=small,
                      fill=TEXT2)
    placed = []  # tag boxes so far: passages close together stack their tags instead of overlapping
    for items, color, tag in ((highs, TEAL, "H"), (lows, RUST, "L")):
        for i, e in enumerate(items, 1):
            xa, xb = X(e["start"] + lag), X(e["end"] + lag)
            if xb - xa < 3:
                xa, xb = xa - 2, xb + 2
            c.create_rectangle(xa, py0 - 6, xb, py1 + 4, fill=mix(PANEL, color, 0.28), outline="")
            ty = py0 - 3
            while any(xa + 4 < b[2] + 2 and b[0] < xa + 30 and ty < b[3] and b[1] < ty + 14 for b in placed):
                ty += 15
            label = c.create_text(xa + 4, ty, text=f"{tag}{i}", anchor="nw", font=font("typewriter", 10, "bold"),
                                  fill=INK)
            placed.append(c.bbox(label))
    typical = res["summary"]["typical"]
    c.create_line(px0, Y(typical), px1, Y(typical), fill=TEXT2, dash=(5, 3))
    c.create_text(px1 - 2, Y(typical) - 3, text="typical", anchor="se", font=small, fill=TEXT2)
    for series, color, width in ((raw, FAINT, 1), (sm, INK, 2.5)):
        run = []
        for tv, v in zip(list(t) + [None], series + [None]):  # one line per stretch without gaps
            if v is None or tv is None:
                if len(run) >= 4:
                    c.create_line(*run, fill=color, width=width, smooth=True, capstyle="round", joinstyle="round")
                run = []
            else:
                run += [X(tv), Y(v)]
    step = next((s for s in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800) if (t1 - t0) / s <= 8), 3600)
    tick = math.ceil(t0 / step) * step
    while tick <= t1 + 1e-6:
        c.create_line(X(tick), y1, X(tick), y1 + 4, fill=INK, width=1.5)
        c.create_text(X(tick), y1 + 6, text=hms(tick), anchor="n", font=small, fill=TEXT2)
        tick += step
    return y1 + 22


class Feedback:
    """The card after the talk. result: res (attention profile), episodes (attention_insights passages),
    tips, report (path or None), n_words, lag."""
    W, PAD, HEADER = 720, 24, 34

    def __init__(self, win, result, on_close, sound=True):
        self.win, self.on_close = win, on_close
        self.rw = rw = RetroWindow(win, self.W, "PRESENTATION FEEDBACK", self.HEADER, 7, self.leave)
        c, W, PAD = rw.c, self.W, self.PAD
        res, lag = result["res"], result["lag"]
        highs, lows = self.subjects(result)

        y = self.HEADER + PAD - 6
        c.create_text(PAD, y, anchor="nw", text="How the room felt", font=font("display", 22, "bold"), fill=INK)
        raw = [v for v in res["profile"]["score"] if v == v]
        mean = sum(raw) / len(raw)
        unit = 0 <= min(raw) and max(raw) <= 1
        bits = [hms(res["summary"]["duration_s"]), f"{result['n_words']} words heard" if result["n_words"] else
                "no transcript", f"average attention {round(mean * 100)}%" if unit else f"average {mean:.3g}"]
        if res["peaks"]:
            bits.append(f"most engaged at {hms(res['peaks'][0]['time'])}")
        y = c.bbox(c.create_text(PAD, y + 34, anchor="nw", text="  ·  ".join(bits),
                                 font=font("typewriter", 12), fill=TEXT2))[3] + 14
        y = draw_profile(c, PAD, y, W - PAD, y + 180, res, highs, lows, lag) + 10

        col_w = (W - 2 * PAD - 28) / 2
        bottoms = []
        for x, title, color, items, tag, empty in (
                (PAD, "MOST ENGAGING", TEAL, highs, "H", "Nothing stood out as clearly more engaging."),
                (PAD + col_w + 28, "LEAST ENGAGING", RUST, lows, "L", "Nothing stood out as clearly less engaging.")):
            bottoms.append(self.column(x, y, col_w, title, color, items, tag, empty, result["n_words"]))
        y = max(bottoms) + 6

        if result["tips"]:
            y = self.chip(PAD, y + 8, "TIPS", MUSTARD) + 8
            for tip in result["tips"][:4]:
                y = c.bbox(c.create_text(PAD + 4, y, anchor="nw", text=f"•  {tip}", width=W - 2 * PAD - 8,
                                         font=font("typewriter", 12), fill=INK))[3] + 6

        y += 14  # buttons, right-aligned
        f = font("display", 12, "bold")
        x = W - PAD - (f.measure("THANKS!") + 36) - 3
        if result["report"]:
            report = Path(result["report"])
            x -= f.measure("OPEN REPORT") + 36 + 14
            rw.button(x, y, "OPEN REPORT", False, lambda: subprocess.Popen(["open", str(report)])
                      if sys.platform == "darwin" else os.startfile(str(report)))  # noqa: S606
            x = W - PAD - (f.measure("THANKS!") + 36) - 3
        rw.button(x, y, "THANKS!", True, self.leave)
        cw, ch = rw.finish(y + 32 + PAD)

        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        self.target = ((sw - cw) // 2, max(30, int(sh * 0.47 - ch / 2)))
        self.start = (self.target[0], self.target[1] - 40)
        win.geometry(f"{cw}x{ch}+{self.start[0]}+{self.start[1]}")
        chime(sound)
        self.t0, self.phase = time.monotonic(), "in"
        win.after(16, self.tick)

    @staticmethod
    def subjects(result):
        """Up to three most and three least engaging passages (attention_insights), else the profile's moments."""
        eps = result["episodes"]
        highs = sorted([e for e in eps if e["kind"] == "high"], key=lambda e: -e["z"])[:3]
        lows = sorted([e for e in eps if e["kind"] == "low"], key=lambda e: e["z"])[:3]
        if not eps:  # no transcript passages: fall back to the attention profile's high and low moments
            lag, res = result["lag"], result["res"]
            for key, items, name in (("peaks", highs, "High point"), ("dips", lows, "Low point")):
                for ev in res[key][:3]:
                    items.append({"start": ev["start"] - lag, "end": ev["end"] - lag, "label": name, "keywords": [],
                                  "text": ev.get("said", ""), "why": ""})
        return highs, lows

    def chip(self, x, y, text, color):
        c = self.rw.c
        f = font("display", 11, "bold")
        w = f.measure(text) + 20
        c.create_polygon(round_rect(x, y, x + w, y + 22, 11), fill=color, outline=INK, width=2)
        c.create_text(x + w / 2, y + 11, text=text, font=f, fill=PAPER if color != MUSTARD else INK)
        return y + 22

    def column(self, x, y, w, title, color, items, tag, empty, n_words):
        c = self.rw.c
        y = self.chip(x, y, title, color) + 10
        if not items:
            note = empty if n_words else empty + " (No transcript, so no subjects.)"
            return c.bbox(c.create_text(x, y, anchor="nw", text=note, width=w, font=font("typewriter", 12),
                                        fill=TEXT2))[3]
        for i, e in enumerate(items, 1):
            when = f"{tag}{i}   {hms(e['start'])}–{hms(e['end'])}"
            y = c.bbox(c.create_text(x, y, anchor="nw", text=when, font=font("typewriter", 10, "bold"), fill=TEXT2))[3] + 1
            label = e["label"][:1].upper() + e["label"][1:]
            y = c.bbox(c.create_text(x, y, anchor="nw", text=label, width=w, font=font("display", 15, "bold"),
                                     fill=INK))[3] + 2
            if e.get("keywords"):
                related = "related: " + " · ".join(e["keywords"][:5])
                y = c.bbox(c.create_text(x, y, anchor="nw", text=related, width=w, font=font("typewriter", 11),
                                         fill=TEXT2))[3] + 2
            if e.get("why"):
                y = c.bbox(c.create_text(x, y, anchor="nw", text=e["why"], width=w, font=font("typewriter", 11),
                                         fill=INK))[3] + 2
            if e.get("text"):
                text = e["text"] if len(e["text"]) <= 150 else e["text"][:147].rsplit(" ", 1)[0] + " …"
                y = c.bbox(c.create_text(x, y, anchor="nw", text=f"“{text}”", width=w,
                                         font=font("typewriter", 11), fill=TEXT2))[3]
            y += 12
        return y

    def tick(self):
        e = time.monotonic() - self.t0
        if self.phase == "out":
            alpha = max(0.0, 1 - (time.monotonic() - self.t_out) / 0.22)
            self.win.attributes("-alpha", alpha)
            if alpha <= 0:
                self.on_close()
                return
        else:
            if e < 0.6:  # drop in with a little bounce
                p = ease_out_back(e / 0.5)
                self.win.geometry(f"+{self.target[0]}+{round(self.start[1] + (self.target[1] - self.start[1]) * p)}")
            self.win.attributes("-alpha", min(1.0, e / 0.18))
            if e >= 0.6:
                return  # settled: no more frames needed until "out"
        self.win.after(16, self.tick)

    def leave(self):
        if self.phase != "out":
            self.phase, self.t_out = "out", time.monotonic()
            self.win.after(16, self.tick)


class App:
    """Meter while the feed streams; when it ends (or the meter is closed) finish() runs in a thread, then the
    feedback card replaces the meter. Closing the card quits."""

    def __init__(self, root, feed, finish, sound):
        import threading
        self.root, self.feed, self.finish, self.sound, self.threading = root, feed, finish, sound, threading
        self.meter = Meter(root, self.end_now)
        root.attributes("-alpha", 1.0)
        self.idx, self.ema, self.t_ema, self.lo, self.hi = 0, None, None, None, None
        self.history, self.values = deque(), []
        self.result, self.state, self.dots = None, "live", 0
        root.after(50, self.tick)

    def end_now(self):
        self.feed.done = True  # closing the meter early: feedback on what came in so far

    def norm(self, v):
        if self.lo is None or (0 <= self.lo and self.hi <= 1):
            return min(max(v, 0.0), 1.0)
        return (v - self.lo) / ((self.hi - self.lo) or 1.0)

    def tick(self):
        f = self.feed
        with f.lock:
            new = list(zip(f.prof.t[self.idx:], f.prof.s[self.idx:]))
            self.idx += len(new)
        for t, v in new:  # attention smoothed over ~2 s of talk time, so the column moves but doesn't flicker
            self.lo, self.hi = (v, v) if self.lo is None else (min(self.lo, v), max(self.hi, v))
            if self.ema is None:
                self.ema, self.t_ema = v, t
            else:
                k = 1 - math.exp(-max(t - self.t_ema, 1e-3) / 2.0)
                self.ema, self.t_ema = self.ema + k * (v - self.ema), t
            self.history.append((t, self.ema))
            self.values.append(v)
            while self.history and self.history[0][0] < t - 5:
                self.history.popleft()
        if self.state == "live":
            if self.ema is None:
                self.meter.show(None, None, 0, "waiting for data")
            else:
                level = self.norm(self.ema)
                trend = level - self.norm(self.history[0][1])
                values = sorted(self.values)
                self.meter.show(level, self.norm(values[len(values) // 2]), trend, "live")
            if f.done:
                self.state = "analysing"
                self.threading.Thread(target=self.run_finish, daemon=True).start()
        elif self.state == "analysing":
            self.dots = (self.dots + 1) % 24
            self.meter.show(None, None, 0, "analysing" + "." * (self.dots // 6))
            if self.result is not None:
                self.state = "feedback"
                if self.result is False:  # not enough data for feedback
                    self.root.destroy()
                    return
                import tkinter as tk
                self.root.withdraw()
                Feedback(tk.Toplevel(self.root), self.result, self.root.destroy, self.sound)
                return
        self.root.after(40, self.tick)

    def run_finish(self):
        try:
            self.result = self.finish() or False
        except Exception as e:  # noqa: BLE001 - never leave the presenter with a frozen meter
            print(f"feedback failed: {e}", file=sys.stderr)
            self.result = False


def run(feed, finish, sound=True):
    """Show the meter while `feed` streams; when it ends, call finish() -> result dict for the feedback card."""
    import tkinter as tk
    root = tk.Tk()
    App(root, feed, finish, sound)
    root.mainloop()
