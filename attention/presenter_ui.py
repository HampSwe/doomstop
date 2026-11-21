"""
presenter_ui.py - what the presenter sees: a small attention meter during the talk, a feedback card after it.

Same warm retro look as message/popup.py (cream paper, ink outlines, hard shadows, Futura + American Typewriter),
drawn with tkinter only.

  Meter      a small always-on-top card in the bottom-right corner (drag it anywhere): an LED eye, like the LED
             ear of classroom noise meters, that lights up and turns from red to green as attention rises. The
             eye is the only indicator: no numbers to read mid-talk.
  Feedback   when the stream has ended: the attention over the whole talk. Hovering over the graph shows the
             topic at that moment. "OPEN REPORT" opens the full attention_insights report.

live_insights.py runs this; run(feed, finish) is the entry point.
"""
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from attention_profile import hms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "message"))
from popup import (INK, MUSTARD, ORANGE, ORANGE_HOVER, PAPER, RUST, TEAL, TEXT2,  # noqa: E402
                   ease_out_back, round_rect)

PANEL, FAINT = "#efe3c9", "#cbbd9d"   # chart background, grid / topic changes
DISPLAY, TYPEWRITER = ("Futura", "Arial Rounded MT Bold"), ("American Typewriter", "Courier New")
SOUND = "/System/Library/Sounds/Hero.aiff"
LED_RED, LED_AMBER, LED_GREEN = "#ff4a2e", "#ffb31f", "#6fe04a"  # the meter's eye: glowing LEDs on dark
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


def led_color(level):
    """Attention 0..1 -> LED colour: red (lost them), amber, green (with you)."""
    stops = ((0.0, LED_RED), (0.5, LED_AMBER), (1.0, LED_GREEN))
    level = min(max(level, 0.0), 1.0)
    for (l0, c0), (l1, c1) in zip(stops, stops[1:]):
        if level <= l1:
            return mix(c0, c1, (level - l0) / (l1 - l0))
    return LED_GREEN


class Meter:
    """The small live meter, after the LED ear of classroom noise meters but as an eye on a dark panel.

    The whole eye takes the attention's colour (red -> amber -> green), and as attention rises the iris ring and
    the eyelids fill up from the middle of the eye out to its sides: a dim red eye with only its centre lit is a
    room that has drifted off, a fully lit green eye is a room that is with you. The pupil always glows."""
    W, HEADER = 176, 26
    LIDS, IRIS = 13, 14  # LEDs along each eyelid (corners shared), LEDs on the iris ring

    def __init__(self, win, on_close):
        self.rw = rw = RetroWindow(win, self.W, "ATTENTION", self.HEADER, 5, on_close, stripes=False)
        c = rw.c
        pad, ph = 12, 104
        top = self.HEADER + pad
        c.create_polygon(round_rect(pad, top, self.W - pad, top + ph, 10), fill=INK, outline=INK, width=2)
        cx, cy, a, b = self.W / 2, top + ph / 2, (self.W - 2 * pad) / 2 - 14, 38  # eye centre, half width / height

        def lid(u, sign):  # an almond: two parabolas meeting in the corners
            return cx + a * u, cy + sign * b * (1 - u * u)

        lids = {sign: [lid(i / 40 - 1, sign) for i in range(81)] for sign in (-1, 1)}
        self.tracks = [c.create_line(*[v for p in pts for v in p], width=8, capstyle="round", smooth=True)
                       for pts in lids.values()]  # both tracks first, so neither covers the corner LEDs
        self.leds = []  # (halo, core, rank): rank 0 on the eye's vertical centre line, 1 at its sides
        for sign, pts in lids.items():
            # LEDs evenly spaced along the lid's length; the lower lid leaves the corners to the upper one
            dist = [0.0]
            for p, q in zip(pts, pts[1:]):
                dist.append(dist[-1] + math.dist(p, q))
            for k in range(self.LIDS) if sign < 0 else range(1, self.LIDS - 1):
                d = dist[-1] * k / (self.LIDS - 1)
                i = min(range(len(dist)), key=lambda j: abs(dist[j] - d))
                self.leds.append(self.led(*pts[i], 3.2, abs(pts[i][0] - cx) / a))
        ri = b * 0.62  # iris ring
        self.tracks.append(c.create_oval(cx - ri, cy - ri, cx + ri, cy + ri, width=7, outline=""))
        for k in range(self.IRIS):
            ang = 2 * math.pi * k / self.IRIS
            self.leds.append(self.led(cx + ri * math.cos(ang), cy + ri * math.sin(ang), 2.6, abs(math.cos(ang))))
        self.pupil = self.led(cx, cy, 9.5, None)
        self.glint = c.create_oval(cx - 5.5, cy - 5.5, cx - 1.5, cy - 1.5, outline="", fill=INK)

        y = top + ph + 16  # status line under the eye: "waiting for data" / "analysing", empty while live
        self.status = c.create_text(self.W / 2, y, text="waiting for data", font=font("typewriter", 10),
                                    fill=TEXT2)
        cw, ch = rw.finish(y + 16)
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{cw}x{ch}+{sw - cw - 24}+{sh - ch - 90}")
        self.shown, self.drawn = 0.0, None
        self.paint(None)

    def led(self, x, y, r, rank):
        """One LED: a soft halo (only when lit) around a bright core."""
        c, h = self.rw.c, r * 1.9
        halo = c.create_oval(x - h, y - h, x + h, y + h, outline="", fill=INK)
        core = c.create_oval(x - r, y - r, x + r, y + r, outline="", fill=INK)
        return halo, core, rank

    def paint(self, level):
        """Light the eye for attention `level` (None: no data yet, everything off)."""
        c = self.rw.c
        color = led_color(level) if level is not None else "#8c8170"
        off, glow = mix(INK, color, 0.2), mix(INK, color, 0.38)
        for track in self.tracks:
            c.itemconfigure(track, **{"outline" if c.type(track) == "oval" else "fill": mix(INK, color, 0.12)})
        for halo, core, rank in self.leds + [self.pupil]:
            on = level is not None and (rank is None or rank <= level / 0.95)  # fully lit from 95 %
            c.itemconfigure(halo, fill=glow if on else INK)
            c.itemconfigure(core, fill=color if on else off)
        c.itemconfigure(self.glint, fill=mix(color, "#ffffff", 0.75) if level is not None else off)

    def show(self, level, status=""):
        """level in 0..1 (None = nothing new). `status` is a short line under the eye ('' while live)."""
        if level is not None:
            self.shown += (level - self.shown) * 0.25  # the eye brightens and dims smoothly instead of jumping
            step = round(self.shown * 100)
            if step != self.drawn:
                self.paint(step / 100)
                self.drawn = step
        self.rw.c.itemconfigure(self.status, text=status)


def topic_spans(result):
    """(start, end, topic) in attention time: the topic markers in the data if there were any, else the
    transcript chunks (spoken at t, seen in the attention at t + lag)."""
    res = result["res"]
    if res["summary"]["topics_from"] == "markers":
        return sorted((d["start"], d["end"], d["topic"]) for d in res["topics"])
    lag = result["lag"]
    return [(c["start"] + lag, c["end"] + lag, c["topic"]) for c in result.get("chunks", []) if c.get("topic")]


def topic_at(spans, t, reach=3.0):
    """The topic at attention time t: the span around it, else the nearest one within `reach` seconds."""
    best, gap = None, reach
    for a, b, name in spans:
        d = 0.0 if a <= t <= b else min(abs(t - a), abs(t - b))
        if d <= gap:
            best, gap = name, d
    return best


class Feedback:
    """The card after the talk: the attention over the whole talk. Hovering over the graph shows the topic at
    that moment. result: res (attention profile), chunks (transcript chunks with topics), report (path or None),
    n_words, lag."""
    W, PAD, HEADER, GRAPH_H = 720, 24, 34, 280

    def __init__(self, win, result, on_close, sound=True):
        self.win, self.on_close = win, on_close
        self.rw = rw = RetroWindow(win, self.W, "PRESENTATION FEEDBACK", self.HEADER, 7, self.leave)
        c, W, PAD = rw.c, self.W, self.PAD
        res = result["res"]
        self.spans = topic_spans(result)

        y = self.HEADER + PAD - 6
        c.create_text(PAD, y, anchor="nw", text="How the room felt", font=font("display", 22, "bold"), fill=INK)
        raw = [v for v in res["profile"]["score"] if v == v]
        mean = sum(raw) / len(raw)
        unit = 0 <= min(raw) and max(raw) <= 1
        bits = [hms(res["summary"]["duration_s"]), f"average attention {round(mean * 100)}%" if unit else
                f"average {mean:.3g}", "click the graph to see the topic" if self.spans else
                "no transcript, so no topics"]
        y = c.bbox(c.create_text(PAD, y + 34, anchor="nw", text="  ·  ".join(bits),
                                 font=font("typewriter", 12), fill=TEXT2))[3] + 14
        y = self.graph(PAD, y, W - PAD, y + self.GRAPH_H, res, unit)

        y += 14  # buttons, right-aligned
        f = font("display", 12, "bold")
        x = W - PAD - (f.measure("THANKS!") + 36) - 3
        if result["report"]:
            report = Path(result["report"])
            rw.button(x - (f.measure("OPEN REPORT") + 36 + 14), y, "OPEN REPORT", False,
                      lambda: subprocess.Popen(["open", str(report)])
                      if sys.platform == "darwin" else os.startfile(str(report)))  # noqa: S606
        rw.button(x, y, "THANKS!", True, self.leave)
        cw, ch = rw.finish(y + 32 + PAD)

        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        self.target = ((sw - cw) // 2, max(30, int(sh * 0.47 - ch / 2)))
        self.start = (self.target[0], self.target[1] - 40)
        win.geometry(f"{cw}x{ch}+{self.start[0]}+{self.start[1]}")
        chime(sound)
        win.focus_force()  # so the arrow keys move the graph's marker
        self.t0, self.phase = time.monotonic(), "in"
        win.after(16, self.tick)

    def graph(self, x0, y0, x1, y1, res, unit):
        """The attention trajectory in a panel: the smoothed attention over time, faint lines where the topic
        changes, time axis underneath; hovering shows the topic. Returns the y below the axis labels."""
        c = self.rw.c
        prof = res["profile"]
        t = prof["t"]
        sm = [v if v == v else None for v in prof["smoothed"]]  # NaN -> None (gap)
        vals = [v for v in sm if v is not None]
        lo, hi = (0.0, 1.0) if unit else (min(vals), max(vals))
        if hi - lo < 1e-9:
            lo, hi = lo - 1, hi + 1
        t0, t1 = t[0], max(t[-1], t[0] + 1)
        c.create_polygon(round_rect(x0, y0, x1, y1, 8), fill=PANEL, outline=INK, width=2)
        px0, px1, py0, py1 = x0 + 46, x1 - 16, y0 + 16, y1 - 12
        X = lambda v: px0 + (min(max(v, t0), t1) - t0) / (t1 - t0) * (px1 - px0)  # noqa: E731
        Y = lambda v: py1 - (v - lo) / (hi - lo) * (py1 - py0)  # noqa: E731

        small = font("typewriter", 9)
        for k in (0, 0.25, 0.5, 0.75, 1):
            v = lo + k * (hi - lo)
            c.create_line(px0, Y(v), px1, Y(v), fill=FAINT, dash=(2, 4))
            c.create_text(px0 - 8, Y(v), text=f"{round(v * 100)}%" if unit else f"{v:.3g}", anchor="e",
                          font=small, fill=TEXT2)
        for (_, _, prev), (a, _, name) in zip(self.spans, self.spans[1:]):
            if name != prev and t0 < a < t1:
                c.create_line(X(a), py0, X(a), py1, fill=FAINT, width=1.5)
        run = []
        for tv, v in zip(list(t) + [None], sm + [None]):  # one curve (and fill) per stretch without gaps
            if v is None or tv is None:
                if len(run) >= 4:
                    c.create_polygon(run[0], py1, *run, run[-2], py1, fill=mix(PANEL, TEAL, 0.22), outline="")
                    c.create_line(*run, fill=INK, width=2.5, smooth=True, capstyle="round", joinstyle="round")
                run = []
            else:
                run += [X(tv), Y(v)]
        step = next((s for s in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800) if (t1 - t0) / s <= 8), 3600)
        tick = math.ceil(t0 / step) * step
        while tick <= t1 + 1e-6:
            c.create_line(X(tick), y1, X(tick), y1 + 4, fill=INK, width=1.5)
            c.create_text(X(tick), y1 + 6, text=hms(tick), anchor="n", font=small, fill=TEXT2)
            tick += step

        # the marker: shown from the start; a click or drag places it (it stays there), hovering previews other
        # moments when the window gets mouse moves, and the arrow keys step it
        self.points = [(X(tv), Y(v), tv, v) for tv, v in zip(t, sm) if v is not None]
        self.box, self.panel, self.unit, self.marked = (px0, py0, px1, py1), (x0, y0, x1, y1 + 22), unit, None
        # start at the first moment with a topic (the talk's first seconds often have none)
        self.pinned = next((k for k, p in enumerate(self.points) if topic_at(self.spans, p[2])), 0)
        for ev in ("<Motion>", "<Button-1>", "<B1-Motion>"):
            c.bind(ev, lambda e, ev=ev: self.pointer(e.x, e.y, pin=ev != "<Motion>"))
        c.bind("<Leave>", lambda _e: self.mark(self.pinned))
        step = max(1, len(self.points) // 80)
        self.win.bind("<Left>", lambda _e: self.pin(self.pinned - step))
        self.win.bind("<Right>", lambda _e: self.pin(self.pinned + step))
        self.mark(self.pinned)
        return y1 + 22

    def pointer(self, mx, my, pin):
        """Mouse at (mx, my): anywhere on the graph's panel (axis labels too), mark the moment nearest in time
        (and keep it there on a click / drag); elsewhere, show the kept one."""
        x0, y0, x1, y1 = self.panel
        if not self.points:
            return
        if x0 <= mx <= x1 and y0 <= my <= y1:
            i = min(range(len(self.points)), key=lambda k: abs(self.points[k][0] - mx))
            self.pin(i) if pin else self.mark(i)
        else:
            self.mark(self.pinned)

    def pin(self, i):
        if self.points:
            self.pinned = min(max(i, 0), len(self.points) - 1)
            self.mark(self.pinned)

    def mark(self, i):
        """A guide line, a dot on the curve and a tooltip with the time and topic of moment i."""
        if not self.points or i == self.marked:
            return
        self.marked = i
        c, unit = self.rw.c, self.unit
        c.delete("hover")
        px0, py0, px1, py1 = self.box
        x, y, tv, v = self.points[i]
        c.create_line(x, py0, x, py1, fill=INK, dash=(3, 3), tags=("hover",))
        c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=ORANGE, outline=INK, width=2, tags=("hover",))

        topic = topic_at(self.spans, tv)
        level = f"{round(v * 100)}% attention" if unit else f"attention {v:.3g}"
        head = c.create_text(0, 0, anchor="nw", text=f"{hms(tv)}  ·  {level}", font=font("typewriter", 10),
                             fill=TEXT2, tags=("hover",))
        body = c.create_text(0, 0, anchor="nw", text=topic or "no topic here", width=240, tags=("hover",),
                             font=font("display", 14, "bold") if topic else font("typewriter", 11),
                             fill=INK if topic else TEXT2)
        hb, bb = c.bbox(head), c.bbox(body)
        w = max(hb[2] - hb[0], bb[2] - bb[0]) + 24
        h = (hb[3] - hb[1]) + (bb[3] - bb[1]) + 22
        tx = min(max(x - w / 2, px0 - 30), px1 + 10 - w)
        ty = y - h - 16 if y - h - 16 >= py0 - 10 else y + 16  # above the dot, below it near the top
        c.create_polygon(round_rect(tx + 3, ty + 3, tx + w + 3, ty + h + 3, 7), fill=INK, outline=INK, width=2,
                         tags=("hover",))
        card = c.create_polygon(round_rect(tx, ty, tx + w, ty + h, 7), fill=PAPER, outline=INK, width=2,
                                tags=("hover",))
        c.coords(head, tx + 12, ty + 9)
        c.coords(body, tx + 12, ty + 11 + hb[3] - hb[1])
        c.tag_raise(head, card)
        c.tag_raise(body, card)

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
        for t, v in new:  # attention smoothed over ~2 s of talk time, so the eye changes but doesn't flicker
            self.lo, self.hi = (v, v) if self.lo is None else (min(self.lo, v), max(self.hi, v))
            if self.ema is None:
                self.ema, self.t_ema = v, t
            else:
                k = 1 - math.exp(-max(t - self.t_ema, 1e-3) / 2.0)
                self.ema, self.t_ema = self.ema + k * (v - self.ema), t
        if self.state == "live":
            if self.ema is None:
                self.meter.show(None, "waiting for data")
            else:
                self.meter.show(self.norm(self.ema))
            if f.done:
                self.state = "analysing"
                self.threading.Thread(target=self.run_finish, daemon=True).start()
        elif self.state == "analysing":
            self.dots = (self.dots + 1) % 24
            self.meter.show(None, "analysing" + "." * (self.dots // 6))
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
