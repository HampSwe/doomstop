#!/usr/bin/env python3
"""
popup.py - a friendly pop-up card with a warm retro look (tkinter, standard library only).

    python message/popup.py                          # a random friendly "stop scrolling" message
    python message/popup.py --title "Hi there" --text "Back to the meeting?" --seconds 8
    python message/popup.py --position top-right     # top-right corner instead of the centre of the screen
    python message/popup.py --icon smiley            # the drawn smiley instead of the LED eye
    python message/popup.py --voice sound/doomscroll_detected.mp3   # a spoken clip instead of the chime
    python message/popup.py --check                  # test: build every message off-screen, check the layout

listener.py starts this in its own process for every signal (style "card", the default). The card drops into
the centre of the screen with the classic Mac chime and closes itself after --seconds; hovering over it
pauses the countdown. "I'M BACK!" closes it with a small thank-you, the × box closes it right away.

The look: cream paper, dark-brown ink outlines with a hard offset shadow, a title bar with 70s stripes, the
lecture tool's LED eye fully lit (attention/presenter_ui.py: a focused eye) or a drawn smiley, Futura +
American Typewriter, and an LCD-style row of blocks counting down.
"""
import argparse
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time

# kind of pop-up -> friendly messages (title, text). listener.py picks one at random, never the same one twice
# in a row, so the nudge stays kind and doesn't get stale. Add messages or new kinds here.
MESSAGES = {
    "scrolling": [
        ("Psst… look up for a sec", "Your feed will still be there later. The good stuff is happening right here."),
        ("Your thumbs deserve a break", "Put the phone down for a bit. You might catch the best part."),
        ("Hey there, eyes up!", "The scroll can wait. Your people are right here."),
        ("Quick breather?", "Rest your eyes from the feed and tune back in. You've got this."),
        ("Small win, right now", "Phone face down, eyes up. Future you says thanks."),
    ],
}
LABEL, BUTTON = "FRIENDLY REMINDER", "I'M BACK!"
THANKS = ("THANK YOU", "Groovy, welcome back!", "Enjoy the rest of it.")  # label, title, text after "I'm back"

W, PAD, RADIUS, SHADOW = 460, 24, 14, 7  # card width, padding, corner radius, hard-shadow offset (points)
HEADER, FACE_R, BLOCKS = 34, 30, 20      # title-bar height, smiley radius, number of countdown blocks
INK, PAPER, TEXT2, EMPTY = "#2b2118", "#f6eedc", "#4d3d2f", "#e6d9bd"
RUST, ORANGE, MUSTARD, TEAL = "#c8442c", "#ec7a30", "#f2b23a", "#2f7d74"
ORANGE_HOVER = "#f39552"
LED_GREEN = "#6fe04a"                    # the lecture tool's eye when the room is fully with you
EYE_W, EYE_H = 92, 64                    # the dark panel the LED eye sits on
SOUND = "/System/Library/Sounds/Sosumi.aiff"  # the classic Mac alert sound
# the sound Windows itself plays for a notification (the "Asterisk" event is only a soft thud)
WINDOWS_SOUND = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Media", "Windows Notify System Generic.wav")


def mix(a, b, t):
    """Blend two #rrggbb colours: t = 0 -> a, t = 1 -> b."""
    ca, cb = ([int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def playable(path):
    """The file itself, or a copy without its ID3v2 tag: Windows' player refuses some of those tags
    ("a problem occurred in initializing MCI"), and plays the same file fine without it."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:3] != b"ID3" or len(data) < 10:
        return path
    size = 10 + ((data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]) + (10 if data[5] & 0x10 else 0)
    copy = os.path.join(tempfile.gettempdir(), "kgnugen_popup_voice" + os.path.splitext(path)[1])
    with open(copy, "wb") as f:  # one fixed name, overwritten each time: nothing piles up
        f.write(data[size:])
    return copy


def play_file(path):
    """Play an audio file (mp3 or wav) and wait until it has finished. Does nothing if it cannot."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            import ctypes
            mci, alias = ctypes.windll.winmm.mciSendStringW, f"popup{os.getpid()}"
            kind = "" if path.lower().endswith(".wav") else "type mpegvideo "
            if mci(f'open "{playable(path)}" {kind}alias {alias}', None, 0, None) == 0:
                mci(f"play {alias} wait", None, 0, None)
                mci(f"close {alias}", None, 0, None)
    except OSError:
        pass


def play_sounds(voice=None):
    """The spoken clip `voice` (a file) if there is one, and nothing else; otherwise the chime."""
    if voice and os.path.exists(voice):
        play_file(voice)
        return
    if sys.platform == "darwin" and os.path.exists(SOUND):
        subprocess.run(["afplay", SOUND], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform == "win32":  # Windows has no Sosumi: its own notification sound instead
        import winsound
        if os.path.exists(WINDOWS_SOUND):
            winsound.PlaySound(WINDOWS_SOUND, winsound.SND_FILENAME)
        else:
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)


def ease_out_back(p):
    """0 -> 1 with a small overshoot at the end: the card lands with a little bounce."""
    p = min(max(p, 0.0), 1.0) - 1
    return 1 + 2.70158 * p ** 3 + 1.70158 * p ** 2


def round_rect(x1, y1, x2, y2, r, top=True, bottom=True):
    """Polygon points of a rectangle with circular corners; top / bottom = False keeps those corners square."""
    r = min(r, (x2 - x1) / 2, (y2 - y1) / 2)
    pts = []
    for cx, cy, a0, rounded, corner in ((x2 - r, y1 + r, -90, top, (x2, y1)), (x2 - r, y2 - r, 0, bottom, (x2, y2)),
                                        (x1 + r, y2 - r, 90, bottom, (x1, y2)), (x1 + r, y1 + r, 180, top, (x1, y1))):
        if not rounded or r <= 0:
            pts += corner
            continue
        n = max(6, int(r / 1.5))
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts += [cx + r * math.cos(a), cy + r * math.sin(a)]
    return pts


class Card:
    def __init__(self, title, text, seconds=12.0, sound=True, position="center", check=False, icon="eye",
                 voice=None):
        import tkinter as tk
        from tkinter import font as tkfont

        self.total = self.remaining = max(float(seconds), 1.0)
        self.phase, self.hover, self.pressed, self.alpha, self.lit, self.face_dy = "show", False, False, 0.0, BLOCKS, 0.0
        root = self.root = tk.Tk()
        root.overrideredirect(True)  # no title bar
        root.attributes("-alpha", 0.0)
        root.attributes("-topmost", True)
        if sys.platform == "darwin":  # transparent window -> rounded corners and a free-standing shadow
            root.attributes("-transparent", True)
            clear = "systemTransparent"
        elif sys.platform == "win32":
            clear = "#010203"
            root.attributes("-transparentcolor", clear)
        else:
            clear = PAPER
        root.configure(bg=clear)
        c = self.c = tk.Canvas(root, width=W + SHADOW + 2, height=200, bg=clear, highlightthickness=0, bd=0)
        c.pack()

        families = set(tkfont.families())

        def font(candidates, size, weight="normal"):
            family = next((name for name in candidates if name in families), None)
            if family:
                return tkfont.Font(family=family, size=size, weight=weight)
            fallback = tkfont.nametofont("TkDefaultFont").copy()
            fallback.configure(size=size, weight=weight)
            return fallback

        display, typewriter = ("Futura", "Arial Rounded MT Bold"), ("American Typewriter", "Courier New")
        self.fonts = f = {"label": font(typewriter, 11, "bold"), "title": font(display, 20, "bold"),
                          "body": font(typewriter, 13), "button": font(display, 13, "bold"),
                          "close": font(display, 12, "bold")}

        e = 1.25  # half the outline width: keeps the 2.5 pt outlines inside the canvas
        # back to front. The card's height is known only after the text layout, so card + shadow get it later.
        self.shadow = c.create_polygon(0, 0, 1, 0, 1, 1, fill=INK, outline=INK, width=2.5)
        self.card = c.create_polygon(0, 0, 1, 0, 1, 1, fill=PAPER, outline=INK, width=2.5)
        c.create_polygon(round_rect(e, e, W - e, HEADER, RADIUS, bottom=False), fill=INK, outline=INK, width=2.5)

        # title bar: label, 70s stripes, close box
        mid = (HEADER + e) / 2
        self.label = c.create_text(PAD, mid, anchor="w", text=LABEL, font=f["label"], fill=PAPER)
        bx1, by1, bx2, by2 = W - 32, mid - 8, W - 16, mid + 8
        self.closebox = c.create_rectangle(bx1, by1, bx2, by2, fill=INK, outline=PAPER, width=1.5, tags=("close",))
        self.closex = c.create_text((bx1 + bx2) / 2, (by1 + by2) / 2, text="×", font=f["close"], fill=PAPER,
                                    tags=("close",))
        sx0, sx1 = c.bbox(self.label)[2] + 14, bx1 - 12
        self.stripes = [c.create_rectangle(sx0, mid - 11 + 6 * i, sx1, mid - 7 + 6 * i, fill=color, outline="")
                        for i, color in enumerate((RUST, ORANGE, MUSTARD, TEAL))]

        # the picture on the left: everything in it is tagged "face", so it can hop
        self.eye_r = self.wink = None
        if icon == "smiley":  # on its own little hard shadow, which stays put when the face hops
            fx, fy = PAD + FACE_R + 2, HEADER + PAD + FACE_R - 2
            c.create_oval(fx - FACE_R + 3, fy - FACE_R + 3, fx + FACE_R + 3, fy + FACE_R + 3, fill=INK,
                          outline=INK, width=2.5)
            self.face = c.create_oval(fx - FACE_R, fy - FACE_R, fx + FACE_R, fy + FACE_R, fill=MUSTARD,
                                      outline=INK, width=2.5, tags=("face",))
            c.create_oval(fx - 11, fy - 11, fx - 5, fy - 2, fill=INK, outline="", tags=("face",))
            self.eye_r = c.create_oval(fx + 5, fy - 11, fx + 11, fy - 2, fill=INK, outline="", tags=("face",))
            self.wink = c.create_arc(fx + 3, fy - 10, fx + 13, fy - 2, start=20, extent=140, style="arc",
                                     outline=INK, width=2.5, state="hidden", tags=("face",))
            c.create_arc(fx - 17, fy - 15, fx + 17, fy + 16, start=200, extent=140, style="arc", outline=INK,
                         width=3, tags=("face",))
            icon_right, icon_bottom = fx + FACE_R + 3, fy + FACE_R + 3
        else:
            icon_right, icon_bottom = self.led_eye(PAD, HEADER + PAD - 4)

        # text column: title, text, button - each under the previous one, so longer texts still fit
        tx = icon_right + 22
        wrap = W - tx - PAD
        self.title = c.create_text(tx, HEADER + PAD - 6, anchor="nw", text=title, font=f["title"], fill=INK,
                                   width=wrap)
        self.body = c.create_text(tx, c.bbox(self.title)[3] + 4, anchor="nw", text=text, font=f["body"],
                                  fill=TEXT2, width=wrap)
        y = c.bbox(self.body)[3] + 18
        bw, bh = f["button"].measure(BUTTON) + 44, 36
        self.btn_shadow = c.create_polygon(round_rect(tx + 3, y + 3, tx + bw + 3, y + bh + 3, 8), fill=INK,
                                           outline=INK, width=2.5)
        self.btn = c.create_polygon(round_rect(tx, y, tx + bw, y + bh, 8), fill=ORANGE, outline=INK, width=2.5,
                                    tags=("btn",))
        self.btn_label = c.create_text(tx + bw / 2, y + bh / 2 + 1, text=BUTTON, font=f["button"], fill=INK,
                                       tags=("btn",))

        # LCD-style countdown: a row of blocks that switch off one by one
        sy = max(y + bh + 3, icon_bottom) + 18
        block = (W - 2 * PAD - (BLOCKS - 1) * 4) / BLOCKS
        self.blocks = [c.create_rectangle(PAD + i * (block + 4), sy, PAD + i * (block + 4) + block, sy + 7,
                                          fill=TEAL, outline="") for i in range(BLOCKS)]
        self.H = H = int(sy + 7 + PAD - 4)
        c.coords(self.shadow, *round_rect(e + SHADOW, e + SHADOW, W - e + SHADOW, H - e + SHADOW, RADIUS))
        c.coords(self.card, *round_rect(e, e, W - e, H - e, RADIUS))
        cw, ch = W + SHADOW + 2, H + SHADOW + 2
        c.configure(height=ch)

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        if position == "top-right":
            self.target = (sw - cw - 18, 44)
            self.start = (self.target[0] + 70, self.target[1])
        else:  # centre, a little above the middle where the eye lands, dropping in from slightly higher
            self.target = ((sw - cw) // 2, int(sh * 0.45 - ch / 2))
            self.start = (self.target[0], self.target[1] - 40)
        root.geometry(f"{cw}x{ch}+{self.start[0]}+{self.start[1]}")

        hand = "pointinghand" if sys.platform == "darwin" else "hand2"

        def recolor(pairs, cursor):
            return lambda _e: ([c.itemconfigure(item, fill=color) for item, color in pairs], c.configure(cursor=cursor))

        c.tag_bind("btn", "<Enter>", recolor([(self.btn, ORANGE_HOVER)], hand))
        c.tag_bind("btn", "<Leave>", recolor([(self.btn, ORANGE)], ""))
        c.tag_bind("btn", "<ButtonPress-1>", lambda _e: self.press())
        c.tag_bind("close", "<Enter>", recolor([(self.closebox, ORANGE), (self.closex, INK)], hand))
        c.tag_bind("close", "<Leave>", recolor([(self.closebox, INK), (self.closex, PAPER)], ""))
        c.tag_bind("close", "<Button-1>", lambda _e: self.leave())
        c.bind("<Enter>", lambda _e: setattr(self, "hover", True))  # hovering pauses the countdown
        c.bind("<Leave>", lambda _e: setattr(self, "hover", False))
        root.bind("<Escape>", lambda _e: self.leave())
        if check:
            return
        if sound:  # in the background, so the card drops in while the sound plays
            threading.Thread(target=play_sounds, args=(voice,), daemon=True).start()
        self.t_start = self.last = self.hop_t0 = time.monotonic()
        root.lift()
        root.after(16, self.tick)

    def led_eye(self, x0, y0):
        """The lecture tool's LED eye (Meter in attention/presenter_ui.py) on its dark panel, fully lit and
        green: a focused eye. Returns the panel's right and bottom edge."""
        c, color = self.c, LED_GREEN
        glow, track = mix(INK, color, 0.38), mix(INK, color, 0.12)
        self.face = c.create_polygon(round_rect(x0, y0, x0 + EYE_W, y0 + EYE_H, 9), fill=INK, outline=INK,
                                     width=2.5, tags=("face",))
        cx, cy, a, b = x0 + EYE_W / 2, y0 + EYE_H / 2, EYE_W / 2 - 11, 21  # eye centre, half width / height

        def led(x, y, r):  # a soft halo around a bright core
            h = r * 1.9
            c.create_oval(x - h, y - h, x + h, y + h, outline="", fill=glow, tags=("face",))
            c.create_oval(x - r, y - r, x + r, y + r, outline="", fill=color, tags=("face",))

        lids = {sign: [(cx + a * (i / 40 - 1), cy + sign * b * (1 - (i / 40 - 1) ** 2)) for i in range(81)]
                for sign in (-1, 1)}  # an almond: two parabolas meeting in the corners
        for pts in lids.values():  # both tracks first, so neither covers the corner LEDs
            c.create_line(*[v for p in pts for v in p], width=5, capstyle="round", smooth=True, fill=track,
                          tags=("face",))
        n = 9  # LEDs along each eyelid, evenly spaced along its length; the lower lid leaves the corners out
        for sign, pts in lids.items():
            dist = [0.0]
            for p, q in zip(pts, pts[1:]):
                dist.append(dist[-1] + math.dist(p, q))
            for k in range(n) if sign < 0 else range(1, n - 1):
                d = dist[-1] * k / (n - 1)
                led(*pts[min(range(len(dist)), key=lambda j: abs(dist[j] - d))], 2.2)
        ri = b * 0.62  # iris ring
        c.create_oval(cx - ri, cy - ri, cx + ri, cy + ri, width=4.5, outline=track, tags=("face",))
        for k in range(10):
            ang = 2 * math.pi * k / 10
            led(cx + ri * math.cos(ang), cy + ri * math.sin(ang), 1.8)
        led(cx, cy, 5.5)  # the pupil, with its glint
        c.create_oval(cx - 3.4, cy - 3.4, cx - 0.8, cy - 0.8, outline="", fill=mix(color, "#ffffff", 0.75),
                      tags=("face",))
        return x0 + EYE_W, y0 + EYE_H

    def tick(self):
        now = time.monotonic()
        dt, self.last = now - self.last, now
        e = now - self.t_start
        if self.phase == "out":
            alpha = self.out_from * max(0.0, 1.0 - (now - self.t_out) / 0.22)
            if alpha <= 0.0:
                self.root.destroy()
                return
        else:
            if e < 0.6:  # drop in with a little bounce
                p = ease_out_back(e / 0.5)
                x = self.start[0] + (self.target[0] - self.start[0]) * p
                y = self.start[1] + (self.target[1] - self.start[1]) * p
                self.root.geometry(f"+{round(x)}+{round(y)}")
            alpha = min(1.0, e / 0.18)
            if self.phase == "show" and not self.hover:
                self.remaining -= dt
                lit = math.ceil(BLOCKS * max(self.remaining, 0.0) / self.total)
                if lit != self.lit:
                    for i, block in enumerate(self.blocks):
                        self.c.itemconfigure(block, fill=TEAL if i < lit else EMPTY)
                    self.lit = lit
                if self.remaining <= 0:
                    self.leave()
        if abs(alpha - self.alpha) > 1e-3:
            self.root.attributes("-alpha", alpha)
        self.alpha = alpha
        b = now - self.hop_t0  # the smiley hops when the card lands, and again after "I'm back"
        dy = -abs(math.sin(b * math.pi * 2.6)) * 7 * (1 - b / 1.2) if b < 1.2 else 0.0
        if dy != self.face_dy:
            self.c.move("face", 0, dy - self.face_dy)
            self.face_dy = dy
        self.root.after(16, self.tick)

    def press(self):
        if self.phase != "show" or self.pressed:
            return
        self.pressed = True
        self.c.move("btn", 3, 3)  # pushed down into its shadow
        self.c.itemconfigure(self.btn_shadow, state="hidden")
        self.root.after(120, self.thanks)

    def thanks(self):
        if self.phase != "show":
            return
        self.phase = "thanks"
        c = self.c
        for item, value in zip((self.label, self.title, self.body), THANKS):
            c.itemconfigure(item, text=value)
        for item in ["btn", self.btn_shadow] + self.blocks:
            c.itemconfigure(item, state="hidden")
        if self.wink:
            c.itemconfigure(self.eye_r, state="hidden")
            c.itemconfigure(self.wink, state="normal")  # ;)
        c.configure(cursor="")
        self.hop_t0 = time.monotonic()
        self.root.after(1500, self.leave)

    def leave(self):
        if self.phase != "out":
            self.phase, self.out_from, self.t_out = "out", self.alpha, time.monotonic()

    def run(self):
        self.root.mainloop()

    def layout_problems(self):
        """Everything inside the card and in its own place, nothing overlapping (used by --check)."""
        c = self.c
        c.update_idletasks()
        box = {name: c.bbox(item) for name, item in (
            ("label", self.label), ("stripes", self.stripes[0]), ("close box", self.closebox), ("smiley", "face"),
            ("title", self.title), ("text", self.body), ("button", self.btn), ("button label", self.btn_label),
            ("countdown", self.blocks[0]), ("countdown end", self.blocks[-1]))}
        problems = [f"{n} sticks out of the card {b}" for n, b in box.items()
                    if b[0] < -1 or b[1] < -1 or b[2] > W + 1 or b[3] > self.H + 1]
        problems += [f"{n} is not inside the title bar" for n in ("label", "stripes", "close box")
                     if box[n][1] < 0 or box[n][3] > HEADER + 1]
        problems += [f"{n} reaches into the title bar" for n in ("smiley", "title") if box[n][1] < HEADER]
        for a, b in (("label", "stripes"), ("stripes", "close box"), ("title", "text"), ("text", "button"),
                     ("button", "countdown"), ("smiley", "title"), ("smiley", "text"), ("smiley", "button"),
                     ("smiley", "countdown")):
            p, q = box[a], box[b]
            if p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]:
                problems.append(f"{a} overlaps {b}")
        if box["button label"][0] < box["button"][0] or box["button label"][2] > box["button"][2]:
            problems.append("button label wider than the button")
        if box["stripes"][2] - box["stripes"][0] < 24:
            problems.append("no room for the stripes")
        lines = {n: round((box[k][3] - box[k][1]) / self.fonts[n].metrics("linespace"))
                 for n, k in (("title", "title"), ("body", "text"))}
        return problems, lines


def check_all(extra=()):
    """Every message with both pictures, plus any (title, text) in `extra`."""
    bad = 0
    for kind, messages in MESSAGES.items():
        for icon, (title, text) in [(i, m) for i in ("eye", "smiley") for m in messages + [THANKS[1:]] + list(extra)]:
            card = Card(title, text, check=True, icon=icon)
            problems, lines = card.layout_problems()
            print(f"{'ok ' if not problems else 'BAD'} {icon:<6} {title:<30} card {W}x{card.H}, "
                  f"title {lines['title']} line(s), text {lines['body']} line(s)"
                  + "".join(f"\n      - {p}" for p in problems))
            bad += bool(problems)
            card.root.destroy()
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Show a friendly retro pop-up card.")
    p.add_argument("--kind", default="scrolling", choices=sorted(MESSAGES), help="pick a random message of this kind")
    p.add_argument("--title", help="headline (default: from the random message)")
    p.add_argument("--text", help="smaller text under the headline (default: from the random message)")
    p.add_argument("--seconds", type=float, default=12, help="close by itself after this long (hover pauses it)")
    p.add_argument("--position", choices=["center", "top-right"], default="center")
    p.add_argument("--icon", choices=["eye", "smiley"], default="eye",
                   help="the picture on the card: the lecture tool's LED eye (default) or the drawn smiley")
    p.add_argument("--voice", help="an audio file (mp3 or wav) to play instead of the chime")
    p.add_argument("--no-sound", action="store_true", help="no chime and no voice")
    p.add_argument("--check", action="store_true", help="test: build every message off-screen and check the layout "
                                                        "(with --title, that title too)")
    a = p.parse_args(argv)
    title, text = random.choice(MESSAGES[a.kind])
    if a.check:
        sys.exit(check_all([(a.title, a.text or text)] if a.title else ()))
    Card(a.title or title, a.text or text, a.seconds, not a.no_sound, a.position, icon=a.icon, voice=a.voice).run()


if __name__ == "__main__":
    main()
