# message

Shows a friendly pop-up when another part of the project sends a signal.

| signal | pop-up |
|---|---|
| `stop scrolling` (or `doomscrolling`) | a kind retro card, e.g. **Psst… look up for a sec** |

Case and punctuation don't matter (`"Stop scrolling!"` works too). Unknown signals are ignored.

The card (`popup.py`, plain tkinter) has a warm retro look: cream paper, ink outlines with a hard shadow, a
title bar with 70s stripes and a drawn smiley. It drops into the centre of the screen with the classic Mac
chime. It closes itself after 12 s (a row of blocks counts down), and hovering over it pauses the
countdown. **I'M BACK!** closes it with a wink and a thank-you, and the × box closes it right away. The text
is picked at random from a few friendly variants, never the same one twice in a row, so it doesn't feel
like nagging.

## Run

```bash
python message/listener.py                  # keep this running in its own terminal
python message/send.py "stop scrolling"     # from another terminal: the card appears
python message/popup.py                     # just preview the card
python message/popup.py --position top-right   # preview it in the corner instead of the centre
```

Listener options: `--style alert` / `--style banner` (plain macOS alert or notification instead of the
card), `--seconds 12`, `--cooldown 20` (at most one pop-up of the same kind per 20 s), `--no-sound`,
`--dry-run` (print instead of pop up), `--port 8766`.

## Send from your own detector

No import needed. Add this where your code decides the user is scrolling:

```python
import socket
socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"stop scrolling", ("127.0.0.1", 8766))
```

It returns immediately and doesn't fail if the listener isn't running. Sending every frame is fine:
the listener's cooldown stops the pop-ups from spamming.

## Change the wording or add a signal

- Messages are in `MESSAGES` at the top of `popup.py`, as `(title, text)`. Run
  `python message/popup.py --check` afterwards: it builds every card off-screen and checks that nothing
  overflows or overlaps.
- Signals are in `SIGNALS` in `listener.py`, mapping a signal to a kind of message.
