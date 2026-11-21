# message

Shows a pop-up when another part of the project sends a signal.

| signal | pop-up |
|---|---|
| `stop scrolling` (or `doomscrolling`) | **Scrolling detected** |

Case and punctuation don't matter (`"Stop scrolling!"` works too). Unknown signals are ignored.

## Run

```bash
python message/listener.py                  # keep this running in its own terminal
python message/send.py "stop scrolling"     # from another terminal: the pop-up appears
```

Options: `--style banner` (macOS notification instead of a window), `--seconds 10` (the window closes
itself after this long), `--cooldown 20` (at most one pop-up of the same kind per 20 s),
`--dry-run` (print instead of pop up), `--port 8766`.

## Send from your own detector

No import needed. Add this where your code decides the user is scrolling:

```python
import socket
socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"stop scrolling", ("127.0.0.1", 8766))
```

It returns immediately and doesn't fail if the listener isn't running. Sending every frame is fine:
the listener's cooldown stops the pop-ups from spamming.

To add a new signal, add a line to `SIGNALS` in `listener.py`.
