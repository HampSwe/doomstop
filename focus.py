"""The productivity tool: the dashboard, plus a friendly pop-up card when you doomscroll.

    python focus.py                 # finds the LogiInsight colour and infrared cameras
    python focus.py --device 1      # or name the colour camera index

This is dashboard.py with --popup switched on, so everything dashboard.py
shows stays on screen: both streams, the attention score and the signals. When
the attention score stays below 0.5 for 2 seconds, a card drops into the
top-right corner: the lecture tool's LED eye, DOOMSCROLL DETECTED and "Get back
to work." A voice says "doomscroll detected" (sound/doomscroll_detected.mp3);
there is no other sound. The card closes itself
after 12 seconds, or at once with I'M BACK! or the x box, and the next card
comes 30 seconds after the last one at the earliest.

Every dashboard.py option works here too; the ones for the card:

    --popup-below 0.4        attention below this counts as scrolling (default 0.5)
    --popup-after 4          ... for this many seconds before the card appears (default 2)
    --popup-cooldown 60      seconds from one card to the next (default 30)
    --popup-position center  the middle of the screen instead of the top-right corner
    --no-popup-sound         no voice

Nothing leaves this computer. The other main program, for a talk, is
call/score_sender.py, which sends the score to the presenter (see DEMO.md).

Needs a fitted model (models/attention_model.json): without one there is no
attention score, and so no card.
"""

import sys

import dashboard

if __name__ == "__main__":
    if "--popup" not in sys.argv:
        sys.argv.insert(1, "--popup")
    dashboard.main(__doc__)
