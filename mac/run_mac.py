#!/usr/bin/env python3
"""MAC ONLY. Run dashboard.py or collect.py on macOS with the LogiInsight colour camera.

    sudo .venv/bin/python mac/run_mac.py dashboard [dashboard.py's options]
    sudo .venv/bin/python mac/run_mac.py collect   [collect.py's options]

It swaps engine.Engine for mac/engine_mac.MacEngine, then runs the normal
script, so dashboard.py and collect.py themselves are unchanged. See
mac/README.md for why this is needed and what differs. Windows and Linux: run
dashboard.py / collect.py directly.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "logiinsight_api"))
sys.path.insert(0, str(ROOT / "detector"))      # dashboard.py, collect.py, engine.py ...

SCRIPTS = {"dashboard": "dashboard.py", "collect": "collect.py"}


def main():
    if sys.platform != "darwin":
        sys.exit("mac/run_mac.py is for macOS only. On Windows/Linux run detector/dashboard.py or detector/collect.py directly.")
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        sys.exit(__doc__)
    script = ROOT / "detector" / SCRIPTS[sys.argv[1]]

    import engine
    from mac.engine_mac import MacEngine
    engine.Engine = MacEngine          # dashboard.py / collect.py import Engine from here

    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
