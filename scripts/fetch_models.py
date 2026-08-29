"""Pre-stage every enrichment model so a demo never downloads anything live.

    python scripts/fetch_models.py

Object detection weights (~9MB) come from the HF hub on first use and are then
cached. That is fine on a laptop and a bad idea on a stage, so run this during
setup. OCR weights ship inside the rapidocr wheel and need no download.

Exit code is 0 even when a model is unavailable: the pipeline is designed to run
without them, and this script reports rather than enforces.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vke.detect import warmup  # noqa: E402


def main() -> int:
    print("staging enrichment models...")
    state = warmup()
    ready = [k for k, v in state.items() if v == "ready"]
    missing = {k: v for k, v in state.items() if v != "ready"}

    print()
    if ready:
        print(f"ready: {', '.join(sorted(ready))}")
    for name, reason in sorted(missing.items()):
        print(f"unavailable: {name} - {reason}")
        print(f"  the pipeline will still run; units will record this reason "
              f"instead of a silent empty list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
