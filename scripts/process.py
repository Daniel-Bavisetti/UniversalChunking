"""Process a video into Knowledge Units from the command line.

Usage:
    python scripts/process.py data/fixture.mp4
    python scripts/process.py talk.mp4 --id my_talk --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vke.config import CONFIGS, ensure_dirs  # noqa: E402
from vke.pipeline import process_video  # noqa: E402
from vke.store import VideoStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a video into Knowledge Units")
    parser.add_argument("video", type=Path)
    parser.add_argument("--id", dest="video_id", default=None,
                        help="video id (default: the filename stem)")
    parser.add_argument("--force", action="store_true",
                        help="ignore the cached extraction and redo it")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"not found: {args.video}")
        return 2

    ensure_dirs()
    video_id = args.video_id or args.video.stem
    store = VideoStore(video_id)

    # The store owns a copy so the API can stream it regardless of where the
    # original lives.
    dest = store.dir / args.video.name
    if not dest.exists() or dest.stat().st_size != args.video.stat().st_size:
        shutil.copy2(args.video, dest)
        print(f"staged -> {dest}")

    def progress(stage: str, percent: int, message: str) -> None:
        print(f"  {percent:3d}%  {message}")

    meta, units_by_config, traces = process_video(
        dest, video_id, progress=progress, force=args.force
    )

    print(f"\n{meta.filename}  {meta.width}x{meta.height}  {meta.duration:.1f}s")
    print(f"\n{'config':<18}{'units':>7}{'mean':>9}   boundaries")
    for key, cfg in CONFIGS.items():
        units = units_by_config[key]
        mean = sum(u.span.duration for u in units) / max(len(units), 1)
        edges = [f"{u.span.start:.1f}" for u in units[1:]]
        shown = ", ".join(edges[:8]) + ("..." if len(edges) > 8 else "")
        print(f"{cfg.label:<18}{len(units):>7}{mean:>8.1f}s   {shown}")

    total = sum(t.seconds for t in traces)
    print(f"\ntotal {total:.1f}s  ({meta.duration/max(total,1e-9):.1f}x realtime)")
    for t in traces:
        print(f"  {t.stage:<12}{t.seconds:>7.2f}s  {t.detail}")
    print(f"\nstored in {store.dir}")
    print(f"start the UI with:  uvicorn vke.api:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
