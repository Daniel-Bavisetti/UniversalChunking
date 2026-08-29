"""Per-video artifact storage.

One directory per video holding plain JSON. That is the whole storage design: at
a few hundred units per video a database buys nothing, and a directory you can
open in an editor is worth a lot when debugging on a deadline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import STORE_DIR
from .schemas import (
    FrameFeature,
    KnowledgeUnit,
    SceneCut,
    SpeakerTurn,
    StageTrace,
    Utterance,
    VideoMeta,
    VisualObservation,
)


class VideoStore:
    def __init__(self, video_id: str, root: Path | None = None) -> None:
        self.video_id = video_id
        self.dir = (root or STORE_DIR) / video_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "keyframes").mkdir(exist_ok=True)

    # --- paths ------------------------------------------------------------- #
    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def extraction_path(self) -> Path:
        return self.dir / "extraction.json"

    @property
    def observations_path(self) -> Path:
        return self.dir / "observations.json"

    @property
    def units_path(self) -> Path:
        return self.dir / "units.json"

    @property
    def curves_path(self) -> Path:
        return self.dir / "curves.json"

    @property
    def graph_path(self) -> Path:
        return self.dir / "graph.json"

    @property
    def traces_path(self) -> Path:
        return self.dir / "traces.json"

    def keyframe_path(self, unit_id: str) -> Path:
        return self.dir / "keyframes" / f"{unit_id}.jpg"

    def video_path(self) -> Path | None:
        meta = self.load_meta()
        if meta is None:
            return None
        candidate = self.dir / meta.filename
        return candidate if candidate.exists() else None

    # --- io ---------------------------------------------------------------- #
    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic; a half-written artifact is worse than none

    @staticmethod
    def _read(path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    # --- meta -------------------------------------------------------------- #
    def save_meta(self, meta: VideoMeta) -> None:
        self._write(self.meta_path, meta.model_dump())

    def load_meta(self) -> VideoMeta | None:
        blob = self._read(self.meta_path)
        return VideoMeta(**blob) if blob else None

    # --- extraction (the expensive stage; cached) --------------------------- #
    def save_extraction(
        self,
        utterances: list[Utterance],
        features: list[FrameFeature],
        cuts: list[SceneCut],
        providers: dict[str, str],
        turns: list[SpeakerTurn] | None = None,
    ) -> None:
        self._write(self.extraction_path, {
            "utterances": [u.model_dump() for u in utterances],
            "features": [f.model_dump() for f in features],
            "cuts": [c.model_dump() for c in cuts],
            "turns": [t.model_dump() for t in (turns or [])],
            "providers": providers,
        })

    def load_extraction(self) -> tuple[
        list[Utterance], list[FrameFeature], list[SceneCut],
        dict[str, str], list[SpeakerTurn],
    ] | None:
        blob = self._read(self.extraction_path)
        if not blob:
            return None
        return (
            [Utterance(**u) for u in blob["utterances"]],
            [FrameFeature(**f) for f in blob["features"]],
            [SceneCut(**c) for c in blob["cuts"]],
            blob.get("providers", {}),
            [SpeakerTurn(**t) for t in blob.get("turns", [])],
        )

    # --- observations (enrichment; cached, and independent of chunk config) -- #
    def save_observations(
        self, observations: list[VisualObservation], status: dict[str, str]
    ) -> None:
        self._write(self.observations_path, {
            "observations": [o.model_dump() for o in observations],
            "status": status,
        })

    def load_observations(self) -> tuple[list[VisualObservation], dict[str, str]] | None:
        blob = self._read(self.observations_path)
        if not blob:
            return None
        return (
            [VisualObservation(**o) for o in blob.get("observations", [])],
            blob.get("status", {}),
        )

    # --- graph + traces ---------------------------------------------------- #
    def save_graph(self, payload: dict[str, Any]) -> None:
        self._write(self.graph_path, payload)

    def load_graph(self) -> dict[str, Any] | None:
        return self._read(self.graph_path)

    def save_traces(self, traces: list[StageTrace]) -> None:
        self._write(self.traces_path, [t.model_dump() for t in traces])

    def load_traces(self) -> list[StageTrace]:
        blob = self._read(self.traces_path)
        return [StageTrace(**t) for t in blob] if blob else []

    # --- units, keyed by config -------------------------------------------- #
    def save_units(self, units_by_config: dict[str, list[KnowledgeUnit]]) -> None:
        self._write(self.units_path, {
            key: [u.model_dump() for u in units]
            for key, units in units_by_config.items()
        })

    def load_units(self) -> dict[str, list[KnowledgeUnit]]:
        blob = self._read(self.units_path)
        if not blob:
            return {}
        return {
            key: [KnowledgeUnit(**u) for u in units] for key, units in blob.items()
        }

    # --- signal curves, for the UI strip ----------------------------------- #
    def save_curves(self, payload: dict[str, Any]) -> None:
        self._write(self.curves_path, payload)

    def load_curves(self) -> dict[str, Any] | None:
        return self._read(self.curves_path)

    # --- lifecycle --------------------------------------------------------- #
    @property
    def is_processed(self) -> bool:
        return self.units_path.exists() and self.meta_path.exists()

    def delete(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def list_videos(root: Path | None = None) -> list[VideoMeta]:
    base = root or STORE_DIR
    if not base.exists():
        return []
    out: list[VideoMeta] = []
    for child in sorted(base.iterdir()):
        if child.is_dir():
            meta = VideoStore(child.name, root=base).load_meta()
            if meta is not None:
                out.append(meta)
    return out
