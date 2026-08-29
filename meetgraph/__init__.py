"""Vendored from sensein/meetgraph (Apache-2.0) — the portable audio core.

Upstream: https://github.com/sensein/meetgraph @ 1f039ad6a976
License:  Apache-2.0 (see LICENSE.md in this directory)

meetgraph is a live meeting-capture application; most of it (the PyQt UI,
SQL/Mongo mirroring, email delivery, the RDF store) is that application's
concern, not ours. Two modules are genuinely portable and are vendored
verbatim:

* ``transcribe``  — transcription engines behind one interface, with
  hardware-aware resolution: CUDA when an NVIDIA GPU is present, **MLX on
  Apple-Silicon GPUs**, CPU otherwise. The MLX path matters here: this is the
  difference between transcribing a meeting on the GPU and pinning a CPU core
  for minutes.
* ``diarize``     — speaker labelling via **Resemblyzer** voice embeddings
  (a learned speaker model that ships with the package, no token, no gated
  download) clustered online by nearest centroid. A learned voice embedding
  separates similar voices far better than hand-rolled spectral features.

Cleave uses these from ``cleave/ingest_audio.py``: timestamps come from the
whisper engines, speaker labels from ``SpeakerLabeler``, and the temporal
chunker downstream turns speaker turns into chunk boundaries.
"""

from __future__ import annotations
