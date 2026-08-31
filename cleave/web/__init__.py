"""HTTP layer: job registry, upload handling, routes and search.

Split out of a single 540-line ``app.py`` that held the job model, disk
persistence, the whole ingest→chunk→enrich orchestration and every route. The
orchestration now lives in ``cleave.pipeline``, which imports nothing from
FastAPI, so the layering the architecture doc describes is real rather than
aspirational.
"""
