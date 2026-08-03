"""Compatibility wrapper for the retired exploratory UBOS processor.

The authoritative pipeline is ``build_final_dataset.py``. This entrypoint is
kept so old commands fail clearly instead of silently producing a mismatched
district-level table.
"""
from __future__ import annotations

raise SystemExit(
    "The legacy UBOS processor is exploratory only. Use "
    "PYTHONPATH=src python scripts/build_final_dataset.py."
)
