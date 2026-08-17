#!/usr/bin/env python3
"""Run the homogeneous seasonal multi-crop authoritative analysis."""

from __future__ import annotations

import sys

from run_final_analysis import main

if __name__ == "__main__":
    raise SystemExit(
        main(["--dataset", "multi_crop_seasonal", "--quick", *sys.argv[1:]])
    )
