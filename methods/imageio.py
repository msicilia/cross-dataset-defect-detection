"""Parallel image loading for the detectors.

Decoding and resizing run on a thread pool (PIL releases the GIL for both), so
the accelerator is not left waiting on one CPU core. Results keep the input
order and each image is processed exactly as in a sequential loop.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)


def load_all(load, paths) -> list:
    """[load(p) for p in paths], computed in parallel."""
    return list(_POOL.map(load, paths))
