"""Shared pytest setup.

LightGBM (recall model tests) and PyTorch (Waymo tests) each bring their own OpenMP runtime on
macOS; with both loaded in one process, multi-threaded OpenMP segfaults. Tests are tiny, so one
thread costs nothing. Must be set before either library is imported.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
