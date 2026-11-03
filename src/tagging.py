"""Smart Tagging intervals in video frames (thin wrapper around starter/load_smart_tagging.py)."""

import os
import sys

from src.atd import ROOT

sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402


def load():
    """[{code, start_frame, end_frame, tags}]; end_frame is exclusive."""
    return load_smart_tagging.load()
