"""Aggregate diagnostics of SET PIECES windows vs a submission (no labels, no rows printed).

    python -m src.setpiece_report submissions/v3.csv

For 5 s bins across the window: share of frames predicted D, and among live frames the share
predicted for the tagged team (team-tagged windows only).
"""

import sys

import numpy as np

from src import proxy_eval, tagging

FPS = 25
TEAM = {"barcelona": "H", "atletic club": "A"}

if __name__ == "__main__":
    labels = proxy_eval.read(sys.argv[1])
    events = [e for e in tagging.load() if (e["code"] or "").strip().lower() == "set pieces"]
    bins = range(-10, 40, 5)
    print("bin_s     D_share  tagged_team_share  (n windows={})".format(len(events)))
    for b in bins:
        d, live, agree = 0, 0, 0
        for e in events:
            s = e["start_frame"] + b * FPS
            seg = labels[max(s, 0):max(s + 5 * FPS, 0)]
            seg = seg[seg != "X"]
            d += np.sum(seg == "D")
            team = TEAM.get((e["tags"].get("Team") or "").strip().lower())
            if team:
                lv = seg[seg != "D"]
                live += len(lv)
                agree += np.sum(lv == team)
        tot = d + sum(np.sum((labels[max(e["start_frame"] + b * FPS, 0):max(e["start_frame"] + (b + 5) * FPS, 0)] != "X")
                             & (labels[max(e["start_frame"] + b * FPS, 0):max(e["start_frame"] + (b + 5) * FPS, 0)] != "D"))
                      for e in events)
        print("{:>3}..{:<3}  {:.2f}     {:.2f}".format(b, b + 5, d / max(tot, 1), agree / max(live, 1)))
    types = {}
    for e in events:
        t = e["tags"].get("Type") or "?"
        types.setdefault(t, {"teamed": 0, "n": 0})
        types[t]["n"] += 1
        types[t]["teamed"] += (e["tags"].get("Team") or "").strip().lower() in TEAM
    print("team tag present by type:", types)
