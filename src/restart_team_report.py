"""Held-out check of the team right after team-tagged restarts (aggregates only).

    python -m src.restart_team_report submissions/v4.csv [more.csv ...]

Phase agreement (src/proxy_eval.py definition: Smart Tagging attacking/defending phases, never fed
to the model) restricted to the first 20 s after each team-tagged SET PIECES start, and the share
of those live frames given to the tagged restarting team (the latter is NOT independent when the
tag is used by the model).
"""

import sys

import numpy as np

from src import proxy_eval, tagging

FPS = 25
WINDOW_S = 20


def phase_expected(events, n):
    exp = np.full(n, "", dtype="<U1")
    for e in events:
        code = (e["code"] or "").strip().lower()
        team = proxy_eval.TEAM.get((e["tags"].get("Team") or "").strip().lower())
        if team is None or code not in proxy_eval.ATTACK | proxy_eval.DEFEND:
            continue
        want = team if code in proxy_eval.ATTACK else proxy_eval.OTHER[team]
        sl = slice(max(e["start_frame"], 0), max(e["end_frame"], 0))
        exp[sl] = np.where(exp[sl] == "", want, exp[sl])
    return exp


if __name__ == "__main__":
    events = tagging.load()
    for path in sys.argv[1:]:
        lab = proxy_eval.fill_x(proxy_eval.read(path))
        n = len(lab)
        g = proxy_eval.graded_mask(n)
        exp = phase_expected(events, n)
        win = np.zeros(n, bool)
        tagged = np.full(n, "", dtype="<U1")
        for e in events:
            if (e["code"] or "").strip().lower() != "set pieces":
                continue
            t = proxy_eval.TEAM.get((e["tags"].get("Team") or "").strip().lower())
            if t:
                s = max(e["start_frame"], 0)
                win[s:s + WINDOW_S * FPS] = True
                tagged[s:s + WINDOW_S * FPS] = t
        live = g & win & np.isin(lab, ["H", "A"])
        ph = live & (exp != "")
        print(path, "phase_agree_after_restart {:.3f} (n={})  tagged_team_share {:.3f}  live_share {:.3f}".format(
            np.mean(lab[ph] == exp[ph]), int(ph.sum()), np.mean(lab[live] == tagged[live]),
            live.sum() / max((g & win).sum(), 1)))
