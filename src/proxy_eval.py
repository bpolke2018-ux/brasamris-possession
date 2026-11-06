"""Ground-truth-free development metric. Uses NO labels from the public ground truth.

    python -m src.proxy_eval submissions/v1.csv [more.csv ...] [--log NAME --note "..."]

(a) Phase agreement -- held out: the Smart Tagging *phase* tags (attacking and defending phases,
    each with a Team) are never fed to the model, so they act as an independent, noisy, automatic
    reference. Share of frames inside an attacking phase where the prediction equals the tagged
    team (Barcelona = H, Atletic Club = A), and the mirror for defending phases, counted over frames
    predicted H or A.
(b) Restart alignment: share of SET PIECES tag starts with a predicted dead-to-live transition
    within 2 s, plus the median signed offset (prediction minus tag). NOTE: the model uses the
    set-piece windows as weak D evidence, so (b) is not fully independent.
(c) Structure: possession count, duration percentiles, share under 2 s, share of graded frames D.
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd

from src import tagging
from src.atd import ROOT
from games import GAME, half_video_blocks

FPS = 25
TEAM = {"barcelona": "H", "atletic club": "A"}
ATTACK = {"build up", "progression", "creating chances", "attacking transition",
          "ball in final third", "ball in the box"}
DEFEND = {"defending in defensive third", "defending in middle third",
          "defending in attacking third"}
OTHER = {"H": "A", "A": "H"}
LOG = os.path.join(ROOT, "notes", "experiments.md")


def read(path):
    labels = pd.read_csv(path, dtype={"label": str})["label"].str.strip().str.upper().to_numpy()
    return labels.astype("<U1")


def graded_mask(n):
    m = np.zeros(n, bool)
    for _, first, last in half_video_blocks(GAME):
        m[first:last + 1] = True
    return m


def fill_x(labels):
    """Forward-fill X inside each half (back-fill at the head), exactly as the official scorer
    does before its temporal metrics: 'H X X H' is one possession, not two."""
    out = labels.copy()
    for _, first, last in half_video_blocks(GAME):
        w = out[first:last + 1]
        known = np.flatnonzero(w != "X")
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(w != "X", np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        out[first:last + 1] = w[carry]
    return out


def segments(labels):
    """(label, start, end_exclusive) runs within each half, graded frames only."""
    out = []
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(seg)]))
        out.extend((seg[s], first + s, first + e) for s, e in zip(starts, ends))
    return out


def evaluate(labels, events):
    n = len(labels)
    graded = graded_mask(n)
    res = {}

    expected = np.full(n, "", dtype="<U1")
    for e in events:
        code = (e["code"] or "").strip().lower()
        team = TEAM.get((e["tags"].get("Team") or "").strip().lower())
        if team is None or code not in ATTACK | DEFEND:
            continue
        want = team if code in ATTACK else OTHER[team]
        sl = slice(max(e["start_frame"], 0), max(e["end_frame"], 0))
        expected[sl] = np.where(expected[sl] == "", want, expected[sl])
    live = graded & np.isin(labels, ["H", "A"]) & (expected != "")
    res["phase_agree"] = float(np.mean(labels[live] == expected[live])) if live.any() else np.nan
    res["phase_frames"] = int(live.sum())

    segs = segments(fill_x(labels))      # structure is judged as the scorer sees it
    restarts = np.array([s for lab, s, _ in segs if lab in ("H", "A")])
    prev = {s: lab for lab, s, _ in segs}
    # a restart is an H/A segment that follows a D segment
    d_to_live = []
    for i in range(1, len(segs)):
        if segs[i][0] in ("H", "A") and segs[i - 1][0] == "D" and segs[i - 1][2] == segs[i][1]:
            d_to_live.append(segs[i][1])
    d_to_live = np.array(d_to_live)
    starts = [e["start_frame"] for e in events
              if (e["code"] or "").strip().lower() == "set pieces" and graded[min(max(e["start_frame"], 0), n - 1)]]
    if len(starts) and len(d_to_live):
        offs = np.array([d_to_live[np.argmin(np.abs(d_to_live - s))] - s for s in starts])
        res["restart_within_2s"] = float(np.mean(np.abs(offs) <= 2 * FPS))
        res["restart_median_offset_s"] = float(np.median(offs) / FPS)
    else:
        res["restart_within_2s"], res["restart_median_offset_s"] = np.nan, np.nan

    poss = np.array([e - s for lab, s, e in segs if lab in ("H", "A")]) / FPS
    res["possessions"] = len(poss)
    res["dur_p10"], res["dur_p50"], res["dur_p90"] = (np.percentile(poss, [10, 50, 90])
                                                      if len(poss) else (np.nan,) * 3)
    res["short_lt2s"] = float(np.mean(poss < 2)) if len(poss) else np.nan
    res["share_D"] = float(np.mean(fill_x(labels)[graded] == "D"))
    res["share_X"] = float(np.mean(labels[graded] == "X"))
    return res


COLUMNS = ["phase_agree", "restart_within_2s", "restart_median_offset_s", "possessions",
           "dur_p10", "dur_p50", "dur_p90", "short_lt2s", "share_D", "share_X"]


def fmt(name, r):
    vals = []
    for c in COLUMNS:
        v = r[c]
        vals.append(str(v) if isinstance(v, int) else "{:.3f}".format(v))
    return "| {} | {} |".format(name, " | ".join(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+")
    parser.add_argument("--log", help="append the results to notes/experiments.md under this name")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    events = tagging.load()
    header = "| run | " + " | ".join(COLUMNS) + " |"
    rows = []
    for path in args.submissions:
        name = os.path.splitext(os.path.basename(path))[0]
        rows.append(fmt(name, evaluate(read(path), events)))
    print(header)
    print("|" + "---|" * (len(COLUMNS) + 1))
    print("\n".join(rows))

    if args.log:
        new = not os.path.isfile(LOG)
        with open(LOG, "a", encoding="utf-8") as handle:
            if new:
                handle.write("# Experiments (proxy metric only -- no ground truth)\n\n"
                             "Columns: see src/proxy_eval.py. phase_agree is the held-out Smart "
                             "Tagging phase agreement.\n\n")
            handle.write("## {} — {}\n\n{}\n\n{}\n{}\n{}\n\n".format(
                args.log, datetime.date.today().isoformat(), args.note, header,
                "|" + "---|" * (len(COLUMNS) + 1), "\n".join(rows)))


if __name__ == "__main__":
    main()
