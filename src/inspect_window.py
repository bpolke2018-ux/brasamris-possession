"""Plot one window of the match for a PERSON to look at. Shows match data.

    python -m src.inspect_window --half 1 --start 2 --end 4
    python -m src.inspect_window --half 2 --start 30 --end 31.5 --sub cache/exp/my_try.csv

FOR BRANDON'S EYES ONLY. The image shows tracking data, so it is saved under outputs/
(gitignored) and must not be committed, shared, or shown to any AI assistant. The development
rules of this repo also say: never pick a version because it looks better on the public
5-minute ground truth. The ground-truth band is drawn only so you can understand what the labels
mean, not to tune against.

--start/--end are minutes from the start of the half's graded play (half 1 graded play begins at
video frame 11185; the public ground truth covers half 1, minutes 0-5).

Panels, top to bottom, sharing one time axis (minutes into the half):
  1. our labels (H blue, A red, D grey)       2. public ground truth, when the window is inside it
  3. distance from the ball to the nearest home and nearest away player (m)
  4. ball speed (m/s), with the 8 m/s in-flight gate and the 1.5 m/s "still" line
  5. ball seen by the tracker (1) or missing (0)
  6. dead-ball evidence (nats/frame; > 0 pushes towards D), and the team evidence k*llr/2
  7. Smart Tagging SET PIECES windows
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src import atd, context, features, model, proxy_eval, run, tagging
from games import GAME, half_video_blocks

FPS = features.FPS
COLOURS = {"H": "#2b6cb0", "A": "#c53030", "D": "#a0aec0", "X": "#ffffff"}
GT_CSV = os.path.join(atd.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
GT_META = os.path.join(atd.ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public_meta.json")
OUT_DIR = os.path.join(atd.ROOT, "outputs")


def band(ax, t, labels, title):
    """Coloured strip, one colour per label run."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=COLOURS.get(labels[s], "#ffffff"), lw=0)
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--half", type=int, choices=(1, 2), required=True)
    parser.add_argument("--start", type=float, required=True, help="minutes into the half")
    parser.add_argument("--end", type=float, required=True, help="minutes into the half")
    parser.add_argument("--sub", default=run.OUT, help="submission CSV to show")
    parser.add_argument("--k-team", type=float, default=1.815,
                        help="team-evidence weight to draw (src.run prints it; v6 = 1.815)")
    parser.add_argument("--no-open", action="store_true", help="save only, do not open")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    first = [f for h, f, _ in half_video_blocks(GAME) if h == args.half][0]
    lo, hi = first + int(args.start * 60 * FPS), first + int(args.end * 60 * FPS)
    if hi <= lo:
        raise SystemExit("--end must be after --start")

    a = atd.load()
    f = features.compute(a)
    n = len(a["bx"])
    sl = slice(lo, hi)
    t = (np.arange(lo, hi) - first) / FPS / 60

    labels = proxy_eval.read(args.sub)[sl]
    sp_dead, _ = run.set_piece_evidence(n)
    dead = model.dead_evidence(f, sp_dead)[sl]
    team = model.team_evidence(f, model.CONFIG)
    team_total = args.k_team * team
    if model.CONFIG["context"]:
        team_total = team_total + model.CONFIG["context_scale"] * context.context_llr(a, f)

    gt = None
    with open(GT_META) as handle:
        covered = json.load(handle)["covered_video_frames"]
    if any(c0 <= lo and hi <= c1 + 1 for c0, c1 in covered):
        gt = pd.read_csv(GT_CSV, dtype={"label": str})["label"].to_numpy().astype("<U1")[sl]

    fig, axes = plt.subplots(7, 1, figsize=(14, 11), sharex=True,
                             gridspec_kw={"height_ratios": [0.5, 0.5, 1.2, 1.2, 0.5, 1.2, 0.5]})
    band(axes[0], t, labels, "ours\n" + os.path.basename(args.sub))
    if gt is not None:
        band(axes[1], t, gt, "public GT")
    else:
        axes[1].text(0.5, 0.5, "outside the public ground-truth window", ha="center",
                     va="center", transform=axes[1].transAxes)
        axes[1].set_yticks([])

    ax = axes[2]
    ax.plot(t, f["d_home"][sl], color=COLOURS["H"], lw=0.8, label="nearest home")
    ax.plot(t, f["d_away"][sl], color=COLOURS["A"], lw=0.8, label="nearest away")
    ax.axhline(model.CONFIG["control_dist_m"], color="k", ls=":", lw=0.8)
    ax.set_ylim(0, 15)
    ax.set_ylabel("distance\nto ball (m)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    ax.plot(t, f["ball_speed"][sl], color="k", lw=0.8)
    ax.axhline(model.CONFIG["max_control_ball_speed"], color="#dd6b20", ls="--", lw=0.8,
               label="in-flight gate")
    ax.axhline(model.CONFIG["still_speed"], color="#38a169", ls="--", lw=0.8, label="still")
    ax.set_ylim(0, 30)
    ax.set_ylabel("ball speed\n(m/s)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[4]
    ax.fill_between(t, f["ball_seen"][sl].astype(float), step="mid", color="#4a5568")
    ax.set_yticks([0, 1])
    ax.set_ylabel("ball seen")

    ax = axes[5]
    ax.plot(t, dead, color="#4a5568", lw=0.8, label="dead-ball evidence (D)")
    ax.plot(t, team_total[sl] / 2, color="#805ad5", lw=0.8, label="team evidence /2 (+H, -A)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_ylabel("evidence\n(nats/frame)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[6]
    for e in tagging.load():
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        s, en = e["start_frame"], e["end_frame"]
        if en < lo or s > hi:
            continue
        ax.axvspan((max(s, lo) - first) / FPS / 60, (min(en, hi) - first) / FPS / 60,
                   color="#ecc94b", alpha=0.6, lw=0)
        ax.text((max(s, lo) - first) / FPS / 60, 0.5, " " + (e["tags"].get("Type") or "set piece")
                + (" / " + e["tags"]["Team"] if e["tags"].get("Team") else ""), fontsize=7,
                va="center")
    ax.set_yticks([])
    ax.set_ylabel("set pieces")
    ax.set_xlabel("minutes into half {}".format(args.half))

    axes[0].legend(handles=[Patch(color=COLOURS[k], label=k) for k in "HAD"], ncol=3,
                   loc="lower right", bbox_to_anchor=(1, 1), fontsize=8)
    fig.suptitle("Half {}, {:.2f}-{:.2f} min  (FOR YOUR EYES ONLY - do not share with an AI)".format(
        args.half, args.start, args.end))
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "window_h{}_{:g}-{:g}.png".format(args.half, args.start, args.end))
    fig.savefig(out, dpi=110)
    print("saved", os.path.relpath(out, atd.ROOT))
    if not args.no_open and hasattr(os, "startfile"):
        os.startfile(out)


if __name__ == "__main__":
    main()
