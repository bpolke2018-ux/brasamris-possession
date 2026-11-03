"""Build submissions/v1.csv from the provided data. Deterministic: no randomness anywhere.

    python -m src.run
"""

import argparse
import os

import numpy as np

from src import atd, features, model, tagging
from games import GAME, half_video_blocks

OUT = os.path.join(atd.ROOT, "submissions", "v1.csv")


def set_piece_mask(n):
    mask = np.zeros(n, bool)
    for e in tagging.load():
        if (e["code"] or "").strip().lower() == "set pieces":
            mask[max(e["start_frame"], 0):max(e["end_frame"], 0)] = True
    return mask


def possessions(labels):
    """(count, mean duration s) of maximal H/A runs, per half, graded frames only."""
    lengths = []
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        start = 0
        for i in range(1, len(seg) + 1):
            if i == len(seg) or seg[i] != seg[start]:
                if seg[start] in ("H", "A"):
                    lengths.append(i - start)
                start = i
    return len(lengths), (np.mean(lengths) / features.FPS if lengths else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--k-team", type=float, default=model.K_TEAM,
                        help="team-evidence weight (nats/frame); see model.py")
    args = parser.parse_args()
    model.K_TEAM = args.k_team

    a = atd.load()
    f = features.compute(a)
    n = len(a["bx"])
    E = model.emissions(f, set_piece_mask(n))
    logT = model.transition_logs()

    labels = np.full(n, "X", dtype="<U1")      # ungraded frames: never scored
    for _, first, last in half_video_blocks(GAME):
        labels[first:last + 1] = model.LABELS[model.viterbi(E[first:last + 1], logT)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    count, mean_s = possessions(labels)
    graded = labels != "X"
    print("wrote {}".format(os.path.relpath(args.out, atd.ROOT)))
    print("graded frames: H {:.1f}%  A {:.1f}%  D {:.1f}%".format(
        *(100 * np.mean(labels[graded] == s) for s in "HAD")))
    print("possessions (full match): {}  mean duration {:.1f} s".format(count, mean_s))


if __name__ == "__main__":
    main()
