"""Model-expected macroF1 of a submission, using the HMM's own posteriors as soft truth.

    python -m src.abstain_report submissions/v4.csv cache/exp/with_x.csv

No labels. For every graded frame the expected confusion row is the posterior over (H, A, D),
added to the column of the predicted label (or the X column). This is self-referential -- it
trusts the model's confidence -- so it only answers "if the posteriors were calibrated, would
these X's help?", which is the question the abstention rule is built on.
"""

import sys

import numpy as np

from src import atd, context, features, model, proxy_eval, run
from games import GAME, half_video_blocks


def posteriors_all():
    a = run.load_tracking()
    f = features.compute(a)
    n = len(a["bx"])
    team_llr = model.team_evidence(f, model.CONFIG)
    sp_dead, sp_team = run.set_piece_evidence(n)
    kickoff = model.kickoff_formation(a, f) if model.CONFIG["kickoff_nats"] else None
    dead = model.dead_evidence(f, sp_dead, kickoff)
    extra = model.CONFIG["sp_team_nats"] * sp_team
    if model.CONFIG["context"]:
        extra = extra + model.CONFIG["context_scale"] * context.context_llr(a, f)
    still = model.ball_still(f)
    if model.CONFIG["hold_nats"]:
        still = still | model.throw_hold(f)
    if kickoff is not None:
        still = still | kickoff
    still = still.astype(np.float64)
    k = run.calibrate(team_llr, dead, n, extra, still=still)
    E = model.emissions(team_llr, dead, k, extra, still)
    post = np.full((n, 3), np.nan)
    logT = model.transition_logs()
    for _, first, last in half_video_blocks(GAME):
        post[first:last + 1] = model.posteriors(E[first:last + 1], logT)
    return post


def expected_macro_f1(labels, post, graded):
    col = {"H": 0, "A": 1, "D": 2, "X": 3}
    cm = np.zeros((3, 4))
    cols = np.array([col[c] for c in labels[graded]])
    p = post[graded]
    for c in range(4):
        cm[:, c] = p[cols == c].sum(axis=0)
    f1 = []
    for k in range(3):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)), [round(x, 4) for x in f1]


if __name__ == "__main__":
    post = posteriors_all()
    for path in sys.argv[1:]:
        lab = proxy_eval.read(path)
        g = proxy_eval.graded_mask(len(lab))
        m, per = expected_macro_f1(lab, post, g)
        print(path, "expected macroF1 {:.4f}  per class H/A/D {}".format(m, per))
