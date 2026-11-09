"""Paired bootstrap of score gaps on the Metrica games (aggregates only).

    python -m src.other_games_boot v7 no_still no_context ...     # first tag = reference

Reads the labels saved by `python -m src.other_games --save <tag>`. For each draw it resamples
the ground-truth possessions (spellF1) or ground-truth segments (macroF1) of each game with
replacement, the SAME picks for every version, so the interval is for the gap itself, not two
separate intervals. This mirrors possession_metrics.bootstrap_spell_f1 / bootstrap_macro_f1 (each
version's predicted-possession count stays fixed within a draw). Games are averaged with equal
weight. Reports the mean gap over the three games and its 5-95% interval.
"""

import os
import sys

import numpy as np

from src import other_games  # noqa: F401  (puts the official scorer on sys.path)
from src.atd import CACHE_DIR
import possession_metrics as pm

DIR = os.path.join(CACHE_DIR, "other_games")
GAMES = (1, 2, 3)
DRAWS = 2000
SEED = 0


def _prep(gt, pred):
    runs = pm.valid_runs(gt)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gs), t) for t in pm.SPELL_IOUS])
    segs = pm.segments(gt, runs)
    seg_gt = np.array([c for _, _, c in segs], np.int64)
    contrib = np.zeros((len(segs), pm.N_LABELS), np.int64)
    for i, (s, e, _) in enumerate(segs):
        contrib[i] = np.bincount(pred[s:e].astype(np.int64), minlength=pm.N_LABELS)
    return flags, len(ps), seg_gt, contrib


def _spell(flags, n_pred, picks):
    n_gt = len(picks)
    f1s = []
    for row in flags:
        tp = int(row[picks].sum())
        fp, fn = max(0, n_pred - tp), n_gt - tp
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def _macro(seg_gt, contrib, picks):
    cm3 = np.zeros((pm.N_SCORED, pm.N_LABELS), np.int64)
    np.add.at(cm3, seg_gt[picks], contrib[picks])
    return pm.macro_f1(cm3)


def main(tags):
    data = {}
    for g in GAMES:
        gt = pm.label_codes(np.load(os.path.join(DIR, "gt_g{}.npy".format(g))))
        data[g] = {t: _prep(gt, pm.label_codes(np.load(os.path.join(DIR, "{}_g{}.npy".format(t, g)))))
                   for t in tags}
    rng = np.random.RandomState(SEED)
    spell = np.zeros((DRAWS, len(tags)))
    macro = np.zeros((DRAWS, len(tags)))
    point_s = np.zeros(len(tags))
    point_m = np.zeros(len(tags))
    for g in GAMES:
        ref = data[g][tags[0]]
        n_gt, n_seg = ref[0].shape[1], len(ref[2])
        for j, t in enumerate(tags):
            fl, npred, sg, co = data[g][t]
            point_s[j] += _spell(fl, npred, np.arange(n_gt)) / len(GAMES)
            point_m[j] += _macro(sg, co, np.arange(n_seg)) / len(GAMES)
        for d in range(DRAWS):
            ps, pg = rng.randint(0, n_gt, n_gt), rng.randint(0, n_seg, n_seg)
            for j, t in enumerate(tags):
                fl, npred, sg, co = data[g][t]
                spell[d, j] += _spell(fl, npred, ps) / len(GAMES)
                macro[d, j] += _macro(sg, co, pg) / len(GAMES)
    print("reference: {}  mean over games {}: spellF1 {:.3f}  macroF1 {:.3f}  ({} draws)".format(
        tags[0], GAMES, point_s[0], point_m[0], DRAWS))
    print("| version | spellF1 gap vs ref | 5-95% | macroF1 gap vs ref | 5-95% |")
    print("|---|---|---|---|---|")
    for j, t in enumerate(tags[1:], 1):
        ds, dm = spell[:, j] - spell[:, 0], macro[:, j] - macro[:, 0]
        print("| {} | {:+.3f} | {:+.3f} .. {:+.3f} | {:+.3f} | {:+.3f} .. {:+.3f} |".format(
            t, point_s[j] - point_s[0], *np.percentile(ds, [5, 95]),
            point_m[j] - point_m[0], *np.percentile(dm, [5, 95])))


if __name__ == "__main__":
    main(sys.argv[1:])
