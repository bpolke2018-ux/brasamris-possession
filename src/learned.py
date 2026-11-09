"""Per-frame evidence LEARNED from the three Metrica games (round 20), instead of hand-set.

    python -m src.learned            # leave-one-game-out on Metrica (+ saves labels for the
                                     # bootstrap), then fits on all 3 games -> models/learned.json

Two small logistic regressions (context.fit_logistic: ridge, Newton, deterministic), fitted on the
labels derived from Metrica's hand-coded events (src/other_games.py):

  dead model  D vs live, on every graded frame. Features: player-speed z (as in model.py), ball
              missing, ball fast inside the pitch, ball still -- the clues both benches have.
              Off-pitch ball, throw-in hold, Smart Tagging and pre-kickoff stay hand-set: the
              Metrica tracking has no ball beyond the lines and no tags, so it cannot teach them.
  team model  H vs A, on live frames. Features: the unit ball-control evidence
              (model.team_evidence) and the team-shape context logit (src/context.py).

A discriminative model gives P(state | frame); the HMM wants a likelihood ratio, so the training
base rate is subtracted from the logit. Frames are strongly correlated, so a per-frame logit
over-counts; it is divided by its own integrated autocorrelation time on the training games (the
rule context.py already uses). The team side is still scaled by the K_TEAM calibration.
Weights are learned from other games only (RULES.md allows it); nothing here touches BAR-ATH labels.
"""

import json
import os

import numpy as np

from src import atd, context, features, model

MODEL_PATH = os.path.join(atd.ROOT, "models", "learned.json")
MODEL_PATH_NOMISS = os.path.join(atd.ROOT, "models", "learned_nomiss.json")    # 20d
DEAD_NAMES = ["speed_z", "ball_missing", "ball_fast", "ball_still"]
TEAM_NAMES = ["control", "context"]


def dead_features(f):
    z = np.clip((model.WALK_SPEED - np.nan_to_num(f["mean_speed"], nan=model.WALK_SPEED))
                / model.SPEED_SCALE, -3, 3)
    seen_in = f["ball_seen"] & ~f["ball_out"]
    fast = seen_in & (np.nan_to_num(f["ball_speed"]) > model.BALL_FAST)
    if model.CONFIG["learned_no_missing"]:
        return np.column_stack([z, fast, model.ball_still(f)]).astype(np.float64)
    return np.column_stack([z, ~f["ball_seen"], fast, model.ball_still(f)]).astype(np.float64)


def _dead_names():
    return [n for n in DEAD_NAMES if not (model.CONFIG["learned_no_missing"] and n == "ball_missing")]


def model_path():
    return MODEL_PATH_NOMISS if model.CONFIG["learned_no_missing"] else MODEL_PATH


def team_features(a, f):
    ctx = context.context_llr(a, f) if model.CONFIG["context"] else np.zeros(len(f["bx"]))
    return np.column_stack([model.team_evidence(f, model.CONFIG), ctx]).astype(np.float64)


def _fit(X, y):
    w = context.fit_logistic(X, y.astype(float))
    logit = np.column_stack([np.ones(len(X)), X]) @ w
    prior = np.log(y.mean() / (1 - y.mean()))
    return w, prior, logit


def fit(games):
    """games: [(Xd, Xt, gt labels)]. Returns a JSON-able dict of weights."""
    Xd, yd, Xt, yt, tau_d, tau_t = [], [], [], [], [], []
    for xd, xt, gt in games:
        graded = gt != "X"
        live = np.isin(gt, ["H", "A"])
        Xd.append(xd[graded]); yd.append(gt[graded] == "D")
        Xt.append(xt[live]); yt.append(gt[live] == "H")
    wd, pd_, _ = _fit(np.concatenate(Xd), np.concatenate(yd))
    wt, pt, _ = _fit(np.concatenate(Xt), np.concatenate(yt))
    for xd, xt, gt in games:                       # autocorrelation of the fitted logits, per game
        g = gt != "X"
        tau_d.append(context.autocorr_time((np.column_stack([np.ones(len(xd)), xd]) @ wd)[g]))
        tau_t.append(context.autocorr_time((np.column_stack([np.ones(len(xt)), xt]) @ wt)[g]))
    return {"dead": {"w": wd.tolist(), "names": ["bias"] + _dead_names(), "prior": float(pd_),
                     "tau": float(np.mean(tau_d))},
            "team": {"w": wt.tolist(), "names": ["bias"] + TEAM_NAMES, "prior": float(pt),
                     "tau": float(np.mean(tau_t))}}


def dead_core(f, m):
    """Learned D log-score per frame (relative to live), per-frame nats. Scaled by 1/tau (round
    20) or, if the model carries one, by a "scale" chosen on the training games (round 20b)."""
    d = m["dead"]
    logit = np.column_stack([np.ones(len(f["bx"])), dead_features(f)]) @ np.array(d["w"])
    return (logit - d["prior"]) * (d["scale"] if "scale" in d else 1.0 / d["tau"])


def team_core(a, f, m):
    """Learned H-vs-A log-odds per frame, unit weight (K_TEAM still calibrates its scale)."""
    t = m["team"]
    logit = np.column_stack([np.ones(len(f["bx"])), team_features(a, f)]) @ np.array(t["w"])
    return (logit - t["prior"]) / t["tau"]


def load(path=None):
    with open(path or model_path()) as handle:
        return json.load(handle)


def main():
    from src import other_games as og
    import possession_metrics as pm
    cache = {}
    for g in (1, 2, 3):
        a = og.load_tracking(g)
        gt, blocks = og.gt_labels(g, len(a["bx"]))
        og.point_at(blocks)
        f = features.compute(a)
        cache[g] = (a, blocks, gt, dead_features(f), team_features(a, f))
    out = os.path.join(atd.CACHE_DIR, "other_games")
    os.makedirs(out, exist_ok=True)
    for mode in ("dead", "team", "both"):
        model.CONFIG["learned"] = mode
        for g in (1, 2, 3):                        # leave one game out
            m = fit([(cache[h][3], cache[h][4], cache[h][2]) for h in (1, 2, 3) if h != g])
            a, blocks, gt, _, _ = cache[g]
            pred, k = og.predict(a, blocks, learned=m)
            np.save(os.path.join(out, "learn_{}_g{}.npy".format(mode, g)), pred)
            r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
            print("LOGO {} game {}: spellF1 {:.3f} macroF1 {:.3f} poss {} (gt {}) D {:.3f} (gt {:.3f}) K {}".format(
                mode, g, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                np.mean(pred[gt != "X"] == "D"), np.mean(gt[gt != "X"] == "D"), k))
    model.CONFIG["learned"] = "off"
    m = fit([(cache[h][3], cache[h][4], cache[h][2]) for h in (1, 2, 3)])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", newline="\n") as handle:
        json.dump(m, handle, indent=1)
    print("all-games model ->", os.path.relpath(MODEL_PATH, atd.ROOT))
    for side in ("dead", "team"):
        print(side, dict(zip(m[side]["names"], np.round(m[side]["w"], 3))),
              "prior", round(m[side]["prior"], 3), "tau_frames", round(m[side]["tau"], 1))


SCALES = (0.025, 0.05, 0.1, 0.2)      # 20b grid, fixed beforehand (hand speed weight / learned ~ 0.07)
SCALES_20D = (0.05, 0.1, 0.2, 0.4, 0.8)   # 20d grid, fixed before its run (20b picked its top edge)


def main_scale():
    """Round 20b: learned dead-side weights, one overall scale picked on the training games by
    decoded spellF1 (leave one game out), team side hand-set as in v8."""
    from src import other_games as og
    import possession_metrics as pm
    cache = {}
    for g in (1, 2, 3):
        a = og.load_tracking(g)
        gt, blocks = og.gt_labels(g, len(a["bx"]))
        og.point_at(blocks)
        f = features.compute(a)
        cache[g] = (a, blocks, gt, dead_features(f), team_features(a, f))
    model.CONFIG["learned"] = "dead"
    scales = SCALES_20D if model.CONFIG["learned_no_missing"] else SCALES
    score = {}                               # (held-out-fit g, scale, eval game h) -> spellF1
    preds = {}
    fits = {g: fit([(cache[h][3], cache[h][4], cache[h][2]) for h in (1, 2, 3) if h != g])
            for g in (1, 2, 3)}
    fits[0] = fit([(cache[h][3], cache[h][4], cache[h][2]) for h in (1, 2, 3)])
    for g in (1, 2, 3):
        for s in scales:
            m = json.loads(json.dumps(fits[g]))
            m["dead"]["scale"] = s
            for h in (1, 2, 3):              # training games (h != g) pick s; g is held out
                a, blocks, gt, _, _ = cache[h]
                pred, _ = og.predict(a, blocks, learned=m)
                r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
                score[(g, s, h)] = r["spell_f1_mean"]
                preds[(g, s, h)] = pred
    out = os.path.join(atd.CACHE_DIR, "other_games")
    chosen = {}
    for g in (1, 2, 3):
        train = [h for h in (1, 2, 3) if h != g]
        best = max(scales, key=lambda s: (np.mean([score[(g, s, h)] for h in train]), -s))
        chosen[g] = best
        tag = "learn_dead_nm" if model.CONFIG["learned_no_missing"] else "learn_dead_s"
        np.save(os.path.join(out, "{}_g{}.npy".format(tag, g)), preds[(g, best, g)])
        print("held out game {}: scale {} chosen on games {} (train spellF1 {}), held-out spellF1 {:.3f}".format(
            g, best, train, [round(np.mean([score[(g, s, h)] for h in train]), 3) for s in scales],
            score[(g, best, g)]))
    m = fits[0]
    m["dead"]["scale"] = max(scales, key=lambda s: (np.mean([score[(g, s, g)] for g in (1, 2, 3)]), -s))
    print("all-games scale (from the held-out scores of each scale):", m["dead"]["scale"])
    print("dead weights", dict(zip(m["dead"]["names"], np.round(m["dead"]["w"], 3))))
    with open(model_path(), "w", newline="\n") as handle:
        json.dump(m, handle, indent=1)
    model.CONFIG["learned"] = "off"


if __name__ == "__main__":
    import sys
    if "--no-missing" in sys.argv:
        model.CONFIG["learned_no_missing"] = True
    main_scale() if "--dead-scale" in sys.argv else main()
