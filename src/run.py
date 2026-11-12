"""Build a submission from the provided data. Deterministic: no randomness anywhere.

    python -m src.run                                   # -> submissions/Brasamris.csv (= v9)
    python -m src.run --set persist_frames=10 --out cache/exp.csv
    python -m src.run --set POSSESSION_S=20 --out cache/exp.csv   # model.py constants too

v1 (committed earlier) is reproduced by: --k-team 1.0 --set max_control_ball_speed=None
"""

import argparse
import ast
import os

import numpy as np

from src import atd, context, features, model, tagging
from games import GAME, half_video_blocks

OUT = os.path.join(atd.ROOT, "submissions", "Brasamris.csv")   # team name (SUBMISSION.md); = v9


TEAM_TAG = {"barcelona": 1.0, "atletic club": -1.0}     # +1 home (H), -1 away (A)


def set_piece_evidence(n, cfg=model.CONFIG):
    """(dead_mask, team_sign) from Smart Tagging SET PIECES windows.

    Aggregate diagnostics (src/setpiece_report.py) show the restart sits near the START of each
    fixed 35 s window: predicted D share is ~0.6 in the 10 s before it and ~0.1 from 5 s after.
    dead_mask: "full" = whole window (v1-v3); "start" = from 10 s before to 3 s after the tag start.
    team_sign: +1/-1 for the tagged restarting team, over the first `sp_team_s` seconds after the
    tag start (corners, goal kicks, free kicks carry a Team; throw-ins and kickoffs do not).
    """
    fps = 25
    dead = np.zeros(n, bool)
    team = np.zeros(n, np.float64)
    for e in tagging.load():
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        s, end = max(e["start_frame"], 0), max(e["end_frame"], 0)
        if cfg["sp_dead_window"] == "start":
            dead[max(s - 10 * fps, 0):s + 3 * fps] = True
        else:
            dead[s:end] = True
        sign = TEAM_TAG.get((e["tags"].get("Team") or "").strip().lower(), 0.0)
        if sign and cfg["sp_team_s"] > 0:
            team[s:s + int(cfg["sp_team_s"] * fps)] = sign
    return dead, team


def restart_penalty(n, cfg=model.CONFIG):
    """(n, 2) cost of D -> H / D -> A at each frame from team-tagged SET PIECES (or None)."""
    if not cfg["sp_restart_pen"]:
        return None
    fps = 25
    pen = np.zeros((n, 2))
    for e in tagging.load():
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        sign = TEAM_TAG.get((e["tags"].get("Team") or "").strip().lower(), 0.0)
        if not sign:
            continue
        s = e["start_frame"]
        lo = max(s - int(cfg["sp_restart_before_s"] * fps), 0)
        hi = max(s + int(cfg["sp_restart_after_s"] * fps), 0)
        pen[lo:hi, 1 if sign > 0 else 0] = cfg["sp_restart_pen"]   # forbid the other side
    return pen


def prekick_mask(n):
    """Bool per frame: graded frames of a half before its Kick Off (Start) tag (ball not yet in
    play). Only the half-start kickoffs; kickoffs after goals follow a D stretch anyway."""
    m = np.zeros(n, bool)
    starts = [e["start_frame"] for e in tagging.load()
              if (e["code"] or "").strip().lower() == "set pieces"
              and (e["tags"].get("Type") or "").strip().lower() == "kick off (start)"]
    for _, first, last in half_video_blocks(GAME):
        for k in starts:
            if first < k <= last:
                m[first:k] = True
    return m


def load_tracking(cfg=model.CONFIG):
    """Tactical ATD arrays, with the ball filled from the aligned broadcast if configured."""
    a = atd.load()
    if cfg["broadcast_ball"] != "off":
        from src import broadcast
        mbx, mby, _ = broadcast.mapped_ball(use_players=cfg["broadcast_ball"] == "players")
        fill = ~np.isfinite(a["bx"]) & np.isfinite(mbx)
        a = dict(a)
        a["bx"] = np.where(fill, mbx, a["bx"]).astype(np.float32)
        a["by"] = np.where(fill, mby, a["by"]).astype(np.float32)
    return a


def decode(E, n, abstain=False):
    labels = np.full(n, "X", dtype="<U1")      # ungraded frames: never scored
    logT = model.transition_logs()
    pen = restart_penalty(n)
    for _, first, last in half_video_blocks(GAME):
        block_pen = None if pen is None else pen[first:last + 1]
        if model.CONFIG["live_phases"] > 1 or model.CONFIG["dead_phases"] > 1:
            path = model.viterbi_chain(E[first:last + 1], block_pen)
        else:
            path = model.viterbi(E[first:last + 1], logT, block_pen)
        seg = model.enforce_min_possession(
            model.LABELS[path], int(model.CONFIG["min_possession_s"] * features.FPS))
        if abstain and model.CONFIG["abstain_below"] > 0:
            codes = np.array(
                [{"H": 0, "A": 1, "D": 2}[c] for c in seg], np.int8)
            post = model.posteriors(E[first:last + 1], logT)
            seg = seg.copy()
            seg[model.abstain(codes, post, model.CONFIG["abstain_below"])] = "X"
        labels[first:last + 1] = seg
    return labels


def possessions(labels):
    """(count, mean duration s) of maximal H/A runs, per half, graded frames only."""
    lengths = []
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(seg)]))
        lengths.extend(e - s for s, e in zip(starts, ends) if seg[s] in ("H", "A"))
    return len(lengths), (np.mean(lengths) / features.FPS if lengths else 0.0)


def calibrate(team_llr, dead, n, extra=0.0, target_s=None, steps=12, still=None):
    """K_TEAM such that the decoded match's own mean possession length is target_s.

    Larger K_TEAM -> more evidence -> more, shorter possessions, so bisect in log space.
    Deterministic: fixed bracket, fixed number of steps.
    """
    if target_s is None:
        target_s = model.POSSESSION_S          # read at call time so --set POSSESSION_S=.. works
    lo, hi = np.log(0.05), np.log(20.0)
    for _ in range(steps):
        mid = (lo + hi) / 2
        _, mean_s = possessions(decode(model.emissions(team_llr, dead, np.exp(mid), extra, still), n))
        if mean_s > target_s:
            lo = mid
        else:
            hi = mid
    return float(np.round(np.exp((lo + hi) / 2), 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override model.CONFIG entries")
    parser.add_argument("--k-team", type=float, help="fixed K_TEAM instead of calibrating")
    args = parser.parse_args()
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass                               # bare string, e.g. sp_dead_window=start
        if key in model.CONFIG:
            model.CONFIG[key] = value
        elif key.isupper() and hasattr(model, key):
            setattr(model, key, value)         # module constant, e.g. POSSESSION_S=20
        else:
            raise KeyError("unknown option {!r}; have {} or a model.py CONSTANT".format(
                key, sorted(model.CONFIG)))

    a = load_tracking()
    f = features.compute(a)
    n = len(a["bx"])
    team_llr = model.team_evidence(f, model.CONFIG)
    sp_dead, sp_team = set_piece_evidence(n)
    kickoff = model.kickoff_formation(a, f) if model.CONFIG["kickoff_nats"] else None
    prekick = prekick_mask(n) if model.CONFIG["prekick_nats"] else None
    lmode = model.CONFIG["learned"]
    lm = None
    if lmode != "off":
        from src import learned
        lm = learned.load()
    dead = model.dead_evidence(f, sp_dead, kickoff, prekick,
                               learned.dead_core(f, lm) if lmode in ("dead", "both") else None)

    extra = model.CONFIG["sp_team_nats"] * sp_team
    if model.CONFIG["context"]:
        extra = extra + model.CONFIG["context_scale"] * context.context_llr(a, f)
    if lmode in ("team", "both"):
        team_llr = learned.team_core(a, f, lm)    # control + context, learned
        extra = model.CONFIG["sp_team_nats"] * sp_team

    still = model.ball_still(f)
    if model.CONFIG["hold_nats"]:
        still = still | model.throw_hold(f)
    if kickoff is not None:
        still = still | kickoff
    still = still.astype(np.float64)
    k_team = args.k_team if args.k_team is not None else calibrate(team_llr, dead, n, extra,
                                                                    still=still)
    labels = decode(model.emissions(team_llr, dead, k_team, extra, still), n, abstain=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    count, mean_s = possessions(labels)
    graded = labels != "X"
    print("wrote {}  (K_TEAM {}, config {})".format(
        os.path.relpath(args.out, atd.ROOT), k_team, model.CONFIG))
    print("graded frames: H {:.1f}%  A {:.1f}%  D {:.1f}%".format(
        *(100 * np.mean(labels[graded] == s) for s in "HAD")))
    print("possessions (full match): {}  mean duration {:.1f} s".format(count, mean_s))


if __name__ == "__main__":
    main()
