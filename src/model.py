"""Per-frame evidence -> 3-state (H, A, D) HMM -> Viterbi.

Every constant below is set from general football reasoning or from unlabelled aggregate
statistics of this match's tracking (notes/exploration.md), never from the public ground truth.
The reasoning is written next to each one. Options that were tried are switches in CONFIG, so an
experiment is one flag (see notes/experiments.md for which were kept).
"""

import numpy as np

from src.features import FPS

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# --- transitions -----------------------------------------------------------------------------
# A possession in elite football lasts ~10-15 s on average; take 12.5 s. Roughly a third of
# possessions end in a stoppage (out of play, foul, goal) and two thirds in a turnover.
POSSESSION_S = 12.5
SHARE_ENDING_IN_STOPPAGE = 1.0 / 3.0
# Dead-ball stretches (throw-ins ~15-20 s, goal kicks / free kicks / corners 25-45 s, longer
# for goals and injuries): ~30 s on average. The restart is equally likely to go to either side.
DEAD_BALL_S = 30.0

# --- team evidence weight --------------------------------------------------------------------
# K_TEAM scales all H-vs-A evidence. It is not hand-picked: src/run.py calibrates it (bisection)
# so that the decoded match's OWN mean possession length equals POSSESSION_S. No labels involved.
# v1 picked 1.0 by hand from this sweep: 0.2 -> 32.6 s, 0.4 -> 22.4 s, 0.8 -> 15.3 s,
# 1.0 -> 12.9 s, 1.6 -> 9.5 s.
K_TEAM = 1.0

# --- dead-ball evidence ----------------------------------------------------------------------
# Live play: outfield players average ~1.8-2 m/s over 90 minutes (10-11 km). During stoppages
# they walk (~1 m/s). Walking pace, 1.3 m/s, is the midpoint; 0.4 m/s is the spread.
WALK_SPEED = 1.3
SPEED_SCALE = 0.4
K_SPEED = 0.04          # nats/frame per unit z, z clipped to +/-3 -> at most ~3 nats per second
K_BALL_MISSING = 0.02   # the tracker loses the ball more often when it is dead (out, held)
K_BALL_OUT = 0.15       # ball seen beyond the lines (by > 1 m): strong, but tracker can be wrong
K_BALL_FAST = 0.05      # ball travelling > 3 m/s inside the pitch: someone kicked it, play is live
BALL_FAST = 3.0
K_SET_PIECE = 0.01      # inside a Smart Tagging SET PIECES window (35 s): a restart is nearby

CONFIG = {
    # A player within a couple of metres of the ball, with the nearest opponent further away, is
    # evidence of control. Weight decays with the controller's distance and saturates in the
    # margin. v1: 3 m (a stride or two).
    "control_dist_m": 3.0,
    "margin_scale_m": 2.0,
    # A ball travelling faster than this is in flight (passes and shots run 10-30 m/s, a dribble
    # 2-6 m/s): the nearest player is not controlling it. None = v1 (no gate). Kept in v2.
    "max_control_ball_speed": 8.0,
    # Controlled possession: the same team must be nearest for this many consecutive ball-seen
    # frames before its evidence counts in full (ramps up linearly). 0 = v1.
    "persist_frames": 0,
    # Add team-shape context (src/context.py), learned from the algorithm's own pseudo-labels.
    "context": True,     # kept in v3
    # Multiplier on the context log-odds (1 = as derived: per-frame logit / autocorrelation time).
    "context_scale": 1.0,
}


def transition_logs():
    leave = 1.0 / (POSSESSION_S * FPS)
    to_d = leave * SHARE_ENDING_IN_STOPPAGE
    to_other = leave - to_d
    restart = 1.0 / (DEAD_BALL_S * FPS)
    T = np.array([[1 - leave, to_other, to_d],
                  [to_other, 1 - leave, to_d],
                  [restart / 2, restart / 2, 1 - restart]])
    return np.log(T)


def _run_lengths(sign):
    """For each frame, how many consecutive frames (ending here) share the same nonzero sign."""
    out = np.zeros(len(sign), np.int32)
    count, last = 0, 0
    for i, s in enumerate(sign.tolist()):
        if s != 0 and s == last:
            count += 1
        elif s != 0:
            count = 1
        else:
            count = 0
        last = s
        out[i] = count
    return out


def team_evidence(f, cfg=CONFIG):
    """Unit-weight H-vs-A log-likelihood ratio per frame (> 0 favours H)."""
    seen_in = f["ball_seen"] & ~f["ball_out"]
    dmin = np.nan_to_num(np.fmin(f["d_home"], f["d_away"]), nan=99.0)
    w = np.where(seen_in, np.exp(-dmin / cfg["control_dist_m"]), 0.0)
    if cfg["max_control_ball_speed"] is not None:
        w = np.where(np.nan_to_num(f["ball_speed"]) > cfg["max_control_ball_speed"], 0.0, w)
    margin = np.nan_to_num(f["margin"])
    if cfg["persist_frames"] > 0:
        sign = np.where(w > 0, np.sign(margin), 0).astype(np.int8)
        w = w * np.minimum(_run_lengths(sign) / cfg["persist_frames"], 1.0)
    return w * np.tanh(margin / cfg["margin_scale_m"])


def dead_evidence(f, set_piece_mask):
    """D log-score per frame relative to the live states."""
    seen_in = f["ball_seen"] & ~f["ball_out"]
    z = np.clip((WALK_SPEED - np.nan_to_num(f["mean_speed"], nan=WALK_SPEED)) / SPEED_SCALE, -3, 3)
    out = K_SPEED * z
    out += K_BALL_MISSING * ~f["ball_seen"]
    out += K_BALL_OUT * f["ball_out"]
    out -= K_BALL_FAST * (seen_in & (np.nan_to_num(f["ball_speed"]) > BALL_FAST))
    out += K_SET_PIECE * set_piece_mask
    return out


def emissions(team_llr, dead, k_team=None, extra_llr=0.0):
    """(n, 3) log-scores. Only differences between states matter.

    team_llr is scaled by K_TEAM; extra_llr (e.g. shape context) is already in nats and is not.
    """
    k = K_TEAM if k_team is None else k_team
    total = k * team_llr + extra_llr
    E = np.zeros((len(dead), 3), np.float64)
    E[:, H] = total / 2
    E[:, A] = -total / 2
    E[:, D] = dead
    return E


def viterbi(E, logT):
    """Most likely state path for one contiguous block. Uniform initial distribution."""
    n = len(E)
    back = np.zeros((n, 3), np.int8)
    score = E[0].copy()
    t = logT.tolist()
    for i in range(1, n):
        s0, s1, s2 = score
        new = np.empty(3)
        for j in range(3):
            c0, c1, c2 = s0 + t[0][j], s1 + t[1][j], s2 + t[2][j]
            if c0 >= c1 and c0 >= c2:
                new[j], back[i, j] = c0, 0
            elif c1 >= c2:
                new[j], back[i, j] = c1, 1
            else:
                new[j], back[i, j] = c2, 2
        score = new + E[i]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path
