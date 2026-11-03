"""v1: per-frame evidence -> 3-state (H, A, D) HMM -> Viterbi.

Every constant below is set from general football reasoning or from unlabelled aggregate
statistics of this match's tracking (notes/exploration.md), never from the public ground truth.
The reasoning is written next to each one.
"""

import math

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

# --- team evidence (ball seen, in bounds) ----------------------------------------------------
# A player within a couple of metres of the ball, with the nearest opponent further away, is
# evidence of control. The weight decays with the controller's distance (3 m scale: beyond a
# stride or two the ball is loose or in flight) and saturates in the margin (2 m scale).
# K_TEAM was first set to 0.2 nats/frame (~2 s of clear control to overturn a possession), but
# the ball is unseen on 41% of frames and the controller is rarely within 1 m, so the effective
# evidence was much weaker than intended: the decoded match had 125 possessions of 32.6 s, far
# from the 10-15 s football prior. K_TEAM is therefore set so that the decoder's OWN output
# matches that prior (no labels involved): 0.2 -> 32.6 s, 0.4 -> 22.4 s, 0.8 -> 15.3 s,
# 1.0 -> 12.9 s, 1.6 -> 9.5 s. 1.0 is the round value inside the 10-15 s band.
K_TEAM = 1.0
CONTROL_DIST_M = 3.0
MARGIN_SCALE_M = 2.0

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


def transition_logs():
    leave = 1.0 / (POSSESSION_S * FPS)
    to_d = leave * SHARE_ENDING_IN_STOPPAGE
    to_other = leave - to_d
    restart = 1.0 / (DEAD_BALL_S * FPS)
    T = np.array([[1 - leave, to_other, to_d],
                  [to_other, 1 - leave, to_d],
                  [restart / 2, restart / 2, 1 - restart]])
    return np.log(T)


def emissions(f, set_piece_mask):
    """(n, 3) log-scores. Only differences between states matter."""
    n = len(f["margin"])
    E = np.zeros((n, 3), np.float64)

    seen_in = f["ball_seen"] & ~f["ball_out"]
    dmin = np.fmin(f["d_home"], f["d_away"])
    w = np.where(seen_in, np.exp(-np.nan_to_num(dmin, nan=99.0) / CONTROL_DIST_M), 0.0)
    llr = K_TEAM * w * np.tanh(np.nan_to_num(f["margin"]) / MARGIN_SCALE_M)
    E[:, H] += llr / 2
    E[:, A] -= llr / 2

    z = np.clip((WALK_SPEED - np.nan_to_num(f["mean_speed"], nan=WALK_SPEED)) / SPEED_SCALE, -3, 3)
    E[:, D] += K_SPEED * z
    E[:, D] += K_BALL_MISSING * ~f["ball_seen"]
    E[:, D] += K_BALL_OUT * f["ball_out"]
    fast = seen_in & (np.nan_to_num(f["ball_speed"]) > BALL_FAST)
    E[:, D] -= K_BALL_FAST * fast
    E[:, D] += K_SET_PIECE * set_piece_mask
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
