"""Team-shape context: who has the ball, judged from the 22 players alone (no ball needed).

Football fact behind it: the team in possession spreads out and pushes up the pitch; the team
without it compacts and drops. That shape is visible even while the tracker has lost the ball,
which is 41% of graded frames.

Learned WITHOUT human labels. Pseudo-labels come from the algorithm itself, on frames where the
ball evidence is unambiguous: ball seen inside the pitch, not in flight (< 8 m/s), a player within
1.5 m of it and the nearest opponent at least 3 m further away. A logistic regression maps the
shape features to those pseudo-labels. It is trained on one half and checked on the other (and
vice versa), so the reported accuracy is out-of-sample -- against pseudo-labels, not ground truth.

The per-frame log-odds are then divided by the autocorrelation time of the signal (in frames):
shape changes slowly, so consecutive frames are not independent evidence, and adding them up
frame by frame would count the same observation many times.
"""

import warnings

import numpy as np

from games import GAME, half_video_blocks
from src.features import FPS, PITCH, nan_smooth

PSEUDO_DIST_M = 1.5
PSEUDO_MARGIN_M = 3.0
PSEUDO_MAX_BALL_SPEED = 8.0
L2 = 1e-2          # ridge penalty on standardised features; fixed, small
NAMES = ["territory", "spread_diff", "depth_diff", "advance_speed", "speed_diff"]


def shape_features(a, f):
    px, py, ps, team = a["px"], a["py"], a["ps"], a["team"]
    L, _ = PITCH
    n = len(px)
    home, away = team == 0, team == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx_h, cx_a = np.nanmean(px[:, home], 1), np.nanmean(px[:, away], 1)
        cx_all = np.nanmean(px, 1)
        spread_h = np.sqrt(np.nanvar(px[:, home], 1) + np.nanvar(py[:, home], 1))
        spread_a = np.sqrt(np.nanvar(px[:, away], 1) + np.nanvar(py[:, away], 1))
        depth_h = np.nanpercentile(px[:, home], 90, 1) - np.nanpercentile(px[:, home], 10, 1)
        depth_a = np.nanpercentile(px[:, away], 90, 1) - np.nanpercentile(px[:, away], 10, 1)
        sp_h = np.nanmean(np.clip(ps[:, home], 0, 12), 1)
        sp_a = np.nanmean(np.clip(ps[:, away], 0, 12), 1)

    # Attack direction per half, unsupervised: over a whole half each team sits nearer its own
    # goal on average, so home attacks towards +x if its mean x is below the away team's.
    direction = np.zeros(n)
    for _, first, last in half_video_blocks(GAME):
        sl = slice(first, last + 1)
        direction[sl] = 1.0 if np.nanmean(cx_h[sl]) < np.nanmean(cx_a[sl]) else -1.0

    s = FPS // 2
    cx_s = nan_smooth(cx_all, s)
    adv = np.full(n, np.nan)
    adv[FPS:] = (cx_s[FPS:] - cx_s[:-FPS]) * direction[FPS:]      # m/s towards home's attack
    X = np.column_stack([
        nan_smooth((cx_all - L / 2) * direction / (L / 2), s),     # +: play in home's attacking half
        nan_smooth(spread_h - spread_a, s),
        nan_smooth(depth_h - depth_a, s),
        adv,
        nan_smooth(sp_h - sp_a, s),
    ])
    return X, direction


def pseudo_labels(f):
    """+1 home, -1 away, 0 no pseudo-label."""
    dmin = np.fmin(f["d_home"], f["d_away"])
    ok = (f["ball_seen"] & ~f["ball_out"]
          & (np.nan_to_num(f["ball_speed"], nan=99) < PSEUDO_MAX_BALL_SPEED)
          & (np.nan_to_num(dmin, nan=99) < PSEUDO_DIST_M)
          & (np.abs(np.nan_to_num(f["margin"])) > PSEUDO_MARGIN_M))
    return np.where(ok, np.sign(np.nan_to_num(f["margin"])), 0).astype(np.int8)


def fit_logistic(X, y, iters=50):
    """Ridge logistic regression by Newton's method. Deterministic. y in {0, 1}."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    reg = L2 * np.eye(len(w))
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + reg @ w
        Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb / len(y) + reg
        w -= np.linalg.solve(Hm, g)
    return w


def autocorr_time(x, max_lag=50 * FPS):
    """Integrated autocorrelation time in frames (sum of ACF up to its first drop below 0)."""
    x = x[np.isfinite(x)]
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    tau = 1.0
    for lag in range(1, min(max_lag, len(x) // 2), 5):
        r = np.dot(x[:-lag], x[lag:]) / (len(x) - lag) / var
        if r <= 0:
            break
        tau += 2 * r * 5          # step of 5 frames
    return tau


def context_llr(a, f, report=None):
    """Per-frame H-vs-A log-odds from shape, already divided by the autocorrelation time."""
    X, _ = shape_features(a, f)
    y = pseudo_labels(f)
    finite = np.all(np.isfinite(X), 1)
    mu = np.nanmean(X[finite], 0)
    sd = np.nanstd(X[finite], 0) + 1e-9
    Z = np.where(finite[:, None], (X - mu) / sd, 0.0)

    halves = half_video_blocks(GAME)
    in_half = [np.zeros(len(X), bool) for _ in halves]
    for m, (_, first, last) in zip(in_half, halves):
        m[first:last + 1] = True
    # out-of-sample check: train on one half, test on the other
    accs = []
    for tr, te in ((0, 1), (1, 0)):
        trm, tem = in_half[tr] & (y != 0) & finite, in_half[te] & (y != 0) & finite
        w = fit_logistic(Z[trm], (y[trm] > 0).astype(float))
        pred = np.column_stack([np.ones(tem.sum()), Z[tem]]) @ w > 0
        accs.append(float(np.mean(pred == (y[tem] > 0))))

    graded = in_half[0] | in_half[1]
    trm = graded & (y != 0) & finite
    w = fit_logistic(Z[trm], (y[trm] > 0).astype(float))
    logit = np.column_stack([np.ones(len(Z)), Z]) @ w
    logit[~finite] = 0.0
    tau = autocorr_time(logit[graded])
    if report is not None:
        report.update(pseudo_frames=int(trm.sum()),
                      home_share=float(np.mean(y[trm] > 0)),
                      cross_half_acc=accs, tau_s=tau / FPS,
                      weights=dict(zip(["bias"] + NAMES, np.round(w, 3).tolist())))
    return logit / tau
