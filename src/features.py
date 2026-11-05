"""Per-frame evidence computed from the ATD arrays (see src/atd.py). All distances in metres."""

import warnings

import numpy as np

from games import GAME, GAMES

PITCH = GAMES[GAME]["pitch"]
FPS = GAMES[GAME]["fps"]
MAX_PLAYER_SPEED = 12.0   # m/s; faster than any sprinter, so anything above is an ID switch


def nan_smooth(x, half_width):
    """Centred moving mean over +/- half_width frames, ignoring NaN. NaN where nothing is seen."""
    if half_width <= 0:
        return x.astype(np.float32)
    ok = np.isfinite(x)
    vals = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cnts = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half_width)
    hi = np.minimum(len(x), i + half_width + 1)
    c = cnts[hi] - cnts[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (vals[hi] - vals[lo]) / c
    out[c == 0] = np.nan
    return out.astype(np.float32)


def compute(a):
    px, py, ps, team = a["px"], a["py"], a["ps"], a["team"]
    bx, by = a["bx"], a["by"]
    L, W = PITCH

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist = np.hypot(px - bx[:, None], py - by[:, None])          # (n, k), NaN if either missing
        d_home = np.nanmin(np.where(team[None, :] == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == 1, dist, np.nan), axis=1)

        n_players = np.isfinite(px).sum(axis=1)

        # Player movement. The tracker's own speed field, clipped against identity-switch spikes;
        # if it is absent, fall back to frame-to-frame displacement.
        speed = ps
        if not np.isfinite(speed).any():
            speed = np.vstack([np.full((1, px.shape[1]), np.nan, np.float32),
                               np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * FPS])
        speed = np.clip(speed, 0, MAX_PLAYER_SPEED)
        mean_speed = np.nanmean(speed, axis=1)

    # Ball speed from consecutive sightings, lightly smoothed (+/- 2 frames = 0.2 s window).
    step = np.full(len(bx), np.nan, np.float32)
    step[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS
    ball_speed = nan_smooth(step, 2)

    seen = np.isfinite(bx)
    ball_out = seen & ((bx < -1) | (bx > L + 1) | (by < -1) | (by > W + 1))

    return dict(
        d_home=d_home.astype(np.float32),
        d_away=d_away.astype(np.float32),
        margin=(d_away - d_home).astype(np.float32),     # > 0: a home player is closer
        n_players=n_players,
        mean_speed=nan_smooth(mean_speed, FPS // 2),     # 1 s window
        ball_speed=ball_speed,
        ball_seen=seen,
        bx=bx,
        by=by,
        ball_out=ball_out,
    )
