"""Label-free dead-ball / restart check from a signal the model does NOT use (aggregates only).

    python -m src.restart_report submissions/v3.csv [more.csv ...]

Football fact: almost every restart (free kick, corner, goal kick, kickoff, many throw-ins) is a
ball that sits still for a couple of seconds and is then struck. In open play the ball is almost
never seen motionless for >= 2 s. So "stationary ball, then a kick" (a *placed-ball kick*, PBK) is
an independent marker of a dead-ball stretch ending at the kick. The model's D evidence uses mean
player speed, ball missing/out/fast and set-piece windows -- not ball stillness.

Reported per submission (no rows, no frames):
  n_pbk            placed-ball kicks found on graded frames (same for every submission)
  d_before         share of the 2 s before the kick predicted D (want high)
  live_after       share of 1..3 s after the kick predicted live (want high)
  restart_1s       share of PBKs with a predicted D->live switch within 1 s of the kick
  restart_off_med  median offset (s) of the nearest D->live switch (prediction - kick)
  by zone          the same restart_1s split by where the ball sat (corner, goal area,
                   touchline, centre spot, other)
"""

import sys

import numpy as np

from src import atd, features, proxy_eval
from games import GAME, half_video_blocks

FPS = features.FPS
STILL_SPEED = 1.5      # m/s on the 0.2 s-smoothed ball speed (tracker jitter on a resting ball)
STILL_S = 1.5          # seconds the ball must sit still
KICK_SPEED = 6.0       # m/s within KICK_WINDOW_S after the still stretch ends
KICK_WINDOW_S = 1.0


def placed_ball_kicks(f, graded):
    """[(kick_frame, x, y)]: ball seen and still >= STILL_S, then faster than KICK_SPEED."""
    spd = f["ball_speed"]
    still = f["ball_seen"] & (np.nan_to_num(spd, nan=99) < STILL_SPEED)
    out = []
    edges = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    w = int(KICK_WINDOW_S * FPS)
    for s, e in zip(starts, ends):
        if e - s < STILL_S * FPS or e + w >= len(spd) or not graded[e]:
            continue
        after = np.nan_to_num(spd[e:e + w], nan=0)
        if after.max() < KICK_SPEED:
            continue
        out.append((e, float(f["bx"][e - 1]), float(f["by"][e - 1])))
    return out


def zone(x, y):
    L, W = features.PITCH
    dx = min(x, L - x)
    if dx < 3 and (y < 3 or y > W - 3):
        return "corner"
    if dx < 7 and abs(y - W / 2) < 11:
        return "goal_area"
    if y < 2 or y > W - 2:
        return "touchline"
    if abs(x - L / 2) < 2 and abs(y - W / 2) < 2:
        return "centre"
    return "other"


def d_to_live(labels):
    out = []
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        for c in ch:
            if seg[c - 1] == "D" and seg[c] in ("H", "A"):
                out.append(first + c)
    return np.array(out)


def report(labels, kicks):
    switches = d_to_live(labels)
    rows = {}
    d_before, live_after, offs, zones = [], [], [], {}
    for k, x, y in kicks:
        d_before.append(np.mean(labels[k - 2 * FPS:k] == "D"))
        live_after.append(np.mean(np.isin(labels[k + FPS:k + 3 * FPS], ["H", "A"])))
        off = (switches[np.argmin(np.abs(switches - k))] - k) / FPS if len(switches) else np.nan
        offs.append(off)
        zones.setdefault(zone(x, y), []).append(abs(off) <= 1)
    offs = np.array(offs)
    rows["n_pbk"] = len(kicks)
    rows["d_before"] = float(np.mean(d_before))
    rows["live_after"] = float(np.mean(live_after))
    rows["restart_1s"] = float(np.mean(np.abs(offs) <= 1))
    rows["restart_off_med"] = float(np.nanmedian(offs))
    rows["zones"] = {z: "{}/{}".format(int(np.sum(v)), len(v)) for z, v in sorted(zones.items())}
    return rows


def throw_reentries(f, graded):
    """[(hold_start, reentry_frame)]: ball held beyond a touchline (model.throw_hold, episodes
    merged over 3 s gaps, >= 0.2 s), then the first frame it is seen back inside the pitch
    within 10 s. Re-entry is pure geometry: play restarts when the throw brings the ball in."""
    from src import model
    L, W = features.PITCH
    m = model.throw_hold(f) & graded
    e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    eps = []
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if eps and s - eps[-1][1] < 3 * FPS:
            eps[-1][1] = t
        else:
            eps.append([s, t])
    with np.errstate(invalid="ignore"):
        inside = f["ball_seen"] & (f["by"] > 0) & (f["by"] < W) & (f["bx"] > 0) & (f["bx"] < L)
    out = []
    for s, t in eps:
        if t - s < 5:
            continue
        nxt = np.flatnonzero(inside[t:t + 10 * FPS])
        if len(nxt):
            out.append((s, t + int(nxt[0])))
    return out


def throw_report(labels, eps):
    switches = d_to_live(labels)
    offs = np.array([(switches[np.argmin(np.abs(switches - r))] - r) / FPS for _, r in eps])
    d_hold = np.mean([np.mean(labels[s:r] == "D") for s, r in eps])
    return {"n_throw": len(eps), "throw_d_during_hold": round(float(d_hold), 3),
            "throw_restart_1s": round(float(np.mean(np.abs(offs) <= 1)), 3),
            "throw_off_med": round(float(np.median(offs)), 2)}


def main(paths):
    a = atd.load()
    f = features.compute(a)
    graded = proxy_eval.graded_mask(len(a["bx"]))
    kicks = placed_ball_kicks(f, graded)
    throws = throw_reentries(f, graded)
    for p in paths:
        lab = proxy_eval.fill_x(proxy_eval.read(p))
        r = report(lab, kicks)
        r.update(throw_report(lab, throws))
        print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
