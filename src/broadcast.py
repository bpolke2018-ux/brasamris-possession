"""Align the broadcast tracking to the tactical timeline by code, and borrow its ball.

    python -m src.broadcast          # aggregate alignment diagnostics (no rows, no frames)

Why: the tactical tracker loses the ball on 41% of graded frames, and the model is weakest
there. The broadcast camera follows the ball; its tracking (same pitch coordinates, same team
ids) sees the ball on ~50% of its own frames. The two recordings are NOT synchronised: the
broadcast has its own clock, cuts, replays and slow motion, and no mapping is provided.

How (no labels, nothing by hand):
1. Cut the broadcast into chunks of CHUNK_S seconds.
2. For each chunk, slide its ball track along the tactical ball track and compute, for every
   offset at once (FFT cross-correlation of masked signals), the mean squared distance between
   the two balls over the frames where both see it. The best offset is the one with the smallest
   distance among offsets with at least MIN_OVERLAP shared ball frames.
3. Accept a chunk only if the match is tight (median distance < ACCEPT_M at the best offset) and
   unique (the best offset further than 2 s away is at least UNIQUE_RATIO times worse). Replays in
   slow motion, close-ups and cuts fail these tests on their own.
4. Ball alignment needs the tactical ball, so it fails exactly where we need help. Second pass,
   on players: every 2 s piece of the broadcast is tried at the offsets of the nearest accepted
   chunks before and after it, and accepted if the broadcast players sit on tactical players of
   the same team (median nearest-same-team distance < PLAYER_ACCEPT_M, at least 6 players) AND
   that fit is a sharp minimum (SHARP times worse 1-2 s either side). Players are tracked in
   both feeds whether or not the ball is.
5. Broadcast frame b maps to tactical frame b + offset. Where the tactical ball is missing and
   the mapped broadcast ball is seen, the broadcast ball is used.
"""

import os

import numpy as np

from src import atd
from src.features import FPS

CHUNK_S = 10
MIN_OVERLAP = 40          # shared ball frames (1.6 s) for an offset to count
ACCEPT_M = 1.5            # median ball distance at the best offset
UNIQUE_RATIO = 4.0        # runner-up (> 2 s away) mean squared distance / best
PIECE_S = 2
PLAYER_ACCEPT_M = 0.6
SHARP = 2.5
MIN_PLAYERS = 6
CACHE = os.path.join(atd.CACHE_DIR, "broadcast_alignment.npz")


def align_chunks(t, b):
    """[(b_start, b_end, offset, median_dist_m, overlap, unique_ratio, accepted)] per chunk."""
    tx, ty = t["bx"].astype(np.float64), t["by"].astype(np.float64)
    tm = np.isfinite(tx).astype(np.float64)
    tx0, ty0 = np.nan_to_num(tx), np.nan_to_num(ty)
    tsq = tx0 ** 2 + ty0 ** 2
    bx, by = b["bx"].astype(np.float64), b["by"].astype(np.float64)
    L = CHUNK_S * FPS
    F_tm, F_tx, F_ty, F_tsq = (np.fft.rfft(v, 1 << (len(tx) + L - 1).bit_length())
                               for v in (tm, tx0, ty0, tsq))
    size = 1 << (len(tx) + L - 1).bit_length()

    def corr(F_full, chunk):
        C = np.fft.rfft(chunk[::-1], size)
        return np.fft.irfft(F_full * C, size)[L - 1:len(tx)]

    rows = []
    for s in range(0, len(bx) - L + 1, L):
        cx, cy = bx[s:s + L], by[s:s + L]
        cm = np.isfinite(cx).astype(np.float64)
        if cm.sum() < MIN_OVERLAP:
            rows.append((s, s + L, 0, np.nan, int(cm.sum()), np.nan, False))
            continue
        cx0, cy0 = np.nan_to_num(cx), np.nan_to_num(cy)
        csq = cx0 ** 2 + cy0 ** 2
        count = corr(F_tm, cm)
        ssd = corr(F_tm, csq) + corr(F_tsq, cm) - 2 * (corr(F_tx, cx0) + corr(F_ty, cy0))
        ok = count >= MIN_OVERLAP - 0.5
        msd = np.where(ok, ssd / np.maximum(count, 1), np.inf)
        d = int(np.argmin(msd))
        far = np.abs(np.arange(len(msd)) - d) > 2 * FPS
        runner = np.min(msd[far]) if far.any() else np.inf
        uniq = float(runner / max(msd[d], 1e-6))
        # exact median distance at the best offset
        seg = slice(d, d + L)
        both = np.isfinite(cx) & np.isfinite(tx[seg])
        med = float(np.median(np.hypot(cx[both] - tx[seg][both], cy[both] - ty[seg][both])))
        offset = d - s
        rows.append((s, s + L, offset, med, int(both.sum()), uniq,
                     bool(med < ACCEPT_M and uniq > UNIQUE_RATIO)))
    return rows


def player_error(t, b, s, e, off, step=5):
    """Median (over sampled frames) of the median nearest-same-team distance, in metres, from
    broadcast players in [s, e) to tactical players at frame + off. inf if not measurable."""
    n = len(t["px"])
    meds = []
    for i in range(s, e, step):
        j = i + off
        if not 0 <= j < n:
            return np.inf
        ok_b = np.isfinite(b["px"][i])
        if ok_b.sum() < MIN_PLAYERS:
            continue
        dists = []
        for team in (0, 1):
            sel = ok_b & (b["team"] == team)
            ok_t = np.isfinite(t["px"][j]) & (t["team"] == team)
            if not sel.any():
                continue
            if not ok_t.any():
                dists.extend([99.0] * int(sel.sum()))
                continue
            dx = b["px"][i][sel][:, None] - t["px"][j][ok_t][None, :]
            dy = b["py"][i][sel][:, None] - t["py"][j][ok_t][None, :]
            dists.extend(np.min(np.hypot(dx, dy), axis=1))
        meds.append(np.median(dists))
    return float(np.median(meds)) if len(meds) >= 4 else np.inf


def piece_offsets(t, b, rows):
    """Per broadcast frame: tactical offset, or the sentinel NO for frames not mapped.

    Pass-2 acceptance for a 2 s piece at candidate offset o: player error at o below
    PLAYER_ACCEPT_M, and at least SHARP times larger 1 s and 2 s either side (players move, so
    the right offset is a sharp minimum; a static scene or a replay is not)."""
    NO = np.iinfo(np.int64).min
    off = np.full(len(b["bx"]), NO, np.int64)
    acc = [(s, e, o) for s, e, o, med, ov, uq, a in rows if a]
    for s, e, o in acc:
        off[s:e] = o
    starts = np.array([s for s, _, _ in acc])
    P = PIECE_S * FPS
    for s in range(0, len(off) - P + 1, P):
        if (off[s:s + P] != NO).all():
            continue
        k = np.searchsorted(starts, s)
        cands = sorted({acc[i][2] for i in (k - 1, k) if 0 <= i < len(acc)})
        best = None
        for o in cands:
            err = player_error(t, b, s, s + P, o)
            if err >= PLAYER_ACCEPT_M:
                continue
            side = min(player_error(t, b, s, s + P, o + d) for d in (-2 * FPS, -FPS, FPS, 2 * FPS))
            if side >= SHARP * err and (best is None or err < best[1]):
                best = (o, err)
        if best is not None:
            off[s:s + P] = np.where(off[s:s + P] == NO, best[0], off[s:s + P])
    return off, NO


def mapped_ball(refresh=False, use_players=False):
    """(bx, by) on the tactical timeline from the broadcast (NaN where none), plus chunk rows.

    use_players=False: ball-aligned chunks only (pass 1). Diagnostic: where both feeds see the
    ball, pass-1 frames agree within 0.6 m (median), but player-aligned pass-2 pieces disagree by
    more than 3 m on 31% of frames -- those pieces are exactly where ball alignment failed, so their
    broadcast ball is suspect. Default is therefore pass 1 only.
    """
    cache = CACHE.replace(".npz", "_players.npz" if use_players else "_balls.npz")
    if not refresh and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            return z["mbx"], z["mby"], z["rows"]
    t = atd.load()
    b = atd.load(source="broadcast")
    rows = align_chunks(t, b)
    if use_players:
        off, NO = piece_offsets(t, b, rows)
    else:
        NO = np.iinfo(np.int64).min
        off = np.full(len(b["bx"]), NO, np.int64)
        for s, e, o, med, ov, uq, a in rows:
            if a:
                off[s:e] = o
    n = len(t["bx"])
    mbx = np.full(n, np.nan, np.float32)
    mby = np.full(n, np.nan, np.float32)
    idx = np.flatnonzero((off != NO) & np.isfinite(b["bx"]))
    tgt = idx + off[idx]
    keep = (tgt >= 0) & (tgt < n)
    mbx[tgt[keep]] = b["bx"][idx[keep]]
    mby[tgt[keep]] = b["by"][idx[keep]]
    arr = np.array([(s, e, off, med if np.isfinite(med) else -1, ov,
                     uniq if np.isfinite(uniq) else -1, acc) for s, e, off, med, ov, uniq, acc in rows],
                   dtype=np.float64)
    os.makedirs(atd.CACHE_DIR, exist_ok=True)
    np.savez(cache, mbx=mbx, mby=mby, rows=arr)
    return mbx, mby, arr


def report():
    from src import proxy_eval
    mbx, mby, rows = mapped_ball(refresh=True)
    t = atd.load()
    g = proxy_eval.graded_mask(len(t["bx"]))
    acc = rows[:, 6] > 0
    print("broadcast chunks: {} ; with enough ball to try: {} ; accepted: {}".format(
        len(rows), int((rows[:, 4] >= MIN_OVERLAP).sum()), int(acc.sum())))
    if acc.any():
        offs = rows[acc, 2]
        print("accepted offsets (tactical - broadcast frames): p10/p50/p90 {}".format(
            np.percentile(offs, [10, 50, 90]).round(0)))
        print("  median ball distance at accepted offsets: p50 {:.2f} m, p90 {:.2f} m".format(
            np.percentile(rows[acc, 3], 50), np.percentile(rows[acc, 3], 90)))
        steps = np.diff(offs)
        print("  consecutive accepted chunks with the same offset (+/-2 frames): {:.2f}".format(
            np.mean(np.abs(steps) <= 2)))
    b = atd.load(source="broadcast")
    print("note: a second pass maps 2 s pieces by player positions (see module docstring)")
    tseen = np.isfinite(t["bx"])
    fill = g & ~tseen & np.isfinite(mbx)
    print("graded frames: tactical ball missing {:.3f}; filled from broadcast {:.3f} of missing".format(
        np.mean(~tseen[g]), fill.sum() / max((g & ~tseen).sum(), 1)))
    both = g & tseen & np.isfinite(mbx)
    dist = np.hypot(mbx[both] - t["bx"][both], mby[both] - t["by"][both])
    print("where both see it after mapping: {} frames, distance p50 {:.2f} m, p90 {:.2f} m, "
          "share > 3 m {:.3f}".format(int(both.sum()), *np.percentile(dist, [50, 90]),
                                    np.mean(dist > 3)))


if __name__ == "__main__":
    report()
