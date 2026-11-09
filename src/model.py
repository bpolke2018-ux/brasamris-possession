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
    # Set-piece windows as D evidence: "full" 35 s window (v1-v3) or "start" (-10 s..+3 s).
    "sp_dead_window": "full",
    # Restarting team from the set-piece Team tag: H/A evidence over the first sp_team_s seconds
    # after the tag start, sp_team_nats per frame. The tag is automatic, so moderate: 0.02 nats
    # per frame is ~5 nats over 10 s -- enough to pick the side at a restart, not enough to hold
    # a possession against clear ball evidence. 0 = off (v1-v3).
    "sp_team_nats": 0.0,
    "sp_team_s": 10.0,
    # Minimum controlled possession, seconds. CHALLENGE.md: a touch, deflection or won header
    # without control is not a change of possession, and "keeps the ball for a few seconds" is
    # the suggested proxy for control. Live segments shorter than this are given back to the
    # surrounding live team (or merged with the following one after a stoppage). 0 = off.
    "min_possession_s": 0.0,
    # Ball lost next to a line: while the ball is missing, if it was last seen within
    # line_dist_m of a touchline or goal line (and no more than line_max_age_s ago), add
    # line_nats per frame of D evidence. Football reasoning: balls leave play over a line, and the
    # tracker often loses them exactly there (out of the camera's pitch area). 0 = off.
    "line_nats": 0.0,
    "line_dist_m": 2.0,
    "line_max_age_s": 20.0,
    # Ball-control evidence says WHICH team, not WHETHER the ball is live: at a free kick or goal
    # kick a player stands next to the placed ball. d_team_share in [0, 1] gives the D state
    # that share of the best live state's control score (|team llr| / 2). 0 = v1-v3 (control
    # evidence also votes for live), 1 = control evidence is neutral between live and D.
    "d_team_share": 1.0,      # v4
    # Where d_team_share applies: "all" frames, or only "still" frames (ball_still below): a
    # dribbled ball next to a player is live play; a motionless one next to a player may be placed.
    "d_team_where": "still",  # v4
    # Placed ball: ball seen inside the pitch and nearly motionless (every smoothed speed within
    # +/- still_half_s below still_speed). In open play the ball is almost never still for a
    # second; at a restart it is placed and waits for the kick. still_nats per frame for D.
    # Aggregate check (src/restart_report.py): v3 calls 95% of such stretches live, and 42% of
    # them sit inside a SET PIECES window. 0 = off (v1-v3).
    "still_nats": 0.1,        # v4
    "still_speed": 1.5,
    "still_half_s": 0.5,
    # X abstention (macroF1 tie-break only; spellF1 is unchanged by construction). The scorer
    # forward-fills X with the previous label before matching possessions, so an X on a frame
    # whose label equals the previous frame's leaves every possession exactly as it was. On
    # macroF1 a wrong label costs a false negative for the true class AND a false positive for
    # the predicted one; an X costs only the false negative. Abstaining on a frame we would get
    # right with probability p changes F1_k by about (F1_k - 2p) / S_k, so it pays when
    # p < F1_k / 2. With F1 ~ 2/3 that is p < 1/3. p is the HMM's own posterior (forward-backward),
    # which is overconfident (frames are not independent), so this abstains rarely. 0 = off.
    "abstain_below": 0.0,
    # Restarting team from the set-piece Team tag, as a constraint on the restart itself rather
    # than as evidence on fixed frames (e5): inside [tag start - before, tag start + after], a
    # D -> live transition to the NON-tagged team costs sp_restart_pen nats. Later turnovers are
    # untouched. Football: the first possession after a restart belongs to the team taking it.
    # 0 = off.
    "sp_restart_pen": 5.0,    # v5
    "sp_restart_before_s": 10.0,
    "sp_restart_after_s": 20.0,
    # Throw-in hold: ball seen beyond a touchline (any amount), slower than 3 m/s, a player
    # within 2.5 m -- the thrower holding it. The ball is out of play by definition; play
    # restarts when it comes back in. hold_nats per frame for D, and ball control is neutral
    # there (like a placed ball). Aggregate: v5 calls only 35% of these frames D. 0 = off.
    "hold_nats": 0.15,        # v6 (same weight as K_BALL_OUT)
    # Borrow the broadcast tracking's ball where the tactical ball is missing (src/broadcast.py,
    # aligned by code). "off", "balls" (ball-aligned chunks only) or "players" (+ player-aligned
    # 2 s pieces, less reliable).
    "broadcast_ball": "off",
    # Kickoff formation: at most one tracked player outside his own half, at least 16 tracked,
    # held >= 1 s (episodes merged over 2 s gaps). The ball is dead until the kick, by the Laws.
    # Aggregate check: 7 such episodes, covering all 5 graded kickoff tags; v6 calls only half of
    # their frames D. kickoff_nats per frame for D, and ball control is neutral there. 0 = off.
    "kickoff_nats": 0.0,
    # Before the half's kickoff: by the Laws the ball is not in play until the kick, so graded
    # frames from the start of a half's grading up to its Kick Off (Start) Smart Tagging tag are
    # D. prekick_nats per frame of D evidence (50 = a hard rule; ball control cannot outvote it).
    # The tag time base is checked against both kickoffs (DATA.md). 0 = off (v1-v6).
    "prekick_nats": 50.0,     # v7
    # Duration-aware states (HSMM-style, by state chaining). A plain HMM makes every stretch's
    # length geometric: the single most likely possession lasts one frame. Replacing a state by a
    # chain of r identical sub-states (each left with prob r / mean) makes its length a negative
    # binomial with the same mean and a mode above zero, so a one-touch flicker becomes unlikely
    # a priori while long possessions keep their cost. live_phases for H and A, dead_phases for D.
    # 1 = plain HMM (v1-v7).
    "live_phases": 1,
    "dead_phases": 1,
}


def ball_lost_near_line(f, cfg=CONFIG):
    """Bool per frame: ball missing, last seen within line_dist_m of a line, recently."""
    from src.features import PITCH
    L, W = PITCH
    bx, by, seen = f["bx"], f["by"], f["ball_seen"]
    n = len(seen)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    ok = (~seen) & (last >= 0)
    lx = np.where(last >= 0, bx[np.maximum(last, 0)], np.nan)
    ly = np.where(last >= 0, by[np.maximum(last, 0)], np.nan)
    dist = np.fmin(np.fmin(np.abs(lx), np.abs(L - lx)), np.fmin(np.abs(ly), np.abs(W - ly)))
    age = np.arange(n) - last
    return ok & (np.nan_to_num(dist, nan=99) < cfg["line_dist_m"]) & (age < cfg["line_max_age_s"] * FPS)


def enforce_min_possession(seg_labels, min_frames):
    """Relabel H/A runs shorter than min_frames inside one half. Deterministic, left to right.

    A short run between two live runs of the same team is a touch inside that team's possession;
    a short run next to a D stretch is assigned to the neighbouring live run (previous if live,
    otherwise next). Runs that cannot be merged (live neighbours of both teams, or only D around)
    are left alone: those are genuine quick changes or restarts.
    """
    if min_frames <= 0:
        return seg_labels
    lab = seg_labels.copy()
    for _ in range(3):                       # a merge can create a new short run; repeat a little
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(lab)]))
        runs = list(zip(lab[starts], starts, ends))
        changed = False
        for i, (v, s, e) in enumerate(runs):
            if v not in ("H", "A") or e - s >= min_frames:
                continue
            prev = runs[i - 1][0] if i > 0 else None
            nxt = runs[i + 1][0] if i + 1 < len(runs) else None
            if prev in ("H", "A") and nxt in ("H", "A"):
                target = prev if prev == nxt else None
            elif prev in ("H", "A"):
                target = prev
            elif nxt in ("H", "A"):
                target = nxt
            else:
                target = None
            if target is not None and target != v:
                lab[s:e] = target
                changed = True
        if not changed:
            break
    return lab


def transition_logs():
    leave = 1.0 / (POSSESSION_S * FPS)
    to_d = leave * SHARE_ENDING_IN_STOPPAGE
    to_other = leave - to_d
    restart = 1.0 / (DEAD_BALL_S * FPS)
    T = np.array([[1 - leave, to_other, to_d],
                  [to_other, 1 - leave, to_d],
                  [restart / 2, restart / 2, 1 - restart]])
    return np.log(T)


def chain_layout():
    """(parent state of each sub-state, first sub-state of H, A, D, last sub-state of D)."""
    rl, rd = CONFIG["live_phases"], CONFIG["dead_phases"]
    parent = np.array([H] * rl + [A] * rl + [D] * rd, np.int8)
    return parent, (0, rl, 2 * rl), 2 * rl + rd - 1


def chain_transition_logs():
    """(S, S) log transitions of the chained HMM; same means and exit splits as transition_logs."""
    rl, rd = CONFIG["live_phases"], CONFIG["dead_phases"]
    parent, (h1, a1, d1), dlast = chain_layout()
    S = len(parent)
    T = np.zeros((S, S))
    p_live = rl / (POSSESSION_S * FPS)
    p_dead = rd / (DEAD_BALL_S * FPS)
    for team, first, other in ((H, h1, a1), (A, a1, h1)):
        for j in range(rl):
            i = first + j
            T[i, i] = 1 - p_live
            if j < rl - 1:
                T[i, i + 1] = p_live
            else:
                T[i, other] = p_live * (1 - SHARE_ENDING_IN_STOPPAGE)
                T[i, d1] = p_live * SHARE_ENDING_IN_STOPPAGE
    for j in range(rd):
        i = d1 + j
        T[i, i] = 1 - p_dead
        if j < rd - 1:
            T[i, i + 1] = p_dead
        else:
            T[i, h1] = T[i, a1] = p_dead / 2
    with np.errstate(divide="ignore"):
        return np.log(T)


def viterbi_chain(E, restart_pen=None):
    """Viterbi over the chained states; returns the parent-state path (H/A/D codes).

    E is the usual (n, 3) emission; every sub-state copies its parent's column. restart_pen as in
    viterbi(): extra cost of the D -> H / D -> A transition into frame i (last D sub-state into
    the first H / A sub-state).
    """
    parent, (h1, a1, _), dlast = chain_layout()
    logT = chain_transition_logs()
    n, S = len(E), len(parent)
    Ex = E[:, parent]
    back = np.zeros((n, S), np.int16)
    score = Ex[0].copy()
    for i in range(1, n):
        cand = score[:, None] + logT
        if restart_pen is not None:
            cand[dlast, h1] -= restart_pen[i, 0]
            cand[dlast, a1] -= restart_pen[i, 1]
        b = np.argmax(cand, axis=0)
        back[i] = b
        score = cand[b, np.arange(S)] + Ex[i]
    path = np.empty(n, np.int16)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return parent[path]


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


def dead_evidence(f, set_piece_mask, kickoff=None, prekick=None):
    """D log-score per frame relative to the live states."""
    seen_in = f["ball_seen"] & ~f["ball_out"]
    z = np.clip((WALK_SPEED - np.nan_to_num(f["mean_speed"], nan=WALK_SPEED)) / SPEED_SCALE, -3, 3)
    out = K_SPEED * z
    out += K_BALL_MISSING * ~f["ball_seen"]
    out += K_BALL_OUT * f["ball_out"]
    out -= K_BALL_FAST * (seen_in & (np.nan_to_num(f["ball_speed"]) > BALL_FAST))
    out += K_SET_PIECE * set_piece_mask
    if CONFIG["line_nats"]:
        out += CONFIG["line_nats"] * ball_lost_near_line(f)
    if CONFIG["still_nats"]:
        out += CONFIG["still_nats"] * ball_still(f)
    if CONFIG["hold_nats"]:
        out += CONFIG["hold_nats"] * throw_hold(f)
    if CONFIG["kickoff_nats"] and kickoff is not None:
        out += CONFIG["kickoff_nats"] * kickoff
    if CONFIG["prekick_nats"] and prekick is not None:
        out += CONFIG["prekick_nats"] * prekick
    return out


def throw_hold(f):
    """Bool per frame: ball held just beyond a touchline by a nearby player."""
    from src.features import PITCH
    L, W = PITCH
    bx, by = f["bx"], f["by"]
    with np.errstate(invalid="ignore"):
        beyond = f["ball_seen"] & ((by < 0) | (by > W)) & (bx > 0) & (bx < L)
    dmin = np.nan_to_num(np.fmin(f["d_home"], f["d_away"]), nan=99.0)
    return beyond & (np.nan_to_num(f["ball_speed"], nan=99) < 3.0) & (dmin < 2.5)


def kickoff_formation(a, f):
    """Bool per frame: both teams in their own halves (<= 1 player across), >= 16 tracked, for
    episodes lasting >= 1 s. Attack direction per half from src/context.py (unsupervised)."""
    import warnings
    from src import context
    from src.features import PITCH
    L = PITCH[0]
    _, direction = context.shape_features(a, f)
    side = np.where(direction > 0, 1.0, -1.0)[:, None]
    px, team = a["px"], a["team"]
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        hx = (px[:, team == 0] - L / 2) * side          # home's own half: < 0
        ax = (px[:, team == 1] - L / 2) * side          # away's own half: > 0
        wrong = np.sum(hx > 1.0, 1) + np.sum(ax < -1.0, 1)
    tracked = np.isfinite(px).sum(1)
    m = (wrong <= 1) & (tracked >= 16)
    e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    out = np.zeros(len(m), bool)
    eps = []
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        if eps and s - eps[-1][1] < 2 * FPS:
            eps[-1][1] = t
        else:
            eps.append([s, t])
    for s, t in eps:
        if t - s >= FPS:
            out[s:t] = True
    return out


def ball_still(f, cfg=CONFIG):
    """Bool per frame: ball seen in the pitch and below still_speed over +/- still_half_s."""
    from numpy.lib.stride_tricks import sliding_window_view
    h = int(cfg["still_half_s"] * FPS)
    ok = f["ball_seen"] & ~f["ball_out"] & (np.nan_to_num(f["ball_speed"], nan=99) < cfg["still_speed"])
    padded = np.concatenate((np.zeros(h, bool), ok, np.zeros(h, bool)))
    return sliding_window_view(padded, 2 * h + 1).all(axis=1)


def emissions(team_llr, dead, k_team=None, extra_llr=0.0, still=None):
    """(n, 3) log-scores. Only differences between states matter.

    team_llr is scaled by K_TEAM; extra_llr (e.g. shape context) is already in nats and is not.
    """
    k = K_TEAM if k_team is None else k_team
    total = k * team_llr + extra_llr
    E = np.zeros((len(dead), 3), np.float64)
    E[:, H] = total / 2
    E[:, A] = -total / 2
    share = CONFIG["d_team_share"]
    if share and CONFIG["d_team_where"] == "still":
        share = share * still
    E[:, D] = dead + share * np.abs(total) / 2
    return E


def posteriors(E, logT):
    """(n, 3) state marginals by scaled forward-backward. Uniform initial distribution."""
    n = len(E)
    T = np.exp(logT)
    lik = np.exp(E - E.max(axis=1, keepdims=True))
    alpha = np.empty((n, 3))
    a = lik[0] / lik[0].sum()
    alpha[0] = a
    for i in range(1, n):
        a = (a @ T) * lik[i]
        a /= a.sum()
        alpha[i] = a
    post = np.empty((n, 3))
    b = np.ones(3)
    post[-1] = alpha[-1]
    for i in range(n - 1, 0, -1):
        b = T @ (lik[i] * b)
        b /= b.sum()
        p = alpha[i - 1] * b
        post[i - 1] = p / p.sum()
    return post


def abstain(seg_path, post, below):
    """Bool mask: frames to write as X. Never the first frame of a run, so the scorer's
    forward fill restores exactly seg_path."""
    if below <= 0:
        return np.zeros(len(seg_path), bool)
    conf = post[np.arange(len(seg_path)), seg_path]
    same = np.concatenate(([False], seg_path[1:] == seg_path[:-1]))
    return same & (conf < below)


def viterbi(E, logT, restart_pen=None):
    """Most likely state path for one contiguous block. Uniform initial distribution.

    restart_pen: optional (n, 2) extra log-cost of the D -> H and D -> A transitions into frame i.
    """
    n = len(E)
    back = np.zeros((n, 3), np.int8)
    score = E[0].copy()
    t = logT.tolist()
    pen = restart_pen.tolist() if restart_pen is not None else None
    for i in range(1, n):
        s0, s1, s2 = score
        new = np.empty(3)
        for j in range(3):
            c0, c1, c2 = s0 + t[0][j], s1 + t[1][j], s2 + t[2][j]
            if pen is not None and j < 2:
                c2 -= pen[i][j]
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
