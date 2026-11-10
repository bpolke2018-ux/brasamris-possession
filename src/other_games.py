"""Held-out robustness test on Metrica Sports' public sample games 1, 2 and 3.

    python -m src.other_games                          # v7's tag-free core on games 1-3
    python -m src.other_games --set still_nats=0       # the same with a change undone

Data: github.com/metrica-sports/sample-data, downloaded by Brandon to ../other_games (outside the
repo, never committed). Tracking at 25 fps in normalised coordinates; hand-coded event data.
Games 1-2: CSV tracking per team and CSV events. Game 3: EPTS text tracking (player slots remapped
at substitutions, per the metadata) and JSON events, converted to the same event table.

Ground truth here is DERIVED from Metrica's event data (rules in notes/other_games.md and in
gt_labels below). It is a different labelling convention from the hackathon's, so absolute numbers
are not comparable with BAR-ATH; what matters is whether a change helps or hurts on games the
model has never seen. These games are used to MEASURE robustness only, never to choose BAR-ATH
settings. Prints aggregates only.

Metrica has no Smart Tagging, so the set-piece evidence, the tagged-restart penalty and the
pre-kickoff rule are off (pre-kickoff frames are ungraded here anyway); everything else is v7
(the kickoff-formation option, off in v7, can be switched on with --set kickoff_nats=..).
The pipeline reads the match layout from module globals (half blocks, pitch), so this harness
points those at the Metrica game it is running; src/run.py on BAR-ATH is unaffected.
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd

from src import atd, context, features, model, run

DATA = os.path.join(os.path.dirname(atd.ROOT), "other_games", "data")
PITCH = (105.0, 68.0)           # Metrica sample games: normalised to a 105 x 68 m pitch
FPS = 25
sys.path.insert(0, os.path.join(os.path.dirname(atd.ROOT), "more-than-a-hack-2027", "evaluation"))
import possession_metrics as pm  # noqa: E402


def _read_side(path):
    t = pd.read_csv(path, skiprows=2)
    cols = list(t.columns)
    names = [c for c in cols[3:] if not c.startswith("Unnamed") and c != "Ball"]
    xy = np.stack([np.stack([t[c].to_numpy(np.float64),
                             t[cols[cols.index(c) + 1]].to_numpy(np.float64)], 1) for c in names], 1)
    i = cols.index("Ball")
    ball = np.stack([t["Ball"].to_numpy(np.float64), t[cols[i + 1]].to_numpy(np.float64)], 1)
    return t["Frame"].to_numpy(), t["Period"].to_numpy(), xy, ball


def _speeds(px, py, per):
    """No tracker speed field: per-player displacement speed, smoothed over +/- 2 frames."""
    step = np.full(px.shape, np.nan, np.float32)
    step[1:] = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * FPS
    step[1:][per[1:] != per[:-1]] = np.nan
    return np.stack([features.nan_smooth(step[:, k], 2) for k in range(px.shape[1])], 1)


def _parse_epts(g):
    """Game 3: EPTS text tracking; the 22 slots map to different players in each segment."""
    import xml.etree.ElementTree as ET
    d = os.path.join(DATA, "Sample_Game_{}".format(g))
    root = ET.parse(os.path.join(d, "Sample_Game_{}_metadata.xml".format(g))).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]          # first team = home
    players = [p.get("id") for p in root.iter("Player")]
    team = np.array([team_ids.index(p.get("teamId")) for p in root.iter("Player")], np.int8)
    chan = {c.get("id"): players.index(c.get("playerId")) for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        slots = [chan[r] for r in refs[0::2]]                     # x channel of each pair
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))
    specs.sort()
    n = max(e for _, e, _ in specs)
    px = np.full((n, len(players)), np.nan, np.float32)
    py = np.full((n, len(players)), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    si = 0
    with open(os.path.join(d, "Sample_Game_{}_tracking.txt".format(g))) as handle:
        for line in handle:
            parts = line.rstrip().split(":")
            fr = int(parts[0])
            while specs[si][1] < fr:
                si += 1
            slots = specs[si][2]
            for k, grp in enumerate(parts[1].split(";")):
                x, y = grp.split(",")[:2]
                if x != "NaN" and y != "NaN":
                    px[fr - 1, slots[k]], py[fr - 1, slots[k]] = float(x), float(y)
            b = parts[2].split(",")
            if b[0] != "NaN" and b[1] != "NaN":
                bx[fr - 1], by[fr - 1] = float(b[0]), float(b[1])
    ev = events(g)
    p2 = int(ev.loc[ev["Period"] == 2, "Start Frame"].min()) - 1
    per = np.where(np.arange(n) < p2, 1, 2)
    return px, py, team, bx, by, per


def load_tracking(g):
    """Arrays in the src/atd.py layout plus `period`; cached under cache/ (gitignored)."""
    path = os.path.join(atd.CACHE_DIR, "metrica_g{}.npz".format(g))
    if os.path.isfile(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    if g == 3:
        px, py, team, bx, by, per = _parse_epts(g)
        assert np.nanmax(px) < 1.5 and np.nanmin(px) > -0.5, "expected normalised coordinates"
        px, py = px * PITCH[0], py * PITCH[1]
        out = dict(px=px, py=py, ps=_speeds(px, py, per).astype(np.float32), team=team,
                   bx=bx * PITCH[0], by=by * PITCH[1], bs=np.full(len(px), np.nan, np.float32),
                   period=per.astype(np.int8))
        os.makedirs(atd.CACHE_DIR, exist_ok=True)
        np.savez(path, **out)
        return out
    d = os.path.join(DATA, "Sample_Game_{}".format(g))
    fh, per, xh, ball = _read_side(os.path.join(d, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(g)))
    fa, _, xa, _ = _read_side(os.path.join(d, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(g)))
    assert np.array_equal(fh, fa) and np.array_equal(fh, np.arange(1, len(fh) + 1)), "frame index"
    xy = np.concatenate([xh, xa], 1)
    px = (xy[:, :, 0] * PITCH[0]).astype(np.float32)
    py = (xy[:, :, 1] * PITCH[1]).astype(np.float32)
    team = np.array([0] * xh.shape[1] + [1] * xa.shape[1], np.int8)
    ps = _speeds(px, py, per)
    out = dict(px=px, py=py, ps=ps.astype(np.float32), team=team,
               bx=(ball[:, 0] * PITCH[0]).astype(np.float32),
               by=(ball[:, 1] * PITCH[1]).astype(np.float32),
               bs=np.full(len(px), np.nan, np.float32), period=per.astype(np.int8))
    os.makedirs(atd.CACHE_DIR, exist_ok=True)
    np.savez(path, **out)
    return out


STOP_AT_END = ("OFFSIDE", "HAND BALL", "END HALF", "REFEREE HIT")


def events(g):
    """Event table with columns Team (Home/Away), Type, Subtype, Period, Start Frame, End Frame."""
    d = os.path.join(DATA, "Sample_Game_{}".format(g))
    if g != 3:
        return pd.read_csv(os.path.join(d, "Sample_Game_{}_RawEventsData.csv".format(g)))
    import json
    with open(os.path.join(d, "Sample_Game_{}_events.json".format(g))) as handle:
        data = json.load(handle)["data"]
    home = "FIFATMA"            # first team in the metadata (Team A), as in _parse_epts

    def sub(e):
        s = e["subtypes"]
        if s is None:
            return np.nan
        return "-".join(t["name"] for t in (s if isinstance(s, list) else [s]))
    return pd.DataFrame({
        "Team": ["Home" if (e["team"] or {}).get("id") == home else "Away" for e in data],
        "Type": [e["type"]["name"] for e in data],
        "Subtype": [sub(e) for e in data],
        "Period": [e["period"] for e in data],
        "Start Frame": [e["start"]["frame"] for e in data],
        "End Frame": [e["end"]["frame"] for e in data]})


def gt_labels(g, n):
    """(labels H/A/D/X per frame, blocks [(name, first, last)]) from the event data.

    Rules (football definitions, fixed before looking at any result):
      * graded: from each period's first KICK OFF to the last event end of that period; else X.
      * the team in possession is the team of the latest PASS / CARRY / SHOT / SET PIECE /
        RECOVERY (a possession changes when the other team gains control; CHALLENGE and BALL
        LOST alone do not change it). CARRY exists only in game 3's events.
      * dead ball from a stoppage -- BALL OUT (its end), a SHOT that goes out or is a goal (its
        end), FAULT RECEIVED (its start), a BALL LOST by offside / hand ball / end of half /
        referee hit (its end) -- to the next restart: a SET PIECE or a GOAL KICK pass. A stoppage
        is ignored if live play (a non-restart on-ball event) comes first (e.g. advantage).
    """
    ev = events(g).sort_values("Start Frame", kind="stable").reset_index(drop=True)
    lab = np.full(n, "X", dtype="<U1")
    blocks = []
    for p in sorted(ev["Period"].unique()):
        e = ev[ev["Period"] == p]
        ko = e[(e["Type"] == "SET PIECE") & e["Subtype"].fillna("").str.startswith("KICK OFF")]
        first = int(ko["Start Frame"].min()) - 1
        last = int(e["End Frame"].max()) - 1
        points = []          # (frame index, label, kind) kind: "own" possession, "restart", "stop"
        for _, r in e.iterrows():
            typ, sub = r["Type"], str(r["Subtype"]) if pd.notna(r["Subtype"]) else ""
            team = "H" if r["Team"] == "Home" else "A"
            s, t = int(r["Start Frame"]) - 1, int(r["End Frame"]) - 1
            restart = typ == "SET PIECE" or (typ == "PASS" and sub == "GOAL KICK")
            if typ in ("PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY"):
                points.append((s, team, "restart" if restart else "own"))
            if typ == "BALL OUT" or (typ == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
                points.append((t, "D", "stop"))
            elif typ == "FAULT RECEIVED":
                points.append((s, "D", "stop"))
            elif typ == "BALL LOST" and any(k in sub for k in STOP_AT_END):
                points.append((t, "D", "stop"))
        points.sort(key=lambda q: q[0])
        keep = []
        for i, q in enumerate(points):
            if q[2] == "stop":
                nxt = next((z for z in points[i + 1:] if z[2] != "stop"), None)
                if nxt is None or nxt[2] != "restart":
                    continue
            keep.append(q)
        # each change point holds until the next one; before the first, its label (the kicker)
        seg = np.full(last - first + 1, keep[0][1] if keep else "D", dtype="<U1")
        ends = [q[0] for q in keep[1:]] + [last + 1]
        for (f0, label, _), f1 in zip(keep, ends):
            lo, hi = max(f0, first), min(f1, last + 1)
            if hi > lo:
                seg[lo - first:hi - first] = label
        lab[first:last + 1] = seg
        blocks.append(("period {}".format(p), first, last))
    return lab, blocks


def point_at(blocks):
    """Point the pipeline's module globals (pitch, half blocks) at a Metrica game."""
    features.PITCH = context.PITCH = PITCH
    run.half_video_blocks = context.half_video_blocks = lambda game: blocks
    model.CONFIG["sp_restart_pen"] = 0.0     # no Smart Tagging here
    model.CONFIG["prekick_nats"] = 0.0


def predict(a, blocks, learned=None):
    """v7's tag-free core on one game (learned: weights from src/learned.py, used as
    model.CONFIG["learned"] says)."""
    from src import learned as lrn
    point_at(blocks)
    f = features.compute(a)
    n = len(a["bx"])
    team_llr = model.team_evidence(f, model.CONFIG)
    kickoff = model.kickoff_formation(a, f) if model.CONFIG["kickoff_nats"] else None
    lmode = model.CONFIG["learned"] if learned is not None else "off"
    dead = model.dead_evidence(f, np.zeros(n, bool), kickoff, None,
                               lrn.dead_core(f, learned) if lmode in ("dead", "both") else None)
    extra = 0.0
    if model.CONFIG["context"]:
        extra = model.CONFIG["context_scale"] * context.context_llr(a, f)
    if lmode in ("team", "both"):
        team_llr = lrn.team_core(a, f, learned)
        extra = 0.0
    still = model.ball_still(f)
    if model.CONFIG["hold_nats"]:
        still = still | model.throw_hold(f)
    if kickoff is not None:
        still = still | kickoff
    still = still.astype(np.float64)
    k = run.calibrate(team_llr, dead, n, extra, still=still)
    return run.decode(model.emissions(team_llr, dead, k, extra, still), n), k


KEYS = ["spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive",
        "n_gt_spells", "n_pred_spells", "gt_mean_spell_duration_s", "mean_spell_duration_s"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--games", default="1,2,3")
    parser.add_argument("--name", default="v7 core")
    parser.add_argument("--save", help="tag: save predicted labels to cache/other_games/<tag>_g<g>.npy "
                                       "(labels only; for src/other_games_boot.py)")
    args = parser.parse_args()
    for item in args.set:
        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        if key in model.CONFIG:
            model.CONFIG[key] = value
        elif key.isupper() and hasattr(model, key):
            setattr(model, key, value)
        else:
            raise KeyError(key)
    rows = []
    lm = None
    if model.CONFIG["learned"] != "off":
        from src import learned
        lm = learned.load()
        print("NOTE: learned evidence (v9+) uses the all-games model here, i.e. IN-SAMPLE on these "
              "games. The honest Metrica score is leave-one-game-out: python -m src.learned "
              "--dead-scale --no-missing --restart-live (saves learn_dead_nm_rl_g*.npy).")
    for g in [int(x) for x in args.games.split(",")]:
        a = load_tracking(g)
        n = len(a["bx"])
        gt, blocks = gt_labels(g, n)
        pred, k = predict(a, blocks, learned=lm)
        if args.save:
            out = os.path.join(atd.CACHE_DIR, "other_games")
            os.makedirs(out, exist_ok=True)
            np.save(os.path.join(out, "{}_g{}.npy".format(args.save, g)), pred)
            np.save(os.path.join(out, "gt_g{}.npy".format(g)), gt)
        r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
        graded = gt != "X"
        r["gt_share_D"] = float(np.mean(gt[graded] == "D"))
        r["pred_share_D"] = float(np.mean(pred[graded] == "D"))
        r["ball_seen"] = float(np.mean(np.isfinite(a["bx"][graded])))
        r["k_team"] = k
        rows.append((g, r))
    cols = KEYS + ["gt_share_D", "pred_share_D", "ball_seen", "k_team"]
    print("| run | game | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for g, r in rows:
        print("| {} | {} | {} |".format(args.name, g, " | ".join(
            str(r[c]) if isinstance(r[c], (int, np.integer)) else "{:.3f}".format(r[c]) for c in cols)))


if __name__ == "__main__":
    main()
