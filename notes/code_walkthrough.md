# Code walkthrough: one frame's journey

This note follows a single video frame from the raw tracking file to the letter (`H`, `A` or `D`)
we write for it in `submissions/v9.csv`. Each stop shows a short excerpt of the real code, what it
does, and why it exists: which football problem or metric issue it solves, with the experiment
number from `notes/experiments.md`.

The pipeline in one line:

```
atd.load → features.compute → context.context_llr ┐
                                model.team_evidence ├→ model.emissions → model.viterbi (per half) → letters
          model.dead_evidence (+ ball_still, hold) ┘        ↑ transition_logs, restart_penalty
```

Everything is started by `src/run.py main()`.

---

## 0. Three ideas you need first: log-probabilities, nats, Viterbi

**Log-probabilities.** The model multiplies many small probabilities: one per frame, 80,000
frames per half. Multiplying that many numbers below 1 underflows to 0 on a computer, so we take
logarithms and **add** instead. `log(p × q) = log p + log q`.

**Nats.** A *nat* is the unit of a natural-log quantity. Evidence of "+1 nat for H" means the
frame makes H `e¹ ≈ 2.7` times more likely than before. Useful anchors:

| nats | odds |
|---|---|
| 0.1 | 1.1 : 1 (a nudge) |
| 1 | 2.7 : 1 |
| 2.3 | 10 : 1 |
| 5 | 150 : 1 (the v5 restart penalty) |
| 6.15 | 470 : 1 (the cost of a possession switch in one frame, below) |

Because only *differences* between states matter, we can set any one state's score to 0 and
express the others relative to it.

**Viterbi** finds the single most likely *sequence* of states, not the most likely state frame by
frame. Each frame adds its emission score (how well the frame fits each state), and each move from
one state to another adds a transition score (how plausible that change is). Viterbi keeps, for
every state, the best total score of any path ending there, and remembers where it came from. At
the end it walks back along those pointers.

### Worked example: 3 frames, 3 states (made-up numbers)

Transitions: stay = 0.9 (log −0.105), each of the two changes = 0.05 (log −3.0).

Emissions (log-scores, D fixed at 0):

| frame | H | A | D | what is going on |
|---|---|---|---|---|
| 1 | +0.5 | −0.5 | 0 | home player on the ball |
| 2 | −0.2 | +0.2 | 0 | an away player is briefly closer (a touch, a deflection) |
| 3 | +0.5 | −0.5 | 0 | home player on the ball again |

A **frame-by-frame** choice picks H, A, H: three possessions, two of them wrong. On spellF1 that
is a disaster.

**Viterbi.** The best score of a path ending in each state:

- Frame 1: H 0.5, A −0.5, D 0.
- Frame 2:
  - H: best of (H→H 0.5−0.105 = 0.395, A→H −0.5−3 = −3.5, D→H 0−3 = −3.0) is 0.395, from H. Adding the emission −0.2 gives **0.195** (from H).
  - A: best of (H→A 0.5−3 = −2.5, A→A −0.5−0.105 = −0.605, D→A −3.0) is −0.605, from A. Adding +0.2 gives **−0.405** (from A).
  - D: best of (−2.5, −3.5, D→D 0−0.105) is −0.105, from D. Adding 0 gives **−0.105** (from D).
- Frame 3:
  - H: best of (0.195−0.105 = 0.09, −0.405−3, −0.105−3) is 0.09, from H. Adding +0.5 gives **0.59**.
  - A: best of (0.195−3, −0.405−0.105 = −0.51, −0.105−3) is −0.51, from A. Adding −0.5 gives −1.01.
  - D: best of (0.195−3, −0.405−3, −0.105−0.105 = −0.21) is −0.21, from D. Adding 0 gives −0.21.
- The best end is **H (0.59)**. Walking back: frame 3 H came from H, frame 2 H came from H, so
  the answer is **H H H**.

The A blip offered +0.4 nats in total, but crossing to A and back costs 3 + 3 = 6 nats. The
sequence stays with H. This is how a touch without control gets ignored (CHALLENGE.md: "a simple
touch of the ball without control" is not a change).

Our real numbers are much stickier, because there are 25 frames per second: a switch costs
6.15 nats against staying (§9).

---

## 1. `src/atd.py`: raw text → arrays

```python
def parse(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    ...
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            ...
            for slot, g in enumerate(parts[1].split(";")[:k]):
                if g and g[0] != "N":
                    px[f, slot], py[f, slot], ps[f, slot] = _group(g)
            b = parts[2]
            if b and b[0] != "N":
                bx[f], by[f], bs[f] = _group(b)
    px *= length
    py *= width
```

- Each line of the file is one frame: `frame_id : players ; … : ball`. We split it and store
  x, y and speed per *track slot* (46 slots) plus the ball. Missing values stay `NaN`.
- Coordinates arrive as 0–1 fractions of the pitch, so we multiply by 105.3 × 68.0 m.
- `load()` caches the result in `cache/*.npz`, which turns a 1–2 minute parse into one second.
- **Why:** everything after this works in metres and arrays, never in text. Slots are *not*
  people: the tracker switches identities, so later code only uses team-level aggregates.

## 2. `src/features.py`: per-frame quantities

```python
        dist = np.hypot(px - bx[:, None], py - by[:, None])          # (n, k), NaN if either missing
        d_home = np.nanmin(np.where(team[None, :] == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == 1, dist, np.nan), axis=1)
        ...
        speed = np.clip(speed, 0, MAX_PLAYER_SPEED)
        mean_speed = np.nanmean(speed, axis=1)

    # Ball speed from consecutive sightings, lightly smoothed (+/- 2 frames = 0.2 s window).
    step = np.full(len(bx), np.nan, np.float32)
    step[1:] = np.hypot(np.diff(bx), np.diff(by)) * FPS
    ball_speed = nan_smooth(step, 2)

    seen = np.isfinite(bx)
    ball_out = seen & ((bx < -1) | (bx > L + 1) | (by < -1) | (by > W + 1))
```

- `d_home` / `d_away`: the distance from the ball to the nearest home and nearest away player.
  `margin = d_away − d_home` is positive when a home player is closer.
- `mean_speed`: the average player speed, clipped at 12 m/s (faster means an identity-switch
  glitch), and smoothed over 1 s.
- `ball_speed`: the ball's movement between frames × 25 fps, smoothed over 0.2 s.
- `ball_out`: the ball is seen more than 1 m beyond a line. The 1 m allows for tracker noise.
- **Why:** these are the raw ingredients of every later decision. They are vectorised over all
  199,319 frames at once (numpy), which is why the whole pipeline runs in about a minute.

## 3. `src/context.py`: who has the ball, from team shape (v3, e4)

```python
def pseudo_labels(f):
    """+1 home, -1 away, 0 no pseudo-label."""
    dmin = np.fmin(f["d_home"], f["d_away"])
    ok = (f["ball_seen"] & ~f["ball_out"]
          & (np.nan_to_num(f["ball_speed"], nan=99) < PSEUDO_MAX_BALL_SPEED)
          & (np.nan_to_num(dmin, nan=99) < PSEUDO_DIST_M)
          & (np.abs(np.nan_to_num(f["margin"])) > PSEUDO_MARGIN_M))
    return np.where(ok, np.sign(np.nan_to_num(f["margin"])), 0).astype(np.int8)
...
    w = fit_logistic(Z[trm], (y[trm] > 0).astype(float))
    logit = np.column_stack([np.ones(len(Z)), Z]) @ w
    logit[~finite] = 0.0
    tau = autocorr_time(logit[graded])
    ...
    return logit / tau
```

- **The problem:** the ball is missing on 41% of graded frames. Without the ball, nearest-player
  evidence is silent.
- **The idea:** the team in possession spreads out and pushes up the pitch, and the other team
  compacts. Five shape features (territory, spread difference, depth difference, advance speed,
  speed difference) are visible whenever the players are tracked.
- **Training without human labels:** `pseudo_labels` keeps only frames where control is obvious
  (ball within 1.5 m of a player, opponent at least 3 m further away, ball not in flight). A
  logistic regression (a model that turns a weighted sum into a probability) learns shape →
  team from those frames. `fit_logistic` uses Newton's method, which is deterministic. Checked
  on the other half, it agrees with the pseudo-labels on 84–85% of frames, against a 62% base rate.
- **`/ tau`:** shape changes slowly. Neighbouring frames repeat the same information, so we
  divide by the signal's autocorrelation time (about 31 s), roughly "how many frames per
  independent look". Without this, one observation would count hundreds of times.
  phase_agree went from 0.708 to 0.720 (e4); ×3 the weight was no better (e4b).

## 4. `model.team_evidence`: ball control → H vs A (v2, e2)

```python
def team_evidence(f, cfg=CONFIG):
    """Unit-weight H-vs-A log-likelihood ratio per frame (> 0 favours H)."""
    seen_in = f["ball_seen"] & ~f["ball_out"]
    dmin = np.nan_to_num(np.fmin(f["d_home"], f["d_away"]), nan=99.0)
    w = np.where(seen_in, np.exp(-dmin / cfg["control_dist_m"]), 0.0)
    if cfg["max_control_ball_speed"] is not None:
        w = np.where(np.nan_to_num(f["ball_speed"]) > cfg["max_control_ball_speed"], 0.0, w)
    margin = np.nan_to_num(f["margin"])
    ...
    return w * np.tanh(margin / cfg["margin_scale_m"])
```

- `w` is how much this frame counts. It is 1 when a player is on the ball and fades with distance
  (`exp(−d/3 m)`: about 0.37 at 3 m). It is 0 when the ball is not seen or is out of bounds.
- `tanh(margin / 2 m)` is the direction and confidence: +1 if home is clearly nearer, −1 if away
  is, near 0 when contested. Football rule: "loose or contested, possession stays with the last
  team", and a near-zero value leaves the HMM where it was.
- **The in-flight gate (e2, the biggest single gain, 0.664 → 0.708):** a ball moving faster than
  8 m/s is a pass, clearance or shot. The nearest player is usually the opponent it is flying
  past, not a controller, so `w = 0`.
- Dropped variants: a 1.5 m control radius (e1) and the same team nearest for 10 frames (e3).

## 5. `model.dead_evidence` + `ball_still` + `throw_hold`: is the ball dead? (v1, v4, v6)

```python
def dead_evidence(f, set_piece_mask):
    """D log-score per frame relative to the live states."""
    seen_in = f["ball_seen"] & ~f["ball_out"]
    z = np.clip((WALK_SPEED - np.nan_to_num(f["mean_speed"], nan=WALK_SPEED)) / SPEED_SCALE, -3, 3)
    out = K_SPEED * z
    out += K_BALL_MISSING * ~f["ball_seen"]
    out += K_BALL_OUT * f["ball_out"]
    out -= K_BALL_FAST * (seen_in & (np.nan_to_num(f["ball_speed"]) > BALL_FAST))
    out += K_SET_PIECE * set_piece_mask
    ...
    if CONFIG["still_nats"]:
        out += CONFIG["still_nats"] * ball_still(f)
    if CONFIG["hold_nats"]:
        out += CONFIG["hold_nats"] * throw_hold(f)
    return out
```

```python
def ball_still(f, cfg=CONFIG):
    h = int(cfg["still_half_s"] * FPS)
    ok = f["ball_seen"] & ~f["ball_out"] & (np.nan_to_num(f["ball_speed"], nan=99) < cfg["still_speed"])
    padded = np.concatenate((np.zeros(h, bool), ok, np.zeros(h, bool)))
    return sliding_window_view(padded, 2 * h + 1).all(axis=1)
```

Each line adds or subtracts a small number of nats for D on each frame.

- **Player speed:** players average about 2 m/s in play and walk (about 1 m/s) in stoppages.
  `z` measures how far below walking pace (1.4 m/s since v8; 1.3 m/s before) they are. At 0.04 nats per unit, z = +3
  (everyone nearly still) gives 0.12 nats/frame, about 3 nats per second.
- **Ball missing (+0.02), ball out (+0.15), ball fast inside the pitch (−0.05)**, plus a tagged
  set-piece window (+0.01).
- **`ball_still` (v4, e12):** true when the ball has been seen, in the pitch, below 1.5 m/s for
  every frame within ±0.5 s. `sliding_window_view(...).all()` is a fast way to say "for the
  whole second". **Why:** at a free kick or goal kick the ball sits still and waits for the kick.
  In open play it almost never does. Before v4 the model called 95% of placed balls live.
- **`throw_hold` (v6, e15):** the ball is seen just beyond a touchline, slow, with a player within
  2.5 m (the thrower). A ball over the line is out of play by definition.

### 5b. v9: the first four clues are learned, plus a restart rule (rounds 20d, 20e)

In v9, CONFIG `learned = "dead"` swaps four of the hand weights above for weights learned from three
other games (`src/learned.py`, stored in `models/learned_nomiss_rl.json`). The four are player
speed, ball fast, ball still and the bias. The rest stays hand-set because Metrica's data cannot
teach it: ball missing (+0.02), ball out, set pieces, throw-in hold and pre-kickoff.

```python
def dead_core(f, m):                       # src/learned.py
    d = m["dead"]
    logit = np.column_stack([np.ones(len(f["bx"])), dead_features(f)]) @ np.array(d["w"])
    return (logit - d["prior"]) * d["scale"]
```

- **What it is:** a logistic regression's log-odds of "dead ball" for each frame, with the
  training base rate taken off (`prior`, so it acts as evidence, not as a prior). It is then
  multiplied by `scale` (0.2). The scale was picked on held-out games: each game was decoded by a
  model fitted on the other two.
- **The learned weights per frame, before scaling:**

  | clue | weight |
  |---|---|
  | speed z | 0.83 |
  | ball fast | −1.71 |
  | ball still | −0.40 |

  After scaling, speed counts about 4× more than the hand weight did.
- **Why ball missing is not learned:** Metrica's tracking blanks the ball during stoppages
  (90–100% missing when dead). On BAR-ATH it is missing in a quarter of live play, so that clue
  does not transfer.

```python
def kick_from_still(f, still_s=1.0, kick_speed=6.0, within_s=1.0, live_s=3.0):   # model.py
    ...  # the 3 s after a still ball (>= 1 s) is struck (> 6 m/s within 1 s)
out -= CONFIG["restart_live_nats"] * kick_from_still(f)                          # 1.0 nat/frame
```

- **The restart rule (20e):** under the Laws, a restart ball is in play once it is kicked and
  clearly moves. Without this rule, the stronger speed clue kept the ball "dead" for a few
  seconds after restarts, because players are still slow just after a kick. 1 nat per frame is
  about twice what every other dead clue can add together, so it acts as a rule. The thresholds
  were fixed from that football logic before the run.

## 6. `model.emissions`: put the evidence in one table

```python
def emissions(team_llr, dead, k_team=None, extra_llr=0.0, still=None):
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
```

- `E` is an `(n, 3)` table: one row per frame, columns H, A, D. This is exactly the "emissions"
  table in the worked example.
- `total` is the H-vs-A evidence: ball control × `K_TEAM` plus team shape. Splitting it ±half
  between H and A keeps the H−A difference equal to `total`.
- `K_TEAM` (1.815 in v6) is **not hand-picked**: `run.calibrate` bisects it until the decoded
  match's own mean possession is 12.5 s (v2).
- **The v4 line (`share * |total| / 2`):** on still-ball frames, D gets the same control score as
  the better live state. "A player is next to the ball" then tells us *which team*, but no longer
  *whether play is live*: someone always stands over a placed ball. Applying this to every frame
  (e11) was too blunt: D carried on after kicks, and possessions fell to 294.

## 7. `model.transition_logs`: how sticky the states are

```python
def transition_logs():
    leave = 1.0 / (POSSESSION_S * FPS)
    to_d = leave * SHARE_ENDING_IN_STOPPAGE
    to_other = leave - to_d
    restart = 1.0 / (DEAD_BALL_S * FPS)
    T = np.array([[1 - leave, to_other, to_d],
                  [to_other, 1 - leave, to_d],
                  [restart / 2, restart / 2, 1 - restart]])
    return np.log(T)
```

- If a possession lasts 12.5 s on average, then on any one frame (1/25 s) the chance it ends is
  `1 / (12.5 × 25) = 0.0032`. One third of those endings are stoppages and two thirds turnovers.
  A dead ball lasts 30 s on average and restarts to either team with equal chance.
- In nats per frame (rounded):

| from \ to | H | A | D |
|---|---|---|---|
| H | −0.0032 | −6.15 | −6.84 |
| A | −6.15 | −0.0032 | −6.84 |
| D | −7.31 | −7.31 | −0.0013 |

- **Reading it:** switching H→A in one frame costs about 6.15 nats more than staying. The H/A
  evidence therefore has to favour the new team by about 6 nats in total before the decoder
  switches. A clean frame of control (ball at 0.5 m, opponent 4 m further) is worth
  1.815 × 0.82 ≈ 1.5 nats, so about 4–5 such frames are enough. A one-frame touch is not.

## 8. `run.restart_penalty`: who restarts after a tagged set piece (v5, e14)

```python
def restart_penalty(n, cfg=model.CONFIG):
    """(n, 2) cost of D -> H / D -> A at each frame from team-tagged SET PIECES (or None)."""
    ...
    for e in tagging.load():
        ...
        sign = TEAM_TAG.get((e["tags"].get("Team") or "").strip().lower(), 0.0)
        if not sign:
            continue
        s = e["start_frame"]
        lo = max(s - int(cfg["sp_restart_before_s"] * fps), 0)
        hi = max(s + int(cfg["sp_restart_after_s"] * fps), 0)
        pen[lo:hi, 1 if sign > 0 else 0] = cfg["sp_restart_pen"]   # forbid the other side
    return pen
```

- For corners, goal kicks and free kicks the automatic tag names the taking team. From 10 s
  before to 20 s after the tag, a restart to the *other* team costs an extra 5 nats (150:1).
- It changes a *transition* (D→team), not the evidence on fixed frames. A turnover a second after
  the restart is still free. The earlier version (e5), which pushed the evidence for 10 s, had no
  effect. This one raised held-out phase agreement after those restarts from 0.525 to 0.580.

## 9. `model.viterbi`: the decoder

```python
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
```

- This is the worked example from §0 in code. `score` holds the best total so far for H, A and D.
  For each next state `j`, `c0/c1/c2` are "arrive from H / A / D". The best one is kept, and
  `back[i, j]` remembers where it came from. Then the frame's emissions `E[i]` are added.
- `c2 -= pen[i][j]` is the v5 restart penalty: it only touches D → H and D → A.
- The last loop walks the pointers backwards to recover the path.
- It is a plain Python loop over about 80,000 frames per half: roughly a second per decode.

## 10. `run.decode` and `run.main`: put it together

```python
def decode(E, n, abstain=False):
    labels = np.full(n, "X", dtype="<U1")      # ungraded frames: never scored
    logT = model.transition_logs()
    pen = restart_penalty(n)
    for _, first, last in half_video_blocks(GAME):
        path = model.viterbi(E[first:last + 1], logT,
                             None if pen is None else pen[first:last + 1])
        seg = model.enforce_min_possession(
            model.LABELS[path], int(model.CONFIG["min_possession_s"] * features.FPS))
        ...
        labels[first:last + 1] = seg
    return labels
```

- Every frame starts as `X` (no answer). Frames outside the two halves (before kickoff, half
  time, after the whistle) are never graded, so `X` there costs nothing.
- Each half is decoded separately, so the 13-minute half-time gap cannot create a fake transition.
- `enforce_min_possession` is off (e8/e9 merged short possessions but produced only 219–276
  possessions in a match). The abstention branch is off too (e13: X was tested and dropped).
- `main()` does: `atd.load` → `features.compute` → `team_evidence`, `dead_evidence`,
  `context_llr` → `calibrate` (12 decodes to find `K_TEAM`) → `decode` → write the CSV.
  `--set key=value` overrides any `CONFIG` entry or `model.py` constant.

---

## Hands-on (PowerShell)

Open a terminal in VS Code (Terminal → New Terminal) and run these once per session:

```powershell
cd <your clone of this repo>          # the folder that holds README.md
$py = ".venv\Scripts\python.exe"     # the virtual environment made in the README
```

**(a) Run the pipeline** (about 1 minute). It writes `submissions\Brasamris.csv` and prints K_TEAM, the
label shares and the possession count:

```powershell
& $py -m src.run
& $py evaluation\validate_submission.py submissions\Brasamris.csv
```

**(b) Change one setting and compare.** Always write experiments to `cache\exp\` so you never
overwrite the real submission:

```powershell
& $py -m src.run --set max_control_ball_speed=6 --out cache\exp\try_gate6.csv
& $py -m src.run --set POSSESSION_S=20 --out cache\exp\try_poss20.csv
& $py -m src.proxy_eval submissions\v5.csv submissions\v6.csv cache\exp\try_gate6.csv cache\exp\try_poss20.csv
& $py -m src.restart_report submissions\v6.csv cache\exp\try_gate6.csv
& $py -m src.restart_team_report submissions\v6.csv cache\exp\try_gate6.csv
```

Read `phase_agree` *together with* `possessions`, `dur_p50` and `share_D`. Higher phase
agreement with far fewer possessions is the trap from e8/e9, not a win.

**(c) Look at a window yourself** (opens a PNG; half 1, minutes 1 to 3):

```powershell
& $py -m src.inspect_window --half 1 --start 1 --end 3
& $py -m src.inspect_window --half 2 --start 30 --end 31.5 --sub cache\exp\try_gate6.csv
```

The image is saved in `outputs\`, which is gitignored. **It shows match data: it is for your
eyes only. Don't commit it, and don't paste or upload it to any AI assistant.**
The public ground-truth band appears only for half 1, minutes 0–5. Use it to understand what the
labels mean, never to choose settings (RULES.md).

**(d) See what changed between versions:**

```powershell
git log --oneline
git show --stat efa76fa                       # which files the v6 commit touched
git diff 4ce1b53 efa76fa -- src/model.py      # v5 -> v6 code change
git diff 87b427d efa76fa -- notes/experiments.md
& $py -c "import pandas as p; a=p.read_csv('submissions/v5.csv'); b=p.read_csv('submissions/v6.csv'); print('frames changed:', (a.label!=b.label).mean())"
```

### Three small experiments to try

1. **How long is a possession?** Run `POSSESSION_S=10`, `15` and `20`. `K_TEAM` re-calibrates
   to each target. Watch `possessions` and `phase_agree` move in opposite directions. This shows
   why the proxy rewards long possessions and why we always check the structure numbers too.
2. **The in-flight gate.** Try `max_control_ball_speed=5`, `12`, and `None` (no gate, as in v1).
   Football question: at what speed is a ball "passing by" rather than "being controlled"? Use
   `inspect_window` on a minute of play to see how often the ball speed line crosses each value.
3. **What counts as a placed ball?** Try `still_speed=1.0` and `still_speed=2.5`, or
   `still_half_s=1.0` (2 s of stillness). Compare with `src.restart_report`: `restart_1s` and
   `restart_off_med` say whether our restarts line up with the kick, and `live_after` must stay
   near 0.99.

If one looks better, write it up as a new round in `notes/experiments.md` (what, why, numbers,
kept or dropped) before changing any default in `src/model.py`.
