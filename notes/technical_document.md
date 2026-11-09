# Ball possession from automatic tracking — technical document (draft)

*More than a Hack 2027, Phase 1. Draft; to be finalised before 17 November 2026.*

## 1. Approach

We label every frame of the tactical video as home (`H`) possession, away (`A`) possession, or
dead ball (`D`). Each frame gets a small amount of evidence for each state. A **three-state
hidden Markov model** turns that evidence into the most likely sequence of states (Viterbi
decoding, run separately for each half).

Why this design and not a per-frame classifier:

- The ranking metric (spellF1) scores **possessions as whole runs**. A per-frame decision that
  flickers between teams invents dozens of short possessions, and each one costs a match. The
  starter baseline shows this: with no smoothing it predicts ~1,950 possessions.
- The ball is missing on 41% of graded frames, usually in long stretches: the median gap is 11 s,
  and 44% of missing frames are in gaps longer than 30 s. A sequence model carries the last
  confident state through a gap instead of guessing frame by frame.
- Transition probabilities are an easy place to encode football knowledge: how long possessions
  and stoppages last.

### 1.1 Evidence

All quantities are in metres (pitch 105.3 × 68.0 m, from `games.py`).

| evidence | for | definition |
|---|---|---|
| ball control | H vs A | `exp(-d_min / 3 m) · tanh(margin / 2 m)`, where `d_min` is the distance from the ball to its nearest player and `margin` is (nearest away − nearest home). Applied only when the ball is seen inside the pitch **and moving slower than 8 m/s** |
| team shape | H vs A | logistic regression on 5 team-shape features (see 1.2), divided by its autocorrelation time |
| player speed | D | mean player speed relative to walking pace: (1.3 m/s − speed) / 0.4 m/s, clipped to ±3 |
| ball missing | D | small constant bonus |
| ball out | D | ball seen more than 1 m beyond a line |
| ball fast in pitch | against D | ball moving faster than 3 m/s inside the pitch means someone kicked it |
| set-piece window | D | inside a Smart Tagging `SET PIECES` interval |
| placed ball | D | ball seen inside the pitch and below 1.5 m/s for a full second (±0.5 s). On these frames ball control is **neutral** between live and D |
| throw-in hold | D | ball seen beyond a touchline, below 3 m/s, a player within 2.5 m (the thrower). Ball control is neutral here too |

**The in-flight gate.** A ball travelling faster than 8 m/s is a pass, clearance or shot. The
player nearest to it is not controlling it; they are usually an opponent it is passing. Removing
control evidence on those frames was the largest single improvement we found (§5).

**The placed ball (v4).** A player standing next to the ball is evidence of *which* team, not of
*whether* the ball is live: at a free kick or goal kick someone stands over the placed ball. In
v3 the ball was seen on only 5% of predicted dead-ball frames, so "dead" in practice meant "ball
missing", and placed balls were called live. Since v4, when the ball is motionless, the control
evidence counts equally for live and dead, and stillness itself is mild dead-ball evidence. The
check (§4): play now restarts a median 0.5 s before the kick that follows a placed ball, where
v3 restarted 6.4 s before it.

### 1.2 Team-shape context, learned without human labels

The team in possession spreads out and pushes up; the team without it compacts and drops back.
This is visible from the 22 players even when the ball is not tracked. The features are:

- how far play is into the home team's attacking half;
- the difference in the two teams' spread;
- the difference in their depth (10th to 90th percentile in x);
- how fast the players' centroid moves towards home's attack;
- the difference in the two teams' mean speed.

Attacking direction is found per half without labels: over a half, each team sits nearer its own
goal on average.

**Training labels are pseudo-labels made by the algorithm itself, never by a person.** These are
frames where ball control is unambiguous: ball seen inside the pitch, slower than 8 m/s, a player
within 1.5 m of it, and the nearest opponent at least 3 m further away. That gives 24,631 frames,
62% of them home. A ridge-penalised logistic regression is fitted by Newton's method, which is
deterministic. Trained on one half and tested on the other, it agrees with the pseudo-labels on
84% and 85% of frames, against a 62% base rate.

Shape changes slowly, so consecutive frames are not independent. The per-frame log-odds are
therefore divided by the signal's integrated autocorrelation time (31 s). Otherwise one observation
would be counted hundreds of times.

### 1.3 Transitions and calibration

- **Possessions:** mean 12.5 s. Elite football has ~10–15 s, i.e. 300–400 possessions in a match.
  One third end in a stoppage and two thirds in a turnover.
- **Stoppages:** mean 30 s (throw-ins ~15–20 s; goal kicks, free kicks and corners 25–45 s). Play
  restarts to either team with equal probability. We checked this. Across stoppage types "the
  other team restarts" is close to a coin flip: corners go back to the attacking team, goal kicks
  to the other one.
- **Restarting team (v5):** for the 22 set pieces whose automatic tag names the team (corners,
  goal kicks, free kicks), restarting to the *other* team costs 5 nats inside a window from 10 s
  before to 20 s after the tag. Only the restart is constrained: a turnover a second later is
  still free. 5 nats is a soft constraint, so clear tracking evidence can still overrule a wrong
  tag.

The overall weight of the H/A evidence (`K_TEAM`) is **calibrated without labels**: bisection sets
it so that the decoded match's own mean possession length equals the 12.5 s prior.

## 2. Assets used and dropped

| asset | use |
|---|---|
| Tactical ATD (players and ball) | everything above |
| Smart Tagging, `SET PIECES` | weak dead-ball evidence |
| Smart Tagging, phase codes (Build Up, Progression, Defending in …) | **held out** — never fed to the model; used only as an independent development check (§4) |
| Smart Tagging, set-piece `Team` tag | v5: constrains which team the restart goes to (§1.3). An earlier per-frame version had no effect |
| Broadcast ATD | aligned to the tactical timeline by code (`src/broadcast.py`: FFT cross-correlation of ball tracks in 10 s chunks; balls agree within 0.6 m after alignment). Borrowing its ball fills only 3.7% of the frames where the tactical ball is missing, and lowered our checks, so it is not used in the submission |
| Homography, broadcast and tactical video | not used: the ATD is already in pitch coordinates |

## 3. Missing and noisy tracking

- **Missing ball (41% of graded frames):** no control evidence is produced. The HMM keeps the
  current state unless the team shape or dead-ball evidence says otherwise.
- **Missing players and identity switches:** all features are team-level aggregates (nearest per
  team, centroids, spreads). The 46 track slots are never treated as people.
- **Speed outliers:** player speeds are clipped at 12 m/s. Ball speed comes from consecutive
  sightings with 0.2 s smoothing.

## 4. Development without ground truth

The public 5 minutes were **not** used to choose anything. Instead, `src/proxy_eval.py` measures:

- **Phase agreement:** the share of live frames inside a team-tagged attacking phase that we give
  to that team, plus the mirror for defending phases. These tags are automatic and noisy, but
  independent of the model.
- **Structure:** possession count, duration percentiles, the share of possessions under 2 s, and
  the share of frames labelled D.
- **Placed-ball kicks** (`src/restart_report.py`): 51 moments where the ball sits still for at
  least 1.5 s and is then struck at more than 6 m/s. We ask whether our dead ball ends at the kick.
  This shares the stillness signal with the v4 evidence, so it is a sanity check, not a score.
  The same script times throw-ins: from a ball held beyond a touchline to the first frame it is
  seen back inside the pitch (geometry only), against our D→live switch.
- **Phase agreement after tagged restarts** (`src/restart_team_report.py`): phase agreement over
  the 20 s after each team-tagged set piece.

A change is kept only if phase agreement improves **and** the structure stays footballing. One
lesson: phase agreement on its own rewards long possessions, because the tags last 15–30 s. A
rule that merged short possessions raised it while pushing the match down to ~220 possessions, so
the structure check is essential.

## 5. Experiments (full log in `notes/experiments.md`)

| change | phase agree | possessions | kept |
|---|---|---|---|
| starter baseline_smooth125 | 0.703 | 398 | — |
| v1: evidence + HMM, K_TEAM 1.0 | 0.667 | 334 | — |
| control radius 1.5 m | 0.672 | 346 | no |
| same team nearest for 10 frames | 0.669 | 349 | no |
| **no control while ball > 8 m/s (v2)** | **0.708** | 350 | yes |
| **+ team-shape context (v3)** | **0.720** | 350 | yes |
| context ×3 | 0.718 | 352 | no |
| set-piece Team tag picks the restarter | 0.721 | 353 | no (noise) |
| D evidence only around set-piece start | 0.718 | 354 | no |
| merge possessions < 2 s / 3 s | 0.726 / 0.733 | 248 / 219 | no (implausible structure) |
| ball control neutral for live vs D everywhere | 0.731 | 294 | no (D runs on after kicks) |
| **placed ball = dead, control neutral on still frames (v4)** | **0.727** | 340 | yes |
| X where the posterior is < 1/3 | — | 340 | no (§6) |
| **tagged team constrains the restart (v5)** | **0.735** | 338 | yes; 0.525 → 0.580 in the 20 s after tagged restarts |
| **throw-in hold = dead (v6)** | 0.734 | 337 | yes; restart within 1 s of the ball re-entering: 13% → 47% (15 throw-ins) |
| ball filled from aligned broadcast tracking | 0.726 | 339 | no |
| kickoff formation (both teams in own halves) = dead | 0.732 | 335 | no (covers part of one stretch, splits it) |
| **graded frames before the half's kickoff = dead (v7)** | 0.733 | 330 | yes; that 114 s stretch was 85% live in v6 |

## 6. Possession rules implemented

- **Control:** the nearest player's team gains evidence, scaled by closeness and by the margin over
  the nearest opponent. A ball in flight gives no evidence, so passes, clearances and deflections
  do not hand over possession by themselves.
- **Loose and contested balls:** small margins give little evidence, so the previous team keeps
  possession, as CHALLENGE.md requires.
- **Dead ball:** players slow to walking pace, the ball is lost or out of the pitch, or a set piece
  is tagged.
- **Restarts:** either team, decided by the evidence that follows.
- **Abstention:** we write `X` only on ungraded frames. We tested a principled rule: X only
  inside one of our own runs, so the scorer's forward fill gives back exactly the same
  possessions and spellF1 cannot change. On macroF1 an X costs one false negative, where a wrong
  label costs a false negative and a false positive, so X pays when P(correct) < F1/2 ≈ 1/3.
  Taken at face value, the HMM's posterior says 5% of frames qualify. But on those frames, the
  held-out phase tags agree with our label 64% of the time, against 73% elsewhere. That is
  worse, but far above the break-even, so the posterior is miscalibrated there and we dropped
  the rule.

## 7. Limitations and next steps

- **Restart timing:** only placed-ball restarts can be checked without labels (§4). Corners and
  throw-ins usually lose the ball from the tracking, so their timing still comes from player
  speed. Smart Tagging windows are fixed 35 s blocks that start near the restart, not at it.
- **Throw-ins (26 of 59 set pieces) carry no team tag**, so the side at a throw-in comes only from
  tracking.
- **Long ball gaps:** the shape context helps but is slow (31 s autocorrelation), so turnovers
  inside long gaps are found late.
- **Next:** a duration-aware model (HSMM); detecting the regroup-then-burst pattern of a restart;
  the broadcast feed is now aligned by code, but its ball adds little where we need it (§2).

### 7.1 Which assumptions matter

We re-ran v6 with each football prior nudged (log round 11). The mean stoppage length (20–40 s)
and the share of possessions ending in a stoppage (0.2–0.5) change fewer than 0.5% of frames. Two
priors carry the result:

- **Mean possession length** (12.5 s): K_TEAM is calibrated to it, so it sets the possession count
  (10 s → 423, 15 s → 277).
- **Walking pace** (1.3 m/s): it sets the dead-ball share (1.1 → 27%, 1.5 → 38%).

Both were fixed from general football knowledge in v1, before any development. They were never
tuned.

## 8. Reproduction

```
pip install -r requirements.txt                    # numpy, pandas
# data: ./data, ../more-than-a-hack-2027/data, or MTAH_DATA_DIR
python -m src.run                                  # -> submissions/v7.csv (deterministic)
python evaluation/validate_submission.py submissions/v7.csv
python -m src.proxy_eval submissions/v7.csv        # development check, no ground truth
python -m src.restart_report submissions/v7.csv    # placed-ball restart check
python -m src.restart_team_report submissions/v7.csv
```

The first run parses the tracking into `cache/` (1–2 minutes). Later runs take about a minute. The
same data always produces a byte-identical file.

## 9. Training data

No human-labelled data of any kind. The only learned component (§1.2) is fitted on pseudo-labels
produced by the algorithm from the provided tracking.
