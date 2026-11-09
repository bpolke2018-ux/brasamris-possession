# Experiments (proxy metric only -- no ground truth)

Columns: see src/proxy_eval.py. phase_agree is the held-out Smart Tagging phase agreement.

## baselines and v1 — 2026-11-02

Reference points: the four starter baselines (smooth25 omitted) and v1 (K_TEAM=1.0).

| run | phase_agree | restart_within_2s | restart_median_offset_s | possessions | dur_p10 | dur_p50 | dur_p90 | short_lt2s | share_D | share_X |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_raw | 0.642 | 0.138 | 4.520 | 1949 | 0.200 | 0.920 | 4.840 | 0.693 | 0.407 | 0.000 |
| baseline_smooth50 | 0.678 | 0.155 | 3.280 | 594 | 0.440 | 4.600 | 18.048 | 0.313 | 0.323 | 0.000 |
| baseline_smooth125 | 0.703 | 0.293 | -0.500 | 398 | 0.520 | 8.140 | 32.100 | 0.276 | 0.223 | 0.000 |
| v1 | 0.667 | 0.138 | 3.860 | 334 | 2.212 | 7.460 | 32.136 | 0.078 | 0.319 | 0.000 |

## round 1: control rules — 2026-11-02

All runs auto-calibrate K_TEAM to a 12.5 s mean possession. e0 = v1 rules. e1 control radius 1.5 m (was 3). e2 no team evidence while the ball moves > 8 m/s (in flight). e3 nearest team must persist 10 frames. KEPT: e2 (phase_agree +0.044; football-correct: the nearest player to a ball in flight is not controlling it). e1/e3: within noise, dropped.

| run | phase_agree | restart_within_2s | restart_median_offset_s | possessions | dur_p10 | dur_p50 | dur_p90 | short_lt2s | share_D | share_X |
|---|---|---|---|---|---|---|---|---|---|---|
| e0_calib | 0.664 | 0.138 | 3.860 | 343 | 2.120 | 7.160 | 31.192 | 0.087 | 0.319 | 0.000 |
| e1_dist15 | 0.672 | 0.121 | 3.200 | 346 | 1.920 | 7.200 | 31.680 | 0.116 | 0.321 | 0.000 |
| e2_flight8 | 0.708 | 0.121 | 3.620 | 350 | 1.316 | 7.220 | 31.520 | 0.174 | 0.311 | 0.000 |
| e3_persist10 | 0.669 | 0.138 | 3.860 | 349 | 2.000 | 6.800 | 30.928 | 0.095 | 0.320 | 0.000 |

## round 2: team-shape context — 2026-11-02

Shape-context log-odds from pseudo-label logistic regression (cross-half pseudo acc 0.84/0.85, base 0.62), divided by its 31 s autocorrelation time. KEPT as v3: e4 (x1). x3 no better.

| run | phase_agree | restart_within_2s | restart_median_offset_s | possessions | dur_p10 | dur_p50 | dur_p90 | short_lt2s | share_D | share_X |
|---|---|---|---|---|---|---|---|---|---|---|
| e2_flight8 | 0.708 | 0.121 | 3.620 | 350 | 1.316 | 7.220 | 31.520 | 0.174 | 0.311 | 0.000 |
| e4_context | 0.720 | 0.121 | 3.880 | 350 | 1.280 | 7.220 | 32.456 | 0.171 | 0.309 | 0.000 |
| e4b_context_x3 | 0.718 | 0.121 | 3.880 | 352 | 1.280 | 7.120 | 32.584 | 0.173 | 0.308 | 0.000 |

## round 3: set-piece Team tag and window timing — 2026-11-04

Diagnostic (src/setpiece_report.py, aggregates): restart sits at the START of the fixed 35 s window (predicted D share 0.53 in 0..5 s, ~0.1 after 5 s); Team tag present on corners, goal kicks, free kicks only (27/59), never throw-ins or kickoffs. e5 tagged team gets 0.02 nats/frame for 10 s; e6 D evidence -10..+3 s instead of the whole window; e7 both. All within noise of v3 -> DROPPED (code kept, off by default).

| run | phase_agree | restart_within_2s | restart_median_offset_s | possessions | dur_p10 | dur_p50 | dur_p90 | short_lt2s | share_D | share_X |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 0.720 | 0.121 | 3.880 | 350 | 1.280 | 7.220 | 32.456 | 0.171 | 0.309 | 0.000 |
| e5_spteam | 0.721 | 0.121 | 4.140 | 353 | 1.280 | 7.080 | 32.304 | 0.173 | 0.307 | 0.000 |
| e6_spstart | 0.718 | 0.121 | 3.620 | 354 | 1.280 | 7.220 | 32.236 | 0.172 | 0.305 | 0.000 |
| e7_both | 0.720 | 0.121 | 4.060 | 355 | 1.280 | 7.080 | 32.168 | 0.172 | 0.303 | 0.000 |

## round 4: minimum controlled possession (instead of X abstention) — 2026-11-04

X abstention reasoning: under spellF1 an X is filled with the previous label, so X == extending the previous label; under macroF1 it is also a miss. A merge rule therefore dominates X wherever X would help, and v3 writes no X on graded frames. e8: merge H/A runs shorter than 1/2/3 s with K_TEAM re-calibrated -- calibration hits its bracket (cannot reach 12.5 s after merging), results mixed. e9: same merge with v3's K_TEAM fixed: phase_agree +0.006/+0.013 BUT possessions drop to 248/219 (17.7/20.0 s), outside the 300-400 a match normally has. The phase tags last 15-30 s, so phase_agree is biased towards fewer, longer possessions (the same reason baseline_smooth125 scores well on it). DROPPED; structure has to guard the proxy.

| run | phase_agree | restart_within_2s | restart_median_offset_s | possessions | dur_p10 | dur_p50 | dur_p90 | short_lt2s | share_D | share_X |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 0.720 | 0.121 | 3.880 | 350 | 1.280 | 7.220 | 32.456 | 0.171 | 0.309 | 0.000 |
| e8_min1 | 0.708 | 0.121 | 3.620 | 351 | 1.600 | 7.080 | 32.680 | 0.145 | 0.307 | 0.000 |
| e8_min2 | 0.712 | 0.121 | 3.880 | 276 | 2.860 | 9.800 | 36.880 | 0.011 | 0.302 | 0.000 |
| e8_min3 | 0.723 | 0.121 | 3.880 | 231 | 3.920 | 12.040 | 42.520 | 0.030 | 0.302 | 0.000 |
| e9_fixk_min2 | 0.726 | 0.121 | 3.880 | 248 | 3.268 | 11.260 | 41.000 | 0.008 | 0.309 | 0.000 |
| e9_fixk_min3 | 0.733 | 0.121 | 3.880 | 219 | 3.952 | 14.400 | 42.904 | 0.014 | 0.309 | 0.000 |

## round 5: ball lost next to a line -> dead ball — 2026-11-04

While the ball is missing and was last seen < 2 m from a line within 20 s: +0.03 / +0.10 nats/frame for D. phase_agree 0.717/0.718 vs 0.720 -> DROPPED (off by default). Caveat: phase_agree only scores H/A frames, so the proxy is nearly blind to dead-ball changes; a label-free D check is needed before more D work.

| run | phase_agree | restart_within_2s | restart_median_offset_s | possessions | dur_p10 | dur_p50 | dur_p90 | short_lt2s | share_D | share_X |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 0.720 | 0.121 | 3.880 | 350 | 1.280 | 7.220 | 32.456 | 0.171 | 0.309 | 0.000 |
| e10_line3 | 0.717 | 0.121 | 3.880 | 345 | 1.320 | 7.280 | 32.696 | 0.165 | 0.319 | 0.000 |
| e10_line10 | 0.718 | 0.121 | 3.620 | 339 | 1.400 | 7.280 | 32.472 | 0.162 | 0.331 | 0.000 |

## round 6: placed ball (still ball) as dead-ball evidence — 2026-11-05

New label-free check, src/restart_report.py: a *placed-ball kick* (PBK) is the ball seen and nearly still (< 1.5 m/s) for >= 1.5 s, then struck (> 6 m/s within 1 s). There are 51 on graded frames; 11 are within 15 s of a SET PIECES tag start (the others are probably untagged free kicks plus some open-play pauses). The model never used ball stillness before. Finding: in v3 the ball is seen on only 5% of predicted-D frames, so v3's D is in effect "ball missing", and v3 calls placed balls live (D share 0.06 in the 2 s before a PBK; nearest D->live switch a median 6.4 s *before* the kick). Cause: ball-control evidence (a player next to the ball) also votes for live over D, and a placed ball always has a player next to it.
e11: d_team_share=1 on all frames (control evidence neutral between live and D) -> D 42%, 294 possessions, and D lingers after kicks (live_after 0.80). Too blunt. e12: neutral ONLY on still-ball frames (3.9% of graded) plus still_nats D evidence. KEPT e12a as v4: PBK restart within 1 s 0.06 -> 0.28, median offset -6.4 -> -0.5 s, live 1-3 s after the kick unchanged (0.99), phase_agree 0.727, structure unchanged (340 possessions). Caveats: the PBK check shares the stillness signal with the change, so it is partly circular; the set-piece tag alignment fell from 7/58 to 4/58 (those tags are loose 35 s windows).

| run | phase_agree | restart_within_2s | restart_median_offset_s | possessions | dur_p10 | dur_p50 | dur_p90 | short_lt2s | share_D | share_X | pbk_restart_1s | pbk_d_before | pbk_live_after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 0.720 | 0.121 | 3.880 | 350 | 1.280 | 7.220 | 32.456 | 0.171 | 0.309 | 0.000 | 0.059 | 0.059 | 0.993 |
| e11a_dneutral (all) | 0.731 | 0.052 | 6.100 | 294 | 1.532 | 7.600 | 31.652 | 0.133 | 0.420 | 0.000 | 0.235 | 0.441 | 0.805 |
| e11b_still only | 0.722 | 0.121 | 3.880 | 349 | 1.280 | 7.200 | 32.472 | 0.175 | 0.311 | 0.000 | 0.059 | 0.059 | 0.993 |
| e11c_both (all) | 0.739 | 0.052 | 6.100 | 279 | 1.696 | 7.520 | 31.400 | 0.122 | 0.446 | 0.000 | 0.255 | 0.564 | 0.696 |
| **e12a = v4** (neutral on still + still 0.1) | 0.727 | 0.069 | 4.300 | 340 | 1.320 | 6.820 | 32.472 | 0.168 | 0.332 | 0.000 | 0.275 | 0.227 | 0.993 |
| e12b (still 0.05) | 0.725 | 0.069 | 4.300 | 345 | 1.296 | 6.680 | 32.632 | 0.174 | 0.325 | 0.000 | 0.196 | 0.168 | 0.993 |
| e12c (share 0.5 all + still) | 0.721 | 0.103 | 4.300 | 341 | 1.320 | 7.320 | 32.440 | 0.161 | 0.330 | 0.000 | 0.137 | 0.133 | 0.993 |
| e12d (neutral on still, no still nats) | 0.721 | 0.121 | 4.140 | 348 | 1.280 | 7.140 | 32.488 | 0.172 | 0.313 | 0.000 | 0.078 | 0.074 | 0.993 |

## round 7: X abstention from the HMM posterior — 2026-11-05

Rule (src/model.py abstain, src/run.py decode): write X on a frame only if its label equals the previous frame's (so the scorer's forward fill restores the exact v4 labels -> spellF1 unchanged, verified: filled == v4 on every graded frame) and the forward-backward posterior of that label is below a threshold. macroF1 reasoning: a wrong label costs FN(true) + FP(pred), an X costs FN only, so X pays when P(correct) < F1_k / 2 (about 1/3).
Model-expected macroF1 (src/abstain_report.py, posteriors as soft truth): v4 0.890, X<0.33 0.905, X<0.5 0.906. But 0.89 is far above what any real system scores, so the posteriors are miscalibrated, and the model cannot vouch for itself. Independent check: v4's phase agreement (held-out tags) on the frames that would be abstained:

| threshold | X share of graded | abstained frames' phase agreement | kept frames' agreement |
|---|---|---|---|
| 0.05 | 0.005 | 0.471 | — |
| 0.10 | 0.009 | 0.556 | — |
| 0.20 | 0.016 | 0.608 | — |
| 0.33 | 0.051 | 0.640 | 0.732 |
| 0.50 | 0.085 | 0.624 | 0.738 |

With tag noise e implied by the overall agreement (q(1-e)+(1-q)e = 0.73 at a plausible q ~0.85 -> e ~0.17), even the 0.05 set implies P(correct) ~0.45, above the ~1/3 break-even. Low-posterior frames are near our own boundaries, where the Viterbi label is still right more often than the marginals say. DROPPED: abstain_below stays 0 (code kept). Also fixed src/proxy_eval.py to forward-fill X before the structure metrics, exactly as the scorer does (no effect on X-free files).

## round 8: restarting team as a restart constraint — 2026-11-05

Aggregate prior check first (v4 team before the stoppage vs tagged restarter): corners same team 12/0, goal kicks other team 8/2, free kicks 3/2; v4's own D stretches 41 same / 41 other. Over all stoppage types "the other team restarts" is close to a coin flip, so a generic switch prior was NOT built. Instead e14: inside [tag start - 10 s, tag start + 20 s] of the 22 team-tagged SET PIECES (corners, goal kicks, free kicks), the D -> live transition to the non-tagged team costs sp_restart_pen nats (time-varying Viterbi transition); later turnovers untouched. Held-out check src/restart_team_report.py: phase agreement in the 20 s after those tag starts (phase tags never fed to the model). KEPT e14a (5 nats, soft: strong evidence can still override a tagging error) as v5. 5 vs 20 nats are equal on the proxy; 5 chosen by the softness argument, not by score. Caveat: 22 events, so the 20 s agreement is noisy.

| run | phase_agree | phase_agree_20s_after_tagged_restart | tagged_team_share (not independent) | possessions | short_lt2s | share_D |
|---|---|---|---|---|---|---|
| v4 | 0.727 | 0.525 | 0.674 | 340 | 0.168 | 0.332 |
| **e14a = v5** (pen 5, -10..+20 s) | 0.735 | 0.580 | 0.725 | 338 | 0.163 | 0.332 |
| e14b (pen 20) | 0.734 | 0.579 | 0.745 | 340 | 0.162 | 0.332 |
| e14c (pen 20, -10..+10 s) | 0.732 | 0.565 | 0.720 | 340 | 0.165 | 0.333 |

## round 9: throw-ins — 2026-11-05

Throw-in team (not built): the tracker sees the ball held just beyond a touchline with a player within 2.5 m in 17 episodes (~9 of the 26 tagged throw-ins); v5 already gives the first 2 s of live play after them to the holder's team 93% of the time (1/17 wrong). Nothing to gain -> not built. But v5 called only ~30-35% of the hold frames D, although a ball beyond the line is out of play.
e15: throw_hold (ball seen beyond a touchline, < 3 m/s, player within 2.5 m) adds hold_nats D evidence and makes ball control neutral there (like a placed ball). New label-free check in src/restart_report.py: the D->live switch vs the first frame the ball is seen back inside the pitch (re-entry: pure geometry) after each hold, 15 episodes. KEPT e15a (0.15 nats, the same weight as a ball > 1 m out; 0.4 was a little better on 15 events, 9 vs 7 hits, but stronger than the existing ball-out weight and riskier for balls dribbled along the line) as v6.

| run | phase_agree | phase_agree_20s_after_tagged_restart | possessions | short_lt2s | share_D | throw_d_during_hold | throw_restart_1s | throw_off_med_s | pbk_restart_1s |
|---|---|---|---|---|---|---|---|---|---|
| v5 | 0.735 | 0.580 | 338 | 0.163 | 0.332 | 0.301 | 0.133 | -1.24 | 0.275 |
| **e15a = v6** (hold 0.15) | 0.734 | 0.580 | 337 | 0.157 | 0.335 | 0.513 | 0.467 | -0.20 | 0.294 |
| e15b (hold 0.4) | 0.734 | 0.577 | 338 | 0.154 | 0.336 | 0.685 | 0.600 | -0.16 | 0.294 |

## round 10: ball from the broadcast tracking, aligned by code — 2026-11-05

src/broadcast.py aligns the unsynchronised broadcast ATD to the tactical timeline with no labels: 10 s broadcast chunks, FFT cross-correlation of the masked ball tracks over all offsets; accept if the median ball distance at the best offset is < 1.5 m and the runner-up (> 2 s away) is 4x worse. 186 of 414 chunks with enough ball are accepted; offsets are 4013-4018 frames (first half) and ~26921 (second half); 69% of consecutive accepted chunks share the offset (cuts and replays in between). Where both feeds see the ball after mapping, they agree within 0.58 m (median; p90 1.9 m; 3.7% > 3 m). But only 3.7% of the tactical missing-ball graded frames get filled: the broadcast mostly sees the ball when the tactical camera does too. A second, player-based pass (2 s pieces, nearest-neighbour fit of broadcast players to tactical players at the neighbouring chunks' offsets, sharp-minimum test) maps more frames (fills 9.6%), but its balls disagree with the tactical ball by > 3 m on 31% of shared frames, so it is unreliable.
e16a (fill from ball-aligned chunks) and e16b (+ player pieces): both lower phase agreement and post-restart agreement. DROPPED (broadcast_ball = "off"; the alignment code is kept). Restart checks unchanged (they read the tactical ball).

| run | phase_agree | phase_agree_20s_after_tagged_restart | possessions | short_lt2s | share_D |
|---|---|---|---|---|---|
| v6 | 0.734 | 0.580 | 337 | 0.157 | 0.335 |
| e16a_bc_balls | 0.726 | 0.569 | 339 | 0.153 | 0.332 |
| e16b_bc_players | 0.713 | 0.566 | 343 | 0.143 | 0.325 |

## round 11: sensitivity of v6 to its football priors (no change) — 2026-11-05

Each prior nudged on its own, K_TEAM re-calibrated each time (to POSSESSION_S, as always). "boundaries" = share of the variant's segment starts within 1 s of one of v6's.

| prior (v6 value) | variant | K_TEAM | frames changed | boundaries within 1 s of v6 | possessions | share_D | phase_agree |
|---|---|---|---|---|---|---|---|
| DEAD_BALL_S (30) | 20 / 40 | 1.815 / 1.815 | 0.001 / 0.001 | 0.99 / 1.00 | 339 / 337 | 0.334 / 0.336 | 0.734 / 0.734 |
| SHARE_ENDING_IN_STOPPAGE (1/3) | 0.2 / 0.5 | 1.763 / 1.924 | 0.001 / 0.004 | 0.99 / 0.98 | 340 / 340 | 0.334 / 0.333 | 0.734 / 0.731 |
| POSSESSION_S (12.5) | 10 / 15 | 3.399 / 1.230 | 0.022 / 0.022 | 0.82 / 0.98 | 423 / 277 | 0.333 / 0.344 | 0.719 / 0.734 |
| WALK_SPEED (1.3) | 1.1 / 1.5 | 2.468 / 1.479 | 0.073 / 0.052 | 0.85 / 0.90 | 368 / 314 | 0.272 / 0.379 | 0.724 / 0.743 |

Reading: the dead-ball duration and stoppage share hardly matter. Two priors carry the result: POSSESSION_S sets the possession count through calibration (277-423), and WALK_SPEED sets the dead-ball share (27-38%). Neither can be validated without labels, and phase_agree is biased (it rises with fewer, longer possessions and with more D, since it only scores live frames). Both stay at the values fixed a priori in v1 (2026-11-02), before any of this. They are NOT to be tuned on the public window. Disclosure: the public GT meta JSON, which states its window's label counts and mean possession, was printed once on 2026-11-05 while building inspect_window. That is one more reason to leave these two priors exactly where they were set beforehand.

## round 12 (parked): kickoff detection — 2026-11-05

Feasibility only: frames with >= 95% of each team's tracked players (>= 8 per team) in their own half, attack direction from src/context.py, merged into episodes >= 1 s: zero episodes, with or without the ball near the centre spot, while there are 5 graded kickoff tags. The shape test is probably too strict for automatic tracking (a stray track or an identity switch on the wrong side), or the half convention needs checking. Parked: other-game validation has priority (see notes/other_games.md).

## round 13 (stopped mid-way, not decided): kickoff formation as dead-ball evidence — 2026-11-06

Second attempt (round 12's test had a bug: NaN slots counted as players on the wrong side). model.kickoff_formation: <= 1 tracked player outside his own half, >= 16 tracked, episodes >= 1 s -> 7 episodes covering all 5 graded kickoff tags; v6 calls only 50% of their frames D. e17 adds kickoff_nats D evidence (control neutral there). restart_report now also times the end of each formation (the kick) against our D->live switch. STOPPED on Brandon's request before deciding; kickoff_nats stays 0 (v6 output verified unchanged).

| run | phase_agree | phase_agree_20s_after_tagged_restart | possessions | share_D | ko_d_during | ko_restart_2s | ko_off_med_s |
|---|---|---|---|---|---|---|---|
| v6 | 0.734 | 0.580 | 337 | 0.335 | 0.499 | 0.429 | -14.76 |
| e17a (0.15) | 0.732 | 0.580 | 335 | 0.341 | 1.000 | 0.571 | 0.00 |
| e17b (0.4) | 0.732 | 0.580 | 335 | 0.341 | 1.000 | 0.571 | 0.00 |


**Round 13 decision (2026-11-09): e17 DROPPED** (kickoff_nats stays 0, code kept). Per-episode aggregates: of the 7 formation episodes, 3 sit in the 114 s of graded play before the 2nd-half kickoff (players warming up in their own halves, v6 D 0.00 there), and the other 4 at kickoffs after goals, where v6 is already 0.68-0.99 D. So e17's gain is mostly that one pre-kickoff stretch, and it only covers 19 s of it: it splits the stretch into more pieces (label changes there 3 -> 6) and lowers live_after at placed-ball kicks (0.993 -> 0.973). Round 14 fixes the real problem directly.

## round 14: before the half's kickoff the ball is dead — 2026-11-09

What: graded frames from the start of a half's grading up to that half's Kick Off (Start) Smart Tagging tag get 50 nats/frame of D evidence (a hard rule; prekick_nats in src/model.py, prekick_mask in src/run.py). Football: by the Laws the ball is not in play before the kickoff. The 2nd half is graded from 114 s before its kickoff tag (games.half_video_blocks; FAQ.md says these frames are D, public documentation, not ground truth); the 1st half's grading starts at its kickoff, so nothing changes there. Found because v6 labelled 85% of that stretch live (H 0.76, A 0.09) with 3 label changes. New check: pre-kickoff share (scratch, aggregates; by construction 1.0 for e18, so not evidence on its own).
KEPT e18a as **v7**. Everything else unchanged; phase_agree within noise (it only scores H/A frames); 7 fewer possessions (2 spurious ones removed, the rest from K_TEAM re-calibration, 1.70). The kick lands where the formation breaks (ko_off_med -14.8 -> 0.04 s). e18b (with e17's 0.15 on top) moves the kickoff switch 6 s late and lowers live_after: dropped.

| run | phase_agree | phase_agree_20s_after_tagged_restart | possessions | short_lt2s | share_D | prekick_D | ko_d_during | ko_off_med_s | pbk_live_after |
|---|---|---|---|---|---|---|---|---|---|
| v6 | 0.734 | 0.580 | 337 | 0.157 | 0.335 | 0.15 | 0.499 | -14.76 | 0.993 |
| e17a (formation 0.15) | 0.732 | 0.580 | 335 | 0.149 | 0.341 | 0.37 | 1.000 | 0.00 | 0.973 |
| **e18a = v7** (prekick 50) | 0.733 | 0.580 | 330 | 0.148 | 0.351 | 1.00 | 0.928 | 0.04 | 0.993 |
| e18b (prekick + formation) | 0.732 | 0.580 | 331 | 0.151 | 0.352 | 1.00 | 1.000 | 6.32 | 0.973 |

## round 15: duration-aware states (HSMM by state chaining) — 2026-11-09

What: each state (H, A, D) can be replaced by a chain of r identical sub-states (model.chain_transition_logs, model.viterbi_chain; CONFIG live_phases / dead_phases, 1 = v7). Same mean lengths (12.5 s, 30 s) and exit splits, but the length of a stretch becomes negative-binomial with a mode above zero instead of geometric (where the single most likely length is one frame). Football: a team that wins the ball usually keeps it for at least a touch or two, and a restart takes some seconds. a-priori shapes: 2 for live (the smallest non-geometric choice), 3 for dead balls (their lengths vary less). Verified: r = 1 reproduces viterbi() exactly (synthetic input) and v7 is byte-identical.
DROPPED: every variant lowers phase_agree (0.733 -> 0.721-0.724). Why: K_TEAM is re-calibrated to keep the 12.5 s mean, and since the chain prior makes stretches longer the calibration doubles the evidence weight (1.70 -> 3.3-6.0), which makes the decoder twitchier wherever there is evidence; even the share of possessions under 2 s goes up (0.148 -> 0.163 for live 2). A 3-phase D removes short dead stretches: D share 0.351 -> 0.29, placed-ball restarts within 1 s 0.29 -> 0.14-0.18. The 20 s after tagged restarts improves (0.580 -> 0.59-0.60) only because more of it is called live. Under a fixed-mean calibration a duration prior and the evidence weight trade off against each other, so the prior bought nothing. Code kept, off by default.

| run | K_TEAM | phase_agree | phase_agree_20s_after_tagged_restart | possessions | short_lt2s | share_D | pbk_restart_1s | throw_restart_1s |
|---|---|---|---|---|---|---|---|---|
| v7 | 1.70 | 0.733 | 0.580 | 330 | 0.148 | 0.351 | 0.294 | 0.467 |
| e19a (live 2) | 3.34 | 0.721 | 0.548 | 337 | 0.163 | 0.337 | 0.275 | 0.467 |
| e19b (dead 3) | 2.47 | 0.724 | 0.598 | 359 | 0.195 | 0.290 | 0.176 | 0.333 |
| e19c (live 2, dead 3) | 4.36 | 0.723 | 0.591 | 365 | 0.200 | 0.285 | 0.137 | 0.333 |
| e19d (live 3, dead 3) | 5.99 | 0.724 | 0.598 | 366 | 0.199 | 0.284 | 0.137 | 0.333 |

## round 16: held-out robustness on two other games (Metrica sample games 1, 2) — 2026-11-09

What: src/other_games.py runs v7's tag-free core (no Smart Tagging there: set-piece windows, tagged restart penalty and pre-kickoff rule off) on Metrica's public sample games and scores it with the official possession_metrics against labels derived from Metrica's hand-coded events (rules in notes/other_games.md, fixed before any result; labels look football-plausible: 352 / 311 possessions, 11.4 / 13.3 s, D 0.30 / 0.26). Then each kept change is undone one at a time, and the dropped tag-free ideas are switched on. Different labelling convention from the hackathon, so compare rows only with each other; with ~300 possessions per game, spellF1 gaps below ~0.01-0.02 are noise. Used to measure robustness only; no BAR-ATH setting changed.
Findings: the flight gate (v2) generalises clearly (spellF1 -0.040 / -0.009 and macroF1 -0.031 / -0.045 without it). Team-shape context (v3) generalises (spellF1 -0.012 / -0.013 without it). The throw-in hold (v6) never fires there (identical output: Metrica's ball is not tracked beyond the touchline). The still-ball change (v4) is a wash: spellF1 +0.010 / +0.003 without it, macroF1 -0.017 / +0.010; v7 calls ~4 points more D than these labels (part of that may be the labels, whose dead stretches only start at a coded stoppage). Dropped ideas stay dropped: 3-phase D chain clearly worse (-0.033 / -0.026), live 2-phase chain and kickoff formation neutral. K_TEAM calibrates to 4.4 / 7.7 there (1.70 on BAR-ATH; the ball is seen on only ~60% of frames).

| run | game | spell_f1_mean | macro_f1 | alive_dead_f1 | team_acc_both_alive | gt_poss | pred_poss | pred_share_D (gt) |
|---|---|---|---|---|---|---|---|---|
| v7 core | 1 | 0.544 | 0.789 | 0.903 | 0.872 | 352 | 303 | 0.336 (0.295) |
| v7 core | 2 | 0.485 | 0.776 | 0.892 | 0.896 | 311 | 313 | 0.306 (0.264) |
| no throw-in hold (v5) | 1 / 2 | 0.544 / 0.485 | 0.789 / 0.776 | same | same | | 303 / 313 | same |
| no still ball (v3-like) | 1 / 2 | 0.554 / 0.488 | 0.772 / 0.786 | 0.895 / 0.900 | 0.875 / 0.896 | | 317 / 318 | 0.302 / 0.292 |
| no context (v2-like) | 1 / 2 | 0.532 / 0.472 | 0.784 / 0.772 | 0.903 / 0.892 | 0.866 / 0.890 | | 303 / 313 | 0.336 / 0.306 |
| no flight gate | 1 / 2 | 0.504 / 0.476 | 0.758 / 0.731 | 0.902 / 0.891 | 0.825 / 0.826 | | 302 / 313 | 0.334 / 0.301 |
| live 2-phase (e19a) | 1 / 2 | 0.547 / 0.481 | 0.786 / 0.772 | 0.904 / 0.889 | 0.869 / 0.897 | | 307 / 314 | 0.325 / 0.304 |
| dead 3-phase (e19b) | 1 / 2 | 0.511 / 0.459 | 0.779 / 0.754 | 0.906 / 0.894 | 0.863 / 0.879 | | 321 / 337 | 0.294 / 0.248 |
| kickoff formation (e17) | 1 / 2 | 0.544 / 0.482 | 0.788 / 0.775 | 0.903 / 0.891 | 0.872 / 0.896 | | 303 / 311 | 0.336 / 0.307 |

## round 17: Metrica game 3 added, paired bootstrap on the gaps — 2026-11-09

What: src/other_games.py now also reads Metrica game 3 (EPTS text tracking with player slots remapped at each substitution; JSON events converted to the same table; CARRY counts as an on-ball possession event, the same rule as a pass). Event-derived labels for game 3: 376 possessions, 12.0 s, D 0.21; v7's team accuracy there is 0.90, which confirms the home/away mapping. New src/other_games_boot.py: paired bootstrap on the gap between two versions. Each draw resamples the ground-truth possessions (spellF1) or ground-truth segments (macroF1) of each game, the same picks for every version, as the official scorer's own bootstrap does (predicted-possession count fixed), and averages the three games; 2000 draws, 5-95% interval. Caveat: with 3 games it does not capture game-to-game variation, and holding the predicted count fixed understates the spread a little.
Result: v7 core averages spellF1 0.502 / macroF1 0.773 over the three games. Gaps when a change is undone (negative = v7 is better):
- team-shape context: spellF1 -0.014 [-0.024, -0.005] -> it generalises.
- flight gate: macroF1 -0.033 [-0.042, -0.025] -> it generalises; its spellF1 gap is uncertain (game 3 goes the other way).
- still ball (v4): spellF1 +0.006 [+0.000, +0.012], macroF1 0.000 [-0.016, +0.013] -> on these games it costs a little spellF1 (borderline). Kept on the coordinator's instruction (it helps BAR-ATH's restart timing, and these labels start a dead ball only at a coded stoppage, so they may under-count dead time); flagged for Brandon.
- throw-in hold: exactly 0 (never fires; the ball is not tracked beyond the line here). Kickoff formation: ~0.
- duration chains (round 15): dead 3-phase spellF1 -0.031 [-0.042, -0.020]; live 2-phase macroF1 -0.008 [-0.014, -0.002]. Confirms they stay dropped.
No BAR-ATH setting changed.

| version (undone / added) | game 3 spellF1 | game 3 macroF1 | 3-game spellF1 gap [5-95%] | 3-game macroF1 gap [5-95%] |
|---|---|---|---|---|
| v7 core (reference) | 0.478 | 0.754 | (0.502) | (0.773) |
| no throw-in hold | 0.478 | 0.754 | +0.000 | +0.000 |
| no still ball | 0.482 | 0.761 | +0.006 [+0.000, +0.012] | +0.000 [-0.016, +0.013] |
| no context | 0.459 | 0.747 | -0.014 [-0.024, -0.005] | -0.005 [-0.010, +0.000] |
| no flight gate | 0.490 | 0.730 | -0.012 [-0.032, +0.007] | -0.033 [-0.042, -0.025] |
| live 2-phase | 0.465 | 0.737 | -0.004 [-0.011, +0.002] | -0.008 [-0.014, -0.002] |
| dead 3-phase | 0.444 | 0.714 | -0.031 [-0.042, -0.020] | -0.024 [-0.037, -0.013] |
| kickoff formation | 0.477 | 0.753 | -0.001 [-0.003, +0.000] | -0.001 [-0.001, -0.000] |

## round 18: the two sensitive settings on both benches; v8 — 2026-11-09

Rule from the coordinator for this and later rounds: a change is kept only if it does not hurt the Metrica paired-bootstrap gap AND does not hurt the BAR-ATH proxy / structure / restart checks. RULES.md allows hand-labelled (here: event-derived) data from other games for training and validation, if the technical document explains it.
What: POSSESSION_S (mean possession, sets the possession count) 10 / 15 s around 12.5, and WALK_SPEED (the player speed at which dead and live are equally likely, sets the D share) 1.1 / 1.5 m/s around 1.3. Each on BAR-ATH (proxy) and the three Metrica games (official scorer, paired bootstrap vs v7, 5-95%).
Results: POSSESSION_S 12.5 s holds up: 10 s hurts both benches (Metrica spellF1 -0.026 [-0.035, -0.017]; BAR-ATH phase_agree 0.719, 415 possessions, 23% under 2 s); 15 s is a wash on Metrica (+0.007 [-0.004, +0.016], macroF1 -0.004) and lowers the BAR-ATH 20 s-after-restart check (0.580 -> 0.551) with 271 possessions. WALK_SPEED 1.1 hurts both. 1.5 helps Metrica clearly (+0.021 [+0.010, +0.031]) and BAR-ATH phase_agree (0.743), but lowers the 20 s-after-tagged-restart check (0.580 -> 0.541) and live play after placed-ball kicks (0.993 -> 0.978): fails the rule.
Artefact check before choosing: Metrica has no speed field, so its speeds come from positions; on BAR-ATH the displacement speed and the tracker's speed agree (median 1.49 vs 1.52 m/s), so the benches are on the same scale. The Metrica labels give an independent measurement: median team speed 1.87-2.00 m/s in live play and 0.92-1.01 m/s in dead ball, midpoint ~1.4. Tested that derived value (not a grid pick): WALK_SPEED 1.4 -> Metrica spellF1 +0.009 [+0.003, +0.016], macroF1 +0.004 [-0.002, +0.010]; BAR-ATH phase_agree 0.733 -> 0.743, 20 s-after-tagged-restart 0.580 -> 0.579, 314 possessions, under-2 s 0.148 -> 0.134, D 0.351 -> 0.380, placed-ball checks equal or better; throw-in restart within 1 s 7/15 -> 6/15 (one event). KEPT as **v8**. (Round 11 had said WALK_SPEED must not be tuned on this match; it still isn't: the value comes from other games' measured speeds.)

| run | phase_agree | after_tagged_restart_20s | possessions | short_lt2s | share_D | pbk_restart_1s | pbk_live_after | throw_restart_1s | Metrica spellF1 gap [5-95%] | Metrica macroF1 gap [5-95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| v7 (walk 1.3, poss 12.5) | 0.733 | 0.580 | 330 | 0.148 | 0.351 | 0.294 | 0.993 | 0.467 | (0.502) | (0.773) |
| poss 10 | 0.719 | 0.548 | 415 | 0.227 | 0.348 | 0.294 | 0.993 | 0.600 | -0.026 [-0.035, -0.017] | -0.003 [-0.006, -0.002] |
| poss 15 | 0.735 | 0.551 | 271 | 0.085 | 0.361 | 0.294 | 0.993 | 0.400 | +0.007 [-0.004, +0.016] | -0.004 [-0.009, -0.000] |
| walk 1.1 | 0.725 | 0.581 | 362 | 0.185 | 0.288 | 0.235 | 0.993 | 0.467 | -0.034 [-0.045, -0.023] | -0.032 [-0.045, -0.021] |
| **walk 1.4 = v8** | 0.743 | 0.579 | 314 | 0.134 | 0.380 | 0.294 | 0.992 | 0.400 | +0.009 [+0.003, +0.016] | +0.004 [-0.002, +0.010] |
| walk 1.5 | 0.743 | 0.541 | 309 | 0.126 | 0.391 | 0.314 | 0.978 | 0.467 | +0.021 [+0.010, +0.031] | +0.009 [+0.001, +0.017] |

Candidate (2), short ball-gap interpolation: NOT RUN, ruled out by an aggregate count. The ball is missing on 41% of BAR-ATH graded frames, but in 150 long gaps (~17 s on average); gaps of 2 s or less hold 0.1% of the missing frames (0.3-1.1% on the Metrica games). Filling short gaps cannot move the result; filling long ones would invent the ball.

## round 19: controlled possession by one player — 2026-11-09

What: a player's control evidence counts in full only after the SAME player has been nearest to the ball for persist_player_frames consecutive frames (linear ramp; features.near_id, model.team_evidence; off by default, v8 verified byte-identical). Football: a touch, deflection or a ball rolling past a defender is not control (CHALLENGE.md); control means one player keeps the ball. A-priori lengths 5 frames (0.2 s, a controlled first touch) and 10 frames (0.4 s). Measured on both benches with the keep rule.
DROPPED: it hurts the Metrica games (spellF1 -0.009 [-0.015, -0.004] for 5 frames, -0.007 [-0.014, +0.000] for 10) and slightly lowers BAR-ATH phase_agree (0.743 -> 0.739 / 0.740); restart checks unchanged, structure unchanged. Likely reason: the ramp delays the new team's evidence exactly at a turnover (the winner has to "hold" before it counts), so possession boundaries arrive late, and the nearest-player identity flips between team-mates in a crowd. Round 1's team-level version was noise for the same reason.

| run | phase_agree | after_tagged_restart_20s | possessions | short_lt2s | share_D | pbk_restart_1s | Metrica spellF1 gap [5-95%] | Metrica macroF1 gap [5-95%] |
|---|---|---|---|---|---|---|---|---|
| v8 | 0.743 | 0.579 | 314 | 0.134 | 0.380 | 0.294 | (0.512) | (0.776) |
| same player 5 frames | 0.739 | 0.579 | 317 | 0.129 | 0.379 | 0.294 | -0.009 [-0.015, -0.004] | -0.000 [-0.002, +0.002] |
| same player 10 frames | 0.740 | 0.579 | 316 | 0.123 | 0.380 | 0.314 | -0.007 [-0.014, +0.000] | -0.001 [-0.003, +0.001] |

## round 20: per-frame evidence learned from the Metrica games — 2026-11-09

What: src/learned.py fits two deterministic ridge logistic regressions on the three Metrica games' event-derived labels. Dead model (D vs live, all graded frames): player-speed z, ball missing, ball fast inside the pitch, ball still. Team model (H vs A, live frames): unit ball-control evidence and the team-shape context logit. Only clues both benches have; off-pitch ball, throw-in hold, Smart Tagging, pre-kickoff stay hand-set (model.CONFIG["learned"] = "dead" / "team" / "both"; off by default, v8 byte-identical). The HMM wants a likelihood ratio, so the training base rate is subtracted, and the per-frame logit is divided by its own autocorrelation time (the rule src/context.py uses). Validated leave-one-game-out (fit on two games, decode the third) with the paired bootstrap against v8's core.
FAILED, clearly: learned dead side spellF1 -0.099 [-0.118, -0.081], macroF1 -0.227; learned team side -0.206 [-0.233, -0.181]; both -0.316. Not applied to BAR-ATH.
Why: the scale. The fitted logits are slow (autocorrelation time 568 frames = 23 s on the dead side, 318 on the team side, because ball-missing stretches and team shape last many seconds), so dividing by it shrinks the dead evidence ~15-30x below the hand weights: D share 0.02-0.07 against 0.21-0.30 true. On the team side K_TEAM hits its bracket (20) and still cannot reach 12.5 s possessions (106-117 possessions against 311-376). And a per-frame classifier cannot weigh a slow clue (team shape) against a fast one (ball control) the way an HMM that adds evidence over time needs: it gives context ~1/30 of the hand ratio. All-games dead weights (per frame, before scaling): speed_z 0.58, ball_missing 2.00, ball_fast -1.03, ball_still 0.21 (hand: 0.04, 0.02, -0.05, 0.10): the learned model leans far more on ball missing.

| run (leave one game out) | game 1 spellF1 / macroF1 | game 2 | game 3 | 3-game spellF1 gap [5-95%] | macroF1 gap [5-95%] |
|---|---|---|---|---|---|
| v8 core (reference) | 0.555 / 0.790 | 0.489 / 0.769 | 0.491 / 0.770 | (0.512) | (0.776) |
| learned dead | 0.418 / 0.549 | 0.430 / 0.596 | 0.390 / 0.505 | -0.099 [-0.118, -0.081] | -0.227 [-0.269, -0.188] |
| learned team | 0.317 / 0.744 | 0.311 / 0.746 | 0.287 / 0.728 | -0.206 [-0.233, -0.181] | -0.037 [-0.047, -0.028] |
| learned both | 0.214 / 0.596 | 0.186 / 0.596 | 0.185 / 0.495 | -0.316 [-0.346, -0.288] | -0.214 [-0.255, -0.177] |

**20b: dead side with the scale picked on the training games.** Same learned relative weights, but one overall scale chosen from a fixed grid (0.025 / 0.05 / 0.1 / 0.2) by decoded spellF1 on the two training games, then the held-out game is decoded (`python -m src.learned --dead-scale`). Every fold picked 0.2 (the grid's top edge). Metrica held out: spellF1 +0.080 [+0.064, +0.095], macroF1 +0.037 [+0.017, +0.055] against v8's core: a big gain. BAR-ATH with the all-games model (models/learned.json): phase_agree 0.743 -> 0.749, after-tagged-restart 0.579 -> 0.587, 304 possessions, short 0.134 -> 0.102, D 0.380 -> 0.404 -- BUT the placed-ball checks collapse (D in the 2 s before a placed-ball kick 0.267 -> 0.055, restart within 1 s 0.294 -> 0.059, offset -0.5 -> -5.2 s) and throw-in holds called D 0.58 -> 0.39: it relearned "ball seen = live", undoing v4. Fails the keep rule.
**20c: 20b + the hand still-ball term on top.** BAR-ATH placed-ball restart within 1 s only 0.137 (v8 0.294), offset -5.2 s; phase_agree 0.753, 298 possessions. Still fails.
**Why (aggregate check):** Metrica's tracking blanks the ball during stoppages: ball missing on 90-100% of dead frames against 14-19% of live frames, so "ball missing = dead" is nearly a rule there and the learned model leans on it (raw weight 2.00, ten times the speed weight). On BAR-ATH the ball is missing on 24% of Smart Tagging live-phase frames and only 43% of the first 10 s of set-piece windows: a weak hint. The weight does not transfer, and the Metrica bench overrates any ball-missing evidence (a caveat for every later round). -> 20d: learn only the clues whose meaning is the same on both trackers.
