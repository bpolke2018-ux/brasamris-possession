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

