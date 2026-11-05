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

