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

