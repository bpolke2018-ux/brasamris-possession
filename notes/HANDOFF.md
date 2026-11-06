# Handoff (2026-11-03, updated 2026-11-05 evening)

## Latest: v6 → `submissions/v6.csv`

- v4 (round 6): a still (placed) ball counts as dead-ball evidence, and ball control is neutral
  for live vs D on still frames. Placed-ball restarts went from 6.4 s early to 0.5 s early.
- e13 (round 7): posterior X abstention was tested and dropped. The held-out tags show those
  frames are right about 64% of the time, far above the ~1/3 break-even.
- v5 (round 8): the set-piece Team tag constrains the restart (D -> other team costs 5 nats,
  -10..+20 s). Phase agreement in the 20 s after tagged restarts went 0.525 -> 0.580; overall 0.735.
- v6 (round 9): a ball held beyond a touchline (the thrower) counts as dead, 0.15 nats. Throw-in
  restart within 1 s of re-entry: 13% -> 47% (15 episodes). A throw-in team rule was not needed (v5 already 93%).
- New checks: `src/restart_report.py`, `src/restart_team_report.py`, `src/abstain_report.py`.
- Brandon is learning alongside: report before and after each experiment in plain language
  (what, football reason, how measured, file), under ~8 lines.
- Next ideas: kickoff detection (ball still at the centre spot, both teams in their own halves); HSMM.


## Current best: v3 → `submissions/v3.csv`

Run `python -m src.run` (about 1 min once `cache/` exists; the first run parses the ATD, 1–2 min more).

The pipeline:
1. **Ball evidence:** nearest-player team evidence, with no credit while the ball moves faster than
   8 m/s.
2. **Team-shape context:** `src/context.py`, a logistic regression learned from the algorithm's own
   pseudo-labels, divided by its ~31 s autocorrelation time.
3. **Dead-ball evidence:** player speed, ball missing, ball out of bounds, set-piece windows.
4. **Decoding:** a 3-state HMM and Viterbi, with `K_TEAM` auto-calibrated to a 12.5 s mean
   possession.

## Proxy results (no ground truth; `src/proxy_eval.py`, log in `notes/experiments.md`)

| version | phase_agree (held-out tags) | possessions | <2 s share | D share |
|---|---|---|---|---|
| baseline_smooth125 | 0.703 | 398 | 0.276 | 0.223 |
| v1 | 0.667 | 334 | 0.078 | 0.319 |
| v2 (flight gate) | 0.708 | 350 | 0.174 | 0.311 |
| **v3 (+ shape context)** | **0.720** | 350 | 0.171 | 0.309 |

- **Context model:** cross-half accuracy against pseudo-labels is 0.84 and 0.85, against a base
  rate of 0.62.
- **Dropped:** control radius 1.5 m (0.672) and a 10-frame persistence rule (0.669). Context ×3
  (0.718) was no better than ×1.

## Update 2026-11-05

The set-piece Team tag (e5), start-of-window D timing (e6, e7) and minimum-possession merges
(e8, e9) were all tried and dropped. Numbers and reasons are in `notes/experiments.md`. v3 is
still the best version.

`notes/technical_document.md` is drafted.

X abstention was reasoned out rather than tested: under spellF1 an X is the same as extending the
previous label, and under macroF1 it is a miss, so on graded frames it cannot gain.

The proxy is biased towards long possessions, so always read it together with the structure
numbers.

Next: an HSMM or restart-burst detector. Then a single public-window sanity score of the final
version, and push to a public GitHub repo (needs Brandon's sign-in).

## Half-done / not started (as of 2026-11-03)

- Restart alignment is poor for every version (~12–14% within 2 s). Before trusting it, check
  where the restart sits inside the fixed 35 s SET PIECES windows. Use aggregate offsets only.
- The set-piece Team tag (who restarts) is not used yet.
- No X abstention yet. The plan is posterior-based: forward-backward, never at likely hand-overs.
- `notes/technical_document.md` is not started. Base it on `notes/v1.md` and this file.
- The public-window score has NOT been computed for v2 or v3. Do it once, only for the version
  you would submit, and label it a sanity check.

## Exact next step

Add the set-piece Team tag as evidence for the restarting side, applied only at the end of each
D stretch inside a set-piece window. Run `python -m src.proxy_eval` on it against v3, then keep
or drop it.

## Rules followed

- No raw data rows, frames or plots were viewed.
- No hand labels.
- Nothing was tuned on the public ground truth.
- Commits are local only; nothing has been pushed.

## Update 2026-11-05 (2)

e10 (ball lost next to a line as D evidence) was dropped: no gain. The proxy cannot judge D changes, because phase_agree only scores H/A frames. Next step: build a label-free dead-ball check before any more D work (for example, agreement with stoppage signs that the model does not use), then try an HSMM.

