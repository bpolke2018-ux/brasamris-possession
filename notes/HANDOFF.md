# Handoff (2026-11-03)

## Current best: v3 → `submissions/v3.csv`

Run `python -m src.run` (about 15 s once `cache/` exists; the first run parses the ATD, 1–2 min).

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
