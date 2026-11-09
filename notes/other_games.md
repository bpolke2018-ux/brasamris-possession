# Other games: held-out robustness test (Metrica Sports sample data)

**Source.** github.com/metrica-sports/sample-data. Brandon downloaded it himself on 2026-11-09 into
`D:\Barcelona Hackathon\other_games`, outside the repo, and it is never committed. Three games, all
tracked at 25 fps in normalised coordinates (we scale to 105 x 68 m), with no player-speed field
(we compute it from positions) and with hand-coded event data. Games 1 and 2 come as CSV per team
plus CSV events. Game 3 comes as EPTS text, like our own match, with its 22 player slots remapped
to different players at each substitution (11 segments in the metadata), plus JSON events, which
we convert to the same table. Game 3's events also have CARRY, which counts as an on-ball action
for possession, the same as a pass.

**Purpose.** To measure whether each change we kept on BAR-ATH also helps on games the model has
never seen. These games are not used to choose BAR-ATH settings, beyond dropping a change that
clearly fails to generalise, and that only with Brandon's say-so. Only aggregates are reported.

**Run.** `python -m src.other_games [--set key=value ...] [--name label] [--save tag]` (about
2 minutes for the three games; it caches the parsed tracking in `cache/metrica_g{1,2,3}.npz`,
which is gitignored). `python -m src.other_games_boot ref tagA tagB ...` gives paired-bootstrap
intervals on the gaps between saved versions: the same resampled ground-truth possessions are used
for every version, as in the official scorer's bootstrap, and the three games are averaged.

## Labels derived from the events (rules fixed before any result was seen)

- **Graded frames:** from each period's first KICK OFF to the end of that period's last event.
  All other frames are X.
- **Team in possession:** the team of the latest PASS, CARRY, SHOT, SET PIECE or RECOVERY. CHALLENGE and
  BALL LOST alone do not change possession, because control has to change hands.
- **Dead ball:** runs from a stoppage to the next restart (a SET PIECE, or a PASS with subtype
  GOAL KICK). A stoppage is any of:
  - BALL OUT (at its end);
  - a SHOT that goes out or scores (at its end);
  - FAULT RECEIVED (at its start);
  - a BALL LOST for offside, hand ball, end of half or hitting the referee (at its end).

  A stoppage is ignored if live play (a non-restart on-ball event) comes before the next restart,
  for example when advantage is played.
- **Sanity check of these labels:**

  | | game 1 | game 2 | game 3 |
  |---|---|---|---|
  | possessions | 352 | 311 | 376 |
  | mean possession | 11.4 s | 13.3 s | 12.0 s |
  | D share | 0.30 | 0.26 | 0.21 |

  These sit within football norms (300–400 possessions of 10–15 s). The D share is probably a
  little low, because a dead stretch only starts at a coded stoppage event.
- **Not comparable with the hackathon's ground truth:** the hackathon labels follow a different
  convention, so the absolute scores here cannot be set against BAR-ATH. Compare rows here only
  with each other.

## What runs

v7's core on BAR-ATH settings. Three parts are off, because they need Smart Tagging, which
Metrica does not have:
- set-piece windows;
- the tagged-restart penalty;
- the pre-kickoff rule. Pre-kickoff frames are ungraded here anyway.

K_TEAM is auto-calibrated per game to a 12.5 s mean, as on BAR-ATH.

## Results

See notes/experiments.md, rounds 16 and 17.
