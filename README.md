# Brasamris — ball possession from automatic tracking

Team **Brasamris**'s entry to the More than a Hack 2027 challenge
([starter repository](https://github.com/metrica-sports/more-than-a-hack-2027)).

**What it does.** For every frame of the FC Barcelona vs Athletic Club match (game id `BAR-ATH`,
25 frames per second, 199,319 frames of the high "tactical" camera), it writes one label:

| label | meaning |
|---|---|
| `H` | the home team (Barcelona) has the ball |
| `A` | the away team (Athletic Club) has the ball |
| `D` | dead ball: the game is stopped (throw-in, free kick, goal kick, injury...) |
| `X` | not graded (before kickoff, halftime, after the final whistle) |

**How, in one paragraph.** Each frame gets a little evidence for each label from the player and
ball tracking: who is nearest the ball, how the two teams are spread out, how fast the players are
moving, whether the ball is still or out of the pitch. A *hidden Markov model* adds what we know
about football (a possession lasts about 12.5 s on average, a stoppage about 30 s), and the
*Viterbi algorithm* picks the single most likely sequence of labels for each half. No labels from
this match are used anywhere.

- **Submission file:** [`submissions/Brasamris.csv`](submissions/Brasamris.csv) (final version
  v10, git tag `final-v10`).
- **Technical document:** [`notes/technical_document.md`](notes/technical_document.md), also as
  [`Brasamris.pdf`](Brasamris.pdf).
- **New here?** Read [`notes/code_walkthrough.md`](notes/code_walkthrough.md) first. Its §0
  explains log-probabilities and Viterbi with a worked example, then follows one frame through
  the code. The [glossary](#glossary) below explains the terms.

## The match data is not in this repository

The match data is released only to enrolled participants, under the competition terms, and must
not be redistributed. It is gitignored here (`data/`, `cache/`, `*.csv` except our own label
files, `*.mp4`, `*.zip`, `outputs/`, `other_games/`). Enrolled participants get it from the
link in the organisers' email (see `DATA.md` in the starter repository).

## Quick start

You need Python 3.11 (we used 3.11.9) and the data bundle. All commands run from the folder that
holds this README.

**1. Get the code and install** — Windows (PowerShell):

```powershell
git clone https://github.com/bpolke2018-ux/brasamris-possession.git
cd brasamris-possession
python -m venv .venv
.venv\Scripts\Activate.ps1        # if PowerShell refuses: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt   # pinned: numpy 2.4.6, pandas 3.1.0
```

macOS / Linux:

```bash
git clone https://github.com/bpolke2018-ux/brasamris-possession.git
cd brasamris-possession
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Put the data in place.** Unpack the bundle so that it has this layout (only the two files
marked `*` are read):

```
data/
`-- tactical/
    |-- BAR-ATH_tactical_atd.txt            * tracking: players and ball, 25 fps
    |-- BAR-ATH_tactical_atd.xml              tracking metadata
    |-- BAR-ATH_tactical_smart_tagging.xml  * automatic event tags (set pieces, phases)
    `-- ...                                   video and homography (not used)
```

`games.py` looks for that `data/` folder in this order:

1. the folder named in the `MTAH_DATA_DIR` environment variable, if it is set
   (PowerShell: `$env:MTAH_DATA_DIR = "D:\mtah2027"`; macOS/Linux: `export MTAH_DATA_DIR=/path/to/mtah2027`);
2. `./data` inside this repository;
3. `../more-than-a-hack-2027/data`, the data folder of a starter-repository clone next to this one.

**3. Build the submission and check it:**

```
python -m src.run
python evaluation/validate_submission.py submissions/Brasamris.csv
```

`src.run` takes about a minute (plus 1–2 minutes the first time, while it parses the tracking into
`cache/`). It ends by printing three lines like these:

```
wrote submissions\Brasamris.csv  (K_TEAM 1.1669, config {...})
graded frames: H 34.9%  A 22.3%  D 42.8%
possessions (full match): 291  mean duration 12.5 s
```

and the validator ends with `Valid.`

## Reproducibility

- **Deterministic.** There is no randomness anywhere, so the same data gives the same file byte
  for byte. We checked this from a fresh clone, with a new virtual environment and no `cache/`.
- **Fingerprint.** SHA-256 of `submissions/Brasamris.csv`:
  `47c16c959349ee4dfa7de7c976344cd67d95fba42e52c5d09f08247183765d96`
  (PowerShell: `Get-FileHash submissions\Brasamris.csv`; macOS/Linux: `shasum -a 256 submissions/Brasamris.csv`).
- **Tested on** Windows 10 x86-64, Python 3.11.9, the pip wheels pinned in `requirements.txt`.
  Another CPU or maths library could in principle differ in the last decimal place of a sum. If
  your file differs, run `python -m src.run --k-team 1.1669`: it skips the one step (the
  calibration search) where such a tiny difference could matter.
- **Stale cache.** `cache/` holds the parsed tracking. Delete it if you change the data or
  `src/atd.py`.
- **No other-game data needed.** The weights learned from three other games ship in
  `models/learned_nomiss_rl.json`.

## Optional: re-learn the weights from the other games

The learned dead-ball weights come from Metrica Sports' public sample data: three anonymised
matches with hand-coded events, not BAR-ATH. To re-learn them:

```
git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../other_games
python -m src.learned --dead-scale --no-missing --restart-live   # -> models/learned_nomiss_rl.json
python -m src.other_games                  # the pipeline's core on the 3 games
python -m src.other_games_boot <ref> <tag> # paired bootstrap between saved versions
```

`notes/other_games.md` describes how the event data becomes H/A/D labels.

## Layout

**Makes the submission** (`python -m src.run` uses only these):

| file | role |
|---|---|
| `src/run.py` | entry point: evidence, calibration, decoding, writing the CSV |
| `src/atd.py` | reads the tracking file into arrays (cached in `cache/`) |
| `src/tagging.py` | reads the Smart Tagging file |
| `src/features.py` | per-frame features: nearest player per team, speeds, ball still/out |
| `src/context.py` | team-shape model (which team looks like it has the ball) |
| `src/model.py` | evidence, hidden Markov model, Viterbi; every setting is in `CONFIG`, with its reason |
| `src/learned.py` | applies the dead-ball weights learned on the other games |
| `games.py` | match constants and data paths (copied from the starter repository) |
| `models/learned_nomiss_rl.json` | the learned dead-ball weights used by the final version |

**Checks and experiments** (not needed to rebuild the file):

- `src/proxy_eval.py`, `src/restart_report.py`, `src/restart_team_report.py`: label-free checks
  used during development (§4 of the technical document);
- `src/other_games.py`, `src/other_games_boot.py`: the three-other-games bench;
- `src/abstain_report.py`, `src/broadcast.py`, `src/setpiece_report.py`, `src/context_report.py`,
  `src/explore.py`: experiments and diagnostics, all switched off or unused in the final version;
- `src/inspect_window.py`: plots a match window for a person to look at. It saves to the
  gitignored `outputs/`; the plots show match data, so never commit or share them;
- `models/learned.json`, `models/learned_nomiss.json`: weights from dropped experiments.

**Notes:** `notes/technical_document.md` (the write-up), `notes/code_walkthrough.md`,
`notes/experiments.md` (every round, kept or dropped), `notes/other_games.md`.

**Submissions:** `submissions/Brasamris.csv` is the final file. `v1.csv` to `v10.csv` are the
earlier versions, kept for comparison. Each file contains labels only.

**Copied from the starter repository** (MIT, see LICENSE): `evaluation/`, `games.py`, `starter/`.

## Glossary

| term | meaning |
|---|---|
| ATD | the tracking file: x/y position of every player and the ball, 25 times a second |
| Smart Tagging | automatic event tags in the bundle (set pieces, attacking/defending phases) |
| tactical / broadcast | the two camera recordings; the tactical one is the frame reference |
| spellF1 | the ranking metric: how well predicted possessions (whole runs of H or A) match the true ones |
| macroF1 | the secondary metric: per-frame F1 averaged over H, A and D |
| HMM | hidden Markov model: states (H, A, D) that persist over time, with switching costs |
| evidence / emission | how much one frame's tracking favours each state |
| Viterbi | the algorithm that finds the single most likely state sequence |
| nats | units of log-probability; +1 nat means "about 2.7 times more likely" |
| LLR | log-likelihood ratio: evidence for home minus evidence for away, in nats |
| K_TEAM | overall weight of the home/away evidence, calibrated so possessions average 12.5 s |
| POSSESSION_S | that 12.5 s prior: average possession length in elite football |
| pseudo-labels | labels our own algorithm makes on obvious frames, used to train the team-shape model |
| Metrica sample data | three public, anonymised matches with hand-coded events (other games, not BAR-ATH) |
| v1 ... v10 | our successive versions; "round N" and "eN" are the experiments in `notes/experiments.md` |
