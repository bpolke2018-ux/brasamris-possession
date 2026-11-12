# Brasamris — ball possession from automatic tracking

Team **Brasamris**'s solution for the More than a Hack 2027 challenge (starter repo:
https://github.com/metrica-sports/more-than-a-hack-2027). It labels every frame of the BAR-ATH
tactical video as home possession (`H`), away possession (`A`) or dead ball (`D`). The method is a
three-state hidden Markov model over evidence from the tracking, decoded by Viterbi.

- **Final version:** v9 (git tag `final-v9`).
- **Submission file:** `submissions/Brasamris.csv`.
- **Approach, data, rules, limitations:** [notes/technical_document.md](notes/technical_document.md),
  also as `Brasamris.pdf`.

## The data is not in this repository

The match data is released only to enrolled participants under the competition terms, must not
be redistributed, and is gitignored here (`data/`, `cache/`, `*.csv` except our own label files,
`*.mp4`, `*.zip`, `outputs/`, `other_games/`).

## Install

Python 3.11 (tested with 3.11.9):

```
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt   # pinned: numpy 2.4.6, pandas 3.1.0
```

## Put the data in place

Unpack the competition data bundle so that `games.py` finds it. It looks in this order:
1. `./data`
2. `../more-than-a-hack-2027/data` (the starter clone next to this repo)
3. the folder named in the `MTAH_DATA_DIR` environment variable

Only the tactical ATD and the Smart Tagging file are read.

## Reproduce the submission

```
python -m src.run                                         # -> submissions/Brasamris.csv
python evaluation/validate_submission.py submissions/Brasamris.csv
```

- **Determinism:** the run is deterministic. On the same data it rebuilds
  `submissions/Brasamris.csv` byte for byte. We checked this from a fresh clone; the file is
  identical to `submissions/v9.csv`.
- **Runtime:** the first run parses the tracking into `cache/` (1–2 minutes); later runs take
  about a minute.
- **No other-game data needed:** the weights learned from the other games ship in
  `models/learned_nomiss_rl.json`.

## Optional: re-learn the weights from the other games

The learned dead-ball weights come from Metrica Sports' public sample data (three anonymised
matches with hand-coded events). They are other games, not BAR-ATH. To re-learn them:

```
git clone --depth 1 https://github.com/metrica-sports/sample-data.git ../other_games
python -m src.learned --dead-scale --no-missing --restart-live   # -> models/learned_nomiss_rl.json
python -m src.other_games                  # v9's core on the 3 games (in-sample, see the note it prints)
python -m src.other_games_boot v9 <tag>    # paired bootstrap between saved versions
```

`notes/other_games.md` describes how the event data becomes H/A/D labels.

## Layout

- **`src/`** — our pipeline:
  - `run.py`: entry point;
  - `atd.py`, `features.py`: tracking to per-frame features;
  - `context.py`: team shape;
  - `model.py`: evidence, HMM, Viterbi;
  - `learned.py`: weights learned on other games;
  - `other_games.py`, `other_games_boot.py`: the other-game bench;
  - `proxy_eval.py`, `restart_report.py`, `restart_team_report.py`: label-free checks.
- **`models/`** — learned weights (from public Metrica data only).
- **`notes/`:**
  - `technical_document.md`;
  - `code_walkthrough.md`: one frame's journey through the code;
  - `experiments.md`: every round, kept or dropped;
  - `other_games.md`.
- **`submissions/`** — `Brasamris.csv` (final, equal to `v9.csv`) and the earlier versions v1–v8,
  for comparison. Each file contains labels only.
- **Copied from the starter repository** (MIT, see LICENSE): `evaluation/`, `games.py`,
  `starter/`.
- **`src/inspect_window.py`** — plots a match window for a person to look at. It saves to the
  gitignored `outputs/`; the plots show match data, so never commit or share them.
