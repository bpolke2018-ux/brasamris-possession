# More than a Hack 2027 — team solution

Our work for the More than a Hack 2027 challenge
(starter repo: https://github.com/metrica-sports/more-than-a-hack-2027).

**The match data is not in this repository.** It is released only to enrolled participants under
the competition terms, must not be redistributed, and is gitignored here (`data/`, `*.mp4`,
`*.zip`, `*.csv`). Keep it in the starter clone's `data/` folder, or point `MTAH_DATA_DIR` at it.

## Run

```
pip install -r requirements.txt            # numpy, pandas
python -m src.explore                      # aggregate stats -> notes/exploration.md
python -m src.run                          # -> submissions/v1.csv
python evaluation/validate_submission.py submissions/v1.csv
```

`games.py` finds the data in `./data`, then `../more-than-a-hack-2027/data`, unless
`MTAH_DATA_DIR` is set. The first run parses the tracking into `cache/` (gitignored).

## Layout

- `src/` — our pipeline (atd parsing, features, HMM, run). Approach: [notes/v1.md](notes/v1.md)
- `evaluation/`, `games.py`, `starter/` — copied from the starter repository (MIT, see LICENSE)
