"""Raw ATD -> numpy arrays, cached.

Parses the FIFA/EPTS raw text once ("<frame_id>:<p1>;<p2>;...:<ball>", each group "x,y,speed",
NaN for missing) into dense arrays indexed by *video* frame (frame_id - 1), converts coordinates
to metres, and caches the result under cache/ (gitignored) so later runs take a second.

Arrays (n = 199319 frames, k = 46 track slots):
    px, py   (n, k) float32   player position in metres, NaN when the slot is empty
    ps       (n, k) float32   player speed in m/s as reported by the tracker
    team     (k,)   int8      0 home, 1 away
    bx, by   (n,)   float32   ball position in metres, NaN when the ball is not seen
    bs       (n,)   float32   ball speed field as reported (NaN if absent)
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """0/1 per track slot; the first team in the metadata is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _group(text):
    """'x,y,s' -> (x, y, s) floats; NaN for missing pieces."""
    fields = text.split(",")
    vals = [float(f) if f and f[0] != "N" else np.nan for f in fields[:3]]
    while len(vals) < 3:
        vals.append(np.nan)
    return vals


def parse(game=GAME, data_dir=None, source="tactical"):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    n = source_frames(source, game)
    length, width = GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    k = len(team)

    px = np.full((n, k), np.nan, np.float32)
    py = np.full((n, k), np.nan, np.float32)
    ps = np.full((n, k), np.nan, np.float32)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    bs = np.full(n, np.nan, np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n:
                continue
            for slot, g in enumerate(parts[1].split(";")[:k]):
                if g and g[0] != "N":
                    px[f, slot], py[f, slot], ps[f, slot] = _group(g)
            b = parts[2]
            if b and b[0] != "N":
                bx[f], by[f], bs[f] = _group(b)

    px *= length
    py *= width
    bx *= length
    by *= width
    return dict(px=px, py=py, ps=ps, team=team, bx=bx, by=by, bs=bs)


def load(game=GAME, data_dir=None, refresh=False, source="tactical"):
    """Cached arrays; parses on first use (a minute or two). source: "tactical" or "broadcast"
    (the broadcast recording runs on its own clock; see src/broadcast.py)."""
    path = os.path.join(CACHE_DIR, "{}_{}_atd.npz".format(game, source))
    if not refresh and os.path.isfile(path):
        with np.load(path) as z:
            return {key: z[key] for key in z.files}
    arrays = parse(game, data_dir, source)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, **arrays)
    return arrays
