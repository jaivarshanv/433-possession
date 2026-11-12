"""Metrica Sports' open sample games: download them, and load them as plain arrays + events.

    github.com/metrica-sports/sample-data   (pinned to config.METRICA_COMMIT)

Three anonymised full matches, 25 fps, CORRECTED tracking (every player, ball almost always
present) plus hand-made event data. They are the closest thing to our match that is openly
licensed: same provider, same coordinate convention (0..1 of the pitch), and game 3 is even in
the same FIFA/EPTS text format as our ATD.

This file only downloads and reads. Turning the events into H/A/D labels lives in
event_labels.py, making the tracking look like ours lives in degrade.py.
"""

import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config

HOME, AWAY = 0, 1

# Both pitch sizes come from the dataset itself (README for games 1-2, metadata XML for game 3).
PITCH_M = (105.0, 68.0)

RAW_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/{commit}/data/{path}"

# Every file we need, with its sha256, so a corrupted or changed download is caught at once.
FILES = {
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv":
        "097dda6345e17390dca095331458fd00e47960c480c76f4356fd4edf2575ef05",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv":
        "0a3d0eff7785950379c587d3597b86a2d91c2492281b4652e1abb97c99b6b364",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv":
        "0a3df68a92af0290ede472e62de3b6489ddee29223da09bdf0cf76c8e0ab0618",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv":
        "edf31a18599265b77a8baf150f2ce6d89456fb0324b62ea7a391229657a619ba",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv":
        "3f9990ac204b07133e4202441d6fcbb386897c7c7f81e1bc3ce4d35bc8644694",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv":
        "aa639f40762eade46c389020fe4458266b5ba14027b1e96e22e75db62803718b",
    "Sample_Game_3/Sample_Game_3_events.json":
        "28a53bfe9eaf32535362335553d517cdce0eabb588af6166009081753751f4e0",
    "Sample_Game_3/Sample_Game_3_metadata.xml":
        "009c7dcfcb66192014917ca1559c158f612353e390eafb85f8214b06a1c7dec1",
    "Sample_Game_3/Sample_Game_3_tracking.txt":
        "570e6ce82c2d93bc92723af331d93c31110bec1d6dc0be07c6d44ca3f981d658",
}


def sha256_of(path):
    """Hex sha256 of a file, read in chunks so big files do not fill memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force=False, verbose=True):
    """Fetch every file in FILES into config.METRICA_DIR, checking each sha256.

    Already-present files with the right hash are skipped. Raises if a hash does not match, so
    the harness can never silently run on different data. Returns the list of local paths.
    """
    paths = []
    for relative, expected in FILES.items():
        target = config.METRICA_DIR / relative
        if target.exists() and not force and (expected is None or sha256_of(target) == expected):
            paths.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = RAW_URL.format(commit=config.METRICA_COMMIT, path=relative)
        if verbose:
            print("downloading", relative)
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, partial)     # write to .part, rename when complete
        partial.replace(target)
        actual = sha256_of(target)
        if expected is not None and actual != expected:
            raise IOError("{}: sha256 {} != pinned {}".format(relative, actual, expected))
        paths.append(target)
    return paths


# ---------------------------------------------------------------------------------------------
# Loading. Every game comes out in ONE common shape, whatever its file format:
#
#   px, py   (N, P) float64  normalised position of each PLAYER (a person, not a tracklet),
#                            NaN while that player is not on the pitch (substitutes)
#   team     (P,)   int8     0 home, 1 away
#   ball     (N, 2) float64  normalised ball x, y; NaN when the ball is not tracked. In this
#                            corrected tracking that is almost exactly "ball out of play".
#   period   (N,)   int8     1 or 2
#   events   DataFrame       one row per event, see load_events()
#
# Array row i is tracking frame i + 1 (the files count frames from 1), same as our ATD.
# ---------------------------------------------------------------------------------------------

def game_dir(number):
    """Folder holding the raw files of sample game 1, 2 or 3."""
    return config.METRICA_DIR / "Sample_Game_{}".format(number)


def read_csv_team(path):
    """One team's tracking CSV (games 1-2) -> (period, frame, px, py, ball).

    The file has three header lines: team, jersey number, then column names. Each player
    takes two columns, "PlayerN" (x) and an unnamed one after it (y); the ball is last.
    """
    table = pd.read_csv(path, skiprows=2)
    columns = list(table.columns)
    players = [name for name in columns if name.startswith("Player")]
    px = np.stack([table[name].values for name in players], axis=1).astype(np.float64)
    py = np.stack([table[columns[columns.index(name) + 1]].values for name in players],
                  axis=1).astype(np.float64)
    ball_col = columns.index("Ball")
    ball = table.iloc[:, [ball_col, ball_col + 1]].values.astype(np.float64)
    return table["Period"].values.astype(np.int8), table["Frame"].values, px, py, ball


def load_tracking_csv(number):
    """Games 1-2: join the home and away CSVs into the common arrays."""
    folder = game_dir(number)
    stem = "Sample_Game_{}_RawTrackingData_{}_Team.csv"
    period, frame_h, hx, hy, ball = read_csv_team(folder / stem.format(number, "Home"))
    _, frame_a, ax, ay, _ = read_csv_team(folder / stem.format(number, "Away"))
    expected = np.arange(1, len(frame_h) + 1)
    if not (np.array_equal(frame_h, expected) and np.array_equal(frame_a, expected)):
        raise ValueError("game {}: home/away frames do not line up as 1..N".format(number))
    team = np.array([HOME] * hx.shape[1] + [AWAY] * ax.shape[1], dtype=np.int8)
    return {"px": np.hstack([hx, ax]), "py": np.hstack([hy, ay]), "team": team,
            "ball": ball, "period": period}


def load_tracking_epts(number):
    """Game 3: FIFA/EPTS text tracking, the same line format as our ATD (x,y per group).

    Unlike our ATD the GROUP ORDER CHANGES during the match: the metadata has one
    <DataFormatSpecification> per stretch of frames (a new one after each substitution),
    listing which player each group belongs to. We map every group back to its player so
    the output has one fixed column per person, like games 1-2.
    """
    folder = game_dir(number)
    root = ET.parse(str(folder / "Sample_Game_{}_metadata.xml".format(number))).getroot()

    home_id = root.find(".//Score").get("idLocalTeam")
    player_team = {p.get("id"): (HOME if p.get("teamId") == home_id else AWAY)
                   for p in root.iter("Player")}
    player_ids = sorted(player_team, key=lambda pid: (player_team[pid], pid))
    column_of = {pid: i for i, pid in enumerate(player_ids)}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    # For every frame stretch: the player column of each group, in file order.
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        x_refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")
                  if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [column_of[channel_player[r]] for r in x_refs]))

    lines = (folder / "Sample_Game_{}_tracking.txt".format(number)).read_text().split()
    n = len(lines)
    px = np.full((n, len(player_ids)), np.nan)
    py = np.full((n, len(player_ids)), np.nan)
    ball = np.full((n, 2), np.nan)
    for line in lines:
        frame_text, groups_text, ball_text = line.split(":")
        frame = int(frame_text)
        row = frame - 1
        columns = next(cols for start, stop, cols in specs if start <= frame <= stop)
        for column, group in zip(columns, groups_text.split(";")):
            x, y = group.split(",")
            px[row, column] = float(x)   # float("NaN") is NaN
            py[row, column] = float(y)
        bx, by = ball_text.split(",")
        ball[row] = float(bx), float(by)

    # Periods from the provider parameters (first_half_start ... second_half_end).
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    period = np.zeros(n, dtype=np.int8)
    for half, key in ((1, "first_half"), (2, "second_half")):
        start, stop = int(params[key + "_start"]), int(params[key + "_end"])
        period[start - 1:stop] = half
    team = np.array([player_team[pid] for pid in player_ids], dtype=np.int8)
    return {"px": px, "py": py, "team": team, "ball": ball, "period": period}


def subtype_text(raw):
    """Metrica subtypes come as NaN, a string, a dict or a list of dicts -> one string.

    Joined with '-' exactly as games 1-2 spell them ("HEAD-INTERCEPTION"), "" when absent.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    if isinstance(raw, str):
        return raw.strip().upper()
    if isinstance(raw, dict):
        raw = [raw]
    return "-".join(item["name"].strip().upper() for item in raw)


def load_events(number):
    """Events of one game as a DataFrame sorted by start frame, columns:

        team (H/A)  type  subtype  period  start  end  player  to

    start / end are ARRAY ROWS (tracking frame - 1), the same index as the tracking arrays.
    player / to are only used to recognise "the same player" within one game.
    """
    folder = game_dir(number)
    rows = []
    if number in (1, 2):
        table = pd.read_csv(folder / "Sample_Game_{}_RawEventsData.csv".format(number))
        for _, event in table.iterrows():
            rows.append({
                "team": "H" if event["Team"] == "Home" else "A",
                "type": str(event["Type"]).strip().upper(),
                "subtype": subtype_text(event["Subtype"]),
                "period": int(event["Period"]),
                "start": int(event["Start Frame"]) - 1,
                "end": int(event["End Frame"]) - 1,
                "player": event["From"] if isinstance(event["From"], str) else "",
                "to": event["To"] if isinstance(event["To"], str) else "",
            })
    else:
        meta = ET.parse(str(folder / "Sample_Game_{}_metadata.xml".format(number))).getroot()
        home_id = meta.find(".//Score").get("idLocalTeam")
        with open(folder / "Sample_Game_{}_events.json".format(number)) as handle:
            data = json.load(handle)["data"]
        for event in data:
            rows.append({
                "team": "H" if event["team"]["id"] == home_id else "A",
                "type": event["type"]["name"].strip().upper(),
                "subtype": subtype_text(event.get("subtypes")),
                "period": int(event["period"]),
                "start": int(event["start"]["frame"]) - 1,
                "end": int(event["end"]["frame"]) - 1,
                "player": (event.get("from") or {}).get("name") or "",
                "to": (event.get("to") or {}).get("name") or "",
            })
    events = pd.DataFrame(rows)
    # Instantaneous events are sometimes written with end frame 0; an event never ends early.
    events["end"] = np.maximum(events["end"], events["start"])
    return events.sort_values("start", kind="stable").reset_index(drop=True)


def clean_cache_path(number):
    """Where the parsed clean tracking of one sample game is kept (external/derived/)."""
    return config.EXTERNAL_CACHE_DIR / "metrica{}_clean.npz".format(number)


def load_game(number):
    """One sample game in the common shape (dict, see top of section). Parses once, then caches.

    Also adds: name ("metrica1"), pitch (length, width in metres), fps.
    """
    path = clean_cache_path(number)
    if not path.exists():
        download(verbose=True)
        arrays = load_tracking_csv(number) if number in (1, 2) else load_tracking_epts(number)
        config.EXTERNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
    with np.load(path) as stored:
        game = {key: stored[key] for key in stored.files}
    game["events"] = load_events(number)
    game["name"] = "metrica{}".format(number)
    game["pitch"] = PITCH_M
    game["fps"] = config.FPS
    return game
