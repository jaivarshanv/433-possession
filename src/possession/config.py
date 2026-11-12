"""Every path and match constant used by the project.

Nothing else in the code base spells out a filename or a frame number: import it from here.
That way a moved data folder or a corrected constant is a one-line change.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------

# Project root = two levels above this file (src/possession/config.py -> project root).
ROOT = Path(__file__).resolve().parents[2]

# The organisers' public repository, checked out as a git submodule.
OFFICIAL_DIR = ROOT / "official"

# Where the match bundle lives. MTAH_DATA_DIR overrides it, same as in the official repo.
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))

# The official code reads MTAH_DATA_DIR too. Set it so both code bases look in the same place.
os.environ["MTAH_DATA_DIR"] = str(DATA_DIR)

# Derived arrays (gitignored, never shown to the model). MTAH_CACHE_DIR moves it, e.g. to
# prove the submission is rebuilt identically from an empty cache (run.py --check-repro).
CACHE_DIR = Path(os.environ.get("MTAH_CACHE_DIR", ROOT / "cache"))
SUBMISSIONS_DIR = ROOT / "submissions"  # our CSVs (gitignored)
EXTERNAL_DIR = ROOT / "external"        # open data from other games (gitignored)

# Metrica Sports' open sample games (github.com/metrica-sports/sample-data): our external
# validation set. Pinned to one commit so the files can never change under us.
METRICA_DIR = EXTERNAL_DIR / "metrica"
METRICA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
METRICA_GAMES = (1, 2, 3)
# Parsed external games are cached here (re-built on demand); MTAH_EXTERNAL_CACHE_DIR moves it.
EXTERNAL_CACHE_DIR = Path(os.environ.get("MTAH_EXTERNAL_CACHE_DIR", EXTERNAL_DIR / "derived"))

# The public 5-minute ground truth. For CHECKING the pipeline only, never for tuning.
PUBLIC_GT = OFFICIAL_DIR / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
OFFICIAL_RESULTS = OFFICIAL_DIR / "evaluation" / "results"

# ---------------------------------------------------------------------------------------------
# Match constants (from DATA.md / games.py)
# ---------------------------------------------------------------------------------------------

GAME = "BAR-ATH"
FPS = 25                        # frames per second
N_FRAMES = 199_319              # tactical video length = submission length
N_FRAMES_BROADCAST = 164_488    # broadcast video length (its own timeline)
PITCH_LENGTH_M = 105.3          # metres, measured from corrected tracking
PITCH_WIDTH_M = 68.0

# Graded frames, inclusive on both ends, in video frames.
HALVES = {
    1: (11_185, 90_651),
    2: (110_808, 190_197),
}
# Kickoff frames. Note half 2 is graded from 110808 but kicks off at 113742:
# the 2934 frames in between are graded and all dead ball.
KICKOFF = {1: 11_185, 2: 113_742}

# Public ground truth window (inclusive).
PUBLIC_WINDOW = (11_185, 18_684)

# Label codes, identical to the official scorer (possession_metrics.LABELS).
H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")

# ---------------------------------------------------------------------------------------------
# Asset file names
# ---------------------------------------------------------------------------------------------

SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}


def asset(name, source="tactical"):
    """Path of one provided file, e.g. asset("atd_raw") or asset("video", "broadcast").

    Does not check that the file exists -- see scripts/00_check_data.py for that.
    """
    if source not in ("tactical", "broadcast"):
        raise ValueError("source must be 'tactical' or 'broadcast', got {!r}".format(source))
    if name == "smart_tagging" and source != "tactical":
        raise ValueError("Smart Tagging exists only for the tactical recording")
    return DATA_DIR / source / "{}_{}{}".format(GAME, source, SUFFIXES[name])


def n_frames_of(source="tactical"):
    """Number of video frames for a recording."""
    return N_FRAMES if source == "tactical" else N_FRAMES_BROADCAST


def add_official_to_path():
    """Make the official modules importable: games, possession_metrics, starter loaders.

    Called by the modules that wrap official code. Adding to sys.path only once.
    """
    for folder in (OFFICIAL_DIR, OFFICIAL_DIR / "evaluation", OFFICIAL_DIR / "starter"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    if not (OFFICIAL_DIR / "games.py").exists():
        raise FileNotFoundError(
            "official/ is empty. Run:  git submodule update --init --recursive")
