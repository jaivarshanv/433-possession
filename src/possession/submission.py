"""Write a submission CSV and validate it with exactly the official checks."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def sha256_of(path):
    """Hex sha256 of a file -- how we prove two runs produced the identical submission."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def codes_to_labels(codes):
    """int codes 0..3 -> array of 'H'/'A'/'D'/'X'."""
    return np.array(config.LABELS)[np.asarray(codes)]


def validate_file(path):
    """Run the official validation on a CSV. Returns None if valid, else the reason string."""
    config.add_official_to_path()
    import possession_metrics as pm
    frame = pm.read_label_csv(str(path))
    return pm.validate_submission(frame, config.N_FRAMES)


def write_submission(labels, name, path=None):
    """Write submissions/<name>.csv (or `path`) from an array of N_FRAMES labels (strings or codes).

    Validates the written file immediately and raises if it fails, so a broken file can never
    reach the upload form unnoticed. Returns the path.
    """
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":            # integer codes -> letters
        labels = codes_to_labels(labels)
    if len(labels) != config.N_FRAMES:
        raise ValueError("need {} labels, got {}".format(config.N_FRAMES, len(labels)))

    path = Path(path) if path else config.SUBMISSIONS_DIR / "{}.csv".format(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(config.N_FRAMES), "label": labels}).to_csv(path, index=False)

    problem = validate_file(path)
    if problem:
        raise ValueError("{} failed validation: {}".format(path.name, problem))
    return path
