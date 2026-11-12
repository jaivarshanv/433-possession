"""Helpers about frames: which are graded, which half they are in, and run-length encoding.

Everything here works on plain numpy arrays indexed by video frame.
"""

import numpy as np

from . import config


def half_of_frame(n_frames=config.N_FRAMES):
    """int8 array: 1 or 2 for frames inside a graded half, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, (first, last) in config.HALVES.items():
        half[first:last + 1] = number
    return half


def graded_mask(n_frames=config.N_FRAMES):
    """bool array: True on the 158,857 frames that are scored."""
    return half_of_frame(n_frames) > 0


def runs(values):
    """Run-length encode a 1-D array.

    Returns a list of (value, start, stop) with stop exclusive, e.g.
        runs(np.array([0, 0, 1, 1, 1, 0]))  ->  [(0, 0, 2), (1, 2, 5), (0, 5, 6)]
    """
    values = np.asarray(values)
    if len(values) == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1   # where a new run begins
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(values)]))
    return [(values[s].item(), int(s), int(e)) for s, e in zip(starts, stops)]


def possession_runs(codes):
    """Possessions (maximal H or A runs) inside graded halves, as (code, start, stop) tuples.

    Same definition as the official spellF1: D runs are stoppages, not possessions, and runs
    never cross the halftime gap.
    """
    out = []
    for first, last in config.HALVES.values():
        for value, start, stop in runs(codes[first:last + 1]):
            if value in (config.H, config.A):
                out.append((value, start + first, stop + first))
    return out


def describe_possessions(codes):
    """Aggregate summary of a label array: possession count, mean / median duration in seconds.

    Safe to print -- it contains no per-frame information.
    """
    spells = possession_runs(codes)
    if not spells:
        return {"n_possessions": 0, "mean_s": 0.0, "median_s": 0.0}
    lengths = np.array([stop - start for _, start, stop in spells]) / config.FPS
    return {
        "n_possessions": len(spells),
        "mean_s": round(float(lengths.mean()), 2),
        "median_s": round(float(np.median(lengths)), 2),
        "share_under_1s": round(float((lengths < 1.0).mean()), 3),
    }
