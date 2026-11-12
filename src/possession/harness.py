"""External validation harness: run any predictor on degraded external games, score officially.

This is where every model decision is made (CLAUDE.md 0.2: never on the public 5 minutes).

THE PREDICTOR CONTRACT. A predictor is any function

    predict(track, info) -> codes

  track  dict in the format of atd.load_cache(): ball (N,3), px, py (N,S), ps (N,S), team (S,),
         seen (N,). For an external game this is the DEGRADED tracking (degrade.py).
  info   dict of what the organisers also give us for our match:
           name     game name
           pitch    (length_m, width_m)
           fps      25
           half     (N,) int8: 1 or 2 on graded frames, 0 elsewhere
           kickoff  {1: frame, 2: frame}
  codes  (N,) integer array of label codes H=0, A=1, D=2, X=3 (config.H ... config.X)

The same function runs on our match via target_inputs(), so nothing is ever written twice.

SCORING. Each game is scored with the official evaluate() against its derived labels
(event_labels.py). The headline is POOLED: all games laid end to end, separated by ungraded X
frames, and evaluated once -- one long "match" with six halves, so every possession of every
game counts once (a micro-average), exactly as the official scorer would rank a full match.
"""

import numpy as np

from . import config, degrade, event_labels, frames, metrica, scoring

SEPARATOR = 1000    # ungraded frames between games in the pooled evaluation


def external_inputs(number, replicate=0, clean=False):
    """(track, info, gt_codes) for one degraded Metrica sample game.

    clean=True hands the predictor the CLEAN tracking instead -- a diagnostic only, to see how
    much of a score the degradation costs. Never select a model on it.
    """
    game = metrica.load_game(number)
    gt, _ = event_labels.derive_labels(game)
    track = degrade.clean_tracking(game) if clean else degrade.degrade(game, gt, replicate)
    half = np.where(gt != config.X, game["period"], 0).astype(np.int8)
    events = game["events"]
    kickoffs = events[(events["type"] == "SET PIECE")
                      & events["subtype"].str.startswith("KICK OFF")]
    info = {
        "name": "{}#{}".format(game["name"], replicate),
        "replicate": replicate,
        "pitch": game["pitch"],
        "fps": game["fps"],
        "half": half,
        "kickoff": {p: int(kickoffs.loc[kickoffs["period"] == p, "start"].min()) for p in (1, 2)},
    }
    return track, info, gt


def target_inputs():
    """(track, info) for OUR match, in exactly the same shapes. Reads the cache (match data):
    whatever a script does with the result, it may only print aggregates."""
    from . import atd
    info = {
        "name": config.GAME,
        "pitch": (config.PITCH_LENGTH_M, config.PITCH_WIDTH_M),
        "fps": config.FPS,
        "half": frames.half_of_frame(),
        "kickoff": dict(config.KICKOFF),
    }
    return atd.load_cache("tactical"), info


def build_inputs(games=config.METRICA_GAMES, replicates=1, clean=False):
    """[(track, info, gt)] for every game and degradation replicate, built once."""
    return [external_inputs(number, replicate, clean)
            for number in games for replicate in range(1 if clean else replicates)]


def check_prediction(codes, n_frames, name):
    """Raise a readable error unless codes is a length-N integer array of 0..3."""
    codes = np.asarray(codes)
    if codes.shape != (n_frames,):
        raise ValueError("{}: predictor returned shape {}, expected ({},)".format(
            name, codes.shape, n_frames))
    if codes.dtype.kind not in "iu" or codes.min() < 0 or codes.max() > config.X:
        raise ValueError("{}: predictor must return integer codes 0..3".format(name))
    return codes.astype(np.int8)


def pooled_arrays(pairs):
    """Lay (gt, pred) pairs end to end with SEPARATOR ungraded frames between them."""
    gap = np.full(SEPARATOR, config.X, dtype=np.int8)
    gts, preds = [], []
    for gt, pred in pairs:
        gts.extend([gt, gap])
        preds.extend([pred, gap])
    return np.concatenate(gts), np.concatenate(preds)


def evaluate(predict, inputs=None, bootstrap_n=0):
    """Score one predictor on every prepared (track, info, gt). Returns a dict:

        per_game   {name: official evaluate() result}
        pooled     official evaluate() result on all games end to end (the headline)
        interval   (low, high) 5-95% bootstrap of the pooled spellF1, or None

    The interval is computed on degradation replicate 0 only: replicates re-use the same true
    possessions, so resampling them as if independent would make the interval too narrow.
    inputs defaults to build_inputs(); pass a pre-built list to score many predictors without
    rebuilding the degraded games each time.
    """
    if inputs is None:
        inputs = build_inputs()
    per_game, pairs, first = {}, [], []
    for track, info, gt in inputs:
        pred = check_prediction(predict(track, info), len(gt), info["name"])
        per_game[info["name"]] = scoring.score_codes(gt, pred)
        pairs.append((gt, pred))
        if info.get("replicate", 0) == 0:
            first.append((gt, pred))
    pooled = scoring.score_codes(*pooled_arrays(pairs))
    interval = None
    if bootstrap_n:
        boot = scoring.score_codes(*pooled_arrays(first), bootstrap_n=bootstrap_n)
        interval = (boot["spell_bootstrap_low"], boot["spell_bootstrap_high"])
    return {"per_game": per_game, "pooled": pooled, "interval": interval}
