"""Training data and leave-one-game-out (LOGO) for anything fitted on external labels.

Every learned piece of a predictor (duration priors, rule-cell probabilities, classifiers) is
fitted ONLY on degraded external games (CLAUDE.md 0.3). When the harness scores external game
k, the model is fitted on the other games, so the score is out-of-sample. On our match the
model is fitted on all the external games.

    training_set(exclude)        [(features, info, gt)] of the external games except `exclude`
    LogoPredictor(make_model)    wraps a model class into a harness predictor (see harness.py)

A model is any object with  fit(train) -> self  and  predict(features, info) -> codes.
"""

from functools import lru_cache

from . import config, features, harness

TRAIN_REPLICATES = 1        # degradation replicates of each training game


@lru_cache(maxsize=None)
def external_example(number, replicate):
    """(features, info, gt) of one degraded external game. Cached: built once per process."""
    track, info, gt = harness.external_inputs(number, replicate)
    return features.compute(track, info), info, gt


def training_set(exclude=None, replicates=TRAIN_REPLICATES):
    """Feature/label triples of every external game whose name is not `exclude`."""
    return [external_example(number, replicate)
            for number in config.METRICA_GAMES if "metrica{}".format(number) != exclude
            for replicate in range(replicates)]


class LogoPredictor:
    """A harness predictor that fits one model per held-out game.

    predict(track, info): computes the features, fits (once) a model on every external game
    except info["game"], and returns that model's codes. For our match info["game"] is the
    match name, which excludes nothing: the model is fitted on all external games.
    """

    def __init__(self, make_model, replicates=TRAIN_REPLICATES):
        self.make_model = make_model
        self.replicates = replicates
        self.models = {}

    def model_for(self, game):
        """The model fitted without `game` (built on first use)."""
        if game not in self.models:
            self.models[game] = self.make_model().fit(training_set(game, self.replicates))
        return self.models[game]

    def __call__(self, track, info):
        feats = features.compute(track, info)
        return self.model_for(info["game"]).predict(feats, info)
