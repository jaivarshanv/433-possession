"""decoder.py (HSMM) on SYNTHETIC scores and labels."""

import numpy as np

from possession import config, decoder

H, A, D, X = config.H, config.A, config.D, config.X
DURATIONS = {"possession": (np.log(8.0), 1.0), "dead": (np.log(15.0), 1.0)}


def labels_with_segments(pattern, length):
    """X frame, then segments of `length` frames following `pattern`, then X."""
    return np.array([X] + [s for s in pattern for _ in range(length)] + [X], dtype=np.int8)


def test_duration_table_is_normalised():
    table = decoder.duration_table(DURATIONS, block=5, max_seconds=60)
    assert table.shape == (3, 300)
    np.testing.assert_allclose(np.logaddexp.reduce(table, axis=1), 0.0, atol=1e-9)
    assert np.argmax(table[D]) > np.argmax(table[H])          # stoppages are longer


def test_fit_durations_and_transitions():
    gt = labels_with_segments([H, A, D, H, A, D, H], 100)     # 100 frames = 4 s each
    durations = decoder.fit_durations([gt])
    assert np.isclose(durations["possession"][0], np.log(4.0))
    assert durations["possession"][1] == decoder.MIN_SIGMA   # spread 0 is floored
    log_t = decoder.fit_transitions([gt])
    # possession -> other team: H->A twice, plus Laplace 1 = 3; -> dead: twice + 1 = 3
    assert np.isclose(np.exp(log_t[H, A]), 0.5) and np.isclose(log_t[A, H], log_t[H, A])
    assert np.isclose(np.exp(log_t[D, H]), 0.5)
    assert log_t[H, H] == decoder.CONTINUE_LOG


def test_viterbi_recovers_clear_segments():
    scores = np.zeros((60, 3))
    scores[:20, H] = scores[20:40, D] = scores[40:, A] = 3.0
    log_dur = decoder.duration_table(DURATIONS, block=5, max_seconds=60)
    log_t = decoder.fit_transitions([labels_with_segments([H, D, A], 50)])
    segs = decoder.viterbi_blocks(scores, log_dur, log_t)
    assert [(s, a, b) for s, a, b in segs] == [(H, 0, 20), (D, 20, 40), (A, 40, 60)]


def test_refine_moves_boundary_to_the_true_frame():
    scores = np.zeros((40, 3))
    scores[:13, H] = 1.0
    scores[13:, A] = 1.0
    refined = decoder.refine_boundaries([(H, 0, 10), (A, 10, 40)], scores, block=5)
    assert refined == [(H, 0, 13), (A, 13, 40)]


def test_decode_marks_outside_halves_x():
    n = 1000
    half = np.zeros(n, dtype=np.int8)
    half[100:900] = 1
    scores = np.zeros((n, 3))
    scores[:, H] = 1.0
    scores[500:900, H] = 0.0
    scores[500:900, A] = 1.0
    codes = decoder.decode(scores, half, DURATIONS, decoder.fit_transitions(
        [labels_with_segments([H, A, D], 50)]))
    assert (codes[:100] == X).all() and (codes[900:] == X).all()
    assert (codes[100:500] == H).all() and (codes[500:900] == A).all()
