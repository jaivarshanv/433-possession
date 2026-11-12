"""Hidden semi-Markov decoding: per-frame evidence over {H, A, D} -> whole possessions.

Why semi-Markov: the ranking metric scores whole possessions, and the truth has realistic
durations (possessions of ~1-60 s, stoppages of ~3-120 s). A plain HMM implies geometric
durations, i.e. it is most comfortable with ultra-short segments; an HSMM gives every segment
an explicit duration probability, so a 0.2 s "possession" made of one noisy frame costs a lot.

Model
  states       H, A, D (config codes 0, 1, 2)
  durations    log-normal in seconds, one for possessions (H and A share it: the model must
               not prefer a side) and one for stoppages; FITTED ON EXTERNAL LABELS only
  transitions  H<->A, H/A -> D, D -> H/A, again fitted on external labels and side-symmetric.
               A segment may also continue in its own state at CONTINUE_LOG cost, so segments
               longer than MAX_SECONDS can still be represented (same label either side)
  emissions    a per-frame score for each state, built by the evidence modules

Viterbi runs on blocks of BLOCK frames (fast), then each boundary is moved to the best frame
within +/- BLOCK (full frame rate). One graded half at a time; frames outside halves get X.
"""

import numpy as np

from . import config

H, A, D, X = config.H, config.A, config.D, config.X
N_STATES = 3
BLOCK = 5                 # frames per decoding step (0.2 s)
MAX_SECONDS = 180.0       # longest single segment the duration model represents
CONTINUE_LOG = -8.0       # cost of chaining a segment onto one of the same state
MIN_SIGMA = 0.3           # floor on the log-normal spread, so a fit never becomes a spike


# --- fitting durations and transitions on external labels -----------------------------------

def segments_of(codes):
    """[(state, start, stop)] constant runs of H/A/D inside the graded (non-X) frames.

    Runs never cross an X gap (halftime), like the official scorer's segments().
    """
    out = []
    graded = codes != X
    edges = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    for first, last in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        window = codes[first:last]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            out.append((int(window[a]), int(first + a), int(first + b)))
    return out


def fit_durations(label_arrays, fps=config.FPS):
    """{'possession': (mu, sigma), 'dead': (mu, sigma)} of log(duration in seconds).

    The first and last segment of each half are left out: they are cut by the half's edges
    (the pre-kick-off stretch is not a normal stoppage).
    """
    logs = {"possession": [], "dead": []}
    for codes in label_arrays:
        for state, start, stop in segments_of(codes):
            cut = start == 0 or codes[start - 1] == X or stop == len(codes) or codes[stop] == X
            if not cut:
                logs["dead" if state == D else "possession"].append(np.log((stop - start) / fps))
    return {key: (float(np.mean(v)), max(MIN_SIGMA, float(np.std(v)))) for key, v in logs.items()}


def fit_transitions(label_arrays):
    """3x3 log transition matrix between consecutive segments, side-symmetric, fitted on labels.

    From a possession: P(to the other team) vs P(to dead ball), pooled over H and A.
    From dead ball: H and A equally likely (the evidence decides who restarts).
    Disallowed moves get -inf. Self-continuation gets CONTINUE_LOG (see module docstring).
    """
    to_other, to_dead = 1, 1                                  # Laplace smoothing
    for codes in label_arrays:
        segs = segments_of(codes)
        for (s1, _, stop1), (s2, start2, _) in zip(segs[:-1], segs[1:]):
            if stop1 != start2 or s1 == D:
                continue
            if s2 == D:
                to_dead += 1
            else:
                to_other += 1
    p_other = to_other / (to_other + to_dead)
    log_t = np.full((N_STATES, N_STATES), -np.inf)
    log_t[H, A] = log_t[A, H] = np.log(p_other)
    log_t[H, D] = log_t[A, D] = np.log(1 - p_other)
    log_t[D, H] = log_t[D, A] = np.log(0.5)
    for s in range(N_STATES):
        log_t[s, s] = CONTINUE_LOG
    return log_t


def duration_table(durations, block=BLOCK, fps=config.FPS, max_seconds=MAX_SECONDS):
    """(3, max_blocks) log-probability of a segment lasting d = 1..max_blocks blocks."""
    max_blocks = int(round(max_seconds * fps / block))
    seconds = np.arange(1, max_blocks + 1) * block / fps
    table = np.empty((N_STATES, max_blocks))
    for state, key in ((H, "possession"), (A, "possession"), (D, "dead")):
        mu, sigma = durations[key]
        log_pdf = (-np.log(seconds * sigma * np.sqrt(2 * np.pi))
                   - (np.log(seconds) - mu) ** 2 / (2 * sigma ** 2))
        log_mass = log_pdf + np.log(block / fps)                  # density x bin width
        table[state] = log_mass - np.logaddexp.reduce(log_mass)  # normalise over 1..max
    return table


# --- Viterbi ---------------------------------------------------------------------------------

def viterbi_blocks(block_scores, log_dur, log_trans, log_start=None):
    """Best segmentation of T blocks. Returns [(state, start_block, stop_block)].

    block_scores (T, 3): summed emission scores per block. For a segment of state s that ends
    at block t after lasting d blocks:
        best[t, s] = max_d  cum[t, s] - cum[t-d, s] + log_dur[s, d] + enter[t-d, s]
    where cum is the cumulative score and enter[u, s] is the best way to START state s at
    block u: the start prior when u = 0, otherwise the best segment ending at u plus the
    transition into s. We store start[u, s] = enter[u, s] - cum[u, s] so that the inner max
    over d is one vectorised operation per block.
    """
    n_blocks = len(block_scores)
    max_d = log_dur.shape[1]
    if log_start is None:
        log_start = np.full(N_STATES, -np.log(N_STATES))
    cum = np.vstack([np.zeros((1, N_STATES)), np.cumsum(block_scores, axis=0)])
    columns = np.arange(N_STATES)

    best = np.full((n_blocks + 1, N_STATES), -np.inf)
    start = np.full((n_blocks + 1, N_STATES), -np.inf)
    came_from = np.zeros((n_blocks + 1, N_STATES), dtype=np.int8)
    length = np.zeros((n_blocks + 1, N_STATES), dtype=np.int32)
    start[0] = log_start
    dur_by_d = log_dur.T                                      # row d-1 = duration d

    for t in range(1, n_blocks + 1):
        d_hi = min(t, max_d)
        # row k of `window` <-> the segment started at u = t-1-k, i.e. lasted k+1 blocks
        window = start[t - d_hi:t][::-1] + dur_by_d[:d_hi]
        pick = np.argmax(window, axis=0)
        best[t] = cum[t] + window[pick, columns]
        length[t] = pick + 1
        options = best[t][:, None] + log_trans                # (from state, to state)
        came_from[t] = np.argmax(options, axis=0)
        start[t] = options[came_from[t], columns] - cum[t]

    out = []                                                   # walk back from the end
    t, state = n_blocks, int(np.argmax(best[n_blocks]))
    while t > 0:
        d = int(length[t, state])
        out.append((state, t - d, t))
        t, state = t - d, int(came_from[t - d, state])
    return out[::-1]


def refine_boundaries(segments, frame_scores, block=BLOCK):
    """Move each boundary between two different states to the best frame within +/- block.

    segments: [(state, start_frame, stop_frame)] covering frame_scores (n, 3) contiguously.
    For a boundary at f between s1 | s2, moving it to f' changes the total score by
    S(f) - S(f'), where S(k) = sum over t < k of (score[t, s2] - score[t, s1]) locally.
    So the best f' is where S is smallest; on a tie the boundary stays where it is.
    """
    segs = [list(s) for s in segments]
    for i in range(len(segs) - 1):
        s1, a1, f = segs[i]
        s2, _, b2 = segs[i + 1]
        if s1 == s2:
            continue
        lo, hi = max(a1 + 1, f - block), min(b2 - 1, f + block)
        if hi <= lo:
            continue
        diff = frame_scores[lo:hi, s2] - frame_scores[lo:hi, s1]
        cumulative = np.concatenate(([0.0], np.cumsum(diff)))  # S(k) at index k - lo
        k = int(np.argmin(cumulative))
        if cumulative[k] < cumulative[f - lo]:
            segs[i][2] = segs[i + 1][1] = lo + k
    return [tuple(s) for s in segs]


def decode(frame_scores, half, durations, log_trans, block=BLOCK):
    """Codes (N,) from per-frame state scores (N, 3). X outside graded halves.

    frame_scores[t, s] = evidence for state s at frame t (log scale, any offset per frame).
    Each half must be one contiguous run of frames.
    """
    n = len(frame_scores)
    codes = np.full(n, X, dtype=np.int8)
    log_dur = duration_table(durations, block)
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        first, last = int(rows[0]), int(rows[-1]) + 1
        scores = frame_scores[first:last]
        n_blocks = int(np.ceil(len(scores) / block))
        padded = np.zeros((n_blocks * block, N_STATES))
        padded[:len(scores)] = scores
        block_scores = padded.reshape(n_blocks, block, N_STATES).sum(axis=1)
        segs = viterbi_blocks(block_scores, log_dur, log_trans)
        frame_segs = [(s, a * block, min(b * block, len(scores))) for s, a, b in segs]
        for state, a, b in refine_boundaries(frame_segs, scores, block):
            codes[first + a:first + b] = state
    return codes
