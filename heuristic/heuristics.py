"""All heuristic implementations.

Each heuristic is a callable with signature:
    h(state_info: dict, valid_moves: collection) -> int (position 0-15)

``state_info`` must contain:
    mask_p0  : int
    mask_p1  : int
    last_pos : int
    level    : int
    rel_masks: array of 16 ints

Factory functions (make_oracle_heuristic, make_tree_heuristic) close over
heavier objects (SolutionDB, sklearn model) so the returned callables still
match the simple signature above.
"""
import random
import numpy as np
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                         # heuristic/ (for features)
sys.path.insert(0, os.path.dirname(_HERE))        # project root (for okiya_core)
from features import compute_move_features, IDX

# ---- convenience index aliases ----
_W   = IDX['wins_immediately']
_B   = IDX['blocks_opp_3']
_OV  = IDX['opp_valid_after']
_MB  = IDX['me_best_pattern']
_MP3 = IDX['me_patterns_3']
_OP3 = IDX['opp_patterns_3']
_MP2 = IDX['me_patterns_2']
_OP2 = IDX['opp_patterns_2']


def _feats(state_info, valid_moves):
    """Compute feature dict {pos: feature_vector} for all valid moves."""
    L  = state_info['level']
    m0 = state_info['mask_p0']
    m1 = state_info['mask_p1']
    rm = state_info['rel_masks']
    is_p0    = (L % 2 == 0)
    mask_me  = m0 if is_p0 else m1
    mask_opp = m1 if is_p0 else m0
    return {p: compute_move_features(mask_me, mask_opp, rm, p, L)
            for p in valid_moves}


# ---------------------------------------------------------------------------
# H0 — Random baseline
# ---------------------------------------------------------------------------

def h_random(state_info, valid_moves):
    """Uniform random move."""
    return random.choice(list(valid_moves))


# ---------------------------------------------------------------------------
# H1 — Win / Block / Random
# Simple 2-rule heuristic: the minimum viable decision tree.
# Rule: "Win if you can; block if you must; otherwise pick randomly."
# ---------------------------------------------------------------------------

def h_win_block(state_info, valid_moves):
    f = _feats(state_info, valid_moves)
    moves = list(valid_moves)

    wins   = [p for p in moves if f[p][_W] == 1]
    if wins:
        return random.choice(wins)

    blocks = [p for p in moves if f[p][_B] == 1]
    if blocks:
        return random.choice(blocks)

    return random.choice(moves)


# ---------------------------------------------------------------------------
# H2 — Win / Block / Constrain
# Win > block > minimise opponent's moves after our play.
# One additional rule on top of H1: prefer tiles that leave the opponent
# fewer options (sticky constraint / stuck-threat play).
# ---------------------------------------------------------------------------

def h_constraint(state_info, valid_moves):
    f = _feats(state_info, valid_moves)
    moves = list(valid_moves)

    wins = [p for p in moves if f[p][_W] == 1]
    if wins:
        return random.choice(wins)

    blocks = [p for p in moves if f[p][_B] == 1]
    pool = blocks if blocks else moves

    # Minimise opp_valid_after; break ties randomly
    min_val = min(f[p][_OV] for p in pool)
    best    = [p for p in pool if f[p][_OV] == min_val]
    return random.choice(best)


# ---------------------------------------------------------------------------
# H3 — Win / Block / Pattern + Constrain
# Win > block > maximise my best pattern > minimise opp valid moves.
# Combines threat-building with constraint.
# ---------------------------------------------------------------------------

def h_pattern_constraint(state_info, valid_moves):
    f = _feats(state_info, valid_moves)
    moves = list(valid_moves)

    wins = [p for p in moves if f[p][_W] == 1]
    if wins:
        return random.choice(wins)

    blocks = [p for p in moves if f[p][_B] == 1]
    pool   = blocks if blocks else moves

    best = max(pool, key=lambda p: (f[p][_MB], -f[p][_OV]))
    return best


# ---------------------------------------------------------------------------
# H4 — Full Priority (most complex hand-crafted rule)
# Win > block > me_patterns_3 > min opp_valid > me_patterns_2 > min opp_pat2
# ---------------------------------------------------------------------------

def h_full_priority(state_info, valid_moves):
    f = _feats(state_info, valid_moves)
    moves = list(valid_moves)

    wins = [p for p in moves if f[p][_W] == 1]
    if wins:
        return random.choice(wins)

    blocks = [p for p in moves if f[p][_B] == 1]
    pool   = blocks if blocks else moves

    best = max(pool, key=lambda p: (
        f[p][_MP3],          # maximise my 3-in-a-row threats
        -f[p][_OV],          # minimise opponent's options
        f[p][_MP2],          # maximise my 2-in-a-row progress
        -f[p][_OP2],         # reduce opponent's 2-in-a-row progress
    ))
    return best


# ---------------------------------------------------------------------------
# Factory: Oracle (minimax lookup)
# ---------------------------------------------------------------------------

def make_oracle_heuristic(db):
    """Return a heuristic that always picks the minimax-optimal move.

    Pre-loads all solution arrays eagerly into plain numpy arrays
    (NpzFile lazy decompression is catastrophically slow per-call).
    """
    print("[Oracle] Pre-loading solution arrays...")
    _levels = {}
    for L in range(17):
        d = db.load_level(L)
        if d is not None:
            _levels[L] = {
                'states':  np.array(d['states']),
                'values':  np.array(d['values']),
                'offsets': np.array(d['children_offsets']) if 'children_offsets' in d else None,
                'indices': np.array(d['children_indices']) if 'children_indices' in d else None,
            }
    print("[Oracle] Ready")

    def h_oracle(state_info, valid_moves):
        m0    = state_info['mask_p0']
        m1    = state_info['mask_p1']
        lp    = state_info['last_pos']
        level = state_info['level']

        data = _levels.get(level)
        if data is None:
            return random.choice(list(valid_moves))

        state = np.uint64((int(m0) << 21) | (int(m1) << 5) | int(lp))
        idx   = int(np.searchsorted(data['states'], state))
        if idx >= len(data['states']) or data['states'][idx] != state:
            return random.choice(list(valid_moves))

        offsets = data['offsets']
        indices = data['indices']
        if offsets is None or indices is None:
            return random.choice(list(valid_moves))
        start, end = int(offsets[idx]), int(offsets[idx + 1])
        if start >= end:
            return random.choice(list(valid_moves))

        nxt = _levels.get(level + 1)
        if nxt is None:
            return random.choice(list(valid_moves))

        c_idx = indices[start:end]
        c_vals  = nxt['values'][c_idx]
        c_moves = (nxt['states'][c_idx] & 0x1F).astype(int)

        best_val   = int(c_vals.max()) if (level % 2 == 0) else int(c_vals.min())
        best_moves = c_moves[c_vals == best_val].tolist()
        return random.choice(best_moves) if best_moves else random.choice(list(valid_moves))

    h_oracle.__name__ = 'Oracle'
    return h_oracle


# ---------------------------------------------------------------------------
# Factory: Decision-tree heuristic
# ---------------------------------------------------------------------------

def make_tree_heuristic(clf, rel_masks, name='Tree'):
    """Return a heuristic backed by a scikit-learn DecisionTreeClassifier.

    The classifier must have been trained on features produced by
    ``compute_move_features``.  At inference time it scores each candidate
    move independently and picks the one with the highest P(optimal) estimate.
    """
    def h_tree(state_info, valid_moves):
        L  = state_info['level']
        m0 = state_info['mask_p0']
        m1 = state_info['mask_p1']
        moves   = list(valid_moves)
        is_p0   = (L % 2 == 0)
        mask_me  = m0 if is_p0 else m1
        mask_opp = m1 if is_p0 else m0

        X = np.array([
            compute_move_features(mask_me, mask_opp, rel_masks, p, L)
            for p in moves
        ])
        probs    = clf.predict_proba(X)[:, 1]
        best_idx = int(np.argmax(probs))
        return moves[best_idx]

    h_tree.__name__ = name
    return h_tree


# ---------------------------------------------------------------------------
# Registry: all hand-crafted heuristics in evaluation order
# ---------------------------------------------------------------------------

HEURISTIC_REGISTRY = [
    ('H0_Random',              h_random),
    ('H1_WinBlock',            h_win_block),
    ('H2_Constraint',          h_constraint),
    ('H3_PatternConstraint',   h_pattern_constraint),
    ('H4_FullPriority',        h_full_priority),
]
