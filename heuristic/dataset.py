"""Load labeled (state, optimal_moves) records from solution/ NPZ files.

Each record describes a non-terminal game state together with the set of
moves that maintain the minimax value (optimal) and those that don't
(suboptimal).  Only states with at least one suboptimal move are useful for
learning — pure-losing or all-equally-optimal states are filtered out when
``informative_only=True`` (default).
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from okiya_core import SolutionDB, decode_state


def load_dataset(db: SolutionDB,
                 max_per_level: int = 20_000,
                 levels=None,
                 informative_only: bool = True,
                 seed: int = 42):
    """Return a list of state-record dicts for non-terminal game states.

    Parameters
    ----------
    db              : SolutionDB instance (canonical board already loaded)
    max_per_level   : max states to sample per level (None = all)
    levels          : iterable of levels to include (default 0–14)
    informative_only: if True, skip states where all moves have equal value
                      (no suboptimal moves exist, useless for learning)
    seed            : RNG seed for level-level sampling

    Each returned dict has keys:
        state       : int  – encoded uint64 game state
        level       : int  – game level (0–16)
        value       : int  – minimax value (+1 / -1)
        mask_p0     : int
        mask_p1     : int
        last_pos    : int
        valid_moves : list[int]  – all candidate move positions
        optimal_moves: list[int] – subset that maintain the minimax value
    """
    if levels is None:
        levels = range(0, 15)   # L=15/16 are near-terminal, less useful

    rng = np.random.default_rng(seed)
    records = []

    for L in levels:
        data = db.load_level(L)
        if data is None:
            continue
        nxt = db.load_level(L + 1)
        if nxt is None:
            continue

        states       = data['states']
        values       = data['values']
        terminal     = data['terminal'].astype(bool)
        offsets      = data['children_offsets']
        child_idx    = data['children_indices']
        nxt_states   = nxt['states']
        nxt_values   = nxt['values']

        non_term = np.where(~terminal)[0]
        if max_per_level is not None and len(non_term) > max_per_level:
            non_term = rng.choice(non_term, max_per_level, replace=False)
            non_term.sort()

        for idx in non_term:
            start = int(offsets[idx])
            end   = int(offsets[idx + 1])
            if start >= end:
                continue

            c_idxs  = child_idx[start:end]
            c_states = nxt_states[c_idxs]
            c_vals   = nxt_values[c_idxs].astype(int)

            # Move position = last_pos encoded in child state (bits 4-0)
            c_moves = [int(cs) & 0x1F for cs in c_states]

            parent_val = int(values[idx])

            if informative_only and len(set(c_vals)) == 1:
                # All children have the same value — nothing to learn here
                continue

            optimal = [pos for pos, cv in zip(c_moves, c_vals) if cv == parent_val]
            if not optimal:
                optimal = c_moves  # fallback (shouldn't normally occur)

            m0, m1, lp = decode_state(states[idx])
            records.append({
                'state':        int(states[idx]),
                'level':        L,
                'value':        parent_val,
                'mask_p0':      m0,
                'mask_p1':      m1,
                'last_pos':     lp,
                'valid_moves':  c_moves,
                'optimal_moves': optimal,
            })

    return records


def build_training_data(records, rel_masks):
    """Convert a list of state records into (X, y, levels) arrays for ML.

    For each (state, move) pair:
        X[i]      = feature vector (N_FEATURES,)
        y[i]      = 1 if move is optimal, 0 otherwise
        levels[i] = game level of the parent state

    Only samples from states where there is at least one suboptimal move are
    included (states where every move is optimal contribute no gradient).

    Returns
    -------
    X      : np.ndarray  shape (N, N_FEATURES)
    y      : np.ndarray  shape (N,) int8
    levels : np.ndarray  shape (N,) int8
    """
    from features import compute_move_features, N_FEATURES

    X_rows = []
    y_rows = []
    L_rows = []

    for rec in records:
        L       = rec['level']
        m0, m1  = rec['mask_p0'], rec['mask_p1']
        is_p0   = (L % 2 == 0)
        mask_me  = m0 if is_p0 else m1
        mask_opp = m1 if is_p0 else m0
        optimal  = set(rec['optimal_moves'])

        for p in rec['valid_moves']:
            feat = compute_move_features(mask_me, mask_opp, rel_masks, p, L)
            X_rows.append(feat)
            y_rows.append(1 if p in optimal else 0)
            L_rows.append(L)

    X      = np.array(X_rows, dtype=np.float32)
    y      = np.array(y_rows, dtype=np.int8)
    levels = np.array(L_rows, dtype=np.int8)
    return X, y, levels
