"""Feature extraction for (state, candidate_move) pairs.

Each feature vector characterises the quality of playing move p in a given
board state from the current player's point of view.  Used by both the
hand-crafted heuristics (for move scoring) and the decision-tree learner.
"""
import numpy as np
import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from okiya_core import WIN_PATTERNS

# Pre-materialise win-pattern values as plain Python ints for tight inner loops
_WP = [int(wp) for wp in WIN_PATTERNS]

# How many win patterns contain each board position (positional value proxy)
_N_PATTERNS_AT = [sum(1 for wp in _WP if wp & (1 << p)) for p in range(16)]

FEATURE_NAMES = [
    'wins_immediately',  # 0  claiming p completes a win pattern right now
    'blocks_opp_3',      # 1  p is the sole missing tile in a pattern where opp has 3
    'opp_valid_after',   # 2  opponent's valid-move count after I play p (lower = tighter constraint)
    'me_best_pattern',   # 3  max tiles I'd have in any single pattern after claiming p
    'opp_best_pattern',  # 4  max tiles opp has in any single pattern (state context, same for all moves)
    'me_patterns_3',     # 5  count of patterns where I'd have exactly 3 after p
    'opp_patterns_3',    # 6  count of patterns where opp has exactly 3 (state context)
    'me_patterns_2',     # 7  count of patterns where I'd have exactly 2 after p
    'opp_patterns_2',    # 8  count of patterns where opp has exactly 2 (state context)
    'n_patterns_p',      # 9  how many win patterns contain position p
    'level',             # 10 current game level 0-16
]

N_FEATURES = len(FEATURE_NAMES)

# Index shortcuts for heuristics.py
IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def compute_move_features(mask_me, mask_opp, rel_masks, p, level):
    """Compute the feature vector for playing move *p* from the current player.

    Parameters
    ----------
    mask_me   : int  – current player's claimed-position bitmask (16 bits)
    mask_opp  : int  – opponent's claimed-position bitmask
    rel_masks : array-like of 16 ints – precomputed relation masks per position
    p         : int  – candidate move position (0–15)
    level     : int  – current game level (0–16)

    Returns
    -------
    np.ndarray of shape (N_FEATURES,), dtype float32
    """
    bit_p          = 1 << p
    mask_me_after  = mask_me | bit_p
    occupied       = mask_me | mask_opp          # before playing p
    occupied_after = mask_me_after | mask_opp

    wins    = 0
    blocks3 = 0
    me_best = 0
    me_p3   = 0
    me_p2   = 0
    opp_best = 0
    opp_p3   = 0
    opp_p2   = 0

    for wp in _WP:
        me_in  = bin(mask_me_after & wp).count('1')
        opp_in = bin(mask_opp & wp).count('1')

        # ---- my pattern progress after claiming p ----
        if me_in == 4:
            wins = 1
        if me_in > me_best:
            me_best = me_in
        if me_in == 3:
            me_p3 += 1
        elif me_in == 2:
            me_p2 += 1

        # ---- opponent pattern status (independent of p) ----
        if opp_in > opp_best:
            opp_best = opp_in
        if opp_in == 3:
            opp_p3 += 1
            # blocks_opp_3: p is the only unclaimed tile left in this pattern
            remaining = wp & ~occupied & 0xFFFF
            if remaining == bit_p:
                blocks3 = 1
        elif opp_in == 2:
            opp_p2 += 1

    # Opponent's valid-move count after I play p
    # (they must play a tile related to p that is still unclaimed)
    opp_valid = bin(int(rel_masks[p]) & ~occupied_after & 0xFFFF).count('1')

    return np.array([
        wins,    blocks3,   opp_valid,
        me_best, opp_best,
        me_p3,   opp_p3,
        me_p2,   opp_p2,
        _N_PATTERNS_AT[p],
        level,
    ], dtype=np.float32)
