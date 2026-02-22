#!/usr/bin/env python3
"""Shared game logic for Okiya: SolutionDB, GameState, pure helpers.

Imported by okiya_query.py (CLI) and okiya_gui.py (Pygame GUI).
"""

import numpy as np
import json
import os

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(SCRIPT_DIR, 'solution')
BOARD_PATH   = os.path.join(SCRIPT_DIR, 'board.json')

# ---------------------------------------------------------------------------
# Board-independent constants
# ---------------------------------------------------------------------------

WIN_PATTERNS = np.array([
    61440, 3840, 240, 15,           # rows
    34952, 17476, 8738, 4369,       # columns
    33825, 4680,                    # diagonals (main, anti)
    52224, 26112, 13056,            # 2x2 quads (top)
    3264, 1632, 816,               # 2x2 quads (mid)
    204, 102, 51                   # 2x2 quads (bot)
], dtype=np.uint32)

BORDER_MASK = np.uint32(63903)  # all except positions 5,6,9,10

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def popcount(x):
    return bin(int(x)).count('1')


def encode_state(mask_p0, mask_p1, last_pos):
    return np.uint64((int(mask_p0) << 21) | (int(mask_p1) << 5) | int(last_pos))


def decode_state(state):
    s = int(state)
    return (s >> 21) & 0xFFFF, (s >> 5) & 0xFFFF, s & 0x1F


def value_label(v):
    if v == 1:  return 'P0 wins'
    if v == -1: return 'P1 wins'
    return 'Draw'


def compute_pos_relation_masks(init_tiles):
    """For each board position, bitmask of positions with related tile types."""
    masks = np.zeros(16, dtype=np.uint32)
    for p in range(16):
        t = init_tiles[p]
        a, b = t // 4, t % 4
        for q in range(16):
            if q == p:
                continue
            tq = init_tiles[q]
            if tq // 4 == a or tq % 4 == b:
                masks[p] |= np.uint32(1 << q)
    return masks


# ---------------------------------------------------------------------------
# SolutionDB — wraps solution access with reload support
# ---------------------------------------------------------------------------

class SolutionDB:
    def __init__(self, solution_dir=SOLUTION_DIR, board_path=BOARD_PATH):
        self._solution_dir = solution_dir
        self._board_path = board_path
        self._cache = {}
        self._meta = None
        self._win_rates = None  # lazy: level -> float64 array
        self._load_board_config()
        self._load_metadata()

    def _load_board_config(self):
        """Load tile layout from board.json (always available)."""
        with open(self._board_path) as f:
            board = json.load(f)
        self._init_tiles = np.array(board['tiles'], dtype=np.int32)
        self._pos_relation_masks = compute_pos_relation_masks(self._init_tiles)
        attrs = board.get('attributes', {})
        self._tile_a_names = attrs.get('a', ['Maple', 'Cherry', 'Pine', 'Iris'])
        self._tile_b_names = attrs.get('b', ['Sun', 'Tanzaku', 'Bird', 'Rain'])

    def _load_metadata(self):
        """Try to load solution metadata. Sets _meta=None if absent."""
        path = os.path.join(self._solution_dir, 'metadata.json')
        if not os.path.exists(path):
            self._meta = None
            return
        with open(path) as f:
            self._meta = json.load(f)

    @property
    def has_solution(self):
        return self._meta is not None

    def reload(self):
        """Re-read solution + board config and clear caches (call after solve)."""
        self._cache.clear()
        self._win_rates = None
        self._load_board_config()
        self._load_metadata()

    def load_board(self):
        """Re-read board.json and delete stale solution (call after shuffle)."""
        self._load_board_config()
        self.clear_solution()

    def clear_solution(self):
        """Delete solution files and clear all caches."""
        import glob
        for f in glob.glob(os.path.join(self._solution_dir, 'level_*.npz')):
            os.remove(f)
        meta_path = os.path.join(self._solution_dir, 'metadata.json')
        if os.path.exists(meta_path):
            os.remove(meta_path)
        self._cache.clear()
        self._win_rates = None
        self._meta = None

    @property
    def meta(self):
        return self._meta

    @property
    def init_tiles(self):
        return self._init_tiles

    @property
    def pos_relation_masks(self):
        return self._pos_relation_masks

    @property
    def tile_a_names(self):
        return self._tile_a_names

    @property
    def tile_b_names(self):
        return self._tile_b_names

    def tile_name(self, tile_type):
        a, b = tile_type // 4, tile_type % 4
        return f"{self._tile_a_names[a]}+{self._tile_b_names[b]}"

    def load_level(self, level):
        if level not in self._cache:
            path = os.path.join(self._solution_dir, f'level_{level:02d}.npz')
            if not os.path.exists(path):
                return None
            self._cache[level] = np.load(path)
        return self._cache[level]

    def lookup(self, mask_p0, mask_p1, last_pos, top_n=None):
        """Look up a state and return (value, [(pos, child_value), ...])."""
        if not self.has_solution:
            return None, []
        level = popcount(mask_p0) + popcount(mask_p1)
        state = encode_state(mask_p0, mask_p1, last_pos)

        data = self.load_level(level)
        if data is None:
            return None, []

        states = data['states']
        idx = int(np.searchsorted(states, state))
        if idx >= len(states) or states[idx] != state:
            return None, []

        value = int(data['values'][idx])

        moves = []
        if 'children_offsets' in data and 'children_indices' in data:
            offsets = data['children_offsets']
            indices = data['children_indices']
            start, end = int(offsets[idx]), int(offsets[idx + 1])

            if start < end:
                nxt = self.load_level(level + 1)
                if nxt is not None:
                    child_idxs = indices[start:end]
                    for ci in child_idxs:
                        cs = nxt['states'][ci]
                        cv = int(nxt['values'][ci])
                        _, _, clp = decode_state(cs)
                        moves.append((clp, cv))

        # sort: best moves first
        is_p0_turn = (level % 2 == 0)
        moves.sort(key=lambda m: m[1], reverse=is_p0_turn)

        if top_n is not None and top_n > 0:
            moves = moves[:top_n]

        return value, moves

    def get_valid_moves(self, mask_p0, mask_p1, last_pos):
        """Return set of valid board positions for the current state."""
        occupied = mask_p0 | mask_p1
        level = popcount(mask_p0) + popcount(mask_p1)

        if level == 0:
            constraint = int(BORDER_MASK)
        else:
            constraint = int(self._pos_relation_masks[last_pos])

        available = constraint & ~occupied
        return {p for p in range(16) if available & (1 << p)}

    def check_win(self, mask):
        """Return True if the given player mask contains a winning pattern."""
        m = int(mask)
        for wp in WIN_PATTERNS:
            if (m & int(wp)) == int(wp):
                return True
        return False

    # -------------------------------------------------------------------
    # Win-rate computation (uniform random play)
    # -------------------------------------------------------------------

    def _ensure_win_rates(self):
        """Lazily compute P0 win probability for every state, bottom-up."""
        if self._win_rates is not None:
            return
        if not self.has_solution:
            return
        # find max level
        max_level = 0
        for L in range(17):
            if self.load_level(L) is not None:
                max_level = L
            else:
                break

        rates_by_level = {}
        for L in range(max_level, -1, -1):
            data = self.load_level(L)
            n = len(data['states'])
            rates = np.full(n, 0.5, dtype=np.float64)

            terminal = data['terminal'].astype(bool)
            values = data['values']
            rates[terminal & (values == 1)] = 1.0
            rates[terminal & (values == -1)] = 0.0

            # non-terminal: mean of children's rates
            if (L < max_level
                    and 'children_offsets' in data
                    and 'children_indices' in data):
                nxt_rates = rates_by_level.get(L + 1)
                if nxt_rates is not None:
                    offsets = data['children_offsets']
                    indices = data['children_indices']
                    edge_counts = np.diff(offsets)
                    has_edges = (edge_counts > 0) & ~terminal
                    if has_edges.any() and len(indices) > 0:
                        child_rates = nxt_rates[indices]
                        parent_ids = np.repeat(
                            np.arange(n, dtype=np.int64), edge_counts)
                        rate_sums = np.zeros(n, dtype=np.float64)
                        np.add.at(rate_sums, parent_ids, child_rates)
                        rates[has_edges] = (
                            rate_sums[has_edges]
                            / edge_counts[has_edges].astype(np.float64))

            rates_by_level[L] = rates

        self._win_rates = rates_by_level

    def get_move_win_rates(self, mask_p0, mask_p1, last_pos):
        """Return {move_pos: win_rate_for_current_mover} for each move.

        Win rate = probability of winning under uniform random play.
        """
        if not self.has_solution:
            return {}
        self._ensure_win_rates()
        if self._win_rates is None:
            return {}
        level = popcount(mask_p0) + popcount(mask_p1)
        is_p0_turn = (level % 2 == 0)
        state = encode_state(mask_p0, mask_p1, last_pos)

        data = self.load_level(level)
        if data is None:
            return {}
        states = data['states']
        idx = int(np.searchsorted(states, state))
        if idx >= len(states) or states[idx] != state:
            return {}
        if 'children_offsets' not in data or 'children_indices' not in data:
            return {}

        offsets = data['children_offsets']
        indices = data['children_indices']
        start, end = int(offsets[idx]), int(offsets[idx + 1])
        if start >= end:
            return {}

        nxt = self.load_level(level + 1)
        nxt_rates = self._win_rates.get(level + 1)
        if nxt is None or nxt_rates is None:
            return {}

        result = {}
        for ci in indices[start:end]:
            cs = nxt['states'][ci]
            _, _, clp = decode_state(cs)
            p0_rate = float(nxt_rates[ci])
            result[clp] = p0_rate if is_p0_turn else (1.0 - p0_rate)
        return result


# ---------------------------------------------------------------------------
# GameState — game history with undo/redo
# ---------------------------------------------------------------------------

class GameState:
    def __init__(self):
        self.history = [(0, 0, 16)]  # (m0, m1, lp)
        self.cursor = 0

    @property
    def mask_p0(self):
        return self.history[self.cursor][0]

    @property
    def mask_p1(self):
        return self.history[self.cursor][1]

    @property
    def last_pos(self):
        return self.history[self.cursor][2]

    @property
    def level(self):
        return popcount(self.mask_p0) + popcount(self.mask_p1)

    @property
    def is_p0_turn(self):
        return self.level % 2 == 0

    def play_move(self, pos):
        m0, m1, _ = self.history[self.cursor]
        if self.is_p0_turn:
            m0 |= (1 << pos)
        else:
            m1 |= (1 << pos)
        # truncate redo future
        self.history = self.history[:self.cursor + 1]
        self.history.append((m0, m1, pos))
        self.cursor += 1

    def can_undo(self):
        return self.cursor > 0

    def can_redo(self):
        return self.cursor < len(self.history) - 1

    def undo(self):
        if self.can_undo():
            self.cursor -= 1

    def redo(self):
        if self.can_redo():
            self.cursor += 1

    def reset(self):
        self.history = [(0, 0, 16)]
        self.cursor = 0
