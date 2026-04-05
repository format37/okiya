"""Evaluation framework: move accuracy and win-rate comparison.

Two evaluation modes
--------------------
1. Move accuracy
   For every sampled non-terminal state, ask the heuristic to pick a move
   and check whether it matches the minimax-optimal move.  Reported per level L.

2. Win rate (game simulation)
   Simulate full games: each player follows its heuristic; terminal conditions
   (win patterns, stuck) are checked after every move.  No solution DB needed
   except for the Oracle heuristic.
"""
import numpy as np
import random
import os
import sys
import csv

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from okiya_core import SolutionDB, WIN_PATTERNS, BORDER_MASK


_WP = [int(wp) for wp in WIN_PATTERNS]
_BORDER = int(BORDER_MASK)


# ---------------------------------------------------------------------------
# Low-level game helpers (no SolutionDB needed)
# ---------------------------------------------------------------------------

def _check_win_fast(mask: int) -> bool:
    for wp in _WP:
        if (mask & wp) == wp:
            return True
    return False


def _get_valid_moves_fast(m0: int, m1: int, last_pos: int, rel_masks) -> list:
    occupied = m0 | m1
    if last_pos == 16:          # initial state, P0's opening
        avail = _BORDER & ~occupied & 0xFFFF
    else:
        avail = int(rel_masks[last_pos]) & ~occupied & 0xFFFF
    return [p for p in range(16) if avail & (1 << p)]


# ---------------------------------------------------------------------------
# Game simulation
# ---------------------------------------------------------------------------

def play_game(h_p0, h_p1, rel_masks):
    """Simulate one game from scratch.  Returns +1 if P0 wins, -1 if P1 wins."""
    m0, m1, lp = 0, 0, 16

    for _ in range(17):  # max 16 moves, +1 for safety
        level = bin(m0).count('1') + bin(m1).count('1')
        is_p0 = (level % 2 == 0)
        valid = _get_valid_moves_fast(m0, m1, lp, rel_masks)

        if not valid:
            return -1 if is_p0 else 1

        state_info = dict(mask_p0=m0, mask_p1=m1, last_pos=lp,
                          level=level, rel_masks=rel_masks)
        move = (h_p0 if is_p0 else h_p1)(state_info, valid)

        if is_p0:
            m0 |= (1 << move)
            if _check_win_fast(m0):
                return 1
        else:
            m1 |= (1 << move)
            if _check_win_fast(m1):
                return -1
        lp = move

    return 0  # shouldn't happen


def simulate_games(h_p0, h_p1, rel_masks, n_games: int = 1000, seed: int = 42):
    """Run n_games and return P0's win fraction."""
    random.seed(seed)
    p0_wins = sum(1 for _ in range(n_games)
                  if play_game(h_p0, h_p1, rel_masks) == 1)
    return p0_wins / n_games


# ---------------------------------------------------------------------------
# Move accuracy (on pre-loaded records)
# ---------------------------------------------------------------------------

def _accuracy_on_records(heuristic_fn, records, rel_masks):
    """Return per-level accuracy dict from already-loaded records."""
    correct_by_L = {}
    total_by_L   = {}
    for rec in records:
        L = rec['level']
        state_info = dict(mask_p0=rec['mask_p0'], mask_p1=rec['mask_p1'],
                          last_pos=rec['last_pos'], level=L,
                          rel_masks=rel_masks)
        chosen  = heuristic_fn(state_info, rec['valid_moves'])
        optimal = set(rec['optimal_moves'])
        correct_by_L[L] = correct_by_L.get(L, 0) + (1 if chosen in optimal else 0)
        total_by_L[L]   = total_by_L.get(L, 0) + 1
    return {L: correct_by_L[L] / total_by_L[L]
            for L in sorted(total_by_L) if total_by_L[L] > 0}


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(db: SolutionDB,
                        named_heuristics: list,
                        n_games: int = 1000,
                        max_per_level: int = 10_000,
                        output_dir: str = None,
                        seed: int = 42):
    """Compute move-accuracy table and pairwise win-rate matrix.

    Parameters
    ----------
    db               : SolutionDB
    named_heuristics : list of (name: str, heuristic_fn) tuples
    n_games          : games per pair in the win-rate simulation
    max_per_level    : states sampled per level for accuracy
    output_dir       : directory for CSV output
    seed             : RNG seed
    """
    from dataset import load_dataset

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    rel_masks = db.pos_relation_masks
    names = [n for n, _ in named_heuristics]

    # ---- Load dataset ONCE ----
    print("[Eval] Loading evaluation dataset (once for all heuristics)...")
    records = load_dataset(db, max_per_level=max_per_level, seed=seed,
                           informative_only=True)
    print(f"[Eval] {len(records):,} informative states loaded")

    # ---- Move accuracy ----
    print("\n=== Move Accuracy ===")
    accuracy_table = {}
    for name, fn in named_heuristics:
        random.seed(seed)
        acc = _accuracy_on_records(fn, records, rel_masks)
        accuracy_table[name] = acc
        vals = list(acc.values())
        mean = sum(vals) / len(vals) if vals else 0
        print(f"  {name:<26s}  mean={mean:.4f}")

    # Print accuracy table
    levels = sorted({L for acc in accuracy_table.values() for L in acc})
    print(f"\n{'Heuristic':<26}", end='')
    for L in levels:
        print(f"  L{L:02d}", end='')
    print("   mean")
    print("-" * (26 + 6 * len(levels) + 7))
    for name in names:
        acc = accuracy_table[name]
        vals = []
        print(f"{name:<26}", end='')
        for L in levels:
            v = acc.get(L, float('nan'))
            vals.append(v)
            print(f" {v:5.3f}" if not np.isnan(v) else "   — ", end='')
        valid = [v for v in vals if not np.isnan(v)]
        mean = sum(valid) / len(valid) if valid else float('nan')
        print(f"  {mean:.3f}")

    # Save accuracy CSV
    acc_path = os.path.join(output_dir, 'move_accuracy.csv')
    with open(acc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['heuristic'] + [f'L{L:02d}' for L in levels] + ['mean'])
        for name in names:
            acc  = accuracy_table[name]
            vals = [acc.get(L, '') for L in levels]
            valid = [v for v in vals if v != '']
            mean = sum(valid) / len(valid) if valid else ''
            writer.writerow([name] + vals + [mean])
    print(f"[Eval] Saved: {acc_path}")

    # ---- Pairwise win rates ----
    print("\n=== Pairwise Win Rates (P0 win fraction) ===")
    n = len(names)
    win_matrix = np.full((n, n), np.nan)

    total_pairs = n * (n - 1)
    done = 0
    for i, (ni, fi) in enumerate(named_heuristics):
        for j, (nj, fj) in enumerate(named_heuristics):
            if i == j:
                win_matrix[i, j] = 0.5
                continue
            done += 1
            print(f"  [{done}/{total_pairs}] {ni} vs {nj}...", end='', flush=True)
            wr = simulate_games(fi, fj, rel_masks, n_games=n_games, seed=seed)
            win_matrix[i, j] = wr
            print(f" {wr:.3f}")

    # Print win matrix
    col_w = max(len(n) for n in names)
    label = 'P0 vs P1'
    header = f"{label:<{col_w}}"
    for name in names:
        header += f"  {name[:10]:>10}"
    print(header)
    print("-" * len(header))
    for i, ni in enumerate(names):
        row = f"{ni:<{col_w}}"
        for j in range(n):
            v = win_matrix[i, j]
            row += f"  {v:10.3f}" if not np.isnan(v) else "         — "
        print(row)

    # Save win-rate CSV
    wr_path = os.path.join(output_dir, 'win_rates.csv')
    with open(wr_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['p0_heuristic', 'p1_heuristic', 'p0_win_rate'])
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                writer.writerow([ni, nj, win_matrix[i, j]])
    print(f"[Eval] Saved: {wr_path}")

    return accuracy_table, win_matrix
