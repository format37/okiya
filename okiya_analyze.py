#!/usr/bin/env python3
"""Single-board analysis of the solved Okiya game.

Loads existing solution/ via SolutionDB. Computes Tier 1 metrics (no GPU):
  - f_minimax(L): fraction of states at level L with value == +1
  - f_random(L):  mean P0 win probability under uniform random play
  - First-difference delta_f(L) for sawtooth effect
  - Critical move ratio per level
  - Terminal breakdown (pattern wins vs stuck) per level

Usage:
    python3 okiya_analyze.py [--plot] [--output analysis.json]
"""

import numpy as np
import json
import os
import argparse

from okiya_core import SolutionDB

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_metrics(db):
    """Compute all Tier 1 metrics from the solved game tree."""
    meta = db.meta
    max_level = max(int(k) for k in meta['level_counts'].keys())

    levels = []
    n_states = []
    f_minimax = []
    f_random = []
    n_terminal = []
    n_pattern_wins = []
    n_stuck = []
    critical_move_ratio = []

    # ensure win rates are computed
    db._ensure_win_rates()

    for L in range(max_level + 1):
        data = db.load_level(L)
        if data is None:
            break

        levels.append(L)
        n = len(data['states'])
        n_states.append(n)

        values = data['values']
        terminal = data['terminal'].astype(bool)

        # f_minimax: fraction with value == +1
        f_m = float((values == 1).sum()) / n if n > 0 else 0.0
        f_minimax.append(f_m)

        # f_random: mean P0 win probability under uniform random play
        if db._win_rates is not None and L in db._win_rates:
            f_r = float(db._win_rates[L].mean())
        else:
            f_r = 0.5
        f_random.append(f_r)

        # terminal breakdown
        n_term = int(terminal.sum())
        n_terminal.append(n_term)

        # pattern wins: terminal states where value != 0
        # (value==+1 means P0 won by pattern, value==-1 means P1 won by pattern)
        # stuck states: terminal but value comes from no-moves rule
        # We distinguish by checking if the mover mask matches a win pattern
        if n_term > 0:
            term_idx = np.where(terminal)[0]
            term_states = data['states'][term_idx]
            term_values = values[term_idx]

            m0 = ((term_states >> 21) & 0xFFFF).astype(np.uint32)
            m1 = ((term_states >>  5) & 0xFFFF).astype(np.uint32)

            # who moved last to reach this level?
            if L > 0:
                prev_mover = (L - 1) % 2  # 0=P0, 1=P1
                mover_mask = m0 if prev_mover == 0 else m1

                from okiya_core import WIN_PATTERNS
                is_pattern_win = np.zeros(len(term_idx), dtype=bool)
                for wp in WIN_PATTERNS:
                    is_pattern_win |= (mover_mask & int(wp)) == int(wp)

                n_pw = int(is_pattern_win.sum())
                n_st = n_term - n_pw
            else:
                # level 0 has no terminal states normally
                n_pw = 0
                n_st = 0
        else:
            n_pw = 0
            n_st = 0

        n_pattern_wins.append(n_pw)
        n_stuck.append(n_st)

        # critical move ratio: fraction of parent→child edges where
        # child value != parent value (i.e., the move was a "mistake")
        if ('children_offsets' in data and 'children_indices' in data
                and L + 1 <= max_level):
            nxt = db.load_level(L + 1)
            if nxt is not None:
                offsets = data['children_offsets']
                indices = data['children_indices']
                non_term = ~terminal
                edge_counts = np.diff(offsets)
                has_children = (edge_counts > 0) & non_term

                if has_children.any() and len(indices) > 0:
                    nxt_values = nxt['values']
                    # for each edge, compare parent value to child value
                    parent_ids = np.repeat(
                        np.arange(n, dtype=np.int64), edge_counts)
                    parent_vals = values[parent_ids]
                    child_vals = nxt_values[indices]
                    mistakes = parent_vals != child_vals
                    total_edges = len(parent_vals)
                    cmr = float(mistakes.sum()) / total_edges if total_edges > 0 else 0.0
                else:
                    cmr = 0.0
            else:
                cmr = 0.0
        else:
            cmr = 0.0
        critical_move_ratio.append(cmr)

    # first-difference
    delta_f_minimax = [0.0] + [f_minimax[i] - f_minimax[i-1]
                                for i in range(1, len(f_minimax))]
    delta_f_random = [0.0] + [f_random[i] - f_random[i-1]
                               for i in range(1, len(f_random))]

    return dict(
        board_name=meta.get('board_name', ''),
        init_tiles=meta.get('init_tiles', []),
        total_states=meta.get('total_states', 0),
        game_value=int(db.load_level(0)['values'][0]),
        levels=levels,
        n_states=n_states,
        f_minimax=f_minimax,
        f_random=f_random,
        delta_f_minimax=delta_f_minimax,
        delta_f_random=delta_f_random,
        n_terminal=n_terminal,
        n_pattern_wins=n_pattern_wins,
        n_stuck=n_stuck,
        critical_move_ratio=critical_move_ratio,
    )


def plot_analysis(metrics, output_dir=None):
    """Generate matplotlib figures from single-board analysis."""
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, 'results', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    L = metrics['levels']

    # Plot 1: Win probability gradient (f_minimax and f_random)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(L, metrics['f_minimax'], 'o-', color='#d62728', label='f_minimax (optimal)')
    ax.plot(L, metrics['f_random'], 's--', color='#1f77b4', label='f_random (uniform)')
    ax.set_xlabel('Level L')
    ax.set_ylabel('P0 win fraction')
    ax.set_title('Win Probability Gradient f(L)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max(L) + 0.5)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'win_probability_gradient.png'), dpi=150)
    plt.close(fig)

    # Plot 2: First-difference sawtooth
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(L, metrics['delta_f_minimax'], color='#d62728', alpha=0.7,
           label='Δf_minimax')
    ax.plot(L, metrics['delta_f_random'], 's--', color='#1f77b4',
            label='Δf_random')
    ax.set_xlabel('Level L')
    ax.set_ylabel('Δf(L) = f(L) − f(L−1)')
    ax.set_title('First-Difference (Turn-Parity Sawtooth)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'first_difference_sawtooth.png'), dpi=150)
    plt.close(fig)

    # Plot 3: State counts per level
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(L, metrics['n_states'], color='#2ca02c', alpha=0.8)
    ax.set_xlabel('Level L')
    ax.set_ylabel('Number of states')
    ax.set_title('State Count per Level')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'state_counts.png'), dpi=150)
    plt.close(fig)

    # Plot 4: Terminal breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(L, metrics['n_pattern_wins'], color='#ff7f0e', alpha=0.8,
           label='Pattern wins')
    ax.bar(L, metrics['n_stuck'], bottom=metrics['n_pattern_wins'],
           color='#9467bd', alpha=0.8, label='Stuck (no moves)')
    ax.set_xlabel('Level L')
    ax.set_ylabel('Number of terminal states')
    ax.set_title('Terminal State Breakdown by Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'terminal_breakdown.png'), dpi=150)
    plt.close(fig)

    # Plot 5: Critical move ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(L, metrics['critical_move_ratio'], 'o-', color='#8c564b')
    ax.set_xlabel('Level L')
    ax.set_ylabel('Critical move ratio')
    ax.set_title('Critical Move Ratio by Level')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'critical_move_ratio.png'), dpi=150)
    plt.close(fig)

    print(f"  Saved 5 figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Single-board analysis of solved Okiya game')
    parser.add_argument('--plot', action='store_true',
                        help='Generate matplotlib figures')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: results/analysis.json)')
    parser.add_argument('--solution-dir', type=str, default=None,
                        help='Path to solution directory')
    parser.add_argument('--board', type=str, default=None,
                        help='Path to board.json')
    args = parser.parse_args()

    sol_dir = args.solution_dir or os.path.join(SCRIPT_DIR, 'solution')
    board_path = args.board or os.path.join(SCRIPT_DIR, 'board.json')
    db = SolutionDB(solution_dir=sol_dir, board_path=board_path)

    if not db.has_solution:
        print("ERROR: No solution found. Run okiya_solve.py first.")
        return

    print("=== Single-Board Analysis ===")
    print(f"Board: {db.meta.get('board_name', '?')}")
    print(f"Total states: {db.meta.get('total_states', 0):,}")

    metrics = compute_metrics(db)

    print(f"\nGame value: {metrics['game_value']:+d}")
    print(f"\nPer-level metrics:")
    print(f"  {'L':>3s}  {'N_states':>10s}  {'f_minimax':>9s}  {'f_random':>9s}  "
          f"{'Δf_mm':>7s}  {'n_term':>7s}  {'n_wins':>7s}  {'n_stuck':>7s}  {'CMR':>6s}")
    for i, L in enumerate(metrics['levels']):
        print(f"  {L:3d}  {metrics['n_states'][i]:10d}  "
              f"{metrics['f_minimax'][i]:9.6f}  {metrics['f_random'][i]:9.6f}  "
              f"{metrics['delta_f_minimax'][i]:+7.4f}  "
              f"{metrics['n_terminal'][i]:7d}  {metrics['n_pattern_wins'][i]:7d}  "
              f"{metrics['n_stuck'][i]:7d}  {metrics['critical_move_ratio'][i]:6.4f}")

    # save JSON
    out_path = args.output
    if out_path is None:
        out_dir = os.path.join(SCRIPT_DIR, 'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'analysis.json')

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved analysis to {out_path}")

    if args.plot:
        print("\nGenerating plots...")
        plot_analysis(metrics)


if __name__ == '__main__':
    main()
