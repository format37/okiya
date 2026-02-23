#!/usr/bin/env python3
"""Batch visualization: reads CSV from okiya_batch.py, generates publication figures.

Figures:
  1. P0 vs P1 win distribution (bar chart)
  2. Win probability gradient + confidence bands
  3. First-difference sawtooth with error bars
  4. State count distribution (histogram)
  5. Winning opening moves distribution (histogram)
  6. Critical move ratio by level (mean + confidence band)
  7. Stuck-state fraction by level
  8. Relation graph metrics vs outcome (scatter)

Usage:
    python3 okiya_plot.py --input results/batch.csv --output-dir results/figures/
"""

import numpy as np
import pandas as pd
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_batch(csv_path):
    """Load batch CSV into pandas DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def plot_all(df, output_dir):
    """Generate all 8 publication-quality figures."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'figure.dpi': 150,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
    })

    os.makedirs(output_dir, exist_ok=True)
    n_configs = len(df)

    # --- Figure 1: P0 vs P1 win distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['game_value'].value_counts().sort_index()
    labels = {1: 'P0 wins', -1: 'P1 wins', 0: 'Draw'}
    colors = {1: '#d62728', -1: '#1f77b4', 0: '#7f7f7f'}
    for val in [-1, 0, 1]:
        c = counts.get(val, 0)
        ax.bar(labels[val], c, color=colors[val], alpha=0.85)
        if c > 0:
            ax.text(labels[val], c + n_configs * 0.01, str(c),
                    ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Number of configurations')
    ax.set_title(f'Game Outcome Distribution (N={n_configs})')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '1_win_distribution.png'))
    plt.close(fig)

    # --- Helpers: per-level column arrays ---
    max_L = 16
    levels = np.arange(max_L + 1)

    def col_array(prefix):
        """Stack per-level columns into (n_configs, 17) array."""
        cols = [f'{prefix}_L{L}' for L in range(max_L + 1)]
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return None
        return df[cols].values

    f_mm = col_array('f_minimax')
    n_states = col_array('n_states')
    n_term = col_array('n_terminal')
    n_pw = col_array('n_pattern_wins')
    n_st = col_array('n_stuck')
    cmr = col_array('critical_move_ratio')

    # --- Figure 2: Win probability gradient + confidence bands ---
    if f_mm is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_f = f_mm.mean(axis=0)
        std_f = f_mm.std(axis=0)
        ax.plot(levels, mean_f, 'o-', color='#d62728', label='Mean f_minimax(L)')
        ax.fill_between(levels, mean_f - std_f, mean_f + std_f,
                         color='#d62728', alpha=0.15, label='±1 std')
        ax.set_xlabel('Level L')
        ax.set_ylabel('P0 win fraction (f_minimax)')
        ax.set_title(f'Win Probability Gradient (N={n_configs})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, max_L + 0.5)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '2_win_probability_gradient.png'))
        plt.close(fig)

    # --- Figure 3: First-difference sawtooth ---
    if f_mm is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        delta_f = np.diff(f_mm, axis=1)  # shape (n, 16)
        mean_delta = delta_f.mean(axis=0)
        std_delta = delta_f.std(axis=0)
        delta_levels = levels[1:]
        ax.bar(delta_levels, mean_delta, color='#d62728', alpha=0.7,
               yerr=std_delta, capsize=3, label='Mean Δf_minimax')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Level L')
        ax.set_ylabel('Δf(L) = f(L) − f(L−1)')
        ax.set_title(f'Turn-Parity Sawtooth (N={n_configs})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '3_sawtooth.png'))
        plt.close(fig)

    # --- Figure 4: State count distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df['total_states'], bins=30, color='#2ca02c', alpha=0.8,
            edgecolor='white')
    ax.axvline(df['total_states'].mean(), color='black', linestyle='--',
               label=f"Mean = {df['total_states'].mean():,.0f}")
    ax.set_xlabel('Total reachable states')
    ax.set_ylabel('Count')
    ax.set_title(f'State Count Distribution (N={n_configs})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '4_state_count_distribution.png'))
    plt.close(fig)

    # --- Figure 5: Winning opening moves distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    max_open = int(df['n_winning_openings'].max())
    bins = np.arange(-0.5, max_open + 1.5)
    ax.hist(df['n_winning_openings'], bins=bins, color='#ff7f0e', alpha=0.85,
            edgecolor='white', rwidth=0.85)
    ax.set_xlabel('Number of winning opening moves (for P0)')
    ax.set_ylabel('Count')
    ax.set_title(f'Winning Opening Moves Distribution (N={n_configs})')
    ax.set_xticks(range(0, max_open + 1))
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '5_winning_openings.png'))
    plt.close(fig)

    # --- Figure 6: Critical move ratio by level ---
    if cmr is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_cmr = cmr.mean(axis=0)
        std_cmr = cmr.std(axis=0)
        ax.plot(levels, mean_cmr, 'o-', color='#8c564b', label='Mean CMR')
        ax.fill_between(levels, mean_cmr - std_cmr, mean_cmr + std_cmr,
                         color='#8c564b', alpha=0.15, label='±1 std')
        ax.set_xlabel('Level L')
        ax.set_ylabel('Critical move ratio')
        ax.set_title(f'Critical Move Ratio by Level (N={n_configs})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '6_critical_move_ratio.png'))
        plt.close(fig)

    # --- Figure 7: Stuck-state fraction by level ---
    if n_term is not None and n_st is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        # stuck fraction = n_stuck / n_terminal (where n_terminal > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            stuck_frac = np.where(n_term > 0, n_st / n_term, 0.0)
        mean_sf = stuck_frac.mean(axis=0)
        std_sf = stuck_frac.std(axis=0)
        ax.plot(levels, mean_sf, 'o-', color='#9467bd', label='Mean stuck fraction')
        ax.fill_between(levels, mean_sf - std_sf, mean_sf + std_sf,
                         color='#9467bd', alpha=0.15, label='±1 std')
        ax.set_xlabel('Level L')
        ax.set_ylabel('Stuck / Total terminal')
        ax.set_title(f'Stuck-State Fraction by Level (N={n_configs})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '7_stuck_fraction.png'))
        plt.close(fig)

    # --- Figure 8: Relation graph metrics vs outcome ---
    if 'mean_degree' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        color_map = {1: '#d62728', -1: '#1f77b4', 0: '#7f7f7f'}
        colors = df['game_value'].map(color_map)

        for ax, col, label in zip(axes,
                                   ['mean_degree', 'std_degree', 'avg_clustering'],
                                   ['Mean degree', 'Std degree', 'Avg clustering']):
            for val, c in color_map.items():
                mask = df['game_value'] == val
                lbl = {1: 'P0 wins', -1: 'P1 wins', 0: 'Draw'}[val]
                ax.scatter(df.loc[mask, col], df.loc[mask, 'total_states'],
                           c=c, alpha=0.5, s=15, label=lbl)
            ax.set_xlabel(label)
            ax.set_ylabel('Total states')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Relation Graph Metrics vs Game Size (N={n_configs})',
                     fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '8_relation_graph.png'))
        plt.close(fig)

    print(f"  Saved figures to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication figures from batch Okiya results')
    parser.add_argument('--input', type=str, default=None,
                        help='Input CSV path (default: results/batch.csv)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for figures (default: results/figures/)')
    args = parser.parse_args()

    csv_path = args.input or os.path.join(SCRIPT_DIR, 'results', 'batch.csv')
    out_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'results', 'figures')

    if not os.path.exists(csv_path):
        print(f"ERROR: Input CSV not found: {csv_path}")
        print("Run okiya_batch.py first.")
        return

    print(f"=== Okiya Batch Visualization ===")
    print(f"Input: {csv_path}")

    df = load_batch(csv_path)
    print(f"Loaded {len(df)} configurations")
    print(f"\nGenerating figures...")

    plot_all(df, out_dir)
    print("Done.")


if __name__ == '__main__':
    main()
