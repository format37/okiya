"""Mutual-information analysis: which features best predict the optimal move?

Runs a per-level MI analysis on a stratified sample of (state, move) pairs,
writes ``results/mi_by_level.csv``, and prints the top-3 features per level.
"""
import numpy as np
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from features import FEATURE_NAMES, N_FEATURES
from dataset import load_dataset, build_training_data


def run_mi_analysis(db,
                    max_per_level: int = 20_000,
                    output_dir: str = None,
                    seed: int = 42):
    """Compute MI(feature, is_optimal_move) stratified by level L.

    Parameters
    ----------
    db            : SolutionDB
    max_per_level : states to sample per level (controls speed vs. accuracy)
    output_dir    : directory for CSV output (default: heuristic/results/)
    seed          : RNG seed

    Returns
    -------
    mi_table : np.ndarray  shape (n_levels, N_FEATURES)
               MI values; rows correspond to sorted unique levels in sample.
    levels_seen : list[int]
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError:
        raise ImportError("scikit-learn is required for MI analysis: pip install scikit-learn")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("[MI] Loading dataset...")
    records = load_dataset(db, max_per_level=max_per_level, seed=seed)
    print(f"[MI] {len(records):,} informative states loaded")

    X, y, levels_arr = build_training_data(records, db.pos_relation_masks)
    print(f"[MI] {len(X):,} (state, move) training samples")

    unique_levels = sorted(set(levels_arr.tolist()))
    mi_table   = np.zeros((len(unique_levels), N_FEATURES))

    print("[MI] Computing MI per level...")
    for i, L in enumerate(unique_levels):
        mask   = levels_arr == L
        X_L    = X[mask]
        y_L    = y[mask]
        n_pos  = int(y_L.sum())
        n_neg  = int((~y_L.astype(bool)).sum())
        if n_pos < 5 or n_neg < 5:
            # Not enough samples of each class
            continue
        mi = mutual_info_classif(X_L, y_L, discrete_features=False,
                                 random_state=seed)
        mi_table[i] = mi

    # ---- Print top-3 features per level ----
    print("\n=== Top-3 features per level (by MI) ===")
    print(f"{'Level':>5}  {'#samples':>8}  Top features")
    print("-" * 60)
    for i, L in enumerate(unique_levels):
        mask  = levels_arr == L
        n_smp = int(mask.sum())
        top3  = np.argsort(mi_table[i])[::-1][:3]
        feat_str = ', '.join(
            f"{FEATURE_NAMES[j]}={mi_table[i, j]:.4f}" for j in top3
        )
        print(f"  L={L:>2}  {n_smp:>8,}  {feat_str}")

    # ---- Save CSV ----
    import csv
    csv_path = os.path.join(output_dir, 'mi_by_level.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['level'] + FEATURE_NAMES)
        for i, L in enumerate(unique_levels):
            writer.writerow([L] + mi_table[i].tolist())
    print(f"\n[MI] Saved: {csv_path}")

    return mi_table, unique_levels
