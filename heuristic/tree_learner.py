"""Decision-tree training and human-readable rule extraction.

Trains two classifiers (depth-4 and depth-8) on the labelled (state, move)
dataset from the canonical board, exports plain-text rule listings, and
pickles the models for use by make_tree_heuristic().
"""
import numpy as np
import os
import sys
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from features import FEATURE_NAMES
from dataset import load_dataset, build_training_data


def train_trees(db,
                max_per_level: int = 35_000,
                depths: tuple = (4, 8),
                output_dir: str = None,
                seed: int = 42):
    """Train decision trees and save models + rule text.

    Parameters
    ----------
    db            : SolutionDB
    max_per_level : states sampled per level (larger = more accurate trees)
    depths        : tuple of tree max_depth values to train
    output_dir    : directory for output files (default: heuristic/results/)
    seed          : RNG seed

    Returns
    -------
    dict mapping depth -> fitted DecisionTreeClassifier
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("[Tree] Loading dataset...")
    records = load_dataset(db, max_per_level=max_per_level, seed=seed)
    print(f"[Tree] {len(records):,} informative states loaded")

    X, y, levels = build_training_data(records, db.pos_relation_masks)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"[Tree] {len(X):,} samples  |  optimal={n_pos:,}  suboptimal={n_neg:,}")

    clfs = {}
    for depth in depths:
        print(f"\n[Tree] Training DecisionTree depth={depth}...")
        clf = DecisionTreeClassifier(
            max_depth=depth,
            class_weight='balanced',   # compensate for class imbalance
            random_state=seed,
        )
        clf.fit(X, y)

        y_pred = clf.predict(X)
        acc    = accuracy_score(y, y_pred)
        print(f"[Tree]   Training accuracy = {acc:.4f}")

        # Save model
        pkl_path = os.path.join(output_dir, f'tree_depth{depth}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"[Tree]   Saved model: {pkl_path}")

        # Export human-readable rules
        rules_text = export_text(clf, feature_names=FEATURE_NAMES)
        txt_path   = os.path.join(output_dir, f'rules_depth{depth}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Decision tree depth={depth}  |  training_accuracy={acc:.4f}\n")
            f.write("=" * 60 + "\n")
            f.write(rules_text)
        print(f"[Tree]   Saved rules: {txt_path}")

        clfs[depth] = clf

    return clfs


def load_tree(depth: int, output_dir: str = None):
    """Load a previously trained tree from disk."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    pkl_path = os.path.join(output_dir, f'tree_depth{depth}.pkl')
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
