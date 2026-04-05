"""Main pipeline entry point.

Usage
-----
  cd /home/alex/projects/okiya
  python heuristic/run_pipeline.py [--skip-mi] [--skip-trees] [--skip-eval]
                                   [--n-games N] [--max-per-level N]

Stages
------
1. Load canonical board + solution DB
2. MI analysis  → results/mi_by_level.csv
3. Train decision trees (depth 4, 8) → results/tree_depth*.pkl + rules_*.txt
4. Evaluate all heuristics → results/move_accuracy.csv + win_rates.csv
5. Print summary

All output lands in  heuristic/results/
"""
import argparse
import os
import sys
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from okiya_core import SolutionDB
from heuristic.analysis    import run_mi_analysis
from heuristic.tree_learner import train_trees, load_tree
from heuristic.heuristics  import (
    HEURISTIC_REGISTRY,
    make_oracle_heuristic,
    make_tree_heuristic,
)
from heuristic.evaluate    import run_full_evaluation

RESULTS_DIR = os.path.join(HERE, 'results')


def parse_args():
    p = argparse.ArgumentParser(description='Okiya heuristic discovery pipeline')
    p.add_argument('--skip-mi',    action='store_true', help='Skip MI analysis')
    p.add_argument('--skip-trees', action='store_true', help='Skip tree training')
    p.add_argument('--skip-eval',  action='store_true', help='Skip evaluation')
    p.add_argument('--n-games',       type=int, default=1000,
                   help='Games per pair in win-rate simulation (default 1000)')
    p.add_argument('--max-per-level', type=int, default=20_000,
                   help='States sampled per level for dataset (default 20000)')
    p.add_argument('--seed', type=int, default=42, help='Global RNG seed')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load solution DB
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Okiya Heuristic Discovery Pipeline")
    print("=" * 60)
    print("\n[1] Loading SolutionDB...")
    db = SolutionDB()
    if not db.has_solution:
        print("ERROR: No solution found in solution/.  Run okiya_solve.py first.")
        sys.exit(1)
    total = sum(db.meta.get('level_sizes', {}).values())
    print(f"    Board: {db.init_tiles.tolist()}")
    print(f"    States: {total:,}")

    # ------------------------------------------------------------------
    # 2. MI analysis
    # ------------------------------------------------------------------
    if not args.skip_mi:
        print("\n[2] Mutual-Information Analysis")
        run_mi_analysis(db,
                        max_per_level=args.max_per_level,
                        output_dir=RESULTS_DIR,
                        seed=args.seed)
    else:
        print("\n[2] MI analysis skipped")

    # ------------------------------------------------------------------
    # 3. Decision tree training
    # ------------------------------------------------------------------
    trained_trees = {}
    if not args.skip_trees:
        print("\n[3] Training decision trees")
        trained_trees = train_trees(db,
                                    max_per_level=args.max_per_level,
                                    depths=(4, 8),
                                    output_dir=RESULTS_DIR,
                                    seed=args.seed)
    else:
        print("\n[3] Tree training skipped — trying to load existing models")
        for depth in (4, 8):
            clf = load_tree(depth, output_dir=RESULTS_DIR)
            if clf is not None:
                trained_trees[depth] = clf
                print(f"    Loaded tree depth={depth}")

    # ------------------------------------------------------------------
    # 4. Build full heuristic list for evaluation
    # ------------------------------------------------------------------
    if args.skip_eval:
        print("\n[4] Evaluation skipped")
        return

    print("\n[4] Evaluation")
    named_heuristics = list(HEURISTIC_REGISTRY)

    # Add tree heuristics if available
    rel_masks = db.pos_relation_masks
    for depth in (4, 8):
        if depth in trained_trees:
            h = make_tree_heuristic(trained_trees[depth], rel_masks,
                                    name=f'H5_Tree{depth}')
            named_heuristics.append((f'H5_Tree{depth}', h))

    # Add oracle last
    named_heuristics.append(('H_Oracle', make_oracle_heuristic(db)))

    run_full_evaluation(
        db,
        named_heuristics,
        n_games=args.n_games,
        max_per_level=min(args.max_per_level, 10_000),
        output_dir=RESULTS_DIR,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete.  Results in:", RESULTS_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
