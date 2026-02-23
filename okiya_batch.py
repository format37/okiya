#!/usr/bin/env python3
"""Batch solver: N random Okiya board configurations → CSV with metrics.

Compiles CUDA once, then for each random tile permutation:
  - uploads board constants
  - runs forward BFS + backward minimax
  - extracts ~100 columns of metrics
  - appends one row to the output CSV

No solution files are saved (each is ~190 MB). Metrics are extracted in-memory.

Usage:
    python3 okiya_batch.py --n 1000 --seed 42 --output results/batch.csv
    python3 okiya_batch.py --n 1000 --seed 42 --start 500  # resume from config 500
"""

import numpy as np
import csv
import os
import time
import argparse
import gc

from okiya_solve import (
    compile_module, upload_board_constants,
    forward_bfs, backward_minimax,
    compute_pos_relation_masks, WIN_PATTERNS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def relation_graph_metrics(pos_relation_masks):
    """Compute mean degree, std degree, and average clustering coefficient
    of the tile-relation graph (16-node undirected graph)."""
    degrees = np.zeros(16, dtype=np.int32)
    adj = np.zeros((16, 16), dtype=bool)
    for p in range(16):
        for q in range(16):
            if p != q and (pos_relation_masks[p] & (1 << q)):
                adj[p][q] = True
                degrees[p] += 1

    mean_deg = float(degrees.mean())
    std_deg = float(degrees.std())

    # average clustering coefficient
    clustering = np.zeros(16, dtype=np.float64)
    for p in range(16):
        neighbors = [q for q in range(16) if adj[p][q]]
        k = len(neighbors)
        if k < 2:
            clustering[p] = 0.0
            continue
        links = 0
        for i in range(k):
            for j in range(i + 1, k):
                if adj[neighbors[i]][neighbors[j]]:
                    links += 1
        clustering[p] = 2.0 * links / (k * (k - 1))
    avg_clustering = float(clustering.mean())

    return mean_deg, std_deg, avg_clustering


def extract_metrics(level_data, tiles, pos_relation_masks):
    """Extract all per-config metrics from the solved game tree."""
    max_level = max(level_data.keys())
    root_value = int(level_data[0]['values'][0])
    total_states = sum(len(level_data[L]['states']) for L in level_data)

    # winning opening moves: at level 0, how many children have value == root_value?
    # Actually: how many children of the root have value == +1 (P0 wins)?
    n_winning_openings = 0
    if 1 in level_data:
        nxt = level_data[1]
        cur = level_data[0]
        if 'offsets' in cur and 'flat_children' in cur:
            offsets = cur['offsets']
            indices = cur['flat_children']
            start, end = int(offsets[0]), int(offsets[1]) if len(offsets) > 1 else 0
            if start < end:
                child_vals = nxt['values'][indices[start:end]]
                n_winning_openings = int((child_vals == 1).sum())

    row = {
        'tiles': ','.join(str(t) for t in tiles),
        'game_value': root_value,
        'total_states': total_states,
        'n_winning_openings': n_winning_openings,
    }

    # per-level metrics
    for L in range(17):
        if L in level_data:
            data = level_data[L]
            n = len(data['states'])
            values = data['values']
            terminal = data['terminal'].astype(bool)

            f_mm = float((values == 1).sum()) / n if n > 0 else 0.0
            n_term = int(terminal.sum())

            # pattern wins vs stuck
            if n_term > 0 and L > 0:
                term_idx = np.where(terminal)[0]
                term_states = data['states'][term_idx]
                m0 = ((term_states >> 21) & 0xFFFF).astype(np.uint32)
                m1 = ((term_states >>  5) & 0xFFFF).astype(np.uint32)
                prev_mover = (L - 1) % 2
                mover_mask = m0 if prev_mover == 0 else m1

                is_pw = np.zeros(len(term_idx), dtype=bool)
                for wp in WIN_PATTERNS:
                    is_pw |= (mover_mask & int(wp)) == int(wp)
                n_pw = int(is_pw.sum())
                n_st = n_term - n_pw
            else:
                n_pw = 0
                n_st = 0

            # critical move ratio
            cmr = 0.0
            if ('children_offsets' in data and 'children_indices' in data
                    and L + 1 in level_data):
                nxt = level_data[L + 1]
                offsets = data['children_offsets']
                indices = data['children_indices']
                edge_counts = np.diff(offsets)
                non_term = ~terminal
                has_children = (edge_counts > 0) & non_term
                if has_children.any() and len(indices) > 0:
                    parent_ids = np.repeat(
                        np.arange(n, dtype=np.int64), edge_counts)
                    parent_vals = values[parent_ids]
                    child_vals = nxt['values'][indices]
                    mistakes = parent_vals != child_vals
                    total_edges = len(parent_vals)
                    cmr = float(mistakes.sum()) / total_edges if total_edges > 0 else 0.0

            row[f'n_states_L{L}'] = n
            row[f'f_minimax_L{L}'] = f_mm
            row[f'n_terminal_L{L}'] = n_term
            row[f'n_pattern_wins_L{L}'] = n_pw
            row[f'n_stuck_L{L}'] = n_st
            row[f'critical_move_ratio_L{L}'] = cmr
        else:
            row[f'n_states_L{L}'] = 0
            row[f'f_minimax_L{L}'] = 0.0
            row[f'n_terminal_L{L}'] = 0
            row[f'n_pattern_wins_L{L}'] = 0
            row[f'n_stuck_L{L}'] = 0
            row[f'critical_move_ratio_L{L}'] = 0.0

    # relation graph metrics
    mean_deg, std_deg, avg_clust = relation_graph_metrics(pos_relation_masks)
    row['mean_degree'] = mean_deg
    row['std_degree'] = std_deg
    row['avg_clustering'] = avg_clust

    return row


def get_fieldnames():
    """Return the ordered list of CSV column names."""
    fields = ['config_id', 'seed', 'tiles', 'game_value', 'total_states',
              'n_winning_openings']
    for L in range(17):
        fields.extend([
            f'n_states_L{L}',
            f'f_minimax_L{L}',
            f'n_terminal_L{L}',
            f'n_pattern_wins_L{L}',
            f'n_stuck_L{L}',
            f'critical_move_ratio_L{L}',
        ])
    fields.extend(['mean_degree', 'std_degree', 'avg_clustering'])
    return fields


def main():
    parser = argparse.ArgumentParser(
        description='Batch Okiya solver: N random configs → CSV')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of random configurations to solve')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: results/batch.csv)')
    parser.add_argument('--start', type=int, default=0,
                        help='Skip first N configs (for resume)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-level BFS/minimax output')
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        out_dir = os.path.join(SCRIPT_DIR, 'results')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'batch.csv')

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    quiet = not args.verbose

    print(f"=== Okiya Batch Solver ===")
    print(f"Configs: {args.n}, seed: {args.seed}, start: {args.start}")
    print(f"Output: {out_path}")

    print("\nCompiling CUDA module...")
    t_compile = time.time()
    mod, expand_fn, minimax_fn = compile_module()
    print(f"  Compiled in {time.time() - t_compile:.1f}s")

    fieldnames = get_fieldnames()

    # determine write mode: append if resuming and file exists, else write new
    write_header = True
    file_mode = 'w'
    if args.start > 0 and os.path.exists(out_path):
        file_mode = 'a'
        write_header = False

    t_total = time.time()

    with open(out_path, file_mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i in range(args.start, args.n):
            t_config = time.time()

            # generate random tile permutation
            rng = np.random.default_rng(args.seed + i)
            tiles = rng.permutation(16).astype(np.int32)

            pos_relation_masks = compute_pos_relation_masks(tiles)
            upload_board_constants(mod, pos_relation_masks)

            print(f"\n[{i+1}/{args.n}] Config {i} "
                  f"(tiles={tiles.tolist()[:4]}...)")

            level_data = forward_bfs(expand_fn, pos_relation_masks, quiet=quiet)
            backward_minimax(level_data, minimax_fn, quiet=quiet)

            row = extract_metrics(level_data, tiles, pos_relation_masks)
            row['config_id'] = i
            row['seed'] = args.seed + i

            writer.writerow(row)
            csvfile.flush()

            root_val = level_data[0]['values'][0]
            n_total = sum(len(level_data[L]['states']) for L in level_data)
            elapsed = time.time() - t_config

            print(f"  value={root_val:+d}  states={n_total:,}  "
                  f"time={elapsed:.1f}s")

            # free memory before next config
            del level_data
            gc.collect()

    elapsed_total = time.time() - t_total
    print(f"\n=== Done ===")
    print(f"Solved {args.n - args.start} configs in {elapsed_total:.0f}s "
          f"({elapsed_total/3600:.1f}h)")
    print(f"Results: {out_path}")


if __name__ == '__main__':
    main()
