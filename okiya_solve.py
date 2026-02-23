#!/usr/bin/env python3
"""Okiya complete GPU brute-force solver.

Forward BFS to enumerate all reachable states, then backward minimax
to compute the optimal value for every state. Outputs per-level .npz
files and metadata.json to the solution/ directory.
"""

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import json
import os
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Board configuration (loaded from board.json)
# ---------------------------------------------------------------------------

def load_board(path):
    with open(path) as f:
        return json.load(f)


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
# Game constants (independent of board layout)
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

SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)

EXPECTED_COUNTS = None  # board-dependent; recomputed by solver

# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------

KERNEL_SRC = r"""
__constant__ unsigned int c_win_patterns[19];
__constant__ unsigned int c_pos_relation_masks[16];
__constant__ unsigned int c_border_mask;

__global__ void expand_states(
    const unsigned long long *parent_states,
    const unsigned char      *parent_terminal,
    const int                 n_parents,
    const int                 level,
    unsigned long long       *child_states,
    signed char              *child_values
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_parents * 16) return;

    int parent_idx = tid >> 4;
    int cand_pos   = tid & 15;

    /* default: invalid */
    child_states[tid] = 0xFFFFFFFFFFFFFFFFULL;
    child_values[tid] = -2;

    if (parent_terminal[parent_idx]) return;

    unsigned long long state = parent_states[parent_idx];
    unsigned int mask_p0  = (unsigned int)((state >> 21) & 0xFFFFU);
    unsigned int mask_p1  = (unsigned int)((state >>  5) & 0xFFFFU);
    unsigned int last_pos = (unsigned int)(state & 0x1FU);
    unsigned int occupied = mask_p0 | mask_p1;

    /* position must be unclaimed */
    if (occupied & (1u << cand_pos)) return;

    /* move-validity constraint */
    if (level == 0) {
        if (!(c_border_mask & (1u << cand_pos))) return;
    } else {
        if (!(c_pos_relation_masks[last_pos] & (1u << cand_pos))) return;
    }

    /* build child state */
    unsigned int new_p0 = mask_p0;
    unsigned int new_p1 = mask_p1;
    int mover = level & 1;          /* 0 = P0's turn, 1 = P1's turn */

    if (mover == 0) new_p0 |= (1u << cand_pos);
    else            new_p1 |= (1u << cand_pos);

    unsigned long long child =
        ((unsigned long long)new_p0 << 21) |
        ((unsigned long long)new_p1 <<  5) |
        (unsigned long long)cand_pos;
    child_states[tid] = child;

    /* check if mover just won */
    unsigned int mover_mask = (mover == 0) ? new_p0 : new_p1;
    for (int w = 0; w < 19; w++) {
        if ((mover_mask & c_win_patterns[w]) == c_win_patterns[w]) {
            child_values[tid] = (mover == 0) ? 1 : -1;
            return;
        }
    }
    child_values[tid] = 0;   /* non-terminal so far */
}


__global__ void minimax_backward(
    const int        *offsets,
    const int        *flat_children,
    const signed char *next_values,
    signed char      *cur_values,
    const unsigned char *cur_terminal,
    const int         n_states,
    const int         level
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_states) return;
    if (cur_terminal[tid]) return;

    int start = offsets[tid];
    int end   = offsets[tid + 1];
    if (start == end) return;

    signed char best;
    if ((level & 1) == 0) {
        /* P0 maximises */
        best = -127;
        for (int i = start; i < end; i++) {
            signed char cv = next_values[flat_children[i]];
            if (cv > best) best = cv;
            if (best == 1) break;
        }
    } else {
        /* P1 minimises */
        best = 127;
        for (int i = start; i < end; i++) {
            signed char cv = next_values[flat_children[i]];
            if (cv < best) best = cv;
            if (best == -1) break;
        }
    }
    cur_values[tid] = best;
}
"""

# ---------------------------------------------------------------------------
# Compile and set up constants
# ---------------------------------------------------------------------------

def compile_module():
    """Compile CUDA + upload board-independent constants. Call once."""
    mod = SourceModule(KERNEL_SRC)
    drv.memcpy_htod(mod.get_global("c_win_patterns")[0], WIN_PATTERNS)
    drv.memcpy_htod(mod.get_global("c_border_mask")[0],
                    np.array([BORDER_MASK], dtype=np.uint32))
    expand_fn  = mod.get_function("expand_states")
    minimax_fn = mod.get_function("minimax_backward")
    return mod, expand_fn, minimax_fn


def upload_board_constants(mod, pos_relation_masks):
    """Upload per-board relation masks to GPU constant memory."""
    drv.memcpy_htod(mod.get_global("c_pos_relation_masks")[0],
                    pos_relation_masks)


def compile_kernels(pos_relation_masks):
    """Backward-compatible wrapper."""
    mod, expand_fn, minimax_fn = compile_module()
    upload_board_constants(mod, pos_relation_masks)
    return expand_fn, minimax_fn

# ---------------------------------------------------------------------------
# Forward BFS
# ---------------------------------------------------------------------------

def forward_bfs(expand_fn, pos_relation_masks, quiet=False):
    """Expand all reachable game states level by level.

    Returns dict  level -> {states, values, terminal, offsets, flat_children}
    """
    level_data = {}

    # level 0: single root state  (mask_p0=0, mask_p1=0, last_pos=16)
    init_state = np.uint64(16)
    level_data[0] = dict(
        states   = np.array([init_state], dtype=np.uint64),
        values   = np.array([0], dtype=np.int8),
        terminal = np.array([False]),
    )

    BLOCK = 256

    for L in range(16):
        pdata    = level_data[L]
        parents  = pdata['states']
        n_parents = len(parents)
        if n_parents == 0:
            break

        n_threads = n_parents * 16
        grid = ((n_threads + BLOCK - 1) // BLOCK, 1, 1)

        # ---------- GPU expand ----------
        d_parents  = gpuarray.to_gpu(parents)
        d_terminal = gpuarray.to_gpu(pdata['terminal'].astype(np.uint8))
        d_child_st = gpuarray.empty(n_threads, dtype=np.uint64)
        d_child_val = gpuarray.empty(n_threads, dtype=np.int8)

        expand_fn(d_parents, d_terminal,
                  np.int32(n_parents), np.int32(L),
                  d_child_st, d_child_val,
                  block=(BLOCK, 1, 1), grid=grid)

        child_st  = d_child_st.get()
        child_val = d_child_val.get()

        # free GPU memory
        del d_parents, d_terminal, d_child_st, d_child_val

        # ---------- filter invalids ----------
        valid = child_st != SENTINEL
        v_states  = child_st[valid]
        v_values  = child_val[valid]
        v_parents = (np.arange(n_threads, dtype=np.int32)[valid]) >> 4

        del child_st, child_val

        if len(v_states) == 0:
            pdata['offsets']       = np.zeros(n_parents + 1, dtype=np.int32)
            pdata['flat_children'] = np.array([], dtype=np.int32)
            break

        # ---------- dedup children ----------
        unique, first_idx, inverse = np.unique(
            v_states, return_index=True, return_inverse=True)
        n_children = len(unique)

        # win values (from expand kernel)
        win_vals = v_values[first_idx].copy()

        # ---------- no-moves check (CPU, vectorised) ----------
        m0  = ((unique >> 21) & 0xFFFF).astype(np.uint32)
        m1  = ((unique >>  5) & 0xFFFF).astype(np.uint32)
        lp  = (unique & 0x1F).astype(np.int32)
        occ = m0 | m1
        rel = pos_relation_masks[lp]
        avail = rel & ~occ
        no_moves = avail == 0

        values   = win_vals.copy()
        terminal = win_vals != 0          # wins already terminal

        stuck_player   = (L + 1) % 2     # who must move at child level
        no_moves_val   = np.int8(-1 if stuck_player == 0 else 1)
        nm_nonterminal = no_moves & ~terminal
        values[nm_nonterminal]   = no_moves_val
        terminal[nm_nonterminal] = True

        # ---------- CSR for level L ----------
        child_idx = np.searchsorted(unique, v_states).astype(np.int32)
        edge_cnt  = np.bincount(v_parents, minlength=n_parents)
        offsets   = np.zeros(n_parents + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(edge_cnt)

        # edges are already in parent-order (kernel output order preserved
        # through boolean mask), so child_idx IS the flat CSR array.
        pdata['offsets']       = offsets
        pdata['flat_children'] = child_idx

        del v_states, v_values, v_parents, child_idx

        # ---------- store level L+1 ----------
        level_data[L + 1] = dict(
            states   = unique,
            values   = values,
            terminal = terminal,
        )

        term_wins = int((win_vals != 0).sum())
        term_nm   = int(nm_nonterminal.sum())
        if not quiet:
            print(f"  Level {L+1:2d}: {n_children:>10d} states  "
                  f"({term_wins} wins, {term_nm} no-moves, "
                  f"{n_children - term_wins - term_nm} open)")

    # last level has no children → set empty CSR
    max_L = max(level_data.keys())
    d = level_data[max_L]
    if 'offsets' not in d:
        n = len(d['states'])
        d['offsets']       = np.zeros(n + 1, dtype=np.int32)
        d['flat_children'] = np.array([], dtype=np.int32)

    return level_data

# ---------------------------------------------------------------------------
# Backward minimax
# ---------------------------------------------------------------------------

def backward_minimax(level_data, minimax_fn, quiet=False):
    max_L = max(level_data.keys())
    BLOCK = 256

    for L in range(max_L - 1, -1, -1):
        cur  = level_data[L]
        nxt  = level_data[L + 1]
        n    = len(cur['states'])
        offs = cur.get('offsets')
        flat = cur.get('flat_children')
        if offs is None or len(flat) == 0:
            continue

        d_offs = gpuarray.to_gpu(offs)
        d_flat = gpuarray.to_gpu(flat)
        d_nv   = gpuarray.to_gpu(nxt['values'])
        d_cv   = gpuarray.to_gpu(cur['values'])
        d_ct   = gpuarray.to_gpu(cur['terminal'].astype(np.uint8))

        grid = ((n + BLOCK - 1) // BLOCK, 1, 1)
        minimax_fn(d_offs, d_flat, d_nv, d_cv, d_ct,
                   np.int32(n), np.int32(L),
                   block=(BLOCK, 1, 1), grid=grid)

        cur['values'] = d_cv.get()
        del d_offs, d_flat, d_nv, d_cv, d_ct

        # summary
        v = cur['values']
        p0w = int((v ==  1).sum())
        p1w = int((v == -1).sum())
        dr  = int((v ==  0).sum())
        if not quiet:
            print(f"  Level {L:2d}: P0-wins={p0w}  P1-wins={p1w}  draws={dr}")

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(level_data):
    ok = True

    # 1. state counts (expected counts are for the default board only)
    print("\n--- State counts ---")
    for L in sorted(level_data.keys()):
        n = len(level_data[L]['states'])
        print(f"  Level {L:2d}: {n:>10d}")

    # 2. structural invariants
    print("\n--- Structural invariants ---")
    for L in sorted(level_data.keys()):
        states = level_data[L]['states']
        m0 = ((states >> 21) & 0xFFFF).astype(np.uint32)
        m1 = ((states >>  5) & 0xFFFF).astype(np.uint32)
        lp = (states & 0x1F).astype(np.int32)
        # masks don't overlap
        overlap = (m0 & m1) != 0
        if overlap.any():
            print(f"  Level {L}: {overlap.sum()} states with overlapping masks!")
            ok = False
        # popcount matches level
        pc = np.zeros(len(states), dtype=np.int32)
        for bit in range(16):
            pc += ((m0 >> bit) & 1).astype(np.int32)
            pc += ((m1 >> bit) & 1).astype(np.int32)
        bad_pc = pc != L
        if bad_pc.any():
            print(f"  Level {L}: {bad_pc.sum()} states with wrong popcount!")
            ok = False
        # last_pos in correct mask (for L > 0)
        if L > 0:
            mover_prev = (L - 1) % 2  # who moved to create this level
            if mover_prev == 0:
                in_mask = (m0 >> lp) & 1
            else:
                in_mask = (m1 >> lp) & 1
            bad_lp = in_mask != 1
            if bad_lp.any():
                print(f"  Level {L}: {bad_lp.sum()} states where last_pos "
                      f"not in mover's mask!")
                ok = False

    if ok:
        print("  All structural checks passed.")

    # 3. minimax consistency
    print("\n--- Minimax consistency ---")
    for L in sorted(level_data.keys()):
        cur = level_data[L]
        offs = cur.get('offsets')
        flat = cur.get('flat_children')
        if offs is None or L + 1 not in level_data:
            continue
        nxt = level_data[L + 1]
        vals = cur['values']
        nvals = nxt['values']
        term = cur['terminal']
        is_p0 = (L % 2 == 0)
        bad = 0
        # check a sample of non-terminal states
        indices = np.where(~term)[0]
        for idx in indices[:min(len(indices), 50000)]:
            s = offs[idx]
            e = offs[idx + 1]
            if s == e:
                continue
            cv = nvals[flat[s:e]]
            expected = cv.max() if is_p0 else cv.min()
            if vals[idx] != expected:
                bad += 1
        if bad:
            print(f"  Level {L}: {bad} minimax mismatches!")
            ok = False

    if ok:
        print("  All minimax checks passed.")
    return ok

# ---------------------------------------------------------------------------
# Save solution
# ---------------------------------------------------------------------------

def save_solution(level_data, output_dir, board_cfg, init_tiles, pos_relation_masks):
    os.makedirs(output_dir, exist_ok=True)

    meta = dict(
        board_name   = board_cfg.get('name', ''),
        init_tiles   = init_tiles.tolist(),
        attributes   = board_cfg.get('attributes', {}),
        win_patterns = WIN_PATTERNS.tolist(),
        border_mask  = int(BORDER_MASK),
        pos_relation_masks = pos_relation_masks.tolist(),
        level_counts = {},
        total_states = 0,
    )

    for L in sorted(level_data.keys()):
        d = level_data[L]
        n = len(d['states'])
        meta['level_counts'][str(L)] = n
        meta['total_states'] += n

        save_kw = dict(
            states   = d['states'],
            values   = d['values'],
            terminal = d['terminal'],
        )
        if 'offsets' in d:
            save_kw['children_offsets'] = d['offsets']
            save_kw['children_indices'] = d['flat_children']

        path = os.path.join(output_dir, f'level_{L:02d}.npz')
        np.savez_compressed(path, **save_kw)

    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {len(level_data)} levels to {output_dir}/")
    print(f"  Total states: {meta['total_states']:,}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Okiya GPU brute-force solver')
    parser.add_argument('--board', type=str,
                        default=os.path.join(SCRIPT_DIR, 'board.json'),
                        help='Path to board layout JSON (default: board.json)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(SCRIPT_DIR, 'solution'),
                        help='Output directory (default: solution/)')
    args = parser.parse_args()

    t0 = time.time()

    # load board
    board_cfg = load_board(args.board)
    init_tiles = np.array(board_cfg['tiles'], dtype=np.int32)
    pos_relation_masks = compute_pos_relation_masks(init_tiles)

    print(f"Board: {board_cfg.get('name', args.board)}")
    print(f"Tiles: {init_tiles.tolist()}")

    print("\nCompiling CUDA kernels...")
    expand_fn, minimax_fn = compile_kernels(pos_relation_masks)

    print("\n=== Phase 1: Forward BFS ===")
    level_data = forward_bfs(expand_fn, pos_relation_masks)

    print("\n=== Phase 2: Backward Minimax ===")
    backward_minimax(level_data, minimax_fn)

    print("\n=== Phase 3: Verification ===")
    verify(level_data)

    print("\n=== Phase 4: Save Solution ===")
    save_solution(level_data, args.output, board_cfg,
                  init_tiles, pos_relation_masks)

    t1 = time.time()
    root_val = level_data[0]['values'][0]
    label = {1: 'P0 (Red) wins', -1: 'P1 (Black) wins', 0: 'Draw'}
    print(f"\nGame value: {root_val}  ({label.get(root_val, '?')})")
    print(f"Total time: {t1 - t0:.1f}s")


if __name__ == '__main__':
    main()
