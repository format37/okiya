# Okiya (Niya) — Complete GPU Solver

Brute-force solver for the board game [Okiya (Niya)](https://boardgamegeek.com/boardgame/125311/okiya) using CUDA on an NVIDIA GPU.
Enumerates every reachable game state (~9.8 million), computes the minimax-optimal value for each, and provides an interactive query interface to explore perfect play.

**Result for the default board:** P0 (Red) wins with perfect play. The only winning opening move is position 2 (tile Maple+Rain).

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- `pycuda`, `numpy`, `Pillow`

```
pip install pycuda numpy Pillow
```

## Files

| File | Description |
|---|---|
| `board.json` | Board layout — tile placement on the 4×4 grid |
| `okiya_solve.py` | GPU solver: forward BFS + backward minimax |
| `okiya_query.py` | Query interface: lookup any state, show best moves |
| `images/` | Tile artwork (16 tiles + 2 player markers) |
| `solution/` | Solver output (generated, git-ignored) |

## Quick Start

**1. Solve the game:**

```bash
python okiya_solve.py
```

Reads `board.json`, solves all ~9.8M states in ~20 seconds (RTX 4090), writes results to `solution/`.

**2. Query a position:**

```bash
# Initial position — show all moves
python okiya_query.py --state "0,0,16"

# With board image
python okiya_query.py --state "0,0,16" --image

# Interactive mode
python okiya_query.py

# Limit to top 3 moves
python okiya_query.py --top-n 3
```

**3. Interactive mode commands:**

| Command | Action |
|---|---|
| `<pos>` | Play move at position 0–15 |
| `<m0> <m1> <lp>` | Jump to specific state |
| `start` | Reset to initial position |
| `img` | Save board image to `images/board.png` |
| `img path.png` | Save board image to custom path |
| `q` | Quit |

## Board Configuration

`board.json` defines the tile layout:

```json
{
  "name": "Okiya (Niya)",
  "tiles": [8, 1, 3, 15, 9, 10, 12, 2, 14, 6, 5, 0, 11, 4, 7, 13],
  "attributes": {
    "a": ["Maple", "Cherry", "Pine", "Iris"],
    "b": ["Sun", "Tanzaku", "Bird", "Rain"]
  }
}
```

- `tiles[i]` — tile type at board position `i` (row-major, 0=top-left, 15=bottom-right)
- Each tile type `t` has two attributes: `a[t // 4]` and `b[t % 4]`
- Two tiles are **related** if they share either attribute

To solve a different tile arrangement, edit `tiles` in `board.json` and re-run the solver.

## Game Rules

- 4×4 board with 16 unique tiles, each having 2 attributes (4 plants × 4 weather)
- Two players (P0=Red, P1=Black) alternate turns; P0 goes first
- **Turn 0:** Must pick from the 12 border positions (not the 4 center squares)
- **Later turns:** Must pick an unclaimed tile that is *related* to the last-played tile
- **Win:** Claim any 4 in a row, 4 in a column, or 2×2 square (17 patterns total)
- **No valid moves:** The stuck player loses

## How It Works

### State Encoding

Each game state is packed into a `uint64`:

```
bits 37–21:  mask_p0   (16-bit bitmask of P0's claimed positions)
bits 20–5:   mask_p1   (16-bit bitmask of P1's claimed positions)
bits 4–0:    last_pos  (0–15 = board position, 16 = none)
```

### Phase 1 — Forward BFS

Starting from the empty board, expands all reachable states level by level (level = number of tiles placed). A CUDA kernel (`expand_states`) tests all 16 candidate positions for each parent state in parallel, checks move validity and win conditions. States are deduplicated with `np.unique` on CPU. A no-moves check marks stuck-player states as terminal.

### Phase 2 — Backward Minimax

Processes levels in reverse (15 → 0). A CUDA kernel (`minimax_backward`) computes each non-terminal state's value from its children via a CSR (Compressed Sparse Row) graph. P0 maximizes (+1), P1 minimizes (−1), with early exit on best possible value.

### Verification

The solver checks three invariants:
1. **State counts** match the reference BFS enumeration
2. **Structural:** masks don't overlap, popcount matches level, last_pos is in the correct player's mask
3. **Minimax consistency:** every non-terminal state's value equals max/min of its children

### Output

Per-level `.npz` files in `solution/` containing:
- `states` — sorted `uint64` array
- `values` — `int8` minimax values (+1 = P0 wins, −1 = P1 wins)
- `terminal` — boolean flags
- `children_offsets`, `children_indices` — CSR graph edges

Plus `metadata.json` with board config, game constants, and level statistics.

## Stats

| Metric | Value |
|---|---|
| Total unique states | 9,800,376 |
| Peak level (12) | 2,248,309 states |
| Solve time (RTX 4090) | ~20 seconds |
| Solution size on disk | ~190 MB |
| Game outcome | P0 wins |
| Winning first moves | 1 out of 12 |
