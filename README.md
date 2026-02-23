# Strongly Solving Okiya

GPU-accelerated brute-force solver for the board game
[Okiya](https://boardgamegeek.com/boardgame/125311/okiya) (originally
published as Niya), providing a **strong solution**: the minimax value is
computed for every reachable state (~9.5M per configuration).

> **Paper:**
> A. Iurasov, "Strongly Solving Okiya: Complete Game-Theoretic Analysis and
> Statistical Win Dynamics via GPU-Accelerated Brute Force," 2026.
> See [`paper/`](paper/) for the full LaTeX source.

## Key Results

| Metric | Value |
|---|---|
| Reachable states per config | ~9.5M (8.7% of upper bound) |
| Solve time (single config, RTX 4090) | ~17 s |
| Configs solved (batch) | 1,000 (seed 42, ~1 h total) |
| P0 (first-player) win rate | 67.2% |
| Draws | 0 across ~9.48 B aggregate states |
| Mean winning openings | 1.37 / 12 |

## Repository Structure

```
.
├── okiya_core.py        # Shared game logic: SolutionDB, GameState, helpers
├── okiya_solve.py       # GPU solver: forward BFS + backward minimax
├── okiya_query.py       # CLI query interface: lookup any state, show best moves
├── okiya_gui.py         # Pygame GUI: interactive play with move visualization
├── okiya_batch.py       # Batch solver: solve N random configurations
├── okiya_analyze.py     # Per-board analysis: extract f(L), CMR, terminal stats
├── okiya_plot.py        # Generate figures from batch results
├── board.json           # Board layout (tile permutation on the 4×4 grid)
├── paper/               # LaTeX source for the arXiv paper
│   └── paper.tex
├── images/              # Tile artwork (16 tiles + 2 player markers)
├── results/             # Batch CSV, analysis JSON, generated figures
│   ├── batch.csv
│   ├── analysis.json
│   └── figures/
└── solution/            # Solver output per level (generated, git-ignored)
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- `pycuda`, `numpy`, `Pillow`, `pygame`

```
pip install pycuda numpy Pillow pygame
```

## Quick Start

**1. Solve the game:**

```bash
python3 okiya_solve.py
```

Reads `board.json`, solves all ~9.5M states in ~17 seconds (RTX 4090), writes results to `solution/`.

**2. Play in the GUI:**

```bash
python3 okiya_gui.py
```

Opens a resizable Pygame window with the 4×4 board. Click tiles to play moves. Features:

- Move quality overlays — win probability (%) under uniform random play, color-coded green (favorable) / red (unfavorable)
- Undo / Redo / Reset — buttons or `Ctrl+Z` / `Ctrl+Y` / `Ctrl+R`
- **Shuffle** — randomize the tile layout (clears current solution)
- **Solve** — run the GPU solver in the background (~20s), solution loads automatically
- Win and stuck detection with game-over display
- Works without a solution (no move recommendations, but play is still possible)

**3. Query a position (CLI):**

```bash
# Initial position — show all moves
python3 okiya_query.py --state "0,0,16"

# With board image
python3 okiya_query.py --state "0,0,16" --image

# Interactive mode
python3 okiya_query.py

# Limit to top 3 moves
python3 okiya_query.py --top-n 3
```

**4. Batch solve and analyze:**

```bash
# Solve 1,000 random configurations
python3 okiya_batch.py

# Analyze a single board in detail
python3 okiya_analyze.py

# Generate all figures
python3 okiya_plot.py
```

## Game Rules

[Okiya](https://boardgamegeek.com/boardgame/125311/okiya) is a two-player
abstract strategy game by Bruno Cathala (Blue Orange Games, 2013):

- 4×4 board with 16 unique tiles, each having 2 attributes (4 plants × 4 weather)
- Two players (P0=Red, P1=Blue) alternate turns; P0 goes first
- **Turn 0:** Must pick from the 12 border positions (not the 4 center squares)
- **Later turns:** Must pick an unclaimed tile that is *related* to the last-played tile (shares a plant or weather attribute)
- **Win:** Claim any 4 in a row, 4 in a column, 4 on a diagonal, or 2×2 square (19 patterns total)
- **No valid moves:** The stuck player loses

## How It Works

### State Encoding

Each game state is packed into a `uint64` (37 bits):

```
bits 36–21:  mask_p0   (16-bit bitmask of P0's claimed positions)
bits 20–5:   mask_p1   (16-bit bitmask of P1's claimed positions)
bits  4–0:   last_pos  (0–15 = board position, 16 = none)
```

### Phase 1 — Forward BFS

Starting from the empty board, expands all reachable states level by level (level = number of tiles placed). A CUDA kernel (`expand_states`) tests all 16 candidate positions for each parent state in parallel, checks move validity and win conditions. States are deduplicated with `np.unique` on CPU. A no-moves check marks stuck-player states as terminal.

### Phase 2 — Backward Minimax

Processes levels in reverse (15 → 0). A CUDA kernel (`minimax_backward`) computes each non-terminal state's value from its children via a CSR (Compressed Sparse Row) graph. P0 maximizes (+1), P1 minimizes (−1), with early exit on best possible value.

### Verification

The solver checks three invariants:
1. **Structural:** masks don't overlap, popcount matches level, last_pos is in the correct player's mask
2. **Minimax consistency:** every non-terminal state's value equals max/min of its children
3. **State-count comparison:** per-level counts printed for manual inspection

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

## License

MIT
