#!/usr/bin/env python3
"""Query interface for the solved Okiya game.

Usage:
    python okiya_query.py                            # interactive mode
    python okiya_query.py --state "mask_p0,mask_p1,last_pos"
    python okiya_query.py --top-n 3                  # limit moves shown
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

from okiya_core import (SolutionDB, popcount, encode_state, decode_state,
                         value_label)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared solution database
# ---------------------------------------------------------------------------

_db = SolutionDB()

# Backward-compatible aliases
INIT_TILES         = _db.init_tiles
POS_RELATION_MASKS = _db.pos_relation_masks
TILE_A_NAMES       = _db.tile_a_names
TILE_B_NAMES       = _db.tile_b_names


def tile_name(tile_type):
    return _db.tile_name(tile_type)


def lookup(mask_p0, mask_p1, last_pos, top_n=None):
    return _db.lookup(mask_p0, mask_p1, last_pos, top_n=top_n)

# ---------------------------------------------------------------------------
# Board image renderer
# ---------------------------------------------------------------------------

IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')
FONT_PATH  = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"

CELL   = 150          # cell size in pixels
TILE   = 138          # tile artwork scaled to this
PAD    = (CELL - TILE) // 2
MARGIN_L = 55         # left margin for row labels
MARGIN_T = 45         # top margin for column labels
BOARD_PX = CELL * 4
LEGEND_H = 80         # space below the board for legend text


def _load_tile_image(tile_type):
    # Images are indexed as weather*4+plant, tile types are plant*4+weather
    img_idx = (tile_type % 4) * 4 + (tile_type // 4)
    path = os.path.join(IMAGES_DIR, f'lowgardentile{img_idx}.png')
    return Image.open(path).convert('RGBA').resize(
        (TILE, TILE), Image.LANCZOS)


def draw_board_image(mask_p0, mask_p1, last_pos,
                     moves=None, output_path=None):
    """Render the board state as a PNG with actual tile artwork.

    Parameters
    ----------
    mask_p0, mask_p1 : int   – 16-bit position bitmasks for each player
    last_pos         : int   – last-played position (0-15, or 16=none)
    moves            : list  – [(pos, child_value), ...] valid moves to annotate
    output_path      : str   – where to save (default: images/board.png)

    Returns the path to the saved image.
    """
    level = popcount(mask_p0) + popcount(mask_p1)
    mover = 'P0 (Red)' if level % 2 == 0 else 'P1 (Black)'

    w = MARGIN_L + BOARD_PX + 10
    h = MARGIN_T + BOARD_PX + LEGEND_H
    img  = Image.new('RGBA', (w, h), (235, 230, 215, 255))
    draw = ImageDraw.Draw(img)

    # fonts
    fnt_hdr    = ImageFont.truetype(FONT_PATH, 18)
    fnt_pos    = ImageFont.truetype(FONT_PATH, 13)
    fnt_player = ImageFont.truetype(FONT_PATH, 44)
    fnt_move   = ImageFont.truetype(FONT_PATH, 18)
    fnt_legend = ImageFont.truetype(FONT_PATH, 14)
    fnt_tid    = ImageFont.truetype(FONT_PATH, 12)

    # column / row headers
    for col in range(4):
        cx = MARGIN_L + col * CELL + CELL // 2
        draw.text((cx, 12), f"C{col}", fill=(80, 80, 80),
                  font=fnt_hdr, anchor="mt")
    for row in range(4):
        cy = MARGIN_T + row * CELL + CELL // 2
        draw.text((12, cy), f"R{row}", fill=(80, 80, 80),
                  font=fnt_hdr, anchor="lm")

    # build a set of move positions for fast lookup
    move_map = {}
    if moves:
        for pos, val in moves:
            move_map[pos] = val

    # ------ draw each cell ------
    for pos in range(16):
        row, col = pos // 4, pos % 4
        tile_type = int(INIT_TILES[pos])
        x0 = MARGIN_L + col * CELL
        y0 = MARGIN_T + row * CELL

        # paste tile artwork
        tile_img = _load_tile_image(tile_type)
        img.paste(tile_img, (x0 + PAD, y0 + PAD), tile_img)

        is_p0   = bool(mask_p0 & (1 << pos))
        is_p1   = bool(mask_p1 & (1 << pos))
        is_last = (pos == last_pos) and (last_pos < 16)
        claimed = is_p0 or is_p1

        # ---- coloured overlay for claimed tiles ----
        if claimed:
            ov = Image.new('RGBA', (TILE, TILE), (0, 0, 0, 0))
            ov_d = ImageDraw.Draw(ov)
            if is_p0:
                ov_d.rectangle([0, 0, TILE - 1, TILE - 1],
                               fill=(210, 35, 35, 110))
            else:
                ov_d.rectangle([0, 0, TILE - 1, TILE - 1],
                               fill=(35, 55, 195, 110))
            region = img.crop((x0 + PAD, y0 + PAD,
                               x0 + PAD + TILE, y0 + PAD + TILE))
            region = Image.alpha_composite(region, ov)
            img.paste(region, (x0 + PAD, y0 + PAD))

            # large player letter
            cx, cy = x0 + CELL // 2, y0 + CELL // 2
            letter = 'R' if is_p0 else 'B'
            # shadow
            draw.text((cx + 2, cy + 2), letter, fill=(0, 0, 0, 180),
                      font=fnt_player, anchor="mm")
            draw.text((cx, cy), letter, fill=(255, 255, 255, 230),
                      font=fnt_player, anchor="mm")

        # ---- gold border on last move ----
        if is_last:
            for i in range(3):
                draw.rectangle(
                    [x0 + PAD - i - 1, y0 + PAD - i - 1,
                     x0 + PAD + TILE + i, y0 + PAD + TILE + i],
                    outline=(255, 200, 0, 255))

        # ---- position number (top-left) ----
        draw.text((x0 + PAD + 3, y0 + PAD + 1), str(pos),
                  fill=(255, 255, 60), font=fnt_pos,
                  stroke_width=1, stroke_fill=(0, 0, 0))

        # ---- tile type id (top-right) ----
        tid_str = f"t{tile_type}"
        bb = draw.textbbox((0, 0), tid_str, font=fnt_tid)
        tw = bb[2] - bb[0]
        draw.text((x0 + PAD + TILE - tw - 3, y0 + PAD + 1), tid_str,
                  fill=(255, 255, 60), font=fnt_tid,
                  stroke_width=1, stroke_fill=(0, 0, 0))

        # ---- tile attributes (bottom, on unclaimed only) ----
        if not claimed:
            a_short = TILE_A_NAMES[tile_type // 4][:3]
            b_short = TILE_B_NAMES[tile_type % 4][:3]
            attr_str = f"{a_short}+{b_short}"
            bb2 = draw.textbbox((0, 0), attr_str, font=fnt_tid)
            aw = bb2[2] - bb2[0]
            draw.text((x0 + PAD + (TILE - aw) // 2,
                       y0 + PAD + TILE - 15), attr_str,
                      fill=(255, 255, 200), font=fnt_tid,
                      stroke_width=1, stroke_fill=(0, 0, 0))

        # ---- move outcome indicator ----
        if not claimed and pos in move_map:
            mv = move_map[pos]
            cx, cy = x0 + CELL // 2, y0 + CELL // 2
            r = 17
            if mv == 1:
                fill_c = (20, 170, 20, 210)
                label  = "+1"
            else:
                fill_c = (190, 30, 30, 210)
                label  = "\u20131"   # minus-1 with en-dash
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=fill_c, outline=(255, 255, 255, 240), width=2)
            draw.text((cx, cy), label, fill=(255, 255, 255),
                      font=fnt_move, anchor="mm")

    # ------ grid lines ------
    grid_c = (90, 70, 50, 255)
    for i in range(5):
        x = MARGIN_L + i * CELL
        draw.line([(x, MARGIN_T), (x, MARGIN_T + BOARD_PX)],
                  fill=grid_c, width=2)
    for i in range(5):
        y = MARGIN_T + i * CELL
        draw.line([(MARGIN_L, y), (MARGIN_L + BOARD_PX, y)],
                  fill=grid_c, width=2)

    # ------ legend / status below board ------
    ly = MARGIN_T + BOARD_PX + 8

    value, _ = lookup(mask_p0, mask_p1, last_pos)
    val_str = f"{value:+d} ({value_label(value)})" if value is not None else "?"
    status = (f"Level {level}   Turn: {mover}   "
              f"Value: {val_str}")
    draw.text((MARGIN_L, ly), status,
              fill=(40, 40, 40), font=fnt_legend)

    ly += 20
    legend_items = [
        ("\u25a0 R = P0 (Red)", (210, 35, 35)),
        ("\u25a0 B = P1 (Black)", (35, 55, 195)),
        ("\u25cb = last move", (200, 170, 0)),
    ]
    lx = MARGIN_L
    for text, col in legend_items:
        draw.text((lx, ly), text, fill=col, font=fnt_legend)
        bb = draw.textbbox((0, 0), text, font=fnt_legend)
        lx += (bb[2] - bb[0]) + 18

    ly += 18
    draw.text((MARGIN_L, ly),
              "\u25cf green +1 = P0 wins    \u25cf red \u20131 = P1 wins",
              fill=(60, 60, 60), font=fnt_legend)

    # ------ save ------
    out = img.convert('RGB')
    if output_path is None:
        output_path = os.path.join(IMAGES_DIR, 'board.png')
    out.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# ASCII display (kept for terminals without image viewers)
# ---------------------------------------------------------------------------

def draw_board(mask_p0, mask_p1, last_pos):
    """ASCII board with tile types and player markers."""
    lines = []
    lines.append("     Col0  Col1  Col2  Col3")
    lines.append("    +-----+-----+-----+-----+")
    for row in range(4):
        parts = [f"R{row} |"]
        for col in range(4):
            pos = row * 4 + col
            tile = int(INIT_TILES[pos])
            if mask_p0 & (1 << pos):
                marker = f" [R] " if pos == last_pos else "  R  "
            elif mask_p1 & (1 << pos):
                marker = f" [B] " if pos == last_pos else "  B  "
            else:
                marker = f" {tile:>2}  "
            parts.append(f"{marker}|")
        lines.append("".join(parts))
        lines.append("    +-----+-----+-----+-----+")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified display
# ---------------------------------------------------------------------------

def show_state(mask_p0, mask_p1, last_pos, top_n=None, image=False):
    level = popcount(mask_p0) + popcount(mask_p1)
    mover = 'P0 (Red)' if level % 2 == 0 else 'P1 (Black)'

    print(f"\n{'='*50}")
    print(f"Level: {level}   Turn: {mover}   last_pos: {last_pos}")
    print(f"mask_p0: {mask_p0:#06x}  mask_p1: {mask_p1:#06x}")
    print()
    print(draw_board(mask_p0, mask_p1, last_pos))

    value, moves = lookup(mask_p0, mask_p1, last_pos, top_n=top_n)

    if value is None:
        print("\n  State not found in solution database.")
        return

    print(f"\nMinimax value: {value}  ({value_label(value)})")

    if not moves:
        print("  (terminal state, no moves)")
    else:
        print(f"\nAvailable moves ({len(moves)} shown):")
        print(f"  {'Pos':>3s}  {'Row,Col':>7s}  {'Tile':>4s}  {'Type':<18s}  {'Outcome'}")
        print(f"  {'---':>3s}  {'-------':>7s}  {'----':>4s}  {'----':<18s}  {'-------'}")
        for pos, cv in moves:
            r, c = pos // 4, pos % 4
            tile = int(INIT_TILES[pos])
            tname = tile_name(tile)
            print(f"  {pos:>3d}  ({r},{c})     {tile:>2d}  {tname:<18s}  {cv:+d} ({value_label(cv)})")

    if image:
        # get ALL moves (not limited by top_n) for the image annotation
        _, all_moves = lookup(mask_p0, mask_p1, last_pos)
        path = draw_board_image(mask_p0, mask_p1, last_pos, moves=all_moves)
        print(f"\nBoard image saved: {path}")

# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive(top_n=None, image=False):
    print("Okiya Query Interface")
    print("Commands:")
    print("  <pos>              - play move at position (0-15)")
    print("  <m0> <m1> <lp>     - jump to state (decimal or 0x hex)")
    print("  start              - reset to initial position")
    print("  img                - save board image")
    print("  img <path>         - save board image to custom path")
    print("  q                  - quit\n")

    cur_m0, cur_m1, cur_lp = 0, 0, 16  # initial state

    while True:
        show_state(cur_m0, cur_m1, cur_lp, top_n=top_n, image=image)
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() == 'q':
            break

        if raw.lower() == 'start':
            cur_m0, cur_m1, cur_lp = 0, 0, 16
            continue

        # img command
        if raw.lower().startswith('img'):
            parts = raw.split(None, 1)
            out = parts[1] if len(parts) > 1 else None
            _, all_moves = lookup(cur_m0, cur_m1, cur_lp)
            path = draw_board_image(cur_m0, cur_m1, cur_lp,
                                    moves=all_moves, output_path=out)
            print(f"Board image saved: {path}")
            continue

        parts = raw.replace(',', ' ').split()

        if len(parts) == 1:
            try:
                pos = int(parts[0], 0)
            except ValueError:
                print("Invalid input.")
                continue
            level = popcount(cur_m0) + popcount(cur_m1)
            if level % 2 == 0:
                cur_m0 |= (1 << pos)
            else:
                cur_m1 |= (1 << pos)
            cur_lp = pos
            continue

        if len(parts) == 3:
            try:
                m0 = int(parts[0], 0)
                m1 = int(parts[1], 0)
                lp = int(parts[2], 0)
            except ValueError:
                print("Invalid input.")
                continue
            cur_m0, cur_m1, cur_lp = m0, m1, lp
            continue

        print("Expected 1 or 3 values. Try again.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Query solved Okiya game')
    parser.add_argument('--state', type=str, default=None,
                        help='State as "mask_p0,mask_p1,last_pos" (decimal or hex)')
    parser.add_argument('--top-n', type=int, default=None,
                        help='Show only top N moves')
    parser.add_argument('--image', action='store_true',
                        help='Generate board image on every state display')
    parser.add_argument('--image-path', type=str, default=None,
                        help='Custom output path for the board image')
    args = parser.parse_args()

    if args.state:
        parts = args.state.replace(',', ' ').split()
        m0 = int(parts[0], 0)
        m1 = int(parts[1], 0)
        lp = int(parts[2], 0)
        show_state(m0, m1, lp, top_n=args.top_n, image=args.image)
        if args.image_path:
            _, all_moves = lookup(m0, m1, lp)
            path = draw_board_image(m0, m1, lp, moves=all_moves,
                                    output_path=args.image_path)
            print(f"Board image saved: {path}")
    else:
        interactive(top_n=args.top_n, image=args.image)


if __name__ == '__main__':
    main()
