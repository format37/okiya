#!/usr/bin/env python3
"""Pygame GUI for Okiya — interactive play with move visualization."""

import pygame
import subprocess
import threading
import json
import os
import random

from okiya_core import SolutionDB, GameState, popcount, value_label

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')
BOARD_PATH = os.path.join(SCRIPT_DIR, 'board.json')
FONT_PATH  = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"

# Reference layout (scale=1.0 at CELL=125)
DEFAULT_CELL    = 125
DEFAULT_PANEL_W = 280
BAR_H           = 50
MIN_PANEL_W     = 180

# Colors
BG_COLOR      = (235, 230, 215)
PANEL_BG      = (220, 215, 200)
GRID_COLOR    = (90, 70, 50)
P0_OVERLAY    = (210, 35, 35, 110)
P1_OVERLAY    = (35, 55, 195, 110)
P0_COLOR      = (210, 35, 35)
P1_COLOR      = (35, 55, 195)
GOLD          = (255, 200, 0)
BROWN         = (140, 80, 20)
TEXT_COLOR     = (40, 40, 40)
MUTED_COLOR   = (100, 100, 100)
BTN_COLOR     = (180, 175, 160)
BTN_HOVER     = (200, 195, 180)
BTN_DISABLED  = (160, 155, 145)
BTN_TEXT       = (30, 30, 30)
BTN_TEXT_DIS   = (120, 115, 105)
VALID_HIGHLIGHT = (255, 255, 100, 60)
GREEN_MOVE    = (20, 170, 20)
RED_MOVE      = (190, 30, 30)
WHITE         = (255, 255, 255)
BLACK         = (0, 0, 0)


class TileRenderer:
    """Loads hi-res tile images and draws cells with rescaling support."""

    def __init__(self):
        self._originals = {}
        for tile_type in range(16):
            # Images are indexed as weather*4+plant,
            # tile types are plant*4+weather
            img_idx = (tile_type % 4) * 4 + (tile_type // 4)
            path = os.path.join(IMAGES_DIR, f'higardentile{img_idx}.png')
            self._originals[tile_type] = pygame.image.load(path).convert_alpha()
        self._tiles = {}
        self._tile_px = 0

    def rescale(self, tile_px):
        if tile_px == self._tile_px:
            return
        self._tile_px = tile_px
        self._tiles = {}
        for tile_type, img in self._originals.items():
            self._tiles[tile_type] = pygame.transform.smoothscale(
                img, (tile_px, tile_px))

    def get_tile(self, tile_type):
        return self._tiles[tile_type]


class Button:
    """Simple clickable button."""

    def __init__(self, text, callback):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.text = text
        self.callback = callback
        self.font = None
        self.enabled = True

    def draw(self, surface, mx, my):
        hover = self.enabled and self.rect.collidepoint(mx, my)
        if not self.enabled:
            color = BTN_DISABLED
            tcolor = BTN_TEXT_DIS
        elif hover:
            color = BTN_HOVER
            tcolor = BTN_TEXT
        else:
            color = BTN_COLOR
            tcolor = BTN_TEXT
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, GRID_COLOR, self.rect, 1, border_radius=4)
        txt = self.font.render(self.text, True, tcolor)
        tx = self.rect.centerx - txt.get_width() // 2
        ty = self.rect.centery - txt.get_height() // 2
        surface.blit(txt, (tx, ty))

    def handle_click(self, mx, my):
        if self.enabled and self.rect.collidepoint(mx, my):
            self.callback()
            return True
        return False


class OkiyaGUI:
    """Main Pygame application."""

    def __init__(self):
        pygame.init()

        # Initial window size
        init_w = DEFAULT_CELL * 4 + DEFAULT_PANEL_W  # 780
        init_h = DEFAULT_CELL * 4 + BAR_H            # 550
        self.screen = pygame.display.set_mode(
            (init_w, init_h), pygame.RESIZABLE)
        pygame.display.set_caption("Okiya")
        self.clock = pygame.time.Clock()

        # Core
        self.db = SolutionDB()
        self.game = GameState()
        self.tiles = TileRenderer()

        # State
        self.solving = False
        self._reload_pending = False
        self._solve_error = None
        self.status_msg = ""
        self.status_timer = 0

        # Buttons (positions set by _recompute_layout)
        self.btn_undo = Button("Undo", self._on_undo)
        self.btn_redo = Button("Redo", self._on_redo)
        self.btn_reset = Button("Reset", self._on_reset)
        self.btn_shuffle = Button("Shuffle", self._on_shuffle)
        self.btn_solve = Button("Solve", self._on_solve)
        self.buttons = [self.btn_undo, self.btn_redo, self.btn_reset,
                        self.btn_shuffle, self.btn_solve]

        # Compute layout (sets all metrics, fonts, overlays, button rects)
        self._recompute_layout(init_w, init_h)

        # Cached move data for current state
        self._cached_value = None
        self._cached_moves = None
        self._cached_valid = None
        self._cached_win_rates = {}
        self._cached_state_key = None
        self._update_cache()

    # -- Layout --

    def _recompute_layout(self, win_w, win_h):
        """Recompute all layout metrics from window dimensions."""
        self.win_w = win_w
        self.win_h = win_h

        # Board is square; limited by height and width minus panel
        board_max_h = win_h - BAR_H
        board_max_w = win_w - MIN_PANEL_W
        board_side = max(min(board_max_h, board_max_w), 160)

        self.cell = board_side // 4
        self.board_px = self.cell * 4
        self.tile_px = max(self.cell - 10, 20)
        self.pad = (self.cell - self.tile_px) // 2
        self.panel_w = win_w - self.board_px
        self.scale = self.cell / DEFAULT_CELL

        # Fonts (scaled, with minimum sizes)
        def sz(base):
            return max(6, int(base * self.scale))

        self.font_big    = pygame.font.Font(FONT_PATH, sz(20))
        self.font_med    = pygame.font.Font(FONT_PATH, sz(16))
        self.font_sm     = pygame.font.Font(FONT_PATH, sz(13))
        self.font_tiny   = pygame.font.Font(FONT_PATH, sz(11))
        self.font_btn    = pygame.font.Font(FONT_PATH, sz(14))
        self.font_player = pygame.font.Font(FONT_PATH, sz(40))
        self.font_tile   = pygame.font.Font(FONT_PATH, sz(14))

        # Tile images
        self.tiles.rescale(self.tile_px)

        # Overlays
        self._p0_overlay = pygame.Surface(
            (self.tile_px, self.tile_px), pygame.SRCALPHA)
        self._p0_overlay.fill(P0_OVERLAY)
        self._p1_overlay = pygame.Surface(
            (self.tile_px, self.tile_px), pygame.SRCALPHA)
        self._p1_overlay.fill(P1_OVERLAY)
        self._valid_overlay = pygame.Surface(
            (self.tile_px, self.tile_px), pygame.SRCALPHA)
        self._valid_overlay.fill(VALID_HIGHLIGHT)

        # Button positions
        bw = max(50, int(72 * self.scale))
        bh = max(22, int(30 * self.scale))
        gap = max(4, int(6 * self.scale))
        by = self.board_px + (BAR_H - bh) // 2

        bx = 10
        for btn in [self.btn_undo, self.btn_redo, self.btn_reset]:
            btn.rect = pygame.Rect(bx, by, bw, bh)
            btn.font = self.font_btn
            bx += bw + gap

        bx2 = self.board_px + 20
        self.btn_shuffle.rect = pygame.Rect(bx2, by, bw + 10, bh)
        self.btn_shuffle.font = self.font_btn
        bx2 += bw + 10 + gap
        self.btn_solve.rect = pygame.Rect(bx2, by, bw, bh)
        self.btn_solve.font = self.font_btn

    # -- State cache --

    def _state_key(self):
        return (self.game.mask_p0, self.game.mask_p1, self.game.last_pos)

    def _update_cache(self):
        key = self._state_key()
        if key == self._cached_state_key:
            return
        self._cached_state_key = key
        m0, m1, lp = key
        self._cached_value, self._cached_moves = self.db.lookup(m0, m1, lp)
        self._cached_valid = self.db.get_valid_moves(m0, m1, lp)
        self._cached_win_rates = self.db.get_move_win_rates(m0, m1, lp)

    def _game_over_reason(self):
        """Return a game-over string, or None if game is ongoing."""
        m0, m1 = self.game.mask_p0, self.game.mask_p1
        lp = self.game.last_pos
        if self.db.check_win(m0):
            return "P0 (Red) wins!"
        if self.db.check_win(m1):
            return "P1 (Blue) wins!"
        if self.game.level > 0 and not self._cached_valid:
            # No valid moves — the stuck player loses
            mover = "P0" if self.game.is_p0_turn else "P1"
            # Debug: print state for verification
            empty = [p for p in range(16)
                     if not ((m0 | m1) & (1 << p))]
            tile_lp = int(self.db.init_tiles[lp]) if lp < 16 else -1
            print(f"\n=== {mover} STUCK ===")
            print(f"  m0={m0:#06x} m1={m1:#06x} lp={lp} "
                  f"(tile {tile_lp}: {self.db.tile_name(tile_lp)})")
            print(f"  Empty positions: {empty}")
            for ep in empty:
                t = int(self.db.init_tiles[ep])
                related = bool(self.db.pos_relation_masks[lp]
                               & (1 << ep))
                print(f"    pos {ep}: tile {t} "
                      f"({self.db.tile_name(t)}) "
                      f"{'RELATED' if related else 'not related'}")
            if self.game.is_p0_turn:
                return "P0 stuck \u2014 P1 (Blue) wins!"
            else:
                return "P1 stuck \u2014 P0 (Red) wins!"
        return None

    def _set_status(self, msg, duration_ms=3000):
        self.status_msg = msg
        self.status_timer = pygame.time.get_ticks() + duration_ms

    # -- Button callbacks --

    def _on_undo(self):
        self.game.undo()
        self._update_cache()

    def _on_redo(self):
        self.game.redo()
        self._update_cache()

    def _on_reset(self):
        self.game.reset()
        self._update_cache()
        self._set_status("Game reset.")

    def _on_shuffle(self):
        with open(BOARD_PATH) as f:
            board = json.load(f)
        random.shuffle(board['tiles'])
        with open(BOARD_PATH, 'w') as f:
            json.dump(board, f, indent=2)
        self.db.load_board()  # updates tiles + deletes solution
        self.game.reset()
        self._cached_state_key = None
        self._update_cache()
        self._set_status("Shuffled. No solution \u2014 press Solve.")

    def _on_solve(self):
        if self.solving:
            return
        self.solving = True
        self._solve_error = None
        self._set_status("Solving... (~20s)", duration_ms=60000)
        t = threading.Thread(target=self._solve_worker, daemon=True)
        t.start()

    def _solve_worker(self):
        try:
            result = subprocess.run(
                ['python3', os.path.join(SCRIPT_DIR, 'okiya_solve.py')],
                timeout=300)
            if result.returncode != 0:
                self._solve_error = f"Solver exited with code {result.returncode}"
        except Exception as e:
            self._solve_error = str(e)
        self._reload_pending = True

    # -- Drawing --

    def _draw_glow_text(self, surface, font, text, x, y,
                        fg=BLACK, glow=WHITE):
        """Render text with a white glow (outline) behind black text."""
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                g = font.render(text, True, glow)
                surface.blit(g, (x + dx, y + dy))
        t = font.render(text, True, fg)
        surface.blit(t, (x, y))

    def _draw_board(self):
        screen = self.screen
        m0, m1, lp = self.game.mask_p0, self.game.mask_p1, self.game.last_pos
        init_tiles = self.db.init_tiles
        game_over = self._game_over_reason()

        cell = self.cell
        tile_px = self.tile_px
        pad = self.pad
        board_px = self.board_px
        s = self.scale

        # Build move map
        move_map = {}
        if self._cached_moves and not game_over:
            for pos, val in self._cached_moves:
                move_map[pos] = val

        for pos in range(16):
            row, col = pos // 4, pos % 4
            tile_type = int(init_tiles[pos])
            x0 = col * cell
            y0 = row * cell

            # Tile image
            screen.blit(self.tiles.get_tile(tile_type), (x0 + pad, y0 + pad))

            is_p0 = bool(m0 & (1 << pos))
            is_p1 = bool(m1 & (1 << pos))
            is_last = (pos == lp) and (lp < 16)
            claimed = is_p0 or is_p1

            # Player color overlay
            if claimed:
                ov = self._p0_overlay if is_p0 else self._p1_overlay
                screen.blit(ov, (x0 + pad, y0 + pad))

                # Player letter
                letter = 'R' if is_p0 else 'B'
                cx, cy = x0 + cell // 2, y0 + cell // 2
                # Shadow
                shadow = self.font_player.render(letter, True, (0, 0, 0))
                soff = max(1, int(2 * s))
                screen.blit(shadow, (cx - shadow.get_width() // 2 + soff,
                                     cy - shadow.get_height() // 2 + soff))
                txt = self.font_player.render(letter, True, WHITE)
                screen.blit(txt, (cx - txt.get_width() // 2,
                                  cy - txt.get_height() // 2))

            # Valid move highlight
            if not claimed and not game_over and pos in self._cached_valid:
                screen.blit(self._valid_overlay, (x0 + pad, y0 + pad))

            # Gold border on last move
            if is_last:
                for i in range(3):
                    pygame.draw.rect(screen, GOLD,
                        (x0 + pad - i - 1, y0 + pad - i - 1,
                         tile_px + 2 * (i + 1), tile_px + 2 * (i + 1)), 1)

            # Position number (top-left) — white glow + black text
            self._draw_glow_text(screen, self.font_tile, str(pos),
                                 x0 + pad + 3, y0 + pad + 1)

            # Tile type id (top-right) — white glow + black text
            tid_surf = self.font_tile.render(f"t{tile_type}", True, BLACK)
            tid_x = x0 + pad + tile_px - tid_surf.get_width() - 3
            tid_y = y0 + pad + 1
            self._draw_glow_text(screen, self.font_tile, f"t{tile_type}",
                                 tid_x, tid_y)

            # Tile attributes (bottom, unclaimed only) — white glow + black
            if not claimed:
                a_short = self.db.tile_a_names[tile_type // 4][:3]
                b_short = self.db.tile_b_names[tile_type % 4][:3]
                attr_str = f"{a_short}+{b_short}"
                attr_measure = self.font_tile.render(attr_str, True, BLACK)
                ax = x0 + pad + (tile_px - attr_measure.get_width()) // 2
                ay = y0 + pad + tile_px - attr_measure.get_height() - 3
                self._draw_glow_text(screen, self.font_tile, attr_str,
                                     ax, ay)

            # Move quality circle — win percentage for current mover
            if not claimed and pos in self._cached_win_rates:
                rate = self._cached_win_rates[pos]
                pct = int(round(rate * 100))
                cx, cy = x0 + cell // 2, y0 + cell // 2
                r = max(8, int(19 * s))
                # Color: green if >55%, red if <45%, gray between
                if rate > 0.55:
                    fill_c = GREEN_MOVE
                elif rate < 0.45:
                    fill_c = RED_MOVE
                else:
                    fill_c = MUTED_COLOR
                label = f"{pct}%"
                pygame.draw.circle(screen, fill_c, (cx, cy), r)
                pygame.draw.circle(screen, WHITE, (cx, cy), r, 2)
                ltxt = self.font_tiny.render(label, True, WHITE)
                screen.blit(ltxt, (cx - ltxt.get_width() // 2,
                                   cy - ltxt.get_height() // 2))

        # Grid lines
        for i in range(5):
            pygame.draw.line(screen, GRID_COLOR,
                (i * cell, 0), (i * cell, board_px), 2)
            pygame.draw.line(screen, GRID_COLOR,
                (0, i * cell), (board_px, i * cell), 2)

    def _draw_panel(self):
        screen = self.screen
        board_px = self.board_px
        panel_w = self.panel_w
        s = self.scale
        px = board_px + int(10 * s)
        py = int(15 * s)

        # Panel background
        pygame.draw.rect(screen, PANEL_BG,
            (board_px, 0, panel_w, board_px))
        pygame.draw.line(screen, GRID_COLOR,
            (board_px, 0), (board_px, board_px), 2)

        game_over = self._game_over_reason()
        line_h = int(30 * s)
        line_sm = int(22 * s)

        # Turn
        if game_over:
            turn_txt = self.font_med.render(game_over, True, BROWN)
            screen.blit(turn_txt, (px, py))
        else:
            turn_str = "P0 (Red)" if self.game.is_p0_turn else "P1 (Blue)"
            turn_color = P0_COLOR if self.game.is_p0_turn else P1_COLOR
            label = self.font_med.render("Turn: ", True, TEXT_COLOR)
            screen.blit(label, (px, py))
            turn_txt = self.font_med.render(turn_str, True, turn_color)
            screen.blit(turn_txt, (px + label.get_width(), py))
        py += line_h

        # Level
        level_txt = self.font_med.render(
            f"Level: {self.game.level}", True, TEXT_COLOR)
        screen.blit(level_txt, (px, py))
        py += line_h

        # Value (relative to current mover)
        has_sol = self.db.has_solution
        if has_sol and self._cached_value is not None:
            val = self._cached_value
            p0t = self.game.is_p0_turn
            mover_wins = (val == 1 and p0t) or (val == -1 and not p0t)
            mover_loses = (val == -1 and p0t) or (val == 1 and not p0t)
            if mover_wins:
                vstr = "Win"
                vcolor = GREEN_MOVE
            elif mover_loses:
                vstr = "Lose"
                vcolor = RED_MOVE
            else:
                vstr = "Draw"
                vcolor = MUTED_COLOR
            vlbl = self.font_med.render("Value: ", True, TEXT_COLOR)
            screen.blit(vlbl, (px, py))
            val_txt = self.font_med.render(vstr, True, vcolor)
            screen.blit(val_txt, (px + vlbl.get_width(), py))
        elif not has_sol:
            screen.blit(self.font_sm.render(
                "No solution", True, MUTED_COLOR), (px, py))
        py += line_h

        # Last pos
        lp = self.game.last_pos
        if lp < 16:
            tile = int(self.db.init_tiles[lp])
            lp_str = f"pos {lp} (t{tile})"
        else:
            lp_str = "none"
        lp_label = self.font_med.render("Last: ", True, TEXT_COLOR)
        screen.blit(lp_label, (px, py))
        lp_txt = self.font_med.render(lp_str, True, TEXT_COLOR)
        screen.blit(lp_txt, (px + lp_label.get_width(), py))
        py += int(25 * s)

        # State masks (for debugging / verification)
        m0, m1 = self.game.mask_p0, self.game.mask_p1
        state_str = f"R:{m0:#06x} B:{m1:#06x}"
        screen.blit(self.font_tiny.render(
            state_str, True, MUTED_COLOR), (px, py))
        py += line_sm

        # Move legend
        if not game_over and self._cached_win_rates:
            mover = "Red" if self.game.is_p0_turn else "Blue"
            screen.blit(self.font_sm.render(
                f"Win % for {mover}:", True, TEXT_COLOR), (px, py))
            py += line_sm
            cr = max(4, int(8 * s))
            pygame.draw.circle(screen, GREEN_MOVE,
                (px + cr + 2, py + cr), cr)
            screen.blit(self.font_sm.render(
                ">55%  favorable", True, GREEN_MOVE),
                (px + cr * 2 + 8, py))
            py += line_sm
            pygame.draw.circle(screen, RED_MOVE,
                (px + cr + 2, py + cr), cr)
            screen.blit(self.font_sm.render(
                "<45%  unfavorable", True, RED_MOVE),
                (px + cr * 2 + 8, py))
            py += line_h

        # Moves list sorted by win rate
        if not game_over and self._cached_win_rates:
            py += int(5 * s)
            screen.blit(self.font_sm.render(
                "Moves (best %):", True, TEXT_COLOR), (px, py))
            py += int(20 * s)
            line_entry = int(16 * s)
            # Sort moves by win rate descending
            sorted_moves = sorted(
                self._cached_win_rates.items(),
                key=lambda x: x[1], reverse=True)
            for pos, rate in sorted_moves[:8]:
                r, c = pos // 4, pos % 4
                tile = int(self.db.init_tiles[pos])
                pct = int(round(rate * 100))
                if rate > 0.55:
                    color = GREEN_MOVE
                elif rate < 0.45:
                    color = RED_MOVE
                else:
                    color = MUTED_COLOR
                line = f"p{pos}({r},{c}) t{tile} {pct}%"
                screen.blit(self.font_tiny.render(line, True, color), (px, py))
                py += line_entry
                if py > board_px - int(20 * s):
                    break

        # Status message
        if self.status_msg and pygame.time.get_ticks() < self.status_timer:
            py = board_px - int(25 * s)
            screen.blit(self.font_sm.render(
                self.status_msg, True, BROWN), (px, py))

    def _draw_bar(self):
        board_px = self.board_px
        # Button bar background
        pygame.draw.rect(self.screen, BG_COLOR,
            (0, board_px, self.win_w, BAR_H))
        pygame.draw.line(self.screen, GRID_COLOR,
            (0, board_px), (self.win_w, board_px), 2)

        mx, my = pygame.mouse.get_pos()

        # Update button enabled states
        self.btn_undo.enabled = self.game.can_undo() and not self.solving
        self.btn_redo.enabled = self.game.can_redo() and not self.solving
        self.btn_reset.enabled = (self.game.level > 0) and not self.solving
        self.btn_shuffle.enabled = not self.solving
        self.btn_solve.enabled = not self.solving

        for btn in self.buttons:
            btn.draw(self.screen, mx, my)

        # Solving indicator
        if self.solving:
            dots = "." * ((pygame.time.get_ticks() // 500) % 4)
            stxt = self.font_btn.render(f"Solving{dots}", True, BROWN)
            sx = self.btn_solve.rect.right + 10
            sy = self.btn_solve.rect.centery - stxt.get_height() // 2
            self.screen.blit(stxt, (sx, sy))

    def _handle_board_click(self, mx, my):
        if self.solving:
            return
        if mx >= self.board_px or my >= self.board_px:
            return
        game_over = self._game_over_reason()
        if game_over:
            return

        col = mx // self.cell
        row = my // self.cell
        pos = row * 4 + col
        if pos > 15:
            return

        if pos in self._cached_valid:
            self.game.play_move(pos)
            self._update_cache()

    def run(self):
        running = True
        while running:
            # Check reload flag (from solve thread)
            if self._reload_pending:
                self._reload_pending = False
                self.solving = False
                if self._solve_error:
                    self._set_status(
                        f"Solve error: {self._solve_error[:60]}", 8000)
                else:
                    self.db.reload()
                    self._cached_state_key = None
                    self._update_cache()
                    self._set_status("Solution loaded!", 5000)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    w, h = event.w, event.h
                    # Enforce minimum size
                    w = max(w, 520)
                    h = max(h, 350)
                    self.screen = pygame.display.set_mode(
                        (w, h), pygame.RESIZABLE)
                    self._recompute_layout(w, h)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    # Check buttons first
                    clicked = False
                    for btn in self.buttons:
                        if btn.handle_click(mx, my):
                            clicked = True
                            break
                    if not clicked:
                        self._handle_board_click(mx, my)
                elif event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_CTRL:
                        if event.key == pygame.K_z:
                            self._on_undo()
                        elif event.key == pygame.K_y:
                            self._on_redo()
                        elif event.key == pygame.K_r:
                            self._on_reset()

            # Draw
            self.screen.fill(BG_COLOR)
            self._draw_board()
            self._draw_panel()
            self._draw_bar()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


if __name__ == '__main__':
    OkiyaGUI().run()
