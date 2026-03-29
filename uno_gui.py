"""
=============================================================
  UNO GAME AI  —  Pygame GUI
  AI 2002 Assignment #2
  
  Run:  python uno_gui.py
  Requires: pip install pygame
=============================================================
"""

import pygame
import random
import copy
import sys
import time
import math
from collections import defaultdict

# ─────────────────────────────────────────────
#  CONSTANTS & THEME
# ─────────────────────────────────────────────

W, H = 1200, 750

# Deep dark card-table theme
BG_DARK      = (10,  22,  18)
BG_TABLE     = (18,  42,  32)
FELT_GREEN   = (22,  58,  40)
GOLD         = (212, 175,  55)
GOLD_DIM     = (140, 110,  30)
WHITE        = (245, 242, 235)
OFF_WHITE    = (200, 195, 180)
GRAY_DARK    = ( 40,  40,  40)
GRAY_MID     = ( 80,  80,  80)
SHADOW       = (  0,   0,   0, 120)

# UNO card colors
CARD_COLORS = {
    'Red'    : (214,  48,  49),
    'Blue'   : ( 45, 100, 200),
    'Green'  : ( 39, 174,  96),
    'Yellow' : (241, 196,  15),
}
CARD_BG      = (248, 244, 230)
CARD_W, CARD_H = 72, 108
CARD_RADIUS  = 10

# Player strip colors
PLAYER_COLORS = [
    (214,  48,  49),   # P1 — red
    ( 45, 100, 200),   # P2 — blue
    ( 39, 174,  96),   # P3 — green
]

PLAYER_LABELS = [
    "P1 · Minimax  (Defensive)",
    "P2 · Expectimax (Offensive)",
    "P3 · Minimax  (Simulation)",
]

# Simulation step delay (seconds)
STEP_DELAY = 1.2

# ─────────────────────────────────────────────
#  CARD & GAME ENGINE  (same logic as notebook)
# ─────────────────────────────────────────────

COLORS_LIST = ['Red', 'Blue', 'Green', 'Yellow']
NUMBERS     = list(range(0, 10))

class Card:
    def __init__(self, color, value):
        self.color = color
        self.value = value

    def __repr__(self):
        return f"{self.color} {self.value}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.color == other.color and self.value == other.value

    def is_skip(self):
        return self.value == 'Skip'

    def label(self):
        return "⊘" if self.is_skip() else str(self.value)

    def __hash__(self):
        return hash((self.color, self.value))


def generate_deck():
    deck = []
    for color in COLORS_LIST:
        for number in NUMBERS:
            deck.append(Card(color, number))
    for color in COLORS_LIST:
        deck.append(Card(color, 'Skip'))
        deck.append(Card(color, 'Skip'))
    random.shuffle(deck)
    return deck


def create_initial_state():
    deck = generate_deck()
    p1   = [deck.pop() for _ in range(5)]
    p2   = [deck.pop() for _ in range(5)]
    p3   = [deck.pop() for _ in range(5)]
    top  = deck.pop()
    while top.is_skip():
        deck.insert(0, top)
        top = deck.pop()
    return {
        'p1_hand': p1, 'p2_hand': p2, 'p3_hand': p3,
        'top_card': top, 'deck': deck,
        'current_player': 0, 'skip_next': False, 'winner': None
    }


def get_hand_key(idx):
    return f'p{idx+1}_hand'


def get_valid_moves(hand, top_card):
    valid = [c for c in hand if c.color == top_card.color or c.value == top_card.value]
    valid.append('Draw')
    return valid


def apply_move(state, move, player_index):
    ns       = copy.deepcopy(state)
    hand_key = get_hand_key(player_index)
    if move == 'Draw':
        if ns['deck']:
            ns[hand_key].append(ns['deck'].pop())
    else:
        ns[hand_key].remove(move)
        ns['top_card'] = move
        if move.is_skip():
            ns['skip_next'] = True
    if len(ns[hand_key]) == 0:
        ns['winner'] = f'Player {player_index + 1}'
    next_p = (player_index + 1) % 3
    if ns['skip_next'] and move != 'Draw' and move.is_skip():
        ns['skip_next'] = False
        next_p = (next_p + 1) % 3
    ns['current_player'] = next_p
    return ns


# ─── Evaluation ───
DEFENSIVE_W = {'base': 50, 'c_ai': -6, 'c_opp': 2,  'skip': 4}
OFFENSIVE_W = {'base': 50, 'c_ai': -5, 'c_opp': 3,  'skip': 2}

def evaluate(state, player_index, weights):
    hk    = get_hand_key(player_index)
    hand  = state[hk]
    c_ai  = len(hand)
    opps  = [len(state[get_hand_key(i)]) for i in range(3) if i != player_index]
    c_opp = sum(opps) / len(opps)
    s     = sum(1 for c in hand if c.is_skip())
    return round(weights['base'] + weights['c_ai']*c_ai + weights['c_opp']*c_opp + weights['skip']*s, 2)


def is_terminal(state):
    return state['winner'] is not None


# ─── Minimax ───
def minimax(state, depth, is_max, ai_idx, alpha=float('-inf'), beta=float('inf')):
    if depth == 0 or is_terminal(state):
        return evaluate(state, ai_idx, DEFENSIVE_W), None
    cp   = state['current_player']
    hk   = get_hand_key(cp)
    moves = get_valid_moves(state[hk], state['top_card'])
    if is_max:
        best_s, best_m = float('-inf'), None
        for m in moves:
            cs = apply_move(state, m, cp)
            s, _ = minimax(cs, depth-1, False, ai_idx, alpha, beta)
            if s > best_s: best_s, best_m = s, m
            alpha = max(alpha, best_s)
            if beta <= alpha: break
        return best_s, best_m
    else:
        best_s, best_m = float('inf'), None
        for m in moves:
            cs = apply_move(state, m, cp)
            nim = (cs['current_player'] == ai_idx)
            s, _ = minimax(cs, depth-1, nim, ai_idx, alpha, beta)
            if s < best_s: best_s, best_m = s, m
            beta = min(beta, best_s)
            if beta <= alpha: break
        return best_s, best_m


def best_minimax(state, player_idx):
    hk    = get_hand_key(player_idx)
    moves = get_valid_moves(state[hk], state['top_card'])
    best_s, best_m = float('-inf'), None
    for m in moves:
        cs = apply_move(state, m, player_idx)
        nim = (cs['current_player'] == player_idx)
        s, _ = minimax(cs, 2, nim, player_idx)
        if s > best_s: best_s, best_m = s, m
    return best_m


# ─── Expectimax ───
def get_draw_probs(deck):
    if not deck: return {}
    total  = len(deck)
    counts = defaultdict(int)
    for c in deck: counts[(c.color, c.value)] += 1
    probs, seen = {}, set()
    for c in deck:
        k = (c.color, c.value)
        if k not in seen:
            probs[c] = counts[k] / total
            seen.add(k)
    return probs


def expectimax(state, depth, node_type, ai_idx):
    if depth == 0 or is_terminal(state):
        return evaluate(state, ai_idx, OFFENSIVE_W), None
    cp  = state['current_player']
    hk  = get_hand_key(cp)
    if node_type == 'max':
        moves = get_valid_moves(state[hk], state['top_card'])
        best_s, best_m = float('-inf'), None
        for m in moves:
            if m == 'Draw':
                s, _ = expectimax(state, depth-1, 'chance', ai_idx)
            else:
                cs = apply_move(state, m, cp)
                s, _ = expectimax(cs, depth-1, 'opponent', ai_idx)
            if s > best_s: best_s, best_m = s, m
        return best_s, best_m
    elif node_type == 'chance':
        probs = get_draw_probs(state['deck'])
        if not probs: return evaluate(state, ai_idx, OFFENSIVE_W), None
        exp = 0.0
        for card, prob in probs.items():
            cs = copy.deepcopy(state)
            if card in cs['deck']: cs['deck'].remove(card)
            cs[hk].append(card)
            cs['current_player'] = (cp + 1) % 3
            s, _ = expectimax(cs, depth-1, 'opponent', ai_idx)
            exp += prob * s
        return round(exp, 2), 'Draw'
    else:
        moves = get_valid_moves(state[hk], state['top_card'])
        m     = random.choice(moves)
        cs    = apply_move(state, m, cp)
        nnt   = 'max' if cs['current_player'] == ai_idx else 'opponent'
        return expectimax(cs, depth-1, nnt, ai_idx)


def best_expectimax(state, player_idx):
    hk    = get_hand_key(player_idx)
    moves = get_valid_moves(state[hk], state['top_card'])
    best_s, best_m = float('-inf'), None
    for m in moves:
        if m == 'Draw':
            s, _ = expectimax(state, 2, 'chance', player_idx)
        else:
            cs = apply_move(state, m, player_idx)
            s, _ = expectimax(cs, 2, 'opponent', player_idx)
        if s > best_s: best_s, best_m = s, m
    return best_m


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_rounded_rect(surf, color, rect, radius, border=0, border_color=None):
    """Draw a filled rounded rectangle, with optional border."""
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surf, border_color, rect, border, border_radius=radius)


def draw_card(surf, card, x, y, selected=False, facedown=False, small=False):
    """
    Renders a UNO card at (x, y).
    selected  — draws a gold glow beneath it
    facedown  — renders card back pattern
    small     — renders at 60% scale for opponent hands
    """
    cw = int(CARD_W * 0.60) if small else CARD_W
    ch = int(CARD_H * 0.60) if small else CARD_H
    r  = 6 if small else CARD_RADIUS

    # Glow / selection halo
    if selected:
        glow_rect = pygame.Rect(x - 4, y - 4, cw + 8, ch + 8)
        pygame.draw.rect(surf, GOLD, glow_rect, border_radius=r+3)

    shadow_rect = pygame.Rect(x + 3, y + 3, cw, ch)
    pygame.draw.rect(surf, (0, 0, 0, 80), shadow_rect, border_radius=r)

    rect = pygame.Rect(x, y, cw, ch)

    if facedown:
        # Card back: dark with gold cross-hatch pattern
        draw_rounded_rect(surf, (30, 30, 60), rect, r)
        draw_rounded_rect(surf, GOLD_DIM, rect, r, border=2, border_color=GOLD_DIM)
        # Cross-hatch lines
        for i in range(0, cw, 8):
            pygame.draw.line(surf, (50, 50, 90), (x+i, y), (x, y+i), 1)
        for i in range(0, ch, 8):
            pygame.draw.line(surf, (50, 50, 90), (x, y+i), (x+i, y), 1)
        return

    color = CARD_COLORS.get(card.color, (150, 150, 150))

    # Card body
    draw_rounded_rect(surf, CARD_BG, rect, r)
    draw_rounded_rect(surf, CARD_BG, rect, r, border=2, border_color=color)

    # Colored oval in center
    inner_margin = 5 if small else 8
    inner = pygame.Rect(x + inner_margin, y + inner_margin,
                        cw - inner_margin*2, ch - inner_margin*2)
    pygame.draw.ellipse(surf, color, inner)

    # Value text
    font_size = 18 if small else 28
    font      = pygame.font.SysFont('consolas', font_size, bold=True)
    label     = card.label()
    txt       = font.render(label, True, WHITE)
    tr        = txt.get_rect(center=(x + cw//2, y + ch//2))
    surf.blit(txt, tr)

    # Corner mini-labels
    tiny = pygame.font.SysFont('consolas', 10 if small else 14, bold=True)
    tl   = tiny.render(label, True, color)
    surf.blit(tl, (x + 4, y + 3))


def draw_deck_pile(surf, x, y, count):
    """Draw a stack of face-down cards representing the draw pile."""
    # Stack shadow cards
    for i in range(min(4, count)):
        draw_card(surf, None, x - i*2, y - i*2, facedown=True)
    # Count badge
    font  = pygame.font.SysFont('consolas', 14, bold=True)
    label = font.render(f"{count}", True, GOLD)
    surf.blit(label, (x + CARD_W//2 - label.get_width()//2, y + CARD_H + 6))


def draw_panel(surf, rect, title=None, alpha=200):
    """Semi-transparent panel with optional title."""
    panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    panel.fill((10, 30, 20, alpha))
    surf.blit(panel, (rect.x, rect.y))
    pygame.draw.rect(surf, GOLD_DIM, rect, 1, border_radius=8)
    if title:
        font = pygame.font.SysFont('consolas', 13, bold=True)
        t    = font.render(title, True, GOLD)
        surf.blit(t, (rect.x + 10, rect.y + 6))


def pulse_alpha(t, speed=2.5, lo=140, hi=255):
    """Returns a pulsing alpha value for animations."""
    return int(lo + (hi - lo) * (0.5 + 0.5 * math.sin(t * speed)))


# ─────────────────────────────────────────────
#  GAME STATE MACHINE
# ─────────────────────────────────────────────

class GameScreen:
    """Handles menu screen."""

    def __init__(self, surf):
        self.surf   = surf
        self.choice = None  # 'simulation' or 'manual'

    def draw(self, t):
        s = self.surf
        s.fill(BG_DARK)

        # Felt table circle
        pygame.draw.circle(s, FELT_GREEN, (W//2, H//2), 320)
        pygame.draw.circle(s, (26, 70, 50), (W//2, H//2), 320, 3)

        # Title
        big  = pygame.font.SysFont('impact', 88)
        sub  = pygame.font.SysFont('consolas', 18)

        alpha = pulse_alpha(t, speed=1.8)
        title = big.render("UNO", True, GOLD)
        title.set_alpha(alpha)
        s.blit(title, title.get_rect(center=(W//2, H//2 - 120)))

        tag = sub.render("A I  ·  A D V E R S A R I A L  S E A R C H", True, OFF_WHITE)
        s.blit(tag, tag.get_rect(center=(W//2, H//2 - 40)))

        # Buttons
        self._btn(s, "▶  SIMULATION MODE", (W//2, H//2 + 50),  'simulation', t)
        self._btn(s, "✎  MANUAL MODE",     (W//2, H//2 + 120), 'manual',     t)

        credit = pygame.font.SysFont('consolas', 12).render(
            "AI 2002  ·  Assignment #2  ·  Minimax + Expectimax", True, GRAY_MID)
        s.blit(credit, credit.get_rect(center=(W//2, H - 24)))

    def _btn(self, s, text, center, key, t):
        font  = pygame.font.SysFont('consolas', 20, bold=True)
        mx, my = pygame.mouse.get_pos()
        bw, bh = 320, 52
        bx, by = center[0] - bw//2, center[1] - bh//2
        rect   = pygame.Rect(bx, by, bw, bh)
        hover  = rect.collidepoint(mx, my)
        bg     = GOLD if hover else GRAY_DARK
        fc     = BG_DARK if hover else GOLD
        draw_rounded_rect(s, bg, rect, 8)
        pygame.draw.rect(s, GOLD_DIM, rect, 1, border_radius=8)
        lbl = font.render(text, True, fc)
        s.blit(lbl, lbl.get_rect(center=rect.center))

    def handle_click(self, pos):
        bw, bh = 320, 52
        for text, center, key in [
            ("sim", (W//2, H//2 + 50),  'simulation'),
            ("man", (W//2, H//2 + 120), 'manual'),
        ]:
            bx, by = center[0] - bw//2, center[1] - bh//2
            if pygame.Rect(bx, by, bw, bh).collidepoint(pos):
                self.choice = key


class UnoGame:
    """
    Main game screen.
    Handles both simulation and manual modes.
    """

    def __init__(self, surf, mode):
        self.surf         = surf
        self.mode         = mode           # 'simulation' or 'manual'
        self.state        = create_initial_state()
        self.log          = []             # Event log messages
        self.selected_idx = None           # Index of selected card in P3 hand (manual)
        self.ai_thinking  = False
        self.done         = False
        self.restart      = False
        self.last_step    = time.time()
        self.anim_t       = 0.0
        self.played_card_anim = None       # (card, x, y, timer) for played card flash
        self.turn_count   = 0

        self.add_log("Game started!", GOLD)
        self.add_log(f"Mode: {mode.upper()}", OFF_WHITE)
        self.add_log(f"Top card: {self.state['top_card']}", OFF_WHITE)

    def add_log(self, msg, color=OFF_WHITE):
        self.log.append((msg, color))
        if len(self.log) > 14:
            self.log.pop(0)

    # ──── AI step ────────────────────────────────

    def ai_step(self):
        """Compute and apply one AI move."""
        cp = self.state['current_player']
        if cp == 0:
            move = best_minimax(self.state, 0)
            algo = "Minimax"
        elif cp == 1:
            move = best_expectimax(self.state, 1)
            algo = "Expectimax"
        else:
            move = best_minimax(self.state, 2)
            algo = "Minimax"

        self.turn_count += 1
        pcolor = PLAYER_COLORS[cp]
        self.add_log(f"P{cp+1} [{algo}]: plays {move}", pcolor)
        self.state = apply_move(self.state, move, cp)

        if self.state['winner']:
            self.done = True
            self.add_log(f"🏆  {self.state['winner']} WINS!", GOLD)

    def manual_play(self, card_or_draw):
        """Apply a manual move for P3."""
        cp = self.state['current_player']
        self.turn_count += 1
        self.add_log(f"P3 [You]: plays {card_or_draw}", PLAYER_COLORS[2])
        self.state = apply_move(self.state, card_or_draw, cp)
        self.selected_idx = None

        if self.state['winner']:
            self.done = True
            self.add_log(f"🏆  {self.state['winner']} WINS!", GOLD)

    # ──── Update ─────────────────────────────────

    def update(self, dt):
        self.anim_t += dt
        if self.done:
            return

        cp = self.state['current_player']

        if self.mode == 'simulation':
            # Auto-step with delay
            if time.time() - self.last_step > STEP_DELAY:
                self.ai_step()
                self.last_step = time.time()

        else:  # manual
            if cp != 2:
                # AI turn
                if time.time() - self.last_step > STEP_DELAY:
                    self.ai_step()
                    self.last_step = time.time()
            # else: wait for user click

    # ──── Draw ───────────────────────────────────

    def draw(self):
        s = self.surf
        s.fill(BG_DARK)

        # ── Felt table oval ──
        pygame.draw.ellipse(s, FELT_GREEN, pygame.Rect(W//2 - 420, H//2 - 260, 840, 520))
        pygame.draw.ellipse(s, (26, 70, 50), pygame.Rect(W//2 - 420, H//2 - 260, 840, 520), 3)

        self._draw_center()
        self._draw_player_zones()
        self._draw_log_panel()
        self._draw_hud()

        if self.done:
            self._draw_winner_overlay()

    def _draw_center(self):
        s = self.surf
        cx, cy = W//2, H//2

        # Deck pile
        deck_x = cx - CARD_W - 50
        draw_deck_pile(s, deck_x, cy - CARD_H//2, len(self.state['deck']))

        # Deck label
        font = pygame.font.SysFont('consolas', 12)
        lbl  = font.render("DRAW", True, GOLD_DIM)
        s.blit(lbl, (deck_x + CARD_W//2 - lbl.get_width()//2, cy - CARD_H//2 - 18))

        # Top card with pulsing glow
        top   = self.state['top_card']
        top_x = cx + 20
        top_y = cy - CARD_H//2

        # Glow ring
        alpha  = pulse_alpha(self.anim_t, speed=2.0, lo=60, hi=180)
        gcolor = CARD_COLORS.get(top.color, GOLD)
        gsurf  = pygame.Surface((CARD_W + 24, CARD_H + 24), pygame.SRCALPHA)
        pygame.draw.rect(gsurf, (*gcolor, alpha),
                         (0, 0, CARD_W + 24, CARD_H + 24), border_radius=14)
        s.blit(gsurf, (top_x - 12, top_y - 12))

        draw_card(s, top, top_x, top_y)

        lbl2 = font = pygame.font.SysFont('consolas', 12)
        lbl2 = lbl2.render("TOP CARD", True, GOLD_DIM)
        s.blit(lbl2, (top_x + CARD_W//2 - lbl2.get_width()//2, top_y - 18))

        # Turn arrow indicator
        cp     = self.state['current_player']
        arrow  = pygame.font.SysFont('consolas', 22, bold=True)
        turn_s = arrow.render(f"▸  Player {cp+1}'s Turn", True, PLAYER_COLORS[cp])
        s.blit(turn_s, turn_s.get_rect(center=(cx, cy - CARD_H//2 - 40)))

    def _draw_player_zones(self):
        s   = self.surf
        cp  = self.state['current_player']
        t   = self.anim_t

        # ── P1 — left side ──
        self._draw_ai_hand(
            hand      = self.state['p1_hand'],
            label     = PLAYER_LABELS[0],
            cx        = 115,
            cy        = H//2,
            vertical  = True,
            player_idx= 0,
            active    = (cp == 0)
        )

        # ── P2 — top ──
        self._draw_ai_hand(
            hand      = self.state['p2_hand'],
            label     = PLAYER_LABELS[1],
            cx        = W//2,
            cy        = 60,
            vertical  = False,
            player_idx= 1,
            active    = (cp == 1)
        )

        # ── P3 — bottom ──
        self._draw_p3_hand(active=(cp == 2))

    def _draw_ai_hand(self, hand, label, cx, cy, vertical, player_idx, active):
        s     = self.surf
        color = PLAYER_COLORS[player_idx]
        font  = pygame.font.SysFont('consolas', 13, bold=True)

        cw = int(CARD_W * 0.60)
        ch = int(CARD_H * 0.60)
        gap = 6

        n = len(hand)

        if vertical:
            total_h = n * (ch + gap)
            start_y = cy - total_h // 2
            for i, card in enumerate(hand):
                x = cx - cw // 2
                y = start_y + i * (ch + gap)
                draw_card(s, card, x, y, facedown=True, small=True)

            # Label strip
            lbl = font.render(label, True, color)
            s.blit(lbl, (cx - lbl.get_width()//2, cy - total_h//2 - 22))
            count = pygame.font.SysFont('consolas', 12).render(
                f"{n} cards", True, OFF_WHITE)
            s.blit(count, (cx - count.get_width()//2, cy + total_h//2 + 6))
        else:
            total_w = n * (cw + gap)
            start_x = cx - total_w // 2
            for i, card in enumerate(hand):
                x = start_x + i * (cw + gap)
                y = cy
                draw_card(s, card, x, y, facedown=True, small=True)

            lbl = font.render(label, True, color)
            s.blit(lbl, (cx - lbl.get_width()//2, cy - 20))
            count = pygame.font.SysFont('consolas', 12).render(
                f"{n} cards", True, OFF_WHITE)
            s.blit(count, (cx + total_w//2 + 8, cy + ch//2 - 6))

        # Active player glow badge
        if active:
            alpha = pulse_alpha(self.anim_t, speed=3.0, lo=100, hi=220)
            badge = pygame.Surface((80, 22), pygame.SRCALPHA)
            badge.fill((*color, alpha))
            if vertical:
                s.blit(badge, (cx - 40, cy - 6))
            else:
                s.blit(badge, (cx - 40, cy + ch + 4))

    def _draw_p3_hand(self, active):
        s     = self.surf
        hand  = self.state['p3_hand']
        cp    = self.state['current_player']
        font  = pygame.font.SysFont('consolas', 14, bold=True)

        n     = len(hand)
        gap   = 10
        total_w = n * (CARD_W + gap)
        start_x = W//2 - total_w // 2
        y       = H - CARD_H - 52

        # Panel behind hand
        panel_rect = pygame.Rect(start_x - 14, y - 14,
                                 total_w + 28, CARD_H + 48)
        draw_panel(self.surf, panel_rect, alpha=160)

        # Label
        lbl = font.render(PLAYER_LABELS[2], True, PLAYER_COLORS[2])
        self.surf.blit(lbl, (W//2 - lbl.get_width()//2, y - 26))

        valid_set = set()
        if cp == 2 and self.mode == 'manual':
            vm = get_valid_moves(hand, self.state['top_card'])
            for m in vm:
                if m != 'Draw':
                    valid_set.add(id(m))

        mx, my = pygame.mouse.get_pos()

        for i, card in enumerate(hand):
            x      = start_x + i * (CARD_W + gap)
            is_sel = (self.selected_idx == i)
            card_r = pygame.Rect(x, y, CARD_W, CARD_H)

            # Hover lift
            hover = card_r.collidepoint(mx, my) and cp == 2 and self.mode == 'manual'
            dy    = -12 if (hover or is_sel) else 0

            # Highlight valid cards
            valid = any(card == vm_card for vm_card in
                        get_valid_moves(hand, self.state['top_card'])
                        if vm_card != 'Draw')

            if cp == 2 and self.mode == 'manual' and valid:
                glow_r = pygame.Rect(x - 3, y + dy - 3, CARD_W + 6, CARD_H + 6)
                pygame.draw.rect(s, PLAYER_COLORS[2], glow_r, border_radius=12)

            draw_card(s, card, x, y + dy, selected=is_sel)

        # Draw button
        draw_bx = W//2 - total_w//2 + total_w + 20
        draw_by = y + CARD_H//2 - 20
        draw_rect = pygame.Rect(draw_bx, draw_by, 80, 40)
        hover_draw = draw_rect.collidepoint(mx, my)
        bg   = GOLD if hover_draw else GRAY_DARK
        fc   = BG_DARK if hover_draw else GOLD
        draw_rounded_rect(s, bg, draw_rect, 8)
        pygame.draw.rect(s, GOLD_DIM, draw_rect, 1, border_radius=8)
        dfont = pygame.font.SysFont('consolas', 14, bold=True)
        dtxt  = dfont.render("DRAW", True, fc)
        s.blit(dtxt, dtxt.get_rect(center=draw_rect.center))

        # Active indicator
        if active and self.mode == 'manual':
            alpha = pulse_alpha(self.anim_t, speed=3.0, lo=80, hi=200)
            ind   = pygame.Surface((total_w + 28, 4), pygame.SRCALPHA)
            ind.fill((*PLAYER_COLORS[2], alpha))
            s.blit(ind, (start_x - 14, y + CARD_H + 30))

    def _draw_log_panel(self):
        s    = self.surf
        rect = pygame.Rect(W - 255, 10, 245, 300)
        draw_panel(s, rect, title="  EVENT LOG", alpha=210)

        font = pygame.font.SysFont('consolas', 12)
        y    = rect.y + 26
        for msg, color in self.log[-12:]:
            # Wrap long messages
            words  = msg
            lbl    = font.render(words[:36], True, color)
            s.blit(lbl, (rect.x + 10, y))
            y += 18

    def _draw_hud(self):
        s    = self.surf
        font = pygame.font.SysFont('consolas', 13, bold=True)

        # Turn counter
        tc = font.render(f"Turn: {self.turn_count}", True, OFF_WHITE)
        s.blit(tc, (12, 12))

        # Deck counter
        dc = font.render(f"Deck: {len(self.state['deck'])}", True, OFF_WHITE)
        s.blit(dc, (12, 32))

        # Mode badge
        mode_font = pygame.font.SysFont('consolas', 13)
        mode_lbl  = mode_font.render(
            f"Mode: {self.mode.upper()}", True, GOLD_DIM)
        s.blit(mode_lbl, (12, 52))

        # Restart hint
        hint = pygame.font.SysFont('consolas', 12).render(
            "R — restart   ESC — menu", True, GRAY_MID)
        s.blit(hint, (12, H - 22))

    def _draw_winner_overlay(self):
        s = self.surf
        # Dark vignette
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 160))
        s.blit(veil, (0, 0))

        # Winner box
        box = pygame.Rect(W//2 - 260, H//2 - 120, 520, 240)
        draw_rounded_rect(s, (12, 28, 20), box, 16)
        pygame.draw.rect(s, GOLD, box, 2, border_radius=16)

        big  = pygame.font.SysFont('impact', 64)
        med  = pygame.font.SysFont('consolas', 22)
        sml  = pygame.font.SysFont('consolas', 15)

        w_text = self.state['winner'] or "Draw"
        title  = big.render("🏆  WINNER", True, GOLD)
        name   = med.render(w_text, True, WHITE)
        turns  = sml.render(f"Completed in {self.turn_count} turns", True, OFF_WHITE)
        hint   = sml.render("Press R to restart  ·  ESC for menu", True, GRAY_MID)

        s.blit(title,  title.get_rect(center=(W//2, H//2 - 60)))
        s.blit(name,   name.get_rect(center=(W//2, H//2)))
        s.blit(turns,  turns.get_rect(center=(W//2, H//2 + 42)))
        s.blit(hint,   hint.get_rect(center=(W//2, H//2 + 80)))

    # ──── Input ──────────────────────────────────

    def handle_click(self, pos):
        if self.done:
            return
        cp = self.state['current_player']
        if cp != 2 or self.mode != 'manual':
            return

        hand  = self.state['p3_hand']
        n     = len(hand)
        gap   = 10
        total_w = n * (CARD_W + gap)
        start_x = W//2 - total_w // 2
        y       = H - CARD_H - 52

        # Check card clicks
        for i, card in enumerate(hand):
            x     = start_x + i * (CARD_W + gap)
            rect  = pygame.Rect(x, y - 12, CARD_W, CARD_H + 12)
            if rect.collidepoint(pos):
                # Check if valid
                vm = get_valid_moves(hand, self.state['top_card'])
                if card in vm:
                    if self.selected_idx == i:
                        # Double-click to confirm play
                        self.manual_play(card)
                    else:
                        self.selected_idx = i
                return

        # Check Draw button
        draw_bx = W//2 - total_w//2 + total_w + 20
        draw_by = y + CARD_H//2 - 20
        draw_rect = pygame.Rect(draw_bx, draw_by, 80, 40)
        if draw_rect.collidepoint(pos):
            self.manual_play('Draw')


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def main():
    pygame.init()
    pygame.display.set_caption("UNO  ·  AI Adversarial Search")
    screen = pygame.display.set_mode((W, H))
    clock  = pygame.time.Clock()

    # Screens
    menu = GameScreen(screen)
    game = None
    t    = 0.0

    while True:
        dt = clock.tick(60) / 1000.0
        t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game = None   # Return to menu
                    menu.choice = None
                if event.key == pygame.K_r and game is not None:
                    game = UnoGame(screen, game.mode)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game is None:
                    menu.handle_click(event.pos)
                    if menu.choice:
                        game = UnoGame(screen, menu.choice)
                else:
                    game.handle_click(event.pos)

        if game is None:
            menu.draw(t)
        else:
            game.update(dt)
            game.draw()

            # Auto-restart to menu if done and restart flagged
            if game.restart:
                game = None

        pygame.display.flip()


if __name__ == '__main__':
    main()
