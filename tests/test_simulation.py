"""Simulation de bout en bout : la vraie boucle du bot (run_loop) contre un faux NTE.

Le faux jeu reproduit les écrans observés en jeu (captures du 07/11) :
invite « Fishing », panneau « Fishing Preparations », lancer, bannière « Fish on
the hook », jauge (zone verte mobile, ligne jaune pilotée par Q/D), F de fin,
stock d'appâts consommé à chaque lancer, « Equip bait before fishing », fenêtre
« Switch Bait », vente (coquillages), boutique, « Rewards », et sortie du mode
pêche si un Échap est de trop.

Chaque scénario enchaîne plusieurs cycles complets, en 2560×1600 (résolution
des images de référence) et en 1920×1080 (interface réduite).

Lancer : python -m unittest tests.test_simulation
"""

import argparse
import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hook  # noqa: E402
from session import HumanConfig, Humanizer, Stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TPL = ROOT / "templates"
GREEN_BGR = (60, 200, 60)
YELLOW_BGR = (0, 230, 255)
GREY_BGR = (60, 60, 60)


def equipped_slot(empty_slot):
    img = empty_slot.copy()
    h, w = img.shape[:2]
    cv2.circle(img, (w // 2, int(h * 0.36)), int(w * 0.25), (150, 90, 230), -1)
    img[int(h * 0.8):, :] = 25
    cv2.putText(img, "Universal Bait", (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    return img


def greyed(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.min()) / max(1.0, g.max() - g.min()) * 70 + 40
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


class FakeWindow:
    title = "NTE"

    def __init__(self, width, height):
        self.size = (width, height)

    def is_foreground(self):
        return True

    def client_region(self):
        return (0, 0) + self.size


class FakeNTE:
    def __init__(self, width=2560, height=1600, mode="fishing", equipped=True, stock=50,
                 shells=0, sell_gives=100, bite_after=3, gauge_frames=25):
        self.W, self.H = width, height
        self.s = height / 1600 if (width, height) != (2560, 1600) else 1.0
        self.mode, self.equipped, self.stock = mode, equipped, stock
        self.shells, self.sell_gives = shells, sell_gives
        self.bite_after, self.gauge_frames = bite_after, gauge_frames
        self.origin = "fishing"
        self.menu_origin = "fishing"
        self.ticks = 0
        self.fish = 0
        self.left_fishing = False
        self.keys, self.clicks, self.events = [], [], []
        self.drawn = []
        self.ghost = None
        raw = {p.stem: cv2.imread(str(p)) for p in (TPL / "restock").glob("*.png")}
        raw["cube_tab"] = greyed(raw["cube_tab"])
        raw["equipped"] = equipped_slot(raw["bait_slot_empty"])
        for name in ("fishing_prompt", "fish_on_hook", "equip_bait_banner"):
            raw[name] = cv2.imread(str(TPL / f"{name}.png"))
        self.images = {k: self._scaled(v) for k, v in raw.items()}

    def _scaled(self, img):
        if self.s == 1.0:
            return img
        return cv2.resize(img, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------ écran
    def layout(self):
        """(nom, x relatif, y relatif) des éléments affichés."""
        m = self.mode
        if m == "spot":
            return [("fishing_prompt", 0.55, 0.62)]
        if m == "prep":
            return [("equipped" if self.equipped else "bait_slot_empty", 0.84, 0.62),
                    ("start_fishing", 0.78, 0.82)]
        if m == "bite":
            return [("fish_on_hook", 0.26, 0.18)]
        if m == "equip_banner":
            return [("equip_bait_banner", 0.39, 0.40)]
        if m == "dialog":
            return [("cancel", 0.33, 0.62), ("switch" if self.stock else "purchase", 0.56, 0.62)]
        one = {"menu": "quick_submit", "submit_rewards": "rewards_close", "menu2": "cube_tab",
               "sell_tab": "quick_sell", "sell_confirm": "confirm", "sell_rewards": "rewards_close",
               "shop": "universal_bait", "shop_item": "max_quantity", "shop_max": "purchase",
               "buy_confirm": "confirm", "buy_rewards": "rewards_close"}.get(m)
        return [(one, 0.45, 0.70)] if one else []

    def grab(self, region=None):
        self.ticks += 1
        if self.mode == "casting" and self.ticks >= self.bite_after:
            self.mode = "bite"
        if self.mode == "gauge" and self.ticks >= self.gauge_frames:
            self.mode = "caught"
        canvas = np.full((self.H, self.W, 3), (120, 90, 40), np.uint8)     # mer
        if self.mode == "gauge":
            self._draw_gauge(canvas)
        if self.ghost is not None and self.mode in ("shop", "shop_item", "shop_max"):
            x, y = self.ghost
            img = self.images["purchase"]
            canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        self.drawn = []
        for name, rx, ry in self.layout():
            img = self.images[name]
            x, y = int(rx * self.W), int(ry * self.H)
            canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
            self.drawn.append((name, (x, y, img.shape[1], img.shape[0])))
            if name == "purchase" and self.mode == "dialog":
                self.ghost = (x, y)
        if region is None:
            return canvas
        x, y, w, h = region
        return canvas[y:y + h, x:x + w].copy()

    def _draw_gauge(self, canvas):
        W, H = self.W, self.H
        y0, y1 = int(0.06 * H), int(0.072 * H)
        cv2.rectangle(canvas, (int(0.25 * W), y0), (int(0.75 * W), y1), GREY_BGR, -1)
        self.green = 0.5 * W + 0.08 * W * math.sin(self.ticks * 0.2)
        half = 0.04 * W
        cv2.rectangle(canvas, (int(self.green - half), y0), (int(self.green + half), y1), GREEN_BGR, -1)
        x = int(max(0.25 * W, min(0.75 * W, self.yellow)))
        t = max(2, int(0.0012 * W))
        cv2.rectangle(canvas, (x - t, y0 - int(0.01 * H)), (x + t, y1 + int(0.01 * H)), YELLOW_BGR, -1)

    # ---------------------------------------------------------------- clavier
    name = "fake"

    def prepare(self, keys):
        pass

    def down(self, key):
        self.keys.append(key)
        self.events.append(key)
        m = self.mode
        if key == "f":
            if m == "spot":
                self.mode = "prep"
            elif m == "fishing":
                if self.equipped and self.stock > 0:
                    self.stock -= 1
                    self.mode, self.ticks = "casting", 0
                else:
                    self.equipped = False
                    self.mode = "equip_banner"
            elif m == "bite":
                self.mode, self.ticks = "gauge", 0
                self.yellow = 0.5 * self.W + 0.06 * self.W
            elif m == "caught":
                self.fish += 1
                self.mode = "fishing"
        elif key in ("q", "d") and m == "gauge":
            self.yellow += (-1 if key == "q" else 1) * 0.015 * self.W
        elif key == "e" and m in ("fishing", "equip_banner"):
            self.origin, self.mode = "fishing", "dialog"
        elif key == "q" and m in ("fishing", "equip_banner"):
            # Vente seulement en mode pêche : Q ne fait rien sur le panneau de
            # préparation (capture du 08/11 à 10:14).
            self.menu_origin = "fishing"
            self.mode = "menu"
        elif key == "esc":
            if m in ("shop", "shop_item", "shop_max", "buy_confirm", "shop_done"):
                self.mode = "dialog"
            elif m == "dialog":
                self.mode = self.origin
            elif m in ("menu", "menu2", "sell_tab", "sell_confirm", "menu_after"):
                self.mode = self.menu_origin
            elif m in ("fishing", "equip_banner", "casting", "bite"):
                self.left_fishing = True            # un Échap de trop : sortie du mode pêche
                self.mode = "prep"
            elif m == "prep":
                self.mode = "spot"

    def up(self, key):
        pass

    # ----------------------------------------------------------------- souris
    def click(self, box):
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        hit = None
        for name, (bx, by, bw, bh) in self.drawn:
            if name == "rewards_close":
                hit = name if cy < by else hit
                continue
            if name in ("bait_slot_empty", "equipped"):
                bh *= 0.7
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                hit = name
        self.clicks.append(hit or "?")
        self.events.append(f"clic {hit or '?'}")
        m = self.mode
        if hit in ("bait_slot_empty", "equipped"):
            self.origin, self.mode = "prep", "dialog"
        elif hit == "start_fishing" and self.equipped:
            self.mode = "fishing"
        elif hit == "cancel":
            self.mode = self.origin
        elif hit == "switch":
            self.equipped = True
            self.mode = self.origin
        elif hit == "purchase" and m == "dialog":
            self.mode = "shop"
        elif hit == "purchase" and m == "shop_max":
            if self.shells >= 5:
                self.shells -= 5
                self.mode = "buy_confirm"
        elif hit == "confirm":
            if m == "sell_confirm":
                self.shells += self.sell_gives
                self.mode = "sell_rewards"
            elif m == "buy_confirm":
                self.stock += 99
                self.mode = "buy_rewards"
        elif hit == "rewards_close":
            self.mode = {"submit_rewards": "menu2", "sell_rewards": "menu_after",
                         "buy_rewards": "shop_done"}[m]
        elif hit is not None:
            self.mode = {"quick_submit": "submit_rewards", "cube_tab": "sell_tab",
                         "quick_sell": "sell_confirm", "universal_bait": "shop_item",
                         "max_quantity": "shop_max"}.get(hit, m)


class SimulationTest(unittest.TestCase):
    RESOLUTIONS = {"2560x1600": (2560, 1600), "1920x1080": (1920, 1080)}

    def setUp(self):
        self.saved = (hook.grab, hook.keyboard, hook.human, hook.target, hook.click_box,
                      hook.UI.candidates, hook.UI.locked, hook.UI.on_lock,
                      hook._key_state, hook._visible_cursor_in_corner)
        # Indépendant du vrai clavier et de la vraie souris : Échap, F8 ou la souris
        # dans un coin pendant les tests (ou le bot lancé en parallèle) ne doivent
        # pas interrompre la simulation.
        hook._key_state = lambda vk: 0
        hook._visible_cursor_in_corner = lambda: False

    def tearDown(self):
        (hook.grab, hook.keyboard, hook.human, hook.target, hook.click_box,
         hook.UI.candidates, hook.UI.locked, hook.UI.on_lock,
         hook._key_state, hook._visible_cursor_in_corner) = self.saved

    def run_bot(self, game, cycles):
        hook.grab = game.grab
        hook.keyboard = game
        hook.click_box = game.click
        hook.human = Humanizer(HumanConfig(enabled=False, reaction_median=0.0))
        hook.target = FakeWindow(game.W, game.H)
        cfg = hook.Config(config_path=None, stats_path=None)
        cfg.human.enabled = False
        cfg.session.breaks_enabled = False
        cfg.start_delay = cfg.reaction_delay = cfg.cycle_pause = (0.0, 0.0)
        cfg.action_press_duration = (0.001, 0.001)
        cfg.press_duration = (0.001, 0.003)
        cfg.press_cooldown = (0.0, 0.0)
        cfg.restock_step_delay = (0.0, 0.0)
        cfg.scan_interval = 0.0
        cfg.gauge_lost_grace = 0.0
        cfg.banner_timeout, cfg.gauge_appear_timeout, cfg.restock_step_timeout = 4.0, 3.0, 3.0
        if (game.W, game.H) != (2560, 1600):
            cfg.resolution = (game.W, game.H)          # comme « .\start.bat -r 1920x1080 »
        hook.configure_ui_scale(cfg)
        stats = Stats(None)
        banner = hook.BannerDetector(cfg.banner_template, cfg.banner_threshold, cfg.banner_scales)
        try:
            hook.run_loop(cfg, argparse.Namespace(cycles=cycles), banner, hook.GaugeDetector(cfg), stats)
            error = None
        except hook.BlockedStop as exc:
            error = exc
        return stats, error

    def check_no_exit(self, game):
        self.assertFalse(game.left_fishing, f"sorti du mode pêche : {game.events}")

    def for_each_resolution(self, scenario):
        for label, (w, h) in self.RESOLUTIONS.items():
            with self.subTest(resolution=label):
                scenario(w, h)

    # ---------------------------------------------------------------------
    def test_fishing_mode_with_bait_three_fish(self):
        def scenario(w, h):
            game = FakeNTE(w, h, mode="fishing", equipped=True, stock=50)
            stats, error = self.run_bot(game, cycles=3)
            self.assertIsNone(error)
            self.assertEqual(game.fish, 3, game.events)
            self.assertEqual([r.outcome for r in stats.records], ["unknown"] * 3)
            self.assertTrue(all(r.in_zone_ratio > 0.3 for r in stats.records))
            self.check_no_exit(game)
        self.for_each_resolution(scenario)

    def test_launched_on_preparations_panel_with_bait(self):
        """Cas signalé le 08/11 (PC en 1080p) : lancement sur « Fishing Preparations »."""
        def scenario(w, h):
            game = FakeNTE(w, h, mode="prep", equipped=True, stock=50)
            stats, error = self.run_bot(game, cycles=2)
            self.assertIsNone(error)
            self.assertEqual(game.clicks[0], "start_fishing")
            self.assertEqual(game.fish, 2, game.events)
            self.check_no_exit(game)
        self.for_each_resolution(scenario)

    def test_launched_at_spot_with_fishing_prompt(self):
        def scenario(w, h):
            game = FakeNTE(w, h, mode="spot", equipped=True, stock=50)
            stats, error = self.run_bot(game, cycles=2)
            self.assertIsNone(error)
            self.assertEqual(game.keys[0], "f")
            self.assertEqual(game.fish, 2, game.events)
            self.check_no_exit(game)
        self.for_each_resolution(scenario)

    def test_preparations_panel_bait_in_stock_not_equipped(self):
        def scenario(w, h):
            game = FakeNTE(w, h, mode="prep", equipped=False, stock=50)
            stats, error = self.run_bot(game, cycles=3)
            self.assertIsNone(error)
            self.assertEqual([r.outcome for r in stats.records], ["restock", "unknown", "unknown"])
            self.assertEqual(game.fish, 2, game.events)
            self.assertEqual(game.shells, 0)                 # rien vendu, rien acheté
            self.check_no_exit(game)
        self.for_each_resolution(scenario)

    def test_runs_out_of_bait_sells_buys_equips_and_continues(self):
        def scenario(w, h):
            game = FakeNTE(w, h, mode="fishing", equipped=True, stock=1)
            stats, error = self.run_bot(game, cycles=3)
            self.assertIsNone(error, game.events)
            self.assertEqual([r.outcome for r in stats.records], ["unknown", "restock", "unknown"])
            self.assertEqual(game.fish, 2, game.events)
            self.assertIn("clic quick_sell", game.events)
            self.assertLess(game.events.index("clic quick_sell"),
                            game.events.index("clic universal_bait"))   # vente avant achat
            self.check_no_exit(game)
        self.for_each_resolution(scenario)

    def test_launched_at_spot_without_bait_buys_from_panel(self):
        """Cas du 08/11 à 10:14 : invite « Fishing » -> F -> panneau sans appât.
        Pas de vente possible ici : achat direct, équipement, Start Fishing, pêche."""
        def scenario(w, h):
            game = FakeNTE(w, h, mode="spot", equipped=False, stock=0, shells=100)
            stats, error = self.run_bot(game, cycles=3)
            self.assertIsNone(error, game.events)
            self.assertEqual([r.outcome for r in stats.records], ["restock", "unknown", "unknown"])
            self.assertNotIn("clic cancel", game.events)
            self.assertNotIn("clic quick_submit", game.events)   # pas de vente depuis le panneau
            self.assertEqual(game.fish, 2, game.events)
        self.for_each_resolution(scenario)

    def test_launched_at_spot_without_bait_nor_shells(self):
        def scenario(w, h):
            game = FakeNTE(w, h, mode="spot", equipped=False, stock=0, shells=0)
            stats, error = self.run_bot(game, cycles=3)
            self.assertIsInstance(error, hook.BlockedStop)
            self.assertIn("coquillages", str(error))
            self.assertNotIn("clic quick_submit", game.events)
        self.for_each_resolution(scenario)

    def test_no_bait_no_shells_stops_inside_fishing_mode(self):
        def scenario(w, h):
            game = FakeNTE(w, h, mode="fishing", equipped=False, stock=0, sell_gives=0)
            stats, error = self.run_bot(game, cycles=3)
            self.assertIsInstance(error, hook.BlockedStop)
            self.assertEqual(game.fish, 0)
            self.check_no_exit(game)
        self.for_each_resolution(scenario)


if __name__ == "__main__":
    unittest.main()
