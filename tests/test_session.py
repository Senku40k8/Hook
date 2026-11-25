"""Tests du comportement humain, des sessions, des statistiques, de l'apprentissage
du contrôleur et d'un cycle complet contre un faux jeu (sans écran ni clavier réels).

Lancer : python -m unittest discover -s tests
"""

import random
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hook  # noqa: E402


def setUpModule():
    """Tests indépendants du vrai clavier et de la vraie souris (Échap, F8, souris
    dans un coin) : un appui pendant les tests, ou le bot lancé en parallèle,
    ne doit pas les interrompre. PauseKeyTest remplace _key_state à sa façon."""
    global _saved_inputs
    _saved_inputs = (hook._key_state, hook._visible_cursor_in_corner, hook.layout_char)
    hook._key_state = lambda vk: 0
    hook._visible_cursor_in_corner = lambda: False
    hook.layout_char = lambda k: k          # clavier QWERTY, quel que soit le PC
    # Pas d'images de diagnostic dans debug/ : elles écraseraient celles du vrai jeu.
    hook._first_track_done = hook._gauge_miss_saved = hook._banner_miss_saved = True
    hook._sale_saved.update({"confirm", "rewards"})


def tearDownModule():
    hook._key_state, hook._visible_cursor_in_corner, hook.layout_char = _saved_inputs


from session import (  # noqa: E402
    CycleRecord, HumanConfig, Humanizer, SessionConfig, SessionPlanner, Stats,
    format_duration, parse_duration,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# --------------------------------------------------------------------------- #
# Comportement humain
# --------------------------------------------------------------------------- #
class HumanizerTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.h = Humanizer(HumanConfig(), random.Random(42), self.clock)

    def test_reaction_times_are_human_like(self):
        times = [self.h.reaction() for _ in range(2000)]
        lo, hi = HumanConfig().reaction_bounds
        self.assertTrue(all(lo <= t <= hi for t in times))
        median = sorted(times)[len(times) // 2]
        self.assertTrue(0.22 <= median <= 0.36, median)
        self.assertTrue(any(t > 0.6 for t in times), "quelques réactions lentes attendues")

    def test_within_stays_in_bounds_even_when_tired(self):
        self.clock.t = 3 * 3600   # 3 h sans pause : fatigue maximale
        values = [self.h.within((0.5, 1.5)) for _ in range(2000)]
        self.assertTrue(all(0.5 <= v <= 1.5 for v in values))

    def test_fatigue_slows_reactions_then_resets(self):
        fresh = self.h.fatigue()
        self.clock.t = 40 * 60
        tired = self.h.fatigue()
        self.assertGreater(tired, fresh)
        self.assertLessEqual(tired, HumanConfig().fatigue_max)
        self.h.rest()
        self.assertEqual(self.h.fatigue(), 1.0)

    def test_cycle_pause_never_below_three_seconds(self):
        pauses = [self.h.cycle_pause((3.0, 4.5)) for _ in range(3000)]
        self.assertGreaterEqual(min(pauses), 3.0)
        short = sum(p <= 4.5 for p in pauses) / len(pauses)
        self.assertTrue(0.75 <= short <= 0.95, short)   # surtout courtes
        self.assertTrue(any(p >= 10 for p in pauses))    # parfois très longues

    def test_imperfections_happen_sometimes(self):
        skips = sum(self.h.skip_correction() is not None for _ in range(5000))
        overs = sum(self.h.press_factor() > 1.2 for _ in range(5000))
        self.assertTrue(100 <= skips <= 350, skips)
        self.assertTrue(150 <= overs <= 400, overs)

    def test_disabled_is_plain_uniform(self):
        h = Humanizer(HumanConfig(enabled=False), random.Random(1), self.clock)
        self.assertIsNone(h.skip_correction())
        self.assertEqual(h.press_factor(), 1.0)
        self.assertTrue(all(3.0 <= h.cycle_pause((3.0, 4.5)) <= 4.5 for _ in range(500)))


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
class SessionPlannerTest(unittest.TestCase):
    def test_breaks_follow_play_time(self):
        clock = FakeClock()
        planner = SessionPlanner(SessionConfig(play_minutes=(20, 20), break_minutes=(5, 5)),
                                 random.Random(0), clock)
        clock.t = 19 * 60
        self.assertIsNone(planner.break_due())
        clock.t = 20 * 60 + 1
        self.assertAlmostEqual(planner.break_due(), 300)
        planner.after_break()
        self.assertIsNone(planner.break_due())

    def test_no_breaks_option(self):
        clock = FakeClock()
        planner = SessionPlanner(SessionConfig(breaks_enabled=False), random.Random(0), clock)
        clock.t = 10 * 3600
        self.assertIsNone(planner.break_due())

    def test_limits(self):
        clock = FakeClock()
        planner = SessionPlanner(SessionConfig(max_duration=3600, max_fish=50),
                                 random.Random(0), clock)
        self.assertIsNone(planner.finished(10))
        self.assertIn("poissons", planner.finished(50))
        clock.t = 3600
        self.assertIn("durée", planner.finished(0))

    def test_parse_duration(self):
        self.assertEqual(parse_duration("2h"), 7200)
        self.assertEqual(parse_duration("1h30"), 5400)
        self.assertEqual(parse_duration("90m"), 5400)
        self.assertEqual(parse_duration("45min"), 2700)
        self.assertEqual(parse_duration("1h 15m"), 4500)
        self.assertEqual(parse_duration("30"), 1800)       # nombre seul = minutes
        for bad in ("", "abc", "2x"):
            with self.assertRaises(ValueError):
                parse_duration(bad)
        self.assertEqual(format_duration(5400), "1 h 30 min")


# --------------------------------------------------------------------------- #
# Statistiques
# --------------------------------------------------------------------------- #
class StatsTest(unittest.TestCase):
    def test_counts_csv_and_consecutive_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logs" / "catches.csv"
            stats = Stats(path)
            stats.add(CycleRecord(1, "success", 2.0, 8.0, 5, 6, 0.9))
            stats.add(CycleRecord(2, "no_banner"))
            stats.add(CycleRecord(3, "fail", 1.0, 5.0, 3, 2, 0.4))
            self.assertEqual(stats.consecutive_failures, 2)
            stats.add(CycleRecord(4, "unknown", 1.5, 7.0, 4, 4, 0.8))
            self.assertEqual(stats.consecutive_failures, 0)
            self.assertEqual(stats.fish, 2)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 5)                 # en-tête + 4 cycles
            self.assertIn("outcome", lines[0])
            summary = stats.summary()
            self.assertIn("2 poisson(s)", summary)
            self.assertIn("échec 1", summary)

    def test_fons_per_hour_from_sales(self):
        """Ventes lues à l'écran : Fons encaissés, Fons par poisson appris entre deux
        ventes, Fons par heure bruts et nets (appâts payés)."""
        clock = FakeClock()
        stats = Stats(None, clock=clock, bait_price=5)
        stats.add_sale(1000)                 # poissons d'avant la session : rien appris
        self.assertIsNone(stats.fons_per_fish)
        for i in range(100):
            stats.add(CycleRecord(i, "success"))
        stats.add_sale(2000)                 # 100 poissons -> 20 Fons par poisson
        self.assertEqual(stats.fons_per_fish, 20)
        clock.t += 3600
        summary = stats.summary()
        self.assertIn("Fons encaissés : 3 000 en 2 vente(s), soit 3 000 Fons/heure", summary)
        self.assertIn("~2 000 Fons/heure (~20 par poisson), ~1 500 Fons/heure", summary)

    def test_amount_parsing(self):
        self.assertEqual(hook.parse_amount(["x", "1972"]), 1972)
        self.assertEqual(hook.parse_amount(["*1,615"]), 1615)
        self.assertEqual(hook.parse_amount(["xl9O4"]), 1904)
        self.assertIsNone(hook.parse_amount(["Income", "Details"]))
        self.assertIsNone(hook.parse_amount([]))


# --------------------------------------------------------------------------- #
# Apprentissage du contrôleur
# --------------------------------------------------------------------------- #
def state(yellow_x, left=150, right=250):
    return hook.GaugeState(left, right, 25, 36, yellow_x)


class ControllerLearningTest(unittest.TestCase):
    def setUp(self):
        self.cfg = hook.Config()
        self.ctrl = hook.GaugeController(self.cfg)

    def press(self, expected, duration, x_before, x_after, t):
        self.ctrl._pending = (expected, duration, x_before, t)
        self.ctrl.observe(state(x_after), t + 0.01)

    def test_learns_line_speed_and_doses_presses(self):
        for i in range(3):
            self.press(+1, 0.1, 100, 150, i)                # 50 px en 0,1 s -> 500 px/s
        self.assertAlmostEqual(self.ctrl.line_speed, 500, delta=1)
        key, duration = self.ctrl.decide(state(100))       # écart 100 px
        self.assertEqual(key, "d")
        expected = 100 * self.cfg.correction_fraction / 500
        self.assertAlmostEqual(duration, expected, delta=expected * 0.2)

    def test_auto_inverts_when_line_goes_the_wrong_way(self):
        self.assertFalse(self.cfg.invert_controls)
        for i in range(5):
            self.press(+1, 0.1, 150, 120, i)                # attendu à droite, part à gauche
        self.assertTrue(self.cfg.invert_controls)
        self.assertTrue(self.ctrl.inverted_now)
        self.assertEqual(self.ctrl.decide(state(100))[0], "a")

    def test_overestimated_speed_recovers(self):
        """Cas du 09/11 (autre PC) : vitesse mémorisée très surestimée -> appuis
        minuscules, la ligne bouge à peine (< 3 px) et ne suit pas. Ces petits
        mouvements comptent désormais : la vitesse redescend en 2 appuis."""
        self.cfg.line_speed = 5000.0                        # aberrant (largeur verte 100 px)
        self.ctrl = hook.GaugeController(self.cfg)
        self.assertLess(self.ctrl.decide(state(100))[1], 0.04)   # appui minuscule
        self.press(+1, 0.05, 100, 101, 0)                   # 1 px : quasiment rien
        self.press(+1, 0.05, 101, 102, 1)
        self.assertLess(self.ctrl.line_speed, 100)
        self.assertGreater(self.ctrl.decide(state(100))[1], 0.15)

    def test_presses_lengthen_when_line_does_not_get_closer(self):
        saved_tap, hook.tap = hook.tap, lambda key, rng: None
        try:
            self.cfg.press_cooldown = (0.0, 0.0)
            self.cfg.line_speed = 1000.0                    # premier appui court (80 ms)
            self.ctrl = hook.GaugeController(self.cfg)
            durations = []
            for _ in range(4):                              # la ligne reste à 100
                self.ctrl._next_press_at = 0
                self.ctrl.step(state(100))
                durations.append(self.ctrl._pending[1])
        finally:
            hook.tap = saved_tap
        self.assertGreater(durations[2], durations[0] * 1.5)
        self.assertLessEqual(max(durations), self.cfg.press_duration[1])

    def test_line_crossing_the_bar_preferred_to_small_yellow_item(self):
        img = np.full((120, 600, 3), 20, np.uint8)
        cv2.rectangle(img, (250, 50), (350, 66), GREEN_BGR, -1)
        cv2.rectangle(img, (296, 50), (305, 58), YELLOW_BGR, -1)     # petit élément jaune, au centre
        cv2.rectangle(img, (219, 40), (221, 76), YELLOW_BGR, -1)     # la vraie ligne, qui traverse
        found = hook.GaugeDetector(hook.Config()).detect(img)
        self.assertAlmostEqual(found.yellow_x, 220, delta=2)

    def test_learned_speed_is_remembered(self):
        for i in range(3):
            self.press(+1, 0.1, 100, 140, i)
        self.cfg.config_path = None
        hook.learn_from(self.cfg, self.ctrl)
        self.assertAlmostEqual(self.cfg.line_speed, 400, delta=1)
        restarted = hook.GaugeController(self.cfg)
        self.assertGreaterEqual(restarted.speed_samples, self.cfg.line_speed_min_samples)


# --------------------------------------------------------------------------- #
# Cycle complet contre un faux jeu
# --------------------------------------------------------------------------- #
GREEN_BGR = (170, 202, 44)          # turquoise de la vraie jauge
YELLOW_BGR = (0, 230, 255)


def text_image(text, size=(220, 900)):
    img = np.full((size[0], size[1], 3), 30, np.uint8)
    cv2.putText(img, text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return img


class FakeGame:
    """Simule le jeu : F lance la ligne, la bannière apparaît, F ferre, la jauge
    s'affiche et sa ligne jaune bouge avec Q/D ; puis message de réussite."""

    def __init__(self, gauge_frames=40):
        self.phase = "idle"
        self.grabs = 0
        self.yellow = 90
        self.gauge_left = 2
        self.keys = []
        self.banner = text_image("Fish on the hook! Press the button to reel it!")
        self.success = text_image("Nice catch!")
        self.gauge_frames = gauge_frames

    # --- clavier -----------------------------------------------------------
    name = "fake"

    def prepare(self, keys):
        pass

    def down(self, key):
        self.keys.append(key)
        if key == "f":
            self.phase = {"idle": "cast", "banner": "gauge", "done": "idle"}.get(self.phase, self.phase)
            self.grabs = 0
        elif key == "d":
            self.yellow += 30
        elif key == "a":
            self.yellow -= 30

    def up(self, key):
        pass

    # --- écran -----------------------------------------------------------------
    def grab(self, region=None):
        self.grabs += 1
        if self.phase == "cast" and self.grabs > 3:
            self.phase = "banner"
        if self.phase == "banner":
            return self.banner
        if self.phase == "gauge":
            if self.grabs > self.gauge_frames:
                self.phase = "done"
            else:
                img = np.full((60, 400, 3), 20, np.uint8)
                cv2.rectangle(img, (20, 25), (380, 35), (60, 60, 60), -1)
                cv2.rectangle(img, (150, 25), (250, 35), GREEN_BGR, -1)
                x = int(max(25, min(375, self.yellow)))
                cv2.rectangle(img, (x - 1, 18), (x + 1, 42), YELLOW_BGR, -1)
                return img
        if self.phase == "done":
            return self.success
        return np.full((220, 900, 3), 30, np.uint8)


class FullCycleTest(unittest.TestCase):
    def setUp(self):
        self.game = FakeGame()
        self.saved = (hook.grab, hook.keyboard, hook.human, hook.target)
        hook.grab = self.game.grab
        hook.keyboard = self.game
        hook.human = Humanizer(HumanConfig(enabled=False))
        hook.target = None
        self.tmp = tempfile.TemporaryDirectory()
        tdir = Path(self.tmp.name)
        cv2.imwrite(str(tdir / "fish_on_hook.png"), self.game.banner[85:135, 20:880])
        cv2.imwrite(str(tdir / "result_success.png"), self.game.success[85:135, 20:260])

        cfg = hook.Config(config_path=None, templates_dir=tdir,
                          banner_template=tdir / "fish_on_hook.png",
                          banner_region=(0, 0, 900, 220), gauge_region=(0, 0, 400, 60))
        cfg.start_delay = cfg.reaction_delay = (0.0, 0.0)
        cfg.action_press_duration = (0.001, 0.001)
        cfg.press_duration = (0.001, 0.005)
        cfg.press_cooldown = (0.0, 0.0)
        cfg.gauge_lost_grace = 0.0
        cfg.scan_interval = 0.0
        cfg.result_timeout = 0.5
        self.cfg = cfg

    def tearDown(self):
        hook.grab, hook.keyboard, hook.human, hook.target = self.saved
        self.tmp.cleanup()

    def test_full_cycle(self):
        cfg = self.cfg
        banner = hook.BannerDetector(cfg.banner_template, cfg.banner_threshold, cfg.banner_scales)
        record = hook.run_cycle(cfg, banner, hook.GaugeDetector(cfg), 1,
                                hook.result_detector(cfg), hook.stop_detector(cfg))
        keys = self.game.keys
        self.assertEqual(keys[:2], ["f", "f"])             # lancer puis ferrer
        self.assertEqual(keys[-1], "f")                     # F final après la jauge
        self.assertIn("d", keys)                            # ligne ramenée vers le vert
        self.assertEqual(record.outcome, "success")
        self.assertGreater(record.presses_right, 0)
        self.assertGreater(record.in_zone_ratio, 0.5)       # la ligne a rejoint le vert

    def test_stop_message_blocks_before_casting(self):
        stop_dir = self.cfg.templates_dir / "stop"
        stop_dir.mkdir()
        cv2.imwrite(str(stop_dir / "inventaire_plein.png"), self.game.banner[85:135, 20:880])
        self.game.phase = "banner"                          # message bloquant à l'écran
        banner = hook.BannerDetector(self.cfg.banner_template, 0.8, (1.0,))
        with self.assertRaises(hook.BlockedStop):
            hook.run_cycle(self.cfg, banner, hook.GaugeDetector(self.cfg), 1,
                           hook.result_detector(self.cfg), hook.stop_detector(self.cfg))
        self.assertEqual(self.game.keys, [])                # aucune touche envoyée


# --------------------------------------------------------------------------- #
# Souris
# --------------------------------------------------------------------------- #
class MousePathTest(unittest.TestCase):
    def test_path_ends_exactly_on_target_and_is_curved(self):
        rng = random.Random(3)
        curved = 0
        for _ in range(50):
            path = Humanizer(HumanConfig(), rng).mouse_path((100, 100), (900, 500), 40)
            x, y, _ = path[-1]
            self.assertAlmostEqual(x, 900, delta=0.01)
            self.assertAlmostEqual(y, 500, delta=0.01)
            # écart max. à la ligne droite
            dev = max(abs((px - 100) * 400 - (py - 100) * 800) / 894.4 for px, py, _ in path)
            curved += dev > 5
        self.assertGreater(curved, 25)

    def test_duration_follows_fitts_law(self):
        rng = random.Random(5)
        h = Humanizer(HumanConfig(), rng)

        def avg(dist, size):
            return sum(sum(d for _, _, d in h.mouse_path((0, 0), (dist, 0), size))
                       for _ in range(200)) / 200
        self.assertLess(avg(100, 60), avg(1200, 60))     # plus loin = plus long
        self.assertLess(avg(600, 120), avg(600, 15))     # plus petit = plus long
        self.assertTrue(0.15 < avg(600, 60) < 1.0)

    def test_speed_profile_slows_down_near_target(self):
        path = Humanizer(HumanConfig(), random.Random(9)).mouse_path((0, 0), (800, 0), 40)
        xs = [p[0] for p in path]
        steps = [abs(b - a) for a, b in zip(xs, xs[1:])]
        third = len(steps) // 3
        self.assertGreater(sum(steps[third:2 * third]), sum(steps[-third:]))


# --------------------------------------------------------------------------- #
# Réapprovisionnement contre un faux jeu
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RESTOCK_DIR = ROOT / "templates" / "restock"
EQUIP_BANNER = ROOT / "templates" / "equip_bait_banner.png"


def equipped_slot(empty_slot):
    """Case d'appât après « Switch » : un appât dans le carré, son nom dessous."""
    img = empty_slot.copy()
    h, w = img.shape[:2]
    cv2.circle(img, (w // 2, int(h * 0.36)), int(w * 0.25), (150, 90, 230), -1)
    img[int(h * 0.8):, :] = 25
    cv2.putText(img, "Universal Bait", (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2)
    return img


def greyed(img):
    """Icône sur onglet gris (non sélectionné) au lieu d'orange."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.min()) / max(1.0, g.max() - g.min()) * 70 + 40
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


class FakeShop:
    """Faux jeu pour les menus d'appâts, calqué sur les captures du 07/11 :

    - panneau « Fishing Preparations » (« slot_empty ») : clic sur la case vide
      -> fenêtre « Switch Bait » ; en mode pêche, E ouvre la même fenêtre ;
    - « Switch Bait » : Cancel / Switch s'il reste des appâts, Cancel / Purchase
      sinon ; Cancel ou Échap ramène à l'écran d'origine ;
    - vente : Q, Quick Submit (+ Rewards), onglet cube, Quick Sell, Confirm,
      Rewards, Échap ; rapporte des coquillages ;
    - boutique : Universal Bait, double chevron, Purchase (sans coquillages : rien
      ne se passe), Confirm, Rewards (fermée par un clic dans le vide, pas par
      Échap), puis Échap x2 ;
    - en mode pêche, un Échap de trop fait sortir du mode pêche (« left_fishing »).
    """

    def __init__(self, stock=0, shells=0, sell_gives=100):
        self.screen = "idle"
        self.origin = "fishing_mode"           # où ramène la fermeture de « Switch Bait »
        self.stock, self.shells, self.sell_gives = stock, shells, sell_gives
        self.equip_works = True
        self.confirm_on_buy = True             # False : petit achat, « Rewards » direct
        self.menu_key = "q"                    # touche qui ouvre le menu du marchand
        self.extra_rewards = 0                 # « Rewards » supplémentaires après la vente
        self.tall_rewards = False              # fenêtre « Rewards » jusqu'au texte
        self.ghost_purchase = False            # 1er Purchase visible derrière la boutique
        self.ghost_box = None
        self.keys, self.clicks, self.events = [], [], []
        self.drawn = []                        # (bouton, rectangle) de l'écran courant
        self.images = {p.stem: cv2.imread(str(p)) for p in RESTOCK_DIR.glob("*.png")}
        self.images["cube_tab"] = greyed(self.images["cube_tab"])
        self.images["equipped"] = equipped_slot(self.images["bait_slot_empty"])
        self.images["fishing_prompt"] = cv2.imread(str(ROOT / "templates" / "fishing_prompt.png"))
        self.banner = cv2.imread(str(EQUIP_BANNER))

    # --- ce qui est affiché ---------------------------------------------------
    def buttons(self):
        s = self.screen
        if s == "dialog":
            return ["cancel", "switch" if self.stock else "purchase"]
        return {
            "slot_empty": ["bait_slot_empty"], "spot": ["fishing_prompt"],
            "prep_equipped": ["start_fishing"], "start_panel": ["start_fishing"],
            "panel_unknown": ["start_fishing"],     # panneau, case d'appât non reconnue
            "equipped": ["equipped"],
            "menu": ["quick_submit"], "submit_rewards": ["rewards_close"],
            "menu2": ["cube_tab"], "sell_tab": ["quick_sell"], "sell_confirm": ["confirm"],
            "sell_rewards": ["rewards_close"],
            "shop": ["universal_bait"], "shop_item": ["max_quantity"], "shop_max": ["purchase"],
            "buy_confirm": ["confirm"], "buy_rewards": ["rewards_close"],
        }.get(s, [])

    # --- clavier ---------------------------------------------------------------
    name = "fake"

    def prepare(self, keys):
        pass

    def down(self, key):
        self.keys.append(key)
        self.events.append(key)
        s = self.screen
        if key == "f" and s == "idle":
            self.screen = "equip_banner"
        elif key == "f" and s == "spot":
            self.screen = self.after_spot
        elif key == "e" and s in ("fishing_mode", "equip_banner"):
            self.origin = "fishing_mode"
            self.screen = "dialog"
        elif key == self.menu_key and s in ("fishing_mode", "equip_banner"):
            # Q (vente) ne marche qu'en mode pêche, pas sur le panneau de
            # préparation (capture du 08/11 à 10:14).
            self.menu_origin = "fishing_mode"
            self.screen = "menu"
        elif key == "esc":
            menu_origin = getattr(self, "menu_origin", "fishing_mode")
            shop = ("shop", "shop_item", "shop_max", "buy_confirm", "shop_done")
            menus = ("menu", "menu2", "sell_tab", "sell_confirm", "menu_after")
            if s in shop:
                self.screen = "dialog"                 # la boutique se ferme sur « Switch Bait »
            elif s in menus:
                self.screen = menu_origin
            else:                                      # « Rewards » : Échap sans effet
                self.screen = {"dialog": self.origin, "fishing_mode": "left_fishing"}.get(s, s)

    def up(self, key):
        pass

    # --- souris ------------------------------------------------------------------
    def click(self, box):
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        hit = None
        for name, (bx, by, bw, bh) in self.drawn:
            if name == "rewards_close":
                inside_text = bx <= cx <= bx + bw and by <= cy <= by + bh
                # Fenêtre « Rewards » haute (vente) : elle descend jusqu'au
                # texte, un clic juste au-dessus tombe dedans (sans effet).
                in_window = self.tall_rewards and cy < by and cy > by - 4 * bh
                if not inside_text and not in_window:   # zone vide
                    hit = name
                continue
            if name == "bait_slot_empty":
                bh = bh * 0.7                  # seul le carré « + » réagit, pas le texte
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                hit = name
        self.clicks.append(hit or "?")
        self.events.append(f"clic {hit or '?'}")
        s = self.screen
        if hit == "bait_slot_empty":
            self.origin, self.screen = "slot_empty", "dialog"
        elif hit == "cancel":
            self.screen = self.origin
        elif hit == "switch":
            if self.equip_works:
                self.stock -= 1
                self.screen = "start_panel"
            else:
                self.screen = "slot_empty"
        elif hit == "start_fishing" and s == "panel_unknown":
            pass                                     # sans appât : sans effet
        elif hit == "start_fishing":
            self.screen = "equipped" if s == "start_panel" else "fishing_mode"
        elif hit == "purchase" and s == "dialog":
            self.screen = "shop"
        elif hit == "purchase" and s == "shop_max":
            if self.shells >= 5:
                self.shells -= 5
                if self.confirm_on_buy:
                    self.screen = "buy_confirm"
                else:
                    self.stock = 99
                    self.screen = "buy_rewards"
        elif hit is not None:
            self.screen = {
                "quick_submit": "submit_rewards", "cube_tab": "sell_tab",
                "quick_sell": "sell_confirm", "universal_bait": "shop_item",
                "max_quantity": "shop_max",
            }.get(hit, self.screen)
            if hit == "confirm":
                if s == "sell_confirm":
                    self.shells += self.sell_gives
                    self.screen = "sell_rewards"
                else:
                    self.stock = 99
                    self.screen = "buy_rewards"
            if hit == "rewards_close" and s == "sell_rewards" and self.extra_rewards:
                self.extra_rewards -= 1          # la récompense suivante s'affiche
            elif hit == "rewards_close":
                self.screen = {"submit_rewards": "menu2", "sell_rewards": "menu_after",
                               "buy_rewards": "shop_done"}[s]

    # --- écran ----------------------------------------------------------------------
    def grab(self, region=None):
        canvas = np.full((1000, 1000, 3), 35, np.uint8)
        if self.screen == "equip_banner":
            h, w = self.banner.shape[:2]
            canvas[20:20 + h, 200:200 + w] = self.banner
        if self.ghost_purchase and self.ghost_box and self.screen in ("shop", "shop_item", "shop_max"):
            gx, gy, gw, gh = self.ghost_box
            canvas[gy:gy + gh, gx:gx + gw] = self.images["purchase"]
        self.drawn = []
        shift = 5 * (len(self.clicks) % 8)     # les boutons bougent un peu d'un écran à l'autre
        for i, name in enumerate(self.buttons()):
            img = self.images.get(name)
            if img is None:
                continue                           # bouton retiré par le test
            h, w = img.shape[:2]
            x, y = 250 + 330 * i + shift, 300 + 8 * shift
            canvas[y:y + h, x:x + w] = img
            self.drawn.append((name, (x, y, w, h)))
            if name == "purchase" and self.screen == "dialog":
                self.ghost_box = (x, y, w, h)
        return canvas


PREPARATIONS = ROOT / "templates" / "originals" / "fishing preparations.jpg"


# --------------------------------------------------------------------------- #
# Touche F8 (pause / reprise)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(PREPARATIONS.exists(), "captures de référence absentes")
class RodLevelTest(unittest.TestCase):
    """Canne lue sur le panneau « Fishing Preparations » : plus elle est faible,
    plus le suivi est précis (tolérance, rythme des corrections, imperfections)."""

    def setUp(self):
        self.saved = (hook.human, hook.UI.candidates, hook.UI.locked, hook.UI.on_lock)
        hook.human = Humanizer(HumanConfig())
        hook.UI.on_lock = None

    def tearDown(self):
        hook.human, hook.UI.candidates, hook.UI.locked, hook.UI.on_lock = self.saved

    def test_rod_read_on_real_panel_at_several_scales(self):
        img = cv2.imread(str(PREPARATIONS))                  # Fishing Rod III
        for k, size in ((1.0, (2560, 1600)), (0.675, (1728, 1080))):
            hook.UI.configure(*size)
            frame = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            self.assertEqual(hook.RodDetector(hook.Config()).level(frame), 3, size)

    def test_each_rod_recognised_and_nothing_elsewhere(self):
        hook.UI.configure(2560, 1600)
        rods = hook.RodDetector(hook.Config())
        for level in (1, 2, 3, 4):
            frame = np.full((1600, 2560, 3), 40, np.uint8)
            tpl = cv2.imread(str(ROOT / "templates" / "rods" / f"rod_{level}.png"))
            frame[1000:1000 + tpl.shape[0], 1800:1800 + tpl.shape[1]] = tpl
            self.assertEqual(rods.level(frame), level)
        self.assertIsNone(rods.level(cv2.imread(str(ROOT / "templates" / "originals" / "rewards.jpg"))))

    def test_weaker_rod_means_more_precise_but_still_human(self):
        cfg = hook.Config()
        settings = {}
        for level in (4, 3, 2, 1):
            hook.apply_rod_level(cfg, level)
            settings[level] = (cfg.deadzone_ratio, cfg.press_cooldown[1], hook.human.precision)
        for a, b in ((4, 3), (3, 2), (2, 1)):
            self.assertTrue(all(x > y for x, y in zip(settings[a], settings[b])), (a, b))
        self.assertEqual(settings[4], (0.15, 0.10, 1.0))           # niveau 4 : inchangé
        self.assertGreater(hook.human.precision, 0)                # toujours des imperfections
        factors = {round(hook.human.press_factor(), 3) for _ in range(50)}
        self.assertGreater(len(factors), 10)                        # appuis toujours variés


@unittest.skipUnless(PREPARATIONS.exists(), "captures de référence absentes")
class FrenzyTest(unittest.TestCase):
    """frenzy.bat : panneau -> clic « Start Fishing » (pas F), invite « Fishing » -> F,
    sinon clic près du centre (jamais pile dessus), à environ 2 clics par seconde."""

    def setUp(self):
        self.saved = (hook.grab, hook.search_region, hook.click_box, hook.tap,
                      hook.sleep, hook.human, hook.target)
        self.events, self.waited = [], []
        hook.search_region = lambda: (0, 0, 2560, 1600)
        hook.click_box = lambda box: self.events.append(("click", box))
        hook.tap = lambda key, d: self.events.append(("key", key))
        hook.sleep = lambda s: self.waited.append(s)
        hook.human = Humanizer(HumanConfig())
        hook.target = None

    def tearDown(self):
        (hook.grab, hook.search_region, hook.click_box, hook.tap,
         hook.sleep, hook.human, hook.target) = self.saved

    def run_on(self, frame):
        hook.grab = lambda region=None: frame
        frenzy = hook.Frenzy(hook.Config())
        self.events.clear()
        frenzy.step()
        return frenzy

    def test_panel_clicks_start_fishing_without_f(self):
        self.run_on(cv2.imread(str(PREPARATIONS)))
        self.assertEqual(len(self.events), 1)
        kind, (x, y, w, h) = self.events[0]
        self.assertEqual(kind, "click")
        self.assertGreater(x, 1280)                 # bouton en bas à droite du panneau

    def test_close_banner_is_dismissed_first(self):
        """« Press empty area to close » (vraie capture) : Échap (un second seulement
        si la fenêtre reste) ; au tour suivant, l'écran est réexaminé (puis « Start
        Fishing »)."""
        frames = [cv2.imread(str(ROOT / "templates" / "originals" / "rewards.jpg"))]
        hook.grab = lambda region=None: frames[0]
        frenzy = hook.Frenzy(hook.Config())
        hook.tap = lambda key, d: (self.events.append(("key", key)),
                                   frames.__setitem__(0, np.full((1600, 2560, 3), 60, np.uint8)))
        frenzy.step()
        self.assertEqual(self.events, [("key", "esc")])           # fermée au premier Échap
        self.assertEqual(frenzy.until_check, 0)
        hook.grab = lambda region=None: cv2.imread(str(PREPARATIONS))
        frenzy.step()
        self.assertEqual(self.events[-1][0], "click")              # Start Fishing

    # --- Outils (filet, bombe) : temps simulé ------------------------------------
    @staticmethod
    def game_frame(tools, size=(2560, 1600)):
        """Écran de partie avec les icônes des outils (vraies captures, à l'échelle
        de l'interface pour cette résolution : 0,675 en 1920×1080)."""
        w, h = size
        frame = np.full((h, w, 3), 60, np.uint8)
        scale = h / 1600
        x = int(w * 0.45)
        for tool in tools:
            icon = cv2.imread(str(ROOT / "templates" / "originals" / f"{tool}.png"))
            if scale != 1:
                icon = cv2.resize(icon, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            y = int(h * 0.72)
            frame[y:y + icon.shape[0], x:x + icon.shape[1]] = icon
            x += icon.shape[1] + int(40 * scale)
        return frame

    def play(self, frame, seconds):
        """Fait tourner le mode frenzy sur `frame` pendant `seconds` secondes de
        temps simulé. Renvoie le Frenzy et les événements horodatés."""
        now = [1000.0]
        hook.sleep = lambda s: now.__setitem__(0, now[0] + s)
        hook.click_box = lambda box: self.events.append((now[0], "click", None))
        hook.tap = lambda key, d: self.events.append((now[0], "key", key))
        hook.grab = lambda region=None: frame
        frenzy = hook.Frenzy(hook.Config())
        frenzy.clock = lambda: now[0]
        frenzy.started_at = now[0]
        self.events.clear()
        while now[0] < 1000.0 + seconds:
            frenzy.step()
        return frenzy

    def throws(self):
        """(instant, outil) de chaque lancer : un clic, l'outil étant celui de la
        dernière touche W/E envoyée."""
        out, selected = [], None
        names = {key: tool for tool, (key, _, _) in hook.FRENZY_TOOLS.items()}
        for when, kind, key in self.events:
            if kind == "key":
                selected = names[key]
            else:
                out.append((when, selected))
        return out

    def check_cooldowns(self, throws):
        for tool, (_, cooldown, _) in hook.FRENZY_TOOLS.items():
            times = [w for w, t in throws if t == tool]
            gaps = [b - a for a, b in zip(times, times[1:])]
            # Jamais avant la fin du temps de recharge ; jamais beaucoup plus.
            self.assertTrue(all(cooldown + 0.29 < g < cooldown + 3.0 for g in gaps),
                            (tool, gaps))
        moments = [w for w, _ in throws]
        self.assertTrue(all(b - a >= 0.79 for a, b in zip(moments, moments[1:])), moments)

    def test_net_alone_w_then_one_click_after_each_cooldown(self):
        """Filet seul : W plusieurs fois (l'outil n'est pas toujours prenable tout
        de suite), puis un clic à chaque fin de recharge (5 s + 0,3 à 1 s), sans
        rappuyer sur W ; fin de partie : Échap, et la suivante repart sans filet."""
        frenzy = self.play(self.game_frame(["net"]), 60)
        keys = [k for _, kind, k in self.events if kind == "key"]
        self.assertTrue(5 <= len(keys) <= 8, keys)
        self.assertEqual(set(keys), {hook.FRENZY_TOOLS["net"][0]})
        self.assertEqual([kind for _, kind, _ in self.events[len(keys):]],
                         ["click"] * (len(self.events) - len(keys)))
        throws = self.throws()
        self.assertGreaterEqual(len(throws), 9)
        self.check_cooldowns(throws)
        gaps = [b[0] - a[0] for a, b in zip(throws, throws[1:])]
        self.assertTrue(all(5.3 <= g <= 6.0 for g in gaps), gaps)
        self.assertGreater(len({round(g, 2) for g in gaps}), len(gaps) // 2)   # irrégulier
        self.assertTrue(frenzy.net_mode)
        frenzy.until_check = 0
        hook.grab = lambda region=None: cv2.imread(
            str(ROOT / "templates" / "originals" / "rewards.jpg"))
        hook.tap = lambda key, d: (self.events.append((0, "key", key)),
                                   setattr(hook, "grab", lambda region=None:
                                           np.full((1600, 2560, 3), 60, np.uint8)))
        frenzy.step()
        self.assertEqual(self.events[-1][1:], ("key", "esc"))
        self.assertFalse(frenzy.net_mode)

    def check_net_and_bomb(self, size):
        frenzy = self.play(self.game_frame(["net", "bomb"], size), 90)
        self.assertEqual(frenzy.tools, ["net", "bomb"])
        throws = self.throws()
        self.assertGreaterEqual(len(throws), 12)
        tools = [t for _, t in throws]
        self.assertEqual(tools, (["net", "net", "bomb"] * 20)[:len(tools)])
        self.check_cooldowns(throws)
        # Touche de l'outil seulement quand il change : W avant le filet qui suit
        # une bombe, E avant chaque bombe, rien entre deux filets.
        w, e = hook.FRENZY_TOOLS["net"][0], hook.FRENZY_TOOLS["bomb"][0]
        before, keys = [], []
        for _, kind, key in self.events:
            if kind == "key":
                keys.append(key)
            else:
                before.append(keys)
                keys = []
        self.assertTrue(5 <= len(before[0]) <= 8 and set(before[0]) == {w}, before[0])
        for i, ks in enumerate(before[1:], 1):
            changed = tools[i] != tools[i - 1]
            expected = {e} if tools[i] == "bomb" else {w}
            if changed:
                self.assertTrue(1 <= len(ks) <= 3 and set(ks) == expected, (i, ks))
            else:
                self.assertEqual(ks, [], i)

    def test_net_and_bomb_cycle_two_nets_one_bomb(self):
        """Filet et bombe : deux filets, une bombe, etc. ; chaque outil après sa
        recharge (5 s filet, 10 s bombe) plus une marge aléatoire."""
        self.check_net_and_bomb((2560, 1600))

    def test_net_and_bomb_cycle_in_1920x1080(self):
        hook.search_region = lambda: (0, 0, 1920, 1080)
        hook.UI.configure(1920, 1080)
        try:
            self.check_net_and_bomb((1920, 1080))
        finally:
            hook.UI.configure(2560, 1600)

    def test_bomb_seen_later_joins_the_cycle(self):
        """Bombe vue après le filet (débloquée en cours de partie) : elle rejoint le
        cycle, sans lancer avant la fin des recharges."""
        frames = [self.game_frame(["net"])]
        now = [1000.0]
        hook.sleep = lambda s: now.__setitem__(0, now[0] + s)
        hook.click_box = lambda box: self.events.append((now[0], "click", None))
        hook.tap = lambda key, d: self.events.append((now[0], "key", key))
        hook.grab = lambda region=None: frames[0]
        frenzy = hook.Frenzy(hook.Config())
        frenzy.clock = lambda: now[0]
        self.events.clear()
        while now[0] < 1020:
            frenzy.step()
        frames[0] = self.game_frame(["net", "bomb"])
        while now[0] < 1080:
            frenzy.step()
        self.assertEqual(frenzy.tools, ["net", "bomb"])
        throws = self.throws()
        self.assertIn("bomb", [t for _, t in throws])
        self.check_cooldowns(throws)

    def test_tool_icons_not_seen_elsewhere(self):
        """Ni filet, ni bombe, ni « fish pond is empty » sur les autres écrans
        (panneau, fin de partie, vente, partie avec les outils)."""
        frenzy = hook.Frenzy(hook.Config())
        dets = dict(frenzy.tool_icons, pond=frenzy.pond)
        frames = {name: cv2.imread(str(ROOT / "templates" / "originals" / name))
                  for name in ("fishing preparations.jpg", "rewards.jpg", "sale rewards.jpg",
                               "quick sell confirm.jpg", "switch bait stock.jpg",
                               "gauge forest.png")}
        frames["outils"] = self.game_frame(["net", "bomb"])
        for name, frame in frames.items():
            for what, det in dets.items():
                if not (name == "outils" and what != "pond"):
                    self.assertFalse(det.detect(frame), (what, name))

    # --- « The fish pond is empty » : arrêt -----------------------------------
    @staticmethod
    def pond_frame(size=(2560, 1600)):
        """Écran avec la vraie bannière « The fish pond is empty » (à l'échelle de
        l'interface : 0,675 en 1920×1080)."""
        w, h = size
        frame = np.full((h, w, 3), 70, np.uint8)
        banner = cv2.imread(str(ROOT / "templates" / "originals" / "fish pond empty.png"))
        scale = h / 1600
        if scale != 1:
            banner = cv2.resize(banner, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bw = min(w, banner.shape[1])
        x = (w - bw) // 2
        y = int(h * 0.12)
        frame[y:y + banner.shape[0], x:x + bw] = banner[:, :bw]
        return frame

    def test_pond_empty_stops_frenzy(self):
        """Bannière affichée : le script s'arrête, sans touche ni clic."""
        hook.grab = lambda region=None: self.pond_frame()
        hook.Frenzy(hook.Config()).run()          # revient au lieu de boucler
        self.assertEqual(self.events, [])

    def test_pond_empty_in_1920x1080(self):
        hook.search_region = lambda: (0, 0, 1920, 1080)
        hook.UI.configure(1920, 1080)
        try:
            hook.grab = lambda region=None: self.pond_frame((1920, 1080))
            hook.Frenzy(hook.Config()).run()
            self.assertEqual(self.events, [])
        finally:
            hook.UI.configure(2560, 1600)

    def test_pond_empty_after_start_fishing_no_new_start(self):
        """Bannière juste après « Start Fishing » (message éphémère, panneau encore
        affiché) : arrêt, pas de second « Start Fishing »."""
        frames = [cv2.imread(str(PREPARATIONS))]
        hook.grab = lambda region=None: frames[0]
        hook.click_box = lambda box: (self.events.append(("click", box)),
                                      frames.__setitem__(0, self.pond_frame()))
        hook.Frenzy(hook.Config()).run()
        self.assertEqual([e[0] for e in self.events], ["click"])

    def test_tools_aim_a_little_above_center(self):
        """Lancers (filet, bombe) : un tout petit peu au-dessus du centre."""
        frenzy = hook.Frenzy(hook.Config())
        for size in ((2560, 1600), (1920, 1080)):
            hook.search_region = lambda s=size: (0, 0, *s)
            for _ in range(200):
                x, y, w, h = frenzy.aim_box()
                cx, cy = (x + w / 2) / size[0], (y + h / 2) / size[1]
                self.assertTrue(0.48 <= cx <= 0.52, cx)
                self.assertTrue(0.445 <= cy <= 0.475, cy)
        # Les lancers passent bien par là.
        hook.search_region = lambda: (0, 0, 2560, 1600)
        aimed = []
        real_aim = hook.Frenzy.aim_box
        hook.Frenzy.aim_box = lambda self_: aimed.append(1) or real_aim(self_)
        try:
            self.play(self.game_frame(["net"]), 30)
        finally:
            hook.Frenzy.aim_box = real_aim
        self.assertEqual(len(aimed), len(self.throws()))
        self.assertGreaterEqual(len(aimed), 4)

    @unittest.skipUnless(sys.platform == "win32", "Windows uniquement")
    def test_net_key_is_the_key_labelled_w(self):
        """Le filet se prend avec la touche marquée W du clavier (en bas à gauche en
        AZERTY, scancode 0x2C), pas la position QWERTY du W (marquée Z en AZERTY)."""
        import ctypes
        code = hook.ScancodeKeyboard().scancode(hook.FRENZY_NET_KEY)
        vk = ctypes.WinDLL("user32").MapVirtualKeyW(code, 1)       # MAPVK_VSC_TO_VK
        self.assertEqual(vk, ord("W"))
        self.assertEqual(_saved_inputs[2](hook.FRENZY_NET_KEY), "w")

    def test_prompt_presses_f(self):
        frame = np.full((1600, 2560, 3), 60, np.uint8)
        prompt = cv2.imread(str(ROOT / "templates" / "fishing_prompt.png"))
        frame[900:900 + prompt.shape[0], 1500:1500 + prompt.shape[1]] = prompt
        self.run_on(frame)
        self.assertEqual(self.events, [("key", "f")])

    def test_clicks_near_but_not_at_center(self):
        frame = np.full((1600, 2560, 3), 60, np.uint8)
        hook.grab = lambda region=None: frame
        frenzy = hook.Frenzy(hook.Config())
        for _ in range(200):
            self.waited.clear()
            frenzy.step()
        clicks = [e[1] for e in self.events]
        self.assertEqual(len(clicks), 200)
        centers = [(x + w / 2, y + h / 2) for x, y, w, h in clicks]
        for cx, cy in centers:
            self.assertLess(abs(cx - 1280), 2560 * 0.16)
            self.assertLess(abs(cy - 800), 1600 * 0.16)
        off = [max(abs(cx - 1280) / 2560, abs(cy - 800) / 1600) for cx, cy in centers]
        self.assertGreater(min(off), 0.015)          # jamais pile au centre
        self.assertGreater(len({c for c in clicks}), 150)   # endroits variés


class PauseKeyTest(unittest.TestCase):
    """Simule GetAsyncKeyState : 0x8000 = F8 enfoncée, 0x0001 = pressée depuis la
    dernière lecture."""

    def setUp(self):
        self.saved = (hook._key_state, hook._pause_key_was_down, hook.keyboard)
        hook._pause_key_was_down = False
        hook.keyboard = FakeGame()                     # pour release_all_keys

    def tearDown(self):
        hook._key_state, hook._pause_key_was_down, hook.keyboard = self.saved

    def feed(self, f8_states):
        states = iter(f8_states)
        hook._key_state = lambda vk: next(states, 0) if vk == hook.PAUSE_VK else 0

    def test_press_pauses_and_second_press_resumes(self):
        # appui (pause), maintenu, relâché, attente, nouvel appui (reprise)
        self.feed([0x8001, 0x8000, 0, 0, 0, 0x8001])
        with self.assertRaises(hook.ResumeRequested):
            hook.check_stop()

    def test_brief_press_between_two_reads_is_not_missed(self):
        # F8 pressée puis relâchée entre deux lectures : seul le bit 0x0001 le dit.
        self.feed([0x0001, 0, 0x0001])
        with self.assertRaises(hook.ResumeRequested):
            hook.check_stop()

    def test_holding_f8_does_not_toggle_twice(self):
        hook._key_state = lambda vk: 0x8000 if vk == hook.PAUSE_VK else 0
        hook._pause_key_was_down = True                # déjà enfoncée au dernier passage
        hook.check_stop()                              # pas de pause répétée

    def test_keys_released_during_pause(self):
        hook._held_keys.add("d")
        self.feed([0x8001, 0, 0x8001])
        with self.assertRaises(hook.ResumeRequested):
            hook.check_stop()
        self.assertEqual(hook._held_keys, set())


@unittest.skipUnless(PREPARATIONS.exists() and (ROOT / "templates" / "originals" / "rewards.jpg").exists(),
                     "captures de référence absentes")
class RealScreenshotTest(unittest.TestCase):
    """Sur une vraie capture du panneau « Fishing Preparations » (07/11) : la case
    vide et « Start Fishing » sont reconnues, aucun autre bouton ne l'est."""

    def test_fishing_preparations_panel(self):
        img = cv2.imread(str(PREPARATIONS))
        restocker = hook.Restocker(hook.Config())
        found = {name for name, det in restocker.buttons.items() if det.detect(img)}
        self.assertEqual(found, {"bait_slot_empty", "start_fishing"})
        # L'invite « Fishing » n'est pas sur ce panneau : pas de F en trop.
        self.assertFalse(hook.fishing_prompt_detector(hook.Config()).detect(img))

    def test_switch_bait_with_bait_in_stock(self):
        """Vraie capture du 07/11 à 17:07 : « Switch Bait » avec 99 appâts en stock.
        Le bot doit choisir « Switch » (le bouton à droite), pas « Purchase »."""
        img = cv2.imread(str(ROOT / "templates" / "originals" / "switch bait stock.jpg"))
        restocker = hook.Restocker(hook.Config())
        self.assertTrue(restocker.buttons["switch"].detect(img))
        self.assertFalse(restocker.buttons["purchase"].detect(img))
        saved = hook.grab
        hook.grab = lambda region=None: img
        try:
            self.assertEqual(restocker.wait_for_any(("switch", "purchase"), timeout=2.0), "switch")
        finally:
            hook.grab = saved

    def test_small_purchase_goes_straight_to_rewards(self):
        """Vraie capture du 08/11 à 10:43 : achat de 2 appâts, pas de « Confirm »,
        « Rewards » directement. Le bot doit choisir Rewards (pas attendre Confirm)."""
        img = cv2.imread(str(ROOT / "templates" / "originals" / "rewards small purchase.jpg"))
        restocker = hook.Restocker(hook.Config())
        self.assertFalse(restocker.buttons["confirm"].detect(img))
        saved = hook.grab
        hook.grab = lambda region=None: img
        try:
            self.assertEqual(restocker.wait_for_any(("confirm", "rewards_close"), timeout=2.0),
                             "rewards_close")
        finally:
            hook.grab = saved

    def test_sale_income_read_on_real_captures(self):
        """Vraies captures de fin de vente (21/11) : le total en Fons, sous « Income
        Details », est lu, même à côté d'une autre monnaie, et à d'autres
        résolutions. Rien n'est lu sur la fenêtre de confirmation."""
        hook.ocr_words(np.full((40, 40, 3), 255, np.uint8))
        if hook._ocr_unavailable:
            self.skipTest("OCR Windows indisponible")
        cases = {"sale rewards.jpg": 1972, "sale rewards two currencies.jpg": 1994}
        saved = (hook.UI.candidates, hook.UI.locked, hook.UI.on_lock)
        hook.UI.on_lock = None
        try:
            for size in ((2560, 1600), (1920, 1080)):
                hook.UI.configure(*size)
                restocker = hook.Restocker(hook.Config())
                for name, amount in cases.items():
                    img = cv2.imread(str(ROOT / "templates" / "originals" / name))
                    if size[0] != 2560:
                        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    self.assertEqual(restocker.read_income(img), amount, f"{name} {size}")
            confirm = cv2.imread(str(ROOT / "templates" / "originals" / "quick sell confirm.jpg"))
            self.assertIsNone(restocker.read_income(confirm))
        finally:
            hook.UI.candidates, hook.UI.locked, hook.UI.on_lock = saved

    def test_rewards_popup_after_confirm(self):
        """Vraie capture de la fenêtre « Rewards » (07/11, 16:59) : seul le texte
        « Press empty area to close » est reconnu, et la zone cliquée pour la
        fermer est hors de la fenêtre (vide), au-dessus du texte."""
        img = cv2.imread(str(ROOT / "templates" / "originals" / "rewards.jpg"))
        restocker = hook.Restocker(hook.Config())
        found = {name for name, det in restocker.buttons.items() if det.detect(img)}
        self.assertEqual(found, {"rewards_close"})
        _, (x, y, w, h) = restocker.buttons["rewards_close"].locate(img)
        popup_bottom = 1000          # bas de la fenêtre « Rewards » sur cette capture
        click_top = int(y - 2.6 * h)
        self.assertGreater(click_top, popup_bottom)
        self.assertLess(click_top + int(1.4 * h), y)   # au-dessus du texte


@unittest.skipUnless(PREPARATIONS.exists(), "captures de référence absentes")
class ResolutionTest(unittest.TestCase):
    """Autres résolutions que 2560×1600 (cas du PC en 1920×1080, 08/11) : les
    vraies captures sont réduites comme si le jeu tournait dans cette résolution."""

    SHOTS = {"fishing preparations.jpg": {"bait_slot_empty", "start_fishing"},
             "switch bait stock.jpg": {"switch", "cancel"},
             "rewards.jpg": {"rewards_close"}}
    # Réellement présent sur la capture, mais assombri derrière la fenêtre
    # « Rewards » : le Purchase de la boutique (sans effet, le bot n'y cherche
    # que « Press empty area to close »).
    ALSO_VISIBLE = {"rewards.jpg": {"purchase"}}

    def setUp(self):
        self.saved = (hook.UI.candidates, hook.UI.locked, hook.UI.on_lock)

    def tearDown(self):
        hook.UI.candidates, hook.UI.locked, hook.UI.on_lock = self.saved

    def test_candidates_cover_height_and_width_scaling(self):
        hook.UI.configure(1920, 1080)
        self.assertIn(round(1080 / 1600, 3), hook.UI.candidates)
        self.assertIn(round(1920 / 2560, 3), hook.UI.candidates)
        hook.UI.configure(2560, 1600)
        self.assertEqual(hook.UI.scales((0.9, 1.0, 1.1)), (0.9, 1.0, 1.1))   # inchangé

    def test_parse_resolution(self):
        self.assertEqual(hook.parse_resolution("1920x1080"), (1920, 1080))
        self.assertEqual(hook.parse_resolution("2560×1440"), (2560, 1440))
        with self.assertRaises(ValueError):
            hook.parse_resolution("1080p")
        args = hook.parse_args(["-r", "1920x1080"])
        self.assertEqual(hook.build_config(args).resolution, (1920, 1080))

    def test_default_is_2560x1600_without_scaling(self):
        """Sans -r : 2560×1600 et tolérance d'origine, même si la fenêtre diffère."""
        saved_target = hook.target

        class Window:
            def client_region(self):
                return (0, 0, 1920, 1080)
        hook.target = Window()
        try:
            hook.configure_ui_scale(hook.Config(config_path=None))
            self.assertEqual(hook.UI.scales((0.9, 1.0, 1.1)), (0.9, 1.0, 1.1))
            cfg = hook.Config(config_path=None, resolution=(1920, 1080))
            hook.configure_ui_scale(cfg)
            self.assertIn(round(1080 / 1600, 3), hook.UI.scales((1.0,)))
        finally:
            hook.target = saved_target

    def test_saved_scale_is_reused(self):
        hook.UI.configure(1920, 1080, saved=0.7)
        self.assertEqual(hook.UI.scales((1.0,))[1], 0.7)

    def check_resolution(self, factor):
        hook.UI.configure(1920, 1080)
        locked = []
        hook.UI.on_lock = locked.append
        restocker = hook.Restocker(hook.Config())
        for name, expected in self.SHOTS.items():
            img = cv2.imread(str(ROOT / "templates" / "originals" / name))
            small = cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            found = {n for n, det in restocker.buttons.items() if det.detect(small)}
            found -= self.ALSO_VISIBLE.get(name, set())
            self.assertEqual(found, expected, f"{name} à l'échelle {factor:.3f}")
        self.assertTrue(locked, "échelle non déterminée")
        self.assertAlmostEqual(locked[0], factor, delta=0.05)

    def test_1080p_interface_scaled_by_height(self):
        self.check_resolution(1080 / 1600)

    def test_1080p_interface_scaled_by_width(self):
        self.check_resolution(1920 / 2560)

    def test_fish_banner_at_1080p(self):
        hook.UI.configure(1920, 1080)
        tpl = cv2.imread(str(ROOT / "templates" / "fish_on_hook.png"))
        f = 1080 / 1600
        small = cv2.resize(tpl, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        screen = np.full((400, 1000, 3), 30, np.uint8)
        screen[150:150 + small.shape[0], 200:200 + small.shape[1]] = small
        det = hook.BannerDetector(ROOT / "templates" / "fish_on_hook.png", 0.8, (0.9, 1.0, 1.1))
        self.assertTrue(det.detect(screen))


@unittest.skipUnless(RESTOCK_DIR.is_dir() and EQUIP_BANNER.exists(), "templates absents")
class RestockTest(unittest.TestCase):
    SELL = ["cancel", "quick_submit", "rewards_close", "cube_tab", "quick_sell", "confirm",
            "rewards_close"]
    BUY = ["purchase", "universal_bait", "max_quantity", "purchase", "confirm", "rewards_close"]
    EQUIP = ["switch", "start_fishing"]

    def setUp(self):
        self.shop = FakeShop()
        self.saved = (hook.grab, hook.keyboard, hook.human, hook.target, hook.click_box)
        hook.grab = self.shop.grab
        hook.keyboard = self.shop
        hook.click_box = self.shop.click
        hook.human = Humanizer(HumanConfig(enabled=False, reaction_median=0.0))
        hook.target = None
        cfg = hook.Config(config_path=None)
        cfg.restock_step_delay = (0.0, 0.0)
        cfg.action_press_duration = (0.001, 0.001)
        cfg.start_delay = (0.0, 0.0)
        cfg.banner_timeout = 2.0
        cfg.scan_interval = 0.0
        cfg.restock_step_timeout = 2.0
        cfg.banner_region = (0, 0, 1000, 700)
        self.cfg = cfg

    def tearDown(self):
        (hook.grab, hook.keyboard, hook.human, hook.target, hook.click_box) = self.saved

    def cycle(self, may_restock=True):
        cfg = self.cfg
        no_bait, restocker = hook.restock_tools(cfg)
        banner = hook.BannerDetector(cfg.banner_template, cfg.banner_threshold, cfg.banner_scales)
        return hook.run_cycle(cfg, banner, hook.GaugeDetector(cfg), 1, None, None,
                              no_bait, restocker, may_restock=may_restock,
                              fishing_prompt=hook.fishing_prompt_detector(cfg))

    # --- plus d'appâts : vendre, acheter, équiper ----------------------------------
    def test_no_bait_from_fishing_mode_sells_buys_equips(self):
        """Cas du 07/11 : plus d'appâts en pêchant. Vente d'abord (coquillages),
        puis achat, puis équipement, sans jamais sortir du mode pêche."""
        record = self.cycle()                               # F -> « Equip bait before fishing. »
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.clicks, self.SELL + self.BUY + self.EQUIP)
        self.assertEqual(self.shop.keys, ["f", "e", "q", "esc", "e", "esc", "esc", "e"])
        self.assertNotIn("left_fishing", self.shop.screen)  # jamais sorti du mode pêche
        self.assertEqual(self.shop.screen, "equipped")
        self.assertEqual(self.shop.stock, 98)               # 99 achetés, 1 équipé

    def test_no_bait_from_preparations_panel_buys_directly(self):
        """Cas du 08/11 à 10:14 : sur le panneau de préparation, Q (vente) ne fait
        rien. Le bot achète directement avec ses coquillages, sans Cancel ni Q."""
        self.shop.shells = 100
        self.shop.screen = "slot_empty"
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        slot = ["bait_slot_empty"]
        self.assertEqual(self.shop.clicks, slot + self.BUY + slot + self.EQUIP)
        self.assertNotIn("q", self.shop.keys)
        self.assertEqual(self.shop.screen, "equipped")

    def test_small_purchase_without_confirm_then_equip(self):
        """Cas du 08/11 à 10:43 : petit achat (2 appâts) -> pas de « Confirm »,
        « Rewards » directement. Le bot ferme Rewards, Échap x2, puis équipe."""
        self.shop.shells = 10
        self.shop.confirm_on_buy = False
        self.shop.screen = "slot_empty"
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        buy = [c for c in self.BUY if c != "confirm"]
        slot = ["bait_slot_empty"]
        self.assertEqual(self.shop.clicks, slot + buy + slot + self.EQUIP)
        self.assertEqual(self.shop.screen, "equipped")

    def test_no_bait_no_shells_on_panel_stops_with_clear_message(self):
        self.shop.screen = "slot_empty"
        self.cfg.restock_step_timeout = 0.5
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle()
        self.assertIn("coquillages", str(ctx.exception))
        self.assertNotIn("q", self.shop.keys)

    def test_sell_comes_before_any_purchase(self):
        self.cycle()
        clicks = self.shop.clicks
        self.assertLess(clicks.index("quick_sell"), clicks.index("universal_bait"))
        self.assertGreaterEqual(self.shop.shells, 0)

    def test_two_escapes_right_after_purchase_confirm(self):
        self.cycle()
        events = self.shop.events
        i = max(k for k, e in enumerate(events) if e == "clic confirm")   # confirmation d'achat
        self.assertEqual(events[i + 1:i + 4], ["clic rewards_close", "esc", "esc"])

    def test_first_purchase_still_visible_behind_shop(self):
        """Le 1er Purchase reste affiché derrière la boutique : le bot clique le
        nouveau (après double chevrons), pas celui de derrière."""
        self.shop.ghost_purchase = True
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.clicks, self.SELL + self.BUY + self.EQUIP)
        self.assertEqual(self.shop.stock, 98)

    def test_no_shells_even_after_selling_stops_without_leaving_fishing(self):
        """Vente sans effet (rien à vendre) : l'achat échoue ; le bot referme les
        menus avec Échap tant qu'un menu est ouvert, mais pas un de plus."""
        self.shop.sell_gives = 0
        self.cfg.restock_step_timeout = 0.5
        with self.assertRaises(hook.BlockedStop):
            self.cycle()
        self.assertNotEqual(self.shop.screen, "left_fishing")
        self.assertEqual(self.shop.screen, "fishing_mode")

    # --- appâts en stock : équiper seulement ---------------------------------------
    def test_bait_in_stock_is_equipped_without_buying(self):
        """Cas de 17:07 : 99 appâts en stock, non équipés -> Switch directement."""
        self.shop.stock = 99
        self.shop.screen = "slot_empty"
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.clicks, ["bait_slot_empty"] + self.EQUIP)
        self.assertEqual(self.shop.keys, [])                # ni vente, ni achat, ni Échap
        self.assertEqual(self.shop.screen, "equipped")

    def test_bait_in_stock_from_fishing_mode_with_e(self):
        self.shop.stock = 99
        self.shop.screen = "fishing_mode"
        _, restocker = hook.restock_tools(self.cfg)
        restocker.run()
        self.assertEqual(self.shop.keys, ["e"])
        self.assertEqual(self.shop.clicks, self.EQUIP)

    def test_not_equipped_is_reported(self):
        self.shop.stock = 99
        self.shop.equip_works = False                       # Switch sans effet
        self.shop.screen = "slot_empty"
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle()
        self.assertIn("pas équipés", str(ctx.exception))

    def test_equipped_slot_does_not_trigger(self):
        self.shop.screen = "equipped"
        _, restocker = hook.restock_tools(self.cfg)
        self.assertFalse(restocker.bait_missing())
        self.shop.screen = "slot_empty"
        self.assertTrue(restocker.bait_missing())

    # --- invite « Fishing » -----------------------------------------------------------
    def test_fishing_prompt_presses_f_then_start_fishing(self):
        self.shop.screen, self.shop.after_spot = "spot", "prep_equipped"
        _, restocker = hook.restock_tools(self.cfg)
        self.assertTrue(hook.enter_fishing_mode(
            self.cfg, hook.fishing_prompt_detector(self.cfg), restocker))
        self.assertEqual(self.shop.keys, ["f"])
        self.assertEqual(self.shop.clicks, ["start_fishing"])
        self.assertEqual(self.shop.screen, "fishing_mode")

    def test_launched_on_preparations_panel_with_bait(self):
        """Cas du PC en 1080p : bot lancé sur « Fishing Preparations » avec un appât
        équipé -> « Start Fishing » est cliqué avant d'appuyer sur F."""
        self.shop.screen = "prep_equipped"
        _, restocker = hook.restock_tools(self.cfg)
        self.assertTrue(restocker.start_fishing_if_shown())
        self.assertEqual(self.shop.clicks, ["start_fishing"])
        self.assertEqual(self.shop.screen, "fishing_mode")
        self.assertFalse(restocker.start_fishing_if_shown())    # plus de panneau

    def test_no_fishing_prompt_no_key(self):
        self.shop.screen = "fishing_mode"
        _, restocker = hook.restock_tools(self.cfg)
        self.assertFalse(hook.enter_fishing_mode(
            self.cfg, hook.fishing_prompt_detector(self.cfg), restocker))
        self.assertEqual(self.shop.keys, [])

    def test_fishing_prompt_without_bait_goes_to_restock(self):
        self.shop.shells = 100                        # achat direct depuis le panneau
        self.shop.screen, self.shop.after_spot = "spot", "slot_empty"
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.keys[0], "f")
        self.assertEqual(self.shop.screen, "equipped")

    # --- échecs et sécurité -------------------------------------------------------------
    def test_own_escape_is_not_a_stop_request(self):
        hook._ignore_escape_until = 0.0
        hook.tap("esc", (0.001, 0.001))
        self.assertGreater(hook._ignore_escape_until, hook.time.perf_counter())
        self.assertFalse(hook._escape_pressed())

    def test_failure_does_not_wait_for_focus(self):
        """Échec pendant que le jeu n'est plus au premier plan : arrêt immédiat."""
        class Game:
            title = "NTE"
            lost = False                              # passe en arrière-plan à l'échec

            def is_foreground(self):
                return not self.lost

            def client_region(self):
                return (0, 0, 1000, 1000)
        game = Game()
        hook.target = game
        self.shop.screen = "nothing"                  # E sans effet, rien à l'écran
        self.cfg.restock_step_timeout = 0.3
        restocker = hook.Restocker(self.cfg)
        restocker._save_failure = lambda *a, **k: setattr(game, "lost", True)
        started = hook.time.perf_counter()
        with self.assertRaises(hook.RestockFailed):
            restocker.run()
        self.assertLess(hook.time.perf_counter() - started, 5)
        self.assertEqual(self.shop.keys, ["e"])                  # aucun Échap hors du jeu

    def test_failure_closes_menus_only_while_open(self):
        """Bouton introuvable dans « Switch Bait » : un seul Échap (la fenêtre se
        ferme), pas de second qui ferait quitter le mode pêche."""
        self.shop.stock = 99
        self.shop.images.pop("switch")                      # bouton jamais reconnu
        self.shop.screen = "fishing_mode"
        self.cfg.restock_step_timeout = 0.3
        _, restocker = hook.restock_tools(self.cfg)
        with self.assertRaises(hook.RestockFailed):
            restocker.run()
        self.assertEqual(self.shop.keys, ["e", "esc"])
        self.assertEqual(self.shop.screen, "fishing_mode")

    def azerty_cycle(self):
        saved = hook.layout_char
        hook.layout_char = lambda k: {"a": "q", "q": "a"}.get(k, k)     # AZERTY
        try:
            return self.cycle()
        finally:
            hook.layout_char = saved

    def test_menu_opens_with_the_players_key(self):
        """En AZERTY, le joueur ouvre le menu avec la touche marquée Q (« a » du jeu,
        position QWERTY) : essayée en premier, et retenue."""
        self.shop.menu_key = "a"
        record = self.azerty_cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.keys[:4], ["f", "e", "a", "esc"])
        self.assertEqual(self.cfg.sell_menu_key, "a")

    def test_menu_key_is_checked_and_alternative_remembered(self):
        """Cas du 09/11 : la touche n'ouvre pas le menu du marchand. Le bot le voit
        (pas de Quick Submit), réessaie (appui perdu ?), puis essaie l'autre
        touche, et retient celle qui marche."""
        self.shop.menu_key = "q"
        record = self.azerty_cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.clicks, self.SELL + self.BUY + self.EQUIP)
        self.assertEqual(self.shop.keys[:6], ["f", "e", "a", "a", "q", "esc"])
        self.assertEqual(self.cfg.sell_menu_key, "q")

    def test_menu_never_opens_stops_cleanly(self):
        self.shop.menu_key = "x"                            # aucune touche ne marche
        self.cfg.restock_step_timeout = 0.5
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle()
        self.assertIn("marchand", str(ctx.exception))
        self.assertEqual(self.shop.screen, "fishing_mode")  # toujours en mode pêche

    def test_several_rewards_windows_in_a_row(self):
        """Cas du 11/11 (autre PC) : après la vente, plusieurs « Rewards » se suivent.
        Le bot les ferme une à une au lieu de conclure qu'elle ne se ferme pas."""
        self.shop.extra_rewards = 4
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        sell = self.SELL[:-1] + ["rewards_close"] * 5            # 1 + 4 fenêtres
        self.assertEqual(self.shop.clicks, sell + self.BUY + self.EQUIP)
        self.assertEqual(self.shop.stock, 98)

    def test_tall_rewards_window_closed_by_clicking_elsewhere(self):
        """Cas du 12/11 (autre PC) : la fenêtre « Rewards » de la vente descend jusqu'au
        texte ; un clic au-dessus du texte tombe dedans. Le bot clique ailleurs (sous le
        texte, sur les côtés) et la ferme."""
        self.shop.tall_rewards = True
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.stock, 98)

    def test_unexpected_menu_is_closed_after_banner_timeout(self):
        """Cas du 12/11 : « Fish Market » ouvert par surprise, F n'y fait rien, plus de
        bannière. Le bot ferme le menu (Échap, un seul) au lieu d'attendre en vain."""
        self.shop.screen = "menu"
        record = self.cycle()
        self.assertEqual(record.outcome, "no_banner")
        self.assertEqual(self.shop.keys, ["f", "esc"])
        self.assertEqual(self.shop.screen, "fishing_mode")

    def test_preparations_panel_gets_no_keys(self):
        """Cas du 08/11 (autre PC) : panneau « Fishing Preparations » ouvert, case
        d'appât non reconnue. Là, F, E, R… ne font rien : le bot ne doit envoyer
        aucune touche, et s'arrêter clairement s'il ne trouve pas où cliquer."""
        self.shop.screen = "panel_unknown"
        self.cfg.restock_step_timeout = 0.5
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle()
        self.assertEqual(self.shop.keys, [])
        self.assertIn("seul un clic", str(ctx.exception))

    def test_no_second_restock_in_a_row(self):
        self.shop.screen = "slot_empty"
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle(may_restock=False)                   # déjà tenté au cycle précédent
        self.assertIn("coquillages", str(ctx.exception))
        self.assertEqual(self.shop.clicks, [])


if __name__ == "__main__":
    unittest.main()
