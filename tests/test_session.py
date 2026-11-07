"""Tests du comportement humain, des sessions, des statistiques, de l'apprentissage
du contrôleur et d'un cycle complet contre un faux jeu (sans écran ni clavier réels).

Lancer : python -m unittest discover -s tests
"""

import random
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hook  # noqa: E402
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
        self.assertEqual(self.ctrl.decide(state(100))[0], "q")

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
GREEN_BGR = (60, 200, 60)
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
        elif key == "q":
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
RESTOCK_DIR = Path(__file__).resolve().parents[1] / "templates" / "restock"
NO_BAIT = Path(__file__).resolve().parents[1] / "templates" / "no_bait.png"


def greyed(img):
    """Icône sur onglet gris (non sélectionné) au lieu d'orange."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.min()) / max(1.0, g.max() - g.min()) * 70 + 40
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


class FakeShop:
    """Menus du jeu : chaque écran montre un bouton ; un clic dessus (et
    seulement dessus) ou la bonne touche fait passer à l'écran suivant."""

    SCREENS = ["no_bait", "menu", "quick_submit", "cube_tab", "quick_sell", "sold",
               "shop", "universal_bait", "max_quantity", "purchase", "bought", "fishing"]
    KEYS = {"no_bait": "q", "sold": "esc", "shop_closed": "r", "bought": "esc"}

    def __init__(self):
        self.screen = "idle"
        self.keys, self.clicks = [], []
        self.button_box = None
        self.images = {p.stem: cv2.imread(str(p)) for p in RESTOCK_DIR.glob("*.png")}
        self.images["cube_tab"] = greyed(self.images["cube_tab"])
        self.no_bait = cv2.imread(str(NO_BAIT))

    # --- clavier ---
    name = "fake"

    def prepare(self, keys):
        pass

    def down(self, key):
        self.keys.append(key)
        transitions = {("idle", "f"): "no_bait", ("no_bait", "q"): "quick_submit",
                       ("sold", "esc"): "closed", ("closed", "r"): "universal_bait",
                       ("bought", "esc"): "fishing"}
        self.screen = transitions.get((self.screen, key), self.screen)

    def up(self, key):
        pass

    # --- souris ---
    def click(self, box):
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        bx, by, bw, bh = self.button_box
        self.clicks.append(self.screen)
        if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
            return  # clic à côté : rien ne se passe
        nxt = {"quick_submit": "cube_tab", "cube_tab": "quick_sell", "quick_sell": "sold",
               "universal_bait": "max_quantity", "max_quantity": "purchase",
               "purchase": "bought"}
        self.screen = nxt[self.screen]

    # --- écran ---
    def grab(self, region=None):
        canvas = np.full((700, 1000, 3), 35, np.uint8)
        if self.screen == "no_bait":
            h, w = self.no_bait.shape[:2]
            canvas[300:300 + h, 200:200 + w] = self.no_bait
        elif self.screen in self.images:
            img = self.images[self.screen]
            h, w = img.shape[:2]
            x, y = 300 + 7 * len(self.clicks), 250 + 11 * len(self.clicks)
            canvas[y:y + h, x:x + w] = img
            self.button_box = (x, y, w, h)
        return canvas


@unittest.skipUnless(RESTOCK_DIR.is_dir() and NO_BAIT.exists(), "templates absents")
class RestockTest(unittest.TestCase):
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

    def cycle(self, restocker):
        cfg = self.cfg
        no_bait, _ = hook.restock_tools(cfg)
        banner = hook.BannerDetector(cfg.banner_template, cfg.banner_threshold, cfg.banner_scales)
        return hook.run_cycle(cfg, banner, hook.GaugeDetector(cfg), 1, None, None,
                              no_bait, restocker)

    def test_full_restock_sequence(self):
        record = self.cycle(hook.Restocker(self.cfg))
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.screen, "fishing")       # tout le parcours effectué
        self.assertEqual(self.shop.keys, ["f", "q", "esc", "r", "esc"])
        self.assertEqual(self.shop.clicks, ["quick_submit", "cube_tab", "quick_sell",
                                            "universal_bait", "max_quantity", "purchase"])

    def test_own_escape_is_not_a_stop_request(self):
        hook._ignore_escape_until = 0.0
        hook.tap("esc", (0.001, 0.001))
        self.assertGreater(hook._ignore_escape_until, hook.time.perf_counter())
        self.assertFalse(hook._escape_pressed())

    def test_missing_button_aborts_and_closes_menus(self):
        self.shop.images.pop("purchase")                    # bouton jamais affiché
        self.cfg.restock_step_timeout = 0.3
        with self.assertRaises(hook.BlockedStop):
            self.cycle(hook.Restocker(self.cfg))
        self.assertEqual(self.shop.keys[-2:], ["esc", "esc"])

    def test_no_second_restock_in_a_row(self):
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle(None)                                # déjà tenté au cycle précédent
        self.assertIn("coquillages", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
