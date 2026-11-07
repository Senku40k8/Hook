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


class FakeShop:
    """Menus du jeu : chaque écran montre un bouton ; un clic dans sa zone
    cliquable (et seulement là) ou la bonne touche fait passer à l'écran suivant."""

    # Comme dans le jeu (captures du 07/11) :
    # - panneau « Fishing Preparations » : clic sur la case vide -> « Switch Bait » ;
    # - en mode pêche, E ouvre directement « Switch Bait » (Cancel / Purchase) ;
    # - la bannière « Equip bait » n'ouvre rien ;
    # - après « Switch », le panneau propose « Start Fishing ».
    CLICKS = {"slot_empty": "purchase1",
              "purchase1": "universal_bait", "universal_bait": "max_quantity",
              "max_quantity": "purchase2", "purchase2": "confirm", "confirm": "rewards",
              "switch": "start_fishing", "start_fishing": "equipped"}
    KEYS = {("idle", "f"): "equip_banner", ("equip_banner", "e"): "purchase1",
            ("bought", "esc"): "bought_esc", ("bought_esc", "esc"): "fishing_mode",
            ("fishing_mode", "e"): "switch"}
    IMAGE = {"slot_empty": "bait_slot_empty",
             "purchase1": "purchase", "purchase2": "purchase",
             "rewards": "rewards_close"}

    def __init__(self, ghost_purchase=False):
        self.screen = "idle"
        self.keys, self.clicks, self.events = [], [], []
        self.button_box = None
        # Le 1er « Purchase » reste visible derrière la boutique (cas réel).
        self.ghost_purchase = ghost_purchase
        self.ghost_box = None
        self.images = {p.stem: cv2.imread(str(p)) for p in RESTOCK_DIR.glob("*.png")}
        self.images["equipped"] = equipped_slot(self.images["bait_slot_empty"])
        self.banner = cv2.imread(str(EQUIP_BANNER))

    # --- clavier ---
    name = "fake"

    def prepare(self, keys):
        pass

    def down(self, key):
        self.keys.append(key)
        self.events.append(key)
        self.screen = self.KEYS.get((self.screen, key), self.screen)

    def up(self, key):
        pass

    # --- souris ---
    def click(self, box):
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        bx, by, bw, bh = self.button_box
        self.clicks.append(self.screen)
        self.events.append("clic " + self.screen)
        if self.screen == "rewards":
            # « Press empty area to close » : seul un clic hors du texte, au-dessus
            # (zone vide), ferme la fenêtre. Échap n'a aucun effet.
            if cy < by:
                self.screen = "bought"
            return
        if self.screen not in self.CLICKS:
            return             # rien de cliquable sur cet écran
        if self.IMAGE.get(self.screen) == "bait_slot_empty":
            bh = bh * 0.7      # case d'appât : seul le carré « + » réagit, pas le texte
        if bx <= cx <= bx + bw and by <= cy <= by + bh:
            self.screen = self.CLICKS[self.screen]

    # --- écran ---
    def grab(self, region=None):
        canvas = np.full((1000, 1000, 3), 35, np.uint8)
        name = self.IMAGE.get(self.screen, self.screen)
        if self.screen == "equip_banner":
            h, w = self.banner.shape[:2]
            canvas[20:20 + h, 200:200 + w] = self.banner
        if self.ghost_purchase and self.ghost_box and self.screen in (
                "universal_bait", "max_quantity", "purchase2"):
            gx, gy, gw, gh = self.ghost_box
            canvas[gy:gy + gh, gx:gx + gw] = self.images["purchase"]
        if name in self.images:
            img = self.images[name]
            h, w = img.shape[:2]
            x, y = 300 + 7 * len(self.clicks), 100 + 70 * len(self.clicks)
            canvas[y:y + h, x:x + w] = img
            self.button_box = (x, y, w, h)
            if self.screen == "purchase1":
                self.ghost_box = self.button_box
        return canvas


PREPARATIONS = ROOT / "templates" / "originals" / "fishing preparations.jpg"


# --------------------------------------------------------------------------- #
# Touche F8 (pause / reprise)
# --------------------------------------------------------------------------- #
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


@unittest.skipUnless(RESTOCK_DIR.is_dir() and EQUIP_BANNER.exists(), "templates absents")
class RestockTest(unittest.TestCase):
    EXPECTED_CLICKS = ["slot_empty", "purchase1", "universal_bait", "max_quantity",
                       "purchase2", "confirm", "rewards", "switch", "start_fishing"]

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

    def test_empty_slot_before_cycle_buys_and_equips_bait(self):
        self.shop.screen = "slot_empty"
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.screen, "equipped")      # tout le parcours effectué
        self.assertEqual(self.shop.clicks, self.EXPECTED_CLICKS)
        # F jamais pressé (pas de pêche sans appât) ; E rouvre le panneau après les Échap.
        self.assertEqual(self.shop.keys, ["esc", "esc", "e"])

    def test_two_escapes_right_after_purchase(self):
        self.shop.screen = "slot_empty"
        self.cycle()
        events = self.shop.events
        i = events.index("clic purchase2")
        self.assertEqual(events[i + 1:i + 5], ["clic confirm", "clic rewards", "esc", "esc"])

    def test_first_purchase_still_visible_behind_shop(self):
        """Bug « bloqué après double chevrons » : le 1er Purchase reste affiché
        derrière la boutique ; le bot doit cliquer le nouveau, puis Échap x2."""
        self.shop.ghost_purchase = True
        self.shop.screen = "slot_empty"
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.clicks, self.EXPECTED_CLICKS)
        self.assertEqual(self.shop.screen, "equipped")
        events = self.shop.events
        i = events.index("clic max_quantity")
        self.assertEqual(events[i + 1:i + 6],
                         ["clic purchase2", "clic confirm", "clic rewards", "esc", "esc"])

    def test_purchase_failure_still_escapes_twice(self):
        self.shop.CLICKS = dict(FakeShop.CLICKS, max_quantity="nothing")  # Purchase n'apparaît pas
        self.shop.screen = "slot_empty"
        self.cfg.restock_step_timeout = 0.4
        with self.assertRaises(hook.BlockedStop):
            self.cycle()
        events = self.shop.events
        i = events.index("clic max_quantity")
        self.assertEqual(events[i + 1:], ["esc", "esc"])

    def test_equipped_slot_does_not_trigger(self):
        self.shop.screen = "equipped"
        _, restocker = hook.restock_tools(self.cfg)
        self.assertFalse(restocker.bait_missing())
        self.shop.screen = "slot_empty"
        self.assertTrue(restocker.bait_missing())

    def test_equip_bait_banner_triggers_restock(self):
        record = self.cycle()                               # F -> « Equip bait before fishing. »
        self.assertEqual(record.outcome, "restock")
        # F, puis E qui ouvre directement « Switch Bait » (cas du blocage) : pas de
        # case à cliquer, on enchaîne sur Purchase.
        self.assertEqual(self.shop.keys, ["f", "e", "esc", "esc", "e"])
        self.assertEqual(self.shop.clicks, self.EXPECTED_CLICKS[1:])
        self.assertEqual(self.shop.screen, "equipped")

    def _spot(self, after_f):
        """Point de pêche : invite « Fishing » ; F ouvre le panneau de préparation."""
        self.shop.images["spot"] = cv2.imread(str(ROOT / "templates" / "fishing_prompt.png"))
        self.shop.images["prep_equipped"] = self.shop.images["start_fishing"]
        self.shop.KEYS = {**FakeShop.KEYS, ("spot", "f"): after_f}
        self.shop.CLICKS = dict(FakeShop.CLICKS, prep_equipped="fishing_mode")
        self.shop.screen = "spot"

    def test_fishing_prompt_presses_f_then_start_fishing(self):
        self._spot("prep_equipped")
        _, restocker = hook.restock_tools(self.cfg)
        entered = hook.enter_fishing_mode(self.cfg, hook.fishing_prompt_detector(self.cfg),
                                          restocker)
        self.assertTrue(entered)
        self.assertEqual(self.shop.keys, ["f"])
        self.assertEqual(self.shop.clicks, ["prep_equipped"])
        self.assertEqual(self.shop.screen, "fishing_mode")

    def test_no_fishing_prompt_no_key(self):
        self.shop.screen = "fishing_mode"
        _, restocker = hook.restock_tools(self.cfg)
        self.assertFalse(hook.enter_fishing_mode(
            self.cfg, hook.fishing_prompt_detector(self.cfg), restocker))
        self.assertEqual(self.shop.keys, [])

    def test_fishing_prompt_without_bait_goes_to_restock(self):
        self._spot("slot_empty")                     # F -> panneau avec case vide
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.keys[0], "f")
        self.assertEqual(self.shop.clicks, self.EXPECTED_CLICKS)

    def test_bait_in_stock_is_equipped_without_buying(self):
        """Bug du 07/11 à 17:07 : 99 appâts en stock mais non équipés. « Switch
        Bait » affiche Cancel / Switch (pas Purchase) : le bot doit cliquer
        Switch, sans rien acheter."""
        self.shop.CLICKS = dict(FakeShop.CLICKS, slot_empty="switch")
        self.shop.screen = "slot_empty"
        record = self.cycle()
        self.assertEqual(record.outcome, "restock")
        self.assertEqual(self.shop.clicks, ["slot_empty", "switch", "start_fishing"])
        self.assertEqual(self.shop.keys, [])                # ni achat, ni Échap
        self.assertEqual(self.shop.screen, "equipped")

    def test_bait_in_stock_from_fishing_mode_with_e(self):
        """En mode pêche (pas de panneau) : E ouvre « Switch Bait », puis Switch."""
        self.shop.KEYS = {**FakeShop.KEYS, ("fishing_mode", "e"): "switch"}
        self.shop.screen = "fishing_mode"
        _, restocker = hook.restock_tools(self.cfg)
        restocker.run()
        self.assertEqual(self.shop.keys, ["e"])
        self.assertEqual(self.shop.clicks, ["switch", "start_fishing"])

    def test_bought_but_not_equipped_is_reported(self):
        """Après Switch, la case est toujours vide : le bot ne repart pas pêcher
        comme si de rien n'était, il signale l'échec."""
        self.shop.CLICKS = dict(FakeShop.CLICKS, switch="slot_empty")
        self.shop.screen = "slot_empty"
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle()
        self.assertIn("pas équipés", str(ctx.exception))

    def test_own_escape_is_not_a_stop_request(self):
        hook._ignore_escape_until = 0.0
        hook.tap("esc", (0.001, 0.001))
        self.assertGreater(hook._ignore_escape_until, hook.time.perf_counter())
        self.assertFalse(hook._escape_pressed())

    def test_failure_does_not_wait_for_focus(self):
        """Échec pendant que le jeu n'est plus au premier plan : arrêt immédiat,
        sans attendre le jeu pour envoyer Échap (le « bloqué » du log)."""
        class Game:
            title = "NTE"
            lost = False                              # passe en arrière-plan à l'échec

            def is_foreground(self):
                return not self.lost

            def client_region(self):
                return (0, 0, 1000, 1000)
        game = Game()
        hook.target = game
        self.shop.screen = "idle_no_panel"           # case jamais visible, E sans effet
        self.cfg.restock_step_timeout = 0.3
        restocker = hook.Restocker(self.cfg)
        restocker._save_failure = lambda *a: setattr(game, "lost", True)
        started = hook.time.perf_counter()
        with self.assertRaises(hook.RestockFailed):
            restocker.run()
        self.assertLess(hook.time.perf_counter() - started, 5)   # pas d'attente du jeu
        self.assertEqual(self.shop.keys, ["e"])                  # aucun Échap hors du jeu

    def test_missing_button_aborts_and_closes_menus(self):
        self.shop.images.pop("switch")                      # bouton jamais affiché
        self.shop.screen = "slot_empty"
        self.cfg.restock_step_timeout = 0.3
        with self.assertRaises(hook.BlockedStop):
            self.cycle()
        self.assertEqual(self.shop.keys[-2:], ["esc", "esc"])

    def test_no_second_restock_in_a_row(self):
        self.shop.screen = "slot_empty"
        with self.assertRaises(hook.BlockedStop) as ctx:
            self.cycle(may_restock=False)                   # déjà tenté au cycle précédent
        self.assertIn("coquillages", str(ctx.exception))
        self.assertEqual(self.shop.clicks, [])


if __name__ == "__main__":
    unittest.main()
