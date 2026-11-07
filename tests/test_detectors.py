"""Tests hors écran des détecteurs et du correcteur, sur images synthétiques.

Lancer : python -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hook  # noqa: E402
from hook import (  # noqa: E402
    BannerDetector, Config, GaugeController, GaugeDetector,
    hsv_range_from_sample, load_config_file, save_config_file, vivid_pixels,
)

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "fish_on_hook.png"

GREEN_BGR = (60, 200, 60)
YELLOW_BGR = (0, 230, 255)
GREY_BGR = (60, 60, 60)


def make_gauge(green_x0: int, green_x1: int, yellow_x: int | None) -> np.ndarray:
    """Jauge 400x60 : fond sombre, barre grise, zone verte, ligne jaune."""
    img = np.full((60, 400, 3), 20, np.uint8)
    cv2.rectangle(img, (20, 25), (380, 35), GREY_BGR, -1)
    cv2.rectangle(img, (green_x0, 25), (green_x1, 35), GREEN_BGR, -1)
    if yellow_x is not None:
        cv2.rectangle(img, (yellow_x - 1, 18), (yellow_x + 1, 42), YELLOW_BGR, -1)
    return img


class GaugeDetectorTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.detector = GaugeDetector(self.cfg)

    def test_detects_green_and_yellow(self):
        state = self.detector.detect(make_gauge(150, 250, 120))
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state.green_left, 150, delta=2)
        self.assertAlmostEqual(state.green_right, 251, delta=2)
        self.assertAlmostEqual(state.yellow_x, 120, delta=2)

    def test_yellow_inside_green_does_not_split_bar(self):
        state = self.detector.detect(make_gauge(150, 250, 200))
        self.assertAlmostEqual(state.green_left, 150, delta=2)
        self.assertAlmostEqual(state.green_right, 251, delta=2)
        self.assertAlmostEqual(state.yellow_x, 200, delta=2)

    def test_no_gauge(self):
        self.assertIsNone(self.detector.detect(np.full((60, 400, 3), 20, np.uint8)))

    def test_static_yellow_icon_does_not_bias_line(self):
        """Une icône jaune trapue à gauche (bug « ne va qu'à droite ») est ignorée :
        seule la ligne verticale compte."""
        img = make_gauge(150, 250, 220)
        cv2.rectangle(img, (25, 20), (45, 40), YELLOW_BGR, -1)   # icône 21x21
        state = self.detector.detect(img)
        self.assertAlmostEqual(state.yellow_x, 220, delta=2)

    def test_green_without_line_is_not_a_gauge(self):
        self.assertIsNone(self.detector.detect(make_gauge(150, 250, None)))
        state = self.detector.detect(make_gauge(150, 250, None), search=False)
        self.assertIsNotNone(state)
        self.assertIsNone(state.yellow_x)

    def test_search_ignores_green_scenery_without_yellow(self):
        """Plein écran : une grande bande verte du décor (sans ligne jaune) est
        ignorée au profit de la vraie jauge, plus petite mais avec indicateur."""
        screen = np.full((400, 1200, 3), 20, np.uint8)
        cv2.rectangle(screen, (50, 50), (650, 80), GREEN_BGR, -1)        # décor
        screen[300:360, 700:1100] = make_gauge(150, 250, 200)             # jauge
        state = self.detector.detect(screen, search=True)
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state.green_left, 850, delta=2)
        self.assertAlmostEqual(state.yellow_x, 900, delta=2)

    def test_search_without_yellow_finds_nothing(self):
        screen = np.full((400, 1200, 3), 20, np.uint8)
        cv2.rectangle(screen, (50, 50), (650, 80), GREEN_BGR, -1)
        self.assertIsNone(self.detector.detect(screen, search=True))

    def test_gauge_lock_region_contains_bar(self):
        state = hook.GaugeState(850, 951, 325, 336, 900)
        left, top, w, h = hook.gauge_lock_region(self.cfg, state, (0, 0, 1200, 400))
        self.assertEqual((left, w), (0, 1200))   # toute la largeur : la zone verte bouge
        self.assertLessEqual(top, 325)
        self.assertGreaterEqual(top + h, 336)


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.ctrl = GaugeController(self.cfg)
        self.det = GaugeDetector(self.cfg)

    def decide(self, yellow_x):
        return self.ctrl.decide(self.det.detect(make_gauge(150, 250, yellow_x)))

    def test_left_of_zone_presses_right(self):
        self.assertEqual(self.decide(100)[0], "d")

    def test_right_of_zone_presses_left(self):
        self.assertEqual(self.decide(300)[0], "q")

    def test_centered_no_press(self):
        self.assertIsNone(self.decide(200))

    def test_inside_zone_but_off_center_corrects(self):
        """La ligne suit le centre de la zone, pas seulement ses bords."""
        self.assertEqual(self.decide(240)[0], "q")
        self.assertEqual(self.decide(160)[0], "d")

    def test_longer_press_when_further(self):
        lo, hi = self.cfg.press_duration
        near = max(self.decide(170)[1] for _ in range(20))
        far = min(self.decide(30)[1] for _ in range(20))
        self.assertLess(near, far)
        self.assertTrue(lo <= near <= hi and lo <= far <= hi)

    def test_invert(self):
        self.cfg.invert_controls = True
        self.assertEqual(self.decide(100)[0], "q")

    def test_anticipates_moving_green_zone(self):
        """Zone verte qui part vers la droite : la ligne au centre doit déjà
        pousser à droite pour ne pas prendre de retard."""
        state0 = hook.GaugeState(150, 250, 25, 36, 200)
        state1 = hook.GaugeState(170, 270, 25, 36, 220)
        self.ctrl.observe(state0, 0.0)
        self.ctrl.observe(state1, 0.05)   # 400 px/s vers la droite
        self.assertGreater(self.ctrl.velocity, 0)
        self.assertGreater(self.ctrl.target(state1), 220)
        self.assertEqual(self.ctrl.decide(state1)[0], "d")


class BannerDetectorTest(unittest.TestCase):
    def test_template_match(self):
        screen = np.full((300, 800, 3), 30, np.uint8)
        cv2.putText(screen, "Fish on the hook! Press the button to reel it!", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        template = screen[120:165, 30:740]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "banner.png"
            cv2.imwrite(str(path), template)
            det = BannerDetector(path, 0.8, (1.0,))
            self.assertTrue(det.detect(screen))
            self.assertFalse(det.detect(np.full((300, 800, 3), 30, np.uint8)))

    @unittest.skipUnless(TEMPLATE.exists(), "template absent")
    def test_real_template_found_on_other_background(self):
        """Le template réel (texte seul) est retrouvé sur un fond différent."""
        tpl = cv2.imread(str(TEMPLATE))
        h, w = tpl.shape[:2]
        screen = np.zeros((h + 200, w + 300, 3), np.uint8)
        screen[:] = (180, 140, 90)  # fond uniforme quelconque
        screen[100:100 + h, 150:150 + w] = tpl
        det = BannerDetector(TEMPLATE, 0.8, (0.9, 1.0, 1.1))
        score, box = det.locate(screen)
        self.assertGreater(score, 0.95)
        self.assertEqual(box[:2], (150, 100))
        self.assertLess(det.score(np.full_like(screen, 128)), 0.8)


class ScreenCaptureTest(unittest.TestCase):
    def test_grab_returns_bgr_image(self):
        try:
            frame = hook.grab((0, 0, 64, 32))
        except Exception as exc:  # pas d'écran (CI, session distante)
            self.skipTest(f"capture indisponible : {exc}")
        self.assertEqual(frame.shape, (32, 64, 3))


class SetupHelpersTest(unittest.TestCase):
    def test_hsv_range_covers_sample(self):
        sample = np.zeros((10, 10, 3), np.uint8)
        sample[..., 0] = np.linspace(55, 65, 100).reshape(10, 10)
        sample[..., 1] = 200
        sample[..., 2] = 180
        lower, upper = hsv_range_from_sample(sample)
        self.assertLessEqual(lower[0], 55)
        self.assertGreaterEqual(upper[0], 65)
        self.assertLessEqual(lower[1], 200)
        self.assertLessEqual(lower[2], 180)

    def test_setup_ranges_detect_gauge(self):
        """Plages mesurées comme dans --setup -> la jauge est détectée."""
        img = make_gauge(150, 250, 120)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cfg = Config()
        cfg.green_lower, cfg.green_upper = hsv_range_from_sample(hsv[27:34, 160:240])
        # Rectangle autour de la ligne jaune, qui contient aussi du gris et du fond.
        cfg.yellow_lower, cfg.yellow_upper = hsv_range_from_sample(
            vivid_pixels(hsv[15:45, 116:125]))
        state = GaugeDetector(cfg).detect(img)
        self.assertIsNotNone(state)
        self.assertAlmostEqual(state.yellow_x, 120, delta=2)

    def test_config_roundtrip(self):
        cfg = Config()
        cfg.gauge_region = (1, 2, 3, 4)
        cfg.green_lower = (50, 60, 70)
        cfg.invert_controls = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            save_config_file(cfg, path)
            loaded = Config()
            self.assertTrue(load_config_file(loaded, path))
        self.assertEqual(loaded.gauge_region, (1, 2, 3, 4))
        self.assertEqual(loaded.green_lower, (50, 60, 70))
        self.assertIsNone(loaded.banner_region)
        self.assertTrue(loaded.invert_controls)


@unittest.skipUnless(sys.platform == "win32", "Windows uniquement")
class GameWindowTest(unittest.TestCase):
    def test_foreground_window_has_client_region(self):
        hwnd = hook.GameWindow.foreground_hwnd()
        if not hwnd:
            self.skipTest("aucune fenêtre au premier plan")
        window = hook.GameWindow(hwnd)
        self.assertTrue(window.is_foreground())
        region = window.client_region()
        if region is not None:
            self.assertGreater(region[2], 0)
            self.assertGreater(region[3], 0)
        self.assertIn(window.is_elevated(), (True, False, None))

    def test_no_target_means_full_screen(self):
        self.assertIsNone(hook.target)
        self.assertEqual(hook.search_region(), hook.full_screen_region())


@unittest.skipUnless(sys.platform == "win32", "Windows uniquement")
class ScancodeKeyboardTest(unittest.TestCase):
    def test_scancodes_resolved(self):
        kb = hook.ScancodeKeyboard()
        codes = {k: kb.scancode(k) for k in "fqd"}
        self.assertTrue(all(codes.values()))
        self.assertEqual(len(set(codes.values())), 3)


if __name__ == "__main__":
    unittest.main()
