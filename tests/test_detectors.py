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

from hook import BannerDetector, Config, GaugeController, GaugeDetector  # noqa: E402

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


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.ctrl = GaugeController(self.cfg)
        self.det = GaugeDetector(self.cfg)

    def decide(self, yellow_x):
        return self.ctrl.decide(self.det.detect(make_gauge(150, 250, yellow_x)))

    def test_left_of_zone_presses_right(self):
        self.assertEqual(self.decide(100), "d")

    def test_right_of_zone_presses_left(self):
        self.assertEqual(self.decide(300), "q")

    def test_centered_no_press(self):
        self.assertIsNone(self.decide(200))

    def test_invert(self):
        self.cfg.invert_controls = True
        self.assertEqual(self.decide(100), "q")


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


if __name__ == "__main__":
    unittest.main()
