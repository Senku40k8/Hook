"""
hook.py - Boucle de test d'interface automatisée (séquence "pêche").

Cycle exécuté en boucle :
    1. Appui sur F puis courte latence aléatoire (0.5 s - 1.5 s).
    2. Surveillance de l'écran jusqu'à l'apparition de la bannière
       "Fish on the hook! Press the button to reel it!", puis nouvel appui sur F.
    3. Suivi en temps réel de la jauge (zone verte + indicateur jaune) :
       pressions courtes sur Q (gauche) / D (droite) pour garder l'indicateur
       dans la zone verte, jusqu'à la disparition de la jauge.
    4. Pause aléatoire (3.0 s - 4.5 s) puis retour à l'étape 1.

Arrêt propre :
    - Touche Échap (vérifiée en permanence, y compris pendant les pauses).
    - Fail-safe PyAutoGUI : amener la souris dans un coin de l'écran.
    - Ctrl+C dans le terminal.

Usage :
    python hook.py                    # lance la boucle
    python hook.py --cycles 5         # s'arrête après 5 cycles
    python hook.py --calibrate        # capture + masques de debug pour régler les couleurs
    python hook.py --gauge-region 600 900 700 80 --banner-region 400 200 1100 200
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pyautogui

log = logging.getLogger("hook")

Region = Tuple[int, int, int, int]  # (left, top, width, height) en pixels écran
HSV = Tuple[int, int, int]
Range = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- Touches -----------------------------------------------------------
    key_action: str = "f"
    key_left: str = "q"
    key_right: str = "d"
    # Mettre à True si Q/D font bouger l'indicateur dans le sens opposé.
    invert_controls: bool = False

    # --- Zones de capture (None = écran principal entier) ------------------
    # Restreindre ces zones accélère la capture et évite les faux positifs.
    banner_region: Optional[Region] = None
    gauge_region: Optional[Region] = None

    # --- Détection de la bannière (template matching) ----------------------
    banner_template: Path = Path(__file__).with_name("templates") / "fish_on_hook.png"
    banner_threshold: float = 0.80          # score TM_CCOEFF_NORMED minimal
    banner_scales: Tuple[float, ...] = (0.9, 1.0, 1.1)  # tolérance d'échelle

    # --- Détection de la jauge (seuillage HSV, échelle OpenCV H:0-179) -----
    green_lower: HSV = (40, 80, 80)
    green_upper: HSV = (85, 255, 255)
    yellow_lower: HSV = (20, 120, 150)
    yellow_upper: HSV = (35, 255, 255)
    gauge_min_width: int = 30               # largeur min. de la zone verte (px)
    gauge_min_aspect: float = 2.5           # largeur/hauteur min. (barre horizontale)
    gauge_bridge_px: int = 15               # comble la coupure créée par la ligne jaune
    yellow_min_pixels: int = 6              # nb de pixels jaunes min. pour valider l'indicateur
    deadzone_ratio: float = 0.20            # marge intérieure (fraction de la largeur verte)

    # --- Temporisations (secondes, bornes min/max des tirages aléatoires) ---
    start_delay: Range = (0.5, 1.5)         # après le premier F
    reaction_delay: Range = (0.12, 0.30)    # entre la bannière et le second F
    press_duration: Range = (0.03, 0.08)    # durée d'une pression Q/D
    press_cooldown: Range = (0.04, 0.10)    # intervalle min. entre deux corrections
    cycle_pause: Range = (3.0, 4.5)         # fin de cycle
    startup_countdown: float = 3.0          # temps pour remettre le focus sur la fenêtre

    # --- Délais de garde ----------------------------------------------------
    banner_timeout: float = 60.0            # abandon du cycle si pas de bannière
    gauge_appear_timeout: float = 10.0      # abandon si la jauge n'apparaît pas
    gauge_lost_grace: float = 0.6           # absence continue = jauge disparue
    gauge_max_duration: float = 120.0       # sécurité : durée max. du suivi
    scan_interval: float = 0.03             # pause entre deux captures en attente


# --------------------------------------------------------------------------- #
# Arrêt propre : Échap + fail-safe PyAutoGUI
# --------------------------------------------------------------------------- #
class StopRequested(Exception):
    """Levée quand l'utilisateur demande l'arrêt (Échap)."""


_VK_ESCAPE = 0x1B


def _escape_pressed() -> bool:
    """Lit l'état de la touche Échap via l'API Win32 (aucune dépendance externe)."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_ESCAPE) & 0x8000)


def check_stop() -> None:
    """Point de contrôle appelé dans toutes les boucles."""
    if _escape_pressed():
        raise StopRequested("Touche Échap pressée")
    # Lève pyautogui.FailSafeException si la souris est dans un coin.
    pyautogui.failSafeCheck()


def sleep(seconds: float) -> None:
    """time.sleep interruptible (Échap / fail-safe vérifiés toutes les 20 ms)."""
    end = time.perf_counter() + seconds
    while True:
        check_stop()
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(0.02, remaining))


def rand(bounds: Range) -> float:
    return random.uniform(*bounds)


# --------------------------------------------------------------------------- #
# Entrées clavier
# --------------------------------------------------------------------------- #
def tap(key: str, duration: Range) -> None:
    """Pression courte d'une touche, durée de maintien aléatoire."""
    pyautogui.keyDown(key)
    try:
        sleep(rand(duration))
    finally:
        pyautogui.keyUp(key)  # relâche la touche même en cas d'interruption


# --------------------------------------------------------------------------- #
# Capture d'écran
# --------------------------------------------------------------------------- #
def grab(region: Optional[Region] = None) -> np.ndarray:
    """Capture (une zone de) l'écran et renvoie une image BGR pour OpenCV."""
    shot = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.asarray(shot), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
# Détection de la bannière
# --------------------------------------------------------------------------- #
class BannerDetector:
    """Repère la bannière par template matching multi-échelle en niveaux de gris."""

    def __init__(self, template_path: Path, threshold: float, scales: Sequence[float]):
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(
                f"Template introuvable : {template_path}\n"
                "Faites une capture de la bannière (voir README, section Calibration)."
            )
        self.threshold = threshold
        self.templates = [
            cv2.resize(template, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if s != 1.0 else template
            for s in scales
        ]

    def score(self, frame_bgr: np.ndarray) -> float:
        """Meilleur score de correspondance (0..1) sur toutes les échelles."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        best = 0.0
        for tpl in self.templates:
            if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(result.max()))
        return best

    def detect(self, frame_bgr: np.ndarray) -> bool:
        return self.score(frame_bgr) >= self.threshold


# --------------------------------------------------------------------------- #
# Détection de la jauge
# --------------------------------------------------------------------------- #
@dataclass
class GaugeState:
    green_left: int
    green_right: int
    green_top: int
    green_bottom: int
    yellow_x: Optional[int]  # None si l'indicateur n'est pas visible


class GaugeDetector:
    """Localise la zone verte et l'indicateur jaune par seuillage HSV."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.gauge_bridge_px, 1))
        self._denoise = np.ones((3, 3), np.uint8)

    def masks(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Renvoie (hsv, masque_vert, masque_jaune) — utile aussi pour la calibration."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array(self.cfg.green_lower), np.array(self.cfg.green_upper))
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, self._denoise)
        # La ligne jaune peut couper la zone verte en deux : on referme horizontalement.
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, self._bridge)
        yellow = cv2.inRange(hsv, np.array(self.cfg.yellow_lower), np.array(self.cfg.yellow_upper))
        return hsv, green, yellow

    def detect(self, frame_bgr: np.ndarray) -> Optional[GaugeState]:
        cfg = self.cfg
        _, green, yellow = self.masks(frame_bgr)

        # 1) Zone verte : plus grand contour en forme de barre horizontale.
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= cfg.gauge_min_width and h > 0 and w / h >= cfg.gauge_min_aspect:
                if best is None or w * h > best[2] * best[3]:
                    best = (x, y, w, h)
        if best is None:
            return None
        x, y, w, h = best

        # 2) Indicateur jaune : recherché sur toute la largeur, dans la bande
        #    horizontale de la jauge (élargie verticalement car la ligne
        #    dépasse souvent de la barre).
        pad = max(3, h)
        y0, y1 = max(0, y - pad), min(yellow.shape[0], y + h + pad)
        band = yellow[y0:y1, :]
        column_counts = np.count_nonzero(band, axis=0)
        yellow_x = None
        if column_counts.sum() >= cfg.yellow_min_pixels:
            # Centre de masse horizontal des pixels jaunes.
            yellow_x = int(round(np.average(np.arange(band.shape[1]), weights=column_counts)))

        return GaugeState(x, x + w, y, y + h, yellow_x)


# --------------------------------------------------------------------------- #
# Régulation de l'indicateur
# --------------------------------------------------------------------------- #
class GaugeController:
    """Correcteur tout-ou-rien avec zone morte et cadence limitée."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._next_press_at = 0.0

    def decide(self, state: GaugeState) -> Optional[str]:
        """Touche à presser pour recentrer l'indicateur, ou None."""
        if state.yellow_x is None:
            return None
        width = state.green_right - state.green_left
        margin = width * self.cfg.deadzone_ratio
        if state.yellow_x < state.green_left + margin:
            key = self.cfg.key_right   # indicateur trop à gauche -> pousser à droite
        elif state.yellow_x > state.green_right - margin:
            key = self.cfg.key_left    # indicateur trop à droite -> pousser à gauche
        else:
            return None
        if self.cfg.invert_controls:
            key = self.cfg.key_left if key == self.cfg.key_right else self.cfg.key_right
        return key

    def step(self, state: GaugeState) -> None:
        now = time.perf_counter()
        if now < self._next_press_at:
            return
        key = self.decide(state)
        if key is None:
            return
        log.debug("Correction %s (jaune=%s, vert=[%d,%d])",
                  key.upper(), state.yellow_x, state.green_left, state.green_right)
        tap(key, self.cfg.press_duration)
        self._next_press_at = time.perf_counter() + rand(self.cfg.press_cooldown)


# --------------------------------------------------------------------------- #
# Étapes du cycle
# --------------------------------------------------------------------------- #
def wait_for_banner(cfg: Config, detector: BannerDetector) -> bool:
    log.info("Attente de la bannière…")
    deadline = time.perf_counter() + cfg.banner_timeout
    while time.perf_counter() < deadline:
        check_stop()
        if detector.detect(grab(cfg.banner_region)):
            log.info("Bannière détectée.")
            return True
        sleep(cfg.scan_interval)
    return False


def wait_for_gauge(cfg: Config, detector: GaugeDetector) -> bool:
    log.info("Attente de la jauge…")
    deadline = time.perf_counter() + cfg.gauge_appear_timeout
    while time.perf_counter() < deadline:
        check_stop()
        if detector.detect(grab(cfg.gauge_region)) is not None:
            log.info("Jauge détectée, début du suivi.")
            return True
        sleep(cfg.scan_interval)
    return False


def track_gauge(cfg: Config, detector: GaugeDetector) -> None:
    """Suit la jauge jusqu'à sa disparition (absence continue > gauge_lost_grace)."""
    controller = GaugeController(cfg)
    started = time.perf_counter()
    lost_since: Optional[float] = None
    frames = 0

    while True:
        check_stop()
        now = time.perf_counter()
        if now - started > cfg.gauge_max_duration:
            log.warning("Durée max. de suivi atteinte, arrêt du suivi.")
            return

        state = detector.detect(grab(cfg.gauge_region))
        frames += 1
        if state is None:
            lost_since = lost_since or now
            if now - lost_since >= cfg.gauge_lost_grace:
                elapsed = now - started
                log.info("Jauge disparue (%.1f s, %.0f img/s).", elapsed, frames / max(elapsed, 1e-6))
                return
            continue
        lost_since = None
        controller.step(state)


def run_cycle(cfg: Config, banner: BannerDetector, gauge: GaugeDetector) -> None:
    # Étape 1 : lancement
    log.info("Étape 1 : appui sur %s", cfg.key_action.upper())
    tap(cfg.key_action, cfg.press_duration)
    sleep(rand(cfg.start_delay))

    # Étape 2 : bannière -> second appui
    if not wait_for_banner(cfg, banner):
        log.warning("Bannière non détectée après %.0f s, cycle abandonné.", cfg.banner_timeout)
        return
    sleep(rand(cfg.reaction_delay))
    tap(cfg.key_action, cfg.press_duration)

    # Étape 3 : régulation de la jauge
    if not wait_for_gauge(cfg, gauge):
        log.warning("Jauge non détectée après %.0f s, cycle abandonné.", cfg.gauge_appear_timeout)
        return
    track_gauge(cfg, gauge)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibrate(cfg: Config, out_dir: Path) -> None:
    """Enregistre capture, masques et image annotée pour régler seuils et zones."""
    out_dir.mkdir(parents=True, exist_ok=True)
    full = grab()
    cv2.imwrite(str(out_dir / "screen.png"), full)

    frame = grab(cfg.gauge_region)
    detector = GaugeDetector(cfg)
    _, green, yellow = detector.masks(frame)
    state = detector.detect(frame)
    annotated = frame.copy()
    if state:
        cv2.rectangle(annotated, (state.green_left, state.green_top),
                      (state.green_right, state.green_bottom), (255, 0, 255), 2)
        if state.yellow_x is not None:
            cv2.line(annotated, (state.yellow_x, 0), (state.yellow_x, annotated.shape[0]), (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "gauge_frame.png"), frame)
    cv2.imwrite(str(out_dir / "gauge_green_mask.png"), green)
    cv2.imwrite(str(out_dir / "gauge_yellow_mask.png"), yellow)
    cv2.imwrite(str(out_dir / "gauge_annotated.png"), annotated)
    log.info("Jauge : %s", state)

    try:
        banner = BannerDetector(cfg.banner_template, cfg.banner_threshold, cfg.banner_scales)
        log.info("Score bannière : %.3f (seuil %.2f)", banner.score(grab(cfg.banner_region)),
                 cfg.banner_threshold)
    except FileNotFoundError as exc:
        log.warning("%s", exc)
    log.info("Fichiers de calibration écrits dans %s", out_dir.resolve())


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Boucle de test d'interface (séquence pêche).")
    p.add_argument("--cycles", type=int, default=0, help="nombre de cycles (0 = infini)")
    p.add_argument("--calibrate", action="store_true", help="capture de calibration puis sortie")
    p.add_argument("--banner-region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    p.add_argument("--gauge-region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    p.add_argument("--template", type=Path, help="image de la bannière")
    p.add_argument("--invert", action="store_true", help="inverse Q et D")
    p.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.banner_region:
        cfg.banner_region = tuple(args.banner_region)
    if args.gauge_region:
        cfg.gauge_region = tuple(args.gauge_region)
    if args.template:
        cfg.banner_template = args.template
    if args.invert:
        cfg.invert_controls = True
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = build_config(args)

    pyautogui.FAILSAFE = True   # souris dans un coin = arrêt immédiat
    pyautogui.PAUSE = 0         # les délais sont gérés explicitement

    try:
        if args.calibrate:
            calibrate(cfg, Path("debug"))
            return 0

        banner = BannerDetector(cfg.banner_template, cfg.banner_threshold, cfg.banner_scales)
        gauge = GaugeDetector(cfg)

        log.info("Démarrage dans %.0f s — placez le focus sur la fenêtre cible. "
                 "Échap ou souris dans un coin pour arrêter.", cfg.startup_countdown)
        sleep(cfg.startup_countdown)

        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            log.info("===== Cycle %d =====", cycle)
            run_cycle(cfg, banner, gauge)
            pause = rand(cfg.cycle_pause)
            log.info("Fin de cycle, pause de %.1f s.", pause)
            sleep(pause)
    except StopRequested as exc:
        log.info("Arrêt demandé : %s", exc)
    except pyautogui.FailSafeException:
        log.info("Arrêt : fail-safe PyAutoGUI (souris dans un coin).")
    except KeyboardInterrupt:
        log.info("Arrêt : Ctrl+C.")
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    finally:
        # Sécurité : aucune touche ne doit rester enfoncée.
        for key in (cfg.key_action, cfg.key_left, cfg.key_right):
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
