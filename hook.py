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
    python hook.py --setup            # réglage interactif des zones et couleurs (une fois)
    python hook.py                    # lance la boucle
    python hook.py --cycles 5         # s'arrête après 5 cycles
    python hook.py --calibrate        # capture + masques de debug pour vérifier le réglage
"""

from __future__ import annotations

import argparse
import ctypes
import json
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

CONFIG_FILE = Path(__file__).with_name("hook_config.json")


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
    # "scancode" (SendInput, recommandé pour les jeux), "pyautogui" ou "auto".
    input_backend: str = "auto"

    # --- Zones de capture ----------------------------------------------------
    # banner_region : None = écran entier (plus lent mais fonctionnel).
    # gauge_region  : obligatoire pour la boucle (le décor contient du vert).
    banner_region: Optional[Region] = None
    gauge_region: Optional[Region] = None

    # --- Détection de la bannière (template matching) ----------------------
    banner_template: Path = Path(__file__).with_name("templates") / "fish_on_hook.png"
    banner_threshold: float = 0.80          # score TM_CCOEFF_NORMED minimal
    banner_scales: Tuple[float, ...] = (0.9, 1.0, 1.1)  # tolérance d'échelle

    # --- Détection de la jauge (seuillage HSV, échelle OpenCV H:0-179) -----
    # Valeurs par défaut génériques : utiliser --setup pour les mesurer en jeu.
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


# Champs enregistrés dans hook_config.json (réglages propres à la machine).
PERSISTED_FIELDS = (
    "banner_region", "gauge_region",
    "green_lower", "green_upper", "yellow_lower", "yellow_upper",
    "invert_controls", "input_backend",
)


def load_config_file(cfg: Config, path: Path) -> bool:
    """Applique les réglages enregistrés (s'ils existent) sur cfg."""
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    for name in PERSISTED_FIELDS:
        if name in data:
            value = data[name]
            setattr(cfg, name, tuple(value) if isinstance(value, list) else value)
    return True


def save_config_file(cfg: Config, path: Path) -> None:
    data = {name: getattr(cfg, name) for name in PERSISTED_FIELDS}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
class PyAutoGuiKeyboard:
    """Touches virtuelles via PyAutoGUI (suffisant pour les applis classiques)."""

    name = "pyautogui"

    def prepare(self, keys: Sequence[str]) -> None:
        pass

    def down(self, key: str) -> None:
        pyautogui.keyDown(key)

    def up(self, key: str) -> None:
        pyautogui.keyUp(key)


if sys.platform == "win32":
    from ctypes import wintypes

    _ULONG_PTR = ctypes.c_size_t

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", _ULONG_PTR)]

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", _ULONG_PTR)]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD),
                    ("wParamH", wintypes.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]

    class _INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


class ScancodeKeyboard:
    """Touches matérielles via SendInput + scancodes (Windows).

    Les jeux (DirectInput / Raw Input) ignorent souvent les touches virtuelles
    de PyAutoGUI ; les scancodes sont reçus comme un vrai clavier. Le scancode
    est calculé avec la disposition clavier active : "q" désigne donc bien la
    touche marquée Q, en AZERTY comme en QWERTY.
    """

    name = "scancode"
    _INPUT_KEYBOARD = 1
    _KEYEVENTF_KEYUP = 0x0002
    _KEYEVENTF_SCANCODE = 0x0008
    _MAPVK_VK_TO_VSC = 0

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Le mode scancode n'est disponible que sous Windows.")
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        u32.VkKeyScanW.argtypes = [wintypes.WCHAR]
        u32.VkKeyScanW.restype = ctypes.c_short
        u32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        u32.MapVirtualKeyW.restype = wintypes.UINT
        u32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
        u32.SendInput.restype = wintypes.UINT
        self._u32 = u32
        self._codes: dict[str, int] = {}

    def scancode(self, key: str) -> int:
        if key not in self._codes:
            if len(key) != 1:
                raise ValueError(f"Touche non gérée en mode scancode : {key!r}")
            vk = self._u32.VkKeyScanW(key)
            if vk == -1:
                raise ValueError(f"Touche absente de la disposition clavier : {key!r}")
            sc = self._u32.MapVirtualKeyW(vk & 0xFF, self._MAPVK_VK_TO_VSC)
            if sc == 0:
                raise ValueError(f"Pas de scancode pour la touche {key!r}")
            self._codes[key] = sc
        return self._codes[key]

    def prepare(self, keys: Sequence[str]) -> None:
        for key in keys:
            log.debug("Touche %s -> scancode 0x%02X", key.upper(), self.scancode(key))

    def _send(self, key: str, flags: int) -> None:
        inp = _INPUT(type=self._INPUT_KEYBOARD)
        inp.ki = _KEYBDINPUT(0, self.scancode(key), self._KEYEVENTF_SCANCODE | flags, 0, 0)
        if self._u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def down(self, key: str) -> None:
        self._send(key, 0)

    def up(self, key: str) -> None:
        self._send(key, self._KEYEVENTF_KEYUP)


def make_keyboard(backend: str):
    if backend == "pyautogui" or (backend == "auto" and sys.platform != "win32"):
        return PyAutoGuiKeyboard()
    return ScancodeKeyboard()


# Clavier actif (remplacé dans main() selon la configuration).
keyboard = PyAutoGuiKeyboard()


def tap(key: str, duration: Range) -> None:
    """Pression courte d'une touche, durée de maintien aléatoire."""
    keyboard.down(key)
    try:
        sleep(rand(duration))
    finally:
        keyboard.up(key)  # relâche la touche même en cas d'interruption


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
                "Faites une capture du texte de la bannière (voir README)."
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
# Réglage interactif (--setup)
# --------------------------------------------------------------------------- #
def hsv_range_from_sample(hsv_pixels: np.ndarray, h_margin: int = 6,
                          sv_margin: int = 40) -> Tuple[HSV, HSV]:
    """Plage HSV couvrant un échantillon de pixels (percentiles 2-98 + marge)."""
    px = hsv_pixels.reshape(-1, 3).astype(int)
    lo = np.percentile(px, 2, axis=0)
    hi = np.percentile(px, 98, axis=0)
    lower = (max(0, int(lo[0]) - h_margin), max(0, int(lo[1]) - sv_margin),
             max(0, int(lo[2]) - sv_margin))
    upper = (min(179, int(hi[0]) + h_margin), 255, 255)
    return lower, upper


def vivid_pixels(hsv_pixels: np.ndarray, keep_ratio: float = 0.2) -> np.ndarray:
    """Garde les pixels les plus saturés/lumineux (ex. la fine ligne jaune
    au milieu d'un rectangle qui contient aussi du vert ou du gris)."""
    px = hsv_pixels.reshape(-1, 3)
    score = px[:, 1].astype(int) + px[:, 2].astype(int)
    threshold = np.percentile(score, 100 * (1 - keep_ratio))
    return px[score >= threshold]


def select_box(title: str, image: np.ndarray) -> Optional[Region]:
    """Rectangle tracé à la souris (Entrée/Espace = valider, C = ignorer)."""
    print(f"\n>>> {title}\n    Tracez un rectangle puis Entrée (C pour ignorer).")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return None
    return int(x), int(y), int(w), int(h)


def setup(cfg: Config, path: Path) -> int:
    """Capture l'écran avec la jauge visible, fait tracer les zones et mesure les couleurs."""
    log.info("Affichez la jauge à l'écran : capture dans %.0f s…", cfg.startup_countdown)
    sleep(cfg.startup_countdown)
    screen = grab()

    banner_region = select_box("1/4 - Zone ou apparait la banniere (C = ecran entier)", screen)
    gauge_region = select_box("2/4 - Zone de la jauge (un peu plus large que la barre)", screen)
    if gauge_region is None:
        log.error("La zone de la jauge est obligatoire, réglage annulé.")
        return 1

    gx, gy, gw, gh = gauge_region
    gauge_img = screen[gy:gy + gh, gx:gx + gw]
    hsv = cv2.cvtColor(gauge_img, cv2.COLOR_BGR2HSV)

    green_box = select_box("3/4 - Rectangle a l'interieur de la zone verte", gauge_img)
    yellow_box = select_box("4/4 - Rectangle serre autour de la ligne jaune", gauge_img)

    cfg.banner_region = banner_region
    cfg.gauge_region = gauge_region
    if green_box:
        x, y, w, h = green_box
        cfg.green_lower, cfg.green_upper = hsv_range_from_sample(hsv[y:y + h, x:x + w])
    else:
        log.warning("Zone verte ignorée : couleurs par défaut conservées.")
    if yellow_box:
        x, y, w, h = yellow_box
        cfg.yellow_lower, cfg.yellow_upper = hsv_range_from_sample(
            vivid_pixels(hsv[y:y + h, x:x + w]))
    else:
        log.warning("Ligne jaune ignorée : couleurs par défaut conservées.")

    save_config_file(cfg, path)
    log.info("Réglages enregistrés dans %s", path)
    log.info("Vert  : %s -> %s", cfg.green_lower, cfg.green_upper)
    log.info("Jaune : %s -> %s", cfg.yellow_lower, cfg.yellow_upper)

    # Vérification immédiate sur la capture.
    state = GaugeDetector(cfg).detect(gauge_img)
    if state is None or state.yellow_x is None:
        log.warning("Vérification : jauge ou ligne jaune non détectée (%s). "
                    "Relancez --setup ou contrôlez avec --calibrate.", state)
    else:
        log.info("Vérification OK : %s", state)
    return 0


# --------------------------------------------------------------------------- #
# Calibration (vérification visuelle)
# --------------------------------------------------------------------------- #
def calibrate(cfg: Config, out_dir: Path) -> None:
    """Enregistre capture, masques et image annotée pour vérifier seuils et zones."""
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
    p.add_argument("--setup", action="store_true",
                   help="réglage interactif des zones et couleurs, enregistré dans hook_config.json")
    p.add_argument("--calibrate", action="store_true", help="capture de vérification puis sortie")
    p.add_argument("--config", type=Path, default=CONFIG_FILE, help="fichier de réglages")
    p.add_argument("--banner-region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    p.add_argument("--gauge-region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    p.add_argument("--template", type=Path, help="image de la bannière")
    p.add_argument("--invert", action="store_true", help="inverse Q et D")
    p.add_argument("--input", choices=("auto", "scancode", "pyautogui"),
                   help="méthode d'envoi des touches (auto = scancode sous Windows)")
    p.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if load_config_file(cfg, args.config):
        log.info("Réglages chargés depuis %s", args.config)
    # Les options en ligne de commande priment sur le fichier.
    if args.banner_region:
        cfg.banner_region = tuple(args.banner_region)
    if args.gauge_region:
        cfg.gauge_region = tuple(args.gauge_region)
    if args.template:
        cfg.banner_template = args.template
    if args.invert:
        cfg.invert_controls = True
    if args.input:
        cfg.input_backend = args.input
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    global keyboard
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
        if args.setup:
            return setup(cfg, args.config)
        if args.calibrate:
            calibrate(cfg, Path("debug"))
            return 0

        if cfg.gauge_region is None:
            log.error("Zone de la jauge non définie : lancez d'abord "
                      "« python hook.py --setup » (ou passez --gauge-region L T W H).")
            return 1

        keyboard = make_keyboard(cfg.input_backend)
        keyboard.prepare((cfg.key_action, cfg.key_left, cfg.key_right))
        log.info("Envoi des touches : %s", keyboard.name)

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
    except (FileNotFoundError, ValueError, OSError) as exc:
        log.error("%s", exc)
        return 1
    finally:
        # Sécurité : aucune touche ne doit rester enfoncée.
        for key in (cfg.key_action, cfg.key_left, cfg.key_right):
            try:
                keyboard.up(key)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
