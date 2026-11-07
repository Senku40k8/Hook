"""
hook.py - Boucle de test d'interface automatisée (séquence "pêche").

Cycle exécuté en boucle :
    1. Appui sur F puis courte latence aléatoire (0.5 s - 1.5 s).
    2. Surveillance de l'écran jusqu'à l'apparition de la bannière
       "Fish on the hook! Press the button to reel it!", puis nouvel appui sur F.
    3. Suivi en temps réel de la jauge (zone verte + indicateur jaune) :
       pressions courtes sur Q (gauche) / D (droite) pour garder l'indicateur
       dans la zone verte, jusqu'à la disparition de la jauge.
    4. Jauge disparue : résultat (réussite/échec) lu si possible, appui sur F.
    5. Pause (au moins 3 s, parfois plus longue) puis retour à l'étape 1.

Plus d'appâts (case « Not Selected » avant un cycle, ou bannière « Equip bait
before fishing ») : achat et équipement automatiques via les menus du jeu
(Switch Bait par la case d'appât ou E, Purchase, Universal Bait, quantité
max., Purchase, Échap x2, Switch Bait, Switch, Start Fishing si affiché), avec
des gestes de souris naturels.

Comportement humain (module session.py) : temps de réaction réalistes, fatigue,
petites imperfections, sessions de 20-45 min entrecoupées de pauses de 5-15 min.

Contrôle :
    - F8 : pause / reprise (le cycle repart de zéro à la reprise).
    - Échap, souris dans un coin de l'écran ou Ctrl+C : arrêt.
    - Arrêt automatique (avec alerte sonore) après plusieurs cycles ratés de
      suite ou si un message de blocage (templates/stop/*.png) apparaît.

Usage :
    python hook.py                    # lance la boucle (zones détectées automatiquement)
    python hook.py --duration 2h      # s'arrête après 2 heures
    python hook.py --max-fish 100     # s'arrête après 100 poissons
    python hook.py --cycles 5         # s'arrête après 5 cycles
    python hook.py --setup            # facultatif : réglage manuel des zones et couleurs
    python hook.py --calibrate        # capture + masques de debug pour vérifier le réglage
    python hook.py --reset            # oublie les zones mémorisées
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np
import pyautogui

from session import (
    CycleRecord, HumanConfig, Humanizer, SessionConfig, SessionPlanner, Stats,
    format_duration, parse_duration,
)

log = logging.getLogger("hook")

Region = Tuple[int, int, int, int]  # (left, top, width, height) en pixels écran
HSV = Tuple[int, int, int]
Range = Tuple[float, float]

CONFIG_FILE = Path(__file__).with_name("hook_config.json")
TEMPLATES_DIR = Path(__file__).with_name("templates")


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
    # None = recherche sur tout l'écran ; la zone est alors verrouillée
    # automatiquement à la première détection et mémorisée dans hook_config.json.
    banner_region: Optional[Region] = None
    gauge_region: Optional[Region] = None
    auto_lock_regions: bool = True
    # Titre de la fenêtre du jeu : mémorisé au premier lancement, le bot la met
    # ensuite lui-même au premier plan.
    window_title: Optional[str] = None
    config_path: Optional[Path] = None      # où mémoriser les zones (non persisté)
    record_dir: Optional[Path] = None       # --record : images annotées du suivi
    record_max_frames: int = 600

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
    yellow_min_pixels: int = 6              # nb de pixels min. d'une ligne jaune
    yellow_min_height: int = 5              # hauteur min. de la ligne jaune (px)
    yellow_min_aspect: float = 1.5          # hauteur/largeur min. (ligne verticale)
    yellow_search_factor: float = 4.0       # portée de recherche (x largeur verte, de chaque côté)
    gauge_lock_vertical_pad: int = 60       # marge verticale de la zone verrouillée (px)

    # --- Régulation --------------------------------------------------------
    deadzone_ratio: float = 0.15            # tolérance autour de la cible (fraction de la largeur verte)
    lead_time: float = 0.15                 # anticipation du mouvement de la zone verte (s)
    velocity_smoothing: float = 0.5         # lissage de la vitesse estimée (0..1)
    press_gain: float = 0.30                # s d'appui par largeur verte d'écart (avant apprentissage)
    # Réglage automatique : vitesse de la ligne jaune (px/s d'appui) mesurée
    # pendant le jeu, mémorisée dans hook_config.json.
    line_speed: Optional[float] = None
    line_speed_min_samples: int = 3         # mesures avant d'utiliser la vitesse apprise
    correction_fraction: float = 0.8        # part de l'écart corrigée par un appui
    auto_invert: bool = True                # inverse Q/D si la ligne part à l'opposé

    # --- Temporisations (secondes, bornes min/max des tirages aléatoires) ---
    start_delay: Range = (0.5, 1.5)         # après le premier F
    reaction_delay: Range = (0.12, 0.30)    # entre la bannière et le second F
    action_press_duration: Range = (0.08, 0.15)  # durée d'appui sur F (lu à chaque image par le jeu)
    press_duration: Range = (0.03, 0.20)    # durée min./max. d'une pression Q/D
    press_cooldown: Range = (0.04, 0.10)    # intervalle min. entre deux corrections
    cycle_pause: Range = (3.0, 4.5)         # fin de cycle
    startup_countdown: float = 1.0          # délai après l'arrivée du jeu au premier plan
    setup_countdown: float = 10.0           # temps pour afficher la jauge avant la capture --setup

    # --- Délais de garde ----------------------------------------------------
    banner_timeout: float = 60.0            # abandon du cycle si pas de bannière
    gauge_appear_timeout: float = 10.0      # abandon si la jauge n'apparaît pas
    gauge_lost_grace: float = 0.6           # absence continue = jauge disparue
    gauge_max_duration: float = 120.0       # sécurité : durée max. du suivi
    scan_interval: float = 0.03             # pause entre deux captures en attente

    # --- Résultat de la prise et blocages (templates facultatifs) ------------
    # templates/result_success.png, templates/result_fail.png : messages de fin.
    # templates/stop/*.png : messages qui arrêtent le bot (inventaire plein…).
    templates_dir: Path = TEMPLATES_DIR
    message_threshold: float = 0.80
    result_timeout: float = 1.5             # attente max. du message de résultat (s)
    max_consecutive_failures: int = 3       # arrêt après N cycles ratés de suite

    # --- Réapprovisionnement en appâts --------------------------------------
    # Déclenché avant chaque cycle si la case d'appât est vide (« Not Selected »),
    # ou par la bannière « Equip bait before fishing ». Étapes : voir EQUIP_STEPS
    # (appâts en stock) et BUY_STEPS (plus d'appâts).
    # Invite « Fishing » du point de pêche : F pour s'installer (au lancement ou
    # si le bot est sorti du mode pêche).
    fishing_prompt_template: Path = TEMPLATES_DIR / "fishing_prompt.png"

    restock_enabled: bool = True
    bait_menu_key: str = "e"                # ouvre la sélection d'appât (« Select Bait »)
    restock_dir: Path = TEMPLATES_DIR / "restock"
    no_bait_template: Path = TEMPLATES_DIR / "equip_bait_banner.png"
    restock_threshold: float = 0.80         # boutons texte (niveaux de gris)
    restock_icon_threshold: float = 0.60    # icônes (contours)
    restock_step_timeout: float = 6.0       # attente max. de chaque bouton (s)
    restock_step_delay: Range = (0.35, 0.8) # pause humaine entre deux étapes
    # Sans templates/equip_bait_banner.png : tenter un réapprovisionnement quand la
    # bannière « Fish on the hook » n'arrive pas. Inutile tant que la case vide est
    # vérifiée avant chaque cycle.
    restock_on_no_banner: bool = False

    # --- Session, comportement humain, statistiques -------------------------
    human: HumanConfig = field(default_factory=HumanConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    stats_path: Optional[Path] = Path(__file__).with_name("logs") / "catches.csv"
    pause_vk: int = 0x77                    # F8 : pause / reprise


# Champs enregistrés dans hook_config.json (réglages propres à la machine).
PERSISTED_FIELDS = (
    "banner_region", "gauge_region",
    "green_lower", "green_upper", "yellow_lower", "yellow_upper",
    "invert_controls", "input_backend", "window_title", "line_speed",
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
# Arrêt (Échap, fail-safe), pause (F8) et alertes
# --------------------------------------------------------------------------- #
class StopRequested(Exception):
    """Levée quand l'utilisateur demande l'arrêt (Échap)."""


class BlockedStop(StopRequested):
    """Arrêt automatique sur blocage (cycles ratés, message de blocage)."""


class ResumeRequested(Exception):
    """Levée à la reprise après une pause F8 : le cycle repart de zéro."""


_VK_ESCAPE = 0x1B
PAUSE_VK = 0x77                     # remplacé par cfg.pause_vk dans main()
_pause_key_was_down = False


def _key_state(vk: int) -> int:
    """État brut d'une touche (API Win32 GetAsyncKeyState) :
    bit 0x8000 = enfoncée maintenant, bit 0x0001 = pressée depuis la dernière lecture."""
    if sys.platform != "win32":
        return 0
    return ctypes.windll.user32.GetAsyncKeyState(vk) & 0xFFFF


def _key_is_down(vk: int) -> bool:
    return bool(_key_state(vk) & 0x8000)


_ignore_escape_until = 0.0         # Échap envoyé par le bot lui-même (menus)


def _escape_pressed() -> bool:
    if time.perf_counter() < _ignore_escape_until:
        return False
    return _key_is_down(_VK_ESCAPE)


def _pause_toggled() -> bool:
    """True au moment où la touche de pause est pressée (front montant).

    Le bit « pressée depuis la dernière lecture » rattrape un appui bref fait
    pendant une capture d'écran ou une analyse d'image (entre deux lectures)."""
    global _pause_key_was_down
    state = _key_state(PAUSE_VK)
    down = bool(state & 0x8000)
    toggled = (down or bool(state & 0x0001)) and not _pause_key_was_down
    _pause_key_was_down = down
    return toggled


def _pause_loop() -> None:
    """Pause F8 : relâche les touches, attend F8 (reprise) ou Échap (arrêt)."""
    release_all_keys()
    log.info("PAUSE. F8 pour reprendre, Échap pour arrêter.")
    while True:
        time.sleep(0.05)
        if _escape_pressed():
            raise StopRequested("Touche Échap pressée")
        pyautogui.failSafeCheck()
        if _pause_toggled():
            break
    log.info("REPRISE.")
    raise ResumeRequested()


def check_stop() -> None:
    """Point de contrôle appelé dans toutes les boucles."""
    if _escape_pressed():
        raise StopRequested("Touche Échap pressée")
    # Lève pyautogui.FailSafeException si la souris est dans un coin.
    pyautogui.failSafeCheck()
    if _pause_toggled():
        _pause_loop()


def alert(kind: str) -> None:
    """Signal sonore : 'error' (blocage, 3 bips aigus) ou 'done' (fin, 1 bip)."""
    if sys.platform != "win32":
        print("\a", end="", flush=True)
        return
    import winsound
    try:
        if kind == "error":
            for _ in range(3):
                winsound.Beep(1000, 220)
                time.sleep(0.08)
        else:
            winsound.Beep(660, 300)
    except RuntimeError:
        winsound.MessageBeep()


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

    SPECIAL_SCANCODES = {"esc": 0x01, "escape": 0x01, "enter": 0x1C, "space": 0x39}

    def scancode(self, key: str) -> int:
        if key in self.SPECIAL_SCANCODES:
            return self.SPECIAL_SCANCODES[key]
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


# Clavier actif et comportement humain (remplacés dans main() selon la configuration).
keyboard = PyAutoGuiKeyboard()
human = Humanizer(HumanConfig(enabled=False))
_held_keys: set = set()


def release_all_keys() -> None:
    for key in list(_held_keys):
        try:
            keyboard.up(key)
        except Exception:
            pass
        _held_keys.discard(key)


def tap(key: str, duration: Range) -> None:
    """Pression courte d'une touche, durée de maintien tirée dans duration.

    N'envoie rien tant que la fenêtre du jeu n'est pas au premier plan."""
    global _ignore_escape_until
    wait_for_focus()
    hold = human.within(duration)
    if key in ("esc", "escape"):
        # Notre propre Échap ne doit pas être pris pour une demande d'arrêt.
        _ignore_escape_until = time.perf_counter() + hold + 0.4
    keyboard.down(key)
    _held_keys.add(key)
    try:
        sleep(hold)
    finally:
        keyboard.up(key)  # relâche la touche même en cas d'interruption
        _held_keys.discard(key)


# --------------------------------------------------------------------------- #
# Fenêtre du jeu
# --------------------------------------------------------------------------- #
class GameWindow:
    """Fenêtre cible (Windows) : premier plan, zone client, privilèges.

    Les recherches d'images sont limitées à cette fenêtre : une bannière
    visible ailleurs (éditeur, navigateur…) ne peut donc pas être détectée.
    """

    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        self.title = self.window_title(hwnd)

    @staticmethod
    def _user32():
        u32 = ctypes.windll.user32
        u32.GetForegroundWindow.restype = wintypes.HWND
        return u32

    @classmethod
    def foreground_hwnd(cls) -> int:
        return cls._user32().GetForegroundWindow() or 0

    @classmethod
    def window_title(cls, hwnd: int) -> str:
        u32 = cls._user32()
        length = u32.GetWindowTextLengthW(wintypes.HWND(hwnd))
        buf = ctypes.create_unicode_buffer(length + 1)
        u32.GetWindowTextW(wintypes.HWND(hwnd), buf, length + 1)
        return buf.value

    @classmethod
    def class_name(cls, hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        cls._user32().GetClassNameW(wintypes.HWND(hwnd), buf, 256)
        return buf.value

    # Fenêtres du shell qui passent au premier plan pendant un changement de
    # fenêtre (barre des tâches, Alt+Tab, Win+Tab, bureau) : jamais le jeu.
    SHELL_CLASSES = {
        "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "MultitaskingViewFrame",
        "XamlExplorerHostIslandWindow", "ForegroundStaging", "TaskSwitcherWnd",
        "Progman", "WorkerW", "Windows.UI.Core.CoreWindow",
    }

    @classmethod
    def is_candidate(cls, hwnd: int) -> bool:
        """Fenêtre d'application : visible et hors shell Windows."""
        return bool(hwnd) and bool(cls._user32().IsWindowVisible(wintypes.HWND(hwnd))) \
            and cls.class_name(hwnd) not in cls.SHELL_CLASSES

    @classmethod
    def list_windows(cls) -> list[int]:
        """Fenêtres d'application visibles ayant un titre."""
        u32 = cls._user32()
        found: list[int] = []
        proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def callback(hwnd, _):
            if cls.is_candidate(hwnd) and cls.window_title(hwnd).strip():
                found.append(hwnd)
            return True

        u32.EnumWindows(proc(callback), 0)
        return found

    @classmethod
    def find(cls, title: str) -> Optional["GameWindow"]:
        """Fenêtre au titre exact, sinon première dont le titre le contient (sans casse)."""
        windows = cls.list_windows()
        for hwnd in windows:
            if cls.window_title(hwnd) == title:
                return cls(hwnd)
        for hwnd in windows:
            if title.lower() in cls.window_title(hwnd).lower():
                return cls(hwnd)
        return None

    def is_foreground(self) -> bool:
        return self.foreground_hwnd() == self.hwnd

    def activate(self, timeout: float = 3.0) -> bool:
        """Met la fenêtre au premier plan (restaure si réduite)."""
        u32 = self._user32()
        hwnd = wintypes.HWND(self.hwnd)
        if u32.IsIconic(hwnd):
            u32.ShowWindow(hwnd, 9)  # SW_RESTORE
        # Windows refuse SetForegroundWindow à un programme en arrière-plan, sauf
        # juste après une entrée clavier : un appui sur Alt lève ce verrou.
        u32.keybd_event(0x12, 0, 0, 0)
        u32.keybd_event(0x12, 0, 0x0002, 0)
        u32.SetForegroundWindow(hwnd)
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.is_foreground():
                return True
            sleep(0.05)
        return False

    def client_region(self) -> Optional[Region]:
        """Zone client (sans bordures ni barre de titre) en coordonnées écran."""
        u32 = self._user32()
        hwnd = wintypes.HWND(self.hwnd)
        if u32.IsIconic(hwnd):
            return None
        rect = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not u32.GetClientRect(hwnd, ctypes.byref(rect)) or \
                not u32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
        if rect.right <= 0 or rect.bottom <= 0:
            return None
        return origin.x, origin.y, rect.right, rect.bottom

    def is_elevated(self) -> Optional[bool]:
        """True si le jeu tourne en administrateur, None si impossible à savoir."""
        pid = wintypes.DWORD()
        self._user32().GetWindowThreadProcessId(wintypes.HWND(self.hwnd), ctypes.byref(pid))
        k32, adv = ctypes.windll.kernel32, ctypes.windll.advapi32
        k32.OpenProcess.restype = wintypes.HANDLE
        process = k32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not process:
            return None
        token = wintypes.HANDLE()
        try:
            if not adv.OpenProcessToken(wintypes.HANDLE(process), 0x0008, ctypes.byref(token)):
                return None  # refus d'accès : processus protégé ou élevé
            elevation, size = wintypes.DWORD(), wintypes.DWORD()
            if not adv.GetTokenInformation(token, 20, ctypes.byref(elevation), 4, ctypes.byref(size)):
                return None  # 20 = TokenElevation
            return bool(elevation.value)
        finally:
            if token:
                k32.CloseHandle(token)
            k32.CloseHandle(wintypes.HANDLE(process))


def running_as_admin() -> bool:
    return sys.platform == "win32" and bool(ctypes.windll.shell32.IsUserAnAdmin())


# Fenêtre du jeu (définie dans main() ; None = pas de restriction).
target: Optional[GameWindow] = None


def wait_for_focus() -> None:
    """Met le bot en pause tant que la fenêtre du jeu n'est pas au premier plan."""
    if target is None or target.is_foreground():
        return
    log.warning("« %s » n'est plus au premier plan : bot en pause.", target.title)
    while not target.is_foreground():
        sleep(0.2)
    log.info("Jeu revenu au premier plan, reprise.")
    sleep(0.5)


def wait_for_clicked_window(stable_for: float = 0.5) -> GameWindow:
    """Attend qu'une autre application que le terminal passe au premier plan et
    y reste stable_for secondes (ignore barre des tâches, Alt+Tab, etc.)."""
    start = GameWindow.foreground_hwnd()
    candidate, since = 0, 0.0
    while True:
        hwnd = GameWindow.foreground_hwnd()
        if hwnd != start and GameWindow.is_candidate(hwnd):
            if hwnd != candidate:
                candidate, since = hwnd, time.perf_counter()
            elif time.perf_counter() - since >= stable_for:
                return GameWindow(hwnd)
        else:
            candidate = 0
        sleep(0.05)


def pick_target(cfg: Config, settle: float) -> GameWindow:
    """Fenêtre du jeu : retrouvée par son titre et activée par le bot, sinon
    choisie en cliquant dessus (son titre est alors mémorisé)."""
    window = GameWindow.find(cfg.window_title) if cfg.window_title else None
    if cfg.window_title and window is None:
        log.warning("Fenêtre « %s » introuvable (jeu fermé ?).", cfg.window_title)

    if window is not None:
        log.info("Fenêtre du jeu : « %s », mise au premier plan…", window.title)
        if not window.activate():
            log.warning("Windows a refusé la mise au premier plan : cliquez dans le jeu.")
            while not window.is_foreground():
                sleep(0.1)
    else:
        log.info("Cliquez dans la fenêtre du jeu pour démarrer…")
        window = wait_for_clicked_window()
        log.info("Fenêtre du jeu : « %s » (type %s).",
                 window.title, GameWindow.class_name(window.hwnd))

    if window.title and cfg.window_title != window.title:
        log.info("Titre mémorisé : le bot activera lui-même cette fenêtre aux prochains lancements.")
        cfg.window_title = window.title
        remember_regions(cfg)
    sleep(settle)
    return window


def warn_if_input_blocked(window: GameWindow) -> None:
    """Windows ignore sans erreur les touches envoyées à un jeu lancé en
    administrateur depuis un terminal qui ne l'est pas."""
    if running_as_admin():
        return
    elevated = window.is_elevated()
    if elevated:
        log.warning("Le jeu tourne en administrateur mais pas ce terminal : Windows "
                    "bloquera les touches. Relancez le terminal en administrateur.")
    elif elevated is None:
        log.warning("Privilèges du jeu inconnus (processus protégé ?). Si F n'a aucun "
                    "effet, relancez le terminal en administrateur.")


# --------------------------------------------------------------------------- #
# Capture d'écran
# --------------------------------------------------------------------------- #
_screen = None


def _sct():
    """Instance mss partagée (capture rapide, sans dépendre de Pillow)."""
    global _screen
    if _screen is None:
        _screen = getattr(mss, "MSS", None)() if hasattr(mss, "MSS") else mss.mss()
    return _screen


def move_mouse(tx: float, ty: float, target_size: float) -> None:
    """Déplace la souris le long d'un trajet humain (courbe, vitesse en cloche)."""
    start = pyautogui.position()
    for x, y, delay in human.mouse_path((start[0], start[1]), (tx, ty), target_size):
        if delay > 0:
            sleep(delay)
        pyautogui.moveTo(int(round(x)), int(round(y)), _pause=False)


def click_box(box: Region) -> None:
    """Clic souris « humain » dans un rectangle (coordonnées écran) : point
    visé autour du centre (loi normale), trajet naturel, court survol, appui
    de durée variable."""
    x, y, w, h = box
    ox = max(-0.35, min(0.35, human.rng.gauss(0, 0.12))) * w
    oy = max(-0.30, min(0.30, human.rng.gauss(0, 0.12))) * h
    wait_for_focus()
    move_mouse(x + w / 2 + ox, y + h / 2 + oy, min(w, h))
    sleep(human.within((0.04, 0.15)))           # survol avant de cliquer
    pyautogui.mouseDown(_pause=False)
    try:
        sleep(human.within((0.05, 0.12)))
    finally:
        pyautogui.mouseUp(_pause=False)


def full_screen_region() -> Region:
    """Zone couvrant l'écran principal."""
    mon = _sct().monitors[1]
    return mon["left"], mon["top"], mon["width"], mon["height"]


def search_region() -> Region:
    """Zone de recherche quand aucune zone n'est verrouillée : la fenêtre du jeu."""
    if target is not None:
        region = target.client_region()
        if region is not None:
            return region
    return full_screen_region()


def grab(region: Optional[Region] = None) -> np.ndarray:
    """Capture (une zone de) l'écran et renvoie une image BGR pour OpenCV."""
    left, top, width, height = region or full_screen_region()
    shot = _sct().grab({"left": left, "top": top, "width": width, "height": height})
    return cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)


def clip_region(left: int, top: int, right: int, bottom: int, bounds: Region) -> Region:
    """Rectangle (coordonnées écran) rogné aux limites de bounds, au format Region."""
    bl, bt, bw, bh = bounds
    left, top = max(left, bl), max(top, bt)
    right, bottom = min(right, bl + bw), min(bottom, bt + bh)
    return int(left), int(top), int(right - left), int(bottom - top)


def remember_regions(cfg: Config) -> None:
    """Mémorise les zones verrouillées pour les prochains lancements."""
    if cfg.config_path is not None:
        save_config_file(cfg, cfg.config_path)


# --------------------------------------------------------------------------- #
# Détection de la bannière
# --------------------------------------------------------------------------- #
def edge_map(gray: np.ndarray) -> np.ndarray:
    """Contours épaissis : compare des formes indépendamment de leurs couleurs
    (ex. icône sur un onglet gris ou orange selon qu'il est sélectionné)."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8))


class BannerDetector:
    """Repère une image (bannière, bouton…) par template matching multi-échelle.

    edges=False : comparaison en niveaux de gris (texte, boutons).
    edges=True  : comparaison des contours (icônes dont la couleur change).
    """

    def __init__(self, template_path: Path, threshold: float, scales: Sequence[float],
                 edges: bool = False):
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(
                f"Template introuvable : {template_path}\n"
                "Faites une capture recadrée de l'élément (voir README)."
            )
        self.threshold = threshold
        self.edges = edges
        scaled = [
            cv2.resize(template, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if s != 1.0 else template
            for s in scales
        ]
        self.templates = [edge_map(t) if edges else t for t in scaled]

    def locate(self, frame_bgr: np.ndarray) -> Tuple[float, Optional[Region]]:
        """Meilleur score (0..1) et rectangle (x, y, w, h) correspondant dans l'image."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.edges:
            gray = edge_map(gray)
        best, box = 0.0, None
        for tpl in self.templates:
            if tpl.shape[0] > gray.shape[0] or tpl.shape[1] > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, value, _, (x, y) = cv2.minMaxLoc(result)
            if value > best:
                best, box = float(value), (x, y, tpl.shape[1], tpl.shape[0])
        return best, box

    def score(self, frame_bgr: np.ndarray) -> float:
        return self.locate(frame_bgr)[0]

    def detect(self, frame_bgr: np.ndarray) -> bool:
        return self.score(frame_bgr) >= self.threshold

    def locate_all(self, frame_bgr: np.ndarray, max_results: int = 5) -> List[Tuple[float, Region]]:
        """Toutes les occurrences au-dessus du seuil (ex. deux boutons « Purchase »
        visibles en même temps), de la meilleure à la moins bonne."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.edges:
            gray = edge_map(gray)
        found: List[Tuple[float, Region]] = []
        for tpl in self.templates:
            th, tw = tpl.shape[:2]
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            result = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            for _ in range(max_results):
                _, value, _, (x, y) = cv2.minMaxLoc(result)
                if value < self.threshold:
                    break
                found.append((float(value), (x, y, tw, th)))
                # Efface ce pic pour trouver l'occurrence suivante.
                result[max(0, y - th // 2):y + th // 2 + 1, max(0, x - tw // 2):x + tw // 2 + 1] = -1
        found.sort(key=lambda f: f[0], reverse=True)
        unique: List[Tuple[float, Region]] = []
        for score, box in found:   # même bouton trouvé à plusieurs échelles : garder le meilleur
            if all(not boxes_overlap(box, other) for _, other in unique):
                unique.append((score, box))
        return unique[:max_results]


def box_center(box: Region) -> Tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def boxes_overlap(a: Region, b: Region) -> bool:
    """Vrai si les centres sont à moins d'une demi-taille l'un de l'autre."""
    (ax, ay), (bx, by) = box_center(a), box_center(b)
    return abs(ax - bx) < max(a[2], b[2]) / 2 and abs(ay - by) < max(a[3], b[3]) / 2


class MessageDetector:
    """Reconnaît des messages du jeu à partir d'images facultatives.

    Chaque image fournie devient un nom de message (nom du fichier sans
    extension) ; les images absentes sont simplement ignorées.
    """

    def __init__(self, paths: Sequence[Path], threshold: float, scales: Sequence[float]):
        self.detectors: Dict[str, BannerDetector] = {}
        for path in paths:
            if path.exists():
                self.detectors[path.stem] = BannerDetector(path, threshold, scales)

    def __bool__(self) -> bool:
        return bool(self.detectors)

    def names(self) -> List[str]:
        return sorted(self.detectors)

    def find(self, frame_bgr: np.ndarray) -> Optional[str]:
        """Nom du message visible (meilleur score au-dessus du seuil), sinon None."""
        best_name, best_score = None, 0.0
        for name, detector in self.detectors.items():
            score = detector.score(frame_bgr)
            if score >= detector.threshold and score > best_score:
                best_name, best_score = name, score
        return best_name


def result_detector(cfg: Config) -> MessageDetector:
    """templates/result_success.png et templates/result_fail.png (facultatifs)."""
    return MessageDetector([cfg.templates_dir / "result_success.png",
                            cfg.templates_dir / "result_fail.png"],
                           cfg.message_threshold, cfg.banner_scales)


def stop_detector(cfg: Config) -> MessageDetector:
    """templates/stop/*.png : messages bloquants (inventaire plein, plus d'appât…)."""
    stop_dir = cfg.templates_dir / "stop"
    paths = sorted(stop_dir.glob("*.png")) if stop_dir.is_dir() else []
    return MessageDetector(paths, cfg.message_threshold, cfg.banner_scales)


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

    def yellow_lines(self, yellow: np.ndarray) -> list:
        """Formes jaunes verticales (x, y, w, h) : candidates pour la ligne jaune.
        Les éléments jaunes trapus (icônes, texte) sont écartés."""
        cfg = self.cfg
        count, _, stats, _ = cv2.connectedComponentsWithStats(yellow, connectivity=8)
        lines = []
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if area >= cfg.yellow_min_pixels and h >= cfg.yellow_min_height \
                    and h >= cfg.yellow_min_aspect * w:
                lines.append((int(x), int(y), int(w), int(h)))
        return lines

    def detect(self, frame_bgr: np.ndarray, search: bool = True) -> Optional[GaugeState]:
        """Localise la jauge : barre verte horizontale + ligne jaune verticale.

        Une barre verte n'est retenue que si une ligne jaune la croise ou se
        trouve à proximité : le vert du décor est ainsi ignoré, et la jauge est
        considérée comme disparue dès que l'ensemble n'est plus visible.
        search=False accepte une barre verte sans ligne jaune (calibration).
        """
        cfg = self.cfg
        _, green, yellow = self.masks(frame_bgr)

        # 1) Zones vertes candidates : contours en forme de barre horizontale.
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= cfg.gauge_min_width and h > 0 and w / h >= cfg.gauge_min_aspect:
                candidates.append((x, y, w, h))
        candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        lines = self.yellow_lines(yellow)

        for x, y, w, h in candidates:
            # 2) Ligne jaune : forme verticale qui chevauche la barre en hauteur,
            #    à moins de yellow_search_factor x largeur de la zone verte ;
            #    la plus proche du centre de la zone verte.
            center = x + w / 2
            reach = w / 2 + cfg.yellow_search_factor * w
            best = None
            for lx, ly, lw, lh in lines:
                overlaps = ly <= y + h + h and ly + lh >= y - h
                line_x = lx + lw / 2
                if overlaps and abs(line_x - center) <= reach:
                    if best is None or abs(line_x - center) < abs(best - center):
                        best = line_x
            if best is None and search:
                continue  # barre verte sans ligne jaune : probablement le décor
            yellow_x = None if best is None else int(round(best))
            return GaugeState(x, x + w, y, y + h, yellow_x)
        return None


# --------------------------------------------------------------------------- #
# Régulation de l'indicateur
# --------------------------------------------------------------------------- #
class GaugeController:
    """Fait suivre à la ligne jaune la zone verte, qui se déplace.

    - La cible est le centre de la zone verte, décalé dans le sens de son
      mouvement (vitesse estimée d'une image à l'autre) pour ne pas être en retard.
    - Hors de la tolérance, une pression Q ou D est envoyée, d'autant plus
      longue que l'écart est grand (à-coups courts quand on est proche).
    - Réglage automatique : après chaque appui, le déplacement réel de la ligne
      donne sa vitesse (px par seconde d'appui), qui sert ensuite à doser les
      appuis. Si la ligne part systématiquement à l'opposé, Q et D sont inversés.
    - Comportement humain (facultatif) : corrections parfois un peu en retard,
      appuis parfois un peu trop longs.
    """

    def __init__(self, cfg: Config, humanizer: Optional[Humanizer] = None):
        self.cfg = cfg
        self.human = humanizer
        self._next_press_at = 0.0
        self._last: Optional[Tuple[float, float]] = None  # (instant, centre vert)
        self.velocity = 0.0                                # px/s de la zone verte
        self.line_speed = cfg.line_speed                   # px/s d'appui (appris)
        self.speed_samples = cfg.line_speed_min_samples if cfg.line_speed else 0
        self.presses: Dict[str, int] = {cfg.key_left: 0, cfg.key_right: 0}
        self.inverted_now = False                          # inversion faite pendant ce suivi
        # Appui en attente de mesure : (sens attendu ±1, durée, x avant, fin de l'appui)
        self._pending: Optional[Tuple[int, float, int, float]] = None
        self._directions: List[bool] = []                  # True = sens attendu respecté

    def observe(self, state: GaugeState, now: float) -> None:
        """Met à jour la vitesse de la zone verte et mesure l'effet du dernier appui."""
        center = (state.green_left + state.green_right) / 2
        if self._last is not None:
            dt = now - self._last[0]
            if dt > 1e-3:
                instant = (center - self._last[1]) / dt
                s = self.cfg.velocity_smoothing
                self.velocity = s * self.velocity + (1 - s) * instant
        self._last = (now, center)
        self._learn(state, now)

    def _learn(self, state: GaugeState, now: float) -> None:
        if self._pending is None or state.yellow_x is None or now < self._pending[3]:
            return
        expected, duration, x_before, _ = self._pending
        self._pending = None
        delta = state.yellow_x - x_before
        if abs(delta) < 3 or duration <= 0:
            return  # mouvement trop faible pour conclure
        same_direction = (delta > 0) == (expected > 0)
        self._directions = (self._directions + [same_direction])[-6:]
        if same_direction:
            sample = abs(delta) / duration
            if self.line_speed is None:
                self.line_speed = sample
            else:
                self.line_speed = 0.7 * self.line_speed + 0.3 * sample
            self.speed_samples += 1
        elif self.cfg.auto_invert and len(self._directions) >= 5 \
                and self._directions.count(False) >= 4:
            self.cfg.invert_controls = not self.cfg.invert_controls
            self.inverted_now = True
            self._directions.clear()
            log.warning("La ligne jaune part à l'opposé des appuis : Q et D inversés "
                        "automatiquement (mémorisé).")

    def _direction_of(self, key: str) -> int:
        """Sens attendu du mouvement de la ligne pour cette touche (+1 = droite)."""
        right = key == self.cfg.key_right
        if self.cfg.invert_controls:
            right = not right
        return 1 if right else -1

    def target(self, state: GaugeState) -> float:
        width = state.green_right - state.green_left
        center = (state.green_left + state.green_right) / 2
        lead = self.velocity * self.cfg.lead_time
        return center + max(-width, min(width, lead))

    def decide(self, state: GaugeState) -> Optional[Tuple[str, float]]:
        """(touche, durée d'appui) pour rapprocher la ligne jaune de la cible, ou None."""
        if state.yellow_x is None:
            return None
        cfg = self.cfg
        width = max(1, state.green_right - state.green_left)
        error = self.target(state) - state.yellow_x
        if abs(error) <= width * cfg.deadzone_ratio:
            return None
        key = cfg.key_right if error > 0 else cfg.key_left
        if cfg.invert_controls:
            key = cfg.key_left if key == cfg.key_right else cfg.key_right
        lo, hi = cfg.press_duration
        if self.line_speed and self.speed_samples >= cfg.line_speed_min_samples:
            # Vitesse connue : appui qui couvre une part de l'écart.
            duration = abs(error) * cfg.correction_fraction / self.line_speed
        else:
            duration = lo + cfg.press_gain * abs(error) / width
        factor = self.human.press_factor() if self.human else random.uniform(0.85, 1.15)
        duration = min(hi, max(lo, duration * factor))
        return key, duration

    def step(self, state: GaugeState) -> None:
        now = time.perf_counter()
        self.observe(state, now)
        if now < self._next_press_at:
            return
        decision = self.decide(state)
        if decision is None:
            return
        if self.human is not None:
            late = self.human.skip_correction()
            if late is not None:
                self._next_press_at = now + late   # réaction un peu tardive
                return
        key, duration = decision
        log.debug("%s %.0f ms (jaune=%s, vert=[%d,%d], cible=%.0f, vitesse=%.0f px/s, "
                  "ligne=%s px/s)", key.upper(), duration * 1000, state.yellow_x,
                  state.green_left, state.green_right, self.target(state), self.velocity,
                  f"{self.line_speed:.0f}" if self.line_speed else "?")
        x_before = state.yellow_x
        tap(key, (duration, duration))
        self.presses[key] = self.presses.get(key, 0) + 1
        end = time.perf_counter()
        self._pending = (self._direction_of(key), duration, x_before, end)
        self._next_press_at = end + rand(self.cfg.press_cooldown)


# --------------------------------------------------------------------------- #
# Étapes du cycle
# --------------------------------------------------------------------------- #
def banner_lock_region(match: Region, frame_region: Region) -> Region:
    """Zone de la bannière : rectangle trouvé + marge (tolérance de position/échelle)."""
    mx, my, mw, mh = match
    left, top = frame_region[0] + mx, frame_region[1] + my
    pad_x, pad_y = int(mw * 0.15), mh
    return clip_region(left - pad_x, top - pad_y, left + mw + pad_x, top + mh + pad_y,
                       frame_region)


def gauge_lock_region(cfg: Config, state: GaugeState, frame_region: Region) -> Region:
    """Zone de la jauge : bande horizontale sur toute la largeur de la fenêtre,
    pour suivre la zone verte où qu'elle aille, et limitée en hauteur."""
    cy = frame_region[1] + (state.green_top + state.green_bottom) // 2
    half_h = (state.green_bottom - state.green_top) // 2 + cfg.gauge_lock_vertical_pad
    left, _, width, _ = frame_region
    return clip_region(left, cy - half_h, left + width, cy + half_h, frame_region)


def annotate(frame: np.ndarray, state: Optional[GaugeState],
             target_x: Optional[float] = None) -> np.ndarray:
    """Image de debug : zone verte (magenta), ligne jaune (rouge), cible (cyan)."""
    out = frame.copy()
    if state is not None:
        cv2.rectangle(out, (state.green_left, state.green_top),
                      (state.green_right, state.green_bottom), (255, 0, 255), 2)
        if state.yellow_x is not None:
            cv2.line(out, (state.yellow_x, 0), (state.yellow_x, out.shape[0]), (0, 0, 255), 2)
        if target_x is not None:
            cv2.line(out, (int(target_x), 0), (int(target_x), out.shape[0]), (255, 255, 0), 1)
    return out


def wait_for_banner(cfg: Config, detector: BannerDetector,
                    no_bait: Optional[BannerDetector] = None) -> Optional[str]:
    """'banner' (poisson ferré), 'no_bait' (plus d'appâts) ou None (délai dépassé)."""
    log.info("Attente de la bannière…")
    deadline = time.perf_counter() + cfg.banner_timeout
    scans = 0
    while time.perf_counter() < deadline:
        check_stop()
        wait_for_focus()
        region = cfg.banner_region or search_region()
        score, box = detector.locate(grab(region))
        if score >= detector.threshold:
            log.info("Bannière détectée (score %.2f).", score)
            if cfg.banner_region is None and cfg.auto_lock_regions and box:
                cfg.banner_region = banner_lock_region(box, region)
                remember_regions(cfg)
                log.info("Zone de la bannière mémorisée : %s", cfg.banner_region)
            return "banner"
        # « Equip bait before fishing » : vérifié sur toute la fenêtre, 1 fois sur 5.
        if no_bait is not None and scans % 5 == 0 and no_bait.detect(grab(search_region())):
            log.warning("Bannière « Equip bait before fishing » : plus d'appâts.")
            return "no_bait"
        scans += 1
        sleep(cfg.scan_interval)
    return None


def wait_for_gauge(cfg: Config, detector: GaugeDetector) -> bool:
    log.info("Attente de la jauge…")
    deadline = time.perf_counter() + cfg.gauge_appear_timeout
    while time.perf_counter() < deadline:
        check_stop()
        wait_for_focus()
        searching = cfg.gauge_region is None
        region = cfg.gauge_region or search_region()
        state = detector.detect(grab(region))
        if state is not None:
            log.info("Jauge détectée, début du suivi.")
            if searching and cfg.auto_lock_regions:
                cfg.gauge_region = gauge_lock_region(cfg, state, region)
                remember_regions(cfg)
                log.info("Zone de la jauge mémorisée : %s", cfg.gauge_region)
            return True
        sleep(cfg.scan_interval)
    return False


@dataclass
class TrackResult:
    disappeared: bool                       # False = durée max. atteinte
    duration: float
    presses_left: int
    presses_right: int
    in_zone_ratio: float                    # part des images où la ligne était dans le vert


def track_gauge(cfg: Config, detector: GaugeDetector) -> TrackResult:
    """Suit la jauge jusqu'à sa disparition (absence continue > gauge_lost_grace)."""
    controller = GaugeController(cfg, human if human.cfg.enabled else None)
    started = time.perf_counter()
    lost_since: Optional[float] = None
    frames = recorded = seen = in_zone = 0
    if cfg.record_dir is not None:
        cfg.record_dir.mkdir(parents=True, exist_ok=True)

    def result(disappeared: bool) -> TrackResult:
        learn_from(cfg, controller)
        return TrackResult(disappeared, time.perf_counter() - started,
                           controller.presses.get(cfg.key_left, 0),
                           controller.presses.get(cfg.key_right, 0),
                           in_zone / seen if seen else 0.0)

    while True:
        check_stop()
        now = time.perf_counter()
        if now - started > cfg.gauge_max_duration:
            log.warning("Durée max. de suivi atteinte, arrêt du suivi.")
            return result(False)

        wait_for_focus()
        frame = grab(cfg.gauge_region or search_region())
        state = detector.detect(frame)
        frames += 1
        if cfg.record_dir is not None and recorded < cfg.record_max_frames:
            target_x = controller.target(state) if state else None
            cv2.imwrite(str(cfg.record_dir / f"track_{recorded:04d}.png"),
                        annotate(frame, state, target_x))
            recorded += 1
        if state is None:
            lost_since = lost_since or now
            if now - lost_since >= cfg.gauge_lost_grace:
                elapsed = now - started
                log.info("Jauge disparue (%.1f s, %.0f img/s).", elapsed, frames / max(elapsed, 1e-6))
                return result(True)
            continue
        lost_since = None
        if state.yellow_x is not None:
            seen += 1
            in_zone += state.green_left <= state.yellow_x <= state.green_right
        controller.step(state)


def learn_from(cfg: Config, controller: GaugeController) -> None:
    """Mémorise la vitesse de ligne apprise (et une éventuelle inversion Q/D)."""
    changed = controller.inverted_now
    if controller.line_speed and controller.speed_samples >= cfg.line_speed_min_samples:
        if cfg.line_speed is None or abs(controller.line_speed - cfg.line_speed) > 1:
            cfg.line_speed = round(controller.line_speed, 1)
            changed = True
            log.debug("Vitesse de la ligne jaune : %.0f px/s d'appui.", cfg.line_speed)
    if changed:
        remember_regions(cfg)


def check_blockers(cfg: Config, stoppers: MessageDetector) -> None:
    """Arrête le bot si un message bloquant (templates/stop/*.png) est visible."""
    if not stoppers:
        return
    name = stoppers.find(grab(search_region()))
    if name:
        raise BlockedStop(f"message bloquant détecté à l'écran : « {name} »")


def wait_for_result(cfg: Config, results: MessageDetector) -> str:
    """'success', 'fail' ou 'unknown' selon le message affiché en fin de jauge."""
    if not results:
        return "unknown"
    deadline = time.perf_counter() + cfg.result_timeout
    while time.perf_counter() < deadline:
        check_stop()
        name = results.find(grab(search_region()))
        if name:
            return "success" if name.endswith("success") else "fail"
        sleep(cfg.scan_interval)
    return "unknown"


# --------------------------------------------------------------------------- #
# Réapprovisionnement en appâts
# --------------------------------------------------------------------------- #
# (action, valeur) :
#   "key"         touche à presser
#   "click"       bouton à cliquer (templates/restock/<nom>.png), obligatoire
#   "bait_dialog" ouvre la fenêtre « Switch Bait » : clic sur la case d'appât si
#                 le panneau « Fishing Preparations » est affiché, sinon touche E
#   "optional"    bouton cliqué seulement s'il est affiché
#   "dismiss"     fenêtre qui se ferme en cliquant dans une zone vide (Échap
#                 ne la ferme pas) : repérée par son texte « Press empty area
#                 to close », on clique juste au-dessus, hors de la fenêtre
#
# La fenêtre « Switch Bait » (ouverte par la case d'appât ou E) affiche :
#   - « Cancel / Switch »   s'il reste des appâts : on équipe directement ;
#   - « Cancel / Purchase » s'il n'en reste plus  : on achète, puis on équipe.
EQUIP_STEPS: Tuple[Tuple[str, str], ...] = (
    ("click", "switch"),                 # équipe l'appât en stock
    ("optional", "start_fishing"),       # panneau « Fishing Preparations » encore ouvert
)
BUY_STEPS: Tuple[Tuple[str, str], ...] = (
    ("click", "purchase"),               # ouvre l'achat d'appâts
    ("click", "universal_bait"),         # Universal Bait à 5 coquillages
    ("click", "max_quantity"),           # double chevron : quantité maximale
    ("click", "purchase"),               # achat…
    ("click", "confirm"),                # … à confirmer
    ("dismiss", "rewards_close"),        # fenêtre « Rewards » : clic dans une zone vide
    ("key", "esc"),
    ("key", "esc"),
    ("bait_dialog", "bait_slot_empty"),  # de nouveau « Switch Bait »…
) + EQUIP_STEPS                          # … puis Switch pour équiper l'appât acheté
RESTOCK_BUTTONS = sorted({name for action, name in BUY_STEPS if action != "key"}
                         | {"bait_slot_empty"})
ICON_TEMPLATES: set = set()     # modèles comparés par contours (icônes à couleur variable)
# Seuils plus stricts : la case vide ne doit jamais être confondue avec une case
# équipée (sinon achat d'appâts à chaque cycle).
TEMPLATE_THRESHOLDS: Dict[str, float] = {"bait_slot_empty": 0.88}
# Partie cliquable d'un modèle (fractions gauche, haut, droite, bas) quand ce
# n'est pas tout le rectangle : pour la case d'appât, le carré « + », pas le texte.
CLICK_AREAS: Dict[str, Tuple[float, float, float, float]] = {
    "bait_slot_empty": (0.2, 0.1, 0.8, 0.65),
}
# La case d'appât est cherchée par Restocker.find_slot (ouverture avec E si
# besoin, position précédente en dernier recours). Les autres boutons déjà
# cliqués ne sont jamais recliqués au même endroit : le premier « Purchase »
# peut rester visible derrière la boutique.


def click_area(name: str, box: Region) -> Region:
    left, top, right, bottom = CLICK_AREAS.get(name, (0.0, 0.0, 1.0, 1.0))
    x, y, w, h = box
    return (int(x + w * left), int(y + h * top),
            max(1, int(w * (right - left))), max(1, int(h * (bottom - top))))


class RestockFailed(Exception):
    """Un bouton du réapprovisionnement n'est pas apparu à temps."""


class Restocker:
    """Équipe un appât (et l'achète d'abord s'il n'en reste plus), en vérifiant
    chaque bouton avant de cliquer."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.buttons: Dict[str, BannerDetector] = {}
        for name in RESTOCK_BUTTONS:
            icon = name in ICON_TEMPLATES
            threshold = TEMPLATE_THRESHOLDS.get(
                name, cfg.restock_icon_threshold if icon else cfg.restock_threshold)
            self.buttons[name] = BannerDetector(
                cfg.restock_dir / f"{name}.png", threshold, cfg.banner_scales, edges=icon)

    def wait_for_any(self, names: Sequence[str], timeout: Optional[float] = None) -> str:
        """Attend que l'un des boutons soit visible et immobile ; renvoie son nom."""
        deadline = time.perf_counter() + (timeout or self.cfg.restock_step_timeout)
        previous: Dict[str, Optional[Region]] = {name: None for name in names}
        frame = None
        while time.perf_counter() < deadline:
            check_stop()
            wait_for_focus()
            region = search_region()
            frame = grab(region)
            for name in names:
                score, box = self.buttons[name].locate(frame)
                if score < self.buttons[name].threshold or box is None:
                    previous[name] = None
                    continue
                last = previous[name]
                if last is not None and abs(last[0] - box[0]) <= 3 and abs(last[1] - box[1]) <= 3:
                    return name
                previous[name] = box
            sleep(0.08)
        for name in names:
            self._save_failure(name, frame, self.buttons[name])
        raise RestockFailed("ni « " + " » ni « ".join(names) + " » dans la fenêtre « Switch Bait »")

    def bait_missing(self) -> bool:
        """True si la case d'appât vide (« Not Selected ») est visible."""
        return self.buttons["bait_slot_empty"].detect(grab(search_region()))

    def open_bait_dialog(self) -> None:
        """Ouvre la fenêtre « Switch Bait » : clic sur la case d'appât vide si le
        panneau « Fishing Preparations » est affiché, sinon touche E (en mode
        pêche, E ouvre directement cette fenêtre)."""
        try:
            box = self.wait_for_button("bait_slot_empty", timeout=1.0, quiet=True)
        except RestockFailed:
            log.info("  touche %s (fenêtre « Switch Bait »)", self.cfg.bait_menu_key.upper())
            tap(self.cfg.bait_menu_key, self.cfg.action_press_duration)
            return
        sleep(human.reaction())
        log.info("  clic sur la case d'appât")
        click_box(click_area("bait_slot_empty", box))

    def dismiss_popup(self, name: str) -> None:
        """Ferme une fenêtre du type « Rewards » en cliquant dans une zone vide."""
        try:
            hint = self.wait_for_button(name, timeout=4.0, quiet=True)
        except RestockFailed:
            log.info("  pas de fenêtre « Rewards » à fermer")
            return
        x, y, w, h = hint
        empty_area = (x + w // 6, int(y - 2.6 * h), w * 2 // 3, int(1.4 * h))
        for _ in range(3):                      # au plus 3 essais
            sleep(human.reaction())
            log.info("  clic dans une zone vide (fermeture de « Rewards »)")
            click_box(empty_area)
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                sleep(0.15)
                if not self.buttons[name].detect(grab(search_region())):
                    return
        raise RestockFailed("la fenêtre « Rewards » ne se ferme pas")

    def wait_for_button(self, name: str, timeout: Optional[float] = None,
                        avoid: Sequence[Region] = (), quiet: bool = False) -> Region:
        """Attend que le bouton soit visible et immobile (fin d'animation) ;
        renvoie son rectangle en coordonnées écran.

        avoid : positions déjà cliquées. Si plusieurs boutons identiques sont
        visibles (ex. deux « Purchase »), on prend celui qui n'y est pas."""
        detector = self.buttons[name]
        timeout = timeout or self.cfg.restock_step_timeout
        deadline = time.perf_counter() + timeout
        # Pendant la première moitié du délai, n'accepter qu'un bouton situé
        # ailleurs qu'aux positions déjà cliquées ; ensuite, la même position
        # est admise (le nouveau bouton peut s'afficher au même endroit).
        strict_until = time.perf_counter() + timeout / 2 if avoid else 0.0
        previous: Optional[Region] = None
        frame, region = None, None
        while time.perf_counter() < deadline:
            check_stop()
            wait_for_focus()
            region = search_region()
            frame = grab(region)
            candidates = [(s, (region[0] + b[0], region[1] + b[1], b[2], b[3]))
                          for s, b in detector.locate_all(frame)]
            fresh = [c for c in candidates
                     if not any(boxes_overlap(c[1], old) for old in avoid)]
            if fresh:
                chosen = fresh[0][1]
            elif candidates and time.perf_counter() >= strict_until:
                chosen = candidates[0][1]
            else:
                chosen = None
            if chosen is not None:
                if previous is not None and abs(previous[0] - chosen[0]) <= 3 \
                        and abs(previous[1] - chosen[1]) <= 3:
                    if len(candidates) > 1:
                        log.debug("  %d « %s » visibles, choix de %s", len(candidates), name, chosen)
                    return chosen
                previous = chosen
            else:
                previous = None
            sleep(0.08)
        if not quiet:
            self._save_failure(name, frame, detector)
        raise RestockFailed(f"bouton « {name} » introuvable")

    def _save_failure(self, name: str, frame: Optional[np.ndarray],
                      detector: BannerDetector, message: Optional[str] = None) -> None:
        """Capture d'écran pour comprendre pourquoi une étape a échoué."""
        if frame is None:
            return
        try:
            out = Path("debug") / f"restock_{name}_{time.strftime('%H%M%S')}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), frame)
            if message:
                log.warning("  %s Capture : %s", message, out)
            else:
                log.warning("  « %s » introuvable (meilleur score %.2f, seuil %.2f). Capture : %s",
                            name, detector.score(frame), detector.threshold, out)
        except Exception:
            pass

    def run(self) -> None:
        log.info("Aucun appât équipé : ouverture de « Switch Bait »…")
        try:
            sleep(human.within(self.cfg.restock_step_delay))
            self.open_bait_dialog()
            if self.wait_for_any(("switch", "purchase")) == "switch":
                log.info("Appâts en stock : équipement direct.")
                self._run_steps(EQUIP_STEPS)
            else:
                log.info("Plus d'appâts : achat d'Universal Bait puis équipement.")
                self._run_steps(BUY_STEPS)
        except RestockFailed:
            # Referme les menus restés ouverts avant d'abandonner, mais seulement
            # si le jeu est au premier plan (sinon on attendrait indéfiniment).
            for _ in range(2):
                time.sleep(human.within((0.3, 0.6)))
                if target is None or target.is_foreground():
                    tap("esc", self.cfg.action_press_duration)
            raise
        sleep(human.within((0.8, 1.5)))
        if self.bait_missing():
            # Toujours « Not Selected » : l'appât n'a pas été équipé.
            self._save_failure("not_equipped", grab(search_region()),
                               self.buttons["bait_slot_empty"],
                               "La case d'appât affiche toujours « Not Selected ».")
            raise RestockFailed("appâts pas équipés (case toujours vide)")
        log.info("Appât équipé, reprise de la pêche.")

    def _run_steps(self, steps: Sequence[Tuple[str, str]]) -> None:
        """Exécute une liste d'étapes (voir EQUIP_STEPS et BUY_STEPS)."""
        clicked: Dict[str, List[Region]] = {}
        for action, value in steps:
            # Rythme humain : courte pause, parfois une hésitation.
            sleep(human.within(self.cfg.restock_step_delay) + human.hesitation())
            if action == "key":
                log.info("  touche %s", value.upper())
                tap(value, self.cfg.action_press_duration)
                continue
            if action == "bait_dialog":
                self.open_bait_dialog()
                continue
            if action == "dismiss":
                self.dismiss_popup(value)
                continue
            before = clicked.get(value, [])
            if action == "optional":
                try:
                    box = self.wait_for_button(value, timeout=2.0, quiet=True)
                except RestockFailed:
                    continue              # pas affiché : rien à faire
            else:
                # Bouton déjà cliqué (2e « Purchase ») : surtout pas le même.
                box = self.wait_for_button(value, avoid=before)
            clicked.setdefault(value, []).append(box)
            sleep(human.reaction())       # temps de repérer le bouton apparu
            log.info("  clic sur %s", value)
            click_box(click_area(value, box))


def fishing_prompt_detector(cfg: Config) -> Optional[BannerDetector]:
    if not cfg.fishing_prompt_template.exists():
        return None
    return BannerDetector(cfg.fishing_prompt_template, cfg.message_threshold, cfg.banner_scales)


def enter_fishing_mode(cfg: Config, prompt: Optional[BannerDetector],
                       restocker: Optional[Restocker]) -> bool:
    """Invite « Fishing » visible (pas encore installé au point de pêche) :
    appuie sur F, puis lance la pêche avec « Start Fishing » si le panneau de
    préparation s'ouvre et qu'un appât est équipé (sinon le réapprovisionnement
    s'en charge)."""
    if prompt is None or not prompt.detect(grab(search_region())):
        return False
    log.info("Invite « Fishing » visible : appui sur %s pour s'installer.", cfg.key_action.upper())
    sleep(human.reaction())
    tap(cfg.key_action, cfg.action_press_duration)
    sleep(human.within((1.0, 1.8)))             # ouverture du panneau de préparation
    if restocker is not None and not restocker.bait_missing():
        try:
            box = restocker.wait_for_button("start_fishing", timeout=2.0, quiet=True)
        except RestockFailed:
            return True                         # pas de panneau : déjà en mode pêche
        sleep(human.reaction())
        log.info("Clic sur « Start Fishing ».")
        click_box(box)
        sleep(human.within((1.0, 1.8)))
    return True


def restock_or_stop(cfg: Config, restocker: Optional[Restocker]) -> None:
    if restocker is None:
        raise BlockedStop("plus d'appâts : réapprovisionnement désactivé, ou déjà tenté "
                          "juste avant sans effet (plus de coquillages ?)")
    try:
        restocker.run()
    except RestockFailed as exc:
        raise BlockedStop(f"réapprovisionnement impossible ({exc})") from None


def run_cycle(cfg: Config, banner: BannerDetector, gauge: GaugeDetector, number: int,
              results: Optional[MessageDetector] = None,
              stoppers: Optional[MessageDetector] = None,
              no_bait: Optional[BannerDetector] = None,
              restocker: Optional[Restocker] = None,
              may_restock: bool = True,
              may_restock_blind: bool = False,
              fishing_prompt: Optional[BannerDetector] = None) -> CycleRecord:
    """Un cycle complet ; renvoie son résultat pour les statistiques.

    may_restock       : False juste après un réapprovisionnement (pas deux d'affilée).
    may_restock_blind : sans bannière « Equip bait » connue, tenter un
                        réapprovisionnement si la bannière de prise n'arrive pas."""
    results = results or MessageDetector([], cfg.message_threshold, cfg.banner_scales)
    stoppers = stoppers or MessageDetector([], cfg.message_threshold, cfg.banner_scales)
    check_blockers(cfg, stoppers)

    # Prérequis : invite « Fishing » visible -> F pour s'installer au point de pêche.
    enter_fishing_mode(cfg, fishing_prompt, restocker)

    # Avant le cycle : case d'appât vide -> achat et équipement d'appâts.
    if restocker is not None and restocker.bait_missing():
        log.warning("Aucun appât équipé (case « Not Selected »).")
        restock_or_stop(cfg, restocker if may_restock else None)
        return CycleRecord(number, "restock")

    # Étape 1 : lancement
    log.info("Étape 1 : appui sur %s", cfg.key_action.upper())
    tap(cfg.key_action, cfg.action_press_duration)
    sleep(human.within(cfg.start_delay))

    # Étape 2 : bannière -> second appui après un temps de réaction humain
    waited = time.perf_counter()
    seen = wait_for_banner(cfg, banner, no_bait)
    if seen == "no_bait":
        restock_or_stop(cfg, restocker if may_restock else None)
        return CycleRecord(number, "restock")
    if seen is None:
        log.warning("Bannière non détectée après %.0f s, cycle abandonné.", cfg.banner_timeout)
        check_blockers(cfg, stoppers)
        if may_restock_blind and may_restock and restocker is not None:
            log.info("Peut-être plus d'appâts : tentative de réapprovisionnement.")
            restock_or_stop(cfg, restocker)
            return CycleRecord(number, "restock")
        return CycleRecord(number, "no_banner")
    banner_wait = time.perf_counter() - waited
    sleep(human.reaction() if human.cfg.enabled else rand(cfg.reaction_delay))
    tap(cfg.key_action, cfg.action_press_duration)

    # Étape 3 : régulation de la jauge
    if not wait_for_gauge(cfg, gauge):
        log.warning("Jauge non détectée après %.0f s, cycle abandonné.", cfg.gauge_appear_timeout)
        return CycleRecord(number, "no_gauge", banner_wait)
    track = track_gauge(cfg, gauge)
    record = CycleRecord(number, "timeout", banner_wait, track.duration,
                         track.presses_left, track.presses_right, track.in_zone_ratio)
    if not track.disappeared:
        return record

    # Étape 4 : jauge disparue -> résultat, puis appui sur F
    record.outcome = wait_for_result(cfg, results)
    sleep(human.reaction() if human.cfg.enabled else rand(cfg.reaction_delay))
    log.info("Étape 4 : jauge disparue, appui sur %s", cfg.key_action.upper())
    tap(cfg.key_action, cfg.action_press_duration)
    if record.outcome == "unknown" and results:
        record.outcome = wait_for_result(cfg, results)   # message affiché après F ?
    return record


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
    log.info("Affichez la jauge à l'écran : capture dans %.0f s…", cfg.setup_countdown)
    sleep(cfg.setup_countdown)
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
    annotated = annotate(frame, state)
    log.info("Lignes jaunes verticales trouvées : %d", len(detector.yellow_lines(yellow)))
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
    p.add_argument("--record", action="store_true",
                   help="enregistre les images annotées du suivi de jauge dans debug/track/")
    p.add_argument("--reset", action="store_true",
                   help="oublie les zones mémorisées (nouvelle détection automatique)")
    p.add_argument("--config", type=Path, default=CONFIG_FILE, help="fichier de réglages")
    p.add_argument("--banner-region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    p.add_argument("--gauge-region", type=int, nargs=4, metavar=("L", "T", "W", "H"))
    p.add_argument("--template", type=Path, help="image de la bannière")
    p.add_argument("--window", metavar="TITRE",
                   help="titre (ou partie du titre) de la fenêtre du jeu")
    p.add_argument("--list-windows", action="store_true",
                   help="affiche les titres des fenêtres ouvertes puis quitte")
    p.add_argument("--duration", metavar="DURÉE",
                   help="arrêt après cette durée (ex. 2h, 1h30, 90m)")
    p.add_argument("--max-fish", type=int, metavar="N", help="arrêt après N poissons")
    p.add_argument("--max-failures", type=int, metavar="N",
                   help="arrêt après N cycles ratés de suite (défaut 3)")
    p.add_argument("--no-breaks", action="store_true",
                   help="pas de pauses de 5-15 min entre les sessions de jeu")
    p.add_argument("--robotic", action="store_true",
                   help="désactive le comportement humain (délais uniformes, sans imperfections)")
    p.add_argument("--invert", action="store_true", help="inverse Q et D")
    p.add_argument("--input", choices=("auto", "scancode", "pyautogui"),
                   help="méthode d'envoi des touches (auto = scancode sous Windows)")
    p.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(config_path=args.config)
    if load_config_file(cfg, args.config):
        log.info("Réglages chargés depuis %s", args.config)
    if args.reset:
        cfg.banner_region = cfg.gauge_region = None
        log.info("Zones oubliées : elles seront redétectées automatiquement.")
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
    if args.window:
        cfg.window_title = args.window
    if args.record:
        cfg.record_dir = Path("debug") / "track"
    if args.duration:
        cfg.session.max_duration = parse_duration(args.duration)
    if args.max_fish:
        cfg.session.max_fish = args.max_fish
    if args.max_failures:
        cfg.max_consecutive_failures = args.max_failures
    if args.no_breaks:
        cfg.session.breaks_enabled = False
    if args.robotic:
        cfg.human.enabled = False
    return cfg


def take_break(seconds: float) -> None:
    """Pause longue entre deux sessions de jeu (interruptible : Échap, F8)."""
    resume_at = time.strftime("%H:%M", time.localtime(time.time() + seconds))
    log.info("Pause de session : %s (reprise vers %s). F8 pour reprendre tout de suite.",
             format_duration(seconds), resume_at)
    sleep(seconds)
    human.rest()


def restock_tools(cfg: Config) -> Tuple[Optional[BannerDetector], Optional[Restocker]]:
    """Détecteur « Equip bait before fishing » et réapprovisionneur, si disponibles."""
    no_bait = None
    if cfg.no_bait_template.exists():
        no_bait = BannerDetector(cfg.no_bait_template, cfg.banner_threshold, cfg.banner_scales)
    if not cfg.restock_enabled:
        return no_bait, None
    try:
        restocker = Restocker(cfg)
    except FileNotFoundError as exc:
        log.warning("Réapprovisionnement indisponible : %s", exc)
        return no_bait, None
    log.info("Réapprovisionnement automatique en appâts : activé%s.",
             "" if no_bait else " (sans templates/no_bait.png : tenté si la bannière n'arrive pas)")
    return no_bait, restocker


def run_loop(cfg: Config, args: argparse.Namespace, banner: BannerDetector,
             gauge: GaugeDetector, stats: Stats) -> None:
    """Boucle principale : cycles, pauses, sessions, plafonds et arrêts sur blocage."""
    planner = SessionPlanner(cfg.session)
    results, stoppers = result_detector(cfg), stop_detector(cfg)
    if results:
        log.info("Messages de résultat reconnus : %s", ", ".join(results.names()))
    if stoppers:
        log.info("Messages bloquants surveillés : %s", ", ".join(stoppers.names()))
    no_bait, restocker = restock_tools(cfg)
    fishing_prompt = fishing_prompt_detector(cfg)

    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        reason = planner.finished(stats.fish)
        if reason:
            log.info("Fin : %s.", reason)
            alert("done")
            return
        try:
            pause = planner.break_due()
            if pause:
                take_break(pause)
                planner.after_break()
                wait_for_focus()

            cycle += 1
            log.info("===== Cycle %d =====", cycle)
            # Pas deux réapprovisionnements d'affilée : évite une boucle d'achats
            # quand il n'y a plus de coquillages.
            just_restocked = bool(stats.records) and stats.records[-1].outcome == "restock"
            record = run_cycle(
                cfg, banner, gauge, cycle, results, stoppers, no_bait, restocker,
                may_restock=not just_restocked,
                may_restock_blind=no_bait is None and cfg.restock_on_no_banner,
                fishing_prompt=fishing_prompt)
            stats.add(record)
            log.info("%s", stats.line(record))
            if stats.consecutive_failures >= cfg.max_consecutive_failures:
                raise BlockedStop(f"{stats.consecutive_failures} cycles ratés de suite")

            wait = human.cycle_pause(cfg.cycle_pause)
            log.info("Pause de %.1f s.", wait)
            sleep(wait)
        except ResumeRequested:
            planner.after_break()           # une pause manuelle compte comme une pause
            human.rest()
            log.info("Nouveau cycle après la reprise.")
            sleep(human.within((1.0, 2.0)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    global keyboard, target, human, PAUSE_VK
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cfg = build_config(args)
    except ValueError as exc:
        log.error("%s", exc)
        return 1
    human = Humanizer(cfg.human)
    PAUSE_VK = cfg.pause_vk
    _pause_toggled()            # ignore un appui sur F8 antérieur au lancement
    stats = Stats(cfg.stats_path)

    pyautogui.FAILSAFE = True   # souris dans un coin = arrêt immédiat
    pyautogui.PAUSE = 0         # les délais sont gérés explicitement

    try:
        if args.list_windows:
            for hwnd in GameWindow.list_windows():
                print(f"  {GameWindow.window_title(hwnd)}")
            return 0
        if args.setup:
            return setup(cfg, args.config)
        if args.calibrate:
            calibrate(cfg, Path("debug"))
            return 0

        keyboard = make_keyboard(cfg.input_backend)
        keyboard.prepare((cfg.key_action, cfg.key_left, cfg.key_right, cfg.bait_menu_key, "esc"))
        log.info("Envoi des touches : %s", keyboard.name)

        banner = BannerDetector(cfg.banner_template, cfg.banner_threshold, cfg.banner_scales)
        gauge = GaugeDetector(cfg)

        log.info("F8 : pause/reprise. Échap ou souris dans un coin de l'écran : arrêt.")
        log.info("Comportement humain : %s. Pauses de session : %s.",
                 "activé" if cfg.human.enabled else "désactivé",
                 "activées" if cfg.session.breaks_enabled else "désactivées")
        if sys.platform == "win32":
            target = pick_target(cfg, cfg.startup_countdown)
            warn_if_input_blocked(target)
        else:
            sleep(cfg.startup_countdown)

        if cfg.gauge_region is None or cfg.banner_region is None:
            log.info("Zones non mémorisées : recherche dans la fenêtre du jeu au premier "
                     "cycle, puis verrouillage automatique.")

        try:
            run_loop(cfg, args, banner, gauge, stats)
        except ResumeRequested:
            pass  # F8 pressé avant le premier cycle : rien à reprendre
    except ResumeRequested:
        log.info("Pause/reprise pendant le démarrage : relancez le bot.")
    except BlockedStop as exc:
        log.error("Arrêt automatique : %s.", exc)
        alert("error")
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
        if stats.records:
            log.info("\n%s", stats.summary())
            if cfg.stats_path is not None:
                log.info("Détail des cycles : %s", cfg.stats_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
