"""
session.py - Comportement humain, sessions de jeu et statistiques.

Logique pure (aucune capture d'écran ni touche) : tout est testable hors jeu.

- Humanizer      : temps de réaction réalistes, fatigue, petites imperfections,
                   pauses entre cycles plus variées.
- SessionPlanner : alternance jeu / pause (ex. 20-45 min puis 5-15 min) et
                   plafonds (durée totale, nombre de poissons).
- Stats          : résultat de chaque cycle, journal CSV, résumé de session.
"""

from __future__ import annotations

import csv
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

Range = Tuple[float, float]
Clock = Callable[[], float]


# --------------------------------------------------------------------------- #
# Comportement humain
# --------------------------------------------------------------------------- #
@dataclass
class HumanConfig:
    enabled: bool = True
    # Temps de réaction (bannière -> F) : loi log-normale, comme chez un humain
    # (la plupart autour de 250-350 ms, quelques réactions nettement plus lentes).
    reaction_median: float = 0.28
    reaction_sigma: float = 0.25
    reaction_bounds: Range = (0.16, 2.0)
    slow_reaction_chance: float = 0.07
    slow_reaction_extra: Range = (0.3, 1.0)
    # Fatigue : les réactions ralentissent légèrement au fil d'une session.
    fatigue_per_minute: float = 0.004       # +0.4 % par minute de jeu
    fatigue_max: float = 1.25               # au plus +25 %
    # Imperfections pendant la jauge.
    skip_correction_chance: float = 0.04    # correction remise à un peu plus tard
    late_correction_delay: Range = (0.08, 0.25)
    overshoot_chance: float = 0.05          # appui un peu trop long
    overshoot_factor: Range = (1.3, 1.7)
    # Pauses entre cycles : surtout courtes, parfois longues.
    long_pause_chance: float = 0.12
    long_pause: Range = (5.0, 10.0)
    very_long_pause_chance: float = 0.03
    very_long_pause: Range = (10.0, 20.0)


class Humanizer:
    def __init__(self, cfg: Optional[HumanConfig] = None,
                 rng: Optional[random.Random] = None, clock: Clock = time.monotonic):
        self.cfg = cfg or HumanConfig()
        self.rng = rng or random.Random()
        self.clock = clock
        self._rested_at = clock()

    # -- fatigue ---------------------------------------------------------------
    def rest(self) -> None:
        """Après une pause : la fatigue repart de zéro."""
        self._rested_at = self.clock()

    def fatigue(self) -> float:
        if not self.cfg.enabled:
            return 1.0
        minutes = (self.clock() - self._rested_at) / 60
        return min(self.cfg.fatigue_max, 1.0 + self.cfg.fatigue_per_minute * minutes)

    # -- tirages ---------------------------------------------------------------
    def within(self, bounds: Range) -> float:
        """Valeur dans [min, max]. En mode humain, plus souvent vers le bas de
        l'intervalle (loi bêta asymétrique) et décalée par la fatigue."""
        lo, hi = bounds
        if hi <= lo:
            return lo
        if not self.cfg.enabled:
            return self.rng.uniform(lo, hi)
        u = self.rng.betavariate(2.0, 3.0)
        u = min(1.0, u * self.fatigue())
        return lo + (hi - lo) * u

    def reaction(self) -> float:
        """Temps de réaction à un signal visuel (s)."""
        c = self.cfg
        lo, hi = c.reaction_bounds
        if not c.enabled:
            return c.reaction_median
        t = self.rng.lognormvariate(math.log(c.reaction_median), c.reaction_sigma)
        if self.rng.random() < c.slow_reaction_chance:
            t += self.rng.uniform(*c.slow_reaction_extra)
        return min(hi, max(lo, t * self.fatigue()))

    def cycle_pause(self, base: Range) -> float:
        """Pause entre deux cycles, jamais sous base[0] (3 s par défaut)."""
        c = self.cfg
        if not c.enabled:
            return self.rng.uniform(*base)
        r = self.rng.random()
        if r < c.very_long_pause_chance:
            pause = self.rng.uniform(*c.very_long_pause)
        elif r < c.very_long_pause_chance + c.long_pause_chance:
            pause = self.rng.uniform(*c.long_pause)
        else:
            pause = self.within(base)
        return max(base[0], pause)

    # -- imperfections -----------------------------------------------------------
    def skip_correction(self) -> Optional[float]:
        """Délai si cette correction doit être faite un peu en retard, sinon None."""
        if self.cfg.enabled and self.rng.random() < self.cfg.skip_correction_chance:
            return self.rng.uniform(*self.cfg.late_correction_delay)
        return None

    def press_factor(self) -> float:
        """Multiplicateur de durée d'appui : léger bruit, parfois un dépassement."""
        if not self.cfg.enabled:
            return 1.0
        if self.rng.random() < self.cfg.overshoot_chance:
            return self.rng.uniform(*self.cfg.overshoot_factor)
        return self.rng.uniform(0.85, 1.15)


# --------------------------------------------------------------------------- #
# Sessions de jeu
# --------------------------------------------------------------------------- #
@dataclass
class SessionConfig:
    breaks_enabled: bool = True
    play_minutes: Range = (20.0, 45.0)
    break_minutes: Range = (5.0, 15.0)
    max_duration: Optional[float] = None    # s ; None = illimité
    max_fish: Optional[int] = None          # None = illimité


class SessionPlanner:
    """Décide quand faire une pause et quand s'arrêter."""

    def __init__(self, cfg: Optional[SessionConfig] = None,
                 rng: Optional[random.Random] = None, clock: Clock = time.monotonic):
        self.cfg = cfg or SessionConfig()
        self.rng = rng or random.Random()
        self.clock = clock
        self.started_at = clock()
        self.next_break_at = self._schedule()

    def _schedule(self) -> float:
        return self.clock() + self.rng.uniform(*self.cfg.play_minutes) * 60

    def break_due(self) -> Optional[float]:
        """Durée de la pause (s) si c'est le moment d'en faire une, sinon None."""
        if not self.cfg.breaks_enabled or self.clock() < self.next_break_at:
            return None
        return self.rng.uniform(*self.cfg.break_minutes) * 60

    def after_break(self) -> None:
        self.next_break_at = self._schedule()

    def finished(self, fish: int) -> Optional[str]:
        """Raison de l'arrêt si un plafond est atteint, sinon None."""
        elapsed = self.clock() - self.started_at
        if self.cfg.max_duration is not None and elapsed >= self.cfg.max_duration:
            return f"durée maximale atteinte ({format_duration(self.cfg.max_duration)})"
        if self.cfg.max_fish is not None and fish >= self.cfg.max_fish:
            return f"nombre maximal de poissons atteint ({self.cfg.max_fish})"
        return None


_DURATION_RE = re.compile(
    r"^\s*(?:(?P<h>\d+(?:[.,]\d+)?)\s*h)?\s*(?:(?P<m>\d+(?:[.,]\d+)?)\s*m(?:in)?)?"
    r"\s*(?:(?P<s>\d+(?:[.,]\d+)?)\s*s)?\s*$", re.IGNORECASE)


def parse_duration(text: str) -> float:
    """'2h', '1h30', '90m', '45min', '30s' ou un nombre seul (minutes) -> secondes."""
    text = text.strip()
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        return float(text.replace(",", ".")) * 60
    m = re.fullmatch(r"(\d+)\s*h\s*(\d+)", text, re.IGNORECASE)   # « 1h30 »
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60
    m = _DURATION_RE.match(text)
    if not text or not m or not any(m.groupdict().values()):
        raise ValueError(f"Durée invalide : « {text} » (ex. 2h, 1h30, 90m, 45min)")
    num = {k: float(v.replace(",", ".")) if v else 0.0 for k, v in m.groupdict().items()}
    return num["h"] * 3600 + num["m"] * 60 + num["s"]


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


# --------------------------------------------------------------------------- #
# Statistiques
# --------------------------------------------------------------------------- #
OUTCOME_LABELS = {
    "success": "réussite",
    "fail": "échec",
    "unknown": "terminé",           # jauge finie, résultat non vérifiable
    "no_banner": "pas de bannière",
    "no_gauge": "pas de jauge",
    "timeout": "jauge trop longue",
}
CATCH_OUTCOMES = {"success", "unknown"}             # comptés comme poissons
FAILURE_OUTCOMES = {"fail", "no_banner", "no_gauge", "timeout"}


@dataclass
class CycleRecord:
    cycle: int
    outcome: str
    banner_wait: float = 0.0
    gauge_duration: float = 0.0
    presses_left: int = 0
    presses_right: int = 0
    in_zone_ratio: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class Stats:
    def __init__(self, csv_path: Optional[Path] = None, clock: Clock = time.monotonic):
        self.csv_path = csv_path
        self.records: List[CycleRecord] = []
        self.clock = clock
        self.started_at = clock()
        self.consecutive_failures = 0

    @property
    def fish(self) -> int:
        return sum(r.outcome in CATCH_OUTCOMES for r in self.records)

    def add(self, record: CycleRecord) -> None:
        self.records.append(record)
        if record.outcome in FAILURE_OUTCOMES:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        if self.csv_path is not None:
            self._write(record)

    def _write(self, record: CycleRecord) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[f.name for f in fields(CycleRecord)],
                                    delimiter=";")
            if new_file:
                writer.writeheader()
            row = asdict(record)
            row["in_zone_ratio"] = f"{record.in_zone_ratio:.2f}"
            row["banner_wait"] = f"{record.banner_wait:.2f}"
            row["gauge_duration"] = f"{record.gauge_duration:.2f}"
            writer.writerow(row)

    def line(self, record: CycleRecord) -> str:
        """Résumé d'un cycle pour le journal."""
        text = f"Cycle {record.cycle} : {OUTCOME_LABELS.get(record.outcome, record.outcome)}"
        if record.gauge_duration:
            text += (f" — jauge {record.gauge_duration:.1f} s, ligne dans le vert "
                     f"{record.in_zone_ratio:.0%}")
        return text + f" | total : {self.fish} poisson(s)"

    def summary(self) -> str:
        n = len(self.records)
        elapsed = self.clock() - self.started_at
        if not n:
            return f"Aucun cycle terminé ({format_duration(elapsed)})."
        counts = {k: sum(r.outcome == k for r in self.records) for k in OUTCOME_LABELS}
        gauges = [r for r in self.records if r.gauge_duration]
        lines = [
            f"Session : {format_duration(elapsed)}, {n} cycle(s), {self.fish} poisson(s)"
            f" ({self.fish / n:.0%}).",
            "Détail : " + ", ".join(f"{OUTCOME_LABELS[k]} {v}" for k, v in counts.items() if v),
        ]
        if gauges:
            avg_d = sum(r.gauge_duration for r in gauges) / len(gauges)
            avg_z = sum(r.in_zone_ratio for r in gauges) / len(gauges)
            lines.append(f"Jauge : {avg_d:.1f} s en moyenne, ligne dans le vert {avg_z:.0%}.")
        if elapsed > 60:
            lines.append(f"Rythme : {self.fish / (elapsed / 3600):.0f} poissons/heure.")
        return "\n".join(lines)
