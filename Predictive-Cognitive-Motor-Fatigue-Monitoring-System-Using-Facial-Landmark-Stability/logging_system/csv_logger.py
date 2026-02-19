"""
logging_system/csv_logger.py
=============================
Daily CSV logger for the Driver Fatigue Monitoring System.

Logs all per-frame features, FI, alert level, and confidence to a
daily-rotated CSV file.  Also integrates with Python's built-in
`logging` module for structured text logs.

CSV columns:
    timestamp, ear_avg, ear_relative, perclos, blinks_per_min,
    blink_duration_ms, blink_variance, mar, mouth_opening_index,
    is_yawning, head_pitch, head_yaw, head_roll, head_deviation,
    gaze_drift, microsaccade_variance, neck_stability,
    physical_fatigue, cognitive_fatigue, fatigue_index,
    fatigue_index_smooth, fatigue_level, confidence, alert_level,
    is_recovering
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from features.eye_features import EyeFeatures
from features.mouth_features import MouthFeatures
from features.head_features import HeadFeatures
from features.gaze_features import GazeFeatures
from features.neck_features import NeckFeatures
from models.fatigue_model import FatigueResult
from models.confidence import ConfidenceResult
from alerts.alert_manager import AlertState
from recovery.recovery_tracker import RecoveryState


_CSV_HEADER = [
    "timestamp",
    "ear_avg", "ear_relative", "perclos", "blinks_per_min",
    "blink_duration_ms", "blink_variance",
    "mar", "mouth_opening_index", "is_yawning",
    "head_pitch", "head_yaw", "head_roll", "head_deviation",
    "gaze_drift", "microsaccade_variance", "neck_stability",
    "physical_fatigue", "cognitive_fatigue",
    "fatigue_index", "fatigue_index_smooth", "fatigue_level",
    "confidence", "alert_level", "is_recovering",
]


class CSVLogger:
    """
    Daily CSV file logger.

    Creates a new file each day in the log directory:
        ``logs/fatigue_log_YYYY-MM-DD.csv``

    Rate-limited to write at most once every ``interval`` seconds
    to reduce I/O overhead during real-time operation.

    Parameters
    ----------
    log_dir : str or Path
        Directory for CSV files (created if absent).
    interval : float
        Minimum seconds between writes.
    log_level : str
        Python logging level for the text logger.
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        interval: float = 1.0,
        log_level: str = "INFO",
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._interval = interval
        self._last_write: float = 0.0

        self._current_date: Optional[str] = None
        self._csv_file = None
        self._csv_writer = None

        # Python text logger
        self._logger = logging.getLogger("fatigue_system")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        if not self._logger.handlers:
            console = logging.StreamHandler()
            console.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self._logger.addHandler(console)

            # File handler (rotating daily in same log dir)
            fh = logging.FileHandler(
                self._log_dir / "fatigue_system.log",
                encoding="utf-8",
            )
            fh.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s — %(message)s",
                )
            )
            self._logger.addHandler(fh)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # ──────────────────────────────────────────────────────────────────
    # CSV rotation
    # ──────────────────────────────────────────────────────────────────

    def _ensure_csv(self) -> None:
        """Open / rotate the daily CSV file."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._close_csv()
            self._current_date = today
            path = self._log_dir / f"fatigue_log_{today}.csv"
            write_header = not path.exists()
            self._csv_file = open(path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow(_CSV_HEADER)
                self._logger.info(f"CSV log started: {path}")

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def log(
        self,
        eye: EyeFeatures,
        mouth: MouthFeatures,
        head: HeadFeatures,
        gaze: GazeFeatures,
        neck: NeckFeatures,
        fatigue: FatigueResult,
        confidence: ConfidenceResult,
        alert: AlertState,
        recovery: RecoveryState,
    ) -> None:
        """
        Write one row to the daily CSV (rate-limited).
        """
        now = time.monotonic()
        if (now - self._last_write) < self._interval:
            return

        self._last_write = now
        self._ensure_csv()

        if self._csv_writer is None:
            return

        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            f"{eye.ear_avg:.4f}",
            f"{eye.ear_relative:.4f}",
            f"{eye.perclos:.4f}",
            f"{eye.blinks_per_minute:.1f}",
            f"{eye.blink_duration_ms:.1f}",
            f"{eye.blink_variance:.2f}",
            f"{mouth.mar:.4f}",
            f"{mouth.mouth_opening_index:.2f}",
            str(int(mouth.is_yawning)),
            f"{head.pitch:.2f}",
            f"{head.yaw:.2f}",
            f"{head.roll:.2f}",
            f"{head.deviation_score:.4f}",
            f"{gaze.gaze_drift:.4f}",
            f"{gaze.microsaccade_variance:.4f}",
            f"{neck.stability_score:.4f}",
            f"{fatigue.physical_fatigue:.4f}",
            f"{fatigue.cognitive_fatigue:.4f}",
            f"{fatigue.fatigue_index:.4f}",
            f"{fatigue.fatigue_index_smooth:.4f}",
            fatigue.level.name,
            f"{confidence.composite:.4f}",
            str(int(alert.level)),
            str(int(recovery.is_recovering)),
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._close_csv()
        self._logger.info("CSV logger closed.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
