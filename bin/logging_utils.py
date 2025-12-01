# bin/logging_utils.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import BASE_DIR, LoggingConfig

_cfg = LoggingConfig()


def _build_file_template() -> Path:
    """
    Baut den Template-Pfad für Logfiles auf Basis von LoggingConfig.

    Beispiel:
      _cfg.path = "log"
      _cfg.log_file = "{name}.log"

    -> BASE_DIR / "log" / "{name}.log"
    """
    base = Path(_cfg.path) / _cfg.log_file
    if not base.is_absolute():
        base = BASE_DIR / base
    return base


def setup_logging() -> None:
    """
    Setzt zentrales Logging auf Basis der LoggingConfig auf.
    Wird nur einmal ausgeführt.
    """
    root = logging.getLogger()
    if root.handlers:
        # schon konfiguriert
        return

    level_name = _cfg.level
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console-Handler
    if _cfg.to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    # File-Handler (rotierend)
    # Variante A: eine zentrale Datei (wenn KEIN {name} im Dateinamen)
    # Variante B: per-Logger-Dateien (wenn {name} im Dateinamen, dann macht get_logger das)
    if _cfg.to_file:
        file_template = _build_file_template()

        # Wenn KEIN {name} im Dateinamen steckt → zentraler Root-File-Handler
        if "{name}" not in file_template.name:
            file_template.parent.mkdir(parents=True, exist_ok=True)

            fh = RotatingFileHandler(
                file_template,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=5,
                encoding="utf-8",
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root.addHandler(fh)

    # Noise reduzieren
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str = "rag") -> logging.Logger:
    """
    Hole einen Logger mit globaler Konfiguration.

    Wenn LoggingConfig.log_file z.B. '{name}.log' ist,
    bekommt jeder Logger eine eigene rotierende Logdatei unter <path>/<loggername>.log.
    """
    setup_logging()
    logger = logging.getLogger(name)

    if _cfg.to_file:
        file_template = _build_file_template()

        # Nur wenn {name} im Dateinamen → per-Logger-File-Handler
        if "{name}" in file_template.name:
            level_name = _cfg.level
            level = getattr(logging, level_name, logging.INFO)

            # Prüfen, ob wir schon einen per-Logger-FileHandler gesetzt haben
            has_handler = any(
                isinstance(h, RotatingFileHandler) and getattr(h, "_per_logger", False)
                for h in logger.handlers
            )
            if not has_handler:
                # Dateiname mit Loggernamen ersetzen
                filename = file_template.name.format(name=name)
                log_path = file_template.with_name(filename)

                log_path.parent.mkdir(parents=True, exist_ok=True)

                formatter = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )

                fh = RotatingFileHandler(
                    log_path,
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
                fh.setLevel(level)
                fh.setFormatter(formatter)
                # Marker, damit wir ihn später wiedererkennen
                fh._per_logger = True  # type: ignore[attr-defined]
                logger.addHandler(fh)

    return logger
