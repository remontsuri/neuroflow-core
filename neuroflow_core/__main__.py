"""NeuroFlow CLI entry point — `python -m neuroflow_core`.

Starts the Telegram ingestor with default config when run directly.
"""

import logging
import os

from neuroflow_core.telegram_ingestor import config_from_env


def _setup_logging() -> None:
    """Configure structured logging from env."""
    level = os.environ.get("NEUROFLOW_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main() -> None:
    """Entry point: prints config summary for CLI invocation."""
    _setup_logging()
    config = config_from_env()
    logging.getLogger(__name__).info(
        "Starting ingestor (poll_interval=%ds)", config.poll_interval_s
    )
    # TelegramIngestor is started by Hermes gateway — this CLI is a stub
    # for future `neuroflow run` / `neuroflow ingest` / `neuroflow export`.


if __name__ == "__main__":
    main()
