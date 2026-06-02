"""NeuroFlow CLI entry point — `python -m neuroflow_core`.

Starts the Telegram ingestor with default config when run directly.
"""

from neuroflow_core.telegram_ingestor import config_from_env


def main() -> None:
    config = config_from_env()
    print(f"[neuroflow] Starting ingestor (poll_interval={config.poll_interval_s}s)")
    # TelegramIngestor is started by Hermes gateway — this CLI is a stub
    # for future `neuroflow run` / `neuroflow ingest` / `neuroflow export`.


if __name__ == "__main__":
    main()
