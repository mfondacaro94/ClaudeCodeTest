"""Shared utilities: logging, serialization, rate-limited requests."""

import logging
import time
import json
import os
import ndjson
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_ODDS = PROJECT_ROOT / "data" / "odds"
MODELS_SAVED = PROJECT_ROOT / "models" / "saved"
LOGS_DIR = PROJECT_ROOT / "logs"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(LOGS_DIR / "mlb_predictor.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class RateLimitedSession:
    """Requests session with built-in delay and retry logic."""

    def __init__(self, delay: float = 5.0, max_retries: int = 5):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        self.delay = delay
        self.max_retries = max_retries
        self._last_request = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=30, **kwargs)
                self._last_request = time.time()
                if resp.status_code == 429:
                    wait = min(120, self.delay * (2 ** attempt))
                    logging.warning(f"Rate limited (429). Waiting {wait:.0f}s (attempt {attempt}/{self.max_retries})...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise
                wait = self.delay * (2 ** attempt)
                logging.warning(f"Request failed ({e}). Retry {attempt}/{self.max_retries} in {wait:.0f}s")
                time.sleep(wait)

        # All retries exhausted (e.g. repeated 429s)
        raise requests.RequestException(f"All {self.max_retries} retries exhausted for {url}")


def load_progress(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            return ndjson.load(f)
    return []


def save_progress(path: Path, records: list[dict]):
    with open(path, "w") as f:
        ndjson.dump(records, f)


def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
