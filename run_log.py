import json
import os
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RUN_LOG_PATH = Path(__file__).parent / "output" / "run_history.jsonl"


def record_run(event, path=None):
    log_path = Path(path or os.getenv("RUN_LOG_PATH") or DEFAULT_RUN_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
