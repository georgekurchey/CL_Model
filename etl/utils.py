from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_write_csv(df, path: Path, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if meta:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path
