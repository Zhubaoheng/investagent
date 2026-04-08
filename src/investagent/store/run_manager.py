"""Run lifecycle manager: creation, isolation, resume, completion.

Each run gets an isolated directory under data/runs/ with a unique ID.
Resume finds the latest incomplete run matching the given criteria.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    run_id: str
    mode: str  # "overnight" | "backtest"
    status: str  # "running" | "completed" | "failed"
    started_at: str
    finished_at: str | None = None
    as_of_date: str | None = None
    config: dict = field(default_factory=dict)
    progress: dict = field(default_factory=dict)
    error: str | None = None


class RunManager:
    """Manages run lifecycle under data/runs/."""

    def __init__(self, data_root: Path) -> None:
        self._runs_dir = data_root / "runs"

    def create_run(
        self,
        mode: str,
        config: dict | None = None,
        as_of_date: str | None = None,
    ) -> RunMetadata:
        """Create a new isolated run directory."""
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        short_id = uuid.uuid4().hex[:4]
        run_id = f"{mode}_{ts}_{short_id}"

        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)

        meta = RunMetadata(
            run_id=run_id,
            mode=mode,
            status="running",
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            as_of_date=as_of_date,
            config=config or {},
        )
        self._write_meta(run_id, meta)
        logger.info("Created run %s at %s", run_id, run_dir)
        return meta

    def find_resumable(
        self,
        mode: str,
        as_of_date: str | None = None,
    ) -> RunMetadata | None:
        """Find the latest incomplete run matching criteria."""
        if not self._runs_dir.exists():
            return None

        candidates: list[RunMetadata] = []
        for run_dir in self._runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            meta = self._read_meta(run_dir.name)
            if meta is None:
                continue
            if meta.mode != mode:
                continue
            if meta.status != "running":
                continue
            if as_of_date is not None and meta.as_of_date != as_of_date:
                continue
            candidates.append(meta)

        if not candidates:
            return None

        # Sort by started_at descending, return latest
        candidates.sort(key=lambda m: m.started_at, reverse=True)
        latest = candidates[0]
        logger.info("Found resumable run: %s (started %s)", latest.run_id, latest.started_at)
        return latest

    def get_run_dir(self, run_id: str) -> Path:
        return self._runs_dir / run_id

    def complete_run(self, run_id: str) -> None:
        meta = self._read_meta(run_id)
        if meta:
            meta.status = "completed"
            meta.finished_at = datetime.now(tz=timezone.utc).isoformat()
            self._write_meta(run_id, meta)
            logger.info("Run %s completed", run_id)

    def fail_run(self, run_id: str, error: str) -> None:
        meta = self._read_meta(run_id)
        if meta:
            meta.status = "failed"
            meta.finished_at = datetime.now(tz=timezone.utc).isoformat()
            meta.error = error
            self._write_meta(run_id, meta)
            logger.info("Run %s failed: %s", run_id, error)

    def update_progress(self, run_id: str, progress: dict) -> None:
        meta = self._read_meta(run_id)
        if meta:
            meta.progress = progress
            self._write_meta(run_id, meta)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _meta_path(self, run_id: str) -> Path:
        return self._runs_dir / run_id / "run.json"

    def _write_meta(self, run_id: str, meta: RunMetadata) -> None:
        path = self._meta_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_meta(self, run_id: str) -> RunMetadata | None:
        path = self._meta_path(run_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return RunMetadata(**raw)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
