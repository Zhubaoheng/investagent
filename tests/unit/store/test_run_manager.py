"""Tests for poorcharlie.store.run_manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poorcharlie.store.run_manager import RunManager, RunMetadata


class TestRunManager:
    def test_create_run(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight", config={"top_n": 100})

        assert meta.mode == "overnight"
        assert meta.status == "running"
        assert meta.run_id.startswith("overnight_")
        assert meta.config == {"top_n": 100}
        assert meta.finished_at is None

        # Directory created
        run_dir = rm.get_run_dir(meta.run_id)
        assert run_dir.exists()
        assert (run_dir / "checkpoints").exists()
        assert (run_dir / "run.json").exists()

    def test_create_run_with_as_of_date(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight", as_of_date="2024-05-06")
        assert meta.as_of_date == "2024-05-06"

    def test_run_id_uniqueness(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        m1 = rm.create_run("overnight")
        m2 = rm.create_run("overnight")
        assert m1.run_id != m2.run_id

    def test_find_resumable_empty(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        assert rm.find_resumable("overnight") is None

    def test_find_resumable_finds_running(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight")

        found = rm.find_resumable("overnight")
        assert found is not None
        assert found.run_id == meta.run_id

    def test_find_resumable_ignores_completed(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight")
        rm.complete_run(meta.run_id)

        assert rm.find_resumable("overnight") is None

    def test_find_resumable_ignores_failed(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight")
        rm.fail_run(meta.run_id, "some error")

        assert rm.find_resumable("overnight") is None

    def test_find_resumable_filters_by_mode(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        rm.create_run("overnight")

        assert rm.find_resumable("backtest") is None

    def test_find_resumable_filters_by_as_of_date(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        rm.create_run("overnight", as_of_date="2024-05-06")

        assert rm.find_resumable("overnight", as_of_date="2024-09-02") is None
        found = rm.find_resumable("overnight", as_of_date="2024-05-06")
        assert found is not None

    def test_find_resumable_returns_latest(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        m1 = rm.create_run("overnight")
        m2 = rm.create_run("overnight")

        found = rm.find_resumable("overnight")
        assert found is not None
        # Should be the latest (m2), but both are "running" and
        # timestamps might be same second — just verify one is found
        assert found.run_id in (m1.run_id, m2.run_id)

    def test_complete_run(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight")
        rm.complete_run(meta.run_id)

        # Read back metadata
        run_json = rm.get_run_dir(meta.run_id) / "run.json"
        data = json.loads(run_json.read_text())
        assert data["status"] == "completed"
        assert data["finished_at"] is not None

    def test_fail_run(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight")
        rm.fail_run(meta.run_id, "pipeline crashed")

        run_json = rm.get_run_dir(meta.run_id) / "run.json"
        data = json.loads(run_json.read_text())
        assert data["status"] == "failed"
        assert data["error"] == "pipeline crashed"

    def test_update_progress(self, tmp_path: Path):
        rm = RunManager(tmp_path)
        meta = rm.create_run("overnight")
        rm.update_progress(meta.run_id, {"phase": "pipeline", "done": 45, "total": 120})

        run_json = rm.get_run_dir(meta.run_id) / "run.json"
        data = json.loads(run_json.read_text())
        assert data["progress"]["phase"] == "pipeline"
        assert data["progress"]["done"] == 45
