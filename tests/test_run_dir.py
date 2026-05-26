from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_reward_density.run_dir import create_run_dir


def test_create_run_dir_creates_expected_layout(tmp_path: Path):
    rd = create_run_dir(tmp_path, run_id="test-001")
    assert rd.path == tmp_path / "test-001"
    assert (rd.path / "checkpoints").is_dir()
    assert rd.env.is_file()
    assert rd.pip_freeze.is_file()
    assert rd.metrics.is_file()
    assert not rd.config.exists()  # only written when config is supplied


def test_env_snapshot_contains_expected_keys(tmp_path: Path):
    rd = create_run_dir(tmp_path, run_id="test-002")
    env = json.loads(rd.env.read_text())
    assert env["run_id"] == "test-002"
    for key in (
        "python_version",
        "platform",
        "executable",
        "git_sha",
        "git_branch",
        "git_dirty",
        "argv",
        "env",
        "created_at_utc",
    ):
        assert key in env, key


def test_create_run_dir_writes_config_when_supplied(tmp_path: Path):
    config = {"lr": 1e-4, "rank": 4, "regime": "outcome"}
    rd = create_run_dir(tmp_path, run_id="test-003", config=config)
    assert json.loads(rd.config.read_text()) == config


def test_create_run_dir_refuses_to_overwrite(tmp_path: Path):
    create_run_dir(tmp_path, run_id="dup")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, run_id="dup")


def test_default_run_id_is_utc_timestamp(tmp_path: Path):
    rd = create_run_dir(tmp_path)
    # Format: YYYYMMDDTHHMMSSZ → 16 chars
    assert len(rd.run_id) == 16
    assert rd.run_id.endswith("Z")
    assert "T" in rd.run_id
