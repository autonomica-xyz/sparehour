from __future__ import annotations

import pytest

from sparehour.cli import main
from sparehour.store import Store


def test_revoke_token_cli_disables_credential(tmp_path, capsys):
    db = tmp_path / "control.sqlite3"
    store = Store(db)
    raw = store.issue_token(
        role="agent",
        subject="worker-a",
        token_id="agent-worker-a",
    )
    store.close()

    assert main([
        "revoke-token", "--db", str(db), "--token-id", "agent-worker-a"
    ]) == 0
    assert "revoked token agent-worker-a" in capsys.readouterr().out

    store = Store(db)
    with pytest.raises(PermissionError):
        store.authenticate(raw)
    store.close()

    assert main([
        "revoke-token", "--db", str(db), "--token-id", "agent-worker-a"
    ]) == 1
    assert "not found or already disabled" in capsys.readouterr().err


def test_agent_inventory_advertises_normalized_arch(tmp_path, capsys, monkeypatch):
    import json

    import sparehour.cli as cli_mod

    for machine, expected in (("x86_64", "amd64"), ("aarch64", "arm64"), ("riscv", "riscv")):
        monkeypatch.setattr(cli_mod.platform, "machine", lambda m=machine: m)
        assert cli_mod.main([
            "agent-inventory",
            "--agent-id", "w",
            "--capability", "demo",
            "--fake-gpus", "1",
            "--slot-label", "arch=amd64",
            "--slot-label", "memory=discrete",
        ]) == 0
        slots = json.loads(capsys.readouterr().out)
        assert slots[0]["labels"]["arch"] == expected  # hardware wins over operator
        assert "memory" not in slots[0]["labels"]  # UMA tag stripped from labels


def test_agent_run_registration_advertises_normalized_arch(tmp_path, monkeypatch):
    import sparehour.cli as cli_mod

    monkeypatch.setattr(cli_mod.platform, "machine", lambda: "x86_64")
    real_config = cli_mod.AgentConfig
    captured = []

    def fake_config(*args, **kwargs):
        registration = kwargs.get("registration") or (
            args[0].registration if args else None
        )
        captured.append(registration)
        return real_config(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "AgentConfig", fake_config)
    monkeypatch.setattr(cli_mod.AgentRunner, "run", lambda self, once=False: None)
    cli_mod.main([
        "agent",
        "--agent-id", "w",
        "--capability", "demo",
        "--fake-gpus", "1",
        "--workdir", str(tmp_path / "w"),
        "--endpoint", "http://localhost:1",
        "--token", "t",
        "--label", "arch=arm64",
    ])
    registration = captured[0]
    assert registration.labels["arch"] == "amd64"
    assert registration.slots[0].labels["arch"] == "amd64"
