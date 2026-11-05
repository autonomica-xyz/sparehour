from __future__ import annotations

import json
import subprocess
import threading

import pytest

from sparehour.agent import (
    AgentConfig,
    AgentError,
    AgentRunner,
    ExecutionError,
    _LeaseKeeper,
    _load_output_manifest,
    discover_nvidia_slots,
)
from sparehour.models import AgentRegistration, SlotInventory


def test_nvidia_inventory_becomes_one_slot_per_gpu():
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                "0, GPU-a, NVIDIA RTX 4090, 24564, 8.9\n"
                "1, GPU-b, NVIDIA L4, 23034, 8.9\n"
            ),
            stderr="",
        )

    slots = discover_nvidia_slots(
        "tower-a",
        capabilities=["image-generate"],
        model_packs=["sdxl-core-v1"],
        runner=runner,
    )

    assert [slot.id for slot in slots] == ["tower-a-gpu-0", "tower-a-gpu-1"]
    assert slots[0].vram_mb == 24564
    assert slots[1].model == "NVIDIA L4"


def test_lease_keeper_renews_for_non_process_executors():
    heartbeat_seen = threading.Event()

    class Client:
        def heartbeat_job(self, *_args):
            heartbeat_seen.set()
            return False

    keeper = _LeaseKeeper(
        Client(),
        job_id="job-a",
        slot_id="slot-a",
        lease_token="secret",
        heartbeat_seconds=0.01,
        lease_seconds=1,
    )
    keeper.start()
    assert heartbeat_seen.wait(0.5)
    keeper.stop()
    keeper.check()


def test_lease_keeper_surfaces_cancel_and_renewal_failure():
    cancelled_seen = threading.Event()

    class CancelledClient:
        def heartbeat_job(self, *_args):
            cancelled_seen.set()
            return True

    cancelled = _LeaseKeeper(
        CancelledClient(),
        job_id="job-a",
        slot_id="slot-a",
        lease_token="secret",
        heartbeat_seconds=0.01,
        lease_seconds=1,
    )
    cancelled.start()
    assert cancelled_seen.wait(0.5)
    with pytest.raises(ExecutionError, match="cancellation") as exc_info:
        cancelled.check()
    assert exc_info.value.code == "cancelled"
    cancelled.stop()

    failure_seen = threading.Event()

    class FailedClient:
        def heartbeat_job(self, *_args):
            failure_seen.set()
            raise AgentError("control plane unavailable")

    failed = _LeaseKeeper(
        FailedClient(),
        job_id="job-b",
        slot_id="slot-b",
        lease_token="secret",
        heartbeat_seconds=0.01,
        lease_seconds=1,
    )
    failed.start()
    assert failure_seen.wait(0.5)
    with pytest.raises(ExecutionError, match="renewal failed") as exc_info:
        failed.check()
    assert exc_info.value.code == "worker_lost"
    failed.stop()


def test_output_manifest_cannot_escape_job_workdir(tmp_path):
    workdir = tmp_path / "job"
    workdir.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text("{}", encoding="utf-8")
    (workdir / "outputs.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "name": "result",
                        "path": "../secret.json",
                        "media_type": "application/json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExecutionError, match="escapes workdir"):
        _load_output_manifest(workdir / "outputs.json", workdir)


def test_fatal_model_service_error_fails_job_then_drains_agent(tmp_path):
    slot = SlotInventory(
        id="agent-a-gpu-0", gpu_index=0, gpu_uuid="GPU-a", model="fake",
        vram_mb=8192, capabilities=["libretranslate"],
    )
    registration = AgentRegistration(
        agent_id="agent-a", provider="salad", hostname="host-a", slots=[slot]
    )

    class Client:
        def __init__(self):
            self.failed = []
            self.draining = []

        def confirm(self, *_args):
            return None

        def heartbeat_job(self, *_args):
            return False

        def fail(self, *args):
            self.failed.append(args)

        def heartbeat_agent(self, agent_id, *, draining=None):
            self.draining.append((agent_id, draining))

    client = Client()
    runner = AgentRunner(
        client,
        AgentConfig(registration=registration, workdir_root=tmp_path),
    )

    def unavailable(*_args, **_kwargs):
        raise ExecutionError("model_server_unavailable", "connection refused")

    runner._execute = unavailable
    runner._run_job(slot, {"id": "job-a", "lease_token": "lease"})

    assert client.failed[-1][-2:] == ("model_server_unavailable", "connection refused")
    assert client.draining == [("agent-a", True)]
    assert runner.stop.is_set()


def test_command_executor_uses_the_agent_python(tmp_path):
    slot = SlotInventory(
        id="agent-a-gpu-0",
        gpu_index=0,
        gpu_uuid="GPU-a",
        model="fake",
        vram_mb=1,
        capabilities=["demo-gpu"],
    )
    registration = AgentRegistration(
        agent_id="agent-a",
        provider="local",
        hostname="host-a",
        slots=[slot],
    )
    runner = AgentRunner(
        object(),
        AgentConfig(registration=registration, workdir_root=tmp_path),
    )

    class Lease:
        def check(self):
            return None

    outputs = runner._execute_process(
        slot,
        {
            "id": "job-a",
            "lease_token": "secret",
            "payload": {"message": "hello"},
            "executor": {
                "type": "command",
                "command": ["{agent_python}", "-m", "sparehour.demo_executor"],
                "environment": {},
                "timeout_seconds": 10,
            },
        },
        tmp_path,
        Lease(),
        container=False,
    )

    assert outputs[0]["name"] == "result"
    result_path = tmp_path / "result.json"
    assert json.loads(result_path.read_text(encoding="utf-8"))["message"] == "hello"


# -- unified-memory (integrated GPU) discovery -------------------------------

_GIB = 1024 * 1024 * 1024


def _nvidia_runner(stdout: str):
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    return runner


def test_uma_na_memory_becomes_budgeted_unified_slot(monkeypatch):
    monkeypatch.setattr(
        "sparehour.agent._cuda_device_table",
        lambda: {"gpu-abc": (True, 128 * _GIB)},
    )
    monkeypatch.setattr(
        "sparehour.agent._mem_available_bytes", lambda: 120 * _GIB
    )

    slots = discover_nvidia_slots(
        "spark",
        capabilities=["llm"],
        model_packs=[],
        runner=_nvidia_runner("0, GPU-ABC, NVIDIA GB10, [N/A], 12.1\n"),
    )

    assert len(slots) == 1
    slot = slots[0]
    # Budget = min(80% of device total, 80% of MemAvailable) in MiB.
    assert slot.vram_mb == int(0.8 * 120 * _GIB) // (1024 * 1024)
    assert slot.vram_mb < int(0.8 * 128 * _GIB) // (1024 * 1024)
    assert slot.labels["memory"] == "unified"
    assert slot.gpu_index == 0
    assert slot.compute_capability == "12.1"


def test_uma_slots_bind_by_uuid_not_ordinal(monkeypatch):
    monkeypatch.setattr(
        "sparehour.agent._cuda_device_table",
        lambda: {"gpu-a": (True, 120 * _GIB), "gpu-b": (True, 120 * _GIB)},
    )
    monkeypatch.setattr(
        "sparehour.agent._mem_available_bytes", lambda: 120 * _GIB
    )
    # CUDA-visible order is irrelevant; nvidia-smi index/UUID bind the slot.
    stdout = "0, GPU-b, NVIDIA GB10, [N/A], 12.1\n1, GPU-a, NVIDIA GB10, [N/A], 12.1\n"

    slots = discover_nvidia_slots(
        "spark",
        capabilities=[],
        model_packs=[],
        runner=_nvidia_runner(stdout),
    )

    assert [slot.id for slot in slots] == ["spark-gpu-0", "spark-gpu-1"]
    assert [slot.gpu_uuid for slot in slots] == ["GPU-b", "GPU-a"]
    # Two integrated devices share one pool: the budget is divided, not doubled.
    expected = int(0.8 * 120 * _GIB) // 2 // (1024 * 1024)
    assert all(slot.vram_mb == expected for slot in slots)
    assert all(slot.labels["memory"] == "unified" for slot in slots)


def test_uma_uuid_hidden_from_cuda_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "sparehour.agent._cuda_device_table", lambda: {"gpu-a": (True, 8 * _GIB)}
    )
    monkeypatch.setattr("sparehour.agent._mem_available_bytes", lambda: 8 * _GIB)

    with pytest.raises(AgentError, match="cannot see GPU GPU-hddn"):
        discover_nvidia_slots(
            "spark",
            capabilities=[],
            model_packs=[],
            runner=_nvidia_runner("0, GPU-hddn, NVIDIA GB10, [N/A], 12.1\n"),
        )


def test_na_memory_on_non_integrated_device_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "sparehour.agent._cuda_device_table", lambda: {"gpu-a": (False, 24 * _GIB)}
    )
    monkeypatch.setattr("sparehour.agent._mem_available_bytes", lambda: 32 * _GIB)

    with pytest.raises(AgentError, match="not an integrated"):
        discover_nvidia_slots(
            "tower",
            capabilities=[],
            model_packs=[],
            runner=_nvidia_runner("0, GPU-a, RTX 4090, [Not Supported], 8.9\n"),
        )


@pytest.mark.parametrize("vram", ["oops", "-5"])
def test_invalid_discrete_memory_is_rejected(monkeypatch, vram):
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, stdout=f"0, GPU-a, RTX 4090, {vram}, 8.9\n", stderr=""
        )

    with pytest.raises(AgentError, match="invalid nvidia-smi memory.total"):
        discover_nvidia_slots(
            "tower", capabilities=[], model_packs=[], runner=runner
        )


def test_low_mem_available_fails_closed_on_uma_path(monkeypatch):
    monkeypatch.setattr(
        "sparehour.agent._cuda_device_table", lambda: {"gpu-a": (True, 128 * _GIB)}
    )
    monkeypatch.setattr("sparehour.agent._mem_available_bytes", lambda: 512 * 1024)

    with pytest.raises(AgentError, match="below 1 MiB"):
        discover_nvidia_slots(
            "spark",
            capabilities=[],
            model_packs=[],
            runner=_nvidia_runner("0, GPU-a, NVIDIA GB10, [N/A], 12.1\n"),
        )


def test_unreadable_mem_available_fails_closed_on_uma_path(monkeypatch):
    monkeypatch.setattr(
        "sparehour.agent._cuda_device_table", lambda: {"gpu-a": (True, 128 * _GIB)}
    )

    class BrokenPath:
        def read_text(self, *_args, **_kwargs):
            raise OSError("no meminfo")

    monkeypatch.setattr("sparehour.agent.Path", lambda _p: BrokenPath())

    with pytest.raises(AgentError, match="MemAvailable"):
        discover_nvidia_slots(
            "spark",
            capabilities=[],
            model_packs=[],
            runner=_nvidia_runner("0, GPU-a, NVIDIA GB10, [N/A], 12.1\n"),
        )


def test_operator_memory_label_is_stripped_on_discrete_slots():
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, stdout="0, GPU-a, RTX 4090, 24564, 8.9\n", stderr=""
        )

    slots = discover_nvidia_slots(
        "tower",
        capabilities=[],
        model_packs=[],
        labels={"memory": "unified", "team": "x"},
        runner=runner,
    )

    assert slots[0].labels == {"team": "x"}


def test_cuda_driver_failure_fails_closed_on_uma_path(monkeypatch):
    def broken_table():
        raise AgentError("CUDA driver query cuInit failed with error 999")

    monkeypatch.setattr("sparehour.agent._cuda_device_table", broken_table)

    with pytest.raises(AgentError, match="cuInit"):
        discover_nvidia_slots(
            "spark",
            capabilities=[],
            model_packs=[],
            runner=_nvidia_runner("0, GPU-a, NVIDIA GB10, [N/A], 12.1\n"),
        )
