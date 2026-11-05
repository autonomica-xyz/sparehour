"""GPU host agent and trusted executor boundary.

One agent process registers one slot per physical GPU. Each slot claims and
executes jobs independently, so a four-GPU host naturally runs up to four jobs
without requiring the jobs themselves to know about distributed execution.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from .models import AgentRegistration, SlotInventory


class AgentError(RuntimeError):
    pass


class ExecutionError(AgentError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

# Declared input names are validated by the workload contract, but the agent
# also treats them as filenames under workdir/inputs; the same bound keeps
# path construction safe even against a corrupt job document.
_INPUT_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,62}$")

# Unavailable-memory markers nvidia-smi prints when a field cannot be read.
_UNAVAILABLE_MEMORY_MARKERS = frozenset(
    {"[n/a]", "[not supported]", "[insufficient permissions]"}
)
# CU_DEVICE_ATTRIBUTE_INTEGRATED: nonzero when the CUDA device draws memory
# from the host's unified DRAM pool (e.g. Grace-Blackwell GB10) instead of
# dedicated VRAM, which is why nvidia-smi cannot report a memory.total.
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
_MIB = 1024 * 1024


class _CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


def _load_libcuda() -> ctypes.CDLL:
    names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise AgentError(
        "CUDA driver library not loadable; cannot probe unified-memory GPUs"
    )


def _cu_check(rc: int, what: str) -> None:
    if rc != 0:
        raise AgentError(f"CUDA driver query {what} failed with error {rc}")


def _format_cu_uuid(uuid: _CUuuid) -> str:
    hexdigest = bytes(uuid.bytes).hex()
    return (
        f"gpu-{hexdigest[0:8]}-{hexdigest[8:12]}-{hexdigest[12:16]}-"
        f"{hexdigest[16:20]}-{hexdigest[20:32]}"
    )


def _cuda_device_table(lib: ctypes.CDLL | None = None) -> dict[str, tuple[bool, int]]:
    """Lowercase "gpu-..." UUID -> (integrated, total memory bytes), all CUDA-visible devices.

    Device-level driver queries only (cuInit/cuDeviceGet*); no CUDA context is
    created, so probing neither allocates GPU memory nor disturbs running jobs.
    Visibility follows CUDA_VISIBLE_DEVICES, which is why nvidia-smi rows are
    matched by UUID here, never by ordinal.
    """
    if lib is None:
        lib = _load_libcuda()
    _cu_check(lib.cuInit(0), "cuInit")
    count = ctypes.c_int(0)
    _cu_check(lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    get_uuid = getattr(lib, "cuDeviceGetUuid_v2", None) or lib.cuDeviceGetUuid
    table: dict[str, tuple[bool, int]] = {}
    for ordinal in range(count.value):
        dev = ctypes.c_int(0)
        _cu_check(lib.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        uuid = _CUuuid()
        _cu_check(get_uuid(ctypes.byref(uuid), dev), "cuDeviceGetUuid")
        integrated = ctypes.c_int(0)
        _cu_check(
            lib.cuDeviceGetAttribute(
                ctypes.byref(integrated), _CU_DEVICE_ATTRIBUTE_INTEGRATED, dev
            ),
            "cuDeviceGetAttribute(INTEGRATED)",
        )
        total = ctypes.c_size_t(0)
        _cu_check(lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev), "cuDeviceTotalMem")
        table[_format_cu_uuid(uuid)] = (bool(integrated.value), int(total.value))
    return table


def _mem_available_bytes() -> int:
    """Current MemAvailable in bytes; AgentError when unreadable (fail closed).

    Only consulted on the unified-memory discovery path where the budget must
    reflect real host memory; MemAvailable excludes swap by definition.
    """
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                if kb > 0:
                    return kb * 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    raise AgentError(
        "cannot read MemAvailable from /proc/meminfo; cannot budget unified-memory GPUs"
    )


def discover_nvidia_slots(
    agent_id: str,
    *,
    capabilities: Sequence[str],
    model_packs: Sequence[str],
    labels: dict[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[SlotInventory]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = runner(command, text=True, capture_output=True, check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise AgentError(f"could not discover NVIDIA GPUs with nvidia-smi: {exc}") from exc
    # The `memory` label is hardware-owned: only slots whose integrated status
    # was verified by the CUDA probe carry memory=unified, so operator labels
    # can never spoof unified-memory accounting on discrete GPUs.
    base_labels = dict(labels or {})
    base_labels.pop("memory", None)
    slots: list[SlotInventory] = []
    uma_rows: list[tuple[int, str, str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise AgentError(f"unexpected nvidia-smi row: {line!r}")
        index, gpu_uuid, model, vram_mb, compute_capability = parts
        if vram_mb.lower() in _UNAVAILABLE_MEMORY_MARKERS:
            uma_rows.append((int(index), gpu_uuid, model, compute_capability))
            continue
        try:
            vram = int(float(vram_mb))
        except ValueError:
            raise AgentError(
                f"invalid nvidia-smi memory.total {vram_mb!r} in row: {line!r}"
            ) from None
        if vram <= 0:
            raise AgentError(
                f"invalid nvidia-smi memory.total {vram_mb!r} in row: {line!r}"
            )
        slots.append(
            SlotInventory(
                id=f"{agent_id}-gpu-{int(index)}",
                gpu_index=int(index),
                gpu_uuid=gpu_uuid,
                model=model,
                vram_mb=vram,
                compute_capability=compute_capability,
                capabilities=list(capabilities),
                model_packs=list(model_packs),
                labels=dict(base_labels),
            )
        )
    if uma_rows:
        slots.extend(
            _unified_memory_slots(
                agent_id,
                uma_rows,
                base_labels,
                capabilities=capabilities,
                model_packs=model_packs,
            )
        )
    if not slots:
        raise AgentError("nvidia-smi returned no GPUs")
    return slots


def _unified_memory_slots(
    agent_id: str,
    rows: Sequence[tuple[int, str, str, str]],
    base_labels: dict[str, str],
    *,
    capabilities: Sequence[str],
    model_packs: Sequence[str],
) -> list[SlotInventory]:
    """Slots for integrated (unified-memory) GPUs, failing closed.

    nvidia-smi cannot report memory.total for integrated devices, so the CUDA
    driver (device-level queries, no context) verifies the device really is
    integrated and how much of the shared DRAM pool it sees, matched by UUID
    because CUDA ordinals diverge from nvidia-smi indices under
    CUDA_VISIBLE_DEVICES. Slot budget = min(80% of the device's total, 80% of
    MemAvailable) divided by the number of integrated devices, since they all
    draw from one physical pool and must not count it multiple times.
    """
    table = _cuda_device_table()
    probes: list[tuple[int, str, str, str, int]] = []
    for index, gpu_uuid, model, compute_capability in rows:
        probe = table.get(gpu_uuid.lower())
        if probe is None:
            raise AgentError(
                f"CUDA cannot see GPU {gpu_uuid} (hidden by CUDA_VISIBLE_DEVICES "
                "or unrecognized UUID); cannot establish its memory budget"
            )
        integrated, total = probe
        if not integrated:
            raise AgentError(
                f"GPU {gpu_uuid} reports unavailable nvidia-smi memory but is not "
                "an integrated device; refusing to guess its memory"
            )
        if total <= 0:
            raise AgentError(f"CUDA reports no total memory for GPU {gpu_uuid}")
        probes.append((index, gpu_uuid, model, compute_capability, total))
    pool_bytes = min(
        int(0.8 * max(total for *_, total in probes)),
        int(0.8 * _mem_available_bytes()),
    )
    per_slot_bytes = pool_bytes // len(probes)
    slots = []
    for index, gpu_uuid, model, compute_capability, total in probes:
        vram_mb = min(int(0.8 * total), per_slot_bytes) // _MIB
        if vram_mb <= 0:
            raise AgentError(
                f"unified-memory budget for GPU {gpu_uuid} is below 1 MiB; "
                "refusing to advertise an unusable slot"
            )
        slots.append(
            SlotInventory(
                id=f"{agent_id}-gpu-{index}",
                gpu_index=index,
                gpu_uuid=gpu_uuid,
                model=model,
                vram_mb=vram_mb,
                compute_capability=compute_capability,
                capabilities=list(capabilities),
                model_packs=list(model_packs),
                labels={**base_labels, "memory": "unified"},
            )
        )
    return slots


class ControlPlaneClient:
    def __init__(self, endpoint: str, token: str, *, timeout: float = 60.0):
        self.client = httpx.Client(
            base_url=endpoint.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {token}"},
        )

    def close(self) -> None:
        self.client.close()

    def _json(self, response: httpx.Response, operation: str) -> Any:
        if response.status_code < 200 or response.status_code >= 300:
            raise AgentError(f"{operation} failed: HTTP {response.status_code}: {response.text[:1000]}")
        return response.json() if response.content else None

    def register(self, registration: AgentRegistration) -> dict[str, Any]:
        return self._json(
            self.client.post("/v1/agents/register", json=registration.model_dump(mode="json")),
            "agent registration",
        )

    def heartbeat_agent(self, agent_id: str, *, draining: bool | None = None) -> None:
        body = {} if draining is None else {"draining": draining}
        self._json(
            self.client.post(f"/v1/agents/{agent_id}/heartbeat", json=body),
            "agent heartbeat",
        )

    def claim(self, slot_id: str, lease_seconds: int) -> dict[str, Any] | None:
        return self._json(
            self.client.post(
                f"/v1/slots/{slot_id}/claim", json={"lease_seconds": lease_seconds}
            ),
            "claim",
        )

    def confirm(self, job_id: str, slot_id: str, lease_token: str) -> None:
        self._lease_post(job_id, "confirm", slot_id, lease_token)

    def heartbeat_job(
        self, job_id: str, slot_id: str, lease_token: str, lease_seconds: int
    ) -> bool:
        result = self._lease_post(
            job_id,
            "heartbeat",
            slot_id,
            lease_token,
            params={"lease_seconds": lease_seconds},
        )
        return bool(result.get("cancel_requested"))

    def complete(
        self, job_id: str, slot_id: str, lease_token: str, outputs: list[dict[str, Any]]
    ) -> None:
        self._lease_post(
            job_id, "complete", slot_id, lease_token, json_body={"outputs": outputs}
        )

    def upload_output(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        *,
        name: str,
        path: Path,
        media_type: str,
    ) -> dict[str, Any]:
        with path.open("rb") as handle:
            response = self.client.put(
                f"/v1/jobs/{job_id}/outputs/{name}",
                params={"slot_id": slot_id},
                headers={
                    "X-SPAREHOUR-Lease-Token": lease_token,
                    "Content-Type": media_type,
                },
                content=handle,
            )
        return self._json(response, "output upload")

    def fail(
        self,
        job_id: str,
        slot_id: str,
        lease_token: str,
        code: str,
        message: str,
    ) -> None:
        self._lease_post(
            job_id,
            "fail",
            slot_id,
            lease_token,
            json_body={"error_code": code, "error_message": message[:4000]},
        )

    def release(
        self, job_id: str, slot_id: str, lease_token: str, reason: str
    ) -> None:
        self._lease_post(
            job_id, "release", slot_id, lease_token, json_body={"reason": reason[:500]}
        )

    def _lease_post(
        self,
        job_id: str,
        action: str,
        slot_id: str,
        lease_token: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        response = self.client.post(
            f"/v1/jobs/{job_id}/{action}",
            params={"slot_id": slot_id, **(params or {})},
            headers={"X-SPAREHOUR-Lease-Token": lease_token},
            json=json_body,
        )
        return self._json(response, action)

    def resolve_input_uri(self, uri: str) -> str:
        """Return the controller-relative request path for an input blob URI.

        Submitted input URIs are always ``/v1/projects/.../inputs/...`` (or
        the same path absolute on the controller's own origin). Any external,
        ``file://``, or off-origin URI is rejected here rather than fetched:
        honoring it would smuggle the bearer token to a third party or read
        arbitrary worker-local files.
        """
        if not isinstance(uri, str) or not uri:
            raise AgentError("input uri must be a non-empty string")
        path = uri
        if "://" in uri:
            base = str(self.client.base_url).rstrip("/")
            if uri.startswith(base + "/"):
                path = uri[len(base):]
            else:
                raise AgentError(
                    "input uri must stay on the controller origin; "
                    "external or file URIs are refused"
                )
        elif not uri.startswith("/"):
            raise AgentError("input uri must be an absolute controller path")
        if not path.startswith("/v1/projects/"):
            raise AgentError("input uri does not reference a controller input blob")
        return path

    def download_input(
        self,
        uri: str,
        *,
        job_id: str,
        slot_id: str,
        lease_token: str,
        destination: Path,
        max_bytes: int,
        check: Callable[[], None] | None = None,
    ) -> tuple[int, str]:
        """Stream an authorized input blob to ``destination``.

        Requires the job lease (the controller authorizes only blobs the job
        declared). Writes through a sibling temp file so failures never leave
        a partial blob at the final name, and returns ``(size, sha256)`` for
        the caller to compare against the declared metadata. Remote error
        bodies are never echoed into the exception.
        """
        path = self.resolve_input_uri(uri)
        digest = hashlib.sha256()
        size = 0
        part = destination.with_name(destination.name + f".{os.getpid()}.part")
        try:
            with self.client.stream(
                "GET",
                path,
                params={"job_id": job_id, "slot_id": slot_id},
                headers={"X-SPAREHOUR-Lease-Token": lease_token},
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise AgentError(
                        f"input download failed: HTTP {response.status_code}"
                    )
                with part.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        if check is not None:
                            check()
                        size += len(chunk)
                        if size > max_bytes:
                            raise AgentError(
                                f"input exceeds declared bound max_bytes={max_bytes}"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
            os.replace(part, destination)
        except httpx.HTTPError as exc:
            raise AgentError(f"input download failed: {type(exc).__name__}") from exc
        finally:
            part.unlink(missing_ok=True)
        return size, digest.hexdigest()


@dataclass(frozen=True)
class AgentConfig:
    registration: AgentRegistration
    workdir_root: Path
    heartbeat_seconds: int = 15
    lease_seconds: int = 90
    poll_seconds: float = 1.0


DRAINING_EXECUTION_ERRORS = {
    # Fatal worker conditions: fail the job, then drain the agent so no
    # further work is claimed into a possibly corrupted GPU context or an
    # unreleased VRAM allocation.
    "model_server_unavailable",
    "model_load_failed",
    "cuda_context_failure",
    "plugin_release_failed",
}


class AgentRunner:
    def __init__(self, client: ControlPlaneClient, config: AgentConfig):
        if config.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        if config.heartbeat_seconds >= config.lease_seconds:
            raise ValueError("heartbeat_seconds must be shorter than lease_seconds")
        self.client = client
        self.config = config
        self.stop = threading.Event()
        # One live plugin instance per slot. Plugin executors hold expensive
        # state (GPU weights), so a slot reuses the instance while consecutive
        # jobs use the same handler, and releases it when the handler changes,
        # a non-plugin job needs the GPU, or the agent stops.
        self._plugins: dict[str, tuple[str, Any]] = {}

    def request_drain(self) -> None:
        self.stop.set()
        try:
            self.client.heartbeat_agent(self.config.registration.agent_id, draining=True)
        except AgentError:
            pass

    def run(self, *, once: bool = False) -> None:
        self.config.workdir_root.mkdir(parents=True, exist_ok=True)
        self.client.register(self.config.registration)
        # A prior process marks the durable agent row draining on SIGTERM.
        # Successful registration by a fresh runner explicitly restores
        # readiness; ordinary heartbeats intentionally preserve current state.
        self.client.heartbeat_agent(self.config.registration.agent_id, draining=False)
        previous_handlers: dict[int, Any] = {}
        heartbeat_stop = threading.Event()
        thread_errors: list[BaseException] = []

        def drain_handler(_signum, _frame):
            self.request_drain()

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(self.config.heartbeat_seconds):
                try:
                    self.client.heartbeat_agent(self.config.registration.agent_id)
                except AgentError as exc:
                    print(
                        f"agent {self.config.registration.agent_id}: heartbeat failed: {exc}",
                        flush=True,
                    )

        def slot_main(slot: SlotInventory) -> None:
            try:
                self._slot_loop(slot, once)
            except BaseException as exc:
                thread_errors.append(exc)
                self.stop.set()

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
                previous_handlers[sig] = signal.signal(sig, drain_handler)
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            name=f"heartbeat-{self.config.registration.agent_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            threads = [
                threading.Thread(
                    target=slot_main,
                    args=(slot,),
                    name=f"slot-{slot.id}",
                )
                for slot in self.config.registration.slots
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(
                timeout=max(1.0, float(self.config.heartbeat_seconds) + 1.0)
            )
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            # Slot threads have joined; no executor can still be using a
            # plugin instance. Release leftover GPU state without masking a
            # primary slot error.
            gpu_by_slot = {
                slot.id: slot.gpu_index for slot in self.config.registration.slots
            }
            for slot_id, (_handler, instance) in self._plugins.items():
                release = getattr(instance, "release_gpu", None)
                if not callable(release):
                    continue
                try:
                    release(gpu_by_slot.get(slot_id, 0))
                except Exception as exc:
                    # Logged and surfaced only when no slot raised first —
                    # shutdown cleanup must never mask the primary error.
                    print(
                        f"slot {slot_id}: plugin release_gpu failed during "
                        f"shutdown: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    thread_errors.append(exc)
            self._plugins.clear()
        if thread_errors:
            raise thread_errors[0]

    def _slot_loop(self, slot: SlotInventory, once: bool) -> None:
        while not self.stop.is_set():
            try:
                job = self.client.claim(slot.id, self.config.lease_seconds)
                if job is None:
                    if once:
                        return
                    self.stop.wait(self.config.poll_seconds)
                    continue
                self._run_job(slot, job)
                if once:
                    return
            except AgentError as exc:
                if once:
                    raise
                print(f"slot {slot.id}: {exc}", flush=True)
                self.stop.wait(self.config.poll_seconds)

    def _run_job(self, slot: SlotInventory, job: dict[str, Any]) -> None:
        job_id = job["id"]
        token = job["lease_token"]
        self.client.confirm(job_id, slot.id, token)
        workdir = Path(
            tempfile.mkdtemp(prefix=f"sparehour-{job_id}-", dir=self.config.workdir_root)
        )
        lease = _LeaseKeeper(
            self.client,
            job_id=job_id,
            slot_id=slot.id,
            lease_token=token,
            heartbeat_seconds=self.config.heartbeat_seconds,
            lease_seconds=self.config.lease_seconds,
        )
        lease.start()
        try:
            local_outputs = self._execute(slot, job, workdir, lease)
            lease.check()
            outputs = []
            for output in local_outputs:
                lease.check()
                uri = str(output["uri"])
                if not uri.startswith("file://"):
                    raise ExecutionError(
                        "artifact_transport_failed", "executor output is not a local file"
                    )
                outputs.append(
                    self.client.upload_output(
                        job_id,
                        slot.id,
                        token,
                        name=output["name"],
                        path=Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path)),
                        media_type=output["media_type"],
                    )
                )
            # Keep the lease alive throughout potentially large uploads, then
            # fence completion against any cancellation or renewal failure.
            lease.stop()
            lease.check()
            self.client.complete(job_id, slot.id, token, outputs)
        except ExecutionError as exc:
            lease.stop()
            self.client.fail(job_id, slot.id, token, exc.code, str(exc))
            if exc.code in DRAINING_EXECUTION_ERRORS:
                print(
                    f"slot {slot.id}: draining after fatal worker error {exc.code}",
                    flush=True,
                )
                self.request_drain()
        except AgentError as exc:
            lease.stop()
            self.client.fail(
                job_id,
                slot.id,
                token,
                "artifact_transport_failed",
                str(exc),
            )
        except Exception as exc:
            lease.stop()
            self.client.fail(job_id, slot.id, token, "executor_failed", str(exc))
        finally:
            lease.stop()
            shutil.rmtree(workdir, ignore_errors=True)

    def _execute(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: "_LeaseKeeper",
    ) -> list[dict[str, Any]]:
        executor = job["executor"]
        executor_type = executor["type"]
        if executor_type != "plugin":
            # A plugin from a previous job may still hold this slot's VRAM;
            # release it before a command/container/http job needs the GPU.
            self._release_slot_plugin(slot)
            lease.check()
        self._materialize_inputs(job, workdir, lease)
        lease.check()
        if executor_type in {"command", "container"}:
            return self._execute_process(
                slot,
                job,
                workdir,
                lease,
                container=executor_type == "container",
            )
        if executor_type == "http":
            return self._execute_http(slot, job, workdir, lease)
        if executor_type == "plugin":
            return self._execute_plugin(slot, job, workdir, lease)
        raise ExecutionError("executor_unsupported", f"unsupported executor {executor_type!r}")

    def _release_slot_plugin(self, slot: SlotInventory) -> None:
        """Release the slot's cached plugin instance, then drop it.

        Executors may define ``release_gpu(gpu_index)`` to unload weights for
        only this GPU. The instance stays registered until release succeeds:
        a failed release must not leave the slot believing VRAM is free, so
        the failure is raised as ``plugin_release_failed`` (a draining error —
        the job fails and the slot stops claiming) and the instance is retried
        once more at shutdown.
        """
        entry = self._plugins.get(slot.id)
        if entry is None:
            return
        _handler, instance = entry
        release = getattr(instance, "release_gpu", None)
        if callable(release):
            try:
                release(slot.gpu_index)
            except Exception as exc:
                raise ExecutionError(
                    "plugin_release_failed",
                    f"plugin release_gpu failed: {type(exc).__name__}: {exc}",
                ) from exc
        self._plugins.pop(slot.id, None)

    def _materialize_inputs(
        self, job: dict[str, Any], workdir: Path, lease: "_LeaseKeeper"
    ) -> None:
        """Download and verify every declared input before any executor runs.

        Only blobs the controller granted to this job are fetched, always
        lease-checked, and each entry is rewritten in place so executors see
        ``path`` (absolute file under ``workdir/inputs``) and a local
        ``file://`` ``uri``. Declared size/digest are verified after the
        bounded stream so corrupt transport fails the job as
        ``artifact_transport_failed``.
        """
        inputs = job.get("inputs") or []
        if not inputs:
            return
        contract = {item["name"]: item for item in job.get("input_contract") or []}
        inputs_dir = workdir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, int] = {}
        for item in inputs:
            name = str(item.get("name") or "")
            spec = contract.get(name)
            if spec is None:
                raise ExecutionError(
                    "input_validation_failed", f"undeclared input artifact {name!r}"
                )
            if not _INPUT_NAME_RE.fullmatch(name):
                raise ExecutionError(
                    "input_validation_failed", f"unsafe input name {name!r}"
                )
            declared_size = item.get("size")
            if (
                isinstance(declared_size, bool)
                or not isinstance(declared_size, int)
                or declared_size < 0
            ):
                raise ExecutionError(
                    "input_validation_failed", f"input {name!r} has invalid size"
                )
            if declared_size > int(spec["max_bytes"]):
                raise ExecutionError(
                    "input_validation_failed",
                    f"input {name!r} exceeds max_bytes={spec['max_bytes']}",
                )
            media_types = set(spec.get("media_types") or [])
            if media_types and item.get("media_type") not in media_types:
                raise ExecutionError(
                    "input_validation_failed",
                    f"input {name!r} has disallowed media type {item.get('media_type')!r}",
                )
            uri = item.get("uri")
            try:
                self.client.resolve_input_uri(str(uri))
            except AgentError as exc:
                raise ExecutionError(
                    "input_validation_failed",
                    f"input {name!r} uri is not a controller input blob: {exc}",
                ) from exc
            seen[name] = seen.get(name, 0) + 1
            filename = name if seen[name] == 1 else f"{seen[name]}-{name}"
            destination = inputs_dir / filename
            try:
                size, digest = self.client.download_input(
                    str(uri),
                    job_id=lease.job_id,
                    slot_id=lease.slot_id,
                    lease_token=lease.lease_token,
                    destination=destination,
                    max_bytes=min(declared_size, int(spec["max_bytes"])),
                    check=lease.check,
                )
            except AgentError as exc:
                if isinstance(exc, ExecutionError):
                    raise
                raise ExecutionError(
                    "artifact_transport_failed",
                    f"input {name!r} download failed: {exc}",
                ) from exc
            if size != declared_size:
                raise ExecutionError(
                    "artifact_transport_failed",
                    f"input {name!r} size {size} does not match declared {declared_size}",
                )
            if digest != str(item.get("sha256") or ""):
                raise ExecutionError(
                    "artifact_transport_failed",
                    f"input {name!r} sha256 does not match the declared digest",
                )
            item["path"] = str(destination)
            item["uri"] = destination.resolve().as_uri()

    def _execute_process(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: "_LeaseKeeper",
        *,
        container: bool,
    ) -> list[dict[str, Any]]:
        executor = job["executor"]
        request_path = workdir / "request.json"
        request_path.write_text(
            json.dumps(_process_request(job, workdir, container=container), sort_keys=True),
            encoding="utf-8",
        )
        output_manifest = workdir / "outputs.json"
        environment = {
            **os.environ,
            **executor.get("environment", {}),
            "CUDA_VISIBLE_DEVICES": str(slot.gpu_index),
            "SPAREHOUR_JOB_ID": job["id"],
            "SPAREHOUR_REQUEST_JSON": str(request_path),
            "SPAREHOUR_OUTPUT_DIR": str(workdir),
            "SPAREHOUR_OUTPUT_MANIFEST": str(output_manifest),
        }
        command = [
            sys.executable if argument == "{agent_python}" else argument
            for argument in executor["command"]
        ]
        if container:
            container_environment = []
            for name in sorted(executor.get("environment", {})):
                container_environment.extend(["-e", name])
            command = [
                "docker",
                "run",
                "--rm",
                "--name",
                f"sparehour-{job['id'][:48]}",
                "--gpus",
                f"device={slot.gpu_uuid}",
                "--mount",
                f"type=bind,src={workdir},dst=/job",
                "-e",
                "SPAREHOUR_JOB_ID",
                "-e",
                "SPAREHOUR_REQUEST_JSON=/job/request.json",
                "-e",
                "SPAREHOUR_OUTPUT_DIR=/job",
                "-e",
                "SPAREHOUR_OUTPUT_MANIFEST=/job/outputs.json",
                *container_environment,
                executor["image"],
                *executor["command"],
            ]
        stdout_path = workdir / "executor.stdout.log"
        stderr_path = workdir / "executor.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=workdir,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            timeout = float(executor.get("timeout_seconds") or 3600)
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    lease.check()
                    wait = min(1.0, max(0.0, deadline - time.monotonic()))
                    if wait <= 0:
                        raise ExecutionError(
                            "execution_timeout", f"executor exceeded {timeout:g}s"
                        )
                    try:
                        process.wait(timeout=wait)
                    except subprocess.TimeoutExpired:
                        continue
            except BaseException:
                _terminate_process(process)
                raise
        if process.returncode != 0:
            stdout = _tail_text(stdout_path)
            stderr = _tail_text(stderr_path)
            message = (stderr or stdout or f"exit code {process.returncode}")[-3000:]
            code = "gpu_oom" if "out of memory" in message.casefold() else "executor_failed"
            raise ExecutionError(code, message)
        return _load_output_manifest(output_manifest, workdir)

    def _execute_http(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: "_LeaseKeeper",
    ) -> list[dict[str, Any]]:
        executor = job["executor"]
        contract = job["output_contract"]
        if len(contract) != 1:
            raise ExecutionError(
                "executor_contract_invalid", "generic HTTP executor requires exactly one output"
            )
        spec = contract[0]
        path = workdir / "response.bin"
        size = 0
        try:
            with httpx.stream(
                "POST",
                executor["endpoint"],
                json=job["payload"],
                timeout=float(executor.get("timeout_seconds") or 3600),
            ) as response:
                response.raise_for_status()
                media_type = response.headers.get(
                    "content-type", "application/json"
                ).split(";", 1)[0]
                with path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        lease.check()
                        size += len(chunk)
                        if size > int(spec["max_bytes"]):
                            raise ExecutionError(
                                "output_validation_failed",
                                f"HTTP response exceeds max_bytes={spec['max_bytes']}",
                            )
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            raise ExecutionError("model_server_unavailable", str(exc)) from exc
        return [_artifact_from_path(spec["name"], path, media_type)]

    def _execute_plugin(
        self,
        slot: SlotInventory,
        job: dict[str, Any],
        workdir: Path,
        lease: "_LeaseKeeper",
    ) -> list[dict[str, Any]]:
        handler_spec = job["executor"]["handler"]
        module_name, separator, attribute = handler_spec.partition(":")
        if not separator:
            raise ExecutionError("executor_contract_invalid", "plugin handler must be module:attribute")
        # The previous plugin keeps the slot's VRAM until now; releasing it
        # after the job's inputs are staged keeps download time overlapped
        # with weights already loaded for a warm same-handler run.
        entry = self._plugins.get(slot.id)
        if entry is not None and entry[0] != handler_spec:
            self._release_slot_plugin(slot)
            lease.check()
            entry = None
        if entry is not None:
            instance = entry[1]
        else:
            try:
                instance = getattr(importlib.import_module(module_name), attribute)()
            except (ImportError, AttributeError) as exc:
                raise ExecutionError("executor_unavailable", str(exc)) from exc
            self._plugins[slot.id] = (handler_spec, instance)
        try:
            result = instance.execute(job=job, workdir=workdir, gpu_index=slot.gpu_index)
        except (ImportError, AttributeError) as exc:
            raise ExecutionError("executor_unavailable", str(exc)) from exc
        lease.check()
        return [
            _artifact_from_path(item["name"], Path(item["path"]), item["media_type"])
            for item in result
        ]


class _LeaseKeeper:
    """Renew a job lease while any trusted executor or upload is active."""

    def __init__(
        self,
        client: ControlPlaneClient,
        *,
        job_id: str,
        slot_id: str,
        lease_token: str,
        heartbeat_seconds: int,
        lease_seconds: int,
    ):
        self.client = client
        self.job_id = job_id
        self.slot_id = slot_id
        self.lease_token = lease_token
        self.heartbeat_seconds = heartbeat_seconds
        self.lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._cancelled = threading.Event()
        self._failure: AgentError | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"lease-{job_id}",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._started and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, float(self.heartbeat_seconds) + 1.0))

    def check(self) -> None:
        if self._cancelled.is_set():
            raise ExecutionError("cancelled", "job cancellation was requested")
        if self._failure is not None:
            raise ExecutionError("worker_lost", f"job lease renewal failed: {self._failure}")

    def _run(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                cancelled = self.client.heartbeat_job(
                    self.job_id,
                    self.slot_id,
                    self.lease_token,
                    self.lease_seconds,
                )
            except AgentError as exc:
                self._failure = exc
                self._stop.set()
                return
            if cancelled:
                self._cancelled.set()
                self._stop.set()
                return


def _load_output_manifest(path: Path, workdir: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ExecutionError("output_validation_failed", "executor did not create outputs.json")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        files = document["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ExecutionError("output_validation_failed", f"invalid outputs.json: {exc}") from exc
    outputs = []
    for item in files:
        candidate = Path(item["path"])
        candidate = candidate if candidate.is_absolute() else workdir / candidate
        try:
            candidate.resolve().relative_to(workdir.resolve())
        except ValueError as exc:
            raise ExecutionError("output_validation_failed", "output path escapes workdir") from exc
        outputs.append(_artifact_from_path(item["name"], candidate, item["media_type"]))
    return outputs


def _process_request(
    job: dict[str, Any], workdir: Path, *, container: bool
) -> dict[str, Any]:
    """Job document written to request.json for command/container executors.

    Materialized ``inputs[].path`` values are host-absolute paths under
    ``workdir``; a container only sees ``workdir`` bind-mounted at ``/job``,
    so for containers every input path and file:// URI is rewritten to the
    in-container location. Paths that somehow escaped the workdir are
    rejected instead of silently mapped. The job document is deep-copied, not
    mutated.
    """
    document = copy.deepcopy(job)
    if not container:
        return document
    root = workdir.resolve()
    for index, item in enumerate(document.get("inputs") or []):
        raw = item.get("path")
        if not raw:
            continue
        candidate = Path(str(raw)).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ExecutionError(
                "input_validation_failed",
                f"inputs[{index}] path is outside the job workdir",
            ) from exc
        container_path = Path("/job") / relative
        item["path"] = str(container_path)
        item["uri"] = container_path.as_uri()
    return document


def _artifact_from_path(name: str, path: Path, media_type: str) -> dict[str, Any]:
    if not path.is_file():
        raise ExecutionError("output_validation_failed", f"output file is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {
        "name": name,
        "uri": path.resolve().as_uri(),
        "media_type": media_type,
        "size": size,
        "sha256": digest.hexdigest(),
    }


def _tail_text(path: Path, max_bytes: int = 16 * 1024) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _terminate_process(process: subprocess.Popen[str], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        else:
            process.wait()
