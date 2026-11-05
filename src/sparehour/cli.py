"""Command-line entrypoint for the control plane and GPU host agents."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .agent import AgentConfig, AgentRunner, ControlPlaneClient, discover_nvidia_slots
from .api import create_app
from .artifacts import ArtifactStore
from .controller import FleetPlanner, PoolSpec
from .deploy import DeploymentService, build_drivers_from_env
from .models import AgentRegistration, SlotInventory
from .providers.salad import SaladProvider
from .providers.vast import VastProvider
from .registry import WorkloadRegistry
from .service import Orchestrator
from .store import Store


UTC = timezone.utc


DEFAULT_DB = os.environ.get("SPAREHOUR_DB", "./state/control.sqlite3")
DEFAULT_WORKLOADS = os.environ.get("SPAREHOUR_WORKLOADS", "./workloads")
DEFAULT_ARTIFACTS = os.environ.get("SPAREHOUR_ARTIFACTS", "./state/artifacts")
DEFAULT_POOLS = os.environ.get("SPAREHOUR_POOLS", "./pools")


def _orchestrator(args: argparse.Namespace) -> Orchestrator:
    return Orchestrator(
        WorkloadRegistry.from_directory(args.workloads),
        Store(args.db),
    )


def _json_argument(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _token(args: argparse.Namespace) -> str:
    if getattr(args, "token", None):
        return str(args.token).strip()
    if getattr(args, "token_file", None):
        return Path(args.token_file).read_text(encoding="utf-8").strip()
    return os.environ.get("SPAREHOUR_TOKEN", "").strip()


def cmd_validate(args: argparse.Namespace) -> int:
    registry = WorkloadRegistry.from_directory(args.workloads)
    for name in registry.names():
        workload = registry.get(name)
        print(f"{name}@{workload.version}: {', '.join(sorted(workload.routes))}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    orchestrator = _orchestrator(args)
    # Provider deploys are only offered when workers have a URL they can
    # actually reach back on; without it the endpoints stay unmounted.
    public_url = os.environ.get("SPAREHOUR_PUBLIC_URL", "").strip()
    deployment_service = None
    if public_url:
        deployment_service = DeploymentService(
            registry=orchestrator.registry,
            store=orchestrator.store,
            drivers=build_drivers_from_env(),
            pools_directory=args.pools,
            public_controller_url=public_url,
            wheel_url=os.environ.get("SPAREHOUR_WHEEL_URL", "").strip() or None,
        )
    app = create_app(
        orchestrator,
        ArtifactStore(args.artifacts),
        bootstrap_directory=args.bootstrap_dir,
        deployment_service=deployment_service,
        enable_dashboard=args.dashboard,
        deployment_reconcile_seconds=(
            float(os.environ.get("SPAREHOUR_DEPLOYMENT_RECONCILE_SECONDS", "15"))
            if deployment_service is not None
            else None
        ),
        scaling_evaluate_seconds=(
            float(os.environ.get("SPAREHOUR_SCALING_EVALUATE_SECONDS", "60"))
            if deployment_service is not None else None
        ),
        autonomous_apply=os.environ.get("SPAREHOUR_AUTONOMOUS_APPLY", "").lower() in {"1", "true", "yes"},
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_issue_token(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        scopes: dict[str, list[str]] = {}
        for raw in args.scope:
            key, separator, value = raw.partition("=")
            if not separator or not key or not value:
                raise ValueError("--scope must be KEY=VALUE")
            scopes.setdefault(key, []).append(value)
        expires_at = (
            datetime.now(UTC) + timedelta(hours=args.expires_hours)
            if args.expires_hours
            else None
        )
        print(
            store.issue_token(
                role=args.role,
                subject=args.subject,
                scopes=scopes,
                expires_at=expires_at,
                token_id=args.token_id,
            )
        )
    finally:
        store.close()
    return 0


def cmd_revoke_token(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        if not store.disable_token(args.token_id):
            raise ValueError(f"token {args.token_id} not found or already disabled")
        print(f"revoked token {args.token_id}")
    finally:
        store.close()
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    try:
        job = orchestrator.submit(
            project_id=args.project,
            workload=args.workload,
            route=args.route,
            payload=_json_argument(args.payload),
            priority=args.priority,
            idempotency_key=args.idempotency_key,
        )
        print(json.dumps(job, indent=2, sort_keys=True))
    finally:
        orchestrator.store.close()
    return 0


def cmd_get_job(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        print(json.dumps(store.get_job(args.job_id), indent=2, sort_keys=True))
    finally:
        store.close()
    return 0


def cmd_list_jobs(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        print(
            json.dumps(
                store.list_jobs(
                    project_id=args.project,
                    workload=args.workload,
                    route=args.route,
                    state=args.state,
                    limit=args.limit,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        store.close()
    return 0


def cmd_backlog(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        print(json.dumps(store.backlog(), indent=2, sort_keys=True))
    finally:
        store.close()
    return 0


def cmd_reap(args: argparse.Namespace) -> int:
    store = Store(args.db)
    try:
        print(json.dumps(store.reap_expired_leases(), sort_keys=True))
    finally:
        store.close()
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    orchestrator = _orchestrator(args)
    try:
        pool = PoolSpec.from_yaml(args.pool)
        drivers = {}
        vast_key = _secret_value(args.vast_api_key, args.vast_api_key_file, "VAST_API_KEY")
        if vast_key:
            drivers["vast"] = VastProvider(vast_key, base_url=args.vast_api_base)
        salad_key = _secret_value(args.salad_api_key, args.salad_api_key_file, "SALAD_API_KEY")
        if salad_key:
            if not args.salad_organization or not args.salad_project:
                raise ValueError("Salad planning requires --salad-organization and --salad-project")
            drivers["salad"] = SaladProvider(
                salad_key,
                args.salad_organization,
                args.salad_project,
                base_url=args.salad_api_base,
            )
        plan = FleetPlanner(orchestrator.registry, orchestrator.store).plan(pool, drivers)
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    finally:
        orchestrator.store.close()
    return 0


def cmd_agent_inventory(args: argparse.Namespace) -> int:
    slots = _agent_slots(args)
    print(json.dumps([slot.model_dump(mode="json") for slot in slots], indent=2))
    return 0


def cmd_agent_run(args: argparse.Namespace) -> int:
    token = _token(args)
    if not token:
        raise ValueError("agent token is required via --token, --token-file, or SPAREHOUR_TOKEN")
    slots = _agent_slots(args)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(args.workdir)
    registration = AgentRegistration(
        agent_id=args.agent_id,
        provider=args.provider,
        provider_resource_id=args.provider_resource_id,
        hostname=socket.gethostname(),
        cpu_cores=float(os.cpu_count() or 0),
        ram_mb=_system_ram_mb(),
        disk_free_mb=disk.free // (1024 * 1024),
        labels=_arch_labels(_key_values(args.label)),
        slots=slots,
    )
    client = ControlPlaneClient(args.endpoint, token)
    try:
        AgentRunner(
            client,
            AgentConfig(
                registration=registration,
                workdir_root=Path(args.workdir),
                heartbeat_seconds=args.heartbeat_seconds,
                lease_seconds=args.lease_seconds,
                poll_seconds=args.poll_seconds,
            ),
        ).run(once=args.once)
    finally:
        client.close()
    return 0


def _agent_slots(args: argparse.Namespace) -> list[SlotInventory]:
    capabilities = args.capability or []
    if not capabilities:
        raise ValueError("at least one --capability is required")
    if args.fake_gpus:
        return [
            SlotInventory(
                id=f"{args.agent_id}-gpu-{index}",
                gpu_index=index,
                gpu_uuid=f"FAKE-GPU-{args.agent_id}-{index}",
                model=args.fake_gpu_model,
                vram_mb=args.fake_gpu_vram_mb,
                compute_capability=args.fake_compute_capability,
                capabilities=capabilities,
                model_packs=args.model_pack,
                labels=_arch_labels(_key_values(args.slot_label)),
            )
            for index in range(args.fake_gpus)
        ]
    return discover_nvidia_slots(
        args.agent_id,
        capabilities=capabilities,
        model_packs=args.model_pack,
        labels=_arch_labels(_key_values(args.slot_label)),
    )


def _key_values(values: list[str]) -> dict[str, str]:
    result = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key:
            raise ValueError(f"expected KEY=VALUE, got {raw!r}")
        result[key] = value
    return result


_ARCH_ALIASES = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "amd64": "amd64"}


def _arch_labels(labels: dict[str, str]) -> dict[str, str]:
    """Advertise the hardware arch as a label; it wins over operator values.

    The operator ``memory`` tag is stripped: UMA is a hardware fact owned by
    slot discovery, not a spoofable operator label.
    """
    machine = platform.machine().strip().casefold()
    arch = _ARCH_ALIASES.get(machine, machine)
    return {
        **{key: value for key, value in labels.items() if key != "memory"},
        "arch": arch,
    }


def _secret_value(explicit: str | None, file_path: str | None, env_name: str) -> str:
    if explicit:
        return explicit.strip()
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return os.environ.get(env_name, "").strip()


def _system_ram_mb() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size // (1024 * 1024))
    except (OSError, ValueError):
        return 0


def add_shared(parser: argparse.ArgumentParser, *, workloads: bool = False) -> None:
    parser.add_argument("--db", default=DEFAULT_DB)
    if workloads:
        parser.add_argument("--workloads", default=DEFAULT_WORKLOADS)


def add_agent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--model-pack", action="append", default=[])
    parser.add_argument("--slot-label", action="append", default=[])
    parser.add_argument("--fake-gpus", type=int, default=0)
    parser.add_argument("--fake-gpu-model", default="NVIDIA RTX 4090")
    parser.add_argument("--fake-gpu-vram-mb", type=int, default=24576)
    parser.add_argument("--fake-compute-capability", default="8.9")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparehour", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate and list workload definitions")
    validate.add_argument("--workloads", default=DEFAULT_WORKLOADS)
    validate.set_defaults(func=cmd_validate)

    serve = sub.add_parser("serve", help="run the control-plane API")
    add_shared(serve, workloads=True)
    serve.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    serve.add_argument("--pools", default=DEFAULT_POOLS)
    serve.add_argument(
        "--bootstrap-dir",
        help="optional directory of public, non-secret worker wheel files",
    )
    serve.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SPAREHOUR_DASHBOARD", "1").lower() not in {"0", "false", "no"},
        help="serve the operator workbench UI at /dashboard (default: on)",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8088)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    token = sub.add_parser("token", help="issue a scoped API credential")
    add_shared(token)
    token.add_argument("--role", choices=["admin", "client", "agent", "controller"], required=True)
    token.add_argument("--subject", required=True)
    token.add_argument("--scope", action="append", default=[])
    token.add_argument("--expires-hours", type=float)
    token.add_argument("--token-id")
    token.set_defaults(func=cmd_issue_token)

    revoke_token = sub.add_parser("revoke-token", help="disable a scoped API credential")
    add_shared(revoke_token)
    revoke_token.add_argument("--token-id", required=True)
    revoke_token.set_defaults(func=cmd_revoke_token)

    submit = sub.add_parser("submit", help="admit a job directly into the local control DB")
    add_shared(submit, workloads=True)
    submit.add_argument("--project", required=True)
    submit.add_argument("--workload", required=True)
    submit.add_argument("--route")
    submit.add_argument("--payload", required=True, help="JSON text or @path/to/file.json")
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--idempotency-key")
    submit.set_defaults(func=cmd_submit)

    get_job = sub.add_parser("get", help="get one job from the local control DB")
    add_shared(get_job)
    get_job.add_argument("job_id")
    get_job.set_defaults(func=cmd_get_job)

    jobs = sub.add_parser("jobs", help="list jobs in the local control DB")
    add_shared(jobs)
    jobs.add_argument("--project")
    jobs.add_argument("--workload")
    jobs.add_argument("--route")
    jobs.add_argument("--state")
    jobs.add_argument("--limit", type=int, default=100)
    jobs.set_defaults(func=cmd_list_jobs)

    backlog = sub.add_parser("backlog", help="show queued workload units")
    add_shared(backlog)
    backlog.set_defaults(func=cmd_backlog)

    reap = sub.add_parser("reap", help="recover jobs with expired worker leases")
    add_shared(reap)
    reap.set_defaults(func=cmd_reap)

    plan = sub.add_parser("plan", help="make a read-only route capacity plan")
    add_shared(plan, workloads=True)
    plan.add_argument("--pool", required=True)
    plan.add_argument("--vast-api-key")
    plan.add_argument("--vast-api-key-file")
    plan.add_argument("--vast-api-base", default="https://console.vast.ai/api")
    plan.add_argument("--salad-api-key")
    plan.add_argument("--salad-api-key-file")
    plan.add_argument("--salad-api-base", default="https://api.salad.com/api/public")
    plan.add_argument("--salad-organization")
    plan.add_argument("--salad-project")
    plan.set_defaults(func=cmd_plan)

    inventory = sub.add_parser("agent-inventory", help="show GPU slots an agent would register")
    add_agent_args(inventory)
    inventory.set_defaults(func=cmd_agent_inventory)

    agent = sub.add_parser("agent", help="register this host and execute jobs")
    add_agent_args(agent)
    agent.add_argument("--endpoint", default=os.environ.get("SPAREHOUR_ENDPOINT", "http://127.0.0.1:8088"))
    agent.add_argument("--token")
    agent.add_argument("--token-file")
    agent.add_argument("--provider", default="local")
    agent.add_argument("--provider-resource-id")
    agent.add_argument("--label", action="append", default=[])
    agent.add_argument("--workdir", default="/tmp/sparehour")
    agent.add_argument("--heartbeat-seconds", type=int, default=15)
    agent.add_argument("--lease-seconds", type=int, default=90)
    agent.add_argument("--poll-seconds", type=float, default=1)
    agent.add_argument("--once", action="store_true")
    agent.set_defaults(func=cmd_agent_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
