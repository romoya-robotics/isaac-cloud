from __future__ import annotations

import json
import os
import socket
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import typer
from click.core import ParameterSource

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


APP_NAME = "isaac-cloud"
API_BASE_URL = "https://dashboard.tensordock.com/api/v2"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
DEFAULT_STATE_PATH = Path.home() / ".config" / APP_NAME / "state.json"
DEFAULT_IMAGE = "ubuntu2404_nvidia_570"
DEFAULT_VIEWER_PORT = 8210
DEFAULT_SSH_USER = "user"
DEFAULT_INSTANCE_NAME_PREFIX = "isaac-cloud"
DEFAULT_ISAAC_VERSION = "5.1.0"
DEFAULT_ISAAC_SIGNAL_PORT = 49100
DEFAULT_ISAAC_STREAM_PORT = 47998
DEFAULT_REMOTE_ROOT = "/opt/gpu-orchestrator"
DEFAULT_RUNTIME_DIR = f"{DEFAULT_REMOTE_ROOT}/runtime/docker/isaac-sim"
DEFAULT_VIEWER_APP_DIR = f"{DEFAULT_REMOTE_ROOT}/web-viewer"
DEFAULT_AUTO_VCPU = 4
DEFAULT_AUTO_RAM_GB = 16
MIN_STORAGE_GB = 100

GPU_COMPATIBILITY: dict[str, tuple[str, ...]] = {
    "rtx4080": (
        "geforcertx4080-pcie-16gb",
        "geforcertx4080super-pcie-16gb",
        "geforcertx4090-pcie-24gb",
        "geforcertx5090-pcie-32gb",
        "rtx4000ada-sff-20gb",
        "rtx4500ada-24gb",
        "rtx5000ada-32gb",
        "rtx5880ada-48gb",
        "rtx6000ada-48gb",
        "l40-48gb",
        "l40s-48gb",
    ),
    "rtx4090": (
        "geforcertx4090-pcie-24gb",
        "geforcertx5090-pcie-32gb",
        "rtx5000ada-32gb",
        "rtx5880ada-48gb",
        "rtx6000ada-48gb",
        "l40-48gb",
        "l40s-48gb",
    ),
    "l40s": ("l40s-48gb",),
    "l40": ("l40-48gb", "l40s-48gb"),
    "rtxa4000": ("rtxa4000-pcie-16gb",),
}

RUNNING_STATES = {"running"}
STOPPED_STATES = {"stopped", "stoppeddisassociated"}


app = typer.Typer(
    help="Provision and manage TensorDock GPU instances for NVIDIA Isaac Sim.",
    add_completion=False,
    no_args_is_help=True,
)
sync_app = typer.Typer(
    help="Sync durable Isaac data to or from S3.",
    add_completion=False,
    no_args_is_help=True,
)


class IsaacCloudError(Exception):
    pass


@dataclass
class AppConfig:
    api_token: str
    ssh_key: str
    ngc_api_key: str | None
    ssh_private_key_path: str | None
    ssh_user: str
    default_gpu_class: str
    default_region: str | None
    default_vcpu: int
    default_ram_gb: int
    default_storage_gb: int
    instance_name_prefix: str
    viewer_port: int
    isaac_version: str


@dataclass
class Candidate:
    location_id: str
    city: str
    stateprovince: str
    country: str
    tier: int | None
    gpu_v0_name: str
    gpu_display_name: str
    gpu_count: int
    gpu_price_per_hr: float
    max_vcpus: int
    max_ram_gb: int
    max_storage_gb: int
    per_vcpu_hr: float
    per_gb_ram_hr: float
    per_gb_storage_hr: float
    dedicated_ip_available: bool
    port_forwarding_available: bool
    estimated_hourly_cost: float


@dataclass
class InstanceNetwork:
    public_ip: str | None = None
    ssh_port: int | None = None
    ssh_host: str | None = None
    viewer_port: int = DEFAULT_VIEWER_PORT
    port_forwards: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InstanceSummary:
    id: str
    name: str
    status: str
    public_ip: str | None
    ssh_port: int | None
    network: InstanceNetwork
    raw: dict[str, Any]


def _raise(message: str) -> None:
    raise IsaacCloudError(message)


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        _raise(
            f"Missing config file: {path}. Create config.toml before running the script."
        )
    if tomllib is None:
        _raise("Python tomllib is unavailable; use Python 3.11+.")
    return tomllib.loads(path.read_text())


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def env_or_config(config_data: dict[str, Any], env_name: str, *path: str) -> Any:
    value = os.getenv(env_name)
    if value not in (None, ""):
        return value
    return nested_get(config_data, *path)


def int_or_default(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def load_app_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    config_data = load_toml(config_path)

    api_token = env_or_config(config_data, "TENSORDOCK_API_TOKEN", "tensordock", "api_token")
    ssh_key = env_or_config(config_data, "TENSORDOCK_SSH_KEY", "tensordock", "ssh_key")

    if not api_token:
        _raise(
            "Missing TensorDock API token. Set TENSORDOCK_API_TOKEN or configure "
            f"{config_path}."
        )
    if not ssh_key:
        _raise(
            "Missing TensorDock SSH key reference. Set TENSORDOCK_SSH_KEY or configure "
            f"{config_path}."
        )

    return AppConfig(
        api_token=api_token,
        ssh_key=ssh_key,
        ngc_api_key=env_or_config(config_data, "NGC_API_KEY", "ngc", "api_key"),
        ssh_private_key_path=env_or_config(
            config_data, "ISAAC_CLOUD_SSH_PRIVATE_KEY", "ssh", "private_key_path"
        ),
        ssh_user=env_or_config(config_data, "ISAAC_CLOUD_SSH_USER", "ssh", "user")
        or DEFAULT_SSH_USER,
        default_gpu_class=env_or_config(config_data, "ISAAC_CLOUD_GPU_CLASS", "defaults", "gpu_class")
        or "rtx4080",
        default_region=env_or_config(config_data, "ISAAC_CLOUD_REGION", "defaults", "region"),
        default_vcpu=int_or_default(
            env_or_config(config_data, "ISAAC_CLOUD_VCPU", "defaults", "vcpu"), 0
        ),
        default_ram_gb=int_or_default(
            env_or_config(config_data, "ISAAC_CLOUD_RAM_GB", "defaults", "ram_gb"), 0
        ),
        default_storage_gb=int_or_default(
            env_or_config(config_data, "ISAAC_CLOUD_STORAGE_GB", "defaults", "storage_gb"), 0
        ),
        instance_name_prefix=env_or_config(
            config_data, "ISAAC_CLOUD_INSTANCE_NAME_PREFIX", "defaults", "instance_name_prefix"
        )
        or DEFAULT_INSTANCE_NAME_PREFIX,
        viewer_port=int_or_default(
            env_or_config(config_data, "ISAAC_CLOUD_VIEWER_PORT", "defaults", "viewer_port"),
            DEFAULT_VIEWER_PORT,
        ),
        isaac_version=env_or_config(
            config_data, "ISAAC_CLOUD_ISAAC_VERSION", "defaults", "isaac_version"
        )
        or DEFAULT_ISAAC_VERSION,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_state(state_path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def save_state(state: dict[str, Any], state_path: Path = DEFAULT_STATE_PATH) -> None:
    ensure_parent(state_path)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def clear_state(state_path: Path = DEFAULT_STATE_PATH) -> None:
    if state_path.exists():
        state_path.unlink()


def normalize_status(status: str | None) -> str:
    return (status or "unknown").strip().lower()


def parse_location_label(candidate: Candidate) -> str:
    parts = [candidate.city, candidate.stateprovince, candidate.country]
    return ", ".join(part for part in parts if part)


def format_bool_flag(value: bool) -> str:
    return "yes" if value else "no"


def extract_dict(payload: dict[str, Any], *paths: tuple[str, ...]) -> dict[str, Any]:
    for path in paths:
        value = nested_get(payload, *path)
        if isinstance(value, dict):
            return value
    return {}


def extract_list(payload: dict[str, Any], *paths: tuple[str, ...]) -> list[Any]:
    for path in paths:
        value = nested_get(payload, *path)
        if isinstance(value, list):
            return value
    return []


def first_truthy(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def extract_instance_id(payload: dict[str, Any]) -> str | None:
    value = first_truthy(
        payload.get("id"),
        payload.get("instance_id"),
        payload.get("instanceId"),
        payload.get("server_id"),
        payload.get("serverId"),
        payload.get("vm_id"),
        payload.get("vmId"),
        nested_get(payload, "attributes", "id"),
        nested_get(payload, "attributes", "instance_id"),
        nested_get(payload, "attributes", "instanceId"),
        nested_get(payload, "attributes", "server_id"),
        nested_get(payload, "attributes", "serverId"),
        nested_get(payload, "attributes", "vm_id"),
        nested_get(payload, "attributes", "vmId"),
        nested_get(payload, "data", "id"),
        nested_get(payload, "data", "attributes", "id"),
        nested_get(payload, "data", "attributes", "instance_id"),
        nested_get(payload, "data", "attributes", "instanceId"),
        nested_get(payload, "data", "attributes", "server_id"),
        nested_get(payload, "data", "attributes", "serverId"),
        nested_get(payload, "data", "attributes", "vm_id"),
        nested_get(payload, "data", "attributes", "vmId"),
        nested_get(payload, "data", "instance", "id"),
        nested_get(payload, "data", "server", "id"),
        nested_get(payload, "data", "virtualmachine", "id"),
    )
    if value is None:
        return None
    return str(value)


def describe_payload_shape(payload: dict[str, Any]) -> str:
    keys = sorted(str(key) for key in payload.keys())
    attributes = payload.get("attributes")
    data = payload.get("data")
    parts = [f"top-level keys={keys}"]
    if isinstance(attributes, dict):
        parts.append(f"attributes keys={sorted(str(key) for key in attributes.keys())}")
    if isinstance(data, dict):
        parts.append(f"data keys={sorted(str(key) for key in data.keys())}")
        nested_attributes = data.get("attributes")
        if isinstance(nested_attributes, dict):
            parts.append(
                f"data.attributes keys={sorted(str(key) for key in nested_attributes.keys())}"
            )
    return "; ".join(parts)


def is_retryable_create_error(message: str) -> bool:
    normalized = message.lower()
    retryable_fragments = (
        "ram per gpu exceeds ratio limit",
        "virtual machine deployment failed",
        "out of stock",
        "insufficient capacity",
        "not enough capacity",
        "resource_error",
    )
    return any(fragment in normalized for fragment in retryable_fragments)


class TensorDockClient:
    def __init__(self, api_token: str, base_url: str = API_BASE_URL) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(90.0, connect=10.0),
        )

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        response.raise_for_status()
        if not response.content:
            return {}
        payload = response.json()
        if isinstance(payload, dict):
            status = payload.get("status")
            error = first_truthy(
                payload.get("error"),
                payload.get("message"),
                nested_get(payload, "errors", 0, "detail"),
            )
            if isinstance(status, int) and status >= 400 and error:
                _raise(f"TensorDock API rejected {method} {path}: {error}")
        return payload

    def list_locations(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/locations")
        return extract_list(payload, ("data", "locations"))

    def list_instances(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/instances")
        return extract_list(payload, ("data", "instances"), ("data", "attributes", "instances"))

    def get_instance(self, instance_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/instances/{instance_id}")
        if "data" in payload and isinstance(payload["data"], dict):
            return payload["data"]
        return payload

    def create_instance(self, attributes: dict[str, Any]) -> dict[str, Any]:
        payload = self._request(
            "POST",
            "/instances",
            json={"data": {"type": "virtualmachine", "attributes": attributes}},
        )
        return extract_dict(payload, ("data",)) or payload

    def start_instance(self, instance_id: str) -> dict[str, Any]:
        payload = self._request("POST", f"/instances/{instance_id}/start")
        return extract_dict(payload, ("data",)) or payload

    def stop_instance(self, instance_id: str) -> dict[str, Any]:
        payload = self._request("POST", f"/instances/{instance_id}/stop")
        return extract_dict(payload, ("data",)) or payload

    def delete_instance(self, instance_id: str) -> dict[str, Any]:
        payload = self._request("DELETE", f"/instances/{instance_id}")
        return extract_dict(payload, ("data",)) or payload


def parse_instance_network(instance: dict[str, Any], viewer_port: int) -> InstanceNetwork:
    port_forwards = first_truthy(
        instance.get("portForwards"),
        nested_get(instance, "attributes", "portForwards"),
        nested_get(instance, "networking", "portForwards"),
        [],
    )
    public_ip = first_truthy(
        instance.get("ipAddress"),
        instance.get("public_ip"),
        instance.get("ip_address"),
        nested_get(instance, "networking", "public_ip"),
        nested_get(instance, "attributes", "ipAddress"),
        nested_get(instance, "attributes", "public_ip"),
    )
    ssh_port = None
    if isinstance(port_forwards, list):
        for entry in port_forwards:
            internal_port = entry.get("internal_port") or entry.get("internalPort")
            if internal_port == 22:
                ssh_port = entry.get("external_port") or entry.get("externalPort")
                break
    if public_ip and ssh_port is None:
        ssh_port = 22
    ssh_host = public_ip
    return InstanceNetwork(
        public_ip=public_ip,
        ssh_port=int(ssh_port) if ssh_port not in (None, "") else None,
        ssh_host=ssh_host,
        viewer_port=viewer_port,
        port_forwards=port_forwards if isinstance(port_forwards, list) else [],
    )


def parse_instance_summary(instance: dict[str, Any], viewer_port: int) -> InstanceSummary:
    attributes = instance.get("attributes", {}) if isinstance(instance.get("attributes"), dict) else {}
    status = first_truthy(instance.get("status"), attributes.get("status"), "unknown")
    name = first_truthy(instance.get("name"), attributes.get("name"), "unknown")
    instance_id = extract_instance_id(instance)
    if not instance_id:
        _raise("TensorDock instance response did not include an instance id.")
    network = parse_instance_network(instance, viewer_port)
    return InstanceSummary(
        id=instance_id,
        name=str(name),
        status=str(status),
        public_ip=network.public_ip,
        ssh_port=network.ssh_port,
        network=network,
        raw=instance,
    )


def total_cost(gpu_price_per_hr: float, vcpu: int, ram_gb: int, storage_gb: int, gpu_info: dict[str, Any]) -> float:
    pricing = gpu_info.get("pricing", {})
    return (
        float(gpu_price_per_hr)
        + vcpu * float(pricing.get("per_vcpu_hr", 0.0))
        + ram_gb * float(pricing.get("per_gb_ram_hr", 0.0))
        + storage_gb * float(pricing.get("per_gb_storage_hr", 0.0))
    )


def filter_candidates(
    locations: list[dict[str, Any]],
    *,
    gpu_class: str,
    region: str | None,
    vcpu: int,
    ram_gb: int,
    storage_gb: int,
) -> list[Candidate]:
    accepted_gpu_names = GPU_COMPATIBILITY.get(gpu_class)
    if not accepted_gpu_names:
        _raise(
            f"Unsupported gpu class '{gpu_class}'. Supported values: "
            f"{', '.join(sorted(GPU_COMPATIBILITY))}."
        )

    candidates: list[Candidate] = []
    region_query = (region or "").strip().lower()

    for location in locations:
        location_id = str(location.get("id", ""))
        city = str(location.get("city", ""))
        stateprovince = str(location.get("stateprovince", ""))
        country = str(location.get("country", ""))
        location_blob = " ".join(
            part for part in (location_id, city, stateprovince, country) if part
        ).lower()

        if region_query and region_query not in location_blob:
            continue

        for gpu_info in location.get("gpus", []):
            gpu_name = str(gpu_info.get("v0Name", ""))
            if gpu_name not in accepted_gpu_names:
                continue

            resources = gpu_info.get("resources", {})
            if vcpu > 0 and int(resources.get("max_vcpus", 0)) < vcpu:
                continue
            if ram_gb > 0 and int(resources.get("max_ram_gb", 0)) < ram_gb:
                continue
            required_storage_gb = storage_gb if storage_gb > 0 else MIN_STORAGE_GB
            if int(resources.get("max_storage_gb", 0)) < required_storage_gb:
                continue
            if int(gpu_info.get("max_count", 0)) < 1:
                continue

            network_features = gpu_info.get("network_features", {})
            if not bool(network_features.get("dedicated_ip_available")):
                continue
            candidate = Candidate(
                location_id=location_id,
                city=city,
                stateprovince=stateprovince,
                country=country,
                tier=int(location.get("tier")) if location.get("tier") not in (None, "") else None,
                gpu_v0_name=gpu_name,
                gpu_display_name=str(gpu_info.get("displayName", gpu_name)),
                gpu_count=1,
                gpu_price_per_hr=float(gpu_info.get("price_per_hr", 0.0)),
                max_vcpus=int(resources.get("max_vcpus", 0)),
                max_ram_gb=int(resources.get("max_ram_gb", 0)),
                max_storage_gb=int(resources.get("max_storage_gb", 0)),
                per_vcpu_hr=float(gpu_info.get("pricing", {}).get("per_vcpu_hr", 0.0)),
                per_gb_ram_hr=float(gpu_info.get("pricing", {}).get("per_gb_ram_hr", 0.0)),
                per_gb_storage_hr=float(gpu_info.get("pricing", {}).get("per_gb_storage_hr", 0.0)),
                dedicated_ip_available=bool(network_features.get("dedicated_ip_available")),
                port_forwarding_available=bool(network_features.get("port_forwarding_available")),
                estimated_hourly_cost=total_cost(
                    float(gpu_info.get("price_per_hr", 0.0)),
                    max(vcpu, DEFAULT_AUTO_VCPU),
                    max(ram_gb, DEFAULT_AUTO_RAM_GB),
                    max(storage_gb, MIN_STORAGE_GB),
                    gpu_info,
                ),
            )
            candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda candidate: (
            accepted_gpu_names.index(candidate.gpu_v0_name),
            0 if candidate.dedicated_ip_available else 1,
            0 if candidate.port_forwarding_available else 1,
            candidate.estimated_hourly_cost,
            -(candidate.tier or 0),
            parse_location_label(candidate),
        ),
    )


def build_instance_name(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def resolve_requested_resources(candidate: Candidate, *, vcpu: int, ram_gb: int, storage_gb: int) -> tuple[int, int, int]:
    resolved_vcpu = vcpu if vcpu > 0 else min(DEFAULT_AUTO_VCPU, candidate.max_vcpus)
    resolved_ram_gb = ram_gb if ram_gb > 0 else min(DEFAULT_AUTO_RAM_GB, candidate.max_ram_gb)
    resolved_storage_gb = storage_gb if storage_gb > 0 else MIN_STORAGE_GB

    if resolved_vcpu <= 0 or resolved_ram_gb <= 0:
        _raise(
            f"Selected candidate {candidate.gpu_display_name} does not expose enough CPU or RAM capacity "
            "to build a default launch request."
        )
    if candidate.max_storage_gb < resolved_storage_gb:
        _raise(
            f"Selected candidate {candidate.gpu_display_name} cannot satisfy the minimum storage requirement "
            f"of {resolved_storage_gb} GB."
        )
    return resolved_vcpu, resolved_ram_gb, resolved_storage_gb


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def build_isaac_image_ref(version: str) -> str:
    return f"nvcr.io/nvidia/isaac-sim:{version}"


def build_bootstrap_script(config: AppConfig) -> str:
    if not config.ngc_api_key:
        _raise("Missing NGC API key. Set NGC_API_KEY before launching Isaac Sim instances.")

    isaac_image = build_isaac_image_ref(config.isaac_version)
    ngc_api_key = shell_quote(config.ngc_api_key)
    remote_root = shell_quote(DEFAULT_REMOTE_ROOT)
    runtime_dir = shell_quote(DEFAULT_RUNTIME_DIR)
    viewer_dir = shell_quote(DEFAULT_VIEWER_APP_DIR)
    app_user = shell_quote(config.ssh_user)
    viewer_port = config.viewer_port

    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euxo pipefail
        exec > >(tee -a /var/log/isaac-cloud-bootstrap.log) 2>&1

        export DEBIAN_FRONTEND=noninteractive
        export REMOTE_ROOT={remote_root}
        export RUNTIME_DIR={runtime_dir}
        export VIEWER_DIR={viewer_dir}
        export APP_USER={app_user}
        export ISAAC_SIM_IMAGE={shell_quote(isaac_image)}
        export WEB_VIEWER_PORT={viewer_port}
        export ISAACSIM_SIGNAL_PORT={DEFAULT_ISAAC_SIGNAL_PORT}
        export ISAACSIM_STREAM_PORT={DEFAULT_ISAAC_STREAM_PORT}

        wait_for_apt_lock() {{
            while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
                  fuser /var/lib/dpkg/lock >/dev/null 2>&1 || \
                  fuser /var/cache/apt/archives/lock >/dev/null 2>&1; do
                echo "APT is locked by another process; waiting 10s..."
                sleep 10
            done
        }}

        apt_get_retry() {{
            local attempt=1
            local max_attempts=12
            while true; do
                wait_for_apt_lock
                if apt-get "$@"; then
                    return 0
                fi
                if [ "$attempt" -ge "$max_attempts" ]; then
                    echo "apt-get $* failed after $attempt attempts"
                    return 1
                fi
                attempt=$((attempt + 1))
                echo "apt-get $* failed; retrying in 10s (attempt $attempt/$max_attempts)..."
                sleep 10
            done
        }}

        retry_command() {{
            local max_attempts="$1"
            local delay_seconds="$2"
            shift 2

            local attempt=1
            while true; do
                if "$@"; then
                    return 0
                fi
                if [ "$attempt" -ge "$max_attempts" ]; then
                    echo "command failed after $attempt attempts: $*"
                    return 1
                fi
                attempt=$((attempt + 1))
                echo "command failed; retrying in $delay_seconds seconds (attempt $attempt/$max_attempts): $*"
                sleep "$delay_seconds"
            done
        }}

        install_nodejs() {{
            if command -v node >/dev/null 2>&1; then
                local node_major
                node_major="$(node -p 'parseInt(process.versions.node.split(".")[0], 10)' || echo 0)"
                if [ "${{node_major}}" -ge 20 ]; then
                    return 0
                fi
            fi

            mkdir -p /etc/apt/keyrings
            curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
                gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
            cat >/etc/apt/sources.list.d/nodesource.list <<'EOF'
deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main
EOF
            apt_get_retry update
            apt_get_retry install -y nodejs
        }}

        apt_get_retry update
        apt_get_retry install -y ca-certificates curl gnupg

        curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
        sh /tmp/get-docker.sh
        systemctl enable --now docker

        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            > /etc/apt/sources.list.d/nvidia-container-toolkit.list
        apt_get_retry update
        apt_get_retry install -y nvidia-container-toolkit
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker

        install_nodejs
        npm config set "@nvidia:registry" "https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/" --location=user
        if id -u "$APP_USER" >/dev/null 2>&1; then
            sudo -u "$APP_USER" -H npm config set "@nvidia:registry" "https://edge.urm.nvidia.com/artifactory/api/npm/omniverse-client-npm/" --location=user
        fi

        mkdir -p "$REMOTE_ROOT"
        if id -u "$APP_USER" >/dev/null 2>&1; then
            chown -R "$APP_USER:$APP_USER" "$REMOTE_ROOT"
        fi
        if [ ! -f "$VIEWER_DIR/package.json" ]; then
            rm -rf "$VIEWER_DIR"
            sudo -u "$APP_USER" -H env REMOTE_ROOT="$REMOTE_ROOT" VIEWER_DIR="$VIEWER_DIR" bash -lc '
                set -euo pipefail
                cd "$REMOTE_ROOT"
                npx @nvidia/create-ov-web-rtc-app --name "$(basename "$VIEWER_DIR")" --sample local-sample
            '
        fi

        mkdir -p "$RUNTIME_DIR"/cache/main/ov
        mkdir -p "$RUNTIME_DIR"/cache/main/warp
        mkdir -p "$RUNTIME_DIR"/cache/computecache
        mkdir -p "$RUNTIME_DIR"/config
        mkdir -p "$RUNTIME_DIR"/data/documents
        mkdir -p "$RUNTIME_DIR"/data/Kit
        mkdir -p "$RUNTIME_DIR"/logs
        mkdir -p "$RUNTIME_DIR"/pkg
        chown -R 1234:1234 "$RUNTIME_DIR"

        docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
        printf '%s\\n' {ngc_api_key} | docker login nvcr.io --username '$oauthtoken' --password-stdin
        retry_command 3 10 docker pull "$ISAAC_SIM_IMAGE"

        sudo -u "$APP_USER" -H env VIEWER_DIR="$VIEWER_DIR" bash -lc '
            set -euo pipefail
            cd "$VIEWER_DIR"
            perl -0pi -e "s/signalingServer: '\\''127[.]0[.]0[.]1'\\'',/signalingServer: window.location.hostname,\\n            signalingPort: ${DEFAULT_ISAAC_SIGNAL_PORT},/" src/main.ts
            npm install
            npm run build
        '

        systemctl enable isaac-cloud-isaac.service
        systemctl enable isaac-cloud-viewer.service
        systemctl restart isaac-cloud-isaac.service
        systemctl restart isaac-cloud-viewer.service
        """
    )


def build_isaac_runtime_script(config: AppConfig) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euxo pipefail
        exec > >(tee -a /var/log/isaac-cloud-isaac.log) 2>&1

        export RUNTIME_DIR={shell_quote(DEFAULT_RUNTIME_DIR)}
        export ISAAC_SIM_IMAGE={shell_quote(build_isaac_image_ref(config.isaac_version))}
        export ISAACSIM_SIGNAL_PORT={DEFAULT_ISAAC_SIGNAL_PORT}
        export ISAACSIM_STREAM_PORT={DEFAULT_ISAAC_STREAM_PORT}

        detect_public_ip() {{
            local candidate=""
            for endpoint in \
                "https://api.ipify.org" \
                "https://ifconfig.me" \
                "https://ipv4.icanhazip.com"; do
                candidate="$(curl -fsSL --max-time 10 "$endpoint" 2>/dev/null | tr -d '[:space:]')" || true
                if [[ "$candidate" =~ ^[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+$ ]]; then
                    printf '%s\n' "$candidate"
                    return 0
                fi
            done
            return 1
        }}

        PUBLIC_IP="$(detect_public_ip)"
        export PUBLIC_IP

        docker rm -f isaac-sim >/dev/null 2>&1 || true
        exec docker run --name isaac-sim --entrypoint bash --gpus all --rm --network=host \
            -e "ACCEPT_EULA=Y" \
            -e "PRIVACY_CONSENT=Y" \
            -v "$RUNTIME_DIR/cache/main:/isaac-sim/.cache:rw" \
            -v "$RUNTIME_DIR/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
            -v "$RUNTIME_DIR/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
            -v "$RUNTIME_DIR/config:/isaac-sim/.nvidia-omniverse/config:rw" \
            -v "$RUNTIME_DIR/data:/isaac-sim/.local/share/ov/data:rw" \
            -v "$RUNTIME_DIR/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
            -u 1234:1234 \
            "$ISAAC_SIM_IMAGE" \
            -lc "/isaac-sim/isaac-sim.streaming.sh --merge-config=/isaac-sim/config/open_endpoint.toml --allow-root -v --/exts/omni.kit.livestream.app/primaryStream/publicIp=$PUBLIC_IP --/exts/omni.kit.livestream.app/primaryStream/signalPort=$ISAACSIM_SIGNAL_PORT --/exts/omni.kit.livestream.app/primaryStream/streamPort=$ISAACSIM_STREAM_PORT"
        """
    )


def build_viewer_runtime_script(config: AppConfig) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euxo pipefail
        exec > >(tee -a /var/log/isaac-cloud-viewer.log) 2>&1

        export VIEWER_DIR={shell_quote(DEFAULT_VIEWER_APP_DIR)}
        export WEB_VIEWER_PORT={config.viewer_port}
        cd "$VIEWER_DIR"
        exec npx vite preview --host 0.0.0.0 --port "$WEB_VIEWER_PORT"
        """
    )


def build_isaac_systemd_unit(config: AppConfig) -> str:
    return textwrap.dedent(
        f"""\
        [Unit]
        Description=Run NVIDIA Isaac Sim headless streaming container
        Wants=network-online.target docker.service
        After=network-online.target docker.service

        [Service]
        Type=simple
        ExecStart=/usr/local/bin/isaac-cloud-run-isaac
        ExecStop=/usr/bin/docker stop isaac-sim
        Restart=always
        RestartSec=10
        TimeoutStartSec=1800

        [Install]
        WantedBy=multi-user.target
        """
    )


def build_viewer_systemd_unit() -> str:
    return textwrap.dedent(
        """\
        [Unit]
        Description=Run NVIDIA Omniverse Web SDK viewer for Isaac Sim
        Wants=network-online.target isaac-cloud-isaac.service
        After=network-online.target isaac-cloud-isaac.service

        [Service]
        Type=simple
        ExecStart=/usr/local/bin/isaac-cloud-run-viewer
        Restart=always
        RestartSec=10
        TimeoutStartSec=1800

        [Install]
        WantedBy=multi-user.target
        """
    )


def build_cloud_init(config: AppConfig) -> dict[str, Any]:
    return {
        "package_update": True,
        "packages": ["ca-certificates", "curl", "gnupg"],
        "output": {"all": "| tee -a /var/log/cloud-init-output.log"},
        "final_message": "isaac-cloud cloud-init completed after $UPTIME seconds",
        "write_files": [
            {
                "path": "/usr/local/bin/isaac-cloud-bootstrap",
                "content": build_bootstrap_script(config),
                "owner": "root:root",
                "permissions": "0755",
            },
            {
                "path": "/usr/local/bin/isaac-cloud-run-isaac",
                "content": build_isaac_runtime_script(config),
                "owner": "root:root",
                "permissions": "0755",
            },
            {
                "path": "/usr/local/bin/isaac-cloud-run-viewer",
                "content": build_viewer_runtime_script(config),
                "owner": "root:root",
                "permissions": "0755",
            },
            {
                "path": "/etc/systemd/system/isaac-cloud-isaac.service",
                "content": build_isaac_systemd_unit(config),
                "owner": "root:root",
                "permissions": "0644",
            },
            {
                "path": "/etc/systemd/system/isaac-cloud-viewer.service",
                "content": build_viewer_systemd_unit(),
                "owner": "root:root",
                "permissions": "0644",
            },
            {
                "path": "/usr/local/bin/isaac-cloud-debug-report",
                "content": textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail

                    echo "== cloud-init status =="
                    cloud-init status --long || true
                    echo

                    echo "== cloud-init-output.log =="
                    tail -n 200 /var/log/cloud-init-output.log || true
                    echo

                    echo "== isaac-cloud-bootstrap.log =="
                    tail -n 200 /var/log/isaac-cloud-bootstrap.log || true
                    echo

                    echo "== isaac-cloud-isaac service =="
                    systemctl status isaac-cloud-isaac.service --no-pager || true
                    echo

                    echo "== isaac-cloud-isaac journal =="
                    journalctl -u isaac-cloud-isaac.service --no-pager -n 200 || true
                    echo

                    echo "== isaac-cloud-isaac.log =="
                    tail -n 200 /var/log/isaac-cloud-isaac.log || true
                    echo

                    echo "== isaac-cloud-viewer service =="
                    systemctl status isaac-cloud-viewer.service --no-pager || true
                    echo

                    echo "== isaac-cloud-viewer journal =="
                    journalctl -u isaac-cloud-viewer.service --no-pager -n 200 || true
                    echo

                    echo "== isaac-cloud-viewer.log =="
                    tail -n 200 /var/log/isaac-cloud-viewer.log || true
                    """
                ),
                "owner": "root:root",
                "permissions": "0755",
            },
        ],
        "runcmd": [
            "systemctl daemon-reload",
            "bash -lc /usr/local/bin/isaac-cloud-bootstrap",
        ],
    }


def build_launch_payload(
    *,
    config: AppConfig,
    candidate: Candidate,
    instance_name: str,
    ssh_key: str,
    vcpu: int,
    ram_gb: int,
    storage_gb: int,
) -> dict[str, Any]:
    return {
        "name": instance_name,
        "type": "virtualmachine",
        "image": DEFAULT_IMAGE,
        "resources": {
            "vcpu_count": vcpu,
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "gpus": {candidate.gpu_v0_name: {"count": candidate.gpu_count}},
        },
        "location_id": candidate.location_id,
        "useDedicatedIp": candidate.dedicated_ip_available,
        "ssh_key": ssh_key,
        "cloud_init": build_cloud_init(config),
    }


def wait_for_instance_state(
    client: TensorDockClient,
    instance_id: str,
    *,
    viewer_port: int,
    target_states: set[str],
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> InstanceSummary:
    deadline = time.time() + timeout_seconds
    last_summary: InstanceSummary | None = None

    while time.time() < deadline:
        summary = parse_instance_summary(client.get_instance(instance_id), viewer_port)
        last_summary = summary
        if normalize_status(summary.status) in target_states:
            return summary
        time.sleep(poll_interval_seconds)

    if last_summary is None:
        _raise("Timed out before receiving any instance state from TensorDock.")

    _raise(
        f"Timed out waiting for instance {instance_id} to reach "
        f"{', '.join(sorted(target_states))}. Last status was '{last_summary.status}'."
    )


def check_tcp_connectivity(host: str, port: int, timeout_seconds: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def wait_for_ssh(host: str, port: int, *, timeout_seconds: int, poll_interval_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if check_tcp_connectivity(host, port, timeout_seconds=5.0):
            return True
        time.sleep(poll_interval_seconds)
    return False


def format_ssh_target(config: AppConfig, summary: InstanceSummary) -> str:
    if not summary.network.ssh_host:
        _raise("Instance does not have a public IP yet, so an SSH target cannot be built.")
    port_flag = ""
    if summary.network.ssh_port and summary.network.ssh_port != 22:
        port_flag = f" -p {summary.network.ssh_port}"
    key_flag = ""
    if config.ssh_private_key_path:
        key_flag = f" -i {config.ssh_private_key_path}"
    return f"ssh{key_flag}{port_flag} {config.ssh_user}@{summary.network.ssh_host}"


def format_viewer_url(summary: InstanceSummary) -> str:
    if not summary.public_ip:
        _raise("Instance does not have a public IP yet, so a viewer URL cannot be built.")
    return f"http://{summary.public_ip}:{summary.network.viewer_port}"


def format_viewer_ports() -> str:
    return (
        f"TCP {DEFAULT_VIEWER_PORT} (web viewer UI), "
        f"TCP {DEFAULT_ISAAC_SIGNAL_PORT} (WebRTC signaling), "
        f"UDP {DEFAULT_ISAAC_STREAM_PORT} (WebRTC media)"
    )


def instance_ref_from_state(state: dict[str, Any]) -> str:
    instance_id = nested_get(state, "last_instance", "id")
    if not instance_id:
        _raise("No tracked instance found in local state. Launch an instance first.")
    return str(instance_id)


def record_instance_state(
    summary: InstanceSummary,
    *,
    provider: str,
    gpu_class: str,
    region: str | None,
    vcpu: int,
    ram_gb: int,
    storage_gb: int,
    selected_candidate: Candidate | None,
    state_path: Path = DEFAULT_STATE_PATH,
) -> None:
    state = load_state(state_path)
    state["last_instance"] = {
        "provider": provider,
        "id": summary.id,
        "name": summary.name,
        "status": summary.status,
        "public_ip": summary.public_ip,
        "ssh_port": summary.ssh_port,
        "gpu_class": gpu_class,
        "region": region,
        "vcpu": vcpu,
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "viewer_port": summary.network.viewer_port,
    }
    if selected_candidate is not None:
        state["last_instance"]["selection"] = {
            "location_id": selected_candidate.location_id,
            "location_label": parse_location_label(selected_candidate),
            "gpu_v0_name": selected_candidate.gpu_v0_name,
            "gpu_display_name": selected_candidate.gpu_display_name,
            "estimated_hourly_cost": selected_candidate.estimated_hourly_cost,
            "dedicated_ip_available": selected_candidate.dedicated_ip_available,
            "port_forwarding_available": selected_candidate.port_forwarding_available,
        }
    save_state(state, state_path)


def print_instance_summary(
    summary: InstanceSummary,
    *,
    config: AppConfig,
) -> None:
    typer.echo(f"Instance: {summary.name}")
    typer.echo(f"ID: {summary.id}")
    typer.echo(f"Status: {summary.status}")
    if summary.public_ip:
        typer.echo(f"Public IP: {summary.public_ip}")
    if summary.ssh_port:
        typer.echo(f"SSH Port: {summary.ssh_port}")
    if summary.public_ip:
        typer.echo(f"SSH: {format_ssh_target(config, summary)}")
        typer.echo(f"Viewer URL: {format_viewer_url(summary)}")
        typer.echo(f"Viewer Ports: {format_viewer_ports()}")
        typer.echo("Viewer Access: Browser client loads over TCP 8210 and then connects to Isaac Sim over WebRTC.")


def print_catalog(candidates: list[Candidate], *, gpu_class: str, vcpu: int, ram_gb: int, storage_gb: int) -> None:
    vcpu_label = str(vcpu) if vcpu > 0 else "any"
    ram_label = str(ram_gb) if ram_gb > 0 else "any"
    storage_label = str(storage_gb) if storage_gb > 0 else f"provider-min ({MIN_STORAGE_GB}+)"
    typer.echo(
        "Launchable TensorDock offerings "
        f"for gpu_class={gpu_class}, vcpu>={vcpu_label}, ram_gb>={ram_label}, storage_gb>={storage_label}"
    )
    typer.echo("")
    for candidate in candidates:
        location_label = parse_location_label(candidate)
        typer.echo(
            f"{candidate.gpu_display_name} | {location_label} ({candidate.location_id}) | "
            f"max {candidate.max_vcpus} vCPU / {candidate.max_ram_gb} GB RAM / "
            f"{candidate.max_storage_gb} GB storage | "
            f"dedicated_ip={format_bool_flag(candidate.dedicated_ip_available)} | "
            f"port_forwarding={format_bool_flag(candidate.port_forwarding_available)} | "
            f"est ${candidate.estimated_hourly_cost:.3f}/hr"
        )


def print_instances(summaries: list[InstanceSummary], *, include_all: bool) -> None:
    if include_all:
        typer.echo("TensorDock instances")
    else:
        typer.echo("Running TensorDock instances")
    typer.echo("")
    for summary in summaries:
        details = [summary.status]
        if summary.public_ip:
            details.append(f"ip={summary.public_ip}")
        if summary.ssh_port:
            details.append(f"ssh_port={summary.ssh_port}")
        typer.echo(f"{summary.name} ({summary.id}) | " + " | ".join(details))


def get_client_and_config() -> tuple[TensorDockClient, AppConfig]:
    config = load_app_config()
    return TensorDockClient(config.api_token), config


@app.command()
def catalog(
    gpu_class: str = typer.Option(None, help="Minimum GPU class to request."),
    region: str = typer.Option(None, help="Substring match against location id/city/state/country."),
    vcpu: int = typer.Option(None, help="Minimum vCPU capacity required."),
    ram_gb: int = typer.Option(None, help="Minimum RAM capacity in GB."),
    storage_gb: int = typer.Option(None, help="Minimum storage capacity in GB."),
) -> None:
    try:
        client, config = get_client_and_config()
        effective_gpu_class = gpu_class or config.default_gpu_class
        effective_region = region if region is not None else config.default_region
        effective_vcpu = vcpu or config.default_vcpu
        effective_ram_gb = ram_gb or config.default_ram_gb
        effective_storage_gb = storage_gb or config.default_storage_gb

        try:
            locations = client.list_locations()
            candidates = filter_candidates(
                locations,
                gpu_class=effective_gpu_class,
                region=effective_region,
                vcpu=effective_vcpu,
                ram_gb=effective_ram_gb,
                storage_gb=effective_storage_gb,
            )
            if not candidates:
                _raise(
                    "No compatible TensorDock catalog entries matched the requested constraints. "
                    "Try a different region or smaller resource request."
                )
            print_catalog(
                candidates,
                gpu_class=effective_gpu_class,
                vcpu=effective_vcpu,
                ram_gb=effective_ram_gb,
                storage_gb=effective_storage_gb,
            )
        finally:
            client.close()
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def instances(
    all: bool = typer.Option(False, "--all", help="Include stopped and other non-running instances."),
) -> None:
    try:
        client, config = get_client_and_config()
        try:
            raw_instances = client.list_instances()
            summaries = [parse_instance_summary(instance, config.viewer_port) for instance in raw_instances]
            if not all:
                summaries = [summary for summary in summaries if normalize_status(summary.status) in RUNNING_STATES]
            summaries = sorted(summaries, key=lambda summary: (summary.name.lower(), summary.id))
            if not summaries:
                if all:
                    typer.echo("No TensorDock instances found.")
                else:
                    typer.echo("No running TensorDock instances found.")
                return
            print_instances(summaries, include_all=all)
        finally:
            client.close()
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def launch(
    ctx: typer.Context,
    gpu_class: str = typer.Option(None, help="Minimum GPU class to request."),
    region: str = typer.Option(None, help="Substring match against location id/city/state/country."),
    vcpu: int = typer.Option(None, help="Requested vCPU count."),
    ram_gb: int = typer.Option(None, help="Requested RAM in GB."),
    storage_gb: int = typer.Option(None, help="Requested storage in GB."),
    instance_name: str = typer.Option(None, help="Explicit instance name."),
    timeout_seconds: int = typer.Option(900, help="How long to wait for the instance to reach running."),
    ssh_timeout_seconds: int = typer.Option(120, help="How long to wait for SSH reachability after the instance is running."),
) -> None:
    if all(
        ctx.get_parameter_source(param_name) == ParameterSource.DEFAULT
        for param_name in (
            "gpu_class",
            "region",
            "vcpu",
            "ram_gb",
            "storage_gb",
            "instance_name",
            "timeout_seconds",
            "ssh_timeout_seconds",
        )
    ):
        typer.echo(ctx.get_help())
        raise typer.Exit()

    try:
        client, config = get_client_and_config()
        if not config.ngc_api_key:
            _raise("Missing NGC API key. Set NGC_API_KEY or configure [ngc].api_key before launch.")
        effective_gpu_class = gpu_class or config.default_gpu_class
        effective_region = region if region is not None else config.default_region
        effective_vcpu = vcpu or config.default_vcpu
        effective_ram_gb = ram_gb or config.default_ram_gb
        effective_storage_gb = storage_gb or config.default_storage_gb

        try:
            locations = client.list_locations()
            candidates = filter_candidates(
                locations,
                gpu_class=effective_gpu_class,
                region=effective_region,
                vcpu=effective_vcpu,
                ram_gb=effective_ram_gb,
                storage_gb=effective_storage_gb,
            )
            if not candidates:
                _raise(
                    "No compatible TensorDock location with dedicated IP support matched the requested constraints. "
                    "Try a different region or smaller resource request."
                )
            resolved_name = instance_name or build_instance_name(config.instance_name_prefix)
            selected: Candidate | None = None
            instance_id: str | None = None
            create_failures: list[str] = []

            for index, candidate in enumerate(candidates, start=1):
                requested_vcpu, requested_ram_gb, requested_storage_gb = resolve_requested_resources(
                    candidate,
                    vcpu=effective_vcpu,
                    ram_gb=effective_ram_gb,
                    storage_gb=effective_storage_gb,
                )
                typer.echo(
                    "Selected TensorDock candidate: "
                    f"{candidate.gpu_display_name} in {parse_location_label(candidate)} "
                    f"(estimated ${candidate.estimated_hourly_cost:.3f}/hr)"
                )
                typer.echo(
                    "Requested instance size: "
                    f"{requested_vcpu} vCPU / {requested_ram_gb} GB RAM / {requested_storage_gb} GB storage"
                )
                try:
                    created = client.create_instance(
                        build_launch_payload(
                            config=config,
                            candidate=candidate,
                            instance_name=resolved_name,
                            ssh_key=config.ssh_key,
                            vcpu=requested_vcpu,
                            ram_gb=requested_ram_gb,
                            storage_gb=requested_storage_gb,
                        )
                    )
                    instance_id = extract_instance_id(created)
                    if not instance_id:
                        _raise(
                            "TensorDock create-instance response did not include an instance id. "
                            f"Observed payload shape: {describe_payload_shape(created)}."
                        )
                    selected = candidate
                    break
                except IsaacCloudError as exc:
                    create_failures.append(
                        f"{parse_location_label(candidate)} ({candidate.gpu_display_name}): {exc}"
                    )
                    if index < len(candidates) and is_retryable_create_error(str(exc)):
                        typer.echo(
                            "Create attempt failed for this candidate; trying the next compatible offer..."
                        )
                        continue
                    raise

            if selected is None or instance_id is None:
                if create_failures:
                    _raise("All candidate create attempts failed:\n" + "\n".join(create_failures))
                _raise("No TensorDock candidate could be provisioned.")

            typer.echo(f"Created instance {resolved_name} ({instance_id}). Waiting for running state...")
            summary = wait_for_instance_state(
                client,
                instance_id,
                viewer_port=config.viewer_port,
                target_states=RUNNING_STATES,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=5,
            )

            if summary.network.ssh_host and summary.network.ssh_port:
                typer.echo("Instance is running. Checking SSH reachability...")
                ssh_ready = wait_for_ssh(
                    summary.network.ssh_host,
                    summary.network.ssh_port,
                    timeout_seconds=ssh_timeout_seconds,
                    poll_interval_seconds=5,
                )
                if ssh_ready:
                    typer.echo("SSH port is reachable.")
                else:
                    typer.echo(
                        "SSH did not become reachable within the configured timeout. "
                        "The instance may still be finishing boot."
                    )

            record_instance_state(
                summary,
                provider="tensordock",
                gpu_class=effective_gpu_class,
                region=effective_region,
                vcpu=requested_vcpu,
                ram_gb=requested_ram_gb,
                storage_gb=requested_storage_gb,
                selected_candidate=selected,
            )
            print_instance_summary(summary, config=config)
        finally:
            client.close()
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def status(
    instance_id: str = typer.Option(None, help="Override the instance id instead of using local state."),
) -> None:
    try:
        client, config = get_client_and_config()
        try:
            state = load_state()
            resolved_instance_id = instance_id or instance_ref_from_state(state)
            summary = parse_instance_summary(client.get_instance(resolved_instance_id), config.viewer_port)
            record_instance_state(
                summary,
                provider="tensordock",
                gpu_class=nested_get(state, "last_instance", "gpu_class") or config.default_gpu_class,
                region=nested_get(state, "last_instance", "region"),
                vcpu=int(nested_get(state, "last_instance", "vcpu") or config.default_vcpu),
                ram_gb=int(nested_get(state, "last_instance", "ram_gb") or config.default_ram_gb),
                storage_gb=int(
                    nested_get(state, "last_instance", "storage_gb") or config.default_storage_gb
                ),
                selected_candidate=None,
            )
            print_instance_summary(summary, config=config)
        finally:
            client.close()
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("viewer")
def viewer(
    instance_id: str = typer.Option(None, help="Override the instance id instead of using local state."),
) -> None:
    try:
        client, config = get_client_and_config()
        try:
            resolved_instance_id = instance_id or instance_ref_from_state(load_state())
            summary = parse_instance_summary(client.get_instance(resolved_instance_id), config.viewer_port)
            typer.echo(f"Viewer URL: {format_viewer_url(summary)}")
            typer.echo(f"Viewer Ports: {format_viewer_ports()}")
            typer.echo("Notes: The browser loads the viewer on TCP 8210, then connects over WebRTC on TCP 49100 and UDP 47998.")
        finally:
            client.close()
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc

def mutate_instance_state(action: str, instance_id: str | None) -> None:
    client, config = get_client_and_config()
    try:
        state = load_state()
        resolved_instance_id = instance_id or instance_ref_from_state(state)
        if action == "stop":
            client.stop_instance(resolved_instance_id)
            target_states = STOPPED_STATES
        elif action == "resume":
            client.start_instance(resolved_instance_id)
            target_states = RUNNING_STATES
        elif action == "destroy":
            client.delete_instance(resolved_instance_id)
            clear_state()
            typer.echo(f"Destroyed instance {resolved_instance_id}.")
            return
        else:
            _raise(f"Unsupported action: {action}")

        typer.echo(f"Requested {action} for instance {resolved_instance_id}. Waiting for state change...")
        summary = wait_for_instance_state(
            client,
            resolved_instance_id,
            viewer_port=config.viewer_port,
            target_states=target_states,
            timeout_seconds=600,
            poll_interval_seconds=5,
        )
        record_instance_state(
            summary,
            provider="tensordock",
            gpu_class=nested_get(state, "last_instance", "gpu_class") or config.default_gpu_class,
            region=nested_get(state, "last_instance", "region"),
            vcpu=int(nested_get(state, "last_instance", "vcpu") or config.default_vcpu),
            ram_gb=int(nested_get(state, "last_instance", "ram_gb") or config.default_ram_gb),
            storage_gb=int(nested_get(state, "last_instance", "storage_gb") or config.default_storage_gb),
            selected_candidate=None,
        )
        typer.echo(f"Instance {resolved_instance_id} is now {summary.status}.")
    finally:
        client.close()


@app.command()
def stop(
    ctx: typer.Context,
    instance_id: str = typer.Option(None, help="Stop requires an explicit instance id."),
) -> None:
    try:
        if ctx.get_parameter_source("instance_id") == ParameterSource.DEFAULT:
            typer.echo(ctx.get_help())
            _raise("Pass --instance-id.")
        mutate_instance_state("stop", instance_id)
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def resume(
    ctx: typer.Context,
    instance_id: str = typer.Option(None, help="Resume requires an explicit instance id."),
) -> None:
    try:
        if ctx.get_parameter_source("instance_id") == ParameterSource.DEFAULT:
            typer.echo(ctx.get_help())
            _raise("Pass --instance-id.")
        mutate_instance_state("resume", instance_id)
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def destroy(
    ctx: typer.Context,
    instance_id: str = typer.Option(None, help="Override the instance id instead of using local state."),
    destroy_all: bool = typer.Option(False, "--all", help="Destroy all TensorDock instances."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    try:
        has_instance_id = ctx.get_parameter_source("instance_id") != ParameterSource.DEFAULT
        has_all = ctx.get_parameter_source("destroy_all") != ParameterSource.DEFAULT and destroy_all

        if has_instance_id == has_all:
            typer.echo(ctx.get_help())
            _raise("Pass exactly one of --instance-id or --all.")

        if destroy_all:
            client, _config = get_client_and_config()
            try:
                raw_instances = client.list_instances()
                summaries = [
                    parse_instance_summary(instance, viewer_port=DEFAULT_VIEWER_PORT)
                    for instance in raw_instances
                ]
                if not summaries:
                    typer.echo("No TensorDock instances found.")
                    clear_state()
                    return

                if not yes:
                    confirmed = typer.confirm(
                        f"Destroy all TensorDock instances ({len(summaries)} total)?",
                        default=False,
                    )
                    if not confirmed:
                        raise typer.Exit(code=1)

                for summary in summaries:
                    client.delete_instance(summary.id)
                    typer.echo(f"Destroyed instance {summary.id} ({summary.name}).")
                clear_state()
            finally:
                client.close()
            return

        resolved_instance_id = instance_id
        if resolved_instance_id is None:
            resolved_instance_id = instance_ref_from_state(load_state())

        if not yes:
            confirmed = typer.confirm(f"Destroy TensorDock instance {resolved_instance_id}?", default=False)
            if not confirmed:
                raise typer.Exit(code=1)

        mutate_instance_state("destroy", resolved_instance_id)
    except IsaacCloudError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except httpx.HTTPStatusError as exc:
        typer.echo(
            f"TensorDock API error: {exc.response.status_code} {exc.response.text}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        typer.echo(f"Network error talking to TensorDock: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@sync_app.command("pull")
def sync_pull() -> None:
    typer.echo("sync pull is not implemented yet.")


@sync_app.command("push")
def sync_push() -> None:
    typer.echo("sync push is not implemented yet.")


app.add_typer(sync_app, name="sync")


if __name__ == "__main__":
    app()
