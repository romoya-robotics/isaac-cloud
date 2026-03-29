# isaac-cloud

`isaac-cloud` is a Python CLI for provisioning TensorDock GPU instances intended to run NVIDIA Isaac Sim.

## Current State

The repo now implements the Phase 1 launch skeleton against TensorDock:

- query TensorDock locations
- filter and rank compatible GPU candidates
- create an Ubuntu 24.04 instance
- attach cloud-init that installs Docker and NVIDIA Container Toolkit
- bootstrap the pinned Isaac Sim container plus an Omniverse Web SDK viewer app on first boot
- poll until the instance reaches `running`
- perform a best-effort SSH reachability check
- record the last instance in local state
- print SSH and browser-viewer access details

## Run From Source

This project currently starts with a simple Python script entrypoint instead of an installed console script.

```bash
uv sync
uv run python isaac_cloud.py --help
```

That keeps the first versions easy to iterate on while the TensorDock and Isaac Sim flow is still being validated.

## Configuration

Create `config.toml` next to `isaac_cloud.py` before running the script. A checked-in template lives at `config.example.toml`:

```toml
[tensordock]
api_token = "..."
ssh_key = "ssh-ed25519 AAAA... you@example.com"

[ngc]
api_key = "..."

[ssh]
private_key_path = "/home/you/.ssh/id_ed25519"
user = "user"

[defaults]
gpu_class = "rtx4080"
region = "seattle"
vcpu = 0
ram_gb = 0
storage_gb = 0
instance_name_prefix = "isaac-cloud"
viewer_port = 8210
isaac_version = "5.1.0"
```

You can still override individual settings with environment variables when needed:

```bash
export TENSORDOCK_API_TOKEN=...
export TENSORDOCK_SSH_KEY=...
export NGC_API_KEY=...
```

`TENSORDOCK_API_TOKEN` is the API credential this CLI uses to call the TensorDock API. You can generate or manage it from the TensorDock Developers page:

`https://dashboard.tensordock.com/developers`

`TENSORDOCK_SSH_KEY` should be the literal SSH public key contents to inject into the VM, not the private key.

`NGC_API_KEY` is required for `launch`. The bootstrap flow uses it to authenticate Docker to `nvcr.io` before pulling the pinned Isaac Sim image.

Optional local SSH settings for readiness checks and admin access:

```bash
export ISAAC_CLOUD_SSH_PRIVATE_KEY=~/.ssh/id_ed25519
export ISAAC_CLOUD_SSH_USER=user
```

`ISAAC_CLOUD_SSH_PRIVATE_KEY` is local-only and optional. It is used for generated SSH commands and for best-effort SSH readiness checks.

Optional defaults for launch behavior:

```bash
export ISAAC_CLOUD_GPU_CLASS=rtx4080
export ISAAC_CLOUD_REGION=seattle
export ISAAC_CLOUD_VCPU=0
export ISAAC_CLOUD_RAM_GB=0
export ISAAC_CLOUD_STORAGE_GB=0
```

## Commands

```bash
uv run python isaac_cloud.py catalog
uv run python isaac_cloud.py instances
uv run python isaac_cloud.py launch
uv run python isaac_cloud.py status
uv run python isaac_cloud.py viewer
uv run python isaac_cloud.py stop
uv run python isaac_cloud.py resume
uv run python isaac_cloud.py destroy --yes
```

The last launched instance is stored in `~/.config/isaac-cloud/state.json`.

## Notes

- The GPU compatibility table is intentionally conservative and can be expanded once we validate more TensorDock SKUs.
- TensorDock response shapes for instance networking may vary, so `public_ip` and `ssh_port` parsing is defensive.
- `TENSORDOCK_API_TOKEN` authenticates API requests, while `TENSORDOCK_SSH_KEY` should contain the actual public key material sent to the `/api/v2/instances` endpoint.
- `launch` now expects `NGC_API_KEY` so cloud-init can authenticate to `nvcr.io` and pull the Isaac Sim image during first boot.
- The live `config.toml` file is ignored by git. Start from `config.example.toml`.
- `vcpu = 0`, `ram_gb = 0`, and `storage_gb = 0` mean "do not constrain candidate selection by that resource." Launch then auto-picks a small baseline size for the actual VM request, with storage still respecting the provider minimum.
- SSH readiness checks are best-effort and only run when a local private key path is configured.
- Browser viewer access is not SSH-tunneled. The browser loads the viewer over TCP `8210`, then WebRTC uses TCP `49100` and UDP `47998`.
- `viewer` prints the public access details.
- The bootstrap now runs a direct `nvcr.io/nvidia/isaac-sim:5.1.0` container and a generated Omniverse Web SDK `local-sample` viewer app instead of cloning the public `IsaacSim` repo for Compose.
- For cloud-init debugging on the VM, inspect `/var/log/cloud-init-output.log`, `/var/log/isaac-cloud-bootstrap.log`, `/var/log/isaac-cloud-isaac.log`, `/var/log/isaac-cloud-viewer.log`, or run `/usr/local/bin/isaac-cloud-debug-report`.
- The bootstrap now waits and retries when `apt` is locked by `unattended-upgrades`, which was blocking the NVIDIA container toolkit install on Ubuntu 24 images.
- `sync pull` and `sync push` are still placeholders; Phase 1 only covers discovery and lifecycle basics.
