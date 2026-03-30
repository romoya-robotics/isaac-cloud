# isaac-cloud

`isaac-cloud` is a Python CLI for provisioning TensorDock GPU instances intended to run NVIDIA Isaac Sim.

## Current State

The repo now implements the Phase 1 launch skeleton against TensorDock:

- query TensorDock locations
- filter and rank compatible GPU candidates
- create an Ubuntu 24.04 instance
- attach cloud-init that installs Docker and NVIDIA Container Toolkit
- bootstrap the pinned Isaac Sim container, plus optional MCP and optional web viewer components
- poll until the instance reaches `running`
- perform a best-effort SSH reachability check
- print SSH access details plus optional viewer and MCP connection details

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
isaac_version = "5.1.0"

[viewer]
enabled = false
port = 8210

[mcp]
enabled = false
repo_url = "https://github.com/omni-mcp/isaac-sim-mcp"
extension_port = 8766
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
uv run python isaac_cloud.py instances --all
uv run python isaac_cloud.py launch --viewer
uv run python isaac_cloud.py launch --mcp
uv run python isaac_cloud.py status --instance-id <INSTANCE_ID>
uv run python isaac_cloud.py viewer --instance-id <INSTANCE_ID>
uv run python isaac_cloud.py stop --instance-id <INSTANCE_ID>
uv run python isaac_cloud.py resume --instance-id <INSTANCE_ID>
uv run python isaac_cloud.py destroy --instance-id <INSTANCE_ID> --yes
uv run python isaac_cloud.py destroy --all --yes
```

Commands that inspect or mutate a specific VM require `--instance-id`, unless you use `destroy --all`.

## Viewer Workflow

The web viewer is optional. A launch only builds and starts the Omniverse Web SDK viewer when it is enabled in config or by an explicit launch flag.

1. Launch a VM with viewer enabled.

```bash
uv run python isaac_cloud.py launch --viewer
```

You can combine `--viewer` with your usual launch filters:

```bash
uv run python isaac_cloud.py launch --viewer --region delaware
```

2. Or enable the viewer by default in `config.toml`.

```toml
[viewer]
enabled = true
port = 8210
```

3. When viewer is enabled, launch prints the public viewer URL and the required ports.

4. To print the viewer URL and ports for an existing instance, use:

```bash
uv run python isaac_cloud.py viewer --instance-id <INSTANCE_ID>
```

## MCP Workflow

This repo can launch Isaac Sim with the community `omni-mcp/isaac-sim-mcp` extension enabled inside the Isaac container.

The MCP extension runs on the VM inside Isaac Sim. The Python MCP server from the community repo should run on your local machine and connect through an SSH tunnel to the VM.

1. Launch a VM with MCP enabled.

```bash
uv run python isaac_cloud.py launch --mcp
```

You can combine `--mcp` with your usual launch filters, for example:

```bash
uv run python isaac_cloud.py launch --mcp --region delaware
```

When launch succeeds, the CLI prints the SSH command and MCP tunnel command. If viewer is also enabled, it prints the viewer URL as well.

2. Wait for bootstrap to finish.

The first boot may take a while because cloud-init installs Docker, validates the NVIDIA runtime, optionally builds the web viewer, clones the MCP repo on the VM, and pulls `nvcr.io/nvidia/isaac-sim:5.1.0`.

To check progress:

```bash
uv run python isaac_cloud.py status --instance-id <INSTANCE_ID> --verbose
```

The VM is ready for MCP when:
- `cloud-init` reports `done`
- `isaac-cloud-isaac.service` is active
- the Isaac log shows `Isaac Sim MCP server started on localhost:8766`

3. Open an SSH tunnel from your machine to the VM.

Use the tunnel command printed by `launch`. It will look like:

```bash
ssh -i /home/you/.ssh/id_ed25519_tensordock -N -L 8766:127.0.0.1:8766 user@<VM_IP>
```

Keep that terminal open while using MCP.

4. Clone the community MCP repo locally.

```bash
git clone https://github.com/omni-mcp/isaac-sim-mcp
cd isaac-sim-mcp
```

5. Create a local Python environment and install the community server dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

6. Run the community MCP server locally.

```bash
python isaac_mcp/server.py
```

That local MCP server should connect to `localhost:8766`, which is forwarded through the SSH tunnel to the Isaac Sim MCP extension running on the VM.

7. Point your MCP client at the local server.

The exact config depends on your MCP client, but the shape is:
- command: your local Python executable
- args: `isaac_mcp/server.py`
- working directory: your local clone of `isaac-sim-mcp`

The important point is that your MCP client talks to a local stdio server, and that local server talks through the SSH tunnel to the VM.

## Notes

- The GPU compatibility table is intentionally conservative and can be expanded once we validate more TensorDock SKUs.
- TensorDock response shapes for instance networking may vary, so `public_ip` and `ssh_port` parsing is defensive.
- `TENSORDOCK_API_TOKEN` authenticates API requests, while `TENSORDOCK_SSH_KEY` should contain the actual public key material sent to the `/api/v2/instances` endpoint.
- `launch` now expects `NGC_API_KEY` so cloud-init can authenticate to `nvcr.io` and pull the Isaac Sim image during first boot.
- The live `config.toml` file is ignored by git. Start from `config.example.toml`.
- `vcpu = 0`, `ram_gb = 0`, and `storage_gb = 0` mean "do not constrain candidate selection by that resource." Launch then auto-picks a small baseline size for the actual VM request, with storage still respecting the provider minimum.
- SSH readiness checks are best-effort and only run when a local private key path is configured.
- Browser viewer access is not SSH-tunneled. When viewer is enabled, the browser loads the viewer over TCP `8210`, then WebRTC uses TCP `49100` and UDP `47998`.
- `viewer` prints the public access details for the target instance using its IP address and configured viewer port. It does not verify whether that instance was actually launched with viewer enabled.
- The bootstrap runs a direct `nvcr.io/nvidia/isaac-sim:5.1.0` container. When viewer is enabled, it also builds and runs a generated Omniverse Web SDK `local-sample` viewer app.
- When MCP is enabled, bootstrap clones the configured `omni-mcp/isaac-sim-mcp` repo onto the VM, mounts that repo into the Isaac container as an extension source, and enables `isaac.sim.mcp_extension`.
- The Python MCP server from the community repo is not run inside the Isaac container. For the current prototype flow, launch with `--mcp`, open an SSH tunnel to the configured MCP port, and run the community `isaac_mcp/server.py` separately against the tunneled localhost port.
- For cloud-init debugging on the VM, inspect `/var/log/cloud-init-output.log`, `/var/log/isaac-cloud-bootstrap.log`, `/var/log/isaac-cloud-isaac.log`, `/var/log/isaac-cloud-viewer.log`, or run `/usr/local/bin/isaac-cloud-debug-report`.
- The bootstrap now waits and retries when `apt` is locked by `unattended-upgrades`, which was blocking the NVIDIA container toolkit install on Ubuntu 24 images.
- `sync pull` and `sync push` are still placeholders; Phase 1 only covers discovery and lifecycle basics.
