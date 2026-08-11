# GPU Orchestrator Plan

## Goal

Build a command line utility that can:

1. Allocate a TensorDock GPU VM compatible with NVIDIA Isaac Sim.
2. Bootstrap the VM with Isaac Sim and NVIDIA's browser-based web viewer.
3. Give the user a reliable way to access the viewer from their laptop.
4. Persist user data to S3 so the VM can be stopped or destroyed between sessions.
5. Manage the full lifecycle: launch, inspect, sync, stop, resume, destroy.

## Initial Development Workflow

For the first versions of this project:

- use `uv` for dependency management, virtual environments, lockfile generation, and build tooling
- keep the entrypoint as a simple Python script in the repo rather than a packaged console script
- run the tool as `uv run python main.py <command>`
- defer packaging polish until after the real TensorDock + Isaac Sim flow works end to end

Packaging can come later, but the early milestone should optimize for fast iteration and a minimal execution path.

## Current Recommendation

Target NVIDIA Isaac Sim 5.1.0 as the default supported version.

Reasoning:

- NVIDIA's current docs show 6.0 as an Early Developer Release with incomplete documentation and no normal GA artifact flow yet.
- NVIDIA's 5.1.0 requirements are explicit and stable enough for automation.
- The documented cloud-friendly path is the browser-based WebRTC viewer deployed with Docker Compose.

## Constraints From Research

### NVIDIA Isaac Sim

- Linux container deployment is the supported remote/headless path.
- Minimum supported GPU for x86_64 is a GeForce RTX 4080 with 16 GB VRAM.
- GPUs without RT cores are not supported.
- Ubuntu 22.04 or 24.04 is supported.
- Isaac Sim requires network access for assets and some extensions.
- NVIDIA documents a browser-based web viewer that is started via Docker Compose.
- NVIDIA documents that `--network=host` is required for WebRTC streaming because bridge networking with normal Docker port publishing is insufficient for the media path.
- NVIDIA warns that streaming endpoints do not include built-in auth or TLS and should not be exposed broadly.

### TensorDock

- The API supports:
  - location and hostnode discovery
  - GPU instance creation
  - instance lifecycle management
  - cloud-init on first boot
  - secrets management
- Instance creation supports Ubuntu 24.04 and GPU resource selection.
- Location and hostnode data include GPU inventory, price data, and network feature flags.
- Hostnode/network metadata includes whether port forwarding is available.

## Product Direction

### Infrastructure Provider Scope

v1 will target TensorDock only.

We do not need user-facing provider configuration in the first release, and we should avoid expanding scope into true multi-provider support before the TensorDock flow works well.

However, the implementation should keep a clean architectural boundary between:

- CLI and config
- orchestration/lifecycle logic
- provider-specific API integration
- remote bootstrap logic
- local state tracking

Reasoning:

- we may want to support another GPU provider later
- different providers will expose different lifecycle semantics, networking models, GPU naming schemes, and bootstrapping capabilities
- if TensorDock response shapes leak into the whole codebase, a later provider migration will become a rewrite instead of an adapter

Practical guidance for v1:

- implement TensorDock as the only provider
- keep TensorDock-specific code isolated to a provider adapter layer
- normalize provider data into internal models such as instance spec, instance state, network access, and candidate capacity
- store provider instance IDs separately from our own logical instance metadata

This is an internal architectural constraint, not a user-facing feature requirement for v1.

### Access Model

Use the browser-based viewer as the default access path.

Do not make the native Isaac streaming client the default for v1.

Reasoning:

- Native Isaac streaming relies on both TCP and UDP ports.
- Standard SSH local port forwarding handles TCP, but not UDP media traffic.
- NVIDIA's documented WebRTC flow expects direct network reachability for both signaling and media ports.
- v1 should use a constrained public-port model instead of SSH tunneling for the viewer.

Constrained public-port model:

- keep SSH available for admin access
- expose only the minimum Isaac viewer ports required for the pinned release
- restrict those ports to the user's current public IP or an explicitly configured CIDR
- prefer dedicated IPs when available so viewer access is deterministic
- treat broader public exposure as an opt-in follow-up, not the default

Current implementation note:

- the viewer and Isaac streaming path currently use the VM public IP
- the viewer is not SSH-tunneled
- the MCP extension is reached through an SSH local port forward to `127.0.0.1:8766`
- any persistence or remote sync workflow should assume SSH access is the private control plane for v1

### Storage Model

Use local NVMe/SSD on the VM for runtime performance and S3 for durable persistence.

Do not treat S3 as Isaac Sim's live filesystem.

Reasoning:

- Isaac caches and runtime data are latency-sensitive.
- S3 is suitable for scene files, outputs, user-generated assets, notebooks, and selected logs.
- Shader caches, package caches, and compute caches should remain local and disposable.

## Proposed v1 CLI

The long-term CLI name can still be `isaac-cloud`, but the first implementation should not depend on a packaged binary.

Initial runnable form:

```bash
uv run python main.py <command>
```

Later, if packaging is worth it, we can add a console script that maps to the same command structure.

### Core Commands

- `isaac-cloud launch`
- `isaac-cloud status`
- `isaac-cloud viewer`
- `isaac-cloud sync pull`
- `isaac-cloud sync push`
- `isaac-cloud stop`
- `isaac-cloud resume`
- `isaac-cloud destroy`

### Likely Options

- `--gpu-class rtx4080|rtx4090|l40s|...`
- `--region <location-id or region>`
- `--vcpu <count>`
- `--ram-gb <count>`
- `--storage-gb <count>`
- `--isaac-version <version>`
- `--s3-bucket <bucket>`
- `--s3-prefix <prefix>`
- `--ssh-key <path or key name>`
- `--public-viewer`
- `--instance-name <name>`

## Configuration Model

Use a local config file plus environment variables for secrets.

Suggested config file: `~/.config/isaac-cloud/config.toml`

Example logical fields:

- TensorDock API token
- default SSH key
- default GPU class
- default location preference
- default instance sizing
- Isaac version
- S3 bucket and prefix
- AWS region
- sync include/exclude rules

Secrets should not be committed to disk in plaintext unless the user explicitly chooses that. For v1, environment variables are acceptable:

- `TENSORDOCK_API_TOKEN`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`
- `NGC_API_KEY`

## Deployment Architecture

### Local CLI Responsibilities

The CLI running on the user's machine should:

1. Query TensorDock for available compatible capacity.
2. Choose a target location or hostnode.
3. Submit the instance creation request with cloud-init.
4. Poll until the VM is reachable.
5. Verify the Isaac stack is healthy.
6. Print the SSH command, public viewer endpoint, and the constrained access policy applied to the viewer ports.
7. Coordinate sync before shutdown or destruction.

### Remote VM Responsibilities

The VM should:

1. Install Docker and NVIDIA container runtime dependencies.
2. Verify GPU visibility with `nvidia-smi`.
3. Authenticate to NVIDIA NGC.
4. Create runtime directories.
5. Write a Docker Compose file and environment file.
6. Start Isaac Sim plus the web viewer.
7. Run a small sync helper for S3 pull/push.
8. Optionally run a systemd unit so services survive reboot.

## Provisioning Flow

### Phase 1: Capacity Discovery

Call TensorDock discovery endpoints and rank candidates.

Candidate filters:

- GPU must be at least the requested class.
- GPU must have RT cores and sufficient VRAM.
- RAM must be at least 64 GB by default.
- Storage must be at least 500 GB by default.
- Ubuntu 24.04 image must be available.

Candidate ranking:

- exact or better GPU match
- dedicated IP availability preferred
- otherwise port forwarding support
- lower hourly cost
- preferred location

Open question:

- We need to verify the exact GPU naming scheme returned by TensorDock and define our compatibility mapping. The CLI will likely need a lookup table that says which models satisfy `rtx4080`.

### Phase 2: Instance Creation

Create an Ubuntu 24.04 instance with:

- chosen GPU resource request
- vCPU, RAM, storage
- SSH key
- cloud-init payload

Cloud-init should:

- update apt metadata
- install Docker
- install NVIDIA Container Toolkit
- ensure Docker starts on boot
- create `/opt/gpu-orchestrator`
- write `docker-compose.yml`
- write `.env`
- write sync scripts
- write systemd units
- pull initial user data from S3 if configured
- start the stack

### Phase 3: Readiness Checks

The local CLI should poll for:

1. instance status from TensorDock
2. SSH reachability
3. `nvidia-smi` success
4. Docker daemon readiness
5. Isaac container running
6. web viewer health endpoint or port readiness

### Phase 4: User Access

Once ready, print:

- instance metadata
- SSH command
- viewer URL
- exposed viewer ports
- allowed source IP or CIDR

Default constrained-access shape:

- TCP `49100`
- UDP `47998`
- source allowlist: `<user-public-ip>/32` by default

Then the user opens:

```text
http://<instance-ip>:49100
```

## Remote Filesystem Layout

Suggested layout on the VM:

```text
/opt/isaac-cloud/
  docker-compose.yml
  .env
  sync/
    pull.sh
    push.sh
  systemd/
    isaac-stack.service
  state/
    manifest.json

/var/lib/isaac-cloud/
  isaac/
    cache/
    computecache/
    config/
    data/
    logs/
    pkg/
```

S3-backed durable paths should map to a subset of:

```text
/var/lib/isaac-cloud/isaac/data/
```

We should split durable user data from disposable cache directories early so sync logic stays simple.

## Data Sync Design

### v1 Sync Strategy

Use `aws s3 sync` or the AWS SDK from the CLI/remote helper.

Recommended behavior:

- On launch:
  - if S3 configured, pull prior durable data down to the VM before starting services
- On explicit save or sync:
  - push durable directories to S3
- On stop:
  - push durable directories
  - stop instance
- On destroy:
  - push durable directories
  - destroy instance

### Data Classes

Persist:

- user scenes
- exported assets
- notebooks and scripts
- user config we choose to preserve
- selected logs useful for debugging

Do not persist:

- shader caches
- compute caches
- pulled container layers
- package caches
- temporary runtime artifacts

## Security Posture

Default to private-by-default access.

v1 defaults:

- expose SSH for admin access
- expose only the required viewer ports for the pinned Isaac release
- restrict viewer ports to the user's public IP or configured CIDR
- do not expose unauthenticated viewer endpoints to the general internet
- avoid storing long-lived credentials on disk when possible

Potential follow-up:

- temporary IAM credentials
- encrypted config store
- VPN-based access path
- reverse proxy or TURN/auth layer if we later need broader or more user-friendly access

## Health and Observability

We need enough telemetry to tell whether launch failed because of:

- TensorDock capacity problems
- driver install problems
- Docker or NVIDIA runtime problems
- NGC auth problems
- Isaac startup failures
- web viewer startup failures
- S3 sync failures

Suggested v1 observability:

- structured local CLI logs
- remote bootstrap log file
- `systemd` service logs
- `docker compose logs` capture
- a simple readiness report command

## Major Technical Risks

### 1. GPU Driver State On Fresh TensorDock VMs

This is the biggest unknown.

We need to validate whether TensorDock GPU Ubuntu images already include a compatible NVIDIA driver. If they do not, our bootstrap must install the driver and may require a reboot before Docker and NVIDIA runtime can function correctly.

Mitigation:

- design the bootstrap for an idempotent two-stage flow
- after instance creation, validate `nvidia-smi`
- if driver install is needed, install and reboot, then continue

### 2. Web Viewer Packaging Details

NVIDIA documents the browser-based viewer via Docker Compose, but the exact compose inputs, images, and version pinning need to be validated against the current Isaac release artifacts we intend to support.

Mitigation:

- pin an exact Isaac version in v1
- test the full compose flow manually before automating
- avoid supporting multiple Isaac versions initially

### 3. TensorDock Networking Details

We need to confirm whether the browser viewer works cleanly with:

- dedicated IP
- host networking
- constrained public TCP/UDP exposure with source IP allowlists

We may also need to choose between location-based and hostnode-based provisioning depending on how deterministic port behavior must be.

Mitigation:

- validate a manual end-to-end launch on a real instance
- prefer dedicated IP plus source-IP-restricted viewer ports for predictable access

### 4. Secrets Injection

We need a clean way to get:

- TensorDock API credentials to the local CLI
- NGC API key to the remote VM
- AWS credentials to either the local CLI or the remote VM

Mitigation:

- use local environment variables first
- only inject the minimum remote credentials needed
- prefer short-lived AWS credentials if available

## Suggested Implementation Phases

### Phase 0: Manual Validation

Before writing much code:

1. Provision one TensorDock VM manually.
2. Validate GPU model, driver state, and `nvidia-smi`.
3. Manually install Docker and NVIDIA container runtime.
4. Manually launch pinned Isaac + web viewer.
5. Verify browser access through source-IP-restricted public viewer ports.
6. Validate which directories actually need S3 persistence.

Deliverable:

- a working manual runbook

### Phase 1: Discovery And Launch Skeleton

Implement:

- repo-root `main.py` command entrypoint run through `uv`
- TensorDock API client
- candidate filtering and ranking
- instance creation
- instance polling
- SSH readiness detection

Deliverable:

- `uv run python main.py launch` creates a compatible VM and reports status

### Phase 2: Automated Bootstrap

Implement:

- cloud-init generation
- Docker/NVIDIA runtime installation
- remote file layout
- systemd unit creation
- initial health checks

Deliverable:

- launched VM bootstraps Isaac services automatically

### Phase 3: Viewer Access UX

Implement:

- `viewer` subcommand in `main.py`
- local status output
- viewer URL guidance
- better health diagnostics

Deliverable:

- user can launch and open the browser viewer reliably

### Phase 4: S3 Persistence

Implement:

- bucket/prefix config
- sync include/exclude rules
- `sync pull`
- `sync push`
- automatic push on stop/destroy

Deliverable:

- user data survives instance destruction

### Phase 5: Hardening

Implement:

- retries and resume logic
- partial failure recovery
- better logging
- cost-awareness and cleanup warnings
- optional public-viewer mode with stronger security guardrails

## Suggested Initial Layout

We will implement the tool in Python with a script-first layout.

Initial layout:

- `main.py`
- `src/isaac_cloud/config.py`
- `src/isaac_cloud/tensordock/client.py`
- `src/isaac_cloud/tensordock/models.py`
- `src/isaac_cloud/providers/base.py`
- `src/isaac_cloud/providers/tensordock.py`
- `src/isaac_cloud/planner.py`
- `src/isaac_cloud/cloud_init.py`
- `src/isaac_cloud/remote/ssh.py`
- `src/isaac_cloud/lifecycle.py`
- `src/isaac_cloud/sync/s3.py`
- `src/isaac_cloud/state.py`

The important constraint is that packaging should remain optional during early development. `main.py` can import from `src/isaac_cloud/...` while we validate the product flow.

## Open Questions

1. Does TensorDock's Ubuntu GPU image already have a compatible NVIDIA driver?
2. Which exact TensorDock GPU SKUs should count as satisfying `rtx4080` compatibility?
3. Which exact Isaac/web-viewer image versions should we pin for v1?
4. Do we want remote sync to be driven by the VM or by the local CLI over SSH?
5. Should `stop` preserve the root disk, or should we always destroy and rely on S3?
6. Do we want one instance per project, or one reusable instance with per-project sync prefixes?
7. Which provider capabilities should be modeled explicitly now so we do not hard-code TensorDock assumptions into lifecycle logic?

## Recommended Next Step

The next step should be Phase 0 manual validation on a real TensorDock GPU VM.

Without that validation, the biggest unknowns are driver bootstrap behavior and the exact Isaac/web-viewer compose recipe we should automate.
