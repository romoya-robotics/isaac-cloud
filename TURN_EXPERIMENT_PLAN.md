# TURN Experiment Plan

## Goal

Prove whether a real TURN relay fixes Isaac/WebRTC when the control plane stays on Tailscale and direct media over the tailnet does not come up.

## First Experiment Shape

- Run `coturn` in Docker on the same VM.
- Expose TURN on the VM public IP.
- Keep viewer UI, SSH, and MCP on Tailscale.
- Patch the webviewer to force TURN relay using the hidden StreamKit override path:
  - `turn=turn:<public-ip>:3478,<username>,<password>`
  - `icetransportpolicy=relay`

## Why Same VM First

- Fastest validation path.
- No extra machine to provision.
- If it fails here, we learn quickly whether TURN is even a viable direction.

## Docker Deployment

- Use the `coturn/coturn` container.
- Publish:
  - `3478/tcp`
  - `3478/udp`
- Also publish a relay UDP range, for example:
  - `49152-49232/udp`
- Configure:
  - `realm`
  - static username/password for the test
  - `external-ip=<public-ip>`
  - listening on `0.0.0.0`

## Firewall Changes

- Keep current Tailscale-only rules for viewer and Isaac signaling unless testing requires otherwise.
- Add public allow rules for TURN:
  - `3478/tcp`
  - `3478/udp`
  - chosen relay UDP range
- Prefer source-IP restriction if practical for the experiment.

## Viewer Patch

- Use `configureStreamKitSettings({ overrideData: ... })` to inject TURN settings.
- First try:
  - Tailscale signaling
  - TURN relay for media
- If that fails:
  - test public signaling plus TURN relay

## Verification

- On the VM, watch:
  - TURN container logs
  - TURN port counters
  - Isaac signaling counters
- In browser, inspect WebRTC internals and confirm the selected candidate pair is `relay`.
- Success condition:
  - session starts
  - media flows via TURN instead of direct UDP to Isaac

## Risks

- Same-VM TURN is suitable for proof, not necessarily for production.
- The native Isaac Streaming Client may be harder to steer than the browser viewer.
- TURN may only help the browser path if the native client ignores the same ICE overrides.

## If It Works

- Move TURN to a separate hardened service or VM.
- Add repo support for:
  - TURN enable/disable
  - credentials
  - relay port range
  - firewall automation
