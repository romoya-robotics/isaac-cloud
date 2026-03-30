# Tailscale Support Plan

## Why This Fits

The current launch flow assumes two access models:

- viewer: public IP plus public TCP/UDP ports
- MCP and admin access: direct remote endpoint, with SSH available separately for admin access

Tailscale gives us a third model that is safer and simpler for operator access:

- the VM joins a tailnet during bootstrap
- operators connect to the VM over its Tailscale IP or MagicDNS name
- viewer, MCP, and SSH can all use the private tailnet path instead of broad public exposure

This is a good match for the current codebase because launch is already driven by a single cloud-init bootstrap path in [isaac_cloud.py](/home/keenb/projects/gpu-orchestrator/isaac_cloud.py), and the access details are printed centrally after launch.

## Design Goals

- Make Tailscale optional per launch and configurable by default.
- Keep the existing public-IP flow working unchanged when Tailscale is disabled.
- Prefer Tailscale as an access path for SSH and MCP immediately.
- Support viewer access over Tailscale without requiring public WebRTC exposure.
- Avoid forcing TensorDock-specific network rules into the Tailscale path when they are unnecessary.

## Recommended v1 Scope

Implement Tailscale in two phases.

### Phase 1

- install and start Tailscale on the VM during bootstrap
- join the VM to a tailnet with a supplied auth key
- detect and print the Tailscale IPv4 and MagicDNS name
- prefer Tailscale endpoints in printed SSH and MCP commands when enabled
- keep Isaac Sim and the optional web viewer running as they do today

This phase already improves security and operator ergonomics for SSH and MCP.

### Phase 2

- route viewer access over Tailscale by advertising the Tailscale IP to Isaac streaming instead of the public IP when Tailscale is enabled
- update viewer output to print a tailnet URL and note that the browser must be on the same tailnet
- optionally relax the current dedicated-public-IP requirement during candidate selection when Tailscale-only access is requested

This is the step that fully changes the viewer access model from public internet to private tailnet.

## Configuration Model

Add a new `[tailscale]` section to `config.toml`.

```toml
[tailscale]
enabled = false
auth_key = "tskey-..."
hostname = ""
ssh = true
accept_routes = false
accept_dns = true
advertise_tags = []
```

Add matching environment-variable overrides:

- `ISAAC_CLOUD_TAILSCALE_ENABLED`
- `TAILSCALE_AUTH_KEY`
- `ISAAC_CLOUD_TAILSCALE_HOSTNAME`
- `ISAAC_CLOUD_TAILSCALE_SSH`
- `ISAAC_CLOUD_TAILSCALE_ACCEPT_ROUTES`
- `ISAAC_CLOUD_TAILSCALE_ACCEPT_DNS`
- `ISAAC_CLOUD_TAILSCALE_ADVERTISE_TAGS`

Notes:

- `TAILSCALE_AUTH_KEY` should be treated like a secret, same class as `NGC_API_KEY`.
- `hostname` should default to the generated instance name when unset.
- `ssh = true` means pass `--ssh` to `tailscale up`, so Tailscale SSH is available if tailnet policy permits it.
- `advertise_tags` is useful if these VMs should be governed by tagged ACLs instead of user-bound node keys.

## CLI Surface

Add launch flags:

- `--tailscale/--no-tailscale`
- `--tailscale-hostname <name>`

Do not add auth-key CLI flags. Keep secrets in config or environment variables.

Later, if needed:

- `--tailscale-ssh/--no-tailscale-ssh`
- `--tailscale-only`

`--tailscale-only` should be deferred until we actually stop depending on the public-IP path for readiness checks and output.

## Data Model Changes

Extend `AppConfig` with a nested or flattened Tailscale config:

- `tailscale_enabled: bool`
- `tailscale_auth_key: str | None`
- `tailscale_hostname: str | None`
- `tailscale_ssh: bool`
- `tailscale_accept_routes: bool`
- `tailscale_accept_dns: bool`
- `tailscale_advertise_tags: list[str]`

Extend `InstanceNetwork` with optional Tailscale fields:

- `tailscale_ipv4: str | None`
- `tailscale_name: str | None`

These should not come from the TensorDock API. They should come from VM-side inspection over SSH after the instance becomes reachable, or from a later remote status probe.

## Bootstrap Changes

Update `build_bootstrap_script()` to optionally install and configure Tailscale.

Recommended bootstrap flow:

1. Install Tailscale after basic package setup and before service start.
2. Enable and start `tailscaled`.
3. Run `tailscale up` with the configured auth key and options.
4. Persist a small helper script that prints Tailscale status as JSON for later inspection.

Example shape:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
systemctl enable --now tailscaled
tailscale up \
  --auth-key="$TAILSCALE_AUTH_KEY" \
  --hostname="$TAILSCALE_HOSTNAME" \
  --accept-routes=false \
  --accept-dns=true \
  --ssh
```

If tags are configured, include `--advertise-tags=tag:foo,tag:bar`.

## Runtime Networking Design

### SSH

When Tailscale is enabled and the VM has a Tailscale IP:

- print the preferred SSH target as the Tailscale IP or MagicDNS name
- still print the public SSH command as a fallback if present

We should keep existing TCP-based SSH readiness checks for v1, because they are simple and depend only on TensorDock-reported networking. Tailscale readiness can be a second-stage enhancement.

### MCP

MCP should follow the same access mode as the rest of the VM.

Public mode:

- local MCP server connects directly to `<public-ip>:8766`

Tailscale mode:

- local MCP server connects directly to `<tailscale-ip>:8766` or `<magicdns-name>:8766`

SSH tunneling can still be used manually when an operator wants it, but it should not be the normal CLI workflow.

### Viewer

Viewer is the sensitive part because the Isaac container currently injects `PUBLIC_IP` into:

- `--/exts/omni.kit.livestream.app/primaryStream/publicIp=$PUBLIC_IP`

That value is produced by `detect_primary_ipv4()` in `build_isaac_runtime_script()`, which currently chooses the main non-loopback host IP. On a Tailscale-enabled VM, that logic may still pick the public NIC instead of `tailscale0`.

For viewer-over-tailnet support we should:

1. Add a helper that prefers the `tailscale0` IPv4 when Tailscale is enabled.
2. Fall back to the existing primary-IP logic when Tailscale is unavailable.
3. Print tailnet viewer instructions instead of public-port instructions when the selected stream IP is a Tailscale IP.

This is the critical implementation detail for Phase 2. Without it, the browser may load the viewer UI over Tailscale but fail to complete the WebRTC path because Isaac advertises the wrong endpoint.

## Candidate Selection Impact

Today `filter_candidates()` drops any offer without a dedicated public IP.

That made sense for the original public-viewer design. It is too strict for a future Tailscale-first mode.

Recommendation:

- keep current dedicated-IP filtering unchanged in the first Tailscale pass
- once viewer-over-tailnet is validated, make dedicated IP conditional instead of mandatory

Proposed later rule:

- if viewer public exposure is needed, require dedicated IP
- if access is Tailscale-only, allow candidates without dedicated IP as long as SSH/bootstrap can still complete

This should be a separate step because it changes provisioning assumptions and may expose TensorDock networking edge cases.

## Status And Output Changes

Add a remote status probe command, for example:

```bash
tailscale status --json
```

Use it from verbose `status` once SSH is reachable. Parse and print:

- Tailscale backend state
- self hostname
- Tailscale IPv4
- MagicDNS name

Launch output should become:

- `SSH (Tailscale): ssh user@<magicdns-or-ip>`
- `SSH (Public Fallback): ssh -i ... user@<public-ip>`
- `MCP Endpoint: <preferred-host>:8766`
- `Viewer URL (Tailscale): http://<magicdns-or-ip>:8210`

Keep fallback output when Tailscale is not available or not yet up.

## Failure Handling

Recommended behavior:

- if `tailscale.enabled = true` but no auth key is configured, fail launch before create
- if Tailscale install or `tailscale up` fails during bootstrap, fail the bootstrap and surface that in logs
- do not silently continue with a half-configured node; that would produce confusing access output

This should be strict because networking is a primary reason to enable Tailscale in the first place.

## Security Posture

- Keep the auth key out of CLI arguments and documentation examples that encourage shell history leakage.
- Prefer tagged auth keys for shared infrastructure.
- Treat public viewer exposure and Tailscale access as distinct modes in docs so operators understand which ports are internet-reachable.
- Long term, Tailscale should let us reduce public attack surface by avoiding broad viewer and MCP exposure.

## Implementation Plan

1. Config plumbing
   Add Tailscale fields to `AppConfig`, `load_app_config()`, and `config.example.toml`.
2. Launch plumbing
   Add `--tailscale/--no-tailscale` and `--tailscale-hostname` to `launch`, then thread those into the effective config.
3. Bootstrap install
   Update `build_bootstrap_script()` to install Tailscale, start `tailscaled`, and run `tailscale up` when enabled.
4. Remote inspection
   Add a helper to fetch and parse `tailscale status --json` over SSH after launch and during `status --verbose`.
5. Access output
   Update SSH, MCP, and instance summary printing to prefer Tailscale endpoints when present.
6. Viewer-over-tailnet
   Change the Isaac runtime script to prefer `tailscale0` for the advertised stream IP when Tailscale is enabled.
7. Docs
   Update `README.md` with config, launch examples, access examples, and the new public-vs-tailnet access model.
8. Candidate policy follow-up
   Revisit the dedicated-IP requirement after validating viewer access over Tailscale.

## Acceptance Criteria

- `launch --tailscale` fails fast if no auth key is configured.
- A launched VM joins the expected tailnet during bootstrap.
- `status --verbose` shows Tailscale identity and IP details.
- SSH and MCP instructions prefer Tailscale endpoints when available.
- Viewer works from a client on the same tailnet without needing public WebRTC exposure.
- Existing non-Tailscale launches behave exactly as they do today.

## Main Risks

- Isaac streaming may advertise the wrong IP unless we explicitly prefer `tailscale0`.
- Tailscale SSH availability depends on tailnet policy, so raw TCP SSH should remain as a fallback until policy is standardized.
- If TensorDock instances without dedicated IP behave differently for bootstrap reachability, relaxing the dedicated-IP filter may need additional work.

## Recommended First Cut

The best first implementation is:

- add Tailscale bootstrap and config
- prefer Tailscale for SSH and MCP
- leave candidate selection unchanged
- treat viewer-over-tailnet as the second implementation step, not a same-PR stretch goal

That keeps the first change set small and useful while isolating the only tricky part, which is Isaac WebRTC endpoint advertisement.
