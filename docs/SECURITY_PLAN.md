# Security Plan

## Goal

Improve the security posture of `isaac-cloud` without breaking the Isaac Sim access path.

The immediate priority is to test whether a VM-hosted WireGuard setup works for Isaac/WebRTC where the previous Tailscale approach did not. In parallel, we should fix the concrete security weaknesses in the current bootstrap and exposure model.

## Current State

The current implementation is optimized for bring-up, not secure exposure.

Today the VM bootstrap:

- serves the web viewer over plain HTTP
- exposes viewer and Isaac streaming on the VM public IP
- does not apply host firewall rules to limit viewer or WebRTC ports
- uses `vite preview` as the public viewer server
- runs bootstrap scripts with shell tracing enabled, which risks secret leakage
- makes some persistence paths broader in permission than necessary

Relevant current ports:

- `8210/tcp` for the web viewer UI
- `49100/tcp` for Isaac WebRTC signaling
- `47998/udp` for Isaac WebRTC media
- `22/tcp` for SSH
- `8766/tcp` for MCP when enabled

## Prior Evidence

We already tried a Tailscale-based private access mode and later removed it.

What that tells us:

- SSH tunneling is not a complete answer because it does not carry Isaac UDP media.
- A private overlay model is still attractive if it works.
- We should not assume that Tailscale failing means WireGuard will also fail.
- We also should not assume that WireGuard will work automatically just because it carries UDP.

The WireGuard experiment needs to be treated as a real validation effort, not as a foregone conclusion.

## Security Objectives

1. Reduce public attack surface by default.
2. Avoid exposing unauthenticated Isaac streaming endpoints broadly.
3. Prevent secret leakage during bootstrap and status collection.
4. Make network exposure an explicit mode, not an accidental consequence of launch.
5. Produce a path that is reliable enough to support actual operator workflows, not just theoretical hardening.

## Non-Goals

- Do not redesign the whole product around Kubernetes, OVAS, or a managed API gateway in this phase.
- Do not block current bring-up workflows on a perfect final architecture.
- Do not assume TURN is the first thing to implement. TURN remains a likely fallback if direct private networking still fails.

## Recommended Access Modes

We should formalize access as explicit modes rather than one implicit public model.

### Mode 1: Public Restricted

Keep the current direct Isaac/WebRTC model, but harden it:

- only open the minimum required ports
- require source-IP or CIDR allowlisting
- keep SSH and MCP on a narrower control-plane path where possible
- serve the viewer behind a proper HTTP server and TLS if publicly reachable

This is the practical fallback if private networking experiments fail.

### Mode 2: Private VPN

Use a VPN path for the viewer and Isaac streaming:

- the VM runs a VPN endpoint
- the laptop joins the VPN
- the viewer and Isaac ports are only reachable over the VPN interface
- Isaac advertises the VPN IP, not the public NIC address

This is the first experimental focus, with WireGuard rather than Tailscale.

### Mode 3: TURN-Assisted

If direct media still fails over private networking, move to TURN-backed relay:

- keep control-plane access private where possible
- relay media through TURN
- treat this as the next experiment if WireGuard does not solve the issue

## Why Start With WireGuard

WireGuard is worth testing because it gives us a cleaner topology than the previous Tailscale attempt:

- the VM can be the fixed VPN server with a stable public endpoint
- the laptop can be a simple client
- we can explicitly control peer addressing, allowed IPs, keepalives, and firewall rules
- there is less ambiguity around relays, connection type, and interface selection

This is not a guarantee that Isaac/WebRTC will work over WireGuard. It is a higher-signal experiment than retrying the prior Tailscale path without tighter controls.

## Hypothesis

WireGuard may succeed where Tailscale did not if the main failure mode was one of these:

- Isaac advertised the wrong IP address
- traffic did not consistently stay on the intended private interface
- Tailscale fell back to a relayed path unsuitable for this workload
- ACL or interface-bound firewall behavior was wrong or insufficiently observable

WireGuard is less likely to help if the failure was caused by:

- Isaac or the viewer requiring a routability property not preserved by the VPN overlay
- browser or native client ICE behavior that still rejects the resulting candidate path
- MTU or fragmentation issues that remain unaddressed

## WireGuard Experiment Plan

### Experiment Goal

Validate whether the browser viewer and Isaac WebRTC streaming work end to end when all relevant traffic uses a WireGuard tunnel.

### Topology

- the TensorDock VM runs `wg0`
- the VM remains reachable on its public IP for initial bootstrap and WireGuard handshake
- the laptop connects as a WireGuard peer
- the viewer UI and Isaac streaming ports are reachable only over `wg0`
- SSH may remain available on the public IP for break-glass admin access unless later restricted

### Required Behavior

When WireGuard mode is enabled:

- Isaac must advertise the `wg0` IP as its stream address
- the viewer should connect to the `wg0` host/IP, not the public IP
- host firewall rules must allow viewer and Isaac ports on `wg0`
- host firewall rules must deny those same ports on non-`wg0` interfaces

### First Implementation Shape

1. Add a `[wireguard]` config section.

Suggested fields:

- `enabled = false`
- `server_port = 51820`
- `server_cidr = "10.44.0.1/24"`
- `client_cidr = "10.44.0.2/32"`
- `client_public_key = "..."`
- `server_private_key = "..."` or a local path/secret source
- `persistent_keepalive = 25`

2. Install WireGuard during bootstrap when enabled.

3. Write a `wg0.conf` on the VM and enable `wg-quick@wg0`.

4. Add a small status helper to print:

- `wg show`
- `ip addr show wg0`
- `ip route`
- packet counters for the Isaac ports

5. Change Isaac runtime selection so the advertised IP is explicitly the `wg0` address in WireGuard mode.

6. Bind or present the viewer on the WireGuard path rather than the public path.

7. Apply interface-scoped firewall rules:

- allow `8210/tcp`, `49100/tcp`, and `47998/udp` on `wg0`
- reject those same ports on non-`wg0` interfaces

### Observability Requirements

The experiment will be hard to judge unless we capture evidence. We should collect:

- WireGuard peer handshake state and transfer counters
- `ss -ltnup` output on the VM
- `iptables` or `nft` counters for accepted and rejected viewer/WebRTC traffic
- Isaac logs showing the advertised IP and stream startup
- browser WebRTC internals if available

### Success Criteria

The experiment succeeds only if all of these are true:

- the browser loads the viewer over the WireGuard path
- signaling succeeds on the WireGuard path
- media flows successfully over the WireGuard path
- the session is stable enough for real use, not just a partial connect
- the corresponding public interface rules remain closed for viewer/WebRTC ports

### Failure Criteria

Treat the experiment as failed if any of these occur:

- the viewer loads but media never flows
- Isaac still appears to advertise the public IP
- traffic counters show the browser is not using `wg0`
- the connection only works when public WebRTC ports are opened
- performance or instability makes the result impractical

### If WireGuard Fails

If the experiment fails, record the exact failure mode and move to TURN-assisted testing rather than repeatedly retrying VPN variants without new evidence.

## Hardening Work We Should Do Regardless

These are concrete security fixes that should land whether or not WireGuard works.

### 1. Stop Leaking Secrets In Bootstrap

Current risk areas:

- shell tracing in bootstrap and runtime scripts
- inline secret material in generated scripts

Planned fixes:

- remove `set -x` from bootstrap and service scripts
- disable tracing around secret-handling blocks if tracing must remain elsewhere
- avoid embedding the NGC API key directly in generated script bodies where possible
- keep secret-bearing files `0600` and delete temporary files after use

### 2. Make Exposure Mode Explicit

Add an explicit config model for network exposure, for example:

- `exposure_mode = "public_restricted" | "wireguard_experiment"`

This avoids accidental public exposure when the operator intended private access.

### 3. Add Host Firewall Automation

Bootstrap should configure host firewall policy, not rely on provider defaults.

For the near term:

- default deny for viewer and Isaac streaming ports
- open only what the selected mode requires
- support explicit CIDR allowlists for `public_restricted`
- support interface-scoped allow rules for `wireguard_experiment`

### 4. Replace Public `vite preview`

If the viewer is publicly reachable, `vite preview` is not the right serving model.

Planned direction:

- keep it for local or private experiments only if needed
- for public exposure, build static assets and serve them via Caddy or NGINX
- add TLS and optional HTTP auth on the viewer UI

This does not secure the Isaac streaming ports by itself, but it closes an avoidable weakness on the viewer UI.

### 5. Tighten File Permissions

Planned fixes:

- remove world-writable persistence directory permissions
- keep persistence owned by the app user or a dedicated group
- use owner/group-scoped permissions for state, logs, and config

### 6. Pin External Dependencies More Strictly

Planned fixes:

- pin MCP checkout to a specific commit or release
- pin `@nvidia/create-ov-web-rtc-app` to an exact version
- prefer reproducible Node installs when possible

This is partly supply-chain hygiene and partly reliability.

## Proposed Execution Order

1. Write and agree on the WireGuard experiment shape.
2. Add config plumbing for `wireguard`.
3. Implement VM-side WireGuard bootstrap and status helpers.
4. Make Isaac advertise the WireGuard IP explicitly in experiment mode.
5. Add interface-scoped firewall rules for WireGuard mode.
6. Run a real experiment and capture evidence.
7. Based on the result:
- if successful, decide whether WireGuard becomes the preferred private access mode
- if unsuccessful, move to TURN-assisted testing
8. In parallel or immediately after, land the bootstrap secret-handling and permission fixes.
9. Add `public_restricted` hardening for users who need direct public access.

## Deliverables

- `wireguard` config and launch support
- VM bootstrap support for WireGuard
- improved status output for VPN and port observability
- hardened bootstrap secret handling
- firewall automation
- a documented fallback plan if WireGuard does not solve the WebRTC path

## Open Questions

- Should the VM continue exposing public SSH in WireGuard mode for break-glass access, or should that also become allowlisted?
- Should the first WireGuard implementation support only one client peer, or a small static peer list?
- Do we want the browser path only in the first experiment, or do we also want to test the native Isaac Streaming Client?
- Should the first public hardening pass use `iptables`, `nftables`, or `ufw`?

## Recommendation

Proceed with a WireGuard-first experiment, but keep the scope narrow and evidence-driven.

The right next step is not to reintroduce a generic VPN mode. The right next step is to build a minimal, explicit WireGuard path that:

- pins Isaac to the VPN IP
- restricts viewer/WebRTC ports to the VPN interface
- captures enough evidence to tell us why it works or fails

At the same time, we should fix the security issues in the current bootstrap that are independent of network design.

## WireGuard Experiment Checklist

1. Add minimal config plumbing for a single-client WireGuard experiment.
- Add a `[wireguard]` section to `config.example.toml`.
- Add fields to `AppConfig` for enablement, server port, server address, client address, client public key, server private key source, and keepalive.
- Keep this first pass single-peer only to reduce ambiguity during testing.

2. Add explicit launch-time validation for WireGuard inputs.
- Fail before provisioning if WireGuard is enabled but required keys or CIDRs are missing.
- Treat WireGuard secrets with the same handling level as `NGC_API_KEY`.

3. Install and configure WireGuard on the VM during bootstrap.
- Install `wireguard` and required networking tools.
- Write a root-only `wg0.conf`.
- Enable and start `wg-quick@wg0`.
- Enable IPv4 forwarding only if the experiment design actually requires routed access beyond the VM itself.

4. Add a VM-side status helper for WireGuard observability.
- Print `wg show`.
- Print `ip addr show wg0`.
- Print `ip route`.
- Print `ss -ltnup`.
- Include this output in `status --verbose`.

5. Make Isaac advertise the WireGuard IP explicitly.
- Stop relying on generic host IP auto-detection in WireGuard mode.
- Resolve the `wg0` IPv4 and pass that into Isaac as the advertised stream IP.
- Surface that chosen IP in logs so we can verify what Isaac was told to use.

6. Make the viewer use the WireGuard path.
- Bind the viewer service to the WireGuard IP or otherwise ensure the browser reaches it over WireGuard.
- Ensure the viewer points signaling at the WireGuard host/IP rather than the public IP.

7. Add interface-scoped firewall rules for experiment mode.
- Allow `8210/tcp`, `49100/tcp`, and `47998/udp` on `wg0`.
- Reject those same ports on non-`wg0` interfaces.
- Decide whether `8766/tcp` for MCP should also be WireGuard-only in the experiment.

8. Keep a break-glass admin path.
- Decide whether public SSH remains open during the experiment.
- If it remains open, treat it as temporary and clearly separate it from the WireGuard data path.

9. Create the local client config needed to join the tunnel.
- Generate or accept a client peer config for the laptop.
- Document the exact local steps needed to bring the tunnel up.
- Keep the experiment deterministic by avoiding multi-client complexity.

10. Run a real end-to-end browser test.
- Load the viewer over the WireGuard path.
- Attempt a full streaming session.
- Confirm whether signaling and media both succeed.

11. Capture evidence from the test.
- Record WireGuard handshake and byte counters.
- Record VM firewall counters for viewer and Isaac ports.
- Record Isaac logs around stream startup.
- Record browser WebRTC diagnostics if available.

12. Decide based on evidence, not partial success.
- If viewer UI works but media does not, treat the experiment as not yet successful.
- If media only works when public ports are reopened, treat WireGuard as not solving the core problem.
- If the full session works over `wg0`, then promote WireGuard into a supported private access mode.
