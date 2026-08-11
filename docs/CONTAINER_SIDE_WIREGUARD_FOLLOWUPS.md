# Container-Side WireGuard Follow-Ups

## Goal

Document the next experiments to run inside or around the Isaac streaming container now that the first WireGuard test has shown:

- the WireGuard tunnel itself works
- the viewer can load over the WireGuard IP
- the stream session still fails before media actually starts

The purpose of these follow-ups is to determine whether the failure is caused by Isaac livestream configuration, signaling behavior, or another stack assumption beyond basic VPN transport.

## What We Learned From The First Experiment

- The VM and Windows client established a real WireGuard tunnel.
- The viewer was successfully served from `http://10.44.0.1:8210`.
- Isaac was restarted and configured to advertise `10.44.0.1`.
- Firewall rules restricted viewer and WebRTC ports to `wg0`.
- Browser logs showed repeated stream-start retries.
- VM counters showed some TCP traffic to `8210` and `49100` on `wg0`.
- VM counters showed no UDP traffic to `47998`.
- Browser WebRTC dumps showed peer connections being created, descriptions being exchanged, and then the connections closing before meaningful ICE/media progress.

This suggests the failure is not just “WireGuard cannot carry UDP.” The failure appears earlier in session establishment.

## Hypothesis

The likely problem is in the livestream/signaling stack rather than in basic VPN transport.

Possible causes:

- `open_endpoint.toml` bakes in assumptions optimized for public exposure
- the signaling path is still handing back the wrong address or candidate data
- additional livestream settings besides `primaryStream/publicIp` need to be overridden
- the viewer stack has hidden assumptions about binding, ICE behavior, or signaling host selection

## Follow-Up Experiments

### 1. Retry Without `open_endpoint.toml`

Current startup uses:

```bash
--merge-config=/isaac-sim/config/open_endpoint.toml
```

Experiment:

- restart Isaac without merging `open_endpoint.toml`
- keep the explicit WireGuard advertised IP
- leave the rest of the setup unchanged

Why:

- this config may be forcing a public-facing endpoint model that does not behave well on a private VPN address

Success signal:

- session establishment progresses further than in the first experiment

### 2. Inspect Effective Livestream Configuration

Experiment:

- inspect the livestream-related config files and runtime arguments visible in the container
- identify settings related to:
  - bind address
  - advertised address
  - ICE
  - STUN/TURN
  - signaling endpoint selection

Why:

- we only changed `primaryStream/publicIp`
- other settings may still be inconsistent with a WireGuard-only path

### 3. Capture Signaling Payloads During A Failed Attempt

Experiment:

- capture traffic or logs around the signaling flow during a browser start attempt
- inspect what address/candidate data the server returns

Why:

- if the server is still handing back the public IP, localhost, or another invalid endpoint, that would explain the early failure

Evidence to collect:

- HTTP/WebSocket signaling payloads if visible
- Isaac logs around session start
- packet capture on the VM during the attempt

### 4. Try Explicit Signaling Bind If Supported

Experiment:

- determine whether the livestream service can bind signaling more explicitly to `10.44.0.1`
- if so, restart with signaling bound to the WireGuard IP rather than `0.0.0.0`

Why:

- today signaling listens on all interfaces and is only restricted by firewall
- if the stack makes interface assumptions internally, an explicit bind may behave differently

### 5. Capture Packets During Stream Start

Experiment:

- run packet capture on the VM during a failed browser attempt
- observe:
  - WireGuard traffic on `wg0`
  - TCP traffic to `49100`
  - any UDP traffic to `47998`
  - any STUN/ICE-related traffic

Why:

- we need to know whether the app is even attempting the media phase
- packet capture is a stronger signal than log inference

### 6. Compare Browser And Native Client Behavior

Experiment:

- test NVIDIA’s native Isaac Streaming Client over the same WireGuard tunnel

Why:

- if both browser and native client fail, the problem is likely on the Isaac/livestream side
- if native works but browser fails, the problem is more likely in the web viewer integration

### 7. Inspect Viewer-Side Hidden Overrides

Experiment:

- inspect the generated web viewer for additional StreamKit or WebRTC override hooks
- look for options related to:
  - signaling host
  - ICE transport policy
  - STUN/TURN
  - candidate filtering

Why:

- the browser-side stack may need more than just `window.location.hostname`
- these hooks may be the bridge toward a TURN-based fallback if direct private networking keeps failing

## Recommended Execution Order

1. Capture signaling and packet-level evidence from one more failed attempt.
2. Retry Isaac without `open_endpoint.toml`.
3. Inspect effective livestream config inside the container.
4. If available, try explicit signaling/interface binding to the WireGuard IP.
5. Compare with the native client if needed.
6. If none of the above changes the outcome, stop spending more time on WireGuard-only fixes.

## Exit Criteria

Continue with VPN-focused debugging only if one of these happens:

- session establishment clearly progresses further
- we identify a concrete wrong-address or wrong-candidate bug
- we find a container-side livestream setting that plausibly explains the failure

If none of those happen, treat the WireGuard result as negative and move to TURN-assisted experiments.
