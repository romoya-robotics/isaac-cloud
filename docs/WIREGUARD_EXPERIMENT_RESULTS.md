# WireGuard Experiment Results

## Goal

Determine whether a manually configured WireGuard tunnel on the TensorDock VM can support the Isaac browser viewer and WebRTC streaming path where the previous Tailscale approach did not.

## Environment

- VM provider: TensorDock
- Instance: `romoya-20260401-094906`
- Instance ID: `4e72fd72-0400-4bcd-9c47-b6877007b08a`
- Public IP: `38.224.253.238`
- WireGuard tunnel:
  - VM: `10.44.0.1/24`
  - client: `10.44.0.2/32`

## What We Changed

We performed a manual VM-side WireGuard experiment instead of modifying the repo bootstrap.

Changes on the VM:

- installed `wireguard`
- created and started `wg0`
- patched Isaac to advertise `10.44.0.1`
- patched the web viewer to bind `10.44.0.1:8210`
- applied temporary firewall rules so:
  - `8210/tcp` and `49100/tcp` were only allowed on `wg0`
  - `47998/udp` was only allowed on `wg0`
  - `22/tcp` and `51820/udp` remained publicly reachable

We also ran a follow-up variant where Isaac was restarted without:

```bash
--merge-config=/isaac-sim/config/open_endpoint.toml
```

## What Worked

- the Windows client established a real WireGuard tunnel to the VM
- the VM saw repeated WireGuard handshakes and nonzero transfer counters
- the browser loaded the viewer over:
  - `http://10.44.0.1:8210`
- Isaac restarted successfully while advertising:
  - `10.44.0.1`
- the viewer restarted successfully while binding:
  - `10.44.0.1:8210`
- the livestream service reported:
  - `Status: Ready for connection`
- the public viewer endpoint was blocked as intended once firewall rules were applied

## What Failed

The browser was not able to establish a streaming session over WireGuard.

Observed browser behavior:

- stream start began normally
- repeated retry messages appeared
- connection attempts failed repeatedly
- no successful session was established

Observed browser-side diagnostics:

- `RTCPeerConnection` objects were created
- remote offers and local answers were exchanged
- connections closed quickly after description exchange
- no meaningful ICE/media progress was observed in the captured rtcstats dump

## Most Important VM-Side Finding

On the VM, the firewall counters showed:

- TCP traffic on `wg0` to `8210` and `49100`
- zero UDP traffic on `wg0` to `47998`

Final relevant counter state:

- `wg0` TCP `8210,49100`: nonzero hits
- `wg0` UDP `47998`: `0 packets / 0 bytes`

This remained true even after multiple browser connection attempts.

That means:

- the viewer path and some signaling/control traffic reached the VM over WireGuard
- the media UDP path never started, or never reached the VM

## Additional Container-Side Findings

- `open_endpoint.toml` does not appear to contain livestream networking settings
- it only contained telemetry/open-endpoint settings
- removing `open_endpoint.toml` did not solve the problem
- after restart without it, the same high-level failure remained

We also found that the livestream stack exposes a local HTTP control plane on `8011`, including:

- `/v1/streaming/ready`
- `/v1/streaming/creds`

The NVCF service code indicates support for injecting STUN credentials dynamically, which suggests the intended networking model is broader than a simple `publicIp` override.

## Conclusion

This experiment did not show that WireGuard solves the Isaac browser streaming problem.

More specifically:

- replacing Tailscale with plain WireGuard did not fix session establishment
- binding the viewer and advertised Isaac IP to the WireGuard address was not sufficient
- removing `open_endpoint.toml` was not sufficient
- the failure appears to occur before meaningful media flow begins

## What We Learned

1. The problem is not simply “SSH cannot carry UDP.”
- WireGuard carried the private network successfully.

2. The problem is not simply “Tailscale was the wrong VPN.”
- A plain WireGuard setup still failed.

3. The problem is likely inside the livestream/session model rather than basic tunnel transport.
- Signaling/control reached the VM.
- UDP media never started.

4. There are likely additional livestream assumptions or control-plane requirements beyond:
- viewer bind host
- `primaryStream/publicIp`
- direct port reachability

5. TURN/STUN-assisted approaches are now more compelling than continued direct-VPN tweaking.

## Recommendation

Do not continue investing in “direct browser streaming over a private VPN should just work” as the primary plan.

The next better path is:

1. restore the VM to baseline
2. preserve these findings
3. move to TURN/STUN-assisted experiments

## Follow-Up References

- [`MANUAL_WIREGUARD_EXPERIMENT.md`](/home/keenb/projects/gpu-orchestrator/MANUAL_WIREGUARD_EXPERIMENT.md)
- [`CONTAINER_SIDE_WIREGUARD_FOLLOWUPS.md`](/home/keenb/projects/gpu-orchestrator/CONTAINER_SIDE_WIREGUARD_FOLLOWUPS.md)
- [`SECURITY_PLAN.md`](/home/keenb/projects/gpu-orchestrator/SECURITY_PLAN.md)
