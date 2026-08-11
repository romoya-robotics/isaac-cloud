# Manual WireGuard Experiment

## Goal

Answer the narrow question first:

Can a manually configured WireGuard tunnel on the TensorDock VM support the Isaac browser viewer and WebRTC streaming path where the previous Tailscale approach did not?

This runbook is intentionally manual. The point is to get signal quickly before we commit to productizing WireGuard in the bootstrap and config model.

## Scope

This experiment does not try to fully harden the VM.

It is limited to:

- launching a normal VM with the current flow
- installing and configuring WireGuard by hand
- restarting Isaac and the viewer to use the WireGuard path
- temporarily restricting viewer and WebRTC ports to `wg0`
- testing from a laptop joined to the same WireGuard tunnel

## Assumptions

- the repo can already launch a working Isaac VM
- the laptop can run a WireGuard client
- we are testing the browser viewer path first
- public SSH may remain available during the experiment as break-glass access

## High-Level Flow

1. Launch a normal Isaac VM with the viewer enabled.
2. Wait until the VM is fully provisioned and stream-ready.
3. SSH to the VM over the normal public SSH path.
4. Install WireGuard on the VM.
5. Bring up `wg0` with a simple single-client configuration.
6. Confirm the VM and laptop can reach each other over WireGuard.
7. Restart Isaac so it advertises the `wg0` IP.
8. Make the viewer use the `wg0` host path.
9. Add temporary firewall rules so viewer/WebRTC ports are only reachable on `wg0`.
10. Test the viewer from the laptop over WireGuard.
11. Capture enough evidence to determine whether it really worked.

## Inputs To Prepare

Before touching the VM, have these ready:

- a WireGuard keypair for the laptop
- a WireGuard keypair for the VM, or a plan to generate the VM key on the VM itself
- a small tunnel CIDR, for example:
  - VM `10.44.0.1/24`
  - laptop `10.44.0.2/32`
- a UDP listen port, for example `51820`

## Suggested Minimal Topology

- VM public IP: existing TensorDock public IP
- WireGuard server: the VM
- VM tunnel IP: `10.44.0.1`
- laptop tunnel IP: `10.44.0.2`

Reasoning:

- the VM already has a public IP and is simpler as the fixed endpoint
- the laptop only needs to initiate the tunnel as a client
- this avoids introducing peer discovery or mesh behavior into the experiment

## Step 1: Launch The VM

Launch the VM normally with the viewer enabled.

Example:

```bash
UV_CACHE=/home/keenb/projects/gpu-orchestrator/.venv uv run python isaac_cloud.py launch --viewer
```

## Step 2: Wait For Real Readiness

Do not start the network experiment until Isaac is actually ready for streaming.

Use:

```bash
UV_CACHE=/home/keenb/projects/gpu-orchestrator/.venv uv run python isaac_cloud.py status --instance-id <INSTANCE_ID> --verbose
```

Wait until the logs indicate the stream app is actually loaded, not merely that:

- cloud-init is done
- Docker is installed
- `isaac-cloud-isaac.service` is active

## Step 3: SSH To The VM

Use the normal SSH command printed by launch.

Keep this public SSH path available during the experiment unless we later decide to close it.

## Step 4: Install WireGuard On The VM

Suggested packages:

```bash
sudo apt-get update
sudo apt-get install -y wireguard iptables
```

## Step 5: Create The VM WireGuard Config

Suggested server config shape:

```ini
[Interface]
Address = 10.44.0.1/24
ListenPort = 51820
PrivateKey = <vm-private-key>

[Peer]
PublicKey = <laptop-public-key>
AllowedIPs = 10.44.0.2/32
```

Save it as:

```bash
sudo install -m 600 /dev/null /etc/wireguard/wg0.conf
sudoedit /etc/wireguard/wg0.conf
```

Then start it:

```bash
sudo systemctl enable --now wg-quick@wg0
sudo wg show
ip addr show wg0
```

## Step 6: Configure The Laptop Peer

Suggested client config shape:

```ini
[Interface]
Address = 10.44.0.2/32
PrivateKey = <laptop-private-key>

[Peer]
PublicKey = <vm-public-key>
Endpoint = <vm-public-ip>:51820
AllowedIPs = 10.44.0.0/24
PersistentKeepalive = 25
```

Bring the client up and confirm:

- the VM and laptop can ping each other on `10.44.0.x`
- `wg show` on the VM shows a recent handshake
- transfer counters increase when traffic is sent

## Step 7: Confirm The WireGuard Path Works Before Touching Isaac

From the laptop:

- test basic connectivity to the VM WireGuard IP
- verify the tunnel stays up for more than a transient handshake

From the VM:

```bash
sudo wg show
ip route
```

If this step is flaky, stop and debug WireGuard before changing Isaac.

## Step 8: Restart Isaac To Advertise The `wg0` IP

The experiment depends on Isaac advertising the WireGuard address rather than the public NIC.

Current repo behavior auto-detects the host IP. For the manual experiment, override that behavior manually.

Practical options:

- edit the generated runtime script on the VM and replace the detected stream IP with `10.44.0.1`
- or stop the service and run an equivalent `docker run` command manually with:
  - `--/exts/omni.kit.livestream.app/primaryStream/publicIp=10.44.0.1`

After the change:

```bash
sudo systemctl restart isaac-cloud-isaac.service
sudo journalctl -u isaac-cloud-isaac.service -n 100 --no-pager
sudo tail -n 100 /var/log/isaac-cloud-isaac.log
```

We want evidence of the chosen advertised IP in the service command or logs.

## Step 9: Make The Viewer Use The WireGuard Path

The viewer needs to connect using the WireGuard host path, not the public IP.

Practical options:

- update the viewer config so the signaling host is `10.44.0.1`
- or use the existing `window.location.hostname` behavior and access the viewer at `http://10.44.0.1:8210`

If the viewer service is still bound broadly, that is acceptable for the moment, but we should still test using the WireGuard IP.

Restart the viewer if needed:

```bash
sudo systemctl restart isaac-cloud-viewer.service
sudo journalctl -u isaac-cloud-viewer.service -n 100 --no-pager
sudo tail -n 100 /var/log/isaac-cloud-viewer.log
```

## Step 10: Add Temporary Firewall Rules

Once WireGuard is up and both services are pointing at the WireGuard path, restrict the relevant ports to `wg0`.

Temporary rule shape:

- allow `8210/tcp` on `wg0`
- allow `49100/tcp` on `wg0`
- allow `47998/udp` on `wg0`
- reject those same ports on non-`wg0` interfaces

Do not apply these until:

- WireGuard is confirmed working
- public SSH access is still available as a recovery path

Before and after applying rules, record:

```bash
sudo iptables -S
sudo iptables -L -n -v
```

## Step 11: Run The Browser Test

From the laptop:

1. Open the viewer using the WireGuard IP:

```text
http://10.44.0.1:8210
```

2. Attempt a full streaming session.

3. Observe whether:

- the page loads
- signaling succeeds
- media appears
- the session remains stable

## Step 12: Capture Evidence

On the VM, record:

```bash
sudo wg show
ip addr show wg0
ip route
ss -ltnup
sudo iptables -L -n -v
sudo journalctl -u isaac-cloud-isaac.service -n 200 --no-pager
sudo tail -n 200 /var/log/isaac-cloud-isaac.log
```

In the browser, record:

- WebRTC internals if available
- any ICE candidate information
- whether the selected path appears to use the WireGuard IPs

## Success Criteria

The experiment is a success only if:

- the viewer is loaded over the WireGuard IP
- signaling succeeds
- media succeeds
- the session is usable
- public viewer/WebRTC ports can remain closed while the session still works

## Failure Interpretation

If one of these happens, treat the experiment as not yet successful:

- viewer loads but no media arrives
- Isaac still appears to advertise the public IP
- the connection only works when public ports are reopened
- WireGuard handshakes exist but the actual app traffic does not traverse `wg0`

## What To Do With The Result

If the experiment succeeds:

- codify the setup into bootstrap and config
- make WireGuard an explicit supported private access mode

If the experiment fails:

- write down the exact failure mode
- do not repeatedly tweak VPN settings without new evidence
- move to the TURN-assisted plan as the next serious networking experiment
