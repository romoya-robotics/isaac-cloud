# Vast.ai Provisioning Timeout Experiment Results

Date: 2026-08-11
Question: the CLI waited up to 20 min (`launch --timeout-seconds`, default
1200) before giving up on a Vast instance, with no early-exit. When a
provision is doomed, how early can we detect it, and what does a doomed
container look like?
Method: 7 instances across distinct whole-machine RTX 4090 hosts, polling
`vastai show instance --raw` every 10 s with TCP + SSH-auth probes, logging
every snapshot to JSONL. Arms: 3x healthy Isaac 6.0.1 pulls, 1x tiny image
(`ubuntu:22.04`, schedule/boot floor), 1x nonexistent image tag, 1x bad
registry password, 1x re-rental of a misbehaving host to test recovery.
Total cost: ~$0.60.

## Headline numbers

| Arm | Outcome | Time |
| --- | --- | --- |
| tiny (ubuntu:22.04) | SSH-ready | 89 s |
| isaac-a (healthy pull, BR) | SSH-ready | 255 s |
| isaac-b (healthy pull, AU) | `running` at 524 s; pubkey auth denied forever | never |
| isaac-c (healthy pull, AU) | `running` at 576 s; pubkey auth denied forever | never |
| bad registry password | `status_msg: docker login failed!`, parked in `loading` forever | detected at **+30 s** |
| nonexistent image tag | `status_msg: ...manifest unknown`, parked in `loading` forever | detected at **+50 s** |

- Healthy time-to-SSH: 1.5–10 min (the pull dominates; `running` + port
  mapping + SSH all land in the same poll).
- Every observed failure mode is detectable in **under 2 minutes** — or, for
  the SSH-key mode, within ~5 min of `running` — far inside the 20-min
  timeout.

## Failure modes and their symptoms

### 1. Registry auth failure (`docker login failed!`)

`status_msg` reads exactly `docker login failed!` ~30 s after create
(first poll after the host picks up the job). `actual_status` stays
`loading` forever; Vast does not retry, nothing self-heals. This is what a
bad/expired NGC key looks like.

### 2. Missing image/tag (`manifest unknown`)

`status_msg` carries docker's own error verbatim:
`Error response from daemon: manifest for nvcr.io/nvidia/isaac-sim:X not
found: manifest unknown: manifest unknown` at ~50 s. Vast retried the pull
once (~+110 s), got the same error, then parked in `loading` forever.

### 3. Host fails to inject the account SSH key (intermittent)

Two of three healthy pulls (both created simultaneously in the same AU
datacenter, one shared public IP) reached `running` with the port mapped and
TCP connecting, but pubkey auth returned `Permission denied (publickey)` for
41 straight probes over 16 minutes. The same account key worked on the other
hosts, and a later solo re-rental of the same machine authed fine in 57 s —
so this is intermittent host-side key-injection flakiness (possibly raced by
concurrent creates), not a machine property. It never self-heals.

### What a *healthy* pull looks like (why a naive stall detector fails)

During a healthy pull `status_msg` streams docker layer progress
(`<layer>: Verifying Checksum / Download complete`, then image build/apt
output, then `success, running nvcr.io/nvidia/isaac-sim_6.0.1/ssh`). But it
can legitimately go **6 minutes** between updates while a large layer
downloads (observed max gap: 360 s on a healthy instance). A
"no-progress-for-3-min" bail-out would kill healthy launches; only error
*content*, not message *staleness*, is a safe early signal.

## Changes made to `isaac_cloud.py`

`wait_for_ssh` now fails fast instead of riding out the timeout:

1. **Fatal `status_msg` patterns** (`docker login failed`, `manifest
   unknown`, `pull access denied`, `unauthorized`, `no space left on
   device`) raise immediately — these never self-heal. Detection: ~30–50 s.
2. **Generic `Error response from daemon`** raises only after persisting
   90 s (Vast retries some pulls; transient daemon hiccups get a chance).
3. **`running` + persistent `Permission denied`**: after 45 s of denials the
   key is re-attached once via `vastai attach ssh` (cheap, might rescue the
   race); if denial persists 5 min total, raise. TCP-level failures don't
   count — only auth denials.
4. Doomed failures raise `ProvisioningDoomed`; `launch` catches it and
   **auto-destroys the instance** (it can never become usable and only
   accrues cost). Plain timeouts still leave the instance alive, since a
   slow pull may yet finish — but `launch` now prints that the instance is
   still billing and the exact destroy command on every failure path.
   `resume` never auto-destroys.
5. The timeout error message now includes the last `status_msg` for
   debugging.

The 1200 s default cap is retained as a backstop: slowest healthy
time-to-`running` observed was 576 s, and pull time scales with host
`inet_down` (query floor: 300 Mb/s), so 2x headroom is deliberate. With the
fail-fast rules, the cap should now only be reached by genuinely slow-but-
alive pulls.

## Validation

- Unit-level: fake-provider tests exercised all four branches (immediate
  doom, 90 s persistence rule, healthy-pull messages not misclassified,
  attach-once-then-give-up at ~5 min).
- End-to-end: `ISAAC_CLOUD_ISAAC_VERSION=0.0.0-doesnotexist launch
  --provider vast` on a real rental bailed out and auto-destroyed the
  instance ~1 min after create (vs 20 min before).

## Raw data

JSONL snapshot logs (one `show instance` snapshot per line, plus TCP/SSH
probe results) in the session scratchpad under `run1/`; harness:
`vast_provision_probe.py`, recovery probe: `ssh_recovery_probe.py`.
