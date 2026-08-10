## Status Progress Plan

1. Define bootstrap milestones in one place
- Add a small ordered milestone model in `isaac_cloud.py`.
- Include phases like `ssh reachable`, `apt base packages`, `nvidia driver install`, `dkms build`, `docker installed`, `nvidia container toolkit configured`, `isaac image pulled`, `isaac service started`, `container running`, `rtx compile in progress`, `rtx ready`, `stream app loaded`.
- Remove old Tailscale/Xvfb-specific milestones from the core plan now that those paths are no longer part of the supported flow.

2. Parse remote state into a structured progress summary
- Teach `status` to derive a current phase from:
- `cloud-init status`
- bootstrap log patterns
- service states
- docker/container state
- Distinguish `progressing`, `waiting`, `ready`, and `failed/stalled`.
- Treat `Isaac Sim Full Streaming App is loaded.` as the real stream-ready signal.
- Treat `omni.services.livestream.nvcf ... rtx_ready for streaming` as a near-final milestone.
- Surface the long `Waiting for RtPso async group async compilation` phase explicitly instead of implying the VM is ready once the container is up.

3. Improve default `status`
- Make plain `status` show a compact progress section, not just instance metadata.
- Example output:
- `Bootstrap: in progress`
- `Current Phase: Building NVIDIA DKMS module`
- `Milestones: 4/10 complete`
- `Isaac: compiling RTX shaders`
- `Ready For Streaming: no`

4. Add explicit readiness guidance
- Show that `isaac-cloud-isaac.service` being `active` is not sufficient.
- Show the difference between:
- `service started`
- `container running`
- `stream app loaded`
- Give the user a direct sentence like `Streaming is not ready yet; wait for "Isaac Sim Full Streaming App is loaded."`

5. Improve `status --verbose`
- Keep the raw sections, but prepend a human summary:
- `Last completed milestone`
- `Current phase`
- `Next expected step`
- `Possible blocker` if logs stop advancing or a service fails
- Include the latest meaningful Isaac readiness line when present.

6. Add stall/failure heuristics
- If `cloud-init` is running but the bootstrap log has not advanced for N minutes, mark as `possibly stalled`.
- If known error patterns appear, surface them directly instead of burying them in log tails.
- Consider `SSH reachable but docker missing` and `cloud-init done but service inactive` as distinct failure modes.

7. Keep raw evidence below the summary
- Do not remove the current detailed log/service output from `--verbose`.
- Just make the top of the command answer the real user question first: “is it advancing?”

8. Verify against real boot states
- Test at least these cases:
- early boot, before Docker
- DKMS build in progress
- post-Docker, pre-Isaac
- Isaac healthy
- Isaac service active but still compiling RTX shaders
- Isaac fully loaded and confirmed stream-ready
- known failure case like missing driver or livestream startup error

9. Lessons from the A/B test
- A VM can report `cloud-init: done`, `docker up`, and `isaac-cloud-isaac.service active` while streaming is still unusable.
- Both the pre-Tailscale baseline and the current code path eventually worked once the app reached `Isaac Sim Full Streaming App is loaded.`
- The status UX needs to prevent premature testing by making that final readiness milestone obvious.
