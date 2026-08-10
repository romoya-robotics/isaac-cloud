# AGENTS.md

Repository execution conventions for Codex and other agents.

## Command runner

- Use `uv` for Python commands in this repository.
- Prefer `uv run ...` instead of invoking `python`, `python3`, or `pip` directly.
- If a command needs dependencies from this project, run it through `uv`.

## UV cache

- Set `UV_CACHE=/home/keenb/projects/gpu-orchestrator/.venv` when running repository commands.
- Preferred command form:

```bash
UV_CACHE=/home/keenb/projects/gpu-orchestrator/.venv uv run <command>
```

## Examples

```bash
UV_CACHE=/home/keenb/projects/gpu-orchestrator/.venv uv run python -m py_compile isaac_cloud.py
UV_CACHE=/home/keenb/projects/gpu-orchestrator/.venv uv run python isaac_cloud.py launch --help
```

## Notes

- Do not assume `python` or `python3` has the required dependencies outside `uv run`.
- When verifying CLI behavior, prefer `uv run python isaac_cloud.py ...`.
