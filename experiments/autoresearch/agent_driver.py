#!/usr/bin/env python3
"""
Agent driver for ADN autoresearch.

Purpose:
- standardize how an external coding agent is invoked
- decouple run_agent.py from any specific vendor CLI

Usage:
  python3 experiments/autoresearch/agent_driver.py \
    --workspace <path> \
    --program <path/to/program.md> \
    --trial-id <id> \
    --trial-dir <path>

Configuration:
- Set AUTORESEARCH_AGENT_CMD_TEMPLATE to your agent command.
  Supported placeholders:
    {workspace} {program} {trial_id} {trial_dir} {prompt}

Example:
  export AUTORESEARCH_AGENT_CMD_TEMPLATE='my-agent-cli run --cwd "{workspace}" --prompt-file "{prompt}"'
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import shutil
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_prompt(program_text: str, workspace: Path, trial_id: str) -> str:
    return (
        "You are running one ADN autoresearch trial.\n\n"
        f"Trial ID: {trial_id}\n"
        f"Workspace: {workspace}\n\n"
        "Read and follow this program policy:\n"
        "----- PROGRAM START -----\n"
        f"{program_text}\n"
        "----- PROGRAM END -----\n\n"
        "Task:\n"
        "1) Propose one small hypothesis.\n"
        "2) Apply minimal code/config change in the workspace.\n"
        "3) Keep edits scoped to allowed paths.\n"
        "4) Do not run destructive git operations.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Invoke external agent for one trial.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--trial-id", type=str, required=True)
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    program = args.program.resolve()
    trial_dir = args.trial_dir.resolve()

    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    if not program.exists():
        raise FileNotFoundError(f"Program file not found: {program}")

    program_text = _read_text(program)
    prompt_text = _build_prompt(program_text, workspace, args.trial_id)
    prompt_file = trial_dir / "agent_prompt.txt"
    _write_text(prompt_file, prompt_text)

    template = os.environ.get("AUTORESEARCH_AGENT_CMD_TEMPLATE", "").strip()

    # 1) Explicit template always wins.
    if template:
        command = template.format(
            workspace=str(workspace),
            program=str(program),
            trial_id=args.trial_id,
            trial_dir=str(trial_dir),
            prompt=str(prompt_file),
        )
        print(f"Executing agent command: {command}")
        run = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )
    else:
        # 2) Default to cursor-agent if available.
        cursor_agent = shutil.which("cursor-agent")
        if cursor_agent:
            print("AUTORESEARCH_AGENT_CMD_TEMPLATE not set; using cursor-agent default.")
            run = subprocess.run(
                [
                    cursor_agent,
                    "-p",
                    "--trust",
                    "--workspace",
                    str(workspace),
                    prompt_text,
                ],
                cwd=str(workspace),
                text=True,
                capture_output=True,
                env=os.environ.copy(),
            )
            if run.returncode != 0:
                print(
                    "cursor-agent invocation failed. "
                    "If not authenticated, run: cursor-agent login"
                )
        else:
            print(
                "AUTORESEARCH_AGENT_CMD_TEMPLATE is not set and cursor-agent was not found. "
                "Running no-op agent step."
            )
            print(f"Generated prompt file: {prompt_file}")
            return 0

    # Mirror subprocess output for parent log collectors.
    if run.stdout:
        print(run.stdout, end="")
    if run.stderr:
        print(run.stderr, end="")

    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
