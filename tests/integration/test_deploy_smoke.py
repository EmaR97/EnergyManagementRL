"""Smoke test: run deploy for ~60s, verify both bot and control loop started."""

import subprocess
import sys
import time
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_SCRIPT = PROJECT_ROOT / "energymanagementrl" / "pipeline" / "cli.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TIMEOUT = 75  # seconds (generous)


def test_deploy_bot_and_control_loop_start():
    proc = subprocess.Popen(
        [str(VENV_PYTHON), str(CLI_SCRIPT), "deploy"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    start = time.monotonic()
    output_parts = []
    errors = []

    while time.monotonic() - start < TIMEOUT and proc.poll() is None:
        try:
            line = proc.stdout.readline()
            if not line:
                continue
            output_parts.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        except (IOError, ValueError):
            break

    # Kill process
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    output = "".join(output_parts)

    # --- Assertions ---
    assert proc.returncode in (-15, 0, None), f"Process exited with code {proc.returncode}"

    assert "Starting Control loop" in output, (
        "Control loop never started"
    )
    assert "Bot commands set!" in output, (
        "Bot never set commands (could not reach Telegram API)"
    )

    state_matches = re.findall(r"State: \{[^}]+\}", output)
    assert len(state_matches) >= 2, (
        f"Control loop completed fewer than 2 cycles (found {len(state_matches)})"
    )

    assert "RuntimeError" not in output, (
        "Found RuntimeError in output"
    )
