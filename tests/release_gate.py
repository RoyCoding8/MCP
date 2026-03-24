"""Release gate check for ReasonForge.

Run:
    python -m tests.release_gate
"""

from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> int:
    print("[GATE] " + " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def main() -> int:
    commands = [[sys.executable, "-m", "tests.test_all"]]

    for cmd in commands:
        code = _run(cmd)
        if code != 0:
            print(f"[GATE] FAIL (exit={code})")
            return code

    print("[GATE] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
