"""One-command unit test runner.

Run:
    python -m tests.test_all
"""

from __future__ import annotations

import unittest


SUITES = [
    "tests.test_endpoint_policy",
    "tests.test_ui_security_static",
    "tests.test_code_tool_safety",
]


def main() -> int:
    print("[TEST_ALL] Running suites:")
    for name in SUITES:
        print(f"[TEST_ALL] - {name}")

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromName(name) for name in SUITES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    print(
        f"[TEST_ALL] result ok={result.wasSuccessful()} "
        f"run={result.testsRun} fail={len(result.failures)} err={len(result.errors)}"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
