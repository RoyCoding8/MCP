import unittest
from unittest.mock import patch
import os

from experts.code.tools.code import code_tool


class CodeToolSafetyTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(
            os.environ,
            {"RF_CODE_TOOL_ISOLATION": "process"},
            clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_run_basic_code(self):
        result = code_tool("print(2 + 2)", operation="run")
        self.assertEqual(result.get("stdout"), "4\n")
        self.assertEqual(result.get("exit_code"), 0)

    def test_blocks_open_builtin(self):
        result = code_tool("open('x.txt', 'w').write('secret')", operation="check")
        self.assertFalse(result.get("valid", True))
        blocked = " ".join(result.get("blocked", []))
        self.assertIn("open", blocked)

    def test_timeout_is_clamped(self):
        result = code_tool("print('ok')", operation="run", timeout=999)
        self.assertEqual(result.get("timeout_s"), 15)

    def test_rejects_large_stdin(self):
        result = code_tool("print('x')", operation="run", stdin_data="a" * 5000)
        self.assertIn("stdin_data too large", result.get("error", ""))

    def test_rejects_large_code(self):
        big_code = "print('x')\n" * 3000
        result = code_tool(big_code, operation="check")
        self.assertIn("Code too large", result.get("error", ""))

    def test_isolation_mode_process_when_forced(self):
        with patch.dict(
            "os.environ", {"RF_CODE_TOOL_ISOLATION": "process"}, clear=False
        ):
            result = code_tool("print('ok')", operation="run")
        self.assertEqual(result.get("isolation_mode"), "process")

    def test_isolation_mode_docker_fallback_warning(self):
        with (
            patch("experts.code.tools.code._docker_available", return_value=False),
            patch.dict("os.environ", {"RF_CODE_TOOL_ISOLATION": "docker"}, clear=False),
        ):
            result = code_tool("print('x')", operation="run")
        self.assertEqual(result.get("isolation_mode"), "process")
        self.assertIn("fell back to process", result.get("warning", ""))


if __name__ == "__main__":
    unittest.main()
