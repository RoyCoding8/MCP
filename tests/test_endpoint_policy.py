import os
import unittest
from unittest.mock import patch

from endpoint_policy import validate_endpoint_url


class EndpointPolicyTests(unittest.TestCase):
    def test_localhost_allowed_by_default(self):
        with patch.dict(
            os.environ,
            {
                "RF_ALLOW_REMOTE_ENDPOINTS": "0",
                "RF_ENDPOINT_ALLOWLIST": "localhost,127.0.0.1,::1",
            },
            clear=False,
        ):
            url = "http://localhost:11434/api/chat"
            self.assertEqual(validate_endpoint_url(url), url)

    def test_remote_blocked_by_default(self):
        with patch.dict(
            os.environ,
            {
                "RF_ALLOW_REMOTE_ENDPOINTS": "0",
                "RF_ENDPOINT_ALLOWLIST": "localhost,127.0.0.1,::1",
            },
            clear=False,
        ):
            with self.assertRaises(ValueError):
                validate_endpoint_url("https://api.openai.com/v1/chat/completions")

    def test_remote_allowed_when_flag_enabled(self):
        with patch.dict(
            os.environ,
            {
                "RF_ALLOW_REMOTE_ENDPOINTS": "1",
                "RF_ENDPOINT_ALLOWLIST": "localhost,127.0.0.1,::1",
            },
            clear=False,
        ):
            url = "https://api.openai.com/v1/chat/completions"
            self.assertEqual(validate_endpoint_url(url), url)

    def test_bare_host_gets_http_prefix(self):
        with patch.dict(
            os.environ,
            {
                "RF_ALLOW_REMOTE_ENDPOINTS": "0",
                "RF_ENDPOINT_ALLOWLIST": "localhost,127.0.0.1,::1",
            },
            clear=False,
        ):
            self.assertEqual(
                validate_endpoint_url("localhost:11434/api/chat"),
                "http://localhost:11434/api/chat",
            )


if __name__ == "__main__":
    unittest.main()
