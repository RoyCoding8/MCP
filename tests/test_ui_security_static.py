import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "ui" / "app.py"


class UiSecurityStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")

    def test_chatbot_html_sanitization_enabled(self):
        self.assertIn("sanitize_html=True", self.source)

    def test_server_defaults_to_localhost(self):
        self.assertIn('os.environ.get("RF_SERVER_NAME", "127.0.0.1")', self.source)

    def test_stop_button_cancels_stream_events(self):
        self.assertIn("cancels=[submit_stream, click_stream]", self.source)


if __name__ == "__main__":
    unittest.main()
