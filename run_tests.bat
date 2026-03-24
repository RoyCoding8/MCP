@echo off
cd /d "%~dp0"
uv run python -m tests.release_gate %*
