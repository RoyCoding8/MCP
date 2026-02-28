@echo off
cd /d "%~dp0"
uv run python -m tests.test_math_tools %*
