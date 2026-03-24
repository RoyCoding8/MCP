"""Endpoint validation utilities for outbound model server calls."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse


_BLOCKED_HOSTS = {
    "0.0.0.0",
    "169.254.169.254",
    "metadata.google.internal",
    "metadata",
}


def _allow_remote() -> bool:
    return os.environ.get("RF_ALLOW_REMOTE_ENDPOINTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _endpoint_allowlist() -> set[str]:
    raw = os.environ.get("RF_ENDPOINT_ALLOWLIST", "localhost,127.0.0.1,::1")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def validate_endpoint_url(endpoint_url: str) -> str:
    endpoint = (endpoint_url or "").strip()
    if not endpoint:
        raise ValueError("Endpoint URL cannot be empty.")
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"

    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Endpoint must use http:// or https://")

    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("Endpoint must include a valid hostname.")
    if host in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked endpoint host: {host}")

    if _allow_remote():
        return endpoint

    allowlist = _endpoint_allowlist()
    if host in allowlist:
        return endpoint

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip and ip.is_loopback:
        return endpoint

    allow = ", ".join(sorted(allowlist))
    raise ValueError(
        f"Endpoint host '{host}' is not allowed. "
        f"Set RF_ALLOW_REMOTE_ENDPOINTS=1 or extend RF_ENDPOINT_ALLOWLIST ({allow})."
    )
