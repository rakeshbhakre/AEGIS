"""Small HTTP adapters for public telemetry and space-weather feeds."""
from __future__ import annotations

import json
import urllib.request


USER_AGENT = "aegis-console/0.1"


def get_json(url: str, timeout: float = 7):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def get_text(url: str, timeout: float = 7) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")
