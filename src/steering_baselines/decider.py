"""Client for a strands-decider server. Adapted from strands-decider examples/strands/_client.py (Apache-2.0)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

ENV_URL = "STRANDS_DECIDER_URL"
DEFAULT_URL = os.environ.get(ENV_URL, "http://127.0.0.1:8099")
SERVE_HINT = "strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099"


class ServerUnavailable(RuntimeError):
    pass


class Decider:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 300.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.total_wall_ms = 0.0

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = dict(criteria)
        return question

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps({"state": state, "questions": dict(questions)}).encode()
        request = urllib.request.Request(
            f"{self.url}/v1/systemone", data=body, headers={"content-type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"decider returned HTTP {exc.code}: {exc.read()[:500]!r}") from exc
        except urllib.error.URLError as exc:
            raise ServerUnavailable(f"no decider at {self.url} ({exc}). Start one with:\n  {SERVE_HINT}") from exc
        self.total_wall_ms += (time.perf_counter() - started) * 1000
        self.calls += 1
        self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
        return payload["answers"]

    def health(self) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise ServerUnavailable(f"no decider at {self.url} ({exc}). Start one with:\n  {SERVE_HINT}") from exc
