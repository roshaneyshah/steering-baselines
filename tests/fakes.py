"""Fake decider server (validated against the real schema) and scripted models."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from strands.models.model import Model

UPSTREAM_REPO = Path(os.environ.get("UPSTREAM_DECIDER_REPO", Path(__file__).resolve().parents[2] / "strands-decider"))


def load_upstream_schema():
    # by path: schema.py needs only pydantic
    path = UPSTREAM_REPO / "src" / "strands_decider" / "schema.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("upstream_schema", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["upstream_schema"] = module
    spec.loader.exec_module(module)
    module.SystemOneRequest.model_rebuild()
    return module


def _location_grounded(state: str) -> bool:
    conversation = state.split("--- CONVERSATION ---", 1)[1].split("--- PROPOSED TOOL CALL ---", 1)[0]
    arguments = json.loads(state.split("arguments:", 1)[1].strip())
    location = str(arguments.get("location", "")).lower()
    user_text = " ".join(line for line in conversation.lower().splitlines() if line.startswith("user:"))
    return bool(location) and location in user_text


def make_fake_decider_app() -> FastAPI:
    schema = load_upstream_schema()
    app = FastAPI()
    app.state.requests = []

    @app.get("/health")
    def health():
        return {"status": "ok", "checkpoint": "fake-decider", "base_model": "none", "max_length": 4096, "device": "cpu"}

    @app.post("/v1/systemone")
    def systemone(body: dict):
        if schema is not None:
            try:
                schema.SystemOneRequest.model_validate(body)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        app.state.requests.append(body)
        grounded = _location_grounded(body["state"])
        answers = {}
        for name in body["questions"]:
            if name == "args_grounded":
                p = 0.92 if grounded else 0.08
            elif name == "premature":
                p = 0.10 if grounded else 0.85
            else:
                p = 0.5
            answers[name] = {"type": "noul", "noul": p}
        return {"model": "fake", "answers": answers, "usage": {"input_tokens": 100, "output_tokens": 0}, "latency_ms": 1.0}

    return app


class FakeServer:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        for _ in range(200):
            if self.server.started:
                return self
            time.sleep(0.02)
        raise RuntimeError("fake server did not start")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)


KNOWN_CITIES = ("Paris", "Tokyo", "Lahore")


class EagerWeatherModel(Model):
    """Guesses Seattle; asks after guidance unless stubborn."""

    def __init__(self, stubborn: bool = False) -> None:
        self.stubborn = stubborn
        self.calls = 0

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        last = messages[-1]["content"]
        results = [b["toolResult"] for b in last if isinstance(b, dict) and "toolResult" in b]
        yield {"messageStart": {"role": "assistant"}}
        if results and results[-1].get("status") == "error" and not self.stubborn:
            yield {"contentBlockDelta": {"delta": {"text": "Which city do you mean?"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return
        if results and results[-1].get("status") != "error":
            yield {"contentBlockDelta": {"delta": {"text": "It's 21C and sunny."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return
        request = messages[0]["content"][0]["text"]
        city = next((c for c in KNOWN_CITIES if c.lower() in request.lower()), "Seattle")
        yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.calls}", "name": "get_weather"}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"location": city})}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}


class ReplyModel(Model):
    """Replies with fixed text or text(prompt)."""

    def __init__(self, text) -> None:
        self.text = text
        self.prompts: list[str] = []

    def get_config(self):
        return {}

    def update_config(self, **kwargs):
        pass

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        prompt = messages[-1]["content"][0]["text"]
        self.prompts.append(prompt)
        text = self.text(prompt) if callable(self.text) else self.text
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 50, "outputTokens": 5, "totalTokens": 55}, "metrics": {"latencyMs": 1}}}


def supervisor_rule(prompt: str) -> str:
    """GUIDE when the location isn't in the user's text."""
    state = prompt[prompt.index("A conversation between"):]
    state = state.split("\n\nShould this tool call", 1)[0]
    return "VERDICT: PROCEED" if _location_grounded(state) else "VERDICT: GUIDE"


__all__ = [
    "EagerWeatherModel",
    "FakeServer",
    "ReplyModel",
    "load_upstream_schema",
    "make_fake_decider_app",
    "supervisor_rule",
]
