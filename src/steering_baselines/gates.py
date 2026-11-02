"""Pre-tool-call gates. All share BaseGate so snapshots, guide budget and logging match."""

from __future__ import annotations

import json
import re
import time
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from strands import Agent
from strands.interventions import Guide, InterventionHandler, Proceed
from strands.models.model import Model

from .decider import Decider

def render_upstream(messages: list[dict]) -> str:
    """Same as the upstream example's _render (text blocks only)."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        for block in message.get("content", []):
            if isinstance(block, dict) and "text" in block:
                lines.append(f"{role}: {block['text']}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def render_full(messages: list[dict], pending_tool_use_id: str | None = None) -> str:
    """Text, earlier tool calls and results; skips the pending call."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        for block in message.get("content", []):
            if not isinstance(block, dict):
                continue
            if "text" in block:
                lines.append(f"{role}: {block['text']}")
            elif "toolUse" in block:
                use = block["toolUse"]
                if use.get("toolUseId") == pending_tool_use_id:
                    continue
                lines.append(f"{role} called {use.get('name')}({json.dumps(use.get('input', {}))})")
            elif "toolResult" in block:
                result = block["toolResult"]
                texts = [c.get("text", "") for c in result.get("content", []) if isinstance(c, dict)]
                lines.append(f"tool result ({result.get('status', 'success')}): {' '.join(texts)}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def build_state(conversation: str, tool: str, arguments: Mapping[str, Any]) -> str:
    return (
        "A conversation between a user and an AI assistant is below, followed by a "
        "tool call the assistant now wants to make.\n\n"
        f"--- CONVERSATION ---\n{conversation}\n\n"
        f"--- PROPOSED TOOL CALL ---\ntool: {tool}\n"
        f"arguments: {json.dumps(arguments)}"
    )


@dataclass
class Snapshot:
    tool: str
    arguments: dict[str, Any]
    conversation: str

    @property
    def state(self) -> str:
        return build_state(self.conversation, self.tool, self.arguments)


class BaseGate(InterventionHandler):
    """max_guides: guides allowed per episode, after which calls proceed. None = no limit."""

    name = "base-gate"
    renderer = "full"

    def __init__(
        self,
        guide_message: str,
        gated_tools: Iterable[str] | None = None,
        max_guides: int | None = 1,
    ) -> None:
        self.guide_message = guide_message
        self.gated_tools = set(gated_tools) if gated_tools is not None else None
        self.max_guides = max_guides
        self.records: list[dict[str, Any]] = []
        self.episode = 0
        self._guides = 0

    def new_episode(self) -> None:
        self.episode += 1
        self._guides = 0

    @abstractmethod
    def decide(self, snapshot: Snapshot) -> tuple[bool, dict[str, Any], str | None]:
        """Return (guide, log details, feedback override or None)."""

    def _snapshot(self, event) -> Snapshot:
        tool = event.tool_use["name"]
        arguments = dict(event.tool_use.get("input") or {})
        if self.renderer == "upstream":
            conversation = render_upstream(event.agent.messages)
        else:
            conversation = render_full(event.agent.messages, event.tool_use.get("toolUseId"))
        return Snapshot(tool, arguments, conversation)

    def before_tool_call(self, event, **kwargs):
        tool = event.tool_use["name"]
        if self.gated_tools is not None and tool not in self.gated_tools:
            return Proceed()
        snap = self._snapshot(event)
        record: dict[str, Any] = {
            "gate": self.name,
            "episode": self.episode,
            "tool": tool,
            "arguments": snap.arguments,
            "state": snap.state,
        }
        if self.max_guides is not None and self._guides >= self.max_guides:
            guide, details, feedback, ms = False, {"reason": "guide_budget_spent"}, None, 0.0
        else:
            started = time.perf_counter()
            guide, details, feedback = self.decide(snap)
            ms = (time.perf_counter() - started) * 1000
        record.update(details)
        record["gate_ms"] = round(ms, 2)
        record["verdict"] = "guide" if guide else "proceed"
        self.records.append(record)
        if guide:
            self._guides += 1
            return Guide(feedback=feedback or self.guide_message)
        return Proceed()


class NeverGuide(BaseGate):
    name = "never-guide"

    def decide(self, snapshot):
        return False, {}, None


class AlwaysGuide(BaseGate):
    name = "always-guide"

    def decide(self, snapshot):
        return True, {}, None


# From strands-decider/examples/strands/tool_call_intervention.py @ 13faf52
UPSTREAM_QUESTIONS = {
    "args_grounded": Decider.noul(
        "Are the tool's argument values grounded in facts the user actually provided?",
        {
            "true": "every argument value traces back to something the user said",
            "false": "an argument value was guessed or invented, not stated by the user",
        },
    ),
    "premature": Decider.noul(
        "Is it premature to call this tool now, before clarifying with the user?",
        {
            "true": "the assistant should ask a clarifying question before calling the tool",
            "false": "there is nothing left to clarify; calling now is appropriate",
        },
    ),
}
UPSTREAM_YES = 0.45
UPSTREAM_FEEDBACK_UNGROUNDED = (
    "The arguments are not grounded in anything the user said -- they look "
    "guessed. Ask the user to confirm the values instead of inventing them."
)
UPSTREAM_FEEDBACK_PREMATURE = "It is too early to call this tool. Clarify with the user first, then try again."


class UpstreamReviewer(BaseGate):
    """Upstream weather gate. Upstream has no guide limit; 5 is just a loop guard."""

    name = "upstream-reviewer"
    renderer = "upstream"

    def __init__(self, decider: Decider, gated_tools=None, max_guides: int | None = 5) -> None:
        super().__init__(UPSTREAM_FEEDBACK_UNGROUNDED, gated_tools, max_guides)
        self.decider = decider

    def decide(self, snapshot):
        answers = self.decider.ask(snapshot.state, UPSTREAM_QUESTIONS)
        grounded = answers["args_grounded"]["noul"]
        premature = answers["premature"]["noul"]
        details = {"args_grounded": round(grounded, 4), "premature": round(premature, 4)}
        if grounded < UPSTREAM_YES:
            return True, details, UPSTREAM_FEEDBACK_UNGROUNDED
        if premature >= UPSTREAM_YES:
            return True, details, UPSTREAM_FEEDBACK_PREMATURE
        return False, details, None


@dataclass
class ViolationQuestion:
    question: dict[str, Any]
    violation_when: str  # "yes" or "no"

    def violation_prob(self, p_yes: float) -> float:
        if self.violation_when == "yes":
            return p_yes
        if self.violation_when == "no":
            return 1.0 - p_yes
        raise ValueError(f"violation_when must be 'yes' or 'no', got {self.violation_when!r}")


class ViolationThreshold(BaseGate):
    """Guide when max P(violation) over the questions >= threshold."""

    name = "violation-threshold"

    def __init__(
        self,
        decider: Decider,
        questions: Mapping[str, ViolationQuestion],
        threshold: float,
        guide_message: str,
        gated_tools=None,
        max_guides: int | None = 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        super().__init__(guide_message, gated_tools, max_guides)
        self.decider = decider
        self.questions = dict(questions)
        self.threshold = threshold

    def decide(self, snapshot):
        answers = self.decider.ask(snapshot.state, {k: q.question for k, q in self.questions.items()})
        probs = {k: answers[k]["noul"] for k in self.questions}
        violations = {k: self.questions[k].violation_prob(p) for k, p in probs.items()}
        score = max(violations.values())
        details = {
            "p_yes": {k: round(v, 4) for k, v in probs.items()},
            "violation_score": round(score, 4),
            "threshold": self.threshold,
        }
        return score >= self.threshold, details, None


SUPERVISOR_SYSTEM = (
    "You supervise an AI assistant. Before each tool call you decide whether to let it run "
    "or to stop it and send the assistant back with a short correction. Stop a call only if "
    "letting it run would likely lead to a worse outcome for the user than correcting the "
    "assistant now. Reply with one line and nothing else: VERDICT: PROCEED or VERDICT: GUIDE."
)
_VERDICT = re.compile(r"VERDICT:\s*(PROCEED|GUIDE)\b", re.IGNORECASE)


class LLMSupervisor(BaseGate):
    """Prompted LLM gate. Unparseable replies proceed and are logged with parse_ok=False."""

    name = "llm-supervisor"

    def __init__(
        self,
        model_factory: Callable[[], Model],
        guide_message: str,
        policy: str = "",
        gated_tools=None,
        max_guides: int | None = 1,
    ) -> None:
        super().__init__(guide_message, gated_tools, max_guides)
        self.model_factory = model_factory
        self.policy = policy.strip()

    def _prompt(self, snapshot: Snapshot) -> str:
        policy = f"--- POLICY ---\n{self.policy}\n\n" if self.policy else ""
        return f"{policy}{snapshot.state}\n\nShould this tool call PROCEED, or should you GUIDE the assistant?"

    def decide(self, snapshot):
        judge = Agent(model=self.model_factory(), system_prompt=SUPERVISOR_SYSTEM, callback_handler=None)
        result = judge(self._prompt(snapshot))
        reply = str(result).strip()
        match = _VERDICT.search(reply)
        usage = result.metrics.accumulated_usage
        details = {
            "supervisor_reply": reply[:500],
            "parse_ok": match is not None,
            "supervisor_input_tokens": usage.get("inputTokens", 0),
            "supervisor_output_tokens": usage.get("outputTokens", 0),
        }
        guide = match is not None and match.group(1).upper() == "GUIDE"
        return guide, details, None
