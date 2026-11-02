"""Run the weather scenario under one or more gates.

    python -m steering_baselines.run_weather --gates upstream --cases upstream
    python -m steering_baselines.run_weather --gates never,always,upstream,threshold,llm --repeats 5
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from strands import Agent

from . import __version__
from .decider import DEFAULT_URL, Decider, ServerUnavailable
from .gates import (
    UPSTREAM_QUESTIONS,
    AlwaysGuide,
    BaseGate,
    LLMSupervisor,
    NeverGuide,
    UpstreamReviewer,
    ViolationQuestion,
    ViolationThreshold,
)
from .models import UPSTREAM_MODEL, make_model, model_factory
from .scenarios import weather

GATE_NAMES = ("never", "always", "upstream", "threshold", "llm")
DECIDER_GATES = {"upstream", "threshold"}


def build_gate(kind: str, args, decider: Decider | None) -> BaseGate:
    if kind == "never":
        return NeverGuide(weather.GUIDE_MESSAGE)
    if kind == "always":
        return AlwaysGuide(weather.GUIDE_MESSAGE, max_guides=args.max_guides)
    if kind == "upstream":
        return UpstreamReviewer(decider)
    if kind == "threshold":
        questions = {
            "args_grounded": ViolationQuestion(UPSTREAM_QUESTIONS["args_grounded"], "no"),
            "premature": ViolationQuestion(UPSTREAM_QUESTIONS["premature"], "yes"),
        }
        return ViolationThreshold(decider, questions, args.threshold, weather.GUIDE_MESSAGE, max_guides=args.max_guides)
    if kind == "llm":
        spec = args.supervisor_model or args.model
        return LLMSupervisor(
            model_factory(spec, args.supervisor_temperature),
            weather.GUIDE_MESSAGE,
            policy=weather.POLICY,
            max_guides=args.max_guides,
        )
    raise ValueError(f"unknown gate {kind!r}; choose from {', '.join(GATE_NAMES)}")


def run_episode(gate: BaseGate, case: weather.Case, args) -> dict:
    gate.new_episode()
    first = len(gate.records)
    weather_tool = weather.WeatherTool()
    agent = Agent(
        model=make_model(args.model, args.temperature),
        tools=[weather_tool.tool],
        interventions=[gate],
        system_prompt=weather.SYSTEM_PROMPT,
        callback_handler=None,
    )
    started = time.perf_counter()
    error = None
    reply = ""
    usage = {}
    try:
        result = agent(case.request)
        reply = str(result).strip()
        usage = dict(result.metrics.accumulated_usage)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if args.fail_fast:
            raise
    wall = time.perf_counter() - started
    decisions = gate.records[first:]
    for d in decisions:
        d["case"] = case.id
        if d["verdict"] == "guide":
            d["guide_needed"] = weather.guide_was_needed(case, d["arguments"])
    return {
        "gate": gate.name,
        "case": case.id,
        "city_given": case.city is not None,
        "request": case.request,
        "episode": gate.episode,
        "executed_calls": weather_tool.executed,
        "success": None if error else weather.score(case, weather_tool.executed),
        "n_proposals": len(decisions),
        "n_guides": sum(d["verdict"] == "guide" for d in decisions),
        "n_unneeded_guides": sum(d.get("guide_needed") is False for d in decisions),
        "gate_ms": round(sum(d["gate_ms"] for d in decisions), 2),
        "agent_input_tokens": usage.get("inputTokens", 0),
        "agent_output_tokens": usage.get("outputTokens", 0),
        "wall_s": round(wall, 3),
        "final_reply": reply,
        "error": error,
    }, decisions


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 3), round(centre + half, 3))


def summarise(episodes: list[dict], decisions: list[dict]) -> dict:
    out = {}
    for gate in dict.fromkeys(e["gate"] for e in episodes):
        eps = [e for e in episodes if e["gate"] == gate]
        ok = [e for e in eps if e["error"] is None]
        no_city = [e for e in ok if not e["city_given"]]
        city = [e for e in ok if e["city_given"]]
        decs = [d for d in decisions if d["gate"] == gate]
        timed = [d["gate_ms"] for d in decs if d.get("reason") != "guide_budget_spent"]
        k = sum(e["success"] for e in ok)
        out[gate] = {
            "episodes": len(eps),
            "errors": len(eps) - len(ok),
            "success_rate": round(k / len(ok), 3) if ok else None,
            "success_95ci": wilson(k, len(ok)),
            "success_no_city": round(sum(e["success"] for e in no_city) / len(no_city), 3) if no_city else None,
            "success_city": round(sum(e["success"] for e in city) / len(city), 3) if city else None,
            "guides_per_episode": round(sum(e["n_guides"] for e in ok) / len(ok), 3) if ok else None,
            "unneeded_guides": sum(e["n_unneeded_guides"] for e in ok),
            "missed_bad_calls": sum(1 for e in no_city if e["executed_calls"]),
            "mean_gate_ms_per_decision": round(sum(timed) / len(timed), 1) if timed else 0.0,
            "agent_tokens_per_episode": round(
                sum(e["agent_input_tokens"] + e["agent_output_tokens"] for e in ok) / len(ok), 1
            )
            if ok
            else None,
            "supervisor_unparsed_replies": sum(d.get("parse_ok") is False for d in decs),
        }
    return out


def to_markdown(summary: dict) -> str:
    cols = [
        ("success_rate", "success"),
        ("success_95ci", "95% CI"),
        ("success_no_city", "no city"),
        ("success_city", "city given"),
        ("guides_per_episode", "guides/ep"),
        ("unneeded_guides", "unneeded guides"),
        ("missed_bad_calls", "missed bad calls"),
        ("mean_gate_ms_per_decision", "gate ms"),
        ("errors", "errors"),
    ]
    lines = ["| gate | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for gate, row in summary.items():
        lines.append(f"| {gate} | " + " | ".join(str(row[c[0]]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates", default="upstream", help=f"comma list from: {', '.join(GATE_NAMES)}")
    p.add_argument("--cases", default="all", choices=["all", "upstream"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--model", default=UPSTREAM_MODEL, help="agent model spec (see models.py)")
    p.add_argument("--temperature", type=float, default=None, help="agent temperature (provider default if unset)")
    p.add_argument("--supervisor-model", default=None, help="LLM supervisor spec (default: same as --model)")
    p.add_argument("--supervisor-temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.5, help="violation-threshold gate cut-off")
    p.add_argument("--max-guides", type=int, default=1, help="guide budget per episode for baseline gates")
    p.add_argument("--decider-url", default=DEFAULT_URL)
    p.add_argument("--out", default=None, help="output folder (default runs/<timestamp>)")
    p.add_argument("--fail-fast", action="store_true", help="stop on the first agent error")
    args = p.parse_args(argv)

    kinds = [k.strip() for k in args.gates.split(",") if k.strip()]
    for k in kinds:
        if k not in GATE_NAMES:
            p.error(f"unknown gate {k!r}; choose from {', '.join(GATE_NAMES)}")
    if args.repeats < 1:
        p.error("--repeats must be >= 1")

    decider, health = None, None
    if DECIDER_GATES & set(kinds):
        decider = Decider(args.decider_url)
        try:
            health = decider.health()
        except ServerUnavailable as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"decider: {health.get('checkpoint')} on {health.get('device')} at {decider.url}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"runs/{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    cases = [weather.UPSTREAM_CASE] if args.cases == "upstream" else weather.CASES
    config = {
        "started_utc": stamp,
        "package_version": __version__,
        "python": platform.python_version(),
        "args": vars(args),
        "decider_health": health,
        "cases": [c.__dict__ for c in cases],
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2))

    episodes, decisions = [], []
    with (out / "episodes.jsonl").open("w") as ef, (out / "decisions.jsonl").open("w") as df:
        for kind in kinds:
            gate = build_gate(kind, args, decider)
            for r in range(args.repeats):
                for case in cases:
                    try:
                        ep, decs = run_episode(gate, case, args)
                    except Exception:
                        traceback.print_exc()
                        return 2
                    ep["repeat"] = r
                    episodes.append(ep)
                    decisions.extend(decs)
                    ef.write(json.dumps(ep) + "\n")
                    for d in decs:
                        df.write(json.dumps(d) + "\n")
                    ef.flush()
                    df.flush()
                    mark = "ERR" if ep["error"] else ("ok " if ep["success"] else "BAD")
                    verdicts = ",".join(d["verdict"] for d in decs) or "-"
                    print(f"[{mark}] {gate.name:20s} {case.id:18s} r{r} verdicts={verdicts} ran={ep['executed_calls']}")
                    if ep["error"]:
                        print(f"      error: {ep['error']}")

    summary = summarise(episodes, decisions)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    table = to_markdown(summary)
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table)
    if episodes and all(e["error"] for e in episodes):
        print("\nall episodes failed, see episodes.jsonl")
    if args.cases == "upstream" and episodes:
        print(f"\nfinal reply (last episode):\n{episodes[-1]['final_reply']}")
    print(f"\nresults in {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
