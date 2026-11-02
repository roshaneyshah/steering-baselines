import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
from strands import Agent

import steering_baselines.run_weather as rw
from steering_baselines import gates
from steering_baselines.decider import Decider, ServerUnavailable
from steering_baselines.scenarios import weather

from fakes import (
    UPSTREAM_REPO,
    EagerWeatherModel,
    FakeServer,
    ReplyModel,
    load_upstream_schema,
    make_fake_decider_app,
    supervisor_rule,
)

CASE = {c.id: c for c in weather.CASES}


@pytest.fixture(scope="module")
def server():
    with FakeServer(make_fake_decider_app()) as s:
        yield s


@pytest.fixture
def decider(server):
    server.app.state.requests.clear()
    return Decider(server.url)


@pytest.fixture(autouse=True)
def scripted_agent(monkeypatch):
    monkeypatch.setattr(rw, "make_model", lambda spec, temperature=None: EagerWeatherModel())


def args(**kw):
    base = dict(model="fake:agent", temperature=None, supervisor_model=None, supervisor_temperature=0.0,
                threshold=0.5, max_guides=1, fail_fast=True)
    base.update(kw)
    return SimpleNamespace(**base)


def load_upstream_example():
    folder = UPSTREAM_REPO / "examples" / "strands"
    if not (folder / "tool_call_intervention.py").exists():
        pytest.skip("upstream clone not found; set UPSTREAM_DECIDER_REPO")
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location("upstream_example", folder / "tool_call_intervention.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(folder))


def test_upstream_constants_match_exactly():
    up = load_upstream_example()
    assert gates.UPSTREAM_QUESTIONS == up.QUESTIONS
    assert gates.UPSTREAM_YES == up.YES
    assert weather.SYSTEM_PROMPT == up.SYSTEM_PROMPT
    assert weather.UPSTREAM_CASE.request == up.USER_REQUEST
    src = (UPSTREAM_REPO / "examples" / "strands" / "tool_call_intervention.py").read_text()
    assert '"us.anthropic.claude-haiku-4-5-20251001-v1:0"' in src
    assert rw.UPSTREAM_MODEL == "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_upstream_policy_feedback_and_state_match_exactly():
    up = load_upstream_example()

    class StubDecider:
        def __init__(self, grounded, premature):
            self.answer = {"args_grounded": {"noul": grounded}, "premature": {"noul": premature}}
            self.seen = []

        def ask(self, state, questions):
            self.seen.append((state, questions))
            return self.answer

    messages = [
        {"role": "user", "content": [{"text": "What's the weather?"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "a", "name": "get_weather", "input": {"location": "Seattle"}}}]},
    ]
    event = SimpleNamespace(tool_use={"toolUseId": "a", "name": "get_weather", "input": {"location": "Seattle"}},
                            agent=SimpleNamespace(messages=messages))
    for grounded, premature in [(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.45, 0.4499), (0.4499, 0.45)]:
        a, b = StubDecider(grounded, premature), StubDecider(grounded, premature)
        theirs = up.ToolCallReviewer(a).before_tool_call(event)
        ours = gates.UpstreamReviewer(b).before_tool_call(event)
        assert type(theirs) is type(ours)
        assert getattr(theirs, "feedback", None) == getattr(ours, "feedback", None)
        assert a.seen == b.seen


def test_upstream_render_matches_exactly():
    up = load_upstream_example()
    messages = [
        {"role": "user", "content": [{"text": "What's the weather?"}]},
        {"role": "assistant", "content": [{"text": "Let me check."}, {"toolUse": {"toolUseId": "a", "name": "get_weather", "input": {"location": "Seattle"}}}]},
    ]
    assert gates.render_upstream(messages) == up.ToolCallReviewer._render(messages)
    assert gates.render_upstream([]) == up.ToolCallReviewer._render([])


def test_requests_validate_against_real_schema(decider):
    schema = load_upstream_schema()
    if schema is None:
        pytest.skip("upstream clone not found")
    body = {"state": "x", "questions": gates.UPSTREAM_QUESTIONS}
    schema.SystemOneRequest.model_validate(body)


def test_reproduction_upstream_case(decider, server):
    gate = gates.UpstreamReviewer(decider)
    ep, decs = rw.run_episode(gate, weather.UPSTREAM_CASE, args())
    assert ep["success"] is True
    assert ep["executed_calls"] == []
    assert [d["verdict"] for d in decs] == ["guide"]
    assert decs[0]["args_grounded"] < gates.UPSTREAM_YES
    assert ep["final_reply"] == "Which city do you mean?"
    sent = server.app.state.requests[0]
    assert sent["questions"] == gates.UPSTREAM_QUESTIONS
    assert sent["state"] == gates.build_state("user: What's the weather?", "get_weather", {"location": "Seattle"})


def test_never_guide_lets_guesses_through(decider):
    gate = gates.NeverGuide(weather.GUIDE_MESSAGE)
    results = {c.id: rw.run_episode(gate, c, args())[0] for c in weather.CASES}
    assert results["upstream"]["executed_calls"] == ["Seattle"]
    assert results["upstream"]["success"] is False
    assert results["paris"]["success"] is True
    s = rw.summarise(list(results.values()), gate.records)["never-guide"]
    assert s["success_rate"] == 0.5 and s["missed_bad_calls"] == 3 and s["guides_per_episode"] == 0


def test_always_guide_counts_unneeded_guides(decider):
    gate = gates.AlwaysGuide(weather.GUIDE_MESSAGE)
    ep, decs = rw.run_episode(gate, CASE["paris"], args())
    assert decs[0]["verdict"] == "guide" and decs[0]["guide_needed"] is False
    assert ep["n_unneeded_guides"] == 1
    assert ep["success"] is False


def test_violation_threshold(decider):
    q = {
        "args_grounded": gates.ViolationQuestion(gates.UPSTREAM_QUESTIONS["args_grounded"], "no"),
        "premature": gates.ViolationQuestion(gates.UPSTREAM_QUESTIONS["premature"], "yes"),
    }
    gate = gates.ViolationThreshold(decider, q, 0.5, weather.GUIDE_MESSAGE)
    eps = [rw.run_episode(gate, c, args())[0] for c in weather.CASES]
    assert all(e["success"] for e in eps)
    rec = gate.records[0]
    assert rec["violation_score"] == pytest.approx(max(1 - 0.08, 0.85), abs=1e-4)
    with pytest.raises(ValueError):
        gates.ViolationThreshold(decider, q, 1.5, "x")
    with pytest.raises(ValueError):
        gates.ViolationQuestion({}, "maybe").violation_prob(0.3)


def test_llm_supervisor_rule_and_parse_failure():
    gate = gates.LLMSupervisor(lambda: ReplyModel(supervisor_rule), weather.GUIDE_MESSAGE, policy=weather.POLICY)
    eps = [rw.run_episode(gate, c, args())[0] for c in weather.CASES]
    assert all(e["success"] for e in eps)
    assert all(r["parse_ok"] for r in gate.records)
    assert gate.records[0]["supervisor_input_tokens"] == 50

    broken = gates.LLMSupervisor(lambda: ReplyModel("I think it is fine"), weather.GUIDE_MESSAGE)
    ep, decs = rw.run_episode(broken, weather.UPSTREAM_CASE, args())
    assert decs[0]["parse_ok"] is False and decs[0]["verdict"] == "proceed"
    assert rw.summarise([ep], decs)["llm-supervisor"]["supervisor_unparsed_replies"] == 1


def test_guide_budget_stops_a_stubborn_agent(monkeypatch):
    monkeypatch.setattr(rw, "make_model", lambda spec, temperature=None: EagerWeatherModel(stubborn=True))
    gate = gates.AlwaysGuide(weather.GUIDE_MESSAGE, max_guides=2)
    ep, decs = rw.run_episode(gate, weather.UPSTREAM_CASE, args())
    assert [d["verdict"] for d in decs] == ["guide", "guide", "proceed"]
    assert decs[-1]["reason"] == "guide_budget_spent"
    assert ep["executed_calls"] == ["Seattle"]


def test_budget_resets_each_episode(decider):
    gate = gates.AlwaysGuide(weather.GUIDE_MESSAGE)
    for _ in range(3):
        _, decs = rw.run_episode(gate, weather.UPSTREAM_CASE, args())
        assert decs[0]["verdict"] == "guide"


def test_gated_tools_filter():
    calls = []

    class Recorder(gates.AlwaysGuide):
        def decide(self, snapshot):
            calls.append(snapshot.tool)
            return super().decide(snapshot)

    gate = Recorder(weather.GUIDE_MESSAGE, gated_tools={"issue_refund"})
    tool = weather.WeatherTool()
    Agent(model=EagerWeatherModel(), tools=[tool.tool], interventions=[gate], callback_handler=None)("What's the weather?")
    assert calls == [] and tool.executed == ["Seattle"]


def test_render_full_skips_pending_call():
    msgs = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "old", "name": "lookup", "input": {"id": 1}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "old", "status": "success", "content": [{"text": "found"}]}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "new", "name": "issue_refund", "input": {}}}]},
    ]
    text = gates.render_full(msgs, "new")
    assert "lookup" in text and "found" in text and "issue_refund" not in text


def test_cli_end_to_end(server, tmp_path, monkeypatch):
    monkeypatch.setattr(rw, "model_factory", lambda spec, t=None: (lambda: ReplyModel(supervisor_rule)))
    out = tmp_path / "run"
    code = rw.main(["--gates", "never,always,upstream,threshold,llm", "--repeats", "2",
                    "--decider-url", server.url, "--out", str(out), "--model", "fake:agent"])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"never-guide", "always-guide", "upstream-reviewer", "violation-threshold", "llm-supervisor"}
    assert summary["never-guide"]["success_rate"] == 0.5
    assert summary["always-guide"]["success_rate"] == 0.5
    assert summary["always-guide"]["unneeded_guides"] == 6
    for g in ("upstream-reviewer", "violation-threshold", "llm-supervisor"):
        assert summary[g]["success_rate"] == 1.0 and summary[g]["errors"] == 0
    lines = (out / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 5 * 2 * 6
    config = json.loads((out / "run_config.json").read_text())
    assert config["decider_health"]["checkpoint"] == "fake-decider"


def test_cli_reports_missing_decider(tmp_path):
    assert rw.main(["--gates", "upstream", "--decider-url", "http://127.0.0.1:9", "--out", str(tmp_path)]) == 1


def test_cli_rejects_unknown_gate():
    with pytest.raises(SystemExit):
        rw.main(["--gates", "nope"])


def test_decider_unavailable_message():
    with pytest.raises(ServerUnavailable, match="strands-decider serve"):
        Decider("http://127.0.0.1:9").health()


def test_wilson():
    lo, hi = rw.wilson(5, 10)
    assert lo < 0.5 < hi
    assert rw.wilson(0, 10)[0] == 0.0


def test_model_spec_errors():
    from steering_baselines.models import make_model

    with pytest.raises(ValueError):
        make_model("no-colon")
    with pytest.raises(ValueError):
        make_model("mystery:x")


def test_fake_server_rejects_bad_requests_like_the_real_one(decider):
    with pytest.raises(RuntimeError, match="HTTP 422"):
        decider.ask("state", {"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}})


def test_wilson_empty_is_valid_json():
    assert rw.wilson(0, 0) is None
    json.dumps(rw.summarise([{"gate": "g", "error": "x", "city_given": True}], []), allow_nan=False)
