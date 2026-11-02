# steering-baselines

Weather-example reproduction and baseline gates for the Cohere Labs ML Agents project "Learning When to Intervene".

Gates run before a tool call and return Proceed or Guide:

| gate | rule |
|---|---|
| `never` | always proceed |
| `always` | always guide (up to `--max-guides` per episode) |
| `upstream` | Strands Decider weather example, unchanged: two noul questions, threshold 0.45 |
| `threshold` | same questions; guide if max(1 - P(grounded), P(premature)) >= `--threshold` |
| `llm` | prompted LLM replies `VERDICT: PROCEED` or `VERDICT: GUIDE` |

## Results (Kaggle, 31 Oct 2026)

Decider: `strands-decider-2B-qwen3.5-v1-2610` on CPU. Agent and supervisor: `qwen2.5:7b` via Ollama on a T4. Six requests (three with no city, three with one), 5 repeats, 30 episodes per gate.

Upstream example, original request only: guided 5/5, no guessed call ran.

| gate | success | 95% CI | no city | city given | unneeded guides | missed bad calls | ms / decision |
|---|---|---|---|---|---|---|---|
| never | 0.50 | 0.33 to 0.67 | 0.00 | 1.00 | 0 | 15 | 0 |
| always | 0.67 | 0.49 to 0.81 | 0.93 | 0.40 | 15 | 1 | 0 |
| upstream | 0.93 | 0.79 to 0.98 | 0.87 | 1.00 | 0 | 2 | 5374 |
| threshold (0.5) | 0.90 | 0.74 to 0.97 | 0.80 | 1.00 | 0 | 3 | 5313 |
| llm | 0.93 | 0.79 to 0.98 | 0.87 | 1.00 | 0 | 2 | 260 |

Notes:
- The three selective gates are within noise of each other at n = 30.
- Latency is not comparable: the decider ran on CPU, the supervisor on GPU.
- Every miss is a retry: the agent was guided once, then proposed the same guessed city again. With `--max-guides 1` the retry proceeds by budget.
- The agent is Qwen2.5-7B, not the Claude Haiku 4.5 used upstream.

## Run

Kaggle: add this repo as a dataset, import `notebooks/kaggle_weather.ipynb`, turn on Internet and a T4, run all.

Locally:

```bash
git clone https://github.com/strands-labs/strands-decider
pip install -e strands-decider -e .
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu --port 8099

python -m steering_baselines.run_weather --gates upstream --cases upstream --model ollama:qwen2.5:7b --repeats 5
python -m steering_baselines.run_weather --gates never,always,upstream,threshold,llm --model ollama:qwen2.5:7b --repeats 5
```

Model specs: `bedrock:<id>`, `anthropic:<id>`, `openai:<id>` (set `OPENAI_BASE_URL` for other servers), `ollama:<id>`.

Each run writes `summary.md`, `summary.json`, `episodes.jsonl`, `decisions.jsonl` (includes the exact state each gate saw) and `run_config.json`.

## Tests

```bash
pip install -e ".[test]"
UPSTREAM_DECIDER_REPO=path/to/strands-decider pytest -q
```

The tests run the upstream handler and `UpstreamReviewer` on the same inputs and check that verdicts, feedback and decider requests match. A fake decider validates every request against the real request schema.

## Open questions

- tau-bench retail has no `issue_refund` tool. Candidates: `return_delivered_order_items`, `cancel_pending_order`. Gates take `gated_tools=`.
- Guide budget per episode (default 1) for the refund experiments.

## License

`decider.py` and the upstream questions, threshold, prompt and feedback are from [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider) (Apache-2.0, commit `13faf52`). See `THIRD_PARTY_LICENSE_strands-decider.txt`.
