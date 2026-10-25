# Fuzzing Agent

LLM-guided fuzzing orchestrator on top of AFL++.

## Modules

| File | Role |
|---|---|
| `orchestrator.py` | Entry point. `--once` or `--interval N`. `--agentic` enables the LLM-driven workflow |
| `agent_loop.py` | **Agentic mode**: ReAct loop — the LLM reasons over tool results and chooses its own next actions via OpenAI tool-calling |
| `tools.py` | Tool layer: coverage/mapping/generation/validation exposed as LLM-callable tools with JSON schemas |
| `coverage.py` | Corpus analysis via `afl-showmap` (per-seed edge sets) |
| `mapper.py` | LLM: maps coverage → functionality, finds near-miss logic; proposes new attack surfaces on plateau |
| `generator.py` | LLM: neighbor mode (variants of seeds) / surface mode (new input classes) |
| `validator.py` | Deterministic: replays candidates, keeps only coverage-adding inputs |
| `memory.py` | Cycle history + plateau detection + failed-hypothesis memory |
| `config.py` | All settings (env-driven) |
| `llm.py` | OpenAI chat client, JSON-mode helpers |

## Setup

All configuration lives in a single `.env` file in this directory
(see `.env.example` for every knob and its default). It is loaded
automatically; exported environment variables still take precedence.

```sh
cp .env.example .env      # then edit: API key, target command, paths
pip install -r requirements.txt
```

Optional: `FUZZ_SOURCE_DIR` (target source, improves mapper/generator),
`FUZZ_LLM_MODEL` (default gpt-4o), `FUZZ_GEN_BATCH_SIZE`, `FUZZ_PLATEAU_CYCLES`.

Any OpenAI-compatible router works via `OPENAI_BASE_URL` in `.env`
(e.g. CunAI `https://www.cun.ai/v1`, OpenRouter, Ollama).

## Run

```sh
python -m agent.orchestrator --once              # single fixed-pipeline cycle
python -m agent.orchestrator --interval 3600     # hourly loop (fixed pipeline)
python -m agent.orchestrator --agentic --once    # single LLM-driven cycle (ReAct)
python -m agent.orchestrator --agentic --interval 3600  # hourly agentic loop
python -m agent.orchestrator --langgraph --once  # LangGraph workflow (recommended)
python -m agent.orchestrator --langgraph --interval 3600  # hourly LangGraph loop
python -m tests.smoke_test                       # offline test (mocked LLM)
python -m tests.agentic_test                     # offline test of the ReAct loop
```

In `--agentic` mode the LLM gets a cycle brief (coverage numbers, plateau
status, previously failed hypotheses) plus tools (`get_corpus_coverage`,
`map_coverage_to_functionality`, `propose_new_attack_surfaces`,
`generate_inputs`, `validate_candidates`, `finalize_cycle`) and decides its
own sequence of actions. Guardrails: max `FUZZ_AGENT_MAX_STEPS` tool actions
(default 10) and forced `finalize_cycle` if the budget is exhausted.

## Output

- `./accepted/` — validated, coverage-adding inputs (injection into a live
  AFL++ queue is deliberately deferred)
- `./cycle_report.json` — per-cycle stats and verdicts
- `./agent_memory.json` — hypothesis/plateau history

## Notes

- Target invocation modes: `@@` (argv file), `@-` (file path in place of the
  marker), or no marker → input piped to stdin.
- Candidate inputs are capped at `FUZZ_MAX_INPUT_SIZE` bytes; the validator
  has an execution budget (`FUZZ_VALIDATE_BUDGET`) per cycle.
