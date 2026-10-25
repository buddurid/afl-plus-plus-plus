# Fuzzing Agent

LLM-guided fuzzing orchestrator built on AFL++ (and any OpenAI-compatible LLM endpoint).

The agent is **target-agnostic and corpus-source-agnostic**: everything
specific to a project lives in `agent/.env`. Real-world corpora (from your
own AFL++ sessions or OSS-Fuzz) are prepared by `tools/ossfuzz_pull.py` in
the repo root — the agent itself never changes.

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

- Target invocation modes: `@@` (argv file), `-f @@` (target opens the file
  itself), `@-` (file path in place of the marker), or no marker → input
  piped to stdin.

## Real-world onboarding

Everything below requires **zero code changes** — only `agent/.env` edits.

### 1. Point the agent at your target

```ini
FUZZ_TARGET_CMD=/work/myproj/fuzz_target @@     # or "-f @@" / stdin mode
FUZZ_TARGET_DIR=/work/myproj
FUZZ_SOURCE_DIR=/work/myproj/src                # optional, improves LLM context
```

### 2. Start fuzzing yourself (AFL++)

```sh
afl-fuzz -i seeds/ -o /work/myproj/afl_out -- /work/myproj/fuzz_target @@
```

Then let the agent consume/refine that session:

```ini
FUZZ_CORPUS_DIR=/work/myproj/afl_out   # sync dirs understood natively
```

### 3. Pull a corpus from OSS-Fuzz and continue on it

From the repo root (the tool lives outside `agent/` on purpose):

```sh
# official path (needs gcloud auth; see OSS-Fuzz "Accessing Corpora" docs)
python3 tools/ossfuzz_pull.py --gcloud \
  "gs://<project>-corpus.clusterfuzz-external.appspot.com/libFuzzer/<fuzz_target>" \
  --project <project>

# or any public seed-corpus zip
python3 tools/ossfuzz_pull.py --url https://.../<target>_seed_corpus.zip --project <project>

# or from a local dir / zip / AFL++ session
python3 tools/ossfuzz_pull.py --from ~/afl_out --project mytarget
```

The tool flattens AFL++ sync layouts, drops `fuzzer_stats`/crash/hang
garbage, dedupes by hash, and lands inputs in `agent/corpus/<project>/`.
Then set `FUZZ_CORPUS_DIR=./corpus` (the default) and run the agent.

### 4. Per-project workspaces

Use absolute paths in `.env` so memory/reports don't tangle between projects:

```ini
FUZZ_CORPUS_DIR=/work/myproj/corpus
FUZZ_STAGING_DIR=/work/myproj/staging
FUZZ_ACCEPTED_DIR=/work/myproj/accepted
FUZZ_MEMORY_PATH=/work/myproj/agent_memory.json
FUZZ_REPORT_PATH=/work/myproj/cycle_report.json
```

### 5. Function/line coverage for real targets

Build a gcov copy of the target once and point the agent at it:

```sh
mkdir /work/myproj/gcov && cd /work/myproj/gcov
CC=gcc CFLAGS="--coverage" <your normal build>
```

```ini
FUZZ_GCOV_BUILD_DIR=/work/myproj/gcov
FUZZ_GCOV_TARGET_CMD=./fuzz_target @@
```

### 6. Sanitizer builds

Point `FUZZ_TARGET_CMD` at your ASan/UBSan build; crash inputs get flagged
by the validator (empty edge sets + nonzero rc) and never enter the corpus.
Crash triage is a future module — the validator currently just rejects them.
- Candidate inputs are capped at `FUZZ_MAX_INPUT_SIZE` bytes; the validator
  has an execution budget (`FUZZ_VALIDATE_BUDGET`) per cycle.
