"""Central configuration for the fuzzing agent.

Every knob is read from environment variables, which are loaded from the
.env file in the agent/ directory via python-dotenv. Precedence:
real environment > agent/.env > defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load agent/.env once, before reading any variables.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

_AGENT_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    _dotenv_path = _AGENT_DIR / ".env"
    if _dotenv_path.is_file():
        load_dotenv(_dotenv_path, override=False)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # --- Target under test -------------------------------------------------
    # Command template executed to run the target on one input.
    # "@@" is replaced with the input file path (AFL++ convention).
    target_cmd: str = _env("FUZZ_TARGET_CMD", "./test-instr @@")
    target_dir: str = _env("FUZZ_TARGET_DIR", ".")

    # --- Corpus ------------------------------------------------------------
    corpus_dir: str = _env("FUZZ_CORPUS_DIR", "./corpus")
    # Where generated-but-not-yet-validated candidates go
    staging_dir: str = _env("FUZZ_STAGING_DIR", "./staging")
    # Validated, coverage-adding inputs (injection point decided later)
    accepted_dir: str = _env("FUZZ_ACCEPTED_DIR", "./accepted")

    # --- Coverage ----------------------------------------------------------
    # Path to afl-showmap binary
    showmap_bin: str = _env("FUZZ_SHOWMAP", "afl-showmap")
    # Number of top seeds per call to feed the LLM
    max_seed_samples: int = int(_env("FUZZ_MAX_SEED_SAMPLES", "8"))
    # Optional gcov-instrumented build of the target (directory containing the
    # binary compiled with `gcc --coverage`). Enables function/line coverage:
    # set to "" to disable. Example: /tmp/gcov_build
    gcov_build_dir: str = _env("FUZZ_GCOV_BUILD_DIR", "")
    # Command template for the gcov binary (defaults to target_cmd)
    gcov_target_cmd: str = _env("FUZZ_GCOV_TARGET_CMD", "")
    # Directory containing digest.json from tools/ossfuzz_coverage.py digest
    # (clang source-based coverage of a real-world OSS-Fuzz target). When set,
    # function-level coverage is read from it instead of/in addition to gcov.
    # Example: /path/to/<project>/oss-fuzz-infra/build/coverage_report
    ossfuzz_report_dir: str = _env("FUZZ_OSSFUZZ_REPORT_DIR", "")
    # Live campaign queue dir(s) (AFL -o .../main/queue). When set, their
    # inputs are merged into the agent's coverage snapshot, so a candidate is
    # only "accepted" if it beats everything the running campaign already has.
    live_queue_dirs: str = _env("FUZZ_LIVE_QUEUE_DIRS", "")
    # Cap how many live-queue inputs are sampled per snapshot (newest first).
    live_queue_max: int = int(_env("FUZZ_LIVE_QUEUE_MAX", "6000"))

    # --- Harness structure (structured-input generation) --------------------
    # Some harnesses start their input with a header of control fields (a mode
    # selector, flags, sizes) followed by the payload. Set the header size so
    # the agent can rewrite those control values deliberately (control_sweep),
    # once the LLM has read the harness source to learn the actual layout.
    harness_header_bytes: int = int(_env("FUZZ_HARNESS_HEADER_BYTES", "0"))
    # Control values to sweep (comma/space separated ints worth trying).
    harness_options: str = _env("FUZZ_HARNESS_OPTIONS", "")
    # How many top seeds the sweep rewrites per call.
    harness_sweep_seeds: int = int(_env("FUZZ_HARNESS_SWEEP_SEEDS", "6"))
    # Which header field control_sweep edits by default: "offset:width".
    harness_control_field: str = _env("FUZZ_HARNESS_CONTROL_FIELD", "0:4")
    # Full header field layout for describing the header to the LLM,
    # e.g. "0:4,4:4,8:4" (offset:width, comma-separated). Optional.
    harness_header_fields: str = _env("FUZZ_HARNESS_HEADER_FIELDS", "")
    # Path to the fuzz harness source. Given to the LLM so it can learn the
    # exact input format and scope dark-feature analysis to what this harness
    # can actually reach. The harness's own local #includes are followed too.
    harness_file: str = _env("FUZZ_HARNESS_FILE", "")
    # Max bytes of harness + included helper sources to feed the LLM. Keep it
    # small enough that the generation request fits the provider's request-time
    # limit (free endpoints often time out at ~120s on large prompts).
    harness_context_bytes: int = int(_env("FUZZ_HARNESS_CONTEXT_BYTES", "5000"))
    # Corpus seed preview in generation prompts: how many seeds and how many
    # bytes of each to show (escaped + base64). Kept small for the same reason.
    seed_preview_count: int = int(_env("FUZZ_SEED_PREVIEW_COUNT", "4"))
    seed_preview_bytes: int = int(_env("FUZZ_SEED_PREVIEW_BYTES", "192"))
    # Max bytes of uncovered-function source shown per target in generation
    # prompts (keeps requests small enough for slow providers).
    target_source_bytes: int = int(_env("FUZZ_TARGET_SOURCE_BYTES", "3000"))

    # --- Symbol / entry-point heuristics (target-agnostic) ------------------
    # Comma-separated prefixes that identify the target's PUBLIC API symbols
    # (used to spot entry points in the function list). Empty = no prefix
    # heuristic: every function is treated as a potential entry point.
    # Examples: a project's C API prefix, e.g. "SSL_" (openssl), "av_" (ffmpeg).
    symbol_prefixes: str = _env("FUZZ_SYMBOL_PREFIXES", "")
    # A subsystem at >= this line-coverage % counts as "driven" by the harness
    # (reachable, merely unfinished) rather than "untouched/out of scope".
    subsystem_driven_pct: float = float(_env("FUZZ_SUBSYSTEM_DRIVEN_PCT", "1.0"))

    # --- Plateau detection -------------------------------------------------
    plateau_cycles: int = int(_env("FUZZ_PLATEAU_CYCLES", "3"))
    # If fewer than this many new edges were accepted over the last
    # `plateau_cycles` cycles, we consider coverage to have plateaued.
    plateau_new_edges: int = int(_env("FUZZ_PLATEAU_NEW_EDGES", "5"))

    # --- LLM (OpenAI) ------------------------------------------------------
    openai_api_key: str = _env("OPENAI_API_KEY")
    openai_model: str = _env("FUZZ_LLM_MODEL", "gpt-4o")
    openai_base_url: str = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_temperature: float = float(_env("FUZZ_LLM_TEMPERATURE", "0.8"))
    # Per-request timeout (seconds) for LLM calls. Long generation prompts on
    # slow providers can otherwise hang indefinitely.
    llm_timeout: float = float(_env("FUZZ_LLM_TIMEOUT", "180"))
    # Base backoff (seconds) between retries of transient provider failures
    # (5xx / Cloudflare 524 / rate limits). Retries wait backoff * attempt.
    llm_retry_backoff: float = float(_env("FUZZ_LLM_RETRY_BACKOFF", "10"))
    # Comma-separated fallback models tried when the primary model fails
    # (empty body / 4xx / 5xx). Provider availability is model-specific.
    llm_fallback_models: str = _env(
        "FUZZ_LLM_FALLBACK_MODELS",
        "glm-5.3-flash,deepseek-v4.1,agnes-3.0-flash",
    )
    # Candidate seeds requested from the generator per cycle
    gen_batch_size: int = int(_env("FUZZ_GEN_BATCH_SIZE", "20"))
    # Wall-clock budget (seconds) for ONE generate_for_target call across all
    # prompt variants and retries. Prevents a slow/degraded provider from
    # stalling an entire cycle on a single dark target.
    gen_budget_sec: float = float(_env("FUZZ_GEN_BUDGET_SEC", "300"))
    # Max bytes for a generated input (generated inputs must be small)
    max_input_size: int = int(_env("FUZZ_MAX_INPUT_SIZE", "4096"))

    # --- Source context (for surface mode / mapper) ------------------------
    # Optional directory of target source code given to the LLM
    source_dir: str = _env("FUZZ_SOURCE_DIR", "")

    # --- Paths -------------------------------------------------------------
    memory_path: str = _env("FUZZ_MEMORY_PATH", "./agent_memory.json")
    report_path: str = _env("FUZZ_REPORT_PATH", "./cycle_report.json")
    # Durable research notes written by the pre-campaign research step and
    # read back into every cycle's brief.
    research_path: str = _env("FUZZ_RESEARCH_PATH", "./research_notes.json")
    # Max bytes of dark-source excerpts fed to the research LLM call.
    research_source_bytes: int = int(_env("FUZZ_RESEARCH_SOURCE_BYTES", "9000"))

    # --- Corpus minimization (coverage-preserving) -------------------------
    # Directory of the AFL/ASan build (mounted at /out in the container) used
    # for afl-cmin. Empty disables minimization.
    minimize_build_dir: str = _env("FUZZ_MINIMIZE_BUILD_DIR", "")
    minimize_bin: str = _env("FUZZ_MINIMIZE_BIN", "")
    minimize_image: str = _env("FUZZ_MINIMIZE_IMAGE",
                               "gcr.io/oss-fuzz-base/base-runner")
    minimize_timeout: int = int(_env("FUZZ_MINIMIZE_TIMEOUT", "1800"))
    # Target input placeholder style for the cmin command ("@@" argv file).
    minimize_target_cmd: str = _env("FUZZ_MINIMIZE_TARGET_CMD", "")
    # Skip minimization once the queue is already at/below this size.
    minimize_min_queue: int = int(_env("FUZZ_MINIMIZE_MIN_QUEUE", "200"))

    # --- Safety ------------------------------------------------------------
    # Timeout (seconds) for a single target execution
    exec_timeout: int = int(_env("FUZZ_EXEC_TIMEOUT", "5"))
    # Max candidate executions per validation round
    validate_budget: int = int(_env("FUZZ_VALIDATE_BUDGET", "200"))

    # --- Agentic mode ------------------------------------------------------
    # Max tool actions the agent may take per cycle
    agent_max_steps: int = int(_env("FUZZ_AGENT_MAX_STEPS", "15"))

    # --- Directed hunt -----------------------------------------------------
    # How many dark features the `hunt` tool tries before giving up, and how
    # many input-construction attempts per feature. Hunt stops early as soon as
    # one feature is unlocked (adds coverage).
    hunt_max_targets: int = int(_env("FUZZ_HUNT_MAX_TARGETS", "4"))
    hunt_attempts_per_target: int = int(_env("FUZZ_HUNT_ATTEMPTS_PER_TARGET", "2"))
    # Max candidate files replayed through the coverage build per immediate
    # function-level check (keeps the check fast).
    hunt_replay_max: int = int(_env("FUZZ_HUNT_REPLAY_MAX", "12"))
    # Wall-clock budget (seconds) for one `hunt` call. Each attempt costs a
    # coverage replay AND ~1 LLM call, so this bounds the tool's runtime within
    # the cycle.
    hunt_budget_sec: float = float(_env("FUZZ_HUNT_BUDGET_SEC", "420"))

    # --- Skip-agent heuristic ---------------------------------------------
    # If the campaign's coverage grew more than this percent since the previous
    # tick, the fuzzer is still making progress on its own: skip the agent this
    # tick and let AFL mutate. 0 disables the skip.
    skip_agent_growth_pct: float = float(_env("FUZZ_SKIP_AGENT_GROWTH_PCT", "2.0"))

    # --- Immediate function-level replay (directed verification) -----------
    # Host directory of the clang-coverage build (a libFuzzer binary with llvm
    # source-based coverage), e.g. .../build/out_<project>_coverage. Used to run
    # candidate inputs RIGHT NOW and confirm the targeted dark function is
    # actually reached before injecting anything.
    coverage_dir: str = _env("FUZZ_COVERAGE_DIR", "")
    # Fuzzer binary name inside coverage_dir (mounted at /out).
    coverage_bin: str = _env("FUZZ_COVERAGE_BIN", "")
    # Docker image providing a matching llvm-profdata/llvm-cov + coverage_helper
    # (the host LLVM is usually too old for the OSS-Fuzz profraw format).
    coverage_image: str = _env("FUZZ_COVERAGE_IMAGE",
                               "gcr.io/oss-fuzz-base/base-runner")
    # Timeout (seconds) for one coverage run+digest.
    coverage_timeout: int = int(_env("FUZZ_COVERAGE_TIMEOUT", "600"))

    # Wall-clock budget for one agent cycle. The langgraph router stops the
    # graph once this elapses so the cycle finalizes gracefully instead of
    # being SIGTERM/SIGKILL-killed by the driver's hard cap.
    cycle_budget_sec: float = float(_env("FUZZ_CYCLE_BUDGET_SEC", "1000"))

    def ensure_dirs(self) -> None:
        for d in (self.corpus_dir, self.staging_dir, self.accepted_dir):
            os.makedirs(d, exist_ok=True)


CONFIG = Config()
