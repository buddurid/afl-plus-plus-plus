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
    # Candidate seeds requested from the generator per cycle
    gen_batch_size: int = int(_env("FUZZ_GEN_BATCH_SIZE", "20"))
    # Max bytes for a generated input (generated inputs must be small)
    max_input_size: int = int(_env("FUZZ_MAX_INPUT_SIZE", "4096"))

    # --- Source context (for surface mode / mapper) ------------------------
    # Optional directory of target source code given to the LLM
    source_dir: str = _env("FUZZ_SOURCE_DIR", "")

    # --- Paths -------------------------------------------------------------
    memory_path: str = _env("FUZZ_MEMORY_PATH", "./agent_memory.json")
    report_path: str = _env("FUZZ_REPORT_PATH", "./cycle_report.json")

    # --- Safety ------------------------------------------------------------
    # Timeout (seconds) for a single target execution
    exec_timeout: int = int(_env("FUZZ_EXEC_TIMEOUT", "5"))
    # Max candidate executions per validation round
    validate_budget: int = int(_env("FUZZ_VALIDATE_BUDGET", "200"))

    # --- Agentic mode ------------------------------------------------------
    # Max tool actions the agent may take per cycle
    agent_max_steps: int = int(_env("FUZZ_AGENT_MAX_STEPS", "15"))

    def ensure_dirs(self) -> None:
        for d in (self.corpus_dir, self.staging_dir, self.accepted_dir):
            os.makedirs(d, exist_ok=True)


CONFIG = Config()
