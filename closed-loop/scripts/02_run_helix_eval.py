"""
02_run_helix_eval.py - Automated Benchmark Evaluation across HELIX Harnesses
=============================================================================
Orchestrates execution of 4 representative HELIX AI agent harnesses
(Minimal, Pi Mono, Hermes Agent, OpenCode) across a curated subset of coding tasks
(inspired by LiveCodeBench / SWE-bench Mini).

Records full trajectory traces (prompts, tool calls, thinking steps, results)
to `closed-loop/data/raw_traces/` for downstream Sibling Trajectory extraction.

Usage:
  # Run BUILD phase (Base Model Theta_0):
  python closed-loop/scripts/02_run_helix_eval.py --phase build --limit 10

  # Run REBUILD phase (LoRA-adapted Model Theta_1):
  python closed-loop/scripts/02_run_helix_eval.py --phase rebuild --model helix-lora-adapter --limit 10
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLOSED_LOOP_ROOT = PROJECT_ROOT / "closed-loop"
CONFIGS_DIR = CLOSED_LOOP_ROOT / "configs"
RAW_TRACES_DIR = CLOSED_LOOP_ROOT / "data" / "raw_traces"


# ------------------------------------------------------------------------------
# 1. Benchmark Task Subset (LiveCodeBench-style representative tasks)
# ------------------------------------------------------------------------------
SAMPLE_BENCHMARK_TASKS: List[Dict[str, Any]] = [
    {
        "task_id": "lcb_001_two_sum",
        "difficulty": "easy",
        "title": "Two Sum with Indices",
        "prompt": (
            "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` "
            "that returns the 0-based indices of the two numbers such that they add up to target. "
            "Assume exactly one valid answer exists. Return valid Python code inside ```python ```."
        ),
        "test_code": (
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n"
            "assert two_sum([3, 2, 4], 6) == [1, 2]\n"
            "assert two_sum([3, 3], 6) == [0, 1]\n"
            "print('ALL_PASSED')"
        ),
    },
    {
        "task_id": "lcb_002_valid_parentheses",
        "difficulty": "easy",
        "title": "Valid Parentheses String",
        "prompt": (
            "Write a Python function `is_valid(s: str) -> bool` that determines if the input string "
            "has valid matched brackets '()', '{}', '[]'. Return only clean Python code in ```python ```."
        ),
        "test_code": (
            "assert is_valid('()[]{}') == True\n"
            "assert is_valid('(]') == False\n"
            "assert is_valid('([)]') == False\n"
            "assert is_valid('{[]}') == True\n"
            "print('ALL_PASSED')"
        ),
    },
    {
        "task_id": "lcb_003_lru_cache",
        "difficulty": "medium",
        "title": "LRU Cache Implementation",
        "prompt": (
            "Implement an `LRUCache(capacity: int)` class in Python with `get(key: int) -> int` "
            "and `put(key: int, value: int) -> None` in O(1) average time complexity. "
            "Return valid Python code inside ```python ```."
        ),
        "test_code": (
            "cache = LRUCache(2)\n"
            "cache.put(1, 1)\n"
            "cache.put(2, 2)\n"
            "assert cache.get(1) == 1\n"
            "cache.put(3, 3)\n"
            "assert cache.get(2) == -1\n"
            "print('ALL_PASSED')"
        ),
    },
    {
        "task_id": "lcb_004_longest_palindromic_sub",
        "difficulty": "medium",
        "title": "Longest Palindromic Substring",
        "prompt": (
            "Write a Python function `longest_palindrome(s: str) -> str` that returns the longest "
            "palindromic substring in s. Provide optimal clean Python code inside ```python ```."
        ),
        "test_code": (
            "assert longest_palindrome('babad') in ['bab', 'aba']\n"
            "assert longest_palindrome('cbbd') == 'bb'\n"
            "assert longest_palindrome('a') == 'a'\n"
            "print('ALL_PASSED')"
        ),
    },
    {
        "task_id": "lcb_005_merge_intervals",
        "difficulty": "medium",
        "title": "Merge Overlapping Intervals",
        "prompt": (
            "Write a Python function `merge(intervals: list[list[int]]) -> list[list[int]]` that merges "
            "all overlapping intervals and returns an array of non-overlapping intervals sorted by start time. "
            "Provide clean Python code in ```python ```."
        ),
        "test_code": (
            "assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
            "assert merge([[1,4],[4,5]]) == [[1,5]]\n"
            "assert merge([]) == []\n"
            "print('ALL_PASSED')"
        ),
    },
]


# ------------------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------------------
def load_harnesses_config() -> List[Dict[str, Any]]:
    config_path = CONFIGS_DIR / "harnesses.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing harnesses config at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("harnesses", [])


def extract_python_code(text: str) -> str:
    """Extracts python code blocks from markdown text or raw string."""
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return text.strip()


def run_unit_tests(code: str, test_code: str) -> Dict[str, Any]:
    """Runs generated python code against the task's assertions in a sandboxed subprocess."""
    full_script = f"{code}\n\n# --- UNIT TESTS ---\n{test_code}\n"
    start_time = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        duration = round(time.time() - start_time, 3)
        passed = (proc.returncode == 0) and ("ALL_PASSED" in proc.stdout)
        return {
            "passed": passed,
            "duration_sec": duration,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "duration_sec": 10.0,
            "stdout": "",
            "stderr": "Execution timed out (10s)",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "passed": False,
            "duration_sec": 0.0,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


# ------------------------------------------------------------------------------
# 3. HELIX Subprocess Execution
# ------------------------------------------------------------------------------
def invoke_helix_cli(
    harness_id: str,
    prompt: str,
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """
    Executes HELIX harness via `npm run helix -- run <product>` command.
    Captures JSON trajectory response.
    """
    # Map 'minimal' to 'pi-mono' or execute minimal recipe inspection if needed,
    # as CLI run accepts opencode, pi-mono, nanobot, hermes-agent.
    cli_product = "pi-mono" if harness_id == "minimal" else harness_id
    if cli_product not in ["opencode", "pi-mono", "hermes-agent", "nanobot"]:
        cli_product = "pi-mono"

    cmd = [
        "npm", "run", "helix", "--", "run", cli_product,
        "--provider", provider,
        "--model", model,
        "--prompt", prompt,
        "--max-steps", str(max_steps),
        "--json",
    ]

    if base_url:
        cmd.extend(["--base-url", base_url])
    if api_key:
        cmd.extend(["--api-key", api_key])

    env = os.environ.copy()
    if base_url:
        env["HELIX_LIVE_BASE_URL"] = base_url
    if api_key:
        env["OPENAI_API_KEY"] = api_key
    if model:
        env["HELIX_LIVE_MODEL"] = model
    env["HELIX_LIVE_PROVIDER"] = provider

    start_time = time.time()
    try:
        # Use shell=True on Windows for npm execution
        use_shell = os.name == "nt"
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=env,
            shell=use_shell,
            timeout=180,
        )
        latency = round(time.time() - start_time, 3)

        raw_stdout = proc.stdout.strip()
        # Parse JSON output from HELIX CLI
        parsed_output: Dict[str, Any] = {}
        if raw_stdout:
            try:
                # In case npm prints header logs before JSON
                json_start = raw_stdout.find("{")
                json_end = raw_stdout.rfind("}")
                if json_start != -1 and json_end != -1:
                    json_str = raw_stdout[json_start : json_end + 1]
                    parsed_output = json.loads(json_str)
            except Exception:
                parsed_output = {"raw_stdout": raw_stdout}

        # Extract generated assistant text
        assistant_text = ""
        assistant_parts = parsed_output.get("assistantParts", [])
        if assistant_parts and isinstance(assistant_parts, list):
            for part in assistant_parts:
                if isinstance(part, dict) and "text" in part:
                    assistant_text += part["text"] + "\n"
        elif "transcript" in parsed_output:
            for msg in parsed_output.get("transcript", []):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    assistant_text += msg.get("text", "") + "\n"

        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "latency_sec": latency,
            "steps": parsed_output.get("steps", 1),
            "transcript": parsed_output.get("transcript", []),
            "assistant_text": assistant_text.strip(),
            "raw_output": parsed_output,
            "stderr": proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "latency_sec": 180.0,
            "steps": max_steps,
            "transcript": [],
            "assistant_text": "",
            "raw_output": {},
            "stderr": "HELIX subprocess timed out after 180s",
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "latency_sec": 0.0,
            "steps": 0,
            "transcript": [],
            "assistant_text": "",
            "raw_output": {},
            "stderr": str(e),
        }


# ------------------------------------------------------------------------------
# 4. Main Evaluation Flow
# ------------------------------------------------------------------------------
def run_evaluation(
    phase: str,
    limit: int = 5,
    selected_harness_id: Optional[str] = None,
    override_model: Optional[str] = None,
    override_base_url: Optional[str] = None,
) -> None:
    RAW_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    harnesses = load_harnesses_config()

    if selected_harness_id:
        harnesses = [h for h in harnesses if h["id"] == selected_harness_id]

    provider = os.getenv("HELIX_LIVE_PROVIDER", "openai-compatible")
    model = override_model or os.getenv("HELIX_LIVE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    base_url = override_base_url or os.getenv("HELIX_LIVE_BASE_URL", "http://localhost:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "dummy-key")

    tasks = SAMPLE_BENCHMARK_TASKS[:limit]
    print(f"\n{'='*70}")
    print(f"HELIX CLOSED-LOOP EVALUATION - PHASE: [{phase.upper()}]")
    print(f"Model: {model} | Provider: {provider} | Base URL: {base_url}")
    print(f"Harnesses: {[h['id'] for h in harnesses]} | Tasks count: {len(tasks)}")
    print(f"{'='*70}\n")

    summary_records = []

    for task in tasks:
        task_id = task["task_id"]
        print(f"\n>> Evaluating Task: {task_id} ({task['title']})")

        for harness in harnesses:
            h_id = harness["id"]
            print(f"   |-- Running Harness: [{h_id}] ...", end="", flush=True)

            # Invoke HELIX agent turn
            helix_res = invoke_helix_cli(
                harness_id=h_id,
                prompt=task["prompt"],
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )

            # Extract python solution and verify tests
            code = extract_python_code(helix_res["assistant_text"])
            test_res = run_unit_tests(code, task["test_code"])

            status_str = "PASSED" if test_res["passed"] else "FAILED"
            print(f" {status_str} ({test_res['duration_sec']}s, steps: {helix_res['steps']})")

            # Assemble full trace payload
            trace_record = {
                "phase": phase,
                "timestamp": datetime.utcnow().isoformat(),
                "task_id": task_id,
                "task_title": task["title"],
                "difficulty": task["difficulty"],
                "prompt": task["prompt"],
                "harness_id": h_id,
                "harness_category": harness["category"],
                "model": model,
                "provider": provider,
                "steps": helix_res["steps"],
                "latency_sec": helix_res["latency_sec"],
                "assistant_solution": helix_res["assistant_text"],
                "extracted_code": code,
                "test_result": test_res,
                "passed": test_res["passed"],
                "raw_helix_transcript": helix_res["transcript"],
                "error": helix_res["stderr"] or test_res["stderr"],
            }

            out_filename = f"{phase}_{task_id}_{h_id}.json"
            out_path = RAW_TRACES_DIR / out_filename
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(trace_record, f, indent=2, ensure_ascii=False)

            summary_records.append({
                "phase": phase,
                "task_id": task_id,
                "harness_id": h_id,
                "passed": test_res["passed"],
                "steps": helix_res["steps"],
            })

    print(f"\n[DONE] Successfully saved {len(summary_records)} raw trajectory traces to:")
    print(f"       {RAW_TRACES_DIR}\n")


# ------------------------------------------------------------------------------
# 5. CLI Arguments
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HELIX Evaluation across Harnesses")
    parser.add_argument("--phase", type=str, choices=["build", "rebuild"], default="build",
                        help="Phase of the closed-loop cycle (build = Theta_0, rebuild = Theta_1)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of benchmark tasks to run")
    parser.add_argument("--harness", type=str, default=None,
                        help="Filter by specific harness ID (minimal, pi-mono, hermes-agent, opencode)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Override LLM endpoint base URL")

    args = parser.parse_args()
    run_evaluation(
        phase=args.phase,
        limit=args.limit,
        selected_harness_id=args.harness,
        override_model=args.model,
        override_base_url=args.base_url,
    )
