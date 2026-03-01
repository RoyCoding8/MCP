"""ReasonForge A/B Code Benchmark — HumanEval.

Compares model accuracy with vs without ReasonForge code_tool on
the OpenAI HumanEval dataset (164 Python problems).
"""

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from core import EXPERTS, MAX_ROUNDS, llm_request

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CODE_EXPERT = EXPERTS["Coder"]



def download_humaneval(n: int, seed: int = 42) -> list[dict]:
    """Download HumanEval from HuggingFace and cache locally."""
    cache_file = CACHE_DIR / "humaneval.json"

    if cache_file.exists():
        with open(cache_file) as f:
            problems = json.load(f)
        print(f"  Loaded {len(problems)} problems from cache")
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("  ERROR: 'datasets' package not installed.")
            print("  Run:  uv pip install datasets")
            sys.exit(1)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("  Downloading HumanEval dataset...", end="", flush=True)
        ds = load_dataset("openai_humaneval", split="test")
        problems = [
            {
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            }
            for row in ds
        ]
        print(f" {len(problems)} problems")

        with open(cache_file, "w") as f:
            json.dump(problems, f)

    import random
    rng = random.Random(seed)
    if n < len(problems):
        problems = rng.sample(problems, n)
    return problems




def check_correctness(problem: dict, completion: str, timeout: int = 10) -> dict:
    """Run generated code against HumanEval test suite in a subprocess.

    Constructs: prompt + completion + test + check(entry_point)
    Executes in isolated subprocess with timeout.
    """
    check_program = (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(check_program)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-u", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        passed = result.returncode == 0
        return {
            "task_id": problem["task_id"],
            "passed": passed,
            "result": "passed" if passed else f"failed: {result.stderr[:200]}",
        }
    except subprocess.TimeoutExpired:
        return {
            "task_id": problem["task_id"],
            "passed": False,
            "result": f"timed out ({timeout}s)",
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "passed": False,
            "result": f"error: {e}",
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)




def extract_completion(response: str, prompt: str) -> str:
    """Extract the function body from the model response.

    The model should generate code that completes the function stub in `prompt`.
    We try multiple strategies to extract a clean completion.
    """
    # Strategy 1: If response contains a code block, extract it
    import re
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_blocks:
        code = code_blocks[-1].strip()
        # If the code block contains the full function (including def), extract body
        if code.startswith("def "):
            # Return just the body after the first def line
            lines = code.split("\n")
            # Find the first line after the def header
            body_start = 1
            for i, line in enumerate(lines):
                if line.rstrip().endswith(":") and i == 0:
                    body_start = 1
                    break
                elif line.rstrip().endswith(":"):
                    body_start = i + 1
                    break
            body = "\n".join(lines[body_start:])
            return body + "\n"
        return code + "\n"

    # Strategy 2: If the prompt's function signature appears, take what follows
    # Find the entry function signature in the prompt
    sig_match = re.search(r"(def \w+\(.*?\).*?:)", prompt, re.DOTALL)
    if sig_match:
        sig = sig_match.group(1).split("\n")[0]  # first line of def
        func_name = re.search(r"def (\w+)", sig).group(1)
        # Look for the function in the response
        func_match = re.search(
            rf"def {func_name}\(.*?\).*?:\s*\n(.*?)(?=\ndef |\Z)",
            response,
            re.DOTALL,
        )
        if func_match:
            return func_match.group(1) + "\n"

    # Strategy 3: Just use the raw response as the completion body
    # Indent if not already indented
    lines = response.strip().split("\n")
    if lines and not lines[0].startswith(" ") and not lines[0].startswith("\t"):
        response = textwrap.indent(response.strip(), "    ")
    return response + "\n"






def run_baseline(prompt: str, model: str, url: str, think: bool = True) -> str:
    """Run a problem WITHOUT tools — baseline model completion."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise Python programmer. Complete the given function. "
                "Output ONLY the function body (the code that goes after the def line). "
                "Do NOT include the function signature, imports, or test code. "
                "Wrap your code in a ```python code block."
            ),
        },
        {"role": "user", "content": f"Complete this function:\n\n```python\n{prompt}```"},
    ]
    resp = llm_request(messages, [], model, url, stream=False, think=think)
    return resp["message"]["content"]


def run_reasonforge(
    prompt: str, model: str, url: str, think: bool = True
) -> tuple[str, int, bool]:
    """Run a problem WITH ReasonForge code_tool — tool-augmented completion."""
    expert = CODE_EXPERT
    sys_prompt = (
        expert["system"]
        + "\n\nYou are completing a Python function. "
        "Output ONLY the function body (the code that goes after the def line). "
        "Use code_tool with operation='run' to test your solution before finalizing. "
        "Use code_tool with operation='check' to verify syntax. "
        "Wrap your final answer in a ```python code block."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Complete this function:\n\n```python\n{prompt}```"},
    ]

    used_tools = False
    content = ""
    for round_num in range(1, MAX_ROUNDS + 1):
        resp = llm_request(
            messages, expert["tools"], model, url, stream=False, think=think
        )
        msg = resp["message"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return content, round_num, used_tools

        used_tools = True
        messages.append(msg)

        for tc in tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            if name in expert["dispatch"]:
                try:
                    result = expert["dispatch"][name](**args)
                except Exception as e:
                    result = {"error": f"{name} failed: {e}"}
            else:
                result = {"error": f"Unknown tool: {name}"}

            messages.append({"role": "tool", "content": json.dumps(result)})

    return content, MAX_ROUNDS, used_tools




def main():
    parser = argparse.ArgumentParser(
        description="ReasonForge A/B Code Benchmark (HumanEval)"
    )
    parser.add_argument(
        "--model", default="qwen3:32b", help="Ollama model name (default: qwen3:32b)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434/api/chat",
        help="Ollama endpoint",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of problems to evaluate (default: 20, max: 164)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Execution timeout per problem in seconds (default: 10)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline (no-tools) run.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable thinking mode (for models that support it).",
    )
    args = parser.parse_args()

    print(f"\n{'-' * 56}")
    print(f"  ReasonForge A/B Code Benchmark — {args.model}")
    print(f"  {args.n} problems · seed={args.seed}")
    print(f"{'-' * 56}\n")

    problems = download_humaneval(args.n, args.seed)
    n = len(problems)
    print(f"  Evaluating {n} problems\n")

    baseline_pass = 0
    rf_pass = 0
    delegation_count = 0
    total_rounds = 0
    results = []
    t_start = time.time()

    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        label = f"[{i+1:>{len(str(n))}}/{n}] {task_id}"
        print(f"  {label}  ", end="", flush=True)

        b_ok = False
        rf_ok = False
        rounds = 0
        used = False
        t0 = time.time()

        if not args.skip_baseline:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                b_future = pool.submit(
                    run_baseline, prob["prompt"], args.model, args.url, think=args.think
                )
                rf_future = pool.submit(
                    run_reasonforge, prob["prompt"], args.model, args.url, think=args.think
                )

                try:
                    b_resp = b_future.result()
                    b_completion = extract_completion(b_resp, prob["prompt"])
                    b_result = check_correctness(prob, b_completion, timeout=args.timeout)
                    b_ok = b_result["passed"]
                    baseline_pass += b_ok
                except Exception as e:
                    print(f"B:ERR({e}) ", end="")

                try:
                    rf_resp, rounds, used = rf_future.result()
                    rf_completion = extract_completion(rf_resp, prob["prompt"])
                    rf_result = check_correctness(prob, rf_completion, timeout=args.timeout)
                    rf_ok = rf_result["passed"]
                    rf_pass += rf_ok
                    delegation_count += used
                    total_rounds += rounds
                except Exception as e:
                    print(f"RF:ERR({e}) ", end="")
        else:
            try:
                rf_resp, rounds, used = run_reasonforge(
                    prob["prompt"], args.model, args.url, think=args.think
                )
                rf_completion = extract_completion(rf_resp, prob["prompt"])
                rf_result = check_correctness(prob, rf_completion, timeout=args.timeout)
                rf_ok = rf_result["passed"]
                rf_pass += rf_ok
                delegation_count += used
                total_rounds += rounds
            except Exception as e:
                print(f"RF:ERR({e}) ", end="")

        status = ""
        if not args.skip_baseline:
            status += f"B:{'✓' if b_ok else '✗'} "
        status += f"RF:{'✓' if rf_ok else '✗'}"
        if used:
            status += " T"
        status += f" R{rounds}"

        if not args.skip_baseline and rf_ok and not b_ok:
            status += " ★"
        elif not args.skip_baseline and b_ok and not rf_ok:
            status += " ▼"
        dt = time.time() - t0
        elapsed = time.time() - t_start
        status += f"  {dt:.1f}s  ({elapsed:.0f}s)"

        print(status)

        results.append(
            {
                "index": i,
                "task_id": task_id,
                "baseline_passed": b_ok,
                "rf_passed": rf_ok,
                "rf_rounds": rounds,
                "rf_used_tools": used,
            }
        )



    print(f"\n{'-' * 56}")
    print(f"  Results — {args.model} — {n} problems (HumanEval)")
    print(f"{'-' * 56}\n")

    if not args.skip_baseline:
        b_pct = baseline_pass / n if n else 0
        r_pct = rf_pass / n if n else 0
        delta = r_pct - b_pct

        print(f"  {'':18} {'Baseline':>10}  {'ReasonForge':>12}")
        print(f"  {'Pass@1:':18} {baseline_pass:>7}/{n}  {rf_pass:>9}/{n}")
        print(f"  {'Accuracy:':18} {b_pct:>9.1%}  {r_pct:>11.1%}")
        arrow = "    ▲" if delta >= 0 else "    ▼"
        print(f"  {'':18} {'':>10}  {arrow} {delta:+.1%}")
    else:
        r_pct = rf_pass / n if n else 0
        print(f"  ReasonForge Pass@1:  {rf_pass}/{n} ({r_pct:.1%})")

    d_pct = delegation_count / n if n else 0
    avg_r = total_rounds / n if n else 0
    print(f"\n  Delegation:   {delegation_count}/{n} ({d_pct:.1%}) used tools")
    print(f"  Avg Rounds:   {avg_r:.1f}")

    # Show which problems benefitted from tools
    if not args.skip_baseline:
        wins = [r for r in results if r["rf_passed"] and not r["baseline_passed"]]
        losses = [r for r in results if r["baseline_passed"] and not r["rf_passed"]]
        if wins:
            print(f"\n  ★ ReasonForge wins ({len(wins)}):")
            for r in wins:
                print(f"    {r['task_id']}")
        if losses:
            print(f"\n  ▼ ReasonForge losses ({len(losses)}):")
            for r in losses:
                print(f"    {r['task_id']}")



    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace(":", "_").replace("/", "_")
    out_file = RESULTS_DIR / f"code_{model_safe}_{ts}.json"

    report = {
        "benchmark": "humaneval",
        "model": args.model,
        "n": n,
        "seed": args.seed,
        "timestamp": ts,
        "baseline_pass1": baseline_pass / n if (not args.skip_baseline and n) else None,
        "rf_pass1": rf_pass / n if n else 0,
        "delta": (rf_pass - baseline_pass) / n
        if (not args.skip_baseline and n)
        else None,
        "delegation_rate": d_pct,
        "avg_rounds": avg_r,
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Results → {out_file}")
    print(f"{'-' * 56}\n")


if __name__ == "__main__":
    main()
