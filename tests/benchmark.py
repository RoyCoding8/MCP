"""
ReasonForge A/B Benchmark — MATH-500

Compares model accuracy WITH vs WITHOUT ReasonForge tools on
competition-level math problems from the Hendrycks MATH dataset.

Usage:
  uv run python -m tests.benchmark --model qwen3:8b --n 50
  uv run python -m tests.benchmark --model qwen3:32b --n 500
  uv run python -m tests.benchmark --model qwen3:8b --n 50 --skip-baseline
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

from core import EXPERTS, MAX_ROUNDS, llm_request

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MATH_EXPERT = EXPERTS["Mathematician"]

def download_math(n: int, seed: int = 42) -> list[dict]:
    cache_file = CACHE_DIR / "math_test.json"

    if cache_file.exists():
        with open(cache_file) as f:
            problems = json.load(f)
        print(f"  Loaded {len(problems)} problems from cache")
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("  ERROR: 'datasets' package not installed.")
            print("  Run:  uv add datasets")
            sys.exit(1)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("  Downloading MATH dataset...", end="", flush=True)
        ds = load_dataset("qwedsacf/competition_math", split="train")
        problems = [
            {"problem": row["problem"], "solution": row["solution"],
             "level": row["level"], "type": row["type"]}
            for row in ds
        ]
        print(f" {len(problems)} problems")

        with open(cache_file, "w") as f:
            json.dump(problems, f)

    rng = random.Random(seed)
    if n < len(problems):
        problems = rng.sample(problems, n)
    return problems


def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} from LaTeX text, handling nested braces."""
    if not text:
        return None
    # Walk backwards to find the last \boxed{
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Also try \boxed without braces (rare)
        m = re.search(r"\\boxed\s+(\S+)", text)
        return m.group(1).strip() if m else None

    # Find matching closing brace
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    # Unmatched — take what we have
    return text[start:].strip()


def extract_answer(text: str) -> str | None:
    """Extract the final answer from a model response."""
    # Try \boxed{} first
    ans = extract_boxed(text)
    if ans:
        return ans
    # Try "the answer is X" patterns
    for pattern in [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)",
        r"(?:=|equals)\s*(.+?)(?:\.|$)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def normalize(ans: str) -> str:
    """Normalize an answer string for comparison."""
    if not ans:
        return ""
    s = ans.strip()
    # Remove common LaTeX wrappers
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\text{", "").replace("\\mathrm{", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\ ", " ")
    s = s.replace("\\%", "%")
    # Normalize fractions: \frac{a}{b} → a/b, \dfrac{a}{b} → a/b
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    # Normalize sqrt: \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    # Remove remaining backslashes before known commands
    s = re.sub(r"\\(pi|infty|cdot|times|div|pm|mp)", r"\1", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def grade(predicted: str | None, expected: str | None) -> bool:
    if predicted is None or expected is None:
        return False

    p = normalize(predicted)
    e = normalize(expected)

    if p == e:
        return True

    try:
        pf = float(p.replace(",", ""))
        ef = float(e.replace(",", ""))
        if abs(pf - ef) < 1e-6:
            return True
    except (ValueError, OverflowError):
        pass

    try:
        from sympy import simplify, sympify
        from sympy.parsing.sympy_parser import parse_expr

        p_expr = parse_expr(p.replace("^", "**"))
        e_expr = parse_expr(e.replace("^", "**"))
        if simplify(p_expr - e_expr) == 0:
            return True
    except Exception:
        pass

    if e in p or p in e:
        return True

    return False


def _stream_to_msg(resp) -> dict:
    """Consume a streaming Ollama response and return the final message dict."""
    content = ""
    tool_calls = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        chunk = json.loads(line)
        msg = chunk.get("message", {})
        content += msg.get("content", "")
        if msg.get("tool_calls"):
            tool_calls.extend(msg["tool_calls"])
        if chunk.get("done"):
            break
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def run_baseline(question: str, model: str, url: str, think: bool = True) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. Solve the problem step by step. "
                "Put your final answer in \\boxed{}."
            ),
        },
        {"role": "user", "content": question},
    ]
    resp = llm_request(messages, [], model, url, stream=True, think=think)
    msg = _stream_to_msg(resp)
    return msg["content"]


def run_reasonforge(question: str, model: str, url: str, think: bool = True) -> tuple[str, int, bool]:
    expert = MATH_EXPERT
    sys_prompt = (
        expert["system"]
        + "\n\nAlways put your final answer in \\boxed{}. "
        "Use tools for ALL computations — never compute in your head."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]

    used_tools = False
    for round_num in range(1, MAX_ROUNDS + 1):
        resp = llm_request(
            messages, expert["tools"], model, url, stream=True, think=think
        )
        msg = _stream_to_msg(resp)
        content = msg["content"]
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
        description="ReasonForge A/B Benchmark (MATH-500)"
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
        "--n", type=int, default=50, help="Number of problems to evaluate (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline (no-tools) run. Useful if you already have baseline results.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable thinking mode. Omit this for models that don't support it (e.g. llama3.2).",
    )
    args = parser.parse_args()
    print(f"\n{'-' * 56}")
    print(f"  ReasonForge A/B Benchmark — {args.model}")
    print(f"  {args.n} problems · seed={args.seed}")
    print(f"{'-' * 56}\n")

    problems = download_math(args.n, args.seed)
    n = len(problems)
    print(f"  Evaluating {n} problems\n")

    baseline_correct = 0
    rf_correct = 0
    baseline_score = 0
    rf_score = 0
    total_score = 0
    delegation_count = 0
    total_rounds = 0
    results = []

    for i, prob in enumerate(problems):
        expected = extract_boxed(prob["solution"])
        level_num = prob["level"].replace("Level ", "") if prob["level"] else "?"
        try:
            weight = int(level_num)
        except ValueError:
            weight = 1
        total_score += weight

        label = f"[{i+1:>{len(str(n))}}/{n}] {prob['type']:<20} L{level_num}"
        print(f"  {label}  ", end="", flush=True)

        b_ans, b_ok = None, False
        if not args.skip_baseline:
            try:
                b_resp = run_baseline(prob["problem"], args.model, args.url, think=args.think)
                b_ans = extract_answer(b_resp)
                b_ok = grade(b_ans, expected)
                baseline_correct += b_ok
                if b_ok:
                    baseline_score += weight
            except Exception as e:
                print(f"B:ERR({e}) ", end="")

        rf_ans, rf_ok, rounds, used = None, False, 0, False
        try:
            rf_resp, rounds, used = run_reasonforge(
                prob["problem"], args.model, args.url, think=args.think
            )
            rf_ans = extract_answer(rf_resp)
            rf_ok = grade(rf_ans, expected)
            rf_correct += rf_ok
            if rf_ok:
                rf_score += weight
            delegation_count += used
            total_rounds += rounds
        except Exception as e:
            print(f"RF:ERR({e}) ", end="")

        status = ""
        if not args.skip_baseline:
            status += f"B:{'✓' if b_ok else '✗'} "
        status += f"RF:{'✓' if rf_ok else '✗'}"
        if used:
            status += f" T"
        status += f" R{rounds}"

        if not args.skip_baseline and rf_ok and not b_ok:
            status += " ★"
        elif not args.skip_baseline and b_ok and not rf_ok:
            status += " ▼"

        print(status)

        results.append({
            "index": i,
            "type": prob["type"],
            "level": prob["level"],
            "problem": prob["problem"][:200],
            "expected": expected,
            "baseline_answer": b_ans,
            "baseline_correct": b_ok,
            "rf_answer": rf_ans,
            "rf_correct": rf_ok,
            "rf_rounds": rounds,
            "rf_used_tools": used,
            "weight": weight,
        })

    print(f"\n{'-' * 56}")
    print(f"  Results — {args.model} — {n} problems")
    print(f"{'-' * 56}\n")

    if not args.skip_baseline:
        b_pct = baseline_correct / n if n else 0
        r_pct = rf_correct / n if n else 0
        delta = r_pct - b_pct
        b_score_pct = baseline_score / total_score if total_score else 0
        r_score_pct = rf_score / total_score if total_score else 0
        score_delta = r_score_pct - b_score_pct
        
        print(f"  {'':18} {'Baseline':>10}  {'ReasonForge':>12}")
        print(f"  {'Correct:':18} {baseline_correct:>7}/{n}  {rf_correct:>9}/{n}")
        print(f"  {'Uniform Acc:':18} {b_pct:>9.1%}  {r_pct:>11.1%}")
        arrow = "    ▲" if delta >= 0 else "    ▼"
        print(f"  {'':18} {'':>10}  {arrow} {delta:+.1%}")
        
        print(f"  {'Weighted Score:':18} {baseline_score:>7}/{total_score}  {rf_score:>9}/{total_score}")
        print(f"  {'Weighted Acc:':18} {b_score_pct:>9.1%}  {r_score_pct:>11.1%}")
        score_arrow = "    ▲" if score_delta >= 0 else "    ▼"
        print(f"  {'':18} {'':>10}  {score_arrow} {score_delta:+.1%}")
    else:
        r_pct = rf_correct / n if n else 0
        r_score_pct = rf_score / total_score if total_score else 0
        print(f"  ReasonForge (Uniform):  {rf_correct}/{n} ({r_pct:.1%})")
        print(f"  ReasonForge (Weighted): {rf_score}/{total_score} ({r_score_pct:.1%})")

    d_pct = delegation_count / n if n else 0
    avg_r = total_rounds / n if n else 0
    print(f"\n  Delegation:   {delegation_count}/{n} ({d_pct:.1%}) used tools")
    print(f"  Avg Rounds:   {avg_r:.1f}")

    print(f"\n  By difficulty:")
    for lvl in sorted(set(r["level"] for r in results)):
        lvl_results = [r for r in results if r["level"] == lvl]
        lvl_rf = sum(1 for r in lvl_results if r["rf_correct"])
        lvl_n = len(lvl_results)
        bar = "█" * int(lvl_rf / lvl_n * 20) if lvl_n else ""
        line = f"    {lvl:<10} {lvl_rf:>3}/{lvl_n:<3} {lvl_rf/lvl_n:.0%}  {bar}"
        if not args.skip_baseline:
            lvl_b = sum(1 for r in lvl_results if r["baseline_correct"])
            delta_l = (lvl_rf - lvl_b) / lvl_n if lvl_n else 0
            if delta_l != 0:
                line += f"  ({'+' if delta_l > 0 else ''}{delta_l:.0%})"
        print(line)

    print(f"\n  By category:")
    for typ in sorted(set(r["type"] for r in results)):
        typ_results = [r for r in results if r["type"] == typ]
        typ_rf = sum(1 for r in typ_results if r["rf_correct"])
        typ_n = len(typ_results)
        bar = "█" * int(typ_rf / typ_n * 20) if typ_n else ""
        line = f"    {typ:<24} {typ_rf:>3}/{typ_n:<3} {typ_rf/typ_n:.0%}  {bar}"
        if not args.skip_baseline:
            typ_b = sum(1 for r in typ_results if r["baseline_correct"])
            delta_t = (typ_rf - typ_b) / typ_n if typ_n else 0
            if delta_t != 0:
                line += f"  ({'+' if delta_t > 0 else ''}{delta_t:.0%})"
        print(line)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace(":", "_").replace("/", "_")
    out_file = RESULTS_DIR / f"{model_safe}_{ts}.json"

    report = {
        "model": args.model,
        "n": n,
        "seed": args.seed,
        "timestamp": ts,
        "baseline_accuracy": baseline_correct / n if (not args.skip_baseline and n) else None,
        "rf_accuracy": rf_correct / n if n else 0,
        "delta": (rf_correct - baseline_correct) / n if (not args.skip_baseline and n) else None,
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
