"""ReasonForge A/B Benchmark — MATH-500.

Compares model accuracy with vs without ReasonForge tools on
competition-level math problems from the Hendrycks MATH dataset.
"""

import argparse
import json
import random
import re
import sys
import time

from pathlib import Path

from tqdm import tqdm

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
    
    # Normalize
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\(pi|infty|cdot|times|div|pm|mp)", r"\1", s)
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
    resp = llm_request(messages, [], model, url, stream=False, think=think)
    return resp["message"]["content"]


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


def _load_checkpoint(path):
    """Load completed results from a JSONL checkpoint file."""
    results = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def _append_checkpoint(path, record):
    """Append a single result to a JSONL checkpoint file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds)."""
    t = time.time()
    result = fn(*args, **kwargs)
    return result, round(time.time() - t, 1)


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
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Persistent results directory (e.g. Google Drive). Enables checkpoint/resume.",
    )
    args = parser.parse_args()
    print(f"  ReasonForge A/B Benchmark — {args.model}")
    print(f"  {args.n} problems · seed={args.seed}")
    print(f"{'-' * 56}\n")

    problems = download_math(args.n, args.seed)
    n = len(problems)

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    model_safe = args.model.replace(":", "_").replace("/", "_")
    checkpoint_file = results_dir / f"math_{model_safe}_s{args.seed}.jsonl"

    prior = _load_checkpoint(checkpoint_file)
    done_indices = {r["index"] for r in prior}
    results = list(prior)
    if done_indices:
        print(f"  Resuming: {len(done_indices)}/{n} already completed\n")
        for r in sorted(prior, key=lambda x: x["index"]):
            lvl = r.get("level", "").replace("Level ", "") if r.get("level") else "?"
            label = f"[{r['index']+1:>{len(str(n))}}/{n}] {r.get('type',''):<20} L{lvl}"
            status = ""
            if r.get("baseline_answer") is not None or r.get("baseline_correct"):
                status += f"B:{'✓' if r.get('baseline_correct') else '✗'} "
            status += f"RF:{'✓' if r.get('rf_correct') else '✗'}"
            if r.get("rf_used_tools"): status += " T"
            status += f" R{r.get('rf_rounds', 0)}"
            if r.get("baseline_time_s") is not None and r.get("rf_time_s") is not None:
                status += f"  B:{r['baseline_time_s']:.1f}s RF:{r['rf_time_s']:.1f}s"
            elif r.get("rf_time_s") is not None:
                status += f"  RF:{r['rf_time_s']:.1f}s"
            print(f"  {label}  {status}  (cached)")
        print()
    print(f"  Evaluating {n} problems ({len(done_indices)} cached, {n - len(done_indices)} remaining)\n")

    baseline_correct = sum(1 for r in prior if r.get("baseline_correct"))
    rf_correct = sum(1 for r in prior if r.get("rf_correct"))
    baseline_score = sum(r.get("weight", 1) for r in prior if r.get("baseline_correct"))
    rf_score = sum(r.get("weight", 1) for r in prior if r.get("rf_correct"))
    total_score = sum(r.get("weight", 1) for r in prior)
    delegation_count = sum(1 for r in prior if r.get("rf_used_tools"))
    total_rounds = sum(r.get("rf_rounds", 0) for r in prior)
    t_start = time.time()

    remaining = [(i, p) for i, p in enumerate(problems) if i not in done_indices]
    pbar = tqdm(remaining, desc=args.model, unit="prob",
                initial=len(done_indices), total=n)

    for i, prob in pbar:

        expected = extract_boxed(prob["solution"])
        level_num = prob["level"].replace("Level ", "") if prob["level"] else "?"
        try:
            weight = int(level_num)
        except ValueError:
            weight = 1
        total_score += weight

        label = f"[{i+1:>{len(str(n))}}/{n}] {prob['type']:<20} L{level_num}"
        detail = f"  {label}  "

        b_ans, b_ok = None, False
        rf_ans, rf_ok, rounds, used = None, False, 0, False
        b_time, rf_time = None, None
        t0 = time.time()

        if not args.skip_baseline:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                b_future = pool.submit(
                    _timed, run_baseline, prob["problem"], args.model, args.url, think=args.think
                )
                rf_future = pool.submit(
                    _timed, run_reasonforge, prob["problem"], args.model, args.url, think=args.think
                )

                try:
                    b_resp, b_time = b_future.result()
                    b_ans = extract_answer(b_resp)
                    b_ok = grade(b_ans, expected)
                    baseline_correct += b_ok
                    if b_ok:
                        baseline_score += weight
                except Exception as e:
                    detail += f"B:ERR({e}) "

                try:
                    (rf_resp, rounds, used), rf_time = rf_future.result()
                    rf_ans = extract_answer(rf_resp)
                    rf_ok = grade(rf_ans, expected)
                    rf_correct += rf_ok
                    if rf_ok:
                        rf_score += weight
                    delegation_count += used
                    total_rounds += rounds
                except Exception as e:
                    detail += f"RF:ERR({e}) "
        else:
            try:
                (rf_resp, rounds, used), rf_time = _timed(
                    run_reasonforge, prob["problem"], args.model, args.url, think=args.think
                )
                rf_ans = extract_answer(rf_resp)
                rf_ok = grade(rf_ans, expected)
                rf_correct += rf_ok
                if rf_ok:
                    rf_score += weight
                delegation_count += used
                total_rounds += rounds
            except Exception as e:
                detail += f"RF:ERR({e}) "

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
        dt = time.time() - t0
        if b_time is not None and rf_time is not None:
            status += f"  B:{b_time:.1f}s RF:{rf_time:.1f}s"
        elif rf_time is not None:
            status += f"  RF:{rf_time:.1f}s"
        else:
            status += f"  {dt:.1f}s"

        detail += status
        tqdm.write(detail)

        done_so_far = len(done_indices) + len(results) - len(prior) + 1
        rf_pct = rf_correct / done_so_far if done_so_far else 0
        pbar.set_postfix(rf=f"{rf_pct:.0%}", refresh=True)

        record = {
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
            "baseline_time_s": b_time,
            "rf_time_s": rf_time,
        }
        results.append(record)
        _append_checkpoint(checkpoint_file, record)

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
    b_times = [r["baseline_time_s"] for r in results if r.get("baseline_time_s") is not None]
    rf_times = [r["rf_time_s"] for r in results if r.get("rf_time_s") is not None]
    avg_bt = sum(b_times) / len(b_times) if b_times else 0
    avg_rt = sum(rf_times) / len(rf_times) if rf_times else 0
    print(f"\n  Delegation:   {delegation_count}/{n} ({d_pct:.1%}) used tools")
    print(f"  Avg Rounds:   {avg_r:.1f}")
    if b_times:
        print(f"  Avg Time:     B:{avg_bt:.1f}s  RF:{avg_rt:.1f}s  (Δ{avg_rt - avg_bt:+.1f}s)")
    else:
        print(f"  Avg Time:     RF:{avg_rt:.1f}s")

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

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"{model_safe}_{ts}.json"

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
        "avg_baseline_time_s": avg_bt if b_times else None,
        "avg_rf_time_s": avg_rt if rf_times else None,
        "results": results,
    }

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Results → {out_file}")
    print(f"  Checkpoint → {checkpoint_file}")
    print(f"{'-' * 56}\n")


if __name__ == "__main__":
    main()
