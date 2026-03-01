# Benchmarking ReasonForge

## Quick Start

```bash
# 1. Sanity check (no LLM needed, <2 seconds)
uv run python -m tests.sanity

# 2. Quick benchmark (50 problems, ~30 min on 8B)
uv run python -m tests.benchmark --model qwen3:8b --n 50

# 3. Full benchmark (500 problems, ~2-3 hours on 32B)
uv run python -m tests.benchmark --model qwen3:32b --n 500
```

## What It Tests

The benchmark runs an **A/B comparison** on competition-level math problems from the [MATH dataset](https://github.com/hendrycks/math) (Hendrycks et al.):

| Mode | Tools | Description |
|------|-------|-------------|
| **Baseline** | None | Raw model with chain-of-thought |
| **ReasonForge** | SymPy tools | Model delegates computation to `math_tool`, `calculus_tool`, etc. |

Both modes use the same model, same thinking mode, same prompting style. The only difference is tool access.

## Reading the Output

```
  [1/50] Algebra              L3  B:✗ RF:✓ T R2 ★
  [2/50] Number Theory        L4  B:✓ RF:✓ T R1
  [3/50] Precalculus          L5  B:✓ RF:✗ R1 ▼
```

| Symbol | Meaning |
|--------|---------|
| `B:✓/✗` | Baseline correct/wrong |
| `RF:✓/✗` | ReasonForge correct/wrong |
| `T` | Used tools |
| `R2` | Took 2 rounds (model↔tool) |
| `★` | Tool **helped** (baseline wrong, RF correct) |
| `▼` | Tool **hurt** (baseline correct, RF wrong) |

## Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | % correct answers |
| **Δ** | ReasonForge accuracy − Baseline accuracy |
| **Delegation Rate** | % of problems where the model used ≥1 tool |
| **Avg Rounds** | Mean model↔tool interactions per problem |

## CLI Options

```
--model MODEL    Ollama model name (default: qwen3:8b)
--url URL        Ollama endpoint (default: http://localhost:11434/api/chat)
--n N            Number of problems (default: 50)
--seed SEED      Random seed for reproducibility (default: 42)
--skip-baseline  Skip baseline run (if you already have it or want speed)
```

## Comparing Models

Results are saved to `tests/results/<model>_<timestamp>.json`. To compare:

```bash
# Run on different models with the same seed
uv run python -m tests.benchmark --model qwen3:8b --n 100 --seed 42
uv run python -m tests.benchmark --model qwen3:32b --n 100 --seed 42
```

Same seed = same problems = fair comparison.

## Published Baselines (without tools)

| Model | MATH-500 | Source |
|-------|----------|--------|
| Qwen3-8B (thinking) | ~62% | Qwen blog |
| Qwen3-32B (thinking) | ~79% | Qwen blog |
| Qwen2.5-Math-7B | ~55% | Qwen paper |
| Llama-3.1-8B | ~51% | Meta |

## FAQ

**Q: How long does a full run take?**
- 50 problems on 8B: ~20-40 minutes
- 50 problems on 32B: ~30-60 minutes
- 500 problems on 32B: ~3-5 hours
- A/B mode doubles elapsed time (runs each problem twice)

**Q: Can I run just ReasonForge (skip baseline)?**
Yes: `--skip-baseline` skips the raw model run. Compare against published baselines from the table above.

**Q: The dataset didn't download?**
The benchmark downloads from HuggingFace's API. If it fails, manually download:
```bash
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('hendrycks/competition_math', split='test'); ds.to_json('tests/.cache/math_test.json')"
```

**Q: How is grading done?**
1. Extract `\boxed{answer}` from model output
2. Normalize LaTeX formatting
3. Compare via exact match → numeric match → SymPy symbolic comparison
