# ReasonForge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoyCoding8/MCP/blob/main/ReasonForge_Colab.ipynb)

**Deterministic math tools for small language models.**

ReasonForge gives small LLMs (8B–32B) access to a verified SymPy computation backend via tool calling.
Instead of relying on the model to compute, all math is delegated to deterministic tools — the model only reasons about *what* to compute and *how* to present results.

## Architecture

```
User Question → LLM (Qwen3) → Tool Calls → SymPy Backend → Verified Results → LLM → Final Answer
```

**Two-phase response pipeline:**
1. **Compute** (`/no_think`): Model calls tools with thinking disabled — forced delegation
2. **Present** (thinking ON): Model reasons about verified results, composes the answer

## Tools

| Tool | Operations | Backend |
|------|-----------|---------| 
| `math_tool` | compute, solve, simplify, factor, expand, gcd, lcm, prime_factors, divisors, mod_inverse, nsolve, crt + SymPy builtins (totient, fibonacci, isprime...) | SymPy |
| `calculus_tool` | differentiate, integrate, limit, series, summation, partial_fraction, trigsimp, ode_solve, laplace | SymPy |
| `matrix_tool` | determinant, inverse, eigenvalues, eigenvectors, rank, rref, transpose, multiply, add, trace, nullspace, columnspace, charpoly, norm, adjugate, solve (Ax=b) | SymPy |
| `statistics_tool` | describe, mean, median, mode, std, variance, correlation, regression, percentile, zscore, skewness, kurtosis, geometric_mean, harmonic_mean | Python stdlib |
| `code_tool` | run, check, ast_inspect — sandboxed Python code execution, syntax checking, and structure analysis | subprocess |

## Project Structure

```
MCP/
├── core.py                    # Shared LLM request logic, expert definitions, tool schemas
├── experts/
│   ├── math/
│   │   ├── server.py          # MCP server entry point (math tools)
│   │   └── tools/
│   │       ├── preprocess.py  # Expression parser (^ → **, implicit multiplication)
│   │       ├── algebra.py     # algebra + number theory
│   │       ├── calculus.py    # derivatives, integrals, ODEs
│   │       ├── matrix.py      # linear algebra
│   │       └── statistics.py  # descriptive & inferential stats
│   └── code/
│       ├── server.py          # MCP server entry point (code execution)
│       └── tools/
│           └── code.py        # Sandboxed Python runner & syntax checker
├── tests/
│   ├── sanity.py              # Tool unit tests (16 checks)
│   ├── benchmark.py           # A/B math benchmark (MATH-500 dataset)
│   └── code_benchmark.py      # A/B code benchmark (HumanEval)
├── ui/
│   ├── app.py                 # Gradio chat interface with two-phase pipeline
│   └── style.css              # Custom UI styles (dark mode, thinking blocks)
├── ReasonForge_Colab.ipynb    # One-click Colab deployment notebook
├── pyproject.toml
├── requirements.txt
└── run_ui.bat                 # Local launcher (Windows)
```

## Quick Start (Local)

```bash
# Requires: Ollama running with a supported model (qwen3:8b, qwen3:32b, etc.)
uv sync
uv run python -m ui.app
# Open at http://localhost:7861
```

## Colab Deployment (GPU)

Open `ReasonForge_Colab.ipynb` in Google Colab Pro with an A100 GPU.
It clones this repo, installs Ollama + `qwen3:32b`, and launches the UI with a public Gradio link.

## Benchmarking

```bash
# Math benchmark — MATH-500 (requires Ollama running)
uv run python -m tests.benchmark --model llama3.2:3b --n 10
uv run python -m tests.benchmark --model qwen3:32b --n 50 --think

# Code benchmark — HumanEval (requires Ollama running)
uv run python -m tests.code_benchmark --model qwen3:8b --n 20
uv run python -m tests.code_benchmark --model qwen3:32b --n 164 --think
```

## Running Sanity Tests

```bash
uv run python -m tests.sanity
```

## Tech Stack

- **LLM Backend:** [Ollama](https://ollama.com) (local) or any OpenAI-compatible API
- **Math Engine:** [SymPy](https://sympy.org) — symbolic computation
- **Math Grading:** [math-verify](https://github.com/huggingface/Math-Verify) — deterministic LaTeX parser (Linux/Colab)
- **Code Grading:** Self-contained HumanEval harness (inspired by [openai/human-eval](https://github.com/openai/human-eval))
- **UI:** [Gradio](https://gradio.app) — chat interface with LaTeX rendering
- **Protocol:** [MCP](https://modelcontextprotocol.io) (Model Context Protocol) compatible
