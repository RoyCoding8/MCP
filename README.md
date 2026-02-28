# ReasonForge

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

## Project Structure

```
MCP/
├── experts/math/
│   ├── server.py              # MCP server entry point
│   └── tools/
│       ├── preprocess.py      # Expression parser (^ → **, implicit multiplication, infinity handling)
│       ├── algebra.py         # math_tool — algebra + number theory
│       ├── calculus.py        # calculus_tool — derivatives, integrals, ODEs, etc.
│       ├── matrix.py          # matrix_tool — linear algebra
│       └── statistics.py      # statistics_tool — descriptive & inferential stats
├── tests/
│   └── test_math_tools.py     # Unit tests for all tools
├── ui/
│   ├── app.py                 # Gradio chat interface with two-phase pipeline
│   └── style.css              # Custom UI styles (dark mode, input pill, etc.)
├── ReasonForge_Colab.ipynb    # One-click Colab deployment notebook
├── pyproject.toml
└── run_ui.bat                 # Local launcher (Windows)
```

## Quick Start (Local)

```bash
# Requires: Ollama running with qwen3:8b
uv sync
uv run python -m ui.app
# Opens at http://localhost:7861
```

## Colab Deployment (GPU)

Open `ReasonForge_Colab.ipynb` in Google Colab Pro with an A100 GPU.
It clones this repo, installs Ollama + `qwen3:32b`, and launches the UI with a public Gradio link.

## Running Tests

```bash
uv run python -m tests.test_math_tools
```

## Tech Stack

- **LLM Backend:** [Ollama](https://ollama.com) (local) or any OpenAI-compatible API
- **Math Engine:** [SymPy](https://sympy.org) — symbolic computation
- **UI:** [Gradio](https://gradio.app) — chat interface with LaTeX rendering
- **Protocol:** [MCP](https://modelcontextprotocol.io) (Model Context Protocol) compatible
