# Memorized or Memo-rized?

Do LLMs follow the documentation in their prompt, or what they memorised during
pre-training? This project tests that with **knowledge-conflict** experiments on code
operators.

A model answers simple aggregate questions over small tables (e.g. *"What is the total
number of pages of all books?"* → `SELECT SUM(pages) FROM books`) by writing a SQL or
Pandas query. The prompt may contain operator documentation that either matches what
the model already knows, or has been **swapped** so that, for example, the docs for
`MIN()` now describe `AVG()`. Comparing the model's answers and operator-token
probabilities across these settings shows whether it is anchored to its parametric
knowledge or shifts to the in-context document.

## Experiments

| Experiment | What it does | Code |
|---|---|---|
| **Main** | Conflict experiment on off-the-shelf instruct, coder and thinking models (Llama 3.1, Qwen 2.5 / 2.5-Coder, Qwen 3 / 3-Coder), in SQL and Pandas | `experiments/main/` |
| **Unlearning** | The same experiment after gradient-ascent unlearning of the real operator docs | `experiments/unlearning/` |
| **Talkie** | The same experiment on [Talkie](https://talkie-lm.com/), a 13B model trained only on pre-1931 text (so it has never seen SQL), after teaching it the operators via continued pre-training and/or QA LoRA | `experiments/talkie/` |

Each experiment runs four **use cases**:

| Use case | Documentation in the prompt |
|---|---|
| `nosql` | none — the model answers from its own knowledge (baseline) |
| `reinforce` | the real documentation |
| `counterfactual` | swapped documentation (`SQL_swap_N` / `Dataframe_swap_N`) |
| `instruct` | swapped documentation, plus an explicit instruction to follow it |

Two knobs vary the conflict: the swap **version** `N` (which operator permutation is
used; version 50 renames operators instead, e.g. `SUM()` → `TVN()`), and the **doclen**
(how many of the ten sample-code blocks per operator doc go in the prompt: 1, 5 or 10).

Analysis reports, per model: accuracy / abstention / counterfactual-adoption rates,
**knowledge shift** (change in operator-token probability vs. `nosql`), **knowledge
anchor** scores ((cf − orig) / ½(cf + orig)), and token-probability distributions.

## Repository layout

```
.
├── SQLShift_PandasShift/   data: real + swapped operator docs (SQL, Pandas), QA test/train sets,
│                           and the scripts that generate them
├── src/                    shared code: paths, config, LLM wrapper (generation with token probs,
│                           unlearning, Talkie training), SQLite executor
├── experiments/            runners and analysis scripts (main, unlearning, talkie) + prompt templates
├── results/                raw generations (*.json.gz) and analysis tables/plots
│   ├── main/               raw_{sql,pandas}/, analysis_{sql,pandas}/, rebuttal/{swap,rename}/
│   ├── unlearning/
│   └── talkie/
├── talkie/                 the third-party Talkie inference library (not tracked in git)
├── environment.yml         conda env (Python 3.11, PyTorch, CUDA 12.1)
└── requirements.txt        pip dependencies
```

Each main directory has its own README with full details:

- [`experiments/README.md`](experiments/README.md) — every runner's flags, output paths and examples
- [`SQLShift_PandasShift/README.md`](SQLShift_PandasShift/README.md) — data format, swap construction, generators
- [`src/README.md`](src/README.md) — shared modules and who uses them

## Setup

```bash
conda env create -f environment.yml      # creates rag_faiss_env
conda activate rag_faiss_env
pip install -r requirements.txt
```

Gated Hugging Face models (e.g. Llama) need a token in `src/.env`:

```
HF_TOKEN=hf_...
```

For the Talkie experiments, install the Talkie library into the same environment
(`pip install -e talkie/`, see `talkie/README.md`).

Default model and generation/training settings live in `src/config.yaml`; `--llm`
overrides the model on the command line.

## Quick start

Run everything from the repository root as a module so `src` is importable:

```bash
export PYTHONPATH=$(pwd)

# 1. Baseline: no documentation
python -m experiments.main.conflict --llm Qwen/Qwen2.5-7B-Instruct \
    --use_case nosql --top_p 0.9999 --num_rounds 1

# 2. Swapped documentation, SQL, two swap versions, 5 sample-code blocks
python -m experiments.main.conflict --llm Qwen/Qwen2.5-7B-Instruct \
    --use_case counterfactual --top_p 0.9999 --num_rounds 1 \
    --language sql --versions 1 11 --doclens 5

# 3. Score everything under results/main/raw_sql and write tables + plots
python -m experiments.main.analyse_results --language sql
```

Runners skip output files that already exist, so interrupted jobs can simply be
resubmitted.

## Data notes

- The test set is 104 questions over 8 hand-made tables, with SQL and Pandas answers
  (`SQLShift_PandasShift/QuestionAnswer/simple_tables_{sql,pandas}.jsonl`).
- Talkie's QA-LoRA training set (688 examples) uses separate Wikipedia tables from
  WikiSQL, so training and test tables never overlap.
- The generator scripts overwrite existing data and the swap scripts draw new random
  permutations — re-running them changes the data the stored results were produced with.

