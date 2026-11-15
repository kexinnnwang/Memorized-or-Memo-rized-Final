# experiments

Knowledge-conflict experiments. A model answers questions over small tables by
writing a SQL (or Pandas) query, with or without operator documentation in the
prompt. The documentation either matches what the model already knows or has
been swapped (e.g. the docs for `MIN()` now describe `AVG()`), and the
experiments measure whether the model follows its own knowledge or the
document.

```
experiments/
├── main/            the conflict experiment on off-the-shelf models (SQL and Pandas)
│   ├── conflict.py
│   ├── analyse_results.py
│   └── archive/     previous scripts, kept for reference
├── unlearning/      the same experiment after unlearning the operator docs
│   ├── unlearn.py
│   └── analyse_results_unlearn.py
├── talkie/          the same experiment on Talkie checkpoints after training
│   ├── train_talkie.py
│   ├── analyse_results_talkie.py
│   └── archive/     previous scripts, kept for reference
└── prompts/         prompt templates, one per use case (`*_pandas.txt` for Pandas)
```

Data comes from `SQLShift_PandasShift/` and results go to `results/`. All paths are
in `src/paths.py`.

## Use cases

Every runner tests one or more of these:

| Use case | Documentation in the prompt |
|---|---|
| `nosql` | none: the model answers from its own knowledge (the baseline) |
| `reinforce` | the real documentation |
| `counterfactual` | swapped documentation (`SQL_swap_N`) |
| `instruct` | swapped documentation, plus an instruction to follow it |

`--versions` picks which swap tables (`N`) to use and only affects
`counterfactual` and `instruct`. `--doclens` is how many sample-code blocks of
each operator's documentation go in the prompt (1–10), and does not apply to
`nosql`.

## Running

Run everything from the repository root as a module, so `src` can be imported:

```bash
export PYTHONPATH=$(pwd)
python -m experiments.main.conflict ...
```

Flags marked "must be given" have no default; leaving one out makes the script fail
partway through rather than at start-up.

Every runner skips any output file that already exists, so an interrupted job
can be resubmitted. The model is only loaded if there is something left to run.

---

## main

### `conflict.py`

Runs one use case on one model and writes
`results/main/raw_{sql,pandas}/<use_case>/[version_N/][doclen_D/]<model>.json.gz`.

| Flag | Default | Description |
|---|---|---|
| `--llm` | `llm_model` in `src/config.yaml` | Hugging Face model id |
| `--use_case` | none, must be given | `nosql`, `reinforce`, `counterfactual` or `instruct` |
| `--top_p` | none, must be given | nucleus sampling threshold used when logging token probabilities |
| `--num_rounds` | none, must be given | number of passes over the test set |
| `--language` | `sql` | `sql` or `pandas` |
| `--versions` | `1 11 24 35 37 7 42 27 41 17` | swap versions (counterfactual/instruct only) |
| `--doclens` | `1 5 10` | sample-code blocks per doc (not used by nosql) |
| `--batched` | off | generate in batches (`LLM.generate_with_probs_batch`) |
| `--batch_size` | `8` | batch size, used with `--batched` |

```bash
# no documentation
python -m experiments.main.conflict --llm Qwen/Qwen2.5-7B-Instruct --use_case nosql \
    --top_p 0.9999 --num_rounds 1

# real documentation, three doc lengths
python -m experiments.main.conflict --llm Qwen/Qwen2.5-7B-Instruct --use_case reinforce \
    --top_p 0.9999 --num_rounds 1 --doclens 1 5 10

# swapped documentation, SQL, two versions
python -m experiments.main.conflict --llm Qwen/Qwen2.5-7B-Instruct --use_case counterfactual \
    --top_p 0.9999 --num_rounds 1 --language sql --versions 1 11 --doclens 5

# all flags: Pandas, batched
python -m experiments.main.conflict --llm meta-llama/Llama-3.1-8B-Instruct --use_case instruct \
    --top_p 0.9999 --num_rounds 1 --language pandas --versions 1 11 24 35 37 --doclens 5 \
    --batched --batch_size 16
```

### `analyse_results.py`

Scores everything under `results/main/raw_sql` and/or `raw_pandas` and writes
tables and plots to `results/main/analysis_{sql,pandas}/`. Models, versions and
doclens are read from the folders; `archive/` folders are ignored.

| Flag | Default | Description |
|---|---|---|
| `--language` | both | `sql` or `pandas` |
| `--versions` | all found | Only use these counterfactual versions |
| `--doclens` | all found | Only use these doc lengths |
| `--output-dir` | see below | Save results to this folder instead |

```bash
python -m experiments.main.analyse_results                    # SQL and Pandas
python -m experiments.main.analyse_results --language sql
python -m experiments.main.analyse_results --language pandas
python -m experiments.main.analyse_results --language pandas --versions 1 11 --doclens 5
python -m experiments.main.analyse_results --language pandas --output-dir results/main/my_analysis
```

By default, results go to `results/main/analysis_{sql,pandas}/`. Runs using
`--versions` or `--doclens` write to a separate folder named after the subset,
e.g. `results/main/analysis_pandas_v1_11_d5/`. `--output-dir` replaces either
location; when analysing both languages, results go to `sql/` and `pandas/`
subfolders inside it.

Outputs, in each `analysis_*` folder:

| Folder | Contents |
|---|---|
| `accuracy/` | `acc_<model>`: accuracy, abstention and counterfactual-adoption metrics per use case |
| `knowledge_shift/` | `ks_<model>`: change in operator-token probability vs `nosql` |
| `knowledge_anchor/` | `ka_<slots>`, `ka_by_doclen_<slots>`: mean (cf − orig)/0.5(cf + orig) per model |
| `knowledge_anchor_distribution/` | histograms of that ratio |
| `token_distribution/` | histograms of the original and counterfactual token probabilities |

---

## unlearning

### `unlearn.py`

The same as `conflict.py` (SQL, always batched), but first unlearns the real
operator documentation with gradient ascent (`LLM.model_unlearn_ga_qa`), then
tests. Writes `results/unlearning/raw_unlearn/...`.

| Flag | Default | Description |
|---|---|---|
| `--llm` | `llm_model` in `src/config.yaml` | Hugging Face model id |
| `--use_case` | none, must be given | `nosql`, `reinforce`, `counterfactual` or `instruct` |
| `--top_p` | none, must be given | nucleus sampling threshold |
| `--num_rounds` | none, must be given | number of passes over the test set |
| `--versions` | `1 11 24 35 37` | swap versions (counterfactual/instruct only) |
| `--doclens` | `5` | sample-code blocks per doc (not used by nosql) |
| `--batch_size` | `8` | generation batch size |

```bash
python -m experiments.unlearning.unlearn --llm Qwen/Qwen2.5-7B-Instruct --use_case nosql \
    --top_p 0.9999 --num_rounds 1

# all flags
python -m experiments.unlearning.unlearn --llm Qwen/Qwen2.5-7B-Instruct --use_case counterfactual \
    --top_p 0.9999 --num_rounds 1 --versions 1 11 24 35 37 --doclens 5 --batch_size 8
```

### `analyse_results_unlearn.py`

Same tables and plots as the main analysis, for `results/unlearning/raw_unlearn`,
written to `results/unlearning/analysis_unlearn/`. It has no flags; the versions
(`1 11 24 35 37`) and doclen (`5`) are set at the top of the file.

```bash
python -m experiments.unlearning.analyse_results_unlearn
```

---

## talkie

### `train_talkie.py`

Trains Talkie once, then tests every requested use case (and, for
counterfactual/instruct, every version) on the trained model. One job is one
training setting.

- **Phase 1** (`--phase1`): continued pre-training on the real operator docs
  (`LLM.talkie_train_phase1`).
- **Phase 2** (`--phase2`): QA LoRA on `Talkie_Train/simple_tables_sql_wiki.jsonl`
  (`LLM.talkie_train_phase2`).

With neither flag the untrained model is tested. Test prompts always use doclen 1.
Results go to
`results/talkie/raw_talkie/<use_case>/[version_N/]<runtype>/<epochs>/<doclen>/<model>.json.gz`,
where `<runtype>` is `notrain`, `phase1`, `phase2` or `full`.

| Flag | Default | Description |
|---|---|---|
| `--llm` | `llm_model` in `src/config.yaml` | model name, e.g. `talkie-1930-13b-it` |
| `--phase1` | off | run phase 1 training |
| `--phase2` | off | run phase 2 training |
| `--use_cases` | `nosql reinforce counterfactual instruct` | use cases to test |
| `--top_p` | none, must be given | nucleus sampling threshold |
| `--num_rounds` | none, must be given | number of passes over the test set |
| `--versions` | `1 11 24 35 37` | swap versions (counterfactual/instruct only) |
| `--p1e` | `1` | phase 1 epochs |
| `--p1_doclen` | `1` | sample-code blocks per doc in the phase 1 corpus |
| `--p2e` | `1` | phase 2 epochs |
| `--p2_doclen` | `1` | number of phase 2 training examples |

```bash
# untrained baseline, all use cases
python -m experiments.talkie.train_talkie --llm talkie-1930-13b-it --top_p 0.9 --num_rounds 1

# phase 1 only
python -m experiments.talkie.train_talkie --llm talkie-1930-13b-it --phase1 --p1e 5 --p1_doclen 5 \
    --top_p 0.9 --num_rounds 1

# phase 2 only
python -m experiments.talkie.train_talkie --llm talkie-1930-13b-it --phase2 --p2e 5 --p2_doclen 100 \
    --top_p 0.9 --num_rounds 1

# all flags: both phases, chosen use cases and versions
python -m experiments.talkie.train_talkie --llm talkie-1930-13b-it --phase1 --phase2 \
    --use_cases nosql reinforce counterfactual instruct --versions 1 11 24 35 37 \
    --p1e 5 --p1_doclen 5 --p2e 10 --p2_doclen 100 --top_p 0.9 --num_rounds 1
```

### `analyse_results_talkie.py`

Scores `results/talkie/raw_talkie` and writes to `results/talkie/analysis_talkie/`.
Versions and checkpoints are read from the folders (`archive/` is ignored); the
flags narrow them down. A flag never drops a checkpoint that does not have that
setting (e.g. `--p1e 5` still keeps `notrain` and the `phase2` runs).

- **Phase 1** (`phase1/`): training effectiveness, from the `nosql` runs only.
  `phase1_scores_<model>.csv` plus plots of accuracy against epochs and doclen.
- **Phase 2** (`phase2/`): conflict, per checkpoint, laid out like the main
  analysis (`accuracy/`, `knowledge_shift/`, `knowledge_anchor/`,
  `knowledge_anchor_distribution/`, `token_distribution/`).

Shared code (scoring, averaging, plots) is imported from
`experiments/main/analyse_results.py`; only Talkie's answer parsing and its extra
columns are separate.

| Flag | Default | Description |
|---|---|---|
| `--phase` | `all` | `1`, `2` or `all` |
| `--model` | `talkie-1930-13b-it` | model name in the result file names |
| `--versions` | all found | swap versions to use |
| `--p1e` | all found | phase 1 epochs to use |
| `--p2e` | all found | phase 2 epochs to use |
| `--p1_doclen` | all found | phase 1 doclens to use |
| `--p2_doclen` | all found | phase 2 doclens to use |

```bash
# everything found
python -m experiments.talkie.analyse_results_talkie

# phase 1 only
python -m experiments.talkie.analyse_results_talkie --phase 1

# all flags: phase 2 for epochs 5 and 10, two versions
python -m experiments.talkie.analyse_results_talkie --phase 2 --model talkie-1930-13b-it \
    --versions 1 11 --p1e 5 10 --p2e 5 10 --p1_doclen 1 5 10 --p2_doclen 50 100 200
```
