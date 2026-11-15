# src

Shared code used by the experiments in `experiments/` and the generators in `SQLShift_PandasShift/`.

Everything here is imported as a package (`from src.<module> import ...`), so scripts are
run from the repository root, e.g. `python -m experiments.main.conflict ...`
(the slurm scripts set `PYTHONPATH` for this).

```
src/
├── paths.py                 locations of the project folders (data, prompts, results, config)
├── config.py, config.yaml   experiment settings: default model, generation and training settings
├── llm.py                   LLM class: loads a model, generates answers with token probabilities
│                            (one at a time or batched), unlearning, Talkie training
├── data_loading.py          Document dataclass for the operator docs
├── SQLite/
│   └── sqlite_database.py   builds a question's table in memory and runs SQL on it
├── .env                     HF_TOKEN (Hugging Face access token; keep it out of git)
└── archive/                 previous llm.py and llm_batched.py, kept for reference
```

| File | Used by |
|---|---|
| `paths.py` | all runners and analysis scripts |
| `config.py`, `config.yaml` | all runners (`--llm` overrides the default model) |
| `llm.py` | `experiments/main/conflict.py`, `experiments/unlearning/unlearn.py`, Talkie runners |
| `data_loading.py` | all runners |
| `SQLite/sqlite_database.py` | analysis scripts, `SQLShift_PandasShift/QuestionAnswer/generate_sql.py`, `WikiSQL/generate_data.py` |
