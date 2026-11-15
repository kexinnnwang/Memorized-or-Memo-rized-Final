# SQLShift_PandasShift

Data for the knowledge-conflict experiments in `experiments/`:
operator **documentation** (actual and counterfactual "swapped" versions, for SQL and
Pandas) and the **question–answer** sets the models are trained and tested on.

Experiment code finds everything here through `src/paths.py`
(`DATA_DIR`, `SQL_DIR`, `SQL_ORIGINAL_DIR`, `sql_swap_dir(v)`, `PANDAS_DIR`, `QA_DIR`,
`CUSTOM_TABLES_DIR`, `TALKIE_TRAIN_DIR`). The generator scripts in this directory build
their paths from their own location, so they can be run from any working directory
(except `WikiSQL/generate_data.py` — see below).

## Layout

```
SQLShift_PandasShift/
├── Documentation/
│   ├── SQL/
│   │   ├── SQL_original/          actual docs: SELECT, MIN(), MAX(), SUM(), AVG()
│   │   └── SQL_swap_{1..44,50}/   counterfactual docs + matches.json
│   └── Pandas/
│       ├── Dataframe/             actual docs: [], .min(), .max(), .sum(), .mean()
│       └── Dataframe_swap_{1,11,24,35,37}/   counterfactual docs + matches.json
├── QuestionAnswer/
│   ├── simple_tables_sql.jsonl     test set (SQL answers)
│   ├── simple_tables_pandas.jsonl  test set (Pandas answers)
│   ├── generate_sql.py
│   ├── Custom_Tables/              the hand-made tables + table_template.txt
│   ├── Talkie_Train/               Talkie QA-LoRA training set (+ swapped versions)
│   ├── WikiSQL_Tables/             Wikipedia tables used by the training set
│   │   └── WikiSQL/                raw WikiSQL dataset + generate_data.py
│   └── archive/                    unused files kept for reference
├── scraper.py, beautiful_soup.py   scrape w3schools SQL docs
├── swap.py                         build SQL_swap_N
├── swap_dataframe.py               build Dataframe_swap_N
├── generate_pandas_data.py         build simple_tables_pandas.jsonl
└── generate_wiki_swap_data.py      build Talkie_Train/*_version_N.jsonl
```

## Documentation

One `.txt` file per operator: `OPERATOR`, `DESCRIPTION`, `SYNTAX`, then ten
`--- Sample Code N ---` blocks (each a sample table, code and result). The runners
truncate at `--- Sample Code {k+1} ---` to control how much documentation a model sees.

**`SQL/SQL_original/`** — the real documentation for the five SQL operators.

**`SQL/SQL_swap_N/`** — counterfactual documentation. The five operators' docs are
permuted (a derangement, so no operator keeps its own doc) and every operator name in the
text is renamed to match; files are named `<operator>_v3.txt`. `matches.json` records the
mapping as a list of single-key dicts, e.g.
`[{"MIN()": "AVG()"}, {"SUM()": "MAX()"}, ...]`.
`N` is the *version* passed to the runners as `--counterfactual_version` / `--version`.

**`Pandas/Dataframe/`** and **`Pandas/Dataframe_swap_N/`** — the same idea for Pandas
(`[]`↔`SELECT`, `.min()`, `.max()`, `.sum()`, `.mean()`↔`AVG()`). Each Pandas swap
version reuses the mapping of the SQL swap version with the same number.
Note that the file names start with a dot (`.sum().txt`), so a plain `ls` hides all but
`[].txt` — use `ls -a`.

| Used by | Reads |
|---|---|
| `conflict_simple.py`, `_larger`, `_thinking`, `_unlearn`, `_talkie`, `_talkie_updated` | `SQL_original`, `SQL_swap_N` |
| `conflict_simple_pandas.py` | `Dataframe`, `Dataframe_swap_N` |
| `analyse_results_simple.py`, `_talkie`, `_unlearn` | `SQL_swap_N/matches.json` |
| `analyse_results_pandas.py` | `Dataframe_swap_N/matches.json` |

## QuestionAnswer

All `.jsonl` files have one example per line with `question`, `table`
(`{table_name: [header, row, ...]}`), `gt` (ground-truth answer) and either `sql` or
`pandas`. Every question is a single aggregate (`SUM`/`AVG`/`MIN`/`MAX`) over one numeric
column, e.g. *"What is the total number of pages of all books?"* →
`SELECT SUM(pages) FROM books`.

| File | Rows | What it is | Used by |
|---|---|---|---|
| `simple_tables_sql.jsonl` | 104 | Test set over the 8 custom tables (employees, products, orders, athletes, movies, books, restaurants, students) | all SQL runners; input to `generate_pandas_data.py` |
| `simple_tables_pandas.jsonl` | 104 | Same questions, Pandas answer instead of SQL | `conflict_simple_pandas.py` |
| `Custom_Tables/*.txt` | — | Each custom table rendered as prompt text (column descriptions + rows) | all runners (inserted as `{database}` in the prompt) |
| `Custom_Tables/table_template.txt` | — | Template for a table file: `{table_name}`, `{description}`, `{table}` | `WikiSQL/generate_data.py` |
| `Talkie_Train/simple_tables_sql_wiki.jsonl` | 688 | Talkie Phase-2 (QA-LoRA) training set: 86 Wikipedia tables × 2 numeric columns × 4 aggregates. Kept on different tables from the test set | `experiments/talkie/train_talkie.py` |
| `Talkie_Train/simple_tables_sql_wiki_version_{15..19}.jsonl` | 688 each | The training set with SQL answers rewritten using `SQL_swap_N/matches.json` | not used (was used by the removed `conflict_simple_talkie_ablation.py`) |
| `Talkie_Train/simple_tables_sql_wiki.jsonl.bak` | — | Backup of the training set | not used |
| `WikiSQL_Tables/*.txt` | 93 files | Wikipedia tables rendered as prompt text; 86 are used by the training set | `src/llm.py` (`talkie_train_phase2`) |
| `archive/` | — | `simple_tables_sql_train.jsonl` (36 rows over cars/hospitals/flights), its generator `generate_sql_train.py`, the matching table files `cars.txt`, `flights.txt`, `hospitals.txt`, and `train_wikisql.jsonl` | not used |

## Data Generation

| Script | Produces |
|---|---|
| `scraper.py` (+ `beautiful_soup.py`) | `SQL_original/` |
| `swap.py` | `SQL_swap_1` … `SQL_swap_44` |
| `swap_dataframe.py` | `Dataframe_swap_N` |
| `QuestionAnswer/generate_sql.py` | `simple_tables_sql.jsonl` |
| `generate_pandas_data.py` | `simple_tables_pandas.jsonl` |
| `WikiSQL/generate_data.py` (run from inside `WikiSQL_Tables/WikiSQL/`) | `WikiSQL_Tables/*.txt`, `Talkie_Train/simple_tables_sql_wiki.jsonl` |
| `generate_wiki_swap_data.py` | `Talkie_Train/*_version_N.jsonl` |

These scripts overwrite the existing files, and the swap scripts draw new random
mappings, so re-running them changes the data the existing results were produced with.
