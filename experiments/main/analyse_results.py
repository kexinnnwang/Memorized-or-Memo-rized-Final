import argparse
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gzip
import os
from pathlib import Path
from matplotlib.patches import Rectangle
from src.paths import SQL_DIR, PANDAS_DIR, RESULTS_DIR


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
    
SQL_SLOT_NOISE = " :`{\n"
SQL_OPERATORS = {"SELECT", "MAX", "MIN", "AVG", "SUM"}
SQL_OPERATOR_SLOTS = re.compile(r"^\s*(\S+)\s+([A-Za-z_]+)\s*\(")


def sql_swap_tokens(query: str, mapping: dict) -> str:
    if not mapping:
        return query
    pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda mo: mapping[mo.group(0)], query)


def sql_swap_prob(step, swapped):
    if step["token"].strip().lower() == swapped.strip().lower():
        return step["prob"]
    for alt in step["alternatives"]:
        if alt["token"].strip().lower() == swapped.strip().lower():
            return alt["prob"]
    return 0.0


def sql_is_operator_slot(step) -> bool:
    return step["token"].strip().upper() in SQL_OPERATORS


def sql_same_operator(token: str, expected: str) -> bool:
    return token.strip().rstrip("()").upper() == expected.strip().rstrip("()").upper()


def sql_unswap_slots(sql, expect_head: str, expect_agg: str, target: str):
    if not sql:
        return None
    mo = SQL_OPERATOR_SLOTS.match(sql)
    if mo is None:
        return None
    if not sql_same_operator(mo.group(1), expect_head) or not sql_same_operator(mo.group(2), expect_agg):
        return None
    return sql[:mo.start(1)] + "SELECT" + sql[mo.end(1):mo.start(2)] + target + sql[mo.end(2):]


def sql_scalar_result(mydb, sql):
    out = mydb.execute_query(sql) if sql else False
    rows = [list(row) for row in out] if out else []
    return rows[0][0] if rows and rows[0] else None


SQL_INVENTED_NOISE = SQL_SLOT_NOISE + "\"'"


def sql_is_invented(operator: str) -> bool:    
    return operator.replace("()", "").strip().upper() not in SQL_OPERATORS


def sql_starts_operator(token: str, operator: str) -> bool:    
    text = token.strip(SQL_INVENTED_NOISE).upper()
    if not text:
        return False
    if operator.upper().startswith(text):
        return True
    return text.startswith(operator.upper()) and not text[len(operator)].isalnum()


def sql_spell_end(region, start: int, operator: str):    
    target = operator.upper()
    built = ""
    for i in range(start, len(region)):
        token = region[i]["token"]
        built += token.upper() if built else token.lstrip(SQL_INVENTED_NOISE).upper()
        if built.startswith(target):
            rest = built[len(target):]     # the last token may carry a bracket or spaces
            return i if not rest or not (rest[0].isalnum() or rest[0] == "_") else None
        if not target.startswith(built):
            return None
    return None


def sql_invented_slot(region, start: int, operator: str, original: str):
    end = sql_spell_end(region, start, operator)
    used = region[start:end + 1] if end is not None else [region[start]]
    first = sum(alt["prob"] for alt in region[start]["alternatives"]
                if sql_starts_operator(alt["token"], operator))
    cf = (first + sum(step["prob"] for step in used[1:])) / len(used)
    seen = [p for p in (sql_swap_prob(step, original) for step in used) if p > 0.0]
    orig = sum(seen) / len(seen) if seen else 0.0
    return {"orig": orig, "cf": cf}


def sql_next_real(region, start: int):
    return next((k for k in range(start, len(region))
                 if region[k]["token"].strip(SQL_INVENTED_NOISE)), None)


def sql_spelled_end(region, start: int, known):    
    return next((end for end in (sql_spell_end(region, start, op) for op in known)
                 if end is not None), None)


def sql_invented_probs(region, matches, head_op: str, target: str, agg_op: str) -> dict:
    known = [v.replace("()", "") for v in matches.values()]
    curr = {}
    head = next((k for k, s in enumerate(region) if s["token"].strip(SQL_SLOT_NOISE)), None)
    head_end = None
    if head is not None and sql_is_operator_slot(region[head]):
        head_end = head
    elif head is not None:
        head = sql_next_real(region, head)     # an invented operator may follow a quote
        head_end = None if head is None else sql_spelled_end(region, head, known)
    if head_end is not None:
        curr["SELECT"] = sql_invented_slot(region, head, head_op, "SELECT")

    paren = next((k for k, s in enumerate(region) if "(" in s["token"] and s["token"].strip() != "()"), None)
    if paren is not None and paren >= 1 and sql_is_operator_slot(region[paren - 1]):
        curr["AGG"] = sql_invented_slot(region, paren - 1, agg_op, target)
    elif head_end is not None:
        agg = sql_next_real(region, head_end + 1)
        if agg is not None and sql_spelled_end(region, agg, known) is not None:
            curr["AGG"] = sql_invented_slot(region, agg, agg_op, target)
    return curr


class SQLSpec:
    name = "SQL"
    head_key = "SELECT"      
    head_label = "SELECT"     
    raw_dir = RESULTS_DIR / "main" / "raw_sql"
    out_dir = RESULTS_DIR / "main" / "analysis_sql"

    def swap_dir(self, version):
        return SQL_DIR / f"SQL_swap_{version}"

    def parse_answer(self, text: str):
        text = text.replace("```sql", " ")
        text = text.replace("```", " ")
        text = text.split("</think>")[-1]
        ans_m = re.search(r"ANSWER:\s*([^,}\n]*)", text)
        sql_m = re.search(r"SQL:\s*(.+)", text, re.DOTALL)
        answer = ans_m.group(1).strip() if ans_m else None
        sql = None
        if sql_m:
            sql = sql_m.group(1).strip()
            sql = sql.strip("{}").strip()
            sql = sql.rstrip(" \n;}").strip()
        return answer, sql

    def score_question(self, res, code, matches, steps, path=""):        
        from src.SQLite.sqlite_database import Database

        target = next((t for t in ("MAX", "MIN", "AVG", "SUM") if t in res["gt_query"]), None)
        if target is None:
            raise ValueError(f"{path}: no aggregate found in gt_query {res['gt_query']!r}")

        mydb = Database(res["databases"])
        code_out = sql_scalar_result(mydb, code)

        select_match = matches["SELECT"]
        target_match = matches[f"{target}()"].replace("()", "")
        cf_queries = [sql_swap_tokens(res["gt_query"], {"SELECT": select_match, target: target_match}),
                      sql_swap_tokens(res["gt_query"], {"SELECT": select_match}),
                      sql_swap_tokens(res["gt_query"], {target: target_match})]

        cf_variants = [(select_match, target_match), (select_match, target), ("SELECT", target_match)]
        cf_hits = []
        for expect_head, expect_agg in cf_variants:
            unswapped = sql_unswap_slots(code, expect_head, expect_agg, target)
            cf_hits.append(unswapped is not None and sql_scalar_result(mydb, unswapped) == float(res["gt"]))

        mydb.con.close()

        # slot probabilities: the token after "sql" is SELECT, the token before the first "(" is the aggregate
        sql_idx = None
        for idx, step in enumerate(steps):
            if step["token"].strip().strip("`").strip().lower() == "sql":
                sql_idx = idx
                break

        curr = {}
        if sql_idx is not None:
            region = steps[sql_idx + 1:]

            if sql_is_invented(select_match):
                return code_out, cf_queries, cf_hits, sql_invented_probs(region, matches, select_match, target, target_match)

            select_pos = next((k for k, s in enumerate(region) if s["token"].strip(SQL_SLOT_NOISE)), None)
            paren_pos  = next((k for k, s in enumerate(region) if "(" in s["token"] and s["token"].strip() != "()"), None)

            if select_pos is not None and sql_is_operator_slot(region[select_pos]):
                step = region[select_pos]
                curr["SELECT"] = {"orig": sql_swap_prob(step, "SELECT"), "cf": sql_swap_prob(step, select_match.replace("()", ""))}

            if paren_pos is not None and paren_pos >= 1 and sql_is_operator_slot(region[paren_pos - 1]):
                step = region[paren_pos - 1]
                curr["AGG"] = {"orig": sql_swap_prob(step, target), "cf": sql_swap_prob(step, target_match)}

        return code_out, cf_queries, cf_hits, curr


PANDAS_AGG_FORMS = (".mean()", ".min()", ".max()", ".sum()")
PANDAS_INDEX_FORM = "[]"
PANDAS_SLOT_NOISE = " :`{}\n"
PANDAS_INDEX_TOKEN = re.compile(r"\[(['\"]?|\][.)>}:]?)$")
PANDAS_MARKER = re.compile(r'pandas"?\s*:', re.IGNORECASE)
PANDAS_GT_QUERY_RE = re.compile(r"^\s*(?P<table>\w+)\s*\[\s*'(?P<column>[^']*)'\s*\]\s*\.(?P<agg>mean|min|max|sum)\s*\(\s*\)\s*$")
PANDAS_QUERY_SLOTS = re.compile(
    r"^\s*(?P<table>\w+)\s*"
    r"(?:\[\s*['\"](?P<col1>[^'\"]*)['\"]\s*\]|\.\s*(?P<head>\w+)\s*\(\s*['\"](?P<col2>[^'\"]*)['\"]\s*\))\s*"
    r"(?:\.\s*(?P<agg>\w+)\s*\(\s*\)|(?P<idx>\[\s*\]))\s*$")


def pandas_parse_gt_query(query: str):
    m = PANDAS_GT_QUERY_RE.match(query)
    if m is None:
        return None
    return m.group("table"), m.group("column"), f".{m.group('agg')}()"


def pandas_render_query(table: str, column: str, first: str, second: str) -> str:
    head = f"['{column}']" if first == PANDAS_INDEX_FORM else f"{first[:-1]}'{column}')"
    return f"{table}{head}{second}"


def pandas_unswap_slots(code, expect_head: str, expect_agg: str, target: str):
    if not code:
        return None
    mo = PANDAS_QUERY_SLOTS.match(code)
    if mo is None:
        return None
    head = PANDAS_INDEX_FORM if mo.group("col1") is not None else f".{mo.group('head').lower()}()"
    agg = PANDAS_INDEX_FORM if mo.group("idx") is not None else f".{mo.group('agg').lower()}()"
    if head != expect_head or agg != expect_agg:
        return None
    column = mo.group("col1") if mo.group("col1") is not None else mo.group("col2")
    return pandas_render_query(mo.group("table"), column, PANDAS_INDEX_FORM, target)


def pandas_run(query: str, database: dict):
    df_list = next(iter(database.values()))
    name = next(iter(database.keys()))
    column_name = df_list[0]
    data = df_list[1:]
    query = re.sub(fr'^{name}', 'df', query)
    df = pd.DataFrame(data, columns=column_name)
    try:
        result = eval(query)
    except Exception:
        result = None
    return result


def pandas_operator_of(token: str):
    t = token.strip()
    if PANDAS_INDEX_TOKEN.match(t):
        return PANDAS_INDEX_FORM
    t = t.strip(PANDAS_SLOT_NOISE).lstrip(".")
    for op in PANDAS_AGG_FORMS:
        if t == op.strip(".()"):
            return op
    return None


def pandas_operator_prob(step, operator: str) -> float:
    return sum(alt["prob"] for alt in step["alternatives"] if pandas_operator_of(alt["token"]) == operator)


def pandas_find_slots(steps):
    seen = ""
    region = None
    for idx, step in enumerate(steps):
        seen += step["token"]
        if PANDAS_MARKER.search(seen):
            region = steps[idx + 1:]
            break
    if region is None:
        return {}

    found = [s for s in region if pandas_operator_of(s["token"]) is not None]
    return {key: step for key, step in zip(("INDEX", "AGG"), found)}


class PandasSpec:
    name = "Pandas"
    head_key = "INDEX"
    head_label = "[]"
    raw_dir = RESULTS_DIR / "main" / "raw_pandas"
    out_dir = RESULTS_DIR / "main" / "analysis_pandas"

    def swap_dir(self, version):
        return PANDAS_DIR / f"Dataframe_swap_{version}"

    def parse_answer(self, text: str):
        text = text.replace("```pandas", " ")
        text = text.replace("```", " ")
        text = text.split("</think>")[-1]
        ans_m = re.search(r'"?ANSWER"?\s*:\s*([^,}\n]*)', text, re.IGNORECASE)
        code_m = re.search(r'"?PANDAS"?\s*:\s*(.+)', text, re.DOTALL | re.IGNORECASE)
        answer = ans_m.group(1).strip().strip('"').strip("'").strip() if ans_m else None
        code = None
        if code_m:
            code = code_m.group(1).strip()
            code = code.strip("{}").strip()
            code = code.rstrip(" \n;}").strip()
            code = code.replace("`", "")
            code = code.split("=")[-1]
            code = code.replace("ANSWER:", "")
            code = code.replace("Answer:", "")
            code = code.replace("answer:", "")
            code = code.replace("\\n", "\n")
            code = code.strip()
            code = code.strip('"')
        return answer, code

    def score_question(self, res, code, matches, steps, path=""):        
        parsed = pandas_parse_gt_query(res["gt_query"])
        if parsed is None:
            raise ValueError(f"{path}: gt_query is not the expected template: {res['gt_query']!r}")
        table, column, target = parsed

        code_out = pandas_run(code, res["databases"])

        index_match = matches[PANDAS_INDEX_FORM]
        target_match = matches[target]
        cf_queries = [pandas_render_query(table, column, index_match, target_match),
                      pandas_render_query(table, column, index_match, target),
                      pandas_render_query(table, column, PANDAS_INDEX_FORM, target_match)]

        cf_variants = [(index_match, target_match), (index_match, target), (PANDAS_INDEX_FORM, target_match)]
        cf_hits = []
        for expect_head, expect_agg in cf_variants:
            unswapped = pandas_unswap_slots(code, expect_head, expect_agg, target)
            cf_hits.append(unswapped is not None
                           and pandas_run(unswapped, res["databases"]) == float(res["gt"]))

        # slot probabilities: the first two operator-like tokens after "pandas:", summed over equivalent spellings
        curr = {}
        slots = pandas_find_slots(steps)
        if "INDEX" in slots:
            curr["INDEX"] = {"orig": pandas_operator_prob(slots["INDEX"], PANDAS_INDEX_FORM),
                             "cf": pandas_operator_prob(slots["INDEX"], index_match)}
        if "AGG" in slots:
            curr["AGG"] = {"orig": pandas_operator_prob(slots["AGG"], target),
                           "cf": pandas_operator_prob(slots["AGG"], target_match)}

        return code_out, cf_queries, cf_hits, curr


SPECS = {"sql": SQLSpec(), "pandas": PandasSpec()}


VERSIONED_USE_CASES = ("instruct", "counterfactual")
USE_CASES = ("reinforce", "counterfactual", "instruct")


def numbered_dirs(parent, prefix):    
    if not parent.is_dir():
        return []
    nums = []
    for d in parent.iterdir():
        if d.is_dir() and d.name.startswith(prefix) and d.name[len(prefix):].isdigit():
            nums.append(int(d.name[len(prefix):]))
    return sorted(nums)


def discover_layout(raw_dir):
    doclens = set(numbered_dirs(raw_dir / "reinforce", "doclen_"))
    versions_by_doclen = {}
    for uc in VERSIONED_USE_CASES:
        for v in numbered_dirs(raw_dir / uc, "version_"):
            for d in numbered_dirs(raw_dir / uc / f"version_{v}", "doclen_"):
                doclens.add(d)
                versions_by_doclen.setdefault(d, set()).add(v)
    all_versions = sorted({v for vs in versions_by_doclen.values() for v in vs})
    versions_by_doclen = {d: sorted(vs) for d, vs in versions_by_doclen.items()}

    models = set()
    for path in raw_dir.rglob("*.json.gz"):
        if "archive" not in path.relative_to(raw_dir).parts:
            models.add(path.name[:-len(".json.gz")])

    return sorted(models), sorted(doclens), versions_by_doclen, all_versions


def condition_runs(raw_dir, condition: str, model: str, versions_by_doclen, all_versions):    
    if condition == "nosql":
        return [(v, raw_dir / "nosql" / f"{model}.json.gz") for v in all_versions]
    use_case, doclen = condition.rsplit("_doclen_", 1)
    versions = versions_by_doclen.get(int(doclen), [])
    if use_case in VERSIONED_USE_CASES:
        return [(v, raw_dir / use_case / f"version_{v}" / f"doclen_{doclen}" / f"{model}.json.gz")
                for v in versions]
    return [(v, raw_dir / use_case / f"doclen_{doclen}" / f"{model}.json.gz") for v in versions]


def load_matches(spec, version: int) -> dict:
    with open(spec.swap_dir(version) / "matches.json", 'r') as f:
        raw = json.load(f)
    return {k: v for entry in raw for k, v in entry.items()}


def channel(condition: str) -> str:    
    return "orig" if condition.startswith("reinforce") else "cf"


def load_results(path):
    if os.path.exists(path):
        with gzip.open(path, 'rt') as f:
            return json.load(f)["Round 1"]["questions"]
    return None


def analyse(spec, results, matches, path="", verbose: bool = False):
    probs_dict = []
    answer_acc = 0.0
    code_acc = 0.0
    code_output_acc = 0.0
    code_output_cf_acc = 0.0
    code_output_cf_head_acc = 0.0
    code_output_cf_agg_acc = 0.0
    counterfactual_code_acc = 0.0
    counterfactual_code_headonly_acc = 0.0
    counterfactual_code_aggonly_acc = 0.0
    abstain = 0.0

    for res in results:
        answer, code = spec.parse_answer(res["llm_answer"]["generated_text"])
        if to_float(answer) is None:
            abstain += 1.0

        steps = res["llm_answer"]["steps"]
        think_end = None
        for idx, step in enumerate(steps):
            if step["token"] == "</think>":
                think_end = idx
        if think_end is not None:
            steps = steps[think_end + 1:]

        code_out, cf_queries, cf_hits, curr = spec.score_question(res, code, matches, steps, path)

        if code == res["gt_query"]:
            code_acc += 1.0
        elif verbose:
            print(f"query: {code}")
            print(f"gt query: {res['gt_query']}")
            print(f"gt: {res['gt']}")
            print(f"code_out: {code_out}")

        if to_float(answer) == float(res["gt"]):
            answer_acc += 1.0

        if code_out == float(res["gt"]):
            code_output_acc += 1.0
        elif verbose:
            print(to_float(answer) == float(res["gt"]))
            print(f"query: {code}")
            print(f"gt query: {res['gt_query']}")
            print(f"gt: {res['gt']}")
            print(f"code_out: {code_out}")

        cf_full, cf_head_only, cf_agg_only = cf_queries
        if code == cf_full:
            counterfactual_code_acc += 1.0
            # print(f"original {res['gt_query']}")
            # print(f"Counterfactual {cf_full}")
        else:
            print(f"original {res['gt_query']}")
            print(f"Code {code}")
            print(f"Counterfactual {cf_full}")
        if code == cf_head_only:
            counterfactual_code_headonly_acc += 1.0
            # print(f"original {res['gt_query']}")
            # print(f"{spec.head_label} {cf_head_only}")
        if code == cf_agg_only:
            counterfactual_code_aggonly_acc += 1.0
            # print(f"original {res['gt_query']}")
            # print(f"agg {cf_agg_only}")

        code_output_cf_acc += float(cf_hits[0])
        code_output_cf_head_acc += float(cf_hits[1])
        code_output_cf_agg_acc += float(cf_hits[2])

        probs_dict.append(curr)

    n = float(len(results))
    metrics = {"answer": answer_acc / n,
               "code": code_acc / n,
               "codeout": code_output_acc / n,
               "codeout_cf": code_output_cf_acc / n,
               "codeout_cf_head": code_output_cf_head_acc / n,
               "codeout_cf_agg": code_output_cf_agg_acc / n,
               "abstain": abstain / n,
               "counterfactual_code_acc": counterfactual_code_acc / n,
               "counterfactual_code_headonly_acc": counterfactual_code_headonly_acc / n,
               "counterfactual_code_aggonly_acc": counterfactual_code_aggonly_acc / n}
    return probs_dict, metrics


def average_metrics(runs):
    return {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}


def average_probs(spec, runs):
    lengths = {len(r) for r in runs}
    if len(lengths) != 1:
        raise ValueError(f"versions disagree on question count: {sorted(lengths)}")

    averaged = []
    for i in range(len(runs[0])):
        merged = {}
        for key in (spec.head_key, "AGG"):
            slots = [r[i][key] for r in runs if key in r[i]]
            if slots:
                merged[key] = {w: sum(s[w] for s in slots) / len(slots) for w in ("orig", "cf")}
        averaged.append(merged)
    return averaged


def analyse_condition(spec, condition, model, layout, matches_by_version, verbose=False):
    _, _, versions_by_doclen, all_versions = layout
    cache = {}
    runs = []
    for version, path in condition_runs(spec.raw_dir, condition, model, versions_by_doclen, all_versions):
        if path not in cache:
            loaded_results = load_results(path)
            if loaded_results is None:
                continue
            cache[path] = loaded_results        
        print(f"Version {version}")
        runs.append(analyse(spec, cache[path], matches_by_version[version], path, verbose))
    if not runs:
        print(f"    no result files for {model} / {condition}; skipping")
        return [], None
    probs = [r[0] for r in runs]
    mets = [r[1] for r in runs]
    return average_probs(spec, probs), average_metrics(mets)


def compare(comp, base):
    diff = []
    mul = []

    for i in range(len(comp)):
        for key in comp[i]:
            if key not in base[i]:
                continue
            diff.append(comp[i][key] - base[i][key])
            mul.append((comp[i][key]+1.0)/(base[i][key]+1.0))

    return diff, mul


def flatten(probs_dict, which):
    return [{k: curr[k][which] for k in curr} for curr in probs_dict]


def collect(probs_dict, which, kinds):
    out = []
    for curr in probs_dict:
        for k in kinds:
            if k in curr:
                out.append(curr[k][which])
    return out


RATIO_LIM = 2.0


def collect_ratio(probs_dict, kinds):
    out = []
    for curr in probs_dict:
        for k in kinds:
            if k in curr:
                orig, cf = curr[k]["orig"], curr[k]["cf"]
                if orig + cf == 0.0:
                    continue   # neither operator was logged; ratio is undefined, not 0
                out.append((cf - orig) / (0.5*(orig + cf)))
    return out


def ratio_mean(probs_dicts, kinds):
    vals = [v for probs_dict in probs_dicts for v in collect_ratio(probs_dict, kinds)]
    return sum(vals) / len(vals) if vals else float("nan")


def condition_rows(doclens):
    comp_rows = [f"{uc}_doclen_{d}" for uc in USE_CASES for d in doclens]
    return comp_rows, ["nosql"] + comp_rows


def metric_cols(spec):
    n, h = spec.name, spec.head_label
    return [("Direct Answer Acc", "answer"),
        (f"Orig {n} Str Match Acc", "code"),
        (f"{n} Output Acc", "codeout"),
        (f"CF {n} Output Acc", "codeout_cf"),
        (f"CF {n} ({h}) Output Acc", "codeout_cf_head"),
        (f"CF {n} (AGG) Output Acc", "codeout_cf_agg"),
        ("Abstention Rate", "abstain"),
        (f"CF {n} Str Match Acc", "counterfactual_code_acc"),
        (f"CF {n} ({h}) Str Match Acc", "counterfactual_code_headonly_acc"),
        (f"CF {n} (AGG) Str Match Acc", "counterfactual_code_aggonly_acc")]


COMP_COLS = [("Diff", "diff"), ("Mul", "mul")]


def plot_distributions(models_probs, which, out_path, kinds, comp_rows):
    model_names = list(models_probs.keys())
    base_label = "no_docs_original" if which == "orig" else "no_docs_new"

    fig, axes = plt.subplots(len(model_names), len(comp_rows),
                             figsize=(3.8 * len(comp_rows), 3.2 * len(model_names)), squeeze=False)
    for i, model in enumerate(model_names):
        base = collect(models_probs[model]["nosql"], which, kinds)
        for j, key in enumerate(comp_rows):
            ax = axes[i][j]
            other = collect(models_probs[model][key], which, kinds)
            hi = max(max(base, default=0.0), max(other, default=0.0))
            bins = np.linspace(0, hi if hi > 0 else 1.0, 31)
            key = key.replace("counterfactual", "new")
            key = key.replace("instruct", "new_instruct")
            key = key.replace("reinforce", "repeat")
            if base:
                sns.histplot(base, stat='probability', bins=bins, color='tab:blue', alpha=0.4, label=base_label, ax=ax)
            if other:
                sns.histplot(other, stat='probability', bins=bins, color='tab:orange', alpha=0.4, label=key, ax=ax)
            else:
                ax.text(0.5, 0.5, 'no run', transform=ax.transAxes, ha='center', va='center', fontsize=10, color='0.5')
            ax.set_yscale('log')
            ax.set_xlim(0, 1.0)
            ax.set_ylim(1e-3, 1.0)
            ax.set_xlabel('probability')
            ax.set_ylabel('log(fraction of tokens)')
            if i == 0:
                ax.set_title(key, fontsize=11)
            if j == 0:
                ax.annotate(model, xy=(-0.35, 0.5), xycoords='axes fraction', fontsize=9, fontweight='bold', ha='right', va='center', rotation=90)
            if base or other:
                ax.legend(fontsize=8)

    fig.suptitle(f'{base_label} (blue) vs condition (orange)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ratios(models_probs, out_path, kinds, metric_rows):
    model_names = list(models_probs.keys())
    conditions = metric_rows
    titles = {}

    fig, axes = plt.subplots(len(model_names), len(conditions), figsize=(3.8 * len(conditions), 3.2 * len(model_names)), squeeze=False)
    bins = np.linspace(-RATIO_LIM, RATIO_LIM, 41)
    for i, model in enumerate(model_names):
        for j, cond in enumerate(conditions):
            ax = axes[i][j]
            vals = collect_ratio(models_probs[model][cond], kinds)
            if vals:
                sns.histplot(vals, stat='probability', bins=bins, color='tab:green', alpha=0.6, ax=ax)
            else:
                ax.text(0.5, 0.5, 'no run', transform=ax.transAxes, ha='center', va='center', fontsize=10, color='0.5')
            ax.set_xlim(-RATIO_LIM, RATIO_LIM)
            ax.set_xticks(np.arange(-RATIO_LIM, RATIO_LIM + 0.5, 1.0))
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)
            ax.set_xlabel('(cf - orig)/0.5(cf+orig)', fontsize=9)
            ax.set_ylabel('fraction of tokens', fontsize=9)
            if i == 0:
                ax.set_title(titles.get(cond, cond), fontsize=11)
            if j == 0:
                ax.annotate(model, xy=(-0.3, 0.5), xycoords='axes fraction', fontsize=9, fontweight='bold', ha='right', va='center', rotation=90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_ratio_tables(models_probs, kinds, doclens, metric_rows):
    models = list(models_probs.keys())

    pooled = pd.DataFrame([{"nosql": ratio_mean([models_probs[m]["nosql"]], kinds),
                            **{uc: ratio_mean([models_probs[m][f"{uc}_doclen_{d}"] for d in doclens], kinds)
                               for uc in USE_CASES}}
                           for m in models], index=models)[["nosql", *USE_CASES]]

    by_doclen = pd.DataFrame([{cond: ratio_mean([models_probs[m][cond]], kinds) for cond in metric_rows}
                              for m in models], index=models)[metric_rows]

    for df in (pooled, by_doclen):
        df.index.name = "Model"
    return pooled.round(4), by_doclen.round(4)


def row_values(scored, cols):
    return {t: (float("nan") if scored is None else scored[k]) for t, k in cols}


def build_comps_table(model, comps, comp_rows):
    titles = [t for t, _ in COMP_COLS]
    data = [row_values(comps[model][cond], COMP_COLS) for cond in comp_rows]
    df = pd.DataFrame(data, index=comp_rows)[titles]
    df.index.name = "Use Case"
    return df.round(4)


def build_metrics_table(spec, model, metrics, metric_rows):
    cols = metric_cols(spec)
    titles = [t for t, _ in cols]
    data = [row_values(metrics[model][uc], cols) for uc in metric_rows]
    df = pd.DataFrame(data, index=metric_rows)[titles]
    df.index.name = "Use Case"
    return df.round(4)


def save_flat_table_png(df, path, title=None, idx_w=1.8):
    cols = list(df.columns)
    idx = list(df.index)
    idx_name = df.index.name or ""
    cell_w, cell_h = 2.1, 0.5
    title_h = 0.5 if title else 0.0

    xs = [0.0, idx_w] + [idx_w + cell_w * (i + 1) for i in range(len(cols))]
    total_w = xs[-1]
    total_h = title_h + (1 + len(idx)) * cell_h
    fig, ax = plt.subplots(figsize=(total_w, total_h))
    ax.set_xlim(0, total_w); ax.set_ylim(0, total_h); ax.axis("off")
    top = total_h

    def cell(x0, x1, y0, y1, text, fc="white", weight="normal", size=8):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, facecolor=fc, edgecolor="0.6", linewidth=0.6))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size, fontweight=weight)

    if title:
        ax.text(total_w / 2, top - title_h / 2, title, ha="center", va="center", fontsize=11, fontweight="bold")

    hy1 = top - title_h; hy0 = hy1 - cell_h
    cell(0, idx_w, hy0, hy1, idx_name, fc="#d9e1f2", weight="bold", size=9)
    for i, c in enumerate(cols):
        cell(xs[1 + i], xs[2 + i], hy0, hy1, str(c), fc="#d9e1f2", weight="bold", size=8)
    for r, ridx in enumerate(idx):
        y1 = hy0 - r * cell_h; y0 = y1 - cell_h
        shade = "#f7f7f7" if r % 2 else "white"
        cell(0, idx_w, y0, y1, str(ridx), fc=shade, weight="bold", size=8)
        for i, c in enumerate(cols):
            val = df.iloc[r][c]
            txt = "" if val is None else (f"{val:g}" if isinstance(val, float) else str(val))
            cell(xs[1 + i], xs[2 + i], y0, y1, txt, fc=shade, size=8)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def filter_layout(spec, layout, only_versions, only_doclens):    
    models, doclens, versions_by_doclen, all_versions = layout
    if not only_versions and not only_doclens:
        return layout

    missing_versions = sorted(set(only_versions or []) - set(all_versions))
    missing_doclens = sorted(set(only_doclens or []) - set(doclens))
    if missing_versions:
        print(f"[{spec.name}] versions not found: {missing_versions}")
    if missing_doclens:
        print(f"[{spec.name}] doclens not found: {missing_doclens}")

    kept = {}
    for doclen, versions in versions_by_doclen.items():
        if only_doclens and doclen not in only_doclens:
            continue
        versions = [v for v in versions if not only_versions or v in only_versions]
        if versions:
            kept[doclen] = versions
    return models, sorted(kept), kept, sorted({v for vs in kept.values() for v in vs})


def subset_suffix(only_versions, only_doclens):
    """Output folder suffix for a subset run, so it can't replace the full analysis."""
    suffix = ""
    if only_versions:
        suffix += "_v" + "_".join(str(v) for v in sorted(set(only_versions)))
    if only_doclens:
        suffix += "_d" + "_".join(str(d) for d in sorted(set(only_doclens)))
    return suffix


def run_language(spec, only_versions=None, only_doclens=None, out_dir=None):
    layout = filter_layout(spec, discover_layout(spec.raw_dir), only_versions, only_doclens)
    models, doclens, versions_by_doclen, all_versions = layout
    if not models or not all_versions:
        print(f"[{spec.name}] no results found in {spec.raw_dir}; skipping")
        return

    comp_rows, metric_rows = condition_rows(doclens)
    print(f"[{spec.name}] models: {models}")
    print(f"[{spec.name}] doclens: {doclens}")
    print(f"[{spec.name}] versions by doclen: {versions_by_doclen}")
    print(f"[{spec.name}] nosql/reinforce scored against versions: {all_versions} (nosql), per doclen (reinforce)")

    matches_by_version = {v: load_matches(spec, v) for v in all_versions}
    for v in all_versions:
        print(f"matches v{v}: {matches_by_version[v]}")

    models_probs = {}
    metrics = {}
    comps = {}
    for model in models:
        print(f"Model: {model}")
        all_probs = {}
        metrics[model] = {}
        for condition in metric_rows:
            print(f"  {condition}")
            probs_dict, condition_metrics = analyse_condition(spec, condition, model, layout, matches_by_version)
            all_probs[condition] = probs_dict
            metrics[model][condition] = condition_metrics
        models_probs[model] = all_probs

        comps[model] = {}
        for condition in comp_rows:
            which = channel(condition)
            if not all_probs[condition] or not all_probs["nosql"]:
                comps[model][condition] = None
                continue
            diff, mul = compare(flatten(all_probs[condition], which), flatten(all_probs["nosql"], which))
            if not diff:   # no operator slot was detected in this condition
                comps[model][condition] = None
                continue
            comps[model][condition] = {"diff": sum(diff)/len(diff), "mul": sum(mul)/len(mul)}

    if out_dir is None:
        out_dir = spec.out_dir.with_name(spec.out_dir.name + subset_suffix(only_versions and all_versions, only_doclens and doclens))
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("accuracy", "knowledge_shift", "knowledge_anchor", "knowledge_anchor_distribution", "token_distribution"):
        (out_dir / sub).mkdir(exist_ok=True)

    for model in models:
        safe = model.replace("/", "_")
        mdf = build_metrics_table(spec, model, metrics, metric_rows)
        mdf.to_csv(out_dir / "accuracy" / f"acc_{safe}.csv")
        save_flat_table_png(mdf, out_dir / "accuracy" / f"acc_{safe}.png", title=model)
        print(f"\nMetrics for {model}:")
        print(mdf)

        cdf = build_comps_table(model, comps, comp_rows)
        cdf.to_csv(out_dir / "knowledge_shift" / f"ks_{safe}.csv")
        save_flat_table_png(cdf, out_dir / "knowledge_shift" / f"ks_{safe}.png", title=model)
        print(f"\nComps for {model}:")
        print(cdf)
    print(f"Saved per-model metrics tables to {out_dir}/accuracy/acc_<model>.{{csv,png}}")
    print(f"Saved per-model comps tables to {out_dir}/knowledge_shift/ks_<model>.{{csv,png}}")

    head = spec.head_key
    for name, kinds in [("pooled", [head, "AGG"]), (head.lower(), [head]), ("agg", ["AGG"])]:
        plot_distributions(models_probs, "orig", out_dir / "token_distribution" / f"model_distributions_original_{name}.png", kinds, comp_rows)
        plot_distributions(models_probs, "cf", out_dir / "token_distribution" / f"model_distributions_counterfactual_{name}.png", kinds, comp_rows)
        plot_ratios(models_probs, out_dir / "knowledge_anchor_distribution" / f"ka_distribution_{name}.png", kinds, metric_rows)

        pooled_df, doclen_df = build_ratio_tables(models_probs, kinds, doclens, metric_rows)
        pooled_df.to_csv(out_dir / "knowledge_anchor" / f"ka_{name}.csv")
        save_flat_table_png(pooled_df, out_dir / "knowledge_anchor" / f"ka_{name}.png", idx_w=3.0,
                            title=f"Mean (cf-orig)/0.5(cf+orig) - {name}, averaged over versions and doclens")
        doclen_df.to_csv(out_dir / "knowledge_anchor" / f"ka_by_doclen_{name}.csv")
        save_flat_table_png(doclen_df, out_dir / "knowledge_anchor" / f"ka_by_doclen_{name}.png", idx_w=3.0,
                            title=f"Mean (cf-orig)/0.5(cf+orig) - {name}, averaged over versions")
        print(f"\nMean ratio ({name}), pooled over doclens:")
        print(pooled_df)
    print(f"Saved distribution and ratio figures to {out_dir}/")
    print(f"Saved mean-ratio tables to {out_dir}/knowledge_anchor/ka{{,_by_doclen}}_<kind>.{{csv,png}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["sql", "pandas"], default=None,
                        help="analyse one language; without it both are analysed")
    parser.add_argument("--versions", type=int, nargs="+", default=None,
                        help="only use these counterfactual versions; default: all found")
    parser.add_argument("--doclens", type=int, nargs="+", default=None,
                        help="only use these doc lengths; default: all found")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="save results here instead of the default folder; "
                             "when analysing both languages, each gets a subfolder")
    args = parser.parse_args()

    languages = [args.language] if args.language else ["sql", "pandas"]
    for lang in languages:
        out_dir = args.output_dir
        if out_dir is not None and len(languages) > 1:
            out_dir = out_dir / lang
        run_language(SPECS[lang], args.versions, args.doclens, out_dir)

if __name__ == "__main__":
    main()
