import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from src.SQLite.sqlite_database import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import seaborn as sns
import gzip
import os
from src.paths import SQL_DIR, RESULTS_DIR


def parse_answer(text: str) -> dict:
    #print(text)
    text = text.replace("```sql", " ")
    text = text.replace("```", " ")
    text = text.split("</think>")[-1]  
    ans_m = re.search(r"ANSWER:\s*([^,}\n]*)", text)
    sql_m = re.search(r"SQL:\s*(.+)", text, re.DOTALL)
    answer = ans_m.group(1).strip() if ans_m else None
    sql = None
    if sql_m:        
        sql = sql_m.group(1).strip()
        sql = sql.split("}")[0]        
        sql = sql.strip("{}").strip()    
        sql = sql.rstrip(" \n;}").strip()  
    print(sql)
    return {"ANSWER": answer, "SQL": sql}

def swap_tokens(query: str, mapping: dict) -> str:
    if not mapping:
        return query
    pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda mo: mapping[mo.group(0)], query)


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def swap_prob(step, swapped):           
    if step["token"].strip().lower() == swapped.strip().lower():
        return step["prob"]
    for alt in step["alternatives"]:
        if alt["token"].strip().lower() == swapped.strip().lower():
            return alt["prob"]
    return 0.0

SLOT_NOISE = " :`{\n"
OPERATORS = {"SELECT", "MAX", "MIN", "AVG", "SUM"}


def is_operator_slot(step) -> bool:
    return step["token"].strip().upper() in OPERATORS

OPERATOR_SLOTS = re.compile(r"^\s*(\S+)\s+([A-Za-z_]+)\s*\(")


def same_operator(token: str, expected: str) -> bool:    
    return token.strip().rstrip("()").upper() == expected.strip().rstrip("()").upper()


def unswap_slots(sql, expect_head: str, expect_agg: str, target: str):
    if not sql:
        return None
    mo = OPERATOR_SLOTS.match(sql)
    if mo is None:
        return None
    if not same_operator(mo.group(1), expect_head) or not same_operator(mo.group(2), expect_agg):
        return None
    return sql[:mo.start(1)] + "SELECT" + sql[mo.end(1):mo.start(2)] + target + sql[mo.end(2):]


def scalar_result(mydb, sql):    
    print(sql)
    out = mydb.execute_query(sql) if sql else False
    rows = [list(row) for row in out] if out else []
    return rows[0][0] if rows and rows[0] else None


RESULTS_ROOT = RESULTS_DIR / "unlearning" / "raw_unlearn"
VERSIONS = [1, 11, 24, 35, 37]  
DOCLENS = [5]
VERSIONED_USE_CASES = ("instruct", "counterfactual")


def load_matches(version: int) -> dict:    
    with open(SQL_DIR / f"SQL_swap_{version}" / "matches.json", 'r') as f:
        raw = json.load(f)
    return {k: v for entry in raw for k, v in entry.items()}


def condition_runs(condition: str, model: str):
    if condition == "nosql":
        return [(v, RESULTS_ROOT / "nosql" / f"{model}.json.gz") for v in VERSIONS]
    use_case, doclen = condition.rsplit("_doclen_", 1)
    if use_case in VERSIONED_USE_CASES:
        return [(v, RESULTS_ROOT / use_case / f"version_{v}" / f"doclen_{doclen}" / f"{model}.json.gz")
                for v in VERSIONS]
    return [(v, RESULTS_ROOT / use_case / f"doclen_{doclen}" / f"{model}.json.gz") for v in VERSIONS]


def channel(condition: str) -> str:    
    return "orig" if condition.startswith("reinforce") else "cf"


def load_results(path):
    if os.path.exists(path):
        with gzip.open(path, 'rt') as f:
            return json.load(f)["Round 1"]["questions"]
    return None


def analyse(results, matches, path="", verbose: bool = False):
    probs_dict = []
    answer_acc = 0.0
    sql_acc = 0.0
    sql_output_acc = 0.0
    sql_output_cf_acc = 0.0
    sql_output_cf_select_acc = 0.0
    sql_output_cf_agg_acc = 0.0
    counterfactual_sql_acc = 0.0
    counterfactual_sql_selectonly_acc = 0.0
    counterfactual_sql_aggonly_acc = 0.0
    abstain = 0.0    
    
    for res in results:
        target = next((t for t in ("MAX", "MIN", "AVG", "SUM") if t in res["gt_query"]), None)
        if target is None:
            raise ValueError(f"{path}: no aggregate found in gt_query {res['gt_query']!r}")

        ans_dict = parse_answer(res["llm_answer"]["generated_text"])        
        if to_float(ans_dict["ANSWER"]) is None:
            abstain += 1.0
        databases = res["databases"]
        mydb = Database(databases)
        
        sql_out = scalar_result(mydb, ans_dict["SQL"])

        if ans_dict["SQL"]==res["gt_query"]:
            sql_acc += 1.0
        elif verbose:
            print(f"query: {ans_dict['SQL']}")
            print(f"gt query: {res['gt_query']}")
            print(f"gt: {res['gt']}")
            print(f"sql_out: {sql_out}")

        if to_float(ans_dict["ANSWER"])==float(res["gt"]):
            answer_acc += 1.0

        if sql_out==float(res["gt"]):
            sql_output_acc += 1.0
        elif verbose:
            print(to_float(ans_dict["ANSWER"])==float(res["gt"]))
            print(f"query: {ans_dict['SQL']}")
            print(f"gt query: {res['gt_query']}")
            print(f"gt: {res['gt']}")
            print(f"sql_out: {sql_out}")

        select_match = matches["SELECT"]
        #.replace("()", "")
        target_match = matches[f"{target}()"].replace("()", "")
        counterfactual_sql = swap_tokens(res["gt_query"], {"SELECT": select_match, target: target_match})
        counterfactual_selectonly_sql = swap_tokens(res["gt_query"], {"SELECT": select_match})
        counterfactual_aggonly_sql = swap_tokens(res["gt_query"], {target: target_match})
        if ans_dict["SQL"]== counterfactual_sql:
            counterfactual_sql_acc += 1.0   
            # print(f"True Counterfactual {counterfactual_sql}")
            # print(f"Original {res["gt_query"]}")
            # print(f"Generated Counterfactual {ans_dict["SQL"]}")
        # else:
        #     print(f"True Counterfactual {counterfactual_sql}")
        #     print(f"Original {res["gt_query"]}")
        #     print(f"Generated Counterfactual {ans_dict["SQL"]}")
        if ans_dict["SQL"]== counterfactual_selectonly_sql:
            counterfactual_sql_selectonly_acc += 1.0 
            # print(f"True Counterfactual {counterfactual_sql}")
            # print(f"Original {res["gt_query"]}")
            # print(f"Generated Counterfactual {ans_dict["SQL"]}")
        if ans_dict["SQL"]== counterfactual_aggonly_sql:
            counterfactual_sql_aggonly_acc += 1.0
            if "reinforce" in str(path):
                print(f"True Counterfactual {counterfactual_sql}")
                print(f"Original {res["gt_query"]}")
                print(f"Generated Counterfactual {ans_dict["SQL"]}")
   
        cf_variants = [(select_match, target_match), (select_match, target), ("SELECT", target_match)]
        cf_hits = []
        for expect_head, expect_agg in cf_variants:
            unswapped = unswap_slots(ans_dict["SQL"], expect_head, expect_agg, target)
            cf_hits.append(unswapped is not None and scalar_result(mydb, unswapped) == float(res["gt"]))
        sql_output_cf_acc += float(cf_hits[0])
        sql_output_cf_select_acc += float(cf_hits[1])
        sql_output_cf_agg_acc += float(cf_hits[2])

        mydb.con.close()

        steps = res["llm_answer"]["steps"]
       
        think_end = None
        for idx, step in enumerate(steps):
            if step["token"] == "</think>":
                think_end = idx
        if think_end is not None:
            steps = steps[think_end + 1:]
        
        sql_idx = None
        for idx, step in enumerate(steps):
            if step["token"].strip().strip("`").strip().lower() == "sql":
                sql_idx = idx
                break        

        curr = {}  
        if sql_idx is not None:
            region = steps[sql_idx + 1:]
            # select_pos = next((k for k, s in enumerate(region) if s["token"].strip().lower() == "select"), None)
            # paren_pos = next((k for k, s in enumerate(region) if "(" in s["token"]), None)

            select_pos = next((k for k, s in enumerate(region) if s["token"].strip(SLOT_NOISE)), None)
            paren_pos  = next((k for k, s in enumerate(region) if "(" in s["token"] and s["token"].strip() != "()"), None)

            if select_pos is not None and is_operator_slot(region[select_pos]):
                step = region[select_pos]
                curr["SELECT"] = {"orig": swap_prob(step, "SELECT"), "cf": swap_prob(step, matches["SELECT"].replace("()", ""))}
                #.replace("()", "")

            if paren_pos is not None and paren_pos >= 1 and is_operator_slot(region[paren_pos - 1]):
                step = region[paren_pos - 1]
                curr["AGG"] = {"orig": swap_prob(step, target), "cf": swap_prob(step, matches[f"{target}()"].replace("()", ""))}

        probs_dict.append(curr)
        
    n = float(len(results))
    metrics = {"answer": answer_acc / n,
               "sql": sql_acc / n,
               "sqlout": sql_output_acc / n,
               "sqlout_cf": sql_output_cf_acc / n,
               "sqlout_cf_select": sql_output_cf_select_acc / n,
               "sqlout_cf_agg": sql_output_cf_agg_acc / n,
               "abstain": abstain / n,
               "counterfactual_sql_acc": counterfactual_sql_acc / n,
               "counterfactual_sql_selectonly_acc": counterfactual_sql_selectonly_acc / n,
               "counterfactual_sql_aggonly_acc": counterfactual_sql_aggonly_acc / n}
    return probs_dict, metrics


def average_metrics(runs):    
    return {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}


def average_probs(runs):
    lengths = {len(r) for r in runs}
    if len(lengths) != 1:
        raise ValueError(f"versions disagree on question count: {sorted(lengths)}")

    averaged = []
    for i in range(len(runs[0])):
        merged = {}
        for key in ("SELECT", "AGG"):
            slots = [r[i][key] for r in runs if key in r[i]]
            if slots:
                merged[key] = {w: sum(s[w] for s in slots) / len(slots) for w in ("orig", "cf")}
        averaged.append(merged)
    return averaged


def analyse_condition(condition, model, matches_by_version, verbose=False):
    cache = {}
    runs = []
    for version, path in condition_runs(condition, model):
        if path not in cache:
            loaded_resuls = load_results(path)
            if loaded_resuls is None:
                continue
            cache[path] = loaded_resuls
            print(f"Version {version}")
            runs.append(analyse(cache[path], matches_by_version[version], path, verbose))
    if not runs:
        print(f"    no result files for {model} / {condition}; skipping")
        return [], None
    probs = [r[0] for r in runs]
    mets = [r[1] for r in runs]
    return average_probs(probs), average_metrics(mets)

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
                    continue   
                out.append((cf - orig) / (0.5*(orig + cf)))
    return out

def plot_distributions(models_probs, which, out_path, kinds):   
    model_names = list(models_probs.keys())
    compare_lists = COMP_ROWS
    base_label = "nosql_real" if which == "orig" else "nosql_counterfactual"

    fig, axes = plt.subplots(len(model_names), len(compare_lists),
                             figsize=(3.8 * len(compare_lists), 3.2 * len(model_names)), squeeze=False)
    for i, model in enumerate(model_names):
        base = collect(models_probs[model]["nosql"], which, kinds)
        for j, key in enumerate(compare_lists):
            ax = axes[i][j]
            other = collect(models_probs[model][key], which, kinds)
            hi = max(max(base, default=0.0), max(other, default=0.0))
            bins = np.linspace(0, hi if hi > 0 else 1.0, 31)
            if base:
                sns.histplot(base, stat='probability', bins=bins, color='tab:blue', alpha=0.4, label=base_label, ax=ax)
            if other:
                sns.histplot(other, stat='probability', bins=bins, color='tab:orange', alpha=0.4, label=key, ax=ax)
            else:                
                ax.text(0.5, 0.5, 'no run', transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='0.5')
            ax.set_yscale('log')
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

def plot_ratios(models_probs, out_path, kinds):    
    model_names = list(models_probs.keys())
    conditions = METRIC_ROWS
    titles = {}

    fig, axes = plt.subplots(len(model_names), len(conditions),
                             figsize=(3.8 * len(conditions), 3.2 * len(model_names)), squeeze=False)
    bins = np.linspace(-RATIO_LIM, RATIO_LIM, 41)
    for i, model in enumerate(model_names):
        for j, cond in enumerate(conditions):
            ax = axes[i][j]
            vals = collect_ratio(models_probs[model][cond], kinds)
            if vals:
                sns.histplot(vals, stat='probability', bins=bins, color='tab:green', alpha=0.6, ax=ax)
            else:
                ax.text(0.5, 0.5, 'no run', transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='0.5')
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

COMP_ROWS = [f"{uc}_doclen_{d}" for uc in ("reinforce", "counterfactual", "instruct") for d in DOCLENS]
METRIC_ROWS = ["nosql"] + COMP_ROWS
METRIC_COLS = [("Direct Answer Acc", "answer"),
    ("Orig SQL Str Match Acc", "sql"),
    ("SQL Output Acc", "sqlout"),
    ("CF SQL Output Acc", "sqlout_cf"),
    ("CF SQL (SELECT) Output Acc", "sqlout_cf_select"),
    ("CF SQL (AGG) Output Acc", "sqlout_cf_agg"),
    ("Abstention Rate", "abstain"),
    ("CF SQL Str Match Acc", "counterfactual_sql_acc"),
    ("CF SQL (SELECT) Str Match Acc", "counterfactual_sql_selectonly_acc"),
    ("CF SQL (AGG) Str Match Acc", "counterfactual_sql_aggonly_acc")]
COMP_COLS = [("Diff", "diff"), ("Mul", "mul")]
USE_CASES = ("reinforce", "counterfactual", "instruct")


def ratio_mean(probs_dicts, kinds):
    vals = [v for probs_dict in probs_dicts for v in collect_ratio(probs_dict, kinds)]
    return sum(vals) / len(vals) if vals else float("nan")


def build_ratio_tables(models_probs, kinds):
    import pandas as pd
    models = list(models_probs.keys())

    pooled = pd.DataFrame([{"nosql": ratio_mean([models_probs[m]["nosql"]], kinds),
                            **{uc: ratio_mean([models_probs[m][f"{uc}_doclen_{d}"] for d in DOCLENS], kinds)
                               for uc in USE_CASES}}
                           for m in models], index=models)[["nosql", *USE_CASES]]

    by_doclen = pd.DataFrame([{cond: ratio_mean([models_probs[m][cond]], kinds) for cond in METRIC_ROWS}
                              for m in models], index=models)[METRIC_ROWS]

    for df in (pooled, by_doclen):
        df.index.name = "Model"
    return pooled.round(4), by_doclen.round(4)

def row_values(scored, cols):    
    return {t: (float("nan") if scored is None else scored[k]) for t, k in cols}


def build_comps_table(model, comps):    
    import pandas as pd
    titles = [t for t, _ in COMP_COLS]
    data = [row_values(comps[model][cond], COMP_COLS) for cond in COMP_ROWS]
    df = pd.DataFrame(data, index=COMP_ROWS)[titles]
    df.index.name = "Use Case"
    return df.round(4)

def build_metrics_table(model, metrics):
    import pandas as pd
    titles = [t for t, _ in METRIC_COLS]
    data = [row_values(metrics[model][uc], METRIC_COLS) for uc in METRIC_ROWS]
    df = pd.DataFrame(data, index=METRIC_ROWS)[titles]
    df.index.name = "Use Case"
    return df.round(4)

def save_flat_table_png(df, path, title=None, idx_w=1.8):    
    from matplotlib.patches import Rectangle
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
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True,
                               facecolor=fc, edgecolor="0.6", linewidth=0.6))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
                fontsize=size, fontweight=weight)

    if title:
        ax.text(total_w / 2, top - title_h / 2, title, ha="center", va="center",
                fontsize=11, fontweight="bold")

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

def main():
    # Qwen3-30B-A3B-Thinking-2507 is excluded: no instruct/version_35 or version_37 run.
    # models = ['Qwen2.5-72B-Instruct']
    models = [
    'Qwen2.5-32B-Instruct',     
    'Qwen2.5-Coder-32B-Instruct',  
    'Qwen3-30B-A3B-Instruct-2507',         
    'Qwen3-30B-A3B-Thinking-2507',     
    'Qwen3-Coder-30B-A3B-Instruct']

    #Llama-3.1-8B-Instruct: everything is good
    #Qwen2.5-7B-Instruct: reinforce and instruct as absentation but coutnerfactual does not?
    #Qwen2.5-7B-Coder-Instruct: no abstentiation
    #Qwen3-4B-Instruct-2507: only instruct has abstention
    #Qwen3-4B-Thinking-2507: the only one that gives the swapped operators    

    matches_by_version = {v: load_matches(v) for v in VERSIONS}
    for v in VERSIONS:
        print(f"matches v{v}: {matches_by_version[v]}")

    models_probs = {}
    metrics = {}   # metrics[model][condition] = {answer, sql, sqlout, abstain, ...}
    comps = {}     # comps[model][condition]   = {diff, mul}
    for model in models:
        print(f"Model: {model}")
        all_probs = {}
        metrics[model] = {}
        for condition in METRIC_ROWS:
            print(f"  {condition}")
            probs_dict, condition_metrics = analyse_condition(condition, model, matches_by_version)
            all_probs[condition] = probs_dict
            metrics[model][condition] = condition_metrics
        models_probs[model] = all_probs

        comps[model] = {}
        for condition in COMP_ROWS:
            which = channel(condition)
            if not all_probs[condition] or not all_probs["nosql"]:
                comps[model][condition] = None   # condition (or its baseline) has no run
                continue
            diff, mul = compare(flatten(all_probs[condition], which), flatten(all_probs["nosql"], which))
            comps[model][condition] = {"diff": sum(diff)/len(diff), "mul": sum(mul)/len(mul)}

    out_dir = RESULTS_DIR / "unlearning" / "analysis_unlearn"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("accuracy", "knowledge_shift", "knowledge_anchor", "knowledge_anchor_distribution", "token_distribution"):
        (out_dir / sub).mkdir(exist_ok=True)

    for model in models:
        safe = model.replace("/", "_")
        mdf = build_metrics_table(model, metrics)
        mdf.to_csv(out_dir / "accuracy" / f"acc_{safe}.csv")
        save_flat_table_png(mdf, out_dir / "accuracy" / f"acc_{safe}.png", title=model)
        print(f"\nMetrics for {model}:")
        print(mdf)

        cdf = build_comps_table(model, comps)
        cdf.to_csv(out_dir / "knowledge_shift" / f"ks_{safe}.csv")
        save_flat_table_png(cdf, out_dir / "knowledge_shift" / f"ks_{safe}.png", title=model)
        print(f"\nComps for {model}:")
        print(cdf)
    print(f"Saved per-model metrics tables to {out_dir}/accuracy/acc_<model>.{{csv,png}}")
    print(f"Saved per-model comps tables to {out_dir}/knowledge_shift/ks_<model>.{{csv,png}}")

    for name, kinds in [("pooled", ["SELECT", "AGG"]), ("select", ["SELECT"]), ("agg", ["AGG"])]:
        plot_distributions(models_probs, "orig", out_dir / "token_distribution" / f"model_distributions_original_{name}.png", kinds)
        plot_distributions(models_probs, "cf", out_dir / "token_distribution" / f"model_distributions_counterfactual_{name}.png", kinds)
        plot_ratios(models_probs, out_dir / "knowledge_anchor_distribution" / f"ka_distribution_{name}.png", kinds)

        pooled_df, doclen_df = build_ratio_tables(models_probs, kinds)
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

if __name__ == "__main__":
    main()