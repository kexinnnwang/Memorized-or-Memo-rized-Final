import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SQL_TO_PANDAS = {
    "SELECT": "[]",
    "MAX()": ".max()",
    "MIN()": ".min()",
    "AVG()": ".mean()",
    "SUM()": ".sum()",
}

# A line that is Pandas code: a table indexed by a quoted column, optionally
# followed by an aggregation. Covers both the SYNTAX line and the sample code.
CODE_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\['([A-Za-z_][A-Za-z0-9_]*)'\](\.[a-z]+\(\))?$")


def multi_replace(text: str, mapping: dict) -> str:    
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in keys))
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def selector(operator: str, column: str) -> str:    
    return f"['{column}']" if operator == "[]" else operator.replace("()", f"('{column}')")


# def aggregation(operator: str) -> str:    
#     return operator


def rename(text: str, original_operator: str, new_operator: str, select_target: str) -> str:
    out = []
    for line in text.split("\n"):
        match = CODE_LINE.match(line)
        if match:
            table, column, agg = match.groups()
            code = table + selector(select_target, column)
            if agg is not None:
                code += new_operator
            out.append(code)
        else:
            out.append(multi_replace(line, {original_operator: new_operator,  "[]": select_target}))
    return "\n".join(out)


def build_version(version: int, src_dir: Path, sql_dir: Path, out_dir: Path) -> None:
    with open(sql_dir / "matches.json", "r", encoding="utf-8") as f:
        sql_matches = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_match = []
  
    select_target = next(SQL_TO_PANDAS[v] for e in sql_matches for k, v in e.items() if k == "SELECT")

    for entry in sql_matches:
        (sql_original, sql_new), = entry.items()
        original_operator = SQL_TO_PANDAS[sql_original]
        new_operator = SQL_TO_PANDAS[sql_new]

        src_file = src_dir / f"{original_operator}.txt"
        with open(src_file, "r", encoding="utf-8") as f:
            original_content = f.read()

        new_content = rename(original_content, original_operator, new_operator, select_target)

        out_name = f"{new_operator}_v3.txt"
        with open(out_dir / out_name, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"{out_name}  <--  {src_file.name}")

        all_match.append({original_operator: new_operator})

    with open(out_dir / "matches.json", "w", encoding="utf-8") as f:
        json.dump(all_match, f, indent=2)


def main():
    versions = [int(v) for v in sys.argv[1:]] or [1, 11, 24, 35, 37]

    for version in versions:
        print(f"Permutation (version {version}):")
        build_version(version,
            src_dir=HERE / "Documentation" / "Pandas" / "Dataframe",
            sql_dir=HERE / "Documentation" / "SQL" / f"SQL_swap_{version}",
            out_dir=HERE / "Documentation" / "Pandas" / f"Dataframe_swap_{version}")
        print(f"\nWrote 5 files to Dataframe_swap_{version}/\n")


if __name__ == "__main__":
    main()
