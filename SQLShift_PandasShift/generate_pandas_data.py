import argparse
import random
import re
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def main():
    with open(HERE / "QuestionAnswer" / "simple_tables_sql.jsonl", "r", encoding="utf-8") as f:
        tables = [json.loads(line) for line in f if line.strip()]

    new_tables = []
    for data in tables:
        old_sql = data["sql"]

        split_sql = old_sql.split()
        table_name = split_sql[-1]
        agg_col_name = split_sql[1]
        agg = agg_col_name.split("(")[0]

        if "AVG" in agg:
            new_agg = ".mean()"
        if "MIN" in agg:
            new_agg =  ".min()"
        if "MAX" in agg:
            new_agg =  ".max()"
        if "SUM" in agg:
            new_agg =  ".sum()"

        col_name = agg_col_name.replace(agg, "").replace("(", "").replace(")", "")

        pandas = f"{table_name}['{col_name}']{new_agg}"

        print(f"old_sql: {old_sql}")
        print(f"new_df: {pandas}")
        del data["sql"]
        data["pandas"] = pandas
        new_tables.append(data)
        #break
    

    with open(HERE / "QuestionAnswer" / "simple_tables_pandas.jsonl", "w") as f:
        for dat in new_tables:
            f.write(json.dumps(dat) + "\n")        

        

if __name__ == "__main__":
    main()
