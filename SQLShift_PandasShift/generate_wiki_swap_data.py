import argparse
import random
import re
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def main():
    versions = [15, 16, 17, 18, 19]

    for version in versions:
        with open(HERE / "QuestionAnswer" / "Talkie_Train" / f"simple_tables_sql_wiki_version_{version}.jsonl", "r", encoding="utf-8") as f:
            tables = [json.loads(line) for line in f if line.strip()]
        
        with open(HERE / "Documentation" / "SQL" / f"SQL_swap_{version}" / "matches.json", "r", encoding="utf-8") as f:
             matches = json.load(f)
        matches_dict = {}
        for match in matches:
            key = list(match.keys())[0]
            value = list(match.values())[0]
            matches_dict[key] = value

        new_tables = []
        for data in tables:
            old_sql = data["sql"]

            if "SELECT" in old_sql:
                new_select = matches_dict["SELECT"]
            if "AVG" in old_sql:
                old_agg = "AVG"
                new_agg = matches_dict["AVG()"].replace("()", "")
            if "MIN" in old_sql:
                old_agg = "MIN"
                new_agg = matches_dict["MIN()"].replace("()", "")
            if "MAX" in old_sql:
                old_agg = "MAX"
                new_agg = matches_dict["MAX()"].replace("()", "")
            if "SUM" in old_sql:
                old_agg = "SUM"
                new_agg = matches_dict["SUM()"].replace("()", "")

            mapping = {}

            mapping["SELECT"] = new_select
            print(new_select)
            mapping[old_agg] = new_agg
            print(new_agg)

            pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
            new_sql = pattern.sub(lambda mo: mapping[mo.group(0)], old_sql)

            print(f"old_sql: {old_sql}")
            print(f"new_sql: {new_sql}")
            data["sql"] = new_sql
            new_tables.append(data)
        

        with open(HERE / "QuestionAnswer" / "Talkie_Train" / f"simple_tables_sql_wiki_version_{version}.jsonl", "w") as f:
            for dat in new_tables:
                f.write(json.dumps(dat) + "\n")        

        

if __name__ == "__main__":
    main()
