import argparse
import random
import re
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent


def multi_replace(text: str, mapping: dict) -> str:
    """Apply all replacements in `mapping` in a single pass so they can't cascade
    into each other. Longer keys take priority (e.g. 'AVG()' before 'AVG')."""
    mapping = {k: v for k, v in mapping.items() if k}
    if not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in keys))
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def build_derangement(n: int) -> list[int]:
    indices = list(range(n))
    while True:
        random.shuffle(indices)
        if any(indices[i] == i for i in range(n)):
            continue
        return indices

def apply_derangement(src_dir: Path, out_dir: Path, perm: list[int], files: list[Path], matches: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_match = []
    print(files)

    # The SELECT keyword appears in EVERY file's sample SQL, not just SELECT.txt.
    # Work out what SELECT maps to under this permutation once, so it is remapped
    # consistently everywhere. SELECT is the source for the dest position whose
    # perm entry points at SELECT's index.
    select_target = None
    select_idx = next((k for k, f in enumerate(files) if f.stem == "SELECT"), None)
    if select_idx is not None:
        # full operator name (with brackets) SELECT maps to, e.g. "MIN()"
        select_target = " ".join(files[perm.index(select_idx)].stem.split("_"))

    for i, dest_file in enumerate(files):
        src_file = files[perm[i]]

        with open(src_file, "r", encoding="utf-8") as f:
            original_content = f.read()

        original_operator = " ".join(src_file.stem.split("_"))       # e.g. "SELECT"
        original_operator_nobrack = original_operator.replace("(", "").replace(")", "")  # e.g. "SELECT"

        new_operator =  " ".join(dest_file.stem.split("_"))# e.g. "WHERE"
        new_operator_nobrack = new_operator.replace("(", "").replace(")", "")  # e.g. "WHERE"

        mapping = {
            original_operator: new_operator,                 # e.g. "AVG()" -> "MIN()"
            original_operator_nobrack: new_operator_nobrack,  # e.g. "AVG"   -> "MIN"
        }
        # Also remap the SELECT keyword embedded in every file's sample SQL,
        # using the full bracketed operator name so it matches the file name.
        if select_target is not None:
            mapping["SELECT"] = select_target
        new_content = multi_replace(original_content, mapping)

        out_name = dest_file.stem + "_v3" + dest_file.suffix   # e.g. "SELECT_v2.txt"
        with open(out_dir / out_name, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"{out_name}  <--  {src_file.name}")

    
        all_match.append({original_operator:new_operator})
    Path(matches).parent.mkdir(parents=True, exist_ok=True)
    with open(matches, "w", encoding="utf-8") as f:
        json.dump(all_match, f, indent=2)


def main():
    #parser = argparse.ArgumentParser()
    # parser.add_argument("--src", default="SQL", help="Source directory (relative to this script)")
    # parser.add_argument("--out", default="SQL_swap",help="Output directory (default: SQL_swap)")
    # parser.add_argument("--seed", type=int, default=None)
    # args = parser.parse_args()

    # if args.seed is not None:
    #     random.seed(args.seed)

    past_perms = []

    for j in range(44):
       
        src_dir = HERE / "Documentation" / "SQL" / "SQL_original"
        files = sorted(src_dir.glob("*.txt"))          

        #files = [f for f in files if f.stem in ["SELECT", "SUM()", "AVG()", "MIN()", "MAX()"]]  
        perm = build_derangement(len(files))
        while perm in past_perms:    
            perm = build_derangement(len(files))
        past_perms.append(perm)

        # Print the mapping before touching any files
        print("Permutation:")
        for i, f in enumerate(files):
            print(f"  {f.name}  <--  {files[perm[i]].name}")

        out_dir = HERE / "Documentation" / "SQL" / f"SQL_swap_{j+1}"       
        apply_derangement(src_dir, out_dir, perm, files, out_dir / "matches.json")
        print(f"\nWrote {len(files)} files to {out_dir}/")


if __name__ == "__main__":
    main()
