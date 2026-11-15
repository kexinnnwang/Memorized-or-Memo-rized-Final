from __future__ import annotations

import argparse
import gzip
import json
from src.config import Config
from src.llm import LLM
from pathlib import Path
from src.data_loading import Document
from src.paths import QA_DIR, CUSTOM_TABLES_DIR, SQL_DIR, PANDAS_DIR, CONFIG_PATH, PROMPTS_DIR, RESULTS_DIR


class Conflict:
    def __init__(self, config: Config, language: str, batched: bool, batch_size: int = 8) -> None:
        self.config = config
        self.language = language
        self.batched = batched
        self.batch_size = batch_size

        self.llm = LLM(
            model_name=self.config.llm_model,
            device=self.config.device,
            hf_token=self.config.hf_token,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            quantize=self.config.quantize)

    def generate_documents(self, use_case:str, counterfactual_version:int, doclen:int) -> list[Document]:
        documents = []

        if self.language == "sql":
            actual_path = SQL_DIR / "SQL_original"
            counterfactual_path = SQL_DIR / f"SQL_swap_{counterfactual_version}"
        else:
            actual_path = PANDAS_DIR / "Dataframe"
            counterfactual_path = PANDAS_DIR / f"Dataframe_swap_{counterfactual_version}"

        if use_case == "reinforce":
            for file_path in actual_path.glob("*.txt"):
                operator_name = file_path.stem
                with open(file_path, "r", encoding="utf-8") as f:
                    documentation = f.read()
                    if doclen < 10:
                        documentation = documentation.split(f"--- Sample Code {doclen+1} ---")[0]
                    doc = Document(doc_id=operator_name,text=documentation,source=str(file_path))
                    documents.append(doc)
        else:
            for file_path in counterfactual_path.glob("*.txt"):
                operator_name = file_path.stem
                with open(file_path, "r", encoding="utf-8") as f:
                    documentation = f.read()
                    if doclen < 10:
                        documentation = documentation.split(f"--- Sample Code {doclen+1} ---")[0]
                    doc = Document(doc_id=operator_name,text=documentation,source=str(file_path))
                    documents.append(doc)

        return documents

    def run_conflict_experiment(self, use_case: str, doclen:int, counterfactual_version:int, output_file: str, top_p: float, num_rounds: int):
        output = {}
        documents = self.generate_documents(use_case, counterfactual_version, doclen)

        if self.language == "sql":
            qa_file = QA_DIR / "simple_tables_sql.jsonl"
            query_key = "sql"
            prompt_suffix = ""
        else:
            qa_file = QA_DIR / "simple_tables_pandas.jsonl"
            query_key = "pandas"
            prompt_suffix = "_pandas"

        for i in range(num_rounds):
            with open(qa_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]

            prompts = []
            metadata = []
            total_questions = 0

            for sample in data:
                question = sample["question"]
                gt = sample["gt"]
                gt_query = sample[query_key]
                table = sample["table"]
                table_name = list(table.keys())[0]

                if table_name == "employees":
                    with open(CUSTOM_TABLES_DIR / "employees.txt", "r", encoding="utf-8") as f:
                        database = f.read()
                elif table_name == "products":
                    with open(CUSTOM_TABLES_DIR / "products.txt", "r", encoding="utf-8") as f:
                        database = f.read()
                elif table_name == "orders":
                    with open(CUSTOM_TABLES_DIR / "orders.txt", "r", encoding="utf-8") as f:
                        database = f.read()
                elif table_name == "athletes":
                    with open(CUSTOM_TABLES_DIR / "athletes.txt", "r", encoding="utf-8") as f:
                        database = f.read()
                elif table_name == "movies":
                    with open(CUSTOM_TABLES_DIR / "movies.txt", "r", encoding="utf-8") as f:
                        database = f.read()
                elif table_name == "books":
                    with open(CUSTOM_TABLES_DIR / "books.txt", "r", encoding="utf-8") as f:
                        database = f.read()
                elif table_name == "restaurants":
                    with open(CUSTOM_TABLES_DIR / "restaurants.txt", "r", encoding="utf-8") as f:
                        database = f.read()
                elif table_name == "students":
                    with open(CUSTOM_TABLES_DIR / "students.txt", "r", encoding="utf-8") as f:
                        database = f.read()

                total_questions += 1

                if use_case == "reinforce":
                    with open(PROMPTS_DIR / f"simple_reinforce{prompt_suffix}.txt", "r", encoding="utf-8") as f:
                        user_prompt = f.read()
                elif use_case == "counterfactual":
                    with open(PROMPTS_DIR / f"simple_counterfactual{prompt_suffix}.txt", "r", encoding="utf-8") as f:
                        user_prompt = f.read()
                elif use_case == "instruct":
                    with open(PROMPTS_DIR / f"simple_instruct{prompt_suffix}.txt", "r", encoding="utf-8") as f:
                        user_prompt = f.read()
                elif use_case == "nosql":
                    with open(PROMPTS_DIR / f"simple_nosql{prompt_suffix}.txt", "r", encoding="utf-8") as f:
                        user_prompt = f.read()

                user_prompt = user_prompt.replace("{database}", database)
                user_prompt = user_prompt.replace("{question}", question)

                context_blocks = []
                for d in documents:
                    context_blocks.append(f"{d.text}")
                context = "\n\n".join(context_blocks)
                user_prompt = user_prompt.replace("{operators}", context)

                prompts.append(user_prompt)
                metadata.append({"sample_id": total_questions,
                    "question": question,
                    "gt": gt,
                    "gt_query": gt_query,
                    "databases": table})

            if self.batched:
                print(f"Generating {len(prompts)} questions in batches of {self.batch_size}")
                llm_answers = self.llm.generate_with_probs_batch("", prompts, top_p, False, batch_size=self.batch_size)
                for llm_answer in llm_answers:
                    print(llm_answer["generated_text"])
            else:
                llm_answers = []
                for p in prompts:
                    llm_answer = self.llm.generate_with_probs("", p, top_p, False)
                    print(llm_answer["generated_text"])
                    llm_answers.append(llm_answer)

            question_results = []
            for meta, llm_answer in zip(metadata, llm_answers):
                question_results.append({
                    "sample_id": meta["sample_id"],
                    "question": meta["question"],
                    "gt": meta["gt"],
                    "gt_query": meta["gt_query"],
                    "llm_answer": llm_answer,
                    "databases": meta["databases"]
                })

            output[f"Round {i+1}"] = {
                "experiment_settings": {
                    "llm_model": self.config.llm_model,
                    "use_case": use_case,
                    "counterfactual_version": counterfactual_version
                },
                "questions": question_results,
            }

        output_file = output_file + ".gz"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f'Saved output to {output_file}')



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm")
    parser.add_argument("--use_case", type=str)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--num_rounds", type=int)
    parser.add_argument("--language", choices=["sql", "pandas"], default="sql")
    parser.add_argument("--versions", type=int, nargs="+", default=[1, 11, 24, 35, 37, 7, 42, 27, 41, 17])
    parser.add_argument("--doclens", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    if args.use_case == "instruct" or args.use_case == "counterfactual":
        versions = args.versions
    else:
        versions = [0]

    if args.use_case == "nosql":
        doclen = [0]
    else:
        doclen = args.doclens

    myconfig = Config.create_config(CONFIG_PATH)
    if args.llm:
        myconfig.llm_model = args.llm
    model_name = myconfig.llm_model.split("/")[-1]

    raw_dir = "raw_sql" if args.language == "sql" else "raw_pandas"

    experiment = None
    for version in versions:
        for dl in doclen:
            if args.use_case == "instruct" or args.use_case == "counterfactual":
                output_file = f"{RESULTS_DIR}/main/{raw_dir}/{args.use_case}/version_{version}/doclen_{dl}/{model_name}.json"
            elif args.use_case == "reinforce":
                output_file = f"{RESULTS_DIR}/main/{raw_dir}/{args.use_case}/doclen_{dl}/{model_name}.json"
            else:
                output_file = f"{RESULTS_DIR}/main/{raw_dir}/{args.use_case}/{model_name}.json"

            if Path(output_file + ".gz").exists():
                print(f"Skipping (already done) {output_file}.gz")
                continue

            print(f"Running {output_file}")
            
            if experiment is None:
                experiment = Conflict(myconfig, language=args.language, batched=args.batched, batch_size=args.batch_size)
            experiment.run_conflict_experiment(use_case=args.use_case, doclen=dl, counterfactual_version=version, output_file=output_file, top_p=args.top_p, num_rounds = args.num_rounds)

if __name__ == "__main__":
    main()
