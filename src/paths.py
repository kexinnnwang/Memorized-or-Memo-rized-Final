from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "SQLShift_PandasShift"
SQL_DIR = DATA_DIR / "Documentation" / "SQL"
PANDAS_DIR = DATA_DIR / "Documentation" / "Pandas"
QA_DIR = DATA_DIR / "QuestionAnswer"
CUSTOM_TABLES_DIR = QA_DIR / "Custom_Tables"
TALKIE_TRAIN_DIR = QA_DIR / "Talkie_Train"

CONFIG_PATH = ROOT_DIR / "src" / "config.yaml"
PROMPTS_DIR = ROOT_DIR / "experiments" / "prompts"
RESULTS_DIR = ROOT_DIR / "results"
