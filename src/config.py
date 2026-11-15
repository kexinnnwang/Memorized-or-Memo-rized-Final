from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    #Paths
    data_dir: Path = Path("test_data")
    device: str = "cuda"
    quantize: bool = False

    #Models
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    hf_token: str | None = os.getenv("HF_TOKEN")

    #LLM generation
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = False

    cpt_num_epochs: int =5
    cpt_learning_rate: float = 1e-5
    cpt_max_length: int = 4096
    cpt_use_checkpoint: bool = True
    cpt_save_path: str | None = None

    lora_r: int = 6
    lora_dropout: int = 0.05
    lora_alpha: float = 8
    lora_num_epochs: int = 5
    lora_learning_rate: float = 1e-4
    lora_max_length: int = 4096

    #Experiment
    experiment_rounds: int = 10
  
    @classmethod
    def create_config(cls, path: str | Path) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)