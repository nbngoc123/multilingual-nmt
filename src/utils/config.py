from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class DataConfig:
    lang_pairs: List[str]
    max_samples: int
    val_samples: int
    test_samples: int
    max_length: int

@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    num_epochs: int
    warmup_steps: int

def load_config(path: str) -> dict:
    """Đọc cấu hình từ file YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
