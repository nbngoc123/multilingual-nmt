import logging
from collections import Counter
# pyrefly: ignore [missing-import]
from datasets import load_dataset, DatasetDict, concatenate_datasets
from typing import List

logger = logging.getLogger(__name__)

def load_multilingual_dataset(lang_pairs: List[str], max_samples: int = 50000, val_samples: int = 1000, test_samples: int = 2000, cache_dir: str = "./data/raw"):
    """
    Tải và gộp dữ liệu từ nhiều cặp ngôn ngữ của OPUS-100 theo cấu trúc chuẩn.
    """
    all_train, all_val, all_test = [], [], []
    
    for pair in lang_pairs:
        logger.info(f"Đang tải dữ liệu OPUS-100 cho cặp: {pair}")
        src, tgt = pair.split("-")
        
        # Cấu hình dataset trong OPUS-100 phải xếp theo alphabet
        langs = sorted([src, tgt])
        config_name = f"{langs[0]}-{langs[1]}"
        
        ds = load_dataset("Helsinki-NLP/opus-100", config_name, cache_dir=cache_dir)
        
        # Giới hạn số mẫu
        train = ds["train"].select(range(min(len(ds["train"]), max_samples)))
        val = ds["validation"] if "validation" in ds else ds["test"].select(range(min(len(ds["test"]), val_samples)))
        test = ds["test"].select(range(min(len(ds["test"]), test_samples)))
        
        # Map dữ liệu để chỉ lấy đúng src và tgt
        train = train.map(lambda x: {"pair": pair, "src": x["translation"][src], "tgt": x["translation"][tgt]}, remove_columns=["translation"])
        val = val.map(lambda x: {"pair": pair, "src": x["translation"][src], "tgt": x["translation"][tgt]}, remove_columns=["translation"])
        test = test.map(lambda x: {"pair": pair, "src": x["translation"][src], "tgt": x["translation"][tgt]}, remove_columns=["translation"])
        
        all_train.append(train)
        all_val.append(val)
        all_test.append(test)
        
    final_dict = DatasetDict({
        "train": concatenate_datasets(all_train).shuffle(seed=42),
        "validation": concatenate_datasets(all_val),
        "test": concatenate_datasets(all_test)
    })
    
    # In phân phối dữ liệu
    pair_counts = Counter(final_dict["train"]["pair"])
    logger.info(f"Phân phối dữ liệu Train: {dict(pair_counts)}")
    
    return final_dict
