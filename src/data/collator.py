# pyrefly: ignore [missing-import]
import torch

def custom_collate_fn(features):
    """
    Hàm gom nhóm (batching) tùy chỉnh cho DataLoader.
    Đảm bảo cột 'pair' được giữ lại dưới dạng List[str] để tra cứu Balanced Loss,
    các cột dữ liệu khác chuyển thành Tensor cho mô hình.
    """
    batch = {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }
    
    if "pair" in features[0]:
        batch["pair"] = [f["pair"] for f in features]
        
    return batch
