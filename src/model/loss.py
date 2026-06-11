# pyrefly: ignore [missing-import]
import torch
import torch.nn as nn

class BalancedLoss(nn.Module):
    """
    Hàm mất mát tự cân bằng dựa trên tần suất dữ liệu của từng cặp ngôn ngữ.
    Sử dụng Temperature Smoothing để điều chỉnh hình phạt cho nhóm dữ liệu hiếm.
    """
    def __init__(self, pair_counts: dict, smoothing_factor: float = 0.5):
        super().__init__()
        
        # Tính tổng số mẫu
        total = sum(pair_counts.values())
        
        # Tính toán trọng số cho mỗi ngôn ngữ
        self.weights = {}
        for pair, count in pair_counts.items():
            freq = count / total
            # Công thức làm mịn (smoothing): (1/freq) ^ smoothing_factor
            self.weights[pair] = (1.0 / freq) ** smoothing_factor
            
        # Chuẩn hóa trọng số (Normalize) để tổng trọng số không làm quá tải Gradient
        weight_sum = sum(self.weights.values())
        for pair in self.weights:
            self.weights[pair] = self.weights[pair] / weight_sum * len(self.weights)
            
        # Sử dụng reduction="none" để giữ nguyên loss của từng token chưa cộng gộp
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        
    def forward(self, logits, labels, pairs):
        """
        Args:
            logits: Đầu ra của model (batch_size, seq_len, vocab_size)
            labels: Nhãn đúng (batch_size, seq_len)
            pairs: Danh sách ngôn ngữ (batch_size,) chứa các chuỗi như "en-vi"
        """
        # Tính Loss cơ bản cho tất cả token
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Định hình lại loss về dạng (batch_size, seq_len)
        loss = loss.view(labels.size(0), -1) 
        
        # Chỉ tính toán các token không phải là -100 (tức là không phải Padding)
        mask = (labels != -100).float()
        
        # Tính trung bình loss cho TỪNG CÂU trong lô (batch)
        sample_losses = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        # Áp dụng trọng số (weight) dựa vào ngôn ngữ của từng câu
        device = logits.device
        pair_weights = torch.tensor(
            [self.weights.get(p, 1.0) for p in pairs], 
            dtype=torch.float32, 
            device=device
        )
        
        # Nhân loss của câu với trọng số tương ứng
        weighted_loss = sample_losses * pair_weights
        
        # Trả về trung bình của cả lô (Batch Mean Loss)
        return weighted_loss.mean()
