import random
import numpy as np
import torch
# pyrefly: ignore [missing-import]
from transformers import set_seed as hf_set_seed

def set_seed(seed: int = 42):
    """
    Cố định seed cho tất cả các thư viện để đảm bảo kết quả huấn luyện
    luôn giống nhau ở mỗi lần chạy (Reproducible).
    """
    # 1. Cố định cho thư viện random mặc định của Python
    random.seed(seed)
    
    # 2. Cố định cho Numpy (rất hay dùng khi xử lý data)
    np.random.seed(seed)
    
    # 3. Cố định cho PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Đảm bảo các thuật toán cuDNN của GPU hoạt động một cách deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. Cố định seed cho thư viện Transformers của Hugging Face
    hf_set_seed(seed)
