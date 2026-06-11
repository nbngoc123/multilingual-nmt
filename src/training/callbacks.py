# pyrefly: ignore [missing-import]
from transformers import EarlyStoppingCallback, TrainerCallback
import logging

logger = logging.getLogger(__name__)

def get_callbacks():
    """
    Khởi tạo danh sách các Callbacks sẽ can thiệp vào quá trình Training.
    """
    callbacks = []
    
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    callbacks.append(early_stopping)
    
    # Callback in Log tùy chỉnh
    callbacks.append(CustomLoggingCallback())
    
    return callbacks

class CustomLoggingCallback(TrainerCallback):
    """
    Callback in ra log đẹp mắt hơn trên Terminal.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Lọc bớt log rác
        if logs:
            _ = logs.pop("total_flos", None)
            # Chỉ in ở tiến trình chính (tránh in trùng lặp ở multi-GPU)
            if state.is_local_process_zero:
                logger.info(f"Step {state.global_step}: {logs}")
