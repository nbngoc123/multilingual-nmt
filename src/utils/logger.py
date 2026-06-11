import logging
import sys

def setup_logger(name: str = __name__, level=logging.INFO) -> logging.Logger:
    """
    Thiết lập cấu hình Logger chuẩn cho toàn bộ project.
    In log ra console với format rõ ràng kèm thời gian.
    """
    logger = logging.getLogger(name)
    
    # Tránh việc in log bị nhân đôi nếu hàm này vô tình được gọi nhiều lần
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)
    
    # Định dạng chuẩn của Log: [Thời gian] - [Tên module] - [Cấp độ] - [Nội dung]
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler xuất log ra màn hình console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Tránh lan truyền log lên Root Logger mặc định của Python
    logger.propagate = False
    
    return logger
