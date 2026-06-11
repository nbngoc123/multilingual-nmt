def get_sampling_kwargs(strategy: str, config: dict) -> dict:
    """
    Biến đổi chuỗi `strategy` thành các tham số dùng cho hàm `model.generate()`.
    Dựa hoàn toàn trên logic Top-p và Top-k từ bài học của AIO.
    """
    base_kwargs = {
        "max_length": config.get("max_length", 128)
    }
    
    # Nếu chọn greedy (Tham lam - luôn lấy xác suất cao nhất)
    if strategy == "greedy":
        base_kwargs["do_sample"] = False
        return base_kwargs
        
    # Từ đây trở xuống là các cơ chế có yếu tố Ngẫu nhiên (Sampling)
    base_kwargs["do_sample"] = True
    base_kwargs["temperature"] = config.get("temperature", 0.7)
    
    if strategy == "top_p":
        base_kwargs["top_p"] = config.get("top_p", 0.9)
    elif strategy == "top_k":
        base_kwargs["top_k"] = config.get("top_k", 50)
    elif strategy == "both":
        base_kwargs["top_p"] = config.get("top_p", 0.9)
        base_kwargs["top_k"] = config.get("top_k", 50)
    else:
        raise ValueError(f"Chiến lược '{strategy}' không được hỗ trợ. Vui lòng chọn: top_p, top_k, both, greedy")
        
    return base_kwargs
