import torch

class Config:
    # 模型参数
    vocab_size = 21128  # bert-base-chinese vocab size
    block_size = 512    # 训练时可以先设为512节省显存，正式训练对齐gpt2-small用1024
    n_layer = 12
    n_head = 12
    n_embd = 768
    
    # 训练参数
    batch_size = 8      # 根据显存调整
    learning_rate = 6e-4
    max_iters = 100000
    eval_interval = 500
    save_interval = 2000
    eval_iters = 200
    
    # Kaggle 路径
    tokenizer_path = "bert-base-chinese"
    data_path = "train_data.txt" # 预处理后的文本文件
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "gpt2_zh_latest.pt"
