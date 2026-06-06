import os
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from model import GPT, GPTConfig
from config import Config
from dataset import get_dataloader

def train():
    # 1. 配置与模型初始化
    conf = Config()
    model_config = GPTConfig(
        vocab_size=conf.vocab_size, 
        block_size=conf.block_size, 
        n_layer=conf.n_layer, 
        n_head=conf.n_head, 
        n_embd=conf.n_embd
    )
    model = GPT(model_config).to(conf.device)
    
    # 2. 优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=0.1)
    scaler = GradScaler() # 混合精度
    
    # 3. 数据加载 (假设数据已预处理为 train.bin)
    # 在 Kaggle 上训练前需确保数据文件存在
    if not os.path.exists("train.bin"):
        print("Waiting for train.bin data file...")
        return
        
    loader = get_dataloader("train.bin", conf.block_size, conf.batch_size)
    data_iter = iter(loader)
    
    model.train()
    start_time = time.time()
    
    for step in range(conf.max_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
            
        x, y = x.to(conf.device), y.to(conf.device)
        
        # 混合精度前向传播
        with autocast():
            logits, loss = model(x, y)
            
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        
        # 日志打印
        if step % 100 == 0:
            end_time = time.time()
            print(f"Step {step}: loss {loss.item():.4f}, time per step: {(end_time-start_time)/100:.3f}s")
            start_time = time.time()
            
        # 保存模型
        if step > 0 and step % conf.save_interval == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': model_config
            }
            torch.save(checkpoint, f"ckpt_step_{step}.pt")
            print(f"Saved checkpoint at step {step}")

if __name__ == "__main__":
    train()
