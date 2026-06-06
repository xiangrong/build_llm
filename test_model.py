import torch
from model import GPT, GPTConfig
from config import Config

def test_forward_pass():
    print("Testing GPT-2 Small Forward Pass...")
    conf = Config()
    
    # 使用测试配置（稍微减小参数以便在本地快速运行）
    test_config = GPTConfig(
        vocab_size=conf.vocab_size,
        block_size=128, # 测试用短序列
        n_layer=4,      # 测试用少层数
        n_head=4,
        n_embd=256
    )
    
    model = GPT(test_config)
    model.eval()
    
    # 模拟输入数据 (Batch Size=2, Seq Length=128)
    # 随机生成一些 Token ID
    idx = torch.randint(0, conf.vocab_size, (2, 128))
    targets = torch.randint(0, conf.vocab_size, (2, 128))
    
    with torch.no_grad():
        logits, loss = model(idx, targets)
        
    print(f"Input Shape: {idx.shape}")
    print(f"Logits Shape: {logits.shape}") # 预期: [2, 128, vocab_size]
    print(f"Loss: {loss.item():.4f}")
    
    # 验证维度
    assert logits.shape == (2, 128, conf.vocab_size), "Logits shape mismatch!"
    assert loss is not None, "Loss should not be None when targets are provided!"
    
    print("Forward Pass Test Passed! ✅")

def test_parameter_count():
    print("\nChecking GPT-2 Small Parameter Count (Standard Scale)...")
    conf = Config()
    standard_config = GPTConfig(
        vocab_size=conf.vocab_size,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768
    )
    model = GPT(standard_config)
    
    n_params = sum(p.numel() for p in model.parameters())
    # GPT-2 Small 约为 124M 参数 (由于词表大小不同会有细微差异)
    print(f"Total Parameters: {n_params / 1e6:.2f}M")
    
    # 检查权重共享
    if model.transformer.wte.weight is model.lm_head.weight:
        print("Weight Tying: Success! (WTE and LM_HEAD share weights) ✅")

if __name__ == "__main__":
    test_forward_pass()
    test_parameter_count()
