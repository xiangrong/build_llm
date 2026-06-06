import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
import requests
import json
from torch.nn import functional as F
from transformers import BertTokenizer
from tqdm import tqdm

# ==========================================
# 1. 配置参数 (Config)
# ==========================================
class Config:
    vocab_size = 21128
    block_size = 512
    n_layer = 12
    n_head = 12
    n_embd = 768
    batch_size = 16
    learning_rate = 3e-4
    max_iters = 100000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_name = "bert-base-chinese"
    checkpoint_path = "gpt2_zh_latest.pt"

# ==========================================
# 2. 模型架构 (同前，包含 generate)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        pos = torch.arange(0, idx.size(1), dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h: x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 3. 强力数据准备 (多源下载 + 自动兜底)
# ==========================================
def prepare_data():
    tokenizer = BertTokenizer.from_pretrained(Config.tokenizer_name)
    all_tokens = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 数据源列表
    sources = [
        "https://raw.githubusercontent.com/brightmart/nlp_chinese_corpus/master/datasets/wiki_zh/wiki_zh_small.txt",
        "https://huggingface.co/datasets/shibing624/medical/resolve/main/pretrain_medical_zh_0.json"
    ]
    
    print("Gathering data...")
    for url in sources:
        try:
            print(f"Downloading {url}...")
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                if url.endswith('.json'):
                    data = r.json()
                    for item in tqdm(data[:5000], desc="JSON"):
                        text = item.get('text', '')
                        if text: all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
                else:
                    lines = r.text.split('\n')
                    for line in tqdm(lines[:10000], desc="Text"):
                        if line.strip(): all_tokens.extend(tokenizer.encode(line.strip(), add_special_tokens=False))
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    # 🆘 核心防线：如果下载失败，生成 100 万 Token 的高质量合成语料
    if len(all_tokens) < 1000:
        print("!!! Warning: External data failed. Generating 1M token fallback corpus !!!")
        base_texts = [
            "人工智能是引领未来的战略性技术。", "GPT模型通过深度学习掌握了语言的规律。",
            "我们在Kaggle平台上使用GPU进行高效的大规模模型训练。", "中文语料的质量直接决定了对话机器人的表现。",
            "深度学习领域的研究正在日新月异地发展。", "自然语言处理是人工智能皇冠上的明珠。"
        ]
        # 重复 20000 次以确保数据量足够大（约 100 万 tokens）
        fallback_text = "。".join(base_texts) * 20000
        all_tokens = tokenizer.encode(fallback_text, add_special_tokens=False)

    print(f"Final Data Scale: {len(all_tokens)} tokens")
    # 只有在有数据时才写入
    if len(all_tokens) > 0:
        np.array(all_tokens, dtype=np.uint16).tofile("train.bin")
        print("train.bin saved.")
    else:
        raise ValueError("Critical Error: Failed to collect any tokens!")

# ==========================================
# 4. 训练与生成测试
# ==========================================
def main():
    # 如果文件不存在或太小，重新生成
    if not os.path.exists("train.bin") or os.path.getsize("train.bin") < 1024:
        prepare_data()
        
    data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    tokenizer = BertTokenizer.from_pretrained(Config.tokenizer_name)
    model = GPT(Config).to(Config.device)
    
    if os.path.exists(Config.checkpoint_path):
        print("Loading existing checkpoint...")
        model.load_state_dict(torch.load(Config.checkpoint_path, map_location=Config.device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    
    def get_batch():
        max_idx = len(data) - Config.block_size - 1
        ix = torch.randint(0, max_idx, (Config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+Config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+Config.block_size]).astype(np.int64)) for i in ix])
        return x.to(Config.device), y.to(Config.device)

    print("Training started...")
    model.train()
    for step in range(Config.max_iters):
        t0 = time.time()
        x, y = get_batch()
        with torch.amp.autocast('cuda'):
            logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 50 == 0:
            print(f"Step {step} | Loss {loss.item():.4f} | Time {(time.time()-t0)*1000:.2f}ms")
            
        if step % 500 == 0:
            model.eval()
            context = torch.tensor([tokenizer.encode("人工智能", add_special_tokens=False)], device=Config.device)
            gen = model.generate(context, max_new_tokens=40)
            print(f"\n--- Step {step} Gen: {tokenizer.decode(gen[0].tolist())} ---\n")
            torch.save(model.state_dict(), Config.checkpoint_path)
            model.train()

if __name__ == "__main__":
    main()
