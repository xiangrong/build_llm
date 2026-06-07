#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import argparse
from transformers import BertTokenizer
from model import GPT, GPTConfig
from config import Config

def print_banner():
    banner = """
============================================================
           🚀 GPT-2 Chinese Model Generation CLI 🚀
============================================================
    """
    print(banner)

def load_model(checkpoint_path, device):
    print(f"[*] 正在从 {checkpoint_path} 加载模型...")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 未找到 Checkpoint 文件: {checkpoint_path}")
        print("请确认您的 checkpoint_path 正确，或已放置 gpt2_zh_latest.pt 在当前目录下。")
        sys.exit(1)
        
    try:
        # 加载 checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"[ERROR] 加载文件失败: {e}")
        sys.exit(1)

    # 兼容不同保存格式
    state_dict = None
    model_config = None
    step = "未知"
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            step = checkpoint.get('step', '未知')
            print(f"[+] 检测到包含元数据的 Checkpoint (训练步数: {step})")
        else:
            state_dict = checkpoint
            print("[+] 检测到直接保存的 State Dict 格式")
            
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            print("[+] 成功载入 Checkpoint 包含的模型结构配置")
    else:
        state_dict = checkpoint
        print("[+] 检测到直接保存的 State Dict 格式")

    # 如果没有在 Checkpoint 中找到配置，使用本地 Config 类的参数初始化
    if model_config is None:
        print("[*] 未在 Checkpoint 中找到配置参数，正在使用本地 config.py 的配置初始化...")
        conf = Config()
        model_config = GPTConfig(
            vocab_size=getattr(conf, 'vocab_size', 21128),
            block_size=getattr(conf, 'block_size', 512),
            n_layer=getattr(conf, 'n_layer', 12),
            n_head=getattr(conf, 'n_head', 12),
            n_embd=getattr(conf, 'n_embd', 768)
        )
    
    # 清理 DDP (module.) 或 torch.compile (_orig_mod.) 的前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if k.startswith('_orig_mod.'):
            name = k[10:]
        if name.startswith('module.'):
            name = name[7:]
        clean_state_dict[name] = v

    # 初始化 GPT 实例
    try:
        model = GPT(model_config)
        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()
        print(f"[+] 模型初始化成功。参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    except Exception as e:
        print(f"[ERROR] 模型加载权重失败，请检查配置文件参数是否与模型匹配！错误信息: {e}")
        sys.exit(1)
        
    return model, model_config, step

@torch.no_grad()
def generate_stream(model, tokenizer, idx, max_new_tokens, temperature=0.8, top_k=20, device='cpu'):
    """流式生成器，逐个 token 产生并返回"""
    for _ in range(max_new_tokens):
        # 截断以匹配模型支持的最大 block_size
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        
        # 获取最后一个位置的 logits 并按温度缩放
        logits = logits[:, -1, :] / temperature
        
        # Top-K 采样过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # 计算概率分布并采样
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 追加到当前序列中
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Yield 当前生成的 token id
        yield idx_next[0, 0].item(), idx

def test_inference(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    # 编码 prompt，不添加特殊的 CLS/SEP
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        # 如果 prompt 为空，默认使用 [CLS]
        input_ids = [tokenizer.cls_token_id]
        
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    print("\n" + "="*50)
    print(f"输入提示词: {prompt}")
    print("模型续写中: ", end="", flush=True)
    
    start_time = time.time()
    tokens_generated = 0
    
    # 获取流式生成器
    generator = generate_stream(model, tokenizer, idx, max_new_tokens, temperature, top_k, device)
    
    for token_id, idx_updated in generator:
        char = tokenizer.decode([token_id])
        # 过滤掉 BERT 分词器中的特殊符号或处理子词前缀
        if char == "[PAD]" or char == "[SEP]" or char == "[CLS]":
            continue
        # BERT tokenizer 的英文子词前缀去掉空格
        if char.startswith("##"):
            print(char[2:], end="", flush=True)
        else:
            # 正常输出字符
            print(char, end="", flush=True)
        tokens_generated += 1
        
    elapsed = time.time() - start_time
    speed = tokens_generated / elapsed if elapsed > 0 else 0
    
    print("\n" + "="*50)
    print(f"生成统计: 共生成 {tokens_generated} tokens | 耗时 {elapsed:.2f}秒 | 速度 {speed:.2f} tokens/s\n")

def main():
    parser = argparse.ArgumentParser(description="GPT2-Chinese Model Generation CLI")
    parser.add_argument("--checkpoint", type=str, default="gpt2_zh_latest.pt", help="Checkpoint 路径")
    parser.add_argument("--prompt", type=str, default="", help="测试提示词（若留空则进入交互式命令行模式）")
    parser.add_argument("--max_len", type=int, default=150, help="最大新生成 token 数量")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (温度)，越小生成越确定，越大越随机")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K 采样参数")
    args = parser.parse_args()
    
    print_banner()
    
    # 自动选择最快设备 (MPS 用于 Mac, CUDA 用于 Nvidia GPU, 否则 CPU)
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon MPS (Metal Performance Shaders)"
    else:
        device = "cpu"
        device_name = "CPU"
        
    print(f"[*] 当前使用设备: {device} ({device_name})")
    
    # 加载 Tokenizer
    try:
        conf = Config()
        tokenizer_path = getattr(conf, 'tokenizer_path', 'bert-base-chinese')
        print(f"[*] 正在从 '{tokenizer_path}' 加载分词器...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"[ERROR] 无法加载分词器: {e}")
        print("提示: 请确认网络是否能够连接 Hugging Face，或者已经缓存了 bert-base-chinese。")
        sys.exit(1)
        
    # 加载模型
    model, model_config, step = load_model(args.checkpoint, device)
    
    # 单次生成模式
    if args.prompt:
        test_inference(model, tokenizer, args.prompt, args.max_len, args.temp, args.top_k, device)
        return
        
    # 交互式对话模式
    print("\n[+] 已进入交互模式！输入 'exit' 或 'quit' 退出。")
    print("[+] 您可以直接输入开头，让模型为您续写文本。")
    
    # 默认值
    temp = args.temp
    top_k = args.top_k
    max_len = args.max_len
    
    while True:
        try:
            print(f"[设置: temp={temp}, top_k={top_k}, max_len={max_len}]")
            user_input = input("请输入提示词 >>> ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("[*] 退出程序，再见！")
                break
                
            # 支持动态修改参数，例如以 / 开头的命令
            if user_input.startswith("/temp "):
                try:
                    temp = float(user_input.split()[1])
                    print(f"[*] 温度参数已修改为: {temp}")
                    continue
                except:
                    print("[!] 参数格式错误，例如: /temp 0.7")
                    continue
            elif user_input.startswith("/topk "):
                try:
                    top_k = int(user_input.split()[1])
                    print(f"[*] top_k 已修改为: {top_k}")
                    continue
                except:
                    print("[!] 参数格式错误，例如: /topk 40")
                    continue
            elif user_input.startswith("/len "):
                try:
                    max_len = int(user_input.split()[1])
                    print(f"[*] 生成长度已修改为: {max_len}")
                    continue
                except:
                    print("[!] 参数格式错误，例如: /len 200")
                    continue
            elif user_input.startswith("/help"):
                print("--- 快捷指令说明 ---")
                print("  /temp <float> : 修改采样温度（0-1之间，默认0.8）")
                print("  /topk <int>   : 修改 Top-K 采样数（默认30）")
                print("  /len <int>    : 修改每次生成的最大 token 数量（默认150）")
                print("  exit / quit   : 退出程序")
                print("-------------------")
                continue
                
            test_inference(model, tokenizer, user_input, max_len, temp, top_k, device)
            
        except KeyboardInterrupt:
            print("\n[*] 退出程序，再见！")
            break
        except Exception as e:
            print(f"[!] 发生错误: {e}")

if __name__ == "__main__":
    main()
