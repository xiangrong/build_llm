import os
import glob
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
import json

def build_chinese_data():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    all_tokens = []
    
    # 扫描 Kaggle 所有挂载的数据集
    # Kaggle 挂载路径一般为 /kaggle/input/<dataset-name>/...
    search_path = "/kaggle/input/**/*.txt"
    json_path = "/kaggle/input/**/*.json*"
    
    files = glob.glob(search_path, recursive=True) + glob.glob(json_path, recursive=True)
    
    if not files:
        print("!!! No datasets found in /kaggle/input !!!")
        print("Please click '+ Add Input' and search for 'Chinese Wikipedia'.")
        return

    print(f"Found {len(files)} files. Starting conversion...")
    
    # 限制处理总量，防止 Token 数量过大撑爆内存 (建议先搞个 1000 万 Token)
    max_tokens = 20_000_000 
    
    for fpath in files:
        if len(all_tokens) >= max_tokens: break
        
        print(f"Processing: {fpath}")
        try:
            if fpath.endswith('.txt'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    # 逐行处理防止内存溢出
                    for line in f:
                        text = line.strip()
                        if len(text) > 10:
                            ids = tokenizer.encode(text, add_special_tokens=False)
                            all_tokens.extend(ids)
                            if len(all_tokens) >= max_tokens: break
                            
            elif '.json' in fpath:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            # 尝试常见的中文数据集字段名
                            text = item.get('text', item.get('content', item.get('title', '')))
                            if len(text) > 10:
                                ids = tokenizer.encode(text, add_special_tokens=False)
                                all_tokens.extend(ids)
                                if len(all_tokens) >= max_tokens: break
                        except:
                            continue
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    print(f"Success! Collected {len(all_tokens)} real Chinese tokens.")
    
    # 保存为 train.bin
    if len(all_tokens) > 0:
        np.array(all_tokens, dtype=np.uint16).tofile("train.bin")
        print("train.bin is ready for high-quality training.")
    else:
        print("Failed to collect any tokens. Check dataset structure.")

if __name__ == "__main__":
    build_chinese_data()
