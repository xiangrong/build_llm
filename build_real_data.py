import os
import requests
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm

def download_and_build():
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    all_tokens = []
    
    # 维基百科中文版 20231101 分片下载链接 (Hugging Face 官方 Parquet)
    shards = [
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00000-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00001-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00002-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00003-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00004-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00005-of-00006.parquet"
    ]

    print(f"Starting to download {len(shards)} shards of Chinese Wikipedia...")
    
    for i, url in enumerate(shards):
        local_parquet = f"wiki_zh_{i}.parquet"
        
        # 下载
        if not os.path.exists(local_parquet):
            print(f"Downloading shard {i+1}/6...")
            r = requests.get(url, stream=True)
            with open(local_parquet, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # 读取并分词
        print(f"Processing shard {i+1}...")
        df = pd.read_parquet(local_parquet)
        # 提取 'text' 列
        texts = df['text'].tolist()
        
        # 为了防止内存溢出，我们逐条处理并限制总量
        # 建议先处理前 10 万条作为初次实验，全量训练可以去掉这个限制
        for text in tqdm(texts[:50000], desc=f"Tokenizing shard {i}"):
            if len(text) > 10:
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(ids)
                
        # 及时删除 Parquet 文件节省 Kaggle 磁盘空间
        os.remove(local_parquet)

    print(f"Success! Total tokens collected: {len(all_tokens)}")
    
    # 保存
    if len(all_tokens) > 0:
        np.array(all_tokens, dtype=np.uint16).tofile("train.bin")
        print("--- DONE: train.bin is ready! ---")
    else:
        print("Error: No tokens collected.")

if __name__ == "__main__":
    download_and_build()
