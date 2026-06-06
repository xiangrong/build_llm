import os
import requests
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from config import Config

def download_and_build(max_tokens=50_000_000):
    conf = Config()
    tokenizer = BertTokenizer.from_pretrained(conf.tokenizer_path)
    all_tokens = []
    
    # Hugging Face hosted Parquet shards for Wikipedia zh 20231101
    shards = [
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00000-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00001-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00002-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00003-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00004-of-00006.parquet",
        "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00005-of-00006.parquet"
    ]

    print(f"Starting to download Chinese Wikipedia shards. Token limit: {max_tokens:,}")
    
    for i, url in enumerate(shards):
        if len(all_tokens) >= max_tokens:
            print(f"Reached token limit of {max_tokens:,}. Stopping download.")
            break
            
        local_parquet = f"wiki_zh_{i}.parquet"
        
        # Download shard
        if not os.path.exists(local_parquet):
            print(f"Downloading shard {i+1}/{len(shards)} from {url}...")
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                with open(local_parquet, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                print(f"Failed to download shard {i}: {e}")
                # Fallback to local files if already downloaded or fail gracefully
                if not os.path.exists(local_parquet):
                    continue
        
        # Load and tokenize
        print(f"Processing shard {i+1}...")
        try:
            df = pd.read_parquet(local_parquet)
            texts = df['text'].tolist()
            
            for text in tqdm(texts, desc=f"Tokenizing shard {i}"):
                if len(text) > 10:
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    all_tokens.extend(ids)
                    if len(all_tokens) >= max_tokens:
                        print(f"Reached token limit of {max_tokens:,} during processing.")
                        break
        except Exception as e:
            print(f"Error processing shard {i}: {e}")
        finally:
            # Clean up disk space
            if os.path.exists(local_parquet):
                os.remove(local_parquet)
                print(f"Removed temporary file {local_parquet}")

    print(f"Preprocessing completed! Total tokens collected: {len(all_tokens):,}")
    
    # Save tokens to train.bin
    if len(all_tokens) > 0:
        np.array(all_tokens, dtype=np.uint16).tofile("train.bin")
        print("--- SUCCESS: train.bin is ready! ---")
    else:
        # Emergency fallback data if completely offline or empty
        print("!!! Warning: No tokens collected. Creating a small dummy dataset to prevent crash !!!")
        fallback_texts = [
            "人工智能是引领未来的战略性技术。",
            "GPT模型通过深度学习掌握了语言的规律。",
            "我们在Kaggle平台上使用双卡GPU进行大语言模型分布式训练。"
        ] * 1000
        for text in fallback_texts:
            all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        np.array(all_tokens, dtype=np.uint16).tofile("train.bin")
        print("--- Fallback train.bin created successfully! ---")

if __name__ == "__main__":
    download_and_build()
