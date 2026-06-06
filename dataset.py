import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data_path, block_size):
        # 实际项目中，我们会先用 tokenizer 把文本转成 ids 存为 bin 文件
        # 这里演示加载预处理后的数据
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        # 提取 block_size 长度的序列作为输入 x
        # 提取对应的下一个 token 作为目标 y
        chunk = self.data[i:i + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

def get_dataloader(data_path, block_size, batch_size):
    dataset = TextDataset(data_path, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# 预处理脚本示例 (在 Kaggle 上运行一次即可)
def pretokenize_data(input_file, output_file, tokenizer_name):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    all_ids = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            ids = tokenizer.encode(line.strip(), add_special_tokens=False)
            all_ids.extend(ids)
    
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(output_file)
    print(f"Pretokenization finished. Saved to {output_file}")
