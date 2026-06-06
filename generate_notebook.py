import json
import os

def create_notebook():
    notebook_path = "kaggle_train_ddp.ipynb"
    print("Reading local script files...")
    
    # Read script contents
    def read_file(fname):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Required file {fname} not found!")
        with open(fname, "r", encoding="utf-8") as f:
            return f.read()

    config_code = read_file("config.py")
    model_code = read_file("model.py")
    prepare_code = read_file("prepare_data.py")
    train_ddp_code = read_file("train_ddp.py")

    # Define cells
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 在 Kaggle 使用 Dual T4 GPUs 从零训练中文大模型\n",
                "\n",
                "本笔记本包含以下关键特性：\n",
                "1. **模型架构**: GPT-2 Small (124M参数)，中文词表大小 21,128，且使用 Weight Tying 技术节省显存。\n",
                "2. **双卡 DDP 加速**: 使用 PyTorch `DistributedDataParallel` 充分释放两张 T4 显卡算力，支持多进程并行训练。\n",
                "3. **数据读取**: 使用内存映射文件 `train.bin` 进行流式加载，支持百万至千万级别的中文维基百科（Wikipedia-zh）语料训练。\n",
                "4. **Kaggle 9小时超时防线**: 自动在运行满 8.3 小时前保存最终 checkpoint 并正常退出，避免因超时被系统清理导致产物丢失。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. 安装与检查依赖环境\n",
                "!pip install -q transformers datasets pandas pyarrow tqdm\n",
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"%%writefile config.py\n{config_code}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"%%writefile model.py\n{model_code}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"%%writefile prepare_data.py\n{prepare_code}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"%%writefile train_ddp.py\n{train_ddp_code}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 数据获取与中文分词预处理\n",
                "\n",
                "运行 `prepare_data.py`。该脚本会从 Hugging Face 下载中文维基百科 Parquet 数据分片，分词转换后输出成 `train.bin` 二进制映射文件。\n",
                "\n",
                "*注意：Kaggle 环境下在线下载极快。由于是第一次测试运行，可以在下方传入参数调整 token 的限制量，如在 prepare_data.py 修改 `max_tokens` 参数。*"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 运行数据预处理\n",
                "!python prepare_data.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 双卡分布式训练 (DDP)\n",
                "\n",
                "我们使用 PyTorch 提供的 `torchrun` 工具在一台机器的多张 GPU 上启动分布式训练。针对 Kaggle 的 2x T4 GPUs，设置 `--nproc_per_node=2`。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 启动分布式训练\n",
                "!torchrun --nproc_per_node=2 train_ddp.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 模型生成与推理测试\n",
                "\n",
                "加载保存好的 Checkpoint 并直接使用 `generate` 接口进行中文内容续写测试。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import torch\n",
                "from transformers import BertTokenizer\n",
                "from model import GPT, GPTConfig\n",
                "from config import Config\n",
                "\n",
                "conf = Config()\n",
                "tokenizer = BertTokenizer.from_pretrained(conf.tokenizer_path)\n",
                "\n",
                "model_config = GPTConfig(\n",
                "    vocab_size=conf.vocab_size,\n",
                "    block_size=conf.block_size,\n",
                "    n_layer=conf.n_layer,\n",
                "    n_head=conf.n_head,\n",
                "    n_embd=conf.n_embd\n",
                ")\n",
                "model = GPT(model_config).to(conf.device)\n",
                "\n",
                "if os.path.exists(conf.checkpoint_path):\n",
                "    checkpoint = torch.load(conf.checkpoint_path, map_location=conf.device)\n",
                "    model.load_state_dict(checkpoint['model'])\n",
                "    print(\"Loaded checkpoint successfully!\")\n",
                "    \n",
                "    model.eval()\n",
                "    context_text = \"人工智能是\"\n",
                "    print(f\"输入提示词: {context_text}\")\n",
                "    context = torch.tensor([tokenizer.encode(context_text, add_special_tokens=False)], device=conf.device)\n",
                "    gen = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=20)\n",
                "    print(f\"模型续写结果:\\n{tokenizer.decode(gen[0].tolist())}\")\n",
                "else:\n",
                "    print(f\"未找到 checkpoint 文件: {conf.checkpoint_path}\")"
            ]
        }
    ]

    # Notebook JSON structure
    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Write file
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2, ensure_ascii=False)
        
    print(f"Success! Jupyter notebook created at: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
