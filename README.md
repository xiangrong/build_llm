# build_llm: 在 Kaggle 从零开始训练中文 GPT-2 大模型

本项目提供了一套完整的、基于 **Kaggle 双卡 GPU (Dual T4 DDP)** 且针对**中文语料**从零开始预训练 **GPT-2 Small (124M 参数)** 大模型的极简实战方案。

---

## 🚀 核心特性

1. **标准 GPT-2 Small 架构**：包含 12 层 Transformer Block、12 个注意力头、768 维隐向量空间，采用 **Weight Tying** 共享嵌入层和输出层参数，降低显存占用。
2. **双卡 DDP 分布式并行加速**：使用 PyTorch `DistributedDataParallel` (DDP) 对 Kaggle 双 T4 GPU 进行多进程并行训练，大幅提高数据吞吐量。
3. **中文维基百科语料预处理**：一键拉取 Hugging Face 的 Wikipedia-zh 官方 Parquet 分片，使用预训练的 `bert-base-chinese` 分词器（词表大小 21,128）将语料转化为高效二进制数据 `train.bin`。
4. **9小时超时退出安全卫士**：针对 Kaggle 后台运行（Save & Run All）的最大 9 小时时限，在训练脚本中加入了计时逻辑。满 8.3 小时会自动保存最后的 Checkpoint 并正常退出，确保产物不丢失。
5. **Notebook 代码自动同步器**：本地采用模块化 Python 开发，通过生成脚本自动编译为一键运行的 Jupyter Notebook。

---

## 📁 目录结构

```text
build_llm/
├── config.py             # 核心模型参数与训练参数配置
├── model.py              # GPT 模型主体（含自回归 generate 文本生成方法）
├── prepare_data.py       # 中文 Wikipedia 数据下载与分词预处理脚本
├── train_ddp.py          # PyTorch DDP 双卡并行预训练脚本（含超时保护与断点续跑）
├── generate_notebook.py  # 笔记本生成器：读取上述 Python 文件并构建为 IPYNB
├── kaggle_train_ddp.ipynb# 交付产物：导入 Kaggle 一键运行的 Jupyter Notebook
├── test_model.py         # 本地模型前向传播和参数量的单元测试
├── dataset.py            # 数据集加载类与预处理样例
├── train.py              # 单卡模式训练参考脚本
└── .gitignore            # Git 忽略文件（已屏蔽权重模型 *.pt 及数据 *.bin）
```

---

## 🛠️ 本地开发与同步

1. **克隆项目到本地**：
   ```bash
   git clone git@github.com:xiangrong/build_llm.git
   cd build_llm
   ```

2. **本地修改代码**：
   您可以根据需要，修改模型架构 [model.py](model.py) 或核心配置 [config.py](config.py)。

3. **重新编译生成 Kaggle 笔记本**：
   在本地修改 Python 脚本后，运行以下命令，笔记本将自动提取最新代码：
   ```bash
   python3 generate_notebook.py
   ```

4. **提交并推送至 GitHub**：
   ```bash
   git add .
   git commit -m "Update model configs or scripts"
   git push origin main
   ```

---

## ☁️ Kaggle 部署与训练

### 第一步：导入 Notebook
1. 进入 Kaggle Notebook 页面，新建一个 Notebook。
2. 点击 **File** -> **Import Notebook**，上传本地生成的 [kaggle_train_ddp.ipynb](kaggle_train_ddp.ipynb)。

### 第二步：配置环境（切勿漏掉）
在右侧的 **Settings** 面板中设置：
* **Accelerator**：选择 **GPU T4 x2**（双卡加速）。
* **Internet on**：确保**开启 (On)**（首次需要绑定手机号验证。用于拉取分词器及维基百科数据分片）。

### 第三步：开始训练
您可以选择：
1. **交互式运行**：在 Notebook 界面直接点击 **Run All**。
2. **后台无忧运行（推荐）**：点击右上角的 **Save Version**，选择 **Save & Run All (Commit)**。代码将会自动运行并监控时间，在接近超时前优雅保存并自动结束。运行完毕后，可在 Notebook 的 **Output** 中直接下载训练出的 `gpt2_zh_latest.pt` 模型权重。
