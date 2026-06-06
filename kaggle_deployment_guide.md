# Kaggle平台部署与训练指南

我们已经成功完成了模型适配、DDP多进程脚本以及数据处理逻辑的编写。为了方便您在一键部署到 Kaggle 之前有一个清晰的认识，我们将项目文件和具体的部署运行步骤梳理如下。

---

## 1. 项目文件清单与作用

| 文件名 | 类型 | 说明 |
| :--- | :--- | :--- |
| [config.py](file:///Users/mac/project/build_llm/config.py) | Python 脚本 | 全局核心参数配置，包括 GPT 模型层数、头数、隐向量维度、学习率及 Checkpoint 保存路径等。 |
| [model.py](file:///Users/mac/project/build_llm/model.py) | Python 脚本 | 优化后的 GPT 模型主体。已补全自回归的 [GPT.generate](file:///Users/mac/project/build_llm/model.py#L127) 文本生成函数并支持共享权重（Weight Tying）。 |
| [prepare_data.py](file:///Users/mac/project/build_llm/prepare_data.py) | Python 脚本 | 流式下载 Hugging Face 上的中文维基百科 Parquet 数据分片，使用 BertTokenizer 编码并转换保存为二进制数据 `train.bin`。拥有自动删除缓存以防止磁盘爆满以及数据流控制。 |
| [train_ddp.py](file:///Users/mac/project/build_llm/train_ddp.py) | Python 脚本 | 双卡 DDP 并行训练主脚本。支持多卡数据独立采样、AMP 混合精度、WandB 接口预留、和 **8.3 小时安全退出防丢机制**。 |
| [generate_notebook.py](file:///Users/mac/project/build_llm/generate_notebook.py) | Python 脚本 | 自动构建器，读取本地上述 4 个 Python 脚本并自动组合打包成 Jupyter Notebook。 |
| [kaggle_train_ddp.ipynb](file:///Users/mac/project/build_llm/kaggle_train_ddp.ipynb) | Jupyter Notebook | **最终交付物**。您只需将此文件上传/导入至 Kaggle 即可。 |

---

## 2. Kaggle 平台运行三步走

### 第一步：导入 Notebook
1. 打开 [Kaggle](https://www.kaggle.com/) 并登录您的账号。
2. 点击左侧菜单栏的 **Create** 按钮 -> 选择 **New Notebook**。
3. 进入 Notebook 页面后，在顶部菜单栏点击 **File** -> 选择 **Import Notebook**。
4. 拖入或选择本地路径下的 [kaggle_train_ddp.ipynb](file:///Users/mac/project/build_llm/kaggle_train_ddp.ipynb) 文件进行上传导入。

### 第二步：配置硬件加速与网络（至关重要）
要在 Kaggle 上成功启动双卡 DDP 并行加速，您必须进行以下设置：
1. **GPU 切换**:
   * 点击 Notebook 右侧设置栏的 **Accelerator**。
   * 将默认的 "None" 切换为 **GPU T4 x2**（切勿选错，否则多卡启动 `torchrun` 会报错）。
2. **启用互联网 (Internet)**:
   * 在右侧设置栏中，找到 **Internet on** 开关，并确保其处于**开启 (On)** 状态。
   * *注意：首次使用此功能需要验证手机号，根据提示绑定即可。必须开启 Internet，脚本才能从外部网络下载 Hugging Face 的 Tokenizer 以及中文维基 Parquet 分片。*

### 第三步：启动执行与产物保存
1. 点击顶部工具栏的 **Run All**，或者按顺序手动点击运行每一个单元格。
   * 第一步环境安装需要约 1-2 分钟。
   * 数据下载与分词预处理会流式运行，默认上限为 50,000,000 tokens（可根据您对语料规模的需求在配置中修改）。
   * 接着，双卡 DDP 训练会使用 `torchrun --nproc_per_node=2 train_ddp.py` 启动。
2. **后台运行 (推荐)**:
   * 如果想关闭浏览器让模型自己训练，请点击 Notebook 右上角的 **Save Version**。
   * 选择 **Save & Run All (Commit)** 并点击 **Save**。
   * Kaggle 将在后台独立拉起容器运行，上限 9 小时。运行过程中您可以在 View Active Events 里面查看训练进度。
3. **获取训练好的模型**:
   * 当后台运行安全退出（或者训练完成）后，模型权重 `gpt2_zh_latest.pt` 会保存在 `/kaggle/working` 目录中。
   * 您可以直接在此 Notebook 的 **Output** 面板处将 `.pt` 权重文件下载至本地。

---

## 3. 进阶调优与规模扩充建议

### 3.1 预处理警告说明
在运行数据预处理时，如果控制台输出如下警告：
> *`Token indices sequence length is longer than the specified maximum sequence length... (7728 > 512)`*

**这是完全正常的**。该警告是因为维基百科单篇文章的 Token 数量超过了 BERT 默认的 512 限制。由于我们是将所有 Token 打碎后拼接输出到二进制文件 `train.bin` 中，并不直接作为 BERT 的输入，因此**对此任务没有任何负面影响，请直接忽略。**

### 3.2 语料规模扩充
目前的默认预处理限制了收集的 Token 上限为 `50,000,000` (5千万)，这足够快速跑通流程并验证模型学习语法规则的能力。如果后续您想训练出表达能力更强的大模型：
* **扩大参数**：在 `prepare_data.py` (或 Notebook 中执行该方法的单元格) 将 `max_tokens` 参数上调，例如修改为 `500,000,000` (5亿) 或直接取消限制，以处理完维基百科的全部 6 个 Parquet 分片。
* **修改位置**：
  ```python
  # 修改为更大量的 tokens (例如 5 亿)
  download_and_build(max_tokens=500_000_000)
  ```

### 3.3 多轮迭代续训 (Kaggle 断点续跑)
单次后台运行（Save & Run All）的 9 小时限制结束后，模型会安全退出并保存最新的权重：
1. **下载权重**：后台执行结束后，可以在 Notebook 的 Output 区域下载 `gpt2_zh_latest.pt`。
2. **多轮加载**：将当前阶段产出的 `gpt2_zh_latest.pt` 权重上传到 Kaggle 作为自定义 Dataset 挂载，或者在同一个 Working 目录下保留。当下一次再次启动 Notebook 训练时，脚本检测到该文件存在，便会自动通过 `torch.load` 恢复所有参数、优化器状态和步数，实现无缝断点续训。
