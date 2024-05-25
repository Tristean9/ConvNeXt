# ConvNeXt模型复现

## 数据集

[Mini-Imagenet](https://paperswithcode.com/dataset/mini-imagenet)
[Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/)

## 项目结构

```text
ConvNeXt
├─ 📁dataset            # 数据集目录
|  ├─📁mini-imagenet    # Mini-ImageNet数据集
|  └─📁MRI              # MRI数据集
├─ 📁model              # 模型目录，存放构建的模型架构，包含瓶颈块。
├─ 📁trained_models     # 训练好的模型权重目录
├─ 📁log                # 日志目录，包含模型改造过程中的训练日志和评估结果。
├─ 📁utils              # 工具函数目录
│  ├─ 📄data_utils.py   # 数据处理相关的工具函数
│  └─ 📄log_utils.py    # 日志记录相关的工具函数
├─ 📄engine.py          # 训练和评估模型的引擎
├─ 📄main.py            # 主程序入口，控制整个训练流程
├─ 📄tune.py            # 模型微调脚本
├─ 📄visual.py          # 可视化脚本，用于展示训练过程中的结果
├─ 📄README.md          # 项目说明文档
└─ 📄environment.yml    # 项目conda环境依赖 
```

## 运行项目

1. **环境配置**：首先，请确保你已安装Anaconda或Miniconda。然后，使用以下命令创建项目所需的环境并激活：

    ```bash
    conda env create -f environment.yml
    conda activate ConvNeXt
    ```

2. **模型训练**：运行`main.py`脚本来开始训练模型。你可能需要在此脚本中设置训练相关的参数，例如数据路径、训练轮次（epoch）、学习率等。

    ```bash
    python main.py
    ```

3. **模型微调**：如果需要调优模型参数，可以运行`tune.py`脚本。

    ```bash
    python tune.py
    ```

4. **训练日志查看**：`log `目录下的每个目录中包含相应模型的训练日志，其中`.log`文件是训练的日志文件，其余文件是训练过程中生成TensorBoard日志，可以使用以下命令启动TensorBoard查看训练过程：

    ```bash
    tensorboard --logdir=./log/<相应模型日志目录>
    ```

    然后，你可以在浏览器中访问`http://localhost:6006`来查看训练过程中的各种指标和图表。
