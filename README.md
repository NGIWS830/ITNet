# ITNet: Inductive Transformer Network with Feature Diffusion for UAV Small Object Detection

**ITNet** 是一个专为无人机（UAV）航拍图像中的小目标检测而设计的深度学习网络。本项目基于 [DEIM](https://github.com/Shihoa/Deim) / [RT-DETR](https://github.com/lyuwenyu/RT-DETR) 架构进行了改进，引入了多种新颖模块以增强对微小特征的提取和融合能力。

## 🚀 主要特性 (Key Features)

基于 `configs/cfg-improve/ITNet.yaml` 和项目代码，本项目包含以下核心改进：

* **增强型骨干网络 (Backbone):** 集成了 **MANet** 和 **InceptionDWBlock**，通过多孔径设计增强多尺度特征提取。
* **混合编码器 (Hybrid Encoder):**
    * 引入 **MetaFormer Block** 配合 **SHSA** (Shunted Self-Attention)，提高全局建模能力。
    * 使用 **FocusFeature** 模块强化关键区域特征。
    * 采用 **MFM (Multi-Scale Feature Modulation)** 模块进行更有效的特征融合。
    * 集成 **C2f Block** 提升特征流转效率。
* **高效解码器 (Decoder):** 采用 **DFINE Transformer** 解码器，实现快速且高精度的端到端检测。

## 🛠️ 环境准备 (Installation)

### 依赖项
请确保您的环境满足以下要求（建议使用 Python 3.8+ 和 PyTorch 1.10+）：

```bash
pip install -r requirements.txt

```
(注意：如果根目录没有 `requirements.txt`，请参考 `tools/benchmark/requirements.txt` 或手动安装 `torch`, `torchvision`, `pyyaml`, `tqdm`, `opencv-python`, `scipy` 等基础库)

## 📂 数据准备 (Data Preparation)

本项目支持 **COCO 格式** 的数据集。如果您使用的是 VisDrone2019 或 HIT-UAV 等无人机数据集，请确保已将其转换为标准的 COCO JSON 标注格式。

数据集目录结构建议如下：

```text
dataset/
  ├── annotations/
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  ├── train2017/
  └── val2017/
  ```
在配置文件（如 configs/dataset/visdrone_detection.yml）中修改路径以匹配您的数据位置。

🚅 训练 (Training)
使用 train.py 脚本开始训练。您可以通过 -c 指定配置文件。

单卡训练示例：

```bash
python train.py \
    -c configs/cfg-improve/ITNet.yaml \
    --use-amp \
    --seed 42 \
    --output-dir ./output/ITNet_exp
```
多卡分布式训练示例：

```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py \
    -c configs/cfg-improve/ITNet.yaml \
    --use-amp \
    --output-dir ./output/ITNet_exp
```
主要参数说明：

-c, --config: 配置文件路径 (e.g., configs/cfg-improve/ITNet.yaml)

-r, --resume: 从检查点恢复训练 (e.g., output/checkpoint.pth)

--use-amp: 启用自动混合精度训练 (推荐)

--tuning: 加载预训练权重进行微调

⚡ 推理与可视化 (Inference)
使用 tools/inference/torch_inf.py 对图片或视频进行检测推理。

命令示例：

```bash
python tools/inference/torch_inf.py \
    -c configs/cfg-improve/ITNet.yaml \
    -r output/ITNet_exp/best.pth \
    -i ./path/to/image_or_video \
    -o ./inference_results \
    -t 0.4 \
    -d 0
```
参数说明：

-i, --input: 输入图片路径、视频路径或文件夹路径

-r, --resume: 训练好的模型权重文件 (.pth)

-t, --thrh: 置信度阈值 (默认: 0.2)

-d, --device: 推理设备 (如 0 表示 cuda:0, cpu 表示使用 CPU)

📁 项目结构 (Project Structure)
```bash
ITNet/
├── configs/             # 配置文件 (模型架构, 数据集, 优化器等)
│   ├── cfg-improve/     # ITNet 核心架构配置
│   └── ...
├── engine/              # 核心引擎 (Trainer, Solver, Backbone, Modules)
├── tools/               # 工具脚本
│   ├── inference/       # 推理脚本 (torch_inf.py)
│   ├── deployment/      # ONNX/TensorRT 导出工具
│   └── visualization/   # 可视化工具
├── train.py             # 训练入口脚本
└── requirements.txt     # 依赖列表
```

📜 引用 (Citation)
如果您在研究中使用了本项目，请引用：

```bash
@article{ITNet2026,
  title={ITNet: Improved Transformer Network for Enhanced Small Object Detection in UAV Imagery},
  author={Fei Han},
  journal={Journal Name},
  year={2026}
}
```

🙏 致谢 (Acknowledgements)
本项目基于以下开源项目构建，感谢原作者的贡献：

RT-DETR

DEIM
