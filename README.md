# ITNet: Improved Transformer Network for Enhanced Small Object Detection in UAV Imagery

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
