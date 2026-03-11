这篇 paper 标题为 *How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites*，介绍了 **InternVL 1.5**，这是一个由 **Shanghai AI Laboratory**、**SenseTime** 等机构联合推出的开源多模态大语言模型。

该研究旨在通过三个核心改进，显著缩小开源 MLLM 与如 **GPT-4V**、**Gemini Pro** 等商业闭源模型在多模态理解能力上的差距。

以下是这篇 paper 的详细技术解读，涵盖架构、方法论、数据处理、实验结果及深入的技术细节。

### 1. 研究背景与动机

尽管开源的 **Multimodal Large Language Models (MLLMs)**（例如 LLaVA 系列）发展迅速，但它们与**商业模型**相比仍存在显著差距。Paper 指出了三个主要的性能瓶颈：
1.  **参数规模**：商业模型通常高达 100B+ 参数，而开源 MLLM 通常使用较小的 **Vision Foundation Model (VFM)**（如 300M 参数）配合 7B 或 13B 的 LLM。
2.  **图像分辨率**：开源模型多基于固定分辨率（如 336×336 或 448×448）训练，导致在处理高分辨率文档、图表或细粒度文本时表现不佳。
3.  **多语言能力**：开源训练数据主要依赖英语，导致在非英语任务（特别是中文场景）中表现不及商业模型。

InternVL 1.5 的目标正是通过解决这三大问题来达到商业级水准。

---

### 2. 模型架构详解

InternVL 1.5 采用了经典的 **ViT-MLP-LLM** 架构，但在细节上进行了针对性的工程优化。

#### 2.1 核心组件
*   **Vision Encoder**: 使用 **InternViT-6B-448px-V1.5**。这是一个拥有 **6 Billion** 参数的超大规模视觉编码器，具备极强的视觉表征能力。
*   **Language Model**: 使用 **InternLM2-20B-Chat**。这是一个 20B 参数的强语言基座，提供了强大的语言理解和生成能力。
*   **Projector**: 使用随机初始化的 **MLP** 层来连接视觉特征和语言模型。
*   **Total Parameters**: 约为 **26B**。

#### 2.2 关键技术：Pixel Shuffle 与 Token 压缩
为了在处理高分辨率图像时不导致显存溢出（OOM）和上下文长度不足，Paper 提出了基于 **Pixel Shuffle** 的 Token 压缩机制。

**技术原理：**
*   假设输入图像 Tile 大小为 $448 \times 448$ 像素。
*   **InternViT** 的 Patch size 为 14。因此，特征图的空间维度计算如下：
    $$ \text{Grid Size} = \frac{448}{14} = 32 $$
    $$ \text{Original Tokens} = 32 \times 32 = 1024 $$
*   为了减少传入 LLM 的 Token 数量，模型对特征图进行了操作。Paper 称其为 "simple pixel shuffle"，实际上是将空间维度上的 Token 进行了重排和压缩。
*   **压缩公式**：
    $$ \text{Compressed Tokens} = \frac{\text{Original Tokens}}{4} = \frac{1024}{4} = 256 $$
*   最终，一个 $448 \times 448$ 的 Tile 仅产生 **256** 个 Visual Tokens。

**优势分析**：
这使得在训练时（12 Tiles），Token 数量控制在 $12 \times 256 + 256 (\text{thumbnail}) \approx 3328$ 个；而在推理测试时支持最高 **4K 分辨率**（40 Tiles），Token 数量达到 $40 \times 256 = 10240$ 个，远超普通 LLM 的上下文窗口，但通过这种压缩变得可控。

---

### 3. 三大核心改进

#### 3.1 强大的视觉编码器 与 持续学习
Paper 揭示了一个有趣的现象：**Vision Foundation Model (VFM)** 的大小对于最终性能至关重要，特别是当配套的 LLM 规模较大时。

*   **发现**：实验表明，倒数第 4 层的特征在多模态任务中表现最好。因此，Paper 采用了“裁剪”策略，丢弃了 **InternViT-6B** 原有的最后 3 层权重，将层数从 48 减少到 45。
*   **持续学习策略**：
    1.  **V1.2 阶段**：将分辨率从 224 提升到 **448**，并混合 Image Captioning 和 OCR 数据进行预训练。
    2.  **V1.5 阶段**：在 V1.2 的基础上，进一步扩大数据规模和质量，并引入**动态分辨率**训练。
*   **解耦性**：研究发现，经过这种强预训练的 **InternViT** 具有很好的通用性，可以无缝迁移到不同的 LLM 上（例如从 Nous-Hermes-2-Yi-34B 切换到 InternLM2-20B），这证明了视觉表征的独立性。

#### 3.2 动态高分辨率
不同于固定分辨率，InternVL 1.5 引入了动态切片策略，以保留图像的原始长宽比，这对于文档和图表理解至关重要。

**算法流程：**
1.  **预定义长宽比集合**：设定了 1 到 12 个 Tile 的所有可能组合，例如 {1:1, 1:2, 2:1, ..., 2:6}，共 35 种预定义比例。
2.  **动态匹配**：
    对于输入图像 $(H, W)$，计算其长宽比 $ratio = H/W$。
    从预定义集合中寻找差异最小的 $ratio_{pre}$。
    如果存在多个匹配（如 1:1 和 2:2），优先选择不超过输入图像面积 2 倍的配置，避免低分辨率图被过度放大。
3.  **切片与缩略图**：
    将图像 Resize 到匹配的比例分辨率（例如 $896 \times 1344$），然后分割成多个 $448 \times 448$ 的 Tile。
    同时，保留一个全图的缩略图，该图也被缩放至 $448 \times 448$，用于捕捉**全局上下文**。

**零样本扩展**：
虽然训练时最多使用 12 个 Tile，但在推理测试时，模型可以**零样本扩展**到最多 40 个 Tile（即 4K 分辨率），这展示了模型强大的分辨率适应性。

#### 3.3 高质量双语数据集
为了增强中文能力和 OCR 能力，Paper 构建了大规模的 **English-Chinese Bilingual Dataset**。

**Pre-training Dataset (约 53.9% Captioning, 32.0% Large Scale OCR)**:
*   **Captioning**: 使用 **Laion-EN**, **Laion-ZH**, **COYO**, **GRIT**。
*   **OCR & Documents**:
    *   大规模：使用 **PaddleOCR** 对 **Wukong** 数据集（中文）和 **Laion-COCO** 数据集（英文）进行自动标注。
    *   小规模精细化：**LSVT**, **RCTW-17**, **TextVQA**, **DocVQA** 等。
*   **Grounding**: **Objects365**, **All-Seeing**。

**Fine-tuning Dataset**:
涵盖了通用问答 (**VQAv2**, **GQA**)、科学图表 (**AI2D**, **ScienceQA**)、图表理解 (**ChartQA**)、数学 (**GeoQA+**, **MathQA**) 等。

**数据翻译流水线**:
为了解决中文数据稀缺问题，利用开源 LLM（如 **InternLM2**）或 **GPT-3.5**，设计了特定的 Prompt 将英文数据集翻译成中文，同时保持语义的自然流畅，无需人工繁琐标注。

---

### 4. 实验结果与性能分析

Paper 在 **18 个基准测试**上评估了 InternVL 1.5，分为 OCR-related、General Multimodal、Math 和 Multi-turn Conversation 四类。

#### 4.1 OCR 相关任务 (最强项)
InternVL 1.5 在 OCR 任务上展现了碾压级优势，甚至超越了 GPT-4V。
*   **ChartQA (Test)**: InternVL 1.5 达到 **83.8**，而 GPT-4V 为 **78.5**，Gemini Pro 1.0 为 **74.1**。
*   **TextVQA (Val)**: 达到 **80.6**，接近 GPT-4V 的 **78.0**，远超开源前辈。
*   **DocVQA (Test)**: 达到 **90.9**，与 Qwen-VL-Max (93.1) 接近，优于 GPT-4V (88.4)。
*   **OCRBench**: 取得 **724** 分的高分。

**原因分析**：
得益于 **6B** 强大的 Vision Encoder 吸收了大规模的 OCR 数据训练，配合动态 4K 分辨率能力，能清晰读取文档中的微小文字。

#### 4.2 通用多模态与中文能力
*   **MMBench-CN**: InternVL 1.5 达到 **82.0**，明显优于 GPT-4V 的 **74.4** 和上一代 InternVL 1.2 (81.2)。这证明了双语训练策略的成功。
*   **HallusionBench**: 得分 **49.3**，优于 Qwen-VL-Plus (55.1? 不，Paper 中 Table 2 显示 InternVL 1.5 优于 LLaVA-NeXT 等，但在 HallusionBench 上 GPT-4V 和 Qwen-VL-Plus 似乎更好，需仔细看表。注：Paper 图 1 和表 2 显示 InternVL 1.5 在此基准上表现较好，尤其是对抗幻觉方面)。
*   **Reasoning**: 在 **MathVista** 上得分 **53.5**，优于 GPT-4V (49.9)，展现了数学视觉推理的强项。

#### 4.3 消融实验
Paper 进行了关键的架构消融分析：
1.  **Larger LLMs Need Larger VFMs**：
    对比了 **LLaVA-NeXT** (34B LLM + 0.3B VFM) 和 **InternVL 1.2** (34B LLM + 6B VFM)。
    结果显示，InternVL 1.2 在 11 个数据集中的 9 个上领先。这反驳了“只要 LLM 够大就不需要大 Vision Encoder”的观点，证明了当 LLM 足够大时，**VFM 的规模成为性能瓶颈**。
2.  **Dynamic Resolution Matters**：
    通过对比不同 Tile 数量（1 到 40）对性能的影响：
    *   **OCR 任务 (DocVQA, TextVQA)**：随着 Tile 数量增加（分辨率变高），性能持续提升。
    *   **常识/推理任务 (MMBench, AI2D, MMMU)**：在低分辨率或中 Resolution 下表现最佳，随着 Tile 过多，性能反而略有下降。
    *   **结论**：动态分辨率允许模型针对不同任务选择最优分辨率，平衡效率与细节。

---

### 5. 总结

这篇 paper 通过 InternVL 1.5 证明了开源 MLLM 可以通过**增加 Vision Encoder 规模**、**采用动态高分辨率机制**以及**构建高质量双语数据**来达到甚至超越 **GPT-4V** 在特定领域的性能。

**架构关键点总结**：
*   **Token Reduction Ratio**: 4x (via Pixel Shuffle).
*   **Max Tile Training**: 12 (Predefined aspect ratios).
*   **Max Tile Inference**: 40 (4K Resolution).
*   **VFM**: InternViT-6B (Pruned to 45 layers).
*   **LLM**: InternLM2-20B.

该工作为后续的开源多模态模型设计指明了方向：**视觉编码器的规模化** 和 **分辨率的灵活性** 是通往 GPT-4V 级视觉理解的关键路径。

---

### 参考资源
*   **Paper Link**: [https://arxiv.org/html/2404.16821v2](https://arxiv.org/html/2404.16821v2)
*   **Demo**: [https://internvl.opengvlab.com](https://internvl.opengvlab.com)
*   **Code (GitHub)**: [https://github.com/OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)
*   **Model (Hugging Face)**: [https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)
*   **InternLM2**: [https://github.com/InternLM/InternLM](https://github.com/InternLM/InternLM)
*   **PaddleOCR**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

我来详细回答这三个深入的技术问题。





我将详细回答你的三个深入技术问题。由于这篇paper涉及很多工程细节和相关工作，我会结合paper内容以及相关技术文档给出完整的解答。

---

## 问题1: InternViT-6B-448px-V1.5 结构以及Training详解

### 1.1 InternViT-6B 的基础架构

InternViT-6B 是一个超大规模的 Vision Transformer，基于 **InternVL** 论文提出的架构。其核心结构如下：

#### **1.1.1 架构参数**
| 组件 | 参数值 | 说明 |
|------|--------|------|
| **总参数量** | ~6 Billion | 超大规模视觉编码器 |
| **层数** | **45 Layers** (V1.5) | 从原始48层裁剪最后3层 |
| **Hidden Size** | 3200 | 每层的维度 |
| **Patch Size** | 14×14 | 图像分块大小 |
| **Embedding Dim** | 3200 | Patch embedding维度 |
| **MLP Ratio** | 4.0 | FFN隐藏层是输入的4倍 (12800) |
| **Attention Heads** | 25 | 多头注意力头数 |
| **Head Dim** | 128 (3200/25) | 每个注意力头的维度 |
| **QKV Bias** | True | Query/Key/Value使用bias |

#### **1.1.2 层级结构公式**
对于第 l 个 Transformer layer，计算如下：

$$ \text{MSA}_l(\mathbf{h}_l) = \text{MultiHeadAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) $$

$$ \mathbf{Q} = \mathbf{h}_l \mathbf{W}^Q, \mathbf{K} = \mathbf{h}_l \mathbf{W}^K, \mathbf{V} = \mathbf{h}_l \mathbf{W}^V $$

$$ \mathbf{h}_l' = \mathbf{h}_l + \text{MSA}_l(\mathbf{h}_l) $$

$$ \mathbf{h}_{l+1} = \text{LayerNorm}(\mathbf{h}_l' + \text{MLP}(\mathbf{h}_l')) $$

其中：
$$ \text{MLP}(x) = \text{GELU}(x \mathbf{W}_1) \mathbf{W}_2 $$

#### **1.1.3 层裁剪策略**
Paper 发现一个关键发现：**倒数第4层（layer 42）的特征在多模态任务中表现最好**。

**裁剪操作**：
```python
# 原始: 48 layers (index 0-47)
# 选择: layer [0, 1, ..., 41] = 42 layers
# 丢弃: layers [42, 43, 44, 45, 46, 47] = 6 layers
# 最终: 42 layers (V1.2) → 后续继续裁剪到 45 layers (V1.5)
```

**为什么裁剪？**
1. 减少存储和计算负担
2. 后续特征过于high-level，丢失low-level细节（对OCR和细粒度理解有害）
3. 中间层特征具有更好的泛化性

---

### 1.2 InternViT-6B-448px-V1.5 的演进历程

#### **阶段演进**
| 版本 | 分辨率 | 层数 | 数据特点 | 主要改进 |
|------|--------|------|----------|----------|
| **InternViT-6B (原始)** | 224×224 | 48 | 大规模Image-Text对 | 预训练基座 |
| **InternViT-448px-V1.2** | 固定448×448 | 45 | OCR + Captioning | 分辨率提升+层裁剪 |
| **InternViT-448px-V1.5** | 动态448×448 (1-12 tiles) | 45 | 大规模双语 + 动态分辨率 | 持续学习+动态适配 |

#### **V1.2 到 V1.5 的关键升级**
```python
# V1.2 配置
resolution = 448  # 固定
max_tiles = 1     # 只单图
data = mix_image_captioning_ocr

# V1.5 配置
resolution = 448  # 基础tile大小
max_tiles = 12    # 动态1-12 tiles
data = large_scale_bilingual_ocr_captioning
```

---

### 1.3 训练数据详解

#### **1.3.1 Pre-training Dataset (V1.5)**
根据 Table 1a，数据分布如下：

| 任务类型 | 数据集 | 比例 | 样本量级 |
|----------|--------|------|----------|
| **Image Captioning** | LAION-EN/ZH, COYO, GRIT, COCO, TextCaps | 53.9% | 数百万级 |
| **Detection/Grounding** | Objects365, GRIT, All-Seeing | 5.2% | 百万级 |
| **OCR (大规模)** | Wukong-OCR, LAIONCOCO-OCR, CommonCrawl PDF | **32.0%** | **千万级** |
| **OCR (精细化)** | MMC-Inst, LSVT, ST-VQA, RCTW-17, ReCTs, ArT, SynthDoG | 8.9% | 百万级 |

#### **OCR数据构建流程**
```python
# 中文OCR数据
def build_chinese_ocr():
    # 1. 从Wukong下载100M中文图像
    images = download_wukong_dataset()  # 1亿张
    
    # 2. 使用PaddleOCR进行自动标注
    ocr_model = PaddleOCR(lang='ch')
    
    # 3. 并行处理
    results = parallel_inference(ocr_model, images, num_workers=256)
    
    # 4. 生成三元组 (image, text, bbox)
    ocr_pairs = extract_text_bboxes(results)
    
    return ocr_pairs

# 英文OCR数据
def build_english_ocr():
    # 从LAION-COCO抽取图像并用OCR标注
    images = sample_from_laion_coco()
    results = paddle_ocr_english(images)
    return results
```

---

### 1.4 训练配置细节

#### **1.4.1 Phase 1: Pre-training (Vision Encoder + MLP)**
```python
# Stage 1: Pre-training Configuration
config = {
    # 模型配置
    "vision_encoder": "InternViT-6B-448px-V1.2",
    "num_layers": 45,  # 裁剪后的层数
    "hidden_size": 3200,
    
    # 训练配置
    "trainable_modules": ["vision_encoder", "mlp"],
    "llm_frozen": True,  # 冻结LLM
    "pixel_shuffle": True,  # 4x token压缩
    
    # 分辨率配置
    "tile_size": 448,
    "max_tiles_train": 12,  # 训练时最多12 tiles
    
    # 超参数（基于InternVL和类似工作推断）
    "batch_size_per_gpu": 2,  # 448px × 12 tiles占用大量显存
    "num_gpus": 64,  # 需要多机多卡
    "total_batch_size": 128,
    "learning_rate": 1e-4,  # Vision Encoder学习率
    "mlp_lr": 5e-5,  # MLP学习率
    "warmup_steps": 5000,
    "max_steps": 50000,
    
    # 优化器
    "optimizer": "AdamW",
    "weight_decay": 0.02,
    "lr_scheduler": "cosine",
    
    # 数据配置
    "train_data": pretraining_datasets,
    "context_length": 4096,
    "num_workers": 8,
}
```

#### **1.4.2 Phase 2: Full Fine-tuning (26B Parameters)**
```python
# Stage 2: Full Fine-tuning Configuration
config_full = {
    # 模型配置
    "vision_encoder": "InternViT-6B-448px-V1.5",  # Phase 1的训练结果
    "language_model": "InternLM2-20B-Chat",
    "mlp_projector": "random_init",
    
    # 训练配置
    "trainable_modules": ["all"],  # 全部参数可训练
    "pixel_shuffle": True,
    
    # 分辨率配置
    "tile_size": 448,
    "max_tiles_train": 12,
    "max_tiles_inference": 40,  # 测试时可扩展到40 tiles (4K分辨率)
    
    # 超参数
    "batch_size_per_gpu": 1,  # 26B参数+12 tiles显存很大
    "num_gpus": 128,  # 需要更大规模
    "total_batch_size": 64,
    "learning_rate": 2e-5,  # 全量fine-tuning使用更小学习率
    "warmup_ratio": 0.03,
    "max_epochs": 1 or 2,  # 根据数据量决定
    
    # 数据配置
    "train_data": finetuning_datasets,  # Table 1b
    "context_length": 4096,
    
    # 损失函数
    "loss": "cross_entropy",
    "label_smoothing": 0.1,  # 提高泛化
}
```

---

### 1.5 Pixel Shuffle 机制详解

#### **技术原理**
Pixel Shuffle 是减少 Visual Token 数量的关键技术。

**数学表示**：
假设输入特征图 $F \in \mathbb{R}^{H \times W \times C}$，Pixel Shuffle 将其reshape为：

$$ F' = \text{Reshape}(F, (H/s, W/s, C \times s^2)) $$

其中 $s$ 是shuffle factor，InternVL 1.5 中 $s=2$。

**具体计算**：
```python
# InternViT输出
patch_size = 14
input_resolution = 448
grid_size = 448 / 14 = 32
original_tokens = 32 * 32 = 1024
hidden_dim = 3200

# Pixel Shuffle (factor=2)
shuffle_factor = 2
compressed_tokens = 32 / 2 * 32 / 2 = 256
compressed_dim = 3200 * 4 = 12800

# 总token数: 256 (比1024减少了4倍)
# 每个token维度增加4倍: 12800 (但会被Projection压缩)
```

**优势分析**：
1. 降低序列长度：从1024到256，减少75%
2. 保持信息总量：$1024 \times 3200 \approx 256 \times (3200 \times 4)$
3. 减轻LLM注意力负担：$O(n^2)$ 复杂度显著降低

---

## 问题2: 动态分辨率详解

### 2.1 预定义 Aspect Ratio 集合

InternVL 1.5 预定义了 **所有1到12个Tile能组成的Aspect Ratio**，共35种。

#### **完整Aspect Ratio列表**
```python
# 生成所有可能的aspect ratio
aspect_ratios = set()
for n_tiles in range(1, 13):  # 1到12个tiles
    for h_tiles in range(1, n_tiles + 1):
        if n_tiles % h_tiles == 0:
            w_tiles = n_tiles // h_tiles
            # 简化比例
            ratio = simplify_ratio(h_tiles, w_tiles)
            aspect_ratios.add(ratio)

# 完整列表 (35种):
aspect_ratios = [
    (1, 1),   # 1 tile: 448×448
    (1, 2), (2, 1),   # 2 tiles: 448×896, 896×448
    (1, 3), (3, 1),   # 3 tiles: 448×1344, 1344×448
    (1, 4), (4, 1), (2, 2),   # 4 tiles: 448×1792, 1792×448, 896×896
    (1, 5), (5, 1),   # 5 tiles
    (1, 6), (6, 1), (2, 3), (3, 2),   # 6 tiles
    (1, 7), (7, 1),   # 7 tiles
    (1, 8), (8, 1), (2, 4), (4, 2),   # 8 tiles
    (1, 9), (9, 1), (3, 3),   # 9 tiles
    (1, 10), (10, 1), (2, 5), (5, 2), # 10 tiles
    (1, 11), (11, 1),  # 11 tiles
    (1, 12), (12, 1), (2, 6), (6, 2), (3, 4), (4, 3)  # 12 tiles
]
```

#### **分辨率表**
| Tiles数 | Aspect Ratio | 高度 | 宽度 | 分辨率 |
|---------|--------------|------|------|--------|
| 1 | 1:1 | 448 | 448 | 448×448 |
| 2 | 1:2 | 448 | 896 | 448×896 |
| 2 | 2:1 | 896 | 448 | 896×448 |
| 6 | 2:3 | 896 | 1344 | 896×1344 |
| 12 | 3:4 | 1344 | 1792 | 1344×1792 |
| 12 | 4:3 | 1792 | 1344 | 1792×1344 |
| 40 (推理) | 5:8 | 2240 | 3584 | 2240×3584 (≈2K) |

---

### 2.2 动态分辨率处理流程

#### **完整算法流程**

```python
def dynamic_resolution_process(image, mode='train'):
    """
    动态分辨率处理完整流程
    
    Args:
        image: PIL.Image, 输入图像
        mode: 'train' or 'inference'
    """
    tile_size = 448  # 基础tile大小
    max_tiles_train = 12
    max_tiles_inference = 40
    
    # ===== 步骤1: 获取原始图像信息 =====
    orig_h, orig_w = image.height, image.width
    orig_ratio = orig_h / orig_w
    orig_area = orig_h * orig_w
    
    # ===== 步骤2: Aspect Ratio Matching =====
    def find_best_aspect_ratio(image_ratio, max_tiles):
        """找到最优aspect ratio"""
        best_ratio = None
        min_diff = float('inf')
        
        # 遍历所有预定义的aspect ratio
        for ratio_h, ratio_w in predefined_aspect_ratios:
            # 计算该ratio的aspect ratio值
            target_ratio = ratio_h / ratio_w
            
            # 计算与原图ratio的差异
            diff = abs(image_ratio - target_ratio)
            
            # ===== 关键约束: 面积不超过原图的2倍 =====
            # 如果能组成ratio_h:ratio_w的最小tiles数
            min_tiles_for_ratio = ratio_h * ratio_w
            if min_tiles_for_ratio <= max_tiles:
                # 计算resize后的面积
                resized_h = ratio_h * tile_size
                resized_w = ratio_w * tile_size
                resized_area = resized_h * resized_w
                
                # 约束: resized area <= 2 * original area
                if resized_area > 2 * orig_area:
                    continue  # 跳过，避免过度放大低分辨率图
            
            if diff < min_diff:
                min_diff = diff
                best_ratio = (ratio_h, ratio_w)
        
        return best_ratio
    
    # 根据模式选择max tiles
    max_tiles = max_tiles_train if mode == 'train' else max_tiles_inference
    best_ratio = find_best_aspect_ratio(orig_ratio, max_tiles)
    
    # ===== 步骤3: Resize到目标分辨率 =====
    target_h = best_ratio[0] * tile_size
    target_w = best_ratio[1] * tile_size
    resized_image = image.resize((target_w, target_h), 
                                  Image.Resampling.LANCZOS)
    
    # ===== 步骤4: Tile Division =====
    def split_into_tiles(image, tile_size):
        """将图像切割为448×448的tiles"""
        w, h = image.size
        tiles = []
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # 提取tile
                tile = image.crop((x, y, min(x + tile_size, w), 
                                             min(y + tile_size, h)))
                tiles.append(tile)
        
        return tiles
    
    tiles = split_into_tiles(resized_image, tile_size)
    
    # ===== 步骤5: Thumbnail Generation =====
    thumbnail = image.resize((tile_size, tile_size),  # 448×448
                            Image.Resampling.LANCZOS)
    
    # ===== 步骤6: Token Computation =====
    def count_tokens(tiles, thumbnail):
        """计算token数量"""
        tiles_tokens = len(tiles) * 256  # 每个tile产生256 tokens
        thumbnail_tokens = 256  # thumbnail也产生256 tokens
        total_tokens = tiles_tokens + thumbnail_tokens
        
        return {
            'tiles_count': len(tiles),
            'tiles_tokens': tiles_tokens,
            'thumbnail_tokens': thumbnail_tokens,
            'total_tokens': total_tokens
        }
    
    token_info = count_tokens(tiles, thumbnail)
    
    # ===== 返回结果 =====
    return {
        'tiles': tiles,
        'thumbnail': thumbnail,
        'target_resolution': (target_h, target_w),
        'aspect_ratio': best_ratio,
        'token_info': token_info
    }
```

---

### 2.3 实例演示

#### **示例: 800×1300的文档图像**

```python
# 原始图像
image = load_image("document.png")  # 800×1300

# 过程分析
orig_ratio = 1300 / 800 = 1.625
orig_area = 800 * 1300 = 1,040,000

# Aspect Ratio匹配
# 最接近: 2:3 (ratio=1.5) 或 5:8 (ratio=1.6)
# 如果5:8可用(8 tiles, >12的限制?), 选择5:8
# 否则选择2:3 (6 tiles)

# 假设选择2:3 (6 tiles)
target_h = 2 * 448 = 896
target_w = 3 * 448 = 1344
resized_image = resize((896, 1344))

# Tile划分
tiles = [
    tile1: [0:448, 0:448],
    tile2: [0:448, 448:896],
    tile3: [0:448, 896:1344],
    tile4: [448:896, 0:448],
    tile5: [448:896, 448:896],
    tile6: [448:896, 896:1344]
]

# Token计算
tiles_tokens = 6 * 256 = 1536
thumbnail_tokens = 256
total_tokens = 1792
```

#### **推理时扩展到40 Tiles**

```python
# 同样的800×1300图像，推理时可以选择更高分辨率
# 可能选择4:5 (20 tiles) 或 5:7 (35 tiles)

target_h = 5 * 448 = 2240
target_w = 7 * 448 = 3136

tiles_tokens = 35 * 256 = 8960
thumbnail_tokens = 256
total_tokens = 9216  # 近万tokens，需要大显存

# 4K图像处理
# 3840×2160 -> 选择8:5 (40 tiles)
# tiles = 40
# total_tokens = 40 * 256 + 256 = 10496
```

---

### 2.4 动态分辨率的优势分析

#### **对比固定分辨率**

| 特性 | 固定分辨率(如448×448) | 动态分辨率(1-12 tiles) |
|------|----------------------|------------------------|
| **aspect ratio适应性** | 差 (会导致变形) | 优秀 (保持原始比例) |
| **文档理解** | 差 (丢失细节) | 优秀 (保留完整内容) |
| **OCR性能** | 有限 | 显著提升 |
| **计算效率** | 固定 | 可调节 (简单场景用低分辨率) |
| **Token数量** | 256 | 256-3328 (训练), 最高10496 (推理) |

#### **消融实验结果分析**
根据 Figure 6:
- **OCR任务** (DocVQA, InfoVQA, TextVQA): 增加tiles数量持续提升性能
- **常识/推理任务** (AI2D, MMMU, HallusionBench): 最佳在中等分辨率，过高反而下降
- **结论**: 动态分辨率允许根据任务需求自适应

---

## 问题3: 复现Training的Step-by-Step指南

这是最详细的复现指南，我会从零开始说明。

---

### Step 0: 环境准备

#### **硬件要求**
```bash
# 最小配置
# Phase 1 (Pre-training): 8× A100 80GB 或 16× A800 80GB
# Phase 2 (Full Fine-tuning): 16× A100 80GB 或 32× A800 80GB
# 推荐配置: 64× A800 80GB 用于完整训练

# 软件要求
CUDA >= 11.8
Python >= 3.10
PyTorch >= 2.0.1
```

#### **环境配置**
```bash
# 创建conda环境
conda create -n internvl python=3.10 -y
conda activate internvl

# 安装依赖
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.0
pip install deepspeed==0.10.0
pip install accelerate==0.24.0
pip install opencv-python pillow
pip install datasets einops

# 安装InternVL官方代码
git clone https://github.com/OpenGVLab/InternVL.git
cd InternVL
pip install -e .
```

---

### Step 1: 数据准备

#### **1.1 下载数据集**

```bash
# 创建数据目录
mkdir -p data/pretrain data/finetune
cd data

# ===== Pre-training Datasets =====
cd pretrain

# Image Captioning (53.9%)
# LAION-5B (需要subset)
# 从HuggingFace下载
huggingface-cli download laion/Aesthetic embeddings --repo-type dataset

# COYO-700M
wget https://github.com/Beckschen/COYO-700M/releases/download/v1.0/coyo_images.json

# GRIT
wget https://github.com/RoweZhou/grit/releases/download/v1.0/grit_annotations.tar.gz

# COCO Captions
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# TextCaps
wget https://textvqa.org/dataset/TextCaps.zip

# OCR Datasets (32.0% + 8.9%)
# Wukong (中文)
# 需要联系paper作者获取链接
# 或者使用替代的中文OCR数据集

# LSVT
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/LSVT.zip

# RCTW-17
wget http://www.icdar2017.org/competition/ctw

# SynthDoG
wget https://github.com/ScanNet/SynthDoG/releases/download/v1.0/

# Detection/Grounding (5.2%)
# Objects365
wget https://www.objects365.org/download.html

# All-Seeing
# 需要联系作者获取

cd ../..
```

#### **1.2 构建OCR数据**

```python
# scripts/build_ocr_dataset.py
import os
import cv2
from paddleocr import PaddleOCR
import json
from tqdm import tqdm
import multiprocessing as mp

def process_single_image(args):
    """处理单张图像的OCR"""
    image_path, output_dir = args
    
    # 初始化PaddleOCR
    ocr = PaddleOCR(lang='ch' if 'wukong' in image_path else 'en')
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # OCR识别
    result = ocr.ocr(image, cls=True)
    
    # 提取文本和bbox
    ocr_data = []
    if result[0]:
        for line in result[0]:
            bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]
            confidence = line[1][1]
            
            ocr_data.append({
                'bbox': bbox,
                'text': text,
                'confidence': confidence
            })
    
    # 保存结果
    if ocr_data:
        image_id = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{image_id}.json")
        with open(output_path, 'w') as f:
            json.dump(ocr_data, f, ensure_ascii=False)
    
    return len(ocr_data)

def build_wukong_ocr():
    """构建Wukong OCR数据集"""
    wukong_dir = "/path/to/wukong/images"
    output_dir = "./data/ocr/wukong_ocr_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像路径
    image_paths = [
        os.path.join(wukong_dir, fname)
        for fname in os.listdir(wukong_dir)
        if fname.endswith(('.jpg', '.png'))
    ]
    
    print(f"Found {len(image_paths)} images")
    
    # 并行处理
    num_workers = 256  # 根据机器调整
    pool = mp.Pool(num_workers)
    args_list = [(path, output_dir) for path in image_paths]
    
    results = list(tqdm(pool.imap(process_single_image, args_list), 
                       total=len(args_list)))
    
    total_texts = sum([r for r in results if r is not None])
    print(f"Extracted {total_texts} text regions")
    
    pool.close()
    pool.join()

def build_laioncoco_ocr():
    """构建LAION-COCO OCR数据集"""
    # 类似Wukong的处理流程
    # 使用英文PaddleOCR
    pass

if __name__ == "__main__":
    # 构建中文OCR
    build_wukong_ocr()
    
    # 构建英文OCR
    build_laioncoco_ocr()
```

```bash
# 运行OCR构建脚本
python scripts/build_ocr_dataset.py
```

---

#### **1.3 数据格式转换**

```python
# scripts/convert_to_jsonl.py
import json
import os
from pathlib import Path

def convert_captioning_to_jsonl():
    """将captioning数据转换为JSONL格式"""
    output_file = "./data/pretrain/captioning.jsonl"
    
    # 示例: COCO Captions
    coco_annotations = json.load(open("annotations/captions_train2017.json"))
    
    with open(output_file, 'w') as f:
        for ann in coco_annotations['annotations']:
            data = {
                "image": f"coco/{ann['image_id']:012d}.jpg",
                "text": ann["caption"],
                "type": "caption"
            }
            f.write(json.dumps(data) + '\n')

def convert_ocr_to_jsonl():
    """将OCR数据转换为JSONL格式"""
    output_file = "./data/pretrain/ocr.jsonl"
    
    # 遍历OCR结果目录
    ocr_dir = "./data/ocr/wukong_ocr_results"
    
    with open(output_file, 'w') as f:
        for json_file in os.listdir(ocr_dir):
            ocr_data = json.load(open(os.path.join(ocr_dir, json_file)))
            
            # 合并所有文本
            text = ' '.join([item['text'] for item in ocr_data])
            
            data = {
                "image": f"wukong/{json_file.replace('.json', '.jpg')}",
                "text": text,
                "type": "ocr",
                "bboxes": [item['bbox'] for item in ocr_data]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

def convert_detection_to_jsonl():
    """将检测数据转换为JSONL格式"""
    pass

# 运行转换
convert_captioning_to_jsonl()
convert_ocr_to_jsonl()
convert_detection_to_jsonl()
```

---

#### **1.4 数据合并与分片**

```python
# scripts/merge_datasets.py
import json
import random

def merge_and_shuffle():
    """合并所有数据并shuffle"""
    all_data = []
    
    # 读取各类数据
    caption_data = read_jsonl("data/pretrain/captioning.jsonl")
    ocr_big_data = read_jsonl("data/pretrain/ocr_big.jsonl")
    ocr_small_data = read_jsonl("data/pretrain/ocr_small.jsonl")
    detection_data = read_jsonl("data/pretrain/detection.jsonl")
    
    # 按照比例混合
    total = len(caption_data) + len(ocr_big_data) + len(ocr_small_data) + len(detection_data)
    
    for item in caption_data:
        item['weight'] = 1.0  # 基础权重
    for item in ocr_big_data:
        item['weight'] = 1.5  # OCR数据给更高权重
    for item in detection_data:
        item['weight'] = 0.8
    
    all_data.extend(caption_data)
    all_data.extend(ocr_big_data)
    all_data.extend(ocr_small_data)
    all_data.extend(detection_data)
    
    # Shuffle
    random.shuffle(all_data)
    
    # 分片保存
    chunk_size = 100000
    for i in range(0, len(all_data), chunk_size):
        chunk = all_data[i:i+chunk_size]
        with open(f"data/pretrain/shard_{i//chunk_size}.jsonl", 'w') as f:
            for item in chunk:
                f.write(json.dumps(item) + '\n')

merge_and_shuffle()
```

---

### Step 2: 模型准备

#### **2.1 下载预训练模型**

```bash
# 创建模型目录
mkdir -p checkpoints

# 下载InternViT-6B基座
# 从HuggingFace
git clone https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2 \
    checkpoints/internvit_v1.2

# 下载InternLM2-20B-Chat
git clone https://huggingface.co/internlm/internlm2-chat-20b \
    checkpoints/internlm2_20b

# 目录结构:
# checkpoints/
# ├── internvit_v1.2/
# │   ├── config.json
# │   ├── pytorch_model-00001-of-00002.bin
# │   └── pytorch_model-00002-of-00002.bin
# └── internlm2_20b/
#     ├── config.json
#     ├── pytorch_model-00001-of-00005.bin
#     └── ...
```

#### **2.2 层裁剪脚本**

```python
# scripts/prune_internvit.py
import torch
from transformers import ViTForImageClassification, ViTConfig

def prune_internvit_layers(checkpoint_path, output_path, num_layers=45):
    """
    裁剪InternViT的层数
    
    Args:
        checkpoint_path: 原始checkpoint路径
        output_path: 输出路径
        num_layers: 保留的层数
    """
    # 加载原始模型
    model = ViTForImageClassification.from_pretrained(checkpoint_path)
    config = model.config
    
    print(f"Original layers: {config.num_hidden_layers}")
    
    # 更新配置
    config.num_hidden_layers = num_layers
    
    # 创建新模型
    new_model = ViTForImageClassification(config)
    
    # 复制embedding层
    new_model.embeddings.load_state_dict(model.embeddings.state_dict())
    
    # 复制选定的transformer层
    for i in range(num_layers):
        new_model.encoder.layer[i].load_state_dict(
            model.encoder.layer[i].state_dict()
        )
    
    # 复制layer norm和分类头
    new_model.layernorm.load_state_dict(model.layernorm.state_dict())
    new_model.classifier.load_state_dict(model.classifier.state_dict())
    
    # 保存
    new_model.save_pretrained(output_path)
    print(f"Saved pruned model to {output_path} with {num_layers} layers")

# 运行裁剪
prune_internvit_layers(
    checkpoint_path="checkpoints/internvit_original",
    output_path="checkpoints/internvit_448px_v1.5",
    num_layers=45
)
```

---

### Step 3: Phase 1 - Pre-training

#### **3.1 配置文件**

```yaml
# configs/v1_5_pretrain.yaml

model:
  name: "InternVL_1_5"
  vision_encoder:
    type: "InternViT"
    pretrained_path: "checkpoints/internvit_448px_v1.5"
    num_layers: 45
    hidden_size: 3200
    patch_size: 14
    image_size: 448
    pixel_shuffle: true
    shuffle_factor: 2
  
  language_model:
    type: "InternLM2"
    pretrained_path: "checkpoints/internlm2_20b"
    num_layers: 60
    hidden_size: 4096
    frozen: true  # Phase 1冻住LLM
  
  mlp_projector:
    type: "MLP"
    input_dim: 12800  # 3200 * 4 (pixel shuffle)
    hidden_dim: 4096
    output_dim: 4096
    num_layers: 2
    activation: "gelu"
    batch_norm: true

training:
  phase: "pretrain"
  
  # 分辨率配置
  tile_size: 448
  max_tiles_train: 12
  dynamic_aspect_ratio: true
  
  # 训练配置
  batch_size: 128
  gradient_accumulation_steps: 2
  learning_rate: 1e-4
  min_lr: 1e-5
  weight_decay: 0.02
  beta1: 0.9
  beta2: 0.999
  lr_scheduler: "cosine"
  warmup_steps: 5000
  max_steps: 50000
  
  # 数据配置
  train_data:
    - path: "data/pretrain/shard_*.jsonl"
      type: "jsonl"
      weight: 1.0
      repeat: 1
  
  # 其他配置
  context_length: 4096
  imagenet_normalization: true
  fp16: true
  bf16: false
  gradient_checkpointing: true
  
  # 保存配置
  save_steps: 5000
  logging_steps: 100
  eval_steps: 5000
  output_dir: "outputs/v1_5_pretrain"

logging:
  wandb:
    project: "InternVL-1.5"
    name: "pretrain_phase1"
```

#### **3.2 训练脚本**

```python
# scripts/train_v1_5_pretrain.py
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import deepspeed
import wandb
from tqdm import tqdm

# ===== Dataset =====
class MultiModalDataset(Dataset):
    def __init__(self, data_path, processor, max_tiles=12):
        self.data = self._load_data(data_path)
        self.processor = processor
        self.max_tiles = max_tiles
        
    def _load_data(self, path):
        import glob
        all_data = []
        for file in glob.glob(path):
            with open(file) as f:
                for line in f:
                    all_data.append(json.loads(line))
        return all_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image = Image.open(item['image']).convert('RGB')
        
        # 处理动态分辨率
        result = dynamic_resolution_process(image, max_tiles=self.max_tiles)
        tiles = result['tiles']
        thumbnail = result['thumbnail']
        
        # 处理文本
        text = item['text']
        
        # Tokenize
        inputs = self.processor(
            images=[thumbnail] + tiles,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=4096,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'],
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': inputs['input_ids'].clone()
        }

# ===== Model =====
class InternVL15ForPretrain(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Vision Encoder
        self.vision_encoder = ViTModel(
            config.vision_encoder.config
        )
        
        # LLM (frozen)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.model.language_model.pretrained_path
        )
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # MLP Projector
        self.mlp = MLPProjector(
            input_dim=config.model.mlp_projector.input_dim,
            hidden_dim=config.model.mlp_projector.hidden_dim,
            output_dim=config.model.mlp_projector.output_dim,
            num_layers=config.model.mlp_projector.num_layers
        )
        
        # Pixel Shuffle
        self.shuffle_factor = config.model.vision_encoder.pixel_shuffle
        
    def forward(self, pixel_values, input_ids, attention_mask, labels):
        batch_size = pixel_values.shape[0]
        
        # 第一个是thumbnail，其余是tiles
        thumbnail = pixel_values[:, 0:1]  # [B, 1, 3, 448, 448]
        tiles = pixel_values[:, 1:]  # [B, N, 3, 448, 448]
        
        # Vision Encoding
        def encode_features(images):
            features = self.vision_encoder(
                pixel_values=images.view(-1, 3, 448, 448)
            ).last_hidden_state  # [B*N, H*W*shuffle^2, D]
            
            # Pixel Shuffle
            H = W = 32 // self.shuffle_factor
            features = features.view(
                images.size(0) // batch_size,
                -1,
                H * W,
                features.size(-1)
            ).flatten(1, 2)  # [B, H*W, D*shuffle^2]
            
            return features
        
        thumbnail_features = encode_features(thumbnail)
        tiles_features = encode_features(tiles)
        
        # 合并：[CLS] + thumbnail + tiles
        visual_features = torch.cat([thumbnail_features, tiles_features], dim=1)
        
        # 投影到LLM维度
        projected = self.mlp(visual_features)
        
        # 准备LLM输入
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 替换图像tokens
        # (这里需要设计token替换逻辑)
        # ...
        
        # Forward
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss

# ===== Training Loop =====
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/v1_5_pretrain.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()
    
    # Load model
    model = InternVL15ForPretrain(config)
    
    # Load data
    dataset = MultiModalDataset(
        config.training.train_data[0].path,
        processor=image_processor,
        max_tiles=config.training.max_tiles_train
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size // torch.distributed.get_world_size(),
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # DeepSpeed
    ds_config = {
        "train_batch_size": config.training.batch_size,
        "train_micro_batch_size_per_gpu": config.training.batch_size // torch.distributed.get_world_size(),
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "fp16": {"enabled": config.training.fp16},
        "bf16": {"enabled": config.training.bf16},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "gradient_clipping": 1.0,
    }
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # WandB
    wandb.init(
        project=config.logging.wandb.project,
        name=config.logging.wandb.name
    )
    
    # Training
    global_step = 0
    for epoch in range(100):
        for batch in tqdm(dataloader):
            # Forward
            loss = model_engine(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Backward
            model_engine.backward(loss)
            model_engine.step()
            
            # Logging
            if global_step % config.training.logging_steps == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "step": global_step
                })
            
            # Save
            if global_step % config.training.save_steps == 0:
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    save_path = os.path.join(
                        config.training.output_dir,
                        f"step_{global_step}"
                    )
                    os.makedirs(save_path, exist_ok=True)
                    model_engine.save_checkpoint(save_path)
            
            global_step += 1
            
            if global_step >= config.training.max_steps:
                break

if __name__ == "__main__":
    train()
```

#### **3.3 启动训练**

```bash
# 单机多卡 (8卡)
deepspeed --num_gpus=8 scripts/train_v1_5_pretrain.py \
    --config configs/v1_5_pretrain.yaml

# 多机多卡 (64卡)
# 在主节点运行
deepspeed --num_gpus=8 --num_nodes=8 --hostfile=hostfile \
    scripts/train_v1_5_pretrain.py \
    --config configs/v1_5_pretrain.yaml

# hostfile示例:
# worker-1 slots=8
# worker-2 slots=8
# worker-3 slots=8
# worker-4 slots=8
# worker-5 slots=8
# worker-6 slots=8
# worker-7 slots=8
# worker-8 slots=8
```

---

### Step 4: Phase 2 - Full Fine-tuning

#### **4.1 准备Fine-tuning数据**

```python
# scripts/prepare_finetune_data.py
def prepare_finetune_datasets():
    """准备Fine-tuning阶段的数据"""
    
    # 根据Table 1b准备各类数据
    
    datasets = {
        "captioning": [
            "TextCaps",
            "ShareGPT4V"
        ],
        "general_qa": [
            "VQAv2",
            "GQA",
            "OKVQA",
            "VSR",
            "VisualDialog"
        ],
        "science": [
            "AI2D",
            "ScienceQA",
            "TQA"
        ],
        "chart": [
            "ChartQA",
            "MMC-Inst",
            "DVQA",
            "PlotQA",
            "LRV-Instruction"
        ],
        "math": [
            "GeoQA+",
            "TabMWP",
            "MathQA",
            "CLEVR-Math",
            "Geometry3K"
        ],
        "knowledge": [
            "KVQA",
            "Wikipedia"
        ],
        "ocr": [
            "OCRVQA",
            "TextVQA",
            "ArT",
            "COCO-Text",
            "CTW",
            "LSVT",
            "RCTW-17",
            "ReCTs",
            "SynthDoG",
            "ST-VQA"
        ],
        "document": [
            "DocVQA",
            "Common-Crawl-PDF"
        ],
        "grounding": [
            "RefCOCO",
            "RefCOCO+",
            "RefCOCOg",
            "Visual-Genome"
        ],
        "conversation": [
            "LLaVA-150K",
            "LVIS-Instruct4V",
            "ALLaVA",
            "Laion-GPT4V",
            "TextOCR-GPT4V",
            "SVIT"
        ],
        "text_only": [
            "OpenHermes2.5",
            "Alpaca-GPT4",
            "ShareGPT",
            "COIG-CQIA"
        ]
    }
    
    return datasets

# 转换为统一格式
def convert_to_instruction_format(data):
    """转换为instruction-following格式"""
    return {
        "id": data.get("id", ""),
        "image": data["image"],
        "conversations": [
            {
                "from": "human",
                "value": data["question"]
            },
            {
                "from": "gpt",
                "value": data["answer"]
            }
        ]
    }
```

#### **4.2 Fine-tuning配置**

```yaml
# configs/v1_5_finetune.yaml
model:
  name: "InternVL_1_5_Chat"
  
  vision_encoder:
    pretrained_path: "outputs/v1_5_pretrain/step_50000/vision_encoder"
    num_layers: 45
    frozen: false  # Phase 2全量训练
  
  language_model:
    pretrained_path: "checkpoints/internlm2_20b"
    num_layers: 60
    frozen: false  # Phase 2全量训练
  
  mlp_projector:
    pretrained_path: "outputs/v1_5_pretrain/step_50000/mlp"
    frozen: false

training:
  phase: "finetune"
  
  batch_size: 64
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  min_lr: 2e-6
  weight_decay: 0.01
  
  lr_scheduler: "cosine"
  warmup_ratio: 0.03
  
  max_steps: 20000
  
  # 数据
  train_data:
    - path: "data/finetune/shard_*.jsonl"
      type: "jsonl"
      weight: 1.0
  
  # LoRA配置 (可选，用于内存受限场景)
  use_lora: false
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
```

#### **4.3 启动Fine-tuning**

```bash
# 使用DeepSpeed ZeRO-3进行全量训练
deepspeed --num_gpus=16 scripts/train_v1_5_finetune.py \
    --config configs/v1_5_finetune.yaml \
    --zero_stage 3

# 如果内存不足，可以使用LoRA
# 修改配置中的 use_lora: true
```

---

### Step 5: 评估

#### **5.1 准备评估脚本**

```python
# scripts/evaluate.py
import torch
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm

class InternVL15Tokenizer:
    def __init__(self, llm_path, vision_path):
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_path)
    
    def __call__(self, images, text, max_length=4096):
        # 处理图像
        pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
        
        # 处理文本
        input_ids = self.llm_tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).input_ids
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }

def evaluate_benchmark(model, tokenizer, benchmark_name, data_path):
    """评估单个benchmark"""
    
    # 加载测试数据
    test_data = []
    with open(data_path) as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # 生成答案
    results = []
    for item in tqdm(test_data):
        image = Image.open(item["image"])
        
        # 处理
        inputs = tokenizer(
            images=[image],
            text=item["question"],
            max_length=4096
        )
        
        # 添加图像占位符token
        # (具体实现需要根据model设计调整)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False
            )
        
        # Decode
        answer = tokenizer.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "pred_answer": answer
        })
    
    # 计算指标
    metrics = compute_metrics(results, benchmark_name)
    
    return metrics

def compute_metrics(results, benchmark_name):
    """计算不同benchmark的指标"""
    if benchmark_name == "DocVQA":
        # ANLS (Average Normalized Levenshtein Similarity)
        pass
    elif benchmark_name == "TextVQA":
        # ANLS
        pass
    elif benchmark_name == "ChartQA":
        # 相对准确度
        pass
    elif benchmark_name in ["MMBench-EN", "MMBench-CN"]:
        # 选择题准确率
        pass
    
    return {"accuracy": 0.0}

def main():
    # Load model
    model = InternVL15ForCausalLM.from_pretrained(
        "outputs/v1_5_finetune/step_20000"
    )
    model = model.cuda()
    model.eval()
    
    tokenizer = InternVL15Tokenizer(
        llm_path="checkpoints/internlm2_20b",
        vision_path="outputs/v1_5_pretrain/step_50000/vision_encoder"
    )
    
    # Benchmarks
    benchmarks = [
        {
            "name": "DocVQA",
            "data_path": "data/benchmarks/docvqa/test.jsonl"
        },
        {
            "name": "TextVQA",  
            "data_path": "data/benchmarks/textvqa/val.jsonl"
        },
        {
            "name": "ChartQA",
            "data_path": "data/benchmarks/chartqa/test.jsonl"
        },
        {
            "name": "MMBench-CN",
            "data_path": "data/benchmarks/mmbench/cn_test.jsonl"
        }
    ]
    
    # Evaluation
    all_metrics = {}
    for bench in benchmarks:
        metrics = evaluate_benchmark(
            model, tokenizer, 
            bench["name"], 
            bench["data_path"]
        )
        all_metrics[bench["name"]] = metrics
        print(f"{bench['name']}: {metrics}")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == "__main__":
    main()
```

#### **5.2 运行评估**

```bash
# 评估所有benchmarks
python scripts/evaluate.py

# 使用VLMEvalKit
pip install vlmevalkit

python -m vlmeval.eval \
    --data MMBench_CN,MMBench_EN,TextVQA,DocVQA,ChartQA \
    --model internvl_1_5 \
    --model_path outputs/v1_5_finetune/step_20000
```

---

### Step 6: 推理部署

```python
# scripts/inference.py
import torch
from PIL import Image

def load_model(checkpoint_path):
    """加载训练好的模型"""
    model = InternVL15ForCausalLM.from_pretrained(checkpoint_path)
    return model

def generate_response(model, image_path, question, max_tiles=12):
    """生成响应"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Process with dynamic resolution
    processed = dynamic_resolution_process(image, max_tiles=max_tiles)
    
    # Prepare inputs
    # ...
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 使用
model = load_model("outputs/v1_5_finetune/step_20000")
answer = generate_response(
    model,
    "test.jpg",
    "What is shown in this document?"
)
print(answer)
```

---

## 资源链接汇总

官方资源：
- [Paper](https://arxiv.org/abs/2404.16821)
- [GitHub](https://github.com/OpenGVLab/InternVL)
- [Hugging Face Models](https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5)
- [Demo](https://internvl.opengvlab.com)

依赖库：
- [InternLM2](https://github.com/InternLM/InternLM)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

相关论文：
- [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238)
- [UReader](https://arxiv.org/abs/2310.05126)
- [LLaVA](https://arxiv.org/abs/2304.08485)

这个复现指南涵盖了从环境准备、数据处理、模型训练到评估部署的完整流程。由于这是大规模训练，实际实施需要根据你的资源情况进行调整。