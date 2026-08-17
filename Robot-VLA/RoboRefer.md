---
source_pdf: RoboRefer.pdf
paper_sha256: 43871d961a3d088a85fd660f44f02f25a848f55616a63a913ef9ff25e61c8ac6
processed_at: '2026-08-12T01:24:41-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboRefer 人话版

## 一句话概括

教 VLM 学会"把杯子放在盘子和酱油碟之间，跟杯子logo对齐"这种复杂空间指令，然后用这个能力去控制机器人。

## 为什么这件事难？

想象你跟一个聪明但不识路的朋友说："帮我把酱油碟放到离你最近的盘子和那个酱油瓶中间的空地上"。你的朋友会怎么想？

1. 先看一眼桌子，找到**离你最近的盘子**是谁
2. 再找到**那个酱油瓶**
3. 然后想象这两个东西中间的**空白区域**
4. 最后在那个空白区域里**挑一个具体位置**

这四步是串行的，每一步都依赖上一步的结果。如果第一步找错了盘子，后面全错。

现有 VLM 的问题就在这：它们能做单步空间感知（"哪个杯子在左边"），但**不会做多步空间推理**。你给个复合指令，它要么瞎猜，要么只处理第一个约束就停了。

而且还有个更深的问题：**3D 空间感知本身就不行**。GPT-4o、Gemini 这些模型在 2D 图像上很强，但一问"哪个物体离相机更远"就开始蒙。原因很简单——它们从没真正"看"过 depth。

## RoboRefer 怎么解决的？三个 trick

### Trick 1: 给 depth 配个专属 encoder

以前的做法（SpatialRGPT, SpatialBot）很偷懒，把 depth map 当成普通灰度图，塞给 image encoder 一起处理。这就像让一个只会看彩色照片的人突然去看地形等高线图——他能看到东西，但脑子里的 feature 完全乱了。

后果是 image encoder 被 depth 污染，RGB 推理能力退化。要弥补就得拿 2 倍以上的 RGB-only data 重新 co-train。

RoboRefer 的做法很朴素：**给 depth 单独配一个 encoder**。架构上跟 image encoder 一模一样（都是 SigLIP-so400m-patch14-448），权重也从 image encoder 初始化，但训练时独立更新。两个 encoder 的输出通过各自的 projector 映射到 LLM 的 token space，然后 concat 起来喂给 LLM。

$$\mathbf{h}_{img} = P_{img}(E_{img}(\mathbf{I}_{rgb}))$$
$$\mathbf{h}_{depth} = P_{depth}(E_{depth}(\mathbf{I}_{depth}))$$
$$\mathbf{h}_{fused} = [\mathbf{h}_{img}; \mathbf{h}_{depth}]$$

变量解释：
- $\mathbf{I}_{rgb}, \mathbf{I}_{depth}$：RGB 图和 depth map 输入
- $E_{img}, E_{depth}$：两个独立 encoder（结构相同，权重独立更新）
- $P_{img}, P_{depth}$：两个 linear projector
- $[\cdot;\cdot]$：concat 操作

好处是什么？image encoder 的 general VQA 能力完全保留（Table 4 显示 MMBench、MME 这些通用 benchmark 基本不掉），depth encoder 专注学空间几何。两者井水不犯河水。

参考: NVILA ([arxiv:2412.04468](https://arxiv.org/abs/2412.04468)) 作为 base model，这个 dedicated encoder 的设计可以参考 SpatialVLM ([arxiv:2401.12166](https://arxiv.org/abs/2401.12166)) 的讨论。

### Trick 2: SFT 两步走，先对齐再增强

SFT 分两步，逻辑很清晰：

**第一步：Depth Alignment**
只训练 depth projector $P_{depth}$，其他参数 freeze。目的让 depth embedding 跟 text space 对齐。用 RefSpatial 的 2.5M RGB-D 样本，学习率 1e-4，1 个 epoch。

**第二步：Spatial Understanding Enhancement**
全参数 fine-tune。关键 trick 是：**RefSpatial 同时以 RGB 和 RGB-D 两种形式喂入**。这逼着 image encoder 自己从 RGB 里学空间线索，而不是偷懒去依赖 depth encoder。

Table 16 的 ablation 很能说明问题：

| 训练数据 | 推理输入 | CV-Bench 2D-Relation |
|---------|---------|---------------------|
| 只用 RGB-D | RGB-only | 87.69 |
| RGB + RGB-D 混用 | RGB-only | **96.15** |
| RGB + RGB-D 混用 | RGB-D | **96.31** |

混用训练让 RGB-only 推理从 87.69 跳到 96.15，几乎追平 RGB-D 推理。这说明 image encoder 真的从 RGB 里学到了空间先验，不光靠 depth。

### Trick 3: RFT with Metric-Sensitive Process Reward（最有意思的部分）

SFT 的问题：**会 memorize 而不 generalize**。训练集里见过的空间关系组合能答，没见过的就歇菜。Table 2 的 Unseen 子集（77 个 novel combination）显示 2B-SFT 只有 33.77%。

RFT 的思路是：让模型自己探索，用 reward 信号引导它往正确方向走。RoboRefer 用 GRPO（Group Relative Policy Optimization，来自 DeepSeekMath [arxiv:2402.03300](https://arxiv.org/abs/2402.03300)），对每个问题采样 N 个回答，根据 reward 计算 group-relative advantage。

**Reward 设计是这 paper 最有创意的地方。**

传统 process reward 需要一个 Process Reward Model（PRM）来判断中间步骤对不对。但 PRM 有两个问题：
1. LLM-based PRM 看不了图，没法验证坐标对不对
2. VLM-based PRM 对 text-formatted coordinate 理解很差（参考 [arxiv:2412.01981](https://arxiv.org/abs/2412.01981) 的发现）

RoboRefer 的解法：**用 rule-based reward 直接评估中间感知结果**，绕开 PRM。关键是两个设计：

**Metric-sensitivity**：不同空间属性用不同度量
- Position：L1 距离 < 50 像素 → reward = 1
- Orientation：cosine similarity > 0.8 → reward = 1
- Size：误差 < ±15% → reward = 1

**Order-invariance**：推理过程不必严格顺序。"先找键盘还是先找鼠标"不影响最终理解"键盘和鼠标中间的空地"。这跟传统 PRM 强调 sequential reasoning 很不一样。

完整 reward function：

$$r_i = R_{OF}(a_i) + R_P(a_i) + \alpha R_{PF}(a_i) + \alpha R_{Acc}(a_i)$$

其中 $\alpha = 0.25$，各项含义：
- $R_{OF}$：Outcome Format Reward，输出格式对不对（`<answer>...</answer>`）
- $R_P$：Point L1 Reward，最终预测点离 ground truth 是否 < 50 像素
- $R_{PF}$：Process Format Reward，中间步骤格式对不对（`[Perception Type] [Target Object]: [Value]`）
- $R_{Acc}$：Accuracy Reward，中间感知结果对不对

$\alpha = 0.25$ 是为了防止 process reward 累积过多，压过 outcome reward。

Group-relative advantage 计算：

$$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$

这标准化让 reward 偏高的 response 获得正 advantage，偏低的获得负 advantage，然后更新 policy 强化高 advantage 的 response。

## RefSpatial 数据集：20M QA pair 怎么来的？

这是 paper 的另一个支柱。数据来源三层，层层递进：

| 数据源 | 来源 | 样本数 | 作用 |
|--------|------|--------|------|
| 2D Web Image | OpenImages (466k filtered from 1.7M) | ~466k images | 学基础空间概念 + 室内外 depth 感知 |
| 3D Embodied Video | CA-1M (100k frames from 2M) | ~100k frames | 学室内精细 3D 空间关系 |
| Simulation | Infinigen + Objaverse (3k assets from 46k) | 模拟生成 | 学多步推理过程 |

**2D 数据的 trick**：2D 图像没有真 3D 信息，怎么办？构造 pseudo-3D scene graph。流程是：
1. RAM ([arxiv:2406.09214](https://arxiv.org/abs/2406.09214)) 识别 object label
2. GroundingDINO ([arxiv:2303.05499](https://arxiv.org/abs/2303.05499)) 定位 bbox
3. UniDepth V2 ([arxiv:2503.18990](https://arxiv.org/abs/2503.18990)) 估计 metric depth
4. SAM 2.1 ([arxiv:2408.00714](https://arxiv.org/abs/2408.00714)) 生成 instance mask
5. 组装成 scene graph，节点是 object，边是 spatial relation

**3D 数据的 trick**：CA-1M 有 per-frame 3D oriented bbox，但缺 semantic label。用 GroundingDINO + RAM 重新标注，然后双向 IoU 匹配过滤掉原始的 noisy annotation。

**Simulation 数据的 trick**：用 Infinigen ([arxiv:2406.11827](https://arxiv.org/abs/2406.11827)) 程序化生成室内场景，放入手动筛选的 Objaverse ([arxiv:2307.05663](https://arxiv.org/abs/2307.05663)) 资产。关键是每生成一个任务，对应的 reasoning process 也被程序化记录下来——每行代码翻译成文本，中间结果填入结构化格式。这就自动得到了带 reasoning annotation 的 QA pair。

涵盖 31 种空间关系，远超之前数据集的 15 种。举几个例子：left/right, above/below (world frame), above/below (image frame), front/behind, between, close/far, inside/outside, touching, facing/back, tall/short, big/small, wide/thin, rotation angle, distance metric, free space...

## 实验结果：到底有多强？

### 单步空间理解（Table 1）

| Model | CV-Bench avg | BLINK val | RoboSpatial | SAT | EmbSpatial |
|-------|-------------|-----------|-------------|-----|------------|
| Gemini-2.5-Pro | 91.67 | 89.76 | 77.24 | 70.59 | 76.67 |
| GPT-4o | 84.82 | 80.38 | 77.20 | 68.67 | 63.38 |
| SpatialRGPT-8B | 89.77 | 85.32 | 66.67 | 64.00 | 59.62 |
| **RoboRefer-8B-SFT** | **96.90** | **92.18** | **84.55** | **86.67** | **72.53** |

平均 89.6% success rate，比 Gemini-2.5-Pro 高 5 个点。

### 多步空间推理（Table 2，RefSpatial-Bench）

| Model | Location | Placement | Unseen | Avg |
|-------|----------|-----------|--------|-----|
| Gemini-2.5-Pro | 46.96 | 24.21 | 27.14 | 32.77 |
| RoboPoint-13B | 22.87 | 9.27 | 8.40 | 13.51 |
| Molmo-72B | 45.77 | 14.74 | 21.24 | 27.25 |
| 2B-SFT | 47.00 | 48.00 | 33.77 | 42.92 |
| **2B-RFT** | **52.00** | **54.00** | **41.56** | **49.19** |

2B-RFT 比 Gemini-2.5-Pro 平均高 17.4%。Unseen 子集上 2B-RFT 比 2B-SFT 高 9.1%，证明 RFT 确实带来 generalization。

### 按推理步数拆分（Table 8）

| Benchmark | Step | 2B-SFT | 2B-RFT | Gain |
|-----------|------|--------|--------|------|
| Location | 1 | 63.33 | 66.67 | +3.34 |
| Location | 2 | 39.58 | 43.75 | +4.17 |
| Location | 3 | 27.27 | 36.36 | +9.09 |
| Placement | 4 | 41.67 | 45.83 | +4.16 |
| Placement | 5 | 0.00 | 25.00 | +25.00 |

步数越多，RFT 的 gain 越大。Step 5 上 SFT 直接 0%，RFT 能到 25%。这说明 RFT 学到了真正的 reasoning ability，而不只是 memorize。

### 真实机器人实验（Table 6）

任务："Pick the hamburger closest to the mug nearest the camera and place it in front of the teddy bear."

| Method | Success Rate |
|--------|-------------|
| OpenVLA | 0.00% |
| RoboPoint | 80.00% |
| **RoboRefer** | **80.00%** |

但关键在 dynamic scene：当 mug 被移动后，只有 RoboRefer 能实时更新 target（2.5Hz 重规划），其他方法直接失败。RoboRefer 以 2.5Hz 频率重新预测 target point，所以物体移动后能自动适应。

另一个任务更狠："Pick the apple in front of the leftmost cup's logo side, navigate to the nearest table, and place it aligned with the apple row." 这需要 navigation + manipulation 联合。只有 RoboRefer 能做到 60% success rate，因为 point-based formulation 统一了 navigation waypoint 和 manipulation target。

## 我的几点观察

**1. Depth encoder 该不该单独？**
这篇 paper 的 ablation 很有说服力。但我觉得更深层的问题是：为什么 VLM 需要 explicit depth？人类看一张 2D 照片就能推断深度啊。答案可能是：当前 VLM 的训练数据里，spatial reasoning 的 signal 太弱，explicit depth 是个 shortcut。未来如果 VLM 能从海量视频里学到 strong 3D prior，可能就不需要 explicit depth 了。参考 MM-Spatial ([arxiv:2505.11311](https://arxiv.org/abs/2505.11311)) 的讨论。

**2. Process reward 的 order-invariance 很聪明**
传统 CoT reasoning 强调 sequential，但空间推理天生可以并行——你不用先确定键盘位置再确定鼠标位置才能理解"两者之间"。RoboRefer 用 metric-sensitive + order-invariant 的设计，让模型自由选择推理顺序，只要中间感知结果对就行。这个 insight 可以推广到其他需要 multi-step perception 的任务。

**3. 2B 模型 + RFT 打爆 72B 模型**
Gemini-2.5-Pro 是万亿参数级别的模型，在 spatial referring 上被一个 2B 模型超 17 个点。这再次印证了 Chu et al. 的发现：SFT memorizes, RL generalizes。小模型 + 针对性 RL > 大模型 + 通用 SFT。这个趋势在 reasoning task 上越来越明显，参考 DeepSeek-R1 ([arxiv:2501.12948](https://arxiv.org/abs/2501.12948))。

**4. Point vs Bbox 的哲学选择**
Point-based formulation 更适合 robotics，因为 point 可以直接 lift 到 3D via depth。Bbox 需要额外处理才能得到 grasping target。但 point 也有局限——无法表达 spatial extent，比如"把杯子放到这张纸上"需要知道纸的大小。Future work 可能需要 hybrid representation。

**5. Limitation: 人话理解的缺失**
Paper 里提到的 limitation 很诚实（Appx. G）。人类会说"拿那个朝向饮料的寿司盘"，这种**概率性偏好**和**空间兼容性**推理需要 human intent understanding。当前 RefSpatial 是程序化生成的，缺少这种 intent-aware 数据。未来可能需要从人类对话数据中学习。

## 资源链接

- Project page: [https://zhoues.github.io/RoboRefer/](https://zhoues.github.io/RoboRefer/)
- Base model NVILA: [arxiv:2412.04468](https://arxiv.org/abs/2412.04468)
- GRPO method: [arxiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Depth estimation: [DepthAnything V2](https://arxiv.org/abs/2406.09414)
- 相关工作 RoboPoint: [arxiv:2406.10721](https://arxiv.org/abs/2406.10721)
- 相关工作 SpatialRGPT: [arxiv:2406.09116](https://arxiv.org/abs/2406.09116)
- 相关工作 SoFar: [arxiv:2501.17033](https://arxiv.org/abs/2501.17033)
- SFT memorizes, RL generalizes: [arxiv:2505.20817](https://arxiv.org/abs/2505.20817)

---

总结一句：RoboRefer 的核心贡献是把 spatial referring 从"单步感知"推进到"多步推理"，用 dedicated depth encoder 解决 modality interference，用 metric-sensitive process reward 解决 reasoning generalization，用 20M 数据集填平训练数据鸿沟。2B 模型打爆万亿参数 Gemini 这件事，值得认真思考。

---

# RoboRefer: 3D-aware VLM for Multi-Step Spatial Referring with Reasoning

## 1. Problem Formulation 的核心 intuition

RoboRefer 处理的核心问题可以形式化为一个 mapping:

$$f: (\mathcal{O}, \mathcal{L}) \rightarrow (x, y)$$

其中 $\mathcal{O}$ 是 sensor observation (RGB 或 RGB-D), $\mathcal{L}$ 是 textual instruction encoding spatial constraints, $(x, y) \in [0,1]^2$ 是 image space 中的归一化 2D point。这个 point 可以通过 depth 和 camera intrinsics 自然地 lift 到 3D coordinates，作为 manipulation target, placement location, 或 navigation waypoint。

为什么选 point 而非 bbox？Bbox 在 occlusion scenarios 下常常包含 irrelevant objects，而 point 能精确 localize visible object part。更重要的是，point-based formulation 统一了 navigation, grasping, placement 这三种 task，使得 VLM 可以 multi-task learning。这与 RoboPoint ([arxiv:2406.10721](https://arxiv.org/abs/2406.10721)) 的设计哲学一致。

## 2. Architecture: Dedicated Depth Encoder 的 design intuition

### 现有方案的痛点
SpatialRGPT ([arxiv:2406.09116](https://arxiv.org/abs/2406.09116)) 和 SpatialBot 这类方法把 depth 当 RGB-like input, 共享 image encoder。这带来两个问题:
1. **Modality interference**: depth 的 statistical distribution 与 RGB 截然不同 (depth 是 smooth, piecewise continuous, 而 RGB 是 high-frequency texture)，share encoder 会破坏 pretrained image features
2. **需要 expensive co-training**: 为了 compensate 退化，需要 2x 以上的 RGB-only data

### RoboRefer 的解法
采用 disentangled but dedicated architecture:
- **Image encoder**: $E_{img}$ (SigLIP-so400m-patch14-448, dynamic resolution)
- **Depth encoder**: $E_{depth}$ (structurally mirroring $E_{img}$, initialized from its weights)
- **Separate projectors**: $P_{img}, P_{depth}$ (linear connectors)
- **Shared LLM**: Qwen2 backbone from NVILA ([arxiv:2412.04468](https://arxiv.org/abs/2412.04468))

Forward pass:
$$\mathbf{h}_{img} = P_{img}(E_{img}(\mathbf{I}_{rgb})), \quad \mathbf{h}_{depth} = P_{depth}(E_{depth}(\mathbf{I}_{depth}))$$
$$\mathbf{h}_{fused} = \text{Concat}(\mathbf{h}_{img}, \mathbf{h}_{depth})$$
$$\mathbf{y} = \text{LLM}(\mathbf{h}_{fused}, \text{tokenize}(\mathcal{L}))$$

**关键 design insight**: image encoder 在 RGB-D training 时 freeze update path for image branch (or 极小 learning rate), depth encoder 独立 update。这样 image encoder 的 general VQA capability 完全保留, 不需要 extensive RGB-only co-training。

Table 4 的 ablation 验证: dedicated encoder 在 general VQA benchmarks (MMBench, MME 等) 上保持 comparable 性能, 而 shared encoder 明显下降。

## 3. SFT 两步走: Cold Start Strategy

### Step 1: Depth Alignment
仅训练 depth projector $P_{depth}$:
$$\mathcal{L}_{DA} = -\mathbb{E}_{(\mathcal{O}, \mathcal{Q}, A) \sim \mathcal{D}_{RGBD}} \sum_{t=1}^{T} \log \pi_\theta(y_t | \mathcal{O}, \mathcal{Q}, y_{<t})$$

其中 $\pi_\theta$ 是 token distribution, $y_t$ 是第 $t$ 个 token, $y_{<t}$ 是 prefix tokens。这一步让 depth embedding space 与 textual space 对齐, hyperparameters: lr=1e-4, warmup=0.03, batch=7/GPU (2B), 1 epoch。

### Step 2: Spatial Understanding Enhancement
Full-parameter fine-tuning, 同时使用 RefSpatial (RGB) 和 RefSpatial (RGB-D), 加上 auxiliary data:
- 965k instruction-tuned (LLaVA-1.5, LRV)
- 321k RefCOCO/+/g
- 176k SAT, 127k EmbSpatial

**关键 trick**: RefSpatial 同时以 RGB 和 RGB-D 两种形式喂入, 强制 image encoder $E_{img}$ 学习 spatial cues from RGB alone (不只依赖 depth)。这一点非常关键 - Table 16 ablation 显示, 如果只用 RGB-D 训练, image encoder 会 over-rely depth, 单独 RGB inference 时性能从 96.15 掉到 87.69 (2D-Relation)。

Total SFT data post-slicing: 8.5M samples, lr=5e-5, batch=6/GPU (2B), 1 epoch。

## 4. RFT: GRPO with Metric-Sensitive Process Reward

### 为什么需要 RFT 而非纯 SFT？
SFT 倾向于 memorize training distribution, 限制了 generalization。Table 2 的 Unseen 子集 (77 samples with novel spatial relation combinations) 显示: 2B-SFT 仅 33.77%, 而 2B-RFT 达 41.56%, 提升 9.1%。这与 Chu et al. 的发现 ([arxiv:2505.20817](https://arxiv.org/abs/2505.20817)) 一致: "SFT memorizes, RL generalizes"。

### GRPO Sampling
对每个 input state $s = (\mathcal{O}, \mathcal{Q})$, 从 current policy $\pi_\theta$ (initialized from $\pi_{SFT}$) 采样 N 个 responses:
$$a_i \sim \pi_\theta(a | \mathcal{O}, \mathcal{Q}), \quad i = 1, 2, \ldots, N$$

### Reward 设计 - 这是 paper 的核心创新

**Outcome Format Reward** $R_{OF}$: 强制输出格式 `
