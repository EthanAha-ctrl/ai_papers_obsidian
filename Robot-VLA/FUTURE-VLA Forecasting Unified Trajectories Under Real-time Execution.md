---
source_pdf: FUTURE-VLA Forecasting Unified Trajectories Under Real-time Execution.pdf
paper_sha256: 1b6be837a6cd597b7bffc6ef55ceb0f358560fa5bf0d9caf7d279f9ea860d640
processed_at: '2026-08-19T08:33:05-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 FUTURE-VLA

## 核心问题：机器人为什么傻？

你想想现在 robot 的痛点是什么。机器人看一个画面，立刻输出一组 action，执行完再看下一个画面，再输出。这个叫 reactive policy，听起来 fine，但实际上有个大麻烦。

打个比方，你在开车，如果只能看当前这一瞬间的路况，不能回顾刚才那辆车是不是变道了，不能预判前面路口会不会有行人冲出来，你肯定开不好。你需要两样东西：**memory**（刚才发生了什么）和 **foresight**（接下来会发生什么）。

Robot policy 一样。只看当前 frame 的模型有严重的 **perceptual myopia**（感知短视），尤其在 long-horizon 任务里，比如"收拾桌子"需要连续做几十步决策，每一步只看当前画面肯定不够。

但问题来了：你想给它 memory，就得多塞历史帧；多塞帧，inference 就慢；inference 慢，robot control 的闭环频率就掉，动作就抖。你想给它 foresight，传统 World Model 要生成未来几帧的 pixel-level video，这个计算量爆炸，根本跟不上 action 输出的节奏。

所以大家一直卡在一个 trade-off 里：要么 context 短反应快，要么 context 长反应慢。

---

## FUTURE-VLA 的 trick：两边同时压

这篇文章的核心 idea 就是 **dual-sided efficiency**，input 和 output 两边都做压缩，让 16 帧历史的 token 预算跟单帧 baseline 一样，同时 output 端用 latent space 预测而不是 pixel rollout。

### Input 端：聪明的压缩

你给它 16 帧历史，如果每帧都高分辨率 encode，token 数量爆炸。作者观察到一个很自然的 insight：**最近几帧需要看清细节，远端几帧只需要知道大概发生了什么**。

比如你抓一个 cup，最近 2 帧决定了你 gripper 的精确位置，需要 pixel-level detail；8 帧前那帧只需要告诉你"cup 大概从桌子左边移到了中间"，coarse representation 足够。

所以他们做了一个 temporal pyramid：
- 最近 2 帧：不压缩，64 tokens/frame
- 中间 6 帧：压 1 次，16 tokens/frame
- 最早 8 帧：压 2 次，4 tokens/frame

加起来 256 tokens/view，跟单帧 baseline 一样。然后 Qwen3-VL 自带的 patch merger 再做一次统一 4× 压缩，整个 input 就塞进去了。

这个设计的 inductive bias 就是 information density 在时间维度上不均匀。这不是什么 fancy 数学，就是个工程 trick，但 ablation 证明它 work 得很好。

### Output 端：latent space 而非 pixel space

传统 World Model 生成未来 video，比如生成 256×256 的下一帧，需要预测成千上万个 pixel token。这在 autoregressive generation 下慢得要命。

作者借用了 TiTok 的思路：把每张 image 压成固定 32 个 discrete token。32 个 token！这压缩比夸张到 Figure 7 里的 reconstruction 看起来几乎跟原图无差别。原理是 natural image 在 patch 级别有 massive redundancy，32 个 latent code 足以 capture structural + texture 信息。

这样未来 16 帧的预测就只需要生成 $V \times 16 \times 32$ 个 token（V 是 view 数），在 single forward pass 内就能跟 action chunk 一起 autoregressively 输出。

---

## 关键 trick：unified tokenization

这里有个很巧妙的工程细节。Action 和 visual latent 都要映射成 discrete token 喂给 LLM backbone。一般做法是 expand vocabulary，但这会破坏预训练 embedding matrix 的语义结构。

作者用了个 remapping 策略：把 action vocab (2048) 和 visual vocab (4096) 映射到 Qwen3 tokenizer 的尾部 indices。因为 Qwen3 用 BPE，尾部 tokens 对应自然语言里极罕见的 byte sequence，复用这些 slot 对 linguistic knowledge 干扰最小，还能复用已训练的 embedding matrix。

Action 端用的是 FAST，基于 DCT 的 spectral tokenization。Action trajectory 在时域上往往 smooth + 局部 high-frequency，DCT 把能量集中到少数 frequency coefficient，压缩率远高于时域 discretization。

---

## HIL：让 World Model 真正有用

这部分是 paper 最有意思的 contribution。之前 World Model 在 robot 里大多是 training signal 或者 offline evaluator，跟 policy 是 loosely coupled。FUTURE-VLA 因为能实时生成未来 preview，所以可以做一个 **Human-In-the-Loop** 机制：

模型输出 action chunk + 未来 16 帧的 visual preview → 人类 verifier 看一眼 preview → 决定执行多少步（k ∈ [1, 16]）→ 如果 preview 看起来要失败，直接 reject（k=0）并提高 temperature 重新采样。

这本质上是把 "predict-then-verify" 的人类认知机制嵌进 control loop。在 long-horizon 任务里特别有价值，比如 Table Cleanup 任务需要连续做几十步决策，open-loop execution 累积误差严重，predictive gating 能在出错前就拦截掉。

实验数据很 striking：Table Cleanup 任务 w/o HIL 是 40%，w/ HIL 提升到 64%，提升 24 个百分点。Long-horizon 任务最受益于 foresight。

---

## 为什么 DINOv3 而不是 native VLM encoder

这个选择也很 deliberate。General VLM 的 visual encoder 主要为 semantic understanding 优化，知道"这是 cup"就够了。但 robotic manipulation 需要 object-centric correspondence 和 fine-grained geometry，需要知道"cup 的 handle 在哪个 pixel"。

DINOv3 的 self-supervised feature 有 sharp object boundary 和 part-level affordance alignment。Figure 6 那个可视化很直观：query fork handle 的 patch，similarity heatmap 精确激活旁边 spoon 的 handle，背景几乎零泄漏。这种 sharp object-background contrast 对 grasp planning 至关重要。

作者 frozen DINOv3，只训练后面的 compression 和 LLM backbone。这个 frozen 设计也避免了 fine-tuning 破坏预训练的 geometric property。

---

## Ablation 的几个关键 take-away

**Temporal horizon 不是越长越好**。在 256 tokens 固定预算下，T=16 的 8:6:2 配置达到 91.3%，T=34 掉到 74.4%，T=4 掉到 86.0%。呈 inverted U-shape。太短 context 不够，太长 intermediate states 被压得太狠。

**Uniform compression 远不如 adaptive**。同样 T=16，0:16:0 (uniform k=1) 只有 81.5%，8:6:2 (adaptive) 达到 91.3%。证明 immediate frame 高分辨率保留是 critical 的。

**Token efficiency**：8:6:2 (256 tokens) 几乎匹配 0:0:16 (1024 tokens) 的 92.0%，token 消耗减少 75%。这说明 adaptive compression 几乎没损失 information。

---

## 实验数据的关键观察

LIBERO 上 w/ HIL 达到 99.2%，这几乎饱和了。w/o HIL 91.3% 其实已经很好，HIL 又额外捞了 7.9 个百分点，证明 visual foresight 的 verification value。

RoboTwin 上 bimanual 任务，FUTURE-VLA 75.4% 超过 π₀.5 的 67.9%。精细任务尤其突出，Stack Bowls 94% vs 74%，Pick Dual Bottles 92% vs 75%。说明 unified tokenization 保留了 delicate manipulation 所需的 geometric fidelity。

Real-world 上 78% 超过 π₀.5 的 74%，Table Cleanup 从 40% 到 64% 是最强证据，证明 HIL 在 long-horizon chaining 任务中的实际价值。

---

## 我觉得还差点什么

几个 paper 没说清楚的点：

**Verifier 到底是人还是模型？** Paper 说 "verifier inspects future trajectory"，但没明说是 human judgment 还是 trained failure detector。如果是 human，scalability 受限，没法 autonomous 部署。如果要做 autonomous，需要训练一个 separate failure predictor，这个 paper 没涉及。

**1D tokenizer 的 temporal consistency**。每帧独立 encode 成 32 token，相邻帧的 latent code 可能不 smooth，导致 predicted future video 有 flickering。Figure 4 和 5 看起来 OK，但定量 temporal consistency metric 缺失。这是 long-horizon prediction 的潜在 problem。

**Resampling 的 latency cost**。如果 deadlock 频繁，需要多次 resample，实际 latency 可能高于 claimed 的 single-frame baseline。Paper 没给 resampling 频率统计。

**DINOv3 frozen 的 domain shift**。如果应用 domain 跟 DINOv3 预训练 domain 差异大（比如医疗、工业极端环境），frozen encoder 可能不够，需要 adaptation。Paper 的实验都在 tabletop manipulation，domain 不算极端。

---

## 我的 overall intuition

这篇 paper 的核心贡献其实是工程整合，不是新理论。它把几个已有 idea（DINOv3、FAST、TiTok-style 1D tokenizer、Qwen3-VL backbone）整合成一个 unified architecture，关键创新是 temporal-adaptive compression 和 latent-space autoregression 的对称设计。

更深层的 insight 是：robotic control 的 real-time 瓶颈不在 forecasting 本身，而在 representation space 的选择。Pixel-space rollout 永远跟不上 action frequency，但 latent-space rollout 可以。这其实跟 LeCun 的 JEPA philosophy 一致——放弃 pixel-level prediction，在 abstract latent space 做 prediction。

HIL 机制是个 nice practical contribution，但长期看 autonomous verifier 才是真正的 scalable 方案。我猜 follow-up 工作会往这个方向走：训一个 lightweight failure predictor，用 predicted latent rollout 作为 input，输出 go/no-go signal，替代 human verifier。

另一个值得探索的方向是 longer horizon。Paper 只做了 16 步 look-ahead，如果能扩展到 64+ 步（可能需要 hierarchical latent prediction），对 truly long-horizon 任务（如 assemble furniture）会更有 impact。

---

## Reference

- [FUTURE-VLA Repo (paper 里提到的)](https://github.com/OpenRobotLab/FUTURE-VLA)
- [Qwen3-VL](https://arxiv.org/abs/2511.21631)
- [FAST tokenization](https://arxiv.org/abs/2501.09747)
- [TiTok](https://arxiv.org/abs/2406.07550)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [π₀](https://arxiv.org/abs/2410.24164)
- [π₀.5](https://arxiv.org/abs/2504.16054)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
- [V-JEPA 2 (LeCun's latent prediction philosophy)](https://arxiv.org/abs/2506.09985)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)

---

# FUTURE-VLA: 深度技术讲解

你好 Andrej，让我深入剖析这篇 paper。这个工作试图解决一个根本性矛盾：robotic control 需要长时序历史来避免 perceptual myopia，同时需要 real-time inference frequency 来支持 closed-loop control。作者用"dual-sided efficiency paradigm"巧妙地把这个矛盾化解为 input compression + output latent autoregression 的组合。

---

## 1. 核心问题动机

当前 VLA + WM 领域有三种架构范式：

**(a) Modular Fragmentation**：VLA 和 WM 各自独立训练，representation 不对齐。VLA 看不到未来，WM 不知道 action 语义，导致 perceptual myopia——policy 只能基于当前 frame 反应式输出，长 horizon 任务累积误差。

**(b) Instantaneous Unification**：把 perception + prediction 塞进一个 model，但只能单帧 visual reasoning output。WM 的 anticipatory power 被浪费，因为 pixel-level rollout 太慢无法匹配 action chunking 频率。

**(c) FUTURE-VLA**：input 端用 adaptive compression 把 16 帧历史压成 256 tokens/view（= 单帧 baseline 的 token budget），output 端用 latent autoregression 在单次 forward pass 中同时生成 action chunks 和 future visual previews。

关键 insight：**real-time 的瓶颈不在 forecasting 本身，而在 pixel-space rollout 的维度灾难**。一旦把 future prediction 移到 compact latent space（32 tokens/frame），WM 就能在 closed-loop 频率下提供 actionable foresight。

---

## 2. Architecture 详解

### 2.1 视觉编码路径重构

模型基于 Qwen3-VL，但作者做了几个 surgical modification：

**DINOv3-ViT-Base 替换 native patch embedding**：
- Frozen encoder，提供 dense spatially discriminative features
- 丢弃 [CLS] token 和 register tokens，只保留 2D spatial grid
- 关键动机：manipulation 任务需要 object-centric correspondences 和 fine-grained geometric fidelity，而 general VLM 的 patch embedding 主要为 semantic understanding 优化

Figure 6 的 feature visualization 很说明问题：query 一个 fork handle 的 patch，similarity heatmap 精确激活相邻 spoon 的 handle（part-level affordance alignment），背景几乎零泄漏。这种 sharp object-background contrast 对 grasp planning 至关重要。

### 2.2 Temporally Adaptive Cascaded Compression

这是 input side 的核心创新。设输入序列：

$$\mathbf{x} = [O_1; \ldots; O_T; \text{TEXT}; \text{SYS}]$$

其中 $O_i$ 是第 $i$ 帧的 visual observation，TEXT 是 task instruction，SYS 是 system prompt。

**基础压缩单元**（公式 2）：

$$X^{(k)} = \text{GeLU}(\text{Conv}(X^{(k-1)}))$$

- $X^{(k)}$：第 $k$ 次压缩后的 feature map
- $\text{Conv}$：strided convolution（stride 2）
- $\text{GeLU}$：activation function
- 空间分辨率递归更新：$(H_k, W_k) = (H_{k-1}/2, W_{k-1}/2)$，即每次 spatial dimension 减半

**关键设计——dynamic compression depth $k_t$**：
- $k_t$ 与时间距离成反比
- 远端历史 frame：heavy compression（$k_t$ 大）→ 保留少量 global context token
- 近端 observation：light/no compression（$k_t$ 小或 0）→ 保留高分辨率 spatial detail

实现配置：T=16，分配 8:6:2
- 最早 8 帧：$k=2$，每帧 4 tokens
- 中间 6 帧：$k=1$，每帧 16 tokens
- 最近 2 帧：$k=0$，每帧 64 tokens
- 总计 $8 \times 4 + 6 \times 16 + 2 \times 64 = 32 + 96 + 128 = 256$ tokens/view

再叠加 Qwen3-VL 原生 patch merger（merge size $2 \times 2$），最终 uniform 4× reduction。

**直觉**：robotic manipulation 中，immediate frame 决定精细 grasp pose，需要 pixel-level detail；distant frame 只提供 motion context 和 object trajectory，coarse representation 足够。这是 temporal-resolution pyramid 的 inductive bias。

### 2.3 Unified Tokenization

作者把 action tokens 和 visual latent codes 都 map 到 Qwen3 tokenizer 的尾部，避免 vocabulary expansion 带来的 embedding matrix 重新初始化。

**Spectral Action Tokenization (FAST)**：
- 基于 Discrete Cosine Transform (DCT)
- 把 trajectory chunk 编码到频域
- 优先保留 high-frequency dynamics，这对快速机动动作很重要
- Vocab size $V_{act} = 2048$

**Compact 1D Visual Tokenization**（公式 3）：

$$\mathbf{Z}_{1D} = \text{Enc}(\mathbf{P} \oplus \mathbf{L}) \in \mathbb{R}^{32 \times D}$$

- $\mathbf{P}$：image patchified 后的 patch embeddings sequence
- $\mathbf{L}$：32 个 learnable latent tokens
- $\oplus$：concatenation
- $\text{Enc}$：ViT encoder，通过 attention 把 visual information 聚合到 latent tokens
- 输出只保留 latent 部分，丢弃 patch 部分
- $D$：latent dimension

**离散化**：
- Codebook $\mathcal{E} \in \mathbb{R}^{|\mathcal{V}| \times D}$，$|\mathcal{V}| = 4096$
- 预测长度 32 的 discrete code index 序列 $\mathbf{c} \in \{1, \ldots, |\mathcal{V}|\}^{32}$
- Quantized latents：$\tilde{\mathbf{Z}}_{1D} = \mathcal{E}[\mathbf{c}]$

**重建**（公式 4）：

$$\hat{\mathbf{I}} = \text{Dec}(\tilde{\mathbf{Z}}_{1D} \oplus \mathbf{M})$$

- $\mathbf{M}$：mask token grid，作为 positional anchors 提供目标 layout
- Decoder 从 quantized latents + mask grid 重建 image

**关键 insight**：传统 VQ-GAN 强制 2D spatial correspondence，latent capacity 与 image resolution 绑定。1D tokenizer 解耦这两者，固定 32 tokens 表示任意分辨率 image。Figure 7 显示重建质量几乎无 perceptual deviation，这对 WM 的 rollout crispness 至关重要。

### 2.4 Vocabulary Remapping Strategy

不 expansion 而是 remap：
- Action tokens → Qwen3 vocab 最后 2048 个 indices
- Visual tokens → 紧邻 action tokens 前的 4096 个 indices

原理：Qwen3 用 BPE，vocab 尾部对应自然语言中极罕见的字符序列。复用这些 low-frequency slot 最小化对预训练 linguistic knowledge 的干扰，同时复用已学习的 embedding matrix。

---

## 3. Inference: Predictive Look-ahead + HIL

模型单次 forward pass 输出：
- Action chunk $\mathbf{A}_{1:T}$（T=16）
- Predicted visual trajectory $\hat{\mathbf{O}}_{1:T}$（每帧 32 tokens × V views）

### 3.1 Dynamic Gating

Verifier 检查 $\hat{\mathbf{O}}_{1:T}$ 决定 safe execution horizon $k \in [1, T]$。机器人执行 $k$ 步后刷新 observation，balance open-loop speed 和 closed-loop safety。

### 3.2 Resampling Recovery

如果 $\hat{\mathbf{O}}_{1:T}$ 暗示 failure：
- 设 $k = 0$（reject，不执行任何 action）
- 提高采样 temperature 重新生成
- 探索 alternative trajectory，escape deadlock

**直觉**：WM 从 passive training signal 升级为 active guidance module。这把"predict-then-verify"的 human cognition 机制嵌入到 robotic control loop。

---

## 4. 实验数据深度分析

### 4.1 LIBERO Benchmark (Table 1)

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀+FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| FUTURE-VLA (w/o HIL) | 89.6 | 99.0 | 95.2 | 81.2 | 91.3 |
| FUTURE-VLA (w/ HIL) | **98.6** | **100** | **100** | **98.2** | **99.2** |

关键观察：
- w/o HIL 的 91.3% 已超过 OpenVLA-OFT (95.4%)？不，91.3 < 95.4，但 w/ HIL 的 99.2% 是 SOTA
- HIL 带来 +7.9% 平均提升，Long subset 从 81.2 → 98.2（+17%），证明 long-horizon 任务最受益于 predictive verification
- Object 和 Goal subset 达到 100%，说明 visual foresight 对 deterministic manipulation 几乎完美

### 4.2 RoboTwin (Table 2)

Bimanual manipulation 更难。π₀.5 达到 67.9% avg，FUTURE-VLA w/ HIL 达到 75.4%。

精细任务表现突出：
- Stack Bowls Two: 94% vs π₀.5 的 74%
- Pick Dual Bottles: 92% vs 75%
- Handover Mic: 91% vs 97%（π₀.5 略优）

这验证 unified tokenization 保留了 delicate manipulation 所需的 geometric fidelity。

### 4.3 Real-world Piper (Table 3)

| Model | Sorting | Handover | Table Cleanup | Avg |
|-------|---------|----------|---------------|-----|
| OpenVLA-OFT | 68 | 26 | 6 | 33.3 |
| π₀.5 | 92 | 72 | 58 | 74 |
| FUTURE-VLA (w/o HIL) | 96 | 66 | 40 | 67.3 |
| FUTURE-VLA (w/ HIL) | 98 | 72 | 64 | 78 |

Table Cleanup（long-horizon chaining）从 40 → 64（+24%），是 HIL 价值的最强证据。这任务需要 sequential decision chaining，open-loop execution 累积误差严重，predictive gating 能 preemptive 过滤错误 trajectory。

---

## 5. Ablation Studies 深度解读

### 5.1 Temporal Horizon Trade-off (Table 4)

固定 256 tokens/view budget：

| Allocation | T | Success |
|-----------|---|---------|
| 0:0:4 | 4 | 86.0 |
| 0:8:2 | 10 | 88.6 |
| 0:16:0 | 16 | 81.5 |
| **8:6:2** | **16** | **91.3** |
| 16:4:2 | 22 | 83.1 |
| 32:0:2 | 34 | 74.4 |

**Inverted U-shape**：
- T=4 太短，temporal context 不足
- T=34 太长，intermediate states 被 aggressive compression，丢失关键 motion cue
- 8:6:2 optimal：远端粗略 + 近端精细的 pyramid 结构

注意 0:16:0（uniform k=1）只有 81.5%，证明 uniform compression 远不如 adaptive。

### 5.2 Compression Strategy (Table 5)

固定 T=16：

| Allocation | Tokens | Success |
|-----------|--------|---------|
| 16:0:0 | 64 | 30.2 |
| 12:4:0 | 112 | 65.4 |
| 0:16:0 | 256 | 81.5 |
| **8:6:2** | **256** | **91.3** |
| 8:4:4 | 352 | 90.8 |
| 0:0:16 | 1024 | 92.0 |

**惊人发现**：8:6:2 (256 tokens) 几乎匹配 0:0:16 (1024 tokens) 的 92.0%，token 消耗减少 75%。

而 16:0:0（全部 heavy compression）崩溃到 30.2%，证明 immediate observation 的高分辨率 retention 是 critical 的。

**直觉**：robotic manipulation 的 information density 在时间维度上极不均匀。最近 2 帧包含 grasp pose 的精细 spatial cue，远端 8 帧只需提供 "object was moving left" 级别的 motion context。adaptive compression 正好匹配这种 information distribution。

---

## 6. Training Details

### 6.1 Compact 1D Tokenizer Training

两阶段策略（following TiTok）：
- **Stage 1**：structural warm-up，预测 pre-trained MaskGIT-VQGAN 的 proxy codes
- **Stage 2**：pixel-level reconstruction loss fine-tuning

Batch size 256（32/GPU × 8 GPUs），约束于硬件。

### 6.2 FUTURE-VLA Training

- Framework: ms-swift
- Optimizer: AdamW, $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$
- Weight decay: 0.1
- LR schedule: cosine decay with linear warm-up, peak $1 \times 10^{-4}$
- 2 epochs
- Global batch size: 64

**关键实现技巧**：bypass 标准 text tokenizer for output targets，直接 feed pre-computed action/visual token IDs 进 label sequence。避免 discrete codes 被 re-encode 成 UTF-8 strings 带来的噪声。

---

## 7. Intuition Building: 为什么这个架构 work

### 7.1 Dual-sided Efficiency 的对称性

Input side 和 output side 都面临"信息密度 vs 计算预算"的矛盾，但解法镜像对称：
- Input：temporal-adaptive compression（远压缩近保留）
- Output：latent-space autoregression（抛弃 pixel-level rollout）

这种对称性让 16× context extension 不增加 latency。

### 7.2 为什么 1D Tokenizer 比 2D VQ-GAN 适合 WM

2D VQ-GAN 的 latent code 与 spatial position 一一对应，要生成高分辨率 future frame 必须生成大量 token（如 256×256 image 需要 1024+ tokens）。这在 autoregressive generation 下是 latency disaster。

1D tokenizer 把 image 压成固定 32 tokens，generation cost 与 resolution 解耦。Figure 7 证明 32 tokens 足以重建 fine detail，因为 natural image 有 massive patch-level redundancy。

### 7.3 为什么 Spectral Tokenization 对 Action 有效

Action trajectory 在时域上往往 smooth + 局部 high-frequency（如 grasp瞬间的高速调整）。DCT 把能量集中到低频成分，少量 frequency coefficient 即可重建 trajectory。这比时域 discretization 压缩率高得多。

### 7.4 HIL 的认知科学基础

Human 操作员的角色类似于 "verifier"——模型提供 "what if" preview，人类基于 visual foresight 做 go/no-go 决策。这 mimics 人类自身的 "mental simulation before action" 机制。Resampling Recovery 用 temperature increase 来 escape local mode，类似 human 在卡住时尝试 alternative strategy。

---

## 8. 局限性与未来方向

Paper 没明确讨论的几个点：

1. **Verifier 的定义模糊**：paper 说 "verifier inspects $\hat{\mathbf{O}}_{1:T}$"，但没说 verifier 是 human judgment 还是 automated module。如果是 human，scalability 受限；如果是 automated，需要训练 separate failure detector。

2. **1D Tokenizer 的 temporal consistency**：每帧独立 encode 成 32 tokens，相邻帧的 latent code 可能不 smooth，导致 predicted future video 有 flickering。Paper 的 visualization（Figure 4, 5）看起来 OK，但 quantitative temporal consistency metric 缺失。

3. **HIL 的 latency cost**：resampling 时需要重新 forward pass，如果 deadlock 频繁，实际 latency 可能高于 claimed 的 single-frame baseline。

4. **DINOv3 frozen 的限制**：frozen encoder 无法 fine-tune 到 task-specific feature。虽然 Figure 6 显示 feature 质量好，但 extreme domain shift（如医疗机器人）可能需要 adaptation。

5. **Real-world generalization**：78% avg 在 3 个 task 上，task diversity 有限。需要更大 scale 的 real-world evaluation。

---

## 9. 与相关工作的 positioning

- vs **π₀.5**：π₀.5 是 flow-based VLA，no explicit WM。FUTURE-VLA 把 WM 嵌入 VLA，提供 interpretable foresight。
- vs **WorldVLA**：WorldVLA 是 autoregressive action world model，但 pixel-space prediction。FUTURE-VLA 用 latent space，real-time。
- vs **Genie / DIAMOND**：这些是 pure WM，no action grounding。FUTURE-VLA unify action + prediction。
- vs **TiTok**：FUTURE-VLA 借鉴 TiTok 的 1D tokenizer，但 apply 到 robotic WM context，并集成进 VLA vocab。

---

## Reference Links

- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [FAST: Efficient Action Tokenization for VLA Models](https://arxiv.org/abs/2501.09747)
- [TiTok: An Image is Worth 32 Tokens](https://arxiv.org/abs/2406.07550)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [π₀: Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164)
- [π₀.5: VLA with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [LIBERO Benchmark](https://arxiv.org/abs/2306.03310)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
- [Unified World Models (Zhu et al.)](https://arxiv.org/abs/2504.02792)
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)
- [Cosmos World Foundation Model](https://arxiv.org/abs/2501.03575)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2401.04024)
- [ms-swift framework](https://github.com/modelscope/ms-swift)
- [MaskGIT](https://arxiv.org/abs/2208.04202)

---

## 总结

FUTURE-VLA 的核心贡献是把 robotic control 的 spatiotemporal bottleneck 重新 formulate 为 sequence generation 问题，用 dual-sided efficiency 同时解决 input history explosion 和 output prediction latency。HIL 机制把 WM 从 passive training signal 升级为 active safety guardian，这在 real-world deployment 中有 immediate practical value。

最值得 follow 的 direction：把 verifier 自动化（学习一个 failure predictor 而非依赖 human），以及探索 1D tokenizer 在 longer horizon（如 64+ frames）下的 temporal consistency。
