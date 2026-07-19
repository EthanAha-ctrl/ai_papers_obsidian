---
source_pdf: BitVLA.pdf
paper_sha256: b6ee0423ffaa1050861acbaf9d1b473d75d641bc27e5612886daa42df990518a
processed_at: '2026-07-18T19:26:57-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BitVLA: 1-bit Vision-Language-Action Models 深度解析

## 1. 这篇 paper 的核心 motivation

VLA models (如 OpenVLA 7.5B, π0 3.5B) 在 LIBERO 这类机器人 manipulation benchmark 上表现很强, 但 deployment 到 edge device (比如 NVIDIA RTX 3050 Ti 4GB) 时, memory 是硬约束。已有的 1-bit LLM 工作 ([BitNet b1.58](https://arxiv.org/abs/2402.17764), [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285)) 证明了 ternary weights {-1, 0, 1} 在 2B 规模可以媲美 fp16 LLM, 但是把这套思路搬到 multimodal + action generation 上没有先例。BitVLA 是第一个完全 ternary 的 VLA, 而且 key insight 是: **vision encoder 的 quantization 可以靠 distillation 高效完成, 不需要重新大规模 pretrain**。

---

## 2. Architecture 细节

### 2.1 三大组件

| Component | 选择 | 精度 | 备注 |
|---|---|---|---|
| LLM backbone | BitNet b1.58 2B4T | W1.58 A8 | 所有 linear 层 ternary |
| Vision encoder | SigLIP-L (224×224) | 训练后 W1.58 A8 | 通过 distillation-aware training 量化 |
| Connector | 2-layer MLP + GeLU | W16 A16 | 占比小, 不量化 |
| Action head | MLP | fp | 投影到连续 action space |

这里有个关键细节: SigLIP-L 用 224×224 而不是 384×384, 是为了 visual token 数更短 (576 tokens instead of 729/1024), 这对 1-bit LLM 的 KV cache 和 attention 计算成本影响很大。OpenVLA 也是这么做的, [参见](https://arxiv.org/abs/2406.09246)。

### 2.2 量化公式逐变量解析

**权重 quantization (absmean)**:

$$Q_w(W) = \alpha \cdot \text{RoundClip}\left(\frac{W}{\alpha}, -1, 1\right), \quad \alpha = \frac{1}{nm}\|W\|_1$$

变量含义:
- $W \in \mathbb{R}^{m \times n}$: 某个 linear layer 的 weight matrix, $m$ 是输出维度, $n$ 是输入维度
- $\alpha$: 一个 scalar clipping threshold, 等于 $W$ 所有元素绝对值的均值 (注意是 $\frac{1}{nm}$, 整个 matrix 平均)
- $\text{RoundClip}(x, a, b)$: 先 round 到最近整数, 再 clip 到 $[a, b]$ 范围
- $Q_w(W)$: 输出是 ternary matrix, 每个元素 $\in \{-1, 0, 1\}$, 再乘回 $\alpha$ 作为 scale

**Activation quantization (per-token absmax)**:

$$Q_a(x) = \frac{\beta}{127} \cdot \text{RoundClip}\left(\frac{127 x}{\beta}, -128, 127\right), \quad \beta = \|x\|_\infty$$

变量含义:
- $x \in \mathbb{R}^{n \times 1}$: 一个 token 的 activation vector (注意是 per-token, 不是 per-tensor)
- $\beta = \|x\|_\infty$: 这个 token 内所有元素的最大绝对值 (一个标量)
- 量化到 INT8 范围 $[-128, 127]$, 再 dequantize 回来时乘 $\beta/127$

**为什么 absmean 而不是 absmax for weights**: ternary 只有 3 个值, outlier 会被 clip 掉, 所以 absmean 更鲁棒; 而 activation 一定要 absmax + per-token 才能保 outlier, 这是 [BitNet b1.58 原论文](https://arxiv.org/abs/2402.17764) ablation 出来的结论。

**Forward 计算**:

$$Y = Q_w(W) \cdot Q_a(x)$$

这里有个重要的硬件收益: $Q_w(W)$ 是 ternary, 所以 matmul 可以用 ternary-bit 累加器实现, 而不是 floating point multiply-accumulate, 能耗和 latency 都大幅下降, [bitnet.cpp](https://arxiv.org/abs/2502.11880) 专门做了 CPU 推理优化。

### 2.3 STE (Straight-Through Estimator)

量化函数 round 是非可导的, backprop 用 STE:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial Q_w(W)}, \quad \frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Q_a(X)}$$

直觉: forward 真的走量化函数, backward 假装它是个 identity。latent weight $W$ 保留 fp32, optimizer state 也是 fp32, 这样 Adam 的 momentum 不会因为 ternary 而崩。这个 trick 来自 [Bengio et al. 2013](https://arxiv.org/abs/1308.3432)。

---

## 3. 三阶段训练流程 (Build Intuition)

### Stage I: Connector alignment
- 数据: LLaVA 1.5-558k (image captioning)
- 只训 connector (2-layer MLP), 让 vision feature 和 LLM embedding space 对齐
- LR 1e-3, 25k steps, batch 256, seq len 1024
- 这里 vision encoder 和 LLM 都 frozen

### Stage II: Visual instruction tuning
- 数据: MammoTH-VL 10M 子集 (single-image)
- 训 LLM + connector, 冻 vision encoder
- LR 3e-4, polynomial decay, weight decay 0.1→0 (大 LR 配合 weight decay warmup 是 1-bit 训练的稳定 trick)
- 40k steps, seq len 2048, visual token len 256

### Stage III: Distillation-aware training (核心创新)
- 数据: 5M 子集 (从 Stage II 数据里采), 大约 10B tokens
- **只训 vision encoder**, LLM 和 connector 冻住
- LR 1e-4, 20k steps
- 把 SigLIP-L 从 W16A16 量化到 W1.58A8

这个设计的关键 insight: **量化 vision encoder 不需要 robotics data, 也不需要重新跑整个 multimodal pretraining**。因为 vision encoder 的 representation 质量直接决定 downstream 表现, 用 full-precision encoder 当 teacher 做 latent alignment, 就能在通用 image-text data 上把 1.58-bit encoder 教出来。

---

## 4. Distillation-aware Training 的 Loss 设计

### 4.1 总 loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \gamma \cdot \mathcal{L}_{\text{aux}}$$

$\gamma = 0.1$ 在实验里效果最好。

### 4.2 Language modeling loss

$$\mathcal{L}_{\text{LM}} = -\sum_{\text{token}_i \in \mathcal{T}_{\text{ans}}} \log P_r\left(\mathcal{V}^i \mid \mathcal{V}_{1.58\text{-bit}}, \mathcal{T}^{[:i-1]}\right)$$

变量:
- $\mathcal{T}_{\text{ins}}$ / $\mathcal{T}_{\text{ans}}$: 输入 text 分为 instruction 和 response 两段
- $\mathcal{V}_{1.58\text{-bit}}$: 1.58-bit vision encoder 输出的 visual tokens
- $\mathcal{V}^i$: 第 $i$ 位置的 ground-truth token
- $P_r$: 模型 (with 1.58-bit encoder) 的预测概率
- Loss 只在 response token 上算 (标准 LLM/VLM 做)

### 4.3 Representation alignment loss

$$\mathcal{L}_{\text{aux}} = \frac{1}{n} \sum_{l=1}^{L} \left\| h_{\text{bf16}}^l - h_{1.58\text{-bit}}^l \right\|^2$$

变量:
- $h_{\text{bf16}}^l$: full-precision (bf16) SigLIP 第 $l$ 层的输出 (teacher)
- $h_{1.58\text{-bit}}^l$: 1.58-bit SigLIP 第 $l$ 层的输出 (student)
- $L$: vision encoder 总层数 (SigLIP-L 是 27 层 transformer)
- $n$: hidden dimension (SigLIP-L 是 1024)
- 每层 MSE, 再 sum 起来, 除以 hidden dim 归一化

**为什么每层都 align 而不是只 align 最后一层**: vision encoder 是 transformer stack, 中间层 representation 也要保。如果只 align 最后一层, student 可能在浅层就 drift 了, 最终输出 layer 拼命补偿, 导致 representation 退化。这个观察和 [OneBit](https://arxiv.org/abs/2402.17652) 在 LLM 上做 distillation 的思路一致。

### 4.4 Ablation 数据 (Table 4, 5)

| Tokens | $\mathcal{L}_{\text{aux}}$ | MMMU | SeedBench | SeedBench2+ | MMStar | AI2D | Avg VQA |
|---|---|---|---|---|---|---|---|
| 10B | ✓ | 35.4 | 69.3 | 43.7 | 41.5 | 67.6 | **51.5** |
| 5B | ✓ | 33.3 | 69.1 | 43.3 | 41.4 | 66.4 | 50.8 |
| 5B | ✗ | 32.4 | 52.9 | 38.8 | 30.7 | 57.5 | 42.4 |

**直觉解读**: 
- 没有 $\mathcal{L}_{\text{aux}}$ 时 VQA 平均掉 8.4 个点 (50.8→42.4), 证明纯 quantization-aware training 没有 distillation 是不行的
- 5B vs 10B 只差 0.7 个点, 说明 distillation 是 data-efficient 的, 不需要海量数据
- LIBERO 上 (Table 5) gap 更小, 因为 fine-tuning 会掩盖一些 quantization 误差, 但 $\mathcal{L}_{\text{aux}}$ 在 LIBERO-Goal 上仍有 2.4% 提升

---

## 5. Robotics Fine-tuning (OFT 风格)

BitVLA 没有自己 reinvent robotics fine-tuning, 而是直接用 [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) 的 recipe。关键四件套:

### 5.1 Parallel decoding + bidirectional attention
标准 LLM 用 causal mask, 每个 token 只能看前面。OFT 把这个改成 **bidirectional mask**, 这样一次 forward pass 就能并行 decode $K=8$ 个 action 步, 不需要 autoregressive。

直觉: robot control 的 action chunk 不是一个 sequence generation 问题, 是一个 trajectory regression 问题。bidirectional mask 让 8 个 action token 互相看到, 生成的是 coherent trajectory, 而不是条件独立的 step。

### 5.2 Action chunking
每次 forward 输出 $K=8$ 步连续 action, 执行完整个 chunk 再 re-plan。这降低了 inference frequency, 对 1-bit model 在 edge device 上的 latency budget 更友好。来自 [ACT / Diffusion Policy](https://arxiv.org/abs/2304.13705) 的思路。

### 5.3 MLP action head
不用 LLM 的 token vocabulary + softmax 出离散 action, 而是在 LLM 的 latent 上接一个 MLP 直接投影到连续 7-DoF end-effector pose (position + rotation + gripper)。

### 5.4 L1 loss
$$\mathcal{L}_{\text{action}} = \|a_{\text{pred}} - a_{\text{gt}}\|_1$$

用 L1 而不是 L2, 因为 robot action 有 outlier (比如突然夹紧 gripper), L1 对 outlier 更鲁棒。

### 5.5 输入构造
- Multi-view: wrist camera + external camera, 各自过 SigLIP-L 拼接
- Proprioception (end-effector pose): MLP 投影成 **1 个 token**, 拼到 image tokens 后面
- 没有用 chunk-by-chunk 的 history, 每个 chunk 都是 from scratch 的 re-plan

---

## 6. 实验数据深度解读

### 6.1 Table 1: LIBERO 主结果

| Model | Size | Memory | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|---|
| OpenVLA (w/ pretrain) | 7.5B | 15.1GB (10.79×) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 (w/ pretrain) | 3.5B | 7.0GB (5.0×) | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT (w/ pretrain) | 7.7B | 15.4GB (11×) | 97.6 | 98.4 | 97.9 | 94.5 | **97.1** |
| OpenVLA-OFT (no pretrain) | 7.7B | 15.4GB (11×) | 94.3 | 95.2 | 91.7 | 86.5 | 91.9 |
| **BitVLA** | 3.0B | **1.4GB (1×)** | 97.4 | 99.6 | 94.4 | 87.6 | **94.8** |

**关键观察**:
1. BitVLA (3B, no robotics pretrain) 击败了 π0 (3.5B, w/ pretrain) 平均高 0.6%, 尤其 Long 任务高 2.4%
2. BitVLA vs OpenVLA-OFT (no pretrain) 几乎打平 (94.8 vs 91.9), 但 memory 只有 1/11
3. Object suite 上 BitVLA 居然 99.6%, 比 OpenVLA-OFT w/ pretrain (98.4) 还高, 这有点 surprise, 可能是因为 Object suite 任务相对简单, 1-bit 容量足够
4. Long suite 是最大 gap: 87.6 vs 94.5 (with pretrain), 1-bit LLM 在长 horizon reasoning 上仍是 bottleneck

### 6.2 Table 2: PTQ 对比

| Model | Memory | Avg |
|---|---|---|
| OpenVLA INT8 | 7.4GB (5.29×) | 76.9 |
| OpenVLA-OFT INT8 | 7.7GB (5.5×) | 96.7 |
| OpenVLA INT4 | 4.4GB (3.14×) | 72.7 |
| OpenVLA-OFT INT4 | 4.7GB (3.36×) | 96.9 |
| BitVLA | 1.4GB (1×) | 94.8 |

**关键 insight**: 
- OpenVLA 7B 模型 PTQ 到 INT4 性能从 76.5→72.7 (掉 3.8 点), 说明 post-training quantization 在 VLA 上不稳定
- OpenVLA-OFT 的 INT4 比 INT8 还高一点点 (96.9 vs 96.7), 这个反常可能是因为 OpenVLA-OFT 用了 parallel decoding + action chunking, 对 weight precision 不敏感, 但对模型架构敏感
- BitVLA 是 native 1-bit (训练时就量化), 比相同 memory budget 的 PTQ 模型表现强, 这是 native 1-bit 训练 vs post-training quantization 的核心优势

### 6.3 Table 3: VQA 性能

| Model | MMMU | SeedBench | SeedBench2+ | MMStar | AI2D | Avg |
|---|---|---|---|---|---|---|
| BitVLA w/ 16-bit VE | 37.4 | 70.6 | 45.0 | 43.6 | 68.6 | 53.0 |
| BitVLA w/ 1.58-bit VE | 35.4 | 69.3 | 43.7 | 41.5 | 67.6 | 51.5 |

Vision encoder 从 bf16 到 1.58-bit 只掉 1.5% 平均, 但 memory 从 0.8GB 降到 0.1GB (8× 压缩)。这是 distillation-aware training 的直接证据。

---

## 7. 失败模式分析 (Section 5)

作者把 BitVLA 在 LIBERO 上的失败分三类:

### 7.1 Spatial localization discrepancy (最多)
- Wine bottle 等重心不稳物体, 小 pose 误差就翻
- Grasp pose 不精确导致 transport 时掉落
- Phantom manipulation (没有物体却尝试操作)
- 放置位置偏差 (LIBERO 严格要求 plate 中心)

**Intuition**: 这是 vision encoder 量化后的 spatial understanding 粗糙化。SigLIP 本身 spatial resolution 就有限 (224×224), 再 1.58-bit 后, fine-grained pose 信息丢失。这个是 1-bit vision encoder 的本质 trade-off, 不容易解决, 除非用更高 resolution 或 3D encoder (如 [SpatialVLA](https://arxiv.org/abs/2501.15830))。

### 7.2 Goal misunderstanding
- 任务执行中错误接触非目标物体, 触发新的 rollout
- 视觉/proprioceptive 信息 dominance 导致 goal switching 时混淆
- OpenVLA-OFT 用 FiLM 策略缓解, BitVLA 没有这层

**Intuition**: 这是 LLM capacity 问题。BitNet b1.58 2B4T 是 2B 参数, 比 OpenVLA 用的 Llama-2 7B 小很多。instruction following 能力受限, 尤其在长 horizon task 切换 goal 时。

### 7.3 Trajectory planning failure
- 抽屉打开后机械臂撞到下方
- 没有 collision-free 轨迹规划
- 早期 sub-goal 没考虑后期 sub-goal 的 feasibility

**Intuition**: 这是 VLA paradigm 本身的局限, 不只是 BitVLA。VLA 直接 predict end-effector pose, 不显式建模 collision。Flow matching 模型 (π0) 或 diffusion policy 在这方面更好, 因为它们 sample 多模态 trajectory, 可以避开 collision。BitVLA 的 MLP action head + L1 loss 倾向 deterministic, 在复杂几何场景下吃亏。

---

## 8. 我对这篇 paper 的几个 critical 观察

### 8.1 "no robotics pretraining" 的 framing
作者反复强调 BitVLA 没用 OXE pretraining, 这其实是 **必要的妥协**, 因为 1-bit LLM 在 robotics action token 上的 pretraining 还没人做过。如果要做 1-bit native VLA pretraining, 需要从头训一个 1-bit VLM on OXE, 计算成本极高。distillation-aware training 这一步其实是在做 "minimal-effort 的 VLM adaptation", 让通用 1-bit VLM 能直接 fine-tune 到 robotics。

### 8.2 1.58-bit vision encoder 的 memory 占比
整个 BitVLA 1.4GB, 但 vision encoder 只占 0.1GB, LLM 占 1.3GB。所以 vision encoder 量化的 memory 收益其实没那么显著, 真正省 memory 的是用了 2B 的 1-bit LLM。distillation-aware training 的更大价值是 **保留 representation 质量**, 让 fine-tuning 后性能不掉太多, 而不是 memory。

### 8.3 为什么不用 1-bit 从头训 SigLIP
理论上可以, 但 SigLIP 的 pretraining 数据 (web-scale image-text) 太大, 1-bit 从头训成本高。distillation 是 cost-effective 的 shortcut, 用 5B tokens 就能跑出来, 这是工程上的明智选择, 但也意味着 student 的 ceiling 受 teacher 限制。

### 8.4 Long horizon 的 gap
BitVLA 在 LIBERO-Long 上落后 OpenVLA-OFT (w/ pretrain) 6.9 个点。这个 gap 很难只靠 fine-tuning 弥补, 因为 long-horizon 需要 model 内部的 multi-step planning 能力, 这跟 pretraining data 多样性高度相关。如果未来能做 1-bit native VLA pretraining on OXE, 这个 gap 应该能缩小。

### 8.5 Connector 不量化的合理性
Connector 是 2-layer MLP, 参数量相对整个模型极小 (大概几十 MB), 但它是 vision feature → LLM embedding 的桥, 量化误差会直接污染 LLM input。所以这里保留 fp 是合理的, 但也意味着 BitVLA 严格说不是 "every parameter is ternary" (虽然 abstract 这么写), 而是 "every parameter in LLM and vision encoder is ternary"。

---

## 9. 跟其他 1-bit / VLA 工作的关系

### 9.1 BitNet 谱系
- [BitNet (2023)](https://arxiv.org/abs/2310.11453): 证明 1-bit 在 scale 上能 match fp
- [BitNet b1.58 (2024)](https://arxiv.org/abs/2402.17764): 引入 {-1, 0, 1} 三值
- [BitNet b1.58 2B4T (2025)](https://arxiv.org/abs/2504.12285): 公开的 2B ternary LLM, BitVLA 直接用它
- [BitNet v2 (2025)](https://arxiv.org/abs/2504.18415): Hadamard 变换做 4-bit activation
- [BitNet a4.8 (2024)](https://arxiv.org/abs/2411.04965): 混合精度 activation

BitVLA 是这个谱系第一个把 1-bit 推到 multimodal + action。

### 9.2 VLA 谱系
- [RT-1 / RT-2 / RT-X](https://arxiv.org/abs/2310.12931): Google 奠基工作
- [OpenVLA](https://arxiv.org/abs/2406.09246): 7B 开源 VLA, OXE pretrain
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645): 优化 fine-tuning, BitVLA 直接复用其 recipe
- [π0](https://arxiv.org/abs/2410.24164): flow matching VLA
- [NORA](https://arxiv.org/abs/2504.19854): Qwen2.5-VL 3B 小 VLA
- [CoT-VLA](https://arxiv.org/abs/2503.22020): 加 visual chain-of-thought
- [SpatialVLA](https://arxiv.org/abs/2501.15830): 3D 空间感知 VLA
- [TinyVLA](https://arxiv.org/abs/2409.12514): 1.3B 跳 pretrain 的 VLA

BitVLA 在 efficiency 方向和 TinyVLA、NORA 同赛道, 但用 1-bit 实现了最低 memory。

### 9.3 Distillation + Quantization 谱系
- [OneBit](https://arxiv.org/abs/2402.17652): 1-bit LLM distillation
- [LLM-QAT](https://arxiv.org/abs/2207.03181): data-free QAT with KD
- BitVLA 的 $\mathcal{L}_{\text{aux}}$ 沿用了 layer-wise MSE distillation 思路, 但只 apply 到 vision encoder, 没有对 LLM 做, 因为 LLM 是 native 1-bit 训练好的, 不需要再 align。

---

## 10. 对未来工作的几个 speculation

1. **1-bit native VLA pretraining**: 用 BitNet b1.58 直接在 OXE 上 pretrain 一个 VLA, 理论上能闭合 Long suite 的 gap。计算成本: OXE 约 1M+ episodes, action token 化后可能 10B-50B tokens, 1-bit 训练应该可行。

2. **更高 resolution SigLIP 量化的 trade-off**: 384×384 量化后 spatial error 可能下降, 但 visual token 翻倍, KV cache 增大。值得 ablation。

3. **Flow matching action head on 1-bit LLM**: π0 的 flow matching 可能比 MLP action head 在 trajectory planning failure 上更鲁棒。能不能在 1-bit LLM 上实现 flow matching 是开放问题。

4. **STE 在 vision encoder 上的退化**: 论文没分析 SigLIP 在 STE 训练下哪些层最容易 drift。如果加 weight noise 或者 LR per-layer 调整, 可能更稳。

5. **Edge deployment 实测**: 论文只跑 simulation, 实际 robot 上 1-bit matmul 是否有 bitnet.cpp 那样的 speedup 还需要验证, 尤其是 GPU/CPU 都跑的话。

---

## 11. 总结: 这篇 paper 的真正贡献

我的直觉是, BitVLA 最重要的不是 "1-bit VLA" 本身, 而是 **distillation-aware training 这套把通用 full-precision vision encoder 嫁接到 1-bit LLM 上的 pipeline**。这个 pipeline 的核心是:
- 用 full-precision encoder 当 teacher, 让 student 1.58-bit encoder 的中间层 representation 对齐
- 这个 stage 是 data-efficient (5-10B tokens) 的, 不需要 robotics 数据
- 训完直接接 OFT fine-tuning 就能用

这套 pipeline 给未来 "1-bit multimodal model" 提供了一个 cost-effective 的训练 template, 不仅 robotics 能用, 任何 VLM downstream task 都能受益。这个 generalizable 的方法学贡献, 比具体的 LIBERO 数字更重要。

论文代码: [https://github.com/ustcwhy/BitVLA](https://github.com/ustcwhy/BitVLA)

BitNet b1.58 2B4T: [https://arxiv.org/abs/2504.12285](https://arxiv.org/abs/2504.12285)

OpenVLA-OFT: [https://arxiv.org/abs/2502.19645](https://arxiv.org/abs/2502.19645)

LIBERO benchmark: [https://arxiv.org/abs/2306.03310](https://arxiv.org/abs/2306.03310)

SigLIP: [https://arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343)

bitnet.cpp: [https://arxiv.org/abs/2502.11880](https://arxiv.org/abs/2502.11880)
