---
source_pdf: 3D-MIX for VLA A Plug-and-Play Module for Integrating VGGT-based.pdf
paper_sha256: 07a12561bf6acc06735db0becb3768efab30701a183e5cf1a9f3d89d91fbf516
processed_at: '2026-07-17T09:27:28-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# 3D-MIX for VLA 深度讲解

Andrej, 这篇 paper 做的事情其实很 grounded——把一个 pretrain 的 3D geometry encoder (VGGT) plug 进 modern VLA 架构，并且系统化地比较了 9 种 fusion 方式，最终发现一个简单的 **semantic-conditioned gated fusion** 胜过所有花哨方案。下面我尽量 build your intuition，公式逐变量拆解，并把 paper 没说透的 connection 也铺开。

---

## 1. Motivation: 为什么 VLA 需要 3D？

现代 VLA (OpenVLA [[1](https://openvla.github.io/)], π₀/π₀.₅ [[2](https://www.physical intelligence.company/blog/pi0)], GR00T-N1.6 [[3](https://research.nvidia.com/labs/gear/gr00t-n1_6/)]) 都建立在 MLLM 之上, 而 MLLM 的 visual pretraining 几乎全是 **2D image-text pairs**, 几何监督信号缺位。结果就是 MLLM 的 **spatial intelligence** 天花板低 —— 做抓取的时候 grasp pose、relative depth、object-to-object spatial relation 都依赖从 RGB pixel 隐式推理, 错误率高。

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) [[4](https://vggt.github.io/)] 恰好是一个统一 geometry encoder：输入 single/multi-view RGB，输出 patch-aligned geometric tokens (point map, depth, camera pose, point tracks 等)，这种 **geometrically grounded features** 正好是 manipulation 任务缺的料。问题是：**3D tokens 该从哪里塞、怎么塞？** 之前的工作各自搞一套，没有 systematic comparison，所以 paper 的 pilot study 就是为这件事服务的。

---

## 2. Base VLA Architecture: 两种 style 的技术拆解

### 2.1 GR00T-style (modular dual-system)

```
RGB×V + lang ℓ  ──► MLLM (Qwen3-VL) ──► H ∈ R^{B×L×D}
                                              │
                                              ▼ cross-attention
                            noisy action A_τ ──► DiT flow-matching ──► action A
```

- **MLLM 输出**：只用 **final-layer hidden states** H，shape 是 `B × L × D`，B 是 batch，L 是 token sequence length (image patches + lang tokens)，D 是 hidden dim。
- **Action expert**：DiT (Diffusion Transformer) [[5](https://www.wpeebles.com/DiT)] 用 flow-matching 目标训练。训练时把 ground-truth action chunk A 与高斯噪声 ε 线性插值：
  
  $$\mathbf{A}_\tau = (1-\tau)\,\boldsymbol{\epsilon} + \tau\,\mathbf{A}, \quad \tau \sim \mathrm{Beta}(\alpha,\beta), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

  这里 **τ 是 flow-matching 的时间步 (上标无, 下标 τ)**, α, β 控制 τ 的分布形状 (Beta 参数), ε 是标准高斯噪声, A 是真实 action chunk `a_t, a_{t+1}, ..., a_{t+T}`. 模型预测 velocity $\mathbf{v} = \mathbf{A} - \boldsymbol{\epsilon}$, 用 MSE loss 监督。
  
- **Cross-attention 接口**：每个 DiT layer 都对同一个 H 做 cross-attention (公式 11):
  $$\mathbf{Z}^{(i)} = \mathrm{TransformerBlock}^{(i)}(\mathbf{Z}^{(i-1)}, \mathbf{H})$$
  Z^(i) 是第 i 层 action latent, H 是 K/V source。优点是 modular, backbone-agnostic；缺点是只有 final-layer 信息，hierarchical semantic 没用上。

### 2.2 π-style (layer-wise coupling)

π₀ / π₀.₅ 系列的设计：从 MLLM 的**最后 N_dit 层**抽取 hidden states，一一对应到 DiT 的每一层：

$$\{\mathbf{H}^{(1)}, \dots, \mathbf{H}^{(N_{\mathrm{dit}})}\} = \mathrm{MLLM}_{\text{last-}N_{\mathrm{dit}}\text{-layers}}(\mathcal{I}, \ell)$$

- N_dit 是 DiT 层数 (下标 dit 表示 "DiT"), 上标 (i) 是层索引。
- 第 i 个 DiT block 接受 H^(i) 作为 cross-attention K/V (公式 13):
  $$\mathbf{Z}^{(i)} = \mathrm{TransformerBlock}^{(i)}(\mathbf{Z}^{(i-1)}, \mathbf{H}^{(i)})$$

**Intuition**：浅层 DiT 拿到浅层 MLLM features (低层 visual + syntactic)，深层 DiT 拿到深层 MLLM features (semantic intent + spatial relation)。代价是显存存 N_dit 份 hidden states，cross-attention 计算量也 ×N_dit。这正是 3D-MIX 在 π-style 上要做 layer-wise fusion 的原因。

---

## 3. Pilot Study: 9 种 Fusion Schemes 逐一拆解

这是 paper 的核心 empirical 贡献。9 个 scheme 覆盖了 VLA pipeline 的所有可能 injection point。

### 3.1 AE-Fusion (Action Expert dual cross-attention)

DiT 内部加第二个 cross-attention head，同时 attend MLLM hidden states H 和 projected VGGT features F_VGGT'：
- 优点：late fusion，upstream 不动
- 缺点：要改 action head 源码，破坏 DiT 原始结构
- **结果**：SIMPLER 3.13% (崩溃), LIBERO 97.40% (反而提升)。说明 in-domain LIBERO 不太需要 3D，OOD SIMPLER 则需要 spatial 信息，而直接修改 DiT 让原本训练好的 cross-attention pattern 重新学习，spatial 信号没正确接进去。

### 3.2 Early Fusion (token injection at MLLM input)

把 VGGT tokens 直接 concat 到 MLLM input sequence：
$$\mathbf{X}_{\mathrm{fused}} = [\mathbf{X}_{\mathrm{input}}; \mathbf{W}_{\mathrm{proj}}\mathbf{F}_{\mathrm{VGGT}}] \in \mathbb{R}^{B \times (L+N) \times D}$$

- 优点：让 MLLM 的 self-attention 隐式学 semantic-geometric interaction
- 缺点：input sequence 大幅变长 (L+N)，attention budget 被稀释，而且**几何 tokens 占用 MLLM 的 receptive field，破坏它原有的 language-vision alignment**
- **结果**：SIMPLER 44.53% (掉), LIBERO 86.45% (掉得更惨)。这是个很强的 negative result —— **pretrained MLLM 的 input space 不应该被随便污染**。

### 3.3 Concat Fusion & 3.4 CrossAttn Fusion

两者都先做 **GateMixer preprocessing**：把 VGGT 输出 split 成 frame-specific + global geometry components，用 learnable gate 融合，project 到 MLLM hidden dim 得到 F_geo。

- **Concat Fusion**：直接 concat $\mathbf{H}_{\mathrm{fused}} = [\mathbf{H}_{\mathrm{MLLM}}; \mathbf{F}_{\mathrm{geo}}]$，让 action expert 的 cross-attention 自己学融合。结果 SIMPLER 60.42%, LIBERO 97.75% —— 还不错，比 base 略好。
- **CrossAttn Fusion**：先做 residual cross-attention：
  $$\mathbf{F}_{\mathrm{geo}}' = \mathrm{CrossAttn}(\mathbf{F}_{\mathrm{geo}}, \mathbf{H}_{\mathrm{MLLM}}, \mathbf{H}_{\mathrm{MLLM}}) + \mathbf{F}_{\mathrm{geo}}$$
  结果 SIMPLER 56.25%, LIBERO 83.45% —— 反而崩了。**Intuition**：显式 cross-attention 让 geometric tokens 过度向 semantic tokens 对齐，丢失了几何锐度，spatial precision 反而被磨平。

### 3.5 Gated Fusion (paper 的 winner)

公式 (1)-(5) 的组合。下面把每一行拆透：

**Step 1 — Project VGGT features** (公式 1):
$$\mathbf{F}_{\mathrm{geo}} = \mathbf{W}_{\mathrm{proj}}\mathbf{F}_{\mathrm{VGGT}}, \quad \mathbf{F}_{\mathrm{geo}} \in \mathbb{R}^{B \times N_{\mathrm{patches}} \times D}$$
- F_VGGT: VGGT 输出的 geometry-aware patch tokens, shape `B × N_patches × D_VGGT`
- W_proj: 可学习投影矩阵 `D × D_VGGT`, 把 VGGT 的 hidden dim 映射到 MLLM 的 hidden dim D
- N_patches: patch 数量 (与 VGGT 输入图像分辨率相关)
- D: MLLM hidden dim (例如 Qwen3-VL-4B 是 ~2560)

**Step 2 — Extract global semantic context** (公式 2):
$$\mathbf{s}_{\mathrm{global}} = \frac{1}{L}\sum_{i=1}^{L} \mathbf{H}_{\mathrm{MLLM}}[:, i, :] \in \mathbb{R}^{B \times 1 \times D}$$
- H_MLLM: MLLM 输出的 hidden states, shape `B × L × D`
- i 是 token 序列位置 (下标), L 是序列长度
- 这是 **mean pooling over token dimension**，得到一个 per-sample 的全局 semantic vector。它代表 "整个 prompt + 图像" 的语义摘要，包含 task intent (例如 "把胡萝卜放进碗里" vs "堆叠绿色方块")。

**Step 3 — Broadcast semantic context** to 每个 patch position:
$$\mathbf{S}_{\mathrm{broadcast}} = \mathrm{expand}(\mathbf{s}_{\mathrm{global}}, N_{\mathrm{patches}}) \in \mathbb{R}^{B \times N_{\mathrm{patches}} \times D}$$

**Step 4 — Position-specific gating** (公式 3):
$$\mathbf{g}_j = \sigma\big(\mathbf{W}_{\mathrm{gate}}[\mathbf{S}_{\mathrm{broadcast}}[:, j, :]; \mathbf{F}_{\mathrm{geo}}[:, j, :]]\big) \in \mathbb{R}^{B \times D}$$
- j 是 patch 位置下标 (从 1 到 N_patches)
- W_gate: `D × 2D` 的可学习矩阵, 输入是 concatenation `[semantic_context; geometric_feature]` 长度 2D
- σ: sigmoid, 把每个 channel 的 gate value 压到 (0,1)
- ⊙ 后面要做 element-wise, 所以 g_j 是 per-channel gate, 不是 scalar —— **每个 feature channel 都有自己的 blend ratio**

**Intuition**: gate 既看到 "这个任务需要什么 semantic" (S_broadcast), 又看到 "这个 patch 的几何信息有多 sharp" (F_geo)，从而决定这个位置偏 semantic 还是偏 geometric。例如 "把胡萝卜放进碗里" 的任务, gate 可能在胡萝卜轮廓处偏向 geometric (需要精确 depth), 在背景处偏向 semantic (识别哪个是碗)。

**Step 5 — Adaptive blend** (公式 4):
$$\mathbf{f}_{\mathrm{fused}, j} = \mathbf{g}_j \odot \mathbf{W}_s \mathbf{S}_{\mathrm{broadcast}}[:, j, :] + (1 - \mathbf{g}_j) \odot \mathbf{W}_g \mathbf{F}_{\mathrm{geo}}[:, j, :]$$
- W_s, W_g: 都是 `D × D` 的 projection, 给 semantic 和 geometric 各自一层 MLP
- ⊙: element-wise multiplication
- g_j 偏大 → 偏 semantic; g_j 偏小 → 偏 geometric
- 注意 **g_j 是 per-channel 不是 per-position scalar**, 所以同一位置不同 channel 可以同时混 semantic 和 geometric

**Step 6 — Concat with MLLM hidden states** (公式 5):
$$\mathbf{H}_{\mathrm{cond}} = [\mathbf{H}_{\mathrm{MLLM}}; \mathbf{F}_{\mathrm{fused}}] \in \mathbb{R}^{B \times (L + N_{\mathrm{patches}}) \times D}$$

这个 H_cond 喂给 DiT 当 cross-attention 的 K/V。

**为什么 GatedFusion work**:
1. MLLM 输出**保持不变**，不破坏 pretrained representation
2. 3D tokens 经过 **learnable gate 自适应缩放**, 不会喧宾夺主
3. Semantic context 作为 gating 的 condition, 让 geometric features "被语义引导" —— 这正是 Spatial Forcing / 3D-Tokens 试图通过 distillation 实现但对齐目标不够 explicit 的事情
4. Per-channel gate 让 model 学到 "在哪个 channel 上需要几何精度, 在哪个 channel 上需要语义 confidence"

### 3.6 3D-Tokens (special token alignment)

在 input sequence 末尾加一个 learnable special token `<|vggt|>`，用 alignment loss 把它的 hidden state 拉向 projected VGGT feature:
$$\mathcal{L}_{\mathrm{align}} = 1 - \cos(\mathbf{W}_{\mathrm{align}}\mathbf{h}_{\mathrm{vggt}}, \mathbf{W}_{\mathrm{proj}}\mathbf{f}_{\mathrm{VGGT}})$$
$$\mathcal{L} = \mathcal{L}_{\mathrm{action}} + \lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$$
- h_vggt: special token 经 MLLM 后的 hidden state
- f_VGGT: 对应 patch 的 VGGT feature
- λ_align: 平衡 action loss 和 alignment loss 的权重
- **推理时不需要 VGGT**, zero inference overhead
- **结果**：SIMPLER 56.25%, LIBERO 97.64%。这种 distillation 风格的提升有限 —— 因为 cosine alignment 只是 **把 hidden state 拉到一个方向**, 并不能保证 MLLM 的 attention 真的会用这个方向。

### 3.7 Middle Layer Injection

在 MLLM 的第 k 层 forward hook 注入：
$$\mathbf{H}_{\mathrm{out}}^{(k)} = \mathbf{H}^{(k)} + \alpha \cdot \mathrm{CrossAttn}(\mathrm{LN}(\mathbf{H}^{(k)}), \mathbf{F}_{\mathrm{geo}}, \mathbf{F}_{\mathrm{geo}})$$
- k: 注入的层索引 (下标)
- α: learnable scale
- LN: LayerNorm
- **结果**：SIMPLER 51.82%。中间层注入破坏 MLLM 内部 representation hierarchy，且 VGGT 的 geometric features 与 MLLM 中层 visual features 对齐不完美，造成 representation drift。

### 3.8 Spatial Forcing (Li et al. 2025)

训练时在 MLLM 中间层加 alignment loss:
$$\mathcal{L}_{\mathrm{align}} = -\frac{1}{N}\sum \cos(\mathrm{BN\text{-}MLP}(\mathbf{H}_{\mathrm{vis}}^{(k)}), \mathbf{F}_{\mathrm{VGGT}} + \mathbf{E}_{\mathrm{pos}})$$
- H_vis^(k): 第 k 层的 visual tokens
- BN-MLP: BatchNorm + MLP projection
- E_pos: position encoding 加到 VGGT feature 上 (因为 VGGT patch 没显式位置)
- **推理时 VGGT 完全丢弃**, zero overhead
- **结果**：SIMPLER 58.85%。比 3D-Tokens 略好但仍然不如 GatedFusion。**Intuition**: alignment loss 是 auxiliary supervision, 它只能影响训练梯度, 不能改变推理时 model 实际"如何使用"这些 representation 的方式 —— 你不能强迫一个 attention layer 用某个 channel。

### 3.9 Visual Fusion

在 MLLM 输入前, 让 2D visual tokens 作 query, 3D tokens 作 K/V 做 cross-attention:
$$\mathbf{T}_{\mathrm{2D}}' = \mathrm{LN}(\mathbf{T}_{\mathrm{2D}} + \mathrm{CrossAttn}(\mathbf{T}_{\mathrm{2D}}, \mathbf{T}_{\mathrm{3D}}, \mathbf{T}_{\mathrm{3D}}))$$
- **结果**：SIMPLER 4.69% (崩), LIBERO 73.40% (崩)。**这是 9 个 scheme 里最差的之一**。原因：直接替换 MLLM 的 2D visual tokens，破坏了 vision encoder 与 LLM 之间 alignment (CLIP-style contrastive alignment 经过大量 pretrain 调教)，inject 3D 几何信号反而抹平了 semantic features。

---

## 4. Pilot Study 结果表读法

| Method | SIMPLER Avg | LIBERO Avg | 解读 |
|---|---|---|---|
| Base (Qwen3-VL-4B) | 57.81 | 96.50 | baseline |
| **Gated Fusion** | **68.23 ↑10.42** | **98.05 ↑1.55** | winner |
| Concat Fusion | 60.42 ↑2.61 | 97.75 | 第二好, 简单 concat 也行 |
| Spatial Forcing | 58.85 | 97.72 | distillation 类, 有限 |
| 3D-Tokens | 56.25 | 97.64 | distillation, 略低于 base |
| Middle Layer | 51.82 | 97.82 | 中间层 inject 破坏 |
| CrossAttn Fusion | 56.25 | 83.45 | explicit cross-attn 抹平几何 |
| Early Fusion | 44.53 | 86.45 | MLLM input 被污染 |
| AE-Fusion | 3.13 | 97.40 | DiT 改造破坏 OOD |
| Visual Fusion | 4.69 | 73.40 | visual token 替换毁灭性 |

**关键 pattern**:
- LIBERO (in-domain) 几乎所有 method 都 > 95%, 说明 in-domain 学习 3D 不是必须的，2D features 加 finetune 足够。
- SIMPLER (OOD real2sim) 才是分水岭，**真正考验 spatial generalization**。能显著 > base 的只有 GatedFusion 和 ConcatFusion。其他方法要么持平要么崩。
- **任何"修改 MLLM 内部"的方法都失败** (Early Fusion, Middle Layer, Visual Fusion) —— 因为破坏了 pretrained 的 vision-language alignment。
- **distillation 类 (3D-Tokens, Spatial Forcing) 提升有限** —— 因为只能影响梯度, 不能保证推理时 attention pattern。
- **简单 Concat 比 CrossAttn 好** —— 显式 cross-attention 让 geometric feature 失去 sharpness。

---

## 5. 3D-MIX = GatedFusion 工程化

3D-MIX 把 GatedFusion 包装成 plug-and-play，只需要两个输入：
1. MLLM hidden states H_MLLM
2. VGGT-extracted F_VGGT

不修改 MLLM 或 action expert 的 source code。

### 5.1 GR00T-style integration

很简单：在 MLLM 和 DiT 之间塞 3D-MIX，DiT 的 cross-attention 从 H 变成 H_cond = [H_MLLM; F_fused]：

$$\mathbf{A}_\tau = \mathrm{DiT}(\mathbf{A}_{\tau-1}, \mathbf{H}_{\mathrm{cond}}, \tau)$$

- A_τ: diffusion step τ 时的 noisy action chunk
- τ: flow-matching timestep (下标)
- H_cond: enriched conditioning (含 fused 3D)

### 5.2 π-style integration (layer-wise)

每个 DiT layer 用一份独立的 gating 参数。给定多层 MLLM hidden states {H_MLLM^(1), ..., H_MLLM^(N)}:

公式 (7): 每层各自算 semantic context
$$\mathbf{s}_{\mathrm{global}}^{(i)} = \frac{1}{L}\sum_{j=1}^{L} \mathbf{H}_{\mathrm{MLLM}}^{(i)}[:, j, :]$$

公式 (8): 每层独立 gating
$$\mathbf{F}_{\mathrm{fused}}^{(i)} = \mathrm{GatedFusion}(\mathbf{s}_{\mathrm{global}}^{(i)}, \mathbf{F}_{\mathrm{geo}})$$

公式 (9): 每层 enriched conditioning
$$\mathbf{H}_{\mathrm{cond}}^{(i)} = [\mathbf{H}_{\mathrm{MLLM}}^{(i)}; \mathbf{F}_{\mathrm{fused}}^{(i)}]$$

公式 (10): DiT layer-wise cross-attention
$$\mathbf{Z}^{(i)} = \mathrm{TransformerBlock}^{(i)}(\mathbf{Z}^{(i-1)}, \mathbf{H}_{\mathrm{cond}}^{(i)})$$

**Intuition**: 浅层 DiT 的 gate 可能更偏 semantic (低层抽象), 深层 DiT 的 gate 可能更偏 geometric (空间精度)。每个 layer 自己学 fusion ratio。

---

## 6. 实验：跨 backbone 的一致提升

### 6.1 GR00T-style (Table 2)

跨 6 个 MLLM 系列 9 个变体 (2B-8B)，所有都提升:

| Backbone | Base SIMPLER | +3D-MIX | Δ |
|---|---|---|---|
| Qwen3-VL-4B | 57.81 | 68.23 | +10.42 |
| Qwen3-VL-8B | 58.40 | 63.27 | +4.87 |
| Qwen2.5-VL-3B | 45.05 | 47.92 | +2.87 |
| Qwen2.5-VL-7B | 47.14 | 49.22 | +2.08 (paper 列 +1.83) |
| RoboBrain2.0-7B | 37.83 | 49.22 | +11.39 |
| RynnBrain-8B | 52.60 | 65.11 | +12.51 |
| RynnBrain-2B | 48.18 | 56.25 | +8.07 |
| MimoEmbodied-7B | 52.87 | 63.28 | +10.41 |
| RoboBrain2.5-8B | 64.58 | 65.63 | +1.05 |
| RoboBrain2.5-4B | 56.25 | 63.02 | +6.77 |

**平均 +7.0% on SIMPLER**。LIBERO 基本都 +0.3 到 +1.5 的小提升 (因为 LIBERO in-domain 已 96-98%, 接近天花板)。

**Pattern**:
- Base 越弱的 backbone 提升越大 (RoboBrain2.0-7B base 37.83, +11.39)
- Base 已经很强的 backbone (RoboBrain2.5-8B base 64.58) 提升变小 (+1.05), 说明 3D 信息有 marginal ceiling
- 模型 size scale 不显著影响 3D-MIX 的提升幅度, 7B 和 2B 都能受益

### 6.2 π-style (Table 3)

5 个变体也一致提升，但幅度比 GR00T-style 小 (RoboBrain2.5-4B +2.60, Qwen3-VL-4B +6.77, Qwen3-VL-2B +5.38)。

**Intuition**: π-style 本身 layer-wise coupling 已经把 semantic 信息层层喂给 DiT, 它的 base spatial grounding 比 GR00T-style 强, 3D-MIX 能补充的边际就小。但 layer-wise fusion 仍然 work, 说明 3D 信息在多个抽象层级都有价值。

---

## 7. Ablation Studies 关键洞察

### 7.1 VGGT Frozen vs. Trainable (Figure 3a)

冻结 VGGT 性能 ≈ 或 > fine-tune。**Intuition**: VGGT 在大规模 3D 数据 (megadepth, co3d, blendmvs 等) 上 pretrain, 已经学到 generalizable geometry representation; finetune 在小规模 robot data 上容易 overfit, 损失泛化能力。这与 CLIP visual encoder 在 robotic manipulation 中通常 frozen 的发现一致 [[6](https://github.com/openvla/openvla)]。

### 7.2 3D Information Sensitivity (Figure 3b)

推理时把 VGGT features 替换成:
1. **Zero vectors**: 性能显著下降
2. **Random Gaussian noise**: 性能也下降, 甚至更剧烈

**关键 takeaway**: 3D-MIX 的提升不是来自"多了 N_patches 个 token 维度", 而是来自**真正的几何信息**。如果是 dimension 增加的副作用, zero 替换不会掉这么多。这验证了 3D 几何信号的 causal role。

### 7.3 Sparse Layer Fusion (π-style, Figure 3c)

让 3D feature 每 k 层注入一次 (k=0 是 full layer-wise), 发现 sparse (例如 k=2 或 k=3) 性能与 full 相当, 显存显著降低。

**Intuition**: 相邻 DiT layer 的 abstraction level 接近, 不需要每层都重新 fuse 3D。这是 practical deployment tip —— π-style 上用 sparse 3D-MIX 几乎不牺牲性能, 显存节省接近 1/k。

---

## 8. 与相关工作的 connection & 我的直觉

### 8.1 VGGT 的位置类比

VGGT [[4](https://vggt.github.io/)] 对 3D 几何的意义, 类似 CLIP visual encoder 对 2D semantic 的意义 —— 一个大规模 pretrain 的 foundation encoder, downstream task 冻结使用就够。所以 paper 选择 frozen VGGT 是合理的, 也呼应了 3D-MIX 的轻量定位。

### 8.2 与 Spatial Forcing, 3D-Tokens 的对比

这两类做法本质是 **representation distillation** —— 训练时强迫 MLLM 内部某层 align 到 VGGT feature, 推理时丢弃 VGGT。理论吸引力: zero inference overhead。但 paper 的实验表明, **这种 distillation 的上限有限**。

我的解读: cosine alignment loss 是一个 **弱 supervision** —— 它只要求方向一致, 不要求 magnitude, 更不要求下游 attention 真的会 attend 到这个方向。而 GatedFusion 是 **显式提供 3D tokens 给 action expert cross-attention**, 让 action expert 自己决定如何使用, supervision signal 强很多。这就解释了为什么 GatedFusion 显著优于 distillation 类方法。

### 8.3 与 OpenVLA-OFT, π₀.₅ 中的 3D 探索

OpenVLA 系列 [[1](https://openvla.github.io/)] 后续工作 OpenVLA-OFT 也在探索 action chunking + proprioception; π₀.₅ 引入了 web data pretraining 但 3D grounding 仍弱。3D-MIX 这类工作提供的是 **module-level 3D injection**, 与这些 foundation model 的 training scheme 正交, 可以叠加。

### 8.4 与 Cambrian-S, SpatialTree 等 spatial MLLM 工作

最近 spatial-aware MLLM (Cambrian-S [[7](https://arxiv.org/abs/2511.04670)], SpatialTree [[8](https://arxiv.org/abs/2512.20617)]) 试图在 MLLM 内部增加 depth-aware pretraining。这条路长远看好, 但需要海量 3D-aligned image-text data, 短期内用 frozen VGGT 做 plug-and-play 仍是 pragmatic 的最优解。

### 8.5 gating 机制的更深直觉

公式 (3) 的 gate 既看 semantic context 又看 geometric feature, 这其实是 **FiLM-style conditioning** [[9](https://arxiv.org/abs/1709.07871)] 的变体 —— 用一个 condition vector (semantic) 调制另一个 feature (geometric)。但 3D-MIX 的特殊之处在于 **condition 也参与 blend** (g_j ⊙ W_s ⊙ S_broadcast + (1-g_j) ⊙ W_g ⊙ F_geo), 这其实是一种 **learnable residual 1x1 conv** 的离散化版本 —— 不同 channel 选择不同 source。

这种 per-channel gating 比 scalar gating 强很多。scalar gating 只能决定整张图偏 semantic 还是偏 geometric; per-channel 可以让 foreground channel 偏 geometric, background channel 偏 semantic, 这正是 manipulation 需要的。

### 8.6 关于 OOD vs. in-domain 的对比

LIBERO 提升 0.3-1.5%, SIMPLER 提升 1-12%。这个 gap 是 paper 最重要的发现之一 —— **3D 信息的主要价值在 OOD generalization**。

in-domain 时, 2D features 已经能记住训练 distribution 的 spatial pattern, 3D 只是 redundant。OOD 时, 2D features 失去 memorization 支撑, 只有 true geometric understanding 才能泛化。这与 "LLM memorization vs. generalization" 的争论异曲同工 —— 3D 信息相当于一种 **inductive bias**, 在 distribution shift 下提供额外的 generalization lever。

### 8.7 与 RT-2, Octo 等早期 VLA 的对照

早期 VLA (RT-2 [[10](https://robotics-transformer2.github.io/)], Octo [[11](https://octo-models.github.io/)]) 完全无 3D, 仅依赖 2D + action chunk finetune, OOD 性能差是公认的。3D-MIX 提供了一种 **non-invasive 升级路径** —— 不重新 train MLLM, 只 plug 一个 module, 就能 OOD 平均 +7%。这对工业部署是巨大的 selling point, 因为 MLLM 训一次成本是天文数字。

---

## 9. 局限与可能的下一步

paper 自己没展开谈, 但我看到的几个点:

1. **Pilot study 只在 Qwen3-VL-4B 上做**: 9 种 fusion 选 GatedFusion 是基于 4B 的结果, 大模型 (8B+) 的最优 fusion 不一定相同。大模型本身的 spatial reasoning 可能已经更强, 3D injection 的 best practice 可能变。
2. **VGGT 本身的局限**: VGGT 处理静态 frame, 没有 temporal geometry。manipulation 中 object motion 的预测 (例如抛物线、滑动) 需要时序 3D, VGGT 无法提供。结合 VGGT + video geometry model (DUSt3R [[12](https://dust3r.eu/)] 的 video extension) 可能是下一步。
3. **Single fixed VGGT feature**: paper 只用 VGGT 最后层输出。VGGT 内部多层 feature (类似 ViT 的中间层) 可能携带不同粒度 geometry, 可以做 multi-scale fusion。
4. **GatedFusion 没在 MLLM input 端做**: pilot study 排除了 input-level fusion 因为破坏 alignment, 但用 gated adapter (类似 (IA)³ [[13](https://arxiv.org/abs/2208.03362)]) 可以最小 invasive 地注入。可以再试一波。
5. **缺少 3D-MIX 的 negative result analysis**: 例如, 如果 VGGT 输入是 single-view (multi-view 不可用时), 3D-MIX 还能提升多少? VGGT single-view 的几何信号本身就弱很多。
6. **Action expert 改造可能性**: pilot 中 AE-Fusion 崩了, 但那是因为直接改造 DiT。如果用 **MoE-style dual expert** (一个 semantic-conditioned DiT, 一个 geometric-conditioned DiT, router 选择), 可能 work, 这是 GR00T-MoE [[3](https://research.nvidia.com/labs/gear/gr00t-n1_6/)] 的方向。

---

## 10. Takeaways (build your intuition 的几句话)

- **3D 不能乱塞**。任何破坏 pretrained MLLM vision-language alignment 的方法都失败 (Early Fusion, Visual Fusion)。
- **distillation 类弱**。Alignment loss 只能调梯度, 不能保证 attention pattern。显式 token-level fusion 远胜 distillation。
- **gate 比 cross-attention 好**。Cross-attention 让 geometric features 失去 sharpness, 而 gate 保留两条独立通路, 用 channel-wise blend 选择性使用。
- **injection point 比 mechanism 重要**。在 MLLM 与 action expert 之间是 sweet spot —— 不破坏上游, 给下游足够 conditioning 信息。
- **3D 的价值在 OOD**: in-domain 时 3D 是 redundant, OOD 时是 generalization lever。
- **per-channel gating 是关键**: scalar gating 不够, 要让每个 feature channel 独立选择 semantic 或 geometric source。

---

## References (Web Links)

- [1] OpenVLA: https://openvla.github.io/
- [2] π₀ / π₀.₅: https://www.physicalintelligence.ai/blog/pi0
- [3] GR00T-N1.6: https://research.nvidia.com/labs/gear/gr00t-n1_6/
- [4] VGGT (CVPR 2025): https://vggt.github.io/
- [5] DiT (Peebles & Xie, ICCV 2023): https://www.wpeebles.com/DiT
- [6] OpenVLA GitHub: https://github.com/openvla/openvla
- [7] Cambrian-S (arXiv 2511.04670): https://arxiv.org/abs/2511.04670
- [8] SpatialTree (arXiv 2512.20617): https://arxiv.org/abs/2512.20617
- [9] FiLM (arXiv 1709.07871): https://arxiv.org/abs/1709.07871
- [10] RT-2: https://robotics-transformer2.github.io/
- [11] Octo: https://octo-models.github.io/
- [12] DUSt3R: https://dust3r.eu/
- [13] (IA)³: https://arxiv.org/abs/2208.03362
- [14] SIMPLER: https://simpler-env.github.io/
- [15] LIBERO: https://libero-project.github.io/
- [16] Open X-Embodiment: https://robotics-transformer-x.github.io/
- [17] DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- [18] AdamW: https://arxiv.org/abs/1711.05101
- [19] Spatial Forcing (Li et al. 2025, arXiv 2510.12276): https://arxiv.org/abs/2510.12276
- [20] 3D-MIX GitHub (paper §): https://github.com/ZGC-EmbodyAI/3DMix-for-VLA

---

最后一句: 这篇 paper 的核心 contribution 不是 propose 一个新 module, 而是 **systematic empirical comparison** 把社区从"每个人 invent 一个 fusion"的无序状态拉回到"用 GatedFusion 这个 baseline"。这种 pilot study 风格的 paper 在 VLA 这个快速膨胀的领域非常稀缺, 工程价值高, 我个人很喜欢这种 "let the experiments speak" 的风格。后续工作如果要 propose 新 fusion scheme, 这篇的 9 个 baseline + 2 个 benchmark 是必须 compare 的。
