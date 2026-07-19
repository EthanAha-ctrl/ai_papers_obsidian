---
source_pdf: Afford-VLA Action-Aligned Visual Planning via.pdf
paper_sha256: 5ad8c3a749b409a82dc480dadeb8bb0e655296da4cda668739fb7b18dfd432ba
processed_at: '2026-07-18T03:01:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Afford-VLA 深度讲解：将 Affordance 内化为 VLA 的视觉规划接口

## 1. 大背景：VLA 的 spatial reasoning 短板

现在 VLA 这条线发展非常快，从 RT-1、RT-2 到 OpenVLA、π₀、GR00T，action decoder 的形式越来越多样化——autoregressive tokenization (FAST)、continuous regression (OFT)、flow-matching/diffusion (π₀, GR00T)。但所有这些工作都共享一个 fundamental 的问题：**action head 需要从一个 global 的 vision-language representation 里隐式地推断 "where to interact"**。

这件事在简单任务上还能 work，但一旦场景复杂、object 多、需要 precise spatial grounding，VLA 就会暴露出 spatial reasoning 不足的毛病。比如 LIBERO-Spatial 这种要求把某个物体放到某个特定位置的任务，纯靠 VLM 的 global representation 去推断 interaction region，效果就有限。

作者把这个问题抽象成 **visual planning** —— 即在视觉空间里告诉模型 "where to interact"。这是这篇论文的核心 framing。

## 2. 既有 visual planning 方法的分类与不足

论文 Section 2 把现有方法分成三类，这个 taxonomy 非常清晰：

**(a) Geometry-based methods** (PointVLA, DepthVLA, 4D-VLA)
- 用 point cloud、depth、multi-view 这类 3D 信号 anchor policy
- 问题：提供的是 global、scene-level 的 geometric context，没有显式给出 task-conditioned 的 interaction region

**(b) Symbolic-based methods** (MOKA, KITE, RoboPoint)
- 把视觉 grounding 成中间符号表示（language description、keypoint tokens）
- 问题：indirect，符号化抽象损失了 visual evidence

**(c) Visually grounded methods** (Transporter, CoT-VLA, TraceVLA, CoA-VLA, RT-Affordance, AffordDP, GLOVER++)
- 用 sparse cues（point、bbox、trajectory）或者 dense cues（mask、heatmap）
- 其中 affordance 最相关
- 问题：这些 affordance pipeline 通常和 action learning 是 decoupled 的——要么是 external perception module 生成的，要么作为独立的 supervised target，没有从 action prediction 那里拿到 feedback

## 3. 作者的核心论点：四条性质

作者提出 effective visual planning 应该满足四个性质，这是整篇论文的 conceptual foundation：

1. **Local** —— 聚焦 task-relevant interaction region，而不是 global scene
2. **Visually grounded** —— 直接 tied to visual evidence，不经过 symbolic abstraction
3. **Internally generated** —— 在 VLA model 内部生成，不是 external module
4. **Action-aligned** —— 直接被 action model 消费，影响 downstream decision

这四条性质合起来定义了一种 "perception-action bridge"。Afford-VLA 的所有设计都是围绕同时满足这四条性质来构造的。这个 framing 让我想到当年 NeRF 之于 view synthesis——一旦定义清楚 "what we want"，剩下的就是 engineering。

## 4. 核心方法：Affordance as Internal Visual Planning Interface

### 4.1 形式化定义

Standard VLA 的 policy：

$$
p(\mathbf{a}_{t:t+H} \mid I_t, x, s_t)
$$

这里 $I_t$ 是 RGB observation，$x$ 是 language instruction，$s_t$ 是 proprioceptive state，$\mathbf{a}_{t:t+H}$ 是未来 $H$ 步的 action chunk。注意 $H$ 是 chunk length，本论文 $H=8$。

Afford-VLA 引入一个 task-conditioned visual focus variable $M_t \in [0,1]^{H_I \times W_I}$，表示 affordance region。新 policy：

$$
p(\mathbf{a}_{t:t+H} \mid I_t, x, s_t, M_t)
$$

关键 insight：$M_t$ 不作为 standalone output，而是作为 **internal intermediate representation** 同时 bridge 视觉 grounding 和 control。这一点很重要——它不是输出可视化用的，是真正进入 action 计算路径的。

### 4.2 <AFF> Token 的设计

这是论文最巧妙的设计之一。给定图像 $I_t$ 和语言 $x$，VLM 把它们编码成 image tokens $Q_{\text{img}}$ 和 language tokens $Q_{\text{text}}$。然后 augment 一组 learnable affordance query tokens $Q_{\text{aff}} \in \mathbb{R}^{K_{\text{aff}} \times C_{\text{llm}}}$，其中 $K_{\text{aff}}$ 是 query 数量（论文里 $K_{\text{aff}}=4$），$C_{\text{llm}}$ 是 VLM hidden dimension。

Augmented sequence 经过同一个 VLM backbone：

$$
[H_t, A_t] = f_{\text{VLM}}([Q_{\text{img}}, Q_{\text{text}}, Q_{\text{aff}}])
$$

这里：
- $H_t$：原 image-language tokens 的 contextualized hidden states
- $A_t \in \mathbb{R}^{K_{\text{aff}} \times C_{\text{llm}}}$：在 <AFF> 位置的 hidden states

因为 <AFF> tokens 参与 self-attention，它们的 final state 同时 conditioned on 视觉和语言——这就 encode 了 "在当前 instruction 下应该 search 什么 interaction evidence"。

这个设计让我想到 DETR 的 object queries。本质上是把 "task-conditioned region localization" 这个 problem 转化成 "learnable query tokens + cross-attention" 这个 well-studied pattern。VLM 的 transformer backbone 天然支持这个 pattern，几乎 zero additional architectural cost。

### 4.3 Affordance Head：从 Query 到 Patch-level Mask

光有 query hidden state 还不够，要 ground 回 image space。作者从 vision encoder 里取出 patch-aligned visual features $P_t \in \mathbb{R}^{N \times C_{\text{vis}}}$，其中 $N = H_p W_p$ 是 patch 数（$16 \times 16 = 256$），$C_{\text{vis}}$ 是视觉 feature 维度。

然后一个 lightweight affordance head $\mathcal{D}_{\text{aff}}$ 把 $A_t$ 和 $P_t$ 耦合，输出 patch-level affordance logits：

$$
G_t = \mathcal{D}_{\text{aff}}(A_t, P_t), \quad G_t \in \mathbb{R}^{H_p \times W_p}
$$

实现上是一个 two-way attention decoder，2 层、8 heads、hidden 256。直觉上：
- $A_t$ encode "what to search"（task-conditioned interaction evidence）
- $P_t$ encode "where it appears"（dense spatial features）
- Decoder couples them，给每个 patch 一个 affordance logit

这是一个典型的 query-patch grounding decoder pattern，类似 SAM 的 mask decoder 或者 Mask2Former 的 query-to-mask 路径。

### 4.4 Mask Pooling：从 Mask 到 Action-Consumable Embedding

这一步是真正实现 "action-aligned" 的关键。把 $G_t$ flatten 成 $g_t \in \mathbb{R}^N$，做 hard Top-K：

$$
m_{t,i} = \mathbb{I}[i \in \text{TopK}(g_t, k)], \quad i = 1, \dots, N
$$

这里 $k=16$，即选 top 16 个 patch（在 $16 \times 16 = 256$ 个 patch 里）。然后做 mask pooling：

$$
r_t = W_{\text{aff}} \left( \frac{1}{k} \sum_{i=1}^{N} m_{t,i} P_{t,i} \right), \quad r_t \in \mathbb{R}^{C_{\text{llm}}}
$$

变量解释：
- $m_{t,i} \in \{0, 1\}$：第 $i$ 个 patch 是否被选中
- $P_{t,i} \in \mathbb{R}^{C_{\text{vis}}}$：第 $i$ 个 patch 的视觉 feature
- $W_{\text{aff}}$：从 $C_{\text{vis}}$ 投影到 $C_{\text{llm}}$ 的矩阵
- $r_t$：affordance embedding，是一个 compact 的、聚焦在 interaction region 的视觉特征

这个 $r_t$ 就是 "action-aligned visual planning" 的载体。它直接进入 action head 的 conditioning sequence：

$$
Z_t = [H_t; r_t]
$$

然后 action prediction：

$$
\hat{\mathbf{a}}_{t:t+H} = f_{\text{act}}(Z_t, s_t)
$$

这里 $f_{\text{act}}$ 用的是 GR00T-style flow-matching action head（DiT-B backbone，16 layers，hidden 1024，预测 7-DoF delta joint position，chunk length 8，inference 4 denoising steps）。

直觉上：action head 现在不再需要从 global VLM representation 里 implicit 地推断 "where to act"，而是直接拿到一个 localized、task-relevant 的 visual summary。这就把 "where" 这件事从 implicit 推理变成了 explicit conditioning。

### 4.5 Multi-view 扩展

单 view 公式自然扩展到 multi-view。给定 $\mathcal{T}_t = \{I_t^v\}_{v=1}^V$，每个 view 独立做 affordance generation 和 pooling，得到 view-specific mask $\hat{M}_t^v$ 和 embedding $r_t^v$。然后 action head 的 conditioning：

$$
Z_t = [H_t; r_t^1; \dots; r_t^V]
$$

实验里用了 wrist camera + third-person camera 两个 view，每个 view 4 个 <AFF> query。

## 5. Action-Aligned Training：让梯度从 Action 流回 Affordance

### 5.1 Straight-Through Estimator

这里有一个非常关键的 technical challenge：hard Top-K 是 non-differentiable 的，意味着 action loss 没法通过 mask pooling 反传到 affordance head。如果 affordance head 只能拿到 dense mask supervision，它就和 action prediction decoupled 了。

作者的解法是 straight-through estimator：

$$
r_t = \Phi_{\text{ST}}(P_t, G_t)
$$

- Forward pass：和 hard Top-K mask pooling 完全一样，从 sparse 16 个 patch 聚合
- Backward pass：把 non-differentiable selection 替换成 soft surrogate（softmax with temperature $\tau = 1.0$），让 action loss 能 update $G_t$ 和 affordance head

这是一个非常 elegant 的设计——保留了 hard selection 的稀疏性和 locality（forward），同时获得了 differentiability（backward）。

### 5.2 训练目标

两个 loss：

$$
\mathcal{L}_{\text{aff}} = \text{BCEWithLogits}(G_t, Y_t)
$$

$G_t$ 是 predicted patch-level logits，$Y_t$ 是 ground-truth affordance mask。这里 BCEWithLogits 是标准 binary cross entropy 加 sigmoid。

$$
\mathcal{L}_{\text{act}} = \ell_{\text{FM}}(f_{\text{act}}(Z_t, s_t), \mathbf{a}_{t:t+H})
$$

$\ell_{\text{FM}}$ 是 flow-matching objective，来自 GR00T。直觉上 flow-matching 是 learn 一个 conditional vector field，把 action 从 noise distribution 流到 data distribution。

Joint loss：

$$
\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{aff}}
$$

通过 straight-through mask pooling，$\mathcal{L}_{\text{act}}$ 不仅优化 action prediction，还 feedback 到 affordance pathway。这就实现了 "action-aligned" 这个性质。

### 5.3 两阶段训练策略

这个设计是为了 stability：

**Stage 1（warmup, 4K steps）**：只用 $\mathcal{L}_{\text{aff}}$ 训练 affordance head，freeze VLM 和 action head。这一步给 affordance head 一个稳定的 spatial grounding 初始化。GT mask 用于 pooling。

**Stage 2（joint, 140K steps on LIBERO, 200K on SimplerEnv）**：用 predicted mask 做 pooling（through $\Phi_{\text{ST}}$），action head 用 resulting affordance embedding 训练。Joint loss $\mathcal{L}_{\text{joint}}$ 同时优化。GT mask 只用于 affordance loss，不参与 embedding 构造。

这个设计的 motivation 是 reduce train-inference mismatch：测试时没有 GT mask，所以训练时 action head 应该见到的也是 predicted mask pooling 出来的 embedding。

## 6. 实验结果分析

### 6.1 LIBERO Benchmark

LIBERO 四个 suite：
- Spatial：测 spatial reasoning
- Object：测 object-level generalization
- Goal：测 goal variation
- Long：测 long-horizon execution

主要数据（Table 1）：

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 95.2 | 94.2 | 95.2 | 93.2 | 94.5 |
| GR00T-N1.5 | 96.5 | 98.5 | 91.0 | 91.5 | 94.4 |
| DepthVLA | 96.4 | 98.0 | 95.8 | 89.2 | 94.9 |
| **Afford-VLA** | **97.8** | **99.6** | **97.6** | **94.6** | **97.4** |

几个直觉：
1. Afford-VLA 在 Spatial 上 97.8%，Long 上 94.6%，比 π₀ 的 85.2% 高出 9.4 个点。这正好印证了 spatial reasoning 是 Afford-VLA 的设计目标
2. Object 上达到 99.6%，几乎 saturation
3. 对比 CoA-VLA（affordance chain-of-thought 风格）只有 79.8%，说明 internal + action-aligned 比 external affordance reasoning 强很多

### 6.2 SimplerEnv（real-to-sim）

SimplerEnv 是 Bridge/WidowX 数据训练的，测 visual & spatial generalization。

| Method | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Avg |
|--------|----------|-----------|-------------|--------------|-----|
| π₀ | 29.2 | 62.5 | 29.2 | 91.6 | 53.1 |
| π₀.5 | 49.3 | 64.7 | 44.7 | 69.7 | 57.1 |
| GR00T-N1.6-Bridge | 64.5 | 65.5 | 5.5 | 93.0 | 57.1 |
| **Afford-VLA** | **66.6** | 54.2 | 14.6 | **96.8** | **58.1** |

注意 Stack Block 上 Afford-VLA 只有 14.6%，作者分析这是因为这类任务更依赖 trajectory dynamics 和 long-horizon coordination，localized affordance guidance 帮助有限。这个 limitation 非常 honest。

### 6.3 LIBERO-Plus（zero-shot robustness）

这个 benchmark 用 7 种 perturbation 测 robustness，10030 个 task。训练只在 LIBERO 上做，直接 zero-shot 评估。

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|--------|--------|-------|----------|-------|------------|-------|--------|-------|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| **Afford-VLA** | 56.0 | **56.8** | **91.5** | **96.8** | **97.0** | **80.9** | **78.9** | **78.1** |

特别值得注意的是：
- Layout 78.9%：layout 改变后能 re-localize interaction region
- Background 97.0%：texture 改变对 affordance 影响很小，因为 affordance 聚焦 task-relevant region，自然 ignore 了 background
- Noise 80.9%：image-level corruption 下仍稳定，因为 mask pooling 压缩了 task-relevant evidence

这些结果直接验证了 internal affordance 提供了 robust perception-action interface——只要 task intention 不变，affordance 就能 re-ground 到正确的 interaction region。

### 6.4 Real-World Experiments

两个 task：Cup-to-Plate 和 Fork-in-Bowl，每个 20 trials。
- Afford-VLA: 80% / 70%
- π₀: 较低
- OpenVLA-OFT: 较低

虽然 sample size 小，但 trend 清晰。

## 7. Ablation Studies 深度解析

### 7.1 Affordance Integration Strategies（核心 ablation）

Table 3 是论文最核心的 ablation：

| Integration | Internalized | Action-Aligned | LIBERO |
|-------------|--------------|----------------|--------|
| (a) Baseline | ✗ | ✗ | 95.4 |
| (b) External Affordance w/ Action Condition | ✗ | ✓ | 96.5 |
| (c) Internal Affordance w/o Action Condition | ✓ | ✗ | 95.9 |
| (d) Full Afford-VLA | ✓ | ✓ | 97.4 |

直觉解读：
- (a) → (b): +1.1，说明 explicit localization cues 有帮助
- (a) → (c): +0.5，说明 internal affordance 即使不直接 condition action 也能 improve spatial grounding（通过 shared backbone）
- (b) → (d): +0.9，说明 internal 比 external 强
- (c) → (d): +1.5，说明 action-aligned 比 auxiliary supervision 强
- (a) → (d): +2.0，overall gain

这个 ablation 严格 validate 了四条性质里 internal + action-aligned 的必要性。

### 7.2 Mask Pooling Design（Table 6）

| Pooling | Differentiable | Localized | LIBERO |
|---------|----------------|-----------|--------|
| Hard Region Pooling | ✗ | ✓ | 91.3 |
| Dense Soft Mask Pooling | ✓ | ✗ | 96.0 |
| Sparse Top-K ST Patch Pooling | ✓ | ✓ | 97.4 |

非常 clean 的三选一：
- Hard Region Pooling 不可微，action loss 传不回 affordance head → 91.3%
- Dense Soft Mask Pooling 可微但 dilute affordance feature（平均了 background）→ 96.0%
- Sparse Top-K ST Patch Pooling 鱼和熊掌兼得 → 97.4%

这个 ablation 把 "differentiable + localized" 这两个看似矛盾的要求通过 straight-through estimator 完美 reconcile。

### 7.3 Training Strategy（Table 7）

| Strategy | Cond. Reliability | Action-to-Mask Feedback | Train-Infer Alignment | LIBERO |
|----------|-------------------|-------------------------|----------------------|-------|
| One-Stage w/ GT Pooling | High | ✗ | ✗ | 94.4 |
| One-Stage w/ Predicted Pooling | Low | ✓ | ✓ | 96.4 |
| Two-Stage Warmup + Predicted | High | ✓ | ✓ | 97.4 |

直觉：
- One-stage GT pooling：affordance 还没学好，action head 见到的是 GT embedding，train-infer mismatch 严重 → 94.4%
- One-stage predicted：no warmup，初期 affordance 很 noisy，action head 训练不稳定 → 96.4%
- Two-stage：先 warmup 让 affordance 学好，再让 action head 适应 predicted embedding → 97.4%

这个 ablation 揭示了一个 deep trade-off：GT signal 提供稳定 conditioning 但引入 train-infer gap；predicted signal 提供 alignment 但初期 noisy。两阶段是 sweet spot。

### 7.4 更细的 Integration Ablation（Table 8）

| Integration | Explicit Aff. | Internal Head | Differentiable | LIBERO |
|-------------|---------------|---------------|----------------|--------|
| (a) Baseline | ✗ | ✗ | ✗ | 95.4 |
| (b) Implicit Internal | ✗ | ✓ | ✗ | 95.9 |
| (c) Explicit External | ✓ | ✗ | ✗ | 96.5 |
| (d) Explicit Internal + Hard Pooling | ✓ | ✓ | ✗ | 91.3 |
| (e) Full | ✓ | ✓ | ✓ | 97.4 |

最 striking 的：(d) 比 (a) 还差！这是因为 hard pooling 不可微，affordance head 变成了 brittle bottleneck——action head 完全依赖一个没法被 action loss 优化的 mask。这反向验证了 differentiable pathway 的关键性。

## 8. Implementation 细节与联想

### 8.1 关键 hyperparameters（Table 4）

- VLM: Qwen3-VL-4B-Instruct
- Action head: GR00T-style flow-matching DiT-B, 16 layers, hidden 1024
- Action dim: 7-DoF delta joint position, chunk 8, 4 inference steps
- AFF queries: 4 per view, max 2 views
- Affordance decoder: 2 layers, 8 heads, hidden 256
- Patch grid: 16×16
- Top-K: 16 patches
- ST temperature: τ=1.0
- Warmup: 4K steps; Joint: 140K (LIBERO) / 200K (SimplerEnv)
- Image resolution: 224×224
- Batch size: 16 per GPU
- Optimizer: AdamW, base LR 2.5e-5, VLM LR 1e-5, action head LR 1e-4
- LR schedule: cosine with min LR 1e-6
- Distributed: DeepSpeed ZeRO-2

### 8.2 Affordance 监督数据的构造

这是 appendix B 的细节，很关键。作者用 RAGNet（一个 affordance segmentation model）offline 给 LeRobot-format 数据集的每一帧、每个 view 生成 affordance mask。这些 mask path 写回 parquet metadata，训练时 dataloader 按需读取。

这意味着：
- 不需要 manual mask annotation
- 但有 dependency on RAGNet 的质量（作者在 limitation 里也承认这点）
- Future work 应该探索 self-supervised affordance，或者让 action loss 完全 drive affordance 学习（去掉 $\mathcal{L}_{\text{aff}}$）

## 9. 我的直觉与联想

### 9.1 这篇论文在 VLA 演化里的位置

VLA 这条线一直在解决 "how to inject inductive bias" 的问题：
- RT-2 把 VLM 知识 inject 进 action
- FAST 用 DCT tokenization inject efficient action representation
- OpenVLA-OFT 用 parallel regression inject speed
- DepthVLA/SpatialVLA 用 3D/spatial token inject spatial awareness
- Afford-VLA 用 affordance mask inject "where to interact"

Afford-VLA 的独特之处在于：它 inject 的不是 input feature，而是 **explicit intermediate representation**。这件事让我想起 chain-of-thought——只不过 CoT 是 symbolic，affordance 是 visual & dense。

### 9.2 与 DETR-style Architecture 的联系

<AFF> tokens + affordance decoder 的设计本质上是 DETR 的 object query + mask head pattern 移植到 VLA。这个 pattern 在 detection/segmentation 里已经 mature，移植到 VLA 几乎 zero-cost。这暗示了一件事：**VLA 可以吸收很多来自 vision community 的成熟 pattern**，比如 query-based detection、mask segmentation、dense prediction。Afford-VLA 是这条 path 的 early example。

### 9.3 Straight-Through Estimator 的更广意义

STE 在 quantization、categorical reparameterization、discrete VAE 里都用过。Afford-VLA 把它用在 mask pooling 上，是一个很 elegant 的应用。

更广义地想：VLA 里很多 operation 都是非 differentiable 的（action quantization、token sampling、hard attention）。STE 提供了一个通用 pattern 来 bridge differentiable training 和 non-differentiable inference。这件事我觉得是未来 VLA scaling 的一个关键 trick。

### 9.4 Train-Infer Mismatch 这个 General Problem

Table 7 揭示的 train-infer mismatch 是 RL/imitation learning 里非常 general 的问题。Afford-VLA 的 two-stage 解法让我想到 DAgger、Scheduled Sampling、Professor Forcing 这些 work。本质上都是 "训练时用 GT，测试时用 predicted" 的 gap。Afford-VLA 的解法是 warmup + predicted-pooling，这是一种 scheduled transition。

### 9.5 与 Cognitive Science 的联系

"Affordance" 这个概念来自 Gibson 的 ecological psychology——指环境相对于 observer 的 action possibility。J.J. Gibson 1977 年提出，1979 年在 *The Ecological Approach to Visual Perception* 系统化。

Afford-VLA 在某种意义上是把 Gibson 的 affordance theory 神经化、可学习化、internal 化。这和 robotics 里 manipulation 的核心问题——"where and how to grasp"——天然 align。

### 9.6 Limitations 与 Future Directions

作者自己指出三个 limitation：
1. 依赖 RAGNet 生成的 affordance supervision
2. 只做 2D affordance，没做 3D
3. 真实世界 evaluation 的 diversity 有限

我额外想到的 future direction：
- **Self-supervised affordance**：能否完全靠 action loss 学 affordance，去掉 $\mathcal{L}_{\text{aff}}$？这需要更强的 inductive bias 和更多 data
- **3D Affordance**：把 patch-level mask 升级成 point cloud-level affordance
- **Hierarchical Affordance**：fine-grained affordance（grasp point）vs coarse affordance（interaction region），做 hierarchical planning
- **Affordance as World Model**：affordance 是否可以 predict future，而不只是 current？比如 predict "在 action 之后 affordance 会怎么变"
- **Cross-embodiment Affordance**：不同 robot embodiment 共享 affordance representation
- **Language-to-Affordance**：能否用 LLM 直接生成 affordance specification，再 ground 到 image

### 9.7 与 Visual Chain-of-Thought 的关系

CoT-VLA（Zhao et al. 2025）做 visual chain-of-thought，Afford-VLA 做 affordance chain。两者都是把 reasoning 显式化，但：
- CoT-VLA 是 symbolic（生成 visual token sequence）
- Afford-VLA 是 dense（生成 spatial mask）
- CoT-VLA 不一定 action-aligned
- Afford-VLA 强 action-aligned

这两条 path 在未来可能会 merge——比如用 CoT 推理 "下一步要 grasp 哪个 object"，然后用 Afford-VLA localize 具体 grasp region。

### 9.8 Scaling Properties 的猜想

一个 natural question：这个方法 scale 到更大 VLM（比如 Qwen3-VL-72B）和更多 data 会怎样？

我的猜想：
1. **Positive**：更大 VLM 的 attention 更 expressive，<AFF> token 能学到更复杂的 task-conditioned affordance
2. **Risk**：更大 VLM 自己已经 implicit 学到 spatial reasoning，explicit affordance 的边际收益可能减小
3. **Open question**：是否所有 task 都 benefit from affordance？高 precision task 一定 benefit，但 global coordination task（stack block）可能不 benefit，这个 task-dependent benefit 是 scaling 时要小心的

## 10. 总结：Afford-VLA 的核心贡献

回到最 fundamental 的层面，Afford-VLA 的贡献是：

1. **Conceptual**：formalize 了 visual planning 的四条性质，提供了一个清晰的 design framework
2. **Methodological**：通过 <AFF> tokens + affordance head + mask pooling + straight-through estimator，第一次让 affordance 同时满足 local / visually grounded / internal / action-aligned
3. **Empirical**：在 LIBERO、LIBERO-Plus、SimplerEnv 上 SOTA，real-world 实验 positive
4. **Engineering insight**：two-stage training + ST pooling + predicted-mask-conditioning 这套组合拳解决 train-infer mismatch 和 differentiability 这两个 fundamental issue

更深一层：这篇论文证明了一件事——**VLA 不必把 spatial reasoning 全部 implicit 化在 global representation 里**。通过引入 explicit、internal、action-aligned 的 intermediate representation，可以同时获得 better performance 和 better robustness。这暗示了未来 VLA 的一个重要 design direction：把 perception-action bridge 显式化、模块化、可学习化。

参考资料：
- Afford-VLA 原文: https://arxiv.org/abs/2505.05791 (or 通过 paper key)
- LIBERO benchmark: https://libero-project.github.io/
- SimplerEnv: https://simpler-env.github.io/
- OpenVLA: https://openvla.github.io/
- π₀: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- Qwen3-VL: https://arxiv.org/abs/2505.09388
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- CoA-VLA: https://arxiv.org/abs/2501.06405 (相近 work)
- RT-Affordance: https://arxiv.org/abs/2403.03174 (相关)
- DETR (object query pattern 灵感来源): https://arxiv.org/abs/2005.12872
- SAM (mask decoder pattern): https://arxiv.org/abs/2204.02677
- Straight-Through Estimator (Bengio 2013): https://arxiv.org/abs/1308.3432
- Gibson affordance theory: https://en.wikipedia.org/wiki/Affordance
- LIBERO-Plus: https://arxiv.org/abs/2510.13626

如果你对某个具体细节（比如 flow-matching action head 的数学、straight-through estimator 的具体 surrogate 形式、或者 multi-view attention 的实现）想 dive deeper，告诉我我可以再展开。
