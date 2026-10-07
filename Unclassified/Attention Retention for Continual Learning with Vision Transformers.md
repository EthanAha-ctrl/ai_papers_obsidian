---
source_pdf: Attention Retention for Continual Learning with Vision Transformers.pdf
paper_sha256: 7e74d018b819cfb3de11c21ec50331a6081195edb4d336afac12657041b405fa
processed_at: '2026-07-18T10:56:31-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ARCL-ViT 论文深度讲解

Andrej, 这是一篇挺有意思的 CL paper,来自西北工业大学的 Yue Lu 和 Xiangyu Zhou 团队,以及 Hikrobot。核心 idea 比较干净:在 ViT 上做 continual learning 时,catastrophic forgetting 的主要来源是 attention 的 drift,作者通过 gradient masking 来"冻结"之前任务学到的 attention pattern,从而保留旧知识。

---

## 1. Motivation:Attention Drift 作为 forgetting 的根因

整篇文章的出发点是一个现象学观察,看 Figure 1(a) 的三列对比:

- **第 1 列** (Model-T1):只在 T₁ 上 fine-tune 过的 ViT,attention map 集中在猫的头部等 discriminative region
- **第 2 列** (Model-T10 Seq-FT):顺序 fine-tune 10 个任务之后,attention 完全漂移走了,聚到新任务的 object 上
- **第 3 列** (Model-T10 ARCL-ViT):用他们的方法,10 个任务之后 attention 仍然锚定在原来的 discriminative region

这个观察是 paper 的核心 insight。传统 CL 文献一般把 forgetting 归结为 "权重被覆盖"(EWC 一类),或者 "feature space 互相干扰"(GPM/NSCL 一类的 orthogonal projection)。这篇 paper 把视角拉回到 **attention map** 这个可观测的中间量,认为 ViT 的 forgetting 本质上是 attention 对旧 visual concept 的"失焦"。

神经科学启发:作者引用了 Zhang et al. 2012 关于 V1 区域 saliency map 的工作 —— 人脑 V1 通过神经活动产生 bottom-up saliency map 来 highlight 注意吸引区域,这种 selective attention 在学习新概念时依然保持稳定。这构成生物视觉系统 non-forgetting 的基础。

reference:
- V1 saliency map: https://www.cell.com/neuron/fulltext/S0896-6273(12)00967-7

---

## 2. Setup:冻结 backbone,只调 Q/K/V

ViT 的标准 attention 公式(eq. 1 和 eq. 2):

$$
\mathbf{Q}^l = \mathbf{X}^l \mathbf{W}_q^l, \quad \mathbf{K}^l = \mathbf{X}^l \mathbf{W}_k^l, \quad \mathbf{V}^l = \mathbf{X}^l \mathbf{W}_v^l
$$

其中:
- 上标 $l$ 表示第 $l$ 个 Transformer block($l \in \{1, \ldots, L\}$,ViT-B/16 中 $L=12$)
- $\mathbf{X}^l \in \mathbb{R}^{(N+1) \times D}$ 是 LayerNorm 后的 token 序列,$N$ 是 patch 数(image 224×224,patch 16×16 时 $N=196$),+1 是 cls token
- $D$ 是 embedding dimension(ViT-B 是 768)
- $\mathbf{W}_q^l, \mathbf{W}_k^l, \mathbf{W}_v^l \in \mathbb{R}^{D \times D}$ 是要被 fine-tune 的 projection 权重

Self-attention 分解:
$$
\mathbf{A}^l = D^{-\frac{1}{2}} \mathbf{Q}^l (\mathbf{K}^l)^\top, \quad \mathbf{S}^l = \mathrm{softmax}(\mathbf{A}^l), \quad \mathbf{F}^l = \mathbf{S}^l \mathbf{V}^l
$$

这里 $\mathbf{A}^l$ 是 logits(没有 softmax 过的),$\mathbf{S}^l$ 是 softmax 后的 attention weight matrix,$\mathbf{F}^l$ 是 attention output。这个分解对后面的 gradient masking 推导很关键 —— 他们要对 $\nabla(\mathbf{A})$ 和 $\mathbf{S}$ 做 mask,而不是对最终参数做 mask。

**关键 design choice**:作者只 fine-tune 每层的 $\mathbf{W}_{q,k,v}$ 和 classifier,backbone 的其他参数(FFN、embedding、positional embedding、LayerNorm)全部冻结。这一点和 LoRA 系列方法有神似 —— 都是 minimal tuning set,只动 attention 的 projection。这给了 method 一个天然的"约束面",gradient masking 就在这个小约束面上操作。

---

## 3. 优化目标的精确表述(eq. 3)

作者先写出"理想"目标:

$$
\min_{\mathbf{W}_{\theta,t}^l} \mathbb{E}_{(\mathbf{x}_t, y_t) \sim \mathcal{D}_t} [\ell(g(f^l(\mathbf{x}_t; \mathbf{W}_{\theta,t}^l), y_t))]
$$
$$
\text{s.t.} \quad \lVert \psi^l(\mathbf{x}_{t-1}; \mathbf{W}_{\theta,t}^l) - \psi^l(\mathbf{x}_{t-1}; \mathbf{W}_{\theta,t-1}^l) \rVert = 0
$$

其中:
- $\theta \in \{q, k, v\}$,指代三个 projection 的 index
- $\psi^l(\cdot)$ 是 attention map extraction function
- $\mathbf{W}_{\theta,t}^l$ 是任务 $t$ 上第 $l$ 层第 $\theta$ 个 projection 的权重
- 约束的意思:在新权重下,旧样本 $\mathbf{x}_{t-1}$ 的 attention map 必须等于旧权重下的 attention map

这个约束直接优化不可行:
1. $\psi^l$ 是非线性的(经过 softmax、rollout),闭式解无解
2. 需要存储 $\mathcal{D}_{t-1}$,storage 开销大

作者的近似方案:不直接约束 attention map 本身,而是约束导致 attention 变化的 gradient 方向。这是 paper 的核心 trick。

---

## 4. Gradient Masking 的推导(eq. 4–5)

这是技术上的核心。从 eq. 2 反向求 gradient:

$$
\nabla(\mathbf{Q}_t) = D^{-\frac{1}{2}} \nabla(\mathbf{A}_t) \mathbf{K}_t, \quad \nabla(\mathbf{K}_t) = D^{-\frac{1}{2}} \nabla(\mathbf{A}_t)^\top \mathbf{Q}_t
$$

$$
\nabla(\mathbf{V}_t) = \mathbf{S}_t^\top \nabla(\mathbf{F}_t)
$$

进一步连到 $\mathbf{W}$ 上(因为 $\mathbf{Q} = \mathbf{X}\mathbf{W}_q$):

$$
\nabla(\mathbf{W}_{q,t}) = D^{-\frac{1}{2}} \mathbf{X}_t^\top \cdot \nabla(\mathbf{A}_t) \cdot \mathbf{K}_t
$$
$$
\nabla(\mathbf{W}_{k,t}) = D^{-\frac{1}{2}} \mathbf{X}_t^\top \cdot \nabla(\mathbf{A}_t)^\top \cdot \mathbf{Q}_t
$$
$$
\nabla(\mathbf{W}_{v,t}) = \mathbf{X}_t^\top \cdot \mathbf{S}_t^\top \cdot \nabla(\mathbf{F}_t)
$$

公式变量含义:
- $\nabla(\mathbf{A}_t) \in \mathbb{R}^{(N+1) \times (N+1)}$ 是 attention logits 矩阵的梯度,每行对应一个 query token,每列对应一个 key token
- $\nabla(\mathbf{W}_{q,t})$ 通过 $\nabla(\mathbf{A}_t) \cdot \mathbf{K}_t$ 与 attention 梯度直接耦合
- $\nabla(\mathbf{W}_{v,t})$ 通过 $\mathbf{S}_t^\top$ 与 attention weight 耦合

**关键 intuition**:权重梯度 $\nabla(\mathbf{W}_{\theta,t})$ 是 attention matrix 的 gradient 的线性函数。如果我把 $\nabla(\mathbf{A}_t)$ 或 $\mathbf{S}_t$ 在"旧任务 high-attention 区域"对应的 entry 置零,那么这部分对 $\nabla(\mathbf{W})$ 的贡献就消失了,新权重就不会被"推"向改变旧 attention 的方向。

具体 mask 操作(eq. 5):

$$
\nabla(\mathbf{W}_{q,t})' = D^{-\frac{1}{2}} \mathbf{X}_t^\top \cdot (\nabla(\mathbf{A}_t) \odot \bar{\mathbf{M}}_{t-1}) \cdot \mathbf{K}_t
$$
$$
\nabla(\mathbf{W}_{k,t})' = D^{-\frac{1}{2}} \mathbf{X}_t^\top \cdot (\nabla(\mathbf{A}_t)^\top \odot \bar{\mathbf{M}}_{t-1}^\top) \cdot \mathbf{Q}_t
$$
$$
\nabla(\mathbf{W}_{v,t})' = \mathbf{X}_t^\top \cdot (\mathbf{S}_t^\top \odot \bar{\mathbf{M}}_{t-1}^\top) \cdot \nabla(\mathbf{F}_t)
$$

变量含义:
- $\bar{\mathbf{M}}_{t-1} \in \mathbb{R}^{(N+1) \times (N+1)}$ 是 binary mask:旧任务 attention region 处为 0,background 处为 1
- $\odot$ 是 element-wise 乘
- $\nabla(\cdot)'$ 表示 masked 之后的 gradient

注意 $\bar{\mathbf{M}}_{t-1}$ 的形状是 $(N+1) \times (N+1)$,因为要 broadcast 到每个 query row。论文里讲:把 mask 展平成 vector,在最左边 prepend 一个 0(对应 cls token,保持 cls token 稳定),然后 broadcast 到所有 row。这样所有 query 对那些 high-attention key 的 gradient 都被 zero 掉。

---

## 5. Optimizer-Compatible Scaling(eq. 6–8)

这一步是工程上很关键的细节。直接把 raw gradient 替换成 masked gradient 喂给 Adam,会有问题:Adam 的更新依赖于 first/second moment 的累积统计,masked gradient 在某些位置突然变 0 会扭曲 moment 估计,导致更新幅度异常大或异常小。

作者提出 ratio-preserving scaling:

$$
\frac{\Delta \mathbf{W}_{\theta,t}'}{\Delta \mathbf{W}_{\theta,t}} = \frac{\nabla(\mathbf{W}_{\theta,t})'}{\nabla(\mathbf{W}_{\theta,t})}
$$

即:期望的参数更新 $\Delta \mathbf{W}'$ 与原始更新 $\Delta \mathbf{W}$ 的比值,等于 masked gradient 与原始 gradient 的比值。重排得:

$$
\Delta \mathbf{W}_{\theta,t}' = \frac{\nabla(\mathbf{W}_{\theta,t})'}{\nabla(\mathbf{W}_{\theta,t})} \odot \Delta \mathbf{W}_{\theta,t}
$$

然后:

$$
\mathbf{W}_{\theta,t}^{\langle s \rangle} = \mathbf{W}_{\theta,t}^{\langle s-1 \rangle} - \gamma \Delta \mathbf{W}_{\theta,t}'
$$

变量:
- $\Delta \mathbf{W}_{\theta,t}$ 是 optimizer 用 unmasked gradient 算出来的原始更新(包含 Adam 的一阶、二阶 moment)
- $\Delta \mathbf{W}_{\theta,t}'$ 是按比例缩放后的"期望"更新
- $\langle s \rangle$ 表示第 $s$ 个 optimization step
- $\gamma$ 是 learning rate

这个 trick 的 intuition:不直接动 optimizer 的内部状态,只对最终 step 的 update vector 做一个 element-wise 的 ratio scaling。这样保留了 Adam 的 adaptive 性质,同时实现了 "在旧 attention 区域不更新"的效果。除零风险需要小心处理(实践中会在 $\nabla(\mathbf{W}) \approx 0$ 的地方加一个 $\epsilon$)。

这个思路让我想起 SAT (Saturated Attention Tuning)、PyTorch 的 `torch.clamp` 类的"软约束" —— 不强行禁止,而是按比例衰减。比"硬置零"鲁棒得多。

---

## 6. Adaptive Mask Generation

### 6.1 Layer-wise Rollout(eq. 9)

原始 attention rollout 来自 Abnar & Zuidema 2020,思想是: attention 是"信息流",要衡量 layer $l$ 对最终输出的影响,需要把前面所有层的 attention matrix 乘起来。

作者的 layer-wise rollout:

$$
\hat{\mathbf{S}}_{t-1}^l = \tilde{\mathbf{S}}_{t-1}^1 \cdot \tilde{\mathbf{S}}_{t-1}^2 \cdot \ldots \cdot \tilde{\mathbf{S}}_{t-1}^l
$$

其中 $\tilde{\mathbf{S}}_{t-1}^l = \mathbf{I} + \mathbf{S}_{t-1}^l$,$\mathbf{I}$ 是单位矩阵。

变量含义:
- $\mathbf{S}_{t-1}^l$ 是旧模型第 $l$ 层的 softmax 后 attention matrix
- $\tilde{\mathbf{S}}$ 加 $\mathbf{I}$ 是为了防止跨层的 self-suppression(因为深层 attention 可能让某些 token 几乎完全 attend 到自己,导致累积乘积为 0)
- $\hat{\mathbf{S}}_{t-1}^l$ 是从第 1 层累积到第 $l$ 层的 rollout attention

注意这是 **layer-wise**,即每一层都有自己的 rollout。这与原始 rollout 不同 —— 原始 rollout 只算最后一层的累积,这里对每一层都计算累积到该层的版本,因为后面要对每一层分别生成 mask。

reference:
- Attention rollout: https://arxiv.org/abs/2005.00928

### 6.2 Class Attention Map 提取

从 $\hat{\mathbf{S}}_{t-1}^l \in \mathbb{R}^{(N+1) \times (N+1)}$ 提取 cls token 对所有 image patch 的 attention:

$$
\mathbf{U}_{t-1}^l = \hat{\mathbf{S}}_{t-1}^l[1, 2:]
$$

即取第 1 行(第 1 个 token 是 cls,作为 query),去掉第 1 个元素(cls 自己 attend 自己的部分),剩下 $N$ 个值,reshape 成 $\sqrt{N} \times \sqrt{N} = 14 \times 14$ 的 2D map。这是 standard practice,跟Attention Rollout / GradCAM 类方法的 visualizable map 一致。

### 6.3 Adaptive Thresholding via Second-Order Derivative(eq. 10)

这是 paper 一个很 elegant 的小 trick。作者的观察:由于 softmax 的 sharpening effect,attention map 排序后像一个 sigmoid 曲线 —— 一段低值 plateau,一段急剧上升,一段高值 plateau。前一段是 background,后一段是 attention region。两段中间的拐点就是最佳 threshold。

数学上,把 $\mathbf{U}_{t-1}^l$ 的元素按升序排成 $u_1, u_2, \ldots, u_N$,找二阶差分最小的位置:

$$
k^* = \arg\min_{2 \leq k \leq N-1} (u_{k+1} - 2u_k + u_{k-1})
$$

变量:
- $u_k$ 是 attention map 排序后的第 $k$ 个值
- $u_{k+1} - 2u_k + u_{k-1}$ 是离散二阶差分
- $k^*$ 是二阶差分最小的 index,对应 sigmoid 曲线最平缓的拐点
- $\tau_{t-1}^l = u_{k^*}$ 是该层该样本的 adaptive threshold

直觉:在 sigmoid 上升最陡的地方二阶差分取极值,这就是 background 和 attention 的边界。然后:

$$
\mathbf{M}_{t-1,i}^l = \begin{cases} 0, & \text{if } \mathbf{U}_{t-1,i}^l \geq \tau_{t-1}^l \\ 1, & \text{otherwise} \end{cases}
$$

attention 区域 mask 为 0(gradient 会被 zero 掉),background 区域 mask 为 1(gradient 保留)。

### 6.4 跨样本聚合

实践中,作者把每个样本的 mask 平均到该 class 内,再平均到该 task 内,得到连续值 [0, 1] 的 mask 而不是 binary。这是合理的 —— 不同样本的 attention region 有共性也有差异,平均后得到"该任务总体的 attention pattern"。

---

## 7. 整体算法流程

1. 任务 $\mathcal{T}_{t-1}$ 训练完毕后:
   - 对 $\mathcal{D}_{t-1}$ 中每个样本做 forward,提取每层的 $\mathbf{S}_{t-1}^l$
   - 用 layer-wise rollout 计算 $\hat{\mathbf{S}}_{t-1}^l$
   - 提取 class attention map $\mathbf{U}_{t-1}^l$
   - Adaptive thresholding 生成 $\mathbf{M}_{t-1}^l$
   - 跨样本、跨 class 聚合成 $\bar{\mathbf{M}}_{t-1}^l$

2. 任务 $\mathcal{T}_t$ 训练时:
   - Forward 计算 $\mathbf{A}_t, \mathbf{S}_t, \mathbf{F}_t$
   - Backward 计算 $\nabla(\mathbf{A}_t), \nabla(\mathbf{F}_t)$
   - Apply mask $\bar{\mathbf{M}}_{t-1}$ 得到 $\nabla(\mathbf{W}_{\theta,t})'$
   - 用 Adam 算出 $\Delta \mathbf{W}_{\theta,t}$
   - Ratio scaling 得到 $\Delta \mathbf{W}_{\theta,t}'$
   - 更新参数

3. 多 head 时,eq. 5–11 对每个 head 独立应用。

---

## 8. 实验结果分析

### 8.1 主表(Table 1)

| Benchmark | ARCL-ViT | 次优 (Existing Best) | Seq-FT Baseline |
|-----------|----------|----------------------|-----------------|
| 10S-ImageNet-R | **81.17** | VPT-CPG 78.63 | 49.43 |
| 20S-ImageNet-R | **79.40** | CPrompt-KAC 75.73 | 39.63 |
| 10S-CIFAR-100 | **90.87** | VPT-CPG 90.63 | 52.85 |
| 10S-DomainNet | **83.94** | VPT-CPG 83.21 | 44.06 |

几个要点:
- 在 10S-DomainNet(cross-domain,最有挑战性)上提升明显:83.94 vs VPT-CPG 83.21
- Seq-FT 上 ImageNet-R 是 49.43,经过 gradient masking 提升到 81.17,直接涨 31.74 个点,forgetting 从 44.31 降到 4.84
- Forgetting metric 上不是最低的(最低是 CODA-Prompt 1.64),但 accuracy 最高。这表明 method 在 plasticity-stability 上做了 trade-off,偏向"既能学新东西,又少忘旧东西",而不是"完全不忘但学不进新东西"

值得注意:与 Seq-FT 的对照实验是最 fair 的 —— 唯一区别就是 gradient masking,这直接验证了核心 idea 的有效性。

### 8.2 Ablation:Mask 构造方式(Table 2)

| Index | Rollout 方式 | Threshold | 10S-ImageNet-R Acc | 10S-DomainNet Acc |
|-------|--------------|-----------|---------------------|-------------------|
| 1 | None (Seq-FT) | - | 49.43 | 44.06 |
| 2 | Raw + Fixed | top 20% | 77.63 | 76.28 |
| 3 | Raw + Adaptive | sigmoid拐点 | 80.33 | 82.17 |
| 4 | Naive rollout + Adaptive | | 80.43 | 82.28 |
| 6 | Layer-wise rollout + Adaptive (full) | | **81.17** | **83.94** |

观察:
- 即使是最朴素的 raw attention + fixed threshold 也比 Seq-FT 高 28 个点 → 核心机制有效
- Adaptive vs Fixed threshold 平均涨 5.4 个点 → threshold 策略很重要
- Layer-wise rollout 比 raw / naive rollout 涨 1.3 个点 → rollout 提供 margin

### 8.3 Ablation:Mask 哪个区域(Table 3)

| Masked Region | 10S-ImageNet-R Acc | Forg. |
|---------------|---------------------|--------|
| Random (同数量) | 73.27 | 4.22 |
| Non-attention 区域 | 69.55 | 7.56 |
| **Attention 区域** | **81.17** | 4.84 |

- Random masking 居然也涨不少 → 部分原因可能是 mask 本身起到 regularization 作用,有点像 dropout
- Non-attention masking 也比 Seq-FT 好,作者解释:一个 class 的 non-attention 区域可能 overlap 另一个 class 的 attention 区域,所以间接提供部分保护
- 只有 attention masking 达到最佳

### 8.4 不同 Pre-training Weights(Table 4)

在 DINO-1k 和 iBOT-1k(都是自监督,在 ImageNet-1k 上预训练)上验证:
- DINO-1k: ARCL-ViT 72.37 vs 次优 68.31 (EASE)
- iBOT-1k: ARCL-ViT 76.48 vs 次优 72.16 (VPT-CPG)

意义:方法对 pre-training 方式不敏感,在 self-supervised pre-trained backbone 上同样有效。

reference:
- DINO: https://arxiv.org/abs/2104.14294
- iBOT: https://arxiv.org/abs/2111.07832

### 8.5 Long-Sequence CL(Table 5)

| Setting | ARCL-ViT | 次优 | Seq-FT |
|---------|----------|-------|--------|
| 50S-ImageNet-R | **76.45** | VPT-CPG 73.08 | 22.57 |
| 100S-ImageNet-R | **70.72** | VPT-CPG 64.63 | 10.65 |
| 50S-DomainNet | **74.32** | VPT-CPG 72.27 | 9.89 |
| 100S-DomainNet | **62.35** | VPT-CPG 60.81 | 8.87 |

100 任务的长序列下 Seq-FT 崩塌到 8.87,ARCL-ViT 仍保持 62.35。作者特别指出 forgetting 没有比 VPT-CPG 显著低,但 accuracy 显著高,说明在长序列下 method 通过"controlled forgetting"实现了更好的 plasticity-stability 平衡。

---

## 9. 与 Related Work 的联系

### 9.1 Regularization-based CL

经典方法:
- **EWC** (Kirkpatrick 2017):Fisher 信息矩阵衡量参数重要性,加 quadratic penalty。 https://arxiv.org/abs/1612.00796
- **SI** (Zenke 2017):path integral 估计参数重要性。 https://arxiv.org/abs/1703.04200
- **OWM** (Zeng 2019):投影到旧任务 input 的 null space。 https://arxiv.org/abs/1810.02256
- **GPM** (Saha 2021):gradient projection memory,存 basis of gradient subspace。 https://arxiv.org/abs/2103.09762
- **NSCL** (Wang 2021 / Lu 2024):feature covariance 的 null space。
- **InfLoRA** (Liang & Li 2024):interference-free LoRA,把 LoRA subspace 与旧任务 orthogonal。 https://arxiv.org/abs/2404.00228

ARCL-ViT 在精神上和这些 orthogonal projection 方法接近 —— 都是"约束 gradient 不能往某个方向走"。区别:
- GPM / NSCL / InfLoRA 的约束方向来自 input space 的 null space(几何上)
- ARCL-ViT 的约束方向来自 attention map 的 region(语义上)

两者可以看作是两种不同的"重要度度量":前者度量参数对 feature 的影响,后者度量参数对 attention 的影响。

### 9.2 Replay-based CL

- **iCaRL** (Rebuffi 2017):herding 选 exemplar + NME 分类。 https://arxiv.org/abs/1611.07725
- **Memory Replay GANs** (Wu 2018):生成旧样本

ARCL-ViT 是 rehearsal-free,符合最近 ViT-based CL 不存样本的趋势(privacy / storage)。

### 9.3 Expansion-based CL (Prompt-based)

- **L2P** (Wang 2022b):prompt pool + key-query。 https://arxiv.org/abs/2112.08654
- **DualPrompt** (Wang 2022a):G-prompt + E-prompt。 https://arxiv.org/abs/2204.01558
- **CODA-Prompt** (Smith 2023):attention decomposition prompt。 https://arxiv.org/abs/2210.06943
- **S-Prompts** (Wang 2022c):per-task prompt
- **CPrompt** (Gao 2024):consistent prompting
- **VPT-CPG** (Lu 2025b):mixture-of-experts prompt generator

这些方法通过加新参数(prompt pool)来"扩展"模型容量。ARCL-ViT 走相反路径 —— 不加新参数,只 fine-tune Q/K/V,通过 gradient mask 保护已有 attention。两种思路在不同方向上探索,但实际效果 ARCL-ViT 更胜一筹,可能是因为 prompt-based 方法本质上还是参数扩张,但 ViT 已经被预训练得很好,加 prompt 反而引入了 inductive bias 干扰。

### 9.4 LoRA-based CL

最近 2024-2025 出现了大量 LoRA-based CL:
- **SD-LoRA** (Wu 2025):scalable decoupled LoRA
- **BiLoRA** (Zhu 2025):almost-orthogonal LoRA subspaces
- **LoRA-DRS**:dynamically reweighted subspaces
- **InfLoRA** (Liang 2024):interference-free LoRA
- **KAC** (Hu 2025):Kolmogorov-Arnold Classifier (CVPR'25,把 KAN 引入 CL)

ARCL-ViT 与这些方法的区别:LoRA 方法通过低秩矩阵分解隔离新旧任务的影响,ARCL-ViT 直接在原始 attention weight 上通过 gradient masking 来保护 attention pattern。两者可以理论上结合 —— 比如把 mask 应用到 LoRA 的 A/B matrix 上。

### 9.5 Attention 在 CL 中的其他工作

- **AttriCLIP**:attention-based CLIP CL
- **SLCA**:Slow Learner with Classifier Alignment
- 这些工作也注意到 attention 在 CL 中的重要性,但都没有 ARCL-ViT 这么直接 —— 在 gradient 层面对 attention region 做精确的 surgical masking

---

## 10. 直觉构建(Intuition Building)

### 10.1 为什么 Attention 是 CL 的"记忆载体"

ViT 的 prediction 几乎完全依赖于 cls token 从 image patch 聚合的信息。聚合通过 attention 实现:
- Attention 集中在 object head → cls token 编码 head feature
- Classifier 读 cls token → 输出 "cat"

如果 attention drift 到 background,cls token 编码的内容改变,classifier 读出来就不是 "cat"。所以 attention map 是 prediction 的"指针",指针漂移等于知识丢失。

这一点和 CNN CL 不太一样。CNN 中 feature 是 spatially pooled 的全局 feature,没有显式 attention 概念,所以 CNN CL 主要看 feature representation drift。ViT 中 attention + feature 是耦合的,attention drift 是更上层的现象。

### 10.2 Gradient Masking 为何能"冻结" Attention

考虑 $\mathbf{A}_t = D^{-1/2} \mathbf{Q}_t \mathbf{K}_t^\top$ 在 backward 中:

$$
\frac{\partial \mathbf{A}_t[i,j]}{\partial \mathbf{W}_q} = D^{-1/2} \mathbf{X}_t^\top \cdot \mathbf{K}_t[j] \cdot \mathbf{e}_i
$$

即 attention logit $\mathbf{A}_t[i,j]$ 对 $\mathbf{W}_q$ 的 gradient 是 input $\mathbf{X}_t$ 的转置乘以 $\mathbf{K}_t$ 的第 $j$ 行(被 query $i$ 加权)。

如果在 backward 中,我们 mask 掉 $\nabla(\mathbf{A}_t)[i, j_{\text{old-att}}] = 0$,意思是:不要让 loss 推动 $\mathbf{A}[i, j_{\text{old-att}}]$ 改变。这就保护了旧任务 attention region 不被新任务 loss 推动。

进一步把 $\nabla(\mathbf{A}_t)$ 反传到 $\mathbf{W}_q$,发现 $\nabla(\mathbf{W}_q)$ 是 $\nabla(\mathbf{A}_t) \cdot \mathbf{K}_t$ 的线性组合,所以 mask 也自然传到了 $\mathbf{W}_q$ 上。整条链上,旧 attention 区域对应的 gradient 被精准切除。

### 10.3 Optimizer-Compatible Scaling 的几何意义

Adam 的 update 大约是:

$$
\Delta \mathbf{W} = -\frac{m_t}{\sqrt{v_t} + \epsilon}
$$

如果直接用 masked gradient $\nabla' = \nabla \odot M$,Adam 算出来的 $m_t', v_t'$ 会累积历史 0 值,逐渐扭曲二阶矩估计。Ratio scaling 的本质是:让 optimizer 算它本来要算的东西,然后在最后一步"按 mask 比例缩放"。

这相当于把 mask 看作"软约束":在 mask=0 的位置,update 严格为 0;在 mask=1 的位置,update 与 unmasked 情况完全一样。中间值 mask(连续 [0,1])则线性插值。

这个 trick 其实和 `grad * mask` 不一样 —— 后者扭曲了 optimizer 的内部状态,前者只扭曲最终的 update vector。在 Adam 这种有 memory 的 optimizer 上,差别显著。

### 10.4 Adaptive Thresholding 的 Sigmoid 假设

Adaptive thresholding 假设 attention map 排序后是 sigmoid 形状。这个假设来自 softmax 的 sharpening effect:
- 在 attention 区域,softmax logits 高,softmax 后 attention 显著
- 在 background,logits 低,softmax 后 attention 接近 0
- 排序后,高值区域聚集在末尾,低值区域聚集在开头,中间是过渡

二阶差分最小点就是过渡的中点。这个 trick 比固定 percentile 优雅,因为它适应每张图的 attention 分布差异:有的图 attention 很集中(对象小),有的图 attention 很分散(对象大或多个)。

### 10.5 跨任务 Attention Drift 的可视化(Figure 1b)

作者计算 attention drift:

$$
\text{Drift}(t) = \frac{\lVert \psi^l(\mathbf{x}_1; \mathbf{W}_t) - \psi^l(\mathbf{x}_1; \mathbf{W}_1) \rVert_F}{\lVert \psi^l(\mathbf{x}_1; \mathbf{W}_1) \rVert_F}
$$

变量:
- $\mathbf{W}_t$ 是第 $t$ 个任务训练完的模型权重
- $\mathbf{x}_1$ 是第 1 个任务的样本
- Frobenius norm 衡量两个 attention map 的差异,除以原始 norm 做归一化

结果:
- Seq-FT:25.9% drift
- ARCL-ViT:6.0% drift

差距 4 倍多,直观说明 method 真的"留住"了 attention。

---

## 11. 一些可以进一步思考的点

### 11.1 局限性

1. **只 mask Q/K/V**:method 只 fine-tune 这三个 projection,FFN 完全冻结。这是优点也是局限 —— 如果 task 真的需要新 feature transformation,FFN 不能适应。可以考虑把 method 推广到 FFN 的 gradient masking。
2. **依赖预训练 backbone**:方法是 ViT-based,依赖 strong pre-trained backbone 提供 general representation。在 from-scratch 训练上效果未知。
3. **Mask 跨任务累积**:任务越往后,需要保护的 attention region 越多,可能 mask 太多导致新任务学不进去。Table 5 的 100S-DomainNet 上 forgetting 25.63% 已经偏高,可能是这个原因。
4. **Memory cost**:每个任务都要存一组 mask $\bar{\mathbf{M}}_{t-1}^l$,共 $T \times L$ 个 $(N+1)^2$ 大小的 mask。对长序列会累积。
5. **理论分析**:Appendix 提到 "Theoretical Justification",但 main text 没展开。能否证明 gradient masking 等价于在某个 attention-invariant subspace 上的 optimization?这是个可以深挖的方向。

### 11.2 与其他方法的潜在结合

- **+ LoRA**:把 mask 应用到 LoRA 的 A 或 B matrix 上,可能更 efficient
- **+ Replay**:对 replay 的 sample 也应用 attention masking,可能让 replay 更 surgical
- **+ KAN (KAC)**:Kolmogorov-Arnold 的非线性能不能更好表达 attention pattern?
- **+ MoE**:用 MoE 处理新任务的 attention,旧任务用 mask 保护的 backbone attention
- **MLM / LLM extension**:idea 可以推到 BERT / GPT 的 transformer 上,对 token-level task 做 CL(比如持续学新领域语料)

### 11.3 神经科学类比的有效性

作者引 V1 saliency map 作类比,但人脑的 selective attention 机制远比 ViT 复杂:
- 人脑有 top-down attention(目标驱动)和 bottom-up attention(saliency 驱动)的分离
- ViT 的 attention 是纯 bottom-up 的(class token 可以看作 weak top-down)
- 人脑的 attention 不是"冻结"的,而是 dynamic reweighting with consolidation

paper 的神经科学类比更多是 inspiration 而非严格 mechanism correspondence,这点要清楚。

---

## 12. 总结

ARCL-ViT 是一个干净、有效、可解释的 ViT CL 方法。核心技术贡献:
1. **诊断**:识别 attention drift 是 forgetting 主因
2. **机制**:gradient masking 在 attention matrix 层面 surgical 切除旧任务 region 的 gradient
3. **工程**:optimizer-compatible scaling 让 method 与 Adam 兼容
4. **细节**:layer-wise rollout + adaptive thresholding 让 mask 精确捕捉 discriminative region

实验全面:4 个标准 benchmark + 2 个 pre-training + long-sequence 都达到 SOTA。Ablation 充分证明了每个 component 的必要性。

代码:https://github.com/zugexiaodui/AttentionRetentionCL

reference 集合:
- ViT: https://arxiv.org/abs/2010.11929
- Attention rollout: https://arxiv.org/abs/2005.00928
- EWC: https://arxiv.org/abs/1612.00796
- SI: https://arxiv.org/abs/1703.04200
- OWM: https://arxiv.org/abs/1810.02256
- GPM: https://arxiv.org/abs/2103.09762
- InfLoRA: https://arxiv.org/abs/2404.00228
- iCaRL: https://arxiv.org/abs/1611.07725
- DER: https://arxiv.org/abs/2104.06066
- L2P: https://arxiv.org/abs/2112.08654
- DualPrompt: https://arxiv.org/abs/2204.01558
- CODA-Prompt: https://arxiv.org/abs/2210.06943
- S-Prompts: https://arxiv.org/abs/2207.12819
- DINO: https://arxiv.org/abs/2104.14294
- iBOT: https://arxiv.org/abs/2111.07832
- V1 saliency: https://www.cell.com/neuron/fulltext/S0896-6273(12)00967-7
- ImageNet-R: https://arxiv.org/abs/2006.16241

希望这些细节能帮你 build 出 intuition:method 的本质是在 backward 过程中"外科手术式"地切除会扰动旧 attention 的 gradient,从而让新任务的学习只能在"非旧 attention"的方向上发生,attention 被迫保持稳定,旧 visual concept 被保留下来。
