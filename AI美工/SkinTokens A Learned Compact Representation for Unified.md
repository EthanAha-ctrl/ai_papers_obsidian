---
source_pdf: SkinTokens A Learned Compact Representation for Unified.pdf
paper_sha256: 5b6acdc1ed1cb609046b3999e5ec77a140543f3d1b728d3a962ed3f5c742462b
processed_at: '2026-08-12T07:35:27-07:00'
target_folder: AI美工
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SkinTokens

## 一句话总结

3D 模型越生越多，但给它们绑骨做动画这个活儿还是手工的。这篇 paper 说：**把 skinning 这个大稀疏矩阵压成几个 token，然后跟 skeleton 拼成一条长序列，让 Transformer 一次性全生成**。就这么个事，但效果炸裂。

---

## 为什么要做这个事

### 行业现状

现在 AI 生成 3D 模型贼快（text-to-3D 一堆 paper），但生成出来是个"死"模型——没有 skeleton，没有 skinning weights，没法做动画。要让模型动起来，artist 要手工：
1. 往模型里塞一整套 skeleton（骨头架）
2. 对每个 vertex 说"你受哪几根 bone 影响，各占多少比例"（skinning weights）

一个 production 角色几千个 vertex、几十根 bone，手工搞要几小时甚至几天。**生成侧已经工业化了，rigging 还停留在手工作坊**，这个 gap 就是 bottleneck。

### 之前自动 rigging 怎么做

分两派：

**Template-based**：预先定义好 humanoid skeleton 模板，往上套。对人形角色 OK，遇到四足、怪 物、带翅膀斗篷的完蛋。

**Template-free**：RigNet 这类用 GNN 预测 joint heatmap，再连成 skeleton。灵活但 skeleton 连接容易出错。

最近两年又冒出 **autoregressive skeleton 生成**（UniRig, Puppeteer, MagicArticulate），把 skeleton 拆成 token 序列，用 Transformer 生成。skeleton 这块进步很大。

**但是 skinning 还是老样子**：大家都把它当 downstream regression 任务，用 GNN 从 mesh 几何特征直接回归 N×J 矩阵。没人把它 token 化。

### Skinning 到底难在哪

给你一个 mesh $\mathcal{M} = \{\mathcal{V}, \mathcal{F}\}$，$N$ 个 vertex，skeleton 有 $J$ 个 joint，要预测一个 $N \times J$ 的 weight matrix $\mathcal{W}$。

实际数据量级：
- VRoid Hub: $N=1298$, $J=20$，矩阵 25960 个元素
- Articulation 2.0: $N=16930$, $J=96$，矩阵 162 万个元素

**关键事实**：每个 vertex 通常只受 ≤4 个 bone 影响，所以 90%+ 的元素是 0。Table 1 说稀疏度只有 2-10%。

这就尴尬了。你用 MSE loss 训网络，网络只要输出全 0 就能拿 90% 分。gradient signal 被 trivial zero-prediction 淹没。结果就是预测出来的 weight noisy，动画一跑全是 artifact（穿模、抖动、candy-wrap）。

另外 vertex ordering 任意，没法用传统稀疏矩阵压缩（block sparsity 之类）。必须 learned compression。

---

## SkinTokens：核心 idea

### 直接动机

既然稀疏，那就压缩。把每根 bone 的 skinning weight vector $\mathcal{W}^* = \{w_{(\cdot),j}\}$ 压成几个 discrete token。

这跟 VQ-VAE 压图像、SoundStream 压音频一个思路：**找到低秩 manifold，用 codebook index 表示**。

### 为什么选 FSQ 不选 VQ-VAE

VQ-VAE 经典做法：learnable codebook + commitment loss + EMA update。问题是 codebook collapse——一大半 code 永远不被用，训着训着就废了。

FSQ (Finite Scalar Quantization) 思路完全不同：**不学 codebook，直接量化到 fixed grid**。

具体说，latent 每个维度量化到最近的 level。比如 level sizes $[8,8,8,5,5,5]$，意思每个维度分别切 8、8、8、5、5、5 份，总 codebook size = $8 \times 8 \times 8 \times 5 \times 5 \times 5 = 64000$。没有 codebook collapse，因为 grid 是固定的，只是看 encoder 会不会映射过去。

梯度用 STE（Straight-Through Estimator）：前向用 quantized value，反向 gradient 直接传给 continuous latent，假装量化不存在。

$$L_D = \text{FSQ}(L_W) = \text{round}(L_W) + \text{sg}(L_W - \text{round}(L_W))$$

- $L_W$：encoder 输出的 continuous latent
- $\text{round}(\cdot)$：量化操作
- $\text{sg}(\cdot)$：stop-gradient，反向传播时 gradient 跳过这一项

Table 2 显示 codebook utilization 86.2%，说明 encoder 学到了 diverse distribution，没有 collapse 到几个 code。

### CVAE 的 Conditioning

这是 **Conditional** VAE，conditioning 是 mesh geometry。

两个 encoder，都基于 VecSet (3DShape2VecSet, Zhang et al. 2023)：
- $E_M(\mathcal{M})$ 吃 mesh，输出 shape features
- $E_W(\mathcal{W}^*)$ 吃 skinning weights，输出 latent $L_W$
- Decoder 把 shape features + quantized $L_D$ 拼起来，重建 skinning weights

VecSet 把 mesh 当 unordered point set 处理，permutation invariant。这跟 PointNet 思路一样，绕开 vertex ordering 任意的问题。

**注意 asymmetric 设计**：encoder $E_M$ 训练时只看 uniform sampled points（推理时也这样），但 decoder 训练时看 uniform + dense sampled from active region。这是 importance sampling，让 decoder 每个 batch 都能充分监督 active region。

### Dice Loss：稀疏重建的灵魂

这个细节值得展开讲。

标准 BCE loss：
$$\mathcal{L}_{\text{BCE}} = -\sum_i [w_i \log w_{\text{pred}_i} + (1-w_i) \log(1-w_{\text{pred}_i})]$$

Dice loss：
$$\mathcal{L}_{\text{Dice}} = \sum_{j \leq J} 1 - \frac{2 \sum_i w_{\text{pred}_{i,j}} w_{i,j} + \varepsilon}{\sum_i w_{\text{pred}_{i,j}}^2 + \sum_i w_{i,j}^2 + \varepsilon}$$

变量：
- $i$：vertex index
- $j$：bone index  
- $w_{\text{pred}_{i,j}}$：预测 weight
- $w_{i,j}$：GT weight
- $\varepsilon = 10^{-4}$：防除零

Figure 3 画了 gradient analysis。结论：

当 GT $w=0$ 时（target 是零），BCE 在 $w_{\text{pred}} \in (0,1]$ 范围内 gradient 很小，网络没动力压 false positive。

当 GT $w>0$ 时（active region），Dice 的 gradient 远大于 BCE。

**核心 insight**：Dice loss 把监督信号集中在 10% 的 active region，正好对冲 90% 都是 zero 的 class imbalance。这是 medical image segmentation (V-Net, Milletari et al. 2016) 的老智慧，paper 把它移植到 3D skinning。

Ablation (Table 5) 验证：去掉 Dice loss，IoU 从 87.1% 掉到 82.2%（VRoid Hub）。

### Nested Dropout：信息分层

借鉴 FlexTok (Bachmann et al. 2025) 和 Rippel et al. 2014。训练时随机取 token 序列的 prefix——比如 32 个 token 的序列，训练时随机只看前 8 个或前 16 个，强迫前面的 token 携带主要信息。

效果：Table 2 / Figure 4 显示 **$T_D=4$ 个 token 就能重建出 high fidelity skinning**。这就是说 skinning 的 intrinsic information 极度集中，4 个 discrete token 编码一根 bone 的全部 skinning 信息足够。

类比 JPEG coarse-to-fine，或者 LLM 里"重要信息放前面"。

### 压缩效果

$T_D=32$ tokens，每个 token 是 codebook index，存 2 bytes，总 64 bytes/bone。

原始 FP16：6247 vertices × 34.46 bones × 2 bytes ≈ 432 KB
压缩后：32 × 34.46 × 2 ≈ 2.2 KB
**Compression ratio 183.74×**

而且没明显信息损失。从信息论角度看，4N 个非零值每个 log 精度，intrinsic information 远小于 dense matrix 存储。

---

## TokenRig：把 skeleton 和 skinning 拼成一条序列

### 序列结构

```
<bos> <type_1> dx_1 dy_1 dz_1 ... <type_k> dx_T dy_T dz_T 
D_{1,0} ... D_{1,T_D} ... D_{T,0} ... D_{T,T_D} <eos>
```

- `<bos>` `<eos>`：序列边界
- `<type_k>`：bone chain 类型（比如 mixamo humanoid）
- $dx_i, dy_i, dz_i$：joint 坐标 uniform quantize 成 integer token
- $D_{i,j}$：第 $i$ 个 joint 的第 $j$ 个 SkinToken

**关键顺序**：先输出完整 skeleton，再输出所有 SkinTokens。

为什么这样？因为 Transformer self-attention 让每个 skin token 都能 attend 到所有 joint 位置。生成手指 skinning 时，模型知道肩膀关节在哪。这是全局 conditioning。

之前 RigNet 用 GNN local receptive field，只能看局部邻居。TokenRig 用 Transformer，long-range dependency 自然建模。

### 为什么 unified 比 decoupled 好

UniRig, Puppeteer, MagicArticulate 都是两阶段：
1. 阶段一生成 skeleton
2. 阶段二固定 skeleton，用另一个网络回归 skinning

问题：
- **Error propagation**：skeleton 错了，skinning 永远救不回来
- **No mutual information**：生成 skeleton 时不知道 surface deformation 需求
- **Representation mismatch**：skeleton 用 token，skinning 用 continuous regression，两个 modality 没法 joint optimize

TokenRig 把两者都变成 token，一个 Transformer 全生成。Cross-modal dependency 通过 self-attention 自然捕捉。

### Backbone

Qwen3-0.6B。为什么用 LLM backbone？因为 skeleton + skin token 序列本质就是 1D discrete sequence，跟 text token 没区别。LLM 的所有技术栈直接迁移：
- GQA (Grouped Query Attention)：减 KV cache 内存
- RoPE (Rotary Position Embedding)：长序列外推好
- 0.6B 参数：sweet spot，capacity 够又不过分

Optimizer：Muon 用于 attention layers，AdamW 用于其他。Muon 用 Newton-Schulz 迭代做 orthogonal update，对 attention 的 high-rank gradient 特别有效。

---

## RL 微调：让模型对"野生"模型也能干活

### 为什么需要 RL

Supervised next-token prediction 学的是 training data 的统计分布。对训练集内 OK，对 OOD（in-the-wild）的怪东西会 fallback 到"average solution"：
- 忽略 wings, tails, horns（训练集里没见过的附件结构）
- bone 突出 mesh 表面
- skinning 覆盖不全

这些 failure mode 用 supervised loss 难表达，但可以用 explicit geometric reward 编码。

### 为什么用 GRPO 不用 PPO/DPO

GRPO (Shao et al. 2024, DeepSeekMath)：
- 不需要 critic（PPO 需要训 value function）
- 不需要 preference data（DPO 需要 pairwise ranking）
- 用 group-relative baseline：同一 input 采样 24 个 output，reward 在组内归一化当 advantage

直觉：模型自己跟自己比。对每个 mesh 生成 24 个 rig，谁的 reward 高谁就是正样本。

### 四个 Reward 详解

#### R_vj：Volumetric Joint Coverage

$$R_{vj} = \frac{1}{V} \sum_{i=1}^{V} \exp\left(-\alpha \min_{j=1}^{J} \|v_i - J_j\|_2\right)$$

变量：
- $V$：mesh 被 voxel 化成 $196^3$ grid 后 occupied voxel 数量
- $v_i$：第 $i$ 个体素中心
- $J_j$：第 $j$ 个 joint
- $\alpha = 0.05$：exponential kernel 衰减速度

直觉：对每个 voxel 找最近 joint，距离越小 reward 越大。惩罚"漏放骨"——某区域没 bone，对应 voxel 贡献接近 0。

作者把 $w_{vj}=5$ 设得最高，这是最重要的 reward。

#### R_vk：Bone-Mesh Containment

$$R_{vk} = \frac{1}{J \times (s+1)} \sum_{j=1}^{J} \sum_{i=1}^{s+1} \mathbb{I}[J_{j,i} \in \mathcal{V}]$$

变量：
- $J$：bone 数量
- $s$：每根 bone 上采样点数
- $J_{j,i}$：第 $j$ 根 bone 上第 $i$ 个采样点
- $\mathcal{V}$：voxelized mesh
- $\mathbb{I}[\cdot]$：indicator function，点在 mesh 内返回 1

直觉：直接惩罚 bone 突出 mesh 表面。每根 bone 上采几个点，看在不在 mesh 里。

#### R_sc：Skinning Coverage and Sparsity

$$R_{sc} = 1 - \frac{1}{2} R_z - \frac{1}{2} R_m$$

$$R_z = \left(\frac{1}{|\mathcal{V}|} \sum_i \prod_{j=1}^{J} \mathbb{I}[\mathcal{W}_{i,j} < \beta]\right)^{\alpha_z}$$

$$R_m = \left(\frac{1}{|\mathcal{V}|} \sum_i \mathbb{I}\left[\left(\sum_{j=1}^{J} \mathbb{I}[\mathcal{W}_{i,j} > \beta]\right) > 4\right]\right)^{\alpha_m}$$

变量：
- $\beta = 0.1$：weight threshold
- $R_z$：所有 bone 权重都 < β 的 vertex 比例（unbound vertices）
- $R_m$：受 >4 个 bone 影响的 vertex 比例（over-bound）

直觉：双重约束——不允许 vertex 没主（动画时不动），也不允许 vertex 太多主（LBS 工业标准 ≤4 bone，超过有 candy-wrap artifact）。

这两个 degenerate mode 都是 naive reward optimization 容易陷入的，所以专门设计这个 reward 防止。

#### R_mo：Deformation Smoothness

$$R_{mo} = \left(1 + s \cdot \mathbb{E}_{p \sim \mathcal{P}}\left[\max_{e \in \mathcal{E}}\left(1, \frac{l(\text{LBS}(e))}{l(e) + \varepsilon}\right)\right]\right)^{-1}$$

变量：
- $l(e)$：rest pose 下 edge $e$ 的 L2 长度
- $\text{LBS}(\cdot)$：Linear Blend Skinning 函数
- $p$：从 pose space $\mathcal{P}$ 采样的 pose
- $\mathcal{E}$：mesh edge 集合
- $s$：scaling
- $\varepsilon = 10^{-6}$：数值稳定

直觉：这是 end-to-end animation quality reward。随机 sample 5 个 pose，做 LBS 变形，检查 edge length 变化比例。skinning 错的话变形后 edge 会拉伸/压缩剧烈。

$\max(\cdot, 1)$ 只惩罚拉伸不惩罚压缩（作者选了非对称处理）。

**这个 reward 最重要**，因为它直接优化最终目标——动画质量，而不是中间 metric。

### GRPO Objective

$$\mathcal{L} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left[\frac{\pi_\theta(o_{i,t})}{\pi_{old}(o_{i,t})}, \text{clip}\left(\frac{\pi_\theta(o_{i,t})}{\pi_{old}(o_{i,t})}, 1-\epsilon, 1+\epsilon\right)\right] R_i - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$$

变量：
- $G=24$：group size
- $o_i$：第 $i$ 个采样序列
- $\pi_\theta$：当前 policy
- $\pi_{old}$：采样时 policy
- $\pi_{ref}$：参考 policy（supervised 训好的）
- $\epsilon=0.2$：clip ratio
- $\beta=0.1$：KL penalty 强度

这是 PPO 的 group-relative 版本：用 group 内 reward 归一化替代 critic，clip 防止 policy 偏离过大，KL penalty 保持接近 reference policy。

**关键 trick**：如果生成的 token 序列 invalid（无法 decode 成 rig），直接 $R=0$。避免无效探索浪费。

---

## 实验数据看效果

### Skinning 提升是最大亮点 (Table 4)

ModelsResource L1 Error：
- RigNet: 0.0573
- Puppeteer: 0.0321
- UniRig: 0.0381
- TokenRig (6 tokens, w/ GRPO): **0.0163**

相对 RigNet 提升 251%，相对 UniRig 提升 133%。这就是 paper claim 的 "98%-133% improvement"。

Articulation 2.0 Motion Loss：
- RigNet: 0.0915
- TokenRig: **0.0209**

降 4.4×，动画质量大幅提升。

### Skeleton 提升不如 skinning 大 (Table 3)

J2J Chamfer Distance:
- ModelsResource: TokenRig 2.857 vs UniRig 3.390（提升 15.7%）
- Articulation 2.0: TokenRig 2.515 vs Puppeteer 3.033（提升 17.1%）

paper claim 的 17%-22% 来自这里。

**Insight**：Skeleton 生成这块前人已经做得不错，TokenRig 主要突破在 skinning。

### GRPO 对 OOD 最有效 (Figure 8, 9)

GRPO 对标准 benchmark 提升不大（甚至个别指标略降），但对 OOD 模型（wings, tails, horns, capes）效果显著。

这符合 RL fine-tuning 的预期——supervised 学分布内能力饱和，RL 注入 geometric prior 帮外推。base model 经常忽略翅膀、斗篷、尾巴这种附件结构，GRPO 训完能准确放置 bone 和分配 skinning。

### Latent Space 有语义 (Figure 5)

t-SNE 显示 skin latent $L_W$ 按 anatomical part（Head, Hips, LeftLeg）聚类。

**这意味着 encoder 学到了"腿的 skinning pattern"这种抽象概念**，而不是 memorize 具体 vertex index。这就是为什么能 generalize 到不同 topology 的 mesh——"腿"的概念在不同 mesh 上共享。

### Data Augmentation 重要 (Table 6)

去掉 joint deletion 后 J2J 从 2.857 升到 3.077。模拟 topological imperfection 对 robustness 关键。in-the-wild mesh 经常有缺失的 bone、奇怪的结构，训练时见过类似的扰动才能扛住。

---

## 我的 Takeaway

### 1. Representation 决定 tractability

skinning 难不是因为网络不够强，是 continuous regression formulation 错了。离散化后变 tractable。这跟 Karpathy 你一直说的 "pick the right representation" 思想完全一致。

### 2. Sparsity 需要显式 inductive bias

Dice loss > BCE for <10% sparse targets。Medical segmentation 的老智慧拿到 3D 直接用。

### 3. Unified sequence 击败 decoupled pipeline

cross-modal dependency 通过 self-attention 自然建模，比 two-stage pipeline 更有表达力。Skeleton 和 skinning 互相影响，分开建模有 ceiling。

### 4. RL for OOD generalization

supervised 学分布内，RL 注入 geometric prior 帮分布外。GRPO 避免 critic 训练，工程友好。

### 5. 3D 的 "LLM-ification"

Qwen3 backbone, GQA, RoPE, Muon optimizer 全是 LLM 技术直接迁移。3D 领域越来越像 LLM 范式：discretize → tokenize → autoregressive → RL fine-tune。这跟 image (VAR, Image GPT)、audio (AudioLM, SoundStream) 走的路一样。

### 6. 工程美学

这篇 paper 没发明新架构，只是 reframe 问题，然后用成熟工具栈解决。这种 "right representation makes problem easy" 的做法是好的工程哲学。

---

## 想想局限性

作者自己说三点：
1. **Discrete latent vs continuous latent**：FSQ 在极难 skinning 场景仍有 gap。Continuous token (VAR-style, Li et al. 2024) 可能是未来方向。
2. **没有 user control**：当前 fully autonomous，artist 想指定 topology 没法做。未来应该加 interactive guidance。
3. **RL reward 是 geometric**：没考虑 physics。未来可以加 dynamics-based reward，让变形物理上更真实。

我再加一个：
4. **Codebook 设计是 manual**：$[8,8,8,5,5,5]$ 是手调的。能不能让 codebook size 也 learnable？
5. **Token 数量固定**：$T_D=32$ 是固定上限。能不能根据 bone 复杂度动态分配 token？复杂 bone 给多点 token，简单 bone 少给点。这跟 FlexTok 的 flexible length 思想一致。

---

## 参考 links

- Project page: https://zjp-shadow.github.io/works/SkinTokens/
- FSQ paper: https://arxiv.org/abs/2309.15505
- 3DShape2VecSet (VecSet backbone): https://arxiv.org/abs/2212.10419
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Muon optimizer: https://arxiv.org/abs/2502.16982
- Dice loss (Generalized Dice): https://arxiv.org/abs/1707.03237
- FlexTok (nested dropout inspiration): https://arxiv.org/abs/2506.05075
- Nested dropout original: https://proceedings.mlr.press/v32/rippel14.html
- RigNet (baseline): https://arxiv.org/abs/2005.00559
- UniRig (baseline, TOG 2025): https://arxiv.org/abs/2505.20557
- VQ-VAE original: https://arxiv.org/abs/1711.00937
- CVAE (Sohn et al. 2015): https://papers.nips.cc/paper/2015/hash/8d55a249e7baed5a1ba0d7f85b50f0a8-Abstract.html
- VAR (continuous token inspiration): https://arxiv.org/abs/2404.02905
- Puppeteer: https://arxiv.org/abs/2508.10898
- MagicArticulate (CVPR 2025): https://arxiv.org/abs/2504.06583
- Anymate dataset (Deng et al. 2025): SIGGRAPH 2025
- Auto-Connect (DPO for rigging): https://arxiv.org/abs/2506.11430
- SkinCells (Voronoi skinning): https://arxiv.org/abs/2506.14714

人话说完。这篇 paper 我觉得最有启发的是 **representation reframe 的力量**——skin matrix 那么吓人，token 化之后突然变得能用 LLM 工具栈直接搞。这种"换 representation 让问题变简单"的思维方式，在 3D 领域还能找到更多应用空间。

---

# SkinTokens: 用 Token 化思维解决 Rigging 的表示难题

## 1. 核心直觉：为什么这篇 paper 重要

这篇 paper 的核心 thesis 用一句话概括：**skinning 的瓶颈不是网络架构问题，而是 representation problem**。作者把一个 ill-posed 的连续回归任务（regress N×J 矩阵）重新 formulate 成一个 tractable 的离散 token prediction 任务。这个 reframe 带来的连锁效应非常深：它让 skeleton 和 skinning 可以在同一个 autoregressive sequence 里 joint modeling，还能接上 RL fine-tuning。

对 Karpathy 你来说，这其实就是 LLM 思维方式向 3D 领域的延伸——把连续信号离散化成 token，然后用 next-token prediction 的范式统一建模。类似 VQ-VAE 对图像/音频做的事情，但这里针对的是 skinning weights 的极端稀疏性。

Reference: 论文官方页面 https://zjp-shadow.github.io/works/SkinTokens/

---

## 2. Skinning 为什么本质上是一个 Representation Problem

### 2.1 稀疏性的数学现实

给定 mesh $\mathcal{M} = \{\mathcal{V} \in \mathbb{R}^{3 \times N}, \mathcal{F}\}$ 和 skeleton 有 $J$ 个 joints，skinning matrix $\mathcal{W} \in \mathbb{R}^{N \times J}$。

生产环境里 $N > 10^5$，$J > 10^2$，矩阵元素超过 $10^7$。但每个 vertex 通常只受 ≤4 个 joints 影响，所以非零元素最多 $4N$。

Table 1 的数据非常关键：
- VRoid Hub: avg N=1297.69, avg J=19.87, sparsity=7.40%
- Articulation 2.0: avg N=16929.78, avg J=95.56, sparsity=2.43%  
- ModelsResource: avg N=6247.05, avg J=34.46, sparsity=9.38%

**Intuition**: 稀疏度 <10% 意味着 90%+ 的矩阵元素是 0。用 MSE loss 训练时，网络只要全输出 0 就能拿到 90% 的"正确率"，gradient signal 完全被 trivial zero-prediction 主导。这就是为什么传统 regression 方法（RigNet、NeuroSkinning）会产生 noisy weights，在动画时出现 jarring artifacts。

### 2.2 为什么传统稀疏矩阵压缩不适用

vertex 的 ordering 是任意的（mesh 数据结构没有 canonical ordering），所以无法依赖 structured sparsity（比如 block sparsity、banded structure）来做传统压缩。必须用 **learned** 的方式捕捉 semantic sparsity。

这就是 SkinTokens 的动机：用 VAE 学一个 latent，把每个 bone 对应的 skinning weight vector $\mathcal{W}^* = \{w_{(\cdot), j}\}$ 压缩成少量 discrete tokens。

---

## 3. SkinTokens 的架构：FSQ-CVAE 深度解析

### 3.1 为什么选 FSQ 而不是 VQ-VAE

传统 VQ-VAE (Van Den Oord et al. 2017) 用 learnable codebook + commitment loss + codebook update。问题：
1. Codebook collapse：很多 code 永远不被使用
2. 需要辅助 loss（commitment loss, EMA update）
3. 训练不稳定

FSQ (Mentzer et al. 2023) 的思路完全不同：**不学 codebook，直接量化到 fixed grid**。每个 latent dimension 量化到最近的 level。例如 level sizes $[8, 8, 8, 5, 5, 5]$ 意味着总 codebook size = $8 \times 8 \times 8 \times 5 \times 5 \times 5 = 64000$，无需学习。

**Intuition**: 这就像把 continuous latent space 切成一个 6D grid，每个 grid cell 对应一个 token。没有 codebook collapse 问题，因为所有 cell 都"存在"，只是看 encoder 是否会映射到那里。Table 2 显示 utilization 86.2%，说明 encoder 学到了多样化的 latent distribution。

梯度通过 Straight-Through Estimator (STE) 传递：
$$L_D = \text{FSQ}(L_W) = \text{round}(L_W) + \text{sg}(L_W - \text{round}(L_W))$$
其中 $\text{sg}(\cdot)$ 是 stop-gradient，前向用 quantized value，反向 gradient 直接传给 continuous latent。

Reference: FSQ paper https://arxiv.org/abs/2309.15505

### 3.2 CVAE 的 Conditioning 设计

这是 **Conditional** VAE，不是普通 VAE。Conditioning 是 mesh geometry $\mathcal{M}$。

两个 VecSet encoder (Zhang et al. 2023, 3DShape2VecSet)：
- $E_M(\mathcal{M})$ → shape features（输入：uniformly sampled mesh points $\mathcal{P}_{\text{uniform}}$）
- $E_W(\mathcal{W}^*)$ → latent weight features $L_W$（输入：skinning weights）

注意：$E_M$ 只看 $\mathcal{P}_{\text{uniform}}$ 是为了 match inference condition（推理时没有 skinning GT）。

为什么用 VecSet 而不是 GNN？因为 mesh vertex ordering 任意，VecSet 把 mesh 当作 unordered point set 处理，permutation invariant。这和 PointNet/DeepSet 的思想一脉相承。

Reference: 3DShape2VecSet https://arxiv.org/abs/2212.10419

### 3.3 Dice Loss：稀疏重建的关键

这是 paper 里最有 insight 的设计之一。作者画了 Figure 3 的 gradient analysis：

Dice loss 公式：
$$\mathcal{L}_{\text{Dice}} = \sum_{j \leq J} 1 - \frac{2 \sum_i w_{\text{pred}_{i,j}} w_{i,j} + \varepsilon}{\sum_i w_{\text{pred}_{i,j}}^2 + \sum_i w_{i,j}^2 + \varepsilon}$$

变量解释：
- $i$ 索引 vertex，$j$ 索引 bone
- $w_{\text{pred}_{i,j}}$ 是预测 weight
- $w_{i,j}$ 是 ground truth weight
- $\varepsilon = 10^{-4}$ 防止除零

**Intuition**: 当 $w=0$（target 是零），BCE gradient 在 $w_{\text{pred}} \in (0, 1]$ 范围内很小，网络没有动力把 false positive 压下去。但当 $w > 0$（active region），Dice gradient 远大于 BCE gradient。这正好对冲了 90% 都是 zero 的 class imbalance——把监督信号集中在 10% 的 active region。

Ablation (Table 5) 验证：去掉 Dice loss 后 IoU 从 87.1% 降到 82.2%（VRoid Hub），ModelsResource 从 91.1% 降到 88.2%。这是显著的 degradation。

Reference: Generalized Dice loss https://arxiv.org/abs/1707.03237

### 3.4 Nested Dropout：Compositional Representation

借鉴 Flex-Tok (Bachmann et al. 2025) 和 Rippel et al. 2014，训练时随机取 token 序列的 prefix。这强迫前面的 token 携带最多信息，后面的 token 是 refinement。

**Intuition**: 类似 JPEG 的 coarse-to-fine 编码，或者 LLM 里 "重要信息放前面"。Table 2 显示 $T_D=4$ 就能达到 high fidelity，说明前 4 个 token 已经编码了大部分 skinning 信息。这让 inference 时可以根据计算预算选择 token 数量。

Reference: FlexTok https://arxiv.org/abs/2506.05075, Nested dropout original https://proceedings.mlr.press/v32/rippel14.html

### 3.5 Importance Sampling 的不对称设计

训练时 decoder 看到 $\mathcal{P}_{\text{uniform}} \cup \mathcal{P}_{\text{dense}}$，其中 $\mathcal{P}_{\text{dense}}$ 来自 GT skinning 非零区域。但 encoder $E_M$ 只看 $\mathcal{P}_{\text{uniform}}$。

**Intuition**: 这是 train-test mismatch 的故意打破——训练时让 decoder 有"作弊"的密集采样看 active region，但 encoder 不能看（因为推理时没有 GT）。这种 asymmetric 设计加速收敛，因为每个 batch 都能充分监督 active region，而不是被 90% 的 zero region 稀释。

### 3.6 最终 VAE Loss

$$\mathcal{L}_{\text{VAE}} = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{MSE}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}$$

注意没有 KL term！这其实更像 Conditional Autoencoder 而不是严格 VAE。作者保留了小 MSE term 用于 stability（BCE 在 weight 接近 0 或 1 时 gradient 较小，MSE 补足）。

---

## 4. TokenRig：统一序列建模

### 4.1 序列设计的关键 Insight

这是 paper 最美的部分。完整序列结构：

```
<bos> <type_1> dx_1 dy_1 dz_1 ... <type_k> dx_T dy_T dz_T 
D_{1,0} ... D_{1,T_D} ... D_{T,0} ... D_{T,T_D} <eos>
```

- 前半段：skeleton（joint 坐标 uniform quantize 成 integer tokens，每个 bone chain 用 `<type>` 前缀分类）
- 后半段：所有 bone 的 SkinTokens 顺序拼接

**Intuition**: 这个顺序非常重要。Skeleton 先生成完，然后 skinning 条件化在完整 skeleton 上生成。Transformer self-attention 让每个 skin token 都能 attend 到所有 joint 位置——这是全局 conditioning。对比之前的方法（RigNet 用 GNN local receptive field），TokenRig 能捕捉长程依赖，比如"手指的 skinning"知道"肩膀关节在哪里"。

### 4.2 为什么 Unified 优于 Decoupled

之前的工作（UniRig, Puppeteer, MagicArticulate）都是两阶段：先生成 skeleton，再用另一个网络回归 skinning。问题：
1. **Error propagation**: skeleton 错了，skinning 永远无法修正
2. **No mutual information**: skeleton 生成时不知道 surface deformation 需求
3. **Representation mismatch**: skeleton 用 discrete token，skinning 用 continuous regression，两个 modality 无法在同一框架内 joint optimize

TokenRig 把两者都变成 token，用同一个 Transformer 生成，cross-modal dependency 通过 self-attention 自然建模。

### 4.3 Backbone 选择

Qwen3-0.6B (Yang et al. 2025)：
- Grouped Query Attention (GQA): 减少 KV cache 内存
- Rotary Position Embedding (RoPE): 长序列外推好
- 0.6B 参数量：足够 capacity，又不会过大

**Intuition**: 这其实是把 LLM 直接拿来用，因为 skeleton + skin token 序列就是 1D discrete sequence，和 text token 没本质区别。0.6B 是 sweet spot。

Reference: Qwen3 https://arxiv.org/abs/2505.09388

### 4.4 混合 Optimizer

Muon (Liu et al. 2025a) 用于 attention layers，AdamW 用于其他参数。Muon 是基于 Newton-Schulz 迭代的 orthogonal update，对 attention 的 high-rank gradient 特别有效。

Reference: Muon https://arxiv.org/abs/2502.16982

---

## 5. RL Refinement：GRPO + 4 个 Reward

### 5.1 为什么需要 RL

Supervised next-token prediction 学到的是 training data 的统计分布。对 OOD（out-of-distribution）mesh，模型会 fallback 到 "average" solution，比如：
- 忽略 wings/tails/horns 等非标准结构
- bone 突出 mesh 表面
- skinning 覆盖不全

这些 failure mode 难以用 supervised loss 表达，但可以用 explicit geometric reward 编码。

### 5.2 为什么 GRPO 而不是 PPO/DPO

GRPO (Shao et al. 2024, DeepSeekMath) 的优势：
- 不需要 critic network（PPO 需要 value function）
- 不需要 preference data（DPO 需要 pairwise ranking）
- 用 group-relative baseline：同一 input 采样 G 个 outputs，reward 归一化后作为 advantage

**Intuition**: 这相当于让模型在"自己和自己比"中学习。对每个 mesh 生成 24 个 rig，谁 reward 高谁就是正样本。这避开了训 critic 的不稳定性。

Reference: GRPO https://arxiv.org/abs/2402.03300

### 5.3 四个 Reward 详解

#### (1) Volumetric Joint Coverage $R_{vj}$

$$R_{vj} = \frac{1}{V} \sum_{i=1}^{V} \exp\left(-\alpha \min_{j=1}^{J} \|v_i - J_j\|_2\right)$$

变量：
- $V$：occupied voxel 数量（mesh voxelized 成 $r^3=196^3$ grid）
- $v_i$：第 $i$ 个体素中心
- $J_j$：第 $j$ 个 joint 位置
- $\alpha = 0.05$：控制 exponential kernel 衰减速度

**Intuition**: 对每个 voxel，找最近的 joint，距离越小 reward 越大。这惩罚"遗漏肢体"——如果某个区域没有 bone，对应 voxel 贡献接近 0。作者把 $w_{vj}=5$ 设得最高，说明这是最重要的 reward。

#### (2) Bone-Mesh Containment $R_{vk}$

$$R_{vk} = \frac{1}{J \times (s+1)} \sum_{j=1}^{J} \sum_{i=1}^{s+1} \mathbb{I}[J_{j,i} \in \mathcal{V}]$$

变量：
- $J$：bone 数量
- $s$：每根 bone 上的采样点数
- $J_{j,i}$：第 $j$ 根 bone 上第 $i$ 个均匀采样点
- $\mathbb{I}[\cdot]$：indicator function，点在 voxelized mesh 内返回 1

**Intuition**: 直接惩罚 bone 突出 mesh。这是 hard constraint 的 soft 版本——采样点在 mesh 外就扣分。比直接判断"bone 与 mesh 相交"简单且可微性友好。

#### (3) Skinning Coverage and Sparsity $R_{sc}$

$$R_{sc} = 1 - \frac{1}{2} R_z - \frac{1}{2} R_m$$

$$R_z = \left(\frac{1}{|\mathcal{V}|} \sum_i \prod_{j=1}^{J} \mathbb{I}[\mathcal{W}_{i,j} < \beta]\right)^{\alpha_z}$$

$$R_m = \left(\frac{1}{|\mathcal{V}|} \sum_i \mathbb{I}\left[\left(\sum_{j=1}^{J} \mathbb{I}[\mathcal{W}_{i,j} > \beta]\right) > 4\right]\right)^{\alpha_m}$$

变量：
- $\beta = 0.1$：weight threshold
- $R_z$：所有 bone 权重都 < β 的 vertex 比例（unbound vertices）
- $R_m$：受 >4 个 bone 影响的 vertex 比例（over-bound）
- $\alpha_z, \alpha_m$：惩罚强度 hyperparameter

**Intuition**: 双重约束——不允许 vertex "无主"（unbound 会导致动画时 mesh 不动），也不允许 vertex "太多主"（>4 个 bone 是 LBS 的工业标准上限，超过会出现 candy-wrap artifact）。这两个 degenerate mode 都是 naive reward optimization 容易陷入的。

#### (4) Deformation Smoothness $R_{mo}$

$$R_{mo} = \left(1 + s \cdot \mathbb{E}_{p \sim \mathcal{P}}\left[\max_{e \in \mathcal{E}}\left(1, \frac{l(\text{LBS}(e))}{l(e) + \varepsilon}\right)\right]\right)^{-1}$$

变量：
- $l(e)$：rest pose 下 edge $e$ 的 L2 长度
- $\text{LBS}(\cdot)$：Linear Blend Skinning 函数
- $p$：从 pose space $\mathcal{P}$ 采样的 pose
- $\mathcal{E}$：mesh edge 集合
- $s$：scaling hyperparameter
- $\varepsilon = 10^{-6}$：数值稳定

**Intuition**: 这是 end-to-end 的 animation quality reward。随机 sample 5 个 pose，做 LBS 变形，检查 edge length 变化比例。如果 skinning 错了，变形后 edge 会拉伸/压缩剧烈，$l(\text{LBS}(e))/(l(e)+\varepsilon)$ 远离 1。用 $\max(\cdot, 1)$ 只惩罚拉伸不惩罚压缩（对称处理也行，作者选了非对称）。

这个 reward 最重要，因为它直接优化最终目标——动画质量，而不是中间 metric。

### 5.4 GRPO Objective

$$\mathcal{L} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left[\frac{\pi_\theta(o_{i,t})}{\pi_{old}(o_{i,t})}, \text{clip}\left(\frac{\pi_\theta(o_{i,t})}{\pi_{old}(o_{i,t})}, 1-\epsilon, 1+\epsilon\right)\right] R_i - \beta \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$$

变量：
- $G=24$：group size
- $o_i$：第 $i$ 个采样序列
- $\pi_\theta, \pi_{old}, \pi_{ref}$：当前/采样时/参考 policy
- $\epsilon=0.2$：clip ratio
- $\beta=0.1$：KL penalty 强度

这是 PPO 的 group-relative 版本：用 group 内 reward 归一化替代 critic，clip 防止 policy 偏离过大，KL penalty 保持接近 reference policy（防止 reward hacking）。

**关键细节**: 如果生成的 token 序列 invalid（无法 decode 成 rig），直接 $R=0$。这避免了无效探索。

---

## 6. 实验数据深度解读

### 6.1 Reconstruction Fidelity (Figure 4)

代码本 size $C=[8,8,8,6,5]=15360$ 时，$T_D=4$ tokens 就能达到 high IoU。这证明 skinning 信息高度集中——4 个 discrete token 就够编码一根 bone 的 skinning。

最终选 $C=[8,8,8,5,5,5]=64000$，是 compression 和 accuracy 的 sweet spot。

### 6.2 Compression Ratio (Table 2)

$T_D=32$ tokens，每 token 2 bytes（codebook index），总 64 bytes/bone。对比 FP16 baseline：6247 vertices × 34.46 bones × 2 bytes = ~430KB，压缩到 32×34.46×2 ≈ 2.2KB，ratio 183.74×。

**Intuition**: 这说明 skinning matrix 的 intrinsic information 远小于它的 raw size。从信息论角度，稀疏 4N 个非零值，每个值 log 精度，理论下界远小于 dense matrix 存储。

### 6.3 Skinning Metrics (Table 4) - 最核心结果

ModelsResource 上 L1 Error：
- RigNet: 0.0573
- Puppeteer: 0.0321
- UniRig: 0.0381
- TokenRig (6 tokens, w/ GRPO): 0.0163

相对 RigNet 提升 $(0.0573-0.0163)/0.0163 = 251\%$，相对 UniRig 提升 $133\%$，相对 Puppeteer 提升 $97\%$。这就是 paper claim 的 98%-133% improvement。

Articulation 2.0 上 Motion Loss：
- RigNet: 0.0915
- TokenRig (w/ GRPO): 0.0209

降低 4.4×，说明动画质量显著提升。

### 6.4 Skeleton Metrics (Table 3)

J2J (Joint-to-Joint Chamfer Distance):
- ModelsResource: TokenRig 2.857 vs UniRig 3.390 (提升 15.7%)
- Articulation 2.0: TokenRig 2.515 vs Puppeteer 3.033 (提升 17.1%)

B2B (Bone-to-Bone):
- ModelsResource: TokenRig 1.568 vs UniRig 1.890 (提升 17.1%)

paper claim 的 17%-22% 提升来自这些数字。

**Insight**: Skeleton 提升不如 skinning 提升大，因为 skeleton 生成已经相对成熟（autoregressive skeleton 模型已经存在），真正突破在 skinning。

### 6.5 GRPO 的 OOD 效果 (Figure 8, 9)

GRPO 对标准 benchmark 提升不大（Table 3, 4），但对 OOD（wings, tails, horns, capes）效果显著。这符合 RL fine-tuning 的预期——supervised 学到的 distribution 内能力已饱和，RL 注入 geometric prior 帮助外推。

### 6.6 Latent Space Semantics (Figure 5)

t-SNE 可视化显示 skin latent $L_W$ 按 anatomical part（Head, Hips, LeftLeg）聚类。这说明 encoder 学到了 **semantic structural prior**——"腿的 skinning pattern"这个概念，而不是 memorize 具体 vertex index。

**Intuition**: 这就是为什么模型能 generalize 到不同 topology——"腿"的概念在不同 mesh 上共享，即使 vertex 数量和连接方式不同。

### 6.7 Ablation: Data Augmentation (Table 6)

去掉 joint deletion 后 J2J 从 2.857 升到 3.077（Articulation 2.0 从 2.515 升到 2.818）。这说明模拟 topological imperfection 对 robustness 关键。

---

## 7. 局限与未来方向

作者自己指出三点：
1. FSQ discrete latent vs continuous latent VAE 在极难场景仍有 gap。Continuous token (Li et al. 2024 VAR-style) 可能是未来方向。
2. 当前是 fully autonomous，没有 user control。未来应该加 interactive guidance / template specification。
3. RL reward 是 geometric，未来可以加 physics-based reward（动态真实性）。

---

## 8. Build Intuition 的关键 Takeaway

1. **Representation determines tractability**: skinning 难不是因为网络不够强，是因为 continuous regression formulation 不对。离散化后变 tractable。
2. **Sparsity needs explicit inductive bias**: Dice loss > BCE for <10% sparse targets。这是 medical segmentation 老智慧应用到 3D。
3. **Unified sequence > decoupled pipeline**: cross-modal dependency 通过 self-attention 自然建模，比 two-stage pipeline 更有表达力。
4. **RL for OOD generalization**: supervised 学分布内，RL 注入 geometric prior 帮分布外。GRPO 避免 critic 训练，工程友好。
5. **Discrete token 让 3D 能用 LLM 工具栈**: Qwen3 backbone, GQA, RoPE, Muon optimizer 全是 LLM 技术直接迁移。这是 3D 领域的 "LLM-ification"。

---

## 9. 参考 Links 汇总

- Project page: https://zjp-shadow.github.io/works/SkinTokens/
- FSQ: https://arxiv.org/abs/2309.15505
- 3DShape2VecSet: https://arxiv.org/abs/2212.10419
- Qwen3: https://arxiv.org/abs/2505.09388
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Muon optimizer: https://arxiv.org/abs/2502.16982
- Dice loss: https://arxiv.org/abs/1707.03237
- FlexTok (nested dropout): https://arxiv.org/abs/2506.05075
- RigNet (baseline): https://arxiv.org/abs/2005.00559
- UniRig (baseline): https://arxiv.org/abs/2505.20557
- VQ-VAE original: https://arxiv.org/abs/1711.00937
- CVAE: https://papers.nips.cc/paper/2015/hash/8d55a249e7baed5a1ba0d7f85b50f0a8-Abstract.html
- VAR (continuous token): https://arxiv.org/abs/2404.02905
- Puppeteer: https://arxiv.org/abs/2508.10898
- MagicArticulate: CVPR 2025 https://arxiv.org/abs/2504.06583

这篇 paper 的 elegance 在于它没有发明新的 network architecture，而是重新思考了 representation，然后用成熟的 LLM 工具栈解决。这种 "right representation makes problem easy" 的思维方式，正是 Karpathy 你在 Tesla 和 OpenAI 一直强调的工程哲学。
