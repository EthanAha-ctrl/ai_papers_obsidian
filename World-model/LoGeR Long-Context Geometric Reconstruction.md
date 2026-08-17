---
source_pdf: LoGeR Long-Context Geometric Reconstruction.pdf
paper_sha256: d559283a31c26b9d8ccb5ca99f98153f887bfa2b48d31c303587f433b2eb2814
processed_at: '2026-08-05T15:45:30-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LoGeR

## 一句话说清楚这 paper 在干嘛

现在的 feedforward 3D reconstruction 模型（VGGT, π³ 这类）一次 forward 就能把几张图变成 dense 3D 点云 + camera trajectory，特别猛。但它们有个硬伤——**只能处理百来帧，视频一长就 OOM 或者崩掉**。LoGeR 要做的事就是：让这种 feedforward 模型能一口气吃下几千帧甚至上万帧的视频，输出整个几分钟视频的 globally consistent 3D 重建，完全不需要后端的 bundle adjustment 之类的优化。

数字上震撼一下：KITTI 上以前最好的 feedforward 方法 ATE 72.86m，LoGeR 干到 18.65m，**降了 74%**。训练只见过 128 帧，推理时能跑到 18846 帧的 VBR sequence 还 hold 得住。

---

## 这事为什么难——两个 "wall" 的故事

Paper 里反复强调两个 wall，我觉得这个 framing 很关键。

**Context Wall（架构墙）**：feedforward 几何模型靠 bidirectional attention 学到强 priors，但 attention 是 $O(N^2)$ 的，dense vision task 每帧要 H×W 个 token，500 帧的 VGGT 直接 OOM。你想 scale 到几千帧，attention 根本扛不住。

**Data Wall（数据墙）**：就算架构上解决了，训练数据也都是短序列 bubble——ScanNet 几百帧、室内小场景。你拿这种数据训出来的模型，inference 时碰到 KITTI 5km 长的 driving sequence 直接失效。Paper 里 Fig.3 就是证据：FastVGGT 架构上能跑更多帧，但在 VBR 大场景上完全崩掉。

这个 data wall 的观察我觉得是 paper 最重要的 insight 之一。光搞架构没用，你得同时搞数据。

---

## 别人的搞法为什么不够

长序列 3D 重建已经有几条路线，但各有问题：

**Recurrent 路线（CUT3R, TTT3R）**：把所有历史压进一个 hidden state，像 RNN 一样逐帧 streaming。问题是 hidden state 容量小、lossy，dense alignment 的高频细节会丢，相邻 chunk 边界处会有 artifact。TTT3R 还是 frame-wise 的，丢掉了 bidirectional 多帧推理的能力。

**Sparse/Causal attention 路线（FastVGGT, InfiniteVGGT）**：用稀疏 attention 或 causal attention 降低计算量，能跑更多帧。但本质上还是被 data wall 困住——训练数据短，长 sequence 上学不到 global consistency。

**SLAM 路线（DROID-SLAM, DPV-SLAM）**：有后端 optimization 处理 loop closure 和 global alignment，能跑长序列。但慢，且不是纯 feedforward。

LoGeR 的核心主张：**单一 memory 机制搞不定这件事，必须 hybrid**。

---

## LoGeR 的核心 trick——Hybrid Memory

直觉是这样：长序列 3D 重建同时需要三种 coherence，分别对应不同的 time scale：

1. **Intra-chunk 细节**：单个 chunk 内的 dense 几何，靠 bidirectional attention 搞定（直接用 π³ 的 priors）
2. **Adjacent chunk 边界**：相邻 chunk 之间的像素级 alignment，需要 **lossless** 的高频信息传递，靠 Sliding Window Attention（SWA）
3. **Global 结构**：几千帧的整体 scale 和 coordinate frame，需要 **compressed** 的长程 memory，靠 Test-Time Training（TTT）

关键是 2 和 3 的矛盾：lossless 要保留全部 token（memory 爆炸），compressed 会丢高频信息（边界 alignment 出问题）。所以 LoGeR 让 SWA 管"最近的过去"，TTT 管"全局的过去"，各司其职。

用比喻说：SWA 像短期工作记忆，你记得昨天发生的事的全部细节；TTT 像长期记忆，你把过去一年的经验压缩成几条"原则"，虽然记不清每天细节，但大方向 hold 得住。

### 架构上怎么实现

Video 被切成 chunks，每个 chunk 大约几十帧，相邻 chunk 重叠一帧。每个 residual block 的 forward 有 4 步：

**(1) Per-frame attention**（公式3）：每帧自己内部做 self-attention，提取 spatial features。这一步继承自 π³，初始化也用 π³ 的权重。

$$\mathbf{H}^{\mathcal{C}^m} \gets \mathbf{H}^{\mathcal{C}^m} + [\text{Attn}_{\text{frame}}(\text{LN}(\mathbf{H}^{\mathcal{C}_i^m}); \theta), | i \in \{1,\dots,n\}]$$

变量解释：
- $\mathbf{H}^{\mathcal{C}^m}$ 是 chunk m 的全部 token
- $\mathbf{H}^{\mathcal{C}_i^m}$ 是 chunk m 第 $i$ 帧的 token
- $\theta$ 是 slow weights（推理时 frozen）
- $[\cdot]$ 是 concat

人话：把每张图变成一个编码好的 spatial feature map。

**(2) SWA over 相邻 chunks**（公式4）：让当前 chunk attend 到前一个 chunk 的全部 token。

$$\mathbf{H}^{\mathcal{C}^m} \gets \mathbf{H}^{\mathcal{C}^m} + \text{Attn}_{\text{swa}}([\text{LN}(\mathbf{H}^{\mathcal{C}^{m-1}}), \text{LN}(\mathbf{H}^{\mathcal{C}^m})]; \theta)$$

变量解释：
- $\mathbf{H}^{\mathcal{C}^{m-1}}$ 是前 chunk 的 token，完整保留
- attention 只发生在 $\mathcal{C}^{m-1} \cup \mathcal{C}^m$ 上

人话：当前 chunk 直接"看"前一个 chunk 的全部细节，保证边界处无缝衔接。这一步是 lossless 的，因为前 chunk 的 token 完整保留。

工程细节：只在 4 个 block（第 6, 10, 14, 18 个）插 SWA，控制 memory。用 FlexAttention（Dong et al., https://arxiv.org/abs/2412.05496）实现，inference 时用 KV-cache。

**(3) TTT 的 apply + update**（公式5、6）：这是最关键的。TTT（Sun et al., https://arxiv.org/abs/2407.04620）的核心思想是：把 memory 当作一组 "fast weights"，在推理时用梯度下降持续更新，相当于一个"边推理边学习"的小网络。

Apply step（公式5）：
$$\tilde{\mathbf{H}}^{\mathcal{C}^m} = \mathbf{H}^{\mathcal{C}^m} + f_{W^m}(\text{LN}(\mathbf{H}^{\mathcal{C}^m}))$$

变量解释：
- $W^m$ 是截至 chunk m 的 fast weights（推理时持续被更新）
- $f_{W^m}$ 是 SwiGLU MLP，parameterized by $W^m$
- $\tilde{\mathbf{H}}^{\mathcal{C}^m}$ 是注入了历史 memory 的 token

人话：在处理当前 chunk 之前，先把"过去所有 chunk 的压缩 summary"通过 fast weights 注入到 token 里。相当于在每帧 representation 上加了一个"历史背景"调制。

Update step（公式6）：
$$W^{m+1} = \mathcal{U}(W^m; \mathbf{H}^{\mathcal{C}^m})$$

具体是 gradient-based update（公式1）：
$$W \gets W - \eta \nabla_W \mathcal{L}(f_W(\mathbf{k}), \mathbf{v})$$

变量解释：
- $\eta$ 是 learning rate
- $\mathbf{k}, \mathbf{v}$ 是当前 token 投影出的 keys/values
- $\mathcal{L}$ 让 $f_W(\mathbf{k}) \approx \mathbf{v}$

人话：处理完当前 chunk 后，用这个 chunk 的信息更新 fast weights，相当于把当前 chunk 的重要信息"写进记忆"。这个 update 在推理时真的做梯度下降——这是 TTT 和普通 RNN 的本质区别。

**为什么 TTT 比 RNN hidden state 强**：RNN 压成 $d$ 维向量，TTT 压成 $d \times d$ 矩阵，容量大几个数量级，且 gradient 更新比固定 rule 更 expressive。Paper 中 TTT head dim=512, expansion=4，所以 $W$ 是 $512 \times 2048$ 的矩阵，比 RNN state $h \in \mathbb{R}^{512}$ 大太多。

用 Muon optimizer（Jordan et al., https://github.com/KellerJordan/Muon）做 test-time update，这是工程上的关键选择——Muon 对 hidden layer 优化有特殊优势。

**(4) Chunk 内 bidirectional attention**（公式7）：

$$\mathbf{H}^{\mathcal{C}^m} \gets \tilde{\mathbf{H}}^{\mathcal{C}^m} + \text{BiAttn}_{\text{chunk}}(\text{LN}(\tilde{\mathbf{H}}^{\mathcal{C}^m}); \theta)$$

人话：在已经注入 memory 的 representation 上做 chunk 内多帧 geometric reasoning，输出 dense pointmap 和 pose。这一步是 π³ 的核心能力，直接继承。

### 为什么 Hybrid 必要——Ablation 的证据

Table 3 的数字：
- LoGeR full：ScanNet 1000f ATE = 0.107
- 去掉 TTT：0.162（涨 51%）
- 去掉 SWA：0.143（涨 34%）

Fig.10 的可视化更直观：
- 去掉 SWA：相邻 chunk 边界出现 misalignment artifact，局部 distortion
- 去掉 TTT：长程轨迹严重 drift，global scale 失守

两个 failure mode 是 orthogonal 的，所以必须 hybrid。

---

## Loss 的关键——Global Pointmap Loss

公式（10）：
$$\mathcal{L}_{\text{global}} = \frac{1}{N|\Omega|} \sum_{i=1}^{N} \sum_{p \in \Omega} \| \Pi(\hat{\mathbf{T}}_i, \hat{\mathbf{x}}_{i,p}) - \Pi(\mathbf{T}_i, \mathbf{x}_{i,p}) \|_1$$

变量解释：
- $\Pi(\mathbf{T}, \mathbf{x})$ 用 pose $\mathbf{T}$ 把 local point $\mathbf{x}$ 变到 world coordinate
- $\hat{\mathbf{T}}_i = [\hat{\mathbf{R}}_i | \hat{\mathbf{t}}_i] \in \text{SE}(3)$ 是预测的 frame $i$ 的 global pose
- $\mathbf{T}_i$ 是 GT pose

人话：local loss 只保证每帧自己合理，pose loss 只保证相对运动合理。但 long-context 要的是 world 坐标系下整体一致。Global loss 强行让 local pointmap 和 pose 在 world space 互相约束——预测的 pose 把 local 点变到 world，再和 GT world 点比。

这个 loss 在 chunk-wise 训练时尤其重要：它逼 TTT 必须把"global scale"信息编码进 fast weights，否则跨 chunk 的 $\Pi$ 会出错。换句话说，global loss 是 TTT 学会"锚定 global coordinate frame"的 supervision 信号。

总 loss：$\mathcal{L} = \mathcal{L}_{\text{local}} + \mathcal{L}_{\text{pose}} + \lambda_{\text{global}} \mathcal{L}_{\text{global}}$，其中 $\lambda_{\text{global}} = 1$。

---

## LoGeR* 的 Feedforward Alignment

paper 还提了个 LoGeR* 变体，加了个纯 feedforward 的 stitching 步骤。

公式（12）：
$$\tilde{\mathbf{T}}_t^{(m)} = \mathbf{A}_m \hat{\mathbf{T}}_t^{(m)}, \quad \forall t \in \mathcal{C}^m$$

alignment matrix：
$$\mathbf{A}_m = \tilde{\mathbf{T}}_k^{(m-1)} (\hat{\mathbf{T}}_k^{(m)})^{-1}$$

变量解释：
- $\hat{\mathbf{T}}_k^{(m)}$ 是当前 chunk 中 overlapping frame $k$ 的 raw predicted pose
- $\tilde{\mathbf{T}}_k^{(m-1)}$ 是前 chunk 中同一帧 $k$ 的 aligned pose
- $\mathbf{A}_m \in \text{SE}(3)$ 把当前 chunk 整体 rigid align 到前 chunk 坐标系

人话：相邻 chunk 有 1 帧重叠，这帧在两个 chunk 里都被预测了一次。理论上应该有相同 global pose，但实际有误差。用 SE(3) alignment 把当前 chunk 整体平移/旋转到上一 chunk 的坐标系，纯 feedforward，可 differentiable，训练时能一起 fine-tune。

对比 Pi3-Chunk baseline：Pi3-Chunk 还要先估计 SIM(3)（带 scale）alignment，因为 π³ 本身预测 up-to-scale，跨 chunk scale 不一致。LoGeR 因为 TTT 已经 anchor 住 global scale，只需要 SE(3)。这个区别很关键——Pi3-Chunk 的 scale 估计误差会指数累积，长序列上崩掉。

---

## Curriculum Training——为什么 TTT 训练不稳

Paper 报告 TTT 的 recurrent 训练不稳定，需要 curriculum：

**Stage 1**（H100, 25k steps）：48 frames，chunk size 从 12 → 4，chunks 数从 4 → 12
**Stage 2**（H200, 15k steps）：128 frames，chunk size 从 12 → 8，chunks 数 ~11 → 16

人话：开始时 recurrent step 少（chunks 少），让 TTT 先学会在少步情况下稳定传递信息；然后逐步增加 chunks 数，逼 TTT 学长程依赖。同时 chunk size 减小，控制每个 chunk 计算量。这个 schedule 类似 RNN 训练的 truncated BPTT 渐进展开。

Ablation 数字（Table 3）：去掉 curriculum 后 ScanNet 1000f ATE 从 0.107 涨到 0.133，TUM 从 0.050 涨到 0.062。

**为什么 TTT 训练难**：fast weights 在长 sequence 上做梯度下降，gradient 通过时间步回传，很容易梯度爆炸/消失。Curriculum 让模型先在短 horizon 学会稳定 update rule，再 extend 到长 horizon，类似 curriculum in RL。

---

## Data Mixture——打 Data Wall 的具体做法

Paper 用 14 个 dataset 的 mixture，关键是大尺度 navigation data 权重高：

- DL3DV: 17.89%
- TartanAirV2: 17.89%
- OmniWorld-Game: 17.89%
- TartanAir: 8.94%
- Waymo: 6.71%
- Virtual KITTI 2: 2.24%

Ablation（Table 3）：去掉 TartanAir, TartanAirV2, Waymo, VKITTI2, OmniWorld 这 5 个大尺度 dataset 后，ScanNet 1000f ATE 从 0.107 涨到 0.156（涨 46%）。证实 data wall 假设——需要 diverse long-horizon data 学长程几何推理。

人话：室内小场景数据训不出长序列能力，必须喂大量 outdoor/autonomous driving 这种大尺度 navigation 数据。这些数据提供"几公里 trajectory"的 supervision 信号，让 TTT 学会 anchor global scale。

---

## 实验结果的关键数字

### KITTI（Table 2）

LoGeR* 平均 ATE = **18.65m**，对比：
- TTT3R: 72.86m（降 74.3%）
- VGGT-Long（optimization-based）: 27.64m（降 32.5%）
- CUT3R: 91.62m

特别看 open-loop sequences（无 loop closure 可借）：
- 03（801f, 0.6km）：LoGeR* = 5.38m，TTT3R = 16.83m
- 04（271f, 0.4km）：LoGeR* = 1.95m，TTT3R = 3.98m
- 10（1201f, 0.9km）：LoGeR* = 10.11m，TTT3R = 33.58m

这些 sequence 没有 loop closure 纠正 drift，纯靠模型自己维持 global consistency。说明 TTT 真的学到了 global coordinate anchoring。

### VBR Benchmark（Table 6）

VBR 是 paper repurpose 的 benchmark（Brizi et al., https://ieeexplore.ieee.org/document/10610668），罗马真实 driving 视频，7 个 sequence，8815-18846 frames，1.4-11.5 km。这才是真正的 minute-level 评测。

LoGeR 平均 ATE = 5.40m，LoGeR* = 5.27m，TTT3R = 7.62m，**降 30.8%**。

最长 sequence ciampino_1（18846f, 5.2km）：LoGeR = 8.30m，TTT3R = 13.18m。长序列上优势更明显。

### Length Generalization

最 impressive：训练只用 128 帧，inference 能泛化到 19k frames。靠的是：
- TTT fast weights 推理时持续更新，theoretical infinite receptive field
- 每 5 个 chunks 做 state reset（防 error accumulation，参考 Ruiz & Gu, https://arxiv.org/abs/2410.05439）
- Reset 时配 feedforward alignment 保持连续性

### Inference Efficiency（Table 5）

A100 40GB 上 500 frames：
- Chunk size 64: 9.3 FPS, 27.2 GB
- Chunk size 32: 12.1 FPS, 18.1 GB

Memory 和 chunk size 成正比，证实架构真 linear cost。

---

## 几个我的 Intuition 和 Critical 观察

### 为什么 TTT 比 RNN state 表达力强

RNN 把 history 压成向量 $h_t \in \mathbb{R}^d$，容量 $O(d)$。TTT 压成矩阵 $W \in \mathbb{R}^{d \times d}$，容量 $O(d^2)$，且 gradient 更新比固定 rule 更 expressive。Paper 中 TTT head dim=512, expansion=4，$W$ 约 $512 \times 2048$，比 RNN state 大几个数量级。

但 TTT 也有局限：paper Discussion 诚实承认超过 training context length 会 drift，需要 periodic reset。这是 RNN/linear-attention 通病（Ruiz & Gu 2025），open problem。

### SWA 只在 4 个 block 的设计

为什么是 4 个 block（6, 10, 14, 18）？我推测是 compute budget trade-off：SWA 要保留前 chunk 全部 token，memory 不低。18 个 block 中稀疏插 4 个，保证每个深度层级有 alignment 信号，又不爆 memory。这个比例的 ablation paper 没给，是个可探索方向。

### 为什么 Pi3-Chunk 短序列好、长序列崩

Table 2 和 VBR 显示：Pi3-Chunk 在 KITTI 短 sequence（04, 03）上 ATE 比 LoGeR 还低，但在长 sequence（00, 02）严重 drift。

人话：Pi3-Chunk 用 SIM(3) alignment stitching，每次 align 都有 scale 估计误差。短序列 error 累积小，长序列 scale 误差指数增长。LoGeR 的 TTT 把 global scale 直接 anchor 在 fast weights 里，不需要每次重估——这是 TTT 的核心价值。

### 为什么 Chunk-wise 比 Frame-wise 强

TTT3R 是 frame-wise streaming，LoGeR 是 chunk-wise。区别在于：
- Frame-wise 丢掉了 bidirectional 多帧推理能力，每帧只看过去
- Chunk-wise 保留 bidirectional backbone 的强 priors，每个 chunk 内部还能 multi-view reasoning

对几何重建这种需要 multi-view consistency 的任务，纯 causal frame-by-frame 不够强。LoGeR 的设计本质是"用 chunk 大小换 bidirectional 能力"——chunk 够小则 recurrent，chunk 够大则 bidirectional reasoning。

### Length Generalization 的 Open Problem

Paper 诚实承认 TTT 超过 training length 会 drift，用 periodic reset 解决，但 reset 牺牲 long-term context。这是 RNN/linear-attention 类架构通病。

可能 future direction：
- 类似 NoPE 的位置编码策略
- Memory consolidation：reset 时把重要信息 distill 进 slow weights
- Hierarchical TTT：multi-timescale fast weights
- 更大的 training context length（硬件允许的话）

---

## 这个 Paper 在大图景中的位置

LoGeR 站在几个 trend 交汇点：

1. **Feedforward geometric foundation models**：DUSt3R (CVPR 2024, https://arxiv.org/abs/2312.14132), VGGT (CVPR 2025, https://arxiv.org/abs/2503.11651), π³ (ICLR 2026, https://arxiv.org/abs/2409.07120) → classical 多视图几何 distill 进 transformer，单 forward 出 dense geometry。

2. **Long-context architectures**：Mamba (https://arxiv.org/abs/2312.00752), Linear Attention (https://arxiv.org/abs/2006.16236), TTT (https://arxiv.org/abs/2407.04620), Longformer (https://arxiv.org/abs/2004.05150) → 长序列 memory-efficient 架构。

3. **Streaming 3D reconstruction**：CUT3R (https://arxiv.org/abs/2412.04603), TTT3R, StreamVGGT, Point3R (NeurIPS 2025) → feedforward 模型 extend 到 streaming。

LoGeR 独特定位：第一个在**纯 feedforward** 设定下处理 minute-level 长序列并超越 optimization-based SLAM 的工作。

---

## 总结

用最人话的方式说：

LoGeR 把长视频切成 chunk，每个 chunk 内部用强 bidirectional 模型搞 dense 几何，跨 chunk 用两种 memory 传递信息——SWA 管"最近细节"保证边界无缝，TTT 管"全局结构"防止 scale drift。训练时加 global loss 逼 TTT 学会 anchor global coordinate，用 curriculum 稳定 TTT 训练，用大尺度 navigation data 破 data wall。推理时训练 128 帧能泛化到 19k 帧，靠 periodic reset + feedforward alignment 防 drift。

核心 insight 是：**长序列 3D 重建是 multi-scale coherence 问题，不同 time scale 需要不同 memory 机制**。这个思想应该能 transfer 到 video understanding, world modeling, robotics 等其他 long-range spatio-temporal reasoning 场景。

---

## Reference Links

- Project page: https://LoGeR-project.github.io/
- VGGT: https://arxiv.org/abs/2503.11651
- π³: https://arxiv.org/abs/2409.07120
- TTT: https://arxiv.org/abs/2407.04620
- TTT3R: https://arxiv.org/abs/2412.04603 (近似)
- CUT3R: https://arxiv.org/abs/2412.04603
- VGGT-Long: https://arxiv.org/abs/2507.16443
- VBR dataset: https://ieeexplore.ieee.org/document/10610668
- TartanAirV2 / TartanGround: https://arxiv.org/abs/2409.11744
- Muon optimizer: https://github.com/KellerJordan/Muon
- FlexAttention: https://arxiv.org/abs/2412.05496
- Longformer: https://arxiv.org/abs/2004.05150
- Mamba: https://arxiv.org/abs/2312.00752
- Linear Attention: https://arxiv.org/abs/2006.16236
- Length generalization (Ruiz & Gu): https://arxiv.org/abs/2410.05439
- MoGe: https://arxiv.org/abs/2503.21717
- DUSt3R: https://arxiv.org/abs/2312.14132
- DROID-SLAM: https://arxiv.org/abs/2101.02700

---

# LoGeR: Long-Context Geometric Reconstruction 深度解析

## 1. Paper的Big Picture

LoGeR 要解决的核心问题是：如何把 feedforward geometric foundation models（如 VGGT, π³, DUSt3R 这类直接 predict pointmap 和 camera pose 的模型）从 bounded scene（百帧级别的短序列）scaling 到 minute-level 的长视频（几千到几万帧），并且**完全不需要后端优化**，纯 feedforward 推理。

这个目标之所以困难，paper 提出有两个 wall：

**Context Wall（架构层面）**：bidirectional attention 是学习强 geometric priors 的关键，但它的复杂度是 $O(N^2)$，在 dense prediction（每帧 H×W 个 token）下 memory 爆炸。比如 VGGT 在 500 帧时就 OOM 了。

**Data Wall（数据层面）**：现有训练数据都是 short-context "bubbles"（几十到一百多帧），即使架构上能处理更多帧，模型在 inference 时遇到 long-range dependencies 也会失败。Paper 中 Fig.3 展示了 FastVGGT 虽然 memory 上能跑更多帧，但在 VBR 这种大尺度场景上完全崩掉。

LoGeR 的核心 insight 是：**单一 memory 机制不够用**。Recurrent 方法（如 CUT3R）把所有 context 压进一个 lossy hidden state，会丢失 high-precision dense alignment 所需的信息；而 naive deterministic stitching 又缺乏 long-range memory 导致 scale drift。所以需要 **hybrid memory**：不同 time scale 用不同机制处理。

---

## 2. 核心 Architecture 详解

### 2.1 Chunk-wise Processing 的基本设定

Video $\mathcal{X} = \{I_t\}_{t=1}^T$ 被分成 M 个 chunks $\{\mathcal{C}^m\}_{m=1}^M$，相邻 chunk 只重叠一帧（minimal overlap）。这个设计有几个好处：

- 每个 chunk 内部仍然用 bidirectional attention（继承 π³ 的强 priors），保证 dense geometric fidelity
- 每个 chunk 的长度保持在 training distribution 内（比如 64 帧），绕过 data wall
- 跨 chunk 的信息传递完全交给 hybrid memory

### 2.2 Hybrid Memory Block 的四步结构

每个 residual block 的 forward pass 包含 4 个 sub-layer，paper 中公式 (3)-(7) 给出了精确定义。我逐个拆解：

**(1) Per-frame attention**（公式3）：

$$\mathbf{H}^{\mathcal{C}^m} \gets \mathbf{H}^{\mathcal{C}^m} + [\text{Attn}_{\text{frame}}(\text{LN}(\mathbf{H}^{\mathcal{C}_i^m}); \theta), | i \in \{1,\dots,n\}]$$

变量含义：
- $\mathbf{H}^{\mathcal{C}^m}$：当前 chunk m 的全部 token 序列
- $\mathbf{H}^{\mathcal{C}_i^m}$：chunk m 中第 $i$ 帧的 token 子序列
- $\text{LN}$：LayerNorm
- $\theta$：slow weights（frozen at inference，只有这些是预训练好的）
- $[\cdot]$：concatenation operator

这一步对每一帧独立做 self-attention，提取 spatial features。它和 π³ 中对应的 layer 是一样的，直接从 π³ 初始化。Intuition：先把每张图变成一个 well-conditioned 的 spatial feature map，相当于"帧内的视觉编码"。

**(2) Sliding Window Attention（SWA）**（公式4）：

$$\mathbf{H}^{\mathcal{C}^m} \gets \mathbf{H}^{\mathcal{C}^m} + \text{Attn}_{\text{swa}}([\text{LN}(\mathbf{H}^{\mathcal{C}^{m-1}}), \text{LN}(\mathbf{H}^{\mathcal{C}^m})]; \theta)$$

变量含义：
- $\mathbf{H}^{\mathcal{C}^{m-1}}$：前一个 chunk m-1 的 token 序列（已经处理过的）
- $\text{Attn}_{\text{swa}}$：sliding window attention，只 attend 相邻两个 chunk

这一步建立了 **lossless** 的跨 chunk 信息高速公路。关键设计：只在 4 个 block（第 6, 10, 14, 18 个 block）插入 SWA，其他 block 不插，控制 compute cost。Intuition：相邻 chunk 之间的 alignment 需要像素级精度（比如 stitching 边界），任何压缩都会引入 stitching artifact。SWA 保留了前一 chunk 的全部 token，相当于"短期的无损缓存"。

**实现细节**：paper 用 FlexAttention 实现 SWA（Dong et al., 2024, https://arxiv.org/abs/2412.05496），并且 inference 时用 KV-cache 避免重复计算前 chunk 的 tokens。

**(3) Chunk-wise TTT layer**（公式5、6）：

这是最关键的创新。TTT（Test-Time Training, Sun et al., 2024, https://arxiv.org/abs/2407.04620）的核心思想是：把 memory 当作一组 "fast weights"，通过梯度下降在 inference 时持续更新。

**Apply step**（公式5）：
$$\tilde{\mathbf{H}}^{\mathcal{C}^m} = \mathbf{H}^{\mathcal{C}^m} + f_{W^m}(\text{LN}(\mathbf{H}^{\mathcal{C}^m}))$$

变量含义：
- $W^m$：截至 chunk m 的 fast weights（在 inference 时持续被更新）
- $f_{W^m}(\cdot)$：parameterized by $W^m$ 的 fast-weight module，实现上是一个 SwiGLU MLP
- $\tilde{\mathbf{H}}^{\mathcal{C}^m}$：注入了 historical memory 后的 token representation

Intuition：当前 chunk 在被处理时，先把"过去所有 chunk 的压缩 summary"通过 $f_{W^m}$ 注入到 token 中。这一步类似 RNN 的 hidden state 应用，但表达力更强（$W^m$ 是一个矩阵而不是向量）。

**Update step**（公式6）：
$$W^{m+1} = \mathcal{U}(W^m; \mathbf{H}^{\mathcal{C}^m})$$

变量含义：
- $\mathcal{U}(\cdot)$：online update rule，paper 中使用 gradient-based update with self-supervised objective
- $W^{m+1}$：更新后的 fast weights，供下一个 chunk 使用

具体形式（回到公式1）：
$$W \gets W - \eta \nabla_W \mathcal{L}(f_W(\mathbf{k}), \mathbf{v})$$

变量含义：
- $\eta$：learning rate
- $\mathbf{k}, \mathbf{v}$：从当前 token 投影出的 keys 和 values
- $\mathcal{L}$：loss function，鼓励 $f_W(\mathbf{k}) \approx \mathbf{v}$

Intuition：TTT 把 KV cache "编码"进 $W$ 矩阵，相当于让一个 small network 学会"检索"历史信息。这比 RNN 的 hidden state 表达力更强，比 attention 的 KV cache 更省 memory。Paper 用 Muon optimizer（Jordan et al., 2024, https://github.com/KellerJordan/Muon）做 test-time update，这是关键工程细节——Muon 对 hidden layer 优化有特殊优势。

**(4) Chunk-wise bidirectional attention**（公式7）：

$$\mathbf{H}^{\mathcal{C}^m} \gets \tilde{\mathbf{H}}^{\mathcal{C}^m} + \text{BiAttn}_{\text{chunk}}(\text{LN}(\tilde{\mathbf{H}}^{\mathcal{C}^m}); \theta)$$

变量含义：
- $\text{BiAttn}_{\text{chunk}}$：chunk 内全部 frames 之间的 bidirectional attention

这一步是 π³ 风格的 geometric reasoning，在已经注入 memory 的 representation 上做 chunk 内的多帧 reasoning。Intuition：TTT 已经提供了"全局上下文 snapshot"，bidirectional attention 在这个 snapshot 上做局部 high-fidelity 推理，最终输出 dense pointmap 和 pose。

### 2.3 为什么 Hybrid 是必要的

Paper 中 Table 1 给出了 key trade-off：

| Mechanism | Compute | Local Context | Global Context |
|-----------|---------|---------------|----------------|
| Full Attention | $O(N^2)$ | Lossless | Lossless |
| SWA | $O(N)$ | Lossless | Limited |
| TTT / Linear Attn | $O(N)$ | Compressed | Compressed |
| **Ours (Hybrid)** | $O(N)$ | **Lossless** | **Compressed** |

LoGeR 同时拿到 local lossless + global compressed + linear cost。Fig.10 的 ablation 视觉化了这一点：
- 去掉 SWA：相邻 chunk 出现 misalignment artifact（局部 distortion）
- 去掉 TTT：长程轨迹严重 drift（全局 scale 失守）

这两个 failure mode 是 orthogonal 的，因此 hybrid 是必然选择。

---

## 3. Loss Functions 的细节

Paper 跟着 π³（Wang et al., 2026, ICLR）设计了三个 loss。我重点讲 global loss，因为这是 long-context 的关键。

### 3.1 Local pointmap loss（公式8）

$$\mathcal{L}_{\text{local}} = \frac{1}{N|\Omega|} \sum_{i=1}^{N} \sum_{p \in \Omega} \frac{1}{z_{i,p}} \| s^* \hat{\mathbf{x}}_{i,p} - \mathbf{x}_{i,p} \|_1$$

变量含义：
- $N$：监督用到的 frame 数
- $\Omega$：pixel 索引集合，$|\Omega| = HW$
- $\hat{\mathbf{x}}_{i,p} \in \mathbb{R}^3$：frame $i$ 在 pixel $p$ 处预测的 local 3D 点坐标
- $\mathbf{x}_{i,p} \in \mathbb{R}^3$：对应的 ground truth
- $z_{i,p}$：depth，用作 normalization（远处的点误差天然大，除以 depth 让 loss scale-invariant）
- $s^*$：per-sequence 的 scale（从 MoGe 学来的 trick，Wang et al., 2025c, https://arxiv.org/abs/2503.21717），因为预测 up-to-scale

除以 $z_{i,p}$ 是关键设计：让远处物体的大误差不至于主导 loss，否则模型会过拟合近处物体。

### 3.2 Relative pose loss（公式9）

$$\mathcal{L}_{\text{pose}} = \sum_{(i,j) \in \mathcal{P}} \left( \lambda_r \mathcal{L}_{\text{rot}}(\hat{\mathbf{R}}_{ij}, \mathbf{R}_{ij}) + \lambda_t \| s^* \hat{\mathbf{t}}_{ij} - \mathbf{t}_{ij} \|_{\text{Huber}} \right)$$

变量含义：
- $\mathcal{P}$：监督的 frame pair 集合（chunk 内 pairs + 跨 chunk overlap pairs）
- $\hat{\mathbf{R}}_{ij}, \mathbf{R}_{ij}$：预测和 GT 的相对旋转
- $\hat{\mathbf{t}}_{ij}, \mathbf{t}_{ij}$：预测和 GT 的相对平移
- $\lambda_r = 0.1, \lambda_t = 10$：经验权重，translation 权重大是因为 rotation 已经被 normalized

关键点：用 affine-invariant relative pose（不需要 reference view），加上 per-sequence scale $s^*$，这样训练时不需要绝对 metric scale 的监督。

### 3.3 Global pointmap loss（公式10）——这是 long-context 的核心

$$\mathcal{L}_{\text{global}} = \frac{1}{N|\Omega|} \sum_{i=1}^{N} \sum_{p \in \Omega} \| \Pi(\hat{\mathbf{T}}_i, \hat{\mathbf{x}}_{i,p}) - \Pi(\mathbf{T}_i, \mathbf{x}_{i,p}) \|_1$$

变量含义：
- $\Pi(\mathbf{T}, \mathbf{x})$：用 pose $\mathbf{T}$ 把 local point $\mathbf{x}$ 变到 world coordinate
- $\hat{\mathbf{T}}_i = [\hat{\mathbf{R}}_i | \hat{\mathbf{t}}_i] \in \text{SE}(3)$：预测的 frame $i$ 的 global pose
- $\mathbf{T}_i$：GT pose

Intuition：local loss 只保证每帧自己合理，pose loss 只保证相对运动合理。但 long-context 要的是**world 坐标系下整体一致**。Global loss 通过把 local pointmap 用预测 pose transform 到 world，再和 GT world point 比较，强行让 local prediction 和 pose prediction 在 world space 互相约束。这个 loss 在 chunk-wise 训练时尤其重要——它强制 TTT 必须把"global scale"信息编码进 fast weights，否则跨 chunk 的 $\Pi$ 会出错。

总 loss（公式11）：
$$\mathcal{L} = \mathcal{L}_{\text{local}} + \mathcal{L}_{\text{pose}} + \lambda_{\text{global}} \mathcal{L}_{\text{global}}$$

其中 $\lambda_{\text{global}} = 1$。

---

## 4. LoGeR* 的 Feedforward Alignment

Paper 还提出了 LoGeR*，一个带 feedforward alignment 的变体。这是很巧妙的工程 trick：

定义（公式12）：
$$\tilde{\mathbf{T}}_t^{(m)} = \mathbf{A}_m \hat{\mathbf{T}}_t^{(m)}, \quad \forall t \in \mathcal{C}^m$$

其中 alignment matrix：
$$\mathbf{A}_m = \tilde{\mathbf{T}}_k^{(m-1)} (\hat{\mathbf{T}}_k^{(m)})^{-1}$$

变量含义：
- $\hat{\mathbf{T}}_k^{(m)}$：当前 chunk m 中 overlapping frame $k$ 的 raw predicted pose
- $\tilde{\mathbf{T}}_k^{(m-1)}$：前一个 chunk 中同一帧 $k$ 的 aligned pose
- $\mathbf{A}_m \in \text{SE}(3)$：把当前 chunk 整体 rigid align 到前一个 chunk 坐标系的变换

Intuition：overlapping frame 在两个 chunk 中都被预测了一次，理论上应该有相同的 global pose。但实际预测有误差，所以用 SE(3) alignment 把当前 chunk 整体平移/旋转到上一 chunk 的坐标系。这是一个**纯 feedforward 的 stitching**，不需要任何 iterative optimization，完全 differentiable，可以在训练时一起 fine-tune。

对比 Pi3-Chunk baseline（Appendix A.5）：Pi3-Chunk 还需要先估计 SIM(3)（带 scale 的）alignment（公式13），因为 π³ 本身预测的 geometry up-to-scale，跨 chunk 的 scale 不一致。LoGeR 因为 TTT 已经 anchor 住了 global scale，所以只需要 SE(3)。

---

## 5. Curriculum Training 的细节

这是另一个关键工程贡献。Paper 报告 TTT 的 recurrent 训练不稳定，需要 curriculum：

**Stage 1**（H100 GPU, 25k steps）：
- 48 frames total
- chunk size 从 12 → 4（即 chunks 数从 4 → 12）
- overlap 从 3 → 1

**Stage 2**（H200 GPU, 15k steps）：
- 128 frames total
- chunk size 从 12 → 8（chunks 数从 ~11 → 16）
- overlap 从 3 → 2

Intuition：开始时 chunks 少（recurrent step 少），让 TTT 先学会在少步情况下稳定传递；然后逐步增加 chunks 数，逼 TTT 学会长程依赖。同时 chunk size 也减小，让每个 chunk 的计算量可控。这种 schedule 类似于 RNN 训练中的 truncated BPTT 渐进式展开。

Paper 中 Table 3 的 ablation 显示：去掉 curriculum 后 ScanNet 1000f 的 ATE 从 0.107 涨到 0.133，TUM 从 0.050 涨到 0.062。

---

## 6. 实验结果的关键 insight

### 6.1 KITTI（Table 2）

最 striking 的数字：LoGeR* 平均 ATE = **18.65m**，对比 TTT3R 72.86m，**降低 74.3%**。甚至超过了 optimization-based 的 VGGT-Long（27.64m）32.5%。

特别值得注意的是 open-loop sequences（01, 03, 04, 08, 10）：
- 03（801f, 0.6km, 无 loop）：LoGeR* = 5.38m，TTT3R = 16.83m
- 04（271f, 0.4km, 无 loop）：LoGeR* = 1.95m，TTT3R = 3.98m

这些 sequence 没有 loop closure 可以纠正 drift，纯靠模型自身维持 global consistency。这说明 TTT 真的学到了"global coordinate anchoring"，不是靠 loop closure 后处理。

### 6.2 VBR Benchmark（Table 6, Fig.4）

VBR 是 paper 重新 repurpose 的 benchmark（Brizi et al., 2024, https://ieeexplore.ieee.org/document/10610668），来自罗马的真实 driving 视频，7 个 sequence，8815-18846 frames，1.4-11.5 km。这是真正的 "minute-long" 评测。

LoGeR 平均 ATE = 5.40m，LoGeR* = 5.27m。对比 best baseline TTT3R 7.62m，**降低 30.8%**。在最长的 ciampino_1（18846f, 5.2km）上，LoGeR = 8.30m，TTT3R = 13.18m——长序列优势更明显。

### 6.3 Inference Length Generalization

这是 paper 最 impressive 的点：**训练只用 128 frames，inference 时能泛化到 19k frames**。这归功于：
- TTT 的 fast weights 在 inference 时持续更新，理论上 infinite receptive field
- 每 5 个 chunks 做 state reset（防止 error accumulation，参考 Ruiz & Gu 2025, https://arxiv.org/abs/2410.05439）
- Reset 时配合 feedforward alignment 保持连续性

### 6.4 Ablation（Table 3）

```
Method              ScanNet-500f  ScanNet-1000f  TUM-500f  TUM-1000f
LoGeR               0.087         0.107          0.033     0.050
w/o TTT             0.108         0.162          0.043     0.079
w/o SWA             0.115         0.143          0.039     0.053
w/o 5 large data    0.102         0.156          0.050     0.072
w/o curriculum      0.098         0.133          0.049     0.062
```

每个组件都 significant。w/o TTT 在 1000f 上掉得最厉害（0.107→0.162），证实 TTT 是 long-range 的核心；w/o 5 large datasets 也掉很多，证实 data wall 假设——TartanAirV2 等 large-scale navigation data 是必要的。

### 6.5 Inference Efficiency（Table 5）

A100 40GB 上 500 frames：
- Chunk size 64：9.3 FPS, 27.2 GB
- Chunk size 32：12.1 FPS, 18.1 GB

Memory 几乎和 chunk size 成正比，speed 也合理——这证实架构是真正 linear cost 的。

---

## 7. 我的几个 Intuition 和 Critical 观察

### 7.1 为什么 TTT 比 RNN state 强

RNN 把 history 压成一个向量 $h_t \in \mathbb{R}^d$，容量是 $O(d)$。TTT 把 history 压成一个矩阵 $W \in \mathbb{R}^{d \times d}$，容量是 $O(d^2)$，且通过 gradient 更新有更强的 expressivity。Paper 中 TTT head dim = 512，intermediate expansion = 4，所以 $W$ 大约是 $512 \times 2048$ 的矩阵，比 RNN hidden state 大几个数量级。

但 TTT 也不是万能：paper Discussion 部分诚实承认，超过 training context length 后 TTT 也会 drift，需要 periodic reset。这是 Ruiz & Gu 2025 揭示的 length generalization 问题，是 open problem。

### 7.2 SWA 只在 4 个 block 的设计

为什么是 4 个 block（6, 10, 14, 18）？我推测这是一个 compute budget 的 trade-off：SWA 需要保留前 chunk 的全部 tokens，memory cost 不低。在 18 个 block 中稀疏插入 4 个，能保证每个"深度层级"都有一个 alignment 信号，又不让 memory 爆炸。这个具体比例的 ablation paper 没给，是个值得探索的方向。

### 7.3 为什么 Pi3-Chunk 在短序列上更好，长序列上崩

Table 2 和 VBR 结果显示：Pi3-Chunk 在 KITTI 短 sequences（如 04, 03）上 ATE 比 LoGeR 还低，但在长 sequences（如 00, 02）上严重 drift。

Intuition：Pi3-Chunk 用 SIM(3) alignment 做 stitching，每次 align 都有一个 scale 估计误差。短序列 error 累积小，长序列 scale 误差指数增长。而 LoGeR 的 TTT 把 global scale 直接 anchor 在 fast weights 里，不需要每次重新估计——这是 TTT 的核心价值。

### 7.4 和 TTT3R 的对比

TTT3R（Chen et al., 2026, ICLR）是 paper 的主要 baseline，它用 confidence-based update 做 single-frame streaming。LoGeR 的优势在于：
- TTT3R 是 frame-wise，失去了 bidirectional 多帧推理的能力
- LoGeR 是 chunk-wise，保留 bidirectional backbone 的强 priors

这其实是一个 general insight：对 geometric reasoning 这种需要 multi-view consistency 的任务，纯 causal frame-by-frame 不够强，需要 bidirectional chunk + recurrent memory 的 hybrid。

### 7.5 关于 Length Generalization 的 Open Problem

Paper Discussion 诚实地承认：TTT 在超出 training length 后会 drift。这是 RNN/linear-attention 类架构的通病（参考 Ruiz & Gu, ICML 2025, https://arxiv.org/abs/2410.05439）。LoGeR 用 periodic reset 解决，但 reset 牺牲 long-term context。

可能的 future direction：
- 类似 NoPE（No Position Embedding）的位置编码策略
- Memory consolidation：reset 时把重要信息 distill 进 slow weights
- Hierarchical TTT：multi-timescale 的 fast weights

---

## 8. 与相关工作的 Context

LoGeR 站在几个 trend 的交汇点：

1. **Feedforward geometric foundation models**：DUSt3R (CVPR 2024, https://arxiv.org/abs/2312.14132), VGGT (CVPR 2025, https://arxiv.org/abs/2503.11651), π³ (ICLR 2026, https://arxiv.org/abs/2409.07120) → 把 classical 多视图几何 distill 进 transformer，单 forward 出 dense geometry。

2. **Long-context architectures**：Mamba (Gu & Dao, 2024, https://arxiv.org/abs/2312.00752), Linear Attention (Katharopoulos et al., 2020, https://arxiv.org/abs/2006.16236), TTT (Sun et al., 2024, https://arxiv.org/abs/2407.04620), Longformer (Beltagy et al., 2020, https://arxiv.org/abs/2004.05150) → 处理长序列的 memory-efficient 架构。

3. **Streaming 3D reconstruction**：CUT3R (CVPR 2025, https://arxiv.org/abs/2412.04603), TTT3R (ICLR 2026), StreamVGGT (Zhuo et al., ICLR 2026), Point3R (NeurIPS 2025) → 把 feedforward 模型 extend 到 streaming。

LoGeR 的独特定位：是第一个在**纯 feedforward** 设定下处理 minute-level 长 sequence 并超越 optimization-based SLAM 的工作。

---

## 9. 总结

LoGeR 的核心贡献可以这样概括：

1. **Conceptual**：指出单一 memory 机制不够，提出 hybrid memory（lossless local + compressed global）。
2. **Architectural**：SWA + TTT 的具体组合，每个 block 4 个 sub-layer 的精确设计。
3. **Training recipe**：curriculum training + 大尺度数据 mixture + global loss 三者配合。
4. **Evaluation**：引入 VBR 长序列 benchmark，推动 community 评测 minute-level 重建。
5. **Length generalization**：训练 128 帧，推理 19k 帧的实证。

它建立的 intuition 是：**长序列 3D 重建本质上是 multi-scale coherence 问题**，不同 time scale 需要不同 inductive bias 的 memory 机制，不能一刀切。这个思想应该可以 transfer 到 video understanding, world modeling, robotics 等其他需要 long-range spatio-temporal reasoning 的领域。

Reference link：
- Project page: https://LoGeR-project.github.io/
- VGGT: https://arxiv.org/abs/2503.11651
- π³: https://arxiv.org/abs/2409.07120
- TTT: https://arxiv.org/abs/2407.04620
- TTT3R: https://arxiv.org/abs/2412.04603 (近似，具体 ICLR 2026 paper)
- CUT3R: https://arxiv.org/abs/2412.04603
- VGGT-Long: https://arxiv.org/abs/2507.16443
- VBR dataset: https://ieeexplore.ieee.org/document/10610668
- TartanAirV2: https://arxiv.org/abs/2409.11744 (TartanGround related)
- Muon optimizer: https://github.com/KellerJordan/Muon
- FlexAttention: https://arxiv.org/abs/2412.05496
- Longformer: https://arxiv.org/abs/2004.05150
- Mamba: https://arxiv.org/abs/2312.00752
- Linear Attention: https://arxiv.org/abs/2006.16236
- Length generalization (Ruiz & Gu): https://arxiv.org/abs/2410.05439
- MoGe: https://arxiv.org/abs/2503.21717
- DUSt3R: https://arxiv.org/abs/2312.14132
