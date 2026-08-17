---
source_pdf: PEAfowl Perception-Enhanced Multi-View Vision-Language-Action for Bimanual
  Manipulation.pdf
paper_sha256: 57d2bede78ffe831afe32e4e10356f32380d81d9f077b5b3eee9e6f9b15423d8
processed_at: '2026-08-06T02:34:45-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 PEAfowl

好，我换个讲法，就像咱俩在白板前聊天那种。

---

## 这论文到底在解决什么问题

想象你教一个机器人两只手配合干活——比如把一个杯子放到架子上，左手得扶着架子，右手得拿杯子。这事难在哪？

**第一，两只手互相挡。** 你从正面看，左手挡住右手；从侧面看，右手又挡住左手。单个相机根本看不全，必须多装几个相机。但装了多个相机之后，传统做法就是把每个相机的画面"叠在一起"塞给网络——网络根本不知道"这个相机看到的杯子和那个相机看到的杯子是同一个东西"。它只是盲目地把四张图的 token 拼起来。一旦相机位置稍微动一下，或者光照变了，网络就懵了。

**第二，指令听不进去。** 你跟机器人说"把红色的方块堆起来"和"按颜色排序方块"，画面几乎一模一样，就是任务不一样。传统 VLA 把语言指令当成一个"背景音乐"——全局加一个 text 向量进去，让网络自己 figure out 哪个物体跟指令有关。在简单场景里还能 work，一旦桌上放了十几个东西，attention 就散了，网络不知道该看谁。

PEAfowl 说：**这两个问题我一起治。**

---

## 怎么治的——用大白话

### 几何那块：让网络知道"3D 空间里谁是谁"

传统做法的问题：每个相机独立编码，token 拼起来完事。没有 3D 对齐。

PEAfowl 的思路很直接：**给每个 token 预测一个 depth（深度）的概率分布，然后把它"抬"到 3D 空间里去。**

为什么是概率分布？因为你手里那个 RealSense 深度相机，对着反光表面、透明物体、远处，深度值经常是 missing 或乱跳。你要是 hard assign 一个 depth 值，一旦 sensor 出错，整个 3D 位置就错了。用分布的意思是说："我不确定这个 token 是在 30cm 还是 50cm 处，我两个都考虑，但 40cm 的概率最高。"这样 sensor 出错的时候网络有退路。

抬到 3D 之后干什么？**在 robot 的 base frame（机器人底座坐标系）里找邻居。** 如果相机 A 的某个 token 和相机 B 的某个 token 在 3D 空间里离得很近，那它们大概率看的是同一个东西。那就把信息融合一下——用 distance 做 softmax 权重，取最近的 top-16 个邻居做加权平均，再以一个可学习的 gate 残差加回去。

这事的直觉特别朴素：**两个相机看到同一个杯子的不同角度，那它们的 feature 应该互相 "借力"。** 如果一个相机被挡了，另一个相机还能看到，那被挡的那个 token 就可以从没被挡的 token 那里"借"信息过来。

训练的时候还有个小聪明：用一个大模型（CDM，Camera Depth Model）当老师，离线把训练数据的深度图都修好，然后用修好的深度图当 soft label 监督网络预测的深度分布。**但部署的时候完全不用这个老师，只用原始的 noisy depth。** 等于训练时让网络跟一个"靠谱的人"学怎么从不靠谱的传感器读数里推断真实深度，学到之后自己就能干活了，不需要老师在场。

这就像你跟一个老中医学徒，老中医在旁边告诉你"这个脉象其实是 X 不是 Y"，你慢慢学会了自己判断，以后出师了就不用老中医盯着了。

### 语言那块：让 text 主动去"问"视觉

传统做法：text 是一个全局向量，加到 visual feature 上，网络自己 figure out。问题是 multi-objective 场景下 attention 散掉。

PEAfowl 的做法：**让 text token 当 query，主动 cross-attend 到 CLIP 的 patch token 上，而且迭代 3 次。**

类比一下：传统做法像你进一个房间，脑子里想着"找个杯子"，然后你扫视整个房间，看到什么算什么。PEAfowl 的做法像你进房间后，先想"杯子大概在哪"，看一眼，发现桌上好像有，再想"桌上那个是杯子吗"，再看一眼确认，第三次想"杯子的把手在哪边"，再聚焦一下。三次迭代之后，你拿到一个很 focused 的视觉证据集。

而且 CLIP 是 frozen 的——不微调。为什么？因为 CLIP 在 web-scale 上学到的 vision-language alignment 太宝贵了，你要 fine-tune 它，在小数据集上很容易 catastrophic forgetting。不如让它当"特征提取器"，你在它上面搭一个轻量的 Perceiver readout 就够了。

每个视角都独立做这个 text-as-query readout，最后 concat 起来。为什么不先 fuse 再 query？因为不同视角看到不同的东西——左手 wrist camera 可能看到了一个被遮挡的物体，front camera 看不到。让每个视角独立 grounding，再汇总，比先汇总再 grounding 鲁棒得多。

---

## Policy backbone 怎么干活

这部分基本沿用 SEM 的设计，两个核心点：

**Joint-centric representation**：不用 end-effector pose，用 joint angle + 通过 forward kinematics 算出来的 joint position 和 orientation。为什么？两只 arm 的 coordination 在 joint space 里更自然——左臂的 joint 和右臂的 joint 是独立的，但它们的 spatial effect 是耦合的。在 joint space 预测，用 FK loss 保证 spatial 一致性，两头都顾上了。

**Diffusion decoder + coarse-to-fine**：预测 64 步 trajectory，先预测 8 步的 coarse 包络，再上采样到 16→32→64。部署时执行 32 步，receding horizon。这招在长 horizon 高维 action 空间里很关键——直接预测 64 步太碎了，先有骨架再填细节。

---

## 实验讲了个什么故事

### Simulation (RoboTwin 2.0)

9 个任务，从 short-horizon（放杯子）到 long-horizon（开微波炉、堆三个方块）。两个 setting：Clean（固定环境）和 Domain-Randomized（背景随机、distractor、光照、桌面高度、指令 paraphrase 全来）。

DR 下 PEAfowl 47.1%，最强 baseline SEM 24.1%，**差了 23 个点**。

最 dramatic 的是 long-horizon 任务：堆三个方块，SEM 在 DR 下只有 1%，PEAfowl 34%。为什么？堆三个方块需要反复在 occlusion 下维持 spatial belief——放第一个方块的时候两只手挡来挡去，放第二个的时候更乱，放第三个的时候最乱。GGMVF 的 cross-view aggregation 让网络在遮挡下能"借"其他视角的信息，text-as-query readout 让网络在视觉混乱中保持"我在堆方块不是在排序方块"的 grounding。

还有一个有意思的对比：Stack Blocks Three（堆方块）和 Blocks Ranking RGB（按颜色排序方块）。两个任务视觉极度相似，指令截然不同。π0 和 SEM 都在这两个任务上"偏科"——一个高一个低。PEAfowl 两个都高，说明 text-as-query 真的起到了 disambiguation 作用。

### Real robot

双臂 AgileX Piper，4 个 RealSense D435。为了增加难度，故意用 remap 缩小 FOV 减少视角重叠。

PEAfowl 68.3%，SEM 11.7%。**差了 56 个点**，比 simulation 的 gap 更大。为什么？real sensor 噪声比 sim 大太多了。SEM 的 depth head 在 real fine-tune 时直接发散（Figure 7），因为 noisy depth + multi-task 训练 destabilize 了 depth 学习。PEAfowl 的 local pairwise fusion + cross-view aggregation + depth distillation 三管齐下稳住了。

Depth distillation 单独贡献 31.6 个点（68.3 vs 36.7）。真实 depth sensor 在镜面、透明、远距离物体上 noise 极大，没有 teacher 监督根本学不出稳定的 depth distribution。

---

## 这论文给我什么启发

几个 take-away 我觉得超出了 bimanual manipulation 本身：

**Distribution over point。** 任何 noisy sensor 输入，都该想"能不能用分布表达不确定性"而不是 hard assign。Depth distribution、soft attention、diffusion policy——都是同一个哲学的体现。这在 robotics perception 里是通用 lesson。

**Inductive bias 在 data-scarce regime 下比 scale 更值钱。** PEAfowl 300M trainable params 干翻 RDT-1B 和 π0。在 manipulation 这种数据量远不如 NLP 的领域，"做对先验"比"堆参数"重要得多。这跟 ViT 早期在 JFT-300M 上才能赢 ResNet，但小数据上 ResNet 更强是同一个道理。

**Text-as-query 是 grounding 的正确打开方式。** 不只是 robotics，任何 vision-language task 里，text 主动 query visual feature 比 text 当 global conditioning 都更锐利。DET-R 的 object query、BLIP-2 的 Q-Former、Flamingo 的 cross-attention——都是这个哲学的变体。PEAfowl 把它用到了 multi-view 上。

**Privileged training + raw deployment。** 训练时用 teacher 给 privilege information，部署时不用。这个 asymmetric design 在 sim-to-real 里太实用了。DAgger、asymmetric actor-critic、privileged distillation 都是这一脉。CDM 只在训练时跑，test-time 零开销，这种 design 在工业部署上极度友好。

**Modality-specific encoder + lightweight fusion。** RGB 和 depth 的 statistics 天差地别，用一个 encoder 会互相干扰。分开编码，只在 depth-distribution head 的输入处做 local pairwise fusion，既利用了 depth 的几何提示，又不污染 RGB 的语义 prior。这种"保守 fusion"思想在 multi-modal learning 里很通用。

---

## 一句话总结

PEAfowl 说的事其实很简单：**在 bimanual manipulation 这种 perception-hard、data-scarce 的 setting 下，与其堆参数，不如把 geometric inductive bias 和 language grounding 这两件事做对。** Depth distribution 让你对 noisy sensor 鲁棒，cross-view 3D aggregation 让你对 occlusion 鲁棒，text-as-query iterative readout 让你对 instruction 鲁棒，depth distillation 让你 sim-to-real 鲁棒。四个鲁棒性叠起来，就是 +23 pp 的 simulation 提升和 +56 pp 的 real-world 提升。

如果你要 take one thing away from this paper：**uncertainty is your friend, not your enemy. Model it, don't hide from it.**

---

# PEAfowl 论文详解

Andrej 你好，这篇 paper 我读得很兴奋，因为它在几个关键 design choice 上都"做对了事情"。下面我按动机 → 架构 → 公式细节 → 实验 → 直觉联想的顺序展开。

## 1. TL;DR — 一句话直觉

PEAfowl 在 multi-view bimanual VLA 这个 setting 下，针对两个长期被忽视的 bottleneck 提出对应方案：
- **几何维度**：用 per-token depth distribution 做 differentiable 3D lifting，再在 robot base frame 里做 top-K cross-view neighbor aggregation，把 "view-agnostic token concatenation" 换成 "geometrically grounded cross-view fusion"。
- **语言维度**：用 Perceiver-style text-as-query readout（M=3 latent blocks）替代全局 text conditioning，让 text token 主动 cross-attend 到 frozen CLIP patch token 上，反复累积 instruction-relevant 的视觉证据。
- **训练 trick**：用一个 pretrained Camera Depth Model (CDM) 当 training-only teacher 监督 depth distribution head，test-time 不引入任何额外开销，但能把几何先验蒸馏进策略网络。

在 RoboTwin 2.0 的 Domain-Randomized setting 上，比最强 baseline SEM 提升 **+23.0 pp**（47.1% vs 24.1%）；real-robot 上 68.3% vs 11.7%。

Project page: https://peafowlvla.github.io/

---

## 2. Motivation — 为什么现有 VLA 在 bimanual + clutter 下崩盘

Bimanual manipulation 有三个特殊难点，paper 在 introduction 里讲得很清楚：

1. **High-dimensional tightly-coupled action space**：`a_t = [a_t^L; a_t^R] ∈ R^{d_a}` 把两条 arm 的 action concat 起来，sample complexity 爆炸。
2. **频繁 self-occlusion + inter-object occlusion**：两条臂互相挡、又被物体挡，单视角必然丢信息 → 必须用 multi-view。
3. **Fine-grained language instructions**：如 "Stack Blocks Three" vs "Blocks Ranking RGB"，视觉极度相似、语义任务截然不同，全局 text conditioning 会让 attention 被视觉 dominant term 拉走，grounding 变糊。

现有 multi-view VLA 的两个常见 design flaw：

- **View-agnostic concatenation**：每个 image 独立过 encoder，token 直接 stack/concat 进 policy head。完全没有 cross-view 3D correspondence，对 camera pose drift、calibration error、occlusion 敏感。
- **Global text conditioning**：text 当成一个 global vector 或者少量 token append 进 visual stream，attention 还是 vision-centric。多物体多任务场景里 attention 是 unfocused 的。

PEAfowl 的论点：**在数据有限的前提下，提升 perception efficiency 和 grounding fidelity 比堆参数更重要**。

---

## 3. Architecture 总览

整体可以拆成三块（参考 Figure 2）：

```
Multi-view RGB-D
    │
    ├──► [Geometry Branch] Swin-T (Grounding-DINO init) + ResNet-34 (depth)
    │         │
    │         ├──► Per-token depth distribution (B=128 bins)
    │         ├──► Differentiable 3D lifting → x_bar, g
    │         ├──► Pairwise RGB-D fusion (only in depth branch)
    │         ├──► Cross-view top-K=16 neighbor aggregation (base frame)
    │         └──► Geometry-enhanced tokens
    │
    ├──► [Language Branch] Frozen CLIP ViT-L/14
    │         │
    │         ├──► Patch tokens (attn-last, ClearCLIP style)
    │         ├──► Text tokens (K_txt=64)
    │         ├──► M=3 Perceiver-style latent blocks (text queries patches)
    │         ├──► Attention pooling → R=64 per view
    │         └──► Language-guided context sequence S
    │
    └──► [Policy Backbone] SEM-style
              │
              ├──► Joint-centric state encoder (J_t,i = [θ; p; q] ∈ R^8)
              ├──► Joint-graph attention (link-hop distances G)
              └──► Diffusion transformer (H=64, coarse-to-fine 8→16→32→64)
```

Trainable params 只有 ~300M。CLIP backbone 是 frozen 的，CDM teacher 只在 training 时跑（offline）。

---

## 4. Geometry-Guided Multi-View Fusion (GGMVF) — 详细公式解析

这是论文最有意思的部分。我逐段拆开讲。

### 4.1 Multi-view RGB-D Feature Extraction

- **RGB encoder**：Swin-T (depths [2,2,6,2], heads [3,6,12,24], window=7)，从 Grounding-DINO 初始化 → 强语义 2D prior。
- **Depth encoder**：ResNet-34（1 channel，base_channels=4）→ 轻量，专门处理几何。
- **Neck**：ChannelMapper 把多尺度特征统一到 d=256（RGB）和 d_dep=32（depth）。
- **Cameras**：V=4（front, head, left_wrist, right_wrist），输入 320×256。

为什么要 modality-specific encoder + 共享权重 across cameras？因为 RGB 和 depth 的 statistics 完全不同：RGB 是 dense semantic texture，depth 是稀疏几何信号。强行用一个 encoder 会互相干扰。Camera 之间共享则隐含 multi-view geometry consistency 假设。

### 4.2 Tokenization across Scales

把 L=4 级 feature pyramid flatten 成 token sequence：
- $\mathbf{T}_{\text{rgb}}^{(v)} = \{\mathbf{t}_{\text{rgb},n}^{(v)}\}$ 
- $\mathbf{T}_{\text{dep}}^{(v)} = \{\mathbf{t}_{\text{dep},n}^{(v)}\}$

token 中心坐标 $\mathbf{u}_n^{(v)}$（像素坐标）配合 known intrinsics/extrinsics 用于后续 backprojection。

### 4.3 Depth-Aware 3D Lifting（核心创新点之一）

关键 idea：**不要相信单点 depth 值，而是预测一个 depth 上的离散分布，softly 把 2D token lift 到 3D**。

对 view $v$ 的 token $n$，预测 $B=128$ 个 depth bin 上的分布 $\mathbf{p}_n^{(v)} \in \mathbb{R}^B$（depth range $[0.01, 1.2]$ m，linear bins）。然后：

$$
\bar{\mathbf{x}}_n^{(v)} = \sum_{i=1}^{B} p_{n,i}^{(v)} \, \mathbf{x}_{n,i}^{(v)}, \qquad \mathbf{g}_n^{(v)} = \sum_{i=1}^{B} p_{n,i}^{(v)} \, \phi(\mathbf{x}_{n,i}^{(v)}) \tag{2}
$$

变量解释：
- $\bar{\mathbf{x}}_n^{(v)} \in \mathbb{R}^3$：expected 3D anchor（在 robot base frame），用作 cross-view 匹配的"位置"。
- $\mathbf{g}_n^{(v)}$：depth-aware point embedding，用作后续 fusion 的 geometric context。
- $\mathbf{x}_{n,i}^{(v)} \in \mathbb{R}^3$：token center $\mathbf{u}_n^{(v)}$ 在 depth bin $d_i$ 处 backproject 出来的 3D 点（用 camera intrinsics + extrinsics 算）。
- $p_{n,i}^{(v)}$：第 $i$ 个 bin 的概率（softmax over B bins）。
- $\phi(\cdot)$：3D 坐标 → point feature 的投影（推测是 sinusoidal positional encoding 或 small MLP）。

**为什么这样设计**？直觉有三层：

1. **不确定性建模**：commodity depth sensor 在 reflective/transparent 表面、远距离、边缘处经常 missing 或 noise 巨大。单点 depth 会把噪声直接传到 3D；分布允许 model 表达 "我大概知道在 0.3~0.5m 之间但不确定"。
2. **可微性**：$\bar{\mathbf{x}}$ 对 $\mathbf{p}$ 是 weighted sum，gradient 可以 backprop 到 depth head → 整个 3D lifting 是 differentiable 的。
3. **Teacher distillation 的天然接口**：分布形式可以直接和 CDM 输出的 refined depth 做软标签 BCE，不需要重新设计 loss。

### 4.4 Pairwise RGB-D Token Fusion

这里有一个非常聪明的 design：**RGB-D fusion 只发生在 co-located token pair 上，且只在 depth-distribution 分支**。

$$
[\hat{\mathbf{r}}_n^{(v)}, \hat{\mathbf{d}}_n^{(v)}] = \text{PairAttn}\big([\mathbf{W}_r \mathbf{t}_{\text{rgb},n}^{(v)}, \mathbf{t}_{\text{dep},n}^{(v)}]\big) \tag{3}
$$

- $\mathbf{W}_r$：把 RGB token 投影到 depth feature dim（d_dep=32）。
- PairAttn：2-token multi-head attention（4 heads）+ FFN(32→128→32)。
- 输出 $\hat{\mathbf{r}}, \hat{\mathbf{d}}$ 分别是融合后的 RGB-side 和 depth-side 表示。

**为什么这么保守**？因为 RGB encoder 是 Grounding-DINO 预训练的，2D 语义 prior 非常宝贵。如果让 noisy depth 大规模干扰 RGB stream，会破坏预训练知识。限制 fusion 只在 depth-distribution head 的输入处，main RGB stream 保持原样 → 既能利用 depth 的几何提示预测更准的 depth 分布，又不污染下游的语义 token。

### 4.5 Cross-View 3D Neighbor Aggregation

这是把多视角"对齐"的关键步骤。直觉：**如果两个 token 来自不同相机但 3D anchor 在 base frame 里很近，它们大概率看的是同一片物理区域，应该 aggregate**。

Step 1：计算 base frame 中的 L2 距离：
$$
\delta_{n,m}^{(v,w)} = \|\bar{\mathbf{x}}_n^{(v)} - \bar{\mathbf{x}}_m^{(w)}\|_2
$$

Step 2：对每个 query token $(v,n)$，选 top-K=16 最近邻 $\mathcal{N}_n^{(v)}$（跨所有其他 view 的所有 token）。

Step 3：distance-softmax 权重：
$$
\alpha_{n,m}^{(v,w)} = \frac{\exp(-\delta_{n,m}^{(v,w)} / \tau)}{\sum_{(w',m') \in \mathcal{N}_n^{(v)}} \exp(-\delta_{n,m'}^{(v,w')} / \tau)} \tag{4}
$$
- $\tau = 0.08$：温度，越小越接近 hard nearest neighbor，越大越平均。0.08 是非常 sharp 的选择，说明作者希望"挑出几个最对的 token"。

Step 4：加权聚合 RGB token：
$$
\mathbf{h}_n^{(v)} = \sum_{(w,m) \in \mathcal{N}_n^{(v)}} \alpha_{n,m}^{(v,w)} \mathbf{t}_{\text{rgb},m}^{(w)} \tag{5}
$$

Step 5：gated residual：
$$
\tilde{\mathbf{t}}_{\text{rgb},n}^{(v)} = \mathbf{t}_{\text{rgb},n}^{(v)} + \gamma \, \mathbf{h}_n^{(v)} \tag{6}
$$
- $\gamma$：learnable scalar gate，init 0.5。让 network 自己学融合强度。

Step 6：最终 fusion via MLP：
$$
\text{output}_n^{(v)} = \text{MLP}(\tilde{\mathbf{t}}_{\text{rgb},n}^{(v)} \oplus \hat{\mathbf{d}}_n^{(v)} \oplus \mathbf{g}_n^{(v)})
$$

直觉：
- Soft aggregation 比 hard correspondence（如 superglue）更适合 policy learning，因为 depth 不准时 hard matching 会崩。
- Top-K=16 限制了 attention 复杂度（不是 full pairwise），N×K 而非 N²。
- Gated residual 让原始 RGB 信息始终保留，aggregation 只做"增补"。
- 几何 distance 当 routing signal 是 inductive bias，比纯 attention 学习成本低。

Figure 5 用 t-SNE 验证：aggregation 后同一物理区域的 cross-view token 确实 cluster 在一起，而 SEM 和 PEAfowl pre-aggregation 都还是 view-separated 的 cluster。

### 4.6 Depth Distillation — 让 noisy sensor 不再拖后腿

Training-only teacher：
- 用 Camera Depth Model (CDM, [Liu et al., 2025a]) 离线处理整个训练集，得到 refined depth $\tilde{\mathbf{D}}^{(v)}$。
- 把每个 pixel 的 refined depth 通过 linear interpolation 映射到最近的两个 bin center，得到 2-hot soft target $\mathbf{q}_n^{(v)}$。
- 按 token stride (8/16/32/64) average-pool 成 per-token target。
- 计算每个 token 内 valid pixel 比例当 validity weight $\omega_n^{(v)}$（threshold 0.5）。

Loss：
$$
\mathcal{L}_{\text{depth}} = \frac{1}{VN} \sum_{v \in \mathcal{V}} \sum_{n=1}^{N} \omega_n^{(v)} \cdot \text{BCE}\big(\mathbf{p}_n^{(v)}, \mathbf{q}_n^{(v)}\big) \tag{16}
$$

为什么这个设计漂亮：
- **Inference-time zero overhead**：CDM 从来不在 policy loop 里跑。
- **Soft target** 比 hard label 信息量更大（邻近 bin 之间有过渡）。
- **Validity weight** 自动忽略 missing region，避免 model 学着预测 "invalid"。
- **Plug-and-play**：任何能产生 refined depth 的 teacher 都能用，未来有更强的 model 直接换。

---

## 5. Language-Guided Multi-View Readout

### 5.1 Frozen CLIP Features

- **CLIP ViT-L/14**（完全 frozen，省参数也保留预训练对齐）。
- **Patch token $\mathbf{X}^{(v)}$**：从 attn-last 输出取，遵循 ClearCLIP [Lan et al., 2024]。直觉：last attention layer 的输出保留了更细粒度的 vision-language alignment，而 CLS token 主要是 global summary。
- **Text token $\mathbf{T}_{\text{txt}}$**：CLIP text encoder 输出，保留前 $K_{\text{txt}}=64$ 个，mask 掉 SOT/EOT/PAD。

公式：
$$
\mathbf{T}_{\text{txt}} = \text{CLIP}_{\text{txt}}(\ell) \in \mathbb{R}^{K_{\text{txt}} \times D_c} \tag{7}
$$
$$
\mathbf{X}^{(v)} = \text{CLIP}_{\text{img}}^{\text{attn-last}}(\mathbf{I}^{(v)}) \in \mathbb{R}^{N_p \times D_c} \tag{8}
$$
- $D_c$：CLIP embed dim（ViT-L/14 是 768）。
- $N_p$：patch 数量。
- $K_{\text{txt}}=64$：保留的 text token 数。

### 5.2 Perceiver-Style Text-as-Query Readout

这是对 OTTER [Huang et al., 2025] 的迭代升级。OTTER 用一次性 similarity score 高亮相关 patch，PEAfowl 用 M=3 个 latent block **迭代累积**。

初始化：
$$
\mathbf{Z}^{(v,0)} = \mathbf{T}_{\text{txt}} \tag{9}
$$

迭代（m=0,...,M-1）：
$$
\mathbf{Z}^{(v,m+1)} = \text{LatentBlock}\big(\mathbf{Z}^{(v,m)}, \mathbf{X}^{(v)}\big) \tag{10}
$$

每个 LatentBlock 包含：
1. **Cross-attention**：text latent $\mathbf{Z}$ 当 query，patch token $\mathbf{X}$ 当 key/value → text 主动"问"视觉有哪些证据。
2. **Latent self-attention**：text token 之间互相交流，整合不同 patch 抓到的信息。
3. **FFN** after each attention。
4. **ReZero gated residual**：
   $$\mathbf{z} \leftarrow \mathbf{z} + \alpha \cdot F(\text{LN}(\mathbf{z}))$$
   $\alpha$ init=0，训练时从 0 慢慢长起来 → 深层 stacking 稳定 [Bachlechner et al., 2021]。

最终：$\mathbf{Z}^{(v)} \triangleq \mathbf{Z}^{(v,M)} \in \mathbb{R}^{K_{\text{txt}} \times D_c}$。

### 5.3 Attention Pooling → Context Sequence

为了避免每个 view 都有 $K_{\text{txt}}=64$ 个 token 拖累 policy，attention pooling 把它压缩成 $R=64$ 个 context token：

$$
\mathbf{R}^{(v)} = \text{Pool}(\mathbf{Z}^{(v)}) \in \mathbb{R}^{R \times d} \tag{11}
$$
$$
\mathbf{R}_{\text{txt}} = \text{Pool}(\mathbf{T}_{\text{txt}}) \in \mathbb{R}^{R \times d}
$$

最终 language-guided context：
$$
\mathbf{S} = [\mathbf{R}_{\text{txt}}; \mathbf{R}^{(1)}; \ldots; \mathbf{R}^{(V)}] \in \mathbb{R}^{(R + V \times R) \times d} \tag{12}
$$

直觉：
- **Text as query** 比"text as global conditioning"锐利得多。前者是"我想找这个"，后者是"我带着这个背景看所有"。
- **迭代 M=3 次**让 grounding 有空间迭代 refine，类似 Perceiver IO 的设计哲学 [Jaegle et al., 2022]：用少量 latent 当 bottleneck 强迫 model 抽取最 relevant 的信息。
- **Frozen CLIP** 保留了 web-scale 视觉语言对齐，又避免了 fine-tune 带来的 catastrophic forgetting 和训练成本。
- **Per-view readout** 让每个相机视角独立 grounding，最后 concat。multi-view 的好处是不同视角能看到不同 occluded 物体，让每个视角都做 grounding 比先 fuse 再 ground 更鲁棒。

### 5.4 Ablation 验证（Table 4）

- Full PEAfowl DR: 47.1%
- w/ 1-step SimAttn（OTTER 风格）: 36.1% → 迭代很重要，尤其 DR 下
- w/o VG-Text（只用 instruction summary）: 38.9% → vision-grounded text token 显著 help
- 在 attribute/reference 敏感任务（BR-RGB, PEC）上 drop 最大

---

## 6. Policy Backbone（继承 SEM）

这部分基本沿用 SEM [Lin et al., 2025]。

### 6.1 Joint-Centric State Representation

每个 joint 编码成 8D token：
$$
\mathbf{J}_{t,i} = [\theta_{t,i}; \mathbf{p}_{t,i}; \mathbf{q}_{t,i}], \quad \mathbf{p}_{t,i} \in \mathbb{R}^3, \mathbf{q}_{t,i} \in \mathbb{R}^4 \tag{13}
$$
- $\theta_{t,i}$：joint angle（1D）
- $\mathbf{p}_{t,i}$：joint position in base frame（3D）
- $\mathbf{q}_{t,i}$：joint orientation as quaternion（4D）
- 总共 1+3+4=8

直觉：纯 joint angle 无法表达 spatial relation；通过 forward kinematics 加上 (p, q) 让 policy 能在 spatial frame 里 reason。对 bimanual coordination 尤其重要，因为两条臂的 coordination 是空间关系而非 joint space 关系。

### 6.2 Joint-Graph Attention

- 用 link-hop distance $\mathbf{G} \in \mathbb{R}^{N_j \times N_j}$ 当 graph structure（kinematic tree 上的拓扑距离）。
- 直觉：相邻 joint 物理上联动，attention 应该有这种 inductive bias。

### 6.3 Diffusion Action Decoder

- 输出 H=64 步 joint trajectory。
- Diffusion：DDPM train（1000 steps, squaredcos_cap_v2 schedule, predict sample），DPM-Solver test（10 denoising steps）。
- **Coarse-to-fine upsampling**：base chunk H/c=8，逐步上采样到 16→32→64，channel 256→128→64→8。
- Closed-loop execution：每次 inference 执行 $H_{\text{exec}}=32$ 步，receding horizon control。

直觉：coarse-to-fine 解决长 horizon + 高维 action 的扩散困难。先预测低频包络再补高频细节，类似 audio synthesis 中的 multi-band 思路。

---

## 7. Training Objectives

总 loss：
$$
\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_{\text{fk}} \mathcal{L}_{\text{fk}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} \tag{17}
$$

### 7.1 Diffusion Imitation Loss
$$
\mathcal{L}_{\text{diff}} = \|(\hat{\mathbf{J}} - \mathbf{J}) \mathbf{W}\|_2^2 \tag{14}
$$
- $\mathbf{W}$：对角权重矩阵，对每个 dim 不同 weighting（细节 appendix 没明说，推测对 position quaternion 给更高权重）。

### 7.2 Forward Kinematics Consistency
$$
\mathcal{L}_{\text{fk}} = \|\big(\text{FK}(\hat{\boldsymbol{\theta}}_{t:t+H-1}) - \mathbf{J}_{t:t+H-1}^{\text{pose}}\big) \mathbf{W}_{\text{fk}}\|_2^2 \tag{15}
$$
- 把预测的 joint angle $\hat{\boldsymbol{\theta}}$ 通过 forward kinematics 算回 (p, q)。
- 和 GT pose 比较 → 强制 joint angle 和 spatial pose 一致。

直觉：$\mathcal{L}_{\text{diff}}$ 在 joint space 上算，但 task 在 spatial space 上评价。FK loss 把两个 space 桥接起来，避免 model 学出 joint space 上 close 但 spatial 上 catastrophic 的解。这是 bimanual coordination 的关键。

### 7.3 Depth Distillation Loss
见前面 Eq. (16)。

---

## 8. Experiments 详解

### 8.1 Simulation Setup

- **Benchmark**: RoboTwin 2.0 [Chen et al., 2025]，bimanual manipulation 专门 benchmark，Aloha-AgileX embodiment，4-camera RGB-D。
- **9 tasks**：短 horizon (PBF, PEC)、中 horizon (HM, OL)、长 horizon (OM, SB3, SW3, BR-RGB, BR-Size)。
- **50 demos per task**，VLA 跨任务联合训练。
- **两个 setting**：
  - Clean：固定视觉/物理参数，模板化 instruction → in-distribution 评估。
  - Domain-Randomized (DR)：背景随机、distractors、lighting、texture、tabletop height ±0.05m、instruction paraphrase → 压力测试。
- **100 trials per task per setting**。

### 8.2 Baselines

- **Visuomotor**（per-task 训练）：ACT [Zhao et al., 2023]、DP [Chi et al., 2023]、DP3 [Ze et al., 2024]
- **Multi-task VLA**：π0 [Black et al., 2025]、RDT [Liu et al., 2025b]、SEM [Lin et al., 2025]

### 8.3 主要结果 (Table 1)

| Setting | DP | ACT | DP3 | π0 | RDT | SEM | **PEAfowl** |
|---|---|---|---|---|---|---|---|
| Clean Avg | 26.1 | 40.0 | 34.2 | 22.0 | 10.6 | 51.0 | **69.6** |
| DR Avg | 11.3 | 14.1 | 7.8 | 22.1 | 6.7 | 24.1 | **47.1** |

几个值得注意的点：

1. **Per-task visuomotor (DP, ACT, DP3) 在 DR 下几乎完全失败**。这说明 50 demos + per-task training 在 appearance/layout 变化下不够。需要 multi-task joint training + 更好的 perception。

2. **RDT 在 DR 下反而崩盘（6.7%）**，尽管 RDT-1B 是 bimanual 专门设计的 diffusion foundation model。作者解释是 RDT 主要用 2D RGB + multi-image concatenation，没有 strong geometric supervision，对 randomization 敏感。

3. **π0 表现中规中矩**（22.1%），靠 large-scale vision-language pretraining 提供一定 cross-scene generalization，但 instruction-distinct 任务（BR-RGB vs SB3）表现不稳定，反映 grounding 不够锐利。

4. **SEM 是最强 baseline**（24.1% DR），用 3D position embedding + joint-centric encoder。PEAfowl 在此基础上 +23.0 pp，主要 gain 来自两个分支联合作用。

5. **Long-horizon 增益最大**：
   - DR Long Avg: SEM 16.4% → PEAfowl 39.0%（+22.6 pp）
   - DR SB3: SEM 1% → PEAfowl 34%
   - 直觉：long-horizon 需要反复在 occlusion 下保持 spatial belief，GGMVF 帮助最大。

6. **BR-RGB vs SB3 的对比很有意思**：两个任务视觉极相似（都是堆 blocks），但 instruction 完全不同（按颜色排序 vs 简单堆叠）。π0 和 SEM 都在这两个任务间"偏科"，PEAfowl 同时高分（47% 和 34% in DR），说明 text-as-query readout 真正起到了 grounding 作用。

### 8.4 Generalization to Held-Out Tasks (Table 2)

| Task | Setting | π0 | RDT | SEM | PEAfowl |
|---|---|---|---|---|---|
| SB2 | Clean | 25 | 0 | 49 | **74** |
| SB2 | DR | 34 | 0 | 14 | **51** |
| SW2 | Clean | 83 | 38 | 93 | **97** |
| SW2 | DR | 79 | 23 | 65 | **82** |

SB2 是 SB3 的简化版（stack 2 blocks instead of 3），SW2 同理。在 9-task training set 上没见过，但 PEAfowl 在 DR 上从 SEM 的 14% 提到 51% → spatial 和 semantic information 能 transfer 到 unseen 但相关的 bimanual task。

### 8.5 Real-World Experiments (Table 3)

- **Platform**：dual-arm AgileX Piper。
- **6 tasks**：4 sim-to-real + 2 real-only（PS=Place Shoe, PBD=Put Bottles Dustbin）。
- **100 demos per task**，从 simulation-pretrained model fine-tune。
- **Camera trick**：故意用 remap-based resizing 缩小 effective FOV → 减少 cross-view overlap，让 multi-view fusion 更难。
- **10 trials per task**。

| Task | PEAfowl | PEAfowl w/o DD | SEM |
|---|---|---|---|
| SW3 | 100 (10/10) | 70 | 30 |
| HM | 30 | 10 | 0 |
| PBF | 60 | 30 | 10 |
| PEC | 80 | 50 | 20 |
| PS | 60 | 30 | 10 |
| PBD | 80 | 30 | 0 |
| **Avg** | **68.3** | 36.7 | 11.7 |

观察：
- PEAfowl vs SEM: **+56.6 pp** in real world（比 sim 的 gap 更大）→ real 噪声更大，PEAfowl 的 robust perception 优势更明显。
- Depth distillation 单独贡献 **+31.6 pp**（68.3 vs 36.7）。Real commodity sensor 噪声远大于 sim，distillation 几乎是必需。
- Figure 6 可视化：PEAfowl w/o DD 已经比 SEM 的 depth distribution 干净很多（local pairwise fusion + cross-view aggregation 的功劳），加 DD 后进一步 sharpen + complete missing region。

### 8.6 Ablations (Table 4)

| Variant | Clean | DR |
|---|---|---|
| PEAfowl | 69.6 | 47.1 |
| w/ 1-step SimAttn (OTTER-style) | 63.9 | 36.1 |
| w/o VG-Text | 62.3 | 38.9 |
| Baseline + GGMVF | 57.7 | 35.8 |
| Baseline | 44.7 | 31.3 |

几个关键 take-away：

1. **1-step SimAttn 在 Clean 下只掉 5.7 pp，DR 下掉 11.0 pp**。说明 clutter 和 scene shift 下迭代 grounding 是必须的，clean 下一次性 similarity 够用。
2. **w/o VG-Text 在 attribute/reference 任务（BR-RGB 47→31, PEC 70→60）掉得多**，vision-grounded text token 对 disambiguation 至关重要。
3. **Baseline + GGMVF** 在 DR 下从 31.3 提到 35.8（+4.5 pp），比 language branch 贡献小但显著。GGMVF 在 long-horizon 上贡献大（OM 17→26, SB3 1→7, BR-RGB 11→34）。
4. **Baseline depth-loss divergence**（Figure 7）：在 multi-task 训练下，baseline 的 depth loss 直接发散。GGMVF 通过 local pairwise fusion + cross-view aggregation 稳定 depth distribution 学习。

### 8.7 Multi-View Ablation (Table 7)

| Variant | DR Avg |
|---|---|
| Full (4 cameras) | 47.1 |
| w/o Front | 35.0 |
| Head-only | 26.8 |

观察：
- **Front view 贡献 +12.1 pp**：DR 下 global context 帮助 disambiguate target。
- **Wrist views 贡献 +8.2 pp**（35.0 → 26.8）：fine manipulation（HM, SB3）wrist view 几乎不可少。HM 从 26% → 2% 说明 wrist view 对 hanging mug 这种 fine bimanual 任务 critical。

---

## 9. 我对这篇 paper 的 intuition 联想

### 9.1 为什么 depth distribution 比 point cloud 好

DP3 [Ze et al., 2024] 用 point cloud，但 point cloud 是 hard decision：每个像素 hard assign 一个 depth。一旦 sensor noise，point cloud 直接错位。PEAfowl 用 distribution 做软 lifting，相当于 "每个 token 在 3D 空间里有一团概率云"。这件事让我联想到：

- **MVSNet** [Yao et al., 2018] 早就用 depth probability volume 做 multi-view stereo，PEAfowl 借鉴了这个 idea 到 VLA。
- **NeRF** [Mildenhall et al., 2020] 的 volume density 也是分布式的几何表达。
- **Diffusion model** 本身就是分布学习，policy 用 diffusion decoder 跟 depth distribution 在哲学上 self-consistent。

### 9.2 Perceiver-style 在 VLA 里的妙用

Perceiver IO 原本设计是为了处理 arbitrary structured input（图像、点云、序列等）做 asymmetric attention，用少量 latent token 当 bottleneck。PEAfowl 反过来用：text token 当 latent query，视觉 patch 当 input。这个 trick 让我想起：

- **Flamingo**：用 text cross-attends 到 visual feature，但 Flamingo 是 frozen LM 主导。
- **Q-Former** (BLIP-2)：learnable query 提取 visual feature 给 LM。PEAfowl 用 text 直接当 query，更 task-relevant。
- **DETR**：object query 主动从 image feature 里"抠"出 object。PEAfowl 的 text query 是 DETR query 的语义化版本。

### 9.3 Sim-to-Real 的关键 trick

Depth distillation 的设计让我想到几个相似思路：
- **Domain adaptation via teacher**：CDM teacher 在 sim 和 real 上都能用（supervised on synthetic + domain generalization），把 policy 的 depth head 推到 teacher 的水准。
- **Privileged learning**：训练时用 privileged 信息（CDM refined depth），部署时只用 raw sensor → 类似 DAgger、asymmetric actor-critic 的思路。
- **No test-time overhead** 是 deployment 友好的关键，比那些 runtime 跑大 model 的方案实用得多。

### 9.4 Joint-Centric vs Cartesian Policy

SEM backbone 的 joint-centric representation 配合 FK consistency loss，让我想到：
- **Operational Space Control**：经典机器人学里用 cartesian space 表示 task，joint space 表示 control，需要 Jacobian 桥接。PEAfowl/SEM 让网络在 joint space 预测，但通过 FK loss 让 spatial 一致 → 学习版的 OSC。
- **Diffusion Policy** [Chi et al., 2023] 原版用 end-effector pose，PEAfowl 用 joint angle → bimanual coordination 时 joint space 表达更直接（两条 arm 各自的 joint 独立），但要靠 FK loss 保证 task space 一致。

### 9.5 Limitations 我能想到的

1. **Per-task real-robot fine-tuning**：real-world 不是 multi-task 评估，每个任务单独 fine-tune 100 demos。能否做 multi-task real-robot 是 open question。
2. **300M params**：相对 RT-2、π0 这种 7B+ model 还是很小。scale up 会怎样？是否需要更多 task？
3. **H_exec=32 closed-loop**：每个 inference 执行 32 步，没有显式 temporal consistency across frames。对未来 video-conditioned VLA 怎么扩展？
4. **Static scene assumption**：no dynamic object modeling，对 deformable object / human-in-the-loop 场景如何？
5. **K_txt=64 固定**：长 instruction 怎么处理？多步 instruction（"先开抽屉再放进去再关上"）？
6. **CDM 依赖**：如果 CDM 在某些 real 场景失效（极暗、镜面），distillation 反而引入 bias。

### 9.6 跟最近 work 的 connection

- **3D-VLA** [Zhen et al., 2024]：3D scene representation + VLA，但用 explicit 3D backbone，PEAfowl 用 lightweight 3D lifting 更高效。
- **SpatialVLA** [Qu et al., 2025]：exploring spatial representations for VLA。
- **GeoVLA** [Sun et al., 2025]：3D representation in VLA。
- **VGGT** [Wang et al., 2025]：visual geometry grounded transformer，做 offline 3D。PEAfowl 借鉴思想但轻量化。
- **RVT-2** [Goyal et al., 2024]：multi-view 3D manipulation，但用 voxel 表达，PEAfowl 用 token 表达更适合 transformer policy。
- **π0** [Black et al., 2025]：VLA flow model，PEAfowl 在 bimanual 上超过它，说明在数据有限 setting 下，perception design 比 model scale 重要。
- **OTTER** [Huang et al., 2025]：text-aware visual feature extraction。PEAfowl 把 OTTER 的 1-step 升级成 M=3 iterative。

---

## 10. 总结：PEAfowl 的核心 design philosophy

我觉得 PEAfowl 给 VLA 设计带来几个重要 lesson：

1. **Distribution > Point**：对 sensor noise 鲁棒的最好办法是 model uncertainty。Depth distribution 比 point cloud、soft attention 比 hard matching、diffusion 比 deterministic regression 都是同一哲学。
2. **Inductive bias > brute force**：cross-view geometric aggregation 用 distance 当 routing signal，比 full attention 学习成本低得多。在数据有限 setting 下，先验是朋友。
3. **Iterative > One-shot**：text-as-query readout 迭代 M=3 次显著好于 1 step。深度 grounding 需要时间累积证据。
4. **Privileged training, raw deployment**：CDM teacher 只在训练时用，test-time 零开销。这种 asymmetric design 在 sim-to-real 上非常实用。
5. **Joint-centric for bimanual**：joint space 是 bimanual 的天然表达，FK loss 桥接 spatial task。
6. **Modality-specific encoder**：RGB 和 depth 不能强行融合，需要各自处理后轻量 fusion。保留预训练语义是关键。

PEAfowl 只有 300M trainable params，却大幅超越 1B 的 RDT 和大规模预训练的 π0，证明在 bimanual 这种 data-scarce + perception-hard 的 setting 下，"做对 inductive bias" 比 "scale up" 更重要。这个结论让我想起 Vision Transformer 早期，BiT 和 ViT 争论的也是"先验 vs scale"——在数据有限时先验赢，数据无限时 scale 赢。Bimanual manipulation 现在还在 data 有限阶段。

---

## Reference Links

- Project page: https://peafowlvla.github.io/
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- SEM: https://arxiv.org/abs/2505.16196
- OTTER: https://arxiv.org/abs/2504.15143
- Perceiver IO: https://arxiv.org/abs/2107.14795
- ClearCLIP: https://arxiv.org/abs/2407.12423
- Grounding DINO: https://arxiv.org/abs/2303.05499
- CDM (Camera Depth Model): https://arxiv.org/abs/2509.02530
- DP3: https://arxiv.org/abs/2403.03954
- RDT-1B: https://arxiv.org/abs/2410.09832
- π0: https://arxiv.org/abs/2410.24164
- ReZero: https://arxiv.org/abs/2003.04487
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT (ALOHA): https://arxiv.org/abs/2304.13705
- SpatialVLA: https://arxiv.org/abs/2501.03036 (估计链接)
- GeoVLA: https://arxiv.org/abs/2508.09071
- VGGT: https://arxiv.org/abs/2503.11651
- RVT-2: https://arxiv.org/abs/2407.02479
- MVSNet: https://arxiv.org/abs/1804.02505
- NeRF: https://arxiv.org/abs/2003.08934
- BLIP-2 (Q-Former): https://arxiv.org/abs/2301.12597
- Flamingo: https://arxiv.org/abs/2204.14198
- DETR: https://arxiv.org/abs/2005.12872

如果你想深入聊某个 component（比如 cross-view aggregation 的 temperature schedule、FK loss 的 weighting、或者 Perceiver block 的具体 attention 实现），我可以再展开。
