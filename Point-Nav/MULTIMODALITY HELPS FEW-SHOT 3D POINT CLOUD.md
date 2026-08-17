---
source_pdf: MULTIMODALITY HELPS FEW-SHOT 3D POINT CLOUD.pdf
paper_sha256: eafeb618dcff4a50382c81d7a75412c0e609f57e42e6d4da185a9faa65137ee8
processed_at: '2026-08-05T21:23:05-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍 MM-FSS

好，刚才那个版本太 "paper-style" 了，我换一种讲法，像在 coffee break 跟你 chat 一样。

---

## 这 paper 一句话到底是啥

FS-PCS（few-shot 3D point cloud segmentation）这个领域，之前所有人都在 "只给模型看 point cloud" 这个 setup 下卷。这篇 paper 说：兄弟，你 annotate 的时候本来就写了 class name，ScanNet 这种 dataset 拍 3D 的时候本来就有 RGB image，这俩东西都是免费的，干嘛不用？

然后他们就把 text（class name）和 2D image 这俩 "免费午餐" 都用上了，而且用得特别聪明 —— **inference 的时候根本不需要 image，只用 point cloud + class name**。涨点还特别猛，2-way 5-shot 直接 +8.7 mIoU。

就这么个事。

---

## 为啥这件事 actually make sense

你想啊，few-shot 的核心痛点是什么？**数据太少**。1-shot 就是给模型看一个 "chair" 的 point cloud，然后让它去 query scene 里找 chair。这一个样本能学到的 "chair 长啥样" 信息极其有限，尤其 3D point cloud 又稀疏又没 texture，模型基本是在瞎猜。

但是！如果你告诉模型 "你要找的是 chair" 这句话，模型脑子里（如果它有 LSeg/CLIP 这种 VLM prior 的话）立刻就激活了 chair 的语义邻居：stool、bench、sofa、armchair... 这条 text 通路携带的 "chair 概念" 信息量，远远大于一个稀疏 point cloud。

这就好比：你给一个小孩看一张模糊的 chair 照片问 "这是啥"，他可能猜不出来；但你跟他说 "这是家具，有四条腿，能坐"，他立刻就懂了。**Text 就是那个 "给 hint" 的过程**。

2D image 同理。Point cloud 的 sparsity 是公认的痛 —— 一个 chair 的 point cloud 可能就几千个点，还都集中在表面，detail 全丢。但同一把椅子的 RGB photo 有几百万 pixel，texture、color、shadow 全在。所以 2D feature 本质上是 point cloud feature 的 "高清版"。

问题是：few-shot inference 的时候你未必有 image。S3DIS 这个 dataset 根本没配 image。怎么办？

作者的招数：**pretraining 阶段让 3D feature 去 "模仿" 2D feature**。具体说，ScanNet 里 point cloud 和 image 是配对的（有 camera intrinsic 矩阵可以 project 3D point 到 2D pixel），那就用 LSeg 的 image encoder 提 2D feature，然后训 3D backbone 让它输出的 feature 跟对应 2D pixel 的 feature 对齐（cosine similarity loss）。

训完之后，3D backbone 就 "内化" 了 2D 的表达。你 inference 的时候只喂 point cloud，但模型吐出来的 feature 已经是 "2D-aware" 的了。这个 trick 其实 2DPASS (https://arxiv.org/abs/2207.06989) 和 OpenScene (https://arxiv.org/abs/2211.15694) 都玩过，但用在 few-shot segmentation 是第一次。

而且关键：这个 pretraining **不用任何 semantic label**，纯 feature alignment。所以学到的 weight 是 class-agnostic 的，可以从 ScanNet transfer 到 S3DIS。这就是作者说的 "cost-free" —— 零额外 annotation 成本。

---

## 模型咋搭的

架构看 Figure 2 那张图，我用大白话走一遍数据流。

### Step 0: 两条 feature 通路

Input point cloud 进来，先过一个 shared backbone（Stratified Transformer 前两个 block，https://arxiv.org/abs/2202.10339），出来 general feature。然后分两个 head：

- **IF head** (Intermodal Feature head): 输出 intermodal feature $F^i$，这个 head 在 pretraining 阶段被训练去 align LSeg 的 2D+text embedding space。维度 $D_t=512$（跟 LSeg 对齐）。Pretrain 完就冻结。
- **UF head** (Unimodal Feature head): 输出 unimodal feature $F^u$，纯 point cloud feature，维度 $D=192$。这个 head 在 meta-learning 阶段跟着 task 一起训。

为啥要俩 head？因为 IF head 的训练目标（跟 VLM 对齐）跟 few-shot task 的训练目标（discriminative segmentation）差太远，一起训会 destabilize。所以 IF head pretrain 完冻住当 "稳定 anchor"，UF head 跟 task 一起训当 "task adapter"。这跟 LoRA (https://arxiv.org/abs/2106.09685) 那种 "frozen base + learnable adapter" 哲学一样。

更深一层：**IF head 是 frozen 的，所以它没有 base class bias**（base bias 是 few-shot 模型在 base class 上 fully supervised 训练后产生的偏置，会在 novel class 场景里 "误激活" base class）。UF head 参与了 meta-learning，所以有 bias 但 task-adaptive。两条路互补，后面 TACC 会利用这点。

### Step 1: 算 correlation（support 跟 query 的关系）

Few-shot segmentation 的核心就是 "support 和 query 的关系"。具体做法（沿用 AttMPTI, https://arxiv.org/abs/2104.12000）：

对 support point cloud，按 label 分 foreground / background，每类用 farthest point sampling + clustering 提 100 个 prototype。这样 support 变成一组 "代表点"。

然后 query 的每个 point 跟这 100 个 prototype 算 cosine similarity，得到 correlation matrix。

IF feature 算一遍得到 $C^i$，UF feature 算一遍得到 $C^u$。两路 correlation。

### Step 2: MCF — 融合两路 correlation

公式 (Eq. 4):

$$C_0 = \mathcal{F}_{lin}(C^i) + \mathcal{F}_{lin}(C^u)$$

人话：两路 correlation 各自过一个 linear layer（把 100 个 prototype 的 sim score 投影成 D 维），然后 element-wise 相加。

就这么简单。作者 ablation (Table 6) 试了 1:1 和 0.5:1 两种加权，性能差不多，说明这个 fusion 很 robust，不需要精心调权重。

### Step 3: MSF — 用 text guidance refine correlation

这是最核心的创新。分三步：

**(a) 算 text-image affinity** (Eq. 5):

$$G_q = F_q^i \cdot T^\top$$

$T$ 是 LSeg text encoder 提出来的 class name embedding（包括 "background" 这个类）。$F_q^i$ 是 query 的 intermodal feature。因为 IF head 在 pretrain 时被训练去 align LSeg 的 visual feature，而 LSeg 的 visual 和 text 又是 aligned 的，所以 $F_q^i$ 跟 $T$ 在同一个 space，可以直接 dot product。

$G_q \in \mathbb{R}^{N_Q \times N_C}$ 就是每个 query point 对每个 class 的 "text-image alignment score"。你可以理解为：**不靠 support 的 visual cue，只靠 text 语义和 query 的 visual，模型觉得这个 point 属于每个 class 的概率**。

**(b) 算动态权重** (Eq. 6):

$$W_q = \mathcal{F}_{mlp}(\mathcal{F}_{expand}(G_q) \oplus C_k)$$

人话：把 $G_q$（text guidance）和当前 correlation $C_k$ 拼一起，过 MLP，输出每个 (query point, class) pair 的一个权重 scalar。

这个权重的意思是：**对这个 point，我该信 text 多少**。如果 support 和 query 的 chair 长得很不一样（visual cue 失效），那 $W_q$ 应该大，让 text guidance 起更大作用。如果长得一样（visual cue 够用），$W_q$ 就小，主要靠 visual correlation。

Fig 5 的可视化特别直观：bookcase 那行 support 和 query 差异大，$W_q$ heatmap 在 bookcase 区域明显高；table 那行 support 和 query 差不多，$W_q$ 比较均匀。

**(c) 加权融合 + refine** (Eq. 7-8):

$$C_k' = G_q \odot W_q + C_k$$
$$C_{k+1} = \mathcal{F}_{mlp}(\mathcal{F}_{attention}(C_k'))$$

把 text guidance 按权重加进 correlation，然后过 linear attention + MLP refine。K 个 block 级联（S3DIS 用 2 个，ScanNet 用 4 个）。

Linear attention 用的是 Katharopoulos 2020 (https://arxiv.org/abs/2006.16236)，计算效率比 softmax attention 高，适合 point cloud 这种 $N_Q$ 可能上万的场景。

### Step 4: Decoder 出 prediction

Refined correlation $C_K$ 过一个 KPConv (https://arxiv.org/abs/1904.08868) + MLP，出 prediction logit $P_q$。Cross-entropy loss 跟 ground truth $Y_q$ 算 loss，end-to-end 训。

### Step 5: TACC — test-time 校准（只 inference 时用）

这个我特别喜欢。公式 (Eq. 9):

$$\hat{P}_q = \gamma G_q + P_q$$

就是把 text guidance $G_q$ 按权重 $\gamma$ 加到 prediction logit 上。$\gamma$ 是啥？**support set 上 $G_s$ 预测的 IoU** (Eq. 10):

$$\gamma = \text{IoU}(\arg\max(G_s), Y_s)$$

人话：我先用 text guidance $G_s$ 对 support 做个预测，跟 support 的 ground truth $Y_s$ 算 IoU。如果 IoU 高，说明 "text-image 通路" 这个 episode 是 work 的，那 query 上也信 text guidance，$\gamma$ 大；如果 IoU 低，说明这个 episode 的 class 在 text-image 通路表现差，那就少用，$\gamma$ 小。

这玩意**完全 parameter-free，closed-form，不需要学任何参数**。就是一个 "per-episode 信号质量自评"。

Ablation (Table 3e) 证明：固定系数（1:1, 1:0.5）都不如 adaptive γ。1:0（只用 text guidance）最差，因为 text 通路单独不够 robust。必须 $\gamma G_q + P_q$ 互补。

5-shot 时有 5 个 $\gamma$（每个 shot 算一个），用 max aggregation（Table 5 ablation 证明 max 比 min 好 1.3%，跟 mean 差不多）。

---

## 结果咋样

S3DIS（没 image 的 dataset，靠 ScanNet pretrain 的 IF head transfer 过去）:

| Setting | 前 SOTA COSeg | MM-FSS | 涨幅 |
|---------|--------------|--------|------|
| 1-way 1-shot | 47.77 | 52.09 | +4.32 |
| 1-way 5-shot | 50.41 | 54.21 | +3.80 |
| 2-way 1-shot | 38.07 | 44.30 | +6.23 |
| 2-way 5-shot | 41.49 | 50.16 | +8.67 |

ScanNet（有 image，但 inference 也不用）:

| Setting | COSeg | MM-FSS | 涨幅 |
|---------|------|--------|------|
| 1-way 1-shot | 42.01 | 44.73 | +2.72 |
| 1-way 5-shot | 46.61 | 50.07 | +3.46 |
| 2-way 1-shot | 29.03 | 39.21 | +10.18 |
| 2-way 5-shot | 35.51 | 44.09 | +8.58 |

两个 pattern 很明显：

1. **N-way 越大涨越多**：1-way 涨 3-4%，2-way 涨 6-10%。直觉：class 越多越需要 disambiguate，text semantic prior 的 "锚点" 作用越大。单 class 时 visual cue 勉强够，多 class 时没 text 锚点模型容易懵。

2. **S3DIS 也涨**：虽然 S3DIS 没 image，但用 ScanNet pretrain 的 IF head 直接 transfer 过去，依然受益。证明 IF head 学到的是 scene-general 的 "3D-to-VLM alignment"，dataset-agnostic。

最关键的 ablation 是 COSeg†：给前 SOTA COSeg 换上同样的 2D-aligned backbone，结果几乎没涨（47.21 → 47.77）。**光有好 backbone 不够，必须要有 MCF + MSF + TACC 这套 fusion 机制才能把 multimodal 信息榨出来**。这是 paper 里最 "defensive" 的一个 ablation，堵住了 "你涨点只是因为 backbone 更好" 这个 reviewer 必问的问题。

---

## 我的几点吐槽和联想

### 1. Text encoder 可以更强

paper 用 LSeg 的 text encoder，就是 CLIP ViT-B/32 量级。现在 2024-2025 年了，text encoder 早就不止这个水平。E5-Mistral (https://arxiv.org/abs/2401.00368)、voyage-3、OpenAI text-embedding-3-large 这些都强得多。如果把 LSeg 换成更强的 VLM（比如 SigLIP, https://arxiv.org/abs/2303.15343；或 EVA-CLIP, https://arxiv.org/abs/2303.14689），估计还能再涨一波。paper 在 Appendix Table 4 试了 LSeg vs OpenSeg，差不多，但都是 2022 年的 VLM，没用最新的。

### 2. 2D 蒸馏的 loss 太朴素

Appendix B 说 pretraining 就用 cosine similarity loss 做点-pixel 对齐。现在 contrastive learning 这么成熟，用 InfoNCE 或者 token-level contrastive (像 DINO, https://arxiv.org/abs/2104.14294) 应该能蒸馏出更 rich 的 2D prior。paper 没往这个方向试。

### 3. TACC 的理论深度

$\gamma = \text{IoU}(G_s, Y_s)$ 这个设计很 pragmatic，但理论上没保证。考虑极端情况：某个 episode 的 support 是 outlier（比如视角特别怪的 chair），$G_s$ 在 support 上 IoU 很低，$\gamma$ 接近 0，TACC 退化成不用 text guidance。这种情况下其实 query 上 text guidance 可能还是 work 的（query 是正常视角），但 TACC 误判了。

更 principled 的做法可能是用 Bayesian uncertainty estimation 或者 conformal prediction (https://arxiv.org/abs/2107.07511) 来估 $G_q$ 的 confidence。不过那就复杂多了，不如 IoU 简洁。作者选 IoU 应该是 "simplicity beats complexity" 的工程判断。

### 4. 跟 3D LLM 时代的对接

现在是 2025 年，PointLLM (https://arxiv.org/abs/2312.00925)、LLaVA-3D、GPT-4Point 这些 3D-LLM 都起来了。MM-FSS 用的还是 LSeg 这种 "老 VLM"。如果让一个 3D-LLM 在 inference 时直接读 point cloud + class name prompt，然后输出每个 point 的 class probability，是不是就直接替代 MCF + MSF 的作用了？

我觉得短期内还不能。因为 few-shot 的关键是 "support-to-query knowledge transfer"，LLM 直接读 query 不一定能利用 support 的 few-shot 信息（除非 in-context learning，但 point cloud 的 in-context 不好做）。MM-FSS 的 episodic meta-learning 框架在 support 利用上更 explicit。但长期看，3D-LLM + few-shot prompt engineering 是个值得探索的方向。

### 5. 跟 SAM-3D 的潜在结合

SAM (https://arxiv.org/abs/2304.02643) 出来后 3D 版本也有几个（SAM3D, https://arxiv.org/abs/2305.06322；PointSAM 之类）。如果用 SAM-3D 提供 promptable mask，再让 MM-FSS 的 text guidance 来做 mask 的 class assignment，可能 open-set few-shot segmentation 就有新玩法。这是这篇 paper 的 natural extension。

### 6. 这个 setup 在 autonomous driving 上的想象空间

Waymo / nuScenes 这种 dataset 本来就有 RGB camera + LiDAR + HD map 三模态。现在做 few-shot novel class segmentation（比如突然要识别 "construction cone" 这种新类），完全可以套 MM-FSS 的 setup：text = "construction cone"，2D image 在 pretrain 蒸馏，inference 只用 LiDAR + text。而且 driving 场景的 class name 通常更标准化（"vehicle", "pedestrian", "cyclist"），text prior 更稳定。这是一个很有 industrial potential 的方向。

### 7. 为啥 "N-way 越大涨越多" 这事值得深想

这个 pattern 其实暗示了一个更深的现象：**multimodal prior 的价值不是恒定的，而是跟 task difficulty 强相关**。单 class 时 visual cue 的 signal-to-noise 还够，text prior 的 marginal value 小；多 class 时 visual cue 不够 disambiguate，text prior 的 marginal value 暴涨。

这跟 CLIP 在 ImageNet zero-shot 上的发现类似 (https://arxiv.org/abs/2103.00020)：class 数越多，text prior 的 disambiguation 价值越大。所以 MM-FSS 在 2-way 上的大涨其实是在验证这个 universal 规律。

**预测**：如果有人做 5-way 或 10-way FS-PCS，MM-FSS 的相对优势会更明显。但 5-way FS-PCS 的 benchmark 估计得新建，现有 dataset 的 class 数量不够撑起来。

---

## 最后唠叨一句

这篇 paper 的核心 insight 其实特别简单：**few-shot 数据稀缺的时候，与其在单模态里卷 representation，不如看看 dataset 里还有啥 free modality 没用上**。

这个 framing 比方法本身更有价值。因为方法是可替换的（LSeg 可以换 SigLIP，KPConv 可以换 Point Transformer v3，TACC 可以换更 fancy 的 calibration），但 "找 free modality" 这个思维方式是 transferable 的。

你做 perception 的时候，问自己三个问题：
1. 我的 dataset 采集时还有啥 signal 顺手 got 了（IMU? GPS? timestamp? audio?）？
2. 这些 signal 在 annotation 阶段需要额外成本吗？
3. 这些 signal 能不能 pretrain 蒸馏进主通路，让 inference 时不需要它们？

如果三个答案都是 "yes / 免费 / 能"，那就有 paper 可写。MM-FSS 就是把这三个问题在 FS-PCS 上回答了一遍。

---

相关参考：
- 原 paper: https://arxiv.org/abs/2502.18837
- COSeg (前 SOTA): https://arxiv.org/abs/2403.09431
- AttMPTI (FS-PCS 开山): https://arxiv.org/abs/2104.12000
- LSeg: https://arxiv.org/abs/2011.12766
- CLIP: https://arxiv.org/abs/2103.00020
- 2DPASS (2D 蒸馏思路): https://arxiv.org/abs/2207.06989
- OpenScene (3D VLM): https://arxiv.org/abs/2211.15694
- Stratified Transformer (backbone): https://arxiv.org/abs/2202.10339
- KPConv (decoder): https://arxiv.org/abs/1904.08868
- Linear attention: https://arxiv.org/abs/2006.16236
- LoRA (frozen+adapter 哲学): https://arxiv.org/abs/2106.09685
- DINO (contrastive 蒸馏): https://arxiv.org/abs/2104.14294
- SigLIP (更强 VLM): https://arxiv.org/abs/2303.15343
- SAM: https://arxiv.org/abs/2304.02643
- PointLLM: https://arxiv.org/abs/2312.00925
- Conformal prediction (calibration 理论): https://arxiv.org/abs/2107.07511
- S3DIS: http://buildingparser.stanford.edu/
- ScanNet: http://www.scan-net.org/

希望这个 "人话版" 更好消化。如果你想挑某一块继续深挖（比如 MSF 的 dynamic weighting 跟 cross-attention 的本质区别，或者 TACC 的 Bayesian interpretation），随时说。

---

# MM-FSS: Multimodality Helps Few-Shot 3D Point Cloud Semantic Segmentation 深度解读

Andrej 好，这篇 paper 我读得很开心，因为它在 FS-PCS（few-shot 3D point cloud semantic segmentation）这个相对小众但越来越重要的领域里，做了一个特别 "obvious in hindsight" 的 contribution：**别人只点 cloud，他把 class name 和 2D image 这两个 "free lunch" 模态都捡起来用了，并且做到了 inference 时不需要 2D image**。这种 setup 设计的优雅程度值得花时间消化。下面我从 intuition 到 formula 一层一层剥给你看。

---

## 1. 一句话定位这篇 paper 的 contribution

在 FS-PCS 里，过去的 SOTA（如 COSeg, AttMPTI, QGE）全部只用 point cloud 单模态，作者提出一个 **cost-free multimodal setup**：

- **Textual modality** (class names 如 "chair", "wall"): meta-learning 和 inference 都用，annotation 阶段就免费拿到
- **2D image modality**: **只在 pretraining 用，implicit 蒸馏到 3D feature 里**，meta-learning 和 inference 完全不需要 image

围绕这个 setup，作者设计了 MM-FSS 模型，三大组件：
- **MCF** (Multimodal Correlation Fusion): 融合 intermodal + unimodal 两路 correlation
- **MSF** (Multimodal Semantic Fusion): 用 text embedding 作为 semantic guidance 动态加权 refine correlation
- **TACC** (Test-time Adaptive Cross-modal Calibration): 用 support set 的 IoU 作为自适应指示器 γ，test-time 校准 prediction，缓解 base class bias

最终在 S3DIS 上 1-way 1-shot +4.3 mIoU，2-way 5-shot +8.7 mIoU；ScanNet 上 2-way 1-shot 直接 +10.2 mIoU，2-way 5-shot +8.6 mIoU。N-way 越多收益越大，这本身就是一个有意思的 signal —— 暗示多模态知识在 novel class 越多时越能发挥组合优势。

paper link: https://arxiv.org/abs/2502.18837 (ICLR 2025)

---

## 2. 核心 intuition：为什么 multimodality 对 few-shot 特别重要？

作者在 intro 里援引了神经科学的多条证据 (Meltzoff & Borton, 1979; Kuhl & Meltzoff, 1984; Quiroga et al., 2005; Nanay, 2018)，论证人类认知是 inherently multimodal 的，不同模态的同一概念通过 **synergistic neurons** 形成强对应。

这里我自己的 read 是：few-shot 场景下，模型见到 novel class 的样本极少（1-shot 甚至更少），单模态 visual cue 几乎不足以支撑 robust 的 prototype 构建。如果引入 textual prior（"chair" 这个词在 LLM/VLM 的 embedding space 里已经携带了大量 chair 的语义邻居信息，比如 stool, bench, sofa 等），就相当于给模型一个"语义锚点"，让它在 visual cue 模糊时 fallback 到 text 语义。这跟 CLIP-style alignment 的 motivation 是一脉相承的 (https://arxiv.org/abs/2103.00020)。

更进一步，2D image modality 比 point cloud 有更 dense 的 texture/appearance，而 point cloud 的 sparsity 是众所周知的痛点 (https://arxiv.org/abs/2202.10339)。所以 2D 视觉特征本质上是 3D 特征的"高分辨率增强版"。但是 few-shot setting 下，2D image 在 inference 时未必可得（如 S3DIS dataset 根本没配 image），所以作者做了一个非常聪明的设计：**让 3D feature 在 pretraining 阶段去"模拟"2D feature**，这样 inference 时模型内部已经隐含了 2D 信息，对外部不依赖。

这个思路跟 2DPASS (https://arxiv.org/abs/2207.06989) 和 OpenScene (https://arxiv.org/abs/2211.15694) 里的 2D-prior-distillation-into-3D 是一脉相承的，但应用在 few-shot segmentation 上是 first attempt。

---

## 3. Setup 设计的精妙：什么是 "cost-free"？

我特别欣赏作者把这个 setup 定义清楚。在 §3.1 里他明确写：

> "for the episode introduced above, we additionally have N class names for S, e.g., 'chair', 'table', 'wall', etc. For the 2D image modality, we have 2D RGB images accompanying 3D point clouds during pretraining, but 2D images are not required during meta-learning and inference."

具体来说 cost-free 体现在：

1. **Textual modality**: 既然 support set 必然要 annotate label，那 label 的 class name 就免费 get
2. **2D modality**: ScanNet 等大型 dataset 本来就 captured RGB image + camera intrinsic，pretraining 完成后，weights 是 **class-agnostic** 的（只做 feature alignment，没有用 semantic label），所以可以 transfer 到 S3DIS 这种没有 image 的 dataset

第二点很关键。作者在 Appendix B 里写了：pretraining 的 loss 是 cosine similarity，让 3D point feature 跟它投影到的 2D pixel feature 对齐，**完全不用 semantic label**。所以这个 pretraining 是 **label-free 的 representation alignment**，weights 学到的是"如何让 3D feature 模拟 VLM 的 2D feature 表达"，本身是 transferable 的。

这点在 Table 1 (S3DIS) 也得到验证 —— MM-FSS 在没有 image 的 S3DIS 上同样涨点明显，因为 backbone 和 IF head 是从 ScanNet pretrain 来的，直接 transfer 过去用。

---

## 4. 架构详解：双 head 设计背后的深层动机

整体架构见图 2 (paper Figure 2)。我把数据流梳理一下：

```
Input point cloud X_s/q
        |
   Shared Backbone Φ (Stratified Transformer 第1-2 block)
        |
   F_s/q ∈ R^{N × D}  (general features at 1/4 resolution)
        |
   ┌────────┴────────┐
   │                 │
IF Head H_IF      UF Head H_UF
   │                 │
F^i ∈ R^{N×D_t}   F^u ∈ R^{N×D}    (at 1/16 resolution, 然后插值回 1/4)
   │                 │
   └────────┬────────┘
            |
       Correlation Generation
   (F^i 用 cosine sim 跟 prototype, F^u 用 cosine sim 跟 prototype)
            |
       C^i, C^u
            |
       MCF (Multimodal Correlation Fusion)
            |
       C_0
            |
       MSF (K blocks, 加 text guidance G_q + 动态权重 W_q)
            |
       C_K
            |
       Decoder (KPConv + MLP)
            |
       P_q
            |
       TACC (test-time only)
            |
       P̂_q = γ·G_q + P_q
```

### 4.1 为什么要 two heads？这是这篇 paper 最 subtle 的设计之一

直觉上你可能会问：既然 IF head 学到的是 aligned with VLM 的好 feature，为什么不直接用 IF head 出来 fused correlation？还要 UF head 干什么？

作者的 motivation（在 Appendix B "Training Strategy" 里写得清楚）：

> "Simultaneously training both heads might complicate and destabilize the optimization process due to significant heterogeneity across different modalities and distinct supervision objectives."

也就是说 IF head要在 VLM embedding space 里 align（高维语义空间，LSeg 是 D_t=512, OpenSeg 是 D_t=768），这种 cross-modal alignment 的训练目标跟 episodic few-shot 的训练目标 heterogeneity 太大，**如果一起训很容易 destabilize**。所以策略是：

1. **Pretraining 阶段**：只训 backbone + IF head，loss 是 cosine similarity alignment，跟 2D pixel feature 对齐
2. **Meta-learning 阶段**：**冻结 backbone + IF head**，新加 UF head + MCF + MSF + Decoder，用 cross-entropy 训 few-shot segmentation

这样 IF head 的 VLM-aligned 特征被 "frozen 住"作为稳定 anchor，而 UF head 在 meta-learning 时跟下游 task 一起 optimize，承担"task-adaptive"的角色。

这个设计的 intuition 跟 DINOv2 (https://arxiv.org/abs/2304.07193) 和 MAE (https://arxiv.org/abs/2111.06377) 里 "frozen pretrained feature + learnable head" 的 philosophy 是相通的 —— representation 和 task adaptation 解耦。

### 4.2 为什么 UF head 仍然必要？TACC 的伏笔

Ablation Table 3b 显示：单 IF head 是不够的。原因作者在 §3.5 里点明：

> "G_q is derived from the query intermodal features and text embeddings, which are not updated throughout the meta-learning process. Thus, G_q includes much less bias towards the training categories."

也就是说 IF head 是 frozen 的，没参与 meta-learning，所以**它没有 base class bias**（base bias 是 few-shot 模型在 base class 上 fully supervised 训练后产生的偏置，会"误激活"base class，参见 Lang et al. 2022, https://arxiv.org/abs/2103.00020 类似的 bias 问题）。

而 UF head 参与了 meta-learning，所以**有 base class bias，但 task-adaptive**。

所以两路互补：
- IF head (frozen) → 提供无偏但 underfit task 的信号 → 用作 text-aware semantic guidance G_q 和 TACC 的 calibrator
- UF head (trained) → 提供有偏但 task-fit 的信号 → 用作主要 correlation

TACC 公式 $\hat{P}_q = \gamma \cdot G_q + P_q$ (Eq. 9) 本质上是 **把无偏的 IF 信号按需 mix 进有偏的最终预测**。$\gamma$ 由 support set 的 IoU 决定 —— support 表现好就多用 G_q，表现差就少用。这个 self-adaptive 的 calibration 在 §5 我会展开。

---

## 5. 公式逐个拆解

### 5.1 Feature extraction (Eq. 1)

$$
\mathbf{F}_s^i = \mathcal{H}_{IF}(\mathbf{F}_s) \in \mathbb{R}^{N_S \times D_t}, \quad \mathbf{F}_s^u = \mathcal{H}_{UF}(\mathbf{F}_s) \in \mathbb{R}^{N_S \times D}
$$
$$
\mathbf{F}_q^i = \mathcal{H}_{IF}(\mathbf{F}_q) \in \mathbb{R}^{N_Q \times D_t}, \quad \mathbf{F}_q^u = \mathcal{H}_{UF}(\mathbf{F}_q) \in \mathbb{R}^{N_Q \times D}
$$

变量说明：
- 下标 `s/q`: support / query
- 上标 `i/u`: intermodal / unimodal
- $N_S, N_Q$: support 和 query 中的 point 数量（实验里 block 化后最多 20480 点）
- $D$: unimodal feature 通道数 = 192 (Stratified Transformer 第三 stage 输出维度)
- $D_t$: intermodal feature 通道数，与 VLM embedding 对齐（LSeg=512, OpenSeg=768）

注意这里 $D_t \neq D$，所以两个 head 输出 dimension 不同，下面 MCF 不能直接做 element-wise 操作，需要先 project。

### 5.2 Prototype generation (Eq. 2)

$$
\mathbf{P}_{fg}^i, \mathbf{P}_{bg}^i = \mathcal{F}_{proto}(\mathbf{F}_s^i, \mathbf{Y}_s, \mathbf{L}_s), \quad \in \mathbb{R}^{N_P \times D_t}
$$
$$
\mathbf{P}_{fg}^u, \mathbf{P}_{bg}^u = \mathcal{F}_{proto}(\mathbf{F}_s^u, \mathbf{Y}_s, \mathbf{L}_s), \quad \in \mathbb{R}^{N_P \times D}
$$

- 下标 `fg/bg`: foreground / background
- $N_P = 100$: 每类的 prototype 数量（实验里设的，跟 COSeg 一致）
- $\mathbf{Y}_s$: support 的 point-level label
- $\mathbf{L}_s$: support point 的 3D coordinates

$\mathcal{F}_{proto}$ 用的是 farthest point sampling + points-to-samples clustering（沿用 AttMPTI 的设计，https://arxiv.org/abs/2104.12000）—— 这是为了避免 single prototype 在 intra-class variation 大时不稳定。100 个 prototype 提供 multi-modal 表达。

K-shot (k>1) 时每 shot 取 $N_P/k$ 个 prototype 然后拼接，保持总数 100。

### 5.3 Correlation (Eq. 3)

$$
\mathbf{C}^i = \frac{\mathbf{F}_q^i \cdot \mathbf{P}_{proto}^{i^\top}}{\|\mathbf{F}_q^i\| \|\mathbf{P}_{proto}^{i^\top}\|}, \quad \mathbf{C}^u = \frac{\mathbf{F}_q^u \cdot \mathbf{P}_{proto}^{u^\top}}{\|\mathbf{F}_q^u\| \|\mathbf{P}_{proto}^{u^\top}\|}
$$

这里就是标准 cosine similarity。$\mathbf{P}_{proto}^{i} = \mathbf{P}_{fg}^i \oplus \mathbf{P}_{bg}^i$（$\oplus$ 是 concat）。

输出 shape：$\mathbf{C}^i, \mathbf{C}^u \in \mathbb{R}^{N_Q \times (N_C \times N_P)}$

- $N_C = N + 1$: N 个 novel class + 1 个 background
- $N_P = 100$: 每类 prototype 数

所以每个 query point 对每个 prototype 都有一个 cosine sim score。但 cosine sim 的值域是 $[-1, 1]$，paper 里没明确说是否 normalize 到 $[0,1]$，但从下游看应该不影响（因为是 linear layer 后再 fuse）。

### 5.4 MCF: Multimodal Correlation Fusion (Eq. 4)

$$
\mathbf{C}_0 = \mathcal{F}_{lin}(\mathbf{C}^i) + \mathcal{F}_{lin}(\mathbf{C}^u), \quad \mathbf{C}_0 \in \mathbb{R}^{N_Q \times N_C \times D}
$$

关键操作：
- $\mathcal{F}_{lin}$: linear layer 把 $N_P$ 这一维 project 到 $D$ 维（即从 100 个 prototype sim score 转成 D=192 维的 representation）
- 加号: 直接 sum（Ablation Table 6 验证 1:1 vs 0.5:1 性能差不多，说明 fusion 鲁棒）

这步的 intuition：query 跟 support prototype 的"关系"不再是简单的 scalar，而是被 linear 投影到 D 维 correlation embedding 里。两路 correlation 在同一 D 维空间里 element-wise 相加。

为什么这么 fusion 而不是 concat 然后 MLP？我猜是为了 parameter efficiency（concat 会翻倍），并且 element-wise sum 保留了 "intermodal 和 unimodal 在同一维度上互相补强" 的归纳偏置。

### 5.5 MSF: Multimodal Semantic Fusion (Eq. 5-8)

这是这篇 paper 最核心的创新，逐公式看：

**Step 1: Semantic guidance G_q (Eq. 5)**

$$
\mathbf{G}_q = \mathbf{F}_q^i \cdot \mathbf{T}^\top
$$

- $\mathbf{T} \in \mathbb{R}^{N_C \times D_t}$: text embeddings（含 background）由 LSeg text encoder 生成
- $\mathbf{F}_q^i \in \mathbb{R}^{N_Q \times D_t}$: query 的 intermodal feature
- 输出 $\mathbf{G}_q \in \mathbb{R}^{N_Q \times N_C}$: 每个 query point 对每个 class 的"text-image alignment score"

这里能这么做是因为 IF head 在 pretraining 阶段被训练去 align LSeg 的 2D visual feature，而 LSeg 的 2D 和 text embedding 本身就是 aligned 的（LSeg 是 image+text 的 VLM）。所以 IF head 出来的 3D feature 跟 text embedding 在同一 space，可以直接 dot product 算 affinity。**这个"对齐链"是整套设计的基石**：3D IF → 2D VLM visual → text。**没有 pretraining 阶段的 2D 蒸馏，整条链就断了**。

**Step 2: Dynamic weights W_q (Eq. 6)**

$$
\mathbf{W}_q = \mathcal{F}_{mlp}(\mathcal{F}_{expand}(\mathbf{G}_q) \oplus \mathbf{C}_k), \quad \mathbf{W}_q \in \mathbb{R}^{N_Q \times N_C \times 1}
$$

- $\mathcal{F}_{expand}$: 把 $\mathbf{G}_q$ 从 $\mathbb{R}^{N_Q \times N_C}$ 扩展到 $\mathbb{R}^{N_Q \times N_C \times D}$（last dim repeat D 次）
- $\oplus$: channel-wise concat
- $\mathcal{F}_{mlp}$: MLP，输出 per-point per-class 的 scalar weight

为什么 expand 之后再 concat？因为 $\mathbf{G}_q$ 是个 $N_Q \times N_C$ 的 2D tensor，要跟 $\mathbf{C}_k \in \mathbb{R}^{N_Q \times N_C \times D}$ 的 3D tensor concat，需要先 broadcast 到同 shape。

直觉理解：**对每个 (query point, class) pair，模型动态决定该用多少 text guidance**。比如某个 query point 在 IF feature 上跟 text "chair" 高度 align 但 visual correlation 弱（support chair 长得不像 query chair），那么 $W_q$ 应该更大，让 text guidance 起更大作用。Fig 5 (paper Appendix Figure 5) 的可视化验证了这点：bookcase 行（support 和 query 差异大）$W_q$ 高，table 行（差异小）$W_q$ 均匀。

**Step 3: Weighted fusion (Eq. 7-8)**

$$
\mathbf{C}_k' = \mathbf{G}_q \odot \mathbf{W}_q + \mathbf{C}_k
$$

$$
\mathbf{C}_{k+1} = \mathcal{F}_{mlp}(\mathcal{F}_{attention}(\mathbf{C}_k'))
$$

- $\odot$: Hadamard product (element-wise multiply)，$\mathbf{W}_q$ 在 channel 维上 broadcast
- $\mathcal{F}_{attention}$: linear attention (Katharopoulos et al. 2020, https://arxiv.org/abs/2006.16236) —— 用 linear 而不是 softmax attention 应该是为了计算效率，因为 $N_Q$ 可以很大
- $\mathcal{F}_{mlp}$: MLP refine

K 个 block 级联。S3DIS 用 K=2，ScanNet 用 K=4（ScanNet 场景更复杂需要更深 refinement）。

### 5.6 TACC: Test-time Adaptive Cross-modal Calibration (Eq. 9-10)

**Eq. 9:**

$$
\hat{\mathbf{P}}_q = \gamma \mathbf{G}_q + \mathbf{P}_q
$$

- $\mathbf{P}_q$: decoder 出来的 prediction logit
- $\gamma$: adaptive indicator
- $\mathbf{G}_q$: 上面计算的 semantic guidance

注意这里是 **logit-level 的加性 calibration**，不是 softmax 后的 probability mixture。直觉上，$\gamma \mathbf{G}_q$ 是给 novel class 的 logit 加一个 boost，对抗 base class bias 导致的 false activation。

**Eq. 10: 计算 γ**

$$
\gamma = \frac{\sum_i \mathbf{1}_{\{\mathbf{P}_s(i)=1 \wedge \mathbf{Y}_s(i)=1\}}}{\sum_i \mathbf{1}_{\{\mathbf{P}_s(i)=1 \vee \mathbf{Y}_s(i)=1\}}}
$$

- $\mathbf{P}_s[i] = \arg\max(\mathbf{G}_s[i, :])$: 用 G_s 直接预测的 support label
- $\mathbf{G}_s = \mathbf{F}_s^i \cdot \mathbf{T}^\top$: support 的 intermodal feature 跟 text embedding 的 affinity
- 分子: 正确预测为 foreground 的点数（TP）
- 分母: 预测为 foreground 或真实为 foreground 的点数（TP + FP + FN）

这就是标准 IoU 公式！γ = IoU(G_s 预测, Y_s 真值)。

**为什么用 support 的 IoU 作为 query 的 γ？** 这是这套设计最精彩的地方：

- G_q 和 G_s 都是 frozen IF head + text embedding 计算的，**它们的质量是同质的**。如果 G_s 在 support 上 IoU 高，说明 IF-text affinity 这条通路对当前 episode 的类是 work 的；如果 G_s 在 support 上 IoU 低，说明通路不 work，那 G_q 也不能信
- 所以 γ 是一个 **per-episode 的"信号质量自评"** —— 完全 parameter-free，不需要学任何参数，是 closed-form 的

5-shot 时有 5 个 γ 值，作者比较了 mean / max / min 三种聚合（Appendix Table 5），mean 和 max 都不错，min 显著差（-1.3%）。默认用 max。直觉：max 偏向"至少有一个 shot 信号好"，更乐观。

Ablation Table 3e 显示固定系数 (1:1, 1:0.5) 都不如 adaptive γ —— 说明**自适应**本身比具体的 ratio 更重要。

---

## 6. 实验结果的技术解读

### 6.1 Main results (Table 1, 2)

S3DIS（无 2D image 的 dataset）:

| Setting | COSeg (前SOTA) | MM-FSS | Δ |
|---------|---------------|--------|---|
| 1-way 1-shot | 47.77 | 52.09 | +4.32 |
| 1-way 5-shot | 50.41 | 54.21 | +3.80 |
| 2-way 1-shot | 38.07 | 44.30 | +6.23 |
| 2-way 5-shot | 41.49 | 50.16 | +8.67 |

ScanNet（有 2D image 的 dataset，但 inference 也不用 image）:

| Setting | COSeg | MM-FSS | Δ |
|---------|-------|--------|---|
| 1-way 1-shot | 42.01 | 44.73 | +2.72 |
| 1-way 5-shot | 46.61 | 50.07 | +3.46 |
| 2-way 1-shot | 29.03 | 39.21 | +10.18 |
| 2-way 5-shot | 35.51 | 44.09 | +8.58 |

**两个关键观察**:

1. **N-way 越大，收益越大** (1-way +3~4%, 2-way +6~10%): 这强烈暗示 multimodal 知识在需要 disambiguate 多个 novel class 时更能发挥作用。单 class 时 visual cue 还勉强够；多 class 时 text semantic prior 提供的"语义锚点"对 disambiguation 至关重要。
2. **S3DIS 也涨**（虽然 S3DIS 没有 image）: 验证了 cost-free setup 的核心 claim —— 用 ScanNet pretrain 的 IF head 直接 transfer 到 S3DIS，依然受益。这说明 IF head 学到的是 **scene-general 的 3D-to-VLM alignment**，不是 dataset-specific 的。

### 6.2 COSeg† ablation 的关键性

COSeg† 是作者给 COSeg 套上同样的 2D-aligned pretrained backbone，但 **COSeg† 相比 COSeg 几乎没涨**（Table 1: 47.21 → 47.77）。这是非常关键的 ablation，它说明：**单纯有 2D-aligned 的 backbone 不够，必须要有专门的 fusion modules (MCF, MSF, TACC) 才能把 multimodal 信息真正用起来**。这印证了 §C 里作者的话："how to effectively leverage multimodal features to establish informative correlations ... poses unique challenges"。

### 6.3 Ablation 解读

**Table 3a - Fusion modules**: MCF 单独用涨 ~2%，MSF 单独用涨 ~4%，合在一起涨 ~5%。互补性得到验证。

**Table 3c - MSF blocks 数量**: K 从 1 增到 4 在 ScanNet 上持续涨。但作者没测更大的 K，估计是 marginal return 递减且 FLOPs 考虑。

**Table 3d - 各模态贡献**: 3D only → +image → +text 逐步涨。证实了 image 和 text 各自独立的 contribution。注意 image 这一项 +0.76% (1-shot) / +0.87% (5-shot) 看起来小，但这是在 baseline 已经用了 image-aligned backbone 的基础上 —— 实际上 image 贡献已经 implicit 在 backbone 里了，这里只是说显式有 IF head 走 intermodal correlation 的额外收益。

**Table 3e - TACC 系数**: 0:1 (baseline) → 1:0 (只用 G_q, 性能最差！) → 1:1 (固定) → 1:0.5 (固定) → adaptive γ (最好)。1:0 最差说明 G_q 单独不够，必须跟 P_q 互补。adaptive γ 比所有固定都好，验证了"per-episode 自适应"的必要性。

**Table 3f - MSF weighting**: Default (动态 W_q) > MSF-linear（简单线性融合）。验证了 per-point per-class 的动态加权是必要的，而不是 over-engineering。

**Table 3g - Complexity**: MM-FSS vs COSeg，FLOPs 几乎不变，参数略增（多了 MSF blocks 和 text encoder inference，但 text encoder 只 forward 一次每个 episode）。

### 6.4 VLM 替换的鲁棒性 (Appendix Table 4)

作者用 LSeg (D_t=512) 和 OpenSeg (D_t=768) 两个 VLM 都试了，性能差不多。这暗示这套方法不依赖于特定 VLM 的 magic，是 VLM-agnostic 的 design。这点很重要，因为 VLM 这一两年迭代很快 (https://arxiv.org/abs/2304.01152, https://arxiv.org/abs/2401.04077)，方法的可持续性依赖于对底层 VLM 的可替换性。

---

## 7. 我的 critical thoughts 和 open questions

读到这里，我自己有几个疑问和扩展思考：

### 7.1 Text encoder 的开放性

paper 用 LSeg 的 text encoder。但 LSeg 的 text encoder 是 CLIP ViT-B/32 量级的。如果换成更强的 text encoder 比如 E5-Large (https://arxiv.org/abs/2212.03533) 或者 instruction-tuned 的 LLM encoder（比如 Eurus-7B embedding, https://arxiv.org/abs/2401.05358），会不会有进一步涨幅？paper 没做这个 ablation。

### 7.2 2D modality 的"模拟"程度

作者在 §3.2 强调 IF head 的 feature 是 "intermodal" (aware of 2D + 3D)。但 Appendix B 的 pretraining loss 只用 cosine similarity alignment，没有 contrastive 之类的更复杂 loss。这种 alignment 学到的 3D feature 能 capture 2D image 的多少信息？是否在高 texture-dependent class（如 painting, window）上效果更好，在 geometry-dependent class（如 floor, wall）上效果一般？这个 per-class breakdown 没有 paper 也没做。我看 Fig 6 的可视化里 chair 和 door 涨得明显，可能确实有这个 pattern。

### 7.3 TACC 的理论性质

TACC 的 $\gamma$ 是 IoU(G_s, Y_s)，这是个 **plugin-in estimator**，没有理论保证。一个潜在 risk：当 support 的 K-shot 中某个 shot 是 outlier（比如视角很怪），G_s 的 IoU 可能 misleading。作者用 max aggregation 缓解，但更好的做法可能是用 BOP (Bayes Optimal Predictor) 风格的 confidence estimation。

### 7.4 跟 LLM 时代的契合

paper 用的是 LSeg 这种小 VLM。当下 trend 是 LLM-augmented perception，比如 PointLLM (https://arxiv.org/abs/2305.04804)、LLaVA-3D 类工作。如果让一个 LLM 在 inference 时直接读取 query point cloud 的 visual feature + class name prompt，让 LLM 输出每个 point 的归属，会不会更 elegant？不过这是另一个故事了。

### 7.5 跟 SAM-3D 的潜在结合

SAM (https://arxiv.org/abs/2304.02643) 已经有 3D 扩展工作。如果用 SAM-3D 提供 promptable mask，结合 MM-FSS 的 text guidance，可能 open-set few-shot segmentation 就有新玩法。这是这篇 paper 的一个潜在 downstream direction。

### 7.6 Episodic vs Transductive

AttMPTI 是 transductive（用 query 的全体统计），MM-FSS 是 inductive（每个 query point 独立预测）。在 2-way 设置下 MM-FSS 显著领先，但作者没在 paper 里 explicit 跟 transductive 版本对比 —— 也许 COSeg 也已经是 inductive 的，但 transductive MM-FSS 的潜力还没探索。

---

## 8. 一图总结我的理解

```
                              ┌────────────────────────────┐
                              │   Cost-Free Multimodal     │
                              │   FS-PCS Setup             │
                              │                            │
                              │  • text: class name        │
                              │    (free at annotation)    │
                              │  • 2D image: pretrain only │
                              │    (implicit distillation) │
                              └─────────────┬──────────────┘
                                            │
                       ┌────────────────────┴────────────────────┐
                       │              MM-FSS Model                │
                       │                                          │
            ┌──────────┴──────────┐                ┌──────────────┴──────────┐
            │  Frozen Branch       │                │  Learnable Branch       │
            │  (Pretrain only)     │                │  (Meta-learned)          │
            │                      │                │                          │
            │  Backbone → IF head  │                │  Backbone → UF head      │
            │  → F^i ∈ R^{N×D_t}   │                │  → F^u ∈ R^{N×D}         │
            │  aligned with        │                │  task-adaptive           │
            │  LSeg 2D+text        │                │  but base-biased         │
            │                      │                │                          │
            │  bias-free, task-    │                │                          │
            │  agnostic             │                │                          │
            └──────────┬───────────┘                └──────────────┬───────────┘
                       │                                           │
                       │ ┌─────────────────────────────────────────┘
                       │ │
                       │ │      ┌──── MCF: C^i + C^u → C_0 ────┐
                       │ │ ────►│                                │
                       │ │      │      (Eq. 4)                   │
                       │ │      └──────────────┬─────────────────┘
                       │ │                     │
                       │ │                     ▼
                       │ │      ┌──── MSF: G_q ⊙ W_q + C_k ─────┐
                       │ ├─────►│  (K blocks, linear attention)  │
                       │ │      │  (Eq. 5-8)                     │
                       │ │      └──────────────┬─────────────────┘
                       │ │                     ▼
                       │ │             P_q (decoder out)
                       │ │                     │
                       │ └──────────────────── │
                       │                      ▼
                       │      ┌─── TACC: γ·G_q + P_q ──┐
                       └─────►│  γ = IoU(G_s, Y_s)      │
                              │  (Eq. 9-10)             │
                              └───────────┬─────────────┘
                                          ▼
                                       P̂_q (final)
```

---

## 9. 总结：这篇 paper 给我的三条 takeaway

1. **"Free modality" 是一个值得 systematize 的概念**: 很多 dataset 在采集时已经免费 got 多模态（RGB+Depth+Textual class name+IMU+...），但在 algorithm 设计时常被 ignore。这篇 paper 把 textual + 2D 都拿出来用，是一个很务实的工程化思考。**未来在 4D 视频点云、autonomous driving perception 里这个思路潜力巨大**（比如 Waymo 数据有 RGB + LiDAR + HD map 三模态，做 few-shot novel class segmentation 时完全可以套用）。

2. **Frozen + Learnable 双 branch 设计的普适性**: IF head (frozen, bias-free) + UF head (learnable, biased) 的二分法本质上是一种 **regularization via architecture**。这个 pattern 在 LLM era 也很常见 —— frozen LLM + learnable adapter（https://arxiv.org/abs/2106.09685）是同构思路。作者的 TACC 用 frozen branch 校准 learnable branch，思路类似 LoRA 的 orthogonal regularization。

3. **Self-adaptive indicator without learning**: γ = IoU(G_s, Y_s) 是一个 zero-parameter, closed-form 的"信号质量自评"。这种 parameter-free 设计在 model 越来越大的当下值得提倡。类似的思路在 DDIM sampling step selection (https://arxiv.org/abs/2204.01827) 和 dynamic thresholding (https://arxiv.org/abs/2205.11487) 里也有。**few-shot 本质是数据稀缺，少学点参数、多用点 closed-form heuristic 反而更 generalizable**。

---

## 10. 相关参考文献

- 原 paper: https://arxiv.org/abs/2502.18837
- LSeg: https://arxiv.org/abs/2011.12766
- OpenSeg: https://arxiv.org/abs/2112.12143
- CLIP: https://arxiv.org/abs/2103.00020
- COSeg (前 SOTA): https://arxiv.org/abs/2403.09431
- AttMPTI (FS-PCS 开山): https://arxiv.org/abs/2104.12000 (Zhao et al. 2021)
- Stratified Transformer: https://arxiv.org/abs/2202.10339
- KPConv: https://arxiv.org/abs/1904.08868
- Linear attention: https://arxiv.org/abs/2006.16236
- 2DPASS (类似 2D-prior-distillation 思路): https://arxiv.org/abs/2207.06989
- OpenScene (3D VLM): https://arxiv.org/abs/2211.15694
- CLIP2Scene: https://arxiv.org/abs/2304.00312
- SAM: https://arxiv.org/abs/2304.02643
- PointLLM: https://arxiv.org/abs/2305.04804
- DINOv2 (frozen representation philosophy): https://arxiv.org/abs/2304.07193
- S3DIS dataset: http://buildingparser.stanford.edu/
- ScanNet dataset: http://www.scan-net.org/
- LoRA: https://arxiv.org/abs/2106.09685

---

如果让我一句话总结这篇 paper 给我的最大启发：**在数据稀缺的 few-shot 场景下，与其继续在 unimodal 里 squeeze performance，不如回过头看看 dataset 里已经免费 got 哪些 modality 没用上**。这个 framing 比方法本身更重要，因为它是一个**研究范式 shift 的暗示** —— few-shot learning 应该从 "how to learn from few visual examples" 拓展为 "how to learn from few examples + abundant free cross-modal priors"。这一点跟 CLIP 当年的贡献是同质的：从"用海量 labeled image 学 vision" 拓展为"用海量 image-text pair 学 vision-language joint embedding"。

希望这个解读对你 build intuition 有帮助！如果你想深挖某一块（比如 §5.5 MSF 的 dynamic weighting 跟 cross-attention 的关系，或者 §7 我提的几个 open questions 里挑一个深入聊），我们可以继续展开。
