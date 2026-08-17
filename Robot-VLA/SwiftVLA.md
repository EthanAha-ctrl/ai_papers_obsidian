---
source_pdf: SwiftVLA.pdf
paper_sha256: 023555fee45de5494cf65f02410fbafea3ac66e01665b491d844ab05a90d79b3
processed_at: '2026-08-12T11:39:08-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SwiftVLA

## 先说这 paper 在干嘛

想象你训练一个 robot arm 干活，给它看图、给它听指令，让它输出 action。这件事现在大家都在做，叫 VLA (Vision-Language-Action) model。

问题在哪？要干得好，model 得"看得懂"3D 空间。比如桌上三个碗，指令说"拿最左边那个"，model 得知道哪个是"最左"、有多远、怎么抓。这个能力叫 spatial reasoning。

现在的尴尬局面是这样的：

**大 model 路线**（π0 用 PaliGemma-3B）：3B 参数，spatial reasoning 强，能正确回答"最左边的碗什么颜色"。但跑在 NVIDIA Jetson Orin 这种 edge device 上要 2.97 秒一次 inference，占 16GB memory。Robot 实时控制需要至少 10Hz，这根本没法用。

**小 model 路线**（SmolVLA 用 SmolVLM-0.5B）：0.45B 参数，Jetson 上 0.17 秒，飞快。但 Figure 1 那张图说得很直白——问 SmolVLM "最左边碗什么颜色"，它答错了。它把空间位置搞混了。直接后果就是 Figure 5 里那个尴尬画面：robot arm 没抓准，直接把物体撞飞了。

所以 trade-off 很清楚：要 spatial reasoning 就得大 model，要部署就得小 model，鱼和熊掌。

**3D 增强路线**：大家想，那给小 model 喂点 3D 信息（depth map、point cloud）不就行了？确实有效，但带来新问题：

- 直接把 3D features 塞进 VLM，small VLM 搞不定 cross-modal fusion（小 model 没那个能力整合多 modality）
- 额外加个 3D branch，参数又膨胀了（PointVLA、GeoVLA 路线），违背 lightweight 初衷
- 而且都只 3D，不 4D，没有 temporal 信息

SwiftVLA 的核心 idea 一句话：**训练时用 4D features 当"老师"监督小 model 学几何，inference 时把 4D 老师开掉，只留学到几何知识的小 model 跑**。类似于你请个家教辅导你学几何，考试时家教不在旁边你也会做题。

这个思路我个人觉得挺 elegant 的，它把 "modality distillation via masking" 这个 idea 用得很到位。下面拆开讲。

---

## 拆解三个核心 trick

### Trick 1: 4D Feature Extraction（"家教"是谁）

4D features 从哪来？不能装 depth camera 或 LiDAR，硬件成本太高。SwiftVLA 用了一个 pretrained 的 Streaming 4D Visual Geometry Transformer（arXiv:2507.11539），从普通 RGB 图像推 4D features。

工作流程 Figure 4 画得很清楚。每个 time step $t$，三个 view 的图 $o_t^v$（left / right / front）分别经过 encoder 得到 $F_e^{t,v}$。然后 decoder 用一个 temporal cache $C$ 把历史信息注入：

$$\bigl(F_{4D}^{t,v}, C^{t,k}\bigr) = \text{Decoder}\bigl(\text{CrossAttn}(F_e^{t,v}, C^{t,k-1})\bigr)$$

变量讲清楚：
- $F_e^{t,v}$：time $t$、view $v$ 的 encoder 输出
- $C^{t,k}$：处理第 $k$ 个 view 时 cache 的状态（$k=1,2,3$ 对应 left, right, front）
- $C^{t,0} = C^{t-1}$：cache 从上一帧继承
- CrossAttn 里 query 是 $F_e^{t,v}$，key/value 是 cache

cache 用 FIFO 策略保留最近 $K$ 个 representations。Table 7 显示 random $K \in \{3,4,5,6\}$ 最好（SR=0.53），固定 $K$ 都略差。这个发现挺有意思——variable temporal horizon 训练让 model 学会 adapt 不同时间尺度，类似 random crop size 让 CNN 学 scale invariance。

关键细节：4D extractor 权重 **frozen**，不参与 training。这是个 protection——不让 4D branch 和主 VLM 协同 overfit。它纯粹当 "4D feature oracle" 用。

另一个细节：只有 front view 的 4D features 送进 VLM（Eq. 5），left/right 的 4D features 只用来 update cache。这是 efficiency 决定——三个 view 的 4D tokens 太多，cache 已经 summarize 了多 view 几何，front view 输出相当于 cache 的 summary。

### Trick 2: Fusion Tokens（"怎么让小 model 学会融合"）

这个 trick 解决的是 small VLM 不擅长 multimodal fusion 的问题。

直接把 4D features 喂进 small VLM，效果有限——Table 5 第二行只比第一行（2D only）高 0.04（0.40 vs 0.36）。小 model 注意力分散，4D features 容易被淹没。

SwiftVLA 引入一组 learnable tokens $Q_f$，叫 Fusion Tokens。它们像一组"探针"，通过 cross-attention 主动从 2D / 4D / language / state 池里"打捞"信息：

$$Z_f^t = \mathcal{V}(Q_f, E_s^t, E_l^t, F_{4D}^t, F_{2D}^t)$$

变量：
- $Z_f^t$：VLM $\mathcal{V}$ 的 output sequence
- $Q_f$：Fusion Tokens
- $E_s^t$：proprioceptive state embedding（关节位置等）
- $E_l^t$：language instruction embedding

光有 Fusion Tokens 还不够，需要 supervision 告诉它们"学什么"。SwiftVLA 用 **end-effector future trajectory** 直接 supervise Fusion Tokens 的输出：

$$\hat{\tau}_t = h_{\text{traj}}(Z_f^t), \quad \mathcal{L}_{\text{traj}} = \|\hat{\tau}_t - \tau_t\|_2^2$$

- $\hat{\tau}_t$：predicted 未来轨迹
- $\tau_t$：ground-truth 未来轨迹
- $h_{\text{traj}}$：trajectory decoder

这个设计直觉上很对：Fusion Tokens 必须编码"机器人未来要走的 path"。trajectory 和 action chunk 高度相关，所以 trajectory supervision 和 action supervision 互补——前者 explicit 直接学，后者通过 diffusion 间接学。Fusion Tokens 充当 "action planning bottleneck"。

这让我联想到 BLIP-2 的 Q-Former（https://arxiv.org/abs/2302.14082），但区别是：Q-Former 桥接 frozen image encoder + LLM，Fusion Tokens 桥接 multimodal input → action output；Q-Former 用 image-text contrastive supervision，Fusion Tokens 用 trajectory regression supervision。本质都是 "learnable query tokens 做 cross-modal bottleneck"。

Table 5 ablation 干脆利落：
- 2D only: 0.36
- 2D + 4D（no Fusion Tokens）: 0.40
- 2D + 4D + Fusion Tokens: **0.50**

第三行比第二行高 0.10，说明 small VLM 真的需要 explicit fusion guidance。

### Trick 3: Mask-and-Reconstruct（"怎么把 4D 知识 distill 进 2D path"）

这是最精妙的部分。

**Training 时**：以一定 probability 随机 mask 2D 或 4D features。当 4D 被 mask，VLM attention 排除 4D tokens，但 action expert 仍要 reconstruct 4D features：

$$\mathcal{L}_{4D} = \|h_{4D}(Z_\mathcal{A}^t) - F_{4D}^t\|_2^2$$

- $h_{4D}$：4D reconstruction head
- $Z_\mathcal{A}^t$：action expert 输出的 latent
- $F_{4D}^t$：原本的 4D features（target）

**Inference 时**：完全 drop 4D extractor、reconstruction heads、trajectory head。只留 VLM + action expert + Fusion Tokens，输入仅 2D。

**为什么这能 work？** 三个机制叠加：

**机制 1：Cross-modal distillation**。当 4D 被 mask 但要 reconstruct 4D，model 被迫从 2D features 中"反推"出 4D 信息。这相当于在 VLM 内部隐式学了一个 single-image depth estimation + temporal modeling 网络。

**机制 2：Dropout-style regularization**。随机 mask 防 overfitting on 4D modality。Multimodal learning 有个著名问题叫 "modality laziness"——强势 modality 抑制弱势 modality 学习（参考 https://arxiv.org/abs/2005.03610）。Masking 强制 model 不能完全依赖 4D。

**机制 3：Information bottleneck**。action latent $Z_\mathcal{A}^t$ 必须同时 support action generation 和 reconstruction，所以它必须是个 rich representation，编码所有有用信息。

Table 6 的 ablation 把这三个机制验证得清清楚楚：

| 4D Mask | 2D Mask | Reconstruction | No 4D infer | With 4D infer |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 0.02 | 0.50 |
| ✓ | ✗ | ✗ | 0.40 | 0.48 |
| ✓ | ✗ | ✓ | 0.50 | 0.52 |
| ✓ | ✓ | ✓ | **0.53** | **0.55** |

第一行最震撼：训练时一直给 4D，inference 突然不给，SR 从 0.50 崩到 0.02。这就是 modality dependency collapse——model 把赌注全押在 4D 上，没了 4D 直接瘫痪。

第二行：加 4D masking，partial 恢复到 0.40。

第三行：加 reconstruction，完全恢复到 0.50，**和有 4D inference 持平**。这就是 distillation 的力量。

第四行：再加 2D masking，0.53，**甚至超过原 4D inference**。这个发现挺 interesting——2D masking 强制 model 更主动利用 4D 几何 cues 来 reconstruct 2D，反过来增强 4D representation 质量。互相 distill，互相强化。

这种 "mask + reconstruct" 的精神本质上是 MAE (https://arxiv.org/abs/2111.06377) 在 multimodal setting 上的延伸。

---

## 整体 loss 长什么样

Eq. 9 把所有 loss 加起来：

$$\mathcal{L}_{\text{total}} = \lambda_{2D}\mathcal{L}_{2D} + \lambda_{4D}\mathcal{L}_{4D} + \lambda_{\text{action}}\mathcal{L}_{\text{action}} + \lambda_{\text{traj}}\mathcal{L}_{\text{traj}}$$

- $\lambda$ 都是 balancing coefficient，论文没给具体值
- $\mathcal{L}_{\text{action}}$ 是 diffusion noise prediction loss（标准 DDPM / Flow Matching）

action expert 是 conditional diffusion model：

$$Z_\mathcal{A}^t = \mathcal{A}(\epsilon \mid \{h_\mathcal{V}^{(i)}\})$$

- $\epsilon \sim \mathcal{N}(0, I)$：高斯 noise
- $\{h_\mathcal{V}^{(i)}\}$：VLM 多层 intermediate hidden states，作为 hierarchical condition
- $\mathcal{A}$：action expert 网络

这个设计借鉴 π0——VLM 的多个 transformer block 输出都喂给 action expert，让 condition 信号更丰富。

---

## Training 流程（两阶段）

Appendix B 给了细节：

**Stage 1 (100k steps)**：纯 action supervision，不启用 4D / Fusion Tokens / mask。lr 从 1e-4 cosine decay 到 2.5e-6，200-step warmup。batch size 256。

**Stage 2 (50k steps)**：从 Stage 1 checkpoint 初始化，启用所有 components。lr 5e-5，cosine decay。

Fine-tuning 也分两阶段：Stage 1 (10k steps) 只 action supervision，Stage 2 启用 4D + Fusion Tokens + mask。

Optimizer 用 AdamW，但 β1=0.85, β2=0.9——这 unusual，默认是 0.9 和 0.999。β2 显著低说明对梯度噪声更敏感响应，可能是 multi-task loss 梯度方差大需要适应。这个细节值得注意，参考 https://arxiv.org/abs/1711.05101。

---

## 实验结果讲故事

### Simulation (RoboTwin 2.0)

Table 1，按 Short / Medium / Long horizon 分三类：

- π0 (3B): Avg SR = 0.47
- TinyVLA (1B): 0.07 ← collapse
- SmolVLA (0.45B): 0.29
- SmolVLA†（公平 pretrain）: 0.36
- **SwiftVLA (0.45B, no 4D infer): 0.53**
- SwiftVLA w/ 4D: 0.55

SwiftVLA 用 ~15% π0 参数，SR 反而高 6 个点。

Long-Horizon 上差距最明显：
- SmolVLA: 0.28
- SwiftVLA: 0.56（翻倍）
- π0: 0.52

Long-horizon 需要强 temporal reasoning，SwiftVLA 反超 π0 说明 4D temporal cache 确实 capture 了 long-horizon dependency。

### Real-World

Table 2，三个 task：

- π0: 0.61
- SmolVLA: 0.34
- **SwiftVLA: 0.80**
- SwiftVLA w/ 4D: 0.82

Real-world 提升（0.80 vs 0.61）比 simulation（0.53 vs 0.47）更显著。我猜原因：real-world 几何更复杂（lighting、occlusion），4D awareness 价值更高；另外 simulation 里 SmolVLA† 公平 pretrain 后已经缩小了差距。

### LIBERO

Table 3，SwiftVLA Avg = 94.7，仅次于 OpenVLA-OFT (97.1)、DD-VLA (96.3)、UniVLA (95.4)——这些都是 7B-9B 大 model。SwiftVLA 0.45B 挤进 top 5 很 impressive。

LIBERO-Long 上：
- SwiftVLA: 88.4
- π0: 85.2
- 4D-VLA: 79.1

再次验证 temporal 优势。但 OpenVLA-OFT 在 Long 上 94.5 远超 SwiftVLA——OFT (https://arxiv.org/abs/2502.19645) 用了 specific fine-tuning tricks，这块对比可能不完全公平。

### Edge Deployment (Jetson Orin)

Table 4 是最 practical 的：

| Model | Time | Memory | SR |
|---|---|---|---|
| π0 | 2.966s | 16236 MB | 0.48 |
| SmolVLA | 0.166s | 1397 MB | 0.30 |
| **SwiftVLA** | **0.167s** | **1398 MB** | **0.76** |

SwiftVLA inference time 和 memory 几乎等于 SmolVLA（多 0.001s、0.9MB），但 SR 从 0.30 提升到 0.76——**0.46 的提升完全免费**。这就是 mask-and-reconstruct 训练的 payoff：4D 知识完全 distill 进 2D-only inference path。

vs π0: 18× speedup, 12× memory reduction, SR 反而 +0.28。这个数字组合在 edge robot 部署上很有吸引力。

### Fold the Cloth（deformable object）

Table 9 是 4D 价值的"极限检验"：

- π0: 0.45
- SmolVLA: 0.05 ← collapse
- SwiftVLA: 0.60
- SwiftVLA w/ 4D: 0.65

Cloth folding 需要 fine-grained 4D understanding（cloth 形变随时间演化），SwiftVLA 大幅领先。这 task 是 deformable object manipulation，4D awareness 价值最高。

---

## 我的 intuition 与几点思考

### 这个工作本质上在做什么？

我觉得 SwiftVLA 本质上是在解决 "small model 能不能有大 model 的 geometric reasoning" 这个问题。答案是通过 **structured supervision + distillation via masking** 实现。

这里有个深层 insight：很多 multimodal learning 工作都在想办法把多 modality 都塞进 inference path，SwiftVLA 反其道而行——training 时多 modality supervision，inference 时单 modality。这是 "train heavy, infer light" 哲学，和 DistilBERT、MoE distillation、knowledge distillation 一脉相承。

### Mask-and-reconstruct 和 knowledge distillation 的区别

Standard KD（https://arxiv.org/abs/1503.02531）是 teacher network 输出 logits，student 模仿。SwiftVLA 的 mask-reconstruct 是 teacher (4D extractor) 提供 features，student (VLM + action expert) 在 input 被遮挡时 reconstruct 这些 features。区别在于：
- KD 模仿 output，SwiftVLA reconstruct intermediate representation
- KD 通常需要 teacher 在 inference 时存在（或者预 compute），SwiftVLA 训完就 drop teacher
- KD 是显式 distillation，SwiftVLA 是隐式 distillation via auxiliary objective

### 这个 pattern 能推广吗？

我觉得可以。其他 modality 也能这样 distill：
- **Tactile sensing**：训练时用 tactile sensor 数据监督视觉 representation，inference 时只用视觉。参考 tactile VLA 工作 https://arxiv.org/abs/2410.01379
- **Audio**：训练时 audio + visual fusion，inference 时只 visual
- **Force-torque**：训练时 force feedback 监督，inference 时无 force sensor

原则上，只要 weak modality（inference 时有的）和 strong modality（inference 时无的）之间有可学习的 mapping，都能用 mask-reconstruct distill。

### Small VLM 不能自主 fuse multimodal 的启示

Table 5 第二行（2D + 4D 直接 concat）只比第一行（2D only）高 0.04，这个发现很重要。它说明 small VLM 不是"喂了就懂"，需要 explicit mechanism 引导 fusion。

这背后的原因可能是 small VLM 注意力容量有限，多 modality 容易"互相干扰"或"某 modality 被忽略"。Fusion Tokens + trajectory supervision 提供 explicit fusion guidance，强制 model 学 cross-modal alignment。

这对未来 small multimodal model 设计有普遍意义：**别光想着喂 modality，要设计 mechanism 让 model 主动用 modality**。

### Streaming cache vs replay all frames

SwiftVLA 用 FIFO cache 而不是 replay 所有历史 frames，这设计很 efficient。复炸度从 O(T²) 降到 O(T·K)。而且 random K 训练比固定 K 好，说明 variable temporal horizon 训练提升 model adaptability。

这和 state space model (Mamba, https://arxiv.org/abs/2312.00752)、streaming transformer 思路一致——用 fixed-size memory summarize history。我觉得 VLA 领域这块还有探索空间，比如 learnable cache update、attention-based cache compression 等。

### 几个我没想通的点

**Fusion Tokens 数量没说**。论文没明确 Q_f 多少个。如果 64 个，inference 时 VLM input 多 64 tokens，开销可控；如果 256 个，可能就不"轻"了。

**Masking probability 没给具体值**。Table 6 验证 mask 有效但没说 mask ratio。50%? 30%? 这影响 distillation 效果。

**Stage 2 才 50k steps**。从 Stage 1 切到 Stage 2，模型需要重新调整 representation 适应 4D 信息。50k steps 够不够充分 distill 是个 question。

**Front view only 4D**。Eq. 5 只送 front view 4D features 进 VLM。左右 view 的 4D 信息只在 cache 里间接编码。这是 efficiency trade-off，但 spatial coverage 可能受限。

**4D extractor frozen**。好处是 stability，坏处是 4D features 可能与 robot task domain 不完全 aligned。预训练 4D VGT 是 general scene geometry，不一定 capture task-relevant 几何（如 gripper-object contact 的精细 4D 结构）。

---

## 一句话总结

SwiftVLA 告诉我们：**小 VLA model 通过 mask-and-reconstruct 训练策略，可以在 inference 时完全不用 4D 输入，但仍保有 4D 空间感知能力，从而同时获得大 model 的 spatial reasoning 和小 model 的 inference speed**。

这个 "train heavy, infer light" 模式我觉得会是 lightweight embodied AI 的标准 pattern 之一。如果延伸到 tactile、audio、force 等其他 modality，可能打开一类新的 efficient robot learning 方法。

Project page: https://Swiftvla.github.io  
SmolVLA: https://arxiv.org/abs/2506.01844  
π0: https://arxiv.org/abs/2410.24164  
Streaming 4D VGT: https://arxiv.org/abs/2507.11539  
SmolVLM: https://arxiv.org/abs/2504.05299  
LIBERO: https://arxiv.org/abs/2306.03310  
RoboTwin 2.0: https://arxiv.org/abs/2506.18088  
BLIP-2 Q-Former: https://arxiv.org/abs/2302.14082  
MAE: https://arxiv.org/abs/2111.06377  
Knowledge Distillation: https://arxiv.org/abs/1503.02531  
Modality Laziness: https://arxiv.org/abs/2005.03610  
Mamba: https://arxiv.org/abs/2312.00752  
AdamW: https://arxiv.org/abs/1711.05101  
SigLIP: https://arxiv.org/abs/2303.15343

---

# SwiftVLA 深度讲解

## 一、论文核心 Problem 与 Motivation 的 Intuition

当前 VLA (Vision-Language-Action) 领域存在一个经典的 trade-off trilemma：**parameter efficiency** vs **spatiotemporal reasoning** vs **inference speed**。

让我先 build 一下这个 problem 的 intuition：

**Large VLM 路线 (π0, OpenVLA)**：基于 PaliGemma-3B 这种 3B+ 参数的 VLM，spatial reasoning 能力强（Figure 1 中能正确回答 "What color is the bowl on the far left?"），但 inference latency 在 NVIDIA Jetson Orin 上 ~2.97s，memory ~16GB，对 edge robot 部署 prohibitive。

**Lightweight VLM 路线 (SmolVLA-0.5B, TinyVLA-1B)**：参数小、速度快（Jetson Orin 上 ~0.17s），但 Figure 1 直观展示了 SmolVLM-0.5B 在 spatial reasoning 上的 collapse——它把最左边的 bowl 和其他 bowl 搞混了。这种 spatial reasoning 的 deficit 直接传导到 action quality 上：imprecise localization、collision risk（Figure 5 中 SmolVLA 撞飞物体）。

**3D-augmented 路线 (3D-VLA, SpatialVLA, PointVLA, GeoVLA)**：通过 depth / point cloud / 3D positional encoding 注入几何信息来补强 spatial reasoning，但 Figure 2 画出了三种 suboptimal 的 design：
- (a) 只用 2D → spatial awareness 弱
- (b) 直接在 large VLM 内部 fuse 2D + 3D → 依赖 heavyweight VLM，small VLM 扛不住 cross-modal alignment
- (c) Decoupled design，额外加 spatial branch → 参数膨胀，与 lightweight 目标冲突

而且以上 3D 方法普遍忽略 **temporal dynamics**——只 3D，不 4D。

SwiftVLA 的核心 insight：**能否在 training 时利用 4D spatiotemporal supervision 把几何知识"蒸馏"进 2D representations，而在 inference 时把 4D branch 整个 drop 掉？** 这样既获得了 4D awareness 的好处，又不需要在 deployment 时付出 4D 计算代价。

这个思路让我联想到 knowledge distillation、MAE (Masked Autoencoder)、dropout-as-regularizer 的精神——通过 structured masking 强制 representation 不依赖单一 modality。

Project page: https://Swiftvla.github.io  
arXiv (推断): 基于 references 中的内容，相关工作 π0 在 arXiv:2410.24164，SmolVLA 在 arXiv:2506.01844，Streaming 4D VGT 在 arXiv:2507.11539。

---

## 二、整体 Architecture 解析

Figure 3 给出了 pipeline。SwiftVLA 由四部分组成：

### 1. 2D Feature Extractor
使用 SigLIP encoder（reference [75], arXiv:2303.15343 风格的 sigmoid loss for image-text pretraining）对多视图 $o_t^v$ 提取 2D 特征 $F_{2D}^{t,v}$。多视图集合 $S = [\text{left}, \text{right}, \text{front}]$。

通过 pixel-shuffle 把每帧 token 数压缩到 64，并跳过 SmolVLM 后面部分层，只用前 16 层。这是 SmolVLA 已有的 efficiency trick。

### 2. 4D Feature Extractor（training-only, frozen）
基于 Streaming 4D Visual Geometry Transformer（reference [86], arXiv:2507.11539），由 encoder + decoder + temporal cache 组成，权重冻结。它从普通 2D images 增量提取 4D features，**不需要 depth camera 或 LiDAR**——这是关键，意味着没有额外 sensor 硬件成本。

### 3. Lightweight VLM (SmolVLM-0.5B)
作为主 backbone，输入包含：
- 2D features $F_{2D}^t$
- 4D features $F_{4D}^t$（仅 front view）
- Language embeddings $E_l^t$
- Proprioceptive state embeddings $E_s^t$
- **Fusion Tokens $Q_f$**（learnable tokens，本文创新点之一）

交替使用 self-attention 与 cross-attention layer（follow SmolVLA 设计）。

### 4. Action Expert
Conditional diffusion model（π0 路线），输入是 noise sample $\epsilon$ 与 VLM 中间层 hidden states $\{h_\mathcal{V}^{(i)}\}$ 作为 condition，输出 action latent $Z_\mathcal{A}^t$。两个 head：
- Action prediction head：predict diffusion noise（主任务）
- Reconstruction heads $h_{2D}, h_{4D}$：reconstruct masked features（auxiliary，inference 时丢弃）

总参数 ~450M（VLM ~350M + action expert ~100M），inference 时只保留 VLM + action expert = ~450M。

---

## 三、关键公式逐行讲解

### Eq. (1): Fusion 表示

$$Z_f^t = \mathcal{V}(Q_f, E_s^t, E_l^t, F_{4D}^t, F_{2D}^t)$$

变量解释：
- $Z_f^t$：time step $t$ 时的 fused representation，是 VLM $\mathcal{V}$ 的 output sequence
- $Q_f$：learnable Fusion Tokens（一组可学习 query embeddings，类似 Q-Former 中的 query tokens，参考 BLIP-2）
- $E_s^t$：proprioceptive state（机器人 joint positions / end-effector pose）的 embedding
- $E_l^t$：language instruction 的 embedding
- $F_{4D}^t = F_{4D}^{t,\text{front}}$：仅 front view 的 4D 特征
- $F_{2D}^t = \{F_{2D}^{t,v}\}_{v \in S}$：三个 view 的 2D 特征

**Intuition**：Fusion Tokens 类似一组"探针"，通过 cross-attention 主动从 2D / 4D / language / state 这些 modality 池里"打捞"信息，最后受 trajectory supervision 约束，因此它们的 representation 是 action-aware 的。

### Eq. (2): Action Latent

$$Z_\mathcal{A}^t = \mathcal{A}(\epsilon \mid \{h_\mathcal{V}^{(i)}\})$$

- $\epsilon \sim \mathcal{N}(0, I)$：标准高斯 noise，diffusion forward process 的 noise sample
- $\{h_\mathcal{V}^{(i)}\}$：VLM 多层 intermediate hidden states，作为 hierarchical condition（类似 π0 把 VLM 多个 transformer block 输出都喂给 action expert）
- $\mathcal{A}$：action expert 网络（diffusion denoiser）
- $Z_\mathcal{A}^t$：action latent，会被两个 head 解码

### Eq. (3) + Eq. (4): Incremental 4D Feature Extraction

$$F_e^{t,v} = \text{Encoder}(o_t^v)$$

$$\bigl(F_{4D}^{t,v}, C^{t,k}\bigr) = \text{Decoder}\bigl(\text{CrossAttn}(F_e^{t,v}, C^{t,k-1})\bigr)$$

变量解释：
- $F_e^{t,v}$：view $v$ 在 time $t$ 的 encoder 输出
- $C^{t,k}$：在处理第 $k$ 个 view 时（k=1,2,3 对应 left, right, front）的 temporal cache state
- 初始化 $C^{t,0} = C^{t-1}$：cache 从上一 time step 继承
- $\text{CrossAttn}(F_e^{t,v}, C^{t,k-1})$：当前 view feature 作为 query，cache 作为 key/value，把历史时空信息注入当前
- 最后 $C^t = C^{t,3}$，三个 view 处理完后更新 cache
- FIFO 策略保留最近 $K$ 个 representations

**Intuition**：这个设计避免了把所有历史 frames 都重复 attend 一遍（O(T²) 复杂度），而是用一个固定大小的 cache 来 summarize history，类似 streaming transformer / state space model 思想。$K$ 的 ablation 在 Table 7 显示 Random $K \in \{3,4,5,6\}$ 最好（SR=0.53），说明 variable temporal horizon 训练能提升模型对 time scale 的 adaptability——这是个有趣的发现，让我联想到 data augmentation 中 random crop size 让 CNN 学到 scale invariance。

### Eq. (5): VLM 视觉输入定义

$$F_{2D}^t = \{F_{2D}^{t,v}\}_{v \in S}, \quad F_{4D}^t = F_{4D}^{t,\text{front}}$$

**关键设计**：left / right view 的 4D features 只用来 update cache，**不送进 VLM**。这是为了控制 VLM 的 token budget——4D features 也限制为 64 tokens per modality。这里有个隐含的 efficiency trade-off：full multiview 4D 会三倍 4D token cost，而 cache 已经捕获了 multi-view 几何信息，所以 front view 的 4D features 实际上是 cache 的"summary output"。

### Eq. (6): Trajectory Supervision

$$\hat{\tau}_t = h_{\text{traj}}(Z_f^t), \quad \mathcal{L}_{\text{traj}} = \|\hat{\tau}_t - \tau_t\|_2^2$$

- $\hat{\tau}_t$：predicted future end-effector trajectory
- $\tau_t$：ground-truth 未来 trajectory
- $h_{\text{traj}}$：trajectory decoder head（MSE regression）

**Intuition**：用 future trajectory 直接 supervise Fusion Tokens 的输出，这就强行让 Fusion Tokens 学到的 representation 编码"机器人未来要走的 path"。trajectory 与 action chunk 高度相关，所以 trajectory supervision 与 action supervision 互补——前者 explicit，后者通过 diffusion 间接。这是一种 auxiliary task learning 的设计。

### Eq. (7): Reconstruction Loss

$$\mathcal{L}_{2D} = \|h_{2D}(Z_\mathcal{A}^t) - F_{2D}^t\|_2^2$$
$$\mathcal{L}_{4D} = \|h_{4D}(Z_\mathcal{A}^t) - F_{4D}^t\|_2^2$$

- $h_{2D}, h_{4D}$：feature reconstruction heads（MLP decoders）
- 当 4D 被 mask 时，$h_{4D}$ 必须从 $Z_\mathcal{A}^t$（由剩下的 2D + language + state 推出的 action latent）重建 $F_{4D}^t$——这就强迫 action latent 内部 encode 4D 信息
- $Z_\mathcal{A}^t$ 是 action expert 的输出 latent

**Intuition**：这个 reconstruction objective 实际上是一个 cross-modal distillation 机制。把它想成"autoencoder bottleneck"——当 4D input 被掐掉，但 output 端要求重建 4D，model 被迫把 4D 信息从 2D 信息中"逆向推导"出来。这暗示 2D 与 4D 之间有可学习的 invertible mapping，而这正是 single-image depth estimation 网络做的事——只是这里隐式学到。

### Eq. (8): Action Diffusion Loss

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\left[\|h_{\text{action}}(Z_\mathcal{A}^t) - \epsilon\|_2^2\right]$$

- 标准 DDPM / flow matching noise prediction loss
- $h_{\text{action}}$：noise prediction head

### Eq. (9): Total Loss

$$\mathcal{L}_{\text{total}} = \lambda_{2D}\mathcal{L}_{2D} + \lambda_{4D}\mathcal{L}_{4D} + \lambda_{\text{action}}\mathcal{L}_{\text{action}} + \lambda_{\text{traj}}\mathcal{L}_{\text{traj}}$$

四个 $\lambda$ 是 balancing coefficient。论文没给出具体值，估计在 appendix 或者 code 里。

---

## 四、Mask-and-Reconstruct 策略的深层 Intuition

这是本文最精妙的部分。让我详细剖析：

### Training 时
- 以一定 probability 随机 mask 2D 或 4D features（Figure 3 灰/白/粉块示意）
- 当 4D 被 mask：VLM attention 中排除 4D tokens，但 action expert 仍需 reconstruct 4D features
- 当 2D 被 mask：同理 reconstruct 2D

### Inference 时
- 完全 drop 4D feature extractor、reconstruction heads、trajectory head
- 只保留 VLM + action expert，输入仅 2D

### 为什么这能 work？三个机制叠加

1. **Cross-modal distillation**：mask 4D 时 forced reconstruction 等价于 implicit depth estimation + temporal modeling 学到 2D features 内部
2. **Dropout-style regularization**：随机 mask 防 overfitting on 4D modality，类似 DropPath / Modality Dropout
3. **Information bottleneck**：action latent $Z_\mathcal{A}^t$ 必须同时 support action generation 和 reconstruction，迫使其成为 rich representation

### Table 6 的 ablation 验证了这一点

| 4D Mask | 2D Mask | Reconstruction | SwiftVLA (no 4D infer) | SwiftVLA w/ 4D |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 0.02 | 0.50 |
| ✓ | ✗ | ✗ | 0.40 | 0.48 |
| ✓ | ✗ | ✓ | 0.50 | 0.52 |
| ✓ | ✓ | ✓ | **0.53** | **0.55** |

第一行说明：训练时一直给 4D，inference 突然不给，性能从 0.50 崩到 0.02——典型的 modality dependency collapse（类似 multimodal learning 中的 "modality laziness" 问题，强势 modality 抑制弱势 modality 学习）。

第二行：加入 4D masking，性能 partial 保持 0.40。

第三行：加 reconstruction，性能恢复到 0.50，**完全找回**有 4D inference 时的水平。

第四行：再加 2D masking，0.53，甚至超过原 4D inference！这暗示 2D masking 强制模型更主动地利用 4D 几何 cues 来 reconstruct 2D，反过来增强了 4D representation 的 quality。

**这个 ablation 是本文最有说服力的部分**，清晰展示了"distillation via masking + reconstruction"的工作机制。

参考 modality collapse 问题相关讨论：
- "What Makes Training Multi-Modal Networks Hard?" (CVPR 2020): https://arxiv.org/abs/2005.03610
- MAE original: https://arxiv.org/abs/2111.06377

---

## 五、Incremental 4D Feature Extraction 的 Cache 机制

Figure 4 展示了 Streaming 4D VGT 的工作流程。让我深入讲一下：

```
Time t:
  Encoder: o_t^v → F_e^{t,v}  (per view)
  For k = 1,2,3 (view order: left, right, front):
    CrossAttn(F_e^{t, S_k}, C^{t,k-1})  →  with cache as KV
    Decoder(...) → (F_{4D}^{t,S_k}, C^{t,k})
  C^t = C^{t,3}
  FIFO update: keep recent K
```

Cache 的工作模式类似 streaming attention、retrieval-augmented generation 中的 memory bank。

关键设计要点：
1. **View order 固定** (left → right → front)：让 cache 在 view 序列内有 consistent 累积，类似 autoregressive over views
2. **FIFO K=3..6**：Table 7 显示 random K best，固定 K 中 K=6 略好于 K=3
3. **Frozen weights**：4D extractor 不更新，纯粹作为"4D feature oracle"提供 supervision signal。这避免了 4D branch 与主 VLM 协同 overfitting，也避免训练时 4D branch 被退化。

为什么 frozen？我推测：4D VGT 是 large pretrained model，让它参与 training 容易 dominate 主 VLM 的 learning signal，破坏 lightweight 设计。Frozen 让它纯粹作为 feature provider，主 VLM 学习如何利用而不是改变这些 features。

---

## 六、Fusion Tokens 与 Q-Former 的对比

Fusion Tokens 让我立刻想到 BLIP-2 的 Q-Former（https://arxiv.org/abs/2302.14082）。但有几个关键区别：

| 维度 | BLIP-2 Q-Former | SwiftVLA Fusion Tokens |
|---|---|---|
| 与哪个 backbone 配对 | Frozen heavy image encoder + LLM | Lightweight VLM |
| Modality | Image → text 桥接 | 2D + 4D + language + state → action 桥接 |
| Supervision | Image-text contrastive/matching/generation | End-effector trajectory prediction |
| 数量 | 通常 32-64 queries | 未明确（推测 64，与 4D token 数一致）|
| Inference 是否保留 | 是 | 是（Fusion Tokens 是 VLM 输入的一部分）|

**Intuition**：Fusion Tokens 在这里扮演"action planning bottleneck"的角色——所有 modality 信息必须经过这一组 tokens 的 bottleneck，然后被 trajectory prediction 直接 evaluate。这避免了 lightweight VLM 在 cross-modal fusion 上的不足（small VLM 注意力分散，容易让某个 modality 被忽略）。

Table 5 的 ablation 验证：
- 2D only: 0.36
- 2D + 4D (no fusion tokens): 0.40（提升 0.04，但有限）
- 2D + 4D + Fusion Tokens: 0.50（再提升 0.10）

第二个到第三个的提升（0.10）说明：**单纯把 4D features 喂进 small VLM 收益有限，small VLM 没能力自主 fuse；Fusion Tokens 加 trajectory supervision 提供了 explicit fusion guidance**。这个发现对小 VLA 设计有重要启示。

---

## 七、实验结果深度分析

### Table 1: RoboTwin 2.0 仿真

按 Short / Medium / Long horizon 分三类任务，对比 SR 和 trajectory length。

关键观察：
- π0 (3B params): Avg SR = 0.47
- GO-1: 0.46
- TinyVLA (1B): 0.07 ← collapse
- SmolVLA (0.45B): 0.29
- SmolVLA† (相同 pretrain config): 0.36 ← 即使公平 pretrain 仍远低于 SwiftVLA
- **SwiftVLA (0.45B, no 4D infer): 0.53** ← 比 π0 高 6 个点
- **SwiftVLA w/ 4D: 0.55**

SwiftVLA 在参数 ~15% π0 的情况下，SR 反而更高。这说明 4D distillation + lightweight backbone 是 viable 的。

Long-Horizon 上的差异最显著：
- SmolVLA: 0.28
- SwiftVLA: 0.56（翻倍！）
- π0: 0.52

Long-horizon 任务需要强 temporal reasoning，SwiftVLA 在这里反超 π0，说明 4D temporal cache 确实 capture 了 long-horizon dependency。

### Table 2: Real-world 实验

三个任务：Clean the Desk / Throw the Bottle / Stack Bowls

- π0: Avg 0.61
- SmolVLA: 0.34
- SmolVLA†: 0.53
- **SwiftVLA: 0.80** ← real-world 上大幅领先 π0
- SwiftVLA w/ 4D: 0.82

Real-world 提升（0.80 vs π0 的 0.61）比 simulation 提升（0.53 vs 0.47）更显著。可能原因：
1. Real-world 有更复杂的 geometry (lighting, occlusion)，4D awareness 价值更高
2. SmolVLA† 公平 pretrain 后 simulation 0.36 → real 0.53，差距比 SwiftVLA 的相对优势在 simulation 中已经被"公平 pretrain"部分缩小

### Table 3: LIBERO Benchmark

LIBERO 有 Spatial / Object / Goal / Long 四个 suite。

SwiftVLA Avg = 94.7，仅次于：
- OpenVLA-OFT (7B): 97.1
- DD-VLA (7B): 96.3
- UniVLA (9B): 95.4

但比 π0 (94.1)、GR00T-N1 (93.9)、SpatialVLA (78.1)、4D-VLA (88.6)、QDepth-VLA (94.9) 都好或持平。

特别值得注意的是 **LIBERO-Long**：
- SwiftVLA: 88.4
- SwiftVLA w/ 4D: 89.0
- π0: 85.2
- 4D-VLA: 79.1

Long suite 上 SwiftVLA 再次胜过大型 baseline，进一步验证 temporal modeling 优势。

但需要指出：OpenVLA-OFT 在 LIBERO-Long 上 94.5 远超 SwiftVLA 的 88.4。OFT (https://arxiv.org/abs/2502.19645) 是 7B 模型 + fine-tuning tricks，可能在 long-horizon 上有 specific advantage。

### Table 4: Edge Device Deployment（Jetson Orin）

| Model | Inference Time | Memory | Avg SR |
|---|---|---|---|
| π0 | 2.966s | 16236 MB | 0.48 |
| SmolVLA | 0.166s | 1397 MB | 0.30 |
| SwiftVLA | 0.167s | 1398 MB | **0.76** |

SwiftVLA 的 inference time 和 memory 几乎等于 SmolVLA（多了 0.001s 和 0.9MB），但 SR 从 0.30 提升到 0.76，**完全免费**地获得了 0.46 的 SR 提升。这是 mask-and-reconstruct 训练策略的核心 payoff：训练时学到的 4D knowledge 完全 distill 进 2D-only inference path。

vs π0: 18× speedup, 12× memory reduction，SR 反而 +0.28。

### Table 9: Fold the Cloth（deformable object, long-horizon）

- π0: 0.45
- SmolVLA: 0.05 ← collapse
- SmolVLA†: 0.30
- SwiftVLA: 0.60
- SwiftVLA w/ 4D: 0.65

Deformable object manipulation 需要 fine-grained 4D understanding（cloth 形变随时间演化），SwiftVLA 大幅领先。这个 task 是 4D awareness 价值的"极限检验"。

---

## 八、Training Procedure 细节

两阶段训练（Appendix B）：

### Stage 1: Pretraining (100k steps)
- 不启用 4D input、Fusion Tokens、mask-and-reconstruct
- 只用 robot action supervision
- Global batch 256, lr 1e-4 → 2.5e-6 (200-step warmup, cosine decay)
- AdamW, β1=0.85, β2=0.9
- Image 512×512

### Stage 2: Continued training (50k steps)
- 从 Stage 1 checkpoint 初始化
- 启用 4D input + Fusion Tokens + mask-and-reconstruct
- lr 5e-5, cosine decay

### Fine-tuning
- Stage 1 (10k steps): 只 action supervision, lr 1e-4
- Stage 2: 启用所有 components

AdamW 的 β1=0.85 比默认 0.9 略低，β2=0.9 比默认 0.999 显著低。这是个 unusual setting，可能是为了训练稳定性——小 batch size + multi-task loss 容易有梯度噪声，低 β2 让二阶矩估计更敏感响应。这个细节值得注意。

参考 AdamW 原文: https://arxiv.org/abs/1711.05101

---

## 九、与相关工作的对比与 positioning

让我把 SwiftVLA 放在 landscape 中：

### 与 SmolVLA 的关系
SmolVLA (https://arxiv.org/abs/2506.01844) 是 SwiftVLA 的直接 baseline 与 backbone 来源。SwiftVLA 在 SmolVLA 基础上加了：
1. 4D feature extractor（training-only）
2. Fusion Tokens + trajectory supervision
3. Mask-and-reconstruct training
4. Reconstruction heads（training-only）

Inference 时 architectural 差异只剩 Fusion Tokens（一些 learnable embeddings，开销可忽略）。所以 SwiftVLA inference speed 几乎等于 SmolVLA。

### 与 π0 的关系
π0 (https://arxiv.org/abs/2410.24164) 用 PaliGemma-3B + Flow Matching decoder。SwiftVLA 借鉴了 π0 的：
- Diffusion-based action generation
- Multi-layer VLM hidden states as condition
- Multi-view image input format

但换了 lightweight backbone + 4D distillation 思路。

### 与 4D-VLA 的关系
4D-VLA (https://arxiv.org/abs/2506.22242) 是另一个用 4D 的工作，但：
- 4D-VLA 用 keyframe sampling 引入多帧，inference overhead 高
- 4D-VLA 在 large VLM (4B) 上工作
- 4D-VLA inference 时仍需 4D input

SwiftVLA 通过 distillation 在 inference 时完全去掉 4D input，是关键 architectural difference。

### 与 SpatialVLA / PointVLA / GeoVLA 对比
这些都是 3D-augmented VLA：
- SpatialVLA: 3D positional encoding + adaptive action network (4B)
- PointVLA: point cloud 作为 auxiliary input，decoupled branch
- GeoVLA: parallel branches + modality experts

SwiftVLA 与它们的区别：
1. **4D 而非 3D**（加 temporal）
2. **从 2D images 推 4D，不需 depth sensor**
3. **Inference 时 drop 4D branch**（其他方法都保留）

### 与 Lift3D / VLM4D 的关系
- Lift3D (https://arxiv.org/abs/2411.18623): 用 2D pretrained model lift 到 3D
- VLM4D: spatiotemporal awareness via fine-tuning

SwiftVLA 不需要 fine-tune VLM for spatiotemporal reasoning，而是用 4D feature supervision 间接 distill。

---

## 十、Criticisms 与 Potential Weaknesses

为了 build critical intuition，让我指出几个 potential issues：

1. **Fusion Tokens 数量未明确**：论文没说 Fusion Tokens 有多少个。如果太多（如 64+），inference 开销可能不"轻"；太少可能 bottleneck 不够 expressive。

2. **Masking probability 未给具体值**：Table 6 验证 mask 有效但没说 mask ratio。

3. **Stage 2 training 50k steps 可能不够**：从 Stage 1 (2D-only action supervision) 切换到 Stage 2 (启用 4D + mask + reconstruction)，模型需要重新调整 representation 适应 4D 信息。50k steps 是否够让 distillation 充分是个 question。

4. **4D extractor 是 frozen**：好处是 stability，坏处是 4D features 可能与 robot task domain 不完全 aligned。预训练 4D VGT 是 general scene geometry，不一定 capture task-relevant 几何（如 gripper-object contact）。

5. **LIBERO 上的对比可能不完全公平**：OpenVLA-OFT 在 LIBERO-Long 上 94.5 远超 SwiftVLA 88.4。OFT 用了 specific fine-tuning tricks，可能 SwiftVLA 没用相同 tricks。论文没有控制这个 variable。

6. **Real-world 只有 4 个 tasks**：Clean the Desk / Throw the Bottle / Stack Bowls / Fold the Cloth，覆盖面有限。需要更多 long-horizon, contact-rich, bimanual tasks 验证 generalization。

7. **Front view only 4D**：Eq. (5) 表明只有 front view 的 4D feature 进 VLM。这是 efficiency 决定，但 spatial coverage 可能受限——front view 的 4D features 主要是 front camera visible 区域的几何，左右两侧的 4D 信息只在 cache 里被间接编码。

---

## 十一、Key Takeaways for VLA Design

总结几个对 VLA 领域有普遍意义的 insights：

1. **Modality distillation works**：训练时用强 modality (4D) supervision，inference 时 drop 该 modality，性能基本保持。这是 lightweight multimodal 部署的 viable pattern。可能延伸到 audio / depth / tactile 等其他 modality。

2. **Small VLM 不能自主 fuse multimodal**：必须用 explicit mechanism (Fusion Tokens) + task-relevant supervision (trajectory prediction) 引导 fusion。直接 concat 多 modality 输入效果有限（Table 5 第二行只比第一行高 0.04）。

3. **Streaming cache > Replay all frames**：FIFO cache + cross-attention 比保留所有历史 frames 更高效，且 random cache size 训练提升 generalization。

4. **Two-stage training 有效**：先 2D-only 学 action space adaptation，再启用 4D + auxiliary objectives，避免初期训练崩溃。

5. **4D 不需要额外 sensor**：从 2D images 用 pretrained 4D VGT 推 4D features，避免 depth camera / LiDAR 硬件复杂度。这让方法 deployable 到只配 RGB cameras 的标准 robot。

6. **Inference path 设计优先于 training path**：SwiftVLA 的 architectural choices (4D extractor, reconstruction heads, trajectory head) 都是 training-only，inference 时只留 VLM + action expert + Fusion Tokens。这种"train heavy, infer light"的设计哲学与 DistillBERT、MoE distillation 等思想一脉相承。

---

## 十二、对未来工作的可能启示

基于 SwiftVLA 思路，可能延伸方向：

1. **Other modality distillation**：能否把 tactile / force-torque 信息 training 时注入、inference 时 drop？类似 distill tactile sensing into visual representation。

2. **Self-distillation without external 4D model**：能否用 SwiftVLA 自己（with 4D）作为 teacher distill 给自己（without 4D）？避免依赖外部 4D VGT。

3. **Adaptive cache size at inference**：根据 task complexity 动态调整 cache size K，简单任务小 cache 快，复杂任务大 cache 强。

4. **Fusion Token sparsification**：能否 prune Fusion Tokens 或者用 MoE-style routing 让不同 task 用不同 Fusion Token subset？

5. **3D-aware action chunk**：能否让 action expert 直接 output 3D trajectory + 2D action chunks dual representation，强化 spatial grounding？

6. **World model integration**：4D features 本质是 world state representation，能否与 world model (如 GAIA, DreamerV3) 结合，让 VLA 具备 model-based planning？

参考相关工作：
- World models for robotics: https://arxiv.org/abs/1910.01323 (Dreamer)
- Tactile VLA: https://arxiv.org/abs/2410.01379
- MoE in VLA: https://arxiv.org/abs/2407.06498

---

## 十三、整体评价

SwiftVLA 是一个 **elegant engineering contribution**。它没有提出 fundamentally new architecture 或 learning paradigm，但通过精心组合 existing components (SmolVLM + Streaming 4D VGT + diffusion action expert + Q-Former-style Fusion Tokens + MAE-style mask-reconstruct)，达到了一个非常实用的目标：**在 edge device 上以 0.45B 参数跑出接近或超过 3B-7B VLA 的性能**。

最 impressive 的不是单个数字（如 18× speedup），而是 **整套设计逻辑的自洽性**：每个 component 都对应一个具体问题（small VLM 弱 → 4D 辅助 / 4D 推理慢 → mask-reconstruct distill / small VLM 不会 fuse → Fusion Tokens + trajectory supervision / multi-view 4D 太重 → front-view only + cache summarize），且 ablation 清晰验证每个 component 的 contribution。

这篇 paper 对 real-world robot VLA deployment 的 practical 价值很大。如果 SwiftVLA 的 distillation 思路能扩展到更多 modality 和 task，可能成为 lightweight embodied AI 的标准 pattern 之一。

Project page: https://Swiftvla.github.io  
SmolVLA: https://arxiv.org/abs/2506.01844  
π0: https://arxiv.org/abs/2410.24164  
Streaming 4D VGT: https://arxiv.org/abs/2507.11539  
SmolVLM: https://arxiv.org/abs/2504.05299  
LIBERO benchmark: https://arxiv.org/abs/2306.03310  
RoboTwin 2.0: https://arxiv.org/abs/2506.18088  
BLIP-2 Q-Former: https://arxiv.org/abs/2302.14082  
MAE: https://arxiv.org/abs/2111.06377  
SigLIP: https://arxiv.org/abs/2303.15343
