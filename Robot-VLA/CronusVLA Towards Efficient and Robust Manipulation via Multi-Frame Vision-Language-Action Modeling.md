---
source_pdf: CronusVLA Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action
  Modeling.pdf
paper_sha256: 80511e4e7dcf36f541f50f9d4eb845226387d9b78df838a6be22ecedca74dc89
processed_at: '2026-08-03T17:52:17-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CronusVLA：用大白话讲清楚这篇paper在干啥

## 一句话总结

这paper解决了一个很实在的问题：**现在的robot policy都是看一张图做一次决策，没有"记忆"，碰到状态模糊或者画面被遮挡就歇菜了。但直接把多帧历史图像喂给VLM又会让计算量爆炸。CronusVLA的招数是：先用单帧预训练打好基础，再在feature层面做时间聚合，这样既能用上历史信息，又不会让VLM算到天荒地老。**

好，下面我用最接地气的方式把整个故事讲一遍。

---

## 1. 为什么单帧VLA是个坑

先说现状。OpenVLA、RT-2这些主流VLA怎么做policy的？拿到一张图$I_t$，加上语言指令$l$，输出一个7维action（末端执行器的位姿增量）。Action被离散化成256个bin的token，然后像LLM生成文本一样autoregressive地一个个吐出来。

听起来挺优雅，复用了LLM的整个基础设施。但有两个致命问题：

### 问题一：状态歧义

举个例子，"按红、黄、绿三个按钮"这个long-horizon task。你按红色按钮前和按红色按钮后，相机看到的画面可能几乎一模一样——因为按钮按下去了视觉变化很小，手的位置也差不多。单帧模型完全分不清"我刚刚按了红色，现在该按黄色"和"我还没按红色，该按红色"这两个状态。

结果就是OpenVLA反复按红色按钮，按了又按，卡死在那。Fig.12的case study就是这么挂的。

### 问题二：画面被干扰就完蛋

现实部署中，相机画面经常被干扰——有人走过挡住镜头、灯光闪烁、画面抖动、帧丢失。单帧模型看到一帧坏画面，就输出一帧坏action，没有任何fallback。

Low-level policy（Diffusion Policy、ACT）早就用多帧输入解决这个问题了。但它们模型小，多帧计算扛得住。VLA的VLM backbone动辄7B参数，self-attention是$O(N^2)$的，你塞7帧图像进去，token数量翻7倍，计算量翻49倍。在大规模预训练上谁玩得起？

---

## 2. CronusVLA的核心思路：在Feature层面搞时间聚合

这就是CronusVLA最精髓的地方。它不去碰VLM内部的attention，而是在VLM的输出端动手术。

### 2.1 一个Learnable Feature Token搞定一切

具体做法：在VLM的所有vision token和text token之后，加一个**learnable feature token**$f_t$。这个token的位置很特殊——它在causal attention中能看到前面所有的vision和text token，所以它会"吸收"整个VLM对当前帧的理解。

$$f_t = \mathrm{VL}(I_t, l)$$

这个公式说的是：给VLM输入当前帧$I_t$和指令$l$，它前向跑一遍，输出这个learnable feature $f_t$。注意，**不是输出文本token，是输出这个hidden feature**。维度$d$就是VLM的hidden dimension（Llama2-7B是4096）。

关键点：**VLM还是按单帧方式处理每一帧**。每帧图像独立过VLM，不跨帧attention。这就避开了$O(M^2)$的爆炸。

### 2.2 Feature Chunking：历史信息的FIFO队列

定义一个feature chunking：

$$\mathcal{F}_t^M = \{f_{t-M+1}, \ldots, f_{t-1}, f_t\}$$

就是从$t-M+1$到$t$时刻的M个feature，组成一个chunk。M是历史长度，论文7B用M=7（当前帧+6历史帧）。

训练时，把batch dimension重组成$B \times M$个单帧输入，VLM独立处理每一帧。推理时用FIFO queue：每一步只算当前帧的feature，旧的feature从队列里拿出来用。这样推理速度基本不随M增加而下降。

实测数据（Table 18）：CronusVLA-7B在7帧时8.7 Hz，比OpenVLA单帧的5.2 Hz还快。为什么？因为OpenVLA要autoregressive地decode 7个action token（一个个生成），CronusVLA是single forward pass输出continuous feature，然后diffusion decoder一次性denoise出action chunk。

---

## 3. Cross-Frame Decoder：怎么从多帧Feature生成Action

有了feature chunking $\mathcal{F}_t^M$，怎么生成action？论文用DiT（Diffusion Transformer）架构做decoder。

### 3.1 为什么要用Diffusion

因为action是continuous的（7维向量），diffusion policy在low-level task上已经被证明很好用（Chi et al. 2023）。而且diffusion可以一次生成action chunk（K步future action），不是一步步生成，适合receding horizon control。

### 3.2 Feature Modulator：平衡当前帧和历史帧

这里有个设计细节值得讲。你有1个current feature $f_t$和M-1个past features $f_{t-M+1:t-1}$，数量不平衡。如果直接concat或者一视同仁处理，past features会"淹没"current feature的信号——毕竟当前action主要由当前状态决定，历史只是辅助。

CronusVLA的招数是**channel splitting**：把current feature $f_t$通过一个Linear层扩展维度，再split成M-1份，这样current feature就有M-1个copy，和past features一一对应：

$$\tilde{f}_t = \mathrm{DIV}(f_t), \quad f_t \in \mathbb{R}^d, \tilde{f}_t \in \mathbb{R}^{(M-1) \times d}$$

然后通过MLP融合：

$$Z_f = \mathrm{MD}(\mathcal{F}_t^M) = \mathrm{MLP}(f_{t-M+1:t-1}, \tilde{f}_t)$$

输出$Z_f \in \mathbb{R}^{2(M-1) \times d'}$，$d'$是decoder的hidden dim（768）。

这个$Z_f$在DiT的cross-attention里作为key和value，noised action $\hat{a}$作为query。通过100步diffusion denoising，最终输出clean action chunk $a_{t:t+K-1}$。

### 3.3 为什么用Cross-Attention不用Self-Attention

Table 5的ablation：去掉cross-attention换成self-attention，success rate从70.9%降到68.3%。但更关键的是复杂度。

Self-attention是把所有token（M个feature + K个noised action）放一起做全局attention，复杂度$O((M+K)^2)$。当M增大时平方增长。

Cross-attention是action token去query feature token，复杂度$O(M \times K)$，线性于M。这意味着你可以用很大的M（比如20帧历史）而decoder不会变慢太多。

---

## 4. Multi-Frame Regularization：保护单帧感知能力

这个设计我觉得是全文最聪明的地方。

### 4.1 问题：多帧训练会"污染"单帧感知

如果你直接让VLM backbone同时学两件事——单帧感知（识别物体、理解场景）和多帧时间推理（从历史推断当前phase）——这两者的梯度会互相干扰。VLM在大规模单帧预训练时学到的embodied perception能力，可能被多帧任务"带偏"。

### 4.2 解法：对历史帧Stop-Gradient

做法很直接：**历史帧的feature不回传梯度**。

$$\hat{f}_{t-M+1:t-1} = \{\mathrm{sg}(\mathrm{VL}(I_{t-k}, l)), k=1, \ldots, M-1\}$$

$\mathrm{sg}$是stop-gradient操作。只有当前帧$f_t$的梯度会更新VLM backbone，历史帧只作为"固定输入"提供给decoder。

完整的diffusion loss：

$$\mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \mathbf{I}), i}\left[\left\|\hat{\epsilon}^i - \epsilon_\theta(t, \hat{f}_{t-M+1:t-1}, f_t)\right\|_2\right]$$

- $\epsilon$：从标准正态分布采样的noise
- $i$：diffusion denoising step index（从0到T）
- $\hat{\epsilon}^i$：模型预测的noise
- $\epsilon_\theta$：参数为$\theta$的noise predictor，条件是timestep、stop-gradient的历史features、和当前feature $f_t$

### 4.3 为什么这个设计work

本质上是在说：**VLM backbone的职责是单帧感知，decoder的职责是时间推理，两者各司其职，不要互相干扰。**

这让我想到模块化训练的一般哲学——你让一个模块同时学太多东西，它的learning dynamics会混乱。stop-gradient是一种硬性的"职责划分"。

Table 4的ablation数据：
- 不加regularization（+M.F.+Dec.+V.L.）：67.2%
- 加regularization（完整模型）：70.9%
- Fig.13还显示convergence速度明显加快

从scratch训练（不利用单帧预训练）在50K steps内只能达到10-18%，而CronusVLA从预训练checkpoint出发50K steps达到70.9%。这说明**单帧预训练建立的embodied foundation至关重要**。

---

## 5. 复杂度对比：为什么这个设计真的快

来算笔账。假设VLM把每帧图像切成$P$个token（通常256），instruction是$I$个token。

### Naive多帧方法

直接把M+1帧图像塞进VLM，总token数$(M+1) \cdot P + I$，self-attention复杂度：

$$O\left(((M+1) \cdot P + I)^2\right) \simeq O(M^2 \cdot P^2)$$

因为$P \gg I$，主导项是$(M+1)^2 \cdot P^2$。M=7时比单帧慢49倍。

### CronusVLA

每帧独立过VLM，feature层面做时间聚合：

$$O(M \cdot T_{VLM}) + O(M \cdot T_{decoder}) \simeq O(M \cdot T_{VLM})$$

因为$T_{decoder} \ll T_{VLM}$（decoder只有135M参数，VLM有7B），所以整体线性于M。M=7时比单帧慢7倍（而且推理时用FIFO queue，只算当前帧，所以实际和单帧一样快）。

实测（Table 18）：
- OpenVLA（单帧）：5.2 Hz，192ms latency
- OpenVLA + 7帧（naive）：3.1 Hz，263ms latency
- CronusVLA-7B（7帧）：8.7 Hz，115ms latency

CronusVLA加了6帧历史反而比OpenVLA单帧还快，主要因为：消除了autoregressive action token decoding（7个token一个个生成），换成single forward + diffusion denoise（可以用few-step solver加速）。

---

## 6. SimplerEnv-OR：专门测Robustness的Benchmark

这是论文的另一个contribution。现有benchmark只测干净环境下的success rate，但真实部署的挑战是disturbance。

### 6.1 设计

基于SimplerEnv的WidowX Robot Visual Matching设置，加了24种disturbance，分两个维度：

**Spatial Dimension**（干扰类型）：
- Global：Blurring、Jittering、Frame Dropping、Full Occlusion
- Local：Overexposing、Partial Occlusion
- Discrete：Gaussian Noise、Impulse Noise

**Temporal Dimension**（干扰频率）：
- Constant (1:0)：每帧都干扰
- Cyclic (1:1)：一帧clean一帧干扰交替
- Sparse (1:3, 1:5)：每3-5帧干扰一次

总共2300 trials，120 severity levels。

### 6.2 Robustness Score

$$\text{R-Score}^i = 100 \times \frac{SR^i}{SR}$$

$SR$是原始任务的success rate，$SR^i$是disturbance $i$下的success rate。R-Score=100表示完全不受影响，越低表示退化越严重。

### 6.3 结果分析（Table 3）

几个有意思的发现：

**Temporal Dimension**：
- Constant (1:0) 下：CronusVLA R-Score 61.2，CogACT 53.3，TraceVLA 59.2
- Sparse (1:3) 下：CronusVLA 96.2（几乎不受影响），CogACT 80.2，TraceVLA 78.0

单帧模型（CogACT、SpatialVLA）在高频干扰下输出out-of-distribution action，直接失败。多帧模型（RoboVLMs、TraceVLA）虽然能抵抗，但依赖准确历史信息，历史帧也被干扰时就犹豫不决。CronusVLA因为feature chunking设计，即使当前帧坏了也能从历史推断。

**Spatial Dimension**：
- Global：CronusVLA 85.4，CogACT 60.2，SpatialVLA 57.6
- Local：CronusVLA 96.6，RoboVLMs 83.3
- Discrete：CronusVLA 80.2，CogACT 87.4

**最反直觉的发现**：SpatialVLA在原始SimplerEnv上51.2%，比RoboVLMs的43.5%强。但在SimplerEnv-OR上，SpatialVLA只有54.4 R-Score，RoboVLMs有67.4。**Performance高不等于Robustness高**，multi-frame modeling提供了本质不同的能力维度。

---

## 7. 实验数据全览

### 7.1 SimplerEnv（Table 1）

CronusVLA-7B在12个task上average 70.9%：

| Method | Google VM | Google VA | WidowX VM | Avg |
|--------|-----------|-----------|-----------|-----|
| OpenVLA-7B | 35.1 | 35.9 | 3.1 | 24.7 |
| CogACT-7B | 74.8 | 61.3 | 55.2 | 63.8 |
| TraceVLA-7B | 45.8 | 49.8 | 27.7 | 41.1 |
| RoboVLMs-2B | 57.0 | 30.9 | 42.7 | 43.5 |
| SpatialVLA-3B | 56.0 | 51.8 | 45.8 | 51.2 |
| **CronusVLA-7B** | **78.6** | **73.8** | **60.4** | **70.9** |
| CronusVLA-0.5B | 70.5 | 57.8 | 39.6 | 56.0 |

亮点：
- Put in Drawer（long-horizon）：CronusVLA 64.8%（VM）/65.1%（VA），OpenVLA 0.0%，SpatialVLA 0.0%
- CronusVLA-0.5B（用Qwen2.5-0.5B）超过了很多2B-7B模型

### 7.2 LIBERO（Table 2, 16）

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **CronusVLA** | **97.3** | **99.6** | **96.9** | **94.0** | **97.0** |

CronusVLA只用wrist view（无state输入），30K steps，达到97.0%，和用state+wrist view+150K steps的OpenVLA-OFT（97.1%）持平。

### 7.3 Real-World（Franka Research 3）

三类task：

**Simple Pick-and-Place**：stack cubes，CronusVLA 48% place success，DP3 12%，OpenVLA 28%。

**Long-horizon**：Press buttons in order，CronusVLA 三步96%/92%/88%，OpenVLA 72%/64%/40%。OpenVLA的failure mode就是反复按同一按钮（state ambiguity）。

**Robustness**：Camera occlusion下CronusVLA 64%，OpenVLA 20%，DP3 12%。Generalization to unseen objects 72% vs 60% vs 52%。

---

## 8. 我的一些直觉和联想

### 8.1 Feature Token作为"接口"的普适性

CronusVLA的learnable feature token本质上是一个"接口"——它把VLM的输出压缩成一个向量，供下游模块使用。这和BLIP-2的Q-Former、Flamingo的Perceiver Resampler是同一类思想。

区别在于CronusVLA不是为了compress visual token（VLM的输出还是完整的），而是为了提供一个temporal context的"锚点"。这个pattern可以推广到：
- Video understanding：每帧独立过VLM，feature level做temporal attention
- Multi-view observation：每个view独立编码，feature level融合
- Multi-agent：每个agent的observation独立编码，feature level通信

### 8.2 Stop-Gradient作为模块化训练工具

这个trick让我想到MoCo的momentum encoder、Decision Transformer中对state representation的frozen处理、EMA teacher在self-supervised learning中的使用。

核心思想是：**当你想让一个模块学A，另一个模块学B，但它们有shared pathway时，用stop-gradient切断其中一个方向的梯度流，强制职责分离。**

在CronusVLA中，VLM backbone只从当前帧学单帧感知，decoder从多帧feature学时间推理。如果让backbone也接收历史帧的梯度，它会试图"记住"历史信息，破坏单帧感知的clarity。

### 8.3 Frame Number的Sweet Spot

Fig.5显示：7B在7帧时最好，0.5B在4帧时最好。这暗示**最优frame number是model capacity的函数**。

小模型capacity有限，处理太多frame会overload——冗余信息dilute信号。大模型能handle更多redundancy，从更多历史中提取有用cue。

这让我想到Transformer的context length问题——更长context不一定更好，因为attention会被irrelevant token稀释。但在robotics中，这个效应可能更强，因为相邻帧的信息冗余度极高（物体位置变化很小）。

### 8.4 从Discrete Token到Continuous Feature的"升级"

CronusVLA的两阶段训练（discrete token预训练 → continuous feature post-training）是一种representation升级。类似pattern：
- VQ-VAE（discrete codebook）→ VAE（continuous latent）
- ImageGPT（autoregressive pixel）→ Diffusion（continuous denoising）
- GPT（discrete text token）→ Continuous embedding在某些场景的使用

这种"先学简单表示建立基础，再升级到复杂表示做refinement"的curriculum思想很普遍。CronusVLA的成功说明它也适用于VLA。

### 8.5 Performance ≠ Robustness

SimplerEnv-OR揭示的现象很重要：SpatialVLA在干净环境比RoboVLMs强，但在干扰环境反而更弱。

这让我想到ImageNet分类中的现象——在干净ImageNet上accuracy高的模型，在ImageNet-C（corruption robustness benchmark）上未必强。Robustness是独立的dimension，需要专门的评估和训练策略。

VLA领域可能也需要类似的robustness-oriented研究：adversarial training、temporal consistency loss、disturbance augmentation等。

### 8.6 和π0、GR00T的定位差异

π0和GR00T-N1是concurrent work，用flow matching + state + wrist view从scratch训练。它们的performance很强（LIBERO 94.2%、93.9%），但需要大量compute和multi-modal input。

CronusVLA的定位不同：它从已有的single-frame VLA checkpoint出发，用50K steps的post-training升级到multi-frame。这意味着：
- 可以复用社区checkpoint（OpenVLA、SpatialVLA等）
- 训练成本远低于from scratch
- 只需要third-person view，deployment门槛低

这是"efficient adaptation"和"powerful from scratch"两种哲学的差异。在LLM时代，efficient adaptation（LoRA、adapter、instruction tuning）越来越重要，CronusVLA把这个思路带到了VLA。

---

## 9. 局限性和未来方向

Paper在Section I讨论了局限：

**Inter-frame redundancy**：相邻帧信息冗余度高，LLM处理冗余token浪费compute。未来可以探索inter-frame difference、cache机制。

**Language reasoning underutilization**：CronusVLA没有显式利用LLM的语言推理能力，只是把VLM当feature extractor。未来可以结合chain-of-thought、explicit planning。

**Single-view limitation**：只用third-person view，没有wrist view和proprioceptive state。未来可以扩展到multi-view + state-conditioned temporal modeling。

---

## 10. 最终感想

这篇paper给我的最大启发是：**在VLM时代，efficient adaptation比from-scratch training更重要**。CronusVLA用50K steps就把single-frame VLA升级成multi-frame，且推理更快、performance更高、robustness更强。这种"增量式能力扩展"的思路，可能是未来embodied AI的主流范式。

核心设计——feature-level temporal aggregation + stop-gradient regularization + cross-attention decoder——形成了一个coherent的story：让VLM做它擅长的（单帧感知），让decoder做它擅长的（时间推理），用feature token作为两者的接口。

**Reference Links：**
- CronusVLA项目页：https://cronus-vla.github.io/
- OpenVLA（基础模型）：https://github.com/openvla/openvla
- SimplerEnv benchmark：https://simpler-env.github.io/
- LIBERO benchmark：https://lifelong-robot-learning.github.io/LIBERO/
- Open X-Embodiment：https://robotics-transformer-x.github.io/
- BridgeData V2：https://github.com/Tony-Zhao-Research/BridgeData_V2
- DiT（decoder架构参考）：https://github.com/facebookresearch/DiT
- Diffusion Policy：https://diffusion-policy.cs.columbia.edu/
- π0：https://www.physicalintelligence.company/blog/pi0
- GR00T N1：https://huggingface.co/nvidia/GR00T-N1-2B
- DINOv2（vision encoder）：https://github.com/facebookresearch/dinov2
- SigLIP（vision encoder）：https://github.com/google-research/big_vision

---

# CronusVLA 深度解读：Multi-Frame VLA 的高效适配范式

作为 Karpathy，我读了这篇 paper 后直觉上感受到几个关键的设计哲学，让我逐一拆解给你听。

## 1. 核心问题：为什么 Single-Frame VLA 不够用

当前 VLA 领域的主流范式（OpenVLA、RT-2、SpatialVLA）建立在 VLM 的 single-frame 基础设施上——把 7D continuous action 离散化成 256 个 bins 的 token，做 next-token prediction。这种设计有两个根本局限：

**第一，state ambiguity。** 在 long-horizon task（比如 press buttons in order）中，按按钮前和按按钮后的 observation 在视觉上可能极其相似。Single-frame model 无法区分"已经按过红色按钮，准备按黄色"和"刚开始要按红色"这两种状态。论文 Fig.12 的 case study 直观展示了 OpenVLA 反复按同一个按钮的 failure mode。

**第二，observational robustness。** 当当前帧被 occlusion、blur、noise 污染时，single-frame model 没有任何 fallback。而 multi-frame policy 可以 rely on 历史帧推断当前应该做什么。

Low-level policy（Diffusion Policy、ACT）早就知道这件事，它们天然处理多帧。但 VLA 的问题是：**VLM backbone 的 self-attention 是 O(N²) 的，直接塞 M 帧图像进去，token 数量从 P+I 变成 (M+1)·P+I，复杂度爆炸到 O(M²)**。在大规模 pretraining 上根本玩不起。

这就是 CronusVLA 要解决的核心矛盾。

## 2. 两阶段训练：从 Discrete Token 到 Continuous Feature

### 2.1 Single-Frame Pretraining（Stage 1）

这一步基本沿用 OpenVLA 的配方：用 OXE 的 27 个 dataset，把 action 通过 extended tokenizer 映射到 256 bins，做标准的 autoregressive next-token prediction。Loss 是 cross-entropy：

$$\mathcal{L}_{CE} = -\sum_{t=1}^{T} \log P_\theta(y_t | y_{<t}, x)$$

变量含义：
- $y_t$：第 $t$ 步的 target action token
- $y_{<t}$：前面已经生成的 token 序列
- $x$：输入模态，包含 image $I_t$ 和 language instruction $l$
- $P_\theta$：参数为 $\theta$ 的模型预测的 token 分布

这一步的 output 是一个"basic single-frame VLA"，它的 vision encoder（SigLIP + DINOv2）已经学会感知 embodied scene，LLM backbone 已经学会把视觉和语言 grounding 到 action。

### 2.2 Multi-Frame Post-Training（Stage 2）：核心创新

这里的设计非常精妙。关键 insight 是：**不要把多帧图像塞进 VLM，而是在 feature level 做时间聚合。**

具体来说，CronusVLA 引入一个 **learnable feature token** $f_t \in \mathbb{R}^d$，它被插入到 VLM 的 hidden layer 中，位于所有 vision token 和 text token 之后。VLM 仍然按 single-frame 方式处理每帧图像：

$$f_t = \mathrm{VL}(I_t, l)$$

这里 $\mathrm{VL}$ 表示 VLM backbone 的前向计算（不生成文本，只输出这个 learnable feature）。这个 feature 吸收了 VLM 的 embodied vision-language summarization 能力。

然后定义 **feature chunking**：

$$\mathcal{F}_t^M = \{f_{t-M+1}, \ldots, f_{t-1}, f_t\} = f_{t-M+1:t}$$

它表示从 $t-M+1$ 到 $t$ 时刻的 M 个 learnable feature 的集合。在 training 时，batch dimension 被重组为 $B \times M$ 个 single-frame input，VLM 独立处理每一帧；在 inference 时，用 FIFO queue 缓存历史 feature，每步只需计算当前帧的 feature。

这个设计的优雅之处在于：**VLM 的 computation 是 O(M·T_VLM) 级别，每个 frame 独立处理，attention 不跨帧。** 时间建模被完全 decouple 到一个小型的 cross-frame decoder 中。

## 3. Cross-Frame Decoder：DiT + Cross-Attention

Decoder 的任务是：给定 feature chunking $\mathcal{F}_t^M$，预测一个 action chunking $a_{t:t+K-1}$（K 步 future action）。架构基于 DiT（Diffusion Transformer），用 diffusion loss 训练。

### 3.1 Feature Modulator

这里有一个关键问题：当前帧 $f_t$ 和过去 M-1 帧的 feature 数量不平衡。如果直接 concat 或简单处理，past features 会"淹没"current feature 的信号。

CronusVLA 的解决方案是 **channel splitting + dimensionality expansion**：

$$\tilde{f}_t = \mathrm{DIV}(f_t), \quad \text{where } f_t \in \mathbb{R}^d, \tilde{f}_t \in \mathbb{R}^{(M-1) \times d}$$

- $\mathrm{DIV}$：先经过一个 Linear 层扩展维度，再 split 成 M-1 份
- $f_t$：当前帧的 learnable feature，维度 $d$
- $\tilde{f}_t$：扩展后的当前帧 feature，维度 $(M-1) \times d$，这样能和 M-1 个 past frame 对齐

然后通过 modulator（MLP）融合：

$$Z_f = \mathrm{MD}(\mathcal{F}_t^M) = \mathrm{MLP}(f_{t-M+1:t-1}, \tilde{f}_t)$$

- $Z_f \in \mathbb{R}^{2(M-1) \times d'}$：modulated feature
- $d'$：decoder 的 hidden dimension（768）
- $f_{t-M+1:t-1}$：过去 M-1 帧的 raw feature
- $\tilde{f}_t$：当前帧扩展后的 feature

这个 $Z_f$ 会被送入 cross-attention 的 key/value，而 noised action $\hat{a}$ 作为 query。通过 iterative denoising（100 步 diffusion schedule），最终输出 clean action。

### 3.2 为什么用 Cross-Attention 而不是 Self-Attention

Table 5 的 ablation 显示：去掉 cross-attention 换成 self-attention，平均 success rate 从 70.9% 降到 68.3%。更重要的是复杂度——self-attention 对所有 token（包括 multi-frame feature 和 noised action）做全局 attention，是 O((2F)²) = O(F²)；cross-attention 只让 action token query feature token，是 O(F·A)，其中 A 是 action chunk length，远小于 F。这保证了 frame number 增加时复杂度线性增长。

## 4. Multi-Frame Regularization：保护 Single-Frame Perception

这是一个很巧妙的 training trick。问题在于：如果直接让 backbone 同时学习 single-frame perception 和 multi-frame temporal modeling，backbone 的梯度信号会被 temporal 任务"污染"，破坏预训练得到的单帧感知能力。

解决方案是对 past frames 做 **stop-gradient**：

$$\hat{f}_{t-M+1:t-1} = \{\mathrm{sg}(\mathrm{VL}(I_{t-k}, l)), k=1, \ldots, M-1\}$$

- $\mathrm{sg}$：stop-gradient 操作，阻断反向传播
- $\mathrm{VL}(I_{t-k}, l)$：第 $t-k$ 帧经过 VLM 得到的 learnable feature
- $k$：从 1 到 M-1，覆盖所有历史帧

只有当前帧 $f_t$ 的梯度会回传到 backbone。完整的 diffusion loss：

$$\mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, \mathbf{I}), i}\left[\left\|\hat{\epsilon}^i - \epsilon_\theta(t, \hat{f}_{t-M+1:t-1}, f_t)\right\|_2\right]$$

- $\epsilon$：从标准正态分布采样的 noise
- $i$：diffusion 的 denoising step index
- $\hat{\epsilon}^i$：模型预测的第 $i$ 步 noise
- $\epsilon_\theta$：参数为 $\theta$ 的 noise prediction network，条件是 timestep $t$、stop-gradient 的 past features、和 current feature $f_t$

这个设计有两个好处：
1. **训练效率**：past frames 只需 forward 不需 backward，显存和计算开销大幅减少
2. **收敛稳定性**：backbone 的更新逻辑和 single-frame pretraining 一致，避免了 multi-frame task 的"遗忘"问题

Table 4 的 ablation 显示，加上 regularization 后从 67.2% 提升到 70.9%，convergence 也明显加快（Fig.13）。

## 5. 复杂度对比：O(M²) vs O(M)

论文 Appendix H 给出了清晰的复杂度分析。

**Naive multi-frame 方法**（直接把多帧图像塞进 VLM）：

$$O\left(((M+1) \cdot \mathcal{P} + \mathcal{I})^2\right) \simeq O(M^2)$$

- $M$：历史帧数
- $\mathcal{P}$：每帧图像被切成的 token 数（通常 256 左右）
- $\mathcal{I}$：instruction token 数
- 因为 $\mathcal{P} \gg \mathcal{I}$，所以主导项是 $(M+1)^2 \cdot \mathcal{P}^2$

**CronusVLA**：

$$O(M \cdot T_{VLM}) + O(M \cdot T_{decoder}) \simeq O(M \cdot T_{VLM}) \simeq O(M)$$

- $T_{VLM}$：VLM 单帧推理时间
- $T_{decoder}$：decoder 单帧推理时间
- 因为 $T_{decoder} \ll T_{VLM}$（decoder 只有 135M 参数，VLM 有 7B），所以整体线性于 M

实测结果（Table 18）：CronusVLA-7B 在 7 帧时推理速度 8.7 Hz，而 OpenVLA 单帧只有 5.2 Hz——**加了 6 帧历史反而更快**，因为消除了 autoregressive decoding（discrete action token 一个个生成），换成 single forward pass 的 continuous feature prediction。

## 6. SimplerEnv-OR Benchmark：量化 Robustness

这是论文的一个 contribution。现有 benchmark（SimplerEnv、LIBERO）只测 task success rate，不测 disturbance 下的 robustness。CronusVLA 提出 SimplerEnv-OR，包含：

**Spatial Dimension**（干扰类型）：
- **Global**：Blurring、Jittering、Frame Dropping、Full Occlusion（影响整帧）
- **Local**：Overexposing、Partial Occlusion（影响局部区域）
- **Discrete**：Gaussian Noise、Impulse Noise（稀疏像素 corruption）

**Temporal Dimension**（干扰频率）：
- **Constant (1:0)**：每帧都 disturbed（最严苛）
- **Cyclic (1:1)**：一帧 clean 一帧 disturbed 交替
- **Sparse (1:3, 1:5)**：每 3-5 帧 disturbed 一次

总共 24 种 disturbance × 120 severity levels × 2300 trials。

**Robustness Score 定义**：

$$\text{R-Score}^i = 100 \times \frac{SR^i}{SR}$$

- $SR$：原始 WR-VM 任务的 average success rate
- $SR^i$：在 disturbance setting $i$ 下的 success rate
- R-Score = 100 表示完全 unaffected，<100 表示有 degradation

测试结果（Table 3）很有意思：
- **Temporal**：Constant (1:0) 下 CronusVLA R-Score 61.2，CogACT 只有 53.3，TraceVLA 59.2；Sparse (1:3) 下 CronusVLA 96.2，几乎不受影响
- **Spatial**：Global disturbance 下 CronusVLA R-Score 85.4，CogACT 60.2，SpatialVLA 57.6
- **Total Avg**：CronusVLA 86.9，CogACT 72.1，RoboVLMs 67.4

一个值得注意的现象：SpatialVLA 在原始 SimplerEnv 上比 RoboVLMs 强（51.2 vs 43.5），但在 SimplerEnv-OR 上反而更差（54.4 vs 67.4）。这说明 **single-frame 的高 performance 不等于 robustness**，multi-frame modeling 提供了本质不同的能力。

## 7. 实验结果分析

### 7.1 SimplerEnv（Table 1）

CronusVLA-7B 在 12 个 task 上 average 70.9%：
- Google Robot VM：78.6（+71.6% over TraceVLA，+138.8% over RoboVLMs）
- Google Robot VA：73.8
- WidowX VM：60.4（+41.5% over SpatialVLA）

特别值得注意的是 **Put in Drawer** 这个 long-horizon task：CronusVLA 达到 64.8% (VM) 和 65.1% (VA)，而 OpenVLA 是 0.0%，SpatialVLA 是 0.0%，TraceVLA 是 11.1%。这是 multi-frame modeling 在 long-horizon task 上的直接体现。

CronusVLA-0.5B 的表现也很亮眼：用 Qwen2.5 0.5B 作为 backbone，平均 56.0%，超过了很多 2B-7B 的 model（比如 OpenVLA-7B 的 24.7%）。这说明 **effective modeling 比 parameter scale 更重要**。

### 7.2 LIBERO（Table 2, 16）

LIBERO 有四个 suite：Spatial、Object、Goal、Long。CronusVLA 平均 97.0%，Long suite 达到 94.0%（+40.3% over OpenVLA 的 53.7%）。和 SOTA OpenVLA-OFT（97.1%）几乎持平，但 OpenVLA-OFT 用了 wrist view + state input + 150K steps，CronusVLA 只用 wrist view + 30K steps。

### 7.3 Real-World（Table 19-21）

在 Franka Research 3 上测试了三类 task：

**Simple Pick-and-Place**：CronusVLA 在 stack cubes 上达到 48% place success（DP3 12%，OpenVLA 28%）。

**Long-horizon**：Press buttons in order，CronusVLA 三步分别 96%、92%、88%；OpenVLA 是 72%、64%、40%。OpenVLA 的问题正是 state ambiguity——它分不清"按了红按钮"和"要按红按钮"的状态。

**Generalization & Robustness**：Camera occlusion 下 CronusVLA 64%，OpenVLA 20%，DP3 12%。这是 multi-frame modeling 的直接优势。

## 8. 我的联想与思考

读这篇 paper 时我想到几个更深层的问题：

### 8.1 Feature-Level Temporal Aggregation 的普适性

CronusVLA 的核心 trick——**在 VLM 的 hidden state 上加一个 learnable token，把它当 multi-frame aggregation 的载体**——本质上是一种"adapter"思想。这让我想到 Flamingo 的 perceiver resampler、Q-Former in BLIP-2。区别在于 CronusVLA 不是为了 compress visual token，而是为了提供一个 temporal context 的"接口"。

这个范式可以推广到很多场景：video understanding（把每帧单独过 VLM，feature level 做 temporal attention）、embodied navigation（multi-view observation aggregation）、甚至 multi-agent 系统（每个 agent 的 observation 独立编码，feature level 做 communication）。

### 8.2 Stop-Gradient 作为"模块化训练"的工具

Multi-frame regularization 的 stop-gradient 设计让我想起 EMANet、MoCo 中的 momentum encoder，或者 Decision Transformer 中对 state representation 的 frozen 处理。本质上是在说：**"这个模块负责 X，那个模块负责 Y，我们不让 X 的梯度污染 Y 的 learning dynamics"**。

在 VLA 这个场景下，这个分离特别合理：VLM backbone 的"职责"是 single-frame perception（识别物体、理解场景、grounding language），decoder 的"职责"是 temporal reasoning（从历史推断当前 phase、resolve ambiguity）。如果让 backbone 同时学两件事，会互相干扰；分开后各司其职，反而更高效。

### 8.3 Discrete vs Continuous Action Representation

从 single-frame pretraining 的 discrete action token 到 multi-frame post-training 的 continuous learnable feature，这是一个 representation 的"升级"。这让我想到 VQ-VAE 到 VAE 的演进，或者 ImageGPT 到 diffusion model 的转变。

Discrete token 的好处是可以复用 LLM 的 autoregressive paradigm，坏处是 inference 慢（要一步步 decode）。Continuous feature + diffusion decoder 的好处是 single forward pass（虽然 diffusion 内部要 iterative denoise，但可以用 few-step ODE solver 加速），坏处是失去了 LLM 的"语言"性质。

CronusVLA 的两阶段策略实际上是 **"先用 discrete paradigm 建立 embodied foundation，再切到 continuous paradigm 做 temporal refinement"**。这种"先学简单表示，再升级到复杂表示"的 curriculum 思想在很多地方都见过。

### 8.4 Robustness Benchmark 的生态位

SimplerEnv-OR 这个 benchmark 填补了一个空白。现有 VLA benchmark 都测"干净环境下的 performance"，但 real-world deployment 真正的挑战是 disturbance。这让我想到 ImageNet-C、ImageNet-P 这类 robustness benchmark 在 CV 领域的影响——它们推动了 robust training、augmentation strategy 的研究。

SimplerEnv-OR 可能会推动 VLA 领域对 robustness 的关注。特别是 Table 3 揭示的"performance ≠ robustness"现象（SpatialVLA 强在 SimplerEnv 但弱在 SimplerEnv-OR），说明我们需要重新思考 VLA 的 evaluation 标准。

### 8.5 Frame Number 的 Sweet Spot

Fig.5 显示了一个有意思的现象：CronusVLA-7B 在 7 帧时最好，0.5B 在 4 帧时最好。这让我想到 Transformer 中的 context length 问题——更多 context 不一定更好，因为冗余信息会 dilute 信号。

对于小模型（0.5B），它的 capacity 有限，处理更多 frame 会 overload；对于大模型（7B），它能 handle 更多 frame 中的 redundant 信息，并提取有用 temporal cue。这暗示了一个 scaling law：**optimal frame number 是 model capacity 的函数**。

### 8.6 与 π0、GR00T 的对比

π0 和 GR00T-N1 是 concurrent work，它们用 flow matching / diffusion 做 action generation，同时用 robot state 和 wrist view。CronusVLA 的优势在于：**它从 single-frame pretrained VLA 高效适配到 multi-frame，而不是从 scratch 训练**。这意味着可以用社区已有的 OpenVLA checkpoint 作为起点，节省大量 compute。

但 π0 在 LIBERO 上也达到 94.2%，说明 flow matching + state + wrist view 的组合很强。CronusVLA 的 contribution 是在"无 state、单 third-person view"的 setting 下达到 97.0%，证明 multi-frame modeling 能 compensate 这些缺失的 modality。

## 9. 总结：CronusVLA 的设计哲学

我读完这篇 paper 的整体感受是：它解决了一个 very specific 但 very important 的问题——**如何把 single-frame VLA 高效升级到 multi-frame，而不重新训练整个 model**。

核心 design choices 形成了一个 coherent 的故事：
1. **Feature-level temporal aggregation**：避免 VLM 内部 O(M²) attention
2. **Learnable feature as temporal interface**：复用 VLM 的 summarization 能力
3. **Cross-attention decoder**：action query multi-frame feature，线性复杂度
4. **Stop-gradient regularization**：保护 single-frame perception，加速 convergence
5. **Diffusion loss for continuous action**：避免 autoregressive decoding 的慢速

这些设计加在一起，让 CronusVLA 在 SimplerEnv 上 70.9%、LIBERO 上 97.0%、real-world robustness 72.6%，同时 8.7 Hz 推理（比 OpenVLA 单帧还快）。

**Reference Links：**
- 论文项目页：https://cronus-vla.github.io/
- OpenVLA（基础模型）：https://github.com/openvla/openvla
- SimplerEnv benchmark：https://simpler-env.github.io/
- LIBERO benchmark：https://lifelong-robot-learning.github.io/LIBERO/
- Open X-Embodiment：https://robotics-transformer-x.github.io/
- BridgeData V2：https://github.com/Tony-Zhao-Research/BridgeData_V2
- DiT（decoder 架构）：https://github.com/facebookresearch/DiT
- π0（对比方法）：https://www.physicalintelligence.company/blog/pi0
- GR00T N1：https://huggingface.co/nvidia/GR00T-N1-2B

这篇 paper 给我的最大启发是：**在 LLM/VLM 时代，efficient adaptation 比 from-scratch training 更重要**。CronusVLA 用 50K steps 的 post-training 就把 single-frame VLA 升级成 multi-frame，这种"增量式能力扩展"的思路，可能是未来 embodied AI 的主流范式。
