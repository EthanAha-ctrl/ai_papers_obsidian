---
source_pdf: FTP-1 A Generalist Foundation Tactile Policy Across.pdf
paper_sha256: 2e94db9bfdfce83534c9f606be9119aac0afe8845a16a52f1296b18599e263a4
processed_at: '2026-08-19T08:31:10-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 FTP-1

## 这 paper 在干嘛

一句话：**第一个 tactile foundation model**。

现在 robot community 的 vision policy 已经 scale 起来了——π0、π0.5、OpenVLA、GR00T 这些 work 跑通了 "大数据 + 大模型 + 统一接口" 这条路径。但 tactile 一直没起来。原因是 tactile sensor 的硬件形态太碎了：GelSight 是 224×224 RGB 图像、AetherGlove 是稀疏 array、ATI F/T sensor 就 6 个数字、palm sensor 和 fingertip sensor 的物理 principle 完全不一样。每个 lab 自己的 sensor 自己训自己的 policy，彼此不通用。

FTP-1 想做的是：pretrain 一个 tactile policy，之后你换 sensor、换 robot、换 embodiment，finetune 一下就能用，甚至换一个 pretraining 没见过的 sensor 都能 boost performance。

Project page: https://ftp1-policy.github.io/

---

## 核心问题：tactile 为什么不能直接 scale

robot learning community 一直有个 intuition：tactile 对 contact-rich task 很重要，插 USB、拧瓶盖、按按钮、在手里调整物体——这些 task 没 tactile 基本做不好。但为什么 prior work 都是 single-sensor？

因为 tactile signal across hardware **实在差太多**。GelSight 的 224×224×3 RGB image、AetherGlove 的稀疏 force array、ATI Axia80 的 6D F/T state——这三者的 raw signal space 几乎没有任何 overlap。你不能像 vision 那样直接把不同 camera 的 image 喂进同一个 ResNet 就期待它 work——pixel-level content 在不同 sensor 上语义完全不同。

所以 FTP-1 要解的 first-order problem 是：**怎么把不同 sensor 的信号 align 到同一个 representation space，让 shared model 学到的 knowledge 能跨 sensor transfer**。

---

## 核心 insight：MTTS

FTP-1 的 key abstraction 叫 **Morphology-Aware Tactile Token Space**。直觉是这样：

tactile sensor 的 physical layout 千差万别，但 **functional role 是收敛的**。任何 hand 上都有 "thumb tip 接触区"、"index fingertip 接触区"、"palm 接触区"——不管这个区域是 GelSight image 还是 array 还是单点 force sensor，它在 manipulation 中的 **功能语义** 是一样的。

MTTS 定义了 24 个 functional slot：
- slot 0–14：人手/灵巧手的 15 个 functional region
- slot 15–20：wrist/finger F/T sensor
- slot 21–23：reserved

parallel gripper 只有两个 finger，就直接 map 到 slot 0 (thumb) 和 slot 1 (index)——这步 mapping 让 gripper 和 dexterous hand 在 functional area 层面对齐了。

每个 token 进 tactile expert 之前都加一个 learnable functional-area embedding，这个 embedding 跨所有 sensor shared。所以 "thumb tip 这个区域" 在任何 sensor 上都对应同一个 embedding slot——这是 cross-sensor transfer 的语义基础。

Intuition: 这相当于做了一个 soft registration——不同 sensor 不需要 pixel-level 对齐，只需要在 "functional role" 这层对齐。shared tactile expert 看到的永远是 24 个统一形态的 token slot，跟 sensor 物理形态解耦。

---

## 架构怎么搭

三个 expert 并行：

1. **Vision-Language Expert**：直接用 PaliGemma [22](https://arxiv.org/abs/2407.07726)，处理 RGB + language
2. **Tactile Expert**：300M 的 Transformer，所有 sensor shared
3. **Action Expert**：π0.5 [5](https://arxiv.org/abs/2504.16054) 的 flow-matching head

Attention 拓扑是 **asymmetric**：action expert attends to tactile expert，但 tactile expert 不 attends back。直觉上 tactile 是 observation，应该被 action generation 条件化；反过来 tactile 不需要 condition on action。这样避免 tactile expert 被 action 的 noisy gradient 干扰。

**Tactile encoder 怎么处理不同 sensor**：

- Image-type（GelSight 这种）：sensor-specific 3 层 ViT + shared 9 层 T3 Transformer（用 T3 [18](https://arxiv.org/abs/2406.13640) pretrained weight 初始化），取 [CLS] token
- Array-type（H×W×D）：Fourier encode signal dim → 3 层 CNN → 2 层 MLP → 1 token per functional area
- State-type（D 维向量）：Fourier encode → 3 层 MLP → token

对 unseen sensor，只需要从头训 sensor-specific encoder（image-type 就 3 层 ViT），所有 shared 组件（T3 9 层 + tactile expert 300M + functional-area embedding）都能直接 reuse。

**为什么不用 adapter？** Tactile-VLA [9](https://arxiv.org/abs/2507.09160) 这种 prior work 把 tactile 通过 adapter 注入 VLM expert。FTP-1 给了三个理由反对：

1. adapter capacity 太小，学不到 generalizable tactile skill
2. adapter inject 到 VLM 内部会扰动 pretrained vision-language knowledge
3. 实验上 Tactile-VLA 在多个 task 上比 π0.5 还差——adapter-based fusion 反而 hurt performance

---

## Unified Action Space

robot embodiment 也碎：单臂、双臂、灵巧手、gripper、humanoid 各种 control signal 不同。FTP-1 用 UAS 把所有 action 表达成 fixed-length 稀疏向量：

$$\mathbf{a} = [\mathbf{a}^L, \mathbf{a}^R, \mathbf{a}^{ego}, \mathbf{a}^{sup}] \in \mathbb{R}^D$$

- $\mathbf{a}^L, \mathbf{a}^R$：左右臂，每臂包含 wrist translation (3D) + 6D rotation + 7 个 arm joint + 32 个 hand joint slot
- $\mathbf{a}^{ego}$：head pose (9D)
- $\mathbf{a}^{sup}$：locomotion / waist

hand joint 用 UniDex [13](https://arxiv.org/abs/2603.22264) 的 FAAS space，把不同 dexterous hand 的 actuator 按 functional role 对齐到 32 个 slot。"右食指弯曲" 这个动作在 Allegro、Shadow、Inspire、Leap 上都对应同一个 action index。parallel gripper 用独立 slot 28。

训练时没填的 slot 用 mask $\mathbf{M} \in \{0,1\}^D$ 屏蔽掉 loss。这样 FTP-1 能在一个 unified head 上同时训所有 embodiment。

**Intuition**: MTTS + UAS + FAAS 三层 abstraction 做的事一致：把 hardware heterogeneity 在 functional 层面对齐，让 backbone 学 functional skill 而不是 hardware mapping。这跟 Open X-Embodiment [37](https://arxiv.org/abs/2310.08896) 的 philosophy 一致，但推到了 tactile + dexterous hand 层面。

---

## 数据规模

- **26 个 data source**
- **21 个 sensor**（7 image + 5 array + 9 state）
- **~3000 小时**
- Resample 后比例：20% human + 30% dexterous hand + 50% gripper

主要 source：EgoTac (AetherGlove human data)、OmniSharingDB (PaxiniGlove)、Sharpa North-FTP-1 (paper 自己采的 4000 demo)、HumanoidEveryday [90](https://arxiv.org/abs/2510.08807)、TouchInTheWild [81](https://arxiv.org/abs/2512.13030)、RH20T [92](https://arxiv.org/abs/2307.00595)、RDP [65](https://arxiv.org/abs/2503.02881)、exUMI/Dexumi/ViTaMIn 这些 UMI-style dataset。

Training 用 48 张 H20，50k step，batch 768。50k step 后 saturate——作者归因于 data diversity 有限 + 跟 π0.5 knowledge preservation 的 tradeoff。

---

## 实验结果

### Seen sensor（5 个 setup，14 个 task）

UniVTAC 仿真（Table 1）：
- FTP-1: 66.66% avg
- 第二名：~49%
- **+17.5%**

Real-world（Table 2，Sharpa North + Sharpa&Dexmate）：
- FTP-1: 62.5%
- π0.5: 45.3%（第二）
- **+17.2%**

**Negative finding 重要**: Tactile-VLA 比 π0.5 还差（35.8% vs 45.3%）。naive tactile integration 反而 hurt performance。这个 finding 跟 [14](https://arxiv.org/abs/2604.13015) Touch Dreaming 的 observation 一致——improper tactile integration 扰乱 vision-language perception。这反过来说明 tactile foundation policy 不能只是 "vision policy + tactile head"，需要 native tactile modeling。

### Unseen sensor（核心 contribution）

两个 pretraining 没见过的 sensor：
- FlexivXense（Xense image sensor）
- TactileUMI（Contactile array sensor）

结果（Table 3）：
- FTP-1: 46.6%
- FTP-π0.5（同架构但无 tactile pretraining）: 15.0%
- **+31.6%**

行为上：FTP-1 在 Insert Hanoi 上有 reactive insertion control——hanoi piece misaligned 时会根据 tactile 减速；π0.5 不会，常常硬塞导致 fail。Wipe Board 上 FTP-1 能 maintain stable pressing force，其他 model lose contact。

### Ablation: gain 来自哪里？

两个 hypothesis：
- H1: data distribution 接近 downstream
- H2: transferable tactile knowledge

训练 NTP-1：用 FTP-1 的 data 和 optimization，但 pretraining 时去掉 tactile。Finetune 时再加回 tactile 架构。

结果：UniVTAC 上 NTP-1 (50%) 略好过 FTP-π0.5 (45%)，说明部分 gain 来自 data distribution。但 FTP-1 (66.66%) 显著好过 NTP-1，**FlexivXense 上 FTP-1 比 NTP-1 高 +37.5%**——这是 stronger evidence，说明对 unseen sensor，tactile pretraining 是 essential 的。

**H2 成立**。

---

## 我的 critical take

### 为什么 work

三层 abstraction 的解耦是关键：MTTS 解耦 sensor layout 和 functional role；heterogeneous encoder 解耦 sensor-specific processing 和 shared representation；UAS+FAAS 解耦 embodiment 和 action semantics。shared tactile expert 学到的是 "functional area space 上 tactile 该如何 inform action"——这是 sensor-agnostic skill。

### 真 surprising 的 finding

不是 unseen sensor 能 transfer——sensor 物理相似 + functional area 对齐，transfer 合理。

最 surprising 的是 **naive tactile integration 比 no tactile 还差**。这说明 tactile signal 如果 fuse 不好会主动 hurt policy，tactile foundation policy 需要专门的架构设计，不是简单的 modality addition。

### 局限

1. **Tactile/force servoing 没解决**——FTP-1 学的是 high-level tactile perception，不是 low-level closed-loop force control。Force-based servoing 是 future direction [34-36](https://arxiv.org/abs/2603.05687)。
2. **Unseen sensor 只有 2 个 setup**——sample size 太小。Xense 物理像 GelSight，Contactile 物理像 AetherGlove，transfer 难度其实不高。如果遇到物理 principle 完全不同的 sensor（capacitive vs piezoelectric vs magnetic），transfer 效果未知。
3. **MTTS 是 hand-designed**——24 个 slot 是 human prior。如果未来有 full-arm tactile skin 或 whole-body tactile，这个 partition 要重新设计。Learnable functional partition 可能更好。
4. **Tactile expert 只有 300M**——跟 PaliGemma 3B 比是 minor module。Paper 说 MoE fusion 没 consistent gain，但我怀疑是 data 不够，不是 architecture 不行。
5. **50k step saturate**——暗示 tactile data effective diversity 比 vision 低很多。tactile signal 的 information content 远低于 vision（image 是 224×224×3 ≈ 150K dim，F/T state 就 6 dim），相同 data hour 下 information throughput 小，需要更多 task/sensor diversity 而非更多 hour。
6. **Compute scale 跟 vision foundation model 比小太多**——48 张 H20 50k step 远不到 vision foundation model 的 scale。

### 跟 LLM analogy

FTP-1 现在的状态像 BERT 早期——pretrain task 是 conditional action generation（flow matching），data 是 heterogeneous multi-sensor corpus，scale 还在 "刚证明 pretrain useful" 阶段。下一步应该是：more data scaling、co-training with vision foundation data、online RL finetune、tactile world model（参考 OmniVTA [8](https://arxiv.org/abs/2603.19201)）。

### 跟 Open X-Embodiment 的关系

philosophy 上一脉相承——Open X-Embodiment [37](https://arxiv.org/abs/2310.08896) 想做 cross-embodiment vision generalization，FTP-1 把这个思路推到 tactile modality + dexterous hand。这个方向我认为是对的，tactile sensing 和 dexterous manipulation 是 robot learning 里 contact-rich task 的 long pole，必须 foundation 化才能 scale。

---

## 一句话总结

FTP-1 证明了一件事：**tactile manipulation 可以做 foundation policy，pretrain 之后能跨 sensor transfer，甚至 transfer 到没见过的 sensor**。代价是需要 native tactile modeling（不能 adapter）+ functional area abstraction（MTTS）+ large heterogeneous pretraining data。作为一个 "first foundation baseline"，design choice 和实验都 solid，能成为后续 tactile scaling 的 anchor。

---

# FTP-1: 第一个 Generalist Foundation Tactile Policy 深度讲解

## 1. Big Picture: 为什么 tactile foundation policy 这么难

机器人 community 在 vision-based generalist policy 上已经 scaling 起来了——π0 [1](https://arxiv.org/abs/2410.24164)、π0.5 [5](https://arxiv.org/abs/2504.16054)、GR00T N1 [2](https://arxiv.org/abs/2503.14734)、RDT-1B [29](https://arxiv.org/abs/2410.07764)、OpenVLA [11](https://arxiv.org/abs/2406.09246)、Gemini Robotics [41](https://arxiv.org/abs/2503.20020)——这些 work 把 "scale data + scale model + unified action interface" 这条路径跑通了。但 tactile 一直没起来，根因 paper 第一段就点破了：**tactile signals across hardware 是高度 heterogeneous 的**——modality 不一样（image-based GelSight vs array-based Contactile vs 6-axis F/T state）、resolution 不一样（GelSight 是 224×224 RGB，AetherGlove 是稀疏 array）、morphology 不一样（指尖单个 sensor vs 全手 distributed）、contact response 不一样（elastic deformation vs normal force）。

所以 prior tactile policy 工作 [6-8] 都 stuck 在 single sensor 上，pretraining 的 transfer 也只在 in-domain sensor 内做。FTP-1 想回答的问题是：**能不能有一个 tactile foundation policy，pretrain 完之后，下游 finetune 时换一个完全没见过的 tactile sensor，依然能 boost performance？** Paper 给的答案是 yes，而且 +31% 的 success rate gain——这个 number 在 tactile manipulation 里相当 striking。

Project page: https://ftp1-policy.github.io/

---

## 2. MTTS: Morphology-Aware Tactile Token Space——整个 work 的核心 intuition

### 2.1 关键 insight

tactile sensor 的物理形态千差万别，但 **functional role 是收敛的**——任何 hand 上都存在 "thumb tip 接触区"、"index fingertip 接触区"、"palm 接触区"。FTP-1 的核心 abstraction 就是把 tactile information 从 raw signal space 提升到 functional area space。

MTTS 定义了 **24 个 functional-area slots**：
- **slots 0–14**：15 个 in-hand functional regions（对应人手/灵巧手的不同关节区）
- **slots 15–20**：6 个 wrist/finger force-torque slot（专门给 F/T sensor）
- **slots 21–23**：reserved for future use

对于 parallel gripper，由于只有两个 finger，paper 把 gripper 两侧 sensor 分别 map 到 slot 0（thumb-tip slot）和 slot 1（index-fingertip slot）——这个映射非常重要，因为它让 parallel gripper 和 dexterous hand 在 functional area 这层对齐了，shared tactile expert 学到的 "thumb tip 接触" 知识可以跨 embodiment 迁移。

### 2.2 Functional-area embedding

每个 token 进 tactile expert 之前都会加上一个 learnable functional-area embedding，这个 embedding 在所有 sensor 之间 **shared**。这意味着 "thumb tip" 这个 functional 位置，无论用 GelSight 还是 AetherGlove 测量，都对应同一个 embedding slot——这是 cross-sensor transfer 能发生的语义层基础。

左右手用独立的 functional-area embedding，因为 left thumb 和 right thumb 在 robot coordinate frame 里语义不对称。

**Intuition**: 这相当于在 tactile representation learning 里做了一个 "soft registration"——不同 sensor 不再需要 pixel-level 对齐，只需要在 functional area 这层对齐。这个 abstraction 让 shared tactile expert 看到的是统一形态的 24 个 token slot，跟 sensor 物理形态解耦了。

---

## 3. Heterogeneous Tactile Encoders——技术细节

MTTS 提供了统一 token interface，但 input shape 依然千差万别。Paper 给三种 encoder：

### 3.1 Image-type (e.g., GelSight-Mini [16](https://www.mdpi.com/1424-8220/17/12/2762), Sharpa DTC [10](https://arxiv.org/abs/2506.15953))

Pipeline: tactile image → resize 224×224 → **sensor-specific ViT encoder** (depth=3, width=768, head=12) → **shared T3 Transformer module** (depth=9, width=768, head=12, initialized from pretrained T3 weights [18](https://arxiv.org/abs/2406.13640)) → take [CLS] token。

这里关键设计是 **sensor-specific ViT + shared T3** 的两段式：sensor-specific 部分只学 3 层，专门处理 "如何把这个 sensor 的 raw image 变成 generic tactile embedding"；shared 部分深度 9 层，跨所有 image-type sensor 共享，学到的是 generic tactile dynamics。这个设计让 unseen image sensor 在 finetune 时只需要从头训 3 层 sensor-specific ViT，剩下 9 层 + tactile expert 都能直接 reuse。

### 3.2 Array-type (e.g., Contactile [19](https://arxiv.org/abs/2502.12191), AetherGlove, 3DViTac)

Input shape: (H, W, D)，其中 H、W 是 array spatial resolution，D 是每个 unit 的 signal dimension。Pipeline: **Fourier encoding** on signal dimension → concat with original → **3-layer CNN** → **2-layer ReLU MLP** → 1 token per functional area。

Fourier encoding 这里是关键 [21](https://arxiv.org/abs/2510.14647)：把 raw scalar force value 通过 sin/cos basis expansion 到高维，让 MLP 更容易拟合 high-frequency contact dynamics。这跟 NeRF 里的 positional encoding 是同一思路。

### 3.3 State-type (e.g., ATI Axia80 F/T, FrankaTorque, PaxiniGlove)

Input shape: (D,)。Pipeline: **Fourier encoding** → concat → **3-layer ReLU MLP** → token。

### 3.4 共享策略

如果同一个 sensor 有多个 functional area 的 input shape 相同，那这些 area 共享同一个 encoder——这个 design 既减少 sensor-specific parameter count，又 force 模型对相同 physical layout 的 functional area 用一致的 representation。

### 3.5 Post-processing

所有 token 都过 LayerNorm [85](https://arxiv.org/abs/1607.06450) → 加 functional-area embedding → **2-layer GELU MLP** project 到 tactile expert 的 input dim。

---

## 4. Shared Tactile Expert——modality fusion 架构

### 4.1 整体 multi-expert 架构

FTP-1 建立在 π0.5 [5](https://arxiv.org/abs/2504.16054) 上，三个 expert：

1. **Vision-Language Expert** (from PaliGemma [22](https://arxiv.org/abs/2407.07726))：处理 RGB observation + language instruction
2. **Tactile Expert**：300M parameter Transformer，width=1024, depth=18, MLP dim=4096, 8 attention heads, head dim=256
3. **Action Expert**：flow-matching Transformer，从 π0.5 初始化

### 4.2 Attention 拓扑

关键 design choice：**Action expert attends to tactile expert，但 tactile expert 不 attends back**。

这是一个 asymmetric attention 拓扑，intuition 是：tactile 是 observation modality，应该被 action generation 条件化；反过来 tactile 不需要 condition on action。这避免了 tactile expert 被 action 的 noisy gradient 干扰，让它专注于学 reusable tactile representation。

### 4.3 为什么不用 adapter（vs Tactile-VLA [9](https://arxiv.org/abs/2507.09160)）

Prior work [6, 9, 23, 24] 把 tactile 通过 lightweight adapter 注入 VLM expert。Paper 给了三个理由反对这种 design：

1. **Reuse capacity**: adapter 太小，capacity 不足以学到 generalizable tactile manipulation skill；独立 expert 才能在 unseen sensor finetune 时 reuse 整个 300M tactile module。
2. **避免干扰 VLM**: adapter 直接 inject 到 VLM expert 内部，会扰动 pretrained vision-language knowledge，可能让 VLM 掉点。
3. **Efficiency & performance**: 论文实验（Table 1, 2, 3）显示 Tactile-VLA 在多个 task 上甚至比 π0.5 还差——例如 Sharpa North 的 Twist Cap，Tactile-VLA 只有 10%，π0.5 有 40%，FTP-1 有 65%。这说明 adapter-based tactile injection 在 long-horizon contact-rich task 上反而 hurt performance。

### 4.4 Adaptive RMSNorm Proprioception Injector

proprioception 不作为独立 token，而是通过 **adaptive RMSNorm** [5, 86](https://arxiv.org/abs/2212.09748) 注入 attention block。Pipeline: proprioceptive state → Fourier encoding → 3-layer ReLU MLP → LayerNorm → concat with flow-matching timestep features → adaptive RMSNorm modulation。

这个 design 跟 DiT [86](https://arxiv.org/abs/2212.09748) 用 timestep modulation attention 是同源的——把 proprioception 当成 conditioning signal，让它在每个 attention block 的 scale 和 shift 上起作用。Paper 说这比 "independent proprioceptive token" 在 generalization 和 robustness 上更好。

---

## 5. Unified Action Space (UAS)

### 5.1 公式

Action chunk prediction：

$$\hat{\mathbf{A}}_{t:t+H-1} = \pi_\theta(\ell, \mathcal{T}_t, \mathbf{s}_t, \boldsymbol{\mathcal{X}}_t) \in \mathbb{R}^{H \times D}$$

变量解释：
- $\hat{\mathbf{A}}_{t:t+H-1}$：从 timestep $t$ 开始未来 $H$ 步的 action chunk
- $H$：action horizon
- $D$：Unified Action Space 维度
- $\ell$：language instruction
- $\mathcal{T}_t$：(multi-view) RGB observations
- $\mathbf{s}_t$：proprioception
- $\boldsymbol{\mathcal{X}}_t$：tactile observations

每个 single-step action：

$$\mathbf{a} = [\mathbf{a}^L, \mathbf{a}^R, \mathbf{a}^{ego}, \mathbf{a}^{sup}] \in \mathbb{R}^D$$

- $\mathbf{a}^L, \mathbf{a}^R$：左右臂控制信号
- $\mathbf{a}^{ego}$：head-pose 控制信号
- $\mathbf{a}^{sup}$：locomotion / waist 等额外 control slots

每个 arm $b \in \{L, R\}$：

$$\mathbf{a}^b = [\mathbf{t}_w^b, \mathbf{r}_w^b, \mathbf{q}_{\text{arm}}^b, \mathbf{q}_{\text{hand}}^b]$$

- $\mathbf{t}_w^b \in \mathbb{R}^3$：wrist translation（xyz）
- $\mathbf{r}_w^b \in \mathbb{R}^6$：6D wrist rotation representation（来自 [84](https://arxiv.org/abs/2402.10329) UMI 的 6D rotation）
- $\mathbf{q}_{\text{arm}}^b \in \mathbb{R}^7$：7-DOF arm joints
- $\mathbf{q}_{\text{hand}}^b \in \mathbb{R}^{32}$：32 维 canonical hand joint slot（FAAS space [13](https://arxiv.org/abs/2603.22264)）

### 5.2 FAAS (Function-Actuator-Aligned Space)

hand joint action $\mathbf{q}_{\text{hand}}^b \in \mathbb{R}^{32}$ 来自 UniDex [13]。FAAS 的核心是把不同 dexterous hand（Ability, Allegro, Inspire, Leap, Shadow, Xhand 等）的 actuator 按 functional role 对齐到 32 个 slot。这意味着 "右食指弯曲" 这个动作，在所有 hand 上都对应同一个 action slot index。Figure 8 给了 8 种 hand 的 FAAS mapping visualization。

对 **parallel gripper**，用独立 slot 28。

### 5.3 Masked loss

不同 embodiment 只填它支持的 control slot，其他位置用 mask：

$$\mathbf{M} \in \{0, 1\}^D$$

训练时 mask 掉的 dimension 不参与 loss。这让 FTP-1 能在一个 unified prediction head 上同时训练单臂、双臂、灵巧手、gripper、humanoid 等各种 embodiment，互不干扰。

**Intuition**: UAS + FAAS + MTTS 三层 abstraction 的目的一致——把 heterogeneous embodiment/sensor 在 functional 层面对齐，让 shared backbone 学到的是 functional skill 而不是 hardware-specific mapping。这跟 Open X-Embodiment [37](https://arxiv.org/abs/2310.08896) 想做的事在 philosophy 上一致，但 FTP-1 把它推到了 tactile 和 dexterous hand 层面。

---

## 6. FTP-1-Dataset——3000 小时的 heterogeneous tactile pretraining corpus

### 6.1 规模和组成

- **26 个 data source**
- **21 个 tactile sensor**：7 image-type + 5 array-type + 9 state-type
- **~3000 hours** 总量
- Resample 后比例：**20% human hand + 30% dexterous hand + 50% gripper/UMI**

Human 部分小（20%）是因为 dex hand 和 gripper 数据更 abundant；resample scale 在 Table 4 里给了——例如 OpenTouch sample scale 11.09、EgoTac-Hotel 2.82、Unit 132.83——这个 sample scale 应该是 inverse sampling weight 的某种形式，让小数据 source 在 batch 里出现得更频繁。

### 6.2 关键 source

- **EgoTac series** (AetherGlove array): human egocentric data
- **OmniSharingDB** (PaxiniGlove state): human data from PaXini
- **Sharpa North-FTP-1**: paper 自己采集的 4000 long-horizon dexterous demonstration，用 Sharpa DTC image sensor [10](https://arxiv.org/abs/2506.15953)
- **HumanoidEveryday** [90](https://arxiv.org/abs/2510.08807): UnitreeDex3 state sensor
- **TouchInTheWild** [81](https://arxiv.org/abs/2512.13030): 3DViTac array sensor
- **RH20T** [92](https://arxiv.org/abs/2307.00595): 3 种 config，覆盖 ATI force/torque + uSkin array + FrankaTorque
- **RDP** [65](https://arxiv.org/abs/2503.02881): GelSight-Mini + MCTac + FlexivGripperTorque 多模态
- **exUMI** [72](https://arxiv.org/abs/2509.14688), **Dexumi** [91](https://arxiv.org/abs/2505.21864), **ViTaMIn** [70](https://arxiv.org/abs/2504.06156): UMI-style tactile data

### 6.3 数据预处理

- Wrist/head pose 统一坐标方向
- Language instruction 用 **GPT-4o [95](https://arxiv.org/abs/2410.21276) rewrite** 增加 linguistic diversity
- 每个 dataset 独立计算 normalization statistics
- Heterogeneous encoder 按 sensor 而非 data source 组织——同一 sensor 在不同 source 里共享 encoder

---

## 7. Training Infrastructure——large-scale heterogeneous pretraining 的工程

### 7.1 Pretraining 设置

- **48 张 NVIDIA H20 GPU**
- **50k steps**
- **Global batch size 768**
- **LR: 1e-4 → 5e-5**（decay）
- 50k 后 saturate——作者归因于 tactile data diversity 有限 + 跟 π0.5 knowledge 的 preservation tradeoff

### 7.2 Heterogeneous training infra

Paper 提到自研了一个新 infra：**自动把不同 domain 的 data dispatch 到不同 GPU**，保证同一 GPU batch 内的 sample 同 data format。这让大规模 heterogeneous pretraining 能 efficient parallel 起来。

Gradient 处理：
- **Domain-specific module**: 独立 update
- **Shared module**: gradient merge 后 joint update

这种 design 在 HPT [15](https://arxiv.org/abs/2409.20537) 类工作里也有类似 spirit——把 stem shared，head/encoder per-domain。

### 7.3 Optimizer 选择

Paper 试过 **Muon [88](https://arxiv.org/abs/2502.16982) optimizer**，发现它在 convergence speed 和 offline action MSE 上更好，但 **real-robot rollout 的 generalization 和 robustness 下降**，最终用 AdamW [87](https://arxiv.org/abs/1711.05101)。这个 observation 很有意思——Muon 的 orthogonalization 让 weight update 更 isotropic，可能 overfit 到训练 distribution，反而 hurt robustness。这是一个值得深挖的 finding。

### 7.4 Normalization

Action / state 用 **z-score normalization** [1]，paper 说比 quantile-based normalization [5] 在 contact-rich 的 fine-grained action 上效果更好。Intuition 上 z-score 保留了 action 的 fine structure，quantile-based 把 outlier 压得太狠，contact-rich task 恰恰需要 small force variation 的 sensitivity。

### 7.5 Finetuning

- 8 张 A800 GPU
- 20k steps per dataset
- Batch size 64
- LR: 5e-5 → 5e-6

---

## 8. Seen Sensor 实验——5 个 setup，14 个 task

### 8.1 Setup 概览（Table 5）

| Setup | Domain | Robot | End-effector | Tactile Sensor | Type |
|---|---|---|---|---|---|
| UniVTAC | Sim (unseen), Gripper | Franka | Franka | GelSight-Mini | image |
| Sharpa North | Real, DexHand | Sharpa North | Sharpa Wave | Sharpa DTC | image |
| Sharpa&Dexmate | Real, DexHand | Dexmate (unseen) | Sharpa Wave | Sharpa DTC | image |
| FlexivXense | Real, Gripper | Flexiv | Robotiq | Xense (unseen) | image |
| TactileUMI | Real, Gripper | Franka | 3D-printed (unseen) | Contactile (unseen) | array |

### 8.2 Baselines

- **π0.5** [5](https://arxiv.org/abs/2504.16054)：SOTA open-source VLA，无 tactile——评估 tactile 相对于 strong VLA baseline 的增量
- **Tactile-VLA** [9](https://arxiv.org/abs/2507.09160)：tactile 注入 VLM expert（adapter-based），评估 tactile expert 设计
- **FTP-π0.5**：FTP-1 architecture 但用 π0.5 weight 初始化、不做 FTP-1 pretraining——隔离 large-scale tactile pretraining 的贡献

### 8.3 UniVTAC 仿真结果（Table 1）

| Method | Lift Bottle | Pull-out Key | Lift Can | Put Bottle | Insert Hole | Insert Tube | Avg. | Avg. w/o Lift |
|---|---|---|---|---|---|---|---|---|
| VITaL* [28] | 72 | 47 | 8 | 32 | 25 | 34 | 36.33 | 34.5 |
| UniVTAC-ACT* [27] | 71 | 46 | 29 | 31 | 25 | 56 | 43.00 | 39.5 |
| π0.5 | 97 | 38 | 72 | 16 | 31 | 41 | 49.16 | 31.5 |
| Tactile-VLA | 97 | 32 | 15 | 10 | 41 | 56 | 41.83 | 34.75 |
| FTP-π0.5 | 77 | 30 | 26 | 19 | 47 | 72 | 45.16 | 42 |
| **FTP-1** | **97** | **48** | 65 | **47** | **64** | **79** | **66.66** | **59.5** |

关键 observation：
1. **Lift Bottle / Lift Can 可以 largely 不用 tactile 解掉**（π0.5 97% / 72%）——这两个 task 在仿真里 vision 信息足够。所以 paper 额外报了 "Avg. w/o Lift"。
2. **FTP-1 在所有 contact-critical task 上都最优**：Pull-out Key 48% vs 第二 47%，Put Bottle 47% vs 第二 32%，Insert Hole 64% vs 第二 47%，Insert Tube 79% vs 第二 72%。
3. **Tactile-VLA 反而比 π0.5 差**——adapter-based tactile injection 在 contact-rich task 上 hurt performance，验证了 paper 关于 independent tactile expert 的 design choice。

### 8.4 Real-World Seen Sensor 结果（Table 2）

| Method | Draw Balloon | Fix Hand (Tear) | Fix Hand (Finish) | Twist Cap | Flip Book | Wipe Dish | Avg. |
|---|---|---|---|---|---|---|---|
| π0.5 | 35 | 70 | 35 | 40 | 65 | 30 | 45.3 |
| Tactile-VLA | 20 | 80 | 25 | 10 | 45 | 35 | 35.8 |
| FTP-π0.5 | 25 | 65 | 25 | 20 | 70 | 45 | 41.6 |
| **FTP-1** | **45** | **80** | **40** | **65** | **85** | **60** | **62.5** |

**Surprising finding**: π0.5 (45.3%) 排第二，比两个 tactile baseline 都好。这说明 naive tactile integration 会 hurt——tactile 信号没有 proper fusion architecture，反而干扰了 vision-language perception。FTP-1 的 +17.2% gain 主要来自 proper tactile expert fusion + large-scale pretraining。

行为分析：
- **Tactile-VLA / FTP-π0.5**：contact condition 改变时 action 不稳定，limited robustness
- **π0.5**：在 Sharpa&Dexmate 的 pressing task 上没法 maintain consistent pressing force，会 push against bottle cap without reactive force adjustment
- **FTP-1**：action 更 stable 和 smooth，能 reactive 到 tactile feedback

---

## 9. Unseen Sensor 实验——核心 contribution

### 9.1 Setup

两个全新 sensor：
- **FlexivXense**：Xense image tactile sensor（paper App. E.4 说最像 GelSight-Mini）
- **TactileUMI**：Contactile array sensor（最像 AetherGlove）

Task：
- Insert Hanoi（细圆 hanoi 插 pillar）—— 100 demo
- Insert USB —— 100 demo
- Wipe Board —— 50 demo

### 9.2 结果（Table 3）

| Method | Insert Hanoi | Insert USB | Wipe Board | Avg. |
|---|---|---|---|---|
| π0.5 | 25 | 0 | 20 | 15.0 |
| Tactile-VLA | 0 | 10 | 15 | 8.3 |
| FTP-π0.5 | 5 | 10 | 30 | 15.0 |
| **FTP-1** | **55** | **30** | **55** | **46.6** |

**+31.6% absolute gain over FTP-π0.5**——这是 paper 的 headline number。

### 9.3 行为细节

- **Insert Hanoi**：FTP-1 和 π0.5 都能 exhibit recovery behavior，但 FTP-1 有 **reactive insertion control**——当 hanoi piece misaligned 时，FTP-1 根据 tactile feedback 减速 insertion motion；π0.5 没这个能力，常常 fail。
- **Insert USB**：100 demo 下 data efficiency 是关键。FTP-1 action 更 smooth；其他模型在 insertion 过程中有 small shaking，降低 success rate。
- **Wipe Board**：FTP-1 能 maintain stable pressing force；其他模型 lose tight contact with board。

### 9.4 Transfer 机制

对于 unseen sensor：
- **Sensor-specific encoder 从头训**（Xense 的 sensor-specific ViT-3 层、Contactile 的 CNN-MLP）
- **Reuse pretrained 的 shared component**：
  - Tactile expert（300M Transformer）
  - T3 shared module（9 层 image-type encoder）
  - Functional-area embeddings

这意味着即使你换一个完全新的 sensor，pretraining 学到的 "如何从 tactile representation 做 action prediction" 的 knowledge 依然可以 reuse。Sensor-specific encoder 只需要学 "如何把这个 sensor 的 raw 信号映射到 MTTS"，而 MTTS 之上的所有 reasoning skill 都可以继承。

---

## 10. Ablation：Gain 是来自 pretraining knowledge 还是 data distribution？

这是 paper 的 4.2 节，做得很 critical。

### 10.1 两个 hypothesis

- **H1 (Data Distribution)**：gain 是因为 FTP-1-Dataset 跟 downstream task distribution 更近
- **H2 (Transferable Knowledge)**：gain 是因为 FTP-1 学到了 transferable tactile manipulation knowledge

### 10.2 NTP-1 对照实验

训练一个 **NTP (No-Tactile-Pretraining) checkpoint**：跟 FTP-1 用同样的 data、同样的 optimization，但 **去掉 tactile input 和 tactile architecture**。Finetune 时再加上 tactile 架构，叫 **NTP-1**。

如果 H1 成立，NTP-1 应该跟 FTP-1 接近（因为 data distribution 一样）。
如果 H2 成立，FTP-1 应该显著好过 NTP-1。

### 10.3 结果（Table 6）

| Method | Lift Bottle | Pull-out Key | Lift Can | Put Bottle | Insert Hole | Insert Tube | Avg. | Avg. w/o Lift |
|---|---|---|---|---|---|---|---|---|
| FTP-π0.5 | 77 | 30 | 26 | 19 | 47 | 72 | 45.16 | 42 |
| NTP-1 | 88 | 38 | 66 | 32 | 31 | 45 | 50.00 | 36.5 |
| **FTP-1** | **97** | **48** | 65 | **47** | **64** | **79** | **66.66** | **59.5** |

UniVTAC 上 NTP-1 比 FTP-π0.5 略好（50% vs 45.16% avg），说明部分 gain 来自 data distribution 接近。但 NTP-1 远不如 FTP-1，说明 tactile pretraining 本身贡献明显。

FlexivXense 上（Figure 7）FTP-1 **比 NTP-1 高 +37.5%**——这是 stronger evidence，说明对 unseen sensor，tactile pretraining 是 essential 的。NTP-1 在 key insertion stage 产生 unstable action，robustness 差很多。

结论：**H2 (Transferable Knowledge) 成立**——FTP-1 的 tactile branch 学到了 general tactile manipulation knowledge，可以 transfer 到 downstream contact-rich task，甚至 unseen sensor。

---

## 11. 跟相关 work 的关系

### 11.1 Generalist Policy 谱系

π0 [1](https://arxiv.org/abs/2410.24164) → π0.5 [5](https://arxiv.org/abs/2504.16054) → GR00T N1 [2](https://arxiv.org/abs/2503.14734) → Gemini Robotics [41](https://arxiv.org/abs/2503.20020) → RDT-1B [29](https://arxiv.org/abs/2410.07764) → OpenVLA [11](https://arxiv.org/abs/2406.09246)。FTP-1 是把这条路径推到 tactile modality 的第一个 work。

### 11.2 Tactile Representation Pretraining 谱系

- **Sparsh** [31](https://arxiv.org/abs/2410.24090)：vision-tactile contrastive pretraining
- **AnyTouch** [32](https://arxiv.org/abs/2502.12191)：cross-sensor unified representation
- **T3** [18](https://arxiv.org/abs/2406.13640)：transferable tactile transformer（FTP-1 用了 T3 weight 做 image encoder 的 shared module 初始化）
- **ViTaL** [28](https://arxiv.org/abs/25863877)：visuo-tactile pretraining

这些 work 学的是 representation，不是 end-to-end policy。FTP-1 是直接 pretrained policy，跨 sensor + 跨 embodiment。

### 11.3 VTLA (Vision-Tactile-Language-Action)

Tactile-VLA [9](https://arxiv.org/abs/2507.09160)、VLA-Touch [74](https://arxiv.org/abs/2507.17294)、TacVLA [75](https://arxiv.org/abs/2603.12665)、ForceVLA [76](https://arxiv.org/abs/2603.15169)、OmniVTLA [6](https://arxiv.org/abs/2508.08706)、VTACFormer [10](https://arxiv.org/abs/2506.15953)——这些 work 都是 sensor-specific 的 VLA 扩展，没解决 cross-sensor generalization。FTP-1 是 sensor-agnostic 的 generalist foundation。

### 11.4 HPT 谱系

Heterogeneous Pre-trained Transformer (HPT) [15](https://arxiv.org/abs/2409.20537) 在 proprioceptive-visual modality 上做了 heterogeneous stem + shared trunk 的设计。FTP-1 在 tactile 上做了类似的事，但加了 MTTS 这层 functional area abstraction，这是 HPT 没有的。

---

## 12. 我的 Intuition 和 critical 观察

### 12.1 为什么这个 work 能成

核心在于 **三层 abstraction 的解耦**：
1. **MTTS**：解耦 sensor physical layout 和 functional role
2. **Heterogeneous encoder**：解耦 sensor-specific signal processing 和 shared tactile representation
3. **UAS + FAAS**：解耦 embodiment hardware 和 action semantics

这三层加起来，让 shared tactile expert 学到的是 "在 functional area space 上，tactile signal 应该如何 inform action"——这是一种 **sensor-agnostic tactile reasoning skill**。

### 12.2 真正 surprising 的 finding

对我来说最 surprising 的不是 unseen sensor 能 transfer——毕竟 sensor 物理上相似、functional area 对齐了，transfer 是 reasonable 的。最 surprising 的是：

**Naive tactile integration 比 no tactile 还差**——Table 2 里 Tactile-VLA (35.8%) < π0.5 (45.3%)，Table 3 里 Tactile-VLA (8.3%) << π0.5 (15.0%)。这说明 tactile 信号如果 fuse 不好，会主动 hurt policy。这个 finding 跟 [14](https://arxiv.org/abs/2604.13015) "Touch Dreaming" 的 observation 一致——improper tactile integration 会扰乱 vision-language perception。

这反过来说明 tactile foundation policy 不仅仅是 "vision foundation policy + tactile head"——它需要 native tactile modeling，FTP-1 的独立 tactile expert + shared pretraining 是必要的。

### 12.3 局限和 open question

Paper 自己提了：
1. **没解决 tactile/force-based servoing**——FTP-1 学的是 high-level tactile perception 和 manipulation skill，不是 low-level force control。Force-based closed-loop control 是 future direction [34-36](https://arxiv.org/abs/2603.05687)。
2. **Dataset scale 仍有限**——3000 hour 跟 Open X-Embodiment [37](https://arxiv.org/abs/2310.08896) 的体量比小很多；50k step 就 saturate 也说明 data diversity 不够。
3. **Pretraining compute 跟 π0.5 这种 vision foundation model 比小太多**——48 张 H20 50k step，远不到 vision foundation model 的 scale。

我会补充几个 critical observation：

4. **Unseen sensor 只有 2 个 setup**——虽然 +31% gain 很 striking，但 sample size 太小。Xense 是 image-type 跟 GelSight-Mini 物理相似；Contactile 是 array-type 跟 AetherGlove 物理相似。如果 unseen sensor 的物理 principle 完全不同（比如 capacitive vs piezoelectric vs magnetic），transfer 效果未知。

5. **MTTS 是 hard-coded 的 functional area partition**——24 个 slot、slot 0 是 thumb tip 这种 mapping 是 hand-designed。如果未来有 full-arm tactile skin 或者 whole-body tactile，这个 partition 需要重新设计。Learnable functional area partition 可能是更好的方向。

6. **Tactile expert 300M param 偏小**——跟 vision-language expert (PaliGemma 3B) 和 action expert 比是 minor module。如果真要 scale，tactile expert 应该跟其他 expert 同量级。Paper 说 "更复杂的 MoE fusion 没 consistent gain"，但我怀疑是 data 不够，不是 architecture 不行。

7. **Pretraining saturate at 50k** 这个观察其实挺重要的——它暗示 tactile data 的 effective diversity 比 vision data 低很多。一个可能的解释：tactile signal 的 information content 远低于 vision（image 是 224×224×3 ≈ 150K dim，tactile state 可能就 6 dim），相同 data hour 下 tactile 的 information throughput小，需要更多 task/sensor diversity 而非更多 hour。

### 12.4 跟 LLM 预训练的类比

如果做个 analogy：FTP-1 现在的状态类似 BERT 早期——pretrain task 是 masked action prediction（flow matching 的 conditional generation），data 是 heterogeneous multi-sensor corpus，scale 还在 "刚证明 pretrain useful" 阶段。下一步应该是：
- More data scaling（向 Open X-Embodiment [37](https://arxiv.org/abs/2310.08896) / DROID [43](https://arxiv.org/abs/2403.12945) / Dexumi [91](https://arxiv.org/abs/2505.21864) 等更大规模 dataset 扩展）
- Co-training with vision foundation data（paper 提到了这个 future direction）
- Online RL finetune（参考 [34-36](https://arxiv.org/abs/2603.05687)）
- Tactile world model（参考 OmniVTA [8](https://arxiv.org/abs/2603.19201) 的 visuo-tactile world modeling 方向）

### 12.5 跟 Gemma / PaliGemma 的 architectural inheritance

FTP-1 的 vision-language expert 来自 PaliGemma [22](https://arxiv.org/abs/2407.07726)，action expert 是 π0.5 [5](https://arxiv.org/abs/2504.16054) 的 flow-matching DiT 风格 [86](https://arxiv.org/abs/2212.09748)。这意味着 FTP-1 继承了 PaliGemma 的 SigLIP vision encoder + Gemma 2 language decoder 这套 stack，然后在它上面加了一个 300M tactile expert 和 adaptive RMSNorm proprioception injection。

这个 inheritance 让 FTP-1 能 leverage π0.5 已有的 vision-language grounding，但同时也限制了它的 architectural freedom——比如如果 tactile expert 想做 cross-attention 到 vision token，会跟 PaliGemma 的 decoder-only 结构不太搭。Paper 选了最简单的 "action expert → tactile expert" 单向 attention，应该也是受这个 inheritance 限制。

---

## 13. 总结

FTP-1 的 contribution 我会概括成：

1. **第一次证明 tactile manipulation 可以做 generalist foundation policy**——MTTS + heterogeneous encoder + shared tactile expert 这套 recipe work。
2. **3000 小时 heterogeneous tactile dataset**，跨 21 sensor，是当前最大 tactile manipulation corpus。
3. **+17.2% on seen sensor + +31% on unseen sensor** 的实验结果，加上 NTP-1 ablation 证明 gain 来自 transferable tactile knowledge 而非 data distribution。
4. **Open release** pretrained model + dataset + code，给 community 一个 shared starting point——这个对 tactile research 的 ecosystem 建设很重要。

局限主要在 scale（50k step saturate）、unseen sensor sample size、tactile expert capacity 偏小、没有 low-level force control。但作为一个 "first foundation baseline"，这个 work 的 design choice 和实验设计都很 solid，能成为后续 tactile scaling 的 anchor。

Paper 在 philosophy 上是把 Open X-Embodiment [37](https://arxiv.org/abs/2310.08896) 的 cross-embodiment generalization 思路扩展到 tactile modality + dexterous hand——这个方向我 personal 觉得是对的，因为 tactile sensing 和 dexterous manipulation 是 robot learning 里 contact-rich task 的 long pole，必须 foundation 化才能 scale。

主要参考链接：
- Project page: https://ftp1-policy.github.io/
- π0.5 base: https://arxiv.org/abs/2504.16054
- T3 encoder: https://arxiv.org/abs/2406.13640
- UniDex / FAAS: https://arxiv.org/abs/2603.22264
- Tactile-VLA (baseline): https://arxiv.org/abs/2507.09160
- UniVTAC benchmark: https://arxiv.org/abs/2602.10093
- Sparsh: https://arxiv.org/abs/2410.24090
- HPT: https://arxiv.org/abs/2409.20537
- Open X-Embodiment: https://arxiv.org/abs/2310.08896
