---
source_pdf: Vid2RobotEnd-to-endVideo-conditionedPolicy LearningwithCross-AttentionTransformers.pdf
paper_sha256: 76a20b2d5b2244299ed47ef46dc13f1c29e23c95ac9ee44a87a15604bb72ddea
processed_at: '2026-08-13T00:29:56-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Vid2Robot 人话版

## 一句话说清楚这 paper 干啥

你给 robot 看一段 human 做事的 video，比如"把可乐罐从抽屉里拿出来放桌上"，robot 看完就自己去干了。整个过程 end-to-end，不需要语言 instruction，不需要 teleoperation，就是 video in，action out。

---

## 为啥要做这个

你想想，RT-1、RT-2 这些 model 都靠 language 来 specify task。问题来了——"open drawer"、"open cabinet"、"open jar" 在 language 层面都共享 verb "open"，但 robot 做起来的 motor control 完全不一样。Language 这种 discrete 符号系统在描述连续 motor skill 时信息密度不够。

Video 天然把 **what** 和 **how** 都编码进去了。你看一个人拧 jar 盖，你能 infer 出来"哦这是 screw motion"，看一个人开 drawer，你能 infel 出来"哦这是 pull motion"。Vid2Robot 就是想让 robot 也有这个能力。

更深一层 motivation：internet 上 human video 是海量免费 resource，如果 robot 能直接从这些 video 学，data bottleneck 就缓解了。这个方向的重要性不用多说。

参考 RT-1: https://robotics-transformer1.github.io/
参考 RT-2: https://robotics-transformer2.github.io/

---

## 数据怎么搞的——这里有个聪明的 trick

要训 video-conditioned policy，你需要 pairs：一段 prompt video + 一段 robot 做 same task 的 trajectory。问题是怎么 cheaply 大量搞这种 pair。

作者搞了三种 data source，配比 90:5:5：

### Robot-Robot（90%）
直接拿已有的 robot trajectory dataset，用 text instruction 做 pairing——只要两条 trajectory 的 text instruction 一样就认为是 same task，配成一对。这个几乎免费，因为 robot data 已经有了，你只要 cross-match 一下。Open X-Embodiment dataset 里就有现成的。

### Hindsight Human-Robot（5%）
这个最聪明。你手头有 robot trajectory 的 text instruction，比如"pick coke can from bottom drawer and place on counter"。你找几个 human，让 human 在 robot camera 视角下做这个 task，录下来。这样你就有了 (human video, robot trajectory) pair，完全不用 teleoperation。

好处是 human video 采集便宜，坏处是 task diversity 没增加——你只能 cover robot dataset 里已有的 task instruction。

### Co-located Human-Robot（5%）
Human 和 robot 在同一个 workspace 做 same task，这是 gold standard——背景、lighting 都一致，embodiment gap 还在。但很贵很慢，只能小量搞。

**核心 insight**：90% 便宜 data 让 model 建立 robot domain 的 strong prior，5%+5% human data 桥接 embodiment gap。这个 data mixture philosophy 跟 instruction tuning 里"大量 easy data + 少量 hard data"的配方如出一辙。

参考 Open X-Embodiment: https://robotics-transformer-x.github.io/

---

## 模型长啥样——四个模块串起来

### Module 1: Prompt Video Encoder
输入 16 frames × 224×224 的 human demo video。每帧过 ViT-B/16，得到 16×196=3136 个 token，每个 768 dim。然后过 Perceiver Resampler 压到 64 个 token。输出叫 $z_{prompt}$，64×768。

这个 module 要学会从 video 里 infer "这是什么 task"以及"怎么做"。

### Module 2: Robot State Encoder
输入 8 frames × 224×224，是 robot camera 最近的 history。结构跟 Module 1 完全一样，**weights shared**。输出叫 $z_{state}$，64×768。

这个 module 编码 robot 当前环境里有啥 object、robot arm 在哪、recent history 说明刚做了啥。

### Module 3: State-Prompt Encoder（cross-attention，这是灵魂）
$$z_{state|prompt} = \text{CrossAttn}(Q=z_{state}, K=z_{prompt}, V=z_{prompt})$$

Query 是 state tokens，Key/Value 是 prompt tokens。4 层 cross-attention transformer。

**直觉**：state 说"我现在桌上有个 apple、一个 banana、一个 orange"，prompt 说"把 apple 放篮子里"。Cross-attention 就是让 state 去 "问" prompt："你要哪个 object？"，prompt 回答："apple-like 的那个"。然后 state 就知道该 focus apple 了。

这就是 cross-attention 比 FiLM 强的地方。FiLM 是全局 modulation，用 prompt 的一个 vector 调制整个 state feature，没法做 spatial localization。Cross-attention 能动态决定"看 state 的哪个位置"。

### Module 4: Action Decoder
用 **learnable action position embeddings**（11 个，对应 11 个 action dimension）作为 query，$z_{state|prompt}$ 作为 Key/Value。4 层 cross-attention，输出 11×768，project 到 11×256，softmax over 256 个 bin，argmax 选 action。

11 个 action dimension 是：mode（terminate/arm/base/both）、gripper xyz position、gripper 三个 rotation angle、gripper closedness、base xy displacement、base rotation。

**关键设计**：用 action position embedding 一次 forward pass 预测所有 action dimension，类似 ACT paper 的做法。RT-1 是 autoregressive 一个一个 token 吐，要多次 forward pass。Vid2Robot 这样 inference 快很多。同时 prediction horizon=4，一次预测未来 4 步 action，执行时只用第一步——这减少 action 抖动，让 trajectory 更平滑。

参考 ACT: https://tonyzhaozh.github.io/aloha/
参考 Perceiver IO: https://arxiv.org/abs/2107.14795

---

## 为啥用 cross-attention 不用 self-attention——算力问题

如果把 prompt 和 state 所有 token 拼起来做 self-attention：
- 16×196 + 8×196 = 4704 个 token
- Attention matrix 是 4704×4704 ≈ 22M entries

用 Perceiver Resampler 各自压到 64 个 token 后，cross-attention 只需要：
- Prompt resampler: 16×196×64 ≈ 200k
- State resampler: 8×196×64 ≈ 100k
- State-prompt cross-attn: 64×64 ≈ 4k
- 总共约 300k entries

**降了 70 倍**。这使得 training 用 batch size 2048 成为可能。

---

## 四个 Loss——一个主菜三个配菜

总 loss 是四个 loss 取平均：
$$L = \frac{1}{4}(L_{CE} + L_{TCC} + L_{VVCL} + L_{VTCL})$$

### $L_{CE}$: Action Prediction（主菜）
标准 cross-entropy，把 action 离散化成 256 个 bin 当 classification 做。这个 loss 训所有 module 的参数。

### $L_{TCC}$: Temporal Cycle Consistency（配菜 1，很关键）

这个 loss 来自 Dwibedi 2019 的 TCC paper。目标：让 prompt video 和 robot video 在 **时间维度上对齐**——同一 task phase 的 frame embedding 应该接近。

具体怎么做：对 prompt video 的每一帧 $t$，在 robot video 里找一个 "soft neighbor"（用 embedding 距离的 softmax 加权平均），然后再从这个 soft neighbor cycle back 回 prompt video，看能不能回到 frame $t$ 本身。

公式（Eq 1）：
$$\widetilde{E_{pr}^t} = \sum_k^{L_r} \alpha_k E_r^k, \quad \alpha_k = \frac{e^{-\|E_p^t - E_r^k\|^2}}{\sum_k^{L_r} e^{-\|E_p^t - E_r^k\|^2}}$$

变量解释：
- $E_p^t$: prompt video 第 $t$ 帧的 embedding
- $E_r^k$: robot video 第 $k$ 帧的 embedding
- $L_r$: robot video 帧数
- $\alpha_k$: prompt frame $t$ 对 robot frame $k$ 的 attention weight（基于 embedding 距离）
- $\widetilde{E_{pr}^t}$: prompt frame $t$ 在 robot video 中的 "软对应位置"

然后 cycle-back loss（Eq 2）：
$$L_{TCC} = \sum_{t \in V_p} (\widehat{E_{pr}^t} - t)^2$$

$\widehat{E_{pr}^t}$ 是 cycle back 后预测的 frame index，理想情况等于 $t$。

**直觉**：这个 loss 强制 image encoder 学到 task-progress-aware 的表示。即使 human 用左手、robot 用机械臂，"已经抓起物体"这个 phase 的 frame embedding 应该接近。它让 encoder 对 embodiment、lighting、background invariant，只 encode "task 进展到哪一步了"。

参考 TCC: https://arxiv.org/abs/1904.07846

### $L_{VVCL}$: Video-Video Contrastive（配菜 2，很关键）

用 SigLIP loss。把 prompt video 的 64 个 token 用 attention pooling 压成 1 个 embedding $Z_{prompt}$，robot video 同理得 $Z_{robot}$。batch size B，算 B×B 的 similarity matrix：

$$\hat{Y} = (Z_{robot} \cdot Z_{prompt}^T) \cdot \tau + b$$

- $\tau$: learnable temperature
- $b$: learnable bias
- Label matrix $Y = 2I_B - 1$（对角线 +1，其余 -1）

SigLIP loss：
$$L_{VVCL} = -\sum \log \sigma(Y \cdot \hat{Y})$$

**直觉**：让 same task 的 prompt video embedding 和 robot video embedding 接近，不同 task 的远离。TCC 是 frame-level temporal alignment，VVCL 是 video-level task identity。两者 complementary。

SigLIP 比 InfoNCE 好的地方是用 sigmoid 而不是 softmax over batch，对 batch size 不那么敏感，batch 内每个 pair 独立判别。

参考 SigLIP: https://arxiv.org/abs/2303.15343

### $L_{VTCL}$: Video-Text Contrastive（配菜 3，锦上添花）

让 video embedding 和对应 text instruction embedding 对齐。继承自 BC-Z 的 auxiliary language regression loss。Ablation 发现这个 loss 只贡献 +1-2%，主要靠 TCC + VVCL 撑场子。

---

## 实验结果——几个 take-away

### 主结果（Table I）

| Prompter | Model | Overall Success |
|---|---|---|
| Robot | BC-Z | 52.6% |
| Robot | Vid2Robot | 54.9% |
| Human | BC-Z | 30.6% |
| Human | Vid2Robot | 52.8% |

核心 claim：**human prompt 下 Vid2Robot 比 BC-Z 高 22 个点**。Robot prompt 下两者差不多，因为 90% training data 是 robot-robot pair，robot domain prior 已经很强了。

### Partial Success 分解（Fig 6）

把一个 rollout 拆成 4 个 milestone：reach correct object → grasp → release at correct location → terminate correctly。

| Milestone | BC-Z | Vid2Robot |
|---|---|---|
| Reach | 70% | 78% |
| Grasp | 45% | 65% |
| Release | 40% | 58% |
| Terminate | N/A | 57% |

**Grasp 环节 Vid2Robot 提升 20 个点最大**。Grasp 是 manipulation 最难的部分，需要精确 geometric reasoning。Cross-attention 让 model 能把 prompt 中的 grasp motion pattern 精确 transfer 到当前 observation。

### Cross-Object Motion Transfer（这个最 cool）

训练时 prompt video 里的 object 总是和 robot 当前环境一致。测试时故意不一致——prompt 是 "place coke can upright"，但 robot 面前放 orange、banana、chips bag、soft toy、wrist watch 这些完全不同的 object。

| Model | Overall |
|---|---|
| BC-Z | 17.5% |
| Vid2Robot | 34.2% |

Vid2Robot 能把 "place upright" 这个 **motion 概念** 抽象出来，apply 到没见过的 object 上。比如 prompt 里是 coke can，robot 面前是 green can，它会优先选 green can（形状最接近的）来做 "place upright" 动作。

这是一个 emergent behavior，暗示 model 内部形成了 verb 和 object 的 disentangled representation——verb "place upright" 是一个 abstract motion template，object 是填充 slot。这跟 linguistics 里的 verb argument structure 对应。

BC-Z 在这个 setting 基本崩了，因为它用 FiLM，prompt 的 object 特征和 motion 特征混在一起，没法 transfer 到新 object。

### Ablation（Fig 8）

| Variant | Success Rate |
|---|---|
| Full Vid2Robot | 61% |
| Without Video-Text CL | ~59% |
| Only action prediction loss | 45% |

Auxiliary losses 整体贡献 +16 个点。Video-text CL 只贡献 1-2 个点，主要靠 TCC + VVCL。这说明 **video-video 对齐比 video-language 对齐重要**——毕竟最终目标是 video-conditioned policy，language 只是训练时的 bridge。

---

## 我的几个 intuition

### 1. Cross-attention 为什么比 FiLM 强这么多

FiLM 是全局 modulation：用一个 vector 调制整个 feature map 的 scale 和 shift。对于 "从 {apple, banana, orange} 里 pick apple" 这种需要 spatial localization 的 task，FiLM 没法告诉 model "看 apple 那个位置"。

Cross-attention 可以。State token 作为 query，每个 token 对应 spatial 的一个位置，它去 "问" prompt "你要我这里的 object 吗？"，prompt token 回答 similarity score。这样 spatial 位置和 task relevance 就 coupled 起来了。

本质上 cross-attention 是一种 **learnable routing**——根据当前 state 动态决定从 prompt 里取什么信息。FiLM 是 static modulation，不管 state 是啥都用同一个 modulation pattern。

### 2. Auxiliary loss 的真正作用

100k paired data 训 ViT-B/16 + 4 层 transformer，数据量严重不够。Auxiliary loss 本质是 **regularizer**，给 image encoder 提供额外 supervision signal，防止它过拟合到 spurious correlation。

- TCC: 强制 encoder 学 temporal structure，不能只看单帧
- VVCL: 强制 encoder 学 task identity，不能只看 background
- VTCL: 强制 encoder 学 language grounding

这跟 Vision-Language Pretraining 里 image-text contrastive + captioning + MLM 多任务联训一个思路——data 不够就用 task diversity 补。

### 3. Cross-Object Transfer 暗示了什么

如果这个能力能 scale，意味着 robot 可以从 YouTube 上看 human 做事（用 robot 没见过的 object），然后在自家环境用自己有的 object 复现。这是 general-purpose robot 的关键能力。

更深一层：这说明 video-conditioned policy 可能比 language-conditioned policy 更适合学 motor skill，因为 video 里的 motion information 是连续的、demonstrative 的，language 是离散的、descriptive 的。Verb 这个 linguistic category在 model 内部可能 emergently 对应到某种 motion primitive representation。

### 4. 跟 LLM in-context learning 的类比

Vid2Robot 的 prompt video 机制跟 LLM in-context learning 结构上很像：
- Prompt video = few-shot example
- Robot state = query
- Action = completion

Cross-attention 就是 in-context learning 的实现机制。如果 data scale 上去，可能看到类似的 scaling law——更多 diverse prompt video → 更好 generalization。这个方向值得探索。

参考 BC-Z: https://sites.google.com/view/bc-z
参考 in-context learning scaling: https://arxiv.org/abs/2305.12738

---

## 局限性

1. **Self-occlusion**：robot arm 挡住 camera，IK 解算的 pose 也可能遮挡视野。这个是 hardware limitation，需要 multi-view 或者 wrist camera 解决。
2. **Grasping**：transparent、deformable object 难抓。单目 depth estimation 不完整。
3. **Data 依赖 text pairing**：Hindsight Human-Robot 需要 text instruction 才能 pair，对 in-the-wild YouTube video 不可扩展。需要 self-supervised video retrieval 来 pair。
4. **Long-horizon**：当前用 prompt chaining 手动串起来，没有 end-to-end 训练 long-horizon task。16 frame prompt 也限制了对长 task 的表达。
5. **5-7 Hz inference**：对 fast dynamic task 可能不够。Action chunking（预测 horizon=4）部分缓解了这个问题。

---

## 实现细节

- Jax 实现
- Batch size 2048（很大，需要 TPU 级别算力）
- 200K iterations，AdamW，lr 8e-5 cosine schedule
- Inference: client-server 架构，policy 在 server 端，robot client 维护 8 frame history buffer
- Action prediction 5-7 Hz

数据增强比较保守（crop 0.95-1.0，brightness 0.9-1.1 等），因为 manipulation 对 spatial 信息敏感，过强增强会破坏 task structure。

---

## 总结一句话

Vid2Robot 证明了：**cross-attention transformer + TCC/VVCL auxiliary loss + 90/5/5 data mixture** 这个配方，能让 robot 从 human video 学 manipulation skill，而且能 emergent 出 verb-level 的 motion abstraction 实现 cross-object transfer。这是 video-conditioned robot policy 的一个 clean baseline，也为从 internet human video scale up robot learning 指了条路。

Paper link: https://vid2robot.github.io/

想深入聊哪个 part——TCC 的 numerical stability、cross-attention pattern 可视化、还是 cross-object transfer 的 mechanism——你说了算。

---

# Vid2Robot 论文详解

## I. 核心动机与问题定位

这篇 paper 解决的核心问题：**如何让 robot 通过观察 human demonstration video 来执行 task**，而不依赖 language instruction。

在 RT-1, RT-2, RT-X 等 multi-task manipulation system 中，task specification 通常用 natural language。但 language 存在 polysemy 问题——例如 "open drawer", "open cabinet", "open jar with screw cap" 共享同一个 verb "open"，但需要完全不同的 motor control。Goal image 也不够，例如 "hold the flag" 和 "wave the flag" 终态图像可能相同，但 motion 完全不同。Video 同时携带 **what to do** 和 **how to do it** 的信息。

参考链接：
- RT-1: https://robotics-transformer1.github.io/
- RT-2: https://robotics-transformer2.github.io/
- BC-Z: https://sites.google.com/view/bc-z
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

## II. 数据集设计：三种 Pairing 策略

这是 paper 最关键的 design choice 之一。作者构造了三类 paired data：

| Dataset Name | Prompt Video Embodiment | Prompt vs Robot Scene | 占比 |
|---|---|---|---|
| Robot-Robot | Robot | Different | 90% |
| Hindsight Human-Robot | Human | Different | 5% |
| Co-located Human-Robot | Human | Same | 5% |

总量：~100k robot videos + ~10k human videos，覆盖 RT-1 和 RT-2 的 task suite。Pairing 时每个 robot trajectory 采样 3 个 prompt video，所以总 pairs 数为 ~360k + 15k + 5k。

**关键 insight**：Robot-Robot pairing 几乎免费（直接复用已有 trajectory data），而 Co-located Human-Robot 是 gold standard 但 expensive。Hindsight Human-Robot 是一个聪明的中间方案——让 human 在 robot camera 视角下做 task，避免 teleoperation 成本，同时引入 embodiment diversity。

这种 data mixture 的设计哲学让我联想到 in-context learning 中 "instruction tuning" 的数据配比——大量 "easy" paired data 加少量 "hard" paired data，让 model 学到 invariance 同时保留 task semantics。

---

## III. 模型架构：四个模块详解

输入：
- Prompt video $V$：16 frames × 224×224
- Robot state $S_t = \{x_i\}_{i=t-k-1}^{t}$：8 frames × 224×224，其中 $k=8$ 是 history length
- 输出：11-dim action vector $a_t = [m, g_x, g_y, g_z, \theta_{xy}, \theta_{yz}, \theta_{zx}, c, b_x, b_y, b_\theta]$
  - $m$: mode (terminate / move arm / move base / both)
  - $g_x, g_y, g_z$: gripper position
  - $\theta_{xy}, \theta_{yz}, \theta_{zx}$: gripper orientation (rotation along xy, yz, zx planes)
  - $c$: gripper closedness
  - $b_x, b_y, b_\theta$: base displacement and rotation

每个 dimension 被离散化成 256 个 bins，问题被 cast 成 classification。

### Module 1: Prompt Video Encoder
$$z_{prompt} = \psi_p(\phi_p(V))$$
- $\phi_p$: per-frame ViT-B/16 image encoder，输出 16×196×768
- Reshape 到 3136×768（all space-time tokens 拉平）
- $\psi_p$: Perceiver Resampler (2 layers, 64 latent tokens, 12 heads)，输出 64×768

### Module 2: Robot State Encoder
$$z_{state} = \psi_s(\phi_s(S_t))$$
- 结构与 prompt encoder 完全相同，**weights shared**：$\phi_p = \phi_s = \phi$
- 输入 8 frames → 8×196×768 → reshape 到 1568×768 → Perceiver Resampler → 64×768

### Module 3: State-Prompt Encoder (Cross-Attention)
$$z_{state|prompt} = \text{CrossAttn}(Q=z_{state}, K=z_{prompt}, V=z_{prompt})$$
- 4 层 cross-attention transformer，768 dim, 8 heads
- **Query 是 state tokens, Key/Value 是 prompt tokens**
- Intuition：state 编码当前环境（apple, banana, orange 都在桌上），prompt 编码 task（pick apple in basket），cross-attention 学习从 state 中"挑出" prompt-relevant 的 object 信息

### Module 4: Action Decoder
- 输入 query: **learnable action position embeddings**（11×768，每个 action dimension 一个 token）
- Key/Value: $z_{state|prompt}$
- 4 层 cross-attention transformer
- 输出 11×768 → Linear → 11×256 → softmax → argmax 选 bin

**关键设计**：使用 action position embeddings 让 action 在一次 forward pass 中全部预测出来，类似 ACT (Bimanual Manipulation) 的设计。这避免了 RT-1 的 autoregressive decoding 多次 forward pass 的延迟。同时 prediction horizon = 4（一次预测 4 步 action），执行时只用第一步。

### Cross-Attention vs Self-Attention 的计算优势

如果不压缩，self-attention over 所有 video tokens：
- Total tokens = 8×196 + 16×196 = 4704
- Attention matrix 大小 = 4704² ≈ 22M entries

用 Perceiver Resampler 压缩到 64 latent tokens 后：
- Prompt side: 16×196×64 ≈ 200k
- State side: 8×196×64 ≈ 100k
- 总计约 300k entries ≈ 0.3M

**降低约 70 倍** attention computation。

参考：
- Perceiver IO: https://arxiv.org/abs/2107.14795
- Flamingo: https://arxiv.org/abs/2204.14198
- ACT (Bimanual): https://tonyzhaozh.github.io/aloha/

---

## IV. 训练目标：一个主 Loss + 三个 Auxiliary Loss

总训练目标：
$$L = \frac{1}{4}(L_{CE} + L_{TCC} + L_{VVCL} + L_{VTCL})$$

### (1) Action Prediction Loss $L_{CE}$
$$L_{CE}(a_t, \hat{a}_t) = -\sum_\tau a_t \log \hat{a}_t$$
- $a_t$: expert action (one-hot over 256 bins)
- $\hat{a}_t$: predicted probability distribution
- 标准 cross-entropy classification loss

### (2) Video Alignment Loss (TCC) $L_{TCC}$
来自 Dwibedi et al. 2019 的 Temporal Cycle-Consistency。目标是让 prompt video 和 robot video 在 **temporal dimension 上对齐**，即 task progress 相同的 frame 应该有相似的 embedding。

先对每帧 image embedding 通过 2-layer MLP projector $\Phi$ 得到 $E_i = \{\Phi(v_i^1), \Phi(v_i^2), ..., \Phi(v_i^{L_i})\}$。

**Soft neighbor 计算**（公式 1）：
$$\widetilde{E_{pr}^t} = \sum_k^{L_r} \alpha_k E_r^k$$
$$\alpha_k = \frac{e^{-\|E_i^t - E_j^k\|^2}}{\sum_k^{L_j} e^{-\|E_i^t - E_j^k\|^2}}$$

变量含义：
- $E_i^t$: 第 $i$ 个 video 的第 $t$ 帧 embedding（这里 $i=p$ 即 prompt video）
- $E_j^k$: 第 $j$ 个 video 的第 $k$ 帧 embedding（这里 $j=r$ 即 robot video）
- $L_r$: robot video 的帧数
- $\alpha_k$: softmax weight，衡量 prompt frame $t$ 与 robot frame $k$ 的相似度
- $\widetilde{E_{pr}^t}$: prompt frame $t$ 在 robot video 中的 "soft" 对应位置（一个加权平均的 embedding）

**Cycle-back**：再从 $\widetilde{E_{pr}^t}$ 出发，找到它在 prompt video $E_p$ 中的 soft neighbor，记为 $\widehat{E_{pr}^t}$。理想情况下 $\widehat{E_{pr}^t}$ 应该回到 frame $t$ 本身，即 $\widehat{E_{pr}^t} - t \to 0$。

TCC loss：
$$L_{TCC}(E_p, E_r) = \sum_{t \in V_p} (\widehat{E_{pr}^t} - t)^2$$
$$L_{TCC} = \frac{L_{TCC}(E_p, E_r) + L_{TCC}(E_r, E_p)}{2}$$

对称地双向计算，让 prompt 和 robot video 的 frame embedding 互相 cycle-consistent。

Intuition：这个 loss 强制 image encoder 学到 **task-progress-aware** 表示，对 embodiment / lighting / background 不变。即使 human 用左手做、robot 用机械臂做，"已经拿起物体"这个 phase 的 frame embedding 应该接近。

参考：TCC paper https://arxiv.org/abs/1904.07846

### (3) Prompt-Robot Video Contrastive Loss (VVCL) $L_{VVCL}$
基于 SigLIP loss（sigmoid loss 而非 InfoNCE 的 softmax over batch）。

用 Attention Pooling 把 N=64 个 prompt tokens 压成 1 个 embedding，得到 $Z_{prompt}$ 和 $Z_{robot}$，都是 $B \times d$。

Logit matrix：
$$\hat{Y} = (Z_{robot} \cdot Z_{prompt}^T) \cdot \tau + b$$

变量：
- $\tau$: learnable temperature
- $b$: learnable bias
- Label matrix $Y = 2I_B - 1$（对角线 +1，off-diagonal -1）

SigLIP loss：
$$L_{VVCL} = \sigma'(Z_{prompt}, Z_{robot}) = -\sum \log \sigma(Y \cdot (Z_1 \cdot Z_2^T) \cdot \tau + b)$$

Intuition：让相同 task 的 prompt video embedding 和 robot video embedding 接近，不同 task 的远离。这个 loss 让 model 学习 **task semantic similarity**，捕获 TCC 之外的"全局 task identity"信息。

参考：SigLIP https://arxiv.org/abs/2303.15343

### (4) Video-Text Contrastive Loss (VTCL) $L_{VTCL}$
$$L_{VTCL} = (\sigma'(Z_{prompt}, Z_{text}) + \sigma'(Z_{robot}, Z_{text})) / 2$$

让 video embedding 和对应 language instruction 的 text embedding 对齐。这部分继承自 BC-Z 的 auxiliary language regression loss。

---

## V. 实验结果解析

### Table I: 主要任务成功率（72 trials per row）

| Prompter | Model | Overall |
|---|---|---|
| Robot | BC-Z | 52.6% |
| Robot | Vid2Robot | 54.9% |
| **Human** | **BC-Z** | **30.6%** |
| **Human** | **Vid2Robot** | **52.8%** |

关键观察：
1. Robot prompt 下两者相当（Vid2Robot 略高 2.3%）
2. **Human prompt 下 Vid2Robot 比 BC-Z 高 22.2%**——这是 paper 的核心 claim
3. 单看 task：Human prompt 下 "open middle drawer" BC-Z 0% → Vid2Robot 62.5%；"close middle drawer" 50% → 87.5%；"pick-place on" 12.5% → 50%

为什么 Vid2Robot 在 human prompt 下优势巨大？我的 intuition：
- Cross-attention 让 model 能动态聚焦 prompt 中 task-relevant 的部分（比如 human 手部动作），忽略 embodiment 差异
- TCC + VVCL 两个 auxiliary loss 显式鼓励 embodiment-invariant 表示
- ViT-B/16 比 BC-Z 的 ResNet-18 表达能力强得多
- 90% Robot-Robot training data + 5% Human-Robot data 让 model 先建立 robot domain 的 strong prior，再用少量 human data 桥接 embodiment gap

### Fig 6: Partial Success Rate 分解

| Milestone | BC-Z | Vid2Robot |
|---|---|---|
| Reach correct object | 70% | 78% |
| Grasp | 45% | 65% |
| Release at correct location | 40% | 58% |
| Terminate correctly | N/A | 57% |

Vid2Robot 在 **grasp** 环节提升最大（+20%）。Grasp 是 manipulation 中最难的环节，需要精确的 geometric reasoning。Cross-attention 让 model 能把 prompt 中的 grasp motion pattern transfer 到当前 observation。

### Table II: 统计显著性（314 rollouts）

| Model | place coke can upright | close middle drawer | Overall |
|---|---|---|---|
| BC-Z | 19.4±9.5% | 39.2±10.8% | 30.2±7.1% |
| Vid2Robot | **39.1±9.9%** | **48.7±11.2%** | **43.4±7.2%** |

Confidence interval 不重叠（除 close drawer 接近），统计上显著。

### Table III: Cross-Object Motion Transfer（emergent behavior！）

训练时 prompt video 中出现的 object 总是和 robot 当前 observation 中的 object 一致。但测试时给 prompt "place coke can upright"，robot 面前放 orange / green can / chips bag / banana / soft toy / wrist watch，结果：

| Model | pick | pick-place on | place into | place upright | knock over | Overall |
|---|---|---|---|---|---|---|
| BC-Z | 45.8% | 0.0% | 29.2% | 12.5% | 0.0% | 17.5% |
| Vid2Robot | 45.8% | 25.0% | 54.2% | 16.7% | 29.2% | **34.2%** |

这是一个非常有意思的 emergent behavior：**Vid2Robot 学到了 verb/motion 的 abstract 表示**，能把 prompt 中的 motion pattern（"place upright"）generalize 到完全没在 prompt 中出现过的 object。BC-Z 在这个 setting 下几乎完全失败。

我的 intuition：这是因为 cross-attention 让 model 把 prompt 中的 "motion/style" 信息和 "object identity" 信息分离开——state encoder 看到当前 observation 中有什么 object，prompt encoder 提供 "对某个 object 做什么动作"，State-Prompt Encoder 把两者融合。这有点类似 CLIP 中 image embedding 和 text embedding 的解耦。

### Ablation（Fig 8）

| Variant | Success Rate |
|---|---|
| Full Vid2Robot (all losses) | 61% |
| Without Video-Text CL | ~59-60% |
| Only action prediction loss | 45% |

Auxiliary losses 整体贡献 +16%。其中 video-text contrastive loss 贡献很小（+1-2%），主要贡献来自 TCC + VVCL。这说明 **video-video 对齐**比 video-language 对齐更重要——毕竟最终目标是 video-conditioned policy，language 只是 training 时的 bridge。

### "No-prompt" baseline
不给 prompt video，Vid2Robot 23% → 54.6%（with prompt），BC-Z 5% → 52.6%。说明 prompt 确实在做 task specification 的工作，model 没有单纯 memorize task distribution。

---

## VI. 与相关工作的关系

### 与 BC-Z 的对比
BC-Z 用 ResNet-18 + FiLM conditioning，是 video-conditioned policy 的经典 baseline。Vid2Robot 在以下几个方面做了升级：
1. ViT-B/16 image encoder（vs ResNet-18）
2. Cross-attention transformer（vs FiLM modulation）
3. Action position embedding 一次性预测（vs autoregressive）
4. TCC + VVCL auxiliary losses（vs 只有 language regression）

### 与 RT-1 / RT-2 的关系
RT-1/RT-2 是 language-conditioned。Vid2Robot 是 video-conditioned。Paper 在 §IV.A 论证 video conditioning 是 language 的 complementary，对 polysemy 和 motion-heavy task 有优势。但 video conditioning 也有劣势——raw video 高维、计算贵、prompt video 采集难。

### 与 XSkill, MimicPlay 的对比
- XSkill: 从 unpaired human + robot video 中做 cross-embodiment skill discovery（无监督）
- MimicPlay: hierarchical learning，用 human play video 做 high-level planning，robot teleop demo 做 low-level control
- Vid2Robot: 用 paired human-robot video（基于 text instruction pairing）做 end-to-end policy

Vid2Robot 的 pairing 需要 text instruction 作为 "桥梁"，这是一个 practical limitation。XSkill 完全 unsupervised 但需要更复杂的 skill discovery 机制。

### 与 One-Shot Visual Imitation 文献
Duan et al. 2017、Dasari & Gupta 2021、Mandi et al. 2022 都是 one-shot visual imitation 的经典工作，但多在 simulation 或 restricted setting。Vid2Robot 是 real-world multi-task 的 end-to-end 方案。

参考：
- XSkill: https://arxiv.org/abs/2310.12729
- MimicPlay: https://arxiv.org/abs/2302.12422
- XIRL: https://arxiv.org/abs/2210.06931

---

## VII. 局限性（§V）

1. **Self-occlusion**：robot arm 可能挡住 camera，IK 解算的 pose 也可能遮挡视野（Fig 9 top）
2. **Grasping failures**：transparent / deformable object 难抓（Fig 9 middle）
3. **Distractor + lighting**：可能识别错 object（Fig 9 bottom）
4. **数据规模**：100k robot videos + 10k human videos 对于大 transformer 来说仍不足，依赖 auxiliary loss 防止 overfit
5. **Long-horizon**：当前用 prompt chaining（Fig 11），没有 end-to-end 训练 long-horizon task
6. **Pairing 策略依赖 text**：Hindsight Human-Robot 需要 task instruction 才能 pair，对 in-the-wild video 不可扩展

---

## VIII. 我的 Intuition 与延伸思考

### 1. 为什么 Cross-Attention 比 FiLM 强那么多？
FiLM 是 **全局 modulation**——用一个 vector 调制整个 ResNet feature map。对于 "pick apple from {apple, banana, orange}" 这种需要 **spatially-localized** decision 的 task，FiLM 没法告诉 model "关注 apple 那个位置"。Cross-attention 可以：state token 作为 query 去 "询问" prompt token "你想要哪个 object"，prompt token 回答 "apple-like 的那个"。

### 2. Auxiliary Losses 的角色
数据量 100k 对于 ViT-B/16 + 4-layer transformer 来说严重不足。Auxiliary losses 本质上是 **multi-task regularizer**：
- TCC: 强制 temporal structure 学习
- VVCL: 强制 task identity 学习
- VTCL: 强制 language grounding

这些 loss 都作用在 image encoder / resampler 输出上，相当于给 encoder 提供 "auxiliary supervision signal"，避免它过拟合到 spurious correlation（比如 background pattern）。

这让我联想到 Vision-Language Pretraining 中 image-text contrastive + image captioning + MLM 多任务联合训练的思路。

### 3. Cross-Object Motion Transfer 的深层含义
这是 paper 最 exciting 的发现。它暗示 model 内部形成了 **motion concept** 的 disentangled representation——"place upright" 这个 verb 被编码成某种抽象的 motion pattern，可以脱离具体 object 应用。

这和 linguistic 的 "verb argument structure" 有对应关系：verb 定义 action template，object 填充 slot。Vid2Robot 似乎 emergently 学到了这种结构。

如果这个能力能够 scale，意味着 robot 可以从 YouTube 上看 human 做 task（用未见过的 object），然后在自家环境用自己有的 object 复现——这是通向 general-purpose robot 的关键能力。

### 4. 与 LLM In-Context Learning 的类比
Vid2Robot 的 prompt video 机制和 LLM 的 in-context learning 有结构相似性：
- Prompt video = context example
- Robot state = query
- Action = completion

Cross-attention 就是 in-context learning 的实现机制。这个类比让我猜测：如果 data scale 上去，Vid2Robot 可能展现更明显的 in-context learning scaling law。

### 5. 数据效率 vs 模型规模
100k paired data 训出 50%+ 的 success rate，说明 cross-attention transformer 的 inductive bias 对 imitation learning 很有效。对比 RT-2 用 100k+ robot demo + VLM pretraining 才达到类似水平，Vid2Robot 的 sample efficiency 优势明显。

### 6. 潜在改进方向
- 用 video diffusion model（如 Sora-style）做 world model，提供 "imagined robot video" 作为额外训练信号
- 引入 3D representation（NeRF / 3D Gaussian Splatting）解决 occlusion 问题
- 用更长的 prompt video（当前 16 frames 限制 long-horizon task 表达）
- Self-supervised pairing（替代 text-instruction pairing），用 video retrieval 自动找相似 task

---

## IX. 实现细节（Appendix）

### Hyperparameters (Table V)
- Batch size: **2048**（很大，需要大算力）
- Learning rate: 8e-5, AdamW, cosine schedule, warmup 2000 steps
- Training: 200K iterations
- Image size: 224×224
- Prompt frames: 16
- Robot frames: 8
- Prediction horizon: 4

### Data Augmentation (Table VI)
- Crop: height (0.95, 1.0), width (0.95, 1.0)
- Brightness: (0.9, 1.1)
- Contrast: (0.8, 1.2)
- Hue: (0, 0.03)
- Saturation: (0.8, 1.2)

数据增强比较保守，因为 manipulation task 对 spatial 信息敏感，过强增强会破坏 task structure。

### Inference
- 5-7 Hz action prediction
- Client-server 架构，policy 在 server，robot client query
- 8 frame history buffer 在 client 维护

---

## X. 总结

Vid2Robot 是 video-conditioned robot policy 的一个 solid step。核心贡献：
1. 证明 cross-attention transformer 在 video-conditioned policy 上显著优于 FiLM-based BC-Z（+20% on human prompt）
2. TCC + VVCL 两个 auxiliary loss 对 embodiment-invariant representation 学习关键
3. 发现 cross-object motion transfer 这个 emergent behavior，暗示 model 学到了 abstract verb representation
4. 三种 pairing 策略的数据 mixture 是 practical 的 data scaling 方案

局限也很明显：data 仍依赖 text-instruction pairing，long-horizon 需要手动 chaining，grasping 成功率受 occlusion 限制。但作为 end-to-end video-conditioned policy 的 baseline，它设定了一个清晰的 reference point。

Paper link: https://vid2robot.github.io/
Code/data: 将随 paper 发布
Open X-Embodiment dataset: https://robotics-transformer-x.github.io/

如果你（Andrej）想深入讨论某个 specific aspect——比如 TCC loss 的 numerical stability、cross-attention 的 attention pattern 可视化、或者 cross-object transfer 的 mechanism——我可以展开讲。
