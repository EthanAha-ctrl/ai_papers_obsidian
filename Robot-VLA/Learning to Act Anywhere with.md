---
source_pdf: Learning to Act Anywhere with.pdf
paper_sha256: 1b8551db322a0e31514601d9d84879f85a5c391e4911e40752ea9df7567196f5
processed_at: '2026-08-05T13:48:42-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们用最直白的大白话来拆解 UniVLA。这篇 paper 的核心其实就是在给机器人控制做一件 LLM 早年做过的事：**搞出一个跨 embodiment 的 universal tokenizer，让 policy 能吃下互联网上所有无标签的视频。**

### 1. 核心痛点：Robot 数据的“巴别塔”

现在的 VLA model（像 OpenVLA, RT-2）面临一个死结：
第一，机器人 action labels 极其昂贵，导致数据上不了规模。
第二，不同 robot 的 action space 完全不通约。Franka 7-DoF、WidowX、人类的手、导航 agent，它们的 action 维度和物理意义南辕北辙。你在 Franka 上学的抓取 action，根本没法直接迁移到人类手上。

这就好比 LLM 面对全世界几百种语言，如果每种语言都从头学一套语法，那就永远做不出 GPT-4。

### 2. UniVLA 的解法：把视频变成“动作词元”

UniVLA 的思路极度优雅：既然不同的 robot 干同一件事的视觉表现是一样的（比如“靠近杯子”），那我就从视觉变化里无监督地提取一个离散的 latent action space，作为所有 robot 的通用语言。

这相当于给 visual dynamics 做了一个 tokenizer。LLM 有 BPE tokenizer，UniVLA 有 latent action tokenizer。

#### 2.1 为什么不用像素做预测？
如果直接在 pixel space 做下一帧预测（像 Sora 或者 Genie 那样），模型会分心去关注相机抖动、光照变化、背景里路过的猫。这些对执行任务是纯噪音。

UniVLA 借鉴了 JEPA 的思想，在 DINOv2 的 feature space 里做预测。公式如下：
$$ \mathcal{L}_{recon} = \|\hat{O}_{t+k} - O_{t+k}\|^2 $$
这里的 $O_{t+k}$ 是 $o_{t+k}$ 经过 DINOv2 提取的 patch features。因为 DINOv2 天生 object-centric 且 spatially aware，模型瞬间就屏蔽了高频纹理噪音，专心看语义层面的物理变化。

#### 2.2 神来之笔：两阶段信息解耦
那怎么把“跟任务相关的动作”和“跟任务无关的背景变化”分开？这是全 paper 最 brilliant 的地方，用了一个两阶段 bottleneck 设计。

**Stage 1：吸走 task-irrelevant 噪音**
$$\begin{cases} \text{Encode:} & \hat{a}_{TI} = \mathcal{Z}([O_t; O_{t+k}; a_{TI}; \ell]), \quad \tilde{a}_{TI} = \mathbf{VQ}(\hat{a}_{TI}) \\ \text{Decode:} & \hat{O}_{t+k} = \mathcal{F}([O_t; \tilde{a}_{TI}; \ell]) \end{cases}$$
变量解释：
- $\ell$：语言指令（比如“pick up the cup”）的 embedding。
- $\tilde{a}_{TI}$：容量极小的离散 latent action codebook（只有 16 个 token）。

注意这里 decoder 同时拿到了 $\tilde{a}_{TI}$ 和 语言指令 $\ell$。因为 codebook 容量极小，模型在优化时会走捷径：既然语言已经告诉我任务是“抓杯子”，那我 latent action 就不需要再 encode 任务信息了，我只去 encode 语言没说出来的视觉变化（比如窗帘动了、相机抖了）。这就把 task-irrelevant dynamics 逼到了 $\tilde{a}_{TI}$ 里。

**Stage 2：提炼 task-centric 动作**
$$\begin{cases} \text{Encode:} & \{\hat{a}_{TI}, \hat{a}_{TC}\} = \mathcal{I}([O_t; O_{t+k}; a_{TI}; a_{TC}]) \\ & \tilde{a}_{TI} = \mathbf{VQ}(\hat{a}_{TI}), \quad \tilde{a}_{TC} = \mathbf{VQ}_{TC}(\hat{a}_{TC}) \\ \text{Decode:} & \hat{O}_{t+k} = \mathcal{F}([O_t; \tilde{a}_{TI}; \tilde{a}_{TC}]) \end{cases}$$
Stage 1 训练完后，把 $\mathbf{VQ}$ 冻死。此时新加入一个新 codebook $\mathbf{VQ}_{TC}$，且不给语言指令了。此时模型要预测视觉变化，能用的只有 $\tilde{a}_{TI}$ 和 $\tilde{a}_{TC}$。既然噪音已经被冻死的 $\tilde{a}_{TI}$ 吸走了，那新学的 $\tilde{a}_{TC}$ 只能被迫去 encode “机械臂靠近杯子”这种真正的 task-centric 动作。

这是一种极其干净的信息阻隔。最终输出一个 $16^4$ 的空间，容量极小，信息极纯。

### 3. Policy: 把 Robotics 变成 Next-Token Prediction

有了这套 latent action tokenizer，接下来的 VLA pretraining 就极其顺滑了。

架构基于 Prismatic-7B (SigLIP + DINOv2 + LLaMA-2)。把原来的 LLM vocabulary 扩展 16 个 special tokens $\{\text{ACT}_1, \ldots, \text{ACT}_{16}\}$。

模型输入图像和语言，直接做 next-token prediction：
$$ \mathcal{L} = \mathbb{E}_{o_t, l, a_{z,<i}} \left[ -\sum_{i=1}^{N} \log \pi_\phi(\hat{a}_{z,i} = a_{z,i} | o_t, l, a_{z,<i}) \right] $$
其中 $N=4$，因为一个 latent action 序列由 4 个 token 组成。

**这里有一个极为震撼的 scaling 效率对比**：
OpenVLA 直接在原始连续 action space 上做 discretize，7-DoF 的 action vocabulary 是 $256^7 \approx 7.2 \times 10^{16}$。这让模型学起来极度费劲。
UniVLA 的 vocabulary 是 $16^4 = 65536$。
Action space 缩小了 12 个数量级！这直接让 UniVLA 只用了 OpenVLA 1/20 的 pretraining compute（960 vs 21,500 A100-hours），效果还更好。

### 4. History as Chain-of-Thought (CoT)

传统机器人 policy 要保持时序记忆，通常得堆叠过去 3-4 帧的图像输入。这非常吃 visual tokens 的算力。

UniVLA 发现，只把上一步输出的 4 个 latent action tokens 拼到当前的 instruction 后面，就能起到极好的 context 作用。公式上很简单，就是在输入端增加 history tokens。
这在直觉上跟 LLM 里的 Chain-of-Thought 完美对应。LLM 通过写出中间步骤来 refine 推理，UniVLA 通过读自己上一步输出的 latent action 来告诉自己“我现在处于什么状态，下一步该干嘛”。

在 R2R 导航任务上，加上 history latent action 让成功率从 30.6% 暴涨到 47.1%。

### 5. Post-training: 轻量级翻译器

既然 latent action 是跨 embodiment 的抽象语言，要落到具体机器人上，就需要一个翻译器。
设计了一个超轻量的 attention pooling head：

$$\begin{cases} \text{Visual Embedd.:} & E_v' = \mathcal{A}(Q=q_v, K=V=E_v) \\ \text{Action Embedd.:} & E_a' = \mathcal{A}(Q=q_a + E_v', K=V=E_a) \end{cases} $$

这里有个精妙的直觉：latent action tokens 告诉模型“去抓取”，但它不知道去哪抓。所以用 visual embedding $E_v$ 提取当前场景的 context，去 query latent action embeddings $E_a$，把抽象的动作 plan grounding 到具体的 3D 坐标和力度上。

这个 decoder 只有 10.8M 参数。加上 LoRA，整个 downstream adaptation 可训练参数仅 123M。对于新 robot、新任务，只需极少数据就能 fine-tune。

### 6. 构建更深的 Intuition

如果从大历史观来看 UniVLA，它在构建一条通往 Generalist Embodied AGI 的终南捷径。

**6.1 Cross-embodiment 就是 Cross-lingual Transfer**
BERT 和 GPT 当年证明，把多语言混合在一个 corpus 里 pretrain，模型自己就能学会跨语言的语义对齐。
UniVLA 证明，把 Franka、WidowX、Ego4D 人类视频混在一个 latent action space 里 pretrain，模型也能学会跨 embodiment 的 action 语义对齐。
实验里有个特别震撼的点：只用 Ego4D 人类第一视角视频 pretrain 的 UniVLA，在 LIBERO 机器人任务上跑出了 83.5% 的成功率，干翻了用真机数据训练的 OpenVLA (76.5%)。这证明 latent action 真正学到了"人类抓杯子"和"机械臂抓杯子"共享的那个 invariant representation。

**6.2 与 World Model 的融合潜力**
Latent action model 的 Decoder $\mathcal{F}$ 其实就是一个 World Model：给定当前状态和 latent action，预测下一时刻状态。
Paper 在 Future Work 里提了一嘴，但这其实是个大招。如果结合 test-time scaling，policy 可以在 latent space 里展开一棵 planning tree。类似 AlphaGo 的 MCTS。模型自己 imagine 几个 latent action 序列，用 World Model 看看哪个能达成目标状态，选最优的执行。这就把 LLM 的 System 2 thinking 带到了 Robotics 里。

**6.3 Video as In-Context Learning**
既然 latent action model 本质是 video tokenizer，未来就可以这样玩：给模型看一段人类操作新任务的 Youtube 视频，用 latent action model tokenize 成一系列 action tokens，直接塞到 VLM 的 prompt context 里。Policy 读着这些 in-context tokens，就能 zero-shot 执行这个完全没见过的新任务，不需要任何 fine-tune。这就是 Robotics 版的 few-shot learning。

### 7. 关键 Reference 深度链接

为了把这个领域看透，这几篇是必读的：
- **UniVLA 官方代码库**: https://github.com/OpenDriveLab/UniVLA
- **LAPA** (最直接的 precursor，证明 unsupervised latent action 可行): https://latentactionpretraining.github.io/
- **OpenVLA** (主要 baseline，看它的 action tokenization 瓶颈): https://openvla.github.io/
- **DINOv2** (为什么 feature space 预测这么 work): https://dinov2.metrai.com
- **VQ-VAE** (离散化 bottleneck 的鼻祖): https://arxiv.org/abs/1711.00937
- **JEPA** (Yann LeCun 的 predictive architecture 思想源头): https://arxiv.org/abs/2301.08243
- **Prismatic VLMs** (架构底座): https://github.com/TRI-ML/prismatic-vlms
- **LIBERO Benchmark** (主战场): https://libero-project.github.io/

总而言之，UniVLA 把 robotics 从“依赖大量 action labels 的监督学习”拉到了“利用全网视频无监督预训练的范式”。把离散的动作语义变成了 LLM 可以直接 predict 的 tokens。它给未来的 robot foundation model 指了一条极其清晰且极具扩展性的路。

---

# UniVLA 深度技术讲解

## 1. Paper 的核心 Thesis

UniVLA 解决的是一个根本性的 scaling 问题：现有的 VLA models (RT-2, OpenVLA) 需要大量 action-labeled 数据来训练，而 internet 上的视频（人类活动、各种 robot demo）几乎都没有 action labels，而且不同 embodiment 的 action space 完全 heterogeneous (Franka 7-DoF vs WidowX vs human hand vs navigation agent)。UniVLA 的 key insight 是：**从 visual dynamics 中无监督地提取一个 task-centric 的离散 latent action space，作为 cross-embodiment 的 universal interface**，让 policy 在这个 unified action space 里 planning，然后通过轻量级 decoder 翻译到具体 robot 的 action space。

这个 idea 让 UniVLA 只用 OpenVLA 1/20 的 pretraining compute (960 vs 21,500 A100-hours) 和 1/10 的 downstream data 就能在 LIBERO 上 +18.7%。

GitHub: https://github.com/OpenDriveLab/UniVLA

---

## 2. 为什么需要 Latent Action：问题动机

### 2.1 现有 VLA 的 bottleneck

OpenVLA / RT-2 把 action 当作 LLM vocabulary 中的 token，直接在 low-level action space 上 planning。具体做法是把每个 action 维度均匀 discretize 到 256 个 bins，7-DoF action 就是 $256^7 \approx 7.2 \times 10^{16}$ 的 vocabulary。这有几个严重问题：

1. **Action space 太大**：$256^7$ 的 space 让 next-token prediction 极度 sample inefficient。
2. **Cross-embodiment 不可迁移**：Franka 的 7-DoF end-effector 和 WidowX 的 action dimension 不同，vocabulary 不对齐。
3. **依赖 action labels**：无法利用 Ego4D 这种 internet-scale 无标注视频。

### 2.2 Latent Action 的 promise 和 pitfall

Genie (https://arxiv.org/abs/2402.15391), LAPA (https://latentactionpretraining.github.io/), IGOR (https://arxiv.org/abs/2411.00785) 这些工作已经证明可以从视频中学 latent actions：用一个 Inverse Dynamics Model (IDM) 从 $(o_t, o_{t+k})$ 推 latent action $a_t$，再用 Forward Dynamics Model (FDM) 从 $(o_t, a_t)$ 预测 $o_{t+k}$。

但它们的致命问题是 **naive reconstruction objective 会 capture 所有 visual changes**：相机抖动、背景里其他 agent 的运动、新物体出现、光照变化等等。这些 task-irrelevant dynamics 对 policy 来说是 noise，会 confuse pretraining。Table III 显示，如果用 Genie 风格的 latent action pretrain on Ego4D，LIBERO-Long 只有 69.6% success；如果只用 task-irrelevant latent action，LIBERO-Long 几乎是 0.2%——完全 fail。

UniVLA 的核心创新就是 **decouple task-centric dynamics 从 task-irrelevant visual changes**。

---

## 3. Task-centric Latent Action Learning：两阶段解耦的精髓

这是 paper 最 elegant 的部分。让我详细讲解这个 two-stage 设计背后的 intuition 和 mechanism。

### 3.1 基础架构：IDM-FDM + VQ-VAE

给定一对相隔 k 帧的视频帧 $\{o_t, o_{t+k}\}$，其中 k 根据每个 dataset 的 fps 校准到约 1 秒间隔。

**Encoder (IDM)**：$\mathcal{Z}(a_t | o_t, o_{t+k})$ — 一个 spatial-temporal transformer with causal temporal mask，把 learnable action tokens $a_q \in \mathbb{R}^{N \times d}$ (N=4, d=latent dim) concatenate 到 video features 后面，提取 dynamics。

**Quantization**：$\tilde{a} = \mathbf{VQ}(\hat{a})$ — 用 codebook size $|C|=16$ 的 VQ-VAE (https://arxiv.org/abs/1711.00937) 把连续 latent action 离散化。$16^4 = 65536$ 种组合，远小于 OpenVLA 的 $256^7$。

**Decoder (FDM)**：$\mathcal{F}(o_{t+k} | o_t, a_t)$ — spatial transformer，只接收 quantized action tokens，**不接收历史 frames** (防止模型 memorize dataset 或依赖 context)。

**Loss**：在 DINOv2 feature space 上做 reconstruction：
$$\mathcal{L}_{recon} = \|\hat{O}_{t+k} - O_{t+k}\|^2$$

其中 $O_t, O_{t+k}$ 是 DINOv2 (https://dinov2.metrai.com) 提取的 patch-level features。这是 JEPA (https://arxiv.org/abs/2301.08243) 思想：在 semantic feature space 而非 pixel space 预测，避免 texture/lighting 等高频 noise。

### 3.2 Stage 1: 学 Task-Irrelevant Latent Actions

公式：
$$\begin{cases} 
\text{Encode:} & \hat{a}_{TI} = \mathcal{Z}([O_t; O_{t+k}; a_{TI}; \ell]), \quad \tilde{a}_{TI} = \mathbf{VQ}(\hat{a}_{TI}) \\
\text{Decode:} & \hat{O}_{t+k} = \mathcal{F}([O_t; \tilde{a}_{TI}; \ell])
\end{cases}$$

变量解释：
- $a_{TI}$：task-irrelevant latent action tokens (learnable queries)
- $\ell$：T5 text encoder (https://arxiv.org/abs/1910.10683) 提取的 instruction embedding
- $\hat{a}_{TI}$：encoder 输出的连续 latent action
- $\tilde{a}_{TI}$：VQ 量化后的离散 latent action
- $[;]$：sequence-wise concatenation

**Key insight**：decoder 同时拿到 $\tilde{a}_{TI}$ 和 $\ell$。$\ell$ 提供了 high-level task semantics ("pick up the cup")。因为 codebook 容量有限（$|C|=16$），latent action 会被迫只 encode 那些 $\ell$ 不能 capture 的 visual details — 也就是 task-irrelevant 的 dynamics：新物体出现、其他 agent 运动、相机抖动。

这是 **Information Bottleneck (https://arxiv.org/abs/1612.00410)** 的巧妙应用：用 language 作为 task prior 吸收掉 task-relevant information，剩下的 latent action 自然就是 task-irrelevant 的。

### 3.3 Stage 2: 学 Task-Centric Latent Actions

公式：
$$\begin{cases} 
\text{Encode:} & \{\hat{a}_{TI}, \hat{a}_{TC}\} = \mathcal{I}([O_t; O_{t+k}; a_{TI}; a_{TC}]) \\
& \tilde{a}_{TI} = \mathbf{VQ}(\hat{a}_{TI}), \quad \tilde{a}_{TC} = \mathbf{VQ}_{TC}(\hat{a}_{TC}) \\
\text{Decode:} & \hat{O}_{t+k} = \mathcal{F}([O_t; \tilde{a}_{TI}; \tilde{a}_{TC}])
\end{cases}$$

变量：
- $a_{TC}$：新初始化的 task-centric latent action tokens
- $\mathbf{VQ}_{TC}$：新初始化的 codebook（Stage 1 的 VQ frozen）
- $\hat{a}_{TC}$：encoder 输出的连续 task-centric latent action
- $\tilde{a}_{TC}$：量化后的离散 task-centric latent action

**关键设计**：Stage 1 的 codebook $\mathbf{VQ}$ 冻结，所以 $\tilde{a}_{TI}$ 继续解释 task-irrelevant dynamics。新的 $\mathbf{VQ}_{TC}$ 必须解释剩余的 visual changes — 也就是 task-relevant 的 dynamics (object manipulation, goal-directed motion)。

**Intuition**：这类似 residual decomposition。先让一个 module 解释 $o_t \to o_{t+k}$ 中由 task-irrelevant factors 引起的部分，再让另一个 module 解释剩余部分。由于 task-irrelevant 部分已经被 freeze 的 $\mathbf{VQ}$ 吸收，新 codebook 只能学到 task-centric 信息。

Stage 1 没 language 输入给 decoder（去掉 $\ell$），因为现在 $\tilde{a}_{TC}$ 要 capture task-relevant info，而 language 之前是用来"吸走"这部分 info 的。

### 3.4 这个设计的深层 intuition

让我再深挖一下。这个两阶段设计其实是在做 **counterfactual reasoning**：

- **Stage 1 问**："如果我知道任务是 $\ell$，但 latent action 容量很小，我应该 capture 什么？" 答案是 task-irrelevant 的 visual changes，因为 $\ell$ 已经提供了 task 信息。
- **Stage 2 问**："如果 task-irrelevant 已经被解释，剩余的 visual changes 是什么？" 答案是 task-centric dynamics。

这跟 causal representation learning 里的 front-door / back-door adjustment 思想有相通之处。也跟 disentangled representation learning 中的 "sequential disentanglement" 类似。

### 3.5 Quantitative 验证 (Table III)

在 Ego4D 上 pretrain（强调 task-irrelevant noise），然后 fine-tune 到 LIBERO：

| Latent Action | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| Genie (all dynamics) | 89.8 | 92.8 | 77.2 | 69.6 | 82.3 |
| Task-irrelevant only | 68.0 | 90.4 | 67.2 | **0.2** | 56.5 |
| Task-centric (ours) | 91.2 | 94.2 | 90.2 | 79.4 | **88.7** |

Task-irrelevant 在 LIBERO-Long 上几乎 0% 这点很 dramatic — long-horizon 任务必须依赖 task-relevant action representation，没有 task signal 的 latent action 完全不能 plan。Task-centric 比 Genie 高 6.4% 平均，在 Goal 和 Long 上分别高 13% 和 9.8%，这两个 suite 恰恰最依赖 task-relevant reasoning。

---

## 4. Generalist Policy: 在 Latent Action Space 上 Auto-regressive Planning

### 4.1 架构

基于 Prismatic-7B VLM (https://github.com/TRI-ML/prismatic-vlms)：
- Visual encoder: SigLIP (https://arxiv.org/abs/2303.15343) + DINOv2 双 encoder fusion
- LLM backbone: LLaMA-2 7B (https://arxiv.org/abs/2307.09288)
- Vocabulary 扩展: 加入 $|C|=16$ 个 special tokens $\{\text{ACT}_1, \ldots, \text{ACT}_{16}\}$

Latent action 通过 codebook index 直接映射到 vocabulary。这保留了 VLM 原始 architecture 和 training objective，最大化利用 pretrained knowledge。

### 4.2 Training Objective

$$\mathcal{L} = \mathbb{E}_{o_t, l, a_{z,<i}} \left[ -\sum_{i=1}^{N} \log \pi_\phi(\hat{a}_{z,i} = a_{z,i} | o_t, l, a_{z,<i}) \right]$$

变量解释：
- $o_t$：当前 observation
- $l$：language instruction
- $a_{z,<i}$：前 $i-1$ 个已经 predicted 的 latent action tokens (auto-regressive)
- $a_{z,i}$：第 $i$ 个 ground-truth latent action token
- $\hat{a}_{z,i}$：模型预测的 token
- $\pi_\phi$：参数为 $\phi$ 的 policy
- $N=4$：每个 latent action sequence 4 个 tokens

这是标准 next-token prediction，但是 action space 从 OpenVLA 的 $256^7$ 压缩到 $16^4$，信息密度高 12 个数量级。这就是为什么 UniVLA 只需 960 A100-hours 而 OpenVLA 需要 21,500。

### 4.3 一个 subtle 但重要的点

注意 latent action $a_z$ 是 task-centric 的，跨 embodiment 共享。Franka 机器人抓杯子、人类手抓杯子、navigation agent 走到杯子旁边，它们在 visual dynamics 上都对应"approach and reach cup"这个语义，因此可能映射到同一个或相似的 latent action codebook entry。

Fig. 8 和 Fig. A-1 的可视化很好地印证了这点：同一个 latent action 在不同 embodiment 上呈现 semantic-consistent 行为。比如 Group A 都是 "pick up things"，Group C 把 manipulation 的 wrist-view 和 navigation 的 ego-centric movement align 起来。

---

## 5. Post-training: 从 Latent Action 到 Executable Control

### 5.1 Action Decoder 架构

这个设计很 elegant，解决了 multimodal action distribution 的 ambiguity 问题：

$$\begin{cases} 
\text{Visual Embedd.:} & E_v' = \mathcal{A}(Q=q_v, K=V=E_v) \\
\text{Action Embedd.:} & E_a' = \mathcal{A}(Q=q_a + E_v', K=V=E_a)
\end{cases}$$

变量：
- $E_v, E_a$：VLM 最后一层的 visual 和 action embeddings
- $q_v, q_a$：随机初始化的 learnable queries
- $\mathcal{A}$：multi-head attention (https://arxiv.org/abs/1706.03762)
- $E_v'$：visual 信息被 pooled 成单 token
- $E_a'$：用 $q_a + E_v'$ 作为 query 从 latent action embeddings 中 extract information

**为什么 visual as query 有效**：Table IV 显示，加入 visual query 比 "w/o visual" 高 2.2% avg，在 LIBERO-Long 上高 6%。Intuition 是：latent action $a_z$ 是 task-centric 但 embodiment-agnostic 的，它告诉模型"做什么"但不告诉"在哪做、对准哪个 object"。Visual embedding 提供了 context-specific information，让 decoder 知道当前 scene 下应该怎么 execute。这相当于 **latent action 是 plan，visual embedding 是 grounding**。

### 5.2 Action Chunk Prediction

Latent action 设计为 1 秒时间尺度（k 帧间隔校准到 1s）。这个 temporal structure 让 action chunk prediction (https://arxiv.org/abs/2304.13705) 是 natural choice。实践用 chunk size = 12，real-time 10Hz closed-loop inference on RTX 4090。

对比 OpenVLA 的 latency 问题：单 step 0.18s，chunk size 4 要 0.68s，导致 real-world 执行卡顿，average success rate 只有 38.3%。

### 5.3 Lightweight Decoder

整个 action decoder 只有 10.8M 参数，加上 LoRA fine-tuning 总 trainable params 约 123M。这个 compact 设计之所以 work，是因为 **latent action space 已经是 information-rich representation**，decoder 只需要做 embodiment-specific translation，不需要重新 learn task understanding。

### 5.4 History Latent Actions as Chain-of-Thought

这是 paper 里我个人觉得最 elegant 的 trick 之一：

**设计**：把过去一步的 latent action (4 tokens) append 到 instruction 后面作为 prompt input。除了初始 step，每步都 incorporate 前一步的 latent action。

**Intuition**：这跟 LLM 的 Chain-of-Thought (https://arxiv.org/abs/2201.11903) 完全 analogous。LLM 通过显式写出 reasoning steps 来 refine decision，UniVLA 通过显式输入自己过去的 latent actions 来 enable sequential decision refinement。这建立了一个 **feedback loop**：policy 看到自己之前 "decided" 了什么，再决定下一步。

**为什么不用 historical observations**：多帧 visual tokens 会带来巨大 inference latency 和 redundant information。History latent action 只需 4 tokens 就压缩了上一步的 plan，compact 且 informative。

**Ablation (Table V)**：

| Prompt Input | LIBERO Goal | LIBERO Long | R2R |
|---|---|---|---|
| Instruction-only | 95.0 | 88.1 | 30.6 |
| w/ History Action | 95.6 | 92.0 | 47.1 |

R2R +16.5% (从 30.6 到 47.1) 非常显著，navigation 任务比 manipulation 更需要 history (因为 navigation 是 sequential decision)。LIBERO-Long +3.9%，long-horizon 任务受益。Goal 任务只 +0.6%，因为 goal-reach 任务 history 信号弱。

---

## 6. 实验结果深度分析

### 6.1 LIBERO Benchmark (Table I)

LIBERO (https://libero-project.github.io/) 4 个 suite：

- **Spatial**：推理 spatial relations 放 bowl
- **Object**：同 layout 不同 object 实例
- **Goal**：同 objects 不同任务目标
- **Long**：long-horizon multi-subgoal

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| LAPA* | 78.3 | 85.7 | 73.5 | 53.7 | 72.4 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| UniVLA (Human only) | 91.2 | 90.1 | 81.8 | 78.6 | 83.5 |
| UniVLA (Bridge only) | 95.2 | 95.4 | 91.9 | 87.5 | 92.5 |
| UniVLA (Full) | 96.5 | 96.8 | 95.6 | 92.0 | **95.2** |

几个值得关注的点：

1. **只用 human videos pretrain 就能 beat OpenVLA**：83.5% vs 76.5%，+7%。OpenVLA 用了 in-domain OpenX data，UniVLA 用 Ego4D human videos 完全没 robot 数据，还能超越，证明 latent action 是真正 embodiment-agnostic 的 representation。

2. **只用 Bridge-V2 就能达到 92.5%**：Bridge-V2 是 OpenX 的一个 subset，远小于 OpenX 全集。UniVLA on Bridge-only 超过 OpenVLA on OpenX，说明 latent action 比 raw action supervision 更 sample efficient。

3. **Long suite 提升最 dramatic**：从 OpenVLA 53.7 到 92.0，+38.3%。Long-horizon 任务最依赖 task-centric reasoning，task-irrelevant latent action (像 LAPA) 在 Long 上只有 53.7，差距明显。

### 6.2 Navigation: R2R (Fig. 6)

VLN-CE (https://jacobkrantz.github.io/vln-ce/) 的 R2R benchmark：

- UniVLA: 47.1% oracle success rate
- OpenVLA: 17.5%
- NaVid (用全部 history observations): comparable to UniVLA
- LLaVA-Nav: 14.0%

UniVLA 只用单帧 RGB + history latent action，就能 match 用全部 history 的 NaVid。这印证了 **latent action as history is sufficient and efficient**。

Navigation 和 manipulation 共享同一个 latent action space — 这是 cross-task transfer 的强证据。Navigation 的 "move forward" / "turn left" 和 manipulation 的 "reach and grasp" 都被抽象到 task-centric latent action 中。

### 6.3 Real-World (Fig. 5, Table A-V)

四个真实任务覆盖不同 capability axes：

1. **Store screwdriver** (spatial awareness + precise manipulation)：93.3% success
2. **Clean cutting board** (tool-usage + nonprehensile)：100%
3. **Fold towel twice** (deformable)：46.7% (Diffusion Policy 53.3% 更高)
4. **Stack tower of hanoi** (semantic understanding)：86.7% (DP 只有 6.7%)

UniVLA avg 81.7% vs LAPA 45% vs OpenVLA 38.3% vs DP 33.3%。

**关键 trade-off**：Diffusion Policy 在 towel folding 这种 trajectory-fidelity-critical 任务上更好，因为它是 single-task training，能 fit fixed trajectory。UniVLA score (2.47) 高于 DP (2.33)，因为 UniVLA 能 reliably 完成 intermediate stages (edge selection, partial folding) 即使最终失败。这在 real-world dynamic environment 中是 critical advantage。

**Tower of hanoi** 最 dramatic：DP 6.7% 因为 single-task 模型没法做 semantic reasoning (哪个 tower 大中小)；UniVLA 86.7% 因为 generalist + language understanding。

### 6.4 Generalizability (Table II)

三种 perturbation：

| Setting | DP Succ. | OpenVLA | LAPA | UniVLA |
|---|---|---|---|---|
| Lightning Variation | 20.0 | 13.3 | 26.7 | **66.7** |
| Visual Distractor | 26.7 | 20.0 | 6.7 | **53.3** |
| Novel Object | 26.7 | 26.7 | 53.3 | **86.7** |

- **Lighting**：UniVLA 66.7% 领先，DINOv2 features 对 lighting robust
- **Distractor**：UniVLA 53.3%，semantic-reliant methods (LAPA, UniVLA) 掉得稍多，但绝对值仍最高
- **Novel object**：换 screwdriver 为 marker，UniVLA 只掉 6.6% (从 93.3 到 86.7)，证明 task-centric latent action 学到的是 abstract action semantics 不是 object-specific trajectories

### 6.5 Data Scalability (Fig. 9)

- Bridge only → +OpenX → +Ego4D：LIBERO 上 2.0% avg 提升
- Real-world：Bridge → OpenX +0.3 score → +Ego4D 再 +0.28 score
- R2R 类似趋势

Human videos (无 action labels, embodiment gap 大) 仍能带来持续提升，证明 UniVLA 的核心 thesis：**internet-scale unlabeled videos 是可利用的资源**。

### 6.6 Data Efficiency (Fig. 10)

- 10% data on LIBERO-Goal: UniVLA 86.3% > OpenVLA 79.2% (full data)
- 10% LIBERO-Goal 已经超过 OpenVLA full data
- 50% data on LIBERO-Long 创新 SOTA

这说明 latent action pretraining 让 policy 学到了 transferable task understanding，downstream 只需极少 data 就能 adapt。

### 6.7 Ablation: Decoder Design (Table IV)

| Decoder | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| Auto-regressive | 85.2 | 81.2 | 79.0 | 49.0 | 73.6 |
| Ours w/o Visual | 95.0 | 95.4 | 93.7 | 86.0 | 92.5 |
| Ours (w/ visual) | 96.5 | 96.8 | 95.6 | 92.0 | **95.2** |

- Auto-regressive (像 OpenVLA/LAPA) 比 attention pooling 差 21.6% — 这是巨大差距
- LIBERO-Long 上 auto-regressive 只有 49.0% vs 92.0%，差 43% — long-horizon 任务最受益于 visual-conditioned decoding

**Intuition**：auto-regressive decoder 在 multimodal action distribution 上有 ambiguity 问题。比如 "pick up cup" 可能有多个 valid trajectories，auto-regressive 一次只能 sample 一个，容易 sample 错。Attention pooling 用 visual context as query，相当于 condition on 当前 scene，减少了 distribution 的 multimodality。

---

## 7. 我的 Intuition Building 和延伸思考

### 7.1 Latent Action 作为 "Visual Tokenizer for Robot Control"

让我把这个 idea 推到极致：**latent action model 本质上是 video tokenizer**，把连续 visual dynamics 离散化为 token sequences。这跟 LLM 中的 text tokenizer 完全 analogous。

Paper 在 Future Work 里提到这个方向：用 latent action 把 human demonstration video encode 成 in-context samples，实现 zero-shot skill acquisition。这跟 LLM in-context learning 是同一个 mechanism — 给几个 (video, latent action) 例子，policy 就能 generalize 到新 task。

这个方向如果 work，将彻底改变 robot learning paradigm：不需要 fine-tune，只要给几个 demo 视频就能执行新任务。

### 7.2 与 World Model 的 connection

Latent action model 的 decoder $\mathcal{F}$ 本质是个 world model：给定当前 observation 和 latent action，predict 下一个 observation。Paper 提到可以和 reinforcement learning + test-time scaling (planning trees) 结合。

这跟 Yann LeCun 的 JEPA world model (https://arxiv.org/abs/2301.08243) 思想完全一致：在 latent space 做 predictive modeling，避免 pixel-space prediction 的 sample inefficiency。

潜在应用：MCTS-style planning in latent action space — policy 可以 imagine 多个 latent action sequences，用 world model evaluate，选最优 plan。这就是 AlphaGo (https://deepmind.com/research/breakthroughs/alphago) 的思想应用到 robot control。

### 7.3 与 LLM Cross-lingual Transfer 的类比

Paper 开头提到 LLM cross-lingual transfer (https://arxiv.org/abs/1911.02116)：BERT 在多语言上 pretrain 后，能 zero-shot transfer 到没见过的语言。UniVLA 想做的是 cross-embodiment transfer：在多种 embodiment (robot, human, navigation) 上 pretrain，能 transfer 到新 embodiment。

这个类比很 powerful：语言有 shared semantic structure (universal grammar, common concepts)，所以 cross-lingual 能 work。Robot actions 也有 shared structure — task-centric dynamics。UniVLA 通过 language + DINOv2 features 显式提取这个 shared structure。

### 7.4 Information Bottleneck 视角

Stage 1 设计本质是 Deep Variational Information Bottleneck (https://arxiv.org/abs/1612.00410)：

$$\min I(\tilde{a}_{TI}; O_{t+k} | O_t, \ell) \text{ s.t. } I(\tilde{a}_{TI}; O_{t+k} | O_t) \geq \text{threshold}$$

给定 $\ell$ 后，$\tilde{a}_{TI}$ 应该 minimal information about $O_{t+k}$；但没有 $\ell$ 时应该有 sufficient information。这迫使 $\tilde{a}_{TI}$ 只 encode $\ell$ 不能提供的 information，即 task-irrelevant dynamics。

VQ-VAE 的 discrete bottleneck 天然实现这个：codebook size $|C|=16$ 是个 hard capacity constraint。

### 7.5 与 VQ-BeT / Quest 的对比

VQ-BeT (https://diffusion-policy.github.io/) 和 Quest (https://arxiv.org/abs/2411.17000) 也用 VQ-VAE 学 action representation，但它们在 raw action trajectories 上学，需要 action labels。UniVLA 在 visual dynamics 上学，不需要 action labels。这是本质区别 — UniVLA unlock internet-scale unlabeled videos。

### 7.6 与 Im2Flow2Act / SPOT / FlowBot 的对比

Flow-based methods (https://arxiv.org/abs/2407.02691) 用 optical flow 作为 cross-embodiment representation。UniVLA 用 learned discrete latent action。区别：

- Flow 是 continuous, dense, object-centric
- Latent action 是 discrete, compact, action-centric
- Flow 需要 dense correspondences，latent action 只需 task-relevant dynamics

Latent action 更适合 LLM-based policy，因为可以 tokenize 进 vocabulary。Flow 不能直接 tokenize。

### 7.7 DINOv2 的 critical role

Paper 用 DINOv2 features 作为 prediction target，这是个 subtle 但 critical 的选择。

DINOv2 (https://arxiv.org/abs/2304.07193) 是 self-supervised vision transformer，学到的 features 有 object-centric 和 spatially aware properties。这避免了：

1. Pixel-space prediction 的 high-frequency noise (texture, lighting)
2. CLIP features 的 semantic bias (CLIP 学的是 image-text alignment，可能丢掉 spatial details)

DINOv2 + language conditioning 的组合：DINOv2 提供 spatial/object-centric visual grounding，language 提供 task semantics，latent action 补 task-irrelevant visual dynamics。三者共同构成完整的 visual dynamics decomposition。

### 7.8 1-second Time Scale 的意义

Latent action 校准到 1 秒间隔。这个选择很 deliberate：

- 短于 1 秒：动作太 atomic，policy 需要高频 planning，失去 high-level reasoning
- 长于 1 秒：单个 latent action 跨多个 sub-actions，policy 灵活度不足

1 秒对应人类一个 "action primitive" 的典型 duration (reach, grasp, place)，也对应 chunk size 12 在 10Hz control frequency 下的 1.2 秒 — 完美 align。

### 7.9 Limitations 的诚实评估

Paper 自己提到的 limitations：

1. **Fixed codebook size**：$|C|=16$ 可能对 dexterous hands 不够
2. **Single-arm focus**：没 evaluate 双臂 / 人形
3. **Language granularity**：task-irrelevant 假设 ego-agent movements critical，对 "boiling water with steam" 这种 task 可能不准确
4. **In-context learning 未实现**：future work

我觉得还有一个 limitation paper 没明说：**Latent action 是 task-irrelevant 还是 task-centric 取决于 language 的 richness**。如果 instruction 太 vague ("do something"), Stage 1 的解耦会变 weak。这是 dependency on language annotation quality 的隐含 limitation。

---

## 8. 跟 Sora / Video Generation Models 的潜在 connection

让我做一个大胆的联想。Sora (https://openai.com/sora) 等 video diffusion models 也在学 visual dynamics，但是 generate full pixels。如果用 Sora 作为 world model backbone，latent action 作为 condition，是否能实现更强大的 planning？

UniVLA 的 DINOv2 + VQ 思想可以直接 apply 到 Sora：用 Sora 的 internal features (类似 DINOv2 patch features) 作为 prediction target，加 latent action conditioning，可能实现 controllable video generation + robot policy 的 unified framework。

这也呼应 paper 里提到的 "Integration with world model" future direction。

---

## 9. 关键 References 深度链接

让我列出一些关键的 reference papers 供深入阅读：

- **UniVLA 主页**: https://github.com/OpenDriveLab/UniVLA
- **OpenVLA** (baseline): https://openvla.github.io/ / https://arxiv.org/abs/2406.09246
- **LAPA** (direct predecessor): https://latentactionpretraining.github.io/ / https://arxiv.org/abs/2410.11758
- **Genie** (latent action from videos): https://arxiv.org/abs/2402.15391
- **IGOR** (image-goal representations): https://arxiv.org/abs/2411.00785
- **DINOv2**: https://arxiv.org/abs/2304.07193 / https://dinov2.metrai.com
- **Prismatic VLMs**: https://github.com/TRI-ML/prismatic-vlms
- **VQ-VAE**: https://arxiv.org/abs/1711.00937
- **JEPA / V-JEPA**: https://arxiv.org/abs/2301.08243
- **LIBERO benchmark**: https://libero-project.github.io/
- **CALVIN benchmark**: https://calvinrobot.github.io/
- **Open X-Embodiment**: https://robotics-transformer-x.github.io/
- **VLN-CE / R2R**: https://jacobkrantz.github.io/vln-ce/
- **Ego4D**: https://ego4d-data.org/
- **Diffusion Policy**: https://diffusion-policy.github.io/
- **ACT / Action Chunking**: https://tonyzhaozh.github.io/aloha/
- **VQ-BeT**: https://diffusion-policy.github.io/
- **CrossFormer** (cross-embodiment): https://crossformer.github.io/
- **Octo**: https://octo-models.github.io/
- **NaVid** (video VLN): https://arxiv.org/abs/2407.13700

---

## 10. 总结：UniVLA 的核心 contributions 重新梳理

让我把 UniVLA 的 design philosophy 用一句话概括：**用 language 作为 task prior，用 VQ-VAE 作为 information bottleneck，用 DINOv2 作为 semantic prediction target，在 visual dynamics 上学出 cross-embodiment 的 task-centric latent action space**。

这个 design 让 UniVLA 同时实现：

1. **Scalability**: Internet-scale unlabeled videos 可用
2. **Cross-embodiment**: 同一 latent action space 跨 robot / human / navigation
3. **Efficiency**: 1/20 pretraining compute, 1/10 downstream data
4. **Generalizability**: Lighting / distractor / novel object 都 robust
5. **Inference speed**: 10Hz closed-loop on RTX 4090 via action chunks

这种把 information theory (bottleneck), representation learning (DINOv2), large-scale language modeling (VLM), robotics (action decoder) 跨领域 unify 的工作，是当前 generalist robot policy 研究的 elegant 范例。

对 robot learning 的 future 影响：我预期这个 paradigm 会取代直接 action token prediction 成为 VLA 主流。Latent action pretraining 类似 LLM 的 pretraining phase，downstream adaptation 类似 instruction tuning。这条路径的 ceiling 远高于直接 action supervision。

期待看到 UniVLA 后续工作 — particularly in-context learning with latent action tokens 和 world model integration for test-time scaling。这可能是通向 robot foundation model 的关键路径。
