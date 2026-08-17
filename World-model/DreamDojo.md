---
source_pdf: DreamDojo.pdf
paper_sha256: c8f02e22c8fbe98e0b50215e10e1f6baf2f589b1987e5c6dc07175bfe6fe8587
processed_at: '2026-08-03T23:21:56-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DreamDojo

## 一句话总结

**DreamDojo 是 NVIDIA 做的一个 robot world model——你给它当前画面 + 一个 action，它给你画出未来几秒钟会发生什么。它最骚的地方是预训练数据全是 human 第一人称视频，根本没 action label，但它学到的物理和操作知识能 zero-shot 迁移到 robot 身上。**

类比一下：你在 YouTube 上刷了几万小时人做饭、修东西、开瓶盖的视频，从来没见过 robot。然后突然给你个 GR-1 humanoid，你居然能想象出"如果这只机械手这样动，杯子会怎样"。这就是 DreamDojo 干的事。

---

## 这事为什么难

做 world model 在 game 和 driving 上已经起飞了，比如 Genie 2 能模拟游戏世界，GAIA 能模拟自动驾驶。但 robot manipulation 一直卡着，三个原因纠缠在一起：

**第一，robot data 太少又太贵。** 你想训个能泛化的 world model，需要见过成千上万种物体、场景、操作。但 robot data 要 teleoperation，一个人一小时也就录几条 trajectory。DROID 这个公认的大数据集也就 350 hours、86 个 skill、564 个 scene。跟互联网 video 比就是九牛一毛。

**第二，互联网 video 没 action label。** 你能爬到几百万小时人切菜的视频，但谁给你标了"第 3 秒左手往左转了 15 度"？没有。如果你就 passive 地让模型预测下一帧，它学到的只是"视频长啥样"，学不到"action 导致了什么后果"。这种 model 给它个 action 它根本不听话。

**第三，expert demo 太干净。** 机器人数据全是成功演示，没有"伸手没抓到""打翻了"这些失败 case。world model 如果没见过失败，就想象不出 counterfactual，做 policy evaluation 就不准。

这三个问题看起来死结：要 data 多就没 label，有 label 的 data 又少又窄。

---

## DreamDojo 的解法：三个 trick

### Trick 1：用 human video 当物理先验，管它 embodiment 不一样

核心 insight 很简单：**物理规律是 embodiment-agnostic 的。** 人手抓杯子，杯子被 lift 起来；GR-1 gripper 抓杯子，杯子也被 lift 起来。重力、摩擦、接触力，这些跟你是肉手还是金属爪没关系。

所以 human video 里学到的"物体怎么响应外力""接触发生什么"这些 knowledge，直接能迁移。

他们攒了 **44,711 hours** 的 human egocentric video，三个来源：
- In-lab 55h（戴 Manus手套录的，能 retarget 到 GR-1）
- EgoDex 829h（Apple Vision Pro 录的 dexterous manipulation，公开数据集）
- DreamDojo-HV 43,827h（众包采集，覆盖 household / industrial / retail / educational / administrative）

对比一下：这比之前最大的 robot world model 训练数据集大 **15 倍 duration、96 倍 skills、2000 倍 scenes**。6,015 个 unique skill，9,869 个 unique scene，43,237 个 unique object。

数据多样性不只是"量"的问题，是"分布"的问题。你见过 6000 种 skill，遇到第 6001 种新 skill 时，物理 intuition 还能拼出来。只见过 86 种，遇到第 87 种就崩了。

参考: https://dreamdojo-world.github.io/

---

### Trick 2：Latent Action VAE——没有 label 就自己从 pixels 里抠出 action

这是整篇 paper 最 elegant 的 trick。

问题：human video 没有 action label，怎么让 world model 知道"这一帧到下一帧之间发生了什么 action"？

他们的做法：train 一个 **VAE**（variational autoencoder），输入两连续 frames，encoder 压出一个 32 维的 latent vector $\hat{a}$，decoder 拿这个 latent + 第一帧去重建第二帧。

$$
\mathcal{L} = \mathbb{E}_{q_\phi(\hat{a}|f^{t:t+1})} \log p_\theta(f^{t+1}|\hat{a}, f^t) - \beta D_{KL}(q_\phi(\hat{a}|f^{t:t+1}) \| p(\hat{a}))
$$

翻译成人话：
- Encoder 看两帧，压出 32 维的"动作摘要"
- Decoder 拿"动作摘要"+第一帧，试着画出第二帧
- 如果"动作摘要"信息不够，decoder 画不出来，reconstruction loss 就大
- KL 项把 latent 拉向标准正态分布，防止它把整帧都存进去作弊
- $\beta = 10^{-6}$，极小，主要靠 reconstruction 逼迫，KL 只做轻微正则

**为什么 32 维这么低？** 这是 information bottleneck。你要是给 4096 维，VAE 就把整个第二帧像素都存进去了，latent 就不是"action"了，是"下一帧的压缩版"。32 维低到只够存"最关键的 motion 信息"，逼它 disentangle 出真正的 action。

**为什么这能跨 embodiment？** 想象一下：人手抓杯子 和 GR-1 gripper 抓杯子。视觉上 hand/gripper 长得完全不一样，但"闭合 → 物体被 lift"这个 temporal delta pattern 是类似的。VAE encoder 学到的是"帧间最关键的变化"，这种变化在 physical level 是共享的。所以两个 embodiment 做"同一种 action"时，latent 会落在同一区域。Paper 里 Figure 3 直接可视化了这个：跨 dataset 检索"latent action 最相似"的 frame pair，发现不同 embodiment 在做同一个动作。

这个 latent action 就成了 **unified proxy**：human video 没 label？用 latent action。Robot video 有真 action label？post-training 时切换就行。互联网 video 无限 scale？latent action 也能抽。

参考 AdaWorld（这个 idea 的先驱）: https://adaworld.github.io/

参考 LAPA: https://arxiv.org/abs/2510.07199

---

### Trick 3：Architecture 两个小改动，让 action 真的能控制画面

Base model 是 Cosmos-Predict2.5，一个 latent video diffusion model。它用 WAN2.2 tokenizer 把 4 个 pixel frames 压成 1 个 latent frame，然后用 DiT blocks + flow matching 做 denoising。

DreamDojo 加了两个看起来简单但效果巨大的改动：

**改动 A：Relative Action。** 不输入绝对 joint pose，而是每个 latent frame（4 timesteps）用起点的 pose 做 rebaseline，输入相对 action。

为什么？绝对 pose 分布太散了——手臂在头顶和手臂在身侧，绝对位置差很远，但"往下移 5cm"这个 relative action 是一致的。relative action 收敛到一个更窄的 manifold，模型学起来容易，泛化也好。这就像做 trajectory normalization。

**改动 B：Chunked Action Injection。** WAN2.2 tokenizer 把 4 frames 压成 1 个 latent。DreamDojo 把对应那 4 个 action concat 成一个 chunk，**只注入到那一个 latent frame 的 timestep embedding 里**，而不是把整个 trajectory broadcast 给所有 latent frames。

为什么这事重要？因为 causality。预测第 3 帧时，第 8 帧的 action 对你毫无信息价值，反而引入 noise。Chunked injection 等于把 causality 硬编码进 architecture：每个 latent frame 只看 causally relevant 的那 4 个 action。

Table 5 的 ablation 显示：chunked injection 单独一项在 counterfactual eval 上把 PSNR 从 19.48 提到 20.78，**+1.3 的 jump**，这是所有改动里最大的 single contribution。Counterfactual 测的就是 OOD action，正好打中 chunked injection 要解决的"未来 action leak"问题。

---

## 还加了一个 Loss：Temporal Consistency

原本的 flow matching loss 监督每帧的绝对 velocity：

$$
\mathcal{L}_{\text{flow}} = \| \mathbf{u}(\mathbf{x}_t, t, \mathbf{c}; \theta) - \mathbf{v}_t \|^2
$$

每帧独立预测 velocity，各帧可能各自对了，但帧间 transition pattern 可能错乱。

DreamDojo 加了：

$$
\mathcal{L}_{\text{temporal}} = \mathbb{E}\left[ \sum_{i=1}^{K-1} \big\| (z^{i+1} - z^i) - (v^{i+1} - v^i) \big\|^2 \right]
$$

翻译：强制模型 match ground-truth 的 **frame-to-frame velocity delta**。

Intuition：一个杯子被 pick up，第一帧 velocity 大（开始动），第二帧 velocity 小（稳定持有）。$(v^{i+1} - v^i)$ 这个差分表达的是"acceleration / state transition"，这才是 action 产生的真实 effect。每帧绝对 velocity 各自蒙对了，不等于 transition pattern 对。

$\lambda = 0.1$，辅助 loss。

---

## Post-Training：小数据 fine-tune 到 target robot

预训练完，model 懂了物理和 latent action 语义。但还没见过 GR-1 的真 action space。

Post-training 做的事：重置 action MLP 第一层，full finetune（not LoRA），用小规模 target robot data。因为 latent action 是 continuous 的，post-training 时能平滑过渡到真 action space，不会像 discrete action 那种硬切换。

Table 2 的结果很说话：

| Method | In-lab Eval PSNR |
|--------|------------------|
| 不预训练直接 fine-tune | 20.576 |
| 预训练但不用 action（passive） | 20.797 |
| **预训练 + latent action** | **20.913** |
| 预训练 + ground-truth retargeted action（ideal 上限） | 20.960 |

Latent action 几乎打平 ideal setting。这意味着 **self-supervised action proxy 能达到接近 ground-truth label 的效果**，所以可以放心 scale 到互联网 video。

---

## Distillation：把 50 步压成 4 步，real-time 跑

Teacher model 的问题：bidirectional attention 固定 horizon，50 denoising steps，2.72 FPS。做 policy evaluation 还行，做 live teleoperation 不行。

他们用 **Self Forcing** 范式做 distillation，把 student 改成 causal attention + 4 denoising steps。

两阶段：

**Warmup**: student 用 teacher forcing（context 来自 teacher），回归 teacher 的 ODE 轨迹。

**Distillation**: student 用 **自己生成的 latents** 当 context。这是关键——training 时就用自己略有偏差的输出当输入，inference 时才不会因为 distribution shift 而 compound error。

更进一步：让 student 生成 $N' > N$ frames（比如生成 49 帧），只在随机一个 13-frame window 上算 loss。模拟更长 rollout，让 student 见过更长的"自己的偏差累积"。

结果（Table 6）：

| | PSNR | FPS |
|---|---|---|
| Teacher | 14.086 | 2.72 |
| Student | 13.146 | **10.81** |

4 倍加速，PSNR 只降 1 点。Causal attention 还带来 streaming 能力 + 多帧 context，能从 occlusion 恢复 object，teacher 单帧 condition 完全做不到。

参考 Self Forcing: https://self-forcing.github.io/

---

## 实验验证的三件事

### 1. 能评估 policy（Figure 5a）

Task: AgiBot fruit packing。训了个 GR00T N1.5 policy，拿不同 checkpoint 在真实世界 rollout，再用 DreamDojo 模拟同样 rollout。

结果：
- **Pearson r = 0.995**：real 和 sim 的 success rate 线性相关极强
- **MMRV = 0.003**：rank consistency 极高

意思就是：**DreamDojo 虽然绝对 success rate 偏高（不擅长想象失败），但 policy 之间的相对排序几乎完美**。这已经足够用来做 policy selection 和 benchmarking。

### 2. 能做 model-based planning（Figure 5b）

Ensemble 5 个 policy checkpoint 生成 action proposals，喂给 DreamDojo 预测未来 video，再一个外部 value model（DINOv2 backbone，预测距 subtask completion 还剩几步）选 best proposal。

结果：
- 高 variance policy group: success rate 比 best single checkpoint **+17%**
- 比起 uniform sampling proposals，**约 2x success rate**

World model 让 policy 能"look ahead"，相当于 inference time 做了个轻量 MPC。

### 3. 能 live teleoperation（Figure 6）

PICO VR controller 捕获 upper-body action，本地 RTX 5090 跑 distilled DreamDojo-2B，real-time 10.81 FPS teleoperation 虚拟 G1，持续 1 分钟不 degrade。

---

## 几个关键 Intuition

**为什么 human video 能迁移到 robot？** 不是"人手长得像 gripper"，是"杯子被 lift 的物理过程跟谁 lift 它无关"。World model 学的是 state transition，不是 appearance。Embodiment gap 在 state transition level 被绕开了。

**为什么 latent action 比 MANO / retargeted action 还能打？** MANO 只描述 hand，描述不了 arm locomotion，heavy occlusion 时还估计不准。Latent action 是从全帧信息里 VAE 压出来的，不只看 hand，还看 object 怎么响应。而且 latent action 天生 continuous + cross-embodiment，迁移时几乎无 friction。

**为什么 chunked injection 对 counterfactual 提升最大？** Counterfactual 测的是 OOD action。如果 future action 信息 leak 进当前 prediction，模型学的是 dataset 里的 spurious correlation（"这个 task 通常这么动"），OOD 时直接崩。Chunked injection 切断 leak，强制 model 学"这个 action 导致什么物理后果"，OOD action 反而更鲁棒。

**为什么 distillation 时 student 必须用自己生成的 context？** Teacher forcing 时 student 见到的 context 是 teacher 的"完美"分布。Inference 时 student 只能拿到自己"略有偏差"的输出。这种 distribution shift 在 long rollout 里会 compound 成灾难。Self Forcing 让 training loop 等于 inference loop，消除了这个 mismatch。$N' > N$ 的 trick 进一步 push 这个对齐。

**为什么 Pearson r=0.995 但 absolute success rate 偏高？** World model 是 generative model，倾向于生成"合理"的未来，不擅长生成"失败"的未来（因为训练数据大多是成功的）。所以绝对值偏高。但相对排序对，因为"policy A 比 policy B 好"这个 ordinal 信号不需要精确的 failure modeling，只需要"policy A 的 action 导致的物理后果更接近成功"。这正适合用 rank correlation 而非 absolute calibration 来评估。

---

## 局限和 Open Questions

**罕见 action 差**：slapping、fast waving 这种 fast & uncommon motion，data 少 + 动作剧烈，效果差。

**不擅长生成失败**：world model 倾向 optimistic，absolute success rate 偏高。要解决可能得在 post-training 里混 failure trajectory，或者用 value model 做 calibration。

**不支持 multi-view**：但 SOTA policy 如 GR00T N1.5 multi-view variant 需要多视角。这是个实际 deployment 的痛点。

**Post-training knowledge retention**：full finetune 可能忘掉预训练知识。LoRA / adapter 方案没试。

**Long horizon**：1 分钟已经是 SOTA，但 model-based planning 可能需要 5-10 分钟。latent drift 怎么防？Self-Forcing++ 的 minute-scale generation 是个方向。

**Sim2Real gap 在 video level**：DreamDojo 是 video-space simulator，不是 physics-space。Policy 在 DreamDojo 上 overfit 会不会 generalization 不到 real？Pearson r=0.995 部分回答了，但还没完全 closed。

---

## 我的判断

这 paper 把 **"internet video is a latent simulator"** 这个 idea 工程化到了 product-level。三个 pillar：

1. **Data**: 44k hours human video，lgest to date，diversity 碾压所有 robot dataset
2. **Method**: Latent action VAE 绕开 label bottleneck + relative action + chunked injection + temporal loss
3. **Efficiency**: Self Forcing distillation 跑到 10.81 FPS real-time

三个应用：policy evaluation（r=0.995）、model-based planning（+17%）、live teleoperation。

比起 Genie 2 那种 game-world focus，DreamDojo 把战场拉到了 contact-rich manipulation，用 latent action 这个 elegant trick 解决了 action label 问题。如果后续 multi-view 和 failure simulation 这两个痛点能突破，这方向基本能 close the loop of sim-to-real for manipulation。

---

参考 links：
- DreamDojo: https://dreamdojo-world.github.io/
- Self Forcing: https://self-forcing.github.io/
- AdaWorld: https://adaworld.github.io/
- V-JEPA 2: https://ai.meta.com/vjepa2/
- Cosmos Predict: https://github.com/nvidia-cosmos/cosmos-predict
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoDex: https://egodex.github.io/
- Genie 3: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- DROID: https://droid-dataset.github.io/
- 1X World Model: https://www.1x.tech/1x-world-model.pdf
- LAPA: https://arxiv.org/abs/2510.07199

---

# DreamDojo 深度讲解

Andrej，这篇 NVIDIA 的工作的核心野心是构建一个 **general-purpose robot world model**，并且通过 **大规模 human video pretraining + latent action as unified proxy** 这条路径绕开了 robot data 不足和 action label 缺失的瓶颈。让我从 motivation → 方法 → 实验 → intuition 几个层面来 build your mental model。

---

## 1. Motivation 的本质

World model 在 driving 和 game 上已经爆发（GAIA-1/2、Genie 2、GameNGen），但 contact-rich dexterous manipulation 一直卡在三个相互纠缠的瓶颈：

1. **Data coverage 瓶颈**：robot data 受限于 hardware variability + teleoperation cost。DROID 这种主流大 robot dataset 只有 350 hours / 86 skills / 564 scenes。
2. **Action label 瓶颈**：互联网 video 巨量但没有 action label，passive prediction 学不到 causality。
3. **Stochasticity 瓶颈**：expert demonstration 数据过于确定，policy evaluation/planning 缺少 counterfactual 推理能力。

DreamDojo 的关键 insight 是 **human video 的物理是跨 embodiment 共享的**，于是用 44k hours 的 human egocentric video 做底层物理先验，再用 **latent action VAE** 自监督地从 pixels 中抽出 action proxy，绕开 label。这是个 "physics 共享 + action 表象统一" 的双重视角。

参考：
- https://dreamdojo-world.github.io/
- AdaWorld (latent action world model 的先驱): https://adaworld.github.io/
- Cosmos-Predict2.5 (base video diffusion): https://github.com/nvidia-cosmos/cosmos-predict

---

## 2. Dataset - DreamDojo-HV

**Dataset 是这篇 paper 最重要的 contribution 之一**。看 Table 1 的对比：

| Dataset | Type | Hours | Skills | Scenes |
|---------|------|-------|--------|--------|
| RT-1 | Robot | 900 | 8 | 2 |
| DROID | Robot | 350 | 86 | 564 |
| AgiBot-World | Robot | 2.9k | 87 | 106 |
| Nymeria | Human | 300 | - | 50 |
| EgoDex | Human | 829 | 194 | 5 |
| **DreamDojo-HV** | **Human** | **43,827** | **6,015** | **9,869** |
| **Mixture (final)** | **Human** | **44,711** | **≥6,015** | **≥9,869** |

**比之前最大 world model 数据集大 15x duration, 96x skills, 2000x scenes**。

数据来源三部分：
1. **In-lab (55h)**: Manus gloves + Vive Ultimate Tracker，precise hand pose → 可 retarget 到 GR-1。用于 validate core designs。
2. **EgoDex (829h)**: Apple Vision Pro 采集，包含 high-precision 3D hand/finger poses。丰富 object variety。参考: https://egodex.github.io/
3. **DreamDojo-HV (43,827h)**: crowdsourcing，覆盖 household / industrial / retail / educational / administrative，96× 更多 skills than DROID。

**Intuition**: 这种 diversity 的意义在于，**world model 学到的"物理"不再是某几个 task 的窄分布物理，而是真正可组合的"操作 primitives + object dynamics"**。这也解释了为什么后续 counterfactual eval 和 novel eval 上能 keep 提升。

---

## 3. Architecture 的核心改进

Base model 是 **Cosmos-Predict2.5**（latent video diffusion + WAN2.2 tokenizer + DiT blocks + flow matching loss）。WAN2.2 tokenizer 的 **temporal compression ratio = 4**，意味着 1 个 video latent 对应 4 个 pixel-space frames。

DreamDojo 在这个 base 上做了两个关键 architectural choice：

### 3.1 Relative Action Transformation

不用 absolute robot joint pose，而是 rebaseline 到每个 latent frame 起点（每 4 timesteps 重置一次）。

**Intuition**: 不同 trajectory 的绝对 pose 分布很发散，但 **相对 action 通常收敛到一个更窄的 manifold**，这对 generalization 和 compositional action 都是 friendly 的。本质上类似于在 trajectory 上做 "running normalization"。

### 3.2 Chunked Action Injection

WAN2.2 tokenizer 把 4 frames 压成 1 个 latent。DreamDojo 把 4 个连续 action $a^{i:i+4}$ concat 成一个 chunk，**注入到对应那个 latent frame 的 timestep embedding**，而不是把整个 trajectory 作为一个 global condition broadcast 给所有 latent frames。

**Intuition**: 由于 contact-rich interaction 严格遵循 causality，**未来 action 对当前 timestep prediction 没有信息价值，反而引入 irrelevant noise**。chunked injection 是给模型一个强先验：每个 latent frame 只看 causally relevant 的那 4 个 action。这点和 GR00T N1 的 action chunking 思路是一脉相承的。

---

## 4. Latent Action Model - 这篇 paper 的关键 trick

没有 action label 怎么做 action-conditioned world model？答案是 **train 一个 VAE 从 frames 对中抽 latent action**。

### 4.1 模型结构

- **700M spatiotemporal Transformer** (24 encoder blocks + 24 decoder blocks)
- Latent action dim = **32**（很低维，强制 bottleneck）
- 训练在 human video + in-house robot video 混合上，400k steps，batch size 256

### 4.2 VAE 训练 loss

$$
\mathcal{L}_{\theta,\phi}^{\text{pred}}(f^{t+1}) = \mathbb{E}_{q_\phi(\hat{a}|f^{t:t+1})} \log p_\theta(f^{t+1}|\hat{a}, f^t) - \beta D_{KL}(q_\phi(\hat{a}|f^{t:t+1}) \| p(\hat{a}))
$$

变量含义：
- $f^{t:t+1}$: 两连续 frames（输入）
- $q_\phi(\hat{a}|f^{t:t+1})$: **encoder**，由 $\phi$ 参数化，输出 latent action 分布 $\hat{a} \in \mathbb{R}^{32}$
- $p_\theta(f^{t+1}|\hat{a}, f^t)$: **decoder**，由 $\theta$ 参数化，从 $\hat{a}$ 和 $f^t$ 重建 $f^{t+1}$
- $p(\hat{a})$: latent action 的 prior（标准 normal）
- $D_{KL}$: KL divergence，正则化 latent space 接近 prior
- $\beta = 10^{-6}$: 极小，让 reconstruction 主导，KL 只做 mild regularization

**Intuition**: 这个 VAE 设计有两个关键性质：
1. **Information bottleneck**：32 维极低 + KL 拉向 prior → 强制 encoder 压出"最 critical"的 motion 信息。
2. **跨 embodiment 一致性**：因为 latent 只表达"动作意图"而不绑定具体 morph，所以 GR-1 的 grip 动作和 human 的 grip 动作会落在 latent space 的同一区域。这点在 Figure 3 中被可视化验证：different datasets 中"做同一种 action"的 frame pair 被检索聚到一起。

参考：
- Latent action 原始 idea (LAPA / V-JEPA 2): https://ai.meta.com/vjepa2/
- AdaWorld (本文的灵感来源): https://adaworld.github.io/

### 4.3 Latent Action 注入到 World Model

- Project latent actions 通过 lightweight MLP → match timestep embedding dim
- MLP 最后一层 **zero-init**（避免训练初期扰动 pretrained state，这是个 trick from ControlNet）
- Projected embedding **加到 timestep embedding 上**，再走 adaptive layer norm (AdaLN) 调制 scale/shift/gate

---

## 5. Training Objective - 重要的细节

Cosmos-Predict2.5 原本用 **flow matching loss**：

$$
\mathcal{L}_{\text{flow}}(\theta) = \mathbb{E}_{\mathbf{x},\epsilon,\mathbf{c},t} \| \mathbf{u}(\mathbf{x}_t, t, \mathbf{c}; \theta) - \mathbf{v}_t \|^2
$$

变量：
- $\mathbf{x}$: clean video latent
- $\epsilon$: Gaussian noise
- $\mathbf{v}_t = \epsilon - \mathbf{x}$: ground-truth velocity field（flow matching 的 target）
- $\mathbf{x}_t$: noise-corrupted latent at diffusion timestep $t$
- $\mathbf{c}$: conditions (text, conditional frames, **actions** for world models)
- $\mathbf{u}(\cdot; \theta)$: denoiser (DiT), 参数为 $\theta$
- $t$: diffusion timestep

DreamDojo 在此基础上加了 **temporal consistency loss**：

$$
\mathcal{L}_{\text{temporal}}(\theta) = \mathbb{E}\left[ \sum_{i=1}^{K-1} \big\| (z^{i+1} - z^i) - (v^{i+1} - v^i) \big\|^2 \right]
$$

变量：
- $K$: video latent 总长度
- $z^i$: predicted velocity at frame $i$（denoiser 输出）
- $v^i$: ground-truth velocity at frame $i$

**这个 loss 的 intuition 非常 elegant**：原本 flow matching 只监督每帧的绝对 velocity，但**帧间 transition（差分）才是真正刻画 object dynamics 的**。强制模型 match ground-truth 的 frame-to-frame velocity delta，相当于在 latent space 加了一个二阶约束，让模型把"动作如何传播"也学进去。

Final loss：

$$
\mathcal{L}_{\text{final}}(\theta) = \mathcal{L}_{\text{flow}}(\theta) + \lambda \mathcal{L}_{\text{temporal}}(\theta), \quad \lambda = 0.1
$$

---

## 6. Post-Training 流程

Post-training 在 target robot data 上：
- 视频采样 ~10 Hz（feasible motion）
- 第一帧作 condition，后续 12 帧训练
- **重置 action MLP 第一层 + full finetune**（注意是 full finetune 不是 LoRA）
- 128 H100 × 50k steps, batch size 512

**Key point**: Latent action 的 **continuous 性质**让 fine-tune 后能平滑过渡到 target robot action space，而不是离散 action 那种硬切换。Table 2 的 PSNR 显示 latent action (20.913) 几乎打平 ideal setting 的 retargeted action (20.960) 和 MANO (20.474)。

---

## 7. Distillation - Self Forcing 范式

为什么要 distill？两个原因：
1. Teacher 用 bidirectional attention，**固定 horizon**，不能 streaming
2. 50 denoising steps → 2.72 FPS，real-time 用不了

Distillation 分两阶段：

### 7.1 Warmup Stage

让 student 用 **teacher forcing**（context 来自 teacher 生成的 latents）回归到 teacher 的 ODE 轨迹：

$$
\mathcal{L}_{\text{warmup}}(G_{\text{teacher}}, G_{\text{student}}) = \mathbb{E}_{x,t} \| G_{\text{student}}(x_t, t) - x_0 \|^2
$$

- $x_0$: teacher 的 ODE 轨迹上某个点
- $G_{\text{student}}, G_{\text{teacher}}$: 分别是 student/teacher 生成器
- 10k ODE 轨迹，10k iterations

### 7.2 Distillation Stage

关键差异：student 用 **自己生成的 latents** 作 context（而不是 teacher forcing），消除 train-test distribution mismatch。

Loss 用 **distribution matching distillation** (Yin et al. 2024)：

$$
\mathcal{L}_{\text{distill}} = D_{\text{KL}}(p_{\text{teacher}} \| p_{\text{student}})
$$

直接算 KL 不 tractable，但 gradient 可算：

$$
\nabla \mathcal{L}_{\text{distill}} = -\mathbb{E}_{z,t} \left[ \big( s_{\text{real}}(x_t, t) - s_{\text{fake}}(x_t, t) \big) \frac{dG_{\text{student}}}{d\theta} \right]
$$

变量：
- $z \sim \mathcal{N}(0, I)$: 标准 noise
- $x_t$: 由 student 从 $z_i$ forward diffusion 得到
- $s_{\text{real}}$: real diffusion score，由 **teacher** 估计
- $s_{\text{fake}}$: fake diffusion score，由一个 **在 student predictions 上训练的 model** 估计
- $\frac{dG_{\text{student}}}{d\theta}$: student 的 Jacobian

**额外 trick**: 让 student 生成 $N' > N$ frames，但只在随机一个 $N$-sized window 上算 loss。这模拟更长 rollouts，进一步减少 compounding error。

### 7.3 Architecture 变化

| Component | Teacher | Student |
|-----------|---------|---------|
| Attention | Bidirectional | **Causal** (sliding window size 12) |
| Denoising steps | 35 (inference) | **4** |
| FPS | 2.72 | **10.81** |

Self Forcing 参考：https://self-forcing.github.io/

---

## 8. Experiments 关键发现

### 8.1 Action Conditioning 消融 (Table 2)

| Method | In-lab PSNR ↑ | EgoDex PSNR ↑ |
|--------|---------------|----------------|
| w/o pretrain | 20.576 | 19.952 |
| action-free pretrain | 20.797 | 19.924 |
| **latent action** | **20.913** | **20.344** |
| retargeted action (ideal) | 20.960 | - |
| MANO (ideal) | - | 20.474 |

**Latent action 几乎打平 ideal setting**！这是这篇 paper 的关键 evidence：**self-supervised action proxy 可以达到 ground-truth label 接近的效果**，意味着可以 scale to internet videos。

### 8.2 Data Mixture 消融 (Table 3)

加入更多 human data 单调提升所有 4 个 benchmark 的 PSNR：
- 仅 In-lab: In-lab Eval PSNR 20.913
- +EgoDex: 20.972 (+0.06)
- +DreamDojo-HV: 21.016 (+0.04)
- DreamDojo-14B (full training): 21.413

**Intuition**: 数据 diversity 越多 → OOD generalization 越强 → counterfactual action 越能想象。

### 8.3 Architecture & Loss 消融 (Table 5)

| Relative | Chunked | Temporal | GR-1 Val PSNR | Counterfactual PSNR |
|----------|---------|----------|----------------|----------------------|
| ✗ | ✗ | ✗ | 16.199 | 19.448 |
| ✓ | ✗ | ✗ | 16.522 | 19.482 |
| ✓ | ✓ | ✗ | 17.626 | 20.783 |
| ✓ | ✓ | ✓ | **17.630** | **20.980** |

**Chunked injection 是最大的 single jump**（counterfactual PSNR +1.3），relative action 也明显。temporal loss 在 counterfactual 上贡献 +0.2。

### 8.4 Distillation 效果 (Table 6)

| Method | GR-1 Long Eval PSNR | FPS | Predict len | Context len |
|--------|----------------------|-----|-------------|-------------|
| Teacher | 14.086 | 2.72 | 12 | 12 |
| Student | 13.146 | **10.81** | - | - |

Student **4x 加速**，PSNR 略降（14.09 → 13.15）但接近。Causal attention 还提供 **streaming 能力 + 多帧 context**，能从 occlusion 恢复 object（Figure 11 显示 teacher 单帧 condition 完全无法处理遮挡）。

### 8.5 Downstream Application 1: Policy Evaluation (Figure 5a)

- Task: AgiBot fruit packing (20 scenes, 5 fruits per scene = 100% success)
- Policy: GR00T N1.5 (state-free, single-view)
- 用不同 checkpoints 的 closed-loop rollout
- 评估指标：
  - **Pearson r = 0.995**：DreamDojo vs real-world success rate 线性相关
  - **MMRV = 0.003**：rank consistency 极高

这是个非常 strong 的 evidence：world model 可以做 reliable simulator for policy evaluation。参考 GR00T N1: https://arxiv.org/abs/2503.14734

### 8.6 Downstream Application 2: Model-Based Planning (Figure 5b)

- Ensemble 5 个 checkpoints → 生成 action proposals with variance
- 把 proposals 喂给 distilled DreamDojo-2B → predict future video
- 用一个外部 **value model**（DINOv2 backbone + global attention，预测距 subtask completion 的 normalized steps）选 best proposal
- 结果：
  - 高 variance policy group: success rate 提升 **+17%** over best single checkpoint
  - 比起 uniform sampling，**约 2x success rate**
  - 低 variance group：小一些，但也 ~2x over uniform

**Intuition**: World model 让 policy 能 "look ahead"，相当于在 inference time 做一个 model-predictive-control (MPC) 的轻量版。

### 8.7 Downstream Application 3: Live Teleoperation (Figure 6)

- PICO VR controller 捕获 upper-body action
- NVIDIA RTX 5090 本地部署
- Real-time 10.81 FPS 实时 teleoperation 虚拟 G1 robot
- 持续 1 分钟以上不 degrade

---

## 9. Limitations

诚实的 limitations：
1. **罕见 action**：slapping、fast waving 效果差
2. **Absolute success rate 偏高**：world model 不擅长 generate nuanced failures，这影响 absolute calibration 但 rank correlation 仍然很高
3. **不支持 multi-view**：但 SOTA policies（如 GR00T N1.5 multi-view variant）需要多视角
4. **Post-training knowledge retention**：未深入研究，可能用 LoRA 等会更好

---

## 10. 关联工作图谱

让我帮你建立一些关联：

**Latent Action 谱系**：
- LAPA (Ye et al. 2025): self-supervised latent action pretraining for VLA → https://arxiv.org/abs/2510.07199
- AdaWorld (Gao et al. 2025): latent action world model，本文直接 build on → https://adaworld.github.io/
- V-JEPA 2 (Assran et al. 2025): meta 的 self-supervised world model → https://ai.meta.com/vjepa2/
- CLAM (Liang et al. 2025): continuous latent action for robot learning

**Human Video for Robotics 谱系**：
- EgoVLA (Yang et al. 2025)
- Being-H0 (Luo et al. 2025)
- EgoZero (Liu et al. 2025)
- DexCap (Wang et al. 2024): https://dexcap.stanford.edu/
- DexWM (Goswami et al. 2025): 直接前作，但 scope 小
- Human-to-Robot transfer (Kareer et al. 2025)

**Video World Model 谱系**：
- GameNGen (Valevski et al. 2025): Doom world model
- Genie 2 (DeepMind): https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- 1X World Model: https://www.1x.tech/1x-world-model.pdf
- IRASim (Zhu et al. 2025)
- WorldPlay: long-term geometric consistency
- Genie 3: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/

**Autoregressive Video Distillation 谱系**：
- Self Forcing (Huang et al. 2025): https://self-forcing.github.io/
- DMD (Distribution Matching Distillation, Yin et al. 2024)
- Self-Forcing++ (Cui et al. 2025): minute-scale generation
- Rolling Forcing (Liu et al. 2025)
- ART-V (Weng et al. 2024)
- MotionStream (Shin et al. 2025)

**Robot Foundation Model 谱系**：
- GR00T N1 (Bjorck et al. 2025): https://arxiv.org/abs/2503.14734
- π0 (Black et al. 2024): https://arxiv.org/abs/2410.24164
- UniVLA (Bu et al. 2025)

---

## 11. 几个值得深挖的 Intuition

**Intuition 1: Latent Action 为什么能跨 embodiment transfer？**
当 VAE encoder 看到 "human hand 抓 cup" 和 "GR-1 gripper 抓 cup"，**视觉上的 critical motion**（grip 闭合 + 物体被 lift）的 temporal delta 是类似的，所以 latent embedding 会落在同一区域。World model 学到的是"latent action → 未来帧"的映射，这种映射在 physical level 是共享的。这就是 paper Section 3.3.2 的核心 insight。

**Intuition 2: Temporal Consistency Loss 为什么有用？**
考虑两个连续帧的 velocity $v^i, v^{i+1}$，他们差分 $(v^{i+1} - v^i)$ 表达的是"**acceleration / state transition**"。一个物体被 pick up，第一帧 velocity 大（开始移动），第二帧 velocity 小（已稳定）。这个差分 pattern 是真正的"action 产生的 effect"。Flow matching 单独监督每帧，模型可能各帧 independently 预测对了，但 transition pattern 错乱。Temporal loss 强制 transition pattern 对。

**Intuition 3: Chunked Injection 为什么对 Counterfactual Eval 提升最大？**
Counterfactual eval 测的是 "out-of-distribution action"（如 patting a toy）。如果用 global action condition，未来 action 信息会 leak 进当前 prediction，模型偷懒学 spurious correlation。Chunked injection 把 causality 硬编码进 architecture，对 OOD action 反而更鲁棒。

**Intuition 4: Distillation 时为什么要 student 用自己生成 context？**
这就是 Self Forcing 的精髓。如果用 teacher forcing 训练，student 看到 context 是 teacher 的"完美"分布，但 inference 时只能看到自己"略有偏差"的 context，**这种 distribution shift 会 compound**。让 student 在 training 时就用自己生成的 context，是把 train loop 和 inference loop 对齐。$N' > N$ 的 trick 是进一步 push 这个对齐，让 student 见过更长的自生成轨迹。

**Intuition 5: 为什么用 Human Video 而不是 Robot Video 做预训练？**
Human video 有三个 robot video 不具备的性质：
1. **Scale**: 互联网 human video 是无限的，robot video 受 hardware/cost 限制
2. **Diversity**: 96x skills, 2000x scenes
3. **Stochasticity**: human 自发做各种探索性 action，包括"不成功"的尝试；robot demo 都是 expert 的，缺 stochasticity

物理本身（gravity, friction, contact dynamics）是 embodiment-agnostic 的，所以预训练得到的"physics prior"在 fine-tune 到 robot 时仍然可用。

---

## 12. 一些 Open Questions / 你可能会想 attack 的点

1. **Latent Action 的语义可解释性**：32 维 latent 真的能 capture 全部 action 语义吗？什么时候会 collapse / disentangle 不够？What Do Latent Action Models Actually Learn (Zhang et al. 2025) 是相关讨论。
2. **Multi-view 扩展**：作者在 limitations 中坦白说不支持 multi-view。一种思路是 **用 multi-view VAE tokenizer** 或者 **latent action 加上 view-conditioning**。
3. **Failure mode simulation**：world model 总是 over-optimistic（absolute success rate 偏高）。一个思路是 **混合失败 trajectory** 进 post-training 数据，或者**用 value model 校准**。
4. **Action space 不连续的情况**：locomotion + manipulation 混合 action space，continuous latent 是否还够用？
5. **更长 horizon**: 1 分钟已是 SOTA，但 model-based planning 可能需要更长 horizon（比如 5-10 分钟 task）。如何 prevent latent drift？Self-Forcing++ 的 minute-scale generation 是一个方向。
6. **Sim2Real gap 在 world model 上**：DreamDojo 给的是 video-space simulator，而不是 physics-space。Policy 在 DreamDojo 上 overfit 会不会 generalize 不到 real world？这个 gap 目前用 Pearson r=0.995 部分回答，但还没完全 closed。

---

## 13. 总结 mental model

DreamDojo 的故事可以浓缩为：

> **"If physics is shared across embodiments, and actions can be self-supervised from pixels, then internet-scale human videos are a free lunch for robot world models."**

三个工程 pillar：
1. **Data pillar**: 44k hours human video (largest to date)
2. **Method pillar**: Continuous latent action VAE as unified proxy + relative action + chunked injection + temporal consistency loss
3. **Efficiency pillar**: Self Forcing distillation → 10.81 FPS real-time, 1+ minute rollout

三个应用 showcase：
1. **Policy evaluation** with Pearson r=0.995 vs real-world
2. **Model-based planning** with +17% success rate via ensemble + value model
3. **Live teleoperation** at real-time FPS

这是一个把 **"internet video is a latent simulator"** 这个 idea 工程化到 product-level 的工作。比起 Genie 2 那种 game-world focus，DreamDojo 把战场拉到了 contact-rich manipulation，并且把 action label bottleneck 用 latent action 这个 elegant trick 解决了。

---

希望这个讲解帮你 build 出 mental model。如果你想要更深入某个 aspect（比如 latent action VAE 的 design space、Self Forcing 的 math、value model 的 training details），告诉我我可以再展开。

参考 web links：
- DreamDojo project page: https://dreamdojo-world.github.io/
- Self Forcing: https://self-forcing.github.io/
- AdaWorld: https://adaworld.github.io/
- V-JEPA 2: https://ai.meta.com/vjepa2/
- Cosmos Predict: https://github.com/nvidia-cosmos/cosmos-predict
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoDex: https://egodex.github.io/
- Genie 3: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- DROID dataset: https://droid-dataset.github.io/
- 1X World Model: https://www.1x.tech/1x-world-model.pdf
- LAPA (latent action pretraining): https://arxiv.org/abs/2510.07199
- DreamGen (related NVIDIA work): https://dreamgen-nvidia.github.io/
