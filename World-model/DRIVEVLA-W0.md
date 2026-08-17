---
source_pdf: DRIVEVLA-W0.pdf
paper_sha256: f5c200f838546b9c62ff35d38c21f788ef2db2adab919d6db58ec180c55e4823
processed_at: '2026-08-03T23:43:08-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DRIVEVLA-W0

## 一句话版本

VLA 模型太大，action label 太少，模型"吃不饱"，所以让它顺便预测下一帧画面，给它足够的 supervision signal，结果发现 scaling law 被放大了。

## 聊聊背景

Karpathy 你肯定有这个感觉：现在 autonomous driving 有两条路线在打架。

一条是 BEV 那帮人，搞 geometric prior，multi-camera + LiDAR，一堆 specialized module 拼起来，工程上 work，但是这套东西很难 leverage internet 上的海量数据。你给它 100 万小时 YouTube 视频，它不知道怎么用。

另一条是 VLA 这帮人，直接拿 7B/8B 的 VLM backbone（Qwen2.5-VL, Emu3 这种），fine-tune 一下让它输出 waypoints。模型大，容量大，理论上能 scale，听起来很性感。

但 reality 有点尴尬。

你想想，一个 7B 的 VLM，输入是 high-res image + language + past trajectory，输出是什么？6 个 waypoints，2Hz，3 秒 future。你用 FAST tokenizer 压一下，可能就 5-6 个 action tokens。

一个 training step，forward pass 处理了几千个 visual tokens + 几百个 language tokens，最后 gradient 只来自 5-6 个 action tokens 的 cross-entropy。这 supervision 密度太低了。

这就是 paper 里说的 **"supervision deficit"**。7B 模型的 capacity 大部分在 pretraining 后就 frozen 了，fine-tune 阶段根本喂不饱它。

更扎心的是，paper 里观察到：在小数据上，大 VLA 有时还不如小的 BEV model（TransFuser-50M）。因为小 model capacity 小，sparse supervision 刚好够用；大 model capacity 大，sparse supervision 反而让它 overfit。

## 核心 idea

作者的 idea 很直接：既然 action supervision 太 sparse，那就再加一个 dense 的 supervision task。

什么 task 最 dense？预测下一帧画面。

一帧 image 经过 VQ 大概有 1000+ 个 visual tokens，每个 token 都是一个 next-token prediction loss。这 supervision 密度比 action 高了 2-3 个数量级。

而且这个 task 不是随便选的。要预测下一帧，模型必须理解：
- 当前 scene 有什么 object
- ego vehicle 在做什么 action
- action 会让世界怎么变化
- 其他 dynamic object 会怎么移动

这就是所谓的 **world model**。模型学到的不只是 "看到这个画面，输出这个 action"，而是 "理解这个世界的 dynamics"。

## 两种实现方式

这里有个 engineering 细节。VLA 有两种主流 paradigm，处理 image 的方式不同，所以 world model 也得不同。

### VQ 版本

Emu3 这种 VLA，image 先被 VQGAN 离散化成 visual tokens，跟 text token 一样拼进 sequence。这种情况下，world model 就是标准的 next-token prediction：

给定 $S_{<V_t}$（之前的 language + vision + action），预测当前帧 $V_t$ 的 visual tokens。

这是最自然的 extension，跟 LLM 训练一模一样，只不过 token 类型多了 vision。

总 loss：action loss + $\alpha$ × world model loss。

### ViT 版本

Qwen2.5-VL 这种 VLA，image 是 continuous ViT feature，没有 visual vocabulary。你不能 next-token predict 一个 continuous vector。

所以作者引入 latent diffusion。给定当前帧的 vision feature $F_t^V$ 和 action feature $F_t^A$，训练一个 diffusion model 去 denoise 下一帧 image 的 latent。

关键：是预测下一帧 $I_{t+1}$，不是当前帧 $I_t$。因为 condition 是当前帧 feature，如果 reconstruct 当前帧就退化成 autoencoder 了，学不到 dynamics。预测下一帧强制模型 extrapolate。

总 loss：action loss + $\beta$ × diffusion loss。

## 推理时不用 world model

这里有个很 practical 的设计：训练时 world model 给 dense supervision，推理时直接 bypass 掉，只跑 action head。

为什么？因为生成下一帧 image 很慢（AR 要生成 1000+ tokens，diffusion 要 10-50 步 denoise），real-time driving 不能 afford。

所以 world model 本质上是 **训练时的 auxiliary objective**，目的是让 VLA backbone 学到更好的 representation，推理时只享受这个 representation 的 benefit，不付 generation 的 cost。

这个设计很聪明。跟 MAE 有点像：训练时 mask + reconstruct 很贵，推理时用 encoder feature，不 reconstruct。

## Action Expert：解决 latency

VLA backbone 7-8B，inference 慢（100-240ms）。Real-time driving 要 <100ms。

作者引入一个 500M 的轻量 Action Expert，跟主 VLA 用 Mixture-of-Experts 架构配合。关键是 **Joint Attention**：

两个 expert 的 Q/K/V 拼在一起做一次 attention，然后 split 回去。这样小 expert 能从大 VLA 那里"偷" representation，自己只专注 action generation。

效果：latency 从 117ms 降到 74ms（63%），PDMS 还涨了（85.6 → 88.4）。

这个设计让我想到 DeepSeek-V3 的 MoE，也是用 joint attention 让 expert 之间通信。

## 三个 Action Decoder 的对比

作者用 MoE 框架当 testbed，对比三种 action 生成方式：

1. **Query-based**：一组 learnable queries 跟 VLA context 做 attention，MLP 直接回归 waypoints。最快，单 forward pass。

2. **Autoregressive**：FAST tokenizer 把 trajectory 编码成 tokens，next-token predict。跟 LLM 一样。

3. **Flow Matching**：π0 那套，学一个 vector field 从 noise 到 action，推理时 ODE solve。10 步 denoise。

小数据（NAVSIM 100k frames）上排名：Query > Flow > AR。

大数据（in-house 70M frames）上排名反转：AR > Flow > Query。

这是 paper 最 intriguing 的发现，叫 **scaling law reversal**。

为什么？小数据上 trajectory distribution 简单，continuous decoder（query, flow）的 precision 优势显现，AR 有 quantization loss。大数据上 trajectory distribution 极度复杂，AR 的 modeling capacity + teacher-forced training efficiency 优势显现，flow matching sample-inefficient 收敛慢，query-based 有 representational bottleneck。

这跟 GPT 在 NLP 上的胜利完全一致：简单 model + 大数据 + autoregressive 最终胜出。

## 最重要的实验：Scaling Law 被放大

Table 3 是 paper 的核心证据。

在 in-house dataset 上训，数据从 70k → 700k → 70M frames：

- VLA baseline（只有 action supervision）：70k ADE 2.85 → 70M ADE 1.48，改善 48%，明显 saturation
- VLA-W0（加 world model）：70k ADE 2.75 → 70M ADE 1.06，改善 62%，持续上升

70M frame 这个 scale，VLA-W0 比 baseline ADE 提升 28.8%。这个 gap 在小数据上根本看不到，只有 scale 上去才显现。

这就是 "world model amplifies data scaling law" 的意思：dense supervision 让模型能更好地 utilize 大数据，scaling exponent 变大了。

直觉上：sparse supervision 下，加 data 的 marginal value 递减，因为每个 sample 的 signal 就那么点。Dense supervision 下，每个 sample 能榨出更多信息，所以 data 越多越值。

## Generalization 的证据

还有一个很 striking 的实验（Table 7 / Figure 4）。

在 NuPlan 上 pretrain，NAVSIM 上 fine-tune。两个 dataset 的 visual domain 相似，但 action distribution 不同（NAVSIM 专注 safety-critical long-tail）。

- Baseline VLA：pretrain 让性能 **下降** 9.5%（负迁移）
- VLA-W0：pretrain 让性能 **提升** 6.1%（正迁移）
- TransFuser-7B：pretrain 让性能下降 8.1%

为什么 baseline 负迁移？因为 sparse action supervision 让模型 overfit 到 NuPlan 的 action style，这个 prior 在 NAVSIM 上有害。

为什么 VLA-W0 正迁移？因为 world model 强制学 visual representation，这个 representation 是 action-distribution-agnostic 的，所以迁移性好。

这跟 LLM pretrain 的逻辑完全一样：在 Wikipedia 上 pretrain，fine-tune 到 code，靠的不是学 Wikipedia 的 "article style"，是学 generic language representation。World model 让 VLA 也能学到这种 generic 的 "world representation"。

## NAVSIM Benchmark 结果

Table 1（NAVSIM v1）：

DriveVLA-W0（AR expert）用 **单 front camera** 拿到 PDMS 93.0，击败了用 multi-cam + LiDAR 的 WoTE (88.1)、DiffusionDrive (86.5)、AutoVLA (92.1)。

这说明 dense supervision 学到的 representation 确实强，不需要堆传感器。

Table 2（NAVSIM v2）：

DriveVLA-W0 EPDMS 86.1，击败 DriveSuprem (83.1)、DiffusionDrive (84.5)。

但有个 weakness：Extended Comfort (EC) 只有 58.9，比 baseline 的 87+ 低。可能是 AR decoder 的 quantization 导致 trajectory jitter。这是 paper 的一个 limitation。

## Counterfactual Reasoning

Figure 14 有个很酷的实验。给定一个 "turn right" action（但 GT 是 straight），world model 生成 off-road imagery。

这说明模型不是 memorize trajectory，是真的学了 conditional distribution $p(I_{t+1} | \text{action}, \text{vision})$。它能做 "what if" 的 simulation。

这对 safety-critical driving 很有价值。你可以想象：planning 时用 world model simulate 几个 candidate action 的未来，选最安全的。这跟 Dreamer 的 latent imagination、AlphaGo 的 MCTS 有点像。

但 paper 当前没做这个，inference 时 world model 被 bypass 了。这是 future work 的方向。

## 跟其他工作的关系

跟 **Dreamer** 系列：Dreamer 用 latent world model 做 RL planning，DRIVEVLA-W0 用 pixel-space world model 做 imitation learning 的 auxiliary supervision。目标不同。

跟 **GAIA-1**：GAIA-1 纯做 driving video generation，目标是 data synthesis。DRIVEVLA-W0 的 world model 是为了 representation learning，不是 data generation。

跟 **LAW**：LAW 也用 world model for driving，但是 latent prediction。DRIVEVLA-W0 是 pixel/token-level prediction，supervision 更 dense 更 direct。

跟 **π0 / π0.5**：π0 用 flow matching 做 action generation。DRIVEVLA-W0 的 flow matching expert 完全 follow π0。但 paper 发现大数据下 AR 反超 flow matching，这对 π0 路线是个挑战信号。

跟 **UniVLA / WorldVLA**：这俩也把 action-conditioned AR prediction 引入 VLA。DRIVEVLA-W0 的区别是系统地验证了 scaling law，还对比了 VQ vs ViT、AR vs Diffusion。

跟 **FAST tokenizer**：FAST 把连续 trajectory 编码成 discrete tokens，让 LLM-style AR 训练能用于 action generation。DRIVEVLA-W0 的发现（大数据下 AR 胜出）某种程度上 validate 了 FAST + AR 路线的潜力。

## 我的几个 takeaways

1. **Supervision density 比 model size 更重要**。你堆 7B VLA，没有 dense supervision 喂，capacity 浪费了。这个 lesson 跟 LLM 的发展路径一致：GPT 之所以 work，是因为 next-token prediction 给每个 token 一个 supervision signal。

2. **World model 是 VLA 的 "pretraining objective"**。之前 VLA 的 pretrain 在 VLM 阶段结束，fine-tune 阶段只有 sparse action loss。World model 把 dense self-supervision 延伸到了 fine-tuning 阶段。

3. **Scaling law reversal 是个 general phenomenon**。简单 decoder + 大数据 + AR 训练，这个配方在 NLP 赢了，在 driving 也开始赢了。Flow matching / diffusion 这种 continuous generation 方法可能在 robotics 这种数据 hungry 的领域也会遇到同样问题。

4. **VQ > ViT 的反直觉发现**。VQ-based VLA 在大数据上 ADE 改善 28.8%，ViT-based 只改善 3.7%。可能因为 VQ 把 generation 直接做进 token sequence，supervision 更 direct；ViT 的 diffusion supervision 是 indirect 的（通过 conditioning）。这个值得深挖。

5. **MoE + Joint Attention 是 deploy 大 VLA 的实用 path**。500M Action Expert 通过 joint attention 从 7B VLA 借 representation，latency 砍到 63%，性能还涨。这个设计在其他 real-time robotics 场景也能用。

## 一些 open questions

1. **World model 在 inference time 的价值没发挥**。当前只 train 用，inference bypass。如果能用 world model 做 planning-time lookahead search，可能进一步提升安全性。这跟 Dreamer 的 latent imagination、AlphaGo 的 MCTS 思路一致。

2. **EC 指标偏低**。AR decoder 的 quantization 导致 trajectory jitter，comfort 下降。可能需要 hybrid decoder（AR + residual refinement）。

3. **Dynamic object prediction 不足**（Figure 12）。Complex intersection 的 oncoming vehicle 没预测到。Generative fidelity 在多 object 场景还是 limited。

4. **Closed-loop evaluation 缺失**。NAVSIM 是 non-reactive。真实价值需要在 closed-loop 测试，world model 的 reactive simulation 能力才能体现。

5. **Multi-view world model**。当前只 front camera。工业部署需要 multi-cam + LiDAR，world model 怎么 generate multi-sensor consistent 的 future 是个 challenge。

6. **Long-horizon planning**。6VA context 对应几秒。如果要 plan 30 秒，需要 hierarchical world model。

7. **RL + World Model**。当前用 imitation learning。如果加 RL fine-tuning，world model 可以当 simulated environment，类似 Dreamer 的思路。但 driving 的 RL 很难，reward 设计 + safety constraint 都是坑。

## 最后

这篇 paper 的核心 insight 其实很朴素，跟 LLM scaling 的 lesson 一致：**dense self-supervision + autoregressive + 大数据** 是 generalization 的配方。

Autonomous driving 正在从 specialized perception-planning-control pipeline 转向 "VLM + world model self-supervision" 的 general intelligence paradigm。这个趋势跟你一直讲的 "software 2.0" / "software 3.0" 方向完全吻合。

只是 driving 比 NLP 难的地方在于：数据采集贵、safety 要求高、real-time latency 约束强。但方向是对的。

参考链接：
- Paper: https://github.com/BraveGroup/DriveVLA-W0
- NAVSIM: https://arxiv.org/abs/2406.15349
- Emu3: https://arxiv.org/abs/2409.18869
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- π0: https://arxiv.org/abs/2410.24164
- DreamerV3: https://arxiv.org/abs/2301.04104
- GAIA-1: https://arxiv.org/abs/2309.17080
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- OpenVLA: https://arxiv.org/abs/2406.09246
- AutoVLA: https://arxiv.org/abs/2506.13757
- Flow matching: https://arxiv.org/abs/2209.03003

---

# DRIVEVLA-W0 深度讲解：World Models 如何放大 Autonomous Driving 的 Data Scaling Law

## 1. 核心问题的直觉

### 1.1 Supervision Deficit 的本质

这篇 paper 的出发点是一个很深刻的观察，Karpathy 你应该会有共鸣。在 LLM 时代，模型 capacity 越大，需要的 supervision signal 越密集。GPT 训练时每个 token 都是 supervisory signal，一个 4096 token 的 sequence 就有 4096 个 loss signal。但是 VLA 在 driving 场景下，输入是高维的 sensory data（多帧 high-res image + language instruction + past actions），输出只有几个 low-dimensional 的 waypoints（比如 6 个 waypoints，2Hz，3 秒），supervision 信号是极度 sparse 的。

这就造成了 paper 中的核心论点 **"supervision deficit"**：

- VLA backbone 可能是 8B parameters（Emu3）或者 7B（Qwen2.5-VL）
- 每个 training step 的 gradient signal 只来自几个 action tokens 的 cross-entropy
- 大部分 model capacity 在 pretraining 后被 "frozen" 在了一个 representation 上，并没有被 fine-tuning 充分调动
- 结果：大 VLA 在小数据上甚至可能 underperform 小的 specialized BEV model

这个 deficit 不能靠简单堆 action-only data 解决。因为 action supervision 的信息量是 bottlenecked 的，你给的再多 action labels，每个 sample 的 supervision density 还是稀疏的。这跟 LLM 中 "tokens are all you need" 的 dense supervision paradigm 形成鲜明对比。

### 1.2 World Modeling 作为 Dense Self-supervision

作者的 key insight 是把 future frame prediction 作为 dense self-supervised signal。每一帧图像，假设有 N=256×144/16² = ~1440 个 visual tokens（VQ 化后），就有 1440 个 next-token prediction loss signal。这比 action supervision 的信号量多了 2-3 个数量级。

这个思路在 self-supervised learning 里其实有很长的 lineage：
- BERT (https://arxiv.org/abs/1810.04805) 的 masked language modeling
- SimCLR, MoCo 等 contrastive learning
- Video prediction as self-supervision（比如 https://arxiv.org/abs/1904.06803, https://arxiv.org/abs/2106.05263）
- DreamerV3 (https://arxiv.org/abs/2301.04104) 用 latent world model 学 policy
- GAIA-1 (https://arxiv.org/abs/2309.17080) 在 driving 上做 generative world model
- UniVLA (https://arxiv.org/abs/2506.19850) 把 action-conditioned autoregressive prediction 引入 VLA

但是 DRIVEVLA-W0 的独特之处是把 world modeling 当作 **scaling law 的 catalyst**，而不仅仅是 representation learning 的手段。这是 paper 的核心 contribution。

## 2. 方法详解

### 2.1 VLA Baseline 架构

输入序列的形式化定义：

$$S_t = [L_{t-H}, V_{t-H}, A_{t-H-1}, \ldots, L_t, V_t, A_{t-1}]$$

变量解释：
- $L_t$：时间步 $t$ 的 language instruction（用 VLM native tokenizer 处理）
- $V_t$：时间步 $t$ 的 front-view image（VQ 模型用 discrete visual tokens，ViT 模型用 continuous features）
- $A_{t-1}$：时间步 $t-1$ 的 past action（用 FAST tokenizer https://arxiv.org/abs/2501.09747 转成 discrete tokens）
- $H$：history length，paper 里默认 6 (6VA configuration)
- $S_t$：deeply interleaved multimodal sequence，由 causal attention autoregressive 处理

两个 backbone：
1. **VLA (VQ)**：用 Emu3 (8B) (https://arxiv.org/abs/2409.18869)，图像被 VQGAN/MoVQGAN (https://arxiv.org/abs/2202.09036) 离散化成 tokens
2. **VLA (ViT)**：用 Qwen2.5-VL (7B) (https://arxiv.org/abs/2502.13923)，图像是 continuous ViT features

### 2.2 Action Prediction Loss

$$\mathcal{L}_{\mathrm{Action}} = -\sum_{i=1}^{L} \log P(a_i | S_t, a_{<i})$$

变量解释：
- $a_i$：第 $i$ 个 action token（FAST tokenizer 把连续 trajectory 编码成离散 token sequence）
- $S_t$：当前时刻 multimodal context
- $a_{<i}$：前 $i-1$ 个已生成 action tokens
- $L$：action token 序列长度（NAVSIM 平均 5.6 tokens，in-house dataset 平均 17.8 tokens）
- $P(\cdot)$：模型预测的下一个 token 的 softmax 概率

这是标准的 autoregressive cross-entropy，跟 LLM next-token prediction 一模一样，只不过 token 来自 FAST tokenizer 而不是 BPE。

### 2.3 AR World Model（VQ 版本）

这是 paper 中最自然的 extension：既然图像已经被离散化为 visual tokens，那 next-token prediction 就可以无缝扩展到 visual tokens。

$$\mathcal{L}_{\mathrm{WM-AR}} = -\sum_{i=1}^{N} \log P(v_i | S_{<V_t}, v_{<i})$$

变量解释：
- $v_i$：当前图像 $V_t$ 的第 $i$ 个 visual token
- $S_{<V_t}$：$V_t$ 之前的所有 context（language, past vision, past actions）
- $v_{<i}$：当前图像已生成的前 $i-1$ 个 visual tokens
- $N$：当前图像的 visual token sequence length

总 loss：

$$\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{Action}} + \alpha \mathcal{L}_{\mathrm{WM-AR}}$$

- $\alpha$：balancing coefficient（paper 没明确给具体值，但通常是 1.0 量级）

**Intuition**：这个 task 让模型必须学到 "如果我执行 action $A_{t-1}$，世界会变成什么样"（即 $V_t$）。这比单纯预测 action 强制模型学习更多 environment dynamics。值得注意的是，paper 中预测的是当前帧 $V_t$ 而不是未来帧 $V_{t+1}$，因为 VQ-based 模型是把图像当 sequence token 拼在 $S_t$ 末尾，所以 AR 训练时 teacher-forced 直接预测当前 token。这一点跟 Diffusion 版本不同。

### 2.4 Diffusion World Model（ViT 版本）

ViT-based VLA 没有 discrete visual vocabulary，所以不能直接 next-token prediction。作者用 latent diffusion (https://arxiv.org/abs/2112.10752) 来做 future image generation。

$$\mathcal{L}_{\mathrm{WM-Diff}} = \mathbb{E}_{z_{t+1}, \epsilon, k} \left[ \| \epsilon - \hat{\epsilon}(z_{t+1,k}, k, F_t^V, F_t^A) \|^2 \right]$$

变量解释：
- $z_{t+1}$：future image $I_{t+1}$ 经过 VAE encoder 得到的 latent representation（typically $H/8 \times W/8 \times 4$）
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$：标准高斯 noise
- $k$：从 $\{1, \ldots, K\}$ 随机采样的 diffusion timestep
- $z_{t+1,k}$：forward diffusion process 在 timestep $k$ 加 noise 后的 noised latent：
  $$z_{t+1,k} = \sqrt{\bar{\alpha}_k} z_{t+1} + \sqrt{1 - \bar{\alpha}_k} \epsilon$$
  其中 $\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$ 是 cumulative noise schedule
- $\hat{\epsilon}$：denoiser network，输入是 $(z_{t+1,k}, k, F_t^V, F_t^A)$，输出是 predicted noise
- $F_t^V$：VLA backbone 输出的 current vision features
- $F_t^A$：VLA backbone 输出的 current action features（这是关键 conditioning）

总 loss：

$$\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{Action}} + \beta \mathcal{L}_{\mathrm{WM-Diff}}$$

- $\beta$：balancing coefficient

**Intuition**：
1. Diffusion model 学的是 conditional distribution $p(I_{t+1} | F_t^V, F_t^A)$。要让 denoiser 准确预测 noise，模型必须 "理解" vision features 和 action features 隐含的 environment dynamics。
2. 为什么是 future frame $I_{t+1}$ 而不是 current frame $I_t$？因为 condition 是 current frame 的 features $F_t^V$，如果 reconstruct $I_t$ 就退化成 autoencoder，没有预测 dynamic 的 pressure。预测 $I_{t+1}$ 强制模型 extrapolate。
3. Latent diffusion 相比 pixel diffusion 的优势：计算量小（操作在 compressed latent space 上），生成质量高。这是 Stable Diffusion 的核心 trick。

### 2.5 MoE Action Expert 架构

VLA backbone 太大（7-8B），real-time control 不能 afford 几百 ms 延迟。作者引入轻量级 Action Expert（500M）配合主 VLA Expert，用 Mixture-of-Experts 架构，关键是 **Joint Attention** 机制。

$$Q = [Q_{\mathrm{VLA}}; Q_{\mathrm{AE}}], \quad K = [K_{\mathrm{VLA}}; K_{\mathrm{AE}}], \quad V = [V_{\mathrm{VLA}}; V_{\mathrm{AE}}]$$

变量解释：
- $Q_{\mathrm{VLA}}, K_{\mathrm{VLA}}, V_{\mathrm{VLA}} \in \mathbb{R}^{L_{\mathrm{VLA}} \times d}$：VLA Expert 的 query/key/value matrices
- $Q_{\mathrm{AE}}, K_{\mathrm{AE}}, V_{\mathrm{AE}} \in \mathbb{R}^{L_{\mathrm{AE}} \times d}$：Action Expert 的 query/key/value matrices
- $[;]$：沿 token sequence dimension concatenation
- $d$：hidden dimension

Attention 计算：

$$\mathrm{Attn}([Q;K;V]) = \mathrm{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

输出再 split 回 VLA 部分（前 $L_{\mathrm{VLA}}$ 个 token）和 AE 部分（后 $L_{\mathrm{AE}}$ 个 token），分别 router 回各自 expert。

**Intuition**：这种 joint attention 比传统的 cross-attention 更 symmetric。两个 expert 在同一个 attention space 里互相 query/key/value，相当于 co-evolve。这跟 Gemini / DeepSeek 的 MoE 设计哲学类似（https://arxiv.org/abs/2101.03961）。同时 500M Action Expert 用小 hidden dim，参数量小，inference 快。

### 2.6 三种 Action Decoder

作者把 MoE 框架当作 testbed 来对比三种 action decoding：

**Query-based Action Expert**：
- 用一组 learnable action queries（类似 DETR 的 object queries，https://arxiv.org/abs/2010.04159）
- queries 通过 joint attention 与 VLA context 交互
- MLP head 直接回归 continuous waypoint trajectory
- Loss: L1 distance between predicted and GT trajectory
$$\mathcal{L}_{\mathrm{Query}} = \| \hat{A}_t - A_t^{\mathrm{GT}} \|_1$$

**Autoregressive Action Expert**：
- 与 VLA Baseline 完全相同的 next-token prediction + cross-entropy loss
- 用 FAST tokenizer 编码 trajectory
- 推理时 autoregressive 生成 token，然后 detokenize 回 trajectory

**Flow Matching Action Expert**：
- 基于 flow matching（https://arxiv.org/abs/2209.03003，rectified flow 的变体）
- 学习 conditional vector field $v_\phi$，从 noise distribution 到 action distribution
- 训练：定义 noise $\epsilon$ 到 GT action $a$ 的直线路径，loss 是 MSE：

$$\mathcal{L}_{\mathrm{Flow}} = \mathbb{E}_{t, \epsilon, a} \left[ \| v_\phi(x_t, t, c_t) - (a - \epsilon) \|^2 \right]$$

其中 $x_t = (1-t) \epsilon + t a$ 是直线插值，$c_t$ 是 multimodal context。

- 推理：从 noise 出发，用 ODE solver（Euler）走 $T$ 步，把 noise transform 成 action
- 这是 π0 (https://arxiv.org/abs/2410.24164) 和 π0.5 (https://arxiv.org/abs/2504.16054) 用的方法

## 3. Scaling Law 的核心发现

### 3.1 World Modeling 放大 Data Scaling Law

这是 paper 的 central claim。在 Table 3 中：

| Model | 70k ADE | 70M ADE | 改善 |
|-------|---------|---------|------|
| VLA (VQ) Baseline | 2.8520 | 1.4829 | 48% |
| VLA-W0-VQ | 2.7482 | 1.0563 | 61.6% |
| VLA (ViT) Baseline | 3.1524 | 1.1051 | 65% |
| VLA-W0-ViT | 2.5268 | 1.0640 | 57.9% |

关键观察：
- 在 70k frame 上，VLA-W0 比 baseline 略好（提升 ~4-20%）
- 在 70M frame 上，VLA-W0 比 baseline 提升更显著（VQ: 28.8% ADE 改善，ViT: 15.9% collision rate 改善）
- Baseline 在大数据上表现出 saturation 趋势，而 world model 持续提升

**Intuition**：这符合你之前在 neural scaling laws 上的直觉（https://arxiv.org/abs/2001.08361）。当 supervision signal 是 dense 的，模型 capacity 能被有效利用，scaling 效果是 near power-law 的。当 supervision 是 sparse 的，加再多 data 也 saturated，因为 gradient signal 本身不够 informative。

### 3.2 Generalization Across Action Distributions

Table 7 / Figure 4 是一个非常 striking 的实验。NuPlan pretrain -> NAVSIM fine-tune，visual domain 相似，但 action distribution 不同（NAVSIM 专注于 safety-critical long-tail maneuvers）。

- Baseline（VLA-VQ）：pretrain 让 PDMS 从 68.7 跌到 62.2（**-9.5% 负迁移**）
- VLA-W0-VQ：pretrain 让 PDMS 从 80.7 升到 85.6（**+6.1% 正迁移**）
- TransFuser-7B：pretrain 让 PDMS 从 77.9 跌到 71.6（**-8.1% 负迁移**）

**Intuition**：sparse action supervision 会让模型 overfit 到 pretrain dataset 的 action distribution（比如 NuPlan 的 trajectory style），fine-tune 时这个 prior 反而有害。World modeling 强制模型学 visual representation，这个 representation 是 action-distribution-agnostic 的，所以正迁移。

这跟 LLM 的 phenomenon 类似：在 Wikipedia 上 pretrain 的 LLM，fine-tune 到 code generation，靠的不是学 Wikipedia 的 "article style"，而是学到 generic language representation。

### 3.3 Action Decoder 的 Scaling Law Reversal

Table 4 是这篇 paper 最 intriguing 的发现：

**NAVSIM (103k frames, 小数据)**：
- Query-based: PDMS 88.4
- Flow Matching: PDMS 87.2 (-1.4%)
- Autoregressive: PDMS 85.3 (-3.6%)

**In-house Dataset (70M frames, 大数据)**：
- Query-based: ADE 1.1248, Collision 0.0453
- Flow Matching: ADE 1.0362 (+7.9%), Collision 0.0398 (+12.1%)
- Autoregressive: ADE 1.0069 (+10.5%), Collision 0.0295 (+34.9%)

**Intuition**：
- 小数据上，trajectory distribution 简单（NAVSIM 主要 highway 和 intersection 场景），continuous decoder（query-based, flow matching）的 **precision** 优势显现，因为不需要 quantization。Flow matching 还能 model multi-modal distribution。
- 大数据上，trajectory distribution 变得极度复杂（corner cases, 各种 road geometry）。这时候 **modeling capacity** 变成 dominant 因素。Autoregressive decoder 借鉴 LLM 的成功，能用 teacher-forced training 高效 sample-efficient 地学复杂 distribution。Flow matching 因为要 sample 整条 trajectory 上的 noise-to-action path，在大数据上 converge 慢。Query-based 有 representational bottleneck（固定数量 queries）。

这跟 GPT 在 NLP 上的胜利很类似：简单 model + 大数据 + autoregressive 的组合最终胜出，因为容量 scaling 和数据 scaling 都能同时受益。

### 3.4 6VA vs 6V vs 2VA 的 Ablation

Table 5:
- 1 (no pretrain) -> 2VA fine-tune: PDMS 80.7
- 6V pretrain -> 2VA fine-tune: PDMS 84.1 (+3.4)
- 6VA pretrain -> 2VA fine-tune: PDMS 85.6 (+1.5 over 6V)

**Intuition**：纯 vision pretrain（6V）有用，但是把 action 也拼进 sequence（6VA）更好。因为 6VA 强制模型学 vision-action 因果关系：给定 action A，predict 下一帧 visual state。这是 causal dynamics learning 而不是 generic visual feature learning。

Table 6 sequence length ablation:
- VA: 83.3
- 2VA: 84.2
- 6VA: 85.6

更长的 temporal context -> 更好的 dynamics modeling -> 更好的 representation。这跟 LLM 中 context length scaling 的 trend 一致。

### 3.5 Temporal Interval of World Model

Table 9:
- VA (only current frame): 82.9
- 2VA with 4s interval: 84.3
- 2VA with 1s interval: 85.6

**Intuition**：1s 间隔最佳。VA 缺乏 temporal context，4s 间隔两帧 scene variation 太大，预测任务太难。

## 4. NAVSIM Benchmark 性能对比

### Table 1（NAVSIM v1）

| Method | Sensors | PDMS |
|--------|---------|------|
| UniAD | 6x Cam | 83.4 |
| TransFuser | 3x Cam + L | 84.0 |
| LAW | 1x Cam | 84.0 |
| DiffusionDrive | 3x Cam + L | 86.5 |
| WoTE | 3x Cam + L | 88.1 |
| DriveVLA-W0* (query) | **1x Cam** | 88.4 |
| ReCogDrive | 3x Cam | 89.6 |
| DriveVLA-W0† (query+anchors) | 1x Cam | 90.2 |
| AutoVLA† (best-of-N) | 3x Cam | 92.1 |
| **DriveVLA-W0 (AR)** | **1x Cam** | **93.0** |

SOTA 用单 front camera 打败了 multi-cam + LiDAR 的方案。

### Table 2（NAVSIM v2）

DriveVLA-W0 EPDMS 86.1，击败 DriveSuprem (83.1), ARTEMIS (83.1), DiffusionDrive (84.5)。但注意 EC (Extended Comfort) 只有 58.9，比 baseline 低，这是个 weakness（可能因为 trajectory 平滑性不够）。

## 5. Latency Analysis

Table B.1 文字描述：
- Full VLA backbone: 117.8ms (NAVSIM), 240ms (in-house)
- Query-based Action Expert: 74.3ms（63.1% of baseline）
- Flow Matching Expert: ~145ms
- AR Expert (NAVSIM, 5.6 tokens): 95ms
- AR Expert (in-house, 17.8 tokens): 170ms

**Intuition**：AR latency 正比于 token length L，flow matching 是 constant（10 denoising steps），query-based 是 single forward pass 所以最快。MoE 架构确实实现了 real-time deployment。

## 6. 相关联想与 Broader Context

### 6.1 与 Robotics VLA 的关系

Karpathy 你应该熟悉 π0 (https://arxiv.org/abs/2410.24164) 和 π0.5 (https://arxiv.org/abs/2504.16054)。π0 用 flow matching 做 action generation，π0.5 加入了 open-world generalization。这篇 paper 的 flow matching expert 完全 follow π0 思路。但是 paper 发现大数据下 AR 反超 flow matching，这对 π0 路线是个挑战信号。

类似的还有：
- OpenVLA (https://arxiv.org/abs/2406.09246)
- RDT-1B (https://arxiv.org/abs/2410.07871)
- WorldVLA (https://arxiv.org/abs/2506.21539)

### 6.2 与 World Model Literature 的关系

- Dreamer (https://arxiv.org/abs/1912.01603) / DreamerV3 (https://arxiv.org/abs/2301.04104)：latent world model + RL
- GAIA-1 (https://arxiv.org/abs/2309.17080)：autonomous driving generative world model
- Sora (https://openai.com/sora)：video generation as world simulator
- Genie (https://arxiv.org/abs/2402.15391)：interactive environment world model
- LAW (https://arxiv.org/abs/2406.08481)：latent world model for driving

DRIVEVLA-W0 跟这些不同的点在于：world model 不做 simulation 用，是作为 **dense self-supervised auxiliary objective** 来 pretrain backbone representation。

### 6.3 与 Self-supervised Learning 范式的关系

本质上这个工作呼应了 CV / NLP 中 self-supervised pretraining 的核心思想：
- LLM：next token prediction (GPT, https://arxiv.org/abs/2005.14165)
- Vision：masked image modeling (MAE, https://arxiv.org/abs/2111.06377)，contrastive learning (SimCLR, https://arxiv.org/abs/2002.05709)
- Video：future frame prediction (https://arxiv.org/abs/1904.06803)

但是 VLA 之前的 pretraining 主要是 "pretrain VLM on internet data, fine-tune on actions"，supervision 在 fine-tune 阶段 sparse。DRIVEVLA-W0 是把 self-supervised objective 延伸到了 fine-tuning 阶段。

### 6.4 关于 Chinchilla Scaling Law

Kaplan 2020 (https://arxiv.org/abs/2001.08361) 和 Chinchilla (https://arxiv.org/abs/2203.15556) 都讲 compute-optimal scaling。Chinchilla 的核心 insight 是 model size 和 data size 应该等比 scale。DRIVEVLA-W0 在某种意义上告诉我们：dense supervision 让你训练的 data "更有 value"，所以等量 data 能 better utilize model capacity，相当于提高了 compute efficiency。

### 6.5 关于 FAST Tokenizer

FAST (https://arxiv.org/abs/2501.09747) 是 Karpathy 你应该感兴趣的 recent work，把连续 trajectory 用 frequency decomposition 转成 discrete tokens，让 LLM-style autoregressive 训练可以直接用于 action generation。这跟 π0 用 flow matching 处理 continuous action 是不同的 philosophy。DRIVEVLA-W0 的发现（大数据下 AR 反超 flow matching）某种程度上 validate 了 FAST + AR 路线的潜力。

### 6.6 VQ vs ViT Paradigm 对比

VQ-based VLA（Emu3, https://arxiv.org/abs/2409.18869）的好处：unified token space，AR world model 自然 fit。
ViT-based VLA（Qwen2.5-VL, https://arxiv.org/abs/2502.13923）的好处：保留 continuous visual feature，没有 quantization loss，但要引入 diffusion 来做 world model。

Paper 中 Table 3 显示 VQ 模型在大数据上 ADE 改善 28.8%，ViT 模型改善 3.7%，VQ 反而更好。这跟一般 assumption（continuous feature 更 expressive）相反。可能因为 VQ token 给 VLA backbone 提供了 **generation-style 的 supervision**，而 ViT feature 只给 diffusion 提供了 conditioning，supervision signal 间接。

### 6.7 关于 Counterfactual Reasoning

Figure 14 的 counterfactual experiment 很有意思：给定一个 "turn right" action，world model 生成 off-road imagery，而 GT 是 straight。这证明 model 学到的是 conditional distribution $p(I_{t+1} | \text{action}, \text{vision})$ 而不是简单 memorize trajectory distribution。

这种 counterfactual reasoning 能力对 safety-critical driving 很重要。可以做 "what if I brake / accelerate" 的 simulation，类似 model-based RL 中的 planning。

## 7. 局限性和 Open Questions

从我读到的 paper 内容里能看到几个 limitations：

1. **NAVSIM v2 EC (Extended Comfort) 偏低（58.9）**：相比 baseline 的 87+，DriveVLA-W0 在 trajectory smoothness 上有 deficit。可能是 AR decoder 的 quantization 导致的 jitter，或者 world model supervision 让模型更关注 safety 而非 comfort。

2. **Dynamic Object Prediction 不足**（Figure 12）：world model 在复杂 intersection 未能预测 oncoming vehicles。这说明 generative fidelity 在多 dynamic object 场景下还有提升空间。

3. **Instruction Ambiguity**（Figure 11）：coarse-grained command ("go straight") 在 Y-junction 这种 ambiguous 几何下无法 disambiguate。这是 NAVSIM benchmark 本身的 limitation，但也是 VLA 需要 finer-grained language 的证据。

4. **VQ model 优于 ViT model 的反直觉发现**：paper 没有深入解释，可能因为 VQ 把 generation task 直接做进 token sequence，supervision signal 比 diffusion 的 indirect supervision 更强。但 ViT 保留 continuous feature 在 perception task 上理论上更 fine-grained。这值得进一步研究。

5. **World Model 推理时 bypass**：paper 在 inference 时跳过 visual token generation / diffusion，节省 latency，但也意味着 world model 的 generative 能力没在 inference 时间发挥作用。未来 work 可以用 world model 做 planning 时的 lookahead evaluation。

6. **Closed-loop Evaluation 缺失**：NAVSIM 是 non-reactive benchmark。虽然 in-house dataset 100 个 scenario 算 evaluation，但 closed-loop 测试（https://arxiv.org/abs/2106.15349 提到的）更能体现 world model 的 reactive simulation 价值。

## 8. 对 Karpathy 的 Possible Research Directions

考虑到你对 scaling laws, world models, self-supervised learning, autoregressive models 的持续 interest，这篇 paper 给几个值得深挖的方向：

1. **World model 作为 inference-time planning**：当前 paper 只在 training 时用 world model，inference bypass。可以做类似 AlphaGo 的 lookahead search：用 world model simulate 未来 N 步，evaluate 不同 action 的 outcome。这跟 Dreamer 的 latent imagination 类似。

2. **Hierarchical World Models**：6VA 已经用了 6 step history，但 prediction horizon 只 1 step。可以做 multi-step future prediction，类似 video diffusion（https://arxiv.org/abs/2310.08165 SVD）。

3. **Scaling Law 数学模型**：paper 给的是 empirical observation，没给数学模型。可以参考 Kaplan 2020 的 framework，model world model supervision 对 scaling exponent 的 quantitative 影响。

4. **Token-free Action Generation**：FAST + AR 在大数据上胜出，但 quantization loss 不可忽视。是否有 hybrid 方法（比如 AR + 残差 refinement）能保留 AR 的 capacity 又消除 quantization error？

5. **World Model 与 RLHF/RFT 结合**：paper 用 imitation learning，supervision 是 expert demonstration。如果加 reinforcement fine-tuning（类似 AutoVLA, https://arxiv.org/abs/2506.13757），world model 可以提供 " simulated environment for RL training"。

6. **Multi-view World Models**：当前只 front camera，扩展到 multi-view + LiDAR 是工业部署需要。World model 怎么 handle multi-sensor fusion + generation 是个 challenge。

7. **Long-horizon Planning with World Model**：6VA context 是 6 step，对应几秒。如果要 plan 30 秒 future，需要 hierarchical world model。

## 9. 总结

DRIVEVLA-W0 的核心 contribution 不是发明新 architecture，是把 "world modeling 作为 dense self-supervision" 这个 idea 严谨地在 autonomous driving 大数据场景下 validate 了。它告诉我们：

1. VLA 的大 model capacity 需要 dense supervision 才能被充分利用，sparse action supervision 是 bottleneck。
2. Future frame prediction 提供了 action-conditional 的 dense signal，强制模型学 environment dynamics。
3. World modeling 不只是 representation learning trick，它 **amplify data scaling law**，让大模型的 capacity 能 scale 出来。
4. 在大数据下，简单的 autoregressive decoder 反超复杂的 flow matching / query-based decoder，scaling law reversal 现象值得深入理解。
5. MoE 架构 + joint attention 是 deploy 大 VLA 的实用 path。

这工作的本质 insight 跟 LLM scaling 的核心 lesson 一致：**dense self-supervision + autoregressive + 大数据** 是 generalization 的配方。Autonomous driving 不再是 specialized perception-planning-control pipeline，而是 "VLM + world model self-supervision" 的 general intelligence paradigm。

参考链接汇总：
- Paper code: https://github.com/BraveGroup/DriveVLA-W0
- NAVSIM: https://arxiv.org/abs/2406.15349
- Emu3: https://arxiv.org/abs/2409.18869
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- GAIA-1: https://arxiv.org/abs/2309.17080
- DreamerV3: https://arxiv.org/abs/2301.04104
- AutoVLA: https://arxiv.org/abs/2506.13757
- UniVLA: https://arxiv.org/abs/2506.19850
- WorldVLA: https://arxiv.org/abs/2506.21539
- LAW: https://arxiv.org/abs/2406.08481
- Flow matching: https://arxiv.org/abs/2209.03003
- MoVQGAN: https://arxiv.org/abs/2202.09036
- DETR: https://arxiv.org/abs/2010.04159
- GPT-3: https://arxiv.org/abs/2005.14165
- MAE: https://arxiv.org/abs/2111.06377
- OpenVLA: https://arxiv.org/abs/2406.09246
