---
source_pdf: Humanoid World Models.pdf
paper_sha256: 31635bc4bc711314d3b77ee94f47f797dafe3a952cf97b1f49dc160daf7e71d5
processed_at: '2026-08-05T07:54:30-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Humanoid World Models

Andrej, 好, 抛开公式, 我用最直白的方式把这篇 paper 的故事讲一遍。

---

## 这篇 paper 到底在干嘛

一句话: **教机器人 "做梦"**。

给机器人看一段过去的视频 + 它打算做的动作, 让它 "脑补" 接下来会看到什么画面。就像你打篮球前在脑子里预演 "如果我往左晃一下, 对手会怎么动"——机器人也需要这种 imagination 能力, 才能在复杂环境里 plan。

这个 "脑补机器" 就叫 world model。

---

## 为什么需要这东西

Humanoid robot 长得像人, 能在人的环境里干活。但光长得像没用, 它得会思考——"我如果伸手抓那个杯子, 杯子会不会掉?"

现在主流方案是 VLA model (vision-language-action), 给图给文字直接输出动作。问题是这种模型 spatial reasoning 很烂, 经常犯低级错误, 在 open-world 里不够 reliable。

World model 提供了另一条路: 你不直接决策, 而是先在脑子里 "simulate" 一遍, 看看不同动作会导致什么结果, 再选最好的。就像下棋先在脑子里推演几步。

---

## 现有的 video generation 模型为什么不够用

Sora、MovieGen 这些模型生成视频很漂亮, 但:

- 它们是为 entertainment 设计的, 不关心 physical plausibility
- 大多 closed source
- 不支持 "给过去视频 + 动作, 预测未来视频" 这种 conditioning
- 太大了, 普通人玩不起

NVIDIA 出了个开源的 Cosmos, 算最接近的, 但最小版本 7B 参数, 要 8 张 H100 训练, 推理要 40GB+ VRAM。作者试了一下, 在 2 张 A6000 上生成 121 帧视频要 1 小时+, 根本没法 fine-tune。

所以核心问题是: **能不能用 2 张 GPU, 搞出一个 humanoid 专用的、能用的 world model?**

---

## 两条技术路线

作者试了两种完全不同的生成范式, 就像对比 "两种做梦方式" 哪种更好。

### 路线 A: Masked Transformer (离散 token + 填空)

把视频压成一堆 discrete tokens (用 VQ-VAE, 类似把视频变成 "视频里的单词")。然后随机盖住一部分 tokens, 让 transformer 猜被盖住的是什么。

这就是 BERT 的思路搬到视频上。你给模型看一段 "被打码" 的未来视频 + 完整的过去视频 + 动作, 让它还原被打码的部分。

推理时把未来视频全部打码, 然后分几步并行地 "猜" + "修正", 像雕刻一样从粗到细。

优点: 训练信号尖锐 (cross-entropy), bidirectional context (每个 token 都能看到周围), 并行解码快。

### 路线 B: Flow Matching (连续 latent + 去噪)

把视频压成 continuous latents (不量化, 保持浮点数)。然后从纯高斯噪声出发, 学一个 "速度场" 把噪声慢慢推到真实视频。

和 diffusion 类似但更简洁——diffusion 是 "随机游走回数据", flow matching 是 "直线滑向数据"。

推理时从噪声开始, 走 50 步 Euler ODE, 到达生成的视频。

优点: 理论优雅, 大模型上 scaling 好。缺点: MSE loss 信号模糊, 容易学出 "平均化" 的模糊结果。

---

## 架构设计的核心探索

这是 paper 最有价值的部分。作者系统探索了 transformer block 怎么设计, 沿着三个轴:

### 轴 1: Attention 怎么做

有 4 路 token 流要处理: 过去视频、未来视频、过去动作、未来动作。

**Joint attention**: 全部拼一起做 global self-attention。交互最充分, 但 memory 随 token 数平方涨, video 序列长就很贵。

**Split attention**: 先各自做 self-attention (自己内部消化), 再让未来视频当 query 去查过去视频 + 动作。省 memory, pixel-level 对齐好, 但 cross-modal 自由交互少。

### 轴 2: 参数要不要共享

4 路 token 流, 每路都独立参数? 还是共享?

**不共享 (Base)**: 每路自己一套 weights。表达力最强, 参数最多。

**按 modality 共享**: 过去视频和未来视频共享一套 (它们本来就是同类), 过去动作和未来动作共享一套。参数少 26%, 几乎无损。

**全共享**: 4 路全用同一套 weights。参数少最多, 最省, 略有质量损失。

直觉: 过去视频和未来视频是 "同一种东西在不同时间", 当然可以共享感官系统。过去动作和未来动作是 "同一个 state vector 在不同时间", 共享也自然。

### 轴 3: 前 4 层 separate, 后面共享

这是 hybrid 方案。前几层学 low-level 特征 (视频的视觉特征 vs 动作的 kinematic 特征确实不同), 保持独立。后几层做 cross-modal 推理 (用动作预测视频变化), 这种推理是 modality-agnostic 的, 共享反而帮助 alignment。

这和大脑 "早期感觉皮层分化, 后期联合皮层整合" 的结构同构。

---

## 实验结果说了什么

### Masked 路线的结果

Base Block (joint attention, 不共享) 视觉质量最好, FID 10.13。

Split Attention pixel-level 最准 (PSNR 最高), 但视觉真实度最差 (FID 15.31)。这个 trade-off 很有意思: split attention 鼓励 "保守地复制过去", PSNR 高; joint attention 鼓励 "生成逼真新内容", FID 低。

参数共享方面, modality sharing 几乎无损 (FID 11.67 vs 10.13, 差距很小), 参数省 26%。Full sharing 略有损失但最快最省, 极度资源受限时是最佳选择。

### Flow 路线的结果

被 Masked 全面吊打。FID 110 vs 10, 差 10 倍。PSNR 20 vs 29, 差 9 dB。

为什么差距这么大? 几个原因:

1. **VQ-VAE tokenizer 比 continuous VAE 更适合视频生成**。Discrete tokens + cross-entropy 信号尖锐, 模型学得明确。Continuous latents + MSE 信号模糊, 容易学出模糊平均。MAGVIT2 已经证明 "tokenizer is key"。

2. **Masked prediction 的 bidirectional context 天然适合视频**。视频时空冗余强, mask 后能从邻居推断, inductive bias 占优。

3. **Flow matching 在小 scale 不占优**。它在大模型上才显出比 diffusion 的优势, 这里 1.36B 还不够大。

4. **Cosmos continuous VAE 压得太狠**。Flow-HWM 用 16×16 latent grid (每帧 256 tokens), Masked-HWM 用 32×32 (每帧 1024 tokens), 前者细节丢失太多, 生成视频有 spotted patches 和模糊边缘。

反直觉的是, Flow-HWM 上 Full Sharing 反而全面最优——FID 最低、速度最快、内存最少。这是因为 1.36B 模型对 100 小时数据来说偏大, sharing 起到 regularization 作用, 减少冗余参数反而 generalization 更好。

---

## 最关键的 takeaways

1. **Masked video modeling 在小 scale 完胜 flow matching**。Tokenizer 的选择比生成范式更重要。VQ-VAE + cross-entropy 是 "穷人" 的最佳选择。

2. **Parameter sharing 是免费午餐**。Modality sharing 几乎无损省 26% 参数, full sharing 略损但省最多。这对资源受限的 lab 极其重要。

3. **Joint attention 视觉真实度最好, split attention pixel 对齐最好**。根据下游需求选——做 planning 要 visual plausibility 选 joint, 做 synthetic data 要 pixel accuracy 选 split。

4. **Hybrid sharing (前 separate 后 sharing) 是 sweet spot**。Early layers 学 modality-specific 特征, late layers 做 cross-modal 推理, 各取所长。

5. **2 张 GPU 真能训出能用的 humanoid world model**。0.195B 参数, 单 A6000, 60K steps。这打破了 "world model 必须烧大算力" 的迷思。

---

## 这篇 paper 的 limitations

作者没明说但我觉得重要的问题:

1. **没有 action consistency 量化**: 生成的视频和给定动作是否物理一致? 光看 FID/PSNR 不够, world model 的核心价值是 action-conditioned plausibility, 需要额外 metric。

2. **没有 downstream task eval**: 没展示用 HWM 做 planning 或生成 synthetic data 训 policy 的效果。World model 的最终价值在 downstream, 光看视频质量不够说明问题。

3. **Finger-level 细节缺失**: VQ-VAE 32×32 latent 不足以表达 fine-grained manipulation, 对 humanoid 抓取任务来说是硬伤。

4. **Long horizon 退化**: 只预测 8 帧, 对 long-horizon planning 不够。需要 hierarchical 或 recurrent 结构。

5. **Single embodiment**: 只在 1X EVE 上验证, 跨机器人迁移未测。

---

## 对你的 particular thoughts

Andrej, 你在 nanoGPT 和 LLM101 里强调 simple + hackable + educational 的 philosophy, 这篇 paper 完全 embody 了——0.195B 参数, 单 A6000, 开源, 可复现。这种 democratization 对学术圈价值巨大。

几个你可能会特别有共鸣的点:

- **Masked >> Flow 在小 scale 的胜利**: 和你常说的 "tokenizer is all you need" 一致。Inductive bias 在小数据上比 model elegance 重要得多。

- **Parameter sharing 的 scaling intuition**: 存在一个 sweet spot 随 model size 变化。小模型 sharing 受益 (regularization), 大模型 sharing 可能失效 (容量足够, separate 更好)。这个 transition point 值得研究。

- **World model 作为 differentiable simulator**: 你讲过 model-based RL 的潜力, HWM 这种 lightweight world model 正好可以做 MPC 的 dynamics model, 在 imagination 里 rollout + gradient-based planning。这是接 downstream 的自然路径。

- **"Early layers 分化, late layers 整合"**: 这和 LLM 里 "early syntax, late semantics" 的发现异曲同工, 跨 modality 也成立。Universal pattern。

---

## References

- [Humanoid World Models (GitHub repo)](https://github.com/qasim-ali/Humanoid-World-Models)
- [MaskGIT - masked image generation](https://arxiv.org/abs/2202.04200)
- [MAGVIT - masked video generation](https://arxiv.org/abs/2212.05199)
- [MAGVIT2 - tokenizer is key](https://arxiv.org/abs/2310.05737)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Stable Diffusion 3](https://arxiv.org/abs/2403.03206)
- [DiT - diffusion transformer](https://arxiv.org/abs/2212.09748)
- [NVIDIA Cosmos](https://arxiv.org/abs/2501.03575)
- [VQ-VAE](https://arxiv.org/abs/1711.00937)
- [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [Copilot-4D - noise injection](https://arxiv.org/abs/2311.01017)
- [AuraFlow - parameter sharing](https://blog.fal.ai/auraflow/)
- [DiT-Air - efficient DiT design](https://arxiv.org/abs/2503.10618)
- [pi0 - VLA flow model](https://arxiv.org/abs/2410.24164)
- [1X World Model Challenge dataset](https://github.com/1x-technologies/1xgpt)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)
- [Classifier-free guidance](https://arxiv.org/abs/2207.12587)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [Genie - interactive environments](https://proceedings.mlr.press/v235/bruce24a.html)
- [Sora review](https://arxiv.org/abs/2402.17177)
- [MovieGen](https://arxiv.org/abs/2410.13720)
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [iVideoGPT](https://arxiv.org/abs/2410.08991)
- [Pandora](https://arxiv.org/abs/2406.09455)

想深入哪个部分, 比如具体怎么用 HWM 接 MPC 做 planning, 或者 parameter sharing 在不同 model size 上的 scaling behavior, 我可以展开聊。

---

# Humanoid World Models (HWM) 深度解析

Andrej, 这篇 paper 我读下来感觉是 robotics + video generation 交叉领域一个相当扎实的 "democratization" 工作——把 humanoid-specific world model 压缩到 2 张 A6000 就能跑起来, 同时系统地 ablate 了 transformer block 的设计空间。下面我把它的核心直觉、数学公式、架构图、实验数据都拆开讲一遍。

---

## 1. 核心动机与 problem framing

### 1.1 为什么 humanoid 需要 world model

Humanoid robots (像 1X 的 EVE Android) 的形态本身就和人类环境兼容——bipedal + bimanual, 能直接 imitate human demonstrations ([Zhao et al. 2023, ALOHA](https://arxiv.org/abs/2304.13705))。但要在 open-world 里 reason + plan, 单纯靠 VLA model ([pi0](https://arxiv.org/abs/2410.24164), [OpenVLA](https://arxiv.org/abs/2406.09246)) 有几个痛点: spatial reasoning 弱 ([Tong et al. 2024](https://arxiv.org/abs/2401.06209)), continuous sensorimotor 处理差, prompting 复杂。

World model ([Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122)) 提供了一条 grounded 的路径: 把它当作 action-conditioned video generator, 用它做两件事:
1. **Long-horizon planning**: 把 world model 当 dynamics model, 在 imagination 里 rollout 候选 action 序列, 选最优的 ([Yang et al. 2024](https://arxiv.org/abs/2402.17139))
2. **Synthetic data generation**: 生成合成 trajectory 给 policy 训练用, 提升 data efficiency ([Yang et al. 2023, UniSim](https://arxiv.org/abs/2310.06114))

### 1.2 现有 video generation model 的 gap

Sora ([review](https://arxiv.org/abs/2402.17177))、[MovieGen](https://arxiv.org/abs/2410.13720)、[CogVideoX](https://arxiv.org/abs/2408.06072) 这些大模型视觉效果惊艳, 但:
- 设计目标是 entertainment, 没有 ego-centric physical plausibility
- 大多 closed-source
- 不支持 conditioning on past video (这是 world model 的硬需求)
- compute 极重

[NVIDIA Cosmos](https://arxiv.org/abs/2501.03575) 是少有的开源 video-to-video 模型, 但最小 7B 版本要 8×H100 训练、>40GB VRAM 推理; 作者在 2×A6000 上生成 121 帧要 1 小时+, fine-tune 根本不现实。其他 robot-focused 工作 [IraSim](https://arxiv.org/abs/2406.12802) (机械臂, 无 temporal compression)、[Navigation World Models](https://arxiv.org/abs/2412.03572) (低 DoF, 单帧) 都不适合 humanoid。

**HWM 要回答的核心问题**: 能不能用 2 张 GPU, 搞出一个 physically grounded、humanoid-specific 的 world model?

---

## 2. Formulation: 数学上的精确陈述

预测目标: $f$ 帧 future RGB video $\dot{v}_f \in \mathbb{R}^{f \times 3 \times H \times W}$, conditioned on:
- $p$ 帧 past video $v_p \in \mathbb{R}^{p \times 3 \times H \times W}$
- $p$ 个 past action $a_p \in \mathbb{R}^z$
- $f$ 个 future action $a_f \in \mathbb{R}^z$

其中 action vector $a \in \mathbb{R}^{25}$, 包含 joint velocities、hand closure、wrist/knee/elbow/shoulder/neck/hip 的 pitch-yaw-roll。

视频经过 VAE encoder 压到 latent space: $v_p, v_f \to L_p, L_f$。两种模型分道扬镳:
- **Masked-HWM**: 用 VQ-VAE 把 latents 量化成离散 tokens, vocabulary 大小 $s$
- **Flow-HWM**: 用 continuous VAE, latents 保持连续

实验中: $f=8, p=9, H=W=256$。

---

## 3. Masked-HWM: 离散 token + masked prediction

### 3.1 直觉

Masked video modeling 的核心思想来自 [MaskGIT](https://arxiv.org/abs/2202.04200) 和 [MAGVIT](https://arxiv.org/abs/2212.05199): 把视频压成 discrete token grid, 然后像 BERT 一样随机 mask 掉一部分 tokens, 让 transformer 预测被 mask 掉的。这比 autoregressive (按顺序逐个生成) 有两大优势:
1. **Bidirectional context**: 每个 token 都能看到上下文 (空间 + 时间双向), 表示学习更高效
2. **Parallel decoding**: tokens 可以并行预测, 推理快很多

### 3.2 训练流程

**Step 1: Tokenize**
把 $v_p, v_f$ 通过 VQ-VAE 编码, 得到离散 token 序列 $\mathbf{L_p}, \mathbf{L_f}$, 沿 temporal 维度 concat: $\mathbf{L} = [\mathbf{L_p}; \mathbf{L_f}]$。

**Step 2: Copilot-4D 风格 noise injection**
参考 [Copilot-4D](https://arxiv.org/abs/2311.01017), 用 random token replacement 给 latents 加噪, corruption rate 从 $\mathcal{U}(0, \rho_{max})$ 采样, $\rho_{max}=0.2$。这一步让模型见过 noisy input, 提升 robustness。

**Step 3: Per-frame masking**
对 future latents $\mathbf{L_f}$ 用 per-frame thresholding:
- 每帧采 $r \sim \mathcal{U}(0,1)$
- 用 scheduling function $\gamma(r)$ 算 masking threshold (cosine schedule, 来自 MaskGIT)
- 帧内每个 token 采 $u \sim \mathcal{U}(0,1)$, 若 $u < \gamma(r)$ 则 mask 掉

Per-frame 而不是全局 mask 的直觉: 让不同帧有不同的 mask 比例, 模型学到 "有些帧信息多, 有些帧信息少" 的鲁棒推理。

**Step 4: Cross-entropy loss**

$$\mathcal{L} = -\mathbb{E}_{\mathbf{M}}\left[\sum_i \mathbf{M}_i \log p(\hat{L}_i \mid \mathbf{L_f})\right]$$

变量含义:
- $\mathbf{M}$: binary mask, $\mathbf{M}_i \in \{0,1\}$ 表示位置 $i$ 是否被 mask
- $\hat{L}_i$: 模型对位置 $i$ 的预测 token 分布
- $\mathbf{L_f}$: corrupted + masked 后的 future latents (作为 input)
- 期望 $\mathbb{E}_{\mathbf{M}}$ 是对不同 mask pattern 取平均

直觉上, 这个 loss 强迫模型从 partial observed future + 完整 past 推断 missing tokens——和 BERT MLM 完全同构, 只是搬到 video 的 spatiotemporal token grid 上。

### 3.3 Inference: parallel iterative decoding

Inference 时把 $\mathbf{L_f}$ 全部 mask 掉, 然后:
- **Latent frame by latent frame**: 预测一帧, 把结果 feedback 给下一帧
- **帧内 K 步 refinement** ($K=2$): 每步并行预测所有 tokens, 然后随机 re-mask 一部分置信度低的, 再预测, 迭代改进

这种 "解码 + 重 mask" 的策略让模型像 sculptor 一样逐步 refine, 比 autoregressive 快几个数量级。详细算法见 [MaskGIT paper](https://arxiv.org/abs/2202.04200)。

### 3.4 Architecture (Masked-HWM Base Block)

参考 [Genie](https://proceedings.mlr.press/v235/bruce24a.html) 和 [Pandora](https://arxiv.org/abs/2406.09455), 采用 **factorized spatio-temporal attention**——把 full spatiotemporal attention 拆成 spatial + temporal 两个 cheaper 的 attention, 复杂度从 $O((T \cdot H \cdot W)^2)$ 降到 $O(T \cdot (HW)^2 + HW \cdot T^2)$。

四个 token streams: $v_p, v_f, a_p, a_f$, 各自独立的 MLP weights, 但共享 temporal attention 层:
- **Temporal attention**: 所有 streams 的 tokens $[a_p, a_f, \mathbf{L}]$ 一起 attend (沿时间维)
- **Spatial attention**: 只对 video tokens 做 (沿空间维)
- **RoPE**: spatial 用 2D RoPE, temporal 用 1D RoPE ([Su et al. 2023](https://arxiv.org/abs/2104.09864))

参数: 24 layers, 8 heads, 512-dim tokens, MLP hidden 2048, 0.321B params, 单 A6000 训练 60K steps, batch 16, AdamW lr=3e-5。

---

## 4. Flow-HWM: 连续 latent + flow matching

### 4.1 Flow Matching 的直觉

Flow matching ([Lipman et al. 2023](https://arxiv.org/abs/2210.02747)) 是 diffusion 的近亲, 但有几个优势:
- 直接学 deterministic velocity field, 不用 reverse SDE
- 训练更简单, sampling 更快
- ODE solver 一步就能走很远

直觉上, diffusion 是 "随机游走回数据", flow matching 是 "确定性地从噪声滑到数据"。两者数学上都是 continuous normalizing flow 的特例, 但 FM 的轨迹更直, 数值积分更友好。

### 4.2 数学公式详解

设 $\mathbf{X}_1$ 是 latent space 中的 video sample, $\mathbf{X}_0 \sim \mathcal{N}(0, \mathbf{I})$ 是 Gaussian prior。中间时间 $t \in [0,1]$ 的插值点:

$$\mathbf{X}_t = t \mathbf{X}_1 + (1 - (1-\sigma_{min})t) \mathbf{X}_0 \tag{1}$$

变量含义:
- $t$: flow time, 0 对应纯噪声, 1 对应真实数据
- $\mathbf{X}_1$: 目标 video latent (ground truth)
- $\mathbf{X}_0$: 采样自 $\mathcal{N}(0, \mathbf{I})$ 的噪声
- $\sigma_{min}$: 小正数 (典型 $10^{-4}$ 量级), 确保 $t=1$ 时仍有非零噪声 support, 避免 singular velocity

对 $t$ 求导得到 ground-truth velocity:

$$\mathbf{V}_t = \frac{d\mathbf{X}_t}{dt} = \mathbf{X}_1 - (1-\sigma_{min})\mathbf{X}_0 \tag{2}$$

注意 $\mathbf{V}_t$ 与 $t$ 无关 (linear interpolation 的特性), 这是 flow matching 比 diffusion 简单的关键之一——velocity field 在时间上是常数, 训练信号更稳定。

模型 $u_\theta$ 预测 instantaneous velocity, conditioned on past video $v_p$, past/future actions $a_p, a_f$, time $t$:

$$\mathbb{E}_{t, \mathbf{X}_0, \mathbf{X}_1, a_p, a_f, v_p} = \left[\left\| u_\theta(\mathbf{X}_t, a_p, a_f, v_p, t) - \mathbf{V}_t \right\|^2\right] \tag{3}$$

直觉: 让网络在任何中间时间点 $t$, 给定 noisy $\mathbf{X}_t$ 和 conditioning, 都能预测出 "从噪声到数据的瞬时方向"。训练时随机采 $t$, 模型学会整条轨迹的 velocity field。

### 4.3 Inference

从 $t=0$ 的纯噪声开始, 用 first-order Euler ODE solver 积分到 $t=1$:

$$\mathbf{X}_{t+\Delta t} = \mathbf{X}_t + \Delta t \cdot u_\theta(\mathbf{X}_t, \cdot, t)$$

50 步 denoising, classifier-free guidance scale 3.0 ([Ho & Salimans 2022](https://arxiv.org/abs/2207.12587))。CFG 的作用是: 训练时随机 drop conditioning, 推理时把 conditional 和 unconditional prediction 外推, $u_{guided} = u_{uncond} + s \cdot (u_{cond} - u_{uncond})$, $s=3$。

### 4.4 Architecture (Flow-HWM Base Block)

Inspired by [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) 和 [DiT](https://arxiv.org/abs/2212.09748)。每个 stream ($v_p, v_f, a_p, a_f$) 独立参数, 通过 joint attention 交互。单个 block 流程:

1. **Timestep modulation (pre-attention)**: 每个 stream 用 learnable $\alpha_0, \beta_0$ 对 tokens 做 scale + shift:
   $$h \leftarrow \alpha_0(t) \cdot h + \beta_0(t)$$
   这就是 [Peebles & Xie 2023, DiT](https://arxiv.org/abs/2212.09748) 的 adaLN-Zero 思路。

2. **QKV projection**: 每个 stream 用独立的 $W_{QKV}$ 算 queries/keys/values, 加 positional encoding:
   - Video tokens: 3D RoPE (space_x, space_y, time) — 同 [Cosmos](https://arxiv.org/abs/2501.03575)
   - Action tokens: 1D RoPE (time only)
   - Past + future tokens concat 后再加 PE

3. **Joint attention**: 所有 streams 的 QKV 拼起来做 global self-attention, 让 cross-modal 交互发生

4. **Timestep rescaling**: 用 $\gamma_0(t)$ rescale attention output, 加 residual

5. **FFN modulation**: 再用 $\alpha_1, \beta_1$ modulate, 过 stream-specific MLP, 用 $\gamma_1$ rescale, 加 residual

参数: 17 layers, 1172-dim tokens, 1.36B params, 2×A6000 训练 150K steps, batch 128, AdamW lr=1e-4 cosine schedule, patch size $p_{lw}=2, p_t=1$。

**关键训练 trick**: 
- Final linear layers 用 Xavier init, **不用 zero-init** (zero-init 导致 instability, 与原 DiT 推荐相反)
- **No lr warmup** (warmup 反而让 convergence 变差)

这两个反直觉的发现说明 flow matching 在 video 这种高维 latent 上对初始化敏感, 经验性 tuning 重要。

---

## 5. Transformer Block 设计空间 (核心贡献)

这是 paper 最有价值的部分。三个设计维度:

### 5.1 Dimension 1: Joint vs Split Attention

**Joint Attention**: 所有 4 个 streams 的 tokens 拼起来做 global self-attention。优点是 cross-modal 交互丰富, 缺点是 memory cost $O(N^2)$ 随 token 数平方增长。在 video 上 long sequence 时很贵。

**Split Attention** (two-stage, [Cosmos](https://arxiv.org/abs/2501.03575)/[MovieGen](https://arxiv.org/abs/2410.13720) 风格):
1. 每个 stream 先做 intra-stream self-attention (处理自己内部依赖)
2. 然后 cross-attention: $v_f$ 当 queries, $v_p, a_p, a_f$ 当 keys/values

直觉: future video 的生成显式地 query past observations + intended actions, 避免 full global attention 的开销。这对 video 特别合适, 因为 video token 数量本来就大。

### 5.2 Dimension 2: Parameter Sharing

灵感来自 [AuraFlow (fal.ai)](https://blog.fal.ai/auraflow/) 和 [DiT-Air (Chen et al. 2025)](https://arxiv.org/abs/2503.10618) 的发现: image generation 里 joint attention 的好处可以用更少参数获得, 通过 sharing attention/modulation/MLP weights 实现。

三种变体:
- **Base Block**: 完全 separate, 每个 stream 独立 weights
- **Modality Sharing**: video streams ($v_p, v_f$) 共享 weights, action streams ($a_p, a_f$) 共享 weights
- **Full Sharing**: 所有 4 个 streams 共享 $(\alpha, \beta, \gamma), W_{QKV},$ MLPs

**Hybrid scheme**: 前 4 层用 separate (学 modality-specific representation), 后 $l-4$ 层用 sharing (cross-modal reasoning 时用 compact 参数)。这反映了深度网络 "early layers 提特征, late layers 做抽象推理" 的直觉。

### 5.3 Dimension 3: Token Stream Grouping
- Modality-based: video 一组, action 一组
- Fully separate: 4 个 stream 全独立

---

## 6. 实验数据深度解读

### 6.1 Masked-HWM 结果 (Table 1)

| Metric | Split Attn | Base Block | Modality Share | Full Share |
|---|---|---|---|---|
| Size (B) | 0.220 | 0.321 | 0.237 | 0.195 |
| Peak GPU (GB) | 2.22 | 2.63 | 2.30 | 2.12 |
| Samples/sec | 2.09 | 2.27 | 2.25 | 2.36 |
| FID ↓ | 15.31 | **10.13** | 11.67 | 14.21 |
| PSNR (dB) ↑ | **29.37** | 29.02 | 28.97 | 28.66 |

**关键观察**:

1. **Base Block (joint attn, no sharing) FID 最低 (10.13)**: joint attention 让 cross-modal 交互最充分, 视觉真实度最好。这印证了 SD3 在 image 上的发现搬到 video 仍成立。

2. **Split Attention PSNR 最高 (29.37) 但 FID 最差 (15.31)**: 这是非常有意思的 trade-off。PSNR 衡量 pixel-level fidelity (逐像素 MSE), FID 衡量 distributional realism (用 Inception-V3 feature 算 Fréchet distance)。Split attention 的 cross-attention 结构让 $v_f$ 显式 query $v_p$, pixel-level 对齐更好, 但失去了 token 间的 free interaction, global structure 真实度反而下降。

   直觉: PSNR 偏向 "保守地复制过去", FID 偏向 "生成逼真的新内容"。Split attention 鼓励前者, joint attention 鼓励后者。

3. **Modality Sharing 几乎无损**: FID 11.67 vs Base 10.13 (差 1.5), 但参数少 26% (0.237 vs 0.321B)。说明 video past/future 共享 weights 完全合理——它们本来就是同一种 modality 的不同时间切片。Action past/future 同理。

4. **Full Sharing 略有损失但最快最省**: FID 14.21, 但 0.195B (39% 减少), 2.36 samples/sec。在资源极度受限时是最佳选择。

### 6.2 Flow-HWM 结果 (Table 2)

| Metric | Split Attn | Base Block | Modality Share | Full Share |
|---|---|---|---|---|
| Size (B) | 0.944 | 1.36 | 0.886 | **0.648** |
| Peak GPU (GB) | 4.37 | 5.94 | 4.41 | **3.25** |
| Samples/sec | 1.11 | 1.69 | 1.89 | **1.91** |
| FID ↓ | 111.12 | 111.59 | 112.75 | **110.73** |
| PSNR (dB) ↑ | **20.50** | 20.42 | 20.50 | 20.43 |

**关键观察**:

1. **Flow-HWM 全面被 Masked-HWM 吊打**: FID 110 vs 10 (差 10×), PSNR 20 vs 29 (差 9 dB)。这个差距非常大。

   为什么? 我推测几个原因:
   - **VQ-VAE tokenizer 比 continuous VAE 更适合视频生成**: [MAGVIT2](https://arxiv.org/abs/2310.05737) 已经证明 "language model beats diffusion, tokenizer is key"。Discrete tokens 让 transformer 用 cross-entropy loss, 信号更尖锐; continuous latents 用 MSE loss, 信号更模糊, 容易学出 mean regression (生成模糊平均)。
   - **Masked prediction 的 bidirectional context 天然适合 video**: 视频有强时空冗余, mask 后能从邻居推断, 这是 inductive bias 上的优势。
   - **Flow matching 在小数据/小模型 regime 不占优**: 大模型上 FM 才显示出比 diffusion 更好的 scaling, 这里 1.36B 还不算大。
   - **Cosmos Continuous VAE 的 16×16 latent 太小**: Flow-HWM 用 8x16x16 compression, latent grid 只有 16×16 = 256 tokens/frame; Masked-HWM 用 8x8x8, 32×32 = 1024 tokens/frame。前者 spatial detail 丢失太多, 这从 Figure 5 的 "spotted patches" 和 "blurry edges" 能看出来。

2. **Full Sharing 在 Flow-HWM 上竟然全面最优**: FID 110.73 (最低), 速度最快 (1.91), 内存最少 (3.25 GB), 参数最少 (0.648B)。这是反直觉的——按理 separate 参数应该表达力更强。

   我的解读: Flow-HWM 训练数据相对模型容量来说不够 (100 hours × 30 Hz × 8 frames/sample ≈ 1.3M training samples, 对 1.36B 模型来说 underfitting 风险小但 overfitting 风险大)。Full sharing 起到了 strong regularization 作用, 减少了冗余参数, 反而 generalization 更好。这和 [Chen et al. 2025, DiT-Air](https://arxiv.org/abs/2503.10618) 在 image generation 上的发现一致。

3. **Split Attention PSNR 最高**: 同 Masked-HWM, 显式 cross-attention 让 pixel-level 对齐更好, 但这里 FID 也是最好之一 (111.12 vs Full Share 110.73 几乎平手)。在 Flow-HWM 上 split attention 没有显著劣势, 因为 continuous latents 本身就模糊, FID 差异不大。

4. **训练稳定性敏感**: Paper 提到 zero-init 导致 instability, warmup 让 convergence 变差。这暗示 flow matching 在 video 上对 optimization landscape 敏感, 可能跟 continuous latents 的高维度 + MSE loss 的 flat gradient 有关。

### 6.3 定性结果 (Figures 4, 5)

**Masked-HWM (Figure 4)**:
- Scene structure (家具、小物体) 学得好
- Robot appendages (arm, wheels) 大致准确
- **Fingers 经常模糊/entangled**: 这是 VQ-VAE 的 spatial compression 限制, 32×32 latent 不足以表达 fine-grained manipulation
- **Lighting 鲁棒**: 第 3 (亮) / 4 (暗) sequence 都生成得不错, 说明模型学到了 lighting-invariant 的 scene representation

**Flow-HWM (Figure 5)**:
- 整体 scene structure OK (墙、地、门)
- **明显 blur + spotted artifacts**: continuous VAE + MSE loss 的典型问题
- **Late frames 退化**: 直边变弯, 圆形变形。这说明 temporal consistency 不够, 模型在长 horizon 上 accumulate error
- **Arm 配置 fallback to canonical**: 罕见 arm pose 生成不出来, 默认到训练集最常见的 arm 外观。这是 mode collapse 的表现, CFG scale 3.0 可能还不够强

---

## 7. Parameter Sharing 的 intuition building

为什么 sharing work 得这么好? 我觉得可以从几个角度理解:

### 7.1 表示论角度
Past video $v_p$ 和 future video $v_f$ 在 latent space 里是同一种 distribution 的不同 sample。它们的 "what is a video token" 的底层概念是共享的——edge detector, texture recognizer, motion pattern matcher 这些 low-level feature 完全一致。让它们用同一套 weights 就像让两个 twin 共享同一套感官系统, 自然合理。

Past action $a_p$ 和 future action $a_f$ 同理——它们都是 25-dim 的 joint state vector, 区别只是时间。共享 weights 强迫模型学到 time-agnostic 的 action representation, 然后用 positional encoding 区分 past/future。

### 7.2 信息论角度
4 个 streams 之间有大量 mutual information (同一 scene 的不同视角/时间)。Separate weights 让每个 stream 独立学一套 representation, 浪费容量。Sharing 强制 alignment, 让模型用有限容量学 "真正不同的东西"。

### 7.3 Optimization 角度
Sharing 减少了 effective parameter count, 在有限数据下相当于 strong prior, 减少 overfitting。这解释了为什么 Flow-HWM 上 Full Sharing FID 反而最低——1.36B 模型对 100h 数据来说偏大, sharing 起到 regularization。

### 7.4 Hybrid scheme 的智慧
前 4 层 separate + 后 $l-4$ 层 sharing 的设计很有讲究。Early layers 学 low-level modality-specific feature (video 的 visual feature, action 的 kinematic feature), 这些 feature 确实不同, 应该 separate。Late layers 做 cross-modal reasoning (用 action 预测 video 变化), 这种 reasoning 是 modality-agnostic 的, sharing 反而帮助 alignment。

这和 brain 的 "early sensory cortex 分化, late association cortex 整合" 的结构同构。

---

## 8. 和相关工作的 positioning

| 工作 | Embodiment | Paradigm | Open Source | Compute | Past Video Cond |
|---|---|---|---|---|---|
| [UniSim](https://arxiv.org/abs/2310.06114) | Generic | Diffusion (U-Net) | ❌ | Large | ❌ (text only) |
| [IraSim](https://arxiv.org/abs/2406.12802) | Robot arm | Diffusion | ❌ | Medium | ❌ |
| [Navigation WM](https://arxiv.org/abs/2412.03572) | Mobile robot | Diffusion | ❌ | Medium | ❌ (single frame) |
| [Cosmos](https://arxiv.org/abs/2501.03575) | Generic | Diffusion | ✅ | 8×H100+ | ✅ |
| [Genie](https://proceedings.mlr.press/v235/bruce24a.html) | Game env | Masked | ❌ | Large | ✅ |
| [iVideoGPT](https://arxiv.org/abs/2410.08991) | Generic | Autoregressive | ✅ | Medium | ✅ |
| [Pandora](https://arxiv.org/abs/2406.09455) | Generic | Autoregressive | ✅ | Medium | ✅ |
| **HWM** | **Humanoid** | **Masked + Flow** | **✅** | **1-2 GPUs** | ✅ |

HWM 在 humanoid-specific + lightweight + open 的交集上是独一无二的。

---

## 9. Limitations 和 future directions

### 9.1 显式 limitations (paper 暗示)
- **Finger-level fine detail 缺失**: VQ-VAE 32×32 latent 限制, 需要更高分辨率 tokenizer
- **Long horizon 退化**: 8 帧预测对 long-horizon planning 不够, 需要 hierarchical 或 recurrent 结构
- **Flow-HWM 质量差**: 需要更大数据/模型才能发挥 FM 优势
- **Single embodiment**: 只在 EVE Android 上验证, 跨 humanoid 迁移未测试

### 9.2 隐式 limitations (我的观察)
- **No action consistency check**: 生成的 video 和给定 action 是否物理一致没量化 (只看 FID/PSNR)。World model 的核心价值是 action-conditioned plausibility, 缺这个 metric 让人担心。
- **No downstream task eval**: 没展示用 HWM 做 planning 或 synthetic data generation 训 policy 的效果。World model 的最终价值在 downstream, 光看视频质量不够。
- **No comparison with Cosmos fine-tuned on humanoid data**: 直接比 Cosmos 原版不公平, 应该 fine-tune Cosmos 在 1xGPT 上看 baseline。
- **Dataset bias**: 1xGPT 100 小时主要是 indoor household tasks, outdoor/industrial 场景未覆盖。

### 9.3 Future directions (我的联想)
- **Hierarchical world model**: 高层预测 coarse scene trajectory, 低层 refine细节, 类似 [HunyuanVideo](https://arxiv.org/abs/2412.03603) 的 pyramidal 思路或 [Pyramidal Flow Matching](https://arxiv.org/abs/2410.05954)
- **3D-aware tokenizer**: 用 [3D VAE](https://arxiv.org/abs/2401.12945) 或 neural radiance field 替代 2D VAE, 让 model 隐式学 3D structure, 解决 finger blur
- **Action-conditioned consistency loss**: 加一个 inverse dynamics model 检查 "生成的 video 对应什么 action", 和给定 action 比对
- **Multi-embodiment pretraining**: 用 [Open-X-Embodiment](https://arxiv.org/abs/2310.08864) 思路在多种 robot 上预训练, 然后 humanoid-specific fine-tune
- **World model + VLA joint training**: 让 world model 和 policy 共享 backbone, 互相 distill, 类似 [pi0](https://arxiv.org/abs/2410.24164) 的 flow matching VLA 思路

---

## 10. 对你 (Karpathy) 的 particular thoughts

Andrej, 你在 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 LLM101 里强调 "simple, hackable, educational" 的 implementation philosophy, 这篇 paper 完全 embody 了——0.195B 参数, 单 A6000, 60K steps 就能训出能用的 humanoid world model。这种 democratization 对学术圈和小 lab 极其重要。

几个可能你特别感兴趣的角度:
1. **Masked >> Flow 在小 scale 的胜利**: 这和你经常强调的 "tokenizer is all you need" 一致。VQ-VAE + cross-entropy 的 inductive bias 在小数据上完胜 continuous + MSE。
2. **Parameter sharing 的 scaling law**: 是否存在一个 "sharing ratio" 的 sweet spot 随 model size 变化? 大模型上 sharing 应该失效 (容量足够, separate 更好), 小模型上 sharing 受益。这个 transition point 值得研究。
3. **Hybrid scheme 和 early/late layer 分工**: 这和 LLM 里观察到的 "early layers 做 syntax, late layers 做 semantics" 异曲同工, 不同 modality 之间也成立。
4. **World model 作为 "differentiable simulator"**: 你在 [YouTube 讲过](https://www.youtube.com/watch?v=kCc8FmEb1nY) model-based RL 的潜力, HWM 这种 lightweight world model 正好可以做 model predictive control (MPC) 的 dynamics model, 在 imagination 里 rollout + gradient-based planning。

---

## 参考 links

- Paper 本身: [Humanoid World Models (GitHub)](https://github.com/qasim-ali/Humanoid-World-Models)
- [MaskGIT](https://arxiv.org/abs/2202.04200)
- [MAGVIT](https://arxiv.org/abs/2212.05199)
- [MAGVIT2 (tokenizer is key)](https://arxiv.org/abs/2310.05737)
- [Open-MAGVIT2](https://arxiv.org/abs/2409.04410)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Stable Diffusion 3](https://arxiv.org/abs/2403.03206)
- [DiT](https://arxiv.org/abs/2212.09748)
- [NVIDIA Cosmos](https://arxiv.org/abs/2501.03575)
- [Copilot-4D](https://arxiv.org/abs/2311.01017)
- [VQ-VAE](https://arxiv.org/abs/1711.00937)
- [World Models (Ha & Schmidhuber)](https://arxiv.org/abs/1803.10122)
- [DDPM](https://arxiv.org/abs/2006.11270)
- [Classifier-free guidance](https://arxiv.org/abs/2207.12587)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [AuraFlow (fal.ai)](https://blog.fal.ai/auraflow/)
- [DiT-Air](https://arxiv.org/abs/2503.10618)
- [1X World Model Challenge](https://github.com/1x-technologies/1xgpt)
- [pi0](https://arxiv.org/abs/2410.24164)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Sora review](https://arxiv.org/abs/2402.17177)
- [MovieGen](https://arxiv.org/abs/2410.13720)
- [CogVideoX](https://arxiv.org/abs/2408.06072)
- [Genie (DeepMind)](https://proceedings.mlr.press/v235/bruce24a.html)
- [Pandora](https://arxiv.org/abs/2406.09455)
- [iVideoGPT](https://arxiv.org/abs/2410.08991)
- [HunyuanVideo](https://arxiv.org/abs/2412.03603)
- [Pyramidal Flow Matching](https://arxiv.org/abs/2410.05954)
- [Navigation World Models](https://arxiv.org/abs/2412.03572)
- [UniSim](https://arxiv.org/abs/2310.06114)
- [IraSim](https://arxiv.org/abs/2406.12802)
- [ALOHA / ACT](https://arxiv.org/abs/2304.13705)
- [nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)

如果你对某个具体部分 (比如 flow matching 的 ODE 推导、parameter sharing 的 ablation 细节、或者 downstream planning 怎么接 HWM) 想更深入聊, 我可以展开。
