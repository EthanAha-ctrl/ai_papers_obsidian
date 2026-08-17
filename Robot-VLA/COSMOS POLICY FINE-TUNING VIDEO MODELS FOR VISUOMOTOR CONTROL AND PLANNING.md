---
source_pdf: COSMOS POLICY FINE-TUNING VIDEO MODELS FOR VISUOMOTOR CONTROL AND PLANNING.pdf
paper_sha256: 05696475c9b5481afbf88673a65f80a46cf3fb87a5755c1410e83d0f468cc6e0
processed_at: '2026-08-03T17:32:05-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Cosmos Policy 人话版

Andrej, 简单讲, 这篇 paper 干了这么一件事: **拿一个已经很会生成视频的大模型, 不改任何架构, 把它变成一个机器人 policy**。

---

## 一句话概括

把机器人要输出的 action, 要预测的未来画面, 要估的 reward, 全部 "伪装" 成视频帧, 塞进 video diffusion 的 latent sequence 里。video model 本来就会生成帧, 那就让它顺带学会生成这些 "假帧"。

---

## 为什么要用 video model, 不继续搞 VLA

现在 robot policy 的主流路线是 VLA: 拿一个 vision-language model (在 image-text pair 上预训练的 backbone), 在 robot demo 上 fine-tune, 让它学会输出 action。π0, OpenVLA, RT-2 都是这个路线。

VLA backbone 学到的是 "这张图里有什么东西, 这句话什么意思"。它是从静态图对里学的, 没有时间维度。

Video model 学到的是 "这张图之后, 世界会怎么变"。它从 millions of internet 视频里学到了 temporal causality, 物体怎么运动, 接触怎么发生, 相机一动画面怎么变。这些东西本质上是物理 world dynamics 的近似。

Robotics 就是 "在物理世界里做 sequential decision", video model 的预训练任务跟这个几乎对齐。VLA 的预训练任务跟它差一层。所以 video model 可能是更好的 backbone。

Cosmos Policy 的实验结果支持这个判断: 它没有用任何 robot action 的 pretraining (VLA 路线都用了), 但在 LIBERO, RoboCasa, 真机 ALOHA 上全赢 π0.5, OpenVLA-OFT+, CogVLA 这些。

---

## 核心招数: Latent Frame Injection

### 遇到的问题

Cosmos-Predict2 这个 video model 只会两件事: 吃一张图 + 一句话, 生成一段 video。

但 robotics 需要:
- 输入: 多个 camera 视角 + robot 的 joint state
- 输出: action chunk (连续多步动作) + 未来画面 + value (估计能不能成功)
- 这些东西不是 image, 是 low-dim vector

之前的做法 (UVA, Video Policy 等) 是: 在 video model 后面或者旁边加一个新 module, 专门处理 action。要么多阶段训练, 要么改架构。

### Cosmos Policy 的解法

不动架构。把所有东西都编码成 "latent frame"。

Video model 内部工作流程是: 输入图经过 VAE tokenizer, 变成一串 latent frame, 每帧是一个 $H' \times W' \times C'$ 的 volume。video model 的任务就是在这个 latent space 里 denoise。

既然它处理的是 "一串 latent frame", 那我就可以往这个序列里插入额外的 frame, 告诉模型 "这些也是 latent frame, 你也帮着生成一下"。

具体怎么做: 比如一个 action chunk 是 50 步 × 14 维, 先 normalize 到 [-1, 1], flatten 成一个 700 长度的 vector, 然后 copy 很多次填满一个 $H' \times W' \times C'$ 的 volume。这个 volume 就当成一帧 latent。推理时反过来, 把 volume 里所有 copy 取 average, un-normalize, 就拿回 action。

Proprioception (joint state) 和 value 同理。Value 是 scalar, 就把这个 scalar 复制到整个 volume。

### 序列长什么样

以 ALOHA 为例 (2 个外部 camera + 1 个 wrist camera), latent sequence 有 11 帧:

```
[placeholder] [proprio] [wrist] [cam1] [cam2] [action] [future_proprio] [future_wrist] [future_cam1] [future_cam2] [value]
   占位          s                                     a            s'                                          V(s')
```

顺序就是 $(s, a, s', V(s'))$。这个顺序允许从左到右 autoregressive decode: 先出 action, 再出 future state, 再出 value。

### 为什么这个 hack 能 work

Video diffusion 本来就是一个高维 distribution 学习器。它学的是 "给定前几帧 clean latent, 生成后续 noisy latent 的 denoise 方向"。Action distribution 也是一个 distribution。把 action 编码成 latent frame 后, model 用同一套 denoise 算法学它, 复用了 video model 在互联网视频上学到的所有 capacity。

Extract 时取 average, 是因为 copy 进去的 volume 本来 spatial 上就 constant, denoise 后各位置应该一致, average 相当于降噪, 比单点采样稳。

---

## 同时学三件事: Policy, World Model, Value Function

同一个模型, 同一个 latent sequence, 根据哪些 frame 是 conditioning (clean), 哪些是 target (noisy), 可以训练不同的函数:

- **Policy training**: clean 是 $s$ (frame 1-5), target 是 $a, s', V(s')$ (frame 6-11)。学 "给 state, 出 action + 未来 + value"。
- **World model training**: clean 是 $s, a$ (frame 1-6), target 是 $s', V(s')$ (frame 7-11)。学 "给 state + action, 预测未来"。
- **Value function training**: clean 是 $s, a, s'$ (frame 1-10), target 是 $V(s')$ (frame 11)。学 "给完整 trajectory, 估 value"。

每个 batch 按 50/25/25 分给这三件事。50% 的 batch 来自人类 demo, 训 policy。25%+25% 来自 rollout (包括失败的 demo replay), 训 world model 和 value function。

### 为什么要 rollout data

Demos 只覆盖成功轨迹。World model 和 value function 只看成功的 trajectory, 学不到 "这个 action 会导致失败"。所以要让 policy 跑一批 rollout, 收集失败案例, 再 fine-tune world model 和 value function, 它们才能识别 bad action。

### Auxiliary supervision 是性能支柱

Policy 训练时, target 不只是 action, 是 $(a, s', V(s'))$ 一起。这相当于强迫 policy 内部 "想象" 做完这个 action 后世界变成什么样, 最终能不能成功。

这个设计是性能的支柱。Ablation 里, 如果让 policy 只预测 action (去掉 future state 和 value 的 auxiliary target), RoboCasa 成功率从 67.1% 掉到 44.4%, 掉了 22 个点。比去掉 pretraining (掉 3.9 点) 影响还大。

Intuition: pure action regression 在 multimodal distribution 上有问题。比如抓 candy, 你可以抓左边那颗或右边那颗, 两个 mode。L1 regression 取 median, L2 取 mean, 都会预测到两颗中间去。但如果强迫 policy 同时预测做完 action 后的画面 ($s'$), 那它必须从 latent 里 sample 出一个 coherent 的 (抓左边, 画面里左边那颗消失) pair, 不能 average 两个 mode。video diffusion 作为 joint distribution learner 天然支持这种 coherent sampling。

OpenVLA-OFT+ 在抓 candy 时的失败 (Figure 5 right) 就是这个问题的典型表现: 它 reach 到两颗 candy 中间, 因为 L1 regression 在 bimodal 上 mean 是无意义的中间值。

---

## Noise Schedule 改动: 给 action 加权

这是我觉得最聪明的细节。

### 问题

原 Cosmos-Predict2 用 EDM 的 log-normal noise schedule, weight 集中在 low σ。这对 video generation 没问题: 起始几步 denoise 不准没关系, 后面步骤会修正。

但 action 要求精确。起始 denoise 步不准, 后面是 cascading error 累积, 最终 action 偏很多。机器人执行误差几毫米就可能 grasp 失败。

### 解法

改成 hybrid 分布: 70% 概率从原 log-normal 采样, 30% 从 Uniform[1.0, 85.0] 采样。等于在 high σ 尾部加 weight, 让模型在起始 denoise 步有足够训练信号。

推理时把 $\sigma_{min}$ 从 0.002 提到 4。跳过那些 SNR 太低、预测不准的最后几步。Empirically 降低 L1 loss。

这个 insight 适用于所有把 diffusion 用在 robotics 的场景: image generation 的 noise schedule 是为 "看着好看" 设计的, robotics 需要 "数值精确", schedule 要重调。

---

## Planning: Best-of-N Search

### Dual Deployment

- Policy model: 用原 checkpoint (训在 demos 上), 负责 sample action 候选
- Planning model: 在 rollout data 上 fine-tune 过的 checkpoint, 负责 predict 未来画面 + 估 value

分开是为了确保 planning model 是 on-policy 的 (rollout 由 policy model 跑出来的, 分布匹配)。

### Search 流程

1. Policy model sample 8 个候选 action chunk
2. Planning model 对每个 action 预测未来画面 (ensemble 3 次)
3. 对每个未来画面, planning model 估 value (ensemble 5 次)
4. 每个 action 总共 3×5=15 个 value 估计
5. **Majority mean**: 先用 threshold 把每个 value 二值化 (成功/失败), 看 majority 倾向哪边, 再在 majority group 内取 average
6. 选 value 最高的 action 执行

Majority mean 比 naive average 鲁棒。Value 估计经常是 bimodal (要么接近 1 要么接近 0), naive average 会被 outlier 拉偏。

### 效果

ALOHA 上两个最难的任务 (candies in bowl, candy in ziploc bag), planning 比 base policy 平均高 12.5 个点。

Qualitative 例子 (Figure 6): base world model 在 demos 上训, 预测 "抓 ziploc slider" 总是预测成功 (因为 demos 都成功了)。Fine-tune on rollouts 后, world model 见过抓不住的失败案例, 能预测 "这个 grasp 会 slip", planning 就避开这种 action。

### 代价

8 张 H100 并行, 4.9 秒出一个 action chunk。ALOHA 一个 chunk 执行 2 秒。所以 robot 要 pause 5 秒等下一个 chunk。不适合 dynamic task (比如打乒乓球)。Paper 自己承认这是 limitation。

---

## 结果

### LIBERO (仿真, 单臂 Franka)

Cosmos Policy 98.5% average, 比 π0.5 (96.9%), CogVLA (97.4%), OpenVLA-OFT (97.1%) 都高。Long-horizon suite (LIBERO-10) 上 97.6% vs π0.5 的 92.4%, 差距 5 个点, 这个最有意义因为 long-horizon 最难。

### RoboCasa (仿真, 24 个 kitchen 任务, 测试含 unseen object + unseen style)

Cosmos Policy 用 50 个 demo 达到 67.1%。别人用 300-3000 demo 也就 50-66%。数据效率碾压。

### 真机 ALOHA (双臂, 4 个高难度任务)

Cosmos Policy 93.6% average, 比 π0.5 的 88.6% 高 5 个点。

最难的 ziploc bag 任务 (毫米级 precision, stochastic dynamics): Cosmos Policy 85.4, π0.5 只有 61.5。π0.5 抓不住 slider 的右侧 (Figure 5 left)。

Diffusion Policy 在 fold shirt 上只有 23.5%, 完全崩。OpenVLA-OFT+ 在 candies in bowl 上只有 21.6%, 因为 L1 regression 在 multimodal action 上平均两个 mode。

---

## 推理速度的隐藏 finding

1 步 denoising, RoboCasa 66.4% (5 步是 67.1%, 只掉 0.7%), 但推理只要 0.16 秒 (5 步是 0.61 秒, 快 4 倍)。

这暗示: action latent 的 distribution 在 video model 里学得非常 "well-formed", 一步 denoise 就能跳到 mode 附近。跟 image diffusion 需要 50-1000 步完全不同。

可能原因: action 是 low-dim, latent 里 duplicate 成 volume 后 redundant encoding 很强, denoise 一个 spatially-constant 的 noise pattern 比自然 image 简单很多。

对未来 robotics diffusion policy 有启发: action chunk 的 diffusion 可能不需要多步, 1-step 或 few-step distillation 应该是 default 而不是 exception。

---

## 这篇 paper 的真正 takeaway

我觉得最值得记住的不是某个具体数字, 是这个 design philosophy:

**当你有一个强大的 generative foundation model, 想给它加新能力, 不要加新 module。把新 modality 表示成 model 已经会处理的 format, 复用它的 learning algorithm。**

Latent frame injection 是这个 philosophy 的具体实例: action, value 这些 low-dim modality, 都 encode 成 latent frame, 让 video diffusion 当成 "特殊 pattern 的 frame" 来 denoise。

这个 idea 应该可以推广。比如: 把 LLM 的 token 当成某种 latent, 把其他 modality 也 tokenize 进去; 或者把 audio model 的 latent sequence 里插入 control signal。只要 foundation model 是 sequence-to-sequence 的, 你总能找到办法把新 modality encode 成 sequence element。

第二个 takeaway: **video pretraining 可能是 robotics 的正确 foundation**。比 image-language pretraining 更对齐, 因为 robotics 本质是 physical world 的 sequential prediction 问题, 跟 video generation 是同一个 task family。

第三个 takeaway: **auxiliary supervision (让 policy 同时预测 future state) 是 multimodal action distribution 上的关键**。pure action regression 在 multimodality 上有 fundamental problem, 加 future state prediction 强制 coherent sampling, 是个简单但有效的解。

---

参考链接 (之前列过, 这里再放一次核心几个):

- Cosmos Policy project page: https://research.nvidia.com/labs/dir/cosmos-policy/
- Cosmos World Foundation Model: https://arxiv.org/abs/2501.03575  
- Wan 2.1 VAE: https://arxiv.org/abs/2503.20314
- EDM: https://arxiv.org/abs/2206.00364
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- π0.5: https://arxiv.org/abs/2504.16054
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboCasa: https://arxiv.org/abs/2406.02523

---

# Cosmos Policy 深度解析

Andrej, 这篇是 NVIDIA + Stanford 的工作, 把 Cosmos-Predict2 video foundation model 单阶段 fine-tune 成 robot policy, 同时输出 action / future state / value, 还能用 best-of-N planning。核心 trick 是 **Latent Frame Injection** —— 不改架构, 把新 modality 当作 latent frames 塞进 video diffusion sequence 里。下面我把每个细节都拆开讲。

Project page: https://research.nvidia.com/labs/dir/cosmos-policy/

---

## 1. 核心 Idea 的直觉

Video foundation model 在 millions of internet videos 上学到了 temporal causality, implicit physics, motion patterns。这些 prior 对 robotics 极有价值。VLA 路线 (RT-2, OpenVLA, π0, π0.5) 走的是 image-text pairs pretraining + robot action fine-tuning, backbone 学的是静态语义。Video model 走的是另一条路 —— backbone 本身就是 temporal predictor。

之前 video-based robot policy 工作 (UVA, Video Policy, VIDAR 等) 的痛点:
- 多阶段训练 (先 video fine-tune, 再单独训 action module)
- 引入新架构组件 (separate action diffuser, inverse dynamics model)
- 或者干脆从头训 unified video-action model, 失去 video pretrained prior (UWM)

Cosmos Policy 的主张: **single-stage post-training, zero architectural change**。把 robot proprioception, action chunk, future state images, future state value 全部当作 latent frames 注入到 video diffusion 的 latent sequence 里, 用同一个 EDM denoising objective 训练。video model 本来就在建模高维 multimodal distribution, action distribution 也是 distribution, 直接复用同一套 learning algorithm。

---

## 2. Base Model: Cosmos-Predict2-2B-Video2World

Cosmos-Predict2-2B 是 NVIDIA 的 video foundation model (https://arxiv.org/abs/2501.03575)。结构上是 latent video diffusion:

**Tokenizer**: Wan2.1 spatiotemporal VAE (https://arxiv.org/abs/2503.20314)
- 输入: $(1+T) \times H \times W \times 3$ 的 RGB video sequence (1 是 conditioning image, T 是要预测的 future frames)
- 输出 latent: $(1+T') \times H' \times W' \times 16$, 其中
  - $T' = T/4$ (temporal compression 4x)
  - $H' = H/8$, $W' = W/8$ (spatial compression 8x)
  - 16 latent channels
- 第一帧不参与 temporal compression (单独编码), 这样可以从单张 image condition

**Denoiser**: Diffusion Transformer (DiT, https://arxiv.org/abs/2212.09748)
- Text conditioning via cross-attention (T5-XXL embeddings, https://arxiv.org/abs/1910.10683)
- Noise level σ conditioning via adaptive layer normalization (FiLM, https://arxiv.org/abs/1709.07871)

**Training objective** (EDM formulation, https://arxiv.org/abs/2206.00364):

$$\mathcal{L}(D_\theta, \sigma) = \mathbb{E}_{\mathbf{x}_0, \mathbf{c}, \mathbf{n}} \left[ \| D_\theta(\mathbf{x}_0 + \mathbf{n}; \sigma, \mathbf{c}) - \mathbf{x}_0 \|_2^2 \right]$$

变量解释:
- $\mathbf{x}_0$: clean VAE-encoded image sequence (latent)
- $\mathbf{c}$: text description 编码成 T5-XXL embedding
- $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$: i.i.d. Gaussian noise, 用来 corrupt $\mathbf{x}_0$
- $\sigma$: noise level (scalar), 决定 corruption 强度
- $D_\theta$: diffusion transformer (denoiser), 输入 corrupted latent + σ + c, 输出 clean latent 的估计

**关键机制**: conditioning mask。训练时第一帧对应的 latent frame 保持 clean (不加 noise), 后续 frames 被 noise corrupt。模型学的是 "given clean first frame + noisy future frames, recover clean future frames"。

---

## 3. Latent Frame Injection (核心 Trick)

### 3.1 问题

Base Cosmos-Predict2 只接受 (image, text) 输入, 生成单视角 video。Robotics 需要:
- Robot proprioception 作为 input (joint angles, end-effector pose)
- Action chunk 作为 output
- State value 作为 output (for planning)
- Multi-view camera input

之前的做法是设计新的 module 接入。Cosmos Policy 的做法: **不加 module, 把这些 modality 也编码成 latent frame**, 和 image latent frame 一起送进 diffusion sequence。

### 3.2 序列构造

以 ALOHA 平台为例 (2 个 third-person camera + 1 个 wrist camera), latent sequence 一共 11 个 latent frames:

| Index | 内容 | 角色 |
|-------|------|------|
| 1 | blank placeholder | 占位 (VAE 第一帧特殊处理, 后面解释) |
| 2 | robot proprioception | 当前 state observation (新 modality) |
| 3 | wrist camera image | 当前 state observation |
| 4 | first third-person camera | 当前 state observation |
| 5 | second third-person camera | 当前 state observation |
| 6 | action chunk | 要预测的 action (新 modality) |
| 7 | future robot proprioception | future state (新 modality) |
| 8 | future wrist image | future state |
| 9 | future first third-person | future state |
| 10 | future second third-person | future state |
| 11 | future state value | 要预测的 value (新 modality) |

顺序对应 $(s, a, s', V(s'))$。这个 ordering 很关键, 允许 left-to-right autoregressive decoding: 先 decode action, 再 decode future state, 再 decode value。

### 3.3 编码细节

每个新 modality 都是 low-dim vector, 但 latent frame 是 $H' \times W' \times C'$ 的 volume。处理方法:

以 action chunk 为例, shape 是 $K \times d_{act}$ ($K$ = chunk size, $d_{act}$ = action dim):
1. **Normalize**: 每个 dimension rescale 到 $[-1, +1]$
2. **Flatten**: 变成 $K \times d_{act}$ 长度的 vector
3. **Duplicate**: 复制 $\frac{H' \times W' \times C'}{K \times d_{act}}$ 次
4. **Reshape**: 填回 $H' \times W' \times C'$ 的 volume
5. **Overwrite**: 用这个 volume 替换对应位置的 placeholder latent frame

Proprioception 和 value 同理, 只是初始 shape 不同。Value 是 scalar, 整个 latent volume 就是同一个 scalar 的 normalized 副本。

**推理时反过来**: 从生成的 latent volume 中取所有 duplicate 的 average, un-normalize 回原 scale。注意, 这些 non-image modality 不需要 VAE decode, 直接在 latent space 里 average 就行。这是个很简洁的设计 —— 利用 video model 已经能处理 latent frame 的能力, 让它顺带学一个新的 low-dim distribution。

### 3.4 Placeholder 的细节

Figure 8 揭示了一个微妙的实现细节: VAE tokenizer 把第一帧单独编码, 后续帧按 4 个一组 temporal compress。为了让 current timestep observations 和 future timestep observations 的 latent 结构对称, 作者把 current 和 future observations 都放在 placeholder 之后。同时为了让 "一个 timestep 对应一个 latent frame", 每个 image 复制 4 份作为一组 (对应 temporal compression 的 group of 4)。

这个细节告诉我: latent injection 不是简单的 "插一帧", 而是和 VAE 的 tokenization scheme 紧密耦合的。如果换一个 VAE (比如 CogVideoX 的 3D VAE), 这个 placeholder 和 group-of-4 的处理都要重新设计。

---

## 4. Joint Training Scheme

### 4.1 三个函数, 一套架构

Cosmos Policy 在同一个模型里同时学三个函数:
- **Policy**: $\pi(a, s', V(s') | s)$ — 给 state, 生成 action + future state + value
- **World model**: $\hat{T}(s', V(s') | s, a)$ — 给 state + action, 预测 future state + value
- **Value function**: $V(s' | s, a, s')$ — 给完整 trajectory, 估 value

实现方式: 同一个 latent sequence, 用不同的 **conditioning scheme** (mask 哪些 frame 是 clean conditioning, 哪些是 noisy target) 来训练不同函数。

### 4.2 Batch 分配

每个 training batch 按 50/25/25 划分:

**50% from demonstrations**: 训练 policy
- Conditioning: clean $s$ (frames 1-5)
- Target: noisy $a, s', V(s')$ (frames 6-11) 加噪后 denoise
- 优化 $p(a, s', V(s') | s)$

**25% from rollouts**: 训练 world model
- Conditioning: clean $s, a$ (frames 1-6)
- Target: noisy $s', V(s')$ (frames 7-11)
- 优化 $p(s', V(s') | s, a)$

**25% from rollouts**: 训练 value function
- Conditioning: clean $s, a, s'$ (frames 1-10)
- Target: noisy $V(s')$ (frame 11)
- 优化 $p(V(s') | s, a, s')$

### 4.3 为什么 demo 和 rollout 混着用

Demos 只覆盖成功 trajectory, world model 和 value function 在 demos 上学到的 distribution 太窄。Rollouts (包括失败的) 提供 broader state-action coverage, 让 world model 和 value function 见过失败案例, 才能在 planning 时识别 "这个 action 会导致失败"。

初始时, rollouts dataset 就是 demos 的超集 (加上 failed demos)。LIBERO/RoboCasa 大约 10-20% demos replay 时失败, 这些就成了 rollouts 的一部分。Real-world ALOHA 仔细 teleop, 没有 failed demos, 所以 rollouts == demos。

### 4.4 Auxiliary supervision 的作用

Policy 训练时 target 不只是 $a$, 而是 $(a, s', V(s'))$。World model 训练时 target 不只是 $s'$, 而是 $(s', V(s'))$。这相当于让 policy 同时学 "我做完这个 action 后世界会变成什么样, 最终能拿多少 reward"。

Ablation 数据 (Table 4, LIBERO):
- 完整版: 98.5%
- 去掉 auxiliary losses: 97.0% (-1.5%)
- 从头训: 94.6% (-3.9%)

RoboCasa 上更细的 ablation (Table 5):
- 完整版 (5 denoising steps): 67.1%
- 去掉 value training samples: 66.6%
- 去掉 world model + value training samples: 64.0%
- 再去掉 policy 的 auxiliary value supervision: 62.5%
- **再去掉 policy 的 auxiliary future state supervision (只剩 barebones π(a|s)): 44.4%** (-22.7%!)

最后这个 drop 很惊人。意味着: **让 policy 同时预测 future state 是关键**, 不是 nice-to-have。Intuition: 强迫 model 在 action representation 里编码 "这个 action 会把世界推向什么 state", 这种 representation 比 pure action regression 更结构化, 更能利用 video prior。

### 4.5 Value function 的两种变体

初始训练时, value 预测 conditioned on 完整 $(s, a, s')$, 学的是 $V(s')$。但当 fine-tune on rollouts 准备 planning 时, 可以用 input mask 切换:
- Mask 掉 $(s, a)$, 只看 $s'$ → 学 $V(s')$ (state value), 用于 model-based planning
- Mask 掉 $s'$, 只看 $(s, a)$ → 学 $Q(s, a)$ (state-action value), 用于 model-free planning

实验对比 (Figure 7): model-based $V(s')$ variant 比 model-free $Q(s,a)$ variant 表现更好。作者解释: rollout data 有限, $Q$ 函数 input 维度更高 (包含 image observations), 更容易 overfit; 而 $V(s')$ 借助 world model 提供 dynamics prior, sample efficiency 更高。

---

## 5. Value Function 的数学

在 sparse reward MDP 下, reward 只在 terminal 给:

$$R(s_t, a_t) = 0 \text{ for } t < H, \quad R(s_H, a_H) \in [0, 1]$$

Value function:
$$V^\pi(s) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{k=t}^{H} \gamma^{k-t} R(s_k, a_k) \bigg| s_t = s \right] = \mathbb{E}_{\tau \sim \pi} \left[ \gamma^{H-t} R(s_H, a_H) \bigg| s_t = s \right]$$

变量:
- $\tau$: trajectory, $\tau \sim \pi$ 表示从 policy 采样
- $\gamma \in [0, 1]$: discount factor, 把 terminal reward 沿时间反向传播
- $H$: horizon (terminal timestep)
- $t$: current timestep
- $R(s_H, a_H)$: terminal reward, 任务成功与否

实现用 Monte Carlo return: 每个 transition 标 $\gamma^{H-t} R(s_H, a_H)$。这个标签直接监督 $V(s')$ latent frame 的生成。

---

## 6. Noise Distribution 调整 (很重要的细节)

这是 paper 里我觉得最 "robotics-flavored" 的改动。

### 6.1 问题

原 Cosmos-Predict2 用 EDM 的 log-normal noise schedule:
$$\ln(\sigma) \sim \mathcal{N}(P_{mean}, P_{std}^2), \quad P_{mean} = 1.39, \quad P_{std} = 1.2$$

这个分布在 low σ 处 weight 高, high σ 处 weight 低 (Figure 9 left)。

Diffusion sampling 是从 $\sigma_{max} = 80$ 开始 (纯噪声), 迭代 denoise 到 $\sigma_{min} \approx 0$。每步 model 预测当前 σ 的噪声, 逐步恢复 clean sample。

**对 video generation 没问题**, 因为不精确的初始 denoise 步只影响 high-level structure, 后续步骤会修正。

**对 action generation 致命**: action 必须精确, 小误差导致 robot 灾难性失败。Log-normal 在 high σ 处训练信号不足, 导致初始 denoise 步不准, 后续 cascading errors 累积。

### 6.2 解决

Cosmos Policy 改用 **hybrid log-normal-uniform**:
- 概率 0.7: 从原 log-normal 采样
- 概率 0.3: 从 $\text{Uniform}[1.0, 85.0]$ 采样

效果: 在 high σ 尾部加 weight, 让 model 在起始 denoise 步有足够训练信号 (Figure 9 right)。

0.7/0.3 这个 split 没怎么 tune, 是 "stay close to original while extending tail" 的选择。

### 6.3 推理时 σ_min 调高

原 EDM: $\sigma_{min} = 0.002$ (几乎 0)
Cosmos Policy 推理: $\sigma_{min} = 4$

理由: σ ≈ 0 时 SNR 极低, 最后几步 denoise 反而比中间步骤不准。把 $\sigma_{min}$ 提到 4, 跳过这些不可靠的最后步骤, empirically 降低 action / future state / value 的 L1 loss。

这个 insight 我觉得挺有启发: diffusion 在 robotics 上的应用不能照搬 image generation 的 schedule, 要根据 "哪些 σ regime 对 final precision 重要" 重新设计。

---

## 7. Planning with World Model + Value Function

### 7.1 Dual Deployment

- **Policy model**: 原 Cosmos Policy checkpoint (训在 demos 上), 负责 sample action proposals
- **Planning model**: 在 rollout data 上 fine-tune 的 checkpoint, 负责 predict future state + value

为什么分开? 确保 planning model 是 on-policy 训练的 —— rollout data 由 policy model 收集, planning model 在这个分布上学 dynamics, 评估才准。

Fine-tune 时 batch split 改成 90/10: 90% 给 world model + value function (45% + 45%), 10% 给 policy。Refine world model 和 value 是重点, policy 不动太多。

### 7.2 Best-of-N Search

Algorithm:
1. Policy model sample N 个 candidate action chunks
2. 每个 action, planning model 生成 future state (ensemble of 3 次)
3. 每个 future state, planning model 生成 value (ensemble of 5 次)
4. 每个 action 总共 3 × 5 = 15 个 value 预测
5. **Majority mean aggregation**: 先用固定 threshold 把每个 value 二值化 (success/fail), 看 majority 是 success 还是 failure; 然后在 majority group 内 average
6. 选 highest mean value 对应的 action 执行

为什么不用 naive average? Value 预测 bimodal 或高方差时, outlier 会拉偏 mean。Majority mean 对 outlier 鲁棒。

### 7.3 速度

N=8, 用 8 张 H100 并行 (每个 branch 一张 GPU):
- Action chunk (10 denoising steps)
- 3 个 future state prediction × 5 denoising steps
- 5 个 value prediction × 5 denoising steps per future state
- 总计 4.9 秒 per action chunk

ALOHA 一个 action chunk 是 2 秒 robot execution。所以 planning 的 wall-clock 是 execution 的 2.5 倍, robot 要 pause 5 秒等下一 chunk。这是 paper 自己承认的 limitation —— 不适合 dynamic task。

### 7.4 Planning 实验结果

在 ALOHA "put candies in bowl" 和 "put candy in ziploc bag" 这两个最难的 task 上 (挑战的 initial conditions):
- Base policy (no planning): 基线
- Model-based planning ($V(s')$): +12.5% average score
- Model-free planning ($Q(s,a)$): 略低于 model-based

Figure 6 给了 qualitative 例子: base policy 的 world model 训在 demos 上, 看不到 "抓不住 ziploc slider" 这种失败; fine-tune on rollouts 后, world model 能预测 "这个 grasp 会失败", planning 就避开这种 action。

---

## 8. 实验结果总览

### 8.1 LIBERO (Table 1)

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Dita | 97.4 | 94.8 | 93.2 | 83.6 | 92.3 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| CogVLA | 98.6 | 98.8 | 96.6 | 95.4 | 97.4 |
| **Cosmos Policy** | 98.1 | **100.0** | 98.2 | **97.6** | **98.5** |

训练数据: 每个 task suite 500 demos (10 tasks × 50 demos)。3 random seeds, 6000 trials total。

注意 LIBERO-Object 上 100.0% 是 saturation。LIBERO-Long (long-horizon) 97.6% 比 π0.5 (92.4%) 高 5 个点, long-horizon 最难, 这个 gap 有意义。

### 8.2 RoboCasa (Table 2)

| Method | # Demos/Task | Average SR (%) |
|--------|--------------|----------------|
| GR00T-N1 | 300 | 49.6 |
| UVA | 50 | 50.0 |
| DP-VLA | 3000 | 57.3 |
| GR00T-N1 + DreamGen | 300 + 10000 synthetic | 57.6 |
| GR00T-N1 + DUST | 300 | 58.5 |
| UWM | 1000 | 60.8 |
| π0 | 300 | 62.5 |
| GR00T-N1.5 | 300 | 64.1 |
| Video Policy | 300 | 66.0 |
| FLARE | 300 | 66.4 |
| GR00T-N1.5 + HAMLET | 300 | 66.4 |
| **Cosmos Policy** | **50** | **67.1** |

数据效率碾压: 用 50 demos 打过别人 300-3000 demos。RoboCasa 测试只含 unseen object instances, 5 个 scene 里有 2 个是 unseen style, 所以是 OOD 测试。3 seeds × 24 tasks × 50 trials = 3600 trials。

### 8.3 ALOHA Real-World (Table 3)

| Method | put X on plate | fold shirt | candies in bowl | candy in ziploc | Average |
|--------|----------------|------------|-----------------|-----------------|---------|
| Diffusion Policy | 63.3 | 23.5 | 32.8 | 14.6 | 33.6 |
| OpenVLA-OFT+ | 68.3 | 99.5 | 21.6 | 58.5 | 62.0 |
| π0 | 85.0 | 98.5 | 71.2 | 56.9 | 77.9 |
| π0.5 | 98.3 | 99.5 | 95.2 | 61.5 | 88.6 |
| **Cosmos Policy** | **100.0** | 99.5 | 89.6 | **85.4** | **93.6** |

101 trials 总计, in-distribution + OOD 混合。4 个 task 共 185 demos 训练一个 policy。

注意 ziploc bag 任务: π0.5 只有 61.5, Cosmos Policy 85.4。这个任务需要毫米级 precision 抓 slider, 且 dynamics stochastic (抓的位置差一点就 slip)。Diffusion Policy 和 OpenVLA-OFT+ 在这种高 multimodality + high precision 任务上崩溃。π0.5 抓不住 slider 的右侧 (Figure 5 left), OpenVLA-OFT+ L1 regression 导致它 reach "two candies 中间" 而不是直接抓一个 (Figure 5 right) —— L1 regression 在 multimodal distribution 上 mean 是 mode 之间没意义的平均。

### 8.4 推理 latency

| Denoising steps | Latency (1 H100) | RoboCasa SR |
|----------------|------------------|-------------|
| 5 | 0.61s | 67.1% |
| 1 | 0.16s | 66.4% |

1 步 denoising 只损失 0.5% SR, 速度快 4 倍。这说明 latent diffusion 在 action generation 上不需要多步迭代 —— 1 步就够 capture 主要 mode。这和 recent "consistency model" / "rectified flow" 在 image 上 1 步生成的 trend 一致。

---

## 9. Training Compute

| Benchmark | GPUs | Batch | Steps | Time | Action chunk |
|-----------|------|-------|-------|------|--------------|
| LIBERO | 64 H100 | 1920 | 40K | 48h | 16 |
| RoboCasa | 32 H100 | 800 | 45K | 48h | 32 (execute 16) |
| ALOHA | 8 H100 | 200 | 50K | 48h | 50 (2s @ 25Hz) |

Fully fine-tune (不是 LoRA)。Comparison 用相同 compute budget (8 H100 × 48h): π0/π0.5 训了 400K steps (batch 256), OpenVLA-OFT+ 32K steps (batch 96)。Diffusion Policy 从头训 72K steps (150M params, 远小于其他 2-7B)。

LIBERO 训完后的 single-step L1 loss (不同 σ 下):
- Action: 0.012
- Future proprio: 0.007
- Future wrist image latent: 0.068
- Future third-person image latent: 0.036
- Value: 0.007

Action / proprio / value 学得快 (loss 小), image latent 学得慢。这符合直觉 —— low-dim modality 容易 fit, image latent 高维难。

---

## 10. 我对这篇 paper 的几点思考

### 10.1 Latent injection 的本质

把 low-dim modality duplicate 成 latent volume 是个 hack, 但很巧妙。它利用了一个事实: video diffusion model 已经会处理 "latent frame 内部有 spatial structure" 的 distribution。Duplicate 一个 scalar 到整个 volume 后, 这个 latent frame 内部 spatial 上 constant, 但 model 不需要知道这一点 —— 它就当作一个特殊 pattern 的 latent frame 来 denoise。Extract 时取 average 是因为 duplicate 后所有位置应该一致, average 降噪。

这相当于把 "学一个 action distribution" 的问题转化成 "学一个特殊 image latent frame 的 distribution", 完全复用 video model 的 capacity 和 learning algorithm。

### 10.2 为什么 video prior 比 image-language prior 强

Ablation 数据 (Table 4): 从头训 Cosmos Policy (相同架构, 相同 compute) 掉 3.9% on LIBERO, 掉 18.7% on ALOHA fold shirt。说明 pretrained weight 提供的不是 "随机初始化 + 训练" 能追平的东西。

π0.5 在 ALOHA 上预训练于海量 robot imitation data, OpenVLA-OFT+ 同理。Cosmos Policy 没有这种 robot action pretraining, 但 LIBERO/RoboCasa/ALOHA 都赢。这暗示: video model 学到的 "世界 dynamics" prior, 比 VLA 在 static image-text pair 上学到的 "semantic" prior, 更接近 robotics 需要的东西。Robotics 本质是 sequential decision making in a physical world with temporal causality, video model 正好是这个任务的 pretraining task。

### 10.3 Auxiliary supervision 的深层原因

让 policy 同时预测 $(a, s', V(s'))$ 而不只是 $a$, 这相当于强制 policy 内部 "想象" action 的 consequence。在 barebones ablation 里掉到 44.4% (RoboCasa), 说明这个 consequence prediction 是 policy 表现的支柱。

一个 hypothesis: pure action regression 在 multimodal action distribution 上有 mode collapse 问题 (L1 loss 取 median, L2 取 mean, 都不是 mode)。让 policy 同时预测 $s'$, 等于给它一个 "更结构化的 target" —— 一个特定 mode 的 action 对应一个特定的 $s'$, model 必须从 latent 里 sample 一个 coherent $(a, s')$ pair, 而不能 average 两个 mode 的 action。这是 video diffusion 作为 "joint distribution learner" 的优势。

### 10.4 Best-of-N planning 的开销

5 秒 per action chunk for planning, 这是 real-time control 的瓶颈。Best-of-N 本质是 brute-force search, 没用 MPC / tree search 的轨迹优化。作者 future work 提到 "extending prediction horizon and planning to greater depths" —— 这是 model-based RL 经典方向 (Dreamer, TD-MPC)。但 Cosmos Policy 的 world model 只 predict 一步 future state ($t+K$ 一步, 不是 $t+1, t+2, ...$ 序列), multi-step horizon 要么 autoregressive roll (误差累积), 要么改训练 target (predict 多步)。这都是开放问题。

### 10.5 1 denoising step 的 implication

1 步生成只掉 0.5%, 这说明 action latent 的 distribution 在 video model 里学得非常 "well-formed" —— 一步 denoise 就能跳到 mode 附近。这和 image diffusion 需要 50-1000 步形成鲜明对比。可能原因:
- Action 是 low-dim (K × d_act), distribution 复杂度有限
- Action chunk 在 latent 里 duplicate 成 volume, 等于高度 redundant encoding, 1 步够 recover
- Latent injection 的 target 是 deterministic duplicate, 训练时 noise 加在 duplicate volume 上, model 学的是 "如何 denoise 一个 spatially-correlated noise pattern", 这比自然 image 的 noise pattern 简单

这个 finding 对未来 robotics diffusion policy 设计有启发: action chunk 的 diffusion 可能不需要多步, 1-step / few-step distillation 应该是 default。

---

## 11. 相关工作链接

Base model:
- Cosmos World Foundation Model: https://arxiv.org/abs/2501.03575
- Wan 2.1 VAE: https://arxiv.org/abs/2503.20314
- EDM: https://arxiv.org/abs/2206.00364
- DiT: https://arxiv.org/abs/2212.09748
- FiLM: https://arxiv.org/abs/1709.07871
- T5: https://arxiv.org/abs/1910.10683

Comparison methods:
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ALOHA: https://arxiv.org/abs/2304.13705
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- CogVLA: https://arxiv.org/abs/2508.21046
- UniVLA: https://arxiv.org/abs/2505.06111
- UVA: https://arxiv.org/abs/2503.00200
- UWM: https://arxiv.org/abs/2504.02792
- Video Policy: https://arxiv.org/abs/2508.00795
- FLARE: https://arxiv.org/abs/2505.15659
- GR00T N1: https://arxiv.org/abs/2503.14734
- DreamGen: https://arxiv.org/abs/2505.12705

Benchmarks:
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboCasa: https://arxiv.org/abs/2406.02523
- MimicGen: https://arxiv.org/abs/2310.17596

Model-based RL foundations:
- Dyna (Sutton 1991): https://dl.acm.org/doi/10.1145/122344.122377
- MBPO: https://arxiv.org/abs/1906.08253
- TD-MPC: https://arxiv.org/abs/2203.04955
- TD-MPC2: https://arxiv.org/abs/2310.16828
- Dreamer: https://arxiv.org/abs/1912.01603
- DreamerV3: https://arxiv.org/abs/2301.04104

---

## 12. 总结

Cosmos Policy 的核心贡献:

1. **Latent frame injection** —— 不改架构, 把所有 modality 当 latent frame 注入 video diffusion, 复用 video model 的 learning algorithm 学 action distribution。
2. **Single-stage post-training** —— 不需要 multi-stage, 不需要 separate action module。
3. **Joint policy + world model + value** —— 同一架构, conditioning scheme 决定训哪个函数。Auxiliary supervision (predict $s'$ alongside $a$) 是性能支柱。
4. **Noise schedule 调整** —— hybrid log-normal-uniform 训练, σ_min=4 推理, 让 action generation 精确。
5. **Rollout-based planning refinement** —— best-of-N + majority mean ensemble, model-based $V(s')$ 比 model-free $Q(s,a)$ sample-efficient。

Limitations paper 自己列了:
- Planning 慢 (5s per action chunk), 不适合 dynamic task
- 需要大量 rollout 才能 refine world model
- 只有 1 层 search tree, 没 multi-step horizon

我觉得这篇 paper 最值得借鉴的是 latent injection 这个设计哲学: **当你有一个强大的 generative model, 想给它加新 modality, 不要加新 module, 把新 modality 也表示成 model 已经会处理的 format**。这个 idea 应该可以推广到其他 foundation model adaptation 场景。
