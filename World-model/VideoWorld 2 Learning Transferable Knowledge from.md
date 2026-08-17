---
source_pdf: VideoWorld 2 Learning Transferable Knowledge from.pdf
paper_sha256: 84cd30bea522cb45732c5c274e5c7ecade4f64162d6635001b8eacb6840432a8
processed_at: '2026-08-13T01:01:55-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VideoWorld 2

---

## 一句话总结

让 AI 看 video 学本事, 最大的坑是模型把 "桌子长啥样" 跟 "手怎么动" 搅在一起了。VideoWorld 2 的办法是: 找一个已经很会画画的模型 (pretrained VDM) 负责外观, 让 latent code 只管 "动作是什么", 这样学到的东西就能搬到新环境用。

---

## 问题是什么

你给一个 model 看折纸飞机的视频, 然后让它换个桌面的纸重新折一遍。

人看这个视频, 脑子里记住的是 "把左角折到右边、对折、压平" 这种 action。桌面是木头还是塑料, 纸是红色还是蓝色, 你根本不在意, 换了照样折。

但 current models 做不到。Wan2.2 14B、HunyuanVideo 13B 这些 SOTA video generation models, 你 fine-tune 它, 给它每一步的 text instruction, 它在 step 1 能做到 81% success rate, 但到 step 4 就掉到 10%, step 5 以后直接 0%。

https://VideoWorld2.github.io/

为什么? 因为 model 分不清 "appearance" 和 "dynamics"。它的 latent representation 把桌面纹理、光线、纸张颜色全 encode 进去了, 换个环境它就懵了。

---

## VideoWorld 1 为什么不 work

VideoWorld 1 (CVPR 2025, https://arxiv.org/abs/2505.18034) 在 synthetic environment (Go game、simulated robot) 上能 work, 因为那些环境 appearance 极其简单且 consistent。

它的核心设计是 Latent Dynamic Model (LDM): 用 MAGVITv2-style causal codec, 把 video 压成一组 latent codes $z = \{z_k^n\}$, $k$ 是时间步, $n$ 是 query index。然后用 first frame feature $f_0$ + latent codes 重建后续 frames, loss 是 L2 reconstruction。

问题就出在这个 reconstruction loss 上。只要你让 decoder 去 reconstruct RGB pixels, latent codes 就会被迫 encode 所有 visual information, 包括 task-irrelevant 的 appearance details。在 synthetic environment 这个问题不明显, 到 real-world 就崩了。

Table 1 里 VideoWorld 1 在 paper folding step 7 的 success rate 是 0%, block building 最好也才 33.9%。

---

## VideoWorld 2 的核心 idea

非常 simple 的一个 insight: **不要让 latent codes 干 appearance 的活, 找个 expert 来干**。

这个 expert 就是 pretrained VDM (Cosmos DiT 2B, https://www.nvidia.com/en-us/ai/cosmos/)。VDM 在 massive video data 上 pretrain 过, 它天生就是 appearance generation 的专家。

所以 dLDM (dynamics-enhanced Latent Dynamics Model) 的分工是:
- **Latent codes**: 只 encode "what action is happening" (手怎么动、纸怎么折)
- **VDM**: 负责 "what does it look like" (光线、纹理、颜色)

VDM 自己会 fill in realistic appearance, 你只要告诉它 action 是什么就行。

---

## 怎么实现这个分工

三个 key tricks, 我一个一个讲:

### Trick 1: Stop gradient

VQ-VAE decoder 的 gradient 不流回 latent codes $z$。

为什么? 因为如果让 decoder 的 L2 loss 去 update $z$, $z$ 就会被 incentivize 去 encode appearance details (decoder 要 reconstruct 所有 visual info)。Stop gradient 之后, $z$ 只被 VDM 的 objective 优化, VDM 是 appearance 专家, 它用自己 internalized 的 prior 去 generate 细节, 不需要从 $z$ 里读取。

Table 3a row 2 (no stop-grad) vs row 3 (stop-grad): success rate 从 30.3% 跳到 47.3%, 差了 17%。

### Trick 2: ControlNet-like motion conditioning

VDM 从 latent codes + noise 直接生成 video 会很慢且 motion 不准, 因为 VDM 没见过 paper folding 这种 task, 不知道手应该怎么动。

Solution: 复用 VQ-VAE decoder 的输出作为 condition 喂给 VDM。Decoder 输出是 low-fidelity 但 motion-rich 的, 提供了 coarse motion scaffolding。VDM 只需要在这个 scaffolding 上 refine appearance。

这个 decoder 输出通过 ControlNet-style branch (https://arxiv.org/abs/2302.05543) 注入 VDM, gradient 是 stopped 的。

Table 3a row 3 (no ControlNet) vs row 5 (ControlNet): success rate 从 47.3% 到 68.8%, LPIPS 从 0.275 到 0.205。Motion conditioning 既帮 task performance 又帮 visual quality。

### Trick 3: Causal attention in VDM

生成第 $t$ 帧的时候, VDM 只能 attend 到 time ≤ t 的信息, 不能偷看未来。这是 AR video generation 的标准做法, 但这里特别重要因为我们在做 long-horizon planning, 任何 future leakage 都会让 evaluation 失效。

Table 3c ablation: causal cross-attention vs non-causal cross-attention, 69.8% vs 52.0%, 差了 17.8%。

---

## 训练流程

Paper Sec 5.1 提到一个 warmup strategy, 很关键:

**Phase 1 (warmup)**: 只用 original LDM reconstruction loss 训 latent codes + decoder。让 latent codes 学会 compress dynamics, decoder 学会 reconstruct low-fidelity motion。这个阶段不用 VDM。

**Phase 2 (disentangled)**: Switch 到 full dLDM, 用 VDM 做 appearance generation。因为 decoder 已经能 produce coherent motion cues, VDM 训练就稳定了。

如果不 warmup, VDM 从 random noise 开始 denoise, 训练不稳定。

Training loss:

$$\mathcal{L} = \text{MSE}(\text{rec}, \text{video}) + \text{VDM}(\text{video}, z, \text{rec})$$

- $\text{rec}$: VQ-VAE decoder 的 low-fidelity reconstruction
- $\text{video}$: ground truth
- $z$: quantized latent codes (via FSQ, levels [8,5,5,5], codebook size 1000)
- 第一项: MSE reconstruction, 但 gradient stopped to $z$
- 第二项: VDM denoising loss, conditioned on $z$ + decoder output

---

## AR Transformer 学 long-horizon policy

dLDM 把 video 压成 latent codes sequence, 然后用 Cosmos AR 4B (https://arxiv.org/abs/2501.03575) 做 next-token prediction。

给定 video $x_{0:T}$, dLDM 提取 $\{z_k^n\}_{k=1,n=1}^{K,N}$, flatten 成 sequence, transformer 预测 next latent code, conditioned on first frame $x_0$ + task instruction。

Inference 的时候: 给一张 unseen environment 的 image → transformer 预测 future latent codes → dLDM decode 成 video。因为 latent codes 只 encode dynamics 不 encode appearance, 同一套 codes 可以应用到新环境, VDM 负责填新环境的 appearance。

关键 hyperparameters:
- **Codebook size**: 1000 (FSQ levels [8,5,5,5])。太小不够表达, 太大 encode noise。Table 3d: 8→20.1%, 1000→68.8%, 64000→29.4%, inverted U-shape。
- **Query length N**: 4。Table 3b: N=1→41.9%, N=4→68.8%, N=8→65.0%。
- **Context length T**: 93 frames (~5s @ 16fps)。Table 3e: T=2→19.1%, T=93→68.8%。Long-horizon task 需要 long context。

---

## Benchmark: Video-CraftBench

5 个 long-horizon handcraft tasks:
- Paper airplane folding (40-80s)
- Paper boat folding (40-80s)
- Block tower (20-30s)
- Block horse (20-30s)
- Block person (20-30s)

总共 ~7 hours, ~9.5k clips。Test set ~150 videos, 用没见过的 backgrounds, paper textures, block arrangements。

Evaluation 有两个维度:
1. **Sequential task success rate**: paper folding 拆成 7 个 key steps, 训练 DINOv2-based classifier 检测 completion。一个 step 算成功 only if 所有 preceding steps 都完成。这 measure 的是 long-horizon 累积成功率。
2. **Visual quality**: LPIPS + SSIM

Classifier 的设计很 clever: 只评估 action correctness, 忽略 appearance consistency。Appearance drift 单独用 LPIPS/SSIM 评估。这样能分离两个维度。

---

## 结果有多好

### Video-CraftBench (Table 1)

只 train on Video-CraftBench (没有 OpenX pretraining):

| Method | Paper Step 7 | Block Avg | LPIPS |
|--------|-------------|-----------|-------|
| Wan2.2 14B | 0.0% | 38.7% | 0.237 |
| HunyuanVideo 13B | 0.0% | 33.6% | 0.255 |
| Moto | 0.0% | 10.5% | 0.394 |
| AdaWorld | 0.0% | 16.3% | 0.378 |
| VideoWorld 1 | 0.0% | 28.5% | 0.351 |
| **VideoWorld 2** | **68.8%** | **77.5%** | **0.205** |

VideoWorld 2 从 0% 到 68.8%, 这就是 disentanglement 的威力。

加上 OpenX pretraining:

| Method | Paper Step 7 | Block Avg |
|--------|-------------|-----------|
| CoLA | 40.2% | 47.3% |
| VideoWorld 1 | 31.9% | 52.7% |
| **VideoWorld 2** | **72.3%** | **85.8%** |

### CALVIN (Table 2)

Cross-domain: pretrain on OpenX (1.3M trajectories), finetune on CALVIN (22k):

| Method | Step 5 | Avg Len |
|--------|--------|---------|
| Oracle (full labels) | 23.0% | 2.46 |
| Video pretrain | 30.7% | 2.46 |
| LAPA | 27.0% | 2.51 |
| **VideoWorld 2** | **47.5%** | **2.88** |

VideoWorld 2 在 long-horizon 上大幅领先, 平均完成 2.88 个任务 vs baseline 2.46。

---

## 为什么 work: UMAP 可视化

Fig 7 是最有说服力的 visualization。

从 CALVIN 和 Bridge 各 sample 4000 trajectories, label robot arm action (up/down/left/right), 对 latent codes 做 UMAP。

- **VideoWorld 2**: 同一 action 的 codes 在不同 environment 紧密 cluster, 跨 environment 一致
- **VideoWorld 1**: 同一 action 的 codes 在不同 environment 显著 diverge, 无法 cluster

这就是 transferability 的来源: disentanglement 让 latent codes 跨 environment 一致。VideoWorld 1 的 codes 被 environment-specific appearance 污染, 所以无法 transfer。

---

## 我的几点直觉

### 1. 这其实是 "分工" 的思想

不要让一个 representation 干所有的事。Appearance 有 expert (VDM) 管, dynamics 有 latent codes 管。这跟人类认知很像 —— 你记住的是 action sequence, 不是 pixel sequence。

### 2. Stop gradient 是关键

这个 trick 看起来简单, 但背后 insight 很深: **reconstruction objective 本身就是 entanglement 的来源**。只要你让 latent codes 对 reconstruction 负责, 它就会 encode 所有 visual info。Stop gradient 切断了这个 pressure, latent codes 才能专注 dynamics。

### 3. Long-horizon 需要 long context

Table 3e 的 T=2 vs T=93 对比非常 striking。LAPA-style 的 short-horizon latent action 在 long-horizon task 上根本不够。这跟 LLM 里 context length 的重要性一致 —— 你要看很多 step 才能理解 task structure。

### 4. Scaling 不等于 architecture-agnostic

Wan2.2 14B 参数量是 VideoWorld 2 (Cosmos AR 4B + Cosmos DiT 2B) 的 2 倍多, 但 task performance 完全不如。说明 naive scaling video generation 不会自动给你 world knowledge。Architecture 的 inductive bias 很重要。

### 5. 这跟 LeCun 的 JEPA 形成有趣对比

LeCun 一直说 pixel-level reconstruction 是 wasteful 的, 推 JEPA-style abstract prediction。VideoWorld 2 用 VDM 处理 appearance, latent codes 处理 dynamics —— dynamics 部分类似 JEPA (avoid pixel prediction), appearance 部分是 generative。可能是两个 paradigm 的 sweet spot。

https://arxiv.org/abs/2506.09985 (V-JEPA 2)

---

## Limitations

Paper 没有详细讨论, 但 Appendix C 末尾提了:

> "the VDM processes only 93 frames at a time, while full task sequences can span thousands of frames. Long videos are therefore generated auto-regressively... Because the VDM's inherent reconstruction noise accumulates over time, visual artifacts such as lighting, texture, or color shifts may gradually appear."

Autoregressive long-horizon generation 会 accumulate visual artifacts。Task-relevant dynamics 通过 latent codes 保持准确, 但 visual quality 会 degrade。这有点像 LLM 的 exposure bias。

其他我看到的:
- VDM choice 的 dependency (换 Sora/Veo 会怎样?)
- VDM 需要 task-specific finetune (Table 3f: full finetune > frozen), 降低 zero-shot transferability
- Latent codes 还是 black-box, interpretability 不够
- Discrete codes (FSQ) 对 continuous control 可能 insufficient

---

## 给你的 take-away

Andrej, 我觉得这篇 paper 的 deep insight 就一句话: **disentanglement via prior 比 disentanglement via regularization 更 effective**。

与其设计 clever auxiliary loss 去 force disentangle, 不如直接找一个 pretrained expert 干其中一个 modality 的活。这跟你 "let data do the work" 的哲学一致 —— VDM 的 appearance prior 是从 massive data 里学来的, 比任何 hand-crafted regularization 都强。

而且这个 idea 可能 generalize 到其他 disentanglement 问题: 比如要 disentangle "identity" 和 "expression" (face generation), 找一个 pretrained identity expert 处理 identity, 让另一个 module 只管 expression。

Future direction 我最 excited 的: **把 latent dynamics codes 作为 universal action space 接到 VLA model 上**。Current VLA 用 language-conditioned action prediction, 如果用 dLDM 的 latent codes 作为 action representation, 可能更 sample-efficient, 更 transferable, 跨 embodiment 跨 task。这跟你 "let the data define the program" 的哲学也契合。

你觉得这个方向 worth pursuing 吗?

---

# VideoWorld 2: 从 Real-world Videos 学习 Transferable Knowledge 的深度解析

Andrej, 这篇 paper 我觉得非常对你胃口, 因为它本质上是在问一个非常 fundamental 的问题:**agents 能不能从 raw visual signal 里直接 learn 出可迁移的 world knowledge**, 而完全 bypass language supervision。这其实呼应了你一直强调的 "the bitter lesson" 在 embodied AI 上的延伸 —— 我们想要的是 scalable 的 knowledge acquisition 机制, 而 video 是 internet 上最 abundant 的 modality。

项目主页: https://VideoWorld2.github.io/

---

## 1. Problem Statement: 为什么这件事难

VideoWorld 1 (CVPR 2025, [51]) 在 synthetic domains (Go game records + simulated robotics) 上验证了一个非常 elegant 的 idea: 把 video 当成 demonstration trajectory, 用 autoregressive generation 范式来 learn policy。但在 real-world videos 上, VideoWorld 1 直接崩溃 —— 生成的 hand poses 扭曲, object shapes 错乱, environment appearance drift, 长程 sequence 完全 fail。

Paper 在 Sec 5.3 的 Table 1 里给出了非常 striking 的数据: VideoWorld 1 在 paper folding 第 4 step 的 success rate 直接掉到 21.3%, 第 5 step 之后就是 0.0%。即便是 SOTA 的 video generation models (Wan2.2 14B, HunyuanVideo 13B, Cosmos AR 4B, Cosmos DiT 2B) 在 fine-tuned on Video-CraftBench + detailed text annotations 的条件下, 第 4 step 也都掉到 ≤10.6%。

这个观察非常关键, Karpathy 你肯定会有共鸣: **video generation fidelity 和 task-relevant dynamics learning 是两件完全不同的事**。Wan2.2 14B 在 LPIPS 上达到 0.237, visual quality 最好, 但 task success rate 在 long-horizon 上同样崩溃。这说明 photorealism 并没有给模型带来 task understanding。

### 我的 intuition building

人类看折纸视频的时候, 自动 filter 掉桌面纹理、光线变化、纸张颜色、相机抖动, 只提取 "手把纸的这一角折到对面" 这种 essential action。Current models 做不到这种 attentional filtering, 它们把 visual appearance 和 action dynamics entangle 在一起, 导致 latent representation overfit 到 appearance details。换一个桌面、换一种纸, 模型就 confused。

这背后的 deep reason 我觉得是: reconstruction objective 本身就 incentivize latent codes 去 encode 所有可观测的 visual information, 包括 task-irrelevant 的部分。只要你用 L2/MSE/VAE reconstruction loss, latent code 就一定会被 "污染"。

---

## 2. VideoWorld 1 回顾: Latent Dynamic Model (LDM)

要理解 VideoWorld 2 的创新, 先要理解 VideoWorld 1 的 LDM。Paper Sec 3.1 给了公式:

$$\mathcal{G} = \langle \mathcal{X}, \mathcal{A}, \rho \rangle$$

这里:
- $\mathcal{X}$: observation space (RGB frames)
- $\mathcal{A}$: action space (这里没有显式 action label, 是 implicit 的)
- $\rho$: video generator, 也就是 policy model $\pi(\cdot | x_{0:t}): \mathcal{X} \to \mathcal{A}$

LDM 的核心 idea 是 MAGVITv2-style ([76], https://magvit.github.io/) causal codec。给定一个长度为 $T$ 的 clip $x$, encoder 输出 feature sequence $f_{0:K}$, 其中:

$$K = 1 + \lfloor \frac{T-1}{s} \rfloor$$

- $T$: input clip 的 frame 数
- $s$: temporal downsampling stride
- $K$: compressed 之后的时间步数

然后定义 $N$ 个 learnable query embeddings $q = \{q^n\}_{n=1}^N$, 通过 cross-attention 从 $\{f_{0:k}\}_{k=1}^K$ 中提取 change information, 得到 continuous representation:

$$z = \{z_k^n\}_{k=1,n=1}^{K,N}$$

- $z_k^n$: 第 $k$ 个时间步的第 $n$ 个 latent code
- $k \in [1, K]$: 时间索引
- $n \in [1, N]$: query embedding 的索引

这些 codes 经过 quantization (防止 shortcut copy) 之后, 配合 first frame feature $f_0$, 用 causal decoder 重建后续 frames。Training objective 是 $\ell_2$ reconstruction loss。

**这个设计的核心 insight**: latent codes 只编码 "变化" 而不编码 "静态信息", 因为 first frame $f_0$ 已经包含了 appearance。这有点像 residual coding 的思想, 但 query-based cross-attention 是 learned 的而不是 hand-crafted 的。

VideoWorld 1 在 Go game 和 synthetic robotics 上 work, 是因为这些环境的 appearance 极其简单且 consistent。但 real-world 一上来, latent codes 就被 "污染" 了 —— 它们开始 encode 桌面木纹、纸张颜色、光线变化等 task-irrelevant information。

---

## 3. VideoWorld 2 的核心创新: dynamics-enhanced LDM (dLDM)

Paper Sec 3.2 给出了关键 insight, 我直接引用: **"the insuficient disentanglement of action dynamics and visual appearance"**。

dLDM 的做法非常 clever, Karpathy 你应该会喜欢这种 "用 prior 来 disentangle" 的思路:

### 3.1 架构 overview

dLDM 由四部分组成 (paper Sec A, Alg 1):
1. **Causal encoder**: causal 3D CNN, 提取 visual features $f$
2. **Learnable queries (Q-former style)**: 通过 cross-attention 提取 visual changes, 产生 latent dynamics codes $z$
3. **VQ-VAE decoder**: 从 $z$ + first frame feature 重建 low-fidelity motion-rich frames
4. **Pre-trained VDM (Cosmos DiT 2B)**: 接收 first frame + decoder 输出 + latent codes, 生成 high-fidelity video

### 3.2 为什么用 VDM 作为 appearance prior

这是 paper 最核心的 insight, 我想要更详细地 unpack:

**问题**: 如果只有 LDM + VQ-VAE decoder, latent codes 必须编码足够的 information 让 decoder 能重建 RGB pixels。这个 reconstruction pressure 会 force latent codes 去 encode appearance details。

**Solution**: 把 appearance modeling offload 给一个 pretrained VDM。VDM 已经在 massive video data 上 pretrained (Cosmos DiT 2B, https://www.nvidia.com/en-us/ai/cosmos/), 它天生就是 appearance generation 的 expert。我们只需要让 latent codes encode "what action / dynamics is happening", VDM 自己会 fill in realistic appearance。

这其实是一个分工的设计:
- **Latent codes**: 负责 task-relevant dynamics (手怎么动、纸怎么折、object 怎么 displace)
- **VDM**: 负责 appearance generation (光线、纹理、颜色、细节)

### 3.3 训练 objective

Paper Alg 1 给了 PyTorch-style pseudocode, 训练 loss 是:

$$\mathcal{L} = \text{MSE}(\text{rec}, \text{video}) + \text{VDM}(\text{video}, z, \text{rec})$$

- $\text{rec}$: VQ-VAE decoder 的 low-fidelity reconstruction
- $\text{video}$: ground truth video
- $z$: quantized latent codes
- 第一项: 标准 LDM reconstruction loss, 让 latent codes 能 reconstruct coarse motion
- 第二项: VDM 的 denoising loss, conditioned on latent codes + decoder output

注意一个关键 trick: **stop gradient**。Paper Sec 3.2 强调:

> "we stop the gradient flow of the decoder to the latent codes to prevent the introduction of irrelevant noise"

也就是说, decoder 的 gradient 不会流回 latent codes $z$。为什么? 因为如果让 decoder 的 reconstruction loss 去 update $z$, $z$ 就会被 incentivize 去 encode appearance details (因为 decoder 试图 reconstruct 所有 visual info)。Stop gradient 让 $z$ 只被 VDM 的 objective 优化, 而 VDM 是 appearance expert, 它会用自己 internalized 的 appearance prior 去 generate 细节, 不需要从 $z$ 里读取。

Ablation Table 3a 的 row 2 vs row 3 给了 quantitative 验证: stop gradient 带来 ~20% success rate 的提升。

### 3.4 ControlNet-like motion conditioning

这是另一个关键 trick, paper Sec 3.2 写得比较隐晦, 我来展开:

直接让 VDM 从 latent codes (通过 cross-attention) + noise 生成 video 会很慢且 motion 不准确, 因为 VDM 没见过 target task (paper folding), 它不知道手应该怎么动。

**Solution**: 复用 VQ-VAE decoder 的输出作为 ControlNet-style condition (https://arxiv.org/abs/2302.05543) 给 VDM。这个 decoder 输出是 low-fidelity 但 motion-rich 的, 提供了 coarse temporal cues (手在哪里移动, 物体在哪里 displace)。VDM 只需要在这个 motion scaffolding 上 refine appearance。

这个 design 在 Table 3a row 4 vs row 5 里有 ablation: 只用 stop-grad decoder (row 3, success rate 47.3%) vs 用 ControlNet-like condition (row 5, success rate 68.8%), 差了 ~20%。在 paper folding 这种 long-horizon task 上效果尤其明显, 证明 motion conditioning 对长程生成至关重要。

### 3.5 Causal attention in VDM

Paper Sec 3.2 提到:

> "we enforce causal attention in the VDM so that features at time t attend only to information up to time t"

这是为了防止 information leakage —— 生成第 $t$ 帧的时候, VDM 不能 "偷看" 未来帧的信息。这是 AR video generation 的标准做法, 但在这里特别重要, 因为我们在做 long-horizon planning, 任何 future leakage 都会让 evaluation 失效。

### 3.6 Latent codes 注入 VDM

Latent codes 通过 projection layer (MLP + causal self-attention) 和 causal cross-attention 注入 VDM (paper Sec 3.2, Table 3c ablation)。

Table 3c 显示:
- MLP + cross attention: Paper 52.0, Block 61.3
- + self attention: 52.3, 61.8 (微小提升)
- MLP + causal cross: 69.8, 78.6 (显著提升)
- + self + causal cross: 72.3, 80.9 (最佳)

这说明 causal cross-attention 是关键, 它确保 generation 严格依赖当前 time step 的 latents。

---

## 4. AR Transformer: Long-horizon Policy Learning

### 4.1 训练

Paper 用 NVIDIA Cosmos AR 4B (https://arxiv.org/abs/2501.03575) 作为 AR transformer。给定 video $x_{0:T}$, dLDM 提取 latent codes $\{z_k^n\}_{k=1,n=1}^{K,N}$, flatten 成 sequence, train transformer 做 next-token prediction, conditioned on first frame $x_0$ + task instruction。

这里一个有意思的点: **codebook size 通过 FSQ (Finite Scalar Quantization, [8], https://arxiv.org/abs/2309.15564) 控制, 默认 levels [8, 5, 5, 5], 总 vocabulary = 1000**。

Table 3d 的 ablation 显示:
- codebook size 8: Paper 20.1
- codebook size 1000: Paper 68.8 (最佳)
- codebook size 4096: Paper 50.4
- codebook size 64000: Paper 29.4

这个 inverted U-shape 非常有意思。太小不够表达, 太大模型会 encode noise。VideoWorld 1 paper 里也有类似 finding, 这里在 real-world task 上更明显, 因为 real-world 的 noise 更多。

### 4.2 Query embedding length N

Table 3b ablation:
- $N=1$: Paper 41.9
- $N=2$: Paper 55.1
- $N=4$: Paper 68.8 (最佳)
- $N=8$: Paper 65.0 (开始下降)

$N$ 越大, latent codes 能 encode 更多 information, 但也会 encode 更多 noise, 同时增加 AR transformer 的 sequence length。$N=4$ 是 sweet spot。

### 4.3 Inference: zero-shot transfer

Paper Sec 3.2 最后一段和 Fig 3 right 描述了 inference:
1. 给一张 unseen environment 的 input image
2. AR transformer 预测 future latent dynamics codes
3. dLDM 把 codes decode 成 task execution video

这就是 transferability 的核心: 因为 latent codes 只 encode dynamics 不 encode appearance, 同一套 codes 可以应用到新 environment, VDM 负责 fill in 新 environment 的 appearance。

---

## 5. Video-CraftBench: Benchmark 设计

### 5.1 Dataset 构造

5 个 long-horizon handcraft tasks:
- Paper airplane folding
- Paper boat folding
- Block tower
- Block horse
- Block person

总计 ~7 hours, split into ~9.5k clips。Paper folding 40-80s, block building 20-30s。Test set (~150 videos) 用 training set 没见过的 backgrounds, paper textures, block arrangements。

Appendix B 给了更多 stats:
- 时长分布: 37.3% 在 45-60s, 27.1% 在 60-90s, 短 task (20-30s) 只占 10.9%
- Task type: paper folding 5.2 hours, block building 1.8 hours

这个 benchmark 设计很好, Karpathy 你应该 appreciate —— 它强调 long-horizon, 强调 fine-grained manipulation, 强调 generalization 到 unseen environment。

### 5.2 Evaluation: Sequential task success rate

Paper Sec 4.2 定义了 7 个 key steps for paper folding (Fig 6), 训练一个 DINOv2-based classifier 来 detect completion。这个 classifier 的 design 很 clever:
- 基于 DINOv2-Base (86M params, https://arxiv.org/abs/2304.07193), 因为 DINOv2 有 strong geometric awareness
- 训练数据 ~25k frames (15k from train/test + 10k from model-generated verified trajectories)
- Test accuracy 96.1%
- **关键**: classifier 只评估 action correctness, 忽略 appearance consistency (appearance drift 单独用 LPIPS/SSIM 评估)

这种分离评估很重要: 一个 model 可能 visual quality 很好但 action 完全错, 或者 action 对但 visual quality 差。我们需要分别 measure 两个维度。

Sequential success rate 的定义: 一个 step 算成功 only if 所有 preceding steps 都完成。这 measure 的是 long-horizon 累积成功率, 而不是 single-step accuracy。

---

## 6. 实验结果深度分析

### 6.1 Video-CraftBench 主结果 (Table 1)

让我详细看这个 table, 这里有很多 insight:

**Pre-trained video generation models (row 1-4)**:
- Wan2.2 14B (最强): Step 1: 81.2%, Step 2: 75.0%, Step 3: 30.4%, Step 4: 10.6%, Step 5+: 0.0%
- 即便提供 detailed text annotations for each step (用 Qwen2.5-VL 72B 生成), 这些 models 还是 fail
- Visual quality: SSIM 0.719, LPIPS 0.237 (最好)

Insight: text conditioning 帮助 short-horizon, 但 long-horizon 还是崩溃。说明 single-step text-conditioned generation 缺少 long-horizon policy learning。

**Latent action models (row 6-8)**:
- LAPA [72]: N.A. (因为 structural constraints 导致 long-horizon decoding 严重 degrade)
- Moto [15]: Step 1: 19.1%, Step 2: 11.7%, Step 3+: ~0
- AdaWorld [24]: Step 1: 43.6%, Step 7: 0.0%
- VideoWorld [51]: Step 1: 70.3%, Step 4: 21.3%, Step 5+: 0.0%

Insight: 这些 latent action models 都 struggle with long-horizon。Moto 用 pretrained vision encoder 提 dynamics, 但在 real-world 复杂 dynamics 上 fail。AdaWorld 用 auxiliary diffusion head, 但 latent codes 还是 overfit appearance。

**VideoWorld 2 (row 9, training on Video-CraftBench only)**:
- Step 1: 97.2%
- Step 4: 83.3%
- Step 7 (final): 68.8%
- Block tasks: 70.0-81.5%
- Visual quality: SSIM 0.770, LPIPS 0.205

这个结果非常 striking: **相比于 VideoWorld 1, VideoWorld 2 在 final step 上从 0.0% 提升到 68.8%**, 这是一个巨大的 jump。而且 visual quality 也是最好的。

**Adding OpenX pretraining (row 10-15)**:
- CoLA [62]: Step 7: 40.2%
- VideoWorld: Step 7: 31.9%
- VideoWorld 2: Step 7: 72.3%, Block: 74.0-85.8%

OpenX pretraining 帮 VideoWorld 2 进一步提升, 同时也让 baseline 改善, 但 VideoWorld 2 依然大幅领先。这证明 latent codes 的 transferability —— OpenX 的 robotic manipulation knowledge 可以 transfer 到 paper folding 这种 handcraft task。

### 6.2 CALVIN 结果 (Table 2)

CALVIN (https://arxiv.org/abs/2112.03230) 是 long-horizon robot manipulation benchmark, 34 tasks, 5-task sequential evaluation。

**In-domain latent pretraining (Idx 1-4)**:
- Oracle (22k trajectories + GT action): 80.9 / 55.6 / 44.5 / 31.3, Avg Len 2.36
- Baseline (2k data only): 50.5 / 35.4 / 20.1 / 5.2, Avg Len 1.11
- LAPA + 22k latent pretrain + 2k finetune: 74.4 / 45.8 / 25.2 / 15.3, Avg Len 1.49
- VideoWorld 2 + 22k latent pretrain + 2k finetune: 75.8 / 47.9 / 31.8 / 20.4, Avg Len 1.87

Insight: latent pretraining 显著提升 sample efficiency。VideoWorld 2 在 long-horizon 上比 LAPA 更好 (Avg Len 1.87 vs 1.49)。

**Cross-domain latent pretraining (Idx 5-7)**:
- Video next-token pretrain (OpenX → CALVIN): 85.9 / 60.4 / 46.0 / 30.7, Avg Len 2.46
- LAPA: 84.0 / 58.8 / 46.2 / 35.4, Avg Len 2.51
- VideoWorld 2: 88.5 / 64.6 / 55.8 / 47.5, Avg Len 2.88

Insight: VideoWorld 2 在 cross-domain setting 上大幅领先。Step 5 success rate 47.5% vs LAPA 35.4% vs video pretrain 30.7%。这证明 latent code pretraining 比 raw video pretraining 更 efficient —— latent codes 提取了 transferable dynamics, 而 raw video 包含太多 appearance noise。

### 6.3 UMAP 可视化 (Fig 7)

这个 figure 我觉得是 paper 最有说服力的 visualization:

实验设置: 从 CALVIN 和 Bridge (OpenX 的一部分) 各 sample 4000 trajectories, label 每个 trajectory 的 robot arm action (up/down/left/right), 然后对 latent codes 做 UMAP。

- **VideoWorld 2 (left)**: 同一 action 的 latent codes 在不同 environment (CALVIN vs Bridge) 紧密 align, 形成跨 environment 一致的 cluster
- **VideoWorld 1 (right)**: 同一 action 的 latent codes 在不同 environment 下显著 diverge, 无法有效 cluster

这个 visualization 直观展示了 transferability 的来源: **disentanglement 让 latent codes 跨 environment 一致**。VideoWorld 1 的 latent codes 被 environment-specific appearance "污染", 所以无法 transfer。

---

## 7. 关键 Ablations 深度分析

### 7.1 dLDM 架构 ablation (Table 3a)

让我逐行分析:
- Row 1 (no VDM, no stop-grad, no ControlNet): Paper 0.0, Block 28.5, LPIPS 0.312 — 这是 VideoWorld 1 baseline, 完全 fail on paper folding
- Row 2 (VDM, no stop-grad, no ControlNet): Paper 30.3, Block 45.2, LPIPS 0.297 — VDM 帮 visual quality, 但 latent codes 还是被 noise 污染
- Row 3 (VDM, stop-grad, no ControlNet): Paper 47.3, Block 54.7, LPIPS 0.275 — stop-grad 大幅提升 task performance (~20%)
- Row 4 (VDM, no stop-grad, ControlNet): Paper 51.1, Block 52.0, LPIPS 0.213 — ControlNet 提升 visual quality
- Row 5 (full model): Paper 68.8, Block 77.5, LPIPS 0.205 — 最佳

Key insight: stop-grad 和 ControlNet 都重要, 但 stop-grad 对 task performance 更关键, ControlNet 对 visual quality 更关键。两者协同达到最佳。

### 7.2 Compression length T (Table 3e)

- T=2 (LAPA-like): Paper 19.1, Block 38.7, Avg Len 1.55 — short context 不足以学 long-horizon
- T=9: Paper 55.4, Block 68.7
- T=49: Paper 65.3, Block 76.2
- T=93: Paper 68.8, Block 77.5 (最佳, 对应 Cosmos VDM max context)
- T=177: Paper 69.0, Block 76.8 (plateau)

Insight: long-horizon tasks 需要 long context。T=2 严重不足, 因为它只 capture pairwise transition, 失去 temporal structure。这解释了为什么 LAPA-style latent action models 在 long-horizon 上 fail —— 它们的 design 假设 short-horizon。

### 7.3 Training strategy for VDM (Table 3f)

- Random init VDM: 0.0 / 0.0 — model collapse, 完全无法 generate valid videos
- Freeze VDM: 31.7 / 40.2 — 只 train VQ-VAE components, 不足以 capture fine-grained manipulation
- LoRA: 50.9 / 62.3
- Full finetune: 68.8 / 77.5 (最佳)

Insight: VDM 需要进一步 adapt 到 target task 的 fine-grained manipulation。Frozen VDM 不足, random init 灾难。Full finetune 利用 prior 同时 adapt 到 task。

---

## 8. 与其他工作的比较

### 8.1 与 CoLA [62] (https://arxiv.org/abs/2510.26433) 的区别

CoLA 是 concurrent work, 也用 VDM optimize latent action codes, 但有 key differences:
- CoLA 只 model short 2-frame transitions
- CoLA 忽略 coarse VAE outputs 的 structured temporal cues
- VideoWorld 2 models multi-step dynamics + 用 VAE decoder output 作为 ControlNet condition

Table 1 row 11: CoLA on OpenX + Craft: Step 7: 40.2%, VideoWorld 2: 72.3%。差距 ~32%。

这个对比非常有说服力 —— CoLA 的 design 在 long-horizon 上 fail, 因为它没有 multi-frame modeling。

### 8.2 与 JEPA-style approaches 的区别

JEPA (V-JEPA 2 [3], https://arxiv.org/abs/2506.09985) avoid pixel-level reconstruction, 在 abstract space 预测。Paper Sec 2.2 提到:

> "JEPA-style approaches avoid pixel-level reconstruction, instead forecasting in an abstract space to benefit downstream tasks"

VideoWorld 2 的 dLDM 有点类似 JEPA 的 idea —— 都想 avoid pixel-level reconstruction。但 mechanism 不同: JEPA 用 latent predictive loss + masking, VideoWorld 2 用 VDM appearance prior + dynamics disentanglement。

JEPA 的 advantage: no generation needed, efficient。VideoWorld 2 的 advantage: 可以 generate 可视化 output 用于 evaluation + downstream planning。

### 8.3 与 Dreamer-style world models 的区别

Dreamer (https://arxiv.org/abs/1912.01603) 在 latent space 做 planning, 用 RSSM 等 structured latent。但 Dreamer 通常在 simple environments (Atari, DMC) 上 work, long-horizon real-world tasks 还没 scale 上去。

VideoWorld 2 的 setting 更 ambitious: minute-long real-world tasks with complex visual dynamics。

---

## 9. Limitations 和我的思考

### 9.1 Paper 提到的 limitations

Paper Sec 7 (Conclusion) 比较简短, 没有详细讨论 limitations。但 Appendix C 末尾提了:

> "the VDM processes only 93 frames at a time, while full task sequences can span thousands of frames. Long videos are therefore generated auto-regressively by extending each segment from the final frame of the previous one. Because the VDM's inherent reconstruction noise accumulates over time, visual artifacts such as lighting, texture, or color shifts may gradually appear."

这是一个真实问题: **autoregressive long-horizon generation 会 accumulate visual artifacts**。虽然 task-relevant dynamics 通过 latent codes 保持准确, 但 visual quality 会 degrade。这有点像 LLM 的 exposure bias 问题。

### 9.2 我看到的额外 limitations

**1. 计算成本**: 用 Cosmos DiT 2B + Cosmos AR 4B, 总参数 ~6B, training 需要大量 compute。Paper 没有报告 FLOPs 或 training cost, 但可以想象很高。

**2. VDM choice 的 dependency**: paper 用 Cosmos DiT 2B 作为 VDM。如果换一个 VDM (比如 Sora, Veo), 效果会怎样? Paper 没有做这个 ablation。我猜测 VDM 的 prior quality 会显著影响最终效果。

**3. Task-specific VDM finetune**: Table 3f 显示 full finetune 比 frozen 好。这意味着 VDM 需要 adapt 到 target task, 这降低了一些 zero-shot transferability 的 claim。如果要 transfer 到一个全新的 task, 需要重新 finetune VDM, 这可能 cost 不低。

**4. Latent code interpretability**: paper Fig 5 显示了 "similar latent codes 对应 similar dynamics", 但 latent codes 本身依然是 black-box。能否设计可解释的 latent space?

**5. Action space 的连续性**: paper 用 FSQ discrete codes。对于需要 continuous control 的 tasks (比如 robotic manipulation with precise force), discrete codes 可能 insufficient。

**6. Language conditioning**: paper 在 Video-CraftBench 上用 task instruction 作为 condition。但 instruction 是 task-level ("fold a paper airplane"), 没有 step-level instruction。Step-level language grounding 还没 explore。

### 9.3 与你的工作的关联

Karpathy, 我觉得这篇 paper 跟你的一些 idea 有 strong resonance:

**1. Software 2.0 的延伸**: 你在 "Software 2.0" (https://karpathy.medium.com/software-2-0-a64152b37c3f) 里说, 我们从 hand-crafted code 转向 learned programs。VideoWorld 2 是这个 idea 在 world model 上的体现 —— 从 hand-crafted physics engine 转向 learned video world model。

**2. "Lesson 1: Conception"**: 你在 https://karpathy.ai/zh/zh/lesson1/ 里强调 mental conception 的重要性。VideoWorld 2 的 dLDM 试图让模型 learn 一个 "task-relevant conception" 而不是 raw pixel conception。这其实是一种 implicit abstraction。

**3. LLM101 / nanoGPT 的精神**: 你喜欢 minimal, clean implementation。dLDM 的 design 其实相当 minimal —— VQ-VAE encoder + queries + VDM prior。没有复杂的 reward function, 没有 RL, 纯 generative modeling。这符合你 "let data do the work" 的哲学。

**4. 从 Yann LeCun 的 JEPA 视角看**: Yann 一直 push JEPA-style world model, 反对 generative world model 因为 "pixel-level reconstruction is wasteful"。VideoWorld 2 用 VDM 处理 appearance, latent codes 处理 dynamics, 这其实是一个 hybrid: dynamics 部分类似 JEPA (avoid pixel prediction), appearance 部分是 generative。可能是两个 paradigm 的 sweet spot?

### 9.4 Future directions 我想到的

**1. Scaling VDM**: 如果用更大更强的 VDM (Sora 2 [46], Veo [20]), appearance quality 会更好, latent codes 也更 pure。这是直接的 scaling direction。

**2. Hierarchical latent codes**: 当前 latent codes 是 flat 的 ($N=4$ queries × $K$ timesteps)。如果引入 hierarchical structure (high-level action + low-level dynamics), 可能 capture 更复杂的 long-horizon planning。

**3. Active inference**: 当前 model 是被动的 generate video。如果让 agent 在 environment 中 act, 用 latent codes 作为 action representation, 可能 connect 到 active inference framework。

**4. Multi-modal transfer**: paper 只用 visual data。如果加入 audio (paper folding 有 sound cues)、tactile (用 touch sensors), 可能 enhance dynamics learning。

**5. Cross-embodiment transfer**: OpenX 包含多种 robot embodiments。VideoWorld 2 显示 latent codes 跨 embodiment transferable。这跟 RT-2 (https://arxiv.org/abs/2307.15818), Octo (https://arxiv.org/abs/2405.12213) 等 VLA model 的方向一致, 但更 fundamental —— 不用 language 作为 bridge, 直接用 dynamics latent。

**6. 与 VLA model 的结合**: 当前 VLA model (π0, OpenVLA) 用 language-conditioned action prediction。如果把 dLDM 的 latent codes 作为 action space, 可能更 sample-efficient。这其实是 Moto [15] 的 idea 的进一步发展。

**7. World model for planning**: VideoWorld 2 生成 video execution。如果用这个 video world model 做 model-based planning (类似 Dreamer), 在 latent space rollout 多个 trajectories, 选最优的, 可能 enable 更复杂的 planning。

**8. Self-supervised curriculum**: 当前用 human-recorded tutorials。如果 agent 能自主 generate "practice videos" (类似 AlphaGo 的 self-play), 可能 scale 更好。

---

## 10. 与 video generation scaling laws 的关系

最后我想思考一个更宏大的问题: **video world model 是否会遵循 LLM 的 scaling laws?**

LLM 的 scaling laws (https://arxiv.org/abs/2001.08361) 显示 loss 随 parameters / data / compute power-law decrease。Video generation 也有 scaling laws (https://arxiv.org/abs/2405.06809), 但 task performance (success rate) 的 scaling behavior 还不清楚。

VideoWorld 2 的 Table 1 显示: 即便 14B 的 Wan2.2 在 long-horizon task 上 fail, 而 4B+2B 的 VideoWorld 2 succeed。这说明 **task performance 不只取决于 model size, 更取决于 architecture design 的 inductive bias**。

这可能 hint: **naive scaling video generation models 不会自动 give us world knowledge**。需要像 dLDM 这样的 disentanglement mechanism。这是对你的 "the bitter lesson" 论点的一个 nuance —— scaling 是 necessary, 但合适的 architecture prior 可以让 scaling 更 efficient。

---

## 11. 一些值得深挖的技术细节

### 11.1 FSQ 量化

Paper 用 FSQ (Finite Scalar Quantization) 而不是标准 VQ。FSQ (https://arxiv.org/abs/2309.15564) 的 idea 是: 把 continuous vector 投影到一组 finite levels, 然后用 Cartesian product 形成 codebook。

Levels [8, 5, 5, 5] 意味着: 一个 4-dim vector, 每个维度分别 quantize 到 8/5/5/5 个 levels, 总 codebook size = 8×5×5×5 = 1000。

FSQ 的 advantage over VQ:
- No codebook collapse 问题 (VQ 的常见 issue)
- No auxiliary losses (commitment loss, EMA update)
- Differentiable (用 straight-through estimator)

### 11.2 MAGVITv2-style causal codec

MAGVitv2 (https://arxiv.org/abs/2310.05737) 的 design 用 causal 3D CNN, 确保 temporal causality。这对 long-horizon generation 重要 —— 不能让 future frames influence past frames 的 representation。

### 11.3 Warmup strategy

Paper Sec 5.1 提到一个 training trick:

> "dLDM first applies a short warm-up where the latent codes are optimized solely using the original reconstruction objective. This warm-up is similar to the training strategy of the original VideoWorld LDM, enabling the latent codes to rapidly learn to compress visual changes and motion dynamics, while allowing the decoder to reconstruct low-fidelity video clips containing agent motion trajectories based on the initial frame and codes."

这个 warmup 很关键: 先让 latent codes 学会 compress dynamics, decoder 学会 reconstruct low-fidelity motion, 然后再 switch 到 disentangled scheme (用 VDM)。如果不 warmup, VDM 从 random noise 开始 denoise, 训练不稳定。

这是一种 curriculum learning: 先学 coarse dynamics, 再 refine appearance。

### 11.4 Causal 3D CNN in VDM

Paper 在 VDM 里 enforce causal attention。Cosmos DiT 原本可能是 bidirectional attention (standard video diffusion), paper 改成 causal。这类似 Decoder vs Encoder 架构的区别。Causal VDM 适合 AR generation, 因为生成第 $t$ 帧时不能看未来。

---

## 12. 代码 / 实现细节

Paper 承诺 open-source code, data, models (Sec 1 最后):

> "with all code, data, and models to be open-sourced for further research"

项目主页 https://VideoWorld2.github.io/ 应该会发布。

Training config (Table 4):
- dLDM: AdamW, lr 1e-4, weight decay 0.1, β=(0.9, 0.99), batch 128, 100k iterations
- AR Transformer: AdamW, lr 3e-4, weight decay 0.05, β=(0.9, 0.98), batch 256, 50k iterations

Vocabulary size 1000, embedding length N=4, 默认 context T=93 frames (~5s @ 16fps, 480px)。

---

## 13. 总结: 这篇 paper 的 deep insight

让我 try to synthesize 这篇 paper 的 core insight:

**核心 thesis**: 从 raw video 学习 transferable world knowledge 的 key bottleneck 是 appearance-dynamics entanglement。通过把 appearance modeling offload 给 pretrained VDM, 让 latent codes 专注于 task-relevant dynamics, 可以 achieve transferable, long-horizon, generalizable world knowledge。

**Technical mechanism**:
1. VQ-VAE encoder + learnable queries 提取 visual changes → latent codes
2. VQ-VAE decoder 提供 low-fidelity motion scaffolding (ControlNet-style)
3. Pretrained VDM 负责 appearance generation, conditioned on latent codes + decoder output
4. Stop gradient from decoder to latent codes (prevent appearance noise)
5. AR transformer 在 latent space 学 long-horizon policy

**Why it works**:
- Latent codes 只需要 encode "what action is happening", 不需要 encode "what does the environment look like"
- VDM 作为 appearance expert, fill in realistic details
- 跨 environment, latent codes 保持 consistent (UMAP 可视化验证)
- Long-horizon 通过 AR transformer 在 compact latent space 建模, 避免 long video generation 的 error accumulation

**Key takeaways for the field**:
1. Disentanglement via prior, not via regularization —— 用一个 pretrained expert 处理一个 modality, 比 用 auxiliary loss 强迫 disentangle 更 effective
2. Latent action models 需要 long-horizon context —— short-horizon latent action (LAPA-style) 不足以学 complex task
3. Visual fidelity ≠ task understanding —— SOTA video generation models 在 long-horizon task 上 fail, 说明 task-relevant representation 是 different axis

---

## 14. 一些可能的 connections 和延伸阅读

- **V-JEPA 2** (https://arxiv.org/abs/2506.09985): LeCun 的最新 world model, 用 self-supervised video prediction 在 abstract space。与 VideoWorld 2 形成 interesting contrast。

- **Genie 2** (https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/): DeepMind 的 game world model, 也探索 controllable generation。

- **Sora 2** (https://openai.com/index/sora-2/): OpenAI 的最新 video generation model, 强调 world simulation 能力。

- **Cosmos** (https://www.nvidia.com/en-us/ai/cosmos/): NVIDIA 的 world foundation model, VideoWorld 2 用其作为 VDM 和 AR transformer。

- **LAPA** (https://arxiv.org/abs/2410.11758): Latent Action Pretraining from Videos, VideoWorld 2 的主要 baseline 之一。

- **AdaWorld** (https://arxiv.org/abs/2503.18938): Learning adaptable world models with latent actions。

- **Moto** (https://arxiv.org/abs/2411.14459): Latent motion token as bridging language for robot manipulation。

- **CALVIN** (https://arxiv.org/abs/2112.03230): Long-horizon robot manipulation benchmark。

- **Open-X Embodiment** (https://robotics-transformer-x.github.io/): 大规模 robot learning dataset。

- **DINOv2** (https://arxiv.org/abs/2304.07193): Self-supervised vision features, paper 用其训练 task classifier。

- **MAGVITv2** (https://magvit.github.io/): Video tokenizer, VideoWorld 1/2 的 codec 基础。

- **FSQ** (https://arxiv.org/abs/2309.15564): Finite Scalar Quantization, 替代 VQ 的 quantization 方法。

- **ControlNet** (https://arxiv.org/abs/2302.05543): Adding conditional control to diffusion models, VideoWorld 2 用其注入 motion condition。

- **V-JEPA** (https://arxiv.org/abs/2301.08243): LeCun 的 video joint-embedding predictive architecture。

- **VideoWorld 1** (https://arxiv.org/abs/2505.18034): 这篇 paper 的前作。

- **CoLA** (https://arxiv.org/abs/2510.26433): Concurrent work on co-evolving latent action world models。

- **Dreamer V3** (https://arxiv.org/abs/2301.04104): Mastering diverse domains through world models。

---

## 15. 给你的几个 open questions

Karpathy, 这篇 paper 让我想问你几个问题, 我觉得对你的 thinking 可能有 challenge:

1. **Disentanglement via prior vs disentanglement via objective**: VideoWorld 2 用 pretrained VDM 作为 prior 来 disentangle。这是一个 "two-model" approach。能否用一个 unified model, 通过某种 clever objective 实现 disentanglement? 还是 prior 注入是必要的?

2. **Latent codes 作为 universal action representation**: 如果 latent codes 真的 capture task-relevant dynamics, 它们能否成为 universal action space, 跨 embodiment 跨 task? 这跟你 "let the data define the program" 的哲学契合。

3. **Video world model vs LLM**: LLM 通过 next-token prediction 学到 world knowledge。Video world model 通过 next-frame prediction 学到 world dynamics。两者最终会 converge 吗? 还是永远 different modality?

4. **Embodied AGI 路径**: 你怎么看这种 pure-visual, no-language, no-RL 的路径 toward embodied intelligence? 是 dead end 还是 promising direction?

5. **Open-world scaling**: paper 在 paper folding 和 block building 上验证。如果 scale 到真实世界的所有 handcraft tasks (cooking, sewing, woodworking, ...), 这种 approach 能 scale 吗? 或者需要什么额外的 mechanism?

---

希望这个 detailed breakdown 对你有帮助, Andrej。这篇 paper 我觉得是 world model 领域一个 important milestone, 它给出了一种 concrete mechanism 来解决 visual world knowledge 的 transferability 问题, 同时也 open 了很多 interesting research questions。我 particularly excited about 它跟 VLA model 的结合 potential —— latent dynamics codes 可能成为 language-conditioned action 之外的另一种 action representation, 也许更 sample-efficient, 更 transferable。

如果你想要我 deep dive 某个 specific aspect (比如 implementation details, math derivations, comparison with specific baseline), 让我知道, 我可以继续 expand。
