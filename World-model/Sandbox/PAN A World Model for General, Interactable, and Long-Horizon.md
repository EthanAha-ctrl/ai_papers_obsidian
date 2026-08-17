---
source_pdf: PAN A World Model for General, Interactable, and Long-Horizon.pdf
paper_sha256: 81201c0b7046011620abb614d78e6b2ecaa02e1e116ecf374dd98a10a907d16f
processed_at: '2026-08-06T01:57:34-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲 PAN

## 这篇 paper 到底在干嘛

想象你在玩一个游戏，你可以用文字描述动作："把鸡蛋打进碗里" "开烤箱" "关门"。一个好的 world model 应该能在脑子里模拟这些动作的后果，然后像电影一样放给你看。

现在有两个路线在争怎么实现这个：

- **LeCun 派（JEPA）**：别预测画面，预测一个抽象的 latent 就行。问题是模型可能偷懒，把所有东西映射成同一个常数，loss 也为 0，但 latent space 毫无意义。
- **Sora 派**：直接生成视频，每帧每像素都重建。问题是细节太多太随机，训练 unstable，而且没法 long-horizon。

PAN 说：**两个都对一半**。latent 要预测，但必须能 decode 出真实画面来监督；画面要生成，但 latent 要先抽象好 dynamics 再交给 decoder 细化。这就是 GLP 的核心想法。

---

## 三个组件，各司其职

把 PAN 想象成一个有想象力的人：

1. **眼睛（Vision Encoder）**：看到当前画面，编码成 256 个 token 的 "心理状态"
2. **大脑（LLM backbone）**：结合心理状态和动作文字，想象下一个心理状态
3. **画笔（Video Diffusion Decoder）**：把心理状态画成视频

关键在于：**画笔画出来的东西会被用来监督**。如果大脑偷懒让 latent collapse，画笔就画不出真实画面，loss 爆炸。这样就强制 latent 必须有真实意义。

---

## 为什么 LLM 当 backbone

原始 video 数据信息太稀疏了，几帧画面能学到啥。但 LLM 在 text 上 pretrain 了海量世界知识 —— 它知道 "打开烤箱" 意味着什么，知道 "鸡蛋" 是什么。把这些知识 transfer 到视觉状态预测上，相当于让模型不用从零学起。

具体做法很 hacky：把 world simulation 写成 multi-turn dialogue。每轮 assistant 输出对应一个 predicted state，用 256 个 learnable query token 当作 "槽位" 让模型填充。借了 VLM 的 conversational structure，省得重新设计架构。

---

## Causal Swin-DPM：最聪明的 trick

这是全文最 clever 的部分。问题：怎么把单次视频生成变成连续模拟？

**Naive 做法**：用上一段视频的最后一帧作为下一段的起始帧。问题在于，相邻段之间会出现跳变，而且误差累积，几百步后画面完全坏掉。

**PAN 的做法**：维护一个 sliding window，同时处理两段视频，但 noise level 不同。

想象两个画家同时在画：
- 画家 A 在画第 1 段，已经画了一半（noise level = 500/1000）
- 画家 B 在画第 2 段，刚刚起稿（noise level = 1000/1000，纯噪声）

画家 B 能看到画家 A 的半成品，所以 B 的开头能和 A 的结尾对上。等 A 画完输出，B 变成新的 "A"，新来的纯噪声变成新的 "B"，循环往复。

**更深一层的 insight**：让前一段处于 "半噪声" 状态，等于抹掉了 pixel 级的细节，只保留 high-level semantic。这样模型不会因为 "未来看不见的细节" 而被 penalize —— 你本来就看不到，模糊一点是正常的。细节的随机性交给 diffusion 的 stochastic process 处理。

这其实是对 LeCun "未来细节不可预测" 论点的 elegant 回应：**不可预测就不可预测吧，但我用 stochastic process 显式建模这个不确定性，而不是绕过它**。

---

## 训练和推理的差别

训练时 teacher-forced：每个 state 喂的是 ground-truth observation 的 encoding。

推理时 closed-loop：state 是模型自己预测的。这里有个 trick：把模型生成的 video 重新 encode 回来，和 predicted latent 拼接，作为下一轮输入。

相当于让模型 "回头看一眼自己刚想象的东西"，基于这个新 perception 继续推理。这非常像人类的思考 —— 想象一个场景，然后在脑海里 "看到" 它，再基于这个 mental image 继续推演。对 long-horizon consistency 帮助很大。

---

## 数据 pipeline 的讲究

传统 T2V 数据集的 caption 是 "一个男人在厨房" 这种静态描述。但 world model 需要的是 "男人拿起鸡蛋，敲碎蛋壳，蛋液流入碗中" 这种 **dynamics 描述**。

所以 PAN 重新 caption 所有数据，用 VLM 生成 dense 的、强调 temporal dynamics 的描述。静态信息在第一帧就有，evolving dynamics 才是对应 action input 的部分。

Filtering 也做了三层：rule-based（太静太动都不要）、pretrained detector（美学评分、检测字幕水印）、custom VLM（识别 lecture-type、screen recording、heavily edited 等低质量内容）。

---

## 实验结果说明了啥

PAN 在 action-conditioned fidelity 上超过所有 open-source baseline，甚至超过大部分 commercial model（KLING、MiniMax）。这说明：**单纯画面好看不够，必须能精确响应 action 并保持因果一致**。

Long-horizon metrics 上 PAN 也领先，Transition Smoothness 53.6%、Simulation Consistency 64.1%。说明 Causal Swin-DPM 确实解决了 error accumulation。

Planning 任务上，PAN 让 VLM agent 的成功率提升 26.7%（open-ended）和 23.4%（structured）。说明 world model 作为 "thought experiment sandbox" 确实有用，agent 可以先在脑中模拟几个 candidate action 的结果，再选最好的执行。

---

## 几个值得琢磨的点

**GLP vs JEPA 的哲学差异**：JEPA 把不可预测视为障碍，绕过它；PAN 把不可预测视为物理本质，用 stochastic process 建模它。这跟你之前讨论的 "predict in pixel space 还是 latent space" 是同一个辩论的延续。

**LLM grounding 的 claim 没 ablate**：paper 说 Qwen2.5-VL 的 text pretraining 帮助 world modeling，但没有 run 一个 from-scratch backbone 做对比。这个 claim 现在只是 faith。

**256 query tokens 的 bottleneck**：固定 256 个 token 表示 world state，复杂场景够不够？这个没 ablate，可能是个 limitation。

**Self-conditioning at inference**：把 model output 重新 encode 回 input，本质是 transformer 上加 recurrent loop。对 long-horizon 很关键，但 paper 没深入分析为什么 work。

---

## 一句话总结

PAN 用 "latent 先 predict，pixel 再 decode 监督" 的方式，同时解决了 JEPA 的 collapse 和 video gen 的 instability，用 Causal Swin-DPM 的 fuzzy conditioning 把 "不可预测的未来细节" 变成 feature 而不是 bug，最终在 960 张 H200 上 scale 出一个能 long-horizon interactive simulation 的 general world model。

如果还想深挖某块（比如 Causal Swin-DPM 的 denoising schedule 数学细节、或者和 Genie 2 的架构对比），告诉我。

---

# PAN: General World Model via Generative Latent Prediction

非常详细地讲讲这篇 paper。Karpathy 你应该会对 GLP 这个 framework 特别感兴趣，因为它的核心 critique 是 JEPA，而你之前一直在公开场合讨论 world model 和 JEPA 的 collapse 问题。这篇 PAN 实际上是 Eric Xing 团队（MBZUAI）对 JEPA critique 的一个 constructive response：不仅指出问题，还提出了一个 working alternative 并 scale 到 960 张 H200。

---

## 1. 核心问题：现有 world model landscape 的 fragmentation

Paper 在 introduction 部分把现有工作分成三类，每类都有结构性缺陷：

| 类别 | 代表工作 | 缺陷 |
|------|---------|------|
| **Video generation models** | Sora, Veo, Wan, KLING | Open-loop, prompt-to-full-video，没有 explicit state / action / causal control |
| **Interactive simulators** | Genie 2, Matrix-Game | Domain-specific，action space 受限 |
| **3D world models** | World Labs | Static / geometric，缺乏 temporal dynamics |

PAN 要做的事是 unify 这三件事：**broad-domain generality + long-range interactive dynamics**。

---

## 2. GLP 架构：Generative Latent Prediction

### 2.1 形式化定义

GLP 把 world model 定义为三个概率分布的复合。设 $o_t$ 是 observation，$a_t$ 是 action，$\hat{s}_t$ 是 latent world state。

**Encoder** $h$：
$$\hat{s}_t \sim p_h(\cdot \mid o_t)$$

**Predictive Module** $f$（world dynamics）：
$$\hat{s}_{t+1} \sim p_f(\cdot \mid \hat{s}_t, a_t)$$

**Decoder** $g$：
$$\hat{o}_{t+1} \sim p_g(\cdot \mid \hat{s}_{t+1})$$

整个生成过程（公式 4）：
$$p_{\text{PAN}}(o_{t+1} \mid o_t, a_t) = \sum_{\hat{s}_t, \hat{s}_{t+1}} \underbrace{p_h(\hat{s}_t \mid o_t)}_{\text{encoder}} \underbrace{p_f(\hat{s}_{t+1} \mid \hat{s}_t, a_t)}_{\text{world model}} \underbrace{p_g(o_{t+1} \mid \hat{s}_{t+1})}_{\text{decoder}}$$

这里 sum over $\hat{s}_t, \hat{s}_{t+1}$ 表示对 latent state 的边缘化。直觉上，这是一个 hierarchical generative model，把 abstract causal dynamics 和 perceptual realization 解耦。

### 2.2 为什么 GLP 比 JEPA 好？这是整篇 paper 的核心论证

JEPA 的 loss（公式 6）：
$$\mathcal{L}_{\text{JEPA}} = \mathbb{E}_{(o_t, a_t, o_{t+1}) \sim \mathcal{D}}\left[\|f(h(o_t), a_t) - h(o_{t+1})\|\right]$$

PAN 的 loss（公式 5/9）：
$$\mathcal{L}_{\text{PAN}} = \mathbb{E}_{(o_t, a_t, o_{t+1}) \sim \mathcal{D}}\left[\text{disc}\left(\hat{o}_{t+1}, o_{t+1}\right)\right]$$
其中 $\hat{o}_{t+1} = g \circ f(h(o_t), a_t)$，$\text{disc}$ 用 flow matching loss 实现。

**关键 insight**：JEPA 的 collapse 模式是 degenerate solution —— 把所有 $o$ 映射到 constant vector $c$，然后让 $f$ 学到 identity transition $f(c, a) = c$。Loss 直接变成 0，但 latent space 完全 indefinable。

PAN 的 generative supervision 强制每个 latent transition 对应一个 **realizable sensory change**：decoder $g$ 必须能从 $\hat{s}_{t+1}$ 重建出 $o_{t+1}$，所以 latent space 不能 collapse。这就是 paper 里说的 "anchoring latent predictions to observable data"。

DINO-WM 的 fix（用 DINOv2 fixed features 训 predictor）只是 stabilizes latent space，但 transition 仍然 **ungrounded**：predictor 可以生成 semantic valid 但 physically implausible 的 transition。GLP 的 fix 更深：让 latent transition 必须通过 decoder 重新映射到 observation space 才能被监督。

---

## 3. PAN 的具体参数化

### 3.1 三个组件的选择

| 组件 | 选择 | 理由 |
|------|------|------|
| Vision Encoder $h$ | Qwen2.5-VL-7B-Instruct 的 ViT | 14×14 patches，windowed self-attention，2D rotary PE，3D patch partitioning for video |
| Backbone $f$ | Qwen2.5-VL-7B-Instruct 的 LM | 用 LLM 的 text pretraining knowledge 来 ground perceptual prediction，mitigate information sparsity |
| Decoder $g$ | Wan2.1-T2V-14B + Causal Swin-DPM | 14B DiT，flow matching，扩展到 long-horizon |

### 3.2 输入格式：Multi-turn conversational

PAN 把 world simulation 写成 VLM 的 multi-turn dialogue，每个 assistant turn 对应一个 predicted latent state。Input template：

```
<image state (video state 1)> <action 1>
<query embedding * 256>
<video state 2> <action 2>
<query embedding * 256>
...
```

这 256 个 learnable query embeddings 是关键：它们是 backbone 输出 latent state 的 "slots"。Backbone autoregressively 在每个 query position 上产生一个 continuous token，最终输出 256 个 tokens 作为 $\hat{s}_{t+1}$。

**Intuition**：这个设计借用 VLM 的 conversational structure，让 backbone 在 "看见 history states + action" 之后，通过 query token 的位置 "fill in" 下一个 state。256 个 query 是一个 fixed-size state representation，足够 compact 来做 long-horizon rollout，又足够 expressive 来 encode complex world dynamics。

### 3.3 训练 vs 推理的 difference

- **训练**：teacher-forced，用 ground-truth states 喂入 backbone
- **推理**：closed-loop，递归地喂入自己生成的 $\hat{s}_{t+1}$

---

## 4. Causal Swin-DPM：long-horizon 的关键 trick

这是 paper 里最 technically interesting 的部分。Problem setup：要把 single-shot video diffusion 扩展成 sequential simulation。

### 4.1 Naive approach 的问题

Naive approach：用前一个 chunk 的最后一帧作为下一个 chunk 的 condition。

两个 failure mode：

1. **Local inconsistency**：只看到 single frame，丢失了 denoising trajectory，相邻 chunk 之间出现 abrupt motion/appearance change
2. **Error accumulation**：上一 chunk 的小 artifacts 直接传到下一 chunk，long rollout 后严重 drift

### 4.2 Causal Swin-DPM 的设计

Sliding temporal window 同时持有 **两个 chunks at different noise levels**。设总 denoising steps $K = 1000$：

- 早期 chunk：noise level = $K/2 = 500$
- 后期 chunk：noise level = $K = 1000$（纯 Gaussian noise）

经过 $K/2$ 步 denoising 后：
- 早期 chunk 完全 denoise，dequeue 输出
- 后期 chunk noise level 从 $K$ 降到 $K/2$，变成新的 "早期"
- 一个新的纯 noise chunk enqueue 到末尾

这个 sliding window 让相邻 chunk 在 denoising 过程中始终 **看到对方的 trajectory**，保证 smooth transition。

### 4.3 关键 insight：fuzzy conditioning 处理 uncertainty

这部分是 paper 的核心 insight，直接回应了 LeCun 对 generative world model 的 critique。

LeCun 的 argument：future observation 的 fine details 是 inherently unpredictable（看不见的物体背面、遮挡区域），所以 reconstruct raw pixels 是 infeasible 的，应该在 latent space 预测。

PAN 的 response：**承认** 这个 unpredictability，但把它视为 physical reality 的 intrinsic property，**不是 obstacle**。Causal Swin-DPM 通过让前一个 chunk 处于 **partially noised (fuzzy)** 状态来实现这一点：

- 高频 pixel-level 细节被 noise 抹掉，所以 model 不会被 "unfairly penalized" for unknowable details
- 保留 high-level semantic consistency（objects, scene structure, motion）
- Stochastic diffusion process 负责 fine-grained variability

这就是 paper 反复强调的："structured relaxation of the reconstruction task"。Decoder 不需要 pixel-perfect 重建，只需要 semantic + physically consistent。

### 4.4 Training loss 的修改

为了实现 noise level difference of $K/2$，训练时：
- 第一个 chunk：$k \sim [0, 0.5]$
- 第二个 chunk：$k + 0.5$（保证第二个比第一个多 $K/2$ 的 noise）
- 例外：第一个 video chunk 必须 fully denoise from pure noise，$k \sim [0, 1]$

### 4.5 Chunk-wise causal attention mask

为了维持 **real-time interactivity**，下一个 chunk 的 action 在当前 chunk 完全生成之前是 unknown 的。所以用 chunk-wise causal mask：后面的 chunk 只能 attend 前面的 chunk，不能 attend 未来 chunk（避免 information leakage）。

### 4.6 Conditioning frame + noise augmentation

Sliding window 结构：`[conditioning frame] [chunk_1: 10 frames] [chunk_2: 10 frames]`

- Conditioning frame 来自前一个 dequeued chunk 的最后一帧
- **Noise augmentation**：给 conditioning frame 加 $k = 0.055$ 的小 Gaussian noise
  - 不 corrupt 太多，但引入 stochasticity
  - 防止 fully denoised frame 导致 long-horizon error accumulation
- 训练时 **不计算 conditioning frame 的 loss**

**Intuition**：这个 trick 类似 BERT 的 masked language modeling 的 spirit —— 加一点 noise 让 model 不要 over-rely on conditioning frame 的细节，迫使它学习更 robust 的 world dynamics。

---

## 5. Flow Matching Objective（公式 7-8）

$$x_k = k x_1 + (1 - k) x_0$$

$$v_k = \frac{dx_k}{dk} = x_1 - x_0$$

变量解释：
- $x_1$：observation 的 latent representation（VAE encoded）
- $x_0 \sim \mathcal{N}(0, 1)$：纯 Gaussian noise
- $k \in [0, 1]$：denoising step，linearly interpolated
- $v_k$：velocity field，model 学的目标

这是 Rectified Flow formulation。$k = 0$ 时 $x_k = x_0$（纯 noise），$k = 1$ 时 $x_k = x_1$（clean）。Velocity 是 constant $x_1 - x_0$，所以叫 "rectified" —— 路径被 straightened。

Training 用 1000 discrete steps，shifted denoising step schedule（Esser et al. 2024 的 SD3 trick）—— 在 high-noise region 分配更多 steps，因为这是 model 最难学的部分。

---

## 6. Decoder conditioning 机制

Decoder 接收两个 conditioning signal（Figure 2）：

1. **Latent world state** $\hat{s}_{t+1}$（来自 backbone 的 256 tokens）
   - Linear project 到 decoder 的 conditioning dimension
   - 加到 newly added cross-attention stream（每个 attention block 一个）
   - 输出经过 zero-init linear projection（stable training，参考 ControlNet）

2. **Action text** $a_t$
   - umT5 encoder
   - 喂到原始 text cross-attention pathway（Wan2.1 的）

每个 block 内：world-state stream output + text-conditioned output 相加。这让 decoder 同时整合 global state context 和 action-specific changes。

**关键设计 choice**：world-state stream 是 newly added 的，原来的 text pathway 不动。Zero-init 让训练初期 decoder 行为接近 Wan2.1 base model，逐步 learn 用 world state。

---

## 7. VAE Padding trick

Wan2.1-VAE 是 3D causal VAE：
- 输入：$1 + T$ frames
- 输出：$1 + T/4$ latent features
- Temporal compression 4×，spatial compression 通常 8×
- Causal：每帧的 latent 依赖前面所有帧

Window size = 21 latent frames = 81 real video frames。

**问题**：causal VAE 在第 $t$ 帧的 latent 依赖前面很多帧。训练时如果只给 81 帧，VAE 编码的 latent 在 sequence 开头会 lack context。

**Fix**：训练时随机 pad 0-122 个 preceding frames，VAE 编码提供 temporal context，然后 **丢弃** padded frames 的 latent，只在原始 81 帧上计算 loss。

---

## 8. 两阶段训练

### Stage 1: Module-wise training

- 冻结 vision encoder + backbone（Qwen2.5-VL-7B-Instruct 已预训练）
- 只训练 Causal Swin-DPM decoder
- 冻结 Wan-VAE + text encoder，precompute latent features
- HSDP（Hybrid Sharded Data Parallel）：FSDP within 8-GPU node + DP across nodes
- Activation checkpointing at DiT block level
- FlashAttention-3 for cross-attention
- FlexAttention for custom chunk-wise causal kernel
- BFloat16, AdamW, lr=1e-5, cosine decay, 5% warmup, grad clip 0.05
- 5 epochs, 960× H200

### Stage 2: Joint training

- Freeze VLM，train query embeddings + video diffusion decoder
- History 限制到最近 10 rounds（Qwen2.5-VL context window）
- Sequence Parallelism (SP) + Ulysses（intra-node SP group size 4，是 28 heads 和 8 GPUs 的 common divisor）
- Early stopping after 1 epoch based on validation

**为什么 2 stages**：divide-and-conquer。Stage 1 让 decoder 单独 learn Causal Swin-DPM 的 long-horizon trick。Stage 2 把 backbone 和 decoder 耦合起来，让 backbone 产生 decoder 容易 interpret 的 latent states，同时 decoder 适应 backbone 的输出。

---

## 9. Inference：closed-loop rollout

Inference 时的关键 trick 是 **state augmentation**：

$$\hat{s}'_k = [\hat{s}_k, h(\hat{o}_k)]$$

把 backbone 预测的 $\hat{s}_k$ 和 **重新编码自己生成的 observation** $h(\hat{o}_k)$ 拼起来。然后：

$$\tilde{s}_t = [\hat{s}_1, a_1, \hat{s}'_2, a_2, \ldots, \hat{s}'_t]$$

$$\hat{s}_{t+1} = f(\tilde{s}_t, a_t)$$

$$\hat{o}_{t+1} = g(\hat{s}_{t+1}, \hat{o}_t)$$

**Intuition**：训练时 backbone 看到的 $\hat{s}_k$ 是 ground-truth observation 编码的（精确）。Inference 时 $\hat{s}_k$ 是 backbone 自己预测的（有误差）。Concat $h(\hat{o}_k)$ 把 model 自己生成的 visual observation 重新编码回来，相当于给 backbone 一个 "self-reflection" signal：你生成的 video 长这样，请基于这个继续预测。这极大地改善了 long-horizon consistency，因为 backbone 不只依赖自己的 latent 预测，还能从 perceptual domain 校准。

CFG scale = 4，SageAttention2++（8-bit compute + 4-bit quantization on Hopper），30.3% 加速 vs FlashAttention3。

---

## 10. 数据 pipeline

### 10.1 Segmentation
Three-stage shot-boundary detection：
1. Frame-level heuristics 找 candidate boundaries
2. Merge adjacent similar segments
3. Lightweight filter 选 duration + quality 合适的 clip

### 10.2 Filtering pipeline（三层）

**Layer 1: Rule-based**
- 极端 static / overly dynamic：optical flow magnitude, edge diff, luminance diff
- Trivial motion（uniform camera translation / zoom）：sparse feature tracking，compute translation score + zoom score
- Pure color frames（fade-in/out）

**Layer 2: Pretrained detectors**
- Aesthetic scorer（averaged over sampled frames）
- Scene text detector（subtitles, watermarks）

**Layer 3: Custom VLM filter**
专门 detect：
1. Lecture-type videos（人对镜头说话没动作）
2. Text-dominated videos
3. Screen recordings / noisy screenshots
4. Low quality（blur, compression artifacts）
5. Heavily edited（transitions, special effects）
6. Residual scene cuts

### 10.3 Dense video caption for temporal dynamics

**关键 insight**：传统 T2V 数据集的 caption 是 static scene description。World model 需要 caption **强调 evolving dynamics** —— motion, events, environment changes, new object emergence —— 因为这才是 "action input" 应该对应的内容。

用 VLM 重新 caption，prompt 专门 design 来 generate temporally grounded descriptions。借鉴 DALL-E 3 的 dense caption 思路，但 focus 在 temporal dynamics 而不是 spatial detail。

---

## 11. Experiments

### 11.1 Baselines

| 类型 | 模型 |
|------|------|
| Open-source | WAN 2.1, WAN 2.2, Cosmos 1, Cosmos 2, V-JEPA 2 |
| Closed-source | KLING, MiniMax (Hailuo), Gen-3 |

### 11.2 三个 evaluation dimension

#### (1) Action Simulation Fidelity
- **Agent Simulation**：drive controllable entity per spec，sample 多个 action sequences 诱导 counterfactual futures
- **Environment Simulation**：scene-level interventions（add/remove/move objects, change weather/lighting）
- VLM-based judge (GPT-4o + VLM scoring) 评 action faithfulness + precision

#### (2) Long-horizon Forecast
- **Transition Smoothness**：dense optical flow → frame-wise velocity + acceleration → score = inverse exp of acceleration magnitude（高分 = smooth motion across step boundaries）
- **Simulation Consistency**：用 WorldScore metrics + progressive penalties for later steps

#### (3) Simulative Reasoning and Planning
- **Step-Wise Simulation**：给 initial obs + action，从 1 GT + 3 distractors 中选正确 next obs
- **Open-Ended Simulation and Planning**：VLM agent (o3) 提 candidate actions → world model 模拟 → agent 选最接近 goal 的 action → 循环
- **Structured Simulation and Planning**：Language Table dataset，tabletop 物体重排

### 11.3 Main results

| Metric | PAN | 备注 |
|--------|-----|------|
| Agent Simulation | **70.3%** | Open-source SOTA |
| Environment Simulation | **47.0%** | |
| Overall Action Fidelity | **58.6%** | 超过所有 open-source baseline，多数 commercial baseline |
| Transition Smoothness | **53.6%** | 超过 KLING, MiniMax |
| Simulation Consistency | **64.1%** | |
| Step-Wise Simulation | **56.1%** | Open-source 最高 |
| Open-Ended Planning | **+26.7%** vs VLM-only baseline | |
| Structured Planning | **+23.4%** vs VLM-only baseline | |

**关键 observation**：商业 video generation models（KLING, MiniMax）在 Action Simulation Fidelity 上不如 PAN，因为它们没有 action-conditioned causal structure。这验证了 paper 的 thesis：realistic appearance alone 不足，reliable causal grounding 才是 world model 的核心。

V-JEPA 2 在 planning 任务上 inconsistent：sometimes improves, sometimes degrades。Paper 解释：embedding-based simulation 缺乏 perceptual grounding，会 mislead agent。

---

## 12. 一些我观察到的设计哲学

### 12.1 Hierarchy of abstraction
- Latent dynamics (backbone)：global, causal, long-horizon consistency
- Pixel dynamics (decoder)：local, perceptual, fine-grained
- 这种分工让每个 component 专注自己的 strength，避免 JEPA 的 "全 latent" 或 video gen 的 "全 pixel" 的极端

### 12.2 "Absorb uncertainty, don't short-circuit it"
LeCun 说 future details unpredictable → 不要 reconstruct pixels。PAN 说 future details unpredictable → **用 diffusion 的 stochastic process 把不确定性显式 model 出来**，而不是绕过它。这是 fundamentally different 的哲学立场。

### 12.3 Self-conditioning at inference
$\hat{s}'_k = [\hat{s}_k, h(\hat{o}_k)]$ 这个 trick 在 inference 时把 model 的 output 重新 feed back 进 conditioning，本质上是一个 closed-loop self-correction。类比人类思考：imagined scenario → re-perceive it → 基于新 perception 继续推理。

### 12.4 Conversational framing for world simulation
把 multi-turn dialogue structure 借用过来做 sequential simulation：每个 assistant turn = 一个 predicted state。这让 VLM 的预训练知识直接 transfer 到 world simulation，无需重新 design architecture。

---

## 13. 局限和 future work

Paper 自己提到的：
1. Current implementation 是 streamlined autoregressive backbone，没有 mixed discrete-continuous representation（GLP 原始 paper Xing et al. 2025 设想的）
2. Modality 限制在 video + text，没有 audio, tactile 等
3. 10 rounds history 限制（Qwen2.5-VL context window）

我观察到的：
- 256 query tokens 是 fixed-size state representation，可能 insufficient for very complex scenes
- VAE compression 4× temporal 意味着 motion resolution 受限
- 960× H200 的训练 cost 意味着复现门槛很高

---

## References

- **PAN paper** (this one): MBZUAI Institute of Foundation Models
- **GLP original**: Xing et al., "Critiques of World Models", arXiv:2507.05169, 2025 — https://arxiv.org/abs/2507.05169
- **JEPA**: Assran et al., "Self-Supervised Learning from Images with a JEPA", CVPR 2023 — https://arxiv.org/abs/2301.08243
- **V-JEPA 2**: Assran et al., 2025 — https://arxiv.org/abs/2506.09985
- **DINO-WM**: Zhou et al., "DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning", ICML 2025 — https://dinowm.github.io/
- **Wan2.1**: Wan Team, arXiv:2503.20314 — https://arxiv.org/abs/2503.20314
- **Qwen2.5-VL**: Bai et al., arXiv:2502.13923 — https://arxiv.org/abs/2502.13923
- **Flow Matching**: Lipman et al., arXiv:2210.02747 — https://arxiv.org/abs/2210.02747
- **Rectified Flow**: Liu et al., arXiv:2209.03003 — https://arxiv.org/abs/2209.03003
- **Shift-Window DPM**: Feng et al., "The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control", arXiv:2412.03568 — https://arxiv.org/abs/2412.03568
- **Sora**: Brooks et al., "Video Generation Models as World Simulators", OpenAI 2024 — https://openai.com/research/video-generation-models-as-world-simulators
- **Cosmos**: NVIDIA, arXiv:2501.03575 — https://arxiv.org/abs/2501.03575
- **Genie 2**: Parker-Holder et al., DeepMind 2024 — https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- **WorldScore**: Duan et al., arXiv:2504.00983 — https://arxiv.org/abs/2504.00983
- **WM-ABench**: Gao et al., arXiv:2506.21876 — https://arxiv.org/abs/2506.21876
- **Agibot**: Bu et al., arXiv:2503.06669 — https://arxiv.org/abs/2503.06669
- **SageAttention2++**: Zhang et al., arXiv:2505.21136 — https://arxiv.org/abs/2505.21136
- **FlexAttention**: Dong et al., arXiv:2412.05496 — https://arxiv.org/abs/2412.05496
- **Ulysses / DeepSpeed-SP**: Jacobs et al., arXiv:2309.14509 — https://arxiv.org/abs/2309.14509

---

## 14. 给 Karpathy 的几个思考点

1. **GLP vs JEPA 的哲学对立**：这本质上是你和 LeCun 一直在辩论的 "predict in pixel space vs latent space" 的具体 instance。PAN 的论证是：pure latent 预测会 collapse，pure pixel 预测会 unstable，**pixel-supervised latent prediction** 是 sweet spot。这个论证 empirical 上 working（experiment 数据），但理论上 collapse 问题的根本解决还需要更深入的分析。

2. **Causal Swin-DPM 的 fuzzy conditioning 是非常 elegant 的 trick**：本质上是用 noise level 来 control information bottleneck。前 chunk 处于 $K/2$ noise level 时，high-frequency detail 被抹掉，model 只能依赖 low-frequency semantic structure。这跟你的 "information bottleneck is a feature, not a bug" 的观点 resonance。

3. **Self-conditioning at inference（$\hat{s}'_k = [\hat{s}_k, h(\hat{o}_k)]$）**：这个 trick 让我想到你的 recurrent mode 之于 transformer 的辩论。PAN 在 inference 时把 model 的 output 重新编码回 input，本质上创造了一个 recurrent loop on top of feedforward transformer。这可能是 long-horizon consistency 的关键。

4. **256 query tokens 作为 state representation**：这个 fixed-size bottleneck 是不是太 rigid？对比 Genie 2 的 latent tokens 数量、Cosmos 的 latent dimension，这是个值得 ablate 的 design choice。

5. **LLM 作为 world model backbone 的 grounding 作用**：paper 说 Qwen2.5-VL 的 text pretraining 提供 "real-world knowledge" 来 mitigate information sparsity。但这个 claim 没有被 ablate —— 一个 from-scratch 的 backbone（没有 text pretraining）会怎样？这个 ablation 缺失让我对 "LLM knowledge helps world modeling" 的 claim 保持 skepticism。

如果需要进一步讨论某个具体方面（比如 Causal Swin-DPM 的实现细节、training stability、或者和 Genie 2 / Cosmos 2 的 head-to-head comparison），告诉我。
