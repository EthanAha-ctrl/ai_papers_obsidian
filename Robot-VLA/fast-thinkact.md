---
source_pdf: fast-thinkact.pdf
paper_sha256: 6ed51b092c1ef9e0b53c86df08304e1de8a19024ecf3938c98d1d614b2baae5f
processed_at: '2026-08-04T07:22:04-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话聊聊 Fast-ThinkAct

## 1. 这篇 paper 在干嘛

一句话总结：**把 robot 的"思考过程"从一段啰嗦的英文，压缩成 6 个连续向量，然后还能照样干活，甚至干得更好更快。**

背景是这样。现在的 reasoning VLA（比如 ThinkAct、MolmoAct）让 robot 先 generate 一段 ~250 token 的 chain-of-thought，比如 "我看到桌上有一个红色杯子，目标是要把它放到左边，所以我要先往右伸手，再抓起来，再往左移……"，然后再根据这段思考输出 action。好处是 generalization 强、能 recovery from failure、能做 long-horizon 任务；坏处是生成 250 个 token 在 7B 模型上要 7 秒，robot 控制频率掉到 0.1 Hz，根本没法实时用。

而真实 robot manipulation 需要 1–15 Hz。所以这里有个直接的矛盾：**reasoning 要慢思考，control 要快反应**。

之前有人试过几个折中方案：
- ECoT-Lite 直接把 reasoning dropout 掉——掉点
- RL length penalty 让 LLM 说短点——还是掉点
- 把 reasoning 截断到 6 个 text token——更掉点

直觉上，discrete text 的信息密度太低，6 个 text token 装不下一个 reasoning trace 的内容。但作者发现：**6 个 continuous latent vector 能装下**，甚至比原 teacher 的 250 token text 还要好。

这就是 paper 的核心 finding。

---

## 2. 为什么 continuous latent 比 discrete text 信息密度高

这是整个 paper 最反直觉、也最关键的 insight。我来拆解一下。

一个 7B 的 VLM，hidden size 通常是 2048 或 4096。一个 token 经过 embedding lookup 之后是一个 2048 维向量；但 discrete token 的问题在于：它是从一个 vocab（比如 32K）里选出来的，**梯度没法流回**——你只能通过 embedding 的梯度间接学习，而 embedding 是离散选择后的产物。

而 continuous latent 不经过 vocab lookup，直接是 2048 维的 float vector，每个 dimension 都可以承载信息，梯度可以直接 backprop 到 visual encoder。所以 6 个 latent token × 2048 dim = 12288 个 float，信息容量远大于 6 个 discrete token。

更关键的是：continuous latent 可以学到 dense representation，而 text token 必须从有限 vocab 里选，lossy。

我以前在 OpenAI 做 VAE 相关的工作时就注意到这一点——continuous representation 的 bottleneck 可以比 discrete 更窄但信息保留更多。这篇 paper 在 reasoning 这个场景验证了这个 intuition。

---

## 3. 整个 pipeline 怎么搭起来

他们其实搭了一个 teacher-student 的框架，分成四个训练阶段。

### Stage 1: SFT
拿 Qwen2.5-VL-3B，在 4M 个样本上做 supervised fine-tuning。数据包括 2D visual trajectories（从 OXE 和 AIST 标注出来的）、各种 QA 数据集（PixMo、RoboFAC、RoboVQA、ShareRobot、EgoPlan）。目的是让 VLM 先有基本的 visual understanding 和 embodied 知识。

### Stage 2: CoT-SFT
从 SFT 数据里抽 5%（200K），加上 165K Video-R1-CoT 数据，训练模型生成<answer>...</answer>` 格式的 reasoning。这一步让模型学会"怎么 reason"。

### Stage 3: Teacher-Student 训练（核心）

这是 paper 最巧妙的部分。

**Teacher $\mathcal{F}_\theta^T$**：从 CoT-SFT checkpoint 初始化，用 GRPO 训练，reward 是 ThinkAct 那套 action-aligned visual reward + QA reward。每个 prompt rollout 5 条 reasoning trace，算 group-relative advantage $A(\tau)$。

**Student $\mathcal{F}_\theta$**：也从 CoT-SFT checkpoint 初始化，但不输出 text，而是 autoregressively 生成 6 个 continuous latent vector $z = \{z_m\}_{m=1}^6$，每个 $z_m \in \mathbb{R}^{2048}$。

然后 student 还有 5 个 learnable spatial tokens $s_i$，append 在 latent sequence 后面，每个 token 的 output hidden state 经过一个 MLP，并行 decode 出 5 个 waypoints。

**Verbalizer $\mathcal{V}_\psi$**：拿 Qwen3-0.6B 当 decoder，每层插 cross-attention 接收 latent $z$，把 $z$ 翻译回自然语言。它的作用是给 latent 一个 text-space anchor，避免 latent collapse 成无意义的 free vector。

### Stage 4: Policy Learning
冻结 student $\mathcal{F}_\theta$，接一个 diffusion Transformer action model（DiT-Policy 或 RDT），把 spatial tokens 的 early-layer KV cache 抽出来作为 condition $c_t$，训练 action model 输出 7-DOF 或 14-DOF action chunk。

Inference 时只用 student + action model，verbalizer 丢掉，所以 latency 完全不受 verbalizer 影响。

---

## 4. 三个 loss 各自在干嘛

这是 paper 设计最精巧的地方。三个 loss 分别从语义、几何、动作三个层面约束 latent。

### Loss 1: $\mathcal{L}_{\mathrm{verb}}$ — 语义层（preference DPO）

公式 (4)：

$$\mathcal{L}_{\mathrm{verb}} = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{p_\psi(\tau^+|\mathbf{z})}{p_{\mathrm{ref}}(\tau^+)} - \log\frac{p_\psi(\tau^-|\mathbf{z})}{p_{\mathrm{ref}}(\tau^-)}\right)\right)\right]$$

变量解释：
- $\tau^+$ 是 teacher rollout group 里 advantage 最高的 trace（好 reasoning）
- $\tau^-$ 是 advantage 最低的 trace（差 reasoning）
- $p_\psi(\tau|\mathbf{z})$ 是 verbalizer 在 latent $z$ 条件下生成 $\tau$ 的概率
- $p_{\mathrm{ref}}(\tau)$ 是 reference（verbalizer 不接收 latent 时的 prior）
- $\beta = 0.1$ 控制 preference strength
- $\sigma$ 是 sigmoid

这是 DPO 的变体。原版 DPO 用 $\pi/\pi_{\mathrm{ref}}$ 作为 implicit reward，这里把 reward 拆成 "verbalizer 在 latent z 条件下生成 τ 的对数似然比"。

直觉：希望 latent $z$ 被 verbalizer 解码后，偏向高质量 reasoning $\tau^+$，远离低质量 $\tau^-$。

这里有个很巧妙的设计：**preference label 直接来自 GRPO 的 advantage**，不需要额外标注。GRPO 训 teacher 时本来就 rollout 5 条，advantage 高的就是好 trace，低的就是差 trace。这是把 RL 的信号 reuse 给 distillation。

训练 schedule 也讲究：前 3000 iter 用标准 LM loss warm up verbalizer，让 $z$ 和 verbalizer 先对齐；后 1500 iter 冻结 verbalizer，启用 $\mathcal{L}_{\mathrm{verb}}$ 让 student 主动调整 $z$。

### Loss 2: $\mathcal{L}_{\mathrm{distill}}$ — 几何层（hidden state alignment）

公式 (5)：

$$\mathcal{L}_{\mathrm{distill}} = \|h_t^T - h_t\|_2^2$$

- $h_t^T$ 是 teacher forward 到 `<answer>` token 时的 hidden state（取 $\tau^+$ 对应的那个 rollout）
- $h_t$ 是 student forward 到对应 `<answer>` 位置的 hidden state

这一步只对齐 hidden state，不对齐 token sequence。好处是 student 不需要模仿 verbose text，但能继承 teacher 在 `<answer>` 位置编码的 visual plan 信息。

直觉：ThinkAct 的 `<answer>` token 实际编码的是 "where to move" 的视觉规划，这个信息在 hidden state 里已经 compact 了，不需要再 expand 成 text。直接对齐 hidden state 反而更 efficient。

### Loss 3: $\mathcal{L}_{\mathrm{ans}}$ — 动作层（waypoint prediction）

公式 (6)：

$$\mathcal{L}_{\mathrm{ans}} = \sum_{i=1}^K \|p_i - \hat{p}_i\|_2^2, \quad p_i = \mathrm{MLP}(h'(s_i))$$

- $K=5$ 是 waypoint 数量
- $s_i$ 是第 $i$ 个 learnable spatial token
- $h'(s_i)$ 是 spatial token 经过 student VLM 后的 output hidden
- $p_i \in \mathbb{R}^6$ 格式 $[x_{\mathrm{single}}, y_{\mathrm{single}}, x_{\mathrm{left}}, y_{\mathrm{left}}, x_{\mathrm{right}}, y_{\mathrm{right}}]$
- $\hat{p}_i$ 是 ground-truth waypoints

这一步把 autoregressive 的 60-70 token waypoint sequence（textual teacher 的做法）压缩成 5 个并行 token。代价是损失了 waypoint 之间的顺序 conditioning，但实验证明对 manipulation 影响很小，因为 waypoints 本身是 spatial 目标点，不是 strict temporal sequence。

### 三个 loss 合起来

$$\mathcal{L}_{\mathrm{student}} = \mathcal{L}_{\mathrm{verb}} + \mathcal{L}_{\mathrm{distill}} + \mathcal{L}_{\mathrm{ans}}$$

- $\mathcal{L}_{\mathrm{verb}}$：latent 能被解码为高质量 reasoning（语义层）
- $\mathcal{L}_{\mathrm{distill}}$：latent hidden 对齐 teacher 的视觉规划（几何层）
- $\mathcal{L}_{\mathrm{ans}}$：spatial tokens 直接预测 waypoints（动作层）

三层约束形成了一个从 abstract semantic 到 concrete spatial 再到 executable action 的 hierarchy。这个设计很 elegant。

---

## 5. Policy Learning 的一个小 trick

公式 (7)：

$$\mathcal{L}_{\mathrm{IL}}(\phi) = \ell\left(\pi_\phi(o_t, l, c_t), \hat{a}_t\right)$$

- $\pi_\phi$ 是 diffusion Transformer action model（DiT-Policy 或 RDT）
- $c_t$ 是 visual plan latent，**从 spatial tokens 的 early-layer KV cache 抽取**
- $\ell$ 是 diffusion denoising loss

这里有个反直觉的 ablation：用 early-layer KV cache（89.7 on LIBERO）比 late-layer（88.3）和 output hidden（87.1）都好。

直觉：early layer 还保留更多 spatial/visual detail，late layer 已经偏向 abstract semantics。对于 action prediction，spatial detail 比 abstract semantics 更重要。这跟 Magma、HAMSTER 的发现一致——它们也用 mid-layer features 做 spatial representation。

还有一个工程细节：用 linear projection 把 VLM 的 KV 维度映射到 action model 的维度（DiT 1024, RDT 2048）。这个 projector 是可训练的，其余部分冻结。

---

## 6. 实验结果里最值得关注的几点

### 6.1 Latency 降幅惊人

| Model | Latency (ms) |
|---|---|
| ThinkAct-7B | 7513 |
| ThinkAct-3B | 5674 |
| **Fast-ThinkAct-3B** | **805** |

vs ThinkAct-7B 是 9.3× speedup，vs ThinkAct-3B 是 7× speedup。这个 gain 主要来自：
- 6 个 latent tokens 并行生成（vs 250 tokens autoregressive）
- 5 个并行 spatial tokens（vs 60-70 token autoregressive waypoint sequence）

### 6.2 Student > Teacher（反直觉）

Table 3 显示：
- Fast-ThinkAct-3B: 52.8 (avg on 3 benchmarks)
- Textual Teacher: 49.8
- SFT + CoT-SFT: 45.0

3B student 超过 7B textual teacher，这是 paper 最强的 ablation 之一。解释：DPO preference 过滤掉了 teacher 的低质量 trace。Fig.7 和 Fig.10 显示 teacher 有时会产生 red 错误 step，而 student 的 verbalized reasoning 更 concise 更聚焦。

这其实揭示了一个更深的 insight：**verbose text reasoning 不是最优 representation**。text 里有冗余、有错误、有 self-referential noise。DPO preference + continuous latent 反而能 denoise。

### 6.3 Textual reasoning 怎么压都掉点

Table 6：

| Variant | Avg |
|---|---|
| Textual Teacher FT | 49.8 |
| Inference w/o thinking (0 tokens) | 46.5 |
| Inference w/ 6 textual tokens | 46.3 |
| FT + RL Length-Penalty (~50 tokens) | 47.8 |
| **Fast-ThinkAct-3B (6 latent)** | **53.3** |

直接把 text truncate 到 6 token 反而掉点（46.3 < 49.8），说明 discrete token compression 不可行。只有 continuous latent 才能携带足够信息（53.3 > 49.8）。这是 paper 的核心 claim 的直接验证。

### 6.4 Long-horizon 和 failure recovery 提升

RoboTwin2.0 long-horizon 任务（average length > 270 steps）：
- RDT: 35.0 (easy) / 12.3 (hard)
- ThinkAct: 42.8 / 15.3
- Fast-ThinkAct: **48.8 / 16.8**

RoboFAC failure recovery：
- vs 第二名 +10.9 (simulation) / +16.4 (real-world)

这些任务正是 reasoning VLA 相对 foundation VLA 的优势所在，而 Fast-ThinkAct 在这些任务上比 textual reasoning VLA 还强，说明 latent reasoning 没有损失 long-horizon 和 failure recovery 能力。

### 6.5 Ablation: M=6 最优

- M=1：容量不足，性能下降
- M=6：最优
- M=30, 100：引入冗余/噪声，性能下降

这跟 Coconut 的发现一致：latent steps 太少不够表达，太多会 overfit 噪声。M=6 是个 sweet spot，但 paper 没做 task-adaptive M，这可能是个 future direction。

---

## 7. 几个我觉得巧妙的 design choice

### 7.1 GRPO advantage 当 preference label

这是 paper 最聪明的设计。一般 DPO 需要人工标注 preference pair，或者用 reward model 打分。这里直接 reuse GRPO 的 group-relative advantage：每个 rollout group 里 advantage 最高的就是 $\tau^+$，最低的就是 $\tau^-$。

好处：
- 不需要额外标注
- 信号是 action-aligned 的（reward 本身是 action-aligned visual reward）
- 自动 denoise（group 内归一化消除 baseline 偏置）

### 7.2 Verbalizer 是 training-only auxiliary

verbalizer 只在 training 时存在，inference 时丢掉。这是 "training-time auxiliary decoder" 范式，让人联想到：
- Diffusion 的 classifier-free guidance（训练时用 condition，inference 可选）
- VQ-VAE 的 codebook learning（auxiliary loss 帮 representation 学习）
- Contrastive learning 的 projector head（训练时用，inference 丢）

这种范式的好处是：representation 学习可以借助一个 auxiliary task（这里是把 latent decode 回 text），但 inference 时不承担这个 auxiliary 的 cost。

### 7.3 Spatial tokens 并行预测 waypoints

Textual teacher 要 autoregressive 生成 5 个 waypoints，每个 waypoint 12-14 token，总共 60-70 token。Student 改成 5 个 learnable spatial token 并行 decode。

代价是损失了 waypoint 之间的 sequential conditioning，但好处是巨大的 latency reduction。实验证明对 manipulation 影响小，因为 waypoints 本身是 spatial 目标点，不是 strict temporal sequence——你先到 $p_1$ 还是 $p_2$ 其实是 action model 决定的，plan 层只需要给出目标点。

### 7.4 Early-layer KV cache > Late-layer

反直觉但合理。early layer 保留更多 spatial detail，late layer 偏向 abstract semantics。action prediction 需要的是 "where to move"，是 spatial info，所以 early layer 更合适。

---

## 8. 几个我觉得可疑或可改进的地方

### 8.1 Verbalizer hallucination

作者自己承认 verbalizer 可能产生 plausible but inaccurate 描述。虽然不影响 action（因为 inference 不用 verbalizer），但影响 interpretability。如果想用 verbalized reasoning 做 safety check 或 human supervision，这个 hallucination 是个问题。

未来可以加 grounding-aware loss，或者用 visual grounding 模块约束 verbalizer。

### 8.2 M=6 是 hand-tuned

paper 在 Fig.8 ablation 验证 M=6 最优，但这是 global 选择。不同任务可能最优 M 不同：short-horizon 可能 M=3 就够，long-horizon 可能 M=10 更好。task-adaptive M 或 learnable M 是个 future direction。

### 8.3 训练成本高

16× A100 80GB，四阶段训练。对 academic lab 不友好。未来如果能把 teacher-student 合并成 single-stage online distillation，或者用 offline RL 直接训 latent，会简化 pipeline。

### 8.4 依赖 Molmo + CoTracker3 标注

ground truth waypoint 质量受限于这些标注器。Molmo-72B 和 CoTracker3 本身有 error，这些 error 会传到 student。

### 8.5 只在 simulation 评测

SimplerEnv-Google 虽然与 real 相关，但仍是 sim benchmark。真机 deployment 的 sim-to-real gap 没直接验证。

### 8.6 Long-horizon 的极限没测

RoboTwin2.0 long 任务平均 270-470 steps。更长的任务（比如 1000+ steps）表现如何没测。M=6 的 bottleneck 在超长任务上可能显现。

---

## 9. 延伸联想

### 9.1 和 Coconut 的关系

Coconut（Hao et al., 2024）是 LLM-only 的 continuous CoT work，用 `<bot>`/`<eot>` special token 切换 latent mode。Fast-ThinkAct 借鉴了这个思路，但 key difference 是它需要同时处理 visual spatial info 和 grounding 到 action，所以加了 spatial tokens 和 trajectory distillation。

未来如果能把 Coconut 的 latent BFS search 和 Fast-ThinkAct 的 latent planning 结合，在 latent space 做 tree search，可能进一步提升 long-horizon 能力。

### 9.2 和 DreamerV3 的联系

DreamerV3 在 latent space 做规划，学习 "未来状态" 的压缩表示。Fast-ThinkAct 的 $c_t$ 实际上是个 implicit world model latent——它编码了 "未来要去哪里" 的信息。

如果把 $c_t$ 扩展成 explicit world model（predict future observation 而不只是 waypoint），可能让 action model 有更强的 forward planning 能力。

### 9.3 和 Test-time Compute Scaling 的关系

o1-style test-time compute scaling 是让 model 在 inference 时多思考（longer reasoning chain）。Fast-ThinkAct 反过来，把 test-time compute "内化"到 latent space，避免 inference latency。

这是 test-time compute 的另一种范式：**train-time compute heavy, inference-time compute light**。与 o1 的 **inference-time compute heavy** 互补。未来可能结合：train 时把 reasoning 压缩到 latent，inference 时在 latent space 做 adaptive depth search。

### 9.4 和 MoE 的潜在结合

M=6 个 latent tokens 可以看作 "6 个 reasoning experts"。未来可以探索 MoE-style routing，让不同 latent token 负责不同 reasoning sub-task（spatial reasoning、temporal reasoning、failure recovery 等）。

### 9.5 和 Neural Turing Machine 的呼应

M 个 latent token 类似 "working memory slots"。不过 M=6 太小，未来扩展到 larger memory bank（比如 M=64）可能支持更复杂的 multi-step reasoning。结合 differentiable memory 机制可能让 latent reasoning 有更强的 expressive power。

---

## 10. 给 Karpathy 的核心 takeaway

1. **核心 finding**：6 个 continuous latent vector 在 reasoning VLA 上可以 outperform 250 token 的 textual CoT，同时 7-9× speedup。这验证了 continuous representation 的 information density 优势在 embodied reasoning 场景依然成立。

2. **核心 trick**：用 GRPO 的 group-relative advantage 作为 DPO preference label，把 RL signal reuse 给 distillation，避免在 latent space 直接定义 reward 的难题。

3. **核心 design**：三层 loss（semantic + geometric + action）+ training-only verbalizer + parallel spatial tokens + early-layer KV conditioning。每个组件都对应一个具体的 representation bottleneck。

4. **核心 limitation**：verbalizer hallucination、M hand-tuned、训练成本高、只测 sim、long-horizon 极限未测。

5. **核心 future direction**：latent space tree search、explicit world model、task-adaptive M、single-stage online distillation。

从 representation learning 的角度，这篇 paper 其实是在验证一个老的直觉——**continuous > discrete for information compression**——但放在了 embodied reasoning 这个新场景。结合 DPO preference 和 RL advantage 的 signal reuse 是工程上的亮点。从 reasoning VLA 的 evolution 看，它把 "slow thinking" 内化成 "fast latent"，是 reasoning VLA 走向 real-time 的关键一步。

**Web References**:
- Fast-ThinkAct: https://arxiv.org/abs/2507.16815
- ThinkAct: https://arxiv.org/abs/2507.16815
- Coconut: https://arxiv.org/abs/2412.06769
- CODI: https://arxiv.org/abs/2502.21074
- Soft Thinking: https://arxiv.org/abs/2505.15778
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290
- OpenVLA: https://arxiv.org/abs/2406.09246
- π₀: https://arxiv.org/abs/2410.24164
- RDT-1B: https://arxiv.org/abs/2410.07864
- Magma: https://arxiv.org/abs/2502.13130
- MolmoAct: https://arxiv.org/abs/2508.07917
- CoT-VLA: https://arxiv.org/abs/2503.22020
- TraceVLA: https://arxiv.org/abs/2412.10345
- Embodied CoT: https://arxiv.org/abs/2407.08693
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DreamerV3: https://arxiv.org/abs/2301.04104
- Neural Turing Machines: https://arxiv.org/abs/1410.5401
- Differentiable Neural Computer: https://arxiv.org/abs/1807.03819

---

# Fast-ThinkAct: Verbalizable Latent Planning for Efficient VLA Reasoning

## 1. 问题背景与 Motivation

当前 reasoning VLA（如 ThinkAct、MolmoAct、ECoT）通过 explicit chain-of-thought 显著提升了 generalization 与 failure recovery 能力，但生成 ~250 tokens 的 textual reasoning 会让单次决策耗时达数秒（0.1 Hz）。而真实机器人 manipulation 需要 1–15 Hz 的控制频率，这个 latency gap 成为 embodied AI 的硬瓶颈。

Fast-ThinkAct 的核心 question：**能否把 verbose textual CoT 压缩为少数几个 continuous latent tokens，同时保留 reasoning 能力、可解释性，并最终 grounding 到 low-level action？**

它给出的答案是 **verbalizable latent planning** + **preference-guided distillation** + **trajectory-aligned visual planning**。

参考链接：
- ThinkAct: https://arxiv.org/abs/2507.16815
- Coconut (continuous CoT in LLM): https://arxiv.org/abs/2412.06769
- CODI (self-distillation of CoT): https://arxiv.org/abs/2502.21074
- Soft Thinking: https://arxiv.org/abs/2505.15778
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290

---

## 2. 整体架构图解析（Figure 2）

架构分成两个阶段：

### (a) Teacher–Student Latent Reasoning Distillation

```
Observation o_t + Instruction l
        │
        ├──────────────────────────────────┐
        │                                   │
        ▼ Textual Teacher F_θ^T           ▼ Latent Student F_θ
   (GRPO-trained, explicit CoT)       (generates M=6 latent tokens z ∈ R^d)
        │                                   │
   samples N rollouts {τ_i}            z = {z_m}_{m=1..M}
   computes advantages A(τ_i)              │
   selects τ⁺ = argmax A, τ⁻ = argmin A   │
        │                                   ▼
        │                           Verbalizer LLM V_ψ
        │                           (Qwen3-0.6B + cross-attn on z)
        │                           decodes z → text τ̂
        │                                   │
        └────→ L_verb (DPO on τ⁺ vs τ⁻) ◀──┘
                                           
   h_t^T (teacher's <answer> hidden) → L_distill ← h_t (student's <answer> hidden)
                                           
   K learnable spatial tokens s_i → L_ans → waypoints p_i
```

### (b) Reasoning-Enhanced Policy Learning

```
F_θ (frozen) → spatial token KV cache (early layers)
                 │ c_t (visual plan latent)
                 ▼
          Action Model π_φ (DiT/RDT)
                 │ cross-attn on (state, c_t)
                 ▼
            action chunk a_t (7- or 14-DOF)
```

**Key intuition**: latent tokens `z` 承担 "compressed CoT" 角色，spatial tokens `s_i` 承担 "visual sub-goal waypoints" 角色，二者一起 form 一个 visual plan latent `c_t`，再 conditioning 一个 diffusion-based action model。

---

## 3. 方法详解（含公式变量解释）

### 3.1 Teacher: GRPO with Action-Aligned Visual Rewards

公式 (1) — GRPO objective:

$$\mathcal{L}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{\tau \sim \mathcal{F}_\theta^T}\left[\min\left(r_\theta(\tau)A(\tau),\ \mathrm{clip}(r_\theta(\tau), 1-\epsilon, 1+\epsilon)A(\tau)\right)\right]$$

变量含义：
- $\tau$：一个 reasoning trace（teacher 自回归生成的 textual CoT + answer）
- $\mathcal{F}_\theta^T(\tau)$：teacher 当前 policy 对 τ 的概率
- $\mathcal{F}_{\mathrm{old}}^T(\tau)$：rollout 时使用的旧 policy 对 τ 的概率
- $r_\theta(\tau) = \frac{\mathcal{F}_\theta^T(\tau)}{\mathcal{F}_{\mathrm{old}}^T(\tau)}$：importance sampling ratio（PPO 标准项）
- $A(\tau)$：group-relative advantage
- $\epsilon$：clip 范围（通常 0.2），防 ratio 漂移过远
- $\min(\cdot,\cdot)$：standard PPO pessimistic bound

公式 (2) — Group-relative advantage:

$$A(\tau) = \frac{R_\tau - \mathrm{mean}(\{R_i\}_{i \in G(\tau)})}{\mathrm{std}(\{R_i\}_{i \in G(\tau)})}$$

- $G(\tau)$：同一 prompt 的 rollout group（这里 N=5）
- $R_i$：第 i 个 rollout 的 reward（来自 ThinkAct 的 action-aligned visual reward + QA reward）
- mean/std：在 group 内归一化，消除 baseline 偏置

**Insight**: 这里 advantage 既是 RL 训练 teacher 的信号，又被复用为 **preference label** — `A(τ)` 高意味着该 reasoning 在该 group 中导致了更好的视觉规划对齐 + QA 正确性。

公式 (3) — Preference pair 构造:

$$\tau^+ = \arg\max_{\tau \in G} A(\tau), \qquad \tau^- = \arg\min_{\tau \in G} A(\tau)$$

每个 group 取 advantage 最高与最低两条 trace，作为后续 DPO-style distillation 的正负样本。

---

### 3.2 Student: Verbalizable Latent CoT via Preference Optimization

Student 不输出 text，而是 autoregressively 生成 M=6 个 continuous latent vectors:

$$\mathbf{z} = \{z_m\}_{m=1}^M, \quad z_m \in \mathbb{R}^d$$

- $M=6$：latent reasoning steps（paper 在 Fig.8 ablation 验证 M=6 最优，M=1 容量不足，M=30/100 引入冗余噪声）
- $d$：VLM hidden size（Qwen2.5-VL-3B 为 2048）

**Verbalizer** $\mathcal{V}_\psi$：以 Qwen3-0.6B 为 backbone，每层插入 cross-attention 接收 z 作为 condition，把 z 解码回自然语言 $\hat{\tau}$。它的作用类似 "latent → text translator"，让 latent space 受 text-space preference 约束，避免 latent 退化成无意义的 free vector。

公式 (4) — Verbalizer preference loss (DPO 形式):

$$\mathcal{L}_{\mathrm{verb}} = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{p_\psi(\tau^+|\mathbf{z})}{p_{\mathrm{ref}}(\tau^+)} - \log\frac{p_\psi(\tau^-|\mathbf{z})}{p_{\mathrm{ref}}(\tau^-)}\right)\right)\right]$$

变量含义：
- $p_\psi(\tau|\mathbf{z})$：verbalizer 在 latent z 条件下生成 τ 的概率
- $p_{\mathrm{ref}}(\tau)$：reference model（即 verbalizer 不接收 latent conditioning 时的 prior）
- $\tau^+$, $\tau^-$：来自公式 (3) 的 preference pair
- $\beta = 0.1$：preference strength，控制偏离 reference 的力度
- $\sigma$：sigmoid

这其实是 DPO 的变体。原版 DPO 用 $\pi/\pi_{\mathrm{ref}}$ 作为 implicit reward；这里把 reward 拆成 "verbalizer decode latent z 给出 τ" 的对数似然比。**直觉**：希望 latent z 被 verbalizer 解码后偏向高质量 reasoning τ⁺，而远离低质量 τ⁻。

训练 schedule 很关键（Appendix A.2）：
- 前 3000 iter：用标准 LM loss，target 是 τ⁺，warm up verbalizer 与 z 的 alignment
- 后 1500 iter：冻结 $\mathcal{V}_\psi$，启用 $\mathcal{L}_{\mathrm{verb}}$ 让 student 主动调整 z 朝高质量方向

---

### 3.3 Visual Plan Distillation: Action-Aligned Trajectory Alignment

公式 (5):

$$\mathcal{L}_{\mathrm{distill}} = \|h_t^T - h_t\|_2^2$$

- $h_t^T$：teacher forward 到 `<answer>` token 时的 hidden state（取 τ⁺ 对应的那个 rollout）
- $h_t$：student forward 到对应 `<answer>` 位置的 hidden state

**直觉**：只对齐 hidden state，不对齐 token sequence，避免 student 被迫模仿 verbose text。而 `<answer>` token 在 ThinkAct 中编码的就是 "where to move" 的视觉规划信息，所以这一步本质是把 visual plan capability 蒸到 latent space。

---

### 3.4 Parallel Spatial Tokens: Visual Trajectory Prediction

传统 textual teacher 需要自回归生成 5 个 waypoints，每个 waypoint 用 12-14 tokens 表达（坐标 + 分隔），总共 60-70 tokens。

Student 改成 **K=5 个 learnable spatial tokens** $\{s_i\}_{i=1}^K$，append 到 latent sequence 末尾，每个 token 的 output hidden state 同时经 MLP 投影成 waypoint，并行 decode：

公式 (6):

$$\mathcal{L}_{\mathrm{ans}} = \sum_{i=1}^K \|p_i - \hat{p}_i\|_2^2, \quad p_i = \mathrm{MLP}(h'(s_i))$$

- $K=5$：waypoints 数量
- $s_i$：第 i 个 learnable spatial token
- $h'(s_i)$：spatial token 经过 student VLM 后的 output hidden
- $p_i \in \mathbb{R}^6$：预测的 waypoint，格式 $[x_{\mathrm{single}}, y_{\mathrm{single}}, x_{\mathrm{left}}, y_{\mathrm{left}}, x_{\mathrm{right}}, y_{\mathrm{right}}]$
  - 前 2 维：单臂坐标
  - 后 4 维：双臂左右手坐标
  - 单臂任务时后 4 维 mask 掉
- $\hat{p}_i$：ground-truth waypoints（来自 Molmo-72B + CoTracker3 标注）

**Insight**: 这是把 autoregressive 60-70 token 序列 → 5 个并行 token 的关键设计。它的代价是损失了 waypoint 之间的顺序 conditioning，但实验证明对 manipulation 影响很小。

公式 (Student 总目标):

$$\mathcal{L}_{\mathrm{student}} = \mathcal{L}_{\mathrm{verb}} + \mathcal{L}_{\mathrm{distill}} + \mathcal{L}_{\mathrm{ans}}$$

三项分别承担：
- $\mathcal{L}_{\mathrm{verb}}$：latent 可被解码为高质量 textual reasoning（语义层）
- $\mathcal{L}_{\mathrm{distill}}$：latent hidden state 对齐 teacher 的视觉规划表示（几何层）
- $\mathcal{L}_{\mathrm{ans}}$：spatial tokens 直接预测 waypoints（动作层）

---

### 3.5 Reasoning-Enhanced Policy Learning

公式 (7):

$$\mathcal{L}_{\mathrm{IL}}(\phi) = \ell\left(\pi_\phi(o_t, l, c_t), \hat{a}_t\right)$$

- $\pi_\phi$：diffusion Transformer action model（DiT-Policy 或 RDT）
- $c_t$：visual plan latent，**从 spatial tokens 的 early-layer KV cache 抽取**
- $\ell$：standard diffusion denoising loss
- $\hat{a}_t$：ground-truth action chunk

**关键设计**（Appendix ablation）：
- 用 **early-layer** KV cache（不是 last-layer）效果最好：89.7 vs late-layer 88.3 vs output hidden 87.1（LIBERO）
- 直觉：early-layer 还保留更多 spatial/visual detail，last-layer 已经偏向 abstract semantics
- 训练时冻结 $\mathcal{F}_\theta$ 和 state encoder，只更新 $\pi_\phi$ 和一个 linear projector（把 VLM 的 KV 维度映射到 action model 的维度：DiT 1024, RDT 2048）

---

## 4. 训练 Pipeline 总结

整个训练分四阶段：

| 阶段 | 数据 | 目标 |
|------|------|------|
| **SFT** | 4M samples (visual traces + QA + PixMo) | 基础 VLM 能力 |
| **CoT-SFT** | 200K SFT subset + 165K Video-R1-CoT | 引入 `<think>...</think><answer>...</answer>` 格式 |
| **Teacher-Student** | 50K balanced subset | Teacher: GRPO；Student: $\mathcal{L}_{\mathrm{verb}} + \mathcal{L}_{\mathrm{distill}} + \mathcal{L}_{\mathrm{ans}}$ |
| **Policy Learning** | OXE (DiT) / OXE + ALOHA (RDT) | 训练 $\pi_\phi$ via $\mathcal{L}_{\mathrm{IL}}$ |

**Inference**: 只需要 student $\mathcal{F}_\theta$ + action model $\pi_\phi$。verbalizer $\mathcal{V}_\psi$ 仅用于 training 和可选 interpretability，**inference 不参与**，所以 latency 完全不受 verbalizer 影响。

---

## 5. 实验数据详解

### 5.1 Latency 对比（Figure 3f）

| Model | Params | Latency (ms) |
|---|---|---|
| ThinkAct-7B | 7B | 7513 |
| MolmoAct-7B | 7B | 6723 |
| ThinkAct-3B | 3B | 5674 |
| **Fast-ThinkAct-3B** | 3B | **805** |

- vs ThinkAct-7B：89.3% latency reduction（9.3×）
- vs MolmoAct-7B：88.0% reduction
- vs ThinkAct-3B：7.0× faster

这个 latency gain 主要来自：M=6 latent tokens（并行生成 + KV cache 短）+ K=5 并行 spatial tokens（取代 60-70 token autoregressive waypoint sequence）。

### 5.2 LIBERO 成功率（Figure 3a-e, Table 5）

| Method | LIBERO (avg) | SimplerEnv-Google |
|---|---|---|
| OpenVLA-7B | 76.5 | 40.2 |
| CoT-VLA-7B | 83.9 | — |
| ThinkAct-7B | 84.4 | 68.3 |
| MolmoAct-7B | 86.8 | 64.9 |
| ThinkAct-3B | 83.1 | 64.7 |
| **Fast-ThinkAct-3B** | **89.7** | **68.7** |

注意：3B student 超过 7B textual teacher，说明 latent 压缩反而去除了 teacher reasoning 中的噪声 trace（Fig.7/10 显示 teacher 有时会产生 red 错误 step）。

### 5.3 RoboTwin2.0 Bimanual Manipulation（Table 1）

按任务长度分组（short 80-100, medium 110-220, long 270-470 steps）：

| Model | Easy Avg | Hard Avg | Long Easy | Long Hard |
|---|---|---|---|---|
| DP | 43.1 | 0.6 | — | — |
| ACT | 45.5 | 3.5 | — | — |
| π₀ | 52.9 | 16.3 | — | — |
| RDT | 56.4 | 22.8 | 35.0 | 12.3 |
| ThinkAct | 62.4 | 24.7 | 42.8 | 15.3 |
| **Fast-ThinkAct** | **65.7** | **26.4** | **48.8** | **16.8** |

特别值得注意：在 long-horizon 任务上 Fast-ThinkAct 对 RDT 提升 +13.8 easy / +4.5 hard，验证 latent reasoning 对 long-horizon planning 的关键作用。

### 5.4 Embodied Reasoning（Table 2, Table 4）

| Method | EgoPlan-Bench2 | RoboVQA (B-Avg) | OpenEQA | Overall |
|---|---|---|---|---|
| GPT-4V | 32.6 | 26.8 | 49.6 | 36.4 |
| Gemini-2.5-Flash | 42.4 | 28.9 | 45.3 | 38.9 |
| ThinkAct-3B | 44.0 | 55.3 | 48.9 | 49.4 |
| **Fast-ThinkAct-3B** | **46.4** | **60.8** | **51.2** | **52.8** |
| ThinkAct-7B | 48.2 | 59.8 | 56.2 | 54.7 |
| **Fast-ThinkAct-7B** | 47.5 | **61.1** | **59.0** | **55.9** |

7B 版本上 OpenEQA 提升显著（+2.8 over ThinkAct-7B），说明 latent space 的 scaling 行为良好。

### 5.5 Failure Recovery（Figure 5, RoboFAC）

| Setting | RoboFAC-3B (2nd) | Fast-ThinkAct | Δ |
|---|---|---|---|
| Simulation | baseline | **+10.9** | — |
| Real-world | baseline | **+16.4** | — |

定性示例（Fig.5 右）：物体掉落时，Fast-ThinkAct 生成具体恢复 plan：(1) 手臂后退腾出空间 → (2) 横向调整对齐目标 → (3) 下降到合适高度。这种 structured recovery plan 在传统 VLA 中很难做到。

### 5.6 Few-Shot Adaptation（Figure 6, RoboTwin2.0）

仅用 10 demonstrations/task fine-tune，Fast-ThinkAct 在 medium 与 long-horizon 任务上均超过 π₀ 和 ThinkAct。这说明 latent reasoning 提供的 prior 帮助 small-data adaptation。

### 5.7 Ablation: Loss Components（Table 3, 7）

| Method | EgoPlan | RoboVQA | OpenEQA | Avg |
|---|---|---|---|---|
| Full | 46.4 | 60.8 | 51.2 | 52.8 |
| w/o $\mathcal{L}_{\mathrm{verb}}$ | 42.1 | 53.8 | 49.5 | 48.5 |
| w/o $\mathcal{L}_{\mathrm{verb}}, \mathcal{L}_{\mathrm{distill}}$ | 41.6 | 52.7 | 48.9 | 47.7 |
| Textual Teacher $\mathcal{F}_\theta^T$ | 41.7 | 58.2 | 49.4 | 49.8 |
| SFT + CoT-SFT | 40.0 | 46.1 | 48.8 | 45.0 |
| SFT only | 40.5 | 53.6 | 45.3 | 46.5 |

几个 insight：
1. 移除 $\mathcal{L}_{\mathrm{verb}}$ → EgoPlan 掉 4.3：preference guidance 让 latent 学到 "good reasoning pattern"
2. 再移除 $\mathcal{L}_{\mathrm{distill}}$ → 再掉 0.5：visual plan 对齐作用相对小，但仍有贡献
3. **Student > Teacher**（52.8 vs 49.8）：distillation 反而比原 teacher 更好，因为 DPO preference 过滤掉了 teacher 的低质量 trace
4. CoT-SFT vs SFT：在 OpenEQA 上 CoT-SFT 好但 EgoPlan 上反而差，说明 naive CoT supervision 会引入 verbosity 损害 structured reasoning

### 5.8 Ablation: Latent Steps M（Figure 8 / Table 8）

- M=1：容量不足，性能下降
- M=6：最优
- M=30, 100：引入冗余/噪声，性能下降

这与 Coconut 的发现一致：latent steps 太少不够表达，太多会 overfit 噪声。

### 5.9 与 efficient textual reasoning baselines 对比（Table 6）

| Variant | Avg |
|---|---|
| Textual Teacher FT | 49.8 |
| Inference w/o thinking (0 tokens) | 46.5 |
| Inference w/ 6 textual tokens | 46.3 |
| FT + RL Length-Penalty (~50 tokens) | 47.8 |
| **Fast-ThinkAct-3B (6 latent)** | **53.3** |

直接把 textual reasoning truncate 到 6 token 反而掉点（46.3 < 49.8），说明 discrete token compression 不可行；而 **continuous latent** 才能携带足够信息（53.3 > 49.8 even）。这是 paper 最强的 ablation 之一。

---

## 6. 与相关工作对比与延伸联想

### 6.1 Latent Reasoning in LLM
- **Coconut** (Hao et al., 2024)：把 LLM 的 reasoning 直接放在 continuous hidden state 中，用 `<bot>` / `<eot>` special token 切换 latent mode
- **CODI** (Shen et al., 2025)：self-distillation 把 explicit CoT 蒸到 continuous space
- **Soft Thinking** (Zhang et al., 2025)：在 concept space 生成 weighted 软 token
- **Compressed CoT** (Cheng & Van Durme, 2024)：dense representation 替代 sparse reasoning

Fast-ThinkAct 借鉴了这个思路，但**关键区别**：它需要同时处理 visual spatial information 和 grounding 到 action，所以引入了 spatial tokens + trajectory distillation，而 LLM-only 方法没有这个需求。

### 6.2 RL-based Reasoning
- **GRPO**（DeepSeekMath）：group-relative advantage，去 value network
- **DPO**：把 reward learning 隐式化为 preference 概率比
- **Length-penalty RL**（L1, Just-Enough Thinking, Stable RL for Efficient Reasoning）：鼓励 short CoT

Fast-ThinkAct 巧妙地把 GRPO 产生的 advantage 当作 preference signal 给 DPO，避免了在 latent space 直接定义 reward 的难题。

### 6.3 Reasoning VLAs
- **Embodied CoT** (Zawalski et al., 2024)：用 LLM 生成 pseudo CoT labels
- **Hi-Robot** (Shi et al., 2025)：hierarchical VLA
- **CoT-VLA** (Zhao et al., 2025)：visual goal generation 作为 CoT
- **MolmoAct** (Lee et al., 2025)：spatial representation reasoning
- **EO-1** (Qu et al., 2025)：interleaved V-L-A pretraining
- **ECoT-Lite** (Chen et al., 2025)：reasoning dropout 加速
- **ThinkAct** (Huang et al., 2025)：RL-based reasoning with visual reward

Fast-ThinkAct 与 ThinkAct 的关系最密切——它直接以 ThinkAct 为 teacher，用其 GRPO advantage 作为 preference signal，把 verbal reasoning 压缩成 6 个 latent token。可以把它理解为 "ThinkAct distilled into latent space"。

### 6.4 Foundation VLAs
- **OpenVLA**, **π₀**, **Magma**, **HAMSTER**, **TraceVLA**, **RDT-1B**, **DiT-Policy**

Fast-ThinkAct 的 action model 是 modular 的——可以接 DiT-Policy（用于 SimplerEnv）或 RDT（用于 LIBERO/RoboTwin2.0）。这种 agnostic to action model 的设计提升了通用性。

参考链接：
- OpenVLA: https://arxiv.org/abs/2406.09246
- π₀: https://arxiv.org/abs/2410.24164
- RDT-1B: https://arxiv.org/abs/2410.07864
- Magma: https://arxiv.org/abs/2502.13130
- TraceVLA: https://arxiv.org/abs/2412.10345
- CoT-VLA: https://arxiv.org/abs/2503.22020
- MolmoAct: https://arxiv.org/abs/2508.07917
- ECoT (Robotic Control via Embodied CoT): https://arxiv.org/abs/2407.08693

---

## 7. 关键 Insights 与潜在问题

### 7.1 为什么 latent 比 textual 更好？（53.3 > 49.8）

我的理解：
1. **Denoising effect**：DPO 用 preference pair 训练，自动过滤 teacher 的低质量 trace（teacher 有时产生 red 错误 step，见 Fig.10）
2. **Information density**：continuous vector 每个 dimension 都可以承载信息，6 tokens × 2048 dim = 12288 floats，远多于 6 个 discrete token
3. **No exposure bias**：autoregressive text generation 有 train/inference mismatch，latent 不存在这个问题
4. **Gradient flow**：latent hidden 可以直接 backprop 到 visual encoder，而 text token 经过 embedding lookup 后梯度信息弱

### 7.2 Verbalizer 的角色很妙

它**只在 training 时存在**，承担两个职能：
- 给 latent 一个 text-space anchor，避免 free collapse
- 通过 DPO 把 preference 信号传回 student

inference 时直接丢掉，所以 latency 不受影响。这是一种 "training-time auxiliary decoder" 范式，让人联想到 **diffusion model 的 classifier-free guidance**（训练时用 condition，inference 可选）和 **VQ-VAE 的 codebook learning**（auxiliary loss 帮助 representation 学习）。

### 7.3 Early-layer KV Cache > Late-layer

这个 ablation 反直觉。一般理解 deep layer 更 abstract、更 task-relevant。但这里 early layer 反而更好，说明：
- **spatial/visual detail 在 early layer 还未坍缩**
- late layer 偏向 textual semantic，对 action prediction 用处小
- 这与 Magma/HAMSTER 的发现一致——它们也用早期/mid-layer features 作为 spatial representations

### 7.4 Limitations

1. **Verbalizer hallucination**：作者承认 verbalizer 可能产生 plausible but inaccurate 描述。但因为它不参与 inference，所以不影响 action，只影响 interpretability。这是一个未来 work 方向（grounding-aware loss）。
2. **数据依赖 Molmo + CoTracker3 标注的 2D visual trajectories**：ground truth waypoint 质量受限于这些标注器。
3. **M=6 是 hand-tuned**：不同任务可能最优 M 不同，但 paper 没做 task-adaptive M。
4. **训练成本高**：16× A100 80GB，三阶段训练，对 academic lab 不友好。
5. **只在 simulation 评测 real-world correlation**：SimplerEnv-Google 是 sim benchmark，虽然与 real 相关但不是 real robot deployment。

### 7.5 延伸联想

1. **与 o1-style test-time compute scaling 的关系**：Fast-ThinkAct 实际上是把 test-time compute "内化"到 latent space，避免 inference latency。这是 test-time compute 的另一种范式——而非 scaling up reasoning length，把 reasoning 压缩到 dense representation。

2. **与 MoE 的潜在结合**：M=6 个 latent tokens 可以看作 "6 个 reasoning experts"，未来可探索用 MoE-style routing 让不同 latent token 负责不同 reasoning sub-task。

3. **与 Continuous CoT + Tree Search 的结合**：latent space 不受 token 离散限制，未来可以在 latent space 做 MCTS-style search（类似 Coconut 论文设想的 latent BFS）。

4. **与 World Model 的联系**：latent planning 本质上学到了 "未来状态" 的压缩表示。可以联想到 DreamerV3 这类 world model-based RL，它们也在 latent space 做规划。Fast-ThinkAct 的 c_t 实际上是一个 implicit world model latent。

5. **与 Neural Turing Machine / Differentiable Memory 的呼应**：M 个 latent token 类似 "working memory slots"，可以联想到 DNC、NTM 系列。不过这里 M=6 太小，未来可能扩展到 larger memory banks。

6. **潜在问题——latent forgetting**：M=6 的瓶颈是否会让 student 忘记某些 rare reasoning pattern？Table 3 显示 OpenEQA 上反而提升，可能因为 QA 任务对 reasoning 的需求被压缩得很好；但更复杂的 multi-step reasoning 可能受损。这是值得 future work 探索的。

参考链接：
- DreamerV3: https://arxiv.org/abs/2301.04104
- Neural Turing Machines: https://arxiv.org/abs/1410.5401
- Differentiable Neural Computer: https://arxiv.org/abs/1807.03819

---

## 8. 总结：核心 Take-aways

1. **核心 idea**：把 verbose textual CoT 蒸馏为 6 个 continuous latent tokens，同时保留 reasoning capability + 可解释性（通过 verbalizer）+ 视觉规划对齐。
2. **关键技术**：
   - GRPO advantage 作为 preference signal 给 DPO
   - 三层 loss：$\mathcal{L}_{\mathrm{verb}}$（语义）+ $\mathcal{L}_{\mathrm{distill}}$（视觉规划）+ $\mathcal{L}_{\mathrm{ans}}$（waypoint）
   - Spatial tokens 并行预测 waypoints，取代 autoregressive
   - Early-layer KV cache conditioning action model
3. **效果**：89.3% latency reduction，3B student 超过 7B teacher，long-horizon / failure recovery / few-shot 上都更强。
4. **核心 intuition building**：latent space 比 discrete text 更 information-dense，DPO 比 naive distillation 更能过滤噪声，parallel spatial tokens 比 autoregressive waypoint sequence 更高效。

这是一篇工程性很强的工作，把 RL preference、DPO、teacher-student distillation、latent reasoning、diffusion policy 等多个 thread 整合在一起，是 reasoning VLA 走向 real-time 的重要一步。它的核心 limitation 在于 verbalizer 仍是 LM-based，可能产生 hallucination；以及训练 pipeline 复杂、需要先训 teacher 再训 student。未来若能把 verbalizer 替换为 grounding-aware decoder、或把 teacher-student 合并为 single-stage online distillation，会进一步简化 pipeline。

**Web References**:
- Fast-ThinkAct (本 paper): https://arxiv.org/abs/2507.16815
- ThinkAct: https://arxiv.org/abs/2507.16815
- Coconut: https://arxiv.org/abs/2412.06769
- CODI: https://arxiv.org/abs/2502.21074
- Soft Thinking: https://arxiv.org/abs/2505.15778
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290
- OpenVLA: https://arxiv.org/abs/2406.09246
- π₀: https://arxiv.org/abs/2410.24164
- RDT-1B: https://arxiv.org/abs/2410.07864
- Magma: https://arxiv.org/abs/2502.13130
- MolmoAct: https://arxiv.org/abs/2508.07917
- CoT-VLA: https://arxiv.org/abs/2503.22020
- TraceVLA: https://arxiv.org/abs/2412.10345
- Embodied CoT: https://arxiv.org/abs/2407.08693
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DreamerV3: https://arxiv.org/abs/2301.04104
