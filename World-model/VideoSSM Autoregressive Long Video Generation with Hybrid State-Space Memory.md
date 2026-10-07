---
source_pdf: VideoSSM Autoregressive Long Video Generation with Hybrid State-Space
  Memory.pdf
paper_sha256: 768a44a30b6b1f0b89a183eacb4464952c33ff7272521b8bdcb1274e7a1500c2
processed_at: '2026-08-13T00:59:37-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 VideoSSM

## 一句话说清楚这篇 paper 在干啥

**让 AI 生成一分钟长的视频，不要"飘"也不要"卡"**

飘了 = 主体脸变样了、场景结构崩了（drift）
卡了 = 场景静止不动，像定格动画，或者内容来回重复（frozen / repetitive）

现有方法要么解决飘但导致卡，要么解决卡但导致飘。VideoSSM 的 trick 是：**给模型装两个记忆系统**，一个记最近几秒的细节，一个压缩所有历史信息，两者配合同时搞定飘和卡。

---

## 为什么生成长视频这么难

先 build 一下 intuition。

想象你闭着眼睛画画。画第一帧没问题。画第二帧的时候，你得回忆第一帧画了啥，才能保证连续性。画第 1000 帧的时候，你不可能把前面 999 帧全部在脑子里过一遍 — 那信息量太大了。

现有方法有几种笨办法：

**笨办法 1：全记住**。每一帧都看所有历史帧。这叫 full attention，复杂度 $O(T^2)$，T 是视频长度。视频一长，显存爆了，算力也扛不住。

**笨办法 2：只看最近几帧**。sliding window，只 attend 最近 L 帧。问题在于 — 第 1000 帧的时候，第 1 帧已经从窗口里滚出去了。人物穿的红衣服颜色、脸上的痣，这些 long-range 信息全丢了。所以第 1000 帧里的人物可能突然换衣服，脸也变样 — 这就是 drift。

**笨办法 3：固定记住开头几帧**。attention sink，把第一帧钉死，永远 attend 它。LongLive [57] 这么干。结果是 — 模型过度依赖开头那几帧，场景几乎不动，或者内容反复重复开头出现的东西。这就是 frozen / repetitive。你看 Table 2 里 LongLive 的 Dynamic Degree 只有 37.50，几乎没动态。

---

## VideoSSM 的核心 idea

类比人类大脑。人怎么记事？

- **工作记忆**（working memory）：记住最近几秒发生的事，细节清晰，但容量小
- **长期记忆**（long-term memory）：把过去几个月/几年的事压缩成抽象要点，容量大但细节模糊

VideoSSM 给模型也装了这两套：

- **Local memory** = sliding window attention，记最近 L 帧，full fidelity，负责 motion continuity 和 texture detail
- **Global memory** = SSM state，把所有滚出 window 的历史帧压缩成一个 fixed-size matrix，持续 update，负责 subject identity 和 scene structure

关键在于 global memory 不是静态的。LongLive 把开头几帧钉死当 sink，那是"死记忆"。VideoSSM 的 SSM state 是"活记忆" — 每一帧被 evict 出 window 的时候，它的信息被增量地写入 state，state 随时间 evolving。

---

## SSM state 到底怎么工作的

这是 paper 最核心的技术点，我用最直白的话讲。

SSM state $M$ 是一个 $d \times d$ 的矩阵。可以把它想成一个"知识库" — 你用 key 去查，它返回 value。

**写入新信息的时候**，paper 用了一个叫 Gated Δ-rule 的东西（来自 Gated Delta Networks [56]）。公式 (7) 长这样：

$$
V_{new} = V - \text{Predict}(M_{old}, K, \beta)
$$

$$
M_{new} = \exp(\bar{g}) \cdot M_{old} + K \cdot V_{new}^T
$$

什么意思呢？

第一行：拿到要写入的 value $V$，先用当前 state $M_{old}$ 和 key $K$ 去预测一个 "期望的 value"，把可预测部分减掉，剩下 $V_{new}$ 是"真正新颖、无法预测的部分"。

第二行：state 更新 = 旧 state 衰减一点 + 写入新颖信息。

**直觉**：写日记不写"今天太阳从东边升起"（可预测，已知道），只写"今天发生的新事"（novelty）。这样 state 不会被冗余信息塞满，也不会 collapse。

类比 Hopfield network 的 delta learning rule — 只学 error，不学已有的。这种 online incremental learning 在长序列上特别重要，因为如果每帧都全量写入，state 早就饱和了。

**衰减 gate** $\alpha_t$ 控制 forget rate：

$$
\alpha_t = -\exp(A) \cdot \text{SoftPlus}(W_\alpha H_t^{in} + B)
$$

负号保证 $\exp(\alpha_t) < 1$，每一步都让旧记忆衰减一点。但这个衰减是 cumulative 的（$\bar{g}_t = \sum \alpha_s$），不会一刀切遗忘。

**注入 gate** $\beta_t$ 控制 input rate：

$$
\beta_t = \sigma(W_\beta H_t^{in})
$$

sigmoid 输出 $[0,1]$，决定这帧信息要不要写入、写多少。

---

## 两个 memory 怎么配合

paper 用了一个 position-aware router 公式 (9):

$$
\gamma_t = \sigma(w_{router} \log(\rho_t) + b_{router})
$$

其中 $\rho_t = (t+1)/T$ 是当前位置占比。

**当 t 很小**（视频刚开头）：$\rho_t \to 0$，$\log(\rho_t) \to -\infty$，$\gamma_t \to 0$，global memory 几乎不参与。因为这时候还没什么历史，硬用 global state 会引入噪声。

**当 t 接近 T**（视频末尾）：$\rho_t \to 1$，$\log(\rho_t) \to 0$，$\gamma_t \to \sigma(b_{router})$，global memory 启用。因为此时 evicted tokens 很多，压缩 state 里有大量有用历史。

最终 fusion 公式 (10):

$$
H_t^{fused} = H_t^{local} + \gamma_t \cdot H_t^{global}
$$

local 始终主导，global 渐进加入。这个 log-encoding 的设计很巧妙 — 让早期抑制强、后期释放平稳。

---

## 训练怎么搞

VideoSSM 用 distillation，从一个 bidirectional teacher（Wan 2.1 [45]）学过来。两阶段：

**Stage 1：Causal distillation**

Teacher 是 bidirectional 的，能生成高质量 5s 短 clip。Student 是 causal AR 模型，从 teacher 的 ODE trajectory 学。在 5s 短 segment 上训练，让 student 先学会短时间尺度的高质量生成。

关键点：即使只在 5s 上训练，hybrid memory 已经在学了 — SSM state 每个 timestep 都 update，模型学会"怎么 compress"、"怎么 retrieve"。

**Stage 2：Long video training**

这一步解决 long-horizon 退化。用 DMD loss [59]。

做法：
1. Student 自己 rollout 60s 长视频（完全 self-generated，模拟 inference）
2. 随机抽 5s 窗口
3. 让 teacher 纠正这个窗口的质量

公式 (11):

$$
\mathcal{L}_{DMD} = \mathbb{E}_{i \sim \text{Unif}(1, N-K)} [\nabla_\theta KL(p_\theta^S(z_i) \| p^T(z_i))]
$$

直觉：teacher 只能管 5s，但能在 student 长视频的任意位置插进来纠错。这样 student 学会即使在 long rollout 中也能保持 5s 内的质量。

这就是 Self-Forcing [25] 思想的升级版 — 从 short rollout 扩展到 long rollout + windowed correction。

---

## 实验结果说了什么

**Short video (5s, VBench Table 1)**:

VideoSSM 在 AR 模型里 Total score 最高（83.95），比 LongLive（83.52）、Self-Forcing（83.00）都强。Params 只多 0.1B（SSM state 和 gates）。证明 hybrid memory 不光解决长视频，短视频也变好了。

**Long video (60s, Table 2)** — 这才是重点：

| Metric | Self Forcing | LongLive | VideoSSM |
|---|---|---|---|
| Subject Consistency ↑ | 88.25 | 91.09 | **92.51** |
| Background Consistency ↑ | 91.73 | 93.23 | **93.95** |
| **Dynamic Degree ↑** | 35.00 | 37.50 | **50.50** |

这里才是 paper 的杀手锏。LongLive consistency 高（91.09），但 Dynamic Degree 只有 37.50 — 场景几乎不动的"假稳定"。VideoSSM 同时拿到最高 consistency AND 最高 dynamic degree（50.50），这才是真稳定。

Dynamic Degree 从 37.50 → 50.50，提升 13 个点，同时 consistency 还更高。这证明了 SSM evolving memory 既不 frozen 也不 drift。

**User study (Table 3)**:

40 个人投票，VideoSSM 拿到 41.07% Rank 1，avg rank 1.85 最好。人眼也觉得 VideoSSM 更好。

---

## 为什么 SSM 比 attention sink 好

Attention sink 把开头几帧钉死，相当于"死记忆"。模型反复 attend 同样的 tokens，会强化那些 tokens 的特征，导致生成内容循环重复。

SSM state 是 evolving 的 — 每一帧 evict 出去时都增量 update state。state 里的"知识"随时间演化，反映最新场景状态，不会卡在开头。

更妙的是 Δ-rule 只写 novelty — 不重复写已知信息，state 不会被冗余塞满。这就像人的长期记忆 — 你不会每天重新记一遍"我叫什么"，但会记住"今天认识了新朋友"。

---

## 这篇 paper 的真正 contribution

从我的视角看：

1. **把 SSM 用对了地方**。之前 Mamba 系列想用 SSM 替代 attention 做 language modeling，跟 Transformer 死磕。VideoSSM 把 SSM 定位成"互补的记忆模块" — attention 做 local precise retrieval，SSM 做 global compressed memory。这更符合 SSM 的 strengths（recurrent compression）和 attention 的 strengths（precise retrieval）。

2. **Hybrid memory 是正确方向**。大脑就是这么干的 — hippocampus 做 pattern completion（attention-like），neocortex 做 compressed representation（SSM-like）。Long-range memory 不该是"记住所有细节"，而是"维护 evolving compressed state，需要时 retrieve"。

3. **Dynamic + Consistent 同时拿到**。这打破了之前 consistency 和 dynamic 的 trade-off。LongLive 牺牲 dynamic 换 consistency，Self-Forcing 牺牲 consistency 换 dynamic。VideoSSM 两个都好。

4. **AR streaming 优势保留**。因为仍然是 causal AR，可以实时 streaming 生成，可以 interactive prompt switching。这是 bidirectional 模型做不到的。

---

## 还有什么没解决

Paper 在 conclusion 提了 multi-modal conditioning、camera priors、editing。我补充几点：

**SSM state capacity 上限**。$M \in \mathbb{R}^{d \times d}$ 是 fixed-size，对小时级视频可能 saturate。可能需要 hierarchical memory pool。

**Predict 函数的具体形式**。Paper 里公式 (7) 写了 $\text{Predict}(M_{t-1}, K_t, \beta_t)$ 但没详述具体是线性还是非线性的。这影响 novelty 提取精度。

**超长 generalization**。训练 N=60s，测试 10min 会怎样？没给 ablation。

**SSM update 的 sequential bottleneck**。虽然 attention 是 $O(TL)$，但 SSM update 在时间维度上还是 sequential，长视频难 parallelize。

---

## 一句话总结

VideoSSM 给 AR video generation 装了个"人类式双记忆系统" — sliding window 当工作记忆记细节，SSM 当长期记忆压缩历史，用 Δ-rule 只写新颖信息避免冗余，用 position-aware router 动态平衡两者。结果是一分钟视频既不飘也不卡，还支持实时 streaming 和 prompt 切换。

这条路我觉得很有前途。Hybrid memory with proper gating 是长视频生成的 natural evolution，比死磕 pure attention 或 pure SSM 都合理。

---

# VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory 详解

## 1. Paper 核心问题与 Motivation

这篇 paper 来自 HKU + ByteDance PICO + SUSTech 团队，arXiv 编号约 2510 前后，解决的核心问题是 **minute-scale autoregressive (AR) video generation** 中的三大顽疾：

- **Error accumulation**: AR 模型在 self-rollout 过程中, 前面帧的预测误差会累积到后续帧
- **Motion drift**: 随着序列变长, 主体身份、场景结构会漂移
- **Content repetition**: 长序列中模型容易陷入循环式重复生成

### 1.1 现有 AR DiT 方案的局限

让我对照 Figure 3 的四种 attention 机制来 build intuition:

**(a) Causal Attention**: 每个 query attend 所有 past tokens, 复杂度 $O(T^2)$, T 为 video token 长度。完整 context 但无法 scale 到长视频。

**(b) Window Attention (sliding window)**: 仅 attend 最近 $L$ 个 tokens, 复杂度 $O(TL)$。问题在于 early tokens 被驱逐后, 信息丢失导致 drift。这是 Self-Forcing [25] 和 next-frame prediction 类方法 [17] 的主要做法。

**(c) Attention Sink**: 在 window 外保留一组 fixed "sink" tokens (通常是 first frames), 复杂度 $O(TL)$。LongLive [57] 和 Rolling Forcing [31] 用这个 trick, 但会导致 **frozen global memory** — 因为 sink tokens 是静态的, 模型反复 attend 它们会过度稳定化, 进而抑制 motion dynamics, 产生 repetitive 生成。

**(d) VideoSSM Hybrid Memory**: 用 SSM 把 evicted tokens 压缩成 evolving global state, 配合 local window, 维持 $O(TL)$ 复杂度同时提供动态 global context。

### 1.2 与 3D-coupled Memory 的区别

World model 领域 (VMem [30], WorldMem [53], Context-as-Memory [61]) 用 3D 几何结构 (surfel, camera pose) 来做 long-term memory。这类设计适合 **view-revisit** 场景 (比如 game scene 回访), 但 transfer 到 free-view、open-ended long video generation 时表现差, 因为后者没有 explicit 3D world state。VideoSSM 的关键 insight 是: 在 **latent space** 中维护一个 continuously updated global memory, 而不依赖 explicit 3D 假设。

---

## 2. Hybrid Memory Architecture 深度解析

### 2.1 总体设计哲学

类比人类记忆系统 (Atkinson-Shiffrin model [1] + Baddeley working memory [2]):

- **Local memory** ≈ working memory: 高分辨率、lossless、短时, 用于 fine motion/detail
- **Global memory** ≈ long-term memory: 压缩、抽象、evolving, 用于 scene-level consistency

### 2.2 Local Memory: Sliding Window Self-Attention

公式 (3):
$$
\{Q_t, K_t, V_t\} = \{H_t^{in} W_Q, H_t^{in} W_K, H_t^{in} W_V\}
$$

变量解释:
- $H_t^{in} \in \mathbb{R}^{d}$: 当前 frame $t$ 的 input hidden state, $d$ 为 token dimensionality
- $W_Q, W_K, W_V$: 标准 attention 的 query/key/value projection matrices
- $Q_t, K_t, V_t$: 当前 frame 的 QKV

公式 (4):
$$
H_t^{local} = \text{SelfAttention}(Q_t, K_t^{local}, V_t^{local})
$$

其中:
$$
K_t^{local} = [K_{sink}, K_{t-L+1}:K_t], \quad V_t^{local} = [V_{sink}, V_{t-L+1}:V_t]
$$

这里 $L$ 是 sliding window size, $K_{sink}, V_{sink}$ 是 sink tokens 的 KV (VideoSSM 实际也保留少量 sink 用于稳定 attention 计算, 但不依赖它们做 long-range memory)。

**关键直觉**: Local memory 保留 recent tokens 的完整 K/V, 提供 precise motion cue 和 high-fidelity appearance。

### 2.3 Global Memory: SSM-based Dynamic State

这是 paper 的核心技术贡献。Global memory 把所有 evicted tokens (即超出 window 范围的历史 tokens) 压缩成 fixed-size state $M_t$, 并通过 recurrent update 持续演化。

#### 2.3.1 Synchronized Gate Caching

每个 token 在离开 local window 之前, 先计算两个 gate:

公式 (5) — **Injection gate** $\beta_t$:
$$
\beta_t = \sigma(W_\beta H_t^{in})
$$
- $\sigma$: sigmoid, 输出 $[0,1]$, 控制新 token 对 global state 的注入强度
- $W_\beta \in \mathbb{R}^{d \times d}$: learnable projection

公式 (6) — **Decay gate** $\alpha_t$:
$$
\alpha_t = -\exp(A) \cdot \text{SoftPlus}(W_\alpha H_t^{in} + B)
$$
- $A \in \mathbb{R}^d$: learnable, 决定 decay rate 的基础值
- $\text{SoftPlus}(x) = \log(1 + e^x)$: smooth ReLU, 保证非负
- $-\exp(A)$: 强制 $\alpha_t < 0$, 这样在后续 state update 中, $\exp(\alpha_t) < 1$, 实现遗忘
- 输出 $\alpha_t \in \mathbb{R}^d$, 维度匹配 token

**直觉**: 这两个 gate 借鉴了 Mamba 系列 [56] 的设计, $\alpha_t$ 控制 "forget rate", $\beta_t$ 控制 "input rate"。但 VideoSSM 用 **synchronized caching** — 在 token 还在 window 内时就预计算 gate, 与 KV cache 同步更新, 避免 evict 时重新计算。

#### 2.3.2 Global Memory State Update — Gated Δ-rule

这是最核心的部分, 基于 Gated Delta Networks [56]。公式 (7):

$$
V_{new,t}^{evt} = V_t^{evt} - \text{Predict}(M_{t-1}, K_t^{evt}, \beta_t^{evt})
$$
$$
M_t = \exp(\bar{g}_t) \cdot M_{t-1} + K_t^{evt} \cdot (V_{new,t}^{evt})^T
$$

变量解释:
- $V_t^{evt}, K_t^{evt}, \alpha_t^{evt}, \beta_t^{evt}$: evicted tokens 的聚合 (用 $\text{avg}[\cdot]$ 表示对 sink+1 到 t-L 范围内 evicted tokens 取平均)
- $\text{Predict}(M_{t-1}, K_t^{evt}, \beta_t^{evt})$: 用前一个 state $M_{t-1}$ 和当前 key $K_t^{evt}$, injection gate $\beta_t^{evt}$ 来预测 evicted value 的可预测部分
- $V_{new,t}^{evt}$: **residual / novel component** — 减掉可预测部分后剩下的不可预测信息
- $\bar{g}_t = \sum_{s=0}^{t} \alpha_s^{evt}$: cumulative negative gate, 因为 $\alpha_s^{evt} < 0$, 所以 $\bar{g}_t$ 越来越负
- $\exp(\bar{g}_t)$: decay factor, 随时间衰减
- $M_0 = 0$: 初始 state 为零

**Δ-rule 的直觉** (重要!):

传统 RNN/SSM 的 update rule 类似 $M_t = \lambda M_{t-1} + K_t V_t^T$, 直接把新信息叠加。Delta rule 的核心思想是 **"只学增量"** — 假设新输入 $V_t^{evt}$ 中有一部分是可以从过去 state $M_{t-1}$ 通过 key $K_t^{evt}$ 检索预测出来的, 那只把 **无法预测的部分** $V_{new}^{evt}$ 写入 state。这等价于一个 **online delta learning** 过程, 类似 delta-WC rule 在 Hopfield network 中的作用。

这样做的 benefits:
1. **避免 redundant 信息反复写入**, 减缓 state saturation
2. **更 stable 的 long-range memory**: 不会因为重复信息而 collapse
3. **保留 novelty**: 真正变化的内容才会被 record

类比: 这就像你写日记, 不需要把 "今天太阳还是从东边升起" 记下来, 只记 "今天发生的新事"。

#### 2.3.3 Global Memory Retrieval

公式 (8):
$$
g_t^{out} = \text{Linear}(H_t^{in})
$$
$$
H_t^{global} = \text{Swish}(g_t^{out} \odot \text{RMSNorm}(Q_t M_t))
$$

变量:
- $g_t^{out} \in \mathbb{R}^d$: output gate, 从当前 hidden state 计算
- $Q_t M_t$: query $Q_t$ 与 memory state $M_t$ 做矩阵乘法, 得到 query-aligned memory response (类似 attention 中 $QK^T$, 但这里 state 是 compressed 的, 直接做 retrieve)
- $\text{RMSNorm}(\cdot)$: Root Mean Square Normalization, 稳定 magnitude
- $\text{Swish}(x) = x \cdot \sigma(x)$: 平滑激活, 允许 negative 抑制
- $\odot$: element-wise multiplication

**直觉**: 检索过程是 "用当前 query 去 probe global state M, 拿回相关历史信息, 再用 output gate 调制流量"。RMSNorm 防止 $M_t$ magnitude 随时间爆炸, Swish 提供平滑 gating。

### 2.4 Position-Aware Gated Fusion

公式 (9):
$$
\gamma_t = \sigma(w_{router} \log(\rho_t) + b_{router})
$$

变量:
- $\rho_t = (t+1)/T \in (0, 1]$: relative position ratio, $t$ 是当前 frame index, $T$ 是 total context length
- $w_{router}, b_{router} \in \mathbb{R}^d$: learnable router parameters
- $\gamma_t \in \mathbb{R}^d$: memory gate, 控制全局记忆的注入强度

**关键性质**:
- 当 $t \to 0$: $\rho_t \to 0$, $\log(\rho_t) \to -\infty$, $\gamma_t \to 0$ → global memory 被抑制, 模型主要用 local
- 当 $t \to T$: $\rho_t \to 1$, $\log(\rho_t) \to 0$, $\gamma_t \to \sigma(b_{router})$ → global memory 启用

公式 (10) — Fusion:
$$
H_t^{fused} = H_t^{local} + \gamma_t \cdot H_t^{global}
$$

**为什么这样设计**: 在视频开始时 (t 小), 历史信息很少, global memory 没什么可压缩的, 此时强行用 global memory 会引入噪声。随着序列增长, evicted tokens 越来越多, global memory 才有意义, 此时增大 $\gamma_t$ 让模型 attend global。这是一个 **curriculum-style 的 memory scheduling**, 用 log-encoding 让早期抑制更强, 后期渐近释放。

---

## 3. Training Methodology

### 3.1 两阶段 Distillation 框架

VideoSSM 从一个 pre-trained bidirectional teacher (Wan 2.1-T2V-1.3B [45]) 出发, 通过 CausVid [60] 策略进行 distillation。

**Stage 1: Causal Model Distillation**

让 causal student $G_\theta$ 拟合 teacher $T_\phi$ 的 ODE sampling trajectory:
$$
\mathcal{L} = \|\hat{x}_0 - T_\phi(x_t, t)\|^2, \quad \hat{x}_0 = G_\theta(x_t, t)
$$

- Teacher 用 bidirectional attention 生成 5s 短 clip
- Student 用 causal attention 拟合同样的 trajectory
- 在 5s segment 上训练, 让 student 获得高质量 short-term dynamics
- Gradients 通过 hybrid memory backprop, exposure bias 通过 self-generated history 缓解

**关键 insight**: 即使只在 5s 短 clip 上训练, hybrid memory 已经能让模型获得 long-range capability, 因为 SSM state 在每个 timestep 都更新, 模型学会如何正确 compress 和 retrieve。

**Stage 2: Long Video Training (Self-Rollout + DMD)**

这一步是处理 long-horizon degradation 的关键, 用 Distribution Matching Distillation (DMD) [59] loss。

子步骤 1 — **Long Self-Rollout**:
- Student $G_\theta$ autoregressively 生成 $N = 60$s 长序列 (chunk size 3 frames, 4-step diffusion per chunk)
- 填充 local KV cache 和 global memory cache (通过 $\beta, \alpha$ gates)
- 完全用 self-generated output, 模拟真实 inference 场景

子步骤 2 — **Windowed Teacher Correction**:
公式 (11):
$$
\mathcal{L}_{DMD} = \mathbb{E}_{t, i \sim \text{Unif}(1, N-K)} [\nabla_\theta KL(p_{\theta,t}^S(z_i) \| p_t^T(z_i))]
$$

变量:
- $K = 5$s: 短窗口长度
- $i \sim \text{Unif}(1, N-K)$: 在长序列中均匀随机采样窗口起点
- $z_i$: 从 frame $i$ 开始的 5s 窗口
- $p_{\theta,t}^S(z_i)$: student 在该窗口的 noise prediction distribution
- $p_t^T(z_i)$: teacher 在该窗口的 noise prediction distribution
- $\nabla_\theta KL(\cdot)$: KL divergence 的 gradient, 反向传播到 student

**直觉**: Teacher 虽然只能生成 5s 短 clip, 但它的短 clip 质量很高。我们让 student 自己 rollout 长序列, 然后随机抽 5s 窗口, 用 teacher 来纠正这个窗口的质量。这样 student 在 long-horizon 上的误差能被持续校正, 而不需要 long ground truth。这是 Self-Forcing [25] 思想的 extended 版本, 把它从 short rollout 扩展到 long rollout + windowed correction。

---

## 4. Experiments 深度分析

### 4.1 Short Video (VBench, 5s) — Table 1

| Model | #Params | Total | Quality | Semantic |
|---|---|---|---|---|
| Wan2.1 (bidirectional teacher) | 1.3B | 84.26 | 85.30 | 80.09 |
| CausVid | 1.3B | 81.20 | 84.05 | 69.80 |
| Self Forcing | 1.3B | 83.00 | 83.71 | 80.14 |
| LongLive | 1.3B | 83.52 | 84.26 | 80.53 |
| Self Forcing++ | 1.3B | 83.11 | 83.79 | 80.37 |
| Rolling Forcing | 1.3B | 81.22 | 84.08 | 69.78 |
| **VideoSSM** | 1.4B | **83.95** | **84.88** | 80.22 |

观察:
- VideoSSM 在 AR 模型中 Total/Quality 最高
- 比 teacher Wan2.1 还略高 (84.88 vs 85.30 略低, 但 Total 83.95 vs 84.26 接近)
- 比 4.5B 的 MAGI-1 (79.18) 强很多, 体现 memory 机制的有效性
- params 只增加 0.1B (1.3B → 1.4B), 主要是 SSM state 和 gate parameters

### 4.2 Long Video (60s) — Table 2

| Metric | Self Forcing | LongLive | **VideoSSM** |
|---|---|---|---|
| Subject Consistency ↑ | 88.25 | 91.09 | **92.51** |
| Background Consistency ↑ | 91.73 | 93.23 | **93.95** |
| Dynamic Degree ↑ | 35.00 | 37.50 | **50.50** |
| Aesthetic Quality ↑ | 60.02 | 55.74 | **60.45** |

**关键解读**:
1. **Consistency vs Dynamic 的 trade-off**: LongLive 用 attention sink 取得了 high consistency (91.09), 但 Dynamic Degree 只有 37.50 — 场景几乎静止。VideoSSM 同时拿到 **最高 consistency AND 最高 dynamic degree (50.50)**, 这是 paper 最核心的贡献。证明 hybrid memory 既不 "frozen" 也不 "drift"。

2. **Dynamic Degree = 50.50** 比 LongLive 高 13 points, 说明 SSM global memory 让模型在保持主体一致性的同时, 还能持续演化场景。这正是 attention sink 的痛点。

3. Figure 6 的 qualitative example:
   - Burger: SkyReels-V2 完全 collapse, Self Forcing 严重 drift, VideoSSM 60s 内保持 subject identity
   - Underwater boy: CausVid 几乎静止, LongLive hallucinate 第二个 boy, VideoSSM 保持 forward-swimming motion + high consistency

### 4.3 User Study (Table 3)

40 participants, 8 prompts, 4 methods, 1-minute videos:

| Model | Rank 1 (%) | Avg Rank |
|---|---|---|
| Self Forcing | 11.79 | 3.18 |
| CausVid | 7.50 | 3.03 |
| LongLive | 39.64 | 1.92 |
| **VideoSSM** | **41.07** | **1.85** |

VideoSSM 在 Rank 1 票数和 avg rank 上都最优, 验证 Dynamic + Consistent 的组合比纯 consistency (LongLive) 更受用户偏好。

### 4.4 Interactive Prompt Switching

通过 KV recache [57] 机制, 当用户提供新 prompt 时, 刷新 local memory。Figure 7 展示了 smooth transition across prompt changes。这是 AR 模型的天然优势 — 因为 generation 是 streaming 的, 可以在任意时刻切换 prompt。

---

## 5. Intuition Building: 为什么 SSM 适合 Long Video Memory?

让我从 first principles 来构建直觉:

### 5.1 Attention 的本质问题

Attention 是 **associative memory** — 把 (key, value) pair 存起来, 用 query 检索。但它的 storage 是 **explicit** 的: 每对 (K, V) 都占独立空间, 复杂度 $O(N)$ in both time and space。对于长视频, 这不可持续。

### 5.2 SSM 作为 Compressed Memory

SSM (尤其是 Mamba/GDN 系列) 把 memory 压缩成 **fixed-size state matrix** $M \in \mathbb{R}^{d \times d}$。存储 cost 是 $O(d^2)$, 与序列长度无关。retrieve 是 $O(d^2)$ per query。

但 vanilla SSM 有两个问题:
1. 信息会被 **decay 覆盖**, long-range info 容易丢失
2. 没有选择性, 所有信息同等对待

### 5.3 Δ-rule 的精妙之处

VideoSSM 用的 Gated Δ-rule 解决了上述问题:

$$
M_t = \underbrace{\exp(\bar{g}_t) M_{t-1}}_{\text{controlled forgetting}} + \underbrace{K_t (V_t - \text{Predict}(M_{t-1}, K_t, \beta_t))^T}_{\text{novelty-aware writing}}
$$

- 第一项: decay, 但 $\bar{g}_t$ 是累积的, 控制整体遗忘速率
- 第二项: **只写 novelty** — Predict 部分用 $M_{t-1}$ 检索出 $K_t$ 对应的 "expected value", 减掉实际 $V_t$, 得到 residual

这等价于 **online gradient descent on retrieval error**:
$$
M_t = M_{t-1} - K_t^T \cdot \text{error}
$$
其中 $\text{error} = \text{Predict}(M_{t-1}, K_t, \beta_t) - V_t$

类比: 这跟 Hopfield network 的 **delta learning rule** 同源, 也跟 modern Hopfield network (Ramsauer et al. 2020) 的 retrieval dynamics 类似。本质上是把 attention 升级为 **learned associative memory with online update**。

### 5.4 为什么 Hybrid 比 Pure SSM 好

如果只用 SSM, 模型会失去 fine-grained detail (因为 state 是 compressed 的)。Local window 保留 recent tokens 的完整 KV, 提供 high-fidelity motion cue。Hybrid 设计让两者互补:

- Local: 短期 high-fidelity (motion continuity, texture detail)
- Global: 长期 abstract consistency (subject identity, scene structure)

Position-aware router $\gamma_t$ 用 log-scale position 来动态平衡, 让模型学会 "早期靠 local, 后期加 global"。

---

## 6. 与 Related Work 的对比

### 6.1 vs Self-Forcing [25]
Self-Forcing 用 short AR rollout 来 align train-test gap, 但 memory 仅靠 sliding KV cache。无 global memory, 长序列 drift。VideoSSM 在此基础上加 SSM global memory 并用 DMD loss 做 long-horizon training。

### 6.2 vs LongLive [57]
LongLive 用 first frames 作为 attention sink, 取得 high consistency 但 frozen dynamics (Dynamic Degree 37.50)。VideoSSM 用 evolving SSM state 替代 fixed sink, Dynamic Degree 提升到 50.50, consistency 还更高。

### 6.3 vs Rolling Forcing [31]
Rolling Forcing 通过 rolling context + cache reuse, 在 long-range 上比 pure window attention 好, 但仍是 local memory 范畴。VideoSSM 显式建模 global state。

### 6.4 vs SSM-Video Diffusion [35]
Oshima et al. 把 SSM 直接接入 video diffusion backbone 做 long-term modeling。VideoSSM 的区别在于 hybrid design (local + global) + Δ-rule based update + position-aware fusion, 而且 trained via distillation from bidirectional teacher。

### 6.5 vs Long-context SSM World Models [37]
Po et al. 用 SSM 做 world model。VideoSSM 借鉴了 SSM 作为 memory 的 idea, 但 focus on open-ended free-view video generation 而非 world modeling。

---

## 7. 局限性与 Future Work

Paper 在 Section 6 提到:
1. **Multi-modal conditioning**: 目前只 support text prompt, 未来可加 audio, action 等
2. **Camera-aware priors**: 没用 explicit camera geometry, 对极端 camera motion 可能 still drift
3. **Controllable long-form editing**: 当前是 generation, editing 是 future direction

从我 (Karpathy) 的视角补充几点:

1. **SSM state 的 capacity 上限**: $M \in \mathbb{R}^{d \times d}$ 是 fixed-size, 对极端长视频 (小时级), 仍可能 saturate。可能需要 hierarchical memory pool 或 dynamic state allocation。

2. **Δ-rule 的 Predict 函数**: paper 中没详细说明 Predict 的具体形式 (是否 linear in $M_{t-1}$?), 这影响 novelty 提取的精度。

3. **Long-horizon generalization**: 训练时 N=60s, 测试更长 (e.g. 10min) 时性能如何? Paper 没给 ablation。

4. **Distillation 的 teacher 依赖**: 必须 good bidirectional teacher。如果 teacher 本身 bias, student 也会继承。

5. **Computational cost**: 虽然 attention 是 $O(TL)$, 但 SSM update 在长序列上仍是 sequential, 难以 parallelize across time。对 real-time generation 是潜在 bottleneck。

---

## 8. 总结: 这篇 paper 的核心贡献

1. **Conceptual**: 把 AR video generation 重新 framed 为 **recurrent dynamical process with hybrid memory**, 借鉴人类记忆系统 (working + long-term)。

2. **Technical**: 
   - Hybrid memory architecture (sliding window + SSM global state)
   - Gated Δ-rule for novelty-aware memory update
   - Position-aware gated fusion for dynamic local/global balance
   - Two-stage distillation (causal init + long self-rollout with DMD correction)

3. **Empirical**:
   - SOTA among AR models on VBench short + long video
   - Dynamic Degree 50.50 vs LongLive 37.50, 同时 consistency 更高
   - User study 41.07% Rank 1 preference

4. **Practical**:
   - 1.4B params, 4-step diffusion, chunk size 3, real-time streaming
   - Support interactive prompt switching via KV recache

---

## References (with links)

主要参考的 papers:

1. **Self-Forcing** [25] - https://arxiv.org/abs/2506.08009
2. **LongLive** [57] - https://arxiv.org/abs/2509.22622
3. **Rolling Forcing** [31] - https://arxiv.org/abs/2509.25161
4. **Self-Forcing++** [12] - https://arxiv.org/abs/2510.02283
5. **CausVid** [60] - https://arxiv.org/abs/2504.04128 (CVPR 2025)
6. **Gated Delta Networks** [56] - https://arxiv.org/abs/2412.06464
7. **Mamba2** (foundation of GDN) - https://arxiv.org/abs/2405.21060
8. **DMD (Distribution Matching Distillation)** [59] - https://arxiv.org/abs/2311.18828 (CVPR 2024)
9. **Wan 2.1** [45] - https://arxiv.org/abs/2503.20314
10. **VBench** [26] - https://arxiv.org/abs/2311.10818 (CVPR 2024)
11. **VMem** [30] - https://arxiv.org/abs/2506.18903
12. **WorldMem** [53] - https://arxiv.org/abs/2504.12369
13. **Context-as-Memory** [61] - https://arxiv.org/abs/2506.03141
14. **Long-context SSM World Models** [37] - https://arxiv.org/abs/2505.20171
15. **Attention Sinks (StreamingLLM)** [52] - https://arxiv.org/abs/2309.17453
16. **Mixture of Contexts** [5] - https://arxiv.org/abs/2508.21058
17. **Pack-N-Go** [63] - https://arxiv.org/abs/2504.12626
18. **Long-context AR Video Modeling** [17] - https://arxiv.org/abs/2503.19325
19. **MAGI-1** [42] - https://arxiv.org/abs/2505.13211
20. **SkyReels-V2** [7] - https://arxiv.org/abs/2504.13074

---

## 9. 个人思考 (Karpathy 视角)

这篇 paper 做了一件非常 elegant 的事: **把 SSM 从 "替代 attention" 的角色, 转换为 "complementary memory module" 的角色**。过去 Mamba 系列的工作大多想用 SSM 完全替代 attention, 在 language modeling 上跟 Transformer 激烈竞争。VideoSSM 反其道而行 — attention 仍然做 local high-fidelity processing, SSM 做 long-range compressed memory。这其实更符合 SSM 的 strengths (recurrent compression) 和 attention 的 strengths (precise retrieval)。

这跟大脑的工作方式很像: hippocampus 做 pattern completion / retrieval (attention-like), neocortex 做 compressed representation update (SSM-like)。Long-range memory 不是 "记住所有细节", 而是 "维护一个 evolving compressed state, 能在需要时 retrieve 相关信息"。VideoSSM 的 Δ-rule 实现了这一点 — 只 novelty-aware 写入, controlled forgetting, query-aligned retrieval。

从 scaling 角度看, 这条路很有前途。如果 video context 要 scale 到 1 小时, 1 天, pure attention 完全不行, pure SSM 也容易 collapse。Hybrid memory with proper gating 是 natural 的发展方向。

值得继续思考的方向:
- Memory state $M$ 是否可以 **multi-head**, 类似 multi-head attention, 让不同 head 记不同 type 信息 (identity, motion, scene, ...)
- Hierarchical memory: 短期 SSM + 中期 SSM + 长期 SSM, 不同时间尺度
- Memory-augmented retrieval: 当 query 难以满足时, 主动 retrieve 更多 history
- Cross-modal memory: video + audio + action 共享 memory state

这篇 paper 是 2025 年 long video generation 领域的重要工作, 把 memory mechanism 从启发式 (attention sink) 升级为 principled (SSM with Δ-rule), 同时保持 AR 的 streaming 和 interactive 优势。值得仔细研读。
