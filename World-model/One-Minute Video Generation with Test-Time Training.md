---
source_pdf: One-Minute Video Generation with Test-Time Training.pdf
paper_sha256: 9390623bdd4eef918beb9717151bbf8844f31dc5d5737aed1b62a9b835d50afc
processed_at: '2026-08-05T23:47:52-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 TTT for Video

参考链接：
- 项目网站 (含视频 sample): https://test-time-training.github.io/video-dit
- TTT 原始论文: https://arxiv.org/abs/2407.04620
- CogVideo-X: https://arxiv.org/abs/2408.06072
- Mamba 2: https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2411.14368
- ThunderKittens (kernel): https://arxiv.org/abs/2407.10358
- Titans (后续工作): https://arxiv.org/abs/2501.00663
- FlashAttention-3: https://arxiv.org/abs/2407.08608

---

## 一句话总结

**把 RNN 的 hidden state 从一个 matrix 升级成一个小 MLP 的 weights，每个新 token 来就对这个 MLP 做一步 SGD 更新，这样 hidden state 就能记住几十万 token 的复杂关系。把这个 layer 插进 CogVideo-X 5B，能从 3 秒短片扩展到 1 分钟多场景故事视频。**

---

## 1. 问题在哪儿

2025 年 3 月所有公开 API 的 video 最长就那么几秒：Sora 20s, MovieGen 16s, Ray 2 10s, Veo 2 8s。都做不了多场景复杂故事。

为什么？因为 video token 太多了。一分钟 video 用标准 tokenizer 编码出来超过 **300k tokens**。self-attention 是 O(n²)，生成一分钟 video 比生成 20 个 3 秒 video 慢 11 倍，训练慢 12 倍。没法 scale。

之前 LinGen 用 Mamba 这种 RNN layer 做 minute-length video，跑出来了，但是只有单场景、慢动作，讲不了复杂故事。

作者的 hypothesis 很直白：**Mamba / DeltaNet 这帮 linear attention 变种，hidden state 就是一个 matrix (几千 rank)。把几十万 vector 压进一个 matrix，记不住远距离 token 的深度关系。容量不够。**

---

## 2. TTT 的核心 insight (人话版)

观察一下：self-supervised learning 本来就能把海量训练集压进 NN weights。那把"历史 context"当作"unlabeled dataset"，把"hidden state"当作"NN weights"，update rule 就是"gradient step on self-supervised loss"。

每来一个 token，就对 hidden state (一个小 MLP 的 weights) 做一步 SGD。所以 hidden state 从"一个 matrix"升级成"一个 small brain 的 weights"。容量和表达力都不一样了。

### 公式拆开看

核心 update (Eq 1)：
$$W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)$$

- $t$ = 时间步
- $W_t$ = 第 $t$ 步更新后的 hidden state（注意这是 MLP 的 weights，不是 matrix）
- $W_{t-1}$ = 上一步的 weights
- $\eta$ = inner-loop learning rate
- $\ell$ = self-supervised loss
- $x_t$ = 第 $t$ 个 input token

直白说：每来一个 token，对 MLP weights 做一步 SGD。

Output rule (Eq 2)：
$$z_t = f(x_t; W_t)$$

- $z_t$ = 输出 token
- $f$ = 那个小 MLP
- $W_t$ = 已经被当前 token 更新过的 weights

注意顺序：先 update 再 forward。$z_t$ 反映的是"刚学过 $x_t$ 之后 $f$ 对 $x_t$ 的反应"。

### 自监督任务怎么设计

naive 版本 (Eq 3)：
$$\ell(W; x_t) = \|f(\tilde{x}_t; W) - x_t\|^2$$

- $\tilde{x}_t$ = corrupted input (denoising autoencoder 那种)
- $x_t$ = 原始 input 作为 reconstruction target
- $f(\tilde{x}_t; W)$ = MLP 在 corrupted input 下的输出

$f$ 必须发现 $x_t$ 各维度之间的 correlation 才能从 partial 信息重建自己——这一过程把 token 的结构信息"消化"进 $W$。

但是手工设计这个 corruption 太 brittle，所以 [43] 把它端到端 learn 出来，引入三个 learnable projection matrices $\theta_K, \theta_V, \theta_Q$ (类比 attention 的 K/V/Q)：

Loss (Eq 4)：
$$\ell(W; x_t) = \|f(\theta_K x_t; W) - \theta_V x_t\|^2$$

- $\theta_K$ = Key projection，把 $x_t$ 投影成低维 corrupted input (outer loop learnable)
- $\theta_V$ = Value projection，把 $x_t$ 投影成低维 reconstruction target (outer loop learnable)
- $W$ = 唯一在 inner loop 优化的参数

设计哲学：不是所有 $x_t$ 的信息都值得记忆，所以 reconstruction target 也压缩到 $\theta_V x_t$。$\theta_K, \theta_V$ 决定"学什么"，$W$ 决定"怎么学"。

Output (Eq 5)：
$$z_t = f(\theta_Q x_t; W_t)$$

- $\theta_Q$ = Query projection

为什么需要 $\theta_Q$？因为 $\theta_K x_t$ 维度比 $x_t$ 少，没法直接用 $f(x_t; W_t)$。$\theta_K, \theta_V, \theta_Q$ 在 outer loop 训练，类比 self-attention 的 Q/K/V parameters。

### Inner vs Outer loop

- **Outer loop**: 训练整个 Transformer (含 TTT layer 的 $\theta_K, \theta_V, \theta_Q$)，standard training
- **Inner loop**: 每个 TTT layer 在每个 test sequence 上自己训一遍 $W_1, ..., W_T$

backward $\nabla \ell$ 需要 gradient of gradient (二阶)。MAML 风格的 meta-learning。

### TTT-MLP instantiation (Eq 6)

$$f(x) = x + \text{LN}(f_{MLP}(x))$$

- $f_{MLP}$ = 两层 MLP，hidden dim = 4× input dim，GELU activation
- LN = Layer Norm
- $x + \text{LN}(\cdot)$ = residual connection

LN + residual 保证 TTT inner loop 的稳定性 (SGD 在深网络里容易发散)。

TTT-Linear 基线：$f$ 里是 linear model。用来消融看 non-linearity 在 hidden state 里的作用。

---

## 3. 怎么塞进 CogVideo-X 5B

原始 Transformer block:
$$X' = \text{self\_attn}(\text{LN}(X))$$
$$Y = X' + X$$

修改后 (Eq 8-12):
$$X' = \text{self\_attn}(\text{LN}(X))$$  (local attention)
$$Z = \text{gate}(\text{TTT}, X'; \alpha)$$  (forward TTT)
$$Z' = \text{gate}(\text{TTT}', Z; \beta)$$  (backward TTT)
$$Y = Z' + X$$

### Gating (Eq 6)

$$\text{gate}(\text{TTT}, X; \alpha) = \tanh(\alpha) \otimes \text{TTT}(X) + X$$

- $\alpha \in \mathbb{R}^d$ = learnable gating vector (per-channel)
- $\tanh(\alpha) \in (-1, 1)^d$ = element-wise 压到 (-1, 1)
- $\otimes$ = element-wise 乘
- $+ X$ = residual

初始化 $\alpha = 0.1$，所以 $\tanh(0.1) \approx 0.1$。开始 fine-tune 时 TTT 贡献很小，避免随机初始化的 TTT 破坏 pre-trained CogVideo-X 的预测。随着 fine-tune 推进，$\alpha$ 会被 learned 调整，让 TTT 慢慢 take over。

Flamingo 风格的 gated cross-attention trick，本质是 residual scaling 防止新模块冷启动退化。

### Bi-direction (Eq 7)

Diffusion model 是 non-causal 的，$z_t$ 可以依赖所有 $x_1, ..., x_T$ 包括未来。但 TTT 默认是 causal。

trick：跑两次方向相反：
$$\text{TTT}'(X) = \text{rev}(\text{TTT}(\text{rev}(X)))$$

- $\text{rev}(X) = (x_T, ..., x_1)$ = 时间反转算子
- 内层 $\text{TTT}(\text{rev}(X))$ = 在反转后的序列上跑 TTT
- 外层 $\text{rev}(\cdot)$ = 把输出反转回 chronological 顺序

$\text{TTT}$ 和 $\text{TTT}'$ 共享同一套 $\theta_K, \theta_V, \theta_Q$，只有 gating $\alpha, \beta$ 不同。

### Local attention + Global TTT

**self-attn 限制在 3 秒 segment 内 (local)，TTT 处理整段 60s (global)**。

- self-attn O(n²)，60s 全局爆炸
- TTT O(n) 线性，可以扛长 context
- 3 秒是 CogVideo-X 原生能生成的长度，local self-attn 保留它的短程建模能力

Hybrid 架构：local attention 处理 intra-segment，TTT 处理 inter-segment long-range。**分工明确**。

---

## 4. 数据和 pipeline

### 三种 prompt 格式

- **Format 1**: 5-8 句 plot 摘要 (用户友好)
- **Format 2**: ~20 句，每句对应 ~3 秒 segment
- **Format 3**: storyboard，每段 3-5 句，含背景颜色、镜头运动，用 `<scene start>` / `<scene end>` 严格标记 scene 边界

Fine-tune 和 inference 永远用 Format 3 喂 text tokenizer。1→2→3 的扩展用 Claude 3.7 Sonnet 自动完成。用户只要写 5-8 句话就能得到 1 分钟 video。

### Dataset

- Tom and Jerry 1940-1948 共 81 集，~5 分钟/集，总 ~7 小时
- Video super-resolution (Real-ESGAN) 统一升到 720×480
- Human annotators 拆 scenes → 3-second segments → 每个 segment 写详细 paragraph (Format 3)
- 多阶段拼接出 9/18/30/63 秒训练样本

为什么选 Tom and Jerry？复杂多场景、动态 motion、长程故事依赖——正是当前 video model 短板。这个 proof-of-concept 数据集**重点放在长程 coherence** 而非 photo-realism (因为 Sora 那帮已经解决了)。

### 多阶段 context extension (Table 2)

| Video len | Ctx len | Trainable | LR | Steps |
|---|---|---|---|---|
| 3 s | 18,048 | All (TTT high 1e-4, pre-trained low 1e-5) | Cosine/Constant | 5000 |
| 9 s | 51,456 | TTT + Local Attn (QKVO) | 1e-5 | 5000 |
| 18 s | 99,894 | TTT + Local Attn (QKVO) | 1e-5 | 1000 |
| 30 s | 168,320 | TTT + Local Attn (QKVO) | 1e-5 | 500 |
| 63 s | 341,550 | TTT + Local Attn (QKVO) | 1e-5 | 250 |

- Stage 1: 全模型 fine-tune，TTT/gate 用 1e-4，pre-trained params 用 1e-5。Domain adaptation 到 Tom and Jerry。
- Stage 2-5: 只 fine-tune TTT + gates + self-attn QKVO，pre-trained MLP blocks 冻结，LR 降到 1e-5，避免遗忘 pre-trained world knowledge。

---

## 5. 怎么把它跑快 (systems 优化)

### Mini-batch inner loop (Eq 13, 14)

TTT 默认 causal：$W_t$ 依赖 $W_{t-1}$，没法跨 token 并行。

trick：对 b 个 token (b=64) 一起做 gradient step：
$$W_{ib} = W_{(i-1)b} - \frac{\eta}{b} \sum_{t=(i-1)b+1}^{ib} \nabla \ell(W_{(i-1)b}; x_t)$$

- $i$ = mini-batch index, $i = 1, ..., T/b$
- $W_{ib}$ = 第 $i$ 个 mini-batch 后的 weights
- $W_{(i-1)b}$ = 上一个 mini-batch 后的 weights
- $\eta/b$ = 平均学习率

然后整个 mini-batch 用同一个 $W_{ib}$ 出 output：
$$z_t = f(W_{ib}; x_t), \quad t = (i-1)b+1, ..., ib$$

mini-batch 内 b 个 token 可以**并行** forward/backward。因为 sequence 是 non-causal (diffusion)，所以用 mini-batch 末尾的 $W_{ib}$ 给整个 mini-batch 出 output 是合理的。附带好处：b 个 gradient 平均，降低 variance，更新更稳定。

类比：inner loop 像训一个 NN，b 是 batch size；outer loop 像训整个 Transformer，64 是训练 batch size。

### On-chip Tensor Parallel

GPU 内存层次：
- **HBM**: 全 SM 共享，大但慢 (~3TB/s on H100)
- **SMEM**: 每个 SM 私有，小 (~228KB on H100) 但快 (~30TB/s)
- **SM** = GPU 上的 "core"

FlashAttention / Mamba 的套路：load 进 SMEM，on-chip 计算，只写最终 output 回 HBM (kernel fusion)。

**TTT-MLP 的问题**：hidden state $W^{(1)}, W^{(2)}$ 是两层 MLP 的 weights，太大，单 SM 的 SMEM 装不下。

**解决方案**：把 GPU 当 cluster，SM 当 GPU，做 **Tensor Parallelism across SMs**。
- 第一层 $W^{(1)}$ **column-wise** 分片 (按输出维度切)
- 第二层 $W^{(2)}$ **row-wise** 分片 (按输入维度切)
- GELU 是 elementwise，前向只需一次 reduction 算 inner loss
- 用 NVIDIA Hopper 的 **DSMEM (Distributed Shared Memory)** feature 做 SM 间 AllReduce

效果：hidden state 和 activation 只在初始 load 和最终 output 时读写 HBM，中间全在 SMEM / DSMEM 间流转。

### 进一步 kernel 优化 (Appendix B)

用 **ThunderKittens** 写 kernel：
- **Multi-stage pipelining** (借鉴 FlashAttention-3)：async prefetch 下一个 mini-batch 进 HBM，overlap 数据传输和当前 mini-batch 计算
- **Producer-consumer asynchrony**：专门 warpgroup 当 producer (load data) 或 consumer (compute)
- **Gradient checkpointing along sequence dimension**：省 activation memory
- **TMA (Tensor Memory Accelerator)**：Hopper 的硬件 unit 做 async memory store

### 效率数据 (Figure 6)

| Method | Inference | Training |
|---|---|---|
| Local attention (3s) | 1× | 1× |
| Full attention (300k tokens) | 11× | 12× |
| TTT-MLP | 2.5× | 3.8× |
| Gated DeltaNet | 1.8× | 1.8× |

TTT-MLP 远好于 full attention，但仍比 Gated DeltaNet 慢 (1.4× inference, 2.1× training)。论文承认 kernel 还有提升空间 (register spills, async 指令顺序)。

Training 效率不那么关键——RNN layers 只在 fine-tune 时加，pre-training 占大头。Inference 效率才是 deployment 关键。

---

## 6. 实验结果

### 协议

- 4 个轴 (从 MovieGen 6 个轴里选): Text following, Motion naturalness, Aesthetics, Temporal consistency
- **Pairwise blind comparison** (Elo rating, LMSys Chatbot Arena 系统)
- 100 plots，每方法生成 1 video per plot
- Plot 生成：Claude 3.7 Sonnet 走 Format 1→2→3
- Evaluators: prolific.com, US, English first language, 18-35, 100+ submissions, 98%+ approval
- Demographics: 50.78% male / 47.66% female / 1.56% other; 57.03% White / 23.44% Black / 10.94% Mixed / 5.47% Asian

### Baselines (都加进同一个 CogVideo-X 5B，统一 7.2B 参数)

- **Local attention**: 原架构不动，3s segment 独立 self-attn
- **TTT-Linear**: $f$ 是 linear model 的 TTT
- **Mamba 2**: matrix hidden state, 比 TTT-Linear 大 4×，比 TTT-MLP 小 2×
- **Gated DeltaNet**: DeltaNet + Mamba 2 改进版 update rule
- **Sliding-window attention**: 8192 tokens window (~1.5 秒)

### 63 秒主实验结果 (Table 1)

| Method | Text | Motion | Aesthetics | Temporal | Avg |
|---|---|---|---|---|---|
| Mamba 2 | 985 | 976 | 963 | 988 | 978 |
| Gated DeltaNet | 983 | 984 | 993 | 1004 | 991 |
| Sliding window | 1016 | 1000 | 1006 | 975 | 999 |
| **TTT-MLP** | **1014** | **1039** | **1037** | **1042** | **1033** |

- TTT-MLP avg 1033，第二名 sliding window 999，**+34 Elo**
- 改进最大：**Temporal consistency +38** (vs Gated DeltaNet 1004)，**Motion naturalness +39** (vs sliding window 1000)
- 参考 scale：GPT-4o vs GPT-4 Turbo 是 +29 Elo，GPT-4 vs GPT-3.5 Turbo 是 +46 Elo。+34 是 practically meaningful。

### 18 秒 elimination round (Table 3)

| Method | Avg |
|---|---|
| Local Attention | 962 |
| TTT-Linear | 1001 |
| Mamba 2 | 1005 |
| **Gated DeltaNet** | **1032** |
| SWA | 993 |
| TTT-MLP | 1004 |

**18 秒 (~100k tokens) 时 Gated DeltaNet 最好**，比 Mamba 2 +27，比 TTT-MLP +28。

关键 insight：**短 context 时，linear matrix hidden state 仍然最有效**。TTT 的 expressivity 优势要等到 context 长到几百 k tokens 才显现。

### 定性分析 (Figure 5)

以 "Tom 吃苹果派 / Jerry 偷 / Tom 追" 这段剧情为例：
- **TTT-MLP**: 跨场景、跨镜头 Tom 形象保持一致，动作流畅高质量
- **Sliding window**: 厨房环境改变，房子颜色变，Jerry 偷 pie 场景重复 (3 秒边界附近 collapse)
- **Gated DeltaNet**: 跨镜头 Tom 形象不一致，但厨房环境能保持
- **Mamba 2**: Tom growl 追 Jerry 时外观扭曲，但厨房环境整体保持

### Artifacts (Figure 7)

TTT-MLP 仍有的问题：
1. **Temporal consistency**: 物体在 3 秒 segment 边界 morph (diffusion model 不同 segment 可能采样到不同 mode)
2. **Motion naturalness**: 物体悬空 (重力没建模好)
3. **Aesthetics**: 光照随动作不自然变化，复杂 camera movement (如 parallax) 不准

论文说这些 artifacts 在所有方法都常见，**很可能源自 pre-trained CogVideo-X 5B 本身的能力上限**，不是 TTT 的问题。

---

## 7. 我的直觉理解

### 为什么 TTT 比 Mamba 强？

Mamba 的 hidden state 是一个 $d \times d'$ matrix，所有历史信息线性叠加。300k tokens 压进几千 rank 的 matrix，必然损失结构。

TTT 的 hidden state 是 MLP weights，参数量同尺寸但**非线性 + 4× hidden dim**。MLP 可以拟合任意连续函数，所以记忆容量本质更高。每个新 token 通过梯度更新把 token 的结构信息编码进 weights。

类比：Mamba 像一个固定大小的便签本，TTT 像一个能持续训练的小 brain。

### 为什么短 context 时 TTT 不如 Gated DeltaNet？

短 context (100k tokens) 时，matrix hidden state 容量够用。TTT 的 MLP forward + backward 计算 overhead 在短序列上不划算。只有当 context 长到 matrix 装不下时 (几百 k tokens)，TTT 的 expressivity 才值回票价。

这也提示：**未来 TTT 的 sweet spot 是超长 context** (小时级 video，整个电影)。

### gating + 双向 + local-attn 组合的设计哲学

- **Gating ($\alpha=0.1$ init)**: 保护 pre-trained model，让 TTT 渐进融入，避免 fine-tune 初期灾难性遗忘。参数高效 fine-tune 的经典套路 (类似 LoRA 的 zero-init)。
- **Bidirectional**: 适配 diffusion non-causal 特性，对称扫描。
- **Local self-attn**: 保留 pre-trained CogVideo-X 的短程能力 (它本来就是这么训的)，把 long-range 让给 TTT。**分工**是关键——不指望 TTT 干所有事。

### 局限性的诚实

- TTT-MLP wall-clock 比 Gated DeltaNet 慢 1.4-2.1× (kernel 没优化透)
- 短 video 时 Gated DeltaNet 更好 (TTT 在短 context 不划算)
- Artifacts 可能是 CogVideo-X 5B 本身上限，不是 TTT 问题
- 只验证了 1 分钟 (资源限制)，更长没试

### Future work 特别有意思的方向

1. **Larger hidden state**: $f$ 本身可以是 Transformer！这意味着 TTT layer 内嵌一个 mini-Transformer，每来一个 token 训一步。容量爆炸式增长，可能解锁小时级 video。Titans [2] 已经在这条路上。
2. **Better integration**: bidirectional + gates 只是众多策略之一。AR video model (autoregressive) 可能需要完全不同的 integration 策略。

---

## 8. 一句话再总结 (人话版)

**作者把 RNN layer 的 hidden state 从 matrix 升级成小 MLP weights，每来一个 token 就对这个 MLP 做一步 SGD。这让 hidden state 表达力爆炸，能记住几十万 token 的复杂关系。把这个 layer 通过 gating + bidirectional + local-attn 的组合插进 CogVideo-X 5B，用 5 阶段 context extension 从 3 秒 fine-tune 到 63 秒，在 100 段人类评测上比 Mamba 2 / Gated DeltaNet / sliding window 高 34 Elo points。核心 trick 是 on-chip tensor parallel across SMs + mini-batch inner loop 让它真能跑起来。短 context 时 Gated DeltaNet 仍更好，TTT 的 sweet spot 在超长 context。**

核心 insight 给 Karpathy 听的话：**RNN 的 hidden state 容量决定记忆能力。matrix hidden state 是便签本，neural network hidden state 是小 brain。self-supervised learning 本来就是把海量数据压进 NN weights 的最好工具，那就用它当 RNN 的 update rule。** 这就是 TTT 的全部。

参考链接：
- 项目网站: https://test-time-training.github.io/video-dit
- TTT 原始论文: https://arxiv.org/abs/2407.04620
- Titans (后续): https://arxiv.org/abs/2501.00663
- Fast Weight Programmers (Schmidhuber 1992 起源): https://arxiv.org/abs/2410.23768
- CogVideo-X: https://arxiv.org/abs/2408.06072
- Mamba 2: https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2411.14368
- ThunderKittens: https://arxiv.org/abs/2407.10358
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- LMSys Chatbot Arena: https://lmarena.ai
- MovieGen benchmark: https://arxiv.org/abs/2410.13720

---

# TTT for One-Minute Video Generation 深度解析

这篇 paper 来自 Yu Sun 团队 (Stanford/NVIDIA)，是 TTT (Test-Time Training) 系列工作的 video 生成应用版本。核心 claim 是：把 TTT layers 插到 pre-trained CogVideo-X 5B Diffusion Transformer 里，能从 3 秒短片扩展到一分钟长视频，且能讲多场景复杂故事，比 Mamba 2 / Gated DeltaNet / sliding-window attention 在 100 段视频的人类评测上高 34 Elo points。

参考链接：
- TTT 原始论文 (NeurIPS 2024): https://arxiv.org/abs/2407.04620
- 本项目网站 (含 video samples, code, annotations): https://test-time-training.github.io/video-dit
- CogVideo-X 5B: https://arxiv.org/abs/2408.06072
- Mamba 2: https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2411.14368
- ThunderKittens (kernel 实现): https://arxiv.org/abs/2407.10358
- MovieGen benchmark: https://arxiv.org/abs/2410.13720
- LMSys Chatbot Arena: https://lmarena.ai
- Titans (相关后续，更大 hidden state): https://arxiv.org/abs/2501.00663
- FlashAttention-3: https://arxiv.org/abs/2407.08608

---

## 1. Motivation: 为什么 Transformer 跑不了长视频

到 2025 年 3 月为止公开 API 的最长 video 生成时长：Sora 20s, MovieGen 16s, Ray 2 10s, Veo 2 8s。**没有一个**能自主生成多场景复杂故事 (multi-scene complex stories)。

根本瓶颈在 **long context**：
- Self-attention 复杂度 O(n²)
- 一分钟 video 用标准 tokenizer 编码后超过 **300k tokens**
- 生成一分钟 video = 生成 20 段 3 秒 video 的 11× 时间，训练是 12×

Video 不像 text 可以靠 tokenizer 大幅压缩——动态画面的每个 patch 都要保留。所以不能简单压缩 context。

之前 LinGen [47] 等用 RNN layers (linear complexity) 也能跑 minute-length video，但只做单场景、慢动作，没有复杂故事。论文给出的 hypothesis 是：**linear-attention 变种 (Mamba, DeltaNet) 的 hidden state 只是一个 matrix，表达能力不够**——把几十万 vector 压到几千 rank 的 matrix 里，记不住远距离 token 之间的深度关系。

这就是 TTT 切入的位置。

---

## 2. TTT 的核心思想：hidden state 本身就是 neural network

### 2.1 从 RNN 视角理解

所有 RNN layer 的本质：把历史 context $x_1, ..., x_t$ 压缩进一个 fixed-size hidden state，再按 update rule 转移：

- update rule + output rule 都是 O(1) per token → 整体 O(T)
- 但记忆能力被 hidden state 的容量限制

Mamba / DeltaNet 的 hidden state 是一个 matrix $W \in \mathbb{R}^{d \times d'}$ (rank 几千)。TTT 的核心 insight 来自 [43]：

> **既然 self-supervised learning 能把海量训练集压缩进 NN weights，那就把历史 context 当作 unlabeled dataset，把 hidden state 当作 NN weights，update rule 当作 gradient step on self-supervised loss。**

这把 RNN 的 hidden state 从"matrix"升级成"learnable function"。MLP 的参数空间远大于同尺寸 matrix，非线性也丰富，能存更多结构化信息。

### 2.2 数学公式详解

**核心 update (Eq 1)**：
$$W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)$$

- $t$: 时间步 index (1, 2, ..., T)
- $W_t$: 第 $t$ 步更新后的 hidden state，即 inner model $f$ 的 weights
- $W_{t-1}$: 上一步的 weights
- $\eta$: inner-loop learning rate (TTT-Linear 用 $\eta=1.0$，TTT-MLP 用 $\eta=0.1$)
- $\ell$: self-supervised loss
- $x_t$: 第 $t$ 个 input token

直观：每来一个新 token，就对 $W$ 做一步 SGD。

**Output rule (Eq 2)**：
$$z_t = f(x_t; W_t)$$

- $z_t$: 输出 token
- $f$: inner model (hidden state 是它的 weights)
- $x_t$: 当前 input
- $W_t$: 已经被当前 token 更新过的 weights

注意顺序：先 update 再 forward，所以 $z_t$ 反映了"刚学过 $x_t$ 之后 $f$ 对 $x_t$ 的反应"。

**Naive reconstruction loss (Eq 3)**：
$$\ell(W; x_t) = \|f(\tilde{x}_t; W) - x_t\|^2$$

- $\tilde{x}_t$: corrupted input (像 denoising autoencoder)
- $x_t$: 原始 input 作为 reconstruction target
- $f(\tilde{x}_t; W)$: $f$ 在 corrupted input 下的输出

$f$ 必须发现 $x_t$ 各维度之间的 correlation 才能从 partial information 重建自己。这一过程把 token 的结构信息"消化"进 $W$。

### 2.3 Learnable self-supervised task (Eq 4, 5)

手工设计自监督任务太 brittle。[43] 把它端到端 learn 出来，通过三个 learnable projection matrices $\theta_K, \theta_V, \theta_Q$ (类比 attention 的 K/V/Q)：

**Loss (Eq 4)**：
$$\ell(W; x_t) = \|f(\theta_K x_t; W) - \theta_V x_t\|^2$$

- $\theta_K$: Key projection，把 $x_t$ 投影成低维 corrupted input $\tilde{x}_t = \theta_K x_t$ (outer loop learnable)
- $\theta_V$: Value projection，把 $x_t$ 投影成低维 reconstruction target $\theta_V x_t$ (outer loop learnable)
- $W$: 唯一在 inner loop 优化的参数
- $f(\theta_K x_t; W)$: inner model 在 corrupted input 下的输出

设计哲学：不是所有 $x_t$ 的信息都值得记忆，所以 reconstruction target 也压缩到 $\theta_V x_t$。$\theta_K, \theta_V$ 决定"学什么"，$W$ 决定"怎么学"。

**Output rule (Eq 5)**：
$$z_t = f(\theta_Q x_t; W_t)$$

- $\theta_Q$: Query projection，把 $x_t$ 投影成 query 输入
- $W_t$: 更新后的 hidden state

为什么需要 $\theta_Q$？因为 $\theta_K x_t$ 维度比 $x_t$ 少，没法直接用 Eq 2 的 $f(x_t; W_t)$，所以引入第三个 projection 做 query。$\theta_K, \theta_V, \theta_Q$ 在 outer loop 训练 (训练更大网络时)，类比 self-attention 的 Q/K/V parameters。

### 2.4 Inner vs Outer loop (Meta-learning 视角)

- **Outer loop**: 训练整个网络 (含 TTT layer 的 $\theta_K, \theta_V, \theta_Q$ 和 MLP blocks)，相当于 standard training
- **Inner loop**: 每个 TTT layer 在每个 test sequence 上自己训一遍 $W_1, ..., W_T$，相当于 meta-learning 中的 "learning to learn"

backward $\nabla \ell$ 需要 gradient of gradient (二阶)，是 meta-learning 经典技术。TTT layer 接口和 RNN / self-attention 一样，可以 plug-in 替换。

### 2.5 TTT-MLP Instantiation (Eq 6)

$f$ 包成 $f_{MLP}$ 的 wrapper：
$$f(x) = x + \text{LN}(f_{MLP}(x))$$

- $f_{MLP}$: 两层 MLP，hidden dimension = 4× input dimension，GELU activation
- LN: Layer Norm
- $x + \text{LN}(\cdot)$: residual connection

LN + residual 保证 TTT inner loop 的稳定性 (SGD 在深网络里容易发散)。

TTT-Linear 基线：$f(x) = x + \text{LN}(f_{Linear}(x))$，$f_{Linear}$ 是 linear model。这是为了消融——看 non-linearity 在 hidden state 里的作用。

---

## 3. Architecture: 怎么 plug 进 pre-trained DiT

### 3.1 整体修改

Backbone: CogVideo-X 5B (Diffusion Transformer, expert transformer 结构)，原本只能生成 3 秒 (16fps) 或 6 秒 (8fps)。

修改点：**只在 sequence modeling block 里加东西，MLP block 不动**。

原始 Transformer block:
$$X' = \text{self\_attn}(\text{LN}(X))$$
$$Y = X' + X$$

修改后 (Eq 8-12):
$$X' = \text{self\_attn}(\text{LN}(X))$$  (Eq 8, 局部 attention)
$$Z = \text{gate}(\text{TTT}, X'; \alpha)$$  (Eq 10, forward direction TTT)
$$Z' = \text{gate}(\text{TTT}', Z; \beta)$$  (Eq 11, backward direction TTT)
$$Y = Z' + X$$  (Eq 12)

### 3.2 Gating (Eq 6)

$$\text{gate}(\text{TTT}, X; \alpha) = \tanh(\alpha) \otimes \text{TTT}(X) + X$$

- $\alpha \in \mathbb{R}^d$: learnable gating vector (per-channel)
- $\tanh(\alpha) \in (-1, 1)^d$: element-wise 在每个 channel 上压到 (-1, 1)
- $\otimes$: element-wise 乘
- $+ X$: residual，保留原始 input

初始化 $\alpha = 0.1$，所以 $\tanh(0.1) \approx 0.1$。开始 fine-tune 时 TTT 贡献很小 (~0.1×)，避免随机初始化的 TTT 破坏 pre-trained CogVideo-X 的预测。随着 fine-tune 推进，$\alpha$ 会被 learned 调整，让 TTT 慢慢 "take over"。

这是 Flamingo [1] 风格的 gated cross-attention trick，本质是 residual scaling 防止新模块冷启动退化。

### 3.3 Bi-direction (Eq 7)

Diffusion model 是 **non-causal** 的：output token $z_t$ 可以依赖所有 $x_1, ..., x_T$，包括未来。但 TTT 默认是 causal 的 (chronological)。

trick: 跑两次，方向相反：

$$\text{TTT}'(X) = \text{rev}(\text{TTT}(\text{rev}(X)))$$

- $\text{rev}(X) = (x_T, ..., x_1)$: 时间反转算子
- 内层 $\text{TTT}(\text{rev}(X))$: 在反转后的序列上跑 TTT (从 $x_T$ 扫到 $x_1$)
- 外层 $\text{rev}(\cdot)$: 把输出反转回 chronological 顺序

结果 $\text{TTT}'(X)$ 仍然按时间顺序输出，但 TTT 内部扫描方向反过来，捕获 "future → past" 信息流。两个 TTT (forward + backward) 一起类似 bidirectional RNN / bidirectional Mamba [30]。

注意 $\text{TTT}$ 和 $\text{TTT}'$ 共享同一套 $\theta_K, \theta_V, \theta_Q$ (weight tying)，只有 gating $\alpha, \beta$ 不同。这降低参数量。

### 3.4 Local attention + Global TTT

关键 efficiency 决策：**self-attn 限制在 3 秒 segment 内 (local)，TTT 处理整段 60s (global)**。

原因：
- self-attn O(n²)，60s = 300k tokens 全局爆炸
- TTT O(n) 线性，可以扛长 context
- 3 秒是 CogVideo-X 原生能生成的长度，local self-attn 保留它的短程建模能力

这相当于一个 hybrid 架构：local attention 处理 intra-segment，TTT 处理 inter-segment long-range。

---

## 4. Pipeline 和数据

### 4.1 三种 prompt 格式

- **Format 1**: 5-8 句 plot 摘要 (用户友好)
- **Format 2**: ~20 句，每句对应 ~3 秒 segment
- **Format 3**: storyboard，每段 3-5 句，含背景颜色、镜头运动等细节，用 `<scene start>` / `<scene end>` 严格标记 scene 边界

Fine-tune 和 inference 时**永远用 Format 3** 喂给 text tokenizer。1→2→3 的扩展用 Claude 3.7 Sonnet 自动完成。所以用户只要写 5-8 句话就能得到 1 分钟 video。

### 4.2 输入序列构造

对每个 3-second segment 独立 tokenize：
1. 把 paragraph 的 text 转成 text tokens
2. 把 noisy video 转成 video tokens
3. 拼成 `[text_tokens_seg_i, video_tokens_seg_i]`
4. 全部 segment 拼接：`[text_1, vid_1, text_2, vid_2, ..., text_n, vid_n]`

所以输入是 interleaved text + video tokens 的超长序列。

### 4.3 多阶段 context extension (Table 2)

LLM context extension 的标准做法 [51]，分 5 阶段：

| Video len | Ctx len | Trainable | LR | Steps |
|---|---|---|---|---|
| 3 s | 18,048 | All (TTT high, pre-trained low) | 1e-4 / 1e-5 | 5000 |
| 9 s | 51,456 | TTT + Local Attn (QKVO) | 1e-5 | 5000 |
| 18 s | 99,894 | TTT + Local Attn (QKVO) | 1e-5 | 1000 |
| 30 s | 168,320 | TTT + Local Attn (QKVO) | 1e-5 | 500 |
| 63 s | 341,550 | TTT + Local Attn (QKVO) | 1e-5 | 250 |

- **Stage 1**: 全模型 fine-tune，TTT/gate 用 1e-4，pre-trained params 用 1e-5。Domain adaptation 到 Tom and Jerry。
- **Stage 2-5**: 只 fine-tune TTT + gates + self-attn QKVO，pre-trained MLP blocks 冻结。LR 降到 1e-5，避免遗忘 pre-trained world knowledge。

Cosine schedule 用在 Stage 1 TTT 参数，constant 用在其他；warmup 2% steps。Batch 64，grad clip 0.1，weight decay 1e-4。

### 4.4 Dataset

- **Tom and Jerry** 1940-1948 共 81 集，~5 分钟/集，总 ~7 小时
- Video super-resolution model [49] (Real-ESGAN) 统一升到 720×480
- Human annotators 把每集拆成 scenes，每个 scene 拆 3-second segments
- 每个 segment 写一个详细 paragraph (Format 3)
- 多阶段拼接出 9/18/30/63 秒训练样本，文本同步拼接

为什么选 Tom and Jerry？复杂多场景、动态 motion、长程故事依赖——正是当前 video model 短板。视觉/物理真实度已由 Sora 等解决，所以这个 proof-of-concept 数据集**重点放在长程 coherence** 而非 photo-realism。

---

## 5. Parallelization 和 Systems 优化

### 5.1 Mini-batch inner loop (Eq 13, 14)

TTT 默认 causal：$W_t$ 依赖 $W_{t-1}$，没法跨 token 并行。

trick：对 **b 个 token** 一起做 gradient step (这里 b=64)：

$$W_{ib} = W_{(i-1)b} - \frac{\eta}{b} \sum_{t=(i-1)b+1}^{ib} \nabla \ell(W_{(i-1)b}; x_t)$$

- $i$: mini-batch index, $i = 1, ..., T/b$
- $W_{ib}$: 第 $i$ 个 mini-batch 后的 weights
- $W_{(i-1)b}$: 上一个 mini-batch 后的 weights (作为计算梯度的起点)
- $\eta/b$: 平均学习率 (除以 b 做平均)
- 求和: 把 b 个 token 的梯度累加

然后这整个 mini-batch 用同一个 $W_{ib}$ 出 output：
$$z_t = f(W_{ib}; x_t), \quad t = (i-1)b+1, ..., ib$$

- 每个 mini-batch 内的 b 个 token 可以**并行** forward 和 backward
- 因为 sequence 是 non-causal (diffusion model)，所以用 mini-batch 末尾的 $W_{ib}$ 给整个 mini-batch 出 output 是合理的
- 附带好处：b 个 gradient 平均，降低 variance，更新更稳定

类比：inner loop 像训一个 NN，b 是 batch size；outer loop 像训整个 Transformer，64 是训练 batch size。

### 5.2 On-chip Tensor Parallel (Section 3.5)

GPU 内存层次：
- **HBM (High Bandwidth Memory)**: 全 SM 共享，大但慢 (~3TB/s on H100)
- **SMEM (Shared Memory)**: 每个 SM 私有，小 (~228KB on H100) 但快 (~30TB/s)
- **SM (Streaming Multiprocessor)**: GPU 上的"core"

FlashAttention / Mamba 高效实现的套路：load 进 SMEM，on-chip 计算，只写最终 output 回 HBM (kernel fusion)。

**TTT-MLP 的问题**：hidden state $W^{(1)}, W^{(2)}$ 是两层 MLP 的 weights，太大，单 SM 的 SMEM 装不下 (加上 input 和 activation)。

**解决方案**：把 GPU 当作 cluster，SM 当作 GPU，做 **Tensor Parallelism across SMs**。

- 第一层 $W^{(1)}$ **column-wise** 分片 (按输出维度切)
- 第二层 $W^{(2)}$ **row-wise** 分片 (按输入维度切)
- GELU 是 elementwise，所以前向只需一次 reduction 算 inner loss
- 用 NVIDIA Hopper 的 **DSMEM (Distributed Shared Memory)** feature 做 SM 间 AllReduce

效果：hidden state 和 activation 只在初始 load 和最终 output 时读写 HBM，中间全在 SMEM / DSMEM 间流转。

### 5.3 进一步 kernel 优化 (Appendix B)

用 **ThunderKittens** [42] 写 kernel：

- **Multi-stage pipelining** (借鉴 FlashAttention-3 [38])：async prefetch 下一个 mini-batch 进 HBM，overlap 数据传输和当前 mini-batch 计算
- **Producer-consumer asynchrony**：专门 warpgroup 当 producer (load data) 或 consumer (compute)
- **Gradient checkpointing along sequence dimension**：省 activation memory
- **TMA (Tensor Memory Accelerator)**: Hopper 的硬件 unit 做 async memory store，减少 I/O stall 和 CUDA thread 工作量

### 5.4 效率数据 (Figure 6)

对比 local attention (3s segment 独立)：

| Method | Inference | Training |
|---|---|---|
| Full attention (300k tokens) | 11× | 12× |
| TTT-MLP | 2.5× | 3.8× |
| Gated DeltaNet | 1.8× | 1.8× |

TTT-MLP 远好于 full attention，但仍比 Gated DeltaNet 慢 (1.4× inference, 2.1× training)。论文承认 kernel 还有提升空间 (register spills, async 指令顺序)。

注意：**training 效率不那么关键**——RNN layers 只在 fine-tune 时加，pre-training 占大头。Inference 效率才是 deployment 关键。

---

## 6. 实验结果

### 6.1 评估协议

- 4 个轴 (从 MovieGen 6 个轴里选): Text following, Motion naturalness, Aesthetics, Temporal consistency
- **Pairwise blind comparison** (Elo rating, LMSys Chatbot Arena 系统 [6])
- 100 plots，每方法生成 1 video per plot
- Plot 生成：Claude 3.7 Sonnet 走 Format 1→2→3
- Evaluators: prolific.com, US, English first language, 18-35, 100+ submissions, 98%+ approval
- Demographics: 50.78% male / 47.66% female / 1.56% other; 57.03% White / 23.44% Black / 10.94% Mixed / 5.47% Asian

### 6.2 Baselines (都加进同一个 CogVideo-X 5B，统一 7.2B 参数)

- **Local attention**: 原架构不动，3s segment 独立 self-attn
- **TTT-Linear**: $f$ 是 linear model 的 TTT
- **Mamba 2** [8]: matrix hidden state, 比 TTT-Linear 大 4×，比 TTT-MLP 小 2×
- **Gated DeltaNet** [53]: DeltaNet + Mamba 2 改进版 update rule
- **Sliding-window attention** [3]: 8192 tokens window (~1.5 秒)

### 6.3 63 秒主实验结果 (Table 1)

| Method | Text | Motion | Aesthetics | Temporal | Avg |
|---|---|---|---|---|---|
| Mamba 2 | 985 | 976 | 963 | 988 | 978 |
| Gated DeltaNet | 983 | 984 | 993 | 1004 | 991 |
| Sliding window | 1016 | 1000 | 1006 | 975 | 999 |
| **TTT-MLP** | **1014** | **1039** | **1037** | **1042** | **1033** |

- TTT-MLP avg 1033，第二名 sliding window 999，**+34 Elo**
- 改进最大：**Temporal consistency +38** (vs Gated DeltaNet 1004)，**Motion naturalness +39** (vs sliding window 1000)
- 参考 scale：GPT-4o vs GPT-4 Turbo 是 +29 Elo，GPT-4 vs GPT-3.5 Turbo 是 +46 Elo，所以 +34 是 practically meaningful

### 6.4 18 秒 elimination round (Table 3)

| Method | Avg |
|---|---|
| Local Attention | 962 |
| TTT-Linear | 1001 |
| Mamba 2 | 1005 |
| Gated DeltaNet | 1032 |
| SWA | 993 |
| TTT-MLP | 1004 |

关键发现：**18 秒 (~100k tokens) 时 Gated DeltaNet 最好**，比 Mamba 2 +27，比 TTT-MLP +28。

这说明：**短 context 时，linear matrix hidden state 仍然最有效**。TTT 的 expressivity 优势要等到 context 长到几百k tokens 才显现。Gated DeltaNet 比 Mamba 2 有稳定改进 (两轮都 +20 左右)。

Local attention 和 TTT-Linear 在这轮被淘汰，不进入 63 秒主实验。

### 6.5 定性分析 (Figure 5)

以 "Tom 吃苹果派 / Jerry 偷 / Tom 追" 这段剧情为例：

- **TTT-MLP**: 跨场景、跨镜头 Tom 形象保持一致，动作流畅高质量
- **Sliding window**: 厨房环境改变，房子颜色变，Jerry 偷 pie 场景重复 (3 秒边界附近 collapse)
- **Gated DeltaNet**: 跨镜头 Tom 形象不一致，但厨房环境能保持
- **Mamba 2**: Tom growl 追 Jerry 时外观扭曲，但厨房环境整体保持

### 6.6 Artifacts (Figure 7)

TTT-MLP 仍有的问题：

1. **Temporal consistency**: 物体在 3 秒 segment 边界 morph (因为是 diffusion model，不同 segment 可能采样到不同 mode)
2. **Motion naturalness**: 物体悬空 (重力没建模好)
3. **Aesthetics**: 光照随动作不自然变化，复杂 camera movement (如 parallax) 不准

论文说这些 artifacts 在所有方法都常见，**很可能源自 pre-trained CogVideo-X 5B 本身的能力上限**，不是 TTT 的问题。

---

## 7. 我的 Intuition 总结

### 7.1 为什么 TTT 比 Mamba 强？

Mamba 的 hidden state 是一个 $d \times d'$ matrix，所有历史信息线性叠加。300k tokens 压进几千 rank 的 matrix，必然损失结构。

TTT 的 hidden state 是 MLP weights，参数量同尺寸但**非线性 + 4× hidden dim**。MLP 可以拟合任意连续函数，所以记忆容量本质更高。每个新 token 不是简单"加进去"，而是被"学习"——通过梯度更新把 token 的结构信息编码进 weights。

类比：Mamba 像一个固定大小的便签本，TTT 像一个能持续训练的小 brain。

### 7.2 为什么 mini-batch + bidirectional 是关键工程 trick？

- **Mini-batch (b=64)**: 解决了 TTT causal 依赖无法并行的问题。代价是 mini-batch 内部用同一个 $W_{ib}$，丢了 token-level granularity，但收益是 GPU 利用率 + 梯度方差下降。
- **Bidirectional**: diffusion 是 non-causal 的，单向 TTT 会损失一半信息流。reverse-trick 让 TTT 既看 past 又看 future，对应 diffusion model 的全局 conditioning。

### 7.3 为什么短 context 时 TTT 不如 Gated DeltaNet？

短 context (100k tokens) 时，matrix hidden state 容量够用。TTT 的 MLP forward + backward 计算 overhead 在短序列上不划算。只有当 context 长到 matrix 装不下时 (几百k tokens)，TTT 的 expressivity 才值回票价。

这也提示：**未来 TTT 的 sweet spot 是超长 context** (小时级视频，整个电影)。

### 7.4 gating + 双向 + local-attn 组合的设计哲学

- **Gating ($\alpha=0.1$ init)**: 保护 pre-trained model，让 TTT 渐进融入，避免 fine-tune 初期灾难性遗忘。这是参数高效 fine-tune 的经典套路 (类似 LoRA 的 zero-init)。
- **Bidirectional**: 适配 diffusion non-causal 特性，对称扫描。
- **Local self-attn**: 保留 pre-trained CogVideo-X 的短程能力 (它本来就是这么训的)，把 long-range 让给 TTT。**分工**是关键——不指望 TTT 干所有事。

### 7.5 局限性的诚实

论文承认：
- TTT-MLP wall-clock 比 Gated DeltaNet 慢 1.4-2.1× (kernel 没优化透)
- 短 video 时 Gated DeltaNet 更好 (TTT 在短 context 不划算)
- Artifacts 可能是 CogVideo-X 5B 本身上限，不是 TTT 问题
- 只验证了 1 分钟 (资源限制)，更长没试

### 7.6 Future work 的两个特别有意思的方向

1. **Larger hidden state**: $f$ 本身可以是 Transformer！这意味着 TTT layer 内嵌一个 mini-Transformer，每来一个 token 训一步。容量爆炸式增长，可能解锁小时级 video。
2. **Better integration**: bidirectional + gates 只是众多策略之一。AR video model (autoregressive) 可能需要完全不同的 integration 策略。

---

## 8. 联想：这篇 paper 在大图景里的位置

- **TTT 系列**：从 [43] 的 RNN with expressive hidden state (NeurIPS 2024)，到这篇的 video generation 应用，再到 [2] Titans 用更复杂 optimizer 训 hidden state，是一条完整 line。
- **Linear attention 复兴**：Mamba 2 / DeltaNet / Gated DeltaNet / TTT 都属于 "Fast Weight Programmers" 谱系 (Schmidhuber 1992)。本质上都在问：hidden state 应该多大？多非线性？怎么 update？
- **Video generation 长程问题**：LinGen [47] 用 Mamba 做 minute-level video 但只有单场景慢动作；StoryDiffusion [56] 用 diffusion + 自定义 module 做 story visualization 但不端到端。这篇是第一个用 RNN-layer 做出真正多场景复杂 story 的端到端 video DiT。
- **Context extension for diffusion**：类似 LLM 的 context extension (从 4k 到 1M tokens)，diffusion model 现在也在经历同样的 stage。多阶段 fine-tune (3→9→18→30→63s) 几乎照搬 LLM 的 NTK-aware / progressive extension 套路 [51]。
- **Meta-learning 回归**：TTT 的 inner loop 本质是 MAML 风格的 meta-learning，但用 self-supervised loss 代替 task-specific loss。这让 inner loop 不需要标签，可以用在 test-time 任意 sequence 上。Schmidhuber 在 1990s 提出的 fast weight programmer 思想，30 年后终于因为 GPU 算力 + kernel 技术成熟而 practical。

---

## 9. 实操细节总结 (想复现的人需要看)

- **Diffusion schedule**: v-prediction [34], 1000 steps, Zero-SNR [27]
- **Sampling**: DDIM 50 steps, dynamic CFG 1→4, negative prompts
- **TTT inner LR**: $\eta = 1.0$ (TTT-Linear), $\eta = 0.1$ (TTT-MLP) —— MLP 需要小 LR 防发散
- **Mini-batch**: $b = 64$
- **AdamW**: $(\beta_1, \beta_2) = (0.9, 0.95)$, weight decay 1e-4
- **Dropout**: 10% text prompt zero-out (classifier-free guidance 训练)
- **Precision**: Mixed Precision with PyTorch FSDP2
- **Compute**: 等效 256 H100 × 50 小时 (preliminary optimization，没榨干)
- **ThunderKittens** 写 kernel, Hopper GPU 上跑

整体感受：这是一篇非常 honest 的 proof-of-concept paper。它没有 over-claim，明确说 1 分钟 artifacts 主要来自 base model 限制。真正的 contribution 在两点：(1) 工程上把 TTT-MLP 真的跑通在 300k-token diffusion model 上；(2) 实验上证明了 expressive hidden state 在长程 video coherence 上的优势。下一个里程碑估计是更长 video + 更大 hidden state ($f$ = Transformer) + 更优 kernel。
