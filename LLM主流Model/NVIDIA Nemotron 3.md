---
source_pdf: NVIDIA Nemotron 3.pdf
paper_sha256: cc3456be539fdd4dd39d6f3614b1e89d34d0b5ba7801ae95aa1b5ac9061395ad
processed_at: '2026-08-05T22:48:30-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Nemotron 3 用人话讲

好，抛开公式，用直觉来 build understanding。

---

## 整体 picture：NVIDIA 在造什么

NVIDIA 造了三个 model：Nano（30B-A3B，已开源）、Super、Ultra（后面两个待发布）。这三个 model 共享同一套技术栈，只是 scale 不同。

核心 thesis 一句话：**"用更少的 byte 和更少的 compute，做出一样甚至更好的 accuracy"**。所有技术点都围绕这个 thesis 展开。

---

## 1. Hybrid Mamba-Transformer MoE：核心架构

### Mental model：两种记忆系统

你脑中有两种记忆：
- **Episodic memory**（情节记忆）：你能 vivid recall 昨天晚餐吃了什么，但这个 recall 系统容量有限，越记越多越累
- **Semantic memory**（语义记忆）：你 "知道" 巴黎是法国首都，这个知识 constant size，不随时间增长

Transformer attention 就像 episodic memory——它存一个 KV cache，所有历史 token 都在里面，recall 时要扫一遍，越扫越长。好处是 **perfect fidelity**，坏处是 **cost 线性增长**。

Mamba-2 就像 semantic memory——它把历史 "压缩" 进一个 fixed-size state，recall cost 与序列长度无关。好处是 **constant cost**，坏处是 **有损压缩**，长 context retrieval 会丢信息。

Nemotron 3 的做法：**主体用 Mamba-2（省 cost），少量 layer 用 attention（补 fidelity）**。

### 为什么 Mamba-2 而非 Mamba-1

Mamba-1 的 selective scan 算法是 sequential 的，GPU 上效率差。Mamba-2 发现 SSM 和 attention 存在数学对偶（SSD, Structured State Space Duality），可以重写成 matrix multiplication，直接吃 GPU tensor core。

更通俗说：**Mamba-1 是 "用循环算"，Mamba-2 是 "用矩阵乘算"**，后者在 GPU 上快得多，state size 也能开更大。Mamba-1 的 state 通常是 16-64 维，Mamba-2 可以到 128+ 维，容量更大。

### 具体数字

Nemotron 3 Nano 相比 Qwen3-30B-A3B（纯 Transformer MoE，同样 active params），**throughput 高 3.3×**。序列越长 speedup 越明显，因为 KV cache 的 linear cost 在长序列上 dominate。

### Attention layer 怎么放

Paper 说 attention layer 很少，每个用 GQA with only 2 KV heads。直觉：attention 是 "奢侈品"，少量高 fidelity layer 做 all-to-all information routing，剩下交给 Mamba-2 做 cheap sequence modeling。

---

## 2. LatentMoE：最巧妙的 idea

### Standard MoE 的两个 bottleneck

先建立 intuition：MoE layer 在两种部署场景下 bottleneck 完全不同。

**Latency 场景**（batch size 小，比如几十 tokens）：你生成一个 token 要等 expert weight 从 HBM 读到 SRAM 计算。读 weight 的 bandwidth cost 远大于计算 cost。这是 **memory-bound**。

**Throughput 场景**（batch size 大，几千 tokens）：你把 batch 里的 tokens 发给不同 expert，这个 all-to-all 通信是 **communication-bound**。通信量与 active expert 数 $K$ 和 hidden dim $d$ 成正比，**与 expert 的中间宽度 $m$ 无关**。

### 标准 MoE 怎么花 budget

假设 hidden dim $d=4096$，128 个 expert，top-6 active：
- 每个 expert 是 $4096 \times m \times 4096$ 的 FFN
- 通信量：$6 \times 4096$ per token
- Nonlinear unit 数：$6 \times m$ per token（这是 "表达能力" 的来源）

### LatentMoE 的关键 insight

Paper 发现：**routing 和 communication 的 cost 依赖 $d$，但 expert 的 expressive power 依赖 $K \times m$**。

那如果我把 routing 从 4096 维降到 1024 维呢？
- 通信量从 $6 \times 4096$ 变成 $6 \times 1024$，**省了 4×**
- Bandwidth（读 expert weight）也省 4×（expert 输入维度小了）

省下来的 budget 拿去干嘛？**加更多 expert，加更多 active expert**。

具体：$d=4096 \to \ell=1024$，expert 数 $128 \to 512$（×4），active $6 \to 22$（约 ×4）。

重新算：
- 通信量：$22 \times 1024 = 22528 \approx 6 \times 4096 = 24576$，**几乎不变**
- Active compute：$22 \times (1024 \times m \times 1024) \approx 6 \times (4096 \times m \times 4096)$，**几乎不变**
- Nonlinear unit 数：$22 \times m$ vs $6 \times m$，**增加约 3.7×**！

### 直觉解释

**同样 compute，同样 bandwidth，但你给了模型 3.7× 的 nonlinear capacity，分布在 4× 多的 expert 上**。

为什么这有用？想象标准 MoE 是 6 个 "全才专家"，LatentMoE 是 22 个 "专才专家"。专才虽然脑子小，但分工细，每个专才只处理自己擅长的窄领域，总体效果更好。

实验数据：1T tokens 同样 hyperparameter 训练，MMLU-Pro 从 48.30 涨到 52.87（+4.57）。这是免费 lunch——**用 LatentMoE 的 "降维 routing" 思想，reallocate compute 从 "宽度" 到 "数量"**。

### 联想

这有点像 LoRA 的精神：低秩 projection 后再处理。但 LoRA 是 adapter，LatentMoE 是 routing space 的低秩化。

也像 group convolution 的思想：把 channel 拆组分别处理，减 parameter 但保持 representation power。

**关键 trick**：non-routed computation（gating network、shared expert、non-expert layers）保持原 hidden dim $d$，因为它们不是 bottleneck。只有 routed expert 进入 latent space。

---

## 3. MTP（Multi-Token Prediction）：训练 + 推理双用途

### 训练角度的 intuition

Standard next-token prediction 像蒙着眼睛走迷宫：每步只看 1 个 token 的 loss signal，模型学到 "局部贪心" 策略。

MTP 像让模型同时看未来几步：每步预测 $x_{t+1}, x_{t+2}, ..., x_{t+k}$，强迫模型在 hidden state $h_t$ 里 encode 多步 plan 的信息。

具体：第 $t$ 个 token 的 hidden state $h_t$，经过 $k$ 个独立 prediction head，分别预测 $x_{t+1}, ..., x_{t+k}$。Loss 是 $k$ 个 cross-entropy 的和。

效果：8B MoE 上 1T tokens 训练，平均 +2.4% on benchmarks。MMLU-Pro +2.79, GSM8K +1.97。Reasoning 任务提升更明显，符合 "plan ahead" 的 intuition。

### 推理角度的 intuition

Speculative decoding 标准做法：训一个 small draft model 生成 $k$ 个 candidate tokens，main model 并行验证。Draft model 要额外训，额外存。

MTP 的妙处：**那 $k$ 个 prediction head 直接当 draft model 用**。Head 1 预测 $x_{t+1}$，head 2 预测 $x_{t+2}$（基于 $h_t$ + $x_{t+1}$ embedding），等等。这些 head 与 main model 共享 trunk，所以 draft 与 main 高度一致。

Paper 报告：**first two predicted tokens 97% acceptance rate**。这意味着 inference 时平均能跳过很多 verification step，生成速度显著加快。

### 双重价值

MTP 同时改善 training signal density 和 inference speed，deepseek-v3 也用同样的 trick。Nemotron 3 把 MTP 用在 RL rollout 加速上，rollout 快 → RL 训练快 → 能用更大 batch size 训更多 step。

---

## 4. NVFP4：4-bit 训练

### 格式直觉

FP4 元素只有 4 bit：1 sign + 2 exponent + 1 mantissa，能表示 8 个 positive 值 + 8 个 negative 值 + 0。这显然太粗，所以加 hierarchical scaling：
- 16 个 element 共享一个 E4M3 block scale
- 整个 tensor 共享一个 FP32 global scale

Value = global × block × element，三层 scale 让 FP4 能表示的 dynamic range 足够大。

### 为什么 4-bit 难训

4-bit 的 quantization error 很大，直接量化 gradient 会爆炸。Nemotron 3 的 recipe 有几个关键：

**Random Hadamard Transform (RHT) on wgrad input**：gradient backward 时，input 先乘一个 Hadamard matrix（正交变换）。直觉：Hadamard rotation 像 "打散" 信息，把 outlier 集中的维度均匀化，避免某维度因 outlier 而 4-bit 量化丢太多信息。

**Stochastic rounding**：量化时按概率 round up/down，期望无偏。对 4-bit 这种粗粒度，deterministic rounding 的 systematic bias 会累积，stochastic 让噪声 zero-mean。

### 哪些 layer 敏感

Paper 的 ablation 给出清晰 recipe：
- **Mamba output projection**：NVFP4 下 40% values flush to zero，信息丢失严重 → 改用 **MXFP8**
- **QKV, attention projection**：attention layer 少，fidelity 重要 → 保持 **BF16**
- **Latent projection（LatentMoE）**：保留 BF16
- **MTP layer**：在 network 末端，保持 BF16
- **最后 15% 的 network**：保持高精度，保 training stability

### Scaling law for quantization

A3B model 上 NVFP4 vs BF16 loss gap < 1%；A8B 上 gap < 0.6%。**Model 越大，量化越 robust**。这与 Chen et al. 2025 (https://arxiv.org/abs/2505.14302) 的 quantization scaling law 一致：大 model 的 redundancy 让量化 error 被 "吸收"。

### 硬件背景

Blackwell 上 FP4 GEMM throughput 是 FP8 的 3×。NVFP4 不是单纯 quantization trick，是 hardware-aware training format。NVIDIA 在推 "train in 4-bit" 作为主流。

---

## 5. Long Context：1M tokens

### 关键 trick：不用 RoPE

RoPE 是 long context extension 的主要痛点。RoPE 在训练长度外 extrapolation 不好，要 NTK scaling, YaRN 等技巧 patch。

Nemotron 3 直接不用 RoPE。Mamba 的 SSM 递归结构（$h_t = f(h_{t-1}, x_t)$）本身就带位置信息——同样 token 在不同 position 经过递归会产生不同 state，等于隐式 positional encoding。

这避免了 out-of-distribution RoPE 问题。SWAN (https://aclanthology.org/2025.emnlp-main.123/) 是 NVIDIA 之前探索无 RoPE 长上下文的工作。

### Training recipe

- CPT at 512k sequence length
- SFT at 256k
- RL stage 包含 long context environment up to 32k
- 不需要 staged increase from 8k to 512k，直接 CPT 到 512k

### MoE 比 dense 更好 extrapolate

Table 3 数据有意思：

| Model | 512k | 1M |
|-------|------|-----|
| Nemotron-2-Nano-12B (dense hybrid) | 75.12 | 23.43 |
| Nemotron-3-Nano-30B-A3B (MoE hybrid) | 66.02 | **54.19** |

两者都 train 到 512k，但 1M 上 dense 急剧 drop，MoE graceful degrade。

**直觉猜测**：MoE 的 sparse activation 让长 context 上不同 pattern 路由到不同 expert，reduce interference，比 dense 的 full activation 更 robust to distribution shift。这点值得深挖，paper 没解释。

### NLL 下降验证

Figure 6 显示在 >1M token 的 code data 上，cumulative average NLL 随 position 单调下降。说明模型真的在利用 long context 信息，越往后越容易预测（因为更多 context 可参考）。如果模型只 attend 最近 tokens，NLL 应该随 position 增长（因为越往后越偏离训练分布）。

---

## 6. Multi-environment RL

### Staged vs Simultaneous

之前 Nemotron 2 用 staged approach：先训 reasoning，再训 tool use，再训 chat。结果训完 stage 2 后 stage 1 能力 degrade。DeepSeek-R1 也有类似 issue。

Nemotron 3 改成 **同时训练所有 environment**：
- Math, code, agentic tool use, long context, chat 一起训
- 每个 batch 混合不同 environment 的 sample
- 不同 reward signal 互相 regularize

**为什么更好**：
- 避免 catastrophic forgetting（所有 capability 一直被训练）
- Less reward hacking（某个 environment 的 reward 模式不能 dominate）
- 多任务 regularize 效果

Figure 7 显示所有 benchmark 在 RL 训练中同步上升，没有跷跷板效应。

### GRPO 不用 critic

GRPO 的核心：**group baseline 替代 value function**。

对同一个 prompt，sample $G$ 个 response，计算 reward $r_1, ..., r_G$。Advantage 是 $\hat{A}_i = (r_i - \bar{r}) / \sigma(r)$，纯 group 内 normalization。

不用训 critic network（critic 在 reasoning reward sparse 场景难训）。KL penalty 到 reference policy 保持稳定。

### Asynchronous RL

- Rollout generation 在独立 GPU cluster
- Training 在另一个 cluster
- 用 MTP 做 speculative decoding 加速 rollout
- Hybrid Mamba 架构的 high throughput 在这里体现价值

NeMo-RL（训练）和 NeMo-Gym（environment）开源 Apache 2.0。

---

## 7. Reasoning Budget Control

### 机制

模型在 SFT/RL 阶段学会识别 `` 作为 thinking trace 结束 marker。

Inference 时：
1. 用户指定 budget $B$（max thinking tokens）
2. 模型开始 generate thinking trace
3. 生成到 $B$ 个 tokens，系统强制 append ``
4. 模型基于 partial thinking trace 生成 final answer

### 为什么 work

模型在训练时见过 "partial thinking → final answer" 的 pattern，所以能 graceful handle 提前结束。Figure 8 显示不同 budget 下的 accuracy curve，smooth trade-off。

### 实用价值

这把 reasoning cost 显式化。部署时可以：
- 高优先级 query：大 budget，max accuracy
- 廉价 batch 处理：小 budget，accept 较低 accuracy
- Cost-aware routing：根据 query 难度动态调 budget

---

## 整体 mental model

把所有技术点串起来：

```
[Architecture]
Mamba-2 + sparse attention + MoE
  → high throughput → 大规模 RL rollout 可行

[MoE innovation]
LatentMoE
  → 低维 routing + 多小 expert
  → 同 compute 更高 accuracy

[Training acceleration]
MTP
  → 训练 signal 更密 + 推理加速
  → RL rollout 快

[Numeric format]
NVFP4
  → 4-bit 训练，Blackwell 3× throughput
  → hardware-software co-design

[Long context]
No RoPE + Mamba implicit position
  → 1M tokens without distribution shift

[Post-training]
Multi-environment simultaneous RL
  → 避免 forgetting，避免 reward hacking

[Inference control]
Budget control via 
  → cost-aware deployment
```

每个技术都同时服务多个目标：
- Mamba 既省 throughput 又免 RoPE 问题
- MTP 既改 training signal 又加 inference speed
- LatentMoE 既保 compute budget 又增 accuracy

**核心 intuition：Nemotron 3 的所有设计都在 "重新分配 bottleneck"**——把 cost 从 attention 移到 Mamba，从 expert 宽度移到 expert 数量，从 16-bit 移到 4-bit，从串行 RL 移到并行 RL。每次 reallocation 都在不损失甚至提升 accuracy 的前提下打开新的 efficiency frontier。

如果你想 build 更深 intuition，我建议重点深挖 LatentMoE 和 NVFP4 recipe——前者是架构创新，后者是系统创新，两个点都最能体现 "重新分配 bottleneck" 的思想。

---

` token
- 模型基于 partial thinking trace 继续 generate response
- 提供了 accuracy-efficiency trade-off curve

### 额外联想：

1. **与 DeepSeek-V3 的对比**：
   - DeepSeek-V3 也用 MTP 和 GRPO
   - 但 Nemotron 3 用 hybrid Mamba-Transformer 架构，DeepSeek-V3 是纯 Transformer
   - LatentMoE 是 Nemotron 3 独有

2. **与 Nemotron Nano 2 的对比**：
   - Nano 2 是 dense hybrid
   - Nano 3 是 MoE hybrid
   - MoE hybrid 有更好的 length extrapolation

3. **与 SWAN (NVIDIA 2025) 的关系**：
   - SWAN 也是 long-context language modeling 的工作
   - 同样避免 RoPE

4. **LatentMoE 的理论联系**：
   - 类似于 bottleneck MLP 的思想
   - 类似于 LoRA 的 low-rank adaptation
   - 本质是 dimensionality reduction for routing

5. **与 Jamba (AI21) 的对比**：
   - Jamba 也是 hybrid Mamba-Transformer MoE
   - 但 Nemotron 3 用 Mamba-2 (SSD)，Jamba 用 Mamba-1
   - LatentMoE 是 Nemotron 3 独有

6. **NVFP4 的硬件背景**：
   - Blackwell architecture 引入 FP4 support
   - GB300: FP4 throughput is 3× FP8
   - 这是 hardware-software co-design 的例子

让我组织一个详细的 response，包含：
- 架构 overview
- 每个核心技术的深入分析
- 公式和实验数据
- 相关工作链接
- Intuition building

需要提供 web links for reference。让我列出相关链接：

- Mamba-2: https://arxiv.org/abs/2405.21060
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- MTP original: https://arxiv.org/abs/2408.10737 (Gloeckle et al.)
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- RULER: https://arxiv.org/abs/2404.06654
- SWAN: https://aclanthology.org/2025.emnlp-main.123/
- Jamba: https://arxiv.org/abs/2403.19887
- Nemotron Nano 2: https://arxiv.org/abs/2508.14444
- Nemotron-H: https://arxiv.org/abs/2504.03624
- NVFP4 pretraining: https://arxiv.org/abs/2509.25149
- Scaling law for QAT: https://arxiv.org/abs/2505.14302
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Empirical study of Mamba: https://arxiv.org/abs/2406.07887

让我写一个详细的、技术深入的 response。# NVIDIA Nemotron 3 深度解析

这篇 paper 是 NVIDIA 在 2025 年底放出的关于 Nemotron 3 系列（Nano, Super, Ultra）的技术白皮书。Nemotron 3 的核心定位是 **"efficient and open intelligence"**，主攻 agentic AI 场景。Nano 已经开源，Super 和 Ultra 后续发布。这篇 paper 虽然篇幅不长，但信息密度极高，每个技术点都有对应的 ablation data 支撑。

---

## 1. Hybrid Mamba-Transformer MoE 架构

### 1.1 为什么选 Mamba-2 而不是 Mamba-1 或纯 Transformer

Nemotron 3 的 backbone 主体是 **Mamba-2 + MoE** 交替堆叠，只在少数 layer 插入 self-attention。这个选择背后的 reasoning chain 非常清晰。

**Mamba-2 的 SSM 核心**：

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

变量含义：
- $h_t \in \mathbb{R}^{N \times d}$：第 $t$ 步的 hidden state（Mamba-2 中 $N$ 是 state dimension，可以远大于 Mamba-1）
- $x_t \in \mathbb{R}^{d}$：输入 token embedding
- $\bar{A} \in \mathbb{R}^{N \times N}$：离散化后的 state transition matrix（discretized from continuous $A$ via zero-order hold + bilinear transform）
- $\bar{B} \in \mathbb{R}^{N \times 1}$：discretized input projection
- $C \in \mathbb{R}^{1 \times N}$：output projection
- $y_t \in \mathbb{R}^{d}$：SSM 输出

**Mamba-2 vs Mamba-1 的关键区别**（Dao & Gu, 2024, https://arxiv.org/abs/2405.21060）：Mamba-2 引入了 **Structured State Space Duality (SSD)**，证明了 SSM 与 attention 存在数学对偶。这允许：
1. 使用 multi-head SSM 结构（类似 multi-head attention）
2. 通过 matrix multiplication 实现并行训练（而非 Mamba-1 的 selective scan）
3. 更大的 state dimension $N$（Mamba-2 可以到 128 或更高，Mamba-1 通常 16-64）
4. 利用 GPU tensor core 的 matmul units，训练效率更高

**为什么 hybrid 而非纯 Mamba**：
- Mamba-2 的 constant-size state 在长 context 上做 in-context retrieval 有信息瓶颈
- Attention 的 KV cache 提供 **exact** recall，但在 generation 时 cost 线性增长
- 所以 hybrid 用 **few attention layers for high-fidelity all-to-all routing** + **Mamba-2 for sequence modeling with constant cost**

### 1.2 Inference throughput 的 math

在 generation 阶段（autoregressive decoding, batch size=1）：
- Transformer layer 的 cost：$O(t \cdot d^2)$ compute + $O(t \cdot d)$ KV cache memory bandwidth
- Mamba-2 layer 的 cost：$O(N \cdot d)$ compute + $O(N \cdot d)$ state memory bandwidth（**与 $t$ 无关**）

当 sequence length $t$ 变长（reasoning workload 如 8k input / 16k output），KV cache 读写成为 dominant bottleneck。Mamba-2 把这个 cost 变成 constant，所以 throughput 显著提升。

Paper 中 Figure 2 给出的数据：**Nemotron 3 Nano 30B-A3B 相比 Qwen3-30B-A3B 达到 3.3× throughput**，sequence 越长 speedup 越明显。

### 1.3 Layer pattern

Nemotron 3 Nano 的 layer pattern（Figure 1）：大部分是 interleaved Mamba-2 + MoE，少数 self-attention layer。具体比例 paper 没给完整数字，但提到：
- Attention 层比例小
- 每个 attention 层用 GQA with only 2 KV heads

这种 sparse attention 设计是为了在保持 long-context recall 的同时最小化 KV cache cost。

参考：Walefe et al., 2024, https://arxiv.org/abs/2406.07887 是 NVIDIA 之前对 Mamba-based LM 的 empirical study，结论支持 hybrid 设计。

---

## 2. LatentMoE：核心创新点

这是这篇 paper 最有意思的技术贡献，idea 非常精巧。

### 2.1 MoE 的两个 bottleneck

Paper 把 MoE 部署场景分成两类：

**Latency-bound deployment**（batch size 几十到几百 tokens）：
- Bottleneck: memory bandwidth（读 expert weights）
- 每个 expert 是 $d \times m$ matrix（$d$ = hidden dim, $m$ = FFN intermediate dim）
- 计算量 $O(d \cdot m)$ 远小于读 $O(d \cdot m)$ weights 的 bandwidth cost
- 所以减 $d$ 或减 $m$ 都能减 bandwidth

**Throughput-bound deployment**（batch size 数千 tokens）：
- Bottleneck: all-to-all communication（dispatch tokens to experts + aggregate results）
- Communication volume: $O(K \cdot d)$（$K$ = top-k active experts）
- **与 $m$ 无关**！

### 2.2 关键观察

FFN 的 expressive power 主要由 **nonlinear budget $K \times m$** 控制（每个 token 经过 $K$ 个 expert，每个 expert 有 $m$ 个 nonlinear unit）。

所以设计自由度是：
- 减 $d$（latent dimension）→ 减 bandwidth + 减 communication
- 用节省的 capacity 增 $N$（总 expert 数）和增 $K$（active expert 数）
- 保持 $K \times m$ ≈ constant → 表达能力不降
- 更多 expert → 更细的 specialization → accuracy 提升

### 2.3 LatentMoE 公式

**Standard MoE**：
$$y = \sum_{i \in \text{top-}K} g_i(x) \cdot E_i(x)$$

其中 $E_i: \mathbb{R}^d \to \mathbb{R}^d$ 是第 $i$ 个 expert FFN，$g_i$ 是 gating network 输出。

**LatentMoE**：
$$z = W_{\text{down}} x \in \mathbb{R}^{\ell}$$
$$y = W_{\text{up}} \left( \sum_{i \in \text{top-}K'} g_i(z) \cdot E_i'(z) \right) \in \mathbb{R}^d$$

变量含义：
- $W_{\text{down}} \in \mathbb{R}^{\ell \times d}$：projection 从 hidden dim $d$ 到 latent dim $\ell$（$\ell < d$，典型 $d/\ell \approx 4$）
- $W_{\text{up}} \in \mathbb{R}^{d \times \ell}$：projection 回 $d$
- $E_i': \mathbb{R}^{\ell} \to \mathbb{R}^{\ell}$：latent space 的 expert，输入输出都是 $\ell$ 维
- $g_i$：gating network，**保持 $d$ 维输入**（paper 强调 non-routed computation 留在 $d$）

**Scaling 关系**：
- Expert 数：$N' = N \cdot (d/\ell)$（例如 $d=4096, \ell=1024$ → $N'=4N$）
- Active expert 数：$K' = K \cdot (d/\ell)$（同步缩放）
- 每个 expert 参数量：从 $d \times m$ 降到 $\ell \times m$，缩小 $(d/\ell)^2$ 倍
- Total routed params：$N' \cdot \ell \cdot m = N \cdot d/\ell \cdot \ell \cdot m = N \cdot d \cdot m$（**保持不变**）
- Nonlinear budget：$K' \cdot m = K \cdot d/\ell \cdot m$（增了 $d/\ell$ 倍！）— 这里我重新算一下

等等，让我重新看 paper：paper 说 "improves accuracy per byte while keeping overall inference cost approximately constant"。让我重新读：

> "By shifting routed expert computation and all-to-all traffic to the latent space, both per-expert weight loads and communication payloads are reduced by a factor of $d/\ell$ compared to a standard MoE. We use these parameter and bandwidth savings to increase both the total number of experts from $N$ to $N' = N \cdot d/\ell$ and the top-$K$ active experts per token from $K$ to $K' = K \cdot d/\ell$."

所以 expert $E_i'$ 在 latent space 计算，输入维度 $\ell$。每个 expert 的 FFN intermediate dim 保持 $m$？还是按比例减？

从 Table 1 看：
- Standard MoE: 8.09B active / 72.6B total, $d=4096$, 128 experts, $K=6$
- LatentMoE: 8.02B active / 72.8B total, $\ell=1024$, 512 experts, $K=22$

active params 几乎一样（8.09B vs 8.02B），total params 也几乎一样（72.6B vs 72.8B）。这说明：
- 每个 expert 的 FFN 中间维度 $m$ 大致保持不变（或微调）
- expert 在 $\ell$ 维输入 + $m$ 维 hidden + $\ell$ 维输出，参数量 $\approx 2 \ell m$
- Total routed: $N' \cdot 2\ell m = N \cdot d/\ell \cdot 2\ell m = N \cdot 2dm = $ standard MoE ✓
- Active routed: $K' \cdot 2\ell m = K \cdot d/\ell \cdot 2\ell m = K \cdot 2dm = $ standard MoE ✓

**所以 $K' \times m = K \cdot (d/\ell) \cdot m$，nonlinear budget 实际上增加了 $d/\ell$ 倍！** 但 active compute 保持不变？这有点矛盾……

让我再想：active params 一样 = active compute 一样（每个 token 经过 $K'$ 个 expert，每个 expert 有 $2\ell m$ params）。Standard 是 $K \cdot 2dm$，LatentMoE 是 $K' \cdot 2\ell m = K \cdot d/\ell \cdot 2\ell m = K \cdot 2dm$。**Compute 完全相同**！

但 nonlinear budget（activation 数量）从 $K \cdot m$ 增加到 $K' \cdot m = K \cdot (d/\ell) \cdot m$，增加 $d/\ell$ 倍。这就是 paper 说的 "increase the nonlinear budget and expert diversity"。

**直觉解释**：
- 同样的 compute budget，更多但更小的 expert → 每个 expert 更 specialized
- 同样的 bandwidth budget（all-to-all 通信量 $K \cdot d$ → $K' \cdot \ell = K \cdot d$，相同），但路由更细粒度
- Nonlinear budget 增加但每 nonlinear unit 的维度减小 → 表达能力来自 expert diversity 而非单个 expert 的 width

### 2.4 实验数据

Table 1：1T tokens 训练，相同 hyperparameter，相同 active/total params：

| Metric | Standard MoE | LatentMoE | Δ |
|--------|-------------|-----------|---|
| MMLU-Pro | 48.30 | 52.87 | +4.57 |
| MMLU | 70.10 | 72.11 | +2.01 |
| Code | 51.95 | 55.14 | +3.19 |
| Math | 78.32 | 80.19 | +1.87 |
| Commonsense | 81.73 | 82.10 | +0.37 |

提升幅度不小，且在所有任务上一致。MMLU-Pro 这种 hard benchmark 上 +4.57 非常显著。

### 2.5 联想与相关工作

- **类似 LoRA 的低秩思想**：但 LoRA 是低秩 adapter，LatentMoE 是低秩 routing space
- **类似 bottleneck MLP**：把 MLP 拆成 down-proj + nonlinear + up-proj
- **与 Mixtral / DeepSeek-MoE 的对比**：DeepSeek-MoE 用 fine-grained expert segmentation（把大 expert 拆成小 expert），LatentMoE 是 "fine-grained + low-dim routing"，更进一步
- **可能的理论联系**：Routing 在 latent space 等价于在低维 manifold 上做 expert selection，对 expert specialization 更友好

---

## 3. Multi-Token Prediction (MTP)

### 3.1 MTP 的原始公式（Gloeckle et al., 2024）

Reference: https://arxiv.org/abs/2408.10737

给定 context $x_{1:t}$，预测未来 $k$ 个 tokens $x_{t+1:t+k}$。核心 loss：

$$\mathcal{L}_{\text{MTP}} = \sum_{j=1}^{k} \mathcal{L}_{\text{CE}}(f_j(h_t, x_{t+1:t+j-1}), x_{t+j})$$

变量含义：
- $f_j$：第 $j$ 个 prediction head（参数独立）
- $h_t$：第 $t$ 个 token 的 hidden state
- $x_{t+1:t+j-1}$：前面预测的真实 tokens（teacher forcing）
- $\mathcal{L}_{\text{CE}}$：cross-entropy loss

**直觉**：让模型在每一步预测多个 future token，强迫模型学到 "plan ahead" 的能力，而不是只看 next token。

### 3.2 DeepSeek-V3 的实现（Nemotron 3 采用的版本）

Reference: https://arxiv.org/abs/2412.19437

DeepSeek-V3 的 MTP 是 sequential 结构：
- Head 1 预测 $x_{t+1}$（输入 $h_t$）
- Head 2 预测 $x_{t+2}$（输入 $h_t$ + $x_{t+1}$ 的 embedding）
- Head $k$ 预测 $x_{t+k}$

每个 head 是个 small transformer block + output projection。

**Nemotron 3 的 ablation 数据**（Table 2，8B active MoE，1T tokens）：

| Task | Baseline | + MTP | Δ |
|------|---------|-------|---|
| MMLU | 70.06 | 71.26 | +1.20 |
| MMLU-Pro | 45.05 | 47.84 | +2.79 |
| MBPP-Sanitized | 65.58 | 66.89 | +1.31 |
| ARC-Challenge | 86.43 | 88.05 | +1.62 |
| GSM8K | 82.49 | 84.46 | +1.97 |

平均 +2.4%。

### 3.3 Speculative decoding 集成

MTP 的 inference 价值：head 1, 2, ..., k 的输出直接作为 speculative decoding 的 draft tokens。

- 标准 speculative decoding 需要 separate draft model
- MTP 把 draft model 嵌入主模型，no extra memory
- Paper 报告 **97% acceptance rate on first two predicted tokens**

Speculative decoding 原理（Leviathan et al., 2023, https://arxiv.org/abs/2211.17192）：
1. Draft model 生成 $k$ 个 tokens $x_{t+1:t+k}^{\text{draft}}$
2. Main model 并行验证：计算 $P(x_{t+1:t+k} | x_{1:t})$
3. 接受 token if draft 的 logit 与 main 一致（rejection sampling）
4. MTP head 与 main model 共享 trunk，draft 与 main 高度一致，acceptance rate 高

---

## 4. NVFP4 Training：4-bit 浮点预训练

这是 NVIDIA 在 Blackwell 架构上推的 4-bit 训练格式。Reference: https://arxiv.org/abs/2509.25149

### 4.1 NVFP4 格式细节

NVFP4 是 **hierarchical scaling** 的 FP4：

1. **Element format: E2M1**
   - 2 bit exponent + 1 bit mantissa + 1 bit sign
   - 可表示值：$\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}$
   - 共 16 个值（包括 sign）

2. **Micro-block scaling (16 elements per block)**
   - 每 16 个 element 共享一个 block scale factor
   - Block scale format: E4M3 (8-bit float, 4 bit exponent + 3 bit mantissa)
   - Value = block_scale × E2M1_element

3. **Global scale (second level)**
   - FP32 全局 scale
   - Value = global_scale × block_scale × E2M1_element

公式表示：
$$\hat{x}_{ij} = S_{\text{global}} \cdot S_{\text{block}, i} \cdot x_{ij}^{\text{E2M1}}$$

其中 $x_{ij}^{\text{E2M1}} \in \{0, \pm 0.5, ..., \pm 6\}$, $S_{\text{block}, i}$ 是第 $i$ 个 block 的 E4M3 scale, $S_{\text{global}}$ 是 FP32 scale。

### 4.2 Recipe 关键技术

**2D block scaling for weights**：weights 在 row + column 两个维度都做 block scaling，capture 不同维度的 outlier distribution。

**Random Hadamard Transform (RHT) on inputs to wgrad**：
$$\tilde{X} = X H / \sqrt{d}$$

其中 $H$ 是 Hadamard matrix（正交矩阵），$d$ 是 input dim。

**RHT 的作用**：rotation 让 outlier 平均分散到所有维度，避免某些维度因 outlier 而 quantization error 过大。等价于在频域做 quantization，Hadamard 是 fast Walsh-Hadamard transform（$O(d \log d)$）。

**Stochastic rounding on gradients**：
$$\text{round}_{\text{stoch}}(x) = \lfloor x \rfloor + \text{Bernoulli}(x - \lfloor x \rfloor)$$

避免 systematic bias 在低位 quantization 下的累积。

### 4.3 敏感 layer 处理

Paper 的关键 finding（Figure 4）：
- **Mamba output projection** 在 NVFP4 下有 40% flushes to zero（在 Nano 上）→ 用 **MXFP8** 替代
- **QKV 和 Attention projections** 保持 **BF16**（attention layer 少，fidelity 重要）
- **最后 15% 的网络**保持高精度（stability）
- Super/Ultra 的 LatentMoE projection 和 MTP layer 也保持 BF16

**Recipe 效果**：
- A3B: <1% relative loss gap (NVFP4 vs BF16)
- A8B: <0.6% relative loss gap
- 支持 scaling law for quantization（Chen et al., 2025, https://arxiv.org/abs/2505.14302）：**model 越大，量化 loss gap 越小**

### 4.4 硬件背景

GB300 (Blackwell Ultra) 的 FP4 throughput 是 FP8 的 3×（NVIDIA Blackwell Ultra Datasheet, https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet）。

这是 hardware-software co-design 的典型例子：4-bit format 在 Blackwell 上有 native GEMM 支持，quantization 不再是 inference-only optimization，而是 pretraining 时就用。

---

## 5. Long Context：1M tokens 支持

### 5.1 关键设计：不用 RoPE

RoPE 是 long context extension 的主要 hurdle，因为：
- RoPE 的 rotation frequency 在训练长度外 extrapolate 不好
- 需要 NTK-aware scaling, YaRN 等技巧

**Nemotron 3 的做法**：attention layer 不用 RoPE。Mamba layer 通过 SSM 的递归结构提供 **隐式 positional information**。

这避免了 out-of-distribution RoPE issue。Reference: SWAN paper (Puvvada et al., 2025, https://aclanthology.org/2025.emnlp-main.123/) 是 NVIDIA 之前关于无 RoPE 长上下文的工作。

### 5.2 Training recipe

- **CPT (Continued Pre-Training)** at 512k sequence length
- **SFT** at 256k sequence length
- **RL stage** 包含 long-context environment（up to 32k tokens）
- **不需要 staged increase** from 8k to 512k（直接 CPT 到 512k 就 work）

Synthetic data 包括：long-range retrieval, multi-hop reasoning, multi-document aggregation。

### 5.3 MoE vs Dense 在 length extrapolation 上

Table 3：RULER scores（Hsieh et al., 2024, https://arxiv.org/abs/2404.06654）

| Model | 128k | 256k | 512k | 1M |
|-------|------|------|------|-----|
| Nemotron-Nano-12B-v2-Base (dense) | 85.13 | 79.85 | 75.12 | 23.43 |
| Nemotron-3-Nano-30B-A3B-Base (MoE) | 74.48 | 71.67 | 66.02 | **54.19** |

**关键观察**：
- Dense model 在 512k→1M 急剧 drop（75→23）
- MoE model **graceful degradation**（66→54）
- 两者都 train 到 512k，但 MoE extrapolation 更好

**直觉**：MoE 的 sparse activation 可能让 long context 上的不同 information pattern 被路由到不同 expert，reduce interference，改善 extrapolation。这点非常值得深究。

### 5.4 NLL 分析

Figure 6 显示 Nemotron 3 Nano 在 >1M token 的 code data 上，cumulative average NLL（negative log-likelihood）随 token position 单调下降。说明模型**真的在用** long context 信息，而非只 attend 最近的 tokens。

---

## 6. Multi-environment RL Post-training

### 6.1 GRPO 算法

Nemotron 3 用 GRPO（Group Relative Policy Optimization, Shao et al., 2024, https://arxiv.org/abs/2402.03300）。

GRPO 的 loss（简化版）：

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i\right)\right] - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

变量含义：
- $\rho_i = \pi_\theta(a_i|s_i) / \pi_{\text{old}}(a_i|s_i)$：importance ratio
- $\hat{A}_i = (r_i - \bar{r}) / \sigma(r)$：group-normalized advantage（**不需要 critic**！）
- $r_i$：第 $i$ 个 response 的 reward
- $\bar{r}, \sigma(r)$：group 内的 mean 和 std of reward
- $\epsilon$：clip range
- $\beta$：KL penalty 系数
- $\pi_{\text{ref}}$：reference policy（通常 SFT model）

**为什么不用 critic**：group baseline 替代 value function，减 critic 训练 cost，特别适合 reasoning 这种 reward sparse 的场景。

### 6.2 Masked importance sampling

Paper 提到 "masked importance sampling to account for discrepancies between the training and rollout policies"。

**直觉**：在 asynchronous RL 中，rollout policy $\pi_{\text{rollout}}$ 与 training policy $\pi_\theta$ 有 lag。Standard PPO 的 importance ratio $\rho = \pi_\theta / \pi_{\text{rollout}}$ 假设两者接近。当 lag 大时，gradient noise 大。Masked importance sampling 是对 ratio 异常大的 sample 做 mask（或 down-weight），提升 stability。

### 6.3 Multi-environment 同时训练

Paper 的核心 claim：所有 RL environments 同时训练，而非 staged。

之前 NVIDIA Nemotron 2 用 staged approach（先训 reasoning，再训 tool use，etc.），导致：
- 训完 reasoning 后训 tool use，reasoning 能力 degrade
- DeepSeek-R1 也有类似 issue

**同时训练的好处**：
- 避免 catastrophic forgetting
- 不同 environment 的 reward signal 互相 regularize
- Less reward hacking（不同 environment 的 reward pattern 互相约束）

Figure 7 显示 training step 增加，所有 benchmark（math, code, agentic, etc.）accuracy 都稳步上升。

### 6.4 异步 RL 架构

- **Decouple training from inference**：rollout generation 在独立 GPU cluster，training 在另一个
- **MTP 加速 rollout**：用 MTP head 做 speculative decoding，rollout generation 快 2-3×
- **High inference throughput** 是关键优势（hybrid Mamba 架构的好处）

Reference: NeMo-RL 和 NeMo-Gym 是开源 RL stack。

---

## 7. Reasoning Budget Control

### 7.1 训练机制

模型在 SFT/RL 阶段被训练识别 `` token 作为 thinking trace 结束的 marker。

**Inference time**：
1. 用户指定 max thinking tokens $B$
2. 模型开始 generate thinking trace
3. 当生成到 $B$ 个 tokens，强制 append `` token
4. 模型基于 partial thinking trace 生成 final response

### 7.2 Accuracy-efficiency trade-off

Figure 8 展示 trade-off curve。这给 user 粒度控制：
- Cost-sensitive 部署：低 budget，accept 较低 accuracy
- High-stakes 推理：高 budget，max accuracy

这种 "thinking budget" 的设计非常实用，把 reasoning cost 显式化。

---

## 8. 整体技术 picture

### 8.1 各组件协同

Nemotron 3 的技术不是孤立堆叠，而是互相强化：

- **Mamba-2 + sparse attention** → high inference throughput → 使大规模 RL rollout 可行
- **MoE** → sparse parameter scaling → 同 compute 下更高 accuracy
- **LatentMoE** → 在 MoE 基础上提升 expert diversity，不改 compute budget
- **MTP** → 加速 inference + 加速 RL rollout（speculative decoding）
- **NVFP4** → 4-bit 训练，Blackwell 上 3× throughput，进一步加速训练
- **No RoPE** → 避免 long context extension 的 distribution shift
- **Multi-environment RL** → 统一 post-training，避免 forgetting
- **Budget control** → inference 时的 cost knob

### 8.2 与同期工作的对比

**vs DeepSeek-V3** (https://arxiv.org/abs/2412.19437)：
- DeepSeek-V3 是纯 Transformer MoE + MTP + GRPO
- Nemotron 3 是 hybrid Mamba-Transformer MoE + LatentMoE + MTP + GRPO + NVFP4
- LatentMoE 和 NVFP4 是 Nemotron 3 独有
- DeepSeek-V3 用 RoPE + context extension，Nemotron 3 直接不用 RoPE

**vs Jamba** (AI21, https://arxiv.org/abs/2403.19887)：
- Jamba 也是 hybrid Mamba-Transformer MoE
- 但用 Mamba-1，Nemotron 3 用 Mamba-2 (SSD)
- Jamba 无 LatentMoE，无 MTP

**vs Nemotron Nano 2** (https://arxiv.org/abs/2508.14444)：
- Nano 2 是 dense hybrid Mamba-Transformer
- Nano 3 是 MoE hybrid
- Nano 3 在 length extrapolation 上显著优于 Nano 2

**vs Nemotron-H** (https://arxiv.org/abs/2504.03624)：
- Nemotron-H 也是 hybrid Mamba-Transformer，主要是 dense model
- Nemotron 3 在此基础上加 MoE + LatentMoE + NVFP4

### 8.3 关键 research questions 留下

1. **LatentMoE 的 $\ell/d$ 比例 sweet spot**：paper 用 1/4，更小（如 1/8）会怎样？是否有 scaling law？
2. **MoE 为何 length extrapolation 更好**：paper 给了 empirical 数据但无理论解释
3. **NVFP4 在更大 model（Super/Ultra）上的 gap**：scaling law 预测 gap 继续减小，但实际数据待验证
4. **Multi-environment RL 的 reward conflict**：paper 说同时训练 less reward hacking，但具体如何避免 negative transfer？

---

## 9. 总结与直觉构建

Nemotron 3 的核心 thesis：**efficiency is the new accuracy**。通过：
- **架构层面**：Mamba-2 把 sequence modeling cost 变 constant，MoE 把 parameter scaling 变 sparse
- **数值层面**：NVFP4 把 4-bit 训练 mainstream 化
- **训练层面**：MTP 加速 inference + RL rollout
- **数据层面**：multi-environment RL 避免 capability forgetting
- **部署层面**：budget control 把 reasoning cost 显式化

LatentMoE 是最有创新性的部分：**通过 dimensionality reduction 重新分配 MoE 的 capacity**，在 compute 和 bandwidth 不变的情况下增加 expert diversity，这是 "reparameterize the MoE bottleneck" 的巧妙思路。本质上类似 channel-wise bottleneck 思想，但用于 expert routing 而非 token mixing。

如果你想 build 更深 intuition，建议从以下几个方向深挖：
1. **SSD (Structured State Space Duality)** 的数学：理解 Mamba-2 与 attention 的对偶
2. **LatentMoE 的 ablation**：自己复现 small scale，验证 nonlinear budget vs expert diversity 的 trade-off
3. **NVFP4 的 RHT**：理解为什么 Hadamard rotation 对 quantization 有帮助
4. **GRPO without critic**：理解 group baseline 在 reasoning reward 上的优势

Reference links:
- Mamba-2: https://arxiv.org/abs/2405.21060
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- MTP original: https://arxiv.org/abs/2408.10737
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- RULER: https://arxiv.org/abs/2404.06654
- SWAN: https://aclanthology.org/2025.emnlp-main.123/
- Jamba: https://arxiv.org/abs/2403.19887
- Nemotron Nano 2: https://arxiv.org/abs/2508.14444
- Nemotron-H: https://arxiv.org/abs/2504.03624
- NVFP4 pretraining: https://arxiv.org/abs/2509.25149
- Scaling law for QAT: https://arxiv.org/abs/2505.14302
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Empirical Mamba LM study: https://arxiv.org/abs/2406.07887
- Blackwell Ultra datasheet: https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet
