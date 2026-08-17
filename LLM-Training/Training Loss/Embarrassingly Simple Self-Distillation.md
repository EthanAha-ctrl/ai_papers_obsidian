---
source_pdf: Embarrassingly Simple Self-Distillation.pdf
paper_sha256: 96536a682ffd8f35632f2b3d775c649b67455c0f8f2a3e4218249e17bbfeab32
processed_at: '2026-08-04T03:19:35-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 SSD

---

## 一句话版

**模型自己采样自己，然后 SFT，就完事了。** 没有 verifier，没有 teacher model，没有 RL，没有 reward。Qwen3-30B 从 42.4% 涨到 55.3% pass@1。

---

## 方法有多荒谬地简单

你有一个 frozen model $p_\theta$。你拿一堆 competitive programming 的题（~10K 道，来自 rSTARcoder），让 model 在 $T_{\text{train}}=1.6$ 下每题采样**一条** solution。采出来的东西不管对错，不做任何 filtering，直接当训练数据，跑标准 cross-entropy SFT。完了。

$$y \sim \text{Decode}_{T_{\text{train}}, \rho_{\text{train}}}[p_\theta(\cdot \mid x)]$$

$$\mathcal{L}(\theta) = -\mathbb{E}_{(x,y)} \sum_t \log p_\theta(y_t \mid x, y_{<t})$$

变量解释：$x$ 是题目，$y$ 是 model 自己采的 solution，$T_{\text{train}}$ 是采样温度，$\rho_{\text{train}}$ 是 top-k/top-p 截断配置，$y_t$ 是第 $t$ 个 token，$y_{<t}$ 是前面的 prefix。

推理时用另一个温度 $T_{\text{eval}}=0.9$。就这样。整个 pipeline 三行代码。

参考：https://github.com/apple/ml-ssd

---

## 核心矛盾：代码里有两种 token，它们要的东西打架

这是整篇 paper 的 intuition 核心。

### Lock 位置：必须精确

你写 `if n == ` 之后，下一个 token 必须是某个特定数字。model 知道是哪个，但 vocab 里还有一堆 syntactically plausible 的 distractor（其他数字、运算符）挂在那儿，携带非零概率。这种位置我叫它 **lock** —— 分布 sharply peaked，但有一条 diffuse 的 distractor tail。

### Fork 位置：必须探索

函数体第一行，你可以写 `for i in range(...)`，可以写 `def helper():` 递归，可以写 `from collections import defaultdict` 然后建图。每个选择通向**完全不同的解法**（quicksort vs mergesort vs built-in sort）。分布真正 spread 在多个 head token 上。这种位置我叫它 **fork**。

### 打架的地方

推理温度 $T_{\text{eval}}$ 对整个分布做 $p(v)^{1/T}$ 的 power transform：

- 降 $T_{\text{eval}}$ → 放大 peak 之间的 gap → lock 安全了（distractor 被压死）→ 但 fork 的 head 被压成 singleton，探索饿死
- 升 $T_{\text{eval}}$ → flatten head → fork 能探索了 → 但 lock 的 distractor tail 回潮，一个错误 token 整条 trajectory 崩

**你不可能用一个全局 temperature 同时满足两者。** 这就是 precision-exploration conflict。

类比：这就像开车，lock 是看路标（必须精确读数），fork 是路口选路（可以探索）。你只有一根油门踏板，同时控制刹车和转向，必然 compromise。

---

## 为什么 naive self-training 是 fixed point

如果 $T_{\text{train}}=1$ 且不做 truncation，你自己采自己训，期望梯度恒等于零：

$$\mathbb{E}_{v \sim p_\theta(\cdot|s)}[\nabla_\theta \log p_\theta(v|s)] = \nabla_\theta \sum_v p_\theta(v|s) = \nabla_\theta 1 = 0$$

变量：$s$ 是 context，$v$ 是 next token。score function gradient 在自身分布下取期望，因为概率归一化 telescopes 到 1 的梯度。**你自己采自己训，什么也学不到。**

SSD 之所以能 break 这个 fixed point，全靠 **temperature + truncation 扭曲了 target distribution**。signal 不来自外部，来自 decoding rule 本身。

---

## SSD 做了什么：把 decoding distortion 蒸馏进 weights

### 三项分解（公式 4/15，paper 最 elegant 的部分）

在单个 context $s$ 上，SSD loss 可以写成：

$$\mathcal{L}_s(\theta) = \underbrace{-\log \text{KeptMass}_\theta(s)}_{\text{Term 1: support compression}} + \underbrace{(1-T) H_{1/T}(p_\theta(\cdot|s, S_s))}_{\text{Term 2: within-support reshaping}} + \underbrace{T \cdot \text{KL}(q_s \| p_{\theta,T}(\cdot|s,S_s))}_{\text{Term 3: alignment}} + \text{const}$$

变量逐一解释：
- $\theta$：student 参数（被优化的）
- $s$：当前 decoding context
- $S_s$：**retained support** —— teacher 经过 temperature + top-k + top-p 后存活的 token 集合
- $T = T_{\text{train}}$：训练温度
- $\text{KeptMass}_\theta(s) = \sum_{v \in S_s} p_\theta(v|s)$：student 在 retained support 上的总概率质量
- $p_\theta(\cdot|s, S_s)$：student 在 $S_s$ 上的条件分布
- $H_{1/T}$：Rényi entropy of order $1/T$，定义为 $H_\alpha(\pi) = \frac{1}{1-\alpha}\log\sum_v \pi(v)^\alpha$
- $q_s$：teacher 的 truncated-tempered target
- $p_{\theta,T}(\cdot|S_s)$：student 在 $S_s$ 上 tempered 后的分布

### Term 1：support compression（gate term）

$$-\log \text{KeptMass}_\theta(s)$$

最大化 student 在 $S_s$ 上的质量。因为 $q_s$ 在 $S_s$ 外严格为 0，最优解要求 outside logits $\to -\infty$。**tail suppression 是持续压力，训练永远不满足。**

logit-level gradient 对 $v \notin S_s$ 直接给出 $+p_\theta(v|s)$ —— 明确把这些 logit 往下推。

### Term 2：within-support reshaping

$$(1-T) H_{1/T}(p_\theta(\cdot|s, S_s))$$

- $T > 1$（SSD 典型 setting）：$(1-T) < 0$，最小化 loss = 最大化 $H_{1/T}$ → 在 retained set 内**平滑**分布
- $T = 1$：此项消失，回到 fixed point

Rényi entropy $H_{1/T}$ 的 order 是 $1/T$。$T>1$ 时 order $<1$，落在 sub-Shannon regime —— 对 diffuse tail 更敏感、对集中 peak 更宽容。它会在 fork 处把多个 viable continuation 摊平，又不会过度鼓励 lock 处的低质量 distractor。

### Term 3：KL anchor

$$T \cdot \text{KL}(q_s \| p_{\theta,T}(\cdot|s,S_s))$$

防止 student 漂离 teacher 的相对偏好。smoothing 只发生在 teacher 已认为值得保留的 token 之间，不会无中生有。

### 关键：同一公式在 lock 和 fork 上表现不同

**Lock 处**：分布 sharply peaked，truncation 后 $|S_s|$ 很小（1-2 个 token）。Rényi 项 $H_{1/T} \leq \log|S_s|$ 几乎没发挥空间。Term 1 主导 —— 把 distractor tail 压死。Lock 变成 sharp spike，对 $T_{\text{eval}}$ 几乎免疫。

**Fork 处**：分布 spread，truncation 后 $|S_s|$ 较大（4-5 个 viable continuation）。Term 1 仍活跃但 head 已含大部分有用质量。Term 2 有发挥空间 —— 把几个 viable alternative 摊平成 plateau，同时**不会** reopen 已被 truncation 切掉的 tail。Fork 变成 broad plateau，$T_{\text{eval}}$ 在其上有真正 leverage。

**这就是 SSD break precision-exploration conflict 的方式：它 context-adaptively 在 lock 处压 tail、在 fork 处平 head。全局一个 objective，局部两种行为，靠 $S_s$ 的几何形状自动切换。**

---

## 为什么 decode-only tuning 追不上 SSD

这是 paper 最 elegant 的 impossibility result。

### Normal form（Proposition B.5）

任何 fixed ordering 的 temperature + top-k + top-p 操作，最终都 collapse 成：

$$\mu_s^\sigma((i)) = \frac{p_{(i)}(s)^\alpha \cdot \mathbf{1}[i \leq m_s^\sigma]}{\sum_{j=1}^{m_s^\sigma} p_{(j)}(s)^\alpha}$$

变量：$p_{(1)} \geq p_{(2)} \geq \cdots$ 是 frozen model 的 ranking，$\alpha = 1/T_{\text{eval}}$，$m_s^\sigma$ 是 prefix 长度（取决于操作顺序 $\sigma$）。

**无论你怎么 reorder temperature/top-k/top-p，最终 decoder 都是对 frozen ranking 的某个 prefix 做单一 power transform。** reorder 只能移动 prefix boundary，不能改变 transformation 的形式。

### 两个 rigidity

**Prefix rigidity（Corollary B.6）**：要保留 rank-$r$ 的 token，必须同时保留所有 rank $< r$ 的 token，哪怕它们是 distractor。

**Power rigidity（Corollary B.7）**：所有 surviving pair 的 log-odds 被同一个 global factor $\alpha$ 缩放：

$$\log \frac{\mu((i))}{\mu((j))} = \alpha \log \frac{p_{(i)}}{p_{(j)}}$$

你**不能**在 fork 处 flatten head 的同时在 lock 处 sharpen peak —— 同一个 $\alpha$ 同时作用两者。

### SSD 拥有的额外自由度

SSD 把 $p_0(\cdot|s) \to p_\theta(\cdot|s)$，**改变了 decoder 看到的 cumulative curve**：

$$S_{s,m}(\tau, k; p_0) \longrightarrow S_{s,m}(\tau, k; p_\theta)$$

decode-only reordering 无法触碰这条 curve。SSD 在 lock 处和 fork 处**分别**移动这条 curve（lock 处压缩 tail，fork 处清理 head）。这是 SSD 能 break conflict 的根本原因。

**直觉**：decode-only tuning 就像你只有一根油门踏板，同时控制刹车和转向。SSD 相当于改装了车本身 —— 让刹车自动生效（lock 安全），你就可以放心用油门去转向（fork 探索）。

---

## Bad Data 实验：最反直觉的 stress test

这个实验直接挑战 "SSD 靠训练在 correct code 上" 的解释。

设置：$T_{\text{train}}=2.0$，**完全关闭 truncation**（top-k = vocab size, top-p = 1.0）。

结果：
- ~62% 输出**根本无法 extract code**
- 看似 coherent 的输出 mid-sequence 退化成 multilingual gibberish（意大利语、俄语、阿拉伯语混杂）
- training loss 飙到 11.29

但 fine-tuned model 仍达到：

| Metric | Base | +SSD (bad data) | Δ |
|---|---|---|---|
| pass@1 | 42.4% | 48.1% | +5.7pp |
| pass@5 | 53.5% | 64.0% | +10.5pp |
| Hard pass@1 | 18.3% | 25.6% | +7.3pp |
| Hard pass@5 | 31.1% | 44.9% | +13.8pp |

**这证明了什么**：SSD 的有用 signal 主要来自 **distributional reshaping**，来自 token-level distribution shape 里的 structure，来自 Rényi shaping term 对 viable alternatives 的摊平。correctness 不是 signal 来源。即使 training data 几乎是 gibberish，只要 sampling 引入了非平凡的温度扭曲，reshaping 仍提供 signal。

Evaluation-time truncation（top-k=20, top-p=0.95）负责在 inference 时把训练时未被压制的 distractor tail 清理掉。

---

## 实验数据表（Table 2，LCB v6）

| Model | Base pass@1 | +SSD pass@1 | Δ | Hard pass@1 Δ | Hard pass@5 Δ |
|---|---|---|---|---|---|
| Qwen3-30B-Instruct | 42.4 | 55.3 | **+12.9** | +15.3 | +23.0 |
| Qwen3-4B-Instruct | 34.0 | 41.5 | +7.5 | +5.7 | +17.6 |
| Qwen3-4B-Thinking | 54.5 | 57.8 | +3.3 | +4.1 | +7.3 |
| Qwen3-30B-Thinking | 66.1 | 68.2 | +2.1 | +5.2 | +6.1 |
| Llama-3.1-8B-Instruct | 12.7 | 16.2 | +3.5 | +1.6 | +2.5 |

两个关键 pattern：

1. **越难增益越大**：30B-Instruct 在 easy 只 +6.5pp，hard 却 +15.3pp。排除了 "SSD 只是把简单题刷更稳" 的解释。
2. **pass@5 增益 > pass@1 增益**：30B-Instruct hard 上 pass@1 +15.3pp 但 pass@5 +23.0pp。SSD **没有** collapse 多样性，反而扩展了 coverage。单纯 sharpening 模式会压低 pass@5，SSD 反而提升了它。

---

## Effective Temperature Composition

在 local ideal-fit 近似下（$p_\theta(\cdot|s) = q_s$），student 在 evaluation 时表现为：

$$q_{s,\tau}(v) = \frac{\mathbf{1}\{v \in S_s\} \cdot p_0(v|s)^{1/(T_{\text{train}} \cdot \tau)}}{\sum_{u \in S_s} p_0(u|s)^{1/(T_{\text{train}} \cdot \tau)}}$$

student 在 retained set 内表现为 teacher 在 **product temperature** $T_{\text{eff}} = T_{\text{train}} \cdot T_{\text{eval}}$ 下的行为。

证明极简（Lemma B.1）：
$$\text{Temper}_{T_2}[\text{Temper}_{T_1}[p]](v) \propto (p(v)^{1/T_1})^{1/T_2} = p(v)^{1/(T_1 T_2)}$$

renormalization 常数 cancel。这是 power law 的结合律。

Figure 3 的实验验证：无 truncation 时 performance 几乎只由 $T_{\text{eff}}$ 决定（$R^2=0.75$），quadratic peak 在 $T_{\text{eff}} \approx 1.2$。

---

## Entropy 悖论的化解

SSD 后 model 的 total entropy 下降（更集中），但 pass@5 反而上升（更 explorable）。看似矛盾，公式 30 分解清楚：

$$H(p_\theta(\cdot|s)) = \underbrace{h_2(\text{KeptMass}_\theta(s))}_{\text{gate entropy}} + \underbrace{\text{KeptMass}_\theta(s) \cdot H(p_\theta(\cdot|s,S_s))}_{\text{head entropy}} + \underbrace{(1-\text{KeptMass}_\theta(s)) \cdot H(u_\theta(\cdot|s))}_{\text{tail entropy}}$$

变量：$h_2(\pi) = -\pi\log\pi - (1-\pi)\log(1-\pi)$ 是 binary entropy，$u_\theta$ 是 student 在 $S_s$ 补集上的条件分布。

**Total entropy 下降的来源**：gate entropy（$\text{KeptMass} \to 1$ 时 $h_2 \to 0$）+ tail entropy（$1-\text{KeptMass} \to 0$ 时整体 $\to 0$）。这两项作用在整个 vocabulary 上，幅度大。

**Head entropy 可以局部上升**：在 fork-like context，$T_{\text{train}} > 1$ 让 retained head 更平坦 → $H(p_\theta(\cdot|s,S_s))$ 上升。但被 $|S_s|$ 上界 bound，幅度小。

**Evaluation-time temperature 只作用于 head entropy**（公式 31）：

$$\frac{d}{d\tau} H(\text{Temper}_\tau^{S_s}[p_\theta(\cdot|s,S_s)]) = \frac{\text{Var}_{\text{Temper}_\tau^{S_s}}[\log p_\theta(v|s,S_s)]}{\tau^3} \geq 0$$

如果 head 是 singleton 或 nearly uniform，variance 接近 0，temperature 几乎无效。如果 head 是几个非等同 token，variance 大，temperature 强力调整 operational policy。

**直觉**：SSD 去掉了 "wrong kind of uncertainty"（diffuse tail），保留了 "right kind of uncertainty"（fork head 的 viable alternatives）。total entropy 和 useful exploration 是两个不同对象。

---

## 给你的直觉总结

### 1. decoding rule 是免费的 signal source

你在 makemore / nanoGPT lecture 里讲 temperature、top-k、top-p 是 inference-time knob。这篇 paper 给出反直觉结论：**inference-time knob 的能力有结构上限**（Proposition B.5 的 normal form + prefix/power rigidity）。要突破上限，必须把 decoding rule "bake into" model 本身。

SSD 本质上是 **把 decoding-time 的 temperature-truncation distortion 蒸馏进 weights**。model 不仅生成 code，还把自己的 inference procedure 编译进自己的 weights。这是 Software 2.0 的 self-referential 闭环。

### 2. signal 在 distribution shape 里，不在 correctness 里

Bad data 实验是最有力的证据。$T_{\text{train}}=2.0$ 无 truncation 采样，62% 输出无法 extract code，剩下的大部分 mid-sequence 退化成 gibberish。但 fine-tuned model 仍 +5.7pp pass@1。

这暗示 post-training 的 "signal" 不一定要来自外部（human label / verifier / reward model）。**decoding rule 本身的 procedural distortion 就是 signal**。它是免费的、无限的。

### 3. SSD vs RL

DeepSeek-R1 用 GRPO 在 verifiable reward 上做 RL，达到 reasoning SOTA。但 GRPO 训练不稳定，entropy collapse 是已知 failure mode（Cui et al. 2025, https://arxiv.org/abs/2505.22617）。

SSD 用 supervised learning 内化 decoding distortion，完全绕开 RL 的 instability。但 SSD 的 ceiling 比 RL 低 —— thinking model 增益比 instruct model 小（30B-Thinking +2.1pp vs 30B-Instruct +12.9pp）。一个自然的 follow-up：SSD + RL 串行或交替。

### 4. 与 DPO 的联想

DPO 从 RLHF 的 KL-regularized objective 出发，闭式求解最优 policy，得到 implicit reward。SSD 的公式 (15) 也可以反向理解：从 "training on temperature-shifted self-samples" 出发，闭式分解出 Rényi entropy 项 —— 相当于一个 implicit entropy regularizer，order 是 $1/T$ 而不是 1。

### 5. 一个值得做的实验

paper 没探索 SSD 是否可以 **iterate**：SSD 后的 student 作为新 teacher，再采样再 SFT。如果每轮都 break fixed point，可能形成 self-improvement loop。但 risk 是 mode collapse —— 每轮 support compression 都在缩小 retained set。

直觉预测：iterate 1-2 轮可能继续增益，3 轮以上收益递减并开始 collapse。Rényi entropy term 在第 $n$ 轮的有效 order 是 $1/T^n$，会迅速趋近 0（max entropy regime），让 reshaping 失去 context-sensitivity。

---

## References

- Paper: https://github.com/apple/ml-ssd
- LiveCodeBench: https://livecodebench.github.io/
- rSTARcoder: https://arxiv.org/abs/2505.21297
- Hinton Distillation: https://arxiv.org/abs/1503.02531
- Kim & Rush Seq-Level KD: https://arxiv.org/abs/1606.07947
- STaR: https://arxiv.org/abs/2203.14465
- ReST-EM: https://arxiv.org/abs/2312.06585
- On-Policy Distillation: https://arxiv.org/abs/2406.13666
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath/GRPO: https://arxiv.org/abs/2402.03300
- Forking Paths: https://openreview.net/forum?id=yp3siPHsJU
- Rényi Entropy: https://projecteuclid.org/euclid.bsmsp/1200512181
- Truncation as Desmoothing: https://arxiv.org/abs/2210.15191
- vLLM: https://arxiv.org/abs/2309.06180
- Holtzman Curious Case: https://arxiv.org/abs/1904.09751
- Entropy Collapse in RL: https://arxiv.org/abs/2505.22617

---

# Embarrassingly Simple Self-Distillation: 深度技术讲解

Andrej，这篇 paper 我猜你会喜欢，因为它和你常强调的 "simplicity wins" 哲学高度契合 —— 方法本身只需要一个 frozen model、一批 prompts、标准 cross-entropy，zero verifier、zero teacher、zero RL，却把 Qwen3-30B-Instruct 在 LiveCodeBench v6 上从 42.4% 拉到 55.3% pass@1。下面我把机制讲透，重点 build 你对 "precision-exploration conflict" 和 "support compression + within-support reshaping" 的 intuition。

---

## 1. 方法本体：极简到反直觉

### 1.1 Pipeline 三步

```
frozen p_θ  --(T_train, ρ_train)-->  raw samples y  --(vanilla SFT)-->  p_{θ*}  --(T_eval, ρ_eval)-->  ŷ
```

公式 (1) 数据合成：
$$y \sim \mathsf{Decode}_{T_{\mathsf{train}}, \rho_{\mathsf{train}}}\big[p_\theta(\cdot \mid x)\big]$$

- $x$：competitive programming prompt（来自 rSTARcoder seed subset，去重后 ~10K 题）
- $T_{\mathsf{train}}$：训练时采样 temperature（Qwen3-30B-Instruct 用 1.6）
- $\rho_{\mathsf{train}}$：truncation 配置，即 top-k 和 top-p（这里 top-k=20, top-p=0.8）
- $N=1$：每个 prompt 只采样一条，不做任何 correctness filtering

公式 (2) 训练就是教科书 cross-entropy：
$$\mathcal{L}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\mathsf{SSD}}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$$

- $y_t$：第 $t$ 个 token
- $y_{<t}$：前 $t-1$ 个 token 的 prefix
- $|y|$：序列长度（最长 128K）

公式 (3) inference：
$$\hat{y} \sim \mathsf{Decode}_{T_{\mathrm{eval}}, \rho_{\mathsf{eval}}}\big[p_{\theta^*}(\cdot \mid x)\big]$$

关键 trick：**训练时用 high temperature 采样，推理时用相对低但非平凡的 temperature**。对 Qwen3-30B-Instruct：$T_{\mathsf{train}}=1.6, T_{\mathsf{eval}}=0.9$。

### 1.2 与其他 paradigm 的对比（Table 1）

| Method | Dense Signal | No Teacher | No Verifier | No Privileged Info |
|---|---|---|---|---|
| SFT on External Data | ✓ | ✗ | ✓ | ✗ |
| GRPO | ✗ | ✓ | ✗ | ✓ |
| On-Policy Distillation | ✓ | ✗ | partial | ✗ |
| On-Policy Self-Distillation | ✓ | ✗ | ✓ | ✗ |
| **SSD (Ours)** | **✓** | **✓** | **✓** | **✓** |

SSD 是唯一同时打四个勾的。这点很关键 —— 因为没有任何外部信号，所有 learning signal 必须来自 "decoding rule 本身改变了 target distribution" 这一事实。

---

## 2. 实验数据：哪些数字最值得记住

### 2.1 主结果（Table 2，LCB v6）

| Model | Base pass@1 | +SSD pass@1 | Δ | Hard pass@1 Δ | Hard pass@5 Δ |
|---|---|---|---|---|---|
| Qwen3-30B-Instruct | 42.4 | 55.3 | **+12.9** | +15.3 | +23.0 |
| Qwen3-4B-Instruct | 34.0 | 41.5 | +7.5 | +5.7 | +17.6 |
| Qwen3-4B-Thinking | 54.5 | 57.8 | +3.3 | +4.1 | +7.3 |
| Qwen3-30B-Thinking | 66.1 | 68.2 | +2.1 | +5.2 | +6.1 |
| Llama-3.1-8B-Instruct | 12.7 | 16.2 | +3.5 | +1.6 | +2.5 |

两个非平凡 pattern：
1. **越难增益越大**：30B-Instruct 在 easy 只 +6.5pp，hard 却 +15.3pp。这立刻排除了 "SSD 只是把简单题刷更稳" 的解释。
2. **pass@5 增益 > pass@1 增益**：30B-Instruct hard 上 pass@1 +15.3pp 但 pass@5 +23.0pp。说明 SSD 没有collapse多样性，反而**扩展了 coverage**。这点对理解机制至关重要 —— 单纯 sharpening 模式会压低 pass@5。

### 2.2 Decode-only sweep 拉不开（Figure 2）

在 base model 上 sweep $T_{\mathsf{eval}}$，pass@1 只在 41.3%–43.5% 之间波动（spread 2.2pp）。SSD 仍然 +11.8pp 超过 best-tuned base。这证明 SSD 改变的是 model 本身，不是 decoding policy。

### 2.3 Hyperparameter 交互（Figure 3）

定义 **effective temperature**：
$$T_{\mathsf{eff}} = T_{\mathsf{train}} \cdot T_{\mathsf{eval}}$$

无 truncation 时，performance 几乎只由 $T_{\mathsf{eff}}$ 决定（$R^2=0.75$），quadratic peak 在 $T_{\mathsf{eff}} \approx 1.2$。有 truncation 时 ceiling 抬升，但 $T_{\mathsf{eff}}$ band 依然成立。最佳点 $T_{\mathsf{train}}=2.0, T_{\mathsf{eval}}=1.1, \text{top-k}=10$ → 49.7% pass@1。

---

## 3. 核心 Hypothesis：Precision-Exploration Conflict

这是整篇 paper 的灵魂。代码生成本质上交织两类 token positions：

### 3.1 Fork positions（exploration-bound）

函数体开头那行 —— 可能 `for i in range(...)`，可能 `def helper(...)` 递归，可能 `from collections import defaultdict` 然后建图。每个选择对应**根本不同的 solution approach**（quicksort vs mergesort vs built-in sort）。这些 continuation 都是 viable 的，distribution 真正 spread 在多个 head token 上。

### 3.2 Lock positions（precision-bound）

`if n ==` 之后必须接特定值。model 知道是哪个，但 vocab 里仍有一长串 syntactically-plausible 的 distractor tail（其他数字、其他运算符）携带非零概率。

### 3.3 为什么这是 conflict

Inference temperature $T_{\mathsf{eval}}$ 作用于整个分布 $p_T(v) \propto p(v)^{1/T}$：
- **Lower $T_{\mathsf{eval}}$**：放大 peak 之间的 gap → locks 安全（distractor 被压）→ 但 fork head 被压成 singleton，exploration 饿死
- **Higher $T_{\mathsf{eval}}$**：flatten head → fork 能探索 → 但 lock 的 distractor tail 回潮，一个错误 token 就让整条 trajectory 崩

全局单一 $T_{\mathsf{eval}}$ 必然是 compromise。这就是 Figure 4 的图示。

**直觉类比**：这就像你写编译器时，lexer 需要确定性（lock），parser 需要回溯（fork）。你不可能用一个全局策略同时优化两者。

---

## 4. 为什么 naive self-training 是 fixed point

这是理解 SSD 为什么能 work 的第一步。如果 $T_{\mathsf{train}}=1$ 且无 truncation，self-training 的期望梯度恒等于零（公式 9）：

$$\mathbb{E}_{v \sim p_\theta(\cdot \mid s)}\big[\nabla_\theta \log p_\theta(v \mid s)\big] = \nabla_\theta \sum_v p_\theta(v \mid s) = \nabla_\theta 1 = 0$$

变量：
- $s = (x, y_{<t})$：context
- $v$：next token
- $p_\theta(\cdot \mid s)$：当前 model 分布

score function gradient 在自身分布下取期望，由于概率归一化 telescopes 到 1 的梯度。**naive self-training 没有任何 learning signal**。任何有用的 signal 必须来自 decoding rule 对 target 的扭曲。

---

## 5. SSD 的理论分解（公式 4 / 15）

这是 paper 最 elegant 的部分。把 SSD loss 在单个 context $s$ 上分解：

$$\mathcal{L}_s(\theta) = \underbrace{-\log \mathsf{KeptMass}_\theta(s)}_{\text{support compression (via } \rho_{\mathsf{train}})} + \underbrace{(1-T) H_{1/T}\big(p_\theta(\cdot \mid s, S_s)\big)}_{\text{within-support reshaping (via } T_{\mathsf{train}})} + \underbrace{T \cdot \mathrm{KL}\big(q_s \| p_{\theta,T}(\cdot \mid s, S_s)\big)}_{\text{alignment to base}} + \mathrm{const}$$

### 5.1 变量与符号

- $\theta$：student 参数（被优化）
- $s$：单个 decoding context
- $S_s$：**retained support** —— teacher 在 context $s$ 上经过 temperature + top-k + top-p 后存活的 token 集合
- $T \equiv T_{\mathsf{train}}$：训练 temperature
- $\mathsf{KeptMass}_\theta(s) = \sum_{v \in S_s} p_\theta(v \mid s)$：student 当前在 retained support 上的总质量
- $p_\theta(\cdot \mid s, S_s) = p_\theta(\cdot \mid s) \cdot \mathbf{1}\{v \in S_s\} / \mathsf{KeptMass}_\theta(s)$：student 在 retained support 上的条件分布
- $H_{1/T}(\pi) = \frac{1}{1-1/T} \log \sum_v \pi(v)^{1/T}$：**Rényi entropy of order $1/T$**
- $q_s$：teacher 的 truncated-tempered target distribution（公式 7）
- $p_{\theta,T}(\cdot \mid S_s)$：student 在 $S_s$ 上 tempered 后的分布

### 5.2 三项的物理意义

**Term 1: Support compression（gate term）**
$$-\log \mathsf{KeptMass}_\theta(s)$$

最大化 student 在 $S_s$ 上的质量 → 等价于把 distractor tail（$S_s$ 外的 token）的概率压到 0。由于 $q_s$ 在 $S_s$ 外严格为 0，最优解只在 outside logits $\to -\infty$ 时达到 —— 训练永远不满足，**tail suppression 是持续压力**。

logit-level gradient（公式 17）对 $v \notin S_s$ 直接给出 $+p_\theta(v \mid s)$：明确地把这些 logit 往下推。

**Term 2: Within-support reshaping**
$$(1-T) H_{1/T}\big(p_\theta(\cdot \mid s, S_s)\big)$$

- $T > 1$（典型 SSD setting）：$(1-T) < 0$，最小化 loss 等于**最大化** $H_{1/T}$ → 在 retained set 内**平滑**分布
- $T < 1$：反向，sharpen
- $T = 1$：此项消失，回到 fixed point

**关键**：Rényi entropy $H_{1/T}$ 的 order $1/T$。当 $T>1$，order $<1$，落在 sub-Shannon regime —— 对 diffuse tail 更敏感、对集中 peak 更宽容。这恰好让它在 fork 处把多个 viable continuation 摊平，又不会过度鼓励 lock 处的低质量 distractor。

**Term 3: KL anchor**
$$T \cdot \mathrm{KL}\big(q_s \| p_{\theta,T}(\cdot \mid s, S_s)\big)$$

防止 student 漂离 teacher 的相对偏好。它把 reshaping 限制在"teacher 已经认为值得保留"的 token 集合内 —— smoothing 只发生在 viable alternatives 之间，不会无中生有。

### 5.3 为什么这个分解 non-trivial

标准 knowledge distillation（Hinton et al. 2015）匹配 full-vocabulary teacher，**没有 gate term**，因此没有机制驱动 tail suppression。SSD 的 gate term 来自 truncation，是 distillation 没有的额外自由度。

Policy-gradient RL（GRPO, PPO）通过 external return 加权 score function 来 break fixed point。SSD 不用 return，而是**直接改变 target distribution 本身**，保持 supervised learning 的 positive normalized weights 形式 —— 这是它稳定且简单的根源。

---

## 6. Lock vs Fork：同一 global objective 的 context-adaptive 行为

这是 mechanism story 的核心。同一公式 (15) 在不同 context 上表现完全不同，因为 $S_s$ 的几何形状不同。

### 6.1 At locks：support compression dominates

Lock 处分布 sharply peaked，truncation 后 $|S_s|$ 很小（往往 1-2 个 token）。Rényi 项 $H_{1/T} \leq \log |S_s|$ 几乎没有发挥空间。学习信号主要来自 gate term —— 把 distractor tail 的质量搬进 $S_s$。

效果：lock 变成 sharp spike，对 $T_{\mathsf{eval}}$ 几乎免疫。Figure 5b 直观展示了这一点。

### 6.2 At forks：within-support reshaping dominates

Fork 处分布 spread，truncation 后 $|S_s|$ 较大（4-5 个 viable continuation）。Gate term 仍活跃但 head 已含大部分有用质量。Rényi 项有发挥空间 —— $T>1$ 时把这几个 viable alternative 摊平成 plateau，同时**不会** reopen 已被 truncation 切掉的 tail。

效果：fork 变成 broad plateau，$T_{\mathsf{eval}}$ 在其上有真正的 leverage。Figure 5a 直观展示。

### 6.3 协同：训练 + decoding 互补

toy simulation（Section 4.2, Figure 14）讲得最清楚：
- **训练单独不够**：SSD 训练后 lock 更安全，但 fork 仍需要 evaluation-time temperature 来探索
- **decoding 单独不够**：调 base model 的 $T_{\mathsf{eval}}$ 给 fork 多样性，会同时让 lock 的 distractor 回潮
- **两者协同**：训练让 lock 安全 → decoding 可以放心提高 $T_{\mathsf{eval}}$ 去 explore fork → 最优 $T_{\mathsf{eval}}$ 从 0.63 上移到 2.09

这解释了为什么 Figure 2 的 base-model temperature sweep 如此 flat —— 它卡在 compromise 里，调哪里都到不了 SSD 的点。

---

## 7. Entropy 悖论的化解（公式 30）

SSD 后 model 的 total entropy 下降（更集中），但 pass@5 反而上升（更 explorable）。看似矛盾，公式 30 把它分解清楚：

$$H\big(p_\theta(\cdot \mid s)\big) = \underbrace{h_2(\mathsf{KeptMass}_\theta(s))}_{\text{gate entropy}} + \underbrace{\mathsf{KeptMass}_\theta(s) \cdot H\big(p_\theta(\cdot \mid s, S_s)\big)}_{\text{head entropy}} + \underbrace{(1-\mathsf{KeptMass}_\theta(s)) \cdot H\big(u_\theta(\cdot \mid s)\big)}_{\text{tail entropy}}$$

变量：
- $h_2(\pi) = -\pi\log\pi - (1-\pi)\log(1-\pi)$：binary entropy
- $u_\theta(v \mid s)$：student 在 $S_s$ 补集上的条件分布

**总 entropy 下降的来源**：gate entropy（KeptMass→1 时 $h_2$→0）+ tail entropy（$1-\mathsf{KeptMass}$→0 时整体→0）。这两项作用在整个 vocabulary 上，幅度大。

**Head entropy 可以局部上升**：在 fork-like context，$T_{\mathsf{train}}>1$ 让 retained head 更平坦 → $H(p_\theta(\cdot \mid s, S_s))$ 上升。但被 $|S_s|$ 上界 bound，幅度小。

**Evaluation-time temperature 只作用于 head entropy**：公式 31
$$\frac{d}{d\tau} H\big(\mathsf{Temper}_\tau^{S_s}[p_\theta(\cdot \mid s, S_s)]\big) = \frac{\mathrm{Var}_{\mathsf{Temper}_\tau^{S_s}}[\log p_\theta(v \mid s, S_s)]}{\tau^3} \geq 0$$

如果 head 是 singleton 或 nearly uniform，variance 接近 0，temperature 几乎无效。如果 head 是几个非等同 token，variance 大，temperature 强力调整 operational policy。

**直觉**：SSD 去掉了 "wrong kind of uncertainty"（diffuse tail），保留了 "right kind of uncertainty"（fork head 的 viable alternatives）。total entropy 和 useful exploration 是两个不同对象 —— 这就是 paradox 的解。

---

## 8. Effective Temperature Composition（Lemma B.1, Prop B.2）

在 local ideal-fit 近似下（$p_\theta(\cdot \mid s) = q_s$），student 在 evaluation 时表现为：

$$q_{s,\tau}(v) = \frac{\mathbf{1}\{v \in S_s\} p_0(v \mid s)^{1/(T_{\mathsf{train}} \tau)}}{\sum_{u \in S_s} p_0(u \mid s)^{1/(T_{\mathsf{train}} \tau)}}$$

也就是说 student 在 retained set 内表现为 teacher 在 product temperature $T_{\mathsf{eff}} = T_{\mathsf{train}} \cdot T_{\mathsf{eval}}$ 下的行为。这解释了 Figure 3 的 $T_{\mathsf{eff}}$ band。

Lemma B.1 的证明极简：
$$\mathsf{Temper}_{T_2}[\mathsf{Temper}_{T_1}[p]](v) \propto \big(p(v)^{1/T_1}\big)^{1/T_2} = p(v)^{1/(T_1 T_2)}$$

renormalization 常数 cancel。这是 power law 的结合律。

---

## 9. Decode-Only Tuning 为什么追不上 SSD（Section B.5）

这是 paper 最 elegant 的 impossibility result。

### 9.1 Normal form（Proposition B.5）

任何 fixed ordering 的 temperature + top-k + top-p 操作，最终都 collapse 成：

$$\mu_s^\sigma((i)) = \frac{p_{(i)}(s)^\alpha \mathbf{1}[i \leq m_s^\sigma]}{\sum_{j=1}^{m_s^\sigma} p_{(j)}(s)^\alpha}$$

其中 $p_{(1)} \geq p_{(2)} \geq \cdots$ 是 frozen model 的 ranking，$\alpha = 1/T_{\mathsf{eval}}$，$m_s^\sigma$ 是 prefix 长度。

无论你怎么 reorder 操作，最终 decoder 都是**对 frozen ranking 的某个 prefix 做单一 power transform**。reorder 只能移动 prefix boundary，不能改变 transformation 的形式。

### 9.2 两个 rigidity

**Corollary B.6 (Prefix rigidity)**：要保留 rank-$r$ token，必须同时保留所有 rank $< r$ 的 token，哪怕它们是 distractor。

**Corollary B.7 (Power rigidity)**：所有 surviving pair 的 log-odds 被同一个 global factor $\alpha$ 缩放：
$$\log \frac{\mu_s^\sigma((i))}{\mu_s^\sigma((j))} = \alpha \log \frac{p_{(i)}(s)}{p_{(j)}(s)}$$

你**不能**在 fork 处 flatten head 的同时**在** lock 处 sharpen peak —— 同一个 $\alpha$ 同时作用两者。这就是 Figure 2 sweep 曲线 flat 的结构原因。

### 9.3 SSD 拥有的额外自由度

SSD 把 $p_0(\cdot \mid s) \to p_\theta(\cdot \mid s)$，**改变了 decoder 看到的 cumulative curve**：
$$S_{s,m}(\tau, k; p_0) \longrightarrow S_{s,m}(\tau, k; p_\theta)$$

decode-only reordering 无法触碰这条 curve。SSD 在 lock 处和 fork 处**分别**移动这条 curve（lock 处压缩 tail，fork 处清理 head），让 equation (40) 的可行区间变宽。这是 SSD 能 break precision-exploration conflict 的根本原因。

---

## 10. Bad Data, Good Results（Section 4.4）—— 最反直觉的 stress test

这个实验直接 challenge "SSD 靠训练在 correct code 上" 的解释。

设置：$T_{\mathsf{train}}=2.0$，**完全关闭 truncation**（top-k = vocab size, top-p = 1.0）。

结果：
- ~62% 输出**根本无法 extract code**
- 看似 coherent 的输出 mid-sequence 退化成 multilingual gibberish（Figure 7a 的例子从 line 13 开始变成意大利语、俄语、阿拉伯语混杂）
- training loss 飙到 11.29

但 fine-tuned model 仍达到：
- 48.1% pass@1（+5.7pp over 42.4% baseline）
- 64.0% pass@5（+10.5pp over 53.5%）
- Hard pass@1 +7.3pp, hard pass@5 +13.8pp

**这证明了什么**：SSD 的有用 signal 主要来自 distributional reshaping，**不是**来自训练在 correct program 上。即使 training data 几乎是 gibberish，只要 sampling 引入了非平凡的温度扭曲，Rényi shaping term 仍提供 signal —— 把 head 内的 viable alternatives 摊平。Evaluation-time truncation（top-k=20, top-p=0.95）负责在 inference 时把训练时未被压制的 distractor tail 清理掉。

这个实验对理解 SSD 的本质极其重要：**signal 在 token-level distribution shape 里，不在 program correctness 里**。

---

## 11. Toy Simulation（Section 4.2, Appendix C.4）

为了 isolate mechanism，作者构造了一个 16-token vocab 的 FSM（Figure 12）：
- Root：tok2 (FAIL) 概率最高，tok0/tok1 分别进入两条对称 success path
- 每条 path 经过 1 个 fork + 3 个 lock
- Fork：correct 在 rank-2，4 个 head token near-tied
- Lock：correct 在 rank-1，75% 质量，剩 25% 在 15-token geometric tail

精确闭式 success probability：
$$P = [q_{\text{root}}(A) + q_{\text{root}}(B)] \cdot q_{\text{fork}}(\text{correct}) \cdot q_{\text{lock}}(\text{correct})^3$$

结果（Figure 14）：
- Teacher 最优：$T^*=0.639$, $P=8.32\%$
- Student 最优：$T^*=2.091$, $P=13.77\%$（+5.4pp）
- Student 的最优 $T$ 上移 3 倍

在最优 $T$ 处，teacher 的 fork nucleus 是 descending $[48.2, 17.8, 17.0, 17.0]\%$，student 是 plateau $[32.1, 22.9, 22.5, 22.5]\%$ —— student 给 lower-ranked correct continuation 多得多的 mass。

robustness check：sweep top-p $\in \{0.65, ..., 0.90\}$，student 始终领先 +1.4 到 +5.4pp。

---

## 12. Out-of-Domain Transfer（Section C.3, Table 5）

只用 competitive programming data 训练，会不会伤害其他能力？

**30B 模型基本稳定**：Qwen3-30B-Instruct 在 AIME '24/'25、HumanEval Py/Sh、CruxEval Input/Output、MMLU 上变化都在 ±2pp 内。MMLU 只掉 0.1pp。

**小模型 tradeoff 更明显**：
- Qwen3-4B-Instruct：AIME '24 掉 6.3pp，HumanEval Shell 掉 15.8pp，但 HumanEval Py 涨 0.6pp，CruxEval 涨 1.3pp
- Llama-3.1-8B：AIME 几乎归零（因为开始输出 code block 而不是数字答案），但 HumanEval 和 CruxEval 上涨

**直觉**：30B 容量大，能把 competitive programming 的 distributional improvement 局部化；4B 容量小，SSD 信号 overflow 到相邻 domain。

---

## 13. 与相关工作的定位

### 13.1 vs. STaR / ReST$^{\text{EM}}$（Zelikman et al. 2022; Singh et al. 2024）

STaR 用 correctness filtering 把 self-generated correct solutions 作为 SFT target。ReST$^{\text{EM}}$ 类似。两者都依赖 verifier。SSD 直接训练在 raw outputs 上，zero filtering。Bad-data stress test 证明 correctness 不是 signal 来源。

### 13.2 vs. On-Policy Distillation（Agarwal et al. 2024）

On-policy distillation 用 teacher model 提供 on-policy distribution 给 student 监督，需要 external teacher。SSD 的 teacher 就是 frozen self，无外部教师。

### 13.3 vs. Unsupervised RLVR / Entropy Minimization（He et al. 2026; Agarwal et al. 2025; Zuo et al. 2025）

这批工作用 majority vote / entropy / self-certainty 作为 intrinsic reward 做 RL。两个核心区别：
1. SSD 用 Rényi entropy of order $1/T$，**不是** Shannon entropy。order 随 $T$ 连续变化。
2. SSD **不是**直接优化 entropy objective —— 它仍然是 supervised learning with positive normalized weights。RL 的 signed policy-gradient weight 在 long training 下容易 reward hacking / collapse（Zhang et al. 2025）。SSD 没有这个 failure mode。

### 13.4 vs. Standard Knowledge Distillation（Hinton et al. 2015; Kim & Rush 2016）

KD 匹配 full-vocabulary teacher distribution。**没有 gate term**。所以 KD 不能驱动 tail suppression —— 它只是 reshaping。SSD 多了 truncation 带来的 support compression。

### 13.5 vs. Forking Tokens 文献（Bigelow et al. 2025; Wang et al. 2025b）

这批工作识别 reasoning 中 high-entropy / forking tokens 是 RL 中 disproportionate 重要的 decision points。SSD 接受这个 framing，但问的问题不同：**plain cross-entropy on self-outputs 能走多远，为什么 reshape distribution 的方式 decode-only 模仿不了**。

---

## 14. 对你（Karpathy）的直觉启示 + 联想

### 14.1 与 makemore / nanoGPT lecture 的连接

你在 makemore 里讲 temperature、top-k、top-p 时强调它们是 inference-time knob。这篇 paper 给出一个反直觉结论：**inference-time knob 的能力有结构上限**（Proposition B.5 的 normal form + 两个 rigidity）。要突破这个上限，必须把 decoding rule "bake into" model 本身 —— 这就是 SSD 的本质。

直觉上，SSD 像是 "把 decoding policy 蒸馏进 weights"。但它蒸发的不是某个具体 policy，而是 "经过 temperature + truncation 扭曲后的 self-distribution"。这个 target 已经被 decoding rule pre-shaped，所以训练等价于让 weights 内化这种 shaping。

### 14.2 与 "The State of GPT" talk 的连接

你在 State of GPT 里区分 pre-training、SFT、RLHF 三个 stage，并强调 RLHF 让 model 从 predicting human 转向 aligning with preference。SSD 提供了一个有趣的第四选项：**用 self-distorted distribution 作为 SFT target**。它不需要 preference model，不需要 reward，只需要 decoding rule 本身。

这暗示了一个更广的可能性：post-training 的 "signal" 不一定要来自外部（human label / verifier / reward model），可以来自 inference-time procedural distortion。decoding rule 是免费的、无限的 signal source。

### 14.3 与 DeepSeek-R1 / GRPO 的对比

DeepSeek-R1 用 GRPO 在 verifiable reward 上做 RL，达到 reasoning SOTA。但 GRPO 训练不稳定，entropy collapse 是已知 failure mode（Cui et al. 2025）。SSD 提供了一个互补方向：用 supervised learning 内化 decoding distortion，**完全绕开 RL 的 instability**。

不过 SSD 的 ceiling 比 RL 低 —— Table 2 显示 thinking model 增益比 instruct model 小（30B-Thinking +2.1pp vs 30B-Instruct +12.9pp）。这暗示 SSD 对已经过 RL 训练的 model 边际收益递减。一个自然的 follow-up：SSD + RL 串行或交替，可能比单独 RL 更稳定。

### 14.4 与 DPO 的联想

DPO 用 preference pair 直接优化 log-ratio，避开 reward model。SSD 在精神上类似 —— 避开 verifier，直接用 self-signal。但 DPO 仍需要 preference label，SSD 连这都不需要。

如果联想到 DPO 的 derivation：它从 RLHF 的 KL-regularized objective 出发，闭式求解最优 policy，得到一个 implicit reward。SSD 的公式 (15) 也可以反向理解：从 "training on temperature-shifted self-samples" 出发，闭式分解出 Rényi entropy 项 —— 这相当于一个 implicit entropy regularizer，但 order 是 $1/T$ 而不是 1。

### 14.5 与 Anthropic Constitutional AI 的对比

Constitutional AI 用 AI feedback 替代 human feedback，但仍然需要 reward model 训练。SSD 完全不用 feedback loop，单 pass SFT 就够。这是 simplicity 的极致。

### 14.6 与 mixture-of-experts 的交互

Qwen3-30B-A3B 是 MoE（30B total / 3B active）。SSD 在 MoE 上同样有效（+12.9pp）说明 mechanism 不依赖 dense architecture。但 paper 没分析 SSD 是否改变了 expert routing distribution —— 这是一个 open question。直觉上，gate term 可能让 router 更 sharp，within-support reshaping 可能让 chosen expert 内部分布更平坦。值得 probe。

### 14.7 与 "grokking" / "lazy training" 的联想

SSD 只跑 2500 iterations（instruct）/ 300 iterations（thinking），LR 极小（$5 \times 10^{-6}$）。这暗示 model 已经 "知道" 正确 distribution，SSD 只是在做小修正 —— 把 decoding-time distortion 内化。这与 lazy training 的图景一致：大模型在小 signal 下倾向于小 weight update。也呼应 paper 结论："strong code models contain latent capability that can be unlocked without a verifier"。

### 14.8 与 you 的 "Software 2.0" 论述的连接

你曾论述 Software 2.0 是 weights 取代 code。SSD 提供一个新维度：**decoding rule 本身可以被视为一种 "soft code"，可以被蒸馏进 weights**。这是 Software 2.0 的 self-referential 闭环 —— model 不仅生成 code，还把自己的 inference procedure 编译进自己的 weights。

### 14.9 一个值得做的实验

paper 没探索的是：**SSD 是否可以 iterate**？即 SSD 后的 student 作为新 teacher，再采样再 SFT。如果每轮都 break fixed point（通过新的 decoding distortion），可能形成 self-improvement loop。但 risk 是 mode collapse —— 每轮 support compression 都在缩小 retained set，多轮后可能失去 diversity。这与你讨论过的 "self-play 何时 diverge 何时 collapse" 问题相关。

直觉预测：iterate 1-2 轮可能继续增益，3 轮以上收益递减并开始 collapse。Rényi entropy term 在第 $n$ 轮的有效 order 是 $1/T^n$，会迅速趋近 0（max entropy regime），让 reshaping 失去 context-sensitivity。

---

## 15. 局限与开放问题

1. **Domain-specific**：只在 code 上验证。Code 的 lock/fork 结构特别明显（语法强制 lock，算法选择强制 fork）。在 math reasoning、creative writing 上 fork/lock 边界模糊，SSD 效果未知。
2. **Thinking model 增益小**：30B-Thinking +2.1pp vs 30B-Instruct +12.9pp。可能因为 thinking model 已经被 RL 训练过，distribution 已经被 reshape 过，SSD 边际收益小。
3. **小模型 transfer 退化**：4B 模型在 AIME / HumanEval Shell 上掉点严重，说明 SSD signal 在 capacity 受限时 overflow。
4. **$T_{\mathsf{train}}, T_{\mathsf{eval}}, \rho$ 的选择仍需 grid search**：paper 给了 per-model 配置（Table 3）但没有 principled 选择方法。$T_{\mathsf{eff}} \approx 1.2$ 的经验法则是一个 anchor，但 truncation 的最优配置仍 empirical。
5. **Theory 假设 local ideal-fit**：公式 (23) $p_\theta(\cdot \mid s) = q_s$ 在实践中只近似成立。真实 model capacity / optimization 限制下的偏差未被分析。
6. **MoE 的 expert routing 未被 probe**：SSD 在 MoE 上 work，但内部 routing 是否变化、如何变化，paper 没说。

---

## 16. 总结：一句话直觉

**SSD 把 decoding-time 的 temperature-truncation distortion 蒸馏进 weights，让 model 在 lock context 内化 tail suppression、在 fork context 内化 head flattening，从而把 evaluation-time temperature 从 "全局 compromise knob" 解放成 "局部 exploration knob"。decode-only tuning 做不到这点，因为它受 prefix rigidity 和 power rigidity 双重约束 —— 这两个 rigidity 是任何 fixed decoding operator ordering 的不可避免 normal form。**

---

## References

- Paper GitHub: https://github.com/apple/ml-ssd
- LiveCodeBench: https://livecodebench.github.io/ (Jain et al. 2024, https://arxiv.org/abs/2403.07974)
- rSTARcoder: https://arxiv.org/abs/2505.21297 (Liu et al. 2025)
- Hinton et al. 2015 Distillation: https://arxiv.org/abs/1503.02531
- Kim & Rush 2016 Seq-Level KD: https://arxiv.org/abs/1606.07947
- STaR (Zelikman et al. 2022): https://arxiv.org/abs/2203.14465
- ReST$^{\text{EM}}$ (Singh et al. 2024): https://arxiv.org/abs/2312.06585
- On-Policy Distillation (Agarwal et al. 2024): https://arxiv.org/abs/2406.13666
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- Forking Paths in Text Generation (Bigelow et al. 2025): https://openreview.net/forum?id=yp3siPHsJU
- High-Entropy Minority Tokens (Wang et al. 2025b): NeurIPS 2025
- Rényi Entropy (1961): https://projecteuclid.org/euclid.bsmsp/1200512181
- Truncation as Desmoothing (Hewitt et al. 2022): https://arxiv.org/abs/2210.15191
- vLLM: https://arxiv.org/abs/2309.06180
- Curious Case of Neural Text Degeneration (Holtzman et al. 2020): https://arxiv.org/abs/1904.09751
- Entropy collapse in RL (Cui et al. 2025): https://arxiv.org/abs/2505.22617
- Unsupervised RLVR (He et al. 2026): https://arxiv.org/abs/2603.08660

如果你想 probe 任何具体一块（比如 Rényi entropy vs Shannon entropy 在 lock/fork 上的 differential behavior、或 MoE routing 的变化、或 iterate SSD 的 collapse dynamics），我可以继续展开。
