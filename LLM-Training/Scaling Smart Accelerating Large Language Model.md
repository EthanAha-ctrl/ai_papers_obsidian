---
source_pdf: Scaling Smart Accelerating Large Language Model.pdf
paper_sha256: ae1dc0912d5de7e760ce27522122343004993f904f9a2ee69e3be9db90b458a8
processed_at: '2026-08-12T03:42:43-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 HyperCloning

## 一句话总结

你有个已经训好的 small model（比如 1B），你想训一个 large model（比如 3B）。传统做法是从 random init 开始训 large model，烧钱烧 time。HyperCloning 说：**把 small model 的 weights 按照某种结构复制粘贴到 large model 里，让 large model 从 step 0 开始就拥有 small model 的能力**，然后继续训，省 3-4 倍的 GPU hours。

就这么简单。核心就是"复制粘贴 + 稍微调一下让数学 work"。

---

## 为什么要这么干

训 large LLM 三个 pain：

1. **贵**。训个 12B 要 72000 GPU hours，public cloud 烧几百万美元。
2. **容易 fail**。loss spike、hardware 挂了、LR 调错了，run 就废了。Llama 3 paper 里专门讲他们怎么 struggle 的。
3. **慢**。Random init 意味着 model 从一个 random point 出发，要花一堆 tokens 才能走到 loss landscape 的 "good region"。

Small model 便宜，但 accuracy 不够。所以 idea 很自然：**能不能让 small model 当 "老师"，但不是通过 distillation 那种 forward 一遍 teacher 的方式，而是直接把 weights 搬过去**？

---

## 怎么搬

这是技术细节的关键。假设 small model 某个 hidden representation 是 $x_S \in \mathbb{R}^d$（维度 $d$），large model 想要的 hidden dimension 是 $2d$。最 naive 的想法：把 $x_S$ 复制两份拼起来：

$$x_D = \begin{bmatrix} x_S \\ x_S \end{bmatrix} \in \mathbb{R}^{2d}$$

这就是 **vector cloning**。整个 HyperCloning 的 goal：让 large model 每一层的 hidden state 都是这个 cloned form，这样 input 到 output logits 完全 match small model。

### Linear layer 三种情况

Linear layer $y = Wx + b$ 要 clone，分情况：

**Case 1: 只 input expand**（比如 unembedding layer，output 是 vocab size 不变）

Large 的 $W_D$ 是把 small 的 $W_S$ 拆两半加 noise：

$$W_D = \begin{bmatrix} \frac{W_S}{2} + \eta_1 & \frac{W_S}{2} - \eta_1 \end{bmatrix}$$

你算一下 $W_D x_D$：第一个 block $\frac{W_S}{2} x_S$ + 第二个 block $\frac{W_S}{2} x_S$ = $W_S x_S$。noise 在两个 block 是相反符号，cancel 掉。所以 output = small model output。✓

为什么要 $\frac{1}{2}$？因为 input 复制了两份，如果不 normalize，output 会变成 2 倍。$\frac{1}{n}$ 这个 normalization 也刚好满足 Glorot init 的 std 要求。

**Case 2: 只 output expand**（比如 embedding layer，input 是 token id 不 expand）

直接堆叠：

$$W_D = \begin{bmatrix} W_S \\ W_S \end{bmatrix}$$

因为 input 没复制，所以不用 $\frac{1}{2}$，直接堆就行。

**Case 3: input 和 output 都 expand**（hidden layer 中间层）

最 general 的情况：

$$W_D = \begin{bmatrix} \frac{W_S}{2} + \eta_1 & \frac{W_S}{2} - \eta_1 \\ \frac{W_S}{2} + \eta_2 & \frac{W_S}{2} - \eta_2 \end{bmatrix}$$

两行各自独立算出来都等于 $y_S$，所以 output 是 $\begin{bmatrix} y_S \\ y_S \end{bmatrix}$。✓

这里的 $\eta_1, \eta_2$ 是两个 **不同的** noise tensor，用来打破 weight 之间的 symmetry。后面会讲为什么这个重要。

### Attention layer 有个坑

Attention 不是纯 linear，有 dot product 的非线性。如果你把 query/key 复制两份：

$$q_D = \begin{bmatrix} q_S \\ q_S \end{bmatrix}, \quad k_D = \begin{bmatrix} k_S \\ k_S \end{bmatrix}$$

算 attention score：

$$a_D = \frac{q_D k_D^T}{\sqrt{2d}} = \frac{q_S k_S^T + q_S k_S^T}{\sqrt{2d}} = \frac{2 q_S k_S^T}{\sqrt{2d}} = \sqrt{2} \cdot a_S$$

哎呀，多了一个 $\sqrt{2}$！因为 numerator 多了一倍但 denominator 只多 $\sqrt{2}$。

**Fix**: 把 query weights 缩放 $\frac{1}{\sqrt{2}}$，一般化就是 $\sqrt{\frac{d_S}{d_D}}$。这个细节如果漏了 attention pattern 就 distort 了。

另一种方式：直接增加 head 数量（比如 16 head → 32 head），每个 head 独立算，没有 scaling 问题。

### LayerNorm 直接复制

LayerNorm 的 $\gamma, \beta$ 直接复制两份。为什么 work？因为 $\begin{bmatrix} x_S \\ x_S \end{bmatrix}$ 的 mean 和 variance 跟 $x_S$ 一样（重复的值不改变统计量），所以 normalize 之后还是 $\begin{bmatrix} l(x_S) \\ l(x_S) \end{bmatrix}$。

### Positional embedding 直接复制

复制 n 份拼起来。对 learned positional embedding 就是直接 tile，对 RoPE 这种可能要 custom 处理。

---

## 实验设了三个场景

| 场景 | Base model | Target | Base 训了多少 | 想模拟什么 |
|---|---|---|---|---|
| OPT | 350M | 1.3B | 30B tokens | "我自己 quick 训一个小 base" |
| Pythia | 410M | 1.4B | 250B tokens (Pile) | "用别人训好的 base，但我换 dataset" |
| OLMO | 1B | 2.9B | 2.4T tokens (DOLMA) | "base 已经训得很充分" |

三个场景测的是不同现实情况：自己 quick 训一个 base、用公开 base 但换数据、base 训得超充分。

Training 配置：AdamW，64 GPUs，ZeRO-2，DOLMA dataset。

---

## 关键结果

### 1. Speedup 2.2x - 4x

Figure 2 显示 HyperCloning init 的 model 很快达到 random init 的 final accuracy，然后继续提升。

- OPT: 2.2x
- Pythia: 3x
- OLMO: 4x

Base model 训得越充分（OLMO 训了 2.4T tokens），speedup 越大。这符合直觉：base 携带的 knowledge 越多，warm start 起点越好。

### 2. Catastrophic forgetting 现象

OLMO 实验里有个有意思的现象：训练初期 HyperCloning 的 accuracy **先掉**，然后才 recover。

为什么？Function preservation 只在 step 0 严格成立。一开始 LR warmup 扰动 weight，新加的 parameter 还没 calibrate 好，加上 data order 变了，导致短暂的 forgetting。但 recover 之后还是大幅超越 random init。

这个现象提示：**初期可以用 lower LR 或者一些 regularization 来减轻 forgetting**，paper 说这是 future work。

### 3. Symmetry 会自动 break

Case 3 init 后，weight matrix 的两个 horizontal block 完全一样（除了 noise）。这有个 concern：duplicated neurons 可能学不到不同的东西，等于浪费 parameter（[LEMON paper](https://arxiv.org/abs/2310.07999) 提过这个 worry）。

Paper 测了 cosine similarity 在训练过程中的 evolution：
- Step 0: similarity = 1（完全对称）
- 训练后: 大多 layer similarity 跌到 0.5 以下

**结论**: symmetry 自动 break 了。Dropout、data diversity、optimizer noise 这些 randomness 让 duplicated block 逐渐 specialize。

### 4. Rank 会 recover

Init 时 cloned weight matrix 的 rank = source matrix rank = destination dimension 的一半（因为有一半是 duplicate）。Concern: 模型一直 low-rank，没用到全部 capacity。

Figure 6 显示训练后的 singular value distribution：HyperCloning init 的 model 和 random init 的 model **几乎一样**。Rank 完全 recover 了。

这个结果反驳了一个 criticism："HyperCloning 只是让 large model 当 small model 用"。实际上训练之后 large model fully utilize 了 enlarged space。

---

## Ablation 的几个 insight

### 哪种 expansion 策略最好

试了四种 function-preserving 的方式：

1. **Symmetric** (paper 默认): $W_D = \begin{bmatrix} \frac{W_S}{2} & \frac{W_S}{2} \\ \frac{W_S}{2} & \frac{W_S}{2} \end{bmatrix}$
2. **Diagonal**: $W_D = \begin{bmatrix} W_S & 0 \\ 0 & W_S \end{bmatrix}$（[Shen et al.](https://arxiv.org/abs/2107.11437) 提的）
3. **Noisy symmetric**: Symmetric + noise
4. **Noisy diagonal**: Diagonal + noise

结果：**Symmetric 最好**，Diagonal 最差。

我的理解：Diagonal init 有 zero block，gradient flow 不顺畅，需要时间把 zero "warm up"。Symmetric 让所有 block 都有信号，gradient 立刻能 flow。Noise 加上去 gain 不大，所以 paper 选 noise-free 省一个 hyperparameter。

### Base model 的影响

用不同训练程度的 OPT-350M checkpoint init OPT-1.3B：
- Base 训得越好，target 起步越好
- 但随着训练进行，gap 缩小
- Final accuracy 都比 random init 好

结论：**base model 主要影响收敛速度，不是 final accuracy 的 upper bound**。

### 可以 stacked 用

OPT-5.3B 可以从 OPT-1.3B（2-fold）或 OPT-350M（4-fold）init。结果 OPT-1.3B init 更好（base 更大更准确）。

**Implication**: 可以递归用 HyperCloning：350M → 1.3B → 5.3B → 21B → ...，每一步都从上一个 stage warm start。

---

## 凭什么 work — 我的 intuition

### 为什么 function preservation 是关键

如果你随便 init 一个 large model，它的 weight 在 loss landscape 里是 random point。要训到 good region 走很长的路。

HyperCloning 让 large model 从一个 **"已知 good" 的 basin 附近**出发（因为 step 0 loss = small model loss）。后续训练相当于 local refinement + capacity expansion。

类比：登山找最低谷。Random init 是空投到随机点。HyperCloning 是把你空投到一个 "已经测出来比较低" 的位置，你只需要 local search 或 escape 到 nearby basin。

### 为什么 width 而不是 depth

Paper 选 width expansion。Depth expansion（[Weight Subcloning](https://arxiv.org/abs/2312.09299)、[Progressive Stacking](https://arxiv.org/abs/1902.07116)）也是 model growth，但 width 有几个优势：

- **Inference latency 不变**：depth 不增加，sequential 计算量不增
- **Per-layer capacity 增加**：每层能 encode 更多 pattern
- **和 depth expansion 互补**：可以组合使用

实际工业部署里 inference cost 是大头，width expansion 对 latency 友好。

### Catastrophic forgetting 怎么解释

Function preservation 只在 step 0 成立。一开始 LR warmup 让 weight 扰动大，新加的 parameter 还没 calibrate，导致短暂 forgetting。但 recover 之后还是赢。

这个 forgetting 是 trade-off：你 pay 一个 initial cost 来 re-calibrate，long-term gain 是更快收敛 + 更好 final accuracy。

### 跟 distillation 的区别

Knowledge distillation 是训 student 的时候加一个 distillation loss，要 forward teacher，改 training loop，复杂。

HyperCloning **只改 init**，training loop 完全不变。这让它极其 drop-in，工业界 adoption 友好。你 existing 的 training pipeline 一行不改，只换 init function。

### 跟 μP 的区别

[μP](https://arxiv.org/abs/2203.03466) (Yang et al.) 也是 width scaling 相关，但角度不同：
- μP：调整 init scale 和 LR 让不同 width 的 model 训练 dynamics 一致，可以从 small model **tune hyperparameter** 然后 transfer 到 large
- HyperCloning：直接 **transfer weights**

理论上可以 combine：用 μP 训 small model，然后 HyperCloning init large model，再用 μP 的 LR。

### Scaling law 视角

[Chinchilla](https://arxiv.org/abs/2203.15556) 说 optimal 是 model size : data size = 1:1。HyperCloning 的 implication：

假设你想训 N parameter 的 model 用 D tokens：
- Random init：训 N model 用 D tokens
- HyperCloning：用 N/4 的 model 训 D/4 tokens，然后 HyperCloning init N model 再训 D/2 tokens

总 cost 远低于 random init。Paper 实验里 OLMO 用 250B tokens (HyperCloning) 达到比 2.4T tokens (random init) 还好的 accuracy，~4x speedup。

---

## Limitations

1. **Expansion factor 必须 integer**：2x, 4x 可以，1.5x 不直接支持
2. **Width only**：不能加 depth（但可以和 depth expansion 组合）
3. **Architecture constraint**：destination hidden dim 必须 source 的整数倍
4. **Catastrophic forgetting**：paper 承认这个现象，没深入研究 mitigation
5. **Beyond transformer**：对 Mamba、SSM 这种非 transformer 架构需要重新设计

---

## 实操 recipe

如果我要用 HyperCloning 训一个 7B：

1. 先训一个 1.75B base model（4x smaller）用 standard recipe
2. HyperCloning 4-fold expand 到 7B：
   - Linear layer: Case 3 with $\frac{W_S}{4}$ blocks
   - Attention: 复制 head 数量最简单
   - LayerNorm: 复制 $\gamma, \beta$
   - Positional embedding: 复制 4 次
3. 用 **完全相同的 training loop** 训 7B，但用更少 tokens
4. 监控 cosine similarity 确认 symmetry 在 break
5. 监控 singular value distribution 确认 rank 在 recover

预期 3-4x speedup，更好 final accuracy。

---

## 我的 take

这篇 paper 的 elegance 在于：**用一个很简单的 weight manipulation trick，把 expensive optimization 问题转化成 cheaper 的问题**。

它不发明新的 optimizer、不改 training loop、不引入新的 loss function。它就是改 init。但这种"小 trick 大效果"的工作往往是最有工业价值的——因为 adoption 成本几乎为零。

从 research angle，它打开了几个有意思的方向：
- Catastrophic forgetting 能不能 mitigate（用 EWC、rehearsal、lower initial LR）
- Depth + Width 怎么 optimal 组合
- Non-integer expansion 用 low-rank approximation 实现
- Online growth（训练中动态 grow）
- Beyond transformer（Mamba、RWKV 怎么 expand）

最后一句：**Function preservation 是关键**。这个 principle 让 warm start 真正成立，避免了 "init 之后比 random 还差" 的尴尬。Paper 用 rank recovery 和 symmetry breaking 的 analysis 证明 large model 没卡在 "当 small model 用" 的 suboptimal region，最终 fully utilize 了 enlarged space。

---

## References

- [HyperCloning paper](https://arxiv.org/abs/2410.14914)
- [Net2Net (Chen 2015)](https://arxiv.org/abs/1511.05641)
- [Weight Subcloning (Samragh 2023)](https://arxiv.org/abs/2312.09299)
- [Stacking your transformers (Du 2024)](https://arxiv.org/abs/2405.15319)
- [LEMON (Wang 2023)](https://arxiv.org/abs/2310.07999)
- [BERT2BERT (Chen 2021)](https://arxiv.org/abs/2110.07143)
- [Progressive Stacking (Gong 2019)](https://arxiv.org/abs/1902.07116)
- [Pythia](https://arxiv.org/abs/2301.09041)
- [OLMO](https://arxiv.org/abs/2402.00838)
- [OPT](https://arxiv.org/abs/2205.01068)
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- [μP (Yang et al.)](https://arxiv.org/abs/2203.03466)
- [FLM-101B](https://arxiv.org/abs/2309.03852)
- [Llama 3](https://arxiv.org/abs/2407.21783)
- [Glorot init](https://proceedings.mlr.press/v9/glorot10a.html)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [DOLMA dataset](https://allenai.org/dolma)
- [Shen et al. diagonal init](https://arxiv.org/abs/2107.11437)

---

# HyperCloning: 用 Small Model Initialize Large Model 的 Function-Preserving Width Expansion

## TL;DR

HyperCloning 是 Apple 提出的一种 **width expansion** 方法：把一个已经 pretrain 好的 small model 的 weights 按照特定结构复制到一个 hidden dimension 更大的 large model 里，使得 large model 在 **step 0 的 forward pass 输出 logits 与 small model 完全一致**（function preservation）。这样 large model 从一个"已知 good"的 loss basin 附近开始训练，而不是从 random point 出发。实验显示 **2.2x–4x 的 speedup**，并且 final accuracy 也更好。

Paper: [arxiv.org/abs/2410.14914](https://arxiv.org/abs/2410.14914) (Apple, 2024)

---

## 1. Motivation: 为什么这件事值得做

Training large LLM from scratch 有三个痛点：

1. **Cost**: 训一个 12B model 大约 72,000 GPU hours（[Biderman et al., Pythia](https://arxiv.org/abs/2301.09041)）。
2. **Failure risk**: loss spike、hardware failure、LR mis-tuning 都可能让 run 失败（[Llama 3 技术报告](https://arxiv.org/abs/2407.21783) 里详细描述了这些 pain）。
3. **Slow convergence**: random init 让 model 一开始处于 loss landscape 的 random point，要花大量 tokens 走到 good region。

Small model 训练便宜，但 accuracy 不够。HyperCloning 的核心 insight: **把 small model 的 knowledge 通过结构化的 weight init "graft" 到 large model 里**，让 large model 起步就具备 small model 的能力。

这跟 **Net2Net** ([Chen et al., 2015](https://arxiv.org/abs/1511.05641)) 的 idea 一脉相承，但 HyperCloning 专门针对 decoder-style transformer，处理了 attention、layer norm、positional embedding 等所有 component。

---

## 2. Method: HyperCloning 的核心构造

### 2.1 Vector Cloning — 最基础的概念

定义 source network 的 hidden representation 为 $x_S \in \mathbb{R}^d$，destination network 的对应 hidden representation 是它的 **n-fold cloned version**：

$$x_D = \begin{bmatrix} x_S \\ x_S \\ \vdots \\ x_S \end{bmatrix} \in \mathbb{R}^{nd}$$

整个 HyperCloning 的 goal 就是: **让 destination network 的每一层 hidden state 都是 source network 对应层的 cloned version**。这样从 input 到 output logits 都 preserve。

### 2.2 Linear Layer Cloning — 三种 case

这是最关键的部分。一个 linear layer $y = Wx + b$，根据 input/output dimension 是否 expand，分三种情况：

#### **Case 1: 只 expand input dimension**（例如 unembedding layer，output 是 vocab logits 不 expand）

- Source: $y_S = W_S x_S + b_S$，其中 $W_S \in \mathbb{R}^{d_{out} \times d}$, $x_S \in \mathbb{R}^d$
- Destination: $x_D = \begin{bmatrix} x_S \\ x_S \end{bmatrix} \in \mathbb{R}^{2d}$, $y_D = y_S \in \mathbb{R}^{d_{out}}$
- Init:
  $$W_D = \begin{bmatrix} \frac{W_S}{2} + \eta_1 & \frac{W_S}{2} - \eta_1 \end{bmatrix} \in \mathbb{R}^{d_{out} \times 2d}, \quad b_D = b_S$$
  其中 $\eta_1$ 是随机 noise tensor，shape 与 $W_S$ 相同。

**验证 function preservation**:
$$y_D = W_D x_D + b_D = \left(\frac{W_S}{2} + \eta_1\right) x_S + \left(\frac{W_S}{2} - \eta_1\right) x_S + b_S = W_S x_S + b_S = y_S \checkmark$$

注意 $\eta_1$ 在两个 block 里是相反符号，所以 cancel out。这里 $\frac{1}{2}$ 是因为 $n=2$，general case 是 $\frac{1}{n}$。这个 normalization 保证 Glorot init 的 std 要求（[Glorot & Bengio, 2010](https://proceedings.mlr.press/v9/glorot10a.html)）。

#### **Case 2: 只 expand output dimension**（例如 embedding layer，input 是 token id 不 expand）

- Source: $y_S = W_S x_S + b_S \in \mathbb{R}^d$
- Destination: $x_D = x_S$, $y_D = \begin{bmatrix} y_S \\ y_S \end{bmatrix} \in \mathbb{R}^{2d}$
- Init:
  $$W_D = \begin{bmatrix} W_S \\ W_S \end{bmatrix} \in \mathbb{R}^{2d \times d_{in}}, \quad b_D = \begin{bmatrix} b_S \\ b_S \end{bmatrix}$$

**验证**:
$$y_D = W_D x_D + b_D = \begin{bmatrix} W_S x_S + b_S \\ W_S x_S + b_S \end{bmatrix} = \begin{bmatrix} y_S \\ y_S \end{bmatrix} \checkmark$$

这里 **没有 $\frac{1}{n}$ normalization**，因为 input 没有被复制（只有一个 $x_S$），直接堆叠 $W_S$ 就能 preserve。

#### **Case 3: input 和 output 都 expand**（hidden layers，attention/FFN 的中间层）

- Source: $y_S \in \mathbb{R}^d$
- Destination: $x_D = \begin{bmatrix} x_S \\ x_S \end{bmatrix}$, $y_D = \begin{bmatrix} y_S \\ y_S \end{bmatrix}$
- Init:
  $$W_D = \begin{bmatrix} \frac{W_S}{2} + \eta_1 & \frac{W_S}{2} - \eta_1 \\ \frac{W_S}{2} + \eta_2 & \frac{W_S}{2} - \eta_2 \end{bmatrix}, \quad b_D = \begin{bmatrix} b_S \\ b_S \end{bmatrix}$$

**验证**: 第一行输出 = $(\frac{W_S}{2} + \eta_1)x_S + (\frac{W_S}{2} - \eta_1)x_S + b_S = W_S x_S + b_S = y_S$。同理第二行。✓

这里的 $\eta_1, \eta_2$ 是两个 **独立的** noise tensor，用来打破 weight 的 symmetry（后面 ablation 会讲为什么这重要）。

### 2.3 Attention Layer Cloning — 两种 sub-case

Attention 比较特殊，因为有 dot product 的非线性。

#### Sub-case A: 扩大 head dimension

如果 head dimension 从 $d$ 变成 $2d$，query/key/value 都按 Case 3 复制：

$$q_D = \begin{bmatrix} q_S \\ q_S \end{bmatrix}, \quad k_D = \begin{bmatrix} k_S \\ k_S \end{bmatrix}$$

计算 attention score:
$$a_S = \frac{q_S k_S^T}{\sqrt{d}}, \quad a_D = \frac{q_D k_D^T}{\sqrt{2d}} = \frac{q_S k_S^T + q_S k_S^T}{\sqrt{2d}} = \frac{2 q_S k_S^T}{\sqrt{2d}} = \sqrt{2} \cdot a_S$$

问题出现了: $a_D \neq a_S$，差了 $\sqrt{2}$ 倍！

**Solution**: scale query weights by $\frac{1}{\sqrt{2}}$，更一般地 by $\sqrt{\frac{d_S}{d_D}}$，其中 $d_S, d_D$ 是 source/destination 的 head dimension。这样 $q_D$ 整体缩小 $\sqrt{\frac{d_S}{d_D}}$，抵消掉 attention score 的 scaling。

这个细节非常重要，否则 attention pattern 会被 distort，破坏 function preservation。

#### Sub-case B: 增加 head 数量

直接复制 heads，没有任何 scaling issue。每个 head 独立计算 attention，多复制几个就行。

### 2.4 Layer Norm Cloning

Layer Norm 公式:
$$l(x_S) = \frac{x_S - \mathbb{E}(x_S)}{\sqrt{\text{var}(x_S) + \epsilon}} \cdot \gamma_S + \beta_S$$

Init:
$$\gamma_D = \begin{bmatrix} \gamma_S \\ \gamma_S \end{bmatrix}, \quad \beta_D = \begin{bmatrix} \beta_S \\ \beta_S \end{bmatrix}$$

**验证**: 关键 observation 是 $\mathbb{E}\left(\begin{bmatrix} x_S \\ x_S \end{bmatrix}\right) = \mathbb{E}(x_S)$ 且 $\text{var}\left(\begin{bmatrix} x_S \\ x_S \end{bmatrix}\right) = \text{var}(x_S)$（因为重复的值不改变 mean 和 variance），所以:

$$l(x_D) = \frac{\begin{bmatrix} x_S \\ x_S \end{bmatrix} - \mathbb{E}(x_S)}{\sqrt{\text{var}(x_S) + \epsilon}} \cdot \begin{bmatrix} \gamma_S \\ \gamma_S \end{bmatrix} + \begin{bmatrix} \beta_S \\ \beta_S \end{bmatrix} = \begin{bmatrix} l(x_S) \\ l(x_S) \end{bmatrix} \checkmark$$

同样适用于 RMSNorm、BatchNorm、GroupNorm。

### 2.5 Positional Embedding Cloning

直接复制 n 次:
$$P_D(X_D, i) = \begin{bmatrix} P_S(x_S, i) \\ P_S(x_S, i) \\ \vdots \\ P_S(x_S, i) \end{bmatrix}$$

对 learned positional embedding 很直接。对 RoPE 这种 rotary embedding，可能需要额外处理（paper 里用 Pytorch custom layer 实现）。

---

## 3. 实验设置

### 3.1 三个 model family

| Model Family | Base Model | Target Model | Base Training Tokens | Dataset |
|---|---|---|---|---|
| OPT | OPT-350M | OPT-1.3B | 30B (self-trained) | DOLMA |
| Pythia | Pythia-410M | Pythia-1.4B | 250B (Pile) | DOLMA (不同!) |
| OLMO | OLMO-1B | OLMO-2.9B | 2.4T (DOLMA) | DOLMA (相同) |

Architecture detail:

| Model | #Layers | #Heads | $d_{model}$ | $d_{FFN}$ |
|---|---|---|---|---|
| OPT-350M (base) | 24 | 16 | 1024 | 4096 |
| OPT-1.3B (target) | 24 | 32 | 2048 | 8192 |
| OLMO-1B (base) | 16 | 16 | 2048 | 16384 |
| OLMO-2.9B (target) | 16 | 32 | 4096 | 16384 |

注意 **layer 数量不变**，只是 hidden dim 翻倍 — 这是 width expansion 的定义。

Training hyperparameters:
- Optimizer: AdamW, weight decay 0.05, $\beta_1 = 0.9$, $\beta_2 = 0.999$
- LR: warmup 25k steps, then cosine decay to 1/10
- ZeRO-2 for memory ([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054))
- 64 GPUs

### 3.2 三个 scenario 模拟

- **OPT**: base 和 target 同 dataset，base 训得少 → 模拟 "quick prototype" 场景
- **Pythia**: base 用 Pile，target 用 DOLMA → 模拟 "dataset 迁移" 场景
- **OLMO**: base 和 target 同 dataset，base 训很多 → 模拟 "充分预训练" 场景

---

## 4. 关键结果

### 4.1 Speedup 和 Final Accuracy

| Model | Speedup vs Random Init | Final Accuracy Gain |
|---|---|---|
| OPT-1.3B | ~2.2x | 显著 |
| Pythia-1.4B | ~3x | 显著 |
| OLMO-2.9B | ~4x | 显著 |

Figure 2 显示 HyperCloning init 的模型很快达到 random init 的 final accuracy，然后继续提升。

### 4.2 Catastrophic Forgetting 现象

OLMO 实验里有个 interesting observation: 训练初期 HyperCloning 的 accuracy **先下降**，然后才 recover 并超越。这是因为:

1. Function preservation 只在 **step 0** 严格成立，一旦 weights 更新就不再 preserve。
2. Base model (OLMO-1B) 在 2.4T tokens 上收敛到某个 solution，新的 OLMO-2.9B 虽然初始 logits 一致，但 optimization trajectory 不同。
3. 新增的 parameters 需要 adapt，这个过程会暂时 disrupt 已有的 knowledge。

但这不是 fatal: 最终 HyperCloning 还是大幅胜过 random init。

### 4.3 Weight Symmetry 的 Evolution (Section 3.2)

HyperCloning init 后，weight matrix 有重复结构（Case 3 里每行的两个 horizontal block 相同）。理论上这会导致 "duplicated neurons 学不到不同东西" 的 concern（[Wang et al., LEMON](https://arxiv.org/abs/2310.07999) 提过这个问题）。

**Metric**: 对 Case 3 的每一行，计算两个 block 的 cosine similarity，然后取平均。

**Observation** (Figure 5):
- Step 0: cosine similarity = 1（完全对称）
- 训练后: 大多数 layer 的 cosine similarity **decay 到 0.5 以下**

这说明 symmetry 自然 break 了，模型 effectively 利用了全部参数空间。原因可能是 dropout 等随机操作引入的 asymmetry。

### 4.4 Rank Recovery (Section 3.3)

Init 时 cloned weight matrix 的 rank 至多等于 source matrix 的 rank，即 destination rank = source rank（destination dimension 是 source 的 2 倍，所以 rank utilization 只有 50%）。

**Question**: 训练后 rank 会 recover 吗？

**Answer** (Figure 6): 训练后，HyperCloning init 的 model 的 singular value distribution 和 random init 的 model **几乎一致**。这说明:
- 初始的 low-rank 结构是临时的
- 训练过程让 weight matrix "填满" 了它的 capacity
- 最终模型 fully utilize 了 enlarged parameter space

这个结果很重要，因为它反驳了 "HyperCloning 只是让 large model 当 small model 用" 的 criticism。

---

## 5. Ablation Studies

### 5.1 Expansion Strategy 比较 (Section 3.4)

四种 function-preserving 的 init 方式:

1. **Symmetric** (paper 默认):
   $$W_L = \begin{bmatrix} \frac{W_S}{2} & \frac{W_S}{2} \\ \frac{W_S}{2} & \frac{W_S}{2} \end{bmatrix}$$

2. **Diagonal** ([Shen et al., 2022](https://arxiv.org/abs/2107.11437)):
   $$W_L = \begin{bmatrix} W_S & 0 \\ 0 & W_S \end{bmatrix}$$

3. **Noisy symmetric**:
   $$W_L = \begin{bmatrix} \frac{W_S}{2} + \eta_1 & \frac{W_S}{2} - \eta_1 \\ \frac{W_S}{2} + \eta_2 & \frac{W_S}{2} - \eta_2 \end{bmatrix}$$

4. **Noisy diagonal**:
   $$W_L = \begin{bmatrix} W_S + \eta_1 & -\eta_1 \\ \eta_2 & W_S - \eta_2 \end{bmatrix}$$

**结果** (Figure 7):
- Symmetric 和 Noisy symmetric 最好
- Diagonal 最差（因为有 zero block）
- Noise 带来的 gain 在 symmetric 上 minimal，所以 paper 选 noise-free 版本避免多一个 hyperparameter (SNR)

我的 interpretation: Diagonal init 让 off-diagonal block 是 zero，gradient flow 不顺畅，需要额外时间把这些 zero "warm up"。Symmetric init 让所有 block 都有 signal，gradient 能立刻 flow。

### 5.2 Base Model Accuracy 的影响 (Section 3.5)

用不同训练程度的 OPT-350M checkpoint (16B, 32B, 64B tokens) init OPT-1.3B。

**结果** (Figure 8):
- 所有 HyperCloning 变体都比 random init 好
- Base model 越准确，target model 起步越好
- 但随着训练进行，gap 缩小 → 说明最终 accuracy 有 upper bound，base model 主要影响 **收敛速度**

### 5.3 Base Model Size 的影响 (Section 3.6)

OPT-5.3B 可以用:
- OPT-1.3B (2-fold cloning)
- OPT-350M (4-fold cloning)

**结果** (Figure 9): OPT-1.3B init 比 OPT-350M init 更好。因为更大的 base model 提供更好的 starting point。

这说明 **可以 stacked/recursive 使用 HyperCloning**: 350M → 1.3B → 5.3B → ...，每次都从上一个 stage 的 model init。

---

## 6. 我的 Intuition 和 Analysis

### 6.1 为什么 Function Preservation 是关键

如果不 preserve function，large model 在 step 0 的 loss 会很高（你在 distort 一个 good solution），可能比 random init 还差。Function preservation 保证 step 0 loss = small model loss，这是一个 **warm start**。

类比: 你在 mountain 上找最低点。Random init 是随机空投到一个位置。HyperCloning 是把你放在一个 "已知较低点" 附近，你只需要做 local search 或者 escape 到 nearby basin。

### 6.2 为什么 Width Expansion 而不是 Depth

Paper 明确选择 width expansion。对比 depth expansion (如 [Weight Subcloning](https://arxiv.org/abs/2312.09299), [Gong et al.](https://arxiv.org/abs/1902.07116)):

| Aspect | Width Expansion | Depth Expansion |
|---|---|---|
| Inference latency | 不变（depth 不变） | 增加（更多 sequential layer） |
| Implementation | 复杂（每层都要 clone） | 简单（复制 block） |
| Function preservation | 需要仔细设计 | Residual connection 让它自然成立 |
| Capacity increase | Per-layer capacity | Sequential computation |

Width expansion 对 **inference efficiency** 友好，这对 deployment 重要。Paper 也提到可以和 depth expansion **组合使用**。

### 6.3 Symmetry Breaking 的机制

Paper 说 symmetry 通过 dropout 等 randomness 自然 break。我的 deeper intuition:

1. **Dropout**: 每次前向传播随机 drop 不同的 neuron，让 duplicated block 接受不同的 gradient。
2. **Data diversity**: 不同 batch 的数据让 model 学到不同的 pattern，duplicated block 逐渐 specialize。
3. **Optimizer noise**: AdamW 的 momentum 和 variance estimate 在 duplicated block 间可能 diverge（虽然有 symmetry，但 floating point 和 batch 顺序的微小差异会放大）。

这个 symmetry breaking 的速度（Figure 5 显示的 cosine similarity decay）可以作为 **model health 的 diagnostic metric**。如果 symmetry 不 break，说明 model 卡在 suboptimal region。

### 6.4 Catastrophic Forcing 的解释

OLMO 实验 (base 训了 2.4T tokens) 显示初期 forgetting。我的 hypothesis:

1. **LR warmup 扰动**: 25k steps 的 warmup 让 weight 有较大扰动，打破 function preservation。
2. **新增 parameter 的 uninitialized "knowledge"**: 虽然结构上 preserve，但新增的 dimension 对 data 的 response还没有 "calibrate"。
3. **Dataset shift**: 虽然 OLMO 用同 dataset，但 data order 不同，让 model re-learn。

这个 forgetting 是 trade-off: 你要 pay 一个 initial cost 来 "re-calibrate"，但 long-term gain 是更快的 convergence 和更好的 final accuracy。

### 6.5 和 μP / Parameter Transfer 的关系

[μP](https://arxiv.org/abs/2203.03466) (Yang et al.) 也是关于 width scaling 的，但角度不同:
- μP: 调整 **init scale 和 LR** 让不同 width 的 model 训练 dynamics 一致，从而可以从 small model **tune hyperparameter**，transfer 到 large model。
- HyperCloning: 直接 **transfer weights**，不只是 hyperparameter。

理论上可以 **combine**: 用 μP 训练 small model，然后用 HyperCloning init large model，再用 μP 的 LR。

### 6.6 和 Knowledge Distillation 的区别

Paper 强调 HyperCloning 不需要改 training loop，这和 distillation 不同:
- Distillation: 加一个 distillation loss，需要 student-teacher forward，改 training setup。
- HyperCloning: 只改 init，training loop 完全不变。

这让 HyperCloning 极其 **drop-in compatible**，对工业界 adoption 友好。

### 6.7 Scaling Law 视角

[Chinchilla](https://arxiv.org/abs/2203.15556) 说 optimal training 是 model size 和 data size 1:1 ratio。HyperCloning 的 implication:

- 如果你想训一个 $N$ parameter 的 model 用 $D$ tokens，
- Random init: 需要 $D$ tokens from scratch。
- HyperCloning: 用 $\frac{N}{4}$ 的 model (random init) 训 $\frac{D}{4}$ tokens，然后 HyperCloning init $N$ model，再训 $\frac{D}{2}$ tokens。
- 总 tokens: $\frac{D}{4} + \frac{D}{2} = \frac{3D}{4}$，但 small model 训练 cheaper。

这个估算很 rough，但 paper 的实验显示 250B tokens (HyperCloning) vs 2.4T tokens (random init) 达到更好 accuracy，speedup ~4x。

### 6.8 Limitation 和 Open Questions

1. **Expansion factor 限制**: 2-fold, 4-fold 可以，但 1.5x 这种 fractional expansion 不直接支持。
2. **Width only**: 不能直接增加 depth（但可以和 [Weight Subcloning](https://arxiv.org/abs/2312.09299) 组合）。
3. **Architecture constraint**: destination 的 hidden dim 必须是 source 的 integer 倍。
4. **Catastrophic forgetting**: 论文承认这个现象，但没深入研究 mitigation。
5. **Beyond transformer**: 方法基于 linear layer + attention + norm 的 decomposition，对其他 architecture（如 Mamba, SSM）需要重新设计。

---

## 7. Related Work Context

| 方法 | Type | Function Preserving? | Notes |
|---|---|---|---|
| [Net2Net](https://arxiv.org/abs/1511.05641) (Chen 2015) | Width + Depth | Yes | 最早的 model growth 工作 |
| [Progressive Stacking](https://arxiv.org/abs/1902.07116) (Gong 2019) | Depth | No | BERT 的 layer 逐步增加 |
| [BERT2BERT](https://arxiv.org/abs/2110.07143) (Chen 2021) | Width + Depth | Partial | Encoder model |
| [LEMON](https://arxiv.org/abs/2310.07999) (Wang 2023) | Width | Yes | 但有 symmetry concern |
| [Weight Subcloning](https://arxiv.org/abs/2312.09299) (Samragh 2023) | Depth | Yes (via residual) | 重复 block |
| [Stacking your transformers](https://arxiv.org/abs/2405.15319) (Du 2024) | Depth + Width | No | 系统比较 |
| [FLM-101B](https://arxiv.org/abs/2309.03852) (Li 2023) | Depth | No | 100k budget 训 101B |
| **HyperCloning** (本文) | Width | **Yes** | Decoder transformer, drop-in |

HyperCloning 的 niche: **function-preserving width expansion for decoder LLM, drop-in compatible**。

---

## 8. Practical Recipe (我的总结)

如果我要用 HyperCloning 训一个 7B model，recipe 大概是:

1. 先训一个 1.75B 的 base model（4x smaller）用 standard recipe。
2. 用 HyperCloning 4-fold expand 到 7B：
   - Linear layer: Case 3 with $\frac{W_S}{4}$ blocks + noise。
   - Attention: 复制 head 数量（更简单）或扩大 head dimension（需要 $\sqrt{\frac{d_S}{d_D}}$ scaling）。
   - LayerNorm: 复制 $\gamma, \beta$。
   - Positional embedding: 复制 4 次。
3. 用 **相同的 training loop** 训 7B，但用更少的 tokens。
4. 监控 cosine similarity 来确认 symmetry breaking。
5. 监控 singular value distribution 来确认 rank recovery。

预期: ~3-4x speedup，更好的 final accuracy。

---

## 9. Open Directions (paper 提到 + 我的想法)

1. **Catastrophic forgetting mitigation**: 用 EWC、rehearsal、或 lower initial LR 来缓解初期 forgetting。
2. **Depth + Width combination**: 先 HyperCloning width，再 Weight Subcloning depth，或反之。
3. **Non-integer expansion**: 用 interpolation 或 low-rank approximation 支持 1.5x 等任意 expansion。
4. **Adaptive expansion**: 不同 layer 用不同 expansion factor（有的 layer 需要更多 capacity）。
5. **Online growth**: 训练过程中动态 grow，而不是一次性 init。
6. **Beyond transformer**: Mamba, RWKV, hybrid architecture 的 expansion。
7. **Theory**: 为什么 function-preserving init 在 loss landscape 上更好？是否有 formal guarantee？

---

## References

- **HyperCloning paper**: [arxiv.org/abs/2410.14914](https://arxiv.org/abs/2410.14914)
- Net2Net: [arxiv.org/abs/1511.05641](https://arxiv.org/abs/1511.05641)
- Weight Subcloning: [arxiv.org/abs/2312.09299](https://arxiv.org/abs/2312.09299)
- Stacking your transformers: [arxiv.org/abs/2405.15319](https://arxiv.org/abs/2405.15319)
- LEMON: [arxiv.org/abs/2310.07999](https://arxiv.org/abs/2310.07999)
- BERT2BERT: [arxiv.org/abs/2110.07143](https://arxiv.org/abs/2110.07143)
- Progressive Stacking: [arxiv.org/abs/1902.07116](https://arxiv.org/abs/1902.07116)
- Pythia: [arxiv.org/abs/2301.09041](https://arxiv.org/abs/2301.09041)
- OLMO: [arxiv.org/abs/2402.00838](https://arxiv.org/abs/2402.00838)
- OPT: [arxiv.org/abs/2205.01068](https://arxiv.org/abs/2205.01068)
- Chinchilla: [arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- Scaling Laws: [arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
- μP: [arxiv.org/abs/2203.03466](https://arxiv.org/abs/2203.03466)
- FLM-101B: [arxiv.org/abs/2309.03852](https://arxiv.org/abs/2309.03852)
- Llama 3: [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)
- ZeRO: [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- Glorot init: [proceedings.mlr.press/v9/glorot10a](https://proceedings.mlr.press/v9/glorot10a.html)
- lm-evaluation-harness: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- DOLMA dataset: [allenai.org/dolma](https://allenai.org/dolma)

---

**Final intuition**: HyperCloning 本质上是把 "训练 large model from scratch" 这个 expensive optimization 问题，转化成 "训练 small model + 在 good basin 附近 refine large model" 这个 cheaper 问题。Function preservation 是关键 trick，确保你不会 lose 已经学到的 knowledge。Symmetry breaking 和 rank recovery 证明 large model 最终 fully utilize 了它的 capacity，没有卡在 "当 small model 用" 的 suboptimal region。这个方法和 scaling law、distillation、depth growth 是互补的，可以组合成更高效的训练 pipeline。
