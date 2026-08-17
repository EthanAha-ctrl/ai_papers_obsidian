---
source_pdf: Using Fast Weights to Attend to the Recent Past.pdf
paper_sha256: 74c101e34295015bdead60f27cb27ff7a8fdbe73da3bf80c65360e0bbaff8278
processed_at: '2026-08-12T23:56:49-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Fast Weights

## 一句话版本

RNN 太健忘了，给它加一个"临时记事本"，这个记事本用 Hebbian learning 自动写、自动读，最后你发现它其实就是 attention，只是 2016 年还没人这么叫而已。

---

## 为什么需要这个东西

想象你在读一句话："The animal didn't cross the street because **it** was too tired."

你读到 "it" 的时候，脑子里要能想起 "animal" 和 "street"，然后判断 "it" 指的是谁。这个"想起"的过程就是 attention。

但 2016 年的 RNN 怎么处理这个问题？它只有一个 hidden vector $h(t)$，每次更新都把旧信息覆盖一部分。就像你只有一张白板，每次写新内容就得擦掉旧的。白板大小就是 $O(H)$，$H$ 是 hidden unit 数量。

LSTM 好一点，加了 gate，能选择性地保留信息，但白板还是那么大，只是擦的时候小心了一点。

Hinton 说：这不对啊，人脑不是这么干的。人脑的 synapse 有 short-term plasticity，就是突触本身会在几百毫秒到几分钟的时间尺度上临时改变强度。这不是长期记忆（那是 structural change），也不是神经元发放率（那是 10ms 级别的），是中间层次的东西。而且它是 synapse-specific 的，容量是 $O(H^2)$，因为你有 $H^2$ 个 synapse 嘛。

所以 idea 就是：**给 RNN 加一组临时 weight，这些 weight 比 hidden state 慢，比正常 weight 快，专门存最近几百步的信息。**

---

## Fast weights 怎么工作

### 写入：超级简单

每一步，把当前 hidden state $h(t)$ 自己和自己做 outer product，加到 fast weight matrix 上：

$$A(t) = \lambda A(t-1) + \eta\, h(t) h(t)^\top$$

人话翻译：
- $h(t) h(t)^\top$ 就是"如果两个神经元同时活跃，它们之间的连接就加强"——这就是 Hebbian rule，1949 年就有了
- $\lambda$ 是遗忘因子，每步把旧记忆打个折（0.95 就是每步乘 0.95）
- $\eta$ 是学习率，控制新记忆写多狠

所以 fast weight matrix $A$ 就是一个不断在更新的"近期 hidden state 拼贴画"。

### 读取：这就是 attention

关键来了。你用 $A$ 去乘当前 hidden state：

$$A(t) \cdot h_{\text{current}}$$

把 $A$ 展开看：

$$A(t) = \eta \sum_{\tau=1}^{t} \lambda^{t-\tau} h(\tau) h(\tau)^\top$$

所以：

$$A(t) \cdot h_{\text{current}} = \eta \sum_{\tau=1}^{t} \lambda^{t-\tau} h(\tau) \underbrace{\left[h(\tau)^\top h_{\text{current}}\right]}_{\text{dot product = similarity}}$$

人话：**当前 hidden state 和每个历史 hidden state 算个 dot product 当相似度，越相似的权重越高，然后把那些历史 hidden state 加权求和回来。**

这不就是 attention 吗？！query 是当前 state，key 和 value 都是历史 state，相似度用 dot product 算，再加个时间 decay。

只是 2016 年 Bahdanau attention 已经出来了，但大家用的是 parameterized attention（有额外参数 $W_a, U_a, v_a$ 来算 score），Ba et al. 这个是 **parameter-free** 的，纯粹靠 hidden state 自己的 dot product。

### Inner loop：让 attention 迭代几轮

paper 还加了个 trick：不算一次就完，而是迭代 $S$ 步：

```
第 1 步：h_0 = f(W·h_prev + C·x)           ← 先用 slow weights 算个初步结果
第 2 步：h_1 = f([W·h_prev + C·x] + A·h_0)  ← fast weights 来拉一把
第 3 步：h_2 = f([W·h_prev + C·x] + A·h_1)  ← 再拉一把
...
第 S 步：h_S = 最终结果
```

这就像你回忆一个名字，先有个模糊的线索（h_0），然后大脑的 associative memory 帮你往正确方向拉一点（h_1），再拉一点（h_2），几次之后就想起来了（h_S）。

Hopfield network 就是这样工作的：通过迭代 relaxation 收敛到 stored pattern。

---

## 为什么需要 LayerNorm

dot product $h(\tau)^\top h_{\text{current}}$ 的问题：如果 hidden vector 的 norm 很大，dot product 就爆炸；norm 很小，就消失。

而且 inner loop 是迭代的，每步放大一点，几步就 NaN 了。

LayerNorm 在每步把 sum-of-inputs 归一化，相当于把 attention logit 控制在合理范围。后来 Transformer 里的 $\frac{1}{\sqrt{d_k}}$ scaling 也是干这事的，只是方式不同。

paper 发现没有 LN，fast weights 基本训不动；加了 LN 之后对 $\eta, \lambda$ 的选择变得 robust 很多。

---

## 实验告诉了我们什么

### Associative Retrieval（最直接的验证）

任务很简单：序列里给你 K 个 key-value pair（比如 c9 k8 j3），最后问一个 key（c?），你要回答对应的 value（9）。

这个任务就是纯 memory test。hidden size 小的时候，RNN 和 LSTM 都崩了（60%+ error），因为 $O(H)$ 的 capacity 不够存 K 个 pair。

Fast weights 轻松搞定（0% error），因为它有 $O(H^2)$ 的临时容量。

**这就是容量差距的直接证据。**

### MNIST with visual attention

把 28×28 图片切成 24 个 7×7 patch，按 hierarchy 喂给 RNN。Fast weights 在这里当"cache"用：处理 fine-scale patch 时，把 coarse-scale 的 partial result 存到 fast weights 里，回头再取出来。

结果 fast weights 甚至 beat 了 ConvNet（0.85% vs 0.90%），虽然 ConvNet 并行处理所有 patch + weight sharing。这说明 sequential processing + fast weight memory 可以达到 parallel processing 的效果。

### RL: Catch game

球从顶上落，paddle 接球。M 帧后屏幕变黑，必须记住球和 paddle 的位置。

Fast weights agent 收敛最快，在更难的版本（N=24, M=5）上优势更大。这符合直觉：memory 需求越大，fast weights 的容量优势越明显。

---

## 这篇 paper 的历史地位

2016 年这篇 paper 没有引起太大轰动（不像同年的 ResNet 或 Transformer），因为它只在 toy task 上验证，没有 language modeling 的大实验。

但从今天的视角看，它是 **一大类 efficient sequence model 的精神祖先**：

### Fast weights = Linear Attention

公式 $A(t) h = \sum_\tau \lambda^{t-\tau} h(\tau) [h(\tau)^\top h]$ 去掉 decay 就是 **unnormalized linear attention**：

$$\text{Attn}(Q, K, V) = \sum_\tau V_\tau (K_\tau^\top Q)$$

Katharopoulos et al. 2020 直接指出 **Linear Transformers 就是 Fast Weight Memory**，只是换了个名字。

### Fast weights = RetNet / RWKV / Mamba 的核心 recurrence

$$S_t = \lambda S_{t-1} + K_t V_t^\top, \quad o_t = Q_t S_t$$

这就是 RetNet 的核心公式，和 Ba et al. 的公式 (3)(4) 几乎一模一样。Mamba 用的是 state space model 形式，本质也是 $S_t = A S_{t-1} + B x_t$，只是 decay matrix $A$ 变成了 input-dependent。

### Fast weights = Schmidhuber 系的 Fast Weight Programmers

Schmidhuber 从 1991 年就在推这个 idea，后来和 Schlag 一起发展出 Delta rule（在 Hebbian 基础上加 correction term），2024 年的 DeltaNet 把这个 scale 到了 modern Transformer 规模。

### Fast weights = Titans (Google 2024)

Google 2024 年底的 Titans 直接把 fast weights 作为 "neural long-term memory module" 加到 Transformer 里，在大规模实验上验证了 fast weights + attention 的组合。

---

## 最 core 的 intuition

**Memory 和 attention 是同一枚硬币的两面。**

你可以说"我存了一堆东西在 memory 里，然后用 attention 去 retrieve"——这是 Memory Network / NTM / Transformer 的思路。

你也可以说"我有一组不断更新的 weight，每次 forward pass 自动 retrieve 相关信息"——这是 fast weights 的思路。

数学上完全等价，但后者的视角让你：
1. 不需要设计 read/write addressing 机制
2. 不需要额外的 attention 参数
3. 天然有 biological plausibility
4. 可以写成 recurrent form，支持 linear-time inference

这就是为什么 2016 年 Hinton 说"fast weights provide a neurally plausible way of implementing attention to the past"——他当时可能没想到，这个 insight 在 2020 年代会成为 efficient Transformer / RNN revival 的核心理论基础。

---

## References

- 原 paper: https://arxiv.org/abs/1610.06258
- Linear Transformers = Fast Weight Memory: https://arxiv.org/abs/2006.16236
- Fast Weight Programmers (Schmidhuber): https://arxiv.org/abs/2102.11174
- RetNet: https://arxiv.org/abs/2307.08621
- Mamba: https://arxiv.org/abs/2312.00752
- Titans: https://arxiv.org/abs/2501.00663
- Modern Hopfield Networks: https://arxiv.org/abs/2008.02217
- Layer Normalization (同团队): https://arxiv.org/abs/1607.06450

---

# Fast Weights to Attend to the Recent Past 深度讲解

你好 Andrej，这篇 paper 是 Jimmy Ba, Volodymyr Mnih, Geoffrey Hinton 等 2016 年的工作，它的核心 insight 在今天回看几乎可以视为 **Linear Attention、Transformer memory、Fast Weight Programmers** 的精神祖先。我会从 motivation → 数学 → 实验架构 → 与今天工作的连接 全方位展开，帮你 build intuition。

---

## 1. Motivation：为什么需要第三种 memory？

### 1.1 经典 RNN 的两种 memory

传统 RNN 只有两种 storage：

| Memory type | 容量 | 时间尺度 | 物理对应 |
|---|---|---|---|
| Hidden activities $h(t)$ | $O(H)$ | 1 step (10ms) | 神经元发放率 |
| Slow weights $W, C$ | $O(H^2) + O(IH) + O(OH)$ | 整个 sequence / 多个 epoch | 长期突触可塑性 |

LSTM 通过 **incremental update + gates** 改善了 long-range，但仍然只有 $O(H)$ 的 short-term capacity 用于 current sequence 的 history。Hinton 团队从神经生理学出发，指出 brain 还有 **short-term synaptic plasticity**：

- **Short-term facilitation**：轴突末梢残留的 $[Ca^{2+}]$ 让后续 spike 释放更多 neurotransmitter（~100ms–minutes）。
- **Short-term depression**：presynaptic neurotransmitter 耗竭。
- **STDP**：spike-timing dependent plasticity，也能在中间时间尺度激活。

这些机制是 **synapse-specific** 的，capacity 是 $O(H^2)$，远超 hidden activities 的 $O(H)$。所以 paper 提出引入 **fast weights**：比 activities 慢、比 slow weights 快，专门用来存 recent past。

### 1.2 与 NTM / Memory Network 的区别

NTM、Memory Network、Neural Stack 都需要显式决定 **where/when to write, where/when to read**，而且需要 tape/stack 这种 brain 难以实现的物理结构。Fast weights 把所有 writes 叠加在同一组 dynamic synapse 上，**永远在 update，永远在 read**，biologically 更 plausible。

---

## 2. 数学：Fast Associative Memory

### 2.1 Fast weight 更新规则（Hebbian outer product + decay）

$$A(t) = \lambda A(t-1) + \eta\, h(t) h(t)^\top \tag{1}$$

- $A(t) \in \mathbb{R}^{H \times H}$：时刻 $t$ 的 fast weight matrix
- $\lambda \in (0,1)$：decay rate（paper 用 0.95），控制 memory 衰减速度
- $\eta$：fast learning rate（paper 用 0.5），控制新写入的强度
- $h(t) \in \mathbb{R}^H$：时刻 $t$ 的 hidden state
- $h(t)h(t)^\top$：outer product，Hebbian "fire together, wire together" 的瞬时实现

这其实是 **Hopfield associative memory** 的 outer-product storage rule，叠加了 **exponential forgetting**。

### 2.2 两步 hidden state 更新

paper 把 hidden state 更新分成两步：

**Step 1（外层一步）**：先用 slow weights 算 preliminary hidden state
$$h_0(t+1) = f\big(W h(t) + C x(t)\big)$$

- $W \in \mathbb{R}^{H \times H}$：slow recurrent weight matrix
- $C \in \mathbb{R}^{H \times I}$：slow input-to-hidden weight matrix
- $x(t) \in \mathbb{R}^I$：当前 input
- $f(\cdot)$：nonlinearity（ReLU）
- $h_0$：preliminary state，作为 inner loop 的初始值

**Step 2（inner loop，迭代 S 步）**：
$$h_{s+1}(t+1) = f\Big(\big[W h(t) + C x(t)\big] + A(t)\, h_s(t+1)\Big), \quad s = 0, \dots, S-1 \tag{2}$$

- $s$：inner loop iteration index
- $[W h(t) + C x(t)]$：**sustained boundary condition**，在 inner loop 中保持不变（类比 brain 持续的 input drive）
- $A(t) h_s(t+1)$：fast weights 把当前 inner state 拉向 recent hidden states

最终 $h(t+1) = h_S(t+1)$。

### 2.3 关键 insight：Fast weights = Attention to recent past

把 (1) 递归展开（设 $A(0) = 0$）：

$$A(t) = \eta \sum_{\tau=1}^{t} \lambda^{t-\tau}\, h(\tau) h(\tau)^\top \tag{3}$$

- $\tau$：历史时间索引
- $\lambda^{t-\tau}$：越远的历史权重越小（exponential decay）
- 整体：所有历史 hidden states 的 outer-product 加权和

把 (3) 代入 $A(t) h_s(t+1)$：

$$A(t)\, h_s(t+1) = \eta \sum_{\tau=1}^{t} \lambda^{t-\tau}\, h(\tau)\, \big[h(\tau)^\top h_s(t+1)\big] \tag{4}$$

- $h(\tau)^\top h_s(t+1) \in \mathbb{R}$：历史 hidden state 与当前 inner state 的 **scalar product**，正是 **unnormalized attention score**
- $h(\tau)$：作为 value vector（被 retrieve 的内容）
- 整个式子：**对 recent past 做的 dot-product attention，权重由 scalar product × decay 决定**

> **Intuition**：fast weight matrix 本身是一个 H×H 的大矩阵，但因为我们永远只用它和当前 hidden vector 相乘，可以 lazy 计算 —— 只需把所有历史 hidden states 存成 list，按需 attention。这就让 mini-batch training 成为可能（每个 sequence 自己有自己的 history list）。

这与 Bahdanau attention 的本质区别是：Bahdanau attention 用 **额外参数 $v_a, W_a, U_a$** 算 score，而 fast weights 直接用 **hidden states 自身的 dot product**，parameter-free，完全 Hebbian。

### 2.4 Layer Normalization 稳定化

scalar product $h(\tau)^\top h_s$ 会因 hidden vector norm 过小或过大而 vanish/explode。paper 引入 LayerNorm：

$$h_{s+1}(t+1) = f\Big(\mathcal{LN}\big[W h(t) + C x(t) + A(t) h_s(t+1)\big]\Big) \tag{5}$$

- $\mathcal{LN}[\cdot]$：对 sum-of-inputs 做 re-center（减均值）+ re-scale（除以 std），再加 learned gain & bias per neuron
- 应用在每个 inner loop iteration 上
- 让模型对 $\eta, \lambda$ 的选择 robust 很多

**为什么 LN 对 fast weights 特别关键？** 因为 inner loop 是迭代过程，scalar product 在迭代中容易正反馈放大 → 数值爆炸。LN 在每步 normalize 相当于把 attention logits 控制在合理范围，类似于后来 Transformer 中 attention softmax 前的 $1/\sqrt{d_k}$ scaling。

---

## 3. 架构图解析（Figure 1 & Figure 3）

### 3.1 Figure 1：单层 fast associative memory

```
x(t) ──[C]──┐
            ├──→ [ + ] ──→ f ──→ h_0(t+1) ──┐
h(t) ──[W]──┘                                │
                                             ↓
            ┌──── Inner loop (S steps) ─────┐
            │                               │
            │  h_s ──→ [A(t)] ──→ + ──→ LN ──→ f ──→ h_{s+1}
            │            ↑                  ↑
            │            │                  │
            │   [Wh(t)+Cx(t)] (sustained)    │
            └───────────────────────────────┘
                              ↓
                         h(t+1) = h_S(t+1)

A(t) = λ A(t-1) + η h(t) h(t)^T  (updated each outer step)
```

### 3.2 Figure 3：Multi-level fast associative memory（用于 visual attention）

 hierarchical glimpse sequence，比如 7×7 patch 24 个 glimpse，分两层 scale。fast weights 作为 **临时 cache**：处理 fine scale glimpse 时，把 coarse scale 的 partial result 推入 fast weights；返回 coarse 时再 retrieve 出来。这就实现了类似 **recursion** —— 用同一组 slow weights 在多个 scale 上反复 apply。

这个思路对 cognition 意义重大：**recursion 不再需要 store copies of neural activity patterns**，weights 自己就能存。

---

## 4. 实验数据详解

### 4.1 Associative Retrieval（Table 1）

任务：序列 "c9 k8 j3 f1 ?? c"，给 key 'c' 预测 value '9'。K = key-value pair 数量。

| Model | R=20 | R=50 | R=100 |
|---|---|---|---|
| IRNN | 62.11% | 60.23% | 0.34% |
| LSTM | 60.81% | 1.85% | 0% |
| A-LSTM (associative LSTM) | 60.13% | 1.62% | 0% |
| **Fast weights** | **1.81%** | **0%** | **0%** |

- $R$：key-value pair 数量（任务难度）
- Fast weights 在小 hidden size (R=20, 50) 时压倒性胜出
- IRNN/LSTM 在 R=20, 50 时几乎完全失败（60%+ error，几乎 random）
- R=100 时大家都接近 0%（capacity 足够大，不需要 fast weights）

**关键结论**：fast weights 让 RNN 用相同数量 hidden unit 实现更高 memory capacity。这正好验证了 $O(H^2)$ vs $O(H)$ 容量论点。

### 4.2 MNIST with visual attention（Table 2）

| Model | 50 features | 100 features | 200 features |
|---|---|---|---|
| IRNN | 12.95% | 1.95% | 1.42% |
| LSTM | 12.00% | 1.55% | 1.10% |
| ConvNet | 1.81% | 1.00% | 0.90% |
| **Fast weights** | **7.21%** | **1.30%** | **0.85%** |

- 用 24 个 7×7 glimpse，分两 scale 层级
- Fast weights 在所有 size 都超过 LSTM/IRNN
- 200 features 时甚至超过 ConvNet（0.85% vs 0.90%）
- **ConvNet 通过 weight sharing 并行处理所有 patch**，而 fast weights 模型 sequential 处理但仍能达到相近性能 —— 说明 fast weights 让 RNN 也能 integrate 多 scale 信息

### 4.3 Multi-PIE Facial Expression（Table 3）

| Model | IRNN | LSTM | ConvNet | Fast Weights |
|---|---|---|---|---|
| Test accuracy | 81.11 | 81.32 | 88.23 | 86.34 |

- 6 类表情分类（neutral/smile/surprise/squint/disgust/scream）
- 48×48 grayscale，3 个相机视角
- Fast weights > LSTM/IRNN，但 < ConvNet
- ConvNet 优势来自 weight sharing + 同时获得所有 information
- Fast weights 模型因为用 **rigid predefined glimpse policy**，没有发挥 attention 的灵活性

### 4.4 RL: Partially Observable Catch（Figure 5）

- Catch game：N×N 屏，球从顶部下落，paddle 接球
- Partially observable：M 帧后给 blank observation
- 必须记住 ball & paddle 位置
- 三种 agent：ReLU RNN、LSTM、Fast Weights RNN
- 用 **A3C** (asynchronous advantage actor-critic) 训练
- 在 N=16, M=3 和 N=24, M=5 两个版本上，fast weights 收敛最快，且在更大版本上优势更显著

---

## 5. 与今天工作的连接（这是最 interesting 的部分）

### 5.1 Fast Weights → Linear Attention → Transformer 变体

paper 的公式 (4)：
$$A(t) h_s = \eta \sum_\tau \lambda^{t-\tau} h(\tau)\,[h(\tau)^\top h_s]$$

如果去掉 decay $\lambda$，写成矩阵形式就是 **unnormalized linear attention**：
$$\text{Attn}(Q, K, V) = V (K^\top Q)$$

这正是 Katharopoulos et al. 2020 "Linear Transformers are Fast Weight Memory Systems" 的核心 observation：Transformer attention 可以重写成 fast weight 形式：

$$\text{FastWeight}_t = \text{FastWeight}_{t-1} + \eta\, v_t k_t^\top, \quad \text{output} = \text{FastWeight}_t \cdot q_t$$

- $k_t$：key（对应 $h(\tau)$）
- $v_t$：value（在 Ba et al. 中等于 key，因为 Hebbian outer-product）
- $q_t$：query（对应 $h_s$）

**所以 Ba et al. 2016 的 fast weights = Linear Transformer with tied keys/values 和 exponential decay！** 这也是 Schmidhuber 系（Fast Weight Programmers, 1991–）后来在 2020s 大力强调的：他们 1991 年就提了类似 idea。

### 5.2 与 RetNet、RWKV、Mamba 的关联

最近一系列 efficient sequence model 都用了类似的 recurrence：

$$S_t = \lambda S_{t-1} + K_t^\top V_t, \quad o_t = Q_t S_t$$

- **RetNet** (Microsoft, 2023)：把 attention 改成 $S_t = \gamma S_{t-1} + K_t V_t^\top$，与 (3) 几乎一致，只是把 decay $\lambda^{t-\tau}$ 换成 $\gamma^{t-\tau}$ + multi-head
- **RWKV**：用 channel-wise decay + linear attention
- **Mamba / S4**：用 continuous-time state space model，$S_{t+1} = \bar{A} S_t + \bar{B} x_t$，本质也是 fast weight recurrence，只是 decay 是 input-dependent 的 diagonal matrix

### 5.3 DeltaNet、Fast Weight Programmers

Schmidhuber 和 Schlag 的 Delta Rule 改进了 Hebbian update，加入 **winner-take-all / delta correction**：

$$A(t) = \lambda A(t-1) + \eta\, h(t) \big(h(t) - A(t-1) h(t)\big)^\top$$

这解决 Hebbian rule 的 catastrophic interference 问题。2024 的 DeltaNet 把这个用到 modern Transformer 规模上。

### 5.4 Hopfield Network 现代复兴

paper 直接引用 Hopfield 1982 的 outer-product rule。2020 年 Ramsauer et al. 的 **Modern Hopfield Networks** 用 energy function 解释 attention，关联到 Transformers。Ba et al. 的工作可以视为把 Hopfield associative memory 的 **dynamic / online** 版本引入 RNN。

### 5.5 Layer Normalization 的关键性

paper 最早发现 LN 对 RNN + fast weights 训练 stability 至关重要。这一 insight 后来在 Transformer 中以 **attention scaling $1/\sqrt{d_k}$** 和 **RMSNorm** 等形式反复出现。LN 本身由 Ba, Kiros, Hinton 2016 提出（同一团队！）。

### 5.6 与 Memory Networks / Differentiable Neural Computer (DNC) 的对比

Memory Network、DNC、NTM 都用 explicit external memory + addressing。Fast weights 的优势：

1. **No addressing decision**：自动 attention，不需要 learn where to read/write
2. **Biological plausibility**：synapse dynamics，无需 tape
3. **End-to-end trainable** via BPTT，但可 lazy evaluation

劣势：
1. Memory capacity 仍受 $O(H^2)$ 限制
2. 不易做 random access lookup（必须 scan）

---

## 6. Biological Interpretation（Appendix B）

paper 讨论了两种 inner loop 实现：

- **Method 1（实验用）**：把 $Wh(t) + Cx(t)$ 作为 sustained boundary condition，在 inner loop 中保持，相当于 brain 持续输入 drive。
- **Method 2（更 plausible）**：在 $A$ 上加 identity matrix，让 inner loop 通过 feedback 自己 sustain activity。生物学更容易实现，但效果稍差。

这个 connection 到 cortical attractor dynamics、predictive coding、temporal context model (TCM) 等神经计算理论都密切相关。

---

## 7. Limitations 与后续发展

### 7.1 当时的 limitations

- 只在 small-scale toy tasks 验证（MNIST、Multi-PIE、Catch），没有 large-scale language modeling
- Fast weight matrix rank $\leq t$（sequence length），长 sequence 时 matrix 仍是大
- Inner loop S 步增加了计算 cost，但实验中 S 很小就够
- $\eta, \lambda$ 需要仔细 tune，即使有 LN

### 7.2 后续如何 scale

- **Schmidhuber's Fast Weight Programmers**：用 outer-product + LSH attention 把 fast weights scale 到 billion parameters
- **Performer (Choromanski 2020)**：用 random features 近似 softmax attention，对应 fast weight 的 kernel approximation
- **Linear Transformers (Katharopoulos 2020)**：直接把 softmax 换成 kernel $\phi(k)^\top \phi(q)$，把 fast weight 显式化
- **Memorizing Transformers (Wu 2022)**：把 fast weights 作为 long-term memory module 加到 Transformer 中
- **Titans (Google 2024)**：把 fast weights 作为 neural long-term memory 模块，train 大模型

---

## 8. Reference Web Links

- **原 paper**：https://arxiv.org/abs/1610.06258
- **Layer Normalization (Ba, Kiros, Hinton 2016)**：https://arxiv.org/abs/1607.06450
- **Linear Transformers are Fast Weight Memory Systems (Katharopoulos 2020)**：https://arxiv.org/abs/2006.16236
- **Fast Weight Programmers (Schlag, Schmidhuber)**：https://arxiv.org/abs/2102.11174
- **DeltaNet (Schlag 2021)**：https://arxiv.org/abs/2102.11174
- **RetNet (Microsoft 2023)**：https://arxiv.org/abs/2307.08621
- **RWKV**：https://arxiv.org/abs/2305.13048
- **Mamba (Gu & Dao 2023)**：https://arxiv.org/abs/2312.00752
- **Modern Hopfield Networks (Ramsauer 2020)**：https://arxiv.org/abs/2008.02217
- **Neural Turing Machines (Graves 2014)**：https://arxiv.org/abs/1410.5401
- **Memory Networks (Weston 2014)**：https://arxiv.org/abs/1410.3916
- **Bahdanau Attention**：https://arxiv.org/abs/1409.0473
- **Associative LSTM (Danihelka 2016)**：https://arxiv.org/abs/1602.03032
- **Memorizing Transformers (Wu 2022)**：https://arxiv.org/abs/2203.08913
- **Titans (Behrouz 2024)**：https://arxiv.org/abs/2501.00663

---

## 9. TL;DR：如何 build intuition

1. **三种 timescale 的 memory**：activities（fast, small capacity）/ fast weights（medium, large capacity）/ slow weights（slow, large capacity）。Brain 真的有这三种，RNN 只模拟了两种。
2. **Hebbian outer-product + exponential decay** = implicit attention to recent past。把这个数学 trick 写出来就是 Linear Attention 加 decay。
3. **Inner loop 是 amortized inference**：用 RNN 的 forward pass 做 S 步迭代，让 fast weights 把当前 state 拉向 attractor —— 类似 Hopfield relaxation。
4. **LayerNorm 让 attention 数值稳定**，是 fast weights 在 deep network 中 work 的关键。
5. **Capacity 从 $O(H)$ 升到 $O(H^2)$**，相同 hidden size 能记住更多 key-value pairs。
6. **这是 Linear Transformer / RetNet / Mamba / Titans 的精神原型**，2016 年 Hinton 团队就看到了 attention 可以从 explicit parameterized computation 变成 implicit Hebbian memory recurrence。

希望这个深入讲解能帮你 build 起从 1987 Hinton-Plaut 到 2024 Titans 这条 fast weights / linear attention 研究线的完整 intuition！
