---
source_pdf: VLAKnowsIts Limits.pdf
paper_sha256: 87be8adc046971056b5039e292811a9bc3bb387aed928d02b680643519280900
processed_at: '2026-08-13T02:52:59-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好，抛开那些公式和 notation，咱就说最核心的事儿。

## 这论文到底在说啥

做 robot policy 的现在都流行 **action chunking**——模型一次预测一串动作，比如 50 个连续的 motor command，然后 robot 只执行前面几个就重新看摄像头、重新规划。这个"执行几个"的数字，paper 里叫 **execution horizon**（$e$）。

问题来了：这个 $e$ 到底设多少，大家都是**拍脑袋**定的。有人设 5，有人设 10，有人设 50，全靠调参试出来。

这篇 paper 说：这事儿没那么简单，$e$ 选错了性能能差 20-30 个百分点，而且**不同时刻的最优 $e$ 不一样**。所以作者搞了个方法，让模型自己告诉自己"我这几步靠谱，后面我不信了"。

## 先说为啥这事儿重要

你看 Table 1 里 $\pi_{0.5}$ 在 LIBERO 上的数据，$p=50$（预测 50 步）的时候：

- 执行 10 步（$e=10$）：成功率 94.9%
- 执行 50 步（$e=50$，全执行）：71.5%

差了 23 个点！就因为多执行了几步。

更扎心的是，不同 task 的最优 $e$ 完全不同（Table 6）：
- LIB-Spatial 最爱 $e=5$
- LIB-Object 最爱 $e=13$
- LIB-Goal 最爱 $e=15$

没有通用解，一个固定值搞不定所有场景。

## 打个比方

就像 GPS 导航给你规划了 50 步路线：
- 你完全照着走（$e=50$）：可能路上堵车了你还闷头往前开，越走越偏
- 你每走 1 步就重新规划（$e=1$）：路线一直在跳，车开得抖抖索索
- 中间某个值最舒服

而且——高速上你可以多走几步再重新规划（路况稳定），市中心得频繁重规划（变化快）。**一个固定值搞不定所有路况**。

## 作者怎么发现这个问题的

他们去看 $\pi_{0.5}$ 内部的 **attention weights**，发现两件特别有意思的事：

### 发现 1：chunk 里的 action 们看的是同一张"图"

正常你想，预测第 1 个 action 和第 30 个 action，模型关注的东西应该不一样吧？毕竟环境变了嘛。

**不是**。Figure 3 显示，chunk 里所有 action 对 vision tokens 的 attention pattern 几乎一模一样。第 1 个 action 看哪块图像区域，第 30 个 action 还看那块区域。

这说明啥？模型预测后面那些 action 的时候，**根本没在 adapt**。它就是在第 1 个 action 的时候看了一眼图，然后后面 49 个 action 全靠"惯性"推出来。环境要是变了，后面那些 action 就废了。

> 顺便还发现一个 language attention sink——第一个 language token 收获了超高 attention，但 Table 8 做实验把 language attention 全 mask 掉，性能几乎不掉。说明 $\pi_{0.5}$ 的 vision-language pretraining 已经把语言信息压进 vision representation 里了，显式看 language token 基本是多余的。

### 发现 2：首尾两个 action 像两个"锚点"

Figure 4 是 action 对 action 的 self-attention，发现：
- 第 1 个 action token 和最后 1 个 action token 被所有人盯着
- 中间的 action 对这两个 boundary 的 attention 一开始挺高，然后迅速衰减
- 衰减完就进入一个 low-attention plateau（大家都差不多低）

作者管这叫 **radial action sinks**，像两个黑洞把 attention 都吸过去了。

为啥？两个直觉解释：
- 初始 action 是模型最有信心的，error 最小，后面 action 拿它当"基准锚"
- 训练时 expert demo 从随机 timestamp 切片，首尾 action 要保证 chunk 之间能衔接上，模型学会把它们当 continuity anchor

### 把两个发现合起来

- 后面 action 不 adapt 环境（发现 1）
- 但它们一开始还会盯着首尾 anchor（发现 2）
- 当 attention 对 anchor 衰减了，说明模型开始"自言自语"——只看自己前面生成的 action，不再 grounded 到感知

**这就是模型"自己知道自己的极限"的信号**。attention 衰减点 = 模型信自己的边界。

## AutoHorizon 怎么做

核心想法：**从 attention pattern 里读出这个衰减点，把它当 execution horizon**。

具体步骤用人话说：

**Step 1**: 拿到 action 之间的 attention matrix（$p \times p$），每行归一化。

**Step 2**: 算每一行的 entropy。entropy 低 = attention 集中 = 信息量大；entropy 高 = attention 散开 = 没主见。只保留低 entropy 的行（取最低 10%，$q=0.9$）。这一步是过滤掉那些"我不知道该看哪"的 action，留下有明确 opinion 的。

**Step 3**: 对每一行算一个"期望位置" $\mu_t[i]$——就是这一行的 attention 主要指向第几个 action。比如 row $i$ 主要 attend 到 row $i+5$，那 $\mu_t[i] \approx i+5$，意思是模型觉得"我能可靠地预测到 5 步以后"。加个 cumulative max 保证单调递增。

**Step 4**: 看相邻行的 $\mu$ 差值 $\Delta\mu_t[i]$。当差值小于 threshold $\tau=0.3$，说明 attention "不往前走了"——plateau 开始了。

**Step 5**: 找第一个同时满足"低 entropy"和"plateau 开始"的行，它的 $\mu$ 值就是 forward horizon $N_f$。

**Step 6**: 反向再扫一遍（从 chunk 末尾往前），得到 backward horizon $N_b$。

**Step 7**: 如果 $N_f + N_b \geq p$，说明整个 chunk 都被两个 sink 的 attention 覆盖了，模型全程 confident，$N = p$ 全执行；否则只执行 $N_f$ 步。

### 直觉上

两个 pointer 从 chunk 两头往中间扫：
- Forward pointer 从开头找"初始 anchor 的 attention 还能撑多远"
- Backward pointer 从末尾找"终止 anchor 的 attention 还能撑多远"
- 如果两头覆盖了整个 chunk → 全执行
- 如果中间有断档 → 只执行前半截

## 公式细节（满足你的要求）

归一化 entropy（Eq. 5）：
$$H_t[i] = -\frac{1}{\log p} \sum_j \mathbf{S}_t[i,j] \log \mathbf{S}_t[i,j]$$

- $H_t[i]$: 第 $i$ 个 action query 的 normalized entropy
- $\mathbf{S}_t[i,j]$: 第 $i$ 个 query 对第 $j$ 个 key 的 attention weight
- $p$: prediction horizon，$\log p$ 做 normalization 让结果落在 $[0,1]$

Forward expected position（Eq. 6）：
$$\mu_t[i] = \max\left(\sum_{j=0}^{p-1} j \cdot \mathbf{S}_t[i,j], \max_{k \leq i} \mu_t[k]\right)$$

- $\mu_t[i]$: 第 $i$ 行的 expected look-ahead position
- $j \cdot \mathbf{S}_t[i,j]$: 位置 $j$ 乘以 attention weight，加权求和得期望
- 外层 $\max(\cdot, \max_{k \leq i} \mu_t[k])$: 强制单调，防止 backward jump

Plateau detection（Eq. 7）：
$$P_t = \{i \mid \Delta\mu_t[i] < \tau\}$$
$$\Delta\mu_t[i] = \mu_t[i] - \mu_t[i-1]$$

- $\Delta\mu_t[i]$: 相邻行 $\mu$ 的增量
- $\tau$: threshold（0.3），小增量 = attention 不再 advance

Forward horizon（Eq. 8）：
$$N_f = \lfloor \mu_t[\min(R_t \cap P_t)]\rfloor + 1$$

- $R_t \cap P_t$: 同时低 entropy 和 plateau 的行集合
- $\min(\cdot)$: 第一个这样的行
- $\lfloor \cdot \rfloor + 1$: 取整 + 1-indexed

## 实验结果咋样

**LIBERO + $\pi_{0.5}$ (p=50)**:
| Method | LIB-Spatial | LIB-Object | LIB-Goal | LIB-10 |
|---|---|---|---|---|
| 最好固定 $e$（Static Oracle+，调参搜出来的） | 96.4 | 97.6 | 93.9 | 91.9 |
| AutoHorizon | **96.5** | **98.0** | **94.4** | **92.1** |

每个 task 都比精心调参的 baseline 还好一丢丢，但**不用调参**。

**Real-world**（Franka robot，3 个 pick-and-place task）:
| Task | Static Oracle+ | AutoHorizon |
|---|---|---|
| Cucumber Plate | 97.0 | **98.0** |
| Cube Plate | 81.5 | **92.0** |
| Cube Bowl | 97.5 | **99.0** |

Cube Plate 提升了 10 个点。而且 Table 9-10 显示 $q$ 和 $\tau$ 怎么调都稳，对 hyperparameter 不敏感。

## 我觉得最 cool 的几点

1. **几乎零开销**：只在第一个 sampling step 算一次 attention，剩下的 rollout 都用这个 horizon。不需要像 uncertainty estimation 那样 sample 4 次（Table 5 里 Uncertainty Proxy 慢得多）。

2. **Generalize 跨架构**：$\pi_{0.5}$ 和 GR00T N1.5 架构差异很大（GR00T 用 alternating cross-attn 和 self-attn block），但 radial action sink 都出现了。说明这是 flow-based VLA 的 intrinsic property，不是某个架构的 artifact。

3. **Explainability 的实际用途**：attention analysis 通常只用来"解释"模型，这篇直接把 attention pattern 当 control signal 用，从 explainability 到 actionable insight。这思路在 LLM interpretability 里也该多探索。

4. **和 LLM attention sink 的呼应**：StreamingLLM 发现 LLM 里 initial token 是 attention sink，这里发现 VLA 里 initial 和 terminal action token 都是 sink。VLA 的 sink 可能是 inter-chunk continuity 需求驱动的，和 LLM 的 positional/structural 解释不同——但都是"模型把某些 token 当 anchor"的体现。

## 可能的局限和我的疑问

1. **Attention 真的等于 confidence 吗？** attention 高可以表示关注，但关注不等于 prediction 准。有些工作（attention is not explanation）质疑 attention 作为 explanation 的可靠性。这里用作 confidence proxy 是 empirical 有效的，但理论基础还可以更扎实。

2. **$e \log e$ 的 divergence 假设**：Proposition 1 用 $\delta^d_j(e) = ke\log e$，但没给 empirical evidence 证明 divergence 真长这样。换成 $e^2$ 或别的增长形式，结论可能不同。这是 existence proof 的 caveat。

3. **只对 flow-based VLA 测过**：autoregressive VLA（如 OpenVLA, RT-2）每个 action 是一个 token，没有 chunk 的概念，attention pattern 可能完全不同。radial sink 还存不存在？得验证。

4. **为什么 language attention 几乎没用（Table 8）却还存在？** 这个 redundancy 很 intriguing。是训练时 distillation 进 vision 了，还是 VLA 根本没真用 language？如果是后者，VLA 的 language grounding 可能比想象中弱。这和有些工作质疑 VLM 的真正 language understanding 是一个路子。

5. **Attention sink 和 chunk 内 attention invariance 是 cause 还是 symptom？** 作者把它们当 cause 来用（从 attention 读出 horizon），但这些 pattern 可能是 training 的 epiphenomenon——模型恰好学成这样，但最优解未必是这样。如果重新训练一个 explicitly 鼓励后面 action adapt vision 的 model，性能会不会更好？

## 相关工作和延伸

- **MPC 的 control horizon**: 控制论里 control horizon < prediction horizon 是常识，但 MPC 有 dynamics model 可以算最优 horizon。VLA 没有 model，只能从内部 signal 估。
- **StreamingLLM**: attention sink phenomenon in LLM，方法被这篇借鉴到 VLA。
- **BID (Liu et al.)**: 同样 trade off 的理论分析，但用 rejection sampling 解决，计算贵。AutoHorizon 是 attention-based 的 zero-cost 替代。
- **Diffusion Policy**: action chunking 的开山之作，但 execution horizon 还是手调。
- **VAR (Visual Attention Redistribution, Kang et al.)**: 发现 VLM 有 visual attention sink，提出 redistribute。Table 8 用 VAR-L-0.5 提升性能，说明 VLA 也有类似冗余。

## Reference Links

- [π0.5 paper](https://arxiv.org/abs/2504.16054)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [ACT / ALOHA](https://arxiv.org/abs/2304.13705)
- [StreamingLLM](https://arxiv.org/abs/2309.17453)
- [BID (Bidirectional Decoding)](https://arxiv.org/abs/2408.13755)
- [Visual Attention Sink (Kang et al.)](https://arxiv.org/abs/2502.04239)
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [DROID dataset](https://arxiv.org/abs/2403.12945)
- [Project page (video demos)](https://vlaknowsitslimits.github.io/)

---

一句话总结：**模型 attention 里已经藏着"我信自己到第几步"的信号，去读它就行**。这思路又简单又优雅，和 LLM 里 attention sink、early exit、speculative decoding 一类工作在哲学上一脉相承——内部 representation 已经包含了你想要的 control signal，别外求，往内看。

---

# VLA Knows Its Limits — 深度解析

Andrej，这篇 paper 触及了 action chunking policies 中一个被长期忽视但极其关键的问题：**execution horizon 的选择**。我尽量从多个角度深入展开。

## 1. 核心问题与动机

### 1.1 Action Chunking 的基本设置

在 flow-based VLA 模型（如 π0.5, GR00T N1.5）中，policy 不再预测单个 action，而是预测一个包含 p 个连续 action 的 chunk：

$$\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+p-1}]$$

- $\mathbf{A}_t$: time step $t$ 预测的 action chunk
- $\mathbf{a}_{t+i}$: 第 $i$ 个 future action
- $p$: **prediction horizon**（chunk 总长度）

执行时只执行前 $e$ 个 action，然后重新观测、重新规划：
- $e$: **execution horizon**（实际执行的前缀长度）
- 通常 $e < p$，剩余的 action 被丢弃

这种 closed-loop prediction 机制在 reactivity 和 temporal consistency 之间做 trade-off。

### 1.2 问题的发现

作者发现一个 striking 的现象：**固定 $e$ 在整个 rollout 中是 suboptimal 的**。如图 1 所示，在 LIBERO benchmark 上用 π0.5 测试，变化 $e$ 导致 success rate 大幅波动，呈现明显的 "先上升后下降" 的 peaked pattern。

例如 Table 1 中 $p=50$ 的 LIB-Spatial：
- $e=0.2p=10$: 94.9%
- $e=0.4p=20$: 92.4%
- $e=0.6p=30$: 87.1%
- $e=0.8p=40$: 81.2%
- $e=1.0p=50$: 71.5%

这个 drop 非常剧烈。而且不同 task 的最优 $e$ 完全不同（见 Table 6: LIB-Spatial 最优 $e=5$, LIB-Object 最优 $e=13$, LIB-Goal 最优 $e=15$）。

### 1.3 为什么固定 horizon 不行

直觉上，task 的不同阶段需要不同的 reactivity：
- **Reaching toward coffee carafe**: 环境稳定，长 horizon 促进 smooth motion
- **Pouring into cup**: 需要短 horizon 增强 responsiveness

这就是 Figure 2 右侧展示的：robot 在 reaching/moving 时 horizon 变大，在 grasping/placing 时 horizon 变小。

## 2. 理论分析：为什么存在最优 horizon

### 2.1 Proposition 1 的设置

作者把 rollout 的总误差建模为：

$$\mathcal{L}(e) = \sum_{i=0}^{m-1} \delta^c + \sum_{j=0}^{m} \delta_j^d(e)$$

变量解释：
- $e$: execution horizon
- $m = \lceil L/e \rceil$: 总共执行的 chunk 数量，$L$ 是总 action 数
- $\delta^c$: **chunk transition loss** — 每次 chunk 切换时由于重新观测、重新规划造成的 reward loss，假设与 $e$ 独立
- $\delta_j^d(e)$: j-th chunk 与对应 expert trajectory 的 **divergence loss**，随 $e$ 增大而增大

第一项 $\sum \delta^c$ 有 $m-1$ 项（chunk 之间的 transition 数量），随着 $e$ 增大而减小（chunk 数减少）；第二项 $\sum \delta_j^d(e)$ 随着 $e$ 增大而增大（chunk 越长，后面 action 的 divergence 越大）。

### 2.2 关键假设

假设 $\delta_j^d(e) = k \cdot e \log e$，其中 $k > 0$ 是 scaling factor。

这个形式很有意思：
- $e \log e$ 意味着 divergence 随 $e$ 增长比线性还快一点
- 这反映了 compounding error 的特性

### 2.3 求解过程

代入后：
$$\mathcal{L}(e) = \left(\frac{L}{e} - 1\right)\delta^c + Lk \log e$$

对 $e$ 求导：
$$\frac{\partial \mathcal{L}(e)}{\partial e} = -\frac{L\delta^c}{e^2} + \frac{Lk}{e} = \frac{L}{e^2}(ke - \delta^c)$$

令导数为 0，得到唯一驻点：
$$\hat{e} = \frac{\delta^c}{k}$$

二阶导数：
$$\frac{\partial^2 \mathcal{L}(\hat{e})}{\partial e^2} = \frac{L\delta^c}{\hat{e}^3} > 0$$

所以 $\hat{e}$ 是全局最小值点，$\mathcal{L}(e)$ 在 $(0, \hat{e})$ 严格递减，在 $(\hat{e}, \infty)$ 严格递增。由于 $e \in \mathbb{N}$ 且 $1 \leq e \leq p$：

$$e^* = \text{clamp}\left(\lceil \frac{\delta^c}{k} \rceil \text{ or } \lceil \frac{\delta^c}{k} \rceil - 1, 1, p\right)$$

### 2.4 Intuition

这个理论揭示了 trade-off：
- $\delta^c$ 大（chunk transition 代价高）→ $e^*$ 大 → 长期 consistency
- $k$ 大（intra-chunk divergence 增长快）→ $e^*$ 小 → 短期 reactivity

但作者也强调这只是 existence proof，不直接指导 method design，因为 $\delta^c$ 和 $k$ 在实际中难以估计。

### 2.5 小 prediction horizon 的极端情况

Figure 7 显示 $p=10$ 时，最优 $e$ 通常在边界 $e=p$。这是因为 $p$ 小时，policy 训练时见过完整 chunk，能准确预测 short trajectory，使得 $k \to 0$。但 $e > p$ 时性能急剧下降，因为 train-test mismatch。

## 3. Attention 分析：两个关键现象

这是论文最 insightful 的部分。作者通过分析 attention weights 来解释为什么性能呈现 peaked pattern。

### 3.1 现象 ❶: Intra-chunk Actions 对 Vision-Language 的 Invariant Attention

Figure 3 可视化了 π0.5 最后一个 sampling step 的 cross-attention map：
- 前 768 tokens: vision
- 接下来 200 tokens: language  
- 剩余: action

**关键观察**：chunk 内不同位置的 action（rows）对 vision-language tokens（columns 的前 968 个）的 attention pattern 几乎完全相同。

这意味着什么？
- 虽然模型预测了时间上 extended 的 action sequence，但后面的 actions 没有适应性地调整对环境的关注
- 它们反复依赖对早期 action 有用但对后期 action 越来越 outdated 的 static visual-linguistic features
- 执行后期 actions 变得 redundant 甚至有害

**Language attention sink**: 作者还发现第一个 language token 接收异常高的 attention，类似 LLM 中的 attention sink phenomenon。但实验（Table 8）显示完全 mask 掉 language tokens 性能只略微下降，说明大部分 linguistic semantics 已经被 vision tokens 吸收。

### 3.2 现象 ❷: Radial Action Sinks

Figure 4 可视化了 action tokens 之间的 self-attention（不同 $p$ 下）：

**关键观察**：
- 初始 action token 和终止 action token 接收极高的 attention
- 这种高 attention 在短 span 内保持，然后迅速衰减
- 形成一个 low-attention plateau

作者称这两个 boundary tokens 为 **radial action sinks**。

### 3.3 为什么会有 radial action sinks

作者给出两个解释：

**原因 1**: 初始 action 有最低的 cumulative error，作为 stable anchor。后续 actions 跟随这个 anchor 的指引，同时逐渐 attend 到相邻 tokens，产生 smooth trajectories。

**原因 2**: Policy 训练时 expert demonstrations 从随机 timestamp 开始，初始和终止 action 都在保持 inter-chunk continuity 中起作用，反映模型确保 chunk 边界 smooth transition 的 implicit objective。

### 3.4 Action self-attention 作为 predictive limit 的 indicator

结合 ❶ 和 ❷：
- 当 attention 对 radial action sinks 保持高 → 模型 confident，predicted actions 仍 aligned with anchors
- 当 attention 衰减 → 模型转向 self-referential dependence，compounding error 放大

**这就是 "VLA knows its limits" 的核心 insight**：attention weight 的衰减标志着模型 predictive capability 的边界。

### 3.5 在 GR00T N1.5 上的验证

Figure 8 和 10 显示 GR00T N1.5 也有相同 pattern，尽管架构很不同：
- π0.5: 标准 transformer attention
- GR00T N1.5: alternating modality fusion（cross-attention block 后接 self-attention block，重复多次）

这说明 radial action sink 是 flow-based VLA 的 intrinsic property，不依赖于具体架构。

## 4. AutoHorizon 方法

### 4.1 整体思路

用 attention weights 作为 proxy，**per-chunk** 动态估计 execution horizon。核心是找到 attention mass 停止 advancing 并开始 plateau 的 turning point。

### 4.2 算法步骤详解

**Step 1: Attention extraction 和 normalization**

在每个 sampling step $t$：
1. 提取 action self-attention maps
2. 跨所有 transformer blocks 和 attention heads 平均
3. Row-wise normalization 使每行 sum to 1
4. 得到 $\mathbf{S}_t \in \mathbb{R}^{p \times p}$

$\mathbf{S}_t[i,j]$ 表示 i-th query action 对 j-th key action 的 attention weight。

**Step 2: Low-entropy row filtering (Eq. 4, 5)**

$$H_t[i] = -\frac{1}{\log p} \sum_j \mathbf{S}_t[i,j] \log \mathbf{S}_t[i,j]$$

- $H_t[i]$: i-th row 的 normalized entropy，范围 $[0, 1]$
- $p$ 在 log 中作为 normalization
- 低 entropy → attention 集中，信息量大
- 高 entropy → attention 分散，uncertain

$$R_t = \{i \mid H_t[i] \leq Q_q(H_t)\}$$

- $Q_q$: q-quantile 函数，实验中 $q=0.9$
- 只保留 entropy 最低的 10% rows
- 这步过滤掉 attention uniformly diffused 的 actions，保留 sharp、confident patterns

**Step 3: Forward soft-pointer (Eq. 6)**

$$\mu_t[i] = \max\left(\sum_{j=0}^{p-1} j \cdot \mathbf{S}_t[i,j], \max_{k \leq i} \mu_t[k]\right)$$

- $\mu_t[i]$: i-th query 的 expected predictive position
- $j \cdot \mathbf{S}_t[i,j]$: 位置 $j$ 乘以 attention weight，计算期望
- 外层 max：强制 $\mu_t$ 非递减，防止 backward jumps
- 内层 max：cumulative max over previous rows

直觉：如果 row $i$ 主要 attend 到 row $i+5$，那么 $\mu_t[i] \approx i+5$，表示模型 "look ahead" 5 步。

**Step 4: Plateau detection (Eq. 7)**

$$\Delta\mu_t[i] = \mu_t[i] - \mu_t[i-1]$$
$$P_t = \{i \mid \Delta\mu_t[i] < \tau\}$$

- $\Delta\mu_t[i]$: incremental change in expected position
- $\tau$: threshold（实验中 $\tau=0.3$）
- 小的 $\Delta\mu$ 表示 attention mass 停止 advancing → plateau 开始

**Step 5: Forward horizon (Eq. 8)**

$$N_f = \lfloor \mu_t[\min(R_t \cap P_t)]\rfloor + 1$$

- $R_t \cap P_t$: 同时满足 low-entropy 和 plateau 的 rows
- $\min$: 取第一个这样的 row
- $\lfloor \cdot \rfloor + 1$: 转成 1-indexed integer

**Step 6: Backward pointer**

对 reversed attention matrix $\tilde{\mathbf{S}}_t$ 重复 Step 3-5，得到 $N_b$。

**Step 7: Horizon fusion**

```
if N_f + N_b >= p: N = p  # full coverage
else: N = N_f  # forward prefix only
```

经验上，$p$ 小时前者常见，$p$ 大时后者 dominate。

### 4.3 Intuition 上的解释

这个 bidirectional pointer 机制可以这样理解：
- Forward pointer 从 chunk 开头扫描，找 attention 从 initial sink 开始能延伸多远
- Backward pointer 从 chunk 末尾扫描，找 attention 从 terminal sink 开始能延伸多远
- 如果两者覆盖整个 chunk → 模型对整个 chunk confident → 执行全部
- 否则只执行 forward prefix → 保留 reactivity

### 4.4 Hyperparameter 选择

实验中固定使用：
- $q = 0.9$（保留 entropy 最低 10%）
- $\tau = 0.3$（plateau 检测 threshold）
- 在第一个或第三个 sampling step 操作

Table 9 和 10 的 ablation 显示对这些 hyperparameter 不敏感。

## 5. 实验详解

### 5.1 LIBERO Benchmark (Table 1)

**π0.5, p=10**:
- 最优 $e$ 在边界 $e=p$（见 Table 6: LIB-Spatial=10, LIB-Object=8, LIB-Goal=10, LIB-10=10）
- Random baseline 也表现不错（因为模型 overfit 到 short chunks）
- AutoHorizon: 99.1, 99.2, 97.5, 91.6 — 全部最好

**π0.5, p=50**:
- 明显 peaked pattern
- $e=0.2p=10$ 往往最好（Static Oracle）
- AutoHorizon: 96.5, 98.0, 94.4, 92.1 — 比 Static Oracle+ 还好或持平

### 5.2 GR00T N1.5 (Table 2)

- $p=16$，同样 peaked pattern
- AutoHorizon: 96.7, 98.7, 96.0, 92.7 — 全部最好

### 5.3 RoboTwin (Table 3)

Bimanual tasks，更复杂：
- Pick Bottles 是 horizon-sensitive task
- AutoHorizon 在所有 7 个任务上达到 100.0, 68.0, 91.0, 92.0, 85.3, 84.7, 75.0

### 5.4 Real-World (Table 4)

Franka Research 3 robot，DROID setup，p=50:
- Cucumber Plate: AutoHorizon 98.0 vs Static Oracle+ 97.0
- Cube Plate: AutoHorizon 92.0 vs Static Oracle+ 81.5
- Cube Bowl: AutoHorizon 99.0 vs Static Oracle+ 97.5

Real-world 中观察到的行为：
- $e \in [1,5]$: robot hesitates，低 amplitude movements
- $e \in [20,40]$: overreaches, collisions
- $e > 40$: 频繁掉物体

### 5.5 Ablation 精选

**Table 7: vs Nearest Static Oracle**
比较 AutoHorizon 均值 $m$ 附近的固定 horizon：
- LIB-10: AutoHorizon 92.1 vs $\lfloor m \rfloor$ 89.1 vs $\lceil m \rceil$ 91.9
- 均值接近最优 static，但偶尔选大/小 horizon 处理 corner cases

**Table 8: Language tokens 的作用**
- Mask Lang（完全 mask language attention）只略微下降
- VAR-L-0.5（redistribute 50% language attention）有时甚至提升
- 证明 language attention 高度 redundant

**Figure 5: Estimated horizon distribution**
AutoHorizon 产生 wide distribution，适应不同 input conditions。大部分值在 moderately low 范围（favoring reactivity），偶尔大值（facilitating fast completion）。

## 6. 关联与延伸思考

### 6.1 与 MPC 的关系

这个 setup 和 Model Predictive Control (MPC) 几乎是同构的：
- $p$ ↔ prediction horizon
- $e$ ↔ control horizon
- MPC 中 control horizon 通常 < prediction horizon

MPC 理论中也有类似的 trade-off 分析，但 VLA 中由于 policy 是 learned 而非 model-based，trade-off 的来源不同。

### 6.2 与 StreamingLLM 的关联

Radial action sink 直接让人联想到 StreamingLLM (Xiao et al., 2024) 发现的 attention sink phenomenon：
- LLM 中 initial tokens 接收 disproportionate attention
- 保留这些 tokens 的 KV cache 可以做 infinite-length generation

VLA 中的 sink 可能来源不同：
- LLM sink: 可能编码 positional/structural info
- VLA sink: 可能与 expert demo 的 random start timestamp 有关，为了 inter-chunk continuity

### 6.3 与 Diffusion Policy 的关系

Diffusion Policy (Chi et al.) 引入了 action chunking 到 robot learning，使用 weighted averaging 处理 overlapping chunks。但 execution horizon 仍需人工设定。

### 6.4 与 BID 的关系

BID (Liu et al., 2025) 提供了 action chunking 的理论分析：
- 短 horizon → reactivity 但 instability
- 长 horizon → smoothness 但 loss of responsiveness
- 提出 rejection sampling

AutoHorizon 可以看作是 BID 思想的 attention-based 实现，避免了 rejection sampling 的计算开销。

### 6.5 π0.5 和 Flow Matching

π0.5 使用 flow matching 而非 diffusion：
- Flow matching: 学习 vector field 直接 transform simple distribution 到 complex distribution
- 比 diffusion 更 sample-efficient
- Action generation 通过 iterative sampling

AutoHorizon 在 first 或 third sampling step 操作，说明 attention pattern 早期就稳定了。

### 6.6 Compositional Generalization 的 hint

Language attention 的高度 redundancy（Table 8）很值得深思：
- Vision-language pretraining 已经把 linguistic semantics 压入 visual representation
- 推理时 explicit language attention 是 superfluous
- 这暗示 VLA 的 compositional generalization 可能更多地依赖 vision grounding

### 6.7 与 Token-level Uncertainty Estimation

Uncertainty Proxy baseline（Table 5）通过 Monte Carlo variance 估计 uncertainty，但需要 4x 计算。AutoHorizon 用 attention 作为 proxy，几乎零开销。

这让人联想到 LLM 中的 entropy-based early exit 和 speculative decoding — 都是用内部 signal 做 adaptive computation。

### 6.8 与 Chain-of-Thought in VLA 的对比

CoT-VLA (Zhao et al.) 在 action 前生成 reasoning tokens。AutoHorizon 的 attention-based horizon 选择可以看作是 "implicit" version — 不显式生成 reasoning，而是从 attention pattern 中读出 model confidence。

### 6.9 可能的局限

1. **Attention 作为 confidence proxy 的可靠性**：attention 高不一定意味着 prediction 准确，attention 可以是 misleading
2. **Chunk transition loss $\delta^c$ 的假设**：假设与 $e$ 独立可能过强
3. **$e \log e$ 的 divergence model**：是建模选择，empirical 验证不足
4. **只针对 flow-based VLA**：是否能扩展到 autoregressive VLA（如 OpenVLA）？

## 7. Reference Links

- [π0 paper (Physical Intelligence)](https://arxiv.org/abs/2410.24164)
- [π0.5 paper](https://arxiv.org/abs/2504.16054)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [ACT (ALOHA)](https://arxiv.org/abs/2304.13705)
- [StreamingLLM (attention sink)](https://arxiv.org/abs/2309.17453)
- [BID (Bidirectional Decoding)](https://arxiv.org/abs/2408.13755)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [LIBERO benchmark](https://arxiv.org/abs/2306.03310)
- [DROID dataset](https://arxiv.org/abs/2403.12945)
- [Visual Attention Sink (Kang et al.)](https://arxiv.org/abs/2502.04239)
- [GR00T N1](https://arxiv.org/abs/2503.14734)
- [RoboTwin](https://arxiv.org/abs/2504.02098)
- [3D Diffusion Policy](https://arxiv.org/abs/2403.03654)
- [Project page (video demos)](https://vlaknowsitslimits.github.io/)

## 8. 总体评价

这篇 paper 的核心贡献在于：
1. **Empirical discovery**: execution horizon 对性能影响巨大但被忽视
2. **Mechanistic insight**: 通过 attention analysis 揭示了两个关键现象，特别是 radial action sink
3. **Practical method**: AutoHorizon 几乎零开销，generalize across π0.5 和 GR00T N1.5

最 impressive 的地方是 attention analysis 的 depth — 把 attention pattern 和 model 的 predictive limit 直接联系起来，这是一个非常漂亮 的 interpretability 工作。

如果 Karpathy 你在构建 VLA 系统，这个 insight 可能比 method 本身更有价值：**model 内部的 attention 已经告诉你它 confident 到哪里了，你只需要去读它**。这和 LLM 中的很多 interpretability 发现一脉相承，说明 transformer-based models 有共性的 internal confidence signaling 机制。
