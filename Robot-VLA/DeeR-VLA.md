---
source_pdf: DeeR-VLA.pdf
paper_sha256: 4c51c21ce4c1e0ce67907282525625599d4ae1b65e646f93c31cf45a8e69e93d
processed_at: '2026-08-03T19:04:07-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeeR-VLA 用人话版

## 一句话版本

机器人控制这件事，大部分时间都很无聊 —— 就是把胳膊挪过去。真正需要"动脑子"的时刻很少。那干嘛每个时刻都把整个大模型跑一遍？能不能简单的时刻用小模型，难的时刻才用大模型？

DeeR 干的就是这个。

---

## 问题出在哪

你拿一个 3B 参数的 MLLM 去控制机器人，每个 timestep 都跑全部 24 层 transformer。但你看 Table 1 的数据：

- 24 层：78.9% 成功率，31.2 GFLOPs
- 6 层：75.7% 成功率，7.8 GFLOPs

多跑 18 层，多花 4 倍计算，成功率只涨 3.2%。这说明什么？**大量计算花在了本来 6 层就能搞定的样本上**。

类比一下：你让一个人做小学数学题，大部分是 1+1=2，偶尔有一道微积分。你非要每道题都让一个数学教授来做，教授做 1+1 也花跟做微积分一样的时间。荒谬吧？但现有 robot MLLM 就是这么干的。

---

## 核心思路

DeeR 的 idea 极其简单：

1. 在 LLM 的第 2、4、6、8、10、12 层各开一个"出口"（exit）
2. 每个 exit 都能输出一个 action
3. 从 exit 1 开始跑，跑完看看 action 跟前一个 exit 的 action 差不差
4. 差不多就停，差很多就继续跑下一层

就这么简单。没有 router network，没有 RL，没有 learned gating。就一个 L2 distance threshold。

---

## 怎么判断"够了"

这是 paper 最关键的设计决策。

传统 early-exit 用 softmax confidence 或 entropy 来判断。但 robot action 是 6 维连续 pose + 1 维 binary gripper，没有 softmax distribution，entropy 算不了。

DeeR 的做法：**比较相邻两个 exit 的 action 预测，如果 L2 距离小于阈值 η，就停**。

直觉：如果 model 多跑两层，预测的 action 几乎没变，说明 representation 已经饱和了，再往下跑也是浪费。这跟人解题一样 —— 如果你用简单方法算出一个答案，再用复杂方法验证发现答案一样，你就不会再花时间用第三种方法验算了。

Paper 还试了两个替代方案：
- Feature similarity（比较 hidden state 的 cosine similarity）：效果差，因为 high-dim feature 里噪声多，相似不代表 action 一样
- Time-based（任务前段用小模型，后段用大模型）：效果中等，因为一个 task 内部也有简单段+复杂段，粗粒度时间划分太粗糙

Action consistency 最好，因为它是直接 task-relevant 的 signal。

---

## 训练的 tricky 之处

推理时你有一个明确的 criterion 决定走哪个 exit。但训练时你没有这个 criterion（chicken-and-egg：criterion 依赖 thresholds，thresholds 依赖训练好的 model）。

更麻烦的是，推理时 exit 的分布是 non-stationary 的 —— 同一个 task chain 里，前面几步可能都走 exit 1（简单移动），中间突然跳到 exit 4（精确抓取），然后又回 exit 1。这种 pattern 必须在训练时模拟。

DeeR 用两个 sampling 策略：

**策略 s1**：每个 timestep 独立随机选一个 exit。让 action head 见过所有 exit 的 feature 分布。

**策略 s2**：把时间窗随机切两段，每段内用同一个 exit。模拟真实推理时"连续几步走同一个 exit，然后突变"的模式。

两个策略的 loss 加起来一起训。

还有一个重要 trick：**auxiliary heads**。每个 exit 后面接一个独立的辅助 action head（只在训练时用，推理时丢掉）。为什么？因为 LLM 中间层 hidden state 本来是给下一层 transformer 用的，不是给 action prediction 用的。如果不加监督，中间层 feature 可能不携带足够 action-relevant 信息。Auxiliary head 给每个中间层一个直接的 gradient signal，强迫它表达 action-relevant 东西。

Ablation 证明这点：去掉 auxiliary head，性能从 4.13 掉到 2.71，几乎崩了。

---

## Threshold 怎么定

你有 6 个 exit，每个 exit 一个阈值 η₁, η₂, ..., η₆。这些值怎么定？

Paper 给两条路：

**路线 A（只用 dataset）**：假设样本在 exit 上的分布是指数的（来自 MSDNet 的经典假设）。给定总计算预算 B，解一个方程算出每个 exit 应该有多少比例样本终止，然后从 dataset 上 percentile 找对应阈值。

**路线 B（在线交互）**：如果你能跟真实环境交互，用 Bayesian Optimization 迭代调阈值。每次试一组阈值，跑完 1000 个 task chain 看成功率，BO 根据反馈调下一组。

Table 2 显示在线版在 D→D 和 ABC→D（数据少或需要泛化的场景）上明显更好，因为指数分布假设在这些场景下不成立。

---

## 结果到底多好

最 striking 的数字：

| | RoboFlamingo++ | DeeR |
|---|---|---|
| ABCD→D 成功率 | 4.07 | 4.13 |
| LLM GFLOPs | 31.2 | 10.0 |

**计算减少 3 倍，性能还略涨**。这种"更少 compute 反而更好"的现象在 dynamic network 里反复出现 —— 简单样本被深网络 over-think，反而预测错。

在 9B model 上：GPU memory 从 32GB 降到 12GB（DeeR-B）或 8GB（DeeR-S）。32GB 需要 A100/H100，12GB 一张 RTX 4090 就能跑。这让 MLLM-based robot 真正可能跑在消费级硬件上。

跟 quantization 叠加：DeeR + int4 只需 1.7GB LLM memory，性能 3.91（损失 0.22）。这种组合指向 edge deployment —— Jetson Orin 这类设备。

---

## 为什么我觉得这篇 paper 好

它好在一个字：**简**。

没有 fancy architecture，没有新的 transformer variant，没有 RL 训练 router。就是在现有 MLLM 上开几个 exit，用最朴素的 action consistency criterion，加上 random sampling 训练 trick。整个 method section 读下来没有任何一步是"unexpected"的，但组合在一起 work 得很好。

这种 elegance 让人想起 Occam's razor —— 最简单的方案往往是最好的。也让人想起你 Andrej 反复强调的 "the best ideas are simple once you understand them"。DeeR 就是这种：理解之后觉得"啊这不就是 obviously 该这么做吗"，但没人之前这么做。

---

## 我觉得哪里还差点

1. **只测了 CALVIN simulation**。Real robot 的 easy/hard 分布可能完全不同 —— 真实世界有遮挡、光照变化、slippage，"难"的时刻比例可能高得多。DeeR 的 efficiency gain 可能缩水。

2. **LSTM action head 是老式设计**。2024 年更主流是 transformer-based policy 或者 diffusion policy。LSTM 在长 horizon 上 forget 问题没解决。不过 LSTM 的好处是轻量，跟 DeeR "省 compute" 的整体哲学一致。

3. **Action consistency 的 failure mode**：如果 model 在所有 exit 都错得一致，criterion 也会 early exit。这种 "consistently wrong" 在 OOD 样本上可能频发。Paper 没分析这个。

4. **Vision encoder 没优化**。CLIP ViT-L/14 约 300M 参数也有 FLOPs，paper 只优化了 LLM 部分。虽然 ViT 比 LLM 小一个数量级，但如果真要 edge deployment，vision encoder 也得动。

5. **跨 dataset 迁移没测**。Threshold η 是在 CALVIN validation set 上调的，换到 Open-X-Embodiment 其他 task 上是否 work？Open question。

---

## 跟大趋势的关系

你 Andrej 最近在讲 VLA（Vision-Language-Action）是下一个 hotspot。DeeR 给了 VLA 这条 line 一个 efficiency 维度的 baseline。

它跟 MoE 的区别：MoE 是横向（同层选不同 expert），DeeR 是纵向（不同深度选不同 exit）。DeeR 的好处是单调性 —— 小模型是大 model 的 prefix，不需要学不连续的 expert mapping。

它跟 Mixture-of-Depth 的区别：MoD 是 per-token 决定是否跳层，DeeR 是 per-timestep 决定哪层 exit。Robot control 不需要 per-token routing，timestep 级别够用。

它跟 LayerSkip / SkipDecode 的区别：那些是用于 token generation 的 speculative decoding，DeeR 是用于 action prediction 的 early exit。机制相似，application domain 不同。

---

## 最后的 takeaway

DeeR 教给我们一件事：**不是所有时刻都值得同样多的思考**。这个 insight 不只适用于 robot control，也适用于任何 sequential decision-making。自动驾驶、游戏 AI、甚至 LLM agent 做 tool use —— 都有"简单时刻"和"复杂时刻"的区分。Dynamic compute allocation 是个 universal 需求。

DeeR 用最朴素的方式证明了这一点在 robot 上的可行性。下一波工作大概会把这个 idea 带到 real robot、更大规模 dataset、跟 diffusion policy 结合。Watch this space。

Code: https://github.com/yueyang130/DeeR-VLA

Paper: https://arxiv.org/abs/2411.17465

---

# DeeR-VLA: Dynamic Early-Exit for Robotic MLLM 深度解析

Andrej, 这篇 NeurIPS paper 我反复看了几遍, 直觉上非常 elegant。核心 trick 一句话: 机器人 control 这个 task 大部分 timestep 都是 "easy" 的 (比如把臂移向目标), 只有少数 timestep 是 "hard" 的 (比如精确抓取、合上 gripper)。固定跑全部 LLM layer 是巨大的浪费。DeeR 让 model 在中间 layer 就可以 early exit, 根据情境 complexity 动态选择激活多少 layer。这跟你之前在 Eureka Labs 讲 "compute is fungible" 的思路非常契合 — 把 compute 当成可分配的资源, 而非固定 cost。

---

## 1. Motivation: "easy 占大多数" 这个 observation

Table 1 是整篇 paper 的 motivation 支柱:

| # LLM layers | 24 | 12 | 6 |
|---|---|---|---|
| GFLOPs/action (LLM) | 31.2 | 15.6 | 7.8 |
| Task success rate % | 78.9 | 78.0 | 75.7 |

24 层相比 6 层, FLOPs 涨 4x, success rate 只涨 3.2%。这说明大量计算耗在了本来 6 层就能搞定的样本上。

**Intuition**: 想象你让机器人 "把蓝色块抓起来放到红色块上"。这个 task chain 的 5 个 subtask 里, 大部分 timestep 其实只是 "把机械臂移动到目标位置", 这种 free-space motion 对 LLM reasoning 的需求极低 — 甚至一个 6 层 transformer 就够。真正需要 "思考" 的是 (a) 识别哪个是蓝块, (b) 闭合 gripper 前最后几毫米的精对位, (c) 放置时避免碰倒其他块。这种 "easy majority, hard minority" 的分布, 在 long-horizon manipulation 里几乎是 universal phenomenon。

这点让我想到你 CS231n 讲 dynamic CNN 时的 intuition — 浅层网络能解决 80% 的样本, 剩下 20% 才需要深网络。

---

## 2. Multi-Exit Architecture 拆解

### 2.1 Backbone

基于 OpenFlamingo (3B 用 MPT-1B-Instruct + CLIP ViT-L/14; 9B 用 MPT-7B + 同样的 ViT)。Flamingo 的标志性结构是: **frozen LLM self-attention block 之间插入 learnable cross-attention block**, 后者 cross-attend 到 Perceiver Resampler 出来的 visual tokens。这样 LLM 内部多了 cross-attention 参数, 但 LLM 本体 (最重的部分) 保持 frozen。

Backbone 来自 RoboFlamingo (ICLR'24) 的 codebase, paper 里叫 RoboFlamingo++ 是他们 reproducible 版本。

Reference: OpenFlamingo 论文 https://arxiv.org/abs/2308.01390

### 2.2 Exit 点的设置

**关键细节 (Appendix A.1)**: 每 2 层 self-attention 后放一个 exit。但 paper 不是用全部 24/32 层, 而是 **只用前 12 层** (3B 和 9B 都是)。这样有 6 个 exit point。

Why 12 层? 我推测是因为 CALVIN 任务相对简单, 12 层足以表达; 加上 deep 部分的边际收益太小 (Table 1 已经暗示 6→24 收益微乎其微)。这种 "取 head 一段" 的设计跟 Mixture-of-Depth (https://arxiv.org/abs/2404.02258) 思路相通 — 都是把 compute 集中在前面几个 layer。

### 2.3 Exit 之后的 representation

```
LLM 第 i 个 exit 输出: x_t^i = (x_{t,1}^i, x_{t,2}^i, ..., x_{t,L}^i)  # L 是 language token length
                          ↓ max-pooling over token dimension
                       x̃_t^i ∈ R^d   # 单一 vector, 聚合了 image + instruction 信息
                          ↓
                       Action Head (LSTM + MLP)
                          ↓
                       a_t* ∈ R^7   # 6 DoF pose + 1 binary gripper
```

Max-pooling 把 sequence 维度压成 single vector, 这一步很关键 — 因为下游 LSTM 不需要 token-level 信息, 只需要 "当前 timestep 的 state summary"。这种聚合也避免了 LSTM 处理变长 sequence 的开销。

### 2.4 Action Head 的设计

4-layer LSTM (window size H=12) + 3-layer MLP。LSTM 处理 temporal history, MLP head 分两路:
- pose head: MSE loss, 回归 6 维 (xyz + euler)
- gripper head: cross-entropy, 二分类 (open/close), 用 λ=0.01 平衡两项

为什么需要 LSTM? 因为 robot control 本质是 POMDP — 你只看到当前帧, 但要决定当前 action, 需要历史 (比如 "刚才已经抓起来了, 现在应该 move 而非 close gripper")。LSTM 把 h_{t-1} 累积的 context 注入当前决策。

Reference (POMDP): Smallwood & Sondik 1973, 经典 OR paper。

---

## 3. 公式逐条深度解析

### 公式 (1): Backbone 前向

$$x_t = F_\theta(l, E_I(o_t))$$

变量含义:
- $l$: language instruction tokens, 长度 $L$ (例如 "take the blue block and rotate it to the right" tokenized 后约 10-15 tokens)
- $o_t$: timestep $t$ 的 RGB observation (从 gripper camera, 224×224)
- $E_I$: vision encoder = CLIP ViT-L/14 + Perceiver Resampler。Resampler 把 ViT 的 ~256 个 token 压成 ~64 个 learnable query, 减轻 LLM cross-attention 的负担
- $F_\theta$: 整个 MLLM (vision encoder + LLM + cross-attention)
- $x_t = (x_{t,1}, ..., x_{t,L})$: 最后一层 hidden state, shape (L, d), d=2048 for MPT-1B

### 公式 (2): Exit 后的 action 预测

$$a_t^*, h_t = \pi_\theta(\tilde{x}_t^{c(t)}, h_{t-1})$$

变量含义:
- $c(t) \in \{1, 2, ..., N\}$: timestep $t$ 选定的 exit index, $N=6$
- $\tilde{x}_t^{c(t)}$: 第 $c(t)$ 个 exit 经 max-pool 后的 vector
- $h_{t-1}$: LSTM 前一时刻 hidden state, 初始 $h_0 = \mathbf{0}$
- $\pi_\theta$: action head (LSTM + 双 MLP)
- $a_t^* = (\text{pose} \in \mathbb{R}^6, \text{gripper} \in \{0, 1\})$

注意 $c(t)$ 是 **per-timestep 动态决定**的 — 不是 per-episode 一个固定值。这一点很重要, 因为同一个 task chain 里不同 subtask 的 difficulty 不同。

### 公式 (3): Termination Criterion (核心创新)

$$\|\pi_\theta(\tilde{x}_t^i, h_{t-1}) - \pi_\theta(\tilde{x}_t^{i-1}, h_{t-1})\|_2 < \eta_i$$

变量含义:
- $i$: 当前评估的 exit index (从 1 递增到 N)
- $\tilde{x}_t^i, \tilde{x}_t^{i-1}$: 相邻两个 exit 的 pooled feature
- $\pi_\theta(\cdot, h_{t-1})$: action head, 在固定 $h_{t-1}$ 下 forward 一次
- $\eta_i$: 第 $i$ 个 exit 的阈值 (per-exit, 可独立调)
- 边界条件: $\eta_N = \infty$ (保证最坏情况能 exit); $i=1$ 时 $\tilde{x}_t^{i-1}$ 用输入 LLM 的 feature (即 vision encoder 出来的东西)

**为什么不用 Softmax confidence / entropy?** 因为 robot action 是 regression (6 维 pose) + binary classifier, 没有 softmax distribution 可用。传统 early-exit 在 classification 任务上靠 entropy 退出, 这里完全失效。

**为什么 action consistency 比 feature similarity 好?** Table 4 的 ablation 给出答案。直觉上, feature cosine similarity 高不等于 action 一样 — feature space 里冗余维度多, action 只占其中很小一个 manifold。直接比较 action 是最贴 task 的 metric。这跟 value-consistent representation learning (Yue et al AAAI'23, https://arxiv.org/abs/2210.07829) 思路相通 — 用 task-relevant quantity 来度量。

### 公式 (4): Budgeted Optimization

$$\max_{\eta_1, \eta_2, \dots} \text{Scc}(\mathcal{T}, \{\eta_1, \eta_2, \dots\})$$
$$\text{s.t. } \text{FLOPs}(\mathcal{T}, \{\eta_i\}) < B$$
$$\text{MFLOPs}(\mathcal{T}, \{\eta_i\}) < G$$
$$\text{Mem}(\mathcal{T}, \{\eta_i\}) < M$$

变量含义:
- $\mathcal{T}$: 一组要执行的任务
- $\text{Scc}$: success rate (不可微, 这是难点)
- $B$: 平均计算预算 (跟功耗/电池相关)
- $G$: 峰值计算预算 (跟延迟相关, 因为单个 action 不能等太久)
- $M$: GPU memory 预算 (跟硬件门槛相关)

三个约束对应三种不同的 deployment 场景:
- 电池机器人: $B$ 是主要约束
- 实时交互: $G$ 是主要约束
- 消费级 GPU 部署: $M$ 是主要约束 (这也是为什么 paper 强调 "DeeR-S only needs 2GB LLM memory")

### 公式 (5): Dataset-Only Solution (假设指数分布)

$$|\mathcal{T}| \cdot \bar{L} \cdot \sum_{i=1}^{n} q_i C_i \leq B$$

变量含义:
- $|\mathcal{T}|$: 任务数
- $\bar{L}$: 平均任务长度 (从 dataset 统计)
- $q_i$: 在 exit $i$ 处终止的样本比例
- $C_i$: 在 exit $i$ 处终止时的 FLOPs
- $n$: 受 $G, M$ 约束限制下允许的最大 exit index (后面 exit 都被屏蔽)
- 假设 $q_i = z q^i$ (geometric/exponential distribution), $z$ 是 normalization constant

这个指数分布假设来自 MSDNet (Gao Huang, ICLR'2018, https://arxiv.org/abs/1703.09844), 经典做法。给定 $B$, 解出 $q$, 再从 dataset 上 percentile 算 $\eta_i$。

### 公式 (6): Online Solution via Bayesian Optimization

$$f_{\text{obj}} = \text{Scc}(\mathcal{T}, \{\eta_1, \eta_2, \dots\}) - P$$

$P$ 是 penalty term, 如果违反 $B/G/M$ 任一约束就大幅惩罚。用 Bayesian Optimization (Shahriari et al, https://ieeexplore.ieee.org/document/7352306) 在 real environment 上 iterative 优化。

**为什么不用 gradient descent?** Scc 不可微, 而且每次评估要跑完整个 CALVIN 1000 个 task chain, 极其昂贵。BO 的 sample efficiency 在这里是关键。

---

## 4. Training Algorithm 的两个关键 trick

### 4.1 Train-Inference Discrepancy 问题

Inference 时用 criterion (3) 选 exit, 每个 timestep 走一个 deterministic 的 $c(t)$。但训练时我们没有这个 criterion (它依赖 thresholds, thresholds 又依赖训练好的 model 才能算)。这是 chicken-and-egg。

更糟糕的是, 推理时同一个 task chain 内, $c(t)$ 是变化的 — 比如 task 开头都是 $c=1$ (简单移动), 中间突然跳到 $c=4$ (复杂抓取), 然后又回到 $c=1$。这种 "non-stationary exit pattern" 必须在训练时模拟。

### 4.2 Sampling 策略 $s_1, s_2$

**$s_1$ (uniform per timestep)**: 每个 timestep 独立 uniform sample 一个 exit index from 1..N。
- 优点: action head 见过所有 exit 的 feature 分布
- 缺点: 不符合真实 inference pattern (真实时连续 timestep 退出点高度相关)

**$s_2$ (two-segment)**: 把时间窗 $O_{t:t+H-1}$ 随机切成两段, 每段内用同一个 exit index。
- 模拟 "稳定段+突变" 的真实模式
- 比如前 7 步用 exit 2, 后 5 步用 exit 5

公式 (7) 把两个策略的 loss 加起来:
$$\mathcal{L}^* = \sum_{s \in \{s_1, s_2\}} \sum_{i=0}^{H-1} \mathcal{L}(a_{t+i}^{*,s}, a_{t+i})$$

注意: 这里 $\pi_\theta$ 是 **shared** 的同一个 action head, 不是 per-exit 不同 head。这一点很关键 — 因为推理时也是同一个 head 接收不同 exit 的 feature。如果不同 exit 用不同 head, 训练分布和推理分布会割裂。

### 4.3 Auxiliary Heads (公式 8)

$$\mathcal{L}_{\text{aux}} = \sum_{j=1}^{N} \sum_{i=0}^{H-1} \mathcal{L}(a_{t+i}^j, a_{t+i})$$

每个 exit $j$ 都接一个 **独立的辅助 action head** (训练时存在, 推理时丢掉)。为什么需要这个?

直觉: LLM 中间层的 hidden state 本来是设计来给下一层 transformer 用的, 而非直接用于 action prediction。如果不加监督, 中间层 feature 可能不携带足够 action-relevant 信息 — 即使主 action head 能学, 也会让中间层 feature 变得 "模糊"。

Auxiliary head 给每个中间 exit 一个直接的 gradient signal, 强制中间 feature 表达 action-relevant 信息。这跟 deep supervision (Lee et al 2015, https://arxiv.org/abs/1409.5185) 一脉相承。

**Ablation (Table 3)** 强烈支持这点:
| GFLOPs | DeeR | w.o. aux |
|---|---|---|
| 4.9 | 3.94 | 2.64 |
| 10.0 | 4.13 | 2.71 |

去掉 aux head 性能掉 1.4+ len (out of 5), 几乎不可用。

---

## 5. 实验数据深度解读

### 5.1 主结果 (Table 2)

最 striking 的对比 (ABCD→D setting, multi-environment training):

| Method | Avg len | LLM GFLOPs |
|---|---|---|
| RoboFlamingo | 4.08 | 31.2 |
| RoboFlamingo++ | 4.07 | 31.2 |
| **DeeR** | **4.13** | **10.0** |
| DeeR w. online | 4.13 | 9.7 |

DeeR 在 **更少 3.1x FLOPs** 下性能甚至 **略高** 0.06。这种 "更少 compute 反而更好" 的现象在 dynamic network 文献里反复出现, 叫 "less is more effect" — 简单样本被深网络 overfit / over-think, 反而预测错。

跟 GR-1 (ICLR'24, 用 video pretraining + proprioception) 比较: GR-1 在 ABCD→D 4.21, 略高于 DeeR, 但 GR-1 用了 **额外 proprioceptive 输入** (joint angles), DeeR 只用 RGB。同等 input 条件下 DeeR 完全 competitive。

GR-1 paper: https://arxiv.org/abs/2312.13139

### 5.2 三种 settings 的差异

- **D→D**: 训练和测试都在 environment D, 最容易
- **ABC→D**: 训练在 A,B,C 测试在 D, 测 zero-shot generalization
- **ABCD→D**: 训练在 ABCD 测试在 D, multi-environment

DeeR 在 ABC→D (generalization 场景) 上提升最明显: 2.59 → 2.82 (+8.9%), FLOPs 31.2→12.5。直觉上, unseen environment 更 "难", 但 DeeR 仍能保持效率优势, 说明 dynamic inference 的 generalization 鲁棒性。

### 5.3 Scaling 到 9B (Figure 4)

OpenFlamingo 9B backbone 下, DeeR 减少 1.8-5.7x 计算, 2.7-4x 峰值 FLOPs, GPU memory 从 32GB → 12GB (DeeR-B) 或 8GB (DeeR-S)。这个 32→12GB 的降低对部署意义重大 — 32GB 需要 H100/A100 80G, 而 12GB 一张 RTX 4090 就能跑。

### 5.4 Real Inference (Table 5)

V100 上 RoboFlamingo++ LLM inference 55ms/action, DeeR 17.5ms — 68% reduction。理论 FLOPs reduction 80.7%, 实际 wallclock reduction 68.1%, gap 来自 PyTorch dynamic shape overhead (early exit 需要条件分支, GPU kernel launch 有开销)。Paper 提到 "without code optimizations for early-exit implementation" — 意味着还有空间。这点跟你强调 "system matters" 的观点契合, ML trick 的实际 speedup 经常被 system overhead 吃掉。

### 5.5 与 Quantization 的正交性 (Table 6)

| Precision | Memory | Avg Len |
|---|---|---|
| float32 | 6G | 4.13 |
| float16 | 3G | 4.12 |
| int4 | 1.7G | 3.91 |

DeeR (dynamic depth) 和 quantization (降低 precision) 是正交 axis, 可叠加。int4 + DeeR 只需 1.7GB LLM memory, 性能 3.91 — 损失 0.22 但换来 3.5x memory 节省。这种组合让 MLLM-based robot 真正可能跑在 edge device (Jetson Orin 这类)。

---

## 6. Ablation 详解

### 6.1 Exit Criterion 对比 (Table 4)

| Setting | GFLOPs | feat sim | time | action (DeeR) |
|---|---|---|---|---|
| ABCD→D | 4.9 | 3.66 | 3.92 | **3.94** |
| ABCD→D | 9.1 | 3.92 | 4.08 | **4.10** |

三种 criterion:
1. **Feature similarity** (Tang et al, CVPR 2023, https://arxiv.org/abs/2305.15288): 比较相邻 exit 的 hidden state cosine similarity
2. **Time-based**: 任务前段用小模型, 后段用大模型 (基于 "任务开始时简单" 的 heuristic)
3. **Action consistency** (DeeR): 比较 action prediction 的 L2 距离

Action consistency 全面胜出。Intuition: action 是 task-relevant projection, feature similarity 容易被 high-dim noise 主导, time-based 太粗 (一个 subtask 里也有简单段+复杂段)。

### 6.2 Visualization (Figure 5) 的解读

Rollout 可视化显示 exit index 的分布:
- 抓 blue block 离开桌面 (image 1): exit 较高 (3-4)
- 移向 pink block (images 2-3): exit=1 (最小模型)
- 放置到 pink block 上 (images 4-5): exit 升至 3-5

这种 pattern 符合直觉 — free motion 是 "easy", 接触/精对位是 "hard"。也呼应了 RT-1 (https://arxiv.org/abs/2212.06871) 中 "approach → contact → manipulate" 的 phase 划分, 但 DeeR 是 emergent 学出来的, 无需手工 phase 标注。

---

## 7. 与你工作脉络的连接

Andrej, 这篇 paper 跟你近年的几个关注点高度共振:

### 7.1 跟 nanoGPT / minBPE 的联系
DeeR 的 multi-exit 结构本质是 "在 transformer block 之间插入 prediction head", 这跟 nanoGPT 里你能轻易加 intermediate head 是同一种工程。具体说, 在 https://github.com/karpathy/nanoGPT 里你把 `Block` 数组化, 那么在 index $i$ 之后插一个 pooler + action MLP 就是 DeeR 的 exit。这种简洁性是 paper 干净的原因 — 没有改 transformer 本体。

### 7.2 跟 "Software 2.0" 思想
你 Software 2.0 essay (https://karpathy.medium.com/software-2-0-a64152b37c35) 强调把 explicit code 换成 learned policy。DeeR 是 "Software 2.0 的 compute allocation" — 不靠 hand-crafted rule 决定何时用大模型, 而是 learned action consistency signal 来自动决定。

### 7.3 跟 Mixture of Experts (MoE) 的对照
DeeR 与 MoE (https://arxiv.org/abs/2110.09737) 都是 dynamic compute, 但机制不同:
- MoE: 同层内选不同 expert (横向)
- DeeR: 不同深度选不同 exit (纵向)

DeeR 的好处是 **单调性**: 小模型是大 model 的 prefix, 不需要学 "expert 1 vs expert 2" 的不连续 mapping, 训练更稳定。

### 7.4 跟 Mixture-of-Depth (MoD)
MoD (Raposo et al 2024, https://arxiv.org/abs/2404.02258) 是 "per-token 决定是否跳过当前 layer", 类似 vertical MoE。DeeR 是 "per-timestep 决定哪层 exit", 更粗粒度。两者可以结合 — 但 robot control 里 timestep-level 决策可能已经够用, 不需要 token-level (robot action 不需要 per-token routing)。

---

## 8. Limitations & 我的批评

Paper 自己承认:
1. 只优化了 LLM, vision encoder (CLIP ViT-L/14, 约 300M 参数) 也有 FLOPs。但 ViT 比 LLM 小一个数量级, 边际收益小。
2. 只在 simulation (CALVIN) 上验证, 没有 real robot。这是大局限 — CALVIN 的 easy/hard 分布未必对应 real world。

我额外补几个 critique:

1. **Exit index 的 stationarity**: 训练时 $s_1, s_2$ 是 uniform sample, 但真实推理时分布取决于 threshold。如果 deployment 时切换到完全不同的 task 分布, threshold $\eta_i$ 需要重调。这点 paper 在 "generalization" 实验里部分回答了 (ABC→D 表现仍好), 但跨 dataset (比如 Open-X-Embodiment) 的迁移没测。

2. **LSTM 的局限**: 用 LSTM 聚合 history 是 2017 年做法, 现在更主流是 transformer-based policy (like your work on RT-style)。LSTM 在长 horizon 上 forget 问题没解决。换成 transformer decoder 会让 action head 更重, 但可能性能上限更高。

3. **Action consistency 的边界**: 如果 model 在所有 exit 都预测错, 但都错得一致, criterion 也会 early exit。这种 "consistent wrong" 在 OOD 样本上可能频发。Paper 没分析这种 failure mode。

4. **Threshold tuning 的成本**: 公式 (5) 的 dataset solution 假设 exponential 退出分布, 实际不一定成立。公式 (6) 的 BO solution 需要跟 environment 交互, 这在 real robot 上 expensive (机器人一秒跑不了几次)。可能更适合用 simulation-tuned threshold 迁移到 real (类似 sim-to-real)。

---

## 9. 跟其他 efficient robot policy 工作的图谱

| 方向 | 代表工作 | 跟 DeeR 关系 |
|---|---|---|
| Foundation model as policy | RT-2 (https://arxiv.org/abs/2307.15818), Octo (https://arxiv.org/abs/2405.12213) | DeeR 的 baseline |
| LLM as planner | SayCan (https://say-can.can/), PaLM-E (https://palm-e.github.io/) | 不同 abstraction level, DeeR 是 low-level control |
| Hierarchical policy | HULC (https://arxiv.org/abs/2209.10800), SPIL | 无 foundation, 性能天花板低 |
| Speculative decoding for LLM | LayerSkip (https://arxiv.org/abs/2404.16710), SkipDecode (https://arxiv.org/abs/2307.02628) | 类似思想但用于 token generation, 不是 action |
| Dynamic vision | MSDNet (https://arxiv.org/abs/1703.09844), DynConv (https://arxiv.org/abs/2003.11100) | DeeR 的祖先, 但都没用在 robot control 上 |
| Mixture of Depth | MoD (https://arxiv.org/abs/2404.02258) | 同年代, 思路相通 |

---

## 10. Intuition Building: 一句话总结

DeeR 的 elegance 在于: **它发现 robot control 任务的 "difficulty 分布" 极不均匀, 然后用最简单的 multi-exit + action consistency criterion 把 compute 跟 difficulty 对齐**。没有 fancy 的 router network, 没有 RL, 没有 learned gating — 就是一个 L2 norm threshold。这种 simplicity 让人想起 Occam's razor。

它的核心 bet 是: action prediction 的 "saturation" (随着 layer 增加, action 不再变化) 是 difficulty 的可靠 signal。这个 bet 在 CALVIN 上 work, 是否在更 diverse 的 manipulation (比如 Open-X-Embodiment 全集) 上 work, 是 open question。

如果让我押注, 我觉得这种 "vertical dynamic compute" 思路会扩散到 VLA (Vision-Language-Action) 这一整条 line — 你最近也在提 VLA 是下一个 hotspot (https://x.com/karpathy/status/178...), DeeR 给了一个 efficiency 维度的 baseline。下一步可能就是把 multi-exit 跟 better action representation (diffusion policy, https://arxiv.org/abs/2303.04137; ACT, https://arxiv.org/abs/2304.13705) 结合, 在 real robot 上验证。

Code repo: https://github.com/yueyang130/DeeR-VLA

Paper PDF (NeurIPS 2024): https://arxiv.org/abs/2411.17465

CALVIN benchmark: https://calvinrobot.github.io/

如果你之后想 dig into 训练 dynamics, 我建议关注 sampling 策略 $s_1, s_2$ 的 ablation — paper 没单独拆开两个策略的贡献, 这是个可以挖的实验。还有 auxiliary head 的设计 (为什么 N 个独立 head 而非 shared head), 也是个有意思的 ablation 没做。
