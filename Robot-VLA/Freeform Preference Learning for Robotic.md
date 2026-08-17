---
source_pdf: Freeform Preference Learning for Robotic.pdf
paper_sha256: 7f358874b5f269d3faad538af1781da42fa81b9401cd2e4498edf593618c789c
processed_at: '2026-08-04T10:28:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FPL 人话版

Karpathy, 我用大白话再讲一遍, 核心就一个 idea, 我保证你听完会拍大腿。

---

## 一句话说清楚

**教 robot, 别问人 "哪个更好", 要问 "在哪个方面更好"。**

就这么简单。剩下的全是 engineering 细节。

---

## 问题出在哪

想象你训练 robot 摆餐桌。两条 trajectory:

- **A**: 快得飞起, 3秒摆完, 但是盘子 "哐" 一声砸上去, 刀叉对着人
- **B**: 慢吞吞, 15秒, 但是每个东西轻拿轻放, 摆得整整齐齐

你拿这两条问 annotator: "哪个更好?"

Annotator: "......?" 

这问题根本没法答。**快和慢不可比, 粗糙和细致不可比**。Binary preference 就是逼人在 apples 和 oranges 之间选一个, 选出来的 label 充满噪声。

而且 long-horizon 任务更惨。摆餐桌要放盘子、放杯子、放刀叉, 4个 subtask。Binary reward 只在 trajectory 结尾给一个 "整体好/坏", reward model 根本不知道**哪段时间对应哪个 subtask**, 最后学出来的 reward 在 episode 末尾蹦一个大 spike, 中间全是平的。这就叫 credit assignment 崩了。

---

## FPL 的 insight

与其逼 annotator collapse 成 1 bit, **不如让 annotator 自己说 "我想比较哪个维度"**。

同样两条 trajectory, annotator 可以说:

- "比 speed: A 更快"
- "比 carefulness: B 更 careful"  
- "比 formality: B 更 formal"
- "比 cup placement: B 摆得更好"

一条 video pair, 出 4 个 label, 每个 label 对应一个明确的 axis。**信息量直接 ×4, 而且每个 label 都 unambiguous**。

这就叫 **freeform preference** —— axis 不预先固定, annotator 用 natural language 自己定义。

---

## 怎么用这些 label

三个步骤, 逻辑很清楚:

### Step 1: 训一个能听懂语言的 reward model

这个 reward model 吃两个 input:
- 一条 trajectory (一堆图像)
- 一句话描述 axis ("speed", "quality of 2nd fold", "formality of setup")

输出一个 scalar score。

关键设计: **所有 axis 共用一个 network**, 通过 language condition 切换。这样 "speed" / "fast" / "efficient" 这种同义词自然 share representation, 不用训练 3 个独立 model。

数学上就是把 Bradley-Terry loss 加个 language condition:

$$\mathcal{L} = -\mathbb{E}\left[\sum_k \log \sigma\big((2y_k - 1)(r_\phi(\tau_i | l_k) - r_\phi(\tau_j | l_k))\big)\right]$$

翻译: 对每个 axis $l_k$, 如果 annotator 说 prefer $\tau_i$, 就让 $r_\phi(\tau_i | l_k) > r_\phi(\tau_j | l_k)$。就这么回事。

还有个细节: trajectory reward 不是给整条一个 score, 而是 **每个 timestep 算一个 score 再加起来** (Eq 4)。这保证 reward 有 temporal structure, 不是黑盒吐一个数。

### Step 2: 用 reward model 给整库 trajectory 打分

拿训好的 reward model, 对 dataset $\mathcal{D}$ 里每条 trajectory, 在每个 axis 上算一个 score。然后 standardize 一下 (减均值除标准差), 让数值落在 reasonable range。

### Step 3: 训一个 reward-conditioned policy

这里 FPL 偷了个懒, 不用 PPO (real-world PPO 样本效率太差), 而是 **直接 supervised learning**。

把 reward score 塞进 VLA 的 text prompt:

```
fold the shorts, 
Quality of 1st fold: 2.2, 
Wrinkle of 1st fold: 2.1, 
Quality of 2nd fold: 0.7, 
Fast: 1.2, 
Smooth: 1.2
```

Policy 就是在这个 conditioning 下模仿 trajectory 的 action。本质是 **Decision Transformer 的 multi-axis 版本**。

Reference: Decision Transformer, https://arxiv.org/abs/2106.01345

---

## 为什么这招 work 得这么好

三个 emergent property, 一个比一个 cool:

### 1. Reward 自动 dense 了, 不用人工切 subtask

Binary reward 在 episode 末尾一个 spike, 中间平的。

FPL 的 axis reward 自动在对应 subtask 时刻出现 peak。比如 "placement of big plate" 这个 axis 的 reward, 会在 robot 放大盘子那个 timestep 蹦起来, 其他时候平的。

为什么? 因为 annotator 比较 "big plate placement" 时, 只关注放盘子那段。Reward model 学到的就是 "这个 axis 的 reward 应该 localize 到放盘子的时刻"。**Language 充当了 attention prior**。

Figure 6 里这个对比特别 striking, 你一眼就能看出 binary reward 的 credit assignment 是错的, FPL 是对的。

### 2. Policy 能 compose 训练数据里没有的行为

Simulation 里有个实验特别巧妙。数据集:
- Left peg: 有快 demo, 有慢 demo
- Right peg: **只有慢 demo**

结果 FPL 在 right peg 上跑出了 **比训练数据里任何 right peg demo 都快** 的速度。

怎么做到的? FPL 把 reward 拆成 "fast" 和 "right peg" 两个独立 axis。Policy 学到的是 "给定 fast reward 高 + right peg reward 高, 怎么 act"。即使训练数据里没同时出现 "快 + 右 peg" 的 trajectory, **这个 conditioning 组合 policy 见过**, 所以能 generalize 过去。

这就是 **disentangled representation 的威力**。Binary reward 把所有 axis 揉成一团, policy 没法拆开重组。Multi-axis reward 天然 disentangled, 支持组合。

### 3. Test-time 能 steer, 不用重训

Place cube in bowl 任务: 3个碗 (blue/orange/yellow), 训练时各种 trajectory 都有。

FPL 训完后, test time 直接改 prompt:

```
Blue bowl: -0.5, Orange bowl: 2.0, Yellow bowl: -0.3
```

Policy 就把 cube 放 orange 碗。改成 `Yellow bowl: 2.0` 就放 yellow 碗。**同一个 policy, 不重训, 改 prompt 就 steer**。

为什么能这样? 因为 FPL 训练时 **不过滤 trajectory**, good 和 bad 的都喂给 policy。所以 policy 见过整个 reward spectrum 的 conditioning, test-time 改 conditioning 不会 out-of-distribution。

Binary preference 不行, 因为它 implicit 把 reward 压成 scalar, 训练分布没覆盖 "不同 target + 高 reward" 的联合分布。

---

## 数字说话

Real-world 4个任务:

| 方法 | 平均 success rate |
|---|---|
| BC (imitation learning) | 31% |
| Filtered BC (加成功 rollout) | 37% |
| Single Preference (传统 RLHF) | 33-34% |
| **FPL** | **75%** |

**比第二名高 38 percentage points**。在 Plate Toast 任务上, single preference 直接 0% success —— binary signal 太 noisy, reward model 学崩了。

而且 annotation cost: FPL 每个 label 比 single preference **快 1.85 倍**。因为看一对 video 的 overhead 被 K 个 axis label 摊薄了。

**Supervision 更 dense, cost 更低, performance 更高**。Pareto improvement 三杀。

---

## 为什么以前没人做

其实 multi-objective RL 早就有人做 (Moffaert 2013, Roijers 2013), 但都假设 **predefined axes + predefined reward functions**。你要先 design "speed reward = xxx, safety reward = yyy", 这本身又回到 reward engineering 的坑。

FPL 的真正创新是: **axes 也从 human preference 里学出来**。Annotator 自己说 "我想比 speed", "我想比 wrinkle level", 不用你预先 design。这是 freeform 的核心 —— axis 本身是 data-driven 的, 甚至会随 iteration 演化 (早期关注 "fold 是否发生", 晚期关注 "有没有褶皱")。

加上 VLA 时代的 enabler: π_0.5 这种 model 已经有 language grounding, 所以 text conditioning 直接 work, 不用从头训。这是 FPL 能在 real-world 跑通的关键 infrastructure。

Reference: π_0.5, https://arxiv.org/abs/2504.16054

---

## 一句话 mental model

**Preference 不是 1 维的, 是 K 维的。以前强迫人 collapse 成 1 维, 信息全丢了。FPL 让 K 维显式化, reward model 学得准, policy 学得活, 还能 test-time steer。**

核心就这么个 insight, 剩下的都是怎么把这个 insight 落地到 VLA + real-world robot 的 engineering。

Paper 主页有 video: https://freeform-pl.github.io/fpl.website/

---

# Freeform Preference Learning (FPL): 技术深度解析

Karpathy 你好，这篇 paper 是 Stanford 的 Marcel Torne, Anubha Mahajan, Abhijnya Bhat 和 Chelsea Finn 做的工作，核心 idea 是把 binary preference learning 扩展为 **language-conditioned multi-axis preference learning**。下面我会从 motivation → 数学公式 → architecture → 实验 → intuition 全方位拆解，帮你 build up mental model。

---

## 1. Motivation: 为什么 binary preference 在 long-horizon robot learning 上会崩

传统 RLHF / preference learning 框架（Christiano 2017）假设 annotator 在两个 trajectory 之间给一个 **scalar "overall quality" preference** $y \in \{0,1\}$。这个假设在 LLM 单轮对话场景勉强 work，但放到 long-horizon robotic manipulation 上立刻崩溃，原因有三个：

1. **多 axis 不可比**: 一条 trajectory 快但 rough，另一条慢但 careful。"which is better overall" 根本没有 well-defined 答案，annotator 只能瞎选。
2. **信用分配错乱**: 在 long-horizon 任务里，binary preference 只告诉 policy "整条轨迹好/坏"，不告诉它 **哪个 subtask event 在哪个 timestep 发生**。结果 reward model 学到的 reward 通常在 episode 末尾出现一个大 spike（credit 集中到结尾），中间 dense progress 信号丢失。
3. **Compositionality 缺失**: 二元 reward 把所有 axis 压成 scalar，policy 无法发现 "快" + "右 peg" 这种原本在数据中未同时出现的组合。

FPL 的核心 insight 是 **让 annotator 在 natural language 中自由定义 axis**（"speed", "smoothness", "formality of setup", "quality of 2nd fold" ...），然后对每个 axis 单独收集 pairwise preference。这本质上把 RLHF 从 1-D preference space 推到 K-D axis-conditioned preference space。

Paper 主页: https://freeform-pl.github.io/fpl.website/

---

## 2. 数学公式拆解

### 2.1 Bradley-Terry 基础 (Eq 1-2)

传统 BT 模型假设偏好概率由 reward 差的 logistic 决定：

$$P(s_i \succ s_j) = \frac{\exp(r_\phi(s_i))}{\exp(r_\phi(s_i)) + \exp(r_\phi(s_j))} = \sigma(r_\phi(s_i) - r_\phi(s_j))$$

- $s_i, s_j$: 两个被比较的 states / segments
- $r_\phi$: parameterized reward model, $\phi$ 是其参数
- $\sigma$: logistic sigmoid $\sigma(x) = 1/(1+e^{-x})$
- $\succ$: 偏好关系（读作 "preferred over"）

NLL loss (Eq 2):

$$\mathcal{L}_{\text{BT}}(\phi) = -\mathbb{E}_{(s_i, s_j, y) \sim \mathcal{P}}\left[\log \sigma\big((2y-1)[r_\phi(s_i) - r_\phi(s_j)]\big)\right]$$

- $\mathcal{P}$: pairwise preference dataset
- $y \in \{0,1\}$: preference label, $y=1$ 表示 prefer $s_i$, $y=0$ 表示 prefer $s_j$
- $(2y-1)$ 是把 $\{0,1\}$ 映射到 $\{-1, +1\}$ 的 trick：当 $y=1$ 时 loss 鼓励 $r_\phi(s_i) > r_\phi(s_j)$；$y=0$ 时反向。注意这里是 **positive direction 写法**，跟 Ouyang 2022 (InstructGPT) 的 $(1-2y)$ 写法相反，看实现不同。

Reference: Christiano et al. 2017, https://arxiv.org/abs/1706.03741

### 2.2 FPL 多 axis reward (Eq 3)

FPL 把 BT loss 扩展为 axis-conditioned：

$$\mathcal{L}_{\text{FPL-Reward}}(\phi) = -\mathbb{E}_{(\tau_i, \tau_j, \{(l_k, y_k)|k=1,\ldots,K_{ij}\}) \sim \mathcal{P}}\left[\sum_{k=1}^{K_{ij}} \log \sigma\big((2y_k-1)(r_\phi(\tau_i|l_k) - r_\phi(\tau_j|l_k))\big)\right]$$

变量含义：
- $\tau_i, \tau_j$: 两条完整 trajectories（注意是 trajectory-level 不是 segment-level）
- $l_k$: 第 $k$ 个 preference axis 的 natural language label（如 "speed", "quality of 2nd fold"）
- $y_k \in \{0,1\}$: 沿 axis $l_k$ 哪条 trajectory 更好
- $K_{ij}$: 该 trajectory pair 上的 preference axes 总数，**每个 pair 可以不同**（这很重要，是 "freeform" 的核心 flex）
- $r_\phi(\tau|l_k)$: conditioned reward function, 输入 trajectory + axis text，输出 scalar

关键设计点：**单一 reward network 共享参数**，由 $l_k$ 通过 language condition 进来，而不是训练 K 个独立的 reward model。这避免了 "speed" vs "fast" vs "efficient" 这种语义重复问题，并允许跨 axis 的 representation sharing。

### 2.3 Trajectory-level reward = sum of per-prefix scores (Eq 4)

$$r_\phi(\tau, l_k) = \sum_{i=1}^{T} g_\phi(o_{1:i}, l_k)$$

- $\tau = o_1, \ldots, o_T$: trajectory，$o_t$ 是 observation
- $o_{1:i}$: observation prefix（从 step 1 到 step $i$）
- $g_\phi$: multimodal transformer, 输入 prefix + axis text, 输出该 timestep 的 scalar score
- 最终 trajectory reward 是所有 prefix score 的 sum

这个设计有几个微妙之处：
- 不是给整条 trajectory 一个 score（那样会丢 temporal locality），而是 **每个 timestep 一个 score 再加起来**。这本质上是把 reward 看作 "cumulative score" 的 Monte Carlo 估计。
- $g_\phi$ 用 **causal mask transformer**，token 可以 attend 到 text + 历史图像，不能 attend 未来图像。
- Trajectory reward 的 sum 形式保证了 **additivity**：trajectory 越长、每个 sub-event 越得到 reward，最终 score 越高，符合 "more progress = more reward" 直觉。

### 2.4 Policy extraction (Eq 5)

FPL 不直接用 PPO（real-world sample inefficient），而是用 **reward-conditioned supervised policy learning**（类似 Decision Transformer / RvS）：

$$\mathcal{L}_{\text{FPL-Policy}}(\theta) = -\mathbb{E}_{\tau_i \sim \mathcal{D}}\left[\sum_{t=1}^{T_i} \log \pi_\theta\big(\mathbf{a}_i^t \big| s_i^t, l_1, r_\phi(\tau_i|l_1), \ldots, l_{K_\pi}, r_\phi(\tau_i|l_{K_\pi})\big)\right]$$

- $\pi_\theta$: parameterized policy（实际用 $\pi_{0.5}$ VLA + flow matching head）
- $\mathcal{D}$: offline + on-policy rollout 混合 dataset
- $\mathbf{a}_i^t$: action at timestep $t$
- $s_i^t$: state (camera images)
- $L = \{l_1, \ldots, l_{K_\pi}\}$: 选定的 representative axes（从 freeform 标注里聚合而来）
- $r_\phi(\tau_i|l_k)$: trajectory-level reward along axis $l_k$，作为 conditioning 一起塞进 policy

这个 supervised formulation 本质上是 **behavior cloning with return conditioning**，但 return 是 **multi-dimensional axis-specific** 而非 scalar。可以看作 Decision Transformer 在 multi-objective return-conditioning 下的扩展。

Reference: Decision Transformer, https://arxiv.org/abs/2106.01345
Reference: Reward-Conditioned Policies (Kumar 2019), https://arxiv.org/abs/1912.13465

### 2.5 Reward standardization for test-time conditioning

因为 $r_\phi$ 输出 unbounded，直接 conditioning 不易在 test-time 选合适目标值。FPL 用 **per-axis standardization**:

$$r_{\text{cond}} = \frac{r - \mu}{\sigma}, \quad \mu, \sigma \text{ 来自 } \mathcal{D} \text{ 上 inference 的统计}$$

然后 round 到 1 位小数，作为 text 拼进 VLA prompt，比如:

```
## Fold Shorts
fold the shorts, Quality of 1st fold: 2.2, Wrinkle of 1st fold: 2.1, 
Quality of 2nd fold: 0.7, ..., Fast: 1.2, Smooth: 1.2, 
Damage to environment: -0.1
```

这是把 **reward 当作 natural-language prompt 的一部分** 塞给 VLA——一个相当 elegant 的设计：不需要单独的 reward-conditioning network，直接复用 VLA 的 language understanding 能力。

### 2.6 Baseline losses

**Advantage Conditioning** (π_0.6 baseline, Eq 6-8):

$$r_t = \begin{cases} 0 & \text{if } t=T \text{ and trajectory succeeded} \\ R_{\text{fail}} & \text{if } t=T \text{ and trajectory failed} \\ -1 & \text{otherwise} \end{cases}$$

- $R_{\text{fail}} = -100$ (large penalty for failure)
- $T$: trajectory length

$$V(s_t) = \sum_{k \geq t} r_k \quad \text{(no discount)}$$

$$A_t = V(s_{t+1}) - V(s_t)$$

这是 sparse signal：只从最终 success/fail 反推 advantage。在 long-horizon 任务上 credit assignment 难。

**Weighted Regression** (AWR-like, Eq 9):

$$\mathcal{L}_{\text{AWR}}(\theta) = -\frac{\mathbb{E}_{\tau_i \sim \mathcal{D}}\left[\sum_{t=1}^{T_i} \log \pi_\theta(\mathbf{a}_i^t | s_i^t) \exp\big(\sum_{k}^{K_\pi} r_\phi(\tau_i|l_k)\big)\right]}{\sum_{j=0}^{|\mathcal{D}|} \sum_{k}^{K_\pi} r_\phi(\tau_j|l_k)}$$

这是 **把 multi-axis reward 压成 scalar 加权 BC**——用来 ablate "保留 multi-dim conditioning 是否真的有用"。结果显示这个 baseline 显著弱于 FPL，证明 benefit 来自 **保留 multi-dim conditioning** 而非仅仅是 multi-axis reward learning。

Reference: AWR (Peng 2019), https://arxiv.org/abs/1910.00177
Reference: π_0.6 (Physical Intelligence), https://arxiv.org/abs/2511.14759

---

## 3. Architecture 细节

### 3.1 Reward Model 架构 (Figure 2)

- **Backbone**: Qwen 3.5 VL 4B（real-world）；simulation 用小 MLP from scratch
- **Input format**: text prompt "What is the score for $l_k$?" + strided image sequence (wrist + third-person)
- **Frozen**: vision encoder
- **Fine-tuned**: rest of model (LM 部分)
- **Causal masking**: 所有 token 可 attend text + 过去图像，不能 attend 未来图像
- **Output extraction**: 每对图像取 **second image 的 final vision separator token**，通过 single linear layer 投影到 scalar，然后 sum 跨所有 timestep 得到 trajectory-level reward

实现上有一个 subtle 决策：**stride** 而不是把所有 frame 塞进去（Plate Toast stride=20，Fold Shorts stride=60 等），这平衡了 temporal resolution 和 context length。

### 3.2 Policy Architecture

- **VLA backbone**: $\pi_{0.5}$ open-source weights
- **Action head**: flow matching（不是 diffusion policy 但同源思想）
- **Action chunk size**: 16（一次预测 16 个 action）
- **Action rollout**: 8（执行 8 个再重新预测）
- **Training**: full fine-tune, 30000 steps, batch size 32
- **Conditioning input**: 通过 text 形式（不是 vector embedding）—— 这是非常关键的设计，复用了 VLA 的 in-context text understanding

### 3.3 Simulation 细节差异

Simulation 不用预训练 model，直接 from scratch：
- Reward model: K 个 head，每个 axis 一个 head（无 language conditioning）
- Policy: flow matching DiT
- Conditioning: float vector 直接进 policy（不是 text）
- **必须加 noise 0.2 到 standardized reward**（否则 overfitting；real-world 不需要，因为 VLA 已经预训练好）

这个差异其实揭示了一个有趣的点：**预训练 VLA 的 language grounding 让 text conditioning 天然 generalize**，from-scratch 小网络就不行。

Reference: π_0 / π_0.5, https://arxiv.org/abs/2410.24164 and https://arxiv.org/abs/2504.16054
Reference: Diffusion Policy, https://arxiv.org/abs/2305.19507
Reference: Qwen-VL, https://arxiv.org/abs/2309.16609

---

## 4. 实验结果分析

### 4.1 Real-world 任务（4个）

| Task | 关键挑战 | Preference axes |
|---|---|---|
| Put cube in bowl | steerability test | blue/orange/yellow bowl, speed, smoothness |
| Fold shorts | deformable object | quality of 1st/2nd/3rd fold, wrinkle of 1st/2nd/3rd fold, alignment, speed, smoothness, damage |
| Plate toast | tool use | speed, smoothness, cleanliness, quality, damage |
| Setup table | long-horizon | placement quality × 4 items, formality, smoothness, carefulness, speed, damage |

### 4.2 主结果表（Table 8）

| Method | Setup Table | Plate Toast | Fold Shorts | Place Cube | **Average** |
|---|---|---|---|---|---|
| BC | 0.68 | 0.15 | 0.10 | 0.33 | 0.31 |
| Filtered BC | 0.65 | 0.30 | 0.25 | 0.29 | 0.37 |
| Single Pref (match pairs) | 0.64 | 0.00 | 0.25 | 0.43 | 0.33 |
| Single Pref (match comparisons) | 0.77 | 0.10 | 0.30 | 0.19 | 0.34 |
| **FPL** | **0.94** | **0.70** | **0.55** | **0.81** | **0.75** |

平均 FPL **比 second-best 高 38 pp**。注意 Single Preference 在 Plate Toast 直接 0.00 success——这是因为 task 太复杂，binary "overall" preference 给不出 consistent signal，reward model 学到噪声。

### 4.3 Simulation 主结果（Table 5）

| Method | Obj Rearrange (success) | Bimodal Square (throughput) | Bimodal Square Inv (throughput) |
|---|---|---|---|
| BC | 0.04 | 0.71 | 0.25 |
| Filtered BC | 0.09 | 0.71 | 0.00 |
| Weighted Reg | 0.18 | 0.83 | 0.09 |
| Advantage Cond | 0.17 | 0.75 | 0.08 |
| Single Pref | 0.73 | 0.67 | 0.78 |
| Single Pref (match labels) | 0.79 | 0.73 | 0.74 |
| **FPL** | **0.84** | **1.19** | **1.24** |

Bimodal Square Inv 体现 **steerability**：FPL 在同一个 trained policy 上通过改 test-time conditioning，能在 left peg 上达到 1.24 throughput（比训练目标 right peg 还快），其他 baseline 在 inverted 任务上几乎全崩。

### 4.4 Compositionality 实验（Bimodal Square）

数据集 composition：
- Left peg: 50% fast + 50% slow demos
- Right peg: 100% slow demos（**没有 fast right-peg demo**）

结果：FPL 在 right peg 上达到了 **比训练数据中任何 right peg demo 都快** 的速度，而 single preference baseline 没超过 demo 分布。这证明 FPL 的 multi-axis reward model 学到 "fast" 是一个可分离的 attribute，可以组合到 "right peg" 上。

Figure 4 体现了 throughput 数据，FPL 显著右移。本质上是 reward conditioning 提供了一个 **disentangled latent space**，policy 可以通过 conditioning 在这个 latent space 上做组合。

### 4.5 Dense reward 定性结果（Figure 6）

在 Setup Table rollout 上可视化 reward：
- FPL 的 axis-conditioned reward 在 subtask 完成时刻（放大盘、放小盘、放杯、放餐具）出现 **temporally localized peak**——即使没有任何 explicit subtask segmentation 标注
- Binary preference 的 reward 在 episode 末尾出现一个 giant spike，**即使末尾没有 subtask 完成**——典型的 credit assignment 错误

这是一个非常 striking 的 qualitative finding：**multi-axis preference 隐式地解决了 long-horizon credit assignment**。原因可能是每个 axis（如 "placement of big plate"）只在对应 subtask 期间被比较，所以 reward model 自然学到 temporal localization。

### 4.6 Iterative preference evolution（Figure 10）

Fold Shorts 任务的 iteration 进展：
- 早期 iteration：annotator 关注 "1st/2nd/3rd fold 是否发生"（粗粒度）
- 晚期 iteration：annotator 转向 "wrinkle", "alignment"（细粒度）

这是 FPL 的一个 **emergent curriculum** 性质：因为 axes 是 freeform 的，annotator 自然会随着 policy 改进迁移到更细的 axis。这跟 active learning 的 spirit 接近。

### 4.7 Annotation cost（Figure 9b）

Freeform preferences per label **比 single preferences 快 1.85 倍**。原因是看一对 video 的 overhead 被 K 个 axis label 摊薄了。所以 FPL 不仅 supervision 更 dense，cost-per-label 还更低——一个 Pareto improvement。

---

## 5. Compositionality 与 Steerability 的本质

这是 paper 最 fascinating 的两个 emergent property，我想 build 你的 intuition：

### 5.1 为什么 FPL 能 compose？

考虑 4D conditioning space $\{l_{\text{fast}}, l_{\text{slow}}, l_{\text{left}}, l_{\text{right}}\}$。训练数据里：
- (fast, left) 有
- (slow, left) 有
- (slow, right) 有
- (fast, right) **没有**

Standard BC / single reward 只能 reproduce 训练分布里见过的 (mode, target) 组合。

FPL 把 reward 拆成 axis-independent 后，**policy 在 (fast, right) 这个组合的 conditioning 上看到过类似的 axis-conditioning 信号**（fast 的高 reward + right 的高 reward），即使该具体组合在 trajectory space 不存在。Reward conditioning 等价于一个 **factored latent**，policy 学到的是 "在给定 axis reward 下，怎么 act"——这种 axis-factorized representation 天然支持 interpolation/extrapolation 到没见过的 axis 组合。

这跟 multi-objective RL 的 Pareto frontier 思想是相通的，但 FPL 的 axis 来自 freeform language 而非 pre-defined。

### 5.2 为什么 FPL 能 steer？

Standard reward-conditioned policy 只在 train 时见过 (state, target_reward) pair，test-time 改 target reward 是在 distribution shift（如果数据没覆盖该 reward 区间）。

FPL 在训练时 **不过滤 trajectory**，所以 policy 见过 high/low reward 的 trajectory 全谱。Test-time 改 reward conditioning 不会 OOD，因为训练 distribution 已经覆盖整个 reward range。这就是 place-cube-in-bowl 任务上 FPL 能 test-time switch bowl 的根本原因。

对比 Advantage Conditioning 等 baseline：它们 implicit 把 reward 压成 single scalar，训练 distribution 没有覆盖 "advantage 高 + 不同 target" 的联合分布，所以 test-time 改 advantage 不能改 target。

---

## 6. 与 Related Work 的关系网

### 6.1 RLHF 谱系
- Christiano 2017 (Deep RLHF): FPL 的基础，但 binary preference
- PEBBLE (Lee 2021): 加 active selection，但仍是 binary
- Contrastive Preference Learning (Hejna 2024): 不用 RL，直接 contrastive，但单 axis
- DPO (Rafailov 2023): 不学 reward 直接优化 policy，单 axis
- GRAPE (Zhang 2024): 用 VLM 给 substep preference，仍 binary

Reference: PEBBLE, https://arxiv.org/abs/2106.05098
Reference: DPO, https://arxiv.org/abs/2305.18290
Reference: GRAPE, https://arxiv.org/abs/2411.19309

### 6.2 Fine-grained feedback
- Wu 2023 (Fine-grained HF for LLM): 多个独立 reward model per error category, **固定 categories**
- Liang 2024 (Rich HF for text-to-image): in-context rich feedback, 不是 RL
- FPL 区别: **单个 language-conditioned reward model**, axes 不预定义

Reference: Fine-grained HF, https://arxiv.org/abs/2310.05318

### 6.3 Multi-objective RL
- MORL (Moffaert 2013, Roijers 2013): 预定义 objective weights → scalarize
- Promptable Behaviors (Hwang 2024): 用 natural language 指定 multi-objective preference, **但需要预定义 reward functions**
- FPL 区别: **axes 也从 freeform preferences 学出来**

Reference: Promptable Behaviors, https://arxiv.org/abs/2312.09371

### 6.4 Real-world RL on VLAs
- π_0.5/0.6 (Physical Intelligence): advantage conditioning + sparse success signal
- OpenVLA (Kim 2024): open VLA, RL 待补
- GR00T N1 (NVIDIA 2025): foundation humanoid model
- Diffusion Policy (Chi 2023): action head 思想
- MEM (Torne 2026): multi-scale memory for VLA

Reference: π_0, https://arxiv.org/abs/2410.24164
Reference: OpenVLA, https://arxiv.org/abs/2406.09246
Reference: GR00T N1, https://arxiv.org/abs/2503.14734
Reference: Diffusion Policy, https://arxiv.org/abs/2305.19507

### 6.5 Natural language for robot feedback
- Sharma 2022 (Correcting plans with language): language as corrective feedback
- Peng 2024 (Preference-conditioned abstraction): language as state abstraction
- FPL 区别: language as **explicit axis definition interface**

---

## 7. Intuition Building: 为什么 FPL work？

我帮你总结几个关键 intuition：

### Intuition 1: Annotation 不确定性是 information bottleneck
Binary preference 强迫 annotator 把多维判断 collapse 成 1 bit。每次 collapse 都损失 information。FPL 让每个 axis 一个 bit，**信息容量从 1 bit/label 提升到 K bit/label**。Reward model 见到的 supervision 直接 dense K 倍。

### Intuition 2: Axis-conditioning 隐式做 credit assignment
Binary reward 在 trajectory end 给 spike 是因为 reward model 不知道**哪段 trajectory 对应哪个 sub-event**。FPL 的 axis like "placement of big plate" 天然告诉 reward model **关注 big plate placement 事件**，所以 reward 自动 localize 到那个 timestep。这是 **language 作为 attention prior** 的工作。

### Intuition 3: Multi-axis reward 是 disentangled latent space
Single reward 把所有 axis 压成 scalar = entangled representation。Policy 必须在 entangled 空间上做 RL，组合泛化难。Multi-axis reward = **axis-disentangled representation**，policy 通过 reward conditioning 自然获得 compositionality 和 steerability。这跟 disentangled VAE / β-VAE 的思想异曲同工。

### Intuition 4: Freeform > Predefined axes
如果固定 axes（"speed, quality, safety"），会限制 annotator 的表达力。FPL 允许 axis 本身是动态的（早期 "fold happens" vs 晚期 "wrinkle level"），形成自动 curriculum。Predefined axes 需要人来 design curriculum，freeform 让 curriculum 从 annotation 数据涌现。

### Intuition 5: 复用 VLA 的 language grounding 是关键
Real-world 不需要 vector conditioning（直接 text），是因为 π_0.5 已经在 huge language+robot 数据上预训练。Simulation from-scratch 必须用 vector + noise。这说明 **VLA 的 language pretraining 是 FPL 在 real-world 可行的根本 enabler**。这个 connection 在 paper 里是 implicit 的，但很关键。

### Intuition 6: Reward-conditioned policy > PPO 在 real-world
PPO 在 real-world 不可行（sample efficiency 太差）。Reward-conditioned supervised learning（RvS 谱系）是 offline-friendly 的。FPL 选这条路 + 加 multi-axis conditioning = real-world tractable + compositionality。

---

## 8. Limitations & 我自己的思考

Paper 自己列的 limitations:
1. 仍需 human preference（vs unsupervised）
2. Test-time reward value 需 manual selection
3. 固定 K_π axes（不能 variable-length conditioning）

我觉得还有几个值得深入思考的方向：

1. **Reward model 的 calibration**: 不同 axis 的 standardized reward 是否真的 comparable？比如 "speed 1.2" 和 "smoothness 1.2" 对 policy 行为的影响权重一致吗？Paper 没仔细讨论这个。
2. **VLM hallucination**: Reward model 基于 Qwen-VL，它可能 hallucinate 哪个 trajectory 更快——尤其当两个 trajectory 速度差异 subtle 时。Paper 没报告 reward model 自身的 preference accuracy。
3. **Curse of axis dimensionality**: 如果 axes 数量增长（比如 50 个 axis），text prompt 会长到 VLA context limit。如何 select 子集是个 open problem。
4. **Multi-annotator disagreement**: 不同 annotator 用不同 axis 名（"fast" vs "speed"），paper 靠 VLM 的 semantic embedding 解决，但如果两个 annotator 对 "formality" 定义不同怎么办？
5. **Reward hacking across axes**: Policy 可能学会同时让所有 axis reward 高的捷径行为，绕过 task 真正的物理目标。Paper 没看到这个 failure mode 的讨论。

---

## 9. 给 Karpathy 的 TL;DR

如果你 30 秒读完，这是我推荐的 mental model:

**FPL = RLHF × Multi-axis Language Conditioning × Reward-conditioned VLA fine-tuning**

三个关键 choice:
1. Annotator 用 natural language 定义任意 axis，每对 trajectory 给 K 个 axis-wise preference（vs 1 个 binary）
2. 单个 language-conditioned reward model 学所有 axis 的 reward（共享 backbone，text condition）
3. Policy 以 axis-reward pairs 作为 text prompt 一部分 conditioning，学 multi-axis reward-conditioned policy

三个 emergent property:
- **Dense reward on long-horizon tasks**（隐式 credit assignment via axis localization）
- **Compositionality**（axis-disentangled representation 支持训练分布外的 axis 组合）
- **Test-time steerability**（训练不过滤 trajectory，test-time 改 conditioning 不 OOD）

实测: real-world 4 task avg 38pp 提升，annotation cost-per-label 降 1.85x。

---

## 主要 References

1. **FPL 主页**: https://freeform-pl.github.io/fpl.website/
2. **Christiano et al. 2017 (Deep RLHF)**: https://arxiv.org/abs/1706.03741
3. **π_0 / π_0.5 (Physical Intelligence)**: https://arxiv.org/abs/2410.24164, https://arxiv.org/abs/2504.16054
4. **π_0.6 (advantage conditioning baseline)**: https://arxiv.org/abs/2511.14759
5. **DROID dataset**: https://droid-dataset.github.io/
6. **Decision Transformer**: https://arxiv.org/abs/2106.01345
7. **Reward-Conditioned Policies (Kumar 2019)**: https://arxiv.org/abs/1912.13465
8. **Diffusion Policy**: https://arxiv.org/abs/2305.19507
9. **OpenVLA**: https://arxiv.org/abs/2406.09246
10. **DPO**: https://arxiv.org/abs/2305.18290
11. **Qwen-VL Technical Report**: https://arxiv.org/abs/2309.16609
12. **Robomimic**: https://arxiv.org/abs/2108.03298
13. **Fine-grained HF (Wu 2023)**: https://arxiv.org/abs/2310.05318
14. **Promptable Behaviors**: https://arxiv.org/abs/2312.09371
15. **GR00T N1**: https://arxiv.org/abs/2503.14734
16. **PPO**: https://arxiv.org/abs/1707.06347

---

希望这个拆解帮你 build 起对 FPL 的 mental model。整个工作的核心 insight 是 **"preference 的 axis structure 本身就是 information"**——binary preference 强迫 annotator 把这个 structure 抹掉，而 freeform preference 让 structure 显式化并直接进入 reward model 与 policy。在 VLA 时代，language 既是 human-AI interface 又是 model 内部 conditioning representation，FPL 把这两个角色统一起来，这是为什么它在 real-world 机器人任务上能 work 得这么好。
