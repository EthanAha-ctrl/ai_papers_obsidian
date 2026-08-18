---
source_pdf: Fast LeWorldModel.pdf
paper_sha256: 2714f69dd4b279fd77b046b8497602b61049f41587310d1637fbabb691f1d9ff
processed_at: '2026-08-18T12:38:54-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Fast LeWorldModel

---

## 一句话版本

LeWM 像"走一步看一步"地规划，每走一步要等前一步的结果才能算下一步；Fast-LeWM 像"一眼看五步"，同时看到走1步、2步、3步、4步、5步后分别会到哪里，所有预测都锚在现在的状态上，互不依赖，所以又快又准。

---

## 这篇 paper 到底在解决什么问题

想象你在下棋。LeWM 的做法是：先想象走第1步后的棋盘，再从那个想象的棋盘想象第2步，再从第2步的想象棋盘想象第3步……一直嵌套到第5步。每个想象都是近似值，想象里有误差，误差会被一层层放大。更糟的是，如果你是 CEM 这种 planner，要同时评估几百个候选走法，每个走法都要这么一步步滚下来，慢得要命。

Fast-LeWM 的 idea 很简单也很漂亮：我直接从当前真实棋盘跳到"走完前k步后的棋盘"，k=1,2,3,4,5 全部并行算出来。中间棋盘的预测互不喂回，全部 anchor 在当前真实观察到的状态上。这就像你站在棋盘前，脑子里同时浮现出"走1步是这样，走2步是那样，走3步是另一种……"，每个画面都直接来源于现在这个真实棋盘，而不是从上一个想象叠加下一个想象。

参考 LeWorldModel 原文: https://arxiv.org/abs/2603.19312

---

## 两种 rollout 拓扑的对比

### LeWM：chain composition（链式）

```
z_t ──a_t──> ẑ_{t+1} ──a_{t+1}──> ẑ_{t+2} ──a_{t+2}──> ẑ_{t+3} ...
              (近似)              (基于近似)          (基于近似的近似)
```

每一步 forward 都依赖上一步的 output。第5步的 ẑ_{t+5} 实际上嵌套了5次 F_φ 调用，误差像 Euler method 积分一样累积。

公式长这样：
$$\hat{\mathbf{z}}_{t+k} = G_{\text{LeWM}}(G_{\text{LeWM}}(\cdots G_{\text{LeWM}}(\mathbf{z}_t, \mathbf{a}_t), \cdots), \mathbf{a}_{t+k-1})$$

下标 $t+k$ 是时间步，$\mathbf{z}_t$ 是当前真实 latent，$\mathbf{a}_t$ 是当前 action，$G_{\text{LeWM}}$ 是 one-step transition function，$\hat{\mathbf{z}}$ 是 predicted latent（帽子表示 prediction）。

### Fast-LeWM：star fan-out（星形扇出）

```
                  ──prefix_1──> ẑ_{t+1}
                  ──prefix_2──> ẑ_{t+2}
z_t (真实 anchor) ──prefix_3──> ẑ_{t+3}     全部并行
                  ──prefix_4──> ẑ_{t+4}
                  ──prefix_5──> ẑ_{t+5}
```

公式：
$$\hat{\mathbf{z}}_{t+k} = G_{\text{Fast-LeWM}}(\mathbf{z}_t, \mathbf{a}_{t:t+k-1}), \quad k=1,\ldots,H$$

这里 $\mathbf{a}_{t:t+k-1} = (\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+k-1})$ 是长度为 $k$ 的 action prefix，$\mathbf{z}_t$ 是 anchor latent（来自 encoder，真实的）。每个 horizon $k$ 直接从 anchor 跳过去，互相之间没有依赖关系。

这个 topological 变化同时解决三个问题：
1. **速度**：5次串行 forward 变成1次并行 forward，GPU parallelism 友好。
2. **误差累积**：每个 prediction 都 anchor 在真实 latent 上，不依赖前一个近似 prediction。
3. **representation 质量**：dense supervision 迫使 latent 保留多 horizon 都用得上的物理信息。

---

## 训练时的 dense supervision 为什么重要

训练数据里，我们有真实的 $(o_t, o_{t+1}, o_{t+2}, \ldots, o_{t+H})$ 这一串 observation。把它们全部 encode 成 latent $\mathbf{z}_t, \mathbf{z}_{t+1}, \ldots, \mathbf{z}_{t+H}$ 作为 target。

Fast-LeWM 的 loss 是每个 prefix 都被监督：
$$\mathcal{L}_{\text{prefix}} = \frac{1}{H}\sum_{k=1}^{H}\|\hat{\mathbf{z}}_{t+k} - \mathbf{z}_{t+k}\|_2^2$$

下标 $t+k$ 是时间步，$H$ 是最大 horizon，$\hat{\mathbf{z}}_{t+k}$ 是 model 预测的 latent，$\mathbf{z}_{t+k}$ 是从真实 observation encode 出来的 target latent，$\|\cdot\|_2^2$ 是欧氏距离平方，外层 $\frac{1}{H}\sum$ 是对 $H$ 个 horizon 取平均。

这个 loss 告诉 model："走完前1步应该到 $\mathbf{z}_{t+1}$，走完前2步应该到 $\mathbf{z}_{t+2}$，走完前3步应该到 $\mathbf{z}_{t+3}$……"所有中间 outcome 都被显式监督。

对比 LeWM 只监督 one-step：$\|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_2^2$，后续 horizon 是通过 composition 间接学习的。

实验里 ablation（Table 4）证实 dense supervision 是关键。只监督 terminal 的 Terminal-only Fast-LeWM 在4个 task 平均只有 84.5%，加上 dense prefix supervision 升到 90.5%。

参考 ablation 数据 Table 4: Long-Action LeWM 71.0% avg, Terminal-only Fast-LeWM 84.5% avg, Full Fast-LeWM 90.5% avg。

---

## Action-Prefix Encoder 在干嘛

给定 candidate action sequence $(\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1})$，encoder 要输出 $H$ 个 prefix token $(\mathbf{p}_{t,1}, \mathbf{p}_{t,2}, \ldots, \mathbf{p}_{t,H})$。

要求：$\mathbf{p}_{t,k}$ 只包含前 $k$ 个 action 的累积信息，不能偷看后面的 action。

实现是 causal Transformer：在位置 $k$ 的输出，attention mask 让它只能 attend 到 position $1, 2, \ldots, k$，看不到 $k+1$ 以后。所以：
$$\mathbf{p}_{t,k} = E_\psi^{(k)}(\mathbf{a}_t, \ldots, \mathbf{a}_{t+k-1})$$

$E_\psi$ 是 encoder，$\psi$ 是参数，上标 $(k)$ 表示"在位置 $k$ 的输出"，输入是前 $k$ 个 action。

**关键 trick：state token prepending。** 同样的 action sequence 在不同 scene 下效果完全不同。比如 PushT 里 agent 起点变了，同样 push 5次，block 飞向的方向完全不一样。Prefix 本身不带 scene 信息。

所以作者用一个 2-layer MLP 把当前 latent $\mathbf{z}_t$ 映射成一个 state token，prepend 到 action token 序列的第0位。encoder 实际上是：
$$\mathbf{p}_{t,k} = E_\psi^{(k)}(\mathbf{a}_t, \ldots, \mathbf{a}_{t+k-1} \mid \mathbf{z}_t)$$

竖线 $|$ 表示 conditional on，意思是 prefix encoding 是以当前 $\mathbf{z}_t$ 为条件算出来的。这让 model 区分"同样的 prefix 在不同 scene 下产生什么效果"。

Ablation 里去掉 state token 性能降 2-4 个点（Table 4, w/o state token）。

参考 causal Transformer / GPT 思想: https://arxiv.org/abs/1706.03762

---

## Parallel Latent Predictor 在干嘛

拿到 anchor latent $\mathbf{z}_t$ 和 $H$ 个 prefix token $(\mathbf{p}_{t,1}, \ldots, \mathbf{p}_{t,H})$，predictor 一次性算出 $H$ 个 future latent：
$$\hat{\mathbf{z}}_{t+k} = G_\phi(\mathbf{z}_t, \mathbf{p}_{t,k}), \quad k=1,\ldots,H$$

$G_\phi$ 是 predictor，$\phi$ 是参数。每个 horizon $k$ 用自己的 prefix token $\mathbf{p}_{t,k}$ 告诉 predictor "应施加多少累积 action effect 到 anchor latent"。

实现是 6-layer action-modulated residual MLP，用 AdaLN-zero modulation。AdaLN-zero 来自 DiT (Diffusion Transformer, https://arxiv.org/abs/2212.09748)，原本是 diffusion model 里用 conditioning 调制每层 normalization 的 scale 和 shift。这里 prefix token 作为 conditioning，调制 anchor latent 在每层 residual block 的 transformation。

总参数 17.9M，跟 LeWM 18.0M checkpoint 几乎一样大——所以性能提升不是堆参数堆出来的。

---

## Planning：CEM 用 Fast-LeWM 怎么评分

CEM (Cross-Entropy Method, 参考 https://link.springer.com/book/10.1007/978-1-4321-0501-0 ) 是一个 sampling-based optimizer。它采样一堆 candidate action sequence，用 cost function 评分，选 elite candidate 更新 sampling distribution，迭代几轮后输出 best sequence。

LeWM 对每个 candidate $m$ 评分：
$$C_{\text{goal}}^{(m)} = \|\hat{\mathbf{z}}_{t+H}^{(m)} - \mathbf{z}_g\|_2^2$$

$\hat{\mathbf{z}}_{t+H}^{(m)}$ 是 candidate $m$ 执行 $H$ 步后预测到达的 latent，$\mathbf{z}_g$ 是 goal observation encode 出来的 latent，距离越小越好。LeWM 要 autoregressive rollout $H$ 次拿到 $\hat{\mathbf{z}}_{t+H}^{(m)}$；Fast-LeWM 只需一次 parallel forward。

**Self-consistency 加料：** 除了直接从长度-$H$ prefix 预测 $\hat{\mathbf{z}}_{t+H}^{(m)}$，还可以"先走一个中间 prefix（比如长度2），拿到中间 latent $\hat{\mathbf{z}}_{t+2}^{(m)}$，再从中间 latent 走剩余 prefix（长度3）"，得到另一个估计 $\tilde{\mathbf{z}}_{t+H}^{(m)}$。两个估计理论上应该一致，不一致说明 model 对该 candidate 的 rollout 不自信。

新 cost：
$$C^{(m)} = C_{\text{goal}}^{(m)} + \beta\|\hat{\mathbf{z}}_{t+H}^{(m)} - \tilde{\mathbf{z}}_{t+H}^{(m)}\|_2^2$$

$\beta \geq 0$ 控制一致性项强度，$\beta=0$ 退化到普通 goal-only CEM，$\beta$ 大时 CEM 偏好 terminal 预测在不同 prefix decomposition 下稳定的 candidate。实验里 $\beta=1$。

直觉：这像 ensemble self-distillation，同一 model 用不同路径预测同一终点，路径间一致说明这条 path 可信。

---

## 实验数据讲人话

### 速度对比（Table 2, NVIDIA 4090）

| Method | Model calls | Dynamics time | CEM time |
|---|---|---|---|
| LeWM | 5次串行 | 31.4秒 | 54.4秒 |
| Fast-LeWM | 1次并行 | 8.0秒 | 28.3秒 |

Dynamics 模块快了 3.9 倍。整体 CEM 快了差不多 2 倍。注意整体没快 3.9 倍，因为 CEM 还有 goal encoding、score computation、data operation 这些开销不在 dynamics 模块里。

### 成功率对比（Table 1）

| Method | Two-Room | Reacher | PushT | Cube | Avg |
|---|---|---|---|---|---|
| PLDM | 97 | 78 | 78 | 65 | 79.5 |
| DINO-WM | 100 | 79 | 74 | 86 | 84.8 |
| LeWM | 87 | 86 | 96 | 74 | 85.8 |
| Fast-LeWM | 98 | 88 | 96 | 80 | 90.5 |
| +Self-Consistency | 98 | 90 | 98 | 82 | 92.0 |

Fast-LeWM 在每个 task 上都不低于 LeWM，平均涨 4.7 个点。Self-consistency 再涨 1.5 个点，在 PushT (96→98) 和 Reacher (88→90) 这种 action effect 路径多样的 task 上特别有用。

### Open-loop 预测误差（Figure 3）

给定真实初始 frame + 真实 action sequence，比较预测 latent 和真实 future latent 的距离随时间变化。

两个观察：
1. **初始误差低**：Fast-LeWM 在 $t=25$ 处（一个 max-horizon）的 latent loss 明显低于 LeWM。
2. **增长斜率小**：作者用最小二乘法拟合 loss curve 取 slope，Fast-LeWM 的 slope 在4个 task 上一致更小。

为什么？LeWM 在 $t=25$ 处已经 chain 5次误差，$t=50$ 处 chain 10次误差。Fast-LeWM 在 $t=25$ 处只做一次 anchor-to-horizon 预测，$t=50$ 处只 chain 两次 anchor-to-horizon 跳跃。chain 长度从 10 降到 2，累积误差自然小。

### Physical probing（Table 3, PushT）

冻 encoder，训练 linear probe 和 MLP probe 预测 agent 位置、block 位置、block 角度。

Linear probe 上 Fast-LeWM 和 LeWM 差不多（block angle 甚至差一点）。MLP probe 上 Fast-LeWM 全面胜出，block angle 的 MSE 从 0.021 降到 0.009。

直觉：linear probe 测"信息是否 linearly 直接可读"，MLP probe 测"信息是否 nonlinearly 可恢复"。Fast-LeWM 把 physical 信息编码得更"非线性可读"——可能因为 dense prefix supervision 逼着 latent 保留多 horizon 都用得上的精细 physical variable，这些 variable 不是 linearly layout 在 latent 上，而是 distributed 在 representation 里。

### Ablation 讲人话（Table 4）

| Variant | Two-Room | Reacher | PushT | Cube |
|---|---|---|---|---|
| Long-Action LeWM | 76 | 70 | 80 | 58 |
| Terminal-only Fast-LeWM | 96 | 80 | 90 | 72 |
| Full Fast-LeWM | 98 | 88 | 96 | 80 |
| w/o state token | 94 | 82 | 92 | 80 |

**Long-Action LeWM 崩盘**：把 LeWM 的 action block 从 5 个 primitive step 扩到 25 个，one-step 直接预测 25 步后。性能从 85.8 avg 掉到 71.0 avg。说明简单拉长 transition 覆盖的 temporal span 行不通——没有中间 supervision，model 学不出可靠的 long-range transition。

**Terminal-only Fast-LeWM 已经 84.5 avg**：即使只监督 terminal latent，prefix representation 本身就比 raw long action block 强得多。因为 causal Transformer 把 sequence 的 order structure 和 cumulative effect 显式 encode 进 token，比把 25 个 action 拼成一个大 vector 喂进去有效。

**Dense supervision 再加 4 个点**：中间 prefix token 被显式约束对应 partial action outcome，把 sequence 内部 order structure 转化为 supervision 信号。

**State token 去掉降 2-4 个点**：同样 action prefix 在不同 scene 下效果不同，没有 state token 提供的 context，model 无法 disambiguate action effect。

---

## 一个最深的直觉

LeWM 的 prediction graph 是一条链，每个节点是近似的，链越长误差越大，planning 时只能串行走链。

Fast-LeWM 的 prediction graph 是一棵星，所有叶子节点都锚在真实的 anchor latent 上，互不串行依赖，可以并行算，误差不在 prediction 内部累积。

这一个 topological 转变同时解决了速度、误差累积、representation 质量三个问题。这不是简单的工程优化，这是对 dynamics interface 的重新设计——把 "transition function" 这个从 classical control 继承来的概念，重新定义成 "action-prefix-conditioned multi-horizon predictor"。

类比一下：这跟 value function learning 里从 TD bootstrap（one-step bootstrap）到 Monte Carlo return（直接预测多步累积 reward）的转变同构。TD 是 local one-step，MC 是 multi-step direct prediction。Fast-LeWM 是 "latent state 的 MC return 预测"。

也跟 video diffusion model 里从 autoregressive frame generation 到 anchor-conditioned multi-frame denoising 的转变同构。参考 Video Diffusion Models: https://arxiv.org/abs/2204.03458

也跟 Decision Transformer (https://arxiv.org/abs/2106.01345) 在 latent space 里的精炼版有联系——causal mask 让每个 position 只看过去，但 Fast-LeWM 在 latent space 做，且用 parallel predictor fan-out 所有 future，避免 autoregressive generation 的累积问题。

---

## 我读出来的局限

1. **Horizon H=5 没测更大的。** Prefix encoder 是 causal Transformer，attention 是 $O(H^2)$。如果 H=20、50，计算和 representation 容量是否还撑得住？

2. **Anchor latent 假设。** 所有 prediction 锚在 $\mathbf{z}_t$ 上。如果 mid-trajectory 发生 contact discontinuity（比如 PushT 里 block 突然撞墙），从 anchor 直接跳 25 步可能丢失突变信息。Self-consistency 部分缓解，但没彻底解决。

3. **State token 只 map 一次。** 同一 anchor latent 被 5 个 prefix 共享。对高度 state-dependent 的 dynamics（比如 robot arm 关节极限附近），可能建模不够细。未来可以做 per-prefix state modulation。

4. **Beta 没自动调节。** Self-consistency 的 $\beta=1$ 是手调的。理想情况下 $\beta$ 应随 CEM iteration 自适应——早期 exploration（低 $\beta$），后期 exploitation（高 $\beta$）。

5. **没测 long-horizon planning。** Paper 里 planning horizon H=5 对应 25 个 primitive step。对几十步、上百步的 long-horizon 任务，prefix 接口是否还能撑住没验证。

代码: https://github.com/Yuntian-Gao/Fast-LeWorldModel

---

## 一句话收尾

Fast-LeWM 的贡献核心是重新定义了 world model 的 prediction interface：从 chain-style one-step transition 变成 star-style multi-horizon prefix prediction。这个 reframe 同时让 planning 快了 2 倍、误差累积慢了、representation 物理信息更丰富。paper 干净、ablation 到位、intuition 清晰——属于"看一眼就记住核心 idea"的那类好 paper。

---

# Fast LeWorldModel (Fast-LeWM) 详解

下面我从背景动机、核心idea、数学形式、架构细节、planning流程到实验数据，逐层拆解这篇 paper，目标是帮你在脑子里搭起完整的 intuition。

---

## 1. 背景与动机：为什么 LeWM 的 rollout 接口是瓶颈

### 1.1 JEPA-style world model 的基本设定

在 reward-free、goal-conditioned visual planning 里，模型从 offline trajectory dataset $\mathcal{D}=\{\tau^{(n)}\}$ 学习，每个 trajectory 是 $(o_t, \mathbf{a}_t)$ pair。其中 $o_t$ 是 pixel observation，$\mathbf{a}_t \in \mathbb{R}^{d_a}$ 是 continuous action。

视觉 encoder 把 observation 映射到 latent：
$$\mathbf{z}_t = f_\theta(o_t), \quad \mathbf{z}_t \in \mathbb{R}^d$$
goal observation $o_g$ 也同样 encode 成 $\mathbf{z}_g$。Planning 就是在 latent space 找一个 action sequence，使预测的 future latent 接近 $\mathbf{z}_g$。

参考 LeWM (LeWorldModel) 论文: https://arxiv.org/abs/2603.19312 ，JEPA 的核心思想是 reconstruction-free，即不重建 pixel，只预测 future embedding。这避免了 latent 被迫保留对 control 无关的视觉细节。JEPA 原始 paper: https://arxiv.org/abs/2301.08243 。

### 1.2 LeWM 的 local one-step transition interface

LeWM 学的是一个 local transition predictor：
$$\hat{\mathbf{z}}_{t+1} = F_\phi(\mathbf{z}_t, \mathbf{a}_t)$$

训练 loss 是 next-latent prediction + SIGReg 防止 collapse：
$$\mathcal{L}_{\text{1step}} = \|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_2^2 + \lambda \cdot \text{SIGReg}(Z)$$

这里 $\|\cdot\|_2^2$ 是欧氏距离平方，$Z$ 是 batch 内 latent 集合，$\lambda$ 是正则权重。SIGReg (Spectral Inertia Gaussian Regularization) 是 Balestriero & LeCun 2025 提出的 anti-collapse regularizer，参考 https://arxiv.org/abs/2503.13926 (LeJEPA)。

### 1.3 两个核心痛点

给定 candidate action sequence $\mathbf{a}_{t:t+H-1}$，要算出 terminal latent $\hat{\mathbf{z}}_{t+H}$，必须 autoregressive 地滚动：
$$\hat{\mathbf{z}}_{t+k} = G_{\text{LeWM}}(G_{\text{LeWM}}(\cdots G_{\text{LeWM}}(\mathbf{z}_t, \mathbf{a}_t), \cdots), \mathbf{a}_{t+k-1})$$

这里 $k=1,\ldots,H$，下标 $t+k$ 表示时间步。这个嵌套 composition 带来两个问题：

**痛点一：慢。** CEM (Cross-Entropy Method) 要评估大量 candidate sequence，每个 sequence 都要 sequential 跑 $H$ 次 forward。dynamics module 时间被 action encoding 和 latent prediction 重复调用占满。

**痛点二：error accumulation。** 早期/中间的 predicted latent $\hat{\mathbf{z}}_{t+1}, \ldots, \hat{\mathbf{z}}_{t+H-1}$ 是 approximated 的，却作为下一步 input 再次喂进去。error 被 recursive 累积，horizon 越长越不可靠。

直觉上，这就像你用 Euler method 积分 ODE——步长固定为1，每步误差累积进下一步。Fast-LeWM 的核心 idea 就是：能不能跳过中间 imagined state，直接从 anchor latent 跳到目标 horizon？

---

## 2. 核心思想：把 "action prefix" 当成 prediction unit

### 2.1 从 single-step transition 到 action-prefix prediction

定义 action prefix 为长度 $k$ 的子序列：
$$\mathbf{a}_{t:t+k-1} = (\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+k-1}), \quad k=1,\ldots,H$$

Fast-LeWM 直接预测执行完整个 prefix 后到达的 future latent：
$$\hat{\mathbf{z}}_{t+k} = G_{\text{Fast-LeWM}}(\mathbf{z}_t, \mathbf{a}_{t:t+k-1}), \quad k=1,\ldots,H$$

这里 $\mathbf{z}_t$ 是 anchor latent（直接来自 encoder，不是 imagined 的），$k$ 是 horizon index。

关键转变：LeWM 是 $G(\mathbf{z}_t, \mathbf{a}_t) \to \mathbf{z}_{t+1}$，然后递归；Fast-LeWM 是 $G(\mathbf{z}_t, \text{prefix}_k) \to \mathbf{z}_{t+k}$，每个 horizon 直接从 anchor 跳过去。这把 multi-horizon state evolution 从 "chain composition" 变成 "star fan-out"。

### 2.2 为什么 prefix supervision 带来 representation 上的好处

不同 prefix 包含不同程度的 accumulated action effect，对应不同 future latent。dense prefix-level supervision 迫使 model 学习：state 在不同长度 action prefix 下如何持续演化。这逼着 encoder 和 dynamics module 保留 fine-grained physical variable（agent 位置、block 角度、contact state 等），因为只有这些变量能预测中间 horizon 的 outcome。

LeWM 只学 $t \to t+1$，representation 只需支撑 local transition；Fast-LeWM 学所有 $k$，representation 必须支撑 "action 序列累积后到达哪里" 的多 horizon query。

---

## 3. 方法详解：Action-Prefix Encoder + Parallel Predictor

### 3.1 Action-Prefix Encoder

输入：candidate action sequence $\mathbf{a}_{t:t+H-1}$，输出 dense prefix token 序列 $\mathbf{p}_{t,1:H}$。

要求：第 $k$ 个 prefix token $\mathbf{p}_{t,k}$ 只能 attend 到 $\mathbf{a}_t, \ldots, \mathbf{a}_{t+k-1}$，防止 future action 信息泄漏到 short prefix。

实现：causal Transformer over action tokens。在位置 $k$ 的 representation 由于 causal mask 只能看到前 $k$ 个 action，得到：
$$\mathbf{p}_{t,k} = E_\psi^{(k)}(\mathbf{a}_t, \ldots, \mathbf{a}_{t+k-1})$$

这里 $E_\psi$ 是 encoder，$\psi$ 是参数，上标 $(k)$ 表示在位置 $k$ 的输出函数。

**关键扩展：state token prepending。** 同样的开环 action 在不同 scene configuration 下会产生完全不同效果——agent 起点不同、object contact 不同。Prefix 本身无法决定 outcome。所以作者用一个 2-layer MLP 把当前 latent $\mathbf{z}_t$ map 成一个 state token，prepend 到 action token sequence 作为第 0 个 token：
$$\mathbf{p}_{t,k} = E_\psi^{(k)}(\mathbf{a}_t, \ldots, \mathbf{a}_{t+k-1} \mid \mathbf{z}_t)$$

这个 conditional 形式让 model 区分 "同样的 prefix 在不同 state 下产生什么效果"。State token 提供 scene geometry、contact constraint 等 context。

架构超参：3 层 Transformer，6 个 attention head，per-head dimension 32，token dimension 192，sinusoidal positional encoding。

### 3.2 Parallel Latent Predictor

给定 anchor latent + prefix token 序列，并行预测所有 future latent：
$$\hat{\mathbf{z}}_{t+1:t+H} = G_\phi(\mathbf{z}_t, \mathbf{p}_{t,1:H})$$

等价形式：
$$\hat{\mathbf{z}}_{t+k} = G_\phi(\mathbf{z}_t, \mathbf{p}_{t,k}), \quad k=1,\ldots,H$$

每个 horizon $k$ 用自己的 prefix token $\mathbf{p}_{t,k}$ 指定 "应施加多少累积 action effect 到 anchor latent"。

实现：6-layer action-modulated residual MLP，latent dimension 192，hidden width 2048，fusion width 768，AdaLN-zero modulation（参考 DiT: https://arxiv.org/abs/2212.09748 ），dropout 0.1。AdaLN-zero 让 prefix token 通过 scale/shift 调制每层 residual block，类似 conditional diffusion model 里的 conditioning 机制。

总参数 17.9M，和 LeWM 18.0M checkpoint 几乎一样——所以性能提升不是堆参数带来的。

### 3.3 Dense Prefix Prediction Objective

训练 segment 是 $(o_t, \mathbf{a}_t, o_{t+1}, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1}, o_{t+H})$，先 encode 所有 observation：
$$\mathbf{z}_{t+i} = f_\theta(o_{t+i}), \quad i=0,\ldots,H$$

dense loss：
$$\mathcal{L}_{\text{prefix}} = \frac{1}{H}\sum_{k=1}^{H}\|\hat{\mathbf{z}}_{t+k} - \mathbf{z}_{t+k}\|_2^2$$

这里每个 prefix 都拿到对应 future latent 作为 target，平均 over $H$ 个 horizon。加上 SIGReg：
$$\mathcal{L}_{\text{AP}} = \mathcal{L}_{\text{prefix}} + \lambda \cdot \text{SIGReg}(Z)$$

这个 dense objective 是 ablation 里最关键的组件——terminal-only variant 性能明显下降（Table 4），说明中间 prefix supervision 是把"action 序列内部 order structure"显式暴露给 model 的核心机制。

---

## 4. Planning：CEM + Self-Consistency

### 4.1 基础 cost function

CEM 采样 candidate sequence，每个 sequence $m$ 算 cost：
$$C_{\text{goal}}^{(m)} = \|\hat{\mathbf{z}}_{t+H}^{(m)} - \mathbf{z}_g\|_2^2$$

Fast-LeWM 直接拿第 $H$ 个 prefix token $\mathbf{p}_{t,H}^{(m)}$ 跑一次 parallel predictor 就拿到 $\hat{\mathbf{z}}_{t+H}^{(m)}$，无需 autoregressive rollout。

CEM 更新 sampling distribution 选 elite candidate，执行 selected action 到下一个 decision point——和 LeWM 完全相同的 MPC schedule，只是 rollout interface 换了。

CEM 原始参考: Rubinstein & Kroese book, https://link.springer.com/book/10.1007/978-1-4757-4321-0 。

### 4.2 Self-consistency penalty

prefix interface 提供一个额外的 model-consistency 信号。除了直接从长度-$H$ prefix 预测 terminal latent $\hat{\mathbf{z}}_{t+H}^{(m)}$，model 还可以"先走一个中间 prefix step，再从中间 latent 预测剩余 horizon"，得到另一估计 $\tilde{\mathbf{z}}_{t+H}^{(m)}$。

consistency penalty：
$$C^{(m)} = C_{\text{goal}}^{(m)} + \beta\|\hat{\mathbf{z}}_{t+H}^{(m)} - \tilde{\mathbf{z}}_{t+H}^{(m)}\|_2^2$$

$\beta \geq 0$ 控制强度。$\beta=0$ 退化为 goal-only CEM；$\beta$ 大时 CEM 偏好那些在不同 prefix decomposition 下 terminal 预测稳定的 candidate。

实验里 $\beta=1$，加在直接预测 25-step latent 和"先走 10-step 中间 latent，再预测剩余"的 discrepancy 之间。

直觉：这就像 ensemble self-distillation——同一 model 用不同路径预测同一终点，path 之间一致说明 model 对该 candidate 的 rollout 有信心。

---

## 5. 实验数据深度解析

### 5.1 Planning success rate（Table 1）

| Method | Two-Room | Reacher | PushT | OGBench-Cube | Avg |
|---|---|---|---|---|---|
| PLDM | 97 | 78 | 78 | 65 | 79.5 |
| DINO-WM | 100 | 79 | 74 | 86 | 84.8 |
| LeWM | 87 | 86 | 96 | 74 | 85.8 |
| Fast-LeWM | 98 | 88 | 96 | 80 | 90.5 |
| Fast-LeWM + Self-Consistency | 98 | 90 | 98 | 82 | 92.0 |

Fast-LeWM 在所有 task 上不低于 LeWM，平均提升 4.7 个点。Self-consistency 再加 1.5 点，特别在 PushT (96→98) 和 Reacher (88→90) 这种 action effect 路径多样的任务上有效。

PLDM 参考: https://arxiv.org/abs/2211.10831 ；DINO-WM: https://arxiv.org/abs/2410.06281 ；OGBench: https://arxiv.org/abs/2410.20092 。

### 5.2 Planning efficiency（Table 2, Two-Room, NVIDIA 4090）

| Method | Model calls | Dynamics time | CEM time |
|---|---|---|---|
| LeWM | 5 | 31.4s | 54.4s |
| Fast-LeWM | 1 | 8.0s | 28.3s |

Model calls 从 5 降到 1——这是从 autoregressive 5 步变成 1 次 parallel forward 的直接结果。Dynamics time 降 3.9×，CEM total time 降 48%。注意 CEM total 降得比 dynamics 降得少，因为 goal encoding、score computation、data operation overhead 不在 dynamics 模块里。

这里有个非平凡点：Fast-LeWM 一次 forward 同时算所有 prefix token，相比 LeWM 5 次串行 forward，单次 forward 计算量更大（要处理 5 个 token 的 Transformer + 5 个 AdaLN MLP），但 GPU parallelism 让这远快于 sequential launch 5 次 kernel。

### 5.3 Open-loop latent prediction loss（Figure 3）

这是验证 error accumulation 的核心实验。给定真实初始 frame + future action，预测 open-loop trajectory 的 latent loss 随时间变化。

关键观察：
1. **初始误差低**：所有 4 个 task 上 Fast-LeWM 在 $t=25$ 处的 loss 明显低于 LeWM。
2. **增长斜率小**：作者用 least squares 拟合 loss curve 取 slope。Fast-LeWM slope 一致小于 LeWM。

为什么这成立？LeWM 的 $\hat{\mathbf{z}}_{t+25}$ 是 5 步 composition，每步误差进入下一步 input；Fast-LeWM 直接从 $\mathbf{z}_t$ 跳到 $\hat{\mathbf{z}}_{t+25}$，中间不喂回 approximated latent。当 horizon 从 25 推到 50（两次 max-horizon 预测 vs LeWM 10 步 autoregressive），Fast-LeWM 的优势进一步放大——因为只有 anchor-to-horizon 跳跃被 chain，而不是每个 primitive step 都 chain。

### 5.4 Physical state probing（Table 3, PushT）

冻 encoder，训练 linear probe 和 MLP probe 预测三个 physical variable：agent location、block location、block angle。

| Property | Model | Linear MSE↓ / r↑ | MLP MSE↓ / r↑ |
|---|---|---|---|
| Agent Loc | LeWM | 0.052 / 0.974 | 0.004 / 0.998 |
| Agent Loc | Ours | 0.048 / 0.976 | 0.001 / 1.000 |
| Block Loc | LeWM | 0.029 / 0.986 | 0.001 / 0.999 |
| Block Loc | Ours | 0.029 / 0.987 | 0.000 / 1.000 |
| Block Angle | LeWM | 0.187 / 0.902 | 0.021 / 0.990 |
| Block Angle | Ours | 0.314 / 0.828 | 0.009 / 0.995 |

Linear probe 上 Fast-LeWM 与 LeWM 相当（block angle linear 甚至差一点，0.314 vs 0.187）。但 MLP probe 上 Fast-LeWM 全面胜出——尤其 block angle 的 MLP MSE 从 0.021 降到 0.009。

直觉解读：linear probe 测的是"信息是否 linearly accessible"，MLP probe 测的是"信息是否 nonlinearly recoverable"。Fast-LeWM 把 physical 信息编码得更"非线性可读"——可能因为这些 variable 是预测多 horizon outcome 所必需的，被 dynamics objective 逼着 encode，但不是 linearly layout 在 latent 上。这是 prefix-level supervision 带来的副作用。

### 5.5 Ablation（Table 4）

| Variant | Two-Room | Reacher | PushT | Cube |
|---|---|---|---|---|
| Long-Action LeWM | 76 | 70 | 80 | 58 |
| Terminal-only Fast-LeWM | 96 | 80 | 90 | 72 |
| Fast-LeWM | 98 | 88 | 96 | 80 |
| w/o state token | 94 | 82 | 92 | 80 |

四个 ablation 揭示三个事实：

**Long-Action LeWM（action skip 从 5 改 25，仍用 one-step interface）性能崩盘。** 说明不能简单地把 LeWM 的 transition 覆盖更长 temporal span 来加速——直接预测 25 步后的 latent 没有中间 supervision，model 学不出 reliable 长程 transition。

**Terminal-only Fast-LeWM 已经远好于 Long-Action LeWM（86 vs 71 avg）。** 说明 prefix representation 本身（即使只监督 terminal）比 raw 长 action block 强——因为 causal Transformer 显式暴露 sequence 的 order structure 和 cumulative effect。

**Dense prefix supervision 在 Terminal-only 基础上再加 4 个点 avg。** 中间 prefix token 被显式约束对应 partial action outcome，这是把 sequence 内部 order structure 转化为 supervision 信号的关键。

**State token 去掉性能降 2-4 个点。** 同样 action prefix 在不同 scene configuration 下效果不同，state token 提供 context 来 disambiguate。

---

## 6. 直觉总结与延伸联想

### 6.1 为什么这个设计 work：从 chain composition 到 parallel fan-out

最深的直觉是关于 prediction graph 的拓扑。LeWM 的 multi-horizon prediction 是一条 chain：
$$\mathbf{z}_t \to \hat{\mathbf{z}}_{t+1} \to \hat{\mathbf{z}}_{t+2} \to \cdots \to \hat{\mathbf{z}}_{t+H}$$

每条 edge 是一次 $F_\phi$ 调用，每个 node 是 approximated。Error 沿 chain 累积，planning 时 sequential。

Fast-LeWM 是 star 拓扑：
$$\mathbf{z}_t \xrightarrow{\text{prefix}_1} \hat{\mathbf{z}}_{t+1}, \quad \mathbf{z}_t \xrightarrow{\text{prefix}_2} \hat{\mathbf{z}}_{t+2}, \quad \ldots, \quad \mathbf{z}_t \xrightarrow{\text{prefix}_H} \hat{\mathbf{z}}_{t+H}$$

所有 prediction 都 anchor 在 observed $\mathbf{z}_t$ 上，互不 sequential 依赖。Error 不在 prediction 内部累积，且所有 edge 可并行算。

这跟 value function learning 里从 bootstrap (TD) 到 Monte Carlo return 的转变有同构性——TD 是 local one-step bootstrap，MC return 是直接预测多步累积 reward。Fast-LeWM 类似"latent state 的 MC return 预测"。

### 6.2 与 diffusion / video prediction 的联系

Action-prefix prediction 让我联想到 video diffusion model 里的 frame conditioning：不是 autoregressive 一帧帧生成，而是用 causal attention 让中间帧 attend 到 anchor frame，一次 denoise 出整段 video。参考 Video Diffusion Models: https://arxiv.org/abs/2204.03458 。

Fast-LeWM 在 latent space 做了类似的事——anchor latent $\mathbf{z}_t$ 像 conditioning frame，prefix token 像 temporal position embedding，parallel predictor 像 multi-frame decoder。

### 6.3 与 Trajectory Transformer / Decision Transformer 的联系

Decision Transformer (https://arxiv.org/abs/2106.01345) 把 RL 转成 sequence modeling，用 causal mask 让每个 position 的 prediction 只看过去。Fast-LeWM 的 action-prefix encoder 本质就是一个 mini Decision Transformer——causal mask 让 position $k$ 的 representation 聚合前 $k$ 个 action。

差异：DT 在 raw state/action space 自回归生成；Fast-LeWM 在 latent space 用 causal encoder 生成 prefix representation，然后用 parallel predictor 一次性 fan-out 所有 future latent。这是 DT 思想在 latent world model planning 里的精炼版。

### 6.4 与 Hopfield network / Energy-based model 的联想

Self-consistency penalty 让我联想到 Hopfield network 的 energy minimization——多个 prediction path 应该收敛到同一 fixed point。$\hat{\mathbf{z}}_{t+H}$ 和 $\tilde{\mathbf{z}}_{t+H}$ 是同一 target 的两个 estimate，discrepancy 是 energy。CEM 选 low-energy candidate，本质是在做隐式 energy-based planning。

这跟 LeCun 一直推的 JEPA + energy-based model 路线（https://openreview.net/pdf?id=BZ5a1r-kVsf ）是同构的——prediction discrepancy 作为 model confidence 信号。

### 6.5 潜在局限

我读出的几个开放问题：

1. **Horizon 固定为 H=5。** Paper 没测更长的 H（比如 20、50）。Prefix encoder 是 causal Transformer，长 sequence 计算量 $O(H^2)$ attention。如果 H 很大，parallel predictor 的 hidden width 是否够容纳多 horizon 区分度？

2. **Anchor latent 强假设。** 所有 prediction 都锚在 $\mathbf{z}_t$ 上。当真实 rollout 中间发生 contact discontinuity（比如 PushT 中 block 突然被阻挡），从 anchor 直接跳 25 步可能丢失这种 mid-trajectory 突变信息。Self-consistency 部分缓解，但不彻底。

3. **State token 只用 MLP map 一次。** 同一 anchor latent 被 5 个 prefix 共享，可能对高度 state-dependent 的 dynamics（比如 robot arm 关节极限附近）建模不足。Future work 可以考虑 per-prefix state modulation。

4. **Beta 选 1。** Self-consistency 的 $\beta$ 是手调的，没有 sweep。理想情况下 $\beta$ 应该随 CEM iteration 自适应——早期 iteration 偏好 exploration（低 $\beta$），后期偏好 exploitation（高 $\beta$）。

### 6.6 对未来 world model 设计的启示

这篇 paper 最大的 conceptual contribution 是：**dynamics model 的 prediction interface 本身是一个 design dimension**。LeWM 把 interface 固定为 one-step transition，是历史惯性（来自 classical control / model-based RL 的 transition function 概念）。Fast-LeWM 表明，interface 可以重新设计成 multi-horizon、prefix-conditioned、parallel-friendly 的形式，且这个设计直接影响 planning 速度、prediction 误差累积、representation 质量。

这给未来 world model 研究打开一个方向：与其只在 representation objective 上创新（I-JEPA, V-JEPA, LeJEPA, DINO-WM 等等），不如同时思考 dynamics interface——比如 hierarchical prefix（不同 temporal scale 的 prefix）、adaptive horizon（根据 state 不确定性动态选 prefix 长度）、或者 cross-trajectory prefix（用其他 trajectory 的 subsequence 作为 prefix query）。

代码: https://github.com/Yuntian-Gao/Fast-LeWorldModel

---

希望这个拆解帮你在脑子里把 "Fast-LeWM = star-topology prefix-anchored parallel latent predictor with dense multi-horizon supervision" 这个核心 intuition 立起来。最值得记住的对照就是 LeWM 的 chain composition vs Fast-LeWM 的 star fan-out——这一个 topological 转变同时解决了速度、误差累积、representation quality 三个问题。
