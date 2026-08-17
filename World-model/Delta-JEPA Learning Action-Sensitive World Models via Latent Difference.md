---
source_pdf: Delta-JEPA Learning Action-Sensitive World Models via Latent Difference.pdf
paper_sha256: 94e394fd9cbffaffb370fdc5f4bb3e2a2833ca41966e02efe800d5afff30371e
processed_at: '2026-08-03T19:22:26-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Delta-JEPA

Andrej，我用最口语的方式重新讲一遍，把学术腔剥掉，看看这篇 paper 到底在干嘛。

---

## 一句话版本

你让一个模型在 latent space 里"想象未来"，它最爱的偷懒方式是把所有东西都压成一个常数点——预测永远准，但啥信息都没有。这篇 paper 的招数是：**让模型从"两张图的差"里反推 action**，这样它就没法偷懒了，因为差值如果是零，action 就推不出来。

---

## 用打比方的方式讲

想象你在玩一个游戏，模型是你的"预言机"。你给它当前画面 + 一个动作（比如"向左走"），它告诉你下一帧画面长啥样（在 latent space 里）。

**问题来了**：如果你只在意"预测准不准"，模型会发现一个 bug——把所有画面都映射成同一个点 $c$，然后永远预测 $c$。预测永远对，loss 永远低，但它啥也没学到。这叫 **representation collapse**。

之前的解法（LeWM）是给 latent space 加一个"散开一点"的约束（SigReg），逼 latent 不要全挤成一坨。这能防住完全 collapse，但有个**更隐蔽的 bug**：

模型可以让 latent "散开但不随 action 变化"。意思是——你给它"向左走"和"向右走"两个 action，它预测的下一帧 latent 几乎在同一个位置。预测还是准（因为下一帧本来就长得差不多），但**action 对 latent 没影响**。

planner 拿这个 rollout 来选 action，发现"向左"和"向右"的想象结果一样——那还选个屁？这就是 **action-insensitive collapse**。

---

## PLDM 的招数和它的漏洞

PLDM 是这么想的："那我加一个 inverse dynamics 模块吧——给它 $z_t$ 和 $z_{t+1}$，让它反推 action 是啥。如果反推得准，说明 latent 里确实编码了 action 信息。"

听起来对，但有个 shortcut：

forward predictor 是 $\hat{z}_{t+1} = P(z_t, a_t)$，它本来就拿到了 $a_t$。在 end-to-end 训练下，模型最 lazy 的做法是——**把 $a_t$ 原封不动地写进 $z_{t+1}$ 的某几个维度里**，像一个小抽屉专门存 action。

然后 inverse decoder 拿到 $[z_t, z_{t+1}]$，直接从 $z_{t+1}$ 的"小抽屉"里读出 $a_t$。完美作弊——inverse loss 很低，但 $z_{t+1}$ 根本没有真正建模"action 怎么改变 state"，它只是把 action 抄了一份。

类比一下：你让学生证明 $A \to B$，学生直接在答案里写"$B$（因为我提前看了答案）"。inverse dynamics 检查"答案对不对"——对啊，B 写对了。但学生根本没做证明过程。

---

## Delta-JEPA 的核心 move

把 inverse decoder 的输入从 $[z_t, z_{t+1}]$（两张图拼一起）改成 $\Delta z_t = z_{t+1} - z_t$（两张图的差）。

$$\hat{a}_t = D_\Theta(\Delta z_t)$$

**为什么这个减法能切断 shortcut？**

回到刚才的"小抽屉"作弊。如果 $z_{t+1} = g(z_t) + h(a_t)$，其中 $h$ 是那个存 action 的小抽屉。那么：

$$\Delta z_t = z_{t+1} - z_t = g(z_t) - z_t + h(a_t)$$

decoder 现在只能看到 $\Delta z_t$。它要恢复 $a_t$，就得**从"差值"里把 $h(a_t)$ 这个部分捞出来**。但差值里还混着 $g(z_t) - z_t$（state 本身的变化）。

关键点：如果模型还想作弊（让 $h$ 原封不动 encode action），那 $g(z_t) - z_t$ 必须是个**不依赖 action 的确定函数**——也就是说 $g$ 必须是一个真正的 dynamics function，不能跟着 action 乱变。

换句话说，**减法把 $z_t$ 的"绝对位置信息"消掉了**，inverse decoder 再也读不到 $z_{t+1}$ 里"抄的 action"，它必须真的去理解"变化量"和 action 的关系。

再用刚才的类比：现在不让学生写"答案是 B"，而是让他写"从 A 到 B 的差是啥"。差是 $\Delta$，你从这个 $\Delta$ 反推过程，学生没法直接抄答案了，因为答案被减掉了。

---

## 三个为什么这个减法 work

### 1. 防 collapse
如果 $z_t \approx z_{t+1}$，那 $\Delta z_t \approx 0$。decoder 拿到一坨零，只能输出常数 action，匹配不上真实的 variable action，loss 爆炸。所以 encoder 被逼着必须让相邻 observation 在 latent 里有区分度。

### 2. 切断 single-state shortcut
如上所述，decoder 看不到绝对位置，只能看变化。action 信息必须编码在"变化的方向和大小"里。

### 3. 让 planner 能用
planner 的工作是：固定 $z_t$，试不同 action，看 rollout 分歧。如果不同 action 产生的 $\Delta z_t$ 朝不同方向走，planner 就能看出"哦，action A 会让 latent 往左上走，action B 会让它往右下走，我选 A"。这就是 **action-sensitive dynamics**。

paper 里 Figure 2 画得很清楚：左边是不同 action 的 next state 挤成一团（action-insensitive），右边是不同 action 像放射状散开（action-sensitive，LDAD 想要的结果）。

---

## Multi-step 版本

光看一步的 displacement 约束太弱。paper 扩展到 N 步：

$$\{\hat{a}_t, \hat{a}_{t+1}, \dots, \hat{a}_{t+N-1}\} = D_\Theta(z_{t+N} - z_t)$$

意思是：给你 N 步之后的累积 displacement，你把 N 个 action 一个个解出来。

实现上用一个小 Transformer + 5 个 learnable query（类似 DETR 的 object query），displacement 通过 AdaLN 注入。直觉是：累积 displacement 是个"压缩包"，里面必须装得下 5 个 action 的信息，才能让 decoder 一个个拆开。

---

## 整个训练 loss

简单到离谱，就两项：

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{action}}$$

- $\mathcal{L}_{\text{pred}}$：forward predictor 预测下一帧 latent 的 MSE
- $\mathcal{L}_{\text{action}}$：从 $\Delta z_t$ 重建 action 的 MSE
- $\lambda$：平衡权重，主实验 10，ablation 发现 Push-T 上 50 最好

**没有 pixel reconstruction，没有 frozen encoder，没有 stop-gradient，没有 VICReg 那一堆 variance/invariance/covariance 三个 term 调来调去。** 就这两个 loss，end-to-end 训。这是我觉得这篇 paper 最优雅的地方。

---

## 实验讲了啥

四个连续控制任务：Two-Room（2D 导航）、Reacher（机械臂到目标点）、Push-T（推 T 形块）、OGB-Cube（3D 抓方块）。

主结果（planning success rate）：

| 方法 | Two-Room | Reacher | Push-T | OGB-Cube |
|---|---|---|---|---|
| PLDM | 93.73 | 64.33 | 76.13 | 57.27 |
| LeWM | 74.93 | 79.87 | 84.53 | 64.13 |
| Sub-JEPA | 90.60 | 81.00 | 63.73 | 62.67 |
| **Delta-JEPA** | **100.00** | **81.33** | **89.07** | **79.27** |

Delta-JEPA 四个全赢。OGB-Cube 提升最大（+15 over LeWM），因为这个 3D 任务 action 维度高，displacement 约束收益最大。

---

## 最关键的 ablation：concat vs displacement

这是整篇 paper 的核心证据。把 LDAD 的输入从 $\Delta z_t$ 换回 $[z_t, z_{t+1}]$，其他都不变：

| 输入 | Push-T |
|---|---|
| concat $[z_t, z_{t+1}]$ | 76.47 |
| displacement $\Delta z_t$ | **89.07** |

**Push-T 上差 12.6 个点**。Push-T 是接触密集的推块任务，action → state change 的 mapping 非线性很强。concat 形式下模型走 shortcut，没学到 contact dynamics；displacement 强制它从"接触引起的位移"反推 action，逼着 latent 学到接触的几何。

这 12.6 个点就是"减法切断 shortcut"的直接收益。

---

## 几个特别漂亮的诊断实验

### Action-conditioned response（Figure 6，我最喜欢的）
固定 512 个 starting history 的 $z_t$，只 sweep action，看 predictor 输出的 $\hat{z}_{t+1}(a)$ 相对于 zero-action 的位移 $\hat{z}_{t+1}(a) - \hat{z}_{t+1}(0)$。

- **Delta-JEPA**：不同 action 产生的 mean marker 清晰分离，action magnitude 越大 shift 越大。**predictor 真的"听"action 了**。
- **LeWM**：所有 action 的 marker 挤在 origin 附近，overlap 严重。**predictor 对 action 基本没反应**。

这个实验直接把"action sensitivity"这个抽象概念变成了可观测的几何量。很 Karpathy 风格——找一个最直接的 probe 测你想测的性质。

### Two-Room trajectory PCA（Figure 5）
两条起点相近但终点不同的 trajectory：
- **Delta-JEPA**：起点近，随时间 fan out，颜色从浅到深是时间推进。清晰的 temporal compositionality。
- **LeWM**：散乱，没结构。

### State-delta probing（Table 5, 9-11）
非常聪明的诊断：从 $\Delta z_t$ 用 linear probe 预测 ground-truth 物理量变化 $\Delta x_t$。

Two-Room 上 Delta-JEPA 的 $\Delta$ Agent Position linear probe MSE=0.016, r=0.992，PLDM 是 0.355，LeWM 是 0.444。

这说明 $\Delta z_t$ 不只是编码了 action，它还**真的对应了物理空间的真实位移**。因为 LDAD 的 supervision 是"从 $\Delta z_t$ 恢复 action"，而 action 又决定 $\Delta x_t$，所以 $\Delta z_t$ 必须包含能推导出 action 的物理变化信息——顺带就把物理 displacement 也 encode 了。

---

## 用你的 intuition 联想一下

Andrej，你一直在 nanoGPT、micrograd 里强调"别让模型作弊"。这个 paper 就是这个 spirit 的一个具体实例。

**减法操作本质上是一个 inductive bias**：它告诉模型"我只关心变化量，不关心绝对值"。这和 CNN 的 translation equivariance 是同一类思想——CNN 告诉模型"pattern 的绝对位置不重要，相对位置才重要"；Delta-JEPA 告诉模型"latent 的绝对位置不重要，变化方向才重要"。

再往深想，$z_{t+1} - z_t$ 在微分意义上就是 $dz$，action 对应 $da$，这其实是让 latent space 学到一个**向量场** $v(z, a)$，action 决定向量场的方向。这和 Lie algebra、symplectic dynamics、Hamilton-Jacobi 那套连续时间动力学的几何语言是同一个东西。N=5 的 multi-step 版本就类似于有限差分近似一个 ODE 积分。

如果哪天有人把这个推广到 Lie group（处理 3D rotation 不用 quaternion 直接减，而是用 $\log(R_{t+1} R_t^{-1})$），那就是 Delta-JEPA 的 Lie 版本。OGB-Cube 的 Block Quaternion probe 结果不好（Table 8: r=0.273），就是因为 quaternion 空间上直接做欧氏减法不对——这里有个明显的 extension opportunity。

---

## 总结成人话

1. **问题**：JEPA world model 容易 collapse，就算不 collapse 也容易"对 action 不敏感"，planner 没法用。
2. **现有解法的漏洞**：inverse dynamics 用 $[z_t, z_{t+1}]$ 会有 shortcut，模型把 action 直接抄进 $z_{t+1}$ 而不学 dynamics。
3. **这篇的招数**：把 inverse decoder 输入改成 $\Delta z_t = z_{t+1} - z_t$，强制 decoder 只看变化量，切断抄 action 的通道。
4. **效果**：loss 极简（就两项），end-to-end 训，四个任务全赢，action sensitivity 诊断清晰可见。
5. **intuition**：减法 = translation equivariance 的 inductive bias，强制 action 编码在"变化方向"里而不是"绝对位置"里。

这 paper 的 elegance 在于——**用一个减法解决了一个看似需要复杂正则化的问题**。这种"用最简单的代数操作引入最强的 inductive bias"的风格，我觉得你 Karpathy 会喜欢。

---

参考链接（便于深挖）：
- I-JEPA (JEPA 起点): https://arxiv.org/abs/2301.08243
- V-JEPA 2 (video extension): https://arxiv.org/abs/2506.09985
- VICReg (对比的正则化思路): https://arxiv.org/abs/2105.04906
- LeWM (主 baseline): https://arxiv.org/abs/2603.19312
- DINO-WM (frozen encoder 路线): https://arxiv.org/abs/2411.04967
- DreamerV3 (reconstruction 路线): https://arxiv.org/abs/2301.04104
- Diffusion Policy (Push-T 的 SOTA policy): https://arxiv.org/abs/2303.04137
- DINOv2 (object emergence): https://arxiv.org/abs/2304.07193
- DETR (object query 灵感): https://arxiv.org/abs/2005.12872
- DiT (AdaLN): https://arxiv.org/abs/2212.09748
- World Models (Ha & Schmidhuber 经典): https://arxiv.org/abs/1803.10122
- LeCun AMI roadmap: https://openreview.net/pdf?id=BZkJaVNkDg
- SigReg / LeJEPA (防 collapse 的另一条路): https://arxiv.org/abs/2511.08544
- OGBench (Cube 任务来源): https://arxiv.org/abs/2410.20092
- DMC Suite (Reacher 来源): https://arxiv.org/abs/1801.00690

---

# Delta-JEPA 深度技术讲解

Andrej，这篇paper的核心insight非常精妙，本质上是用一个**看似微小的代数操作（latent subtraction）**切断了 end-to-end JEPA world model 中 inverse dynamics 的 shortcut 通道。让我从几何直觉、shortcut 机制、loss 设计、架构细节、实验证据五个层次展开。

---

## 1. 问题本质：为什么 latent prediction alone 会 collapse

JEPA 的 elegance 在于它跳过 pixel reconstruction，直接在 representation space 预测未来。但这个 elegance 同时埋下了一个 degenerate solution 的种子。考虑 forward prediction loss：

$$\mathcal{L}_{\text{pred}} = \|\hat{z}_{t+1} - z_{t+1}\|_2^2$$

变量含义：
- $z_{t+1} = f_\theta(o_{t+1}) \in \mathbb{R}^d$：encoder $f_\theta$ 对真实下一帧 observation 的 embedding
- $\hat{z}_{t+1} = P_\phi(z_t, a_t)$：predictor $P_\phi$ 接收当前 latent $z_t$ 和 action $a_t$ 输出的预测
- $\|\cdot\|_2^2$：squared L2 norm

**Collapse 的几何机制**：如果 encoder $f_\theta$ 把所有 $o_t$ 都 map 到同一个常数向量 $c$（即 $f_\theta(o_t) = c, \forall t$），那么 predictor $P_\phi$ 只需学会 $P_\phi(c, a_t) = c$，loss 就精确等于 0。这就是经典 representation collapse。LeWM 用 SigReg (Balestriero & LeCun 2025, https://arxiv.org/abs/2511.08544) 在 latent 上施加一个 Gaussian marginal 约束来防止这个 trivial solution。

但即使防止了完全 collapse，还存在一个**更 subtle 的 action-insensitive collapse**：encoder 可以把"不同 action 下的相邻 observations" map 到 latent space 中彼此邻近的点。形式上，对于同一个 $z_t$，两个不同 action $a_t^{(1)}, a_t^{(2)}$ 产生的 $z_{t+1}^{(1)}, z_{t+1}^{(2)}$ 距离 $\|z_{t+1}^{(1)} - z_{t+1}^{(2)}\|$ 很小。此时 prediction loss 仍然小，但 latent rollout 对 planner 来说毫无信息量——因为 planner 通过比较不同 action 候选对应的 rollout 结果来选 action，rollout 都"挤在一起"就无法区分。

这是 JEPA-based world model 在 planning 任务上的**致命几何缺陷**。可以参考 V-JEPA 2 (https://arxiv.org/abs/2506.09985) 也面临类似问题。

---

## 2. 现有 inverse dynamics 的 shortcut 问题（核心 insight）

PLDM (Sobal et al. 2026, https://arxiv.org/abs/2603.xxxxx) 用 inverse dynamics 来增强 action sensitivity：

$$\hat{a}_t = \bar{D}_\Theta([z_t, z_{t+1}])$$

变量含义：
- $[z_t, z_{t+1}] \in \mathbb{R}^{2d}$：两个相邻 latent state 的 concatenation
- $\bar{D}_\Theta$：inverse dynamics decoder
- $\hat{a}_t \in \mathbb{R}^{d_a}$：predicted action

**为什么这个形式有 shortcut**：注意 forward predictor 是 $\hat{z}_{t+1} = P_\phi(z_t, a_t)$，training 目标让 $\hat{z}_{t+1} \to z_{t+1}$。在 end-to-end 优化下，模型有完全的自由度选择 $z_{t+1}$ 的 internal representation geometry。最 lazy 的 solution 是让 $z_{t+1}$ 包含一个"action 通道"——一个只依赖于 $a_t$ 的 subspace，predictor 把 $a_t$ 信息几乎"原封不动"地写进 $z_{t+1}$ 的某些维度。

用更具体的代数 intuition：假设 representation 学到了近似形式 $z_{t+1} \approx g(z_t) + h(a_t)$（加性 decomposition，虽然真实学到的可能更复杂，但加性是一个 minimum sufficient 的 shortcut 模式）。那么 inverse decoder $\bar{D}_\Theta$ 可以直接读 $z_{t+1}$ 中的 $h(a_t)$ 部分，**完全 bypass 任何 transition 建模**。结果是：action reconstruction loss 很低，但 $\Delta z_t = g(z_t) - z_t + h(a_t)$ 中的 $g(z_t) - z_t$ 部分可能 collapse 或者完全与 action 无关。

具体后果：planner 拿 latent rollout $z_t \to z_{t+1} \to z_{t+2} \to \cdots$ 来比较候选 action，但每个 transition 几乎独立，没有真正的 dynamics chain 信息。PLDM 的 inverse dynamics 失去了"约束 transition geometry"的意图。

---

## 3. LDAD 的核心修复：displacement 切断 shortcut

Delta-JEPA 的关键修改是把 inverse decoder 的输入从 $[z_t, z_{t+1}]$ 换成 $\Delta z_t$：

$$\Delta z_t = z_{t+1} - z_t$$
$$\hat{a}_t = D_\Theta(\Delta z_t)$$
$$\mathcal{L}_{\text{action}} = \|\hat{a}_t - a_t\|_2^2$$

变量含义：
- $\Delta z_t \in \mathbb{R}^d$：latent displacement 向量
- $D_\Theta$：Latent Difference Action Decoder
- $\hat{a}_t$：从 displacement 重建出的 action
- $a_t$：ground-truth executed action
- $\|\cdot\|_2^2$：squared L2

**为什么这个简单减法有效——三个机制**：

### 3.1 Anti-Collapse Effect
如果 $z_t \approx z_{t+1}$（adjacent collapse），那么 $\Delta z_t \approx \mathbf{0}$。decoder $D_\Theta(\mathbf{0})$ 只能输出一个常数 action，无法匹配 variable $a_t$，loss 爆炸。这强制 encoder 必须把相邻 observation 至少在某个 direction 上分开。

### 3.2 切断 single-state cue shortcut
回到前面的加性 decomposition：$z_{t+1} \approx g(z_t) + h(a_t)$。则 $\Delta z_t \approx g(z_t) - z_t + h(a_t)$。decoder 现在只看到 $\Delta z_t$，它必须从 $g(z_t) - z_t + h(a_t)$ 中恢复 $a_t$。如果 $h(a_t)$ 还想原封不动地 encode $a_t$，$g(z_t) - z_t$ 必须是确定的（不依赖于 $a_t$），这就强制 $g$ 是关于 $z_t$ 的一个真实 dynamics function，不能"作弊"。

更深刻地讲：**减法消去了 encoder 可能注入 $z_{t+1}$ 的绝对位置 cue**。在 concat 形式下，inverse decoder 有两份信息可用：$z_t$（已知当前 state）和 $z_{t+1}$（包含 action cue）。理论上最优 inverse decoder 可以用 $z_t$ 解出"当前 state 的什么特征"，用 $z_{t+1}$ 解出"下一 state 的特征"，再做差——但端到端训练不会逼它这么做，它会走最 lazy 的 path。displacement 形式**预先在 decoder 外部做减法**，让 decoder 看不到绝对位置，只能看到相对变化。

### 3.3 Action-sensitive dynamics for planning
对同一个 $z_t$，假设有 $K$ 个候选 action $\{a_t^{(k)}\}_{k=1}^K$，产生 $K$ 个 $\Delta z_t^{(k)}$。要使 $\hat{a}_t^{(k)} = D_\Theta(\Delta z_t^{(k)})$ 各不相同且匹配对应 $a_t^{(k)}$，必须 $\Delta z_t^{(k)}$ 在 latent space 中至少线性可分（per action）。等价地，$z_{t+1}^{(k)} = z_t + \Delta z_t^{(k)}$ 必须彼此分离。这就给 planner 提供了 action-distinguishable rollout——这是 planning 的最低必要条件。

可以用 Figure 2 的 top-left vs top-right 直观对照：top-left 是 action-insensitive geometry（不同 action 的 endpoint 都在 $z_t$ 附近一个 blob 里）；top-right 是 LDAD-induced geometry（不同 action 沿不同方向 push 出去，形成 "star" pattern）。

---

## 4. Multi-step LDAD：让 displacement 编码 action 序列

为了让 displacement 约束覆盖 longer horizon，paper 把 LDAD 扩展到多步：

$$\{\hat{a}_\tau\}_{\tau=t}^{t+N-1} = D_\Theta(z_{t+N} - z_t)$$

变量含义：
- $z_{t+N}$：N 步之后的 latent state
- $z_{t+N} - z_t \in \mathbb{R}^d$：N 步累积 displacement
- $\{\hat{a}_\tau\}_{\tau=t}^{t+N-1}$：N 个连续 action 的预测序列

**架构实现**：$D_\Theta$ 是 3-layer non-causal Transformer，配 N=5 个 learnable action queries（类似 DETR 的 object queries, https://arxiv.org/abs/2005.12872）。displacement $z_{t+N} - z_t$ 通过 **Adaptive Layer Normalization (AdaLN)** 注入每个 query。AdaLN 是 diffusion 模型中广泛使用的技术（DiT, https://arxiv.org/abs/2212.09748），形式为：

$$\text{AdaLN}(h, c) = \gamma(c) \cdot \text{LayerNorm}(h) + \beta(c)$$

其中 $c = z_{t+N} - z_t$ 是 conditioning vector，$\gamma, \beta$ 是从 $c$ 通过一个小 MLP 学到的 scale 和 shift。每个 action query 经过 stack of AdaLN-conditioned Transformer layers 后，输出 head 解码出对应时间步的 $\hat{a}_\tau$。

**直觉**：累积 displacement $z_{t+N} - z_t$ 类似一个"压缩包"，里面必须包含 N 个 action 的信息才能让 decoder 把它们一个个"解包"出来。这等价于强制 latent trajectory 的 endpoint 距离 encode 整个 action 序列。N=5 是一个 sweet spot，horizon 太短约束弱，太长则单个 displacement 难以承载多 action 信息。

---

## 5. 完整 objective 与训练设置

总 loss：

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{action}}$$

变量含义：
- $\lambda > 0$：balancing hyperparameter，paper 主实验用 $\lambda = 10.0$，但 ablation 显示 Push-T 上 $\lambda = 50$ 最佳

**架构规格**：
- Encoder $f_\theta$：ViT-Tiny (randomly initialized, no pretraining)，patch embedding + 12 transformer layers，hidden dim 192
- Predictor $P_\phi$：6-layer causal Transformer，16 attention heads，head dim 64，MLP hidden 2048，action 通过 AdaLN 注入
- Decoder $D_\Theta$：3-layer non-causal Transformer，N=5 learnable action queries，8 heads，head dim 64，FFN hidden 512

**训练配置**：
- 50 epochs from scratch
- Learning rate $5 \times 10^{-5}$
- 3 random seeds
- Dataset 是 reward-free offline trajectories，behavior policy 未知

---

## 6. 实验结果深度分析

### 6.1 主结果（Table 1）

| Method | Two-Room | Reacher | Push-T | OGB-Cube |
|---|---|---|---|---|
| PLDM | 93.73±1.03 | 64.33±2.14 | 76.13±1.70 | 57.27±1.53 |
| LeWM | 74.93±0.42 | 79.87±0.90 | 84.53±1.50 | 64.13±1.89 |
| Sub-JEPA | 90.60±0.53 | 81.00±2.40 | 63.73±0.12 | 62.67±1.45 |
| **Delta-JEPA** | **100.00±0.00** | **81.33±0.50** | **89.07±1.90** | **79.27±1.81** |

几个观察：
- **Two-Room 上 100% 成功**，这是一个 navigation task，需要长 horizon 的 action-conditioned rollout。100% 说明 latent dynamics 在长 horizon 也能保持 action-distinguishability——这是 multi-step LDAD 起作用的关键证据
- **OGB-Cube 提升最大**（+15.14 over LeWM），这是 3D 操作任务，自由度大、action 维度高，displacement-based 约束带来的几何收益最显著
- **Reacher 提升最小**，因为 Sub-JEPA 已经 81.00%，天花板接近

### 6.2 Ablation：concat vs displacement（Table 2，最关键的实验）

| Action-Decoder Input | Two-Room | Reacher | Push-T | OGB-Cube |
|---|---|---|---|---|
| $[z_t, z_{t+1}]$ (concat) | 95.93±0.61 | 80.27±0.81 | 76.47±2.08 | 78.60±3.29 |
| $\Delta z_t$ (LDAD) | **100.00±0.00** | **81.33±0.50** | **89.07±1.90** | **79.27±1.81** |
| Gain | +4.07 | +1.07 | +12.60 | +0.67 |

Push-T 上的 +12.60 是最戏剧化的证据。Push-T 是 contact-rich manipulation（https://arxiv.org/abs/2303.04137），物理接触导致 action $\to$ state change 的 mapping 高度非线性，concat inverse dynamics 容易学到"action 通道"shortcut（直接从 $z_{t+1}$ 读 action 而不通过 contact dynamics），导致 rollout 对 contact 敏感度不足。displacement 强制 decoder 必须从"接触引起的位移"中反推 action，让 latent space 学到 contact 的几何结构。

### 6.3 λ 敏感性（Figure 3）

$\lambda = 0$：完全 collapse，success rate 近 0。证明 LDAD 是 anti-collapse 的关键约束。
$\lambda = 0.1$：信号太弱，仍然差。
$\lambda \in [1, 50]$：稳定 plateau，最佳 $\lambda = 50$。
$\lambda \geq 100$：action reconstruction 过度 dominate，forward prediction 被压制，性能下降。

这个 curve 形态很像 VICReg 论文里 variance/invariance/covariance 三个 term 权重的 trade-off curve (https://arxiv.org/abs/2105.04906)。

### 6.4 LDAD decoding target ablation（Table 3）

| Decoded Target | Success Rate |
|---|---|
| Raw action $a_t$ | 81.33±0.50 |
| $\Delta$ finger position | 64.93±1.10 |
| $\Delta$ joint position | 80.47±2.10 |
| $\Delta$ finger + $\Delta$ joint | 76.40±1.40 |

**直觉**：raw action 最好，因为 action 直接对应 agent 的 control 输入，与 transition 的因果链最紧密。$\Delta$ joint position 也不错，因为 joint 是 agent 的 controllable state，与 action 一一对应。$\Delta$ finger position 差，因为 finger position 是 action 经过 forward dynamics 之后的二级 consequence，间接。Concat 反而下降——这暗示"adding 多余信号会稀释 displacement supervision"。

---

## 7. 诊断实验：直接 probe action-sensitivity

### 7.1 PCA on latent over training（Figure 4）
Push-T 上的 latent representation 从 epoch 1（compact blob）逐渐 expand 成 discernible structure。这是 anti-collapse 的可视化证据。

### 7.2 Two-Room trajectory PCA（Figure 5）
两条 initial states 相近但 endpoint 不同的 trajectory：
- **Delta-JEPA**：起点近，随时间逐渐 fan out，颜色由浅到深呈现清晰的时间 progression。这是 action-conditioned rollout 分化的几何表现。
- **LeWM**：散乱、缺少 temporal compositionality，trajectory 之间没有清晰的 separation pattern。

### 7.3 Action-conditioned predictor response（Figure 6，最直接的诊断）
实验设计：sample 512 个 starting histories，对每个 history 固定 history representation $z_t$，只 sweep 候选 action。对每个 action $a$，计算 $\hat{z}_{t+1}(a) - \hat{z}_{t+1}(0)$，即相对于 zero-action 的 displacement。可视化到 PCA 空间。

- **Delta-JEPA**：不同 action 产生清晰分离的 mean marker，且 action magnitude 越大 shift 越大。这是 **action-conditioned geometry 的直接证明**。
- **LeWM**：所有 action 的 mean marker 挤在 origin 附近，overlap 严重。说明 forward predictor 对 action 输入几乎没有 response，处于 action-insensitive 状态。

这是我最喜欢的实验——它把"action sensitivity"这个抽象几何性质变成了可观测的量。

### 7.4 Physical probing（Table 4-11）
- **Two-Room Agent Pos. linear probe**：Delta-JEPA MSE=0.004, r=0.998，远好于 PLDM (0.078) 和 LeWM (0.085)。说明 latent 直接 linear encode 了 agent 位置。
- **OGB-Cube End-Effector Position**：Delta-JEPA MSE=0.007, r=0.997，LeWM 只有 MSE=0.515。这里 displacement supervision 帮助 encode 末端执行器位置——因为 end-effector position 与 action 紧密相关。

### 7.5 State-delta probing（Table 5, 9-11）
这是非常聪明的诊断：从 $\Delta z_t = z_{t+1} - z_t$ 通过 probe 预测 $\Delta x_t = x_{t+1} - x_t$（ground-truth 物理量变化）。

- **Two-Room $\Delta$ Agent Pos. linear**：Delta-JEPA MSE=0.016, r=0.992，PLDM 是 0.355, LeWM 是 0.444。
- **OGB-Cube $\Delta$ End-Effector Position**：Delta-JEPA MSE=0.010, r=0.995，PLDM 0.608，LeWM 0.678。

这些结果直接验证了 LDAD 让 $\Delta z_t$ 学到了真实的物理 displacement，与 $\Delta x_t$ 高度 aligned。这不是意外——LDAD 的 supervision 信号就是"从 $\Delta z_t$ 恢复 action"，而 action 又直接决定 $\Delta x_t$，所以 $\Delta z_t$ 必须包含足以推导 action 的物理变化信息。

### 7.6 Attention rollout（Figure 7, 8）
Push-T 上 encoder 注意到 T 形 block 和 agent；OGB-Cube 上 layer 5 注意 cube，layer 7 注意 gripper——layer-wise specialization。说明 encoder 在没有 pixel reconstruction supervision 的情况下自然学到了 object-centric attention。这是 JEPA 类方法 object emergence 现象 (https://arxiv.org/abs/2304.07193 DINOv2 也有类似观察) 在 world model setting 下的体现。

---

## 8. 与相关工作的 positioning

- **Dreamer / DreamerV3 (https://arxiv.org/abs/2301.04104)**：用 RSSM + pixel reconstruction，避免 collapse 但 capacity 浪费在 visual detail。
- **DINO-WM (https://arxiv.org/abs/2411.04967)**：用 frozen DINOv2 features，稳定但限制 task adaptation。Delta-JEPA 全 end-to-end，no frozen encoder。
- **LeWM (https://arxiv.org/abs/2603.19312)**：用 SigReg 防 collapse 但不约束 action sensitivity。Delta-JEPA 直接 supervise action via displacement。
- **PLDM (https://arxiv.org/abs/2603.xxxxx)**：VICReg + inverse dynamics with concat——Delta-JEPA 的 displacement 改进直接 target PLDM 的 shortcut 缺陷。
- **V-JEPA 2 (https://arxiv.org/abs/2506.09985)**：self-supervised video model，与 world model planning 接近，但 V-JEPA 2 主要 focus on representation + evaluation，Delta-JEPA focus on action-conditioned latent dynamics for planning。
- **I-JEPA (https://arxiv.org/abs/2301.08243)**：mask image modeling in latent space，是 JEPA family 的 image 起源。

---

## 9. 与你 (Karpathy) 一直强调的 intuition 的连接

这篇 paper 实际上做了一个非常类似"narrow bottleneck for cheating"的事情——类似于 nanoGPT、micrograd 中你常说的"不要让模型走捷径"。displacement operation 在数学上是一个**translation-equivariant 的约束**：encoder 学到的 latent space 在 $z \to z + v$ 这个 translation 下，action 信号必须由 $v$ 本身编码。

可以联想：
- **Causal interventions**：displacement form 强制 $a_t \perp z_t$（在因果意义上），因为 decoder 只看 $\Delta z_t$，action 信号不能"渗漏"进 $z_t$。
- **Equivariant representation theory**：$z_{t+1} - z_t$ 是 translation-equivariant 的最简单形式，相当于在 latent space 上施加 $\mathbb{R}^d$ 加法群的不变性。这与卷积网络的 translation equivariance 在 spirit 上对应。
- **Lie algebra intuition**：$z_{t+1} - z_t$ 类似一个无穷小 generator 在 latent space 上的作用，action 决定 generator direction。这指向 Lie group-based world model 的方向 (TNC, Lie symplectic, etc.)。

还可以联想 **Schmidhuber 的 PowerPlay**、**Hafner 的 RSSM 的 posterior collapse 问题**、**VQ-VAE 的 codebook collapse**、**BYOL 的 implicit collapse 防护 (EMA + stop-grad)**——所有这些都是 "如何让 representation 不 collapse" 的不同面向。Delta-JEPA 的特别之处是它**针对 action-conditioned world model 这个具体场景**，把 collapse 防护与 action sensitivity 统一在一个 displacement objective 里。

---

## 10. Limitations & extensions 我能想到的

1. **加性 decomposition 假设**：paper 论证 shortcut 时用 $z_{t+1} \approx g(z_t) + h(a_t)$ 作为 intuition，但真实的 representation 可能更复杂（multiplicative、attention-based gating）。理论上 shortcut 仍可能通过非线性方式存在，只是更难。可以做更严格的 information-theoretic 分析（用 mutual information bound）。

2. **Continuous action only**：实验都在 continuous control。Discrete action 下 displacement 的语义是否还成立？discrete action 可能让 latent 在不同 mode 间 jump，$\Delta z_t$ 是非光滑的，可能需要不同的 metric。

3. **Horizon N=5 是固定的**：能否让 N 自适应或者用 hierarchical structure（类似 Dreamer 的 H-step rollout vs full trajectory）。

4. **与 diffusion-based policy 的结合**：Diffusion Policy (https://arxiv.org/abs/2303.04137) 已经在 Push-T 上很强。能否用 Delta-JEPA 作为 latent rollout generator，让 diffusion policy 在 latent space 做 action generation，避免 high-dim pixel diffusion？

5. **Stochastic environment**：当前 framework 是 deterministic dynamics + MSE loss。Stochastic environment 下 $z_{t+1}$ 是 distribution，MSE 会 mode-collapse 到 mean。需要换成 distributional loss（KL、Wasserstein），同时 displacement 形式如何延伸到 distributional setting 是个 open question。

6. **Large-scale visual pretraining**：encoder 是 ViT-Tiny from scratch，限制了 visual generality。能否让 encoder 是预训练的 V-JEPA 2 backbone (https://arxiv.org/abs/2506.09985)，只 fine-tune predictor + LDAD？这相当于在 frozen visual foundation 上做 world model，类似 DINO-WM 但 action-sensitive。

7. **3D rotation 的 displacement 表达**：OGB-Cube 的 Block Quaternion probe 结果并不好（Table 8: Delta-JEPA Block Quaternion MSE=1.053, r=0.273；Block Yaw r=0.241）。Quaternion 不是欧氏空间，简单减法不对应 rotation 的 displacement。可能需要 Lie algebra displacement ($\log(R_{t+1} R_t^{-1})$)。

---

## 11. 一句话总结

Delta-JEPA 通过把 inverse dynamics 从"看 absolute latent pair" 改成"看 latent difference"，**在 latent space 上强制一个 translation-equivariant 的 action encoding geometry**，从而同时实现 anti-collapse 和 action-sensitivity 两个目标，loss 极简（只有 forward prediction + action reconstruction 两个 term），不需要 frozen encoder、stop-gradient、distribution matching，是 JEPA-based world model 在 planning 任务上的一个 elegant 进展。

参考链接：
- 原始 JEPA vision paper: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2404.08471
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- VICReg: https://arxiv.org/abs/2105.04906
- DINOv2: https://arxiv.org/abs/2304.07193
- DINO-WM: https://arxiv.org/abs/2411.04967
- DreamerV3: https://arxiv.org/abs/2301.04104
- Dreamer: https://arxiv.org/abs/1912.01603
- PlaNet: https://arxiv.org/abs/1811.04551
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- OGBench: https://arxiv.org/abs/2410.20092
- DMC Suite: https://arxiv.org/abs/1801.00690
- DETR (object queries): https://arxiv.org/abs/2005.12872
- DiT (AdaLN): https://arxiv.org/abs/2212.09748
- LeCun AMI roadmap: https://openreview.net/pdf?id=BZkJaVNkDg
- SigReg / LeJEPA: https://arxiv.org/abs/2511.08544
