---
source_pdf: Beyond Monotonic Progress.pdf
paper_sha256: 28ef4c10b7c06157cfaa5a0274edf3d19ecb7ee297bce4efc8a1169e037b1747
processed_at: '2026-07-18T16:52:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ReTVL: Retry-Supervised Value Learning 深度讲解

## 1. Paper 的核心 motivation

Andrej, 这篇 paper 触及了 robot imitation learning 中一个被长期回避的"dirty secret": 真实 human demonstration data 几乎从来都不干净。Teleoperator 会 overshoot, 会对不准 grasp, 会 unstable contact, 然后会自发地纠正。这些 retry 事件在主流的数据 pipeline 里要么被当成 noise 直接丢弃, 要么被一刀切当作"suboptimal"用 progress label 硬拟合。

**问题的本质** 在于: 几乎所有现有的 value/reward model 都假设 trajectory 的 progress 是一个关于 time 的 **monotonic non-decreasing scalar** —— 第 t 帧的 value 就该等于 t/T。这条假设在 expert demonstration 上是 fine 的, 但在含 retry 的 mixed-quality data 上是 fundamentally wrong 的: 当 robot 手爪在 grasp 前一秒明显偏了, human teleoperator 在那一帧的"task progress"实际上倒退了, 因为 system state 偏离了通向 success 的 manifold。Monotonic label 在这里会强迫 value model 把所有 downward fluctuation 抹平, 让它变成一个对"哪里出错"完全 blind 的光滑曲线。

Paper 的关键 insight 是一句很朴素的话: **retry 事件本身是免费的 supervision signal**。Reasoning chain 是这样的: 

1. Human 决定 retry 的那一刻, 一定是因为他自己感知到了"当前 state 偏了, 必须修"; 
2. 所以 retry keypoint 是 human 内置 value function 的 argmin (一个 local minimum);
3. retry 之前 value 应该 drop, retry 之后 (correction 开始执行) value 应该 rebound; 
4. 这构成了一组 **pairwise preference**: (pre-retry state, retry state) 偏好 pre; (retry state, post-retry state) 偏好 post。

这条 insight 把一个看起来需要 dense per-frame label 的难题, 降维成了只需要标 sparse keypoint (每条 trajectory 平均 26 个) 的问题。Table 5 显示平均每条 trajectory 标注 < 1 min, inter-annotator error 仅 0.19s, 比 5Hz 一帧还短。

## 2. 与现有范式的位置关系

为了 build intuition, 我把这篇 paper 放在更大的 landscape 里看:

- **Progress-based value learning** (RECAP [3], RoboDopamine [40], SARM [13], ARM [33]): 假设 monotonic, 学一个 t/T 的 regressor。ReTVL 在这一类基础上做修正。
- **Preference-based reward learning** (Robometer [29], RLHF [16, 17]): 用 trajectory-level pairwise comparison 学 reward, 不显式建模 progress。ReTVL 借用了 Bradley-Terry preference loss, 但 anchor 是 sparse retry keypoint 而非整条 trajectory 比较。
- **VLM-as-reward** (TOPReward [14], RoboReward [26]): zero-shot 用 VLM 判断 task completion probability。Cheap 但 coarse, 对 local mistake 几乎 zero sensitivity (paper 里 TOPReward 的 Pre > Retry 只有 0.329)。
- **Suboptimal IL** (DemoDICE [23], Discriminator-weighted IL [45], Confidence-aware IL [49]): 在 policy 层面做 reweighting, 但 typically 需要专家参考或 discriminator。ReTVL 把 reweighting 信号从 value model 拉出来, 更直接。

ReTVL 的设计哲学本质上是 **global absolute + local relative** 的双轨制: 用 absolute progress 把全局刻度锚住 (避免 value 乱漂), 用 retry-induced pairwise preference 把局部结构雕刻进去 (避免被单调假设抹平)。这是一种很典型的"分层监督"思想 —— coarse signal 用来定 anchor, fine signal 用来定 shape。

## 3. 方法详解: 公式逐条拆解

### 3.1 Distributional Value Function (Eq. 1)

$$
V_{\theta}(h_t, \ell) = \sum_{k=0}^{K-1} b_k \, p_{\theta}(k \mid h_t, \ell)
$$

**变量解释**:
- $h_t = \mathbf{o}_{t-H+1:t}$: 历史 observation window, 大小 H=8 frames (5Hz 下采样后), 是模型的视觉输入
- $\ell$: language instruction (task 描述)
- $K$: discretized value bins 数量, paper 里 K=64
- $b_k$: 第 k 个 bin 的 center value, 在 [0,1] 区间均匀分布
- $p_\theta(k \mid h_t, \ell)$: 模型预测的属于第 k 个 bin 的 probability
- $\theta$: 模型参数 (Robometer-4B backbone + 2-layer MLP value head, LoRA rank 32)

**为什么用 distributional**: 单 scalar regression 在 value learning 里有个 well-known 问题 —— value 的绝对尺度很难标, 而 categorical distribution 通过 expectation 出来的 scalar 对 head 的 over-confident prediction 鲁棒, 类似 C51 在 RL 里的作用。Bin 数 K=64 给了 0.016 的分辨率, 足够区分 retry 附近的细微 drop。

### 3.2 Absolute Progress Supervision (Eq. 2)

$$
\mathcal{L}_{\mathrm{abs}} = -\mathbb{E}_{(h_t, b_t^{\star})} \log p_\theta(b_t^{\star} \mid h_t, \ell)
$$

**变量解释**:
- $b_t^{\star}$: 把目标 progress $v_t^{\star}$ discretize 后的 target bin index
- 对成功 trajectory: $v_t^{\star} = t/T$, 即第 t 帧在总长 T 的相对位置
- 对失败 trajectory: 终帧 $v_t^{\star} = 0$
- **关键**: 这个 loss 只在 retry neighborhood 之外的帧上施加

**Intuition**: 这部分 loss 的作用是给模型一个"骨架", 告诉它"成功 trajectory 大致是 t/T 的对角线"。但是这条对角线在 retry 附近是错的, 所以那里要让位给 preference loss。

### 3.3 Retry 邻域的三个窗口 (Eq. 3-5)

这是整个方法最讲究的设计:

$$
\mathcal{W}_{i,j}^{\mathrm{pre}} = \{t : r_{i,j} - \Delta_{\mathrm{pre}} \leq t < r_{i,j} - \Delta_{\mathrm{near}}\}
$$

$$
\mathcal{W}_{i,j}^{\mathrm{near}} = \{t : r_{i,j} - \Delta_{\mathrm{near}} \leq t \leq r_{i,j} + \Delta_{\mathrm{near}}\}
$$

$$
\mathcal{W}_{i,j}^{\mathrm{post}} = \{t : r_{i,j} + \Delta_{\mathrm{near}} < t \leq r_{i,j} + \Delta_{\mathrm{post}}\}
$$

**变量解释**:
- $r_{i,j}$: 第 i 条 trajectory 的第 j 个 retry keypoint, 标的是 correction 开始的瞬间
- $\Delta_{\mathrm{near}}$: retry keypoint 周围的小邻域半径, paper 取 1 (5Hz 帧, 即 0.2 秒)
- $\Delta_{\mathrm{pre}}$: pre-retry 区间长度, paper 取 12 (2.4 秒)
- $\Delta_{\mathrm{post}}$: post-retry 区间长度, paper 取 12 (2.4 秒)

**几何 intuition**: 这相当于以 retry keypoint 为中心划了三层"洋葱": 最内层 (near) 是 value 的 local minimum 区, 外面两层 (pre 和 post) 分别是 drop 阶段和 rebound 阶段。

**Pair 构造规则** (paper 里 4 种 pair type):
1. **pre-vs-near**: pre 区的 frame 应该比 near 区的 value 高 (drop 方向)
2. **near-vs-post**: post 区的 frame 应该比 near 区的 value 高 (rebound 方向)
3. **pre-vs-pre**: 在 pre 区内, 离 retry 越远的 value 越高 (单调下降逼近 retry)
4. **post-vs-post**: 在 post 区内, 离 retry 越远的 value 越高 (单调上升远离 retry)

第 3、4 类 pair 是非常聪明的设计 —— 它们把"drop/rebound 的斜率方向"也编码进 preference, 不只是离散的"高/低"。

### 3.4 Soft-Window Weighting (Eq. 6-7)

$$
d(h^+; r_{i,j}) = |t^+ - r_{i,j}|
$$

$$
w(h^+, h^-) = \exp\left(-d(h^+; r_{i,j}) / \tau_w\right)
$$

**变量解释**:
- $t^+$: preference pair 中"应该 value 更高"那一端 (即 $h^+$) 的 endpoint timestamp
- $d(h^+; r_{i,j})$: $h^+$ 离 retry keypoint 的 temporal distance
- $\tau_w = 6.0$: decay 温度, 控制 soft 程度

**Intuition**: 离 retry keypoint 越近的 pair 越可信 (因为 retry 边界本身模糊), 越远的 pair 越可能是噪声。指数衰减给近的 pair 强权重, 远的 pair 弱权重。这相当于把 hard binary window boundary 变成了一个 RBF kernel, 避免 hard boundary 处 supervision 突变带来的训练震荡。

Table 6 提供了一个非常 solid 的 empirical 证据支撑这个设计: 标注者对 retry keypoint 本身的 inter-annotator error 只有 0.19s, 但对 pre/post 边界的 error 是 0.51s 和 0.66s —— 后者大 3 倍。这意味着 pre/post 边界本身就是 ambiguous 的, 用 soft weighting 才能容纳这种 ambiguity。

### 3.5 Preference Loss (Eq. 8)

$$
\mathcal{L}_{\mathrm{pref}} = -\mathbb{E}_{(h^+, h^-)} w(h^+, h^-) \log \sigma\left(\frac{V_\theta(h^+, \ell) - V_\theta(h^-, \ell)}{T_{\mathrm{pref}}}\right)
$$

**变量解释**:
- $h^+, h^-$: preference pair, $h^+$ 应该比 $h^-$ value 高
- $\sigma(\cdot)$: sigmoid function
- $T_{\mathrm{pref}} = 0.1$: temperature, 控制 preference sharpness
- $w(h^+, h^-)$: soft-window weight

这是经典的 **Bradley-Terry model** [6], 跟 RLHF [16] 里训练 reward model 的 loss 一样:
$$
P(h^+ \succ h^-) = \sigma\left(\frac{V(h^+) - V(h^-)}{T}\right)
$$

**关键 insight - 为什么 pairwise 而不是 regression**:

Ablation Table 3 里的 "w/o Preference Loss" 是最 striking 的数据点: 这个 baseline 不是简单地去掉 retry supervision, 而是**手工注入 value drop 到 progress target 里**再回归。直觉上这应该 work —— 你明明告诉模型"在 retry 附近 value 该 drop", 它应该学到。但结果是:

| 指标 | ReTVL | w/o Pref Loss |
|---|---|---|
| Drop AUC | 0.797 | 0.510 |
| Pre > Retry | 0.740 | 0.486 |
| Post > Retry | 0.967 | 0.742 |

也就是说, **直接告诉模型"这里 value 应该是多少"远不如告诉它"A 应该比 B 高"**。这背后的原因我认为是:

1. **Absolute target 是 ambiguous 的**: 你说 retry 时 value 该掉到 0.3 还是 0.5? 没人知道。但 A > B 这个 ordering 是确定的。
2. **Regression 容易让模型记忆 numeric pattern 而非 visual cue**: 模型可能学会"在 retry 附近几帧输出 0.3", 但没学会"识别 visual error 长什么样"。所以 Figure 6 里 drop-regression baseline 在 held-out trajectory 上的 drop 位置常常错位。
3. **Pairwise preference 是 invariant 的**: 对任意 monotonic transformation 不变, 更鲁棒。

这跟 RLHF 的发现是一致的: 给 reward model 评绝对分数很难, 给两个 response 排序就容易得多。

### 3.6 Total Loss (Eq. 9)

$$
\mathcal{L}_{\mathrm{value}} = \lambda_{\mathrm{abs}} \mathcal{L}_{\mathrm{abs}} + \lambda_{\mathrm{pref}} \mathcal{L}_{\mathrm{pref}}
$$

- $\lambda_{\mathrm{abs}} = 1.0$
- $\lambda_{\mathrm{pref}} = 3.0$

**Intuition**: preference loss 权重是 absolute loss 的 3 倍, 说明局部 mistake sensitivity 是重点。但 absolute loss 不能去掉 (Table 3 ablation: w/o Abs. Calibration 导致 VOC 从 0.987 掉到 0.929, S/F Det. 从 1.000 掉到 0.967) —— 没有全局 anchor, value model 会局部 drift, 失去对 success/fail 的判别能力。

采样比例: paper 用固定 $\rho_{\mathrm{pref}}$ 在每个 training step 里混 abs 和 pref 样本。

### 3.7 Value-Guided Weighted BC (Eq. 10-11)

$$
r_t = V_\theta(h_{t+\Delta_a}, \ell) - V_\theta(h_t, \ell)
$$

$$
\alpha_t = \mathrm{clip}\left(\frac{r_t - (\mu - 2\sigma)}{4\sigma + \epsilon}, 0, 1\right)
$$

$$
\mathcal{L}_{\mathrm{wBC}}(\psi) = \frac{\sum_t \alpha_t \, \ell_{\mathrm{BC}}(\pi_\psi(h_t), a_t)}{\sum_t \alpha_t + \epsilon}
$$

**变量解释**:
- $\Delta_a$: action chunk 的 stride, 这里 chunk size=16
- $r_t$: chunk 带来的 value 改进 (advantage)
- $\mu, \sigma$: 整个 dataset 上 $r_t$ 的均值和方差 (offline statistics, 提前算好)
- $\alpha_t \in [0,1]$: chunk 的归一化 weight
- $\pi_\psi$: policy, 这里是 $\pi_0$-style flow-matching VLA, 参数 $\psi$

**Intuition**: 这是 **Advantage-Weighted Regression (AWR)** [34, 37] 在 chunk-level 的应用。 $r_t$ 大表示这个 chunk 把 value 推高 (good action), $r_t$ 小或负表示这个 chunk 拖累了 progress (bad action)。 归一化逻辑是: 把 $r_t$ 标准化到 z-score, 然后线性映射 [-2σ, +2σ] 到 [0, 1]。这意味着只保留 advantage 在 top 2σ 之上的 chunk 权重接近 1, 在 bottom 2σ 之下的接近 0。

Downstream policy 是 flow-matching VLA:
$$
\mathcal{L}_{\mathrm{wBC}} = \frac{\sum_t \alpha_t \| v_\psi(z_s, h_t, s) - u_s \|_2^2}{\sum_t \alpha_t + \epsilon}
$$
其中 $z_s$ 是 flow time s 处的 noisy action chunk, $u_s$ 是 target velocity。Inference 用 10 flow steps 采样。

**为什么 ReTVL 这里效果好**: Figure 4 显示 ReTVL-BC 给 manually annotated bad actions 的平均权重是 0.11, 而 RECAP-BC 是 0.54 (5 倍差)。Table 7 更细: ReTVL 的 Strict-bad Retention 是 0.206, 比 Robometer 的 0.896 低 4 倍多 —— 它能精准过滤掉 harmful chunk 同时保留 useful corrective chunk (Post-retry Weight 0.636, 比 baseline 都高)。

## 4. 实验数据关键解读

### 4.1 Value Evaluation (Table 1)

| Method | VOC↑ | S/F Det↑ | Drop AUC↑ | Drop Prob↑ | Pre>Retry↑ | Post>Retry↑ |
|---|---|---|---|---|---|---|
| TOPReward | 0.903 | 0.844 | 0.264 | 0.612 | 0.329 | 0.917 |
| Robometer | 0.994 | 0.964 | 0.237 | 0.435 | 0.086 | 0.828 |
| RECAP-Value | 0.985 | 1.000 | 0.372 | 0.721 | 0.296 | 0.854 |
| **ReTVL** | 0.987 | 1.000 | **0.797** | **0.874** | **0.740** | **0.967** |

**关键观察**:

1. **Global metrics 几乎打平**: ReTVL 在 VOC (0.987) 和 S/F Det. (1.000) 上跟最强 baseline 持平, 说明它没牺牲全局能力。这是 ablation 里绝对 calibration loss 的功劳。
2. **Local metrics 大幅领先**: Drop AUC 是 baseline 最好的 (RECAP 0.372) 的 2.14 倍; Pre>Retry 是 baseline 最好的 (TOPReward 0.329) 的 2.25 倍。
3. **Robometer 悖论**: 它的 VOC 最高 (0.994), 但 Pre>Retry 最低 (0.086)。**全局 progress 拟合得越好, 局部 mistake sensitivity 越差**。这是一个非常重要的 finding —— 它说明 monotonic progress 是 local mistake 的"敌人", 你越逼模型相信 progress 单调, 它就越倾向于把所有 dip 都抹平。这本质上是 bias-variance trade-off 在 value learning 上的体现。

### 4.2 Policy Learning (Table 2)

| Task | Standard BC | RECAP-BC | ReTVL-BC |
|---|---|---|---|
| Pick up Spoon | 60 | 65 | 85 |
| Stack Blocks | 45 | 80 | 95 |
| Fold Towel | 50 | 65 | 80 |
| Open Drawer | 10 | 40 | 60 |
| Average | 41 | 63 | **80** |

**关键观察**:

1. ReTVL-BC 平均 80%, 比 Standard BC 高 +38.75%, 比 RECAP-BC 高 +17.50%。这个 gap 在 Open Drawer 上最大 (10 → 40 → 60), 因为 Open Drawer 是 articulated object + long-horizon, retry 事件最多最复杂。
2. Stack Blocks 从 45 跳到 95 是惊人的 —— 这个任务对 fine-grained alignment 要求高, human demo 几乎必然有 retry, 而 Standard BC 把所有 retry 都当成要模仿的动作学了, 导致 policy 也会 overshoot 然后 retry。ReTVL-BC 把"纠正动作"保留而把"导致错误需要纠正的动作"压下去, 所以 policy 一开始就不会犯那个错。

### 4.3 Ablation (Table 3)

| Method | VOC | S/F Det | Drop AUC | Drop Prob | Pre>Retry | Post>Retry |
|---|---|---|---|---|---|---|
| ReTVL | 0.987 | 1.000 | 0.797 | 0.874 | 0.740 | 0.967 |
| w/o Pref Loss | 0.994 | 1.000 | 0.510 | 0.836 | 0.486 | 0.742 |
| w/o Soft Window | 0.940 | 1.000 | 0.785 | 0.874 | 0.707 | 0.971 |
| w/o Abs Calib | 0.929 | 0.967 | 0.857 | 0.872 | 0.789 | 0.959 |

**有趣的不对称**:

- w/o Pref Loss: Local metrics 崩 (Drop AUC 0.797 → 0.510), 但 global 还行 (VOC 反而更高 0.994) → 单纯 regression 让模型更"光滑", global 更平滑但 local 失明
- w/o Soft Window: Local 几乎不变, VOC 掉 (0.987 → 0.940) → hard boundary 在 abs/pref 交界处制造 conflict, 拖累全局
- w/o Abs Calib: Local 几乎不变甚至略升, 但 global 大跌 → 没 anchor 时, value 在局部结构正确, 但全局尺度漂移, success/fail 分不开

这是一个非常 clean 的 decoupling: **pref loss 决定 local sensitivity, abs loss 决定 global scale, soft window 决定两者如何不打架**。

### 4.4 Chunk-Weighting Analysis (Table 7)

| Method | Success W↑ | Post-retry W↑ | Success Del↓ | Strict-bad Ret↓ |
|---|---|---|---|---|
| TOPReward | 0.539 | 0.498 | 0.330 | 0.695 |
| Robometer | 0.845 | 0.432 | 0.085 | 0.896 |
| RECAP-Value | 0.716 | 0.438 | 0.150 | 0.743 |
| **ReTVL** | 0.769 | **0.636** | 0.146 | **0.206** |

**这个表是 ReTVL 设计哲学的完美体现**: Robometer 给 success chunk 最高权重 (0.845) 但同时也 retain 了 89.6% 的 strict-bad chunk —— 它分不清"成功的有用动作"和"成功 trajectory 里夹杂的有害动作"。ReTVL 在保留 post-retry 纠正动作 (0.636, 最高) 的同时把 strict-bad 压到 0.206。这就是"selective weighting"而不是"trajectory-level filtering"的价值。

## 5. 我的批判性观察

### 5.1 优点
- **Annotation cost 极低**: 30 trajectories × < 1 min/traj, 这是一个很务实的可扩展性优势
- **Loss 设计 elegant**: 把 sparse keypoint 翻译成 dense pairwise supervision, 兼顾 global 和 local
- **Ablation 干净**: 三个组件 decouple 得很清楚

### 5.2 潜在问题
1. **Drop-and-rebound 假设的局限**: Paper 自己在 Limitations 里承认了 —— exploratory retry (试错性 retry) 和 failed correction (纠正失败) 不符合这个 pattern。在 Open Drawer 这种 articulated object 任务里, 这种 case 估计不少。
2. **Robometer 作为 backbone**: 所有 trainable 方法都用 Robometer-4B 做公平比较, 但这意味着 ReTVL 的 local sensitivity 提升多少来自 backbone 本身的 prior, 多少来自 loss 设计, 不完全清楚。可以想象用一个 from-scratch ViT 做 backbone 会更说明问题。
3. **下游只用 offline BC**: Eq. 10 的 $\alpha_t$ 是一个 static weight, 训练前算好就不再变。这是 offline weighted BC 的局限, 没有闭环反馈。Limitations 里提到了。
4. **Retry keypoint 的定义依赖 human**: "correction 开始的瞬间" 在某些 ambiguous case 上仍可能模糊。Table 6 的 inter-annotator error 0.19s 已经很小, 但对应 5Hz 是接近 1 帧的误差, 在 fast correction 上可能有影响。
5. **只测了 4 个 task**: 都是 tabletop manipulation, 没有 contact-rich (比如插入、拧紧) 任务。在那种任务里 retry 的语义可能更微妙。

### 5.3 跟大方向的关系
这篇 paper 实际上是在 **VLA scale-up** 和 **RLHF-style preference learning** 的交叉点上。最近 $\pi_{0.6}$ [3] 那篇 paper 提到用 experience-based learning 改进 VLA, 但他们走的是 Monte Carlo return 的路子, 仍是 monotonic。ReTVL 提供了一个 orthogonal 的角度: **人类演示里的"失败-纠正"结构本身是金矿**, 不应该被 monotonic progress 假设洗掉。

这与近年 Language model RL 里的一个反思暗合:prm (process reward model) vs ORM (outcome reward model) 之争 —— 只看 outcome 会丢掉 process 的局部信息。ReTVL 在 robot 语境下做的是类似的事情, 但 anchor 是 retry 而不是 step-level correctness label。

参考链接:
- arXiv: paper 没有明确的 arXiv 链接, 但作者来自 Tsinghua + MSR Asia + KAIST, 可以关注 https://arxiv.org/abs/2602.11075 (RISE) 和 https://arxiv.org/abs/2511.14759 (π0.6) 获取同领域近期工作
- RECAP / π0: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- AWR (Advantage-Weighted Regression): https://arxiv.org/abs/1910.00177
- Bradley-Terry model 原始论文: https://www.jstor.org/stable/2334022
- RLHF (Christiano et al.): https://arxiv.org/abs/1706.03741
- Robometer: https://arxiv.org/abs/2603.02115
- SARM: https://arxiv.org/abs/2509.25358
- Demo video: https://youtu.be/6aF6QrPg2To

## 6. 总结直觉

如果让我用一句话总结 ReTVL 的核心 insight: **demo 里的 retry 是 human 给你的免费 value gradient, monotonic progress assumption 把它扔了, ReTVL 用 pairwise preference 把它捡回来。**

把这套思路推广一下, 我能看到几个有趣的延伸:
- 在 **autonomous driving** 里, human driver 的 takeover / correction 事件是不是等价的 sparse value anchor?
- 在 **LLM agent** trajectory 里, human 在某步说"等等, 重新来" 是不是一种文本域的 retry keypoint?
- 在 **VLA self-improvement** 里, 如果让 VLA 自己 retry 失败的动作, 那个 retry 时刻的 state 能不能作为 self-supervised preference pair 的 anchor? 这就实现了 retry-supervised 的 self-play 闭环。

这篇 paper 在我看来最重要的贡献, 是把 robot learning community 从"clean data fixation"中拉出来一点, 提示大家: 模型学不好的部分, 往往就是数据里最 informative 的部分。
