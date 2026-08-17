---
source_pdf: HapticVLA Contact-Rich Manipulation via Vision-Language-Action.pdf
paper_sha256: 12c280760a88534ef68bf94d79a2f336f081d46b7289ce5a716b89686258b71a
processed_at: '2026-08-04T23:27:25-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 HapticVLA

好，把那些公式全扔掉，我给你讲个故事。

---

## 这帮人在干嘛

机器人抓鸡蛋这件事，人做起来毫无压力——你看一眼就知道鸡蛋脆，手就轻了。但 VLA 模型看一眼鸡蛋，跟看一眼铁球，在它眼里差不多。它不知道"脆"是什么感觉。

现在业界的解法基本都是：往 gripper 手指上贴 tactile sensor，让机器人能"摸到"东西，把摸到的信号喂给模型。问题是这玩意儿贵、跟主流 gripper 不兼容、装上去还很麻烦。

这帮人说：**训练的时候让机器人摸，部署的时候把 sensor 拆掉，机器人靠 vision 就能"假装"自己在摸**。

---

## 怎么做到的

分两步。

**第一步：训练一个会摸的 teacher**

他们拿 SmolVLA（一个 0.45B 的小 VLA）当 base，加上 tactile encoder，让它能看 vision + 摸 tactile + 读 proprioception，然后输出 action chunk。这部分不稀奇，别人也做过。

稀奇的是他们怎么 fine-tune 这个 teacher。他们不只用 task success 当 reward，而是从 tactile map 里提取了一堆 safety 信号：

- 捏太用力了 → 扣分
- 捏太轻了（hold 不住）→ 扣分
- 局部压力太高（要刺穿了）→ 扣分
- 压力太集中（单点接触，要 slip 了）→ 扣分
- 左右手压力不对称（歪了）→ 扣分
- 检测到 slip → 扣分

然后他们故意收集了一批"把东西捏坏了"的失败 demo。这些失败 demo 的 reward 是负的，在训练里权重被压得很低，几乎不参与。成功且安全的 demo 权重高，模型主要学这些。

这等于在 action space 里把"安全操作"的 mode 给 sharpen 出来了。Teacher 学到的不止是"怎么抓"，还有"怎么抓才不弄坏"。

**第二步：distill 出一个不用摸的 student**

Teacher 训练完了，但 deployment 时需要 tactile sensor。他们就把 teacher 的 action prediction 全部预先跑一遍存下来（51,251 个 samples），然后训练一个 student SmolVLA（没有 tactile encoder）去模仿 teacher 的输出。

但有个 trick：student 的训练 target 不是纯 teacher output，是 teacher output 跟 ground-truth demo 的 50/50 blend。一半学 teacher 的"tactile-aware 行为"，一半学人类 demo 的"smoothness"，两边互相牵制。

Student 的初始化也巧：直接 copy teacher 的 weights，只丢掉 tactile encoder 那部分，proprioception 的 projection 矩阵只保留前 6 列（对应 proprioception），丢掉后 128 列（对应 tactile embedding）。等于 student 一出生就继承了 teacher 对 proprioception 的理解，只需要学"怎么从 vision 推断 tactile 信息"这一件事。

---

## 结果有多惊艳

三个 task：抓 jar（中等难度）、抓 waffles（高难度，脆）、抓 egg（极高难度，更脆）。

| Model | Jar | Waffles | Egg | Mean |
|-------|-----|---------|-----|------|
| SmolVLA base | ~75% | ~35% | ~10% | ~40% |
| Teacher (有 tactile) | 85% | 85% | 55% | 75% |
| X-VLA (0.9B) | 0% | 0% | 0% | 0% |
| VLA-0 | 0% | 0% | 0% | 0% |
| **HapticVLA (没 tactile)** | 75% | 90% | 95% | **86.7%** |

两个 observations：

**第一，X-VLA 和 VLA-0 两个更大的 baseline 在这些 task 上完全 0% 成功率。** 这帮人没解释，但我的猜测是：大模型在小数据上 fine-tune 容易出问题，加上这些模型预训练里可能没怎么见过"脆的东西"这个概念，纯靠 vision + 几百 episodes 学不会 force-aware behavior。

**第二，也是更离谱的：student 反超 teacher。** Teacher 有 tactile sensor，86.7% vs 75%。没 sensor 的反而比有 sensor 的高 11 个百分点。

---

## 为什么没 sensor 反而更好

这是整篇 paper 最有意思的地方。Paper 自己没深挖，但我的几个猜测：

**Tactile sensor 是个 noisy signal source。** 120Hz、10×10 taxel array，每个 reading 都有 measurement noise。Teacher online condition 在这些 noisy reading 上，action 会 jitter。Student 用 vision 替代，vision 信号比 tactile 稳定得多。

**Vision 本身就 carry 了大部分 material property 信息。** 你看到 egg 的形状、纹理、颜色，VLA 内部的 vision encoder 已经能 activate "fragile" 这个 concept。Tactile 在很多场景下是 redundant verifier，不是必要 input。

**Blended target 起 low-pass filter 作用。** 50% GT + 50% teacher，GT 是人类 demo（很 smooth），blending 等于把 teacher 的 jitter 给 average 掉了。

**Teacher 过度依赖 tactile 导致它没好好学 vision。** 既然有 tactile 信号，teacher 就懒了，不好好从 vision 推断 material property。Student 被迫从 vision 推断，反而学得更 robust。

这四个因素哪个主导，paper 没 ablate，但综合效果是明确的：**online tactile conditioning 不一定比 offline tactile distillation 好**。

---

## 这跟别的工作什么关系

最接近的是 FD-VLA，也是 force distillation，也是 inference 时没 force sensor。区别是 FD-VLA 没有 RL component，只是 supervised distillation。HapticVLA 多了 SA-RWFM 这一层 RL，让 teacher 学到的是 sharpened safe distribution，student 继承这个 sharpened distribution。

跟 Tactile-VLA / OmniVTLA / VTLA 这些的区别更明显：那些都要 tactile sensor 在 inference 时存在。HapticVLA 把硬件成本从 deployment 移到 training。

跟 Gano et al. (IROS 2025) 思想相似——他们做 visuo-tactile pre-training 然后 disable sensor at inference。但 Gano 是直接 disable，没有 explicit distillation mechanism。HapticVLA 的 distillation 是更 structured 的 transfer。

---

## 我的看法

**好的地方：**

Reward design 是 paper 最有价值的部分。把"safety"拆成 6 个 axis（force、peak、concentration、asymmetry、slip、holding），每个独立 penalize，用 ReLU² 让 penalty 平滑可微。这个 decomposition pattern 可以推广到任何 contact-rich task。

"Training 时用 hardware，inference 时不用"这个范式很 elegant。跟 RLHF 里"训练 reward model，部署只用 policy"结构同构。

故意收集 failure demo 这个设计点很好。大多数 offline RL dataset 只有成功 demo，failure 提供的"什么是 unsafe"信息缺失。HapticVLA 的 70 success + 20 faulty 比例值得学习。

**存疑的地方：**

310 real episodes + 1000 sim episodes，数据量太小。Scaling curve 没给。如果只在 SO-101 + 3 个 pick-and-place task 上 work，generalization 存疑。

Reward 有 6 个 axis + 7 个 thresholds + 6 个 weight coefficients，全是手工设计。每换一个 task 可能要 re-calibrate。这跟"end-to-end learning"的大方向有点拧。

X-VLA 和 VLA-0 的 0% 成功率没解释。Baseline 是否 properly fine-tuned 存疑。

Sim data 的贡献没 ablate。Tactile 的 sim-to-real gap 在接触动力学上可能很大。

Student 反超 teacher 这个最强 claim 没 ablate 原因。α=0 vs α=0.5 vs α=1 的对比没做。

**更深层的 intuition：**

这篇 paper 在暗示一个有意思的观点：**tactile 信息在 contact-rich task 上是 training-time verifier，不是 inference-time necessity**。你 vision 看到 egg 就知道要轻，tactile 只在 training 时帮你确认"这个 vision 推断是对的"。一旦 VLA 内化了"看到脆的东西就轻"这个 mapping，tactile 就 redundant 了。

这跟人类学习类似：小孩摸过一次鸡蛋碎了，以后看到鸡蛋手就轻了。他不靠每次都摸来调节力度，他靠 vision activate 的 prior。

所以这篇 paper 的 deeper claim 其实是：**contact-rich manipulation 的核心是 vision→material property 的 cross-modal inference 能力，tactile 只是训练这个能力的 supervision signal**。如果这个 claim 对，那未来大规模 tactile dataset 的价值不在于让 robot 用 tactile 做 inference，而在于让 robot 从 tactile supervision 里学到 vision 的 material understanding。

---

## 一句话总结

让机器人训练时戴 tactile glove 学手感，部署时摘掉 glove 靠眼神抓鸡蛋，效果比一直戴着 glove 还好——因为 online tactile 信号是 noise，offline tactile 是 supervision。

代码和数据开源了（paper 说在 project website，markdown 里没给完整 URL，应该找 PDF 的 footnote）。

如果你想我再展开某一块（比如 reward weights 的 sensitivity、async inference 的具体机制、或 student 反超 teacher 的更深入 ablation 设计），说一声。

---

# HapticVLA: Contact-Rich Manipulation without Inference-Time Tactile Sensing — 深度拆解

Karpathy 你好，这篇 paper 切入点很关键也很反直觉：**tactile-aware behavior 不一定要靠 tactile sensor 来 deploy**。整体方法论落在 "offline tactile-aware RL fine-tune → cross-modal distillation → sensorless deployment" 这条链路上，跟近期 FD-VLA 的 force distillation 思路有亲缘关系，但用 reward-weighted flow matching 把 "safe distribution" sharpen 出来这一步是关键差异。下面拆解。

---

## 1. Problem Framing & Why It Matters

传统 VLA pipeline (OpenVLA, π₀, SmolVLA) 是 vision + proprioception → action chunk。问题在 contact-rich 任务里非常明显：你 vision 看到一个 egg，但你不知道 egg 的壳有多脆、gripper 闭合到什么程度会 crack。Paper 列了三种现行方案：

1. **Visual-tactile sensors on fingers** (Tactile-VLA, VTLA, OmniVTLA, VLA-Touch, MLA, BiTLA) — 把 GelSight/DIGIT 类的 visual-tactile 当 vision modality 喂进 VLM。问题：硬件贵、跟主流 gripper 不兼容、低层 force 信号被 vision tokenizer 摊平。
2. **Joint force/torque sensors** (ForceVLA, TaF-VLA) — 用 motor 电流反推 external force。问题还是硬件依赖。
3. **Distillation during training, no sensor at inference** (FD-VLA, Gano et al.) — HapticVLA 落在这一类。

核心 claim：把 tactile 信号在 offline 阶段烧进 policy 的 latent，inference 时纯 vision + state 就能复现 tactile-aware behavior。这等于在说 "tactile 信息有相当一部分是 vision-conditionable 的"——这是个强的可学习性假设，跟人类 cross-modal inference (你看到 egg 就知道要轻) 直觉吻合。

参考：
- π₀: https://arxiv.org/abs/2410.24164
- SmolVLA: https://arxiv.org/abs/2506.01844
- OpenVLA: https://arxiv.org/abs/2406.09246
- FD-VLA: https://arxiv.org/abs/2602.02142
- Tactile-VLA: https://arxiv.org/abs/2507.09160
- MLA: https://arxiv.org/abs/2509.26642

---

## 2. 三阶段 Pipeline 总览

```
[ Episode dataset (vision, state, tactile, success/damage labels) ]
              │
              ▼
(1) Offline Tactile Reward Calculation   ──► per-step r_t + R_episode
              │
              ▼
(2) SA-RWFM Teacher Training (SmolVLA + tactile encoder + reward-weighted FM loss)
              │
              ▼
(3) Tactile Distillation ──► Student SmolVLA (no tactile encoder, predicts action from vision+state only)
              │
              ▼
       Deployment: vision + state → action chunk, no tactile hardware
```

直觉上这是一个 **"先让 teacher 看见真实物理，再让 student 模仿 teacher 的行为而看不见物理"** 的 cross-modal distillation。Teacher 不只是模仿 demo，而是被 tactile reward 重塑过 action distribution；student 拿到的是被重塑过的 soft target，所以 student 学到的不止是 "demo 的轨迹"，而是 "在 tactile awareness 下被筛选/修正过的轨迹"。

---

## 3. SA-RWFM: Safety-Aware Reward-Weighted Flow Matching

### 3.1 为什么 Flow Matching 上做 RL 不好做

Flow matching 的训练目标是 velocity field $v_\theta(x_t, t)$ 回归到 $a - x_0$，其中 $x_t$ 是 noisy action，$t$ 是 flow time，$a$ 是 ground-truth action chunk，$x_0$ 是起点 noise。这是个 imitation 信号。要往里塞 reward，最朴素的做法是 PPO 那种 on-policy，但 FM 是 generative model，每次 sample 要 integrate ODE，rollout 成本高，variance 大。Pfrommer et al. 提出的 RWFM 思路是把 reward 当作 per-sample 的 importance weight，重写 imitation loss：

$$
L_{\text{rwfm}} = \frac{\sum_i w_i L_i}{\sum_i w_i}
$$

其中 $L_i$ 是 sample $i$ 的 FM loss，$w_i$ 是从 reward 计算出的权重。这样高 reward 轨迹权重大，低 reward 轨迹权重小，等于把 distribution 朝高 reward 区域拉。本质上是个 off-line weighted regression，比 on-policy RL 简单且稳定。HapticVLA 在这个基础上做了几件事：

- **Reward 设计是 safety-aware**（不光是 task success，还有 contact safety 的多项惩罚）。
- **用 robust group-wise normalization** 防止 scale bias。
- **用 anchor regularization** 防止 mode collapse。
- **Mixed local/global returns**，让 advantage 同时考虑短 chunk 和整 episode。

参考：
- RWFM (Pfrommer): https://arxiv.org/abs/2507.15073
- ARFM: https://arxiv.org/abs/2509.04063
- ReinFlow: https://arxiv.org/abs/2505.22094

### 3.2 Tactile Reward 的设计 — 这是 paper 的精华

每步拿到左右两个 tactile map $M_t^L, M_t^R \in [0,1]^{H\times W}$。$H=W=10$（10×10 taxel array），每个 pixel 强度对应 1–9N 的 force 量级。从这里提取 5 个统计量：

**Mean force proxy** (Eq. 1)：
$$
f_t^s = \frac{1}{HW}\sum_{i,j} M_{t,ij}^s
$$
- $s \in \{L, R\}$：active tactile side
- $H, W$：tactile map 高宽（=10）
- $M_{t,ij}^s$：side $s$ 在时刻 $t$ 像素 $(i,j)$ 的 normalized force 读数

**Peak pressure** (Eq. 2)：
$$
p_t^s = \max_{i,j} M_{t,ij}^s
$$
- max over pixels，捕捉局部高压点

**Pressure concentration** (Eq. 3)：
$$
c_t^s = \frac{p_t^s}{HW \cdot f_t^s + \varepsilon}
$$
- 这个比值大意味着 force 集中在一个小区域（比如尖点接触），小意味着 force 均匀分布
- 物理意义：均匀分布是稳定 grasp 的特征，集中往往是单点接触 / 即将 slip 的前兆

**Center of Pressure (CoP)** (Eq. 4)：
$$
\text{cop}_{x,t}^s = \frac{\sum_{i,j} M_{t,ij}^s X_{ij}}{\sum_{i,j} M_{t,ij}^s + \varepsilon}, \quad \text{cop}_{y,t}^s = \ldots
$$
- $(X_{ij}, Y_{ij}) \in [0,1]^2$ 是 normalized grid coordinate
- CoP 是 force 加权的 spatial centroid，类似 image 的 brightness centroid
- 后面 slip detection 要用它的 jump

**Slip detection** (Eq. 5-6)：
$$
\text{slip}_t = \mathbb{I}[h_t=1] \cdot \mathbb{I}\left[\max_{s\in\mathcal{A}} \Delta\text{cop}_t^s > c_{op} \;\vee\; \min_{s\in\mathcal{A}} \Delta f_t^s < -d_f\right]
$$
- $h_t \in \{0,1\}$：holding state（gripper 闭合且稳定接触）
- $\Delta\text{cop}_t^s = \|(\text{cop}_{x,t}^s, \text{cop}_{y,t}^s) - (\text{cop}_{x,t-1}^s, \text{cop}_{y,t-1}^s)\|_2$：CoP 在相邻帧的 L2 jump
- $c_{op}, d_f$：calibrated 阈值
- 两个 OR 条件：要么 CoP 突然跳（接触点滑动），要么 force 突然掉（gripper 在物体上滑脱）
- $\mathcal{A} \subseteq \{L,R\}$：active side 集合，calibration 后只考虑真正接触的那一面

**Per-step reward** (Eq. 7)：一个 stack of ReLU² penalties：

$$
\begin{aligned}
r_t = & -\sum_{s\in\mathcal{A}} \Big(
  \lambda_{high} \text{ReLU}(f_t^s - f_{max})^2 \\
  & + \lambda_{low} \mathbb{I}[h_t=1] \text{ReLU}(f_{min} - f_t^s)^2 \\
  & + \lambda_{peak} \text{ReLU}(p_t^s - p_{max})^2 \\
  & + \lambda_{conc} \text{ReLU}(c_t^s - c_{max})^2 \Big) \\
  & - \lambda_{asym} \mathbb{I}[L,R\in\mathcal{A}] \text{ReLU}(|f_t^L - f_t^R| - \delta)^2 \\
  & - \lambda_{slip} \text{slip}_t
\end{aligned}
$$

变量含义：
- $f_{min}, f_{max}$：safe force band 上下限（calibrated from dataset quantiles）
- $p_{max}$：peak pressure 上限
- $c_{max}$：concentration 上限
- $\delta$：inter-pad force asymmetry tolerance
- $\lambda_{\cdot}$：每项的权重系数
- ReLU² 而非 ReLU：让惩罚平滑可微，gradient 在 threshold 附近不为 0

直觉：这六项分别在说：
1. **过大力**（捏碎脆弱物体）
2. **过小力 while holding**（hold 不住，要 drop）
3. **peak 过高**（局部刺穿，即使平均力 OK）
4. **concentration 过高**（单点压强，单点破损）
5. **左右不对称**（歪斜 grasp，旋转 slip 前兆）
6. **slip 事件**（无论 CoP jump 还是 force drop）

这是个非常 physics-informed 的 reward shaping，每项都对应一种 failure mode。比 "task success" 这种 0/1 信号 dense 得多。

**Episode risk** (Eq. 8)：
$$
\text{risk} = \text{clip}\left( P_{95}(\{e_t : t\in\mathcal{M}\}) + \frac{1}{2T}\sum_{t=1}^T \text{slip}_t, \;0,\;1\right)
$$
- $e_t$：normalized threshold exceedance（max over force, peak, concentration 的超标程度）
- $\mathcal{M}$：holding state = 1 的 timesteps 集合
- $P_{95}$：95th percentile，避免单个 outlier spike 主导
- $\frac{1}{2T}\sum \text{slip}_t$：slip event 的频率，归一化到 [0, 0.5]
- clip 到 [0,1]

用 $P_{95}$ 而不是 $\max$ 是个 robust 选择，避免一个 tactile glitch 就把整个 episode 标记为 risky。

**Episode reward** (Eq. 9-10)：
$$
R_{episode} = R_{step}\frac{1}{T}\sum_t r_t + R_{succ}\cdot \text{success} - R_{drop}\cdot \text{drop} - R_{damage}\cdot \text{damage} - R_{risk}\cdot \text{risk}
$$
- $R_{step}, R_{succ}, R_{drop}, R_{damage}, R_{risk}$：各项 scale 系数
- 同时考虑：step-level comfort，task success，object drop，object damage，near-failure risk

设计上很像一种 **multi-task penalty + bonus** 的 hybrid reward，既给 dense step signal 又给 sparse outcome signal。

### 3.3 Reward-Weighted FM 的具体计算

**Per-sample masked FM loss** (Eq. 11)：
$$
L_i = \frac{\sum_{h,j} m_{i,j} \ell_{i,h,j}}{H\sum_j m_{i,j} + \varepsilon}
$$
- $\ell_{i,h,j}$：sample $i$ 在 chunk step $h$、DoF $j$ 上的 FM loss element
- $m_{i,j}\in\{0,1\}$：task-dependent DoF mask（比如 bimanual task 时只激活 relevant arm 的 DoF）
- $H$：chunk horizon

**Chunk return** (Eq. 12)：
$$
R_t^{chunk} = \sum_{k=0}^{H-1} \gamma^k r_{t+k}, \quad \gamma = 0.99
$$
- $\gamma$：discount factor，0.99 意味着 100 步 horizon，对 50-step chunk 几乎是 undiscounted
- chunk return 是 local credit assignment

**Robust group-wise normalization** (Eq. 13-14)：
$$
z = \frac{x - \text{median}_g(x)}{\text{scale}_g}, \quad \text{scale}_g = \max(1.4826\cdot \text{MAD}_g(x), \varepsilon)
$$
- $g$：group label（比如按 task 分组）
- $\text{median}_g$：group 内 median
- $\text{MAD}_g$：median absolute deviation
- $1.4826$：MAD 到 std 的转换因子（对正态分布，$\sigma \approx 1.4826 \cdot \text{MAD}$）
- 用 median + MAD 而不是 mean + std 是为了 robust to outlier episodes（比如一个爆炸性高 reward 的 episode 不应该把别的都压成低权重）

**Advantage** (Eq. 15)：
$$
A_i = \text{clip}(\beta z_i^{epi} + (1-\beta) z_i^{chunk}, -c_A, c_A)
$$
- $\beta = 0.7$：偏 episode，因为 outcome 更可靠
- $z^{epi}, z^{chunk}$：episode-level 和 chunk-level 的 robust normalized scores
- $c_A = 6$：clip 防止单个 sample advantage 过大

**Weights** (Eq. 16-17)：
$$
w_i^{raw} = \exp(\alpha A_i), \quad w_i^{clip} = \text{clip}(w_i^{raw}, w_{min}, w_{max}), \quad w_i = \frac{w_i^{clip}}{\frac{1}{B}\sum_b w_b^{clip}}
$$
- $\alpha = 0.25$：temperature，控制权重的 sharpness
- $w_{min}=0.25, w_{max}=4.0$：clip 范围，最大最小 16 倍 ratio
- 归一化让 batch 内 weights 均值 = 1

这套 $\alpha \to \text{clip} \to \text{normalize}$ pipeline 很像 **softmax with temperature + clipping**。$\alpha=0.25$ 偏保守，意味着即使 advantage 是 6（最大），$\exp(0.25 \times 6) \approx 4.48$，刚好被 $w_{max}=4$ 截掉。整个 design 选择是 "soft reweighting"，避免极端 sample 主导。

**Anchor regularization** (Eq. 19-20)：
$$
L_{anchor} = \frac{1}{|\Theta|}\sum_{p\in\Theta} \|\theta_p - \theta_p^0\|_2^2
$$
- $\Theta$：所有参数
- $\theta_p^0$：imitation 训练完的初始参数
- $L_2$ distance to initial parameters
- $\lambda_{anchor}$：warm-up schedule，early training 大，后期减小

这是个 **L2 anchor to imitation init**。直觉：reweighting 会让 policy drift 到 high-reward 但 mode-collapsed 的状态，anchor 把它拉回 imitation 的 manifold。跟 RL 中的 KL penalty to reference policy 是同类思想，但这里是对 parameters 直接做 L2，而不是对 actions 做 KL。简单粗暴但有效。

参考 Fan et al. 的 Wasserstein regularization 和 ARFM 的 adaptive scaling 都是同源问题（FM + RL 容易 collapse）的不同解法。

参考：
- Online RWFM with Wasserstein reg: https://arxiv.org/abs/2507.15073
- ARFM: https://arxiv.org/abs/2509.04063

### 3.4 关于 Negative Examples 的关键作用

Paper 在 dataset collection 部分明确说：每 task 收 70 成功 + 20 faulty episodes。faulty 定义为 "excessive grasping force produced permanent structural damage or significant object deformation"。

这是个非常关键的设计选择。RL literature 里大多数 offline dataset 只有成功 demo，失败要么没有要么占比很小。HapticVLA 故意制造 failure，因为这些 failure 提供 "什么是 unsafe action distribution" 的信息。配合 reward weighting：
- $w_i^{raw} = \exp(\alpha A_i)$，failure 的 $A_i$ 是负数，$w_i$ 极小（被 clip 到 0.25）
- success 的 $A_i$ 是正数，$w_i$ 大（最高 4）

结果：FM loss 几乎只拟合 high-reward samples 的 velocity field，low-reward samples 几乎不参与训练。这等于在 action space 里 sharpen 出一个 high-reward mode。

这跟 DPO (Direct Preference Optimization) 的思想是相通的：用 pairwise preference 推出一个 implicit reward，再用这个 reward 重塑 policy。VTLA 用了 DPO loss 处理 insertion task，HapticVLA 在这里走的是 reward-weighted regression 路线，但有相似的偏好建模能力。

---

## 4. Tactile Distillation (TD)

### 4.1 Motivation

Teacher (SmolVLA + tactile encoder + SA-RWFM) 在 inference 时需要 tactile sensors，deployment 受限。Student 是普通 SmolVLA，无 tactile input。要做的是把 teacher 的 tactile-aware action distribution 编码进 student 的 vision+state→action 映射。

### 4.2 Three Stages

**Stage 1: Offline Teacher Target Generation** (Eq. 21)：
$$
\hat{a}_i^T = \pi_T(o_i, s_{T,i}) \in \mathbb{R}^{H\times d_a}
$$
- $\pi_T$：teacher policy
- $o_i = (I, \ell)$：视觉观察 $I$ + 语言指令 $\ell$
- $s_{T,i} = [q; f] \in \mathbb{R}^{134}$：proprioception $q\in\mathbb{R}^6$ + tactile embedding $f\in\mathbb{R}^{128}$
- $H = 50$：action chunk horizon
- $d_a = 6$：per-arm joint dimension
- $N = 51,251$ samples

预先把所有 training samples 跑一遍 teacher inference，缓存 predictions。这把 distillation 变成 offline，不需要 teacher online。

**Stage 2: Teacher Backbone Initialization** (Eq. 22-23)：
$$
\Theta_S^{(0)} = \{\theta \in \Theta_T \mid \theta \notin \Theta_{tactile}\}
$$
$$
W_S = W_T[:, :d_a] \in \mathbb{R}^{32\times 6}, \quad b_S = b_T
$$
- $\Theta_T$：teacher 所有参数
- $\Theta_{tactile}$：tactile encoder 参数（被丢弃）
- $W_T \in \mathbb{R}^{32\times 134}$：teacher 的 state projection，把 [q; f] 投影到 32 维 latent
- $W_S \in \mathbb{R}^{32\times 6}$：取 $W_T$ 前 6 列（对应 proprioception），丢掉后 128 列（对应 tactile）
- $b_S$：bias 复用

这是非常 surgical 的 initialization：把 teacher 学到的 proprioception → latent mapping 完整保留，只丢 tactile 部分。Student 不需要从头学 proprioception encoding。

**Stage 3: Blended Target Training** (Eq. 24-25)：
$$
\tilde{a}_i = (1-\alpha) a_i^{GT} + \alpha \hat{a}_i^T
$$
$$
\mathcal{L}_{distill}(\theta) = \mathbb{E}_{t, x_0, \tilde{a}} \left[\|v_\theta(x_t, t) - (\tilde{a} - x_0)\|^2\right]
$$
- $\alpha \in [0,1]$：blending coefficient，paper 用 0.5
- $a_i^{GT}$：ground truth demo action chunk
- $\hat{a}_i^T$：teacher prediction
- $v_\theta$：student 的 flow matching velocity field
- $x_t$：noisy action at flow time $t$
- $x_0$：noise 起点

Validation 时设 $\alpha = 0$，让 student 完全拟合 GT，得到 unbiased reconstruction quality。

Blending 0.5 的两个互补信号：
1. **GT anchoring (1-α=0.5)**：防止 student 放大 teacher error，保 grounding
2. **Teacher shaping (α=0.5)**：注入 tactile-aware force modulation

这个 trick 很关键。如果只 distill teacher target，student 会学到 teacher 的所有 bias 包括错误；如果只学 GT，distillation 没意义。α=0.5 平衡，类似 PID 里的 setpoint + measurement blend。

---

## 5. Experimental Setup

### 5.1 Hardware

- **Bimanual SO-101 arms** (LeRobot platform)
  - Left: 标准 SO-101，7.4V Feetech STS3215 servos
  - Right: 12V 版本，加 tactile-equiped parallel gripper
- **Cameras**: Intel RealSense D435 (external) + 2× IMX335 5MP wrist cameras, 640×480
- **Tactile**: 10×10 taxel array per fingertip, 200 taxels total, 120Hz, 1-9N range
- **Compute**: NVIDIA Jetson Orin NX 16GB

参考 LeRobot: https://github.com/huggingface/lerobot

### 5.2 Tasks

| Task | Difficulty | Failure mode |
|------|-----------|--------------|
| Jar pick-and-place | 中 | plastic jar 变形 |
| Waffles pick-and-place | 高 | waffles 碎裂 |
| Egg pick-and-place | 极高 | egg 破裂 |

每个 task 20 trials。Success = object intact + placed at target。

### 5.3 Dataset

- Real: 310 episodes (3 tasks)
  - Per single-arm task: 70 success + 20 faulty
  - Bimanual: 100 success + 30 faulty
- Sim (Isaac Sim digital twin): 1000 额外 pick-and-place episodes with randomized object pose / target box

总训练样本：51,251（应该包含 augmentations）。

---

## 6. Main Results

### 6.1 Success Rate Comparison (Fig. 6 数据近似)

| Model | Jar | Waffles | Egg | Mean |
|-------|-----|---------|-----|------|
| SmolVLA (base) | ~75% | ~35% | ~10% | ~40% |
| SmolVLA + SA-RWFM | 85% | 85% | 55% | 75% |
| X-VLA (0.9B) | 0% | 0% | 0% | 0% |
| VLA-0 | 0% | 0% | 0% | 0% |
| **HapticVLA (full)** | **75%** | **90%** | **95%** | **86.7%** |

几个观察：

1. **X-VLA 和 VLA-0 完全 0%** 非常出乎意料。这两个比 SmolVLA 大或预训练更强的 model 在 contact-rich 任务上完全失败。Paper 没有详细分析，但我的猜测：
   - Fine-tune data 太少（几百 episodes），大 model 容易 overfit 或欠 fit
   - 这些 model 可能预训练里没接触过 "fragile object" 的概念，hardness-aware behavior 没法纯靠 vision + 几百 episodes 学出来
   - Action chunking / horizon 跟 SO-101 的 control frequency 不匹配

2. **SmolVLA + SA-RWFM (75%) vs. HapticVLA (86.7%)**：distillation 反超 teacher！这是一个非常反直觉的结果。

3. **Egg task 是最难分水岭**：base SmolVLA 几乎 0%，HapticVLA 95%。tactile-aware reasoning 在这 task 上贡献最大。

### 6.2 Ablation Study (Table I)

| Model | Jar | Waffles | Egg | Mean |
|-------|-----|---------|-----|------|
| w/o TD, async | 16/20 | 18/20 | 15/20 | 81.7% |
| w/o TD (sync) | 11/20 | 17/20 | 17/20 | 75% |
| w/ TD, async | 14/20 | 19/20 | 15/20 | 80% |
| **w/ TD (sync, HapticVLA)** | 15/20 | 18/20 | 19/20 | **86.7%** |

四个 cell 的对比非常 instructive：

**Column 比较 (sync vs async)：**
- w/o TD：sync 75% < async 81.7%（async 更好）
- w/ TD：sync 86.7% > async 80%（sync 更好）

**Row 比较 (TD vs no TD)：**
- sync：no TD 75% < TD 86.7%（TD 大幅帮助）
- async：no TD 81.7% > TD 80%（TD 略微 hurt）

这个交叉很有意思。我的解读：
- **Async inference 引入 temporal misalignment**：tactile observation 跟 action chunk 之间有时间差。w/o TD 时 teacher 有 tactile sensor，async 的 misalignment 让 tactile feedback 滞后导致部分 grasp 失败（但 sync 又因为 latency 太大反而不好... wait）。
- 等等，重新读 paper："We attribute this degradation to temporal misalignment and increased effective latency between tactile observations and control actions." 这是说 **async degradation** 来自 tactile 时序错位。
- 那么 sync + no TD 应该最好，但实际 sync + no TD = 75% < async + no TD = 81.7%。这跟 paper 的解释矛盾。

更合理的解释可能是：async 时 chunk 之间有 overlap，相当于对 action 做 temporal smoothing，对 tactile-related noise 有 average 效应。而 sync 时一次性执行整个 chunk，更容易被单步 tactile reading 误导。

- **TD 的作用**：distillation 把 tactile 信号烧成 vision+state 的 implicit representation。这相当于**去掉了 inference 时序对齐问题**——student 不依赖实时 tactile，所以 sync/async 差异变小。
- TD + sync 反超 w/o TD + async 5个百分点：distillation 不止补回 tactile 信号，还起 regularization 作用。

### 6.3 Distillation Beats Teacher — 为什么？

86.7% (TD) > 81.7% (teacher, async) > 75% (teacher, sync)。这是 paper 最强的 claim 之一。几个可能的解释：

1. **Teacher 过度依赖 tactile signal**：teacher 见到 tactile 数据就 condition on 它。tactile sensor 有 noise（120Hz 是高的，但每个 reading 有 measurement noise），noise 进入 action 生成过程导致 jitter。Student 用 vision 替代 tactile，vision 信号稳定得多。

2. **Blended target 起 regularization**：α=0.5 让 student 同时学 GT 和 teacher。GT 是人类 demo，已经 smooth。Blending 相当于在 action space 做 low-pass filter。

3. **Vision 能 implicit 推断 material property**：看到 egg 的形状、纹理，VLA 内部已经能激活 "fragile" 概念。Tactile 是 redundant signal。Distillation 让 student 学会从 vision 推出 "this needs gentle"，而 teacher 用 tactile 信号 verify。

4. **Anchor regularization 在 student 阶段已经稳定**：student 从 teacher init，权重已经在 good region，distillation loss 只 fine-tune 细节。

这其实指向一个更深的观点：**tactile signal 在某些任务上是 "verifiable" 而不是 "necessary"**。你 vision 看到对象就能推断出大部分 physical property，tactile 只是在线 verify。Distillation 把 verify 步骤 offline 化。

---

## 7. 跟相关工作的差异

### 7.1 vs FD-VLA (最接近)

FD-VLA (arxiv 2602.02142) 是最相似的——也是 force distillation, no force sensor at inference。区别：

| | FD-VLA | HapticVLA |
|---|--------|-----------|
| Modality | Force (joint motor force) | Tactile (fingertip array) |
| Teacher | Force-aligned VLA | SA-RWFM with safety reward |
| Distillation target | Learnable force token prediction | Blended action target |
| Reward signal | 无明确 RL | Explicit RL via RWFM |
| Action expert | 普通 VLA | Flow matching with RWFM |

HapticVLA 的 RL component 是关键差异——FD-VLA 的 force distillation 更接近 supervised learning，HapticVLA 的 SA-RWFM 让 teacher 学到 sharpened safe distribution，student 继承这个 sharpened distribution。

### 7.2 vs Tactile-VLA / OmniVTLA / VTLA

这些都需要 tactile sensor at inference。Hardware cost 高，compatibility 差。HapticVLA 的 deployment-friendly 优势直接，但放弃 online tactile feedback 意味着对 truly novel objects (vision 看不出的 material property) 适应性差。

### 7.3 vs Gano et al.

Gano et al. (IROS 2025) 做 low-fidelity visuo-tactile pre-training + vision-only inference，在 insertion task 上效果好。思想类似但没 distillation——他们直接 pretrain 后 disable sensor。HapticVLA 的 distillation 是更 explicit 的 transfer mechanism。

参考 Gano et al.: https://arxiv.org/abs/2507.09160 (类似思想)

### 7.4 vs RWFM literature

Pfrommer 的 RWFM + Fan 的 Wasserstein regularization + ARFM 的 adaptive scaling + ReinFlow 的 online RL——这些都是 FM + RL 的不同变体。HapticVLA 选 offline RWFM + anchor reg，更保守更稳定，跟 ARFM 的 adaptive scaling 思路相通。

---

## 8. 批判性思考 & 潜在问题

### 8.1 Dataset 规模

Real 310 episodes + sim 1000 episodes。对 VLA 标准来说非常小。但任务也相对窄（3 个 pick-and-place）。Ablation 没做 "数据量 vs 性能" 的 scaling curve。如果只在 SO-101 平台、3 个 task 上有效，generalization 存疑。

### 8.2 Reward 设计的手工性

6 个 reward 项（force, peak, concentration, asymmetry, slip, holding）+ 多个 thresholds ($f_{min}, f_{max}, p_{max}, c_{max}, \delta, c_{op}, d_f$) + 多个 weight ($\lambda_{\cdot}$)。这些全是手工设计，每换一个 task 都可能要 re-calibrate。Paper 说 thresholds 从 dataset quantiles calibrated，但 weights 没说怎么定。如果不同 task 的 reward weights 不同，新 task 需要 reward engineering。

### 8.3 X-VLA / VLA-0 0% success 没解释

这两个 baseline 在 contact-rich 任务上完全失败，但 paper 没分析。可能原因：
- Fine-tune 数据太少，大 model 在小 data 上 overfit / catastrophic forgetting
- Action chunking / control frequency mismatch
- 预训练 distribution 跟 contact-rich manipulation 太远

这个 0% 让 HapticVLA 的对比看起来很漂亮，但 baseline 是否 properly fine-tuned 存疑。

### 8.4 Sim Data 的影响

Paper 加了 1000 sim episodes from Isaac Sim，但没 ablate sim data 的贡献。Sim tactile 跟 real tactile 的 sim-to-real gap 在 tactile modality 上可能很大（接触动力学很难精确仿真）。如果 sim data 主要贡献 vision+state 的多样性而 tactile 部分有 gap，可能影响 reward 计算。

### 8.5 Inference Latency 没数据

Paper 没给 inference latency 数字。SmolVLA 0.45B 在 Jetson Orin NX 上能跑多快？async vs sync chunk 的具体 timing？对 real-time control 这个数字很关键。

### 8.6 Distillation Beats Teacher 的真正原因没挖

86.7% (TD) > 81.7% (teacher async) 这个反超是 paper 最强结论，但没 ablate 原因。可能的 ablation：
- α=0 vs α=0.5 vs α=1：分别看 GT-only、blended、teacher-only
- Distillation 不用 blended，直接学 teacher target (α=1)
- 用 GT-only fine-tune from teacher init (α=0)

如果 α=1 (纯 teacher distillation) 跟 α=0.5 (blended) 接近，说明 teacher shaping 是主要因素；如果 α=0.5 显著更好，说明 blending 起决定作用。

### 8.7 Asymmetric Inference 的细节

"asynchronous SmolVLA inference" 是 SmolVLA 的设计特性，但 paper 里没解释 async 跟 sync 的具体 timing 差异。我的理解：async 是 chunk 之间 overlap execution（边执行边 inference 下一个 chunk），sync 是等 chunk 执行完再 inference。但具体实现细节缺失。

参考 SmolVLA: https://arxiv.org/abs/2506.01844

---

## 9. 直觉总结

把整篇 paper 压成几个直觉：

1. **Tactile 信息在 contact-rich task 上有 RL value**，因为 dense reward 比纯 success 信号 informative 得多。

2. **Tactile 信息在 inference 时不一定必要**——vision + state 已经包含大部分 material property 的可预测信息。tactile 的功能可以从 training-time verifier 转化为 inference-time implicit representation。

3. **Reward-weighted flow matching 是 offline RL on generative policy 的实用工具**——比 PPO 简单，比 DPO flexible，配合 anchor regularization 防 mode collapse。

4. **Negative examples (faulty episodes with object damage) 是关键训练 signal**——给 reward weighting 提供低 reward baseline，sharpen high-reward distribution。

5. **Blended target (α=0.5) 是 distillation 的稳定剂**——纯 teacher target 容易 amplify teacher bias，纯 GT 没 distillation 价值，blend 是 Pareto。

6. **Distillation 不止 transfer knowledge，还能 regularization**——TD 反超 teacher 的现象暗示 teacher 的 online tactile 依赖反而是 noise source。

7. **Hardware-aware ML 的未来方向**：让 hardware (tactile sensor) 只在 training 时出现，inference 用 minimal hardware。这跟 LLM 里的 test-time compute scaling 反向——这里是 **training-time compute/sensor scaling, inference-time minimal**。

8. **Safety reward 设计的 multi-axis decomposition 是个 good pattern**：把 "safe grasp" 拆成 force / peak / concentration / asymmetry / slip 五个 axis，每个独立 penalize。这个 pattern 可以推广到其他 contact-rich task (insertion, wiping, assembly)。

---

## 10. 跟你过去 work 的潜在联系

Karpathy 你在多处提过 "real world data is the long-term bottleneck"。这篇 paper 暗合这个观点——他们用小规模 real dataset (310 episodes) + sim data (1000) + 仔细的 reward design 来弥补 data scarcity。如果未来有 large-scale tactile dataset (类似 Open X-Embodiment 但 with tactile)，这套 SA-RWFM 应该能 scale。

另外，你讲过 VLA 是 "system 1" of robotics foundation model——把 perception→action 编码。这篇 paper 在说 "system 1 内部还藏着 physics reasoning"——通过 distillation 把 tactile 物理 encode 进 vision-language latent。这跟 your intuitions about latent world models 也有 connection——SA-RWFM 的 tactile reward 实际上在训练一个 implicit "what happens when I grip this" 的 forward model，student 通过 distillation 把这个 forward model 抽象成 vision-conditioned prior。

参考 Open X-Embodiment: https://robotics-transformer-x.github.io/
参考 RT-X paper: https://arxiv.org/abs/2310.08864

---

## 11. 可能的后续方向

1. **Cross-embodiment tactile distillation**：把 SO-101 学的 tactile-aware behavior distill 到其他 gripper（没 tactile hardware 的）。Paper 只在 SO-101 验证，跨 embodiment 是否 work 是大问题。

2. **Online tactile distillation**：当前是 fully offline。如果 student 能在 deployment 时偶尔 access tactile sensor (比如换 gripper finger 时)，做 partial online update，可能 capture novel object 的 tactile property。

3. **Tactile reward 的 learned version**：把 reward function 自己用 NN 学，避免手工设计 6 个 axis。Inverse RL 或 reward learning from preferences。

4. **Multi-modal distillation**：tactile 之外，audio (contact sound)、force (joint motor current) 也可以 distill 进 vision+state latent。一个 unified "physics distillation" pipeline。

5. **Failure mode analysis**：HapticVLA 在哪些 object 上失败？是 vision 看不出 material property 的吗（比如看起来硬其实软）？这个分析会揭示 distillation 的真正 limit。

6. **Larger model scaling**：SmolVLA 0.45B 是否足够？用 π₀ 5B 或 OpenVLA 7B 做 teacher，distill 到 SmolVLA student，是否更好？这种 "big teacher, small student" 的 cross-size distillation 在 LLM 里很成熟。

7. **Tactile-aware pretraining dataset**：类似 Open X-Embodiment 但 with tactile sensor readings。这能让 SA-RWFM 有大规模数据 base。

8. **Anchor reg 跟 KL penalty 的对比**：现在 anchor reg 是 L2 on parameters，理论分析跟 KL on action distribution 的关系。这跟 RLHF 里的 reference policy KL 是同源问题，但 FM policy 上还没有完整理论。

---

## 12. Final Take

HapticVLA 在 VLA + tactile 这个 niche 里走了一条 elegant 路径：**让硬件只在 training 时存在**。这套 training-time-modality, inference-time-modality-free 的范式跟 RLHF 里的 "训练 reward model，部署只 policy" 有结构上的相似——都是 "用额外监督信号烧 capability 进 base model，部署时丢掉监督"。

Reward 设计是 paper 最有信息量的部分——把 "safe grasp" 拆成 6 个独立 axis 的 dense penalty，配合 chunk return + episode return 的 mixed credit assignment，让 RL signal 在 FM policy 上稳定 transfer。

Distillation beats teacher 这个 result 很可能成为后续工作的引用点——它说明 generative policy 的 online conditioning 不一定是 optimal，offline distillation 反而能去掉 online noise。这跟 image generation 里 "classifier guidance (online) vs classifier-free guidance (offline distilled)" 的对比有同构性。

代码、model、dataset、digital twin 都开源（paper 说在 project website，但 link 没在 markdown 里给完整 URL，应该找 paper PDF 的 footnote）。

如果让我打分：方法 novelty 7/10（FD-VLA 思路相似，但 RL 部分有新意），实验 completeness 6/10（baseline 0% 没解释，scaling curve 缺），写作 clarity 7/10（公式完整，但 anchor reg schedule 和 async 细节缺失），intuition value 8/10（distillation>teacher 这个点很启发性）。

希望这个 walkthrough 帮你 build intuition，Karpathy。如果你想深挖某一块（比如 anchor reg schedule、async inference 具体实现、或 reward weights 的 sensitivity），我可以再展开。
