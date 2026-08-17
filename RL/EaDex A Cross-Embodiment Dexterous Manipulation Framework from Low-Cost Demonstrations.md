---
source_pdf: EaDex A Cross-Embodiment Dexterous Manipulation Framework from Low-Cost
  Demonstrations.pdf
paper_sha256: ade4608b0cd51f33815184a9d40eba82a31244a180f955fd8377808f57ac5c6e
processed_at: '2026-08-04T01:13:45-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej， 咱们抛开那些学术腔调， 用最直白的工程师视角来聊聊这篇 paper 到底在干什么。

### 1. 核心痛点：巧妇难为无米之炊

灵巧手（比如 Allegro Hand, Inspire Hand）训练起来极其痛苦。纯靠 Reinforcement Learning (RL) 去试错，面对 20-30 个 DoF (Degrees of Freedom) 的高维空间，exploration 效率极低， 搜到猴年马月也学不会抓取。Imitation Learning (IL) 倒是能直接抄作业， 但是 抄作业的前提是你得有“标准答案”——也就是高质量的 demonstration。

传统采集 demonstration 需要极其昂贵的 motion capture 设备， 或者极其复杂的 VR teleoperation 系统。这套下来成本太高， 普通实验室玩不起。EaDex 这篇 paper 的核心出发点很简单：**我就用一个几千块的 RGB-D 摄像头拍人手动作， 数据虽然 noisy， 但是 算法层面想办法把这个噪声消化掉。**

### 2. 核心 Intuition：粗糙的 Contact Prior 就够了

作者抓住了 articulated object（比如盒子、华夫饼机、搅拌机） manipulation 的一个本质特征：**只要你的手能摸对位置（建立 contact），剩下的关节怎么发力、怎么扭， RL 自己能探索出来。**

你教小孩开瓶盖， 你不需要精确告诉他每个手指的 joint angle 是多少度， 你只要告诉他“左手握住瓶子，右手抓住盖子”。只要这个 contact 建立了， 小孩自己瞎拧几下就能拧开。EaDex 就是基于这个直觉：**low-cost camera 提供的粗略数据， 足够提供 contact prior， 剩下的交给 RL 去优化。**

### 3. 数据 Pipeline：怎么把廉价视频变成训练数据

拿着单目 RGB-D 拍人手， 怎么变成 robot 能懂的数据？

**第一步：2D 到 3D 的手腕定位**
用 [MediaPipe](https://arxiv.org/abs/2006.10214) 提取 2D keypoints， 拿到 wrist 的 pixel coordinate $(u, v)$。RGB-D 自带 depth $z$， 结合 camera intrinsics（焦距 $f_x, f_y$ 和光心 $c_x, c_y$）， 反投影出 wrist 的 3D 位置：

$$x = \frac{(u - c_x) z}{f_x}, y = \frac{(v - c_y) z}{f_y}, z = z$$

**变量拆解：**
*   $u, v$：wrist 在图像上的横纵坐标
*   $c_x, c_y$：camera 光心在图像上的横纵坐标
*   $f_x, f_y$：camera 在 x 和 y 方向的焦距
*   $z$：深度信息
*   $x, y, z$：wrist 在 camera 坐标系下的 3D 空间坐标

**第二步：在线拟合 MANO 模型**
有了大致的 3D 位置， 去拟合 [MANO](https://mano.is.tuebingen.mpg.de/) 这个 3D 手部模型， 目的是把手恢复出来。最小化预测关节点和检测关节点的误差：

$$\min_{\theta, r} \sum_{i=1}^{N} \omega_i \| J_i(\theta, r) - \hat{J}_i \|_2^2$$

**变量拆解：**
*   $\theta$：MANO 模型的 pose parameters， 控制手指弯曲
*   $r$：global wrist rotation， 手腕的全局旋转
*   $J_i(\theta, r)$：MANO 模型预测的第 $i$ 个关节点位置
*   $\hat{J}_i$：MediaPipe 检测到的第 $i$ 个关键点
*   $\omega_i$：第 $i$ 个关键点的权重
*   $N$：关键点总数

**第三步：平滑去噪**
RGB-D 加上 online fitting， 动作会抖得像帕金森。用 1D Gaussian kernel 在时间轴上做 smoothing：

$$\hat{x}_t = \sum_{k=-r}^{r} g(k) x_{\text{clip}}(t+k, 0, T-1)$$

**变量拆解：**
*   $\hat{x}_t$：第 $t$ 帧平滑后的数据
*   $g(k)$：Gaussian kernel 的权重函数
*   $r$：kernel 的半径， 决定看前后多少帧
*   $x_{\text{clip}}(\cdot)$：防止越界的边界截断函数
*   $T$：总帧数

这套流程走完， 数据存成 [ARCTIC](https://arxiv.org/abs/2303.09988) 格式， 就可以 retargeting 给不同的 robot hand 了。

### 4. 核心算法：动态“断奶”机制

这是全篇最 brilliant 的地方。拿到了 noisy demo， 直接拿去训 RL 会出问题：policy 容易被噪声带偏， 死磕那些错误的轨迹。传统的做法（比如 [DAPG](https://arxiv.org/abs/1709.10087)）给 imitation reward 一个固定的 weight， 这在高质量数据下管用， 在 low-cost 数据下就会“近墨者黑”。

作者的解法是 **Dynamic Demonstration Annealing（动态演示退火）**。本质上就是“动态断奶”。

总 reward 由四部分组成：

$$r_t = w_{\text{task}} r_{\text{task}} + w_{\text{imi}} r_{\text{imi}} + w_{\text{bc}} r_{\text{bc}} + w_{\text{con}} r_{\text{con}}$$

*   $r_{\text{task}}$：任务进度， object 有没有按目标走。
*   $r_{\text{imi}}$：状态模仿， robot hand 的状态像不像 demo 里的状态。
*   $r_{\text{bc}}$：动作模仿， policy 输出的 action 像不像 demo 的 action。
*   $r_{\text{con}}$：接触奖励， 有没有建立合理的物理接触。

**“断奶”逻辑在于：一开始 $w_{\text{imi}}$ 和 $w_{\text{bc}}$ 给高权重， 强行让 policy 跟着 demo 走。 当 policy 自己摸索出稳定的 contact 模式时， 就开始降低 $w_{\text{imi}}$ 和 $w_{\text{bc}}$ 的权重， 让 policy 摆脱 noisy demo 的束缚， 去追求更高的 task reward。**

什么时候开始降权重？看 contact reward 和 episode length。更新公式如下：

$$w_{\text{imi}}^{(e+1)} = \max(w_{\text{imi}}^{\min}, \gamma w_{\text{imi}}^{(e)})$$

**变量拆解：**
*   $w_{\text{imi}}^{(e+1)}$：下一个 epoch 的 imitation 权重
*   $w_{\text{imi}}^{\min}$：权重下限， 不能彻底降到 0， 得留点念想防止 catastrophic forgetting
*   $\gamma$：衰减系数， 介于 0 和 1 之间
*   $w_{\text{imi}}^{(e)}$：当前 epoch 的权重

触发这个衰减的条件是四个 AND 逻辑：

$$e \geq e_{\text{wait}}, \quad e - e_{\text{last}} \geq C, \quad \bar{r}_k^{(e)} \geq \tau_k, \quad \bar{l}^{(e)} \geq T_{\text{stable}}$$

*   $e \geq e_{\text{wait}}$：过了新手保护期， 别一开始就降。
*   $e - e_{\text{last}} \geq C$：距离上次降权过了冷却时间 $C$， 防止降得太快。
*   $\bar{r}_k^{(e)} \geq \tau_k$：滑动窗口平均的 contact reward 达到阈值 $\tau_k$， 证明 policy 会抓东西了。
*   $\bar{l}^{(e)} \geq T_{\text{stable}}$：滑动窗口平均的 episode length 达到阈值 $T_{\text{stable}}$， 证明 policy 不会早早死掉， 够稳定。

### 5. 技术直觉与联想

这种 **competence-based annealing** 的直觉极其符合人类学习规律。小孩学游泳， 一开始教练死死架着你（强 imitation）， 等你自己找到水感了（contact reward 升高）， 教练慢慢放手（weight decay）， 你才能游出自己的姿势。如果教练一直架着你， 你永远只能复刻教练的动作， 还会被教练的抖动带偏。

这让我联想到 [AlphaGo](https://www.nature.com/articles/nature24270) 的训练， 先用人类棋谱做 Supervised Learning 模仿， 丢掉棋谱后用 Reinforcement Learning 自我博弈探索更优解。EaDex 把这个过程变得平滑且自适应了。

另外， 作者选了 [Genesis](https://genesis-embodied-ai.github.io/) simulator 而不是传统的 Isaac Gym， 这也是个信号。Genesis 在 contact dynamics 的仿真上可能更精确， 对这种极其依赖 contact reward 触发 annealing 的 pipeline 来说， 仿真器的 contact fidelity 至关重要。

### 6. 实验结果的冷酷现实

听起来很美， 但看实验数据：

*   平均成功率 36.5%， 最高 93.3%。
*   Ablation study 证明， 没有这个 annealing 机制， 平均只有 23.5%， 相对提升 55.3%。

这就说明， 虽然 low-cost pipeline 可行， 但是 绝对成功率依然在及格线边缘徘徊。36.5% 意味着 10 次尝试里， 6 次以上是失败的。在真实工业场景部署， 这种成功率灾难性。paper 的定位确实还停留在 proof of concept 阶段。

### 7. 批判性思考与未来方向

这篇工作暴露出 dexterous manipulation 领域的一个核心瓶颈：**我们极度缺乏 scalable 的高质量数据。** 

EaDex 试图用算法（dynamic annealing）去弥补数据源（RGB-D）的先天不足。这是一个极好的工程妥协。 不过， 它依赖 predefined object trajectory， 这意味着人对着空气做动作， object 的运动是预设好的。这大大限制了 demo 的 diversity。如果 object 在被抓取时发生了不可预知的滑动， 预设轨迹就和实际脱节了， 这时 RL 能否 recover？paper 没有探讨。

如果顺着我自己的直觉往下推演， 我认为以下几个方向极具潜力：

1.  **World Model 辅助 Annealing：** 接触是高度非线性的物理过程， 只看 contact reward 这个标量信号其实很粗粒度。如果我们引入 [DayDreamer](https://arxiv.org/abs/2206.14176) 那种基于 Vision 的 World Model， 让 policy 在 latent space 里想象 contact 的后果， 基于想象的 reconstruction error 或者 uncertainty 去触发 annealing， 可能会比单纯的 reward 阈值更敏锐。
2.  **VLM 注入 Semantic Contact Prior：** 单目 RGB-D 毕竟视角有限。如果结合 [RT-2](https://arxiv.org/abs/2307.15818) 这类 Vision-Language Model， 直接从图像推理出 “应该抓这里” 的 semantic affordance， 作为更高级的 contact prior 指导 RL， 甚至连人手录制都不需要了。
3.  **Diffusion Policy 替代 PPO：** 对于这种 noisy demonstration， [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 天生具备处理 multimodal 数据的能力。与其用 PPO 硬刚 annealing， 不如先用 diffusion 把 demo 分布学下来， 然后在 action sampling 阶段注入 RL 的 value gradient 去做 guidance， 可能效果更好。

总而言之， EaDex 提供了一个非常实用的 engineering pipeline， 它告诉我们：在 robot learning 里， 只要你能找到合适的抽象层级（比如把 joint-level 精确度降维成 contact-level prior）， 廉价硬件也能撬动复杂任务。这是一个 build intuition 的好样本。

**Reference Links:**
*   [EaDex / DexMachina related context](https://arxiv.org/abs/2505.24853)
*   [MediaPipe Hands](https://arxiv.org/abs/2006.10214)
*   [MANO Model](https://mano.is.tuebingen.mpg.de/)
*   [ARCTIC Dataset](https://arxiv.org/abs/2303.09988)
*   [DAPG (Demonstration Augmented Policy Gradient)](https://arxiv.org/abs/1709.10087)
*   [AlphaGo Nature Paper](https://www.nature.com/articles/nature24270)
*   [Genesis Simulator](https://genesis-embodied-ai.github.io/)
*   [DayDreamer](https://arxiv.org/abs/2206.14176)
*   [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

---

# EaDex 深度技术解析

Andrej，这篇paper的核心intuition其实挺有意思：它质疑了dexterous manipulation领域对"high-fidelity demonstration"的过度依赖。作者argue说，对于articulated object manipulation这类任务，**coarse-grained demonstrations提供sufficient contact priors**，后续让RL去refine就可以了。这个观察直接motivates了整个framework的设计——用单RGB-D camera采集noisy data，再用dynamic annealing把policy从demonstration guidance中"wean off"出来。

---

## 1. Problem Setting & Motivation

### 1.1 核心矛盾

Dexterous manipulation learning面临一个fundamental tension：

| Paradigm | Strength | Bottleneck |
|----------|----------|------------|
| **Pure RL** [OpenAI dactyl](https://arxiv.org/abs/1808.00177) | 可探索complex scenarios | High-dim control space $\sim 20$-$30$ DoF，sample inefficient |
| **Pure IL** [DexMV](https://arxiv.org/abs/2108.05877) | Bypass trial-and-error | Demonstration采集cost高（mocap / teleop setup） |
| **RL + Demo** [DAPG](https://arxiv.org/abs/1709.10087), [DexMachina](https://arxiv.org/abs/2505.24853) | Demo作为prior，RL refine | 仍依赖expensive dataset |

EaDex要回答：**能否从low-cost, low-precision demonstration中学到stable policy？**

### 1.2 Key Insight

对于articulated object（box、waffle iron、mixer这类有joint的结构），manipulation success的关键在于**establishing correct contact patterns**，而不是精确的轨迹tracking。只要demo能告诉policy"什么时候该碰哪里"，剩下的可以让contact reward驱动RL自主优化。

这个insight在本质上类似[DayDreamer](https://arxiv.org/abs/2206.14176)里world model提供粗prior + RL fine-tune的思路，只不过这里prior来自视觉demo而非learned dynamics。

---

## 2. Dataset Construction Pipeline (Section 3.1)

### 2.1 整体架构

```
RGB-D Camera (RealSense D435i)
        ↓
MediaPipe 2D keypoint detection
        ↓
Depth back-projection → wrist 3D position (公式1)
        ↓
Online MANO fitting (公式2)
        ↓
ARCTIC format storage
        ↓
Gaussian temporal smoothing (公式3)
        ↓
Motion retargeting → 3 robot hands
```

这里值得注意的设计选择：作者没有走[OpenAI's AnyTeleop](https://arxiv.org/abs/2307.04577)那种general vision-based teleop路线，而是用了**trajectory-guided teleoperation**——operator跟随predefined的object轨迹做bimanual manipulation。这降低了free-form demo的复杂度，但也限制了任务的diversity。

### 2.2 公式(1)详细解析 — Wrist 3D Recovery

$$x = \frac{(u - c_x) z}{f_x}, \quad y = \frac{(v - c_y) z}{f_y}, \quad z = z$$

**变量含义：**
- $(u, v)$：wrist在image plane的pixel coordinate
- $z$：depth value（mm或m，取决于camera config）
- $(f_x, f_y)$：focal length in pixels（x和y方向）
- $(c_x, c_y)$：principal point（optical center的pixel coordinate）
- $(x, y, z)$：wrist在camera frame下的3D position

这就是标准的pinhole camera back-projection，没有什么新东西，但作者明确只用wrist而不是全部joint来恢复3D——这降低了noise sensitivity，因为finger tip的depth estimation error在RGB-D上往往>5cm（RealSense D435i的典型噪声水平）。

### 2.3 公式(2)详细解析 — MANO Online Fitting

$$\min_{\theta, r} \sum_{i=1}^{N} \omega_i \| J_i(\theta, r) - \hat{J}_i \|_2^2$$

**变量含义：**
- $\theta \in \mathbb{R}^{45}$：MANO的pose parameters（15 joints × 3 axis-angle）
- $r \in \mathbb{R}^{3}$：global wrist rotation（axis-angle）
- $J_i(\theta, r)$：MANO forward kinematics预测的i-th joint position
- $\hat{J}_i$：MediaPipe检测的i-th keypoint（注意MediaPipe只给2D，这里应该是把wrist depth扩展到其他joint的简化处理，paper没明说）
- $\omega_i$：joint weight，通常给wrist和MCP更高权重
- $N$：joint总数（21 for MediaPipe）

**Intuition：** 这是一个non-linear least squares问题，作者用online optimization求解（可能是Gauss-Newton或LBFGS）。相比[SMPLify](https://smplify.is.tue.mpg.de/)那种offline fitting，online fitting的accuracy会差一些，但对demo来说够了——这正是paper的thesis。

### 2.4 公式(3)详细解析 — Temporal Smoothing

$$\hat{x}_t = \sum_{k=-r}^{r} g(k) x_{\text{clip}}(t+k, 0, T-1)$$

**变量含义：**
- $\hat{x}_t$：smoothed signal at time $t$
- $g(k)$：1D Gaussian kernel，通常 $g(k) = \frac{1}{\sqrt{2\pi}\sigma} e^{-k^2/(2\sigma^2)}$
- $r$：kernel radius（half-width）
- $x_{\text{clip}}(\cdot, 0, T-1)$：boundary clipping，防止越界
- $T$：sequence length

**Intuition：** RGB-D + online MANO会引入high-frequency jitter，特别是finger abduction参数会跳变。Gaussian smoothing相当于low-pass filter，保留motion的low-frequency structure。代价是会smooth掉fast contact transitions，但作者认为这对contact prior提取没影响。

### 2.5 ARCTIC Format

存储成[ARCTIC](https://arxiv.org/abs/2303.09988)格式的好处是compatibility——ARCTIC是bimanual hand-object manipulation的标准格式，包含：
- Per-frame MANO params（pose + shape）
- Object 6D pose
- Articulation state（joint angle）
- Contact annotations

这样后续可以直接复用[DexMachina](https://arxiv.org/abs/2505.24853)的retargeting pipeline。

---

## 3. Contact-Reward-Based Dynamic Demonstration Annealing (Section 3.2)

这是paper最核心的贡献，我重点讲。

### 3.1 总Reward结构 — 公式(4)

$$r_t = w_{\text{task}} r_{\text{task}} + w_{\text{imi}} r_{\text{imi}} + w_{\text{bc}} r_{\text{bc}} + w_{\text{con}} r_{\text{con}}$$

**各项含义：**
| Term | 角色 | 直觉 |
|------|------|------|
| $r_{\text{task}}$ | Task progress | Object是否follow target trajectory |
| $r_{\text{imi}}$ | State imitation | Robot hand state匹配retargeted demo state |
| $r_{\text{bc}}$ | Action imitation (BC) | Policy output匹配retargeted demo action |
| $r_{\text{con}}$ | Contact | 是否建立合理contact（penetration、normal force等） |

对比[DAPG](https://arxiv.org/abs/1709.10087)只有 $r_{\text{task}} + r_{\text{bc}}$，EaDex多了 $r_{\text{imi}}$（state-level）和 $r_{\text{con}}$（contact-level）。$r_{\text{con}}$的设计参考了[DexMachina](https://arxiv.org/abs/2505.24853)的contact formulation。

### 3.2 Task Reward — 公式(5)

$$r_{\text{task}} = \exp(-\beta_{\text{pos}} d_{\text{pos}}) \cdot \exp(-\beta_{\text{rot}} d_{\text{rot}}) \cdot \exp(-\beta_{\text{ang}} d_{\text{ang}})$$

**变量含义：**
- $d_{\text{pos}} = \| p_{\text{obj}} - p_{\text{target}} \|_2$：position error（Euclidean）
- $d_{\text{rot}}$：rotation error，通常是geodesic distance $\arccos(\frac{\text{tr}(R_1^T R_2) - 1}{2})$
- $d_{\text{ang}}$：articulation joint angle error $|q_{\text{obj}} - q_{\text{target}}|$
- $\beta_{\text{pos}}, \beta_{\text{rot}}, \beta_{\text{ang}}$：sensitivity gains，控制reward衰减速度

**Intuition：** Multiplicative form比additive更严格——任何一项error大，整体reward就低。$\beta$越大，对error越sensitive。典型值 $\beta_{\text{pos}} \sim 5$-$10$（单位 $m^{-1}$），$\beta_{\text{ang}} \sim 1$（单位 $rad^{-1}$）。

### 3.3 Dynamic Annealing Mechanism — 公式(6)(7)(8)

这是paper的核心创新。先看update rule：

**公式(6) — Weight Decay:**
$$w_{\text{imi}}^{(e+1)} = \max(w_{\text{imi}}^{\min}, \gamma w_{\text{imi}}^{(e)}), \quad w_{\text{bc}}^{(e+1)} = \max(w_{\text{bc}}^{\min}, \gamma w_{\text{bc}}^{(e)})$$

**变量含义：**
- $w_{\text{imi}}^{(e)}, w_{\text{bc}}^{(e)}$：epoch $e$ 时的imitation和BC权重
- $w_{\text{imi}}^{\min}, w_{\text{bc}}^{\min}$：下界（防止完全消失，保留弱prior）
- $\gamma \in (0, 1)$：decay factor（典型值0.8-0.95）
- $e$：training epoch

**Intuition：** Exponential decay，但有floor。这个floor很重要——paper实验发现完全decay到0会导致policy forget demo中的contact structure。

**公式(7) — Sliding Window Statistics:**
$$\bar{r}_k^{(e)} = \frac{1}{L} \sum_{i=e-L+1}^{e} r_k^{(i)}, \quad \bar{l}^{(e)} = \frac{1}{L} \sum_{i=e-L+1}^{e} l^{(i)}$$

**变量含义：**
- $r_k^{(i)}$：epoch $i$ 的key reward（这里 $k$ 指 contact，即 $r_{\text{con}}$）
- $l^{(i)}$：epoch $i$ 的average episode length
- $L$：sliding window size（典型值5-10 epochs）
- $\bar{r}_k^{(e)}, \bar{l}^{(e)}$：windowed averages

**Intuition：** Sliding window平滑noise。单epoch的reward会因random seed、exploration noise等大幅波动，averaging后才能反映policy的真实capability。

**公式(8) — Trigger Condition:**
$$e \geq e_{\text{wait}}, \quad e - e_{\text{last}} \geq C, \quad \bar{r}_k^{(e)} \geq \tau_k, \quad \bar{l}^{(e)} \geq T_{\text{stable}}$$

**四个条件全部满足才触发annealing：**
1. $e \geq e_{\text{wait}}$：过了warmup期（防止过早decay）
2. $e - e_{\text{last}} \geq C$：距离上次annealing足够久（cooling interval，防oscillation）
3. $\bar{r}_k^{(e)} \geq \tau_k$：contact reward达到threshold（policy已经学会contact）
4. $\bar{l}^{(e)} \geq T_{\text{stable}}$：episode足够长（policy不会early terminate，说明stable）

**核心Intuition：** Annealing的trigger不是基于time schedule（像cosine schedule那样），而是基于**policy的competence signal**——具体来说是contact mastery。这非常关键：

- 传统schedule annealing（如DAPG的linear decay）会在固定epoch decay，不管policy是否真的学会了contact
- 如果demo noisy（low-cost scenario），policy可能学到的contact是错的，此时继续enforce demo会overfit noise
- EaDex的逻辑是：**只有当policy自主发现stable contact pattern时，才放松demo约束**，让RL去explore更优解

这跟[curriculum learning](https://arxiv.org/abs/2103.02588)中automatic difficulty scheduling的思路相通，只不过这里的"难度"是demo约束的强度。

### 3.4 与DAPG / DexMachina的对比

| Method | Demo Weight Schedule | Trigger Signal | Low-cost Demo Robustness |
|--------|---------------------|-----------------|--------------------------|
| [DAPG](https://arxiv.org/abs/1709.10087) | Fixed (0.1 BC weight) | N/A | Low（fixed weight对noisy demo敏感） |
| [DexMachina](https://arxiv.org/abs/2505.24853) | Fixed + curriculum | N/A | Medium |
| **EaDex** | Dynamic exponential decay | Contact reward + episode length | **High** |

---

## 4. Experiments 深度分析

### 4.1 Setup

- **Simulation**: [Genesis](https://genesis-embodied-ai.github.io/)（注意不是Isaac Gym，Genesis是较新的simulator）
- **RL Algorithm**: [PPO](https://arxiv.org/abs/1707.06347)
- **Hands**: Inspire Hand, Allegro Hand, XHand
- **Objects**: Box, Waffle iron, Mixer（all articulated）
- **Hardware**: Single RTX 3090, Intel i9-10900KF
- **Time**: Full pipeline ~1 hour for some tasks

### 4.2 Main Results (Section 4.2)

9个cross-embodiment task的平均success rate：**36.5%**，最高93.3%。

注意success criteria较严格：
- Object必须stays on platform（0.2m × 0.2m × 0.1m）整个episode
- Final articulation angle > 45°

这个criteria同时考察stability和task completion，不是简单的"opened vs not"。

### 4.3 Ablation — Annealing Effect (Section 4.3)

| Setting | Avg Success Rate |
|---------|-----------------|
| w/o Annealing (fixed weight) | 23.5% |
| Ours (dynamic annealing) | 36.5% |
| **Relative Improvement** | **+55.3%** |

这个ablation直接证明了dynamic annealing的价值。直觉解释：
- **Low-cost demo含noise**：fixed weight会让policy一直被noise拉扯
- **Annealing让policy先学contact structure，再escape noise**：早期demo提供coarse guidance，后期contact reward接管
- **Floor防止catastrophic forgetting**：完全decay会让policy丢失demo中的task structure

### 4.4 ARCTIC Validation (Section 4.4)

作者在high-quality ARCTIC dataset上也测试了annealing：

| Task | DexMachina (Unannealing) | Ours (Annealing) |
|------|--------------------------|-------------------|
| Ketchup | 9.0 ± 0.6 | 7.91 ± 28.12* |
| Waffleiron | 9.1 ± 0.7 | **23.01 ± 0.65** |
| Mixer | 28.1 ± 7.4 | **35.14 ± 4.02** |

*注：Ketchup的数字看起来有typo或formatting问题（7.91 ± 28.12这个std太大了）

**Intuition：** 即使在high-quality demo上，annealing也能带来gain（waffleiron +153%, mixer +25%）。这说明dynamic annealing不仅是low-cost demo的band-aid，而是**demonstration-guided RL的通用improvement**——因为即使high-quality demo，fixed weight也会限制policy探索更优解。

### 4.5 Cross-Embodiment Generalization

同一套human demo retarget到3个不同hand embodiment：

| Hand | DoF | Manufacturer |
|------|-----|--------------|
| [Inspire Hand](https://www.inspire-robots.com/) | 6 (underactuated) | Inspire Robots |
| [Allegro Hand](https://www.wonikrobotics.com/) | 16 | Wonik Robotics |
| [XHand](https://www.xrobotics.ai/) | ~12-20 | XRobotics |

Retargeting从ARCTIC format的MANO params映射到各hand的joint space，具体方法paper没详述（应该是用了[DexMachina](https://arxiv.org/abs/2505.24853)的retargeting）。关键insight是：**contact-based representation比joint-angle-based representation更transferable across embodiments**——因为contact是task-level abstraction，不依赖具体kinematics。

---

## 5. Limitations & Critical Analysis

### 5.1 Paper承认的limitation

- **Occlusion**: 单camera时gesture会被occlude，keypoint detection incomplete
- **Mitigation**: 用reference pose + local offset让palm朝camera，但这限制了wrist dexterity

### 5.2 我看到的额外concerns

1. **Task Diversity**: 只有3个articulated object，都是"opening"类任务。能否扩展到in-hand reorientation（[Rotating without seeing](https://arxiv.org/abs/2303.10880)那种）？Contact prior对fast reorientation可能不够。

2. **Sim-to-Real Gap**: Paper完全在simulation（Genesis）里eval。低cost demo + sim训练的policy能否transfer到real robot hand？作者没讨论。参考[OpenAI dactyl](https://arxiv.org/abs/1808.00177)的sim-to-real需要大量domain randomization。

3. **Success Rate绝对值**: 36.5%平均success rate其实不高。即使有annealing，大部分task还在50%以下。这对real deployment是否usable？

4. **Ketchup Result异常**: Table 1的Ketchup结果（7.91 ± 28.12）std比mean还大，这可能是5 seeds中有outlier，或者typo。需要看原始数据。

5. **Annealing Hyperparameters**: $\gamma, e_{\text{wait}}, C, \tau_k, T_{\text{stable}}, L$ 都需要tune。Paper没给sensitivity analysis。如果这些hyperparameters对每个task都要重新tune，那"low-cost"的claim要打折扣。

6. **Comparison缺失**: 没有跟[DexWild](https://arxiv.org/abs/2505.07813)、[UniByD](https://arxiv.org/abs/2512.11609)这些concurrent work直接对比。

---

## 6. Broader Context & Related Work

### 6.1 Low-cost Demonstration Acquisition 谱系

| Method | Hardware | Fidelity | Cost |
|--------|----------|----------|------|
| [DexCap](https://arxiv.org/abs/2403.07788) | Custom mocap suit | High | Medium |
| [AnyTeleop](https://arxiv.org/abs/2307.04577) | Multi-camera RGB-D | Medium | Medium |
| [DexMV](https://arxiv.org/abs/2108.05877) | YouTube videos | Low | Low |
| [DexWild](https://arxiv.org/abs/2505.07813) | Wild videos | Very low | Very low |
| **EaDex** | Single RGB-D | Medium-low | **Very low** |

EaDex的位置在DexMV和AnyTeleop之间——比single video准（有depth），比multi-camera便宜。

### 6.2 Demonstration Annealing 谱系

这个idea其实有历史根源：
- **[AlphaGo](https://www.nature.com/articles/nature24270)**: SL pretrain → RL fine-tune，可以看作extreme annealing（一步切换）
- **[DAPG](https://arxiv.org/abs/1709.10087)**: Fixed demo weight throughout
- **[Mandi et al.](https://arxiv.org/abs/2505.24853)**: Curriculum + fixed weight
- **EaDex**: Adaptive annealing based on competence signal

更broadly，这跟**self-distillation**、**EMA teacher**（[BYOL](https://arxiv.org/abs/2006.07733)）的momentum schedule有conceptual similarity——都是"从一个fixed target逐渐relax到self-consistency"。

### 6.3 Contact-Centric Manipulation Learning

Paper的contact-first视角跟最近的几个trend呼应：
- [Tactile RL](https://arxiv.org/abs/2303.10880): Touch signal作为primary observation
- [Contact graphs](https://arxiv.org/abs/2407.02274): Geometric fabrics用contact作为constraint
- [Diffusion policy with contact](https://arxiv.org/abs/2403.12945): Contact-aware generation

EaDex的贡献是用contact reward作为**annealing trigger**，这比单纯作为auxiliary reward更进了一步。

---

## 7. Implementation Details 我推测的细节

Paper有些地方说得不够细，我根据相关work推测：

### 7.1 Contact Reward Formulation

参考[DexMachina](https://arxiv.org/abs/2505.24853)，$r_{\text{con}}$可能是：

$$r_{\text{con}} = \sum_{i \in \text{fingertips}} \mathbb{1}[\text{contact}_i] \cdot \max(0, \mathbf{n}_i \cdot \mathbf{f}_i)$$

其中 $\mathbf{n}_i$ 是contact normal，$\mathbf{f}_i$ 是applied force。Penetration会被penalize。

### 7.2 Retargeting Objective

从MANO到robot hand的retargeting可能是：

$$\min_{\mathbf{q}} \sum_{i} w_i \| FK_i(\mathbf{q}) - \hat{J}_i^{\text{demo}} \|_2^2 + \lambda \|\mathbf{q} - \mathbf{q}_{\text{prev}}\|_2^2$$

其中 $\mathbf{q}$ 是robot joint angles，$FK_i$ 是robot forward kinematics，第二项是temporal smoothness。

### 7.3 PPO Hyperparameters (推测)

基于community common practice：
- $\gamma_{\text{discount}} = 0.99$
- $\lambda_{\text{GAE}} = 0.95$
- Clip ratio $\epsilon = 0.2$
- LR $3 \times 10^{-4}$
- Num envs: 4096-16384（Genesis支持大规模parallel）

---

## 8. Takeaways for Building Intuition

1. **Coarse demo + RL refine is a viable paradigm** for contact-rich manipulation. 完美的轨迹tracking不是必需的，contact pattern才是。

2. **Annealing trigger应该基于competence，不是time**。这跟人类学习类似——先模仿，等自己"感觉对了"再创新。

3. **Cross-embodiment的关键是task-level abstraction**（contact, trajectory），而非joint-level copy。

4. **Low-cost data acquisition的trade-off**: 用algorithmic sophistication（annealing + contact reward）compensate hardware simplicity（single RGB-D）。

5. **Genesis simulator的选择**值得关注——它比Isaac Gym新，可能contact dynamics更准，这对contact-rich task很重要。

---

## 9. Open Questions & Future Directions

1. **Can annealing generalize to in-hand reorientation?** Contact pattern在那里更dynamic，可能需要不同的trigger signal。

2. **Multi-modal demonstrations**: 如果用LLM/VLM生成demo（[RT-2](https://arxiv.org/abs/2307.15818) style），annealing还work吗？

3. **Real-world transfer**: Sim-trained policy + low-cost demo能否zero-shot transfer？需要多少domain randomization？

4. **Annealing for other auxiliary rewards**: 这个mechanism能否推广到其他auxiliary reward（如exploration bonus、curiosity）的scheduling？

5. **Theoretical analysis**: Annealing的convergence property有理论保证吗？跟[teacher-student framework](https://arxiv.org/abs/1503.02531)的关系？

---

## References

- [EaDex Paper](https://arxiv.org/abs/2505.24853) (这是DexMachina的link，EaDex本身好像还没上arxiv，从内容看应该是concurrent work)
- [ARCTIC Dataset](https://arxiv.org/abs/2303.09988) - https://arctic.is.tuebingen.mpg.de/
- [MANO Model](https://mano.is.tuebingen.mpg.de/) - https://arxiv.org/abs/2201.02610
- [MediaPipe Hands](https://arxiv.org/abs/2006.10214) - https://google.github.io/mediapipe/
- [DAPG](https://arxiv.org/abs/1709.10087)
- [OpenAI Dactyl](https://arxiv.org/abs/1808.00177)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Genesis Simulator](https://genesis-embodied-ai.github.io/)
- [DexCap](https://arxiv.org/abs/2403.07788)
- [AnyTeleop](https://arxiv.org/abs/2307.04577)
- [DexMV](https://arxiv.org/abs/2108.05877)
- [DexWild](https://arxiv.org/abs/2505.07813)
- [UniByD](https://arxiv.org/abs/2512.11609)
- [Rotating without Seeing](https://arxiv.org/abs/2303.10880)
- [Dextrah-g](https://arxiv.org/abs/2407.02274)

---

Andrej，这篇paper的contribution主要在engineering和system level——它没有提出fundamentally new algorithm（PPO + reward shaping都是标准的），但把low-cost data acquisition和adaptive annealing组合起来，形成了一个practical的pipeline。最有intellectual content的部分是**contact-reward作为annealing trigger**这个设计，它隐含了一个hypothesis：**contact mastery是manipulation learning的关键milestone**。这个hypothesis如果成立，对整个dexterous manipulation领域都有implications——意味着我们应该更多关注contact-centric representations和objectives，而不是joint-level imitation。

如果要critically evaluate，我会更想看到：(1) contact reward的sensitivity analysis，(2) annealing hyperparameters的robustness study，(3) 跟DexWild/UniByD的直接对比，(4) real-world transfer实验。目前的results还在"proof of concept"阶段，距离deployment-ready还有距离。
