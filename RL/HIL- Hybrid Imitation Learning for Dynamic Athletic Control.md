---
source_pdf: HIL- Hybrid Imitation Learning for Dynamic Athletic Control.pdf
paper_sha256: 7bcdc9058769ce1bd976953b831d37ec3b219653178d7f5d994259f2697803fe
processed_at: '2026-08-04T23:50:01-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 HIL

好,Karpathy,我抛开学术腔,用最直白的方式给你讲这篇 paper 到底在干嘛。

---

## 一句话版本

这帮人想让虚拟小人跑 parkour,会翻墙、跳箱、滑铲,而且换个场景还能用。他们发现单独用任何一种现有方法都不行,于是把两种方法**同时**塞给一个 network 训,互相补盲点,效果出奇地好。

---

## 问题在哪

你想想 physics-based character animation 这个领域,过去十年有两派:

**Tracking 派**:就像让小孩描红字帖。你有一段 reference motion(比如一个人 vault 翻墙的 mocap),policy 每一帧都要尽量跟 reference 的 pose 对齐。好处是动作漂亮,坏处是描红太死板 — 换一面不同高度的墙,policy 就懵了,因为 reference 里没这面墙。

**AIL 派**(Adversarial Imitation Learning):就像让小孩看了一堆书法作品,然后自己写,有个老师(discriminator)判断写得像不像是那堆作品的风格。好处是灵活,写啥都行;坏处是小孩偷懒,发现反复写一个最简单的字最容易骗过老师,这就是 mode collapse。

Parkour 把这个矛盾放大了:
- 你需要 5 种不同 skill 串起来用(vault, jump, slide, wall-run...)
- 每种 skill 要适配不同 obstacle
- Tracking 死板,mode collapse 又让你只会一招

所以单用哪派都不行。

---

## 他们的 insight

核心 insight 特别简单:**别二选一,两个同时训**。

一个 environment 里 policy 学描红(tracking),旁边另一个 environment 里 policy 学自由发挥但被 discriminator 盯着(AIL)。两个 environment 共享同一个 policy network。

为什么这能 work?因为两种 learning 互相补对方的短板:

- Tracking 告诉 policy:"你得真的会每个 skill,不能糊弄"
- AIL 告诉 policy:"你得能在新场景下用这些 skill,不能只会背 reference"

单看好像理所当然,但真正难的是:**怎么让两种 mode 共享同一个 network?**

---

## 关键 trick:统一 observation

这是 paper 里最 clever 的设计。

传统 tracker 需要看两个东西才能工作:
1. 当前 character state
2. **phase variable**(你现在在 reference 的第几帧)或 **future target pose**(接下来要摆啥姿势)

问题来了 — AIL mode 下没有 reference,phase 和 target pose 都无定义。你给 network 一个 phase=0.3,它不知道你在说哪个 skill 的 0.3。

HIL 的解法:**扔掉 phase 和 target pose,换成 goal condition**。

具体来说,policy 只看三样东西:
- character 当前 state(joint positions, velocities, root height...)
- scene point cloud(周围障碍物的点云)
- target location(你要往哪走)

在 tracking mode,target location 是从 reference 未来 1-2 秒的 root 位置采的。在 AIL mode,target location 是随机采的附近 obstacle 位置加噪声。

**Intuition**: policy 不需要知道"我在 reference 第几帧",它只需要知道"我要往哪去"。从"我要往哪去"+当前 state,policy 能隐式推断出"我该做哪个 skill 的哪个阶段"。

这就像你开车不需要看 GPS 上的路线进度条,你看前方路况 + 目的地方向,自然知道该转弯还是直行。

Fig. 11 做了个实验验证这个 — 在 heading task 上,只用 goal condition 的 tracker 比 pose-conditioned tracker 稍慢但最终收敛到接近的 success rate。说明 goal condition 确实够用。

---

## Reward 长啥样

### Tracking mode reward

公式 2,说白了就是每一帧算 character 和 reference 的 pose 差异,越像越好:

$$r_t^{track} = w_p e^{-\alpha_p ||\hat{p}_t - p_t||} + w_r e^{-\alpha_r ||\hat{q}_t \ominus q_t||} + ... + w_e \sum_j ||\tau_j \dot{q}_j||$$

逐个说:
- $\hat{p}_t$ 是 reference 的 joint positions,$p_t$ 是 character 的,第一项比 position
- $\hat{q}_t$ 是 reference 的 joint rotations,$\ominus$ 是 quaternion 减法,第二项比 rotation
- 后面比 linear velocity $\hat{\dot{p}}_t$、angular velocity $\hat{\dot{q}}_t$、root height $\hat{h}_t$
- 最后一项 $w_e \sum_j ||\tau_j \dot{q}_j||$ 是 energy penalty,$\tau_j$ 是 joint $j$ 的 torque,$\dot{q}_j$ 是 angular velocity,乘起来惩罚 jittering — 你使劲抖动消耗大,扣分
- $w_{\{\cdot\}}$ 是各项 weight,$\alpha_{\{\cdot\}}$ 是 exponential scale,小 error 时快速饱和到 1

具体数字: $w_p=2.5, w_r=1.5, w_v=0.5, w_\omega=0.5, w_h=1, w_e=0.001$,$\alpha_p=1.5, \alpha_r=0.3, \alpha_v=0.12, \alpha_\omega=0.05, \alpha_h=20$

### AIL mode reward

AIL mode 的 reward 是 task reward + style reward 加权:

$$r_t = w^{task} r_t^{task} + w^{style} r_t^{style}$$

其中 $w^{task} = w^{style} = 0.5$。

**Style reward** 来自 discriminator:

$$r_t^{style} = -\log(1 - D(s_{t-n:t}, c_{t-n:t}))$$

- $s_{t-n:t}$ 是过去 $n$ 步($n=10$)的 character state history
- $c_{t-n:t}$ 是过去 $n$ 步的 scene point cloud history
- discriminator 判断"这段 motion 在这个 scene 下自不自然",越像 reference dataset 越高分

**这里有个关键创新**:传统 AMP 的 discriminator 只看 state,不看 scene。HIL 把 scene point cloud 也塞进 discriminator。效果惊人 — ablation(Table 2)显示去掉 scene info 后 skill accuracy 从 0.66 掉到 0.38。

**Intuition**: 没有 scene info,discriminator 觉得"走路"在哪都自然。加上 scene info,discriminator 学到"在墙前面走路不自然,应该 wall-run"。它把 affordance 编码进了 reward signal。

### Parkour task reward

公式 5:

$$r_t^{task} = w^{prog} \Big( ||\hat{p}_{t-1}^{root} - l_{t-1}||_2 - ||\hat{p}_t^{root} - l_t||_2 \Big) + w^{reach} r_t^{reach}$$

- $p_t^{root}$ 是 character pelvis position
- $l_t$ 是 target location
- 第一项:这一步比上一步离 target 近了多少,鼓励往前走
- $r_t^{reach}$ 是到达 target 的一次性 bonus
- 注意 $\Delta$ 被 clamp 到 $[0, 0.05]$,防止 policy 用"冲刺一招"吃 progress reward

### Heading task reward

公式 6:

$$r_t^{heading} = w^{vel} \exp\Big(-\alpha(v^* - \hat{d}_t^\top v_t)^2\Big) + w^{face}(\hat{f}_t^\top \hat{q}_t)$$

- $v_t$ 是 root velocity
- $\hat{d}_t$ 是 target heading direction(2D 单位向量)
- $\hat{f}_t$ 是 target facing direction
- $\hat{q}_t$ 是 character 当前 facing
- $v^* = 1.2$ 是 target speed
- 第一项让速度方向对齐 heading,第二项让身体朝向对齐 facing

---

## Network 架构

看 Fig. 2,policy 是个 Transformer:

```
character state ──[MLP]──> 1 token
scene point cloud ──[PointNet]──> N tokens  
target location ──[MLP]──> 1 token
                              │
                              ▼
              Transformer (2 layers, 2 heads, dim=256)
                              │
                              ▼
                          action a_t
```

为什么用 Transformer?因为 point cloud 是变长输入,而且需要 attention 机制让 character state "关注" 附近相关的 obstacle 点。PointNet 先把 60 个点编码成 feature,然后送进 Transformer 和其他 token 一起 attend。

Critic 是普通 MLP,**额外吃一个 task indicator $k_t \in \{0,1\}$**:
- $k_t = 0$:tracking mode
- $k_t = 1$:AIL mode

这个 flag 只给 critic,不给 policy。**为什么?** 因为 critic 要估 value function,两个 mode 的 reward 量级和语义完全不同,不告诉 critic 当前是哪个 mode 它就估不准。但 policy 不能看这个 flag — 部署时 policy 不知道自己在哪个 mode,必须 unified 处理。

Ablation 显示去掉 $k$ 后 critic loss 大 5 倍,task completion 也略降。

---

## PSI — 被低估的小 trick

Perturbed State Initialization 听起来是个小 trick,但 ablation 说它极其关键。

传统 RSI(Reference State Initialization)从 reference motion 采样初始 state。问题:skill A 结束时的 pose 可能跟 skill B 开始时的 pose 差很远,policy 学不会 A→B transition,于是 RL 优化器偷懒,只用一个简单 skill 走天下 — mode collapse。

PSI 的做法:RSI 采样后加 Gaussian noise。强制 policy 从"偏离 reference 的 state"开始,学会恢复到 reference 附近。

效果:
1. Policy 能处理 transition 之间的 state mismatch
2. Transition 变容易后,policy 不会因为"切换太难"退化为只用简单 skill

Ablation(Table 2):去掉 PSI,task completion 0.74 → 0.52,skill accuracy 0.66 → 0.50。

**Intuition**: PSI 是 tracking 和 AIL 之间的"桥梁"。它让 tracking 学到的 skill 更 robust,从而 AIL mode 下能自由组合这些 skill。

---

## 训练 schedule

两阶段:
1. **Stage 1**:4 billion samples,只训 tracking — 让 policy 先学会 individual skills
2. **Stage 2**:2 billion samples,一半 environment 跑 tracking,一半跑 AIL — 让 policy 学会 adapt 和 compose

为什么不一开始就 joint 训?因为冷启动时 policy 啥都不会,AIL 的 task reward 信号太弱,容易把 tracking 也带崩。

硬件:4×V100,4096 parallel environments,Isaac Gym,simulation 120Hz,policy 30Hz,PPO + GAE,$\gamma=0.99$,GAE $\tau=0.95$。

---

## 数据怎么来的

Parkour mocap 没有,他们从 YouTube 视频搞:

1. 用 **TRAM** 从单目视频估 3D pose + global trajectory
2. TRAM 对 ground 估计不准(parkour body orientation 复杂),用 **body orientation hints** 修正 — 看 view up-vector 和 body up-vector 在 image plane 的夹角
3. 手动 annotation tool 放 box geometry 当 obstacle
4. 用 **MaskedMimic tracker** refine — 消除 jittering, sliding, body-obstacle collision

最终:19 clips,30 秒,15 skills。数据量极小。

Heading task 用 ASE 的 sword-and-shield dataset,7 分钟 mocap。

---

## 结果怎么看

Table 1 主结果(带 noise 的 unseen obstacle):

| Method | Skill Acc | Track Err | Task Comp |
|---|---|---|---|
| Task Reward | 0.00 | 1.82 | 0.81 |
| AMP | 0.06 | 1.49 | 0.11 |
| ASE | 0.03 | 1.63 | 0.00 |
| MaskedMimic | 0.50 | 0.41 | 0.00 |
| Task Reward w/ ws | 0.15 | 0.54 | 0.86 |
| AMP w/ ws | 0.54 | 0.37 | 0.85 |
| **HIL** | **0.66** | **0.31** | 0.74 |

人话解读:

- **Task Reward**:纯 task optimization,task completion 0.81 看着高,但 skill accuracy 0.00 — 小人在地上爬过去,完全不自然。SMPL humanoid actuator 太强,RL exploit simulation 漏洞
- **AMP**:没 warm start,discriminator signal 太弱,policy 卡在 obstacle 前面不动,completion 0.11
- **ASE**:纯 discriminator 没 task guidance,policy 在 obstacle 前 stall,completion 0.00
- **MaskedMimic**:只训 tracking,新 transition 出现就崩,completion 0.00。第一个 obstacle 过了,第二个就摔
- **Task Reward w/ ws**:warm start 后会跑了,但动作还是不自然,skill accuracy 0.15
- **AMP w/ ws**:warm start 帮助大,completion 0.85,但 mode collapse 严重 — Fig. 5 显示对每个 obstacle 都用同一个 vault
- **HIL**:skill accuracy 最高(0.66),tracking error 最低(0.31),completion 0.74 够用。**完美 trade-off**

注意 HIL 的 task completion 不是最高,比 AMP w/ ws 低。这是合理的 — HIL 坚持"用对 skill",有些难 obstacle 用对 skill 反而比"用简单 skill 硬冲"难完成。但 motion quality 高一个档次。

---

## Ablation 讲什么

Table 2:

| Variant | Skill Acc | Track Err | Task Comp |
|---|---|---|---|
| w/o D | 0.53 | 0.36 | 0.62 |
| w/o PSI | 0.50 | 0.37 | 0.52 |
| D w/o scene info | 0.38 | 0.39 | 0.75 |
| w/o k | 0.52 | 0.40 | 0.73 |
| HIL | 0.66 | 0.31 | 0.74 |

人话:

- **w/o D**(去掉 discriminator):两个 mode 各练各的,没 synergy。Tracking 还行但 AIL 退化成纯 task reward
- **w/o PSI**:transition 学不会,completion 掉到 0.52。验证 PSI 是 anti-mode-collapse 关键
- **D w/o scene info**:skill accuracy 掉到 0.38,但 task completion 0.75 几乎不变。说明 scene-conditioned discriminator 主要管"选对 skill",不管"完成任务"
- **w/o k**:critic 区分不了 mode,value 估不准,影响 PPO 更新

---

## 评估方法的小聪明

Frame-by-frame 比 reference 不行,因为 policy 在 perturbed obstacle 上 timing 不会对齐(可能快可能慢)。

他们用 **DTW(Dynamic Time Warping)**:
- 给两条 motion sequence $a$ 和 $b$
- 构造 cost matrix $C(i,j) = ||a_i - b_j||$
- 找最优 warping path 对齐时间轴
- DTW distance = path 上累积 cost

Tracking error = DTW distance / 1000 归一化。

Skill accuracy 判定:对 generated clip 算与所有 15 个 reference clip 的 DTW,取最近的,看是不是 obstacle 对应的 expected skill。

---

## 几个我的直觉

### 1. 这本质上是 regularization

HIL 把 tracking 当 regularizer,把 AIL 当 exploration signal。RL 优化器在"忠实 reference"和"适应 task"之间找 Pareto optimal。这跟 LLM RLHF 里 "reference model KL penalty + task reward" 一模一样的思路 — 都防止 policy 漂离 reference distribution。区别是 RLHF 用 KL,这里用 tracking reward。

### 2. Goal condition 替代 phase 是真 insight

我一直觉得 phase variable 是 tracking 系统的"必要之恶"— 没它 tracker 学不会,有它又没法 generalize。HIL 用 goal condition 隐式表达 progression,一举两得。这个 idea 可能能推广到其他需要"reference-conditioned control"的场景,比如 robotic manipulation with demonstration。

### 3. Scene-conditioned discriminator 把 affordance 编码进 reward

传统 AMP 的 discriminator 是"这动作自不自然",HIL 的是"这动作在这场景自不自然"。这相当于让 reward signal 携带 affordance 信息。policy 通过最大化 style reward 间接学到"什么 skill 适合什么 obstacle"。

### 4. PSI 是 anti-mode-collapse 的隐藏武器

Mode collapse 的根因是 RL 优化器贪心 — 用一个 skill 解决所有任务最容易。PSI 让"切换 skill"的 cost 降低,减少 collapse 的 incentive。这个 insight 可能对其他 multi-skill RL 也有用,比如 robotics 里的 task switching。

### 5. 小数据能 work 说明 framework sample efficiency 好

30 秒 parkour 数据训出 15 skills 的 unified controller。说明 hybrid training 的 regularization 效果让 policy 不会 overfit 到狭窄的 reward signal,而是真正学到"skill 的结构"。

### 6. Hierarchy 不是唯一答案

最近 ASE/CALM/SkillMimic 这些 hierarchical 方法很流行,但 HIL 用 single-stage 反而更灵活。Hierarchy 的瓶颈是 low-level skill latent 不能 OOD generalize,因为 latent 是从 reference 学的。HIL 通过 AIL mode 显式让 policy 练习 OOD,可能这是 single-stage 反而胜出的原因。

---

## Limitations 作者自己承认的

1. SMPL humanoid actuator 过强,policy 能 exploit 出不自然高跳
2. Parkour 用 box geometry,real scene 复杂得多
3. 偶尔有 unnatural recovery 行为(绊倒后恢复)
4. 数据 scaling 卡在 paired motion-scene 获取难
5. 只在 sim 里训,sim-to-real 没碰

---

## 我觉得最 cool 的地方

1. **简单**:核心 idea 就一句话"两个 mode 同时训",implementation 也不复杂
2. **统一 observation**:goal condition 替代 phase 是真 insight,让 tracking 和 AIL 能 share network
3. **Scene-conditioned discriminator**:小改动大效果,affordance 编码进 reward
4. **PSI 作为 bridge**:不只是 robustness trick,更是 anti-mode-collapse 机制
5. **小数据**:30 秒训 15 skills,说明 framework 设计对了

---

## Links

- [HIL video](https://youtu.be/le4248gIMME)
- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html)
- [AMP](https://xbpeng.github.io/projects/AMP/index.html)
- [ASE](https://xbpeng.github.io/projects/ASE/index.html)
- [MaskedMimic](https://research.nvidia.com/labs/toronto-ai/maskedmimic/)
- [CALM](https://research.nvidia.com/labs/toronto-ai/calm/)
- [GAIL](https://arxiv.org/abs/1606.03476)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [TRAM](https://yufu-wang.github.io/tram/)
- [PointNet](https://arxiv.org/abs/1612.00593)
- [DTW book](https://link.springer.com/book/10.1007/978-3-540-74048-3)
- [ANYmal Parkour](https://doi.org/10.1126/scirobotics.adi7566)
- [Humanoid Parkour](https://arxiv.org/abs/2406.10759)
- [3D Diffusion Policy](https://3ddiffusionpolicy.github.io/)
- [Yu et al. body orientation](https://research.nvidia.com/labs/toronto-ai/human_dynamics/)

Karpathy,你直觉上觉得这个 idea 能不能推广到 robotics manipulation?我总觉得"reference demo + task adaptation"的 hybrid 在 manipulation 上也能 work,只是没人这么试。

---

# HIL: Hybrid Imitation Learning for Dynamic Athletic Control — 深入解读

Karpathy 你好，这篇 paper 来自 CMU + NVIDIA + SFU 团队（Jiashun Wang, Xue Bin Peng 等），投在 ACM TOG 2026。我把核心 idea、架构细节、reward 公式、实验数据都拆解一下，目的是让你 build intuition 关于为什么 hybrid 方案能解决 physics-based character animation 长期存在的 tension。

---

## 1. 这篇 paper 解决的核心 tension

Physics-based character animation 一直有两大家族方法，互相补不齐：

**Family A: Motion Tracking (DeepMimic 系 [Peng et al. 2018a](https://xbpeng.github.io/projects/DeepMimic/index.html))**

- 思路：用 reference motion clip $M$ 作为目标，policy 每帧 minimize 与 reference 的 pose/velocity 距离
- 输入需要 phase variable $\phi_t \in [0,1]$ 或 future target pose（比如 [MaskedMimic](https://research.nvidia.com/labs/toronto-ai/maskedmimic/) 用 target pose）
- 优点：motion quality 高，能精确 reproduce 各种 skill
- 缺点：无法 compose skills，遇到 unseen scene 几何就直接 fail，因为 phase/target pose 在新场景中无定义

**Family B: Adversarial Imitation Learning (AMP 系 [Peng et al. 2021](https://xbpeng.github.io/projects/AMP/index.html))**

- 思路：训一个 discriminator $D(s_{t-n:t})$ 区分 reference 和 policy 生成的 motion，policy 拿 $-\log(1-D)$ 作为 style reward
- 优点：分布级匹配，允许 deviation，能 generalize 到 task goal
- 缺点：mode collapse — RL 优化器会找最容易骗过 discriminator 的一个 skill，反复用，比如 AMP 在 parkour 任务上反复用同一个 vault 跨过所有 obstacles

**Parkour 任务把 tension 放大到极致**：
- 需要按 sequence 执行 5 个不同 dynamic skills（vault, jump, wall-run, slide, …）
- 需要每个 skill 适应 obstacle 几何的扰动
- Tracking 不行：phase 变量没法泛化到 perturbed obstacles
- AIL 不行：mode collapse 让所有 obstacle 用同一个 skill

---

## 2. HIL 的核心 insight

HIL 的关键 intuition 是：**这两种方法不是二选一，而是可以同时训练同一个 policy，让它们互相 regularize**。

更具体一点：

- **Motion tracking 提供 "正确性" 信号** — policy 必须能精确 reproduce reference 中的每个 skill，否则 tracking reward 低
- **AIL 提供 "适应性" 信号** — policy 必须能在新 scene 下产生与 reference 分布一致的运动，否则 discriminator reward 低
- 两者通过 **shared observation space** 在同一个 network 中互相传递 knowledge

为什么 naive 的 finetune（先 track 再用 task reward 训）不行？因为 task reward 会立刻 destroy tracking 学到的 motion quality — paper 里 "Task Reward w/ warm start" baseline 就是这样，task completion 0.86 但动作极其不自然（在地上爬过 obstacle）。原因是 SMPL humanoid actuator 太强，policy 会 exploit simulation 漏洞，跳过高 jump 这种现实中不可能的动作。

为什么 naive 的联合训练（不 share observation）也不行？因为 tracker 需要 phase/target pose 输入，AIL 模式下没有这些信息。两个 mode 输入分布不一致，shared policy 会混乱。

HIL 的关键 trick 是 **condition-only observation**：把 phase/target pose 替换成 goal condition（target location + scene point cloud + heading/facing）。这个 condition 在两个 mode 下都可用，且隐式包含 progression 信息。

---

## 3. Method 细节

### 3.1 Goal-Conditioned RL Formulation

标准 RL setting，policy $\pi(a_t | s_t, g_t)$，目标是最大化 discounted return:

$$J(\pi) = \mathbb{E}_{p(\tau|\pi)} \left[ \sum_{t=0}^{T-1} \gamma^t r_t \right] \tag{1}$$

变量解释：
- $\tau = \{s_0, a_0, r_0, ..., s_T\}$ 是 trajectory
- $\gamma \in [0,1]$ 是 discount factor（paper 里 $\gamma=0.99$）
- $g_t$ 是 goal condition，这是区别于标准 RL 的关键
- $p(\tau|\pi) = p(s_0) \prod_t p(s_{t+1}|s_t, a_t) \pi(a_t|s_t, g_t)$ 是 trajectory 似然

### 3.2 Motion Tracking Mode

**核心创新**：放弃 phase variable 和 target pose，用 goal condition 隐式表达 progression。

Tracking reward 形式（公式 2）:

$$r_t^{track} = w_p e^{-\alpha_p ||\hat{p}_t - p_t||} + w_r e^{-\alpha_r ||\hat{q}_t \ominus q_t||} + w_v e^{-\alpha_v ||\hat{\dot{p}}_t - \dot{p}_t||} + w_\omega e^{-\alpha_\omega ||\hat{\dot{q}}_t - \dot{q}_t||} + w_h e^{-\alpha_h ||\hat{h}_t - h_t||} + w_e \sum_j ||\tau_j \dot{q}_j||$$

每一项含义（变量和下标）：
- $\hat{p}_t, p_t$：reference 和 character 的 joint positions
- $\hat{q}_t, q_t$：reference 和 character 的 joint rotations（$\ominus$ 表示 quaternion difference）
- $\hat{\dot{p}}_t, \dot{p}_t$：reference 和 character 的 joint linear velocities
- $\hat{\dot{q}}_t, \dot{q}_t$：angular velocities
- $\hat{h}_t, h_t$：root height
- $w_{\{\cdot\}}, \alpha_{\{\cdot\}}$：每项的 weight 和 exponential scale
- 最后一项 $w_e \sum_j ||\tau_j \dot{q}_j||$ 是 energy penalty，$\tau_j$ 是 joint $j$ 的 torque，$\dot{q}_j$ 是 angular velocity，惩罚 jittering

具体超参（Appendix C.1）：
- $w_p = 2.5, w_r = 1.5, w_v = 0.5, w_\omega = 0.5, w_h = 1, w_e = 0.001$
- $\alpha_p = 1.5, \alpha_r = 0.3, \alpha_v = 0.12, \alpha_\omega = 0.05, \alpha_h = 20$
- 每个 reward term 都是 exponential kernel $e^{-\alpha ||\Delta||}$ 形式，这在 DeepMimic 就用，是为了让 reward 在小 error 时快速饱和到 1，大 error 时平滑衰减

**Intuition**: 目标位置 $l_t$ 在 tracking mode 是从 reference trajectory "1-2 秒未来 root 位置" 采样，相当于 phase variable 的 implicit 替代 — policy 通过看 "我还要往哪走" 来推断 "我在 reference 的哪个阶段"。

### 3.3 Adversarial Imitation Learning Mode

Discriminator loss（公式 3，标准 GAIL [Ho & Ermon 2016](https://arxiv.org/abs/1606.03476) + AMP gradient penalty）:

$$\min_D -\mathbb{E}_{d_M} \log D(s_{t-n:t}, c_{t-n:t}) - \mathbb{E}_{d_\pi} \log(1 - D(s_{t-n:t}, c_{t-n:t})) + w_{gp} \mathbb{E}_{d_M} ||\nabla_\phi D(\phi)||^2$$

变量解释：
- $s_{t-n:t}$ 是 $n$-step state history（paper 中 $n=10$），捕捉 temporal dynamics
- $c_{t-n:t}$ 是 scene point cloud 的 $n$-step history — **这是 HIL 的关键创新**
- $d_M, d_\pi$ 分别是 reference 数据集和 policy 生成的分布
- $w_{gp}$ 是 gradient penalty weight（防止 discriminator 太 sharp，标准 AMP 技巧）

**Scene-conditioned discriminator 的 intuition**: 普通 discriminator 只看 character state，会被骗 — 比如走路状态在平地上自然，但在 obstacle 前面是不对的。加上 scene point cloud 让 discriminator 学到 "什么 motion 适合什么 scene context"，相当于 implicit affordance modeling。

Style reward（与 AMP 一致）:
$$r_t^{style} = -\log(1 - D(s_{t-n:t}, c_{t-n:t}))$$

Total reward（公式 4）:
$$r_t = w^{task} r_t^{task} + w^{style} r_t^{style}$$

paper 中 $w^{task} = w^{style} = 0.5$。**关键 trick**: tracking mode 也加 style reward（同样的 weight），让两个 mode 的 reward signal 进一步 align。

### 3.4 Parkour Task 具体 reward

公式 5:
$$r_t^{task} = w^{prog} \Big( ||\hat{p}_{t-1}^{root} - l_{t-1}||_2 - ||\hat{p}_t^{root} - l_t||_2 \Big) + w^{reach} r_t^{reach}$$

- $p_t^{root}$：character pelvis position
- $l_t$：target location
- 第一项：progress reward — 当前到目标距离比上一步减少多少
- $r_t^{reach}$：到达 target 的一次性 bonus
- 在 Appendix C.1 中提到 $\Delta = ||p_{t-1}^{root} - g_{t-1}|| - ||p_t^{root} - g_t||$ 被 clamp 到 $[0, 0.05]$，防止 policy 通过 "冲刺完成" 来 exploit task reward

### 3.5 Heading Task reward

公式 6:
$$r_t^{heading} = w^{vel} \exp\Big(-\alpha(v^* - \hat{d}_t^\top v_t)^2\Big) + w^{face}(\hat{f}_t^\top \hat{q}_t)$$

- $v_t$：root velocity
- $\hat{d}_t$：target heading direction（2D ground plane unit vector）
- $\hat{f}_t$：target facing direction
- $\hat{q}_t$：character facing vector
- $v^* = 1.2$：target speed
- $w^{vel} = 0.7, w^{face} = 0.3, \alpha = 0.25$

### 3.6 Network Architecture

Policy 是 **Transformer-based**（Fig. 2）：

```
Input:
  character state s_t  ──[MLP 512→256]──> token
  scene point cloud c_t ──[PointNet 512→256]──> N tokens  
  target goal l_t ──[MLP 512→256]──> token
                                          │
                                          ▼
                          Transformer (2 layers, 2 heads, dim=256, FF=512)
                                          │
                                          ▼
                                    action a_t
```

- Transformer 用 2 层、2 个 attention head、latent dim 256、feed-forward dim 512
- PointNet 处理 60 个 closest points from scene（每个 obstacle 采样 15 个点，4 个 obstacle）
- Critic 是 MLP（1024, 512），**额外输入 task indicator $k_t \in \{0,1\}$** — 这是 privileged information，只给 critic 用于区分 tracking mode 和 AIL mode，policy 不接收
- Discriminator 是 MLP（1024, 512），输入是 state transition + scene cloud 的拼接

**为什么 critic 要 task indicator？** Ablation 显示去掉 $k$ 后 critic loss 大 5 倍。两个 mode 的 reward 量级和语义完全不同（tracking 是 dense pose error，AIL 是 task + style），同一个 critic 如果不知道当前是哪个 mode，value function 估计会严重混淆。这是 multi-task RL 中 critic conditioning 的一个小 trick。

**为什么 policy 不要 task indicator？** 因为部署时（test-time）policy 不知道当前在 "tracking" 还是 "adapt"，它应该 unified 处理所有场景。让 policy 不依赖这个 flag 强制它学一个 unified representation。

### 3.7 Action Space

PD controller actuation:
$$\tau_i = k_p \cdot (a_{t,i} - q_{t,i}) - k_d \cdot \dot{q}_{t,i}$$

- $a_{t,i}$ 是 policy 输出（joint target position）
- $q_{t,i}, \dot{q}_{t,i}$ 是当前 joint position 和 velocity
- $k_p, k_d$ 是 PD gains（manual 设定）
- Policy 输出 Gaussian $\pi(a|s,o) = \mathcal{N}(\mu_\pi, \Sigma_\pi)$，$\Sigma_\pi$ 固定，每个 diagonal 元素 $\sigma_\pi = 0.055$

### 3.8 Perturbed State Initialization (PSI)

传统 RSI（Reference State Initialization）从 reference motion 采样初始 state。问题：如果 skill A 的结束 state 与 skill B 的起始 state 不一致，policy 学不会 A→B 的 transition。

PSI 在 RSI 基础上加 Gaussian noise 到初始 state，强制 policy 学会从 "偏离 reference 的 state" 恢复到 reference 附近。这有两个效果：
1. **Robustness**: policy 能处理 transition 之间的 state mismatch
2. **Anti-mode-collapse**: 当 transition 容易学时，policy 不会因为 "transition 太难" 而退化为只用简单 skill

Ablation 数据（Table 2）证实 PSI 关键：去掉后 task completion 从 0.74 → 0.52。

### 3.9 Training Schedule

- Stage 1: 4 billion samples，只 train tracking mode（先让 policy 掌握 individual skills）
- Stage 2: 2 billion samples，并行 train tracking 和 AIL（一半 environments 跑 tracking，一半跑 AIL）

这个两阶段策略是为了避免冷启动困难 — 一开始 policy 什么都不会，AIL 的 task reward 信号太弱，joint 训练会让 tracking 也学不好。

4096 parallel environments on 4×V100, Isaac Gym simulator [Makoviychuk et al. 2021](https://developer.nvidia.com/isaac-gym), simulation 120Hz, policy 30Hz, PPO [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) with GAE [Schulman et al. 2016](https://arxiv.org/abs/1506.02438), $\gamma=0.99$, GAE $\tau=0.95$.

---

## 4. Dataset 构造 pipeline（值得细看）

Parkour mocap 数据稀缺，paper 用 YouTube 视频构造：

1. **TRAM** [Wang et al. 2024b](https://yufu-wang.github.io/tram/) — vision-based 3D pose estimator，从单目视频估计 global trajectory + SMPL pose
2. **Body orientation hints** [Yu et al. 2021](https://research.nvidia.com/labs/toronto-ai/human_dynamics/) — TRAM 对 ground estimation 不准（parkour 中 body orientation 复杂），用 view up-vector 和 body up-vector 在 image plane 的夹角修正 global body orientation
3. **手动 scene annotation tool** — annotator 放置 box geometry 对应 obstacle affordance
4. **Physics-based motion tracker** [Tessler et al. 2024](https://research.nvidia.com/labs/toronto-ai/maskedmimic/) refine — 消除 jittering, sliding, body-obstacle collision，得到 physical-plausible reference

最终：19 clips, 30 seconds total, 15 skills。这个数据量极其小，但效果惊艳 — 说明 method 的 sample efficiency 不错。

Heading task 用 sword-and-shield dataset from ASE [Peng et al. 2022](https://xbpeng.github.io/projects/ASE/index.html)，约 7 分钟 mocap，包含 advancing/retreating/turning/sword-swing 等行为。

---

## 5. 实验结果深入分析

### 5.1 主结果（Table 1）

| Method | Skill Acc ↑ | Track Err ↓ | Task Comp ↑ |
|---|---|---|---|
| Task Reward | 0.00 | 1.82 | 0.81 |
| AMP | 0.06 | 1.49 | 0.11 |
| ASE | 0.03 | 1.63 | 0.00 |
| MaskedMimic | 0.50 | 0.41 | 0.00 |
| Task Reward w/ ws | 0.15 | 0.54 | 0.86 |
| AMP w/ ws | 0.54 | 0.37 | 0.85 |
| **HIL (Ours)** | **0.66** | **0.31** | 0.74 |

关键观察：

1. **Task Reward w/ ws** 的 task completion 最高（0.86）但 skill accuracy 极低（0.15）— 完美的 RL exploitation 案例，policy 找 task reward 的捷径，动作完全不自然
2. **AMP w/ ws** 整体不错但 mode collapse 严重（Fig. 5 显示对每个 obstacle 都用同一个 vault）
3. **MaskedMimic** 完成率 0.00 — 因为它只在 reference-conditioned tracking 下训，无法 generalize 到 "skill A 完成后接 skill B" 这种新 transition
4. **HIL** 的 trade-off 极佳：skill accuracy 0.66（最高），tracking error 0.31（最低），task completion 0.74（够用）

### 5.2 Ablation（Table 2）

| Variant | Skill Acc | Track Err | Task Comp |
|---|---|---|---|
| w/o D | 0.53 | 0.36 | 0.62 |
| w/o PSI | 0.50 | 0.37 | 0.52 |
| D w/o scene info | 0.38 | 0.39 | 0.75 |
| w/o k | 0.52 | 0.40 | 0.73 |
| HIL | 0.66 | 0.31 | 0.74 |

Intuition 提炼：
- **w/o D**: 去掉 discriminator 后，两个 mode 优化目标完全独立，没有 synergy。Tracking 还能学但 AIL 退化成纯 task reward，skill accuracy 0.53 说明 skill 选择不够 intelligent
- **w/o PSI**: 关键！transition 学习变得困难，task completion 从 0.74 → 0.52。验证了 PSI 是 tracking 和 AIL 之间的 "bridge"
- **D w/o scene info**: skill accuracy 大降到 0.38，但 task completion 0.75 几乎不变。说明 scene-conditioned discriminator 主要影响 "选对 skill"，不影响 "完成任务"。这印证了 scene info 让 discriminator 学会 affordance
- **w/o k**: critic 区分不了两个 mode，value estimate 不准，影响 PPO 更新

### 5.3 Robustness 实验

- Noise level $\sigma=0.05$ 时 task completion >70%
- $\sigma=0.1$ 时仍 >50%
- 长序列：训练用 5 obstacles，测试用 20 obstacles + $\sigma=0.03$ noise，仍达 40% completion
- 这些数字说明 policy 学到的是 "generalizable skill composition" 而非 "memorize specific obstacle layout"

### 5.4 Skill coverage（Fig. 6）

用 DTW 距离把 generated motion 对应到最相似的 reference motion，统计 skill 使用频率。
- Task Reward 严重偏斜（少数 skill 主导）
- AMP w/ ws mode collapse 明显（vault 一招走天下）
- HIL 分布相对均匀，覆盖大部分 skills

### 5.5 Heading task 结果（Table 3）

| Method | Dir Score ↑ | Facing Score ↑ | Avg Return ↑ |
|---|---|---|---|
| AMP | 0.95 | 0.94 | 266 |
| ASE | 0.54 | 0.78 | 147 |
| MaskedMimic | 0.79 | 0.72 | 17 |
| HIL | 0.94 | **0.97** | 227 |

HIL 在 facing score 上最好，AMP 略高 return 但 motion 自然度差。ASE 借助 skill embedding 保留 reference 行为但 task performance 弱。MaskedMimic 完全 fail（return 17）说明纯 tracking 训练无法 generalize 到新 goal。

---

## 6. DTW 评估方法

值得讲一下，因为这是 paper 中一个 smart 设计：

Dynamic Time Warping [Müller 2007](https://link.springer.com/book/10.1007/978-3-540-74048-3) 算法：
- 给定两条 feature sequence $a = [a_1, ..., a_T]$ 和 $b = [b_1, ..., b_{T'}]$
- 构造 cost matrix $C(i,j) = ||a_i - b_j||$（Euclidean）
- 找 warping path $W = [(i_1, j_1), ..., (i_K, j_K)]$ 满足：
  - Boundary: $(1,1) \to (T, T')$
  - Continuity: 相邻 step
  - Monotonicity: 时间单调
- DTW distance = $\sum_{(i,j) \in W} C(i,j)$

为什么需要 DTW？因为 policy 在 perturbed obstacle 上执行 skill 时，timing 与 reference 不严格对齐（可能更快或更慢），frame-by-frame comparison 会高估 error。DTW 允许时间 warp 来对齐相同 motion phase。

Paper 中 tracking error 是 DTW distance / 1000 归一化。Skill accuracy 判定：对 generated clip 计算与所有 15 个 reference clip 的 DTW distance，取最近的，看是否匹配 obstacle 对应的 expected skill。

---

## 7. 我的几个 critical 观察 / open questions

### 7.1 Condition replaces phase — 真的 enough 吗？
Fig. 11 显示在 heading task 上，task-conditioned tracker 比 pose-conditioned tracker success rate 低一点（但接近）。这是个 surprising result — 仅靠 "目标方向 + 角色状态" 就能推断 "该执行 reference 哪个 phase"。我直觉是 heading task motion 数据多样，goal 已经隐含 phase。Parkour 上没做 pose-conditioned tracker baseline，可能的 concern 是 parkour skill 极 dynamic，goal-condition 是否真够 informative？这是 paper 留的一个 question。

### 7.2 为什么 scene info 进 discriminator 这么有效？
Ablation 显示 scene info 让 skill accuracy 从 0.38 → 0.66。我的 hypothesis 是：discriminator 学到了 "motion-state-scene" 三元组的 joint distribution。它不只是判 natural motion，更是判 "natural motion in this scene context"。这相当于把 affordance 编码到 discriminator 的 feature space 中，policy 通过 style reward 间接学到 affordance-aware skill selection。

### 7.3 PSI 的深层作用
PSI 不只是 robustness trick，更是 anti-mode-collapse 机制。Mode collapse 的根因是 RL 优化器贪心 — 用一个 skill 解决所有任务最简单。PSI 让 "切换 skill" 这个 action 的 cost 降低，从而减少 collapse 的 incentive。这是一个 generalizable insight，可能对其他 multi-skill RL 也有用。

### 7.4 数据规模的极限
Paper 用 30s parkour 数据训出 15 skills 的 unified controller，scaling 到 sword-and-shield 7 min 数据也 work。但 parkour 的 scaling 卡在 "paired motion-scene" 数据获取难。作者讨论未来用 3D reconstruction 自动化 pipeline。这与 [TRAM](https://yufu-wang.github.io/tram/) 这类工作的进一步发展密切相关。

### 7.5 与 sim-to-real 的关联
作者在 discussion 提到 extending to real humanoid systems 的方向。HIL 框架本身在 sim 里训，但 sim-to-real 的 challenge（perception noise, dynamics gap）仍未解决。最近 [ANYmal parkour](https://doi.org/10.1126/scirobotics.adi7566) 和 [Humanoid Parkour Learning](https://arxiv.org/abs/2406.10759) 都在 quadruped / humanoid real robot 上做 parkour，HIL 的 hybrid idea 可能对 real parkour 也有借鉴意义。

### 7.6 与 diffusion policy / 3D Diffusion Policy [Ze et al. 2024](https://diffusion-policy.cs.columbia.edu/) 的联系
Paper 用 PointNet 处理 scene point cloud，与 [3D Diffusion Policy](https://3ddiffusionpolicy.github.io/) 思路类似 — point cloud 作为 perception input 是 manipulation 和 locomotion 共享的设计 choice。Diffusion policy 在 manipulation 上已经 dominate，但 physics-based character control 用 Transformer + PointNet 还是一个合理选择。未来 hybrid 方案可能融入 diffusion 作为 high-level planner。

### 7.7 与 hierarchical methods 的对比
Paper 反复对比 ASE [Peng et al. 2022](https://xbpeng.github.io/projects/ASE/index.html)、CALM [Tessler et al. 2023](https://research.nvidia.com/labs/toronto-ai/calm/)、SkillMimic 等 hierarchical 方法。这些方法先 train low-level skill latent，再 train high-level goal-conditioned policy 选择 latent。HIL 不用 hierarchy，single-stage 联合训。**Intuition**: hierarchy 的瓶颈是 low-level skill latent 不能 out-of-distribution generalize（因为 latent 是从 reference 学的），而 HIL 通过 AIL mode 显式让 policy 练习 OOD generalization。这可能是 single-stage 反而比 hierarchy 更灵活的原因。

### 7.8 Reward hacking 和 SMPL 限制
作者承认 SMPL humanoid actuator 过强导致 task reward 可以 "exploit" 出不自然动作（高 jump、爬行过 obstacle）。这反映 simulation fidelity 对 RL policy 行为的深刻影响。改进 actuator model 或加入真实生物力学 constraint 可能是下一步。

---

## 8. 总结：HIL 的 take-away intuition

1. **Hybrid training > finetuning** — tracking 和 AIL 同时训能互相 regularize，避免 quality degradation 和 mode collapse
2. **Goal condition 可替代 phase/target pose** — 这让 unified observation space 成为可能，让 reference skill 能 transfer 到 unseen scene
3. **Scene-conditioned discriminator** — 把 affordance 编码到 reward signal 里，自然解决 "选对 skill" 问题
4. **PSI 是 anti-mode-collapse 的 hidden weapon** — 让 transition 易学，policy 不会被 "easy skill" 局部最优困住
5. **Critic task indicator** — multi-task RL 中简单的 trick 但 effective
6. **小数据也能 work** — 30 秒 parkour + 7 分钟 locomotion，说明 framework 对 data efficiency 友好

**最终直觉**: HIL 本质上把 motion tracking 当作 "regularizer"，把 AIL 当作 "exploration signal"，让 RL 优化器在 "忠实 reference" 和 "适应 task" 之间找 Pareto optimal。这和 LLM RLHF 中 "reference model KL penalty" + "task reward" 的思路异曲同工 — 一个用 KL 显式约束，一个用 tracking reward 显式约束，都防止 policy 漂离 reference distribution。

---

## 参考链接

- **HIL project page**: https://youtu.be/le4248gIMME (supplementary video)
- **DeepMimic** (Peng et al. 2018): https://xbpeng.github.io/projects/DeepMimic/index.html
- **AMP** (Peng et al. 2021): https://xbpeng.github.io/projects/AMP/index.html
- **ASE** (Peng et al. 2022): https://xbpeng.github.io/projects/ASE/index.html
- **MaskedMimic** (Tessler et al. 2024): https://research.nvidia.com/labs/toronto-ai/maskedmimic/
- **CALM** (Tessler et al. 2023): https://research.nvidia.com/labs/toronto-ai/calm/
- **GAIL** (Ho & Ermon 2016): https://arxiv.org/abs/1606.03476
- **PPO** (Schulman et al. 2017): https://arxiv.org/abs/1707.06347
- **GAE** (Schulman et al. 2016): https://arxiv.org/abs/1506.02438
- **Isaac Gym**: https://developer.nvidia.com/isaac-gym
- **TRAM** (Wang et al. 2024): https://yufu-wang.github.io/tram/
- **PointNet** (Qi et al. 2017): https://arxiv.org/abs/1612.00593
- **SMPL**: https://smpl.is.tue.mpg.de/
- **DTW book** (Müller 2007): https://link.springer.com/book/10.1007/978-3-540-74048-3
- **ANYmal Parkour** (Hoeller et al. 2024): https://doi.org/10.1126/scirobotics.adi7566
- **Humanoid Parkour Learning** (Zhuang et al. 2024): https://arxiv.org/abs/2406.10759
- **3D Diffusion Policy** (Ze et al. 2024): https://3ddiffusionpolicy.github.io/
- **Yu et al. body orientation hints**: https://research.nvidia.com/labs/toronto-ai/human_dynamics/

如果你 (Karpathy) 想深入某个细节（比如 transformer attention pattern 怎么 attend point cloud tokens，或 PSI 的 noise schedule 怎么设），我可以再展开讲。这个 paper 的设计 choices 里有不少 "看似简单但仔细想想很 clever" 的地方，特别值得 reverse engineering 来 build intuition 关于 multi-task physics-based RL 的 design pattern。
