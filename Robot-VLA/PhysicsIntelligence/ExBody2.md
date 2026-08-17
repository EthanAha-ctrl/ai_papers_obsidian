---
source_pdf: ExBody2.pdf
paper_sha256: 49c7fb908c25631025065d41a35cedfbb7d1c6f7ff146cb39803acf8a87e24fb
processed_at: '2026-08-04T06:08:31-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ExBody2

## 这篇 paper 到底在干嘛

咱就是说,你有个 Unitree G1 humanoid robot,你想让它学人跳舞、打拳、走路、蹲下,各种 expressive 动作。问题在哪?人 motion capture dataset (CMU mocap) 里有 1919 个动作 clip,但里面有大量 robot 根本做不到的 - 翻跟头、handstand、push-up、地上打滚。你硬让 robot 跟这些 reference motion 跟,它学不会,直接摆烂,连能做的动作都做不好。

前作 ExBody 的解法是 "我只跟 upper body",lower body 让 robot 自己保持平衡。但这样 robot 走起路来怪怪的,不像人。H2O、OmniH2O 是 "我全身 keypoint 都跟",但用 global frame tracking,robot 一旦某一帧跟丢了 reference keypoint,下一帧 keypoint 在前面老远,robot 拼命想追,追不上,reward 一直低,学个屁。

ExBody2 干了三件事,用大白话讲:

---

## Trick 1: 用 policy 自己当 filter,自动筛数据

### 问题

你说我手动把 infeasible 动作剔掉不就完了?1919 个 clip 你手动看一遍?而且你怎么判断 "feasible"?人的 cartwheel 对 G1 feasible 吗?你看着办吧,挺主观的。

### 解法

ExBody2 的思路特别巧妙 - 我先在**全部数据**上训一个 base policy $\pi_0$,这个 policy 肯定烂,因为数据里太多垃圾动作。但没关系,我拿这个烂 policy 去跑每个 motion clip,看它跟踪得怎么样。跟踪好的,说明这动作 robot 能做;跟踪烂的,大概率 infeasible。

具体怎么量?对每个 motion sequence $s$:

$$e(s) = \alpha \, E_{\mathrm{key}}(s) + \beta \, E_{\mathrm{dof}}(s)$$

- $E_{\mathrm{key}}(s)$: lower body 的 keypoint 位置误差 (米),抓那种整个下半身飞掉的极端 case
- $E_{\mathrm{dof}}(s)$: lower body 的 joint angle 误差 (弧度),抓关节级不可达
- $\alpha = 0.1, \beta = 0.9$: joint angle 权重大,因为 joint limit + motor torque 是最 hard 的物理约束,keypoint error 是 derived 量

然后排序,选一个 threshold $\tau$,保留 $e(s) \leq \tau$ 的 motion,得到 filtered dataset $\mathcal{D}_\tau$。在 $\mathcal{D}_\tau$ 上 resume train,得到 $\pi_\tau$。试 $\tau \in \{0.075, 0.10, 0.125, 0.15, 0.175\}$ 这五档,看哪个 $\pi_\tau$ 在**整个原 dataset** $\mathcal{D}$ 上表现最好。

### 结果

Figure 4 显示 $\tau = 0.15$ 是 sweet spot。太小 ($\tau = 0.075$) 数据太简单,robot 只学站立和慢走,见没见过的动作就懵;太大 ($\tau = 0.175$) 垃圾动作回来,训练被带坏。

最 striking 的 ablation 是 Table X: 在 250 个精挑的动作上训的 policy,在 1919 个动作全集上测,**居然比直接在 1919 个动作上训的 policy 还好**。这跟 LLM 里 LIMA paper 一个调调 - "1000 条干净数据胜过 100k 条噪声数据"。Humanoid 也是 data quality > data quantity。

### 直觉

这件事本质上是用 policy 当 "feasibility oracle"。你不需要手工判断 "这个动作 robot 能不能做",直接让 policy 跑一遍,做得出来的就是 feasible。这是一种 self-supervised data curation,跟 self-training、bootstrap 思路一脉相承。

---

## Trick 2: 先训 generalist,再 finetune specialist

### 问题

一个 policy 同时学 walk、dance、kungfu、punch,每个动作 reward 信号不同,policy gradient 互相打架,容易 mode collapse,啥都学不精。

### 解法

两阶段:
1. **Generalist**: 用 filtered data $\mathcal{D}_{\tau=0.15}$ 训一个大而广的 policy,啥都能凑合做
2. **Specialist**: 在 generalist 上 finetune,针对 dance、kungfu 这种 specific motion group,获得高精度

公式上没新东西,就是普通 finetune。但关键 ablation 在 Table IV,三个策略对比:
- **Generalist**: $\pi_{\tau=0.15}$ 原样
- **Specialist**: 在 generalist 基础上 finetune 到 dance 等子集
- **Scratch**: 从头训同样总 iterations,作为 fairness 对照

### 结果

$\mathcal{D}_{\mathrm{ACCAD}}$ (OOD, 完全没见过的数据) 上:
- Specialist $E_{\mathrm{mpjpe}} = 0.1402$
- Scratch $E_{\mathrm{mpjpe}} = 0.1609$
- Generalist $E_{\mathrm{mpjpe}} = 0.1716$

注意 **Generalist 居然比 Scratch 还差**?这有点反直觉。但仔细想就懂: Generalist 见过太多 motion,但对 ACCAD 没特化,啥都做不精;Scratch 从头在 dance 上训,但没 generalist prior 兜底,遇到没见过的 ACCAD 动作就崩。**Specialist 既继承 generalist 的 robustness,又有 task-specific precision,所以最好**。

这就是 LLM 里 pretrain → SFT 的同构:
- Pretrain (generalist) → 学一个 motion prior
- SFT (specialist) → 学 task-specific fine details
- 不 pretrain 直接 task train (scratch) → 在 OOD 上崩

### 留个 open question

Paper 自己 limitations section 说,多个 specialist 之间没法平滑切换。跳舞跳到一半想切到打拳,咋办?目前每个 specialist 是独立 policy,切换会跳变。这是个明显 follow-up 方向 - 用 MoE (Mixture of Experts) 加 router,或者用 latent code conditioning 单个 policy,跟 MultiGame Decision Transformer 那套思路。

---

## Trick 3: 把 global keypoint 拆成 local keypoint + velocity

### 问题

OmniH2O / H2O 用 global keypoint tracking,意思是 reference motion 的 keypoint 位置在 world frame 给定,robot 要在 world frame 跟上。

问题在于: 假设第 10 帧 robot 没跟上,reference keypoint 已经在第 11 帧的位置(更前方),robot 此刻还落在后面,下一帧它要 "追赶" 这个越来越远的 target。这是 trajectory tracking 的经典累积误差问题,robot 越追越累,最终摔倒。

直觉上,这种 global tracking 在控制论里是 open-loop 的味道 - reference trajectory 是预先给定,robot 跟不上就崩,没有 feedback 修正机制。

### 解法

ExBody2 把这件事拆成两半:

**(1) Velocity tracking 负责全局移动**

reward $r_{\mathrm{vel}} = \exp(-4.0|v_{\mathrm{ref}} - v|)$, weight 6.0
- $v_{\mathrm{ref}}$: reference motion 的 root linear velocity (m/s)
- $v$: robot 当前 root linear velocity

还有 velocity direction: $\exp(-4.0 \cos(v_{\mathrm{ref}}, v))$, weight 6.0,保证方向也对。

这个 reward 只看速度,不看绝对位置。robot 走错一厘米没事,只要速度跟上就行。这是 **"轨迹跟丢但速度对就 OK"** 的思路。

**(2) Local keypoint tracking 负责局部表达**

把 reference keypoint 用 robot 当前 root pose 转换到 local frame:

$$p_t^{\mathrm{local}} = T(p_t^{\mathrm{ref}}; x_t^{\mathrm{robot}})$$

- $T(\cdot; x_t^{\mathrm{robot}})$: 用 robot 当前 root pose $x_t^{\mathrm{robot}}$ 做 coordinate transform
- $p_t^{\mathrm{ref}}$: world frame 下的 reference keypoint
- $p_t^{\mathrm{local}}$: 转换后,相对于 robot 当前 root 的 keypoint

然后 reward $r_{\mathrm{kp}} = \exp(-|p_{\mathrm{ref}}^{\mathrm{local}} - p|)$, weight 2.0

这意味着: 哪怕 robot 整体 drift 了 1 米,只要相对姿势跟 reference 一致,就算赢。这就把 "全局位置跟丢" 和 "局部姿势模仿" 解耦了。

训练时还允许 small global drift,周期性 reset 到 robot frame,类似 PD 控制器 integral wind-up reset。部署时严格用 local。

### Reward 表细看 (Table I)

| Term | Expression | Weight |
|---|---|---|
| DoF Position | $\exp(-0.7|q_{\mathrm{ref}} - q|)$ | 3.0 |
| Keypoint Position | $\exp(-|p_{\mathrm{ref}} - p|)$ | 2.0 |
| Linear Velocity | $\exp(-4.0|v_{\mathrm{ref}} - v|)$ | 6.0 |
| Velocity Direction | $\exp(-4.0 \cos(v_{\mathrm{ref}}, v))$ | 6.0 |
| Roll & Pitch | $\exp(-|\Omega_{\mathrm{ref}}^{\phi\theta} - \Omega^{\phi\theta}|)$ | 1.0 |
| Yaw | $\exp(-|\Delta y|)$ | 1.0 |

看 weight 分布: velocity (6+6=12) 远高于 keypoint (2) 和 DoF (3)。**velocity 是 dominant signal**。这反映 decoupling 思路 - velocity 用来做全局 navigation, keypoint/DoF 用来做局部 expressiveness。

$\Omega^{\phi\theta}$ 上标 $\phi\theta$ 表示 roll (绕 x 轴) 和 pitch (绕 y 轴) 的角速度, $\Delta y$ 是 yaw (绕 z 轴) 的偏差。这俩 weight 都 1.0,低,因为 root orientation 已经在 velocity tracking 里间接约束了。

### Regularization reward (Table IX) 也值得说

- `DoF position limits`: $\mathbb{1}(d_t \notin [q_{\min}, q_{\max}])$, weight -10 - joint 超限强惩罚
- `Feet air time`: $T_{\mathrm{air}} - 0.5$, weight 10 - 鼓励抬脚 0.5 秒,防拖步,引导正常 step pattern
- `Stumble`: $\mathbb{1}(F_{\mathrm{feet}}^x > 5 \times F_{\mathrm{feet}}^z)$, weight -2 - 脚横向力大于 5 倍垂直力时惩罚,防绊倒
- `Action rate`: $\|a_t - a_{t-1}\|^2$, weight -0.1 - 平滑动作,防 jitter

这些 regularization 都是 sim-to-real 经验堆出来的 - 在 sim 里这些不做,real 上必崩。

### 为什么这个解法 work

直觉: 这套 decoupling 等于把一个 hard constrained trajectory tracking 问题,拆成两个 soft 问题:
- 全局移动用 velocity (软约束,允许 drift)
- 局部姿势用 local keypoint (软约束,允许位置漂移)

每个问题 reward signal 独立,policy gradient 不打架。这跟 LocoMan、UMI on Legs 这些工作里 locomotion-manipulation decouple 的思路一致 - whole-body control 普遍倾向 hierarchical / decoupled 设计,因为 monolithic reward landscape 太崎岖。

---

## Pipeline 完整图景

把整个 ExBody2 pipeline 在脑子里画一遍:

```
CMU/AMASS raw motion (1919 clips)
        │
        ▼
[Retarget to G1 morphology] ── 把人骨架映射到 23-DoF G1
        │
        ▼
Train base policy π₀ (PPO + privileged info)
        │
        ▼
Evaluate π₀ on each clip → e(s) = α·E_key(s) + β·E_dof(s)
        │
        ▼
Rank motions, pick threshold τ (greedy search over {0.075, 0.10, 0.125, 0.15, 0.175})
        │
        ▼
Filtered dataset D_τ (τ=0.15 best)
        │
        ▼
Resume train → Generalist policy π_τ* (teacher with privileged)
        │
        ▼
DAgger distill → Student policy (no privileged, with H=10 history)
        │
        ▼
[Deploy on G1]  or  [Finetune to specialist for dance/kungfu/...]
```

### Teacher vs Student 细节

**Teacher** (在 sim 里训练,有 privileged info):
- 输入: $p_t$ (privileged, 62 维) + $o_t$ (proprioceptive, 75 维) + $g_t$ (tracking target, 69 维)
- $p_t$ 包含真实 root velocity、真实 body link positions、friction、motor strength 这些 sim 才能拿到的 ground truth
- 输出: $\hat{a}_t \in \mathbb{R}^{23}$ (joint PD target)
- 算法: PPO, MLP [512, 256, 128]

**Student** (deploy 用,没 privileged):
- 输入: $o_{t-H:t}$ (history of proprioceptive, $H=10$) + $g_t$
- 用 history 推断 privileged info (隐式 system identification)
- 输出: $a_t \in \mathbb{R}^{23}$
- 训练: DAgger - student rollout,teacher 提供 oracle action label,MSE loss $\|a_t - \hat{a}_t\|^2$
- MLP [1024, 1024, 512] - 比 teacher 大,因为要 encode history

Table XI 的 ablation:
- History $H=0$: $E_{\mathrm{vel}}$ 从 0.2930 飙到 0.4151 (差太多)
- History $H=10$: best
- History $H=100$: 又变差,过长 history 难拟合

DAgger 去掉: $E_{\mathrm{vel}}$ 从 0.2930 → 0.4195,$E_{\mathrm{mpjpe}}$ 从 0.1079 → 0.1496,全线崩盘。

直觉: DAgger 解决 covariate shift。普通 BC (behavior cloning) 只在 expert trajectory 上学,但 student deploy 时自己产生的 state distribution 跟 expert 不一样,会越走越偏。DAgger 在 student 自己 trajectory 上 query teacher label,闭环 fix 这个 mismatch。这是模仿学习里的经典 lesson,ExBody2 老老实实用上。

---

## 真实世界结果

Table III - Unitree G1 real-world:

| Method | $E_{\mathrm{mpjpe}}$ | $E_{\mathrm{mpjpe}}^{\mathrm{upper}}$ | $E_{\mathrm{mpjpe}}^{\mathrm{lower}}$ |
|---|---|---|---|
| ExBody | 0.2178 | 0.1223 | 0.3239 |
| ExBody† (full body) | 0.1465 | 0.1314 | 0.1672 |
| OmniH2O* | 0.1396 | 0.1273 | 0.1533 |
| ExBody2 | **0.1074** | **0.1092** | **0.1054** |

最 striking: ExBody 的 lower body error 是 0.3239,ExBody2 是 0.1054,**3 倍提升**。这主要归功于 lower body tracking + motion-velocity decouple - lower body 终于跟得上了。

部署细节:
- Compute: Jetson Orin NX (onboard)
- Policy inference: 50 Hz
- Low-level PD control: 500 Hz
- Comm delay: 18-30 ms (用 LCM middleware)
- 这个 50 Hz / 500 Hz 分层是 humanoid 标准做法,RL policy 给 PD target,PD 控制器高频追踪

---

## 这套思路跟 LLM 的同构性

我感觉这是这篇 paper 最值得思考的点:

| Humanoid (ExBody2) | LLM |
|---|---|
| Base policy π₀ on full data | Pretraining on internet data |
| Filter by base policy feasibility score | Data cleaning / quality filtering |
| Generalist π_τ* on filtered data | Pretrained foundation model |
| Specialist finetune on dance/kungfu | SFT on task data |
| DAgger distill teacher → student | Distill large model to small |
| Privileged info in sim | Teacher forcing in training |
| Real-world deploy without privileged | Autoregressive inference |
| Multi-specialist MoE (open) | Mixture of Experts |
| Cross-embodiment generalist | Cross-modal foundation model |

这套同构不是巧合 - 任何 "学习一个通用 prior,再 task-specific 特化" 的范式都会收敛到类似结构。humanoid 正在走 LLM 三年前走过的路。

---

## 几个值得吐槽的点

1. **Threshold τ* 跨 dataset 没 verify**: 论文说 "exhibits generalizability to other datasets",但实验只在 CMU + ACCAD 上做。τ* = 0.15 在 H1、Berkeley Humanoid、Apollo 上还 work 吗?没说。

2. **Specialist runtime 切换没解决**: 论文 limitations 自己承认。deployment 时 robot 怎么知道现在该用 dance specialist 还是 kungfu specialist?作者说用 motion classifier / action recognition,但没实现。这是个明显 follow-up。

3. **Reward weight 怎么定的**: Table I 的 weight (3.0, 2.0, 6.0, 6.0, 1.0, 1.0) 是怎么调出来的?没 ablation。看起来 manually tuned。可以想象用 reward learning (RLHF 风格) 学 reward weight。

4. **Generalist 在 hard 数据上提升不大**: Table IV 的 $\mathcal{D}_{\mathrm{hard}}$ 上,Specialist 0.1047 vs Generalist 0.1181,提升约 13%。比 $\mathcal{D}_{\mathrm{easy}}$ 上的提升 (8%) 大,但比预期小。可能 generalist prior 在 hard motion 上本身不强。

5. **没跟 HumanPlus、Hover、HOMIE 比**: baselines 只有 ExBody 系和 OmniH2O*,这几个最近的工作没比。

---

## 一句话总结

ExBody2 = **用 policy 当 data filter** + **generalist-specialist finetune** + **velocity-keypoint decouple**。三件事都不复杂,但组合起来在 Unitree G1 上把 whole-body tracking 做到了 SOTA。这套 recipe 跟 LLM 的 pretrain → SFT 高度同构,暗示 humanoid control 的 scaling path 可能就在这里 - clean data + generalist prior + task specialist + good control decomposition。

接下来值得 follow: MoE humanoid policy、diffusion policy 替换 MLP PD target、vision-conditioned ExBody2、text-to-motion → ExBody2 整条 pipeline、cross-embodiment humanoid foundation model。

主页看 demo video 才有感觉: https://exbody2.github.io

---

# ExBody2 深度解读: Humanoid Whole-Body Control 的范式跃迁

## 一、宏观框架直觉: 为什么这篇 paper 重要

ExBody2 是 UC San Diego (Xiaolong Wang 组, Xuxin Cheng) 与 UC Berkeley、MIT 合作的 ExBody 续作, 针对的是 Unitree G1 平台上 expressive + robust 的 whole-body tracking 问题。它的核心 insight 是三段式 pipeline:

1. **Automated data curation** - 用一个 base policy 当 "filter", 自动剔除 infeasible motions (主要是 lower body 极端动作)
2. **Generalist → Specialist finetune** - 先训一个大而广的 generalist, 再 finetune 出 dancer / kungfu 等 specialist
3. **Motion-velocity decoupling** - 把 global keypoint tracking 转成 local frame + velocity tracking, 避免累积 drift

直觉上, 这套设计思路和 LLM 的 pretrain → SFT 范式高度同构: base policy 像 pretraining 学一个 motion prior, data filtering 像 data cleaning, specialist finetune 像 task-specific SFT。这种类比很重要, 它暗示了 humanoid control 的 scaling 路径可能跟 LLM 类似。

项目主页: https://exbody2.github.io
arXiv 前作 ExBody: https://arxiv.org/abs/2402.16796

---

## 二、Feasibility-Diversity Principle: 数据筛选的核心 insight

### 2.1 现象与 motivation

Human motion dataset (如 CMU Mocap, AMASS) 包含大量机器人做不到的动作: 翻滚、handstand、push-up、somersault。直接拿这些训 policy, reward 信号会塌掉, policy 学会 "give up", 导致连 feasible 的动作也跟踪不好。ExBody 初代用 language label filter (e.g. "dance"), 但 "dance" 这个词下也可能有不可行的极端动作; H2O / OmniH2O 用 SMPL avatar 模拟, 但 SMPL avatar 的 capability 跟 Unitree G1 的 motor + DOF 限制 mismatch。

ExBody2 提出的核心 principle 是: **lower body 严格 feasible, upper body 尽量 diverse**。直觉是 lower body 决定 CoM (center of mass) dynamics 和 support polygon, 一旦不可行整个 trajectory 就废了; upper body 主要影响表达性, 即使 noisy 一点也不会让 robot fall。

### 2.2 Filtering 公式逐项解析

对每个 motion sequence $s \in \mathcal{D}$, 训完 base policy $\pi_0$ 后, 计算:

$$
e(s) = \alpha \, E_{\mathrm{key}}(s) + \beta \, E_{\mathrm{dof}}(s)
$$

变量解释:
- $s$: 一个 motion sequence (e.g. CMU dataset 里的一个 clip)
- $E_{\mathrm{key}}(s)$: lower body 的 mean keybody position error (单位 meter), 主要防止翻转、滚动这种极端 lower body 偏离
- $E_{\mathrm{dof}}(s)$: lower body 的 mean joint-angle tracking error (单位 rad), 度量 joint level 是否可达
- $\alpha = 0.1, \beta = 0.9$: weights, 明显偏向 joint-angle error。直觉是 joint 角度是物理上最 hard 的约束 (joint limit + motor torque), keypoint error 是 derived 量, 把 weight 重压在 dof 上更稳定

然后对整个 dataset 排序得到 empirical distribution $P(e)$, 目标是找 threshold $\tau^*$:

$$
\tau^* = \arg\max_{\tau} \mathbb{E}_{s \in \mathcal{D}}[\mathrm{Performance}(\pi_\tau, s)]
$$

其中 $\pi_\tau$ 是在 filtered subset $\mathcal{D}_\tau = \{s \in \mathcal{D} \mid e(s) \leq \tau\}$ 上 train 的 policy, 但 evaluation 是在 **full dataset** $\mathcal{D}$ 上做的 - 这一点很关键, 评估目标是 generalization, 不是 in-distribution fit。

实际算法: 用 greedy search, 把 $P(e)$ 分成 evenly spaced intervals (论文里选 $\tau \in \{0.075, 0.10, 0.125, 0.15, 0.175\}$, 见 Figure 8 的 empirical CDF), 对每个 $\tau$ resume train base policy 得到 $\pi_\tau$, 选 best。

### 2.3 Ablation 验证 Feasibility-Diversity Principle

Table X 是 Principle 的 ablation:
- $\mathcal{D}_{50}$: 50 个 fundamental 动作 (站立、走路), extreme feasible
- $\mathcal{D}_{250}$: 扩展到 250, 加入 upper limb variation + moderate dynamic lower
- $\mathcal{D}_{CMU}$: 全 1919 sequence, 包含极端动作

在 $\mathcal{D}_{CMU}$ 上 eval, $\mathcal{D}_{250}$-trained policy 居然**胜过** $\mathcal{D}_{CMU}$-trained policy:
- $E_{\mathrm{vel}}$: 0.2834 vs 0.2622 ($\mathcal{D}_{CMU}$ 略好, 但 MPKPE/MPJPE 都差)
- $E_{\mathrm{mpjpe}}^{\mathrm{lower}}$: 0.1335 ($\mathcal{D}_{250}$) vs 0.1512 ($\mathcal{D}_{CMU}$)

在 OOD set $\mathcal{D}_{ACCAD}$ 上, $\mathcal{D}_{250}$ 仍然 best, $E_{\mathrm{mpjpe}} = 0.1421$ vs $\mathcal{D}_{CMU}$ 的 0.1780。这证明了 noisy data 反而 hurts generalization, 跟 LLM 里 "data quality > data quantity" 的 lesson 一致。

---

## 三、Teacher-Student Pipeline 细节

### 3.1 Teacher Policy

公式化 MDP:
- State: $\{p_t, o_t, g_t\}$
  - $p_t \in \mathbb{R}^{62}$: privileged info (Table VI), 包含 DoF difference (23), Keybody difference (36), root velocity (3)
  - $o_t \in \mathbb{R}^{75}$: proprioceptive state (Table V): DoF pos (23) + DoF vel (23) + last action (23) + root angular vel (3) + roll/pitch/yaw (3)
  - $g_t \in \mathbb{R}^{69}$: motion tracking target (Table VII): DoF pos (23) + keypoint pos (36) + root vel (3) + root ang vel (3) + roll/pitch/yaw (3) + height (1)
- Action: $\hat{a}_t \in \mathbb{R}^{23}$ (Unitree G1 有 23 个 actuated joints), 是 joint PD controller 的 target position
- Policy: $\hat{\pi}(\hat{a}_t \mid p_t, o_t, g_t)$
- Objective:

$$
\max_{\hat{\pi}} \mathbb{E}_{\hat{\pi}}\left[\sum_{t=0}^{T} \gamma^t \mathcal{R}(s_t, \hat{a}_t)\right]
$$

- $\gamma = 0.99$ (discount factor, Table VIII), $T$ 是 episode horizon
- 算法: PPO, clip param 0.2, entropy coef 0.005, 5 learning epochs, 4 mini batches, batch size 4096, actor MLP [512, 256, 128], value MLP [512, 256, 128]

### 3.2 Student Policy (DAgger)

Student 去掉 $p_t$, 改用 history $o_{t-H:t}$:

$$
a_t \sim \pi(\cdot \mid o_{t-H:t}, g_t)
$$

$H=10$ 是 history length (Table XI ablation 显示 $H=10$ best, $H=0$ 显著差, $H=100$ 也差 - history 过长难拟合 privileged info)。

Loss:

$$
l = \|a_t - \hat{a}_t\|^2
$$

DAgger 关键 trick: rollout student $\pi$ 自己采数据, 在每个 visited state, teacher $\hat{\pi}$ 提供 oracle action 作 supervision。迭代 minimize $l$ 直到收敛。

Student MLP size: [1024, 1024, 512] - 比 teacher 大, 因为要 encode history 推断 privileged info。

Table XI(b) DAgger ablation: 去掉 DAgger 后 $E_{\mathrm{vel}}$ 从 0.2930 升到 0.4195, $E_{\mathrm{mpjpe}}$ 从 0.1079 升到 0.1496 - DAgger 关键性极强, 没有它 student 学不到 dynamic velocity tracking。

直觉: DAgger 解决的是 covariate shift - student 直接 imitate teacher 在 teacher distribution 上的 action, 但 deploy 时 student 自己产生的 state distribution 会 drift, DAgger 在 student 自己 trajectory 上 query teacher label, 闭环解决这个 mismatch。这跟 BC (behavior cloning) 只用 expert trajectory 的差别。

---

## 四、Motion-Velocity Decoupled Control: 最关键的 methodological 创新

### 4.1 现有问题

H2O / OmniH2O 用 global keypoint tracking, 公式大概是 reward $r = \exp(-\|p_t^{\mathrm{ref}} - p_t^{\mathrm{robot}}\|)$ 其中 $p$ 是 world frame 下的 keypoint。

问题: 一旦某一帧 robot 跟 ref 偏了, 下一帧 ref keypoint 已经在前方某处, robot 要 "赶上去", 但 momentum 让它做不到, 导致 reward 持续低, policy 学到放弃, 或者 robot 摔倒。

### 4.2 ExBody2 的解法

把 global keypoint 转到 robot current frame:

$$
p_t^{\mathrm{local}} = T(p_t^{\mathrm{ref}}; x_t^{\mathrm{robot}})
$$

其中 $T(\cdot; x_t^{\mathrm{robot}})$ 是把 ref keypoint 用 robot 当前 root pose $x_t^{\mathrm{robot}}$ 做坐标变换。然后 student 用:
- Local keypoint tracking: 表达性 motion imitation
- Velocity tracking: 全局移动 guidance

训练时允许小 global drift, 周期性 reset 到 robot frame (类似 PD controller 的 integral reset); 部署时严格用 local + velocity decouple。

Reward Table I:
- DoF Position: $\exp(-0.7|q_{\mathrm{ref}} - q|)$, weight 3.0
- Keypoint Position: $\exp(-|p_{\mathrm{ref}} - p|)$, weight 2.0
- Linear Velocity: $\exp(-4.0|v_{\mathrm{ref}} - v|)$, weight 6.0
- Velocity Direction: $\exp(-4.0 \cos(v_{\mathrm{ref}}, v))$, weight 6.0
- Roll & Pitch: $\exp(-|\Omega_{\mathrm{ref}}^{\phi\theta} - \Omega^{\phi\theta}|)$, weight 1.0
- Yaw: $\exp(-|\Delta y|)$, weight 1.0

观察: **velocity reward weight (6.0+6.0=12.0) 远高于 keypoint (2.0) 和 DoF (3.0)**, 这与 decoupling 思路一致 - velocity 用来保证全局 navigation, keypoint/DoF 用来保证 local 表达。$\Omega^{\phi\theta}$ 上标 $\phi\theta$ 指 roll 和 pitch 角速度, $\Delta y$ 是 yaw 偏差。

Regularization (Table IX) 也很关键, 几个值得注意的:
- `DoF position limits`: $\mathbb{1}(d_t \notin [q_{\min}, q_{\max}])$, weight -10 - hard constraint 软化为强 penalty
- `Stumble`: $\mathbb{1}(F_{\mathrm{feet}}^x > 5 \times F_{\mathrm{feet}}^z)$, weight -2 - 防止脚横向受力 (stumble 检测)
- `Feet air time`: $T_{\mathrm{air}} - 0.5$, weight 10 - 鼓励抬脚 0.5s, 对行走 step pattern 有引导
- `Waist roll pitch error`: $\|p_t^{\mathrm{wrp}} - p_0^{\mathrm{wrp}}\|^2$, weight -0.5 - 防止腰乱晃

---

## 五、Generalist vs Specialist vs Scratch 实验解读

Table IV 是 paper 最有 insight 的 ablation。三个策略在 4 个 dataset ($\mathcal{D}_{\mathrm{easy}}, \mathcal{D}_{\mathrm{moderate}}, \mathcal{D}_{\mathrm{hard}}, \mathcal{D}_{\mathrm{ACCAD}}$) 上比:

$\mathcal{D}_{\mathrm{hard}}$ 上的 $E_{\mathrm{mpjpe}}$:
- Specialist: 0.1047
- Scratch: 0.1188
- Generalist: 0.1181

$\mathcal{D}_{\mathrm{ACCAD}}$ (OOD) 上的 $E_{\mathrm{mpjpe}}$:
- Specialist: 0.1402
- Scratch: 0.1609
- Generalist: 0.1716

注意一个反直觉点: **在 hard 数据上, generalist 反而和 scratch 差不多甚至略好 (vel tracking 上 0.1452 vs 0.1631)**。但 specialist 全面胜出。这说明:
1. Generalist 的 prior 给了 specialist warm start
2. Scratch 因为没见过 diverse motions, 在 hard / OOD 上泛化弱
3. Specialist 继承了 generalist 的 robustness, 又有 task-specific precision

这跟 LLM 里 "general pretrain + task finetune > train from scratch on task" 完全同构。

Figure 5 的 Cha-Cha dance 案例: ExBody2-Specialist (蓝) 全程低于 ExBody2-Scratch (橙) 和 Generalist (绿), 说明 finetune 真的 capture fine-grained details。

---

## 六、Real-World Deployment 细节

- Platform: Unitree G1
- Compute: Jetson Orin NX onboard
- Policy inference: 50 Hz
- Low-level control: 500 Hz
- Comm delay: 18-30 ms
- Comm middleware: LCM (Lightweight Communications and Marshalling, [21])
- Action space: 23-dim (G1 actuated joints)

Real-world 结果 Table III: $E_{\mathrm{mpjpe}}^{\mathrm{lower}}$ 从 ExBody 的 0.3239 降到 ExBody2 的 0.1054, **3x 提升**, 这是 huge gap, 主要归功于 lower body tracking + decouple。

---

## 七、与 Related Work 的定位

把 ExBody2 放在 landscape 里:

**传统 whole-body control**: MIT Humanoid [6], ANYmal [22], Hybrid Zero Dynamics [61], WABOT [27], Honda humanoid [20] - 基于 dynamics model + online QP/MPC, 精确但 brittle, 难以 expressive。

**RL-based locomotion**: RMA [28], AnyMal learning [29], Berkeley Humanoid [35], Real-world humanoid RL [47], Humanoid as next-token prediction [48] - sim-to-real RL, 但 locomotion 为主, 不太 expressive。

**Motion imitation in physics sim**: DeepMimic 系, AMP [44], ASE [45], CALM [56], MaskedMimic [57], PHC [37] - sim 内 character 控制, 难 transfer 到真实硬件。

**Real-world humanoid motion imitation**: ExBody [3, 4], H2O [18], OmniH2O [17], HumanPlus [13], HOMIE, Hover [19], ExBody2 (本文) - 直接打 sim-to-real, 用 human motion dataset 驱动。

ExBody2 在这最后一类里的差异:
- ExBody: only upper body tracking, root 跟 reference, 没 teacher-student
- H2O: global keypoint tracking (会 drift)
- OmniH2O: teleop-oriented, 全身 global keypoint
- HumanPlus: 用 RL + transformer, posture-based
- ExBody2: local keypoint + velocity decouple, automated data curation, generalist-specialist

---

## 八、Possible Extensions 与 Open Questions

Limitations section 自己提到: specialist 之间无法 smooth switch, 没有动态 policy integration。这暗示几个潜在方向:

1. **MoE (Mixture of Experts) for humanoid**: 多个 specialist 作为 experts, 加一个 router (gating network) 根据当前 motion 类别路由 - 跟 Switch Transformer 同构。motion classifier / action recognition model 提供 router input (paper 里提了但没实现)

2. **Continual learning**: 不断 finetune 新 specialist, 避免 catastrophic forgetting - EWC / L2 regularization on generalist params

3. **Diffusion policy integration**: ExBody2 还是 MLP PD target policy。换成 diffusion policy [Chi et al. UMI] 可以 capture multi-modal action distribution, 对 expressive motion (e.g. 同一拍有多个 valid pose) 更友好

4. **Vision integration**: 目前 ExBody2 全 proprioceptive。加 vision (depth / RGB) 处理地形、避障、与人交互 - ExBody2 Appendix I 已经做了 HybrIK + RGB real-time mimic 雏形

5. **Foundation model aspect**: Appendix J 用 CVAE 做 motion synthesis, 可以用 text-to-motion (MDM, MotionGPT) 当 source - 整个 pipeline 变成 "text → motion → ExBody2 policy → robot", 这是 humanoid foundation model 的雏形

6. **Reward learning**: 当前 reward 是手工 design 的 (Table I + IX)。可以用 RLHF / preference learning 让人标 motion quality, 学 reward model - 跟 LLM RLHF 思路一致

7. **Cross-embodiment**: ExBody2 专做 G1, 但 ExBody 一代做了 H1, 如果在不同 humanoid (G1, H1, Apollo, Figure, Optimus) 上 share 一个 generalist + embodiment-specific finetune, 是 cross-embodiment 的 RT-X 方向

---

## 九、Critical Thinking: 论文的 weak points

1. **Filtering 是一次性的**: base policy $\pi_0$ 训完才能 filter, 但 $\pi_0$ 本身 noise 很大, filter 的 quality 取决于 $\pi_0$。可以想象 iterative filtering (像 self-training) - 但论文没探索。

2. **Threshold $\tau^*$ 的 generalization claim 缺验证**: 论文说 "exhibits generalizability and can be effectively applied to other motion datasets", 但实验只在 CMU + ACCAD 上做, 没跨 dataset 验证 $\tau^*$ 迁移性。

3. **Specialist 选择靠 classifier**: 论文提 "motion labels or an action recognition model can classify input motions", 但没实现也没 evaluate。这其实是 deployment 的关键 - runtime 怎么知道输入是 dance 还是 kungfu?

4. **Reward weight tuning**: Table I 的 weight (3.0, 2.0, 6.0...) 看起来 manually tuned, 没有 reward learning 或 ablation 解释为什么这套 weight 最好

5. **Real-world 上没测 velocity tracking**: Table III 只有 $E_{\mathrm{mpjpe}}$, 没 $E_{\mathrm{vel}}$。但 decouple 的卖点之一是 velocity tracking, sim 上 Table II 显示 filter 后 $E_{\mathrm{vel}}$ 略升 (0.2787 → 0.2930), real 上是不是也这样没说

6. **Difficult motions 上 Specialist 比 Scratch 提升不大**: $\mathcal{D}_{\mathrm{hard}}$ 上 Specialist $E_{\mathrm{mpjpe}}$ 0.1047 vs Scratch 0.1188, 提升约 12%; 而 $\mathcal{D}_{\mathrm{easy}}$ 上 Specialist 0.0772 vs Scratch 0.0843, 提升仅 8%。hard 上 generalist prior 的价值没充分体现, 可能是 specialist finetune epochs 不够, 或 generalist prior 在 hard 上本身弱

7. **No comparison with HumanPlus / HOMIE**: baselines 只有 ExBody 系和 OmniH2O, 缺 HumanPlus、Hover、HOMIE 等 SOTA

---

## 十、Reference Links

主论文与项目:
- ExBody2 项目: https://exbody2.github.io
- ExBody (前作): https://arxiv.org/abs/2402.16796
- OmniH2O: https://arxiv.org/abs/2406.08858
- H2O: https://arxiv.org/abs/2403.04436
- HumanPlus: https://arxiv.org/abs/2406.10454
- Hover: https://arxiv.org/abs/2410.21229
- PHC (Universal humanoid motion representation): https://openreview.net/forum?id=OrOd8PxOO2

Dataset 与工具:
- AMASS: https://amass.is.tue.mpg.de
- CMU Mocap: http://mocap.cs.cmu.edu/
- IsaacGym: https://arxiv.org/abs/2108.10470
- HybrIK: https://github.com/Jeff-sjtu/HybrIK
- SMPL: https://smpl.is.tue.mpg.de/

Algorithms:
- PPO: https://arxiv.org/abs/1707.06347
- DAgger: https://arxiv.org/abs/1011.0686
- AMP (Adversarial Motion Prior): https://arxiv.org/abs/2104.02180
- ASE: https://arxiv.org/abs/2205.01906
- MaskedMimic: https://arxiv.org/abs/2409.14393

Related platforms:
- Unitree G1: https://www.unitree.com/g1
- UMI on Legs: https://arxiv.org/abs/2407.10353
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- DexCap: https://arxiv.org/abs/2403.07788
- Berkeley Humanoid: https://arxiv.org/abs/2407.21781
- Humanoid as Next Token Prediction: https://arxiv.org/abs/2402.19469

---

## 十一、Intuition 总结

最核心的三个 intuition, build your mental model:

1. **Data > Method**: Table X 是 paper 最 strong 的 ablation, 干净的 250-action 数据胜过 noisy 的 1919-action 数据。这跟 Chinchilla / LIMA 一系列 LLM 工作结论一致, "data quality beats data quantity"。对 humanoid, "clean" 的定义是 lower body feasible + upper body diverse。

2. **Decoupling 是 silver bullet**: global keypoint tracking 是 monolithic objective, 一旦失败链式崩。Decouple 成 velocity (全局移动) + local keypoint (局部表达) 是把一个硬问题分解成两个 soft 问题, 每个问题都有自己的 reward signal, policy gradient 不会互相打架。这个 decouple 思路也出现在 LocoMan、UMI on Legs 等工作里 - whole-body control 普遍倾向于 hierarchical / decoupled 设计。

3. **Generalist-Specialist 是 scaling 的钥匙**: 单一 policy 没法 simultaneously master walk、dance、kungfu, 因为 reward landscape 多 mode, policy 容易塌到 mode collapse。Generalist 学一个 broad prior (像 ImageNet pretrain), Specialist finetune 到 narrow task (像 downstream task finetune)。这条 path 让 humanoid policy 可以 scale 到无限多 task, 而不会互相干扰。

下一步值得关注: MoE humanoid policy、diffusion policy for humanoid、vision-conditioned ExBody2、cross-embodiment humanoid foundation model、以及 reward model learning 让 humanoid 不靠 hand-crafted reward。
