---
source_pdf: Catch & Carry Reusable Neural Controllers for Vision-Guided.pdf
paper_sha256: 6a94c21149efb7919b09c3d0aef672de9c6f9ed2a4699195f0d24d614170132b
processed_at: '2026-08-03T15:08:58-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Catch & Carry 用人话讲

好,我把前面那篇拆得细的技术拆解放松一点,用更讲故事的方式讲一遍。但 Andrej 你懂的,"人话" 不等于 "弱智化",我会把直觉和关键的技术细节一起讲,只是少堆 jargon,多打比方。

参考链接还是放这里方便点:
- 原文: https://dl.acm.org/doi/10.1145/3386569.3392474
- 前身 NPMP (ICLR 2019): https://arxiv.org/abs/1811.11711
- DeepMimic: https://arxiv.org/abs/1804.02717
- dm_control: https://github.com/deepmind/dm_control

---

## 1. 这篇 paper 在讲一个什么故事

想象你要教一个虚拟人怎么在仓库里搬箱子,还要能接球扔球。直接让它从零开始 RL 学习,它会学出各种奇怪解法 —— 比如用背把球拍进桶里(论文 Video 6 真的有这个 demo),或者摔倒爬不起来。问题在于:

- **奖励太稀疏**:只有"把箱子放到指定位置"才给 +1,中间怎么走怎么抓完全没引导。
- **身体维度太高**:56 个自由度,探索空间巨大,随机探索几乎不可能凑出 "弯腰、伸手、握住、抬起" 这种协调动作。
- **locomotion 和 manipulation 耦合**:你抱一个 10kg 盒子走路的时候,重心变了,步态也得变,不能把走路和搬东西当成两个独立技能拼起来。

作者的核心 idea 就是:**先给虚拟人一些"肌肉记忆",然后再让它在具体任务里用这些肌肉记忆去解决问题**。肌肉记忆是从人类动作捕捉数据里提炼出来的,提炼完以后就跟具体任务无关了,可以反复用。

---

## 2. 三步走:从 mocap 到 task

整篇 paper 的 pipeline 长这样(Figure 2):

```
Stage 1: 单段动作模仿  ->  Stage 2: 蒸馏成肌肉记忆(NPMP)  ->  Stage 3: 任务里复用
几百个小专家         一个通用低层控制器              高层策略指挥低层
```

用比喻讲:

- **Stage 1** 像让一个学徒跟着师傅模仿每一个单独的动作片段 —— 走路片段、抱盒子片段、扔球片段。每个片段都练到肌肉记忆级别,稍微推一下也能回到正确动作。
- **Stage 2** 像把这些动作都 "内化" 成一套通用的运动直觉 —— 你不再需要想 "这是走路的第 3 秒所以左脚要抬起",你只要知道 "我想往前走",身体自己知道怎么动。
- **Stage 3** 像实战:高层大脑下达 "去把那个箱子搬过来" 的指令,低层的肌肉记忆自动翻译成具体的关节动作。

---

## 3. Stage 1: 单段专家怎么训出来

每一段 mocap clip(3-5 秒的小片段),训一个 policy 让虚拟人 robustly 跟踪这段 clip。Reward 长这样(公式 1):

$$
r_t = \exp(-\beta E_{\text{total}} / w_{\text{total}})
$$

人话翻译:
- $r_t$ 是当前时刻的奖励,范围 $(0, 1]$,越接近 1 表示跟踪得越好;
- $E_{\text{total}}$ 是 "当前 pose 离参考 pose 差多远" 的总能量;
- $\beta = 10$ 是个 sharpness 参数,意思是差一点点 reward 就掉得很快,逼着 policy 紧紧贴 reference;
- $w_{\text{total}}$ 是各项能量权重的和,用来做归一化。

$E_{\text{total}}$ 由 7 项组成(公式 2 与 Appendix B):

$$
E_{\text{total}} = w_{\text{qpos}} E_{\text{qpos}} + w_{\text{qvel}} E_{\text{qvel}} + w_{\text{ori}} E_{\text{ori}} + w_{\text{app}} E_{\text{app}} + w_{\text{vel}} E_{\text{vel}} + w_{\text{gyro}} E_{\text{gyro}} + w_{\text{obj}} E_{\text{obj}}
$$

每项的意思:
- $E_{\text{qpos}}$:关节角度的 L1 误差平均;$\vec q_{\text{qpos}}$ 是当前关节角,$\vec q_{\text{qpos}}^\star$ 是参考,$N_{\text{qpos}}$ 是自由度数。
- $E_{\text{qvel}}$:关节角速度的 L1 误差。
- $E_{\text{ori}} = \|\log(\vec q_{\text{ori}} \cdot \vec q_{\text{ori}}^{\star -1})\|_2$:root 朝向误差,用 quaternion log 测地线距离。
- $E_{\text{app}}$:head、hands、feet 这些末端在 pelvis 坐标系下的位置误差。
- $E_{\text{vel}}$:末端线速度误差。
- $E_{\text{gyro}}$:root 角速度误差。
- $E_{\text{obj}}$:object 位置误差 —— 这是本文相对前作新增的项。

权重: $w_{\text{qpos}}=5, w_{\text{qvel}}=1, w_{\text{ori}}=20, w_{\text{app}}=2, w_{\text{vel}}=1, w_{\text{gyro}}=1, w_{\text{obj}}=10$。$w_{\text{obj}}=10$ 设得很高,因为 object 跟踪要强制执行。

两个关键 trick:

**Action noise**:训练时给每个 actuator 加 $\sigma=0.1$ 的高斯噪声(action 范围 $[-1,1]$)。这让 expert 不只是机械重复 clip,而是能在被推一下以后自动回到参考轨迹。这就是 "robustly track" 的来源。

**Mime experts**:对有 object 的 mocap clip,额外训一组"虚拟环境里不放 object"的 expert —— 人在做同样的动作,但手上没东西。这个看似浪费,实际是个 data balancing trick。因为如果 NPMP 训练数据全是 "抱盒子" 的轨迹,decoder 会过度 bias 到 "双手靠在一起" 的姿态(抱盒子时双手一直在盒子两侧)。Mime experts 提供了 "做同样动作但手是空的" 反例,让 decoder 不会过度倾向 carry 姿态。这个 trick 在 paper 里只用一句话提了,但极其 subtle 且重要。

---

## 4. Stage 2: 蒸馏成 NPMP —— 全文灵魂

几百个 single-clip expert 训完以后,它们各自只能跟踪自己那一段 clip,没法 generalization。需要把它们蒸馏成一个 **single conditional policy**,叫 NPMP(Neural Probabilistic Motor Primitives)。

架构很简洁:

- **Encoder** $q(z_t \mid z_{t-1}, s_{t+1 \ldots t+k})$:看过去 latent $z_{t-1}$ 加上未来 $k=5$ 步的 state,输出 latent intention $z_t$。能看未来 $k$ 步是为了 disambiguate 需要提前准备的 action(比如跳之前要下蹲)。
- **Decoder** $\pi(a_t \mid s_t, z_t)$:看当前 state $s_t$ 与 latent $z_t$,输出 action $a_t$。Decoder 就是未来 reusable 的低层控制器。

训练目标(公式 3)是 ELBO:

$$
\mathbb{E}_q \Big[ \sum_{t=1}^T \log \pi(a_t \mid s_t, z_t) + \beta \big( \log \hat p_z(z_t \mid z_{t-1}) - \log q(z_t \mid z_{t-1}, s_{t+1 \ldots t+k}) \big) \Big]
$$

人话讲:
- 第一项 $\log \pi(a_t \mid s_t, z_t)$ 是 "behavioral cloning" —— 给定 latent,decoder 应该能复现 expert 的 action;
- 第二项是 KL-like regularizer:$\log \hat p_z(z_t \mid z_{t-1})$ 是 autoregressive prior,鼓励 latent 在时间上连续平滑;$\log q$ 是 encoder posterior;两者之差约束 latent space 的结构;
- $\beta$ 控制 prior 的权重(注意:这里的 $\beta$ 跟公式 1 里的 $\beta$ 是不同符号,只是巧合同字母)。

### 全文最关键的 design choice

Encoder 训练时能看到 object state,但 decoder **只**接收 humanoid 的 proprioceptive state(关节角度、速度、身体姿态)。

这是个 information bottleneck。意思就是:decoder 学到的是 "给我一个 motor intention $z$ 加身体感觉,我就能产生合适的 muscle activation",完全不依赖环境里的 object。所有 object awareness 必须通过 $z$ 来传递。

为什么这么重要?因为如果你让 decoder 直接看到 object,那它在 warehouse 训练完就 hard-coded 了 warehouse 相关的控制方式,没法迁移到 toss 任务。Decoder 必须 object-agnostic 才能复用。

这是整个工作的灵魂设计。把它跟前面的比喻连起来:你的肌肉记忆不知道你现在在搬箱子还是在接球,它只知道 "我想让手往那个方向伸、身体往那边转",具体怎么伸怎么转由肌肉记忆负责。具体在哪、有什么 object,由高层大脑通过 $z$ 告诉它。

---

## 5. Stage 3: 任务里复用 NPMP decoder

Stage 3 把 NPMP decoder 冻结,作为低层控制器。再训一个高层 task policy $\pi_{\text{task}}(z_t \mid o_t)$,输入是 task-relevant observation,输出是 latent action $z_t$。注意输出的不再是 muscle activation 而是 **latent motor intention**。$z_t$ 喂给冻结的 decoder,decoder 输出 actual actuator command。

这相当于 **学一个新的 action space**。在这个 latent space 里做 RL,比在 raw actuator space 里做 RL 要 tractable 得多,因为 latent space 已经被 prior regularization 成 "human-like behavior manifold" —— 你随便采样一个 $z$ 序列,decoder 都会生成合理的协调动作。

一个工程细节:task policy 的输出被 clip 到 $(-2, 2)$,因为 NPMP 训练时 $z$ 经过 prior regularization 大致分布在 0 附近,过大的 $z$ 会让 decoder out-of-distribution,生成奇怪动作。这就是个 "别让高层说 decoder 听不懂的话" 的限制。

### Task policy 网络架构(Figure 5)

三股 input stream:
1. **Egocentric image**:ResNet preprocessor,来自 humanoid 头上的相机;
2. **Task instruction**:small MLP;warehouse 任务里是 phase one-hot + focal pedestal 相对位置;
3. **Proprioception**:small MLP,humanoid 自己的 joint state。

三股 stream 的 embedding 拼接后进 shared LSTM,从 shared LSTM 分叉出:
- value function head;
- 第二个 LSTM 再接 policy head。

policy 还有 task 与 proprioception stream 的 skip connection —— 类似 "actor 不应该忘记自己刚被告知的目标"。

训练算法是 V-MPO(Song et al. 2020)的变体,from replay buffer 而非纯 on-policy。1000 个 actor 并行,learning rate 1e-4,MPO $\epsilon$ 在 $\{0.5, 1.0\}$ sweep,$\gamma = 0.99$,minibatch size 128,trajectory length 50。V-MPO 是 MPO 家族的 on-policy 变体,用 KL-divergence 约束 policy update,在高维 continuous action space 上比 PPO 的 clip 方法更不容易 collapse。

---

## 6. 两个具体任务

### 6.1 Warehouse task

环境:4 个 pedestal 围一圈,2 个 box。Pedestal 距 origin $U(2.5, 3.5)$m,高度 $U(0.45, 0.75)$m,box 大小乘以 $U(0.75, 1.25)$,box mass $U(2, 7)$kg(真实 mocap 的 box 是 3kg 与 10kg,RL 阶段允许更轻是为了 curriculum)。

四个 phase 的 state machine:

$$
\text{GOTO} \to \text{LIFT} \to \text{CARRY} \to \text{PUTDOWN} \to \text{GOTO} \to \ldots
$$

Success criteria(Table 1):
- GOTO:walker 离 focal pedestal 0.65m 以内;
- LIFT:每只手至少一个 contact point,pedestal 与 box 无接触;
- CARRY:每只手至少一个 contact point 且 walker 离 target pedestal 0.65m 以内;
- PUTDOWN:walker 无 contact,pedestal 与 box 至少 4 个 contact points。

每完成一个 phase 给 reward 1.0(只给一次)。15s 一个 episode。失败条件:walker 摔倒或 box 掉地。

**核心 curriculum trick**:episode 开始时,uniformly 随机初始化四个 phase 之一,并且从 mocap 里 sampled 一个与该 phase 一致的 timestep 作为 initial pose。

为什么这么关键?因为这个 task 的 reward 极度 sparse。如果只从 GOTO 开始,agent 需要 million-step exploration 才能凑巧完成 PUTDOWN 拿到第一次 reward,gradient signal 极弱。从 PUTDOWN phase 开始,只需要学 "放下 box" 就能立即得到 reward,然后反向 chain 起来。这是经典 reverse curriculum 思想,但用 mocap pose 自然实现了。

Figure 9B 的 ablation 证明:只用 pickup 或 walk 单一 phase 初始化,任务学不会。

### 6.2 Toss task

环境:humanoid 站着,ball 在空中 3m 远处以一定速度飞向 humanoid,bucket 在地上。

Ball 初始 velocity 的计算很巧妙(Appendix E):
- $v_x \sim U(1.5, 4.5)$ m/s 朝 humanoid;
- $v_y \sim U(-0.75, 0.75)$ m/s 水平;
- $d_z \sim U(0.1, 0.4)$ m 目标高度;
- $t_{\text{hit}} = d_x / v_x$ 是 ball 到达 humanoid 的时间;
- $v_z = (4.9 t_{\text{hit}}^2 + d_z) / t_{\text{hit}}$。

这里 $4.9 = g/2 = 9.8/2$,来自 $\frac{1}{2} g t^2$ 抛物线公式。也就是说用 kinematic 弹道公式反解出能让 ball 在 $t_{\text{hit}}$ 时刻达到 $d_z$ 高度的 $v_z$。这保证了 ball 大致在 humanoid shoulder height 经过,形成 "strike zone"。

Reward:
- ball 触地或 humanoid 摔倒:episode 终止,large negative reward;
- ball 接近 humanoid 后:shaping reward $\propto 1/\text{dist}_{\text{ball-bucket}}$(只在 x-y 平面),鼓励 catch 后走向 bucket;
- ball 接触 bucket 底部:sparse positive reward。

注意没有显式 "catch" reward。agent 自己悟出:先 catch(否则 ball 落地就 terminate),再走到 bucket,再 drop。

为什么 toss 不需要 phase-based curriculum?因为 ball 飞向 humanoid 这件事本身强制 agent 介入,shaping reward 又把 agent 引到 bucket —— 这是一个"被动触发"的 curriculum,不需要从 mocap 初始化 phase。

---

## 7. 结果里几个反直觉的事

### 7.1 Vision vs State 的差异

Warehouse 任务:vision > state。这反直觉,通常以为 state 更精确应该更好。作者解释:state features 只有 box center position + orientation,缺少 "edge 与 face 相对 body" 的信息。vision 里这种几何信息更 apparent。这是 perception 研究里很经典的 observation —— 你提供给 agent 什么 feature,决定了 RL 的 upper bound。

Toss 任务:state > vision。这里 state features 对 optimal policy 已经足够,vision 让 simulation 变慢、wallclock 更长。

### 7.2 Gaze 是 emergence 出来的

Paper Figure 1 里强调 "character's gaze and posture track the ball"。这是 emergence 的 —— 训练时没有任何 gaze reward、head pose reward。但因为 vision sensor 物理上挂在 head 上,agent 必须学会"把 head 朝向 ball"才能在 image 里看到 ball。这是个 self-supervised gaze emergence。

这跟 biological active sensing 非常像(参考 Merel et al. 2019b,Nature Communications 的 hierarchical motor control 论文 https://www.nature.com/articles/s41467-019-13780-3)。后续 robotics 里的 "look at the object you manipulate" 几乎都是这样 emergent 的。

### 7.3 训得越久动作越丑

作者在 Section 4.2 末尾提了一个有意思的观察:稍微 undertrained 的 policy 看起来更 human-like。Fully trained 的 policy 学会 "hurry",动作变得 fast 与 "extreme"。

因为 reward 只奖励 task completion,不奖励 naturalness。Humanlikeness 主要由 NPMP decoder 的 prior 决定,task policy 拉得越狠,行为越偏离 prior。这是个 alignment 问题 —— 后来很多 work 专门在 "naturalness reward" 上发力(比如 Adversarial Motion Priors、PAS 等)。

### 7.4 From-scratch 学到什么

作者做了 from-scratch 的 baseline:在 toss 任务上,训了 100e9 steps(比正常多一个数量级),学到的策略是 "用背当 paddle 把球拍进桶"(Video 6)。在 warehouse 上,根本学不出完整 cycle。这印证了为什么需要 motor prior。

---

## 8. Ablation 给的 intuition

### 8.1 NPMP expert data ratio (Figure 9A)

三个版本:
- Mixed:自然比例的 warehouse + toss 数据;
- No toss:完全去掉 toss expert;
- Toss++:toss expert 数据 ×2 upsample。

Toss++ 在 toss 任务上 performance 没显著变化但 slightly 更美观;在 warehouse 任务上 learning 曲线更慢、更不稳定。

直觉:skill embedding space 是个 bottleneck,过度 represent 一种 behavior 会挤掉其他 behavior 的 capacity。这是 information-theoretic 上很 reasonable 的 trade-off。

### 8.2 Phase initialization (Figure 9B)

只用单一 phase 初始化,warehouse 任务学不会。直觉:sparse reward 下 reverse curriculum 是必须的。

### 8.3 Box size variation (Figure 9C)

用 wider distribution 训练的 policy 在 large box 上表现更好。直觉:这是 domain randomization 现象(Tobin et al. 2017 https://arxiv.org/abs/1703.06907)—— 训练分布越广,学到 invariant feature 越多,generalization 越好。

---

## 9. 整个 framework 的理论直觉

用 hierarchical RL 的语言讲:

```
Expert policies ---  蒸馏  ---> NPMP (encoder E + decoder D)
   (RL)                    (z 是 information bottleneck,
                            D 只看 humanoid proprioception)
                            |
                            | 冻结 D, 把 z 作为新 action space
                            v
                      Task policy π_task
                      (V-MPO, vision/state input)
```

这是 option framework 的变体:
- $z$ 是 option 的 continuous parameter;
- decoder D 是 intra-option policy;
- $\pi_{\text{task}}$ 是 meta-policy 选 option。

跟经典 HRL(MAXQ、HIRO、Option-Critic)区别:
1. $z$ 是 continuous 而非 discrete;
2. D 通过 supervised behavioral cloning + VAE 训练,而非 end-to-end RL;
3. $z$ 是从 mocap-derived expert distribution 中 distilled 出来的 "human-like motor intention manifold",所以 $\pi_{\text{task}}$ 探索时被自动约束在 "human-like behavior manifold" 上。

类比:你能把它想成 "Diffusion model 作为 policy prior" 的前身。NPMP decoder 就像一个 "behavior prior",$\pi_{\text{task}}$ 输出的 $z$ 是 "conditioning",decoder 把 conditioning 映射到 actual action。后来的工作比如 Diffusion Policy(https://diffusion-policy.cs.columbia.edu/)、Flow Chain 这些更进一步用 diffusion/flow 替换 VAE 来 modeling behavior prior,但 idea 的 root 在这里。

---

## 10. 几个帮你 build intuition 的 mental model

### 10.1 为什么 z 是 multi-dim continuous 而非 discrete one-hot?

如果 $z$ 是 discrete one-hot,那就是 "选一个 primitive",组合性极弱。连续 $z$ 让 decoder 能 interpolate 不同 expert 的 behavior。比如 "carry box 走" 与 "空手走" 都是 $z$ 空间里的点,中间 $z$ 可能对应 "半拿半空手走" 这种 interpolation。这对 robustness 极其重要 —— 当 box 大小 mass 变化时,最优 behavior 不是任何一个 expert 的 reproduction,而是 experts 之间的 interpolation。

### 10.2 为什么 z 需要 autoregressive prior $p_z(z_t \mid z_{t-1})$?

如果没有这个 prior,$z$ 在时间上是 i.i.d.,decoder 看到的 $z$ 序列会非常 jittery,导致动作 jerky。autoregressive prior 强制 $z$ 在时间上平滑,产生 smooth behavior。可以理解为 "soft 的 temporal smoothing regularizer"。

### 10.3 "肌肉记忆" 比喻

你可以把 NPMP decoder 想象成人的脊髓反射弧 —— 脊髓不需要知道你在打篮球还是在搬箱子,它只要接收到大脑发来的 "我想伸手往那个方向",就自动组织多个肌肉群的协调收缩。大脑负责高层 goal 与 spatial awareness,脊髓负责 low-level coordination。NPMP decoder 就是这个 "虚拟脊髓"。

### 10.4 Information bottleneck = 可复用性

Decoder 只看 proprioception,所有 environment awareness 必须通过 $z$ 传 —— 这个 bottleneck 是 reusable skill 的核心。如果你让 decoder 直接看到 object,那它在 warehouse 训练完就 hard-coded 了 warehouse 的 control,无法迁移到 toss 任务。Bottleneck 强制 decoder 学到 "纯运动学" 的能力,跟环境解耦。

---

## 11. Limitations 与 follow-up

作者自己提了几个 limitation:

1. **Sample efficiency 差**:model-free RL + 1000 actors,150e9 steps 大概 3 weeks wallclock。Task-level exploration 还在 dithering。
2. **依赖 mocap initialization 做 curriculum**(warehouse 任务)。
3. **Visual quality trade-off**:generalization 时 motion 不再严格 human-like。
4. **不能直接 sim-to-real deploy**。

后续工作可以延伸看:
- Adobe 的 ASE(Assisted Adversarial Skill Exploration, Peng et al. 2022, SIGGRAPH https://arxiv.org/abs/2112.07191):直接在这个 framework 上引入 adversarial skill discovery;
- DeepMind 的 Adversarial Motion Priors: https://arxiv.org/abs/2105.02180;
- Flow Chain 与 Diffusion Policy 把 $z$ space 变成 diffusion: https://diffusion-policy.cs.columbia.edu/;
- NASA 等针对 kinematic control 的: https://research.nvidia.com/labs/toronto-ai/nasa/

---

## 12. 一句话总结

**先从人类动作里蒸馏一个 object-agnostic 的肌肉记忆模块,然后把"下达 motor intention"作为新的 action space,在这个 space 里训高层任务策略** —— 这就是 Catch & Carry 的全部 essence。Information bottleneck(让 decoder 看不到 object)是可复用性的关键,reverse curriculum(用 mocap pose 初始化不同 task phase)是 sparse reward 任务能学会的关键。

希望这个版本更"人话"一点,Andrej。如果你要继续深入某一块(V-MPO 推导、NPMP ELBO 的变分解释、抛物线 $v_z$ 的反解推导、或者跟后来 diffusion policy 的关系),我接着讲。

---

# Catch & Carry: Reusable Neural Controllers for Vision-Guided Whole-Body Tasks 深度解析

你好 Andrej,这篇 DeepMind 2020 年发在 ACM Transactions on Graphics 上的 paper 是 humanoid whole-body control with object interaction 的一个里程碑式工作。我下面尽量把它拆得很细,让你 build intuition。

参考链接:
- paper official: https://dl.acm.org/doi/10.1145/3386569.3392474
- 项目相关作者页面 Josh Merel: https://research.google/people/author38753/josh-merel/
- 之前的 NPMP paper (ICLR 2019): https://arxiv.org/abs/1811.11711
- DeepMimic (Peng et al. 2018): https://arxiv.org/abs/1804.02717
- V-MPO (Song et al. 2020): https://arxiv.org/abs/1909.12238
- IMPALA: https://arxiv.org/abs/1802.01561
- DReCon (Bergamin et al. 2019): https://arxiv.org/abs/1907.05988
- MCP (Peng et al. 2019): https://arxiv.org/abs/1905.09808
- MuJoCo: https://mujoco.org/
- DeepMind Control Suite: https://github.com/deepmind/dm_control

---

## 1. Problem Setting: 为什么这件事难

这个 paper 想解决的问题是:让一个 56-DoF 的 simulated humanoid 在物理仿真里完成需要 whole-body coordination 与 object interaction 的任务,例如 warehouse box manipulation 与 ball catching/tossing。关键难点:

1. **Task objective 是 sparse 的**。比如 warehouse 任务里只在每个 phase 完成时给 +1 reward。设计 dense shaping reward 非常困难,因为 "怎样走过去、怎样弯腰、怎样用手抓" 都没有简单可写的目标函数。
2. **Humanoid body 是高维的**,直接 RL from scratch 会 explore 到非常奇怪的 manifold,最后学到 "用背当 paddle 把球拍进桶" 这种 solution(论文 Video 6 有真实演示)。
3. **Whole-body manipulation 与 locomotion 强耦合**。你 carry 一个 10kg box 的时候,COM 改变了,gait pattern 也得跟着改变,你没法把它当成两个独立 skill 拼起来。
4. **要让 skill 跨任务复用**。论文标题里 "reusable" 是核心点 —— 同一个 low-level motor module 要既能支持 box manipulation 又能支持 ball catching。
5. **Active perception**。Agent 用 first-person egocentric vision,需要 emergent 地学会把 head 朝向 ball 来 tracking。这跟传统 graphics 里手动给 eye gaze 完全不同。

---

## 2. 核心架构: 三阶段流水线

整篇 paper 的核心 idea 体现在 Figure 2 的三阶段 pipeline:

### Stage 1: Single-clip expert policies
对每一段 motion capture clip(3-5s 的 snippet),训练一个 **time-indexed tracking policy** $\pi_t$,它通过 RL 学会 robustly 跟踪这段 clip。Reward 是:

$$
r_t = \exp(-\beta E_{\text{total}} / w_{\text{total}}) \tag{1}
$$

这里:
- $r_t \in (0, 1]$ 是 timestep $t$ 的 normalized tracking reward;
- $\beta = 10$ 是 sharpness parameter,控制 exponential 衰减有多陡;
- $E_{\text{total}}$ 是当前 simulated pose 与 reference pose 之间的 weighted energy;
- $w_{\text{total}} = \sum_i w_i$ 是所有 energy term 权重之和,用于归一化。

$E_{\text{total}}$ 由 7 项组成(公式 2):

$$
E_{\text{total}} = w_{\text{qpos}} E_{\text{qpos}} + w_{\text{qvel}} E_{\text{qvel}} + w_{\text{ori}} E_{\text{ori}} + w_{\text{app}} E_{\text{app}} + w_{\text{vel}} E_{\text{vel}} + w_{\text{gyro}} E_{\text{gyro}} + w_{\text{obj}} E_{\text{obj}}
$$

各项含义(Appendix B):
- $E_{\text{qpos}}$: joint angle tracking error, $E_{\text{qpos}} = \frac{1}{N_{\text{qpos}}} \sum |\vec q_{\text{qpos}} - \vec q_{\text{qpos}}^\star|$,即当前 joint position 与 reference 的 L1 误差平均。$N_{\text{qpos}}$ 是 DoF 数量。
- $E_{\text{qvel}}$: joint velocity L1 误差。
- $E_{\text{ori}} = \|\log(\vec q_{\text{ori}} \cdot \vec q_{\text{ori}}^{\star -1})\|_2$: root orientation quaternion 误差,用 quaternion log 来度量测地线距离。
- $E_{\text{app}} = \frac{1}{N_{\text{app}}} \sum \|\vec x_{\text{app}} - \vec x_{\text{app}}^\star\|_2$: appendages(head, hands, feet)在 root frame 下相对 pelvis 的 Cartesian 距离平均。这一项保证了末端执行器位置接近 reference。
- $E_{\text{vel}} = 0.1 \cdot \frac{1}{N_{\text{vel}}} \sum |\vec x_{\text{vel}} - \vec x_{\text{vel}}^\star|$: global frame 下 appendage 线速度。
- $E_{\text{gyro}} = 0.1 \cdot \|\vec q_{\text{gyro}} - \vec q_{\text{gyro}}^\star\|_2$: root angular velocity 误差。
- $E_{\text{obj}} = \|\vec x_{\text{obj}} - \vec x_{\text{obj}}^\star\|_2$: object position tracking。这是本文相对之前 Merel 2019c 新增的一项。

权重: $w_{\text{qpos}}=5, w_{\text{qvel}}=1, w_{\text{ori}}=20, w_{\text{app}}=2, w_{\text{vel}}=1, w_{\text{gyro}}=1, w_{\text{obj}}=10$。注意 $w_{\text{obj}}=10$ 设得很高,因为要相对强地 enforce object tracking。

**关键 trick 1**: 训练时加 action noise $\sigma = 0.1$ per actuator(acts 在 $[-1, 1]$ 范围),用来 robustify controller。这是为什么 stage 1 出来的 expert 不仅仅是"复现一段 clip",而是能在 mild perturbation 下回到 reference 轨迹。

**关键 trick 2**: "mime experts" 数据增强。对于有 object 的 motion capture clip,额外训练一组 expert,跟 human reference 动作一样,但 virtual environment 里**不放 object**。这看似奇怪,实际作用:如果 NPMP 训练数据全是 "carry box" 的轨迹,decoder 会过度倾向于 "把双手靠在一起" 的姿态(因为 carry box 时双手一直在 box 两侧)。Mime experts 提供了 "做同样动作但手是空的" 反例,平衡了数据分布,使得 NPMP 不会过度 bias 到 carry 姿态。这是非常 subtle 但重要的一个 data balancing trick。

### Stage 2: Distillation into NPMP (Neural Probabilistic Motor Primitives)
把 stage 1 几百个 expert 的 rollout trajectories 蒸馏成一个 single conditional policy(inverse model)。这个 NPMP 的核心结构:

- **Encoder** $q(z_t \mid z_{t-1}, s_{t+1 \ldots t+k})$:输入是过去 latent $z_{t-1}$ 加上未来 $k$ 步的 state $s_{t+1 \ldots t+k}$(论文里 $k=5$),输出 latent intention $z_t$。Encoder 能"看到未来" $k$ 步,这是为了让它能 disambiguate 需要提前准备的 action(比如跳跃前要下蹲)。
- **Decoder** $\pi(a_t \mid s_t, z_t)$:输入当前 state $s_t$ 与 latent $z_t$,输出 action $a_t$。Decoder 就是未来 reusable 的 low-level controller。

训练目标是 ELBO(公式 3):

$$
\mathbb{E}_q \Big[ \sum_{t=1}^T \log \pi(a_t \mid s_t, z_t) + \beta \big( \log \hat p_z(z_t \mid z_{t-1}) - \log q(z_t \mid z_{t-1}, s_{t+1 \ldots t+k}) \big) \Big]
$$

各项含义:
- 第一项 $\log \pi(a_t \mid s_t, z_t)$ 是 reconstruction / behavioral cloning 项:给定 latent,decoder 应该能 reproduce expert 的 action $a_t$。
- 第二项里:
  - $\log \hat p_z(z_t \mid z_{t-1})$ 是 autoregressive prior,鼓励 latent 序列时间上连续平滑;
  - $\log q(z_t \mid z_{t-1}, s_{t+1 \ldots t+k})$ 是 encoder 的 posterior;
  - 两者之差是 $\log \hat p_z - \log q$ 形式的 KL-like regularizer(其实是 importance-weighted ELBO 的写法)。
- $\beta$ 在这里控制 prior 的权重,既不是固定 1 也不是 β-VAE 那个 annealing schedule。注意:这里的 $\beta$ 与公式 (1) 里的 $\beta$ 是不同符号,只是巧合同字母。

**关键 design choice(论文反复强调)**:Encoder 训练时能看到 object state,但 decoder **只**接收 humanoid 的 proprioceptive state。这保证了 decoder 输出 distribution 只依赖于 humanoid 自身状态,与环境无关 —— 因此可以 plug-and-play 到任何新场景。Decoder 这一边相当于被 forced 学到 "给我一个 motor intention z 与身体感觉,我就能产生合适的 muscle activation",完全 object-agnostic。所有 object awareness 必须通过 z 来传递,而 z 在新任务中由 high-level task policy 产生。

这种 **information bottleneck** 是整个工作的灵魂。如果你让 decoder 直接看到 object,那它在 warehouse 训练完就 hard-coded 了 warehouse 相关的 control,无法迁到 toss 任务。

### Stage 3: Task policy 复用 NPMP decoder
Stage 3 把 NPMP decoder 冻结,作为 low-level controller。新训练一个 high-level task policy $\pi_{\text{task}}(z_t \mid o_t)$,输入是 task-relevant observation(可以是 egocentric image、task instruction、proprioception),输出是 latent action $z_t$ —— 注意输出的不再是 muscle activation 而是 **latent motor intention**。$z_t$ 喂给冻结的 decoder,decoder 输出 actual actuator command。

这是典型的 **hierarchical RL** with **learned action space** 思路。skill embedding space 成了一个 "behaviorally meaningful action space",在这个空间里做 RL 比在 raw actuator space 里做 RL 要 tractable 得多。

**Action range 限制**:task policy 的输出被 clip 到 $(-2, 2)$,因为 NPMP 训练时 z 经过 prior regularization 大致分布在 0 附近,过大的 z 会让 decoder out-of-distribution。这是个简单但重要的工程细节 —— 否则 task policy 探索时会发 "解码器看不懂的指令"。

---

## 3. Task Policy 网络架构 (Figure 5)

输入有三股 stream:
1. **Egocentric image**: ResNet(He et al. 2016 风格)preprocessor;
2. **Task instruction**: small 1-2 hidden-layer MLP;对 warehouse 任务是 phase one-hot + focal pedestal relative position;
3. **Proprioception**: small MLP,包含 humanoid 自己的 joint state。

三个 stream 的 embedding 拼接后喂入一个 shared LSTM,从这个 LSTM 分叉出:
- value function head(给 RL 用);
- 第二个 LSTM 再接 policy head。

shared LSTM 让 value 与 policy 共享 useful representation(类似 A3C 那种 shared trunk)。Policy 还接收 task 与 proprioception stream 的 skip connection —— 这类似 "actor 不应该忘记自己刚被告知的目标"。

训练算法:V-MPO(Song et al. 2020)的变体,from replay buffer 而非纯 on-policy(作者发现 from-replay 更稳定)。V-trace 用于 off-policy correction(IMPALA 风格)。1000 个 actor 并行。learning rate 1e-4,MPO $\epsilon$ 在 $\{0.5, 1.0\}$ sweep,$\gamma = 0.99$,minibatch size 128,trajectory length 50。

直觉:V-MPO 是 MPO(Maximum a Posteriori Policy Optimization,Abdolmaleki et al. 2018)家族的 on-policy 变体,用 KL-divergence 约束 policy update。作者选择它而非 PPO,大概是因为 MPO 在 continuous control 上更稳定(我的经验:KL-constrained 的方法在高维 action space 上比 clip-based 的方法更不容易 collapse)。

---

## 4. 两个核心任务

### 4.1 Warehouse task (Section 4.1, Appendix D)
环境:4 个 pedestal 围一圈,2 个 box。Pedestal 距 origin $\sim U(2.5, 3.5)$m,pedestal 高度 $\sim U(0.45, 0.75)$m,box 大小乘以 $\sim U(0.75, 1.25)$,box mass $\sim U(2, 7)$kg(注意真实 mocap 的 box 是 3kg 与 10kg,这里 RL 阶段允许更轻是为了 curriculum)。

四个 phase 的 state machine:
$$
\text{GOTO} \to \text{LIFT} \to \text{CARRY} \to \text{PUTDOWN} \to \text{GOTO} \to \ldots
$$

Success criteria(Table 1):
- GOTO: walker within 0.65m of focal pedestal;
- LIFT: walker 每只手至少一个 contact point,pedestal 与 box 无接触;
- CARRY: walker 每只手至少一个 contact point 且 walker within 0.65m of target pedestal;
- PUTDOWN: walker 无 contact,pedestal 与 box 至少 4 个 contact points。

每完成一个 phase 给 reward 1.0(只给一次),phase 自动 transition。15s 一个 episode。失败条件:walker 摔倒(非脚部 ground contact)或 box 掉地。

**关键设计**:在 episode 开始时,**uniformly 随机初始化四个 phase 之一**,并且从 mocap 里 sampled 一个与该 phase 一致的 timestep 作为 initial pose。这等价于一个 "natural curriculum":你不用学会从 GOTO 走到 PUTDOWN 一整条 chain,而是从每个 phase 中间状态开始学。这是 warehouse 任务能学会的关键 —— 论文 Figure 9B 显示,如果只从 pickup phase 或 walk phase 初始化,任务学不会。

**直觉**:为什么这个 curriculum 这么关键?因为这个 task 的 reward 极度 sparse,如果你只从 GOTO 开始,agent 需要 million-step exploration 才能凑巧完成 PUTDOWN 拿到第一次 reward,gradient signal 极弱;从 PUTDOWN phase 开始,只需要学 "放下 box" 就能立即得到 reward,然后反向 chain 起来。这是经典 "reverse curriculum" 思想,但用 mocap pose 自然实现了。

### 4.2 Toss task (Section 4.1, Appendix E)
环境:humanoid 站着,ball 在空中 3m 远处,以一定速度飞向 humanoid,bucket 在地上。Ball radius 乘以 $\sim U(0.95, 1.5)$,mass $\sim U(2, 4)$kg。Ball 初始位置在 humanoid 后上方(具体说 "behind the bucket"),距 humanoid 约 $d_x = 3$m。

Ball 初始 velocity 计算(细节很 clever):
- $v_x \sim U(1.5, 4.5)$ m/s 朝 humanoid;
- $v_y \sim U(-0.75, 0.75)$ m/s 水平;
- $d_z \sim U(0.1, 0.4)$ m 目标高度;
- $t_{\text{hit}} = d_x / v_x$ 是 ball 到达 humanoid 的时间;
- $v_z = (4.9 t_{\text{hit}}^2 + d_z) / t_{\text{hit}}$。

这里 $4.9 = g/2 = 9.8/2$,来自 $\frac{1}{2} g t^2$ 抛物线。也就是用 kinematic 弹道公式反解出能让 ball 在 $t_{\text{hit}}$ 时刻达到 $d_z$ 高度的 $v_z$。这个 setup 保证了 ball 大致在 humanoid shoulder height 经过,形成 "strike zone"。

Reward 设计:
- ball 触地或 humanoid 摔倒:episode 终止,large negative reward;
- ball 接近 humanoid 后:shaping reward $\propto 1/\text{dist}_{\text{ball-bucket}}$(只在 x-y 平面),鼓励 catch 后走向 bucket;
- ball 接触 bucket 底部:sparse positive reward,鼓励 drop ball in bucket。

**注意**:这里没有显式 "catch" reward。agent 自己悟出:先 catch(否则 ball 落地就 terminate),再走到 bucket,再 drop。

为什么 toss 任务不需要 phase-based curriculum?因为 ball 飞向 humanoid 这件事本身强制 agent 介入,shaping reward 又把 agent 引到 bucket —— 这是一个"被动触发"的 curriculum,不需要从 mocap 初始化 phase。

---

## 5. 实验结果的关键 insight

### 5.1 Vision vs State performance
- Warehouse(Figure 7A):vision > state。这反直觉。作者解释:state features 只有 box center position + orientation,缺少 "edge 与 face 相对 body" 的信息。vision 里这种几何信息更 apparent。这是 perception 研究里很经典的 observation —— "你提供给 agent 什么 feature,决定了 RL 的 upper bound"。
- Toss(Figure 8A):state > vision。这里 state features 对 optimal policy 已经足够,且 vision 让 simulation 变慢 wallclock 长,样本效率的差别可能部分是 wallclock artifacts。

### 5.2 Robustness heatmap (Figure 7B, 8B)
对 warehouse,作者在 $9 \times 9$ grid 的初始 x-y 位置上,每个位置 10 次 trial,画 pickup success probability 的 heatmap。结论:agent 对大多数 initial position robust,只在"离 pedestal 太近"的少数位置失败 —— 因为太近时初始 velocity/pose 让"无法抬腿"。

对 toss,作者 discretize 初始 ball velocity 空间,画 "strike zone" heatmap。横向速度太大会让 ball 无法被 catch(return = -1)。

这种 evaluation 方式比单点 episode return 更说明问题 —— 类似 OpenAI 的 "robustness sweep" 评估法。

### 5.3 Ablations (Figure 9)
三个 ablation:

**(A) NPMP expert data ratio**:
- "Mixed":自然比例的 warehouse + toss 数据;
- "No toss":完全去掉 toss expert;
- "Toss++":toss expert 数据 ×2 upsample。

结论:Toss++ 在 toss 任务上 performance 没显著变化但 slightly 更美观;在 warehouse 任务上 learning 曲线更慢、更不稳定。**直觉**:skill embedding space 是个 bottleneck,过度 represent 一种 behavior 会挤掉其他 behavior 的 capacity。这是 information-theoretic 上很 reasonable 的 trade-off。

**(B) Phase initialization**:
- Default: all phases
- Pickup only
- Walk only

结论:只用单一 phase 初始化,warehouse 任务学不会。**直觉**:sparse reward 下,reverse curriculum 是必须的。这与 OpenAI ELE丁、Andrychowicz dexterous hand 那个 "from multiple starting states" 的 idea 一脉相承。

**(C) Box size variation**:
- Baseline: box size 在 normal 范围随机
- Large only: 只用大 box

评估时只用大 box。结论:用 wider distribution 训练的 policy 在 large box 上表现更好。**直觉**:这是一个 domain randomization 现象(Tobin et al. 2017)—— 训练分布越广,学到 invariant feature 越多,generalization 越好。

---

## 6. 关于 "为什么不用 MCP" 的讨论 (Section 5)

作者试过 Peng et al. 2019 的 MCP(Multiplicative Compositional Policies,8 个 primitives),结论是在 warehouse 上 MCP **没有比 MLP decoder 更好** —— 反而更慢、更不稳定。

作者的 hypothesis:MCP 用离散 primitives 的乘性组合,适合"行为可以通过少数 discrete mode 切换"的场景;warehouse 任务需要 continuous 多模态 behavior,discrete primitive 切换会成为 bottleneck。

我自己的看法:这个 ablation 不够 strong(只测了 warehouse 没测 toss),且 MCP 原始 paper 用的是不同的 task setup,直接迁移不容易。但作者的 intuition 是合理的 —— 高维 continuous 全身 manipulation 不像 locomotion 那样能分成 "walk/run/jump" 几个 mode。

---

## 7. 整体 framework 的理论直觉

可以这么理解整个 framework:

```
Expert policies  ---  蒸馏  --->  NPMP (encoder E + decoder D)
   (RL)                       (z 是 information bottleneck,
                               D 只看 humanoid proprioception)
                                |
                                | 冻结 D, 把 z 作为新 action space
                                v
                          Task policy π_task
                          (V-MPO, vision/state input)
```

这其实是 **option framework** 的一个变体:
- z 是 option 的 continuous parameter;
- D 是 intra-option policy;
- π_task 是 meta-policy 选 option。

但与经典 option framework 区别在于:
1. z 是 continuous 而非 discrete;
2. D 通过 supervised behavioral cloning + VAE 训练,而非 end-to-end RL;
3. z 是从 mocap-derived expert distribution 中 distilled 出来的 "human-like motor intention manifold",所以 π_task 探索时被自动约束在 "human-like behavior manifold" 上。

类比:你能把它想成一个 "Diffusion model 作为 policy prior" 的前身。NPMP decoder 就像一个 "behavior prior",π_task 输出的 z 是 "conditioning",decoder 把 conditioning 映射到 actual action。后来的工作比如 Diffusion Policy(Florence et al.)、Flow Chain 这些更进一步用 diffusion/flow 替换 VAE 来 modeling behavior prior,但 idea 的 root 在这里。

这个 paper 跟 HRL 经典工作(MAXQ, HIRO, Option-Critic)的区别:那些工作的 sub-policy 是 task-specific 训练的,这里 sub-policy 是 task-agnostic 从 mocap distilled 的。这种 "pretrain skill module then finetune task policy" 的 paradigm 后来在 robotics 被广泛采用,比如 RL + imitation 的各种 skill foundation model。

---

## 8. 一些你没在 paper 里直接看到但能帮 build intuition 的事

### 8.1 为什么 z 是 multi-dim continuous 而非 discrete one-hot?
如果 z 是 discrete one-hot,那就是 "选一个 primitive",组合性极弱。连续 z 让 decoder 能 interpolate 不同 expert 的 behavior。比如 "carry box 走" 与 "空手走" 都是 z 空间里的点,中间 z 可能对应 "半拿半空手走" 这种 interpolation。这对 robustness 极其重要 —— 当 box 大小 mass 变化时,最优 behavior 不是任何一个 expert 的 reproduction,而是 experts 之间的 interpolation。

### 8.2 为什么 z 需要 autoregressive prior $p_z(z_t \mid z_{t-1})$?
如果没有这个 prior,z 在时间上是 i.i.d.,decoder 看到的 z 序列会非常 jittery,导致动作 jerky。autoregressive prior 强制 z 在时间上平滑,产生 smooth behavior。可以理解为 "soft 的 temporal smoothing regularizer"。

### 8.3 关于 first-person vision 的 emergence
Paper Figure 1 里强调 "character's gaze and posture track the ball"。这是 emergence 的 —— 训练时没有任何 gaze reward、head pose reward。但因为 vision sensor 在 head 上,agent 必须学会"把 head 朝向 ball"才能在 image 里看到 ball。这是个 self-supervised gaze emergence。这跟 biological active sensing 非常像(参考 Merel et al. 2019b,Nature Communications 的 hierarchical motor control 论文)。后续 robotics 里的 "look at the object you manipulate" 几乎都是这样 emergent 的。

### 8.4 关于 "task visual quality vs performance" 的反直觉
作者指出(Figure 7 caption 与 Section 4.2 末尾):稍微 undertrained 的 policy 看起来更 human-like。Fully trained 的 policy 学会 "hurry",动作变得 fast 与 "extreme"。这是因为 reward 只奖励 task completion,不奖励 naturalness。Humanlikeness 主要由 NPMP decoder 的 prior 决定,task policy 拉得越狠,行为越偏离 prior。这是个 alignment 问题 —— 后来 much work 专门在 "naturalness reward" 上发力(比如 PAS, Adversarial Motion Priors 等)。论文引用的 Peng et al. 2018 DeepMimic 解决了这个 by always reward tracking mocap;但 Merel 这篇 trade-off 了 tracking 与 task,所以 task policy 能偏离 mocap。

---

## 9. 一些 limitation 与 follow-up 思路

作者自己提到几个 limitation:

1. **Sample efficiency 差**:model-free RL + 1000 actors,150e9 steps 大概 3 weeks wallclock。这是 because task-level exploration 还在 dithering。
2. **依赖 mocap initialization 做 curriculum**(warehouse 任务)。toss 不依赖,但 warehouse 依赖。
3. **Visual quality trade-off**:generalization 时 motion 不再严格 human-like。
4. **不能 sim-to-real 直接 deploy**(虽然 body 是 realistic actuation,但真实 robot 还需要大量 domain randomization 与 system id)。

Follow-up 思路(可以延伸读):
- Adobe 的ASE(Assisted Adversarial Skill Exploration, Peng et al. 2022, SIGGRAPH): https://arxiv.org/abs/2112.07191 直接在这个 framework 上引入 adversarial skill discovery;
- DeepMind 的 Async 设置 with more robots: https://arxiv.org/abs/2207.07675;
- Flow Chain 与 Diffusion Policy 把 z space 变成 diffusion: https://diffusion-policy.cs.columbia.edu/;
- NASA 等针对 kinematic control 的: https://research.nvidia.com/labs/toronto-ai/nasa/

---

## 10. 给你的几个 build-intuition 的要点总结

1. **Information bottleneck 是 reusable skill 的核心**。Decoder 只看 humanoid proprioception,object awareness 全通过 z 传 —— 这是为什么同一个 decoder 可以复用到 warehouse 与 toss 两个完全不同的 task。
2. **Mocap 提供的是 "behavior manifold",不是 "behavior itself"**。Expert 跟踪 mocap → 蒸馏 → 在 manifold 上做 task RL,本质是 "把搜索空间从 raw action space 投影到 human-like manifold"。
3. **Reverse curriculum via mocap pose initialization** 是 sparse reward 任务的关键 trick,本质上是 "用 mocap 告诉 agent 这个 task 的中间状态长什么样"。
4. **Vision 让 agent 自动学会 gaze control**,因为 visual input 物理上挂在 head 上。这是个 cheap 但 effective 的 embodiment-induced emergence。
5. **Task reward 与 motion prior 是 trade-off 关系**:reward 越强,越偏离 human-like;reward 越弱,越像 mocap 但 task performance 低。MCP、NPMP 这些方法本质都在调这个 dial。
6. **V-MPO + V-trace + 1000 actors** 是 DeepMind 那个时代的标准 distributed RL stack,wallclock 3 weeks 是 typical。

如果你想自己 reproduce 或者 explore,DeepMind 的 dm_control + dm_env + MuJoCo humanoid 是公开的,NPMP 架构本身不复杂,关键是 mocap 数据 + 三阶段训练 schedule + curriculum setup。

希望这些细节能让你 build intuition,Andrej。如果你要 push 更深某一节(比如 V-MPO 推导、或者 NPMP ELBO 的变分解释、或者那个抛物线 v_z 的 inversion 推导),跟我说哪一块。
