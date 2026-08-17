---
source_pdf: Imitate and Repurpose.pdf
paper_sha256: a01a3807aac05cfeb1cf94f1d00e1094e91aaef3345a178665cc858c0435d0d8
processed_at: '2026-08-05T09:20:20-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

Andrej，我换个讲法，咱们像聊天一样把这篇 paper 揉开，重点是让你 build intuition，细节公式我照样塞进去。

---

## 这篇 paper 到底干了啥，one sentence

**把 dog 和 human 的 MoCap 动作"蒸"成一个低维的"运动意图旋钮"，机器人拧这个旋钮就能走出自然步态；下游任务只管拧旋钮，不用管关节细节。**

听起来简单，但里面有一堆聪明设计。我从头讲。

---

## Motivation：为什么直接 RL 训 robot locomotion 很痛苦

你自己训过 RL locomotion 就知道——你给个 reward "往前走"，agent 能给你学出各种妖魔鬼怪步态：三条腿着地一条腿乱甩、base 剧烈摇晃、jerk 爆表、torque 拉满。硬件跑两下 actuator 就冒烟。

传统解法是在 reward 里加一堆 shaping term：
- "base 别晃"
- "foot clearance 保持 10cm"
- "joint velocity 平滑"
- "energy 别太高"
- "contact pattern 要像 walk"
- ……

一加就是 20 个 coefficient，每个之间还互相打架。你调高 "foot clearance" 系数，"energy" 那项就崩；调低 "base 稳定" 系数，gait 就变怪。这种 reward engineering 是 legged robot RL 的实际瓶颈，跟 algorithm 本身无关。

paper 给了一个很 elegant 的切入点：**与其在 reward 里加 20 个 shaping，不如把"什么是自然运动"这个知识预先压进一个 network**。用 MoCap 当 prior，先训练一个 universal "skill module"。它输出 joint setpoint，输入是当前 proprioception + 一个低维 latent command $z$。下游任务只训"怎么给 $z$"，joint 层面的事情 skill module 全包了。

---

## Pipeline 四步走，用人话过一遍

### Stage 1：Retarget

拿 dog MoCap（走路、转弯、小跳）和 human CMU dataset，重定向到 ANYmal 和 OP3 的 kinematics 上。ANYmal 跟 dog 形态接近但不完全一样（ANYmal 宽一些），所以 retargeting 是个 least-squares 拟合：

$$
\theta^*, \mathbf{q}_{0..T}^* = \arg\min_{\theta} \sum_{t} \arg\min_{\mathbf{q}_t} \|f(\theta, \mathbf{q}_t) - \mathbf{p}_t^{\mathrm{ref}}\|_2^2
$$

- $\theta$：robot 上对应 marker 的位置（脚、肩、髋、base 中心这些）
- $\mathbf{q}_t$：每帧 joint angle
- $f(\cdot)$：forward kinematics
- $\mathbf{p}_t^{\mathrm{ref}}$：MoCap 上对应 marker 的全局位置

交替优化 $\theta$ 和 $\mathbf{q}_t$，加个小的 regularization 把 pose 拉回 ANYmal 默认站立姿态（$\beta=0.01$），避免腿内折。对称性 augmentation 让 dog 没有 backward walking 也能合成出来——ANYmal 左右前后都对称，所以直接 mirror 一下数据就翻倍。

最终拿到 2.5 小时 ANYmal reference、1.5 小时 OP3 reference。

### Stage 2：Imitation training，把 MoCap 压进 latent

这是核心。架构是 encoder-decoder：

```
reference future 5 frames (relative)
       │
       ▼
encoder π_HL  ──►  latent z_t ∈ R^d  (低维!)
       │
       ▼
decoder π_LL  ──►  joint setpoint a_t
```

encoder 看未来 5 帧 reference（body position + orientation，relative to 当前 pose），输出一个 latent $z_t$。decoder 看 $z_t$ + 当前 proprioception（joint position、IMU、velocity 这些 raw sensor），输出 joint setpoint。

encoder 和 decoder 端到端训，目标是 imitation reward 跟 reference 对齐。但这里有个关键 trick：**latent $z$ 被一个 AR(1) prior 约束着**，KL divergence 加到 loss 里。下面讲为什么。

### Stage 3：Reuse，冻结 decoder，只训 high-level

下游任务（controllable walking、ball dribbling）来时：
- **decoder 完全冻结**
- 换一个新 high-level policy $\hat{\pi}_{\mathrm{HL}}$，输入 task obs（target velocity 或 ball/target 位置）+ proprioception，输出 latent $z_t$
- 这个 $z_t$ 喂给冻结的 decoder，decoder 翻译成 joint setpoint

reward 极简：
- controllable walking：$r_t = \exp(-\|\mathbf{v}_t - \hat{\mathbf{v}}_t\|^2 / \phi)$，只有 velocity tracking
- dribbling：$r_t = \exp(-\|\mathbf{x}_t^{\mathrm{ball}} - \hat{\mathbf{x}}_t\|^2 / \phi)$，只有 ball-target 距离

**没有任何 shaping term**。skill module 本身就是 regularization。

### Stage 4：Sim-to-real

zero-shot transfer，部署到真 ANYmal 和 OP3。靠 actuator network + domain randomization 弥补 sim-real gap。ANYmal 比较顺，OP3 sim-to-real gap 大，paper 主要 demo 在 ANYmal 上。

---

## 关键 trick 1：AR(1) prior，为什么不是 i.i.d. Gaussian

latent $z$ 的 prior 是：

$$
p(z_t \mid z_{t-1}) = \mathcal{N}\!\left(\alpha \cdot z_{t-1},\ (1-\alpha^2) \cdot \mathbf{I}\right)
$$

- $\alpha = 0.95$：时间相关系数
- $(1-\alpha^2)$：incremental variance，让 marginal $p(z_t) = \mathcal{N}(0, \mathbf{I})$，是个 stationary process

用人话说：**$z$ 不会突变，下一时刻的 $z$ 大部分继承上一时刻，加一个小扰动**。

为什么这件事极其重要？

你想想，如果 prior 是 i.i.d. Gaussian（每时刻独立采样），encoder 学出来的 $z$ 序列会跳来跳去。decoder 看到 $z$ 抖动，输出 joint setpoint 也跟着抖，gait 就是高频抽搐的样子。

但自然运动的"指令层"本来就是慢变的——你走路时不会每 20ms 重新决定"我要走还是跑"，意图是连续平滑的。AR(1) 把这个 inductive bias 直接嵌进 latent space。

paper 的 Video L 演示：光从 prior 采样 $z$（不给任何 task reward），ANYmal 就在原地踱步，**skill module 本身就是 locomotion 的 generative model**。这是 NPMP 跟 [Merel 2019](https://arxiv.org/abs/1811.01156) 一脉相承的核心 insight。

---

## 关键 trick 2：KL 在两阶段扮演两个不同角色

这条是 paper 最 subtle 的地方，很多人读 paper 容易漏掉。

训练时 KL term：

$$
\beta \, \mathbb{E}_{z^* \sim \pi_{\mathrm{HL}}} \sum_t \mathrm{KL}\!\left[\pi_{\mathrm{HL}}(z_t \mid z_{t-1}^*, x_t) \,\|\, p(z_t \mid z_{t-1}^*)\right]
$$

- $\beta$：KL coefficient
- $z^*$：reparameterization 采样出的 $z$
- $p$：AR(1) prior

同一个公式，两阶段作用完全不同。

### Imitation 阶段：$\beta$ 控制 information bottleneck 容量

encoder 看的是未来 5 帧 reference（特权信息），decoder 看的是当前 proprioception。信息必须流过 latent $z$。$\beta$ 大 = $z$ 被压向 prior = 信息流不过去 = imitation 学不会。$\beta$ 小 = $z$ 可以塞任意信息 = imitation 容易但 latent 没结构，reuse 时 high-level 没法用。

所以 paper 用 **KL schedule**：

$$
\beta(k) = 0.3 \cdot \left(1 - \left(1 - \min\!\left(1, \frac{k}{1.5 \times 10^{10}}\right)\right)^{0.2}\right)
$$

- $k$：当前 env step
- $1.5 \times 10^{10}$：schedule 起作用的 step（imitation 总共 $3 \times 10^{10}$ step）

前半段 $\beta \to 0$，让 encoder-decoder 先把 imitation 学会；后半段 $\beta \to 0.3$，把 latent 往 prior 拉。paper Figure 7 显示，固定 $\beta=0.3$ 不加 schedule 直接训练失败——还没学会 imitate 就被压垮。

这跟 VAE 训练里 $\beta$-warmup 是同一个道理：**先让信息流，再压缩**。

### Reuse 阶段：$\beta$ 控制 task-vs-style trade-off

这里 $\beta = 0.01$（很小）。直觉：
- $\beta$ 大 = $z$ 强烈贴 prior = motion 自然但响应 task 慢
- $\beta$ 小 = $z$ 自由响应 task = 任务完成好但 motion 风格偏离 MoCap

paper Figure 6 的 multi-objective plot 把这个 trade-off 画得很清楚：x 轴 velocity error，y 轴 energy（sum of squared currents），reuse $\beta$ 一调，整个 Pareto front 就移动。

**这两个 $\beta$ 的语义完全不同**——imitation $\beta$ 是 latent space 的"形状参数"，reuse $\beta$ 是"风格保持强度"。paper 用 $\beta=0.3$（imitation）/ $\beta=0.01$（reuse），跨平台跨任务不变，说明这套系数比较 robust。

---

## 关键 trick 3：decoder 双分支，避免 LSTM 过拟合 latent 序列

这条很 technical 但极其聪明，对任何 hierarchical RL 都通用。

decoder 结构（Figure 8）：

```
proprioception o_t
       │
       ▼
  input norm
       │
       ├────► Branch A: 2 FC + LSTM (256 cell)  ──► state estimation
       │                                           │
       │                                           ▼
       │                           concat with o_t, z_t
       │                                           │
       │                                           ▼
       │                           Branch B: 2 FC  ──► action modulation
       │                                           │
       ▼                                           ▼
   linear combination  ──►  a_t distribution
```

为什么这么拆？

如果 LSTM 直接同时看 $z$ 和 $o$，训练时 LSTM 会"作弊"——直接把 $z$ 当 ground truth 透传，懒得从 proprioception 推断 hidden state（contact phase、external force、actuator deviation）。

到了 reuse 阶段，$z$ 来自新 high-level policy，分布完全变了，LSTM 的内部 state 估计直接崩，decoder 输出乱套。

双分支强制 LSTM **只看 proprioception** 建一个独立的 world-estimator，$z$ 只在最末层调制 action。这样 reuse 阶段 $z$ 分布偏移时，LSTM 内部的 state estimation 依然 robust。

这个思想跟 [Information Asymmetry in KL-RL](https://arxiv.org/abs/1905.01288) 一脉相承：**让某些信息只在训练时可见，部署时不可见，强迫 network 学到 implicit estimation 而不是 memorization**。

---

## 关键 trick 4：actuator network 把 sim-real gap 吃掉

ANYmal 用 Series Elastic Actuator (SEA)，内部 PID 在 drive 里跑 2.5kHz。MuJoCo 内置 motor 模型完全模拟不了这种"PID + 弹性 + 齿轮间隙 + 温度依赖"的复合动态。

paper 学了一个 actuator network（Figure 9）：

```
torque cmd ū, joint vel q̇, position error e
       │
       ▼
  analytical PID  ──►  ref torque τ̂
       │
       ▼
  learned WaveNet-style 1D dilated conv (4 layers, receptive field 8)
       │
       ▼
  residual to previous timestep  ──►  τ (torque) + I (current)
```

- $\bar{\tau}$：open-loop torque command
- $\hat{\tau}$：PID 输出的 reference torque
- $\tau, I$：实际 torque 和电流
- 还输入 temperature $T$ 和 battery voltage $V$

为什么拆 PID + learned residual？这样改 PID gain 或 control mode 不需要重新训 actuator model，analytical 部分通用，learned 部分只补差。

**预测电流** 是个 bonus——让 paper 可以在 sim 里直接优化 energy，imitation reward 里的 $r_{\mathrm{amp}} = -5e\text{-}4 \cdot \sum_i I_i^2$ 就靠这个。Figure 6 的 y 轴 energy use 也是 sum of squared currents，相当于一个 implicit energy penalty 可视化。

OP3 是 Dynamixel servo，用 MuJoCo 内置 actuator + system identification 就够，没单独学 actuator network。这也是为什么 OP3 sim-to-real gap 大——servo 的 backlash、battery 依赖没建模。

---

## 实验结果，挑亮点讲

### Imitation quality

30m 轨迹上 base position 最大偏差 0.23m，sim 和 real 几乎一致。脚的 height 跟 reference 有 systematic gap（dynamic feasibility 让 contact point 近似有偏差），但 gait pattern 保留。

跟 [Peng 2020 "Imitating animals"](https://arxiv.org/abs/2004.00784) 对比：那个是 per-clip 训一个 controller，本文是 **一个 universal controller 覆盖整个 dataset**，这是关键区别。

### Controllable walking

velocity tracking 在大范围内合理。OP3 慢速段偏差大，因为低速平衡本身就难。

最有意思的对比是 Figure 4B/C：步态 contact pattern 跟经典 MPC walking [Bellicoso 2018] 比较，learned 的 base roll/pitch 摆动更大。MPC 显式优化 base 平稳，learned 没这个约束所以"自然晃"，但仍然稳定。

paper 拿这个反驳"四足必须 base 稳"的传统观念——动物本来就是晃的，stable locomotion 不要求 base 平稳。这是 locomotion 社区里 [Lee 2020](https://arxiv.org/abs/2011.05253) / [Miki 2022](https://arxiv.org/abs/2108.13032) 一类工作跟传统 MPC 路线的本质分歧。

### Dribbling

这个最能体现 skill module 的 transfer 能力。MoCap 里**根本没有球或踢球动作**，只有 walk/turn。但 reuse 后 ANYmal 学会用前腿和后腿都拨球——emergent behavior，说明 latent space 上"行走 + 局部 limb 位移"已经被组合出来。

reward 只有 ball-target 距离，没有任何"踢球姿势"shaping。sim 与 real 轨迹高度重合（Figure 5B 绿色 sim / 橙色 real 都贴着蓝色 target）。

这个实验比 controllable walking 更有说服力，因为 dribbling 是 MoCap 分布外的任务——证明 skill module 不是单纯 memorize，而是学到了可组合的 motor primitives。

---

## 你应该 build 的 intuition

我列几条 mental model：

### Intuition 1：skill module = motion manifold 的 implicit parameterization

decoder 把整个 MoCap dataset 的 motion manifold 压成一个 low-dim latent space。这个 manifold 上的每一点对应一段合理的"未来运动意图"。reuse 时 high-level 在这个 manifold 上 search，自然就得到 natural motion。

这跟直接在 joint space 做 RL 的根本区别：joint space 是 12 维或 20 维，绝大部分区域都是"不自然甚至危险"的 motion；latent space 是低维且**整个空间都在 natural motion manifold 附近**（因为 AR(1) prior 让 latent 不会乱跑）。

### Intuition 2：MoCap 当 prior ≠ MoCap 当 demonstration

DeepMimic [Peng 2018](https://arxiv.org/abs/1804.02717) 是把 MoCap 当 demonstration，一个 clip 训一个 controller，目标是 reproduce。

本文把 MoCap 当 **prior**——不要求 reproduce，只要求"运动风格在 MoCap 流形上"。所以 dribbling 这种 MoCap 里没有的任务也能学。这是关键概念升级。

类比一下：demonstration 是"抄答案"，prior 是"学风格"。

### Intuition 3：latent 是新的 control interface

传统 control interface 是 joint torque / joint setpoint，高维且需要动力学知识。NPMP 把 interface 变成低维 latent $z$，$z$ 直接对应"未来一段运动意图"。

这意味着任何 high-level planner（trajectory tracker、joystick、甚至 LLM-based planner）只需要输出低维 $z$，不需要懂 robot 动力学。paper 在 slalom 跟踪任务里就用了个 outer-loop PD：

$$
\hat{\mathbf{v}}_t = \bar{\mathbf{v}}_t + P \cdot (\bar{\mathbf{p}}_t - \mathbf{p}_t)
$$

- $\bar{\mathbf{v}}_t, \bar{\mathbf{p}}_t$：reference trajectory 的速度和位置
- $\mathbf{p}_t$：当前 robot 位置
- $P$：比例增益

这个 PD 输出 velocity command，velocity command 转 $z$，$z$ 转 joint setpoint。**整个 hierarchy 就是 planner → latent → joint**，每一层都 low-dim 且有明确语义。

### Intuition 4：AR(1) prior 让 skill module 同时是 generative model

光从 prior 采 $z$，robot 就在原地踱步。这是因为 AR(1) 让 $z$ 平滑变化，decoder 把 smooth $z$ 序列翻译成 smooth gait。

这意味着 skill module 不光是 control interface，还是一个 **locomotion 的 generative model**。你可以用它生成"合理的随机行走"，做 data augmentation、做 exploration prior、甚至做 unconditional sampling 研究 robot 能做什么。

### Intuition 5：reuse 阶段的探索已经被 prior 启发

paper Video N 演示：不加 skill module 直接 RL，behavior 极其 erratic。因为 high-dim joint space 里随机探索几乎不可能撞到 useful gait。

加了 skill module，policy 训练初期就在"合理 motion"附近探索——prior 采 $z$ 就是 random natural walking。**skill module 把搜索空间从高维 joint space 压缩到自然 motion manifold**，sample efficiency 大幅提升。

这是 hierarchical RL 的真正价值：不是"分层计算节省算力"，而是"下层提供了 inductive bias 让上层探索更高效"。

### Intuition 6：KL schedule 是 VAE 训练的老把戏

imitation 阶段 $\beta$ 从 0 慢慢升到 0.3。先让信息流，再压缩。这跟 $\beta$-VAE 的 warmup 一模一样。如果一开始就 $\beta=0.3$，encoder-decoder 还没学会 imitate 就被压垮，训练直接失败。

这条经验对任何用 information bottleneck 的 RL 都适用：**bottleneck 容量跟任务学习要解耦 schedule**。

### Intuition 7：symmetry 是免费的数据增强

ANYmal 左右对称、前后对称。dog MoCap 没有 backward walking，mirror 一下就有了。几何对称性 augmentation 在 locomotion 上几乎免费，应该 default 开。

这条对任何 legged robot 都通用，跟 NPMP 没直接关系，但 paper 提了，值得记住。

---

## 跟相关工作的关系，串一下

- [DeepMimic](https://arxiv.org/abs/1804.02717)：MoCap 当 demonstration，per-clip 单 controller。本文是 universal controller + prior 而非 demonstration。
- [NPMP (Merel 2019)](https://arxiv.org/abs/1811.01156)：NPMP 框架提出，仿真 humanoid。本文是其在 real robot 上的扩展。
- [CoMic (Hasenclever 2020)](https://arxiv.org/abs/2010.05891)：co-training task 与 imitation，NPMP 思路延伸。
- [AMP (Peng 2021)](https://arxiv.org/abs/2104.02180)：concurrent work，用 adversarial objective 直接在 task 里加 motion prior，不显式训 skill module。两条路都能 work，本文是 explicit skill module 路线。
- [Peng 2020 "Imitating animals"](https://arxiv.org/abs/2004.00784)：MoCap → real robot，但 per-clip，没有 reusable skill module。
- [Catch & Carry (Merel 2020)](https://arxiv.org/abs/2011.10930)：vision-conditioned whole body control，NPMP + perception 的下一步。
- [Behavior Priors (Tirumala 2020)](https://arxiv.org/abs/2010.14274)：AR(1) prior 的理论框架。
- [Hwangbo 2019](https://arxiv.org/abs/1901.08552)：ANYmal actuator network 思路来源。
- [Lee 2020 / Miki 2022](https://arxiv.org/abs/2108.13032)：ANYmal 真实场景 locomotion，perception-based，但用大量 reward shaping——本文是对这条路线的"替代方案"。
- [V-MPO](https://arxiv.org/abs/1909.12238) / [MO-VMPO](https://arxiv.org/abs/2005.07516)：训练算法。
- [Information Bottleneck](https://arxiv.org/abs/physics/0004057)：KL regularization 的理论根基。
- [VAE](https://arxiv.org/abs/1312.6114)：reparameterization trick 让 latent 可 end-to-end 训练。
- [IMPALA](https://arxiv.org/abs/1802.01561)：异步 actor-learner 训练架构。
- [dm_control](https://arxiv.org/abs/2006.12983) / [MuJoCo](https://arxiv.org/abs/2012.06276)：simulation 平台。

---

## 局限，paper 自己说的和我补充的

paper 自己说的：
- "Natural" 不一定 "optimal"——dog 步态对 ANYmal 不一定能耗最优。但实测 smooth、能耗合理，trade-off 可接受。
- MoCap dataset 覆盖度有限——dog 数据主要是 walk/turn，没跑、没跳。所以高速段 velocity tracking 偏差大（Figure 4A）。
- OP3 sim-to-real gap 大——低成本 servo 的 backlash、battery 依赖让 zero-shot transfer 困难。

我补充：
- **Reuse 还是 RL**，依然需要 reward 设计 + 数十亿 step 训练。未来用 diffusion/flow matching 直接学 latent sequence distribution（[Diffusion Policy](https://arxiv.org/abs/2303.04137) 思路）可能更 sample efficient。
- **Perception 没集成**——所有 downstream task 的 $y_t$ 都是 privileged（ground truth ball/target）。接 vision 的下一步在 [Catch & Carry](https://arxiv.org/abs/2011.10930) 已经做过了。
- **Latent dim 选择** paper 没讨论——d 太小 bottleneck 太紧，d 太大 prior 作用弱。这个 sensitivity 没给。
- **跨 morphology transfer 没做**——dog MoCap 只能 retarget 到 quadruped，不能给 biped 用。未来 cross-embodiment transfer（[ROBCOGEN](https://arxiv.org/abs/2407.06268) / [Cross-Em bodiment](https://arxiv.org/abs/2407.06268) 方向）是个 open problem。

---

## 一句话总结

**NPMP = "VAE-style encoder-decoder + AR(1) latent prior"**，MoCap → encoder → 低维 latent $z$ → decoder → joint command。imitation 训 encoder+decoder（KL warm-up），reuse 冻结 decoder 只训 high-level（小 KL 控 style-task trade-off）。skill module 同时是 constraint、exploration tool、control interface，把高 DoF robot 控制转化成低维 latent 指挥问题。

这条思路对任何做 locomotion / whole-body control / hierarchical RL 的人都极其有教益。**核心 insight 是把 MoCap 当 prior 而不是 demonstration**——这一字之差让 skill module 可以 transfer 到 MoCap 分布外的任务（dribbling）。

如果你想动手复现，从 [NPMP 原 github](https://github.com/deepmind/neural_probabilistic_motor_primitives) 起步，把 AR(1) prior 嵌入 latent，然后接你的 downstream task，路径很清晰。

---

### 主要 reference

- 本 paper 项目页：https://bit.ly/robot-npmp
- NPMP (Merel et al. ICLR 2019): https://arxiv.org/abs/1811.01156
- CoMic (Hasenclever et al. ICML 2020): https://arxiv.org/abs/2010.05891
- MO-VMPO (Abdolmaleki et al. ICML 2020): https://arxiv.org/abs/2005.07516
- V-MPO (Song et al. 2019): https://arxiv.org/abs/1909.12238
- Behavior Priors (Tirumala et al. 2020): https://arxiv.org/abs/2010.14274
- DeepMimic (Peng et al. 2018): https://arxiv.org/abs/1804.02717
- AMP (Peng et al. 2021): https://arxiv.org/abs/2104.02180
- Peng 2020 "Imitating animals": https://arxiv.org/abs/2004.00784
- Catch & Carry (Merel et al. 2020): https://arxiv.org/abs/2011.10930
- ANYmal agile skills (Hwangbo et al. 2019): https://arxiv.org/abs/1901.08552
- ANYmal perceptive locomotion (Miki et al. 2022): https://arxiv.org/abs/2108.13032
- Quadruped challenging terrain (Lee et al. 2020): https://arxiv.org/abs/2011.05253
- VAE (Kingma & Welling): https://arxiv.org/abs/1312.6114
- Information Bottleneck (Tishby et al.): https://arxiv.org/abs/physics/0004057
- Information Asymmetry in KL-RL (Galashov et al. ICLR 2019): https://arxiv.org/abs/1905.01288
- IMPALA (Espeholt et al. 2018): https://arxiv.org/abs/1802.01561
- dm_control (Tassa et al. 2020): https://arxiv.org/abs/2006.12983
- MuJoCo (Todorov et al. 2012): https://arxiv.org/abs/2012.06276
- Diffusion Policy (Chi et al.): https://arxiv.org/abs/2303.04137
- RMA (Kumar et al.): https://arxiv.org/abs/2107.04034

---

# Imitate and Repurpose: 从 MoCap 学到可复用的 robot motor skills

这篇 DeepMind 2022 年的工作（arXiv:2207.09105 邻近，[项目页](https://bit.ly/robot-npmp)）核心命题可以一句话概括：**把 human/animal MoCap 数据当作 prior，先蒸一个通用的 "skill module"，下游任务直接在这个 module 的 latent command space 里求解，自然就拿到了 smooth、natural、可 sim-to-real 的运动**。下面我把每层都拆开讲，公式变量都讲清楚。

---

## 1. 为什么需要这条路：问题背景

传统 legged robot control 的几条路各有死穴：

- **Modular hand-tuned controller**（如 ANYmal 的 MPC [1]）：模块化、每个子模块都依赖简化模型（单刚体、linear inverted pendulum 之类），只能在近似有效的 state region 工作。
- **Trajectory optimization + tracking**：discrete contact 让 NLP/TO 很难实时。
- **直接 RL**：reward shaping 一旦给不到位，学出来的 gait 高 jerk、高 torque、不 energy efficient，硬件快速磨损；reward 本身又要堆很多 term（joint regularization、CoM、foot clearance…），term 之间还会打架。

paper 给出的关键洞察是：与其在 reward 里加 20 个 shaping term 去逼出 "natural motion"，不如直接拿真实生物的 motion（dog 和 human）当 prior，把"natural motion manifold"压缩进一个 decoder，下游任务只在 latent 上 search。Skill module 充当了 inductive bias，搜索空间天然就在 natural motion 流形上。

参考：
- 经典 RL locomotion [Emergence of Locomotion](https://arxiv.org/abs/1707.02286)
- DeepMimic [Peng et al. 2018](https://arxiv.org/abs/1804.02717)
- AMP [Peng et al. 2021](https://arxiv.org/abs/2104.02180)

---

## 2. 整体 pipeline：四阶段

Figure 1 描述了完整流程，理解清楚这四步是后面所有细节的基石：

| Stage | 输入 | 输出 | 关键点 |
|---|---|---|---|
| 1. Retarget | MoCap clip (dog 或 human) | retargeted reference trajectory on robot kinematics | point-cloud 拟合，加 symmetry augmentation |
| 2. Imitation training | retargeted clips + robot sim | encoder π_HL + decoder π_LL（即 skill module） | end-to-end，information bottleneck + AR(1) KL |
| 3. Reuse | task reward + 冻结 decoder | high-level task policy 输出 latent z | 只在 latent 空间做 RL |
| 4. Sim-to-real | 冻结整套 policy | real robot 行为 | actuator network + domain randomization |

为什么这套 pipeline 有效？因为 stage 2 在做"压缩"：把高维 reference trajectory 压到低维 latent z，并强制 z 满足 AR(1) 时间结构。这意味着 decoder 学到的是"如何从当前的 proprioception 出发，按 z 的指示实现一段未来运动"。下游任务只需要"指挥"——给出 z 的序列，不再需要操心关节层面的细节。

---

## 3. 核心架构 NPMP：encoder-decoder + latent prior

这是整篇 paper 的灵魂。架构图 Figure 8 我重新画一下逻辑：

```
Stage 2 (imitation):
  reference x_t (future 5 frames of body pos/orient, relative)
            │
            ▼
  encoder π_HL(z_t | z_{t-1}, x_t)  ──[AR(1) KL prior]──►  latent z_t ∈ R^d
            │
            ▼
  decoder π_LL(a_t | o_{≤t}, z_t, h_{t-1})  ──► joint setpoint a_t

Stage 3 (reuse):
  task obs y_t + proprio o_{≤t}
            │
            ▼
  high-level π̂_HL(z_t | z_{t-1}, o_{≤t}, y_t)  ──[AR(1) KL]──►  z_t
            │
            ▼
  (frozen) decoder π_LL ──► a_t
```

### 3.1 Encoder 参数化 (式 3)

$$
\pi_{\mathrm{HL}}(z_t \mid z_{t-1}, x_t) = \mathcal{N}\!\left(\mu_{\mathrm{HL}}(z_{t-1}, x_t) + \alpha \cdot z_{t-1},\ \Sigma_{\mathrm{HL}}(z_{t-1}, x_t)\right)
$$

变量解释：
- $z_t \in \mathbb{R}^d$：latent command，d 是 latent dim（本文 ANYmal 与 OP3 都是相同架构，维度见表 8）。
- $z_{t-1}$：上一时刻的 latent，**显式进入 mean**，自带时间相关性。
- $x_t$：context，即未来 5 帧 MoCap 参考的 body position + orientation，全部 relative to 当前 pose。
- $\mu_{\mathrm{HL}}, \Sigma_{\mathrm{HL}}$：一个 2-layer MLP 的输出（每层 1024 unit，LayerNorm）。
- $\alpha = 0.95$：**AR(1) 时间常数**，控制 latent command 的 decay rate。

注意 $\alpha \cdot z_{t-1}$ 是个非常巧妙的硬约束：它直接把 prior 的"惯性"嵌入到 encoder 的 mean 里，使 latent 在结构上对齐 AR(1) 过程，后续 KL 比较好优化。

### 3.2 Decoder 参数化 (式 4)

$$
\pi_{\mathrm{LL}}(a_t \mid o_t, z_t, h_{t-1}) = \mathcal{N}\!\left(\mu_{\mathrm{LL}}(o_t, z_t, h_{t-1}),\ \Sigma_{\mathrm{LL}}(o_t, z_t, h_{t-1})\right)
$$

- $o_t$：**proprioception only**——joint position、position setpoint、IMU 角速度、linear 加速度、roll/pitch 估计。都是 raw noisy sensor reading，跟实际部署能拿到的一致。
- $h_{t-1}$：LSTM hidden state（256 cell），给 decoder 提供记忆以隐式做 state estimation 与 system identification。
- $a_t$：joint position setpoint（ANYmal 12 DoF、OP3 20 DoF）。

#### Decoder 的双分支设计（关键 trick）

paper 在 Materials and Methods 里特别提到 decoder 拆成两个 branch：

- **Branch A**：2 FC + 1 LSTM，只吃 proprioception $o_t$。它的职责是从 noisy observation 推断 hidden state（contact phase、external force、actuator 性能 deviation 等）。
- **Branch B**：把 Branch A 的 output 和 normalized proprioception + latent $z_t$ 拼起来，过 2 个 FC。
- 最后两个 branch 输出线性组合得 $a_t$ 分布。

为什么这么拆？直觉是：**纯 proprioception 的 system identification 不应该被 latent 序列带偏**。如果 LSTM 一开始就同时看 $z$ 和 $o$，训练时 LSTM 容易"作弊"——直接把 $z$ 当 ground truth 输出，导致 reuse 阶段一旦 $z$ 来自新 policy（分布偏移），LSTM 的内部状态估计全垮。双分支强迫 LSTM 独立建一个 world-estimator，$z$ 只在最末层调制。这跟 [Information asymmetry in KL-regularized RL](https://arxiv.org/abs/1905.01288) 的精神一致。

### 3.3 为什么这是 "information bottleneck"

decoder 部署时只有 proprioception 和 $z$，没有 ground-truth state。encoder 训练时看到的是 reference $x_t$（包含未来 5 帧的"特权"信息）。让信息流过 latent $z$ 时被 KL 压缩，等价于强迫 encoder 把 reference 压缩成"对未来运动最有用的低维指令"，而 decoder 必须仅靠 proprioception + $z$ 完成。

paper 显式提到这等价于用 AR(1) prior 作 variational distribution 的 VAE（参考 [VAE](https://arxiv.org/abs/1312.6114)、[Information Bottleneck](https://arxiv.org/abs/physics/0004057)、[Behavior Priors](https://arxiv.org/abs/2010.14274)）。

---

## 4. AR(1) prior：latent 的"时间形状"

(式 5)

$$
p(z_t \mid z_{t-1}) = \mathcal{N}\!\left(\alpha \cdot z_{t-1},\ (1-\alpha^2) \cdot \mathbf{I}\right)
$$

变量解释：
- $\alpha \in [0, 1)$：时间相关系数，本文 0.95。
- 协方差 $(1-\alpha^2) \mathbf{I}$：stationary variance 是 1（让 prior 是 unit-variance process），$1-\alpha^2$ 是 incremental variance，保证 marginal $p(z_t) = \mathcal{N}(0, \mathbf{I})$。

为什么是 AR(1) 不是 i.i.d. Gaussian？这是 NPMP [Merel 2019](https://arxiv.org/abs/1811.01156) 的关键发现：
- 独立 Gaussian prior 让 latent 在时间上跳跃，decoder 看到高频抖动 $z$，输出 jerk 大；
- AR(1) 让 $z$ 平滑变化，对齐自然运动"指令层"本就应该慢变的事实（你不会每 20ms 重新决定走路还是跑步）。

数学上 AR(1) 的 conditional variance 是 $(1-\alpha^2)$，所以 marginal stationary 是 unit Gaussian。在 reuse 阶段，如果完全从 prior 采样 $z$，会得到一段合理的 "random natural locomotion"——paper Video L 就演示了 ANYmal 在没有任何 task reward 下，光从 prior 采 $z$ 就能稳定踱步。这是 "skill module 同时是 generative model" 的体现。

---

## 5. Imitation training：MO-VMPO + 多目标 reward

### 5.1 算法选择

paper 用 **MO-VMPO**（[Abdolmaleki et al. 2020](https://arxiv.org/abs/2005.07516)）训练 imitation policy。MO-VMPO 是 [V-MPO](https://arxiv.org/abs/1909.12238) 的多目标扩展，每个 reward $r_k$ 配一个 constraint $c_k$，对应一个独立的 value head。优点是不要手动标量加权 reward term，而是让算法在 Pareto front 上自动平衡。

直觉上 V-MPO 是 on-policy maximum a posteriori 算法，比 PPO 更稳定（用 top-50% advantage 做 E-step、用 KL trust region 做 M-step），适合 long horizon、high-dim 的 locomotion 训练。

### 5.2 Imitation reward (式 9-14)

总 reward 形如：

$$
r = \tfrac{1}{2} r_{\mathrm{trunc}} + \tfrac{1}{2}\left(a\,r_{\mathrm{com}} + r_{\mathrm{vel}} + b\,r_{\mathrm{app}} + c\,r_{\mathrm{quat}}\right)
$$

各项（注意 $\exp(-\|\cdot\|^2 / \text{scale})$ 这种 kernel 形式让 reward 自然 in $[0,1]$）：

- $r_{\mathrm{trunc}} = 1 - \delta/d$，其中 $\delta$ 是 termination metric（式 2），$d=0.3$。这一项鼓励"远离 termination 边界"。
- $r_{\mathrm{com}} = \exp(-d \cdot \|p_{\mathrm{com}} - p_{\mathrm{com}}^{\mathrm{ref}}\|_2^2)$：CoM 位置跟踪。
- $r_{\mathrm{vel}} = \exp(-e \cdot \sum_i \|q_{\mathrm{vel},i} - q_{\mathrm{vel},i}^{\mathrm{ref}}\|^2)$：每个 joint velocity 跟踪。
- $r_{\mathrm{app}} = \exp(-f \cdot \sum_{i \in \mathcal{E}} \|p_{\mathrm{app},i} - p_{\mathrm{app},i}^{\mathrm{ref}}\|^2)$：end-effector（foot）位置。
- $r_{\mathrm{quat}} = \exp(-g \cdot \sum_{i \in \mathcal{B}} \|q_{\mathrm{quat},i} \ominus q_{\mathrm{quat},i}^{\mathrm{ref}}\|^2)$：每段 body 的 quaternion 距离。

变量集合：
- $\mathcal{B}$：body index 集合。
- $\mathcal{E}$：end-effector index。
- $\mathcal{T}$：joint index。
- $\ominus$：quaternion difference。
- $a, b, c, d, e, f, g$：平台相关 coefficient（Table 5），比如 ANYmal 上 $f=80$ 对脚的位置非常严格，因为脚离地高度直接决定 gait 自然度。

ANYmal 还额外加 $r_{\mathrm{amp}} = -5e\text{-}4 \cdot \sum_i I_i^2$（式 14），惩罚电流平方——电流由 actuator network 预测，相当于一个软的 energy penalty，用来压制高频抖动。

### 5.3 Termination metric (式 2)

$$
\delta = \frac{1}{3|\mathcal{B}|}\sum_{i \in \mathcal{B}} \|p_i - p_i^{\mathrm{ref}}\|_1 + \frac{1}{|\mathcal{T}|}\sum_{j \in \mathcal{T}} \|q_j - q_j^{\mathrm{ref}}\|_1
$$

注意用 **L1 norm**，对 outlier 没那么敏感（避免单次小偏离直接终止）。当 $\delta > \eta = 0.3$ 终止 episode。这项 reward 与 termination 共同构成"behavioral corridor"：让 policy 在 reference 附近的椭球内停留，超出就重启，鼓励 faithful imitation。

### 5.4 KL schedule (Table 7)

imitation 阶段 KL coefficient：

$$
\beta(k) = 0.3 \cdot \left(1 - \left(1 - \min\!\left(1, \frac{k}{1.5 \times 10^{10}}\right)\right)^{0.2}\right)
$$

变量：
- $k$：当前 env step 数。
- $1.5 \times 10^{10}$：schedule 起作用的 step（imitation 总共跑 $3 \times 10^{10}$ step，所以前一半主要训 imitation）。

这是一个"先模仿后压缩"的 schedule：
- 训练早期 $\beta \to 0$，让 encoder-decoder 先把 imitation 学会（信息自由流过 latent）；
- 训练后期 $\beta \to 0.3$，把 latent 朝 AR(1) 拉近，强迫 $z$ 慢变、smooth。

paper Figure 7 显示，固定 $\beta = 0.3$ 不加 schedule 训练直接失败——encoder 太早被压缩，信息流不过去，policy 无法学会 imitation。这是一个很经典的 "capacity vs compression" 平衡问题，VAE 训练里也常出现（warmup $\beta$）。

---

## 6. Reuse phase：在 latent 空间做 RL

### 6.1 新 high-level policy (式 7)

$$
\pi_{\mathrm{HL}}(z_t \mid z_{t-1}, o_{\le t}, y_t) = \mathcal{N}\!\left(\theta\,\mu_{\mathrm{HL}} + (1-\theta)\,z_{t-1},\ \Sigma_{\mathrm{HL}}\right)
$$

变量：
- $y_t$：task-specific observation。Controllable walking 是 3D target velocity；dribbling 是 ball + target 的 egocentric 3D 位置。
- $\theta \in [0, 1]$：**learned filtering constant**，初始化为 $\alpha = 0.95$。意思是 high-level 一开始完全继承 AR(1) 的慢变结构，训练过程中可以学到一个更小的 $\theta$ 来更激进地响应 task。
- $\mu_{\mathrm{HL}}, \Sigma_{\mathrm{HL}}$：MLP+LSTM 输出，$\Sigma_{\mathrm{HL}}$ 初始 0.5。

这个参数化的精妙之处：**policy 初始化等于 prior**。一开始 training 时从 prior 采 $z$，等于从 skill module "随机抽一段 natural locomotion"，policy 在此基础上学习"何时偏离 prior 来达成 task"。这也是为什么 reuse 阶段不需要大量 reward shaping——探索起点本身就在合理 motion 流形上。

### 6.2 Reuse reward 极简

- Controllable walking：$r_t = \exp(-\|\mathbf{v}_t - \hat{\mathbf{v}}_t\|^2 / \phi)$，$\phi = 0.5$ (ANYmal) 或 $0.05$ (OP3)。**只有 velocity tracking，没有任何 shaping**。
- Dribbling：$r_t = \exp(-\|\mathbf{x}_t^{\mathrm{ball}} - \hat{\mathbf{x}}_t\|^2 / \phi)$，$\phi = 1$ 或 $0.5$。**只有 ball-target 距离，跟 robot 状态完全解耦**，是个相当 sparse 的 reward。

paper 直接在主文里说：reuse 时不加 skill module 直接 RL，"训练出非常 erratic 不安全的行为"（Video N）。这说明 skill module 同时是 exploration 工具与 hard constraint——它把动作空间收缩到自然流形上。

### 6.3 Trajectory tracking controller

为了部署 ANYmal 跟随 slalom 轨迹（Figure 2B），加了个外部 PD：

$$
\hat{\mathbf{v}}_t = \bar{\mathbf{v}}_t + P \cdot (\bar{\mathbf{p}}_t - \mathbf{p}_t)
$$

- $\bar{\mathbf{v}}_t, \bar{\mathbf{p}}_t$：reference trajectory 的速度和位置。
- $\mathbf{p}_t$：当前机器人位置（来自 MoCap 或 state estimator）。
- $P$：比例增益。

这是一个简单的 outer-loop controller，证明 high-level latent command 接口"可以接任何 task planner"，skill module 起到了类似 "MPC tracking controller" 的角色，但接口是 latent 而不是 trajectory。

---

## 7. Sim-to-real：actuator network 是关键

ANYmal 用 Series Elastic Actuator (SEA)，simulator 内置 motor 模型不够准。paper 学了一个 [actuator network](https://arxiv.org/abs/1901.08552)（Figure 9）：

### 7.1 两段式 architecture

```
torque cmd ū, joint vel q̇, position error e
        │
        ▼
   analytical PID  ──►  ref torque τ̂
        │
        ▼
   learned WaveNet-style 1D dilated conv (4 layers, receptive field 8 steps)
        │
        ▼
   residual to previous timestep  ──►  τ (torque) + I (current)
```

变量：
- $\bar{\tau}$：open-loop torque command。
- $\hat{\tau}$：PID 后的 reference torque。
- $\tau, I$：实际 torque 与电流。
- 输入还包括 temperature $T$ 和 battery voltage $V$。

为什么拆 PID + learned residual？这样改 PID gain 或 control mode 不需要重新收集数据训 actuator model，analytical 部分通用，learned 部分只补 residual。这是工程上的实用考虑。

### 7.2 训练

- 0.5 小时 robot 数据，10:1:1 split，5 小时训练数据 @400Hz。
- BPTT unroll 1600 步（4 秒），batch 16。
- ADAM lr 1e-3。
- 测试 RMSE：torque 0.54 Nm，current 1.64 A。

预测电流让 paper 可以在 simulation 里直接优化 energy——前面 $r_{\mathrm{amp}}$ 与 Figure 6 的 "energy use (sum of squared currents)" 都依赖于此。

OP3 的 Dynamixel servo 用 MuJoCo 内置 actuator + system identification 就够，所以没单独学 actuator network。这也解释了为什么 OP3 sim-to-real gap 更大。

---

## 8. 实验结果拆解

### 8.1 Imitation quality (Figure 3)

- 30m 轨迹上 base position 最大偏差 0.23m（sim 和 real 都一样），这本身是相当好的 tracking。
- 脚的 height 跟 reference 有系统 gap——paper 解释是 dynamic feasibility 让接触点近似略有偏差，但 qualitative gait pattern 保留。

对比基线 [Peng et al. 2020 "Learning agile robotic locomotion skills by imitating animals"](https://arxiv.org/abs/2004.00784) 是 **per-clip 训单独 controller**，本文是一个 universal controller 覆盖整个 dataset。

### 8.2 Controllable walking (Figure 4)

- Figure 4A：velocity tracking 在很大速度范围内合理（OP3 慢速段偏差大，因为低速平衡本身难）。
- Figure 4B：步态 contact pattern 与经典 MPC walking [Bellicoso 2018] 对比——learned 的 swing/stance pattern 也合理，但 base roll/pitch 摆动更大（Figure 4C）。这个对比有意思：MPC 显式优化 base 平稳，learned 没这个约束所以"自然晃"，但仍然稳定。paper 拿这个反驳"四足必须 base 稳"的传统观念，呼应动物也是会晃的。

### 8.3 Dribbling (Figure 5)

这个任务最能体现 skill module 的 transfer 能力。MoCap 里**根本没有球或踢球动作**，只有 walk/turn。但 reuse 后 robot 学会用前腿和后腿都拨球——这是 emergent behavior，说明 latent space 上"行走 + 局部 limb 位移"已经被组合出来了。

reward 只有 ball-target 距离，没有任何"踢球姿势"shaping。sim 与 real 的轨迹高度重合（Figure 5B 绿色 sim / 橙色 real 都贴着蓝色 target）。

### 8.4 Skill space 分析 (Figure 6, 7) — paper 最有教益的部分

Figure 6 是 multi-objective plot：
- x 轴：mean squared velocity error（task performance）。
- y 轴：mean sum of squared currents（energy use）。
- 颜色：KL 系数 $\beta$。

两个子图：
- **6A**：variation 是 **imitation phase** 的 $\beta$。结论：imitation phase 的 $\beta$ 越大，reuse 越接近 Pareto front；但有 cutoff——$\beta$ 太大 imitation 学不会，reuse 也无从谈起。**imitation 的 regularization 决定 reuse 的 motion 质量**。
- **6B**：variation 是 **reuse phase** 的 $\beta$。结论：reuse 的 $\beta$ 直接 control "energy vs accuracy" 的 trade-off。$\beta$ 高 → $z$ 慢变 → motion 更接近 natural 但响应慢；$\beta$ 低 → $z$ 跟 task 走得紧 → 速度快但能耗高、风格偏离 MoCap。

这个分析给出的核心 intuition：**KL 在两阶段扮演完全不同的角色**。imitation 阶段它是"信息瓶颈容量调节器"，决定 latent space 的形状；reuse 阶段它是"风格保持 vs 任务完成度"的 trade-off knob。最终 paper 用 $\beta=0.3$（imitation）/ $\beta=0.01$（reuse），跨平台、跨任务都不变，说明这套系数比较 robust。

---

## 9. 一些容易忽略但重要的细节

### 9.1 Retargeting 优化 (式 8)

$$
\theta^*, \mathbf{q}_{0..T}^* = \arg\min_{\theta} \sum_{t} \arg\min_{\mathbf{q}_t} \|f(\theta, \mathbf{q}_t) - \mathbf{p}_t^{\mathrm{ref}}\|_2^2
$$

- $\theta$：robot 上对应 marker 的位置（足、肩、髋、base 中心）。
- $\mathbf{q}_t$：每帧的 joint position。
- $f(\cdot)$：forward kinematics。
- $\mathbf{p}_t^{\mathrm{ref}}$：MoCap 上对应 marker 的全局位置。

**交替优化**：固定 $\theta$ 优化 $\mathbf{q}_t$（least square with known Jacobian），再固定 $\mathbf{q}_t$ 优化 $\theta$。再加 $\beta \|\mathbf{q}_t - \mathbf{q}^{\mathrm{ref}}\|^2$ 正则化到 stable standing pose（$\beta=0.01$），避免因为 ANYmal 比 dog 宽导致腿内收不稳。

对称性 augmentation：dog MoCap 没 backward walking，但通过 left-right、front-back mirror 任意组合可以合成出来，**几何对称性直接数据增强**。

### 9.2 观测噪声与延迟 (Table 1)

ANYmal 的 angular velocity noise std 是 $[0.1, 0.2, 0.8]$——注意这是 per-axis 不同！yaw 轴 noise 最大（0.8），因为 ANYmal 的 yaw 估计本来就最不准。这种平台特化的 noise model 让 sim-to-real 更接近。

### 9.3 Domain randomization (Table 2)

随机化内容覆盖：body mass scale $\mathcal{U}(-0.3, 0.3)$、CoM offset、joint position offset、joint damping、friction loss、friction coefficient、P gain、torque limit。注意 friction loss 公式是 $(1+s_g)(1+s_i) \cdot c$，global 和 per-joint 都随机化，避免 over-fitting 到某个 nominal 模型。

### 9.4 Target velocity random process (式 15)

$$
x_{k+1} = x_k - w_k \cdot (x_k - y_k \cdot z_k)
$$

- $x_k$：当前 target velocity。
- $y_k \sim \mathcal{U}(-a, a)$：candidate velocity。
- $w_k \sim \mathrm{Bern}(p)$：是否更新（per component 不同 $p$）。
- $z_k \sim \mathrm{Bern}(0.5)$：是否保留上一帧 target。

这个设计巧在让 target velocity **per component 独立变化**——不会出现三个分量同时跳，逼 policy 学会解耦响应。这跟普通"每 N 秒采样新 3D vector"不同，更接近真实 joystick 操作分布。

### 9.5 Control rate 与 first-order hold

- ANYmal：agent 50Hz、main loop 400Hz、per-drive 2.5kHz。Agent 输出 50Hz 的 setpoint，main loop 用 delayed first-order hold 在 400Hz 插值。这避免了 setpoint 阶跃激发 drive 内部动态。
- OP3：33Hz agent、200Hz main、per-servo 内部。无插值，因为 Dynamixel 自己响应慢。

这种 control rate 设计本身是 sim-to-real 不可忽视的细节——simulator 里 agent 输出直接以 50Hz zero-order hold 作用于 actuator，跟 reality 的多级插值完全不一样。paper 在 sim 里 emulate 这个 delay+hold pipeline 来弥补。

---

## 10. 我的 take / intuition

下面是我从这篇 paper 提炼的几条 core intuition，也是你 build mental model 时应该记的：

### 10.1 "Skill module 是一种 implicit regularization，比 reward shaping 更优雅"

传统 RL locomotion 需要在 reward 里堆 20 个 shaping term，term 之间互相打架，调一个系数可能搞坏另一个。NPMP 把"natural motion"从 reward 里挪到了 architecture 与 prior 里——通过 imitation 阶段把 MoCap 的 motion manifold 压进 decoder，reuse 阶段 KL 把 policy 锚在这个 manifold 上。这样下游 reward 就只剩 task 本身（velocity error / ball-target distance）。**调参问题从 "20 个 reward coefficient" 变成了 "1 个 KL coefficient"**。

### 10.2 "Latent command 是一种新的 control interface"

OP3 / ANYmal 这种高 DoF robot，传统 control interface 是 joint torque 或 joint setpoint，维度高且需要动力学知识。NPMP 把 interface 变成低维 latent $z$，且 $z$ 直接对应 "未来一段运动意图"。这意味着任何 high-level planner（trajectory tracker、joystick、甚至 LLM-based planner）只需要输出低维 $z$，不需要懂 robot 动力学。这跟 [Catch & Carry](https://arxiv.org/abs/2011.10930) 的 vision-conditioned whole body control 是同一思路。

### 10.3 "AR(1) prior 是 latent 的 inductive bias，比 i.i.d. Gaussian 强很多"

AR(1) 让 latent 在时间上相关，对齐"运动指令应该慢变"的物理直觉。这点对 locomotion 至关重要，对 manipulation 可能没那么重要。这也解释了为什么 behavior priors 在 locomotion 上效果显著。

### 10.4 "KL schedule 在两阶段做不同事"

- Imitation 阶段：warm-up $\beta$ 让信息先流通起来再压缩。否则 encoder-decoder 还没学会 imitate 就被压垮。
- Reuse 阶段：固定小 $\beta$，让 task policy 可以偏离 prior 但不至于偏离太远。

这两个 schedule 的不对称是 paper 的实操经验。

### 10.5 "Decoder 双分支避免 LSTM 过拟合 latent sequence"

这条对任何 hierarchical RL 都通用：低层记忆不应依赖高层指令的特定分布。双分支强制 LSTM 独立建 world model，$z$ 只在末端调制。这避免了 reuse 阶段 high-level policy 分布偏移时 LSTM 内部 state estimator 崩溃。

### 10.6 "Actuator network 让 simulator 学到 'motor+driver' 的真实响应"

ANYmal 用 SEA（弹性串联执行器），内部 PID 在 drive 里跑 2.5kHz，simulator 用内置 motor 模型完全不准。Actuator network 把 PID 之外的非线性（齿轮间隙、温度、battery 状态对扭矩传递的影响）学掉，同时预测电流。**预测电流**这点很重要，让 paper 可以在 sim 里直接优化 energy——一种 implicit reward shaping。

### 10.7 "Reuse 阶段的探索已经被 prior 启发"

paper Video L 里演示从 prior 采 $z$ 得到 random walking。这意味着 task policy 训练初期就在"合理 motion"附近探索，不需要 epsilon-greedy 或随机扰动来发现 useful behavior。这是 skill module 作为 exploration 工具的本质：**它把搜索空间从高维 joint space 压缩到自然 motion manifold**。

### 10.8 "Symmetry augmentation 是免费的数据增强"

ANYmal 是 left-right、front-back 双对称的。dog MoCap 没有倒着走，但 mirror 一下就有了。这种几何对称性 augmentation 在 locomotion 上几乎免费，应该 default 开启。

---

## 11. 与相关工作的对照

| 工作 | 与本文关系 |
|---|---|
| [DeepMimic](https://arxiv.org/abs/1804.02717) | 单 clip 单 controller，本文一个 controller 覆盖整 dataset |
| [Peng 2020 "Imitating animals"](https://arxiv.org/abs/2004.00784) | MoCap → real robot，但 per-clip，没有 reusable skill module |
| [NPMP (Merel 2019)](https://arxiv.org/abs/1811.01156) | NPMP 框架的提出，本文是其在 real robot 上的扩展 |
| [CoMic (Hasenclever 2020)](https://arxiv.org/abs/2010.05891) | Co-training task 与 imitation，本文 NPMP 思路的延伸 |
| [AMP (Peng 2021)](https://arxiv.org/abs/2104.02180) | 并行工作，用 adversarial objective 直接在 task 里加 motion prior，不显式训 skill module |
| [Behavior Priors (Tirumala 2020)](https://arxiv.org/abs/2010.14274) | AR(1) prior 的理论框架 |
| [Hwangbo 2019](https://arxiv.org/abs/1901.08552) | ANYmal actuator network 思路来源 |
| [V-MPO](https://arxiv.org/abs/1909.12238) / [MO-VMPO](https://arxiv.org/abs/2005.07516) | 训练算法 |
| [Lee 2020 / Miki 2022](https://arxiv.org/abs/2108.13032) | ANYmal 真实场景 locomotion，perception-based，但用大量 reward shaping |
| [Information Bottleneck](https://arxiv.org/abs/physics/0004057) | KL regularization 的理论根基 |
| [VAE](https://arxiv.org/abs/1312.6114) | reparameterization trick，让 latent 可 end-to-end 训练 |
| [IMPALA](https://arxiv.org/abs/1802.01561) | 异步 actor-learner 训练架构 |
| [dm_control](https://arxiv.org/abs/2006.12983) / [MuJoCo](https://arxiv.org/abs/2012.06276) | simulation 平台 |

---

## 12. 局限与未来方向

paper 自己列了几条，我补充几条：

- **MoCap dataset 覆盖度**：dog 数据主要是 walk/turn，没跑、没跳。所以高速段 velocity tracking 偏差大（Figure 4A）。补充 MoCap 或用 trajectory optimization 数据补 [Brakel 2021](https://arxiv.org/abs/2111.00262) 是一条路。
- **"Natural" 不等于 "optimal"**：dog 的步态对 ANYmal 不一定能耗最优。但实测 motion smooth、能耗合理，trade-off 可接受。
- **OP3 sim-to-real gap**：低成本 servo 的 backlash、battery 依赖让 zero-shot transfer 困难。可以叠 [RMA-style online adaptation](https://arxiv.org/abs/2107.04034) 来补。
- **Reuse 时 high-level 还是 RL**：依然需要 reward 设计 + 数十亿 step 训练。未来用 diffusion/flow matching 直接学 latent sequence distribution（[Diffusion Policy](https://arxiv.org/abs/2303.04137) 思路）可能更 sample efficient。
- **Perception 没集成**：所有 downstream task 的 $y_t$ 都是 privileged（ground truth ball/target）。接 vision 的下一步在 [Catch & Carry](https://arxiv.org/abs/2011.10930) 已经做过了。

---

## 13. 总结：你应该带走什么

如果只记三件事：

1. **NPMP = "VAE-style encoder-decoder + AR(1) latent prior"**。MoCap → encoder → low-dim latent $z$ → decoder → joint command。Imitation 训 encoder+decoder，reuse 冻结 decoder 只训 high-level。
2. **KL 在两阶段是两个不同的 knob**：imitation 阶段控制 latent bottleneck 容量（需要 warm-up schedule），reuse 阶段控制 task-vs-style 的 trade-off（直接调小值即可）。
3. **Skill module 同时是 constraint、exploration tool、control interface**。它把"高 DoF robot 控制"问题转化成"低维 latent 指挥"问题，下游 reward 极简，sim-to-real 容易。

这套思想对任何做 locomotion / whole-body control / hierarchical RL 的人都极其有教益，把 MoCap 当 prior 而不是 demonstration 的视角尤其深刻。如果你想动手复现，从 NPMP 原始 [github](https://github.com/deepmind/neural_probabilistic_motor_primitives) 起步、把 AR(1) prior 嵌入 latent、然后接你的 downstream task，路径很清晰。

---

### 参考 / 进一步阅读

- 本 paper 项目页：https://bit.ly/robot-npmp
- NPMP (Merel et al. ICLR 2019): https://arxiv.org/abs/1811.01156
- CoMic (Hasenclever et al. ICML 2020): https://arxiv.org/abs/2010.05891
- MO-VMPO (Abdolmaleki et al. ICML 2020): https://arxiv.org/abs/2005.07516
- V-MPO (Song et al. 2019): https://arxiv.org/abs/1909.12238
- Behavior Priors (Tirumala et al. 2020): https://arxiv.org/abs/2010.14274
- DeepMimic (Peng et al. 2018): https://arxiv.org/abs/1804.02717
- AMP (Peng et al. 2021): https://arxiv.org/abs/2104.02180
- Peng 2020 "Imitating animals": https://arxiv.org/abs/2004.00784
- Catch & Carry (Merel et al. 2020): https://arxiv.org/abs/2011.10930
- ANYmal agile skills (Hwangbo et al. 2019): https://arxiv.org/abs/1901.08552
- ANYmal perceptive locomotion (Miki et al. 2022): https://arxiv.org/abs/2108.13032
- Quadruped challenging terrain (Lee et al. 2020): https://arxiv.org/abs/2011.05253
- VAE (Kingma & Welling): https://arxiv.org/abs/1312.6114
- Information Bottleneck (Tishby et al.): https://arxiv.org/abs/physics/0004057
- Information Asymmetry in KL-RL (Galashov et al. ICLR 2019): https://arxiv.org/abs/1905.01288
- IMPALA (Espeholt et al. 2018): https://arxiv.org/abs/1802.01561
- dm_control (Tassa et al. 2020): https://arxiv.org/abs/2006.12983
- MuJoCo (Todorov et al. 2012): https://arxiv.org/abs/2012.06276
- Diffusion Policy (Chi et al.): https://arxiv.org/abs/2303.04137
- RMA (Kumar et al.): https://arxiv.org/abs/2107.04034
