---
source_pdf: DextrAH-G.pdf
paper_sha256: 5a9558f11916984c71f8a763de841d156509b94f944b9dd0dc882db07d49028d
processed_at: '2026-08-03T20:55:52-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DextrAH-G 人话版

Andrej，我换个说法，用大白话再讲一遍，像我们在 Tesla 走廊里白板讨论那样。

---

## 这篇 paper 到底在干啥

想象你有一个 23 个电机的机器手臂+手（Kuka 7-DoF arm + Allegro 16-DoF hand），桌上放了个水杯，你要让它抓起来放进旁边的 bin 里。

这件事难在哪：
1. **23 维 action space**——你不可能让 RL 从头学怎么协调 23 个 motor
2. **sim 到 real gap**——sim 里学得再好，real world 总有 noise、friction 不对、相机校准误差
3. **安全**——RL policy 万一抽风，23 个 motor 一起乱动，几千美金的 Allegro 就废了
4. **感知**——policy 看到的就是一张 160×120 的 depth image，要从中推断"物体在哪、怎么抓"

这篇 paper 的核心 recipe：**用 geometric fabric 当"安全骨架"，用 RL 学"抓取策略"，用 distillation 把视觉感知塞进去**。

项目页: https://sites.google.com/view/dextrah-g

---

## Geometric Fabric 是什么——一句话版

Fabric 就是一个**人造的物理引擎**，你给它一个目标位置，它会算出一条平滑、不撞东西、不超过 joint limit 的轨迹。RL policy 只负责告诉 fabric "去哪"，fabric 负责算"怎么安全地去"。

这跟传统 RL 直接输出 motor torque 的区别：
- 纯 RL: policy 输出 23 个 motor 命令 → 万一抽风就撞桌/超 joint limit
- Fabric-guided: policy 输出 11 维目标（palm 去哪 + 手指怎么弯）→ fabric 算出安全 motor 命令

用比喻：fabric 像一条**虚拟弹簧 + 避障系统**，policy 只是拉弹簧的方向，弹簧自己会绕开障碍平滑收缩。

数学上就是公式 (1)，本质 $\mathbf{M}\ddot{\mathbf{q}} = -\mathbf{F}$，跟牛顿第二定律一样，但 $\mathbf{M}$ 和 $\mathbf{F}$ 都是设计出来的，可以保证 stability 和 composability。

参考: https://arxiv.org/abs/2405.02250

---

## Collision Avoidance 怎么做——人话

机器人被近似成一堆 sphere（Figure 5）。对每个 sphere，检查它离桌面/其他 sphere 多近：

- 离得远 → 啥也不做
- 离得近 → 排斥力开始起作用，越近越强（$1/d_i$）
- 而且只有当 sphere **正在朝障碍移动**时才排斥（那个 velocity gate $s_i$）

这个 velocity gate 特别聪明：你远离障碍时，排斥力完全关掉，不浪费能量；只有你要撞上去的瞬间才打开。

metric $\mathbf{M}_b$ 用外积 $\hat{\mathbf{n}}_i \otimes \hat{\mathbf{n}}_i$ 构造——意思就是"在朝障碍那个方向上加权大，其他方向加权小"。这样 fabric 在避障时优先响应危险方向。

---

## PCA Action Space——为什么这是灵魂

这部分我觉得是全 paper 最 clever 的设计。

### 问题

Allegro hand 有 16 个 finger joint。如果让 RL 直接输出 16 个 joint target，会怎样？

Appendix C 给了答案：训练慢 4 倍，多物体训练**完全失败**。而且学到的是**诡异的抓取策略**——比如用中指和无名指夹物体，或者手指扭成奇怪形状。

为什么？因为 16 个 joint 之间没有 prior，RL 要从头发现"手指应该协调运动"这件事。

### 解决方案

从 DexYCB 数据集（人类抓取动作）retarget 到 Allegro hand，得到一批 Allegro 的抓取 motion data，然后跑 PCA。

发现：**前 5 个主成分解释 98% 方差**。

也就是说，人类抓取时手指 16 个 joint 的运动，其实主要活在 5 维 submanifold 上。

RL policy 现在只需要输出 5 维 PCA 系数，fabric 负责把 PCA 系数映射回 16 个 joint target。这 5 维天然就是"power grasp / precision grasp"这种人类常用 hand shape。

加上 palm 的 6 维（3 位置 + 3 朝向），总 action = 11 维。从 23 降到 11，砍了一半多，且每一维都有物理意义。

这就是你在教学里反复讲的 **inductive bias**——别让网络从零学起，给它一个合理的低维表达空间。

参考 DexYCB: https://datasets.k2si.com/

---

## RL 训练怎么搞

### Asymmetric Actor Critic

Critic 看到所有 privileged info（joint force、contact force、true object pose/velocity）——这样 value estimate 准。

Policy 只看有限的 observation（不含真值）——这样学到的行为不依赖"作弊信息"，部署时不会崩。

直觉：critic 是"教练"，可以看回放和传感器数据；policy 是"运动员"，只能看比赛现场。教练用完美信息估分，运动员只学到能现场执行的策略。

参考: https://www.roboticsproceedings.org/rss14/p08.html

### Reward 非常简洁

只有 6 个 term，全是 positive reward，核心是"指尖靠近物体 → 抬起 → 送到目标"。

`minimize(e)` 函数特别关键：只在 error 创历史新低时给 reward。这避免了 policy 站在原地不动刷分。

$r_{success} \times (T_{max} - T)$ 这个设计也很巧妙——越早成功奖励越多，逼迫 policy 加快。

### Domain Randomization

mass [0.3, 3.0] 倍随机、friction [0.5, 1.1] 倍随机、joint stiffness/damping loguniform、gravity 加 noise、observation/action 都加 noise。

随机 wrench perturbation：0.1 概率给物体一个随机力/力矩，逼 policy 学 robust grasp。

摩擦系数特意降到 0.7，防止 policy 学到"靠摩擦粘住物体"的 fragile 策略。

### Cspace Damping Curriculum

低 profile 物体（pot、cup）需要手指贴近桌面。但 fabric 的 collision avoidance 会让 policy 害怕近桌面。

解法：训练开始 cspace damping=0（允许碰撞、最大探索），success rate > 90% 后 +0.1，重复到 10。让 policy 先学会"啥都能碰"，再学会"安全地碰"。

### 训练规模

Isaac Gym, 8192 robots 并行/GPU × 4 V100，68 小时，4.7B frames，相当于 10 年仿真时间。PPO, lr=5e-4, γ=0.998。

参考 Isaac Gym: https://arxiv.org/abs/2108.10470

---

## Network 架构的两个小细节

Teacher policy: MLP → LSTM(1024) → output，**LSTM 外面套 skip connection**。

这个 skip 跟 ResNet 一个道理——LSTM 训练不稳，让网络可以选择"绕过 LSTM"，把 LSTM 输出当 residual。梯度也能直接回流，避免 BPTT 消失。

Student 用 GRU 替代 LSTM——参数少，显存省（distill 在单卡 3090 上跑），且对短时序够用。

这跟你讲 RNN 时说的"RNN 是隐式 memory"完全对应。抓取时物体被手挡住，depth 看不到，但 GRU 记得"刚才物体在那"，能继续输出合理 action。

---

## Distillation：怎么从 privileged policy 变成 depth policy

Teacher: 输入 privileged state → 输出 11-D action
Student: 输入 depth image + robot state → 输出 11-D action **+ 预测物体 3D 位置**

为什么让 student 也预测物体位置？因为 deployment 时有个 state machine 要用这个信息判断"物体已经抬起了，该转 transport 模式了"。

Loss = action MSE + 0.1 × position MSE。

DAgger online distillation：student 跑 6 步，BPTT 回传 6 步，每个 episode 结束 update。12 小时 / 4.67 仿真天，单卡 3090。

Depth augmentation 很关键：sim depth 完全干净，real depth 有各种 noise。他们加了 dropout、random pixel、line artifact（模拟线缆）、相机 placement 扰动。Figure 13 直观对比。

参考 DAgger: https://proceedings.mlr.press/v15/ross11a.html

---

## Real-World Deployment 架构

分层 ROS2 node：

- Arm joint PD: 1 kHz
- Hand joint PD: 333 Hz
- Geometric fabric: 60 Hz
- FGP policy: 15 Hz
- Depth stream: ~30 Hz

关键设计：**fabric node 跟 FGP node 解耦**。即使 FGP policy 崩了、model 输出垃圾、node 死掉，fabric node 还在 60 Hz 持续输出安全命令，机器人不会失控。

论文原话："over many hours of testing DextrAH-G (and a variety of ill-behaved FGPs), no hardware was damaged."

这是 fabric 最大的实用价值——**让 RL 可以安全地探索和部署**。

---

## 实验结果到底多强

### Single Object（11 个 standard 物体，每个 5 trials）

DextrAH-G 在 9/11 物体达到 100%，平均 92.7%。baselines（DexDiffuser、ISAGrasp、Matak）普遍 60-80%。

### Bin Packing（30 物体连续抓取）

这是真正的 application-level 测试：
- 87% success rate over 256 attempts
- 5.63 picks-per-minute (PPM)
- 平均连续成功 6.56 次，最高 27 次
- median cycle time 8 秒

对比人类估计 16.53 PPM——DextrAH-G 已经到人类 1/3 速度，且能连续工作不累。

### 失败模式

主要 8% 是"不小心把物体推出工作区"（fabric 的边界太严），3% 是"对某些难物体反复抓不到"。

最难物体：small bottle（小）、green cup（滑）、pot（大）、apple（滚）、sanitizer bottle（透明 depth 看不见）。

---

## 限制和未来方向

Paper 自己承认：
1. PCA 限制 dexterity——只能 power/precision grasp，做不了 in-hand reorientation
2. Obstacle avoidance 还是 model-based，应该让 RL 学 sensory-based
3. RL 探索困难近 high-cost region，低 profile 物体差
4. 只能单物体，不能处理 clutter

我额外想到的：
- **No tactile sensing**——Allegro 没有触觉传感器，全靠 vision + proprioception。drop rate 1% 可能靠 tactile 能降到 0
- **One-hot object embedding 是 sim-only 概念**——teacher 学时知道是哪个物体，student 靠 depth 隐式推断。real test 用了部分 training 物体，真正 novel object generalization 还没完全 stress test
- **Single camera**——occlusion 严重，全靠 GRU memory 撑

---

## 跟你 Andrej 视角的共鸣点

### Inductive Bias 是王道

这篇 paper 通篇都在"给网络先验"：
- Fabric 给"安全 + 平滑"先验
- PCA 给"人类 hand shape"先验
- Asymmetric AC 给"不依赖作弊信息"先验
- Skip connection 给"梯度直接回流"先验

纯 end-to-end 在 23-DoF 空间会死。这跟你一直强调的"give the network the right inductive bias"完全一致。

### Teacher-Student = Knowledge Distillation

这跟 NLP 里的 distillation（Hinton 2015）同构——大 teacher 蒸馏到小 student。只不过这里 teacher 有 privileged info，student 只有 raw sensor。视觉感知是 distillation 过程中**强迫学进去**的。

### State-Based Model 处理 Occlusion

GRU 在这里扮演"短期记忆"——物体被手挡住时，GRU 记得刚才物体的位置和速度，继续输出合理 action。这跟你讲 RNN/LSTM 时说的"隐式 state"完全对应。

如果换成 transformer + causal mask + history window，理论上也能 work，但 GRU 在这个 scale 更轻量。

### Reward 设计 = Implicit Curriculum

`minimize(e)` 只奖励"打破历史最佳"。这避免了 reward hacking 的局部最优（站在原地不动）。跟 self-imitation learning、RLHF 里的 baseline 思想相通。

---

## 一句话总结

**Fabric 当安全骨架压缩 action space，RL 学 task policy，distillation 把视觉塞进去。三个 trick 缝合起来，让 23-DoF dexterous grasping 第一次做到 5.6 PPM + 87% success 的真实可用水平。**

这个 recipe 下一步自然是让 fabric 本身也 learnable（Neural Geometric Fabrics, Xie 2023），fabric 提供 structure，neural net 提供 adaptation。NVIDIA 这条线在往那走。

参考 Neural Fabrics: https://arxiv.org/abs/2303.02502

---

希望这版"人话"更直观。如果你想我再展开哪块（比如 velocity gate 的 tanh 为什么这样设计、PCA 5 维为什么是 98%、GRU vs LSTM 在这个任务的 trade-off），随时问。

---

# DextrAH-G 深度技术解析

Andrej，这篇 paper 是 NVIDIA、Stanford、Utah 联合的工作，本质上是把 Geometric Fabrics（一种结构化 controller）+ RL + Teacher-Student Distillation 三者缝合起来，做出一个 23-DoF dexterous arm-hand 的 pixels-to-action 抓取系统。让我从底层数学一直讲到 sim2real。

## 1. 整体架构哲学

这篇工作有一个非常清晰的设计理念：**把"安全约束 + 行为先验"放进 controller，把"任务目标"留给 RL 学习**。这是相对于纯 end-to-end RL 的反叛。纯 RL 在 23-DoF 空间里探索会非常慢，且容易学到 weird behavior（比如 paper Appendix C 提到的"用中指和无名指夹物体"这种诡异策略）。

通过 geometric fabric 提供 inductive bias，RL 只需要在 11 维 action space（6 维 palm pose + 5 维 PCA hand）上学习，且 reward function 可以非常简洁——只需要 fingertip-object contact 和 lift-to-goal。这其实呼应了你在 Tesla/教学里常说的"give the network the right inductive bias"思想。

参考: 
- Geometric Fabrics 原始 paper: https://arxiv.org/abs/2405.02250
- 之前的 Fabrics: https://arxiv.org/abs/2309.07368

---

## 2. Geometric Fabrics 核心方程

公式 (1) 是整个系统的根：

$$\mathbf{M}_f(\mathbf{q}_f, \dot{\mathbf{q}}_f) \ddot{\mathbf{q}}_f + \mathbf{f}_f(\mathbf{q}_f, \dot{\mathbf{q}}_f) + \mathbf{f}_\pi(\mathbf{a}) = 0$$

变量逐个拆解：
- $\mathbf{M}_f \in \mathbb{R}^{n \times n}$: positive-definite **metric matrix**（mass-like），表达"在哪些方向上响应应该更激烈"。这类似 Riemannian metric，方向上越重要 metric 越大。
- $\mathbf{q}_f, \dot{\mathbf{q}}_f, \ddot{\mathbf{q}}_f \in \mathbb{R}^n$: fabric 自己的"虚拟" position/velocity/acceleration，下标 $f$ 强调这是 fabric state 而非 robot state
- $\mathbf{f}_f \in \mathbb{R}^n$: nominal geometric force，构造出来的"路径生成力"——collision avoidance、joint limit repulsion 这些都靠它
- $\mathbf{f}_\pi(\mathbf{a}) \in \mathbb{R}^n$: policy action $\mathbf{a} \in \mathbb{R}^m$ 注入的 driving force，下标 $\pi$ 表示 policy

**Intuition**: 这个方程本质就是 $\mathbf{M}\ddot{\mathbf{q}} = -\mathbf{F}$，跟牛顿第二定律同构。但 $\mathbf{M}$ 和 $\mathbf{F}$ 都是**人为设计**的，可以保证 Lyapunov stability、composability（多个 term 加起来不会爆炸）。Fabric state $\mathbf{q}_f$ 通过 PD control 让真实 robot 跟随，实现"虚拟动力学引导真实机器人"。

fabric 在 60 Hz 上 forward integrate（二阶 Runge-Kutta），且 velocity target 始终设为 0（这是一个 trick，让 robot 总是试图"停下"，配合 fabric 的 attractor 实现稳态收敛）。

---

## 3. Collision Avoidance 细节

这是 fabric 最有意思的地方。机器人被建模为 sphere 集合（Figure 5），每个 sphere 通过 forward kinematics 得到 origin：

$$\mathbf{x} = \phi_{fk}(\mathbf{q}) \in \mathbb{R}^3$$

对每个 collision body $i$：
- $\mathbf{r}_i \in \mathbb{R}^3$: collision body 上离 sphere 最近的点
- $\hat{\mathbf{n}}_i = \frac{\mathbf{r}_i - \mathbf{x}}{\|\mathbf{r}_i - \mathbf{x}\|}$: 方向单位向量
- $d_i$: signed distance
- $\underline{d}_i = \max(d_{min}, d_i)$: lower-bounded distance（避免除零）

**Base acceleration response**（远离 collision 的方向）：

$$\ddot{\mathbf{x}}_b = -\sum_i \frac{1}{\underline{d}_i} \hat{\mathbf{n}}_i$$

直觉：距离越近，1/$d_i$ 越大，排斥越强；方向沿 $\hat{\mathbf{n}}_i$ 远离 collision。

**Geometric term**（HD2，速度不变路径）：
$$\ddot{\mathbf{x}} = k_g \|\dot{\mathbf{x}}\|^2 \hat{\ddot{\mathbf{x}}}_b$$

这里 $k_g$ 是 gain。HD2 是 homogeneous of degree 2 in velocity——加速度正比于速度平方，意味着路径形状不随速度变化，只随几何变化。这是 Riemannian Motion Policies 的核心思想保留下来。

**Forcing term**（在边界附近推开）：
$$\ddot{\mathbf{x}} = k_f \hat{\ddot{\mathbf{x}}}_b - b\dot{\mathbf{x}}$$

$b$ 是 damping scalar，防止振荡。

**Base metric**（关键设计）：
$$\mathbf{M}_b = \sum_i \frac{s_i}{d_i} \hat{\mathbf{n}}_i \otimes \hat{\mathbf{n}}_i$$

外积 $\hat{\mathbf{n}}_i \otimes \hat{\mathbf{n}}_i$ 产生一个 rank-1 matrix，其唯一非零特征向量沿 $\hat{\mathbf{n}}_i$ 方向。求和后，metric 的 eigenvectors 沿所有重要的 collision 方向排布，eigenvalue 大小代表优先级。

**Velocity gate**（特别 clever）：
$$s_i = \frac{1}{2}\tanh(-\alpha_1(v_i - \alpha_2)) + 1$$

其中 $v_i = -\dot{\mathbf{x}} \cdot \hat{\mathbf{n}}_i$，是 signed impact speed（朝向 collision 时为负）。

这个 gate 只在 sphere **正在朝 collision body 移动**时激活 avoidance。这避免了"远离时还浪费能量排斥"的无意义行为。

最终 metric $\mathbf{M} = \frac{\beta}{\tilde{d}^2} \hat{\mathbf{M}}_b$，其中 $\tilde{d} = \min_i\{\underline{d}_i\}$。$\frac{1}{\tilde{d}^2}$ 让最近的 collision 主导整个 metric。

参考 RMP 思想：https://arxiv.org/abs/2103.05922

---

## 4. Action Space 设计——PCA 是灵魂

这是这篇 paper 最值得讲的设计。

### 4.1 为什么不直接用 16-D hand joint space?

Appendix C 显示，直接用 cspace joint position 作 action，RL 训练慢 4 倍，且多物体训练**完全失败**。原因：
1. 高维 action space（23-D）探索困难
2. 16 个 finger joint 之间没有 prior correlation，policy 学到的手指姿态会"deranged"——比如中指和无名指夹物体这种诡异策略
3. underspecified reward 不能约束 finger motion 的"美感"

### 4.2 PCA from Human Retargeting

从 DexYCB dataset 抓取人类抓取动作，retarget 到 Allegro hand（Appendix D）：

retargeting loss:
$$\mathbb{L}(\mathbf{q}_r) = \gamma \|\mathbf{x}_r - \alpha\mathbf{x}_h\|^2 + (1-\gamma)\|\mathbf{x}_r - \mathbf{x}_c\|^2 + \lambda\|\mathbf{q}_r - \mathbf{q}_{reg}\|$$

变量：
- $\mathbf{x}_r \in \mathbb{R}^{12}$: Allegro 4 个指尖的 stacked 位置
- $\mathbf{x}_h \in \mathbb{R}^{12}$: 人类对应指尖位置（scaling $\alpha=1.6$ 因为 Allegro 比人手大）
- $\mathbf{x}_c$: 用来鼓励 power grasp（palm 上的点）或 precision grasp（指尖中心点）
- $\gamma = 1 - \frac{i+1}{n}$: blend factor，trace 开始时全靠 mimic 人类，trace 结束时全靠 grasp target
- $\mathbf{q}_{reg}$: 正则化目标——precision grip 用 opposed thumb + straight fingers，power grip 用 opposed thumb + curled fingers

这个 retargeting 给出 Allegro 上的 grasping motion dataset，然后跑 PCA，发现前 5 个主成分解释 98% 方差（和 [17] 一致）。

### 4.3 最终 action space

$$\widetilde{\mathbf{A}} = [\mathbf{0}, \mathbf{A}] \in \mathbb{R}^{5 \times 23}, \quad \tilde{\mathbf{x}} = \widetilde{\mathbf{A}}\mathbf{q} \in \mathbb{R}^5$$

前 5 列对应 hand（被 PCA 映射），后 7 列是 arm（被 0 padding，不约束 arm joint）。

Attraction fabric（在 PCA taskmap 里）：
$$\ddot{\mathbf{x}} = -k_a \tanh(\alpha_a \|\mathbf{x} - \mathbf{x}_{pca,target}\|) \frac{\mathbf{x} - \mathbf{x}_{pca,target}}{\|\mathbf{x} - \mathbf{x}_{pca,target}\|} - b\dot{\mathbf{x}}$$

$\tanh$ saturation 让远距离时响应弱（避免冲过头），近距离时响应强。

**Palm action**：另外定义一个 21-D taskmap（palm 上 7 个 3D 点），action 是 6-D（3 位置 + 3 欧拉角），transform 成 21-D target。

**总 action space**: 11 维 = 5（PCA hand）+ 6（palm pose）

这种 dimension reduction 极其重要：23-D → 11-D，减少了一半多，且每个维度都有物理意义。

参考:
- DexYCB: https://arxiv.org/abs/2104.04631
- Dexpilot retargeting: https://arxiv.org/abs/2003.05736

---

## 5. RL 训练细节

### 5.1 Asymmetric Actor Critic

这个 trick 来自 Pinto et al. 2018 (RSS)：

- **Critic** $V(\mathbf{s})$: 接收全部 privileged info $\mathbf{s} = [\mathbf{o}_{privileged}, \mathbf{s}_{privileged}]$
- **Teacher policy** $\pi_{privileged}(\mathbf{o}_{privileged})$: 只看有限 observation

`privileged` $\mathbf{s}_{privileged}$ 包含：
- joint forces $\mathbf{f}_{dof} \in \mathbb{R}^{23}$
- fingertip contact forces $\mathbf{f}_{fingers} \in \mathbb{R}^{4 \times 3}$
- true object position $\mathbf{x}_{obj}$
- true object quaternion $\mathbf{q}_{obj}$
- true object linear/angular velocity $\mathbf{v}_{obj}, \mathbf{w}_{obj}$

**为什么 asymmetric**: 训练时 critic 用完美信息估 value，policy 学到的行为不依赖不可观测的信息——这样 distillation 到 student 时，teacher 行为不需要"猜"，可以更好被模仿。

Teacher observation:
$$\mathbf{o}_{privileged} = [\mathbf{o}_{robot}, \mathbf{x}_{goal}, \mathbf{o}_{obj}]$$

$\mathbf{o}_{robot}$:
- $\mathbf{q} \in \mathbb{R}^{23}$: cspace position
- $\dot{\mathbf{q}} \in \mathbb{R}^{23}$: cspace velocity  
- palm 上 3 个 anchor 点位置 $\in \mathbb{R}^{3\times3}$
- 4 个指尖位置 $\in \mathbb{R}^{4\times3}$
- fabric state $[\mathbf{q}_f, \dot{\mathbf{q}}_f, \ddot{\mathbf{q}}_f] \in \mathbb{R}^{3\times23}$

$\mathbf{o}_{obj}$:
- noisy 物体 position $\tilde{\mathbf{x}}_{obj}$
- noisy 物体 quaternion $\tilde{\mathbf{q}}_{obj}$
- **one-hot embedding** $\mathbf{e} \in \{0,1\}^{140}$: 告诉 policy 当前是哪个物体

这个 one-hot embedding 是关键设计——policy 知道"我在抓物体 #47"而非猜测几何。这也意味着部署时 student 必须能从 depth image 推断出"这是哪个物体类型"，或者其实 student 不需要 one-hot（因为 distillation 后 student 用 depth 替代）。

### 5.2 Reward Design with `minimize()` Stateful Function

定义：
$$\text{minimize}(e) = \max(e_{smallest} - e, 0)$$

其中 $e_{smallest}$ 是 episode 内历史最小 error。

只有当 error 低于历史最小值时才给 positive reward，且立刻更新 $e_{smallest}$ 防止"原地踏步"。

Reward terms:
- $r_{to-obj} = \text{minimize}(\|\mathbf{x}_{fingertips} - \mathbf{x}_{obj}\|)$
- $r_{lift} = \text{minimize}(z_{lifted} - z(\mathbf{x}_{obj})) \times (1 - \text{lifted}(\mathbf{x}_{obj}))$
- $r_{lifted}$: 一次性 bonus，首次 lift
- $r_{to-goal} = \text{minimize}(\|\mathbf{x}_{goal} - \mathbf{x}_{obj}\|) \times \text{lifted}(\mathbf{x}_{obj})$
- $r_{reached} = \mathbb{1}(\|\mathbf{x}_{goal} - \mathbf{x}_{obj}\| < d_{success})$
- $r_{success} = \mathbb{1}(r_{reached}=1 \text{ for } T_{success} \text{ consecutive timesteps}) \times (T_{max} - T)$

参数：$z_{lifted} = z_{table} + 0.2m$, $d_{success}=0.1m$, $T_{success}=15$（1 秒）, $T_{max}=150$（10 秒）。

最后一项 $r_{success} \times (T_{max} - T)$ 是非常 clever 的设计：越早成功，剩余时间越多，bonus 越大。这给了 policy "快成功"的激励，避免拖延。

Reward weights 选得让 cumulative reward **递增**：$r_{to-obj}$ 总额 ~2.5, $r_{lift}$ ~10, $r_{lifted}$ 50, $r_{to-goal}$ ~300, $r_{reached}$ max 6000, $r_{success}$ max 15000。这避免了 policy 提前终止 episode 来"卡"某个 reward term。

### 5.3 Environment Modifications for Robustness

1. **Random Wrench Perturbations**: $p=0.1$ 概率施加
   - $\mathbf{f}_{perturb} = f_{scale} m \mathbf{u}_f$ ($f_{scale}=50$)
   - $\boldsymbol{\tau}_{perturb} = \tau_{scale} \mathbf{I} \mathbf{u}_\tau$ ($\tau_{scale}=100$)
   
2. **Pose Noise**: 
   - uncorrelated: 每步采样 $\mathcal{N}(0, 0.02m)$ 位置 / $\mathcal{N}(0, 0.1\text{rad})$ 朝向
   - correlated: 每个 episode 采一次，episode 内固定
   
3. **Friction Reduction**: $\mu = 0.7$（降低，强迫 robust grip）

4. **Domain Randomization**: Table 2 列了 mass scaling [0.3, 3.0]、friction [0.5, 1.1]、joint stiffness loguniform [0.5, 2]、gravity 加噪声等

### 5.4 Curriculum on Cspace Damping

Appendix E.9 的 trick：低 profile 物体（pot、cup）需要手指贴近桌面，但 collision avoidance term 会让 policy 害怕接近桌面。解决：从 cspace damping = 0 开始训练（允许接触、最大探索），success rate > 90% 后 +0.1 damping，重复直到 10。这让 policy 先学会"啥都能碰"，再学会"安全地碰"。

### 5.5 Network Architecture

Teacher:
- Critic: MLP [512, 512, 256, 128] (stateless，因为有 full privileged info)
- Policy: MLP [512, 512] → **LSTM (1024)** → output
- **Skip connection around LSTM** (residual connection)

这个 skip connection 我觉得对你 Andrej 应该很有共鸣——LSTM 训练不稳（BPTT 时梯度爆炸/消失），把 LSTM 输出当作 residual 加到 input 上，相当于给网络"绕过 LSTM"的选项。这跟 ResNet、Transformer 里的 skip 一样思想。

### 5.6 Training Scale

- Isaac Gym, 8192 robots 并行 per GPU
- 4× NVIDIA V100 (32GB each)
- 20k FPS
- 4.7B frames = ~10 年仿真时间
- 68 小时 wall clock
- PPO, lr=5e-4, γ=0.998, clip ε=0.2, horizon=16, 5 mini-epochs/update
- rl_games framework

参考:
- Isaac Gym: https://arxiv.org/abs/2108.10470
- rl_games: https://github.com/Denys88/rl_games
- PPO: https://arxiv.org/abs/1707.06347
- Asymmetric AC: https://www.roboticsproceedings.org/rss14/p08.html

---

## 6. Distillation: Pixels-to-Action Student

Student policy:
$$\pi_{depth}(\mathbf{o}_{depth}) \to (\hat{\mathbf{a}}, \hat{\mathbf{x}}_{obj})$$

输入 $\mathbf{o}_{depth} = [\mathbf{o}_{robot}, \mathbf{x}_{goal}, \mathbf{I}]$，其中 $\mathbf{I} \in [0.5, 1.5]^{160 \times 120}$ m 是 raw depth image。

Loss:
$$\mathcal{L} = \mathcal{L}_{action} + \beta\mathcal{L}_{pos}$$
$$\mathcal{L}_{action} = \|\hat{\mathbf{a}} - \mathbf{a}\|_2, \quad \mathcal{L}_{pos} = \|\hat{\mathbf{x}}_{obj} - \mathbf{x}_{obj}\|_2$$
$\beta = 0.1$

**为什么同时预测 $\hat{\mathbf{x}}_{obj}$**: 给 state machine 用——state machine 用预测的物体 z-coordinate 判断"已抬起，转 transport 状态"。这是把 perception 和 control 整合后顺势产出的 auxiliary head。

### 6.1 Student Architecture

- $\mathbf{o}_{robot}, \mathbf{x}_{goal}$: 3-layer MLP (512, 256, 128) + ELU
- $\mathbf{I}$ (depth): 3 个 Conv2D (16→32→64, kernel=3, stride=1, pad=1) + max-pool (kernel=2, stride=2) + ReLU → 2-layer MLP (128, 128) + ReLU
- Concatenate all encodings
- GRU (1 layer, 1024 units) → predict $\hat{\mathbf{a}}$

用 **GRU 而非 LSTM**（更少参数，省显存）。

整个 student 在 **1× RTX 3090** 上 distill，12 小时 / 140 rollouts × 480 envs / 6.05M frames ≈ 4.67 天仿真时间。

### 6.2 Depth Image Augmentation

仿真渲染的 depth 完全干净（Figure 13 left）。为了跨过 sim2real gap：

1. **Pixel dropout**: $p_{dropout}=0.003$, 设为 0
2. **Random pixel**: $p_{randu}=0.003$, 设为 $(-0.5, -1.3)$ 之间随机值
3. **Stick artifact**: $p_{stick}=0.0025$, 长 ≤18 像素、宽 ≤3 像素的线段（模拟线缆）
4. **Uncorrelated + correlated depth noise**（来自 Handa et al. 2014 ICLR RGB-D benchmark）

还随机扰动相机 placement，给 calibration error 留 margin。

### 6.3 DAgger Online Distillation

Online DAgger（Ross et al. 2011）：
- Student 跑 n=6 步
- BPTT 通过 6 步
- 每 environment done 时 update

时间窗口 0.4 秒，对应 6 个 15Hz control steps。

参考:
- DAgger: https://proceedings.mlr.press/v15/ross11a.html

---

## 7. Joint Constraints via Fabric（B.3）

Fabric 自然支持 closed-form joint acceleration/jerk limit。

QP:
$$L = \frac{1}{2}(\ddot{\mathbf{q}}_f - \ddot{\mathbf{q}})^T \mathbf{M}_f (\ddot{\mathbf{q}}_f - \ddot{\mathbf{q}}) + \frac{\alpha}{2}\ddot{\mathbf{q}}_f^T \mathbf{M}_f \ddot{\mathbf{q}}_f$$

最优解 $\ddot{\mathbf{q}}_f = -(\mathbf{M}_f + \alpha\mathbf{I})^{-1}\mathbf{f}_f$。$\alpha \to \infty$ 时 $\|\ddot{\mathbf{q}}_f\| \to 0$。

通过二分 search 单个 $\alpha$，让所有 joint 的 acceleration/jerk 在 limit 内。Joint jerk limit 由：
$$\overline{\ddot{\mathbf{q}}} = \min\left(\overline{\ddot{\mathbf{q}}}, \frac{\Delta t \overrightarrow{\mathbf{q}}}{2\overline{\ddot{\mathbf{q}}}}\right)$$

Joint position limits 用 repulsion term：metric $\mathbf{M}(\mathbf{x}) = \text{diag}(\max(-\text{sgn}(\dot{\mathbf{x}}), 0) \frac{k_b}{\mathbf{x}})$，只在朝 limit 移动时激活。

---

## 8. Posture Control for Manipulability

因为 action space (11-D) < controlled joints (23-D)，存在 redundancy。用 cspace attractor：

$$\ddot{\mathbf{x}} = -k_a \|\dot{\mathbf{x}}\|^2 \tanh(\alpha_a \|\mathbf{x} - \mathbf{x}_g\|) \frac{\mathbf{x} - \mathbf{x}_g}{\|\mathbf{x} - \mathbf{x}_g\|}$$

注意这是 HD2（$\|\dot{\mathbf{x}}\|^2$ factor）——这个项在 taskmap（PCA + palm pose）已经收敛时**不会干扰**，但在 null space 内会引导 robot 到 elbow-out fingers-curled posture。

$\mathbf{x}_g$ 选 elbow-out + fingers-curled 是为了**kinematic manipulability**——palm 能贴近桌面和 robot base 抓低 profile 物体。

---

## 9. Real-World System Architecture

很分层的 ROS2 node 设计（Figure 2 bottom）：

| Node | Rate | Function |
|---|---|---|
| Joint PD controller (arm) | 1 kHz | track joint targets |
| Joint PD controller (hand) | 333 Hz | track joint targets |
| Geometric fabric | 60 Hz | integrate fabric, output joint targets |
| FGP (policy) | 15 Hz | output 11-D action to fabric |
| Depth camera | ~30 Hz | stream depth to FGP |

这个分层解耦很关键：即使 FGP node 崩了或 model 跑飞，fabric node 仍在 60 Hz 持续输出安全命令。论文强调："over many hours of testing DextrAH-G (and a variety of ill-behaved FGPs), no hardware was damaged." 这是 fabric 的最大价值。

---

## 10. 实验结果深度分析

### 10.1 Simulation Per-Object Performance

训练 140 物体：
- Teacher $\pi_{privileged}$: 85% per-object success
- Student $\pi_{depth}$: 80% per-object, 99% per-batch

per-batch vs per-object 差 19% 是因为成功 episode 后 reset，简单物体被更频繁尝试。

### 10.2 Single Object Assessment (Table 1)

11 个 standard test objects，每个 5 trials：

| Object | DextrAH-G | DexDiffuser | ISAGrasp | Matak |
|---|---|---|---|---|
| Pitcher | 80% | - | - | 67% |
| Pringles | 100% | 60% | 60% | 100% |
| Coffee | 100% | - | - | 67% |
| Container | 100% | - | 40% | - |
| Cup | 80% | 60% | - | 0% |
| Cheezit | 100% | 80% | 80% | 0% |
| Cleaner | 100% | 100% | - | 100% |
| Brick | 100% | - | - | 100% |
| Spam | 100% | - | - | 0% |
| Pot | 100% | - | 80% | - |
| Airplane | 60% | 20% | - | - |

DextrAH-G 基本全胜。

### 10.3 Bin Packing Assessment（核心 metric）

30 个物体，连续抓取：
- **CS (consecutive success)**: $6.56 \pm 2.41$ (mean ± 95% CI)
- **Cycle time**: $10.66 \pm 0.84$ s = **5.63 picks-per-minute (PPM)**
- **Success rate**: 87% over 256 attempts
- 最高连续 27 次成功

对比人类估计 16.53 PPM（Boothroyd-Dewhurst tables）。DextrAH-G 已经达到人类 1/3 速度。

### 10.4 Failure Modes (Figure 15)

- Push out of region: 8%
- Repeated miss on hard object: 3%
- Loose grip drop: 1%
- Poor grasp placement drop: 1%

最难物体：small bottle（小）、green cup（slippery）、pot（大）、apple（rolling）、sanitizer bottle（transparent）。

### 10.5 Speed Distribution (Figure 18)

Median 8 秒 / pick，42% 在 6.0-7.3 秒区间。

---

## 11. 跟你 Andrej 工作可能的联想点

### 11.1 与 Asymmetric Actor Critic 类似 "implicit learning"

你在 nanoGPT / teaching 里强调"网络只学到必要的 representation"。Asymmetric AC 是把这个思想应用到 RL——critic 有更多信息但 policy 没有，policy 不会学到 "依赖 ground truth object pose"这种部署时崩盘的策略。

### 11.2 Teacher-Student = Knowledge Distillation with State

DAgger 是 imitation learning 经典。这里 student 用 GRU 替代 teacher 的 LSTM，input 从 privileged state 切到 depth。这跟你讲 RNN 时强调的"state is implicit memory"完全对应。

### 11.3 PCA Action Space = learned latent

5-D PCA action 等价于一个**固定的 linear autoencoder**。这让人联想到 VAE latent、Diffusion policy 的 latent action。但 PCA 是 deterministic、pre-trained on human data——这种 "frozen prior" 思想在 23-DoF 系统中比让 RL 从头学 latent 更高效。

参考 Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 11.4 Skip Connection Around LSTM

这个跟 ResNet、Transformer 的 skip 完全一个逻辑——"let gradient flow, let the model choose to use the layer"。在 micrograd 风格看，就是 $y = x + \text{LSTM}(x)$ 而非 $y = \text{LSTM}(x)$。

### 11.5 Minimize() Reward 是 Implicit Curriculum

只奖励"打破历史最佳"。这避免了 reward shaping 的局部最优（站在原地不动）。跟 self-imitation learning、asymmetric self-play 思想相通——"past self" 是 baseline。

---

## 12. 局限与未竟之事

Paper 自己点出：
1. **PCA taskmap 限制 dexterity**: power/precision grasp 之外做不到（比如 in-hand reorientation）
2. **Obstacle avoidance 仍 model-based**: 应该让 RL 学 sensory-based avoidance
3. **RL 探索困难近 high-cost region**: fabric collision 排斥让 policy 不敢靠近桌面，低 profile 物体差
4. **单物体场景**: 不能处理 clutter，需要 segmentation

我额外想到：
- **One-hot object embedding 是 sim-only 概念**: real-world deploy 时 student 用 depth 替代，但 teacher 学到的是"知道是哪个物体"的行为。distillation 实际上让 student 学到"从 depth 推断 object identity + action"，但 evaluation 用了部分 training objects 做 real test。generalization 程度其实没有完全 stress test。
- **Allegro Hand 限制**: 4 finger, 16 DoF, 无 tactile sensing。paper 完全靠 vision + proprioception，没有触觉反馈。这是 sim-to-real 的一个盲区。
- **No tactile feedback**: 如果加上 tactile，可能 drop rate 能进一步降。
- **Single camera**: 单 depth 视角易 occlude。当 robot 抓住物体后，depth 几乎看不到物体，全靠 GRU memory 维持估计。这是为什么 paper 强调 state-based model 重要性。

---

## 13. 跟相关 paper 的纵向联系

- **OpenAI Rubik's Cube (2019)**: 类似 sim2real + domain randomization，但用 24-DoF Shadow Hand + LSTM policy + PPO。DextrAH-G 更结构化（fabric + PCA action space），用更少算力达到更高 PPM。
- **Dextreme (Handa 2022)**: in-hand reorientation，类似的 teacher-student + sim2real。DextrAH-G 是 grasping 而非 reorientation。
- **DexNet (Mahler 2017)**: grasp pose prediction 经典，但 model-based planning。DextrAH-G 是 continuous control。
- **Visual Dexterity (Chen 2023)**: 用 depth + point cloud 做 in-hand reorientation。DextrAH-G 借用了它的 object dataset。
- **DexPoint (Qin 2022)**: point cloud conditioned dexterous manipulation。
- **RMP2 (Li 2021)**: structured composable policy，geometric fabrics 的前身。
- **DexDiffuser (2024)**: diffusion-based grasp generation baseline，比 DextrAH-G 慢且成功率低。

参考链接汇总：
- DextrAH-G 项目页: https://sites.google.com/view/dextrah-g
- Geometric Fabrics: https://arxiv.org/abs/2405.02250
- Visual Dexterity: https://www.science.org/doi/abs/10.1126/scirobotics.adc9244
- Dextreme: https://arxiv.org/abs/2210.13702
- OpenAI Rubik: https://arxiv.org/abs/1910.07113
- Isaac Gym: https://arxiv.org/abs/2108.10470
- YCB Object Set: https://arxiv.org/abs/1502.03143
- Asymmetric AC: https://www.roboticsproceedings.org/rss14/p08.html
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- PPO: https://arxiv.org/abs/1707.06347
- DexYCB: https://datasets.k2si.com/
- Dexpilot: https://arxiv.org/abs/2003.05736

---

## 14. 总结：Intuition

如果让我用一句话给你 Andrej 总结这篇 paper 的精髓：

**"用结构化的动力学先验压缩 action space 和 behavior space，让 RL 只学最关键的 task policy；用 sim 的算力 + asymmetric AC 训 teacher；用 DAgger 把视觉感知蒸馏进 student。Fabric 提供安全网让 RL 可以放心探索、student 可以放心部署。"**

这个 formula 是不是可以推广到其他高维机器人任务？我觉得 legged locomotion 已经有类似（RMA、Daydreamer），但 dexterous manipulation 因为接触丰富更难，这篇 paper 展示了 fabric + PCA action space + distillation 这个组合的威力。

下一个 frontier 应该是：**让 fabric 本身也 learnable**（Neural Geometric Fabrics，Xie et al. 2023），同时 fabric 提供 structure，neural net 提供 adaptation。这正是 NVIDIA 这一系列 paper 的走向。

参考 Neural Geometric Fabrics: https://arxiv.org/abs/2303.02502

希望这个 walkthrough 帮你 build intuition。如果对某个具体细节（比如 collision metric 设计、PCA variance 解释、GRU vs LSTM 选择）想更深入聊，可以继续问。
