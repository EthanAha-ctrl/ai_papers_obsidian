---
source_pdf: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation.pdf
paper_sha256: 022f551cef2a5b1f54f26a651fc8fa2832e85ed0a12069f102db953981948eb3
processed_at: '2026-08-13T02:23:32-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 如果用最接地气的人话来讲，就是教一个“瞎子”（只有内部传感器数据的机器人）怎么在虚拟世界里学会走路和拿东西，然后再把这个经验传授给一个“只长了眼睛”的机器人，最后让它直接在现实世界里干活，完全不需要人类手把手教。

下面我按逻辑顺序把这里面的门道给你揉碎了讲，尽量帮你 build intuition。

### 1. 为什么这件事这么难？

你让一个人形机器人去桌子上拿个杯子，再放到另一个地方，这叫 loco-manipulation。在 real world 里，机器人没法像人一样靠感觉随意调整。目前主流的做法是人类戴上 VR 头显去“遥操作”机器人，但这极其昂贵且费时。如果纯粹在现实里收集数据来训练，成本高到无法承受。

Simulation（仿真环境）就不一样了，想要多少数据就有多少，还不用怕摔坏机器人。但是，在 Sim 里面看起来完美的画面和物理规律，一旦搬到 Real 现实里，就会因为光线、摩擦力、摄像头位置的细微差别而彻底崩溃。之前大家能在 Sim 里练练走路，或者站在原地抓个方块，但把走路和抓东西结合起来、并且只靠头顶的一个 RGB 摄像头去判断位置，这还是个未被攻克的难题。VIRAL 就是想把这个流程彻底打通。

### 2. 两步走的“师生架构”

如果直接让一个带着摄像头的机器人去从零开始试错学这么复杂的动作，它会像无头苍蝇一样，永远得不到奖励信号。所以作者用了 Teacher-Student 架构。

**Phase 1: 训练“开了挂”的 Teacher**
在仿真环境里，先训练一个 Teacher policy。这个 Teacher 是“开了挂”的，它能看到所有真实数据。比如它知道目标物体在三维空间里的精确坐标 $(x, y, z)$，知道桌子在哪里。
$$o_t^{\text{priv}} = [o_t^{\text{prop-priv}}, o_t^{\text{exte-priv}}]$$
这里 $o_t^{\text{exte-priv}}$ 就是 Sim 里的上帝视角信息（包括目标位置 $\mathbf{T}_t$ 和相对变换 $\mathbf{O}_t$）。因为开了全图视野，Teacher 很容易通过 PPO 强化学习算法学会怎么走、怎么抓。

**Phase 2: 蒸馏出“只能看视频”的 Student**
Real robot 不可能有上帝视角，它只有自己头顶的 Intel RealSense 摄像头拍的一张 108×192 的 RGB 图像，以及自己的关节角度。所以我们要训练一个 Student，输入只有 RGB 图像和本体感觉（proprioception）。Student 的目标就是：看着眼前的画面，模仿 Teacher 刚才输出的动作。
$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{o_t \sim \rho^o}\left[\|\pi_{\text{teacher}}(o_t^{\text{teacher}}) - \pi_{\text{student}}(o_t^{\text{student}})\|_2^2\right]$$
这个公式的意思是，Student 输出的动作 $\pi_{\text{student}}$ 要尽可能逼近 Teacher 在同样状态下输出的动作 $\pi_{\text{teacher}}$，用均方误差（MSE）来衡量。

### 3. Teacher 练功的三个“外挂”

Teacher 虽然开了全图视野，但人形机器人有几十个自由度，动作空间太大，直接学很容易练不下去。作者用了三个极其有效的 trick：

**Trick 1: Delta Action Space（只学增量）**
过去做机器人腿部控制，通常让 RL 直接输出关节的绝对目标角度（比如抬腿到 30 度）。但在 loco-manipulation 中，手臂需要微调，输出绝对值太难收敛了。VIRAL 让 Teacher 输出的是 **变化量（Delta）**：
$$a_t = (\Delta \mathbf{v}_t, \Delta \omega_t^{\text{yaw}}, \Delta \mathbf{q}_t^{\text{arm}}, \Delta \mathbf{q}_t^{\text{finger}})$$
这里的 $\Delta$ 就是 delta，意思是“在现在的指令基础上加一点点”。比如 $\Delta \mathbf{q}_t^{\text{arm}}$ 就是手臂关节角度的增量。这种做法让 policy 的搜索空间瞬间变小，收敛极快。

**Trick 2: Reference State Initialization (RSI，参考状态初始化)**
如果一个机器人从站立状态开始学“走过去-抓东西-转身”，它可能在学会抓东西之前，永远碰不到抓东西的奖励，最后啥也学不会。作者收集了 200 次人类遥操作的录像，每次 reset 的时候，直接把机器人和物体的状态“快进”到录像中的某一个随机中间点。
这就好比你学打游戏，每次开局都直接把你扔到 Boss 面前，你不需要从第一关打起，你能直接练习打 Boss 的动作。这个 trick 让 success rate 从不到 10% 飙升到了 95%。

**Trick 3: 把低层动作打包给 WBC**
Teacher 不需要去想“我要怎么控制电机电流让脚不摔倒”。团队用了一个预训练好的 Whole-Body Controller (WBC) 叫 HOMIE。Teacher 只需要输出高层的速度指令和手臂目标，底层的平衡全交给 HOMIE 去做。这样 Teacher 只需要专注“去哪、手怎么动”，大大降低了维度。

### 4. Student 怎么学才不会变傻？

Student 看着画面模仿 Teacher，这叫 Behavior Cloning (BC)。但 BC 有个致命弱点：如果 Student 犯了个小错，导致它处于一个它没见过的画面状态，它就不知道怎么办了，错误会滚雪球一样越来越大。

所以作者用了一半 BC + 一半 DAgger 的混合策略，比例 $\alpha = 0.5$。
$$\rho^o \triangleq \alpha \rho_{\pi_{\text{teacher}}}^o + (1 - \alpha) \rho_{\pi_{\text{student}}}^o$$
- 当 $\alpha = 1$ 时，全看 Teacher 走的完美轨迹（BC），学得快但很脆。
- 当 $\alpha = 0$ 时，全看 Student 自己走的轨迹（DAgger），Student 能看到自己犯错时的画面，并学习 Teacher 会怎么去纠正。但早期 Student 太烂，收集到的都是垃圾数据。
- $\alpha = 0.5$ 就是 Sweet spot，既学了完美动作，又学会了“救场”。

此外，Student 的视觉编码器用了极其强大的 DINOv3，把图片变成 128 维的 feature，这让 Student 能在 Sim 和 Real 不同的视觉风格中找到共性。

### 5. 从 Sim 搬到 Real 的“土办法”

即使 Student 在 Sim 里练得再好，Real world 的光照、机械误差还是会搞砸一切。作者的思路很简单：**既然没法做到一模一样，那就让 Sim 里的世界足够乱。**

**狂加 Domain Randomization（域随机化）**
在 Sim 里训练时，每次画面都不一样：灯光从 800 到 2000 lux 乱变，光照方向 360 度转，桌子的材质从木头变到地毯，机器人的反光度也乱变，甚至给图像加上高斯模糊、噪点、改变对比度。连摄像头的安装位置都加了 $\pm 5$ 厘米和 $\pm 0.1$ 弧度的随机噪声。
这样逼出来的 Student，对环境的细微变化完全免疫。

**硬核调校机械手**
Unitree G1 的三指灵巧手用的是高减速比电机，Sim 里的模型跟现实差了十万八千里。作者做了一个 SysID（系统辨识）：在现实里抓一下，在 Sim 里重放同样的指令，然后慢慢调 Sim 里的 stiffness 和 damping 参数，直到 Sim 里的关节轨迹跟现实完全重合。

**算力就是正义**
最残酷的现实是：如果你的 GPU 不够多，以上全白搭。
Figure 14 显示，用 1-2 块 GPU 训练 Teacher，成功率死活上不去；用 8-16 块 GPU 才能冲破 90%。
Figure 15 显示，Student 训练从 1 块 GPU 加到 64 块 GPU（8 节点），收敛速度飙升，最终表现也更好。
为什么？因为视觉状态空间太大了。几千个环境同时跑，才能让 policy 见识到足够多的画面变异，防止它过拟合到某一个模拟器的特定视觉 bug 上。Scale 在这里就是 necessary condition。

### 6. 结果与作者的诚实反思

在真实世界的 59 次连续测试中，VIRAL 成功了 54 次（91.5%），单次循环耗时 20.2 秒。这甚至比一个有 1000 小时经验的专家遥操作员还要快一点（专家 21.4 秒，成功率 100%）。

但是，这篇 paper 最精彩的其实是 Section 6 的 Limitations。作者非常清醒地泼了一盆冷水：**Sim-to-real 的天花板可能很快就到了。**

原因有四个：
1. **物理盲区**：你能花几个月在 Sim 里完美模拟“舀米饭”，但你没法把现实世界里成千上万种物体的阻尼、摩擦力、软硬程度全部建模出来。工程量太大了。
2. **任务盲区**：你能模拟洗碗，但你没法穷举家里所有的家务，更没法模拟“宠物突然跑过来干扰”这种未知的边缘情况。
3. **Reward 难写**：给一个任务设计强化学习的 Reward 是门玄学，太密了机器人会投机取巧，太稀疏了它学不到。这无法 scale 到成千上万个任务上。
4. **硬件鸿沟**：灵巧手在现实里有很多摩擦、背隙、发热降频的问题，在 Sim 里完美执行的动作，到了现实里常常抓不住。

所以作者最终的 outlook 是：未来的 general-purpose loco-manipulation，绝对不是 Sim 全能包打天下，而是 Sim-to-real 与 real-world imitation learning（比如现在的 VLA 大模型 $\pi_0.5$）互相结合、互补短板。Sim 负责提供海量的基础常识，Real data 负责补齐长尾的物理和任务细节。

**Reference Links:**
- VIRAL Project Page: https://viral-humanoid.github.io
- DAgger Original Paper (Ross et al.): https://arxiv.org/abs/1011.0686
- DINOv3 Visual Backbone: https://arxiv.org/abs/2508.10104
- HuggingFace TRL (Training Framework): https://github.com/huggingface/trl
- HOMIE WBC Paper: https://arxiv.org/abs/2502.13013

---

# VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation 详解

Paper link: https://viral-humanoid.github.io
arXiv: https://arxiv.org/abs/2502.20396 (相关Lin et al.工作)

## 1. 核心问题与动机

Humanoid robot 部署到 real-world 的核心障碍是缺少 autonomous loco-manipulation skill——即 locomotion（行走）与 manipulation（抓取放置）tight coordination 的 long-horizon 能力，并且必须依赖 onboard perception（机载感知，RGB camera）。VIRAL 想回答一个 question：**能不能完全在 simulation 里训练一个 RGB-driven policy，然后 zero-shot 部署到 real humanoid 上做连续的 loco-manipulation？**

为什么这个 question 重要：
- Real-world teleoperation data collection 对于 humanoid 极其昂贵（高 DoF、safety constraint、teleop stack 工程开销），如果把 humanoid mobile manipulation 当作纯 data problem，data 规模是 prohibitive 的
- Simulation 提供了 marginal cost 极低的数据来源，已经在 locomotion sim-to-real 上成熟，但 manipulation sim-to-real 还基本停留在 tabletop setting
- Locomotion 与 manipulation 通常被 isolated 研究，缺少 end-to-end 的 vision-based loco-manipulation 系统

Intuition：VIRAL 不发明新 algorithm，而是构建一个 full-stack recipe，回答"哪些 design matter、在哪里 fail、它们之间如何 interact"。

---

## 2. 整体框架：Teacher-Student Privileged Learning

参考 Learning by Cheating (Chen et al., CoRL 2020): https://arxiv.org/abs/1911.00298

### Phase 1: Privileged RL Teacher
Teacher $\pi_{\text{teacher}}(a_t | o_t^{\text{priv}})$ 拿到 **full privileged state**（含 ground-truth object pose、table transform 等 sim-only 信息），用 PPO 学 long-horizon loco-manipulation。在 2 个 8-GPU L40S node（16 GPU）上训练，不渲染 visual frame，纯 state-based RL，所以 throughput 高。

### Phase 2: Vision-based Student Distillation
Student $\pi_{\text{student}}$ 只看 RGB image + real-available proprioception，通过 DAgger + BC mixture 从 teacher 蒸馏。在 8 个 8-GPU node（64 GPU）上用 Isaac Lab 的 tiled rendering 大规模训练。

关键 insight：teacher 不需要 sim-to-real，因为它只在 sim 里跑；student 不需要 reward engineering，因为它只是 imitate teacher 的 action。这种 decomposition 把"reward 难设计"与"vision 难 transfer"两个难题拆开。

---

## 3. Teacher Policy 细节

### 3.1 Action Space：Delta Action
$$a_t = (\Delta \mathbf{v}_t, \Delta \omega_t^{\text{yaw}}, \Delta \mathbf{q}_t^{\text{arm}}, \Delta \mathbf{q}_t^{\text{finger}})$$

变量含义：
- $\Delta \mathbf{v}_t \in \mathbb{R}^2$：base 线速度命令的 delta（x, y 方向增量）
- $\Delta \omega_t^{\text{yaw}} \in \mathbb{R}$：yaw 角速度命令的 delta
- $\Delta \mathbf{q}_t^{\text{arm}} \in \mathbb{R}^7$：arm 关节目标位置的 delta
- $\Delta \mathbf{q}_t^{\text{finger}} \in \mathbb{R}^4$：finger 关节目标位置的 delta（Unitree 3-finger hand）

这些 delta 累加成 WBC command，传给 pre-trained HOMIE WBC policy。WBC 负责低层 motor control，teacher 只学 high-level command。这种 design 把 action 限制在 WBC 的 safe region 内，提升 deployability。

**为什么 delta 比 absolute 好**（ablation Figure 9）：absolute joint target 让 policy 必须从 absolute pose 空间搜索，梯度信号 noisy；delta action 本质是 residual / velocity-like command，policy 只需学增量方向，sample efficiency 显著提升。这跟 legged locomotion RL 习惯用 absolute target 是相反的——因为 loco-manipulation 的 arm 需要精细的 incremental 调整。

### 3.2 Privileged Observation
$$o_t^{\text{priv}} = [o_t^{\text{prop-priv}}, o_t^{\text{exte-priv}}]$$

Proprioception（teacher 版，含真值）:
$$o_t^{\text{prop-priv}} = [\mathbf{v}_t, \omega_t, \mathbf{g}_t, a_{t-1}, \mathbf{q}_t, \dot{\mathbf{q}}_t, \mathbf{f}_t^{\text{finger}}]$$

- $\mathbf{v}_t$：base linear velocity（3 维）
- $\omega_t$：base angular velocity（3 维）
- $\mathbf{g}_t$：projected gravity（3 维，用来判断 body orientation）
- $a_{t-1}$：上一 step action（31 维，用来做 action smoothing）
- $\mathbf{q}_t, \dot{\mathbf{q}}_t$：joint position 与 velocity（各 43 维，含 finger）
- $\mathbf{f}_t^{\text{finger}}$：fingertip force（12 维，4 finger × 3 axis）

Exteroception（teacher 独享的 privileged 信息）:
$$o_t^{\text{exte-priv}} = [e_t, \mathbf{T}_t, \mathbf{O}_t]$$

- $e_t$：当前 stage（one-hot，5 个 stage：walk / pre-place / place / grasp-lift / turn）
- $\mathbf{T}_t$：placement 与 lift target 的位置
- $\mathbf{O}_t$：object 与 table 相对 robot 的 transform（9 维 = 3D position + rotation flatten）

总维度 226（Table 1）。

### 3.3 Stage-based Reward
任务被 decompose 成 5 个 stage 的 loop：walk → pre-place → place → grasp-lift → turn，然后重复。每 step reward：
$$r_t = \sum_{i=0}^{4} w_i \mathbb{1}[s_t = i] r_t^{(i)}, \quad w_i > 0$$

$s_t$ 是当前 stage，$w_i$ 是 stage weight，$r_t^{(i)}$ 是 stage-specific shaping reward。这种 stage-gated 设计避免 reward 互相干扰（比如 walk stage 不应该被 grasp reward 干扰）。

关键 reward 项（Table 3）：

1. **Walking toward object**：
$$r_{\text{walk}} = \exp(-4(\|p_{\text{robot}} - p_{\text{GraspObj}}\| - 0.45)^2)$$
$p_{\text{robot}}, p_{\text{GraspObj}}$ 是 robot pelvis 与目标 object 的位置。0.45 是 ideal grasping distance，exp wrapping 让 reward 在接近目标距离时 saturation，避免 policy 冲过去撞 object。

2. **Place object**：
$$r_{\text{place}} = -\|\mathbf{f}_{\text{PlaceObj}}\| \cdot \mathbb{1}(\|p_{\text{PlaceObj}} - p_{\text{tray}}\| < 0.3)$$
$\mathbf{f}_{\text{PlaceObj}}$ 是 finger 与 place object 之间的接触力。只有当 object 离 tray 0.3m 内时才惩罚接触力——防止 finger 把 object 推走。

3. **Grasp lift**：
$$r_{\text{grasp-z}} = \min(h_{\text{GraspObj}} - h_{\text{table}}, 0.15), \quad r_{\text{grasp-goal}} = \exp(-10\|p_{\text{GraspObj}} - p_{\text{goal}}\|^2)$$
$h$ 是高度。grasp-z 鼓励把 object 抬起，但 cap 在 0.15m（一旦抬起足够高就不再额外 reward）；grasp-goal 引导 object 到 lift target position。

4. **Turn**：
$$r_{\text{turn}} = -|y_{\text{robot}} - y_{\text{desired}}|$$
$y$ 是 yaw heading angle。turn 完成后回到 walk stage，loop 继续。

还有大量 penalty：termination (-2000)、action rate、DoF velocity、torque limit、output smoothness、finger primitive limit 等。这种 dense reward + heavy penalty 的组合是 sim-to-real RL 的标准 recipe，目的是让 policy 输出 smooth、safe 的 trajectory。

### 3.4 Reference State Initialization (RSI)
收集 200 个 teleoperated simulation demonstration，作为 state-initialization buffer。每个 episode reset 时，sample 一个 demonstration snapshot，把 robot / object / table 都 reset 到那个 snapshot 状态。

**Intuition**：long-horizon task 从 scratch 学几乎不可能，因为 policy 在学会 grasp 之前永远走不到 grasp stage，reward 信号缺失。RSI 让 policy 从 task trajectory 的任意中间点开始 practice，相当于 curriculum learning 的逆向版本——直接 expose 到 rewarding state，再逐步学会如何 reach 这些 state。这跟 DeepMimic (Peng et al., 2018) https://arxiv.org/abs/1804.02717 的 idea 类似。

Ablation Figure 9 显示：无 RSI 的 teacher success rate 卡在 10% 以下，有 RSI 达到 95%。

---

## 4. Student Policy 细节

### 4.1 DAgger + BC Mixture
蒸馏 objective：
$$\rho^o \triangleq \alpha \rho_{\pi_{\text{teacher}}}^o + (1-\alpha) \rho_{\pi_{\text{student}}}^o$$
$$\mathcal{L}_{\text{distill}} = \mathbb{E}_{o_t \sim \rho^o}\left[\|\pi_{\text{teacher}}(o_t^{\text{teacher}}) - \pi_{\text{student}}(o_t^{\text{student}})\|_2^2\right]$$

变量含义：
- $\rho_{\pi_{\text{teacher}}}^o$：teacher rollout 产生的 observation distribution（"干净"的 near-optimal trajectory）
- $\rho_{\pi_{\text{student}}}^o$：student rollout 产生的 observation distribution（含 student 自己的错误）
- $\alpha \in [0,1]$：BC vs DAgger 的混合比例
- $o_t^{\text{teacher}}$：teacher 看到的 privileged observation
- $o_t^{\text{student}}$：student 看到的 RGB + proprioception

**关键 intuition**：
- $\alpha=1$（纯 BC）：student 只看 teacher 的"完美"trajectory，loss 下降快，但部署时一旦 student 犯小错，就进入 OOD state，无人纠正，compounding error 灾难性
- $\alpha=0$（纯 DAgger）：student 看 student rollout，能学到 error correction，但早期 student 太差，rollout 都是 garbage，学得慢
- $\alpha=0.5$：BC 提供快速 prior，DAgger 提供 error recovery，两者互补

Ablation Figure 11：纯 BC 在 Isaac-to-MuJoCo transfer 与 real-world 上都崩，$\alpha=0.5$ 的 mixture 是 sweet spot。这跟 DAgger 原始论文 (Ross et al., 2011) https://arxiv.org/abs/1011.0686 的理论一致：DAgger 通过 on-policy data collection 解决 covariate shift。

### 4.2 Vision Backbone
用 DINOv3 (Simeoni et al., 2025) https://arxiv.org/abs/2508.10104 作为 image encoder，输出 128 维 visual feature，与 proprioception concat 后送入 policy head。

Student observation（Table 2，共 113 维 state + 108×192 RGB）：
- 去掉了 finger 相关的 proprioception（real robot 上 finger force sensing 不可靠）
- 去掉了所有 privileged exteroception（object pose、table transform 等）

Ablation Figure 10：DINOv3 比 ResNet 等传统 backbone 显著更好——pre-trained visual representation 对 sim-to-real 至关重要，因为 sim 渲染的 image distribution 与 real image 之间有 gap，strong pre-trained feature 能 bridge 这个 gap。

### 4.3 History Architecture
对比 single-step MLP、feed-forward history、LSTM。Ablation Figure 12 显示 history-aware model 一致优于 single-step。原因：sim-to-real 的 sensor delay、partially observable 的 object pose（被遮挡时）需要 temporal context 来推断。

### 4.4 Distributed Training System
用 TRL (von Werra et al., 2020) https://github.com/huggingface/trl + Accelerate (Gugger et al., 2022) https://github.com/huggingface/accelerate 跨 GPU/node 分布式训练。Teacher 32768 envs（2048×8 GPU×2 node），Student 65535 envs（1024×8 GPU×8 node）。

**Scaling 是 critical**：
- Figure 14：Teacher 从 1 GPU 到 16 GPU，success rate 从卡在低水平到 >90%。早期甚至 super-linear speedup——更多 parallel env → 更宽 state-space coverage per wall time → 更快 discover rewarding behavior
- Figure 15：Student 从 1 GPU 到 64 GPU，distillation loss 更快收敛，success rate 更高更稳。原因是大规模 experience collection 提供 richer state coverage，对 vision policy 的 robustness 至关重要

Intuition：visual sim-to-real 的 state space 极大（image 是高维），少量 env 的 rollout 无法 cover 足够的 visual variation，policy 容易 overfit 到某个 visual pattern。scale 是 cover 这个 space 的必要条件。

---

## 5. Sim-to-Real Transfer 三要素

### 5.1 Dexterous Hand SysID
Unitree G1 的 3-finger hand 用 high gear ratio motor，sim-to-real mismatch 严重（不像 QDD locomotion actuator 那么好 model）。

方法：定义 real-world grasp-release primitive，在 real robot 上 replay 一个 action sequence，记录 joint trajectory；在 sim 里 replay 同样 action，tune finger 的 armature / stiffness / damping 参数，让 sim joint trajectory match real measurement。Figure 5 显示 SysID 后 alignment 显著改善。

### 5.2 Camera Extrinsics Alignment + Randomization
Camera intrinsics 按 manufacturer spec match。但 extrinsics（安装位置）每台 robot 不同，甚至会 drift。所以：
1. Lightweight real-to-sim calibration：visual matching rendered 与 real image
2. Training 时 randomize camera extrinsics（Table 7：position noise ±2-5cm，rotation noise ±0.05-0.1 rad）

### 5.3 Visual & Physical Randomization
Table 7 完整列出：
- **Image augmentation**：brightness、contrast、hue、saturation、Gaussian noise、Gaussian blur——模拟 camera sensor noise 与 lighting variation
- **Dome lighting**：intensity 800-2000 lux、yaw rotation 全范围、texture map 从 Indoor/Clear/Cloudy/Night/Studio 中 random
- **Material**：robot/floor/table/object 的 roughness、metallic、specular 全部 randomize，texture 从 Wood/Carpet/Masonry/Metals/Stone 等 random
- **Table physical**：height、depth、width、thickness 都 randomize，模拟不同 table geometry
- **Camera extrinsics**：position + rotation noise

Ablation Figure 13：去掉全部 randomization success rate 下降 35.1%，去掉任意单个 component（M/D/E）都 degrade，说明这些 randomization 是 complementary 的。

Intuition：sim 的渲染永远不可能完全 photorealistic match real。Domain randomization 的 logic 是——与其让 policy overfit 到 sim 的某个 visual appearance，不如让 policy 见过足够多的 appearance variation，从而在 real 的某个 specific appearance 上也能 generalize。这是 OpenAI Dactyl (Andrychowicz et al., 2020) https://arxiv.org/abs/1808.00177 与 Dextreme (Handa et al., 2023) https://arxiv.org/abs/2210.13702 的同款思路，但 VIRAL 把 scale 推到了 humanoid loco-manipulation。

---

## 6. Real-World Results

### 6.1 Robustness
Unitree G1（29 DoF body + 7 DoF × 2 dexterous hand），Intel RealSense D435i，RTX 4090 推理。连续 loco-manipulation（walk-place-grasp-turn 循环）59 次 trial，成功 54 次（91.5%）。

对比 teleoperator：
- Expert（1000+ 小时 G1 经验）：100% success，21.4s/cycle
- VIRAL：91.5% success，20.2s/cycle（比 expert 还快一点）
- Non-expert（1 小时经验）：73% success，更慢

VIRAL 接近 expert 水平且更快，远超 non-expert。这意味着 autonomous policy 可以在 assisted teleoperation 场景下 reduce human workload。

### 6.2 Generalization
系统性 vary：tray position、robot start pose、table height、table type、tablecloth color、lighting、object category。VIRAL 都能 complete task without tuning。归因于 domain randomization + RL 的 state-space coverage。

### 6.3 Object Generalization
Figure 16：single-object training（cylinder only）vs multi-object training（10 objects）。测试时在 10 objects 上 evaluate，multi-object policy 在每个 category 上都更好。说明 object diversity in training 对 grasp generalization 重要。

---

## 7. Limitations 与 Outlook（Section 6 很精彩）

作者识别出 sim-to-real 在 general-purpose loco-manipulation 上的 4 个 coverage gap：

### 7.1 Physics Coverage Gap
Simulator 理论上能 model deformable body、fluid-structure interaction，但工程 effort 不可 scale。可以 bespoke tune 一个"舀米饭"场景，但无法 scale 到 daily life 的长尾（每个 cardboard box 的 damping、每件衣服的 stiffness、每种食物的 granular mechanics）。Engineering cost 可能 exceed 直接 collect real data。

### 7.2 Task Coverage Gap
即使 physics 完美，task generation 也是瓶颈。洗碗需要 model 不止 geometry，还有 affordance、state（dirty/clean）、interaction logic。Scale 到 household 的 thousands of chores 是 massive content generation bottleneck。而且 simulation 受限于 human imagination，无法 simulate "unknown unknowns"（宠物的干扰、行动不便的用户等 edge case）。

### 7.3 Reward Engineering Bottleneck
"Goldilocks" regime——dense enough 引导 exploration，sparse enough 避免 specification gaming——对 single task 可行，对 thousands of tasks intractable。Imitation learning 通过 human demonstrator 隐式提供 reward，bypass 这个 problem。这是为什么几天高质量 teleop data 常常 beat 几个月 sim-to-real engineering。

### 7.4 Hardware-Simulation Gap
QDD actuator for locomotion 相对 well-modeled，但 dexterous hand 有 unmodeled friction、backlash、thermal throttling、sensor noise。依赖 precise finger positioning / force feedback 的 policy 在 real hardware 上常 fail。

### Outlook
作者认为 sim-to-real 不会 replace real-world learning，而是 complement。下一个 frontier 是 integrate sim-to-real with real-world imitation learning + foundation model，发现 synergy。这跟近期 π0.5 (Physical Intelligence, 2025) https://arxiv.org/abs/2504.16054、GR00T N1 (Bjorck et al., 2025) https://arxiv.org/abs/2503.14734 等工作的方向一致。

---

## 8. 我的 Intuition Building 总结

这篇 paper 的核心 takeaway 我觉得有几点：

**第一，scale 是 sim-to-real 的 necessary condition，不是 nice-to-have**。Figure 14/15 的 scaling curve 很 striking——1-2 GPU 的 teacher 卡在低 success，8-16 GPU 才到 90%+。这说明 humanoid loco-manipulation 的 state space 太大，少量 rollout 无法 cover。这对整个 sim-to-real community 是个重要 data point：visual sim-to-real 不只是 algorithm 问题，是 compute 问题。

**第二，teacher-student decomposition 把两个 hard problem 拆开**。Reward engineering 在 privileged state space 下 tractable（teacher），vision transfer 在 imitation framework 下 tractable（student）。这比直接 end-to-end RGB RL 现实得多。

**第三，delta action + RSI 是 humanoid RL 的两个 secret sauce**。Delta action 让 policy 学增量而非 absolute pose，sample efficiency 暴涨；RSI 让 policy 从 task 中间点开始 practice，bypass long-horizon exploration 难题。这两个 design choice 对任何 high-DoF long-horizon RL 都应该有参考价值。

**第四，DAgger-BC mixture 的 $\alpha=0.5$ 是 empirical sweet spot**。纯 BC 有 covariate shift，纯 DAgger 早期太差。Mixture 让 BC 提供 prior、DAgger 提供 error recovery。这个 0.5 不是理论推导的，是 ablation 出来的——说明 sim-to-real distillation 还有很多 empirical tuning 空间。

**第五，limitations section 诚实且深刻**。作者承认 sim-to-real 无法 cover open-ended physics/task/reward/hardware diversity，认为未来是 sim + real data synergy。这种 self-aware 的 discussion 比 over-claim 的 paper 有价值得多。

相关 reference：
- HOMIE (WBC backbone): https://arxiv.org/abs/2502.13013
- Isaac Lab: https://arxiv.org/abs/2511.04831
- DAgger: https://arxiv.org/abs/1011.0686
- DeepMimic: https://arxiv.org/abs/1804.02717
- Dactyl: https://arxiv.org/abs/1808.00177
- Dextreme: https://arxiv.org/abs/2210.13702
- Learning by Cheating: https://arxiv.org/abs/1911.00298
- PPO: https://arxiv.org/abs/1707.06347
