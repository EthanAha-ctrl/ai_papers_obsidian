---
source_pdf: Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer.pdf
paper_sha256: 9998e70b2f77514949c5bcde230ac86e7e1b8da01a761c3f89f08b91f60edaee
processed_at: '2026-08-06T00:45:36-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DoorMan

好，我用更口语的方式聊聊这篇 paper。先把核心 thesis 抛出来：**NVIDIA GEAR 团队（Jim Fan 和 Yuke Zhu 领衔，project page: https://doorman.github.io/）做了一个纯 RGB 的人形机器人开门 policy，完全在 simulation 里训练，直接 zero-shot 部署到真实 Unitree G1 上，能开各种门，比人类 VR teleop 还快 31.7%。** 这个数字是最反直觉的——开门这么"日常"的事，竟然比 humanoid 后空翻还难，而 RL policy 居然比人还强。

下面我分几层 intuition 讲。

---

## 1. 为什么开门比后空翻还难

你想想，humanoid 后空翻、kung fu 这些 demo 看起来很炫，但本质上都是 locomotion，可以用 motion capture 数据做 imitation learning，或者纯 RL + reward shaping 就能搞定。perception 部分相对简单——你只要不撞到东西就行。

开门完全不一样。开门是一个 **perception + balance + contact + articulated object** 全耦合的 long-horizon 任务：

1. **Perception**：机器人头上的相机是 egocentric 的，会随着机器人走动而抖动。你要从这个抖动的视频流里找到 handle 的位置，而且 handle 还可能是 lever、knob、push bar 各种形状。
2. **Contact-rich manipulation**：handle 是 spring-loaded 的，你要旋转它才能 unlock latch。旋转方向、力度、时机都要对。转错了或者用力过猛，机器人会失去平衡。
3. **Articulated object tracking**：门板是绕 hinge 转的，一旦 latch 解锁，门板会做 compliant 圆弧运动。机器人要跟踪这个圆弧，同时身体要配合移动。
4. **Whole-body coordination under external force**：推门的时候，hinge 会给机器人反作用力。机器人要 lean forward 对抗这个力，同时保持平衡。lean 多少、什么时候 lean，都需要 fine-grained closed-loop control。

任何一环失败，整个 skill 就崩掉。比如 perception 漂了一点，手抓空了；或者 contact force 估计错了，机器人被门拽倒。这就是为什么 DARPA Robotics Challenge 那么多年过去，door opening 一直是 humanoid 的硬骨头，大多数系统要么用 depth camera、要么 hard-code motion primitive、要么依赖 operator intervention（参考 Oh et al. 2017, https://ieeexplore.ieee.org/document/7463077）。

---

## 2. 三阶段 Pipeline：Teacher → DAgger → GRPO

DoorMan 的核心 pipeline 在 IsaacLab 里交互式跑完，分三阶段。我用一个比喻：你想教一个盲人（student，只有 RGB 视觉）开门，但你有一个开了 god-mode 的 sighted 导师（teacher，能看到所有 ground truth）。

### Phase 1: Teacher Policy（god-mode RL）

Teacher $\pi_T(a|s)$ 能看到 simulation 里所有 privileged info：
- $\xi_{RD}$：robot root 到 door 的 rigid transform（机器人相对门在哪）
- $\xi_{LD}, \xi_{RH_D}$：左右手到 handle 的 transform（手离 handle 多远）
- $\tau_H \in \mathbb{R}^{18 \times 6}$：18 个 hand body 上的 net contact wrench（6D：3 force + 3 torque）
- $v_R \in \mathbb{R}^3$：root 线速度

用 PPO 训练，reward 按 6 个 stage 分解：0 Walk to door, 1 Pre-grasp, 2 Grasp, 3 Open, 4 Swing, 5 Pass through。每个 stage 激活对应的 reward term（Appendix A, Table 2）。

关键 insight：teacher 完全 bypass 了 perception 难题。所有 state estimation、object pose estimation 都不用做，因为 sim 直接给你 ground truth。这样 policy 只需要解决"怎么控制"这个硬问题，不需要同时解决"怎么感知"和"怎么控制"两个难题。后面 distillation 时再让 student 从 RGB 自己重发现这些信息。

他们 build on top of Homie（Ben et al. 2025, https://arxiv.org/abs/2502.13013）pretrained 的 whole-body controller，这样 legged locomotion 这部分不用从零学，policy 只需 focus 在 manipulation 和与 WBC 的协调上。

### Phase 2: Student Distillation via DAgger

Student $\pi_S(a|o)$ 只有 RGB image + proprioception（joint angles $q$、joint velocities $\dot{q}$、root angular velocities $\dot{\omega}$）。

架构很 standard：
1. **ResNet vision encoder**（He et al. 2015, https://arxiv.org/abs/1512.03385），jointly fine-tuned
2. Vision latent + proprioception features → concat
3. **2-layer LSTM**（512 units each）—— temporal context 关键，因为单帧 RGB 无法推断 handle 的 spring-loaded 状态或 door 当前的 hinge angle
4. **3-layer MLP** (512, 256, 128) → target joint angles

distillation 用 **DAgger**（Ross et al. 2011, https://arxiv.org/abs/1011.0686）。这里有个 subtle 但重要的点：为什么不用 plain behavioral cloning？

BC 只在 teacher 的 state distribution 上做监督。但 student 跟 teacher 看到的东西不一样——student 有 partial observability，它会跑到 teacher 从未见过的 state，比如"handle 被自己手挡住看不见"这种 pose。一旦 student 偏离 teacher distribution，BC 的 error 会 compound（covariance shift），越跑越偏。

DAgger 的做法是：让 student 跑自己的 rollout，但每一步 query teacher 拿 label。这样 student 直接在自己实际会访问的 state distribution 上被监督。对人形 loco-manipulation 特别关键，因为 partial observability 让 student 很容易跑到 teacher 没见过的地方。

### Phase 3: GRPO Fine-tuning（最 interesting 的部分）

即使 DAgger distill 完，student 仍然只有 50-70% 成功率，比 teacher 的 80-90% 低一截。paper 称为 **non-recoverable observability gap**。

为什么 DAgger 不够？因为 student 看不到 teacher 能看到的 privileged info，比如 handle 的精确 transform。BC/DAgger 永远学不到"如何主动调整 pose 让 handle 重新进入视野"这种 compensatory behavior，因为 teacher 从来不需要这样做——teacher 永远看得见 handle。

这就是 GRPO 登场的 motivation。GRPO（Group Relative Policy Optimization，Shao et al. 2024, DeepSeekMath, https://arxiv.org/abs/2402.03300）是 actor-only PPO 变体，no value function，baseline 从 group 内 trajectory return 估计。

公式：
- Group 内 G 条 rollout $\{\tau_i\}_{i=1}^G$，每条 return $R_i$
- Group-relative advantage：$\hat{A}_i = \frac{R_i - \mathrm{mean}(R)}{\mathrm{std}(R)}$
  - $\mathrm{mean}(R)$、$\mathrm{std}(R)$ 是 group 内 return 的均值和标准差，当成 natural baseline
- Clipped surrogate loss：
$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{i,t}\Big[\min\big(r_{i,t}(\theta)\hat{A}_i, \; \mathrm{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\big)\Big]$$
  - $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t}|o_{i,t})}{\pi_{\text{old}}(a_{i,t}|o_{i,t})}$ 是 importance sampling ratio
  - $\epsilon$ 是 clip 范围（通常 0.2）

**intuition**：GRPO 不需要 value network。对 33+ DoF 人形机器人这种高维 action space + partial observability，value function 极难训准。group-relative baseline 利用"同一初始条件 rollout 多次"的对称性，把 group 内的 mean return 当 baseline，方差小、稳定性高。

reward 信号超级简单：**binary task success** + 基础 shaping（joint velocity、acceleration、action rate penalty）。几乎不需要额外 reward engineering，可以当作 drop-in refinement 套到任何"已经有非零成功率"的 base policy 上。

empirically，student 学到了 teacher 从没演示过的策略：
- 把 manipulated object 保持在视野中央
- 调整 end-effector pose 维持 visibility
- 在 off-nominal camera pose 下 recover

这个三段式 pipeline 跟 LLM 里 SFT → RLHF 的思路异曲同工，只是这里 SFT 换成 DAgger，RLHF 换成 GRPO。

---

## 3. Staged Reset：把 long-horizon 任务免费 decompose

Phase 1 teacher 训练里有个关键 trick，叫 staged-reset exploration。

问题：door opening 是 multi-stage 任务，stage 之间是 narrow bridge。比如从"接近门"到"成功 grasp handle"，bridge crossing probability $p_{\text{bridge}} \ll 1$。policy 如果一开始不会正确 rotate handle，grasp 就会触发 collision penalty、torque penalty，policy 反而 **unlearn** grasping behavior——它学会"不要去碰 handle"，因为碰了就惩罚。这就陷入 local optimum，永远进不到 stage 2。

inspired by Go-Explore（Ecofet et al. 2021, https://www.nature.com/articles/s41586-020-03157-9），他们利用 simulator 的 superpower：**simulator 可以任意 set state**，这在 real world 是不可能的。

具体做法：
- 当 environment 进入新 stage，把最近 100 个 snapshot 缓存到 rolling buffer
- reset 时按概率随机回到某个 middle stage（从 buffer sample snapshot 加载）
- snapshot 包含所有 articulated 和 rigid object 的 generalized coordinates

形式化一点。state space 划分为 disjoint stage subset $\{S_1, \ldots, S_K\} \in \mathcal{S}$，stage 之间通过 bridge $B_{y, y+1} \in S_y$ 连接。

定义 staged reset law：

$$\alpha = (\alpha_1, \ldots, \alpha_K), \quad \sum_{y=1}^K \alpha_y = 1$$

$\alpha_y$ 是 reset 到 stage $y$ 的概率。新的初始分布：

$$\tilde{\rho}_\alpha = \sum_{y=1}^K \alpha_y \rho_y$$

对应的 discounted occupancy measure：

$$d_\pi^\alpha(s) = (1-\gamma) \sum_{t=0}^\infty \gamma^t \mathrm{Pr}(s_t = s \mid s_0 \sim \tilde{\rho}_\alpha, \pi)$$

- $\gamma \in [0,1)$：discount factor
- $\mathrm{Pr}$：marginal probability
- $d_\pi^\alpha(s)$：在初始分布 $\tilde{\rho}_\alpha$ 和 policy $\pi$ 下，state $s$ 被访问的 discounted 频率

**intuition**：staged reset 把 occupancy measure 重新加权到 later-stage regions。later stage 不需要从 $\rho_0$ 走很远才能 reach，policy 在 later stage 上能拿到 dense supervision，避免 long-horizon credit assignment 的方差爆炸。这相当于免费把 long-horizon 任务 decompose 成多个 short-horizon 任务。

ablation（Figure 6b）非常说服力：
- buffer size 100：500 iter 内 reach 大部分 stage，1700 iter 全部 reach
- buffer size 10：4000+ iter 才完成 exploration
- buffer size 0：完全无法进入 stage 2，policy 陷入"avoid grasping"的 local optimum

这跟 curriculum learning 不一样——curriculum 通常是 schedule 化的"先学简单再学难"，staged reset 是始终 maintain 一个混合的 reset distribution，让 later-stage state 始终被高频采样。这个 trick 在任何 multi-stage RL 任务里都该试。

---

## 4. Domain Randomization：关键是 coverage，不是 pixel-perfect

paper 反复强调："不重建真实场景，而是暴露 broad variability envelope"。所有真实评估场景都是 unseen during training。

### Physical randomization（Table 3）

5 种 door type，3 大类：
- pushing door with rotational handle
- pulling door with rotational handle  
- pushing door with push bar

物理参数全部 randomize：
- Panel width 0.8-1.1 m，height 1.9-2.2 m
- Handle height 0.85-0.95 m，handle-to-edge 0.04-0.1 m
- Weight 80-120 kg
- Hinge max force 20-30 Nm，damping 5-10 (kg·m²)/(s²·°)，stiffness 10-20
- Handle max force 1-3 Nm，damping 0.1-0.6，stiffness 30-50
- **Latch 用 mimic joint** 跟 handle angle 联动——模拟"handle 转到一定角度才 unlock hinge"的真实物理。这个细节很重要，它捕捉了开门瞬间 whole-body dynamics 的 abrupt change
- Handle actuator target -5°（向上），模拟 spring-loaded tension

### Visual randomization

- OmniPBR materials，对每个 material 随机化 sub-identifier、texture transform、albedo color、tint
- 每 0.9-1.1 秒，scene 内一个 geometry 重新 draw material（动态变化，逼 policy 学 invariant feature 而不是 memorize appearance）
- **5233 个 dome light texture**（Omniverse 内置 + Poly Haven），覆盖室内/室外/不同 time-of-day。这个数字是 killer feature
- RTX Real-Time renderer performance mode + motion blur + auto white balance
- Camera extrinsics/intrinsics 轻微 randomize

ablation（Table 1）揭示一个反直觉的 insight：

| 配置 | Push Lever | Pull Lever | Push Bar |
|---|---|---|---|
| No rand | 10.8 | 5.0 | 20.0 |
| Solid-color + dome | 67.5 | 65.8 | 70.0 |
| +10% texture, no dome | 58.3 | 50.8 | 76.7 |
| +10% texture + dome | 79.2 | 77.5 | 77.5 |
| +100% texture, no dome | 73.3 | 55.8 | 76.7 |
| +100% texture + dome | **85.8** | **80.8** | **85.0** |

观察：
1. **Dome light randomization 最关键**。去掉 dome light 掉 15-30%。因为 lighting 直接决定 PBR material 在 camera 里的 appearance distribution。同一张红色 texture，在暖光和冷光下看起来完全不一样。
2. **10% texture 已经接近 100% texture 的效果**。说明 PBR material 的多样性只要 cover 主要 appearance mode 就够，不需要穷举所有 texture。这跟 LLM 里 data scaling 的"long tail 不重要，head distribution 覆盖到就够"的直觉一致。
3. **Long-horizon task 对 visual randomization 最敏感**。Pull lever（最 long-horizon）去掉 dome light 掉 25%，push bar 只掉 9%。因为 partial observability 累积——你看不清 handle 一步，后面整个 trajectory 都偏。
4. **No randomization 直接崩到 5-20%**。证明 sim-to-real gap 是真实存在且必须用 DR 解决的。

**核心 intuition**：sim-to-real gap 主要不是 texture 细节，而是 appearance distribution 的 coverage。dome light 去掉比 texture 去掉影响大，因为 lighting 直接决定整个 scene 的 appearance distribution。未来 sim-to-real 不一定要追求 pixel-perfect，而是要追求 appearance distribution coverage。

这跟早期 domain randomization（Tobin et al. 2017, https://arxiv.org/abs/1703.06907）相比，photorealism 是关键——早期只能 solid color randomize，现在有 PBR + ray tracing，能模拟 harsh real-world 的 contact switching 下的相机抖动。

---

## 5. 为什么能 beat human teleop

主结果（Figure 5）：
- **Success rate**: DoorMan 83%，expert teleop 80%，non-expert 60%
- **Completion time**: DoorMan 比 expert 快 23.8%，比 non-expert 快 31.7%

实验设置很公平：同一套 WBC（HOMIE），同一个 robot，区别只在 high-level policy 是 DoorMan 还是 VR teleop。teleop baseline 用 PICO 4 Ultra headset + 两个 handheld controller，输出 3 个 upper-body SE(3) pose + finger joint angles + planar nav command（$v \in \mathbb{R}^2$ + $\omega \in \mathbb{R}$），用 Pinocchio（Carpentier et al. 2019, https://arxiv.org/abs/1807.09160）做 IK。

teleop 失败的 qualitative 原因很直观：
1. **无法 gauge spring-loaded handle 和 hinge 的反作用力大小**。VR 手柄没有 force feedback，人感觉不到 handle 什么时候 unlock、门什么时候开始转。
2. **无法判断 robot 是否 lean 得合适**。推门时机器人要 lean forward 对抗 hinge 反作用力，lean 多少、什么时候 lean，需要 fine-grained closed-loop feedback，VR + headset + joystick 这种 coarse 接口做不到。
3. **无法 track 门板的 revolving path**。门板做 compliant 圆弧运动，机器人要配合移动，但人通过 VR 看到的视角跟机器人实际 body frame 不完全对齐，spatial reasoning 很难。

这些都需要 fine-grained closed-loop feedback，VR 接口做不到。但 RL policy 在 sim 里直接以 50Hz 闭环 + 几百万步训练，能学到这种 **implicit force-awareness**——policy 网络里 encode 了"当前 joint configuration + contact wrench 历史 → 应该施加多少 torque"的映射，人脑通过 VR 接口根本无法实时计算这个。

这跟 AlphaGo 击败人类的逻辑类似：不是 RL policy 比 human 更"聪明"，而是 RL policy 在 sim 里见过的 state space 和试过的 action sequence 比人类一辈子能试的多几个数量级。

---

## 6. Reward 设计：hand-engineered 但有逻辑

reward 是 stage-conditioned 的，28 个 term（Appendix A, Table 2）。我挑几个关键的讲 intuition。

**通用 penalty**（全 stage 开启）：
- Termination: $-1000$（fall over 等灾难性终止，强烈惩罚）
- Upright penalty: $\|R_{\text{torso}}[0,0,1]^\top - [0,0,1]^\top\|_2^2$，weight $-1.0$（保持 torso 朝上）
- Door frame/panel contact: $\sum\|f_{\text{doorframe/panel}}\|_2$，weight $-0.1$（避免撞门）

**Stage 0: Walk to door**
- Walk to door: $\exp(-\|v_{\text{robot}} - v_{\text{target}}\hat{d}_{\text{door}}\|_2^2 / (2 \cdot 0.15^2))$，weight 5.0
  - $v_{\text{robot}}$：robot 当前速度
  - $v_{\text{target}}$：目标速度
  - $\hat{d}_{\text{door}}$：朝向 door 的单位向量
  - Gaussian tracking reward，鼓励 robot 沿 door 方向以 target speed 前进
- Face door: $|\mathrm{wrap}_\pi(\|\mathrm{axis-angle}(R_{\text{door}})\|_2)|$，weight $-1.0$
  - $R_{\text{door}}$：当前朝向与 door 朝向的差异
  - $\mathrm{wrap}_\pi$：把角度 wrap 到 $[-\pi, \pi]$

**Stage 1: Pre-grasp**
- Hand-handle orientation: $\exp(-\|\mathrm{wrap}_\pi(\|\mathrm{axis-angle}(R_{\text{hand}}R_{\pm 90})\|_2)\|^2 / (2 \cdot 0.6^2))$，weight 3.0
  - $R_{\text{hand}}$：手朝向
  - $R_{\pm 90}$：预定义的"正确 grasp 朝向"（左手/右手差 180°）
- Penalty not standing still: $\|u_{\text{HOMIE,[0:3]}}\|_2$，weight $-15.0$
  - $u_{\text{HOMIE,[0:3]}}$：HOMIE WBC 接口的 planar navigation command
  - pre-grasp 阶段要求 robot 站定不动，weight 很大确保不乱走

**Stage 3: Open**
- Push door handle: $\dot{\theta}_{\text{handle}} + \mathrm{clip}(\theta_{\text{handle}}, 0, 45°)/45°$，weight 6.0
  - $\dot{\theta}_{\text{handle}}$：handle 角速度
  - $\theta_{\text{handle}}$：handle 当前角度
  - 奖励"持续转动 handle"+"已经转过的比例"
- Push door hinge: $10\dot{\theta}_{\text{hinge}} + \mathrm{clip}(\theta_{\text{hinge}}, 0, 90°)/90°$，weight 6.0
  - 系数 10 说明 hinge 旋转更难，需要更大 reward slope

**Stage 4-5: Swing & Pass through**
- Don't push door handle: $-\dot{\theta}_{\text{handle}} + (45° - \theta_{\text{handle}})/45°$，weight 3.0
  - 开门后 handle 应该回弹（spring-loaded），不要继续推
- Target root distance: $\mathrm{track}(v_{\text{root}} \cdot \hat{d}_{\text{target}}, v_{\text{target}}, \sigma=0.2) + \mathrm{track}(\|p_{\text{root}} - p_{\text{target}}\|, 0, \sigma=0.2)$，weight 12.0
  - 鼓励 root 沿目标方向以 target speed 前进，最终到达 target 位置
- Penalty standing still: $\exp(-\|u_{\text{HOMIE,[0:3]}}\|_2^2 / (2 \cdot 0.05^2))$，weight $-1.0$
  - 跟 stage 1 相反，现在要求必须走，停下来会被惩罚

**Always-on**:
- Stage progress: $\mathrm{stage}_{\text{current}}$，weight 1.0（鼓励进入更后 stage）
- Task completion: $\mathbb{1}_{\{\text{complete}\}}$，weight 4.0
- Success save time: $\mathbb{1}_{\{\text{success}\}} \cdot$ remaining time ratio，weight 0.5（鼓励快完成）

整个 reward 表 hand-engineered 程度很高，这是 paper 的 limitation——future work 说想用 high-capacity BC teacher 替代这种 reward engineering，这跟 LeVERB（Xue et al. 2025, https://arxiv.org/abs/2506.13751）思路一致。

---

## 7. 我的 take-away 和联想

### Sim-to-real 从 research problem 变成 engineering scale problem

这篇 paper 给我最大的 intuition：**sim-to-real 不再是研究难题，而是工程规模问题**。当 DR 覆盖到 5233 个 lighting condition + 5 种 door + 全物理参数 randomize，sim-to-real 就能 work。这跟 LLM scaling law 类似——不是算法变聪明了，而是数据规模 + 多样性到了 critical threshold。

NVIDIA 的 IsaacLab（https://arxiv.org/abs/2511.04831）+ Omniverse RTX renderer 在这个方向上有 structural advantage，因为 photorealism 是 GPU 公司的天然护城河。别的团队要复现，要么用 NVIDIA 的工具栈，要么自己 build photorealistic sim，后者成本极高。

### Teacher-student-bootstrap 跟 LLM RLHF 的类比

DoorMan 的三段式 pipeline 跟 LLM 的 SFT → RLHF 结构惊人地相似：
- **Phase 1 (Teacher RL)**：相当于用 ground truth reward 训一个 strong policy，类似 LLM 里用 verifier reward 训 strong model
- **Phase 2 (DAgger distillation)**：相当于 SFT，把 strong policy 蒸馏到 student
- **Phase 3 (GRPO fine-tuning)**：相当于 RLHF/RLAIF，让 student 在自己的 distribution 上 self-improve

关键 insight 是 Phase 3 解决了 Phase 2 留下的 information gap——student 学到了 teacher 没演示过的 compensatory behavior。这个 pattern 我觉得会扩散到整个 robot learning 领域。

### Sim 的 recoverability 是免费资源

staged reset 这个 trick 利用了 simulator 可以任意 set state 的能力，这在 real world 是不可能的。这等于把 long-horizon 任务免费 decompose 成多个 short-horizon 任务，每段都 dense supervision。这跟 Go-Explore（https://www.nature.com/articles/s41586-020-03157-9）思路一致，但用在了 contact-rich humanoid 任务上。

这个 trick 在任何 multi-stage RL 任务里都该试，特别是 robot learning 里"探索 bottleneck"普遍存在。想象一下 bussing a table（多 stage：approach table, pick plate, move to tray, place plate），或者 making a bed（多 stage：approach bed, grab sheet corner, pull, smooth），都可以套用 staged reset。

### Photorealism 的真正价值

Table 1 的 ablation 给我一个重要 intuition：**sim-to-real gap 主要不是 texture 细节，而是 appearance distribution 的 coverage**。

去掉 texture 只保留 solid color + dome light，还能到 65-70%。但去掉 dome light，即使 100% texture，pull lever 直接掉到 55.8%。说明 lighting 分布覆盖比 texture 细节重要得多。

这跟人类 vision 的 intuition 一致——人眼在不同 lighting 下认出同一个物体，靠的是 shape、geometry、relative lighting pattern 这些 invariant feature，不是 pixel-perfect texture。RL policy 在大规模 dome light randomization 下学到的也是这种 invariant feature。

### 局限性

1. **Reward engineering 极重**。28 个 term，每个 stage 都要 tune。future work 说想用 BC teacher 替代，但 BC teacher 又需要 human demo data，回到 data 采集的老问题。可能的出路是用 VLA model（如 RT-2、π0、LeVERB）作为 teacher。
2. **只测了门这一类 articulated object**。drawer、knob、latch、elevator button 等虽然 paper 提了但没实测。不同 articulated object 的物理特性差异大（drawer 是 prismatic joint，knob 是 continuous rotation），pipeline 能否直接 transfer 未知。
3. **Camera 是 fixed egocentric**。GRPO 学到的 compensatory behavior 还是 reactive 的（"handle 看不见就调整 pose 让它重新进入视野"），没有 active vision（"主动探索最佳视角"）。如果机器人能主动控制 camera pose（比如转头），可能能进一步提性能。
4. **跟 human teleop 比用的是 VR + joystick 这种相对 coarse 的接口**。跟 exoskeleton teleop（Homie，https://arxiv.org/abs/2502.13013）或直接模仿 human motion 的系统（HDMI，https://arxiv.org/abs/2509.16757）没直接比。不过 Homie 本身是 teleop cockpit，不是 autonomous policy，比较维度不同。
5. **5233 dome light + 5 door type 的 variability envelope 在更复杂场景是否够还未知**。比如多层楼的 fire door（很重、有 closer mechanism）、旋转门、电梯门，这些 physical dynamics 跟 paper 里的 door 差异大，可能需要扩展 physical randomization 范围。

### 相关工作的串联

整个 humanoid WBC 领域最近进展很快，DoorMan 是这条线上的一个 milestone：
- **Homie** (https://arxiv.org/abs/2502.13013)：isomorphic exoskeleton cockpit 做 teleop 和 WBC 训练，DoorMan 直接 build on top
- **ASAP** (https://arxiv.org/abs/2502.01143)：aligning sim and real physics，做 agile humanoid whole-body skill
- **LeVERB** (https://arxiv.org/abs/2506.13751)：photorealistic demo + latent vision-language instruction，coarse whole-body task
- **Sonic** (https://arxiv.org/abs/2511.07820)：supersizing motion tracking for natural humanoid WBC
- **OmniH2O** (https://arxiv.org/abs/2406.08858)：universal human-to-humanoid teleop
- **HDMI** (https://arxiv.org/abs/2509.16757)：learning interactive humanoid WBC from human videos

door opening 专题的对比：
- **DoorGym** (https://arxiv.org/abs/1908.01887)：早期 scalable door opening env，但 wheeled platform + privileged sensing
- **StageAct** (https://arxiv.org/abs/2509.13200)：stage-conditioned imitation for humanoid door opening，跟 DoorMan 最直接对比，但 confined to same scene
- **Legged manipulator door opening** (https://proceedings.mlr.press/v270/zhang25g.html)：sim-to-real door，但 privileged sensing

visual sim-to-real 经典：
- **Domain randomization** (https://arxiv.org/abs/1703.06907)：最早的 DR
- **Dextreme** (https://arxiv.org/abs/2210.13739)：dexterous in-hand manipulation sim-to-real
- **Dextrah-RGB** (https://arxiv.org/abs/2412.01791)：stereo image zero-shot dexterous grasping，跟 DoorMan 思路最像
- **Extreme Parkour** (https://arxiv.org/abs/2309.14341)：vision-based legged parkour
- **ANYmal perceptive locomotion** (https://www.science.org/doi/10.1126/scirobotics.abk2822)：robust perceptive locomotion in the wild

RL fine-tuning 方向：
- **RoboCat** (https://arxiv.org/abs/2306.11706)：self-improving generalist agent
- **TransIC** (https://arxiv.org/abs/2405.10315)：sim-to-real with online correction
- **Self-improving visuomotor** (https://arxiv.org/abs/2303.01488)
- **Residual RL** (https://arxiv.org/abs/1812.03201)：经典 residual RL

procedural generation：
- **Infinigen-Sim** (https://arxiv.org/abs/2505.10755)：procedural articulated simulation-ready assets，DoorMan 借鉴但 IsaacLab-native 实现物理更准

---

## 8. 最终直觉

DoorMan 证明了一件事：**只要 sim 足够 photorealistic + DR 足够 broad + pipeline 设计合理，纯 RGB sim-to-real 在 contact-rich whole-body task 上能超过 human teleop**。这意味着 humanoid robot learning 的 bottleneck 正在从"real data 采集"转移到"sim photorealism + DR pipeline + RL algorithm"。

整个 pipeline 的 intuition 层层递进：
1. Teacher 先用 privileged info 把"控制难题"解决
2. DAgger 把 teacher 蒸馏到 RGB student，解决 state distribution mismatch
3. GRPO 让 student self-improve，解决 observation information gap
4. Staged reset 解决 long-horizon exploration bottleneck
5. 大规模 DR 解决 sim-to-real appearance gap

每一步都在解一个特定的 bottleneck，组合起来就 close the loop。这个 pattern 我觉得会成为未来 humanoid policy learning 的 standard recipe，就像 ResNet 之于 image classification，Transformer 之于 sequence modeling 一样。

paper 给我最大的启发是 GRPO 这一步——它本质上是在说"distillation 有上限，student 需要 self-bootstrap 才能超越 teacher"。这跟 Sutton 在 "The Bitter Lesson"（http://www.incompleteideas.net/IncIdeas/BitterLesson.html）里讲的"compute > human knowledge"一脉相承。teacher 的 privileged info 是 human knowledge 的体现，但 student 通过 self-play 式的 GRPO RL，能在自己的 observation distribution 上发现 teacher 没见过的策略。这是 scaling 的胜利，不是 algorithmic trick 的胜利。

---

# DoorMan 详解：从 Sim 到 Real 的人形机器人开门 Policy

## 1. 总览与 motivation

这篇 paper 来自 NVIDIA GEAR 团队（Linxi "Jim" Fan 和 Yuke Zhu 领衔），核心成果是一个纯 RGB 感知的人形机器人 loco-manipulation policy，叫 **DoorMan**，完全在 photorealistic simulation 里训练，zero-shot 部署到真实 Unitree G1 上，能开多种类型的门，并且比人类 teleop 还快 31.7%。这听起来有点反直觉——开门这么"日常"的事，竟然比后空翻还难。原因在于开门是一个 perception + balance + contact + articulated-object interaction 紧耦合的 long-horizon 任务：要从一个 egocentric 移动相机里识别 grasp location，旋转 spring-loaded handle，跟踪门板的 compliant 圆弧运动，还要在 hinge 反作用力下保持平衡。任何一环失败都会让整个 skill 崩掉。

paper 的核心 thesis 是：**photorealistic simulation + 大规模 domain randomization + teacher-student bootstrap + GRPO 自我精修**，这套组合能让一个完全 sim-trained 的 vision policy 在真实 articulated object 上超过 human teleop。这跟传统 robot learning 里"小数据 BC + 精心标定真实场景"的路子完全不同。

project page: https://doorman.github.io/  
IsaacLab: https://arxiv.org/abs/2511.04831

---

## 2. 核心架构：三阶段 Teacher-Student-Bootstrap Pipeline

整个 pipeline 在 IsaacLab 里交互式跑完，分三阶段。

### Phase 1: Teacher Policy（privileged RL）

Teacher policy $\pi_T(a|s)$ 能看到 simulation 里一切 ground-truth 信息：
- $\xi_{RD}$：robot root 到 door 的 rigid transform
- $\xi_{LD}, \xi_{RH_D}$（paper 里写成 $\xi_{RD}$ 是 typo，应该是 right-hand-to-door-handle）：左右手到 handle 的 transform
- $\tau_H \in \mathbb{R}^{18 \times 6}$：18 个 hand body 上的 net contact wrench（6D：3 force + 3 torque）
- $v_R \in \mathbb{R}^3$：root 线速度

用 PPO 训练，reward 按 6 个 stage 分解：0 Walk to door, 1 Pre-grasp, 2 Grasp, 3 Open, 4 Swing, 5 Pass through。每个 stage 有独立的 shaping reward，并且只有进入对应 stage 才激活（Table 2）。

关键 insight：teacher 不需要估计这些 privileged 量，所以可以把所有 perception 难题（estimator、state estimation）直接 bypass，把"控制难题"先解决掉。后面 distillation 时再让 student 自己从 RGB 重发现这些信息。

他们 build on top of Homie（Ben et al. 2025, https://arxiv.org/abs/2502.13013）pretrained 的 whole-body controller，这样 legged locomotion 这部分不用从零学，policy 只需 focus 在 manipulation 和与 WBC 的协调上。

### Phase 2: Student Distillation via DAgger

Student policy $\pi_S(a|o)$ 输入：
- RGB image（egocentric camera）
- proprioception：joint angles $q$、joint velocities $\dot{q}$、root angular velocities $\dot{\omega} \in \mathbb{R}^3$

架构：
1. **Vision encoder**：ResNet（He et al. 2015, https://arxiv.org/abs/1512.03385），jointly fine-tuned
2. Vision latent + proprioception features → concat
3. **2-layer LSTM**（512 units each）—— temporal context 很关键，因为单帧 RGB 无法推断 handle 的 spring-loaded 状态或 door 当前的 hinge angle
4. **3-layer MLP** (512, 256, 128) → target joint angles

注意：Unitree G1 有 29 body joints + 14 hand joints，paper 写 action space dimension 33，这个数字对不上 29+14=43，我推测实际控制的是 subset（可能 23 body + 10 finger primitives，或者 HOMIE 接口压缩了 finger DoF），paper 这里有 inconsistency。

distillation 用 **DAgger**（Ross et al. 2011, https://arxiv.org/abs/1011.0686），而不是 plain behavioral cloning。差别在于：BC 只在 teacher 的 state distribution 上做监督，student 一旦在 deployment 偏离 teacher distribution，error 会 compound（covariance shift）；DAgger 让 student 跑自己的 rollout，但 query teacher 拿 label，所以 student 直接在自己实际会访问的 state distribution 上被监督。这对人形 loco-manipulation 特别关键，因为 partial observability 让 student 很容易跑到 teacher 从未见过的"看不清 handle"的 pose。

### Phase 3: GRPO Fine-tuning

这是 paper 最有意思的部分。即使 DAgger distill 完，student 仍然有 50-70% 成功率，比 teacher 的 80-90% 低一截，paper 称为 **non-recoverable observability gap**。原因是 student 看不到 teacher 能看到的 privileged info，比如 handle 的精确 transform，BC/DAgger 永远学不到"如何主动调整 pose 让 handle 重新进入视野"这种 compensatory behavior，因为 teacher 从来不需要这样做。

GRPO（Group Relative Policy Optimization，Shao et al. 2024, DeepSeekMath, https://arxiv.org/abs/2402.03300）是一个 actor-only PPO 变体，no value function，baseline 从 group 内的 trajectory return 估计。公式如下：

给定 batch 内 G 条 rollout $\{\tau_i\}_{i=1}^G$，每条 return $R_i$，定义 group-relative advantage：

$$\hat{A}_i = \frac{R_i - \mathrm{mean}(R)}{\mathrm{std}(R)}$$

这里 $\mathrm{mean}(R)$ 和 $\mathrm{std}(R)$ 是 group 内 return 的均值和标准差，相当于把 baseline 用同 group 其他 rollout 估计出来，避免了训 value function。

policy update 用 clipped surrogate：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{i,t}\Big[\min\big(r_{i,t}(\theta)\hat{A}_i, \; \mathrm{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\big)\Big]$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t}|o_{i,t})}{\pi_{\text{old}}(a_{i,t}|o_{i,t})}$ 是 importance sampling ratio，$\epsilon$ 是 clip 范围（通常 0.2）。

intuition：GRPO 不需要 value network，对高维 action space（33+ DoF）的人形机器人特别友好，因为 value function 在 33 维 action + 部分可观测 observation 下极难训准。group-relative baseline 利用了"同一初始条件 rollout 多次"的对称性，把 group 内的 mean return 当成 natural baseline，方差小、稳定性高。

reward 信号很简单：binary task success + shaping（joint velocity、acceleration、action rate penalty）。这意味着 GRPO 阶段几乎不需要额外 reward engineering，可以当作 drop-in refinement 套到任何"已经有非零成功率"的 base policy 上。empirically，student 学到了"把 manipulated object 保持在视野中央"、"调整 end-effector pose 维持 visibility"这些 teacher 从没演示过的策略。

---

## 3. Staged-Reset Exploration：解决 long-horizon credit assignment

这是 Phase 1 teacher 训练的关键 trick。问题在于：door opening 是 multi-stage 任务，stage 之间是 narrow bridge（比如从"接近门"到"成功 grasp handle"），bridge crossing probability $p_{\text{bridge}} \ll 1$。如果 policy 一开始不会正确 rotate handle，grasp 就会触发 collision penalty、torque penalty，policy 反而 "unlearn" grasping behavior，永远进不到 stage 2。

inspired by Ecofet et al. 2021 "First return, then explore"（Go-Explore, https://www.nature.com/articles/s41586-020-03157-9），他们利用 simulator 的 full recoverability：当 environment 进入新 stage，把最近 100 个 snapshot 缓存到 rolling buffer；reset 时按一定概率随机 reset 到某个 middle stage。

形式化：把 state space 划分为 disjoint 的 stage subset $\{S_1, \ldots, S_K\} \in \mathcal{S}$，stage 之间通过 bridge $B_{y, y+1} \in S_y$ 连接。

定义 staged reset law：

$$\alpha = (\alpha_1, \ldots, \alpha_K), \quad \sum_{y=1}^K \alpha_y = 1$$

这里 $\alpha_y$ 是 reset 到 stage $y$ 的概率。新的初始分布：

$$\tilde{\rho}_\alpha = \sum_{y=1}^K \alpha_y \rho_y$$

其中 $\rho_y$ 是 stage $y$ 的 reset distribution（从 buffer 里 sample snapshot）。

对应的 discounted occupancy measure：

$$d_\pi^\alpha(s) = (1-\gamma) \sum_{t=0}^\infty \gamma^t \mathrm{Pr}(s_t = s \mid s_0 \sim \tilde{\rho}_\alpha, \pi)$$

这里 $\gamma \in [0,1)$ 是 discount factor，$\mathrm{Pr}$ 是 marginal probability，$d_\pi^\alpha(s)$ 表示在初始分布 $\tilde{\rho}_\alpha$ 和 policy $\pi$ 下，state $s$ 被访问的 discounted 频率。

关键 insight：staged reset 把 occupancy measure 重新加权到 later-stage regions，增加这些 state 的有效 gradient update 频率和幅度。从 RL 角度看，这相当于改变了 effective horizon——later stage 不需要从 $\rho_0$ 走很远才能 reach，policy 在 later stage 上能拿到 dense supervision，避免 long-horizon credit assignment 的方差爆炸。

ablation（Figure 6b）非常直观：
- buffer size 100：500 iter 内 reach 大部分 stage，1700 iter 全部 reach
- buffer size 10：4000+ iter 才完成 exploration
- buffer size 0：完全无法进入 stage 2，policy 陷入"avoid grasping"的 local optimum

这跟 curriculum learning 类似但不一样——curriculum 通常是 schedule 化的"先学简单再学难"，staged reset 是始终 maintain 一个混合的 reset distribution，让 later-stage state 始终被高频采样。

---

## 4. Massive-Scale Domain Randomization

paper 强调"不重建真实场景，而是暴露 broad variability envelope"。所有真实评估场景都是 unseen during training。

### Physical randomization（Table 3）
5 种 door type，覆盖 3 大类：
- pushing door with rotational handle
- pulling door with rotational handle  
- pushing door with push bar

物理参数全部 randomize：
- Panel width 0.8-1.1 m，height 1.9-2.2 m
- Handle height 0.85-0.95 m，handle-to-edge 0.04-0.1 m
- Weight 80-120 kg
- Hinge max force 20-30 Nm，damping 5-10 (kg·m²)/(s²·°)，stiffness 10-20
- Handle max force 1-3 Nm，damping 0.1-0.6，stiffness 30-50
- Latch 用 mimic joint 跟 handle angle 联动，模拟"handle 转到一定角度才 unlock hinge"的真实物理
- Handle actuator target -5°（向上），模拟 spring-loaded tension

### Visual randomization
- OmniPBR materials，对每个 material 随机化 sub-identifier、texture transform、albedo color、tint
- 每 0.9-1.1 秒，scene 内一个 geometry 重新 draw material（动态变化，逼 policy 学 invariant feature）
- **5233 个 dome light texture**（Omniverse 内置 + Poly Haven），覆盖室内/室外/不同 time-of-day
- RTX Real-Time renderer performance mode + motion blur + auto white balance
- Camera extrinsics/intrinsics 轻微 randomize

这跟早期 domain randomization（Tobin et al. 2017, https://arxiv.org/abs/1703.06907）相比，关键是 photorealism——早期只能 solid color randomize，现在有 PBR + ray tracing，能模拟 harsh real-world 的 contact switching 下的相机抖动。

ablation（Table 1）说明 visual randomization 的影响：
| 配置 | Push Lever | Pull Lever | Push Bar |
|---|---|---|---|
| No rand | 10.8 | 5.0 | 20.0 |
| Solid-color + dome | 67.5 | 65.8 | 70.0 |
| +10% texture, no dome | 58.3 | 50.8 | 76.7 |
| +10% texture + dome | 79.2 | 77.5 | 77.5 |
| +100% texture, no dome | 73.3 | 55.8 | 76.7 |
| +100% texture + dome | **85.8** | **80.8** | **85.0** |

观察：
1. Dome light randomization 最关键（去掉掉 15-30%），因为 lighting 直接决定 texture 在 camera 里的 appearance distribution
2. 10% texture 已经接近 100% texture 的效果，说明 PBR material 的多样性只要 cover 主要 appearance mode 就够
3. Long-horizon task（pull lever）对 visual randomization 最敏感，因为 partial observability 累积
4. No randomization 直接崩到 5-20%，证明 sim-to-real gap 是真实存在的、必须用 DR 解决

---

## 5. Reward 设计深度解析（Appendix A, Table 2）

reward 是 stage-conditioned 的，每个 stage 只激活对应项。我用几个关键 term 解释 intuition。

**通用 penalty**（全 stage 开启）：
- Termination: $-1000$（fall over 等灾难性终止，强烈惩罚）
- Delta action rate: $\|\Delta a_t\|_2^2$，weight $-0.01$（平滑控制）
- DoF velocity: $\|\dot{q}_{\text{upper, non-finger}}\|_2^2$，weight $-10^{-3}$
- Door frame/panel contact: $\sum\|f_{\text{doorframe/panel}}\|_2$，weight $-0.1$（避免撞门）
- Upright penalty: $\|R_{\text{torso}}[0,0,1]^\top - [0,0,1]^\top\|_2^2$，weight $-1.0$（保持 torso 朝上）

**Stage 0: Walk to door**
- Walk to door: $\exp(-\|v_{\text{robot}} - v_{\text{target}}\hat{d}_{\text{door}}\|_2^2 / (2 \cdot 0.15^2))$，weight 5.0
  - $v_{\text{robot}}$ 是 robot 当前速度，$v_{\text{target}}$ 是目标速度，$\hat{d}_{\text{door}}$ 是朝向 door 的单位向量。Gaussian tracking reward，鼓励 robot 沿 door 方向以 target speed 前进。
- Face door: $|\mathrm{wrap}_\pi(\|\mathrm{axis-angle}(R_{\text{door}})\|_2)|$，weight $-1.0$
  - 用 axis-angle 表示当前朝向与 door 朝向的差异，$\mathrm{wrap}_\pi$ 把角度 wrap 到 $[-\pi, \pi]$

**Stage 1: Pre-grasp**
- Hand-handle orientation: $\exp(-\|\mathrm{wrap}_\pi(\|\mathrm{axis-angle}(R_{\text{hand}}R_{\pm 90})\|_2)\|^2 / (2 \cdot 0.6^2))$，weight 3.0
  - $R_{\text{hand}}$ 是手朝向，$R_{\pm 90}$ 是预定义的"正确 grasp 朝向"（左手/右手差 180°），axis-angle 衡量偏差
- Pre-grasp target distance: $\mathrm{track}(\|p_{\text{hand}} - p_{\text{pre-grasp}}\|, 0, \sigma=0.2) + \mathrm{track}(\|v_{\text{hand}} - v_{\text{target}}\hat{d}\|, 0, \sigma=0.15)$，weight 6.0
  - 这里 $\mathrm{track}(x, \mu, \sigma) = \exp(-(x-\mu)^2/(2\sigma^2))$，同时 track 位置和速度
- Penalty not standing still: $\|u_{\text{HOMIE,[0:3]}}\|_2$，weight $-15.0$
  - $u_{\text{HOMIE,[0:3]}}$ 是 HOMIE WBC 接口的 planar navigation command（前 3 维），pre-grasp 阶段要求 robot 站定不动，weight 很大确保不乱走

**Stage 2: Grasp**
- Grasp finger pose: $\mathrm{track}(q_{\text{finger}}, q_{p1}, \sigma_{\text{pos}}=0.3) + \mathrm{track}(\dot{q}_{\text{finger}}, 0.6, \sigma_{\text{vel}}=0.2)$，weight 3.0
  - $q_{p1}$ 是预定义的 grasp finger configuration，同时 track 位置和速度
- Grasp force: $\sum(-|f_{\text{palm},y,z}| + f_{\text{palm},x})$，weight 0.2
  - 鼓励沿手掌法向（x 方向）的压力，惩罚切向（y, z）滑动

**Stage 3: Open**
- Push door handle: $\dot{\theta}_{\text{handle}} + \mathrm{clip}(\theta_{\text{handle}}, 0, 45°)/45°$，weight 6.0
  - $\dot{\theta}_{\text{handle}}$ 是 handle 角速度，$\theta_{\text{handle}}$ 是 handle 当前角度。奖励"持续转动 handle"+"已经转过的比例"
- Push door hinge: $10\dot{\theta}_{\text{hinge}} + \mathrm{clip}(\theta_{\text{hinge}}, 0, 90°)/90°$，weight 6.0
  - 类似，但是 hinge 角度（门板开的角度），系数 10 说明 hinge 旋转更难，需要更大 reward slope
- Push door force: $\mathrm{clip}(f_{\text{hand},x}, 0, 20)$，weight 0.3
  - 鼓励沿开门方向的推力，clip 到 20N 防止暴力推

**Stage 4-5: Swing & Pass through**
- Don't push door handle: $-\dot{\theta}_{\text{handle}} + (45° - \theta_{\text{handle}})/45°$，weight 3.0
  - 开门后 handle 应该回弹（spring-loaded），不要继续推，否则会损坏或卡住
- Target root distance: $\mathrm{track}(v_{\text{root}} \cdot \hat{d}_{\text{target}}, v_{\text{target}}, \sigma=0.2) + \mathrm{track}(\|p_{\text{root}} - p_{\text{target}}\|, 0, \sigma=0.2)$，weight 12.0
  - 鼓励 root 沿目标方向以 target speed 前进，最终到达 target 位置
- Penalty standing still: $\exp(-\|u_{\text{HOMIE,[0:3]}}\|_2^2 / (2 \cdot 0.05^2))$，weight $-1.0$
  - 跟 stage 1 相反，现在要求必须走，停下来会被惩罚

**Always-on**:
- Stage progress: $\mathrm{stage}_{\text{current}}$，weight 1.0（鼓励进入更后 stage）
- Task completion: $\mathbb{1}_{\{\text{complete}\}}$，weight 4.0
- Success save time: $\mathbb{1}_{\{\text{success}\}} \cdot$ remaining time ratio，weight 0.5（鼓励快完成）

整个 reward 表有 28 个 term，非常 hand-engineered，这是 paper 唯一的"弱点"——他们 future work 也提到想用 high-capacity BC teacher 替代这种 reward engineering。

---

## 6. 实验结果：超过 human teleop

主结果（Figure 5）：
- **Success rate**: DoorMan 83%，expert teleop 80%，non-expert 60%
- **Completion time**: DoorMan 比 expert 快 23.8%，比 non-expert 快 31.7%

这是用同一套 WBC（HOMIE）+ 同一个 robot，区别只在 high-level policy 是 DoorMan 还是 VR teleop。teleop 失败的 qualitative 原因：
1. 无法 gauge spring-loaded handle 和 hinge 的反作用力大小
2. 无法判断 robot 是否 lean 得合适以维持 smooth opening speed
3. 无法 track 门板的 revolving path

这些都需要 fine-grained closed-loop feedback，VR 手柄 + joystick + headset 这种 coarse 接口做不到。但 RL policy 在 sim 里直接以 50Hz 闭环 + 数百万步训练，能学到这种 implicit force-awareness。

实验设置：
- Robot 放在 door 前 1m，朝向 door 中心
- Yaw perturbation $\sim \text{Uniform}(-0.3, 0.3)$ rad（约 ±17°）
- Success 定义：穿过门并到达门框另一侧 1m 外

3 个 door category：
- Push lever（最简单）
- Pull lever（long-horizon，constrained space）
- Push bar（需要 forceful interaction 克服 spring hinge）

GRPO ablation（Figure 6a）：
- Teacher 上限 80-90%
- DAgger 后 student 卡在 50-70%（observability gap）
- GRPO 后 student 达到 80.8-85.8%，plateau 跟 teacher 上限对齐
- 说明 GRPO 确实 close 了 partial observability 造成的 gap

Hardware（Appendix D）：
- Unitree G1，29-DoF body + two 7-DoF 3-finger dexterous hands
- Intel RealSense D435i（禁用 depth，只输出 RGB）
- Intel i9-14900K + RTX 4090
- 50Hz inference

Teleop baseline（Appendix C）：
- PICO 4 Ultra + 两个 handheld controller
- 输出：3 个 upper-body SE(3) pose（head + 两手腕）、finger joint angles、waist height、planar nav command（$v \in \mathbb{R}^2$ + $\omega \in \mathbb{R}$）
- 用 Pinocchio（Carpentier et al. 2019, https://arxiv.org/abs/1807.09160）做 IK

---

## 7. 联想与相关工作链接

### 同一团队的 humanoid 工作
- **Homie** (Ben et al. 2025, https://arxiv.org/abs/2502.13013)：isomorphic exoskeleton cockpit 做 teleop 和 WBC 训练，DoorMan 直接 build on top
- **ASAP** (He et al. 2025, https://arxiv.org/abs/2502.01143)：aligning sim and real physics，做 agile humanoid whole-body skill
- **LeVERB** (Xue et al. 2025, https://arxiv.org/abs/2506.13751)：photorealistic demo + latent vision-language instruction，coarse whole-body task
- **Sonic** (Luo et al. 2025, https://arxiv.org/abs/2511.07820)：supersizing motion tracking for natural humanoid WBC
- **OmniH2O** (He et al. 2024, https://arxiv.org/abs/2406.08858)：universal human-to-humanoid teleop

### Loco-manipulation 方向
- **VBC** (Liu et al. 2024, https://arxiv.org/abs/2403.16967)：visual whole-body control for legged loco-manipulation，hierarchical distillation
- **RoboDuet** (Pan et al. 2025)：cooperative policy for whole-body legged loco-manipulation
- **UMI on Legs** (Ha et al. 2024, https://arxiv.org/abs/2407.10353)：manipulation-centric WBC 让 manipulation policy mobile
- **Wococo** (Zhang et al. 2024, https://arxiv.org/abs/2406.06005)：whole-body humanoid control with sequential contacts

### Door opening 专题
- **DoorGym** (Urakami et al. 2019, https://arxiv.org/abs/1908.01887)：早期 scalable door opening env，但用 wheeled platform + privileged sensing
- **StageAct** (Lee et al. 2025, https://arxiv.org/abs/2509.13200)：stage-conditioned imitation for humanoid door opening，跟 DoorMan 最直接对比，但 confined to same scene
- **Adaptive mobile manipulation** (Xiong et al. 2024, https://arxiv.org/abs/2401.14403)：open-world articulated object，但 adaptation-heavy
- **Legged manipulator door opening** (Zhang et al. 2025, https://proceedings.mlr.press/v270/zhang25g.html)：sim-to-real door，但 privileged sensing

### Visual sim-to-real 经典
- **Domain randomization** (Tobin et al. 2017, https://arxiv.org/abs/1703.06907)：最早的 DR
- **CAD2RL** (Sadeghi & Levine 2016, https://arxiv.org/abs/1611.04201)：单张 CAD 图训练即迁移
- **Dextreme** (Handa et al. 2023, https://arxiv.org/abs/2210.13739)：dexterous in-hand manipulation sim-to-real
- **Dextrah-RGB** (Singh et al. 2024, https://arxiv.org/abs/2412.01791)：stereo image zero-shot dexterous grasping，跟 DoorMan 思路最像
- **Extreme Parkour** (Cheng et al. 2023, https://arxiv.org/abs/2309.14341)：vision-based legged parkour
- **Perceptive locomotion** (Miki et al. 2022, ANYmal)：robust perceptive locomotion in the wild

### RL fine-tuning 方向
- **RoboCat** (Bousmalis et al. 2023, https://arxiv.org/abs/2306.11706)：self-improving generalist agent
- **TransIC** (Jiang et al. 2024, https://arxiv.org/abs/2405.10315)：sim-to-real with online correction
- **Residual RL** (Johannink et al. 2018, https://arxiv.org/abs/1812.03201)：经典 residual RL
- **Self-improving visuomotor** (Sharma et al. 2023, https://arxiv.org/abs/2303.01488)

### Procedural generation
- **Infinigen-Sim** (Joshi et al. 2025, https://arxiv.org/abs/2505.10755)：procedural articulated simulation-ready assets，DoorMan 借鉴但 IsaacLab-native 实现物理更准

---

## 8. 我的 intuition 与 take-away

这篇 paper 的核心 insight 在我看来有三层：

**第一层：teacher-student 不是简单蒸馏，而是 bootstrap**。传统 teacher-student 把 privileged policy 蒸馏到 visual policy 就结束了。DoorMan 在 distillation 之后加 GRPO 让 student 在自己的 partial observation 分布上 self-improve，这关键解决了"teacher 永远不会演示 compensatory behavior"的问题。从 RL 角度看，DAgger 解决了 state distribution mismatch，但没解决 observation information gap——GRPO 这一步是补上 information gap 的关键。这跟 LLM 里 RLHF 的"先 SFT 后 RL"思路异曲同工，但这里 SFT 是 DAgger，RL 用 GRPO（无 value function，对人形高维 action 友好）。

**第二层：sim 的 recoverability 是免费资源**。staged reset 利用了 simulator 可以任意 set state 的能力，这在 real world 是不可能的。这等于把 long-horizon 任务免费 decompose 成多个 short-horizon 任务，每段都 dense supervision。这跟 Go-Explore 思路一致，但用在了 contact-rich humanoid 任务上。这个 trick 在任何 multi-stage RL 任务里都该试，特别是 robot learning 里"探索 bottleneck"普遍存在。

**第三层：photorealism 是 sim-to-real 的 enabler，不是 nice-to-have**。Table 1 ablation 显示 no randomization 直接 5-20% 成功率，但即使 solid color + dome light 也能到 65-70%。说明 sim-to-real gap 主要不是 texture 细节，而是 lighting/appearance distribution 的 coverage。dome light 去掉掉 15-30%，比 texture 影响大，因为 lighting 直接决定 PBR material 在 camera 里的 appearance distribution。这给我们的 intuition：未来 sim-to-real 不一定要追求 pixel-perfect，而是要追求 appearance distribution coverage。

**潜在 limitation**：
1. Reward engineering 极重，28 个 term，每个 stage 都要 tune。future work 说想用 BC teacher 替代，这跟 LeVERB 思路一致。
2. 只测了门这一类 articulated object，drawer、knob、latch 等虽然 paper 提了但没实测
3. Camera 是 fixed egocentric，没处理主动 view adjustment 的更复杂场景（GRPO 学到的 compensatory behavior 还是 reactive 的）
4. 跟 human teleop 比用的是 VR + joystick 这种相对 coarse 的接口，跟 exoskeleton teleop（Homie）或直接模仿 human motion 的系统没直接比
5. 5233 dome light + 5 door type 的 variability envelope 在更复杂场景（多层楼、电梯、旋转门）是否够还未知

**对 future humanoid policy 的启示**：DoorMan 证明了一件事——只要 sim 足够 photorealistic + DR 足够 broad + pipeline 设计合理，纯 RGB sim-to-real 在 contact-rich whole-body task 上能超过 human teleop。这意味着 humanoid robot learning 的 bottleneck 正在从"real data采集"转移到"sim photorealism + DR pipeline + RL algorithm"。NVIDIA 的 IsaacLab + Omniverse RTX renderer 在这个方向上有 structural advantage，因为 photorealism 是 GPU 公司的天然护城河。

paper 给我最大的 intuition 是：**sim-to-real 不再是 research problem，而是 engineering scale problem**。当 DR 覆盖到 5233 个 lighting condition + 5 种 door + 全物理参数 randomize，sim-to-real 就能 work。这跟 LLM scaling law 类似——不是算法变聪明了，而是数据规模 + 多样性到了 critical threshold。
