---
source_pdf: From Reach to Insert Tactile-Augmented Precision Assembly under Sub-Millimeter
  Tolerances.pdf
paper_sha256: 3ff08430fb826521b802d82bb5f6b64577388cd5386775867421ce57421abfd5
processed_at: '2026-08-04T10:52:21-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，咱们用最接地气的大白话把这篇 paper 捋一遍。

其实这篇 paper 讲的事儿特别直白：**让机器人像人一样，闭着眼睛仅靠手感把一根针插进 0.05 毫米缝儿的针眼儿里，还既快又安全，不把针怼断。**

这篇 paper 的核心 motivation 来自于对人类行为的观察：你闭着眼摸黑插 USB 接口的时候，肯定先靠记忆大概摸到洞口，等碰到边缘了，再靠手指的触觉一点点拧巴着怼进去。这篇 paper 就是把这个直觉写成了算法。

---

### 1. 难点到底在哪？

0.05 mm 是什么概念？一根头发丝大概 0.07 mm。这就意味着机器人如果偏了一点点，插头就会卡死。一旦卡死，机器人要继续往下怼，力就会瞬间飙升到几十牛顿，不仅任务失败，搞不好还把传感器或者机械臂给弄坏了。

以前大家搞这个任务，要么纯靠视觉（看不清这么细微的误差），要么纯靠模仿学习（人类演示得再好，机器人自己一抖动，误差积累起来就懵了，而且它不会自己纠正），要么直接上 RL 在真机上硬训（太慢了，而且 RL 早期乱动容易把机器搞坏）。

---

### 2. 第一招：分两步走 (Reach + Insert)

既然一步到位太难，那就拆开。人怎么干，机器就怎么干。

**第一步叫 Reach (视觉主导)**：
就像你摸黑找 USB 洞。先用 Diffusion Policy 这个模型，看着摄像头，抓起随机放在桌子上的插头，直接导航到洞口附近。这步只管大方向，靠视觉完全够了，动作可以很丝滑。这步把巨大的探索空间（整个桌子）直接缩小到了洞口周围几厘米。

**第二步叫 Insert (触觉主导)**：
一旦到了洞口附近（算法里用一个 AABB 包围盒判断是否切换阶段），立马切断视觉主导，切到 RL 策略。这时候主要靠触觉反馈一点点试错、旋转、往下压。因为探索空间已经被第一步缩小了，所以 RL 学得特别快，真机上半小时就能学会。

---

### 3. 第二招：触觉信号的巧用 (Tactile Baseline & Asymmetric AC)

触觉这个东西，在机器人学习里其实是个双刃剑。你手上捏着插头，每次捏的位置哪怕偏一毫米，传感器读出来的基础力矩都不一样。如果直接把这个原始读数丢给 RL 的 Actor（动作网络），Actor 就会晕掉：这到底是接触力变了，还是我刚才抓歪了？它分不清，动作就会乱抖。

作者的解法特别漂亮：
首先，搞个 **Tactile Baseline (触觉基线)**。
公式是 $\Delta \mathbf{f}_t = \mathbf{f}_t - \bar{\mathbf{f}}$
- $\mathbf{f}_t$ 是当前时刻的 6 维力/力矩读数 $[F_x, F_y, F_z, M_x, M_y, M_z]^\top$
- $\bar{\mathbf{f}}$ 是刚抓完插头、还没动时候的读数平均值。
- 俩一减，得到了 $\Delta \mathbf{f}_t$。
这相当于把“抓歪了带来的初始偏差”给滤掉了，只保留“怼到洞口边缘产生的新接触力”。干干净净，网络最爱吃这种数据。

然后，搞个 **Asymmetric Actor-Critic (非对称演员-评论家)**。
RL 里有两个网络：决定动作的 Actor，和打分的 Critic。作者让 Actor 只看视觉，把 $\Delta \mathbf{f}_t$ 偷偷塞给 Critic 看。
为啥这么干？因为 Actor 如果看了触觉，容易学出一些奇怪的短路反应（比如一碰就猛退）。Critic 看了触觉，就能在心底默默评估：“哎哟，这个动作产生的接触力太大了，容易卡死，我给你打个低分”。Actor 听了 Critic 的低分，自然就学会了温柔的动作。这叫“只做不说”，Actor 保持着视觉的稳定输出，Critic 在背后用触觉做价值辅导。

---

### 4. 第三招：数据采样均衡 (Tactile Group Sampling)

RL 训练时有个坑：如果机器人某一种抓取姿势特别顺手，它就会一直用那种姿势，buffer 里存的全是那种姿势的数据。那些抓得比较歪、力矩偏大的数据很少。结果就是，只要某次抓歪了，Critic 没见过这种情况，直接傻眼给个错分，任务失败。

作者的解法是 **分桶采样**。
公式是 $g(\bar{\mathbf{f}}) = (b_x(\bar{M}_x), b_y(\bar{M}_y))$
- $\bar{M}_x, \bar{M}_y$ 是抓取后的基线力矩在 x 和 y 方向的分量。
- $b_x, b_y$ 是分桶函数，把力矩范围切成 $2 \times 4 = 8$ 个格子（bins）。
每次训练抽数据，先在这 8 个桶里均匀抽，再在桶里抽具体数据。这样不管抓得正还是歪，每个姿势都有均等的上场机会。Critic 见过了大世面，打分就准了，Q-disagreement（两个 Critic 网络的打分分歧）直线下降。

---

### 5. 技术细节深挖 (给 Karpathy 的干货)

咱们再看点更细节的东西，build 一下 intuition。

#### 5.1 Critic Warmup (评论家预热)
Sparse reward 是 RL 的噩梦。一开始 Q 网络全是随机初始化的，Actor 乱动，得不出 reward，就没法更新。作者用了一招 Monte Carlo (MC) Warmup。
公式是 $\mathcal{L}_{\text{MC}}(\phi) = \mathbb{E}_{(s,a,G) \sim \mathcal{D}_{\text{demo}}} \left[ \left( Q_\phi(s,a) - G \right)^2 \right]$
- $\mathcal{D}_{\text{demo}}$ 是人类演示数据。
- $G$ 是人类演示轨迹的总回报（就是 1，因为插进去了）。
- $Q_\phi(s,a)$ 是 Critic 当前的打分。
说白了，这就是在有标签的监督学习阶段，硬把 Q 网络往“人类最终得分”上去拟合。这一步把 Critic loss 从 0.5 直接干到 0.002。等真正开始 RL 探索的时候，Critic 已经是个有基本判断力的老手了，Actor 跟着它走少走很多弯路。

#### 5.2 RLPD Style Buffer 管理
作者用了两个 buffer：$\mathcal{D}_{\text{demo}}$（装人类演示和干预数据）和 $\mathcal{D}_{\text{on}}$（装机器人自己探索的数据）。每次更新，一半从 Demo 里抽，一半从 Online 里抽。这保证了策略永远不会跑偏太远，人类的高质量数据像一根缰绳一直拽着它。

#### 5.3 Human-in-the-loop (HITL)
真机训练 0.05 mm，很容易卡死炸机。作者接了个 SpaceMouse，看机器人要卡死了，人接管一下，推一把，这个 transition 也存进 $\mathcal{D}_{\text{demo}}$。这是一种极其高效的安全探索机制，30 分钟、1000-2000 步就能收敛。

---

### 6. 实验结果直觉

看 Table I 和 Fig 9 的数据：
- 纯 IL 在 0.05 mm 成功率 0%。因为误差积累直接崩盘。
- 纯 IL+RL (没触觉) 成功率 52%，但峰值力超过 50N，力矩超 160 N·mm。这叫“暴力怼”，虽然怼进去了，但机器和工件都受内伤。
- **Full 方法 (带触觉 Critic + 分组采样)**：成功率 67%，峰值力降到 20N，力矩降到 90 N·mm。力降了 60%，力矩降了 44%。

这个结果非常 Intuitive。因为 Critic 里有触觉，它知道“这条轨迹虽然插进去了，但是中间力太大了，扣分！”。Actor 慢慢就学会了避开高力的路径，寻找那种顺滑滑进去的动作。Safety 和 Success rate 都保住了。

---

### 7. 联想与 Hallucination (脑洞大开)

顺着这个思路，咱们可以发散很多有意思的方向：

1. **VLA (Vision-Language-Action) 的触觉化**：现在 VLA 模型（比如 RT-2, OpenVLA）火得不行，但全靠视觉。如果把这篇 paper 的 Asymmetric AC 思想搬过去，在 VLA 做 RL fine-tuning 的时候，给 Critic 接上触觉甚至本体感受的 privileged information，Actor 还是只输出 action chunk，会不会能在 contact-rich 任务上起飞？参考 [VLA-Touch](https://arxiv.org/abs/2507.17294)。
2. **World Model 的触觉注入**：现在 Dreamer 系列 或者 JEPA 这种 world model，如果把 tactile 预测作为 latent space 的一部分，是不是能学出一个“力学常识”的 world model？想象一下，模型在脑子里推演“如果我往左偏一点，接触力矩会变大，然后就卡死”，这比单纯预测下一帧像素强多了。
3. **Tactile Representation Learning**：这篇 paper 还是直接用的 6 维力矩 $\Delta \mathbf{f}$。如果换成 GelSight 那种高分辨率触觉图像，能不能用 MAE (Masked Autoencoder) 先在大量无监督抓取数据上 pre-train 一个 tactile encoder，然后再接进去？这绝对是个 low-hanging fruit。参考 [Dexterity from Touch](https://arxiv.org/abs/2303.12076)。
4. **Biological Plausibility**：人类大脑皮层处理触觉和视觉是分开的区域的，但在做精细操作时，是如何做 sensor fusion 的？这篇 paper 说 Actor 不用触觉只用视觉，Critic 用触觉。这有点像人脑的“小脑”负责无意识的精细动作调节（Critic 隐式参与），而“大脑”显式的决策只靠视觉规划？

---

### 8. 总结

把这篇 paper 浓缩成一句话：**人在插东西时，视觉负责导航，触觉负责微调；作者把触觉信号的 grasp offset 滤掉，只喂给 Critic 做安全打分，并且按抓取姿势分桶训练，让机器人在 0.05mm 的极端公差下，用半小时学会了温柔且高效地插拔。**

其实逻辑非常顺畅，没有太多花里胡哨的 trick，全都是对物理世界痛点的精准打击。Right, this is exactly how good robotics research should be done --- grounded in physical intuition, implemented with clean algorithmic design.

---
### Reference Web Links
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [RLPD (Efficient Online RL with Offline Data)](https://arxiv.org/abs/2205.12721)
- [Human-in-the-loop RL (Science Robotics 2025)](https://www.science.org/doi/10.1126/scirobotics.adl5774)
- [Dexterity from Touch (Tactile Pre-training)](https://arxiv.org/abs/2303.12076)
- [VLA-Touch](https://arxiv.org/abs/2507.17294)

---

# Paper 讲解: From Reach to Insert — Tactile-Augmented Precision Assembly under Sub-Millimeter Tolerances

这篇 paper 来自 Meng et al.（含 Long Cheng, IEEE Fellow），核心命题是：**在 0.05 mm 这种亚毫米级 clearance 下，如何让机器人自主完成 grasp-to-insert 的 peg-in-hole 任务，同时把 interaction force 压在安全范围内**。下面我会从 motivation → architecture → 公式细节 → 实验 → intuition 几个层面把它讲透。

---

## 1. Motivation & 任务设定的难点

### 1.1 任务设定本身就有挑战

大部分 peg-in-hole 文献假设 peg 已经被 grasp 好、并且已经对齐，只需做最后的 insertion（参考 Fig. 2 的 literature mapping）。这篇 paper 选了一个更真实的 setting：**grasp-to-insert**，即 peg 随机放在 5×5 cm 区域内，机器人要先 grasp 再 insert。

这带来两个 error accumulation 源：
- **peg 初始 pose 不确定** → reach 阶段需要 position generalization
- **grasp 引入 in-hand pose error** → 这个误差在 contact 时被放大，导致 eccentric contact、torque 爆炸、jamming

### 1.2 为什么 sub-millimeter 难

clearance $c$ 定义为 hole 直径减 peg 直径的差值（diametral clearance）。当 $c = 0.05\,\text{mm}$ 时：
- 任何 pose error $\delta \theta$ 都会被几何约束放大成 friction force $\sim \mu N \cdot \tan(\delta\theta)$ 量级的 jamming force
- vision 的分辨率（D405 的 depth noise 在几十 cm 距离上约为 0.5–1 mm）远不足以直接对齐
- IL 的 demonstration 覆盖度有限，一旦进入 OOD 状态就崩；RL 在 real robot 上 sample efficiency 差、且 exploration 不安全

这就是为什么 Fig. 2 里把这篇 paper 放在 grasp-to-insert × $c<0.1$ mm 的"最难象限"。

---

## 2. Two-Stage Framework 的设计 intuition

人类操作员的行为模式是：**vision 做开环 reach，contact 一旦发生就切到 tactile feedback 做 closed-loop correction**。这篇 paper 把这个直觉形式化成两阶段：

### Stage 1: Reach — Diffusion Policy (IL)

Reach policy $\pi_{\text{reach}}$ 用 diffusion policy 训练，输入是 end-effector pose + multi-view images（wrist D405 + global D435i），输出 short-horizon action sequence：

$$a_{t:t+H-1} \sim \pi_{\text{reach}}(\cdot \mid o_t^{\text{reach}})$$

- $a_{t:t+H-1}$：从时刻 $t$ 开始、长度为 $H=16$ 的 action chunk
- $o_t^{\text{reach}}$：reach 阶段的 observation，包括 proprioception + 两路 camera image
- $H$ 是 action horizon，receding-horizon 执行（类似 MPC 的 receding horizon control）

Diffusion policy 的好处是 **multimodal action distribution** —— 不同 peg 位置对应不同 reach 轨迹，diffusion 能建模这种多模态性，而 BC/Gaussian policy 会把多个 mode 平均掉导致轨迹漂移。100 条 demo × 5 geometries，200 epochs，ResNet-18 encoder，128×128 输入。

**Intuition**：reach 阶段不需要精细 contact 信息，vision 足够。把它从 RL 里剥离出来，能让 RL 只 focus 在 exploration space 极小的 contact-rich 区域，sample efficiency 大幅提升。

### Stage 2: Insert — SAC (RL) with Tactile Augmentation

Insert 阶段建模为 POMDP $\mathcal{M} = \{\mathcal{O}, \mathcal{A}, \rho, \mathcal{P}, r, \gamma\}$：
- $\mathcal{O}$：observation space（vision crop + tactile + proprioception）
- $\mathcal{A}$：action space（7-DoF delta pose）
- $\rho(o_0)$：initial state distribution（由 reach 阶段终止时的状态决定）
- $\mathcal{P}$：unknown transition dynamics（contact physics）
- $r$：sparse binary reward（式 2）
- $\gamma$：discount factor

Reward 极其 sparse：只有 insertion 成功才 $r=1$，否则 $0$。这是 RL 最难学的 setting。

### Stage Switching 机制

用 location-based switching（式 6–7）：

$$I_{\text{ins}}(\mathbf{p}) = \begin{cases} 1, & \mathbf{p} \in \mathcal{X}_{\text{init}} \\ 0, & \text{otherwise} \end{cases}$$

$$\pi(a \mid o) = (1 - I_{\text{ins}}) \pi_{\text{reach}} + I_{\text{ins}} \pi_{\text{insert}}$$

- $\mathbf{p} \in \mathbb{R}^3$：end-effector position
- $\mathcal{X}_{\text{init}} = \text{AABB}(\mathcal{P}_{\text{demo}} \cup \mathcal{P}_{\text{fail}})$：demo 端点 $\mathcal{P}_{\text{demo}}$ 和 reach-failure 点 $\mathcal{P}_{\text{fail}}$ 的并集的 axis-aligned bounding box
- AABB 会动态更新，但对 $\mathcal{P}_{\text{fail}}$ 做 statistical outlier rejection，防止异常 termination 把 bbox 撑爆

**Intuition**：用几何区域而非 contact 检测来切换 stage，是因为 reach policy 本身可能把 peg 带到 hole 附近但还没真正 contact。AABB 的动态扩展让 failure-driven 的 exploration 自然纳入 insert stage 的 jurisdiction。

---

## 3. 核心创新点详解

### 3.1 Tactile Baseline & Tactile Change

这是整篇 paper 最关键的 insight 之一。Paxini PX-6AX GEN3 tactile sensor 输出 6-DoF force/torque $\mathbf{f} = [F_x, F_y, F_z, M_x, M_y, M_z]^\top$。

问题：**不同 grasp 配置下，即使 peg 没动，tactile 读数也差很多**（因为 peg 在 gripper 里的位置每次都不一样，产生不同的 pre-load force/torque）。如果直接把 raw $\mathbf{f}$ 喂给 policy，输入方差极大，学习不稳定。

解决方案（式 8）：

$$\bar{\mathbf{f}} = \frac{1}{K} \sum_{k=0}^{K-1} \mathbf{f}_{t_b + k}$$

- $\bar{\mathbf{f}}$：post-grasp 稳定窗口内的 tactile 均值，作为 **baseline**
- $t_b$：gripper 闭合后进入稳定状态的起始 index
- $K$：窗口长度

然后定义 **tactile change**：

$$\Delta \mathbf{f}_t = \mathbf{f}_t - \bar{\mathbf{f}}$$

这个 $\Delta \mathbf{f}_t$ 把 grasp-induced offset 减掉，只保留 insertion-induced 的 contact 变化。

**Intuition**：这本质上是一个 **per-episode normalization**，类似 batch norm 的思想 —— 把 grasp-specific 的 baseline 当成 "DC component" 滤掉，只让 policy 学习 "AC component"（contact event）。Fig. 4 直观展示了不同 grasp pose 下 raw tactile 差异巨大，但 $\Delta \mathbf{f}$ 把它们对齐了。

### 3.2 Asymmetric Actor-Critic: Tactile in Critic, not Actor

这是另一个反直觉但合理的 design choice：

- **Actor** $\pi_\theta(a \mid o)$：只用 vision（object-centric crop）+ proprioception 生成 action
- **Critic** $Q_\phi(o, a)$：额外接入 $\Delta \mathbf{f}_t$ 做 value estimation

为什么 tactile 不进 actor？作者给的 reasoning：
1. Tactile 信号受 grasp offset 影响大，直接进 actor 会增加输入方差，destabilize policy learning
2. Actor 的 action 是 vision-grounded 的稳定行为，tactile 提供 contact-aware 的 value 评估即可
3. Asymmetric AC 在 literature 里已知能让 critic 学到更丰富的 state representation 而不增加 actor 的输入复杂度

**Intuition**：可以把 critic 看成一个 "teacher"，它能看到更多信息（包括 touch），用来判断 actor 的 action 好不好。Actor 是 "student"，只看 vision 就够了，因为最终执行时 vision 足以 generate corrective motion，tactile 的价值在于让 critic 给出更准确的 gradient 信号。这跟 model-based RL 里 value function 用 privileged info 的思路类似。

### 3.3 SAC 的标准 loss + 它们的变量含义

Critic loss（式 3）：

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(o,a,r,o',d)} \left[ \left( Q_\phi(o,a) - y \right)^2 \right]$$

- $\phi$：critic 网络参数
- $Q_\phi(o,a)$：当前 critic 对 $(o,a)$ 的 Q 值估计
- $y$：TD target（式 4）

Target value（式 4）：

$$y = r + \gamma (1-d) \mathbb{E}_{a' \sim \pi_\theta(\cdot|o')} \left[ Q_{\bar{\phi}}(o', a') - \alpha \log \pi_\theta(a' \mid o') \right]$$

- $r$：当前 reward
- $\gamma$：discount factor
- $d \in \{0,1\}$：terminal flag（episode 是否结束，如 insertion 成功）
- $\bar{\phi}$：target critic 参数（EMA 或 periodic copy，用于稳定训练）
- $\alpha$：temperature coefficient，控制 entropy 项的权重
- $\alpha \log \pi_\theta(a' \mid o')$：entropy 项，鼓励 exploration

Actor loss（式 5）：

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_o \left[ \mathbb{E}_{a \sim \pi_\theta(\cdot|o)} \left[ \alpha \log \pi_\theta(a \mid o) - Q_\phi(o, a) \right] \right]$$

- $\theta$：actor 参数
- 第一项 $\alpha \log \pi$：最大化 entropy（鼓励 stochastic policy）
- 第二项 $-Q_\phi(o,a)$：最大化 Q 值（选择高 value action）

**Intuition**：SAC 的核心是 maximum entropy principle —— 不光要 maximize return，还要 maximize policy entropy。这让 policy 在 contact-rich 场景下保持足够 exploration 来发现 corrective behavior，同时 Q 值引导它往高 return 方向收敛。

### 3.4 Critic Warmup with Monte Carlo Returns

Sparse reward 下，早期 TD learning 的 bootstrap error 极大（因为 target $y$ 依赖一个还没学好的 $Q_{\bar{\phi}}$）。作者用 **MC supervised warmup**（式 9）：

$$\mathcal{L}_{\text{MC}}(\phi) = \mathbb{E}_{(s,a,G) \sim \mathcal{D}_{\text{demo}}} \left[ \left( Q_\phi(s,a) - G \right)^2 \right]$$

- $G$：demo 轨迹的 pre-computed Monte Carlo return（即 $\sum_t \gamma^t r_t$ 沿整条 demo 轨迹累加）
- $\mathcal{D}_{\text{demo}}$：demonstration buffer
- $s$：state（包括 vision + tactile）
- $a$：demo action

**Intuition**：这相当于先用监督学习把 critic 拟合到 expert 的 return surface 上，给 RL 一个好的 value prior。Fig. 7 显示 critic loss 从 0.503 降到 0.002，predicted $Q_{\text{mean}} \approx 0.94$（与 demo 的 MC return $G \approx 0.94$ 一致）。这个初始化避免了早期 RL 的 "blind exploration"，让 actor 一开始就有有意义的 gradient。

### 3.5 Tactile Group Sampling

这是第三个核心创新。问题：off-policy replay buffer 里，某些 grasp 配置（torque 接近 0 的"好 grasp"）会被 over-represented，因为它们更容易成功 → 更多 transition 进 buffer。这导致 critic 对 rare grasp 配置的 value 估计不准。

解决方案（式 10）：

$$g(\bar{\mathbf{f}}) = \left( b_x(\bar{M}_x), b_y(\bar{M}_y) \right)$$

- $\bar{M}_x, \bar{M}_y$：post-grasp baseline torque 的 x/y 分量（这两个对 grasp pose 最敏感）
- $b_x(\cdot), b_y(\cdot)$：discretization functions，把 torque 范围分成 $N$ 和 $M$ 个 bin
- 实验中 $N=2, M=4$，共 $2 \times 4 = 8$ 个 tactile groups
- bin boundary 由 demo 数据的 empirical range 决定

Sampling 策略：**先 uniform 采样 group，再在 group 内 uniform 采样 transition**。

**Intuition**：这本质上是一个 **stratified sampling**，类似 importance sampling 但目标不是修正分布偏差，而是 **主动平衡 replay 分布**。Fig. 8 显示 tactile group sampling 的 Q-disagreement 显著低于 baseline，说明 critic 在不同 contact mode 下的 value 估计更一致。这跟 Prioritized Experience Replay 的思路不同 —— PER 优先 TD error 大的样本，而这里优先 **under-represented 的 grasp mode**。

### 3.6 RLPD-style Training & Human-in-the-Loop

两个 buffer：
- $\mathcal{D}_{\text{demo}}$：demo + human intervention 数据
- $\mathcal{D}_{\text{on}}$：online interaction 数据

每次 update 从两个 buffer 大致等比例采样（RLPD: Reinforcement Learning with Prior Data，[Ball et al. 2023](https://arxiv.org/abs/2205.12721)）。

Human-in-the-loop：当 policy 进入 unrecoverable 状态，操作员用 SpaceMouse 接管，产生 corrective action $a^{\text{inv}}$，这些 transition 存入 $\mathcal{D}_{\text{demo}}$。

**Intuition**：这跟 [Luo et al. 2025 Science Robotics](https://www.science.org/doi/10.1126/scirobotics.adl5774) 的 Human-in-the-loop RL 思路一致 —— 把人类 intervention 当成 high-value demo 注入 buffer，既保证 safety 又加速收敛。30 分钟训练就能达到 zero-intervention success，约 1000–2000 transitions。

---

## 4. 实验结果深度解读

### 4.1 Success Rate（Table I）

| Method | 1.5 mm | 0.25 mm | 0.05 mm |
|--------|--------|---------|---------|
| Pure IL | 76% avg | 24% avg | **0%** |
| IL+RL (no tactile) | ~97% | 79% | 52% |
| **Full (tactile critic + group sampling)** | ~97% | **94%** | **67%** |

关键观察：
- **1.5 mm**：所有方法都还行，因为 tolerance 大，IL 的误差积累不足以导致 failure
- **0.25 mm**：IL 崩到 24%，RL-based 方法开始体现优势
- **0.05 mm**：IL 完全失败（0%），vanilla IL+RL 52%，full method 67%

Pure IL 在 0.05 mm 完全失败的原因：tight tolerance 下，peg 在 gripper 里因 contact force 微小漂移 → in-hand pose error → 累积 → OOD → jamming。IL 没有 recovery 能力。

### 4.2 Interaction Force & Torque（Fig. 9）

这是 safety 维度的关键结果，在 0.05 mm 任务上：
- Vanilla IL+RL：peak normal force > 50 N，peak torque > 160 N·mm
- **Proposed method：peak normal force ≤ 20 N，peak torque ≤ 90 N·mm**
- **Force 降 60%，torque 降 44%**

**Intuition**：tactile critic 让 policy "感知到" contact force 在上升，于是学到 proactively suppress excessive force 的策略。Vanilla policy 因为 critic 看不到 tactile，只靠 vision 判断好坏，于是它可能学到 "硬怼" 的策略 —— 只要最终插进去就给 reward，不管中间 force 多大。Tactile critic 把 force 信息纳入 value 估计，相当于隐式地给 high-force trajectory 打了低分（因为 tactile pattern 异常的 transition 的 Q 值会被 critic 压低）。

### 4.3 Training Efficiency（Fig. 6, 7, 8）

- **First zero-intervention success**：tactile group sampling 在 ~1000 transitions 达到，vanilla 更晚且波动大
- **Critic loss**：稳定在 $10^{-3}$ 到 $10^{-2}$ 范围
- **Q-disagreement**：tactile group sampling 显著低于 baseline

**Intuition**：Q-disagreement 低意味着 ensemble critic 之间一致性高 → value 估计置信度高 → actor 的 gradient 噪声小 → 训练稳定。这是 tactile group sampling 平衡 replay 分布的直接收益。

---

## 5. 整体架构图解析（Fig. 3）

Fig. 3 展示了完整的 training pipeline：

1. **Reach demos** $\mathcal{D}_{\text{demo}}^{\text{reach}}$ → 训练 diffusion policy $\pi_{\text{reach}}$
2. **Insertion demos** $\mathcal{D}_{\text{demo}}^{\text{ins}}$ → critic warmup（MC loss）+ 初始化 $\mathcal{D}_{\text{demo}}$
3. **Online rollout**：
   - 执行 $\pi_{\text{reach}}$ grasp + approach，记录 tactile baseline $\bar{\mathbf{f}}$
   - 当 $\mathbf{p} \in \mathcal{X}_{\text{init}}$，切换到 $\pi_{\text{insert}}$
   - 执行 $a_t^{\text{RL}} \sim \pi_{\text{insert}}(\cdot \mid o_t)$（或 human intervention $a_t^{\text{inv}}$）
   - 存入 $\mathcal{D}_{\text{on}}$（和 $\mathcal{D}_{\text{demo}}$ 如果是 intervention）
4. **Update**：每 $U$ 步，用 tactile group sampling 从 $\mathcal{D}_{\text{demo}} \cup \mathcal{D}_{\text{on}}$ 采样 mini-batch，更新 SAC
5. **Failure handling**：失败时把 endpoint 加入 $\mathcal{P}_{\text{fail}}$，更新 $\mathcal{X}_{\text{init}}$

Algorithm 1 的 line 10 很有意思：**Reset $p_0 \sim \text{Unif}(\mathcal{P}_{\text{fail}})$ if $\mathcal{P}_{\text{fail}} \neq \emptyset$**。这意味着失败后会从 failure 点附近重新开始，而不是从头 grasp。这是一个 **curriculum-like restart strategy** —— 让 policy 在它最弱的区域多练。

---

## 6. 与 Related Work 的定位（Fig. 2）

Fig. 2 把 prior work 按 task setting × tolerance 分类：
- **Peg fixed**（最简单）：[18] Lee et al., [2] Xie et al., [4] Xu et al.
- **Pre-grasped**（中等）：[38] TLA, [22] FORGE, [12] Triyonoputro, [33] TacDiffusion, [19] Luo, [28] Inoue, [29] Dong, [35] MimicTouch, [26] VT-Refine, [20] Schoettler, [23] IndustReal
- **Grasp-to-insert**（最难，本工作）：唯一在 $c < 0.1$ mm 象限的工作

**Intuition**：这个定位很清晰 —— 大部分 tight-tolerance 工作都假设 peg 已经 grasp 好并对齐，把 grasp 引入的误差剥离了。本工作把 grasp 误差纳入考量，更接近真实工业场景。

---

## 7. 我的几个 Intuitive Takeaways

### 7.1 Tactile 的正确用法是 "value guidance" 而非 "action guidance"

这篇 paper 最深的 insight 是：**tactile 信号有 grasp-dependent offset，直接进 actor 会引入方差；但放进 critic 做 value estimation 则能提供 contact-aware 的 gradient 信号**。这跟 privileged learning 的思路相通 —— critic 可以看 "上帝视角" 信息，actor 只看可执行信息。

参考类似思路：
- [Asymmetric Actor Critic for Image-based Robot Learning](https://arxiv.org/abs/2010.02824)
- [Deep RL with Privileged Information for Robotics](https://arxiv.org/abs/2106.05053)

### 7.2 Tactile Baseline 是 per-episode normalization

$\Delta \mathbf{f} = \mathbf{f} - \bar{\mathbf{f}}$ 这个操作看似简单，但它解决了 tactile sensing 在 robot learning 里的一个根本问题 —— **sensor reading 不 stable，受 grasp state 强烈影响**。这跟 vision 里的 image normalization、batch norm 是同一个 family 的 trick。

### 7.3 Group Sampling 是 stratified replay

这不是 PER（prioritized by TD error），而是 **stratified by contact mode**。目标是 distribution balancing 而非 sample efficiency。这在 tactile 这种 long-tail 分布的场景里很重要 —— 大部分 transition 是 "no contact" 或 "light contact"，真正有信息量的 "critical contact" 很少。

### 7.4 Two-Stage 是 modality-aware decomposition

Reach 用 vision（global info），insert 用 tactile（local contact info）。这种 decomposition 跟 human behavior 一致，也跟 sensor 的物理特性一致 —— vision 在远距离有用，contact 时被遮挡且分辨率不够；tactile 只在 contact 时有意义。把两个 stage 用 location-based switching 衔接，是一个干净的工程方案。

---

## 8. 可能的延伸与局限

### 局限
1. **每个 geometry 需要单独训练 reach policy**（100 demos × 5 geometries = 500 demos），generalization 到 unseen geometry 未验证
2. **0.05 mm 下 success rate 67%**，离工业级 99.9% 还有距离
3. **Human-in-the-loop 30 分钟**虽然短，但仍需人工介入，完全 autonomous 的 cold start 未实现
4. **Tactile sensor 是 Paxini PX-6AX**，专用的 high-end sensor，能否迁移到 cheaper tactile sensor（如 GelSight Mini）未讨论

### 延伸方向
1. 把 tactile representation 用 self-supervised pre-training（如 [Dexterity from Touch, Guzey et al.](https://arxiv.org/abs/2303.12076)）替代 raw $\Delta \mathbf{f}$
2. 用 VLA model（如 [TLA](https://arxiv.org/abs/2503.08548)）做 reach stage，实现跨 geometry generalization
3. 把 tactile critic 扩展到 multi-part assembly sequence
4. 用 active tactile exploration（[Active Extrinsic Contact Sensing, Kim & Rodriguez](https://arxiv.org/abs/2011.04922)）替代被动 tactile feedback

---

## 9. 相关 Reference Links

- **Diffusion Policy**: [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/) (Chi et al., RSS 2023)
- **Soft Actor-Critic**: [https://arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290) (Haarnoja et al., ICML 2018)
- **RLPD (Efficient Online RL with Offline Data)**: [https://arxiv.org/abs/2205.12721](https://arxiv.org/abs/2205.12721) (Ball et al., ICML 2023)
- **Human-in-the-loop RL (Science Robotics)**: [https://www.science.org/doi/10.1126/scirobotics.adl5774](https://www.science.org/doi/10.1126/scirobotics.adl5774) (Luo et al., 2025)
- **IndustReal (Sim-to-Real Assembly)**: [https://arxiv.org/abs/2305.17110](https://arxiv.org/abs/2305.17110) (Tang et al., RSS 2023)
- **FORGE (Force-Guided Exploration)**: [https://arxiv.org/abs/2501.07924](https://arxiv.org/abs/2501.07924) (Noseworthy et al., RAL 2025)
- **AutoMate (Specialist & Generalist Assembly)**: [https://arxiv.org/abs/2406.16600](https://arxiv.org/abs/2406.16600) (Tang et al., RSS 2024)
- **Making Sense of Vision and Touch**: [https://arxiv.org/abs/1907.13098](https://arxiv.org/abs/1907.13098) (Lee et al., ICRA 2019)
- **VT-Refine (Visuo-Tactile Assembly)**: [https://arxiv.org/abs/2510.08093](https://arxiv.org/abs/2510.08093) (Huang et al., CoRL 2025)
- **Dexterity from Touch (Tactile Pre-training)**: [https://arxiv.org/abs/2303.12076](https://arxiv.org/abs/2303.12076) (Guzey et al., 2023)
- **Reactive Diffusion Policy**: [https://arxiv.org/abs/2504.12222](https://arxiv.org/abs/2504.12222) (Xue et al., RSS 2025)
- **TacDiffusion**: [https://arxiv.org/abs/2412.16861](https://arxiv.org/abs/2412.16861) (Wu et al., ICRA 2025)
- **Paxini Tactile Sensor**: [http://www.paxini.com/](http://www.paxini.com/)
- **Franka Emika Panda**: [https://www.franka.de/](https://www.franka.de/)
- **Intel RealSense D405**: [https://www.intelrealsense.com/depth-camera-d405/](https://www.intelrealsense.com/depth-camera-d405/)

---

## 10. 总结

这篇 paper 的核心 contribution 可以浓缩成三句话：

1. **Two-stage decomposition** 把 vision-dominant reach 和 tactile-dominant insert 分开，让 RL 只 focus 在 contact-rich 的小空间里，sample efficiency 飙升（30 分钟 real-robot training）。

2. **Tactile critic + tactile baseline** 把 tactile 信号放进 critic 而非 actor，通过 per-episode normalization（$\Delta \mathbf{f} = \mathbf{f} - \bar{\mathbf{f}}$）消除 grasp offset，让 value estimation 变 contact-aware，force 降 60%。

3. **Tactile group sampling** 用 stratified replay 平衡 grasp 配置分布，让 critic 在 rare contact mode 下也有稳定 value 估计，Q-disagreement 显著降低。

在 0.05 mm clearance 这个"地狱难度"下做到 67% success rate 且 force 受控，是一个 solid 的 real-world RL 结果。Limitation 在于 per-geometry training 和离工业级可靠性还有差距，但作为一个 build intuition 的 case study，它清晰地展示了 **如何把 tactile 正确地 inject 到 RL pipeline 里**。
