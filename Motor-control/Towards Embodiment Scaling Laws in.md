---
source_pdf: Towards Embodiment Scaling Laws in.pdf
paper_sha256: e7bf8b5880d24d03365c048b08b6e281d6caa4e71cde5d7b13532ae9acfc7538
processed_at: '2026-08-12T17:04:03-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 一句话总结

这帮人搞了**一千个不同的机器人物理形态**，训练**一个policy**控制它们全部，然后这个policy能直接迁移到真机Go2和H1上——连膝盖关节被人故意卡住都能瘸着走。

核心发现：**增加机器人物理形态的多样性**比**增加同一机器人的数据量**更管用。这是一条新的scaling law。

---

## 为什么要做这件事

想象一下：你训了个Go2的policy，跑得挺好。但Go2装了个新传感器、腿磨损了、或者你想换Anymal来用——policy就废了，得重训。

这很蠢。人类换双鞋不会突然不会走路。

现实世界机器人会变：磨损、改装、损坏、升级。你希望的是**一个policy吃所有形态**。但怎么训？在5个机器人上训不够看出规律，得至少1000个才能看出scaling trend。

之前没人做到过这个规模。real world 上限是10个左右，simulation 上限是100个左右。这篇冲到了 **1012个**。

---

## 怎么搞出1000个机器人

手工建模1000个URDF不现实。他们用**程序化生成**：拿几个base component（Table 4/5），在三个维度上做Cartesian product。

**三个变化维度**：

1. **Topology**：每条腿0/1/2/3个膝盖关节。0个膝盖 = thigh直接连foot（没有calf）。4条腿 × {0,1,2,3} = 已经16种拓扑。

2. **Geometry**：每个link的长度/尺寸scaling factor。比如thigh可以是 {0.4, 0.8, 1.0, 1.2, 1.6} × 基础长度。Foot可以是1×或2×大小。

3. **Kinematics**：膝盖关节的活动范围scaling {0.2, 0.6, 1.0}。0.2就是膝盖几乎僵硬，1.0是正常范围。

把这三个维度组合起来，348个humanoid + 332个quadruped + 332个hexapod = **1012个机器人**。叫**GENBOT-1K**。

每个机器人和真实的Go2/H1都至少差几厘米或一个关节配置——故意不让训练集包含真机，逼policy学泛化能力。

---

## 核心难题：关节数都不一样怎么训

这是这篇最聪明的地方。

不同机器人关节数不同。Quadruped有12个关节，hexapod有18个，humanoid可能20+。传统的policy网络输入输出维度固定，没法直接搞。

他们的解法叫**URMA**，核心思想：**把机器人看成"关节的集合"**，而不是固定维度的向量。

**Observation拆两半**：

- **General observation $o_g$**：固定20维，所有机器人都有。包括躯干速度、重力方向、指令速度、PD gains、总质量、机器人尺寸等。
- **Joint observation $o_j$**：每个关节3维（关节角度、角速度、上一步的action）。关节数变了，这部分维度变。

**关键公式（公式3）**：

$$\bar{z}_{\mathrm{joints}} = \sum_{i \in J} z_j, \quad z_j = \frac{\exp(f_\phi(d_j)/\tau)}{\sum_{L_d} \exp(f_\phi(d_j)/\tau)} f_\psi(o_j)$$

翻译成人话：

- $d_j$：这个关节的"身份证"，18维。包括它在身体里的相对位置、旋转轴、最大扭矩、最大转速、关节限位、PD gains等。
- $f_\phi(d_j)$：把"身份证"编码成一个向量。
- $f_\psi(o_j)$：把这个关节当前的状态（角度、速度）编码。
- 那个softmax fraction：**根据关节类型给attention权重**。膝盖和髋关节权重不同，因为它们作用不同。
- $\bar{z}_{\mathrm{joints}}$：所有关节信息加权汇总成一个全局向量。

**直觉**：这等于告诉网络"这是膝盖，位置在腿中间，最大扭矩300，所以它的动作应该这样"。不同机器人的膝盖通过同样的description encoding在latent space对齐。

**Action生成（公式4）**：

$$a_j = \mu_\nu(g_\omega(d_j), \bar{z}_{\mathrm{action}}, z_j)$$

解码端也是per-joint的。$bar{z}_{\mathrm{action}}$是"全局意图"（比如"以1m/s前进"），$g_\omega(d_j)$告诉decoder"这是膝盖，该输出什么角度"，每个关节独立decode一个action。

这就是"**全局策略头部 + 局部执行机构适配**"模式。一个policy学全局意图，每个关节根据自己身份独立执行。

他们加了**multi-head attention（3个head）**，让网络并行关注不同特征（比如一个head看拓扑、一个看几何、一个看运动学）。

最终网络**只有2.1M参数**——非常小。说明正确的inductive bias比堆参数重要得多。

---

## 训练流程：两阶段

直接在1000个机器人上做RL不现实：变长obs/action让PPO的batch计算崩溃，不同机器人reward scale不同。

他们用**student-teacher蒸馏**：

### Stage 1: 每个机器人单独训RL expert

- 框架：NVIDIA Isaac Lab
- 算法：PPO
- **4096个并行环境** per embodiment
- 160张RTX 4090/3090
- 5天训完所有expert
- 总计 **2万亿仿真步**

每个morphology class共享一套hyperparameter（不可能给1000个机器人逐个调参）。

**Reward（Table 2）18项**，关键的几个：

- 前进速度tracking：$r = \exp(-|v_{xy} - c_{xy}|^2/0.25)$
- 防颠簸：$-|v_z|^2$
- 关节靠近nominal pose：$-|q - q_{\mathrm{nominal}}|^2$
- Action平滑：$-|a_t - 2a_{t-1} + a_{t-2}|^2$（二阶差分，防抖动）
- 对称性：左右脚交替着地，不能同侧乱跳
- Self-collision惩罚

**Curriculum**：domain randomization强度从0线性升到1。先在简单环境学基本步态，再逐步适应扰动。如果没摔倒且tracking误差小，coefficient +0.01，否则-0.01。

**Domain randomization（Table 1）**：电机强度±50%、PD gains ±50%、friction [0.05, 2.0]、附加质量±2kg、重力±2 m/s²、随机推力±1 m/s——覆盖真机部署可能遇到的各种扰动。

### Stage 2: 蒸馏成单一cross-embodiment policy

每个expert跑600步 × 4096 envs → 总共**19.8亿样本**，约5TB h5py数据。

Loss就是MSE：

$$\mathcal{L}_{\mathrm{BC}} = \mathbb{E}_{(s_t, e, a_t) \sim \mathcal{D}} \left[\|\pi(s_t, \phi(e)) - a_t\|^2\right]$$

学生policy通过$\phi(e)$（embodiment descriptor）conditioning，817个机器人共享参数。

**工程细节**：因为不同机器人obs/action维度不同，没法直接拼batch。他们用**gradient accumulation**：8个minibatch累积梯度再更新，平衡各机器人贡献。更好的方法应该是zero-padding到uniform batch，但实现复杂，他们没做。

一张H100训一周搞定。

---

## 实验结果：scaling 真的存在

### Finding 1: Embodiment越多，泛化越好

Figure 4里7条曲线，关键对比是C4 vs C8。

**C4（embodiment scaling）**：固定数据量per embodiment，增加embodiment数量从5%到100%。reward从18涨到接近30，**翻倍**。

**C8（data scaling）**：固定5%embodiment，增加每个embodiment的数据量从1×到20×。0.2就饱和了，后面没收益。

**结论**：**堆数据没用，堆形态有用**。这是这篇paper最核心的发现。

### Finding 2: 跨形态训练比单形态强2-5倍

把只在quadruped上训的policy在混合测试集上跑（C5），差得离谱。把三个class合训的policy在混合测试集上跑（C4），好2-5倍。

说明跨class训练让policy学到更抽象的"机器人控制原理"，比如"维持重心"、"避免自碰撞"——这些原理跨形态通用。

### Finding 3: 越难的形态越受益于scaling

Quadruped和hexapod在100个embodiment就饱和。Humanoid持续提升，没饱和。

Humanoid是两足，天然不稳定，控制空间复杂。需要更多形态变体才能覆盖足够的"动态平衡策略"。这和LLM里难任务受益更多是一个道理。

### Finding 4: 真机zero-shot成功

训了817个仿真机器人的policy直接上真机：

**Go2**：草地、鹅卵石、砾石上都走得稳。前进后退都行。

**最骚的实验**：故意把Go2右后腿膝盖的joint limit卡到正常的20%/40%/60%。

怎么卡的？软件层面：每次policy输出action，如果超出限制范围就project回boundaries，同时把PD gains拉高到$K_p=60, K_d=1$强行把关节推回安全区。

Policy**自动形成了瘸子步态**——把受限腿保持靠后位置，其他腿补偿。这说明policy真的理解了"这条腿活动受限"，不是死记硬背。

**H1 humanoid**：实验室橡胶地面前进后退正常，侧向行走慢但稳定。比Go2稍差，作者说训练集里humanoid多样性还不够。

### Finding 5: Latent space学到了"机器人分类学"

对policy的$\bar{z}_{\mathrm{action}}$做t-SNE：

- 三个morphology class**自然聚类**
- 大簇按knee关节数分小子簇
- 子簇内按geometry/kinematics细划分

Policy**自发学会了**按形态、拓扑、几何组织机器人。这意味着网络内部建了一个"机器人的功能分类树"，不是flat lookup table。

这暗示未来可以在latent space里做morphology co-design——优化机器人结构。

### Finding 6: OOD测试

训练时knee limit scaling只有{0.2, 0.6}，测试时推到{0.1, 0.001}：

| Class | 0.6 | 0.2 | 0.1 | 0.001 |
|---|---|---|---|---|
| Humanoid | 19 | 14 | 16 | 4 |
| Quadruped | 49 | 36 | 45 | 26 |
| Hexapod | 34 | 21 | 28 | 20 |

中度OOD（0.1）下降温和。极端OOD（0.001）humanoid崩了，quadruped/hexapod相对鲁棒——因为四条腿/六条腿有冗余，两腿没有。

---

## 为什么这个工作重要

之前robot foundation model的叙事是"scale up data and tasks"。这篇说**还要scale embodiment**。

Open X-Embodiment搞了22个机器人，已经觉得很大了。这篇直接1012个，跨3个形态学class，并且证明继续scale还能继续涨。

更关键的是给出了**架构recipe**：URMA这种per-joint attention + per-joint decode的模式，2.1M参数就能跨1000个机器人。这意味着不用堆巨无霸模型，正确的inductive bias更重要。

真机实验里policy适应knee limit变化自动调整步态，这指向一个未来：**可重构机器人**（modular robot）的软件基础。你换了个关节、加了个模块，policy还能用。

---

## 局限

1. 只做了平地locomotion。视觉manipulation或loco-manipulation没碰。
2. 生成维度还窄：mass distribution、damping、actuation type都固定。扩大可能进一步提升泛化。
3. 真机只测了Go2和H1两个。modular robot没测。
4. BC的gradient accumulation是工程妥协，zero-padding uniform batch可能更好但没实现。
5. H1表现比Go2差，说明humanoid训练集还需要更多样化。

---

## 我的take

这篇工作最有意思的不是sim-to-real成功——那个大家都做得到。有意思的是**scaling law本身**。

embodiment scaling 和 data scaling 是正交的两个轴。data scaling很快就饱和，embodiment scaling持续有效。这意味着未来robot foundation model的路线图应该是：

1. **程序化生成**大量diverse embodiment（不止locomotion，还有arm/hand/whole-body）
2. **per-component attention架构**（URMA-style）处理变长结构
3. **两阶段训练**：per-embodiment RL expert + cross-embodiment BC distillation
4. **继续scale**：从$10^3$到$10^4$到$10^5$

如果这个trend持续，"generalist robot policy"的可行性就有了empirical support。这篇是第一步的实证。

参考链接：
- 项目主页: https://embodiment-scaling-laws.github.io
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- URMA原paper (CoRL 2024): https://proceedings.mlr.press/
- PPO: https://arxiv.org/abs/1707.06347
- GenLoco: https://proceedings.mlr.press/v205/feng23a.html
- GET-Zero: https://arxiv.org/abs/2407.15002
- π0: https://arxiv.org/abs/2410.24164
- RT-2: https://robotics-transformer2.github.io/
- GR00T N1: https://arxiv.org/abs/2503.14734

---

# Embodiment Scaling Laws in Robot Locomotion 深度解析

## 核心论点：一种新的 scaling 维度

这篇 paper 探讨的是一个此前没有被系统研究过的 scaling 维度。在 vision 和 language 领域，我们习惯了 data scaling 和 model scaling；在 robotics 中，已有大量工作 scaling tasks 和 environments。但 **embodiment scaling**——即增加训练时使用的 **不同机器人形态的数量** ——这个维度还是空白。

作者假设存在一条 **embodiment scaling law**：在更多 diverse 的 embodiment 上训练 policy，能提升对 unseen embodiment 的泛化能力。直觉上，跨 embodiment 训练迫使 policy 学习 **结构上可迁移的控制策略**，而非记住单个机器人特定的动力学。

这个假设的动机很实际：真实机器人在部署中会因损伤、老化、制造差异、工具使用、升级而改变形态。如果 policy 能跨 embodiment 泛化，就能利用异构部署数据形成 data flywheel，这是构建 generalist robot 的关键路径。

参考链接：
- Paper page: https://embodiment-scaling-laws.github.io
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Octo: https://octo-models.github.io/

---

## Embodiment 的形式化定义

Embodiment $e \in \mathcal{E}$ 被定义为一个三元组：

$$e = \langle \mathcal{G}, \mathcal{T}, \mathcal{K} \rangle$$

- $\mathcal{T}$（topology）：joint 的数量和连接关系（例如膝盖关节数 = 0, 1, 2, 3）
- $\mathcal{G}$（geometry）：link 的形状与尺寸（thigh/calf 长度、foot 大小、torso 大小）
- $\mathcal{K}$（kinematics）：joint 类型与运动范围（例如 knee joint limits 的 scaling factor）

每个 embodiment 对应一个 MDP $\mathcal{M}_e = \langle S_e, \mathcal{A}_e, P_e, R_e, H \rangle$。由于 $S_e$ 和 $\mathcal{A}_e$ 大小不同，policy 必须处理 **变长输入输出**。Locomotion 中 policy 额外 conditioned on x-y-yaw velocity command $v_t \in \mathbb{R}^3$：

$$a_t \sim \pi(s_t, \phi(e), v_t)$$

其中 $\phi(e)$ 是 embodiment descriptor（描述该机器人的固定动力学/运动学属性）。

训练目标：

$$\pi_{\mathrm{train}}^* = \arg\max_\pi \mathbb{E}_{e \in \mathcal{E}_{\mathrm{train}}} \mathbb{E}_{\tau \sim \pi} \left[\sum_{t=0}^{H} R_e(s_t, v_t, a_t)\right] \tag{1}$$

测试目标（在 held-out 20% embodiment 上）：

$$J_{\mathrm{test}}(\pi_{\mathrm{train}}^*) = \mathbb{E}_{e \in \mathcal{E}_{\mathrm{test}}} \mathbb{E}_{\tau \sim \pi_{\mathrm{train}}^*} \left[\sum_{t=0}^{H} R_e(s_t, v_t, a_t)\right] \tag{2}$$

**Scaling hypothesis**：$|\mathcal{E}_{\mathrm{train}}|$ 增大 → $J_{\mathrm{test}}$ 增大。

---

## GENBOT-1K：程序化 Embodiment 数据集

为了验证 scaling 假设，需要至少 $\sim 10^3$ 个 embodiment。先前工作上限仅 $\sim 10^1$（real world）或 $\sim 10^2$（simulation）。作者程序化生成了 **1,012 个机器人**：348 humanoids + 332 quadrupeds + 332 hexapods。

### 三个维度的变化（Table 6）

| Variation Type | Parameter | Candidate Values |
|---|---|---|
| Topology | Number of knee joints | {0, 1, 2, 3} |
| Geometry | All link size scaling | {0.8, 1.0, 1.2} |
| Geometry | Thigh link length | {0.4, 0.8, 1.0, 1.2, 1.6} |
| Geometry | Calf link length | {0.4, 0.8, 1.0, 1.2, 1.6} |
| Geometry | Foot link size | {1.0, 2.0} |
| Geometry | Torso link size (humanoid only) | {0.4, 0.8, 1.0, 1.2, 1.6} |
| Kinematics | Knee joint limits scaling | {0.2, 0.6, 1.0} |

**Intuition**：通过 topology × geometry × kinematics 三维度的 Cartesian product，能以很少的 base components（Table 4/5）扩展出大量形态学变体。例如膝盖关节数 0 表示 thigh 直接连接 foot（无 calf link），这模拟了不同构型的腿。Reference robot（1.0× 配置）被排除在训练集外，保证测试集与 Go2/H1 有差异（每个 humanoid 关节偏离几厘米，整体高度差约 10 cm）。

参考链接：
- GenLoco (类似 generation 思想): https://proceedings.mlr.press/v205/feng23a.html
- GET-Zero (graph embodiment): https://arxiv.org/abs/2407.15002

---

## URMA：跨 Embodiment Policy 架构

### 核心问题：变长 observation/action space

不同机器人关节数不同，state 和 action 维度变化。URMA（Unified Robot Morphology Architecture）的核心 trick 是把 observation 分成两类：

- **General observations $o_g$**：固定 20 维，包括 trunk linear velocity, gravity vector, command velocities, PD gains, action scaling, total mass, robot dimensions, #joints, feet size
- **Joint-specific observations $o_j$**：每个关节 3 维，包括 joint angle, joint velocity, previous action of that joint

### Joint Encoding via Attention（公式 3）

$$\bar{z}_{\mathrm{joints}} = \sum_{i \in J} z_j, \quad z_j = \frac{\exp(f_\phi(d_j)/\tau)}{\sum_{L_d} \exp(f_\phi(d_j)/\tau)} f_\psi(o_j) \tag{3}$$

变量含义：
- $J$：当前 embodiment 的关节集合
- $d_j$：joint description vector（18 维：相对笛卡尔位置、旋转轴、nominal angle、max torque、max velocity、position limits、P/D gain、action scaling、robot mass、dimensions）
- $f_\phi$：joint description encoder，输出 latent dimension $L_d$
- $f_\psi$：joint observation encoder
- $\tau$：learnable softmax temperature
- $z_j$：单关节的 latent
- $\bar{z}_{\mathrm{joints}}$：加权求和后的"全局关节 latent"

**Intuition**：这是一个 attention pooling 操作。每个关节根据其 description（类型、位置、能力）得到一个 attention weight，再对 observation encoding 做加权求和。这等价于让 policy "知道" 当前关节在身体里的"角色"，从而把 6-legged hexapod 的某个 hip 和 4-legged quadruped 的 hip 在 latent space 里对齐。

论文扩展原 URMA 为 **multi-head attention（3 heads）**，让 policy 并行关注不同的 joint-level features，更好地捕获 inter-joint 复杂依赖。

### Action Decoding（公式 4）

$$a_j = \mu_\nu(g_\omega(d_j), \bar{z}_{\mathrm{action}}, z_j) \tag{4}$$

- $g_\omega$：action encoder for joint descriptions
- $\bar{z}_{\mathrm{action}} = h_\theta(o_g, \bar{z}_{\mathrm{joints}})$：core network 输出的 action latent
- $\mu_\nu$：最终 action decoder
- 对每个关节，把 action latent 和该关节的 description encoding 拼起来，独立 decode 出该关节的 action

**Intuition**：解码端也是 per-joint 的。$\bar{z}_{\mathrm{action}}$ 是"全局意图"（如"以 1 m/s 前进"），而 $g_\omega(d_j)$ 告诉 decoder "这是膝盖，应该输出什么角度"。这种"global intent + local decoder"模式是跨 embodiment 泛化的关键——一个全局策略头部 + 局部执行机构适配。

### 架构修改（相比原 URMA）
1. multi-head attention（3 heads）
2. 移除 foot-specific attention encoder（不是所有真机都有 foot pressure sensor）
3. 直接输出 action（不输出 std + Gaussian sampling，因为 BC 不需要 stochastic policy）
4. 给 $o_g$ 加额外 encoding layer 投影到更高维 latent
5. feedforward layers 加宽到 2× hidden dimension

最终模型仅 **2.1M 参数**，是 compact 网络，但 inductive bias 极强。

参考链接：
- 原 URMA paper: https://proceedings.mlr.press/
- Attention is All You Need: https://papers.nips.cc/paper/7181-attention-is-all-you-need

---

## Two-Stage Learning Pipeline

### Stage 1: Per-Embodiment RL Expert Training

- 框架：NVIDIA Isaac Lab
- 算法：PPO
- 规模：4096 parallel environments / per embodiment
- 硬件：160 × NVIDIA RTX 4090/3090
- 时间：~5 天
- 总计：**2 trillion simulation steps**
- 每个形态学类共享一套 hyperparameters（避免 1000+ 次调参）

**Reward function（Table 2）共 18 项**，关键项包括：
- T1: $r_{xy} = \exp(-|v_{xy} - c_{xy}|^2 / 0.25)$，系数 2.0
- T2: $r_{yaw} = \exp(-|\omega_{\mathrm{yaw}} - c_{\mathrm{yaw}}|^2 / 0.25)$，系数 1.0
- T3: $-|v_z|^2$，系数 2.0（防止上下颠簸）
- T6: $-|q - q^{\mathrm{nominal}}|^2$，系数 14.4（让关节靠近 nominal pose）
- T11: $-|a_t - a_{t-1}|^2$，action rate penalty，0.12
- T12: $-|a_t - 2a_{t-1} + a_{t-2}|^2$，action smoothness（二阶差分），0.12
- T13: $-|h - h_{\mathrm{nominal}}|^2$，walking height penalty，30.0
- T15: symmetry penalty，0.5（鼓励左右脚对称着地）
- T18: self-collision penalty，1.0

**Performance-based curriculum**：curriculum coefficient 从 0 → 1 线性增加 domain randomization 范围和 penalty 系数。如果 episode 没摔倒且 xy tracking 误差 < 0.4 m/s，coefficient +0.01；否则 -0.01。这让 policy 先在最简单环境学基本步态，再逐步适应更强 randomization。

**Domain randomization（Table 1）**：覆盖 motor strength ±50%, PD gains ±50%, joint position offset ±0.05, friction [0.05, 2.0], added mass ±2 kg, gravity ±2 m/s², random pushes ±1 m/s in xyz, observation noise 等。

### Stage 2: Cross-Embodiment BC Distillation

每个 expert 在 4096 envs 跑 600 steps → 1,985,740,800 samples 总计（约 5 TB h5py 数据）。

Distillation loss：

$$\mathcal{L}_{\mathrm{BC}} = \mathbb{E}_{(s_t, e, a_t) \sim \mathcal{D}} \left[\|\pi(s_t, \phi(e)) - a_t\|^2\right] \tag{5}$$

学生 policy 通过 $\phi(e)$ conditioning，在 817 个训练 embodiment 间共享同一参数。

**训练细节（Table 10）**：
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$)
- Weight decay: $3 \times 10^{-4}$ → 0 (cosine annealing)
- Batch size: 64
- Gradient accumulation: 8 steps（因不同 robot obs/action 维度不同，无法直接拼接，需要 gradient accumulation 平衡 contribution）
- Gradient clipping: max norm 5
- 80 epochs
- 128 GB RAM 维持 in-memory buffer

**Intuition**：两阶段范式让 RL 的探索性 optimization（难，需要 parallel envs 和 careful reward shaping）与大规模 BC 的数据复用性解耦。1000 个 robot 直接 RL 会爆内存且不稳定，但 BC distillation 在单 H100 上一周即可。

参考链接：
- PPO: https://arxiv.org/abs/1707.06347
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Generalist-Specialist learning (Jia et al.): https://proceedings.mlr.press/v162/jia22a.html

---

## 实验结果与 Scaling Curves

### Q1: Embodiment Scaling Laws（Figure 4）

作者在三个层面做 scaling 实验：

**In-class scaling（C1-C3）**：在每个 morphology class 内分别训练/测试
- 训练 embodiment 比例从 0.05 → 1.0
- **每条曲线 reward 几乎翻倍**
- Quadruped 和 Hexapod 在 ~100 个 embodiment 处饱和
- Humanoid 持续提升，没有明显饱和——**更难的 embodiment 从 scaling 中获益更多**

**Cross-class scaling（C4）**：所有 class 合训，混合测试
- Reward 从 18 → 近 30
- 把单 class 模型在混合测试集上评估（C5-C7），表现差
- **C4 最佳点比 C5-C7 高 2-5× reward**

**Data scaling 对比（C8）**：固定 5% embodiment，改变 trajectory 数量
- 0.05 → 0.2（4× data）就几乎饱和
- 之后几乎无收益

**Key finding**：单纯增加 fixed embodiment 上的数据量快速饱和，但增加 embodiment 数量持续有效。**Embodiment scaling 和 data scaling 是不同的轴**。这强烈暗示：要让 robot foundation model 真正泛化到新机器人，必须扩 embodiment 多样性，而不仅是堆同一机器人的数据。

### Q2: Real-World Zero-Shot（Figure 5）

最佳 policy（训练于 817 个 sim embodiment）零样本部署到：

**Unitree Go2**（四足）：
- 草地、鹅卵石、砾石上稳定行走
- Forward/backward locomotion 正常
- 12 种 knee joint limit 配置（20%, 40%, 60% of nominal）
- 通过 software layer + 高增益反推（$K_p=60, K_d=1$）强制限制 knee angle
- Policy 自适应形成 **limping gait**（受限于活动范围的腿保持靠后）

**Unitree H1**（人形）：
- 实验室橡胶地面前进/后退
- 侧向行走比 sim 慢但稳定
- 整体表现略差于 Go2——作者归因于训练集 humanoid 多样性还需提升

**Intuition**：sim-to-real 成功的关键是 URMA 学到了结构化表征，而非记忆特定机器人。新机器人只需提供 URDF → 生成 $d_j$ → 喂给同一 policy。

### Q3: Latent Space Structure（Figure 6）

对 $\bar{z}_{\mathrm{action}}$ 做 t-SNE：
- 三个 morphology class **自然聚类**
- 大子簇按 knee 关节数划分（说明 topology 是显著 factor）
- 子簇内更细分对应 geometric/kinematic 变化

PCA 和 UMAP（Figure 8）也显示 morphology 主导聚类结构，但更 cramped。Joint description latent space（Figure 9）则相对 entangled，说明 joint-level 跨 class 表征更难学。

**Intuition**：policy 自发学到了"语义化"的 embodiment space——humanoids 在一边，quadrupeds 在另一边，hexapods 在第三边。这暗示 policy 内部建了一个 **functional taxonomy of robots**，而不是 flat lookup table。

---

## OOD Generalization 测试（Table 11）

测试 knee joint limits 在训练范围外 {0.6, 0.2} 之外的 {0.1, 0.001}：

| Class | 0.6 | 0.2 | 0.1 | 0.001 |
|---|---|---|---|---|
| Humanoid | 19 | 14 | 16 | 4 |
| Quadruped | 49 | 36 | 45 | 26 |
| Hexapod | 34 | 21 | 28 | 20 |

- 中度 OOD（0.1）：性能下降温和
- 极端 OOD（0.001）：humanoid 暴跌（更不稳定），quadruped/hexapod 相对鲁棒
- Hexapod 因 6 条腿有冗余，鲁棒性最强

---

## 方法学的几个关键设计选择

### 1. 为什么用 BC 而非直接 cross-embodiment RL？
直接在 1000 个 robot 上 RL，需要处理变长 obs/action，PPO 的 GAE 计算复杂，且 batch 内不同 robot 的 reward scale 不同。BC 把 1000 个 expert 蒸馏成单一 student，是计算和稳定性的工程妥协。

### 2. 为什么用 locomotion 作为 testbed？
- Sim-to-real gap 小（perceptual confounders 少）
- 主要依赖 morphology + dynamics
- 避免视觉/渲染干扰，让 scaling 关系更纯净

### 3. 为什么 training/test 按 morphology 分别 80/20 split？
确保每个 class 在测试集中有足够样本评估 in-class 和 cross-class 泛化。

### 4. 为什么 cross-class 比 in-class 强 2-5×？
In-class 模型只见过一种 morphology，遇到其他 morphology 的测试样本几乎失败。Cross-class 模型学习了"机器人"的更抽象概念，能跨 morphology 迁移共享控制原语（如"维持重心"、"避免 self-collision"）。

### 5. 为什么 humanoid 持续受益于 scaling 而 quadruped 饱和？
Humanoid 不稳定，控制空间更复杂，需要更多 embodiment 变体才能覆盖足够多的"动态平衡策略"。这呼应了 LLM scaling law 中 harder task 受益更多的现象。

参考链接：
- Scaling Laws for Neural Language Models: https://arxiv.org/abs/2001.08361
- Chinchilla (compute-optimal): https://arxiv.org/abs/2203.15556
- BridgeData V2: https://proceedings.mlr.press/v229/walke23a.html

---

## 与已有工作的位置

| Work | # Embodiments | Real World | Architecture |
|---|---|---|---|
| GenLoco (Feng 2022) | ~10 quadrupeds | Yes | Fixed obs/action |
| ManyQuadrupeds (Shafiee 2024) | ~10 quadrupeds | Yes | Fixed action abstraction |
| GET-Zero (Patel 2024) | ~100 sim | No | Graph transformer |
| MetaMorph (Gupta 2022) | simplified | No | Transformer |
| URMA (Bohlinger 2024) | 16 | Yes | Attention |
| **GENBOT-1K (this)** | **~1000 sim** | **Yes (Go2, H1)** | **Multi-head URMA** |

这篇工作的独特性：
1. **规模**：从 $10^1$/$10^2$ 跃升到 $10^3$，足以观察 scaling trend
2. **多样性**：跨 3 个 morphology class，覆盖 topology/geometry/kinematics 三维
3. **真实转移**：zero-shot 到 Go2 + H1，包括 kinematic perturbation
4. **可复现**：程序化生成，固定 train/test split（Table 8 列出具体 indices）

---

## 局限与未来方向

1. **任务单一**：仅 flat terrain locomotion。未来扩展到 vision-based manipulation 或 loco-manipulation 是自然方向。
2. **Generation 维度有限**：固定了 mass distribution、damping、actuation type。扩大 generation space 可能进一步提升泛化。
3. **真机测试仅 2 平台**：modular/reconfigurable robot 验证更能体现 embodiment scaling 的价值。
4. **BC 的局限**：gradient accumulation 是工程妥协，zero-padding uniform batching 可能更好。
5. **Humanoid sim-to-real 略差**：暗示需要更多 humanoid 多样性（更多厂商、更广尺寸范围）。

---

## Intuition 总结：这篇 paper 教会我们什么

1. **Embodiment 是一个独立的 scaling 轴**。Data scaling 和 embodiment scaling 不是同一回事；前者饱和快，后者持续受益。
2. **架构 inductive bias 比堆参数更重要**。2.1M 参数的 URMA 通过 attention pooling + per-joint decode 实现了跨 1000 个机器人的泛化，说明把"机器人 = 关节集合"这一先验正确注入架构是关键。
3. **越难的 embodiment 越受益于 scaling**。这和 LLM scaling law 中的现象一致——难任务、长尾任务从规模中获益最多。
4. **程序化生成是 scaling 的 enabler**。手工标注 1000 个机器人 URDF 不现实，但 procedural generation（base unit × variation factor）让 $10^3$ 规模变得可行。
5. **Latent space 学到 functional taxonomy**。Policy 自发按 morphology + topology + geometry 分簇，说明它学到了"机器人是什么"，这暗示未来可以做 morphology co-design——在 latent space 中优化 embodiment 结构。

这篇 work 把 robotics 从 "one robot, one policy" 范式推向 "many robots, one policy" 范式，并提供了第一个 scaling law 的实证。对于 generalist robot foundation model 的方向，这是一个 milestone 式的实验证据。

参考链接：
- π0 (Physical Intelligence): https://arxiv.org/abs/2410.24164
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- GR00T N1: https://arxiv.org/abs/2503.14734
