---
source_pdf: SkyVLNVision-and-LanguageNavigationandNMPCControl for UAVs inUrbanEnvironments.pdf
paper_sha256: 763b7d8affa0b28ea86b7933a9b6a58de4b936b2a1bc61a1c602ab5909ae0970
processed_at: '2026-08-12T07:47:04-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，那我换个角度，用大白话跟你聊聊这篇 paper 到底在干嘛。

---

## 一句话版本

他们想让无人机听懂人话——比如你说"KFC 左边那个麦当劳旁边降落"，无人机自己看风景、自己想路径、自己飞过去，飞的过程中还能躲开移动的车，最后稳稳落到对的地方。

---

## 为什么这件事难

你站在地上导航和在天上导航完全是两码事。

地上走路，你只有前后左右，转弯也就左转右转。无人机在空中，上下左右前后都能动，还能侧着飞不转头，还能 pitch、roll、yaw 三个轴转。自由度一下子从 2D 蹦到 3D，action space 炸了。

城市里高楼一挡，GPS 信号要么没了要么乱跳（multipath effect，信号在楼之间弹来弹去），所以你得靠眼睛——摄像头——来判断自己在哪、高度多少、前面有没有楼。

再加上无人机飞得远，路上有车有人有鸟，天气光照还在变，这就比室内导航难了一个数量级。

以前做 VLN（vision-and-language navigation）的人基本都在室内搞，比如"穿过客厅进厨房拿个东西"。 outdoor 的有 TouchDown，但那是地面上走路的。天上飞的、3D 连续空间的、带动态避障的，基本没人做过。

---

## 他们的系统长啥样

整体就是一个 pipeline，分两层：

**上层是"大脑"**：用 GPT-4 这种 LLM 听指令、看图、想下一步往哪飞。

**下层是"小脑"**：用 NMPC（非线性模型预测控制）把"往哪飞"翻译成具体的电机推力和姿态角，让无人机真的飞过去，飞的过程中还能躲障碍。

这两层之间有个桥梁，就是 WPO（Wayfinding Prompt Optimization），负责把感知到的东西变成 LLM 能理解的文字描述，同时帮 LLM 记住自己走过哪。

---

## 大脑部分怎么工作的

### 第一步：看

无人机前面装了个广角摄像头，只能上下转不能左右转。所以它要看全景就得自己转身子。每转一个角度拍一张，拿到 RGB、depth、semantic 三种图。

然后用 GroundingDINO 这个 open-set detection 模型去框出图里的 landmark——KFC、麦当劳、白色大楼、什么路口之类的。如果同一个 landmark 在好几个视角都出现了，就挑 score 最高的那个视角作为"观察这个 landmark 的最佳角度"。

### 第二步：想

LLM 拿到自然语言指令后，先把指令拆成 sub-goal。比如"飞过 KFC，然后到麦当劳旁边的路口降落"会被拆成 [KFC, 麦当劳, 路口] 三个 landmark。

这一步的好处是 agent 可以一步一步来，走错了还能回退到某个 landmark 重新规划，而不是一条路走到黑。

---

## 两个核心创新

### 创新一：HSD（High-resolution Spatial Descriptor）

问题在哪呢——你跟 LLM 说"前面有条路"，LLM 不知道路是在正前方还是偏左偏右。正前方就直飞，偏左就得斜着飞，action 完全不同。

他们的做法特别朴素：把画面切成 3×3 的九宫格，每格编个号。然后描述变成"路在 #4 格"（正中间，直飞）或者"路在 #2 格"（中右，往右偏）。

就这么一个简单的 grid verbalizer，让 LLM 的空间判断能力直接上了一个台阶。

整个流程是：拍图 → GroundingDINO 框 landmark → 每个候选位置生成文字描述 → LLM 比较这些描述跟原始指令的匹配度 → 排序选最佳。

这个"先两两比较再综合推理"的范式叫 comparing-then-reasoning，模仿人类看图——你不是一眼扫完所有图就下结论，而是先看这张和那张差在哪，再综合判断。

### 创新二：TBMA（TrackBack Memory Array）

这个解决的是另一种问题——指令里压根没 landmark。

比如"左转，然后右转，然后直走"——纯方向指令，没有 KFC 没有麦当劳。LLM 没有参照物，一旦走错了就懵了，只能继续瞎转。

他们的做法是把历史轨迹存成一个 graph：

- 每个 node = 之前见过的 landmark
- 每条 edge = 两个 landmark 之间的导航指令
- 任意两个 node 之间可以用最短路径算法找路

这样 agent 走到死胡同了，可以查 graph 发现"哦我之前在 KFC 那儿，我可以回到 KFC 再换条路走"。而且 LLM 还可以主动问 commander 要澄清——"你说的左转是在 KFC 之后还是之前？"

HSD 和 TBMA 是配合用的：HSD 发现当前 landmark 描述太模糊（比如"白色大楼"城市里一堆），就触发 TBMA 去查历史；TBMA 给了历史 context 后，HSD 重新聚焦相关区域再提取特征。

---

## 小脑部分：NMPC

LLM 说"往那个方向飞 30 米"——这是一个 macro-action。但无人机要真飞过去，得算推力多少、机身倾斜多少度、怎么躲前面那辆车。

他们用 NMPC 来干这个事。

### 动力学模型

核心就几行：

- 位置的变化率 = 速度（废话）
- 速度的变化率 = 推力（在 body frame 里是竖直向上的）经过旋转矩阵投到 world frame + 重力 - 空气阻力
- 姿态角的变化 = 一阶滞后系统（假设底下有个 PID attitude controller 在管电机）

为什么这么简化？因为 quadrotor 没有机翼，水平加速只能靠**倾斜机身**——你往前飞就得 pitch forward，这时候推力的水平分量就推你往前走。空气阻力简单建模成三个轴独立的线性阻尼。

姿态部分不自己解 fast dynamics，而是假设有个低层 PID 在管电机，自己只管"上层"的轨迹跟踪。这样 MPC 的预测步长可以大一点，求解快一点。

### Cost function

三块：

1. **State cost**：离参考轨迹越远罚越多
2. **Input cost**：推力别太大（省电、保护电机）
3. **Input smoothness cost**：控制量别突变（避免暴力机动，乘客/货物舒服，电机也耐用）

每一步求解未来 N 步的优化问题，但只执行第一步，下一步重新测量重新求解。这就是 receding horizon control 的精髓——靠 feedback 来对抗模型不准和外部干扰。

### 障碍物约束

用球体模型表示障碍物：障碍物有个真实半径，外面再套一层 safety radius。无人机不能进入这个"膨胀球"内部。

动态障碍物就是每个预测时间步都加一个这样的约束，把障碍物的整条预测轨迹都塞进求解器。

### 输入变化率约束

roll/pitch 的参考值每个 time step 最多变 $\Delta\phi_{\max}$ / $\Delta\theta_{\max}$——避免避障时突然猛打方向，attitude controller 跟不上。

### 求解器

用 PANOC + OpEn。PANOC 是专门给 embedded MPC 设计的 Newton-type 算法，内存小算得快，能满足无人机 control loop 的实时性要求。

---

## 实验环境

AirSim + Unreal Engine 4 搭的城市。楼顶和外墙都加了很多细节——因为无人机从上往下看，看到最多的是屋顶，如果屋顶全是灰秃秃的方块根本没法做 landmark 识别。

动态元素有车有人，来自 Mirage framework。

Action space 是连续的，可以飞到环境里任意一个点。这跟 R2R、RxR 那种在 nav-graph 上跳 node 的设定完全不同——SkyVLN 是真 3D 连续飞行。

数据集用 AVDN，平均路径 287 米，对话式指令。

---

## 结果怎么样

### 主实验

Full model（HSD + TBMA + NMPC 全开）在 unseen test 上：

- **SPL 28.11%**（Success weighted by Path Length，越高越好）
- **SR 42.37%**（Success Rate）

对比 NavGPT + PID baseline 的 18.9% / 16.6%，提升非常明显。

关键发现：**提升主要来自 unseen 环境**。seen validation 上 Full model 和 w MPC 差不多，但 unseen test 上 Full model 碾压。这说明 HSD + TBMA 的组合主要在**泛化**上发力——在没见过的环境里，靠 fine-grained spatial description 和 memory backtracking 来弥补 lack of training prior。

### 控制器对比

NMPC vs AirSim 默认的 SimpleFlight（PID）：

- 位置跟踪：NMPC 紧贴参考轨迹，PID 抖得厉害，尤其 Z 轴
- 姿态误差：NMPC 快速收敛到 0，PID 持续震荡
- 加了动态障碍物后 NMPC 依然稳

这说明非线性动力学建模在动态场景下确实比线性 PID 强。

### LLM Ablation

- **GPT-4V**：SR 最高 34.9%，NE（Navigation Error）最低 62.35m——原生多模态，直接吃图
- **GPT-4o**：SPL 最高 34.25% 但 SR 只有 20.44%——成功的 case 路径很高效，但失败率高，整体不稳
- **GPT-4 Turbo**：SR 最低 15.62%，NE 最大 127.87m——纯文本模型，要靠 GroundingDINO 把图转成文字再喂，信息损失大

intuition 很直白：**直接看图比先转文字再看强太多了**。多模态原生模型在 spatial reasoning 上有结构性优势。

---

## 我觉得这篇 paper 的真正贡献

不是某个模块多 novel——HSD 的九宫格很朴素，TBMA 的 graph memory 也是标准操作，NMPC for quadrotor 更是十年前的老技术。

真正的贡献是**把整条链路打通了**：

perception（GroundingDINO）→ spatial reasoning（HSD）→ memory（TBMA）→ macro action（LLM）→ trajectory optimization（NMPC）→ motor command

很多 VLN paper 停在"输出 go forward 5m"就完了，control 当黑盒。很多 MPC paper 完全不管 high-level task。SkyVLN 把这两层真正缝在一起了，还在一个相对真实的城市 sim 里跑通了。

---

## 我觉得的几个槽点

**LLM 延迟问题完全没提**。GPT-4 一次调用几百毫秒到秒级，NMPC 要 100+ Hz。这俩时间尺度差三个数量级，怎么协调的？macro-action 执行多久？paper 没说。

**HSD 的九宫格是固定分辨率**。远处 landmark 的角分辨率应该比近处细，固定 grid 在远处可能不够用，近处又浪费。换成 polar grid 或者 learnable spatial tokens 会更合理。

**TBMA 的 graph 膨胀问题**。287 米的路径，如果每走几米就加个 node，graph 会不会撑爆 LLM context window？paper 没讨论 memory management。

**障碍物轨迹是 ground truth**。AirSim 里直接拿到障碍物的精确状态和未来轨迹，real world 里你得自己预测。这个 sim2real gap 挺大的。

**Safety 没兜底**。NMPC 有 safety radius 约束，但 LLM 给的 macro-action 本身可能不安全（指向禁飞区、或者指令本身矛盾）。paper 没讨论这个 level 的 safety。

---

## 一句话总结

这篇 paper 就是把 LLM 时代的 VLN 思想第一次严肃地搬到了天上——无人机听人话、看风景、想路径、躲障碍、稳稳飞。每个模块单独看都不算特别新，但缝在一起跑通了一个完整闭环，还在城市级 sim 里验证了泛化性。后续如果做 sim2real、多机协同、或者把 HSD 换成 learnable spatial representation，都是很自然的 follow-up。

---

# SkyVLN 深度讲解：让 UAV 在城市里听懂人话并安全飞行

## 一、Paper 的核心 motivation：为什么这是一个值得做的问题

SkyVLN 这篇工作要解决一个非常具体的问题：在**复杂城市 3D 环境**里，让 UAV 能够基于**自然语言指令** + **第一人称视觉感知**完成导航任务，并且底层用 **NMPC** 做精确的轨迹跟踪与动态避障。这其实是把三件事缝在一起：(1) LLM/VLM 的多模态推理能力；(2) UAV 在 3D 连续空间的飞行控制；(3) 城市级 GNSS-denied 场景下的视觉定位。

Paper 一开篇就强调了 ground VLN 和 aerial VLN 的根本差异，我觉得这是 build intuition 的关键点：

1. **Action space 从 2D 升到 3D**：地面 robot 一般只有 forward/turn，而 multirotor 要处理 rise up、pan down，还能侧移而不转头，再加上 roll/pitch/yaw，自由度暴涨。
2. **GNSS 在城市峡谷里失效**：高楼密集 + 电磁干扰 → multipath effect，所以必须靠 visual localization 补 depth/pose/height。
3. **长距离飞行 + 动态障碍物**：天气、光照、移动车辆都会让 navigation 难度远高于 indoor VLN。

这三条决定了 SkyVLN 不能简单地把 R2R / RxR 那套 nav-graph 范式搬到天上，必须重新设计 perception → reasoning → control 的全链路。

参考相关背景：
- R2R: https://arxiv.org/abs/1711.07280
- RxR: https://arxiv.org/abs/1907.04975
- Touchdown (outdoor ground VLN): https://arxiv.org/abs/1810.05330
- AVDN dataset: https://aclanthology.org/2023.findings-acl.190

---

## 二、整体架构解析（对应 Figure 2）

SkyVLN 的系统由两大模块组成：

### 2.1 Multimodal Perception Module

UAV 前部装一个**广角相机**，可以沿 pitch 轴上下转 90°，但**不能左右转**。这一点很关键——意味着 UAV 必须靠自身旋转才能拿到 panoramic view，这其实模拟了真实硬件约束（很多商用 quadrotor 的 camera gimbal 就是只有 1-axis 或者 2-axis）。

视觉输入分三路：
$$V_t = \{v_t^R, v_t^D, v_t^S\}$$

- $v_t^R$：RGB image
- $v_t^D$：Depth image
- $v_t^S$：Semantic segmentation image

注意这里 paper 特意 follow 了主流 robotic navigation 的设定，**限制 agent 只能看前向**，需要主动 rotate 来获取其他视角，这就避免了直接喂 panoramic 的作弊式设置。

然后用 **GroundingDINO** 做粗粒度 landmark 检测，如果同一个 landmark 在多个视角出现，就选 score 最高的那个视角作为该 landmark 的 observation viewpoint。

GroundingDINO 论文：https://arxiv.org/abs/2303.05499  
代码：https://github.com/IDEA-Research/GroundingDINO

### 2.2 Sub-goal Extraction

公式 (1)：
$$L = \mathrm{LLM}(T, \, \mathrm{prompt})$$

- $T$：natural language instruction
- $\mathrm{prompt}$：引导 LLM 抽取 landmark 的提示模板
- $L = \{l_1, l_2, \ldots, l_n\}$：抽出来的 landmark 集合

这里 build intuition 的关键是：把一个长 instruction 分解成 sub-goals，每个 sub-goal 对应一个 landmark，这样 agent 可以**逐步推理 + 必要时 backtrack**（对应 Figure 3 的 graph 结构）。这个思想类似 NavGPT 的 explicit reasoning，但在 aerial 场景下多了 graph 化的 memory。

NavGPT: https://arxiv.org/abs/2305.86970

---

## 三、Wayfinding Prompt Optimization (WPO)：让 LLM 看得更细、记得更远

WPO 是这篇 paper 的核心创新之一，分成两个子模块。

### 3.1 High-resolution Spatial Descriptor (HSD)

问题：如果只告诉 LLM "road 在 forward view"，LLM 没法判断 road 是在正前方还是偏左——这两种情况对应的 action 完全不同（一个是直走，一个是斜飞）。

HSD 的做法：把每个视角的画面切成 **3×3 = 9 个 sector**，每个 sector 给一个标签（比如 #0# 是左上，#4# 是中心，#8# 是右下，对应 Figure 4）。

这样描述就变成 "road 在 sector #4#"（中心，应该直行）vs "road 在 sector #2#"（中右，应该向右偏）。这种 grid-based verbalizer 在 VLM grounding 文献里是常用 trick，比如 CLIPort、VIMA 系列都用过类似思路。

具体 pipeline（对应 Figure 4）：

1. 从 observation image 提 visual feature $O$
2. 从 landmark text 提 word feature $B$
3. 对查询 landmark $L_n$，找出 top-K 候选匹配
4. LLM 为每个 (query, candidate) pair 生成一段描述文本
5. 综合评估这些描述文本与原始 query 的相似度，得到 ranking

这是一个 **comparing-then-reasoning** 的范式，paper 强调这是模仿人类做视觉比较时的迭代过程——人类不是一步看完所有图就下结论，而是先做 pairwise delta 描述，再做整体推理。这跟最近 NaVid、EVolNav 等 video VLN 工作里强调的 temporal reasoning 思路一致。

### 3.2 TrackBack Memory Array (TBMA)

这是处理 **ambiguous instruction** 的关键模块。

考虑这种情况：instruction 是 "turn left, then move right, then go straight"——**完全没有 landmark**，LLM 没有任何参照物。如果 agent 走错了，它只能继续盲目探索（对应 Figure 3 左侧的失败 case）。

TBMA 的设计：把历史轨迹和 instruction 存成一个 **graph**：

- **Node** = 历史上遇到过的 landmark
- **Edge** = 两个 landmark 之间的导航指令
- 任意两个 node 之间可以用 shortest path algorithm 找到 navigable path

这样当 agent 意识到自己走错了，可以**回溯到某个历史 node**，再重新规划。对应 Figure 3 右侧的 thinking process——LLM agent 在有 memory graph 时可以做 global exploration 和 path planning，而不是局部贪心。

Paper 还提到一个 formatted prompt 让 LLM 可以**主动向 commander 提澄清问题**（对应 Figure 9），这是 active learning / interactive VLN 的味道，跟 TEACh、HRVLN 这类 dialog-based VLN 工作思路相通。

TEACh: https://arxiv.org/abs/2110.00511

HSD 和 TBMA 是**互补**的：HSD 检测到 ambiguous landmark（比如 "white building" 太模糊）时，触发 TBMA 查询历史 reference；TBMA 给出 temporal context 后，HSD 重新聚焦相关 region 再做特征提取。

---

## 四、NMPC：把 LLM 的高级决策落到飞行动力学上

LLM 给出的是 macro-action（target position 之类），但 UAV 实际飞行要处理 thrust、attitude、dynamic obstacle。Paper 用 **Nonlinear Model Predictive Control** 来 fill 这个 gap。

### 4.1 状态和输入

UAV 状态向量：
$$x = [p, v, \phi, \theta]^T$$

- $p = [x, y, z]^T$：position in world frame
- $v = [u, v, w]^T$：linear velocity in global frame（注意这里的 $u, v, w$ 是速度分量，跟控制输入 $u$ 重名了，是 paper 的 notation 混乱，阅读时要小心）
- $\phi$：roll angle（绕 $x^W$ 轴）
- $\theta$：pitch angle（绕 $y^W$ 轴）

控制输入：
$$\boldsymbol{u} = [T, \phi_{\mathrm{ref}}, \theta_{\mathrm{ref}}]^T$$

- $T \geq 0$：total thrust（四个电机推力之和）
- $\phi_{\mathrm{ref}}$：reference roll
- $\theta_{\mathrm{ref}}$：reference pitch

注意 yaw $\psi$ 没有出现在状态向量里——paper 假设了一个 simplified 4-state model，可能是因为 aerial VLN 任务里 yaw 控制相对独立，或者由低层 controller 处理。这是常见 simplification，在 MPC for quadrotor 文献里比如 Kamel et al. 的 RotorS MPC 也是类似处理。

参考：Kamel et al. MPC for UAV: https://arxiv.org/abs/1609.06753

### 4.2 6-DoF 动力学模型（公式 2）

$$\dot{p}(t) = v(t)$$

简单：位置导数 = 速度。

$$\dot{v}(t) = R(\phi, \theta) \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \\ -g \end{bmatrix} - A\, v(t)$$

- $R(\phi, \theta) \in \mathrm{SO}(3)$：从 body frame 到 world frame 的旋转矩阵（Euler 形式）
- $[0, 0, T]^T$：body frame 下的推力向量（沿 body z 轴向上），旋转到 world frame 后产生水平和垂直分量
- $[0, 0, -g]^T$：重力
- $A = \mathrm{diag}(A_x, A_y, A_z)$：线性气动阻尼矩阵，三个轴独立

build intuition 的方式：quadrotor 没有气动翼面，所以水平加速靠**倾斜机身**——倾斜 $\theta$ 会让 thrust 在 world x 方向产生分量 $T \sin\theta$，倾斜 $\phi$ 产生 y 方向分量。这就是为什么 quadrotor 必须 pitch forward 才能向前飞。

姿态动力学建模成**一阶系统**：

$$\dot{\phi}(t) = \frac{1}{\tau_\phi}\left(K_\phi \phi_{\mathrm{ref}}(t) - \phi(t)\right)$$

$$\dot{\theta}(t) = \frac{1}{\tau_\phi}\left(K_\theta \theta_{\mathrm{ref}}(t) - \theta(t)\right)$$

- $K_\phi, K_\theta$：闭环增益
- $\tau_\phi, \tau_\theta$：时间常数

这个建模假设了 UAV 有一个 **lower-level attitude controller**（比如 PX4/ArduPilot 标配的 PID cascaded controller），它把 thrust/roll/pitch 命令翻译成电机 PWM。MPC 只管"上层"轨迹，把 attitude loop 当作一阶滞后环节。这是 embedded MPC 的标准做法，避免在 MPC 里解 fast attitude dynamics，可以让预测步长更大、求解更快。

Paper 明确提到 AirSim 的 rotor aerodynamic model "相对粗糙"，移植到 real aircraft 会有问题——所以他们用了这个 simplified model 来定义 NMPC，而不直接信任 sim2real 的 motor model。

### 4.3 Cost Function（公式 3）

$$J(\boldsymbol{x}_k, \boldsymbol{u}_k, \boldsymbol{u}_{k-1|k}) = \sum_{j=0}^{N} \underbrace{\|\boldsymbol{x}_{\mathrm{ref}} - \boldsymbol{x}_{k+j|k}\|^2_{Q_x}}_{\text{State cost}} + \underbrace{\|\boldsymbol{u}_{\mathrm{ref}} - \boldsymbol{u}_{k+j|k}\|^2_{Q_u}}_{\text{Input cost}} + \underbrace{\|\boldsymbol{u}_{k+j|k} - \boldsymbol{u}_{k+j-1|k}\|^2_{Q_{\Delta u}}}_{\text{Input smoothness cost}}$$

逐项解释：

- $k$：当前 time step
- $j$：prediction horizon 内的相对索引（0 到 $N$）
- $\boldsymbol{x}_k$：当前状态
- $\boldsymbol{u}_k$：当前控制输入
- $\boldsymbol{u}_{k-1|k}$：上一步的控制输入（用于 smoothness 惩罚）
- $\boldsymbol{x}_{\mathrm{ref}}$：参考状态（来自 LLM 给的 macro-action 经过映射后得到）
- $\boldsymbol{u}_{\mathrm{ref}}$：参考控制输入（一般是 hover 时的推力 $T = mg$，roll/pitch = 0）
- $\boldsymbol{x}_{k+j|k}$：在时刻 $k$ 预测的 $k+j$ 时刻状态
- $\boldsymbol{u}_{k+j|k}$：在时刻 $k$ 决定的 $k+j$ 时刻控制输入
- $Q_x, Q_u, Q_{\Delta u}$：正定权重矩阵（penalize state deviation、input magnitude、input rate）
- $N$：prediction horizon（paper 里用 $T_p$ 表示预测时域，$T_c$ 表示控制时域，一般 $T_c \leq T_p$）

**三项 cost 的 intuition**：
- State cost：让 UAV 跟踪 LLM 给的参考轨迹
- Input cost：不要使蛮力（节能 + 保护电机）
- Input smoothness cost：避免突变控制（避免 aggressive maneuver，提高乘客/货物舒适度，也减小电机损耗）

### 4.4 球形障碍物约束（公式 4）

$$h_{\mathrm{sphere}}(p, \xi_{\mathrm{obs}}) = \big[(r_{\mathrm{obs}} + r_s)^2 - (x - x_{\mathrm{obs}})^2 - (y - y_{\mathrm{obs}})^2 - (z - z_{\mathrm{obs}})^2\big]_+ = 0$$

- $\xi_{\mathrm{obs}} = [r_{\mathrm{obs}}, r_s, p_{\mathrm{obs}}]$：障碍物参数集
- $r_{\mathrm{obs}}$：障碍物实际半径
- $r_s$：附加 safety radius（缓冲带）
- $p_{\mathrm{obs}} = [x_{\mathrm{obs}}, y_{\mathrm{obs}}, z_{\mathrm{obs}}]$：障碍物中心 world frame 坐标
- $[h]^+ = \max\{0, h\}$：positive part operator

intuition：括号里的表达式在 UAV 进入 $(r_{\mathrm{obs}} + r_s)$ 球内时为正，外部为负。$[\cdot]^+$ 把外部映射到 0，内部保持正值。约束 $h = 0$ 等价于 "UAV 不能进入球内"。

对 dynamic obstacle，每个预测 time step 都加一个这样的约束，于是障碍物的整个预测轨迹 $\{p_{\mathrm{obs}}(k), k=0,\ldots,N\}$ 都被 parametrized 进 NMPC 求解器。

### 4.5 输入变化率约束（公式 5）

$$|\phi_{\mathrm{ref}, k+j-1|k} - \phi_{\mathrm{ref}, k+j|k}| \leq \Delta\phi_{\max}$$

$$|\theta_{\mathrm{ref}, k+j-1|k} - \theta_{\mathrm{ref}, k+j|k}| \leq \Delta\theta_{\max}$$

- $\Delta\phi_{\max}, \Delta\theta_{\max}$：每个 time step 内 roll/pitch 参考的最大变化量

这是为了让控制输入在避障时**不会突然跳变**——即使要急转弯，也要在物理可达范围内平滑过渡，避免 attitude controller 跟不上。

### 4.6 求解：PANOC + OpEn

公式 (6) 的标准形式：

$$\min_{z \in Z} f(z, \rho) \quad \text{s.t.} \quad F(z, \rho) = 0$$

- $f$：Lipschitz-differentiable cost
- $F$：vector-valued equality constraint mapping
- $z$：decision variable（all control inputs over horizon）
- $\rho$：参数向量（initial state、references、obstacle trajectory）

**PANOC** (Proximal Averaged Newton for Optimal Control) 是一种 Newton-type method，专门为 embedded MPC 设计——低内存、低计算负载、低 latency。这对 UAV 是关键，因为 control loop 一般要跑 50–200 Hz。

equality constraint 用 quadratic penalty method 处理，逐步把 cost minima 推到约束满足的区域。

OpEn (Optimization Engine) 项目主页：https://alphaville.github.io/optimization-engine/  
PANOC 原始论文：https://arxiv.org/abs/1902.01986

Algorithm 1 整体流程归纳：

```
loop:
  measure x_UAV(t), x_obstacles(t)
  predict future states over horizon T_p
  formulate NLP (cost + dynamics + obstacle + input constraints)
  solve with PANOC -> u*
  apply u*(0)  // 只执行第一步
  t = t + Δt
  if reached destination or emergency: break
```

这就是经典的 **receding horizon control**——每一步求解一个未来 $T_p$ 步的优化问题，但只执行第一步，下一步重新测量、重新求解。这种 feedback 机制让 MPC 对 model mismatch 和 disturbance 有天然鲁棒性。

---

## 五、3D 实验平台：AirSim + UE4

Paper 用 **AirSim**（Microsoft 的 drone simulator）+ **Unreal Engine 4** 搭建城市环境。关键特性：

- Buildings：shopping malls、residential complexes、public facilities
- Streets：lanes、intersections、traffic signals、road markings
- 动态元素：vehicles、pedestrians（来自 Mirage framework，引用 [29]）
- Street furniture、vegetation、urban amenities
- 在高楼顶部和外墙上加了大量细节，方便 aerial agent 识别（这个细节很重要，因为 aerial view 看到最多的是 roof）

Action space 是连续的：target position $[x, y, z]^T$、target velocity $[u, v, w]^T$、target orientation $[\theta, \phi, \psi]^T$，可以飞到 environment 内任意点。这跟 Touchdown、R2R、RxR 这种 graph-based discrete action 完全不同——SkyVLN 是真正 continuous 3D 空间。

AirSim 仓库：https://github.com/microsoft/AirSim  
AirSim 论文：https://arxiv.org/abs/1705.05065

---

## 六、实验结果解析

### 6.1 数据集与硬件

- 数据集：**AVDN** (Aerial Vision-and-Dialog Navigation)，引用 [34]，平均路径长度 **287m**
- LLM：GPT-4（ablation 还测了 GPT-4V、GPT-4o、GPT-4 Turbo）
- 硬件：Intel i9 12代 + RTX 4070
- Depth sensor：100m 感知距离
- Camera FOV：90°

AVDN paper: https://aclanthology.org/2023.findings-acl.190

### 6.2 Table I：主实验

| Model | Seen Val SPL/SR | Unseen Val SPL/SR | Unseen Test SPL/SR |
|---|---|---|---|
| Random | 0.5 / 1.6 | 0.2 / 1.0 | 0.5 / 1.1 |
| CMA | 8.2 / 10.5 | 12.8 / 6.7 | 8.3 / 9.7 |
| Seq2Seq | 5.1 / 6.4 | 7.9 / 4.2 | 5.6 / 5.1 |
| NavGPT w PID | 12.3 / 14.6 | 15.2 / 10.8 | 18.9 / 16.6 |
| w MPC | 14.34 / 17.3 | 16.5 / 20.4 | 12.9 / 15.17 |
| w NMPC | 13.9 / 16.2 | 17.1 / 18.5 | 22.4 / 26.8 |
| w/o HSD | 11.6 / 13.0 | 18.3 / 20.0 | 12.6 / 14.1 |
| w/o TMA | 11.57 / 12.97 | 18.3 / 19.95 | 18.17 / 15.32 |
| **Ours Full** | 14.7 / 17.3 | 16.62 / 20.44 | **28.11 / 42.37** |

几个关键观察：

1. **Unseen Test 上 Full model 大幅领先**：SPL 28.11% vs NavGPT 18.9%，SR 42.37% vs NavGPT 16.6%——这意味着 HSD + TBMA 的组合主要在**泛化**上发力，seen set 提升没那么明显。这跟 paper 强调 "robustness in new environments" 一致。

2. **w MPC vs w NMPC**：MPC 在 unseen test 上反而比 NMPC 差（SPL 12.9 vs 22.4），说明非线性动力学建模在动态障碍物场景下确实有优势。linear MPC 在 attitude 较大时会因为小角度假设失效。

3. **w/o HSD 在 unseen test 上 SR 掉到 14.1%**：HSD 的 spatial verbalizer 对 unseen 环境特别重要，因为没有训练分布的 prior，必须靠 fine-grained spatial description 来 disambiguate。

4. **w/o TMA 的 SR 在 unseen test 15.32%**：TBA 的 backtracking 能力对长指令序列至关重要——AVDN 平均 287m 的路径，agent 走错一步如果回不去就废了。

5. **NavGPT w PID（应该是 paper 里的 "NavGPT [6] w PID" 行）在 seen val 上 SR 14.6，unseen test SR 16.6**——纯 LLM + PID 的 baseline 在 unseen 上其实没掉太多，但 SR 整体偏低，因为没有 spatial reasoning 和 memory。

### 6.3 Table II：LLM Ablation

| LLM | Unseen Test SPL | SR | NE (m) |
|---|---|---|---|
| GPT-4V | 16.62 | 34.9 | 62.35 |
| GPT-4o | 34.25 | 20.44 | 90.11 |
| GPT-4 Turbo | 25.12 | 15.62 | 127.87 |

这里有一个有趣的现象：

- **GPT-4V**：SR 最高 (34.9%)，NE 最低 (62.35m)——成功率最高，失败时也离目标最近。这是**真正 multimodal** 的模型，能直接吃 image。
- **GPT-4o**：SPL 最高 (34.25%)，但 SR 反而只有 20.44%——这是个看似矛盾的数据。可能的解释：GPT-4o 成功的 case 路径非常高效（高 SPL），但失败率高，整体稳定性不如 GPT-4V。NE 90.11m 也偏高。
- **GPT-4 Turbo**：SR 最低 (15.62%)，NE 最大 (127.87m)——这是纯 text 模型，要靠 GroundingDINO 把 image 转成 text description 再喂进去，信息损失大，失败时偏离最严重。

build intuition：**直接 visual grounding 比间接 text-mediated grounding 显著好**。这跟 NaVid、NaviLLM 这些 video-LLM 导航工作的发现一致——多模态原生模型在 spatial reasoning 上有结构性优势。

NaVid: https://arxiv.org/abs/2402.15852

### 6.4 Figure 7/8：6-DOF 轨迹曲线

Paper 对比了 AirSim 默认的 **SimpleFlight**（PID 控制）和 **NMPC**：

- **Position curve**：NMPC（黑线）紧贴 reference（红线），SimpleFlight 抖动明显，Z 轴尤其乱。
- **Attitude curve (roll/pitch/yaw)**：NMPC 误差小且快速收敛，SimpleFlight 有持续震荡。
- **Position error**：NMPC 在 X、Y 轴误差远低于 SimpleFlight；Yaw error 几乎衰减到 0，SimpleFlight 持续偏高。

Figure 8 加了 dynamic obstacle，NMPC 依然稳定——这说明 NMPC 的 obstacle constraint 真的在工作，不是单纯靠 PID 跟踪。

---

## 七、整体 intuition 总结与一些可能的延伸

1. **LLM 负责"想"，NMPC 负责"飞"**：这是 hierarchical robotics 控制的典型范式（类似 SayCan、Code as Policies、VoxPoser 的思路），但 SkyVLN 把它放到 aerial + 3D continuous space，且加入了 HSD/TBMA 这种 spatial memory 机制。

SayCan: https://arxiv.org/abs/2204.01691  
Code as Policies: https://arxiv.org/abs/2209.07753  
VoxPoser: https://arxiv.org/abs/2307.05973

2. **HSD 的 3×3 grid 思想可以扩展**：paper 用 9 个 sector，但完全可以用 polar grid、log-polar、或者 learnable spatial tokens（类似 DETR 的 positional embedding）。这是一个明显的改进方向。

3. **TBMA 的 graph memory 跟 topological SLIDE memory 思路相通**：可以联想到 NeRF-based memory、Cartographer subgraph、或者更近的 MemGPT 风格的 hierarchical memory。如果 TBMA 能结合 visual feature embedding（CLIP feature）而不只存 text label，泛化会更好。

MemGPT: https://arxiv.org/abs/2310.08560

4. **NMPC 的 sim2real gap**：paper 用 simplified first-order attitude model，real quadrotor 的 motor delay、blade flapping、ground effect 都没建模。移植到 real hardware 时可能需要 system identification + robust MPC（tube MPC）或者 learning-based residual policy。

5. **AVDN 287m 平均路径意味着 instruction complexity 很高**：这意味着 LLM 的 long-context reasoning 能力是 bottleneck 之一。可以考虑用 retrieval-augmented planning，把长 instruction 切片成短 sub-instruction，每次只 reason 当前一段。

6. **跟 EmbodiedCity（引用 [26]）的关系**：EmbodiedCity 是同一团队的城市仿真平台，SkyVLN 应该是基于它做的扩展，专门处理 aerial perspective。可以期待后续 multi-agent swarm version。

EmbodiedCity: https://arxiv.org/abs/2406.09941 (or similar)

7. **跟 AerialVLN / OpenUAV 等并发工作的对比**：aerial VLN 这个方向 2024 年集中爆发，比如 OpenUAV、CityNav、AVDN 系列都在抢 benchmark。SkyVLN 的差异化卖点是 **NMPC + LLM 的紧耦合**，而不只是 navigation policy 本身。

8. **安全性问题**：NMPC 的 safety radius $r_s$ 是 hard constraint，但 LLM 给的 macro-action 可能本身就不安全（比如指向禁飞区）。Paper 没讨论 this level of safety，未来的工作可能要加 formal verification 或者 shielded RL 来兜底。

---

## 八、一些值得追的 reference

- GroundingDINO (open-set detection): https://arxiv.org/abs/2303.05499
- NavGPT (explicit reasoning VLN): https://arxiv.org/abs/2305.86970
- NavGPT-2 (LVLM-based VLN): https://arxiv.org/abs/2402.07492
- AVDN (aerial dialog navigation): https://aclanthology.org/2023.findings-acl.190
- AirSim: https://arxiv.org/abs/1705.05065
- OpEn (Optimization Engine): https://alphaville.github.io/optimization-engine/
- PANOC: https://arxiv.org/abs/1902.01986
- MPC for UAV (Kamel et al.): https://arxiv.org/abs/1711.07801
- Touchdown (outdoor VLN): https://arxiv.org/abs/1810.05330
- R2R (original VLN): https://arxiv.org/abs/1711.07280
- SayCan: https://arxiv.org/abs/2204.01691
- VoxPoser: https://arxiv.org/abs/2307.05973

---

## 九、最后的 critical comment

这篇 paper 的核心 contribution 我觉得是**把 VLN 的高层 reasoning 和 MPC 的低层 control 真正打通了**——很多 VLN paper 只停在 action 输出（"go forward 5m"），把 control 当黑盒；很多 MPC paper 又完全不管 high-level task。SkyVLN 用 HSD + TBMA + NMPC 把 perception → spatial reasoning → memory → trajectory optimization → motor command 整个闭环做完了，并且在一个相对真实的城市 sim 里验证。

但它也有几个我没看到的细节：

1. **LLM 推理延迟**：GPT-4 一次调用几百 ms 到秒级，NMPC 要 100+ Hz，这两个 time scale 怎么协调？Paper 没明确说 macro-action 的执行时长。
2. **HSD 的 9 个 sector 是固定的**：在城市环境里，远处 landmark 的 angular resolution 应该比近处的细，固定 grid 在远处可能不够用。
3. **TBMA 的 graph size 上限**：长 episode 下 graph 会不会膨胀到 LLM context window 不够？Paper 没讨论。
4. **dynamic obstacle 的轨迹预测**：NMPC 里 obstacle trajectory 是给定参数，但 real world 里 obstacle 是要预测的（用 Kalman filter / trajectory predictor）。Paper 假设 AirSim 里直接拿到 ground truth obstacle state，这是 sim-only 的 luxury。

整体来说这是一个 engineering 完整度很高的工作，把 LLM 时代的 VLN 思想第一次严肃地放到了 aerial + NMPC 的 setting 里。后续如果做 sim2real、multi-UAV、或者把 HSD 换成 learnable spatial tokens，都会是很自然的 follow-up。
