---
source_pdf: HARMON.pdf
paper_sha256: ed3cc5edce021cd9ab707aff55877c4524c55a60b2a49570c114e5d7d77d7b37
processed_at: '2026-08-04T23:29:39-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy，用最直白的人话讲，HARMON 做的事情类似于：**“让 AI 先根据剧本找一个真人演员演一遍，把动作映射到机器人身上，然后再请一位‘盲人导演’（VLM）看着模糊的剧照，指导机器人怎么补全面部表情和手指细节，并纠正不到位的肢体动作。”**

下面我用大白话结合底层技术细节，给你把这个 process 拆解一下，希望能 build up your intuition。

### 1. 场景类比

假设你给机器人一句指令：“一个人开心地边走边挥手，并用手指着天空”。

**第一步：找真人替身**。我们没有直接从文本生成机器人动作的数据，但有海量人类的动作数据。所以，先用一个文本生成人体动作的模型（PhysDiff）生成一段人类动作。这就像找了个替身演员先把戏演了。

**第二步：套皮**。替身演员的骨骼跟机器人（GR1）不一样高、不一样胖。直接套皮会穿模或者够不到。所以先把机器人和虚拟替身摆成相同的 T-pose，调整替身的身材参数（SMPL 的 $\beta$），让两人的关键骨骼关节对齐。

**第三步：动作重定向**。利用 Inverse Kinematics (IK)，让机器人的手腕、手肘、膝盖等关键部位去“追赶”替身演员对应部位的空间坐标。这就得到了初步的机器人动作。

**第四步：盲人导演修戏**。由于替身演员没演手指和脖子，加上套皮有误差，动作可能跟原台词对不上。这时候请出 GPT-4V 导演。导演看不清连续视频，只看 4 张关键剧照。导演发现：“台词说要指天，但手只举到了肩膀，而且没有手指动作，头也没抬”。导演就会下达指令：“把右手腕再往上抬 10cm，张开食指，头向上仰 15 度”。

**第五步：真机表演**。真机直接按这套路子动会摔倒。于是把动作劈开：下半身走路交给传统的平衡控制器，上半身严格执行导演导好的动作。

---

### 2. 技术深水区

虽然面上看起来像个 pipeline 的拼凑，但每一步的设计都暗含了对当前 Foundation Model 和 Robotics 交互接口的深刻理解。

#### A. Body Shape Optimization: 把 $\beta$ 拟合到 Robot
在这个公式 $\beta^* = \arg\min_{\beta} \sum_{j} \| S_j(\beta) - H_j \|^2$ 里：
*   $\beta$: SMPL 模型的 body shape 参数，10 维向量，控制高矮胖瘦。
*   $S_j(\beta)$: 给定身材 $\beta$ 时，SMPL 模型上第 $j$ 个关节的三维坐标。
*   $H_j$: GR1 机器人上对应关节的三维坐标。
*   $\beta^*$: 优化出来的最匹配 GR1 的虚拟人类身材。
**Intuition**: 绝对不能跳过这步。如果在错误的尺度下做 IK，每一个 frame 都在积累物理偏差，最后语义全变了。

#### B. Task-Space Control Primitives: VLM 与 Robot 的接口设计
这是整篇 paper 最聪明的地方。VLM 完全不懂什么是关节角（joint angle $\mathbf{q}$），你让它直接输出 30 维的角度向量，它会 hallucinate 出天马行空的乱码。

作者设计了一套“Control Primitives”，本质上是一个受限的动作 API。VLM 只能调用类似 `move_left_wrist(direction='up', distance=10cm)` 这样的原语。
*   底层的 pink IK solver 会把这个 task-space 命令转化为 joint velocity，进而更新关节配置 $\mathbf{q}$。
*   **Intuition**: Foundation Model 在 low-level control 上很弱，但在 high-level semantic reasoning 上很强。你要在它们中间放一个“翻译层”。在这个案例里，翻译层就是基于 IK 的 Cartesian space primitives。这跟 Eureka 让 LLM 写 reward function，Code as Policies 让 LLM 写 Python 代码是同一种哲学。

#### C. VLM 跨模态对齐闭环
公式：$Q^* = Q_b \oplus Q_f \oplus Q_h$
*   $Q_b$: 身体关节序列，经过 VLM iterative adjustment。
*   $Q_f$: 12 维手指关节序列，由 VLM 看着渲染图生成。
*   $Q_h$: 3 维颈部关节序列，由 LLM 纯文本推理生成并插值。
*   $\oplus$: Concatenation。
**Intuition**: 为什么只让 VLM 调整 Arm？因为腿脚在真机上由 ZMP controller 接管了，调了也白发指令。把 VLM 的注意力限制在上半身，既减少了搜索空间，又避免了 VLM 干预它不懂的平衡动力学。

---

### 3. 为什么这个方法 Work？我的几点 Intuition

1.  **Embodiment Gap 足够小，使得 Retargeting 成为可能**。如果是四足机器人或者轮式机器人，Human Motion Prior 就废了。Humanoid 恰好处于“跟人类相似度高到可以 borrow prior，但又有差异需要 correction”的黄金分割点。
2.  **VLM is a Critic, not a Generator**。Baseline 里让 LLM 直接从 T-pose 开始生成动作，结果惨不忍睹。VLM 没有 human kinematic dynamics 的 prior，让它从零生成动作违背了它的能力边界。但在一个 80 分的初稿上做 critique 和 refinement，恰好是 VLM 的强项。
3.  **Image is the universal interface between VLM and Physical World**。VLM 读不懂 30 维的浮点数时间序列，但 VLM 读得懂渲染图。通过把 robot state 渲染成 image 喂给 VLM，作者实际上把 robot state 映射到了 VLM 的语义空间里。这是一种极其高效的 sim-to-vision 桥接。

---

### 4. 联想与 Future Directions

顺着这篇 paper，我很自然地产生几个联想：

*   **Sim-to-Real Rendering Gap 的反噬**：VLM 是看着仿真渲染图做判断的。如果渲染图太简陋，VLM 可能会误判深度或者遮挡。如果渲染图太逼真，VLM 可能会把光照阴影误判为动作的一部分。未来可能需要针对 VLM 做特定的 Sim rendering 优化（类似于对抗训练，让渲染图符合 VLM 的分布预期）。
*   **闭环到 Real Robot**：现在 VLM 看的是仿真截图。如果直接把真机上摄像头拍到的画面喂给 VLM 做 critique 呢？这就形成了一个真正的 real-world agentic loop。难点在于真机摄像头的视角和遮挡问题。
*   **Whole-Body RL 替代 ZMP**：文章最后提到了 ExBody 那种基于 RL 的 whole-body control。如果把这个 system 做成 end-to-end，文本 -> Diffusion -> VLM Critic -> RL Policy，不需要把上下半身解耦，机器人就能做出“金鸡独立”这种需要全身动态平衡的复杂动作。
*   **Tokenize the Primitives**：目前的 Control Primitives 是人工硬编码的（上下左右移动）。如果让 VLM 自己学会生成 free-form 的 Python 代码来控制 IK 呢？类似于 Code as Policies，这样就能组合出无限丰富的微调动作。

### 5. Reference Links
这里是相关工作的链接，供你 deep dive：
*   **HARMON Project Page**: https://ut-austin-rpl.github.io/Harmon/
*   **PhysDiff (ICCV 2023)**: https://research.nvidia.com/labs/toronto-ai/PhysDiff/
*   **ExBody (Expressive Whole-Body Control)**: https://expressive-humanoid.github.io/
*   **OmniH2O (Real-Time Teleoperation)**: https://humanoid-teleoperation.github.io/
*   **Eureka (LLM for Reward Design)**: https://eureka-research.github.io/
*   **Code as Policies**: https://code-as-policies.github.io/
*   **SMPL Model**: https://smpl.is.tue.mpg.de/
*   **HumanML3D Dataset**: https://github.com/EricGuo5513/HumanML3D
*   **Yoshida et al. Alter3 (Text to Motion via LLM)**: https://arxiv.org/abs/2312.06571

总而言之，HARMON 并没有发明什么惊天动地的新架构，它更像是一个精巧的 System Engineering，巧妙地把 Generative AI 的 prior 和 Foundation Model 的 reasoning 塞进了一个经典的 Robotics pipeline 里。这种“先验 + 语义纠错器”的范式在缺乏 paired data 的 Robot Learning 领域会越来越常见。

---

# HARMON: 从语言描述生成人形机器人全身动作

这是一篇 UT Austin + NVIDIA Research 的工作，核心思路非常优雅：**把 human motion generation 的成熟成果当作先验，把 VLM 的 commonsense reasoning 当作"打磨器"**，从自由文本生成可在真实 humanoid 上执行的动作。下面从 problem setup、pipeline、关键设计、实验细节四个维度拆解。

---

## 1. Problem Setup

给定一段自然语言描述 $X$，目标是产生一串 robot joint configurations：

$$Q = \{ \mathbf{q}_1, \cdots, \mathbf{q}_T \}, \quad \mathbf{q}_i \in \mathbb{R}^c$$

- $Q$：整个动作序列
- $\mathbf{q}_i$：第 $i$ 个时刻的关节配置向量
- $T$：序列长度（帧数）
- $c$：humanoid 的自由度数量（GR1 上是个不小的数字，论文里没明确但可以推断手部 12 + 颈部 3 + 其他身体关节）

关键挑战：**没有大规模 (language, humanoid motion) 配对数据**。human motion 数据集（HumanML3D [1]、AMASS [2]、CMU MoCap [5]）虽然海量，但都是 SMPL 人体参数，且常常缺头/手动作；直接 retarget 到 humanoid 会丢失语义、丢失表达力。

---

## 2. Pipeline 总览（对应 Figure 2）

整个 pipeline 分三阶段：

```
Text X
  └──> PhysDiff (text-conditioned diffusion) 
         └──> SMPL params P = {(θ_i, t_i)}
                └──> IK retargeting (pink solver)
                       └──> Q_r (初步 humanoid joint configs)
                              └──> VLM editing (GPT-4 / GPT-4V)
                                     ├── finger motion Q_f
                                     ├── head motion Q_h
                                     └── iterative arm adjustment
                                            └──> Q_b
                                                   └──> Q* = Q_b + Q_f + Q_h
                                                          └──> split upper/lower → real robot
```

### 2.1 PhysDiff → SMPL 序列

PhysDiff [6] 是 Yuan 等人在 ICCV 2023 提出的 physics-guided motion diffusion model，它在标准 diffusion (MDM [42]) 之上注入物理约束（地面接触、重力、动量），生成的 human motion 不穿模、不浮空。输出是 SMPL 参数序列：

$$P = \{(\theta_1, \mathbf{t}_1), \cdots, (\theta_T, \mathbf{t}_T)\}$$

- $\theta_i \in \mathbb{R}^{72}$：SMPL 的 24 个关节 × 3 轴 axis-angle 旋转参数（包括 root orientation 在内）
- $\mathbf{t}_i \in \mathbb{R}^3$：根节点的全局平移
- $\beta \in \mathbb{R}^{10}$：body shape 参数，编码身高、体型、四肢比例等

SMPL forward kinematics 把参数映射成关节位置：

$$J_i = S(\theta_i, \mathbf{t}_i, \beta) \in \mathbb{R}^{24 \times 3}$$

$S$ 是 SMPL 蒙皮函数，$J_i$ 是 24 个人体关节在第 $i$ 帧的三维坐标（global frame）。

### 2.2 Body Shape 优化（缩小 embodiment gap）

由于 humanoid（GR1）和默认 SMPL 模型身材不同，如果直接 retarget 会因尺度不匹配而够不到目标位置。作者借鉴 He et al. 的 H2O [27] 做法：让 SMPL 和 humanoid 都摆到 T-pose，选择 17 个对应关节对，最小化位置差：

$$\beta^* = \arg\min_{\beta} \sum_{j \in \text{17 pairs}} \| S_j(\beta) - H_j \|^2$$

- $S_j(\beta)$：SMPL 在 T-pose 下第 $j$ 个关节的位置（依赖 $\beta$）
- $H_j$：humanoid 上对应关节的位置

用 Adam optimizer [60] 迭代得到 $\beta^*$，再把它固化下来计算后续每一帧的 $J_i$。**这一步的直觉是：先把"骨骼尺寸"对齐，再谈运动学对齐，否则后续 IK 永远在错误的尺度上挣扎。**

### 2.3 IK Retargeting

用 pink [61]（基于 Pinocchio 的 Python IK 库）求逆运动学。关键关节（wrist、elbow、shoulder、knee、ankle）被当作 target，IK 解算 joint velocities 逐帧更新 $\mathbf{q}$。最终得到 retargeted joint sequence $Q_r$。

**注意这里的设计哲学**：不在关节角空间直接对齐，而在 task space（关键笛卡尔位置）上对齐。这样语义信息（"手要举到头顶"）才能保留下来。直接在 joint space 对齐会因 joint 树拓扑不同而丢失语义。

---

## 3. VLM-Based Motion Editing（对应 Figure 3）——这篇 paper 的真正创新点

Retarget 后还有三类问题：

1. **手指动作缺失**：SMPL 没有手指
2. **头部动作缺失**：HumanML3D 等数据集基本没有 neck
3. **retargeting 误差**导致语义偏移：比如文本说"挥手过头顶"，IK 解出来手只到肩膀

作者用 GPT-4 / GPT-4V 当 judge + editor，把渲染后的 humanoid motion 当作"语言模型可以读懂的中间表示"。

### 3.1 Finger Motion Generation（上左）

输入：
- 4 张等间隔采样的渲染帧（论文实测 4 帧够，更多帧会降低 VLM 推理质量——这是个有意思的发现，VLM 对长 image sequence 不鲁棒）
- 从原始 $X$ 用 GPT-4 提取的 finger motion description $X_f$

输出：
$$\bar{\mathbf{q}}_i^f \in \mathbb{R}^{n_f}, \quad n_f = 12$$

- $\bar{\mathbf{q}}_i^f$：第 $i$ 个时间间隔的 finger joint configurations
- $n_f = 12$：GR1 单手 6 个手指关节 × 2 只手 = 12（GR1 灵巧手的简化配置）

四个区间分别生成，然后 concat 成 $Q_f$。**协调性靠"看视频"实现**：GPT-4V 看到当前手臂在什么位置，再决定手指要做什么。这比直接从文本硬猜手势要靠谱得多。

### 3.2 Head Motion Generation（上右）

头部动作维度低（3 个 neck joints）、相对独立，所以不走 VLM 视觉通路，直接用 GPT-4 文本推理：

输入：
- Head motion description $X_h$（同样从 $X$ 提取）
- 总帧数 $T$
- FPS

输出：
$$\mathbf{q}_i^h \in \mathbb{R}^3$$

- 3 个 neck joints（yaw / pitch / roll）

GPT-4 自主决定关键帧位置，然后线性插值得到 $Q_h$。让 GPT-4 自己挑 keyframe 是关键：高频率头部运动（比如点头、摇头）才能被表达出来，固定 keyframe 间隔就僵了。

### 3.3 Iterative Arm Adjustment（底部）——这个 loop 设计很精妙

这是整个 method 最 "Agentic" 的部分，本质是个 actor-critic 双 agent 闭环：

**Judgment Agent（GPT-4V）**：
- 输入：4 帧 screenshot + 原始 motion description $X$
- 输出：(a) 对当前动作的 caption 描述；(b) 是否匹配 $X$ 的判断；(c) 改进建议（自然语言）

**Adjustment Agent（GPT-4）**：
- 输入：同样 4 帧 + Judgment Agent 给的建议
- 输出：一组 **control primitives** 的组合

**Control Primitives 的设计**（这是关键工程决策）：

不让 VLM 直接编辑 joint angle——因为 joint → motion 的映射对人来说不直观，VLM 推理会出错。只允许它在 task space 层面操作，预定义一组原语：

> "把左手腕往上移动 10cm"
> "把右手腕往胸口方向拉近"

每个原语底层用 IK 求解对应 wrist 目标位置的偏移。

**为什么只调 upper body**？因为 lower body 在 real robot 上由独立的 ZMP locomotion controller 控制，retarget 出来的 lower body motion 只用来提取 pelvis trajectory 作为 locomotion command，所以编辑它没意义。

**终止条件**：Judgment Agent 确认 aligned，或超过 2 轮迭代。还多加一步"判断当前 motion 是否还能被这些 primitives 改进"，不能就跳过——避免无意义的 loop。

最终：
- 如果有过 adjustment：取最后一轮的 $Q_b$
- 如果没有：$Q_b = Q_r$

合成：
$$Q^* = Q_b \oplus Q_f \oplus Q_h$$

$\oplus$ 表示按关节维度拼接（finger 12 维 + head 3 维 + body 其余维）。

---

## 4. Real Robot Execution（对应 Sec 3.3）

直接把 $Q^*$ 发给真机会摔——因为没有动力学和平衡考虑。借鉴 Cheng et al. ExBody [26] 的做法：

```
Lower body trajectory (pelvis x,y) → ZMP controller [62] → locomotion
Upper body joints (arms + head + fingers) → joint position control
```

- ZMP（Zero Moment Point）是 Vukobratovic 1969 的经典双足步行稳定性判据：保持 ZMP 在支撑多边形内就不会倒
- 这种 decoupling 是当前 humanoid 上体表达性工作的标准范式，OmniH2O、ExBody、H2O 都这么做

代价是：复杂 upper body 动作可能让上身质心偏移过大，破坏 ZMP 假设——论文 conclusion 也承认这点，提到未来要用 RL-based whole-body control。

---

## 5. Experiments

### 5.1 Test Set 构造

- 第一部分：HumanML3D [1] test set 随机采样——主要测 body motion
- 第二部分：用 GPT-4 生成包含 head/finger 的描述——测 whole-body 表达力
- 总共 ~50 条描述

### 5.2 Baselines

1. **VLM-based Motion Generation**（Yoshida et al. [28] 路线）：去掉 human motion prior，从 SMPL T-pose 直接开始 VLM editing；lower body 用 HARMON 的（公平比较）。验证 human motion prior 的价值。

2. **Human Motion Retargeting**：只用 retargeted $Q_r$ + HARMON 的 $Q_f, Q_h$。验证 iterative adjustment 的价值。

3. **HARMON w/o Head or Finger**：ablation，验证 head/finger 生成的价值。

### 5.3 Human Study 结果（Figure 4）

12 个参与者 × 1728 个评估。

| Method | Normalized Score |
|---|---|
| **HARMON** | **81.2%** |
| Human Motion Retargeting | 较低（arm 项明显掉） |
| HARMON w/o Head/Finger | 在 head/finger 项掉 |
| VLM-based Motion Generation | 最低 |

分 body part 看：
- VLM-only 几乎全维度都垫底 → **没有好的初始化，VLM 直接写 joint 配置等于盲写**
- Retargeting-only 在 arm 上掉分 → 证明 iterative adjustment 真的在修语义偏移
- w/o head/finger 在对应项掉，但仍非零 → 因为有些 test case 本来就不涉及这些部位

整体 human evaluators 在 **86.7% 的测试 case 上偏好 HARMON**。

### 5.4 Qualitative Results（Figure 5）

四个 case 都很有启发性：
1. **Clapping**：retargeting 已经足够好（高频周期性动作 human prior 强）
2. **Point + 手放胸口**：finger + head 的 VLM 生成显著提升表达力
3. **Retargeting 误差导致语义漂移**：iterative adjustment 修正回来
4. **PhysDiff 本身就没生成对**：iterative adjustment 也能修正上游 diffusion 的错误

这第四点很关键——说明 VLM loop 不仅修 retargeting 误差，还能修生成模型本身的语义偏差。**HARMON 把 VLM 当成跨模态对齐的"语义纠错器"**。

---

## 6. Intuition Building: 为什么这个方法 work？

我个人读下来的几点直觉：

### 6.1 "Humanoid is close enough to human to borrow, but different enough to need correction"

这是整篇论文的底层假设。Human motion prior 给你 90% 的"自然性 + 物理合理性"，剩下 10% 是 embodiment-specific 的语义对齐问题，丢给 VLM 处理。

### 6.2 VLM 不擅长生成，但擅长 critique

VLM 直接从文本写 joint angle 是灾难（baseline 已证明），但让它**看视频 → 判断对不对 → 提建议**，再让另一个 LLM 把建议翻译成 IK primitives——这个 division of labor 是合理的。VLM 在 perception + commonsense 上强，在 motor control generation 上弱。

### 6.3 Task-space primitives 作为 VLM ↔ robot 的接口

这是我觉得最巧妙的工程决策。joint angle 不是好的 LLM-action 接口（不可解释、维度灾难、不同 robot 不通用）；自然语言动作描述太抽象（"挥挥手"是几度？）；**task-space 上的离散 primitives**（"左腕向上 10cm"）恰好落在 sweet spot：VLM 能理解，robot 能执行，可组合。

这跟 Eureka [56] 的"reward code as LLM-robot interface"、L2R [58] 的"reward as interface"是同一思想谱系——**找一个语义-控制双向友好的中间层**。

### 6.4 Iterative loop 的必要性

单次生成很容易错，但 judgment-adjustment-judgment 的闭环能在少量迭代内收敛。作者限制 2 轮迭代 + "判断是否还能改进"的早期退出，体现了对 LLM loop 成本和不稳定性的 awareness。

---

## 7. Limitations & Future Directions（论文自述 + 我的补充）

- **Primitives 是固定的**：当 human motion 严重偏离文本（比如高频动作），有限 primitives 调不动。作者提到让 VLM 自己生成 free-form primitives——这其实就是把 control interface 也变成 VLM 的输出空间，要做 safety bound。
- **Upper/Lower decoupling**：复杂上身动作可能让机器人失衡。未来用 RL whole-body control（ExBody 路线）替换 ZMP locomotion controller。
- **VLM 推理成本**：每条 motion 要跑多轮 GPT-4V，单次生成可能要几分钟。对 offline motion synthesis OK，对 realtime interaction 不行。
- **仿真渲染保真度**：VLM 看的是仿真渲染图，如果 sim-to-real 在视觉上有 gap，VLM 的判断可能在 sim 中正确但 real 中跑偏。
- **Body shape 优化是全局的**：用单个 $\beta^*$ 适配所有动作。但不同动作可能需要不同身材先验（蹲下 vs 跳跃）。可考虑 per-motion $\beta$。

---

## 8. 与同期工作的定位

- **ExBody [26]** / **OmniH2O [27]**：RL-based humanoid imitation，需要大规模训练，但生成的是 physically-consistent policy。HARMON 不训 policy，只生成 trajectory。
- **Yoshida et al. Alter3 [28]**：纯 LLM 直接生成 humanoid joint，结果是僵硬不自然。HARMON 用 human prior 弥补。
- **PhysDiff [6]** / **MDM [42]** / **MotionDiffuse [43]**：text-to-human-motion 的 SOTA。HARMON 把它们当 backbone。
- **Eureka [56]** / **L2R [58]** / **RL-VLM-F [59]**：用 LLM/VLM 生成 reward function。HARMON 用 VLM 做 trajectory critique，思想类似但接口不同。

---

## References & Useful Links

- **HARMON 项目页**：https://ut-austin-rpl.github.io/Harmon/
- **PhysDiff (Yuan et al. ICCV 2023)**：https://research.nvidia.com/labs/toronto-ai/PhysDiff/ 
- **SMPL**：https://smpl.is.tue.mpg.de/
- **HumanML3D**：https://github.com/EricGuo5513/HumanML3D
- **AMASS**：https://amass.is.tue.mpg.de/
- **ExBody (Cheng et al.)**：https://expressive-humanoid.github.io/
- **OmniH2O / H2O (He et al.)**：https://humanoid-teleoperation.github.io/
- **Yoshida et al. Alter3**：https://arxiv.org/abs/2312.06571
- **pink IK library**：https://github.com/stephane-caron/pink
- **Fourier GR1**：https://www.fourierintelligence.com/products/gr1-dual-form-humanoid-robot
- **MDM (Tevet et al.)**：https://guytevet.github.io/mdm-page/
- **MotionDiffuse**：https://mingyuan-zhang.github.io/projects/MotionDiffuse.html
- **Eureka**：https://eureka-research.github.io/
- **L2R**：https://language-to-reward.github.io/
- **RL-VLM-F**：https://rl-vlm-f.github.io/
- **DeepMimic**：https://xbpeng.github.io/projects/DeepMimic/index.html
- **AMP**：https://github.com/xbpeng/DeepMimic

---

整体看，HARMON 是一个典型的 **"先验 + LLM-as-refiner"** 范式样本：先用成熟 generative model 解决 80% 的分布先验问题，再用 foundation model 的 commonsense 做 20% 的对齐和打磨。这种 pattern 在 robot learning 里会越来越常见——无论是 trajectory generation、policy fine-tuning，还是 reward design，都可以套这个 template。对 humanoid 这个具体领域来说，**embodiment gap 小到可以 retarget + VLM-edit 就能 work**，是这个方法成立的物理基础；对其他形态差异大的 robot（比如 quadruped、无人机），就需要重新设计 prior 和 edit interface 了。
