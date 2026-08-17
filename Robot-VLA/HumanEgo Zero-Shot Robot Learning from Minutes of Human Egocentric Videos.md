---
source_pdf: HumanEgo Zero-Shot Robot Learning from Minutes of Human Egocentric Videos.pdf
paper_sha256: 6ab2959cf3ab8739154687fd1365a30c85b9c196a4d6d320ce15aae157ae84eb
processed_at: '2026-08-05T07:50:58-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍

---

## 一句话版本

**人戴着眼镜干活 30 分钟，录下来，机器人就能学着干同样的活，连 robot data 都不用。**

听起来像作弊，但他们是真做到了 92.5% 成功率。

---

## 核心问题是什么

机器人学技能，传统路子是 teleoperation——人操纵机器人干几十遍，记录 joint trajectory，再训练 policy。慢、贵、累。一个 task 随便就几百条 demonstration。

那能不能直接录人干活？反正人干活又快又自然，30 秒一条，30 分钟能录 60 条。

问题在于 **embodiment gap**——人手和 robot gripper 长得完全不一样，camera 视角不同，kinematics 不同。你直接让 robot 学人的 wrist trajectory，大概率废掉。

---

## 他们怎么解决的

关键 insight 很简单：**机器人不该模仿人的身体，该模仿的是"手和物体之间的空间关系怎么演化"。**

抓杯子这件事，不管你是人手还是 robot gripper，"手指接近杯子→接触→合拢→提起→移动到目标位置"这个 **relative geometry 的变化过程** 是一样的。

所以他们设计了一个叫 ICT 的东西——把场景里每个 entity（手、物体）都用 29 个数字描述，核心是 **"从我这个 entity 看，左手在哪、右手在哪"**。这是 relative transform，跟 camera 在哪、robot 是什么型号完全无关。

这就是为什么能 zero-shot transfer——你换 robot、换 camera、换桌子，ICT 的数值不变，policy 看到的 input 一样，输出自然一样。

---

## 整个 pipeline 长这样

1. **人戴 Aria glasses 录视频**，随便在厨房、客厅、哪都行，不用标定
2. **把人手从画面里抠掉**（SAM2 分割 + LaMa inpainting），再画一个虚拟 gripper 上去——这样视觉上就不像人手了
3. **从视频里提取 ICT**：用 Aria 自带的 stereo hand tracking 拿到 3D 手部 keypoint，用 Grounding DINO + SAM2 + CoTracker3 拿到物体 3D pose，然后算 relative transform
4. **训练一个 flow matching policy**：输入 ICT + RGB，输出未来 50 步的双臂 action
5. **部署到机器人**：直接跑，不用 fine-tune

---

## 几个我觉得 clever 的设计

### MCP joint 而不是 fingertip

把人手 retarget 成 gripper 的时候，naive 想法是 fingertip 中点当 position。但 orientation 怎么定？

如果用 wrist→fingertip 方向当 forward axis，**捏东西的瞬间两个指尖碰到一起，这个 axis 就退化成零向量了**，整个 frame 崩掉。

他们的解法：用 MCP joint（指根关节）而不是 fingertip 来定 orientation。因为 MCP joint 在整个抓取过程中始终保持分开，frame 数值稳定。这是一个很实际的工程 trick，但效果决定成败。

### Flow matching 而不是 diffusion

Diffusion policy 很火，但 inference 要 50-1000 步 denoising，太慢。Flow matching 学的是直线轨迹，20 步 Euler ODE 就够，control loop 能跑 10 Hz。

对 robot 来说，inference latency 是硬约束，不是 "可以优化一下" 的软约束。

### 三个 auxiliary loss 榨干每条视频

30 分钟数据很少。他们给 policy 加了三个额外任务：
- 预测物体未来怎么动
- 预测 2D 像素轨迹
- 预测未来 latent state

三个都 share 同一个 encoder。效果是 encoder 被迫学一个 **mini world model**——理解"我做的事会让世界怎么变"。

15 分钟数据下，这三个 loss 加起来涨了 25 个百分点。数据多了之后收益递减，因为 data 本身够 supervise 了。

---

## 最震撼的实验结果

### Fig 9：visual 不是 bottleneck，spatial state 才是

| 输入 | 成功率 |
|---|---|
| 原始人手 RGB | 7.5% |
| 抠掉人手 + 画 keypoint | 20% |
| **直接用 robot RGB（视觉上零 gap）** | **32.5%** |
| 加上 ICT | 85% |
| 完整系统 | 95% |

就算你把视觉 gap 完全消除——直接用 robot 自己的 camera 看自己——也只有 32.5%。但加上 ICT（29 个数字的 spatial token），直接跳到 85%。

**这说明 manipulation 的核心不在 pixel appearance，在于 3D 空间关系的显式表示。** 你在 Tesla 讲的 vector space 逻辑，这里是 manipulation 版的同构证据。

### Fig 5：8 分钟人视频 > 30 分钟 robot teleop

人干活天然比操纵机器人快、smooth、没 idle time。所以每分钟人视频的 information density 是 teleop 的 ~3.75 倍。

### Appendix E.2：没有 co-training sweet spot

固定 30 分钟 budget，从 0% human 到 100% human，success 单调上升。**纯人视频最好，加一点 robot data 反而变差。**

这挺反直觉的。大家默认觉得 "robot data 对 deployment 是必要的"，但至少在 ICT representation 下，human video 是 strictly superior 的 per minute。

---

## 最大的 limitation

**依赖 stereo hand tracking。** Aria glasses 有两个 SLAM camera，能恢复 metric depth。换成 monocular tracker（WiLoR、HaMeR、MediaPipe），success 从 95% 掉到 45%、32.5%、0%。

原因是 monocular 有 scale ambiguity，5-11cm 的 depth offset 直接污染 ICT 的 reference frame。policy 学到的 grasp 位置全是错的。

这是 hardware lock-in——你得有 Aria glasses 或类似 stereo setup 才能跑通这个 pipeline。

---

## 为什么我觉得这个工作重要

它证明了一个 thesis：**manipulation 的本质是 hand-object interaction geometry 的演化，把这个 geometry 显式编码 + generative policy + dense supervision，可以在极小数据量下学到 deployable skill 并跨 embodiment transfer。**

不需要 internet-scale pretraining，不需要 per-task robot data，不需要 large VLA model。30 分钟，6 层 transformer，92.5%。

这在 VLA 越 scale 越大的趋势里是一个反方向的 data point——**representation 的选择仍然 matter，structured prior 仍然能 punch above its scale。**

当然 limitation 也很明显：stereo dependency、per-frame perception pipeline 脆弱、~1cm 精度 ceiling。但作为一个 "证明 human video 可以 directly become robot policy" 的 starting point，信号很强。

---

# HumanEgo 深度解读

Karpathy 你好，这篇 paper 是 University of Maryland 的工作，核心 claim 很 aggressive：**只用 30 分钟人类第一人称视频，zero-shot 跨 embodiment 部署到机器人，4 个真实任务平均 92.5% 成功率**，而且超过了等量时间的 robot teleoperation。下面我从 representation、learning、experiments 三个层面拆解，重点 build intuition。

---

## 1. Core Thesis：什么真正 transfer 跨 embodiment

传统 imitation learning 思路是 "imitate the body"——retargeting 把人手渲染成 robot gripper，或者直接学习 wrist trajectory。HumanEgo 的 thesis 是 **"recover the task-relevant interaction geometry that transfers across bodies"**。

这句话的几何含义值得展开：抓一个杯子这件事，不论你是用 Aria glasses 在厨房里、还是 Trossen WidowX 在 lab 里，"hand 相对于 cup 的 SE(3) relative transform 在 contact 时刻的演化"是 invariant 的。这就是 ICT 的数学根。

---

## 2. ICT (Interaction-Centric Tokens)：29D 的紧凑 state representation

### 2.1 公式 (1) 逐项拆解

$$\mathrm{ICT}_k = \big[\underbrace{\tau}_{1} \mid \underbrace{T_E^{\mathrm{REF}}}_{9} \mid \underbrace{{}^{E}T_{LH}}_{9} \mid \underbrace{{}^{E}T_{RH}}_{9} \mid \underbrace{g}_{1}\big] \in \mathbb{R}^{29}$$

变量含义：
- $\tau \in \{0,1\}$：entity type flag（0=hand, 1=object）
- $T_E^{\mathrm{REF}} \in SE(3)$ flatten 成 9D：entity $k$ 在共享 reference frame（静态 camera frame）下的 pose。3D translation + 6D rotation representation（来自 [Zhou et al. CVPR 2019](https://arxiv.org/abs/1812.07035)，用 rotation matrix 的前两列，因为连续可微，避免了 quaternion 的 double-cover 和 Euler angle 的 gimbal lock）
- ${}^{E}T_{LH}$：left hand 的 pose，**表达在 entity $k$ 自己的 local frame $E$ 下**——这是关键，意味着 "从 entity $k$ 看，left hand 在哪"
- ${}^{E}T_{RH}$：right hand 同理
- $g \in [0,1]$：grasp state，对 hand 是 normalized thumb-index 距离，对 object 是 sentinel

### 2.2 为什么是 interaction-centric 而非 entity-centric

对比 EgoZero / PointPolicy 等 point-based 方法：它们把 hand 和 object 都当成 3D point cloud 里的点，丢失了 "谁抓谁" 的结构。对比 SPOT 等 object-centric 方法：只 track object 6-DoF，hand 是 implicit 的，丢失了 approach / grasp / release 的相位信息。

ICT 的设计哲学：**每个 entity 都用 "其他 entity 相对于我" 来描述自己**。这相当于在 representation 层面就编码了一个完全图（complete graph）的 relative pose，transformer 的 self-attention 天然适合处理这种 set-of-tokens。

### 2.3 与 camera frame 的解耦

关键：${}^{E}T_{LH}$ 和 ${}^{E}T_{RH}$ 都是 relative transform，camera pose 完全不进入这个 representation。这就是 zero-shot 跨 viewpoint transfer 的数学根——你换 camera、换 robot、换 mounting height，ICT 数值不变。

附录 E.3 有一个有趣的 trade-off：**anchor frame**（pose 相对于第一个被抓的物体）在 low-data regime 更好，因为它注入了 "grasping prior" 的 inductive bias；但 camera frame 在 large-data regime 略胜，因为不继承 upstream perception noise。这其实是一个经典的 bias-variance trade-off 例子。

---

## 3. Visual Preprocessing：把人手 "挖掉" 再 "填回去"

两步 pipeline：
1. **SAM2** 分割人手 + 手臂 → **LaMa** ([Lugaresi et al.](https://arxiv.org/abs/2109.07161)) inpainting 填充背景
2. 渲染 virtual gripper + object keypoints 到 inpainted 图像

这步的 intuition：消除 visual embodiment gap 的最简单方式是直接消除 embodiment 的视觉证据。LaMa 用 Fourier convolutions 处理大 mask，比传统 inpainting 更鲁棒。

但 ablation（Fig 9）揭示了一个反直觉的事实：

| Visual 配置 | Success Rate |
|---|---|
| Raw human RGB | 7.5% |
| Keypoint rendering + arm inpainting | 20% |
| Robot RGB（zero visual mismatch） | 32.5% |
| ICT + raw RGB | 85% |
| Full system | 95% |

**即使完全消除 visual mismatch（用真 robot RGB），也只有 32.5%**。这说明 visual appearance 不是 manipulation 的 bottleneck，explicit spatial state 才是。这与你在 Tesla 强调的 "vector space" 思路异曲同工——pixel 是 appearance，vector space 是 structure。

---

## 4. Hand-to-Gripper Retargeting：一个精巧的 SE(3) 构造

这是 paper 里最容易被忽略但最 critical 的工程细节。把 21-keypoint 人手 retarget 成 6-DoF gripper pose + 1-DoF grasp。

### 4.1 位置

$$\mathbf{p}_{\mathrm{ee}} = \frac{1}{2}(\mathbf{p}_{\mathrm{thumbtip}} + \mathbf{p}_{\mathrm{indextip}})$$

thumb tip 和 index tip 的中点，对应 parallel-jaw grasp center。

### 4.2 朝向——这里是关键 trick

naive 方案是 wrist→fingertip-midpoint 作为 forward axis，但**在 pinch grasp 瞬间两个 fingertip 收敛到同一点，jaw axis 退化**。HumanEgo 的解法是**用 MCP joint 而非 fingertip**：

$$\mathbf{x}_{\mathrm{ee}} = \widehat{\mathbf{p}_{\mathrm{iMCP}} - \mathbf{p}_{\mathrm{tMCP}}}, \quad \mathbf{y}_{\mathrm{ee}} = \tilde{\mathbf{y}} - (\tilde{\mathbf{y}}^\top \mathbf{x}_{\mathrm{ee}})\mathbf{x}_{\mathrm{ee}}, \quad \mathbf{z}_{\mathrm{ee}} = \mathbf{x}_{\mathrm{ee}} \times \mathbf{y}_{\mathrm{ee}}$$

其中 $\tilde{\mathbf{y}} = \frac{1}{2}(\mathbf{p}_{\mathrm{tMCP}} + \mathbf{p}_{\mathrm{iMCP}}) - \mathbf{p}_w$（wrist→MCP midpoint），$\widehat{(\cdot)}$ 是单位化。

intuition：MCP joint 在 pinch grasp 全周期都保持 well-separated，所以 Gram-Schmidt 构造的 frame 数值稳定。这是一个 "anatomically stable landmark" 的选择，类似于 face tracking 用 ear 而非 mouth 作为 anchor。

### 4.3 Grasp scalar

$$g = \mathrm{clip}\left(\frac{\|\mathbf{p}_{\mathrm{thumbtip}} - \mathbf{p}_{\mathrm{indextip}}\| - d_{\min}}{d_{\max} - d_{\min}}, 0, 1\right)$$

$d_{\min}, d_{\max}$ 是 per-user 标定的 closed/fully-open 距离。

---

## 5. Flow Matching Policy：为什么不是 Diffusion

### 5.1 公式 (2)

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[w_p \|\Delta\mathbf{p}\|^2 + w_r \|\Delta\mathbf{r}\|^2 + w_g \|\Delta g\|^2\right]$$

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \quad \Delta(\cdot) = v_\theta(\mathbf{x}_t, t, s_t) - (\mathbf{x}_1 - \mathbf{x}_0)$$

变量：
- $t \sim \mathcal{U}(0,1)$：flow time，与 diffusion 的 timestep 不同，是连续的
- $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：Gaussian prior
- $\mathbf{x}_1$：ground-truth bimanual action chunk（K=50 步，每步 $D_a$ 维，包含双手 6-DoF pose + binary grasp）
- $v_\theta$：6-layer 8-head transformer decoder，embedding 384
- $w_p=5, w_r=1, w_g=10$：dimension-wise reweighting

### 5.2 Flow Matching vs Diffusion 的直觉

Diffusion 学习 reverse SDE，path 是弯曲的，需要很多 denoising step。Flow Matching ([Lipman et al. ICLR 2023](https://arxiv.org/abs/2210.02747); [Liu et al. ICLR 2023, Rectified Flow](https://arxiv.org/abs/2209.03003)) 学习一个 **constant velocity field**，path 是直线，Euler ODE 20 步就够。

数学上：$\frac{d\mathbf{x}}{dt} = v_\theta(\mathbf{x}_t, t, s_t)$，从 $t=0$（noise）积分到 $t=1$（action）。

这个选择对 robot policy 很关键：diffusion policy (Chi et al. RSS 2023) inference 慢（~50-1000 steps），flow matching 20 步，control loop 能跑到 10 Hz re-plan。这与 Stable Diffusion 3 用 Rectified Flow 的动机一致——inference latency 在 closed-loop control 里是 hard constraint。

### 5.3 Multi-modality 的处理

同一 task 有多种 valid strategy（比如 grasp 杯子可以从左边也可以从右边）。Flow matching 通过 sample $\mathbf{x}_0$ 不同，能 generate 多模态 action。这是 generative policy 相比 deterministic regression (ACT) 的优势。

---

## 6. Dense Auxiliary Objectives：从小数据 squeeze 监督信号

这是 paper 最 "data-efficient" 的核心。公式 (3)：

$$\mathcal{L} = \mathcal{L}_{\mathrm{FM}} + \lambda_{\mathrm{OM}}\mathcal{L}_{\mathrm{OM}} + \lambda_{\mathrm{2D}}\mathcal{L}_{\mathrm{2D}} + \lambda_{\mathrm{LC}}\mathcal{L}_{\mathrm{LC}}$$

三个 auxiliary head 共享 context encoder：

| Head | 预测目标 | 所在空间 | 15min 时增益 |
|---|---|---|---|
| Object Motion ($\mathcal{L}_{\mathrm{OM}}$) | 物体未来 6-DoF trajectory | 3D physical | +17.5 pp |
| 2D Trace ($\mathcal{L}_{\mathrm{2D}}$) | entity 未来 2D 投影轨迹 | 2D visual | +5.0 pp |
| Latent Consistency ($\mathcal{L}_{\mathrm{LC}}$) | ICT state K 步后 | latent | +12.5 pp |
| Combined | — | — | +25.0 pp |

**直觉**：一条 30 秒的 demonstration 视频蕴含远超 action label 的信号——物体怎么动、像素怎么变、latent state 怎么演化。三个 head 分别在 3D physical、2D visual、latent space 做 forward dynamics prediction，**共享 encoder 被迫学一个 lightweight world model**。

这与你反复强调的观点一致：intelligence 的本质是预测下一个 state。这里 policy 不只学 "给定 state 输出 action"，还学 "给定 state 预测未来 state 演化"——后者是 world model 的范畴，提供 prior 让 policy 在 small data 下泛化。

### 6.1 为什么 Object Motion 贡献最大（+17.5pp）

因为 manipulation 的因果链是 action → object motion。预测 object motion 等于让 encoder 理解 "我做的事会让物体怎样"，这是 task-relevant 的核心 signal。2D trace 贡献小是因为它 redundant with 3D（投影是 3D 的 lossy 版本）。Latent consistency 中间是因为它是 self-supervised，信号弱但 unbiased。

### 6.2 与 multi-task pretraining 的区别

这里不是 pretraining 一个大模型再 fine-tune，而是**在 policy 训练同时**做 multi-task。优势：auxiliary target 来自同一 perception pipeline，与 policy target 时空对齐；劣势：不能 leverage internet-scale data。HumanEgo 的 bet 是：structured representation + dense supervision > large-scale pretraining for this specific regime。

---

## 7. 实验：几个关键数字

### 7.1 主结果（Fig 4）

| Method | Serve Bread | Downstack Cups | Water Flowers | Adjust Table | Avg |
|---|---|---|---|---|---|
| **HumanEgo (30 min)** | 95 | 87.5 | 95 | 92.5 | **92.5** |
| HumanEgo (15 min) | 80 | 70 | 80 | 70 | 75.0 |
| ACT (30 min teleop) | 52.5 | 47.5 | 55 | 50 | 51.2 |
| EgoZero | 45 | 32.5 | 20 | 27.5 | 31.2 |
| Point Policy | 40 | 25 | 15 | 22.5 | 25.6 |
| ZeroMimic | 35 | 20 | 10 | 17.5 | 20.6 |
| Track2Act | 30 | 15 | 7.5 | 15 | 16.9 |
| SPOT | 25 | 10 | 5 | 12.5 | 13.1 |

### 7.2 Data Efficiency（Fig 5）

- 7 min human video → 50% success
- 8 min human (57.5%) > 30 min ACT teleop (52.5%) → **3.75× data efficiency**
- 30 min → 95%

这说明 human egocentric video 的 per-minute information density 远高于 teleop。原因（Fig 6）：human 动作更 smooth（jerk 低一个数量级）、idle time 接近 0、spatial coverage 更广。teleop 通过 piloting interface 本身就有 latency 和 noise。

### 7.3 Zero-shot Cross-condition（Fig 7, 8）

9 个 OOD 条件：background / lighting / viewpoint / distractors / novel object instance / 4 种 placement variation / 3 种 robot (Trossen/Franka/UR10) / 2 种 camera (RealSense/ZED) → 全部 85-91.25% success，无 retraining。

这是 ICT viewpoint-invariance 的直接验证。

### 7.4 Hand Tracking Study（Appendix E.1）

| Tracker | Stereo? | Success |
|---|---|---|
| Aria-MPS | Yes | 95% |
| WiLoR | No | 45% |
| HaMeR | No | 32.5% |
| MediaPipe | No | 0% |

**Stereo depth 是决定性的**。monocular tracker 有 5-11cm systematic depth offset，直接污染 ICT reference frame。这与 autonomous driving 里 LIDAR vs camera depth 的争论同构——absolute depth 在 manipulation 里是 hard requirement，scale ambiguity 不能容忍。

---

## 8. Co-training Study（Appendix E.2）：一个反直觉的发现

固定 30 min 总 budget，vary human ratio 0/25/50/75/100%：

| Human ratio | Success |
|---|---|
| 0% (pure teleop) | 65% |
| 25% | 72.5% |
| 50% | 77.5% |
| 75% | 90% |
| 100% (pure human) | 95% |

**没有 sweet spot，纯人类数据最好**。即使把 25% teleop 换成 human video（绝对 robot 数据减少 7.5 min），success 反而 +7.5pp。这反驳了 "robot data is necessary for deployment" 的常见假设——至少在 ICT representation 下，human video 是 strictly superior data source per minute。

直觉解释：teleop trajectory 本身有 control latency、operator skill variance、joint limit artifacts 等 noise；human manipulation 是 million-year evolution 优化的 motor policy，signal-to-noise ratio 天然高。

---

## 9. Pipeline 的脆弱性：Limitations

paper 自己承认几个：

1. **依赖 stereo hand tracking**：Aria Gen1 glasses 提供。换 monocular → success 崩溃。这是 hardware lock-in。未来需要更强的 monocular depth estimator（比如 [HaWoR](https://arxiv.org/abs/2501.02973) 这类 world-space hand reconstruction）。

2. **Per-frame object detection**：用 Grounding DINO + SAM2 + CoTracker3 + Orient-Anything V2 串联。任一 module failure 会 cascade。in-hand manipulation 和 fast motion 下 occlusion 处理弱。

3. **~1cm precision ceiling**：contact-rich task 到 sub-cm 精度需要 RL refinement 或 sim fine-tuning。这指向一个 hybrid pipeline：HumanEgo 做 coarse policy bootstrapping，RL 做 precision refinement。

4. **Scene sweep 前置要求**：triangulation 需要多视角，所以 demonstrator 必须先做 1-2 秒 head movement。这是 data collection protocol 的约束。

---

## 10. 在 VLA landscape 中的定位

把这个工作放到 2025-2026 的 VLA 格局里看：

- **Scaling 路线**：π0.5, RT-2, OpenVLA, EgoVLA, EgoScale——靠 internet-scale pretraining + robot post-training。compute heavy，但 generalist。
- **Co-training 路线**：EgoMimic, EgoBridge, ImMimic, Humanoid Policy ≈ Human Policy——human + robot joint training，仍需 per-task robot data。
- **Hierarchical 路线**：MimicPlay, XSkill, H2R——high-level plan from human video，low-level control from robot data。
- **Zero-shot 路线**（HumanEgo 所属）：EgoZero, PointPolicy, Track2Act, ZeroMimic, SPOT——完全无 robot data。

HumanEgo 在 zero-shot 路线里的 differentiator 是 **interaction-centric representation**：其他方法要么 point-based（丢失 hand-object 结构），要么 object-centric（丢失 hand approach 相位），要么 goal-conditioned（需要 explicit goal image）。HumanEgo 的 bet 是：**manipulation 的本质是 hand-object interaction 的几何演化，把这个演化显式编码进 representation，比用大模型隐式学习更 data-efficient**。

这个 bet 的哲学根源可以追溯到 UMD 的传统——Aloimonos lab 一直强调 action representation 和 linguistic/cognitive structure 的关系。ICT 本质上是把 manipulation 表示成一个 entity-relation graph 的 temporal evolution，这与 causal reasoning、affordance learning 都有连接。

---

## 11. 与你（Karpathy）观点的几个共振点

1. **Structured representation > raw pixels for control**：你在 Tesla AI Day 强调 vector space，HumanEgo 的 ICT 是同一哲学在 manipulation 的 instantiation。Fig 9 的 ablation（visual mismatch 消除只到 32.5%，加 ICT 跳到 85%）是这个哲学的硬证据。

2. **World model 作为 policy 的 prior**：你反复讲 "predict next token 是 intelligence 的核心"。HumanEgo 的三个 auxiliary objectives 本质上是让 encoder 学一个 lightweight world model——predict future object motion / 2D trace / latent state。这与 LeCun 的 JEPA、Ha-Schmidhuber 的 world model 传统都有呼应。

3. **Data efficiency 来自 signal density，不来自 model scale**：HumanEgo 用 6-layer transformer + 30 min data 达到 92.5%，没 scale model，而是 scale signal per demonstration。这与你讲 "neural net 是 brutal force 但 representation 选择仍然 matter" 一致。

4. **Flow matching 的选择**：你在 nanoGPT 系列里强调 generative model 的 simplicity。Flow matching 比 diffusion 简单（straight path, 20 step Euler），且足够 expressive capture multi-modality。这与 SD3 用 Rectified Flow 的趋势一致。

---

## 12. 几个可以深挖的方向

1. **ICT 的 graph structure 是否可以用 GNN 替代 transformer？** 当前 ICT 是 set-of-tokens + self-attention，但本质是完全图。GNN with edge features（relative pose）可能更 sample-efficient。

2. **Auxiliary objectives 能否扩展到 cross-task？** 当前是 per-task 训练。如果能 share encoder 跨 task，相当于 multi-task world model pretraining，可能进一步降 data requirement。

3. **Active perception**：当前 scene sweep 是 manual。如果 policy 能 active control camera（比如 head-mounted camera on robot），可以闭环改善 triangulation。

4. **与 VLA 的结合**：ICT 可以作为 VLA 的 action token interface。把 language grounding 加到 ICT 上，可能实现 language-conditioned zero-shot transfer。

5. **Sim2Real 的反向**：ICT 是 embodiment-agnostic，意味着可以在 sim 里 generate ICT labels 大量预训练，再 real fine-tune。这可能突破 30 min 的 ceiling。

---

## Web Links

- **Project page**: https://humanego-ai.github.io/
- **Code**: https://github.com/TX-Leo/HumanEgo
- **Flow Matching (Lipman et al.)**: https://arxiv.org/abs/2210.02747
- **Rectified Flow (Liu et al.)**: https://arxiv.org/abs/2209.03003
- **Project Aria**: https://arxiv.org/abs/2308.13561
- **6D Rotation Representation (Zhou et al.)**: https://arxiv.org/abs/1812.07035
- **SAM2**: https://arxiv.org/abs/2408.00714
- **LaMa Inpainting**: https://arxiv.org/abs/2109.07161
- **CoTracker3**: https://arxiv.org/abs/2403.24601
- **Grounding DINO**: https://arxiv.org/abs/2303.05499
- **Diffusion Policy (Chi et al.)**: https://diffusion-policy.cs.columbia.edu/
- **ACT (Zhao et al.)**: https://tonyzhaozh.github.io/aloha/
- **EgoZero**: https://arxiv.org/abs/2505.20290
- **Track2Act**: https://arxiv.org/abs/2405.01527
- **SPOT**: https://arxiv.org/abs/2411.00965
- **HaWoR (monocular hand reconstruction)**: https://arxiv.org/abs/2501.02973
- **WiLoR**: https://arxiv.org/abs/2409.12259
- **HaMeR**: https://arxiv.org/abs/2312.05253

---

总结一句 build intuition 的话：**HumanEgo 的核心 bet 是——manipulation 的本质是 hand-object interaction 的 SE(3) relative geometry 的 temporal evolution，把这个 geometry 显式编码进 29D token，配合 flow matching 的 generative policy 和三个 forward-dynamics auxiliary objectives，能在 30 min data 下学到 deployable policy 并 zero-shot 跨 embodiment**。这个 bet 的实验验证（Fig 9 的 7.5%→85% 跳跃）是整篇 paper 最有力的 single piece of evidence。
