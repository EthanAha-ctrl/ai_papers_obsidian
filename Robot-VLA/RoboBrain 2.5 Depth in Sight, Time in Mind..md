---
source_pdf: RoboBrain 2.5 Depth in Sight, Time in Mind..pdf
paper_sha256: b1593026b2c5e8ba23a8a1a3f36e283d520eeed6f5fa68fc43b98cb7d6b63c38
processed_at: '2026-08-12T00:19:47-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 RoboBrain 2.5

Andrej，咱们撇开 paper 的 academic framing，直接讲这帮人到底干了啥、为什么这么干、干得漂亮不漂亮。

---

## 一句话总结

BAAI 这帮人发现：**现在的 VLA 模型本质上是"嘴上功夫厉害"——能说会道，但手不行**。你让它讲"把杯子放到左边"，它能讲得头头是道，但真让它输出一个机器人能用的 3D 坐标，它就歇菜了。而且它跑完整个任务才知道成没成，中间走错了也不回头。

RoboBrain 2.5 干了两件事：**给 VLM 装上"3D 眼睛"和"进度感"**。

---

## 第一件事：3D 眼睛

### 问题在哪

你想想现在主流的 VLA model 怎么 output 空间信息的——它给你吐一个 2D pixel 坐标 $(x, y)$。但机器人手臂要的是啥？要的是 3D 空间里的点 $(X, Y, Z)$，而且要是 metric 单位（厘米、毫米），因为你要做碰撞检测、要算 clearance、要规划轨迹。

从 2D pixel 到 3D metric 中间隔着一层 camera geometry。传统做法要么让 VLM 自己隐式学这层映射（很难，数据不够），要么外挂一个 depth estimator（pipeline 一长，error 累积）。

### 他们的解法

非常 simple & elegant：**让 VLM 直接输出 $(u, v, d)$**，就是 image plane 坐标 + 绝对深度。

然后你拿 camera intrinsics（这是已知的，机器人上相机参数都标定好了）做个 back-projection：

$$
Z_c = d, \quad X_c = \frac{u - c_x}{f_x} \cdot d, \quad Y_c = \frac{v - c_y}{f_y} \cdot d
$$

其中 $f_x, f_y$ 是 focal length，$c_x, c_y$ 是 principal point，都是 camera 内参。$d$ 是 depth，$u, v$ 是 pixel 坐标。

**为什么这个解法聪明**：

VLM 已经在 image plane 上做 grounding 做得很好了（YOLO、GroundingDINO 这些 detector 都是 image plane 的），depth estimation 也有成熟工具（UniDepth V2 之类）。RoboBrain 2.5 把这两个 capability **decouple** 开，让 VLM 只管它擅长的 $(u, v)$，depth $d$ 单独 predict。这样 VLM 不用学 camera geometry 的 implicit representation，camera intrinsics 作为已知先验直接用。

**再说个更深的好处**：这个 $(u, v, d)$ 表示天然 support task hierarchy。去掉 $d$ 就是 2D grounding（跟 RoboPoint、RefSpatial 这些 2D dataset 兼容），只保留 start/end point 就是 3D referring，全保留就是 full 3D tracing。这意味着你可以拿 2D data 和 3D data 一起 co-training，multi-task learning 效果更好。

### 但是光有坐标还不够

你给 robot 一个目标点 $(X, Y, Z)$，它怎么过去？直接直线插值可能撞东西。所以 RoboBrain 2.5 让 VLM 直接 output **ordered keypoint sequence** $\tau = \{p_1, p_2, \ldots, p_T\}$，每个 $p_t = (u_t, v_t, d_t)$。

这个 sequence 就是 spatial plan——从 start point 一路 keypoint 到 end point，中间避开障碍。他们管这个叫 "3D Spatial Trace Generation"。

举个例子：指令是"用浇水壶从左到右浇花，壶嘴离每朵花 1-5 cm"。模型要：
1. 识别所有花，按从左到右排序（3D Spatial Referring）
2. 估计每朵花的物理高度，算出 1-5 cm 的 metric offset（3D Spatial Measuring）
3. 生成一条 keypoint trajectory，从第一朵花上方依次到每朵花上方，保持 1-5 cm 间距，中间不撞东西（3D Spatial Tracing）

这三个 sub-skill 形成一个 curriculum，从易到难。

### 数据怎么来的

这是工程上最重的部分。他们 build 了 1.74M samples 的 3D spatial reasoning dataset：
- 3D scanning 数据（CA-1M、ScanNet）：室内场景的 3D bounding box 和 occupancy map
- Manipulation 视频（AgiBot-Beta、DROID、RoboTwin 2.0）：真实和仿真的 tabletop manipulation

每个 sample 都用 Qwen3-VL 做 task decomposition，把 long-horizon task 拆成 subgoal，然后对每个 subgoal 标注 3D keypoint trace。数据清洗很狠——AgiBot-Beta 从 167K 砍到 59K，DROID 从 116K 砍到 24K，只留 camera pose valid、task flow coherent、trajectory clean 的。

---

## 第二件事：进度感

### 问题在哪

现在 VLA model 是 open-loop predictor：它吐出一串 action，然后 robot 执行，执行完才知道成没成。中间如果 slip 了、object 移位了、手抖了，模型完全不知道，继续往下执行，必然失败。

RL 里这叫 sparse reward problem。你想做 dense reward，传统做法要么 hand-craft reward function（很 tedious），要么用 success detector（还是 sparse）。

### 他们的解法

训练一个 vision-language model 当 **reward model / value function**。输入是 multi-view RGB，输出是一个标量，告诉你"当前 state 离 goal 还有多远"。

但这里有个 tricky 的数学问题：怎么设计这个 supervision signal？

#### Naive approach 的问题

最直觉的做法：regress progress delta $\Delta\Phi = \Phi(s_q) - \Phi(s_p)$，然后 iteratively 加起来重建 global progress。

问题：
1. 误差累积——每步预测都有 noise，加 $T$ 步后 noise 也放大 $T$ 倍
2. 越界——重建的 $\Phi^*$ 可能跑出 $[0, 1]$，这就破坏了 "progress" 的语义

#### Hop-based normalization

他们的核心 trick：**不 regress 绝对 delta，而是 regress 相对 delta**。

$$
\mathcal{H}(s_p, s_q) = \frac{\Phi(s_q) - \Phi(s_p)}{\Phi(s_M) - \Phi(s_p)} \quad \text{(progress case)}
$$

这个 hop $\mathcal{H}$ 的意思是："你这步走的距离，占剩下到 goal 的距离的多少比例"。

比如你已经走了 70%（$\Phi(s_p) = 0.7$），goal 是 1.0，你这一步走到 0.75，那 hop = (0.75 - 0.7) / (1.0 - 0.7) = 0.167。意思是你走完了剩下 30% 路的 16.7%。

regress 的情况对称处理：分子还是 delta，分母换成 "已经走过的距离" $\Phi(s_p) - \Phi(s_0)$。

#### 为什么这个 trick 牛

迭代重建时用 multiplicative update：

$$
\Phi^*(s_t) = \Phi^*(s_{t-1}) + \mathcal{H} \cdot [1 - \Phi^*(s_{t-1})] \quad (\mathcal{H} \geq 0)
$$

这其实是 convex combination：$\Phi^*(s_t) = \mathcal{H} + \Phi^*(s_{t-1})(1 - \mathcal{H})$，两项都非负，和为 1，所以结果天然在 $[0, 1]$ 里。

paper Appendix B 给了完整 proof，核心就是这个 multiplicative structure。naive additive update 缺这个 inductive bias，所以会越界。

**Intuition**：这就像你问"离 goal 还多远"，与其直接估绝对距离（容易飘），不如估"我这一步走了剩余路的比例"。比例天然 bounded，而且每一步都 re-normalize，error 不会无限累积。

#### Multi-Perspective Fusion

但单 perspective 还是有 bias：
- **Incremental**：递归加 hop，局部精细，但 long-horizon 会 drift
- **Forward-anchored**：每步都从 initial state 重新算 hop，稳定但早期 progress 小时 noise 大
- **Backward-anchored**：每步都从 goal state 反向算 hop，对接近 goal 的 state 敏感

三者 fusion：$\Phi^*(s_t) = \frac{1}{3}(\Phi_I^* + \Phi_F^* + \Phi_B^*)$。这个 ensemble 思路很简单但 effective，三个 estimator 的 bias 互相 cancel。

#### OOD 防御

RL explore 时会遇到 OOD state，reward model 在 OOD state 可能 hallucinate 高 reward，policy 就会 "reward hacking"——故意往 OOD state 跑去骗 reward。

他们的 trick：**用 forward 和 backward prediction 的 disagreement 当 uncertainty**。如果 model 熟悉这个 state，forward 和 backward 应该一致；如果 OOD，两者会 diverge。

$$
\Delta_{\text{norm}} = \frac{|\Phi_B^* - \Phi_F^*|}{\bar{\Phi}^* + \epsilon}
$$

然后 Gaussian kernel 转成 confidence weight $w_t = \exp(-\alpha \Delta_{\text{norm}}^2)$。confidence 低就 ignore update，保留上一个 state 的 estimate。这是个 semantic filter，防止 reward hacking。

---

## 训练怎么做的

两 stage training，思路是 curriculum：

**Stage 1（Foundational）**：8.3M samples，学 general perception + 2D grounding + planning + temporal comparison（只比较两帧谁先谁后，不 predict 具体 hop value）。

**Stage 2（Specific）**：4.1M samples，学 metric 3D tracing + dense value estimation（predict 具体 hop value）。

防 catastrophic forgetting：Stage 2 里 mix 15% Stage 1 data。

Architecture 是 Qwen3-VL 8B，full model trainable。AdamW，cosine schedule，batch size 1024，TP=2, PP=2，max seq length 16384。

有个 infra 亮点：在 **Moore Threads GPU**（国产非 NVIDIA）上做了完整训练，loss convergence gap 控制在 0.62% 以内，然后 checkpoint 迁移到 NVIDIA 做 eval。这说明 FlagScale 的 cross-accelerator 能力已经 production-ready。还有个 dynamic pre-allocated memory trick 处理 multi-modal long-sequence 的 memory fragmentation，挺实用的。

---

## 效果怎么样

### 3D Spatial Reasoning

最 striking 的数字是 **TraceSpatial Success Rate**：

| Model | Success Rate |
|-------|:---:|
| RoboBrain 2.5 (NV) | **44%** |
| RoboBrain 2.5 (MTT) | 36% |
| Qwen3-VL-8B | 6% |
| Gemini-3-Pro | 7% |
| GPT-5.2 | **0%** |

Success 要求 grasp + placement + collision-free 三者同时满足。GPT-5.2 直接 0%，说明 general VLM 根本做不了 metric-grounded 3D planning。RoboBrain 2.5 的 44% 是 SOTA 级别。

CrossPoint（cross-view point correspondence）上 RoboBrain 2.5 达到 75-76，而 Gemini-3-Pro 只有 38.6，Qwen3-VL 只有 28.4。这说明 RoboBrain 2.5 在 viewpoint-invariant 的点对应上有本质提升，这对 multi-view 3D reasoning 至关重要。

### Temporal Value Estimation

最 revealing 的 test 是 **time-reversal**：把 video 倒放，重新评估 model。如果 model 真的懂 progress 语义，倒放后应该给出反向的 progress。如果 model 只是 fit 了个 spurious feature（比如 frame index），倒放后就会露馅。

| Model | DROID VOC+ (正向) | DROID VOC- (倒放) |
|-------|:---:|:---:|
| RoboBrain 2.5 (MTT) | 93.67 | **89.26** |
| GPT-5.2 | 91.45 | 15.29 |
| Gemini-3-Pro | 90.57 | 44.15 |

GPT-5.2 正向 VOC 91.45 看着很高，但倒放只有 15.29——它根本不懂 progress，只是学了个 "frame 越往后 progress 越大" 的 spurious bias。RoboBrain 2.5 正反向都 89+，说明它真正理解了 task 的 semantic progress。

这个 time-reversal test 应该成为未来 reward model 评估的标配。它就像是 reward model 的 "adversarial example"——能区分 "真懂" 和 "假懂"。

---

## 最让我兴奋的几个点

### 1. (u,v,d) decoupling 的 design pattern

这其实是个 general principle：**不要让 neural network 学你已经知道的东西**。Camera geometry 是已知的，就别让 VLM 隐式学。把它 decouple 出来，用解析解处理。这种 "neural + symbolic" hybrid 在 embodied AI 里应该会更流行。

类似地，physics constraint（collision、kinematic limit）也不该让 VLM 隐式学，应该用 explicit constraint solver 处理。

### 2. Hop-based normalization 的数学美感

这个 multiplicative update 天然 bounded 的性质，让我想起 RL 里的 softmax policy iteration、信息论里的 entropy regularization——都是用数学结构 constrain 估计值的行为。这种 design 比 "regress + clip" 这种粗暴做法优雅得多。

### 3. Time-reversal test 是 reward model 的 "lie detector"

GPT-5.2 在 forward VOC 上 90+ 看着很牛，但倒放就露馅了。这说明现有 VLM 的 "progress understanding" 很多是 spurious correlation。RoboBrain 2.5 通过 hop-based supervision 学到了 time-symmetric representation，这个 test 应该推广到所有 reward model 评估。

### 4. 这本质上是 embodied AI 的 PRM

LLM RLHF 里有 Process Reward Model（PRM），给每个 reasoning step 打分。RoboBrain 2.5 做的其实是 **embodied version 的 PRM**——给每个 manipulation step 打分。这个范式如果 scale 起来，可能成为 embodied RL 的基础设施层，就像 PRM 之于 LLM reasoning。

### 5. Self-Evolving Data Engine 的未来

Paper 最后提到未来方向：用 dense value estimator 自己 filter/annotate 大规模 uncurated video。这就是 reward model 的 self-improvement 飞轮——model 越好，data curation 越准，data 越多 model 越好。这个 closed loop 如果跑起来，data 门槛就大幅降低了。

---

## 与其他工作的关联

- **π0 / π0.5 (Physical Intelligence)**：open-loop VLA 的 SOTA，但仍然是 open-loop，没有 dense feedback。RoboBrain 2.5 走的是 decoupled perception + value estimation 路线，可以 plug 进任何 RL pipeline。
- **Eureka (NVIDIA)**：用 LLM 写 reward function code，是 "symbolic reward"。RoboBrain 2.5 是 "neural reward"，更 flexible 但 less interpretable。
- **VLM as Value Learner (Ma et al.)**：思路类似，但 RoboBrain 2.5 的 hop formulation 更 robust。
- **RoboPoint (UW)**：2D spatial affordance 的 VLM，RoboBrain 2.5 在它基础上加了 depth 维度。
- **SpatialVLM (Stanford)**：之前给 VLM 加 spatial reasoning 的工作，但缺乏 metric grounding。
- **UniDepth V2**：universal monocular depth estimation，RoboBrain 2.5 用它生成 pseudo-3D scene graph。
- **SAM 2**：segment anything，RoboBrain 2.5 用它做 instance mask。
- **Code-as-Monitor**：BAAI 同期的 constraint-aware failure detection，跟 RoboBrain 2.5 互补。
- **Cambrian-1**：vision-centric MLLM，CV-Bench 的来源。

---

## 一句话直觉

RoboBrain 2.5 教 VLM 两件事：**"看准 3D 位置"和"感觉离目标多近"**。前者用 $(u, v, d)$ decoupling 避开学 camera geometry，后者用 hop-based normalization 保证 bounded dense reward。两个加起来，VLM 从 "会说的规划器" 变成 "能动手的执行器"。

参考链接：
- 项目主页: https://superrobobrain.github.io
- RoboBrain 2.0: https://arxiv.org/abs/2507.02029
- Robo-Dopamine: https://arxiv.org/abs/2512.23703
- TraceSpatial: https://arxiv.org/abs/2512.13660
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- π0.5: https://arxiv.org/abs/2504.16054
- Eureka: https://arxiv.org/abs/2310.12931
- RoboPoint: https://arxiv.org/abs/2406.10721
- UniDepth V2: https://arxiv.org/abs/2505.02059
- SAM 2: https://arxiv.org/abs/2408.00714
- Code-as-Monitor: https://arxiv.org/abs/2412.04455
- Cambrian-1: https://arxiv.org/abs/2408.07005

---

想深挖哪一块？我个人觉得 hop-based normalization 的数学结构最值得 follow up，这个 multiplicative update 能不能推广到 LLM reasoning 的 step-level confidence estimation？比如 self-consistency voting 时，不用 majority vote，而是用 hop-based weighted vote？这个 transfer 听起来挺自然的。

---

# RoboBrain 2.5: Depth in Sight, Time in Mind 深度解读

你好 Andrej！这篇 paper 由 BAAI RoboBrain Team 出品，是 RoboBrain 2.0 的延续版本，核心论点非常清晰：当前的 embodied AI foundation models 存在两大根本缺陷——**metric blindness**（空间维度只懂 2D pixel、缺乏绝对深度和 scale）以及 **open-loop prediction**（时间维度只做静态 sequence prediction、缺乏 dense feedback）。RoboBrain 2.5 通过两根支柱来填补这个 gap：Precise 3D Spatial Reasoning 和 Dense Temporal Value Estimation。下面我从 intuition 的角度一层层拆开。

项目主页：https://superrobobrain.github.io  
RoboBrain 2.0 tech report: https://arxiv.org/abs/2507.02029  
RoboBrain (CVPR 2025): https://arxiv.org/abs/2506.09671  
Robo-Dopamine (Dense Value Estimation 的基础): https://arxiv.org/abs/2512.23703  
TraceSpatial (3D Spatial Tracing 的基础): https://arxiv.org/abs/2512.13660  

---

## 1. 核心动机与设计哲学

### 1.1 为什么需要从 2D Grounding 升级到 3D Spatial Reasoning

传统 VLA 模型（如 RT-2、OpenVLA、π0）通常 output 2D pixel coordinates 或者 topological representations，这就直接导致一个工程灾难：你拿到一个 (x_pixel, y_pixel) 后，机器人手臂根本没法直接 actuate，因为手臂需要的是 6-DoF Cartesian pose 或者 joint angles。中间需要再做 depth estimation、camera projection、metric scale recovery，整个 pipeline 的 error 会层层累积。

RoboBrain 2.5 选择一个很优雅的 formulation：直接让 VLM 输出 **decoupled $(u, v, d)$ representation**，其中：
- $(u_t, v_t)$ 是 image-plane coordinates（pixel 坐标）
- $d_t$ 是 absolute depth（绝对深度，metric 单位，例如 cm）

然后通过已知的 camera intrinsics $K$（内参矩阵）做 back-projection 得到 3D camera-frame coordinates：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = d_t \cdot K^{-1} \begin{bmatrix} u_t \\ v_t \\ 1 \end{bmatrix}
$$

这里 $K^{-1}$ 是内参矩阵的逆，$d_t$ 充当 scale factor，因为 normalized image-plane 的 ray direction 是 $\tilde{p} = K^{-1}[u_t, v_t, 1]^T$，乘以 depth 才得到真实 3D 坐标。

**Intuition**：这种 decoupling 的好处在于：
1. VLM **不需要 implicit 学 camera geometry**，camera intrinsics 作为已知先验，直接拿过来用
2. 可以做 **task hierarchy**：去掉 $d$ 就是 2D grounding 任务，保留 start/end point 就是 3D spatial referring 任务，全部保留就是 full 3D spatial tracing
3. 这种 hierarchy 允许 co-training 2D dataset（如 RefSpatial、RoboPoint）和 3D dataset，提升 multi-task learning 效果

### 1.2 为什么需要 Dense Temporal Value Estimation

Embodied manipulation 的 long-horizon task 中，sparse success reward 是非常 inefficient 的，因为 agent 要等整个 episode 跑完才能拿到一个 0/1 信号。但 RL 真正需要的是 **dense reward signal**，告诉 agent 每一个 state transition 是 progress 还是 regress。

RoboBrain 2.5 训练一个 vision-language estimator，输入是 multi-view RGB observations，output 是一个 **hop value** $\mathcal{H} \in [-1, 1]$，描述当前 state 相对 goal 还剩多远。这个 hop 可以用作 general process reward，下游的 RL 直接拿它来训练 policy。

**关键设计**：hop-based normalization（公式 2）让 supervision signal 自然 bounded，并且数学上保证 iteratively reconstructed 的 global progress $\Phi^*(s) \in [0, 1]$，这是相对于 naive relative progress regression 最大的优势。

---

## 2. Precise 3D Spatial Reasoning 的三个 sub-capabilities

RoboBrain 2.5 把 3D spatial reasoning 拆成三个 curriculum skill：

### 2.1 3D Spatial Referring

任务：给一个 spatially constrained instruction（例如 "the third picture frame from the left on the piano"），模型要 resolve spatial relationships 并 localize 出目标 object 的 3D 坐标。

这部分数据来源：
- CA-1M（3D embodied videos）
- OpenImage + UniDepth V2（pseudo-3D scene graph）
- 802K samples from RefSpatial pipeline

### 2.2 3D Spatial Measuring

任务：理解 instruction 中的 absolute metric quantity，例如 "watering can hovering 1-5 cm above each flower"，模型需要 quantifies 每个花的物理 height 以及 1-5 cm 的 metric offset。

这部分是 metric-grounded QA，单位可以是 cm / inch / m。

### 2.3 3D Spatial Trace Generation

任务：从 monocular RGB 输入 + textual instruction 直接生成 ordered keypoint sequence：
$$
\tau = \{p_t\}_{t=1}^T, \quad p_t = (u_t, v_t, d_t)
$$
其中 $T$ 是 trace 长度，$t$ 是 keypoint index（不是 time step，是 trajectory waypoints）。

这个 trace 就是 spatial plan，引导 robot end-effector 从 start point 平滑过渡到 end point，并且要 collision-free。

**Intuition**：这其实就是把 high-level semantic instruction 通过 spatial planning 翻译成 actionable waypoints。比如 "pick up the orange object on the window sill, move it to the spot closest to the right wall on the sink's edge"，模型需要：
1. 在 3D scene graph 中 localize 橙色物体（referring）
2. 估计 sink 边缘到右墙的距离（measuring）
3. 生成从 start point 到 end point 的 collision-free 3D keypoint sequence（tracing）

数据来源：1.74M samples，包含 3D scanning（CA-1M、ScanNet）和 manipulation videos（AgiBot-Beta、DROID、RoboTwin 2.0）。

---

## 3. Dense Temporal Value Estimation 的数学核心

这是 paper 里 mathematically 最 elegant 的部分，我详细展开。

### 3.1 Hop-wise Progress Construction

给定一条 expert trajectory，先用 human-annotated multi-view keyframes $\{K_0, K_1, \ldots, K_N\}$ 把 trajectory 切成 N 个 sub-task segment，其中 $K_0$ 是 initial state，$K_N$ 是 success state，每个 $K_j$ 是一组同步 multi-view frames。

然后在每个 segment 内做 adaptive sampling。设 trajectory 总长度 $L$ 帧（per view），chunk size $C$ 控制采样密度，segment 内的中间点数：
$$
m = \left\lfloor \frac{1}{N} \left\lfloor \frac{L}{C} \right\rfloor \right\rfloor \tag{1}
$$
这里 $\lfloor \cdot \rfloor$ 是 floor operation，$1/N$ 是均匀分配到每个 segment，$L/C$ 是总采样数。

得到 state sequence $\mathcal{S} = \{s_0, s_1, \ldots, s_M\}$，ground-truth global progress 定义为：
$$
\Phi(s_i) = \frac{i}{M}
$$
其中 $i$ 是 state index，$M$ 是 total state count。

### 3.2 Hop-based Relative Progress Normalization（核心公式）

naive approach 是 regress 进度增量 $\Phi_\delta(s_p, s_q) = \Phi(s_q) - \Phi(s_p)$，但问题在于：
- **误差累积**：如果 iteratively 加这些 delta，error 会指数累积
- **越界**：reconstructed $\Phi^*(s)$ 可能跑出 $[0, 1]$ 区间，破坏 reward 的语义

RoboBrain 2.5 引入 hop label $\mathcal{H}(s_p, s_q)$：

$$
\mathcal{H}(s_p, s_q) = 
\begin{cases} 
\frac{\Phi(s_q) - \Phi(s_p)}{\Phi(s_M) - \Phi(s_p)} & \text{if } q \geq p \text{ (PROGRESS)} \\
\frac{\Phi(s_q) - \Phi(s_p)}{\Phi(s_p) - \Phi(s_0)} & \text{if } q < p \text{ (REGRESS)}
\end{cases} \tag{2}
$$

**变量解释**：
- $s_p$：BEFORE state
- $s_q$：AFTER state
- $\Phi(s_p), \Phi(s_q)$：分别是两个 state 的 global progress
- $\Phi(s_M)$：goal state 的 progress（=1）
- $\Phi(s_0)$：initial state 的 progress（=0）

**Intuition**：
- PROGRESS 情况下，hop = 实际 progress / 剩余到 goal 的距离，衡量 "你这步走了剩下路的多大比例"
- REGRESS 情况下，hop = 实际倒退 / 已经走过的距离，衡量 "你倒退了已经走过的多大比例"
- 两者都归一化到 $[-1, 1]$，分别用剩余距离和已走距离做分母

为什么这种 normalization 能保证 bounded？Appendix B 的数学证明给出答案。

### 3.3 Bounded Global Progress Proof（直觉理解）

Theorem：如果 $\Phi^*(s_0) = 0$，并且 hop $H \in [-1, 1]$，那么 iteratively 应用 update rule 后 $\Phi^*(s_t) \in [0, 1]$ 对所有 $t$ 成立。

Update rule（公式 12）：
$$
\Phi^*(s_t) = 
\begin{cases}
\Phi^*(s_{t-1}) + H \cdot [1 - \Phi^*(s_{t-1})] & \text{if } H \geq 0 \\
\Phi^*(s_{t-1}) + H \cdot \Phi^*(s_{t-1}) & \text{if } H < 0
\end{cases}
$$

设 $G = \Phi^*(s_{t-1}) \in [0, 1]$（归纳假设）。

**Case 1: $H \in [0, 1]$（progress）**
$$
\Phi^*(s_t) = G + H(1-G) = H + G(1-H)
$$
这是一个 convex combination：$H$ 和 $G(1-H)$ 都非负，所以 $\Phi^*(s_t) \geq 0$。上界方面，代入 $G=1$：
$$
\Phi^*(s_t) \leq H + 1 \cdot (1-H) = 1
$$
所以 $\Phi^*(s_t) \in [0, 1]$。

**Case 2: $H \in [-1, 0)$（regress）**
$$
\Phi^*(s_t) = G(1+H)
$$
由于 $H \in [-1, 0)$，所以 $1+H \in [0, 1)$，因此 $G(1+H) \in [0, 1]$。

**核心 insight**：这个 hop update rule 本质上是 **multiplicative interpolation**，而 multiplicative 结构天然保持 bounded。naive additive update 缺乏这个 inductive bias，所以会越界。

### 3.4 Multi-Perspective Progress Fusion

单 perspective 估计都有 bias：
- **Incremental**（公式 4, 5）：$\Phi_I^*(s_t) = \Phi^*(s_{t-1}) + \Delta\Phi_{t-1,t}^*$，局部精细但累积误差
- **Forward-anchored**（公式 6）：$\Phi_F^*(s_t) = \mathcal{H}^*(s_{\text{init}}, s_t)$，对初始 state anchor，稳定但对 goal 不敏感
- **Backward-anchored**（公式 7）：$\Phi_B^*(s_t) = 1 + \mathcal{H}^*(s_{\text{goal}}, s_t)$，对 goal anchor，在完成附近敏感但早期 noise 大

公式 4 的细节：
$$
\Delta\Phi_{t-1,t}^* = 
\begin{cases}
[1 - \Phi^*(s_{t-1})] \cdot \mathcal{H}^* & \text{if } \mathcal{H}^* \geq 0 \\
\Phi^*(s_{t-1}) \cdot \mathcal{H}^* & \text{if } \mathcal{H}^* < 0
\end{cases}
$$
这里 $[1-\Phi^*(s_{t-1})]$ 是 remaining progress to goal，乘以 hop 得到实际 progress delta。

融合策略（公式 8）：
$$
\Phi^*(s_t) = \frac{1}{3}\left( \Phi_I^*(s_t) + \Phi_F^*(s_t) + \Phi_B^*(s_t) \right)
$$

**Intuition**：这就像 ensemble of three estimators，每个有不同的 inductive bias。Incremental 捕捉 local dynamics，Forward 锚定起点，Backward 锚定终点。三者平均可以 cancel 各自的 bias。

### 3.5 Bi-directional Consistency Checking（OOD 防御）

在线 RL 中，policy 会 explore unseen regions，model 在 OOD state 可能输出 spurious high signal，导致 **reward hacking**。RoboBrain 2.5 用 forward 和 backward prediction 的 **discrepancy** 作为 uncertainty proxy：

$$
\Delta_{\text{norm}}(s_t) = \frac{|\Phi_B^*(s_t) - \Phi_F^*(s_t)|}{\bar{\Phi}^*(s_t) + \epsilon} \tag{9}
$$

其中 $\bar{\Phi}^*(s_t) = (\Phi_F^* + \Phi_B^*)/2$，$\epsilon$ 是数值稳定常数。除以 $\bar{\Phi}^*$ 让 discrepancy 在早期（progress 小）时被 weighted 更重，因为早期 precise guidance 更重要。

confidence weight（公式 10）：
$$
w_t = \exp\left(-\alpha \cdot (\Delta_{\text{norm}}(s_t))^2\right)
$$
$\alpha$ 是 sensitivity hyper-parameter。Gaussian kernel 形式让 weight 在 discrepancy=0 时为 1，discrepancy 大时快速衰减到 0。

Conservative update（公式 11）：
$$
\Phi^*(s_t) = \Phi^*(s_{t-1}) + \frac{w_t}{2} \cdot \left( \bar{\Phi}^*(s_t) - \Phi^*(s_{t-1}) + \Delta\Phi_{t-1,t}^\star \right)
$$

当 $w_t \to 0$（OOD 不确定），update 被忽略，保留 $\Phi^*(s_{t-1})$；当 $w_t \to 1$（in-distribution 高 confidence），完全信任 estimate。这是一个 **semantic filter**，在 RL 中防止 reward hacking。

---

## 4. Training Data 与 Training Strategy

### 4.1 数据组成（12.4M samples）

| 类别 | 数据量 | 用途 |
|------|--------|------|
| General MLLM Data | 2.83M | 通用 visual perception |
| Spatial Reasoning Data | ~5M | 2D grounding + 3D spatial reasoning |
| Temporal Prediction Data | ~4M | Planning + Dense value estimation |

General MLLM Data 来自 Honey-Data-1M（https://arxiv.org/abs/2510.13795）和 LLaVA-OneVision-1.5（https://arxiv.org/abs/2509.23661），做了 dedup 和 sample packing（sequence length 集中在 2048-8192 token）。

Spatial Reasoning Data 涵盖：
- Visual Grounding (LVIS, 152K images, 86K conversations)
- Object Pointing (Pixmo-Points, 190K QA pairs, 64K images)
- Affordance (PACO-LVIS, 561K QA pairs + RoboPoint 320K QA pairs)
- Spatial Understanding (826K samples, 31 spatial concepts)
- Spatial Referring (802K samples from RefSpatial)
- 3D Spatial Reasoning (1.74M samples, RoboBrain 2.5 new feature)

Temporal Prediction Data 涵盖：
- Ego-View Planning (EgoPlan-IT 50K samples)
- ShareRobot Planning (1M QA pairs, 51K instances)
- AgiBot Planning (9,148 QA pairs)
- Multi-Robot Planning (44,142 samples, 1,659 task types)
- Close-Loop Interaction (OTA trajectories, 120 indoor environments)
- **Dense Value Estimation** (35M raw → 3.5M after down-sampling, RoboBrain 2.5 new feature)

Dense Value Estimation 的数据组成：
- Real-World robot data (~60%)：AGIBot-World、DROID、RoboBrain-X
- Simulation data (~13%)：LIBERO、RoboCasa、RoboTwin
- Human-Centric data (~26%)：EgoDex（https://arxiv.org/abs/2505.11709）

跨 embodiment 设计（从单臂 Franka Emika Panda 到 bimanual humanoid AGIBot-A2D）防止对特定 kinematics overfitting，让 model 学到 **embodiment-invariant** 的 progress signal。

### 4.2 Two-stage Training

| | Stage 1: Foundational | Stage 2: Specific |
|---|---|---|
| Data | 8.3M | 4.1M |
| 内容 | General + 2D Spatial + Planning + Temporal Comparison | Metric 3D Tracing + Dense Value Estimation |
| Loss | Next-token prediction | Next-token prediction |
| LR | ViT 1e-6, LM 1e-5 | ViT 1e-6, LM 1e-5 |
| Optimizer | AdamW, cosine schedule, warmup 0.01 | 同左 |
| Global batch | 1024 | 1024 |
| TP × PP | 2 × 2 | 2 × 2 |
| Max seq | 16384 | 16384 |
| GPU | 64 × 8 (NV) / 128 × 8 (MTT) | 同左 |

**关键 anti-forgetting 策略**：Stage 2 中 random sample 15% 的 Stage-1 data mix 进去，防止 catastrophic forgetting of general capabilities。

---

## 5. Infrastructure

### 5.1 Hybrid Parallelism

Qwen3-VL 是 VLM 异构架构：ViT encoder 轻量但 LLM decoder 大。计算 cost 在 visual-heavy training 中 ViT 也不可忽视。

FlagScale（https://github.com/FlagOpen/FlagScale）用 **uneven pipeline parallelism**：把 ViT 放在 pipeline 前端，第一个 pipeline stage 减少分配的 LLM layer 数，平衡 compute load，减少 bubble。

### 5.2 Dynamic Pre-Allocated Memory

multi-modal long-sequence training 中 sequence length 变化大，PyTorch 默认 CUDA caching allocator 容易 fragment 甚至 OOM。常见 workaround 是每次 iter 前 `torch.cuda.empty_cache()`，但破坏 memory reuse。

RoboBrain 2.5 的 dual-stream dynamic padding 策略：
1. Training 前统计 max sequence length
2. 第一次 iter 全部 pad 到 max length，做一次 memory pre-allocation
3. 后续 iter 复用 pre-allocated memory
4. 只有当 visual token 超过当前 max 时，才 trigger full cache cleanup + re-pad

这个 trick 在 multi-modal long-context training 中应该是个比较通用的优化。

### 5.3 Cross-Accelerator Training

亮点：在 **Moore Threads GPU**（国产非 NVIDIA 加速器）上做了完整训练，loss 收敛 gap 控制在 0.62% 以内，然后 checkpoint 无缝迁移到 NVIDIA 平台做 evaluation。这说明 FlagOS/FlagScale 的 cross-accelerator 能力已经达到 production-ready 水平。

---

## 6. Evaluation 结果分析

### 6.1 2D Spatial Reasoning（Table 2）

| Benchmark | RoboBrain 2.5 (NV) | RoboBrain 2.5 (MTT) | Qwen3-VL-8B | Gemini-3-Pro | GPT-5.2 |
|-----------|:---:|:---:|:---:|:---:|:---:|
| CV-Bench | 94.58 | 93.90 | 92.89 | 92.00 | 86.84 |
| CrossPoint | 75.40 | 76.30 | 28.40 | 38.60 | 33.00 |
| RoboSpatial | 73.03 | 73.00 | 66.90 | 57.96 | 43.78 |
| RefSpatial | 60.50 | 59.00 | 54.20 | 65.50 | 15.00 |
| EmbSpatial | 75.58 | 76.92 | 78.50 | 76.62 | 68.02 |
| AVG | 75.82 | 75.82 | 64.18 | 66.14 | 49.33 |

**关键观察**：CrossPoint（cross-view point correspondence）上 RoboBrain 2.5 达到 75-76，远超所有 baseline（最高 Gemini 38.6）。这是一个 striking result，说明 RoboBrain 2.5 在 viewpoint-invariant 的 point correspondence 上有本质提升。这对 multi-view 3D reasoning 至关重要。

### 6.2 3D Spatial Reasoning（Table 3）

| Benchmark | RoboBrain 2.5 (NV) | Gemini-3-Pro | Qwen3-VL-8B |
|-----------|:---:|:---:|:---:|
| MSMU | 64.17 | 59.44 | 43.48 |
| Q-Spatial | 73.53 | 81.37 | 70.74 |
| TraceSpatial-3D Start | **83** | 19 | 30 |
| TraceSpatial-3D End | **63** | 25 | 20 |
| TraceSpatial-Success | **44** | 7 | 6 |
| VABench-V (RMSE↓) | 0.1281 | 0.1705 | 0.1979 |
| ShareRobot-T (RMSE↓) | 0.1164 | 0.1899 | - |

**关键观察**：TraceSpatial Success Rate 上 RoboBrain 2.5 (NV) 达到 44%，而 Gemini-3-Pro 只有 7%，GPT-5.2 是 0%。这是巨大的 gap，说明 RoboBrain 2.5 的 metric-grounded 3D tracing 能力是 SOTA 级别的。Success 要求 grasp + placement + collision-free 三者同时满足。

### 6.3 Temporal Value Estimation（Table 4）

这个 benchmark 用 VOC (Video-Order Correlation) 评估，并且做了 **time-reversal test**：把 video 倒过来播放，重新评估 model，如果 model 真的理解 progress 而非 spurious bias，reverse VOC 应该也很高。

| Dataset | RoboBrain 2.5 (NV) | RoboBrain 2.5 (MTT) | Gemini-3-Pro | GPT-5.2 |
|---------|:---:|:---:|:---:|:---:|
| AgiBot (VOC+/VOC-) | 88.58/87.36 | 87.48/87.36 | 81.36/58.70 | 90.02/15.91 |
| DROID | 90.82/90.07 | 93.67/89.26 | 90.57/44.15 | 91.45/15.29 |
| Galaxea | 93.38/95.79 | 94.58/94.54 | 88.86/35.34 | 88.76/10.03 |
| EgoDex | 80.67/84.99 | 80.67/81.12 | 80.48/50.15 | 78.12/22.79 |
| LIBERO | 98.97/98.94 | 98.88/98.91 | 98.42/76.31 | 96.97/19.19 |
| RoboCasa | 98.47/98.75 | 98.54/99.58 | 67.89/34.28 | 77.91/10.71 |

**核心 insight**：GPT-5.2 在 forward VOC 上很高（90+），但 reverse VOC 只有 10-22，说明它学到了 "假" progress signal，可能是 frame count 之类的 spurious feature。RoboBrain 2.5 的 forward 和 reverse 都达到 87+ 甚至 98+，证明它真正理解 task progress 的语义，而不是 overfit 视觉捷径。

这个 time-reversal test 应该是评估 progress model 的 gold standard，未来 RL reward model 评估应该都做这个 test。

---

## 7. Real-World RL Rollout（Figure 13）

Paper 在 Appendix A.2 给了一个 striking 的定性例子：Insert Block task 中，policy 训练 20 分钟达到 95% success rate。Rollout 中 human 手动挪走 target slot，robot 错过目标后：

(a) Human 干预移动 target  
(b) Robot 错过目标，inset 显示 **Progress 曲线急剧 drop**（红色 dot 标注）  
(c) Policy 反应，调整 end-effector  
(d) 重新对准 target slot 上方  
(e) 精确对齐  
(f) 成功插入，Progress 达到 peak  

这个例子直观展示了 dense reward signal 如何 enable **closed-loop recovery**——sparse reward 在这种 perturbation 场景下根本无法 recover。

---

## 8. 个人 Intuition 与 Takeaway

### 8.1 (u,v,d) Representation 的深刻意义

这个 decoupling 其实是把 3D prediction 问题 **projection 到 image-plane + scalar depth** 的 latent space。本质上，VLM 已经在 image plane 上做 grounding 做得很好了，depth estimation 也有大量 pretrained model（UniDepth V2 等），RoboBrain 2.5 把这两个 capability 复合起来，避免了让 LLM 学习 camera geometry 的 implicit representation。

这种 design pattern 在 embodied AI 里应该会越来越流行：**不要让 LLM 学 implicit geometry，而是用已知先验把 geometry 解析地引入**。

### 8.2 Hop-based Normalization 的更广意义

公式 2 的 hop formulation 本质上是 **relative-to-remaining** normalization。这种思想在 RL 里其实早就有了（例如 advantage function $A = Q - V$），但 RoboBrain 2.5 把它用到 progress estimation 上，结合 multiplicative update rule 实现了 **bounded iterative reconstruction**。

这个 trick 可以推广到任何需要 dense、bounded、iteratively refinable 的 signal estimation 场景，比如：
- LLM token-level confidence estimation
- Multi-step reasoning 的 progress tracking
- Agent task completion probability estimation

### 8.3 Multi-Perspective Fusion 像 Transformer 的 Multi-Head

Forward / Backward / Incremental 三个 perspective 各自捕捉不同时间尺度的 progress signal，类似 multi-head attention 每个 head 学不同的 representation sub-space。Fusion 通过 average 而非 learned weight，简单但有效。Bi-directional consistency 进一步用 forward vs backward 的 disagreement 作为 OOD detector，这是个 elegant 的 unsupervised uncertainty estimation trick。

### 8.4 Time-Reversal Test 是关键 Insight

GPT-5.2 forward VOC 高但 reverse VOC 低，说明它可能学到了 frame index 之类的 position bias 而非真正的 progress semantics。RoboBrain 2.5 通过 hop-based supervision 和 multi-view fusion 学到了 **time-symmetric** 的 progress representation。这个 test 应该成为未来 reward model 评估的标准。

### 8.5 与 Open-Loop Action Prediction 的对比

当前主流 VLA（OpenVLA、π0、Gemini Robotics）都是 open-loop action prediction，RoboBrain 2.5 走的是另一条路：**decouple perception/planning from action execution**，把 value estimation 作为 separate module。这种 decoupling 让 RoboBrain 2.5 可以 plug 进任何 RL pipeline 作为 general reward model，灵活性极高。

### 8.6 联想到的其他工作

- **π0.5（Physical Intelligence）**：https://arxiv.org/abs/2504.16054，open-world VLA，但仍然是 open-loop
- **Eureka (Ma et al.)**：https://arxiv.org/abs/2310.12931，用 LLM 写 reward function code，但 RoboBrain 2.5 直接做 reward function
- **VLM as Value Learner (Ma et al.)**：https://arxiv.org/abs/2402.06194，类似思路
- **SAM 2**：用于 segmentation，RoboBrain 2.5 用它做 instance mask
- **UniDepth V2**：https://arxiv.org/abs/2505.02059，universal monocular depth estimation，RoboBrain 2.5 用它生成 pseudo-3D scene graph
- **Embodied-Reasoner**：https://arxiv.org/abs/2503.21696，OTA trajectory 思路
- **Code-as-Monitor**：https://arxiv.org/abs/2412.04455，BAAI 同期的 constraint-aware failure detection 工作
- **SpatialVLM (Chen et al.)**：https://arxiv.org/abs/2401.12168，先前的 spatial reasoning VLM
- **RoboPoint**：https://arxiv.org/abs/2406.10721，spatial affordance prediction 的 VLM
- **Cambrian-1**：https://arxiv.org/abs/2408.07005，vision-centric MLLM，CV-Bench 来源

### 8.7 未来方向（paper Section 7 提到）

1. **Unified Generation and Understanding**：合并 image/video prediction（next-stage prediction）做 world model
2. **Mobile Manipulation + Humanoid 部署**：用 3D spatial reasoning 实现 training-free manipulation generalization
3. **Scalable Model Family**：分 Instruction (fast) 和 Thinking (slow) 两个版本，类似 System 1 / System 2 cognitive architecture
4. **Self-Evolving Data Engine**：用 dense value estimator 自己 filter/annotate 大规模 uncurated video，闭环 self-improvement

这个 self-evolving engine 思路非常有意思，相当于把 reward model 用作 data curation 的 verifier，类似 RLHF 里 reward model 的 self-improvement 飞轮，但应用在 embodied data 上。

---

## 9. Summary

RoboBrain 2.5 是一个 **spatially grounded + temporally aware** 的 embodied foundation model，工程上做了大量 data curation 工作（12.4M samples），算法上有两个核心 contribution：

1. **Decoupled (u, v, d) representation** 让 VLM 避免学习 implicit camera geometry，同时 support hierarchical task learning（2D referring → 3D referring → 3D tracing）
2. **Hop-based normalization + multi-perspective fusion + bi-directional consistency** 共同实现了 bounded、robust、OOD-aware 的 dense temporal value estimation

从 scaling perspective 看，这工作实际上是在 build 一个 **general process reward model (PRM)** for embodied AI，类似于 LLM RLHF 中 PRM 的角色，但用的是 vision-language 信号。这个范式如果 scale 起来，可能成为 embodied RL 的基础设施层。

参考链接汇总：
- 项目主页：https://superrobobrain.github.io
- RoboBrain 2.0：https://arxiv.org/abs/2507.02029
- Robo-Dopamine：https://arxiv.org/abs/2512.23703
- TraceSpatial：https://arxiv.org/abs/2512.13660
- Qwen3-VL：https://arxiv.org/abs/2511.21631
- FlagScale：https://github.com/FlagOpen/FlagScale
- RoboBrain GitHub：https://github.com/FlagOpen/RoboBrain-X0

如果你想做 follow-up research，我最看好的方向是把 hop-based formulation 推广到 **token-level / sub-step-level** 的 reasoning progress estimation，让 LLM 在 multi-step reasoning 时也能有 dense feedback signal。这个 transfer 应该是非常自然的。
