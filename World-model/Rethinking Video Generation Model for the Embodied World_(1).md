---
source_pdf: Rethinking Video Generation Model for the Embodied World.pdf
paper_sha256: 6a0d0292b0384fad23d90f24bd47af2a8dc729a2de50fa88c2193987702725f0
processed_at: '2026-08-11T23:44:33-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 的核心直觉非常直白：**现在的 video generation models 虽然生成电影画面很惊艳，但它们根本不懂物理世界，如果直接拿来做 robot 的训练数据或者模拟器，会死得很惨。**

为了把这个问题讲透，作者做了两件事：第一，造了一个能戳破这些 model “物理造假”的测试集 RBench；第二，搞了一个专门教 model 物理常识的超大数据集 RoVid-X。

下面我用更接地气的方式给你拆解这篇 paper 的技术细节和直觉。

---

### 1. 为什么我们需要重新审视 Video Model？

目前 video generation 领域非常火爆，Sora、Wan、HunyuanVideo 等模型生成的画面极其逼真。大家一开始幻想：既然 Sora 懂物理规律，我们直接用它给 robot 生成动作视频，然后用 Inverse Dynamics Models (IDM) 提取出 action，不就可以实现无限数据了吗？

但是如果你真的去拿这些模型生成 robot 操作视频，你会发现全是破绽。现有的 benchmark（比如 VBench）只盯着画面顺不顺滑、清不清晰，完全测不出这些 robot 专属的幻觉。

常见的几类离谱失败模式包括：
- **穿透与悬浮**: robot 的机械臂直接穿过桌子，或者物体悬在半空。
- **非接触吸附**: gripper 还没碰到杯子，杯子就跟着 gripper 飞走了。
- **结构变形**: 单臂 robot 生成着生成着，突然长出了第二只手，或者 quadruped robot 变成了 humanoid。
- **关键动作缺失**: prompt 要求“开冰箱拿苹果关冰箱”，模型只生成了“开冰箱”就卡住了，任务根本没做完。

这篇 paper 就是为了把这些“皇帝的新装”给揭露出来，并给出解决方案。

---

### 2. RBench：专门测机器人的“照妖镜”

RBench 包含 650 个 image-text pairs，横跨 5 个 task domains（Common Manipulation, Long-horizon Planning, Multi-entity Collaboration, Spatial Relationship, Visual Reasoning）和 4 种 embodiment（Dual-arm, Humanoid, Single-arm, Quadruped）。

它的核心亮点在于抛弃了纯像素层面的 metric，设计了 5 个细粒度的 Automatic Metrics。这里最精妙的是 **Motion Amplitude Score (MAS)**，专门用来抓那种“画面很顺滑，但 robot 根本没动”的摆烂情况。

**MAS 公式拆解与直觉：**

首先要算出画面里 robot 的平均位移：
$$ \bar{D}_t = \frac{1}{K} \sum_{k=1}^{K} \|\mathbf{p}_{t,k} - \mathbf{p}_{t-1,k}\|_2 $$
变量含义：
- $\bar{D}_t$：第 $t$ 帧的平均位移
- $K$：在 robot mask 内部 tracking 的关键点总数
- $\mathbf{p}_{t,k}$：第 $k$ 个点在第 $t$ 帧的 2D 坐标
- $\|\cdot\|_2$：欧式距离

这里有个致命问题：相机晃动也会导致 $\bar{D}_t$ 变大。为了让 metric 只反映 robot 的主动运动，作者搞了个 **soft-zero strategy** 减去背景运动：
$$ \hat{D}_t = \begin{cases} \tilde{D}_t - \tilde{D}_t^{bg}, & \tilde{D}_t > \tilde{D}_t^{bg} \\ \tilde{D}_t, & \tilde{D}_t \leq \tilde{D}_t^{bg} \end{cases} $$
变量含义：
- $\hat{D}_t$：剥离了相机运动后的真实位移
- $\tilde{D}_t$：归一化（除以视频对角线长度）后的 robot 位移
- $\tilde{D}_t^{bg}$：背景区域的归一化运动幅度

直觉：如果 robot 看着动了 ($\tilde{D}_t$)，但其实背景动得更快 ($\tilde{D}_t^{bg}$)，说明 robot 其实没动，是镜头在摇。此时直接相减会出现负数，但作者保留了小的 residual value，提高了对 tracking noise 的 robustness。

最后算全局平均：
$$ \text{MAS} = \frac{1}{T} \sum_{t=1}^{T} \min(\hat{D}_t, 1) $$
变量含义：
- $T$：总帧数
- $\min(\hat{D}_t, 1)$：clip 操作，防止极端值干扰。

**Task Completion 的打分逻辑也很有意思**。对于 Visual Reasoning 任务，作者用 MLLM 生成一个 Question Chain，然后看视频回答对了多少题：
$$ \text{Score} = 5 \times \frac{\text{completed questions}}{\text{total questions}} $$
直觉：把“任务做没做对”从非黑即白的 0/1 判断，变成了 partial credit。如果 prompt 要求“挑出蓝色的书放进篮子”，模型挑对了颜色但没放进篮子，依然有部分得分。这比传统的 binary metric 更符合 robotic task 的 hierarchical 结构。

---

### 3. RoVid-X：给 Video Model 喂的“物理补剂”

光有测试集没用，模型表现差是因为没吃过好的物理数据。作者搞了个 4 million clips 的超大数据集 RoVid-X。

它的四阶段 pipeline 很严谨：
1. **Collection**: 20多个开源数据集加网络视频，用 GPT-5 过滤。
2. **Quality Filtering**: Scene segmentation 检测去除非 robot 画面，再用打分系统过滤糊片。
3. **Segmentation & Captioning**: 用 Seed1.5-VL 切片打标签，标准化动作描述（Subject + Object + Operation）。
4. **Physical Property Annotation**: 这是最关键的一步！用 FlashVSR 做超分，用 AllTracker 标注 Optical Flow，用 Video Depth Anything 标注 Depth Maps。

**为什么加 Depth 和 Optical Flow 这么重要？**
Intuition：传统的 video diffusion model 只学习 RGB 像素间的统计规律。它看到机械臂靠近杯子，杯子就跟着动，它就以为“靠近=跟着动”，完全不理解是因为“接触+摩擦力”。加入了 Depth Map，模型就能理解 3D 空间前后关系；加入了 Optical Flow，模型就能精确学习接触瞬间和运动轨迹的物理对应关系。这是向 Physically Grounded Representation 迈进的关键一步。

数据集效果验证表也很能说明问题：
| Model | Spatial. | Total |
|-------|----------|-------|
| Wan2.2_5B | 0.313 | 0.380 |
| Wan2.2_5B+Ours | 0.403 | 0.439 |

使用 RoVid-X 微调后，Spatial Relationship 任务分数从 0.313 暴涨到 0.403。这直接证明了 Depth annotation 对模型 spatial reasoning 能力的提升是立竿见影的。

---

### 4. 25个模型大评测：反直觉的结论

作者测了 25 个模型，得出了几个对整个 AI 圈都极具启发性的结论：

#### Insight A: Sora 的“翻车”与 Media-Simulation Gap
Sora v2 Pro 只排在第 17 名，平均分 0.362。Sora v1 更是排第 22 名，只有 0.266。
Intuition：Sora 被训练得太追求“电影级转场”和“视觉顺滑”了。它极度缺乏对物理接触和 rigid body constraints 的理解。这说明针对 media consumption 优化的 model，在 embodied AI 领域存在严重的 domain gap。能做电影导演，做不了物理世界模拟器。

#### Insight B: Scaling Law 在物理智能上依然成立
Wan 系列从 2.1 到 2.6，分数一路狂飙（0.399 -> 0.507 -> 0.570 -> 0.607）。Seedance 从 1.0 到 1.5 pro，排名从第 6 升到第 2。这说明只要持续迭代和数据喂养，video model 对物理规律的理解是可以被“涌现”出来的。

#### Insight C: 精细操作比粗运动难得多
看 embodiment 维度的分数：
- Quadruped（四足）: Wan 2.6 得分 0.723
- Single arm（单臂）: Wan 2.6 得分 0.666

Intuition：四足走路其实就是周期性的 rhythmic patterns，很容易学；但单臂抓取涉及极其复杂的 contact dynamics 和 fine-grained force control。目前的 video generator 在生成像素时，根本没有力反馈的约束，所以碰上精细操作就拉胯。这也是未来为什么需要 differentiable physics simulation 或 hybrid model 的原因。

#### Insight D: 专有小模型是死路一条
像 UnifoLM-WMA-0 这种针对特定 robot 微调的模型，分数只有 0.123，垫底。Vidar 也只有 0.206。
Intuition：在特定实体上 fine-tune 虽然能获得 control precision，但完全失去了 large-scale pretraining 带来的“World Knowledge”。没有 world knowledge 的先验，模型连什么是桌子、什么是水杯、水杯掉下来会碎都不知道，怎么可能生成合理的物理视频？

---

### 5. 对未来的联想与启发

这篇 paper 其实画出了下一代 World Model 的蓝图。

目前 video diffusion model 本质上是在高维像素空间做曲线拟合。即使给了 Depth 和 Flow，它依然是被动拟合。未来的路径可能有几条：

1. **Video + IDM 的闭环控制**: 论文未来方向提到了 Inverse Dynamics Models (IDM)。目前的 video model 只能看不能动。如果把生成的 video 喂给 IDM，提取出 latent action 或者离散的 joint torques，再丢进 MuJoCo 或者 Isaac Sim 里验证，如果跑不通就反向惩罚 video model。这类似于 Reinforcement Learning 里的 RLHF，只不过这里是 Physics Feedback。

2. **3D-aware DiT Architecture**: 目前大家都在用 2D 的 DiT (Diffusion Transformer)。如果要真正理解物理，可能需要把 3D Gaussian Splatting 或者 Neural Radiance Fields (NeRF) 的表征直接嵌进 transformer 的 token 里。比如在 RoVid-X 里标的 Depth，不仅作为 condition，还要作为 latent state 参与去噪过程。

3. **Latent Action Pretraining**: 像 LAPA (Latent Action Pretraining from Videos) 这种工作指出，可以在预训练阶段强迫 model 预测下一帧的 latent action，这会逼迫 video model 理解因果关系。RoVid-X 这种带细粒度操作标签的数据集，简直是做 Latent Action Pretraining 的完美养料。

总而言之，这篇 paper 给沉迷于“刷分 VBench”的 video generation 社区泼了一盆冷水，也指出了一条明路：想要实现 AGI，必须跨过 Physical AI 这道坎，而 RBench 和 RoVid-X 就是通关的装备。

**参考链接:**
- Project Page: https://dagroup-pku.github.io/ReVidgen.github.io/
- GitHub Repo: https://github.com/DAGroup-PKU/ReVidgen/
- HuggingFace Dataset: https://huggingface.co/datasets/DAGroup-PKU/RoVid-X/
- VBench (对比基准): https://vchitect.github.io/VBench-project/
- Video Depth Anything (数据标注工具): https://github.com/DepthAnything/Video-Depth-Anything
- DreamGen (Robotic Video 模型): https://arxiv.org/abs/2505.12705
- LAPA (Latent Action 预训练): https://openreview.net/forum?id=rk4l54k16r

---

# Rethinking Video Generation Model for the Embodied World - 深度解读

## 1. Paper 核心动机与定位

这篇 paper 来自 Peking University 与 ByteDance Seed,核心 addressing 的问题是:**当前的 video generation models(如 Sora、Wan、HunyuanVideo 等)虽然在 cinematic content 上表现惊艳,但在 embodied AI 场景下生成 robotic 行为时存在严重的物理与任务一致性问题**。作者提出两个核心 contribution:

1. **RBench**:第一个专门为 robotic video generation 设计的 comprehensive benchmark,包含 650 个 image-text pairs,覆盖 5 个 task domains 和 4 种 embodiment types
2. **RoVid-X**:4 million clips 的大规模 robotic video dataset,通过 4-stage pipeline 构造,旨在为训练 physically-grounded video foundation models 提供数据支撑

paper 的关键 insight 在于:video generation models 正在从 **"visual fidelity" paradigm** 向 **"physical intelligence" paradigm** 转移,而现有的 evaluation framework(如 VBench [46])主要关注 perceptual quality,无法 capture robotic scenarios 中的 task-level correctness 和 physical plausibility。

参考链接:
- Project page: https://dagroup-pku.github.io/ReVidgen.github.io/
- GitHub: https://github.com/DAGroup-PKU/ReVidgen
- HuggingFace Dataset: https://huggingface.co/datasets/DAGroup-PKU/RoVid-X

---

## 2. RBench 设计详解

### 2.1 Benchmark 构造逻辑

RBench 总共 650 个 samples,分为两个维度:

**Task-oriented 维度(250 samples)**:
- **Common Manipulation**(50): grasp, place, push, rotate, press 等基础动作
- **Long-Horizon Planning**(50): 多阶段动作序列,如 "open refrigerator → take out box → close door"
- **Multi-Entity Collaboration**(50): Primary Entity(robot)+ Secondary Entity(human/animal/robot)的交互
- **Spatial Relationship**(50): above/below, left/right, front/behind 等空间关系
- **Visual Reasoning**(50): color, count, attribute matching, geometric understanding 等

**Embodiment-specific 维度(400 samples)**:
- **Dual-arm robots**(100): bimanual coordination
- **Humanoid robots**(100): tool use, full-body posture
- **Single-arm robots**(100): precise object interaction
- **Quadruped robots**(100): terrain adaptation, locomotion continuity

这种设计的 intuition 在于:**robotic video generation 的失败模式是 embodiment-specific 的**。例如,humanoid robot 由于在 large-scale human activity datasets 中有大量 pretraining 数据,模型表现往往较好;而 single-arm fine-grained manipulation 由于缺乏 contact dynamics 的建模,表现较差。

### 2.2 关键设计选择

paper 强调三个 critical 设计原则:
1. **避免训练数据泄漏**:evaluation set 中的 videos 不出现在 RoVid-X training data 中,且对每个 reference image 重新设计 task prompts
2. **Human verification**:所有 samples 经过人工 annotation 与过滤
3. **Metadata richness**:记录 manipulated object, embodiment type, camera viewpoint(first-person/third-person)

---

## 3. Automatic Metrics 深度解析

这是 paper 的技术核心。作者设计了 5 个 fine-grained metrics,分为 **Task Completion** 与 **Visual Quality** 两个维度。

### 3.1 Physical-Semantic Plausibility (PSS)

通过 MLLM(Qwen3-VL 或 GPT-5)进行 VQA-style 评估,针对 4 类 common failure modes:

1. **Floating/Penetration**: robot 或 object 悬空或互相穿透
2. **Spontaneous emergence**: 实体无因果地出现/消失
3. **Non-contact attachment**: object 在没有 visible contact 的情况下跟随 robot 移动
4. **Incorrect grasp**: gripper closure 不正确

直觉上,这类错误是 standard perceptual metrics(如 FID, FVD)无法检测的,因为像素层面可能看起来很 smooth,但物理上完全不合理。

### 3.2 Task-Adherence Consistency (TAC)

针对 5 种 task family 设计 task-specific criteria:

**Visual Reasoning task** 使用 Question Chain 机制:
$$\text{Score} = 5 \times \frac{\text{completed questions}}{\text{total questions}}$$

变量含义:
- 5: 满分上限
- completed questions: MLLM 基于 generated video 能正确回答的 verification questions 数量
- total questions: 从 prompt 自动生成的 stepwise verification questions 总数

**Long-Horizon Planning** 使用 Event Completion Rate:
$$\text{Score} = 5 \times \frac{\text{completed events}}{\text{total events}}$$

变量含义:
- completed events: 视频中正确执行的 ordered sub-events 数量
- total events: prompt 定义的 ordered event set 总数

直觉:这种设计把 "task correctness" 从 binary judgment 转化为 **partial credit**,允许模型在部分 sub-goal 完成时获得部分分数,更符合 robotic task 的 hierarchical 结构。

### 3.3 Motion Amplitude Score (MAS)

这是 paper 中最 elegant 的 metric 之一,旨在解决一个 critical failure mode:**video 看起来很 smooth,但 robot 几乎不动**。

完整公式链:

**Step 1 - Frame-level displacement**:
$$\bar{D}_t = \frac{1}{K} \sum_{k=1}^{K} \|\mathbf{p}_{t,k} - \mathbf{p}_{t-1,k}\|_2$$

变量含义:
- $\bar{D}_t$: 第 $t$ 帧所有 tracked points 的平均位移
- $K$: tracked points 总数
- $\mathbf{p}_{t,k}$: 第 $k$ 个 tracked point 在第 $t$ 帧的 2D location
- $\|\cdot\|_2$: L2 norm(Euclidean distance)

**Step 2 - Resolution normalization**:
$$\tilde{D}_t = \frac{\bar{D}_t}{\sqrt{W^2 + H^2}}$$

变量含义:
- $\tilde{D}_t$: normalized displacement
- $W, H$: video 的 width 和 height
- $\sqrt{W^2 + H^2}$: video diagonal length

直觉:不同 resolution 的视频,相同的 pixel displacement 代表不同的 physical motion,需要 normalize。

**Step 3 - Camera-motion compensation (soft-zero strategy)**:
$$\hat{D}_t = \begin{cases} \tilde{D}_t - \tilde{D}_t^{bg}, & \tilde{D}_t > \tilde{D}_t^{bg} \\ \tilde{D}_t, & \tilde{D}_t \leq \tilde{D}_t^{bg} \end{cases}$$

变量含义:
- $\hat{D}_t$: camera-motion compensated displacement
- $\tilde{D}_t^{bg}$: background region 的 normalized motion(通过对 robot mask 取 inverse 后 tracking 估计)

直觉:如果 robot 的 apparent motion 小于 background motion,说明 robot 实际上是 static 的,apparent motion 来自 camera movement。但 soft-zero 保留小的 residual value 而非 hard zero,提高了对 tracking noise 和 partial occlusion 的 robustness。

**Step 4 - Final MAS**:
$$\text{MAS} = \frac{1}{T} \sum_{t=1}^{T} \min(\hat{D}_t, 1)$$

变量含义:
- $T$: total frame number
- $\min(\hat{D}_t, 1)$: clip 操作,stabilize extreme values

**Implementation pipeline**:
1. GroundingDINO [65] 定位 active subject
2. GroundedSAM [81] 生成 temporally stable masks
3. CoTracker [55] 跟踪 dense grid of keypoints within robot mask

参考:https://github.com/IDEA-Research/GroundingDINO
参考:https://github.com/IDEA-Research/Grounded-SAM
参考:https://github.com/facebookresearch/co-tracker

### 3.4 Motion Smoothness Score (MSS)

基于 Q-Align aesthetic score [99] 的 temporal consistency 评估:

**Step 1 - Per-frame quality score**:
对 sliding window(w=3)内的 frames 使用 Q-Align 获得 $\{Q_t\}_{t=1}^{T}$

**Step 2 - Temporal fluctuation**:
$$\Delta Q_t = |Q_t - Q_{t-1}|, \quad t = 2, \ldots, T$$

变量含义:
- $\Delta Q_t$: 相邻帧 quality score 的绝对差
- $Q_t$: 第 $t$ 帧的 Q-Align aesthetic score

**Step 3 - Adaptive threshold**:
$$\tau_s(m) = \begin{cases} 0.01, & m < 0.1 \\ 0.015, & 0.1 \leq m < 0.3 \\ 0.025, & 0.3 \leq m < 0.5 \\ 0.03, & m \geq 0.5 \end{cases}$$

变量含义:
- $m$: Motion Amplitude value(来自 MAS)
- $\tau_s(m)$: piecewise adaptive threshold

直觉:**低 motion video 使用更严格的 threshold**(0.01),因为 subtle temporal inconsistencies 更容易被察觉;**高 motion video 使用更宽松的 threshold**(0.03),避免 penalize 自然快速运动。

**Step 4 - Anomaly detection**:
$$I_t = \mathbb{I}[\Delta Q_t > \tau_s(m)]$$

变量含义:
- $I_t$: indicator function,第 $t$ 帧是否为 temporal anomaly
- $\mathbb{I}[\cdot]$: indicator function

**Step 5 - Final MSS**:
$$\text{MSS} = 1 - \frac{1}{T} \sum_{t=2}^{T} I_t$$

变量含义:
- MSS: Motion Smoothness Score,范围 [0, 1]
- $\sum_{t=2}^{T} I_t$: anomaly frames 总数

### 3.5 Robot-Subject Stability (RSS)

采用 **contrastive VQA** 机制:同时观察 reference frame 与 generated frame,评估 entity 的 appearance, structure, semantics 一致性。

针对两类 failure:
- **Robot structural stability**: humanoid → single-arm, quadruped → humanoid, parallel gripper → dexterous hand 等 morphology drift
- **Subject appearance stability**: color drift, material change, rigid → deformable 等

### 3.6 Score Aggregation

最终将 5 个 metrics 聚合为 2 个 high-level indicators:

**Normalization**:
$$s \leftarrow \text{clip}_{[0,1]} \left(\frac{s - s_{\min}}{s_{\max} - s_{\min}}\right)$$

变量含义:
- $s$: normalized metric value
- $s_{\min}, s_{\max}$: metric 的原始 range

**Motion-amplitude penalty**:
$$P_{MA}(\text{MA}) = \begin{cases} (t - \text{MA}) + \delta, & \text{MA} < t_{\text{low}} \\ t - \text{MA}, & t_{\text{low}} \leq \text{MA} < t \\ 0, & \text{MA} \geq t \end{cases}$$

参数:
- $t = 0.1$: 主 threshold
- $t_{\text{low}} = 0.05$: 严重不足 threshold
- $\delta = 0.1$: 额外 penalty

直觉:当 MAS < 0.05 时,给予额外 $\delta = 0.1$ 的 penalty,因为这种 "almost static" 的 video 在 robotic 场景中是严重 failure。

**Stability-consistency penalty**:
$$p(g) \in \{0.2, 0.4, 0.6, 0.8\} \quad \text{for grades } g \in \{B, C, D, E\}$$

$$P_{RSS} = \begin{cases} \frac{p(g_r) + p(g_o)}{2}, & \text{if both exist} \\ p(g_r), & \text{if only } g_r \text{ exists} \\ 0, & \text{otherwise} \end{cases}$$

变量含义:
- $g_r$: robot stability grade(A-E,A 为最优,零 penalty)
- $g_o$: object stability grade
- $P_{RSS}$: 综合稳定性 penalty

**Final indicators**:
$$\text{TC} = \frac{\text{PSS} + \text{TAC}}{2}$$

$$\text{VQ} = \max(0, \, 0.8 \cdot \text{RSS} + 0.2 \cdot \text{MS} - P_{MA}(\text{MA}) - P_{RSS})$$

变量含义:
- TC: Task Completion score
- VQ: Visual Quality score
- PSS: Physical-Semantic Plausibility
- TAC: Task-Adherence Consistency
- RSS: Robot-Subject Stability
- MS: Motion Smoothness
- $P_{MA}$: motion amplitude penalty
- $P_{RSS}$: stability penalty

直觉:VQ 中 RSS 权重(0.8)远高于 MS(0.2),因为在 robotic 场景中,robot 和 object 的 structural consistency 比 temporal smoothness 更重要。$\max(0, \cdot)$ 确保 VQ 非负。

---

## 4. RoVid-X Dataset 构造

### 4.1 Four-Stage Pipeline

**Stage 1: Robot Video Collection**
- 来源:20+ open-source embodied datasets + internet video platforms
- 使用 GPT-5 [76] 自动过滤,识别 robotic task 相关内容
- 产出:~3M raw robotic video clips

**Stage 2: Video Quality Filtering**
- Scene segmentation detection 移除非 robot 内容
- Multi-dimensional quality scoring:
  - Clarity
  - Dynamic effects
  - Aesthetic performance
  - OCR

**Stage 3: Task Segmentation and Captioning**
- 使用 video understanding model(Seed1.5-VL [34])
- 基于 timestamp 分割为 task segments
- 自动生成 standardized subtitles,包含:
  - Action subject(e.g., "right arm", "left gripper")
  - Manipulated object(e.g., "nameplate", "box")
  - Operation details(e.g., "grasp and move", "remove from table")

**Stage 4: Physical Property Annotation**
- **FlashVSR** [114]: video super-resolution,提升 action details
- **AllTracker** [43]: unified optical flow annotation,确保 cross-scene tracking consistency
- **Video Depth Anything** [16]: relative depth maps,描述 spatial relationships

参考:
- FlashVSR: https://arxiv.org/abs/2510.12747
- AllTracker: https://alltracker.github.io/
- Video Depth Anything: https://github.com/DepthAnything/Video-Depth-Anything

### 4.2 Dataset Statistics 对比

| Dataset | Year | #Videos | #Skills | Resolution | Optical Flow | Diverse Forms | Captions |
|---------|------|---------|---------|-----------|--------------|---------------|----------|
| Open X-Embodiment [78] | 2024 | 1.4M | 217 | 64P-720P | ✗ | ✓ | ✗ |
| Agibot World [12] | 2025 | 1M | 87 | 480P | ✗ | ✗ | ✗ |
| **RoVid-X (Ours)** | 2026 | **4M** | **1300+** | **720P** | **✓** | **✓** | **✓** |

RoVid-X 的关键优势:
1. **规模最大**:4M clips,是 Open X-Embodiment 的 ~2.86 倍
2. **Skill 多样性最高**:1300+ skills,远超其他 dataset
3. **Physical annotations**:唯一提供 optical flow 和 depth maps 的 dataset
4. **Standardized captions**:统一的 task description 格式

---

## 5. 实验结果与 Key Insights

### 5.1 主实验结果(25 个 models)

**Top 5 Models**:

| Rank | Model | Avg. | Type |
|------|-------|------|------|
| 1 | Wan 2.6 | 0.607 | Commercial |
| 2 | Seedance 1.5 pro | 0.584 | Commercial |
| 3 | Wan 2.5 | 0.570 | Commercial |
| 4 | Hailuo v2 | 0.565 | Commercial |
| 5 | Veo 3 | 0.563 | Commercial |

**关键发现**:

#### Insight 1: Iterative Scaling Unlocks Physical Capabilities

Wan 系列 evolution:
- Wan 2.1(Rank 14, 0.399)→ Wan 2.2_A14B(Rank 8, 0.507)→ Wan 2.5(Rank 3, 0.570)→ **Wan 2.6(Rank 1, 0.607)**

性能提升 ~52%,说明 **scaling laws 不仅改善 visual quality,还在 active refining model 对 physics, distinct motion patterns, control logic 的理解**。

#### Insight 2: The Media-Simulation Gap

Sora 系列表现:
- Sora v1: Rank 22, Avg 0.266
- Sora v2 Pro: Rank 17, Avg 0.362

这与公众认知严重不符。Paper 解释为 **"domain gap"**:consumer-oriented models 优化 visual smoothness 和 cinematic transitions,牺牲了 physical fidelity 和 precise motion control。

直觉:creative video generation 的成功 **不能自然 transfer** 到 embodied AI tasks,这突出了 physically-grounded training data 的必要性。

#### Insight 3: Closed-source vs Open-source Gap

Top 7 全部是 commercial closed-source models。最好的 open-source model(Wan2.2_A14B, 0.507)与 SOTA commercial model(Wan 2.6, 0.607)差距 ~16.7%。

#### Insight 4: Specialization Dilemma

- **Cosmos 2.5**(robotics-specific, Rank 9, 0.464): 超过许多更大的 open-source video models
- **Vidar**(Rank 24, 0.206)和 **UnifoLM-WMA-0**(Rank 25, 0.123): 在 specific robot entity 上 fine-tune 的模型表现极差

直觉:**domain-specific data 对 control precision 有价值,但无法补偿 large-scale pretraining 提供的 "World Knowledge" deficit**。这是 embodied AI 的核心 trade-off。

#### Insight 5: Cognitive and Fine-grained Control Bottlenecks

- **Cognitive Gap**: Wan 2.6 在 Visual Reasoning 上仅 0.531,远低于 execution-oriented tasks
- **Manipulation Gap**: 所有 models 在 Quadruped(coarse locomotion)上表现 > Single-arm(fine-grained manipulation)

例如 Wan 2.6:
- Quadruped: 0.723
- Single arm: 0.666
- Visual Reasoning: 0.531

直觉:**fine-grained contact dynamics 比 rhythmic legged locomotion 更难建模**,这是 video generation models 的 fundamental limitation。

### 5.2 Human Preference Study

- 30 participants
- Win/Tie/Loss: 5/3/1 scoring
- **Spearman correlation: ρ = 0.96**(p < 10⁻³)

**Bland-Altman Analysis with LOO Calibration**:

为了评估 absolute agreement(而非仅 rank correlation),作者进行了 leave-one-out linear calibration:

$$(\hat{\alpha}_{-i}, \hat{\beta}_{-i}) = \arg\min_{\alpha, \beta} \sum_{j \in S_{-i}} (B_j - \alpha - \beta H_j)^2$$

变量含义:
- $(\hat{\alpha}_{-i}, \hat{\beta}_{-i})$: 使用除 model $i$ 外的所有 models 估计的 OLS calibration parameters
- $B_j$: model $j$ 的 benchmark score
- $H_j$: model $j$ 的 human score
- $S_{-i} = \{j : j \neq i\}$: leave-one-out training index set

Calibrated score:
$$H_i^* = \hat{\alpha}_{-i} + \hat{\beta}_{-i} B_i$$

Bland-Altman statistics:
$$d_i = B_i - H_i^*, \quad m_i = \frac{B_i + H_i^*}{2}$$

变量含义:
- $d_i$: difference between benchmark and calibrated human score
- $m_i$: mean of the two scores

结果:**Bias = 0.002, LoA = [-0.108, 0.112]**,说明 calibrated benchmark scores 与 human judgments 高度一致。

### 5.3 RoVid-X 有效性验证

使用 200k samples(从 4M 中随机采样)fine-tune Wan2.1_14B 和 Wan2.2_5B,使用 MSE loss:

| Model | Manip. | Long. | Multi. | Spatial. | Reason. | Total |
|-------|--------|-------|--------|----------|---------|-------|
| Wan2.1_14B | 0.344 | 0.335 | 0.282 | 0.268 | 0.205 | 0.399 |
| Wan2.1_14B+Ours | 0.376 | 0.389 | 0.295 | 0.314 | 0.298 | 0.446 |
| Wan2.2_5B | 0.331 | 0.318 | 0.142 | 0.313 | 0.234 | 0.380 |
| Wan2.2_5B+Ours | 0.373 | 0.387 | 0.221 | 0.403 | 0.284 | 0.439 |

所有 task domains 都有稳定提升,其中 **Spatial Relationship 提升最显著**(Wan2.2_5B: +0.090),说明 physical property annotations(特别是 depth maps)对 spatial reasoning 有直接帮助。

---

## 6. Technical Architecture 解析

虽然 paper 主要 focus 在 benchmark 和 dataset,但从 evaluation 结果可以推断一些 architectural insights:

### 6.1 评估的 25 个 models 分类

**Commercial(10)**:
- Wan 2.6, Wan 2.5(Sora-class DiT)
- Seedance 1.5 pro, Seedance 1.0(3D causal VAE, 4×16×16 compression)
- Hailuo v2, Veo 3, Kling 2.6 pro
- Sora v2 Pro, Sora v1

**Open-source(11)**:
- Wan2.2_A14B(MoE architecture,dynamic expert allocation)
- Wan2.2_5B, Wan2.1_14B
- HunyuanVideo 1.5, HunyuanVideo
- LongCat-Video(13.6B,GRPO RLHF)
- LTX-2, LTX-Video(1:192 compression ratio)
- SkyReels, FramePack, CogVideoX_5B

**Robotics-specific(4)**:
- Cosmos 2.5(NVIDIA,flow-based)
- DreamGen(gr1), DreamGen(droid)
- Vidar(MIDM - Masked Inverse Dynamics Model)
- UnifoLM-WMA-0

### 6.2 Architecture-Metric 关联分析

从 Table 2 数据可以观察到:

1. **MoE 架构优势**:Wan2.2_A14B(0.507)> Wan2.2_5B(0.380),MoE 的 dynamic expert allocation 对 temporal understanding 有帮助
2. **3D causal VAE**:Seedance 系列使用 4×16×16 compression ratio,在 long-horizon planning 上表现突出(Seedance 1.5 pro: 0.570)
3. **Flow-based architectures**:Cosmos 2.5 在 physical plausibility 上表现优异(PSS=0.620 in Quadruped),说明 flow-based 方法对 motion modeling 有天然优势

---

## 7. Limitations 与 Future Directions

### 7.1 当前 Limitations

1. **Evaluation 依赖 MLLM**:Qwen3-VL 和 GPT-5 作为 evaluator 可能引入自身的 biases
2. **Action recovery 未实现**:目前只能评估 video quality,无法直接评估 generated video 是否可以转化为 executable robot actions
3. **Simulation-to-real gap**:RBench 评估的是 video fidelity,但 real-world robotic deployment 需要 kinematic/dynamic feasibility

### 7.2 Future Work

Paper Section 6 提出三个方向:

1. **Inverse Dynamics Models (IDM)**:从 generated videos 中 recover executable actions,实现 closed-loop control
   - 相关工作:[4, 22, 89, 112]
   - Latent action models:[88, 107]

2. **Automated physical metrics**:开发更 automated、physically grounded 的 evaluation metrics,评估 kinematic 和 dynamic feasibility

3. **Physical capability training**:训练具有 improved physical capabilities 的 video generation models

参考:
- Genie 3: https://arxiv.org/abs/2501.07062
- GR00T N1: https://arxiv.org/abs/2503.14734
- DreamGen: https://arxiv.org/abs/2505.12705

---

## 8. Intuition Building:为什么这个工作重要

### 8.1 Paradigm Shift 的信号

Paper 最深刻的贡献在于揭示了 video generation 领域的 **paradigm shift**:

**Old paradigm**: Visual Fidelity
- 目标:generate aesthetically pleasing videos
- Metrics:FID, FVD, VBench scores
- 代表:Sora v1, Runway, Pika

**New paradigm**: Physical Intelligence
- 目标:simulate physically realistic robot behaviors
- Metrics:task completion, physical plausibility, action completeness
- 代表:Wan 2.6, Seedance 1.5 pro, Cosmos 2.5

Sora 系列在 RBench 上的 poor performance(Rank 17, 22)是这一 shift 的最强证据。

### 8.2 Data-Centric AI 的体现

RoVid-X 的设计体现了 **data-centric AI** 的核心理念:

1. **Quality > Quantity**:4-stage pipeline 中的 quality filtering 比 raw scale更重要
2. **Physical annotations**:optical flow + depth maps 为 model 提供 multi-modal physical priors
3. **Standardized captions**:统一的 task description 格式降低 cross-dataset co-training 的难度

### 8.3 Benchmark 作为 Research Compass

RBench 的真正价值在于作为 **research compass**:

- **Cognitive Gap**(Visual Reasoning: 0.531 for SOTA)指向需要更好的 reasoning capabilities
- **Manipulation Gap**(Single-arm < Quadruped)指向需要更好的 contact dynamics modeling
- **Media-Simulation Gap**(Sora poor performance)指向需要 physically-grounded training

这些 insights 为下一代 video generation models 的研发提供了明确的 direction。

---

## 9. 与相关工作的 positioning

### 9.1 vs VBench [46]

VBench 评估 16 个 dimensions,包括 subject consistency, background consistency, temporal flickering, motion smoothness 等。但缺乏:
- Task-level correctness 评估
- Physical plausibility 评估
- Embodiment-specific 评估

RBench 通过 5 个 fine-grained metrics 填补这些 gap。

### 9.2 vs VideoPhysics [6] / PhyBench [72, 73]

这些 benchmark 评估 general physical commonsense(如 gravity, collision),但:
- 缺乏 task-specific datasets
- 缺乏 embodiment-specific criteria
- 缺乏 action-goal alignment 评估

RBench 通过 task-oriented 和 embodiment-specific 双维度设计解决这些问题。

### 9.3 vs Open X-Embodiment [78]

Open X-Embodiment 是 robotic learning 最大的 dataset(1.4M),但:
- 无 optical flow annotations
- 无 standardized captions
- Resolution 范围广(64P-720P),质量不均

RoVid-X 通过 4-stage pipeline 提供 4M high-quality(720P)clips with comprehensive physical annotations。

---

## 10. 总结

这篇 paper 的核心贡献可以概括为三点:

1. **Diagnosis**:通过 RBench 系统性诊断了 25 个 SOTA video generation models 在 embodied AI 场景下的 deficiencies,揭示了 visual fidelity ≠ physical intelligence 的关键 insight

2. **Data**:RoVid-X 提供了 4M clips 的大规模、高质量、physically-annotated training data,为下一代 physically-grounded video models 奠定基础

3. **Direction**:通过 Cognitive Gap, Manipulation Gap, Media-Simulation Gap 三个 key insights,为 embodied AI 领域的 video generation 研究指明方向

paper 的影响将体现在:
- 推动商业 video models(如 Sora)向 physical intelligence 方向迭代
- 为 open-source 社区提供 fair comparison 的 benchmark
- 通过 RoVid-X 加速 physically-grounded video foundation models 的训练
- 为 video → action 的 IDM 研究提供高质量视频数据

这是连接 video generation 与 embodied AI 两个领域的重要桥梁工作,值得深度关注。

---

**关键参考资料**:
- Paper: https://dagroup-pku.github.io/ReVidgen.github.io/
- Code: https://github.com/DAGroup-PKU/ReVidgen
- Dataset: https://huggingface.co/datasets/DAGroup-PKU/RoVid-X
- VBench: https://vchitect.github.io/VBench-project/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Cosmos: https://arxiv.org/abs/2511.00062
- DreamGen: https://arxiv.org/abs/2505.12705
