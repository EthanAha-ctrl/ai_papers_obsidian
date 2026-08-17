---
source_pdf: ROBOTIC TASK GENERALIZATION VIA HINDSIGHT TRAJECTORY SKETCHES.pdf
paper_sha256: b2b00e9a23ac2ec401eba606e90b3f5009064c51e26b58cdc8b53d1e658363b9
processed_at: '2026-08-12T01:54:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 咱们用大白话来过一遍这篇 paper。抛开那些学术黑话，它讲了一个非常漂亮且直觉极强的故事。

---

### 1. 痛点：教机器人干活，语言其实很贫乏

假设你给一个 robot policy 输入一大堆训练数据，里面有抓取可乐罐、把苹果放到碗里这种操作，你用 language 指令去 condition 它，比如 RT-1 和 RT-2 那样。训练完，让它去执行“fold towel”（叠毛巾）。

结果它完全傻眼了。

但在物理层面上，叠毛巾这个动作的 arm trajectory，跟“抓起一个物体并放到另一边”的轨迹是非常像的。为什么它做不出来？因为 policy 的 generalization 本质上是在 conditioning representation space 里面做 interpolation。在 language embedding 的流形上，“fold towel” 这个 token sequence 离“pick coke can”十万八千里。对 policy 来说，这就是一个 out-of-distribution 的输入，哪怕底层的 motion 几何上长得一模一样，它也无法泛化过去。

这就像一个只会照着菜单做菜的厨师，客人点了个菜单上没有的“西红柿炒鸡蛋”，哪怕他厨房里有西红柿有鸡蛋，他也懵了，因为他的 representation space 里面没有这个组合。

### 2. Insight：换一种告诉它“干什么”的语言

既然 language 这个 conditioning 太抽象，导致新任务离训练数据的语义距离太远，那我们就换一个 representation。要求它既能表达 motion 细节，又足够粗略，不会像 video 那样信息爆炸难以学习。

这篇 paper 提出的 RT-Trajectory 就是直接在 camera 拍到的 2D 图像上，画一条机械臂末端要走的轨迹线。

这为什么 work？因为这条线直接描述了 motion 的几何形状，同时它是一张 2D image，可以直接喂给 EfficientNet 处理。无论你想执行什么没见过的 task，只要它手部运动的 2D 投影跟训练集中的某些 motion 有几何相似性，policy 就能在 trajectory manifold 上 interpolate 过去。

“fold towel”在 language space 里面离“pick and place”很远，但它们的 2D trajectory sketch 却长得极其相似。

### 3. 数据怎么来？Hindsight 是精髓

最大的工程亮点是：不需要人工去一条条画轨迹来训练。

我们手里有 RT-1 那套 7 万多条的 demonstration dataset。每一条都有机械臂的 proprioception 数据。我们就回过头去，把每帧的 3D end-effector position 投影到 2D image plane 上，自动把这些点连成线。这叫 hindsight trajectory labeling。

你甚至不需要给数据集打额外的 label，直接从已有数据里自动提取 conditioning signal，这就让整个方法 scalable 到任何已有的 robot dataset 上。

### 4. 轨迹长什么样？里面藏了什么机关？

这条轨迹不仅仅是条线，它是一个 RGB image，三个通道被塞满了信息：
- **2D Trajectory 本身**：像素坐标上连起来的线。
- **Red channel**：编码时间进度 $R_t = \frac{t+1}{T}$。$t$ 是当前步，$T$ 是总长度。颜色从暗变亮，告诉 policy 走到哪一步了，隐含了 velocity 信息。
- **Green channel**：编码高度（用于 2.5D 版本）$G_t = \frac{h_{t+1} - h_{min}}{h_{max} - h_{min}}$。$h$ 是末端高度。因为纯 2D 轨迹不知道手该举多高，加了这个就能区分在桌子上抓东西和在椅子上抓东西。
- **Interaction Markers**：用画圈来标记什么时候该 close gripper，什么时候该 open。判定逻辑是计算 $\delta_t = \hat{p}_t - p_t$。$\hat{p}_t$ 是指令要求的 gripper position，$p_t$ 是传感器读到的实际 position。如果指令让它 close 但实际 position 没跟上（$\delta_t > 0$），说明碰到东西了，这就是抓取的关键帧。

### 5. 实验结果：碾压级别的对比

在 7 个完全没见过的 task 上测试（比如抓椅子上的东西、叠毛巾、转椅子）：

| Method | Overall Success Rate |
|---|---|
| RT-Trajectory (2.5D) | 67% |
| RT-Trajectory (2D) | 50% |
| RT-1-Goal (goal image) | 26% |
| RT-1 (language) | 17% |
| RT-2 (language + web VQA) | 11% |

这个表极其震撼。RT-2 喂了海量的 internet VQA data，结果在没见过的机器人 task 上比只吃 robot data 的 RT-1 还差。这印证了如果你 representation 选错了，喂再多外部 data 也可能起负作用，因为它把 policy 在 representation space 里面推得更远离实际的 robot motion manifold 了。

### 6. 测试时怎么画这条线？多样性令人发指

这篇 paper 最有意思的地方是，trajectory 这个 interface 对接了各种输入模态：
1. **Human Drawing**：开发个 GUI，人看着相机画面用鼠标画。
2. **Human Video**：拿手机录一个人叠毛巾的第一视角视频，用 Mediapipe 提取手部 21 个 landmark，映射成机械臂末端轨迹，自动生成 sketch。
3. **LLM Code as Policies**：让 GPT-4 写代码算出 3D waypoints，投影成 2D sketch。LLM 生成的轨迹很直溜，但 RT-Trajectory 依然能 follow，甚至比纯 IK planner 表现还好，因为 policy 能自适应场景中物体的细微朝向。
4. **Image Generation VLM**：用 PaLM-E 风格的模型，输入语言指令和初始图像，直接生成 trajectory sketch 的 tokens。

### 7. Emergent Capabilities：涌现的能力

这篇 paper 让我感到最兴奋的其实是这里：

**Visual Prompt Engineering for Robots**

既然 conditioning 是一条画出来的线，那就意味着你可以像调 LLM prompt 一样去调机器人的行为。如果机器人执行失败了，你完全不需要重新训练模型，也不需要采集新数据。你只需要改一下你画的那条线：比如让它先往上挑高一点再往下放，它就能把东西放到更高的架子上。

这种交互范式太像 LLM 的 zero-shot prompting 了。你是在“prompt 一个 robot”，这为未来的人机交互打开了一扇大门。

**Retry Behavior**

遇到失败，比如抓抽屉把手没抓到，learning-based policy 会自己调整去抓抽屉边缘。这种容错能力是传统 IK planner 绝对做不到的。

### 8. 怎么证明这些任务真的是“没见过”的？

这篇 paper 用了一个非常硬核的数学工具：Fréchet distance。

公式长这样：
$$F_D(\tau, \tau') = \max\left(d(\rho_0, \rho'_0),\, \min\left\{ F_D(\tau[1:], \tau'[1:]),\, F_D(\tau, \tau'[1:]),\, F_D(\tau[1:], \tau') \right\}\right)$$

这里的 $\tau$ 和 $\tau'$ 是两条轨迹，$\rho_0$ 是轨迹的起点。直觉上，这就像两个人各牵一条狗，分别沿着两条轨迹走，狗必须按顺序走，但可以停顿等对方。狗绳最短需要多长，Fréchet distance 就是多少。它只看形状，不在乎速度。

paper 把评估时的轨迹和训练集里所有的轨迹算 Fréchet distance，找出最相似的 10 条。结果发现，对于没见过的 task，最相似的训练轨迹在语义上完全不相干（比如评估的是 place fruit，最像的训练轨迹是 move can near bottle）。这定量证明了 policy 确实是在做 cross-task 的 motion generalization，仅仅因为你在测试时给了它一个对齐到 motion manifold 的 conditioning。

### 9. 总结与狂野联想

这篇 paper 的核心 thesis 极其清晰：**Conditioning representation 的选择决定了 generalization 的边界**。

RT-Trajectory 找到了一个 sweet spot：它比 language 具体，能描述 motion 几何；它比 video 简单，没有背景噪声且易于学习。更妙的是它 hindsight 自动标注，完全 scalable。

往远了想，这种把“任务描述”转化为“空间几何约束”的范式，跟 **VoxPoser** 用 LLM 生成 3D value map 有异曲同工之妙。VoxPoser 是在 3D space 里画价值地图，RT-Trajectory 是在 2D image space 里画运动草图。

如果未来我们把这种 2D sketch 升级成 **4D video generation**（像 Sora 那样生成未来状态的 video），然后再用 policy 去 follow 这个 video，这就走向了 **UniPi** 的路线。但 RT-Trajectory 告诉我们，其实不需要那么高维的 generative planning，一个极度压缩的 2D sketch 加上强大的 visual encoder，可能就是目前最 sample-efficient 且最 controllable 的方案。

Link:
- Paper: https://arxiv.org/abs/2311.12554
- Project: https://rt-trajectory.github.io/
- RT-1: https://robotics-transformer1.github.io/
- RT-2: https://robotics-transformer2.github.io/
- VoxPoser: https://voxposer.github.io/
- UniPi: https://universal-policy.github.io/
- Code as Policies: https://code-as-policies.github.io/

---

# RT-Trajectory: 用 2D 轨迹草图实现 robotic task generalization 的 deep dive

Andrej, 这篇 paper 我读下来觉得有一个相当 elegant 的核心 insight, 我想先把它说出来, 因为整个 paper 的设计都围绕它在转。下面我会从 intuition 开始, 然后展开技术细节、公式、实验, 最后给出一些跨 paper 的联想。

---

## 1. 这篇 paper 真正在解决什么问题

robotic policy 的 generalization 一直被分成几类: 对新 objects、新 backgrounds、新 language phrasing 的 generalization (visual / semantic generalization) 已经有大量工作 (RT-1, RT-2, CLIPort, VIMA 等)。但还有一个被相对忽略的维度: **task generalization**, 也就是泛化到训练时从未见过的 motion 模式。

这里有一个微妙但关键的现象: 一个在 pick-and-place 上训练的 language-conditioned policy, 拿来执行 folding 动作, 即使 folding 的 arm trajectory 在几何上与 pick-and-place 极其相似, 它也会失败。理由是——**policy 的 generalization 是在它的 conditioning representation space 中做 interpolation**, "folding" 这个 language token 在 language embedding manifold 上远离 "pick X near Y" 这类训练指令, 所以它属于 out-of-distribution, policy 无法 interpolate 到它。

paper 的核心 thesis 就是:

> Conditioning representation 的选择, 决定了 policy 在 inference 时能够 interpolate 到哪些 task。如果两个 task 在 chosen representation 上接近, policy 就能 generalize; 否则即使物理 motion 相似, 也无法迁移。

这是一个非常 deep 的观点, 它把 "generalization" 从一个抽象的 capability, 落地为 "test-time input 是否落在 training manifold 的近邻区域" 的几何问题。这种 framing 让我想到 manifold hypothesis 在 generative modeling 中的角色, 也让我想到 GAN / diffusion 中 "training distribution support" 的讨论。

paper: https://arxiv.org/abs/2311.12554
project page: https://rt-trajectory.github.io/

---

## 2. 为什么 trajectory sketch 是 sweet spot

paper 里给了一个非常清晰的 conditioning spectrum (Fig. 2), 我把它整理成一个轴:

| Conditioning 方式 | Specification level | 问题 |
|---|---|---|
| One-hot task ID | under-specified | 无法表达 task 间相似性, 无法泛化 |
| Language (RT-1/RT-2) | under-specified end-state | 无法精细描述 motion; 新 motion 在 language space 上 OOD |
| Goal image | over-specified end-state | 包含 background 等无关像素, 学习困难, 测试时获取成本高 |
| Goal video | over-specified 全 trajectory | 高维, 难学, 难 scale |
| 2D object-centric reps (CLIPort etc.) | intermediate | 需要 object detection 等额外管线 |
| **Trajectory sketch (本文)** | **intermediate, mid-level** | **既表达 motion, 又粗略到允许 policy 自己 interpret** |

trajectory sketch 的妙处在于它是一个 **"coarse but motion-centric"** 的 conditioning: 它告诉 policy "沿着这条 2D 曲线走, 在这些点 grasp/release", 但不告诉它具体的 joint 角度、不告诉它具体的 z-height (2D 版本)、不告诉它背景里有什么。policy 必须结合 visual observation 把这个粗略 sketch "interprete" 成具体动作。这种 under-determination 是关键——它让 policy 能在 scene nuances (object orientation, 干扰物, 高度差) 上自适应, 而不是机械跟随。

这其实是一个 **representation bottleneck 反而有益** 的例子。trajectory sketch 比 video 简单得多 (只是一张 2D 图像), 但足够 informative 到能区分 pick 和 fold, 因为它们的 2D motion 在 image space 上看起来确实不同。

---

## 3. Trajectory sketch 的构造细节

这是 paper 最技术化的部分, 我把它彻底拆开。

### 3.1 三个基本元素

paper 把 trajectory sketch 分解成三个 building blocks:

**(a) 2D Trajectory**: 每个 episode, 从 proprioception 拿到 robot end-effector center 在 robot base frame 下的 3D position, 用 camera extrinsic / intrinsic 投影到 image plane, 得到一个 pixel sequence。把这些点用直线连起来画在 blank image 上。

**(b) Color Grading** (用 RGB 通道编码额外信息):
- Red channel 编码 normalized temporal progress:
  
  $$R_t = \frac{t+1}{T}$$
  
  其中 $t$ 是 current time step, $T$ 是 episode 总长度。这编码了 "我在 trajectory 哪个阶段", 隐含 velocity / direction 信息。
  
- Green channel 编码 normalized height (用于 2.5D 版本):
  
  $$G_t = \frac{h_{t+1} - h_{min}}{h_{max} - h_{min}}$$
  
  其中 $h_{t+1}$ 是 end-effector 在时刻 $t+1$ 相对 robot base 的 height, $h_{min}, h_{max}$ 是 dataset 内的 height 极值。这把 z 轴信息塞进颜色里, 解决纯 2D trajectory 的 depth ambiguity。

**(c) Interaction Markers**: 标记 gripper 何时 grasp / release。判定逻辑很有意思:

定义 target gripper joint position 与 sensed position 的差:
  
$$\delta_t = \hat{p}_t - p_t$$

其中 $\hat{p}_t$ 是 commanded target position, $p_t$ 是实际 sensed position (注意 $p_t$ 随 gripper closing 而增大)。判定 closing 的 key step:

- 条件一: 当前正在 closing: $\delta_t > 0 \land \hat{p}_t > \epsilon$, 其中 $\epsilon$ 是一个 closing threshold。
- 条件二: 上一时刻还没在 closing, 当前开始 closing: 即 $\delta_t < 0 \lor \hat{p}_t \le \epsilon$ 但 $\delta_{t+1} > 0 \land \hat{p}_{t+1} > \epsilon$。

物理直觉: 当 gripper 命令要 close 但 actual position 落后于 target ($\delta_t > 0$), 通常意味着 gripper 碰到物体了, 阻止它继续 close——这就是 contact / grasp 的时刻。opening 的判定同理, 反向。

这些 key steps 在 trajectory 上画 green circle (closing) 或 blue circle (opening)。

### 3.2 两个版本

- **RT-Trajectory (2D)**: 2D trajectory + temporal color + interaction markers (RGB, 但 green 同时被 height 和 marker 占用? 实际上 paper 里 2D 版本的 green 用于 marker, 2.5D 版本的 green 用于 height, 应该是这样区分)
- **RT-Trajectory (2.5D)**: 在 2D 基础上增加 height 编码到 green channel。

2.5D 的必要性在 Pick from Chair 这种 task 上体现: 物体在椅子上, 高度与桌面不同, 2D trajectory 看起来可能像桌面 pick, 但实际 z 完全不同。Table 4 数据印证了这点: Pick from Chair 上, 2D 只有 0%, 2.5D 有 38%。

---

## 4. Hindsight labeling 的妙处

这是整个方法能在 scale 上 work 的关键。paper 不是去采集新的带 trajectory label 的数据, 而是 **从已有 demonstration dataset 自动地、hindsight 地** 抽取 trajectory labels。

这一点非常重要, 因为它意味着:
1. 可以复用 RT-1 的 73K demonstrations;
2. 任何新数据集都能用同样 pipeline 自动得到 trajectory supervision;
3. 没有 annotation bottleneck。

这让我想到 hindsight experience replay (HER) 在 RL 中的精神——给定一条已经成功的 trajectory, 我们 retrospectively 重新 label 它 "完成了什么 goal"。RT-Trajectory 是把这个思想用到 imitation learning 的 conditioning 上: 一条 trajectory "完成了什么 task" 由它自身的 2D 投影来描述, 而不是外部给的 language label。

---

## 5. Policy 架构与训练

paper 用 RT-1 (Brohan et al. 2023b) 作为 backbone, RT-1 本身是一个 Transformer policy, 输入是 6 帧 RGB 图像历史, 用 EfficientNet-B3 作 image tokenizer, 输出离散化的 7-DoF action tokens (包括 gripper)。原 RT-1 用 FiLM 把 language instruction 注入到 image features 里。

RT-Trajectory 的修改:

1. **Input concatenation**: 把 trajectory sketch 图像与每帧 RGB 沿 feature dimension 拼接, 即每个时间步的 input 是 `(RGB image, trajectory sketch)` pair。trajectory sketch 在整个 episode 中是固定的。
2. **EfficientNet first conv layer 扩展**: 因为 input channel 数变了 (RGB 3 + sketch 3 = 6), 新增 channel 的 conv kernel weights 初始化为 **all zeros**。这是个细节, 但很重要——初始化为 0 意味着训练初始时刻, 网络行为等价于原 RT-1, trajectory 信息完全不影响 forward pass, 然后通过梯度逐渐学到如何利用它。这是一种 "additive fine-tuning-friendly" 初始化, 类似 LoRA / adapter 的精神, 避免破坏预训练 feature。
3. **移除 FiLM layers**: 因为不再用 language。

training loss: 标准的 behavior cloning 负 log-likelihood:

$$\mathcal{L} = -\mathbb{E}_{(o_t, a_t) \sim \tau} \left[ \log \pi_\theta(a_t \mid o_t, c_{traj}) \right]$$

其中 $o_t$ 包含 6 帧 RGB history, $c_{traj}$ 是 trajectory sketch, $a_t$ 是 7-DoF action (bin-discretized)。

这里 trajectory sketch 是一个 image, 被同一个 image tokenizer 处理——这种"把 structured conditioning 塞成 image 让 EfficientNet 吃"的设计很简洁, 但也带来一个潜在问题: EfficientNet 是为 natural image 设计的, 处理 trajectory line drawing 这种 sparse signal 可能不是最优。不过 paper 的实验显示它 work 得很好, 这或许是因为 zero-init 让网络渐进学习。

---

## 6. Inference 时的 sketch 来源——这才是真正让人兴奋的部分

paper 探索了 4 种 inference 时生成 trajectory sketch 的方式, 这展示了 trajectory 作为 interface 的多功能性:

### 6.1 Human drawings (GUI)
人看 initial camera image, 在 GUI 上画一条曲线, 标记 grasp/release 点, 可选标 height。简单直觉。

### 6.2 Human demonstration videos
用 Mediapipe (Lugaresi et al. 2019) 检测 first-person video 中的 21 个 hand landmarks, 取 thumb 和 index finger 的 4 个 landmark 模拟 parallel gripper, 用 depth map lift 到 3D, interpolate 得到 end-effector pose, 再转换成 trajectory sketch。

这是一个非常 promising 的方向, 因为人类视频是地球上最丰富的 motion signal 来源。这让我联想到 WHIRL, RH20T, BridgeData 等试图利用人类视频的项目, 以及 Voltron, VideoDex, ROSIE 等用人类视频预训练 representation 的工作。

### 6.3 LLM with Code as Policies (CaP)
让 LLM (GPT-4) 写代码生成 3D waypoints (类似 Liang et al. 2022 的 Code as Policies, Gonzalez Arenas et al. 2023 的 prompt book), 然后把这些 waypoints 投影到 image 上画 trajectory。

LLM 生成的 trajectory 倾向于 "precise and linear"——因为它是为 IK planner 设计的, 不是为人类。但 RT-Trajectory 依然能 follow, 并且因为 policy 有学习到的 scene understanding, 在 diverse pick tasks 上 outperform 纯 IK planner (Table 1b: Pick 89% vs IK 83%)。这是 learning-based policy 相对 hard planner 的优势体现。

### 6.4 Image generation VLMs (PaLM-E style)
用一个 PaLM-E 风格的 VLM, 输入 (initial image, language instruction), 输出 trajectory image 的 vector-quantized tokens (用 ViT-VQGAN, Yu et al. 2022), detokenize 后得到 trajectory sketch。

这条路径目前还很 noisy, 但 paper 展示了 promising qualitative results。如果未来 image-generating VLM 改进, 这条路径会自然受益。这其实是一种 **"用生成模型把 language 翻译成 trajectory"** 的两阶段 pipeline, 与 RT-2 直接 language→action 的端到端形成对比。我个人觉得这种 decomposition 可能更 sample-efficient 且更 controllable, 因为 trajectory 是一个比 action sequence 更结构化的中间 representation。

---

## 7. 实验结果分析

### 7.1 Headline number

7 个 unseen skills, 64 trials 总计:

| Method | Overall |
|---|---|
| RT-Trajectory (2D) | 50% |
| RT-Trajectory (2.5D) | 67% |
| RT-1 (language) | 17% |
| RT-2 (language + VQA co-training) | 11% |
| RT-1-Goal (goal image) | 26% |

差距巨大。一个值得注意的反直觉点: **RT-2 (11%) 反而比 RT-1 (17%) 差**。这听起来奇怪, 因为 RT-2 通常被认为更强。解释是: RT-2 的 VQA co-training 让它在 language instructions 上更 specialized to web-scale language distribution, 但这里的 unseen tasks 在 language space 上完全 OOD, VQA knowledge 不仅没帮助, 可能反而让 representation 偏离了 robot-specific 的 motion manifold。这是 "更多 data 不一定更好, 取决于 representation 是否 align 到 evaluation 分布" 的一个 nice 案例。

### 7.2 Per-skill breakdown (Table 4)

- Place Fruit: 75% / 75% / 0% / 33% / 8% — RT-Traj 大胜, 因为 "place fruit into container" 的 motion 接近训练中的 "place X into receptacle"
- Upright and Move: 33% / 50% / 17% / 0% / 0%
- Move within Drawer: 67% / 100% / 33% / 0% / 17% — 2.5D 达到 100%, 因为 drawer 内 motion 需要精确 z
- Restock Drawer: 92% / 67% / 42% / 17% / 42% — 这里 2D (92%) 反而比 2.5D (67%) 高, 说明 height info 有时反而引入 noise, 取决于 task
- Pick from Chair: 0% / 38% / 0% / 0% / 17% — 2D 完全失败 (chair 高度未在训练分布中, 2D 无法 disambiguate), 2.5D 救了一半
- Fold Towel: 75% / 75% / 0% / 0% / 0% — 所有 baseline 全挂, 只有 RT-Traj 能做
- Swivel Chair: 0% / 70% / 17% / 0% / 50% — 涉及 underactuated system, 2.5D 70% 令人印象深刻

Fold Towel 和 Swivel Chair 这两个 task 上, language 和 goal-image baselines 全部 0% 或接近 0%, 而 RT-Traj 能做 70-75%——这是 trajectory conditioning 的杀手锏证据: 这些 motion 在训练中不存在, 但 trajectory sketch 把它们带到 policy 能理解的 manifold 上。

### 7.3 Diverse trajectory generation methods (Table 1)

Trajectory from human video:
- Pick: IK Planner 42% / Ours 2D 94% / Ours 2.5D 100%
- Fold Towel: IK 25% / Ours 75% / 75%

Trajectory from LLM CaP:
- Pick: IK 83% / Ours 89% / 89%
- Open Drawer: IK 71% / Ours 60% / 60%

Open Drawer 上 IK 反超, 这很合理: drawer 的 motion 非常 structured, IK planner 这种 hard-coded 方法在这种 task 上有优势。RT-Traj 在 pick (unstructured, 需要适应 object orientation) 上反超, 展示 learning-based policy 的 scene adaptation 优势。

---

## 8. Motion similarity 分析——这是 paper 最 undervalued 的部分

paper 用 **Fréchet distance** 量化 evaluation trajectory 与 training trajectories 的相似度, 试图回答: "unseen tasks 真的 unseen 吗? 它们离最近的 training trajectory 多远?"

### 8.1 Fréchet distance 公式

给定两条 trajectory $\tau = \{\rho_0, \rho_1, ..., \rho_m\}$ 和 $\tau' = \{\rho'_0, \rho'_1, ..., \rho'_n\}$, 其中 $\rho_i$ 是 3D waypoint, $d(\rho_i, \rho'_j)$ 是 Euclidean distance, Fréchet distance 递归定义为:

$$F_D(\tau, \tau') = \max\left(d(\rho_0, \rho'_0),\, \min\left\{ F_D(\tau[1:], \tau'[1:]),\, F_D(\tau, \tau'[1:]),\, F_D(\tau[1:], \tau') \right\}\right)$$

变量含义:
- $\tau, \tau'$: 两条 trajectory, 一条是 query (evaluation rollout), 一条是 reference (training sample);
- $\rho_0, \rho'_0$: 各自的第一个 waypoint;
- $\tau[1:]$: 把 $\tau$ 的第一个元素去掉, 返回剩余序列的子 trajectory;
- $d(\cdot, \cdot)$: 两个 waypoint 间的 Euclidean distance;
- $F_D$: 两条 trajectory 的 Fréchet distance。

直觉: 想象有两个人各牵着一条狗, 一人沿 $\tau$ 走, 一人沿 $\tau'$ 走, 都必须按 waypoint 顺序前进 (不能后退), 但可以独立选择何时前进到下一个 waypoint。狗绳长度需要满足让两条狗都能完成 walk 的最小值, 就是 Fréchet distance。它是 order-preserving, parameterization-independent 的——即不关心 trajectory 的速度, 只关心 shape。

递归式中:
- $\max(d(\rho_0, \rho'_0), \ldots)$: 当前起点的 distance 是一个 lower bound;
- $\min\{...\}$: 三种递归情况对应 "两个人都前进", "只有 τ 前进" 不行 (因为 τ' 还在 ρ'_0), 实际上正确递归是 "两人都前进一步 / 只有 τ 前进 / 只有 τ' 前进" 中取 min, 这保证我们选最省力的 walking pattern。

### 8.2 关键发现 (Fig. 10-12)

paper 把每个 evaluation trajectory 与 training set 中所有 trajectory 计算 Fréchet distance, 取 top-10 最相似的, 然后分析:

1. **Semantic relevance** (Fig. 10): 对于 seen skill (close top drawer), 最相似的 training trajectories 都是同 semantic skill; 对于 unseen skill, 最相似的 training trajectories 来自完全不同的 semantic skills。这印证了 paper 的核心论点——unseen task 在 motion space 上能找到邻居, 但这些邻居在 language space 上完全不同。

2. **First-interaction height alignment** (Fig. 11): 比较 evaluation trajectory 的 first grasp z-height 与最相似 training trajectory 的 first grasp z-height。对于 seen skill, 差值接近 0; 对于 unseen skills (尤其 Pick from Chair, Move within Drawer), 差值方差巨大——这意味着即使 2D shape 相似, 关键的 z 信息仍然不同, policy 必须真的 generalize 而不仅是 copy。

3. **Fréchet distance 分布** (Fig. 12): unseen skills 的 query trajectory 到 most similar training trajectory 的距离显著大于 seen skills 的——定量证明这些 task 确实是 OOD motion, 而不是 trivial interpolation。

这部分分析是 paper 中最 "scientific" 的, 它把 "generalization" 这个模糊概念量化了。我特别喜欢这点, 因为很多 robotics paper 都在 claim "我们 generalize 到 unseen X", 但从来不 measure "unseen" 到底多 unseen。

---

## 9. Emergent capabilities

### 9.1 Visual prompt engineering

这是 paper 中最让人拍案叫绝的 emergent behavior。RT-Trajectory 允许一种 **"visual prompt engineering for robot policies"**: 如果某个 trajectory prompt 失败, 实践者可以换一个 sketch 重新 query, 不需要重训或采新数据。这非常像 LLM 的 in-context prompting——同一 model, 改 input 就能改变行为 mode。

Fig. 19 给了一个例子: 把 apple 放到 middle stage vs top stage, 区别只在于 trajectory sketch 是先低后高 vs 直接斜上。这种 "robot 也吃 prompt" 的范式, 把 LLM 的 prompt engineering culture 直接迁移到 robotics, 这是个非常有想象力的方向。

### 9.2 Retry behavior

paper 还展示了 retry behavior (Fig. 20): robot 第一次抓 drawer handle 失败, 自动 retry 抓 drawer edge 成功。这种 emergent recovery 是 learning-based policy 相对 IK planner 的另一优势, 因为 IK planner 没有 "失败后调整" 的概念。

### 9.3 Realistic out-of-distribution scenarios

Fig. 8, 17, 18 展示了在 2 个新 building, 4 个新 room 的 evaluation——新背景、新光照、新家具、新 object、新 furniture geometry。RT-Trajectory 在 moderate prompt engineering 下能完成多种 task, 包括 pivot-hinge cabinet (训练只有 sliding drawer)。这种 simultaneous visual + physical distribution shift 的鲁棒性, 才是真正实用的 generalization。

---

## 10. Limitations 和未来方向

paper 自己点出:

1. 假设 robot stationary, end-effector 是唯一有用的 motion source。扩展到 mobile manipulation, 让 whole-body control 是一个 promising direction。这让我想到 RT-2 + mobile base, 以及 HomeRobot, OK-Robot 等工作。
2. 没法 specify "某些区域严格 enforcement"——比如 fragile object 附近必须严格 follow。trajectory sketch 是 uniform guidance, 缺乏 per-region confidence weighting。未来可以设计 "uncertainty map" 或 "constraint region" 作为额外 channel。
3. 2D representation 的 fundamental ambiguity: 即使 2.5D 加了 height, 还是损失了 6-DoF pose 信息 (gripper orientation)。一个 2D line + scalar height 无法表达 wrist rotation, 这对某些 task (e.g. screwdriver) 会是瓶颈。

我自己的额外思考:

- **Multi-modal trajectory conditioning**: 现在 sketch 是确定的一条 curve。如果让 policy condition on 一个 trajectory distribution (e.g. 多个 candidate sketches), 可能更鲁棒。
- **Sketch hierarchy**: coarse sketch + fine sketch 两层, 让 policy 在 coarse 层做 high-level planning, fine 层做 low-level adjustment。
- **Self-generated sketches**: 让 policy 自己生成 sketch (类似 planner-actor 分离), 或用 VLM 在 loop 里实时 refine sketch。这条线与 UniPi, SuSIE 等 generative planner 工作有连接。
- **3D trajectory sketch**: 直接用 3D point cloud 或 NeRF / Gaussian Splatting 上的 trajectory, 而非 2D image。这避免 camera projection 的信息损失。
- **Sketch + language joint conditioning**: 现在是 either/or, 但 language 提供 semantic context, sketch 提供 motion detail, 联合应该更强。

---

## 11. 跨 paper 的联想

读这篇 paper 时我脑中不断浮现一些相关工作:

- **RT-1 / RT-2** (Brohan et al. 2023b, 2023a): 直接对比 baseline, RT-1 是 transformer policy 的 scale 化, RT-2 是 VLA 把 VLM 知识 transfer 到 robot。RT-Trajectory 在同一 backbone 上换 conditioning, 干净地 isolate 了 conditioning representation 的影响。https://robotics-transformer1.github.io/, https://robotics-transformer2.github.io/
- **Code as Policies** (Liang et al. 2022): 用 LLM 生成 code 来执行 robot task, 是 trajectory generation 的一种自动化方式, paper 里直接 adopt 作为 sketch 生成器之一。https://code-as-policies.github.io/
- **PaLM-E** (Driess et al. 2023): embodied multimodal LLM, paper 用它做 sketch 生成。https://palm-e.github.io/
- **CLIPort** (Shridhar et al. 2021): language + semantic segmentation 的 two-stream policy, 也是 mid-level conditioning 的例子, 但 object-centric 而非 trajectory-centric。https://cliport.github.io/
- **VIMA** (Jiang et al. 2023): multimodal prompt (text + image) conditioning, 在 prompt space 上更 general, 但仍然以 object specification 为主, 不强调 motion。https://vimalabs.github.io/
- **UniPi / SuSIE** (Du et al. 2023): 用 video diffusion model 生成未来 trajectory 视频, 然后 policy follow。这是 video-conditioned policy 的 extreme 版本, paper 中提到 video conditioning 是 over-specified。RT-Trajectory 可以看作 UniPi 的"省略中间步骤"版本——直接 sketch 而非 full video。https://universal-policy.github.io/
- **GenAug** (Chen et al. 2023): 用 generative augmentation 增强 robot data generalization, 是另一个 generative model × robotics 的交叉。
- **BC-Z** (Jang et al. 2021, 2022): 早期 language + goal image conditioning 的 zero-shot generalization 工作, paper 里多次对比。
- **Play-LMP** (Lynch et al. 2019): 从 play data 学 latent plan, goal-conditioned, 是 latent plan 的早期代表。
- **VoxPosPoser** (Huang et al. 2023): 用 LLM 生成 3D value map 作为 motion constraint, 与 RT-Trajectory 在精神上接近 (都生成 mid-level motion representation), 但 VoxPoser 是 3D spatial value map, RT-Trajectory 是 2D+height trajectory。https://voxposer.github.io/
- **RoboCat** (Bousmalis et al. 2023): self-improving foundation agent, 通过 multi-task 训练实现 generalization, 也是 Google DeepMind 的 sister work。

还有一个很深的 conceptual link: 这篇 paper 的核心论点——"generalization 取决于 conditioning representation 是否让 test input 落在 training manifold 近邻"——和 **classifier-free diffusion guidance** 中 conditional vs unconditional 的 manifold navigation 有精神上的呼应。在 diffusion 里, guidance strength 控制我们多靠近 conditional manifold; 在 RT-Trajectory 里, trajectory sketch 的 coarseness 控制我们多 rigidly 锚定到 specific motion。两者的本质都是 representation 选择决定了 interpolation regime。

另一个方向上的联想是 **modality gap** (Liang et al. 2022 在 ICML): 不同 modality (image, language) 的 embedding space 在高维中有 geometric gap, 无法 trivially align。RT-Trajectory 实际上是承认 language manifold 和 motion manifold 之间存在 gap, 然后用 trajectory sketch 作为"桥接 manifold"——它既 motion-aligned (因为是真实 motion 的投影), 又 image-aligned (因为渲染成 2D image 给 EfficientNet 处理)。

---

## 12. 我对这篇 paper 的整体评价

我自己的 take: 这篇 paper 的核心 insight 是"用什么 representation 做 conditioning"这件事不是工程细节, 而是 generalization 的第一性原理。它做了一个非常 clean 的实验对照——同一 RT-1 backbone, 同一 training data, 只换 conditioning modality (language / goal image / trajectory sketch), 就看到 67% vs 17% vs 26% 的巨大 gap。这种 controlled comparison 让结论可信度极高。

trajectory sketch 这个 representation 之所以 work, 是因为它 straddles 两个 manifold: motion manifold (它确实是真实 end-effector trajectory 的投影, 与训练数据的 motion 几何 align) 和 image manifold (它是一张 2D image, 能被 EfficientNet 自然处理)。这种 "straddling" 是 mid-level representation 的威力——它既不像 language 那样抽象远离 motion, 也不像 full video 那样高维难学。

paper 最让我 impressed 的不是 67% 这个数字, 而是 (a) hindsight labeling 让方法完全 scalable, (b) 4 种不同 sketch generation 方法都 work 说明 trajectory 是个 robust interface, (c) Fréchet distance 分析把 "generalization" 量化成可测量的几何量, (d) visual prompt engineering 这个 emergent capability指向一个全新的 robot-policy interaction paradigm。

可以 critique 的地方: 2D 投影丢掉的信息太多, 只能处理 stationary manipulation, 只能处理简单 gripper interaction (open/close), 对 multi-finger dexterous manipulation 不直接适用。但这些都不影响核心 thesis——conditioning representation 决定 generalization manifold——这个 thesis 是 transferable 的。

我很看好这条线, 未来如果把 trajectory sketch 换成 3D point trajectory, 或 4D video (在 RT-2 style VLA 框架里直接用 video token 作 conditioning), 再结合 retrieval-augmented demonstration, 可能就是 robotics foundation model 的下一个台阶。

---

## 主要参考链接

- Paper arXiv: https://arxiv.org/abs/2311.12554
- Project page: https://rt-trajectory.github.io/
- RT-1: https://robotics-transformer1.github.io/
- RT-2: https://robotics-transformer2.github.io/
- Code as Policies: https://code-as-policies.github.io/
- PaLM-E: https://palm-e.github.io/
- VoxPoser: https://voxposer.github.io/
- CLIPort: https://cliport.github.io/
- VIMA: https://vimalabs.github.io/
- UniPi: https://universal-policy.github.io/
- Mediapipe: https://mediapipe.dev/
- Fréchet distance 原始论文 (Eiter & Mannila 1994, discrete 版本): https://www.ams.org/publicoutreach/featurecolumn/fc-2018-09
- Code as Policies (Liang et al. 2022) paper: https://arxiv.org/abs/2209.07753
- Google DeepMind blog 上 RT-Trajectory 的官方介绍: https://www.deepmind.com/blog/rt-trajectory

希望这个 dive 能让你建立起对这篇 paper 的 intuition, Andrej。如果想再 deep dive 某个部分 (比如 EfficientNet 的 zero-init trick 的数学分析, 或 Fréchet distance 的 dynamic programming 实现, 或 CaP prompt 的具体模板), 我可以再展开。
