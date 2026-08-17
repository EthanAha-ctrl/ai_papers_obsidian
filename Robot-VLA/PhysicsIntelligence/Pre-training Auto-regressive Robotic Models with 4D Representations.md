---
source_pdf: Pre-training Auto-regressive Robotic Models with 4D Representations.pdf
paper_sha256: 2c38a85b24b627df24307047154e1f863e2d9f599326d5edd9be78df28274842
processed_at: '2026-08-06T05:34:08-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ARM4R 用人话说

## 一句话总结

**让机器人看人类做饭视频,学会怎么动手。**

听起来很神奇对吧?human做饭和robot抓东西有什么关系?这篇paper的核心就是找到这两者之间的"共同语言"。

## 1. 核心Intuition: 找到一个"翻译器"

想象一下这个场景:

你想训练一个robot去抓杯子。传统方法就是给robot看几千条robot抓杯子的demo,让它学。问题是robot data太贵了,采集一万条demo可能要半年。

那人类做饭视频呢?YouTube上随便download几万个,几乎免费。但怎么用?human hand和robot gripper长得完全不一样,kitchen场景和robot lab也完全不一样。

**ARM4R的insight**: 别看surface长什么样,看**底层几何结构**。

human抓杯子的时候,杯子上某个点(比如杯柄)在3D空间里划出一条trajectory。robot抓杯子的时候,end-effector在3D空间里也划出一条trajectory。这两条trajectory在**4D时空(3D空间+时间)**里是同一类东西——都是3D点随时间移动。

所以如果我们能让模型学会预测"3D点在空间里怎么动",那这个能力既能用到human video上,也能用到robot control上。这就是**shared representation**的威力。

## 2. 为什么这个"翻译"是数学上成立的?

Paper里有个关键论证,用robotics里经典的**Product of Exponentials公式**:

$$g_{1,p}(\theta) = e^{\hat{\xi}_1 \theta_1^d} \cdots e^{\hat{\xi}_i \theta_i^d} g_{i,p}(0)$$

让我humanize一下这个公式在说什么:

- $g_{1,p}(\theta)$: robot身上某个点 $p$ 在base frame(就是机器人底座坐标系)里的位置和朝向
- $\hat{\xi}_j$: 第 $j$ 个joint的"旋转向量"(twist),告诉你这个joint怎么转
- $\theta_j^d$: 第 $j$ 个joint转了多少度
- $e^{\hat{\xi}_j \theta_j^d}$: 这个joint旋转之后产生的4×4变换矩阵
- $g_{i,p}(0)$: 初始时刻点 $p$ 在joint $i$ 坐标系里的位置

**人话翻译**: robot身上任何一点的位置,等于把每个joint的旋转"乘起来"。这是SE(3) Lie group的标准结果(Murray, Li, Sastry那本经典教材里有)。

关键点: **human video里物体运动**和**robot body运动**都是SE(3) transformation。它们在4D空间里是同一类geometric object。这就是为什么pre-training学到的representation可以transfer。

用Karpathy你常说的话: 这是一种**inductive bias**。我们利用robotics的先验知识(SE(3) structure)来design representation,让model不用从头学起。

## 3. 三阶段训练: 像人学技能一样

### Stage 1: 看大量人类视频"打基础"

用Epic-Kitchens100数据集,75,041个kitchen视频,人类做饭、切菜、洗碗等。模型的任务很简单: **预测video里一堆3D点下一帧会跑到哪里**。

这些3D点是怎么来的?用SpatialTracker(另一个paper的工具)把2D video点"lift"到3D。初始化一个 $g \times g$ 的grid,track这些点随时间怎么动。

这一步模型学到什么?**physical world的基本dynamics**——物体怎么动、手怎么抓、东西怎么被推。这个知识是task-agnostic的。

### Stage 2: 在robot场景"适应一下"

Stage 1用的是ego-centric human video(相机戴在头上,会动)。Robot lab里相机是固定的,而且看到的是robot arm不是human hand。这有distribution shift。

所以Stage 2用少量robot视频(1-2K条,约Stage 1的5-10%),还是做3D point tracking任务。这一步像"let me get used to this new environment"。

### Stage 3: 真正学control

这一步把input从"3D points"换成"robot state",output从"next 3D points"换成"next robot state"。模型架构完全不变,只是换input/output的meaning。

因为Stage 1+2已经学到了"3D空间中点怎么随时间演化"的generic能力,Stage 3只需要把这个能力specialize到robot state space。

**类比**: 就像你学了开车(Stage 1),然后换一辆新车适应一下(Stage 2),最后在陌生路上开(Stage 3)。底层skill是transferable的。

## 4. 架构的"小心机"

模型其实不复杂,就三个encoder + 一个causal transformer:

**Language encoder**: frozen CLIP,处理"pick up the cube"这种instruction
**Image encoder**: frozen ViT,处理visual observation
**Point encoder**: 2-layer MLP,处理3D坐标

然后有个attention pooling把image feature和point feature融合。这个attention pooling很关键——它让模型学会"image里哪个pixel对应3D空间里哪个点"。

最后feed进causal transformer做next-token prediction。sequence长这样:

$(z_l, z_0^{obs}, \hat{z}_0, z_l, z_1^{obs}, \hat{z}_1, \cdots)$

每个timestep有三个token: language, current observation, prediction。Loss只在prediction token上算。

这个design和LLM的next-token prediction完全homologous,只是token变成了robot-relevant的representation。我觉得这很Karpathy风格——把LLM的playbook用到robotics上。

## 5. 实验结果: 真的work

### RLBench仿真(Table 1)

ARM4R平均59.47%, 超过PerAct (55.33%), LLARVA (48.33%)。

但有些任务ARM4R明显落后:
- close jar: 24% vs PerAct 60%
- stack blocks: 4% vs PerAct 36%

这两个任务都需要精细rotation和multi-step placement。说明4D point tracking representation对**精细rotation**还是不够expressive。这是limitation。

### Real Kinova robot(Table 2)

ARM4R平均**83.1%**, OpenVLA 37.2%, ATM 6.4%。

这个gap很大。特别是pick & place toy任务:
- ARM4R: 90.7% (spiderman), 94.7% (penguin), 93.3% (pig)
- OpenVLA: 2.7%, 17.3%, 2.7%

为什么ARM4R在toy任务上碾压OpenVLA?我的intuition: toy是非规则形状,OpenVLA的language pre-training难以理解"spiderman toy"的affordance。ARM4R基于geometry,直接看3D结构,对novel object更robust。

ATM只有6.4%特别惨,因为ATM用2D point tracks。Real world有perspective distortion和depth ambiguity,2D信息不够。这强化了paper的核心论点:**3D > 2D for robotics**。

### Cross-robot transfer(Table 4)

Human video pre-train → Kinova video fine-tune → Franka robot control
比直接Kinova → Franka提升19.6%。

这说明human video学到的4D representation有cross-embodiment generalization,因为SE(3) structure是embodiment-agnostic的。

## 6. 诚实的Limitations

Paper自己承认的limitation:

1. **Camera frame vs World frame**: 现在3D tracks在camera coordinate里,把camera motion和object motion耦合在一起。如果用world frame会更好,可以用MonST3R (https://arxiv.org/abs/2410.03825) 或MegaSAM (https://arxiv.org/abs/2412.04463)。

2. **Uniform grid sampling**: 现在第一帧用 $g \times g$ grid均匀采样points。如果用attention机制select task-relevant points会更好,小物体任务分辨率会更高。

3. **Close jar, stack blocks失败**: 精细rotation任务表现差。需要force-aware representation或者更expressive的action space。

## 7. 更大的picture: Robotics FM路线之争

这篇paper让我看到一个tension:

**VLA路线**(OpenVLA, RT-2, π0): 用LLM的reasoning + world knowledge,然后fine-tune到robot
**Physical representation路线**(ARM4R, ATM, MVP): 从physics-grounded representation pre-training

ARM4R的实验暗示: 至少对low-level manipulation, physical representation route可能更effective。Language pre-training学到的是"什么是什么",physical pre-training学到的是"东西怎么动"。

但终极的robotics foundation model估计是hybrid: LLM做high-level planning("先把A放到B上,然后打开drawer"),physical representation model做low-level control(具体怎么抓、怎么放)。

这让我想到Karpathy你在Tesla的autonomous driving work——也是类似问题: 如何从大规模visual data learn driving policy?ARM4R的4D representation approach可能也有启发: 3D point tracks across time可以作为driving scene的compact representation。

## 8. 与World Models的关联

最近world model很热(DreamerV3, Genie, Sora)。ARM4R本质上是个**implicit world model**: 预测future 3D point tracks = 预测physical world未来state。

区别: ARM4R用structured 4D representation,不是raw pixels。这更sample efficient,也更interpretable。

可以想象下一步: 用ARM4R-style architecture做world model,在imagination里plan robot actions。这是model-based RL的思路,但用learned 4D world model替代hand-crafted dynamics model。

## 9. 我会怎么extend这个work

如果我来做next step,几个方向:

1. **World frame tracks**: 用MonST3R生成world coordinate的tracks,解决camera motion entanglement。这是paper自己指出的limitation。

2. **Object-centric points**: 用SAM (Segment Anything) segment出object,然后在每个object上sample points。这样representation是object-centric的,更sample efficient。

3. **Dense 3D motion field**: 不用sparse grid,用neural field表示dense 3D motion。类似Dynamic Gaussian Splatting但用于robotics。

4. **Force-aware**: 加上force-torque sensing,处理contact-rich manipulation。这对screw bulb这种任务必要。

5. **Scale up data**: 从76K video扩展到Ego4D的3000小时,甚至YouTube manipulation videos。如果scaling law成立,效果应该更好。

6. **Hybrid with LLM**: LLM planner + ARM4R controller。LLM分解long-horizon task成sub-goals,ARM4R执行每个sub-goal。

## 10. Reference links

- ARM4R project: https://arm4r.github.io/
- SpatialTracker (生成3D pseudo-labels): https://arxiv.org/abs/2404.04319
- ATM (2D baseline): https://arxiv.org/abs/2401.00025
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0-FAST: https://arxiv.org/abs/2501.09747
- LLARVA (同组前作): https://arxiv.org/abs/2406.11815
- Epic-Kitchens100: https://epic-kitchens.github.io/2019/
- OpenX: https://arxiv.org/abs/2310.08864
- Track2Act: https://arxiv.org/abs/2405.01527
- 3D-VLA: https://arxiv.org/abs/2403.09631
- MonST3R: https://arxiv.org/abs/2410.03825
- MegaSAM: https://arxiv.org/abs/2412.04463
- Murray, Li, Sastry textbook (Product of Exponentials): http://www.cds.caltech.edu/~murray/mlswiki/
- RLBench: https://arxiv.org/abs/1909.12271
- PerAct: https://arxiv.org/abs/2209.05551
- RVT-2: https://arxiv.org/abs/2406.08545
- Octo: https://arxiv.org/abs/2405.12213

## 一句话再总结

ARM4R告诉我们: **让模型先在人类视频里学会"东西在3D空间怎么动"这个universal skill,再specialize到robot control**。这个skill因为是几何的、physics-grounded的,所以能跨embodiment transfer。这比用language pre-training更适合low-level manipulation,因为language学到的是semantic不是dynamics。

---

# ARM4R: Pre-training Auto-regressive Robotic Models with 4D Representations — 深度解读

## 1. 核心Intuition: 为什么用4D representations做pre-training

这篇paper的核心insight其实非常漂亮。我们思考一下robotics pre-training面临的根本困难: robotic data稀缺, 而human video data海量。但直接用human video pre-training robotic model面临两个gap: **embodiment gap**(human hand vs robot gripper)和**task gap**(human kitchen activity vs robot manipulation)。

ARM4R的trick在于: 找到一个**representation space**, 这个space同时满足:
- 可以从human video中unsupervised地extract
- 与robot的low-level control state存在shared geometric structure(可由linear transformation映射)

这就是4D representations = **3D point tracks across time**。这个idea的妙处在于:

**human video** → 3D point tracks (3D空间中的点随时间轨迹)
**robot state** → 也是3D空间中点随时间的演化(机器人的关节、end-effector、attached objects在3D中运动)

这两者本质上都是**4D时空中的点轨迹**, 且都通过rigid body transformations (SE(3)) 描述, 这构成了shared geometric structure。

让我用具体公式来build这个intuition。

## 2. 关键公式与几何论证

### 2.1 4D representation的定义

Paper中Eq.(1)定义3D point tracking:

$$p_t = \{(x_{jt}, y_{jt}, z_{jt}) \mid 0 \leq j < n\}$$

变量解释:
- $p_t$: 时刻 $t$ 的3D point set
- $j$: 第 $j$ 个被track的point, 共 $n$ 个(通常 $n = g^2$, 即 $g \times g$ 的grid)
- $t$: 时间帧index, $0 \leq t < T$
- $(x_{jt}, y_{jt}, z_{jt})$: point $j$ 在时刻 $t$ 在camera coordinate frame下的3D坐标
- 关键约束: 第 $j$ 个点在不同帧中是**同一个物理点**(identity fixed across all frames)

这就是4D: 3D space + 1D time。

### 2.2 关键定理: Robot state与3D points的linear transformation关系

Paper Section 3.1的核心论证。考虑n-DoF open-chain manipulator, 参考configuration $\theta_1 = \theta_2 = \cdots = \theta_n = 0$, 命令到新位置 $\theta_1^d, \theta_2^d, \cdots, \theta_n^d$。

取robot body上任意一点 $p$, 位于joint $i$ 与 $i+1$ 之间, 用 $g_{i,p}(0)$ 描述其在joint $i$ frame下的变换。则 $p$ 在base frame下的新位置:

$$g_{1,p}(\theta) = e^{\hat{\xi}_1 \theta_1^d} \cdots e^{\hat{\xi}_i \theta_i^d} g_{i,p}(0)$$

变量解释:
- $g_{1,p}(\theta)$: point $p$ 在base frame (frame 1) 下的pose, 是4×4 SE(3) matrix
- $\hat{\xi}_j$: 第 $j$ 个joint的twist (在Lie algebra $\mathfrak{se}(3)$ 中), 这是一个6维向量对应的4×4矩阵
- $\theta_j^d$: 第 $j$ 个joint的目标角度
- $e^{\hat{\xi}_j \theta_j^d}$: matrix exponential, 即joint $j$ 旋转 $\theta_j^d$ 产生的SE(3) transformation
- $g_{i,p}(0)$: 初始configuration下 $p$ 在joint $i$ frame中的pose

这就是**Product of Exponentials formula** (Murray, Li, Sastry经典教材中的公式)。这个公式告诉我: robot body上的点经过的状态变化是**SE(3) matrix的连乘**, 即linear transformation of robot state。

**关键insight**: 
- human video中的3D point tracks: 物理世界中的点随时间演化, 物体运动本质也是SE(3) transformations
- robot state: 通过forward kinematics得到end-effector或body上点的3D位置, 也是SE(3) transformations of robot joint states
- 两者shared geometric structure: 都是4D时空中的3D点轨迹, 通过linear transformation(实际上是affine, 但paper近似为linear)连接

这就是为什么pre-training on human video 4D representations能transfer到robot control的数学基础。这个论证其实有更深层的意味: 因为机器人是open-chain rigid body, 它的状态演化空间是SE(3)的子群积, 与世界中物体运动的SE(3) trajectory有相同的几何类型。

### 2.3 Architecture的核心forward pass

Eq.(2): point tracking阶段
$$\pi(l, i_{t-C+1:t}, p_{t-C+1:t}) \to p_{t+1}$$

Eq.(3): control fine-tuning阶段
$$\pi(l, i_{t-C+1:t}, s_{t-C+1:t}) \to s_{t+1}$$

变量解释:
- $l$: language instruction (e.g. "pick up the cube")
- $i_{t-C+1:t}$: 从 $t-C+1$ 到 $t$ 时刻的image observation序列, $C$ 是context window size
- $p_{t-C+1:t}$: 同样时间段的3D point tracks
- $s_{t-C+1:t}$: 同样时间段的robot proprioceptive state
- $p_{t+1}$ / $s_{t+1}$: 预测下一时刻的point tracks / robot state

注意Eq.(2)和Eq.(3)的形式完全一致, 只是输入输出从points切换为states。这正是shared geometric structure所允许的——同一个auto-regressive backbone可处理两种任务。

### 2.4 Loss function

Eq.(4):
$$\mathcal{L}(\hat{p}_{t+1}, p_{t+1}) = \frac{1}{n} \|\hat{p}_{t+1} - p_{t+1}^*\|_1$$

- $\hat{p}_{t+1}$: 预测的3D points
- $p_{t+1}^*$: ground truth 3D points
- $n$: 点的数量
- $L_1$ loss: 比$L_2$更robust to outliers, 在tracking任务中常用

注意control fine-tuning阶段loss function形式不变, 只是把 $p$ 替换为 $s$ (robot state), 这是优雅之处。

## 3. 三阶段训练流程深度解析

### Stage 1: Human video pre-training on Epic-Kitchens100

- 数据: 75,041个episode (从75,886个中过滤掉>256帧的)
- 采样率: 10fps (从原始50fps降采样)
- pseudo-label生成: 用SpatialTracker (Xiao et al., 2024)生成3D point tracks
- grid初始化: $g \times g$的square grid在第一帧, n = $g^2$ points
- 坐标系: **camera coordinate frame**(这是limitation, paper后面也提到)

**为什么Epic-Kitchens100合适**:
- egocentric视角, 与机器人first-person视角相似
- 97个verbs × 300个nouns组合, 丰富的human-object interaction
- 包含kitchen场景中大量的pick, place, push, pour等动作, 这些与robot manipulation任务高度相关

**关键细节**: pseudo-labels在camera frame中, inherently capturing **both object and camera motion**(因为ego-centric video camera也在动)。这是egocentric数据的特殊性。

### Stage 2: Robot video fine-tuning for 3D point tracking

- 数据量: 约5-10% of Stage 1 (1-2K demonstrations)
- 任务: 与Stage 1相同的3D point tracking, 但是用robot setup的视频
- 目的: 弥补embodiment gap和camera dynamics gap

这一步很关键。Stage 1学到的是human video中"动相机+动object"的4D pattern, Stage 2学到的是robot setup中"静相机+动object+动robot"的4D pattern。这一步实现了distribution shift的平滑过渡。

### Stage 3: Robot control fine-tuning

- 数据: 每个task约190 episodes per variation
- 替换: input points → robot state $s_t$, output points → next state $s_{t+1}$
- 预测: next 16 actions, 但execute只第一个(action chunking idea)
- Control mode: end-effector control, $\mathbf{x} = (x, y, z, \theta_x, \theta_y, \theta_z) + \text{binary gripper}$

这个action chunking trick让我想到RT-2和π0的设计——预测一段轨迹但只执行第一步, 可以平衡short-term accuracy和long-term planning。

## 4. Architecture细节解析

### 4.1 Encoder结构

**Language Encoder**: 
- Frozen CLIP text encoder (LAION-2B训练)
- + learnable linear projection → $z_l$

**Image Encoder**:
- ViT (Vision Transformer), frozen during ARM4R training
- Pre-trained with CrossMAE on ImageNet + OpenX
- 这个pre-training组合很巧妙: ImageNet保证general visual representation, OpenX保证robot scene understanding

**4D Representation Encoder**:
- 2-layer MLP处理point coordinates → $z_t^{pts}$
- 与 $z_t^{im}$ 通过**attention pooling**结合 → $z_t^{obs}$
- 单独的MLP编码next timestep的points → $\hat{z}_t$

**Attention pooling的妙处**: 
- 用point features as queries, image features as keys/values
- 自动学习"哪个image region对应哪个point"
- 提供了spatial grounding, 让模型知道3D空间中的点对应image中的哪个区域

**Causal Transformer**:
- ViT-Base, 随机初始化
- Input sequence: $(z_l, z_0^{obs}, \hat{z}_0, z_l, z_1^{obs}, \hat{z}_1, \cdots)$
- Next-token prediction, loss只在 $\hat{z}_t$上计算
- 这是一个decoder-only style的auto-regressive design

### 4.2 Multi-view adaptation

控制fine-tuning阶段如何处理多视角:
- 每个view单独过image encoder → $z_t^{im}$
- 每个view用attention pooling与state embedding融合 → $z_t^{obs}$
- project到half hidden dim (768 → 384)
- concatenate两个view → final image token

这种设计保留了view-specific信息又进行了fusion, 但相比cross-attention fusion显得简单。后续工作可以探索更sophisticated的multi-view fusion。

## 5. 实验结果深度分析

### 5.1 RLBench结果 (Table 1)

ARM4R平均成功率59.47%, 超越PerAct (55.33%), LLARVA (48.33%), ManiGaussian (48.00%)。

**值得注意的细节**:
- 在"put money"任务上: ARM4R 92.0% vs PerAct 44%——大幅领先
- 在"open drawer"上: 88.8% vs PerAct 80%
- 在"close jar"上: 24.0% vs PerAct 60%——明显落后! 这是个interesting failure case
- 在"stack blocks"上: 4.0% vs PerAct 36%——也是落后

**为什么close jar和stack blocks表现差?** 我的推测:
- close jar需要精确的rotation manipulation(screw motion), ARM4R的end-effector representation可能不够expressive
- stack blocks需要multi-step precise placement, 这与ARM4R的predict-first-action策略可能冲突
- 这些任务的失败mode反映了4D point tracking representation在精细rotation任务上的局限性

### 5.2 Real robot Kinova结果 (Table 2)

ARM4R平均83.1%, 远超OpenVLA (37.2%)和ATM (6.4%)。

特别impressive的是:
- pick cube (yellow): 92.6% vs OpenVLA 77.8%
- pick cube (cyan): 100% vs OpenVLA 45.8%
- pick cube (green): 95.8% vs OpenVLA 91.7%
- pick spiderman then place: 90.7% vs OpenVLA 2.7%!

**ATM为什么这么差(6.4%)?** ATM用2D point tracks, 在real world中受perspective distortion和depth ambiguity影响。这强化了paper的核心论点: 3D > 2D for robotics。

**为什么ARM4R在pick & place toy上(90.7%)远超OpenVLA(2.7%)?** 
- Toy是非规则形状物体, VLA基于language pre-training难以理解其affordance
- ARM4R基于geometry的4D representation直接编码spatial dynamics, 不依赖object的semantic category
- 这反映了low-level geometric representation的优势: 对out-of-distribution objects更robust

### 5.3 Cross-robot generalization (Table 4)

Epic pre-train → Kinova fine-tune → Franka control: 73.3% pick, 49.3% stack, 65.3% destack
Epic pre-train → Kinova fine-tune (point track) → Franka control: 93.3%, 56.0%, 97.3%
Kinova fine-tune → Franka control: 81.3%, 52.0%, 73.3%

**Human video pre-training带来19.6%的平均提升**在cross-robot transfer上。这证明4D representation确实有cross-embodiment的geometric invariance。

### 5.4 Robustness分析 (Table 5, 6)

Dynamic disturbance (移动cube): 性能轻微下降(96.0% → 92.0%)
Dim light: 中等下降(96.0% → 86.7%)
Background distractor: 几乎无影响(96.0% → 94.7%)
Tabletop distractor: 较大下降(96.0% → 81.3%)

**Tabletop distractor影响最大**, 说明attention pooling虽然能filter背景, 但对前景distractor仍敏感。这指向未来工作: 更显式的object-centric representation。

## 6. 与相关工作的深度对比

### 6.1 ARM4R vs VLA (OpenVLA, RT-2, π0-FAST)

VLA: 用language decoder pre-trained on VQA/captioning → robot action prediction
- 优势: 利用LLM的reasoning能力, high-level task understanding
- 劣势: 高level pre-training objective与low-level control不匹配

ARM4R: 用3D point tracking pre-trained on human video → robot control
- 优势: pre-training objective直接对应physical world dynamics, 与robot state shared geometric structure
- 劣势: 缺乏high-level reasoning (e.g. "first pick up A, then B, finally C")

paper的实验证明: 在low-level manipulation任务上, ARM4R > VLA, 暗示当前VLA的language pre-training对精细控制帮助有限。但这并不意味着VLA框架错了, 而是**pre-training objective需要更physical**。

### 6.2 ARM4R vs ATM (Any-Point Trajectory Modeling)

ATM (Wen et al., 2024):
- 用2D point tracks on small human demo set
- Hierarchical: track transformer + policy network
- 跨embodiment transfer

ARM4R vs ATM关键区别:
1. **3D vs 2D**: ARM4R用monocular depth lifting → 3D tracks; ATM停留在2D
2. **Data scale**: ARM4R用76K Epic-Kitchens videos大规模pre-training; ATM用small demo set
3. **Architecture**: ARM4R用single auto-regressive model; ATM用hierarchical两个网络
4. **Robot state integration**: ARM4R直接预测robot state(共享geometric structure); ATM用track condition policy

实验上ARM4R碾压ATM(83.1% vs 6.4% in Kinova), 证明3D + scale是关键。

### 6.3 ARM4R vs Track2Act

Track2Act (Bharadhwaj et al., 2024):
- 在Epic-Kitchens100和Something-Something-v2上训练2D point tracker
- Re-purpose for robotic manipulation

相似点: 都用Epic-Kitchens大规模人类视频
不同点: Track2Act用2D, ARM4R用3D。ARM4R的论证是3D提供更physical的grounding, 这与机器人3D control空间自然对齐。

### 6.4 ARM4R vs RVT/RVT-2

RVT (Goyal et al., 2023) and RVT-2 (Goyal et al., 2024):
- 用RGB-D图像reconstruct point cloud
- 直接用3D scene information
- 需要3D传感器输入

ARM4R优势: 只需monocular video (无需RGB-D), 通过pre-training implicit学习3D结构, 更scalable to in-the-wild data collection。

### 6.5 ARM4R vs SpatialTracker

SpatialTracker (Xiao et al., 2024)被ARM4R作为pseudo-label generator:
- Lifts 2D pixels to 3D via monocular depth
- Iteratively refines with as-rigid-as-possible motion priors
- 长程3D point tracking, 鲁棒to occlusion

ARM4R的clever之处: 不需要训练新的3D tracker, 直接用off-the-shelf SpatialTracker生成pseudo-labels, 然后pre-train一个auto-regressive model来预测这些tracks。这绕过了3D tracker本身的速度问题, 让inference更快。

## 7. 我对这篇paper的critique和extension ideas

### 7.1 Strengths

1. **Mathematical elegance**: SE(3) product of exponentials论证很漂亮, 提供了representation transfer的几何基础
2. **Practical scalability**: 只需monocular video + off-the-shelf depth estimator, 数据收集成本极低
3. **Strong empirical results**: 在real robot上83.1% average success rate, 远超VLA baselines
4. **Cross-robot transfer**: 验证了representation的geometric invariance

### 7.2 Limitations and future directions

1. **Camera frame vs World frame**: paper limitation部分提到当前tracks在camera coordinate, 难以disentangle object和camera motion。未来工作可以用MonST3R (Zhang et al., 2024)或MegaSAM (Li et al., 2024b)生成world frame的tracks。这对应了**invariance to camera intrinsics and ego-motion**的关键需求。

2. **Uniform grid sampling**: 当前在第一帧用 $g \times g$ uniform grid采样points。可以改进为**task-relevant point selection**——比如用attention机制select运动物体上的points。这能提高小物体任务的resolution(对应close jar失败案例)。

3. **Multi-view fusion**: 当前是简单的concat, 可以用cross-view attention或epipolar transformer更深度融合multi-view信息, 提高occlusion robustness。

4. **Lack of high-level reasoning**: ARM4R是low-level controller, 缺乏long-horizon planning。可以与LLM-based planner结合: LLM提供sub-goal sequence, ARM4R执行每个sub-goal的low-level control。

5. **Point cloud granularity**: 当前是grid-based sparse points。可以探索**dense 3D motion field**或**neural radiance field dynamics**作为representation。

6. **Action representation**: 当前用absolute end-effector pose。可以尝试**impedance control**或**force-torque aware** representation, 处理需要contact-rich manipulation的任务(screw bulb的失败案例)。

7. **Pre-training data scale**: 76K videos相比LLM/VLM的billions还很小。可以扩展到Ego4D (3000小时), MECCANO, 或者YouTube的manipulation videos。预训练数据规模可能遵循scaling law。

### 7.3 更深的思考: Robotics foundation model的路线之争

这篇paper让我思考robotics foundation model的两个流派:

**流派A (VLA route)**: LLM/VLM + robot data fine-tuning
- 代表: OpenVLA, RT-2, π0
- 核心: 利用LLM的semantic reasoning + 世界知识
- 瓶颈: language pre-training objective与low-level control不匹配

**流派B (Physical representation route)**: 从physics-grounded representation pre-training
- 代表: ARM4R, ATM, Track2Act, MVP
- 核心: 在physical world dynamics上pre-training, geometry-aware
- 瓶颈: 缺乏high-level reasoning, semantic understanding

ARM4R的实验暗示: 至少对low-level manipulation, 流派B可能更有效。但终极的robotics foundation model可能是两者的hybrid: physical representation提供low-level grounding, language model提供high-level planning。

### 7.4 与Karpathy您自己工作的关联

您提到的"software 2.0"思想——用神经网络替代手写规则——与ARM4R有共鸣。ARM4R某种程度上是**learned physics engine**: 通过观察大量human video, 学习物理世界的implicit dynamics, 然后transfer到robot control。

这让我想到您在Tesla的work on driving: 同样面临"如何从大规模visual data中learn driving policy"的问题。ARM4R的4D representation approach可能也对autonomous driving有启发: 3D point tracks across time可以作为driving scene的compact representation, 替代或补充BEV (bird's eye view)。

### 7.5 与world models的关联

最近世界模型的工作越来越受到关注。ARM4R本质上是一个**implicit world model**:
- 预测future 3D point tracks = 预测physical world的未来state
- 与DreamerV3, Genie, Sora等generative world models有思想上的联系
- 区别: ARM4R用structured 4D representation, 而非raw pixels

可以想象**4D world model + control**: 用ARM4R-style architecture作为world model, 在imagination中plan robot actions。这可能是下一篇大paper的方向。

## 8. Reference links

- ARM4R project page: https://arm4r.github.io/
- SpatialTracker (用于生成3D point tracks的pseudo-labels): https://arxiv.org/abs/2404.04319
- ATM (Any-Point Trajectory Modeling): https://arxiv.org/abs/2401.00025
- Track2Act: https://arxiv.org/abs/2405.01527
- OpenVLA: https://arxiv.org/abs/2406.09246
- LLARVA (same group's previous work): https://arxiv.org/abs/2406.11815
- π0-FAST: https://arxiv.org/abs/2501.09747
- Epic-Kitchens100 dataset: https://epic-kitchens.github.io/2019/
- OpenX Embodiment: https://arxiv.org/abs/2310.08864
- PerAct (RLBench baseline): https://arxiv.org/abs/2209.05551
- RLBench: https://arxiv.org/abs/1909.12271
- RVT: https://arxiv.org/abs/2306.13096
- RVT-2: https://arxiv.org/abs/2406.08545
- 3D-VLA: https://arxiv.org/abs/2403.09631
- RoboPoint: https://arxiv.org/abs/2406.10721
- Octo: https://arxiv.org/abs/2405.12213
- Co-Tracker (ATM baseline): https://arxiv.org/abs/2307.07635
- Ego4D: https://arxiv.org/abs/2110.07058
- MonST3R (future work mention): https://arxiv.org/abs/2410.03825
- MegaSAM (future work mention): https://arxiv.org/abs/2412.04463
- Murray, Li, Sastry "A Mathematical Introduction to Robotic Manipulation" (Product of Exponentials公式来源): http://www.cds.caltech.edu/~murray/mlswiki/
- BLIP-2 (frozen VLM相关): https://arxiv.org/abs/2301.12597
- CLIP: https://arxiv.org/abs/2103.00020
- CrossMAE: https://arxiv.org/abs/2310.04930
- LoRA: https://arxiv.org/abs/2106.09685

## 9. 总结

ARM4R是robotics pre-training领域一篇重要的work, 它通过一个deeply geometric insight(4D representations在human video和robot control之间shared SE(3) structure)解决了一个关键问题: 如何用海量human video pre-train robotic model。

核心的Product of Exponentials论证虽然简短, 但抓住了robotics control的本质——robot state演化通过SE(3) Lie group, 与physical world中object motion通过SE(3)在数学上homologous。这种geometric homology允许representation transfer。

实验结果惊艳: 在real Kinova robot上83.1% success rate远超OpenVLA的37.2%, 证明low-level physical representation pre-training可能比language pre-training更适合manipulation control。

未来方向可能是:
1. World frame 3D tracks (解决camera motion entanglement)
2. 与LLM planner结合 (补全high-level reasoning)
3. Scale up pre-training data (Ego4D, YouTube)
4. Dense 3D motion field替代sparse grid points
5. Force-aware representation for contact-rich tasks

ARM4R为robotics foundation model的physical representation route树立了一个strong baseline, 期待后续工作在此基础上推进。
