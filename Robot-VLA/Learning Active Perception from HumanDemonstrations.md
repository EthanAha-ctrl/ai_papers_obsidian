---
source_pdf: Learning Active Perception from HumanDemonstrations.pdf
paper_sha256: fa8d80dd89269e0996a56a6ecb5ca852a9c91741e1c4dcd2b7570e62aa1724dc
processed_at: '2026-08-05T12:40:46-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ViA 用人话讲

## 一句话总结

这篇paper说了一件特别朴实的事：**人拿东西的时候会动脑袋看，但现在的robot都不动脑袋**。ViA让robot学会像人一样主动转头去找东西、对准东西、盯住东西。

就这么简单一个idea，但因为执行起来全是engineering nightmare，所以没人做。Stanford这帮人把hardware、teleoperation、policy learning三层全打通了，终于做出来了。

## 为什么这是个真问题

你想象一下从bag里拿banana这个task。你会怎么做？

1. 先scan一下桌面，找到bag在哪
2. 用手把bag口撑开
3. 脑袋凑过去peek一下，确认banana在哪个位置
4. 盯住banana，手伸进去抓

这四个动作里，三个都涉及"动脑袋"。你的gaze是active的，是task-driven的，是跟着task stage变化的。

但你看看现在的imitation learning系统都怎么做的：
- **ALOHA / Mobile ALOHA**: 两个wrist cameras，camera跟着arm走，viewpoint完全被arm action决定，不是被perception need决定
- **UMI / Diffusion Policy**: 一个fixed的third-person camera，永远从同一个角度看
- **Bunny-VisionPro**: 也是wrist-mounted

这就像你戴着一个head mount，脑袋永远不能动，只能靠手的位置变化来"间接"改变视角。在simple task上还能凑合，一旦有occlusion就完蛋——wrist camera被shelf挡住、被bag挡住、被自己挡住，policy直接瞎了。

更根本的问题是**observation mismatch**: human在teleoperation的时候natural地shift gaze（因为VR headset跟踪head），但robot在deployment的时候用的是fixed camera，看到的view和human当时看到的view不一致。你让policy从demonstration里学什么呢？human的perceptual strategy根本没被记录下来，只有arm action被记录了。

这就像教一个人开车，教练一直在转头看后视镜、看盲区，但学员的脑袋被固定住只能看前方——你只把教练的"手怎么动方向盘"记录下来了，"眼睛怎么观察"这个关键信息全丢了。

## 三个Engineering Challenge

为什么不直接做active perception？因为太难了，有三个坑：

### 坑1: Hardware

Human的gaze control是eye + neck + torso协同的，非常flexible。现有robot neck大多是2-DoF（pan + tilt），这连human neck的motion range都不到，更别说torso的贡献了。

Open Television [14] 用了一个专门的2-DoF neck，ASIMO [11]、HRP-3 [12]也是类似。这些都太limiting了。

### 坑2: VR Teleoperation的Motion Sickness

你想让human teleoperate一个active head camera，最自然的方式就是VR——human戴headset，转头的时候robot的camera跟着转。human直接看到robot看到的东西，shared observation space。

但这里有个killer: **motion-to-photon latency**。

你转头 → VR headset检测到head motion → 发送command给robot → robot neck物理转动 → camera capture新画面 → 传回VR display → 你看到新画面

这整个链条latency轻松超过100ms。而人类VR comfort的threshold是~10ms [59]。超过这个threshold你的visual system和vestibular system打架，你就想吐——这就是motion sickness [55]。

传统approach（Open Television [14]、Chuang et al. [52]）就是这么直接stream RGB的，结果operator戴着VR做了5分钟就想吐，没法scale数据收集。

### 坑3: Gaze Strategy怎么来？

你让robot主动看，看哪里？Prior work有两类approach:
- **Hand-crafted heuristics** [20,21,22]: 设计saliency map、object detector，计算"哪里最值得看"。但每个task都要重新设计，不scalable。
- **RL** [38,39,40,41]: 让robot自己学gaze policy。但real-world RL难deploy，sim-to-real gap大。

## ViA的三个解法

### 解法1: 用6-DoF Arm当Neck

别design专门的humanoid neck了，直接拿一个现成的6-DoF arm（ARX5），末端mount一个iPhone 15 Pro当camera。

这个设计的brilliant之处在于：
- 6-DoF足够approximate human upper body（torso + neck）产生的head motion range
- Off-the-shelf hardware，不用自己design + manufacture
- iPhone同时给你RGB + depth + 自己的VIO pose，省了你一堆sensor fusion的活

总共三个6-DoF ARX5 arm: 一个当neck，两个当manipulation arm，每个arm装fin-ray gripper。简单、cheap、effective。

和Open Television [14]的2-DoF neck对比，ViA的6-DoF neck能做translation（不只是rotation），这对peek into bag这种需要"凑过去"的motion非常重要。

### 解法2: Intermediate 3D Representation — 这是全文最clever的部分

核心insight: **人类对viewpoint latency极度sensitive，但对scene content staleness不sensitive**。

你想想你自己转头看东西的场景。你头转过去的一瞬间，你的眼睛expect立刻看到新视角的内容。但如果新视角里有个object，它位置是10ms前的而不是现在的——你完全察觉不到，你的brain根本care这个。

ViA就是利用这个asymmetry:

**传统方法（同步）**:
```
User转头 → 发command → Robot动 → Camera拍 → 传RGB → VR显示
        ←————————— 100+ ms latency ——————————→
                    User感觉想吐
```

**ViA（异步解耦）**:
```
User转头 → 从point cloud实时render新视角 → VR显示
        ←——— ~7ms，user感觉丝滑 ———→

同时（异步）:
Robot慢慢动 → 更新point cloud → 替换旧的
     ←——— ~100ms，user无感 ———→
```

具体怎么做：

**(1) Point Cloud Construction**

定义world frame W在robot neck base。每个时刻t，iPhone给一个RGB-D frame + 自己的relative pose $^{H}T_{\text{iphone}}(t)$。Robot neck的joint positions通过forward kinematics给初始head pose $^{W}T_H(t_0)$。组合起来:

$$^{W}T_H(t) = ^{W}T_H(t_0) \cdot ^{H}T_{\text{iphone}}(t)$$

- $^{W}T_H(t_0)$: 初始时刻robot head在world frame的pose（从joint positions算）
- $^{H}T_{\text{iphone}}(t)$: iPhone自己估计的relative pose（iPhone的VIO）
- $^{W}T_H(t)$: 时刻t head在world frame的pose

用这个pose把RGB-D transform到world frame，得到point cloud $^{W}X(t)$。这是intermediate 3D scene representation。

**(2) Low-Latency View Rendering**

User戴的VR headset实时给head pose，transform到world frame:

$$^{W}T_{\text{user}}(t+k) = ^{W}T_{\text{VR}}(t+k) \cdot T_{\text{height\_offset}}$$

- $^{W}T_{\text{VR}}(t+k)$: VR headset在时刻 $t+k$ 的head pose transform到world frame
- $T_{\text{height\_offset}}$: 高度offset，因为robot neck base和user站立位置有高度差
- $k$: rendering interval，~7ms

用 $^{W}T_{\text{user}}(t+k)$ 从point cloud $^{W}X(t)$ render stereo RGB给VR display。频率~150Hz。

关键: **render用的是user的最新head pose，但scene content用的是稍微旧的point cloud**。user感觉viewpoint更新是instant的（因为是render），scene content稍微旧一点user无感。

**(3) Asynchronous Point Cloud Updating**

Robot neck的head pose以较慢频率更新:

$$^{W}T_H(t+K) = \text{AggregatedUserHeadPose}(t \to t+K)$$

- $K$: 更长的时间间隔，由robot control latency决定，远大于rendering interval $k$
- Aggregated: 不是mirror user每个瞬间head pose（那样robot追不上），而是aggregate一段时间内的head motion再执行

Point cloud以10Hz更新——robot慢慢动camera，慢慢补全scene。

**为什么这个设计work？**

核心是利用了human perception的一个asymmetry。Human visual system对"我转头了但画面没更新"极度sensitive（vestibular-visual conflict → motion sickness），但对"画面里的东西是100ms前的位置"几乎不sensitive。

这就像你用电脑的时候，鼠标移动latency必须<10ms否则你崩溃，但网页内容是1秒前fetch的还是2秒前fetch的，你完全不care。

ViA把这个principle apply到VR teleoperation: viewpoint跟随是"鼠标级别"的low latency（通过rendering），scene content更新是"网页级别"的acceptable latency（通过robot异步）。

这和human brain的机制也有parallel:
- Saccadic masking: saccade期间视觉被suppress，brain不在意这期间的visual content
- VOR (vestibulo-ocular reflex): gaze稳定即使head在动
- 这些都说明visual system对不同types of latency有不同的tolerance

Wilder-Smith et al. [57] 用radiance field做类似的事，但他们没有physical camera control——user不能purposefully移动camera去看task-relevant区域。ViA保留了purposeful camera control，这是manipulation task必须的。

### 解法3: Shared-Observation + Diffusion Policy

**Shared-observation teleoperation**: Teleoperator只看active head camera的rendered view，和robot deployment时的observation完全一样。这样human的perceptual strategy（searching, tracking, focusing）直接被encode在demonstration里。

这个设计很scalable: 不用hand-design gaze strategy，human intelligence直接transfer。ALOHA用joint-to-joint mapping transfer motor strategy，ViA用shared observation transfer perceptual strategy，思路一脉相承。

**Policy架构**: 基于Diffusion Policy [10]。

每个时刻 $t$:
- **Visual input**: $\mathbf{I}_t \in \mathbb{N}_0^{H \times W \times C}$ — active head camera的RGB
- **Proprioceptive state**: $\mathbf{P}_t \in \mathbb{R}^{23}$
  - Neck end-effector pose (position 3 + quaternion 4 = 7)
  - Left arm end-effector pose (7)
  - Right arm end-effector pose (7)
  - Left gripper width (1)
  - Right gripper width (1)
  - Total: $7 \times 3 + 1 \times 2 = 23$

**Visual encoder**: DINOv2 [60] pretrained ViT，提取384-dim CLS token。

为什么DINOv2而不是ResNet或point cloud？
- DINOv2是self-supervised在large-scale unlabeled data上pretrain的，semantic features强
- Active perception需要semantic understanding才能做intelligent gaze——你得先知道"我看到了banana"才能决定"我该凑近看grasp点"
- ResNet-18 pretrained on ImageNet: supervised labels不够semantic rich
- DP3 [63]用point cloud from scratch: 没有pretrained prior，容易hallucinate

**Action output**: 
$$\mathbf{A}_t = \{a_{t+1}, \ldots, a_{t+16}\} \in \mathbb{R}^{16 \times 23}$$

- $n_p = 16$: predict未来16步action
- $n_a = 8$: 只执行前8步（receding horizon control）
- Policy运行频率: 10 Hz

Diffusion Policy predict action sequence而不是single action，对multi-modal action distribution更友好，temporal smoothing效果好。

## 三个Task

### Bag Task
从bag里拿object。三个stage: open bag → peek inside → take out。
- 150 demos, 5 training objects (banana, carrot, dog, shoe, strawberry)
- Test on 2 unseen objects (blue elephant, green avocado), 10 trials
- 考察interactive perception: 用手open bag减少occlusion + 脑袋凑过去看interior

### Cup Task
从shelf A找cup，handover到另一只手，放到shelf B下的saucer上。
- 125 demos, cup随机在upper/lower tier，sauder随机位置
- 20 test rollouts
- 考察viewpoint switching: cup在shelf深处wrist camera被挡，saucer在shelf B下方

### Lime & Pot Task
找lime放进pot，bimanual grasp pot，precise align到trivet上。
- 260 demos, 10个lime + trivet configurations
- 20 test rollouts
- 考察bimanual coordination + precise alignment

## 实验结果里最有意思的发现

### 发现1: 加Wrist Camera反而变差18%

ViA (head only) vs. ViA + wrist cameras: 加了wrist camera平均**下降18.33%**。

这个counter-intuitive的结果怎么解释？作者的hypothesis:
1. Active head camera alone已经task-complete，teleoperator只用这个view就能完成任务
2. Wrist cameras增加input dimensionality但不增加task-relevant info
3. 在occlusion场景下wrist cameras经常被挡，提供的是noise不是signal
4. ~150 demos的low-data regime下，更多input = 更高overfitting风险

**Intuition**: 这就像给人看一道数学题，你已经把题目看清楚了，再给你三个模糊的、被遮挡的、看不清的辅助图，反而让人confused。Data efficiency和input complexity要match——input越informative，需要的data越少；input越noisy，需要的data越多来filter out noise。

这个发现对future robot system design有implication: **不是camera越多越好**。在low-data regime，sparse informative observation可能比dense multi-view observation更好学。只有在data-rich regime，multi-view fusion才真正发挥优势。

### 发现2: Active Head vs. Chest & Wrist: 45%差距

```
ViA (active head):        ████████████████████ 100%
Chest & Wrist cameras:    ████████░░░░░░░░░░░░  55%
                         -45% success rate
```

Chest & wrist camera的failure mode:
- Cup task: right wrist被upper shelf tier完全occlude
- Bag task: fixed chest camera看不到bag interior
- 即使fisheye lens也救不了fixed viewpoint

**Intuition**: Fixed camera就像一个人脑袋被钉住，只能从固定角度看。你能看到的东西是死的，task需要看的东西是活的。Active camera让robot的perception变成活的，能follow task structure。

### 发现3: DINOv2 >> ResNet >> Point Cloud

```
ViA (DINOv2 ViT):     ████████████████████ Best
ResNet-DP:           ███████████████░░░░░ Worse  
DP3 (Point Cloud):   █████████░░░░░░░░░░░ Worst (hallucination)
```

DP3的failure mode是**hallucination**——policy misinterpret scene，direct arm到empty shelf section。在bag task完全fail因为bag handle grasping不precise。

**Intuition**: Active perception需要semantic-level understanding。你得知道"这是个cup"、"cup可能在shelf上"才能做intelligent search。Point cloud虽然geometrically precise但semantically poor，在partial observation + occlusion下容易confuse。DINOv2的self-supervised pretraining在大规模unlabeled data上学到了rich semantic features，让policy能infer"我看到的这个模糊的轮廓是个cup的边缘"，这是active search的foundation。

## 这个工作的深层意义

### Principle 1: Perception-Action Time Scale Decoupling

ViA的intermediate 3D representation揭示了一个broadly applicable的principle: 当fast feedback和slow actuation共存时，可以通过intermediate representation decouple它们的time scale。

- Viewpoint更新: fast time scale (rendering, 150Hz)
- Scene content更新: slow time scale (robot observation, 10Hz)

这个principle可以apply到很多地方:
- Autonomous driving: sensor fusion fast, planning slow
- Surgical robot: visualization fast, actuation slow  
- Haptic feedback: rendering fast, force actuation slow

### Principle 2: Shared-Observation Transfer

不是transfer motor strategy，而是transfer perceptual strategy。

ALOHA transfer的是"手怎么动"，ViA transfer的是"眼睛怎么动"。这打开了imitation learning一个新维度: 把human的perceptual intelligence transfer给robot。

Future direction: 如果能用AR眼镜直接capture human在日常生活中自然的gaze behavior [4]，连teleoperation都不需要了，scale会大得多。

### Principle 3: Semantic Prior是Active Perception的Foundation

没有semantic understanding做不了intelligent gaze。DINOv2的self-supervised features让policy有了"我看到了什么"的semantic grounding，才能做"我该看哪里"的decision。

这暗示了active perception和representation learning的deep connection:
- 好的representation enable active perception
- Active perception收集的diverse viewpoints反过来improve representation learning
- 这是一个virtuous cycle

Infant就是这样学visual representation的——通过active exploration，不断change viewpoint，从multi-view observations中构建object concept。ViA某种程度上是让robot通过human-guided active exploration来learn，和developmental robotics [25,26]的vision呼应。

## 我觉得最大的Limitation

### Point Cloud Fidelity

Single-frame RGB-D → point cloud，noisy depth + incomplete reconstruction → visualization比RGB streaming差很多。User需要practice adapt，fine-grained manipulation有挑战。

Future direction可能是:
- **3D Gaussian Splatting** [64]: 实时dynamic scene rendering，比point cloud更photo-realistic。但要解决如何保持low-latency rendering。
- **NeRF**: Wilder-Smith et al. [57]的方向，但要解决physical camera control问题。
- **Multi-frame fusion**: 不用single-frame，用sliding window的multi-frame point cloud fusion。但会增加latency。

这是个经典的trade-off: fidelity vs latency。ViA选了latency（因为motion sickness是killer），future work需要找更好的balance。

### 没有Memory

Search任务需要memory——你得记住"我已经看过左边了，lime不在那边，应该去右边找"。

当前ViA用Diffusion Policy的temporal context implicitly handle这个，但explicit的memory mechanism会显著help:
- **Transformer memory**: cross-attention over past observations
- **Episodic memory**: store key viewpoints + observations
- **Topological memory**: build scene graph during exploration

这本质上是POMDP问题，belief state需要维护。这是active perception和passive perception的核心区别之一——passive perception假设observation是complete的，active perception承认observation是partial的，需要通过action来reduce uncertainty。

### 没有Language Conditioning

Language instruction天然imply了perceptual strategy:
- "Pick up the **red** mug **behind** the laptop" → 先find laptop, 再look behind it, 再find red mug
- "Put the cup **on** the **left** saucer" → 先look left, find saucer, 然后align

Language提供high-level goals、spatial cues、object relationships、temporal dependencies，这些都是active perception的prior。结合VLM + active perception是一个huge opportunity。

想象一个VLM-conditioned active perception系统:
- VLM理解language instruction，infer出"我该看哪里"的prior
- Active perception policy执行physical search来verify
- VLM根据新observation更新belief，指导next viewpoint

这相当于把LLM的"reasoning about where to look"能力和robot的"physically looking"能力结合，是通向general-purpose robot的promising direction。

## 总结一句话

ViA告诉我们的最深的insight: **robot perception不该是被动的camera stream，而该是主动的信息gather行为**。这件事Bajcsy 1988年就说了 [1]，但直到2024年才有人真正在bimanual manipulation上做出来system-level的solution。原因是三个challenge（hardware、interface、learning）必须co-design，任何一个weak link都break整个系统。这种system-level contribution比single-algorithm breakthrough更rare也更valuable。

Paper website: https://vision-in-action.github.io

Related reads:
- Open Television (最接近的parallel work): https://arxiv.org/abs/2407.01512  
- Learning to Look Around (另一个active neck work): https://arxiv.org/abs/2411.00704
- Active vision might be all you need (concurrent work): https://arxiv.org/abs/2409.17435
- Diffusion Policy (base framework): https://diffusion-policy.cs.columbia.edu
- DINOv2 (visual backbone): https://arxiv.org/abs/2304.07193

---

# Vision in Action (ViA): 从人类示范中学习主动感知

## 1. Paper的核心问题与Motivation

这篇paper来自Stanford的Shuran Song组，第一作者Haoyu Xiong。核心observation非常biological: **perception is inherently active**。当你从bag里拿banana时，你会先scan环境定位bag，再peek进去identify banana，最后focus在grasp点。这种gaze shifting不是decorative的，是task-critical的。

但今天的主流imitation learning系统（ALOHA [2]、Mobile ALOHA [3]、UMI [6]、UMI on Legs [58]、Bunny-VisionPro [9]、Diffusion Policy [10]）几乎都用wrist cameras或fixed third-person cameras。这里有个fundamental的**observation mismatch**:
- Human在data collection时natural地shift gaze
- Robot在deployment时用fixed/mismatched viewpoint
- Policy学不到human的searching/tracking/focusing行为

这个gap在visual occlusion场景下变得critical——wrist camera被环境挡住，无法capture task-relevant信息。

参考Bajcsy 1988的active perception开创性工作: https://ieeexplore.ieee.org/document/5968

## 2. 三大System-Level Challenges

作者identify了三个为什么active perception被忽视的engineering challenges:

### Challenge 1: Flexible Hardware for Gaze Control
Human通过coordinated eye + neck + torso运动来direct gaze。现有robot neck大多是2-DoF（如Open Television [14]、ASIMO [11]、HRP-3 [12]），无法replicate human whole-upper-body产生的motion range。

### Challenge 2: Synchronized Camera-Gaze Movements
VR teleoperation需要precise mirroring human motion + real-time visual feedback。这要求fast motor control + low-latency data streaming，两者在today's hardware都challenging。Motion-to-photon latency超过10ms就引起motion sickness [56]。

### Challenge 3: Scalable Active Perception Strategies
Human gaze由top-down和bottom-up attention共同驱动 [16,17,18,19]。Prior work用hand-crafted heuristics [20,21,22]或next-best-view的uncertainty reduction [23,36,37]，但难泛化。Reinforcement learning方法 [38,39,40,41,35,42]在real-world难deploy。

## 3. ViA的三大设计选择

### 3.1 Hardware: 6-DoF Arm as Robot Neck

Clever的设计选择: 用off-the-shelf的ARX5 6-DoF arm作为neck，end-effector上mount一个iPhone 15 Pro作为RGB-D sensor。这个设计的elegance在于:
- 不用design复杂的humanoid neck/torso biomechanics
- 6-DoF足够approximate human upper body产生的head motion range
- iPhone同时提供RGB、depth、synchronized camera pose

加上两个6-DoF ARX5 arm + fin-ray gripper做bimanual manipulation，每个arm mount在3D-printed shoulder structure上。

这个思路和Sen et al. [15]、Open Television [14]的humanoid neck方向不同——他们用专门的2-DoF neck，ViA用通用的6-DoF arm。Trade-off是flexibility vs biomimicry。

### 3.2 Teleoperation Interface: Intermediate 3D Representation

**这是这篇paper技术上最clever的部分。**

#### 传统VR teleoperation的问题

传统方法（如Open Television [14]、Chuang et al. [52]）直接mirror human head motion并stream live RGB。Motion-to-photon latency = RGB传输latency + robot control latency，typically > 100ms，远超VR comfort threshold的~10ms [59]，导致motion sickness [55]。

#### ViA的解法: View Decoupling

核心idea: **decouple user's view from robot's view** via intermediate 3D scene representation。让user的viewpoint通过rendering instant更新，不等robot物理移动camera。

具体三个components:

**(a) Point cloud construction in world frame W**:

定义world frame W在robot neck的fixed base。每个RGB-D frame通过camera intrinsics和head pose $^WT_H(t)$ transform到world frame。

$$^WT_H(t) = ^WT_H(t_0) \cdot ^{H}T_{\text{iphone}}(t)$$

其中:
- $^WT_H(t_0)$: 初始时刻robot neck的head pose，从joint positions通过forward kinematics得到
- $^{H}T_{\text{iphone}}(t)$: iPhone实时提供的relative pose（iPhone本身的VIO/SLAM）
- $^WT_H(t)$: 时刻t的head pose w.r.t. world frame

得到point cloud $^WX(t)$作为intermediate 3D scene representation。

**(b) Low-latency view rendering**:

用user最新head pose $^WT_{\text{user}}(t+k)$从point cloud $^WX(t)$ render stereo RGB views给VR display:

$$^WT_{\text{user}}(t+k) = ^WT_{\text{VR}}(t+k) \cdot T_{\text{height\_offset}}$$

其中:
- $k$: short time interval（rendering interval，~7ms级别）
- $^WT_{\text{VR}}(t+k)$: VR device的head pose transform到world frame
- $T_{\text{height\_offset}}$: 高度offset让user视角合理

Render频率~150Hz，保证user端smooth viewpoint updates。

**(c) Asynchronous point cloud updating**:

Robot head pose更新到 $^WT_H(t+K)$，其中 $K \gg k$，由robot control latency决定:

$$^WT_H(t+K) = \text{AggregatedUserHeadPose}(t \to t+K)$$

Point cloud以10Hz频率asynchronously更新。

**这里intuition是什么？**

关键在于: 人类visual system对head motion后的visual update latency极度sensitive（这就是motion sickness的根源），但对scene content的slight staleness不sensitive。所以策略是:
- Viewpoint更新 = ultra low latency（rendering based，~7ms）
- Scene content更新 = can be stale（10Hz asynchronous）

这就像你转头看一个moving object——你的gaze跟随是instant的，但object的实际位置可能在你的perceptual latency内slightly outdated，brain可以tolerate。

这种decoupling本质上是把"perception"和"action"在time scale上分离，类似human brain的dorsal/ventral stream分工，或者saccade期间的temporal masking机制。

#### 对比Wilder-Smith et al. [57]
他们用radiance fields做VR teleoperation，但**没有physical camera control**——user不能purposefully control camera。ViA允许user purposefully control camera来maintain task-relevant visibility，这是关键区别。

### 3.3 Policy Learning: Diffusion Policy + DINOv2

#### Observation和Action空间

每个time step $t$:
- Visual input: $\mathbf{I}_t \in \mathbb{N}_0^{H \times W \times C}$，active head camera的RGB
- Proprioceptive state: $\mathbf{P}_t \in \mathbb{R}^{23}$，包括:
  - Neck end-effector pose (position + quaternion): $\mathbb{R}^7$
  - Left arm end-effector pose: $\mathbb{R}^7$
  - Right arm end-effector pose: $\mathbb{R}^7$
  - Two gripper widths: $\mathbb{R}^2$
  - Total: $7 + 7 + 7 + 2 = 23$

Action output: $\mathbf{A}_t = \{a_{t+1}, \ldots, a_{t+n_p}\} \in \mathbb{R}^{n_p \times 23}$

其中:
- $n_p = 16$: prediction horizon
- $n_a = 8$: execution horizon（只有前8个actions执行）
- $n_a \leq n_p$: receding horizon control
- Policy运行频率: 10 Hz

#### Visual Encoder: DINOv2 ViT

用DINOv2 [60] pretrained ViT作为visual encoder，提取384-dim classification token (CLS token)作为compact semantic representation。

为什么DINOv2? DINOv2是self-supervised learning在large-scale unlabeled image上预训练的，产生rich semantic features。相比ResNet-18 [61] pretrained on ImageNet [62]，DINOv2的features更robust、更semantic meaningful，这对active perception至关重要——policy需要理解"我在看什么"才能决定"我接下来应该看哪里"。

Diffusion Policy [10] framework预测future action sequence，相比single-step prediction有temporal smoothing效果，对manipulation的multi-modal action distribution更友好。

#### 关键设计哲学: Shared-Observation Teleoperation

Teleoperator用和robot完全相同的observation space——只看active head camera的rendered view。这样human的perceptual strategies（searching, tracking, focusing）被直接encode在demonstration中，policy通过straightforward behavior cloning就能学到。

这是一个scalable的设计: 不用hand-design gaze strategy，让human intelligence直接transfer。类似ALOHA的joint-to-joint mapping思路，但apply到perception层面。

## 4. 三个评测任务

### 4.1 Bag Task: Interactive Perception
- (1) Open bag, (2) Peek inside to locate target, (3) Take it out
- 150 demonstrations，5 training objects (banana, carrot, dog, shoe, strawberry)
- Test on 2 unseen objects (blue elephant, green avocado)，10 trials total
- 考察interactive perception: 通过physical interaction (opening bag)减少occlusion + active head movement检查interior

### 4.2 Cup Task: Active Viewpoint Switching
- (1) Find & pick up cup from shelf A (right hand), (2) Hand over to left hand, (3) Place on saucer beneath shelf B
- 125 demonstrations，cup随机在shelf A的upper/lower tier，saucer随机在shelf B下
- 考察multi-stage viewpoint switching: cup在shelf深处（upper tier挡wrist camera），saucer在shelf B下方
- 20 test rollouts

### 4.3 Lime & Pot Task: Bimanual Coordination + Precise Alignment
- (1) Find & place lime into pot, (2) Bimanual grasp lift pot, (3) Precise align onto trivet
- 260 demonstrations，pot固定，10 different lime + trivet configurations
- Lime可能在workspace任一侧，需要先决定用哪个arm grasp
- 20 test rollouts

## 5. 实验结果

### 5.1 Camera Setup Comparison

| Setup | Visual Input | Avg. Performance |
|-------|-------------|------------------|
| ViA (Ours) | Active head camera only | **Best** |
| Active Head & Wrist Cameras | Head + 2 wrist | -18.33% vs ViA |
| Chest & Wrist Cameras | Fixed chest + 2 wrist | -45% vs ViA |

**Surprising finding**: 加wrist cameras反而decrease 18.33% performance。Hypotheses:
1. Active head camera alone已经task-complete（teleoperator只用这个view完成任务）
2. Wrist cameras增加input dimensionality但不贡献task-relevant info
3. Wrist cameras在occlusion场景下引入redundant/noisy observations
4. Low-data regime下增加complexity导致overfitting或distraction

这是个counter-intuitive但reasonable的结果。在data-rich regime下更多cameras可能help，但在~150 demos的low-data regime下，sparse informative observation更利于学习。这和data scaling laws [8]的发现一致——input complexity和data量需要co-scale。

Chest & wrist cameras的failure mode:
- Right wrist camera被upper shelf tier完全occlude
- Fixed chest camera缺乏target visibility
- 即使fisheye lens也救不了fixed viewpoint

ViA的active head camera通过dynamic viewpoint adjustment gather更多信息，平均提升45% success rate。

### 5.2 Visual Representation Comparison

| Method | Visual Backbone | Pretraining | Performance |
|-------|----------------|-------------|-------------|
| ViA (Ours) | DINOv2 ViT | Self-supervised | **Best** |
| ResNet-DP | ResNet-18 | ImageNet supervised | Worse |
| DP3 [63] | Point cloud encoder | From scratch | Worst (hallucination) |

ViA得益于DINOv2的strong semantic understanding，能先active find object再initiate arm action。例如Lime & Pot task中，ViA能做long-horizon active search找lime。

DP3的failure mode是**hallucination**——policy misinterpret scene，direct arm到empty shelf section。在bag task上完全fail因为open stage的bag handle grasping不precise。Root cause是DP3 from scratch training，缺乏pretrained visual priors。

**Intuition**: Active perception需要semantic-level understanding才能做intelligent gaze。Point cloud虽然geometric precise但缺乏semantic prior，在occlusion和partial observation下容易hallucinate。DINOv2的semantic features让policy能infer"我应该往哪里看"，这是active search的核心。

### 5.3 Teleoperation Interface Comparison

8个participants的user study:
- 6/8 prefer ViA (point cloud rendering)
- ViA slightly longer collection time（trade-off）
- ViA significantly less motion sickness

这validate了intermediate 3D representation的设计：user接受slightly slower data collection换取comfort。

## 6. Limitations和Future Directions

### 6.1 Teleoperation Interface
- Point cloud从single-frame RGB-D重建，noisy depth导致visualization fidelity低于RGB streaming
- User需要practice adapt
- Fine-grained manipulation有挑战
- Future: dynamic scene fusion + rendering techniques，如4D Gaussian Splatting [64]
- AR glasses [4]可能替代physical teleoperation

### 6.2 Hardware
- 6-DoF arm作为neck虽然effective但不是full human biomechanics
- Future: 更human-like的hardware design [65,66,27]
- Mobile manipulation platform升级，active perception在mobile场景更challenging

### 6.3 Policy Learning
- 当前简单concatenate encoded features，应该explore representation learning融合multi-camera到shared space
- 没有language conditioning——language可以提供high-level goals、spatial cues、object relationships、temporal dependencies
- 没有memory机制——search任务需要记住已搜索区域避免重复，当前policy不支持

## 7. 我的Intuition Building

### 7.1 为什么这个工作重要？

这篇paper击中了robot learning一个被忽视但fundamental的问题: **perception不是passive的，是active的**。Human通过gaze control来actively gather task-relevant information，但今天几乎所有imitation learning系统都假设perception是passive的（fixed/wrist cameras）。

这个gap在occlusion-heavy的real-world manipulation task中变得critical。从bag里拿东西、在cluttered shelf找cup、precise alignment——这些都是日常生活中无处不在的scenario，但现有系统都fail。

ViA展示了: 通过正确的hardware + interface + learning framework设计，可以capture并transfer human的active perceptual strategies到robot，这打开了一个新的research方向。

### 7.2 Intermediate 3D Representation的深层意义

这个设计的elegance不仅是解决motion sickness。它实际上揭示了一个更深的principle: **perception和action在time scale上可以decouple**。

Human brain就是这样work的:
- Saccadic eye movement期间visual是suppressed的（saccadic masking）
- Vestibulo-ocular reflex (VOR) 让gaze稳定即使head在动
- Attention shifting和scene parsing在不同time scale

ViA的point cloud rendering本质上是把这个principle apply到robotics: viewpoint更新在fast time scale（rendering），scene content更新在slow time scale（robot observation）。这让user的perception感觉natural，即使robot的action是slow的。

这个principle可能更broadly applicable: 任何需要low-latency feedback + slow physical actuation的system都可以用这种decoupling。例如:
- Autonomous driving的sensor fusion + planning
- Surgical robot的visualization + control
- Haptic feedback的rendering + actuation

### 7.3 Active Perception和LLM/VLM的Future Integration

Paper的limitation提到language conditioning没做。这是巨大的opportunity。

想象一个VLM-conditioned active perception系统:
- Language instruction: "Pick up the red mug behind the laptop"
- Policy需要actively search场景找red mug
- VLM的semantic understanding + active perception的physical search能力结合

这相当于把LLM的"reasoning about where to look"能力和robot的"physically looking"能力结合。DINOv2已经展现了strong semantic features，下一步是把这些features和language grounding结合。

参考相关方向:
- EgoMimic [4]: egocentric video for imitation learning - https://arxiv.org/abs/2410.24221
- SPIN [42]: simultaneous perception interaction navigation - https://arxiv.org/abs/2501.05420
- Learning to Look [38]: information seeking for decision making - https://arxiv.org/abs/2410.18964

### 7.4 和Developmental Robotics的connection

Active perception在developmental robotics [25,26]有long history。Infant通过active exploration学习visual representation和object concepts。ViA的shared-observation teleoperation某种程度上是让robot通过human-guided active exploration来learn，这和infant通过caregiver的joint attention学习有parallel。

这个connection提示: active perception不仅improve task performance，还可能improve representation learning本身。如果robot在training时actively explores diverse viewpoints，学到的visual representation可能更robust、更generalizable。这和active learning in supervised learning的思路类似，但apply到perceptual representation learning。

### 7.5 关于Point Cloud Rendering的Limitation

Paper承认point cloud fidelity低于RGB streaming。这个limitation可能可以通过:

1. **NeRF/Gaussian Splatting**: Wilder-Smith et al. [57]的方向，但需要解决physical camera control问题
2. **3D Gaussian Splatting [64]**: 实时dynamic scene rendering，可能比point cloud更photo-realistic
3. **Depth completion**: 用learning-based方法补全noisy depth
4. **Multi-frame fusion**: 不用single-frame，用sliding window的multi-frame point cloud fusion

但这些都会增加latency，破坏ViA的core design principle。所以是个open problem: 如何在low-latency constraint下做high-fidelity 3D scene representation?

### 7.6 Memory和Active Search

Paper提到search任务需要memory。这是active perception和passive perception的关键区别之一。Active search需要:
- 记住已经搜索过的区域
- 推断未搜索区域的可能位置
- Based on prior, 决定next viewpoint

这本质上是POMDP (Partially Observable Markov Decision Process)问题，其中belief state需要维护。当前ViA用DINOv2 CLS token + Diffusion Policy的temporal context implicitly handle这个，但explicit的memory mechanism（如transformer memory、episodic memory、topological memory）可能显著improve search efficiency。

相关方向:
- Active vision reinforcement learning under limited observability [40] - https://arxiv.org/abs/2306.00975
- Learning to look around [41] - https://arxiv.org/abs/1709.00507

## 8. 总结

ViA是一个system-level contribution，而不是单一algorithm breakthrough。它的价值在于把三个orthogonal的设计选择coherently integrate:

1. **Hardware**: 用6-DoF arm作为neck，simple but effective
2. **Interface**: Intermediate 3D representation decouple perception-action time scales
3. **Learning**: Shared-observation teleoperation + Diffusion Policy + DINOv2

这三个设计互相reinforce: hardware enable flexible gaze, interface enable natural human demonstration capture, learning enable scalable strategy acquisition。任何一个环节weak，整个system都break。

这给未来的robot learning system design一个template: 要解决一个被忽视但fundamental的问题，需要co-design hardware、interface、algorithm三个layer，光在algorithm layer innovate不够。

Paper website: https://vision-in-action.github.io

相关arXiv links:
- Open Television (对比工作): https://arxiv.org/abs/2407.01512
- Learning to Look Around (对比工作): https://arxiv.org/abs/2411.00704
- Active vision might be all you need: https://arxiv.org/abs/2409.17435
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu
- DINOv2: https://arxiv.org/abs/2304.07193
- ALOHA: https://arxiv.org/abs/2304.13705
- Mobile ALOHA: https://mobile-aloha.github.io
- EgoMimic: https://arxiv.org/abs/2410.24221
- GELLO: https://arxiv.org/abs/2309.13037
- 4D Gaussian Splatting: https://arxiv.org/abs/2403.11142
- Active Perception (Bajcsy 1988): https://ieeexplore.ieee.org/document/5968
