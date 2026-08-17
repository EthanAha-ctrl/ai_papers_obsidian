---
source_pdf: Polybot.pdf
paper_sha256: 63e86a9989fdb4ee8b18178f5f09a8a960d5ff6ef61c23ed216be28cb79d38f4
processed_at: '2026-08-06T05:18:09-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Polybot - 用人话讲讲这paper在搞什么

## 一句话版本

三个完全不一样的robot arm（Franka、Sawyer、WidowX），你给其中一个robot录了点数据，能不能让另一个robot也学会这个task？**可以，但前提是你要在三个层面把domain shift对齐掉。**

## 问题到底有多恶心

你想想这个场景：Franka上你录了50个pick and place的demo，现在你想在Sawyer上做同一个task。你naive的想法是——直接把Franka的image和action都灌进一个policy网络里训，然后deploy到Sawyer上。

这肯定会爆炸，因为：

**Camera看到的完全不一样**。Franka是白色的，Sawyer是黑色的，WidowX是蓝色的。你用exterior camera拍的话，整个画面里arm占了很大一块，不同robot的arm长得完全不同，policy会学到一堆"这是Franka的白色link所以往下走"这种spurious correlation。

**Action的语义不一样**。你说"末端往前移动1cm"，Franka和Sawyer的controller对这句话的理解完全不同。Franka可能直接joint-space插值，Sawyer可能用cartesian impedance，WidowX可能就是个简单的PD。你发出的同一个action command，三个robot执行出来的trajectory完全不是一回事。

**Workspace shape不一样**。WidowX的arm短，能reach的范围小，Franka的arm长能reach的范围大。同样是"把marker放进杯子"这个task，Franka可能是直线过去，WidowX可能得绕一下避开singularity。

所以你如果想naive地把数据混在一起训，policy会confused——它看到Franka的画面和Sawyer的画面，本来task progress是一样的，但visual features完全不同；它想输出一个action，但同一个action在三个robot上效果完全不同。这个policy什么也学不到。

## 他们怎么想的——三层alignment

核心insight特别简单：**你不可能把所有东西都对齐，但你可以对齐足够多的东西，剩下的让learning自己figure out。**

这跟做人有点像——你跟一个法国人合作，你不会说完美的法语，他不会说完美的英语，但你们可以约定用英语交流（align language），各自保留自己的文化习惯（not align culture），然后把事情做成（learn the rest）。

### 第一层：让camera看到的东西尽量一样

他们用了wrist camera。这个decision的intuition特别漂亮：

exterior camera的position需要6个参数来specify（3 translation + 3 rotation），你换个lab、换个setup，这6个参数全变了，policy看到的画面天差地别。

wrist camera的position只需要2个参数来specify：camera离end-effector多高、camera的俯仰角多少。因为camera是挂在arm上的，arm怎么动它就怎么动，workspace是什么样、table是什么颜色、背景是什么pattern，全都看不到了。视野里只有end-effector和你要操作的那个object。

更妙的是：**不同robot的arm颜色、shape、length这些差异全消失了**，因为wrist camera只能看到末端的gripper，gripper都是同一个parallel jaw gripper。Franka的白色、Sawyer的黑色、WidowX的蓝色，wrist camera全都看不到。

这就相当于把一个6维的variation source压缩成了2维，policy学习的负担大大减轻。

代价是什么？你看不到global scene。但对于closed-loop manipulation这种reactive task来说，你只关心眼前的object和gripper的关系，global scene其实不重要。

参考Hsu et al. 2022的paper "Vision-based manipulators need to also see from their hands"（https://sites.google.com/view/see-from-their-hands）——这篇已经论证过wrist camera对single robot的robustness有帮助，Polybot把它扩展到multi-robot setting。

### 第二层：action space怎么对齐——这是个陷阱

这是paper里最纠结的部分，我慢慢讲。

**天真方案1**：把所有robot的action都统一成"末端pose的delta"——$\Delta p_t = p_{t+1} - p_t$，translation和rotation都是。然后所有robot共享一个policy head。

听起来很美，但实际deploy的时候，你发出的command是 $p_t + \Delta p_t^c$，这个target pose被送给robot的controller，controller执行完之后到达的实际pose是 $p_{t+1}$。问题在于：$p_{t+1} \neq p_t + \Delta p_t^c$。误差来源：

- IK solver的数值误差
- Joint limit导致某些pose根本reach不到
- Controller本身的dynamics（PD gain、friction、inertia）让实际trajectory和commanded trajectory有偏差
- 不同robot的link length不同，同一个pose command对应的joint configuration完全不同，碰到joint limit的概率也不同

Paper在Appendix Figure 9-11里画了这个error随time的变化。WidowX的error最大（平均好几cm），Sawyer次之，Franka最小。这个error是高度non-linear的，你relabel action也救不了。

**天真方案2（更激进）**：用blocking controller。什么叫blocking？就是发一个command，等robot完全停下来不再动了，再发下一个。这样可以保证 $p_{t+1} \approx p_t + \Delta p_t^c$。

但blocking也有问题：你teleop的时候用的是continuous control，trajectory是密集的。如果用blocking，你得把continuous trajectory subsample成sparse waypoints，每个waypoint等robot到位。这个subsampling会丢失大量信息，而且对于需要smooth 6-DoF rotation的任务（比如shelf manipulation要把book插进shelf），blocking会让motion变成jerky的step function，一个robot无法精确复现另一个robot的smooth rotation。

实验结果很残忍（Table 4）：在shelf task上，blocking controller成功率0%。在simple pick/place上还行（0.8 vs 0.9），但一旦需要rotation就崩。

**他们的实际方案**：shared high-level action space + multiheaded policy。

高层action space是统一的：一个7维向量，3个translation + 3个rotation + 1个gripper command，每个value都是[-1, 1]归一化的。这个command被送进一个shared environment，environment把它转换成"target pose = current pose + delta"，然后送给每个robot自己的lower-level controller。

但policy网络本身是multiheaded的：shared encoder $\tilde{f}(o)$ 提取visual feature，然后每个robot有一个自己的head $\pi_r(a | \tilde{f}(o), t)$ 输出action。训练的时候，Franka的数据只更新Franka的head，Sawyer的数据只更新Sawyer的head，但所有数据都更新shared encoder。

这样做的intuition：**encoder学到的是"task progress长什么样"这种robot-agnostic的visual feature，head学到的是"我这个robot应该怎么动"这种robot-specific的control dynamics**。

这跟你管理一个跨国团队有点像：所有人共享同一个project plan（encoder），但每个国家的office用自己的方式执行（head）。你不能强迫日本office用美国office的执行方式，但你可以让大家对"项目进展到哪一步了"有共同的理解。

### 第三层：internal representation怎么对齐——这是最巧妙的部分

前两层alignment之后，visual input和action output都尽量对齐了。但还有一个问题：**encoder学到什么feature，完全取决于BC loss怎么训练**。如果只用BC loss，encoder可能学到一堆spurious correlation——比如Franka的data里gripper总是出现在画面左下角，它就学到"左下角有gripper → 往右动"这种feature。这个feature在Sawyer上完全不成立。

他们用contrastive pretraining来强制encoder学习"task-progress-relevant"的feature。核心idea：

即使action space不能完全对齐，但proprioceptive signal（end-effector pose）是可以通过shared IK solver对齐的。两个robot做同一个task，如果它们的end-effector做完全相同的motion，那它们的pose trajectory只差一个constant translation $k$（因为base frame不同）。

$$\forall t, \quad p_{r_1, t} - p_{r_2, t} = k$$

这是一个非常强的invariant。利用这个invariant，你可以定义"两个observation是否处于task的同一个阶段"：只要它们的proprioceptive pose（相对于某个reference point）接近，它们就是positive pair。

**reference point怎么选？** 这是paper的一个关键创新。他们定义了"fixed state"——subtask完成的时刻，比如grasp成功、place成功。所有successful demonstration都必须经过这些fixed state。

然后对每个trajectory里的每个state $p_i$，计算它到**下一个**fixed state的pose difference：

$$d(p_i, p_{t_f})^{xyz} = p_i^{xyz} - p_{t_f}^{xyz}$$
$$d(p_i, p_{t_f})^{quat} = p_{t_f}^{quat} \cdot (p_i^{quat})^{-1}$$

- $p_i^{xyz}$: 第$i$个timestep的3D position
- $p_i^{quat}$: 第$i$个timestep的orientation（用四元数表示）
- $p_{t_f}$: 下一个fixed state的pose
- $d^{xyz}$: position的向量减法
- $d^{quat}$: 四元数乘法，表示"从$p_i$到$p_{t_f}$需要做的rotation"

用"下一个"fixed state而不是"上一个"很巧妙——这衡量的是"还要走多远才能到达下一个milestone"，比"已经走了多远"更符合task completion的语义。一个快robot和一个慢robot，即使它们的absolute time不同，只要它们离下一个milestone的距离相同，就处于task的同一个阶段。

**Positive/Negative sampling**：给定一个anchor $(p_a, o_a)$，从其他robot的trajectory里采样positive $(p_+, o_+)$，条件是：

$$\|d(p_a)^{xyz} - d(p_+)^{xyz}\|_2^2 < \epsilon^{xyz}$$
$$\cos^{-1}(\langle d(p_a)^{quat}, d(p_+)^{quat}\rangle^2 - 1) < \epsilon^{quat}$$

- $\epsilon^{xyz}$: position距离阈值
- $\epsilon^{quat}$: orientation角度阈值（geodesic distance）
- $\langle \cdot, \cdot \rangle$: 四元数内积

Negative就是不符合这个条件的samples。

**Triplet loss**：

$$L(o_a, o_+, o_-) = \max(0, m + \|\tilde{f}_\theta(o_a) - \tilde{f}_\theta(o_+)\|_2^2 - \|\tilde{f}_\theta(o_a) - \tilde{f}_\theta(o_-)\|_2^2)$$

- $o_a, o_+, o_-$: anchor、positive、negative的image observation
- $\tilde{f}_\theta$: encoder，参数$\theta$
- $m = 0.5$: margin
- $\|\cdot\|_2^2$: L2距离平方

这个loss的含义：让anchor和positive的embedding距离比anchor和negative的embedding距离小至少$m$。

**为什么这比标准contrastive learning好？** 标准的contrastive learning（TCN、R3M、VIP）用time-based或augmentation-based positives——同一个trajectory的相邻frame是positive，或者同一个frame的不同augmentation是positive。这些方法学到的feature可能跟task progress无关。

Polybot用proprioceptive distance定义positive，直接告诉encoder："不管visual appearance差异多大，只要task progress相同，embedding就要靠近。" 这强制encoder忽略robot-specific的visual artifact，关注task-progress-relevant的visual cue（比如object相对gripper的位置、grasp的角度等）。

Figure 7的nearest neighbor可视化很impressive：给一个Franka的image，能在Sawyer和WidowX的dataset里找到task-progress对应的frame，即使三个robot的visual appearance完全不同。

## 实验——到底work不work

### Few-shot transfer（Table 1）——核心结果

你在new robot上只录5个demo，能不能学会new task？

| Robot | Method | Pick/Place avg | Shelf avg |
|-------|--------|---------------|-----------|
| Franka | **Polybot** | 0.87 | 0.95 |
| Franka | Naive Multi-Robot | 0.33 | 0.00 |
| Franka | Single Robot | 0.13 | 0.00 |

数字说明一切：
- Single Robot完全学不会shelf task——5个demo根本不够学6-DoF motion
- Naive Multi-Robot（exterior camera + 无contrastive）在shelf上也完全失败——跨robot数据变成noise
- Polybot在所有task上都>70% success rate

**这个improvement的本质**：5个demo提供的是"new task的visual cue是什么"，而其他robot的shared data提供的是"task progress长什么样、general的grasp motion长什么样"。contrastive pretraining让encoder能reuse后者，BC finetune让head学前者。

### Ablation: contrastive到底加了多少（Table 3）

| Method | Franka avg | Sawyer avg | WidowX avg |
|--------|-----------|-----------|-----------|
| Polybot | 0.90 | 0.82 | 0.82 |
| Polybot w/o Contrastive | 0.66 | 0.58 | 0.70 |

平均19%的提升。这看起来不算dramatic，但你要注意：multiheaded training本身已经能学到一些shared feature（因为encoder是共享的），contrastive是在这个基础上的额外boost。

**我的intuition**：contrastive pretraining的作用更像是一种regularization——防止encoder overfit到某个robot的spurious visual feature。在数据量小的情况下，这种regularization的价值会放大。

### Blocking vs Multiheaded（Table 4）——action alignment的核心实验

| Task type | Polybot | Contr. + Blocking |
|----------|---------|-------------------|
| Simple pick/place (S1, S2) | 0.85 | 0.77 |
| Rotation needed (S3, S4, S5) | 0.87 | 0.00 |

这个对比一针见血：blocking controller在simple translational task上还行，一旦需要6-DoF rotation就完全崩溃。

**为什么？** 想象一下shelf task：你要把book从container里拿出来，旋转90度，插进shelf。不同robot的wrist link长度不同，旋转的radius不同。Franka的旋转半径可能是15cm，WidowX可能是10cm。你用blocking controller让WidowX复现Franka的rotation trajectory，WidowX要么撞到东西，要么reach不到目标pose。

而multiheaded policy不需要精确复现——每个robot的head学自己的dynamics，shared encoder只提供"现在处于rotation的中段"这种task-progress信息。

## 我的intuition和联想

### 这paper本质在做什么

我觉得Polybot最深的insight是：**cross-embodiment transfer不是要消除所有domain shift，而是要把domain shift控制在一个policy learning能handle的范围内**。

完全消除domain shift是impossible的（不同robot就是不同robot），完全ignore domain shift是失败的（naive multi-robot training证明了）。Polybot找到一个sweet spot：通过wrist camera把visual variation从6维压到2维，通过shared action representation把action semantic的大部分对齐，通过multiheaded policy把remaining dynamics分离，通过contrastive pretraining把encoder的feature空间对齐。

这有点像做翻译——你不需要把中文和英文的所有语法差异都消除，你只需要找到一个"中间表示"（比如meaning），让两种语言都能map到这个中间表示上，然后各自从中间表示生成自己的表达。

### 跟你（Karpathy）可能关心的几个connection

**1. Scaling law会怎样？**

3个robot、60小时数据，work了。10个robot、600小时数据，还work吗？我的intuition：
- Contrastive的benefit会随robot数量增加而增加——更多positive pair，更好的representation
- Multiheaded的cost也会随robot数量增加而增加——head数量爆炸，每个head的数据量变少
- 可能需要hypernetwork design：用一个robot embedding作为hypernetwork的input，生成head weights，而不是每个robot一个独立head

**2. 跟Decision Transformer的关系**

Polybot本质上是BC + contrastive pretraining，没有用到sequence modeling或RL。能否把multi-robot data放进Decision Transformer的框架？每个robot的trajectory作为不同的"domain"，用robot embedding作为condition？这可能能capture更long-horizon的task structure。

**3. 跟V-JEPA、video pretraining的关系**

Polybot的contrastive用的是proprioceptive ground truth，而video pretraining用的是predictive objective。能否结合？先用video pretrain encoder（得到general visual feature），再用proprioceptive contrastive fine-tune（得到task-progress-aware feature）？这可能能解决Polybot的zero-shot limitation——对完全new task无法zero-shot。

**4. Foundation model时代的Polybot**

如果用CLIP或VLM作为encoder backbone，Polybot的contrastive pretraining可能就不需要了——CLIP已经学到了足够的visual feature。只需要multiheaded finetune + 少量data。但CLIP的feature是image-level的，不是task-progress-level的，所以可能还是需要proprioceptive supervision来fine-tune。

**5. Embodiment-invariant representation的哲学问题**

人类motor cortex能控制非常不同的effector——手、脚、工具、甚至prosthetic。这暗示了某种universal motor representation。Polybot的shared encoder本质上就是在学这种universal representation——"task progress"是embodiment-invariant的，只有"如何执行"是embodiment-specific的。

这可能跟neural science里的"affordance competition hypothesis"有关系——大脑可能同时encode多个affordance，然后根据当前embodiment选择执行哪个。Polybot的multiheaded policy某种程度上mimic了这个process。

## Limitations他们没明说但我看到的

1. **Fixed state需要人工定义**：在quasi-static task里这OK（grasp成功、place成功很明确），但在dynamic task里（pouring、flipping、stacking）"fixed state"就很模糊。能否用automatic keyframe detection？比如用trajectory clustering或change point detection。

2. **Proprioceptive alignment假设base frame一致**：不同robot的base frame其实是不一样的，paper用shared IK solver隐式处理了这个，但对calibration error敏感。如果两个robot的base frame有5度偏差，所有的pose alignment都会有systematic error。

3. **Gripper morphology没显式处理**：实验里三个robot都是parallel jaw gripper。如果换成suction gripper、soft gripper、dexterous hand，这个方法还work吗？suction gripper的"grasp success"和parallel jaw的"grasp success"在visual上完全不同。

4. **Wrist camera的limitation**：对于需要global context的任务（navigate to different table、search for object in a cluttered scene），wrist camera不够。能否用wrist + exterior双camera，让policy自己学习attend to哪个？

5. **Task condition的scalability**：现在是one-hot task index，如果task数量到1000个怎么办？可能需要language embedding或VLM-based task condition。

## 一些你可能感兴趣的web links

- **Polybot project page**: https://sites.google.com/view/polybot-multirobot
- **Hsu et al. "Vision-based manipulators need to also see from their hands"** (wrist camera的motivation): https://sites.google.com/view/see-from-their-hands
- **GNM (General Navigation Model, Shah et al.)** (cross-embodiment navigation): https://general-navigation-models.github.io/
- **R3M (Nair et al.)** (human video pretraining for manipulation): https://r3m.github.io/
- **VIP (Ma et al.)** (value-implicit pretraining): https://sites.google.com/view/vip-rl
- **BC-Z (Jang et al.)** (multi-task BC): https://sites.google.com/view/bc-z
- **Bridge Data (Ebert et al.)** (cross-domain robot data): https://sites.google.com/view/bridgedata
- **RoboNet (Dasari et al.)** (multi-robot dataset): https://www.robonet.wiki/
- **RT-1 (Brohan et al.)** (robotics transformer): https://robotics-transformer1.github.io/
- **Time-Contrastive Networks (Sermanet et al.)** (TCN, 早期的contrastive learning for robot): https://sermanet.github.io/tcn/
- **Ego4D** (egocentric video benchmark): https://ego4d-data.org/
- **V-JEPA** (video joint embedding predictive architecture): https://facebookresearch.github.io/jepa/

## 最后吐槽一句

这paper最让我impressed的是engineering taste——他们没有invent任何新architecture或新loss function，就是用了standard CNN + standard triplet loss + standard BC。他们的contribution全在"仔细分析domain shift的source，然后逐一对齐"这种engineering decision上。

这种taste在robot learning领域太稀缺了。大部分paper都在拼architecture novelty或loss function novelty，但真正deploy到real robot上的瓶颈往往是"camera怎么mount"、"controller怎么interface"、"action space怎么定义"这种system-level decision。

Polybot把system-level decision提升到了method-level contribution的高度——wrist camera vs exterior camera、blocking vs multiheaded、contrastive positive的定义方式——每一个decision都有清晰的reasoning和ablation支持。这种"engineering rigor + theoretical insight"的结合，是robot learning从"demo-driven"走向"science-driven"的关键。

---

# Polybot: 跨Robot通用Policy训练深度解析

## 1. Big Picture: 这篇paper解决什么问题

Robot learning最大的瓶颈是data efficiency。收集real-world robot demonstration极其昂贵(60小时数据在这篇paper里已经算大规模),所以如果能reuse一个robot platform的数据来加速另一个robot platform的学习,价值巨大。

跨robot transfer的核心难点是**domain shift**,来自4个source of variation:
- **Control scheme**: 不同robot的low-level controller语义不同
- **Camera viewpoint**: 不同setup的相机角度不同
- **Kinematic configuration**: 关节配置不同(link length, workspace shape)
- **End-effector morphology**: gripper形态不同

Polybot的核心insight: 如果你能**align enough dimensions of variation**,剩余的variation可以让policy自己学。这是一个partial alignment的思路 - 不强求完美统一,而是在关键维度上对齐,然后让learning填补剩余gap。

## 2. 三层Alignment的设计哲学

Paper的framework围绕三个层面进行alignment,这是整篇paper的核心架构思想:

### 2.1 Observation Space Alignment: Wrist Camera

**Intuition**: exterior camera需要6个DOF来完全specify(position + orientation),而wrist camera只需要2个DOF(camera相对end-effector的height + angle)。这是一个dimensionality reduction的alignment。

更深层的原因:
- Wrist camera天然invariant到robot arm的visual appearance(Franka白色、Sawyer黑色、WidowX蓝色这些差异消失)
- 视野中只包含end-effector + workspace,不包含arm的link
- 这本质上是一种"egocentric representation",类似人类手眼协调
- Reference: 之前Hsu et al.的工作"Vision-based manipulators need to also see from their hands"(ICRA 2022)已经论证过wrist camera的重要性,但Polybot把它放到multi-robot setting下系统化

**关键trade-off**: wrist camera牺牲了global scene information(看不到整个table布局),换来了cross-robot consistency。对于closed-loop manipulation来说,这个trade-off是值得的,因为policy可以reactive地调整。

### 2.2 Action Space Alignment: Shared IK + Multiheaded Policy

这是最微妙的design decision。paper尝试了两种方案:

**方案A (理想但失败)**: Blocking controller + unified action space
- 思路: 让每个robot执行blocking motion(完成一个command再发下一个),把所有action重新label为$\Delta p_t = p_{t+1} - p_t$(achieved pose差),这样所有robot共享同一个action space
- 为什么失败? 即使是blocking controller,$p_t + \Delta p_t^c \neq p_{t+1}$,因为:
  - Hardware的joint limit限制可达workspace
  - IK solver有数值误差
  - Link length不同导致相同的pose command产生不同的实际trajectory
- 实验数据(Table 4): blocking controller在Shelf Manipulation任务上完全失败(0% success),因为shelf任务需要6-DoF的rotational motion,一个robot无法精确复现另一个robot的rotation radius

**方案B (实际采用)**: Shared high-level environment + robot-specific heads
- 高层action space统一: 7维向量(3 translation + 3 rotation + 1 gripper)
- 共享IK solver把pose command转换成joint target
- 但每个robot有独立的policy head $\pi_r(a | \tilde{f}(o), t)$学自己的dynamics
- Encoder $\tilde{f}$是shared的,学到robot-agnostic visual features
- Heads学到robot-specific的control dynamics

这个设计哲学本质上是:**share what can be shared, separate what must be separated**。

### 2.3 Internal Representation Alignment: Contrastive Pretraining

这是paper最有technical深度的部分。

## 3. Contrastive Learning的数学细节

### 3.1 核心思路

即使action space无法完全统一,proprioceptive signal(end-effector pose)经过shared IK solver后是consistent的。两个trajectory如果end-effector做完全相同的motion,它们的pose只差一个constant translation $k$:

$$\forall t, \quad p_{r_1, t} - p_{r_2, t} = k$$

这是一个非常强的invariant。paper利用这个invariant来定义"state similarity",然后用contrastive loss把这个similarity注入到visual encoder中。

### 3.2 Fixed State概念

这是paper的一个关键创新。在一个quasi-static trajectory中,定义"fixed states"为subtask完成的时刻(比如grasp success, place success)。所有successful demonstration都必须经过这些states。

为什么用fixed state? 因为它提供了一个**task-progress的universal reference frame**。不同robot的trajectory可能在time上不对齐(一个快一个慢),但在task progress上是对齐的。

### 3.3 公式逐项解析

**Pose difference定义**:
$$d(p_i, p_{t_f})^{xyz} = p_i^{xyz} - p_{t_f}^{xyz}$$
$$d(p_i, p_{t_f})^{quat} = p_{t_f}^{quat} (p_i^{quat})^{-1}$$

- $p_i$: trajectory中第$i$个timestep的proprioceptive pose
- $p_{t_f}$: fixed state的pose
- 上标$xyz$: 3D Cartesian position
- 上标$quat$: 4D quaternion orientation
- $d^{xyz}$: 直接向量减法,得到relative translation
- $d^{quat}$: quaternion乘法$p_{t_f} \cdot p_i^{-1}$,得到relative rotation。注意quaternion乘法顺序: 这是"从fixed state到current state"的旋转

**Closest fixed state difference**:
$$d(p_i) := d(p_i, p_{t_f})$$
其中$t_f$是$\geq i$的第一个fixed state timestep。

这个定义很巧妙: 用**前向**的fixed state作为reference,这样$d(p_i)$衡量的是"还要走多远才能到达下一个里程碑"。这比用过去最近的fixed state更符合task completion的语义。

**Positive/Negative sampling**:

$$P_{p_a} := \{(p_+, o_+); \|d(p_a)^{xyz} - d(p_+)^{xyz}\|_2^2 < \epsilon^{xyz}, \cos^{-1}(\langle d(p_a)^{quat}, d(p_+)^{quat}\rangle^2 - 1) < \epsilon^{quat}\}$$

- $p_a$: anchor pose
- $p_+$: positive pose
- $\epsilon^{xyz}, \epsilon^{quat}$: 预设的阈值
- Position部分: $L_2$距离小于阈值
- Orientation部分: geodesic distance $\cos^{-1}(\langle p, q\rangle^2 - 1)$小于阈值。这里$\langle p, q\rangle$是quaternion内点积,geodesic distance衡量unit quaternion之间的angular distance

Negative set $N_{p_a}$的定义在paper中看起来有typo,应该是不满足上述条件的samples。

**Triplet Loss**:
$$L(o_a, o_+, o_-) = \max(0, m + \|\tilde{f}_\theta(o_a) - \tilde{f}_\theta(o_+)\|_2^2 - \|\tilde{f}_\theta(o_a) - \tilde{f}_\theta(o_-)\|_2^2)$$

- $o_a, o_+, o_-$: anchor, positive, negative observations
- $\tilde{f}_\theta$: encoder参数为$\theta$
- $m = 0.5$: margin
- 这就是标准的triplet loss,目标是让anchor-positive的embedding距离 + margin < anchor-negative的embedding距离

### 3.4 为什么这个比标准contrastive learning好?

标准contrastive learning(如TCN, R3M, VIP)通常用time-based或augmentation-based positives。Polybot用**proprioceptive distance**定义positives,这是task-aware的:

- 同一task在不同robot上,即使visual appearance差异大,只要proprioceptive state相似,embedding就应该相似
- 这强制encoder学到"task-progress-relevant"的visual features,忽略robot-specific的visual artifacts

Figure 7的nearest neighbor可视化展示了这个效果: 给一个Franka的image,能在Sawyer和WidowX的dataset里找到task-progress对应的frame。

## 4. Architecture Details

### 4.1 Encoder (Figure 5)

```
Input: RGB (64, 64, 3)
  ↓ Conv 3x3, 16 channels, stride 1, padding 1
  ↓ ReLU + MaxPool 2x2
(32, 32, 16)
  ↓ Conv 3x3, 16 channels
  ↓ ReLU + MaxPool 2x2
(16, 16, 16)
  ↓ Conv 3x3, 16 channels
  ↓ ReLU + MaxPool 2x2
(8, 8, 16) → flatten → 1024
  ↓ FC
512 (z representation)
  ↓ FC
```

这是一个非常shallow的CNN(只有3层conv),适合小数据量场景。Image augmentation用了random crop和color jitter,这是为了增强visual robustness。

### 4.2 Decoder (Figure 6)

```
z (512) ⊕ one-hot task index
  ↓ MLP
  ↓ 3 robot-specific heads (each outputs 7-dim action)
```

Task index是one-hot encoding,用来condition policy。每个robot有自己的head,所以forward pass时只有对应robot的head被激活。

这个设计的一个重要implication: 训练时所有robot的data都在一个batch里,但gradient只flow到对应robot的head,encoder的gradient则来自所有robot。

## 5. Experimental Results深入分析

### 5.1 Few-shot Transfer (Table 1)

| Robot | Method | S1 | S2 | S3 | S4 | S5 |
|-------|--------|-----|-----|-----|-----|-----|
| Franka | Polybot | 0.9 | 0.8 | 0.9 | 1.0 | 0.9 |
| Franka | Naive Multi-Robot | 0.4 | 0.3 | 0.3 | 0.0 | 0.0 |
| Franka | Single Robot | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 |

关键观察:
1. **Single Robot完全失败在Shelf任务**: 5个demo不足以学习6-DoF的complex motion,policy很快fall out of distribution
2. **Naive Multi-Robot在Shelf也失败**: exterior camera + 无contrastive alignment,跨robot data反而成了noise
3. **Polybot在所有任务上>70%**: 这证明了三层alignment的cumulative effect

### 5.2 Ablation: Contrastive的作用 (Table 3)

| Method | Franka avg | Sawyer avg | WidowX avg |
|--------|-----------|-----------|-----------|
| Polybot | 0.90 | 0.82 | 0.82 |
| Polybot w/o Contr. | 0.66 | 0.58 | 0.70 |

平均19%的提升。这个数字看起来不dramatic,但要注意:
- W/o contrastive,encoder只通过BC loss学习,容易overfit到single robot的visual features
- Contrastive pretraining提供了一个**pretext task**,强制encoder学习task-progress-relevant features
- 这本质上是一种representation regularization

### 5.3 Blocking vs Multiheaded (Table 4)

| Task | Polybot | Contr. + Blocking |
|------|---------|-------------------|
| S1 (Pick/Place distractor) | 0.9 | 0.8 |
| S3 (New Container, needs rotation) | 0.9 | 0.0 |
| S4 (Shelf reversed) | 1.0 | 0.0 |
| S5 (Shelf new compartment) | 0.9 | 0.0 |

这个对比非常有说服力:
- 简单的translational Pick/Place,blocking还行(0.8 vs 0.9)
- 一旦需要6-DoF rotation,blocking完全崩溃
- 原因: 不同robot的wrist link长度不同,rotation radius不同,blocking controller无法精确复现

Figure 9-11的error plot展示了这个问题: WidowX的commanded vs achieved pose error最大,Sawyer次之,Franka最小。这个error是高度非线性的,无法用简单的relabeling解决。

### 5.4 Zero-shot Transfer (Table 2)

Polybot在S1 (New Distractor)上zero-shot能达到0.4-0.6,这是scene generalization(只是background变化,action相同)。对于task generalization(S3, S4, S5),zero-shot基本失败,这是expected的,因为这些任务需要新的action pattern。

## 6. 与Related Work的Connection

### 6.1 与GNM (Shah et al., ICRA 2023)的对比

GNM用unified action space做cross-embodiment navigation(wheeled, quadruped, drone)。Polybot为什么不能用类似方法?

- Navigation的action space是low-dimensional(通常是linear + angular velocity)
- Manipulation的action space是7-DoF(6 pose + 1 gripper),且每个robot的interpretation差异巨大
- Navigation对action precision要求低,manipulation要求高(mm级别的误差就grasp失败)

### 6.2 与BC-Z, RT-1的对比

- BC-Z (Jang et al., CoRL 2021): single robot, multi-task
- RT-1 (Brohan et al., 2022): single robot type (Google robot), massive data
- Polybot: multi-robot, limited data per robot

Polybot的setting更接近realistic deployment: 你有3-5个不同的robot,每个的数据量都不大,怎么最大化reuse。

### 6.3 与R3M, VIP的对比

R3M (Nair et al., CoRL 2022)和VIP (Ma et al., ICLR 2023)用human video学universal visual representation。Polybot的contrastive learning思路类似,但:
- R3M/VIP用human video作为pretraining source
- Polybot用multi-robot data本身,通过proprioceptive signal作为supervision
- Polybot的representation更task-specific,因为proprioceptive signal直接关联task progress

## 7. Limitations & 我的Intuition

### 7.1 Paper承认的Limitations

1. **需要shared dataset**: 必须有一个task在所有robot上都有data,才能学习correspondence。对于全新robot,无法zero-shot transfer。
2. **Zero-shot只在structurally similar tasks上work**: 对于motion pattern完全不同的新任务,zero-shot失败。

### 7.2 更深层的Limitations(我的分析)

1. **Fixed state需要人工定义**: 在quasi-static任务里这OK,但在dynamic任务(比如pouring, flipping)里,"fixed state"的定义就很模糊。能否用automatic keyframe detection?
2. **Proprioceptive alignment假设base frame一致**: 实际上不同robot的base frame可能不同,paper用shared IK solver隐式处理了这个,但对于calibration误差敏感。
3. **Wrist camera牺牲了global context**: 对于需要long-horizon planning的任务(比如navigate to a different table),wrist camera可能不够。
4. **3个robot的scale**: 真实world可能有几十种robot,这个方法的scalability未验证。每个robot一个head,head数量爆炸。
5. **Gripper morphology差异没有显式处理**: paper说gripper可以vary,但实验里3个robot都是parallel jaw gripper。对于suction gripper, soft gripper,这个方法是否work?

### 7.3 可能的Extension

1. **Hypernetwork for heads**: 用robot embedding作为hypernetwork的input,生成head weights,而不是每个robot一个独立head。这样能scale到更多robot。
2. **Cross-task contrastive**: 目前contrastive只在同任务内,能否跨任务定义positive? 比如所有"grasp"动作的state都应该是positive。
3. **Learned fixed state**: 用VLM或trajectory clustering自动发现fixed states。
4. **3D representation**: 用NeRF或3D scene representation替代2D image,可能更invariant到camera viewpoint。
5. **Foundation model integration**: 用CLIP或VLM encoder替代custom CNN,可能zero-shot能力更强。

## 8. 对你的Intuition Building

作为Karpathy,你可能关心的几个点:

1. **Scaling law for multi-robot**: 这篇paper的3个robot、60小时数据,是否在10个robot、600小时data上还能work? 我的intuition是: contrastive learning的benefit会随着robot数量增加而增加(更多positive pairs),但head数量爆炸会成为瓶颈。

2. **与Decision Transformer的关系**: Polybot本质上是BC + contrastive pretraining,没有用到RL或sequence modeling。能否用Decision Transformer的框架,把multi-robot data作为不同"domains"来condition?

3. **与Ego4D, V-JEPA等video pretraining的关系**: Polybot的contrastive用的是proprioceptive ground truth,而video pretraining用的是time-based或predictive objective。能否结合两者? 用video pretrain encoder,再用proprioceptive contrastive fine-tune?

4. **Embodiment-invariant representation**: 这篇paper的holy grail是学一个"embodiment-invariant"的visual representation。从neural science角度,人类motor cortex能控制非常不同的effectors(hand, tool, prosthetic),这是否暗示了某种universal representation?

## 9. Web Links for Reference

- **Polybot Project Page**: https://sites.google.com/view/polybot-multirobot
- **GNM (General Navigation Model)**: https://general-navigation-models.github.io/
- **R3M**: https://r3m.github.io/
- **VIP**: https://sites.google.com/view/vip-rl
- **BC-Z**: https://sites.google.com/view/bc-z
- **Bridge Data**: https://sites.google.com/view/bridgedata
- **RT-1**: https://robotics-transformer1.github.io/
- **RoboNet**: https://www.robonet.wiki/
- **Ego4D**: https://ego4d-data.org/
- **Time-Contrastive Networks (TCN)**: https://sermanet.github.io/tcn/
- **Vision-based manipulators need to also see from their hands (Hsu et al.)**: https://sites.google.com/view/see-from-their-hands

## 10. 总结

Polybot的核心贡献是一个**pragmatic的multi-robot transfer framework**: 不追求完美的action unification,而是通过三层alignment(observation, action, representation)最大化shareable information,然后用multiheaded policy处理remaining dynamics。这个design philosophy可以总结为:

**Align what you can, separate what you must, learn what remains.**

实验结果证明了这个哲学的有效性: 5个demo就能在全新robot上达到70%+ success rate,这是sample efficiency的巨大提升。Limitations也很clear: 需要shared dataset,zero-shot只在相似任务上work。但这些limitations指向了清晰的future work方向。

整个paper最让我impressed的是: 他们没有用任何fancy的architecture或loss function,只是仔细地分析了domain shift的source,然后逐一对齐。这种**engineering rigor + theoretical insight**的结合,是robot learning领域最需要的工作。
