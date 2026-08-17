---
source_pdf: Learning Dexterous Manipulation Using Contact.pdf
paper_sha256: 51d3b3e147ebbe6ed2dd0e2d8fd037e05228336cef31476548f1f197040c3c5d
processed_at: '2026-08-05T12:51:34-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CHORD

好，Andrej，咱把那些公式先放一边，我用最直白的话把这个故事重新讲一遍。

---

## 一句话版本

教机器人灵巧操作，光让它"模仿人的手怎么动"不行，因为机器人的手跟人的手长得完全不一样。CHORD 的做法是：**别管手怎么动，管"物体被怎么折腾"就行**。

---

## 为什么模仿动作这条路走不通

想象你教一个三指机器人抓手铳子开盒子。人是用大拇指顶住盖子底下，往上掀。你把人的手部动作通过 IK 映射到机器人上，机器人的三指手可能根本够不到盖子底下那个位置。就算勉强够到了，接触面的朝向也完全不对——人是从下往上顶，机器人可能变成从侧面蹭。

结果就是：动作看起来"差不多"，物体纹丝不动。

这就像你让一个左撇子照着右撇子的挥拍动作学打网球，动作一模一样，但球就是打不到一个点上。因为身体构造不同，**同一个动作产生完全不同的物理效果**。

---

## 核心洞察：关注"物体感受到了什么"

CHORD 说：别盯着手看了，看物体。

物体被手碰到的时候，它感受到的是什么？是一组力和扭矩——学术上叫 **wrench**。力决定物体往哪平移，扭矩决定物体怎么转。

人的大拇指从盖子底下往上顶，产生一个"向上的力 + 让盖子翻转的扭矩"。机器人的食指从侧面推盖子边缘，只要也能产生"向上的力 + 同样的翻转扭矩"，**对物体来说效果完全一样**。

这就是 wrench space 的魔力：它是 **object-centric** 的，跟谁施加的、用几根手指、在哪里接触，统统无关。只看"物体被怎么搞了"。

参考：Murray-Li-Sastry 的经典教材把这个讲得最清楚
https://www.crcpress.com/A-Mathematical-Introduction-to-Robotic-Manipulation/Murray-Li-Sastry/p/book/9780849379810

---

## 具体怎么比较"两组接触"

现在问题来了：人有一组接触，机器人也有一组接触，怎么判断它们"在 wrench 层面等价"？

难的地方在于——这两组接触的数量不一样、位置不一样、方向不一样。你不能一个一个对上比。

CHORD 用了个特别聪明的技巧。想象你在暗处摸一个立体雕塑，你怎么判断摸到的是不是同一个雕塑？你会从各个方向去摸，记录每个方向上最远能摸到哪。如果从 512 个方向摸的结果都一样，那基本就是同一个形状。

这就是 **support function** 的直觉。CHORD 预先采样 512 个 6D 方向，把人的 wrench 能力和机器人的 wrench 能力分别投影到这 512 个方向上，每个方向取最大值。这样就把"一组接触"变成了"一个 512 维的向量"。

两个 512 维向量一比，就知道两组接触在功能上有多接近了。

---

## Reward 怎么设计

光"接近"还不够，还得告诉机器人什么叫"够好"。

CHORD 设了一个 **tolerance band**：机器人的 wrench 能力只要落在人的 $(1 \pm \beta)$ 倍范围内，就算合格。比如人能产生 10 牛顿的力，机器人产生 8 到 12 牛顿之间都 OK。

为什么要有 band？因为人的五指手和机器人的三指手能力不可能 exact match。你给个容忍度，机器人才能找到"功能等价但不完全相同"的接触方案。

还有两个细节特别关键：

**第一，不能只有下限，还得有上限。** 如果只奖励"机器人至少能产生人所需的 wrench"，机器人会学会用整个手掌死死抱住物体——什么 wrench 都能产生，reward 拿满，但这根本不是人演示的操作方式。所以还得惩罚"多出来的 wrench 能力"，抑制过度抓握。

**第二，得惩罚"幽灵接触"。** 人没碰的时候机器人碰了，或者人碰了机器人没碰，都得扣分。否则机器人会发现"啥都不碰"也能避免很多 penalty。

---

## 训练的另一个坑：探索太难

即便有了 reward，RL 训练还有个大问题：**一开始物体根本不动**。

机器人随机乱动手指，碰不到物体，物体纹丝不动，task reward 永远是 0。这就像蒙着眼睛学开锁——你不知道锁在哪儿，乱摸一辈子也摸不到。

CHORD 借用了 DexMachina 的 **Virtual Object Controller (VOC)** 技巧。简单说就是：训练早期，偷偷给物体施加一个外力，让它沿着人演示的轨迹自己走。这样机器人即使没碰到物体，也能看到"物体在动"，得到一些 reward signal，知道大概该往哪个方向努力。

然后随着训练进行，慢慢把这个外力减弱到零，让机器人最终自己接管。

这就像教小孩骑自行车时装辅助轮——一开始帮你稳着，慢慢拆掉。

但 VOC 有个副作用：机器人可能学会"反正 VOC 会推物体，我随便碰碰就行"。所以 contact wrench reward 在这里起约束作用——**你可以靠 VOC 推物体，但你接触的质量必须达标**。两者配合才 work。

DexMachina paper:
https://dexterousmachina.github.io/

---

## 退化方案：当人的演示太烂时

有时候人的演示是从普通 RGB 视频重建的，contact 信息全是噪声。这时候直接 match 人的 wrench 会把机器人带沟里去。

CHORD 有个 fallback：放弃精确 match，只要求机器人能**抵抗任意方向的力**。这在 grasping 领域叫 **force closure**——最最基本的 grasping 条件。

用大白话说：我不管你具体怎么抓了，你只要抓得稳，哪个方向拉都不会掉，就行。

代价是丢失了"忠实模仿人类操作方式"的能力——机器人只会稳定抓握，不会 push、slide、lever 这些非抓握操作。但总比被噪声带歪强。

---

## 扩展到全身操作

CHORD 还能扩展到 humanoid 机器人全身操作。两种情况：

**情况一：只有手的演示**（比如戴眼镜的摄像头拍的）
- 用一个 inpainting 模块从手腕轨迹预测全身运动
- 然后在这个预测的全身轨迹上跑 CHORD RL

**情况二：有全身演示但手部不准**（比如第三人称视频重建）
- 手指重建噪声太大，直接用 force closure 退化方案

这个 inpainting 模块用的是 MotionBricks 架构，本质是个 keyframe-conditioned 的 autoregressive 生成模型。给它手腕的轨迹，它给你生成合理的全身运动——包括走路、弯腰、重心转移。

MotionBricks 相关：
https://research.nvidia.com/labs/toronto-ai/

---

## Benchmark：4739 个任务

这篇 paper 另一个贡献是搞了个大规模 benchmark。从 7 个公开数据集（ARCTIC, TACO, HOT3D, OakInk2, DexYCB, GRAB, H2O）里收集人类操作视频，统一处理成机器人能训练的格式。

关键处理步骤：
1. 把各种数据集的格式统一（wrist pose + 21 MANO joints + contact + 6DoF object pose）
2. 用 differential IK 把人的手部动作 retarget 到 Sharpa 机器人手
3. 质量检查：检查穿透（手穿过物体 >2cm 就淘汰）+ 在 Isaac Lab 里 replay 300 步看能不能跑通

最后在 1831 个任务上测，平均成功率 82.12%。这个规模在 dexterous manipulation RL 领域是前所未有的。

ARCTIC dataset:
https://arctic.is.tuebingen.mpg.de/

---

## 实验结果的人话解读

几个关键数字：

**82.12%** —— 1831 个任务的平均成功率。考虑到这些任务包括双臂协调、articulated object 操作、多物体交互，这个数字相当 impress。

**90.77%** —— 全身操作的成功率。把人的五指手演示 transfer 到 G1 人形机器人的三指 Dex3 手，还能 work。**这是 wrench space embodiment-agnostic 特性的最大胜利**。

**Pearson r ≈ 0.80** —— contact wrench reward 和 task success 的相关性。说明这个 reward 确实是"做对事情"的 good proxy，不是 reward hacking。

**Long horizon 40-48 秒** —— 在接近一分钟长的任务上还能维持高精度，baseline 方法随时间增长显著 degrade。因为 wrench reward 是局部 dense signal，每一步都给反馈，不像 trajectory tracking 那样误差会累积。

---

## 真机部署

在 Dexmate 机器人上部署，两只 Sharpa 手，总共 61 个关节。用 Vicon 动捕系统 @100Hz 跟踪物体和机器人位姿，ONNX 模型 @20Hz 推理，底层控制器 @500Hz 执行。

有个细节值得注意：policy 在"policy environment frame"里预测动作，但机器人在"robot base frame"里执行。需要一个 calibration 把两个 frame 对齐。CHORD 用的是最朴素的 trick——让两个 frame 看同一个物体的初始位姿，反解出 frame 之间的 transform。这就是公式 6 和 7 的全部含义。

部署支持 open-loop（执行 action chunk）和 closed-loop（根据实时状态推理）两种模式。

---

## 跟 Teleoperation 的对比特别有意思

Paper 里做了个小实验：让人类操作员用遥操作界面试着复现演示轨迹。

结果发现，就算是"开盒子"这种看起来简单的任务，人遥操作都特别费劲。三个主要原因：

1. **IK 不保真**：操作员想的手指配置，经过 IK 解算后可能完全变了样，尤其手张开很大或接近关节极限时
2. **没有 haptic feedback**：操作员只能靠看，但接触状态经常被遮挡看不见
3. **没法直接控制力**：遥操作界面控制的是位置，不是力，操作员没法精确说"这根手指用这个力顶这个点"

这说明 RL 训练出来的 policy 确实能做到人类实时遥操作做不到的事情——精确的 contact timing + force control。这是 RL + simulation 的独特价值。

---

## 完整的 Video-to-Robot Pipeline

Appendix C 描述了从原始 RGB 视频到机器人执行的完整流程，用了一个 brush 操作任务做演示：

1. **MoGe** 估单目深度，作为 metric anchor
2. **Grounding DINO** 检测物体
3. **SAM 2** 分割视频里的物体
4. **SAM 3D** 从单帧重建 textured mesh
5. **FoundationPose** 跟踪物体 6-DoF 位姿
6. **ViPE + Dyn-HaMR** 重建人手 MANO 参数
7. **DROID-SLAM** 估相机轨迹
8. **3D Gaussian Splatting** 联合优化所有输出

最终 grounding quality AUC 0.647——不算高，但证明了 end-to-end pipeline 能跑通。

MoGe:
https://wangruisheng.github.io/moge/

FoundationPose:
https://nvlabs.github.io/FoundationPose/

---

## 这篇 Paper 真正的贡献

抛开技术细节，CHORD 的核心贡献在概念层面：

**它把 "imitation learning" 从 "imitate motion" 升级到了 "imitate physical effect"。**

传统 imitation learning 的隐含假设是：expert 的 action 是最好的监督信号。但在 embodiment gap 存在时，这个假设崩了——expert 的 action 对你的 embodiment 可能毫无意义。

CHORD 说：真正该 imitate 的不是 action，不是 motion，甚至不是 contact location，而是 **contact 对物体产生的 mechanical effect**。这个 effect 是 embodiment-agnostic 的，是真正可迁移的"任务知识"。

这个 insight 不仅适用于 dexterous manipulation，对整个 robot learning 领域都有启发意义。任何涉及 embodiment gap 的 transfer 问题，都可以想想：**什么层面的表示是 embodiment-agnostic 的？**

---

## 我觉得还存在的问题

**Support function 的 512 个方向是稀疏采样。** 6D wrench space 用 512 个方向覆盖，可能有 high-frequency 结构被 alias 掉。两个 wrench cone 在这 512 个方向上 support 一致，实际形状可能不同。增大 $b$ 会更准但更贵。

**Friction cone 的 polyhedral 近似精度未知。** Paper 没说用几个 edge force 近似 Coulomb cone。太少会 conservative，太多计算贵。

**VOC annealing schedule 需要手调。** 什么时候开始衰减、衰减多快，都是 hyperparameter。不同任务难度可能需要不同 schedule，但 paper 说用同一套 hyperparameter 跑所有任务——这要么说明方法 robust，要么说明 benchmark 任务难度分布比较集中。

**Real-world 还是 state-based。** 用 Vicon 动捕，不是 vision-based。真正 deploy 到野外还得解决 visual perception 的噪声问题。

**Object pose error 作为 success metric 不完美。** 有些任务物体位姿差一点就 functional failure（比如插钥匙），有些任务位姿差很多也没事（比如把东西扔进筐里）。需要更 task-aware 的 metric。

---

## 给你的 Take-away

Andrej，如果你要从这篇 paper 里带走一个 intuition，那就是：

**当你面临 embodiment gap 时，找到那个"gap 消失的抽象层"。**

对手部操作来说，这个层就是 wrench space——力和扭矩的空间。在这个空间里，人的五指手和机器人的三指手没有区别，只有"能对物体施加什么机械效果"这一件事。

这个思维方式跟你在 neuroarchitecture 和 representation learning 里常说的"找正确的 inductive bias"是一脉相承的。Wrench space 就是 manipulation 领域的一个 physically-grounded inductive bias。

参考资源汇总：
- CHORD project page: https://nvidia-isaac.github.io/video_to_data/chord/
- Isaac Lab: https://github.com/isaac-sim/IsaacLab
- Murray-Li-Sastry 教材: https://www.crcpress.com/A-Mathematical-Introduction-to-Robotic-Manipulation/Murray-Li-Sastry/p/book/9780849379810
- Ferrari-Canny 1992: https://ieeexplore.ieee.org/document/192040
- DexMachina: https://dexterousmachina.github.io/
- ManipTrans: https://kailinli.github.io/ManipTrans-Page/
- SPIDER: https://stanford-spider.github.io/spider/
- ARCTIC dataset: https://arctic.is.tuebingen.mpg.de/
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- SAM 2: https://github.com/facebookresearch/sam2

---

# CHORD: Contact Wrench Guidance from Human Demonstration 深度解析

Andrej，这篇 NVIDIA 的工作触及了 dexterous manipulation 里一个非常本质的问题：**如何把 human demonstration 的 "意图" 而非 "动作" transfer 给 morphologically 完全不同的 robot hand**。我尽量把里面的 intuition、math 和 engineering choices 都拆开来给你讲清楚。

Project page: https://nvidia-isaac.github.io/video_to_data/chord/
Isaac Lab (训练框架): https://github.com/isaac-sim/IsaacLab
ARCTIC dataset (主要 benchmark 来源): https://arctic.is.tuebingen.mpg.de/
Murray-Li-Sastry 经典 wrench space 教材: https://www.crcpress.com/A-Mathematical-Introduction-to-Robotic-Manipulation/Murray-Li-Sastry/p/book/9780849379810
Ferrari-Canny grasp metric 原始 paper: https://ieeexplore.ieee.org/document/192040

---

## 1. 核心 Insight: 为什么 Contact Location 不够用

直觉上，把 human demo transfer 给 robot 的最朴素方法是 retargeting——把 MANO keypoints 通过 IK 映射到 robot joint configurations，然后让 robot 跟踪。但问题在于 morphology gap：human thumb 接触 lid 底面和 robot thumb 接触 lid 侧面，3D 位置可能很接近，但产生 object motion 的能力完全不同（normal 方向不一致）。

CHORD 的核心 insight 是：**contact 的 "含义" 应该用 "这个 contact 能让 object 做什么 motion" 来定义，而非 "这个 contact 在 object 表面上的位置"**。

这是 grasping literature 里的经典 wrench space 概念的延伸。在 grasping 里，wrench space 用来分析 force closure；CHORD 第一次把它当作 human-robot motion similarity 的 metric 用在 RL training 里。

一个 contact 在 position $p$ 施加 force $f$，对 object 产生的 wrench 是 6D vector:
$$w = \begin{bmatrix} f \\ p \times f \end{bmatrix} \in \mathbb{R}^6$$

- $f \in \mathbb{R}^3$: 3D force
- $p \times f \in \mathbb{R}^3$: 3D torque（关于 object frame 的 origin）
- 前三维是 net force，后三维是 net torque

关键：wrench 是 **object-centric** 的，与施加 contact 的 embodiment 解耦。Human thumb 和 robot thumb 在不同位置施加不同 forces，只要 wrench 一致，object 的 motion 就一致。

---

## 2. Wrench Matrix 构造 (公式 1)

公式 (1):
$$\mathcal{W}_{h,k} = \begin{bmatrix} w_{h,k}^{1,1} & \cdots & w_{h,k}^{1,d} & \cdots & w_{h,k}^{c_{h,k},1} & \cdots & w_{h,k}^{c_{h,k},d} \end{bmatrix} \in \mathbb{R}^{6 \times (c_{h,k} d)}$$

变量含义：
- 下标 $h$ = human (相应有 $r$ = robot)
- 下标 $k \in \{1, \ldots, K\}$ = object part index（articulated object 有多个 rigid part）
- $c_{h,k}$ = hand-part pair 上 human contact points 的数量
- $d$ = 每个 contact 的 friction cone 用 polyhedral cone 近似时的 edge 数（典型取 4 或 8，Coulomb cone 用正多面体离散化）
- 上标 $i \in \{1, \ldots, c_{h,k}\}$ = contact point index
- 上标 $j \in \{1, \ldots, d\}$ = friction cone 的 edge force index
- $w_{h,k}^{i,j} = \begin{bmatrix} f_{h,k}^{j} \\ p_{h,k}^{i} \times f_{h,k}^{j} \end{bmatrix}$ = 第 $i$ 个 contact 在第 $j$ 个 edge force 下产生的 primitive wrench

**Intuition**: $\mathcal{W}_{h,k}$ 是 wrench cone 的 generator matrix。它的 positive span（非负线性组合）构成 human 在 part $k$ 上能施加的所有 wrench 集合。注意每一列的 norm 大致一致（edge force 都是 unit magnitude），所以 matrix 的 "形状" 由 contact 位置和 normal 决定。

为什么用 polyhedral cone 近似？Coulomb friction cone $\{f : \|f_t\| \leq \mu f_n, f_n \geq 0\}$ 在 3D 是一个圆锥，没法用有限 generator 表达。用 $d$ 个 edge force 做内接多面体近似是标准技巧（参见 Bicchi 1995, Murray-Li-Sastry Ch. 5）。$d$ 越大近似越精确但计算越贵。

---

## 3. Support Function: 解决 "Matrix 比较难" 的问题 (公式 2)

直接比较两个 wrench matrix $\mathcal{W}_h$ 和 $\mathcal{W}_r$ 有两个问题：
1. 列数不同（$c_{h,k} d \neq c_{r,k} d$，因为 contact count 不同）
2. 列的 ordering 任意（没有对应关系）

CHORD 用 convex geometry 里的 **support function** 把 wrench matrix 投影成 fixed-dimension vector:

$$\sigma_{h,k} = \max_{\text{col}}\left(B^\top \mathcal{W}_{h,k}\right) \in \mathbb{R}^b$$

变量：
- $B \in \mathbb{R}^{6 \times b}$ = 预采样的 $b$ 个 6D unit directions（paper 里用 $b = 512$）
- $B^\top \mathcal{W}_{h,k} \in \mathbb{R}^{b \times (d c_{h,k})}$: 把每个 primitive wrench 投影到 $b$ 个 directions 上
- $\max_{\text{col}}$ = 对每行独立取 max
- 输出 $\sigma_{h,k} \in \mathbb{R}^b$ = 在每个 basis direction 上 wrench cone 的最大投影值

**Intuition**: Support function $\sigma_{\mathcal{W}}(d) = \max_{w \in \mathcal{W}} \langle d, w \rangle$ 完全刻画一个 convex set。给定足够多的 directions $b$，两个 wrench cones 的 support function 一致当且仅当 cone 一致（其实严格意义上需要所有 directions，$b=512$ 是近似）。

这把 "比较两个不同 cardinality 的 cone" 变成了 "比较两个 fixed-length vector"——非常 elegant 的 trick。

---

## 4. Contact Wrench-Space Reward (公式 3)

$$r_{\text{cws}}^k = \exp\left(-\frac{\|\max(0, (1-\beta)\sigma_{h,k} - \sigma_{r,k})\|_2^2}{v_{\text{cws}}} - \frac{\|\max(0, \sigma_{r,k} - (1+\beta)\sigma_{h,k})\|_2^2}{v_{\text{cws}}}\right)$$

变量：
- $\beta \in [0, 1)$ = relative tolerance（容忍 robot support 落在 $[(1-\beta)\sigma_h, (1+\beta)\sigma_h]$ band 内）
- $\sigma_{h,k} \in \mathbb{R}^b$ = human support function
- $\sigma_{r,k} \in \mathbb{R}^b$ = robot support function
- $v_{\text{cws}}$ = exponential kernel variance（控制 reward sharpness）

两个 penalty term 的含义：
- **第一项** $\max(0, (1-\beta)\sigma_h - \sigma_r)$: penalty 当 robot support **不足**（低于下界）。意思是 robot 在某方向上**应该能**产生 wrench 但**没产生**——比如 human 用拇指顶住 lid 产生向上 torque，robot 没接触。
- **第二项** $\max(0, \sigma_r - (1+\beta)\sigma_h)$: penalty 当 robot support **过多**（高于上界）。意思是 robot **多出** human 没有的 wrench 能力——比如 robot 用整个手掌包住 object，产生 human demo 里不存在的 extra wrench。这抑制 "过度抓握"。

$\max(0, \cdot)$ 是 element-wise hinge，只在违反 tolerance 时才惩罚。当 robot support 完全落在 band 内，两个 penalty 都是 0，reward = 1。

**Intuition**: 这是个 **tolerance band** 而非 exact match。为什么？因为 human 和 robot 的 contact count、finger geometry 都不同，exact match 不现实也不必要。$\beta$ 给了一个 "只要 robot 能产生 human 所需 wrench 的 $(1 \pm \beta)$ 范围" 的 slack。

---

## 5. 额外的 Contact Penalty

CWS reward 是 exponential kernel，**总是非负**，policy 可以通过不接触来 avoid penalty（虽然也不会得高分）。所以补两个 penalty:

- $r_{\text{unintend}}^k$：当 $\sigma_{h,k} = 0$ 但 $\sigma_{r,k} > 0$，惩罚 "ghost contact"（human 没碰但 robot 碰了）
- $r_{\text{miss}}^k$：当 $\sigma_{h,k} > 0$ 但 $\sigma_{r,k} = 0$，惩罚 "missing required contact"

这两个是 hinge penalty，防止 policy 通过 trivial 解决方案（完全不接触或乱接触）刷 reward。

---

## 6. Total Reward

$$r = r_{\text{task}} + r_{\text{imit}} + r_{\text{contact}}$$

其中：

**Task reward**:
$$r_{\text{task}} = \exp\left(-\frac{\sum_{k=1}^K \|x_t^{\text{object},k} \ominus s_t^{\text{object},k}\|_2^2}{\text{var}_{\text{obj}}}\right) + r_{\text{relative}}$$

- $x_t^{\text{object},k} \in SE(3)$ = reference object part pose at time $t$
- $s_t^{\text{object},k} \in SE(3)$ = rollout object part pose
- $\ominus$ = SE(3) pose difference（log map 到 $\mathbb{R}^6$）
- $\text{var}_{\text{obj}}$ = exponential kernel variance

**Relative reward** (用于 multi-object interaction):
$$r_{\text{relative}}(t) = m(t) \exp\left(-\frac{e_p(t) \cdot e_R(t)}{\text{var}_{\text{rel}}}\right)$$

- $m(t) \in \{0, 1\}$ = mask，只在 human demo 里两个 object 在交互的 phase 才开启
- $e_p(t) = \|p_{1|0}(t) - \bar{p}_{1|0}(t)\|_2$ = relative translation error
- $e_R(t) = d_{\text{geo}}(R_{1|0}(t), \bar{R}_{1|0}(t))$ = geodesic rotation error
- $p_{1|0}$, $R_{1|0}$ = object 1 相对 object 0 的位姿

关键设计：mask $m(t)$ 基于 **demonstration** 的 inter-object distance，**而非** policy 的当前 inter-object distance。这防止 policy 通过 "把两个 object 推远" 来 trivially avoid reward（approach phase 时不需要精确 relative pose，所以 mask 自动关闭）。

**Imitation reward**:
$$r_{\text{imit}} = \exp\left(-\frac{\|\bar{x}_t^{\text{robot}} - s_t^{\text{robot}}\|_2^2}{\text{var}_{\text{imit}}}\right)$$

- $\bar{x}_t^{\text{robot}}$ = retargeted human motion (IK prior)
- $s_t^{\text{robot}}$ = rollout robot configuration

这是 regularizer，把 robot 拉向 IK 解（防止 RL 学到完全不合理的姿势）。

---

## 7. VOC (Virtual Object Controller)

VOC 来自 DexMachina [49]，本质是个 "training wheel"：

- 在 RL 训练早期，object 还没被 robot 抓住，task reward 是 sparse 的（object 不动 → reward 低）
- VOC 在 object 上施加一个 auxiliary wrench，让 object 沿 reference trajectory 移动
- 这样即使 robot 还没找到正确的 contact，也能得到 "object 在动" 的 dense reward signal
- 通过 curriculum 把 VOC gain anneal 到 0，让 policy 最终自己接管 object motion

问题：VOC 容易陷入 local optimum——policy 学会 "靠 VOC 推动 object，自己随便接触"。这就是为什么需要 contact wrench reward 来约束 contact 的**质量**，而不只是 object 是否在动。

CHORD 的训练 trick：
1. **Reset to random reference frame** + 短暂保持 VOC 满功率：让 robot 从任意中间状态恢复 contact
2. **Object perturbation from human wrench matrix** $\mathcal{W}_{h,k}$：用 human demo 的 wrench 作为 disturbance source，让 policy 学会抵抗 task-relevant disturbances（而非 random noise）
3. **Residual action space** with retargeted motion as prior
4. **VOC annealing curriculum**

---

## 8. Force-Closure Reward (退化版本)

当 human demo 太 noisy（如从 RGB video 重建），contact wrench 不可靠，CHORD 退化到一个 reduced objective:

$$r_{\text{fc}}^k = \frac{1}{B}\sum_{b=1}^B \mathbb{1}[\sigma_{r,k,b} > \epsilon]$$

- $\sigma_{r,k,b}$ = robot support value for part $k$ along basis direction $b$
- $\epsilon$ = small threshold（防止数值噪声触发）
- $\mathbb{1}[\cdot]$ = indicator function

**Intuition**: 这是 "robot 至少在每个方向上都能产生一点 wrench"。最大化这个 reward 等价于 **force closure**（grasp 能抵抗任意方向 external wrench）。这是 wrench space 的最弱约束——不要求 wrench magnitude 匹配 human，只要求 wrench cone **覆盖所有方向**。

Ablation A.1 显示：当 noise 很大时，$r_{\text{fc}}$ 反而比 $r_{\text{cws}}$ 好（因为 corrupted $\sigma_h$ 反而误导）。但 $r_{\text{fc}}$ 牺牲了 dexterity——它只鼓励 stable grasp，不鼓励 faithful imitation of human manipulation（比如 pushing, sliding 这些 non-force-closure phases）。

---

## 9. Whole-Body Manipulation 扩展

两种场景：

**Hand-only references** (e.g. egocentric video)：
- 用 MotionBricks [42] 的 inpainting module 从 end-effector trajectory 预测 full-body motion
- MotionBricks 是个 modular latent generative model，keyframe-conditioned autoregressive
- 把 wrist trajectory $\mathcal{T}^{\text{EE}} = \{(p_h^{\star,t}, q_h^{\star,t}) | t \in [1,T], h \in \{L, R\}\}$ 作为 dense constraint
- 训练 tokenizer, root module, pose module 预测 global root trajectory 和 full-body joint motion
- 然后在 predicted whole-body reference 上 apply CHORD RL

**Whole-body references** (e.g. third-person video)：
- Full body 可用但 finger reconstruction 不准
- 直接用 $r_{\text{fc}}$ 而非 $r_{\text{cws}}$

公式 (4), (5) 描述 EE constraint:
$$\mathcal{T}^{\text{EE}} = \{(p_h^{\star,t}, q_h^{\star,t}) | t \in [1,T], h \in \mathcal{H}\}$$
$$\mathcal{T}_{\text{gt}}^{\text{EE}} = \{(p_h^t, q_h^t) | t \in [1,T], h \in \mathcal{H}\}$$

- $\mathcal{H} = \{L, R\}$ = left/right wrist body indices
- $p_h^{\star,t} \in \mathbb{R}^3$ = target wrist position
- $q_h^{\star,t}$ = 6D rotation representation（continuous rotation representation，Zhou et al. 2019）

---

## 10. Benchmark 统计

CHORD benchmark 是这篇 paper 的另一个 contribution：4,739 个 bimanual dexterous manipulation tasks。

数据来源：
- ARCTIC: bimanual hand-object manipulation
- TACO: tool-action-object
- HOT3D: egocentric multi-view hand-object tracking
- OakInk2: bimanual hands-object
- DexYCB: hand grasping
- GRAB: whole-body grasping
- H2O: first-person hand-object
- + in-house 视频重建

处理流程：
1. Dataset loader → unified representation (wrist pose, 21 MANO joints, per-link contact, 6-DoF object pose)
2. HOT3D 的长视频被切成 atomic clips (294 recordings → ~4045 clips)
3. Differential IK with QP optimizer（200 iter/frame，residual < 1e-6 退出）
4. Quality check: penetration check (capsule model, threshold 2cm) + replay check (Isaac Lab 300 steps)

Diversity metrics（对比 prior work）：
- **Time horizon length**: CHORD 更长（接近 1 分钟）
- **Contact events per task**: CHORT 更密集
- **Ferrari-Canny Epsilon** $\varepsilon = \min h_{h,k}$: force closure 的距离度量，$\varepsilon > 0$ 表示 force closure

---

## 11. 实验结果分析

### Table 1: Baseline Comparison

| Task Suite | Metric | Ref. Method | Ref. Score | CHORD Score |
|------------|--------|--------------|-------------|-------------|
| DM (DexMachina) | AUC | DexMachina | 0.232±0.214 | **0.687±0.358** |
| MT (ManipTrans) | MT-SR | ManipTrans | 0.428 | 0.639 |
| SP (SPIDER) | SP-SR | SPIDER | 0.333±0.488 | 0.359±0.482 |
| Ours-1 | AUC | DexMachina | 0.211±0.138 | **0.895±0.052** |
| Ours-1 | SP-SR | SPIDER | 0.133±0.327 | **0.999±0.000** |
| Ours-2 | SP-SR | SPIDER | 0.533±0.503 | **0.982±0.022** |

注意：每个 row 用对应 baseline 的 metric，所以跨 row 不可比。CHORD 在 SP 的原始 suite 上只是 marginal 提升（0.333 → 0.359），但在自己的 harder suite 上 (Ours-2) 把 SP-SR 从 0.533 拉到 0.982。这说明 CHORD 在长 horizon 和 dense contact 上优势更明显。

### Table 2: Ablation on Contact Guidance

| Sequence | CHORD (SR) | Position Only (SR) | No Contact (SR) |
|----------|-------------|----------------------|------------------|
| box grab | **0.702±0.257** | 0.334±0.141 | 0.384±0.206 |
| mixer use | **0.894±0.023** | 0.624±0.166 | 0.423±0.273 |

CHORD vs Position Only 是核心比较。Position Only（DexMachina-style）reward 只 match 3D contact location，结果显示 wrench space guidance 显著更好。No Contact baseline 比 Position Only 还差，确认 contact guidance 整体有必要。

### Table 3: Whole-body Ablation

| Category | n | CHORD (SR) | Position Only (SR) |
|----------|----|-------------|---------------------|
| Rigid | 4 | **0.994±0.008** | 0.460±0.487 |
| Articulated | 4 | **0.914±0.107** | 0.000±0.000 |
| Multi-object | 4 | **0.866±0.129** | 0.192±0.166 |
| Overall | 12 | **0.925±0.104** | 0.217±0.333 |

这是 cross-embodiment 的 ablation（5-finger human hand → G1 + Dex3 3-finger hand）。Position Only 在 articulated object 上直接 **0.000**——完全失败，因为 3-finger hand 根本无法在同样的 3D 位置接触。而 wrench space alignment 还能找到等价的 contact 产生相同 object motion。**这是 wrench space 的核心价值**：embodiment-agnostic。

### Figure 6: Correlation Analysis

Pearson $r \approx 0.80$ between normalized CWS reward and task success rate，within each dataset $r = 0.76 \sim 0.89$。说明 CWS reward 是 task success 的 good proxy。Monotonic 但 saturating——意味着 reward 到一定阈值后 task success 不再线性提升。

### Figure 7: Long Horizon

CHORD 在 horizon 接近 40-48 秒时仍维持 ADD-AUC ≈ 0.85-0.98，baseline 方法随 horizon 增加显著 degrade。这验证 wrench space guidance 在 long-horizon 上更 robust——因为它给的是局部 dense signal 而非 global trajectory tracking。

---

## 12. Real-World Deployment

Hardware: Dexmate robot + 2 Sharpa dexterous hands
- 61-DOF commanded joint vector（每 arm 7 DOF + torso 3 DOF + 每 Sharpa hand 22 DOF）
- Vicon 6-camera motion capture @ 100 Hz
- ONNX model inference @ 20 Hz，低层 controller ZOH @ 500 Hz

Frame alignment calibration（公式 6, 7）：
$$T_E^B = T_O^B (T_O^E)^{-1}$$
$$T_V^E = T_O^E (T_O^V)^{-1}$$

- $V$ = Vicon world frame
- $B$ = robot base frame
- $E$ = policy environment frame
- $O$ = object frame
- $T_E^B$ = environment→base transform（用一次 startup calibration 固定）
- $T_V^E$ = Vicon→environment transform

这个 calibration 用 "policy frame 和 robot frame 看到同一个 object pose" 的 constraint，非常标准的 hand-eye calibration 思路。

---

## 13. Video-to-Simulation Pipeline (Appendix C)

从 monocular RGB video 到 simulation 的完整 pipeline 很值得看：

1. **MoGe** [41]: monocular depth estimation，作为 metric anchor
2. **Grounding DINO** [17]: open-vocabulary detection，定位 object
3. **SAM 2** [33]: video segmentation
4. **SAM 3D** [36]: 单帧 textured mesh 重建
5. **FoundationPose** [44]: 6-DoF object pose tracking
6. **ViPE** [10] + **Dyn-HaMR** [46]: monocular MANO hand pose recovery
7. **DROID-SLAM** [40]: monocular SLAM for camera trajectory
8. **3D Gaussian Splatting** [12]: joint refinement with photometric + segmentation + depth losses

最终 brush manipulation task 的 grounding quality: AUC 0.647。这个数字不算高，但说明 end-to-end pipeline 能 work。

参考链接：
- MoGe: https://wangruisheng.github.io/moge/
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- SAM 2: https://github.com/facebookresearch/sam2
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM

---

## 14. 关键 Intuition 总结

**Intuition A: Wrench 是 "what the object feels"**
不要想 "robot 的手指在哪里"，要想 "object 在被什么 force/torque 作用"。Wrench 把 embodiment 抽象掉，只留下 object-centric 的 mechanical effect。

**Intuition B: Support function 是 convex set 的 fingerprint**
两个 wrench cone 即使 cardinality 不同，support function 在足够多 directions 上一致就近似等价。这是把 set comparison 降维到 vector comparison 的标准技巧。

**Intuition C: Tolerance band 是 morphology gap 的 explicit slack**
$(1 \pm \beta)\sigma_h$ 这个 band 承认 robot 和 human 的 wrench 能力不会 exact match，只要在 tolerance 内就算"功能等价"。

**Intuition D: Penalty on excessive wrench 抑制 over-grasping**
只 reward "robot 产生 human 所需 wrench" 不够，还要 penalize "robot 产生 human 没有的额外 wrench"。否则 policy 会用整个手掌包住 object，stable 但不符合 demo 风格。

**Intuition E: Force closure 是 noisy demo 下的 degenerate wrench metric**
当 human demo 不可信，退一步只要求 "robot 能抵抗任意方向 disturbance"——这是 wrench space 的最弱约束，等价于 classical force closure。

**Intuition F: VOC + Contact reward 互补**
VOC 解决 "object 不动 → sparse reward" 的 exploration 问题；contact reward 解决 "VOC 推动 object，policy 随便接触" 的 local optimum 问题。两者缺一不可。

---

## 15. Limitations & 我的思考

Paper 自己列的 limitations：
1. Real-world 是 state-based（用 Vicon），不是 vision-based
2. 需要 clean demo，noisy demo 要退化到 force closure
3. Object pose error 作为 metric 不完美（小 pose error 可能 functional failure，大 pose error 可能 task success）

我会额外思考几个点：

**a. Support function 是 lossy projection**
$b = 512$ 个 directions 对 6D space 是稀疏采样。两个 wrench cone 可能在 512 directions 上 support 一致但实际形状不同（high-frequency 结构）。增加 $b$ 会更精确但计算更贵——这其实是个 fidelity-efficiency trade-off。

**b. Friction cone 的 polyhedral 近似**
$d$ 个 edge forces 是 Coulomb cone 的 inner approximation。$d$ 太小会 conservative（认为 human 能力比实际小），$d$ 太大计算贵。Paper 没明确说 $d$ 取多少，常见是 4-8。

**c. Reward shaping 的 non-stationarity**
VOC annealing 让 effective MDP 随训练改变。这违反 standard RL 假设，但 curriculum learning 一直这么做。难点在于 annealing schedule 需要手动 tune。

**d. 与 Diffusion Policy 的关系**
CHORD 是 RL-based，imitation-based 方法如 Diffusion Policy 在 bimanual manipulation 上也很强。区别在于 CHORD 用 simulation 来 explore（可以试错），diffusion policy 需要 expert demonstration。对于 contact-rich task，sim-to-real 的 dynamics gap 是 risk；imitation 的 demo coverage 是 risk。CHORD 选了前者，并用 wrench space reward 来 reduce exploration 难度。

**e. 与 RLDG (Reinforcement Learning from Demonstration Guidance) 的关系**
CHORD 本质是 RLDG——用 demo 作为 reward shaping，而非直接 imitation。这个范式的好处是 demo 不需要 exact match（embodiment gap 下也不肯能 exact match），只需要在某个 abstract space（这里是 wrench space）上 align。

**f. 为什么不直接用 differentiable simulation?**
Differentiable physics（如 Brax, MJX, DiffTaichi）可以让 gradient 直接 backprop through contact。但 contact dynamics 的 non-smoothness（hybrid dynamics, complementarity）让 gradient 很 noisy。CHORD 用 RL + reward shaping 绕过这个问题，代价是 sample efficiency。

---

## 16. 一句话总结

CHORD 把 "human demonstration transfer" 问题转化为 "wrench cone alignment" 问题，用 support function 把 cone 比较变成 vector 比较，用 tolerance band 容忍 morphology gap，用 VOC 解决 exploration 难题，最终在 1831 个 task 上做到 82.12% success rate。最 elegant 的点是 wrench space 天然 embodiment-agnostic——这是为什么 5-finger human hand 能 transfer 到 3-finger Dex3 hand 还能 work。

如果你想 build intuition，我建议从 grasping literature 的 wrench space 分析（Ferrari-Canny 1992, Murray-Li-Sastry Ch. 5）开始，再看 Bicchi 的 force closure 工作，最后回到 CHORD 会发现它把经典 grasping 数学用到 RL reward shaping 里——这个 bridge 做得非常漂亮。

参考资料：
- Murray-Li-Sastry 教材: https://www.crcpress.com/A-Mathematical-Introduction-to-Robotic-Manipulation/Murray-Li-Sastry/p/book/9780849379810
- Bicchi 1995 "On the Closure Properties of Robotic Grasping": https://journals.sagepub.com/doi/10.1177/027836499501400402
- Ferrari-Canny 1992: https://ieeexplore.ieee.org/document/192040
- DexMachina: https://dexterousmachina.github.io/
- ManipTrans: https://kailinli.github.io/ManipTrans-Page/
- SPIDER: https://stanford-spider.github.io/spider/
- EgoMimic: https://egomimic.github.io/
- Sonic (whole-body controller used in CHORD): https://arxiv.org/abs/2511.07820
- MotionBricks (inpainting module): 见 paper ref [42]
