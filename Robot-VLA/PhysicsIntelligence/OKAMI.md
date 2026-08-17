---
source_pdf: OKAMI.pdf
paper_sha256: 5f5cb32c2fc00a9be46b107eb30429e6bab4ed32b626255bed6e1d14ae639a2a
processed_at: '2026-08-05T23:01:37-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OKAMI 用人话说

## 1. 一句话概括

OKAMI 让一个 humanoid robot 看一段人做事的视频(就一段, 不需要动作标签), 然后它就能学着做同样的事, 而且 object 换个位置也能适应。核心 trick 是"把人的动作搬过来, 再根据当前 object 在哪儿把动作扭一下"。

类比一下: 你看 YouTube 上一个老外做了道菜, 你自己要做的时候厨房台面布局不一样、菜板位置不同, 但你脑子知道"先去这里拿刀, 再到那里切菜" —— 你不照搬老外的手部轨迹, 你照搬"动作相对于菜板的关系"。OKAMI 就是给 robot 装了这么个直觉。

## 2. 为什么这件事难

让 robot 从视频学动作, 听起来简单, 实际上有几个坑:

**第一个坑: 视频里没有 robot action label**。视频就是 RGB(+depth) 像素流, robot 该发什么 joint command 完全要自己推断。跟 supervised imitation learning 不一样 —— 后者你录的时候手柄就记录了 action, 直接 BC 就行。这里你得"反推"出人发了什么 action, 再把这个 action"翻译"到 robot 上。

**第二个坑: 人和 robot 尺寸不一样**。人手臂 70cm, GR1 手臂可能 50cm, joint 数也不同, 不能 1:1 copy joint angles。这个叫 **kinematic retargeting** 问题, computer graphics 里研究了 30 年(Gleicher 1998, https://dl.acm.org/doi/10.1145/280814.280820)。

**第三个坑: object 位置变了怎么办**。视频里 cup 在桌面左边, test time cup 在右边。你不能让人手的轨迹直接 replay, 否则抓空。这个是 **spatial generalization** 问题。

**第四个坑: humanoid DoF 太多, 优化空间爆炸**。GR1 这种 humanoid 全身 30+ DoF, 你要 end-to-end 优化 robot action 去重现 object trajectory (ORION 那套), 计算量扛不住, 而且 kinematic redundancy 让解不唯一。

OKAMI 的核心 insight 是: 既然人和 humanoid 形状像, 我们不如直接把人的动作 retarget 过来作为 prior, 然后只在"位置不匹配"的地方做局部修正。这比从头优化 action 便宜得多。

## 3. OKAMI 怎么拆这个问题

### Stage 1: 把视频"读懂"

OKAMI 先把视频过一遍, 抽取三样东西:

**(a) 哪些 object 是任务相关的**

GPT-4V (https://openai.com/gpt-4) 看几帧视频, 吐出 object 名字: "salt bottle", "bowl"。这个 step 看似简单, 实际上很重要 —— 之前的方法要么假设已知 object (closed-world), 要么用无监督方法在简单背景下能 work, 一旦背景复杂就崩。GPT-4V 的 common-sense 让它在杂乱场景也能识别"哪个才是 salt bottle 而不是旁边的酱油瓶"。

拿到名字后, 用 **Grounded-SAM** (https://github.com/IDEA-Research/Grounded-Segment-Anything) 在第一帧做 segmentation, 用 **Cutie** (https://github.com/hkchengrex/Cutie) 在后续帧做 video object segmentation, 把 mask propagate 到整段视频。

**(b) 人的动作是什么**

这里 OKAMI 用了三件套 stack:

- **4D Humans** (https://humans4d.is.tue.mpg.de/, Goel et al. ICCV 2023): 给出 body 粗略 3D pose, 但是手是平的(flat hands, 没有 finger articulation)
- **HaMeR** (https://geopavlakos.github.io/hamer/, Pavlakos et al. CVPR 2024): 从 2D hand keypoints 估 3D hand pose
- **Modified SLAHMR** (https://vye15.github.io/slahmr/): joint refine, 同时优化 body pose + hand pose + body location, 用 reprojection loss + temporal smoothness

为什么搞这么复杂的 stack? 因为没有单一模型能同时给你"全身 pose + 精细 hand pose + 时序一致"。每个模型 specialize 在自己擅长的部分, 然后 SLAHMR 把它们缝起来。

输出是 SMPL-H 模型序列 —— SMPL-H 是 SMPL body + MANO hand 的合体模型 (https://smpl.is.tue.mpg.de/)。

**(c) 任务怎么分成 subgoals**

用 **CoTracker** (https://github.com/facebookresearch/co-tracker) 在 object 上 track keypoints, 算每帧平均 keypoint velocity, 然后用 changepoint detection (Killick et al. 2012, https://www.tandfonline.com/doi/abs/10.1080/01621459.2012.737745) 找 velocity 突变点 —— 这些点就是 subgoal 边界。

每个 subgoal 里 OKAMI 标记两个 object:
- **Target object**: 当前 step 被操作的(移动的)object
- **Reference object**: 提供空间参考的 object (例如把零食放盘子里, 零食是 target, 盘子是 reference)

### Stage 2: 在新环境里执行

Test time robot 面对新场景, OKAMI 重新用 vision pipeline 找到 object 当前位置, 然后做两件事:

**(a) Retarget body motion (arm)**

从 SMPL-H 提取 shoulder / elbow / wrist pose, 用 **Pink** (https://github.com/stephane-caron/pink, 一个 IK 库) 算出 humanoid 各 joint 角度, 让 humanoid 的 shoulder / elbow / wrist 跟人差不多。

注意这里有个 trade-off: wrist position 权重 1.0, shoulder/elbow orientation 权重 0.04~0.08。说明 OKAMI 优先保证"手腕到目标位置", 其他 joint 自由发挥。这是合理的 —— 末端到达对 manipulation 是硬约束, 中间姿态只要不别扭就行。

**(b) Retarget hand pose (fingers) separately**

这部分用 **dex-retargeting** (https://github.com/dexsuite/dex-retargeting, Qin et al. AnyTeleop)。它把 SMPL-H 的 finger joint angles map 到 Inspire dexterous hand 的 finger joints。

为什么 arm 和 hand 要分开? 因为它们的"参考系"不一样:
- Arm motion 是相对 world / base frame 的, 跟 object 在哪儿有关 (extrinsic)
- Hand pose 是相对 palm frame 的, 跟 object shape 和 grasp type 有关 (intrinsic)

如果你 end-to-end 一起学, 模型得 implicitly 学到这个解耦, 数据效率低。OKAMI 直接显式 factorize, 等于注入了一个 strong inductive bias。这是 Karpathy 你经常讲的 "software 2.0 vs software 1.5" 那个 spectrum 上, OKAMI 偏 1.5 —— 用人类先验 decompose, 然后只在子问题里用 neural network (DinoV2 在 BC policy 里, HaMeR 在 vision 里)。

## 4. 核心公式直觉: Trajectory Warping

这是 paper 最 mathy 的部分, 我把直觉讲清楚。

设原 retargeted trajectory 是 $\tau^{robot}(t)$, 起点 $p_{\text{start}} = \tau^{robot}(t_i)$, 终点 $p_{\text{end}} = \tau^{robot}(t_{i+1})$, 都在 SE(3) (3D 位置 + 朝向) 空间里。

Test time, object 位置变了, 对应的"应该的"起点终点变成 $T_{\text{start}} \cdot p_{\text{start}}$ 和 $T_{\text{end}} \cdot p_{\text{end}}$, 这里 $T_{\text{start}}, T_{\text{end}} \in SE(3)$ 是 object 新位置对应的 transform (从原位置到新位置的 rigid transform)。

OKAMI 给的 warped trajectory 公式:

$$\hat{\tau}^{robot}(t) = \frac{\tau^{robot}(t) - p_{\text{start}}}{p_{\text{end}} - p_{\text{start}}} \cdot (T_{\text{end}} \cdot p_{\text{end}} - T_{\text{start}} \cdot p_{\text{start}}) + T_{\text{start}} \cdot p_{\text{start}}$$

变量含义:
- $\tau^{robot}(t)$: 原轨迹在 t 时刻的位置/朝向
- $p_{\text{start}}, p_{\text{end}}$: 原轨迹起终点
- $T_{\text{start}}, T_{\text{end}}$: object 位置变化对应的 SE(3) transform (起点对应 target object, 终点对应 reference object, 或者反过来, 看 subgoal 类型)
- $\hat{\tau}^{robot}(t)$: warp 后的新轨迹

**人话翻译**: 
- $\frac{\tau^{robot}(t) - p_{\text{start}}}{p_{\text{end}} - p_{\text{start}}}$ 这个分数是 "t 时刻在原轨迹中走多远了" 的归一化进度, 在 0 到 1 之间 (起=0, 终=1)
- 然后乘以新的"总位移" $(T_{\text{end}} \cdot p_{\text{end}} - T_{\text{start}} \cdot p_{\text{start}})$, 加上新的起点 $T_{\text{start}} \cdot p_{\text{start}}$
- 效果: "保持原轨迹的时间节奏, 但把起终点平移/旋转到新的 object 位置"

**几何类比**: 想象你在一张纸上画了一条曲线 (原轨迹), 现在你想把曲线两端粘到桌上两个新位置 (object 新位置), 你怎么做? 你拉伸/扭曲这张纸, 让曲线两端对上。OKAMI 做的就是这个, 不过是 in SE(3)。

**关键假设**: 这个公式假设 trajectory shape 可以通过起终点的 affine transform 重 parameterize。换言之, "中间过程只是起终点之间的某种插值, 没有独立的中间结构"。这对以下任务 work:
- pick-and-place (伸手→抓→收回→放下, 起终点是关键)
- pouring (拿杯子→到碗上方→倒, 起终点决定 path)

对以下任务会 break:
- **In-hand reorientation**: pen 在手里转, 起点终点都是 pen 在手里, 但是中间有复杂的 finger 协调, affine warp 没法 capture
- **Force-controlled task**: push drawer 要持续 force, position-only warp 丢掉 force profile
- **Bimanual coordinated**: 两只手要配合 (例如拧瓶盖), 一只手的运动依赖另一只手的位置, 不是独立 warp

## 5. OKAMI vs ORION 的本质区别

ORION (https://arxiv.org/abs/2405.20321) 的思路是: 视频里有 object motion, 我让 robot 的 palm 重现这个 object motion (考虑 new object location)。把人当成"object motion 的载体", 人本身怎么动不重要, 重要的是 object 怎么动。

OKAMI 的思路是: 视频里有 human body motion, 这个 motion 自身包含 affordance 信息(从哪个方向抓、wrist 怎么转), 我把 body motion retarget 过来, 再根据 new object location 微调。

差别为什么这么大? 看 experiment:
- Place-snacks-on-plate: OKAMI 75% vs ORION 0%
- Close-the-laptop: OKAMI 83.3% vs ORION 41.2%
- Sprinkle-salt (sim): OKAMI 82% vs ORION 0%
- Close-the-drawer (sim): OKAMI 84% vs ORION 10%

ORION 在 pour salt 上 0% 是因为: 它只知道"palm 要移动到 bowl 上方", 不知道"wrist 要转 180 度把盐倒出来" —— 因为 salt 是粉末, 没有可 track 的 keypoint motion。ORION 的 object-centric 视角丢失了 wrist rotation 这个 affordance。

OKAMI 保留 body motion, 所以 wrist rotation 信息直接 retarget 过来。

**Intuition**: 人的 body motion 是 affordance 的高带宽 channel。Robot 看 human 动作, 不仅仅是在看"object 怎么动", 更是在看"human 如何 approach object, 如何 grasp, 如何 manipulate"。这些信息 encode 在 shoulder / elbow / wrist 的小动作里, 全是 free signal, 弃之可惜。

类比: 你看一个老外切菜的视频, 你不仅知道"菜从位置 A 到位置 B", 你还看到他怎么握刀、怎么用腕力。这些是 implicit demonstration, 你不需要额外标注就拿到了。

## 6. 实验告诉我们什么

### 6.1 主要数字

6 个 task, 平均 71.7% success rate, 比 ORION 高 58.3% (绝对数字)。具体 failure mode 论文里写得比较模糊, 大概是 grasp failure + motion execution failure + vision failure 三类。

### 6.2 Vision 是 bottleneck

仿真实验对比 OKAMI with vision vs without vision (假设 ground truth object pose):

| Method | Sprinkle-salt | Close-the-drawer |
|---|---|---|
| OKAMI (w/ vision) | 82% | 84% |
| OKAMI (w/o vision) | 100% | 100% |
| ORION | 0% | 10% |

Ground-truth object pose 让 performance 从 82~84% 飙到 100%, 说明 vision pipeline 是主要 error source。这跟整体 robotics + foundation model 的 trend 一致: perception accuracy 是当前 manipulation 的 bottleneck, 不只是 control policy。

具体 vision 噪声来源:
- RGB-D 在反光/透明表面的 depth noise
- Cutie 在 object 被手遮挡时的 tracking drift
- Grounded-SAM 在 transparent object 上的 segmentation failure

### 6.3 Demonstrator 多样性

3 个不同人录视频, Close-the-laptop 没显著差异, Place-snacks-on-plate 最低比最高低 16.7%。发现是 demonstrator 2 动作太快, SLAHMR reconstruction 噪声大。说明 vision pipeline 对 motion speed 敏感。

这个发现挺有意思 —— 如果你要 scale up 到 internet video, video 里的人动作快慢千差万别, OKAMI 的 SLAHMR 可能崩。这个 limitation 没在 limitation section 里明说, 但其实挺关键。

### 6.4 BC from OKAMI rollouts

OKAMI 生成的 trajectory 拿来训 ACT (https://tonyzhaozh.github.io/aloha/) + DinoV2 backbone 的 visuomotor policy:

- Sprinkle-salt: 50 traj → 65%, 100 traj → 75%
- Bagging: 50 traj → 50%, 100 traj → 80%

这说明 OKAMI rollouts 是有效的 BC training data —— 可以替代人工 teleoperation。Teleoperation 数据收集有多贵看 HumanPlus (https://humanoid-ai.github.io/) 和 Open-Television (https://opentv.github.io/) 就知道, 一个 lab 要收集几百条 humanoid teleoperation demo 是个工程苦力活。OKAMI 把这个 cost 从"人 teleop 几百小时"降到"录一段 video + 跑 OKAMI 自动生成"。

## 7. 几个我想多聊的联想

### 7.1 OKAMI 是"Software 1.5" 路线

Karpathy 你提出的 "software 1.0 vs 2.0 vs 1.5" 框架 (https://karpathy.medium.com/software-2-0-a801460c6e5a), OKAMI 偏 1.5:
- Software 1.0 部分: SE(3) trajectory warping 公式、IK solver、temporal segmentation algorithm、changepoint detection
- Software 2.0 部分: HaMeR (hand pose net)、4D Humans (body pose net)、Grounded-SAM (open vocab seg)、Cutie (VOS net)、DinoV2 (visual backbone)、ACT (transformer policy)、GPT-4V (VLM)
- 接口设计: 把 2.0 模型当"感知 primitive", 把 1.0 算法当"glue"

这种 design pattern 在 robotics foundation model 时代很常见 —— 你不可能纯 end-to-end (sample efficiency 不够), 也不能纯 hand-crafted (generalization 不够), 折中是 structured prior + foundation model primitives。

对比 RT-2 (https://robotics-transformer2.github.io/), RT-2 把所有东西塞进一个 VLA transformer, OKAMI 把感知和决策 decompose 成多个 module, 用 foundation model 做 primitive, 用经典算法做 composition。Trade-off:
- RT-2: 灵活, 但需要海量数据
- OKAMI: 高 sample efficiency, 但受限于 retargeting 假设

### 7.2 SMPL-H 作为 "interlingua"

OKAMI 用 SMPL-H 作为 human motion 的中间表示。这个 design 有个 nice property: 不同 demonstrator 都映射到同一 SMPL-H 空间, retargeting 时只看 SMPL-H, 不关心具体是谁录的。这就是为什么 demonstrator 实验能 work —— motion 先被抽象到 SMPL-H, 个体差异被 wash out。

这个思路可以推广: 任何 imitation learning 系统都应该有个 "interlingua" layer, 把不同 embodiment / 不同 demonstrator 的 motion normalize 到统一表示。Karpathy 你在 Tesla 做 autonomous driving 也有类似思路 —— 不同 driver 行为不同, 但最终都映射到 "trajectory in ego frame" 这个 interlingua, 然后训 policy。

更激进的 interlingua 可以是: motion token (MotionGPT, https://motion-gpt.github.io/, Jiang et al. NeurIPS 2024), 或者 latent motion code (MoMask, https://github.com/EricGuo5513/momask)。这些更 abstract, 但也 lose 了 explicit joint correspondence, 不利于 IK-based retargeting。

### 7.3 人脑 mirror neuron 类比

OKAMI 让我想到人脑的 **mirror neuron** (https://en.wikipedia.org/wiki/Mirror_neuron) —— 猴子看到人抓花生, 自己大脑里抓花生的 motor cortex 也激活。从 watch 到 do 的直接 mapping。

人脑里这个 mapping 大概是 innate 的 (后天 fine-tune 但基本结构先天有), 而且 cross-embodiment (人看猴抓也能学), 这跟 OKAMI 跨 embodiment (human → humanoid) 模仿的精神一致。

但人脑的 mirror neuron 不只做 motion copy, 还做 goal inference —— 看人伸手, 不是映射"伸手这个动作", 而是映射"伸手为了拿杯子"这个 goal, 然后自己根据当前情况决定怎么拿。OKAMI 现在还做不到这个, 它做的是 motion-level mapping + warping, 不是 goal-level reasoning。Goal-level imitation 是 future work —— 可能需要 VLM 不仅做 object naming, 还要做 affordance 和 subgoal reasoning。

### 7.4 跟 VLA 路线的合流

OKAMI 现在是 open-loop (execute reference plan, 不闭环修正)。但 BC policy 部分 (Section 4.3) 已经在训 closed-loop visuomotor policy, 用 DinoV2 + ACT。这相当于: OKAMI 当 "data generator", VLA policy 当 "consumer"。

这跟 LLM 时代 pretraining paradigm 类比: 你可以用规则/弱模型生成大量 synthetic data, 然后用大模型 fine-tune。这里 OKAMI 是"弱规则生成器", DinoV2+ACT policy 是"大模型消费者"。如果 OKAMI 能 scale 到 internet video (去掉 RGB-D 假设), 就变成 "human video → synthetic robot rollout → VLA training data" 的 pipeline, 这是从 internet leverage 数据到 robot foundation model 的关键 link。

类似思路在 OpenAI VPT (https://openai.com/index/vpt/, Baker et al. 2022) 里见过: 从 YouTube Minecraft video 反推 action, 训 downstream policy。OKAMI 是这个 idea 在 humanoid manipulation 上的 version, 只是当前 limited to 受控 RGB-D video。

### 7.5 失败的 case 还能给我们什么 intuition

OKAMI 的 failure mode 包括:
- Grasp failure (hand 形状 mismatch)
- Motion execution failure (warp 后 IK 求解失败)
- Vision failure (object localization 错)

第一个让我想到一个深层问题: humanoid dexterous hand 设计还远未收敛。Inspire hand 6-DoF 跟人 hand (27-DoF if count wrist) 差距巨大, retarget 必然 lossy。如果 hand DoF 太少, 复杂 grasp (三指捏、power grasp 切换) 做不了。这个不是 OKAMI 能解决的, 是硬件 limitation。

第二个让我想到 IK feasibility 问题。SE(3) warp 后的 trajectory 不一定在 robot reachable space 里, 或者 self-collision, 这时 IK 解不出来。OKAMI 没显式处理这种情况, 实际部署时这种 failure 会常见。可以用 **cuRobo** (https://github.com/NVlabs/curobo, NVIDIA 的 GPU-accelerated IK/motion planning) 做 global IK feasibility check, 或者用 learned IK (https://github.com/JeffreyHN/cycle_consistent_robotarium)。

### 7.6 跟 imitation learning 的更深思考

人从单次观察学新技能的能力是个奇迹。心理学叫 "deferred imitation" (https://en.wikipedia.org/wiki/Deferred_imitation), 婴儿 9 个月就能做到。这个能力需要:
- Episodic memory (记住刚才看到的)
- Motor imagery (在大脑里 simulate 动作)
- Cross-modal mapping (视觉动作 → motor command)
- Goal inference (理解对方在干什么)

OKAMI 实现了一个非常浅化的版本: 视频当 episodic memory, SMPL-H retargeting 当 motor imagery, SE(3) warping 当 cross-modal mapping, subgoal segmentation 当 goal inference。每个模块都很 brittle, 但组合起来能在特定任务上 work。

要把这个能力 push 到婴儿水平, 估计需要:
1. Memory: 长视频理解 (现在 OKAMI 假设短演示视频), 用 long-context transformer 或者 retrieval-augmented memory
2. Motor imagery: better human → robot retargeting, 可能需要 neural, 不只是 IK
3. Cross-modal mapping: 不只 SE(3) affine, 还要 learn task-specific warping function (data-driven)
4. Goal inference: VLM 真正理解 task semantics (不只是 object naming)

每一项都是 active research direction。

### 7.7 关于 "Factorized" 的更深层思考

OKAMI factorize 成 body motion + hand pose 两部分。这个 factorization 是 task-agnostic 的 (任何 manipulation task 都适用)。

更细的 factorization 可能更 powerful:
- Approach phase (reaching) vs Contact phase (grasping) vs Manipulation phase (in-hand)
- Each phase 用不同 coordinate frame: world frame (approach), object frame (grasping), palm frame (in-hand)

OKAMI 现在是 subgoal-level factorization (CoTracker changepoint), 但每个 subgoal 内部还是统一 SE(3) warp。如果 subgoal 内部再做 phase-level factorization, 可能能 handle in-hand manipulation 这种 case。

参考 **DexCap** (https://dexcap.cs.columbia.edu/, Wang et al. 2024) 做的 dexterous manipulation 数据收集, 他们就把 motion 分成 approach + grasp + manipulate 三段, 用不同 representation。OKAMI 借这个思路可能能扩展到 in-hand reorientation task。

### 7.8 一个 wild 联想: gestalt 心理学

Gestalt psychology (https://en.wikipedia.org/wiki/Gestalt_psychology) 讲 "整体大于部分之和"。在 robot imitation 里的对应: human motion 不是 joint angle 序列, 是 affordance-driven 的 goal-directed behavior。

OKAMI 现在做 joint-level retargeting, 是 reductionist 的。如果做得 gestalt 一点: 先理解 "人在抓杯子喝水" 这个 high-level goal, 然后根据 robot 自己的 embodiment 决定怎么实现这个 goal。这需要 VLM 不仅做 object naming, 还要做 action understanding + reasoning。

这个方向上有 **EgoExo4D** (https://egoexo4d-data.org/, Grauman et al. CVPR 2024) 这种数据集 (egocentric + exocentric video pair), 可以用来训 VLM 理解 first-person action。如果 OKAMI 接入这种 VLM, 就能从 "看 third-person video 模仿 motion" 升级到 "理解 first-person goal 然后 execute"。

## 8. 我觉得 OKAMI 最值得记住的几个点

1. **Object-aware retargeting** = human motion retarget + SE(3) trajectory warping。前者用 human body 作为 prior source, 后者用 object 作为 anchor point。这是 single-video imitation 的关键 trick。

2. **Factorized retargeting** (body + hand): 用不同 coordinate frame 处理不同 subproblem。这是 strong inductive bias, 让 sample efficiency 飙升。

3. **Foundation model 当 perception primitive, 经典算法当 glue**: HaMeR/4D Humans/Grounded-SAM/Cutie/CoTracker/DinoV2/ACT 这些 model 是 perception primitive, IK + SE(3) warping + changepoint detection 是 glue。这是 humanoid foundation model 时代的 practical design pattern。

4. **Open-loop retargeting → closed-loop BC**: OKAMI rollouts 是 BC training data 的廉价来源, 可以替代 teleoperation 数据收集。

5. **Limitations**: 受限于 SE(3) affine assumption (in-hand manipulation 不行)、stationary camera + RGB-D (internet video 不行)、upper body only (loco-manipulation 不行)、SLAHMR 对 motion speed 敏感。这些 limitations 正好 map 到 future work 方向。

## References

- OKAMI project: https://ut-austin-rpl.github.io/OKAMI/
- ORION paper: https://arxiv.org/abs/2405.20321
- 4D Humans: https://humans4d.is.tue.mpg.de/
- HaMeR: https://geopavlakos.github.io/hamer/
- SLAHMR: https://vye15.github.io/slahmr/
- SMPL-H / MANO: https://smpl.is.tue.mpg.de/
- Grounded-SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything
- Cutie: https://github.com/hkchengrex/Cutie
- CoTracker: https://github.com/facebookresearch/co-tracker
- Pink (IK): https://github.com/stephane-caron/pink
- dex-retargeting: https://github.com/dexsuite/dex-retargeting
- AnyTeleop: https://anyteleop.github.io/
- ACT / ALOHA: https://tonyzhaozh.github.io/aloha/
- DinoV2: https://dinov2.metamind.io/
- Gleicher 1998 (motion retargeting): https://dl.acm.org/doi/10.1145/280814.280820
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a801460c6e5a
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- OpenAI VPT: https://openai.com/index/vpt/
- DexCap: https://dexcap.cs.columbia.edu/
- HumanPlus: https://humanoid-ai.github.io/
- Open-Television: https://opentv.github.io/
- Expressive Whole-Body Control: https://expressive-humanoid.github.io/
- Berkeley Humanoid: https://berkeley-humanoid.com/
- cuRobo: https://github.com/NVlabs/curobo
- EgoExo4D: https://egoexo4d-data.org/
- MimicPlay: https://mimic-play.github.io/
- MotionGPT: https://motion-gpt.github.io/
- MoMask: https://github.com/EricGuo5513/momask
- Changepoint detection (Killick 2012): https://www.tandfonline.com/doi/abs/10.1080/01621459.2012.737745
- Mirror neuron (Wikipedia): https://en.wikipedia.org/wiki/Mirror_neuron
- Deferred imitation (Wikipedia): https://en.wikipedia.org/wiki/Deferred_imitation
- Gestalt psychology: https://en.wikipedia.org/wiki/Gestalt_psychology

---

# OKAMI: 从单个视频教 humanoid robot 学 manipulation skills

## 1. High-level Idea

OKAMI 来自 UT Austin RPL + NVIDIA Research, 解决的核心问题: 给 humanoid robot 一个单条 RGB-D 人类操作视频, 让 robot 模仿执行同一个任务, 并且要 generalize 到不同 object 位置/不同背景/不同 object instance。核心 trick 是 **object-aware retargeting** —— 把人类 body motion retarget 到 humanoid 的时候, 用 object 的实时位置作为 anchor 来 warp trajectory。这跟之前 ORION (https://arxiv.org/abs/2405.20321) 那种"只看 object motion 轨迹重建 + 优化 robot action"的路线是分叉的, OKAMI 走的是"retarget human kinematics + 用 object 信息 refine" 的混合路线。

Project page: https://ut-austin-rpl.github.io/OKAMI/

## 2. Problem Setting: "Open-World Imitation from Observation"

OKAMI 框架下 task 被建模成 MDP $M = (S, A, P, R, \gamma, \mu)$:
- $S$: raw RGB-D observation space (robot + object states 都隐式表示在 image/depth 里)
- $A$: humanoid joint command space (这里是 joint position command, 400Hz 执行)
- $P(\cdot|s,a)$: transition dynamics (未显式建模, 通过 real robot roll out)
- $R(s)$: sparse reward, task 完成 = 1
- $\gamma \in [0,1)$: discount factor (这里实际上没用 RL, 所以 γ 只是形式化)
- $\mu$: initial state distribution, 强调 generalization across configurations

输入: 单条 RGB-D 视频 $V$, 输出: 一个 humanoid policy $\pi$ 完成 $V$ 演示的任务。

"Open-world" 含义: robot 不知道 object 的 category 或者 ground-truth state, "from observation" 含义: $V$ 不带 robot action labels。成功判据: rollout 最终 state 匹配 $V$ 最后一帧的 state。

两个假设:
1. $V$ 里所有 frame 都 capture 人类全身
2. 相机视角 static

这两个假设其实限制了应用场景 —— 不能从 Internet 任意 video 学, 必须是受控录制的。文章 limitations 里也提到了这点。

## 3. Method Pipeline 详解

### 3.1 Stage 1: Reference Plan Generation

这一阶段从 video $V$ 里提取三类信息: (a) task-relevant objects 是哪些, (b) human motion 序列, (c) 把 task 分成 subgoals 形成 plan。

#### 3.1.1 Object Identification

使用 **GPT-4V** (https://openai.com/gpt-4) 做任务相关 object 命名 —— 通过 concatenating 几个 sampled RGB frames 后 prompt, 要求输出 JSON list。这里的关键 insight: GPT-4V 内化了 common-sense knowledge, 即使 object 不和 robot hand 或其他 object 接触也能识别。例如 "sprinkle salt" 任务里 GPT-4V 能识别 salt bottle 和 bowl 两个 object, 即使 salt 这种粉末 object 在视觉上很 ambiguous。

拿到 object names 后:
- **Grounded-SAM** (https://github.com/IDEA-Research/Grounded-Segment-Anything) 用 text prompt 在 first frame 做 segmentation
- **Cutie** (https://github.com/hkchengrex/Cutie) 做 video object segmentation tracking, propagate mask 到所有 frames

#### 3.1.2 Human Motion Reconstruction

这里用了一个改进版的 **SLAHMR** (https://vye15.github.io/slahmr/) —— 原版 SLAHMR 假设 flat hands (手是平的), OKAMI 扩展为同时优化 hand poses。

具体 stack:
1. **4D Humans** (https://humans4d.is.tue.mpg.de/) 给出 initial body pose estimate (Goel et al., ICCV 2023)
2. **ViTPose** (https://github.com/ViTAE-Transformer/ViTPose) 检测 2D hand keypoints
3. **HaMeR** (https://geopavlakos.github.io/hamer/) 从 2D keypoints 估计 3D hand pose (Pavlakos et al., CVPR 2024)
4. Modified SLAHMR joint optimization: 同时优化 SMPL-H 模型的 body location、body pose、hand pose

SMPL-H (https://smpl.is.tue.mpg.de/) 是 SMPL 加上 hand articulation 的模型。优化目标 minimization 2D reprojection error + temporal smoothness + body-hand consistency。

为什么需要 joint optimization? HaMeR 直接出的 hand pose 跟 body reconstruction 可能 inconsistent (wrist 位置和朝向 mismatch)。Reprojection loss 用 HaMeR 预测的 2D hand keypoints 约束 SMPL-H 的 3D hand keypoints 投影。

Runtime: RTX 3090 (24GB) 上 10s video @ 30fps 处理 10 分钟。这个 runtime 对 offline plan generation 可以接受, 但限制了交互性。

#### 3.1.3 Plan Generation via Temporal Segmentation

用 **CoTracker** (https://github.com/facebookresearch/co-tracker, Karaev et al., 2023) 在 segmented objects 上 track keypoints, 计算每 frame 平均 keypoint velocity, 然后用 **changepoint detection algorithm** (Killick et al., 2012, https://www.tandfonline.com/doi/abs/10.1080/01621459.2012.737745) 检测 velocity 突变 —— 这些突变点就是 subgoal keyframes。

每个 subgoal 提取:
- **Target object**: 当前 step 中被操作的 object (基于 average keypoint velocity)
- **Reference object**: 提供 spatial reference 的 object (geometric heuristics 或 GPT-4V semantic reasoning)
- **SMPL-H trajectory segment** $\tau^{SMPL}_{t_i:t_{i+1}}$

最终 reference plan 是一个序列 $l_0, l_1, \ldots, l_N$, 每个 $l_i$ 包含 target object point cloud $O_{\text{target}}$, reference object point cloud $O_{\text{reference}}$, SMPL-H trajectory segment。

Point cloud 通过 RGB+depth back-projection 得到 (用 Open3D, https://github.com/isl-org/Open3D)。

### 3.2 Stage 2: Object-Aware Retargeting

这是 paper 的核心创新。test time 时 robot 面对新的环境, object 位置变了, 但 plan 结构 (subgoals + relative motion) 不变。

#### 3.2.1 Test-Time Object Localization

测试时用同一套 vision pipeline (Grounded-SAM + Cutie + depth back-projection) 在 robot 视野里 localize 任务相关 object, 得到当前 3D point cloud。这里用 RGB-D 主要是为了得到 metric depth —— 如果只是 RGB 没法直接 3D localize。

#### 3.2.2 Factorized Retargeting

OKAMI 把 retargeting 拆成两部分:
1. **Body motion retarget** (arm): 在 task space 操作, 通过 inverse kinematics 求解 joint angles
2. **Hand pose mapping**: 在 joint configuration space 操作, 直接 map SMPL-H finger joint angles 到 dexterous hand joints

为什么 factorize? 这两个 subproblem 的 coordinate frame 选择不同。Arm motion 应该是 object-centric (相对 object 调整), hand pose 是 intrinsic (手指如何 wrap 物体跟 object 相对位置无关, 跟 object shape 和 grasp type 有关)。

Body retarget 用 **Pink** (https://github.com/stephane-caron/pink) 做 IK, IK weights:
- Shoulder orientation: 0.04
- Elbow orientation: 0.04
- Wrist orientation: 0.08
- Wrist position: 1.0

Wrist position weight 高得多 —— 说明 OKAMI 优先保证末端到达目标, 其他关节保持合理姿态。

Hand pose mapping 用 **dex-retargeting** (https://github.com/dexsuite/dex-retargeting, from Qin et al.'s AnyTeleop, https://anyteleop.github.io/):
1. 从 SMPL-H 提取 3D hand joint locations
2. 计算 SMPL-H 各 finger joint rotation angles
3. Apply 到 canonical SMPL-H (size 与 humanoid robot hardware match)
4. dex-retarget 优化 robot finger joint angles

#### 3.2.3 Trajectory Warping —— 数学细节

这是 OKAMI 的核心公式。设原 retargeted robot trajectory 是 $\tau^{robot}(t)$, 起点 $p_{\text{start}} = \tau^{robot}(t_i)$, 终点 $p_{\text{end}} = \tau^{robot}(t_{i+1})$, 都在 SE(3) 空间。

公式 (1):
$$p_t = p_{\text{start}} + (\tau^{robot}(t) - p_{\text{start}})$$

这个公式其实写得很啰嗦 —— 它等价于 $p_t = \tau^{robot}(t)$, 只是强调 "起点 anchor + 偏移" 这个 decompose 视角。

公式 (2) 是 warp 后的版本:
$$p_t = T_{\text{start}} \cdot p_{\text{start}} + (\hat{\tau}^{robot}(t) - T_{\text{start}} \cdot p_{\text{start}})$$

其中
$$\hat{\tau}^{robot}(t) = \frac{\tau^{robot}(t) - p_{\text{start}}}{p_{\text{end}} - p_{\text{start}}} \cdot (T_{\text{end}} \cdot p_{\text{end}} - T_{\text{start}} \cdot p_{\text{start}}) + T_{\text{start}} \cdot p_{\text{start}}$$

变量解释:
- $T_{\text{start}}, T_{\text{end}} \in SE(3)$: test-time object 位置对应的 SE(3) transform。如果只有 target object 移动, $T$ 由 target object 新位置决定; 如果 reference object 也移动, $T_{\text{end}}$ 由 reference object 新位置决定
- $\frac{\tau^{robot}(t) - p_{\text{start}}}{p_{\text{end}} - p_{\text{start}}}$: 原轨迹在 t 时刻的"归一化 progress" (在 SE(3) 中, 实际是 affine parameter)
- 这个 fraction 乘以 new endpoint difference $(T_{\text{end}} \cdot p_{\text{end}} - T_{\text{start}} \cdot p_{\text{start}})$, 把原轨迹相对起终点的偏移 mapping 到新起终点之间的偏移

Intuition: 这是 **affine interpolation in SE(3) space**。原 trajectory 中 t 时刻相对于起点终点的"线性进度"被保持, 但起终点根据 object 新位置 reposition。这种 warping 假设 trajectory 的整体 shape 在 SE(3) 中可以 linearly reparameterize —— 对 pick-and-place、pour 这类 motion 是合理近似, 对需要复杂 in-hand manipulation 的就不行。

**Caveat**: SE(3) 严格说不是 vector space, rotation 部分 linear interpolation 会产生 non-rigid effect (intermediate poses 不在 SO(3) 上)。Practical 上 robotics 论文经常这么写, 但 principled 做法应该是 SLERP for rotation + linear for translation, 或者用 screw theory / log-map 转到 Lie algebra 后 linear interpolation。OKAMI 代码里具体怎么处理这个细节 paper 没明说, 但从 Pink IK 之后被调用来看, 可能最终 IK 把这个小 inconsistency absorb 掉了。

另一个 implicit assumption: $p_{\text{start}} \neq p_{\text{end}}$, 即 trajectory 不是 "原地操作" (例如在原地把 object 翻转)。这种 corner case 对 in-hand reorientation 任务会 break。

### 3.3 关键 Design Choice: Object-Aware vs Object-Only

这是 OKAMI vs ORION 的本质区别。

- **ORION** 路线: 只 retarget palm trajectory (从 SMPL-H 估计), 用 object 新位置 warp, 然后 IK。完全抛弃 body motion 信息。
- **OKAMI** 路线: retarget 全 body motion (arm + shoulder + elbow + wrist + fingers), 然后用 object 位置 warp 整个 arm trajectory。

为什么 OKAMI 显著好? 因为 human body motion encode 了 affordance —— 比如 "抓 snack bag" 时 human 是从 top-down 抓 (因为 snack bag 形状), "倒盐" 时 wrist 需要大幅 rotate。ORION 只 warp palm position, 不知道 wrist 该怎么 orient, 导致 grasp 位置和方向都不对。

实验数据 (Table 1, simulation):
| Method | Sprinkle-salt | Close-the-drawer |
|---|---|---|
| OKAMI (w/ vision) | 82% | 84% |
| OKAMI (w/o vision) | 100% | 100% |
| ORION | 0% | 10% |

OKAMI w/o vision 假设 ground-truth object pose, 看出 vision 误差带来的 performance gap 是 18% 和 16%, 说明 vision pipeline 是主要 bottleneck —— 这给 future work 指明方向 (更好的 vision foundation model)。

## 4. Experimental Setup

### 4.1 Tasks

六个任务覆盖 manipulation spectrum:
1. **Plush-toy-in-basket**: pick-and-place, 软体 object
2. **Sprinkle-salt**: pouring, 需要 wrist rotation
3. **Close-the-drawer**: articulated object, push
4. **Close-the-laptop**: articulated object, 需要二指捏 hinge 区域
5. **Place-snacks-on-plate**: pick-and-place, deformable bag
6. **Bagging**: bimanual dexterous, multi-subgoal

### 4.2 Hardware

- Robot: Fourier GR1 (https://www.fourier-intelligence.com/)
- Hands: 2 × Inspire 6-DoF dexterous hands (https://www.inspire-robohand.com/)
- Camera: Intel RealSense D435i
- Controller: 400Hz joint position controller, 40Hz command generation + interpolation to 400Hz

### 4.3 Evaluation Protocol

- 12 trials per task
- Object locations randomized in robot arm reachable + camera view intersection
- 多 object 场景 (含 distractor)
- New object generalization tested on 3 tasks

## 5. Results Analysis

### 5.1 Main Results (Figure 4a)

平均 71.7% task success rate across 6 tasks。Failure modes 分了几类 (paper 里没列具体数字, 只说大致分布):
- Grasp failure: hand 形状 mismatch 或 object 形状变化大
- Motion execution failure: trajectory warp 后 IK 无法求解
- Vision failure: object localization 错

### 5.2 OKAMI vs ORION (Real Robot)

- Place-snacks-on-plate: OKAMI 75% vs ORION 0%
- Close-the-laptop: OKAMI 83.3% vs ORION 41.2%

ORION 失败模式: "tries to grasp snack from the sides instead of top-down grasp in human video", "failing to rotate the wrist fully for pouring"。这印证了 affordance 缺失问题。

### 5.3 Different Demonstrators (Figure 4b)

三个不同 demonstrator 录视频:
- Close-the-laptop: 无统计显著差异
- Place-snacks-on-plate: 最低 16.7% below 最高 (50% range)

发现 demonstrator 2 motion 较快导致 SLAHMR reconstruction 噪声大。说明 vision pipeline 对 motion speed 敏感, 一个 bottleneck。

### 5.4 Visuomotor Policy via Behavioral Cloning

OKAMI rollout 被用作 BC training data, 用 **ACT** (https://tonyzhaozh.github.io/aloha/, Zhao et al. 2023) 训练 closed-loop visuomotor policy。

ACT hyperparameters (Table 2):
- KL weight: 10
- Chunk size: 60
- Hidden dim: 512
- Batch size: 45
- Feedforward dim: 3200
- Epochs: 25000
- LR: 5e-5
- Temporal weighting: 0.01

Visual backbone: pretrained **DinoV2** (https://dinov2.metamind.io/)
Input: 1 RGB image + 26-dim joint positions
Output: 26-dim absolute joint position

Results (Figure 5):
- Sprinkle-salt: ~65% with 50 trajectories, ~75% with 100
- Bagging: ~50% with 50 trajectories, ~80% with 100

数据越多效果越好, 但 100 trajectories 已经接近 OKAMI open-loop (71.7% avg) 水平。这说明 OKAMI rollouts 是有效的 BC training data source —— 可以替代 teleoperation 数据收集 (teleoperation 是 labor-intensive 的, 见 HumanPlus https://humanoid-ai.github.io/ 和 Open-Television https://opentv.github.io/)。

## 6. Critical Analysis & Intuitions

### 6.1 Object-Aware Retargeting 的 limitation

OKAMI 假设 trajectory 在 SE(3) 中可以 affine reparameterize。这对 "reach → grasp → move to target" 这类 motion 合理, 但对下列 case 会 break:
- In-hand reorientation (object 在手中旋转, 起点和终点位置近似)
- Bimanual coordinated motion (两只手相互配合, 不是简单 warp)
- 力控任务 (push drawer 需要 force feedback, 仅 position 不够)

### 6.2 Vision Pipeline 的脆弱性

依赖 RGB-D + Grounded-SAM + Cutie 这套 stack。问题:
- Grounded-SAM 在 transparent object (例如玻璃杯) 上 segmentation 容易失败
- Cutie 在 object 被 hand 遮挡时 tracking drift
- Depth noise 在远距离或反光表面严重

DinoV2 在 BC policy 里只用 RGB —— 但 plan generation 还是要 RGB-D, 限制了 in-the-wild 应用。

### 6.3 与 Locomotion 的脱节

OKAMI 只做 upper body, lower body 假设固定。Humanoid loco-manipulation (例如边走边抓) 需要 whole-body controller (WBC), 参考 Expressive Whole-Body Control (https://expressive-humanoid.github.io/, Cheng et al. 2024) 和 Berkeley Humanoid (https://berkeley-humanoid.com/, Liao et al. 2024)。

### 6.4 联想: 与其他 video imitation 工作的关系

- **Ditto** (https://arxiv.org/abs/2403.15203): trajectory transformation-based, 跟 ORION 类似, 没 body retargeting
- **MimicPlay** (https://mimic-play.github.io/): human play video → robot plan, 但 single-arm, 不 humanoid
- **VPP** (Vecchioni et al. 2024, https://humanoid-ai.github.io/vpp): video pretraining for humanoid policy
- **HumanPlus** (Fu et al. 2024, https://humanoid-ai.github.io/): humanoid shadowing from human video, 需要 teleoperation 一次
- **Open-Television** (Cheng et al. 2024, https://opentv.github.io/): teleoperation with immersive feedback

OKAMI 的独特 position: **zero-teleoperation** 的 single-video 教学方案, 适用 humanoid bimanual + dexterous hand。这是 fewest-assumption 路线。

### 6.5 联想: 关于 SMPL-H 作为 Interlingua

OKAMI 用 SMPL-H 作为 human motion 的"中间表示"。这个设计有个 nice property: SMPL-H 是 demography-agnostic 的 —— 可以从不同人录的视频 retarget 到同一个 humanoid。Figure 4b 的实验验证了这一点。

进一步联想: 如果把 SMPL-H 换成 **neural motion tokens** (例如 MotionGPT, https://motion-gpt.github.io/, Jiang et al. NeurIPS 2024) 或者 **AMP** (Adversarial Motion Priors, https://github.com/xbpeng/DeepMimic), 可以 encode 更丰富的 motion style 信息, 但失去了 explicit joint correspondence, 不利于 IK-based retargeting。

### 6.6 联想: Foundation Model 的角色

GPT-4V 在 OKAMI 里只做 object naming 和 reference object reasoning —— 这个 role 比较保守。进一步可以:
1. 让 GPT-4V (或 GPT-5V) 直接输出 task plan 的结构化描述 (subgoal + 期望 object state)
2. 用 VLM 做 in-the-wild failure detection 和 replanning
3. 用 VLM 做 object affordance prediction (如何 grasp 这类 object), 替代目前的 hardcoded finger joint mapping

参考 RT-2 (https://robotics-transformer2.github.io/) 和 OpenVLA (https://openvla.github.io/) 的 VLA 路线, 可以把 OKAMI 看成 "structured prior" 而不是 end-to-end VLA 的对比。

### 6.7 联想: 关于 "Factorized" 的物理直觉

为什么 factorize body + hand 是对的?

考虑人抓 cup 的 motion:
- Arm motion (reach): 由 cup 位置决定, 是 extrinsic / scene-dependent
- Hand pose (grasp): 由 cup shape + grasp type 决定, 是 intrinsic / object-property-dependent

这两个 subproblem 在 coordinate frame 上是 decoupled 的:
- Arm motion 在 world frame 或 base frame
- Hand pose 在 wrist frame 或 palm frame

OKAMI 的 factorization 就是把这个 decoupling 显式化。如果一起 retarget (end-to-end learning), 模型需要 implicitly 学到这个 decoupling, 但数据效率低。Factorized design 是 inductive bias, 加速 learning。

### 6.8 公式 (2) 的几何直觉重述

$\hat{\tau}^{robot}(t) = \frac{\tau^{robot}(t) - p_{\text{start}}}{p_{\text{end}} - p_{\text{start}}} (T_{\text{end}} \cdot p_{\text{end}} - T_{\text{start}} \cdot p_{\text{start}}) + T_{\text{start}} \cdot p_{\text{start}}$

可以 reframe 为:
- "Original progress" $\alpha(t) = \frac{\tau^{robot}(t) - p_{\text{start}}}{p_{\text{end}} - p_{\text{start}}} \in [0, 1]$ (在 SE(3) 中, 这个 ratio 是 symbolic 表达, 实际需 group operation)
- "New trajectory" $\hat{\tau}^{robot}(t) = \text{interp}(\alpha(t); T_{\text{start}} \cdot p_{\text{start}}, T_{\text{end}} \cdot p_{\text{end}})$

即: 保持原 trajectory 的"时间归一化进度", 但用新 object 位置定义的起终点做 linear interpolation。这就是 **shape-preserving trajectory morphing**。

类似思想在 computer graphics 中叫 "motion retiming + warping" (Gleicher 1998, https://dl.acm.org/doi/10.1145/280814.280820) —— OKAMI 把这个 idea 借到 robotics, 关键改进是 "shape preserving" 由 original human motion 提供, "endpoint alignment" 由 object localization 提供, 两个 prior 来源 factorize。

### 6.9 联想: 为什么要 4D Humans + HaMeR + SLAHMR 三层

这套 stack 看起来很重, 为什么不 end-to-end?

- **4D Humans** (https://humans4d.is.tue.mpg.de/): 给 full-body coarse pose, 但 hand 是 flat
- **HaMeR** (https://geopavlakos.github.io/hamer/): 给 detailed hand pose, 但跟 body 可能 inconsistent
- **SLAHMR** (modified): joint refinement + smoothness

这种 "coarse-to-fine + joint refine" pattern 在 vision 里常见 (e.g. SMPLify, https://smplify.is.tue.mpg.de/)。优点: 任何单一模型 fail 时有 fallback; 缺点: 复杂、slow、error 累积。

Future direction: 用一个 single transformer 模型 (类似 4D Humans 但 hand 细节) 替代整个 stack, 这需要更大数据集训练。当前 hand reconstruction 数据集规模还小。

## 7. Summary & 个人 Take

OKAMI 的核心贡献是把 "human motion retargeting" 这个传统 CG/robotics 技术, 跟 "open-world vision foundation models" 结合, 实现 single-video humanoid skill learning。Factorized body+hand design 和 SE(3) trajectory warping 是两个关键 inductive bias, 让方法在 sample efficiency 上远胜 end-to-end 路线 (ORION)。

不过 OKAMI 的 limitation 也很明显:
- 假设 stationary camera + RGB-D (Internet video 不能用)
- 假设 affine trajectory reparameterization (in-hand manipulation 不行)
- Vision pipeline heavy, runtime 10s video 处理 10 min
- 只 upper body, no locomotion

但作为一个 "proof of concept" —— 证明 humanoid 可以从 single video 学 manipulation skill 并且 generalize 到 new layout/instance —— 这篇 paper 做得很 solid。

从 Karpathy 的视角, OKAMI 代表了一种 "structured prior + foundation model" 路线, 跟纯 end-to-end VLA 路线 (RT-2, OpenVLA) 形成 contrast。OKAMI 把 human kinematic structure + object affordance + VLM common-sense 显式组合, 而 VLA 全部 implicit in neural network。两种路线 trade-off:
- OKAMI: high sample efficiency, low flexibility (受限于 retargeting 假设)
- VLA: low sample efficiency, high flexibility (可以学任意 skill)

Hybrid 路线 (用 OKAMI 类方法 bootstrap, 再用 VLA fine-tune) 可能是 humanoid foundation model 的 practical path。

## References & Links

- OKAMI project: https://ut-austin-rpl.github.io/OKAMI/
- ORION paper: https://arxiv.org/abs/2405.20321
- GPT-4V: https://openai.com/gpt-4
- Grounded-SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything
- Cutie: https://github.com/hkchengrex/Cutie
- SLAHMR: https://vye15.github.io/slahmr/
- 4D Humans: https://humans4d.is.tue.mpg.de/
- HaMeR: https://geopavlakos.github.io/hamer/
- ViTPose: https://github.com/ViTAE-Transformer/ViTPose
- SMPL-H / MANO: https://smpl.is.tue.mpg.de/
- CoTracker: https://github.com/facebookresearch/co-tracker
- Open3D: https://github.com/isl-org/Open3D
- Pink (IK): https://github.com/stephane-caron/pink
- dex-retargeting / AnyTeleop: https://github.com/dexsuite/dex-retargeting, https://anyteleop.github.io/
- ACT / ALOHA: https://tonyzhaozh.github.io/aloha/
- DinoV2: https://dinov2.metamind.io/
- Fourier GR1: https://www.fourier-intelligence.com/
- Inspire Hands: https://www.inspire-robohand.com/
- RoboSuite: https://github.com/ARISE-Initiative/robosuite
- HumanPlus: https://humanoid-ai.github.io/
- Open-Television: https://opentv.github.io/
- Berkeley Humanoid: https://berkeley-humanoid.com/
- Expressive Whole-Body Control: https://expressive-humanoid.github.io/
- MimicPlay: https://mimic-play.github.io/
- Ditto: https://arxiv.org/abs/2403.15203
- AMP (Adversarial Motion Priors): https://github.com/xbpeng/DeepMimic
- MotionGPT: https://motion-gpt.github.io/
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- Changepoint detection (Killick et al. 2012): https://www.tandfonline.com/doi/abs/10.1080/01621459.2012.737745
- Gleicher 1998 (motion retargeting): https://dl.acm.org/doi/10.1145/280814.280820
