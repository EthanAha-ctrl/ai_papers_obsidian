---
source_pdf: SPOT.pdf
paper_sha256: 31d6995c356bfc1997b98687fbd488a7a53b3e48c4ecda98d7aaae01a0dabaf5
processed_at: '2026-08-12T10:16:00-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SPOT

## 先讲个故事

想象你在教一个外国厨师做一道中国菜——**番茄炒蛋**。

**方法 A**：你给他看 1000 个中国厨师做番茄炒蛋的视频，每个视频都标好"这一秒应该用多大的力、什么角度握锅铲、手腕怎么转"。他看完背下来，然后自己炒。问题是——他用的锅不一样、灶不一样、铲不一样，背下来的手势对不上。这就是 **end-to-end imitation learning** 的困境。

**方法 B（SPOT）**：你告诉他"番茄从砧板到盘子，中间要经过锅里翻炒，番茄的整体姿态轨迹是这样的"。你给他看 8 个视频，但只标注番茄这个物体在 3D 空间里的位置和朝向随时间变化。他自己想办法用他的锅、他的铲让番茄按这个轨迹走。

区别在哪？方法 B 把"**任务的本质**"和"**执行任务的躯体**"解耦了。番茄从 A 到 B 经过 C 这个轨迹是任务的本质，用什么手去实现是执行细节。

SPOT 就是方法 B。

[Project page](https://nvlabs.github.io/object-centric-diffusion)

---

## End-to-end 到底痛在哪

你训练一个神经网络，输入是相机画面 $O_t$，输出是机器人动作 $A_t$。听起来直接，实际上有三个大坑：

**坑 1：数据贵得离谱**
要遥控 Franka Panda 录 1000 条 demo，每条几分钟，一个 task 一个 task 录。YouTube 上有海量人类做菜视频，但没有 robot action label，end-to-end 直接用不了。

**坑 2：换个机器人就废**
在 Franka 上录的 demo 训练出来的 policy，迁移到 Kinova 上基本不行。因为 $A_t$ 是 end-effector 的 pose，不同 robot 的 workspace、kinematics 都不一样。 embodiment 强耦合。

**坑 3：约束学不会**
倒水任务里水壶必须竖直。你给它 1000 条 demo，它可能学到"大部分时候水壶是竖直的"，但偶尔会歪一下水洒出来。为什么？因为 end-to-end 是黑盒，没有任何结构性 prior 告诉它"这个约束是硬约束"。

前人怎么解决？有人用 object pose [YODA](https://yoda-robot.github.io/)，有人用 keypoint [KPAM](https://kpm.yale.edu/)，但他们只关注**最后那一寸**——手已经抓到物体后怎么放。前面的"怎么接近、怎么保持姿态"还是要手写规则。这就像教做菜只教最后装盘那一步，前面的翻炒火候全靠厨师自己悟。

SPOT 的核心贡献：**把整个轨迹都交给数据来学，不要手写规则**。

---

## SPOT 的三个组件，用大白话讲

### 组件一：从视频里挖出"物体轨迹"

拿 iPhone 在厨房录 8 段视频，视频里是**人手**（不是机器人）拿水壶往杯子里倒水。用 [FoundationPose](https://foundationpose.github.io/) 这个 6D pose 估计模型，自动跟踪水壶和杯子在每帧的 3D 位置和朝向。

输出是什么？一串数字：
$$\hat{\tau} = \{\hat{T}_b^a\}_0^l$$

这里 $\hat{T}_b^a$ 是"水壶（$a$）相对于杯子（$b$）的 SE(3) 变换"，$l$ 是视频长度。SE(3) 就是 6 自由度的刚体变换——3 个平移 + 3 个旋转，用 7 维表示（quaternion 4 维 + translation 3 维）。

**为什么用相对 pose 而不是绝对 pose？**

你在厨房录视频，杯子在桌上某个位置；机器人在实验室执行，杯子在工作台另一个位置。绝对坐标完全对不上。但"水壶相对于杯子的位置和朝向"这个关系是 task-intrinsic 的，永远不变。这就是 canonicalization——把所有 demo 归一化到 target 物体的坐标系下。

**Keyframe selection**：人手晃得厉害，dense trajectory 噪声大。PerAct 的方法：物体速度为零（方向改变）或位移旋转超阈值时才存一个 keyframe。这样把 100 帧的轨迹压成 10 个关键点，干净多了。

### 组件二：训练一个 diffusion 模型生成轨迹

这个模型干的事：给它现在的物体 pose，让它生成物体**未来的 pose 序列**。注意，生成的是物体的 pose，不是机器人的 action。

原始 [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 的公式：

$$A_t^{k-1} = \alpha\left(A_t^k - \gamma\,\varepsilon_\theta(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I)\right) \quad (1)$$

人话翻译：
- $A_t^k$ 是 action 在第 $k$ 步去噪过程中的 noisy 版本
- $k$ 是 diffusion 的 step index，从 $K$（纯噪声）一步步去到 0（干净 action）
- $\varepsilon_\theta$ 是神经网络，预测"应该减掉多少噪声"
- $\alpha, \gamma, \sigma$ 是去噪调度器的常数，控制每步去多少噪声、加多少随机性
- $O_t$ 是相机观察，用来 condition 网络生成合适的 action

SPOT 把它改成：

$$\bar{T}_t^{k-1} = \alpha\left(\bar{T}_t^k - \gamma\,\varepsilon_\theta(\bar{T}_t, T_{t+1}^k, k) + \mathcaps{N}(0, \sigma^2 I)\right) \quad (3)$$

变化：
- 条件从 $O_t$（相机画面）换成 $\bar{T}_t$（物体过去的 pose 历史）
- 生成目标从 $A_t$（机器人 action）换成 $T_{t+1}$（物体未来的 pose）

训练 loss 就是标准 DDPM：

$$\mathcal{L} = \text{MSE}\left(\varepsilon_k,\, \varepsilon_\theta\left(T_t,\, T_{t+1}^0 + \varepsilon_k,\, k\right)\right) \quad (4)$$

- $\varepsilon_k$ 是 ground truth 噪声，从 $\mathcal{N}(0, I)$ 采样
- $T_{t+1}^0$ 是 demo 里的真实物体 pose
- $T_{t+1}^0 + \varepsilon_k$ 是加噪后的 pose
- 网络预测噪声 $\varepsilon_\theta$，跟真实噪声做 MSE

**这里有个 subtle 的点**：物体 pose 在 SE(3) 上，是个流形，不是欧式空间。直接在 7 维 quaternion+translation 上做 diffusion 理论上有问题（quaternion 要在 $S^3$ 上归一化）。论文没细说怎么处理，估计是 MLP encoder 把它投影到 64 维欧式空间后就在欧式空间做 diffusion 了——pragmatic 但不严谨。

**End state 预测**：另开一个 MLP，输入当前 pose，输出 [0,1] 表示"任务是否完成"。Deployment 时大于 0.95 就让 gripper 张开。为什么不直接看 pose 是否到 goal？因为多阶段任务里"什么时候算这一阶段完成"是个判断，需要学。

**Multi-task**：用 [CLIP](https://openai.com/research/clip) 把任务描述（如"倒水进杯子"）编码成 text embedding，concatenate 到 pose 输入上。一套 weights 跑 13 个 task。

### 组件三：把物体轨迹变成机器人动作

推理时三个 step 循环：

**Step 1: 跟踪物体**
FoundationPose 实时跟踪物体 6D pose，输出 $T_{cam}^{obj}$（相机坐标系下的物体 pose）。需要物体 3D mesh（用 [BundleSDF](https://bundlesdf.github.io/) 从 demo 重建）和第一帧 bounding box（[YOLO-World](https://github.com/AILab-CVC/YOLO-World) 提供）。

**Step 2: 生成未来轨迹**
Diffusion 模型输入当前 pose $T_t$，输出未来 $N$ 步的 pose trajectory $T_{t:t+h}$。

**Step 3: 转成 action**
假设抓到物体后 gripper 和物体是 rigid attachment（固定连接），那么 end-effector pose 和 object pose 之间差一个固定变换 $T_{EE}^{obj}$。

$$A_{t:t+h} = T_{EE}^{obj} \cdot T_{t:t+h}$$

$A_{t:t+h}$ 就是 end-effector 要走的轨迹，直接送给 robot 的 Cartesian controller。

**Receding Horizon Control**：模型预测 $N$ 步（比如 16），只执行前 $K$ 步（比如 4），然后重新观察、重新规划。这就是闭环控制——物体在 gripper 里滑了、target 被人挪了，下一步重新规划就能跟上。

---

## 实验结果讲讲

### Simulation: RLBench 13 个 task

| Method | 平均成功率 | 相机配置 |
|--------|-----------|---------|
| RVT2 | 76.4% | 4 个相机 |
| 3D Diffuser Actor | 54.5% | 1 个相机 |
| **SPOT** | **79.4%** | 1 个相机 |

SPOT 用 1 个相机干翻了 RVT2 的 4 个相机配置。重点看几个 task：

**Insert Peg（插销钉）**：SPOT 78.7% vs RVT2 44.0% vs 3D-DA 9.3%。这种高精度任务，object-centric 的相对 pose 直接编码了 peg-hole 几何关系。end-to-end 在 raw 3D scene 上学这个关系需要海量数据。

**Stack Blocks（叠积木）**：SPOT 94.0% vs RVT2 81.3%。长 horizon 多阶段任务，object-centric 自然 decompose 成子策略。

**Drag Stick（拖棍子）**：SPOT 80.0% vs RVT2 98.7%。细长物体 FoundationPose 跟丢率高，这是 SPOT 的硬伤。

**Screw Bulb（拧灯泡）**：SPOT 48.0% vs RVT2 86.7%。小物体 + gripper 严重遮挡，tracker 容易 lost，整个 pipeline 就崩了。

**结论**：SPOT 在高精度、长 horizon 任务上赢；在 thin/small/occluded object 任务上输。**bottleneck 是 pose tracker，不是 diffusion model**。

[RLBench](https://github.com/stepjam/RLBench) | [RVT2](https://robot-virtual-transformer.github.io/)

### Real-world: iPhone demo → Kinova 机器人执行

4 个 task，每个 8 个 iPhone demo（人在厨房/客厅录的 RGBD 视频），10 trials 测试。环境、光照、相机视角都跟 demo 时完全不同。

四个任务：
1. **mug-on-coaster**：杯子放杯垫，普通 pick-and-place
2. **plant-in-vase**：仙人掌插花瓶，高精度低 tolerance
3. **pour-water**：水壶倒水，必须保持竖直
4. **put-plate-into-oven**：盘子放烤箱，必须保持水平 + narrow insertion

对比 baseline：[CoTracker](https://cotracker.github.io/) point tracking——在第一帧均匀采样 keypoint，用 RANSAC 估计物体变换。这是 [RoboTAP](https://robotap.github.io/) 和 [Track2Act](https://track2act.github.io/) 那一类方法的代表。

**关键结果**：
- **Mug-on-coaster**：所有方法都 OK，无约束任务
- **Plant-in-vase**：point tracking 噪声大，plant 进不去 vase
- **Pour-water**：point tracking orientation 抖动，水洒；SPOT trajectory 平滑保持竖直
- **Put-plate-into-oven**：point tracking orientation 漂移，食物掉；SPOT 保持水平

**核心 insight**：point tracking 是 redundant + noisy（每个点 3D 位置独立噪声），pose trajectory 是 compact + manifold-constrained（SE(3) 上的刚体变换天然约束）。约束任务（保持水平/竖直）在 pose 空间是个低维子流形，diffusion 容易学；在 point 空间是个高维约束，难学。

---

## 为什么这是"对"的方向

回到番茄炒蛋 analogy。SPOT 揭示一个深层道理：**选对 representation 比改进任何 single component 更重要**。

Rigid object manipulation 的最小充分统计量就是 SE(3) pose。用 pose 作为 perception、planning、control 之间的 interface，每个 module 都可以独立替换：
- Pose estimation：FoundationPose → 未来可能用 [DUSt3R](https://dust3r.eu/) 或 [MASt3R](https://arxiv.org/abs/2406.09756)
- Trajectory model：Diffusion → 可能用 [Flow Matching](https://arxiv.org/abs/2210.02747) 或 [Consistency Model](https://arxiv.org/abs/2303.01469)
- Controller：RHC → 可能用 RL fine-tuning

类比自动驾驶：Tesla 在 [AI Day](https://www.youtube.com/watch?v=j0z4FweC4eU) 讲的"vector space"——所有 sensor 投影到统一 3D 空间后 planning 才能 scale。SPOT 的 object pose 就是 manipulation 的 vector space。

类似思路：
- [UniAD](https://uniad.github.io/)：autonomous driving 用 planning 作为统一接口
- [Wayve VASA](https://wayve.ai/thinking/vasa/)：用 driving scene state space
- [GenSim](https://gen-sim.github.io/)：用 object state 作为 simulation 接口

整个 AI 领域都在收敛到"找对中间 representation"这个方向。End-to-end 听起来酷，但 intermediate representation 才是 scaling 的关键。

---

## 还有什么坑

论文自己承认的：
1. **只 handle prehensile rigid object**——绳子、衣服、 articulated object（抽屉）都不行
2. **依赖 6D pose tracking**——thin/small/occluded object 上 FoundationPose 跟丢
3. **需要 reconstructed mesh**——虽然 BundleSDF 降低门槛，仍是 friction
4. **Rigid attachment 假设**——in-hand manipulation 不行
5. **不可控 force/velocity**——polishing、tight insertion 不支持

我猜测 Karpathy 你会更关心的深层问题：

**Object-centric prior 太强**：contact-rich task（擦拭、打磨）contact patch 和 force 比 pose 更重要，pose trajectory 反而是 misleading representation。论文用 insertion task 验证，但都是 quasi-static 的。

**8 demos 的 data efficiency 可疑**：4 task × 8 demo = 32 demo，能在 100 task 上保持吗？scaling law 没探讨。可能 object pose 的 compactness 帮了 few-shot，但 large-scale 时反而限制表达力。

**FoundationPose 的 generalization bound 未知**：zero-shot 在论文测试物体上 OK，没量化 distribution shift 下的 decay。新物体、新材质、新光照下的 robustness 是个 question mark。

**End state classifier 的 false positive**：0.95 threshold hand-tuned，新 task 未必 robust。多阶段任务的 stage transition 依赖这个 classifier，一旦早触发整个 task 就废了。

**Quaternion sign ambiguity**：$q$ 和 $-q$ 表示同一旋转。Diffusion 训练时如果不 canonicalize，网络会困惑。论文没提怎么处理。

**Mode collapse 风险**：multi-modal demo（同一 task 多种解法）在 diffusion 上虽然比 regression 好，但 high-precision task 仍可能 average 成 failure trajectory。

---

## 我的判断

SPOT 不是 single component breakthrough，是 system integration 的胜利。它把现有 SOTA 组件（FoundationPose、Diffusion Policy、CLIP、RHC）组装成一个 embodiment-agnostic、action-less demo compatible、constraint-aware 的 pipeline。

**真正贡献是 representation choice**：SE(3) pose trajectory 作为 manipulation 的 universal interface。这个思想值得 build intuition on top of——未来 force-aware、deformable、bi-manual 的扩展都会围绕"什么是最小充分 representation"这个核心问题展开。

如果要做 follow-up，我会关注：
1. **Contact-aware representation**：pose + contact patch 联合表示
2. **Deformable object**：用 [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 替代 mesh，support non-rigid
3. **Foundation model 替代 CLIP**：用 [VLM](https://openai.com/research/gpt-4v) 生成 richer task embedding
4. **Latent diffusion on pose manifold**：直接在 SE(3) 流形上做 diffusion，不用投影到欧式空间
5. **Active perception**：tracker confidence 低时让 robot 主动调整视角

[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [GPT-4V](https://openai.com/research/gpt-4v)

---

一句话总结：SPOT 告诉我们，**与其让神经网络直接学"手怎么动"，不如让它学"物体该怎么走"**。前者是肌肉记忆，后者是任务理解。肌肉可以换，任务理解是通用的。

---

# SPOT: SE(3) Pose Trajectory Diffusion for Object-Centric Manipulation 深度解析

## 一、Core Intuition: 为什么需要 Object Pose Trajectory 作为 Intermediate Representation

经典的 imitation learning 范式存在一个**核心矛盾**：end-to-end visuomotor policy（如 Diffusion Policy, RVT, PerAct）直接将 $O_t \mapsto A_t$ 紧密耦合，导致：

1. **Embodiment coupling**：robot demonstration 数据的 action space（joint torques / end-effector poses）与特定 hardware 强绑定，Kinova Gen3 的 demo 没法直接迁移到 Franka Panda；
2. **Cross-camera gap**：training 用 front-facing camera，deployment 用 shoulder-mounted camera，distribution shift 巨大；
3. **Action-less data 浪费**：海量 YouTube 视频、人类手持 iPhone 录制的视频没有 action label，end-to-end 方法直接报废；
4. **Constraint 难以学习**：pouring water 时 kettle 必须保持 upright，这种**中间状态约束**藏在 trajectory 的 sequential structure 里，end-to-end 需要海量数据才能隐式学到，而 object-centric 方法（如 YODA, KPAM）又只关注 last-inch manipulation，前面的 motion planning 要手写规则。

SPOT 的核心 insight：**Object pose trajectory 在 SE(3) 中 inherently encode 了所有 task constraint**。kettle upright 约束等价于 kettle pose relative to target 的 rotation 中 z-axis 始终对齐 gravity；insert peg 约束等价于 peg 的 pose trajectory 沿着 hole 的 axis 方向平移。这种 representation 把"任务是什么"和"用什么 embodiment 执行"彻底解耦。

[Project page](https://nvlabs.github.io/object-centric-diffusion)

---

## 二、Architecture 解析：从 iPhone 视频到 Robot Action 的全 Pipeline

整个系统分为 **Training Phase** 和 **Deployment Phase** 两条 path。

### 2.1 Training Phase: Demonstration Video Parsing

输入是 RGBD video（iPhone Pro + LiDAR 录制，length $l$）。对每个 task 涉及的 object pair $V = \{v_a, v_b\}$（graspable source $v_a$ + target $v_b$），提取 pose trajectory：

$$\hat{\tau} = \{\hat{T}_b^a\}_0^l$$

其中 $\hat{T}_b^a \in SE(3)$ 是 source object 在 target object 坐标系下的相对 pose。这一步用 **FoundationPose** [Wen et al., CVPR 2024] zero-shot 6D object pose estimation。

关键设计：**为什么用 relative pose 而不是 absolute world pose？**
- 把所有 demonstration trajectory normalize 到 target object 的 canonical frame；
- 自动 decouple 了 background / camera extrinsic / scene setup 的差异；
- kitchen 录的 demo 和 office 录的 demo 自动对齐。

**Keyframe selection**（继承 PerAct）：dense trajectory 噪声大、human hand shaking 严重。规则是：
- relative velocity 为零（方向改变点）→ 添加 keyframe；
- 与上一个 keyframe 的 translation 或 rotation 超过 threshold → 添加 keyframe。

[FoundationPose](https://foundationpose.github.io/) | [BundleSDF](https://bundlesdf.github.io/) | [PerAct](https://peract.github.io/)

### 2.2 Training Phase: Object Trajectory Diffusion

这是 SPOT 的核心模型。**不是预测 robot action，而是预测 object 的 future SE(3) pose trajectory**。

原始 Diffusion Policy 的 reverse process：

$$A_t^{k-1} = \alpha\left(A_t^k - \gamma\,\varepsilon_\theta(O_t, A_t^k, k) + \mathcal{N}(0, \sigma^2 I)\right) \quad (1)$$

变量解释：
- $A_t^k$: timestep $t$ 的 action 在 diffusion step $k$ 的 noisy 版本；
- $k \in \{K, K-1, \ldots, 0\}$: diffusion step index，$K$ 是总 steps，$A_t^K \sim \mathcal{N}(0, I)$ 是纯噪声；
- $\varepsilon_\theta$: 神经网络（score function / denoising network），参数为 $\theta$；
- $\alpha, \gamma, \sigma$: denoising scheduler 的常数，控制 step size 和 noise injection；
- $O_t$: observation conditioning（RGBD 或 3D scene）。

SPOT 的修改版本：

$$\bar{T}_t^{k-1} = \alpha\left(\bar{T}_t^k - \gamma\,\varepsilon_\theta(\bar{T}_t, T_{t+1}^k, k) + \mathcal{N}(0, \sigma^2 I)\right) \quad (3)$$

关键变化：
1. **Conditioning** 从 $O_t$（raw sensory）换成 $\bar{T}_t$（historical object pose，过去若干帧 pose）；
2. **Generation target** 从 $A_t$（end-effector action）换成 $T_{t+1}$（object future pose）；
3. 这样模型 $p(T_{t+1} | T_t)$ 是 embodiment-agnostic 的，纯几何动力学。

训练 loss：

$$\mathcal{L} = \text{MSE}\left(\varepsilon_k,\, \varepsilon_\theta\left(T_t,\, T_{t+1}^0 + \varepsilon_k,\, k\right)\right) \quad (4)$$

变量解释：
- $\varepsilon_k \sim \mathcal{N}(0, I)$: 在 diffusion step $k$ 注入的 ground-truth noise；
- $T_{t+1}^0$: clean object pose（ground truth from demo）；
- $T_{t+1}^0 + \varepsilon_k$: noised object pose；
- 模型 $\varepsilon_\theta$ 预测噪声，与 $\varepsilon_k$ 做 MSE，标准 DDPM 训练目标。

**Pose Feature Encoder**：3-layer MLP + LayerNorm + projection head → 64-dim feature。SE(3) pose 通常用 7-dim 表示（quaternion 4 + translation 3），encoder 把它映射到更 dense 的 feature space 便于 diffusion model 处理。

**End State Prediction**：单独的 MLP binary classifier，输入 current pose $T_t$，输出 [0, 1] 表示 task 是否完成。BCE loss 监督。Deployment 时阈值 0.95 触发 gripper open。这个 head 解决了"什么时候停"的问题——object pose trajectory 本身没有 explicit 终止信号。

**Multi-task Language Conditioning**：用 CLIP text encoder 得到 sentence embedding，concatenate 到每个 object pose 输入上。这样一套 weights 可以处理 13 个 task。

**Scheduler**：DDIM，训练 100 timesteps，推理 10 timesteps。3000 epochs，batch size 128。

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | [DDIM](https://arxiv.org/abs/2010.02502) | [CLIP](https://openai.com/research/clip)

### 2.3 Deployment Phase: Closed-loop Policy Execution

三个 component 串联：

**(1) Object Pose Tracking**:
- FoundationPose 实时跟踪所有 task-relevant object 的 6D pose；
- 需要 object 3D mesh（用 BundleSDF 从 demo 重建）和第一帧 2D detection（YOLO-World 提供 bounding box）；
- 输出 $T_{cam}^{obj}$（camera frame 下的 object pose）；
- 转换到 target frame: $T_b^a = (T_{cam}^b)^{-1} \cdot T_{cam}^a$。

**(2) Trajectory Synthesizing**:
- Diffusion model 输入 current pose $T_t$（+ history），输出 future pose trajectory $T_{t:t+h}$（horizon $h$ 步）；
- **Receding Horizon Control (RHC)**：预测 $N$ 步，执行前 $K$ 步（$K < N$），然后 re-plan。这是 closed-loop 的关键——grasping 时 object 在 gripper 里滑动、target 被移动都能 react。

**(3) Action Plan Generation**:
- 关键 transform：$T_{EE}^{obj} \in SE(3)$ 是 object 到 end-effector 的固定 rigid transform（grasping 时确定）；
- 给定 camera extrinsic $T_{cam}^{EE}$ 和 object pose $T_{obj}^{cam}$（即 $T_{cam}^{obj}$ 的 inverse）：
  $$T_{EE}^{obj} = T_{cam}^{EE} \cdot T_{obj}^{cam}$$
- Action sequence：
  $$A_{t:t+h} = T_{EE}^{obj} \cdot T_{t:t+h}$$
- 这样 object pose trajectory 被映射成 end-effector pose trajectory，可以直接送给 robot 的 Cartesian controller。

这里有个 subtle 但重要的点：**$T_{EE}^{obj}$ 假设是 rigid attachment**（gripper 抓住后 object 和 EE 之间固定）。这个假设在 precision assembly 中通常成立，但 in-hand manipulation 或 slipping 时会 break——这也是 RHC 的价值，每步重新 tracking，slippage 会被 detect 并 compensate。

[Receding Horizon Control](https://en.wikipedia.org/wiki/Model_predictive_control) | [YOLO-World](https://github.com/AILab-CVC/YOLO-World) | [SAM-6D](https://sam6d.github.io/)

---

## 三、Experiment 数据深度分析

### 3.1 Simulation: RLBench (Table I)

13 个 task，single Franka Panda + 单 camera（除 RVT2 用 4 cameras）。100 demos 训练，25 unseen configs 测试。

| Method | Avg Succ. | Camera Setup | Key Strength |
|--------|-----------|--------------|--------------|
| RVT2 | 76.4% | 4 cameras | Multi-view 补全 occlusion |
| 3D-DA | 54.5% | 1 camera | 3D scene feature |
| **SPOT** | **79.4%** | 1 camera | Object-centric + trajectory |

**Per-task 分析**：
- **Insert Peg**: SPOT 78.7% vs RVT2 44.0% vs 3D-DA 9.3%。高精度 task，object-centric relative pose 直接编码了 peg-hole 几何关系，远胜 raw 3D scene 的 implicit learning；
- **Place Cups**: SPOT 62.7% vs RVT2 33.3% vs 3D-DA 4.0%。stacking 需要 sequential constraint，trajectory diffusion 自然 capture；
- **Stack Blocks**: SPOT 94.0% vs RVT2 81.3%。Long-horizon 多 stage 任务，object-centric decomposes 成 sub-policies；
- **Drag Stick**: SPOT 80.0% vs RVT2 98.7%。Thin object 是 limitation——FoundationPose 对细长物体 tracking 不稳；
- **Sort Shape**: SPOT 32.0% vs RVT2 49.3%。Mix of symmetric objects，pose estimation 有 ambiguity（symmetric object 的 pose 不唯一）；
- **Screw Bulb**: SPOT 48.0% vs RVT2 86.7%。小物体 + gripper 严重 occlusion，tracker 容易 lost。

这些 failure case 揭示了 SPOT 的 fundamental limitation：**整个 pipeline 的 bottleneck 是 pose tracker**。Tracker 一旦 fail，所有 downstream 都 fail。

[RLBench](https://github.com/stepjam/RLBench) | [RVT2](https://robot-virtual-transformer.github.io/) | [3D Diffuser Actor](https://3d-diffuser-actor.github.io/)

### 3.2 Real-world: iPhone Demo → Robot Execution (Fig. 4)

4 个 task，每个 8 iPhone demos（RGBD + LiDAR，human hand 执行），10 trials 测试。Robot: Kinova Gen3 7-DoF + RealSense D415。

**Failure mode 分类**（设计很 careful）：
- **Tracking failure**: pose tracker lost object，rollout 终止；
- **Placing failure**: object 没到 goal position；
- **Task constraint failure**: object 到了 goal 但中间 constraint violated（如 kettle 倾倒水洒了）。

**Key findings**:
- **Mug-on-coaster**: 所有方法表现类似，pick-and-place 无 constraint；
- **Plant-in-vase**: 高精度 + 低 tolerance，point-tracking baseline 因 noise 导致 plant 进不去 vase；
- **Pour-water**: kettle 必须 upright，SPOT 自动学到这个 constraint（trajectory 数据里 inherent），point-tracking baseline orientation 抖动导致水洒；
- **Put-plate-into-oven**: plate 必须 upright + narrow insertion，SPOT 输出 smooth trajectory，point-tracking baseline orientation 漂移食物掉落。

Point-tracking baseline 用 CoTracker 在第一帧均匀采样 keypoint，RANSAC 估计 rigid transform。这个 baseline 揭示了**点轨迹 vs pose 轨迹**的本质差异：
- 点轨迹 redundant（每个点 3D 位置）+ noisy（个别点 tracking 失败拖累整体）；
- Pose trajectory compact（7-dim）+ 严格 SE(3) manifold 约束（避免 non-rigid drift）。

[CoTracker](https://cotracker.github.io/)

---

## 四、为什么 Object Pose Trajectory 是"正确"的 Representation

Karpathy 你肯定会问：为什么 pose 而不是 flow、keypoint、neural field？让我从 information theory 角度推演。

**Rigid object manipulation 的 sufficient statistic**：一个 rigid object 在 3D 空间的 complete state 就是它的 SE(3) pose（6 DoF）。任何更多 redundant representation（dense flow, point cloud）都是 pose 的 function，但引入了额外的 noise dimension。

数学上：
- Object state: $T \in SE(3)$，7-dim representation（quaternion + translation）；
- Dense point cloud: $N \times 3$ dim，但由 pose 唯一确定（rigid 假设下）；
- 估计 $N \times 3$ 时 noise 在每个 point 独立采样，aggregation 时部分 cancel，但 estimation error 的 manifold 结构复杂；
- 估计 pose 时直接在 SE(3) manifold 上 optimize，natural Riemannian structure。

**Trajectory 编码 constraint**：
- Pouring kettle upright：$R_z(t) \approx I$ for all $t$ until接近 mug；
- 这等价于 trajectory 在 SE(3) 的子流形 $\{T : R \in SO(2)\}$ 上；
- Diffusion model 直接学习 trajectory distribution，constraint 变成 implicit prior；
- 而 last-inch 方法需要手工 hard-code 这个 constraint。

**Diffusion 在 trajectory 上的优势**：
- Multi-modal trajectory distribution（同一 task 多种解法）；
- 比监督回归更 robust to demonstration noise；
- DDIM 10 步推理足够 real-time（~10Hz 控制频率）。

---

## 五、与相关工作的对比定位

| Method | Representation | Last-inch only? | Action-less demo? | Cross-embodiment? | Closed-loop? |
|--------|----------------|-----------------|-------------------|-------------------|--------------|
| Diffusion Policy | End-effector action | No | No | No | Yes |
| RVT2 | 3D virtual view | No | No | No | Yes |
| PerAct | 3D voxel | No | No | No | No |
| YODA | Object pose | Yes | Partial | Yes | No |
| KPAM | Keypoint | Yes | No | Yes | Yes |
| RoboTAP | Point track | No | Yes | No | Yes |
| Track2Act | Point track | No | Yes | No | Yes |
| **SPOT** | **SE(3) pose trajectory** | **No** | **Yes** | **Yes** | **Yes** |

SPOT 的 unique 位置：**唯一同时满足** action-less demo + cross-embodiment + closed-loop + 全 horizon trajectory modeling 的方法。

[YODA](https://arxiv.org/abs/2206.16277) | [KPAM](https://kpm.yale.edu/) | [RoboTAP](https://robotap.github.io/) | [Track2Act](https://track2act.github.io/)

---

## 六、Limitation 与 Future Direction 的 Honest Analysis

论文自己承认的：
1. **只 handle prehensile rigid object**——non-rigid（衣服、绳索）、articulated（抽屉、剪刀）、non-prehensile（pushing）都不行；
2. **依赖 6D pose tracking**——thin object、small object、severe occlusion 时 tracker 失败整个系统崩；
3. **依赖 reconstructed object mesh**——虽然 BundleSDF/DUSt3R 已经降低门槛，但仍是 friction；
4. **Rigid attachment 假设**——in-hand manipulation、compliant object 无法处理；
5. **Constant velocity / force 不可控**——polishing、insertion with force feedback 这类 task 不支持。

我（推测 Karpathy 视角）会补充的更深层 limitation：
- **Object-centric 的 prior 太强**：当 task 本质是 contact-rich（wiping、scrubbing）时，contact patch 和 force 比 pose 更重要，pose trajectory 反而是 misleading representation；
- **Diffusion model 的 mode collapse 风险**：高精度 task 上 multi-modal demo 容易被 average 成失败 trajectory（虽然比 regression 好但仍存在）；
- **FoundationPose 的 generalization bound 未知**：论文用 zero-shot 但没量化 distribution shift 下的性能 decay；
- **8 demos 的 data efficiency 值得怀疑**：4 个 task × 8 demos = 32 demos 总共，是否能在更大 task suite 上保持？scaling law 没有探讨；
- **End state classifier 的 false positive**：threshold 0.95 是 hand-tuned，新 task 未必 robust。

潜在 future work：
1. **加入 force/torque modality**：把 wrist F/T sensor 数据作为额外 condition，处理 contact-rich task；
2. **Neural radiance field as object representation**：替代 mesh，支持 non-rigid；
3. **Tactile-driven closed-loop**：slipping detection 用 tactile 而非 vision；
4. **VLM-guided task description**：替代 CLIP text，用 GPT-4V 类模型生成 richer task embedding；
5. **Hierarchical diffusion**：high-level 生成 sub-goal sequence，low-level 生成 sub-trajectory，处理更长 horizon；
6. **Incorporate uncertainty estimation**：diffusion model 输出 trajectory distribution 而非 point estimate，downstream controller可以 risk-aware planning；
7. **3D foundation model 替换 FoundationPose**：用 DUSt3R / MASt3R 等 metric 3D reconstruction 直接提供 pose，绕过 mesh 重建步骤。

[DUSt3R](https://dust3r.eu/) | [MASt3R](https://arxiv.org/abs/2406.09756)

---

## 七、Implementation Details 你可能会问的

**为什么 DDIM 10 step 推理够用？**
Object pose 是 7-dim compact representation，远比 raw image 或 3D voxel 低维，diffusion trajectory 在低维 manifold 上 10 step 已经收敛。对比 Diffusion Policy 在 2D action space 上也用 ~10 step。

**为什么用 quaternion 不用 Euler angle？**
Euler 有 gimbal lock，且不构成 Lie group，diffusion 在 Euler 空间会有 singularity。Quaternion 是 $S^3$ 流形，配合 LayerNorm 后的 MLP encoder 隐式 handle 了 normalization。

**为什么 RHC 用 $K < N$？**
预测 $N$ 步（如 16），执行 $K$ 步（如 4），然后 re-plan。Trade-off：
- $K$ 太大 → 闭环慢，dynamic uncertainty 反应不及时；
- $K$ 太小 → re-planning 频率高，diffusion 推理成本累积；
- 实践中 $K=4, N=16$ 是 Diffusion Policy 默认值，SPOT 继承。

**Pose feature encoder 为什么 64-dim？**
论文没明说，但推测：
- 7-dim pose → 64-dim feature 是 ~9x expansion，足够保留信息；
- 与 CLIP embedding（512-dim）concatenate 后仍 manageable；
- 64-dim 足够 capture multi-modal trajectory distribution（empirically）。

**FoundationPose 的 latency？**
原 paper 报告 ~10Hz tracking（GPU），与 diffusion 10 step 推理匹配，整体控制频率 ~5-10Hz，对于 quasi-static manipulation 足够。

---

## 八、总结：SPOT 的真正贡献

SPOT 不是 single novel component 的突破，而是 **system-level integration** 的胜利。它把：
- Object pose estimation（FoundationPose）
- Trajectory diffusion（Diffusion Policy 改造）
- Receding horizon control（经典 MPC）
- CLIP language conditioning
- Keyframe selection（PerAct）

组装成一个 **embodiment-agnostic、action-less demo compatible、constraint-aware** 的 manipulation pipeline。

真正的 insight 是：**选择正确的 intermediate representation 比改进任何单 component 更重要**。SE(3) pose trajectory 是 rigid object manipulation 的 minimal sufficient statistic，用它作为 interface 把 perception、planning、control 解耦，每个 module 都可以用 SOTA 方法替换。

这让我联想到你在 [Tesla AI Day](https://www.youtube.com/watch?v=j0z4FweC4eU) 讲的 "vector space" 概念——把所有 sensor data 投影到统一的 vector space 后 planning 才 scale。SPOT 的 object pose trajectory 就是 manipulation 领域的 "vector space"。

类似思想在 autonomous driving 也有：[UniAD](https://uniad.github.io/) 用 planning as unified interface，[VASA](https://wayve.ai/thinking/vasa/) 用 driving scene state space。Manipulation 领域 SPOT 走的是同一条路。

[UniAD](https://uniad.github.io/) | [Wayve VASA](https://wayve.ai/thinking/vasa/)

---

如果你要复现，我建议的路径：
1. 先跑通 FoundationPose 在 iPhone RGBD 视频上的 tracking，确认 pose 估计稳定；
2. 用合成数据（如 RLBench 的 ground truth pose）训练 diffusion model，验证 model capacity；
3. 再迁移到 real iPhone demo，处理 noise；
4. 最后 deploy 到 robot，调 RHC 参数。

最容易踩的坑：
- FoundationPose 的 mesh quality 决定一切，iPhone LiDAR 的 mesh 在 thin structure 上很差；
- Quaternion 的 sign ambiguity（$q$ 和 $-q$ 表示同一 pose）会导致 diffusion 训练不稳定，需要 canonicalize；
- End state classifier 在 multi-stage task 上需要 per-stage 训练。

希望这个解读帮你 build 出 SPOT 的 intuition！如果想要深入某个 component（比如 FoundationPose 内部、Diffusion Policy 的 noise schedule 选择、或 RHC 的具体实现），可以继续追问。
