---
source_pdf: odyssey.pdf
paper_sha256: 942a47d9ad4f2b66592043358654f23ed6a4d7546318ea93688ceb318b03bbef
processed_at: '2026-08-05T22:59:25-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ODYSSEY 用人话说

好，我把 jargon 都剥掉，用大白话重新讲一遍，核心讲"这帮人到底干了什么、为什么这么干、结果怎么样"。

## 一句话说清楚

他们造了一只机器狗背上一条机械臂，能听人话、自己规划路线、走过去抓东西、放东西、拉抽屉、拖小车，室内室外都能干活。整套系统从"听懂人话"到"肌肉怎么动"是一条完整的 pipeline，并且真的跑到了真机上。

项目页：https://kaijwang.github.io/odyssey.github.io/

---

## 为什么要做这件事

之前 mobile manipulation 有几个痛点：

1. **navigation 和 manipulation 是分开做的**。一只机器狗先走到目标旁边停下来，再单独执行抓取，两个 policy 互不通信。问题是，机器狗背上加了一条 arm 之后，arm 一动会推得 base 晃，base 一晃 arm 就抓不准。两个 system 互相打架。

2. **LLM / VLM 用来做 task planning** 这事在 tabletop 上玩得很好了，比如 RT-1、OpenVLA、OmniManip。但是换到 mobile platform 上，camera 是 egocentric（机器人第一人称视角），视角一直在变，VLM 看不全物体，推理能力大幅退化。

3. **之前没有像样的 benchmark**。大家各搞各的 simulation，任务都很短（pick-and-place），没有真正测"robot 听一句话、自己走 10 米、抓东西、再走回去、放到柜子里"这种长程任务。

ODYSSEY 想把这三件事一起解决。参考 RT-1: https://arxiv.org/abs/2212.06817 ; OpenVLA: https://arxiv.org/abs/2406.09246

---

## 系统长什么样

把整个系统想象成三层指挥：

**顶层 GPT-4.1（大脑，慢思考）**
你跟它说："把厨房桌上的可乐拿到客厅的桌子上。"它会把这句话拆成一系列原子动作：
1. navigate to 厨房桌子
2. pick 可乐
3. navigate to 客厅桌子
4. place 可乐

它怎么知道可乐在哪？靠一个实时建出来的"instance graph"——就是机器人边走边用 LiDAR + RGB 摄像头把每个看到的物体打标签、给 3D bounding box，形成一个"我周围有什么"的结构化清单。GPT 就对着这个清单做规划。

**中层 Qwen2.5-VL-72B（眼睛+手，中等速度）**
GPT 说"pick 可乐"，但"pick"具体是 gripper 伸到哪个点、从哪个方向夹？这就要 VLM 上场。它看 wrist 上的 RGB-D 摄像头，输出：
- 一个 2D contact point（图像上点一下要抓哪）
- gripper 的 closing direction 和 approach direction 的描述

然后作者很聪明地加了一层**几何约束**把 VLM 这个模糊输出"修整"成精确 6D pose。比如：
- 如果物体有个主轴（抽屉的滑轨方向），gripper 的 x 轴和 z 轴都得垂直于这个主轴
- 如果物体贴在一个平面（桌面上），gripper 的 z 轴要跟桌面法向量对齐

这一步挺关键的。VLM 自己想精确预测 6D pose 其实不太行，但 VLM 能给出"我大概想从上面抓"这种粗 hint，再用几何约束一投影，就精确了。这种"VLM 出 hint + 几何约束 refine"的思路跟 OmniManip 很像。参考 OmniManip: https://arxiv.org/abs/2502.13143

**底层 RL policy（肌肉，快）**
拿到 6D end-effector target 和 base velocity command 之后，一个 18-DoF 的神经网络同时输出 12 条腿 + 6 个 arm 的关节角度目标，PD controller @ 200Hz 把这些目标转成力矩执行。

这里有个关键设计：**single unified policy**。之前的 RoboDuet 是 locomotion 和 manipulation 两个 policy 分开学，ODYSSEY 把它们放在一个网络里，让网络自己学 arm 和 leg 之间的 dynamic coupling。结果就是 arm 动的时候 base 不晃，base 在斜坡上走的时候 arm 还能稳。参考 RoboDuet: https://arxiv.org/abs/2501.17691 ; Deep WBC: https://arxiv.org/abs/2303.09095

---

## 训练怎么搞

底层 policy 用 RL 在 simulation 里训练，关键 trick 有三个：

**1. 两阶段 curriculum**
- Stage 1（前 2000 iterations）：arm 关掉不动，只学走路。加 gait reward 强制 trot gait（对角腿同步触地），加 frequency reward 强制步频在目标频率附近。这一阶段让 robot 先学会稳稳地走。
- Stage 2：arm 解锁，加入 end-effector pose tracking reward，开始学走路 + 抓东西的联合控制。

为什么这么做？因为 18 DoF action space 太大，一开始就全开学，policy 会卡在 local minimum，比如"蹲着不动让 arm 乱伸"。先学走路再开 arm，类似 baby 先会爬再会抓东西。

**2. Terrain-Invariant End-Effector Sampling（最关键创新）**
训练时给 policy 的 end-effector target 怎么采样？RoboDuet 是在 arm local frame 里采样，问题是 base 在斜坡上 pitch 了，target 跟着 base 一起转，policy 学到的是"在 base 平的条件下抓哪"——一上斜坡就废。

ODYSSEY 改成：在 **world frame** 采样 target，**z 高度在 world frame 固定**，再 transform 到 base frame 给 policy 看。这样 policy 看到的 target 会随 base 姿态"漂移"，policy 自然学到"arm 要怎么补偿 base 的 pitch 才能让 EE 在 world frame 稳住"。

直觉就是：把"在斜坡上保持 EE 不动"这件事的难度，从 policy 学习里移到了 sampling strategy 里。policy 只需要学静态 mapping，不需要学 terrain-dependent 的复杂映射。

**3. Domain Randomization**
训练时把摩擦系数（0.4-2.0）、base 质量（±5kg）、电机 gain（0.8-1.2x）、EE 末端质量（0-0.2kg）等大量随机化，让 policy 对物理参数不敏感。±5kg 对 Go2（15kg 标称）是很大的扰动比例，这种激进随机化是 sim-to-real 成功的关键。

---

## Benchmark 设计

他们做了一个新 benchmark，包括：

- **10 个场景**：5 个 indoor home、2 个超市、1 个餐厅、2 个 outdoor 院子（带斜坡和楼梯）
- **物体资产**：50 个 rigid object、15 个 container、30 个 articulated structure（柜子门）、10 个 draggable 物体（小车椅子）
- **4 个 ARNOLD short-horizon task**：pickup、reorient、open cabinet、close cabinet
- **8 个 long-horizon task**：indoor collect、room navigation、cart delivery、cabinet storage、restocking、shopping、outdoor collect、outdoor delivery

每个 long-horizon task 有 2-3 个 subgoal，总共 246 个 indoor variation + 58 个 outdoor variation。

评估很细，不只看任务整体成败，还拆成每个 atomic action 的成功率。比如 CARTDELIVERY 拆成：nav to object → pick object → nav to cart → place object → drag cart → nav to goal，每一步都监控。

---

## 实验结果解读

### Short-Horizon vs PerAct (Table 1)

PerAct 是 ARNOLD 上的 SOTA，但它是 closed-set 3D voxel 方法，训练时见过那些物体。ODYSSEY 用单个 egocentric camera + VLM。

| 任务 | PerAct Seen | ODYSSEY Seen | PerAct Novel | ODYSSEY Novel |
|---|---|---|---|---|
| Pick | 94.0 | 60.5 | 25.7 | 45.2 |
| Reorient | 19.5 | 51.3 | 8.2 | 52.1 |
| Open Cab | 31.1 | 56.3 | 16.6 | 51.1 |
| Close Cab | 60.8 | 74.3 | 41.3 | 79.5 |

人话解读：
- **Seen 上** PerAct pick 高（94 vs 60），因为 ARNOLD 训练集里见过那些物体，PerAct 记住了
- **Novel 上** ODYSSEY 全线碾压，因为 VLM 是 open-world 的，没见过的物体也能识别和推理
- **Reorient** PerAct 直接崩盘（19→8），因为 voxel 方法对动态物体旋转很弱
- **Cabinet** ODYSSEY 跨场景稳定在 50-80，VLM 的语义理解起了大作用

更细的 Table 8 显示，给 ODYSSEY 用 ground-truth grasp pose（Phase II）时，pick/reorient/close cabinet 都能到 80%+。说明**底层 control 没问题，瓶颈在 VLM 的 grasp pose 预测**。

### Long-Horizon (Table 2)

8 个 task 的 overall success rate：

| 任务 | Overall | 最难一步 |
|---|---|---|
| Room Nav | 69.8% | push 94.1% |
| Indoor Collect | 66.7% | pick 72.7% |
| Outdoor Collect | 63.3% | pick 69.0% |
| Restocking | 56.7% | pick 83.3%, place 79.2% |
| Shopping | 47.5% | drag 79.2% |
| Outdoor Delivery | 46.4% | pick 72.7%, drag 85.7% |
| Cabinet Storage | 44.9% | pick 79.6%, place 83.8% |
| Cart Delivery | 41.0% | drag 69.2% |

人话解读：
- 长程任务 success 是每一步 success 的乘积。就算每步 80%，5 步串联就只剩 32%。CARTDELIVERY 41% 其实已经很可观
- 最难的 CARTDELIVERY 涉及 navigate + pick + place + drag，drag 69% 是瓶颈
- 每个 atomic action 基本都 >60%，说明系统各模块都 work，只是叠加起来损耗
- outdoor 任务（63%, 46%）和 indoor（67%, 45%）相当，说明 terrain-adaptive 真的起作用了

### Low-Level vs RoboDuet (Table 3)

| 指标 | RoboDuet 站着 | ODYSSEY 站着 | RoboDuet 走着 | ODYSSEY 走着 |
|---|---|---|---|---|
| $e_x$ (cm) | 0.32 | 0.08 | 9.70 | 0.36 |
| $e_y$ (cm) | 0.34 | 2.69 | 15.42 | 2.31 |
| $e_\omega$ (deg) | 0.32 | 0.26 | 60.59 | 0.79 |
| $D_{pos}$ (cm) | 11.08 | 11.48 | 10.75 | 10.57 |
| $D_{ori}$ (deg) | 47.14 | 46.93 | 47.53 | 47.15 |

人话解读：
- **base tracking 走着的时候**：RoboDuet 直接崩了（$e_\omega$ 从 0.32 飙到 60.59），ODYSSEY 几乎不变（0.32→0.79）。这就是 single policy 学到 coupling 的体现——arm 动的时候 base 不会乱晃
- **EE tracking**：两者差不多，~11cm 位置误差，~47° 朝向误差
- 47° 朝向误差听起来很大，但其实是因为 Euler angle 表示对某些 rotation 很敏感，而且 gripper 自身有 ±20-30° 的抓取容差，实际抓取能成功
- ODYSSEY 训练时 workspace 比 RoboDuet 小（Table 4），evaluation 时用更大的 workspace，仍然 work，说明 generalization 强

---

## Sim-to-Real

硬件配置：
- Unitree Go2（12 DoF leg, 15kg, 标称 payload 8kg）
- Arx5 arm（6 DoF, 3.35kg）背在 Go2 上
- Unitree L1 LiDAR（built-in, 朝下）建 occupancy map
- Livox MID-360 做 high-level localization
- RealSense D435i 在 head 上做 RGB scene understanding
- RealSense D405 在 gripper 上做 close-range RGB-D manipulation

控制：50Hz policy，200Hz PD

真机跑了 "navigate to pick" 和 "pick and place" 两个长程任务，5 种物体，成功 transfer。失败主要在小物体抓取——EE tracking 误差 + depth 感知不准叠加起来。

参考 Unitree Go2: https://www.unitree.com/go2 ; RealSense D405: https://www.intelrealsense.com/depth-camera-d405/

---

## 我觉得哪几个设计最聪明

1. **Terrain-invariant sampling**：简单但 powerful。把"斜坡上保持 EE 稳定"的复杂度从 policy 移到 sampling strategy，policy 只学静态 mapping。这种"把 hard problem 的 structure 提取出来让 learner 只学简单部分"的思路在 RL 里通用。

2. **VLM + 几何约束**：VLM 单独预测 6D pose 不靠谱，但 VLM 给粗 hint + 几何约束 refine 成精确 pose，是 pragmatic 的工程方案。比纯 VLM 稳，比纯 analytic 方法 general。

3. **Single unified policy**：避开了 dual policy 的协调问题，让 network 自己学 coupling。代价是训练更难（需要 curriculum），但 deployment 简单。

4. **Instance graph + GPT-4.1 task decomposition**：把 LLM 当成"有限 action set 的 program synthesizer"，比让 LLM 直接输出 continuous action 更可靠。

---

## 我觉得哪里还有问题

1. **$D_{ori} \approx 47°$ 误差**：在 long-horizon 上会累积，CARTDELIVERY 41% 的 bottleneck 可能在这里。底层 EE orientation tracking 还有很大提升空间。

2. **VLM latency**：Qwen2.5-VL-72B 一帧推理可能 1-2s，paper 完全没提这个。对 reactive 任务（物体掉了要马上调整）这种延迟很致命。实际部署要么蒸馏小模型，要么用 VLM 做一次规划 + 轻量 policy 做 closed-loop correction。

3. **VLM 对 slim handle / partial occlusion 弱**：failure analysis 里提到这是主要失败原因。而日常任务里这种场景很多（拉抽屉细把手、被遮挡的杯子）。VLM 的 fine-grained spatial reasoning 还得继续提升。

4. **Gait reward / frequency reward 没 ablation**：加了这么多 shaping reward，但没做 ablation 证明每个都必要。可能有些 reward 其实没用甚至有害。

5. **Domain randomization 范围**：friction [0.4, 2.0] 没覆盖冰面（0.1）或湿地（0.2）。真实 outdoor 可能 OOD。

6. **Long-horizon success 41-70%**：距离"通用 home robot"还远。但作为 first comprehensive benchmark，这个数字建立了 baseline，后续工作可以对着打。

---

## 跟其他工作的关系

- **vs RoboDuet**：ODYSSEY 用 single policy + terrain-invariant sampling，RoboDuet 用 dual policy + base-centric sampling。ODYSSEY 在 base tracking 上完胜，EE tracking 持平
- **vs OmniManip**：都用 VLM + 几何约束做 pose grounding，但 OmniManip 是 tabletop，ODYSSEY 扩展到 mobile + terrain
- **vs Mobile ALOHA**：Mobile ALOHA 用模仿学习从遥操数据学，需要大量人类 demo；ODYSSEY 用 RL + VLM，不需要 demo 但需要 sim
- **vs UMI on Legs**：UMI on Legs 把 manipulation policy 从 demo 学的，再 attach 到 locomotion controller；ODYSSEY 是 end-to-end 学联合控制
- **vs WildLMA**：WildLMA 也做 long-horizon loco-manipulation，但用 per-action policy 从 demo 学，缺 compositionality；ODYSSEY 用 atomic action + VLM grounding 更 scalable

参考 Mobile ALOHA: https://arxiv.org/abs/2401.02117 ; WildLMA: https://arxiv.org/abs/2411.15131 ; UMI on Legs: https://arxiv.org/abs/2407.10353

---

## 一句话评

ODYSSEY 把 LLM task planning、VLM contact grounding、几何约束 pose refine、terrain-invariant RL policy 这四块拼成一个能 sim-to-real 的完整 mobile manipulation 系统。技术 contribution 主要是 terrain-invariant sampling 和 geometry-constrained VLM grounding 这两个工程 trick。但 long-horizon success 41-70% 说明离通用 home robot 还远，主要瓶颈在 VLM 的 fine-grained spatial reasoning 和底层 EE orientation tracking 精度。作为 first comprehensive benchmark，它建立了 baseline，后续工作可以对着打。

如果让我赌一个最有可能突破的方向，是 paper 5.2 提到的 **active perception**——robot 在 navigate 阶段就主动调整视角去 peek 物体的侧面背面，VLM 看到更多 view 再 manipulate。当前"走过去停下再抓"的范式限制了 VLM 的信息量，而 mobile platform 天生适合多视角主动感知。这种"先 look 再 act"的 exploration 可能是下一个 big thing。参考 Active Perception: https://arxiv.org/abs/2306.16661

---

# ODYSSEY 深度技术解析

Andrej, 这篇 paper 我仔细读完了, 是一个相当完整的 mobile manipulation system, 把 hierarchical VLM planning 和 terrain-adaptive whole-body RL control 缝合在一起。下面我从架构直觉、关键公式、实验数据三个层面展开, 尽量 build 你的 intuition。

## 1. 整体架构直觉

ODYSSEY 的核心 pipeline 可以拆成三层, 每层解决不同时间尺度的问题:

- **Slow loop (秒级到分钟级)**: GPT-4.1 在 instance-level semantic graph 上做 long-horizon task decomposition, 输出 atomic action sequence (navigate / pick / place / push-pull-drag)
- **Medium loop (百毫秒级)**: Qwen2.5-VL-72B 在 wrist-mounted RGB-D 上做 geometry-constrained end-effector pose grounding
- **Fast loop (20ms, 50Hz)**: 单一 18-DoF RL policy 同时输出 12 leg joints + 6 arm joints 的 PD targets

这里的关键 insight 是: 不像 RoboDuet 用 dual policy (locomotion policy + manipulation policy 分开), ODYSSEY 用 **single unified policy**。为什么 single policy 更好? 因为 quadruped + arm 之间存在强 dynamic coupling——arm 的惯性反作用会扰动 base pose, 而 base 的 pitch/roll 又会改变 arm 的 reachable workspace。把两者放在一个 network 里, policy 可以学到这种 coupling 的 compensator。

参考链接:
- RoboDuet: https://arxiv.org/abs/2501.17691
- Deep Whole-Body Control (Fu et al. 2023): https://arxiv.org/abs/2303.09095
- OmniManip (Pan et al. 2025): https://arxiv.org/abs/2502.13143
- ARNOLD benchmark: https://arnold-benchmark.github.io/
- UMI on Legs: https://arxiv.org/abs/2407.10353

## 2. Hierarchical Planner 的核心设计

### 2.1 Instance-level Semantic Graph 构建 (Appendix A.1)

这个 module 是 long-horizon planning 的 semantic substrate。流程:
1. LiDAR @ 10Hz → 全局 point cloud map (SLAM-based)
2. RGB + RAM++ (multi-grained tagger) → 候选 object labels
3. Grounding-SAM (用 Mobile-SAM 替换标准 SAM 提速) → instance mask $m_i$
4. MaskCLIP → 每个 mask 的 visual-semantic descriptor $\mathbf{f}_i \in \mathbb{R}^d$
5. Mask 反投影到 LiDAR frame → per-object 3D point segment
6. 跨时间融合: semantic + geometric 双重匹配

**Semantic Similarity (Eq. 9)**:
$$\mathrm{sim}_{sem}(\mathbf{f}_i, \mathbf{f}_j) = \frac{\mathbf{f}_i^\top \mathbf{f}_j}{\|\mathbf{f}_i\| \|\mathbf{f}_j\|} > \tau_{sem}, \quad \tau_{sem} = 0.8$$

- $\mathbf{f}_i, \mathbf{f}_j \in \mathbb{R}^d$: 是 mask-pooled CLIP feature, $d$ 通常是 512 (CLIP ViT-B) 或 768
- $\tau_{sem} = 0.8$: cosine similarity threshold, 这个值在 ConceptFusion / ConceptGraphs 体系里是经验值, 0.8 大概对应"明显同类"
- $^\top$: 转置, 即内积

**Geometric Similarity (Eq. 10)**:
$$\frac{1}{|\mathcal{P}_i|} \sum_{\mathbf{p} \in \mathcal{P}_i} \mathbb{I}\left[\min_{\mathbf{q} \in \mathcal{P}_j} \|\mathbf{p} - \mathbf{q}\| < \epsilon\right] > \tau_{geo}, \quad \tau_{geo} = 0.8$$

- $\mathcal{P}_i, \mathcal{P}_j$: 两个 candidate 的 3D point set
- $\mathbb{I}[\cdot]$: indicator function, 满足条件为 1, 否则 0
- $\epsilon$: 邻域半径, 没明说但通常是 5-10cm 量级
- 阈值 0.8 意味着 80% 的点能找到对应对面, 这是单向的 (i→j), 没有做 symmetric chamfer, 但作者应该按时间顺序只在新 detection 出现时检查是否合并到旧 instance

直觉: semantic 防止把不同 class 的物体合并 (例如椅子 vs 凳子), geometric 防止把同类但不同 instance 合并 (例如两把椅子)。两个 criterion 都满足才 merge, 体现保守策略。

参考 ConceptGraphs: https://concept-graphs.github.io/

### 2.2 GPT-4.1 Task Decomposition

输入: user instruction (template-free NL) + instance graph summary
输出: atomic action sequence from {navigate, pick, place, push/pull/drag}, 每个附带 language description + coarse waypoint (对 navigate/drag)

Coarse waypoint 投影到 2D occupancy map, 在 waypoint 周围 local search 找 collision-free goal pose。这里其实就是把 LLM 当成有限的"program synthesizer", action space 是 closed set, 这跟 SayCan / Code as Policies 思路类似, 但加了 spatial grounding。

参考 SayCan: https://say-can.github.io/
参考 Code as Policies: https://code-as-policies.github.io/

### 2.3 Geometry-Constrained Local Manipulation

这是我觉得这篇 paper 最有意思的设计。VLM (Qwen2.5-VL-72B) 不直接输出 6D pose, 而是输出:
- 2D contact point $p^* \in \mathbb{R}^2$ 在 image space
- gripper closing direction (x-axis) 和 approach direction (z-axis) 的描述

然后用**几何约束** refine 成完整 $\mathbf{R}_{ee} \in SO(3)$。

**Axis-alignment constraint (Eq. 1)**:
$$\mathbf{r}_x^\top \mathbf{a} = 0, \quad \mathbf{r}_z^\top \mathbf{a} = 0$$

- $\mathbf{r}_x, \mathbf{r}_z \in \mathbb{R}^3$: end-effector frame 的 x 轴和 z 轴在 world/robot frame 下的单位向量
- $\mathbf{a} \in \mathbb{R}^3$: target object 的 dominant axis (比如抽屉的滑轨方向, 杯子的把手方向)
- $^\top$ 内积等于 0 意味着正交

直觉: gripper 不能沿 object 的主轴方向 "夹歪", 必须垂直地接近, 这样夹住后才能 pull/push 沿 $\mathbf{a}$ 方向。

**Surface-normal constraint (Eq. 2)**:
$$\mathbf{r}_z \parallel \mathbf{n}, \quad \text{s.t.} \quad \mathbf{r}_z^\top \mathbf{a} = 0$$

- $\mathbf{n} \in \mathbb{R}^3$: 物体附着平面的 normal (e.g. 桌面 normal = [0,0,1])
- $\mathbf{r}_z$ 要 align 到 $\mathbf{n}$, 同时不能违反 axis-alignment

直觉: gripper 的 approach 方向应该垂直 surface (从上往下抓桌上物体), 但还要保持对 object 主轴的正交关系。这两个约束联立定义了 $\mathbf{r}_z$ 在 surface plane 上的可能方向集合 (一个圆), 再用 VLM 的 hint 选一个具体角度。

这种"VLM 给 hint + 几何约束做 projection"思路, 与 OmniManip 的 object-centric interaction primitives 思想相通, 都是把 VLM 的弱空间推理用解析约束变成精确的 6D pose。

contact point 投影公式隐含的:
$$\mathbf{P}_{ee} = \pi^{-1}(p^*, D(p^*))$$
其中 $D$ 是 aligned depth image, $\pi^{-1}$ 是 back-projection。

## 3. Whole-Body RL Policy

### 3.1 Policy formulation

**Observation (Eq. 4)**:
$$\mathbf{a}_t = \pi(\mathbf{c}_t, \mathbf{e}_t, \mathbf{s}_t, \mathbf{g}_t, \mathbf{m}_t, \mathbf{a}_{t-1})$$

变量:
- $\mathbf{c}_t = (\hat{x}, \hat{y}, \hat{\omega})$: base velocity command, 线速度 2D + 角速度
- $\mathbf{e}_t$: 6-DoF end-effector target (position + orientation)
- $\mathbf{s}_t = (\mathbf{q}_t, \dot{\mathbf{q}}_t) \in \mathbb{R}^{36}$: proprioceptive state, 18 joints 的位置 + 速度
- $\mathbf{g}_t$: projected gravity vector in base frame, 3 维, 提供 base pitch/roll 信息
- $\mathbf{m}_t$: local ground height map, 编码 terrain
- $\mathbf{a}_{t-1} \in \mathbb{R}^{18}$: previous action, 用于 smoothness

**Action**:
$$\mathbf{q}_t^{target} = \mathbf{q}^{default} + \mathbf{a}_t$$

直觉: action 是 default joint pose 的 offset, 这是 Fu et al. 2023 (Deep WBC) 提出的 trick, 让 policy 学习相对默认姿态的小偏移, 而不是从零开始学绝对 pose。好处:
1. 初始化时机器人不会乱跳
2. policy 输出空间被 implicit bias 到合理 region
3. sim-to-real 时如果 sim 的 default pose 和 real 不完全一致, offset 的形式仍然能用

PD controller @ 200Hz 把 target joint position 转成 torque:
$$\tau = K_p (\mathbf{q}^{target} - \mathbf{q}) + K_d (0 - \dot{\mathbf{q}})$$
($K_p, K_d$ 是 gain, paper 没明说, 但 Unitree Go2 标准 PD gain 大约 $K_p \approx 20-60$ N·m/rad, $K_d \approx 0.5-2$ N·m·s/rad)

### 3.2 Two-Stage Curriculum

**Stage 1 (0 ~ 2k iterations)**: arm joints 固定, 只学 locomotion
**Stage 2 (2k+)**: 解锁 arm, 全 18 DoF

Stage 1 的关键 reward:

**Gait reward (Eq. 5, 11-17)**:
$$r_{gait} = \prod_{i,j \in \text{sync pairs}} r_s(i,j) \cdot \prod_{k,l \in \text{async pairs}} r_a(k,l)$$

- sync pairs = {(FL, RR), (FR, RL)}: 对角腿, walk/trot gait 下应该同步触地
- async pairs = 其他所有组合 (FL,FR), (FL,RL), (FR,RR), (RL,RR): 不应该同时触地

子项:
- $t_{air}^s(A,B) = \text{clip}((A_{air} - B_{air})^2, 0, 0.04)$: 两腿 air time 差, clip 在 0.04 (相当于 0.2s 的平方)
- $t_{cont}^s(A,B) = \text{clip}((A_{cont} - B_{cont})^2, 0, 0.04)$: 两腿 contact time 差
- $r_s(FL,RR) = e^{-(t_{air}^s + t_{cont}^s)}$: sync pair 奖励
- async pair 同理, 但用 cross term $(A_{air} - B_{cont})^2$, 鼓励一腿在 air 时另一腿在 contact

直觉: clip 在 0.04 是 saturation, 防止偶尔大扰动破坏整个 reward 信号; exponential form 让 reward 始终为正且对 small error 敏感, 对 large error 鲁棒。乘积形式 (而非和) 意味着任何一个 pair 不协调都会拖累整体, 强制所有 pair 都满足约束。

**Frequency reward (Eq. 6-8)**:
$$\text{err}(\mathbf{leg}) = (f(\mathbf{leg}) - f_{target})^2$$
$$r_f(\mathbf{leg}) = \exp(-0.5 \cdot \text{err}(\mathbf{leg}))$$
$$r_{fre} = \prod_{\mathbf{leg} \in \{FL, FR, RL, RR\}} r_f(\mathbf{leg})$$

- $f(\mathbf{leg}) = 1 / (t_k^{cont} - t_{k-1}^{cont})$: 当前 stride 频率, 即相邻两次触地时间间隔的倒数
- $f_{target}$: 目标步频, 通常 2-3 Hz for quadruped
- $0.5$: temperature, 控制 reward 对频率偏差的宽容度

直觉: 单独的 gait reward 只约束 contact pattern, 不约束步频, robot 可能学到很慢或很快的步态。frequency reward 把 cadence 拉到目标频率附近。

**Tracking reward (Appendix B.1)**:
- $r_{xy}^{track} = \exp(-((\hat{\mathbf{v}}_{xy} - \mathbf{v}_{xy})^2 / \gamma_{xy}))$, weight 2.75
- $r_{yaw}^{track} = \exp(-((\hat{\omega} - \omega)^2 / \gamma_{\omega}))$, weight 1.50

$\gamma_{xy}, \gamma_{\omega}$ 是 scale factor, 控制宽容度。

Stage 2 加入:
- $r_{ee-pos}^{track} = \sqrt{(\hat{\mathbf{p}}_{ee} - \mathbf{p}_{ee})^2}$, weight -1.20 (注意是负数, 表示 penalty, L2 distance)
- $r_{ee-ori}^{track} = \sqrt{(\hat{\phi}_{ee} - \phi_{ee})^2}$, weight -1.50

Stage 1 weight=0, Stage 2 启用, weight 为负说明写成 penalty 形式而非 reward, 用 negative weight 拉近 EE pose。

**Regularization** (smoothness, torque, power):
- $r_{smooth} = -\sqrt{(\mathbf{a}_t - \mathbf{a}_{t-1})^2}$, weight -0.02, 鼓励 action 平滑
- $r_{torque}^{base} = -|\tau^{base}|^2$, weight -2.0e-4
- $r_{power}^{arm} = -|\tau^{arm}| \cdot |\dot{\mathbf{q}}^{arm}|$, weight -2.0e-4

直觉: 所有 regularization 都用很小 weight, 主要起 fine-grained 作用, 防止 policy 用极端 torque 或高频抖动来满足 tracking reward。

### 3.3 Terrain-Invariant End-Effector Sampling (核心创新)

这是与 RoboDuet 的关键差异。RoboDuet 在 **arm local frame** 采样 target, 导致:
- 当 base 在斜坡上 pitch 时, target 跟着 base 转, 实际 world-frame 位置变了
- robot 必须学: "我想抓 world-frame 高度 0.5m 的物体, 但我现在 pitch=15°, 所以 arm target 要相应 transform"
- 这个 transform 增加了 policy 学习负担

ODYSSEY 改成:
1. 在 **world frame** 采样, 但 z 高度固定 (例如 0.3-0.8m)
2. 转换到 base frame 时, base 的 pitch 影响 transform matrix, 但 z 在 world frame 是 ground truth
3. policy 看到的 target 在 base frame 里"漂移", 但漂移量精确反映了 base 的当前姿态
4. policy 学到的是 "arm joint 怎么变来补偿 base 姿态以保持 world-frame EE pose"

直觉: 这其实把 "在斜坡上保持 EE 不动" 这个任务的 complexity 从 policy 移到了 sampling strategy。policy 只需要 learn 静态 mapping "base frame target → joint pose", 不需要 learn terrain-dependent 映射。

Table 4 显示 evaluation range 比 train range 大很多 (例如 $\hat{x} \in [-1.5, 1.5]$ vs train $[-1.0, 1.0]$), 验证 generalization。

### 3.4 Domain Randomization

关键参数:
- Friction: [0.4, 2.0], 涵盖滑/涩地面
- Base Mass: $\pm 5$ kg additive, 模拟 arm 重量变化
- Base Pushing: 持续 interval force $[-0.5, 0.5]$ N
- Actuator Gains: $[0.8, 1.2]$ scale, 模拟 motor 老化
- Ee Link Mass: $[0, 0.2]$ kg additive, 模拟抓取物体重量

$\pm 5$ kg 对 Go2 (15kg) 是很大的扰动比例, 这个激进的 randomization 是 sim-to-real 成功的关键之一。

## 4. 实验结果直觉

### 4.1 Short-Horizon (ARNOLD, Table 1, 8)

ODYSSEY vs PerAct (PerAct 用 5 个 external cameras, ODYSSEY 用单个 egocentric):

| Task | PerAct (Seen) | ODYSSEY (Seen) | PerAct (Novel) | ODYSSEY (Novel) |
|---|---|---|---|---|
| PICKUPOBJECT | 94.03 | 60.45 | 25.70 | 45.24 |
| REORIENTOBJECT | 19.48 | 51.32 | 8.23 | 52.09 |
| OPENCABINET | 31.09 | 56.30 | 16.62 | 51.09 |
| CLOSECABINET | 60.81 | 74.32 | 41.32 | 79.50 |

直觉: 
- Seen 上 PICKUPOBJECT PerAct 大幅领先 (94 vs 60), 因为 PerAct 在 ARNOLD 训练集上见过这些物体
- Novel 上 ODYSSEY 全面碾压, 因为 VLM 的开放世界泛化 >> closed-set 3D voxel 方法
- REORIENTOBJECT (旋转物体) PerAct 几乎失败 (19/8), 说明 voxel-based 方法对动态 reorientation 很弱
- ODYSSEY 在 CABINET 任务上跨场景稳定 (51/79), VLM 的语义理解起作用

Table 8 更细: Phase II (用 GT grasp pose) 上 ODYSSEY 在 PICKUPOBJECT/REORIENTOBJECT/CLOSECABINET 上都达到 80%+, 说明低层 control 没问题, 失败主要来自 high-level grasp pose prediction。

### 4.2 Long-Horizon (Table 2)

8 个 task, overall success rate 41-70%:
- INDOORCOLLECT 66.7% (navigate 97.4, pick 72.7, place 96.8)
- ROOMNAVIGATION 69.8% (navigate 86.6, push 94.1) — 简单任务, 只有 nav + push
- CARTDELIVERY 41.0% — 最难, 因为 drag + pick + place + nav
- SHOPPING 47.5%, OUTDOORCOLLECT 63.3%, OUTDOORDELIVERY 46.4%

直觉: 长程任务 overall success 是各 atomic action success 的乘积。即使每个 atomic action 80%+, 5 步串联就只剩 32%。CARTDELIVERY 41% 已经很可观。

### 4.3 Low-Level (Table 3)

vs RoboDuet:

| Metric (stand still) | RoboDuet | ODYSSEY |
|---|---|---|
| $e_x$ | 0.32 | 0.08 |
| $e_y$ | 0.34 | 2.69 |
| $e_\omega$ | 0.32 | 0.26 |
| $D_{pos}$ | 11.08 | 11.48 |
| $D_{ori}$ | 47.14 | 46.93 |

| Metric (move) | RoboDuet | ODYSSEY |
|---|---|---|
| $e_x$ | 9.70 | 0.36 |
| $e_y$ | 15.42 | 2.31 |
| $e_\omega$ | 60.59 | 0.79 |
| $D_{pos}$ | 10.75 | 10.57 |
| $D_{ori}$ | 47.53 | 47.15 |

直觉:
- $e_x, e_y, e_\omega$ 单位应该是 cm 和 deg (没明说, 但 0.08 这种量级应该是 cm/deg)
- 移动时 RoboDuet base tracking 严重退化 (9.7, 15.4, 60.59), ODYSSEY 几乎不变 (0.36, 2.31, 0.79)
- 这说明 single policy 学到了 locomotion-manipulation coupling, arm 动作不会扰动 base
- $D_{pos}, D_{ori}$ 两者持平 (~11cm, 47°), 这其实误差挺大, 但 paper 说 EE tracking 主要靠 high-level VLM 提供 target, 低层误差被 absorbed

47° 的 orientation error 听起来很大, 但实际是因为 Euler angle 表示对某些 rotation 敏感 (gimbal lock 附近), 而且抓取容差 gripper 自身 ±20-30° 都能成功。

## 5. Sim-to-Real

硬件: Unitree Go2 (12 DoF legs) + Arx5 (6 DoF arm, 3.35kg), 总负载远超 Go2 8kg 标称 payload 的边缘, 但能 work。

传感器:
- Unitree L1 LiDAR (built-in, 倾斜向下)
- Livox MID-360 (high-level localization)
- RealSense D435i (head, RGB)
- RealSense D405 (gripper, RGB-D)

控制频率: 50Hz policy, 200Hz PD。

D435i 在 head 上、D405 在 gripper 上, 这个配置很关键——head camera 给 wide FoV scene understanding, gripper camera 给 fine-grained manipulation 的 close-up depth。VLM (Qwen2.5-VL-72B) 用 gripper camera 的 RGB-D 做 contact point grounding, 这样 depth 精度高 (D405 close-range depth error < 1cm)。

参考 Unitree Go2: https://www.unitree.com/go2
参考 Intel RealSense D405: https://www.intelrealsense.com/depth-camera-d405/

## 6. 我的整体评价与潜在改进

**优点**:
1. VLM + 几何约束 + RL 三层耦合设计干净
2. Terrain-invariant sampling 是个简单但 powerful 的 insight
3. Single unified policy 避开 dual policy 的 coordination 问题
4. Benchmark 覆盖 indoor + outdoor, 是真实的 progress

**潜在问题**:
1. $D_{ori} \approx 47°$ 的 orientation error 在 long-horizon 上会累积, CARTDELIVERY 41% 部分原因可能在这里
2. VLM 推理 latency 没讨论, Qwen2.5-VL-72B 一帧推理可能 1-2s, 对 reactive 任务太慢
3. failure analysis 提到 "slim handles" "partial occlusion" 是 VLM 的弱项, 这些恰恰是日常任务的高频场景
4. paper 没有给 gait reward $r_{gait}$ 和 frequency reward $r_{fre}$ 的 ablation, 难以判断这些 shaping 真正起的作用
5. domain randomization 中 friction $[0.4, 2.0]$ 没覆盖冰面 (0.1) 或草地 (3+), 真实 outdoor 可能 OOD
6. table 8 phase II 中 CLOSECABINET ODYSSEY 94.93% (any state), 比 phase I 83.33% 高 11 个点, 说明 grasp pose prediction 是 bottleneck, 用更好的 grounding (e.g. VLM + 3D-aware) 还能提升

**未来方向** (paper 5.2 提到的 active perception): 这是个对的直觉。当前 VLM 在 "static frame" 上做 grounding, 但 robot 完全可以在 navigate 阶段就主动调整视角去 peek object 的背面/侧面, 然后再靠近 manipulate。这种"先 look 再 act"的 exploration 是 mobile manipulation 的天然优势。

参考 Active Perception for Manipulation: https://arxiv.org/abs/2306.16661

## 7. 一句话总结

ODYSSEY 把 LLM 的 task decomposition、VLM 的 contact grounding、几何约束的 pose projection、terrain-invariant 的 RL policy 这四件事拼成一个能 sim-to-real 的 mobile manipulation 系统, 关键贡献是 terrain-invariant sampling 和 geometry-constrained VLM grounding, 但 long-horizon overall success 41-70% 表明距离"通用 home robot"还很远, 主要瓶颈在 VLM 的 fine-grained spatial reasoning。

参考 paper 项目页: https://kaijwang.github.io/odyssey.github.io/

如果你想 build 进一步 intuition, 我建议重点看 Appendix B.1 的 reward weight 表 (Table 5), 那些 tiny weight 的 regularization term 在 sim-to-real 上往往比 task reward 更重要——它们决定了 policy 在 OOD 物理参数下的 robustness, 这是 RL 部署到真实 robot 上最 non-obvious 的 craft knowledge。
