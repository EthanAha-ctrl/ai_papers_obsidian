---
source_pdf: GeoManip Geometric Constraints as General Interfaces for Robot Manipulation.pdf
paper_sha256: bf786ee1b302f8e87c265388f5291d8f2cfc2395dea521303be446f8c7556383
processed_at: '2026-08-04T21:14:05-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GeoManip

## 一、这篇 paper 到底在干嘛

先想象一个场景：你对 robot 说 "cut the carrot with the knife"。

一个 naive 的做法是训一个 VLA model，把 image + language 直接映射成 gripper 的 action sequence。这个做法的问题你自己也吐槽过 — 需要海量数据，black box，OOD 就废。

GeoManip 的思路完全不一样。它说：cut carrot 这个 task 其实有 **非常明确的 geometric 规律**，只要 knife blade 垂直于 carrot 长轴、knife 平行于桌面、knife 在 carrot 上方 5cm，然后往下切就行。这些 geometric 关系可以用语言描述，也可以用数学 equation 写，那为啥不把这个 **中间的 geometric 关系** 显式写出来，作为 language 和 action 之间的 bridge？

所以 GeoManip 的核心就是一句话：**让 VLM 把 "cut carrot" 翻译成一组 geometric constraint equations，然后让 numeric solver 解出 gripper 轨迹**。

VLM 负责 reasoning，solver 负责 precision，各司其职。

类比一下：VLA model 是 "you give me image+text, I give you action tokens, trust me bro"，GeoManip 是 "you give me image+text, I show you my work — 这里是 3 条 geometric constraint，这里是 cost function，最后 solver 给你 trajectory，你可以随时打断我说 'height too large'，我就改 constraint 重算"。

参考：项目主页 http://geoconstraintmanip.github.io

---

## 二、Pipeline 的 4 个 step

整条 pipeline 就像做菜的分步：

**Step 1: Task Decomposition**

VLM 把 "filled macaroni" 这种复杂 task 拆成 6 个 sub-task：grasp spoon → scoop macaroni → move to pan → pour into pan → ... 还能加 loop（"repeat tilting until pan is filled"）。

这部分纯粹靠 GPT-4o 的 planning 能力，没什么花活。

**Step 2: Geometry Parser（最 clever 的部分）**

这是论文的 novelty。问题是：要写 "knife blade 垂直 carrot axis" 这种 constraint，你得先在 image 里找到 "knife blade" 和 "carrot" 的 point cloud。

但 open-vocab segmentation model（LISA、OV-Seg）在这种 fine-grained part 上 fail 严重。你让它分割 "knife blade"，它可能给你整个 knife 或者只给你一块 metal 区域，不准确。

GeoManip 的 trick 是 **select-process 两步走**：

- 先用 SAM (Segment Anything, https://arxiv.org/abs/2304.02643) 把 image 切成 N 个 class-agnostic mask。SAM 不懂语义，但它能给你所有可能的 region 边界。
- 然后把 image 和每个 mask 配对 $\{(I, \bar{M}_i)\}_{i=1}^N$ 喂给 VLM，问 "which mask best matches 'the blade of knife'?" VLM 选出 $M^*$。
- 但 $M^*$ 可能还不够精确（比如包含了部分 handle），于是再让 VLM **生成一段 Python 代码** $g(\cdot)$ 来 refine：比如用 Canny edge 提取边界、取 leftmost column 之类。最终 $M' = g(M^*)$。

这就像：SAM 给你一堆 raw candidates，VLM 既负责 semantic 判断（选哪个），又负责 pixel-level refinement（怎么改）。

这种 SAM + VLM 分工的设计真的 elegant — 用 SAM 的 strong boundary prior + VLM 的 strong semantic prior，互相补足。

**Step 3: Constraint Generator**

拿到 geometric components 的 point cloud 之后，VLM 根据 task description 生成 constraints。每个 constraint 是个 tuple：

$$\text{Constraint} = (\{\text{GeoComp}_1, \text{GeoComp}_2, \ldots\}, \text{ConsDesc}, \text{type})$$

- $\text{GeoComp}_i$：涉及哪些 geometric component（"the blade of knife"、"the axis of carrot"）
- $\text{ConsDesc}$：语言描述的关系（"perpendicular to"、"directly above by 10cm"）
- $\text{type}$：sub-goal（终点满足）或 path（全程满足）

比如 cut carrot 的 stage 2 有 3 条 constraint：
1. `(knife blade heading, carrot axis, perpendicular)` — sub-goal
2. `(knife blade heading, table normal, parallel)` — sub-goal  
3. `(knife center, carrot center, directly above by 5cm)` — sub-goal

prompt 里有几个 in-context examples（pouring tea、open door、cut cucumber...），VLM 学着 pattern 生成新的 constraint。

**Step 4: Cost Function + Solver**

这一步把 symbolic constraint 翻译成可优化的 cost function。还是 VLM 来做 — 它写一段 Python 代码 $f: \mathcal{P} \to \mathbb{R}^+$，输入是相关 component 的 point cloud，输出是非负 violation score（0 表示完美满足）。

举个具体例子，"knife blade perpendicular to carrot axis" 的 cost function 长这样：

```python
def stage_2_subgoal_constraint1():
    pc1 = get_point_cloud("the body of the cucumber", -1)
    pc2 = get_point_cloud("the blade of the kitchen knife", -1)
    
    # cucumber 的 axis = PCA 最大特征值对应的特征向量
    cov1 = np.cov(pc1.T)
    eigvals1, eigvecs1 = np.linalg.eig(cov1)
    cucumber_axis = eigvecs1[:, np.argmax(eigvals1)]
    
    # knife surface normal = PCA 最小特征值对应的特征向量
    cov2 = np.cov(pc2.T)
    eigvals2, eigvecs2 = np.linalg.eig(cov2)
    knife_normal = eigvecs2[:, np.argmin(eigvals2)]
    
    dot = np.dot(cucumber_axis, knife_normal)
    cost = (1 - abs(dot)) * 5  # perpendicular ⟺ normal 平行
    return cost
```

这里几个技术细节值得拆解：

- **为什么 cucumber axis 是最大特征值的方向？** 因为长条形物体沿长轴 variance 最大，PCA 第一主成分就是长轴方向。这是经典 geometry。
- **为什么 knife normal 是最小特征值方向？** 因为 surface 是 2D 平面，垂直 surface 方向 variance 最小（点云薄薄一层），最小特征值方向就是 normal。
- **为什么 `if axis[argmax_abs] < 0: axis = -axis`？** 强制最大分量非负，保证向量方向 consistency，避免 dot product 因符号翻转产生 ambiguous cost。
- **`get_point_cloud(..., -1)` 和 `-2` 的 timestamp 索引**：`-1` 是当前帧，`-2` 是上一帧。这让 rotation constraint 可以算 "相对于 grasping 时刻" 的相对变换。

所有 cost function 收集起来形成 $\mathcal{F}^s$（sub-goal cost set）和 $\mathcal{F}^p$（path cost set）。

---

## 三、核心优化公式（Eq. 1）

论文的核心公式长这样：

$$
\min_{\mathbf{R}, \mathbf{t}} \frac{1}{K^s} \sum_{f \in \mathcal{F}^s} f\Big(\mathcal{P}^s \cup (\mathbf{R}\mathbf{R}_0^{-1} \otimes (\mathcal{P}^m \oplus -\mathbf{t}_0) \oplus \mathbf{t})\Big) + \alpha\|\mathbf{t} - \mathbf{t}_0\|_2 + \beta\|\text{euler}(\mathbf{R}\mathbf{R}_0^{-1})\|_1
$$

变量含义拆解：

- $\mathbf{R} \in SO(3), \mathbf{t} \in \mathbb{R}^3$：**优化目标**，gripper 的目标 rotation matrix 和 translation vector
- $\mathbf{R}_0, \mathbf{t}_0$：gripper **上一时刻** 的 pose（previous state，作为正则化 anchor）
- $\mathcal{F}^s$：当前 stage 所有 sub-goal cost function 的集合，$K^s = |\mathcal{F}^s|$
- $f$：单个 cost function，输入 point cloud 集合，输出非负 float
- $\mathcal{P}^s$：**stationary** geometric components 的 point cloud（不会被 gripper 移动的物体，比如 carrot、table）
- $\mathcal{P}^m$：**moving** geometric components 的 point cloud（被 gripper 抓住的物体，比如 knife）
- $\oplus$：set 上对每个 point 做 vector addition（element-wise broadcast）
- $\otimes$：set 上对每个 point 做 matrix-vector product
- $\alpha = 0.02, \beta = 0.075$：translation 和 rotation 正则化权重
- $\text{euler}(\cdot)$：从 rotation matrix 提取 3 个轴的 Euler angle
- $\|\cdot\|_2, \|\cdot\|_1$：L2 和 L1 范数

**这个公式在干嘛？**

moving point cloud $\mathcal{P}^m$ 经历的 rigid body transformation 是：

$$
\mathcal{P}^m_{\text{new}} = \mathbf{R}\mathbf{R}_0^{-1} \otimes (\mathcal{P}^m - \mathbf{t}_0) + \mathbf{t}
$$

拆成 3 步理解：
1. $\mathcal{P}^m - \mathbf{t}_0$：把 moving point cloud 平移回原点（撤销之前 translation）
2. $\mathbf{R}_0^{-1} \otimes (\cdot)$：撤销之前 rotation，回到 canonical frame
3. $\mathbf{R} \otimes (\cdot) + \mathbf{t}$：应用新的 rotation 和 translation

本质上就是把 $\mathcal{P}^m$ 从 previous pose 变换到 candidate new pose，然后和 stationary components 一起喂给 cost function 检查是否满足 geometric constraints。

**两个正则项的意义**：
- $\alpha\|\mathbf{t} - \mathbf{t}_0\|_2$：translation 不要跳太远（保证轨迹平滑、避免 local minima 暴跳）
- $\beta\|\text{euler}(\mathbf{R}\mathbf{R}_0^{-1})\|_1$：rotation 角度变化惩罚

注意 L1 形式的 rotation 正则很关键 — 它鼓励 rotation 集中在一个轴上而不是 spread 在 3 个轴上。大部分 manipulation task 都是 single-axis rotation（开门绕 hinge、倒水绕 handle、切东西垂直下切），L1 的 sparsity 正好对应这种 inductive bias。

求出 $(\mathbf{R}, \mathbf{t})$ 后，在 $(\mathbf{R}_0, \mathbf{t}_0)$ 和 $(\mathbf{R}, \mathbf{t})$ 之间插值出几个 control points，每个 control point 再用 path cost $\mathcal{F}^p$ 做局部优化。这就是个 **bi-level optimization**：外层插值，内层局部 refine。

整个求解用 SLSQP（Sequential Least Squares Programming, Kraft 1988），一种 quasi-Newton 风格的 constrained optimizer。

---

## 四、5 个 generalist features（最 cool 的部分）

这部分是论文最有意思的「上探」。因为 constraint 是 in-context generated 的 first-class object，你可以直接 manipulate 它：

**1. On-the-fly policy adaptation**

用户在 dialogue 里说 "the height is too large"，VLM 直接把 constraint tuple 里的 "20 cm" 改成 "10 cm"，re-run solver。完全不需要 retraining。

这就像给 robot 一个 **verbal knob**，你可以随时调参数。

**2. Learn from failure cases**

上传失败 video，VLM 被问 "Why did the robot fail? How to adjust constraints?"。VLM 识别出 "unsafe grasping position"，追加一条 sub-goal constraint（"grasp position should be close to center of banana"）。这是 **verbal feedback loop** 而不是 gradient-based update。

**3. Long-horizon planning**

任务 "Add the pan with macaroni and water. Add salt with spoon and stir." 被分解成 15+ 个 stage，包括 flow control 里的 "repeat until pan is filled"。每个 stage 独立生成 constraints 和 trajectory。

**4. Learn from human demonstration**

看 human 开盒子的视频，VLM 推理出 "box 应该是 lid-up rotate around edge" 而不是默认的 "drawer pull-out"，修改 sub-goal constraint。这本质是 **video-to-constraint** 的 in-context translation。

**5. Efficient data collection for IL**

两种用法：
- **VLA training data**：对同一 task 不同初始 pose，只用 SAM 分割一次，然后用 CoTracker (https://arxiv.org/abs/2307.07635) 做 point tracking 同步 geometric components，快速批量生成 ground-truth trajectory。论文实验显示用 GeoManip 收集的 30 条 demo 训练 OpenVLA，效果和人工 demo 训练的相当（Table 4: 3/5 vs 3/5, 4/5 vs 5/5）。
- **Reward model training**：cost function 直接当作 reward signal。Train 一个 ViT encoder + MLP head $R(o)$，输入 RGB observation 输出 reward。Inference 时枚举 6 个 candidate actions $\{L, R, F, B, U, D\}$：
$$
a = \arg\max_a R(\text{step}(o, a))
$$
这是 greedy one-step planning，类似 Q-learning 的 greedy policy，但 reward 来自 VLM-generated cost。

---

## 五、实验结果

### MetaWorld (Table 1)

| Method | Avg. | Training? |
|---|---|---|
| BC-Scratch | 15.3% | ✓ |
| Diffusion Policy | 14.8% | ✓ |
| AVDC | 38.0% | ✓ |
| SceneFlow | 59.8% | ✓ |
| **GeoManip** | **71.1%** | ✗ |

GeoManip **训练免费** 还能 +11% over 之前 SOTA。这里关键的 insight 是：MetaWorld 大部分 task 本质是 geometric alignment（按按钮、放物体到指定位置），用 explicit geometric constraints 比用 diffusion policy 学 action distribution 更高效。

但仔细看 per-task：
- `basketball` 73.3% vs SceneFlow 96.0% — GeoManip 反而更低。推测是因为 basketball 需要把球抛进筐，trajectory 是 ballistic motion，cost function 没捕捉到 release timing
- `assembly` 40% vs 46.7% — 插针进孔对 point cloud 精度要求高，VLM-generated cost 在精细 alignment 上不如 trained policy

### OmniGibson (Table 2)

| | open-fridge | typing | put-pen | cut-carrot | overall |
|---|---|---|---|---|---|
| ReKep | 0% | 0% | 60% | 20% | 20% |
| GeoManip | 80% | 40% | 80% | 40% | 60% |

GeoManip 比 ReKep 高 40%。ReKep 用 keypoint constraints，keypoint 是离散点，信息稀疏；GeoManip 用 geometric components（point cloud 上的 axis/normal/plane），信息丰富 + 可由 PCA robustly 估计。

OOD 物体上 ReKep 容易选错 keypoint（比如 fridge handle vs hinge，选错一个就 fail），而 GeoManip 用整个 point cloud 上的 geometric primitive 对 keypoint 错位更鲁棒。

### Real-world (Table 3)

| | pick-place | pour | open | stir | overall |
|---|---|---|---|---|---|
| OpenVLA (30 demos) | 30% | 20% | 10% | 0% | 15% |
| GeoManip (zero-shot) | 90% | 70% | 60% | 40% | 65% |

OpenVLA 即使有 30 demo 训练，real-world 也只有 15% — 30 demos 对 7B 参数 VLA 远远不够，且 OOD object 泛化差。GeoManip 完全 training-free 还能 65%，体现了 **structured representation 在 data efficiency 上的碾压优势**。

`stir` 任务 40% 是最低的 — 推测是因为 stirring 涉及 periodic motion（circular orbit），需要 flow control 重复 12 次绕 30 度，error 累积。

---

## 六、几个我个人觉得值得 discuss 的点

**1. 没有 collision avoidance**

Eq. 1 里没有 collision cost term。这意味着 GeoManip 在 cluttered scene 里可能 plan 出穿过其他物体的轨迹。一个 extension 是让 VLM 也生成一个 collision cost：
```python
def collision_cost(P_moved, P_static_environment):
    return chamfer_distance(P_moved, P_static_environment) if penetration > 0 else 0
```

CuRobo (https://github.com/NVlabs/curobo) 有 parallel collision checking 可以借鉴。

**2. Local minima 问题**

SLSQP 是 local optimizer。Multi-stage task 里每个 stage 独立优化，可能在 stage transition 处出现 deadlock。可以引入 sampling-based warm start（OMPL/RRT）或者 diffusion-based proposal。

**3. 没有 dynamics / deformable objects**

Cost function 假设 rigid body transformation。倒水这种 task 实际涉及 fluid dynamics，GeoManip 把它简化成 "tilt teapot by 30 degrees" + flow control check "is cup filled"。这是 functional approximation 而不是 physics simulation。

**4. VLM 推理 cost**

每个 stage 至少需要：1 次 SAM + 1 次 VLM mask selection + 1 次 VLM code generation + 1 次 VLM constraint generation + 1 次 VLM cost function generation。一个 6-stage task 就要 30+ 次 VLM call，每次 GPT-4o 几秒，total 几十秒到分钟级。这比 diffusion policy (100ms 级 inference) 慢得多。

不过对 data collection for IL 反而是优势 — 一次性生成大量 ground-truth trajectory 给小模型学习。

**5. Point cloud 的 categorical 模糊性**

"knife blade" 在 point cloud 里和 "knife handle" 边界可能模糊。SAM 给出的 mask 不一定 align 到 functional boundary。这导致 cost function 在 PCA 算 axis 时可能混入 handle 点，污染 axis 估计。一个 fix 是用 multiple mask candidates 的 ensemble，取 cost 最低的。

**6. 联想到 Software 2.0**

GeoManip 在某种意义上是 **Software 1.5** — 用 VLM 生成 symbolic code（Software 1.0 的结构），但 code 内容由 learned model 决定（Software 2.0 的 flexibility）。比 VLA 端到端（Software 2.0 pure）多了 structure 和 interpretability，比手写规则（Software 1.0 pure）多了 generalization。这种 hybrid 在 data-sparse domain 应该会成为主流。

参考你这个观点：https://karpathy.medium.com/software-2-0-a64182b37c35

**7. 与 differentiable physics 的连接**

Cost function 是 non-differentiable（用了 abs、argmax、if 之类），所以只能用 SLSQP 这种 gradient-free / numerical gradient 方法。如果让 cost function differentiable，可以用 iLQR / DDP / gradient descent 大幅加速。这暗示了一个方向：**VLM 生成 differentiable cost function**（用 JAX/Torch 写），然后 chain rule 自动求导。

参考 Toussaint et al. 2018 "Differentiable Physics and Stable Modes" https://arxiv.org/abs/1805.08367

---

## 七、最值得 take-away 的几个 idea

1. **Symbolic geometric constraint 作为 language→action 的中间层** — 比 keypoint 丰富，比 pixel-level value map 结构化，比 end-to-end VLA 可解释。这层 abstraction 是整篇论文的灵魂。

2. **Select-process scheme** — SAM + VLM 的分工很 elegant，SAM 给 raw candidates，VLM 做 semantic selection + pixel-level refinement。这种 pattern 可以迁移到其他 fine-grained part segmentation 任务。

3. **In-context constraint editing** — 5 个 generalist features 的本质都是 "把 constraint 当作 first-class object 来 manipulate"，这和 program synthesis 的思路一致。你不需要 retrain model，只需要 edit constraint。

4. **Cost function = Reward function** — 把 VLM-generated cost function 当 reward signal 训 reward model，是 IL→RL 桥梁的优雅设计。这暗示了一种新的 IL pipeline：VLM 生成 cost → cost 当 reward → reward model 蒸馏成 fast policy network。

5. **Training-free SOTA** — 在 real-world 上 zero-shot 65% vs OpenVLA 30-shot 15%，再次印证 **structure beats data** 在小数据 regime 的威力。这点你应该特别有共鸣 — 在 data scarce 的 real-world robotics，inductive bias 比 model scale 重要得多。

整体上这篇工作非常有 2024-2025 neuro-symbolic robotics 的代表性 — VLM 的 reasoning 接上 classical optimization 的 precision，用 in-context 方式实现 zero-shot manipulation。它没有 end-to-end 学，但反而因此获得了 OOD generalization、可解释性、和 on-the-fly adaptation 这三个 VLA model 当前最缺的属性。

参考链接汇总：
- GeoManip 主页: http://geoconstraintmanip.github.io
- ReKep: https://arxiv.org/abs/2409.01652
- Code as Policies: https://arxiv.org/abs/2209.07726
- VoxPoser: https://arxiv.org/abs/2307.05973
- OpenVLA: https://arxiv.org/abs/2406.09246
- CuRobo: https://github.com/NVlabs/curobo
- SAM: https://arxiv.org/abs/2304.02643
- CoTracker: https://arxiv.org/abs/2307.07635
- Logic-Geometric Programming: https://arxiv.org/abs/1707.05818
- Differentiable Physics: https://arxiv.org/abs/1805.08367
- Riemannian Motion Policies: https://arxiv.org/abs/1801.02854
- Software 2.0: https://karpathy.medium.com/software-2-0-a64182b37c35

---

# GeoManip: 把 Geometric Constraints 当作 Language→Action 的中间表示

## 一、论文的核心 intuition

这篇论文的精神气质让我想到你在 Tesla/OpenAI 一直强调的一个观点：**纯 end-to-end 模型缺乏 inductive bias，会浪费数据**。GeoManip 的核心 thesis 是 — robot manipulation 的低层 action 不应该直接由 language 端到端映射（VLA 的做法），而是应该经过一层 **symbolic 的 geometric constraints** 作为 intermediate representation。这层 representation 既能被 VLM 用语言推理生成，又能被 numeric solver 精确执行，构成一种 **neuro-symbolic pipeline**。

具体类比："cut the carrot with knife" 这个 language command，VLM 把它展开成一组符号约束：
1. `heading of knife blade ∥ table surface`
2. `heading of knife blade ⊥ carrot axis`  
3. `center of knife ≈ 5 cm above center of carrot`

这组约束既是 interpretable 的语言，又是数学上可解的 equation。整篇论文的工程就是把这条 pipeline 做端到端可执行。

参考：
- 项目主页 http://geoconstraintmanip.github.io  
- 最相近工作 ReKep: https://arxiv.org/abs/2409.01652  
- Code as Policies: https://arxiv.org/abs/2209.07726  
- VoxPoser: https://arxiv.org/abs/2307.05973

---

## 二、Pipeline 总览（4 个 stage 的 cascade）

整条 pipeline 是 4 个 module 的串行 cascade，每个 module 都由 VLM (GPT-4o) 驱动：

```
Language instruction + RGB image
        │
        ▼
[1] Task Decomposition + Process Control  (VLM)
        │  → list of sub-tasks + flow control (loop/branch)
        ▼
[2] Geometry Parser (select-process scheme)  (SAM + VLM)
        │  → per-stage geometric components (point clouds)
        ▼
[3] Constraint Generator  (VLM)
        │  → symbolic constraints: (GeoComps, ConsDesc, type=sub-goal|path)
        ▼
[4] Cost Function + Trajectory Solver  (VLM generates code → SLSQP)
        │  → per-stage trajectory (R_t, t_t)
        ▼
Low-level gripper action
```

关键 design pattern：**每个 stage 的 constraints 是 in-context generated**，可以随时插入、修改、扩展。这是后面 5 个 generalist features 的根基。

---

## 三、Geometry Parser：select-process scheme 详解

### 3.1 为什么不能直接用 open-vocab segmentation

论文实验显示 LISA、OV-Seg 这类 open-vocabulary part segmentation 方法在 fine-grained geometric component 上 fail 严重（Fig. 3）。原因有两个：
- 训练数据里 "knife blade"、"cup opening"、"hinge axis" 这种 **part-level + functional** 标签太少
- 这些 part 边界往往和 texture/color 边界不对齐（hinge 在 visual 上看不出）

### 3.2 select-process 的两步设计

**Step 1: Select**
$$\{M\}_1^N = SAM(I)$$
SAM 给出 N 个 class-agnostic mask。然后把每个 mask 和 image 拼成 pair $\{(I, \bar{M})\}_1^N$ 喂给 VLM，问 "which mask best matches 'the blade of the knife'?" VLM 选出 $M^*$。

这里利用了 **SAM 擅长边界但不擅长语义** + **VLM 擅长语义但不擅长像素边界** 的互补性。SAM 不需要训练就能给出所有可能的 object part 候选。

**Step 2: Process**

VLM 生成一个 Python 函数 $g: \mathbb{R}^{H\times W} \to \mathbb{R}^{H\times W}$，把 $M^*$ 处理成更精确的 $M' = g(M^*)$。

例子（来自 appendix prompt）：如果 geometric component 是 "hinge of microwave door"，VLM 会生成一段用 Canny edge + 取 leftmost/rightmost column 的代码，从 door mask 中提取出 hinge 那条窄边。这种 geometric primitive 的提取在 symbolic 代码层面非常自然，但在像素级 segmentation model 里很难学。

最后从 $M'$ 反投影到 3D 拿到 point cloud $\mathcal{P}_i \in \mathbb{R}^{N_i \times 3}$。

---

## 四、Constraint Generator：从 language 到 symbolic constraint

### 4.1 Constraint 的数据结构

每个 constraint 是一个 tuple：
$$\text{Constraint} = (\{\text{GeoComp}_1, \text{GeoComp}_2, \ldots\}, \text{ConsDesc}, \text{type})$$

其中 `type ∈ {sub-goal, path}`：
- **sub-goal constraint**：只在 stage 终点需要满足（destination 的目标姿势）
- **path constraint**：在整条 trajectory 上都必须满足（比如切胡萝卜过程中 knife 始终 perpendicular to carrot axis）

这个区分对应 optimal control 里 **terminal cost vs running cost** 的经典区分。

### 4.2 Prompt 的三件套设计

VLM 生成 constraints 时 prompt 包含：
1. **Geometry principles**：基础几何事实，如 "to be ⊥ to a plane is to be ∥ to its normal"
2. **Output rules**：tuple 的格式约束、命名约定（必须出现至少两个 "of"）
3. **In-context examples**：pouring tea / put block / open door / cut cucumber / open drawer / press button 六个例子

这种 **rule + examples** 的 prompt 结构是 in-context learning 的标准做法，配合 GPT-4o 的 reasoning 能力，能泛化到训练例子之外的任务。

### 4.3 Process Control 嵌入到 constraint 里

最妙的是把 control flow（loop / branch）也变成 constraint：
```
<"flow constraints", "the cup is filled with water"> → (goto stage 3 if not, goto stage 4 if yes)
```

在 cost function 阶段，flow constraint 被翻译成 Python 的 `while True: query_GPT(...)` 风格代码。这就把 long-horizon planning + conditional branching 全部统一到 constraint generation 里。

---

## 五、Cost Function + Trajectory Solver（核心公式）

### 5.1 VLM 生成 cost function 代码

每个 constraint 翻译成一个 Python 函数 $f: \mathcal{P} \to \mathbb{R}^+$，输入是相关 geometric components 的 point cloud 集合，输出是非负 violation score（0 = 完美满足）。

例子（cut carrot 中的一个 constraint）：
```python
def stage_2_subgoal_constraint1():
    pc1 = get_point_cloud("the body of the cucumber", -1)
    pc2 = get_point_cloud("the blade of the kitchen knife", -1)
    
    # cucumber axis = eigenvector with largest eigenvalue (PCA)
    cov1 = np.cov(pc1.T)
    eigvals1, eigvecs1 = np.linalg.eig(cov1)
    cucumber_axis = eigvecs1[:, np.argmax(eigvals1)]
    
    # knife surface normal = eigenvector with smallest eigenvalue
    cov2 = np.cov(pc2.T)
    eigvals2, eigvecs2 = np.linalg.eig(cov2)
    knife_normal = eigvecs2[:, np.argmin(eigvals2)]
    
    dot = np.dot(cucumber_axis, knife_normal)
    cost = (1 - abs(dot)) * 5  # perpendicular ⟺ parallel normals
    return cost
```

注意几个关键技术点：
- **PCA 提取几何 primitive**：long-shaped 物体的 axis 是最大特征值对应的特征向量（沿长轴 variance 最大）；surface 的 normal 是最小特征值对应的特征向量（垂直 surface 方向 variance 最小）。这是经典计算机几何方法。
- **符号对齐**：用 `if axis[np.argmax(np.abs(axis))] < 0: axis = -axis` 强制最大分量非负，保证向量方向一致性（避免 dot product 因符号翻转产生 ambiguous cost）。
- **Timestamp 索引**：`get_point_cloud(..., -1)` 是当前帧，`-2` 是上一帧。这让 rotation/orbit constraint 可以计算 "相对于 grasping 时刻" 的相对变换。

### 5.2 核心优化公式（Eq. 1）

$$
\min_{\mathbf{R}, \mathbf{t}} \frac{1}{K^s} \sum_{f \in \mathcal{F}^s} f\Big(\mathcal{P}^s \cup \big(\mathbf{R}\mathbf{R}_0^{-1} \otimes (\mathcal{P}^m \oplus -\mathbf{t}_0) \oplus \mathbf{t}\big)\Big) + \alpha\|\mathbf{t} - \mathbf{t}_0\|_2 + \beta\|\text{euler}(\mathbf{R}\mathbf{R}_0^{-1})\|_1
$$

变量逐个解释：
- $\mathbf{R} \in SO(3), \mathbf{t} \in \mathbb{R}^3$：**待优化的目标** gripper rotation matrix 和 translation vector
- $\mathbf{R}_0, \mathbf{t}_0$：gripper **上一时刻** 的 rotation 和 translation（previous state，作为正则化 anchor）
- $\mathcal{F}^s$：当前 stage 所有 sub-goal cost function 的集合，$K^s = |\mathcal{F}^s|$
- $f$：单个 cost function，输入 point cloud 集合，输出非负 float
- $\mathcal{P}^s$：**stationary** geometric components 的 point cloud 集合（不会被 gripper 移动的物体）
- $\mathcal{P}^m$：**moving** geometric components 的 point cloud 集合（被 gripper 抓住的物体）
- $\oplus$：set 上对每个 point 做 vector addition（element-wise broadcast）
- $\otimes$：set 上对每个 point 做 matrix-vector product
- $\alpha = 0.02, \beta = 0.075$：translation 和 rotation 正则化权重
- $\text{euler}(\cdot)$：从 rotation matrix 提取 3 个轴的 Euler angle
- $\|\cdot\|_2, \|\cdot\|_1$：L2 和 L1 范数

**几何意义拆解**：

moving point cloud 经历的 rigid body transformation 是：
$$
\mathcal{P}^m_{\text{new}} = \mathbf{R}\mathbf{R}_0^{-1} \otimes (\mathcal{P}^m - \mathbf{t}_0) + \mathbf{t}
$$

可以拆成三步理解：
1. $\mathcal{P}^m \oplus -\mathbf{t}_0$：把 moving point cloud 平移回原点（撤销之前的 translation）
2. $\mathbf{R}_0^{-1} \otimes (\cdot)$：撤销之前的 rotation，回到 canonical frame
3. $\mathbf{R} \otimes (\cdot) \oplus \mathbf{t}$：应用新的 rotation 和 translation

所以本质上是把 $\mathcal{P}^m$ 从 previous pose 变换到 candidate new pose，然后和 stationary components 一起喂给 cost function 检查是否满足 geometric constraints。

**正则项的意义**：
- $\alpha\|\mathbf{t} - \mathbf{t}_0\|_2$：translation 不要跳太远（保证轨迹平滑、避免 local minima 暴跳）
- $\beta\|\text{euler}(\mathbf{R}\mathbf{R}_0^{-1})\|_1$：rotation 角度变化惩罚（L1 形式有利于 sparse rotation，倾向于只在一个轴上转）

L1 形式的 rotation 正则很关键 — 它鼓励 rotation 集中在一个轴上而不是 spread 在 3 个轴上，对应大部分 manipulation task 都是 single-axis rotation（开门绕 hinge、倒水绕 handle、切东西垂直下切）。

### 5.3 Trajectory 插值

求出目标 $(\mathbf{R}, \mathbf{t})$ 后，并不是直接命令 gripper 跳过去，而是在 $(\mathbf{R}_0, \mathbf{t}_0)$ 和 $(\mathbf{R}, \mathbf{t})$ 之间插值出几个 **control points**，每个 control point 再用 path cost functions $\mathcal{F}^p$ 做局部优化，确保 path constraint 在中间状态也被满足。

这就是一个 **bi-level optimization**：
- 外层：插值生成 coarse control points
- 内层：每个 control point 解一次 Eq. 1（用 path cost 代替 sub-goal cost）

整个求解用 SLSQP（Sequential Least Squares Programming, Kraft 1988），一种 quasi-Newton 风格的 constrained optimizer。

---

## 六、五个 Generalist Features

这部分是论文最有意思的「上探」— 把 in-context constraint generation 的能力扩展到 5 种 HRI 场景：

### 6.1 On-the-fly policy adaptation
用户在 dialogue 里说 "the height is too large"，VLM 直接修改 constraint tuple 的数值（"20 cm" → "10 cm"），re-run cost function 生成和 optimization。**完全不需要 retraining**。

### 6.2 Learn from failure cases
上传失败 video，VLM 被问 "Why did the robot fail? How to adjust constraints?"。VLM 会识别 "unsafe grasping position" 并追加一条 sub-goal constraint（"grasp position should be close to center of banana"）。这相当于 **verbal feedback loop** 而不是 gradient-based update。

### 6.3 Long-horizon planning
任务 "Add the pan with macaroni and water. Add salt with spoon and stir." 被分解成 15+ 个 stage，包括 flow control 里的 "repeat until pan is filled"。每个 stage 独立生成 constraints 和 trajectory。

### 6.4 Learn from human demonstration
看 human 开盒子的视频，VLM 推理出 "box 应该是 lid-up rotate around edge" 而不是默认的 "drawer pull-out"，修改 sub-goal constraint。这本质上是 **video-to-constraint** 的 in-context translation。

### 6.5 Efficient data collection for IL
两种用法：
1. **VLA training data**：对同一 task 的不同初始 pose，只用 SAM 分割一次，然后用 CoTracker (Karaev 2023, https://arxiv.org/abs/2307.07635) 做 point tracking 同步 geometric components，快速批量生成 ground-truth trajectory。论文实验显示用 GeoManip 收集的 30 条 demo 训练 OpenVLA，效果和人工 demo 训练的相当（Table 4: 3/5 vs 3/5, 4/5 vs 5/5）。
2. **Reward model training**：cost function 直接当作 reward signal。Train 一个 ViT encoder + MLP head $R(o)$，输入 RGB observation 输出 reward。Inference 时枚举 6 个 candidate actions $\{\text{Left, Right, Front, Back, Up, Down}\}$：
$$
a = \arg\max_a R(\text{step}(o, a))
$$
这是一种 greedy one-step planning，类似 Q-learning 的 greedy policy，但 reward 来自 VLM-generated cost。

---

## 七、实验结果深度解读

### 7.1 MetaWorld (Table 1)

| Method | Avg. | Training? |
|---|---|---|
| BC-Scratch | 15.3% | ✓ |
| BC-R3M | 9.8% | ✓ |
| UniPi | 1.8% | ✓ |
| Diffusion Policy | 14.8% | ✓ |
| AVDC | 38.0% | ✓ |
| SceneFlow | 59.8% | ✓ |
| **GeoManip** | **71.1%** | ✗ |

GeoManip **训练免费** 还能 +11% over 之前的 SOTA SceneFlow。这里关键的 insight 是：MetaWorld 任务大部分本质是 geometric alignment（按按钮、放物体到指定位置），用 explicit geometric constraints 比用 diffusion policy 学 action distribution 更高效。

但仔细看 per-task：
- `basketball` 73.3% vs SceneFlow 96.0% — GeoManip 反而更低。推测是因为 basketball 任务需要把球抛进筐，trajectory 不是 simple geometric alignment，而是 ballistic motion，cost function 没捕捉到 release timing
- `assembly` 40% vs 46.7% — 也低。这种 task 需要插针进孔，对 point cloud 精度要求高，VLM-generated cost 在精细 alignment 上不如 trained policy

### 7.2 OmniGibson (Table 2)

| | open-fridge | typing | put-pen | cut-carrot | overall |
|---|---|---|---|---|---|
| ReKep | 0% | 0% | 60% | 20% | 20% |
| GeoManip | 80% | 40% | 80% | 40% | 60% |

GeoManip 比 ReKep 高 40%。论文归因于 "geometric components 比 keypoints 信息更丰富"。我的理解是 ReKep 的 keypoint 选择高度依赖 VLM 推理 keypoints 位置，OOD 物体上 keypoint 容易选错（比如 fridge handle vs hinge，选错一个就 fail）。GeoManip 用整个 point cloud 上的 geometric primitive（PCA axis、surface normal），对 keypoint 错位更鲁棒。

### 7.3 Real-world (Table 3)

| | pick-place | pour | open | stir | overall |
|---|---|---|---|---|---|
| OpenVLA (30 demos) | 30% | 20% | 10% | 0% | 15% |
| GeoManip (zero-shot) | 90% | 70% | 60% | 40% | 65% |

OpenVLA 即使有 30 demo 训练，real-world 也 only 15% — 因为 30 demos 对 7B 参数 VLA 远远不够，且 OOD object 泛化差。GeoManip 完全 training-free 还能 65%，体现了 **structured representation 在 data efficiency 上的碾压优势**。

`stir` 任务 40% 是最低的 — 推测是因为 stirring 涉及 periodic motion（circular orbit），需要 flow control 重复 12 次绕 30 度，error 累积。

---

## 八、与相关工作的关系图

GeoManip 在 robot manipulation with large models 的三个 branch 里属于第三个：

| Branch | 代表工作 | 缺点 |
|---|---|---|
| End-to-end VLA | RT-1/2, OpenVLA, RDT-1B | 需要海量数据，black box |
| VLM as high-level planner | SayCan, Manipulate-Anything, ECoT | 缺乏低层 precision |
| **VLM generates code/constraints** | Code as Policies, VoxPoser, ReKep, **GeoManip** | 依赖 VLM 推理质量 |

在第三个 branch 内部：
- **Code as Policies** (Liang 2023)：VLM 直接生成 Python 控制代码，但没有 geometric constraints 的 abstraction
- **VoxPoser** (Huang 2023b)：VLM 生成 3D value maps，但没有 explicit constraint structure
- **ReKep** (Huang 2024b)：用 keypoint constraints，keypoint 是离散点，信息稀疏
- **GeoManip**：用 geometric components（point cloud 上的 axis/normal/plane/center），信息丰富 + 可由 PCA robustly 估计

更广义的 lineage 可以追到：
- **Logic-Geometric Programming** (Toussaint 2015, https://arxiv.org/abs/1707.05818)：multi-stage spatial optimization with logic constraints
- **CHOMP / TrajOpt / CuRobo** (Sundaralingam 2023, https://github.com/NVlabs/curobo)：optimization-based motion planning with collision
- **Riemannian Motion Policies** (Ratliff 2018, https://arxiv.org/abs/1801.02854)：policy as sum of costs on manifolds

GeoManip 本质是把上面这些 classical optimization-based 方法的 **cost function specification** 用 VLM 自动生成了，并且把 cost 提升到 geometric constraint 这个更 abstract 的 level。

---

## 九、Limitations 和我的延伸思考

论文自述 limitation：
1. Geometric component 在 camera 里看不清就 fail
2. Point cloud 质量差就 fail  
3. VLM 推理 constraints 不完整就 fail
4. Action 不能用语言描述就 fail

我补充几点更深的 concern：

**1. Collision avoidance 缺失**

Eq. 1 里没有 collision cost term。这意味着 GeoManip 在 cluttered scene 里可能 plan 出穿过其他物体的轨迹。ReKep 也类似。CuRobo 之类的方法有 parallel collision checking，可以借鉴。一个 extension 是让 VLM 也生成一个 collision cost function：
```python
def collision_cost(P_moved, P_static_environment):
    return chamfer_distance(P_moved, P_static_environment) if penetration > 0 else 0
```

**2. Local minima 问题**

SLSQP 是 local optimizer。Multi-stage task 里，每个 stage 独立优化，可能在 stage transition 处出现 deadlock（前一 stage 的 terminal config 不是下一 stage 的 good initial guess）。这本质上是 stochastic planning vs deterministic optimization 的取舍。可以引入 sampling-based warm start（OMPL/RRT）或者 diffusion-based proposal。

**3. 没有 dynamics / deformable objects**

Cost function 假设 rigid body transformation。倒水这种 task 实际涉及 fluid dynamics，GeoManip 把它简化成 "tilt teapot by 30 degrees" + flow control check "is cup filled"。这是 functional approximation 而不是 physics simulation。

**4. VLM 推理 cost**

每个 stage 至少需要：1 次 SAM + 1 次 VLM mask selection + 1 次 VLM code generation for mask processing + 1 次 VLM constraint generation + 1 次 VLM cost function generation。一个 6-stage task 就要 30+ 次 VLM call，每次 GPT-4o 几秒，total 几十秒到分钟级。这比 diffusion policy (100ms 级 inference) 慢得多。

不过这点对 data collection for IL 反而是优势 — 一次性生成大量 ground-truth trajectory 给小模型学习。

**5. Point cloud 的 categorical 模糊性**

"knife blade" 在 point cloud 里和 "knife handle" 边界可能模糊。SAM 给出的 mask 不一定 align 到 functional boundary。这导致 cost function 在 PCA 算 axis 时可能混入 handle 点，污染 axis 估计。一个 fix 是用 multiple mask candidates 的 ensemble，取 cost 最低的。

**6. 联想到 Karpathy 的 "Software 2.0"**

GeoManip 在某种意义上是 **Software 1.5** — 用 VLM 生成 symbolic code（Software 1.0 的结构），但 code 内容由 learned model 决定（Software 2.0 的 flexibility）。比 VLA 端到端（Software 2.0 pure）多了 structure 和 interpretability，比手写规则（Software 1.0 pure）多了 generalization。这种 hybrid 在 data-sparse domain 应该会成为主流。

参考你这个观点：https://karpathy.medium.com/software-2-0-a64182b37c35

**7. 与 differentiable physics 的连接**

Cost function 是 non-differentiable（用了 abs、argmax、if 之类），所以只能用 SLSQP 这种 gradient-free / numerical gradient 方法。如果让 cost function differentiable，可以用 iLQR / DDP / gradient descent 大幅加速。这暗示了一个方向：**VLM 生成 differentiable cost function**（用 JAX/Torch 写），然后 chain rule 自动求导。这是 Differentiable Physics + LLM 的结合点。

参考：Toussaint 2018 "Differentiable Physics and Stable Modes" https://arxiv.org/abs/1805.08367

---

## 十、最值得 take-away 的几个 idea

1. **Symbolic geometric constraint 作为 language→action 的中间层** — 比 keypoint 丰富，比 pixel-level value map 结构化，比 end-to-end VLA 可解释。
2. **Select-process scheme** — SAM + VLM 的分工很 elegant，可以迁移到其他 fine-grained part segmentation 任务（medical imaging、autonomous driving lane detection）。
3. **In-context constraint editing** — 5 个 generalist features 的本质都是 "把 constraint 当作 first-class object 来 manipulate"，这和 program synthesis 的思路一致。
4. **Cost function = Reward function** — 把 VLM-generated cost function 当 reward signal 训 reward model，是 IL→RL 桥梁的优雅设计。
5. **Training-free SOTA** — 在 real-world 上 zero-shot 65% vs OpenVLA 30-shot 15%，再次印证 **structure beats data** 在小数据 regime 的威力。

整体上这篇工作非常有 2024-2025 neuro-symbolic robotics 的代表性 — VLM 的 reasoning 接上 classical optimization 的 precision，用 in-context 方式实现 zero-shot manipulation。它没有端到端学，但反而因此获得了 OOD generalization、可解释性、和 on-the-fly adaptation 这三个 VLA 模型当前最缺的属性。

参考链接汇总：
- GeoManip 主页: http://geoconstraintmanip.github.io  
- ReKep: https://arxiv.org/abs/2409.01652  
- Code as Policies: https://arxiv.org/abs/2209.07726  
- VoxPoser: https://arxiv.org/abs/2307.05973  
- OpenVLA: https://arxiv.org/abs/2406.09246  
- CuRobo: https://github.com/NVlabs/curobo  
- CoTracker: https://arxiv.org/abs/2307.07635  
- Logic-Geometric Programming: https://arxiv.org/abs/1707.05818  
- Differentiable Physics: https://arxiv.org/abs/1805.08367  
- Riemannian Motion Policies: https://arxiv.org/abs/1801.02854
