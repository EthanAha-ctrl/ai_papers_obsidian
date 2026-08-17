---
source_pdf: VoxPoser.pdf
paper_sha256: cf1cadcf7a1ee44979908c07698f87755c9b5340a79420db509a92d6d2533d42
processed_at: '2026-08-13T03:25:27-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VoxPoser - 人话版

好，Andrej，我换个讲法，把它当一个故事来讲，中间该上公式的地方还是上，但重点是把 intuition 给你 build 起来。

---

## 先讲一个场景

你给机器人说一句："open the top drawer and watch out for the vase"。

之前大家怎么干呢？LLM 坐在那儿想一下，说：好，第一步 move_to(drawer_handle)，第二步 grasp()，第三步 pull()，第四步 open_gripper()。这就像你给一个人一张 todo list，但每一步都是一个"写死的"动作模板。问题是这个 list 里的每一步都是粗粒度的——它告诉你"去 drawer handle"，但不说具体怎么绕过 vase，也不说从哪个角度去抓，更不说碰到 vase 了怎么办。Vase 这个约束根本就进不了那个 pipeline，因为 primitives 里没有 "avoid" 这个东西。

而且回到 action 本身的频率问题。Robot 控制是 kHz 级的连续信号，LLM 输出是 text token，你让 LLM 直接吐 action 就是把一个高维连续空间硬塞进离散文本，信息量完全 mismatch。

所以真正的问题是：**LLM 脑子里的知识怎么落到 robot 的 continuous 控制空间里？**

VoxPoser 的回答特别有意思。

---

## 核心比喻：把 LLM 当 cost function writer

想象你是个 motion planner，本来你的 cost function 是你自己写的——"target 在哪儿，obstacle 在哪儿，penalty 多大"。这个 cost function 你得手调，每个 task 都不一样，特别烦。

VoxPoser 说：咱别自己写 cost function 了，让 LLM 写。但 LLM 不会直接吐数值数组，它会写 Python 代码。这段 Python 代码干一件事：调几个 perception API 搞清楚 scene 里东西的 3D 位置，然后在一个 $100 \times 100 \times 100$ 的 voxel grid 上，把"好去的地方"填上高值，把"别去的地方"填上低值。

这个 voxel grid 就是 **3D value map**。

所以你看，整个 trick 是把 "LLM 知道什么该抓什么该躲" 这个 semantic knowledge，通过 code + perception API 这条路，**翻译成了 3D 空间里的 cost field**。Cost field 是 motion planner 能直接吃的东西，它本来就是干这个的——给个 cost field 它就能 search 出一条 path。

这个 mapping 的妙处在于：cost function 是 **continuous + dense + composable** 的。Affordance map 和 avoidance map 可以叠在一起（"想去的"和"想躲的"同时存在），motion planner 在这个 joint cost field 上找一条最优 path 就同时解决了"去 drawer"和"躲 vase"两个约束。你用 primitives 是没办法表达这种 spatial composition 的——你不能说 "move_to(handle) AND avoid(vase)"，因为 primitives 是 sequential 的，不是 spatial 的。

---

## 数学上到底是怎么回事

整件事本质上是个 trajectory optimization，写出来是这样：

$$\min_{\tau_i^r} \left\{ \mathcal{F}_{task}(\mathbf{T}_i, \ell_i) + \mathcal{F}_{control}(\tau_i^r) \right\} \quad \text{s.t.} \quad \mathcal{C}(\mathbf{T}_i)$$

逐个讲：
- $\tau_i^r$ 是 robot 在第 $i$ 个 sub-task 下的 trajectory，就是一串 6-DoF end-effector waypoints
- $\mathbf{T}_i$ 是整个 environment state 随时间的演化，trajectory 是它的一部分
- $\ell_i$ 是 sub-task 的 language instruction，比如 "grasp the top drawer handle"
- $\mathcal{F}_{task}$ 衡量 "这个 trajectory 走完之后，task 完成了没"
- $\mathcal{F}_{control}$ 是控制代价，比如鼓励省力、省时间
- $\mathcal{C}(\mathbf{T}_i)$ 是 dynamics 和 kinematics 的硬约束

难的就是 $\mathcal{F}_{task}$——你怎么从一句自然语言算出 trajectory 完成没完成？这中间没有 labeled robot data，没有个能直接训出来的端到端 predictor。

VoxPoser 的 trick 是：定义一个 voxel value map $\mathbf{V} \in \mathbb{R}^{w \times h \times d}$，它告诉你空间里每个点的"价值"。然后把 task cost 近似成：

$$\mathcal{F}_{task} = -\sum_{j=1}^{|\tau_i^e|} \mathbf{V}(p_j^e)$$

- $p_j^e \in \mathbb{N}^3$ 是 "entity of interest" $e$ 在第 $j$ 步的离散 $(x, y, z)$ 坐标
- $\mathbf{V}(p_j^e)$ 是那个 voxel 上的 value
- 负号因为我们在 minimize cost，等价于 maximize accumulated value

$e$ 不一定是 robot end-effector，可以是任何 "我们应该关注它运动的东西"——比如 drawer handle，比如要 push 的 block。这样 task cost 就变成"这条 path 经过的 voxel 价值之和"，trajector optimization 变成了 path-finding in voxel grid。

光有位置还不够，6-DoF 还需要 rotation、gripper、velocity。所以又加了三个 map，都是 LLM 来 compose：

| Map | 输出空间 | $k$ (channel) | 含义 |
|-----|---------|--------------|------|
| $\mathbf{V}_{aff}$ (affordance) | $\mathbb{R}$ | 1 | 高值=吸引 |
| $\mathbf{V}_{avoid}$ (avoidance) | $\mathbb{R}$ | 1 | 高值=排斥 |
| $\mathbf{V}_r$ (rotation) | $SO(3)$ | 4 (quaternion) | 目标朝向 |
| $\mathbf{V}_g$ (gripper) | $\{0, 1\}$ | 1 | 开/关 |
| $\mathbf{V}_v$ (velocity) | $\mathbb{R}$ | 1 | 速度倍率 |

所有 map shape 都是 $(100, 100, 100, k)$，$k$ 根据类型不同。

---

## Pipeline 到底怎么跑起来的

LLM 不直接写一个巨大的 code，而是分成好几层 LMP（Language Model Program），每层是一个独立的 prompt + LLM call：

最上面一层叫 **planner**，吃进 user instruction $\mathcal{L}$，吐出一串 sub-tasks $\ell_1, \ell_2, \ldots, \ell_N$。比如 "open the top drawer and watch out for the vase" 会被拆成 "grasp the top drawer handle" 和 "pull the drawer open" 两步。

中间一层叫 **composer**，吃一个 sub-task $\ell_i$，决定要调哪些 value map LMP。

底下五个 LMP 各管一种 map。每个 LMP 拿到 sub-task 的 language 描述后，自己写 Python 代码。代码长什么样呢？它会调一个 `detect(obj_name)` 函数，这个函数内部跑 perception pipeline：

```
text query → OWL-ViT (open-vocab detector) → bbox
         → SAM (Segment Anything) → mask
         → XMEM → 跨帧 track mask
         → RGB-D back-projection → point cloud + occupancy grid + mean normal
```

返回一个 dict: `{center_pos, occupancy_grid, mean_normal}`。

然后 LLM 写的 code 用 NumPy 操作这个 3D 数组。比如它会说：在 handle 的 center position 周围 radius 2cm 的 voxel 全部填 1.0（affordance map），在 vase 周围 radius 5cm 填 1.0（avoidance map），在 handle 位置填一个 face handle normal 的 quaternion（rotation map）...

这些 map 喂给一个 motion planner，它做的事很简单：拿 affordance 和 avoidance map 加权组合成一个 cost map，在 voxel grid 上 greedy search 找一条 collision-free path，然后沿 path 读取每个点的 rotation/velocity/gripper 约束。最后输出第一个 waypoint 给 OSC (Operational Space Controller, Khatib 1987) 执行，然后 **5 Hz 重新规划一遍**。

参考: https://ieeexplore.ieee.org/document/1087247

---

## 为什么 zero-shot 能 work

这个我觉得是 paper 里最 under-appreciated 的点。作者自己也说 "surprisingly find"。

你看，他们用的 dynamics model 其实特别 trivial——大部分 task 直接假设 scene 是 static 的，连个 learned dynamics 都没有。按理说这种 model 完全没法做 closed-loop control。

但它 work，因为有两个东西在兜底：

第一，**value map 是 dense reward**。它不是"到了 target 才给 reward"的稀疏 reward，而是空间里每个点都有梯度。Greedy search 在 dense gradient 上走，本质上就是在做 gradient descent，不需要 model 多准，跟着梯度走就行。

第二，**MPC 5 Hz re-plan**。每 0.2 秒重新看一次 observation，重新生成 value map（因为 `execute` 接收的 map 是 function 不是 array，每次调都能拿最新 perception），重新规划。这等于把 closed-loop feedback 的职责从 dynamics model 转移到了 perception + re-plan 上。Model 不准没关系，反正每 0.2 秒就修正一次。

所以整个 system 的 robustness 不来自 dynamics model，而是来自"快速感知 + 快速重规划"这个循环。这个 insight 我觉得挺重要——它说明在 manipulation 里，如果你有 reliable perception + 高频 replanning，你不需要一个特别准的 dynamics model 就能做很多 task。

这也解释了为什么 Table 1 里 VoxPoser 在 "Dist."（有干扰）设定下表现还是不错——有人推 robot、有人移动 target、有人甚至把 robot 刚关上的 drawer 又拉开，系统都能 recover，因为每 0.2 秒它都在看新世界、重新规划。

---

## Online Dynamics Learning 这块很聪明

对于 door、fridge、window 这种 contact-rich task，zero-shot 跑不动——因为你不知道 handle 要先按下多少才能拉开，这种 contact 的物理细节 LLM 的 text knowledge 给不了。

标准的做法是 online 学一个 dynamics model $g_\theta$：robot 乱试 action 收集 $(o_t, a_t, o_{t+1})$ transitions，用 L2 loss 训练 $\min_\theta \|g_\theta(o_t, a_t) - o_{t+1}\|^2$。

问题在于 action sampling distribution $P(a_t | o_t)$。如果你在整个 action space $\mathcal{A}$（7-DoF，连续）里 uniform random sample，绝大多数 action 根本碰不到 door handle，更别说 "press down" 这种 meaningful interaction。Table 3 里 "No Prior" 一栏全是 TLE（time limit exceeded，12 小时跑不完），就是这个原因。

VoxPoser 的 trick 是：把 zero-shot 生成的 trajectory $\tau_0^r$ 当作 exploration prior。虽然 $\tau_0^r$ 本身可能不完成 task（zero-shot door 只有 6.7% success），但它至少把 robot 带到 handle 附近、给出大致正确的 press-down 方向。然后只在 $\tau_0^r$ 附近加小扰动做 local exploration：

$$P(a_t | o_t, \tau_0^r) = \tau_0^r + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

效果是爆炸性的：

| Task | Zero-Shot | No Prior (12h limit) | w/ Prior |
|------|-----------|----------------------|----------|
| Door | 6.7% | 58.3% (TLE) | 88.3% in ~142s |
| Window | 3.3% | 36.7% (TLE) | 80% in ~137s |
| Fridge | 18.3% | 70% (TLE) | 91.7% in ~71s |

注意右边时间：**3 分钟以内**。从"12 小时跑不完"到"3 分钟搞定 80-91%"，这个 jump 完全是因为 LLM 的 commonsense 把 exploration space 从 $\mathbb{R}^7$ 压缩到 $\tau_0^r + \text{noise}$ 这一条 1D 流形附近。Sample efficiency 提升了几个数量级。

这个 insight 蛮深远的：LLM 的 knowledge 即使不直接 solve task，也能作为 **exploration prior** 指导 online learning。这跟预训练 + 微调的范式有点像，只不过这里是 "LLM 给 prior + online interaction 做微调"，而不是 "language pretraining + downstream fine-tuning"。

---

## Generalization 的数据挺震撼

Table 2 最值得看的是 UI UA（unseen instruction + unseen attribute）这一行——最难的 setting：

| Method | UI UA Object Int. | UI UA Composition |
|--------|-------------------|------------------|
| U-Net + MP (supervised) | 0.0% | 0.0% |
| LLM + Prim (Code as Policies) | 17.5% | 25.0% |
| VoxPoser | **65.0%** | **76.7%** |

Supervised U-Net 直接崩到 0%，因为它在 train data 上拟合死，unseen 完全不行。LLM + Primitives 还有一点泛化（17-25%），但 primitives 的 vocabulary 限制了它能表达的 spatial 关系。

VoxPoser 在最难 setting 还有 65-77%，几乎跟 seen setting 打平。这个 zero generalization gap 我觉得是整个 paper 最强的一个 claim。

为什么？因为 VoxPoser 把 instruction parsing 和 spatial grounding 彻底解耦了：
- Instruction 的 syntax/semantic 由 LLM 处理（LLM 在 internet text 上预训练，见多识广）
- Spatial grounding 由 VLM 处理（VLM 在 internet image-text pair 上预训练）
- 组合它们的是 code，code 是 deterministic 的

Unseen instruction 不会让 LLM 傻眼（它本来就是在 open vocabulary 上 work 的），unseen attribute 不会让 VLM 傻眼（OWL-ViT 是 open-vocab detector）。所以整个 pipeline 没有任何一处会"见过才能 work"。

---

## 跟其他路线的对比

同期还有几条平行的路线，可以放一起看：

**RT-1 / RT-2 路线（end-to-end robot transformer）**：直接从 (image, instruction) → action token，喂大量 robot data 训练。优势是直接 encode 物理 knowledge，能做 contact-rich；劣势是需要巨量 robot data，generalization 受限于 data coverage。VoxPoser 的 zero-shot 能力在 unseen instruction 上吊打 RT-1 的 generalization，但 contact-rich task 上 VoxPoser 又要靠 online learning 才能赶上 RT-2 这种直接 encode physical knowledge 的方法。

参考 RT-1: https://arxiv.org/abs/2212.06817

**Eureka / Language to Reward 路线（LLM 写 reward code for RL）**：让 LLM 写 reward function 的 Python code，然后在 simulator 里跑 RL。VoxPoser 跟它们 idea 几乎一样（都是 LLM → cost/reward function），但 VoxPoser 的 cost 是 grounded 在 real RGB-D observation 上的 3D voxel map，直接给 MPC 用；Eureka/L2R 的 reward 是 simulator 里的 scalar function，给 RL agent 用。前者 zero-shot deploy 到 real robot，后者还要在 simulator 里训 policy。

参考 Eureka: https://arxiv.org/abs/2310.12931
参考 Language to Reward: https://arxiv.org/abs/2306.08647

**Code as Policies 路线（LLM + primitives）**：VoxPoser 的直接 predecessor。Code as Policies 让 LLM 写 code 调 primitives（move_to, grasp, ...），VoxPoser 让 LLM 写 code compose value map。区别是 primitives 是 discrete + sequential 的，value map 是 continuous + spatial 的。Value map 能表达 "同时被 drawer 吸引 + 被 vase 排斥"，primitives 表达不了。Table 1 里 "Move & Avoid" task 就是专门 show 这个：baseline 0/10，VoxPoser 9/10。

参考 Code as Policies: https://arxiv.org/abs/2209.07753

---

## 那些 Emergent Capabilities 挺有意思

Appendix A.2 里写了几个 LLM knowledge "渗透" 出来的行为：

**"I am left-handed" + "set up the table"**：VoxPoser 理解 left-handed 在 table setting 语境下意味着 fork 该在 bowl 左边，于是把 fork 的 affordance map 从 default 右侧挪到左侧。这种 context-dependent spatial adjustment 是纯 primitives 系统做不到的——primitives 不知道 "left-handed" 跟 fork 位置有什么关系。

**"open the drawer precisely by half"**：VoxPoser 不知道 drawer fully open 是多远（没 object model），于是它自己想了个策略：先 fully open 记录 handle displacement，再 close 回 midpoint。这其实是 visual servoing + program memory 的组合，LLM 自己 emergent 出来的。

**"用 ramp 判断哪个 block 更重"**：VoxPoser 选择把两个 block 都推下 ramp，选 travel distance 更远的为重的。这里有个有趣的 bug：在 frictionless 理想世界里两个 block 应该走一样远（Galileo free-fall thought experiment），LLM 复现了一个人类常见的物理 misconcept。这说明 LLM 的 "physics knowledge" 是从 internet text 里来的 implicit bias，不是真的物理 engine。

---

## 我觉得真正深的一层 intuition

VoxPoser 的核心 insight 可以一句话讲完：

> LLM 的 knowledge 应该被表达成 spatial cost function，不应该被表达成 action sequence。

为什么？因为 cost function 有 action sequence 没有的三个性质：

**Composability**：cost 可以叠加，"去 drawer" + "躲 vase" = 两个 map 加权求和。Action sequence 没法叠加——你不能"去 drawer"的同时"躲 vase"，你得先去再躲，但物理上应该是同时的。

**Density**：cost 覆盖整个 3D 空间，每个点都有梯度信息。Action sequence 只在几个离散点有定义，中间靠 interpolation，遇到 perturbation 就得重新生成。

**Frequency decoupling**：LLM 几秒生成一次 cost map 就够了，MPC 高频执行。Action sequence 要么 LLM 高频生成（做不到），要么 open-loop 执行（不 robust）。

这跟 classical robotics 里的 potential field methods（Khatib 1986）是同一个 idea，只是 cost 的来源从 "engineer 手写" 变成了 "LLM 写 code 生成"。Khatib 的 potential field 在 80 年代就提出来了，但一直受限于 cost function 难设计——你得手调每个 task 的 attractive/repulsive field。LLM 把这个 bottleneck 拆掉了。

参考 Potential Field: https://ieeexplore.ieee.org/document/1087247

所以如果让我总结 VoxPoser 的真正贡献，我会说它做了一个非常 elegant 的 **interface design**：在 LLM 的 symbolic knowledge 和 robot 的 continuous control 之间，找到了 "3D voxel value map" 这个 middle ground 作为 bridge。这个 bridge 同时是 semantic 的（LLM 能理解和生成）、spatial 的（robot 能执行和感知）、composable 的（能组合复杂约束）。

这种 "找对 interface 比 find 对 method 更重要" 的感觉，其实跟 differentiable programming、program synthesis 这些领域的精神一脉相承——你不需要端到端 learn 一个 giant network，你只需要在 symbolic reasoning 和 numerical optimization 之间找对那个连接点。

Project page: https://voxposer.github.io/
Code: https://github.com/huangwl18/VoxPoser

---

需要我再展开哪一块都行——比如那几个 LMP 的 prompt 长什么样、motion planner 的 greedy search 具体怎么实现、value map 怎么从 sparse densify、dynamics model 的 MLP 架构细节，或者你想聊聊这个 framework 的 failure mode 和改进方向。

---

# VoxPoser 深度技术解析

Andrej, 这篇 paper 是 Stanford 的 Wenlong Huang (SayCan, Inner Monologue, Code as Policies 的作者) 在 LLM for robotics 方向的 evolution。我把它放到一个更宏的 trajectory 里来理解:从 SayCan (LLM 做 high-level planning) → Code as Policies (LLM 写代码调 primitives) → VoxPoser (LLM 写代码 compose 3D value maps)。每一步都在把 LLM 的 "knowledge" 往更 low-level 的物理控制层推。

## 1. Core Problem & 关键 Insight

**Problem setting**: 给定 free-form language instruction $\mathcal{L}$ (e.g. "open the top drawer and watch out for the vase"), 在 RGB-D observation 下合成 robot trajectory $\tau^r$ — 一个 dense sequence of 6-DoF end-effector waypoints。

**为什么 LLM 不能直接 output actions?** 这是个 frequency mismatch 问题:
- LLM 输出是 text token, 几 Hz 量级
- Robot 控制需要 kHz 级, 且 action space 是 high-dimensional (e.g. 7-DoF joints + gripper)
- 文本空间到连续 action space 没有 natural mapping

**关键观察**: LLMs excel at 推断 **affordances** 和 **constraints**。比如 "open the top drawer" 这条 instruction, LLM 能推断:
1. top drawer 的 handle 应该被 grasp (affordance)
2. handle 需要 translate outwards (motion direction)
3. robot 应该 stay away from vase (constraint)

**核心 idea**: LLM 不直接 output actions, 而是 output **Python code**, 这段 code 调用 perception APIs (VLMs) 获取 3D 空间几何信息, 然后用 NumPy 操作 manipulate 一个 3D voxel array, 在相关 spatial locations 写入 cost/reward。这个 voxel array 就是 **3D value map**, 直接作为 motion planner 的 objective function。

这本质上是把 LLM 的 commonsense knowledge 转化为一个 potential field / cost field, 类似 Khatib 1986 的 artificial potential field 方法, 但是 cost 的来源从 hand-designed 变成了 LLM-composed。

参考 Khatib 原始 paper: https://ieeexplore.ieee.org/document/1087247

## 2. 数学 Formulation

### 2.1 整体优化问题 (Equation 1)

$$\min_{\tau_i^r} \left\{ \mathcal{F}_{task}(\mathbf{T}_i, \ell_i) + \mathcal{F}_{control}(\tau_i^r) \right\} \quad \text{s.t.} \quad \mathcal{C}(\mathbf{T}_i)$$

变量解释:
- $\tau_i^r$: robot trajectory for sub-task $i$, 是 dense 6-DoF end-effector waypoints
- $\mathbf{T}_i$: environment state evolution, $\tau_i^r \subseteq \mathbf{T}_i$
- $\ell_i$: 第 $i$ 个 sub-task 的 instruction (e.g. "grasp the drawer handle")
- $\mathcal{F}_{task}$: task cost, 衡量 $\mathbf{T}_i$ 完成 $\ell_i$ 的程度
- $\mathcal{F}_{control}$: control cost (e.g. 最小化 effort / time)
- $\mathcal{C}(\mathbf{T}_i)$: dynamics + kinematics constraints

Note: instruction $\mathcal{L}$ 被分解成 sub-tasks $\ell_{1:n}$, 这个 decomposition 由 high-level planner (LLM) 给出。

### 2.2 Value Map 近似 Task Cost

关键 insight: 大量 task 可以被 voxel value map $\mathbf{V} \in \mathbb{R}^{w \times h \times d}$ characterizes, 它 guides 一个 "entity of interest" $e$ 的运动轨迹 $\tau^e$ ($e$ 可以是 end-effector / object / object part)。

$$\mathcal{F}_{task} = -\sum_{j=1}^{|\tau_i^e|} \mathbf{V}(p_j^e)$$

其中:
- $p_j^e \in \mathbb{N}^3$ 是 entity $e$ 在 step $j$ 的 discretized $(x, y, z)$ 位置
- $\mathbf{V}(p_j^e)$ 是该位置 voxel 的 value (高 value = attraction, 低 value = repulsion)
- 负号: value map 越高越好, 我们 minimize cost = maximize accumulated value

这就是把 trajectory optimization 问题变成了 graph search / sampling 在 voxel grid 上的 path-finding。

### 2.3 Trajectory Parametrization 扩展

Position map $\mathbf{V}: \mathbb{N}^3 \to \mathbb{R}$ 只能给 cost。为了完整描述 6-DoF trajectory, 作者引入:
- **Rotation map** $\mathbf{V}_r: \mathbb{N}^3 \to SO(3)$ ($k=4$, quaternion)
- **Gripper map** $\mathbf{V}_g: \mathbb{N}^3 \to \{0, 1\}$ (open/close)
- **Velocity map** $\mathbf{V}_v: \mathbb{N}^3 \to \mathbb{R}$ (scale factor, e.g. 0.5 = 半速)

这些都是 LLM compose 的, shape 为 $(100, 100, 100, k)$, $k$ 根据类型不同。

## 3. 系统架构深度解析

Figure 2 展示的 pipeline 可以拆成 3 层 LMP (Language Model Programs):

### 3.1 LMP 层级 (借鉴 Code as Policies, Liang et al.)

```
User Instruction L
        ↓
[Planner LMP]  → decompose → ℓ_1, ℓ_2, ..., ℓ_N
        ↓
[Composer LMP] (per sub-task ℓ_i)
        ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
↓              ↓              ↓              ↓              ↓
get_affordance get_avoidance  get_rotation   get_gripper   get_velocity
   LMP            LMP            LMP            LMP            LMP
↓              ↓              ↓              ↓              ↓
V_aff (100³,1) V_avoid(100³,1) V_rot(100³,4) V_grip(100³,1) V_vel(100³,1)
        ↓
[Motion Planner + MPC] → trajectory → robot
```

每个 LMP 接收 5-20 个 examples 作为 in-context prompt (few-shot)。

### 3.2 Perception Stack (VLM 调用)

LLM 写的 Python code 调用 `detect(obj_name)` API, 内部 pipeline:

```
obj_name (text)
    → OWL-ViT (open-vocab detector) → bbox
    → SAM (Segment Anything) → mask
    → XMEM (video tracker) → tracked mask over time
    → RGB-D back-projection → point cloud + occupancy grid + mean normal
```

返回 dictionary: `{center_pos, occupancy_grid, mean_normal}`

参考:
- OWL-ViT: https://arxiv.org/abs/2205.06230
- SAM: https://arxiv.org/abs/2304.02643
- XMEM: https://arxiv.org/abs/2207.07115

### 3.3 Value Map Composition APIs (Appendix A.3)

关键的环境 API:
- `detect(obj_name)`: 返回 object 实例 list with 3D info
- `set_voxel_by_radius(voxel_map, voxel_xyz, radius_cm, value)`: 在 voxel map 中以 radius 赋值
- `get_empty_affordance_map()`: 返回全 0 的 affordance map (高 value = 吸引)
- `get_empty_avoidance_map()`: 返回全 0 的 avoidance map (高 value = 排斥)
- `cm2index(cm, direction)`: 物理单位 → voxel 索引
- `pointat2quat(vector)`: 期望指向方向 → target quaternion

### 3.4 LLM 写的代码示例 (mentally reconstruct)

对于 "grasp the top drawer handle":
```python
# Detect handle
handles = detect("top drawer handle")
handle = handles[0]

# Compose affordance map: attract to handle center
aff = get_empty_affordance_map()
set_voxel_by_radius(aff, handle["center_pos"], 2.0, 1.0)
# Smooth with Euclidean distance transform for gradient

# Compose avoidance map: repel from vase
vases = detect("vase")
avoid = get_empty_avoidance_map()
for vase in vases:
    set_voxel_by_radius(avoid, vase["center_pos"], 5.0, 1.0)
# Smooth with Gaussian filter

# Rotation map: face the handle normal
rot = get_empty_rotation_map()
target_quat = pointat2quat(handle["mean_normal"])
set_voxel_by_radius(rot, handle["center_pos"], 2.0, target_quat)

# Gripper map: open then close
grip = get_empty_gripper_map()
set_voxel_by_radius(grip, handle["center_pos"], 2.0, 0)  # open on approach
set_voxel_by_radius(grip, handle["center_pos"], 0.5, 1)   # close at handle

execute(movable=handle, affordance_map=aff, avoidance_map=avoid,
        rotation_map=rot, gripper_map=grip)
```

**Intuition**: LLM 不需要知道 IK, control frequency, joint limits。它只需要知道 "handle 该 grasp, vase 该 avoid" 这种 semantic-level 知识, 然后用 code + perception API 把这个 knowledge 投影到 3D voxel space。剩下的 optimization 由 motion planner 完成。

## 4. Zero-Shot Trajectory Synthesis

### 4.1 Motion Planning 算法

Cost map 计算:
$$\text{CostMap} = -2 \cdot \text{normalize}(\mathbf{V}_{aff}) - 1 \cdot \text{normalize}(\mathbf{V}_{avoid})$$

权重 2:1 表示 affordance 比 avoidance 更重要。Negative 因为 low cost = high value。

**Greedy search** (paper 中说是 zeroth-order optimization + random sampling):
1. 从当前位置出发, 在 voxel grid 上做 greedy search 找 collision-free path $p_{1:N} \in \mathbb{R}^3$
2. 在每个 $p_i$ 上 enforce rotation / velocity / gripper 约束 (从对应 value maps 读取)
3. 输出第一个 waypoint 给 OSC (Operational Space Controller), 然后 re-plan

**MPC at 5 Hz**: 每秒 re-plan 5 次, 用最新 observation。这给了 system 对 dynamic perturbation 的 robustness。

### 4.2 为什么 zero-shot 能 work?

关键观察: "VoxPoser effectively provides dense rewards in the observation space and we are able to replan at every step, we surprisingly find that the overall system can already achieve a large variety of manipulation tasks even with simple heuristics-based models."

也就是说, dense value map + 高频 re-plan = implicit feedback control。即使 dynamics model 是 trivial (e.g. static scene assumption), replan 机制 compensate 了 model 误差。

参考 Operational Space Formulation (Khatib 1987): https://ieeexplore.ieee.org/document/1087247

## 5. Online Dynamics Learning (Sec 3.4)

对于 contact-rich tasks (door, fridge, window), zero-shot 不够, 需要 dynamics model。

### 5.1 问题设定

标准 online learning loop:
1. 收集 transitions $(o_t, a_t, o_{t+1})$, 其中 $a_t = \text{MPC}(o_t)$
2. 训练 dynamics model $g_\theta$ 最小化 $\|g_\theta(o_t, a_t) - o_{t+1}\|_2^2$

**瓶颈**: action sampling distribution $P(a_t | o_t)$。Random sampling over full action space $\mathcal{A}$ 极其 sample-inefficient, 因为大多数 action 不接触 relevant object。

### 5.2 VoxPoser 作为 Exploration Prior

VoxPoser 的 zero-shot trajectory $\tau_0^r$ 编码了 LLM 的 commonsense (e.g. "press handle down first to open door")。把它作为 prior:

$$P(a_t | o_t, \tau_0^r) = \tau_0^r + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

只在 $\tau_0^r$ 附近加小噪声做 local exploration, 而不是 global random exploration。

### 5.3 实验结果 (Table 3)

| Task | Zero-Shot Success | No Prior Success | No Prior Time | w/ Prior Success | w/ Prior Time |
|------|------|------|------|------|------|
| Door | 6.7% ± 4.4% | 58.3% ± 4.4% | TLE (>12h) | **88.3% ± 1.67%** | **142.3 ± 22.4s** |
| Window | 3.3% ± 3.3% | 36.7% ± 1.7% | TLE | **80.0% ± 2.9%** | **137.0 ± 7.5s** |
| Fridge | 18.3% ± 3.3% | 70.0% ± 2.9% | TLE | **91.7% ± 4.4%** | **71.0 ± 4.4s** |

注意:
- "No Prior" 在 12 小时内根本 converge 不了 (TLE)
- "w/ Prior" 用 **3 分钟以内** 的 online interaction 就达到 80-91% success
- 即使 zero-shot 不成功 (6.7% door success), prior 仍然 meaningful — 它指向 correct interaction region, dynamics model 在 local region 内学习就够

**Intuition**: LLM 的 commonsense 把 exploration space 从 7-DoF action space $\mathbb{R}^7$ 压缩到 1D curve $\tau_0^r + \text{noise}$, sample efficiency 提升 ~10⁶ 量级。

## 6. 实验结果深度分析

### 6.1 Real-World (Table 1)

| Task | LLM+Prim. Static | LLM+Prim. Dist. | VoxPoser Static | VoxPoser Dist. |
|------|------|------|------|------|
| Move & Avoid | 0/10 | 0/10 | 9/10 | 8/10 |
| Set Up Table | 7/10 | 0/10 | 9/10 | 7/10 |
| Close Drawer | 0/10 | 0/10 | 10/10 | 7/10 |
| Open Bottle | 5/10 | 0/10 | 7/10 | 5/10 |
| Sweep Trash | 0/10 | 0/10 | 9/10 | 8/10 |
| **Total** | **24.0%** | **0.0%** | **88.0%** | **70.0%** |

Baseline 是 Code as Policies 变种 (LLM + primitives like `move_to_pos`, `open_gripper`)。

**关键对比**:
- "Move & Avoid" 和 "Sweep Trash": baseline 0/10, VoxPoser 9/10 — spatial composition 能力是 primitives 无法 capture 的
- "Dist." (有干扰): baseline 几乎全部归零, 因为 primitives 是 open-loop chain; VoxPoser 靠 MPC re-plan 保持 robust

### 6.2 Simulation Generalization (Table 2)

设置: 13 tasks, 2766 unique instructions, 模板化随机化 attributes。

| Train/Test | Category | U-Net+MP [50] | LLM+Prim. [75] | VoxPoser (Ours) |
|------|------|------|------|------|
| SI SA | Object Int. | 21.0% | 41.0% | **64.0%** |
| SI SA | Composition | 53.8% | 43.8% | **77.5%** |
| SI UA | Object Int. | 3.0% | 46.0% | **60.0%** |
| SI UA | Composition | 3.8% | 25.0% | **58.8%** |
| UI UA | Object Int. | 0.0% | 17.5% | **65.0%** |
| UI UA | Composition | 0.0% | 25.0% | **76.7%** |

- **SI/UI**: Seen / Unseen Instructions
- **SA/UA**: Seen / Unseen Attributes

U-Net baseline 在 unseen 上完全 collapse (0%), 因为它要 supervised 数据。VoxPoser 在 UI UA (最难的 setting) 仍有 65-77%, 几乎没有 generalization gap (相比 SI SA)。

**Intuition**: LLM 把 instruction parsing 和 spatial composition 解耦了 — instruction 的 syntax/semantic 由 LLM 处理, spatial grounding 由 VLM 处理, 两者都是 pre-trained 在 internet-scale data 上, 所以 unseen instruction/attribute 不影响。

### 6.3 Full Simulation Results (Table 4, Appendix)

挑几个有意思的:
- "move to [pos] while moving at [velocity] when within [dist]cm from obj": U-Net 80→0%, LLM+Prim 10→0%, **VoxPoser 100→95%**
- "push the [obj] along the [line]": 全部其他方法 0%, VoxPoser 65→30% — 这个 task 需要 continuous spatial reasoning, primitives 无法表征
- "close the [deixis] drawer by pushing": U-Net 0%, LLM+Prim 60%, VoxPoser 80% — articulated object 交互

### 6.4 Error Breakdown (Figure 4, Sec 4.4)

Paper 把 error 分成三类:
1. **Dynamics error**: dynamics model 预测误差
2. **Perception error**: detector 失败 / 部分 detection 不准
3. **Specification error**: cost / parameter specification 错误 (e.g. U-Net 预测噪声, LLM 参数错误, LLM value map 错误)

结论: VoxPoser 的 specification error 最低 (因为 LLM 推理 robust), 但 perception error 成为主要瓶颈。Real-world 实验中, 大部分失败来自 detector 对 object initial pose 敏感 + object parts 检测不稳定。

## 7. Emergent Capabilities (Appendix A.2)

这部分很有意思, 展示了 LLM 的 world knowledge 如何 "leak through" VoxPoser:

### 7.1 Behavioral Commonsense Reasoning
Task: "set up the table" + "I am left-handed"
- VoxPoser 理解 "left-handed" 在 table setting 语境下意味着 fork 应该在 bowl 左侧, 而不是 default 右侧
- 通过 LLM 写 code 把 fork affordance map 移到 left side

### 7.2 Fine-grained Language Correction
Task: "covering teapot with lid" + "you're off by 1cm"
- VoxPoser 根据 feedback 在 affordance map 上 shift 1cm offset
- 类似 closed-loop correction

### 7.3 Multi-step Visual Program
Task: "open the drawer precisely by half"
- 没有 object model, 不知道 fully open 距离
- VoxPoser 策略: 先 fully open + record handle displacement → close back to midpoint
- 类似 visual servoing + program memory

### 7.4 Estimating Physical Properties
Task: 用 ramp 判断两个 block 哪个更重
- VoxPoser 决定 push both blocks off ramp, 选 travel distance 更远的为更重
- **有趣 note**: 在 frictionless 理想世界中, 两个 block 应该走同样距离 (Galileo 的 free-fall 思想实验) — LLM 复现了 common human misconception, 说明它的 "physics knowledge" 来自 internet text 的 implicit bias

## 8. Limitations & Future Work

1. **依赖 external perception modules**: OWL-ViT + SAM + XMEM 各自有 failure mode。对 fine-grained geometry (e.g. handle 的 exact 3D shape) 不够强
2. **需要 general-purpose dynamics model**: 现在 contact-rich 还是要 online learn dynamics
3. **End-effector trajectory only**: 没有 whole-arm planning, 对 obstacle-avoidance in joint space 弱
4. **Prompt engineering**: 每个 LMP 需要 5-20 examples, 手工设计

Future directions 提到的:
- Multi-modal LLMs (PaLM-E, BLIP-2) 可以做 direct visual grounding, 省掉 perception pipeline
- Alignment methods (RLHF, Constitutional AI) 减少 prompt engineering
- Advanced trajectory optimization (e.g. Riemannian Motion Policies, Trajectory Optimization around obstacles)

参考 PaLM-E: https://arxiv.org/abs/2303.03378
参考 BLIP-2: https://arxiv.org/abs/2301.12597
参考 RMPflow: https://arxiv.org/abs/1801.02854
参考 TrajOpt (convex optimization around obstacles): https://arxiv.org/abs/2205.04422

## 9. 个人思考: 这篇 paper 的真正 contribution

放到 LLM-for-robotics 的更大的图景里看, VoxPoser 解决了一个核心 bottleneck:

**之前的工作** (SayCan, Inner Monologue, Code as Policies, ProgPrompt, etc.): LLM 做 high-level planning, 但 physical interaction 还是靠 hand-designed primitives (e.g. `pick`, `place`, `move_to`). Primitives 的数量和复杂度限制了 system 能做的事。

**VoxPoser 的 trick**: 把 LLM 的 output 从 "primitive selection" 换成 "cost function composition"。Cost function 是 continuous, dense, composable 的, 所以同一个 LLM 能 generate 出任意 task 的 cost map。

这其实跟 reinforcement learning 里 reward design 的思想相通, 但 reward 不再 hand-crafted, 而是由 LLM 的 commonsense 提供。同期/后续的 Eureka (NVIDIA, Ma et al. 2023), Language to Reward (Yu et al. 2023), Text2Reward (Xie et al. 2023) 都在类似方向。

参考 Eureka: https://arxiv.org/abs/2310.12931
参考 Language to Reward: https://arxiv.org/abs/2306.08647
参考 Text2Reward: https://arxiv.org/abs/2309.11489

**VoxPoser vs. 这些 reward-generation 工作**:
- Eureka / L2R / Text2Reward: 在 MuJoCo / Isaac Gym 这种 fully-simulated 环境里生成 reward code, 给 RL 用
- VoxPoser: 在 real robot + real perception 上 generate 3D value map, 给 MPC 用, zero-shot

VoxPoser 的真正价值是 **bridge 到 real-world**: value map 是 grounded 在 RGB-D observation 上的, 不是 abstract reward signal。

## 10. 跟 RT-1 / RT-2 等 end-to-end 方法的对比

同期还有 Google 的 RT-1 / RT-2 / RT-X 等 end-to-end robot transformer 方法。它们的 philosophy 完全相反: 把 robot data 喂给 transformer, 直接 learn 从 (image, instruction) → action。

VoxPoser 的优势:
- Zero-shot generalization (不需要 robot data)
- Compositional (新 task 通过 LLM 推理 + VLM grounding 就能做)
- Interpretable (value map 可视化)

VoxPoser 的劣势:
- 依赖 perception pipeline 的 reliability
- 不学习 contact-rich dynamics (需要 online learning, 不像 RT-2 直接 encode 物理知识)
- 频率限制: 5Hz MPC, 不能做高频 reactive 控制

参考 RT-1: https://arxiv.org/abs/2212.06817
参考 RT-2: https://arxiv.org/abs/2307.15818

## 11. 总结: Build Intuition

VoxPoser 的核心 insight 可以一句话概括:

> **LLM 的 world knowledge 应该被 expressed as spatial cost functions, not as action sequences.**

Cost function 是 composable (affordance + avoidance 可以叠加), dense (覆盖整个 3D observation space), model-agnostic (motion planner 只看 cost), 而且与 frequency 解耦 (LLM 几秒生成一次 cost map, MPC 高频执行)。

这其实跟 classical robotics 里的 potential field methods 是 same idea, 但 cost 的来源从 "engineer hand-designed" 换成了 "LLM-composed via code"。LLM 充当了一个 **semantic-to-spatial translator**。

从 architecture 上看, VoxPoser 是一个 hybrid system:
- LLM (GPT-4): semantic reasoning + code generation
- VLM (OWL-ViT + SAM): open-vocabulary grounding
- Numerical optimization (MPC): low-level control
- Online learning (optional): dynamics for contact-rich

每个 component 都 pre-trained / classical, 通过 code interface 组合, 实现 1+1+1+1 >> 4 的效果。

**最 deep 的 takeaway**: 在 robot learning 还没有 robot foundation model 之前, 把 LLM + VLM 当成 "knowledge source" 而不是 "policy" 来用, 通过 proper interface (code + value map) 让它 grounded 到 physical world, 是一个非常 pragmatic 且 effective 的 design pattern。这种 modular approach 在 generalization 上甚至超过 end-to-end 的 RT-1 (在 unseen instructions / attributes 上)。

Project page: https://voxposer.github.io/
Code: https://github.com/huangwl18/VoxPoser

---

如果你想让我进一步 deep dive 某个部分 (比如 value map LMP 的具体 prompt, motion planner 的 greedy search 实现细节, dynamics learning 的 MLP 架构, 或者跟 Eureka/L2R 的 reward generation 做更细的对比), 我可以继续展开。
