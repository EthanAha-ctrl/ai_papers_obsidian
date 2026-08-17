---
source_pdf: End-to-End Navigation with Vision-Language Models.pdf
paper_sha256: 76f5cbb6b7a62233a63d92c2544ad10fe06bb15dc1196a75184b94046c3e5e31
processed_at: '2026-08-04T04:21:04-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 的核心 idea，用大白话讲就是：**VLM 其实是个纸上谈兵的将军，你直接让它指挥机器人走路，它连东南西北都分不清；但如果你把"走路"变成"做选择题"，它就能瞬间变身战术大师。**

下面我们拆解一下这套"选择题套路"是怎么运作的，以及为什么它能 work。

### 1. 核心套路：把 Navigation 变成 Multiple-Choice QA

VLM（比如 Gemini Flash）很聪明，能看懂图，能理解"去找沙发"是啥意思。但是你如果给它一张第一人称视角的图，问它"下一步该走多少米、转多少度"，它会抓瞎。因为它对 continuous coordinates（连续坐标）的感知能力极差，经常指错位置（可以参考这篇吐槽文：*Vision Language Models are Blind*，https://arxiv.org/abs/2407.06581）。

所以作者搞了个 VLMnav，思路极简：
1. 用 depth 传感器扫一圈，算出哪些方向能走，哪些方向有墙。
2. 从能走的方向里，挑出几个最合适的（尤其是没去过的方向）。
3. 把这几个方向画成带数字编号的箭头，直接 P 在 RGB 图上。
4. 把这张画满箭头的图丢给 VLM，问它："你要找沙发，选 1 号还是 2 号？"
5. VLM 选了 3 号，机器人就执行 3 号对应的位移和旋转。

就这么简单。不需要训练任何神经网络，不需要微调，直接 zero-shot 用现成的 VLM。这就叫把 spatial reasoning 转成了 question answering。

### 2. 内部流程图解析（对应 Figure 2）

整个 pipeline 就像一个流水线，把 VLM 保护起来，只让它做最擅长的事：

- **Navigability（探路）**：机器人有个往下倾斜 25 度的深度相机。它把每个角度的射线打出去，看看多远会撞墙。这生成了一组初始的安全动作 $A_{\text{initial}}$。同时，它维护一个简单的 2D voxel map，2 米内标为"已探索"，2 米外标为"未探索"。
- **Action Proposer（出题）**：这是最精妙的一步。它不能把所有安全动作都画在图上，不然箭头太密 VLM 看不清。它得挑出几个有代表性的。
- **Projection（画图）**：把挑出来的动作，用箭头和带编号的圆圈画在 RGB 图上。如果有个动作是"原地掉头 180 度"，因为它没法在图上画箭头，就单独贴在图旁边，标上 "Turn Around"。
- **Prompting（提问）**：结合 Chain-of-Thought，让 VLM 先描述看到了啥，再定个计划，最后吐出一个数字。注意，这里每次动作只调用一次 VLM，不像 PIVOT 那样要反反复复调多次去迭代高斯分布。

### 3. Action Proposer 的"小心思"（公式人话解析）

这部分是全篇的技术核心。作者怎么挑候选动作呢？核心思想是**偏心"未探索区域"**。

**公式 (1) - 打标签**：
$$e_i = \begin{cases} 1 & \text{if region } (\theta_i, r_i) \text{ is unexplored} \\ 0 & \text{if region } (\theta_i, r_i) \text{ is explored} \end{cases}$$
- $\theta_i$ 和 $r_i$ 是第 $i$ 个候选动作的极坐标角度和距离。
- $e_i$ 就是个 flag。如果走这个动作会到达"未探索"区域，$e_i=1$；如果还在"已探索"区域，$e_i=0$。

**公式 (2) & (3) - 采样密度的双重标准**：
$$A_{\text{final}} \leftarrow A_{\text{final}} \cup \{(\theta_i, r_i) \mid e_i = 1 \text{ and } |\theta_i - \theta_j| \geq \theta_\delta, \forall (\theta_j, r_j) \in A_{\text{final}}\}$$
$$A_{\text{final}} \leftarrow A_{\text{final}} \cup \{(\theta_i, r_i) \mid e_i = 0 \text{ and } |\theta_i - \theta_j| \geq \theta_\Delta, \forall (\theta_j, r_j) \in A_{\text{final}}\}$$
- 这俩公式意思是：加入新动作时，要保证和已经选中的动作之间有足够的角度间距。
- 玄机在阈值上：$\theta_\Delta > \theta_\delta$。也就是说，通向未探索区域的动作，允许排得密一点（比如相差 15 度就可以选）；通向已知区域的动作，必须排得很疏（比如相差 45 度才选）。
- 这就硬生生造出了一个 **explore bias（探索偏好）**。图上密密麻麻的箭头都指向没去过的地方，寥寥无几的箭头指向去过的地方。VLM 瞎选一个都有很大概率在探索新区域。

**公式 (4) - 留一手安全边界**：
$$r_i \gets \min\left(\frac{2}{3} \cdot r_i, r_{\max}\right)$$
- $r_i$ 是动作距离。这里把它砍掉三分之一（乘以 2/3），或者不超过一个最大值 $r_{\max}$。
- 为啥？因为 VLM 判距不准，机器人有惯性，留个安全冗余防撞墙。

### 4. 凭什么这套把戏能 work？

构建直觉的关键在于：**Geometry constrains, VLM discriminates（几何来约束，VLM 来判别）。**

纯靠 VLM 做导航，死路一条，因为它没有 metric 概念。
纯靠经典 Frontier-based exploration（探索边界法），太死板，只会像贪吃蛇一样往没去过的地方钻，找不到特定的目标（比如沙发）。

VLMnav 把两者缝起来了：
1. 底层用 geometry（深度图、voxel map）把所有不靠谱的动作全砍掉，顺便用 explore bias 把 VLM 往未知区域引。
2. VLM 只需要在剩下的几个靠谱选项里，靠它的 semantic 知识挑一个最可能找到沙发的方向。

这其实是 Set-of-Mark prompting（https://arxiv.org/abs/2310.11441）在具身智能里的极致应用。你把选择题的选项直接画在图上，VLM 的 visual grounding 能力就被激活了。

### 5. 实验数据里的猫腻与 Intuition

看实验结果不能只看数字，得看数字背后的隐喻：

- **PIVOT 被按在地上摩擦**：VLMnav 在 ObjectNav 上 SR 50.4%，PIVOT 只有 24.6%。PIVOT 是每次随机撒一堆箭头，根据 VLM 反馈调整高斯分布，迭代好几次。VLMnav 只调一次 VLM，但箭头是用深度图精心算出来的。这说明：**与其让 VLM 在一堆垃圾选项里反复纠结，不如提前帮它把垃圾选项删了。**
- **"碰碰车"真相**：关掉 simulator 的 `allow slide`（允许贴墙滑行）参数，SR 立刻从 50.4% 暴跌到 12.9%。说明 VLMnav 其实经常撞墙，只是 simulator 允许它沿着墙滑过去才没死。因为 Navigability 只看射线能不能穿透，没考虑机器人本身的物理体积。这要在真实世界，早卡在死角哭爹喊娘了。
- **宽视野是王道**：FOV 从 82 度增加到 131 度，SR 线性上升 15%。直觉很简单：视野越宽，角落里的箭头越不容易被裁掉，VLM 看到的选项越全，越不容易钻牛角尖。
- **历史记忆是个坑**：给 VLM 看过去 5 步、10 步、15 步的图和动作，性能不仅没涨，反而跌了。因为现在的 VLM 对多图的时间序列推理很弱，塞了一堆历史图进去，反而分散了它看当前图的注意力。
- **RGB 就够了**：用 Segformer 做个地板分割来代替深度图，性能只掉 3%。用 ZoeDepth 预测深度，掉 10%。直觉是：地板分割是个 binary task，很稳；深度预测是 continuous regression，稍有误差，射线法就算错，导致机器人撞墙。所以这玩意儿以后连深度相机都不用，普通 RGB 摄像头加个分割模型就能跑。

### 6. 致命缺陷与未来脑洞

作者很坦诚，承认了几个硬伤：
1. **距离感太差**：13.9% 的情况下，VLM 在距离目标 1 到 1.5 米的地方就早早喊停了（阈值是 1 米）。因为 VLM 根本估不准距离。
2. **卡死无法自救**：如果不让滑行，机器人一旦进死角，由于 Action Proposer 没有生成"后退"动作，直接死锁。

如果顺着这篇 paper 往后想，有几个特别自然的脑洞：
- **把 voxel map 也画给 VLM 看**：既然历史图看不懂，那把 top-down 的小地图直接画在 prompt 里，让 VLM 知道自己在这个房间的啥位置，这不就是给它装了个 GPS 吗？
- **分层调度**：High-level VLM 看 360 度全景图和 voxel map，决定"去那个房间"；Low-level VLM 看当前视角带箭头的图，决定"走哪个箭头"。这绝对能解决过早停止和卡死角的问题。
- **Action value estimation**：别让 VLM 直接吐选项，让它给每个选项打个分，估个 Q-value。这可能比硬做分类更鲁棒。

总之，这篇 paper 的 intuition 极其漂亮：**承认 VLM 是个瘸子，然后给它造个轮椅（几何投影+探索偏好），让它坐着轮椅去打怪。** 这比硬逼着 VLM 学走路（微调输出连续坐标）聪明得多。

参考链接：
- Paper 项目主页：https://jirl-upenn.github.io/VLMnav/
- Set-of-Mark prompting 原理：https://arxiv.org/abs/2310.11441
- PIVOT 基线对比：https://arxiv.org/abs/2402.07872
- VLM 是瞎子的证据：https://arxiv.org/abs/2407.06581
- GOAT-Bench 挑战赛：https://arxiv.org/abs/2406.18209

---

# VLMnav 深度解析：把 Navigation 转成 Question Answering

## 1. Core Insight: Why this works at all

这篇paper的核心**audacious claim**是：一个没有任何navigation training的off-the-shelf VLM (Gemini Flash)，可以直接作为end-to-end policy zero-shot工作。这听起来有点unreasonable，但作者的关键reformulation是：

> **Navigation is fundamentally a spatial reasoning problem, and VLMs excel at QA about images. So make navigation look like image QA.**

具体怎么做的？通过projection把候选actions画在first-person view image上（用箭头+编号circle），然后问VLM"which number should I go to?"。这个reformulation让VLM从"generate a continuous coordinate"这种它不擅长的任务，变成"pick from a discrete set"这种它天然擅长的multiple-choice任务。

类似思想可以追溯到Set-of-Mark prompting (Yang et al. 2023, https://arxiv.org/abs/2310.11441)，但这里的关键是**如何选择这组离散候选actions**，这是整篇paper的真正贡献。

## 2. Architecture Walkthrough

### 2.1 Pipeline Overview (Figure 2 解析)

四个组件串联，每个组件解决一个具体failure mode：

```
RGB-D + pose ξ
    │
    ▼
[Navigability] ──► voxel map (explored/unexplored) + A_initial
    │
    ▼
[Action Proposer] ──► A_final (discrete action set with explore bias)
    │
    ▼
[Projection] ──► Î (image with arrows + numbered circles)
    │
    ▼
[Prompting] ──► VLM call ──► action a*
    │
    ▼
[Optional: Termination call] ──► stop or continue
```

**Key design choice**: 不像PIVOT那样iterative refinement（多次VLM call refine一个Gaussian distribution），VLMnav每次step只做**一次VLM call**就定action。这是latency上很大的节省。代价是action sampling必须非常clever，因为refine的机会没了。

### 2.2 Navigability Module 详解

输入：depth image $D \in \mathbb{R}^{H \times W}$ + agent pose $\xi = (x, y, \theta_{yaw})$

输出：
- Navigability mask $M \in \{0, 1\}^{H \times W}$
- Action set $A_{\text{initial}} = \{(\theta_i, r_i)\}_{i=1}^{N}$
- Voxel map $V \in \{0, 1, \text{unexplored}\}^{H' \times W'}$

**算法描述**：
对每个polar angle $\theta \in [-\text{FOV}/2, +\text{FOV}/2]$ discretized to $\theta_1, \theta_2, \ldots, \theta_N$：

1. Ray-cast from camera optical center along direction $\theta$
2. 沿着这条ray在depth image找第一个obstacle distance $d_{\text{obs}}(\theta)$
3. 该方向上最大可走distance $r_i = \min(d_{\text{obs}}(\theta), r_{\text{max}})$
4. 如果 $r_i > r_{\text{min}}$（threshold），加入 $A_{\text{initial}}$

同时用SLAM-like approach更新voxel map：
- agent周围2米内 → explored
- 2米外可见区域 → unexplored

注意这里作者用的voxel resolution较粗，本质是occupancy grid。这与Chaplot et al.的Object-Goal Navigation with Active Mapping (https://arxiv.org/abs/2006.13348) 思路一致。

**Camera pitch design trick**: paper提到camera tilted down 25°。这个角度让depth image能直接看到floor region，对navigability mask计算至关重要。如果camera平视，会miss掉近处的obstacle。

### 2.3 Action Proposer - 真正的核心创新

这是paper里最subtle也最重要的部分。先看三个公式：

**公式 (1) - Exploration indicator**:

$$
e_i = \begin{cases} 1 & \text{if region } (\theta_i, r_i) \text{ is unexplored} \\ 0 & \text{if region } (\theta_i, r_i) \text{ is explored} \end{cases}
$$

变量解释：
- $\theta_i$：第$i$个候选action的polar angle (相对robot yaw)
- $r_i$：第$i$个候选action的radial distance
- $e_i \in \{0, 1\}$：binary exploration indicator
- "region $(\theta_i, r_i)$ is unexplored"：意思是该action执行后的终点位置在voxel map里属于unexplored voxel

**公式 (2) - Adding unexplored actions**:

$$
A_{\text{final}} \leftarrow A_{\text{final}} \cup \{(\theta_i, r_i) \mid e_i = 1 \text{ and } |\theta_i - \theta_j| \geq \theta_\delta, \forall (\theta_j, r_j) \in A_{\text{final}}\}
$$

变量解释：
- $\theta_\delta$：unexplored actions之间要求的最小angular spacing
- 这个constraint是greedy的：从unexplored actions里逐个add，要求new action与existing $A_{\text{final}}$中所有action的角度差至少 $\theta_\delta$

**公式 (3) - Adding explored actions with wider spacing**:

$$
A_{\text{final}} \leftarrow A_{\text{final}} \cup \{(\theta_i, r_i) \mid e_i = 0 \text{ and } |\theta_i - \theta_j| \geq \theta_\Delta, \forall (\theta_j, r_j) \in A_{\text{final}}\}
$$

变量解释：
- $\theta_\Delta > \theta_\delta$：explored actions用更大的angular spacing threshold
- 这样explored方向actions密度低，unexplored方向actions密度高 = explore bias

**公式 (4) - Safety clipping**:

$$
r_i \gets \min\left(\frac{2}{3} \cdot r_i, r_{\max}\right) \quad \forall (\theta_i, r_i) \in A_{\text{final}}
$$

变量解释：
- $r_{\max}$：hard cap on action displacement
- $\frac{2}{3}$：safety margin factor（保留1/3 distance作为buffer，避免VLM判错或者robot inertia导致的collision）

**Special case**: 如果 $A_{\text{initial}} = \emptyset$ (agent被stuck in corner)，加入 $(\pi, 0)$ action，即原地旋转180°。这个trick很重要，因为没它agent在corner就dead-locked。

### 2.4 Projection (类似 Set-of-Mark)

把每个 $(\theta_i, r_i)$ action project到first-person RGB image上：

1. 用camera intrinsics + pose把action终点(在robot frame下)project到pixel coordinate $(u_i, v_i)$
2. 在image上从底部camera center画arrow指向$(u_i, v_i)$
3. 终点画numbered circle

**为什么visual annotation有效？** 这是set-of-mark prompt的本意。VLM对image中**显式标注**的位置有更好的spatial grounding能力，相比让它在"raw image"上指"那个红色沙发左边一点"。

参考：
- Set-of-Mark: https://arxiv.org/abs/2310.11441
- What does CLIP know about a red circle? (Shtedritski et al.): https://arxiv.org/abs/2304.06712

特殊处理：$(\pi, 0)$ turn-around action因为终点在agent后面无法project，单独画在image侧边并标"Turn Around"。

### 2.5 Prompting (Figure 1)

完整的prompt结构：
```
[system prompt]
You are a robot navigating in a 3D environment...
The arrows labeled 1-N on the image show possible movements.
Arrow 0 means turn around.

[action prompt]
Task: Find the {object_name}
Goal: Get within 1 meter of {object_name}
Available actions: numbered arrows shown on the image

[chain-of-thought instructions]
First, describe what you see in the image.
Then, make a high-level plan.
Finally, choose an action number.

[image input: Î (annotated)]
[goal image if image-based goal]
```

这个prompt遵循了CoT (Chain-of-Thought) prompting的best practice (Wei et al. 2022: https://arxiv.org/abs/2201.11903, Kojima et al. 2022: https://arxiv.org/abs/2205.09121)，强制VLM先verbalize spatial reasoning再做decision。

### 2.6 Termination (Figure 4)

单独的VLM call判断是否stop：

```
Look at this image. Are you within 1 meter of {object_name}?
Answer with just "STOP" or "CONTINUE".
```

**为什么separate call而不是同一个call里return stop？** 作者给两个理由：
1. Annotated arrows + circles增加image noise，降低stop判断准确性
2. Task interference：action selection和stop judgment任务不同，分开更clean

**Stop decision rule**：连续两次VLM call stop才真的terminate。第一次stop后，关闭navigability和explore bias，让agent原地（小范围）再观察。这避免了过早stop。

## 3. Experimental Analysis

### 3.1 ObjectNav Results (Table 1)

| Method | SR | SPL |
|---|---|---|
| **VLMnav (Ours)** | **50.4%** | **0.210** |
| Ours w/o nav | 33.2% | 0.136 |
| Prompt Only | 29.8% | 0.107 |
| PIVOT | 24.6% | 0.106 |
| Ours w/o sliding | 12.9% | 0.063 |

**Important deltas**:
- VLMnav vs PIVOT: **+25.8% SR, +0.104 SPL** (主要贡献来自action proposal的exploration-aware sampling)
- VLMnav vs Ours w/o nav: **+17.2% SR** (这是Navigability + Action Proposer模块的纯贡献)
- VLMnav vs Prompt Only: **+20.6% SR** (visual annotation的作用)
- VLMnav vs Ours w/o sliding: **-37.5% SR** (说明agent实际有collisions被slide救了)

**Insight**: Ours w/o nav 比 Prompt Only 仅高3.4%，说明**只是均匀spacing的visual annotation**只带来微小提升，真正work的是navigability-aware action sampling。

### 3.2 GOAT Results (Table 2)

| Method | SR | SPL | Image SR | Object SR | Desc SR |
|---|---|---|---|---|---|
| **VLMnav** | **16.3%** | **0.066** | 14.3% | 20.5% | 13.4% |
| Ours w/o nav | 11.8% | 0.054 | 7.8% | 16.5% | 10.2% |
| Prompt Only | 11.3% | 0.037 | 7.7% | 15.6% | 10.1% |
| PIVOT | 8.3% | 0.038 | 7.0% | 11.3% | 5.9% |

GOAT harder than ObjectNav (16.3% vs 50.4%) because:
- 多任务序列 (5-10 subtasks per episode)
- 3种goal modalities (image, object name, description)
- Description goal最难 (13.4%) 因为需要fine-grained spatial-semantic reasoning

**Image SR vs Description SR**：VLMnav在image goal上比description goal好(14.3 vs 13.4)。这intuitively合理 - VLM可以directly match两个images的visual similarity，而description需要更多reasoning。

### 3.3 Comparison with Specialized SOTA (Table 3)

| Method | SR | SPL | Type |
|---|---|---|---|
| SenseAct-NN | 29.5% | 0.113 | RL trained |
| Modular GOAT | 24.9% | 0.172 | Modular + low-level policy |
| Ours w/ sliding | 16.3% | 0.066 | Zero-shot VLM |
| Ours (no sliding) | 6.9% | 0.049 | Zero-shot VLM |

**Key observation**: Specialized systems still dominate, but零代价zero-shot仍能reach ~50% of trained system performance。考虑到VLM capabilities每年指数增长，这种zero-shot baseline极有scalability。

### 3.4 FOV Scaling Study (Figure 6)

| FOV | SR | SPL |
|---|---|---|
| 82° | ~35% | ~0.13 |
| 100° | ~42% | ~0.17 |
| 115° | ~48% | ~0.19 |
| 131° | 50.4% | 0.210 |

**Strong positive scaling**：FOV增宽50% (82→131)，SR增加~15个百分点。这与人类直觉吻合 - 更宽视野意味着更少local minima，更少需要turn-around。

这也suggests：宽FOV sensor对VLM-based navigation是关键，可能是未来robot hardware design consideration。

### 3.5 History Length Ablation (Table 4)

| History | SR | SPL |
|---|---|---|
| No history | 46.8% | 0.193 |
| 5 | 42.7% | 0.180 |
| 10 | 45.4% | 0.196 |
| 15 | 40.4% | 0.170 |

**Counter-intuitive result**: 增加history不仅没帮助，反而轻微hurt performance。这与普通ML中的intuition相反。

可能的解释：
1. Naive concatenation: 只是把过去K步的image和action串起来，VLM很难从中extract有用spatial memory
2. Attention dilution: 更多images分摊VLM的attention，当前observation被under-weighted
3. VLM天生不擅长temporal reasoning

这指向一个**open research direction**：如何给VLM提供有效的"spatial memory"。Voxel map是low-level memory，但VLM读不到它。一个可能的方向是把voxel map也render成image给VLM看。

### 3.6 Depth Quality Ablation (Table 5)

| Method | SR | SPL |
|---|---|---|
| GT Depth | 50.4% | 0.210 |
| Segformer [floor segmentation] | 47.2% | 0.183 |
| ZoeDepth [metric depth estimation] | 39.1% | 0.161 |

**Surprising finding**: 用Segformer检测floor然后estimate距离（粗略），只比GT depth低3.2%。但ZoeDepth（learned metric depth）差11.3%。

**Why Segformer better than ZoeDepth here?** 我推测：
- Floor segmentation是binary task（floor vs not-floor），VLM/Segformer在这任务上非常robust
- ZoeDepth estimation有continuous error，对小obstacle容易误判
- Segformer的"像素数 × 系数"虽然粗糙但对navigability mask足够

这个结果很有practical value：**production deployment不需要LiDAR或active depth sensor，只需RGB + floor segmentation**。

### 3.7 Failure Mode Analysis

作者观察到：**13.9%的episodes中VLM在距离target 1-1.5m处premature stop**。这个failure pattern暴露了VLM的fundamental limitation：缺乏fine-grained metric spatial awareness。这与"Vision Language Models are Blind" (Rahmanzadehgervi et al. 2024, https://arxiv.org/abs/2407.06581) 和 "Does Spatial Cognition Emerge in Frontier Models" (Ramakrishnan et al. 2024, https://arxiv.org/abs/2410.06468) 的发现一致。

## 4. Broader Context & Connections

### 4.1 与PIVOT的关键差异

| 维度 | PIVOT | VLMnav |
|---|---|---|
| Action sampling | Iterative Gaussian refinement | One-shot from voxel map |
| VLM calls per action | Many (iterative refinement) | 1 (+1 for stop) |
| Depth usage | No | Yes (via Navigability) |
| Exploration guidance | No explicit bias | Explore bias via voxel map |
| Latency | High | Low |
| Action space expressivity | Limited by Gaussian | Polar coordinates with full coverage |

PIVOT的iterative refinement本质是**用VLM的logits作为目标函数，做iterative optimization**。这个思路elegant但latency高。VLMnav走另一条路：**让geometry先做大部分工作，VLM只做最后的semantic judgment**。

### 4.2 与Modular Approaches (e.g., Modular GOAT)的差异

Modular GOAT (https://arxiv.org/abs/2311.06430) 用VLM做object detection + semantic mapping + low-level policy。性能好但需要多个component training。

VLMnav的philosophical difference：**don't train anything**。直接用off-the-shelf VLM作为policy。Trade-off是性能低一些，但generalize到任何task。

### 4.3 与RL Approaches (e.g., SenseAct-NN)的差异

SenseAct-NN是RL trained policy，需要大量interaction data。性能最好但specific to training distribution。

VLMnav的pro：零训练数据cons，性能比trained policy低~13%。

### 4.4 与Vision-Language-Action Models (OpenVLA, RT-2)的差异

OpenVLA (https://arxiv.org/abs/2406.09246) 和RT-2 (https://arxiv.org/abs/2307.15818) fine-tune VLM with robot data。这些方法performance强但：
1. 需要robot data
2. 限定了action space
3. Fine-tuning可能破坏VLM的general knowledge

VLMnav保留VLM的full generality，task-agnostic。

### 4.5 与End-to-End Learning (ViNG, GNM, ViNT)的对比

ViNT (https://arxiv.org/abs/2306.14846) 和ViNG (https://arxiv.org/abs/2104.05859) 训练navigation foundation models。这些方法用大量navigation data train，robust但需要data。

VLMnav的philosophy是：**VLM本身已经是navigation foundation model的隐式形式**，只要prompting方法对就能unlock这个能力。

### 4.6 与Frontier-Based Exploration的connection

VLMnav的explore bias本质上是**Frontier-based exploration (Yamauchi 1997)**的prompting版本。经典FBE在occupancy grid上找explored-unexplored boundary，drive到那里。VLMnav把这个heuristic encode到action proposal里，让VLM隐式执行FBE。

参考：Topiwala et al. 2018: https://arxiv.org/abs/1806.03581

### 4.7 Visual Prompting lineage

整个visual prompting lineage:
1. Red circle prompting (Shtedritski et al. 2023): https://arxiv.org/abs/2304.06712
2. Set-of-Mark (Yang et al. 2023): https://arxiv.org/abs/2310.11441
3. VisualWebArena (Koh et al. 2024): https://arxiv.org/abs/2401.13649
4. CoNVOI (Sathyamoorthy et al. 2024): https://arxiv.org/abs/2403.15637
5. PIVOT (Nasiriany et al. 2024): https://arxiv.org/abs/2402.07872
6. **VLMnav (this paper)**

VLMnav在这个lineage里的位置：第一次把visual prompting真正用于**embodied navigation的end-to-end action selection**（之前CoNVOI做的是path planning但还需要low-level controller，PIVOT做的是single-step但是iterative）。

## 5. Limitations & Future Directions

### 5.1 Stated Limitations

1. **Sliding dependency**: 关掉sliding参数，SR从50.4%暴跌到12.9%。这意味着navigability mask没有考虑robot size，导致实际碰撞。
2. **Premature stopping**: 13.9%的runs在1-1.5m处过早stop。VLM缺乏fine-grained distance judgment。
3. **Performance gap**: 比specialized systems差2-3倍SR。

### 5.2 Implicit Limitations

1. **VLM latency**: Gemini Flash每次call可能1-3秒，对real-time navigation太慢
2. **VLM cost**: 大规模deployment的API成本可能高
3. **No real-world experiments**: 全部在simulator (Habitat-Matterport)
4. **No fine-grained control**: action space只有forward + rotate，没有strafe，没法做tight maneuver
5. **Memory**: 没有真正的spatial memory，只有2m范围的voxel map

### 5.3 Future Directions I'd speculate

1. **VLM with voxel map input**: 把top-down voxel map也render给VLM看，让VLM有真正的spatial memory
2. **Hierarchical prompting**: high-level VLM做planning，low-level VLM做action selection，可能解决history问题
3. **Action value estimation**: 让VLM不直接选action，而是estimate Q-value for each action，再argmax。可能解决premature stop
4. **Fine-tuning with navigation traces**: zero-shot是pro，但如果有少量数据，VLM-based imitation learning可能大幅提升
5. **Active depth recovery**: 利用VLM semantic understanding推断depth，而不是依赖depth sensor。这比ZoeDepth可能更好，因为VLM有object size priors

## 6. Intuition Building Summary

如果让我提炼这篇paper的**核心intuition**：

> **VLMs already "know" how to navigate - they just need to be asked the right question.**

具体decompose：

1. **VLM knows semantics**: 它能识别sofa, chair, bed在哪
2. **VLM knows spatial layout**: 它能describe "there's a hallway on the left, a wall in front"
3. **VLM doesn't know continuous coordinates**: 它不擅长output "go to (1.2, -0.5, 0.3)"
4. **Solution**: 把action selection从continuous regression → discrete classification
5. **Bonus**: 用geometry (navigability + explore bias) 来constrain这个classification问题，让VLM的"先验"更容易align到physical feasibility

整个pipeline其实是：
**Geometry constrains → VLM discriminates → Agent executes**

Geometry提供scaffolding，VLM提供semantic judgment，两者分工。这比纯geometry (FBE) 和纯VLM (naive prompt) 都好。

这与Andrej你的"Software 2.0/3.0"思想非常aligned - 不是取代classical methods，而是让neural model和classical method协作，neural部分处理semantic reasoning，classical部分处理metric precision。

## 7. Final Thoughts

这篇paper是**VLM-as-policy的early proof-of-concept**。Performance不是best，但展示了zero-shot的潜力。它真正贡献的是一个**general prompting strategy**：用visual annotation把spatial decision problem转成multiple-choice QA。这个framework可以extend到：
- Manipulation (pick which grasp)
- Multi-agent coordination (pick which agent goes where)
- Web navigation (already shown by VisualWebArena)

未来VLM能力提升后，这种approach可能会surpass specialized systems。作者自己在conclusion里也hypothesize这一点。Whether or not that comes true，这个paper提供了一个clean的baseline和design study，值得follow-up。

参考链接汇总：
- Project page: https://jirl-upenn.github.io/VLMnav/
- PIVOT (Nasiriany et al. 2024): https://arxiv.org/abs/2402.07872
- Set-of-Mark (Yang et al. 2023): https://arxiv.org/abs/2310.11441
- GOAT-Bench (Khanna et al. 2024): https://arxiv.org/abs/2406.18209
- Modular GOAT (Chang et al. 2023): https://arxiv.org/abs/2311.06430
- ObjectNav (Batra et al. 2020): https://arxiv.org/abs/2006.13171
- Habitat-Matterport: https://arxiv.org/abs/2210.05633
- CoNVOI (Sathyamoorthy et al. 2024): https://arxiv.org/abs/2403.15637
- Vision Language Models are Blind: https://arxiv.org/abs/2407.06581
- Frontier-based exploration: https://arxiv.org/abs/1806.03581
- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- Zero-shot CoT: https://arxiv.org/abs/2205.09121
- ViNT foundation model: https://arxiv.org/abs/2306.14846
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
