---
source_pdf: RoboGen.pdf
paper_sha256: 4146b2443205d6e4f8137e14fb20a91165da9a70c12306a300bc71fcae53b022
processed_at: '2026-08-12T01:01:54-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboGen 用人话讲

好，我重新用大白话讲一遍，但还是会保留技术细节，因为细节才是 build intuition 的关键。

---

## 一句话说清楚

RoboGen 干的事情就是：**让 GPT-4 当 "导演"，让 simulator 当 "片场"，让 robot 当 "演员"，自动拍无数部 "机器人技能教学片"。**

GPT-4 负责：想任务、搭场景、写剧本（reward function）。
Simulator 负责：提供物理世界、跑 RL 训练。
Robot 负责：在 sim 里学技能。

整个 loop 可以无限跑下去，源源不断产出 skill demonstration。

---

## 为什么这个 idea 现在才 work

要理解 RoboGen，得先理解之前 robot learning 的痛点。

### 痛点 1：Simulation data 很便宜，但搭场景很贵

在 simulation 里训 robot，exploration cost 几乎为零，可以 parallel 跑成千上万个 env。问题是——你得先搭场景。

搭一个场景需要：
- 想一个有意义的 task
- 找到合适的 3D assets（物体模型）
- 把物体摆到合理的位置
- 写 reward function
- 设置初始状态

一个 task 搭下来可能要一个 PhD 学生好几天。你想 scale 到 1000 个 task？那得一个团队干半年。

### 痛点 2：LLM 直接输出 action 不靠谱

之前很多人试过让 LLM 直接输出 robot action 或者 policy code。代表 work：
- **Code as Policies** (Liang et al., 2022): LLM 写 Python 代码控制 robot
- **VoxPoser** (Huang et al., 2023): LLM 生成 3D value map引导 robot
- **RT-2** (Brohan et al., 2023): VLM 直接输出 tokenized action

这些 work 在简单 task 上还行（pick-and-place、推东西），但一到 contact-rich 的 task 就不行了。原因很简单：**LLM 的训练数据里没有 physical dynamics**。它不知道拧一个 knob 需要多大的 torque，不知道 rolling dough 的时候面团会怎么变形，不知道 quadruped 跳起来的时候四条腿应该怎么协调。

LLM 擅长的是 semantics 和 common sense：
- 知道 microwave 用来 heat food
- 知道 drawer 可以 store things
- 知道 oven 通常放在地上
- 知道把 book 放进 drawer 需要 drawer 比 book 大

RoboGen 的 insight 就是：**extract LLM 擅长的东西，把不擅长的交给 physics simulator。**

参考：
- Code as Policies: https://arxiv.org/abs/2209.07753
- VoxPoser: https://arxiv.org/abs/2307.05973
- RT-2: https://arxiv.org/abs/2307.15818

---

## 整个 pipeline 走一遍

我用 paper 里的 "Throw Trash Away" 这个例子，从头走一遍 pipeline，你就明白了。

### Step 1: Task Proposal

系统先从 object pool 里随机抽一个东西，比如 TrashCan（来自 PartNetMobility dataset）。

GPT-4 拿到的 input：
```
对象：TrashCan
articulation tree: joint_0 (revolute) 连接 link_0 和 link_1
semantics: link_0 = hinge door, link_1 = trashcan_body
```

GPT-4 输出：
```
Task Name: Throw Trash Away
Description: The robotic arm places an item of trash inside the trash can
Additional Objects: A pile of trash
Links: link_0 (lid, 需要打开)
Joints: joint_0 (控制 lid 开合)
```

就这么简单。GPT-4 看了 trashcan 的结构，用 common sense 想出了 "扔垃圾" 这个 task，还自动推理出需要额外的 "trash" 对象，以及要操作 link_0 / joint_0。

PartNetMobility: https://sapien.ucsd.edu/browse

### Step 2: Scene Generation

这一步要搭出完整的 3D 场景。分成几个小步：

**2a. 找 assets**

GPT-4 说需要 TrashCan 和 Trash。但光有这两个东西场景太空了，不像 real world。于是再问 GPT-4："这个场景里还有什么东西是 semantically relevant 的？"

GPT-4 回答：broom、dustpan、recycling bin、soda can。

然后去 Objaverse（80万个 3D 模型的数据库）里 search。用 Sentence-BERT 做 text embedding retrieval，取 top-10 candidate。

但 Objaverse 里的东西很杂（很多不是 household object），所以再用 Gemini-Pro（VLM）看一眼 retrieved object 的 render image，生成 caption，然后喂回 GPT-4 判断 "这个东西适合这个 task 吗"。

Objaverse: https://objaverse.allenai.org/
Sentence-BERT: https://arxiv.org/abs/1908.10084

**2b. 修正大小**

Objaverse 里的 asset 大小经常离谱。比如一个 toilet 可能只有 0.2m 高。

GPT-4 根据 common sense 修正：
- trashcan: 0.6m（合理）
- trash: 0.05m（之前 0.1m 太大了，塞不进去）

还会考虑 relative size：drawer 要比 book 大，cup 要比 faucet 小。

**2c. 初始 joint 状态**

Articulated object 的 joint 需要设对初始状态。

对 "Throw Trash Away"：robot 要学打开 lid，所以 lid 初始应该是 closed 的。

GPT-4 输出：joint_0 = 0（lower limit = closed）

Paper 里的 convention：0 = lower limit = natural state（closed/unpushed），1 = upper limit = open/pushed。

**2d. 空间布局**

GPT-4 决定每个物体放哪：
- TrashCan: (1.5, 1.5, 0) world coordinate（在地上）
- Trash: (0.5, 0.5, 0) table coordinate（在桌上）

Robot 在 (1, 1, 0)，table 在 (0, 0, 0)，所以 trashcan 放 (1.5, 1.5) 避免碰撞。

如果检测到 collision，沿 collision normal 反方向推 center of mass 来 resolve。

### Step 3: Training Supervision Generation

**3a. Task decomposition**

GPT-4 把 "Throw Trash Away" 拆成 7 个 sub-step：
1. grasp trash can lid
2. open trash can lid
3. grasp trash
4. put trash into trash can
5. release trash
6. grasp lid again
7. close lid

**3b. Algorithm selection**

对每个 sub-step，GPT-4 从 3 个 algorithm 里选：

- **Motion planning primitive**：适合 grasp / approach / release。这些动作需要一个 collision-free path 到目标位置，用 BIT* 算法。
- **RL (SAC)**：适合需要 continuous interaction 的动作，比如 open lid（需要接触 + 旋转）、close lid。
- **Trajectory optimization**：适合 soft-body 的 fine-grained shaping（比如把 dough 捏成 baguette 形状）。

对 "Throw Trash Away"：
- sub-step 1 (grasp lid): motion planning primitive
- sub-step 2 (open lid): RL（因为需要接触 + 旋转 joint）
- sub-step 3 (grasp trash): motion planning primitive
- sub-step 4 (put trash in): RL（需要把 trash 移到 trashcan 内部）
- sub-step 5 (release): motion planning primitive
- sub-step 6 (grasp lid): motion planning primitive
- sub-step 7 (close lid): RL

**为什么不能全用 RL？** Figure 5 的 ablation 显示，12 个 articulated object task 里，纯 RL 大部分完全失败。原因：RL 从零学 "grasp 一个东西" 的 sample efficiency 极差。Grasping 这个事情用 motion planning 几秒就能解出来，RL 可能要几百万 step 还学不好。

**为什么不能全用 motion planning？** 因为很多动作（旋转 knob、开 door）需要 contact-rich 的 continuous interaction，motion planning 只能做 kinematic path planning，处理不了接触后的 force interaction。

**3c. Reward function generation**

对需要 RL 的 sub-step，GPT-4 写 Python reward function。Paper 给了 3 个 in-context examples 来教 GPT-4 怎么写。

GPT-4 可以用一组 simulator API：
- `get_position(obj)` → 物体质心坐标 [x, y, z]
- `get_joint_state(obj, joint)` → joint angle 值
- `get_joint_limit(obj, joint)` → (lower, upper) limit
- `get_link_state(obj, link)` → link 质心坐标
- `get_bounding_box(obj)` → AABB 的 min/max corner
- `in_bbox(pos, min, max)` → bool，pos 是否在 AABB 内
- `get_eef_pos()` → end-effector 位置
- `gripper_close_to_object(obj)` → bool

Sub-step 2 (open lid) 的 reward function 长这样：

```python
def _compute_reward(self):
    # Dense shaping: 让 end-effector 靠近 lid
    eef_pos = get_eef_pos(self)[0]
    lid_pos = get_link_state(self, "TrashCan", "link_0")
    reward_near = -np.linalg.norm(eef_pos - lid_pos)
    
    # Task reward: 让 joint angle 趋向 upper limit (fully open)
    joint_angle = get_joint_state(self, "TrashCan", "joint_0")
    joint_low, joint_high = get_joint_limit(self, "TrashCan", "joint_0")
    target = joint_high
    diff = np.abs(joint_angle - target)
    reward_joint = -diff
    
    # 合起来
    reward = reward_near + 5 * reward_joint
    
    # Success condition
    success = diff < 0.1 * (joint_high - joint_low)
    return reward, success
```

这个 reward 的 intuition 很清楚：
- `reward_near` 是 dense shaping signal，让 robot 知道 "往 lid 靠近"
- `reward_joint` 是真正的 task signal，让 robot 知道 "把 joint 转到 open 位置"
- 权重 5 让 task reward 占主导
- success 用 joint angle 的 relative distance 判断（10% of range）

Sub-step 4 (put trash in) 的 reward 更复杂一点：

```python
def _compute_reward(self):
    trash_pos = get_position(self, "Trash")
    eef_pos = get_eef_pos(self)[0]
    reward_near = -np.linalg.norm(eef_pos - trash_pos)
    
    # 拿 trashcan body 的 bounding box
    min_aabb, max_aabb = get_bounding_box_link(self, "TrashCan", "link_1")
    
    # 缩小一点 bbox（避免边界 case）
    diff = max_aabb - min_aabb
    min_aabb = min_aabb + 0.05 * diff
    max_aabb = max_aabb - 0.05 * diff
    center = (max_aabb + min_aabb) / 2
    
    # trash 在 trashcan 里 → +1
    reward_in = 0
    if in_bbox(self, trash_pos, min_aabb, max_aabb):
        reward_in += 1
    
    # Dense signal: trash 靠近 trashcan 中心
    reward_reaching = -np.linalg.norm(center - trash_pos)
    
    success = in_bbox(self, trash_pos, min_aabb, max_aabb)
    reward = 5 * reward_in + reward_reaching + reward_near
    return reward, success
```

注意这里的 reward 结构：`5 * reward_in + reward_reaching + reward_near`。`reward_in` 是 sparse 的 binary signal（trash 在不在 trashcan 里），权重最大；`reward_reaching` 是 dense 的距离 signal，引导 robot 往 trashcan 方向移；`reward_near` 让 eef 跟着 trash。

这种 "sparse big reward + dense shaping" 的 pattern 在 RL 里非常常见，GPT-4 通过 in-context examples 学会了这个 pattern。

### Step 4: Skill Learning

有了 scene + decomposition + reward，就开始训了。

**RL 的训练细节：**
- Algorithm: SAC (Soft Actor-Critic)
- Network: MLP [256, 256, 256] for policy 和 Q
- Learning rate: 3e-4
- 1M env steps per sub-task
- Horizon: 100 steps, frame skip 2（所以一个 episode 实际跑 50 个 action）
- Action space: 6D = 3D translation (delta 或 target) + 3D rotation (delta axis-angle)

SAC 的核心 idea 是 maximum entropy RL：不仅要 maximize reward，还要 maximize policy 的 entropy（鼓励 exploration）。

Objective:
$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$$

- $\gamma$: discount factor，通常 0.99，表示 future reward 的折扣
- $r_t$: step $t$ 的 reward
- $\alpha$: temperature，控制 exploration 强度，越大越 random
- $\mathcal{H}(\pi(\cdot|s_t))$: policy distribution 的 entropy
- 整个 objective 是 reward + entropy 的加权和

SAC 用了 4 个 network：
- Policy network $\pi_\phi(a|s)$：输出 action distribution
- Two Q networks $Q_{\theta_1}, Q_{\theta_2}$：估计 Q-value（double Q trick 降 variance）
- Target Q networks $Q_{\bar{\theta}_1}, Q_{\bar{\theta}_2}$：soft update 的 target

Q loss:
$$L(\theta_i) = \mathbb{E}\left[\left(Q_{\theta_i}(s,a) - y\right)^2\right]$$
$$y = r + \gamma \left(\min_{j=1,2} Q_{\bar{\theta}_j}(s', a') - \alpha \log \pi_\phi(a'|s')\right)$$

- $y$: target value
- $r$: reward
- $\gamma$: discount
- $\min_{j} Q_{\bar{\theta}_j}$: 取两个 Q network 的较小值（clipped double Q，overestimation bias 小）
- $\alpha \log \pi_\phi(a'|s')$: entropy term
- $a' \sim \pi_\phi(\cdot|s')$: next action 从 policy sample

Policy loss:
$$L(\phi) = \mathbb{E}\left[\alpha \log \pi_\phi(a|s) - Q_{\theta}(s, a)\right]$$
- 要 minimize 这个 loss → maximize $Q - \alpha \log \pi$ → reward 高 + entropy 高

SAC paper: https://arxiv.org/abs/1801.01290

**Motion planning 的细节：**

用 BIT* (Batch Informed Trees)。BIT* 是 sampling-based optimal motion planner，核心 idea 是用 heuristic-guided batch sampling 渐近搜索 implicit random geometric graph。

对 grasping primitive：
1. 在目标 object surface 上随机 sample 一个 point $p$
2. 计算 $p$ 处的 surface normal $\hat{n}$
3. Gripper pose: y 轴对齐 $\hat{n}$，位置在 $p + 0.03 \cdot \hat{n}$（pre-contact pose，离 surface 3cm）
4. BIT* 找一条从当前 gripper pose 到 pre-contact pose 的 collision-free path
5. 沿 path 移动到 pre-contact pose
6. 沿 $\hat{n}$ 方向继续移动直到 contact

BIT* 的 cost function:
$$g_T(x) = c(x_{start}, x) + h(x, x_{goal})$$
- $c(x_{start}, x)$: 从起点到 $x$ 的实际 cost（已经走过的 path length）
- $h(x, x_{goal})$: heuristic，$x$ 到 goal 的估计 cost（Euclidean distance，admissible）
- $g_T(x)$: total estimated cost through $x$

BIT* 维护一个 priority queue 按 $g_T$ 排序，每次 expand $g_T$ 最小的 node。用 batch sampling：一次 sample 一批 random configuration，build 一个 implicit graph，然后在 graph 上做 A*-like search。

BIT* paper: https://ieeexplore.ieee.org/document/7139620
OMPL (Open Motion Planning Library): https://ompl.kavrakilab.org/

**Trajectory optimization (soft body) 的细节：**

对 soft body manipulation（比如把 dough 捏成 baguette），用 gradient-based trajectory optimization。

Cost function 是 Earth Mover's Distance (EMD) between current shape 和 target shape:
$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x,y) \sim \gamma} \|x - y\|$$

- $P$: current soft body particle distribution（一个 point cloud）
- $Q$: target shape particle distribution（另一个 point cloud）
- $\Pi(P, Q)$: 所有 coupling（joint distribution with marginal $P$ and $Q$）
- $\gamma$: transport plan，$\gamma(x, y)$ 表示从 $x$ 搬多少 mass 到 $y$
- $\|x - y\|$: transport cost
- $W_1$: optimal transport cost

为什么用 EMD 而不是 Chamfer distance？

Chamfer distance:
$$D_{Chamfer}(P, Q) = \sum_{p \in P} \min_{q \in Q} \|p - q\| + \sum_{q \in Q} \min_{p \in P} \|q - p\|$$

Chamfer 只看每个点到对方最近点的距离，对 density 不敏感。比如 $P$ 有 1000 个点集中在某个区域，$Q$ 有 10 个点散开，Chamfer 可能很小（每个 $P$ 点都能找到一个近的 $Q$ 点），但 shape 其实差很远。

EMD 考虑 mass transport，所以 density mismatch 会惩罚。

Optimization 用 Adam:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_{t+1} = \theta_t - \eta \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}$$

- $g_t$: gradient at step $t$
- $m_t$: first moment (momentum)
- $v_t$: second moment (adaptive learning rate per parameter)
- $\beta_1 = 0.9, \beta_2 = 0.999$: decay rates
- $\eta = 0.05$: learning rate
- $\epsilon = 10^{-8}$: numerical stability
- $(1 - \beta_1^t), (1 - \beta_2^t)$: bias correction（因为 $m_0 = v_0 = 0$，初期 estimate 有 bias）

需要 differentiable simulation（用 Genesis）才能 backpropagate gradient through physics。300 gradient steps，horizon 150-200。

Adam paper: https://arxiv.org/abs/1412.6980

**Sequential sub-task learning:**

长 horizon task 的 7 个 sub-step 是 sequential 学的。每个 sub-step 跑 N=8 次，取 reward 最高的那次的 end state 作为下一个 sub-step 的 initial state。

这个设计简单但 fragile：如果某个 sub-step 8 次都失败了，整个 chain 就断了。但实践中 77.4% 的成功率说明大部分时候能跑通。

---

## Soft body 的 asset 怎么来

Soft body manipulation 需要 target shape 的 mesh（比如 "把 dough 捏成 baguette" 需要一个 baguette 的 3D model）。

Objaverse 里不一定有合适的，所以 RoboGen 用了 generation pipeline：

1. GPT-4 生成 target shape 的 text description: "a baguette, no background, top-view"
2. **Midjourney** 做 text-to-image（生成白底俯视图）
3. **Zero-1-to-3** 做 image-to-3D mesh
4. **DMTet** (Deep Marching Tetrahedra) 做 mesh refinement

Zero-1-to-3 的核心 idea：用 large-scale 2D diffusion model 的 prior 来做 novel view synthesis，然后从 multi-view 重建 3D。给定一张 image，生成其他视角的 image，然后用 NeRF 或 mesh reconstruction。

DMTet 用 tetrahedral grid 来 represent 3D shape，可以 hybrid 做 SDF + mesh representation，支持 high-resolution 3D synthesis。

Zero-1-to-3: https://arxiv.org/abs/2303.11328
DMTet: https://arxiv.org/abs/2111.13209
Midjourney: https://www.midjourney.com/

---

## 实验结果讲讲

### Task Diversity (Table 1)

RoboGen 生成了 106 个 task，和几个人工 benchmark 比 diversity：

| | RoboGen | Behavior-100 | RLBench | MetaWorld | ManiSkill2 | GenSim |
|---|---------|-------------|---------|-----------|------------|--------|
| Self-BLEU ↓ | **0.284** | 0.299 | 0.317 | 0.322 | 0.674 | 0.378 |
| SentenceBert sim ↓ | **0.165** | 0.210 | 0.200 | 0.263 | 0.194 | 0.288 |
| ViT image sim ↓ | **0.193** | 0.389 | 0.375 | 0.517 | 0.332 | 0.717 |
| CLIP image sim ↓ | **0.762** | 0.833 | 0.864 | 0.867 | 0.828 | 0.932 |

**Self-BLEU** 是 text generation 的 diversity metric。对一组 generated text，算每个 text 和其他 text 的 BLEU score 的平均。BLEU 测 n-gram overlap，越高说明越相似。Self-BLEU 低 = text 之间 overlap 少 = diverse。

公式（对 $N$ 个 generated text）：
$$\text{Self-BLEU} = \frac{1}{N} \sum_{i=1}^{N} \text{BLEU}(t_i, \{t_j\}_{j \neq i})$$

- $t_i$: 第 $i$ 个 generated task description
- $\text{BLEU}(t_i, \{t_j\})$: 把 $t_i$ 当 candidate，其他所有 $t_j$ 当 reference 算的 BLEU
- Self-BLEU: 平均的 "每个 task 和其他 task 的相似度"

**Embedding similarity** 用预训练 model（SentenceBert / ViT / CLIP）把 text/image encode 成 vector，然后算 pairwise cosine similarity 的平均。低 = diverse。

$$\text{Sim} = \frac{2}{N(N-1)} \sum_{i < j} \cos(\mathbf{e}_i, \mathbf{e}_j)$$

- $\mathbf{e}_i$: 第 $i$ 个 task 的 embedding
- $\cos(\mathbf{e}_i, \mathbf{e}_j)$: cosine similarity
- Sim: 所有 pairwise similarity 的平均

RoboGen 在所有 metric 上都是最低的（最 diverse），包括比 concurrent work GenSim 好很多。原因：
1. Objaverse 有 800k assets，GenSim 只用 Ravens 的小 pool
2. RoboGen 做 articulated object + soft body + locomotion，GenSim 只做 table-top pick-and-place
3. RoboGen 自动 retrieve diverse distractor objects

Self-BLEU paper: https://arxiv.org/abs/1803.04871
CLIP: https://arxiv.org/abs/2103.00020
ViT: https://arxiv.org/abs/2010.11929

### Scene Validity (Figure 4)

用 BLIP-2 score 评估 retrieved object 和 text description 的 alignment。BLIP-2 是 VLM，可以算 image-text matching score。

BLIP-2 score 高 = retrieved object 的 visual appearance 和 text description 匹配。

两个 ablation：
- **w/o object verification**: 不用 VLM 过滤 retrieved object → score 下降，variance 增大
- **w/o size verification**: 不用 GPT-4 修正 size → score 下降最多（默认 size 经常离谱到搞笑）

结论：object verification 和 size verification 都很重要。

BLIP-2: https://arxiv.org/abs/2301.12597

### Skill Learning Success Rate

69 个 task 的平均成功率 77.4%。

分类：
- 50 个 articulated object manipulation: 平均 74.5%
- 7 个 soft body: 60%-100%
- 12 个 locomotion: 40%-100%

 locomotion 的 "Climb up stairs" 只有 40%，因为 contact-rich + 需要精确的 foot placement。

 locomotion 用的是 CEM (Cross-Entropy Method) 而不是 RL。CEM 是 model-based planning：用 ground-truth simulator 作为 dynamics model，optimize action sequence。

CEM 的流程：
1. Initialize action sequence distribution $\mathcal{N}(\mu_0, \sigma_0^2)$
2. Sample $K$ 个 action sequence $\{a_1, ..., a_K\}$
3. 用 simulator roll-out 每个 action sequence，得到 return $\{R_1, ..., R_K\}$
4. Select top-$k$ elite sequences（return 最高的 $k$ 个）
5. Update $\mu, \sigma$ 用 elite 的 mean 和 std
6. Repeat

CEM 不学 policy network，每次都重新 optimize。好处：不需要 training data，sample efficient。坏处：inference 慢（每次都要 sample + roll-out），且不能 transfer 到不同 dynamics（因为用的是 ground-truth model）。

CEM paper: https://www.cs.utexas.edu/~ai-lab/pubs/CEM-nips-04.pdf

### Pure RL vs. Mixed Algorithm (Figure 5)

12 个 articulated object task，如果只用 RL（不用 motion planning primitive），大部分完全失败。

这验证了 algorithm selection 的重要性。Grasping / approaching 这种 "找到一条 collision-free path" 的事情，motion planning 几秒就能解，RL 从零学可能要几百万 step。

### Failure Analysis (Table 8)

155 个 task 里 19 个失败：
- 13 个 scene generation 失败
  - 6 个：asset 功能不支持（比如 printer 没有可移动 tray）
  - 4 个：joint state semantic mapping 错误（GPT-4 不知道 joint=0 是 open 还是 closed）
  - 3 个：precise spatial relationship 找不到匹配 asset（比如 stapler + staples 尺寸不配）
- 6 个 reward generation 失败
  - 2 个：undefined variables
  - 2 个：reward encode 错误 behavior（比如 "fold chair" 的 reward 实际上鼓励 unfold）
  - 2 个：连续运动难以 reward（比如 "knock door" 的 back-and-forth motion）

这些 failure 给了很明确的 future work 方向：
- 用 VLM 验证 joint state mapping
- 把 error message 喂回 LLM 做 self-correction（Eureka 的思路）
- 用 environment feedback 来 verify reward correctness

---

## 我觉得这个 work 最 clever 的地方

### 1. 正确地划分了 LLM 和 simulator 的职责

LLM 做 semantics（task idea, scene layout, reward design），simulator 做 physics（contact, dynamics, control）。这比让 LLM 直接输出 action 靠谱得多。

### 2. Algorithm selection

让 GPT-4 根据子任务类型选择 RL / motion planning / trajectory optimization。这个看起来简单但实际上非常关键。Figure 5 证明纯 RL 在很多 task 上完全不行。把 motion planning 当作 "primitive" 嵌入 RL pipeline 里，避免了 RL 浪费 sample 去学 "怎么 grasp 一个东西" 这种 motion planning 秒解的事情。

### 3. Pipeline 的 modular 设计

每个 component（LLM, VLM, generative model, simulator, RL algorithm）都是可替换的。GPT-4 可以换成更好的 LLM，Gemini-Pro 可以换成更好的 VLM，Genesis 可以换成 MuJoCo 或 Isaac Gym。这保证了 RoboGen 可以随着 foundation model 的进步持续变强。

### 4. Suction cup gripper 的简化

用 suction cup 而不是 parallel jaw gripper，把 grasping 变成 deterministic primitive（只要 approach + attach）。这回避了 grasping 这个 robotics 的 open problem，让 pipeline 更 robust。代价是：不能做需要精细 grasp 的 task（比如 in-hand manipulation）。

---

## 和相关 work 的关系

### Eureka (Ma et al., 2023)

Eureka 也是 LLM 生成 reward function，但：
- Eureka 需要 human 提供 task，RoboGen 自动生成 task
- Eureka 用 evolutionary search + LLM iteration refine reward（generate multiple candidates → evaluate in env → feed results back to LLM → generate better rewards）
- RoboGen 是 single-shot generation（no iteration）

Eureka 的 reward quality 可能更高（因为 iterative refinement），但 RoboGen 更 end-to-end automated。

两者结合是最好的：RoboGen 生成 task + scene + initial reward → Eureka-style iteration refine reward。

Eureka: https://arxiv.org/abs/2310.12931

### GenSim (Wang et al., 2023a)

Concurrent work，也用 LLM 生成 task。但：
- 只做 table-top rigid object pick-and-place
- LLM 直接写 manipulation code script
- Asset pool 小（Ravens dataset）

RoboGen diversity 高很多，且更 general（articulated + soft body + locomotion）。

GenSim: https://arxiv.org/abs/2310.01361

### RT-X / Open-X-Embodiment (Google DeepMind, 2023)

RT-X 是 real robot data 的大规模 dataset。RoboGen 生成 sim data。

互补关系：
- RoboGen: unlimited sim data，但 sim-to-real gap
- RT-X: real data，但 scale 有限且 cost 高

Future: RoboGen 生成 sim data → pre-train policy → RT-X fine-tune → sim-to-real

Open-X-Embodiment: https://robotics-transformer-x.github.io/

### PaLM-E / RT-2

VLA (Vision-Language-Action) model。RoboGen 生成的 data 可以用来 pre-train VLA，特别是 articulated object interaction 这种 real data 稀缺的 task。

PaLM-E: https://palm-e.github.io/
RT-2: https://robotics-transformer2.github.io/

### Language to Reward (Yu et al., 2023b)

也是 LLM → reward，但需要 human 提供 task，且 mapping 是固定 template。RoboGen 自动生成 task，更 general。

LtR: https://arxiv.org/abs/2306.08647

### DreamerV3 / World Models

RoboGen 用 ground-truth physics simulator。另一个方向是 learn a world model 然后 "dream"。

两者可以结合：用 RoboGen 生成的 data 训 world model → world model 里做更多 imagination → generate even more data。

DreamerV3: https://arxiv.org/abs/2301.04104

### Scaling Laws for Robotics

NLP 和 CV 都有 scaling laws（Chinchilla, Kaplan）。Robotics 的 scaling law 是什么样的？

RoboGen 提供了一种 generate data 的方式，可以用来研究：robot policy 的 performance 随 data 量、task diversity、robot embodiment diversity 怎么 scale。

Chinchilla: https://arxiv.org/abs/2203.15556
Kaplan scaling laws: https://arxiv.org/abs/2001.08361

---

## Limitations 和我看到的瓶颈

### 1. Asset quality

Objaverse 的 asset quality 参差不齐。很多 asset 的 annotation 有 noise，geometry 有问题（non-manifold mesh, missing textures），articulation 不对。

Paper 用 Gemini-Pro 做 verification 来 filter，但这是 best-effort，不是彻底解决。

Future direction: 用更好的 text-to-3D generative model（比如 Point-E, Shap-E, 未来的 better models）直接生成高质量 asset。

Point-E: https://arxiv.org/abs/2212.08751
Shap-E: https://arxiv.org/abs/2305.02463

### 2. Reward correctness

6/155 的 reward 有错。对更复杂的 task（multi-step cooking, long-horizon assembly），错误率可能上升。

Eureka 的 iterative refinement 可以帮助：generate reward → evaluate → feed back → regenerate。RoboGen 目前是 single-shot。

### 3. Verification of learned skills

怎么知道学到的 skill 是 "correct" 的？现在靠 human eval（看 video）。这在 scale 下不可持续。

Future: 用 VLM 自动评估 skill video 和 task description 的 alignment。甚至可以用 VLM 做 success detector：给 VLM 看 skill execution 的 video，问 "robot 是否成功完成了 [task description]？"

### 4. Sim-to-real gap

Paper 承认这是 limitation。生成的 skill 在 sim 里 work，但 real robot 有：
- Friction / mass / inertia 不准
- Sensor noise
- Actuator delay
- Visual appearance 不同

需要 domain randomization, system identification, tactile sensing 等。

Domain randomization: https://arxiv.org/abs/1703.06907

### 5. Suction cup simplification

Suction cup 让 grasping 变 trivial。但 real robot 经常用 parallel jaw 或 dexterous hand。扩展到这些 gripper 需要：
- 更复杂的 grasping primitive（force closure analysis）
- 可能需要 grasp generation network（比如 GraspNet, AnyGrasp）

GraspNet: https://arxiv.org/abs/2004.05106
AnyGrasp: https://arxiv.org/abs/2212.08333

### 6. Sequential sub-task learning 的 fragility

N=8 次 retry + best end state 传递。如果某个 sub-task 失败，chain 断掉。

Future: hierarchical RL (/options framework，把 sub-task 当 macro-action)，或者 curriculum learning（从简单 task 到复杂 task）。

Options framework: https://arxiv.org/abs/1606.01460

### 7. Locomotion 用 CEM 而不是 RL

CEM 用 ground-truth simulator 当 dynamics model。这意味着 locomotion skill 是 "planned" 的，不是 "learned" 的。每次 inference 都要重新 optimize，且不能 transfer 到不同 dynamics。

Future: 用 RL + domain randomization 训 locomotion policy，可以 zero-shot transfer。

### 8. Task 的 "interestingness"

现在 task proposal 是 random seeding + LLM extrapolation。生成的 task 可能有很多 trivial variation（"open door" / "close door" / "partially open door"）。

Future: 加一个 "interestingness" scorer，优先 generate 更 challenging / more novel 的 task。可以用 LLM 自己来 score，或者用 learned skill 的 success rate 来 infer difficulty。

---

## 我觉得未来的方向

### 1. Self-improvement loop

现在是 single-pass pipeline。想象一个 closed loop：

```
LLM propose task → generate scene → generate reward → learn skill 
    → VLM evaluate skill quality → feedback to LLM 
    → LLM improve task/reward → re-learn → iterate
```

这其实就是 Eureka 的思路 + RoboGen 的 task generation + VLM verification。

### 2. Foundation policy training

RoboGen 生成的 data 可以用来训一个 generalist robot policy（类似 RT-X 但用 sim data）。

想象：RoboGen 跑几周，生成 100 万个 task + skill demonstration → 训一个 VLA 或 diffusion policy → 得到一个 "foundation policy" → 在 real robot 上 fine-tune。

Diffusion Policy: https://arxiv.org/abs/2303.04137

### 3. Interactive task generation

Robot 在环境里探索，发现 "这个地方我做得不好" → 自动 generate 相关 task 来 practice。

这类似 curriculum learning，但 curriculum 是 robot 自己 generate 的。

### 4. Multi-robot / multi-agent

现在 RoboGen 只用 single robot (Franka arm 或 quadruped)。扩展到 multi-robot collaboration（bimanual, dual-arm, swarm）会大大增加 task diversity 和 complexity。

### 5. Real-world asset scanning

用 iPhone LiDAR 或 NVIDIA Omniverse 扫描 real object → import 到 sim → 让 RoboGen 在 real asset 上生成 task。可以缩短 sim-to-real gap。

NVIDIA Omniverse: https://www.nvidia.com/en-us/omniverse/

### 6. Language-conditioned skill library

RoboGen 生成的每个 skill 都有 natural language description。可以 build 一个 "skill library"：robot 遇到新 task → LLM decompose → 从 library 里 retrieve 对应 skill → 组合执行。

这类似 SayCan (Ahn et al., 2022) 的思路，但 skill library 是自动生成的。

SayCan: https://arxiv.org/abs/2204.01691

---

## 最后的总结

RoboGen 的核心贡献是 **paradigm-level 的**：它展示了 "LLM 设计 + simulator 执行 + robot 学习" 这个 loop 可以 fully automate，generate diverse 的 skill data。

具体的技术细节（reward function pattern, algorithm selection, asset retrieval）都是可替换的 module，未来可以持续改进。

最重要的是它给 community 指了一个方向：**robot learning 的 data scaling 问题，可以通过 generative simulation 来 attack。** 这和 LLM 的 data scaling（web crawling）、CV 的 data scaling（image scraping）是同一个级别的 insight，但之前在 robotics 里一直缺一个 practical 的实现。

Paper 的 project page: https://robogen-ai.github.io/

---

希望这个版本更 readable 了。如果你想深入某个具体 component（比如 reward function 的 prompt engineering 细节、SAC 的实现 trick、或者 sim-to-real 的具体策略），我可以再 drill down。

---

# RoboGen: 深入解析 Generative Simulation 范式

Andrej, 这篇paper 在我看来是 robotics + foundation model 交叉领域里一个范式级的 work，它把 "data scaling for robot learning" 这个问题重新 frame 了一次。我来讲讲我的理解和一堆相关联想。

---

## 1. 核心思想的 intuition

整篇paper的核心 insight 其实很简洁：**foundation models 不擅长 physical dynamics，但擅长 semantics / affordances / common sense，所以应该让它们做擅长的事，把 physical reasoning 交给 simulator。**

之前的 work 比如 Code as Policies (Liang et al., 2022)、VoxPoser (Huang et al., 2023)、RT-2 (Brohan et al., 2023) 都试图让 LLM/VLM 直接输出 low-level actions 或者 policy code。但 LLM 的训练数据里几乎没有 dynamics、actuation、contact-rich interaction 这些东西，所以它们在 physical reasoning 上是先天不足的。

RoboGen 选择把 LLM 当成 "task/scene/reward 的生成器"，让 simulator 当成 "物理引擎 + skill learner"。这其实呼应了 Sutton 在 "The Bitter Lesson" 里的观察——把 computation 放在 search 和 learning 上，而不是 human knowledge 上。LLM 提供的是 human knowledge (semantics), simulator + RL 提供的是 computation (search over actions)。

参考：
- The Bitter Lesson: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Eureka (Ma et al., 2023): https://arxiv.org/abs/2310.12931
- Code as Policies: https://arxiv.org/abs/2209.07753
- VoxPoser: https://arxiv.org/abs/2307.05973

---

## 2. Pipeline 架构详解

RoboGen 是一个 propose-generate-learn 的自循环，4 个 stage：

### Stage A: Task Proposal

输入：robot type + 一个从 PartNetMobility (Xiang et al., 2020) 或 RLBench (James et al., 2020) pool 里 sample 的 object。

具体地，对于 articulated object（比如 microwave），LLM 拿到的 input 包含：
1. object category
2. articulation tree（来自 URDF）
3. link 的 semantic annotations（比如 link 0 是 door，link 1 是 timer knob）
4. 一个 in-context example

LLM 输出的 task 包含 4 个字段：task name、自然语言描述、additional objects needed、relevant joints/links。

这里有个设计选择值得注意：对 articulated object 用 object-based seeding，对 legged locomotion 和 soft-body 用 example-based seeding（11 个预定义 task）。这是因为 locomotion 没有 "object" 可以 seed。

PartNetMobility: https://sapien.ucsd.edu/browse
RLBench: https://sites.google.com/view/rlbench

### Stage B: Scene Generation

这是最复杂的 stage，分成 4 个 sub-component：

**B1. Relevant assets retrieval**
- 从 task proposal 拿到 asset query list
- 再问 GPT-4 多生成一些 semantically relevant distractor objects（比如 "open storage, put toy inside" 会额外生成 mat、lamp、book、chair）
- 用 Sentence-BERT (Reimers & Gurevych, 2019) 在 Objaverse (Deitke et al., 2023, 800k+ 3D assets) 上做 embedding retrieval，取 top-k=10
- 用 Gemini-Pro 做 VLM verification，把 caption + task description + asset description 喂回 GPT-4 判断是否合适

Sentence-BERT: https://arxiv.org/abs/1908.10084
Objaverse: https://objaverse.allenai.org/

**B2. Asset size correction**
- Objaverse 里的 asset 大小经常离谱（比如 toilet 0.2m）
- GPT-4 根据常识和 task 要求调整 size，保证 plausible + relative sizes consistent（比如 drawer 要比 book 大）

**B3. Initial asset configuration**
- articulated object 的 joint angle 需要正确初始化
- 比如 "close the window" → window 初始是 open 状态
- GPT-4 输出 joint angle value（normalized 到 [0, 1]，0=lower limit, 1=upper limit）

**B4. Scene spatial configuration**
- 每个物体的 3D location
- 特殊 spatial relationship（比如 "in safe", "on chopping board"）
- Collision-free placement（如果检测到 collision，沿 collision normal 反方向 push center of mass）

对于 soft-body manipulation，asset 走的是 generation pipeline：
- GPT-4 生成 target shape 的 text description
- Midjourney 做 text-to-image（白底、front view 或 top-down view）
- Zero-1-to-3 (Liu et al., 2023b) 做 image-to-mesh
- DMTet (Shen et al., 2021) 做 mesh refinement

Zero-1-to-3: https://arxiv.org/abs/2303.11328
DMTet: https://shenyuan.com/dmtet

### Stage C: Training Supervision Generation

**C1. Task decomposition**
LLM 把长 horizon task 拆成 sub-tasks，比如 "throw trash away" 拆成：
1. grasp trash can lid
2. open trash can lid
3. grasp trash
4. put trash into trash can
5. release trash
6. grasp lid again
7. close lid

**C2. Algorithm selection**
对每个 sub-task，GPT-4 从 3 个 algorithm 里选一个：

- **Reinforcement Learning (SAC)**：适合 contact-rich、continuous interaction，比如 turning oven knob、legged locomotion
- **Action primitive + motion planning (BIT\*)**：适合 approach/grasp/release 这种需要 collision-free path 的
- **Gradient-based trajectory optimization (Adam)**：适合 soft-body 的 fine-grained shaping（用 differentiable simulation）

这个 design choice 很关键。Figure 5 显示，如果只用 RL，12 个 articulated object manipulation task 里大部分完全失败。因为 RL 从零学 motion planning 类型的事情 sample efficiency 太差。

**C3. Reward function generation**
对 RL sub-task，GPT-4 写 Python reward function。Paper 给了 3 个 in-context examples（refrigerator fetch、oven temperature、box open）。Reward function 用一组 simulator API：
- `get_position(obj_name)`：物体质心位置
- `get_joint_state(obj_name, joint_name)`：joint angle
- `get_joint_limit(obj_name, joint_name)`：(lower, upper) limit
- `get_link_state(obj_name, link_name)`：link 质心位置
- `get_bounding_box(obj_name)`：AABB
- `in_bbox(pos, bbox_min, bbox_max)`：判断 pos 是否在 AABB 内
- `gripper_close_to_object(obj_name)`：gripper 是否接近物体

典型 reward pattern：
```python
reward = reward_near + alpha * reward_task
```
其中 `reward_near = -||eef_pos - target_pos||` 是 dense shaping，`reward_task` 是 task-specific 的 sparse reward（比如 joint angle 距离 target 的 negative L1）。

Soft-body 的 reward 是 Earth Mover's Distance (Wasserstein-1)：
$$W_1(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \mathbb{E}_{(x, y) \sim \gamma} \|x - y\|$$

变量解释：
- $P$: 当前 soft body particle distribution
- $Q$: target shape particle distribution  
- $\Pi(P, Q)$: 所有 marginal 为 $P$ 和 $Q$ 的 joint distribution（couplings）集合
- $\gamma$: transport plan，表示如何把 $P$ 的 mass 搬到 $Q$
- $\|x - y\|$: 搬运 cost
- $W_1$: 最小总搬运 cost

EMD 比 Chamfer distance 在 soft body 上更好，因为它对 particle density 敏感，而 Chamfer 只看最近点距离。

参考 EMD: https://arxiv.org/abs/1701.07875
参考 DiffSkill (Lin et al., 2022): https://arxiv.org/abs/2203.17275

### Stage D: Skill Learning

对长 horizon task，sequential 学习：每个 sub-task 跑 N=8 次，取 reward 最高的 end state 作为下一个 sub-task 的 initial state。

**RL 细节**：SAC (Haarnoja et al., 2018)
- Policy 和 Q network 都是 MLP [256, 256, 256]
- Learning rate: 3e-4
- 1M env steps per sub-task
- Horizon 100, frame skip 2
- Action space: 6D（3D translation + 3D delta-axis-angle rotation）
- 用 suction cup gripper 简化 grasping

SAC 的 objective（maximum entropy RL）：
$$J(\pi) = \mathbb{E}_{(s_t, a_t) \sim \pi, \pi_{env}} \left[ \sum_{t=0}^{T} \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right) \right]$$

变量解释：
- $\pi$: stochastic policy
- $\gamma$: discount factor（通常 0.99）
- $r(s_t, a_t)$: reward
- $\alpha$: temperature，控制 exploration vs exploitation 的 trade-off
- $\mathcal{H}(\pi(\cdot | s_t))$: policy 的 entropy，鼓励 exploration
- $T$: horizon

SAC paper: https://arxiv.org/abs/1801.01290

**Motion planning 细节**：BIT\* (Gammell et al., 2015)
- 在 OMPL (Sucan et al., 2012) 里实现
- 先在 surface 上 sample 一个 point，计算对齐 normal 的 gripper pose
- Pre-contact pose 在 normal 方向上 0.03m 处
- BIT\* 找 collision-free path 到 pre-contact pose
- 然后沿 normal 移动直到 contact

BIT\* 的核心 idea 是用 heuristic-guided batch sampling 来渐近最优地搜索 implicit random geometric graph。Cost function：
$$g_T(x) = c(x_{start}, x) + h(x, x_{goal})$$
- $c(x_{start}, x)$: cost-to-come（从起点到 $x$ 的实际 cost）
- $h(x, x_{goal})$: admissible heuristic（$x$ 到 goal 的估计 cost，通常用 Euclidean）
- $g_T$: total estimated cost

BIT\*: https://ieeexplore.ieee.org/document/7139620
OMPL: https://ompl.kavrakilab.org/

**Trajectory optimization 细节**：
- Adam optimizer (Kingma & Ba, 2014)
- 300 gradient steps
- Learning rate 0.05
- Horizon 150 或 200
- 需要 differentiable simulation（用 Genesis）

Adam update rule：
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t)$$
$$\hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

变量解释：
- $g_t$: gradient at step $t$
- $m_t$: first moment estimate (momentum)
- $v_t$: second moment estimate (uncentered variance)
- $\beta_1, \beta_2$: decay rates (typically 0.9, 0.999)
- $\hat{m}_t, \hat{v}_t$: bias-corrected estimates
- $\eta$: learning rate
- $\epsilon$: numerical stability (1e-8)

Adam: https://arxiv.org/abs/1412.6980
Genesis simulator: paper 里 footnote 1 提到，是作者们自己的 differentiable physics platform。可以参考 Chuang Gan 组的其它 work，比如 FluidLab (Xian et al., 2023b): https://arxiv.org/abs/2303.02346

---

## 3. 实验

### Task Diversity (Table 1)

| Metric | RoboGen | Behavior-100 | RLBench | MetaWorld | ManiSkill2 | GenSim |
|--------|---------|--------------|---------|-----------|------------|--------|
| # Tasks | 106 | 100 | 106 | 50 | 20 | 70 |
| Self-BLEU ↓ | **0.284** | 0.299 | 0.317 | 0.322 | 0.674 | 0.378 |
| SentenceBert sim ↓ | **0.165** | 0.210 | 0.200 | 0.263 | 0.194 | 0.288 |
| ViT image sim ↓ | **0.193** | 0.389 | 0.375 | 0.517 | 0.332 | 0.717 |
| CLIP image sim ↓ | **0.762** | 0.833 | 0.864 | 0.867 | 0.828 | 0.932 |

RoboGen 在所有 diversity metric 上都是最低（最好）。Self-BLEU 测的是文本生成里的 n-gram overlap，越低越 diverse。embedding similarity 用 ViT 和 CLIP 在 rendered scene image 上算，越低说明 visual diversity 越高。

Self-BLEU: https://arxiv.org/abs/1803.04871
CLIP: https://arxiv.org/abs/2103.00020

### Scene Validity (Figure 4)

用 BLIP-2 score (Li et al., 2023b) 评估 retrieved object 和 text description 的 alignment。

BLIP-2: https://arxiv.org/abs/2301.12597

两个 ablation：
- **w/o object verification**: BLIP-2 score 显著下降，variance 大
- **w/o size verification**: BLIP-2 score 下降最剧烈（默认 size 经常离谱）

155 个 task 里 13 个 scene generation 失败，分类：
1. Asset functionality 不支持（比如 printer 没有可移动 tray）
2. Articulated object joint state 的 semantic mapping 错误（比如 joint=0 到底是 open 还是 closed）
3. 极端 precise spatial relationship 找不到 matching asset（比如 stapler + staples）

### Skill Learning Success Rate

总体平均 0.774（69 个 task）。

 articulated object manipulation: 平均 0.745（50 task）
 soft-body: 7 task（0.6-1.0）
 locomotion: 12 task（0.4-1.0）

Figure 5 显示：纯 RL 在 12 个 articulated object task 上大部分完全失败。混合 algorithm（RL + motion planning）显著提升。

### Failure Analysis (Table 8)

19 个 failure case：
- 13 个 scene generation 失败
- 6 个 reward generation 失败（undefined variables、reward encode 错误 behavior、连续 motion 难以 reward）

---

## 4. 我的 Intuition 和 Critique

### 4.1 为什么这个范式 work

我觉得 RoboGen 的成功在于它正确地识别了 LLM 的 capability boundary。LLM 在 physical reasoning 上不行，但在以下几个方面很强：

1. **Object affordances**: LLM 知道 microwave 可以 heat food，drawer 可以 store things
2. **Task decomposition**: LLM 能把 "make coffee" 拆成合理的 sub-steps
3. **Reward shaping**: LLM 能写出 reasonable 的 dense reward（用 -||eef_pos - target|| 这种 pattern）
4. **Common sense scene layout**: LLM 知道 oven 在地上，mug 在桌上

而 simulator + RL 在以下方面强：
1. **Contact-rich manipulation**: 真正的物理交互
2. **Continuous control**: joint torque, end-effector trajectory
3. **Search over action space**: 通过 exploration 找到 solution

把两者结合起来，就 avoid 了各自的 weakness。

### 4.2 与 Eureka 的对比

Eureka (Ma et al., 2023) 也是用 LLM 生成 reward function，但有几个关键区别：
- Eureka 需要 human 提供 task specification（reward template），RoboGen 自动 generate task
- Eureka 用 evolutionary search + LLM iteration 来 refine reward，RoboGen 是 single-shot generation
- Eureka 不做 scene generation 和 task decomposition

RoboGen 更 ambitious，它试图 automate 整个 pipeline。但 Eureka 在 reward quality 上可能更好（因为它有 iterative refinement + environment feedback）。

### 4.3 与 GenSim 的对比

GenSim (Wang et al., 2023a) 是 concurrent work，也用 LLM 生成 task。但 GenSim：
- 只做 table-top rigid object pick-and-place
- LLM 直接写 manipulation code script（不是 reward）
- Asset pool 小（Ravens dataset）

RoboGen 的 diversity 高很多（Table 1 证实），因为：
- 支持 articulated object、soft body、locomotion
- 用 Objaverse（800k assets）而不是小 pool
- 用 reward + RL 而不是直接 code

### 4.4 Scaling 的 bottleneck

RoboGen 声称 "infinite data"，但实际有几个 bottleneck：

1. **Asset quality**: Objaverse 里很多 asset 不适合 robot manipulation（noisy annotation、geometry 问题）
2. **LLM reward 的 correctness**: 6/155 失败率不算高，但对于更复杂的 task（比如 multi-step cooking）可能显著上升
3. **Sim-to-real gap**: Paper 在 limitations 里承认了。生成的 skill 在 sim 里 work，但 transfer 到 real robot 还需要 domain randomization、system identification 等
4. **Verification**: 怎么知道学到的 skill 是 "correct" 的？现在靠 human eval。未来可能需要 VLM-based verification

### 4.5 联想到的相关方向

**1. RT-X / Open-X-Embodiment (Google DeepMind, 2023)**
RoboGen 生成的是 simulation data，RT-X 是 real robot data。两者互补：RoboGen 可以 generate unlimited sim data，RT-X 提供 real world distribution。Future work 可能是 RoboGen 生成 sim data → 用 RT-X fine-tune → sim-to-real。

Open-X-Embodiment: https://robotics-transformer-x.github.io/

**2. PaLM-E / RT-2**
这些是 VLA (Vision-Language-Action) model。RoboGen 生成的 data 可以用来 pre-train VLA，特别是 articulated object interaction 这种 real data 稀缺的 task。

PaLM-E: https://palm-e.github.io/
RT-2: https://robotics-transformer2.github.io/

**3. DreamerV3 / World Models**
RoboGen 用真实 simulator。另一个方向是 learn a world model 然后 dream。两者可以结合：用 RoboGen 生成的 data 训 world model，然后 world model 里做更多 imagination。

DreamerV3: https://arxiv.org/abs/2301.04104

**4. Language to Reward (Yu et al., 2023b)**
这个 work 也是 LLM → reward，但需要 human 提供 task，且 mapping 是固定的 template。RoboGen 更 general。

Language to Reward: https://arxiv.org/abs/2306.08647

**5. Scaling laws for robotics**
OpenAI 的 rubber hand 实验显示 RL 在机器人上需要海量 data。RoboGen 提供了一种 generate data 的方式。问题是：robotics 的 scaling law 是什么样的？Chinchilla-style 的 compute-optimal 训练在 robotics 上如何 apply？

Rubber hand: https://arxiv.org/abs/1910.07113
Chinchilla: https://arxiv.org/abs/2203.15556

**6. Foundation models for physics**
RoboGen 用传统 physics engine。如果未来有 learned physics simulator（比如 learned from YouTube videos），可以和 RoboGen 结合，生成更 photorealistic + physically accurate 的 data。

**7. Self-improvement loop**
现在 RoboGen 是 single-pass。可以想象一个 self-improvement 版本：
- RoboGen 生成 task → 学 skill → 用 VLM 评估 skill quality → feedback 给 LLM 改进 task/reward → 迭代

这其实是 Eureka 的思路 + RoboGen 的 task generation 的结合。

### 4.6 一些 technical nitpicks

1. **Suction cup gripper**: 这是个 simplification。Real robot 用 parallel jaw 或 dexterous hand 会困难很多。Suction cup 让 grasping 变成 deterministic primitive，回避了 grasping 这个 open problem。

2. **CEM for locomotion**: Paper 在 Appendix A.3 提到 locomotion 用 CEM (De Boer et al., 2005) 而不是 RL。CEM 用 ground-truth simulator 作为 dynamics model。这其实是个 model-based planning approach，不是 learning。这意味着 locomotion skill 不能 transfer 到 perturbed dynamics（除非加 domain randomization）。

CEM: https://www.cs.utexas.edu/~ai-lab/pubs/CEM-nips-04.pdf

3. **Sequential sub-task learning**: 用 N=8 次的 best end state 作为下一个 sub-task 的 initial state。这个简单但可能 suboptimal。如果某个 sub-task 失败，整个 chain 就断了。Future work 可以用 curriculum learning 或 hierarchical RL。

4. **Reward function 的 expressiveness**: 现在 reward 都是 hand-crafted pattern（reward_near + alpha * reward_task）。更复杂的 task（比如 making coffee with multiple constraints）可能需要更 expressive 的 reward specification，比如 temporal logic 或者 programmatic reward with state machine。

---

## 5. 总结

RoboGen 是一个 paradigm-shifting 的 work。它把 robot learning 从 "human designs everything" 推向 "LLM designs + simulator executes + robot learns"。Diversity 超过 human-crafted benchmark，pipeline 自动化程度高，failure rate 可接受。

更重要的是它给 community 提供了一个 framework：未来更好的 LLM/VLM/generative model 可以直接 plug-in，持续改进。这是一个 "infrastructure" 类型的 contribution。

Potential future directions：
1. 加入 VLM-based skill verification 和 feedback loop
2. 扩展到 dexterous manipulation、bimanual、mobile manipulation
3. 结合 real robot data 做 sim-to-real
4. 用生成的 data 训 foundation policy（比如 RT-X style）
5. 加入 interactive task generation（robot 主动 ask for clarification）

---

希望这些 detail 和 intuition 对你有帮助。如果你想深入某个 component（比如 reward generation 的 prompt engineering、Genesis simulator 的 differentiable physics、或者 sim-to-real 的具体策略），我可以再展开。
