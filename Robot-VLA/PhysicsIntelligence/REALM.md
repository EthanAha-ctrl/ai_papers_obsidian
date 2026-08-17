---
source_pdf: REALM.pdf
paper_sha256: b78cc7d3a818c6a45d6a8d9a4bec048acf23220d120d244a710b18a777965ffe
processed_at: '2026-08-11T21:27:57-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# REALM 用人话说

## 这 paper 在干嘛

一句话：**造一个能靠谱评估 robot policy 的 simulation benchmark**。

为啥这事难？因为 robot 这领域有个老 problem — 你训了个 policy，想测它 generalization 怎么样，得在 real world 跑成百上千次。贵、慢、不可复现。大家都说自己 model 厉害，但没法横向比。

那用 simulation 行不行？以前不行，因为 sim 跟 real 差太远 — 画面假、robot 动得不对。你 sim 里测出来的结果跟 real world 根本对不上。

REALM 的核心 contribution 就是：**把 sim 做得足够真实（画面 + 控制），然后用 ~800 对 real-sim rollout 验证说"你看，sim 里测的结果跟 real world 高度相关"**。这样大家就能放心用 sim 大规模测了。

Project page: https://martin-sedlacek.com/realm/

## 怎么把 sim 做真实的

两件事：

### 画面

用 IsaacSim 渲染，visual fidelity 足够高。论文没做 texture matching 之类的 trick，直接验证说 attention map 对比 sim 和 real 的 cosine similarity 有 0.85/1（Fig. 7）。意思是 π₀ 看模拟画面和看真实画面时，attention 关注的 image patch 几乎一样。

这说明 visual gap 基本不是问题了。

### 控制对齐

这个是真有技术含量的部分。问题是：你给 robot 同样的 action command，real robot 和 sim robot 走出来的 joint trajectory 不一样。因为 sim 里的 friction, armature（motor 的 reflected inertia）这些物理参数跟真实 robot 对不上。

做法是 system identification：
- 把 controller 在 IsaacSim 里重写一遍，留 14 个 free parameters（friction + armature）
- 拿 3 条 real robot 的 trajectory replay 数据
- 用 CMA-ES 优化这些参数，让 sim trajectory 跟 real trajectory 的 L2 距离最小

公式就一行：
$$\mathcal{L} = \sum_{i=1}^{N} \sum_{t=1}^{T} \|\mathbf{q}_{i,t}^{\text{real}} - \mathbf{q}_{i,t}^{\text{sim}}\|_2^2$$

$\mathbf{q}$ 是 7D joint angle，$i$ 是 episode index，$t$ 是 timestep。N=3 是因为 heterogeneous parallel sim 在他们的 IsaacSim 版本不支持，只能用少量数据。

Fig. 4 看得出来效果很好 — 默认 controller 下 sim 跟 real 差很远，aligned control 后基本重合。

为啥用 CMA-ES 不用 gradient descent？因为 physics simulator 不可微，black-box optimization 最合适。CMA-ES 是进化算法里调 continuous parameters 的标准工具。

参考: https://en.wikipedia.org/wiki/CMA-ES

## Benchmark 长啥样

- **7 个 skills**: pick, put, push, rotate, stack, open, close
- **10 个 tasks**: 8 个 base (pick-place 类) + 2 个 articulated (开抽屉)
- **15 个 perturbations**: 这是最核心的，分三类
- **3500+ objects**

### Perturbation 三大类

**Visual** — 改画面，不改任务：
- V-AUG: 加 blur, 改 contrast
- V-SC: 加 distractor objects
- V-VIEW: 挪相机
- V-LIGHT: 改灯光

**Semantic** — 改语言指令：
- S-PROP: "拿那个红色的"（用属性指代）
- S-LANG: "grab" 换成 "take"，去掉 "the"
- S-MO: "把杯子放到盘子左边"（空间关系）
- S-AFF: "我渴了"（人类需求）
- S-INT: "拿那个可回收的"（世界知识）

**Behavioral** — 改物理/对象，需要 motor 适配：
- B-HOBJ: 改 object mass
- VB-POSE: 改 object 位姿
- VB-MOBJ: 改 object size/shape
- SB-NOUN: 换 scene 里另一个已知 object
- SB-VRB: 换 skill（比如 pick 换成 push）
- VSB-NOBJ: 换成完全没见过的新 object

### Tiered progression metric

不搞 binary success/fail，搞分阶段打分。

比如 `Put` task:
```
Reach → Grasp → Lift → Move Close → IsInside
 0.2    0.4     0.6      0.8         1.0
```

每一步到了就加 0.2。这样能看出 model 是挂在哪一步 — 是 reach 不到、grasp 不到、还是 final placement 错了。比 binary success 信号丰富得多。

## Real-to-Sim 验证结果

这是 paper 最关键的支撑章节。如果 sim 跟 real 不相关，整个 benchmark 就没意义。

做了近 800 对 rollout，3 个 model，7 个 task，5 个 perturbation。用三个 metric：

1. **Pearson correlation r**: sim 和 real task progression 的线性相关性
2. **p-value**: t-test，看相关性是否显著
3. **MMRV**: sim 里 policy A > B 但 real 里 A ≤ B 的 rank violation，越低越好

Fig. 6 结果：r 很高，p < 0.001，MMRV 很低。而且这还是在 "digital cousin"（不重建原 scene，生成视觉相似但参数不同的 scene）上测的，说明 generalization 也在。

关键 take-away: **simulation 可以作为 real-world performance 的 reliable proxy**。

## 实验发现：VLA 模型到底行不行

测了三个 model: π₀, π₀-FAST, GR00T N1.5。每个 model 跑了约 4000 个 sim rollout。用 RMSD 量化 perturbation 效果：

$$\text{RMSD}(p) = \sqrt{\frac{1}{MT} \sum_{m=1}^{M} \sum_{t=1}^{T} (r_{m,t}^{p} - r_{m,t}^{\text{default}})^2}$$

$M=3$ models, $T=10$ tasks, $r$ 是 task progression，每个 $(m,t,p)$ 用 25 rollouts 平均。Ideal generalist 的 RMSD 应该接近 0。

### 发现 1: Visual perturbation 整体 OK 但不均匀

- **轻**: V-AUG (blur/contrast), V-LIGHT (光照) — RMSD 低。为啥？DROID training data 本身 visual diversity 高，加上 VLM backbone 预训练见过各种图像。
- **重**: V-VIEW (相机视角), V-SC (distractor)

有意思的是 V-VIEW 有时候反而**提升** performance（Fig. 8），说明 default 视角未必是最优的，model 在别的视角下偶尔表现更好。

### 发现 2: Semantic perturbation 暴露 π₀ 的软肋

这是最 striking 的 finding。π₀ 在 semantic perturbation 下 drop 明显大于 π₀-FAST。

Hypothesis: **π₀ 用 diffusion objective 生成 action，可能损害了 VLM backbone 的 language understanding**。π₀-FAST 用 FAST tokenization（离散化 action token，autoregressive 生成），更"友好"于 backbone 的 language capability。

具体排序：
- 最难: S-INT (世界知识), S-AFF (人类需求) — 真需要 Internet-scale text pretraining
- 意外简单: S-MO (空间关系) — 尽管 DROID 没显式标注空间关系

S-MO 为啥简单？可能 model 直接从 visual cues 读空间关系，不需要 language 帮忙。

参考 FAST: https://arxiv.org/abs/2501.09747
参考 π₀: https://arxiv.org/abs/2410.24164

### 发现 3: Behavioral 是最难的一关

- **SB-VRB** (换 skill，同 object): 两 π models 都还行 — motor pattern 在 known object 上 transfer 得了
- **SB-NOUN** (scene 里已知 object): 中等 drop
- **VSB-NOBJ** (完全没见过的 object): 大 drop — 这才是真 generalization 瓶颈

直觉理解：model 在训练数据里的 object 上学到了 "visual appearance → motor pattern" 的 mapping。换新 object 这个 mapping 就失效了。Object-centric generalization 是当前 VLA 的核心瓶颈。

VB-POSE (改 object 位姿) 排名第 2 难，drop 0.12 — 位姿一变，visual 和 motor planning 都得重来。

### 发现 4: GR00T 大幅落后

GR00T N1.5 是 NVIDIA 的 humanoid foundation model。论文自己 fine-tune 到 π 的 action space，但 success rate 远低于两 π models，time to completion 也长（~30s vs ~20s），variance 大。

可能原因:
1. GR00T 原生优化 humanoid bimanual，单臂 DROID setup 不是它强项
2. 自己 fine-tune 不如 Physical Intelligence 官方的 native DROID fine-tune

### 发现 5: 整体 robustness 远未到 deployment 级别

即使最好的 π₀-FAST，很多 task success rate 也不高。Completion time 20-30s 对这些简单 manipulation task 来说太长了。说明 model 在 unseen environments 下还在 "挣扎"，离 autonomous deployment 有距离。

## 论文最后给的 6 个 lessons

用大白话:

1. **Sim 能当 real proxy** — 高保真 + 控制对齐做到了，correlation 强
2. **VLM backbone 也没救了 semantic** — 即使预训练见过 Internet-scale text，semantic perturbation 下还是 drop
3. **相机视角 sensitivity 还在** — 尽管 DROID 已经多视角了
4. **跨 object 的 behavioral generalization 最难** — unseen object 是最大瓶颈
5. **同 object 换 skill 相对 OK** — motor skill 在 known object 上能 transfer
6. **Robustness 整体不行** — success rate 偏低，远没到能部署的水平

还有一个 hypothesis 没列入 lessons 但我觉得很重要: **Full fine-tuning on DROID 可能损害 VLM backbone 的 generalization**。意思是你在 fine-tune VLA 时把整个 backbone 都更新了，可能把预训练学到的 language/visual knowledge 给 "覆盖" 了。这指向未来应该用 LoRA / adapter / partial fine-tuning 之类的策略。

## 我的几个 intuition

### Diffusion vs Autoregressive action generation

π₀ 用 flow matching (diffusion 变体) 生成 continuous action，π₀-FAST 用 FAST tokenizer 把 action 离散化后 autoregressive 预测。

从 REALM 结果看，**autoregressive 在 semantic generalization 上明显占优**。这暗示 diffusion objective 训练时梯度信号可能"覆盖"了 VLM backbone 的 language representation，而 autoregressive token prediction 跟 VLM 原生的 next-token prediction 形式一致，backbone 的 language capability 保留得更好。

这是个值得深挖的方向: **action representation 的形式会影响 VLM backbone 的 capability retention**。

参考 FAST tokenization: https://arxiv.org/abs/2501.09747

### Object-centric vs Scene-centric

VSB-NOBJ 比 SB-NOUN 难很多，说明 "scene 中 known object 的 motor transfer" 和 "unseen object 的 visual-grounded motor learning" 是两个不同难度的问题。

未来 VLA 可能需要:
- 更强的 object perception module（neural field, 3D reconstruction, pose estimation）
- Object-centric representation（不能只靠 2D pixel + language）
- Explicit object property grounding（mass, material, geometry）

### System identification 的可推广性

他们用 N=3 trajectory 做 system ID 就够了，说明 DROID 这种 Franka arm 的 dynamics 相对简单。但换成 humanoid (GR00T 原生场景) 或 bimanual，N=3 可能不够。Heterogeneous parallel sim 一旦支持，可以做更大规模 system ID，甚至 per-joint 细粒度参数。

### Benchmark 设计哲学

REALM 的 tiered progression metric 是个好 design pattern。Binary success 在 long-horizon task 上信号太弱。Rubric-based scoring (Reach → Grasp → ...) 给 debugging 信号。

这跟 LLM evaluation 的趋势一样 — 从 exact match 到 rubric scoring (如 LLM-as-judge)。Robot learning 也在走这条路。

### Sim-to-Real 的"够用就行"哲学

REALM 没追求 sim 跟 real 完全一致，而是追求"correlation 足够强"。这跟 NVIDIA的 "digital twin" 思路不同 — "digital cousin" 不重建原 scene，生成视觉相似但参数不同的 scene，仍能保持 correlation。

这降低 sim 重建成本，让 benchmark scalable。参考: https://digital-cousins.github.io/

## 这 paper 对 robot learning 圈的意义

1. **给 VLA 评估立了个 standard** — 以前大家都自己搭 real-world setup 测，没法比。REALM 提供一个可复现的 sim benchmark。

2. **证明 sim evaluation trustworthy** — 不是 hand-wavy 说"sim 跟 real 差不多"，而是用 800 对 rollout + Pearson + MMRV 量化验证。

3. **暴露了 VLA 的真问题** — visual generalization 还行，semantic 不均（π₀ diffusion 有问题），behavioral 跨 object 是死穴。这些 finding 之前散在小规模 real-world test 里，现在 systematically 量化了。

4. **给未来研究指方向**:
   - Action representation 如何保留 VLM capability
   - Object-centric representation for motor generalization
   - Partial fine-tuning vs full fine-tuning
   - Multi-view robustness training

5. **开放性**: 支持更多 embodiment (现在只 DROID)、更多 skill、更多 perturbation 是未来工作。

## 一句话总结

REALM 用高保真 visuals + aligned control + digital cousin scene 做了个 VLA generalization benchmark，用 800 对 real-sim rollout 验证 sim 结果跟 real 高度相关，然后测了 π₀/π₀-FAST/GR00T 发现：visual 还行、semantic 上 diffusion-based π₀ 明显弱于 autoregressive π₀-FAST、unseen object 的 behavioral generalization 是最大瓶颈、整体 robustness 远没到能部署的水平，full fine-tuning 可能损害了 VLM backbone 的预训练 capability。

参考链接汇总:
- REALM project: https://martin-sedlacek.com/realm/
- SIMPLER: https://simpler-env.github.io/
- DROID: https://droid-dataset.github.io/
- π₀: https://arxiv.org/abs/2410.24164
- π₀-FAST: https://arxiv.org/abs/2501.09747
- GR00T N1: https://arxiv.org/abs/2503.14734
- ⋆-Gen Taxonomy: https://arxiv.org/abs/2503.01238
- Digital Cousins: https://digital-cousins.github.io/
- CMA-ES: https://en.wikipedia.org/wiki/CMA-ES

---

# REALM: Real-to-Sim Validated Benchmark for VLA Generalization

## 1. 论文定位与核心动机

REALM 是一个面向 Vision-Language-Action (VLA) 模型 generalization 评估的 simulation benchmark。核心 thesis 是：通过 high-fidelity visuals + aligned robot control，simulation evaluation 结果能 strongly correlate with real-world performance，从而提供一个 reproducible、scalable、systematic 的评估代理。

这个工作填补了几个关键空白：
- Real-world evaluation (如 RoboArena [15]) 难以 reproducible 且 expensive，几百个 rollouts 就不可持续
- 现有 simulation benchmarks (GemBench [17], VLABench [18], COLOSSEUM [19], SIMPLER [20]) 要么 visual fidelity 不足，要么 perturbation 数量有限，要么 control 不对齐
- VLA 模型 (π₀, π₀-FAST, GR00T N1.5) 的 generalization 能力缺乏 systematic probing

参考链接：
- 项目主页: https://martin-sedlacek.com/realm/
- SIMPLER: https://simpler-env.github.io/
- COLOSSEUM: https://github.com/arc-l/c-more
- DROID: https://droid-dataset.github.io/

## 2. Benchmark Design 深入

### 2.1 Skills 与 Tasks 的分层

论文区分了两个层级：
- **Skills**: 7 个 generic manipulation primitives — picking, putting, pushing, rotating, stacking, opening, closing
- **Tasks**: skill 的具体 instantiation — 特定 object 在特定 scene 中的操作

Benchmark 分为两个 task set：
- **REALM-base**: 8 个 pick-place tasks（Fig. 2 显示 6 个）
- **REALM-articulated**: 2 个 open/close tasks on cabinet drawers

### 2.2 Perturbation Taxonomy

15 个 perturbations 分为三个 category，这是 ⋆-Gen taxonomy [16] 的扩展：

**Visual (V)**: 仅改变 pixel space observation，不需行为调整
- V-AUG: blur & contrast randomization
- V-SC: spawn distractors
- V-VIEW: external camera pose shifts
- V-LIGHT: illumination color/intensity（REALM 新增）

**Semantic (S)**: 改变 language instruction，需理解自然语言
- S-PROP: 用 object property reference
- S-LANG: 同义 verbs + 去除 articles
- S-MO: spatial relationships
- S-AFF: human needs/use cases
- S-INT: world facts 需 Internet-scale knowledge

**Behavioral (B)**: 需 motor control 适配
- B-HOBJ: 改变 object mass
- VB-POSE / VB-MOBJ / SB-NOUN / SB-VRB / VSB-NOBJ: 组合 perturbations

### 2.3 Tiered Progression Metric

这是论文一个关键 design choice — binary success rate 太粗。Tiered progression $r \in [0, 1]$ 是 ordered discrete states 的等权和：

以 `Put` skill 为例：
```
Reach → Grasp → Lift → Move Close → IsInside
 0.2    0.4     0.6      0.8          1.0
```

以 `Open` 为例：
```
Reach → Touch & Move → Open 50% → Open 75% → Open 95%
```

这比 binary success 更 informative，可以区分 "完全失败"、"approach 阶段成功但 grasp 失败"、"grasp 成功但 placement 失败" 等。

## 3. System Identification: Control Gap 缓解

这是论文技术含量最高的部分。目标：让 simulated robot 执行同一 action sequence 后，joint trajectory 与 real robot 对齐。

### 3.1 参数化

在 IsaacSim [30] 中 re-implement controller，留 14 个 free parameters：
- $\boldsymbol{\theta}_{\text{friction}}$: joint friction（mechanical resistance to motion）
- $\boldsymbol{\theta}_{\text{armature}}$: joint armature（reflected inertia of motors，helps simulation stability）

### 3.2 Optimization Objective

Dataset 定义：
$$\mathcal{D} = \left\{ \{(\mathbf{q}_{i,t}^{\text{real}}, \mathbf{q}_{i,t}^{\text{sim}}); \mathbf{q}_{i,t} \in \mathbb{R}^7\}_{t=1}^{T} \right\}_{i=1}^{N}$$

变量含义：
- $i$: episode pair index
- $t$: timestep within episode
- $N$: episode pair 总数（实际用 N=3）
- $T$: episode 长度
- $\mathbf{q}_{i,t}^{\text{real}}$: real robot 7D joint angle vector at timestep t
- $\mathbf{q}_{i,t}^{\text{sim}}$: simulated robot 7D joint angle vector at timestep t

Loss function:
$$\mathcal{L}(\boldsymbol{\theta}_{\text{friction}}, \boldsymbol{\theta}_{\text{armature}}) = \sum_{i=1}^{N} \sum_{t=1}^{T} \|\mathbf{q}_{i,t}^{\text{real}} - \mathbf{q}_{i,t}^{\text{sim}}\|_2^2$$

这是标准 L2 trajectory matching loss。

### 3.3 优化策略

- **Algorithm**: CMA-ES [49] (Covariance Matrix Adaptation Evolution Strategy)
- **为什么用 CMA-ES**: 黑盒优化，无需 gradient；适合 non-differentiable physics simulator
- **Initialization**: 用 CMA-ES 得 initial estimates
- **Refinement**: parameter search with annealing values
- **Constraint**: heterogeneous parallel simulation 在所用 IsaacSim 版本中不原生支持，故只能用 N=3 trajectories

Fig. 4 对比显示 aligned control 显著改善 trajectory following，blue (sim) 与 yellow (real) 高度重合。

参考：CMA-ES https://en.wikipedia.org/wiki/CMA-ES

## 4. Real-to-Sim Validation 方法论

### 4.1 Validation Setup

- 3 个 VLA models: π₀, π₀-FAST, GR00T N1.5
- 7 tasks
- 5 perturbations
- ~800 real-sim rollout pairs
- 用 "digital cousin" [36] — 不重建原 scene，而生成视觉相似但参数不同的 scene（Fig. 5）

### 4.2 三个 Metrics

**(i) Pearson correlation coefficient r**: 
$$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \cdot \sum_i (y_i - \bar{y})^2}}$$
其中 $x_i$ 是 real task progression，$y_i$ 是 sim task progression。越高越好。

**(ii) p-value**: two-sided t-test on observed data。越低越好。论文报告 $p < 0.001$。

**(iii) MMRV (Mean Maximum Rank Violation)** [20]: 
- rank violation: policy A 在 sim 中胜过 B，但 real 中 A 不胜 B
- average over max pairwise rank violations
- 越低表示 sim 与 real 的 policy ranking 越一致

Fig. 6 显示：
- 整体 strong Pearson correlation，datapoints 接近 identity line
- 各 perturbation 下 correlation 也高
- MMRV 较低，说明 policy ranking 一致

### 4.3 Visual Gap 的独立验证

除 task progression correlation 外，还做了 attention map 对比（Fig. 7）：
- Replay same trajectory in real 和 sim
- 提取 π₀ action expert 的 attention maps（across layers, heads, ~280 frames）
- 计算 cosine similarity
- 得 0.85/1 的高分

这印证 [50] 提出的 OOD indicator — visual gap 不足以让模型 predictions 大幅偏离。

直觉理解：π₀ 的 attention 在 sim 渲染上仍指向相同 spatial patches，说明 photorealism 已足够。

## 5. Experimental Results 详解

### 5.1 RMSD Definition

$$\text{RMSD}(p) = \sqrt{\frac{1}{MT} \sum_{m=1}^{M} \sum_{t=1}^{T} (r_{m,t}^{p} - r_{m,t}^{\text{default}})^2}$$

变量含义：
- $p$: active perturbation
- $M$: 模型数 (M=3)
- $T$: 任务数 (T=10)
- $r_{m,t}^{p}$: model $m$ 在 task $t$ 下 perturbation $p$ 的平均 task progression
- $r_{m,t}^{\text{default}}$: model $m$ 在 task $t$ 默认设置下的平均 task progression
- 每个 $(m, t, p)$ triplet 用 25 rollouts 平均
- $r \in [0, 1]$
- RMSD $\in [0, 1]$, ideal generalist 应接近 0

### 5.2 Factor I: Visual Generalization

Visual perturbations (V-AUG, V-VIEW, V-SC, V-LIGHT) 平均 RMSD ≥ 0.12。

排序（Fig. 9）：
- **最不 impactful**: V-AUG (Blur & Contrast), V-LIGHT (Illumination)
  - 假设：DROID training data visual diversity + VLM backbone 提供了 robustness
- **最 impactful**: V-VIEW (viewpoint), V-SC (distractors)

有意思的 observation：V-VIEW 有时反而**提升** absolute task progression（Fig. 8），暗示 DROID 多视角 diversity 让模型在某些视角变化下表现 OK，甚至 default 视角未必是最优。

### 5.3 Factor II: Semantic Generalization

最 striking 的 finding：semantic perturbations 对 π₀ 的影响显著大于 π₀-FAST。

Hypothesis：**π₀ 的 diffusion training objective 损害了 language understanding**。π₀-FAST 用 FAST tokenization [10] — 一种离散化 action token 方法 — 可能保留了 VLM backbone 的 language capability。

排序：
- **最 impactful**: S-INT (world facts), S-AFF (human needs) — 需 Internet-scale knowledge
- **surprisingly low impact**: S-MO (spatial relationships) — 尽管 DROID 中未显式表示

直觉：spatial reasoning 可能通过 visual cues 间接解决，不需 explicit language representation。而 S-INT/S-AFF 真的需要 large-scale text pretraining 来 support。

### 5.4 Factor III: Behavioral Generalization

Behavioral perturbations 是最 challenging 的 category。

关键 observations：
- **SB-VRB** (change skill, same object): 两 π models 都 generalize 相对好 — 同 object 的 motor pattern 适配容易
- **SB-NOUN, VSB-NOBJ** (change object): 显著 drops
  - SB-NOUN: scene 中 known object — DROID 中 well represented
  - VSB-NOBJ: unseen object — 对两 model 都显著 harder
- **VB-POSE** (object pose): 排名第 2，drop 0.12 — pose 改变让 visual + motor planning 都失效
- **VB-MOBJ, B-HOBJ** (size/shape/mass): 中等影响

直觉：motor control 在 known object 上是 "transferable skill"，但 unseen object 需要 visual-grounded control re-learning，这是当前 VLA 的瓶颈。

### 5.5 Factor IV: Robustness & Task Completion

Fig. 10 Bayesian posterior of success rate（uniform Beta prior）：
- π₀-FAST 在 9/10 tasks 上 success rate 最高
- π₀ 在 4 tasks 上 better or comparable
- GR00T 大幅落后

Time to completion：
- π models: ~20s
- GR00T: ~30s with large variance

关键 insight：**这些简单 manipulation tasks 20-30s 的 completion time 表明 models 在 unseen environments 下仍 struggle**，远未到 autonomous deployment 级别。

### 5.6 GR00T 表现低的原因

论文 fine-tune GR00T 到 π 的 action space，但仍大幅落后。两个可能原因：
1. GR00T N1.5 (NVIDIA) 的 humanoid 优化可能在 bimanual/arm-specific action space 上不擅长
2. Fine-tuning quality 不如 π 的 native DROID fine-tuning

## 6. 核心发现总结

论文 Section VI 列了 6 个 lessons：

(i) High-fidelity sim + aligned control 是 real-world 的 valuable proxy
(ii) VLM backbone 在 Internet-scale pretraining 后仍有 semantic perturbation 下显著 drop
(iii) Camera view sensitivity 仍然存在，尽管 DROID 多视角
(iv) Behavioral generalization across objects/properties 是最 challenging
(v) Known skills 在 same object 上 generalize 较好
(vi) Robustness 远未 solved，success rates 整体偏低

最后一个 hypothesis 也很重要：**Full fine-tuning on DROID 可能损害 VLM backbone 的 generalization capability**。这暗示可能需要 partial fine-tuning 或 adapter-based methods (如 LoRA)。

## 7. 与相关工作的联系

### 7.1 Sim-to-Real 方向

- **SIMPLER** [20]: 唯一 established fully-sim evaluation of real-trained policies；REALM 在此基础上扩展更多 skills, objects, perturbations, multi-view
- **Digital Cousins** [36]: REALM 用此方法生成 sim scene
- **SureSim** [33]: 用少量 real eval + sim augmentation
- **Real-is-Sim** [31]: Dynamic digital twin
- **RoboArena** [15]: 分布式 real-world evaluation — 无 systematic perturbation probing

### 7.2 VLA 模型谱系

- **π₀** [9]: VLA flow model，diffusion-based action generation
- **π₀-FAST** [10]: FAST tokenization, autoregressive 而非 diffusion
- **GR00T N1.5** [11]: NVIDIA humanoid foundation model
- **RT-2** [12]: 早期 VLA，web knowledge transfer
- **OpenVLA** [14]: 开源 VLA
- **Gemini Robotics** [13]: Google 最新 VLA

### 7.3 World Models 的局限

论文提到 action-conditioned world models (如 Ctrl-World [46]) 的问题：缺乏 granular control over object mass, pose, illumination。Explicit physics simulator 仍提供不可替代的 controllability。

## 8. 局限性

论文承认：
1. 部分 model 在某些 task 上 performance 太低，使 perturbation uninformative
2. 仅支持 DROID embodiment（未来扩展）
3. System identification 用 N=3 trajectories（computational constraint）

## 9. 对 VLA 研究的直觉 Insight

基于这篇论文，我们可以构建几个 intuition：

1. **Diffusion vs Autoregressive action generation**: π₀ (diffusion) 在 semantic generalization 上明显弱于 π₀-FAST (autoregressive)。这暗示 diffusion objective 可能 "覆盖" VLM 的 language understanding，而 autoregressive token prediction 更"友好"于 backbone。

2. **Visual generalization 主要靠 VLM backbone**: V-AUG, V-LIGHT 低 RMSD 说明 visual robustness 主要继承自预训练。但 V-VIEW, V-SC 高 RMSD 说明 spatial arrangement 和 distractor 处理仍是瓶颈。

3. **Object-centric generalization 是核心瓶颈**: SB-NOUN vs VSB-NOBJ 的差距证明 VLA 在 "known object 上的 motor skill transfer" 与 "unseen object 的 visual-grounded motor learning" 是不同难度的问题。这指向未来需要 explicit object-centric representation 或 scene grounding。

4. **Sim 是 affordable 评估途径**: 4000 rollouts/model 的规模在 real-world 不可行；REALM 证明 high-fidelity sim 可以 trustworthy。

5. **Tiered progression metric 是好 design**: 比 binary success 多了 debug signal，能定位失败 stage。

6. **Fine-tuning 全量可能损害 generalization**: 这是重要 caveat — VLA 的 VLM backbone 可能在 fine-tuning 后丢失了 pretraining 的某些 capability。

参考链接：
- π₀ paper: https://arxiv.org/abs/2410.24164
- π₀-FAST paper: https://arxiv.org/abs/2501.09747
- GR00T N1: https://arxiv.org/abs/2503.14734
- ⋆-Gen Taxonomy: https://arxiv.org/abs/2503.01238
- RoboArena: https://arxiv.org/abs/2501.09029 (近似)

## 10. 未来方向推测

基于 REALM 的发现，几个 promising 方向：

1. **Partial fine-tuning experiments**: 系统性对比 full FT vs LoRA vs frozen VLM + action head
2. **Object-centric pretraining**: 在 fine-tuning 前加入 object property 预训练
3. **Multi-view robust training**: 在 VLA training 中显式 augment camera view
4. **Sim-to-real curriculum**: 用 REALM 做 continual learning 评估
5. **扩展到 bimanual / humanoid**: 支持更多 embodiment
6. **World model + simulator hybrid**: 用 world model 加速 sim，用 simulator 保证 controllability
7. **Long-horizon tasks**: REALM 当前都是 short skills，扩展到 multi-step tasks 测 planning

## 11. 关键数据表总结

| Metric | 值 |
|---|---|
| Skills | 7 |
| Tasks (base + articulated) | 8 + 2 = 10 |
| Perturbations | 15 (V/S/B = 6/8/7) |
| Objects | 3500+ |
| Real-sim validation pairs | ~800 |
| Sim rollouts per model | ~4000 |
| Rollouts per (m,t,p) | 25 |
| System ID trajectories | N=3 |
| Attention cosine similarity (sim-real) | 0.85 |
| Pearson r p-value | < 0.001 |

## 12. 一句话总结

REALM 通过高保真 visuals + aligned control + digital cousins + extensive real-to-sim validation，建立了一个 trustworthy 的 VLA generalization benchmark；用它评估 π₀/π₀-FAST/GR00T 揭示：VLA 在 visual 上有部分 robustness（继承自 VLM backbone），在 semantic 上 diffusion-based π₀ 明显弱于 autoregressive π₀-FAST（暗示 diffusion objective 损害 language understanding），在 behavioral 上 unseen object 是最大瓶颈，整体 robustness 远未达 autonomous deployment 级别，full fine-tuning 可能损害 VLM backbone 的 pretraining capability。
