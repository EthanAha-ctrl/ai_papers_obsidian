---
source_pdf: InternData-A1 Pioneering High-Fidelity Synthetic Data.pdf
paper_sha256: 666de30178cc2b3e73aac3dde4f014e2f86835b66b38214bf89ff69034030eec
processed_at: '2026-08-05T10:10:41-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 InternData-A1

## 一句话版本

Shanghai AI Lab 这帮人搞了个纯仿真数据集，630k 条 robot 操作轨迹，7400 小时，然后拿这个去 pre-train 一个 VLA model，结果在真机和仿真上都能打平 Physical Intelligence 那个用上万小时真机数据训出来的 π0。

这件事的意义是：**以前大家都觉得"仿真数据上不了台面"，现在这个观念得改了**。

---

## 为什么这件事重要

先讲背景。robot learning 这两年有个尴尬的现实：

Physical Intelligence 的 π0 是目前公认最强的 general robot policy，他们用了一个叫 π-dataset 的东西，据说是上万小时的真机遥操数据，关起门来不开放。你想复现？没门。你想研究 "到底多少数据够用"？也没门。整个社区被这个 closed-source dataset 卡着脖子。

真机数据为什么这么难搞？你想想，一条 bimanual trajectory，一个熟练操作员大概 1-5 分钟能录一条，一天 8 小时也就 100-300 条。要堆到 10k hours，你得养一个工厂的操作员队伍，配几十台机器人，搞半年。全世界能干这事的 lab 不超过 5 个。

那仿真呢？仿真便宜啊。一张 4090 一天能跑 200 小时数据，电费加折旧算下来一条 trajectory 不到 3 分钱人民币。问题是——**仿真数据一直被认为"不顶用"**。为什么？

1. 图太假，model 学不到真实视觉特征
2. 物理不对，接触、摩擦、柔软体都模拟不准
3. Task 太窄，以前的数据集基本就是 pick-and-place
4. 没有 sim-to-real 证据，大家觉得仿真训出来的 model 上真机必崩

InternData-A1 这篇 paper 说：这四个问题我们都解决了，而且有数据有证据。

---

## 他们到底做了什么

### 1. 数据规模与多样性

630k 条轨迹，覆盖：
- **4 个机器人**：Franka 单臂、ARX Lift-2 双臂、Agilex Split Aloha 双臂、AgiBot Genie-1 双臂
- **70 个 task**：不是 70 种"换不同的物体 pick 一下"，是 70 个真正不同的任务，包括开微波炉、折叠衣服、倒水、拧瓶盖、做三明治、扫垃圾
- **227 个场景**：厨房、书房、餐厅、客厅
- **3185 个刚体 + 321 个铰接体 + 20 件衣物 + 流体**
- **18 个原子 skill**：pick、place、push、pour、fold、rotate、stack 等等

关键设计是**把 task 当 Lego 拼起来**。18 个原子 skill 就像积木块，70 个 task 是积木搭出来的不同造型。比如"做三明治"= pick 面包 + place 到盘子 + pick 牛肉 + place 到面包上 + pick 第二片面包 + place 到牛肉上。这样 18 个 skill 通过 sequential 和 parallel 两种组合方式，能指数级扩展出 task diversity。

这点很重要。之前的仿真数据集要么是"一招鲜"（GraspVLA 10M 条全是 pick），要么 task 数量少（RoboCasa 100 个但都是 teleop 录的），覆盖不了 VLA model 真正需要的 action space。

### 2. Pipeline 怎么这么快

这是工程上的核心。传统仿真数据生成有个蛋疼的问题：planning 和 rendering 串在一起，planning 是 CPU 活，rendering 是 GPU 活，串起来 CPU 等 GPU、GPU 等 CPU，两个都不饱和。而且 task 越复杂，planning 失败率越高，失败的 trajectory 还白渲染一遍。

他们的解法是 producer-consumer 模式：
- **Planner（CPU）**算 trajectory，算成功的扔进 queue
- **Renderer（GPU）**从 queue 拿 trajectory 渲染
- 两个 stage 独立 batch，中间有个 dynamic scheduler 根据积压情况调 batch size

再加一个 "Stack Render" trick（论文没细讲，我推测是把多个 scene 在一个 render pass 里批量处理），还有集群级别的 Balancer 和 Supervisor 做容错。

结果：8 张 4090 一天 209.7 小时数据，2-3 倍 speedup。

你想想，这个 throughput 是真机遥操的**至少 1000 倍**。成本上完全降维打击。

### 3. 为什么 sim-to-real 能 work

这部分是我觉得最 interesting 的。

他们做了个实验：拿 4 个 task，分别用 200 条真机数据和 200-1600 条仿真数据 fine-tune，对比性能。

结果：
- **Sort Rubbish、Wipe Stain**（简单 pick-place）：200 仿真 ≈ 200 真机，**1:1**
- **Flip Package、Instructional Pick**（复杂 task）：1600 仿真 ≈ 200 真机，**8:1**

8:1 听起来仿真效率低？但你算算成本：1600 条仿真，8 张 4090 不到 1 小时就生成完；200 条真机，一个操作员得录一整天。换算成钱，仿真依然便宜 10 倍以上。

更 striking 的是 Figure 7：他们额外挑了 6 个 task，只用 500 条仿真数据 fine-tune，直接上真机测 30 次：
- Close Microwave: 87% 成功率
- Close Box: 63%
- Handover: 57%
- Make Sandwich: 50%
- Pack: 50%
- Sweep: 60%

**70 个 task 里有 10 个直接 zero-shot sim-to-real 成功率超过 50%**。这在 VLA 社区是第一次。GraspVLA 之前也做过 sim-to-real，但只敢做 pick 这一种 task。

为什么能 work？我的理解是三个原因叠加：

**第一，robot 本身有 tolerance**。PD controller 有 feedback，小控制误差被物理执行层 absorb 了。你不用学到完美的 contact dynamics，学个大概就行。

**第二，domain randomization 把 real 视作 sim 的子集**。174 个环境光照、±5° 相机噪声、随机物体位姿……相当于训练时把 real world 可能出现的情况都 cover 了。Test-time real 只是训练分布里的一个 sample，policy 自然 robust。

**第三，waypoint 抽象把 physics gap 隔离了**。他们的 skill 输出的是 6D end-effector pose，不是 joint action。Joint action 由 CuRobo 重新 solve IK。这意味着 model 学的是"看到这个场景，手该去这个位置"，而不是"看到这个场景，第 3 个关节转 0.3 弧度"。前者是 task-space 抽象，跨 embodiment 和跨 sim-real 都更 robust。

---

## 最关键的 ablation

他们把数据集拆成四块：PnP（30.61%）、Base（35.95%）、Long（21.77%）、Art（11.67%），每次去掉一块训练，看性能掉多少。

| Config | Easy | Hard |
|---|---|---|
| Full | 58.0 | 25.0 |
| w/o PnP | 57.0 | 22.5 |
| w/o Art | 55.5 | 19.5 |
| w/o Base | 52.5 | 20.5 |
| w/o Long | 54.0 | 19.0 |

**PnP 占 30%，去掉影响最小**。这说明光堆 pick-place 数据没用，GraspVLA 那种 10M 条全 pick 的路线在 general VLA 上是不够的。

**Base 和 Long 去掉影响最大**。这俩是 multi-skill composition 的主体。说明真正驱动 VLA generalization 的是 **trajectory diversity**，不是数据量本身。

这个 insight 对未来 dataset 设计影响很大。你不能只追"我有多少 hours"，你得问"我覆盖了多少 action space 分布"。10M 条 pick 不如 100k 条 70 种 task。

---

## 和 π0 怎么比

公平起见，他们用了完全一样的架构（PaliGemma + flow-matching action expert），完全一样的训练步数（680k vs official 700k），完全一样的 fine-tuning 流程。唯一变量就是 pre-training data 来源。

结果：
- **49 个 sim task 上**：InternData-A1 比 official π0 高 5%（Easy）/ 6.5%（Hard）
- **9 个 real task 上**：regular task 高 6.2%，dexterous task comparable
- **vs 开源数据集**：OXE、Agibot World、RoboCasa 全部被 InternData-A1 打败，尤其 real task 上 RoboCasa 只有 13-23%，InternData-A1 有 60-90%

Hard 模式这个结果特别有意思：训练用 clean data，测试用 cluttered data，InternData-A1 反而优势更大。说明仿真阶段学到的 robustness 不会被 fine-tune 抹掉，它会作为 prior 沉淀下来。

---

## 还有什么没解决

1. **60 个 task sim-to-real 表现未知**。10/70 成功不代表剩下 60 个也行，可能选了 favorable 的 task。作者没说失败 task 的成功率，这点有点 cherry-picking 嫌疑。

2. **Fluid 和 deformable 没真机验证**。倒水、折衣服这类 task 对 physics simulator 要求最高，论文里真机实验主要还是 rigid + articulation。

3. **Skill engineering 的人力没算进成本**。18 个 atomic skill 是手写 scripted policy，每个 task 还要调 spatial range。这个 "minimal manual tuning" 到底多 minimal，论文没量化。

4. **π-dataset 不可得，fair comparison 有上限**。你只能用 official checkpoint，看不到他们的训练曲线、数据分布、ablation。如果 PI 哪天 release 了，可能发现 InternData-A1 没那么接近。

5. **跨 embodiment 的真正上限**。ARX AC One 是 unseen embodiment，但只是 fine-tune 后 comparable，没测 zero-shot 跨 embodiment。π0 本身的 cross-embodiment 能力有多强，这个 baseline 缺失。

---

## 我怎么看这篇 paper

从更宏观角度，这篇 paper 触到了一个有意思的趋势拐点：

过去 2 年，robot learning 社区一直在追"real data scaling"，从 RT-1 的 130k，到 Open X-Embodiment 的 1M+，到 π0 的上万小时。每次 record 都意味着更多 teleop operator、更多机器人、更多钱。这条路本质上不可持续，因为能玩得起的 lab 越来越少。

InternData-A1 证明了一件事：**如果你愿意在 simulation pipeline 工程上投入，把 rendering fidelity、domain randomization、skill composition、task diversity 都做到位，sim data 完全可以替代 real data 做 pre-training**。

这个观念转变会带来几个连锁效应：

1. **Data moat 从"谁有最多真机"变成"谁有最好的 sim pipeline"**。Shanghai AI Lab 这个 pipeline 一天产 200 小时数据，一年 7 万小时，超过 Physical Intelligence 现有数据一个数量级。如果他们持续投入，data scaling 曲线会指数甩开 real-first 的 lab。

2. **Skill library 会成为新的"模型权重"**。18 个 atomic skill 是 hand-engineered 的，这本身是一种 IP。未来可能有 open-source skill library，类似 HuggingFace 之于 NLP model。

3. **Long-horizon task 的 generation 会被 LLM 自动化**。现在 task composition 是手写 YAML config，下一步显然是用 LLM 自动 generate skill sequence，甚至自动 verify + retry。这会让 task diversity 进一步 explode。

4. **Real data 的价值会从"训练用"转向"验证用 + 弥补 sim gap"**。你不需要 10k 小时 real data 训练，但你需要几百小时 real data 来做 sim-to-real gap 评估和最后的 fine-tune polish。

我觉得这篇 paper 是 robot learning 这个领域一个 small but important turning point。它不一定是 SOTA，但它把"仿真能不能替代真机"这个问题从 "probably no" 推到了 "definitely worth pursuing aggressively"。

接下来的 12 个月，我预期会看到至少 3-5 个 lab 跟进，用类似 pipeline 把 sim data 推到 millions of hours 量级。如果 physics simulator 同步进步，能稳定模拟 deformable 和 fluid，那 real data 在 VLA pre-training 阶段可能真的会被边缘化。真机会退回到 "部署 + 验证 + 长尾兜底" 的角色。

Karpathy 你怎么看？我觉得这件事对 NVIDIA Isaac Lab、SimNet、MuJoCo MJX 这些 simulator 生态是个大利好——sim pipeline 工程化会成为下一个 infra 级别的机会。

---

# InternData-A1 深度技术解析

## 一、Paper 核心定位

这篇 paper 来自 Shanghai AI Laboratory 与 Peking University 的合作团队，一作包括 Yang Tian、Yuyin Yang、Yiman Xie、Zetao Cai、Xu Shi，通讯作者为 Jia Zeng、Hao Dong 与 Jiangmiao Pang。核心 thesis 一句话概括：**pure synthetic data 首次能在 VLA pre-training 上 match 当前最强的 closed-source real-robot dataset（即 Physical Intelligence 的 π-dataset）**，并且在 sim-to-real 上展示出令人意外的 robustness。

这是一个相当强的 claim，因为 π0 (Black et al., RSS 2024) 是目前公认的 strongest open-world VLA model，其 π-dataset 规模据称达到 10k hours 量级 real teleoperation data。InternData-A1 用 7433 hours 的纯仿真数据，以同样的 PaliGemma + flow-matching action expert 架构，在 49 个 sim tasks + 9 个 real tasks 上达到甚至超过 official π0 checkpoint。

参考链接：
- π0 official paper: https://arxiv.org/abs/2410.24164
- PaliGemma: https://arxiv.org/abs/2407.07726
- CuRobo (motion planner): https://arxiv.org/abs/2310.17274
- AnyGrasp: https://arxiv.org/abs/2212.08333
- MimicGen: https://arxiv.org/abs/2310.17596
- RoboTwin 2.0: https://arxiv.org/abs/2501.05098 (推测链接)
- GRUtopia: https://arxiv.org/abs/2404.15043
- Objaverse: https://arxiv.org/abs/2212.05639
- OmniObject3D: https://arxiv.org/abs/2306.06005

---

## 二、Dataset 规模与 Composition

从 Table 1 与 Table 5 可以拆出关键数据：

| 维度 | 数量 |
|---|---|
| Trajectories | 637,498 |
| Frames | 401,430,981 |
| Hours | 7,433.91 |
| Embodiments | 4 (Franka, ARX Lift-2, Agilex Split Aloha, Genie-1) |
| Tasks | 70 |
| Scenes | 227 (kitchen/study/dining/living) |
| Rigid objects | 3,185 (107 categories) |
| Articulated objects | 321 (14 categories) |
| Garments | 20 (scan digitized) |
| Skills | 18 atomic primitives |
| Long-horizon tasks | 18 (≥3 sequential skills) |

四个任务大类占比：
- **Pick-and-Place (PnP)**: 195,133 traj, 30.61%
- **Base (含 ≤3 skills 非 PnP)**: 229,168 traj, 35.95%
- **Long-horizon (≥3 skills)**: 138,782 traj, 21.77%
- **Articulation**: 74,415 traj, 11.67%

值得注意的是 trajectory 在 1k–10k 区间分布相对均匀（56/70 任务落在这个区间），这是一个刻意为之的 near-uniform 设计，避免长尾任务 dominated 训练信号。

成本方面：**8× RTX 4090 GPU 可日产 209.7 hours robot data**，单 episode 成本 < 0.003 USD。这个 throughput 与 real teleoperation（一条 bimanual trajectory 通常需要 1–5 分钟人工）相比有 4–5 个数量级的成本优势，是这篇 paper 工程上最 striking 的数字。

---

## 三、Pipeline 架构解析

Figure 3 描述的 pipeline 是四个 decoupled stage：

### Stage 1: Environment Construction
- Robot: USD-format embodiment，验证过 contact dynamics 一致性
- Scene: GRUtopia 的 GRScenes-100 子集，annotated with manipulation-area metadata
- Object library:
  - **Rigid**: 来自 OmniObject3D + Objaverse，每个 object 附带 canonical pose + AnyGrasp 自动生成的 grasp poses
  - **Articulated**: 来自 GRUtopia, GAPartNet, GenSim2, Infinite Mobility, ArtVIP，annotated with joint axes, part poses, damping, stiffness
  - **Deformable**: EinScan Rigel Pro 扫描的 20 件真实衣物，remeshed，用 **Vertex Block Descent (VBD, Chen et al. TOG 2024)** 模拟
  - **Fluid**: particle-based dynamics，容器内 adaptively generate particles，isosurface rendering 重建液面，PBD materials

### Stage 2: Skill Composition
每个 atomic skill 是一个 scripted policy，输入为：
- Object states: $o = (p_o, q_o, j_o)$，其中 $p_o \in \mathbb{R}^3$ 为位置，$q_o \in \mathbb{R}^4$ 为四元数 orientation，$j_o \in \mathbb{R}^k$ 为 articulated joint state
- Robot states: $r = (p_{base}, T_{ee})$，base pose 与 end-effector 6D pose
- User constraints $c$（如 align_axis, ratio_range）

输出为 waypoint sequence $W = \{w_i\}_{i=1}^N$，每个 waypoint $w_i = (p_i \in \mathbb{R}^3, q_i \in SO(3))$ 是 target end-effector 6D pose。

这种 waypoint 抽象非常关键：**high-level skill logic 与 low-level motion execution 完全 decouple**。例如 Pick skill：
```
pre_grasp → grasp → post_grasp
```
对应三个 waypoint，具体如何插值由 CuRobo 解决。

### Stage 3: Domain Randomization
论文里 randomization 颗粒度分四个 level：
1. **Camera**: 主视图与腕部视图 ±5° rotation，±5cm translation
2. **Lighting**: 174 个 environment maps，色温与强度 randomized
3. **Layout**: 同类 object 替换，桌面与背景布局 randomized
4. **Trajectory-level**: object 位置/朝向在 task-specific spatial region 内 sampling；grasp pose 从 AnyGrasp 输出的 top-40 high-confidence candidates 中 random sample

### Stage 4: Generation & Storage
- **CuRobo**（Sundaralingam et al. 2023）将 waypoints 插值为 dense joint-space actions，做 collision-free minimum-jerk motion
- 物理仿真验证 trajectory，**只保留成功的 trajectory** 才进入 rendering
- 输出为 LeRobot 格式（huggingface LeRobot standard）：multi-view RGB + camera intrinsics/extrinsics + proprioceptive state + action labels + language instruction

---

## 四、Framework Optimization（这是工程核心）

### Bottleneck 诊断
传统 pipeline 把 planning 和 rendering 串行放在一个 stage 里，有两个 fundamental 问题：

1. **Planning success rate 随 task complexity 下降**：失败 trajectory 仍然会浪费渲染算力
2. **Compute 特性 mismatch**：planning 是 CPU-bound serial workload，rendering 是 GPU-bound parallel workload，串行执行导致硬件利用率低下

### 解决方案
论文提出 multi-level system optimization：

**1. Stage Decoupling + Pipelined Architecture**

Planner (CPU) 与 Renderer (GPU) 解耦成两个独立 stage，中间用 message queue 通信。Planner 产出一个成功的 trajectory 就 push 到 queue，Renderer 从 queue 消费。这就实现了经典 producer-consumer pattern，CPU 与 GPU 可以并行 saturate。

**2. Dynamic Resource Scheduling**

不同 task 的 planning/rendering 时间比不一样（如 bimanual long-horizon task planning 重，simple pick task rendering 重）。引入 dynamic scheduling algorithm 根据当前 queue 积压动态调整 batch size。

**3. Stack Render**

这个 trick 没有详细展开，但根据上下文推测：把多个 scene 的渲染请求 batch 在同一个 render pass 里，利用 GPU 的 SIMT 特性，类似 NVidia Isaac Sim 的 Multi-Threaded Rendering。具体实现可能是在一个 scene graph 里同时实例化多个 robot+object 子图，用一个 camera 渲染多个 viewport。

**4. Balancer + Supervisor**

Balancer 做集群负载分配，Supervisor 做监控与 failover。这是大规模分布式系统标准组件。

**最终效果：2–3× end-to-end speedup**，支持长时间稳定运行。

---

## 五、VLA Model 架构细节

InternData-A1 直接复用 π0 架构（这是 fair comparison 的前提）：

### 5.1 Vision-Language Backbone: PaliGemma
- 3B parameter VLM
- SigLIP vision encoder + Gemma LLM
- 输入 image $I \in \mathbb{R}^{H \times W \times 3}$ → vision tokens $V \in \mathbb{R}^{N_v \times d}$
- 输入 language $L$ → text tokens $T \in \mathbb{R}^{N_t \times d}$
- Cross-attention fusion

### 5.2 Action Expert: Flow Matching

π0 的 action expert 用 flow matching（Lipman et al. 2023 的 Flow Matching framework），公式：

给定状态 $s = (V, T, r_t)$（vision tokens, language tokens, robot proprioception），action $a \in \mathbb{R}^{D_a}$，flow matching 学习一个 time-conditioned vector field $v_\theta(a, t | s)$：

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1),\, a_1 \sim q(a_1|s),\, a_t = \phi_t(a_0, a_1)} \left\| v_\theta(a_t, t, s) - u_t(a_t | a_1) \right\|^2$$

变量解释：
- $t \in [0,1]$：flow time，从 noise ($t=0$) 到 target action ($t=1$)
- $a_0 \sim \mathcal{N}(0, I)$：flow 起点，standard Gaussian
- $a_1$：target action，来自 demonstration
- $a_t = (1-t) a_0 + t a_1$：linear interpolation（OT conditional flow）
- $u_t(a_t | a_1) = a_1 - a_0$：conditional vector field target
- $v_\theta$：神经网络预测的 vector field

Inference 时从 $a_0 \sim \mathcal{N}(0,I)$ 出发，用 Euler method $a_{t+\Delta t} = a_t + \Delta t \cdot v_\theta(a_t, t, s)$ 积分到 $a_1$。π0 通常用 10 步 Euler。

### 5.3 Action Chunking
π0 预测未来 $H$ 步 action $a_{t:t+H}$ 而不是单步，这是 ACT (Zhao et al. 2023) 提出的关键技术，可以减少 compounding error。InternData-A1 训练时也沿用这一设计。

### 5.4 训练 hyperparameters (Table 6)
| Hyperparam | Pre-training | Fine-tuning |
|---|---|---|
| Batch size | 512 | 128 |
| LR | 5e-5 | 2.5e-5 |
| Schedule | Constant | Cosine decay |
| Steps | 680k | 30k (regular) / 100k (dexterous) |
| Hardware | 32× A100 | 8× GPU |

Pre-training 680k steps 与 official π0 的 700k steps 接近，这是控制变量。

---

## 六、核心实验结果

### 6.1 vs π-dataset (Table 2)

49 个 RoboTwin 2.0 sim tasks：

| Method | Easy avg | Hard avg |
|---|---|---|
| π0 (Scratch) | 23.5% | 2.5% |
| π0 (official) | 55.0% | 20.0% |
| **π0 (InternData-A1)** | **60.0%** | **26.5%** |

InternData-A1 在 Easy 高 5%，Hard 高 6.5%。Hard 模式是 clean training + cluttered evaluation，说明 InternData-A1 学到的 robustness 在 fine-tune 干净数据后依然保留。

### 6.2 Real-world (Figure 5)

9 个 real tasks 跨 3 个 embodiments（Genie-1, ARX Lift-2, ARX AC One）：
- 5 个 regular tasks: Heat Sandwich, Sort Rubbish, Place Markpen, Pass Bottle, Sweep Trash
- 4 个 dexterous tasks: Sort Parts, Unscrew Cap, Fold Cloths, Zip Bag

Regular tasks: InternData-A1 比 π-dataset 高 **6.2%**
Dexterous tasks: comparable（ARX AC One 是双方都 unseen 的 embodiment）

### 6.3 vs Open-source Datasets (Table 3)

| Dataset | Domain | 49 Sim Easy/Hard | Sort Rubbish | Pass Bottle |
|---|---|---|---|---|
| OXE | Real | 32.5/11.0 | 40.0 | 36.7 |
| Agibot World | Real | 52.5/12.0 | 53.3 | 56.7 |
| RoboCasa | Sim | 50.0/11.0 | 23.3 | 13.3 |
| **InternData-A1** | Sim | **60.0/26.5** | **90.0** | **60.0** |

RoboCasa 在 sim 上 competitive，但 real 上崩了（只有 13–23%），InternData-A1 在 real 上 60–90%。这印证 photorealism + domain randomization 对 sim-to-real 的 critical 作用。

### 6.4 Sim-to-Real Ratio 实验 (Figure 6)

四个 task 做 sim vs real 等效性实验：
- **Sort Rubbish / Wipe Stain**: 200 sim ≈ 200 real (1:1)
- **Flip Package / Instructional Pick**: 1600 sim ≈ 200 real (8:1)

1600 sim samples 在 8×4090 上 < 1 小时就能生成，200 real samples 至少 5–10 小时人工，成本依然有数量级优势。

### 6.5 Additional 6 Sim-to-Real (Figure 7)

500 sim episodes，30 rollouts 评估：
- Make Sandwich: 50%
- Pack: 50%
- Close Box: 63%
- Close Microwave: 87%
- Sweep: 60%
- Handover: 57%

**10/70 tasks 直接 sim-to-real 达 >50% success**，这是 VLA 社区第一个 demonstrate diverse & complex tasks 能 zero-shot sim-to-real transfer 的工作（GraspVLA 只做了 pick）。

---

## 七、Ablation：什么真正重要？(Table 4)

把数据集分成四个部分，每次 remove 一个，训练 0.5 epoch，在 RoboTwin 2.0 上评估：

| Config | Easy | Hard |
|---|---|---|
| Full | 58.0 | 25.0 |
| w/o PnP | 57.0 | 22.5 |
| w/o Art | 55.5 | 19.5 |
| w/o Base | 52.5 | 20.5 |
| w/o Long | 54.0 | 19.0 |

两个关键 insight：
1. **PnP 占 30.61%，但移除影响最小**——说明单一 pick-place skill 无法支撑 VLA generalization，验证了 GraspVLA 那种 billion-scale pick-only 数据的局限
2. **Base 与 Long 移除影响最大**——表明 task diversity 与 multi-skill composition 比 skill 总量更重要

作者提出 hypothesis：**trajectory diversity 才是 pre-training 的 core driver**。这个观点和 Helix (Cui et al. 2025)、GR-3 等工作的 trend 一致——单纯堆数据量已经不够，diversity 在 action space 上的覆盖度才是关键。

---

## 八、为什么 Sim-to-Real Work？我的 Intuition 构建

读完这篇 paper，我觉得 sim-to-real work 的核心原因是三个：

### 8.1 Action Prior 抽象层级
VLA model 不是在学 exact physics，而是学 **visuomotor mapping** $a = f(I, L, r)$。当 robot hardware 容忍 small control inaccuracy（PD controller 有 feedback），且 task 允许 approximate contact strategy 时，sim 与 real 的物理 gap 在 policy 层被 absorb。这解释为什么 Sort Rubbish 这种 simple pick-place 在 200 sim = 200 real。

### 8.2 Domain Randomization 的 "Cover the Real" 思想
174 env maps + ±5° camera noise + random lighting + random object pose，相当于把 real world 视作 sim distribution 的一个 sample。Real test-time 分布是 sim 训练分布的子集，policy 在子集上表现自然 robust。

### 8.3 Waypoint 抽象的 Sim-Real Bridge
Waypoint $w = (p, q)$ 是 task-space 6D pose，与 robot joint 配置无关。同一段 waypoint sequence 在不同 embodiment 上由 CuRobo 重新 solve IK，这就是 cross-embodiment transfer 的 mechanism。ARX AC One 是 unseen embodiment，但 action expert 学的是 waypoint → joint action mapping，physics gap 在这一层被部分 decouple。

---

## 九、Limitation 与未来方向

作者承认：physics simulator 限制高 dexterous task（系鞋带、穿针引线）。这其实是所有 simulation-based 方法的通病——柔软体接触、精细摩擦、瞬态力学依然难以准确模拟。

从更宏观角度看，这篇 paper 暗示了几个 trajectory：

1. **Real data 的不可替代性在被打破**：如果 sim 数据足够 diverse + photorealistic，real data 的 marginal value 在下降
2. **Synthetic data pipeline 工程化是 moat**：8×4090 日产 200 hours，这个 throughput 远超任何 real teleoperation farm
3. **Action space diversity > Skill repetition**：这对未来 dataset 设计有指导意义
4. **Compositional skill design**：18 个 atomic skill × 70 task composition，这种 Lego 式构造可能是 manipulation data scaling 的正确路径

---

## 十、与同期工作的对比 Positioning

- vs **π0.5 (Intelligence et al., 2025)**: π0.5 用 web-scale VQA + real data，强调 open-world generalization；InternData-A1 强调 action prior
- vs **GR00T N1 (Bjorck et al., 2025)**: 同样 humanoid focus，但用 hybrid sim+real
- vs **GR-3 (Cheang et al., 2025)**: video-language pre-training 路线
- vs **InternVLA-M1 (Chen et al., 2025b)**: 同一 lab 前作，spatially guided，只 244k traj
- vs **UniVLA (Bu et al., 2025b)**: task-centric latent action
- vs **OpenHelix (Cui et al., 2025)**: dual-system VLA，Helix 是 Figure 公司的 real-data 路线

InternData-A1 的独特 position 是：**pure synthetic + general VLA + first sim-to-real match real-data**。

---

## 十一、Potential Weakness & Open Questions

1. **Sim-to-real 的 cherry-picking**？只有 10/70 tasks direct transfer，剩 60 tasks 失败比例未知
2. **Physics gap 在 fluid/deformable 上有多严重**？paper 没给 fluid/deformable task 的 real 评估
3. **Sim tuning effort**：spatial range 需要手工调整，"minimal manual tuning" 的 minimal 是多少？
4. **Cost 比较 base**：0.003 USD/episode 是 GPU 折旧成本，未计 asset creation 与 skill engineering 人力
5. **Closed-source π-dataset 的 mystery**：fair comparison 受限于 π-dataset 不可得，作者只能用 official checkpoint

---

## 十二、代码与数据

论文承诺 open-source dataset 和 generation pipeline。从 Appendix B 的 YAML config 示例（Listing 1）可以看出 pipeline 是高度 declarative 的：用户写一个 YAML 描述 robot + objects + skills + randomization，框架自动生成 trajectory。

LeRobot 格式输出意味着可以直接用 huggingface LeRobot 的 dataloader，与 community 工具链兼容。

参考：
- LeRobot: https://github.com/huggingface/lerobot
- π0 unofficial JAX impl: https://github.com/markvdm/amberjaxpi0 (推测)
- CuRobo: https://github.com/NVlabs/curobo
- AnyGrasp: https://github.com/graspnet/anygrasp_sdk

---

## 总结性 Intuition

InternData-A1 的核心贡献证明了一件事：**在 manipulation VLA 领域，data scaling law 的 "data" 不再必须是 real**。通过 compositional skill design + photorealistic rendering + extensive domain randomization + scale of 600k+ trajectories，sim data 可以学到一个 action prior 至少与最强 real dataset 同等 expressive。这暗示未来 VLA 的 data moat 不再是 real teleoperation farm，而是 simulation pipeline engineering。

下一步值得探索的方向：
- 是否存在一个 unified skill library 可以覆盖 95% 人类 manipulation task？
- Sim data 与 real data 的 best mixing ratio？
- Action prior 在 unseen embodiment 上的 generalization 边界？
- 是否能用 LLM 自动 compose skill → 自动验证 → 自动生成 task？

这篇 paper 对 community 的最大价值是把 "simulation 不行" 这个旧观念打破了一次，给未来 synthetic data scaling 打开了 conceptual gate。
