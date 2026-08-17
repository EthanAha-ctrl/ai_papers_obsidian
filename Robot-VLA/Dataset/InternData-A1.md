---
source_pdf: InternData-A1.pdf
paper_sha256: 666de30178cc2b3e73aac3dde4f014e2f86835b66b38214bf89ff69034030eec
processed_at: '2026-08-05T10:12:02-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 InternData-A1

## 这篇 paper 到底在说啥

一句话：**有一帮人用纯 simulation 造了 63 万条 robot 操作数据，然后证明只用这些 sim 数据 pre-train 出来的 VLA 模型，跟用真实世界数据训练的最强 π0 打平手，甚至在某些场景还略胜一筹**。

这事以前没人做到。之前 sim data 的名声一直是"看着假""物理不准""只能做 grasping 这种简单任务"。这帮人第一次把 sim data 推到一个临界点——scale 够大、fidelity 够高、diversity 够广——结果就 work 了。

Paper: https://arxiv.org/abs/2507.22022
Code: https://github.com/OpenDataLab/InternData-A1

---

## 为什么这事以前做不到

你想想 robot data 为什么难搞。Real robot data 要人 teleop，一台机器人配一个熟练操作员，一条 trajectory 录起来又慢又贵。Google 那个 Open X-Embodiment、Physical Intelligence 的 π-dataset，都是这么硬录出来的，成本吓人。而且 π-dataset 还是 closed-source，外人根本用不了。

Sim 数据呢？理论上可以无限生成，成本几乎为零。但实际做出来一直有两个问题：

1. **任务太窄**：GraspVLA 生成了 1000 万条 grasping 轨迹，但只有一个 task。模型学到的东西没法 transfer 到别的 task。
2. **Sim-to-real gap 太大**：RoboCasa 这种 sim 数据集，在 simulation 评测里分数不错，一到真实机器人上就崩。rendering 不够真实，物理仿真不够准。

所以一直有个 open question：**到底 sim data 能不能 scale 到 match real data？** 这篇 paper 给了第一个肯定答案。

GraspVLA: https://arxiv.org/abs/2502.20300
RoboCasa: https://arxiv.org/abs/2406.02523

---

## 他们到底怎么搞出来的

这是 paper 真正的工程亮点。我分几个层面讲。

### Asset 这块下了血本

他们攒了一大堆 3D asset：

- **Rigid object**: 3185 个，107 类，从 OmniObject3D 和 Objaverse 拿的
- **Articulated object**: 321 个，14 类——微波炉、抽屉、笔记本电脑这种有 joint 的东西，从 GRUtopia、GAPartNet、GenSim2、Infinite Mobility、ArtVIP 五个来源凑
- **Garment**: 20 件真衣服，用 EinScan Rigel Pro 扫描仪扫的
- **Fluid**: 用 particle-based dynamics 模拟，isosurface 渲染水面
- **Scene**: 227 个室内场景（厨房、书房、餐厅、客厅），从 GRUtopia 的 GRScenes-100 切出来的

每个 object 都带物理 annotation：joint axis、damping、stiffness、grasp pose（AnyGrasp 自动生成）等等。这块工作量巨大，但是是地基。

OmniObject3D: https://omni3d.github.io/
Objaverse: https://objaverse.allenai.org/
GRUtopia: https://github.com/OpenRobotLab/GRUtopia
AnyGrasp: https://arxiv.org/abs/2212.12976

### Skill 抽象是关键 insight

他们把每个 atomic skill 定义成一个函数：

$$
\text{skill}: (s_{\text{obj}}, s_{\text{robot}}, \mathcal{C}) \rightarrow \{w_1, w_2, ..., w_N\}
$$

解释一下变量：
- $s_{\text{obj}}$: object 的状态，包括 6D pose $SE(3)$ 和 joint state（articulated object 才有）
- $s_{\text{robot}}$: robot 状态，base pose + joint configuration
- $\mathcal{C}$: user 定的 constraint，比如"抓握轴要对齐 y 轴"
- $w_i$: waypoint，就是一个 6D end-effector pose + gripper 开合状态

**这个抽象的妙处在于把"做什么"和"怎么做"彻底分开了**。Skill 只负责算"要去哪儿"（waypoints），motion planner 负责算"怎么去"（joint trajectory）。

所以同一个 `pick` skill，换个 robot body 就能直接用，因为 waypoints 是 embodiment-agnostic 的。这让他们能 4 个 embodiment 共用一套 skill library。

然后 task 就是 skill 的组合。写个 YAML config，声明式地说"右手 pick → handover → 左手 place"，一个 long-horizon task 就出来了。论文 Appendix B 有个完整 config 例子，挺直观的。

### Domain Randomization 做得很狠

5 个 axis 全部 randomize：

1. Camera extrinsic: ±5° rotation, ±5cm translation
2. 光照: 174 个 environment map，temperature 和 intensity 都随机
3. Object: 同 category 内随机替换
4. 桌面和背景布局: 随机
5. Grasp pose: 从 AnyGrasp top-40 candidate 里随机选一个

这个思路跟 OpenAI 那个 Hand 论文一脉相承——让 policy 学到对 visual 和 spatial variation 都 invariant 的 representation，而不是 overfit 到某个 specific sim 环境。

OpenAI Hand: https://arxiv.org/abs/1808.00177

### Pipeline 工程优化是真功夫

传统 sim data pipeline 是这样的：planner 算 trajectory → 渲染 → 存。串行的。问题在于：

1. Task 复杂了 planning 成功率下降，失败的 trajectory 不用渲染，但串行架构还是会浪费时间
2. Planning 是 CPU-bound 串行，rendering 是 GPU-bound 并行，串一起 hardware utilization 很差

他们的优化：
- **Stage decoupling**: Planner 和 Renderer 拆成两个独立 process，中间用 pipeline 串起来
- **Dynamic scheduling**: 根据 task 的 heterogeneous time-cost 动态分配资源
- **Stack Render**: 把多个 scene 的渲染 request 堆叠一起送 GPU（这个 trick 论文没细说，我猜是 batched rendering call）
- **Balancer + Supervisor**: 大规模集群部署的负载均衡和监控

**结果：8 张 RTX 4090 一天产 209.7 小时 robot data，每条 episode 成本 < $0.003**。

我算了一下：630k episodes × $0.003 ≈ $1,890 总成本。7433 小时 / 209.7 小时/天 ≈ 35 GPU-days，8 卡并行大概 4-5 天就跑完了。

这个成本对学术界是 game-changer。对比真实数据采集的成本——一台 robot 配一个操作员，一天能录几个小时就不错了——sim 路线的优势在这里第一次展现出压倒性。

---

## 结果：用数据说话

### vs π-dataset（49 个 sim task，RoboTwin 2.0 benchmark）

| | Easy mode | Hard mode |
|---|---|---|
| π0 (Scratch, 没 pre-train) | 23.5% | 2.5% |
| π0 (official, 用 π-dataset) | 55.0% | 20.0% |
| **π0 (InternData-A1, 纯 sim)** | **60.0%** | **26.5%** |

**Easy mode +5%, Hard mode +6.5%**。纯 sim 数据居然赢了真实数据。

Hard mode 领先更明显这事很关键。Hard mode 是 cluttered + domain randomization 评测，而 fine-tune 用的是 clean data。说明 **InternData-A1 的 domain randomization 学到的 robustness 在 clean fine-tune 之后仍然保留**。这个现象跟 Google DeepMind 那波 sim-to-real 的发现一致。

### vs 其他 open-source dataset

| Dataset | 49 Sim (E/H) | 2 Real Task |
|---|---|---|
| OXE (real) | 32.5/11 | Sort Rubbish 40%, Pass Bottle 36.7% |
| Agibot World (real) | 52.5/12 | 53.3%, 56.7% |
| RoboCasa (sim) | 50.0/11 | 23.3%, 13.3% |
| **InternData-A1 (sim)** | **60/26.5** | **90%, 60%** |

**RoboCasa 在 sim 评测里只差 10%，但 real-world 评测差 57.7%**。说明 sim data 要 sim-to-real work，光 sim 评测好没用，rendering fidelity 和数据量是关键。InternData-A1 这两点都做到了。

### Sim-to-real 数据效率

这个实验我觉得最 informative。他们从同一个 π0(InternData-A1) checkpoint 出发，分别用 sim 和 real 数据 post-train，看多少 sim 数据能抵多少 real 数据：

| Task | Sim:Real Ratio |
|---|---|
| Sort Rubbish (简单 pick-place) | **1:1** |
| Wipe Stain (简单) | **1:1** |
| Flip Package (复杂, dynamic object) | 8:1 |
| Instructional Pick (复杂, language grounding) | 8:1 |

**简单任务 sim 数据已经跟 real 数据一样 efficient 了**。复杂任务需要 8 倍 sim 数据，但 sim 数据成本远低于 real 数据的 1/8，所以还是赚的。

而且他们发现 **sim-to-real 不需要视觉完全一致**——背景、光照、object texture、table layout 都不用 exact match，只要 camera view 和 joint action space 对齐就行。这跟 OpenAI Hand 的结论一致，但这次扩展到了 70 个 task。

### Zero-shot sim-to-real（无 real data fine-tune）

10 个 task 用纯 sim data post-train 后直接在真实机器人上跑，平均 success rate > 50%：

| Task | Success Rate |
|---|---|
| Close Microwave | 87% |
| Close Box | 63% |
| Sweep | 60% |
| Handover | 57% |
| Make Sandwich | 50% |
| Pack | 50% |

**这是 GraspVLA 之后第一次证明 complex multi-skill task 也能 zero-shot sim-to-real**。GraspVLA 只做了 grasping 一个 task。InternData-A1 做到了 articulation、bimanual coordination、long-horizon assembly。

---

## π0 架构简单回顾

为了 context，π0 是 Physical Intelligence 的 VLA 模型，由两部分组成：

1. **PaliGemma**: 3B 参数的 vision-language model，输入 image(s) + text prompt
2. **Flow-matching action expert**: 在 PaliGemma representation 之上用 flow matching 生成 continuous action chunk

Flow matching 的核心公式：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]
$$

变量解释：
- $t \sim U[0,1]$: flow 的时间参数，0 是 Gaussian noise，1 是 target action
- $x_0 \sim \mathcal{N}(0, I)$: base distribution，标准高斯
- $x_1 \sim q(x_1)$: target distribution，就是真实 action chunk
- $x_t = (1-t) x_0 + t x_1$: linear interpolation 的中间状态
- $v_\theta$: 神经网络学的 vector field，conditioned on PaliGemma feature $c$
- 目标是让 $v_\theta$ 拟合 conditional vector field $x_1 - x_0$

Inference 时从 Gaussian 采 $x_0$，用 ODE $\frac{dx_t}{dt} = v_\theta(x_t, t, c)$ 积分到 $t=1$ 得到 action chunk。

Flow Matching 原论文: https://arxiv.org/abs/2210.02747
π0: https://arxiv.org/abs/2410.24164
PaliGemma: https://arxiv.org/abs/2407.07726

InternData-A1 训练设置：32× A100 GPU，680k iteration pre-train，batch size 512，lr 5e-5 constant。Fine-tune 用 8× A100，30k iteration (regular task) 或 100k (dexterous task)，batch 128，lr 2.5e-5 cosine decay。

---

## Ablation：什么真正驱动了 pre-training 效果

这个 ablation 我觉得是 paper 最 informative 的部分之一。他们把 InternData-A1 分成 4 块：

- **PnP** (Pick-and-Place): 30.61%
- **Art** (Articulation): 11.67%
- **Base** (简单 task, <3 skills): 35.95%
- **Long** (long-horizon, ≥3 skills): 21.77%

然后每次去掉一块，pre-train 0.5 epoch，在 RoboTwin 上评测：

| Variant | Easy / Hard |
|---|---|
| Full | 58.0 / 25.0 |
| w/o PnP | 57.0 / 22.5 |
| w/o Art | 55.5 / 19.5 |
| w/o Base | 52.5 / 20.5 |
| w/o Long | 54.0 / 19.0 |

**两个关键发现**：

1. **PnP 占了 30% 但去掉影响最小**。单纯堆 pick-and-place 数据对 VLA pre-training 价值有限。这解释了 GraspVLA 1000 万条 grasping 轨迹为什么泛化不好——action distribution 太窄。

2. **Long-horizon 和 Articulation 去掉影响大**。Long 只占 21.77%，去掉降 4% / 6%。Art 只占 11.67%，去掉降 2.5% / 5.5%。说明 **trajectory diversity 比 sample count 重要得多**。

我的 hypothesis：**VLA pre-training 的 effective signal 是 action distribution 的 diversity，不是 sample count**。Long-horizon 自然涵盖更多 skill 组合，articulation 涉及不同 joint geometry 和 distorted arm configuration，都贡献了 action distribution 的 diversity。单纯 pick-and-place 即使 sample 量大，action distribution 也是 narrow 的。

这个 insight 对未来 sim data 生成有指导意义：**别堆量，堆多样性**。

---

## 我对这篇 paper 的整体直觉

### 为什么 sim data 终于 work 了

我归纳 5 个条件，缺一不可：

1. **Photorealistic rendering**: 不是"看起来像"，是让 VLA 学到的 visual feature 在 real domain 仍然 useful
2. **Diverse atomic skills + composition**: 让 model 学到 transferable action priors
3. **Domain randomization**: 让 model 学到 invariant representation
4. **Scale**: 630k trajectories + 7433 hours，达到 critical mass
5. **Embodiment diversity**: 4 个 robot body，让 model 不 overfit 到单一硬件

之前的工作最多做到 2-3 个条件，InternData-A1 第一次全部满足。

### Sim data 的角色会怎么演变

我的 sense：**sim-only 在 pre-train 阶段已经站住脚了**。Real data 的角色会逐渐转移到两个地方：

1. **Fine-tune**: 用少量 real data 把 sim-pretrained model 适配到 specific deployment scenario
2. **Evaluation**: real-world 评测作为 ground truth

Long-term 看 hybrid (sim + real + web video + VQA) 应该还是最优，比如 π0.5 和 GR00T N1 在做的方向。但 sim 的 scaling 成本优势太大，pre-train 阶段 sim-only 已经是可行路线。

π0.5: https://arxiv.org/abs/2504.16054
GR00T N1: https://arxiv.org/abs/2503.14734

### Limitations 的诚实评估

作者承认的 limitation：**高度 dexterous tasks（系鞋带、穿针引线）sim 还做不到**。背后是 physics simulator 的精度问题——deformable object + tight contact + friction 的耦合在当前 PBD/FEM 模型下误差太大。

我推测还有几个 paper 没说的 limitation：

1. **Tactile-heavy tasks**: 没有 tactile sensor，sim 里做不了触觉反馈
2. **Multi-step tight contact**: 比如插 USB 这种 sub-mm precision + compliance 的任务
3. **Long-horizon planning over truly novel objects**: sim 里 object pool 还是 finite 的，遇到分布外的物体可能还是不行
4. **Dynamic scene changes**: 人走过来碰一下桌子这种，sim 里很难模拟

### 我直觉上觉得应该探索的方向

1. **Procedural object generation**: 不依赖 scanned asset，程序化生成 diverse physics 的 object。Infinite Mobility (https://arxiv.org/abs/2503.14314) 是这个方向。

2. **Neural physics surrogate**: 当前 PBD/FEM fidelity 上限有限，用 learned physics model 可能 simulate 更 dexterous 的 task。

3. **Sim-text-video co-training**: sim trajectory + 互联网 instructional video + web VQA 三者 co-train，让 VLA 同时有 action prior + semantic grounding + long-horizon planning。π0.5 在往这走。

4. **Active sim-to-real**: 用 real-world failure case 反过来 targeted 生成 sim data，iteratively close gap。这跟 DAgger 的思路类似但用在 data generation 上。

5. **Closed-loop sim evaluation**: 当前评测是 open-loop action chunk，应该加 closed-loop rollout evaluation——让 sim model 在 sim 里实际执行并 measure success rate，这样能更早发现 sim-to-real gap。

---

## 几个我好奇但 paper 没说清的点

1. **Stack Render 具体怎么实现**？是 batched rendering call 还是别的？这个 trick 能不能复用到其他 sim pipeline？

2. **680k iteration 为什么选这个数**？跟 official π0 的 700k 对齐是为了公平比较，但 680k 之后的 scaling law 是什么样的？继续训会不会更好？

3. **Action chunk length** $H$ 是多少？π0 paper 里是 50，这里沿用了吗？

4. **Failure recovery 数据怎么生成**？Table 1 说 InternData-A1 有 failure recovery 标签，但 pipeline 看上去只 generate 成功 trajectory。是不是 post-hoc 在成功 trajectory 上 inject perturbation？

5. **Cross-embodiment action space normalization** 怎么做的？PaliGemma 输入是 image+text，但 action expert 怎么处理不同 embodiment 的不同 joint dimension？是 pad 到统一长度还是用 separate action head？

---

## 给 Karpathy 的 takeaway

1. **Synthetic data 在 VLA pre-training 上终于立住了**。630k sim trajectories 可以匹配 closed-source π-dataset。这是 sim data 第一次在 scale + diversity + fidelity 三个 axis 都达到 critical mass。

2. **Pipeline engineering 比 model architecture 更 critical**。CuRobo + VBD + AnyGrasp + domain randomization + pipeline decoupling + stack render，每一步都是 engineering 细节，叠起来把 sim data 生成成本压到 $0.003/episode。

3. **Diversity > Quantity**。Ablation 证明 PnP 单纯堆量没用，关键是 task/skill/action diversity。这个 insight 对未来 sim data 生成有指导意义。

4. **Sim-to-real gap 不在 visual fidelity 而在 interface alignment**。只要 camera view 和 action space 对齐，coarse alignment 就够。

5. **可复现**。数据 + pipeline 都开源，8 张 4090 就能跑 200+ hours/day。如果 Karpathy 你想自己 reproduce 或者改进，门槛已经压到很低了。

GitHub: https://github.com/OpenDataLab/InternData-A1
Dataset: https://opendatalab.com/OpenDataLab/InternData-A1
ArXiv: https://arxiv.org/abs/2507.22022

这篇 paper 对 build intuition 的价值在于：**它第一次把 sim data 推到了跟 real data 平起平坐的位置，告诉我们 sim data 的 ceiling 在哪，以及还需要往哪个方向推**。如果 Karpathy 你对某一块（flow matching action expert 实现、CuRobo trajectory interpolation、VBD garment 仿真细节、或者 sim-to-real 的 interface alignment 具体怎么做）想更深入聊，可以告诉我，我再展开。

---

# InternData-A1: 纯 Synthetic Data 匹配 π-dataset 的突破性证据

Karpathy 你好，这篇 paper 我读得很兴奋。它在 robotics synthetic data 这个方向上立了一个里程碑：**第一次证明 pure simulation data 可以 match 最强 real-robot dataset（π-dataset）做 VLA pre-training 的效果**。下面我尽量把 technical details 撑开讲，希望能 build 起你关于这个方向的 intuition。

---

## 1. 论文核心论点（Thesis）

**Claim**: 一个 VLA model 如果在足够 high-fidelity、足够 diverse、足够大规模的 simulation data 上 pre-train，它**可以匹配** (而不是接近) 在 closed-source π-dataset 上 pre-train 的 official π0 model。

**证据**:
- 49 simulation tasks (RoboTwin 2.0 benchmark): Easy mode +5%, Hard mode +6.5%
- 9 real-world tasks (5 regular + 4 dexterous): comparable, 平均领先 +6.2% (regular tasks)
- 10 个 task 实现 zero-shot sim-to-real (无 real data fine-tune)，平均 success rate > 50%
- Sim-to-real 数据效率：1,600 sim episodes ≈ 200 real episodes（8:1 ratio，简单任务甚至 1:1）

Paper link: https://arxiv.org/abs/2507.22022 (InternData-A1)
Homepage: https://opendatalab.com/OpenDataLab/InternData-A1
GitHub: https://github.com/OpenDataLab/InternData-A1
π0 paper: https://arxiv.org/abs/2410.24164

---

## 2. Dataset 规模与组成（构建 intuition）

### 规模对照（vs prior synthetic datasets）

| Dataset | Traj. | Hours | Skill | Task | Scene | Embodiment | Fluid | Deform. | Open |
|---|---|---|---|---|---|---|---|---|---|
| MimicGen | 50k | — | 1 | 18 | 1 | 4 | × | × | ✓ |
| RoboCasa | 77k | — | 8 | 100 | 120 | 2 | × | × | ✓ |
| GraspVLA | 10M | — | 1 | 1 | 1 | 1 | × | × | × |
| RoboTwin 2.0 | 100k | — | — | 50 | 1 | 5 | × | × | ✓ |
| InternVLA-M1 | 244k | — | 2 | 1 | 1 | 1 | × | × | ✓ |
| **InternData-A1** | **630k** | **7,433** | **18** | **70** | **227** | **4** | **✓** | **✓** | **✓** |

注意 GraspVLA 的 10M trajectories 看上去吓人，但它只有 1 个 task（grasping），实际 skill diversity 极窄。InternData-A1 走的是 **breadth + depth 双重 scaling**。

### Embodiment 分布（按 trajectory 数量）

- **Franka Emika Panda**: 23.3% — tabletop 单臂研究主力
- **AgileX Split Aloha (Piper-100 arms)**: 30.8% — 当前 VLA real-world deployment 的主流双臂
- **ARX Lift-2 (R5a arms)**: 37.8% — 同上
- **AgiBot Genie-1**: 7.9% — 大规模 real data collection factory 用的双臂

**我的 intuition**: 选这 4 个 embodiment 不是偶然。它们覆盖了"实验室单臂" + "工业级双臂 factory" + "开源 deployment-ready 双臂"三种典型场景。Cross-embodiment 的关键是 action space 的统一表示——论文里用 waypoints (6D end-effector pose) 作为 embodiment-agnostic 中间层，让同样的 scripted skill 可以跨 embodiment 复用。

### Object 类型覆盖

- **Rigid**: 3,185 个，107 类，来自 OmniObject3D + Objaverse
- **Articulated**: 321 个，14 类，来自 GRUtopia + GAPartNet + GenSim2 + Infinite Mobility + ArtVIP
- **Deformable (garment)**: 20 个真实衣物用 EinScan Rigel Pro 扫描得到
- **Fluid**: particle-based dynamics + isosurface rendering

**Reference links**:
- OmniObject3D: https://omni3d.github.io/
- Objaverse: https://objaverse.allenai.org/
- GRUtopia: https://github.com/OpenRobotLab/GRUtopia
- GAPartNet: https://arxiv.org/abs/2211.00348
- GenSim2: https://arxiv.org/abs/2402.13061
- ArtVIP: https://arxiv.org/abs/2502.08024
- AnyGrasp: https://arxiv.org/abs/2212.12976

---

## 3. 合成 Pipeline 详解（这是 paper 真正的工程亮点）

Pipeline 分 4 个 stage，关键在于 **fully decoupled + compositional**。

### 3.1 Environment Construction
给定 task description template，retrieve 三个 asset：
- **Robot USD**: 验证过 contact dynamics 稳定
- **Scene**: 来自 GRUtopia 的 GRScenes-100，带 manipulation-area metadata
- **Object**: 带 canonical pose + grasp poses（AnyGrasp 自动生成）

每种 asset 有不同的物理 annotation：
- Articulated object: joint axes + part poses + damping/stiffness
- Deformable: 用 Vertex Block Descent (VBD) simulate
- Fluid: PBD (Position-Based Dynamics) particle 系统

VBD reference: https://arxiv.org/abs/2402.05944 (Vertex Block Descent, SIGGRAPH 2024)

### 3.2 Skill Composition —— 核心抽象

**核心 insight**: 把 skill 定义为 **state → waypoints** 的 mapping，waypoints 是 6D end-effector pose。这样做的妙处在于：

1. **Embodiment-agnostic**: 同一个 `pick` skill 的 waypoint 序列 (pre-grasp, grasp, post-grasp) 可以喂给任何 embodiment 的 motion planner
2. **Decoupled from execution**: 高层逻辑（"先 pick 再 handover 再 place"）和底层 motion planning 完全分离
3. **Compositional**: 用 YAML config 声明式地组合 skill

**Skill 的数学形式**:

$$
\text{skill}: (s_{\text{obj}}, s_{\text{robot}}, \mathcal{C}_{\text{user}}) \rightarrow \{w_i\}_{i=1}^{N}
$$

其中：
- $s_{\text{obj}} \in SE(3) \times \mathcal{J}$: object pose + joint state（articulated object 时 $\mathcal{J}$ 非空）
- $s_{\text{robot}} \in SE(3) \times \mathbb{R}^{n_j}$: base pose + joint configuration
- $\mathcal{C}_{\text{user}}$: user-specified constraints，如 `align_pick_obj_axis = [0,1,0]`（要求 pick axis 对齐 y 轴）
- $w_i \in SE(3) \times \{open, close\}$: 6D target pose + gripper state

比如 `Insert Flower In Vase` 任务里要求 stem 保持 upright，就用 constraint: `align_pick_obj_axis = [0,1,0]`, `align_place_obj_axis = [0,0,1]`, tolerance 10°。

### 3.3 Domain Randomization

随机化覆盖 5 个 axis：

1. **Camera extrinsic**: ±5° rotation, ±5 cm translation
2. **Lighting**: 174 environment maps，temperature + intensity randomize
3. **Object replacement**: 同 category 内 swap
4. **Layout**: tabletop + background 都随机
5. **Contact point**: grasp pose 从 AnyGrasp top-40 high-confidence candidate 中随机选

**Intuition**: 这种 randomization 的目标跟 OpenAI Hand 那篇文章一致——让 policy 学到 **invariant representation**，而非 overfit 到某个 specific sim 的 visual/dynamics。这也是为什么后面 sim-to-real 用 "coarse alignment" 就能 work。

OpenAI Hand reference: https://arxiv.org/abs/1808.00177

### 3.4 Generation & Storage

- **CuRobo** 做 waypoint → dense joint actions 的 interpolation
- 验证 physics simulation，丢弃失败 trajectory
- 渲染成功 trajectory，存为 LeRobot format
- 可选记录: depth, grounding annotations, bounding boxes

CuRobo 是 NVIDIA 的 GPU-accelerated motion planner:
$$
\pi_{\text{motion}}: (w_i, w_{i+1}, \text{robot}) \rightarrow \{q_k\}_{k=0}^{K}
$$
其中 $q_k \in \mathbb{R}^{n_j}$ 是 joint configuration，用 minimum-jerk interpolation + collision-free 优化。

CuRobo: https://arxiv.org/abs/2310.17274
LeRobot: https://github.com/huggingface/lerobot

---

## 4. Framework 优化（这部分对实操者最有价值）

传统 sim data pipeline 的瓶颈是 **trajectory planning 和 visual rendering 单 stage 串行**。问题：

1. **失败率随 task complexity 上升**：planning 失败的 trajectory 不需要渲染，但单 stage 架构仍会产生冗余渲染
2. **Compute type mismatch**: planning 是 CPU-bound 串行，rendering 是 GPU-bound 并行，串行执行导致 hardware utilization 差

论文的优化方案：

### 4.1 Stage decoupling + pipelined architecture
把 Planner 和 Renderer 拆成两个独立 process，建立 pipeline。

### 4.2 Dynamic resource scheduling
针对不同 task 的 heterogeneous time-cost ratio，Planner 和 Renderer 内部都做 batch parallelism + dynamic scheduling。

### 4.3 Stack Render
把多个 scene 的渲染 request 堆叠一起送 GPU，提高 GPU utilization。这个 trick 我没见过详细描述，推测是把多个 trajectory 的不同 timestep batch 成一个 batched rendering call。

### 4.4 Cluster stability
- **Balancer module**: load distribution
- **Supervisor module**: monitoring + control

**结果**: 2-3× end-to-end speedup，**8× RTX 4090 GPU 上 209.7 hours robot data per day**，**< $0.003 per episode**。

**我的计算**:
- 630k episodes × $0.003 = ~$1,890 total cost
- 7,433 hours / 209.7 hours-per-day ≈ 35.4 GPU-days ≈ 4.4 天 on 8× RTX 4090

这个 throughput 对学术界简直是 game-changer。对比 Open X-Embodiment 那种 real data collection 成本，sim 路线在这里第一次展现出压倒性优势。

---

## 5. π0 架构回顾（为了 context）

π0 由两部分组成：

1. **PaliGemma** (vision-language backbone): 3B 参数，输入 image(s) + text prompt
2. **Flow-matching action expert**: 在 PaliGemma 的 representation 之上，用 flow matching 生成 continuous action chunks

### Flow Matching 公式细节

给定：
- $q(x_1)$: data distribution over action chunks（real robot trajectories）
- $p(x_0) = \mathcal{N}(0, I)$: 简单 base distribution
- Conditional path: $x_t = (1-t) \cdot x_0 + t \cdot x_1$, $t \in [0,1]$
- Conditional vector field: $u_t(x_t | x_0, x_1) = x_1 - x_0$

训练目标（Conditional Flow Matching, OT-CFM）:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim U[0,1], x_0 \sim \mathcal{N}, x_1 \sim q} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]
$$

其中：
- $v_\theta$: 神经网络学的 vector field（conditioned on PaliGemma features $c$）
- $x_t = (1-t) x_0 + t x_1$ 是 linear interpolation 的中间 state
- $x_1 \in \mathbb{R}^{H \times n_j}$: action chunk（horizon $H$ 个 future action，每个 $n_j$ 维）
- $t \in [0,1]$: 时间，0 是 base distribution（Gaussian），1 是 target action distribution

Inference 时从 $\mathcal{N}(0,I)$ 采样 $x_0$，用 ODE $\frac{dx_t}{dt} = v_\theta(x_t, t, c)$ 积分到 $t=1$ 得到 action chunk。

Flow Matching reference: https://arxiv.org/abs/2210.02747 (Lipman et al., ICLR 2023)
π0 paper: https://arxiv.org/abs/2410.24164
PaliGemma: https://arxiv.org/abs/2407.07726

### InternData-A1 训练设置

| Hyperparameter | Pre-training | Fine-tuning |
|---|---|---|
| Batch Size (Total) | 512 | 128 |
| Learning Rate | 5e-5 | 2.5e-5 |
| LR Schedule | Constant | Cosine Decay |
| Training Steps | 680k | 30k (regular) / 100k (dexterous) |
| Hardware | 32× A100 GPU | 8× A100 GPU |

---

## 6. 实验：用数据说话

### 6.1 vs π-dataset（49 sim tasks, RoboTwin 2.0）

| Method | Hanging Mug (E/H) | Lift Pot (E/H) | Pick Dual Bottles (E/H) | Place Object Stand (E/H) | Shake Bottle (E/H) | Turn Switch (E/H) | **Avg. 49 Tasks (E/H)** |
|---|---|---|---|---|---|---|---|
| π0 (Scratch) | 5/2 | 26.5/0 | 1.5/0.5 | 9/0 | 55/2 | 9/9.5 | 23.5/2.5 |
| π0 (official) | 11/6.5 | 17/1.5 | 58/16 | 43/14 | 96.5/55 | 27.5/30 | 55/20 |
| **π0 (InternData-A1)** | **24.5/20** | **63.5/2.5** | **62/19** | **48.5/29.5** | **98/64** | **40.5/32.5** | **60/26.5** |

**+5% (Easy), +6.5% (Hard)**。Hard mode 领先更明显很关键，因为 hard mode 是 cluttered + domain randomization 评测，而 fine-tune 用的是 clean data。这说明 **InternData-A1 的 domain randomization 学到的 robustness 在 clean fine-tune 之后仍然保留**——这点跟 GCD (Google DeepMind) 那波 sim-to-real 的发现一致。

### 6.2 vs Open-source datasets

| Dataset | Domain | 49 Sim Tasks (E/H) | Sort Rubbish (real) | Pass Bottle (real) |
|---|---|---|---|---|
| OXE | Real | 32.5/11 | 40.0 | 36.7 |
| Agibot World | Real | 52.5/12 | 53.3 | 56.7 |
| RoboCasa | Sim | 50.0/11 | 23.3 | 13.3 |
| **InternData-A1** | **Sim** | **60/26.5** | **90** | **60** |

**关键发现**: RoboCasa 在 sim 评测里跟 InternData-A1 只差 10%，但 real-world 评测差 57.7%。说明 RoboCasa 的 visual fidelity 不够，sim-to-real gap 大。InternData-A1 的 photorealistic rendering + 大数据量是这里的关键。

RoboCasa reference: https://arxiv.org/abs/2406.02523
OXE: https://arxiv.org/abs/2310.08896
Agibot World: https://arxiv.org/abs/2507.08845

### 6.3 Sim-to-Real 数据效率（4 tasks 深入研究）

| Task | Real-Data 需求 | Sim-Data 等效量 | Sim:Real Ratio |
|---|---|---|---|
| Sort Rubbish | 200 episodes | 200 sim episodes | 1:1 |
| Wipe Stain | 200 episodes | 200 sim episodes | 1:1 |
| Flip Package | 200 episodes | 1,600 sim episodes | 8:1 |
| Instructional Pick | 200 episodes | 1,600 sim episodes | 8:1 |

**关键 insight**: 
- 简单 pick-and-place + move 任务，sim 已接近 real 的 data efficiency
- 复杂 task（dynamic object + language grounding），sim 需要 8× more data
- 但 8× sim data 的 cost 远低于 8× real data（后者需要 teleop 人力）
- **Coarse alignment 足够**——背景/光照/纹理无需 exact match，只要 camera view 和 joint action space 对齐

这点很 powerful。意味着 sim-to-real 的关键 bottleneck **不是 visual fidelity 本身，而是 observation/action interface 的 alignment**。这跟 earlier sim-to-real 工作的结论高度一致。

### 6.4 Zero-shot Sim-to-Real（10 tasks total）

6 个额外 task 用 500 sim episodes post-train 后直接 sim-to-real（无 real data）：

| Task | Success Rate |
|---|---|
| Make Sandwich | 50% |
| Pack | 50% |
| Close Box | 63% |
| Close Microwave | 87% |
| Sweep | 60% |
| Handover | 57% |

**这是 GraspVLA 之后第一次证明 complex multi-skill tasks 也能 zero-shot sim-to-real**。GraspVLA 只做了 grasping。

GraspVLA: https://arxiv.org/abs/2502.20300

---

## 7. Ablation：什么真正驱动了 pre-training 效果？

把 InternData-A1 分成 4 个 component：
- **PnP** (Pick-and-Place): 30.61%
- **Art** (Articulation): 11.67%
- **Base** (<3 skills beyond PnP): 35.95%
- **Long** (≥3 skills, long-horizon): 21.77%

Ablation：训练 0.5 epoch，去掉各 component 看 RoboTwin 评测：

| Variant | Easy / Hard |
|---|---|
| Full | 58.0 / 25.0 |
| w/o PnP | 57.0 / 22.5 |
| w/o Art | 55.5 / 19.5 |
| w/o Base | 52.5 / 20.5 |
| w/o Long | 54.0 / 19.0 |

**两个核心发现**:

1. **PnP 占 30% 但去掉影响最小**。说明单纯 pick-and-place 数据对 VLA pre-training 价值有限。这解释了为什么 GraspVLA (10M pick-only) 无法泛化。

2. **Long-horizon 和 Base (multi-skill) 去掉影响大**。即便 Long 只占 21.77%，去掉它降了 4% / 6%。说明 **trajectory diversity** 比 task count 更重要。

3. **Articulation 数据少但影响大**。只占 11.67%，去掉降 2.5% / 5.5%。Hypothesis: articulated manipulation 涉及 diverse action spaces（不同 joint geometries + distorted arm configurations），对 action prior 学习贡献大。

**我的 hypothesis**: VLA pre-training 的 effective signal 不是 "more pick-and-place samples"，而是 "more diverse action distributions"。Long-horizon 和 articulation 自然提供了这种 diversity，而单纯 pick-and-place 即使 sample 量大，action distribution 也比较 narrow。

---

## 8. 我对这篇 paper 的思考 & 联想

### 8.1 为什么 sim data 终于 work 了？

之前 sim-to-real 的 max success cases 多停留在 grasping。这次能 work 到 70 tasks，我归纳几个条件：

1. **Photorealistic rendering**: 不只是"看起来像"，是让 VLA 学到的 visual feature 在 real domain 仍然 useful
2. **Diverse atomic skills + composition**: 让 model 学到 transferable action priors
3. **Domain randomization**: 让 model 学到 invariant representation
4. **Scale**: 630k trajectories + 7,433 hours，达到 critical mass
5. **Embodiment diversity**: 让 model 不 overfit 到单一 robot

### 8.2 跟其他相关工作的脉络

**Sim-only lineage**:
- GraspVLA (10M grasping trajectories): 证明 sim data 规模可行，但 task narrow
- InternData-A1: 证明 sim data 可以 diverse + match real data

**Hybrid sim+real lineage**:
- GR00T N1 (NVIDIA): https://arxiv.org/abs/2503.14734
- π0.5 (Physical Intelligence): https://arxiv.org/abs/2504.16054

**Real-data-only lineage**:
- RT-1/2: https://arxiv.org/abs/2212.06817
- OpenVLA: https://openvla.github.io/

我的 sense：sim-only 在 pre-train 阶段已经站住脚，real data 的角色会逐渐转移到 fine-tune 和 evaluation。Long-term 看 hybrid 应该还是最优，但 sim 的 scaling 成本优势太大。

### 8.3 Limitations 的诚实评估

作者承认的 limitation：**高度 dexterous tasks（tie shoelaces, thread a needle）sim 还做不到**。这背后是 physics simulator 的精度问题——deformable object + tight contact + friction 的耦合在当前 PBD/finite element 模型下误差太大。

我推测还会有的 limitation（paper 没说）：
- **Tactile-heavy tasks**: 没有触觉 sensor，sim 里强行做 tactile feedback 都是 fake 的
- **Multi-step contact dynamics**: 比如插拔 USB 这种，依赖 sub-mm precision + compliance
- **Long-horizon planning over novel objects**: sim 里的 object pool 仍 finite

### 8.4 我直觉上觉得应该探索的方向

1. **Procedural object generation**: 不依赖 scanned assets，procedurally generate object with diverse physics。Infinite Mobility (https://arxiv.org/abs/2503.14314) 是这个方向。

2. **Neural physics simulator**: 当前 PBD/FEM 仿真 fidelity 上限有限。如果用 learned physics surrogate，可能 simulate 更 dexterous tasks。

3. **Sim-text co-training**: 把 sim trajectory + 互联网 instructional video + web VQA 三者 co-train，让 VLA 同时有 "action prior" + "semantic grounding" + "long-horizon planning"。π0.5 已经在往这走。

4. **Closed-loop sim evaluation**: 当前评测是 open-loop action chunks，应该加 closed-loop rollout evaluation（让 sim model 在 sim 里执行并 measure success）。

5. **Active sim-to-real**: 用 real-world failure cases 反过来 generate targeted sim data，iteratively close the gap。

---

## 9. 几个我好奇但 paper 没说清的点

1. **Stack Render 具体怎么实现**？是 batched rendering call 还是别的？这个 trick 在其他 sim pipeline 里能复用吗？

2. **680k iteration 为什么选这个数**？跟 official π0 的 700k 对齐是为了公平比较，但 680k 之后的 scaling law 是什么样的？

3. **Action chunk length** $H$ 是多少？π0 paper 里是 50，这里沿用了吗？

4. **Failure recovery 数据怎么生成**？Table 1 说 InternData-A1 有 failure recovery，但 pipeline 看上去只 generate 成功 trajectory。是不是 post-hoc 在成功 trajectory 上 inject perturbation？

5. **不同 embodiment 的 action space normalization** 怎么做的？PaliGemma 输入是 image+text，但 action expert 怎么处理 cross-embodiment 的不同 joint dim？

---

## 10. 总结：对 Karpathy 你可能关心的 takeaway

1. **Synthetic data 在 VLA pre-training 上终于 "立住了"**：630k sim trajectories 可以匹配 closed-source π-dataset。这是 sim data 第一次在 scale + diversity + fidelity 三个 axis 都达到 critical mass。

2. **Pipeline engineering 比 model architecture 更 critical**：CuRobo + VBD + AnyGrasp + domain randomization + pipeline decoupling + stack render，每一步都是 engineering 细节，但叠起来让 sim data 生成成本压到 $0.003/episode。

3. **Diversity > Quantity**: Ablation 证明 PnP 单纯堆量没用，关键是 task/skill/action diversity。这个 insight 对未来 sim data 生成有指导意义。

4. **Sim-to-real gap 不在 visual fidelity 而在 interface alignment**：只要 camera view 和 action space 对齐，coarse alignment 就够。这跟 OpenAI Hand 那波 conclusion 一致，但这次扩展到了 70 tasks。

5. **可复现**: 数据 + pipeline 都开源。如果 Karpathy 你想自己 reproduce 或者改进，门槛已经压到很低了——8 张 4090 就能跑 200+ hours/day。

GitHub: https://github.com/OpenDataLab/InternData-A1  
Dataset: https://opendatalab.com/OpenDataLab/InternData-A1  
ArXiv: https://arxiv.org/abs/2507.22022

希望这些细节能帮到你 build 起对这个领域的 intuition。如果你对某一块（比如 flow matching action expert 的具体实现、CuRobo 的 trajectory interpolation、VBD 在 garment 上的细节）想更深入聊，可以告诉我，我再展开。
