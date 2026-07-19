---
source_pdf: A Careful Examination of Large Behavior Models.pdf
paper_sha256: c9ba44579974208aa43d60fce2e0294c0fee1f1b8d604abd64613568f9c5a102
processed_at: '2026-07-17T09:37:27-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# TRI LBM Paper 深度解析

## 1. 核心问题与动机

这篇 paper 来自 Toyota Research Institute (TRI)， tackling 一个在 robot learning 领域极其重要但常被回避的问题：**multitask pretraining 到底有没有用，能多用多少数据？** 在 LLM 和 vision foundation model 大爆发的背景下，robotics 社区也在追随 scaling hypothesis，涌现出 RT-1/RT-2、Octo、π0、OpenVLA、Gemini Robotics 等大量 generalist manipulation policy 工作。但 TRI 团队指出一个尖锐的现实——**大部分论文的 evaluation 不够 rigorous，统计功效不足，难以区分真实信号和噪声**。

paper 的 project page: https://toyotaresearchinstitute.github.io/lbm1/

## 2. 方法论：Diffusion Policy + DiT

### 2.1 生成模型选择

LBM 采用 DDIM (Denoising Diffusion Implicit Models) 作为 action generator。给定 K 个 denoising steps，从高斯噪声 $A_t^K \sim \mathcal{N}(0, I)$ 出发，逐步去噪到连续 action $A_t^0$。条件化的 DDIM update 公式：

$$A_t^{k-1} = \alpha \left( A_t^k - \gamma \cdot \epsilon_\theta(O_t, A_t^k, k) \right) \tag{1}$$

变量解释：
- $A_t^k$: 在 time step $t$、denoising step $k$ 处的 noisy action sequence
- $O_t$: observation（visual + proprioceptive + language）
- $k$: diffusion timestep（上标，表示去噪迭代深度）
- $\alpha, \gamma$: 由 noise schedule 决定的标量参数，随 $k$ 变化
- $\epsilon_\theta$: noise prediction network，参数为 $\theta$

训练目标是最朴素的 DDPM MSE loss：

$$\mathcal{L}(\theta) = \|\epsilon_k - \epsilon_\theta(O_t, A_t^k, k)\|_2^2 \tag{2}$$

其中 $\epsilon_k$ 是添加到 clean action $A_t^0$ 上的 step-dependent Gaussian noise，network 学习从 noisy action 反推 noise。这个范式直接继承自 Chi et al. 的 Diffusion Policy (https://diffusion-policy.cs.columbia.edu/)。

### 2.2 网络架构

LBM 用 Diffusion Transformer (DiT, Peebles & Xie, https://arxiv.org/abs/2212.09748) 替换原版 Diffusion Policy 中的 U-Net backbone。具体配置：

- **Vision encoder**: CLIP ViT-B/16 (https://openai.com/research/clip)，取 CLS token 输出，训练时 fine-tune
- **Language encoder**: CLIP text encoder，frozen，但上面加一个可训练的 projection layer
- **Proprioception**: end-effector pose（相对 station base frame 和相对另一只手）、gripper width
- **Diffusion timestep embedding**: sinusoidal positional embedding + 2-layer MLP，通过 adaptive LayerNorm (adaLN) 注入
- **DiT 主体**: 8 个 DiT block，embedding dim 768
- **Observation history**: $n_{obs}=2$（两帧历史）
- **Action prediction horizon**: $n_{horizon}=15$（预测 16 个 timestep 的 action，每个 20 维，总 output size $A_t = 320$）

观察特征拼接后 size 为 6,732。Deployment 时 policy 以 10 Hz 运行，执行前 8 个预测 timestep 后 replan（receding horizon control 思路）。

Action space 设计上有意思：使用 6D rotation representation（rotation matrix 的 top 2 rows），避免了 quaternion 的 discontinuity 问题，也避免了 Euler 角的 gimbal lock。

### 2.3 Data Normalization

这部分有个值得深究的设计。对每个 feature 维度、每个 timestep 独立计算 normalization：

$$y_i = \min\left(\max\left(-1.5, 2\frac{x_i - x^{0.02}}{x^{0.98} - x^{0.02}} - 1\right), 1.5\right) \tag{3}$$

变量解释：
- $x_i \in \mathcal{D}$: 原始数据样本
- $x^{0.02}, x^{0.98}$: 该 feature 在该 timestep 上的第 2 和第 98 百分位
- $y_i$: normalized value，范围 clip 到 $[-1.5, 1.5]$

这种 percentile-based normalization 把 [2%, 98%] 区间映射到 $[-1, 1]$，保留少量 outlier 但保留中心高密度区域的分辨率。**关键 intuition**: 因为 action 是相对于当前 observation 表示的，越远的 future action spread 越大，per-timestep 独立 normalize 能更好地保留 near-future action 的精度。

paper 还坦白了一个尴尬的 bug——pretraining 时部分 datagram 用了错误 data source 的 normalization 参数。他们做了一个小规模 ablation (Fig. S21) 评估影响，发现 nominal 条件下差异不大，但 distribution shift 下 corrected 版本在 4/16 tasks 和 aggregate 上更好。这种 honest reporting 在 robotics 论文中相当罕见。

## 3. 数据集：Ramen

预训练数据集 Ramen 总计约 **1695 小时** demonstration，分两大部分：

### 3.1 TRI-Ramen（545 小时）
- **TRI-Ramen-Real**: 468 小时，362 tasks，46,063 demos，9 个 hardware stations 收集
- **TRI-Ramen-Sim**: 45 小时，41 tasks，7,348 demos，2 个 simulation stations
- **TRI-Ramen-UMI**: 32 小时，129 tasks，10,851 demos，用 Universal Manipulation Interface (UMI, https://universal-manipulation-interface.github.io/) 在 in-the-wild 环境收集

### 3.2 OXE-Ramen（1150 小时）
来自 Open X-Embodiment (https://robotics-transformer-x.github.io/) 的子集，基于 object/environment diversity 和 episode 数量筛选。Unimanual 数据通过 zero-padding 和随机左右翻转转成 bimanual 格式。

Batch balancing 用经验权重（Table S7），lbm_real 权重 0.5，lbm_sim 0.25，lbm_umi 0.05，等。

## 4. Evaluation Protocol：这是 paper 真正的杀手锏

### 4.1 Blind A/B Testing

所有 real-world evaluation 都采用 blind testing——evaluator 不知道被测的是哪个 policy。每次 rollout 顺序随机化，每个 "bundle" 包含所有要对比的 policy，对应同一个 initial condition，顺序随机。这样能控制 lighting drift、operator fatigue 等时变干扰。

### 4.2 Initial Condition 控制

用 image overlay 让 operator 把真实场景对齐到 reference image（来自 simulation 或 pre-recorded real photo）。用 homographic projection 把不同 station 的图像 canonicalize。这种 setup 保证了对比的公平性，但代价是单次 rollout 耗时增加。

### 4.3 Rubrics 和 Task Completion

仅 success rate 不能反映 policy 的真实能力（"差一点就成功"和"完全不动"在天壤之别）。paper 设计了 milestone-based rubrics：

- **Real-world**: 人工填写 binary yes/no 问题（如 "robot grasped the apple"）
- **Simulation**: 基于仿真状态的 predicates 自动计算

Task Completion (TC) = 成功完成的 milestone 数 / 总 milestone 数。比如 CutAppleInSlices 有 14 个 milestones，从 "grasped the apple" 到 "placed knife in utensil crock"。

### 4.4 QA 流程

对约 27% 的 real-world rollouts 做二次审查，success rate discrepancy 2.31%，overall rubric question discrepancy 6.25%。这个数字给整个 evaluation 的可信度提供了 baseline。

### 4.5 统计方法

**对于 binary success/failure**: 用 Bayesian posterior，uniform Beta prior $\text{Beta}(1,1)$，violin plot 展示后验分布。这是 Kress-Gazit et al. (https://arxiv.org/abs/2409.09491) 推荐的做法。

**对于 multi-policy comparison**: 
- 共 $K(K-1)/2$ 个 pairwise tests，Bonferroni 校正维持全局 95% confidence
- 用 Compact Letter Display (CLD, Piepho 2004) 可视化——共享字母表示不可区分，不同字母表示统计显著分离
- Sequential hypothesis testing (Lai 1988, https://projecteuclid.org/journals/annals-of-statistics) 用于 small sample binary data，严格 Type-I error 控制
- Welch's t-test 用于 task completion（categorical data）

### 4.6 实验规模

- **Real-world**: 1,800 次 blind A/B trials，每 task 每 policy 每 condition 50 rollouts
- **Simulation**: 47,000+ rollouts，每 task 每 policy 每 condition 200 rollouts

这种规模在 robotics 实证工作里非常罕见。Agarwal et al. (NeurIPS 2021, "Deep RL at the Edge of the Statistical Precipice", https://arxiv.org/abs/2105.09551) 论证过小样本统计在 RL 评估中的危险性，TRI 这套 protocol 直接回应了这个痛点。

## 5. 核心实验结果

### 5.1 "Seen" Tasks（预训练时见过的任务）

**主要发现 1**: Finetuned LBM 在 aggregate 上统计显著优于 single-task baseline，在 nominal 和 distribution shift 条件下均成立。

具体数字（sim 16 个 tasks）：
- Nominal: finetuned LBM 在 3/16 tasks 显著优于 single-task，剩余 13/16 不可区分
- Distribution shift: finetuned LBM 在 10/16 tasks 显著优于 single-task

**主要发现 2**: Pretrained-only LBM（不 finetune）在 seen tasks 上 success rate > 0，aggregate 上与 single-task 在 sim 中不可区分，但在 real-world 上略差。作者归因于：
1. Real-world 任务更难
2. Data normalization bug
3. Language steering brittleness（CLIP 文本编码器容量有限）

### 5.2 "Unseen" Tasks（预训练时未见过的任务）

这是最 compelling 的实验。设计了 5 个 long-horizon real-world tasks：

1. **BikeRotorInstall**: 装自行车 rotor，需要 bimanual coordination + 工具使用
2. **CutAppleInSlices**: 用 corer 去核、刀切半、切 3 片以上、擦刀、收刀，最长可达 3 分钟
3. **SetBreakfastTable**: 开柜门，按顺序放置 9 个物件，cereal 要倒入 bowl
4. **CleanLitterBox**: 双手协作清理猫砂
5. **ClearKitchenCounter**: 工具收纳、清洁 cutting board、扫垃圾

**主要发现 3**: Finetuned LBM 在 task completion 上统计显著优于 single-task baseline（4/5 real-world tasks，4/5 sim tasks）。Success rate 上虽然数字低，但在能分离的 tasks 上都是 LBM 胜出。

**主要发现 4 — Data Efficiency**: LBM finetune 只用 15% 数据，就统计显著优于用 100% 数据训练的 single-task baseline（SetBreakfastTable, Fig. 5）。Aggregate sim 数据显示 LBM 达到 single-task baseline 等价性能只需 < 30% 数据，即 **3-5x data efficiency**。

### 5.3 Pretraining Scaling Laws

Fig. 7 的 scaling 实验非常有意思。构造 4 个预训练数据集：
1. Full Ramen (TRI-Ramen + OXE-Ramen)
2. TRI-Ramen only
3. 50% of TRI-Ramen tasks
4. 25% of TRI-Ramen tasks

然后用不同 fraction (0%, 15%, 50%, 100%) 的 task-specific fine-tuning data 测试。关键观察：

- 用 15% finetune data 时，5 个模型全部统计显著分离，performance 随 pretraining data 单调上升
- Pretraining 和 fine-tuning data 之间存在 tradeoff：finetune data 越少，pretraining diversity 越重要
- **没有看到性能不连续或 sharp inflection point**——这是 robotics scaling 的重要负面发现，意味着不存在"emergent ability"那种相变

这个 finding 对应 LLM 里的 scaling laws debate (Kaplan et al. 2020, Hoffmann et al. 2022 "Chinchilla")。Robotics 由于数据成本极高，研究者特别关心"多少数据够用"。TRI 的曲线显示在他们的 scale 范围内，仍是平滑提升。

## 6. 一些值得深究的细节

### 6.1 Time-to-Motion 问题

Section X 和 Fig. S13 揭示了一个微妙现象：pretrained LBM 和 finetuned LBM 在 simulation 中常常 rollout 开始后久久不动（甚至 timeout）。但 single-task policy 没这个问题。更奇怪的是，在 distribution shift 条件下，LBM 反而更快开始动。

作者的 hypothesis：低 motion frames 在 pretraining data 中比例较高（sim 11.7%, real 3.3%）。如果 pretraining 时 filter 掉这些 frames，LBM 启动更快，但语言可 steer 能力下降——会 commit 到错误任务。这个 tradeoff 暴露了 multitask model 的一个根本张力：**等待 language 信号 disambiguate 与及时行动之间的权衡**。

### 6.2 Sim-to-Real Performance Reversal

PutKiwiInCenterOfTable 和 TurnMugRightsideUp 两个 task，real-world success rate 高于 simulation。Table S6 显示，如果对 real rollouts 强制施加 simulation timeout，real performance 接近 sim。原因是 real-world operator 会等机器人"试探"和"恢复"，而 sim 是无条件 fixed timeout。这个 finding 暴露了 sim-real comparison 中一个被低估的 confound。

### 6.3 Dataset Filtering 的反直觉效果

Section XII 揭示：filter 低 motion frames 让 single-task 在 sim 上更好，但让 pretrained LBM 的 language steering 变差。作者的解释：低 motion frames 集中在 rollout 开头，此时场景视觉上最 ambiguous（benchmark 设计使然），policy 必须依靠语言来 disambiguate。Filter 掉这些 frames 等于减少了"听语言信号"的训练压力。这个解释很漂亮，说明了 data filtering 不只是 data cleaning，而是隐式地 shaping policy 的 attention pattern。

## 7. Limitations 和 Field-level Caveats

作者自己点出的几个关键 limitations：

1. **未对训练 stochasticity 建模**：
$$p(\text{success}|\text{dataset}) = \int p_{\text{eval}}(\text{success}|w) p_{\text{train}}(w|\text{dataset}) dw$$
其中 $w$ 是 policy weights。他们的 CI 只覆盖第一项 $p_{\text{eval}}$，未对 $p_{\text{train}}$ 建模。这等于假设训练 deterministic，而实际上 SGD seed 会引入显著方差。这个 gap 在 (https://arxiv.org/abs/2105.09551) 中详细讨论过。

2. **Language encoder 容量有限**: 用 CLIP text encoder 比较小，language steerability 不足。作者提到更大的 VLA prototype 有改善迹象，但未做严格对比。

3. **许多 robotics 论文可能在测统计噪声**: 由于 effect size 小、实验噪声大，sample size 不够时极易 false positive。

4. **Design decisions（如 data normalization）的影响常常超过 architecture/algorithm 改动**——这是 field-level 的反思，呼应了 machine learning reproducibility crisis 的讨论。

## 8. 我的 Intuition 和相关联想

### 8.1 与 LLM Pretraining 的类比

LBM 的 finetune superiority over single-task training，与 LLM 里 "pretrain-then-finetune" 优于 "train from scratch" 完全对应。但有一个重要差异：**robotics data 的 marginal cost 远高于 text data**。1,700 小时 demonstration 在 robotics 算大规模，但对应 token 数远不及 LLM 的 pretraining corpus。这意味着 LBM 的 scaling curve 还远未饱和，未来的瓶颈在 data collection infrastructure。

值得对比的工作：
- **OpenVLA** (https://openvla.github.io/): 970k OpenX demos + 直接 VLA transformer
- **π0 / π0.5** (https://www.physicalintelligence.company/): flow matching VLA
- **Octo** (https://octo-models.github.io/): 800k OpenX demos transformer
- **GR00T N1** (https://arxiv.org/abs/2503.14734): NVIDIA humanoid foundation model
- **Gemini Robotics** (https://arxiv.org/abs/2503.20020): DeepMind 的 VLA

TRI 的 LBM 在架构上更保守（DiT + CLIP，没有大规模 LLM backbone），但 evaluation 上更严谨。

### 8.2 Diffusion vs. Autoregressive Token Prediction

LBM 选择 diffusion 而非 tokenized action 的 autoregressive prediction（如 RT-2, OpenVLA 的做法）。Diffusion 的优势在于：
- 连续 action space 的精度
- Multi-modality 自然支持
- 不需要 action tokenizer 设计

但代价是 inference 速度（多次 denoising step）。LBM 在 10 Hz 运行，已经能满足 bimanual tabletop manipulation 的需求。这个 tradeoff 在 Brooks et al. (https://arxiv.org/abs/2410.24164) 的 π0 里用 flow matching 部分缓解。

### 8.3 Sim-Real Co-training

paper 提到用 sim data 主要用于 evaluation，但 pretraining 时也 co-train sim 和 real。这个领域最近有重要进展：
- Wei et al. (https://arxiv.org/abs/2503.22634): sim-and-real co-training empirical analysis
- Maddukuri et al. (https://arxiv.org/abs/2503.24361): sim-and-real co-training recipe
- SIMPLER (https://arxiv.org/abs/2405.05941): real2sim evaluation framework

LBM 的 sim-real performance reversal 提醒我们：sim 的"完美可控"也是它的局限——operator judgment 包含大量 implicit domain knowledge。

### 8.4 Evaluation Methodology 的深远影响

TRI 这套 protocol 给 robotics 社区树立了一个标杆。值得对比的工作：

- **AutoEval** (https://arxiv.org/abs/2503.24278): autonomous evaluation
- **Snyder et al.** (https://arxiv.org/abs/2503.10966): near-optimal stopping for policy comparison
- **BEHAVIOR** (https://arxiv.org/abs/2108.03332): benchmark for everyday household
- **RoboCasa** (https://arxiv.org/abs/2406.02523): large-scale everyday task simulation

如果社区普遍采用 blind A/B + Bayesian posterior + CLD + sequential testing，许多"显著提升"的论文可能会失去显著性。这种 methodological rigor 对 field 的长期健康至关重要，类似 ImageNet 之于 CV、GLUE 之于 NLP 的标准化作用。

### 8.5 Action Representation 的细节

paper 用 6D rotation representation（rotation matrix 的 top 2 rows），这是 Zhou et al. 2019 "On the Continuity of Rotation Representations in Neural Networks" (https://arxiv.org/abs/1812.07035) 的推荐做法。这个细节很容易被忽视，但对 manipulation 的稳定性影响巨大——quaternion 在 antipodal 点不连续，Euler 角有 gimbal lock，直接预测 9D rotation matrix 又过参数化。

Action 表示用 relative-to-current-observation 的方式（来自 UMI paper），让 policy 学 relative motion 而非 absolute pose，这增强了 cross-embodiment 的可能性，也使 action distribution 在不同初始条件下更稳定。

### 8.6 关于 Pretraining-Finetuning Tradeoff 的深层思考

Fig. 7 揭示的 tradeoff（finetune data 越少，pretraining diversity 越关键）让我联想到 LLM 里的 "data mix" 研究。当 finetune data 充足时，pretraining 的 marginal benefit 下降；当 finetune data 稀缺时，pretraining 提供的 inductive bias 成为 performance bottleneck。

这对工业部署有直接含义：
- 高频重复任务（如工厂 pick-and-place）：少 pretraining 也够，多收 task data
- 长尾 diverse 任务（如家庭服务）：必须 large-scale diverse pretraining

这也是 OpenVLA、π0、GR00T N1 这些项目押注的方向——大模型 + 大数据，希望覆盖足够广的任务 manifold。

### 8.7 未解决的开放问题

paper 没有但我觉得重要的几个方向：

1. **Continual learning**: LBM finetune 后是否会 catastrophic forget 其他 task？未测
2. **Compositionality**: 能否通过 language compose 出 pretraining 时未见的 skill sequence？
3. **Safety guarantees**: distribution shift 下 policy 行为是否可预测 bounded？
4. **Causal understanding**: LBM 是否有 object affordance 的隐式 model，还是纯 pattern matching？
5. **Long-horizon planning**: 当前 16-step action horizon 对 truly long-horizon 任务（>1 分钟）够吗？

## 9. 总结性 Intuition

这篇 paper 真正的价值不在于 SOTA 数字——它的架构是已知 component 的组合（DiT + CLIP + Diffusion Policy），数据规模也不是最大。**它的价值在于把 empirical rigor 带回 robotics evaluation**。

关键 take-away：
1. Multitask pretraining 在 robotics 上 work，效果可量化、可统计显著验证
2. Data efficiency 提升 3-5x，distribution shift 下优势放大
3. Scaling 在测试范围内平滑，无相变
4. Pretrain vs. finetune data 之间存在 tradeoff，部署场景应决定数据策略
5. Evaluation protocol 的 cost 不低（1800 real rollouts），但远低于错误结论的代价
6. Architecture 选择的影响力常常被 data normalization 这种 "工程细节" 盖过

对 Karpathy 你可能特别有意思的几点：
- **Pretraining 的本质在 robotics 里仍然成立**——这是一个非平凡 finding，因为 robotics 的 action distribution 比 text token distribution 复杂得多（高维、连续、多模态、embodiment-specific）
- **Smooth scaling without emergent abilities**——和 LLM 的 phase transition 现象形成对比，可能因为 (a) 数据规模不够，(b) 任务 manifold 不如 language 那般有 compositional structure，(c) action space 的几何性质不同
- **Language steering 的 brittleness**——暗示当前 VLA 的 LLM 部分（CLIP encoder）还不够强，"用语言来 disambiguate visually ambiguous scenes"这个能力还需要更大的 language model

一些相关的 paper 可以延伸阅读：
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OpenVLA: https://openvla.github.io/
- π0: https://arxiv.org/abs/2410.24164
- DROID dataset: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Drake (simulation): https://drake.mit.edu/
- UMI: https://universal-manipulation-interface.github.io/
- Kress-Gazit et al. evaluation best practices: https://arxiv.org/abs/2409.09491
- Agarwal et al. statistical precipice: https://arxiv.org/abs/2105.09551

paper 本身写得很诚实，bug（normalization error）、反直觉发现（sim-real reversal, filtering 反效果）、limitations 都和盘托出。这种风格在如今 robotics 论文"刷榜"风气中相当难得。如果社区能广泛采纳 TRI 这套 evaluation protocol，robotics 的 empirical credibility会大幅提升。
