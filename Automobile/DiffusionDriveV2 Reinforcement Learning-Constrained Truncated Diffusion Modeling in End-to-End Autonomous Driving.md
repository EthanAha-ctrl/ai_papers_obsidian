---
source_pdf: DiffusionDriveV2 Reinforcement Learning-Constrained Truncated Diffusion
  Modeling in End-to-End Autonomous Driving.pdf
paper_sha256: 076ce47e0b0a323c9bb0072432e7f1219ccdcf645b8ff6db172f1bb9efb85023
processed_at: '2026-08-03T21:58:22-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DiffusionDriveV2

## 一、这事儿到底在解决什么

开车这件事本质上是个多选题。同一个路口，你可以直行、可以变道、可以右转，三条路都合理。但传统的 end-to-end driving model 只会输出一条 trajectory，等于考试只填一个答案，遇到复杂路口就懵了。

后来有人用 diffusion model 来生成多条 trajectory，理论上可以生成 20 条不同的候选。但 diffusion 有个臭名昭著的毛病叫 **mode collapse**——模型偷懒，生成的 20 条最后全塌成几乎一模一样的那条，因为那条概率最高、loss 最小。你让它给多样化，它给你复制粘贴 20 次。

DiffusionDrive V1 想了个挺巧的招：先把 expert 司机的轨迹用 K-Means 聚类，发现大概有 $N_{\text{anchor}}$ 种典型 intent（直行、左转、变道……），每种 intent 挑一条代表轨迹叫 **anchor**。然后让模型在每个 anchor 附近加噪、去噪，生成轨迹。这样等于把 action space 切成了几个子空间，每个子空间对应一种 driving intent，强制 model 在不同 intent 下生成不同的轨迹。diversity 算是保住了。

参考：DiffusionDrive V1 paper https://arxiv.org/abs/2411.15249

## 二、V1 的死穴在哪

V1 的训练 paradigm 是 imitation learning (IL)。每个场景只有一条 GT trajectory，训练时只能监督离 GT 最近的那条 anchor（positive mode），其他 anchor 生成的轨迹统统不管。结果就是：模型一边吐出高质量直行轨迹，一边吐出一堆撞墙的右转轨迹——反正右转那个 anchor 没 GT，怎么生成都没人管，loss 不会惩罚。

Fig. 1(b) 那张图很直观：绿色高质量轨迹旁边一圈红色圈的撞车轨迹。diversity 有了，quality 的 floor 没了。

最后怎么办？靠下游一个小小的 classifier 从 20 条里挑一条执行。问题在于这个 classifier 参数量比 generator 小得多，generalization 能力弱。一旦遇到 OOD scenario，classifier 自己先挂了，整个系统炸掉。这就好比你养了个能写 20 篇作文的学生，但只有一个小老师批改，小老师哪天病了，20 篇里挑不出一篇能用的。

这就是 paper 反复说的 **diversity vs consistent high quality dilemma**——V1 选了 diversity，丢了 quality floor。

## 三、为什么 RL 是自然的解

IL 的本质是"模仿 expert"，永远 bound 在 expert data 的质量上。expert 开得保守，model 就开得保守。expert 数据里没有的 maneuver，model 永远学不会。

RL 的本质是"自己探索 + reward 反馈"，跟 AlphaGo 一个道理。reward 可以是 rule-based 的（撞了扣分、开得快加分、舒适度加分），model 通过探索找到得分高的 policy，甚至能找到比 expert 更好的开法。这正是 EP (Ego Progress) 指标能从 82.2 涨到 87.5 的原因——model 探索到了 expert 没见过的大胆策略。

参考：AlphaGo https://www.nature.com/articles/nature24270

而且 RL 天然能处理多模态约束：对所有 anchor 下生成的轨迹都施加 reward，撞墙的扣分、安全的加分，不管你是直行 anchor 还是右转 anchor。这就把 quality floor 抬起来了。

## 四、直接搬 GRPO 会踩的坑

DeepSeek-R1 让 GRPO 火了，LLM 里的做法是：同一个 prompt 让模型生成 G 个 response，互相比较谁好谁坏，group-relative advantage 算出来做 policy gradient。

参考：DeepSeek-R1 https://arxiv.org/abs/2501.12948  
参考：GRPO 原始 paper DeepSeekMath https://arxiv.org/abs/2402.03300

如果 driving 这边照搬：每个场景下从所有 anchor 各采几条 trajectory，放一个 group 里算 relative advantage。听起来挺自然，对吧？

**但这里有坑**。anchor 的设计初衷就是把 action space 切成不同 intent 子空间。右转和直行是两种合理 intent，应该共存，不应该互相比较优劣。你把它们放一个 group 里，model 会发现"直行更常见、更安全、reward 更高、advantage 更高"，于是 policy gradient 会让 model 越来越偏向直行——**又回到 mode collapse 了**，前面 anchor 设计白做了。

paper 里 Fig. 2 那张图红绿轨迹就是这个意思：绿色直行 anchor 的 reward 系统性比红色右转 anchor 高，naive GRPO 会把红色 reward 压下去，model 最后只输出绿色。

## 五、Intra-Anchor GRPO：核心 insight 一句话

**右转跟右转比，直行跟直行比，不在 intent 之间比。**

每个 anchor 自己组一个 group，group size G，内部算 group-relative advantage：

$$
A^{k,i} = \frac{r^{k,i} - \text{mean}(\{r^{k,1}, \ldots, r^{k,G}\})}{\text{std}(\{r^{k,1}, \ldots, r^{k,G}\})}
$$

变量解释：
- $k$：anchor 索引（第 k 个 intent）
- $i$：group 内第 i 个 sample，$i \in [1, G]$
- $r^{k,i}$：第 k 个 anchor 下第 i 条 trajectory 的 reward
- mean/std：在 group 内算，不让跨 anchor 的 reward 污染 advantage

直觉：在每个 intent 内部问"哪条 trajectory 更好"，但绝不问"右转和直行哪个更好"。multimodality 保住了。

这个 ablation (Tab. 5) 很说明问题：跨 anchor 比 PDMS 89.2，intra-anchor 比 90.1，差 0.9 个点。0.9 在 NAVSIM 这种 saturate benchmark 上已经不小。

## 六、Inter-Anchor Truncated GRPO：堵另一个漏洞

但只做 Intra-Anchor 又出问题：完全隔离 group 之后，advantage 失去 global 可比性。想象两个场景：

- Anchor A（直行）下 5 条 trajectory 全 suboptimal 但 safe，group 内相对好的那条 advantage 是正的
- Anchor B（右转）下 5 条 trajectory 全 collide，其中 collide 晚点的那条相对最好，advantage 也是正的

模型会收到信号："撞车晚点的右转" advantage 正，"安全的直行" advantage 也正，两个都被鼓励。这等于鼓励 model 在烂 group 里挑相对烂的，而不是绝对避开撞车。

Inter-Anchor Truncated GRPO 的哲学是 **"奖励相对进步，只惩罚绝对失败"**：

$$
A_{\text{trunc}}^{k,i} = \begin{cases} -1 & \text{if collision} \\ \max(0, A^{k,i}) & \text{otherwise} \end{cases}
$$

两层意思：
- 负 advantage 全部 clip 到 0，"烂里挑好"不给奖励
- 撞车 trajectory 不管在哪个 anchor 都强制 -1，绝对惩罚

这样保留 intra-anchor 的 multimodal 特性（正 advantage 还在 group 内相对算），但引入 global safety floor（撞车跨 anchor 一票否决）。Ablation Tab. 6：不用这个 89.5，用了 90.1。

## 七、Multiplicative Noise：小细节大智慧

探索时给 trajectory 加噪声，最直觉的做法是 additive Gaussian：每个 waypoint 独立加一个 $\epsilon \sim \mathcal{N}(0, \sigma^2)$。但 trajectory 有个 spatial scale 各向异性——近端 waypoint（车前 2 米）坐标值小，远端 waypoint（车前 30 米）坐标值大。同样 $\sigma=0.5$ 的噪声，近端被搅得稀烂，远端几乎纹丝不动，trajectory 变成折线，shape 完全被破坏。

multiplicative noise：

$$
\tau' = (1 + \epsilon_{\text{mul}}) \cdot \tau, \quad \epsilon_{\text{mul}} = (\epsilon_{\text{long}}, \epsilon_{\text{lat}})
$$

只有两个标量：纵向缩放因子 + 横向缩放因子。整条 trajectory 按比例缩放，远端被放大同样比例，shape 完整保留，trajectory 仍然光滑。

这个设计跟 trajectory 这个数据结构的内在 anisotropy 完美匹配。换到 image diffusion 你不会用 multiplicative noise，因为 image pixel 没这种"远端 scale 大"的特性。这是典型的 **design choice 必须匹配 data structure** 的例子。

Fig. 3 对比图很直观：additive 是折线锯齿，multiplicative 是平滑变形。

## 八、把 Denoising 视作 MDP

paper 借鉴 DPPO 的思路，把 conditional denoising chain 看成 Markov Decision Process：

$$
\pi_\theta(\tau_{t-1}^k \mid \tau_t^k, z, \mathbf{a}^k) = \mathcal{N}\bigl(\tau_{t-1}^k; \mu_\theta(\tau_t^k, t, z, \mathbf{a}^k), \eta(1-\alpha_t)\mathbf{I}\bigr)
$$

- $\tau_t^k$：state（第 t 步的 noisy trajectory）
- $\tau_{t-1}^k$：action（去噪一步，变成下一个 state）
- $\mu_\theta$：diffusion decoder 预测的均值
- $\eta$：exploration noise scale，训练时 $\eta=1$（stochastic，能算 likelihood），推理时 $\eta=0$（deterministic DDIM）

参考：DPPO https://arxiv.org/abs/2409.00588

这里有个工程细节：如果推理也用 stochastic $\eta=1$，DDIM 退回 DDPM，要更多步数才能去噪干净，inference 慢。所以训练 stochastic 探索 + 推理 deterministic 部署，是"训练时探索，部署时稳"的 pattern。

policy gradient 用 REINFORCE：

$$
\nabla_\theta \mathcal{L} = \mathbb{E}\left[\sum_{t=1}^{T_{\text{trunc}}} \nabla_\theta \log \pi_\theta(\tau_{t-1}^k \mid \tau_t^k) \cdot A^{k,i} \cdot \gamma^{t-1}\right]
$$

- $\gamma^{t-1}$：discount factor，paper 设 $\gamma=0.8$，让早期 noisy step 的 gradient 贡献小（早期 state 噪声大，policy 输出方差大，不 discount 会梯度爆炸）
- $A^{k,i}$：trajectory-level advantage，应用到整条 denoising chain 的每个 step

reward 只在最终 clean trajectory $\tau_0^{k,i}$ 上评估，但梯度通过 $\log \pi_\theta$ 反传到每一步 denoising。这是 reward sparsity 的标准处理。

## 九、Loss 长啥样

最终 loss：

$$
\mathcal{L} = \mathcal{L}_{RL} + \lambda \mathcal{L}_{IL}, \quad \lambda = 0.1
$$

- $\mathcal{L}_{RL}$：上面推导的 policy gradient loss
- $\mathcal{L}_{IL}$：原 DiffusionDrive 的 IL loss（reconstruction for positive mode + BCE for mixture weight），相当于 GRPO 里的 KL regularizer，防止 policy 漂太远忘掉基本驾驶能力

训练流程：先用 DiffusionDrive V1 的 IL pretrain weight 做 cold start，再 10 epoch RL fine-tune。这跟 LLM RLHF 的 SFT→RL 范式完全一致——先让 model 有基本能力，再用 RL 抬上限和下限。

## 十、Mode Selector 的角色

generator 输出多 anchor × 多 trajectory，最后还要选一条执行。selector 设计借鉴 DriveSuprim 的两阶段：coarse scorer 先选 top-k，fine-grained scorer 精选。

辅助 loss 是 Margin-Rank：

$$
\mathcal{L}_{\text{rank}} = \frac{1}{N} \sum_{i,j} \max\bigl(0, -\text{sign}(s_i - s_j) \cdot (\hat{s}_i - \hat{s}_j) + m\bigr)
$$

- $s_i, s_j$：GT 排序关系
- $\hat{s}_i, \hat{s}_j$：predicted score
- $m$：margin 超参

直觉：让 predictor 学相对排序，避免直接 regress 绝对值。ranking 是 easier learning target。

参考：DriveSuprim https://arxiv.org/abs/2506.06659

但 Tab. 9 的 ablation 很关键：DiffusionDrive 加上 V2 的复杂 selector 也只涨 1 分（88.1→89.1），DiffusionDriveV2 + 同样 selector 是 91.2。**说明 V2 的提升主要来自 generator 质量本身**，selector 不是主因。这反驳了"性能提升来自更复杂 selector"的质疑。

## 十一、NAVSIM metric 讲讲

NAVSIM v1 的 PDMS：

$$
\text{PDMS} = NC \times DAC \times \frac{5 \cdot EP + 5 \cdot TTC + 2 \cdot C}{12}
$$

- $NC$ (No At-Fault Collisions)：无责任碰撞率，hard gate
- $DAC$ (Drivable Area Compliance)：不出道路率，hard gate
- $EP$ (Ego Progress)：前进效率
- $TTC$ (Time to Collision)：碰撞时间
- $C$ (Comfort)：舒适度

直觉：前两项是 hard gate（撞车或出路面直接归零），后三项加权平均。NAVSIM v2 的 EPDMS 加了 DDC、TL、LK、HC、EC，维度更多更接近真实驾驶。

参考：NAVSIM https://github.com/autonomousvision/navsim

## 十二、结果解读

**NAVSIM v1 (Tab. 1)**：91.2 PDMS，SOTA。比 DiffusionDrive V1 高 3.1，比 DIVER（同样 RL-based 但用在 vanilla diffusion 上）高 2.9，比 GoalFlow（用更大 backbone V2-99）高 0.9。EP 从 82.2 → 87.5 涨 5.3 分，证明 RL 探索到了比 expert 更激进的策略。

**NAVSIM v2 (Tab. 2)**：85.5 EPDMS，SOTA。EC (Extended Comfort) 91.0 远超 DriveSuprim 的 77.0 和 Hydra-MDP++ 的 70.9。这很有意思——IL 没见过的 comfort metric 仍然 work，说明 RL 学到的是更 general 的 driving principle，不只是 fit 训练 metric。

**Diversity vs Top-K PDMS (Tab. 3) 这张表是 paper 灵魂**：

| Method | Div. | PDMS@1 | PDMS@5 | PDMS@10 |
|---|---|---|---|---|
| Vanilla (TransfuserTD) | 0.1 | 85.7 | 85.7 | 85.7 |
| DiffusionDrive (IL) | 42.3 | 93.5 | 84.3 | 75.3 |
| DiffusionDriveV2 (RL) | 30.3 | 94.9 | 91.1 | 84.4 |

人话解读：
- Vanilla：diversity 0.1，全 collapse 成一条，Top-1=Top-10，质量中等
- DiffusionDrive V1：diversity 42.3 拉满，但 Top-10 只剩 75.3——9 条 trajectory 都是垃圾，全靠 selector 抽中那 1 条 93.5 的撑场面。selector 一旦失手就翻车
- DiffusionDriveV2：diversity 从 42.3 降到 30.3（牺牲一点），但 Top-10 从 75.3 涨到 84.4——**所有 trajectory 都质量过关**，不用把命押在 selector 上。Top-1 还从 93.5 涨到 94.9，upper bound 也抬高了

这就是 paper 反复说的 **"diversity 不是目的，diversity + consistent high quality 才是"**。RL 把所有 mode 拉到 quality floor 上，diversity 略损失但 floor 大幅提升，整体 trade-off 最优。

## 十三、几个我特别想强调的 intuition

**Intuition 1：GRPO 的 group 定义是 domain-specific decision**。LLM 里 group = 同 prompt 多 response，可以 cross-compare。Driving 里 group = 同 intent 内多 trajectory，不能 cross-compare。迁移 GRPO 时必须想清楚 group 的语义边界。这个 insight 对所有 RL + structured output 的工作都有借鉴价值。

**Intuition 2：Truncated advantage 是 multi-objective RL 的一种实现**。Intra-Anchor 保留 multimodality（鼓励探索不同 intent），Inter-Anchor Truncated 引入 safety floor（惩罚绝对失败）。两个 objective 用不同 advantage 计算方式实现，这种"分目标分 advantage"的设计很优雅。

**Intuition 3：IL 的本质局限是"只有一个正确答案"**。每场景只有一条 GT，model 只学到 positive mode 的质量。RL 的本质优势是"对所有 action 都有 reward 信号"，可以约束所有 mode。这就是 IL→RL 范式转换的根本原因。

**Intuition 4：multiplicative noise 反映 data structure awareness**。trajectory 有 spatial scale anisotropy，所以用 multiplicative；image pixel 各向同性，用 additive。design choice 跟 data structure 对齐才能 work。

**Intuition 5：cold start from IL, fine-tune with RL**。完全 from scratch RL 训 diffusion 极难收敛，先用 IL 让 model 有基本能力，再 RL fine-tune 抬 floor 和 ceiling。这跟 AlphaGo 的 SL policy network + RL policy network、LLM 的 SFT+RLHF 是同一个套路。

## 十四、几个联想

**AlphaGo 类比**：SL policy network 学人类棋谱（IL），RL policy network 自我对弈超越人类（RL）。DiffusionDriveV2 是 IL cold start + RL fine-tune，结构完全同构。EP 指标超越 expert 就是"超越人类"的证据。

**RLHF 类比**：SFT → Reward Model → PPO。DiffusionDriveV2 简化成 IL → RL，因为 driving reward 是 rule-based 可直接计算，不需要单独训 reward model。但 future work 如果 reward 也用 learned model（比如 human preference），就完全对应 RLHF。

**Flow Matching 方向**：GoalFlow 用 flow matching 替代 diffusion 做 multimodal trajectory 生成，但仍是 IL-based。DiffusionDriveV2 的 RL 思想完全可以迁移到 flow matching model——把 flow 的 ODE solver 视作 MDP，policy gradient 应该同样 work。这是个潜在的 next step。

参考：GoalFlow https://arxiv.org/abs/2503.04031

**Robotics diffusion + RL**：DPPO 之后 robotics 领域已经有人做 diffusion + RL，DiffusionDriveV2 是这个范式在 driving 上的特化。driving 的特殊性在于 anchor structure（多 intent partition），robotics manipulation 可能没有这种天然 partition，所以不需要 Intra-Anchor 这种设计。

**Mode collapse 在 image generation**：image diffusion 用 CFG (Classifier-Free Guidance) 缓解 mode collapse，但代价是 sample quality trade-off。driving 有 anchor structure 可以更直接 enforce multimodality，不需要 CFG 这种间接手段。

**GTRS trajectory vocabulary**：ablation 提到训练 selector 时从 GTRS vocabulary 里采 1% trajectory 做数据增强，说明 anchor-based 和 vocabulary-based 是 hybrid 互补的。VADv2、Hydra-MDP 走 vocabulary 路线，DiffusionDrive 走 anchor + diffusion 路线，未来可能是混合。

参考：VADv2 https://arxiv.org/abs/2402.13243  
参考：Hydra-MDP https://arxiv.org/abs/2406.06978

**Reward hacking 风险**：reward 是 NAVSIM simulator 给的 rule-based 加权，OOD scenario 下 reward 可能不 robust。如果 reward 本身有 bias，RL 会 exploit 这个 bias。比如 reward 偏好"不撞车"甚于"前进"，model 可能学会原地不动刷 NC=100。paper 没讨论这个，是 future work 的潜在坑。

**Sim-to-real gap**：NAVSIM 是 non-reactive simulation，其他 agent 不响应 ego action。真实闭环里其他 agent 会反应。RAD (Gao et al. 2025) 用 3DGS 环境训 end-to-end policy 是更接近 real 的方向，但 reward 信号更难定义。

参考：RAD https://arxiv.org/abs/2502.13144

**Anchor 数量选择**：$N_{\text{anchor}}$ 是 K-Means 聚类 expert behavior 得到的固定数量，可能不覆盖 long-tail intent。比如紧急避让、U-turn 这种罕见 maneuver 可能没对应 anchor。dynamic anchor 或者 hierarchical anchor 是潜在改进方向。

## 十五、一句话总结

DiffusionDriveV2 把 GRPO 从 LLM 借到 driving，关键 insight 是"右转跟右转比、直行跟直行比、不在 intent 之间比"，用 Intra-Anchor GRPO 保 multimodality + Inter-Anchor Truncated GRPO 引入 global safety floor，配 multiplicative noise 保 trajectory 平滑性，最终让 truncated diffusion 同时拿到 diversity 和 consistent high quality，NAVSIM v1/v2 双 SOTA。本质上是把 IL 的"只有一个正确答案"换成 RL 的"所有 action 都有约束"，把 model 的 lower bound 和 upper bound 同时抬起来。

---

# DiffusionDriveV2: 从 IL Dilemma 到 RL-Constrained Truncated Diffusion

## 1. 这篇 paper 在解决什么本质问题

End-to-end autonomous driving (E2E-AD) 里 trajectory generation 的核心矛盾在于：vanilla diffusion model 容易 **mode collapse**——所有 trajectory 都 collapse 到一个 high-probability mode 上，失去 diversity。DiffusionDrive 用 anchor-based Gaussian Mixture Model (GMM) prior 把 action space 按 driving intent 分割（左转/直行/变道各自一个 anchor），强制 multimodal。但是它采用 imitation learning (IL)，每个 scene 只有一条 GT trajectory，只能选离 GT 最近的 anchor 作为 positive mode 做监督，其他 anchor 都是 negative modes 完全没监督。结果 generator 一边吐高质量 trajectory，一边吐一堆 collision trajectory，全靠下游一个参数量很小的 selector/classifier 兜底。一旦 OOD，selector 挂掉整个系统就炸了。

DiffusionDriveV2 的核心论点是：**RL 是 IL 之外更自然的 training paradigm**，因为它能 (a) 对所有 mode（包括 negative modes）施加 constraint，把 lower bound 抬高；(b) 鼓励 exploration，去 IL expert 数据集之外找更好 policy，把 upper bound 抬高。这篇 paper 的工作是把 GRPO (Group Relative Policy Optimization) 从 LLM 借过来，正确地迁移到 anchored truncated diffusion 上。

Reference: 
- DiffusionDrive V1: https://arxiv.org/abs/2411.15249
- DiffusionDriveV2 repo: https://github.com/hustvl/DiffusionDriveV2
- DeepSeek-R1 (GRPO 启发源): https://arxiv.org/abs/2501.12948
- GRPO 原始 paper (DeepSeekMath): https://arxiv.org/abs/2402.03300

---

## 2. Truncated Diffusion 的数学骨架

先回顾 DiffusionDrive 把 trajectory 分布写成 GMM：

$$
p(\tau \mid z) = \sum_{k=1}^{N_{\text{anchor}}} s(\mathbf{a}^k \mid z)\, p(\tau^k \mid \mathbf{a}^k, z)
$$

变量解释：
- $z$：scene context，由 perception backbone (ResNet-34) 提取的 BEV + agent/map features
- $\mathbf{a}^k$：第 $k$ 个 anchor trajectory，由 K-Means 聚类 expert driving behaviors 得到的代表 trajectory，每个对应一种 driving intent
- $s(\mathbf{a}^k \mid z)$：mixture weight，scene context $z$ 下选择第 $k$ 个 intent 的概率
- $p(\tau^k \mid \mathbf{a}^k, z) = \mathcal{N}(\tau^k \mid \mathbf{a}^k + \mu^k(z), \Sigma^k(z))$：以 anchor 为中心的 Gaussian，模型预测的是相对 anchor 的 offset $\mu^k(z)$
- $N_{\text{anchor}}$：anchor 总数

Truncated diffusion 把标准 noise schedule 截断到 $T_{\text{trunc}} \ll T$ 步，对每个 anchor 加噪：

$$
\tau_t^k = \sqrt{\bar{\alpha}_t}\, \mathbf{a}^k + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$：cumulative noise schedule coefficient
- $t \in [1, T_{\text{trunc}}]$：truncated diffusion step index
- $\tau_t^k$：第 $t$ 步 noisy trajectory for anchor $k$

直觉上，truncated diffusion 不从纯 Gaussian noise 开始 denoise，而是从 "anchor + 少量 noise" 开始，这样 2 步就能 denoise 完，大幅降低 inference latency，同时 GMM prior 强制不同 anchor 走不同 region。

---

## 3. 把 Denoising 视作 MDP（DPPO 思想）

DiffusionDriveV2 借鉴 DPPO (Ren et al. 2024) 把 conditional denoising chain 视作 Markov Decision Process，每一步 denoise 是一个 Gaussian policy：

$$
\pi_\theta(\tau_{t-1}^k \mid \tau_t^k, z, \mathbf{a}^k) = \mathcal{N}\bigl(\tau_{t-1}^k;\, \mu_\theta(\tau_t^k, t, z, \mathbf{a}^k),\, \eta(1-\alpha_t)\mathbf{I}\bigr)
$$

变量：
- $\tau_t^k$ → state at step $t$ (noisy trajectory)
- $\tau_{t-1}^k$ → action (the denoised step that becomes next state)
- $\mu_\theta(\cdot)$ → mean of Gaussian policy, predicted by the diffusion decoder network
- $\alpha_t$ → noise schedule coefficient at step $t$
- $\eta$ → exploration noise scale. **关键技巧**：训练时 $\eta = 1$（等价 DDPM，stochastic sampler）保证 $\pi_\theta$ 有正常数方差以计算 Gaussian likelihood；inference 时 $\eta = 0$（等价 DDIM deterministic sampler），保证 deterministic 推理。

这里有一个细节值得注意：如果直接用 DDIM 的 $\eta=0$，policy 退化为 Dirac delta，REINFORCE 的 $\log \pi_\theta$ 就无定义。所以 $\eta$ 的切换是 training/inference 不一致的设计，类似 stochastic policy gradient 训练但 deterministic policy 推理。

Reference: 
- DPPO paper: https://arxiv.org/abs/2409.00588
- DDIM: https://arxiv.org/abs/2010.02502
- DDPM: https://arxiv.org/abs/2006.11239

---

## 4. Policy Gradient 推导

利用 REINFORCE，policy gradient 写成：

$$
\nabla_\theta \mathcal{T}(\pi_\theta^k) = \mathbb{E}_{\pi_\theta^k}\!\left[\sum_{t=1}^{T_{\text{trunc}}} \nabla_\theta \log \pi_\theta^k(\tau_{t-1}^k \mid \tau_t^k)\, A_t^k \right]
$$

- $A_t^k$：advantage function at step $t$ for anchor $k$
- 每个 trajectory 用一个 trajectory-level reward $R(\tau_0^{k,i})$ 反传给所有 denoising step（reward sparsity 通过 discount $\gamma^{t-1}$ 处理）

这里的 trick 是 reward 只在最终 clean trajectory $\tau_0^{k,i}$ 上评估，但梯度通过 $\log \pi_\theta$ 反传到每个 denoising step。一个 trajectory 级 reward 当作每个 step 的 reward，再乘 $\gamma^{t-1}$ 让早期 noisy step 贡献小（避免高方差，因为早期 step 输入噪声大，policy 输出方差大）。

---

## 5. Scale-Adaptive Multiplicative Exploration Noise（非常重要的细节）

这是 paper 里我最喜欢的设计直觉。Trajectory 在 BEV 坐标系下，近端 waypoint（车前方几米）的 absolute coordinate 很小，远端 waypoint（几十米外）coordinate 很大。如果用 additive Gaussian noise：

$$
\epsilon_{\text{add}} = \{(\epsilon_{x,n}, \epsilon_{y,n})\}_{n=1}^{N_f}, \quad \tau' = \tau + \epsilon_{\text{add}}
$$

每个 waypoint 独立采样 noise，相同 scale 的 noise 在近端是巨大扰动，在远端是微小扰动，导致 trajectory 变成 "broken line"，失去平滑性。远端 shape 都被 noise 拍碎了，本质上 destroy 了 trajectory 的结构。

Multiplicative noise：

$$
\tau' = (1 + \epsilon_{\text{mul}})\,\tau, \quad \epsilon_{\text{mul}} = (\epsilon_{\text{long}}, \epsilon_{\text{lat}})
$$

- $\epsilon_{\text{long}}$：纵向缩放 noise (scalar)
- $\epsilon_{\text{lat}}$：横向缩放 noise (scalar)
- 只有两个标量 noise，分别沿 trajectory 纵向/横向方向缩放

直觉：noise 与 trajectory 自身 scale 成比例，远端 waypoint 被放大同样比例，shape 完整保留，trajectory 仍然光滑。这是 trajectory 这个数据结构特化的 exploration strategy——你不会在 image diffusion 用 multiplicative noise，因为 image pixel 没有 trajectory 这种"远端 scale 大"的内在 anisotropy。

论文报告 multiplicative 比 additive 在 PDMS 上提升 0.4（90.1 vs 89.7），看似不大但稳定。

---

## 6. Intra-Anchor GRPO（核心创新 1）

GRPO 在 LLM 里做法：sample G 个 response，计算 group-relative advantage $A^i = (r^i - \text{mean})/\text{std}$，避免训练 value model。

DiffusionDriveV2 的关键 insight：**naive 把所有 anchor 下的 sample 放进同一 group 做 GRPO 会破坏 multimodality**。因为 anchor 设计的初衷就是 partition action space into 不同 intent subspaces，如果 "右转" anchor 的 trajectory 和 "直行" anchor 的 trajectory 在同一 group 比较 advantage，policy gradient 会 favor 更 frequent 的 "直行" mode，逐步 collapse 到单一 intent——这就回到了 vanilla diffusion 的 mode collapse 问题。

所以 Intra-Anchor GRPO 的做法：**每个 anchor 单独组 group，group 内做 advantage normalization**。

$$
A^{k,i} = \frac{r^{k,i} - \text{mean}(\{r^{k,1}, r^{k,2}, \ldots, r^{k,G}\})}{\text{std}(\{r^{k,1}, r^{k,2}, \ldots, r^{k,G}\})}
$$

- $k$：anchor index
- $i$：sample index within group，$i \in [1, G]$
- $G$：group size，每个 anchor 下采 $G$ 个 trajectory variations
- $r^{k,i} = R(\tau_0^{k,i})$：trajectory-level reward for $i$-th sample in $k$-th anchor

RL loss：

$$
\mathcal{L}_{RL} = -\frac{1}{N_{\text{anchor}}} \sum_{k=1}^{N_{\text{anchor}}} \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_{\text{trunc}}} \sum_{t=1}^{T_{\text{trunc}}} \gamma^{t-1} \log \pi_\theta^k(\tau_{t-1}^{k,i} \mid \tau_t^{k,i})\, A^{k,i}
$$

- $\gamma^{t-1}$：denoising discount，paper 设 $\gamma = 0.8$，downweight 早期 noisy step（早期 step state 噪声大，policy 输出方差大，会导致梯度方差爆炸）
- $A^{k,i}$ 是 trajectory-level advantage，应用在整条 denoising chain 上的每个 step

直觉：在同一 anchor 内做相对比较，"在你这个 intent 下，哪个 trajectory 更好"，但绝不在 intent 之间做比较。这保留 GMM 的 multimodal prior 不被 RL 损坏。

Ablation Tab. 5: 不用 Intra-Anchor (即跨 anchor 比较) PDMS 89.2 vs 用 90.1，证明 insight 正确。

Reference: GRPO 原始 paper (DeepSeekMath) https://arxiv.org/abs/2402.03300

---

## 7. Inter-Anchor Truncated GRPO（核心创新 2）

但是 Intra-Anchor 完全隔离 modes 又有新问题：**advantage 失去 global comparability**。假设：
- Anchor A (直行) 下有 5 个 sample，全是 suboptimal 但 safe，group-relative advantage 有正有负
- Anchor B (右转) 下有 5 个 sample，全是 collide，但其中一个 collide 得稍微晚一点，相对最好，group-relative advantage 是正的

这会给 model 误导信号：collide 的 trajectory 反而比 safe 的 advantage 高。RL 会鼓励 collide。

Inter-Anchor Truncated GRPO 的设计哲学："reward relative improvements, but only penalize absolute failures"。具体实现：

$$
A_{\text{trunc}}^{k,i} = \begin{cases} -1 & \text{if collision} \\ \max(0, A^{k,i}) & \text{otherwise} \end{cases}
$$

- 负 advantage 全部 clip 到 0（不奖励"在烂 group 里相对好"的 trajectory）
- 但是 collide 的 trajectory 强制赋 -1（绝对惩罚）

这个设计有两层意思：
1. **保留 Intra-Anchor 的 multimodality**：正 advantage 仍然只在 group 内相对计算，不在 cross-intent 比较
2. **但引入 global floor**：collide 不论在哪个 anchor 都该罚，用 absolute penalty 而不是 relative penalty

直觉上，这是把 "exploration 鼓励" 留在 anchor 内部，把 "safety constraint" 跨 anchor 全局化。它实现了 paper abstract 里说的 "constrain low-quality modes and explore for superior trajectories" 的两目标分离。

Ablation Tab. 6: 不用 Inter-Trunc PDMS 89.5 vs 用 90.1。

---

## 8. Combined Loss & Training Pipeline

最终 loss：

$$
\mathcal{L} = \mathcal{L}_{RL} + \lambda \mathcal{L}_{IL}
$$

- $\lambda \in (0, 1)$：BC loss weight，paper 设 $\lambda = 0.1$
- $\mathcal{L}_{IL}$：原 DiffusionDrive 的 IL loss (Eq. 4)，包括 reconstruction loss for positive mode 和 BCE for mixture weight

这里的 $\mathcal{L}_{IL}$ 类比 GRPO 原始 paper 里的 KL divergence regularizer，防止 policy 漂移太远破坏 general driving capability。Cold-start 时用 DiffusionDrive pre-trained weights，再做 10 epoch RL fine-tuning。

训练超参（Tab. 7）：
- AdamW, lr $2\times 10^{-4}$, weight decay $10^{-4}$, cosine schedule + 10% linear warmup
- Batch size 512, 10 epochs
- Multiplicative noise min std = 0.04 (防止 entropy collapse)
- log $\pi$ 评估时 Gaussian std 至少 0.1 (防止大梯度，stability trick)
- $\gamma = 0.8$ denoising discount
- 8× NVIDIA L20 GPUs

这里有两个 engineering 细节值得注意：
1. **Multiplicative noise min std = 0.04**：RL 训练容易 entropy collapse，policy 越来越 deterministic，exploration 消失。强制最小 noise std 保持 exploration。这是 entropy regularization 的工程实现。
2. **log $\pi$ min std = 0.1**：当 policy variance 接近 0 时，$\log \pi$ 的梯度数值上会爆炸（Gaussian density 在 variance → 0 时趋向 delta）。clip min std 防止梯度爆炸。

---

## 9. Mode Selector 架构

Generator 输出多 anchor 多 trajectory，最后还要选一条执行。Mode selector 的设计：
- Trajectory 坐标作为 query
- Deformable spatial cross-attention 与 BEV features 交互
- Cross-attention 与 agent/map queries 交互
- MLP 输出 score
- 两阶段 coarse-to-fine（借鉴 DriveSuprim）：coarse scorer 先选 top-k，fine-grained scorer 精选

辅助 Margin-Rank loss：

$$
\mathcal{L}_{\text{rank}} = \frac{1}{N} \sum_{i,j} \max\bigl(0, -\text{sign}(s_i - s_j) \cdot (\hat{s}_i - \hat{s}_j) + m\bigr)
$$

- $s_i, s_j$：GT 排序关系
- $\hat{s}_i, \hat{s}_j$：predicted score
- $m$：margin hyperparameter (positive)
- $\text{sign}(s_i - s_j)$：indicator 表示 GT 中哪个 trajectory 更好

这个 loss 的直觉：让 predictor 学**相对排序**，避免直接 regress 绝对 continuous 值。Ranking 是 easier learning target。Reward 是 non-differentiable，但 selector 是 supervised 的，可以直接用 GT rank 训练。

Ablation Tab. 8：Coarse2Fine +0.2 PDMS, Rank Loss 再 +0.2 PDMS。

Reference: DriveSuprim https://arxiv.org/abs/2506.06659

---

## 10. NAVSIM 评估 Metric 拆解

NAVSIM v1 的 PDMS (PDM Score)：

$$
\text{PDMS} = NC \times DAC \times \left(\frac{5 \cdot EP + 5 \cdot TTC + 2 \cdot C}{12}\right)
$$

- $NC$ (No At-Fault Collisions)：无责任碰撞率
- $DAC$ (Drivable Area Compliance)：不出道路率
- $TTC$ (Time to Collision)：碰撞时间
- $C$ (Comfort)：舒适度（jerk 等）
- $EP$ (Ego Progress)：前进效率
- $5,5,2$ 是权重，反映重要性

直觉：前两项 (NC, DAC) 是 hard gate（hard safety violation → 整个 score 归零），后三项是 soft metric（加权平均）。

NAVSIM v2 扩展为 EPDMS：

$$
\text{EPDMS} = NC \times DAC \times DDC \times TL \times \frac{5 \cdot TTC + 2 \cdot C + 5 \cdot EP + 5 \cdot LK + 5 \cdot EC}{22}
$$

- $DDC$ (Driving Direction Compliance)
- $TL$ (Traffic Lights Compliance)
- $LK$ (Lane Keeping)
- $HC$ (History Comfort)
- $EC$ (Extended Comfort)

NAVSIM v2 增加更多 compliance 维度，更接近真实 driving。

Reference: 
- NAVSIM: https://github.com/autonomousvision/navsim
- NAVSIM paper: https://arxiv.org/abs/2406.15349

---

## 11. 实验结果深度解读

### Main results NAVSIM v1 (Tab. 1)
- DiffusionDriveV2: **91.2 PDMS** (ResNet-34, 21.8M params)
- vs DiffusionDrive: +3.1 PDMS
- vs DIVER (also RL-based): +2.9 PDMS
- vs GoalFlow (V2-99 backbone, 96.9M params, 4.5× params): +0.9 PDMS

特别值得注意的是 EP (Ego Progress) 从 82.2 → 87.5（+5.3）。EP 提升 5 个点意味着 model 探索到比 expert 更激进的 driving policy——IL expert 通常 conservative，RL 让 model 超越 expert 上限。这是 RL 在 driving 上比 IL 更有优势的实证。

### NAVSIM v2 (Tab. 2)
- DiffusionDriveV2: **85.5 EPDMS** (SOTA)
- vs DriveSuprim 83.1
- vs Hydra-MDP++ 81.4

特别值得注意的是 EC (Extended Comfort) 91.0，远超其他方法（DriveSuprim 77.0, Hydra-MDP++ 70.9）。这说明 model 在新加的 comfort 维度上 generalization 好——IL 没见过的 metric 仍然 work，说明 RL 学到的是更 general 的 driving principle。

### Diversity vs Top-K PDMS (Tab. 3) **这个表是 paper 的灵魂**

| Method | Div. | PDMS@1 | PDMS@5 | PDMS@10 |
|---|---|---|---|---|
| TransfuserTD (vanilla) | 0.1 | 85.7 | 85.7 | 85.7 |
| DiffusionDrive (IL) | 42.3 | 93.5 | 84.3 | 75.3 |
| DiffusionDriveV2 (RL) | 30.3 | 94.9 | 91.1 | 84.4 |

直觉分析：
- **TransfuserTD**: Div 0.1 → mode collapse，所有 trajectory 都一样，Top-1=Top-10，质量中等
- **DiffusionDrive**: Div 42.3 → 高度 diverse，但 Top-10 只有 75.3 → 9 条 trajectory 都是垃圾，全靠 selector 兜底，Top-1 93.5 比较高但 selector 是脆弱的
- **DiffusionDriveV2**: Div 30.3 → diversity 降低一些（从 42.3 → 30.3），但 Top-10 PDMS 从 75.3 → 84.4，**lower bound 大幅提升**

这正是 paper 的核心论点：**diversity 不是目的，diversity + consistent high quality 才是**。DiffusionDriveV2 通过 RL constraint 把所有 mode 拉到 high quality floor 上，diversity 略有损失但 lower bound 巨大提升，整体 trade-off 最优。Top-1 94.9 也比 DiffusionDrive 高 1.4，说明 upper bound 也抬高了——RL 探索到了更好的 policy。

### Mode Selector Ablation (Tab. 9)

| Model | Selector | PDMS |
|---|---|---|
| DiffusionDrive | × | 88.1 |
| DiffusionDrive | ✓ (V2 selector) | 89.1 |
| DiffusionDriveV2 | ✓ | 91.2 |

DiffusionDrive 加上 V2 selector 只涨 1 分，DiffusionDriveV2 比 DiffusionDrive+V2 selector 高 2.1 分。**说明 V2 的提升主要来自 generator，不是来自更强的 selector**。这反驳了 "performance gain 来自更复杂 selector" 的质疑，强化 "generator 本身质量提升" 的论点。

---

## 12. 整体 Architecture 图解析 (Fig. 2)

Pipeline 走向：
1. **Perception backbone (ResNet-34)**：3 个 cropped camera + rasterized BEV LiDAR → BEV features + agent/map queries
2. **Anchor prior**：$N_{\text{anchor}}$ 个 predefined anchor $\{\mathbf{a}^k\}$
3. **Truncated diffusion 加噪**：每个 anchor 加噪得到 $\{\tau_t^k\}$，从 anchored Gaussian 采样
4. **Multiplicative exploration noise**：在 trajectory 上加 $(1+\epsilon_{\text{mul}})\tau$ 保持平滑
5. **Diffusion decoder**：输入 noisy trajectories $\{\tau_t^k\}$ + scene context $z$，输出 denoised trajectories $\{\hat{\tau}^k\}$ 和 mixture scores $\hat{s}^k$
6. **RL training**: Intra-Anchor GRPO + Inter-Anchor Truncated GRPO 计算 advantage，REINFORCE 更新
7. **Mode selector (inference)**：两阶段 coarse-to-fine + Margin-Rank loss 选出 final trajectory
8. **Inference**: 2 步 denoising (DDIM, $\eta=0$)

关键点：
- 训练时 $\eta=1$ 加 multiplicative noise 探索；推理时 $\eta=0$ deterministic
- RL loss 和 IL loss 联合训练 ($\mathcal{L}_{RL} + 0.1\mathcal{L}_{IL}$)
- Mode selector 单独 stage 训练 20 epochs，with data augmentation (multiplicative noise + 1% GTRS trajectory)

---

## 13. 与同期 RL-based Diffusion 工作对比

- **DIVER (Song et al. 2025)**: RL + vanilla diffusion, 也用 GRPO，但直接在 vanilla diffusion 上做，没处理 mode collapse 问题。DiffusionDriveV2 比 DIVER 高 2.9 PDMS。
- **AlphaDrive (Jiang et al. 2025)**: VLM + RL + reasoning, 不同 paradigm。
- **RecogDrive (Li et al. 2025)**: RL + cognitive framework.
- **DiffusionDriveV2 的 unique**: 第一个把 GRPO 正确迁移到 truncated diffusion + GMM prior，关键洞察是 intra-anchor vs inter-anchor advantage 计算的 separation。

Reference:
- DIVER: https://arxiv.org/abs/2507.04049
- AlphaDrive: https://arxiv.org/abs/2503.07608
- RecogDrive: https://arxiv.org/abs/2506.08052

---

## 14. 直觉总结 & 我的联想

这篇 paper 的 intuition 可以提炼成几条 principle：

**Principle 1: Diversity 和 Quality 不是 trade-off 的两端，而是两个 axis**。DiffusionDrive 用 GMM prior 在 diversity axis 上拉满，但 quality axis 没控制住。DiffusionDriveV2 用 RL 在 quality axis 上拉 floor，同时不让 diversity 损失太多（30.3 vs 42.3 仍然很高）。

**Principle 2: RL 的优势是能超越 expert**。IL 永远 bound 在 expert data quality 上，RL 可以探索到 expert 没见过的更好 policy。EP 指标从 82.2 → 87.5 就是 evidence。

**Principle 3: Group-relative advantage 在 multimodal 场景下要小心 group 的定义**。LLM 里 group 是 "同一 prompt 的多个 response"，可以 cross-compare。Trajectory generation 里不同 anchor 是不同 intent，不能 cross-compare。这是迁移 GRPO 时需要做的 domain-specific adaptation。

**Principle 4: Multiplicative vs Additive noise 反映 data structure**。trajectory 有 spatial scale anisotropy (近小远大)，image pixel 没有。所以 trajectory diffusion 需要 scale-adaptive noise scheme。这跟 positional encoding 用 sinusoidal 适配 sequence length 的哲学一致——design choice 必须匹配 data structure。

**Principle 5: Training/inference 不一致可以 work**。$\eta=1$ 训练 $\eta=0$ 推理，类似 stochastic policy gradient + deterministic policy evaluation。这种"训练 stochastic 探索，推理 deterministic 部署"是 RL diffusion 的通用 pattern。

**Principle 6: Cold-start from IL, RL fine-tune**。完全 from scratch RL 训练 diffusion 极难收敛，先用 IL pretrain 让 model 有基本 driving capability，再 RL fine-tune 抬高 floor 和 ceiling。这是 LLM RLHF 的思路在 driving 上的复刻——SFT 先，RL 后。

联想到的其他方向：
- **AlphaGo 的 policy network + value network 范式**：纯 IL 是 supervised，纯 RL 是 self-play，AlphaGo 二者结合。DiffusionDriveV2 也是 IL cold start + RL fine-tune，结构相似。
- **InstructGPT 的三阶段**：SFT → Reward Model → PPO。DiffusionDriveV2 简化为 IL → RL，因为 driving reward 是 rule-based 可直接计算，不需要单独训 reward model。
- **Flow Matching (GoalFlow) vs Diffusion**：GoalFlow 用 flow matching 也是 generative multimodal trajectory，但是 IL-based。DiffusionDriveV2 的 RL 思想完全可以迁移到 flow matching model 上。
- **Diffusion + RL 在 robotics**：DPPO 之后还有更多工作，DiffusionDriveV2 是这个范式在 driving 上的特化。
- **Mode collapse 在 image generation 也是问题**：Image diffusion 用 CFG (Classifier-Free Guidance) 缓解，但代价是 sample quality trade-off。Driving trajectory 因为有 anchor structure 可以更直接 enforce multimodality。
- **GTRS trajectory vocabulary**：Ablation 提到从 GTRS (Generalized Trajectory Scoring) vocabulary 里 sample 1% trajectory 训 mode selector。说明 anchor-based 和 vocabulary-based 是 hybrid 的。

---

## 15. 潜在局限和未来方向

Paper 没讨论但我觉得重要的：

1. **Reward 设计仍然依赖 rule-based**：$R(\tau_0^{k,i})$ 是 NAVSIM simulator 提供，本质上是 NC/DAC/TTC 等规则的加权。这种 reward 在 OOD scenario 可能不 robust。如果 reward model 本身有 bias，RL 会 exploit 这个 bias。
2. **Sim-to-real gap**：NAVSIM 是 non-reactive simulation，其他 agent 不响应 ego action。真实 closed-loop 中其他 agent 会反应。RAD (Gao et al. 2025) 用 3DGS environment 训 end-to-end policy，是更接近 real 的方向。
3. **Anchor 的数量选择**：$N_{\text{anchor}}$ 是 K-Means 聚类 expert behavior 得到的，固定数量。可能不覆盖 long-tail intent。
4. **Inference 仍需 selector**：虽然 generator 质量提升让 selector 不那么 critical，但最终还是要选一条执行。理想情况是 generator 直接输出 high-confidence trajectory，不需要 selector。
5. **GRPO 的 group size G**：每个 anchor 采 G 个 sample，paper 没说具体 G 值，但 G 大 training cost 高。这个 trade-off 值得 ablation。
6. **Mode Selector 的 OOD robustness**：Ablation 9 用 data augmentation 提升鲁棒性，但 selector 仍然是 supervised trained on dataset，OOD 上仍然是 weak link。

Reference:
- RAD (3DGS + RL driving): https://arxiv.org/abs/2502.13144

---

## 16. 代码 & 资源

- Paper 主页 / repo: https://github.com/hustvl/DiffusionDriveV2
- DiffusionDrive V1: https://github.com/hustvl/DiffusionDrive
- NAVSIM benchmark: https://github.com/autonomousvision/navsim
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DPPO: https://arxiv.org/abs/2409.00588
- DIVER: https://arxiv.org/abs/2507.04049
- AlphaDrive: https://arxiv.org/abs/2503.07608
- VADv2: https://arxiv.org/abs/2402.13243
- Hydra-MDP: https://arxiv.org/abs/2406.06978
- GoalFlow: https://arxiv.org/abs/2503.04031
- DriveSuprim: https://arxiv.org/abs/2506.06659
- DiffusionDrive V1 paper: https://arxiv.org/abs/2411.15249

---

## 17. 一句话总结

DiffusionDriveV2 把 GRPO 从 LLM 借过来，识别出"不同 anchor 是不同 intent，不能 cross-compare advantage"这个关键 insight，设计 Intra-Anchor GRPO 保留 multimodality + Inter-Anchor Truncated GRPO 引入 global safety floor，配 scale-adaptive multiplicative noise 保持 trajectory 结构，最终在 NAVSIM v1/v2 都刷 SOTA，把 truncated diffusion model 的 diversity vs consistent quality dilemma 给解了。
