---
source_pdf: Video-R1.pdf
paper_sha256: 69574ff35e6bf69c726d7afa34b0e0a49074b685b95f5687aff0361d508e2178
processed_at: '2026-08-13T00:43:21-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Video-R1 的"人话"版本

## 故事的开头

想象一个侦探学校训练学员破案。传统方法是给学员看一段监控录像，让他说出"谁干了什么"，答对就给奖励。听起来合理，但有个 bug —— 学员可能根本没认真看视频的**时间顺序**，而是瞄了一眼某个画面就瞎猜对了。比如问"谁先动手"，学员可能只是看了一眼"某人举拳"的画面就猜，根本没注意谁先谁后。

这就是这篇 paper 要解决的核心问题。DeepSeek-R1 用 rule-based RL 在文本推理上大获成功，大家就想把这套搬到视频上。但直接搬会踩坑，因为视频多了一个**时间维度**，而 GRPO 这个算法压根不关心模型有没有用时间信息。

---

## 两个核心痛点

### 痛点一：模型爱走捷径

GRPO 的逻辑很简单：生成一堆回答，对的给 reward，错的惩罚。问题在于它只看**结果**不看**过程**。

举个具体例子（Figure 1 里展示的）：
- 问："视频中谁先到达终点？"
- 正确答案："穿红衣服的人"
- 模型 A 看了所有帧的时间顺序，发现红衣服先过线 → 答对了
- 模型 B 只看了最后一帧，发现红衣服在终点庆祝，猜是红衣服 → 也答对了

对 GRPO 来说，A 和 B 拿一样的 reward。但 A 是真正理解了视频，B 是在投机取巧。训练多了，模型就学会走 B 这条路，因为更省力。

Video-UTR (https://arxiv.org/abs/2502.12081) 这篇 paper 也发现了类似问题，管这叫 "hackable temporal reward"。

### 痛点二：数据不够

视频推理的高质量训练数据太少了。现有的视频数据集大多是"这是什么动作"、"车里坐了几个人"这种感知级问题，很少需要复杂推理。

而 image reasoning 的数据反而很多 —— 数学题、图表解读、OCR、空间逻辑，各种难度都有。

---

## T-GRPO：用一个"作弊检测器"逼模型动真格

这是全文最核心的技术创新，"人话"版如下：

### 核心比喻

继续侦探学校的比喻。老师想出一招来检测学员有没有真看视频：

每次出题出两份 —— 一份正常顺序的录像，一份打乱顺序的。然后比较学员在两份卷子上的正确率：

- 如果学员正常顺序答对的多 → 说明他真的用了时间信息 → 给**额外奖励**
- 如果两份正确率差不多 → 说明他在投机 → **不给额外奖励**

公式长这样：

$$r_t = \begin{cases} \alpha, & \text{if } p \geq \tilde{p} \\ 0, & \text{otherwise} \end{cases} \quad \text{(Eq. 1)}$$

变量解释：
- $p$：正常顺序 group 里正确回答的比例
- $\tilde{p}$：打乱顺序 group 里正确回答的比例
- $\alpha = 0.3$：额外奖励的幅度
- $r_t$：temporal reward，就是那个"额外奖励"

### 为什么要这样设计

关键 insight 是这相当于做了一个**因果测试**。如果模型的推理真的依赖时间顺序，那打乱顺序后它应该变差。如果打乱了它还是一样，那它根本没看时间顺序。

这比直接训"你必须用时间信息"聪明得多 —— 你不需要标注"这道题需要时间推理"，模型自己就会学会区分。

### 两个细节

**细节一**：temporal reward 只加在正确答案上。

$$R_i = \begin{cases} r_i + r_t, & \text{if } o_i \text{ is correct} \\ r_i, & \text{otherwise} \end{cases} \quad \text{(Eq. 2)}$$

为什么？如果错的也加，就等于"只要用时间信息，错也奖励"，这会 dilute signal。正确 + 用时间 → 双重奖励，错就是错，不给糖。

**细节二**：只对 video data 用 T-GRPO，image data 不用。因为图像没有时间维度，打乱了也没意义。

### Group size 的选择

Ordered group $G = 8$，shuffled group $\tilde{G} = 4$（一半，省算力）。

直觉是：正常 group 需要**精确估计**优势，所以多采几个；打乱 group 只是当 baseline 对照，少一点也能用。

---

## 数据策略：用图像给视频"补课"

### 比喻

假设你想训练一个能分析足球比赛录像的 AI。直接用足球录像训？数据太少。聪明做法是先让它学一堆图像级的球类知识 —— 球的形状、球员姿态、场地布局，再迁到视频上学"动作先后"。

Video-R1 干的就是这事儿。数据集 Video-R1-260k 组成：

- **Video (116k)**：通用视频，学时间推理
- **Image (146k)**：分六类补基础推理能力
  - Math 37k：公式、几何、多步符号推理
  - Knowledge 37k：世界知识 + 视觉
  - Chart 21k：图表解读
  - Spatial 20k：空间推理
  - OCR 16k：文字识别 + 推理
  - General 15k：通用视觉理解

关键比例：image 占 56%，video 占 44%。不能全是 video（推理基础不够），也不能全是 image（学不到时间维度）。

### 为什么 reasoning skill 能从 image transfer 到 video

数学推理、图表解读、空间逻辑这些能力本质上是 **modality-agnostic** 的。你在静态图上学懂了"这个柱状图说明 A 比 B 大"，到视频里看到"这个曲线在涨"，逻辑是一样的。

Ablation（Table 2）证明这点：去掉 image data，VideoMMMU 掉 4 分，MMVU 掉 3.6 分 —— 这些都是 knowledge-intensive 任务，最依赖 general reasoning skill。

---

## 训练 pipeline：两步走

### Stage 1: SFT Cold Start

用 Qwen2.5-VL-72B 给 260k 数据生成 CoT rationale，过滤后得到 165k，SFT 训 1 epoch。

这一步像是让模型先"背"一些推理范例，建立基础 reasoning pattern。没有这一步直接 RL（Table 2 的 zero variant），性能明显掉 —— VSI-Bench 从 34.6 掉到 31.8。

参考 Chu et al. "SFT memorizes, RL generalizes" (https://arxiv.org/abs/2501.17161) —— SFT 提供起跑姿势，RL 才是真正的探索。

### Stage 2: T-GRPO RL

只训 1k steps（扩展到 10k），15 小时 8×A100。1k steps 就有显著提升，说明 SFT 的初始化质量很高，RL 主要做 refine。

### Length Reward：防止模型"想太多"或"想太少"

$$R_i = \begin{cases} R_i + \omega, & \text{if correct and } l_{min} \leq \text{len}(o_i) \leq l_{max} \\ R_i, & \text{otherwise} \end{cases} \quad \text{(Eq. 5)}$$

参数：$\omega = 0.2$, $l_{min} = 320$, $l_{max} = 512$ tokens。

为什么需要这个？DeepSeek-R1 训练时 response length 会爆炸 —— 模型发现"想越长越容易对"就拼命写。但太长浪费 inference 算力。这个 reward 设个甜区，鼓励深度但不鼓励啰嗦。

Appendix A.2 的 ablation 显示：去掉 length reward，response 变短，性能反而掉。说明 reasoning depth 本身是个重要 signal。

---

## 结果"人话"解读

### 最亮眼的结果

Video-R1-7B 在 VSI-Bench（视频空间推理）上 37.1%，超过 GPT-4o 的 34.0%。

但要 critical 一点：GPT-4o 没专门优化 video frame sampling，这个比较有点 PR 意味。不过 7B 模型能达到这个水平还是 impressive。

### RL 比 SFT 更 generalizable

Table 1 有个有意思的细节：SFT 后有些 benchmark 反而掉（VideoMME 从 59.6 → 58.8），但 RL 后全升（→ 61.4）。

这再次印证 SFT 容易 overfit 训练分布，RL 探索出来的 policy 更 robust。

### More frames = better reasoning

从 16 frames → 64 frames，几乎所有 benchmark 都升。VSI-Bench 从 34.6 → 37.1。这直觉很对 —— 视频信息越完整，推理越准。但 training 时只用 16 frames（显存限制），inference 时才上 64 frames。

### T-GRPO 真的让模型用时间信息了

Figure 6 的量化：用 Qwen2.5-VL-72B 评估 responses 是否包含 temporal reasoning。

- 有 T-GRPO：75.0% 的 responses 用了 temporal reasoning
- 没 T-GRPO：只有 60.2%

14.8 个百分点的提升，直接 evidence 说明 T-GRPO 有效。

---

## 几个有意思的细节

### Response length 的 U 型曲线

Figure 5(c) 训练曲线：length 先掉后升再稳定。

"人话"解释：模型一开始沿用 SFT 学的 reasoning 风格（很长），然后发现这风格在 RL 上 reward 不高，开始"忘掉"它（length 掉），探索新风格（length 升），最后找到平衡点（稳定）。

这和 DeepSeek-R1 训练中观察到的 dynamics 类似，但在 video 上更明显 —— 因为 video input 信息密度高，模型需要重新 calibrate。

### Aha Moment

Section 3.4 说 Video-R1 会自言自语 "Wait, let me reconsider..."，重新审视视频解读。

这是 DeepSeek-R1 那个著名的 emergent behavior 在 video domain 首次观察到。直觉上，当模型遇到 temporal ambiguity（某帧模糊），反思能带来正确答案，RL 自然强化了这种 self-correction 行为。

### $\alpha$ 的 sensitivity

Figure 7 测了 $\alpha$ 从 0.1 到 0.4。0.2 和 0.3 最好，0.1 和 0.4 略差。

说明 temporal reward 太弱（0.1）信号不够，太强（0.4）会 override correctness reward。0.3 是个甜点。

---

## 通俗的总结

Video-R1 的故事可以浓缩成三句话：

1. **把文本 R1 搬到视频上会踩坑**，因为模型可能只看静态画面就猜对答案，压根不用时间信息
2. **用"打乱帧 vs 正常帧"做对比**，逼模型证明自己真用了时间顺序，否则不给额外奖励
3. **用图像推理数据打基础**，因为图像的 reasoning 数据又多又好，skill 能 transfer 到视频

这个 contrastive 思路其实挺 generalizable —— 任何 RL 任务如果有 shortcut policy 风险，都可以设计类似的 negative control。比如 code reasoning 可以打乱代码行顺序看模型有没有真理解逻辑，math reasoning 可以交换条件看模型有没有真推理。

---

## 几个我会质疑的点

1. **16 frames training 太少**。VSI-Bench 是 short video，优势明显；VideoMME 是 long video，提升最小（只有 1.8）。Long video temporal reasoning 才是真挑战。

2. **完全 random shuffle 可能不够严格**。更狠的 negative control 是 reverse frames（倒着播），或者用其他视频的 random frames 替换。完全 shuffle 保留了每帧的 local motion，模型可能还能从单帧信息猜。

3. **"超过 GPT-4o" 这个说法要打折扣**。GPT-4o 没有针对 video frame sampling 优化，对比不完全公平。

4. **T-GRPO 的 generalization**。Ablation 显示 general benchmark（MVBench 等）去掉 T-GRPO 掉分不多，主要受益来自 image data 和 overall RL。T-GRPO 的核心价值在 temporal-intensive 任务（VSI-Bench）。

5. **1k steps 就有提升**，说明 SFT 已经很关键。如果 base model 更弱，T-GRPO 的相对贡献可能更难 isolate。

---

## 联想到的工作

- **STAR-R1** (https://arxiv.org/abs/2505.15804)：CUHK 同组的，做 spatial transformation reasoning，和 Video-R1 互补 —— spatial + temporal 是 video reasoning 的两个 axis
- **Wu et al. "Reinforcing Spatial Reasoning with Visual Drawing"** (https://arxiv.org/abs/2506.09965)：让 model 在 image 上画图辅助推理，如果搬到 video 上画箭头标 motion direction 会很有趣
- **SophiAVL-R1** (https://arxiv.org/abs/2505.17018)：引入 "thinking reward"，和 Video-R1 的 length reward 思路相通
- **Video-UTR** (https://arxiv.org/abs/2502.12081)：另一种防 temporal shortcut 的设计，和 T-GRPO 可以结合
- **Kimi k1.5** (https://arxiv.org/abs/2501.12599)：image reasoning 的 R1 paradigm，Video-R1 是它的 video 版本延伸

---

## 相关 Links

- Paper: https://github.com/tulerfeng/Video-R1
- DeepSeek-R1 原始 paper: https://arxiv.org/abs/2501.12948
- GRPO 算法来源 (DeepSeekMath): https://arxiv.org/abs/2402.03300
- VSI-Bench: https://arxiv.org/abs/2412.14171
- VideoMMMU: https://arxiv.org/abs/2501.13826
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161

---

最后一句话总结：这篇 paper 的 elegance 在于它没有硬训"用时间信息"，而是设计了一个让模型自己证明"我用了"的机制 —— 这种 contrastive reward 思路比直接监督更优雅，也更 generalizable，是 R1 paradigm 在 video domain 一个很自然的延伸。

---

# Video-R1 深度解析

## 1. 研究动机与问题定位

Andrej，这篇 Video-R1 是一个相当有意思的工作，它把 DeepSeek-R1 的 rule-based RL paradigm 第一次系统地迁移到 **video reasoning** 这个领域。让我从最根本的问题开始 build intuition。

### 1.1 为什么 video reasoning 比文本 reasoning 更难

核心问题在于 **temporal shortcut**。当模型面对一个 video QA 时，它实际上有两种可能的 reasoning policy：

- **Policy A (temporal-aware)**：利用 frame 之间的时间依赖关系、motion、causal dynamics 来推理
- **Policy B (snapshot shortcut)**：只看某一帧或几帧的 superficial visual pattern，直接 "猜" 答案

GRPO 这种 outcome-only reward 机制无法区分这两种 policy —— 只要最终答案对了，reward 就给。这导致模型会倾向于 Policy B，因为 shortcut 总是更低 entanglement、更容易学。这就是 Figure 1 展示的核心问题。

Video-UTR [40] (https://arxiv.org/abs/2502.12081) 也发现了类似的 "hackable" 现象，这篇工作和它是互补的视角。

### 1.2 与已有 R1-for-vision 工作的关系

让我梳理一下 R1 paradigm 在 multimodal 上的发展脉络：

- **DeepSeek-R1** [11] (https://arxiv.org/abs/2501.12948): text domain, GRPO + rule-based reward
- **Kimi k1.5** [33] (https://arxiv.org/abs/2501.12599): image reasoning，long CoT scaling
- **Skywork R1V** [39]: image reasoning
- **Vision-R1** [14] (https://arxiv.org/abs/2503.06749): image reasoning
- **Video-R1** (本文): 第一个 video reasoning 的 R1 工作

Gap 非常清晰 —— image reasoning 已经被探索过，但 video reasoning 因为 temporal dimension 的复杂性一直被回避。

---

## 2. T-GRPO 算法深度解析

这是这篇 paper 的核心技术贡献，让我详细拆解。

### 2.1 GRPO 回顾

原始 GRPO [30] (https://arxiv.org/abs/2402.03300) 的核心思想：对于每个 question $q$，采样 G 个 responses $\{o_i\}_{i=1}^G$，用 group-relative advantage 来代替 critic 网络：

$$A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$

这里 $R_i$ 是第 $i$ 个 response 的 reward，mean 和 std 都在 group 内计算。

Policy update 用 clipped objective + KL penalty：

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

变量含义：
- $\pi_\theta$: 当前 policy (要优化的)
- $\pi_{\theta_{old}}$: 采样时的旧 policy
- $\pi_{ref}$: reference policy (通常是 SFT model), 用于 KL regularization 防止 drift
- $\epsilon$: PPO clip ratio
- $\beta$: KL penalty 系数 (本文 0.04)

### 2.2 T-GRPO 的核心创新：Contrastive Temporal Reward

T-GRPO 的关键 insight 是：**用 shuffled frames 作为 "negative control"，通过 contrast 来检测 model 是否真的利用了 temporal information**。

具体机制：

对每个 video question，模型产生两组 responses：

1. **Ordered group** $\{o_i\}_{i=1}^G$：输入正常时间顺序的 frames
2. **Shuffled group** $\{\tilde{o}_i\}_{i=1}^{\tilde{G}}$：输入随机打乱顺序的 frames

设 $p$ = ordered group 中正确答案比例，$\tilde{p}$ = shuffled group 中正确答案比例。

Temporal reward 定义为：

$$r_t = \begin{cases} \alpha, & \text{if } p \geq \tilde{p} \\ 0, & \text{otherwise} \end{cases} \quad \text{(Eq. 1)}$$

其中 $\alpha = 0.3$ 是 hyperparameter 控制 temporal reward 的 magnitude。

**直觉解释**：
- 如果模型在 ordered frames 上比 shuffled frames 上表现更好（$p \geq \tilde{p}$），说明模型确实依赖了 temporal order —— 给正向 reward
- 如果两者表现一样（$p < \tilde{p}$），说明模型在用 snapshot shortcut，不依赖时间顺序 —— 不给额外 reward

这是一个非常巧妙的 contrastive 设计 —— 它不是直接监督 "你应该用 temporal information"，而是通过比较来"测"模型是否在用。

### 2.3 Reward Shaping 的细节

最终的 augmented reward：

$$R_i = \begin{cases} r_i + r_t, & \text{if } o_i \text{ is correct} \\ r_i, & \text{otherwise} \end{cases} \quad \text{(Eq. 2)}$$

这里 $r_i$ 是 base reward，包含 correctness reward 和 format reward。

**关键设计决策**：$r_t$ 只 applied 到 correct responses。为什么？

如果也 applied 到 incorrect responses，那么：
- 正确 + 用 temporal → $r_i + r_t$ （高 reward）
- 错误 + 用 temporal → $r_i + r_t$ （这里 $r_i$ 已经是低/负的，但加了 $r_t$ 反而把错误的 reward 抬高了）

这会 dilute signal。所以只在 correct 上加 temporal reward，使得 advantage 计算时（Eq. 3），correct 且 temporal 的样本获得更高 advantage，被 reinforce 更强。

### 2.4 Length-based Reward

另一个细节是 length reward：

$$R_i = \begin{cases} R_i + \omega, & \text{if } o_i \text{ correct and } l_{min} \leq \text{len}(o_i) \leq l_{max} \\ R_i, & \text{otherwise} \end{cases} \quad \text{(Eq. 5)}$$

参数：$\omega = 0.2$, $l_{min} = 320$, $l_{max} = 512$ tokens。

**直觉**：这解决了 "overthinking" 问题。DeepSeek-R1 训练时经常出现 response length 暴涨，因为模型发现 "想得越长越容易对"。但过长会浪费 compute、降低 inference efficiency。这个 reward 设定了一个 sweet spot —— 既鼓励 reasoning depth，又限制过度膨胀。

Appendix A.2 的 ablation 显示，没有 length reward 时 response length 会下降，性能也下降 —— 说明 length 本身是一个重要的 reasoning signal。

---

## 3. 数据集构造的深度分析

### 3.1 为什么需要 image data

这是 paper 的另一个关键 insight。Video reasoning data 太稀缺了，特别是 high-quality、需要 long reasoning path 的样本。

Image reasoning data 的作用：
- **数学公式、几何、多步符号推理** (Math, 37k)
- **图表解读、定量逻辑** (Chart, 21k)  
- **OCR + 文本嵌入推理** (OCR, 16k)
- **世界知识 + 视觉** (Knowledge, 37k)
- **空间推理** (Spatial, 20k)

这些 reasoning skill 是 **modality-agnostic** 的 —— 一旦在 image 上学会，可以 transfer 到 video。

数据分布（Video-R1-260k）：
- General Video: 116k (44.6%)
- General Image: 15k
- Chart: 21k
- OCR: 16k
- Math: 37k
- Knowledge: 37k
- Spatial: 20k

总 image: 146k，总 video: 116k。Image 占 56%。这个比例很关键 —— 不能太多 image（否则模型不学 video），也不能太少（否则 reasoning 基础不够）。

### 3.2 Reward 设计的多样性

Paper 列出了 5 种 correctness reward function：

| Data Type | Reward Function |
|-----------|----------------|
| Multiple Choice | Exact match |
| Numerical QA | Binary exact match |
| OCR | Word Error Rate (WER) |
| Free-form QA | Average of ROUGE-1, ROUGE-2, ROUGE-L |
| Regression | $1 - \text{relative error}$ |

**关键观察**：绝大部分训练数据是 multiple choice 和 numerical QA，因为它们可以精确 verifiable。这对应 DeepSeek-R1 的核心 insight —— rule-based reward 必须是 reliable 且 precise 的。

但 free-form 和 regression 这种 continuous reward 在 temporal reward 计算时需要 threshold（比如 ROUGE > 0.5 才算 correct）。

### 3.3 SFT Cold Start 的重要性

用 Qwen2.5-VL-72B-Instruct 为 Video-R1-260k 生成 CoT，过滤后得到 Video-R1-CoT-165k。

Ablation Table 2 显示：跳过 SFT 直接 RL (Video-R1-7B-zero) 性能明显下降（VSI-Bench 31.8 vs 34.6）。

这印证了 [5] (https://arxiv.org/abs/2501.17161) "SFT memorizes, RL generalizes" 的发现 —— SFT 提供基础 reasoning pattern 的初始化，RL 在此基础上探索和 generalize。没有 SFT cold start，RL 从 random reasoning pattern 开始探索，搜索空间太大，1k steps 不够。

---

## 4. 训练策略与超参数

### 4.1 两阶段训练

**Stage 1: SFT Cold Start**
- Dataset: Video-R1-CoT-165k
- Base model: Qwen2.5-VL-7B-Instruct
- Epochs: 1
- Time: ~40 hours
- 输出: Qwen2.5-VL-7B-SFT

**Stage 2: T-GRPO RL**
- Dataset: Video-R1-260k
- Steps: 1k (扩展到 10k)
- Time: ~15 hours for 1k steps
- 输出: Video-R1-7B

### 4.2 Frame 设置的细节

- Training: max 16 frames, resolution 128 × 28 × 28 pixels
- Inference: 16~64 frames, resolution 256 × 28 × 28 pixels

为什么 training frame 数少？显存 + 计算效率。T-GRPO 需要同时跑 ordered 和 shuffled 两组，每个 group 8 个 responses，shuffled group 4 个 responses —— 总共每个 question 要生成 12 个 responses，frame 不能太多。

为什么 inference frame 多？提升性能。Table 1 显示从 16 → 64 frames，VSI-Bench 从 34.6 提升到 37.1。

### 4.3 Group Size 的选择

Ordered group $G = 8$，shuffled group $\tilde{G} = 4$（一半，为了 efficiency）。

GRPO 对 group size 敏感 —— 太小 advantage estimate 噪声大，太大 compute 太贵。$G=8$ 是 DeepSeek-R1 经验值。Shuffled group 减半是合理的 —— 它只是作为 baseline 对照，不需要那么精确的 estimate。

---

## 5. 实验结果深度解读

### 5.1 Main Results (Table 1)

关键观察：

| Model | VSI-Bench | VideoMMMU | MMVU (mc) | MVBench | TempCompass | VideoMME |
|-------|-----------|-----------|-----------|---------|------------|----------|
| GPT-4o | 34.0 | 61.2 | 75.4 | - | 71.9 | - |
| Qwen2.5-VL-7B (CoT) 64f | 31.4 | 50.4 | 60.0 | 59.2 | 72.9 | 59.6 |
| Qwen2.5-VL-7B-SFT 64f | 34.8 | 49.4 | 61.6 | 60.6 | 70.0 | 58.8 |
| **Video-R1-7B 64f** | **37.1** | **52.4** | **63.8** | **64.8** | **73.2** | **61.4** |

几个 interesting 的点：

1. **VSI-Bench 超过 GPT-4o**：37.1 vs 34.0，但要注意 GPT-4o 没有用 frame sampling 优化，这个比较不完全公平。但仍然说明 7B model + RL 可以达到 frontier 性能。

2. **SFT 不一定提升**：VideoMME 上 SFT 从 59.6 → 58.8 反而下降。这印证了 SFT memorize 不 generalize 的观点。但 RL 之后提升到 61.4。

3. **MVBench 提升 4.6 个点** (60.6 → 64.8)：MVBench 是 general benchmark，含 perception + reasoning。说明 RL 带来的 reasoning 提升 generalize 到 perception 任务。

### 5.2 Ablation Study (Table 2)

三个 ablation variant：

| Variant | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|---------|-----------|-----------|------|---------|-------------|----------|
| wo-image | 32.3 | 45.8 | 60.6 | 60.9 | 69.8 | 53.8 |
| wo-temporal | 32.7 | 48.3 | 62.1 | 61.1 | 71.3 | 54.5 |
| zero (no SFT) | 31.8 | 49.5 | 63.8 | 60.4 | 70.9 | 53.8 |
| **Full** | **34.6** | **49.8** | **64.2** | **62.7** | **72.6** | **57.4** |

**关键发现**：
- **wo-image**: 下降最多在 VideoMMMU (-4.0) 和 MMVU (-3.6)。这些是 knowledge-intensive 任务，说明 image reasoning data 提供的 general reasoning 能力是基础。
- **wo-temporal**: 下降最多在 VSI-Bench (-1.9) 和 VideoMME (-2.9)。VSI 是 spatial reasoning over video，强烈依赖 temporal cue。
- **zero**: VSI-Bench 下降 2.8，证明 SFT cold start 的必要性。

### 5.3 Temporal Reasoning 量化 (Figure 6)

用 Qwen2.5-VL-72B 评估 responses 是否包含 temporal reasoning：

- Video-R1 (with T-GRPO): **75.0%**
- Video-R1-wo-temporal (without T-GRPO): **60.2%**

提升了 14.8 个百分点。这是 T-GRPO 最直接的 evidence —— 它确实让模型更倾向于用 temporal information。

### 5.4 Training Dynamics (Figure 5)

三个曲线：
- **(a) Accuracy reward**: 持续上升，符合预期
- **(b) Temporal reward $r_t$**: 持续上升 —— 说明模型 progressively 采用 temporal-aware policy
- **(c) Response length**: 先下降后上升再稳定

(c) 的 U 型曲线很有意思。Paper 解释：模型 first discard sub-optimal SFT reasoning style (length drop)，然后 explore new reasoning policy (length rise)，最后 stabilize。

这个 dynamics 和 DeepSeek-R1 训练中观察到的类似，但更明显 —— 因为 video input 信息更 dense，模型需要重新 calibrate reasoning length。

### 5.5 Scaling RL (Table 3)

从 1k → 10k steps：

| Model | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|-------|-----------|-----------|------|---------|-------------|----------|
| 1k 64f | 37.1 | 52.4 | 63.8 | 64.8 | 73.2 | 61.4 |
| 10k 64f | 37.8 | 51.4 | 65.0 | 65.5 | 74.2 | 61.8 |

大部分 benchmark 提升明显（MVBench +0.7, TempCompass +1.0, MMVU +1.2），但 VideoMMMU 反而下降 1.0。这说明 RL scaling 不是 monotonic 提升，可能需要 learning rate schedule 或者 curriculum。

---

## 6. Aha Moment 现象

Section 3.4 提到 Video-R1 展现了 "aha moment" —— self-reflective 行为，模型会 revisit 对 video 的解读，特别是面对 ambiguous temporal cue 或 multi-step inference 时。

这是 DeepSeek-R1 训练中著名的 emergent behavior 在 video domain 的首次观察。Figure 4 给了一个例子。

**直觉解释**：当模型遇到 temporal ambiguity（比如某帧 motion 模糊），它会在 reasoning 中产生 "Wait, let me reconsider..." 类的反思。这种 self-correction 是 RL 探索过程中自然涌现的，因为反思能带来正确答案从而获得 reward。

---

## 7. Limitations 与未来方向

Paper Section E 列了 5 个 limitation，让我评估一下：

### 7.1 Frame Number 限制
16 frames training 限制了 long-range temporal modeling。**这是当前 video-LLM 的通病**。未来需要：
- Hierarchical temporal encoding (类似 TimeSformer)
- Token compression (类似 LLaMA-VID 的 dual-token)
- Sliding window attention 处理长 video

### 7.2 T-GRPO 计算开销
每个 question 要生成两组 responses（ordered + shuffled），且 shuffled group 也要过 vision encoder。这个 overhead 在大规模训练时显著。

可能的优化：
- **vLLM acceleration** (paper 提到)
- **Cached shuffled encoding**：vision encoder 输出可以 cache（permutation invariant pooling 层）
- **Curriculum on temporal reward**：训练后期才开 T-GRPO

### 7.3 Adaptive Length Control
当前 length reward 是 fixed $[320, 512]$。更好的做法：
- 根据 question 难度动态调整
- 用 reward model 估计 optimal length
- 类似 OpenAI o1 的 "thinking budget" 机制

### 7.4 Image-to-Video Transfer 的原理性
目前是简单混合，没有显式的 transfer 机制。可能的方向：
- **Cross-modal contrastive learning**：让 image reasoning pattern 和 video reasoning pattern 在 representation space 对齐
- **Curriculum learning**：先 image 后 video
- **Meta-learning**：显式学习 "如何 transfer reasoning skill"

### 7.5 Generalist Video Reward Model
当前是 rule-based reward，但很多 video 任务（如 video captioning、video QA 的 open-ended）没有 ground truth for rule verification。需要训练一个 video reward model —— 这是 multimodal RLHF 的核心问题。

参考相关工作：
- **Skywork-Reward** (https://github.com/Skywork/Skywork-Reward-Llama-3.1-8B)
- **RLAIF-V** (https://arxiv.org/abs/2407.05065)

---

## 8. 与相关工作的对比与启发

### 8.1 Video-UTR [40] 的关系
Video-UTR 也针对 video reasoning 的 temporal shortcut 问题，但思路不同：
- **Video-UTR**: Unhackable Temporal Reward，设计更复杂的 reward 来避免 model gaming
- **Video-R1**: Contrastive approach，用 shuffled 作为 control

两者可以结合 —— T-GRPO 提供对比信号，Video-UTR 的 reward 设计防止 reward hacking。

### 8.2 STAR-R1 [24] (https://arxiv.org/abs/2505.15804)
STAR-R1 做 spatial transformation reasoning，也是 R1 paradigm 在 multimodal reasoning 的应用。和 Video-R1 是互补的 —— spatial reasoning + temporal reasoning 是 video reasoning 的两个 axis。

### 8.3 SophiAVL-R1 [6] (https://arxiv.org/abs/2505.17018)
这篇引入 "thinking reward"，和 Video-R1 的 length reward 思路类似 —— 都是 explicit reward 来引导 reasoning style。

### 8.4 Wu et al. "Reinforcing Spatial Reasoning with Visual Drawing" [35] (https://arxiv.org/abs/2506.09965)
这是 CUHK 同组的工作，让 model 通过 visual drawing (在 image 上画图) 来辅助 reasoning。和 Video-R1 结合会很有趣 —— model 可以在 video frame 上画箭头标记 motion direction。

---

## 9. Intuitive Summary

让我用一个比喻来总结 Video-R1 的核心 idea：

想象一个学生学解 video 推理题。普通 GRPO 训练像只看最终答案对不对的学生 —— 他可能背下了一些 visual pattern，但不懂为什么。

T-GRPO 像是给学生两份考卷：一份是正常顺序的 video clip，一份是打乱顺序的。如果学生在两份卷子上分数一样，说明他根本没看时间顺序（在用 shortcut）。只有当他在正常顺序上分数更高时，才说明他真的理解了 video 的 temporal structure。

这个 contrastive mechanism 本质上是一个 **causal test** —— 它在测试 model 的 reasoning 是否 causally depends on temporal order。这是非常 elegant 的设计，比直接监督 "你必须用 temporal" 更 generalizable。

加上 image reasoning data 提供 general reasoning skill 基础，length reward 控制 reasoning depth，最终形成一个完整的 video reasoning training pipeline。

---

## 10. 我的 critical thoughts

虽然这篇工作 impressive，但有几个我会质疑的点：

1. **VSI-Bench vs GPT-4o 的比较**：GPT-4o 不是专门优化 video 的，且没说用了多少 frames。这个 comparison 的 PR 意味 > 技术意味。

2. **1k RL steps 是不是足够**：Table 3 显示 10k 更好，但 1k 已经有显著提升 —— 这说明 SFT stage 已经提供了 strong initialization，RL 主要在 refine。

3. **T-GRPO 的 generalization**：只在 video reasoning 任务上有效？还是 temporal-aware policy 会 generalize 到其他任务？Table 1 显示在 general benchmark (MVBench, TempCompass, VideoMME) 也提升，但 ablation 显示 wo-temporal 在这些 task 上下降不多 —— 说明 general benchmark 主要受益于 image data 和 overall RL，不是 T-GRPO 本身。

4. **Frame shuffling 的合理性**：完全 random shuffle 破坏 temporal order，但保留 local motion（每帧内 motion 仍存在）。可能更严格的 negative control 是 reverse frames 或者用 random frames from other videos。

5. **Long video 的处理**：16 frames 训练显然不够 long video。VSI-Bench 主要是 short video，这个 benchmark 的优势不一定 transfer 到 VideoMME 这种 long video benchmark。Table 1 也显示 VideoMME 提升最小 (1.8)。

---

## 参考 Links

- Paper GitHub: https://github.com/tulerfeng/Video-R1
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Kimi k1.5: https://arxiv.org/abs/2501.12599
- Video-UTR: https://arxiv.org/abs/2502.12081
- VSI-Bench: https://arxiv.org/abs/2412.14171
- VideoMMMU: https://arxiv.org/abs/2501.13826
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- STAR-R1: https://arxiv.org/abs/2505.15804
- SFT memorizes, RL generalizes: https://arxiv.org/abs/2501.17161
- Vision-R1: https://arxiv.org/abs/2503.06749
- MME-reasoning: https://arxiv.org/abs/2505.21327

Andrej，整体上我觉得这是一个 solid 的 foundational work。它最大的贡献是把 R1 paradigm 的边界从 text/image 推到 video，并且用 contrastive temporal reward 解决了 video-specific 的 shortcut 问题。这个 contrastive 思路其实可以推广 —— 任何 reasoning task 如果有"shortcut policy"风险，都可以设计相应的 negative control 来 force model 走 deep reasoning path。这可能是这类工作的 general methodology。
