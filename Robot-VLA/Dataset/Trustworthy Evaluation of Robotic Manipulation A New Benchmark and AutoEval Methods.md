---
source_pdf: Trustworthy Evaluation of Robotic Manipulation A New Benchmark and AutoEval
  Methods.pdf
paper_sha256: 8755031dd2b41d4da6809583564bd9064036f54c45fd5fa9d0a1d3cb006cfe9b
processed_at: '2026-08-12T18:19:24-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

好，我换个角度，用更直白的方式讲讲这篇 paper 到底在说什么，why it matters，以及背后那些作者没明说但很重要的 intuition。

---

## 一句话说清楚这篇 paper 干了啥

现在机器人圈有个 dirty secret：everyone 报 success rate，但 "success" 这个词水分很大。

这篇 paper 说：**咱们得重新定义什么叫"成功"，并且得能验证这个"成功"到底是机器人自己干的，还是人在后面偷偷帮忙。**

他们做了两件事：
1. 造了一个 dataset（Eval-Actions），里面不光有成功的 trajectory，还有失败的，还标了每个 trajectory 的质量分数和 reasoning
2. 训了一个 evaluator（AutoEval），它能看视频就能打分，还能 99.6% 准确率判断这个视频是 policy 生成的还是 human teleoperation 的

---

## 为什么要搞这个？现在的 evaluation 坏在哪？

### Gap 1: Success rate 是个谎言

想象你在评两个 robot policy：

- **Policy A**：机械臂抖了 5 次，撞了 2 次桌子，最后勉强把杯子放到位置 → Success = 1
- **Policy B**：一次流畅完成，优雅得像芭蕾舞 → Success = 1

现有 metric 给两个都打 1 分。这完全疯了。你 deploy Policy A 到真实世界，它可能把人撞伤。

这就是 paper 说的 **Execution Quality Ambiguity**：binary success rate 把 quality information 全扔了。

### Gap 2: 你怎么知道这是 robot 自己干的？

更阴险的问题。某 lab 发 paper 说他们 policy success rate 95%，但视频里那个流畅的动作，到底是 policy 生成的，还是有个 grad student 在后台用 joystick teleoperation？

你不知道。我也不知道。reviewer 不知道。这就是 **Source Authenticity Ambiguity**。

这两个 gap 加起来，意味着整个 field 的 evaluation credibility 是有 crisis 的。你没法 fair comparison，因为大家报的数字不在同一个 semantics 上。

---

## Eval-Actions 这个 dataset 为什么和别的不一样

看 Table I 就很清楚：现有的 dataset（OXE, DROID, BridgeData）全是 training-centric，几百万条 trajectory，但全是成功的人类 demonstration，没有 failure，没有 quality score，没有 reasoning。

Eval-Actions 反着来：trajectory 数量少（13k），但每条都带 dense annotation。

**关键 insight**：training 需要大量 data，evaluation 需要高质量 annotation。这是两个不同的 problem，不该用同一个 dataset 设计哲学。

### 三种 label 的直觉

- **Expert Grading (EG)**：让 10 个 expert 打分（1-10），取平均。简单直接，但 human rating 有 noise。
- **Rank-Guided (RG)**：让 expert 给一批视频排序（谁好谁差），然后用 genetic algorithm 反推每个 kinematic metric 的权重。为什么这样？因为 human 判断"谁比谁好"比"这个打 7 分还是 8 分"更可靠。相对判断比绝对判断 noise 小。
- **Chain-of-Thought (CoT)**：expert 写下打分的 reasoning process。这能训练 evaluator 不只打分，还能 explain why。

---

## Rank-Guided Weight Optimization 到底在干嘛

这部分其实在做一件事：**把物理量（velocity variance, acceleration variance 等）和人的直觉对齐**。

人看视频能判断"这个动作抖，那个动作顺"，但人说不出具体 velocity variance 是 0.3 还是 0.5。那怎么办？

公式 (1) 把 kinematic metrics 加权求和，得到 raw score：
$$S_{raw}(\theta) = \frac{\sum w_i \cdot s_i'}{\sum w_i}$$

如果 safety violation 发生，对应 metric 除以 $\lambda$（penalty divisor）。用除法而非减法，这样 violation 越严重，score 被压得越低，这是 multiplicative penalty。

公式 (2) 优化 ranking 一致性：
$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{k=1}^N|R_{human}^{(k)} - R_{raw}^{(k)}(\theta)|$$

只优化 ranking，不管 absolute score。用 genetic algorithm 搜最优权重 $\theta^*$。

公式 (3) 做 Z-score alignment，把 raw score 的分布对齐到 human score 的分布（比如 1-10 scale）：
$$S_{final} = \mu_{human} + \sigma_{human} \cdot \frac{S_{raw} - \mu_{raw}}{\sigma_{raw}}$$

这样最终得到的 score 既是 algorithmically derived 的（客观），又 aligned with human judgment（直观）。

---

## AutoEval-S 的核心 trick：Spatio-Temporal Aggregation

这部分我觉得是 paper 里最 clever 的工程创新。

### 问题

VLM 处理视频有个痛点：你想多给几帧让 model 看清 motion detail，但每加一帧就多几百个 token，VRAM 爆了。传统做法是 sparse sampling——比如从 32 帧里抽 8 帧给 model。但这样把中间的高频 motion 信息（jitter, hesitation）丢了。

### AutoEval-S 的方案

在两个 keyframe 之间，取 k 个中间帧，**把它们 spatially 拼接成一张大图**，再 resize 回标准分辨率喂给 VLM。

比如 2×2 configuration：取 target keyframe + 3 个中间帧，拼成 2×2 grid，resize，当成一张图喂给 encoder。

**为什么这有效**：VLM 的 vision encoder 本来就是处理 2D image 的。你把时间维度"折叠"进空间维度，encoder 的 attention 机制自然会在这些 patches 之间做 cross-frame reasoning。相当于免费获得了 temporal modeling capability，不用改架构。

**直觉类比**：就像把电影胶片剪下来并排贴在一张大图上。你看这张大图，能一眼看出动作的连续变化。

### Ablation 的 insight

Table V 的结果很有意思：
- 2×2：SRCC 0.84（最佳）
- 3×3：SRCC 0.77
- 4×4：SRCC 0.60（崩了）

4×4 为什么崩？因为 16 张帧拼成一张图再 resize，每帧的空间分辨率被压到 1/16，细节全没了。**信息密度有个 sweet spot**：太稀疏丢 temporal 信息，太密集丢 spatial 信息。2×2 是最佳平衡。

---

## Kinematic Calibration Signal 的设计哲学

公式 (4)(5) 从 joint trajectory 提取物理量：

$$\mathbf{v}_t = \mathbf{q}_t - \mathbf{q}_{t-1}, \quad \alpha_t = \mathbf{v}_t - \mathbf{v}_{t-1}$$

这是 standard first-order difference，算 angular velocity 和 acceleration。

然后公式 (5) 用 **max** 而非 mean 来聚合 joint variance：
$$\mathcal{U}_v = \max_{j \in \{1...J\}}(\mathbf{s}_v[j])$$

**直觉**：一个 7-DoF 机械臂，6 个 joint 都很平滑，但第 7 个 joint 严重 jitter。用 mean 会被 6 个平滑 joint 稀释掉，你检测不到问题。用 max 就能抓住这个 worst-case instability。

这些物理量被序列化成文本 prompt $I_{phys}$，和视频一起喂给 VLM。

### Table IX 的关键发现

- w/o Visual（只给物理量）：SRCC 0.54（暴跌）
- w/o Physics Prompt（只给视频）：SRCC 0.81（仍然很高）
- Full（视频+物理量）：SRCC 0.84

**这个 ablation 说明了什么**：视觉是主导信号，物理量是 calibration。VLM 已经能从视频提取 semantic quality，物理量的作用是 compensate 视频压缩带来的 high-frequency motion 信息丢失。

这很重要，因为它证明了 AutoEval 不是 "cheating" by 依赖精确的 kinematic data——它真的在看视频理解动作。

---

## AutoEval-P 和 GRPO：为什么 SFT 不够用

### 问题

CoT task 要求 model 先生成 reasoning text，再给出 score。但 SFT 训练的 model 会出现 **logic-score misalignment**：

Fig. 6 bottom 的例子很直观：QwenVL3-4B 看到双臂递毛巾的视频，毛巾掉地上了（task failed），但 model 生成 "seamless transfer"，然后打 8 分。Reasoning 和 score 完全脱节——语言在 hallucinate，分数随便给。

SFT 只能教 model "generate text that looks like reasoning"，但没法 enforce reasoning 和 score 之间的 causal consistency。

### GRPO 的 solution

GRPO 的核心 idea：**用 RL reward 强制 reasoning 和 score 对齐**。

公式 (8) 是 Gaussian kernel soft reward：
$$R_{score} = \exp\left(-\frac{(S - \hat{S})^2}{2\sigma^2}\right)$$

- $S$：ground truth score
- $\hat{S}$：predicted score
- $\sigma$：sensitivity parameter

这是 Gaussian kernel，$\hat{S} = S$ 时 reward = 1，error 越大 reward 指数衰减。比 binary reward（0/1）提供更 dense 的 gradient signal。

公式 (9) 加上 success 和 source 的 indicator reward：
$$R_{acc} = \omega_{score} \cdot R_{score} + \omega_{succ} \cdot R_{succ} + \omega_{src} \cdot R_{src}$$

公式 (11) 是 GRPO 的关键 trick——group-wise normalization：
$$A_i = \frac{R_{total}(y_i) - \mu_{group}}{\sigma_{group} + \epsilon}$$

对同一个 input，sample G 个 outputs，算 group mean/std，用 group statistics 代替 value network 估计 baseline。这比 PPO 省了一个 value network，computationally cheaper。

公式 (12) 加 KL penalty：
$$\mathcal{I}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G\left(\frac{\pi_\theta}{\pi_{old}}A_i - \beta\mathbb{D}_{KL}(\pi_\theta||\pi_{ref})\right)\right]$$

KL penalty 防止 catastrophic forgetting——RL exploration 不应该把 model 原本的语言能力搞坏。

### 为什么 GRPO 能 fix logic-score misalignment

SFT 优化的是 "生成和 ground truth 类似的 text"，但 reasoning text 和 final score 是分别生成的 token，SFT loss 不会显式惩罚 "reasoning 说失败但 score 给 8" 这种矛盾。

GRPO 的 reward function 同时 evaluate reasoning 内容和 score 数值。如果 reasoning 说 "task failed" 但 score 给 8，那 $R_{score}$ 会很低（因为和 ground truth 不匹配），整个 reward 被惩罚。Model 在 RL exploration 中会学会让 reasoning 和 score 保持一致。

Fig. 6 bottom 展示了效果：AutoEval-P 正确识别 "towel is dropped"，score 给 5，reasoning 和 score 一致。

---

## 实验结果的关键 insight

### Table II：Zero-shot VLM 完全失败

InternVL3.5-4B 没有 SFT 时 SRCC ≈ 0.01-0.02。这意味着 general-purpose VLM 对 robotic action quality 几乎零理解。这很合理——它们训练数据里没有 robot manipulation quality assessment 这种 task。

### Model scaling 有效但不是全部

SRCC 随 model size 上升：SmolVLM2.2B (0.41) → QwenVL2.5-3B (0.62) → InternVL3.5-4B (0.80)。但 AutoEval-S 用 InternVL3.5-4B 作为 backbone，达到了 0.84。说明 architecture innovation（Spatio-Temporal Aggregation）在 scaling 之外还提供了额外 gain。

### 99.6% Source Prediction Accuracy

这个数字非常 striking。它说明 policy-generated trajectory 和 human teleoperation 之间存在 statistical signature 差异。可能的原因：

- Policy 学到的是 human demonstration 的 approximation，distribution 上有 gap
- Human 有 micro-adjustments, anticipatory motion, subtle hesitation——这些 policy 很难完美模仿
- Policy 在 uncertain state 可能产生特定的 jitter pattern

这个 result 打开了 "policy fingerprinting" 的可能性——未来可能通过分析 trajectory 判断是哪个 policy 生成的。

### CoT 下的 Information Dilution

Table II 显示，所有 model 在 CoT protocol 下 SRCC 都下降。QwenVL3-4B 从 0.82 降到 0.64。

Paper 的解释：生成 reasoning tokens 占用 attention capacity，挤压了 numerical regression 精度。这是 interpretability 和 precision 之间的 trade-off。

AutoEval-P 通过 GRPO 缓解了这个问题（0.70），但还是比 AutoEval-S 的 0.84 低。作者认为这个 trade-off 值得——换一点 precision 换取 critical interpretability。

### Cross-Embodiment Generalization

Table III 在 unseen Franka embodiment 上测试：
- AutoEval-S (RG)：SRCC 0.75（从 0.84 下降 11%）
- Source Prediction：90%（从 99.6% 下降）

这说明 model 学到的不完全是 embodiment-specific feature，有一定 cross-embodiment 的 generalizable action quality signature。但 domain gap 确实存在，未来需要更多 embodiment diversity 的 training data。

### Table IV：Frame Density Ablation

从 8 帧 → 16 帧，小模型获益更大：
- SmolVLM2.2B：0.41 → 0.55（+34%）
- QwenVL3-4B：0.82 → 0.81（基本持平）

小模型 intrinsic temporal reasoning capacity 不足，external temporal information 补偿了这个缺陷。大模型本身 temporal reasoning 已经不错，extra frame 的 marginal benefit 递减。

---

## 那些 paper 没明说但很重要的 point

### 1. Evaluation 本质上是个 learning problem

传统 evaluation 是 measurement problem——定义一个 metric，测量。但这篇 paper 把 evaluation 转化为 learning problem——训一个 model 来做 evaluation。

好处：可以 leverage VLM 的 scaling law，evaluator 会越来越强。
坏处：evaluator 本身需要 validation，可能引入新的 bias。SRCC 0.84 是相对 human judgment 的 correlation，但 human judgment 本身有 noise。

### 2. Evaluator 可能成为 robotic system 的标配

现在 robot system 有 perception module, control module, planning module。未来可能有 evaluation module 作为第四个 component。Real-time evaluation 能 detect policy degradation, safety risk, 并 trigger recovery。

### 3. RL for Evaluation 是个新范式

GRPO 在 math reasoning 上的成功已经证明了 RL 能 align reasoning 和 output。这篇 paper 把这个 idea 迁移到 robotic evaluation，说明 RL for evaluation 是个通用 technique，可能适用于任何需要 reasoning-output alignment 的 task。

### 4. Data Collection 范式转变

从 "collect more successes" 到 "collect diverse quality with dense annotation"。这比单纯堆 data volume 难得多，但对 field 长期发展更重要。

### 5. Adversarial Robustness 的 open question

99.6% source discrimination 很强，但如果有人故意 train 一个 policy 来 mimic human teleoperation 的 statistical signature，这个 accuracy 还能维持吗？这是 trustworthy evaluation 的终极测试——对抗性 robustness。

Paper 没讨论这个，但这是 future work 的 critical direction。

---

## 最终的 intuition

这篇 paper 建立了一个新的 evaluation paradigm，核心思想可以浓缩为：

**Evaluation is not just measuring success—it's understanding behavior.**

传统 evaluation 把 robot behavior 压缩成一个 bit（success/fail）。这篇 paper 说，这个 compression 丢了太多 information。Behavior 有 quality dimension, 有 safety dimension, 有 efficiency dimension，还有 authenticity dimension。

要 capture 这些 dimension，你需要：
1. Dense annotation 的 dataset（Eval-Actions）
2. 能理解 spatio-temporal dynamics 的 model（AutoEval-S 的 aggregation）
3. 能做 logical reasoning 并保持 reasoning-output consistency 的 model（AutoEval-P 的 GRPO）

这和 LLM evaluation 的发展轨迹很像——从 BLEU/ROUGE 这种 surface metric，到 human evaluation，再到 LLM-as-judge。Robotic evaluation 正在走类似的路，从 success rate 到 fine-grained quality assessment。

---

## 参考

- [项目主页](https://term-bench.github.io/)
- [DeepSeek-R1 GRPO paper](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath GRPO](https://arxiv.org/abs/2402.03300)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Mobile ALOHA](https://arxiv.org/abs/2401.02117)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0](https://arxiv.org/abs/2410.24164)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [DROID dataset](https://droid-dataset.github.io/)
- [UMI](https://universal-manipulation-interface.github.io/)
- [InternVL3.5](https://arxiv.org/abs/2508.18265)
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388)

---

# Trustworthy Evaluation of Robotic Manipulation 深度解析

这篇 paper 触及了当前 robotic learning 领域一个被严重低估的问题：**evaluation crisis**。在 VLA/VA 模型狂飙突进的时代，everyone reports success rates，但几乎没人追问 "success" 这个词背后隐藏了多少 ambiguity。我分几个层次来剖析。

---

## 1. 核心问题诊断：Evaluation 的双重 Ambiguity

paper 识别出两个 critical gap：

### Gap 1: Execution Quality Ambiguity

传统 binary success rate 把 "jerky success"（抖动地完成）和 "smooth success"（优雅地完成）都打 1 分。这掩盖了 safety risk。想象一个场景：Model A 经历多次 trial-and-error，机械臂严重 jitter，最终把杯子放到位置；Model B 一次流畅完成。两者都 "success"，但 deployment readiness 天差地别。

### Gap 2: Source Authenticity Ambiguity

更阴险的问题：一段 "successful" demonstration 可能是 robust policy 生成的，也可能是人在幕后 teleoperation 的。当前 benchmark 无从验证，这让 fair comparison 变得几乎不可能——你可以声称你的 policy 很强，但可能是 human in the loop。

这两个 gap 叠加，构成了 paper 所称的 **"crisis of evaluation credibility"**。

---

## 2. Eval-Actions Benchmark 设计哲学

对比 Table I 的 dataset landscape，可以看到一个清晰的设计 shift：

| Dataset | Focus | Raw Traj. | Failures | Scoring | CoT |
|---------|-------|-----------|----------|---------|-----|
| Open X-Embodiment | Training | 1M+ | ✗ | ✗ | ✗ |
| DROID | Training | 76k | ✗ | ✗ | ✗ |
| RoboMIND | Training | 107k | 1.6k (re-grasp) | ✗ | ✗ |
| **Eval-Actions (Ours)** | **Evaluation** | 13k | **2.8k** | **✓** | **✓** |

关键 design choice：**不是最大化 trajectory count，而是最大化 annotation density**。13k trajectory 听起来比 OXE 的 1M+ 小很多，但每个 trajectory 都携带 dense multimodal supervision signals。

值得注意的细节：RoboMIND 的 "failures" 主要是 re-grasping events，而非 genuine terminal failures。Eval-Actions 的 2.8k failures 是真正的失败案例，这对训练 evaluator 至关重要——一个只见过 success 的 evaluator 永远学不会识别 failure mode。

### 三种监督信号的设计动机

**Expert Grading (EG)**：10 个 expert，三档评分（Excellent/Good/Poor），取平均降低 individual bias。简单直接但受主观影响。

**Rank-Guided preferences (RG)**：这是最 clever 的设计。EG 给绝对分数容易受 rater scale drift 影响；RG 只要求 expert 做 ranking（相对判断比绝对判断更可靠），然后用 Genetic Algorithm 反推 weight。

**Chain-of-Thought (CoT)**：提供 reasoning process，让 evaluator 不只是打分，而是 "explain why"。

### Fine-Grained Action Quality 四维定义

- **Success Rate**：binary indicator
- **Smoothness**：joint angular velocity 和 acceleration variance
- **Safety**：collision / hazardous interaction 检测
- **Efficiency**：normalized task completion time（按空间区域归一化以公平比较）

这四个维度构成了 paper 的 Fig. 2 中的 "Quality Radar Chart"。

---

## 3. Rank-Guided Weight Optimization 数学详解

这里有一个 elegant 的工程化设计：如何把 kinematic metrics 和 human intuition 对齐？

### 公式 (1)：Raw Score 计算

$$S_{raw}(\theta) = \frac{\sum w_i \cdot s_i'}{\sum w_i}, \quad s_i' = \begin{cases} s_i/\lambda & \text{if violation} \\ s_i & \text{otherwise} \end{cases}$$

变量含义：
- $\theta = \{w_{vel}, ..., w_{len}, \lambda_{coll}, \lambda_{fail}\}$：可调参数集合
- $w_i$：第 i 个 kinematic metric 的权重
- $s_i$：normalized kinematic metric（如 velocity variance, trajectory length 等）
- $\lambda$：penalty divisor，当 safety 或 success constraint 被违反时施加惩罚
- $s_i'$：penalized metric

关键 insight：用 divisor 而非 subtraction 来施加 penalty，这样 violation 越严重，metric 值被压缩得越厉害，score 越低。这是一个 multiplicative penalty scheme。

### 公式 (2)：Ranking Discrepancy Loss

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{k=1}^{N}|R_{human}^{(k)} - R_{raw}^{(k)}(\theta)|$$

- $R_{human}^{(k)}$：第 k 个样本在 human ranking 中的位置
- $R_{raw}^{(k)}(\theta)$：第 k 个样本在 algorithmic ranking 中的位置
- Loss 是 Mean Absolute Rank Difference

注意：这里优化的是 **ranking** 而非 absolute score。这是 learning-to-rank 的经典思路——human 对相对优劣的判断比对绝对分数的判断更可靠。

### 公式 (3)：Z-score Distribution Alignment

$$S_{final} = \mu_{human} + \sigma_{human} \cdot \left(\frac{S_{raw}(\theta^*) - \mu_{raw}}{\sigma_{raw}}\right)$$

变量含义：
- $\mu_{raw}, \sigma_{raw}$：raw scores（使用 optimal weights $\theta^*$ 计算）在 dataset 上的均值和标准差
- $\mu_{human}, \sigma_{human}$：expert annotations 的均值和标准差

这是一个标准的 Z-score normalization + rescaling：把 raw score 的分布对齐到 human score 的分布。因为公式 (2) 只优化 ranking，最终 $S_{raw}$ 的 magnitude 可能和 human scale（如 0-10）不一致，需要这一步 alignment。

---

## 4. AutoEval 架构深度解析

### Task Formulation 和 Kinematic Feature Extraction

公式 (4) 是标准的 discrete first-order difference：

$$\mathbf{v}_t = \mathbf{q}_t - \mathbf{q}_{t-1}, \quad \alpha_t = \mathbf{v}_t - \mathbf{v}_{t-1}$$

- $\mathbf{q}_t \in \mathbb{R}^J$：time step t 的 joint configuration
- $J \in \{7, 14\}$：DoF 数量（单臂 7 DoF，双臂 14 DoF）
- $\mathbf{v}_t$：instantaneous angular velocity
- $\alpha_t$：angular acceleration

公式 (5) 是最关键的 kinematic metric 定义：

$$\mathcal{U}_v = \max_{j \in \{1...J\}}(\mathbf{s}_v[j]), \quad \mathcal{U}_\alpha = \max_{j \in \{1...J\}}(\mathbf{s}_\alpha[j])$$
$$\mu_v = \frac{1}{T \cdot J}\sum_{t=1}^T\sum_{j=1}^J|v_{t,j}|$$

- $\mathbf{s}_v, \mathbf{s}_\alpha \in \mathbb{R}^J$：velocity 和 acceleration 的 temporal variance vectors（每个 joint 一个 variance 值）
- $\mathcal{U}_v, \mathcal{U}_\alpha$：worst-case uniformity metrics，取所有 joints 中的最大 variance
- $\mu_v$：global motion intensity（mean absolute velocity）

**设计直觉**：用 max 而非 mean 来捕获 instability。一个机械臂可能在 6 个 joint 上都很平滑，但第 7 个 joint 严重 jitter——这种局部 instability 必须被检测到。mean 会平滑掉这种 worst-case behavior，而 max 保留了它。

### 公式 (6)：多模态输入融合

$$(\hat{S}, \hat{O}, \hat{C}) = \Phi_\theta(\mathcal{F}, I_{phys}(\mathcal{U}_v, \mathcal{U}_\alpha, \mu_v))$$

- $\hat{S}$：action quality score prediction
- $\hat{O}$：task success prediction（Success/Failure）
- $\hat{C}$：trajectory source classification（Policy vs. Teleoperation）
- $\mathcal{F}$：visual keyframe sequence
- $I_{phys}$：physics-aware prompt，作为 Kinematic Calibration Signal

关键设计：$I_{phys}$ 是 **auxiliary** signal，用于 compensate video compression artifacts。视频压缩可能平滑掉高频 jitter，而 kinematic statistics 保留了真实 motion 信息。

---

### AutoEval-S: Spatio-Temporal Aggregation Strategy

这是这篇 paper 最 clever 的工程创新之一。

**问题**：增加 frame 数 N 会增强 temporal reasoning，但导致 visual tokens 爆炸，VRAM 不够用。

**传统方案**：sparse keyframe sampling——丢掉中间帧。问题：丢掉了高频 motion dynamics（如 jitter、hesitation）。

**AutoEval-S 方案**：
1. 在两个 keyframe $f_i$ 和 $f_{i+1}$ 之间，取 k 个 intermediate frames
2. 将这 k 个 intermediate frames 和 target keyframe $f_{i+1}$ **空间拼接**成一张 composite image
3. Resize composite 到标准 encoder 分辨率
4. 得到 refined sequence $\mathcal{F}' = \{f_i'\}_{i=1}^N$

**为什么这有效**：VLM 的 vision encoder 对图像的 spatial structure 很敏感。把多帧空间拼接后，encoder 会 "看到" 一个时间序列的 snapshot，相当于把 temporal information 编码进 spatial layout。然后 transformer 的 attention 机制可以在这些 spatial-temporal patches 间做 cross-frame reasoning。

这是一种 **token budget 重新分配** 技术：不增加总 token 数，但改变 token 内部的信息密度。

公式 (7) 是标准 SFT loss：

$$\mathcal{L} = -\sum_{t=1}^L \log P_\theta(y_t | y_{<t}, \mathcal{F}', I_{phys})$$

- $L$：target text sequence 长度
- $y_t$：第 t 个 token
- $y_{<t}$：preceding context tokens

模型被训练来 autoregressively 生成序列化后的 score、success label、source label。

---

### AutoEval-P: GRPO-based CoT Reasoning

这部分是 paper 最有思想性的地方。

**问题**：CoT 生成要求模型在打分前生成 reasoning text。但 SFT 训练的模型经常出现 **logic-score misalignment**——reasoning 说 "task failed"，但 score 给 8 分。这就是 Fig. 6 bottom 展示的 hallucination 问题：QwenVL3-4B 描述 "seamless transfer" 但其实是 towel drop。

**GRPO 方案**：用 RL 强制 reasoning 和 score 之间的 causal consistency。

公式 (8)：Gaussian kernel-based soft regression reward

$$R_{score} = \exp\left(-\frac{(S - \hat{S})^2}{2\sigma^2}\right)$$

- $S$：ground truth score
- $\hat{S}$：predicted score（从 CoT output 解析出来）
- $\sigma$：sensitivity hyperparameter

这是一个 Gaussian kernel，最大值 1（当 $\hat{S} = S$），随 error 增大指数衰减。相比 binary reward（0 或 1），这个 soft reward 提供了 dense gradient signal，对 regression task 更友好。

公式 (9)：Content Accuracy Reward

$$R_{acc} = \omega_{score} \cdot R_{score} + \omega_{succ} \cdot R_{succ} + \omega_{src} \cdot R_{src}$$

- $R_{succ} = \mathbb{I}(\hat{O} = O)$：success indicator（binary）
- $R_{src} = \mathbb{I}(\hat{C} = C)$：source indicator（binary）
- $\omega_{score}, \omega_{succ}, \omega_{src}$：task-specific weights

公式 (10)：Total reward

$$R_{total} = (1-\gamma) \cdot R_{acc} + \gamma \cdot R_{fmt}$$

- $\gamma$：balancing factor
- $R_{fmt}$：format reward，约束输出文本结构

公式 (11)：Group-wise normalized advantage

$$A_i = \frac{R_{total}(y_i) - \mu_{group}}{\sigma_{group} + \epsilon}$$

- $\mu_{group}, \sigma_{group}$：sampled group 内的 reward 均值和标准差
- $\epsilon$：数值稳定常数

这是 GRPO 相比 PPO 的核心简化：用 group statistics 代替 value network 来估计 baseline。对每个 input，sample G 个 outputs，用 group mean 作为 baseline，normalize advantage。

公式 (12)：GRPO objective

$$\mathcal{I}_{GRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[\frac{1}{G}\sum_{i=1}^G\left(\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}A_i - \beta\mathbb{D}_{KL}(\pi_\theta||\pi_{ref})\right)\right]$$

- $G$：group size
- $\pi_\theta$：current policy
- $\pi_{old}$：old policy（用于 importance sampling ratio）
- $\pi_{ref}$：reference policy（通常是 SFT 后的 model）
- $\beta$：KL penalty coefficient

KL penalty 防止 catastrophic forgetting——RL exploration 不应该破坏 model 原有的 language fluency。

---

## 5. 实验数据深度解读

### Table II：核心 benchmark 结果

几个 critical observations：

**1. Zero-shot VLM 完全失败**：
- InternVL3.5-4B (w/o SFT)：SRCC ≈ 0.01-0.02
- 这意味着 off-the-shelf VLM 对 robotic action quality 几乎零理解
- 强 motivation for domain-specific fine-tuning

**2. Model scaling effect**：
- SmolVLM2.2B：SRCC 0.41 (EG)
- QwenVL2.5-3B：SRCC 0.62 (EG)
- InternVL3.5-4B：SRCC 0.80 (EG)

SRCC 随 model size 单调上升，说明 action quality assessment 需要相当的 reasoning capacity。

**3. AutoEval-S SOTA**：
- EG：SRCC 0.81, $R_{\ell_2}$ 3.45
- RG：SRCC 0.84, $R_{\ell_2}$ 3.49
- Source Prediction：99.6% accuracy（RG）

99.6% 的 source discrimination 非常 striking。这意味着 policy-generated 和 teleoperated trajectories 之间存在可学习 的 statistical signature——可能是 jitter pattern、decision smoothness、或 micro-pauses。

**4. CoT 协议下的 Information Dilution Effect**：
- QwenVL3-4B：EG 0.82 → CoT 0.64（下降 22%）
- AutoEval-P：CoT 0.70（相对最 robust）

paper 把这解释为 "informational dilution"：生成 reasoning tokens 占用 attention capacity，挤压了 numerical regression 的精度。这是一个 **interpretability vs. precision** 的 trade-off。

### Table III：Cross-Embodiment Generalization

在 unseen Franka embodiment 上：
- AutoEval-S (RG)：SRCC 0.75（vs. 原始 0.84，下降 11%）
- Source Prediction：90% accuracy（vs. 99.6%）

这表明 framework 学到的不只是 embodiment-specific features，而是某种 cross-embodiment 的 "action quality signature"。

### Table IV：Frame Density Ablation

从 8 frames → 16 frames：
- SmolVLM2.2B：0.41 → 0.55（+34%）
- QwenVL2.5-3B：0.62 → 0.71（+15%）
- AutoEval-S：0.81 → 0.82

小模型从 dense frame 获益更大，因为它们 intrinsic temporal reasoning capacity 不足，external temporal information 补偿了这一缺陷。大模型本身已有较强 temporal reasoning，所以增益递减。

### Table V：Aggregation Grid Size

- 2×2：SRCC 0.84（最佳）
- 3×3：SRCC 0.77
- 4×4：SRCC 0.60（严重下降）

这个结果 counter-intuitive：stitched frames 越多反而越差？解释：4×4 拼接后每帧被 downsample 到 1/16 分辨率，spatial fidelity 严重损失。**2×2 是 sweet spot**：既保持了 temporal density，又没有过度压缩 spatial 信息。

### Table IX：Modality Importance

- w/o Visual：SRCC 0.54（暴跌）
- w/o Physics Prompt：SRCC 0.81（仍然很高）
- Full：SRCC 0.84

这个 ablation 揭示了一个重要 insight：**visual context 是主导信号，physics prompt 是 fine-grained calibration**。模型不是依赖 $I_{phys}$ 才能评估——它已经能从视频提取 semantic action quality。$I_{phys}$ 的作用是 refine motion stability 量化，补偿视频压缩损失。

---

## 6. 更深层的技术讨论

### Spatio-Temporal Aggregation vs. Video Transformer

AutoEval-S 的拼接策略让我想到 VideoLLaMA 等视频理解模型的 temporal encoding 方案。本质区别：
- Video Transformer：用 3D attention 或 temporal positional encoding 处理 frame sequence
- AutoEval-S：把 temporal 维度 "折叠" 进 spatial 维度，用 2D attention 处理

后者的 advantage 是不需要修改 VLM architecture，直接复用 image encoder。disadvantage 是拼接帧数受限于 spatial resolution budget。

### GRPO 在 Vision-Language 场景的应用

GRPO 最初是为 LLM 数学推理设计的（DeepSeekMath, DeepSeek-R1）。这篇 paper 把它迁移到 VLM 的物理推理，关键 modification 是 reward function：
- 数学推理：correctness reward
- 物理推理：hybrid reward（score regression + success classification + source classification）

这种 multi-objective reward design 是 robotic evaluation 特有的——不像数学题只有一个正确答案，robotic evaluation 需要同时判断多个维度。

### Source Authenticity 的哲学意义

99.6% 的 source discrimination accuracy 暗示了一个深刻问题：**policy behavior 和 human behavior 在微观层面是统计可区分的**。这可能是因为：
1. Policy 学到的 trajectory distribution 是 human demonstration 的 approximation，必然有 distributional gap
2. Human 有 micro-adjustments 和 anticipatory motions，policy 通常更 "直接"
3. Policy 在 uncertain state 可能产生 jitter 或 hesitation patterns

这为未来 "policy fingerprinting" 技术打开了大门。

---

## 7. Limitations 和 Future Directions

paper 自己承认两个 limitation：
1. 没有量化 generalization capabilities（text generalization score, task-scope generalization score）
2. Policy-generated data 规模和多样性有限

我认为还有几个值得探讨的方向：

**Evaluation of Evaluation**：AutoEval 本身如何被 validate？SRCC 0.84 是相对 human judgment 的 correlation，但 human judgment 本身有 noise（10 个 expert 的平均）。是否需要 "meta-evaluation" framework？

**Sim-to-Real Gap in Evaluation**：如果 evaluator 主要在 real-world data 上训练，它能否评估 sim2real policy 的质量？paper 没有明确讨论这一点。

**Adversarial Robustness**：如果有人故意生成 "looks like teleoperation" 的 policy behavior 来欺骗 source classifier，AutoEval 的 99.6% accuracy 还能维持吗？这是 trustworthy evaluation 的终极测试。

---

## 8. 对领域的启示

这篇 paper 对 robotic learning community 有几个重要 implications：

1. **Success rate 时代的终结**：binary metric 已经不够用了，future benchmark 必须包含 fine-grained quality assessment

2. **Evaluator as a Component**：未来 robotic system 可能包含 "evaluation module" 作为标配，就像现在包含 perception 和 control module 一样

3. **Data Collection 范式转变**：从 "collect more successes" 转向 "collect diverse quality with dense annotation"

4. **RL for Evaluation**：GRPO 在 evaluation task 上的成功暗示 RL 可能是 alignment evaluation model 和 human judgment 的通用方法

---

## 相关参考资源

- **项目主页**：https://term-bench.github.io/
- **GRPO 原始 paper**：[DeepSeekMath](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- **ACT 和 Mobile ALOHA**：[Mobile ALOHA](https://arxiv.org/abs/2401.02117)
- **Diffusion Policy**：[Diffusion Policy RSS 2023](https://diffusion-policy.cs.columbia.edu/)
- **OpenVLA**：[OpenVLA paper](https://arxiv.org/abs/2406.09246)
- **π0**：[π0 paper](https://arxiv.org/abs/2410.24164)
- **Open X-Embodiment**：[OXE paper](https://arxiv.org/abs/2310.08864)
- **DROID dataset**：[DROID project](https://droid-dataset.github.io/)
- **UMI**：[UMI paper](https://universal-manipulation-interface.github.io/)
- **InternVL3.5**：[InternVL3.5 paper](https://arxiv.org/abs/2508.18265)
- **Qwen3**：[Qwen3 technical report](https://arxiv.org/abs/2505.09388)
- **Action Quality Assessment 综述背景**：[Surgical skill assessment](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Towards_Unified_Surgical_Skill_Assessment_CVPR_2021_paper.pdf)

---

## Intuition 总结

这篇 paper 的核心 intuition 可以浓缩为几点：

1. **Evaluation 是一个被严重欠拟合的 problem**，需要从 binary classification 升级到 multi-dimensional regression + classification

2. **Fine-grained quality 来自 temporal density**——Spatio-Temporal Aggregation 是一种 token-efficient 的 temporal encoding 方案

3. **Logic-score alignment 需要 RL 而非 SFT**——SFT 教 model 说什么，GRPO 强制 model 说的和打的一致

4. **Source authenticity 是可验证的**——policy 和 human behavior 有统计可区分的 signature

5. **Evaluator 本身是 VLM**——这意味着 robotic evaluation 可以从 VLM 的 scaling law 中获益

这篇工作为 robotic manipulation 的 evaluation 建立了一个新的范式，本质上把 evaluation 从 "measurement problem" 转化为 "learning problem"。
