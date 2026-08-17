---
source_pdf: Video-OPD Efficient Post-Training of Multimodal Large Language Models
  for Temporal Video Grounding via On-Policy Distillation.pdf
paper_sha256: 37a6e4ba8ad279504711a82c47507a6bbe39fa86b78a5d4abc11f2b590a4e326
processed_at: '2026-08-13T00:40:34-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 Video-OPD

## 要解决啥问题

你给AI一段视频，问它"两个人什么时候开始打架的"，它要告诉你一个时间段，比如"第15秒到第23秒"。这就叫 Temporal Video Grounding（TVG）。

## 以前两种方法都不行

### 方法一：SFT（像背标准答案）

你拿一堆"视频+正确时间段"的标注数据，让AI模仿着学。问题在于——AI学的时候看的全是**别人走对的路线**，但实际用的时候它自己一旦某一步走偏了，就彻底懵了，因为从来没练过"偏了之后怎么补救"。

打个比方：驾校教练只教你沿着理想路线开，从来没教过你"压线了怎么回正"。一上路稍微偏一点，你不会修，越偏越离谱。

### 方法二：GRPO（像只看期末总分）

让AI自己试，试完给它打一个总分（基于预测时间段跟真实时间段的重叠程度）。问题有两个：

**问题1：不知道哪步对哪步错。** 一条预测可能有好几十个token，最后只得到一个"65分"。到底是第3个token错了还是第8个token错了？模型完全不知道。它只能把65分平均分给每个token，该奖励谁该惩罚谁完全搞不清。视频越长这个问题越严重。

**问题2：太费钱了。** 为了让打分稳定一点，同一个视频要生成8遍（8次rollout），每遍都要在长视频上跑一遍完整生成。又慢又贵。

## Video-OPD 的做法

核心思路三句话：
1. **AI自己做题**（on-policy，跟GRPO一样避免distribution mismatch）
2. **旁边坐个老师，每写一个字就给一个评分**（dense token-level supervision，解决credit assignment）
3. **只做一遍**（single rollout，省掉8倍开销）

具体来说：

- Student先自己生成一条预测序列
- 拿这条序列给Teacher看，Teacher对每个token都给一个概率分数
- 如果Teacher觉得这个token"应该出现"，就给正奖励；觉得"不该出现"，就给负奖励
- Student根据每个token各自的奖励做更新

关键是：**Teacher不自己生成内容，只在Student的轨迹上打分**。这样保证Student学的是"在自己的状态下该怎么走"，而不是"在别人的状态下该怎么走"。

## TVDF：怎么挑训练数据

还有个聪明的小trick。标注数据不用来直接监督（那样又变成SFT了），而是用来**验证Teacher靠不靠谱**：

1. 先看Teacher在某道题上预测得准不准（用标准答案检查）
2. 如果Teacher自己都答错，这道题就不练了（Teacher都靠不住，学它反而有害）
3. 剩下Teacher靠谱的题里，优先挑**"Teacher很强但Student很弱"**的题——也就是Student跟Teacher分歧最大的那些

这样训练数据既保证质量（Teacher可靠），又最有信息量（Student最需要学的地方）。

## 结果怎样

- 比GRPO平均高17%（GRPO约12%）
- 训练速度是GRPO的5倍（因为只用1次rollout不用8次）
- 数据量只需GRPO的五分之一
- 多轮训练后Student能超过Teacher本身——这跟传统distillation完全不同，传统上student不会超过teacher

## 为啥Student能超过Teacher

传统distillation里Student天花板就是Teacher。但Video-OPD不一样：Student是在自己的轨迹上做优化，Teacher只是提供一个"引力场"指引方向。Student探索的状态空间可以比Teacher更广，多轮迭代后自然能青出于蓝。

## 一句话总结

**让Student自己开车，让Teacher实时点评每一把方向盘，用标准答案筛掉不靠谱的Teacher、专挑分歧大的弯路练。又快又好，还能超过老师。**

---

# Video-OPD：Temporal Video Grounding 的高效 post-training 框架

## 一、整体框架与 motivation 的三层递进

这篇 paper 的核心 insight 可以从一张图（Figure 1）的三列读出来：SFT、GRPO、Video-OPD 各自有不同的 trade-off。让我先从 task 本身讲起，这样能 build 你的 intuition。

### 1.1 TVG 的形式化定义

TVG (Temporal Video Grounding) 的目标是：给定 video $v$ 和 natural-language query $q$，预测一个 temporal boundary $[t_s, t_e]$，使得这个时间区间对应 query 描述的内容。例如 query "两个人打架" 时，模型需要输出打架发生的 start time $t_s$ 和 end time $t_e$。

MLLM (Multimodal Large Language Model) 把 TVG 形式化为 autoregressive decision process：
- state: $s_t = (v, q, a_{<t})$，即当前 step 的 context 包含 video、query 以及之前已经生成的 actions
- action: $a_t \in \mathcal{A}$，其中 $\mathcal{A}$ 是 discrete temporal action space（比如 tokenized 的时间戳）
- trajectory: $\tau = (a_1, a_2, \ldots, a_T)$，即模型逐步生成的 token 序列

关键点在于：**early action 的错误会 propagate 到后续 step**，因为后续 state 依赖前面生成的 action。这和传统 language generation 不同，TVG 的 token 之间存在强耦合的 temporal dependency。

### 1.2 SFT 为何失败：off-policy distributional mismatch

SFT 的优化目标是：
$$\min_\theta \mathbb{E}_{\tau \sim p_{\text{data}}(\tau)} [\mathcal{L}_{\text{CE}}(\theta; \tau)]$$

这里 trajectory $\tau$ 来自固定数据分布 $p_{\text{data}}$，即 ground-truth demonstration。但 inference 时执行的是 $\tau \sim \pi_\theta(\tau)$，即模型自己的 policy 采样。

**Compounding error 的本质**：模型训练时看到的所有 state 都是"expert 在正确轨迹上产生的 state"，但 inference 时一旦某一步偏离 expert，state 就变成 expert 从未见过的 distribution，后续预测会越来越离谱。这在 TVG 里尤其严重，因为 temporal prediction 的 step 之间强耦合——如果 $t_s$ 预测偏了 0.5 秒，后续 $t_e$ 的预测条件概率分布就完全错位。

你可以把这个想象成开车：SFT 教你沿着高速公路中线开，但没人教过你"已经偏出车道了怎么办"。一旦偏一点，模型 panic，越来越偏。

### 1.3 GRPO 为何也不够：两个 bottleneck

Time-R1 和 TVG-R1 用 GRPO 做 post-training，优化目标（Eq. 1）：

$$\max_\theta \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ R(\tau) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

变量含义：
- $\pi_{\theta_{\text{old}}}$：上一个 optimization step 的 policy，用于 sample trajectory
- $\pi_{\text{ref}}$：reference policy（通常是 pretrained model），anchor 住防止 policy 跑偏
- $\beta$：KL regularization 强度
- $R(\tau)$：trajectory-level reward

Group-relative normalized reward（Eq. 2）：
$$R(\tau) = \frac{1}{G} \sum_{i=1}^{G} \frac{\pi_\theta(\tau_i)}{\pi_{\theta_{\text{old}}}(\tau_i)} \cdot \hat{r}_G(\tau_i)$$

- $G$：group size，即同一 sample rollout 几次
- $\{\tau_i\}_{i=1}^G$：从 $\pi_{\theta_{\text{old}}}$ 采样的 $G$ 条 trajectory
- $\pi_\theta(\tau_i)/\pi_{\theta_{\text{old}}}(\tau_i)$：importance sampling ratio（PPO-style）
- $\hat{r}_G(\tau_i)$：group-normalized reward，通常是 $(r_i - \bar{r})/\sigma_r$

GRPO 的 **Bottleneck 1：Sparse reward 导致 credit assignment 失败**

Policy gradient（Eq. 3）：
$$\nabla_\theta \mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ R(\tau) \nabla_\theta \log \pi_\theta(\tau) \right]$$

由于 $R(\tau)$ 是 trajectory-level scalar，对所有 timestep $t \in [1, T]$ 都是同一个值，它被 uniformly propagated 到每个 token。无论 trajectory 长度 $T$ 多大，每个 episode 只提供 $\mathcal{O}(1)$ 的 feedback。这意味着：
- 如果 trajectory 最终 IoU=0.6，模型不知道是 step 3 的预测错了还是 step 8 的预测错了
- 梯度方差随 $T$ 增长（后面 Appendix A 会证明）
- 长 trajectory（视频长）下尤其灾难

GRPO 的 **Bottleneck 2：Multi-rollout overhead**

为了降低 reward 估计的方差，GRPO 需要 $G$ 次 rollout（论文里 $G=8$）。每次 rollout 都要 condition 在长 video context（论文里 8192 video tokens，768 frames @ 2 FPS）上做 full autoregressive generation。算力成本随 $T \times G$ 缩放，长视频下 prohibitive。

### 1.4 设计需求

从 SFT 和 GRPO 的失败得到三个 design requirement：
1. **On-policy**：必须在自己的 policy 上 sample trajectory，避免 distributional mismatch
2. **Dense supervision**：每个 token 都要有 per-step 的 supervision，避免 credit assignment
3. **Single rollout**：每个 sample 只 sample 一次 trajectory，避免 multi-rollout overhead

Video-OPD 同时满足这三点。

---

## 二、Video-OPD 的方法核心

### 2.1 四步 pipeline

**Step 1: On-Policy Trajectory Sampling**（Eq. 4）

$$\tau = (a_1, \ldots, a_T) \sim \pi_\theta(\cdot \mid v, q)$$

Student 自己 sample trajectory，同时记录每个 token 的 log-probability $\log \pi_\theta(a_t \mid s_t)$，用于后续 importance weighting。

这一步保证 training 和 inference 的 state distribution 一致，这是 on-policy 的核心。

**Step 2: Teacher Evaluation on Student Trajectory**

固定一个 high-capacity teacher $\pi_{\text{tea}}$，对 student sample 出来的 trajectory 上每个 token 计算 conditional log-probability $\log \pi_{\text{tea}}(a_t \mid s_t)$。

**Critical subtlety**：teacher 自己不生成 token！teacher 只在 student 的 on-policy state distribution 上做 scoring。这是 on-policy distillation 和 off-policy distillation 的关键区别——off-policy distillation 中 teacher 评估的是 ground-truth trajectory 上的 token，分布对应 corpus 而非 student policy。

**Step 3: Dense Token-Level Supervision**（Eq. 5）

$$r_t = -\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{tea}}(a_t \mid s_t) \right)$$

这是 reverse KL divergence 的 pointwise contribution。让我拆开看：
- $\log \pi_\theta(a_t \mid s_t)$：student 给自己采的 token $a_t$ 的 log-prob
- $\log \pi_{\text{tea}}(a_t \mid s_t)$：teacher 给同一个 token 的 log-prob
- $r_t > 0$ 当 teacher 觉得这个 token "应该"（prob 比 student 高）
- $r_t < 0$ 当 teacher 觉得这个 token "不该"（prob 比 student 低）

直觉上：teacher "支持" 的 token，$r_t$ 给正奖励，鼓励 student 增加 prob；teacher "反对" 的 token，$r_t$ 给负奖励，惩罚 student。

**Step 4: Policy Update**（Eq. 6）

$$-\mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^T r_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

变量：
- $\pi_\theta(a_t \mid s_t)/\pi_{\theta_{\text{old}}}(a_t \mid s_t)$：importance sampling ratio，让旧 policy sample 的 trajectory 可以用新 policy 的 gradient 更新
- $r_t$：per-token reward，替换 GRPO 中的 $R(\tau)$

每个 action $a_t$ 都有 step-specific reward $r_t$，实现 fine-grained credit assignment。

### 2.2 与 GRPO 的本质区别

| 维度 | GRPO | Video-OPD |
|------|------|-----------|
| Reward 来源 | 外部 verifiable reward (IoU) | Teacher log-prob 差 |
| Reward 粒度 | Trajectory-level ($\mathcal{O}(1)$) | Token-level ($\mathcal{O}(T)$) |
| Rollout 数量 | $G$ 次 (论文 $G=8$) | 1 次 |
| Supervision 类型 | Sparse scalar | Dense per-token signal |
| Credit assignment | Ambiguous | Precise |

### 2.3 与 off-policy distillation 的区别

Paper 在 Appendix C 实现了两个 baseline：OP-RKD (Off-Policy Reverse KL) 和 OP-FKD (Off-Policy Forward KL)。

OP-RKD 的 reward 公式和 Video-OPD 完全相同（Eq. 24 vs Eq. 5）：
$$r_t = -(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\text{tea}}(a_t \mid s_t))$$

唯一区别：OP-RKD 评估的 token 是 ground-truth corpus token $a_t \sim p_{\text{data}}$，而非 student sample 的 token。这意味着 teacher 提供的 supervision aligned with corpus distribution，导致训练时 state distribution 和 inference 时 mismatch。

OP-FKD 用 forward KL（Eq. 27）：
$$\ell_t = \text{KL}(P_{\text{tea}}(\cdot \mid s_t) \| P_{\text{stu}}(\cdot \mid s_t)) = \sum_{w \in \mathcal{V}} P_{\text{tea}}(w \mid s_t) \log \frac{P_{\text{tea}}(w \mid s_t)}{P_{\text{stu}}(w \mid s_t)}$$

变量：
- $\mathcal{V}$：vocabulary
- $w$：vocabulary 中的某个 token
- $P_{\text{tea}}$ / $P_{\text{stu}}$：teacher / student 在 state $s_t$ 下的 token 分布

Forward KL 是 **mode-covering**（mean-seeking），鼓励 student 把 mass 分散到 teacher 所有可能的 mode；reverse KL 是 **mode-seeking**，鼓励 student 集中在 teacher 的 peak。Table 1 实验显示 OP-RKD 和 OP-FKD 都显著弱于 Video-OPD，证明 on-policy 本身是关键。

---

## 三、理论分析（Appendix A）—— 这部分对你的 intuition 很关键

### 3.1 Reward density 的比较

GRPO 的 policy gradient（Eq. 7）：
$$\nabla_\theta J_{\text{GRPO}} = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ R(\tau) \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

由于 $R(\tau)$ 与 $t$ 无关，所有 time step 共享同一个 scalar reward。

Video-OPD 的 policy gradient（Eq. 9）：
$$\nabla_\theta J_{\text{OPD}} = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^T r_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]$$

每个 time step 有独立的 $r_t$，reward 密度从 $\mathcal{O}(1)$ 提升到 $\mathcal{O}(T)$。

### 3.2 Variance reduction 的证明

Policy gradient 的 variance 分解（Eq. 10）：
$$\text{Var}\left(\sum_{t=1}^T A_t \nabla_\theta \log \pi_t\right) = \sum_{t=1}^T \text{Var}(A_t \nabla_\theta \log \pi_t) + 2 \sum_{t < t'} \text{Cov}(A_t \nabla_\theta \log \pi_t, A_{t'} \nabla_\theta \log \pi_{t'})$$

- GRPO 中 $A_t \equiv R(\tau)$，对所有 $t$ 相同，导致 cross-time covariance 项 $\text{Cov}(\cdot, \cdot)$ 很大且 positive correlated，方差随 $T$ 平方级增长
- Video-OPD 中 $A_t = r_t$ 只依赖 local $(s_t, a_t)$，teacher 是固定的且 evaluate on student state distribution，所以 $r_t$ 跨 time step 大致 decorrelated，cross-covariance 项大幅减小

### 3.3 与 KL minimization 的等价性

Eq. 11 证明：
$$\mathbb{E}_{a_t \sim \pi_\theta} \left[ r_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \nabla_\theta D_{\text{KL}}(\pi_\theta(\cdot \mid s_t) \| \pi_{\text{tea}}(\cdot \mid s_t))$$

这意味着 Video-OPD 等价于在每个 visited state $s_t$ 上做 reverse KL 的 stochastic gradient descent。Reverse KL 是 smooth、convex-in-student 的目标（在 exponential family 下），收敛性比 GRPO 的 trajectory-level non-convex objective 好得多。

### 3.4 Convergence rate

对于 smooth objective + stochastic gradient，optimality gap 以 $\mathcal{O}(1/\sqrt{K})$ 收敛，常数项 $\propto \sqrt{\text{Var}}$。Video-OPD 方差低 + smooth KL 目标，所以收敛更快。

---

## 四、TVDF：Teacher-Validated Disagreement Focusing

TVDF 是一个 lightweight curriculum，把 ground-truth annotation 当 validation signal 而非 supervision target。两部分：

### 4.1 TRPV (Teacher Reliability Pre-Validation)

对每个 (video, query) pair $(v, q)$：
- Teacher 预测 top-$k$ 个 temporal boundary
- 计算 top-$k$ IoU 的 mean，与 ground-truth annotation 比较
- 若 mean IoU 低于阈值 $\tau_{\text{tea}}$，标记 sample 为 teacher-unreliable，从 training 中剔除

直觉：如果 teacher 自己都预测不准，distill 它的 log-prob 反而会害 student。所以先用 ground-truth annotation 过滤一遍。

### 4.2 DBTP (Disagreement-Based Trajectory Prioritization)

对 teacher-reliable 的 sample：
- 计算 teacher IoU $\tau_i$（teacher 预测和 ground-truth 的 IoU）
- 计算 student IoU $\sigma_i$（student 预测和 ground-truth 的 IoU）
- 定义 disagreement $\delta_i = \tau_i - \sigma_i$
- 优先选 $\delta_i$ 大的 sample（即 teacher 强但 student 弱）

附录 B 提供四种 sampling 策略：

**DSUS (Difference-Sorted Uniform Sampling)** — Eq. 13-15：按 $\delta_i$ 降序排序，然后均匀采样 $k$ 个 sample，保持多样性。

**Top-k DBS** — Eq. 16：直接选 $\delta_i$ 最大的前 $k$ 个 sample。论文默认采用这个。

**BBDS (Bucket-Balanced Difference Sampling)** — Eq. 17-20：把 $\delta_i$ 范围分 $B=5$ 个 bucket，每个 bucket 内按 $\delta_i$ 降序均匀采样，保证 disagreement 各级都有。

**GWDS (Gaussian-Weighted Difference Sampling)** — Eq. 21-22：以中心 $c=0.9$、标准差 $\sigma=0.2$ 的高斯分布加权采样：
$$p_i = \frac{1}{Z} \exp\left(-\frac{(\delta_i - c)^2}{2\sigma^2}\right), \quad Z = \sum_j \exp\left(-\frac{(\delta_j - c)^2}{2\sigma^2}\right)$$

Table 7 显示 Top-k DBS 在三个 benchmark 上全面最优。

### 4.3 TVDF 与传统 active learning 的区别

传统 active learning：用 model uncertainty 选 sample，但 uncertainty 不一定 = informativeness。

TVDF 的关键 insight：**ground-truth annotation 不再用于 SFT supervision**（因为 SFT off-policy），而是用来 verify teacher 是否可靠。一旦 teacher 可靠，就可以放心用 teacher log-prob 做 dense supervision。这样 ground-truth 只用一次（验证），后续 supervision 来自 teacher，比直接 SFT 更 on-policy。

---

## 五、实验结果详解

### 5.1 主实验 Table 1

在三个 TVG benchmark 上比较：Charades-TimeLens、ActivityNet-TimeLens、QVHighlights-TimeLens。Metric 是 R@0.3, R@0.5, R@0.7（IoU 阈值下的 Recall）。

关键对比：
- **vs Gemini-2.5-Pro（closed-source SOTA）**：在 QVHighlights 上 Video-OPD 的 R@0.7 = 50.4，超过 Gemini-2.5-Pro 的... 等等，看 Table 1：Gemini-2.5-Pro 在 QVHighlights R@0.7 = 61.1，Video-OPD 是 50.4。所以还是落后于最强 closed-source。但已超过 Gemini-2.5-Flash 的 55.0。论文 abstract 说"approaches Gemini-2.5-Flash, even surpasses on several benchmarks"。

- **vs GRPO [Qwen3-VL-8B]**：Video-OPD 全面优于 GRPO。比如 QVHighlights mIoU（看 D.4 节有完整数）：
  - GRPO: mIoU = 54.3 (作为 teacher)
  - Video-OPD Round 1: mIoU = 62.3

平均改进超过 17%，而 GRPO 大约 12%。

- **vs OP-RKD / OP-FKD**：off-policy distillation 全面弱于 Video-OPD，证明 on-policy 是关键。OP-RKD 在 QVHighlights mIoU 57.4（reverse KL 但 off-policy）vs Video-OPD 61.0（reverse KL + on-policy）。

### 5.2 Generalization 到 general video understanding

Figure 4 显示在 TempCompass、MVBench、Video-MME 上：
- OP-RKD / OP-FKD（off-policy distillation）明显 degrade（因为 corpus distribution mismatch 影响其他能力）
- GRPO 和 Video-OPD 都保持或略提升性能
- Video-OPD SOTA

**Why OP-RKD/FKD degrade?** Off-policy distillation 让 student 适配 corpus trajectory distribution，偏离了 general video understanding 的 natural distribution，catastrophic forgetting。On-policy 让 student 只在自己的 distribution 上学习，不破坏其他能力。

### 5.3 Ablation: TVDF 的两个组件（Table 2）

在 Charades/ActivityNet/QVHighlights 上 R@0.7：
- Base (无 TVDF): 31.1 / 33.5 / 47.7
- +TRPV only: 31.7 / 34.2 / 48.2
- +DBTP only: 31.3 / 34.6 / 50.0
- +TRPV+DBTP: 32.4 / 35.8 / 50.4

两个组件单独都有效，组合后最优。DBTP 贡献略大于 TRPV。

### 5.4 Multi-round training（Figure 5, Table 6）

TVDF 可以迭代应用，每轮基于当前 student 重新计算 disagreement：
- Round 1: QVHighlights mIoU = 62.3
- Round 2: mIoU = 63.1
- Round 3: mIoU = 64.6

Round 3 已经超过 teacher (Qwen3-VL-32B-GRPO mIoU = 58.4)。这证明 Video-OPD 不受 teacher capacity 上界约束——student 可以超过 teacher。这是个有意思的现象，paper 解释为 multi-round 的 iterative refinement 让 student 学到 teacher 在不同 state 上的"集成"信息。

### 5.5 不同 teacher 的影响（Table 3, Table 5）

Student = Qwen3-VL-8B-Instruct，QVHighlights 上：

| Teacher | Teacher mIoU | Student mIoU | Gap |
|---------|--------------|--------------|-----|
| 4B-GRPO | 54.9 | 62.5 | +7.6 |
| 8B-GRPO | 54.3 | 62.3 | +8.0 |
| 32B-GRPO | 62.8 | 62.9 | +0.1 |

观察：
1. 4B teacher 就能让 8B student 大幅提升（+7.6）——证明 Video-OPD 对 teacher 规模不敏感
2. 32B teacher 几乎被 student 追平——单 round 即可
3. **Student 不被 teacher 上界限制**：4B teacher (54.9) → 8B student (62.5)，student 远超 teacher。这违反了传统 distillation 的直觉——传统上 student ≤ teacher。

**Why student > teacher?** 我的解读：
- Teacher 是 GRPO 训练的，有 sparse reward 引起的 sub-optimality
- Video-OPD 给 student 提供 dense supervision，让 student 在 teacher 的基础上做更精细的优化
- 多轮迭代相当于 ensemble 多个 teacher state distribution 的信息

### 5.6 Thinking mode 反而有害（Table 8）

- No-thinking: Charades mIoU = 52.0
- Thinking: Charades mIoU = 48.7（下降 6.3%）

TVG 是 visual perception 主导的任务，explicit thinking process（CoT）不仅无益还引入 noise。这与 TimeLens、VideoChat-R1 的观察一致。Insight：visual perception 任务不需要 verbal reasoning chain。

### 5.7 Convergence & Cost（Figure 6）

- 左中图：Video-OPD 收敛速度远快于 GRPO，最终性能更高
- 右图：相同 training step 下，Video-OPD 用时仅为 GRPO 的 ~20%

GRPO 用 8 rollouts，每个 rollout 都要 full autoregressive generation on long video context。Video-OPD 用 1 rollout，但 teacher 只需要 forward pass（无 generation），所以快 5x 是合理的。

### 5.8 vs TimeLens-8B（Table 4）

TimeLens-8B: 先 SFT 再 GRPO，用 12k samples。
Video-OPD: pure on-policy distillation，2.5k samples（Round 1）。

| Model | QVHighlights mIoU |
|-------|-------------------|
| TimeLens-8B | 63.0 |
| Video-OPD R1 | 61.0 |
| Video-OPD R2 | 63.1 |
| Video-OPD R3 | 64.6 |

Round 3 时用 7.5k samples（vs TimeLens 12k）超过 TimeLens 1.6 mIoU。Video-OPD 在 sample efficiency 上碾压。

---

## 六、Intuition 总结与 critical reflection

### 6.1 整体 intuition

你可以把 Video-OPD 想象成"自驱动 + 专家旁听"：
- Student 自己开赛车（on-policy sampling），自己犯错、自己纠正
- Teacher 是副驾驶的资深教练，每个动作给即时评分（dense token-level reward），但不替 student 开车
- GRPO 是只有终点计时（trajectory-level reward），教练只能看总成绩
- SFT 是教练按教材开车，student 跟着模仿，但 student 一旦偏了教材路线就完了

### 6.2 Reverse KL vs Forward KL 的直觉

- Reverse KL（Video-OPD/OP-RKD）：$\text{KL}(\pi_\theta \| \pi_{\text{tea}}) = \mathbb{E}_{a \sim \pi_\theta}[\log \pi_\theta - \log \pi_{\text{tea}}]$。Student 在自己采样到的 token 上要求 teacher 概率高。Student 是 mode-seeking：只覆盖 teacher 的高概率区，避开 teacher 低概率区。
- Forward KL（OP-FKD）：$\text{KL}(\pi_{\text{tea}} \| \pi_\theta) = \mathbb{E}_{a \sim \pi_{\text{tea}}}[\log \pi_{\text{tea}} - \log \pi_\theta]$。要求 student 在 teacher 高概率的所有 token 上都给高概率。Student 是 mode-covering：尽量覆盖 teacher 的所有 mode，包括低概率的。

TVG 这种精确 prediction 任务更适合 mode-seeking（reverse KL），因为只要 teacher 高概率的 prediction 正确就够。

### 6.3 我对这篇 paper 的几个 critical observations

1. **Teacher 必须有 GRPO 增强**：所有实验的 teacher 都是 Qwen3-VL-xB-GRPO。如果 teacher 是 vanilla instruct model，效果可能差很多。这说明 Video-OPD 实际上是"GRPO 的成果 transfer 给 student"，比从头 GRPO 更 efficient。这有点像 RLHF + distillation 的组合。

2. **Ground-truth 的间接使用**：TVDF 用 ground-truth IoU 来 verify teacher，这本质是利用了 ground-truth 但不直接 supervision。这种 indirect use 很聪明——避免了 off-policy distributional mismatch，又利用了 annotation 信息。但 require ground-truth IoU，所以本质还是 supervised setting，不是完全 unsupervised。

3. **Multi-round 超过 teacher 的现象**：这个其实有理论解释。On-policy distillation 在 student 自己 sample 的 trajectory 上做 KL minimization，这相当于让 student 在 teacher 的"guidance field" 下做 self-consistent optimization。多轮迭代让 student 探索 teacher 没有充分覆盖的 state space，最终 student 可以学得比 teacher 更精细。这是 RL 中 "iterative policy improvement" 的体现。

4. **是否真的需要 frontier teacher?** Limitation section 说可以 online distill from 多个 domain-specific experts。但实验只展示了 GRPO-trained teacher。如果 teacher 是 SFT-only model，效果未知。

5. **Thinking mode 无益**：这是 TVG 作为 perception task 的特性，对 reasoning-heavy 的 video QA 任务可能相反。Paper 的结论不能外推到所有 video task。

### 6.4 与 broader literature 的联系

- **Thinking Machines Lab 的 on-policy distillation blog**（[thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation)）：Video-OPD 直接 cite 这个工作。TML 的 insight 是 unify on-policy RL + distillation，Video-OPD 是把这个 idea 应用到 TVG。
- **MiMo-V2-Flash、Qwen3、Qwen3-VL**：这些 production model 都用了 on-policy distillation 做 post-training，证明这个 paradigm 的 industrial scalability。
- **Time-R1, TVG-R1, Tempo-R0**：TVG 上的 RL 先驱，但都用 GRPO，受 sparse reward 限制。
- **TimeLens** ([github.com/TencentARC/TimeLens](https://github.com/TencentARC/TimeLens))：提供了 corrected benchmark 和 100K 训练数据。

### 6.5 与 GRPO 在数学上的关键差异再总结

让我重新写一遍两种 policy gradient 的对比，强调 credit assignment：

**GRPO**（trajectory-level reward $\bar{R}$）：
$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[\bar{R} \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

每个 token 的 gradient 都是 $\bar{R} \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)$，$\bar{R}$ 对所有 $t$ 相同。如果 $\bar{R} > 0$，所有 token 的 log-prob 都被 increase；如果 $\bar{R} < 0$，所有 token 都被 decrease。但实际只有少数 token 影响最终 IoU（比如 boundary token $t_s, t_e$），其他 reasoning token 是 noise。Credit assignment 失败。

**Video-OPD**（token-level reward $r_t$）：
$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[\sum_{t=1}^T r_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)\right]$$

每个 token 有自己的 $r_t$，由 teacher log-prob 决定。如果 token $a_t$ 是 teacher 高度认可的，$r_t > 0$ 大；如果 teacher 反对，$r_t < 0$。Student 可以精确知道每个 token 该增该减。

这是 **distillation 把 credit assignment 问题转化为 teacher scoring 问题**。Teacher 提供的 dense signal 等价于一个 perfect credit assigner——只要 teacher 本身是 good predictor。

---

## 七、参考资料

1. **Thinking Machines Lab - On-Policy Distillation blog**: https://thinkingmachines.ai/blog/on-policy-distillation （Video-OPD 的理论源头）
2. **TimeLens GitHub** (TencentARC): https://github.com/TencentARC/TimeLens （训练数据 + benchmark）
3. **Time-R1 paper** (arXiv 2503.13377): https://arxiv.org/abs/2503.13377 （TVG 上的 RL 先驱）
4. **TVG-R1 paper** (Chen et al., 2025): EMNLP 2025 Industry Track
5. **DeepSeekMath / GRPO paper** (arXiv 2402.03300): https://arxiv.org/abs/2402.03300 （GRPO 原文）
6. **Qwen3-VL technical report** (arXiv 2511.21631): https://arxiv.org/abs/2511.21631 （teacher model）
7. **Qwen3 technical report** (arXiv 2505.09388): https://arxiv.org/abs/2505.09388
8. **MiMo-V2-Flash technical report** (arXiv 2601.02780): https://arxiv.org/abs/2601.02780
9. **VideoChat-R1** (arXiv 2504.06958): https://arxiv.org/abs/2504.06958 （video RL fine-tuning）
10. **MVBench** (CVPR 2024): https://arxiv.org/abs/2403.00476
11. **TempCompass** (arXiv 2403.00476): https://arxiv.org/abs/2403.00476
12. **Video-MME** (CVPR 2025): https://arxiv.org/abs/2503.00476
13. **Charades-STA / ActivityNet / QVHighlights** (Gao et al. 2017, Caba Heilbron et al. 2015, Lei et al. 2021): TVG 经典 benchmark

---

## 八、延伸思考

### 8.1 为什么 on-policy 比 off-policy 重要？

Off-policy 的根本问题：state distribution mismatch。在 TVG 这种 long-horizon autoregressive setting，每一步都依赖前一步的输出，error compounding 严重。On-policy 通过让 student 在自己的 distribution 上 sample trajectory，保证 training state 和 inference state 同分布。

### 8.2 为什么 dense supervision 比 sparse reward 重要？

Sparse reward 的根本问题：variance 和 credit assignment。Trajectory-level reward 给所有 token 同一个 scalar，gradient variance 随 $T$ 平方增长，长 trajectory 下 gradient noise 淹没 signal。Dense per-token reward 把 variance 从 $\mathcal{O}(T^2)$ 降到 $\mathcal{O}(T)$（diagonal term only），且每个 token 有独立 supervision。

### 8.3 为什么 single rollout 够？

Video-OPD 不需要 estimate trajectory-level reward 的 mean/variance，因为 reward 来自 teacher log-prob，本身就是 deterministic function of (state, action)。每个 sample 单次 rollout 就能给 dense signal。GRPO 需要 8 rollouts 是为了 normalize reward 减少 variance，Video-OPD 不需要这个 trick。

### 8.4 Video-OPD 的局限

- 需要一个 strong teacher（虽然不必 frontier，但至少 GRPO-trained）
- Teacher forward pass 仍然有 cost（虽然比 generation 快）
- TVDF 依赖 ground-truth IoU 做 teacher validation，所以仍需 annotated data
- 4B teacher 实验显示，若 teacher 本身太弱，student 提升空间有限（虽然 student 仍能超过 teacher）

### 8.5 可能的扩展

- **Multiple teachers ensemble**：不同 teacher 覆盖不同 state，aggregation 提升信号质量
- **Self-distillation**：student 自己经过几轮 GRPO 后变成 teacher，循环提升（类似 AlphaGo 的 self-play）
- **Cross-modal teacher**：用 video-only model 做 teacher，distill 给 MLLM
- **Active teacher querying**：TVDF 选 disagreement 大的 sample 不仅用于 training，还可以 query teacher 在这些 sample 上做 fine-tuning

### 8.6 与 DAgger 的联系

Video-OPD 在精神上类似 DAgger (Dataset Aggregation)：DAgger 通过 student 自己 rollout 收集 state，然后 query expert label 这些 state，iteratively 训练。Video-OPD 是 DAgger 的"distillation 版本"——expert 不提供 hard label，而提供 soft log-prob distribution。两者都解决 distributional mismatch，但 Video-OPD 用 dense soft signal 替代 sparse hard label，更 sample efficient。

参考文献 DAgger: Ross, Stefano, et al. "A reduction of imitation learning to structured prediction via agnostic no-regret learning." ICML 2011. https://arxiv.org/abs/1011.0684

---

总结：Video-OPD 是 on-policy distillation 在 TVG 上的成功应用，核心 insight 是把 GRPO 的 trajectory-level sparse reward 替换为 teacher-provided token-level dense supervision，同时保留 on-policy sampling 和 single rollout，三者协同让训练又快又好。TVDF 进一步用 ground-truth 做 teacher validation + disagreement-based prioritization，提升 sample efficiency。最终 student 在多轮训练后超过 teacher，证明 on-policy distillation 不被 teacher capacity 上界约束——这是与传统 distillation 最大的不同。
