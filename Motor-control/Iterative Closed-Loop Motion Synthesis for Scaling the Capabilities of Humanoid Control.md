---
source_pdf: Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of
  Humanoid Control.pdf
paper_sha256: 20e7bd02340c2bcf86e52f5fba89c6a4d718f02b143d8b8f4c1f9f5fa59e787d
processed_at: '2026-08-05T10:38:11-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 咱们用大白话把这个 paper 捋一遍。这篇 paper 的核心直觉其实非常优雅：**把 humanoid tracker 和 dataset 当成两个互相博弈的 player**。Dataset 不断给 tracker 出难题，tracker 努力解题，解开了 dataset 就出更难的题。这就叫 co-evolution (协同进化)。

### 1. 核心痛点：为什么现有 humanoid control 走到了瓶颈？

现在的 physics-based humanoid control (比如 PHC, MaskedMimic) 极度依赖 MoCap 数据集。最大的数据集是 AMASS，但 AMASS 里面 90% 以上都是走路、站立、日常伸手这种 low dynamic 动作。

用这种低动态数据训练出来的 tracker，遇到 gymnastics 的空翻、martial arts 的回旋踢这种 high dynamic 动作，直接就摔倒了 (collapse)。根本原因在于：**训练数据的 difficulty distribution 有上限**。你拿一堆走路数据，永远练不出后空翻的 policy。而专业的 high dynamic 动作，请演员去 capture 成本极高，且极其危险，根本无法 scale。

### 2. CLAIMS 的直觉：LLM 当教练，MDM 当动作生成器，Tracker 当运动员

CLAIMS (Closed-Loop Automated Motion Synthesis) 构建了一个自动化的闭环。系统里有几个角色：
- **Variable Library (词库)**：一个结构化的专业词汇库。
- **Generator (生成器)**：预训练好的 MDM (Motion Diffusion Model)。
- **Tracker (运动员)**：基于 RL 的物理控制器 (比如 PHC)。
- **VLM (裁判)**：视觉语言模型，看动作做得怎么样。
- **LLM Scheduler (教练)**：Gemini，负责综合反馈并出下一批难题。

每一轮 (loop) 的流程：
1. LLM 教练从词库里挑词，拼出一个 prompt (比如 "The martial artist executed kick -> spin -> land with precise foot placement, following explosive bursts")。
2. MDM 根据 prompt 生成 SMPL 格式的 motion。
3. 过滤掉物理上不合理的动作 (比如脚穿地、全身飞天)。
4. Tracker (运动员) 用 RL 拼命去模仿这个动作。
5. 模仿完之后，系统收集两类反馈：**Physics metrics** (物理指标，如关节误差) 和 **VLM 评分** (裁判觉得这个动作难度多少、连贯性如何)。
6. LLM 教练拿到这些反馈，进行 Chain-of-Thought 推理，决定下一轮怎么出更难的题。

就这样 Loop 0 -> Loop 1 -> ... -> Loop 6，数据越来越难，tracker 越来越强。

### 3. 技术细节拆解

#### 3.1 结构化词库

为了让 LLM 出题不脱离实际，系统设计了一个四维的 difficulty space。每个 domain (martial arts, dance, combat, sports, gymnastics) 都有四个 slots：
- **base action**：原子动作，比如 kick, leap, roll。
- **combo action**：组合链，比如 "roll -> rise -> leap"。
- **detail**：技术细节，比如 "precise foot placement"。
- **speed & rhythm**：节奏，比如 "explosive bursts"。

LLM 必须从这些 list 里挑词填入 template。这个 structured prior 是关键，它防止 LLM 瞎编乱造导致 MDM 生成出垃圾 motion。

#### 3.2 VLM 的低成本评估技巧

怎么让 VLM 评估一个视频动作？如果直接喂 60 帧视频给 GPT-4o 或 Qwen-VL-MAX，计算成本太高。这里有个非常聪明的工程实现：**把 60 帧视频均匀采样后，水平拼接成一张 Base64 长图**。
这样把动态视频变成了静态但有时序顺序的 trajectory，VLM 只要看一眼长图就能判断 action sequence 和 difficulty score。这个设计大大降低了闭环迭代的成本。

#### 3.3 Metrics 公式与变量解释

为了 build intuition，我们看看系统是怎么量化 tracker 表现的。假设一个 motion clip 有 $T$ 帧，每帧有 $J$ 个关节。

**Global MPJPE (g-MPJPE)**：衡量世界坐标系下的绝对追踪误差。
$$ \mathrm{g-MPJPE} = \frac{1}{TJ} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| \mathbf{p}_t^j - \hat{\mathbf{p}}_t^j \right\|_2 $$
- $T$: 总帧数。
- $J$: 总关节数。
- $\mathbf{p}_t^j \in \mathbb{R}^3$: 第 $t$ 帧第 $j$ 个关节的真实世界坐标。
- $\hat{\mathbf{p}}_t^j$: Tracker 预测出来的坐标。
- $\left\| \cdot \right\|_2$: L2 范数 (欧氏距离)。
- 这个公式算的是所有关节在所有帧的平均距离误差。

**Success Rate**：判断动作有没有跟丢。
$$ e_t = \frac{1}{J} \sum_{j=1}^{J} \left\| \mathbf{p}_t^j - \hat{\mathbf{p}}_t^j \right\|_2 $$
$$ s_{\mathrm{clip}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[e_t < \tau] $$
- $e_t$: 单帧的平均误差。
- $\mathbf{1}[\cdot]$: Indicator function，条件成立返回 1，否则返回 0。
- $\tau$: 阈值，这里固定用 0.5m。
- $s_{\mathrm{clip}}$: 整个 clip 中，误差小于 0.5m 的帧所占的比例。

### 4. 实验数据表解析：为什么这个框架有效？

看 Table 1 的核心结果。测试集总共 2201 个 clips，包含 Kungfu, EMDB, AIST++, Video-Convert 四个高难度测试集。

| Method | Kungfu | EMDB | AIST++ | VC | Avg |
|---|---|---|---|---|---|
| AMASS baseline | 47.1 | 53.3 | 67.6 | 31.2 | 58.3 |
| L0 | 37.8 | 31.1 | 68.8 | 33.3 | 55.9 |
| L3 | 59.1 | 64.4 | 82.1 | 50.9 | 72.4 |
| L6 | 60.3 | 64.4 | 88.1 | 58.9 | **76.9** |

直觉解读：
1. **Loop 0 (L0) 居然比 AMASS baseline 还差**。因为一开始 MDM 生成的动作有点 OOD，tracker 还没适应，反而连基础动作都做不好了。
2. **Loop 1 开始反超 baseline**。闭环反馈起作用了，tracker 开始掌握高动态动作的 pattern。
3. **Loop 6 达到 76.9%**。相对于 AMASS baseline 的 58.3%，平均 failure rate (失败率) 从 41.7% 降到了 23.1%，**相对降低了 45%**。
4. 最惊人的是，这个结果**只用了约 400 条数据**，差不多是 AMASS 数据集的 1/10。

### 5. 证明难度确实在升高的证据

为了证明系统确实在生成越来越难的动作，作者用了一个 frozen 的 PHC+ tracker 去跑各个 loop 生成的数据 (Table 5)。

| Data Loop | SR | g-MPJPE | Vel |
|---|---|---|---|
| L0 | 75.3 | 49.78 | 8.54 |
| L3 | 61.2 | 57.24 | 9.95 |
| L6 | 53.6 | 59.61 | 10.97 |

因为 tracker 是固定的，所以 success rate (SR) 单调下降，就证明了数据集的物理难度在单调上升。Velocity (速度) 和 Acceleration 等物理指标也在升高，证实了动作的 dynamic 越来越强。

### 6. 关键 Insight 与直觉延展

我非常欣赏这篇 paper 的几个点：

**1. Prompt Steering 解锁了 OOD Generation**
t-SNE 图显示，用 expert prompts 生成的动作，落在 AMASS manifold 之外，但又在专业 martial arts manifold 附近。这意味着不需要重新训练 MDM，仅靠 prompt engineering 就能引导 latent space 产生 OOD 但物理合理的新动作。MDM 本身的 generalization 能力被低估了。

**2. Multi-modal Feedback 互相补充**
消融实验表明，如果只用 physics metrics，LLM 会缺少对 "动作连贯性、美感、主观难度" 的感知；如果只用 VLM，LLM 又会偏离物理规律。Physics 提供客观下限，VLM 提供主观上限。两者结合形成 robust 的 reward signal。

**3. 把 LLM 当作 Policy 而非 Generator**
LLM 在这里生成的是 prompts，本质上是在一个结构化的 prompt space 里做 policy improvement。这跟 RLHF 中 reward model 与 policy 共同进化的思想如出一辙，只是把 data generation 也拉进来了。

**4. 可能的局限与未来方向**
MDM 的 capacity 终究是有限的。当遇到 extreme high-dynamics (比如连续后空翻三周) 时，MDM 生成的 motion 本身可能就会物理崩溃。未来如果能引入 physics-informed diffusion，或者让 generator 本身也加入这个 closed loop 接收 physics feedback，上限会更高。另外，如果把 LLM scheduler 换成真正的 RL policy，用 tracker improvement 作为 explicit reward，可能会比现在的 prompt-based CoT 更高效。

总结一句话，这篇 paper 把 humanoid control 的 scaling 问题，转化成了 LLM-guided curriculum learning 问题。通过 co-evolution 破解了 data scarcity 和 difficulty ceiling 两大难题。这个 paradigm 在 embodied AI 领域有极大的推广潜力。

参考链接：
- AMASS: https://amass.is.tue.mpg.de/
- PHC: https://zhengyiluo.github.io/phc/
- MDM: https://guytevet.github.io/mdm-page/
- MaskedMimic: https://news.nvidia.com/news/maskedmimic-research

---

# CLAIMS: Iterative Closed-Loop Motion Synthesis for Humanoid Control 深度解析

Andrej, 这篇 paper 的核心 insight 很有意思：**把 humanoid tracker 和 dataset 当成两个对抗的 player，通过 LLM-mediated 的闭环 curriculum 让它们 co-evolve**。本质上是把 self-play / RLHF 中"reward model 与 policy 共同进化"的范式搬到了 motion data 生成上，但 reward model 换成了 multi-modal feedback (physics metrics + VLM scores)，policy 既是 tracker 又是 prompt scheduler。

---

## 1. Problem Framing

现有 physics-based humanoid control (DeepMimic → AMP → ASE → PHC → MaskedMimic) 的 bottleneck 在于：

- **MoCap 数据严重长尾偏斜**：AMASS 90%+ 是日常低动态动作，武术、体操、街舞等专业数据稀缺
- **Fixed difficulty distribution 限制了 policy ceiling**：用低动态数据训练的 controller 在 acrobatic motion 上直接 collapse
- **传统 MoCap 成本不可扩展**：martial arts、gymnastics 这类高危高动态动作很难 capture

CLAIMS 给出的方案是：用 pretrained MDM (在 HumanML3D 上训练) 作为廉价 generator，通过 domain-structured prompt template 把它的 latent space 推向 OOD 的 professional region，然后用 VLM + physics metrics 做双反馈闭环迭代。

参考链接：
- MDM: https://guytevet.github.io/mdm-page/
- PHC: https://zhengyiluo.github.io/phc/
- AMASS: https://amass.is.tue.mpg.de/
- HumanML3D: https://github.com/EricGuo5583/HumanML3D

---

## 2. Architecture 拆解

整个 pipeline 可以拆成 4 个 component：

### 2.1 Difficulty-aware Variable Library (Semantic Prior)

这是整个系统的"语义坐标系"。五大学科 domain × 四维 difficulty axes:

**Domains**: martial arts / dance / combat / sports / gymnastics

**Difficulty axes** (4 orthogonal dimensions):
- **base action**: 原子技能
- **combo action**: 组合链路，例如 "kick → spin → land"
- **detail**: 技术细节
- **speed & rhythm**: 时序节奏

每个 domain 维护 4 个 list，组合空间是 $O(|B_d| \times |C_d| \times |D_d| \times |S_d|)$，规模在 $10^1 \times 10^1 \times 10^1 \times 10^1 = 10^4$ 量级 per domain。这个结构化的 prior 是 LLM scheduler 能够"可控地"提升难度的关键——free-form text 很容易 drift off manifold。

### 2.2 Motion Synthesis (MDM-step50-DistilBERT)

$$
q = G(a; \theta_G)
$$

其中 $a$ 是 prompt，$G$ 是 MDM (50 步 diffusion sampler)，$q$ 是生成的 SMPL motion (22 joints → post-modified 24 joints, 180 frames per clip)。

关键设计：
- **不 fine-tune MDM**，只靠 prompt engineering 把它的 latent 推向 OOD
- Offset height = 0.92m (humanoid root 标准化)
- Post-generation 用 Gaussian filter 平滑 root displacement

物理过滤：
- root height 越界 → 拒绝
- foot penetration below terrain → 拒绝

### 2.3 VLM Semantic Alignment Check

这里有个很巧的工程实现：**把 60-frame 视频均匀 subsample 后 horizontally stitch 成一张 Base64 image**，喂给 Qwen-VL-MAX。这样把 dynamic motion 压成 temporally-ordered static trajectory，既减少了 VLM 推理成本，又保留了完整 action semantics。

VLM 输出：
- action-matching score (语义对齐)
- difficulty score ∈ [0, 10]
- textual descriptors: action sequence / technical complexity / intensity / balance / continuity

接受阈值：`semantic_score ≥ τ_sem`

### 2.4 Tracker Training (PHC single-primitive)

$$
\pi^{trk}_{k+1} = \text{TrainTracker}(\mathcal{D})
$$

PHC 用 PPO，reward 是 dense reward：pose + joint velocities + end effectors + contact events。HP 超参完全沿用 PHC 原文（Table 9）：

| Param | Value |
|---|---|
| Total Steps | $1.5 \times 10^6$ |
| Optimizer | Adam |
| Num Envs | 1024 |
| LR | $1 \times 10^{-4}$ |
| $\gamma$ (discount) | 0.98 |
| $\lambda$ (GAE) | 0.95 |
| $\epsilon$ (clip) | 0.2 |
| Replay Buffer | $2 \times 10^5$ |

---

## 3. Closed-Loop Co-Evolution 数学化

Algorithm 1 是核心。形式化：

$$
o_k = [m_k, v_k, e_k]
$$

- $m_k$: physics tracking metrics (g-MPJPE, l-MPJPE, VelDist, AccDist, Success Rate)
- $v_k$: dual VLM (GPT-4o + Qwen-VL-MAX) difficulty scores + textual analyses
- $e_k = \phi(a_k)$: previous action prompt 的 embedding

LLM policy (Gemini-CoT) 接收 $o_k$ 和 variable library $\mathcal{L}$ + templates $\mathcal{T}$，sample 出下一批 prompts：

$$
A_k = \{a_k^1, \ldots, a_k^M\} \sim \pi_\theta(o_k, \mathcal{L}, \mathcal{T})
$$

对每个 $a_k^j$ 生成 motion $q_k^j$，过滤后加入 $\mathcal{D}$，重新训练 tracker。

**关键 insight**：这里没有显式的 reward function 去优化，而是 implicit curriculum——policy 的"目标"是 improve physical tracking scores **while** steadily raising annotated difficulty。这是 self-reinforcing 的 self-play。

### Metrics 公式详解

设 motion clip 有 $T$ 帧、$J$ 个 joint，root joint index $r$：

**Global MPJPE** (世界坐标系下的平均关节位置误差)：
$$
\text{g-MPJPE} = \frac{1}{TJ} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| \mathbf{p}_t^j - \hat{\mathbf{p}}_t^j \right\|_2
$$
- $\mathbf{p}_t^j \in \mathbb{R}^3$: ground-truth global position of joint $j$ at frame $t$
- $\hat{\mathbf{p}}_t^j$: controller 预测的 global position
- 反映绝对 tracking 精度，对 global drift 敏感

**Local MPJPE** (root-relative，消除 global translation)：
$$
\text{l-MPJPE} = \frac{1}{TJ} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| (\mathbf{p}_t^j - \mathbf{p}_t^r) - (\hat{\mathbf{p}}_t^j - \hat{\mathbf{p}}_t^r) \right\|_2
$$
- $\mathbf{p}_t^r$: root joint 位置
- 主要反映 articulation quality，不受 global drift 干扰

**Velocity Distance** (关节速度差)：
$$
\text{VelDist} = \frac{1}{(T-1)J} \sum_{t=1}^{T-1} \sum_{j=1}^{J} \left\| \mathbf{v}_t^j - \hat{\mathbf{v}}_t^j \right\|_2
$$
- $\mathbf{v}_t^j = \mathbf{p}_{t+1}^j - \mathbf{p}_t^j$: finite difference velocity
- 捕捉 dynamics (speed + direction changes)

**Acceleration Distance** (关节加速度差)：
$$
\text{AccDist} = \frac{1}{(T-2)J} \sum_{t=1}^{T-2} \sum_{j=1}^{J} \left\| \mathbf{a}_t^j - \hat{\mathbf{a}}_t^j \right\|_2
$$
- $\mathbf{a}_t^j = \mathbf{v}_{t+1}^j - \mathbf{v}_t^j$
- 对 abrupt changes 高度敏感，与 motion smoothness / physical plausibility 强相关

**Success Rate** (clip 级别)：
$$
e_t = \frac{1}{J} \sum_{j=1}^{J} \left\| \mathbf{p}_t^j - \hat{\mathbf{p}}_t^j \right\|_2, \quad s_{\text{clip}} = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[e_t < \tau]
$$
- $\tau = 0.5$m (固定阈值)
- $\mathbf{1}[\cdot]$: indicator function

参考 PHC 的 evaluation protocol: https://zhengyiluo.github.io/phc/

---

## 4. Competitive Iteration 的直觉

这是 paper 最有意思的部分。把闭环迭代抽象成 minimax game 的变种：

- **Tracker** (min player): 想要 minimize tracking error on current distribution
- **Dataset** (max player via LLM scheduler): 想要 maximize difficulty while staying semantically valid

跟 GAN 的区别：这里 generator (LLM) 不直接 sample 数据，而是 sample **prompts**，由 MDM 间接 sample。这相当于在 prompt space 而非 data space 上做 adversarial curriculum。

跟 RLHF 中 reward model 的区别：这里没有显式 reward 函数，"reward" 是 multi-modal feedback 的 implicit combination。LLM scheduler 在做的是 **policy improvement over prompt space**，不是 value-based optimization。

Loop-wise progression 的机制：
- Loop $k$ tracker 在 loop $k$ 数据上训练收敛
- Compute $m_k$ + $v_k$ → 形成观察 $o_k$
- LLM 用 CoT 推理：哪些 set 是"low error + low VLM difficulty"（容易，要 hardening），哪些是"high error + high VLM difficulty"（已 hard，小步迭代）
- 生成 loop $k+1$ 的 prompts，sample harder motions
- Tracker 在 $\mathcal{D} \cup M_k$ 上继续训练

---

## 5. 关键实验数据

### 5.1 主实验 (Table 1): PHC single-primitive

测试集 2201 clips：Kungfu (663) + EMDB (45) + AIST++ (1320) + Video-Convert (173)

| Method | Kungfu | EMDB | AIST++ | VC | Avg |
|---|---|---|---|---|---|
| AMASS baseline | 47.1 | 53.3 | 67.6 | 31.2 | 58.3 |
| L0 | 37.8 | 31.1 | 68.8 | 33.3 | 55.9 |
| L1 | 47.7 | 33.3 | 75.3 | 38.7 | 64.0 |
| L2 | 51.7 | 51.1 | 73.3 | 41.6 | 63.8 |
| L3 | 59.1 | 64.4 | 82.1 | 50.9 | 72.4 |
| L4 | 55.8 | 55.6 | 84.0 | 45.1 | 71.8 |
| L5 | 60.6 | 60.0 | 85.2 | 54.3 | 74.8 |
| L6 | **60.3** | **64.4** | **88.1** | **58.9** | **76.9** |

关键观察：
1. **L0 < AMASS baseline**：因为 loop0 用 MDM 生成的数据偏 OOD，初期 tracker 还没适应
2. **L1 > AMASS baseline**：仅一轮迭代就 surpass baseline，说明闭环反馈比单纯数据规模重要
3. **L6 vs AMASS**: average failure rate 从 41.7% 降到 23.1%，**相对降幅 45%**
4. **数据效率**：仅用 ~400 sequences (AMASS 的 1/10)

### 5.2 Cross-tracker 泛化 (Table 2): MaskedMimic

| Dataset | AMASS | loop0 | loop1 |
|---|---|---|---|
| Kungfu | 57.2 | 54.0 | **65.8** |
| EMDB | 53.3 | 48.9 | **71.1** |
| AIST++ | 68.9 | 75.3 | **83.9** |
| VC | 41.6 | 47.4 | **62.4** |

证明 framework 是 tracker-agnostic 的，DeepMimic-style (MaskedMimic) 和 AMP-style (PHC) 都能用。

参考: MaskedMimic https://news.nvidia.com/news/maskedmimic-research

### 5.3 Ablation Studies

**Observation ablation** (Table 3, Loop3):

| Variant | 平均趋势 |
|---|---|
| Full (var + VLM + physics) | 最好 |
| w/o var library | 略差，提示 Gemini-CoT 自由生成不够稳定 |
| w/o VLM | 物理指标是核心 |
| w/o physics metrics | 显著下降 |
| w/o both | 最差 |

排序：**no obs < no physics < no VLM < full** → physics metrics 是诊断 tracker 弱点的最 diagnostic signal，VLM 提供 complementary subjective difficulty cue。

**Variable library ablation**: 没有结构化 variable library，让 Gemini 直接生成 hard prompts，结果在所有 third-party benchmarks 上都更差。说明 structured prior 稳定 prompt 生成、防止 LLM hallucinate 不切实际的描述。

**Feedback iteration vs data scale**: 同等数据规模下，iterative 训练显著优于 one-shot 训练，证明 gain 来自 curriculum 而非单纯数据增多。

### 5.4 Difficulty Verification (Table 5)

用 frozen PHC+ 推理各 loop 数据：

| Loop | SR | g-MPJPE | Acc | Vel |
|---|---|---|---|---|
| L0 | 75.3 | 49.78 | 5.97 | 8.54 |
| L1 | 65.8 | 53.84 | 6.66 | 9.34 |
| L2 | 65.2 | 61.29 | 7.65 | 10.86 |
| L3 | 61.2 | 57.24 | 7.03 | 9.95 |
| L4 | 59.0 | 57.70 | 7.08 | 9.99 |
| L5 | 52.7 | 59.10 | 7.49 | 10.65 |
| L6 | 53.6 | 59.61 | 7.94 | 10.97 |

**SR 单调下降**（75.3 → 53.6）证明 dataset 难度确实随 loop 增加。velocity、acceleration 都有相应上升趋势，说明高 dynamic motion 在数据中比例增加。

### 5.5 VLM Scoring Reliability

盲测 5 tiers × 200 motions = 1000 clips，Qwen 在不读 prompt 的情况下仅看 rendered frames 给 difficulty 1-10 分。结果显示：
- 平均 velocity 随 tier 单调上升
- Qwen 的 difficulty rating 也随 tier 单调上升

证明 VLM 的 difficulty judgment 与物理指标高度 correlated，可以作为有效的 feedback signal。

### 5.6 t-SNE Distribution Analysis (Fig. 7)

- Fig. 7a: expert-prompt 生成 motion 大量 overlap 专业 martial arts 数据 manifold，random prompt 远离
- Fig. 7b: expert-prompt 生成 motion 在 AMASS/HumanAct12 manifold **之外**，random prompt 在 manifold 之内

这是非常关键的发现：**MDM 在 expert prompt 引导下能产生 OOD 但仍 professional-plausible 的 motion**。这说明 MDM 的 latent space 比训练分布更宽广，prompt 是解锁 OOD generation 的钥匙。

---

## 6. 与 Related Work 的位置

CLAIMS 在几个 axis 上定位独特：

| 维度 | 现有工作 | CLAIMS |
|---|---|---|
| Data 来源 | MoCap / video retargeting | LLM-guided MDM synthesis |
| Difficulty 控制 | 静态分布 | 4-axis structured + iterative |
| Curriculum | None / hand-crafted | LLM-mediated closed-loop |
| Evaluation | Single metric | Multi-modal (physics + VLM) |
| Controller 兼容 | Pipeline-specific | Tracker-agnostic |

最相近的工作是 **PARC** (Xu et al., SIGGRAPH 2025)，也是 generate-correct-augment-retrain loop，但 PARC 用单一 evaluation criterion，CLAIMS 用 multi-dimensional (physics + dual VLM)。另外 PARC 不能跨场景泛化。

另一个相关的是 **CLOSD** (Tevet et al., 2024)，把 diffusion 和 RL 闭环做 multi-task control，但还是用 static corpus，没有 difficulty adaptation。

参考：
- PARC: https://dair-aer.github.io/character-physics/
- CLOSD: https://clost-loop.github.io/
- ASE: https://xuebinp.github.io/papers/ase/index.html
- AMP: https://pkuxuxd.github.io/AMP/

---

## 7. Limitations 和未来方向

Paper 自己诚实承认两个限制：

1. **Generator capacity bound**: MDM 在 extreme high-dynamics (e.g., 体操空翻、MMA 投摔) 上还是会 struggle。但 modular design 允许 swap in 未来的更 advanced generator (e.g., MotionGPT-2, Being-M0.5)。

2. **Variable library 是 manually curated**: 缺乏 objective calibration 和 comprehensive coverage。未来需要 automated multi-modal library + domain knowledge graph。

从 research 视角，我认为还有几个潜在方向：

- **Reward hacking risk**: VLM 可能学到 spurious difficulty cues（如画面亮度、动作幅度 proxy），导致 LLM scheduler 找到 reward hacking prompts。需要 active learning 式的 VLM calibration。
- **Sample efficiency**: 当前每 loop 用 $1.5 \times 10^6$ steps 训练 PHC，如果 loop 数量从 6 扩到 60，compute 会爆炸。可以引入 off-policy replay 或 incremental fine-tuning。
- **Real robot transfer**: Paper 全在仿真里，sim-to-real gap 在 humanoid 上是另一大坑。可以借鉴 Humanoid-X / HuBE 的 cross-embodiment 方向。
- **Multi-agent collision**: 当前 framework 假设 single-primitive tracker，多 agent 交互 (combat, partnering dance) 是下一步。

参考: 
- Humanoid-X: https://ir1dX.github.io/Humanoid-X/
- HuBE: https://github.com/shipenglyu/HuBE
- Being-M0.5: https://arxiv.org/abs/2502.05177

---

## 8. 我对这篇 paper 的整体直觉

**最 strong 的点**：
1. Co-evolution framing 非常 elegant——把 dataset 和 controller 当成 minimax player 是个漂亮的 abstraction
2. Multi-modal feedback (physics + VLM) 互相 complement：physics 提供 objective 信号，VLM 提供 subjective/semantic 信号
3. Variable library 是关键的 structured prior，避免 LLM scheduler hallucinate 不切实际的 prompt
4. 框架 tracker-agnostic，PHC 和 MaskedMimic 都能用，证明方法本身的通用性

**最 weak 的点**：
1. MDM 的 OOD generation 能力其实是 prompt steering 的 free lunch，paper 没充分说明为什么 expert prompt 能 push 出 manifold——可能是 DistilBERT 在 HumanML3D 之外有 generalization 能力？这个 mechanism 需要 more analysis
2. VLM 的 evaluation 是 black box，没有给出 calibration study 之外的 systematic analysis
3. 只跑了 6 个 loop，scale up 到 100+ loop 的稳定性未验证
4. Failure case (Fig. 14) 没有更深入分析——是 physics violation 还是 reward shaping 不够？

总体而言，这篇 paper 提供了一个非常 promising 的 paradigm：**LLM as a curriculum orchestrator for embodied AI training**。这与 RLHF、self-improving LLM (SEAL)、robotics 自动 data generation (RoboTwin 2.0) 是同一大趋势。humanoid control 从 "static dataset + static policy" 走向 "dynamic dataset co-evolving with policy"，这是 scaling 的正确方向。

如果你想进一步 push 这个方向，我觉得最值得探索的是：把 LLM scheduler 换成真正的 RL policy (而非 prompt-based CoT)，给它 explicit reward = tracker improvement + difficulty growth，这样能在 prompt space 上做真正的 optimization，而不是 rely on LLM 的 in-context reasoning。

希望这个分析对你 build intuition 有帮助！
