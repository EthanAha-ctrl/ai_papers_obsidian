---
source_pdf: SFT Memorizes RL Generalizes AComparative Study of Foundation Model Post-training.pdf
paper_sha256: ebe6aaf90a9e309732a45c1f809a40808b61ec46f737b0306dba80f21a149c54
processed_at: '2026-08-12T05:25:24-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话版本

SFT 在 **死记硬背** training data，RL 在 **学真正的 rule**。你换一下 rule、换一下 visual style，SFT 直接崩，RL 还能 work。

---

## 到底在研究什么？

现在大家训 LLM/VLM 都是这个 pipeline：pre-train → SFT → RL。但这两步 post-training 到底各干了什么，其实没人讲清楚过。这篇 paper 就是在问一个特别 raw 的问题：

**SFT 和 RL，谁在 memorize，谁在 generalize？**

这个问题为什么重要？因为你如果花几百万美元训一个 model，结果它只是把 training data 记住了，换个问法就答不出，那这 model 就是个昂贵的数据库，不是 intelligence。

---

## 怎么测 memorize vs generalize？

关键是要把 "model 学到了 rule" 和 "model 记住了 specific pattern" 这两件事分开。作者的方法很聪明——**同一个 task，换一个 rule 或换一个 visual style，看 model 还行不行**。

举个例子，GeneralPoints 这个 task：给 model 4 张扑克牌，让它凑出 24。

训练的时候，规则是 `J=Q=K=10`（都当 10 用）。

测试的时候，换成 `J=11, Q=12, K=13`。

如果 model 学到的是 "哦，这个游戏是 parse prompt 里的 rule，然后做 arithmetic"，换 rule 也能 work。

如果 model 记住的只是 "看到 J 就输出 10"，换 rule 就崩。

这个 design 直接把 memorization 和 generalization 给 disentangle 了。很 elegant。

---

## 两个 task，两个 modality

### GeneralPoints
4 张牌凑 24。两个版本：
- **GP-L**：纯文字输入，`Cards: ['A', '3', 'K', '6']`
- **GP-VL**：图片输入，model 得先看图认牌再算

OOD 测试：
- Rule variant：`J=Q=K=10` → `J=11, Q=12, K=13`
- Visual variant：黑桃梅花 → 红桃方块

### V-IRL
真实街景导航。model 看着 street view，按指令走路。
- **V-IRL-L**：纯文字描述 landmark
- **V-IRL-VL**：看 360° 全景图

OOD 测试：
- Rule variant：absolute direction（`north, northeast`）→ relative direction（`left, slightly right`）
- Visual variant：NYC → 全球 9 个城市

两个 task 横跨 arithmetic 和 spatial reasoning，横跨 LLM 和 VLM。如果只在 GP 上看到结论，你可以说这是 task-specific artifact；在 V-IRL 上也看到同样 pattern，说服力就强多了。

---

## 实验结果——直接看数字

### Rule-based generalization

| Task | SFT ΔOOD | RL ΔOOD |
|---|---|---|
| GP-L | **-8.1%**（11.5→3.4） | **+3.5%**（11.5→15.0） |
| GP-VL | **-5.6%**（11.2→5.6） | **+3.0%**（11.2→14.2） |
| V-IRL-L | **-79.5%**（80.8→1.3） | **+11.0%**（80.8→91.8） |
| V-IRL-VL | **-33.2%**（35.7→2.5） | **+9.3%**（35.7→45.0） |

SFT 在 4 个 setting 全崩。RL 在 4 个 setting 全涨。

V-IRL-L 那个 -79.5% 特别触目惊心：SFT 把 model 训成了 absolute direction 的复读机，碰到 relative direction 直接从 80.8% 掉到 1.3%。完全没用。

### Visual generalization

| Task | SFT ΔOOD | RL ΔOOD |
|---|---|---|
| GP-VL（黑→红） | -9.9%（23.6→13.7） | **+17.6%**（23.6→41.2） |
| V-IRL-VL（NYC→全球） | -5.6%（16.7→11.1） | **+61.1%**（16.7→77.8） |

V-IRL-VL 上 RL 直接刷到 77.8%，比原 paper 的 SOTA（44.0%，用 GPT-4 + 两阶段 pipeline + prompt engineering）高了 33.8 个点。用 open-source Llama-3.2-Vision-11B end-to-end 训出来的。

---

## 最有意思的发现：RL 顺便把 visual recognition 也训好了

这个 ablation 我觉得是整篇 paper 最有 insight 的部分。在 GP-VL 上：

- **Scaling RL compute** → visual recognition accuracy **提升**，success rate **提升**
- **Scaling SFT compute** → visual recognition accuracy **下降**，success rate **下降**

SFT 越训，model 越不会看图。这听起来反直觉，但仔细想是有道理的。

SFT 的 loss 是 token-level cross-entropy。看 Figure 11 的 example，`formula` 字段（reasoning tokens）的长度远大于 `cards` 字段（recognition tokens）。所以 gradient signal 里 reasoning token 占主导，model 把所有 capacity 都拿去拟合 reasoning pattern，visual recognition 这个 "低频任务" 就被挤掉了。这就是经典的 catastrophic forgetting（[Zhai et al. 2024b](https://arxiv.org/abs/2310.14004) 报告过类似现象）。

RL 的 reward 只看 final outcome——等式对不对。Model 想拿 reward，就得先认对牌。所以 visual recognition 被迫一起 train。**Outcome-based reward 天然平衡了 recognition 和 reasoning 两个 subtask**。

这个 insight 其实很 deep：**你不需要单独训 visual recognition，只要 reward 结构对了，model 自己会 figure out 哪些能力是 prerequisite**。

---

## 但 SFT 不是没用——它是 format teacher

作者试了一下跳过 SFT 直接对 base Llama-3.2-Vision-11B 跑 RL——**完全失败**。

Failure mode（Figure 20）：base model 输出一堆乱七八糟的东西，甚至写 Python 代码暴力枚举，但就是不输出 structured JSON。没有 JSON 就 parse 不出 action，没有 action 就算不出 reward，没有 reward RL 就没法 train。

所以 SFT 的 role 其实很 narrow：**教 model 输出格式**。一旦 format 稳定了，RL 接手去学真正的 capability。

这和 [LIMA (Zhou et al. 2024)](https://arxiv.org/abs/2305.11206) 的说法一致：SFT 是 "format teacher"，1k 条 data 就够了，capability 是 pre-train 带来的。

但这和 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 的 "SFT 不必要" 结论不矛盾。R1 的 base 是 DeepSeek-V3，已经 instruction-tuned 过了；Llama-3.2-Vision 的 instruction following 不够好。**关键不是有没有 SFT 这一步，而是 base model 会不会 follow instructions**。

---

## Verification steps：越多越好

在 GP-L 上固定 compute budget，变 max verification steps：

| VIter | OOD improvement |
|---|---|
| 1 | +0.48% |
| 3 | +2.15% |
| 5 | +2.99% |
| 10 | +5.99% |

从 1 步到 10 步，generalization 从 +0.48% 涨到 +5.99%。

这就是 o1 和 R1 的思路——**给 model 机会 self-correct**。第一轮答错，verifier 告诉你错哪了，第二轮你换个思路。每次 verification 都是一次 escape memorized pattern 的机会。

这个实验直接把 "test-time compute → generalization" 这个 chain 给 quantified 了。

---

## 我的理解：为什么 RL generalize，SFT memorize？

这部分 paper 没给正式理论，是我自己的推想，但我觉得 intuition 是对的。

### SFT 的 loss

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log p_\theta(y|x) \right]$$

这个 loss 鼓励 $p_\theta(y|x)$ 去逼近 $p_{\text{data}}(y|x)$。在 finite data 下，最简单的 path 就是 memorize $(x, y)$ pair。任何 "学 underlying rule" 的 hypothesis space 都比 "memorize" 的 hypothesis space 大，optimization 上 harder，所以 finite training steps 下 memorize 通常 wins。

### RL 的 loss

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t r_t \right]$$

这个 objective **完全不约束 $p_\theta$ 的 shape**，只约束 expected return。达到 high return 的 path 有无数条，"memorize training data" 只是其中一条，而且可能还不是最优的（因为 training data 只覆盖了 state space 的一小部分）。

更关键的是：**RL 的 reward 是 outcome-based，只看结果不看过程**。同一个 outcome 可以由很多 trajectory 达成。Model 通过 rollouts 看到多样化的 successful paths，自然学到 "what's invariant across all successful paths"——那就是 underlying rule。

SFT 看到的每个 "正确答案" 都是同一个 specific trajectory，model 学到的是 "这个 specific trajectory 长什么样"，不是 "为什么这个 trajectory 是对的"。

### 类比

这就像学下棋：
- **SFT**：看大师棋谱，每步都告诉你 "这步该走马"。你记住了 "这种局面走马"，换个局面就不会了。
- **RL**：自己下，赢了 +1 输了 -1。你得自己 figure out 为什么某些走法赢。下多了，你学到的是 "控制中心""保护 king" 这些 principle，换局面也能用。

---

## Compute 的细节

作者用 [Chinchilla 的公式](https://arxiv.org/abs/2203.15556) 估算：

$$X_{\text{train}} = 6ND_{\text{train}}, \quad X_{\text{inference}} = 2ND_{\text{inference}}$$

- $N$：参数量
- $D_{\text{train}}$：训练 token 数

SFT total compute：

$$X_{\text{SFT}} = 6N(D_{\text{init}} + D_{\text{SFT}})$$

RL total compute（PPO 是 on-policy，要多一次 inference）：

$$X_{\text{RL}} = 6N(D_{\text{init}} + D_{\text{RL}}) + 2ND_{\text{buffer}}$$

$D_{\text{buffer}}$ 的近似：

$$D_{\text{buffer}} \approx \frac{E \bar{d}_i \bar{d}_o}{D_{\text{RL}}} \cdot D_{\text{RL}} = \lambda D_{\text{RL}}$$

- $E$：auto-regressive generation 次数
- $\bar{d}_i, \bar{d}_o$：平均 input/output token 长度
- $\lambda$：GP 上 ≈ 6，V-IRL 上 ≈ 5.1

所以 RL 每 token 大约比 SFT 贵 6-7x。但 Figure 5 是按 GFLOPs 画的——**等 compute 下 RL 还是 beat SFT**，所以这个 advantage 不是 "RL 用了更多 compute" 的 artifact。

---

## Reward 设计细节

### GeneralPoints

| 情形 | Reward |
|---|---|
| 合法等式 = 24 | $r = +5$ |
| 合法但 ≠ 24 | $r = -1$ |
| 超过 max step (5) | $r = -1$ |
| 用了不在卡上的数字 | $r = -2$ |
| 其他非法 | $r = -3$ |
| GP-VL 识别错牌 | $r = -1.5$（额外） |

### V-IRL

| 情形 | Reward |
|---|---|
| action 正确 | $r = +1$ |
| action 错误 | $r = -1$ |
| 超过 max step (2) | $r = -1$ |
| landmark detection 失败 | $r = -1.5$ |

注意：都是 **outcome-based**，没有 process supervision。这和 [Lightman et al. 2023 的 PRM](https://arxiv.org/abs/2305.20050) 是对照——PRM 监督每一步，这篇只看最终结果。作者的 hypothesis 是：**正是这种 "只看结果" 的 reward 结构，逼着 model 自己 discover rule**。如果你监督每一步，model 又退回 mimic pattern 了。

---

## 两个 failure mode 要注意

### 1. SFT 在 GP-VL 上根本 train 不起来

Figure 16 跑了 10 个 learning rate + frozen component 的 ablation，没有一个能超过 30%。SFT 在 vision-language setting 下根本 compete 不过 RL。

### 2. RL 救不了 overfitted checkpoint

Figure 19：从一个 per-step accuracy < 1% 的极度 overfit SFT checkpoint 开始 RL，RL 也 recover 不回来。Model 已经 commit 到 spurious pattern，gradient signal 不够把它拉回来。

**Practical takeaway**：RL 的 generalization 优势依赖 starting checkpoint 还没 memorize。一旦 SFT 训过头，RL 也救不了。所以 multi-round pipeline 里 SFT 要 "just enough"，不能 overdo。

---

## 这篇 paper 和 frontier 的 connection

### 和 o1 / R1 的关系

o1 和 R1 都用 "thinking before answering" + verification。这篇 paper 的 multi-turn + verifier 就是同一个 idea 的 academic version。Figure 10 直接 quantified 了：verification step 越多，generalization 越好。

R1 的 base 是 DeepSeek-V3，已经 instruction-tuned 过，所以可以跳过 SFT 直接 RL。这篇 paper 的 Llama-3.2-Vision instruction following 不够，必须先 SFT。**两者不矛盾，关键是 base model 的 instruction following 能力**。

### 和 Llama 3 的关系

[Llama 3 paper](https://arxiv.org/abs/2407.21783) 的 post-training 是 "multiple rounds of SFT + RLHF"。这篇 paper 给这个设计提供了理论支撑：SFT 教 format，RL 教 generalization，下一轮 SFT fix regression，下一轮 RL 再 generalize。**这种 "format teacher + generalizer" 的 role split 可能是 multi-round pipeline work 的根本原因**。

### 和 AlphaZero 的 connection

AlphaZero 用 self-play + outcome reward 学到了远超人类 master 的策略。这篇 paper 的 RL 其实是 AlphaZero 在 LLM 上的 analog：**不教 model 怎么做，只告诉它做对没有，model 自己 discover 策略**。SFT 就像 "看大师棋谱"，你只能学到大师的水平，超不过。

---

## 我的整体 take

**这篇 paper 做了一件很简单但很重要的事**：用 clean 的实验设计，把 SFT 和 RL 的 functional role 给 disentangle 了。结论很 sharp：

1. **SFT memorize**——它在 mimic training data 的 surface pattern
2. **RL generalize**——它在 discover underlying rule
3. **SFT 是 format teacher**——教 model 输出结构化格式
4. **RL 是 capability learner**——教 model 真正的 reasoning
5. **Verification steps 是 generalization 的 amplifier**——越多越好

这个结论和 o1、R1、Llama 3 的工程经验完全一致。这篇 paper 给它提供了 clean 的 experimental evidence。

**但 open questions 还很多**：
- >11B model 上会不会一样？没测
- "RL 为什么 generalize" 没有正式理论
- Process supervision (PRM) 会怎样？没对比
- 不同 reward shaping 的 robustness？没系统研究

这些都是 future work 的方向。但就当前结果而言，这篇 paper 已经足够说服我：**RL 是 reasoning capability 的主要来源，SFT 只是 format**。这对怎么设计 post-training pipeline 有直接指导意义。

---

## 相关链接

- [Paper: SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2507.07681)
- [RL4VLM (Zhai et al. 2024a)](https://openreview.net/forum?id=nBjmMF2IZU) — 方法论基础
- [Snell et al. 2024 — Scaling test-time compute](https://arxiv.org/abs/2408.03314) — sequential revision formulation
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — 纯 RL 训 reasoning
- [OpenAI o1 system card](https://arxiv.org/abs/2412.16720) — test-time verification 工业实现
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347) — RL 算法
- [LIMA (Zhou et al. 2024)](https://arxiv.org/abs/2305.11206) — SFT 是 format teacher
- [V-IRL (Yang et al. 2024a)](https://arxiv.org/abs/2410.07163) — 导航环境
- [Physics of LLMs (Allen-Zhu & Li 2023)](https://arxiv.org/abs/2309.14316) — memorization 理论
- [PRM (Lightman et al. 2023)](https://arxiv.org/abs/2305.20050) — process supervision 对照
- [Llama 3 (Dubey et al. 2024)](https://arxiv.org/abs/2407.21783) — backbone + multi-round pipeline
- [Chinchilla (Hoffmann et al. 2023)](https://arxiv.org/abs/2203.15556) — compute 估算
- [Cobbe et al. 2021 — verifiers for math](https://arxiv.org/abs/2110.14168) — outcome-based verifier 源头

---

# SFT Memorizes, RL Generalizes — 深度技术解读

Andrej，这篇 paper 我觉得抓住了当前 post-training 阶段最核心的一个 empirical puzzle：**为什么 SFT 把 model 训得越来越像 training data，而 RL 反而能让 model 学到 "transferable principle"？** 这正好对应你过去几年反复强调的 "model 想学的是 underlying algorithm 而非 surface pattern" 的直觉。下面我把方法、公式、实验和我的 intuition 都展开讲。

---

## 1. Paper 的核心 Question

Foundation model 的 post-training 通常分两步：SFT（supervised fine-tuning）+ RL（reinforcement learning，通常是 PPO 或类似的 on-policy 方法）。但这两步的 **functional role** 一直没被严格 disentangle：

- SFT 是在 mimic 一个 supervised target distribution $p_{\text{SFT}}(y|x)$
- RL 是在 optimize 一个 expected return $\mathbb{E}_\pi[\sum_t r_t]$

二者形式上完全不同，但实践中常常被混为一谈（"post-training = alignment"）。这篇 paper 提出的问题非常 sharp：

> SFT/RL 各自是 **memorize** training distribution，还是 **learn generalizable rule**？

为了分离这两种机制，作者设计了 **rule-based generalization**（同一 task 下换 rule）和 **visual generalization**（同一 task 下换 visual appearance）两种 OOD 评测。这个 design 很聪明：因为如果 model 只 memorize 了 "J=10" 这个 specific mapping，换到 "J=11, Q=12, K=13" 就会崩；如果它真的学到了 "parse rule from prompt + do arithmetic"，就能 generalize。

论文链接：[arXiv (作者主页版本)](https://arxiv.org/abs/2507.07681) / [Yuexiang Zhai's RL4VLM (NeurIPS 2024)](https://openreview.net/forum?id=nBjmMF2IZU)

---

## 2. 两个 Benchmark 的设计

### 2.1 GeneralPoints（GP-L 和 GP-VL）

基于 RL4VLM 的 Points24 改造。State 是 4 张扑克牌，target 是凑出 24。

- **GP-L**：纯文本输入，`Cards: ['A', '3', 'K', '6']`
- **GP-VL**：图像输入，VLM 要先 recognize 4 张牌再做 arithmetic

**Rule variants**（核心 OOD 测试）：
- ID：`J=Q=K=10`
- OOD：`J=11, Q=12, K=13`

**Visual variants**：
- ID：黑色花色 ♠♣
- OOD：红色花色 ♥♦

### 2.2 V-IRL（V-IRL-L 和 V-IRL-VL）

来自 [Yang et al. 2024a](https://arxiv.org/abs/2410.07163)，真实世界街景导航。

- **Rule variants**：absolute direction（`north, northeast, ...`）vs relative direction（`left, right, slightly left, slightly right`）
- **Visual variants**：训练 NYC，测试全球 9 个城市

这两个 task 涵盖了 LLM (text-only) 和 VLM (vision-language)，并且都有 rule 和 visual 两个 axis 的 OOD 测试。这个 design 让结论的 generalizability 大大增强 —— 如果只在 GP 上看到 "RL generalizes"，可能被人质疑是 task-specific artifact；同时在 V-IRL 上 reproduce，就 strong 得多。

---

## 3. 方法：Multi-turn RL with Verifier

### 3.1 RL 的标准 formulation

经典 finite-horizon MDP：

$$\max_{\pi \in \Pi} \mathbb{E}_\pi \left[ \sum_{t=0}^{T} r_t \right]$$

- $\pi : \mathcal{S} \to \mathcal{A}$：policy
- $r_t = r(s_t, a_t)$：step reward
- $T$：max steps per episode
- $\pi(a|s) \in [0,1]$：probability of choosing action $a$ at state $s$

### 3.2 适配到 LLM/VLM

作者把 token space $\mathcal{V}$ 作为基础：

- $\mathcal{V}^m$：input text space（$m$ 是 max input token length）
- $\mathcal{V}^n$：output text space（$n$ 是 max output token length）
- $\mathcal{O}$：所有 RGB image 的集合（仅 VLM）

State space：
- VLM: $\mathcal{S} := \mathcal{V}^m \times \mathcal{O}$
- LLM: $\mathcal{S} := \mathcal{V}^m$

Action space: $\mathcal{A} := \mathcal{V}^n$（即 model 输出的 token 序列）

### 3.3 Verifier 和 Outcome-based Reward

关键创新是引入一个 verifier：

$$\text{VER}(\mathbf{v}_t^{\text{out}}) \mapsto (r_t, \mathbf{v}_t^{\text{ver}})$$

- $\mathbf{v}_t^{\text{out}} \in \mathcal{V}^n$：第 $t$ 步 model 的输出
- $r_t \in \mathbb{R}$：outcome-based reward（参考 [Cobbe et al. 2021](https://arxiv.org/abs/2110.14168) 的 verifier 思路）
- $\mathbf{v}_t^{\text{ver}} \in \mathcal{V}^k$：verifier 给出的 textual feedback（比如 "You failed because formula is incorrect"）

这个 verifier 起到三个作用：
1. 计算 reward
2. 生成 textual feedback，作为下一步的 context
3. 把单次 generation 变成 multi-turn 交互

### 3.4 Sequential Revision Formulation

State transition 的关键设计（来自 [Snell et al. 2024](https://arxiv.org/abs/2408.03314)）：

$$\mathbf{v}_{t+1}^{\text{in}} = \text{concat}\left(\mathbf{v}_0^{\text{in}}, [\mathbf{v}_k^{\text{out}}, \mathbf{v}_k^{\text{ver}}]_{k=0}^{t}\right)$$

- $\mathbf{v}_0^{\text{in}}$：system prompt（任务描述 + rule）
- $[\mathbf{v}_k^{\text{out}}, \mathbf{v}_k^{\text{ver}}]_{k=0}^{t}$：所有历史 model output + verifier feedback

Intuition：这相当于给 model 一个 "scratchpad + critic" 的环境。Model 第一次答错，verifier 告诉它错哪了，model 在第二轮可以基于 feedback 修正。这个 multi-turn 结构是 RL 能 generalize 的关键之一（后面 Section 5.5 会验证）。

### 3.5 PPO 作为 backbone

Policy network 就是 foundation model 本身：$\pi_\theta : \mathcal{S} \to \mathcal{V}^n$，用 [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347) 更新。这个选择 follow 了 [Zhai et al. 2024a (RL4VLM)](https://openreview.net/forum?id=nBjmMF2IZU)。

---

## 4. Reward Design 细节

### 4.1 GeneralPoints Reward

| 情形 | Reward |
|---|---|
| 合法等式 = target | $r = +5$ |
| 合法但 ≠ target | $r = -1$ |
| 超过 max verification step (5) | $r = -1$ |
| 用了不在卡上的数字 | $r = -2$ |
| 其他非法等式 | $r = -3$ |
| GP-VL 额外：识别错牌 | $r = -1.5$ (额外) |

注意这个 reward 是 **outcome-based**：只要最终等式对就 +5，对中间过程没奖励。这是和 process supervision（[OpenAI's PRM, Lightman et al.](https://arxiv.org/abs/2305.20050)）的关键区别。作者的猜想是：**正是因为不监督中间过程，model 才有 freedom 去 explore 不同的 reasoning path，从而学到 generalizable 的策略**。

### 4.2 V-IRL Reward

| 情形 | Reward |
|---|---|
| 当前坐标上 action 正确 | $r = +1$ |
| 当前坐标上 action 错误 | $r = -1$ |
| 超过 max step (2) | $r = -1$ |
| landmark detection 失败 | $r = -1.5$ |

V-IRL 的 max verification step 只有 2，比 GP 的 5 小很多。这个差异暗示 navigation task 的 search space 比 arithmetic 更窄。

---

## 5. 主要实验结果

### 5.1 Rule-based Generalization（Figure 5, 6）

| Task | SFT ΔOOD | RL ΔOOD |
|---|---|---|
| GP-L | -8.1% (11.5→3.4) | **+3.5% (11.5→15.0)** |
| GP-VL | -5.6% (11.2→5.6) | **+3.0% (11.2→14.2)** |
| V-IRL-L | -79.5% (80.8→1.3) | **+11.0% (80.8→91.8)** |
| V-IRL-VL | -33.2% (35.7→2.5) | **+9.3% (35.7→45.0)** |

**Observation**：
- RL 在所有 4 个 setting 下都 improve OOD performance
- SFT 在所有 setting 下都 degrade OOD performance
- V-IRL-L 上 SFT 的崩塌最夸张：-79.5%。从 80.8% 掉到 1.3%，几乎完全 memorize 了 absolute direction 这个 specific action space，碰到 relative direction 就完全失效

### 5.2 Visual Generalization（Figure 7）

| Task | SFT ΔOOD | RL ΔOOD |
|---|---|---|
| GP-VL (black→red suits) | -9.9% (23.6→13.7) | **+17.6% (23.6→41.2)** |
| V-IRL-VL (NYC→worldwide) | -5.6% (16.7→11.1) | **+61.1% (16.7→77.8)** |

V-IRL-VL 上 RL 拿到了 **SOTA +33.8% (44.0→77.8)**，超过原 V-IRL paper 报告的最佳结果（且原 SOTA 用了 closed-source GPT-4 + 两阶段 VLM-LLM collaboration + prompt engineering，作者用 open-source Llama-3.2-Vision-11B end-to-end RL 就超过了）。

### 5.3 RL 提升视觉识别能力（Figure 8）

这是 paper 里我觉得最有 insight 的 ablation 之一。在 GP-VL 上：

- **Scaling up RL compute** → visual recognition accuracy **提升**，overall success rate **提升**
- **Scaling up SFT compute** → visual recognition accuracy **下降**，overall success rate **下降**

作者的 hypothesis：SFT 因为 reasoning tokens 频率高于 recognition tokens（看 Figure 11 的 example，`formula` 字段的 token 比 `cards` 字段多），导致 model 在 cross-entropy 上过度 overfit 到 reasoning tokens，反而 "forget" 了 visual recognition。这和 [Zhai et al. 2024b](https://arxiv.org/abs/2310.14004) 报告的 VLM fine-tuning 中 catastrophic forgetting 现象一致。

而 RL 的 outcome reward 不区分 reasoning 和 recognition token —— 它只看 final outcome。所以 model 必须先把 visual recognition 做对，才能拿到 reward，于是 visual perception 被 "顺便" 训练好了。

### 5.4 SFT 对 RL 是必要的（Figure 9）

直接对 base Llama-3.2-Vision-11B 跑 RL（跳过 SFT）→ **完全失败**。

Failure mode（Figure 20）：base model 输出冗长、跑题、无结构的 response（甚至试图写 Python 代码暴力枚举），导致无法 parse 出 task-related info 和 reward signal。

但要注意：这个结论和 [DeepSeek-R1 (DeepSeekAI et al. 2025)](https://arxiv.org/abs/2501.12948) 报告的 "SFT 不必要" 不矛盾。DeepSeek-R1 的 base 是 DeepSeek-V3，本身已经经过大量 instruction tuning；而这里的 base 是 Llama-3.2-Vision，instruction following 不够好。

**Takeaway**：SFT 的 role 是 **format teacher**（[LIMA, Zhou et al. 2024](https://arxiv.org/abs/2305.11206) 的说法），让 model 输出结构化 JSON；RL 在此基础上才能 explore。但 SFT 本身不该被 scale 太多 —— 作者在 Figure 19 展示：如果从一个已经 overfit 的 SFT checkpoint 开始 RL，RL 也救不回来。

### 5.5 Verification Iterations 的 scaling（Figure 10）

在 GP-L 上固定 compute budget，变化 max verification steps：

| VIter | OOD improvement |
|---|---|
| 1 | +0.48% |
| 3 | +2.15% |
| 5 | +2.99% |
| 10 | +5.99% |

更多 verification steps → 更好 generalization。这呼应了 [Snell et al. 2024](https://arxiv.org/abs/2408.03314) 和 [OpenAI o1 (Jaech et al. 2024)](https://arxiv.org/abs/2412.16720) 的 "test-time compute scaling"。

**Intuition**：每个 verification step 是一次 "self-correction" 的机会。Model 在第 1 步可能用了一个 spurious pattern，verifier 否定之后，第 2 步被迫 explore 别的路径。这种 "试错 + 反馈 + 修正" 的过程让 model 跳出 memorized pattern。

---

## 6. 为什么 RL Generalizes 而 SFT Memorizes？我的 Intuition

这是 paper 没有完全给出理论解释的地方，但我可以基于现有证据推几个 hypothesis：

### 6.1 Loss landscape 视角

SFT 最小化：

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log p_\theta(y|x) \right]$$

这个 loss 鼓励 model 把 $p_\theta(y|x)$ 推向 $p_{\text{data}}(y|x)$。在 finite data 下，最直接的 path 是 memorize $(x,y)$ pair。任何 "underlying rule" 的 hypothesis 都得和 memorize 的 hypothesis 竞争 —— 而在 finite training steps 下，memorize 通常 wins（参考 [Allen-Zhu & Li 2023a](https://arxiv.org/abs/2309.14316) 的 Physics of LLM Part 3.1）。

RL 优化的是：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t r_t \right]$$

这个 objective **不直接约束 $p_\theta$ 的形状**，只约束它的 expected return。Model 可以通过无数种 $p_\theta$ 实现 high return，其中只有少数是 "memorize training data" 这条 path。RL 的 stochastic policy + exploration 倾向于找到更 general 的 solution。

### 6.2 Inductive bias 视角

SFT 的 supervised target 是 **one specific solution**（expert trajectory）。Model 看到的所有 "正确答案" 都是同一个 pattern。

RL 的 reward signal 是 **outcome-level**。同一个 outcome 可以由 many different trajectories 达成。Model 通过 rollouts 看到多样化的 successful paths，自然学到 "what's invariant across all successful paths" —— 这就是 generalizable rule。

这点和 [AlphaZero 的 intuition](https://arxiv.org/abs/2405.03553) 很像：self-play / RL 让 agent 看到远比 supervised expert data 多的 state distribution。

### 6.3 Reward sparsity 视角

Outcome-based reward 只在 episode 末尾给信号。这迫使 model 在内部 representation 上做 "credit assignment" —— 它得自己 figure out 哪个中间 token 决定了成败。这个 credit assignment 过程就是 "discover underlying rule" 的过程。

而 SFT 的 dense token-level cross-entropy 不需要 credit assignment —— 每个 token 都有 supervised signal，model 可以 "懒" 地直接 mimic，不需要 discover rule。

---

## 7. 计算量的 FLOPs 估算

作者用了 [Hoffmann et al. 2023 (Chinchilla)](https://arxiv.org/abs/2203.15556) 和 [Snell et al. 2024](https://arxiv.org/abs/2408.03314) 的估算：

$$X_{\text{train}} = 6ND_{\text{train}}, \quad X_{\text{inference}} = 2ND_{\text{inference}}$$

- $N$：model 参数数量
- $D_{\text{train}}$：训练 token 数

SFT 和 RL 的 total compute：

$$X_{\text{SFT}} = 6N(D_{\text{init}} + D_{\text{SFT}})$$

$$X_{\text{RL}} = 6N(D_{\text{init}} + D_{\text{RL}}) + 2ND_{\text{buffer}}$$

PPO 是 on-policy，需要 iterative rollout + optimization，所以多了一项 inference compute $2ND_{\text{buffer}}$。

$D_{\text{buffer}}$ 的近似：

$$D_{\text{buffer}} \approx \frac{E \bar{d}_i \bar{d}_o}{D_{\text{RL}}} \cdot D_{\text{RL}} = \lambda D_{\text{RL}}$$

- $E$：auto-regressive generation 次数
- $\bar{d}_i, \bar{d}_o$：平均 input/output token 长度
- $\lambda$：GeneralPoints 上 ≈ 6，V-IRL 上 ≈ 5.1

所以 RL 大约比 SFT 多消耗 6-7x 的 compute per training token。这是 RL "贵" 的原因。但 Figure 5 的曲线是按 GFLOPs 画的，RL 在等量 compute 下依然 beat SFT，说明这个 generalization 优势不是 "RL 用了更多 compute" 导致的 artifact。

---

## 8. 局限性和作者承认的 failure modes

### 8.1 SFT 在 GP-VL 上完全失败（Figure 16, 17）

作者跑了 10 个不同 learning rate + 不同 frozen component 的 SFT ablation，没有一个能超过 30% success rate。Hypothesis：SFT 局部 overfit 到 reasoning tokens，忽视了 recognition tokens。

### 8.2 RL 救不了 overfitted checkpoint（Figure 19, 21）

如果从一个 per-step accuracy < 1% 的极度 overfit SFT checkpoint 开始 RL，RL 也无法 recover OOD。Model collapse 到 training rule 上，无法 escape。

**这暗示一个重要 practical insight**：RL 的 generalization 优势依赖 starting checkpoint 处于 "还未 memorize" 的 regime。一旦 model 已经 commit 到 spurious pattern，RL 的 gradient signal 不足以把它拉回来。这和 [Kumar et al. 2022 (implicit under-parameterization)](https://arxiv.org/abs/2210.05649) 的观察一致。

### 8.3 只测了 Llama-3.2-Vision-11B

没在更大 model 上验证。可能 RL 的 generalization 优势在 larger model 上更明显（因为 larger model 有更强的 inductive bias 去 discover rule），也可能反而更弱（因为 larger model memorize 能力更强）。这个 scaling 实验留给了 future work。

---

## 9. 和你（Karpathy）之前一些观点的 connection

### 9.1 "Software 2.0" 和 RL

你之前讲过 "Software 1.0 is explicit code, Software 2.0 is weights learned from data"。这篇 paper 进一步指出：**同样在 Software 2.0 框架下，loss function 的选择（cross-entropy vs policy gradient）会产生质的区别**。SFT 和 RL 都是 "学 weights"，但 RL 学出来的是更 "algorithmic" 的 representation。

### 9.2 "micrograd / 对 backprop 的 intuition"

你过去在 micrograd 里强调 backprop 是 "cheap and local"。这篇 paper 的 multi-turn RL 暴露了一个问题：reward 是 sparse 且 global（episode-level），gradient 通过 backprop 传回去时，每个 token 都得到相同的 "advantage signal"。这种 **broadcast** 让 model 学到的 representation 是 "rule-level" 而非 "token-pattern-level"。

### 9.3 和 o1 / DeepSeek-R1 的 connection

这篇 paper 的 multi-turn verification 思路和 [OpenAI o1 (Jaech et al. 2024)](https://arxiv.org/abs/2412.16720)、[DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) 的 "thinking before answering" 是同一回事。R1 用纯 RL（无 SFT warmup，base 是 DeepSeek-V3）训出 strong reasoning，证明：**当 base model 已经会 follow instructions，RL 可以直接教它 reasoning，而不需要 SFT 示范 reasoning trajectory**。这正好是这篇 paper Figure 9 的 counter-example —— 当 base 不会 follow instructions 时，RL 失败；当 base 会 follow instructions（R1 的 case）时，RL 不需要 SFT 就能 generalize。

### 9.4 "Lessons from Llama-3" paper

[Llama 3 herd paper (Dubey et al. 2024)](https://arxiv.org/abs/2407.21783) 里提到他们的 post-training 是 "multiple rounds of RLHF + SFT"。这篇 paper 给这个 multi-round 设计提供了理论支撑：SFT 教 format，RL 教 generalizable capability，然后下一轮 SFT 再 fix 一些 regression，再下一轮 RL 再 generalize。这种 "format teacher + generalizer" 的 role split 可能是 multi-round pipeline 工作的根本原因。

---

## 10. 相关延伸阅读

- [RL4VLM (Zhai et al. 2024a, NeurIPS 2024)](https://openreview.net/forum?id=nBjmMF2IZU) — 这篇 paper 的方法论基础
- [Snell et al. 2024 — Scaling test-time compute](https://arxiv.org/abs/2408.03314) — sequential revision formulation 的来源
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — 在强 base 上纯 RL 的 scaling 极致
- [OpenAI o1 system card (2024)](https://arxiv.org/abs/2412.16720) — test-time verification 的工业级实现
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347) — 用的 RL 算法
- [LIMA (Zhou et al. 2024)](https://arxiv.org/abs/2305.11206) — SFT 是 "format teacher" 的原始论述
- [V-IRL (Yang et al. 2024a)](https://arxiv.org/abs/2410.07163) — V-IRL 环境的原始 paper
- [Physics of LLMs Part 3.1 (Allen-Zhu & Li 2023)](https://arxiv.org/abs/2309.14316) — memorization vs generalization 的理论基础
- [Hoffmann et al. 2023 (Chinchilla)](https://arxiv.org/abs/2203.15556) — compute 估算公式来源
- [Cobbe et al. 2021 (verifiers for math)](https://arxiv.org/abs/2110.14168) — outcome-based verifier 的思路源头
- [Lightman et al. 2023 (PRM, Let's verify step by step)](https://arxiv.org/abs/2305.20050) — process-based supervision，可作为这篇 paper 的对照（这篇用 outcome-based）
- [Llama 3 herd of models (2024)](https://arxiv.org/abs/2407.21783) — backbone model + multi-round post-training 的工业实践
- [FLAN (Wei et al. 2022a)](https://arxiv.org/abs/2109.01652) — SFT 的 zero-shot generalization 早期证据

---

## 11. 我对这篇 paper 的整体评价

**Strengths**：
1. 实验设计干净：rule 和 visual 两个 axis，LLM 和 VLM 两种 modality，结论一致
2. 对 SFT/RL 的 role 做了清晰 disentangle —— SFT 教 format，RL 教 generalization
3. V-IRL 上拿到 SOTA，说明这是 practical finding 而非 toy result
4. Figure 8 的 ablation 揭示 "RL 顺便提升了 visual recognition" 是个让人惊喜的 by-product

**Open questions**：
1. 没在 >11B 的 model 上验证 scaling behavior
2. "RL 为什么 generalize" 缺乏正式理论，目前只是 hypothesis
3. Reward shaping 的 robustness 没系统研究 —— 不同 reward function 下 RL 的 generalization 是否稳定？
4. 和 process-based reward (PRM) 的对比缺失。如果用 PRM 训练，generalization 会变好还是变差？我直觉是变差，因为 PRM 把 "正确路径" 又 constrain 回 specific pattern 了，但需要实验验证

**和当前 frontier 的 alignment**：这篇 paper 的结论和 DeepSeek-R1、OpenAI o1 的工程经验完全一致 —— **RL 是 reasoning capability 的主要来源，SFT 是 format / instruction following 的来源**。这是 2024-2025 年 LLM 训练最重要的 paradigm shift 之一，这篇 paper 给它提供了 clean 的 experimental evidence。
