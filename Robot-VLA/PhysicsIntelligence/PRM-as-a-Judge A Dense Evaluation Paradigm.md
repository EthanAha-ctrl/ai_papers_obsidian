---
source_pdf: PRM-as-a-Judge A Dense Evaluation Paradigm.pdf
paper_sha256: ef82d317cf579a641fdc4d8405dacd3b710a5bac3d6d1f535802652bd63a57c4
processed_at: '2026-08-06T06:25:04-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 一句话说清楚

现在评估机器人做得好不好，基本就是看"成没成功"——一个 0 或 1 的数字。但这篇paper说：这太粗糙了，就像用"活没活"来评价一个人的人生质量。我们应该看整个过程，就像看一部电影而不是只看结局。

---

## 为什么 binary success rate 不够用

想象两个机器人都在做"把杯子挂到钩子上"这个任务：

- **机器人 A**：稳稳地拿起来，平滑地移过去，对准，挂上。整个过程行云流水。
- **机器人 B**：拿起来，掉了；再拿，拿歪了，换角度；好不容易移过去，没对准，退回来重新对；折腾半天终于挂上了。

binary success rate 的评价：两个都是 1，一样好。

这显然不对。而且还有更隐蔽的问题——**失败的两种情况被混为一谈**：

- **Near-miss failure**：杯子都送到钩子旁边了，就差最后 1 毫米没挂上去
- **Early collapse**：刚伸手就把杯子碰飞了

binary SR 对这两种情况都给 0，完全无法区分。

---

## 这篇paper 的核心思路

既然 binary success rate 不够，我们需要一个**密集的、过程感知的评估方法**。这篇paper 的做法是：

### Step 1: 找一个能打分的"裁判"

你需要一个模型，给它看机器人执行过程中的每一帧画面，它都能告诉你："当前离完成任务有多近"。输出一个 0 到 1 的数字，0 是刚开始，1 是完成。

这个数字就是 $\Phi(x_t)$——**progress potential**。

paper 的关键 insight 是：**Process Reward Model (PRM)** 天生适合干这个。PRM 原本是用在 RL training 里提供 dense reward 的，这篇paper 把它"转岗"用来做 evaluator。

### Step 2: 这个裁判必须满足两个条件

**条件一：Macro-Consistency（宏观一致性）**

你从第 0 帧到第 100 帧的 progress，应该等于你从第 0 帧到第 50 帧的 progress 加上第 50 帧到第 100 帧的 progress。

$$
S(x_0, x_{100}) = S(x_0, x_{50}) + S(x_{50}, x_{100})
$$

这听起来像废话，但很多 evaluator 其实做不到。比如 CLIP-based 的方法，它是算两帧之间的 similarity，这个 similarity 没有一个全局一致的坐标系。就像你用温度计量体温，每次量的刻度基准都在漂移，加起来就不对了。

paper 在附录 C 里严格证明了：只要你的 evaluator 输出的是一个 absolute scalar $\Phi(x)$，然后用差值 $S(x_i, x_j) = \Phi(x_j) - \Phi(x_i)$ 来算 progress，就天然满足这个条件。这就像物理学里的保守力场——做功只跟起点和终点有关，跟路径无关。

**条件二：Micro-Resolution（微观分辨率）**

裁判必须能察觉到微小的物理变化。比如夹爪往前移了 1 毫米，这个变化在画面上可能几乎看不出来，但对任务来说可能是关键的一步。

paper 发现，general-purpose VLM（如 GPT-5.2、Gemini 3 Pro）在这一点上表现很差。在 RoboPulse benchmark 的 Small hop 测试中，GPT-5.2 的准确率只有 0.47，基本等于瞎猜。因为这些模型的内部 representation 是为 semantic understanding 服务的，不是为 fine-grained physical reasoning 服务的。

而 trajectory-trained PRM（如 Robo-Dopamine）能达到 0.83，因为它在训练时就见过大量机器人执行轨迹，学会了理解"夹爪靠近物体 1 毫米"这种细微但重要的变化。

---

## OPD Metric System：把过程拆成三层

有了 $\Phi(x_t)$ 这个 progress potential，paper 定义了一组 metrics，把执行过程拆成三个层次来分析。

### 第一层：Outcome（结果层）——走多远

**Milestone Coverage (MC)**：把 0 到 1 的 progress 分成四段：0.25, 0.5, 0.75, 1.0。看机器人最远走到了哪一段。

```
MC = 0.25  →  只完成了第一步（比如接近物体）
MC = 0.50  →  完成了第二步（比如抓住物体）
MC = 0.75  →  完成了第三步（比如移到目标位置）
MC = 1.00  →  完全成功
```

这个比 binary SR 精细多了。两个 policy 的 SR 都是 0，但一个 MC 是 0.75（差最后一步），另一个 MC 是 0.25（一开始就崩了），诊断价值完全不同。

**Max Progress (MP)**：整个 episode 中 progress 的最大值。比 MC 更精细，是连续值。

### 第二层：Process（过程层）——走得有多顺

**Path-weighted Progress Length (PPL)**：

$$
\mathrm{PPL} = \Phi(x_T) \times \frac{\text{净 progress}}{\text{总 progress 波动} + \delta}
$$

分子是终点 progress 减起点 progress（净进展）。分母是每一步 progress 变化的绝对值之和（总波动，包括前进和后退）。

如果机器人一路顺风，progress 单调上升，那总波动就等于净进展，比值接近 1。如果机器人反复折腾——前进一步、退两步、再前进——总波动远大于净进展，比值就很小。

前面乘的 $\Phi(x_T)$ 是一个"完成度门槛"，防止那种"动了一点点就停下来"的 policy 拿到不公平的高分。

**PPL 高 = 走得直、走得顺；PPL 低 = 走得弯、反复折腾。**

### 第三层：Diagnosis（诊断层）——为什么失败

**Cumulative Regret Area (CRA)**：

$$
\mathrm{CRA} = \frac{1}{T+1} \sum_{t=0}^{T} \left(\max_{k \le t} \Phi(x_k) - \Phi(x_t)\right)
$$

每一时刻，算"历史最高 progress"减"当前 progress"。如果机器人到了 0.8 然后掉到 0.2 并一直没爬起来，那 regret 就很大。如果马上恢复了，regret 就很小。

**CRA 高 = 掉下来之后爬不回去（不稳定、recovery 能力差）。**

**Stagnation Ratio (STR)**：

$$
\mathrm{STR} = \frac{\text{progress 几乎没变的时间步数}}{\text{总时间步数}}
$$

**STR 高 = 停在某不动了（犹豫、卡住、decision uncertainty）。**

---

## RoboPulse：专门考裁判的 benchmark

paper 还做了一个 benchmark 来测试各种 evaluator 的微观分辨率。

核心设计很聪明——**Hop-based normalization**：

在任务早期，从"离物体 10 厘米"到"离物体 9 厘米"，绝对 progress 变化很小。在任务晚期，从"离钩子 1 厘米"到"挂上钩子"，绝对 progress 变化可能也就 0.05。但前者的意义远不如后者。

所以 RoboPulse 用相对 normalization：

$$
\mathcal{H} = \frac{\Phi(x_q) - \Phi(x_p)}{\Phi(x_{\text{end}}) - \Phi(x_p)}
$$

分母是"还剩多少路要走"。这样早期的微小位移和晚期的微小位移在 $\mathcal{H}$ 的尺度上变得可比了。

然后 benchmark 把测试分成 Small / Medium / Large 三种 hop 范围，看 evaluator 在不同粒度下的表现。

### 实验结果（Table 2）：

| 方法 | Small hop | Medium hop | Large hop | 平均 |
|------|-----------|------------|-----------|------|
| CLIP ViT-L/14 (I2I) | 0.54 | 0.59 | 0.65 | 0.59 |
| Gemini 3 Pro | 0.54 | 0.67 | 0.77 | 0.66 |
| GPT-5.2 | 0.47 | 0.51 | 0.62 | 0.53 |
| Qwen3-VL-8B | 0.48 | 0.57 | 0.74 | 0.59 |
| VLAC (PRM) | 0.61 | 0.72 | 0.79 | 0.71 |
| GVL (PRM) | 0.63 | 0.72 | 0.78 | 0.71 |
| **Robo-Dopamine (PRM)** | **0.80** | **0.85** | **0.85** | **0.83** |

**关键发现**：在 Small hop（最难的、变化最细微的）条件下，PRM 类方法把 VLM 甩了一大截。Robo-Dopamine 0.80 vs GPT-5.2 0.47。这说明 VLM 的 visual representation 根本无法分辨细微的物理变化，它们的 latent space 缺乏这种 resolution。而 PRM 通过在大量机器人轨迹上做 contrastive learning，学到了一个对物理 progress 敏感的连续 manifold。

---

## 在真实 policy 上的审计结果

paper 在 RoboTwin 2.0 上测了 5 个 policy family：ACT, DP, RDT, $\pi_0$, OpenVLA-OFT。

### 发现 1：不同 policy 的"失败位置"完全不同（RQ2）

以 Blocks Ranking RGB 任务为例：

| Policy | MC@25 | MC@50 | MC@75 | MC@100 |
|--------|-------|-------|-------|--------|
| ACT | 84 | 44 | 22 | 2 |
| DP | 94 | 40 | 18 | 0 |
| RDT | 100 | 62 | 30 | 0 |
| $\pi_0$ | 96 | 66 | 40 | 8 |
| OpenVLA-OFT | 98 | 42 | 6 | 0 |

binary SR 看：RDT、DP、OpenVLA-OFT 都是 0，$\pi_0$ 是 8。看起来差不多？

但 MC@75 告诉你：$\pi_0$ 有 40% 的 episode 走到了 75% 的进度，而 OpenVLA-OFT 只有 6%。$\pi_0$ 是"差最后一口气"，OpenVLA-OFT 是"走了一半就崩了"。

### 发现 2：成功的轨迹质量也天差地别（RQ3）

在 Handover Mic 任务上，只看成功的 episode：

| Policy | MC@100 | PPL | CRA | STR |
|--------|--------|-----|-----|-----|
| DP | 44 | 65.97 | 1.05 | 57.18 |
| RDT | 100 | 84.23 | 1.45 | 39.82 |
| $\pi_0$ | 98 | 88.05 | 1.03 | 42.71 |
| OpenVLA-OFT | 76 | 66.20 | 5.66 | 45.14 |

DP 的 PPL 最低（65.97）但成功时 CRA 也最低（1.05），说明它成功时走得很顺。但它的 MC@100 只有 44——成功率低。这说明 DP 有一个"窄成功域"：一旦它的 diffusion process 走对了路，就一路顺到底；但稍有偏差就崩。

$\pi_0$ 的 MC@100 是 98，几乎总是成功，PPL 也最高（88.05），CRA 极低（1.03）。这说明 $\pi_0$ 既可靠又高效。

OpenVLA-OFT 的 CRA 高达 5.66，说明即使它成功了，过程中也有大量 backtrack 和 correction，走得很挣扎。

### 发现 3：失败模式可以"指纹化"（RQ4）

把失败 episode 的 OPD metrics 做 z-score normalization 后，不同 policy family 有稳定的 failure fingerprint：

- **DP**：高 STR，低 CRA → 停滞型失败，走到一半卡住不动了
- **OpenVLA-OFT**：高 MP + 高 CRA → 后期崩盘型失败，走到很后面然后剧烈后退
- **ACT**：低 MP + 高 STR → 早期停滞型失败，一开始就找不到方向

这些 fingerprint 能告诉你改进方向：
- 高 STR 的 policy 可能需要更好的 exploration mechanism 或 contact maintenance
- 高 CRA 的 policy 可能需要更好的 error recovery 和 corrective control
- 低 PPL 的 policy 可能需要减少 redundant motion

---

## 我的直觉与联想

### 关于 VLM 为什么不行

GPT-5.2 在 Small hop 上只有 0.47 的准确率，基本是瞎猜。我的直觉是：VLM 的 vision encoder 输出的是离散的 semantic token，这些 token 对"门开没开""杯子在不在"这种粗粒度状态有响应，但对"夹爪移动了 1 毫米"这种连续物理变化几乎无感。这就像用 720p 的屏幕去显示 4K 的信号——分辨率不够，细节全丢了。

PRM 通过 contrastive learning 在 trajectory 数据上训练，强制相邻 frame 在 latent space 中靠近、非相邻 frame 远离，这就 carve out 了一条连续的 progress manifold。这条 manifold 的"分辨率"是由训练数据的密度决定的，而机器人轨迹数据天然就是高密度的（每一帧都有标注的 progress）。

### 关于 Diffusion Policy 的窄成功域

DP 在成功时表现极好（低 CRA、路径顺），但成功率低。我的联想是：Diffusion 的 denoising process 是一个 iterative refinement，如果初始 noise sample 恰好落在"正确 basin"里，整个 process 就很顺；如果落在"错误 basin"里，denoising 会收敛到一个 local optimum，产生反复微调但无法推进的 action，表现为高 STR。这跟 diffusion model 在 image generation 里的 mode collapse 问题类似。

### 关于 Potential Field 与 RL Reward Shaping

这个 paper 的理论框架跟 RL 里的 potential-based reward shaping 高度一致：

$$
F(s, s') = \gamma \Phi(s') - \Phi(s)
$$

Ng et al. 1999 证明了这种 reward shaping 不改变 optimal policy。这意味着 PRM-as-a-Judge 评估出来的 progress 信号可以直接用来做 RL training 的 dense reward，而不会引入 bias。paper 最后也提到了这个方向。

### 关于未来方向

这篇paper 打开了一个很有想象力的方向：**用 PRM 不只做 evaluation，还做 training-time diagnosis**。想象一下，你在训练一个 VLA policy，每个 epoch 你都用 PRM-as-a-Judge 看一看 policy 的 OPD fingerprint。如果 STR 在上升，说明 policy 在某个 bottleneck 处卡住了，你可以调 learning rate 或加 exploration noise。如果 CRA 在上升，说明 policy 的 error recovery 在退化，你可能需要加更多 failure recovery 的 training data。

这把 evaluation 从"事后评判"变成了"训练过程中的实时诊断工具"，类似于给 policy 做连续的心电图。

---

## 参考链接

- **项目主页**: https://PRM-as-a-Judge.github.io
- **Robo-Dopamine** (paper 使用的主力 PRM): https://arxiv.org/abs/2512.23703
- **RoboTwin 2.0** (policy 审计用的 benchmark): https://arxiv.org/abs/2506.18088
- **VLAC** (baseline PRM): https://arxiv.org/abs/2509.15937
- **GVL** (baseline PRM, Ma et al. 2024): https://arxiv.org/abs/2403.12945 (相关)
- **Qwen3-VL** (baseline VLM): https://arxiv.org/abs/2511.21631
- **Ng et al. 1999** (potential-based reward shaping 理论基础): https://arxiv.org/abs/9901.0210 (经典)
- **LLM-as-a-Judge** (NLP 领域的灵感来源): https://arxiv.org/abs/2306.05685

---

在这篇paper中，作者提出了一种名为 **PRM-as-a-Judge** 的 dense evaluation paradigm，旨在解决 robotic manipulation 领域长期依赖 binary success rate 所导致的评估信息丢失问题。因为 robotic manipulation 正在从 short-horizon skill 向 long-horizon, contact-rich task 演进，单一的 binary success rate 无法区分 near-miss failure 与 early collapse，同时也无法揭示 policy 在 execution 过程中的 stability 与 efficiency。

为了建立你的 intuition，我将从理论公理、metric 公式解析、benchmark 设计以及实验数据四个维度进行深度拆解，并融入一些关于 representation space 与 policy architecture 的联想。

### 1. Theoretical Foundations: Axioms & Potential Field

要成为一个合格的 dense evaluator，paper 提出了两个必须满足的 axioms：

**Axiom 1: Macro-Consistency via Temporal Additivity**
要求评估器在不同时间尺度上保持一致。假设 $S(x_i, x_j)$ 表示从 state $x_i$ 到 $x_j$ 的 estimated progress，对于任何时间段 $[t_0, t_2]$ 及其内部任意时刻 $t_1$，必须满足：
$$
S(x_{t_0}, x_{t_2}) = S(x_{t_0}, x_{t_1}) + S(x_{t_1}, x_{t_2}) \tag{1}
$$
这里 $x_t$ 表示在时间 $t$ 时 judge 可获取的 task-relevant information state（如 observation）。这个公式意味着局部 progress 的累加必须等于全局 progress，与时间序列如何切分无关。在此基础下，paper 定义了基于 potential 的 progress 计算方式：
$$
S(x_i, x_j) = \Phi(x_j) - \Phi(x_i) \tag{2}
$$
其中 $\Phi(x_t) \in [0, 1]$ 是一个标量势能函数，表示当前 state 相对于 task goal 的 progress。

**Intuition:** 这非常类似于物理学中的保守力场做功，或者强化学习中基于势能的 reward shaping ($F(s,s') = \gamma \Phi(s') - \Phi(s)$)。只要评估器输出的是一个全局一致的 absolute scalar $\Phi(x)$，通过差分得到的局部 progress 天然满足可加性。附录 C 中严格证明了基于 similarity 或 pairwise relative comparison 的评估器（如 CLIP）通常无法满足这个 cocycle identity，因为 observation function $g(s)$ 在 task-equivalent states 上不是单射的，会导致 scale drift。

**Axiom 2: Micro-Resolution of Progress Signals**
要求 evaluator 对细粒度的、task-relevant 的物理演化保持敏感。即使 $\Delta$ 很小，$\Phi(x_{t+\Delta}) - \Phi(x_t)$ 也应该反映出非退化的物理变化，不能全部塌缩为 0。这要求模型在 local geometry 上具有极高的分辨率。

### 2. OPD Metric System: Formulas & Intuitions

基于 $\Phi(x)$，paper 构建了 OPD (Outcome-Process-Diagnosis) metric system。

#### 2.1 Outcome Level
衡量 policy 能走多远。

*   **Milestone Coverage (MC):** 将 progress 空间离散化为 quartiles $\mathcal{Q} = \{0, 0.25, 0.5, 0.75, 1\}$。
    $$
    \mathbf{MC}(\tau) = \max \{q \in \mathcal{Q} \mid \exists t, \Phi(x_t) \ge q\} \tag{3}
    $$
    $\tau$ 是整个 trajectory。$\mathbf{MC}$ 相当于一个 soft success rate。它能区分在 final alignment 阶段失败 ($\mathbf{MC}=0.75$) 的 policy 与在 approach 阶段就失败 ($\mathbf{MC}=0.25$) 的 policy，而 binary SR 将两者都记为 0。

*   **Max Progress (MP):** 整个 episode 中达到的最大势能。
    $$
    \mathbf{MP}(\tau) = \max_{t \in [0, T]} \Phi(x_t) \tag{4}
    $$
    反映了 policy 的能力边界。

#### 2.2 Process Level
衡量执行效率。

*   **Path-weighted Progress Length (PPL):**
    $$
    \mathrm{PPL}(\tau) = \Phi(x_T) \cdot \frac{[\Phi(x_T) - \Phi(x_0)]_+}{\sum_{t=1}^T |\Phi(x_t) - \Phi(x_{t-1})| + \delta} \tag{5}
    $$
    变量解析：$x_T$ 是 terminal state，$x_0$ 是 initial state，$[\cdot]_+$ 表示 $\max(x, 0)$，$\delta$ 是为了防止除以 0 的极小常数（如 $10^{-8}$）。分母 $\sum |\Phi(x_t) - \Phi(x_{t-1})|$ 是 Total Variation (TV)，衡量轨迹在势能场上的总动荡。
    **Intuition:** 如果 policy 在势能场上反复横跳（来回尝试、失败、重试），TV 会非常大，导致 PPL 降低。前面的乘法项 $\Phi(x_T)$ 是一个 completion gate，确保那些仅仅在初始阶段有一点小 progress 然后就停滞的 policy 无法获得高 PPL 分数。

#### 2.3 Diagnosis Level
定位失败的性质。

*   **Cumulative Regret Area (CRA):**
    $$
    \mathbf{CRA}(\tau) = \frac{1}{T+1} \sum_{t=0}^T \left[ \max_{0 \le k \le t} \Phi(x_k) - \Phi(x_t) \right] \tag{6}
    $$
    变量解析：$T$ 是总步数，$\max_{0 \le k \le t} \Phi(x_k)$ 是到时刻 $t$ 为止的历史最高势能 $M_t$。方括号内的项 $R_t = M_t - \Phi(x_t)$ 是瞬时 regret。
    **Intuition:** CRA 衡量的是“跌落神坛后的持续时间与深度”。如果 policy 达到了 0.8 的 progress，然后掉到 0.2 并一直停滞，CRA 会很高；如果它马上恢复到 0.8，CRA 就很低。这比简单的 Regression Rate (RR) 更能捕捉 persistent failure。

*   **Stagnation Ratio (STR):**
    $$
    \mathrm{STR}(\tau) = \frac{1}{T} \sum_{t=1}^T \mathbb{I}(|\Phi(x_t) - \Phi(x_{t-1})| < \epsilon) \tag{7}
    $$
    变量解析：$\mathbb{I}$ 是指示函数，$\epsilon$ 是根据 judge noise 校准的微小阈值。
    **Intuition:** 衡量 policy “发呆”或无效微调的时间比例。高 STR 通常意味着 policy 在某个 bottleneck 处缺乏 escape capability 或陷入了 decision uncertainty。

### 3. RoboPulse Benchmark & Micro-Resolution

为了验证 PRM 的 micro-resolution，paper 提出了 RoboPulse，包含 1800 个 pairwise progress judgment cases。其核心设计是 **Hop-based normalization**：

$$
\mathcal{H}(x_p, x_q) = \frac{\Phi(x_q) - \Phi(x_p)}{\Phi(x_M) - \Phi(x_p)} \quad (\text{Forward Progress})
$$
这里 $x_p$ 是 pre-state，$x_q$ 是 post-state，$x_M$ 是 terminal state。
**Intuition:** 这个设计非常巧妙。在 task 的早期，移动 1 厘米在绝对 progress 上可能只增加 0.01；但在 final insertion 阶段，移动 1 厘米可能意味着从 0.95 到 1.0 的巨大跨越。通过除以剩余距离 $\Phi(x_M) - \Phi(x_p)$，使得早期的微小位移与晚期的微小位移在 $\mathcal{H}$ 的尺度上可比。这迫使 evaluator 必须理解 context-aware 的物理意义，而无法依赖单纯的 visual displacement。

### 4. Experimental Insights & Hallucinations

Table 2 的实验数据极具启发性。在 Small hop scale 下，Robo-Dopamine 达到了 0.80 的 accuracy，而 Gemini 3 Pro Preview 仅为 0.54，GPT-5.2 仅为 0.47（接近随机猜测）。

**Intuition & Hallucination on VLM Failure:**
为什么 frontier VLMs 在 small hop 上彻底失败？因为 VLMs 的 pretraining objective 主要捕捉 semantic coarse coding。它们能识别“门开着”与“门关着”的离散状态，但它们的 internal representation space 缺乏连续的 metric space。对于 visual differences 极小但 physical semantics 巨大的两帧（例如：夹爪刚刚接触物体 vs 夹爪距离物体 1 毫米），VLMs 的 vision encoder 输出的 token 几乎是相同的。由于缺乏 dense robotic trajectory 的 inductive bias，VLMs 无法在 latent space 中 carve out 一条平滑、连续的 progress manifold。这类似于在低分辨率的网格上试图计算微积分，必然导致信息丢失。而 trajectory-trained PRMs 通过 contrastive learning 强制拉近相邻 frame、推远非相邻 frame，从而在 observation space 中蒸馏出了一个连续的 potential field。

**Policy Auditing (Table 3 & Figure 4):**
在 Handover Mic 任务中，观察 DP 与 $\pi_0$ 的对比：
*   **DP:** $\mathbf{MC@100} = 44$, $\mathbf{PPL} = 65.97$ (较低), $\mathbf{CRA} = 1.05$ (极低)
*   **$\pi_0$:** $\mathbf{MC@100} = 98$, $\mathbf{PPL} = 88.05$ (高), $\mathbf{CRA} = 1.03$ (极低)

这揭示了“窄成功域”现象。DP 在成功时的轨迹非常优雅且低 regret，但它的 robustness 极差，一旦偏离 familiar manifold 就彻底崩溃。$\pi_0$ 则展现出了极强的 error recovery capability，其 success regime 覆盖了更广的 manifold 区域。

**Failure Fingerprints (Figure 5):**
观察 OpenVLA-OFT 与 ACT 的 failure mode：
*   **OpenVLA-OFT:** 表现出 high CRA（如 Place Bread Basket 中为 26.3），意味着它常常达到 late stage，然后发生 catastrophic regression 并反复尝试。
*   **ACT:** 表现出 high STR（如 Place Bread Basket 中为 65.4），意味着它在早期就陷入停滞，不断微调却无法推进。

**Hallucination on Architecture Inductive Bias:**
Diffusion Policy (DP) 的高 Stagnation Ratio 可能源于其 iterative denoising 过程。当遇到 OOD observation 时，denoising process 可能会收敛到一个 local optimum，导致 action 产生微小的往复震荡。而 OpenVLA-OFT 基于 auto-regressive next-token prediction，当它预测错误并收到 negative visual feedback 时，由于缺乏 RLHF 或类似的 correction mechanism，它可能会陷入条件概率的循环，反复生成曾经成功过但现在不适用的 action sequence，从而产生高 CRA。

### 5. Conclusion & References

PRM-as-a-Judge 将 robotic evaluation 从 discrete boolean logic 带入了 continuous potential field 的范畴。它证明了基于 dense trajectory supervision 的 PRM 可以作为物理世界 progress 的 caliper，为 policy 的 behavioral signature 提供了不可替代的诊断工具。这暗示了未来的 VLA 模型评估不仅要看 success rate，更需要审视其在 progress manifold 上的 geodesic efficiency 与 topological stability。

**Reference Links:**
*   Project Page: [https://PRM-as-a-Judge.github.io](https://PRM-as-a-Judge.github.io)
*   Robo-Dopamine (Core PRM used): [https://arxiv.org/abs/2512.23703](https://arxiv.org/abs/2512.23703)
*   RoboTwin 2.0 (Benchmark Environment): [https://arxiv.org/abs/2506.18088](https://arxiv.org/abs/2506.18088)
*   VLAC (Baseline PRM): [https://arxiv.org/abs/2509.15937](https://arxiv.org/abs/2509.15937)
*   Qwen3-VL (Baseline VLM): [https://arxiv.org/abs/2511.21631](https://arxiv.org/abs/2511.21631)
*   Gemini 3 Pro (Baseline VLM): [https://blog.google/innovation-and-ai/technology/developers-tools/gemini-3-pro-vision/](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-3-pro-vision/)
