---
source_pdf: ReLaX Reasoning with Latent Exploration for Large Reasoning Models.pdf
paper_sha256: 617813603d852596fb2d64794d877179c9ed9e5fad76b61bf0b4e8f04d8c374c
processed_at: '2026-08-11T22:33:59-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ReLaX 的"人话版"

好，我把那堆公式和术语翻译成大白话，顺便把直觉打通。

---

## 一句话总结

**你盯着 token 的 entropy 折腾半天没用，因为真正的病根在 hidden state 层面——hidden state 的"内部动态"已经僵化成死循环了，再怎么在 token 层面注入随机性也救不回来。ReLaX 用 Koopman operator 这个数学工具给 hidden state 做"频谱分析"，然后直接在 latent space 里鼓励模型多产生几种不同的"动态模式"，治本。**

---

## 1. 现有方法的困境——盯着浪花不看暗流

想象 LLM 生成 token 的过程是一出戏：**token 是演员念出来的台词，hidden state 是演员脑子里的思考过程。**

RLVR 训练中，reward 驱动 model 把 policy 越推越 deterministic。台词越来越固定。Cui et al. 的 KL-Cov paper 给出经验公式：

$$R = -a \cdot \exp(H) + b$$

意思是 token entropy $H$ 越高，reward 反而越低——policy 被推向低 entropy 角落。这就是所谓的 "entropy collapse"。

现有方法（DAPO、KL-Cov、FR3E、R1-zero-Div）都在 token 层面想办法：放宽 clip bound、selectively 惩罚高协方差 token、专门挑高 entropy 的 forking token 扩展 rollout、直接加 entropy regularization……

问题在于：**你可以逼演员念不同的台词（提高 token entropy），但如果他脑子里已经只有一套固定思维模式，他再怎么换台词也是机械重复——台词变了，思考模式没变。**

ReLaX 的洞察：**token entropy collapse 只是症状，latent dynamics 的 rigidification 才是病根。**

---

## 2. Hidden State 是个 dynamical system

Transformer 每次 forward，hidden state $x_t$ 演化：

$$x_t = \mathcal{F}(x_{t-1}, \omega_t)$$

- $x_t$: 第 $t$ 步的 last-layer hidden state，e.g. Qwen2.5-7B 是 3584 维
- $\mathcal{F}$: 整个 transformer 的一次 nonlinear 变换
- $\omega_t$: sampling noise（temperature/top-p/top-k）

这是个 stochastic nonlinear dynamical system。Token 是从这个 hidden state 通过 LM head decode 出来的 projection——损失了大量信息。

**Intuition**: Hidden state 才是真正的 computation 载体。如果 hidden state trajectory 退化到低维 manifold 上来回打转，再怎么在 token space 注入 entropy，stochastic input $\omega_t$ 都没办法 translate 成 diverse 的 latent trajectory——因为 intrinsic dynamics 已经僵了。

所以 ReLaX 的方向：**直接 measure 并 regularize hidden state dynamics 的 flexibility。**

---

## 3. Koopman Operator——给暗流做"频谱分析"

难点：hidden state 高维 + nonlinear，怎么分析？

Koopman theory 的 magic：**任何非线性动力学都能 embed 到 infinite-dimensional observable space 里，在那里 evolution 由一个 linear operator 主导。**

### 类比理解

想象你在看一段复杂的波形（比如声波）。直接看时域信号很难分析，但做 Fourier 变换把它分解成不同频率的 sine wave 叠加，就清晰了——每个 sine wave 是一个 "mode"，有自己的 frequency 和 amplitude。

Koopman operator 是 nonlinear dynamical system 的"广义 Fourier 变换"：
- 把 $x_t$ 映射到 observable $g(x_t)$
- 在 observable space 里，演化变成 linear：$g(x_{t+1}) = K g(x_t)$
- $K$ 的 eigenvalue 对应不同的 dynamical mode

### Eigenvalue 的物理含义

Koopman eigenvalue $\lambda$ 一般是复数：

- $|\lambda| < 1$: decay mode（瞬态扰动衰减）
- $|\lambda| > 1$: growth mode（不稳定方向）
- $|\lambda| \approx 1$, $\text{Im}(\lambda) \neq 0$: oscillation mode
- $\lambda \approx 1$: fixed/steady mode

**如果所有 eigenvalue magnitude 都接近 1，dynamics 退化成 near-identity 的稳定振荡——rigid，exploration 没空间。如果 magnitudes 分散（有 decay 有 growth 有 oscillation），dynamics 富有表达力。**

---

## 4. DSD——"模式种类有多丰富"

这是 paper 的核心 metric：

$$\text{DSD}(x) = \text{Var}(|\Lambda|)$$

就是把 Koopman operator 的所有 eigenvalue 取模长，然后求方差。

- DSD 高 → eigenvalue magnitude 分散 → 多种 mode 共存 → dynamics flexible → 探索能力强
- DSD 低 → eigenvalue magnitude 聚集 → 单一 mode 主导 → dynamics rigid → entropy collapse

**这就是给 hidden state 的"频谱多样性"打分。** 频谱丰富 = 内部计算灵活 = 模型还能探索；频谱贫乏 = 内部计算僵化 = 陷入死循环。

paper 的 Figure 3 显示：vanilla GRPO 训练 50 步内 DSD 急剧下降，紧接着 entropy 跟着崩，reward 停滞。**DSD 崩在前，token entropy 崩在后——latent 是因，token 是果。**

---

## 5. ReLaX 的 Control Loop——怎么用 DSD 引导训练

光测出来 DSD 没用，得把它塞进 GRPO 的 loss 里。ReLaX 设计了一套 control loop：

### 5.1 基础项：鼓励 DSD 涨

$$\mathcal{L}_{xp} = \log\left(\frac{1}{R}\sum_{i=1}^R \exp(-\text{DSD}(x^i))\right)$$

log-mean-exp 是 smooth 版的 max，gradient 稳定。负号表示最小化 loss 等价于最大化 DSD。

**但这样有坑**：如果对所有 trajectory 都鼓励 DSD，model 会在 negative reward direction 上也乱探索，浪费 gradient。

### 5.2 Advantage shaping：只在有用的方向探索

$$\tilde{\mathcal{L}}_{xp} = \log\left(\frac{1}{R}\sum_{i=1}^R \exp(-\text{clip}(\hat{A}^i, 0) \cdot \text{DSD}(x^i))\right)$$

把 DSD 乘上 advantage 的正部分（$\text{clip}(\hat{A}^i, 0)$ 只保留 $\hat{A}^i > 0$ 的部分）。对 advantage 是 0 或负的 trajectory，DSD 不贡献梯度。

**直觉**：探索要在 meaningful subspace 里发生，不是瞎转。Advantage 把 exploration 锚到 reward-positive direction——"这条路是对的，那你在这条路上多产生几种思维模式；那条路是错的，别浪费力气在错路上折腾。"

### 5.3 Adaptive KL：防止过度发散

如果 DSD 涨过头，trajectory 会变得 unstable。ReLaX 只对 DSD 超过 threshold $\xi$ 的 trajectory 加 KL constraint：

$$\mathcal{J}(\theta) = \mathcal{J}_{GRPO}(\theta) + \alpha\tilde{\mathcal{L}}_{xp} + \beta\sum_{i \in \mathcal{T}} D_{KL}(\pi_\theta(o^i) \| \pi_{ref}(o^i))$$

其中 $\mathcal{T} = \{i \mid \overline{\text{DSD}}(x^i) > \xi\}$。

**双重 control**:
1. Advantage shaping 让 exploration 沿 positive direction
2. Adaptive KL 把过度 diverge 的 trajectory 拉回来

效果：conditional latent exploration + stable learning。既不让 model 陷入 rigid pattern，也不让它 sanity drift。

---

## 6. Koopman Dictionary——怎么算 Koopman Operator

实际操作还有一步：选什么样的 observable function $g$ 才能让 Koopman operator 在这个 observable space 里 linearly faithful？

ReLaX 用 ResKoopNet（作者团队前作）学习一个 neural dictionary：

$$g(x) = \sigma(Wx), \quad W \in \mathbb{R}^{d \times m}$$

- $d$: hidden state 维度（3584）
- $m$: Koopman operator 维度（实验中 $m=50$）
- $\sigma$: sigmoid（bounded，数值稳定）

Dictionary $W$ 用 initial policy 的 hidden trajectory 训练，**一次性 fit 完冻结**，整个 policy optimization 中保持 observable space 一致，让 DSD 在可比 frame 下 evolution。

冻结的原因：如果 dictionary 边训边变，DSD 也在边变，就没办法对比 trajectory 之间的 dynamics flexibility。固定 dictionary = 固定参考系。

---

## 7. 实验结果的直觉解读

### 7.1 VLM 上 +5~8 分平均提升

ReLaX-VL-7B 平均 53.2，超 VL-Rethinker-7B (52.5)。3B 模型甚至超过若干 7B baseline。

**关键洞察**：在 EMMA-Physics 这种强视觉 grounding 的 benchmark 上 +7.7 over KL-Cov。因为 MLLM 的 cross-modal alignment 发生在 latent space，token entropy 完全反映不了视觉理解的 rigidity。Token-level 方法在视觉任务上失灵，latent-level 方法才管用。

### 7.2 LLM 上 +4~6 分提升

Qwen2.5-7B-Math: ReLaX 49.1 vs FR3E 42.8 (+6.3)。AMC23 上 ReLaX 88.9 vs FR3E 67.5 (+21.4)——hard competition math 上 latent exploration 特别 beneficial。

### 7.3 训练动力学：DSD 崩在前

GRPO 50 步内 DSD 急剧下降，紧接着 entropy 跟着崩，reward 停滞。ReLaX 维持高 DSD → 维持高 entropy → reward 持续涨。**Latent dynamics collapse 是 entropy collapse 的 upstream cause。**

### 7.4 Ablation 关键发现

- $\alpha=0.1$ 最佳，$\alpha=1.0$ 过度鼓励反而变差——entropy 高不等于 performance 好，需要 "right amount"
- 去掉 advantage shaping：性能严重下降，甚至比 base 还差——**乱探索比不探索还糟**
- 去掉 adaptive KL：uniform penalty 抑制 useful exploration

### 7.5 Case study 揭示的真相

AMC23 题目，ReLaX 和 R1-zero-Div 都答对，都做 self-verification：
- **ReLaX**: 用 law of cosines 数学验证（有意义的推理）
- **R1-zero-Div**: 生成 Python code 验证——但 model 没有 code execution environment，这是 hallucination

**Token entropy 高 ≠ 推理质量好。** 单纯在 token 层面注入随机性，会让 model 表面 diverse 但 content meaningless。

DynaMath（同题视觉变体）上 ReLaX 3 个 variant 全对，KL-Cov 在 variant 2 把 prism 当 flat rectangle，variant 3 用错公式——**latent flexibility 让 model 对视觉变化更 robust。**

---

## 8. 为什么这套设计 work——intuition 总结

让我把整套逻辑串起来：

1. **诊断层面**：Token entropy 是 symptom，latent dynamics rigidification 是 root cause。Token-level 方法治标不治本。

2. **分析工具**：Koopman theory 把 nonlinear dynamics embed 到 linear observable space，让 spectral analysis 变 tractable。Eigenvalue magnitude 反映 dynamics mode 类型。

3. **量化指标**：DSD = eigenvalue magnitude 的 variance，衡量 dynamics 的"模式丰富度"。高 DSD = 多种 mode 共存 = flexible。

4. **干预策略**：把 DSD 作为 regularizer 塞进 GRPO loss。但 naive 鼓励 DSD 会让 model 在错方向乱探索，所以：
   - Advantage shaping：只在 positive reward direction 鼓励 DSD
   - Adaptive KL：对过度 diverge 的 trajectory 加约束

5. **效果**：在 text-only 和 multimodal 都显著提升。Multimodal 上优势尤其大，因为 cross-modal alignment 在 latent space 发生，token-level feedback 无法触及。

---

## 9. 几个值得琢磨的延伸

### 9.1 与 mechanistic interpretability 的 connection

Koopman operator 在找 model 的 intrinsic coordinate。这与 Anthropic 的 sparse autoencoder (SAE) 找 feature direction 异曲同工：
- SAE: unsupervised, find sparse feature
- Koopman: dynamics-aware, find linear evolution mode

两者结合可能很有趣：在 SAE feature space 上算 Koopman operator，既 sparse 又 dynamics-aware。

### 9.2 Test-time compute 的 implication

ReLaX 是 training-time intervention。但 latent dynamics flexibility 应该也影响 test-time scaling (tree search, best-of-N)。如果 model 内部 rigid，再多 sampling 也是同一 trajectory 的变体——这解释了 [Yue et al. arxiv 2504.13837](https://arxiv.org/abs/2504.13837) "RLVR 没真增加 capability 只是提升 sampling efficiency" 的 finding。

### 9.3 Koopman dictionary 冻结的局限

Dictionary 在 init 一次性 fit 然后冻结，但 policy 优化中 model 变了不少，dictionary 是否一直 valid? Paper 没讨论 re-fit 的影响。可能 model 在中后期 dynamics 结构已经 drift，frozen dictionary 不再 faithful——这是潜在的改进点。

### 9.4 Multi-layer Koopman

只用 last-layer hidden state，可能丢掉 mid-layer 的 dynamics。Multi-layer Koopman 也许更 informative，但计算开销会大。

### 9.5 DSD 与 correctness 弱相关

t-SNE 显示 sample-wise DSD 不直接预测答对——但 policy-level consistent 高 DSD 有益。说明 DSD 是 **population-level 的 exploration proxy**，不是 **instance-level 的 quality indicator**。这区分很重要：DSD 衡量的是 model 的 exploration "capacity"，不是单条 trajectory 的 quality。

---

## 10. 一句话再总结

**ReLaX 把 LRM 的 hidden state sequence 当 dynamical system，用 Koopman operator 做"频谱分析"，用 DSD 量化内部计算的灵活性，再用 advantage-gated + adaptive KL 这套 control loop 让 policy 在 meaningful subspace 持续 explore——治 latent rigidification 这个 root cause，自然解决 token entropy collapse 这个 symptom。**

本质上就是：**与其在水面搅浪花，不如让暗流保持丰富。**

---

## Reference Links

- [ReLaX GitHub](https://github.com/ZhangShimin1/ReLaX)
- [Koopman theory review - Brunton et al.](https://arxiv.org/abs/2102.12086)
- [ResDMD - Colbrook & Townsend](https://arxiv.org/abs/2109.06642)
- [ResKoopNet - Xu et al.](https://arxiv.org/abs/2501.00701)
- [KL-Cov - Cui et al.](https://arxiv.org/abs/2505.22617)
- [DAPO - Yu et al.](https://arxiv.org/abs/2503.14476)
- [FR3E - Zheng et al.](https://arxiv.org/abs/2507.07017)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)
- [VeRL framework](https://arxiv.org/abs/2409.19256)
- [Does RLVR really incentivize reasoning? - Yue et al.](https://arxiv.org/abs/2504.13837)

---

# ReLaX: 在 Latent Space 用 Koopman Operator 引导 LRM 探索

Shimin Zhang et al. (Hong Kong Polytechnic University + Shanghai AI Lab) 的这篇工作把 Koopman operator theory 从 dynamical systems 领域搬进 LLM RLVR training，思路很 elegant，但 motivation 和技术细节都比较 layered。我会从 intuition 切入，逐层把公式、架构图、实验数据表都铺开讲。

---

## 1. 大背景：RLVR 与 entropy collapse 的因果链

RLVR (Reinforcement Learning with Verifiable Rewards) 自 DeepSeek-R1 [arxiv 2501.12948](https://arxiv.org/abs/2501.12948) 爆火以来，已经成为训练 Large Reasoning Models (LRMs) 的主流范式。在 verifiable reward (数学答案对错、代码执行通过) 驱动下，policy $\pi_\theta(o|q)$ 通过 GRPO [DeepSeekMath, arxiv 2402.03300](https://arxiv.org/abs/2402.03300) 类算法收敛。

问题：RL 天然偏好 deterministic policy，sparse reward 进一步加剧 entropy collapse。Cui et al. (KL-Cov, [arxiv 2505.22617](https://arxiv.org/abs/2505.22617)) 给出经验关系式：

$$R = -a \cdot \exp(H) + b$$

- $R$: reward
- $H$: token-level policy entropy
- $a, b$: 拟合系数
- 负指数形式：entropy 高 reward 反而下降，policy 被推向低 entropy corner

paper 中 Figure 1 在 text-only LLM 和 VLM 两个 setting 都拟合出这条曲线，scatter 点 + 灰色拟合线。这就是 RLVR 的 fundamental bottleneck：**exploration–exploitation tradeoff 失衡**。

现有方法的局限（在 Supplementary Sec. 1 有 review）：
- **Entropy regularization** [R1-zero-Div, arxiv 2505.23433](https://arxiv.org/abs/2505.23433)：直接最大化 token entropy，但易引起 semantic drift
- **Reward reshaping** [DAPO, arxiv 2503.14476](https://arxiv.org/abs/2503.14476)；clip-higher 放宽 PPO clip bound
- **DCPO** [arxiv 2509.02333](https://arxiv.org/abs/2509.02333)：dynamic clipping
- **KL-Cov / Clip-Cov** [arxiv 2505.22617](https://arxiv.org/abs/2505.22617)：selectively 惩罚 action probability 与 logit variation 协方差大的 token
- **High-entropy token selection**: [Wang et al. arxiv 2506.01939](https://arxiv.org/abs/2506.01939) 只更新 top 20% forking token；**FR3E** [arxiv 2507.07017](https://arxiv.org/abs/2507.07017) 在 high-entropy token 处扩展 rollout；**CURE** [arxiv 2508.11016](https://arxiv.org/abs/2508.11016) stochastic 选择

这些方法都困在 **token space**。ReLaX 的核心 claim：**token-level entropy collapse 只是表象，latent dynamics 的 rigidification 才是病根**。尤其在 MLLM 中，cross-modal internal computation 与 unimodal text-centric output 严重错位，token-level feedback 无法反映 multimodal processing。

---

## 2. Latent dynamics 视角：为什么 hidden state 比 token 更 fundamental

模型生成 CoT 时，hidden state $x_t \in \mathbb{R}^d$ 演化遵循 stochastic nonlinear dynamical system：

$$x_t = \mathcal{F}(x_{t-1}, \omega_t), \quad \omega_t \sim \mathcal{P}_\omega$$  (Eq. 6)

- $\mathcal{F}$: 非线性动力学（一个 transformer forward pass）
- $x_t$: 第 $t$ 步 last-layer hidden state (e.g. $d=3584$ for Qwen2.5-7B)
- $\omega_t$: stochasticity injection (temperature, top-p, top-k sampling)
- $\mathcal{P}_\omega$: sampling distribution

**Intuition**: token 是从 $x_t$ 通过 LM head decode 出来的离散 projection，丢失了大量 information。$x_t$ 才是 computation 的真实载体。如果 hidden state trajectory 退化到 low-dimensional manifold 上来回振荡，再怎么在 token level 注入 entropy 都没用——stochastic input $\omega_t$ 没办法被 translate 成 diverse latent trajectory，因为 intrinsic dynamics 已经 rigid。

所以 ReLaX 的方向：**直接 measure 并 regularize hidden state dynamics 的 flexibility**。

挑战：hidden state 是 high-dim + nonlinear，怎么分析？

---

## 3. Koopman Operator Theory：非线性动力学的线性化魔法

这是 paper 最 deep 的部分。Koopman theory [Brunton et al. arxiv 2102.12086](https://arxiv.org/abs/2102.12086) 的核心 idea：

**任何 nonlinear dynamical system 都可以 embed 到 infinite-dimensional Hilbert space $\mathcal{H}$ 中，在那里 evolution 由一个 linear operator $K$ 主导。**

### 3.1 Koopman operator 定义

对 discrete-time system $x_{t+1} = f_t(x_t)$，Koopman operator $K$ 作用在 observable $g \in \mathcal{H}$ 上：

$$[Kg](x_t) := g(f_t(x_t)) = g(x_{t+1})$$  (Eq. 4)

- $K$: Koopman operator（infinite-dimensional, linear）
- $g$: observable function (Koopman dictionary)，把 state $x$ 映射到某个可观测量
- $f_t$: 原 nonlinear dynamics

**Intuition**: 不直接追踪 $x_t$ 在原空间的 nonlinear evolution，lift 到 observable space 看 $g(x_t)$ 的 linear 演化。Linear system 就能用 eigen-decomposition 分析。

### 3.2 Dynamic Mode Decomposition (DMD) [Schmid 2022](https://www.annualreviews.org/doi/10.1146/annurev-fluid-030121-115849)

DMD 是 Koopman operator 的 finite-dim data-driven approximation。给定 trajectory 的 consecutive snapshots：

- $\mathcal{V} = \{g(x_0), g(x_1), \ldots, g(x_{t-1})\}$: 前置 snapshots
- $\mathcal{V}^+ = \{g(x_1), g(x_2), \ldots, g(x_t)\}$: 后继 snapshots

least-squares 估计 $K$：

$$\mathcal{K} = \arg\min_\mathcal{K} \|\mathcal{V}^+ - \mathcal{K}\mathcal{V}\|_F^2 = \mathcal{V}^+\mathcal{V}^\dagger$$  (Eq. 5)

- $\mathcal{V}^\dagger$: Moore-Penrose pseudoinverse
- $\|\cdot\|_F$: Frobenius norm

### 3.3 DMD 的 spurious eigenvalue 问题

DMD discretization 在 complex continuous spectrum 系统上会产生 spurious eigenvalue，丢失关键 dynamical mode [Williams et al. 2015](https://link.springer.com/article/10.1007/s00332-015-9258-x)。

**ResDMD** [Colbrook & Townsend 2024](https://arxiv.org/abs/2109.06642) 用 residual test 过滤 corrupted eigenvalue。给定 eigenpair $(\lambda, v)$，squared residual：

$$\text{res}(\lambda, v)^2 := v^*[(\mathcal{V}^+)^*\mathcal{V}^+ - \lambda(\mathcal{V}^*\mathcal{V}^+)^* - \bar{\lambda}\mathcal{V}^*\mathcal{V}^+ + |\lambda|^2\mathcal{V}^*\mathcal{V}]v$$  (Eq. 13)

- $\lambda$: eigenvalue（一般 complex）
- $\bar{\lambda}$: complex conjugate
- $v$: eigenvector
- $v^*$: conjugate transpose
- $|\lambda|^2 = \lambda\bar{\lambda}$: eigenvalue magnitude squared

Residual 大 → spurious，删掉。但 ResDMD 在 fixed dictionary 之上做事后 filtering，dictionary 本身怎么选还是 open。

---

## 4. ResKoopNet: 学一个 Neural Koopman Dictionary

**ResKoopNet** [Xu et al. arxiv 2501.00701](https://arxiv.org/abs/2501.00701) 把 dictionary 也参数化，与 residual objective 联合优化。

### 4.1 Dictionary parameterization

$$g(x) = \sigma(Wx), \quad W \in \mathbb{R}^{d \times m}$$  (Eq. 8)

- $W$: 投影矩阵，$d$ 是 hidden state dim (3584 for Qwen2.5-7B)
- $m$: Koopman operator 维度（experiments 中 $m=50$）
- $\sigma$: sigmoid activation（bounded，数值稳定）

只一层 linear + sigmoid，参数量很小，避免 overfit。Dictionary $W$ 用 initial policy 的 hidden trajectory 训练，**一次性 fit 完冻结**，后续 policy optimization 中保持 function space 一致。

### 4.2 Optimization objective

$$W = \arg\min \frac{1}{BR}\|(\mathcal{V}^+ - \mathcal{K}\mathcal{V})\Phi\|_F^2$$  (Eq. 9)

- $B$: policy training batch size
- $R$: GRPO group size（rollout 数）
- $\mathcal{K}$: 当前 dictionary 估计出的 Koopman operator
- $\Phi$: $\mathcal{K}$ 的 eigenvectors
- $\mathcal{V}, \mathcal{V}^+$: 从 hidden state sequence 构造的 snapshot matrices

**Intuition**: 让 dictionary 学到的 observable $g$ 满足 "lifted space 中 $K$ 的 spectral residual 最小"，即 Koopman operator 在这个 observable space 里 linearly faithful。

Algorithm 1 (paper 末尾) 给出完整流程：
- Step 1: vLLM 推理 generate R completions
- Step 2 (first step only): transformers 推理 collect hidden states，fit Koopman dict
- Step 3: 计算 DSD, 计算 ReLaX objective, 更新 policy

---

## 5. DSD (Dynamic Spectral Dispersion): Latent exploration 的 metric

这是 paper 的核心 metric。

$$\text{DSD}(x) = \text{Var}(|\Lambda|), \quad \text{where } K\Phi = \Phi\Lambda$$  (Eq. 7)

- $K$: approximate Koopman operator
- $\Phi$: eigenvector matrix
- $\Lambda$: diagonal eigenvalue matrix（eigenvalues 一般 complex）
- $|\Lambda|$: 取每个 eigenvalue 的 magnitude（模长）
- $\text{Var}$: 对所有 eigenvalue magnitudes 求 variance

### 5.1 为什么 variance of magnitudes？

Koopman eigenvalue 的物理含义：
- $|\lambda| < 1$: decay mode（瞬态扰动衰减）
- $|\lambda| > 1$: growth mode（不稳定方向）
- $|\lambda| \approx 1$, $\text{Im}(\lambda) \neq 0$: oscillation mode
- $\lambda \approx 1$: fixed/steady mode

如果所有 eigenvalue magnitude 都接近 1，dynamics 退化成 near-identity 的稳定振荡——rigid，exploration 没空间。如果 magnitudes 分散（有 decay 有 growth 有 oscillation），dynamics 富有表达力，stochastic input 能被 translate 成 diverse trajectory。

**DSD 高 → latent dynamics 异质 → 内部计算 flexible → 探索能力强**
**DSD 低 → latent dynamics 退化 → 内部计算 rigid → entropy collapse**

paper 在 Supplementary Fig. 7 给了 t-SNE 可视化（DeepSeek-Math-7B），显示 sample-wise DSD 与 correctness 不强相关，但 policy optimization 中 consistent 高 DSD 是 beneficial 的。

---

## 6. ReLaX Objective: 把 DSD 嵌进 GRPO

### 6.1 基础 sequence-level regularization

$$\mathcal{L}_{xp} = \log\left(\frac{1}{R}\sum_{i=1}^R \exp(-\text{DSD}(x^i))\right)$$  (Eq. 10)

- $R$: group size
- $x^i$: 第 $i$ 条 response 的 hidden state sequence
- $\log$-mean-$\exp$：smooth 版的 max，gradient 稳定
- 负号：最小化 $\mathcal{L}_{xp}$ 等价于最大化 DSD

### 6.2 Advantage shaping: 只在 useful 方向 explore

naive 鼓励 DSD 会让 model 在 negative reward trajectory 上也乱探索——浪费 gradient。ReLaX 把 DSD 乘上 truncated positive advantage：

$$\tilde{\mathcal{L}}_{xp} = \log\left(\frac{1}{R}\sum_{i=1}^R \exp(-\text{clip}(\hat{A}^i, 0) \cdot \text{DSD}(x^i))\right)$$  (Eq. 11)

- $\hat{A}^i$: GRPO advantage (Eq. 2)
- $\text{clip}(\hat{A}^i, 0)$: 只保留正 advantage 部分，负的截断为 0
- 对 advantage=0 或负的 trajectory，DSD 不贡献梯度——只在 "this trajectory 是 good direction" 时鼓励 latent flexibility

**Intuition**: exploration 要在 meaningful subspace 里发生，不是瞎转。Advantage 把 exploration 锚到 reward-positive direction。

### 6.3 Adaptive KL: 防止 over-dispersion

如果 DSD 过度膨胀，trajectory 会变得 unstable。ReLaX 只对 DSD 超过 threshold $\xi$ 的 trajectory 加 KL constraint：

$$\mathcal{J}(\theta) = \mathcal{J}_{GRPO}(\theta) + \alpha\tilde{\mathcal{L}}_{xp} + \beta\sum_{i}^{\mathcal{T}} D_{KL}(\pi_\theta(o^i) \| \pi_{ref}(o^i))$$  (Eq. 12)

- $\alpha$: latent exploration strength（experiments 中 $\alpha = 0.1$）
- $\beta$: KL coefficient（$\beta = 0.01$）
- $\mathcal{T} = \{i \mid \overline{\text{DSD}}(x^i) > \xi\}$: DSD 超 threshold 的 trajectory 子集
- $\xi = 10$ in experiments
- $\pi_{ref}$: reference policy (initial policy)
- $D_{KL}$: KL divergence

**双重 control**:
1. Advantage shaping 让 exploration 沿 positive direction
2. Adaptive KL 把过度 diverge 的 trajectory 拉回来

效果：conditional latent exploration + stable learning。

---

## 7. 实验设计

### 7.1 数据与 benchmark

**VLM training**:
- 数据集: ViRL39K [VL-Rethinker, arxiv from paper ref 38]，38,870 multimodal QA
- 7 个 multimodal benchmark: MMMU [arxiv 2311.16502](https://arxiv.org/abs/2311.16502), MM-Star, EMMA, MathVista, MathVerse, MathVision, DynaMath
- 评测: mean@1 greedy decoding

**LLM training**:
- 数据集: DAPO-Math-17K + MATH Level 3-5 subset ≈ 22K
- 6 个 text benchmark: MATH500, Minerva [arxiv 2206.14823](https://arxiv.org/abs/2206.14823), AMC22/23, AIME24/25
- MATH500/Minerva: mean@1; AMC/AIME: mean@32

### 7.2 Base model

- VLM: Qwen2.5-VL-Instruct (3B, 7B)
- LLM: Qwen2.5-3B-Base, Qwen2.5-7B-Base, Qwen2.5-7B-Math
- Supplementary 还验证 Llama3.2-3B-Instruct, Qwen3-4B-Base

### 7.3 关键 hyperparameter (Table 3)

- max response length: 3072
- temperature: 1.0, top-p: 1.0, top-k: -1
- 16 rollouts per prompt
- batch size: 512, generate size: 2048
- AdamW, lr=1e-6, grad clip=1.0
- $\alpha=0.1$, $\beta=0.01$, $\xi=10$, $m=50$

---

## 8. 主结果详解

### 8.1 VLM Table 1

| Model | MathVista | MathVerse | MathVision | DynaMath | MMMU | MMStar | EMMA | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B (base) | 68.2 | 49.2 | 25.1 | 53.2 | 54.3 | 63.9 | 21.5 | 47.9 |
| VL-Rethinker-7B | 74.9 | 54.2 | 32.3 | 55.2 | 56.7 | 64.2 | 29.7 | 52.5 |
| **ReLaX-VL-7B (Ours)** | **77.1** | **55.7** | 30.2 | **55.9** | 57.4 | **65.5** | **30.6** | **53.2** |
| Qwen2.5-VL-3B (base) | 62.3 | 33.5 | 21.2 | 40.0 | 46.3 | 55.9 | 19.2 | 39.8 |
| **ReLaX-VL-3B (Ours)** | 70.7 | 46.2 | 27.6 | 52.2 | 52.2 | 60.7 | 26.9 | 48.1 |

- ReLaX-VL-7B average 53.2，比 base +5.3，超过 VL-Rethinker-7B (52.5)
- ReLaX-VL-3B average 48.1，比 base +8.3，还超过若干 7B 级别 model (R1-VL 40.9, OpenVLThinker 45.3)
- 在 EMMA-Physics 这种强 visual grounding benchmark 上 +7.7 over KL-Cov

### 8.2 LLM Table 2

| Model | MATH500 | Minerva | AMC22 | AMC23 | AIME24 | AIME25 | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Base | 64.6 | 5.7 | 21.4 | 30.0 | 0.3 | 0.1 | 17.4 |
| +SimpleRL (GRPO) | 78.2 | 38.6 | 39.5 | 62.5 | 15.6 | 8.9 | 34.8 |
| +DAPO | 77.8 | 35.3 | - | 60.0 | 18.1 | 11.5 | - |
| +KL-Cov | 80.8 | 38.2 | - | 61.4 | 22.6 | 12.9 | - |
| +FR3E | 79.0 | 39.0 | 49.2 | 67.5 | 25.2 | 14.8 | 39.2 |
| **+ReLaX** | **82.4** | **39.1** | **65.4** | **84.1** | 19.7 | 13.8 | **43.5** |
| Qwen2.5-7B-Math-Base | 52.4 | 10.7 | 25.4 | 52.2 | 16.6 | 6.3 | 23.4 |
| +FR3E | 82.2 | 40.8 | 64.1 | 67.5 | 26.7 | 18.1 | 42.8 |
| **+ReLaX** | **85.6** | **43.4** | **71.9** | **88.9** | **36.9** | 17.3 | **49.1** |

- Qwen2.5-7B-Base: ReLaX 43.5 vs FR3E 39.2 (+4.3)
- Qwen2.5-7B-Math: ReLaX 49.1 vs FR3E 42.8 (+6.3)
- AMC23 上 ReLaX 88.9 比 FR3E 67.5 高出 21.4 分，说明 latent exploration 在 hard competition math 上特别 beneficial

### 8.3 跨 model family (Table 5)

- Llama3.2-3B-Instruct: ReLaX vs GRPO +5.3 average
- Qwen3-4B-Base: ReLaX vs GRPO +8.2，比 HICRA (paper ref 39) 也好

跨 family generalization 表明方法不是 Qwen 专有 trick。

---

## 9. 训练动力学 (Figure 3, Figure 8)

对比 ReLaX (red) vs vanilla GRPO (gray) 在 4 个指标上：
1. **Reward**: ReLaX 持续上升，GRPO 50 步后 saturate
2. **DSD**: GRPO 前 50 步急剧下降（latent dynamics collapse），ReLaX 保持高水平
3. **Entropy**: GRPO 急降，ReLaX 维持较高且 well-regulated
4. **Response length**: ReLaX 略长，但 stable

**关键 insight**: GRPO 在 50 步内 DSD 与 entropy 同时崩盘，reward 停滞。ReLaX 让 DSD 保持稳定 → entropy 维持 → reward 持续涨。证明 latent dynamics 是 entropy collapse 的 upstream cause。

---

## 10. Ablation Studies

### 10.1 系数 $\alpha$ 敏感性 (Figure 4)

- $\alpha=0$ (vanilla GRPO): baseline
- $\alpha=0.1$: best
- $\alpha=0.3$: marginal improvement over GRPO
- $\alpha=1.0$: 性能下降，over-exploration 损害 exploitation

$\alpha$ 增大时 policy entropy 单调上升，但 reward 不单调——证明 entropy 高不等于 performance 好，需要 "right amount"。

### 10.2 Advantage shaping + Adaptive KL (Figure 5)

在 Qwen2.5-7B-Math 上 ablate:
- Full ReLaX: 49.1
- 去掉 adaptive KL（uniform KL penalty）: 性能下降（过度约束 useful exploration）
- 去掉 advantage shaping: **更严重下降**（exploration 变 indiscriminate，包括 negative reward direction，甚至比 base 还差）

证明 advantage shaping 是 ReLaX 的核心 control mechanism。

### 10.3 Koopman 维度 $m$ 与 threshold $\xi$ (Figure 11)

- $m=5$: 太少 spectral mode 无法 capture intrinsic dynamics，DSD 不稳，recede 到 GRPO
- $m=10, 25, 50$: 表现接近，robust
- $\xi=\infty$（无 KL constraint）: DSD 无限制增长，~50 步后 training collapse
- $\xi=10, 25, 50$: 表现 comparable，robust

---

## 11. Computational Cost (Table 6)

| Component | GRPO | ReLaX |
|---|---|---|
| **Qwen2.5-3B-VL** | | |
| Fit Koopman dict | - | 109 s (one-time) |
| Update actor | 490 s | 578 s |
| Total/step | 1052 s | 1161 s (+10.4%) |
| **Qwen2.5-7B-Math** | | |
| Fit Koopman dict | - | 132 s (one-time) |
| Update actor | 449 s | 653 s |
| Total/step | 2030 s | 2273 s (+12.0%) |

- Dict fitting 一次性，~2 分钟
- Actor update 多约 50% 时间（DSD 计算开销）
- Total +10~12% per step——开销 acceptable

---

## 12. Token-level vs Latent-level 对比分析 (Section 4.4)

### 12.1 Multimodal generalization (Figure 6)

在 Qwen2.5-VL-3B 上对比 ReLaX、KL-Cov、Entropy Reg、vanilla GRPO:

- ReLaX 全面领先，在 EMMA-Physics 这种 visual-heavy benchmark 上 +7.7 over KL-Cov
- Entropy Reg: DSD 和 token entropy 都涨，但 semantic drift，performance 下降
- KL-Cov: token entropy 涨但 DSD 没动，在 visual-grounded task 上效果有限

**Insight**: MLLM 中 token-level feedback 与 cross-modal internal computation 错位。Token entropy 涨了不代表 latent dynamics 真的 flexible。KL-Cov 在 math-text benchmark (MathVista, MathVerse) 上还行，因为视觉 grounding 不 critical；在 EMMA 上失败因为视觉信息整合需要 latent flexibility。

### 12.2 Case study: AMC23 (Supplementary Table 7)

ReLaX 和 R1-zero-Div 都答对题，都做 self-verification：
- **ReLaX**: 用 law of cosines 数学验证
- **R1-zero-Div**: 生成 Python code 验证——但模型没有 code execution environment，这是 hallucination

token-level entropy encouragement 会让 model 表面 diverse 但 content meaningless。

### 12.3 Case study: DynaMath (Supplementary Table 8)

DynaMath 用同一题的 visual variant（改数字/视觉细节）测 robustness:

- **ReLaX-3B-VL**: 3 个 variant 都对，公式 $P=4(L+W+H)$ 一致
- **KL-Cov**: variant 2 误识 height，把 prism 当 flat rectangle；variant 3 用错公式

证明 latent-level exploration 让 model 对 visual variation 更 robust。

### 12.4 Case study: MMMU (Supplementary Table 9)

物理题（光在塑料中速度），ReLaX 正确用 Snell's Law: $1 \cdot \sin(53°) = n_2 \cdot \sin(37°)$，得 $n_2 \approx 1.33$，$v = c/n \approx 2.26 \times 10^8$ m/s，选 D。

KL-Cov 错把 $n_2 = 0.75$，velocity 计算错乱。

---

## 13. 方法论 intuition 总结

让我把 ReLaX 的设计哲学 distilled 一下：

1. **Symptom vs Root cause**: Token entropy collapse 是 symptom，latent dynamics rigidification 是 root cause。治标 vs 治本。
2. **Lift 到 linear space**: Koopman theory 把 nonlinear dynamics embed 到 linear observable space，让 spectral analysis 变 tractable。
3. **Spectral dispersion as exploration proxy**: Eigenvalue magnitude 的 variance 直接反映 dynamics mode 的 heterogeneity。
4. **Conditional exploration**: 用 advantage gating 确保只在 useful direction explore，避免 indiscriminate divergence。
5. **Elastic KL bound**: 只对 over-disperse trajectory 加 KL，让有 exploration potential 的 trajectory 自由生长。
6. **Frozen dictionary**: 一次性 fit，整个 policy optimization 中保持 observable space 一致，让 DSD 在可比 frame 下 evolution。

---

## 14. 联想与 limitations

让我做一些 paper 没明说但 relevant 的联想：

### 14.1 与 mechanistic interpretability 的 connection

Koopman operator 本质是在找 model 的 intrinsic coordinate。这与 Anthropic 的 sparse autoencoder (SAE) 找 feature direction 异曲同工。区别：
- SAE: unsupervised, find sparse feature
- Koopman: dynamics-aware, find linear evolution mode

两者结合可能很有趣：在 SAE feature space 上算 Koopman operator。

### 14.2 与 continual learning 的 connection

Koopman eigenvalue 反映 dynamics mode，如果某些 mode 是 catastrophic forgetting 的 source，DSD 可以作为 continual learning 的 monitor metric。

### 14.3 与 test-time compute 的 connection

ReLaX 在 training-time intervention。但 latent dynamics 的 flexibility 也应该影响 test-time scaling (e.g., tree search, best-of-N) 的效果。如果 model 内部 rigid，再多 sampling 也是同一 trajectory 的变体——这解释了 [Yue et al. arxiv 2504.13837](https://arxiv.org/abs/2504.13837) "RLVR 没真增加 capability 只是提升 sampling efficiency" 的 finding。

### 14.4 Multimodal 的真正瓶颈

ReLaX 在 MLLM 上的优势主要在 visual-grounded task，因为 cross-modal alignment 在 latent space 发生。这暗示 MLLM 的 next-token entropy 完全无法反映视觉理解的 rigidity——这是 MLLM RLVR 的核心 issue。

### 14.5 限制

- **Dict freeze 假设**: Koopman dictionary 在 init fit 后冻结，但 policy 优化中 model 本身变了不少，dictionary 是否一直 valid? Paper 没讨论 re-fit 的影响。
- **Last-layer only**: 只用 last-layer hidden state，可能丢掉 mid-layer 的 dynamics。Multi-layer Koopman 也许更 informative。
- **DSD 与 correctness 弱相关**: t-SNE 显示 sample-wise DSD 不直接预测答对——但 policy-level consistent 高 DSD 有益。说明 DSD 是 population-level 的 exploration proxy，不是 instance-level 的 quality indicator。
- **Spectral collapse vs entropy collapse 谁先**: Figure 3 显示两者同时 collapse，paper 没明确证明 latent 是因 token 是果。可以做 experiment：强制 token entropy 高但 DSD 低，看 reward 是否仍 saturate。

---

## 15. Web Links for Reference

- Paper: [ReLaX GitHub](https://github.com/ZhangShimin1/ReLaX)
- Koopman theory review: [Brunton et al. arxiv 2102.12086](https://arxiv.org/abs/2102.12086)
- ResDMD: [Colbrook & Townsend, arxiv 2109.06642 (CPAM 2024)](https://arxiv.org/abs/2109.06642)
- ResKoopNet: [Xu et al. arxiv 2501.00701](https://arxiv.org/abs/2501.00701)
- DMD review: [Schmid, Annual Review of Fluid Mechanics 2022](https://www.annualreviews.org/doi/10.1146/annurev-fluid-030121-115849)
- Extended DMD with dictionary learning: [Williams et al. 2015, Journal of Nonlinear Science](https://link.springer.com/article/10.1007/s00332-015-9258-x)
- GRPO/DeepSeekMath: [arxiv 2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1: [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- DAPO: [arxiv 2503.14476](https://arxiv.org/abs/2503.14476)
- KL-Cov (Cui et al.): [arxiv 2505.22617](https://arxiv.org/abs/2505.22617)
- FR3E: [arxiv 2507.07017](https://arxiv.org/abs/2507.07017)
- R1-zero-Div: [arxiv 2505.23433](https://arxiv.org/abs/2505.23433)
- VL-Rethinker: paper ref 38
- VeRL framework: [arxiv 2409.19256](https://arxiv.org/abs/2409.19256)
- KoopSTD (作者前作): [ICML 2025](https://icml.cc/virtual/2025)
- MMMU benchmark: [arxiv 2311.16502](https://arxiv.org/abs/2311.16502)
- Minerva (MATH): [arxiv 2206.14823](https://arxiv.org/abs/2206.14823)

---

## 16. 一句话 intuition

ReLaX 把 LRM 的 hidden state sequence 当成 stochastic nonlinear dynamical system，用 Koopman operator 在 linear observable space 量化 dynamics 的 spectral dispersion，把 exploration 监控从 token-level 提升到 latent-dynamics level，再用 advantage-gated log-mean-exp + adaptive KL 这套 control loop 让 policy 在 meaningful subspace 持续 explore——治 latent rigidification 这个 root cause，自然解决 token entropy collapse 这个 symptom。
