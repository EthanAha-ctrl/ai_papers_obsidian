---
source_pdf: SEVA Self-Evolving Verification Agent with.pdf
paper_sha256: 34c6bcfde86a93850ed7a9e9205e336aaa213ed7463671dfedde10dbb011fa49
processed_at: '2026-08-12T05:23:47-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SEVA

## 一句话版本

你想让 LLM 当 fact-checker，判断 "这句话能不能从那段原文里推出来"。现在的 verifier 只给你一个 "行/不行"，你不知道它为啥这么判。SEVA 让它给你结构化的东西——哪些对上了、哪步推理有道理、是哪种错误、怎么改——然后发现一个尴尬事：**用 binary reward 训这种结构化输出，RL 直接训不动**。

---

## 问题出在哪

你现在要训一个模型，输出是一坨 JSON，里面有 evidence alignment、reasoning chain、label、confidence、error type、fix suggestion 五六个东西。

你用 GRPO 训。GRPO 干的事是：同一个 prompt 采样 8 个 response，给每个打分，算 advantage = (分数 - 组平均) / 组标准差，然后往高分方向靠。

你用 binary reward：label 对就 1 分，错就 0 分。

问题来了。你的 SFT 模型 28% 的输出 JSON 格式都不对，直接 0 分。剩下 72% 里又有 35% label 错，也是 0 分。所以 8 个 response 里 5-7 个都是 0 分，剩下 1-3 个是 1 分。

组内标准差 $\sigma \approx 0.05$。advantage = $(r_i - \mu) / (\sigma + \epsilon) \approx 0$ 对所有 $i$。gradient 死了。

**你跑 350 步，模型跟 SFT 完全一样，一点没学。**

这跟 model 大小无关，跟 group size 无关。问题是 binary reward 把所有 verification quality 压成一个 bit——一个 reasoning 完美但 label 猜错的 response 跟一个输出乱码的 response 得分一样，都是 0。模型没有任何信号区分 "谁更接近对的"。

---

## 怎么修

把 reward 拆开。你有 5 个 output component，就给 5 个 reward component，每个 0 到 1 独立打分：

- $R_f$：JSON 格式对不对
- $R_a$：evidence alignment 做得好不好（每个 span 对上了吗）
- $R_c$：reasoning chain 做得好不好（每步有 judgment、有 citation、有 explanation 吗）
- $R_l$：label 对不对
- $R_d$：error type 和 fix 对不对

加权 70% process（前三个）+ 30% outcome（后两个），再加一个 calibration term 奖励 "对的时候自信、错的时候不自信"。

现在同样一个 "reasoning 完美但 label 错" 的 response，得分是 0.63 而不是 0。一个 "label 对但 reasoning 烂" 的 response 得 0.28 而不是 1.0。**你的 reward landscape 从一个悬崖变成四级台阶**，group 内终于有 spread 了，GRPO 能学了。

数学上，Proposition 2 说的就是：只要有一个 component 有 variance（比如 format 在训练早期 28% 出错，$R_f$ 就有 variance），总 reward 的 variance 就 $\geq w_f^2 \sigma_f^2 > 0$，gradient 不会死。这跟 components 之间 correlated 还是 independent 没关系——除非你 contrived 到 perfect anti-correlated，否则 cross-terms 也救不了零。

**核心 insight**：reward 的粒度必须匹配 output 的粒度。binary output 配 binary reward 没问题，multi-component output 配 binary reward 就是 advantage collapse。这不是 verification task 特有的事，任何 RL agent 输出多个 component 都继承这个 dichotomy。

---

## 训完之后发生了什么

3B 模型从 SFT 的 64.9 F1 涨到 GRPO 的 69.0 F1，追平 GPT-4o-mini 的 69.8。alignment quality 从 0.917 到 0.997，format compliance 从 72% 到 100%。

有个没 design 的 curriculum 浮现出来：前 150 步 alignment 和 format 就饱和了，后面 200 步 F1 才慢慢爬。模型先学会 "怎么写正确的 JSON 和 align span"，再学会 "怎么真的判断 claim 对不对"。pattern-level skill 比 semantic reasoning 容易，70/30 加权放大了这个 difficulty asymmetry。

这给你个 deployment hint：如果只要 format 可靠就行，step 150 就可以 early stop。

---

## Self-evolution 的意外发现

结构化输出让你能看到模型具体在哪种 error 上错。于是搞个 loop：验证 → 反思（按六类 error 统计 weak spot）→ probe（用 GPT-4o-mini 针对弱类生成 adversarial example，弱类拿 3 倍 budget）→ refine（用这些 probe 训下一轮）。

跑了四轮。期待是模型变成一个越来越强的 generalist。

实际看到的是：**每一轮产生一个 specialist，不是 generalist**。HaluEval 涨 15 pp，TruthfulQA 跌 12 pp，ClearFacts 和 FEVER 基本不动。平均 F1 在 70.5-71.4 之间晃——这哪是 improvement，这是 mass redistribution。

Round 4 用了 7 倍于 Round 2 的数据，HaluEval 只多涨 4 pp。如果是 overfitting 应该 unbounded 增长，这里是 saturation。所以这不是 overfitting，是 **data-distribution-induced specialization**：你的 probe 来自 ClearFacts-style 的 weakness profile，就把模型往那个 distribution 推，远离 TruthfulQA 的 distribution。

**这跟 Self-Refine / STaR 那套 "iterative refinement 越来越好" 的叙事直接冲突**。当 probes 来自单一分布时，specialization 是 dominant mode，不是 generalization。要恢复 generalist 可能需要 heterogeneous probe distributions——这是 future work。

这个 finding 只因为 structured output 暴露 per-category dynamics 才可见。你看 aggregate accuracy 只会觉得 "嗯，没怎么变"，其实里面在剧烈重新分配。

---

## 还有啥坑

$R_d$ 的设计有个 asymmetry：预测 Not Attributable 要给 error type + fix 两部分信号，预测 Attributable 只有一个 scalar。这让模型在不确定时倾向判 NA——false positive rate 35.9%。在 class-balanced benchmark（FEVER、TruthfulQA）上这是优势，在 positive-skewed benchmark（HaluEval）上这是劣势。作者承认这是 deployment 前要修的，方向是 label-conditional reward normalization。

六类 error taxonomy 也有点 over-fit 到 word-level substitution，会把 paraphrase 误判成 scope inflation（比如 "emissions" vs "greenhouse gas output" 被判 NA）。

---

## 你该记住啥

1. **多 component 输出 + binary reward = RL 训不动**。Proposition 1 给数学，Proposition 2 给 fix。这不是 trick，是 structural dichotomy。
2. **Reward 粒度匹配 output 粒度**。5 个 component 就 5 个 reward，独立打分加权求和。
3. **70/30 process/outcome 是个合理起点**，但要在你自己 task 上 sweep。Process 太重 label 不准，outcome 太重结构崩。
4. **Structured output 是 dual-use**：deployment 靠它 audit，training 靠它 diagnose。没有结构化输出，self-evolution 的 specialization finding 根本看不见。
5. **Iterative self-improvement 不必然产生 generalist**。single-sourced probes 下 specialization 是 dominant mode。这跟你从 STaR literature 里得到的直觉相反。

代码在 [github.com/Justin0504/Verifiable_agent](https://github.com/Justin0504/Verifiable_agent)，3B-only pipeline 28 GPU 小时能在单台多卡工作站复现。

---

# SEVA: 深度技术讲解

## 1. Paper 在 research landscape 中的定位

这篇 paper 处于三个 previously uncombined 的 research line 的交叉点：

1. **Fact attribution verification**：MiniCheck (Tang et al., 2024) [arXiv:2404.10774](https://arxiv.org/abs/2404.10774) 和 ClearCheck (Seo et al., 2025) [arXiv:2506.13342](https://arxiv.org/abs/2506.13342) 这条线用 NLI transfer 做 fact verification，accuracy 强但 output 是 opaque binary label，且 SFT-only。
2. **RL for reasoning**：GRPO (DeepSeek-Math, Shao et al., 2024) [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) 和 RL Tango (Zha et al., 2025) [arXiv:2505.15034](https://arxiv.org/abs/2505.15034) 这条线假设 single-answer output，correctness reward 就够。
3. **Process reward models**：PRM800K (Lightman et al., 2024) [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) 和 Math-Shepherd (Wang et al., 2024) 这条线 score sequential step dependencies。

SEVA 的 insight 在于：当 output 变成 multi-component structured 格式时，这三条线的假设全部 break down——binary reward 在 multi-component generation 上会触发 advantage collapse，而 sequential PRM 的 step dependency 假设也不成立（这里是 parallel components）。这是 paper 的 structural contribution。

---

## 2. Structured Verification Schema 的设计

给定 claim $c$ 和 source document $d$，SEVA 要求模型输出 $\mathbf{v} = (A, C, y, \gamma, e, s)$，其中：

- **$A$ (evidence alignment)**：list of $(c_i, d_i, \text{status}_i)$ triples，status $\in$ {match, mismatch, not found}。这迫使模型 anchor 每个 judgment 到具体 text span，而不是 form holistic impression。
- **$C$ (reasoning chain)**：step-by-step verification，每步产出一个 judgment $\in$ {supported, not supported, partially supported} 和 natural language explanation。
- **$(y, \gamma)$**：binary label $y$ 配 calibrated confidence $\gamma \in [0, 1]$。
- **$(e, s)$**：当 $y = \text{Not Attributable}$，$e$ 从六类 taxonomy（numerical exaggeration, negation flip, scope inflation, temporal shift, entity substitution, fabrication）中选一个，配 fix suggestion $s$。

这个 schema 是 dual-use 的：
- **Deployment 面向**：human operator 可以 audit alignments 和 reasoning 来判断 verdict 是否 trustworthy。
- **Training 面向**：当 agent 出错时，structured output pinpoint 哪个 evidence 被 mishandled，这为 self-evolution loop 提供了 diagnostic interface。

---

## 3. Binary Reward 的失败 —— Proposition 1 的数学

### 3.1 现象

在 GRPO group of $G = 8$ responses 下用 binary reward（label 对就 1.0，错就 0.0）：
- SFT 模型只有 72% 的输出能 parse 出 valid JSON，剩下 28% 直接得 0 分
- 在 valid responses 中，约 35% 预测错 label，也得 0 分
- 典型 group 里 5-7 个 response 得 0 分

### 3.2 GRPO advantage 的公式

GRPO 的 normalized advantage 是：

$$\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon}, \quad \mu = \frac{1}{G} \sum_j r_j, \quad \sigma = \text{std}(\{r_j\})$$

这里 $\epsilon$ 是一个 numerical stability 的小常数（通常 $10^{-4}$ 量级）。policy gradient 是：

$$\nabla_\theta J \propto \sum_i \hat{A}_i \nabla_\theta \log \pi_\theta(\mathbf{v}_i)$$

当所有 $r_j$ 几乎相等时，$\sigma \to 0$，所以 $\hat{A}_i \to 0$ for all $i$，gradient vanishes。

### 3.3 Proposition 1 的证明

设 $r_1, \dots, r_G \in \{0, 1\}$ 为 i.i.d. $\text{Bernoulli}(q)$，$q = \Pr[r_j = 1]$。无偏 sample variance estimator：

$$\hat{\sigma}^2 = \frac{1}{G-1} \sum_j (r_j - \mu)^2$$

其期望是：

$$\mathbb{E}[\sigma^2] = \frac{G}{G-1} q(1-q)$$

这个公式直接来自 Bernoulli 分布的方差 $q(1-q)$ 乘以 Bessel 校正 $G/(G-1)$。

- 当 $q \to 0^+$（几乎全部 wrong）或 $q \to 1^-$（几乎全部 correct），$q(1-q) \to 0$，所以 $\sigma \xrightarrow{a.s.} 0$
- 因此 $\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon} \xrightarrow{a.s.} 0$ for all $i$
- Policy gradient $\nabla_\theta J = \mathbb{E}[\sum_i \hat{A}_i \nabla_\theta \log \pi_\theta(\mathbf{v}_i)] \to 0$

**关键 intuition**：这个 failure 是 structural，不是 incidental。增加 group size $G$ 没用——问题是 near-uniform scores，不是 insufficient sampling。任何 multi-component output 在 binary reward 下都会有这个问题，只要 model 不能 reliably produce 所有 components simultaneously。

### 3.4 实证

在 SFT init 时 $q \approx 0.37$：
$$\mathbb{E}[\sigma^2] \leq \frac{8}{7} \cdot 0.37 \cdot 0.63 \approx 0.27, \quad \sigma \lesssim 0.5$$

到 step 350，binary reward 的 advantage spread 衰减到 $\pm 0.05$，mean 只从 0.38 爬到 0.41。**350 步 policy optimization 产生的 gain 是零**——这跟 SFT 完全一样。

---

## 4. Process Reward 的设计 —— Proposition 2 的数学

### 4.1 Reward 函数

$$R = \underbrace{w_f R_f + w_a R_a + w_c R_c}_{\text{process (70\%)}} + \underbrace{w_l R_l + w_d R_d}_{\text{outcome (30\%)}} + R_{\text{cal}}$$

其中：
- $w_f = 0.10$（format）
- $w_a = w_c = 0.30$（alignment, chain）
- $w_l = w_d = 0.15$（label, diagnosis）
- $R_{\text{cal}} = +\hat{\gamma} \cdot 0.15$ if $\hat{y} = y^*$ else $-\hat{\gamma} \cdot 0.10$

每个 $R_x \in [0, 1]$ 从 response $\mathbf{v}$ 的不同 region 独立计算。注意 $R_{\text{cal}}$ 的 asymmetry：reward calibrated correctness (0.15) 大于 penalize calibrated error (0.10)。作者明确说这是 deliberate：在 safety-critical deployment 里，overconfident wrong answer 的代价 > overconfident correct answer 的价值，所以 calibration term 偏向 reward correct calibration。

### 4.2 Per-component scoring rubrics

**$R_f$ (Format)**：$\{0, 0.2, 0.5, 1.0\}$ 取决于 JSON validity。

**$R_a$ (Alignment, per entry $a_i$)**：
$$R_a(a_i) = 0.3 \cdot \mathbb{1}[|\text{claim\_span}| > 0] + 0.3 \cdot \mathbb{1}[|\text{source\_span}| > 0 \lor \text{NOT\_FOUND}] + 0.2 \cdot \mathbb{1}[\text{status} \in \text{VALID}] + 0.1 \cdot \mathbb{1}[3 \leq |\text{claim\_span}| \leq 200] + 0.1 \cdot \mathbb{1}[3 \leq |\text{source\_span}| \leq 500]$$

Final $R_a$ = mean across entries，cap at 1.0。

**$R_c$ (Chain, per step $s_j$)**：
$$R_c(s_j) = 0.3 \cdot \mathbb{1}[\text{judgment} \in \text{VALID}] + 0.3 \cdot \mathbb{1}[|\text{explanation}| \geq 10] + 0.2 \cdot \mathbb{1}[|\text{source\_evidence}| \geq 5] + 0.2 \cdot \mathbb{1}[|\text{claim\_part}| > 0]$$

加 length bonus $\min(|C|/3, 1) \times 0.2$ 来 reward multi-step chains。

**$R_d$ (Diagnosis)**：
$$R_d = \begin{cases} 1.0 & y^* = \mathbf{A}, \text{no err.} \\ 0.3 & y^* = \mathbf{A}, \text{err. present} \\ 0.6 \cdot \mathbb{1}[e \in \mathcal{T}] + 0.4 \cdot \mathbb{1}[|s| \geq 10] & y^* = \mathbf{NA} \end{cases}$$

这里 $\mathcal{T}$ 是六类 error taxonomy。

### 4.3 Reward landscape

| Response quality | Process | Binary |
|---|---|---|
| Correct label + good reasoning | ~1.13 | 1.0 |
| Good reasoning, wrong label | ~0.63 | 0.0 |
| Correct label, poor reasoning | ~0.28 | 1.0 |
| Unparseable output | 0.0 | 0.0 |

**关键 intuition**：binary reward effectively pays for lucky guesses；process reward pays for genuine verification work。"Good reasoning but wrong label" 得 0.63 而不是 0——这个 gap 是 GRPO 能学到东西的根本原因。

### 4.4 Proposition 2 的证明

设 $\bar{R} = \sum_{k=1}^K w_k R_k$，$R_k \in [0, 1]$，$w_k > 0$。由 linear combination 的 variance identity：

$$\sigma^2(R) = \sum_{k=1}^K w_k^2 \sigma_k^2 + 2 \sum_{k < \ell} w_k w_\ell \text{Cov}(R_k, R_\ell)$$

只要 components 不是 perfect anti-correlated，cross-terms 不能 drive $\sigma^2(R)$ to zero，除非每个 $\sigma_k = 0$。

特别地，如果某个 component $k^*$ 满足 $\sigma_{k^*}^2 > 0$ 且与其他 uncorrelated：

$$\sigma^2(R) \geq w_{k^*}^2 \sigma_{k^*}^2 > 0$$

**这个 lower bound 是关键**：在 training 过程中，format errors 在 SFT 起步时约 28%，所以 $\sigma_f > 0$ 在所有观察到的 step 都成立。由 Eq. 7，$\sigma^2(R) \geq w_f^2 \sigma_f^2 = 0.01 \cdot \sigma_f^2 > 0$——GRPO gradient non-vanishing。

### 4.5 为什么 70/30

如果 outcome 占主导，model 会 learn to guess labels 并 wrap 在 incoherent reasoning 里。70% process 让 model 必须先做 substantive verification 才能 "unlock" label 这条 easy lever。这跟 Self-Refine [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) 和 STaR [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) 的"reasoning first"思想呼应，但这里的 decomposition 是 parallel 的，不是 sequential 的。

Ablation（Appendix I）显示：
- 90/10 (process-heavy): F1=67.2, Align=0.998, Format=100%
- **70/30 (ours): F1=69.0, Align=0.997, Format=100%**
- 50/50: F1=68.1, Align=0.985, Format=98%
- 30/70: F1=66.8, Align=0.945, Format=85%
- 0/100 (binary): F1<65, Align<0.92, Format~72%

70/30 是 sweet spot：结构质量近完美，label 准确率最大化。

---

## 5. Implicit Curriculum 的发现

训练动态中浮现了一个没 design 的 ordering：

- **Step ~150 前**：alignment (0.917→0.997), format (72%→100%) saturate
- **Step 150-350**：F1 继续 climb (64.9→69.0)

**Intuition**：模型先 master verification behavior（pattern-level skills：怎么写 JSON、怎么 align spans），再 master verification outcomes（semantic reasoning：判断 claim 真的 attributable 吗）。这是 70/30 加权的自然结果——process 部分权重高，先被压到饱和；outcome 部分权重低，慢慢 refined。

Math PRMs 在 sequential steps 上也观察到类似 effect；SEVA 把这个 principle 扩展到 parallel components。这对 deployment 有直接意义：safety-critical pipeline 可以 early-stop at step 150（format 已经可靠，F1 还没饱和）。

---

## 6. Self-Evolution Loop 的 Surprising Finding

### 6.1 Loop 结构

Verify→Reflect→Probe→Refine：

- **Verify**：当前 model $\pi_{k-1}$ 在 held-out set $\mathcal{D}_{\text{eval}}$ 上产出 structured predictions $\mathcal{V}_k$
- **Reflect**：按六类 taxonomy 聚合 per-category accuracy $\alpha_t$，计算 weakness weight $w_t = (1 - \alpha_t) / \sum_{t'} (1 - \alpha_{t'})$
- **Probe**：用 GPT-4o-mini 按 weakness weight 分配 budget，弱类别拿约 3× strong 的 budget。例如 entity substitution 42% accuracy 拿约 3× fabrication 78% accuracy 的 budget。
- **Refine**：用 adversarial probes + replay set 做 FT 或 LoRA

四轮配置：
- Round 1：rules 注入 prompt（no param update）
- Round 2：LoRA SFT on 1,122 adversarial probes
- Round 3：Full FT on 2,013 mixed samples
- Round 4：mega-FT on 7,787 mixed samples（4× Round 3）

### 6.2 Specialization Fingerprint（这是 paper 最 surprising 的发现）

Table 22 的绝对 F1 数据：

| Round | CF | FEVER | TQA | HE | Avg |
|---|---|---|---|---|---|
| Step150 (seed) | 65.2 | 90.7 | 68.8 | 57.1 | 70.5 |
| Round 1 (rules) | 64.5 | 90.2 | 69.9 | 57.7 | 70.6 |
| Round 2 (LoRA) | 66.5 | 92.3 | 58.6 | 68.0 | 71.4 |
| Round 3 (FT) | 65.2 | 91.9 | 55.0 | 71.4 | 70.9 |
| Round 4 (mega-FT) | 65.1 | 92.2 | 56.4 | 72.0 | 71.4 |
| Δ(Step150→R4) | -0.1 | +1.5 | **-12.4** | **+14.9** | +0.9 |

**关键观察**：
1. HaluEval 单调上升：+10.9 → +14.3 → +14.9 pp
2. TruthfulQA 单调下降：-10.2 → -13.8 → -12.4 pp
3. ClearFacts 和 FEVER 基本持平
4. **Average F1 只在 70.5-71.4 之间移动**——specialization 是 mass redistribution，不是 aggregate gain

### 6.3 为什么不是 overfitting

作者提供三条证据：

1. **单调性**：trade-off 在两个方向都 monotone（+10.9→+14.9 HaluEval，-10.2→-12.4 TruthfulQA）。non-functional loop 会 sign-flip。
2. **预算跟踪**：per-benchmark gains 跟 Probe-stage budget allocation 相关，不是 raw sample count。
3. **数据规模饱和**：Round 4 有 7× Round 2 的 data，但 HaluEval 只多 +4 pp。如果是 data-volume overfitting，应该 unbounded growth。

Round 4 持续 4× scale 还能保持 specialization 排除了 trivial overfitting，确认这是 **data-distribution-induced**：probes 来自 ClearFacts-style weakness profile，把 model 推向那些 failure modes，远离 TruthfulQA 的 distribution。

### 6.4 这跟 Self-Refine / STaR 的张力

Self-Refine [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) 和 STaR [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) 隐含假设 iterative refinement 产生 monotone-improving generalist。SEVA 的 finding 是：当 probes 来自单一 source distribution 时，**specialization 是 iterative refinement 的 dominant mode**。这个 finding 只因为 structured output 暴露了 per-category dynamics 才可见——aggregate accuracy 会隐藏这个 asymmetry。

### 6.5 Mechanism 背后的直觉

TruthfulQA 的 failure mode 主要是 qualifier-level（"significantly improves" vs "improves"），而 ClearFacts/HaluEval 的 failure mode 主要是 entity/number-level。六类 taxonomy 偏向 entity/number 风格（numerical exaggeration, entity substitution 等），adversarial probes 训练 model attend to entity/number perturbations，这让它对 qualifier 更 permissive。Case F3 给了具体例子：carrots/night vision 的 claim 在 Step150 被正确判为 NA（注意 "significantly"），Round 3 后判为 A——decision boundary 被 push 向 ClearFacts/HaluEval-style failures。

---

## 7. 实验结果

### 7.1 ClearFacts 主结果

| Model | Size | Output | F1 |
|---|---|---|---|
| Llama-3.1 (0-shot) | 8B | binary | 67.2 |
| MiniCheck | 7B | binary | 81.2 |
| ClearCheck | 8B | binary | ~84 |
| GPT-4o-mini (0-shot) | - | struct | 69.8 |
| MiniCheck-Flan-T5 | 770M | binary | 68.3 |
| SEVA-SFT | 3B | struct | 64.9 |
| **SEVA-GRPO** | **3B** | **struct** | **69.0** |
| SEVA-SFT (LoRA-128) | 7B | struct | 68.5 |

**关键对比**：3B SEVA-GRPO 匹配 GPT-4o-mini（69.0 vs 69.8 F1），同时输出 substantially richer 的 auditable output。与 MiniCheck-7B 的 gap (81.2 F1) 是真实的，但反映了 data asymmetry（5K structured vs 57K binary annotations），不是 architectural limitation。

### 7.2 跨 benchmark generalization

| Model | CF | FEVER | TQA | HE |
|---|---|---|---|---|
| GPT-4o-mini | 69.8 | 91.0 | 48.6 | 34.0 |
| SEVA-SFT (3B) | 64.9 | 76.3 | 72.1 | 42.0 |
| SEVA-GRPO (3B) | 69.0 | 84.9 | 82.7 | 39.4 |

**TruthfulQA 上 SEVA 比 GPT-4o-mini 高 34 pp（82.7 vs 48.6）**。这直接 trace 到 $R_c$ 的 per-step source-citation 要求：GPT-4o-mini 在 claims "sound right" 时 fallback 到 parametric knowledge，SEVA 被强制 ground 每 step 到 document。

HaluEval 是 exception（-2.6 vs SFT），因为 agent over-predicts "Not Attributable"。这个 bias 来自 $R_d$ 的不对称 reward surface——negative predictions 有两部分 signal（error type + fix），positive 只有一个 scalar。

### 7.3 Structural Quality

| | Align | Chain | Format |
|---|---|---|---|
| SEVA-SFT | 0.917 | 0.917 | 72% |
| SEVA-GRPO | 0.997 | 0.995 | 100% |

**这本身是 load-bearing**：unparseable response 功能上 indistinguishable from wrong response。SFT 的 28% format error 让它无法作为 dependable safety component。

---

## 8. Ablation 验证 process reward 的 necessity

| Configuration | F1 | Align | Format |
|---|---|---|---|
| SEVA-GRPO (process reward) | 69.0 | 0.997 | 100% |
| SEVA-GRPO (binary reward) | <65 | <0.92 | ~72% |
| SEVA-SFT (no RL) | 64.9 | 0.917 | 72% |

Binary reward GRPO 在 350 步后 performance 与 SFT 完全一样——**policy 从未更新**。Process reward 同时提升 alignment 到 0.997、format 到 100%、F1 到 69.0。这个 tri-directional gain 只在 gradient non-degenerate 时可能（Prop. 2）。

---

## 9. Limitations & Failure Modes

### 9.1 Negative-prediction bias

$R_d$ 给 negative predictions 两部分 signal（error type + fix），positive 只有一个 scalar。这暴露更多 "reward surface" 给 negative predictions，在 uncertainty 下 bias policy。ClearFacts 上 false positives 35.9%。

Error type 分布：
- fabrication: 36.7%（catch-all category）
- scope_inflation: 23.0%
- entity_substitution: 15.3%
- numerical_exaggeration: 11.0%
- negation_flip: 8.4%
- temporal_shift: 5.7%

**fix 方向**：label-conditional reward normalization。

### 9.2 Failure cases（Appendix M）

- **Case F1**：paraphrase 被误判为 scope inflation（"emissions" vs "greenhouse gas output"）。taxonomy 在 word-level substitution 上 over-fit。
- **Case F2**：HaluEval 上 near-paraphrase items 被判为 Attributable，应为 NA。alignment threshold 对 proper-noun substitutions 应该 tighter。
- **Case F3**：self-evolution 后 TruthfulQA 上的 qualifier scrutiny 退化。

三个 cases 共享同一 shape：reward surface、taxonomy、probe distribution 都 push 同一方向（更多 negative predictions、更多 entity-style diagnoses）。**fixes 是 coupled 的**，不是 independently composable。

---

## 10. 我对这篇 paper 的 intuition 构建

### 10.1 核心思想：reward granularity 必须匹配 output granularity

这是 paper 的 transferable principle。Binary reward 在 binary output 上工作良好（数学 reasoning、single-answer QA），因为 1 bit 的 reward signal 精确对应 1 bit 的 output。当 output 变成 $K$ 个 independent components 时，reward 也必须分解成 $K$ 个 independent components，否则 GRPO 的 advantage 信号会 collapse。

Proposition 1 和 2 一起给出了 clean 的 mathematical characterization：
- Binary: $\sigma^2 \to q(1-q)$，collapse 在 $q \to 0$ 或 $q \to 1$
- Process: $\sigma^2 \geq w_{k^*}^2 \sigma_{k^*}^2 > 0$ 只要任一 component 有 variance

这不是 verification task 特有的——任何 multi-component generation（agentic planning, tool use with structured args, multi-hop reasoning with intermediate citations）都继承同样的 dichotomy。

### 10.2 Implicit curriculum 是 emergent，不是 designed

70/30 加权产生 behavior-before-outcome 的 ordering 是 pattern-level skills（写 JSON、align spans）和 semantic reasoning 的 natural difficulty asymmetry 的结果。这跟 curriculum learning 的 explicit schedule 不同——这里 schedule 是从 reward decomposition 涌现的。

### 10.3 Self-evolution specialization 是 structural finding

最 provocative 的发现是 specialization fingerprint。这与 STaR 的 monotone improvement 假设直接冲突。Intuition 是：iterative self-improvement 不是"become better at everything"，而是"become specialist at training distribution's failure modes"。当 probes 来自单一分布时，specialization 是 dominant mode。要 recover generalist 需要 heterogeneous probe distributions——这是 paper 留给 future work 的 open question。

这个 finding 只因为 structured output 暴露 per-category dynamics 才可见。Aggregate accuracy 会显示 "average F1 within 1 pp band, looks like nothing happened"。这论证了 structured output 的 dual-use 价值：deployment 需要它做 audit，training 需要它做 diagnosis。

### 10.4 与 MARCH / Dr. Zero 的对比

MARCH (Li et al., 2026) [arXiv:2603.24579](https://arxiv.org/abs/2603.24579) 用 multi-agent reinforced self-check for hallucination，Dr. Zero (Yue et al., 2026) [arXiv:2601.07055](https://arxiv.org/abs/2601.07055) 做 self-evolving search agents without training data。这两者都假设 single-answer output，correctness reward 就够。SEVA 把 functional separation 原则从 across agents（MARCH）应用到 across loop stages，且必须解决 multi-component reward 的 structural 问题。

### 10.5 工程实践含义

对任何想用 RL 训 multi-component agent 的人：
1. 先检查 output 有几个 independent components
2. 设计 reward 时每个 component 一个 $R_k \in [0, 1]$，加权和
3. 用 Prop. 2 检查至少一个 component 在 training 各阶段有 variance
4. 70/30 process/outcome 是个合理 starting point，再做 sweep
5. Monitor advantage spread——如果 collapse 到 $\pm 0.05$ 以下，gradient 已死
6. Self-evolution 时用 per-category dynamics 而非 aggregate accuracy 做 sanity check

代码在 [github.com/Justin0504/Verifiable_agent](https://github.com/Justin0504/Verifiable_agent)。

---

## References

- GRPO: [DeepSeek-Math (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- PRM800K: [Let's Verify Step by Step (Lightman et al., 2024)](https://arxiv.org/abs/2305.20050)
- MiniCheck: [Tang et al., 2024](https://arxiv.org/abs/2404.10774)
- ClearCheck: [Seo et al., 2025](https://arxiv.org/abs/2506.13342)
- Self-Refine: [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- STaR: [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- MARCH: [arXiv:2603.24579](https://arxiv.org/abs/2603.24579)
- Dr. Zero: [arXiv:2601.07055](https://arxiv.org/abs/2601.07055)
- RL Tango: [arXiv:2505.15034](https://arxiv.org/abs/2505.15034)
- Math-Shepherd: [Wang et al., 2024](https://arxiv.org/abs/2312.08935)
- Qwen2.5: [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- veRL/HybridFlow: [arXiv:2409.19256](https://arxiv.org/abs/2409.19256)
- LoRA: [Hu et al., 2022](https://arxiv.org/abs/2106.09685)
- HaluEval: [arXiv:2305.15788](https://arxiv.org/abs/2305.15788)
- TruthfulQA: [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)
- FEVER: [arXiv:1803.05355](https://arxiv.org/abs/1803.05355)
- ANLI: [arXiv:1910.14599](https://arxiv.org/abs/1910.14599)
- AlignScore: [Zha et al., 2023](https://arxiv.org/abs/2211.09579)
- RARR: [Gao et al., 2023](https://arxiv.org/abs/2210.08726)
- FActScore: [Min et al., 2023](https://arxiv.org/abs/2305.04176)

这篇 paper 三个 transferable principle 总结如下：**reward granularity 必须匹配 output granularity；structured output 是 dual-use asset（audit + diagnosis）；iterative self-improvement 在 single-sourced probes 下 drift toward specialization 而非 generalization**。这三个 principle 应该 transfer 到任何需要 explain、justify、improve under audit 的 agent 场景。
