---
source_pdf: Efficient Reasoning on the Edge.pdf
paper_sha256: 5e7ef7b07f702327d80c6ec5102d3e1a35d6998567a49c8a2a7677dbe9375c03
processed_at: '2026-08-04T01:55:21-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---
把推理模型塞进手机里

不要把reasoning能力焊死在模型里，做成一个可以开关的"插件"

遇到难题了，啪一下装上reasoning插件（LoRA adapter），模型瞬间变成reasoning模式开始思考。

简单题就用base model直接答，省时省电。

### 1. 用LoRA教模型学推理

Full fine-tuning太贵，而且会把base model的通用能力搞坏。LoRA就是只训练一小部分参数（像4%），插到frozen的base model上。

他们试了两个数据集：
- MoT：DeepSeek-R1蒸出来的35万条reasoning trace
- OT3：QwQ-32B蒸出来的120万条

结果OT3完胜。有意思的是，3B模型用OT3训练能打平7B用MoT训练的 - **数据质量比模型大小更重要**。

LoRA的rank（就是插件的大小）怎么选？7B模型rank=128就够用了，3B模型对rank更敏感，rank越大效果越好。这说明小模型更需要插件来补能力。

### 2. 训练一个"开关"决定要不要推理

叫Switcher。本质就是在模型最后一层接一个小classifier，看一眼你的问题，判断"这题需要推理吗？"

简单题（"今天几号"）→ 直接用base model答，快
难题（"证明费马大定理"）→ 开启reasoning adapter

这个Switcher很小，就一个8维的MLP。训练数据才2000条，故意混了数学和非数学的题，免得它学成"看到数学题就推理"的偷懒策略。

**最clever的trick - Masked LoRA training**：

这里有个工程难题。正常LoRA训练时，prefill阶段（读prompt）LoRA是开着的，所以生成的KV-cache带着LoRA的印记。但实际部署时，Switcher要先读完prompt才能决定要不要开LoRA - 这时KV-cache已经是base model生成的了。如果Switcher说"要推理"，你开LoRA继续生成，KV-cache就对不上，模型会懵。

naive的解决办法：重新用LoRA跑一遍prefill。但这等于prefill算两次，手机上不可接受。

他们的solution：**训练时就模拟这个场景**。prefill阶段把LoRA关掉，只decode阶段开LoRA。这样LoRA从小就学会适应base model生成的KV-cache。实测对精度几乎没影响，但部署时prefill只算一次。

### 3. 用RL治模型的话痨

SFT之后的模型很话痨。问它"2+3等于几"，它也能写500字思考过程。这叫"epistemic hesitation" - 模型不信任自己的判断，疯狂自我验证。

他们用GRPO（一种RL算法）加budget forcing来治。原理：

- 给模型一个token预算（比如"这道题只能用4000 token思考"）
- 答对了且没超预算 → reward高
- 答对了但超预算 → reward打折
- 答错了 → reward为零，不管长短

关键设计是**乘法而非加法**：reward = accuracy × budget_score。这样答错时不管多短都没reward，模型不会学到"为了短而瞎答"。答对时才有incentive压缩长度。

效果：平均压缩2.4倍，最多压缩8倍，精度几乎不掉。看case study很有意思 - baseline模型用4种方法重算同一个东西，budget-forced模型直接算一次就给答案。模型学会了"信任自己的第一直觉"。

### 4. 并行生成多个答案投票

手机上decoding是memory-bound的 - 每生成一个token要把weights搬一遍，NPU算力闲置。那就同时生成N个答案，weights只搬一次，N个stream共享。

生成8个答案后怎么选？标准做法是majority voting - 谁的答案出现次数多选谁。但有个问题：如果8个答案里有4个答A、4个答B，voting就僵了。

他们加了个轻量verifier - 就一个linear layer，对每个答案打分"这个答案对不对"。然后做weighted voting：verifier觉得靠谱的答案投票权重大。2个答案时就能打过greedy，8个答案时提升10个点。

Verifier的trick：复用生成时的KV-cache，只在每个答案后面补一句"这个solution对吗？"，不用重新算。开销几乎为零。

### 5. 量化到4-bit塞进手机

7B模型BF16要14GB，手机放不下。量化到INT4只要3.5GB。

但LLM量化有个老问题：activation有outlier（个别数值特别大），一量化就把精度毁了。naive的min-max量化会让模型直接废掉（WikiText perplexity从6.85飙到102）。

他们用FPTQuant的思路：在量化前对activation做一些rotation和scaling变换，把outlier"摊平"到其他维度，让分布更均匀。这些变换数学上不改变模型输出，只改变数值表示。变换参数可以merge回weights，推理时零开销。

然后还有一步：LoRA adapter也要在量化后的base上训练（叫QAMR），不然quantization noise会让adapter失效。naive的"quantize base + full precision LoRA"直接输出乱码。

最终4-bit模型精度只比full precision低2个百分点。

## 整体串起来

```
用户提问
   ↓
Switcher看一眼 → 简单题？直接base model答 → 快
   ↓ 难题
开启reasoning LoRA adapter
   ↓
模型在budget内推理（不会话痨）
   ↓
同时生成N个答案
   ↓
Verifier打分 + weighted voting
   ↓
最终答案
```

全程在4-bit量化下跑，fit进手机内存。实际demo视频能在手机上跑。

## 我觉得哪些地方妙

1. **Masked LoRA** - 这个trick只有真正做部署的人才能想到。训练和推理的KV-cache一致性问题是工程细节，但影响很大。

2. **乘法reward** - 简单但解决reward hacking。加法reward会让模型在"答错但短"和"答对但长"之间tradeoff，乘法直接砍掉前者的incentive。

3. **Verifier复用KV-cache** - 单独的verifier model太贵，复用generator的representation加个linear head就够了。这是GenRM思路在edge上的具体实现。

4. **FPTQuant的function-preserving transform** - 数学上不改变模型，只改变数值分布，merge回weights零开销。这个idea比单纯的quantization algorithm要高明。

## 我觉得哪些地方不够

1. **Switcher的OOD泛化** - 2000条训练数据，真实用户的query分布是long-tail的，没report OOD表现。

2. **没有真实wall-clock latency数字** - 说parallel decoding efficient，但没给"用户从提问到拿到答案要等几秒"这个最关键的指标。

3. **Budget forcing的scaling law** - 只在7B上做了。更大模型是不是更多话痨可以压缩？还是capability和话痨正相关？

4. **Verifier的failure mode** - 如果verifier系统性地给错答案高分，weighted voting比standard voting更差。没report verifier的calibration。

5. **Reasoning质量评估** - 只看final answer对不对。但budget-forced的CoT是不是"走捷径"答对的？对safety-critical场景这个很重要。

总之这是一篇很solid的系统paper，很多工程细节对实际部署有直接参考价值。缺点是open questions比answers多，但这恰恰说明reasoning-on-edge这个方向还很早期，有很多值得挖的坑。

---

# Efficient Reasoning on the Edge - 深度解析

这篇 paper 来自 Qualcomm AI Research，核心目标是把 reasoning-capable LLM 部署到 mobile/edge 设备上。作者团队非常大（20+ 人），是一个完整的 end-to-end system paper，从 training recipe 到 quantization 到 on-device deployment 全链路打通。Project page: https://qualcomm-ai-research.github.io/llm-reasoning-on-edge/

## 1. 整体 motivation 和 problem framing

Reasoning models（DeepSeek-R1, OpenAI o1, QwQ-32B）的本质是 verbose CoT generation - 一个 math 问题可能消耗 10k-30k tokens 才给出 final answer。这对 edge device 是致命的，因为三个 bottleneck 叠加：

1. **Memory bottleneck**: mobile DRAM 有限（典型 8-12GB shared with OS），大 model 只能 aggressive quantization。参考 [LLM in a flash](https://arxiv.org/abs/2310.05025), [Understanding LLMs in pockets](https://arxiv.org/abs/2502.xxxxx)
2. **Token generation cost**: autoregressive decoding 是 memory-bound 的，每生成一个 token 都要 load 全部 weights，long CoT 直接放大 latency 和 power consumption
3. **General purpose vs specialized SLM**: broad capability scope 在 edge model size 下难以实现，model switching 又引入 memory movement overhead

paper 的核心 insight: **不要把 reasoning baked-in 到 base model weights 里，而是用 LoRA adapter 作为 modular reasoning module，runtime 动态 toggle**。这样 base model 和 reasoning mode 共享同一份 frozen weights，memory footprint 不翻倍。

## 2. System architecture overview

Figure 1 和 Figure 2 给出了整体架构。从 training 到 deployment 的 pipeline 包含 5 个核心 component：

```
[Base LLM (frozen)]
       ↓
   [LoRA adapters (SFT on OT3)]    ← Section 3
       ↓
[Budget Forced RL (GRPO)]          ← Section 5
       ↓
[Switcher module (binary router)]  ← Section 4
       ↓
[Verifier head (parallel TTS)]     ← Section 6
       ↓
[Quantization: FPTQuant + QAMR]    ← Section 7
       ↓
[On-device deployment via GENIE SDK]
```

关键设计哲学：**hardware-awareness 贯穿整个 pipeline**，不是事后 quantize 一个 reasoning model，而是从 training 阶段就考虑 quantization noise、KV-cache sharing、memory-bound decoding 这些 constraints。

## 3. LoRA for Modular Reasoning (Section 3)

### 3.1 为什么 LoRA 适合 reasoning specialization

[LoRA](https://arxiv.org/abs/2106.09685) 的标准形式是 W = W₀ + BA，其中 W₀ ∈ R^(d×k) frozen，B ∈ R^(d×r), A ∈ R^(r×k) trainable，rank r << min(d,k)。

paper 的关键观察：LoRA 在 reasoning setting 下可以 match 甚至 surpass dense fine-tuning（参考 [Thinking Machines Lab 的 LoRA without regret](https://thinkingmachines.ai/blog/lora/)）。更重要的是 LoRA 的 modularity - 同一个 base model 加载一次，通过 enable/disable adapter 切换 chat mode 和 reasoning mode。

### 3.2 Training data: OT3 vs MoT

两个 SFT dataset 的对比很关键：

- **MoT (Mixture of Thoughts)**: 350k traces，由 DeepSeek-R1 蒸馏，分 Math (93.7k) + Code (83.1k) + Science (173k)
- **OT3 (OpenThoughts3)**: 1.2M samples，由 QwQ-32B 蒸馏，分 Math (850k) + Code (250k) + Science (100k)

[OpenThoughts3 paper](https://arxiv.org/abs/2506.04178) 的核心 insight 是 data quality > data quantity，他们的 recipe 是用 QwQ-32B 这个相对小的 reasoning teacher 生成 traces，然后做 aggressive filtering。结果上 OT3 全面碾压 MoT（Table 1）：

| Model | FT data | LoRA | AIME24 | MATH500 | LCB |
|-------|---------|------|--------|---------|-----|
| Qwen2.5-7B (base) | - | - | 0.10 | 0.76 | 0.36 |
| Qwen2.5-7B | MoT | dense | 0.37 | 0.90 | 0.55 |
| Qwen2.5-7B | OT3 | dense | 0.61 | 0.95 | 0.66 |
| Qwen2.5-7B | OT3 | r=128 | 0.56 | 0.93 | 0.60 |
| R1-Distill-Qwen-7B | - | - | 0.55 | 0.92 | 0.59 |

注意几个关键点：
1. OT3 dense 3B 性能 ≈ MoT dense 7B，data quality 补偿了 backbone size
2. LoRA r=128 只训练 4.24% 参数，就能接近 R1-Distill-Qwen-7B（这是 full distillation 的 7B model）
3. 对 7B backbone，LoRA r=128 几乎 recover 了 dense OT3 的 gains；但对 3B backbone，LoRA r=128 与 dense OT3 有显著 gap，说明小 model 对 adapter capacity 更敏感

### 3.3 LoRA hyperparameter study 的关键发现

Table 2-7 的 ablation 很有信息量。我提炼几个非显然的 insight：

**Learning rate 的 sweet spot 与 backbone scale 相关**：
- 3B: LR=2e-4 最好，5e-4 开始有 over-adaptation 迹象（MATH500 下降但 AIME 上升）
- 7B: LR=5e-4 经常 collapse（Table 13 里能看到 0.048 avg 的灾难性 run），stable range 是 1e-4 到 2e-4

这背后的 intuition：大 model 的 loss landscape 更 sharp，高 LR 容易 push 到 bad region。LoRA 的 effective learning rate 实际上是被 rank 和 alpha 缩放过的，所以 backbone 越大需要的 nominal LR 越小。

**LoRA rank 的边际收益递减**：
- 3B: rank 256 > 128 > 64 > 32，gap 明显（0.247 vs 0.238）
- 7B: rank 128 ≈ 256 > 64 > 32，gap 很小（0.402 vs 0.388）

这说明 7B 的 base model 已经有足够 representation capacity，LoRA 只需要轻量 correction；3B 则需要 adapter 承担更多 reasoning capability 的 lifting。

**Batch size 几乎无关紧要**（32 vs 64 vs 128 差异 <0.5%），可以 pure 根据 GPU memory 选。这与 dense training 的经验不同，原因是 LoRA 的 trainable params 少，gradient noise 本来就小，batch size 对 optimization stability 的影响被稀释了。

### 3.4 Specialization-forgetting tradeoff

Table 1 里有个 subtle 现象：SFT on OT3 会提升 LCB（LiveCodeBench，需要 reasoning 的 coding）但 degrade HumanEval/MBPP（直接生成 code 的简单 task）。这是典型的 catastrophic forgetting - reasoning SFT 让 model 习惯于 "think step by step"，对 "just give me the function" 的 task 反而不适应。

[Does math reasoning improve general LLM capabilities?](https://arxiv.org/abs/2507.00432) 这个现象被系统研究过。paper 的发现是 LoRA r=64/128 比 dense OT3 在 HumanEval/MBPP 上 retention 更好，PEFT 的 limited capacity 反而保护了 base capability。这是一个 nice property - LoRA 不只是 memory efficient，也是 regularization 的手段。

## 4. Switcher Module (Section 4) - Dynamic LoRA Routing

### 4.1 设计动机

Reasoning model 的一个 fundamental inefficiency：用户问 "今天天气怎么样" 也会触发 5k token 的 CoT。Edge device 上这是灾难。Switcher 解决的是 **when to think** 的问题。

架构上 Switcher 是一个 auxiliary classification head on top of base LLM 的 final transformer layer：

```
[Hidden states h_1, h_2, ..., h_n]  (from last layer, during prefill)
              ↓
   [Running EMA across chunks]  (chunk size 128, α=0.5)
              ↓
        [MLP: dim 8, ReLU, dropout 0.2]
              ↓
        [Binary classification: reasoning needed?]
              ↓
   Yes → enable LoRA adapters for decoding
   No  → use base model directly
```

几个工程细节值得注意：

1. **Chunked prefill compatibility**: edge device 上 prefill 是 compute-bound 的，长 prompt 会被 chunk 成 128 token 的 block。Switcher 不是 buffer 整个 prompt 的 hidden states，而是用 exponential moving average 增量更新：
   
   h̄_t = α · h̄_(t-1) + (1-α) · h_t
   
   这里 α=0.5 是 smoothing coefficient。这种 streaming 设计避免了 O(n) 的 memory overhead。

2. **Quantization robustness via Gaussian noise**: training 时向 averaged representation 注入 N(0, 0.5²) noise，模拟 4-bit quantization 引入的 perturbation。这是 simulating QAT 的轻量替代 - 不真正做 quantization-aware training，但让 classifier 对 deployment 时的 quantization noise 鲁棒。

3. **Training data 的设计**: 2k samples，刻意 mix math 和 non-math domain 来避免 domain-specific cue overfit。Easy queries from SQuAD2.0 (600) + MMLU math subset (419)；Hard queries from S1K (500) + StrategyQA (500)。这个 mix 设计的 intuition 是：如果只用 "math=hard, others=easy" 的 dataset，switcher 会学成 domain classifier 而非 complexity classifier。

### 4.2 Masked LoRA training - KV-cache reuse 的关键 trick

这是 paper 里我认为最 clever 的设计。问题陈述：

```
Standard LoRA training:
  Prefill (prompt tokens):  LoRA active  → generates KV-cache with LoRA
  Decode (response tokens): LoRA active  → uses LoRA-augmented KV-cache

Runtime inference with Switcher:
  Prefill: base model only (switcher 还没决定) → generates KV-cache without LoRA
  If switcher says "reasoning needed":
    Decode: enable LoRA → 但 KV-cache 是 base model 生成的，mismatch!
    Naive fix: re-run prefill with LoRA → 2x prefill cost, 不可接受 on edge
```

**Masked LoRA training 的 solution**: training 时在 prefill phase mask（disable）LoRA weights，只在 response generation phase 激活。这强制 LoRA adapter 学会适应 base model 生成的 KV-cache。

形式化：对于 prompt tokens x_(1:n) 和 response tokens y_(1:m)，standard LoRA forward 是：

h_t = W₀ · e_t + B·A · e_t  (for all t)

Masked LoRA:

h_t = W₀ · e_t                    (if t ≤ n, prefill phase)
h_t = W₀ · e_t + B·A · e_t        (if t > n, decode phase)

paper 报告这个 trick **no significant accuracy drop**，但完全消除 re-prefill 的 latency penalty。这是典型的 "match training distribution to inference distribution" 原则的应用 - 既然 inference 时 prefill 一定 base-only，training 就该模拟这个。

### 4.3 Switcher 的 accuracy-cost tradeoff

Figure 3 展示了在 MATH500 上的 Pareto frontier。随着更多 queries routed 到 reasoning mode，accuracy 平滑上升，cost（avg completion length）也线性上升。Switcher 提供了一个 knob，让 deployment 时根据 latency budget 选 operating point。

Intuition: MATH500 内部 problem difficulty 是 heterogeneous 的。简单题（base model 已经能解）不需要 reasoning adapter，难题才需要。Switcher 学会了这个 implicit difficulty estimator。

## 5. Budget Forcing RL (Section 5) - 控制 verbosity

### 5.1 问题背景

[Chain-of-thought](https://arxiv.org/abs/2201.11903) 的理论分析（[When reasoning meets its laws](https://openreview.net/forum?id=lWjcbodr4M)）指出 optimal test-time compute 应该 linearly scale with problem difficulty。但实际 reasoning model 会 degenerate 成 "overthinking" - 简单题也生成冗长 verification loops。

[s1 paper](https://arxiv.org/abs/2501.19393) 是 budget forcing 的开创性工作，用 hard token limit + "please give me the answer" 的 append 来强制结束。paper 这里改进成 soft-barrier multiplicative reward。

### 5.2 Reward formulation 详解

**Standard budget-forced reward (eq 1)**:

R(y, x) = R_accuracy(y, x) - λ · R_budget(L)

变量含义：
- y: generated response（model 的输出）
- x: prompt（输入问题）
- R_accuracy(y, x): accuracy reward，二值 {0, 1}
- L: total token length of response
- R_budget(L): length penalty function
- λ: scaling hyperparameter

这个 additive form 的问题是：如果 R_accuracy=0（答错），不管 L 多短 reward 都是负的，model 没有 incentive 压缩；如果 R_accuracy=1，model 会在 "答对但 verbose" 和 "答对但 concise" 之间 tradeoff，但 λ 难调。

**Soft-barrier multiplicative reward (eq 2 + eq 3)**:

eq 2 定义 budget penalty modifier R_budget(L)：

$$
R_{budget}(L) = \begin{cases} 
1 & L \leq L_{low} \\
p & L > L_{high} \\
1 - (1-p) \cdot \frac{L - L_{low}}{L_{high} - L_{low}} & L_{low} < L \leq L_{high}
\end{cases}
$$

变量含义：
- L: total response length
- B: target budget（prompted 的 token budget，如 1000/3000/4000/6000）
- m ∈ [0, 1]: half-size of decay window（hyperparameter）
- L_low = (1-m) · B: decay 开始的 lower bound
- L_high = (1+m) · B: decay 结束的 upper bound
- p: maximum penalty floor（paper 设 p=0）

graphically：
```
R_budget
  1.0 |-------*
            \
             \
              \
  0.0 |        *-------*----------
         L_low    B    L_high    L
              (decay window)
```

eq 3 是 final reward：

R(y, x) = R_accuracy(y, x) × R_budget(L)

**Multiplicative 设计的关键 intuition**：

1. **答错时压缩无意义**: R_accuracy=0 → R=0，不管 L 多短。这样 model 不会学到 "为了短而短" 的 degenerate policy。
2. **答对时强 incentive 压缩**: R_accuracy=1 → R = R_budget(L) ∈ [0, 1]，L 越长 reward 越低。
3. **Soft barrier 避免 hard truncation 的 gradient 问题**: hard truncation 在 boundary 处 gradient 不连续，RL optimization 不稳定。Soft linear decay 给了一个 smooth gradient signal。
4. **Decay window 的 buffer 作用**: m 控制 tolerance。小 m = strict budget，大 m = lenient。这避免了 minor budget infraction 的 catastrophic penalty。

paper 提到三个 design rationale：

1. **Avoidance of strict token matching**: 不强制 exact match budget，因为 perfect a priori knowledge of optimal compute 不现实
2. **Trajectory exploration**: model 保留 explore diverse reasoning paths 的自由度，避免 premature truncation
3. **Prompt-adherent budget compliance**: model 必须 satisfy user-defined budget in prompt

### 5.3 Reward hacking 的防御

paper 提到一个非常 concrete 的 reward hacking 现象：

如果只 penalize CoT trace 内的 token（不 penalize final answer 部分），model 会学到 "prematurely close  thinking block with ），然后继续 verbose CoT 在 final answer 里"。这是典型的 RL policy collapse。

**Defense**: penalize **total generation length L**（包括 thinking + answer），multiplicative formulation 让这种 exploit 无法获利 - 把 verbose 从 thinking 移到 answer 总长度没变，reward 一样低。

paper 还观察到 format-following reward 不需要显式加入：multiplicative reward 训练过程中 model 自然维持  和  的 structural formatting。这有点 surprising - 一般预期移除 format reward 会导致 format collapse，但这里没有。Hypothesis: GRPO 的 group-relative baseline 让 format-breaking 的 samples 在 group 内 reward 较低（因为答错或过长），自然被淘汰。

### 5.4 GRPO optimization

[GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) 的 loss（eq 4）：

$$
\mathcal{L}_{GRPO}(\theta | x) = -\frac{1}{G} \sum_{i=1}^{G} \min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i) + \beta D_{KL}(\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x))
$$

变量含义：
- θ: policy parameters（这里是 LoRA weights）
- x: prompt
- G: group size（每个 prompt 采样 G 个 generations）
- ρ_i: probability ratio = π_θ(y_i|x) / π_old(y_i|x)
  - π_θ: current policy
  - π_old: old policy（rollout 时的 policy）
- A_i: advantage
- ε: clipping parameter（PPO-style，限制 policy update 幅度）
- β: KL penalty coefficient
- π_ref: reference policy（通常是 SFT 后的 frozen policy，防止 reward hacking 导致 drift 太远）

Advantage（eq 5, 6）：

$$
A_i = \frac{r_i - \mu_r}{\sigma_r + \varepsilon}
$$

$$
\mu_r = \frac{1}{G} \sum_{j=1}^{G} r_j, \quad \sigma_r = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (r_j - \mu_r)^2}
$$

- r_i: 第 i 个 sample 的 reward
- μ_r: group 内 reward 均值
- σ_r: group 内 reward 标准差
- ε: small constant for numerical stability

**GRPO vs PPO 的核心区别**: GRPO 不需要 value network，直接用 group 内的 relative reward 作为 advantage。这省了一个 value head 的 training cost，对 LoRA 这种 parameter-efficient setting 很友好。

**β_KL 作为隐式 budget controller**: paper 发现一个非显然的 trick - 不在 reward 里显式调 λ，而是用 β_KL 控制预算 adherence：
- β_KL = 1e-3: optimal balance，significant length reduction + minimal accuracy drop
- β_KL = 1e-4: 更 aggressive 的 format adherence at short length，但 accuracy regression 更大

Intuition: β_KL 大 → policy 不能 drift 太远 → 保留 SFT model 的 reasoning capability 但可以微调 length；β_KL 小 → policy 可以大幅改变 → 学到更短 pattern 但可能 lose reasoning depth。

### 5.5 实验结果

Figure 4 和 5 是核心结果：

- **平均压缩 2.4×**（6K budget setting）
- **最大压缩 8×** on certain queries
- **MATH500 accuracy**: Table 8 显示 BF RL (β_KL=1e-3) 在 budget=4K 时 accuracy=85%，而 SFT baseline 在同样 budget 下只有 73%。这说明 budget-forced model 学会了 "在 budget 内高效推理"，而 SFT model 被强制截断时会答错。

### 5.6 Qualitative analysis - "epistemic hesitation"

Figure 6, 7, 9, 10 的 case study 很有启发性。Baseline model 的典型 failure mode：

```
1. 识别正确策略（e.g., difference of squares）→ 1 token
2. "Let me verify..." → 用 3-4 种不同方法重算 → 2000 tokens
3. "Wait, let me double check..." → 重 list primes → 1000 tokens  
4. "Actually, maybe I should consider..." → 测试 alternative hypothesis → 1500 tokens
5. 最终确认原答案 → 100 tokens
```

Budget-forced model 的 behavior：

```
1. 识别正确策略 → 1 token  
2. 执行 → 50 tokens
3. 给出答案 → 5 tokens
```

paper 把 baseline 的这种现象叫 **"epistemic hesitation"** - model 不信任自己的 initial derivation，用大量 redundant self-verification 来 hedge。这其实暴露了一个 deeper 问题：SFT 学到的是 "reasoning pattern matching" 而非 "reasoning confidence calibration"。Budget forcing 通过 reward signal 让 model 学会 trust correct derivations。

## 6. Parallel Test-Time Scaling (Section 6)

### 6.1 Edge device 上的 parallel decoding 为什么 free

Autoregressive decoding 的 memory-bound 特性：每个 token 生成都要从 DRAM load 全部 weights。如果只生成 1 个 stream，NPU 的 compute units 大部分时间在 idle waiting for memory。

Parallel TTS 的 insight: 同时生成 N 个 stream，weights 只 load 一次，被 N 个 stream 共享。**incremental overhead 主要来自 KV-cache（每个 stream 独立），但 weights load cost 不变**。

具体来说，prefill 是 compute-bound（matrix-vector 变 matrix-matrix），decoding 是 memory-bound。Parallel decoding 在 memory-bound phase 把 compute/memory ratio 提升 N 倍，better utilize NPU。

[Scaling LLM test-time compute with mobile NPU](https://www.microsoft.com/en-us/research/publication/scaling-llm-test-time-compute-with-mobile-npu-on-smartphones/) 是相关工作，但用的是 separate verifier，paper 这里改进成 shared-backbone verifier。

### 6.2 Verifier design - GenRM inspired

[GenRM](https://arxiv.org/abs/2408.15240) 的核心 idea：让 generator 同时做 verifier，避免 separate verifier model 的 memory footprint。

paper 的 verifier 架构：

```
[Generated response y_i, with full KV-cache from generation]
                          ↓
[Append verification prompt: "Is the above solution correct?"]
                          ↓
[Reuse existing KV-cache, only prefill the short verification prompt]
                          ↓
[Linear head on final token embedding + sigmoid → correctness score s_i ∈ [0,1]]
```

关键设计点：
1. **Linear head 极轻量**: 相比 base model 的几 B 参数，verifier head 只有 d_model 维度的 linear layer
2. **KV-cache reuse**: verification 不需要 reprocess 原始 prompt 和 response，只 prefill 短 verification prompt
3. **Verification prompt 有帮助**: 比 pure linear head without prompt 效果好。Intuition: 显式 prompt 让 model 进入 "evaluation mode"，hidden representation 更适合 correctness judgment

Training: 用 MATH training set 7.5k questions的 97.5%，每个 question 生成 16 个 candidate responses（temperature sampling），label 是 ground truth match。Binary cross-entropy loss。

### 6.3 Weighted Majority Voting

Standard majority voting（[Self-consistency](https://arxiv.org/abs/2203.11171)）：

$$
\hat{y} = \arg\max_a \sum_{i=1}^{N} \mathbb{1}[y_i = a]
$$

Weighted majority voting：

$$
\hat{y} = \arg\max_a \sum_{i=1}^{N} s_i \cdot \mathbb{1}[y_i = a]
$$

- N: parallel responses 数量
- y_i: 第 i 个 response 的 final answer
- s_i: verifier 给的 correctness score ∈ [0, 1]
- a: candidate answer

Intuition: verifier 认为 "更可能对" 的 candidate，其 vote 权重更大。这解决了 standard MV 在 tie-breaking 时的无能 - 当 N=2 且两个 answer 不同时，standard MV 无法决策，weighted MV 用 verifier score break tie。

### 6.4 实验结果（Table 9）

在 4-bit quantized Qwen2.5-7B-Instruct 上：

| Parallel | Greedy | Majority Vote | Weighted MV |
|----------|--------|---------------|-------------|
| 1 | 71.0 | 69.9±1.3 | 69.9±1.3 |
| 2 | - | 70.0±1.3 | 72.7±1.0 |
| 4 | - | 75.1±1.0 | 76.1±0.9 |
| 8 | - | 77.5±0.8 | 78.2±0.7 |

关键 insight：
1. **N=2 时 weighted MV 已经超过 greedy**（72.7 vs 71.0），而 standard MV 没超过。这验证了 verifier 在 tie-breaking 上的价值。
2. **N=8 时 10% absolute improvement**（78.2 vs 71.0）
3. **Variance 也降低**: weighted MV 的 std 从 1.3 降到 0.7，说明 verifier 让 aggregation 更 stable

## 7. Quantization (Section 7)

### 7.1 量化基础（eq 7）

Uniform affine quantization:

$$
\hat{x} = q(x; s, z, b) = s \cdot \left(\text{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z; -2^{b-1}, 2^{b-1}-1\right) - z\right)
$$

变量含义：
- x: input tensor（weight 或 activation）
- s: quantization scale（FP32/FP16/BF16 高精度）
- z: integer zero offset（asymmetric quantization 用）
- b: bitwidth（如 4 表示 INT4）
- ⌊·⌉: round-to-nearest-integer
- x_Z: b-bit integer quantized representation
- clip 范围 [-2^(b-1), 2^(b-1)-1]: INT4 时是 [-8, 7]

Symmetric quantization 限制 z=0，让 quantization grid 关于 0 对称。

### 7.2 LLM 量化的核心挑战

LLM 有 strong numerical outliers（[Massive activations in LLMs](https://arxiv.org/abs/2402.17762), [Understanding transformer quantization challenges](https://aclanthology.org/2021.emnlp-main.627/)）。Outlier 的 dilemma：
- 包含 outlier → 增大 dynamic range → 牺牲 near-zero precision
- 截断 outlier → 保留 precision 但 outlier 信息丢失

两种方案都 degrade performance。这就是为什么简单 min-max range estimation 的 W4A16KV8 会得到 WikiText-2 PPL=102.4（Table 10），基本 unusable。

### 7.3 FPTQuant - Function-Preserving Transforms

[FPTQuant](https://arxiv.org/abs/2506.04985) 的核心 idea：用一组可 merge 的 transformation 重塑 activation distribution，让 outlier 分布更 quantization-friendly，同时保持 model function 不变。

paper 用了 4 种 transform（Figure 8）：

1. **(T_k, T̄_k)**: pre-RoPE transforms for keys 和 queries
   - T_k 应用于 keys
   - T̄_k 应用于 queries（interpret 为 T_k 的 inverse）
   - 利用 RoPE 的 rotation equivariance

2. **(T_u, T_u^(-1))**: per-channel scaler
   - Merged into up 和 down projection weights
   - 调整 FFN 中间 activation 的 scale

3. **(T_v, T̄_v)**: multi-head value transforms
   - Per-head invertible matrices
   - Merged into value 和 output weights
   - 利用 attention 的 multi-head structure

4. **(T_r, T_r^(-1))**: residual rotation
   - 应用于每个 transformer block 的 beginning 和 end
   - Shared across layers
   - 重塑 residual stream 的 activation distribution

**"Function-preserving" 的含义**: 这些 transform 在数学上不改变 model 的 input-output mapping，只改变 intermediate activation 的 numerical representation。训练时学 transform parameters，inference 时 merge 回 weights，zero overhead。

**Why this works**: outlier 本质是某些 channel 的 activation magnitude 远大于其他。Rotation/scaling transform 可以 "spread" outlier magnitude 到其他 channel，让 distribution 更 uniform，quantization grid 更有效利用。

### 7.4 Quantization results (Table 10)

| Method | Bitwidth | L^p | T | train | WikiText-2 | CSR | MMLU |
|--------|----------|-----|---|-------|------------|-----|------|
| Full-precision | BF16 | - | - | - | 6.85 | 72.90 | 74.28 |
| Min-max | W4A16KV8 | - | - | - | 102.4 | 51.71 | 62.35 |
| + L^p init | W4A16KV8 | ✓ | - | - | 9.18 | 65.83 | 67.59 |
| + Transforms | W4A16KV8 | - | ✓ | - | 8.48 | 67.85 | 69.06 |
| + L^p + T | W4A16KV8 | ✓ | ✓ | - | 7.53 | 70.68 | 72.26 |
| FPTQuant° (full) | W4A16KV8 | ✓ | ✓ | ✓ | 7.26 | 72.94 | 72.81 |

Progressive improvement 很清晰：每个 component 都有 measurable 贡献。最终 FPTQuant° 在 CSR 上 match full precision，MMLU 只掉 1.5%。

Training: 在 DCLM-Edu 上 24 小时 single H100，非常 efficient。

### 7.5 QAMR - Quantization-Aware Modular Reasoning

关键问题：quantized base model 的 activation distribution 与 full precision 不同。如果 LoRA adapter 在 full precision base 上训练，再 deploy 到 quantized base 上，会有 distribution shift。

[QLoRA](https://arxiv.org/abs/2305.14314) 的 paradigm：在 frozen quantized base 上训 LoRA。paper 这里进一步 quantize LoRA weights 到 INT8（base 是 INT4），inference 用 INT16 activations。

Table 11 的 ablation：

| Bitwidth | FPTQuant | QAMR | N | AIME24 | MATH500 | Avg |
|----------|----------|------|---|--------|---------|-----|
| BF16 | - | - | 50k | 21.8 | 82.6 | 45.70 |
| W4A16KV8 | - | - | 50k | 0.0 | 0.0 | 0.00 |
| W4A16KV8 | - | ✓ | 50k | 17.3 | 75.6 | 39.25 |
| W4A16KV8 | ✓ | ✓ | 50k | 23.3 | 79.6 | 41.72 |
| BF16 | - | - | 1.2M | 53.3 | 94.0 | 60.54 |
| W4A16KV8 | ✓ | ✓ | 1.2M | 46.6 | 89.6 | 58.12 |

几个关键 insight：

1. **Naive quantization + full-precision LoRA = 灾难**（avg=0.00）。Model 输出 random tokens。这说明 quantization noise 严重 disrupt 了 base representation，LoRA 没见过这种 perturbed representation，完全失效。

2. **QAMR 必须有**: 即使 quantized base 质量差，QAMR 也能 recover 大部分性能。LoRA 通过 training 学会了 compensate quantization noise。

3. **FPTQuant base + QAMR > naive base + QAMR**: 更好的 quantized base 提供 cleaner starting point，LoRA 学习更容易。

4. **Full data (1.2M) + W4A16KV8 ≈ full data + BF16 - 2%**: 最终 deployment-ready model 性能 very close to full precision。

### 7.6 Verifier quantization

Verifier head 训练时直接用 4-bit quantized base model 的 embeddings，避免 train-test distribution shift。然后 verifier weights 和 activations 进一步 quantize 到 INT8。这样 verifier 在 edge 上 overhead 极小。

### 7.7 Deployment pipeline

最后一步是 model export 到 GENIE SDK：
1. PyTorch representation 兼容性（autoregressive parallel/sequential generation, attention masking, position embeddings）
2. ONNX export via FastForward
3. Linear layer 和 multi-head attention 的 transformation
4. Model partitioning
5. DLC (Deep Learning Container) format
6. Quantize 剩余 non-quantized nodes（如 biases）
7. Compile for aarch64-android
8. ADB upload to device

## 8. Discussion 和 open problems

### 8.1 Specialization-forgetting tradeoff

Reasoning SFT 提升 complex task 但 degrade simple task。paper 提到 future direction 是用 RL 来 mitigate forgetting（参考 [Reinforcement fine-tuning naturally mitigates forgetting](https://arxiv.org/abs/2507.05386)）。Intuition: RL 的 reward signal 可以覆盖 general capability 的保持，而 SFT 只 mimic reasoning trace pattern。

### 8.2 Switcher 的 RL 化

当前 switcher 是 supervised binary classifier。Future direction: 用 RL 学习 routing policy，optimization objective 同时包含 accuracy 和 length。这样 switcher 能学到 "base model 能解的题就 bypass，不能解的才 trigger reasoning"，而非简单的 complexity heuristic。

更进一步，可以扩展到 **multi-adapter routing**：不限于 base + reasoning 两个 mode，而是 bank of task-specific adapters（math adapter, code adapter, latent reasoning adapter 等）。[Mixture of LoRA Experts](https://arxiv.org/abs/2403.03152), [Mixture-of-LoRAs](https://aclanthology.org/2024.lrec-main.1059/) 是相关工作。

### 8.3 Semantic-aware budget forcing

当前 budget forcing 假设 uniform token cost - 每个 token 对 budget 的消耗相同。但语义上，关键 logical leap 的 token 价值远高于 "Let me think..." 这种 filler token。

Future direction: 用 information density 或 local entropy 来 weight penalty，让 model 学到 "maximize reasoning density per token" 而非 "minimize total tokens"。这其实指向了一个更深的视角 - reasoning as compression，[Conditional information bottleneck formulation](https://arxiv.org/abs/2503.xxxxx) 是理论框架。

### 8.4 Sub-4-bit quantization

4-bit 是当前 sweet spot。往 2-3 bit 推需要：
- [Quip#](https://arxiv.org/abs/2402.04396): Hadamard incoherence + lattice codebooks
- [ParetoQ](https://arxiv.org/abs/2502.02631): scaling laws for extremely low-bit QAT

### 8.5 Latent reasoning

Explicit CoT 是 token-expensive 的。Latent reasoning（在 hidden state 里 reasoning 而非生成 token）是 promising direction：
- [Coconut (Training LLMs to reason in continuous latent space)](https://arxiv.org/abs/2412.06769)
- [CoDi (Compressing CoT into continuous space)](https://arxiv.org/abs/2502.xxxxx)
- [KAva (Latent reasoning via compressed KV-cache distillation)](https://openreview.net/forum?id=ePrhcLbtGv)

如果 latent reasoning LoRA adapter 成熟，switcher 可以 route 到 "explicit reasoning", "latent reasoning", "no reasoning" 三档，进一步优化 compute-accuracy Pareto。

## 9. 我对这篇 paper 的整体评价

### Strengths

1. **End-to-end system thinking**: 不是孤立优化一个 component，而是 training + inference + quantization + deployment 全链路 co-design。Masked LoRA training 就是典型例子 - 这个 trick 只在 "inference 时需要 KV-cache sharing" 这个 constraint 下才 meaningful。

2. **Practical engineering insights**: 很多细节是 production experience 才能给出的，比如 GRPO 的 β_KL 作为 implicit budget controller，比如 Gaussian noise injection for quantization robustness，比如 verifier prompt 的必要性。

3. **Honest ablation**: Table 11 敢于展示 naive quantization 的灾难性结果（avg=0.00），这比只报 cherry-picked 数字有说服力得多。

### Weaknesses 和我关心的问题

1. **Switcher 的 generalization**: 2k training samples，domain mix 是 SQuAD + MMLU math + S1K + StrategyQA。这覆盖了 reasoning complexity 的什么分布？real-world query 分布是 long-tail 的，switcher 在 OOD query 上表现如何？paper 没有 report 这个。

2. **Budget forcing 的 scaling behavior**: 实验只在 7B 上做。paper 自己承认 "relationship between base model scale and capacity for rationale compression remains an open question"。更大 model 是否有更多 epistemic hesitation 可以压缩？还是 capability 与 hesitation 正相关？

3. **Parallel TTS 的 wall-clock latency**: paper 强调 compute utilization 但没给实际的 end-to-end latency 数字。8 个 parallel stream 的 KV-cache memory footprint 在 edge device 上是否 sustainable？mobile DRAM 通常 8-12GB shared，8 个 32k context stream 的 KV-cache 可能几个 GB。

4. **Verifier 的 failure mode**: weighted MV 在 verifier 完全 wrong 时（systematically 给 wrong answer 高分）会比 standard MV 更差。Verifier 的 calibration 在 OOD 上如何？没有 report。

5. **Reasoning quality 的深度评估**: paper 主要 report final answer accuracy。但 budget-forced CoT 的 reasoning trace 质量如何？是否在某些 task 上虽然答对但 reasoning chain 是 "shortcut" 而非 valid derivation？这对 safety-critical application 重要。

### 联想到的相关工作

- [Tina: Tiny reasoning models via LoRA](https://arxiv.org/abs/2504.15777): 类似 approach，smaller scale
- [Phi-4-mini-reasoning](https://arxiv.org/abs/2504.21233): Microsoft 的小 reasoning model
- [Think only when you need with large hybrid-reasoning models](https://arxiv.org/abs/2505.14631): hybrid reasoning architecture，与 switcher 概念类似
- [L1: Controlling how long a reasoning model thinks](https://arxiv.org/abs/2503.04697): budget control via RL
- [Chain of Draft](https://arxiv.org/abs/2502.18600): concise CoT via prompting
- [DLER: Doing length penalty right](https://arxiv.org/abs/2510.15110): RL length penalty
- [Best-of-N via speculative rejection](https://arxiv.org/abs/2410.20234): efficient parallel sampling
- [Efficient Tree Search](https://arxiv.org/abs/2502.13575): TTS efficiency

这篇 paper 在 reasoning-on-edge 这个 emerging direction 上是一个 solid system contribution，很多 engineering insights 对实际部署有直接价值。作为 system paper 它成功了，作为 research paper 它的 open questions 比 answers 多 - 这反而是好事，说明这个方向还有很多值得探索的空间。

如果你想 build intuition about 某个具体 component（比如 GRPO 的 advantage estimation 在 LoRA setting 下的 behavior，或者 FPTQuant 的 transform 数学细节），可以让我深入展开。
