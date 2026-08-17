---
source_pdf: InstructGPT.pdf
paper_sha256: c1984bb50a5b90fddb895fdc3a0f72e5bc977148c9f63ef6040cbe7a3e1f0d98
processed_at: '2026-08-05T09:59:18-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# InstructGPT 人话版

## 一句话版本

GPT-3 跑歪了，不是因为它不够聪明，是因为它学的目标是错的。OpenAI 花了 GPT-3 预训练 1.6% 的算力，请 40 个人给模型"补课"，结果 1.3B 的小模型把 175B 的 GPT-3 吊起来打。这告诉我们：**对齐比堆参数便宜得多**。

---

## 1. 问题出在哪

GPT-3 的训练目标特别 dumb — "给我一串 web 文本，我预测下一个 token"。这个 objective 训出来的模型，能力是真的有，但它根本不知道你想让它干啥。

你问它 "Write a story about a wise frog"，它可能接着写 "Write a story about a sad dog" — 因为在它训练的 web data 里，"Write a story about X" 后面经常跟着另一个 "Write a story about Y"。它压根没把这句当指令，就是当文本续写。

这就是 paper Section 1 说的 **misalignment** — 你想要的目标 ("follow my instruction") 和模型学的目标 ("predict next token on internet") 根本不是一个东西。Scale 越大，predict next token 越准，但 follow instruction 这个能力并没有被 optimize。

Karpathy 直觉：这就像一个学生读了整个 Wikipedia，倒背如流，但你让他写篇作文他不会 — 他从没被告诉过"写作文"是个啥任务。

---

## 2. 三步走，每步在干嘛

### Step 1: SFT — "老师示范一遍"

找 40 个 contractor，给 prompt，让他们写出**理想答案**。比如 prompt 是 "Summarize this article"，labeler 就真的写一个 summary 出来。然后拿这 ~13k 条 (prompt, demo) pair 对 GPT-3 做 supervised fine-tuning。

这一步的目的是让模型从 "续写模式" 切换到 "听指令模式"。SFT 之后，模型至少知道 "哦，别人问我问题我应该回答，而不是接着编问题"。

**一个反直觉点**：SFT 训 1 epoch 后 validation loss 就开始 overfit 了，但如果你训 16 epochs，human preference 反而更好。Loss 这个 proxy 在这里彻底失效 — 模型在 validation 上 token-level 预测变差了，但 overall behavior 变好了。Karpathy 经常 flag 这件事：**proxy loss 和真实 capability 之间隔了八层**。

### Step 2: Reward Model — "教一个裁判"

SFT 出来的模型虽然能听指令，但它不知道**什么样的回答是好的**。比如同一个 prompt，模型可能写出 4-9 个不同 response，有的好有的烂。怎么让模型知道哪个好？

找 labeler 把这 K 个 response **排序** (rank 1 到 K)。然后用这些 rank 训一个 reward model — 输入 (prompt, response)，输出一个 scalar 分数。这个 RM 就是 "人工偏好的压缩版"。

RM 的 loss (paper Eq. 1)：

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}\left[\log\sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]
$$

人话翻译：取两两一对 (winner $y_w$, loser $y_l$)，让 RM 给 winner 打的分比给 loser 打的分高。sigmoid 把差值压成概率，log 后做 maximum likelihood。除以 $\binom{K}{2}$ 是因为一个 prompt 下有 $\binom{K}{2}$ 个 pair，做个 normalization。

**一个关键工程 trick**：同一个 prompt 下的 K 个 response 是高度相关的 (共享 prompt)，如果一对一对训练 RM 会严重 overfit。OpenAI 的做法是把同一个 prompt 的所有 $\binom{K}{2}$ 个 pair 放进同一个 batch element — 一次 forward 把 K 个 response 都 encode 了，既省 compute 又避免 overfit。这其实是 listwise ranking loss，和 LambdaMART 一个套路。

RM 只训 1 epoch，多了就 overfit 到 labeler 的 idiosyncrasy 上。RM 是 6B 不是 175B — 因为 175B RM 训练不稳定，且后面 PPO 要把 RM 当 value function 用，175B 太贵。**所有 size 的 policy 共用同一个 6B RM**，这是工程上的 sweet spot。

### Step 3: PPO — "让模型自己练，裁判打分"

这一步最玄。把 SFT model 拷贝一份当起点 (RL policy)，让它在 prompt 上自己生成 response，然后 RM 打分，用 PPO 算法 update policy 让它下次拿高分。

但纯 PPO 会出问题 — RM 在训练分布外是 unreliable 的，policy 会找到 RM 的 "漏洞" (adversarial example)，生成一些 RM 给高分但人类看了想骂街的内容。这就是 Goodhart's Law — reward 成了优化目标后就不再是好指标。

所以加两个安全垫：

**KL penalty**：policy 不能漂离 SFT model 太远。每个 token 都算 $\log(\pi^{RL}/\pi^{SFT})$，policy 越偏离 SFT 越被惩罚。这相当于一个 anchor，告诉模型 "你可以变好，但别变怪"。系数 β=0.02。

**Pretraining mix (PPO-ptx)**：混入预训练数据的 next-token prediction gradient，让模型别忘本。系数 γ=27.8。

最终 objective (paper Eq. 2)：

$$
\text{obj}(\phi) = \mathbb{E}[r_\theta(x,y) - \beta \log\frac{\pi^{RL}(y|x)}{\pi^{SFT}(y|x)}] + \gamma \mathbb{E}_{x \sim D_{pretrain}}[\log \pi^{RL}(x)]
$$

人话：第一项 "拿高分"，第二项 "别离 SFT 太远"，第三项 "别忘预训练"。三项的 balance 决定了 alignment 的质量。

---

## 3. PPO-ptx 为什么必要 — "alignment tax" 的故事

纯 PPO (没 ptx mix) 训完后，模型在 SQuAD / DROP / HellaSwag / 翻译这些 public NLP benchmark 上 **变差了**。这是反直觉的 — RLHF 本应让模型更好啊？

原因：RL 把 probability mass 集中到 "instruction-following manifold" 上，挤压了原本在 general NLP 任务上有用的 representation。模型学会了"听指令"，但把"做 NLP 任务"这个 capability 给忘了点。这就是 **alignment tax** — alignment 的代价。

怎么修？最直觉的想法是调大 KL coefficient β，让 policy 更紧地贴着 SFT。但 paper Figure 34 的 ablation 显示：β 从 0.02 加到 2.0 (100 倍)，DROP 和 SQuADv2 还是回不到 GPT-3 水平，而 validation reward 大幅下降。**KL 只能拉住 SFT，但 SFT 本身就已经偏了**。

真正的解法是 PPO-ptx — 把 pretraining 数据的 log-likelihood 也加进梯度。paper Figure 33 显示 γ≥20 时 SQuADv2 和 DROP 基本恢复，γ=27.8 是 1.3B / 6B / 175B 三个 size 都能用的 sweet spot。

**直觉**：KL anchor 是 "别离 SFT 太远"，pretraining mix 是 "别离 GPT-3 太远"。SFT 已经偏了，所以 KL 不够，得回到更原始的 anchor — 就是预训练分布。

这个 ablation 是 paper 里最 important 的工程 finding 之一，后续 DPO / GRPO 的工作都在反复印证这一点：**RL 训练必须有一个 "原始能力" 的 anchor，否则 alignment 会吃掉 capability**。

---

## 4. 数据这件事，比你想的更复杂

### Prompt 分布 (Table 1)

API 用户提交的 prompt，labeler 分类：
- Generation 45.6% (写故事、写邮件、写 rap)
- Open QA 12.4% (谁建了自由女神像)
- Brainstorming 11.2% (给我 5 个重拾职业热情的 idea)
- Chat 8.4%
- Rewrite 6.6%
- Summarization 4.2%
- Classification 3.5%
- Closed QA 2.6%
- Extract 1.9%

**关键发现**：classification + QA 只占 ~18%，剩下 57% 是 generation + brainstorming。但 FLAN / T0 这些 public instruction-tuning dataset 主要就是 classification + QA — 所以它们在 API 分布上完全打不过 InstructGPT (winrate 78% / 79%)。

**Paper 的潜台词**：学术界精心设计的 NLP benchmark 根本不 reflective of 真实用户怎么用 LM。这件事在 ChatGPT 之后被彻底证实 — 真实使用就是开放生成，不是 SQuAD 那种阅读理解。

### Labeler 不是随便找的

40 个 contractor，通过 4 维筛选：
1. 在 sensitive content 标注上和 researcher agreement ≥ 75%
2. 在 ranking 任务上和 researcher agreement ≥ 75%
3. 写 sensitive prompt 的 demo，Likert 评分 ≥ 6/7
4. 自评能识别哪些 cultural group 的 sensitive speech

Inter-labeler agreement ~73%。**这意味着 human preference 本身有 ~25% 是 irreducible disagreement** — RM 训到 73% accuracy 已经接近天花板。所以 paper Section 5.2 老实承认：我们 align 的是这 40 个人 + OpenAI researcher 的偏好，不是什么抽象的 "human values"。

---

## 5. 核心结果，几个数字

| 对比 | 结果 |
|---|---|
| 1.3B InstructGPT vs 175B GPT-3 | 1.3B 胜 (~60% winrate) |
| 175B InstructGPT vs 175B GPT-3 | 85% winrate |
| 175B InstructGPT vs 175B GPT-3 (few-shot) | 71% winrate |
| TruthfulQA truthful+informative | InstructGPT 是 GPT-3 的 ~2× |
| Closed-domain hallucination | 21% (PPO) vs 41% (GPT-3) |
| Toxicity (respectful prompt) | InstructGPT 比 GPT-3 低 25% |
| Toxicity (biased prompt) | InstructGPT 比 GPT-3 **高很多** |
| vs FLAN | 78% winrate |
| vs T0 | 79% winrate |

最后那个 toxicity 结果很关键 — InstructGPT 更听话，让它 toxic 它就更 toxic。这是 paper Section 5.3 自己 flag 的 limitation：**helpfulness 和 harmlessness 在 user explicitly 要 harmful 内容时直接冲突**，而 InstructGPT 把 helpfulness 放第一。

---

## 6. 成本这件事

| 项目 | PF-days |
|---|---|
| GPT-3 pretraining | 3,640 |
| 175B SFT | 4.9 |
| 175B PPO-ptx | 60 |
| **总 alignment 成本** | **~65 (1.8% of pretraining)** |

用 1.8% 的算力，让 1.3B 模型打败 175B。这是 paper 的 punchline — **alignment 的 ROI 是 scaling 的 100×**。

Karpathy 视角：你 2023 年 Stanford talk 里画的那个三步图，说 "SFT is imitation, RLHF is optimization"，本质就是 — SFT 是 maximum likelihood (复制 demo)，RLHF 是 reward maximization (完全不同的 objective)。InstructGPT 第一次把这个套路在百亿参数规模上跑通了。

---

## 7. 一些没明说但能读出来的事

**RM 是 OOD 不可靠的**。paper 用 6B RM 同时给 1.3B / 6B / 175B policy 打分，表面是 compute 考虑，深层是 — RM 在 policy 探索到的 OOD 区域不可靠，训更大 RM 收益递减。Gao et al. 2022 后来量化了这点：RM quality 和 KL distance 是 log-linear 关系。

**Public NLP benchmark 已经不是 LLM 的 North Star**。paper Section 4.1 的 FLAN/T0 对比是个 manifesto — 在真实 API 分布上，public instruction-tuning data 几乎无效。整个 field 从此转向 "real-world usage alignment"。

**Labeler 是 bottleneck**。40 个人，~33k RM comparison，这规模撑不起 ChatGPT 量级的需求。Anthropic 后来搞 Constitutional AI 就是用 AI feedback 替代部分 human feedback，直接回应这个 bottleneck。

**Held-out labeler 只掉 3 个点**。RM 在训练 labeler 上 72.4% accuracy，held-out labeler 上 69.6% — 这说明 RM 学到的不是具体某几个人的 idiosyncrasy，而是某种 group-level 的偏好函数。这是 paper 一个 understated 的 generalization claim。

---

## 8. 最简单的 mental model

把整个 pipeline 想成：

1. **SFT**：教一个读过整个 internet 的小孩，"别人问你问题，你要回答，不要反问"
2. **RM**：请几个老师，给小孩的答卷打分，训一个自动打分器
3. **PPO**：小孩反复答卷，打分器打分，小孩根据分数调整 — 但有两个安全带：不能偏离 SFT 太远 (KL)，不能忘掉原本学的知识 (ptx)

三步走完，一个 1.3B 的小模型就把 175B 的 GPT-3 干翻了。参数没变，objective 变了。这就是 InstructGPT。

---

## 9. 对后续的影响

这篇 paper 是 ChatGPT 的直接技术底座。ChatGPT 2022.11 发布时用的就是 InstructGPT pipeline 的升级版 (更大数据、对话式、更精细 labeler)。

之后的 DPO (2023.5) 把 RM+PPO 两步合成一步，简化了 pipeline，但没有 KL anchor 的 natural mechanism 容易 over-optimize。Llama-2 的 RLHF 部分公开了 PPO 细节。DeepSeek-R1 用 GRPO 把 reward 从 "human preference" 换成 "verifiable correctness"，但 PPO-ptx 的 "保留原始能力" 思路一直在被继承。

所有 2022 之后的 LLM post-training 工作，多多少少都是 InstructGPT 的变体或反叛。这篇 paper 是 LLM alignment 的开山之作，工程上粗糙，方向上对了。

Reference: https://arxiv.org/abs/2203.02155

---

# InstructGPT 深度解读 — Training language models to follow instructions with human feedback

## 1. Paper 的核心 thesis

OpenAI 在 2022 年 3 月挂出这篇 paper (arXiv:2203.02155)。核心论点非常 Karpathy-friendly：**单纯的 scaling 不是 alignment 的解药**。175B 的 GPT-3 在 follow instruction 这件事上被 1.3B 的 InstructGPT 打败，参数量差 100×。这从 engineering 直觉上非常反常 — capacity 不是问题，objective 才是问题。

GPT-3 的预训练目标是 next-token prediction on web text，这个 objective 与 "follow user intent helpfully & safely" 是 **misaligned** 的。InstructGPT 的工作就是把 objective 从 "predict next token" 切换到 "maximize human-preferred reward"，用三步 pipeline 把这件事做成了一个工业流程。

Paper 链接：https://arxiv.org/abs/2203.02155
OpenAI blog: https://openai.com/research/instruction-following
HuggingFace TRL 复现: https://huggingface.co/docs/trl/index

---

## 2. 三步 pipeline 架构图解析

```
                  Step 1: SFT                Step 2: RM                  Step 3: PPO
                  ────────                  ────────                    ────────
Prompt x ──→ Labeler demo ──→ SFT model    Prompt+K outputs ──→ RM     SFT init ──→ PPO policy
  (13k)        y_demo           π_SFT         (K∈[4,9])         r_θ       π_RL  ↑ reward from r_θ
                                                                                  ↓ KL to π_SFT
                                                                                  ↓ ptx mix to π_pretrain
```

### Step 1: Supervised Fine-Tuning (SFT)

- 数据来源：labeler 自己手写的 demo + OpenAI API Playground 上提交给早期 InstructGPT beta 的 prompt。
- 规模：约 13k training prompts (labeler 11,295 + customer 1,430)。
- 训练：16 epochs, cosine LR decay, residual dropout 0.2, Adam (β₁=0.9, β₂=0.95)。
- 关键 insight: **SFT 在 1 epoch 后 validation loss 就 overfit**，但 RM score 和 human preference 仍持续提升。这是 "loss is a proxy of a proxy" 的经典案例 — Karpathy 经常在 lecture 里强调这件事，validation loss 对 downstream behavior 是 weak predictor。

### Step 2: Reward Model (RM)

RM 是把 SFT model 的 unembedding layer 拿掉，换成一个 scalar projection head。在 paper 里只用了 6B RM (不开 175B)，理由是 175B 训练 unstable，且 175B 作为 PPO 的 value function 在 compute 上不可接受。

RM 的 loss function (paper Eq. 1):

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \, \mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log\sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]
$$

变量解释：
- $\theta$: RM 的参数
- $K$: 同一个 prompt 下 labeler 排序的 candidate outputs 数量，$K \in [4, 9]$
- $\binom{K}{2}$: 从 K 个 outputs 中取的所有 pairwise comparisons 数
- $x$: prompt
- $y_w$: pairwise comparison 中 human 偏好的那个 output (winner)
- $y_l$: pairwise comparison 中被 rejected 的 output (loser)
- $r_\theta(x, y)$: RM 输出的 scalar reward
- $\sigma(\cdot)$: sigmoid 函数，把 reward 差转化为 "winner beats loser" 的概率
- $D$: 人类 comparison 数据集

**这里有一个工程上的关键 trick**：因为同一个 prompt 下的 K 个 comparisons 是高度相关的 (共享 prompt x)，如果 shuffle 进 dataset 单条训练会 overfit。论文把同一个 prompt 的所有 $\binom{K}{2}$ 个 comparisons 当作 **一个 batch element** — 这样一次 forward pass 把 K 个 outputs 都编码了 (而不是 $\binom{K}{2}$ 次 forward)，既省 compute 又避免 overfit。这其实是 Pairwise Ranking Loss 的 "list-wise" 变种，对应 LambdaRank / RankNet 家族中的 LambdaMART 思路。

RM 训练超参：1 epoch (再多就严重 overfit), LR=9e-6, batch size=64 prompts (最多 2,304 comparisons per batch)。初始化用 6B GPT-3 在一堆 public NLP 数据集 (ARC, BoolQ, CoQA, DROP, MultiNLI, OpenBookQA, QuAC, RACE, Winogrande) 上 fine-tune 过的 checkpoint — 这是历史包袱，从 SFT 直接初始化效果类似。

RM 训练完后做一个 normalization: **让 labeler demonstrations 的平均 reward = 0**，因为 RM loss 对 shift 是 invariant 的，不做 normalization PPO 会乱跑。

### Step 3: PPO + KL penalty + pretraining mix

PPO 的 objective (paper Eq. 2):

$$
\text{objective}(\phi) = \mathbb{E}_{(x, y) \sim D_{\pi_\phi^{RL}}}\!\left[r_\theta(x, y) - \beta \log\!\frac{\pi_\phi^{RL}(y \mid x)}{\pi^{SFT}(y \mid x)}\right] + \gamma \, \mathbb{E}_{x \sim D_{\text{pretrain}}}\!\left[\log \pi_\phi^{RL}(x)\right]
$$

变量解释：
- $\phi$: 被优化的 RL policy 参数
- $\pi_\phi^{RL}$: 当前 RL policy
- $\pi^{SFT}$: Step 1 训好的 SFT model (frozen, 用作 KL anchor)
- $r_\theta(x, y)$: Step 2 训好的 RM 给出的标量 reward
- $\beta$: KL penalty 系数 (paper 里 β=0.02)，控制 RL policy 偏离 SFT 的程度
- $D_{\pi_\phi^{RL}}$: 由当前 RL policy 自己采样的 prompt-response 分布
- $\gamma$: pretraining mix 系数 (paper 里 γ=27.8)，控制预训练梯度的强度
- $D_{\text{pretrain}}$: GPT-3 的预训练数据分布
- $\log(\pi_\phi^{RL}(y|x)/\pi^{SFT}(y|x))$: per-token KL divergence，逐 token 累加

第一项是主 reward；第二项是 KL penalty，**防止 policy 漂离 SFT 太远** (RL 容易 overoptimize RM — RM 在 OOD 区域 unreliable，policy 会找到 RM 的 "reward hack" 解)；第三项是 PPO-ptx 的核心 — 把 pretraining 的 next-token log-likelihood 也加进梯度，对抗 "alignment tax"。

**alignment tax 现象**：纯 PPO (γ=0) 会让模型在 SQuADv2, DROP, HellaSwag, WMT Fr→En 上 performance 下降。这违反直觉 — RLHF 本应让模型更好。原因：RL 把 probability mass 集中到 "instruction-following" 风格的 manifold 上，挤压了原本在 NLP benchmark 上有用的 representation。PPO-ptx 通过混入 pretraining gradient 让模型 "不要忘本"，在 Figure 33 里可以看到 γ≥20 时 SQuADv2 和 DROP 基本恢复，γ=27.8 是 1.3B / 6B / 175B 三个 size 都能用的 sweet spot。

**为什么不直接调大 β (KL coefficient)？** Paper 在 Figure 34 里做了 ablation: 把 β 从 0.02 加到 2.0 (100×)，DROP 和 SQuAD 仍然回不到 GPT-3 水平，而 validation reward 大幅下降。这说明 KL penalty 只能防止 policy 漂离 SFT，但 SFT 本身就已经偏了；只有 pretraining 数据才能拉回原始能力分布。这是一个非常重要的 finding — KL anchor 的选择决定了 "alignment 的天花板"。

PPO 训练细节：
- 256k episodes，约 31k unique prompts
- batch size = 512, minibatch size = 64, 8 minibatches per batch, 1 inner epoch
- constant LR with 10-iter warmup (从 0.1× peak 起)
- EMA decay 0.992
- GAE 不 discount (γ_GAE=1)，因为 bandit environment 没 temporal structure
- PPO clip ratio = 0.2
- sampling T=1 for rollouts
- value function 用 6B (从 RM 初始化)，所有 policy size 都共用同一个 RM 和 value function

---

## 3. 数据集的细节与设计哲学

### Prompt 来源
- **Labeler-written (bootstrapping 阶段)**: 三类 — Plain (任意任务)、Few-shot (instruction + K-1 query/response pairs)、User-based (从 API waitlist 申请里抽象出的 use case)。这是为了在没有 instruction-style prompt 数据时启动整个 pipeline。
- **API Playground prompts**: 用户在 Playground 上提交给早期 InstructGPT beta 的 prompt。每次切到 InstructGPT model 时会弹窗告知数据可能被用于训练。

### Use case 分布 (Table 1)
| Use case | % |
|---|---|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |
| Closed QA | 2.6% |
| Extract | 1.9% |

关键观察：**classification + QA 只有 ~18%**，剩下 57% 是 generation + brainstorming — 这正是 FLAN/T0 这类 public NLP benchmark 不能 cover 的部分。这也是为什么 paper 里 InstructGPT 大幅领先 FLAN/T0 (winrate 78% / 79%)。Public NLP dataset 不 reflective of real usage。

### Prompt 长度统计 (Table 9)
- SFT train: mean=408 tokens, median=283
- RM train: mean=199 tokens, median=64
- PPO train: mean=166 tokens, median=62
- Test: mean=115 tokens, median=49

Test set 明显更短 — 这暗示用户实际用 InstructGPT 时倾向于短 prompt，而 labeler 写的 demo prompt 偏长。

### Dataset 规模 (Table 6)
| Dataset | Train | Valid |
|---|---|---|
| SFT | 12,725 (labeler 11,295 + customer 1,430) | 1,653 |
| RM | 33,207 (labeler 6,623 + customer 26,584) | 17,887 |
| PPO | 31,144 (全 customer) | 16,185 |

RM 数据里 customer 占绝对多数 (~80%)，因为 RM 需要真实分布的 prompt；SFT 阶段 labeler 数据占多数，因为 bootstrapping 阶段需要高质量 demo。

---

## 4. Labeler 选择 — 这部分最被低估

40 个 contractor，通过 Upwork 和 Scale AI 招募。筛选测试 4 个维度：

1. **Sensitive speech flagging agreement**: 在带 sensitive content (toxic/sexual/violent/judgmental/political) 的 prompt+completion 上，与 researcher 的 agreement ≥ 75%。
2. **Ranking agreement**: 在 API prompt + 多 model completion 的 ranking 任务上，与 researcher agreement ≥ 75%。
3. **Sensitive demonstration writing**: 给 sensitive prompt 写 demo，Likert 1-7 评分，平均分 ≥ 6/7。
4. **Self-assessed sensitivity coverage**: "你能识别哪些 topic/cultural group 的 sensitive speech?" — 用主观回答覆盖 demographic 多样性 (法律上不能按 demographic 直接筛人)。

**Inter-labeler agreement**: training labelers 之间 72.6 ± 1.5%，held-out labelers 77.3 ± 1.3% — 与 Stiennon et al. (2020) summarization 的 researcher-researcher 73±4% 相当。这个数字告诉我们 human preference 本身就有 ~25% 的 irreducible disagreement，所以 RM 在 ~73% accuracy 时已经接近 ceiling。

**Held-out labeler 实验**: 用 5-fold CV 训 5 个 RM (训练在 4/5 labelers 上，测在 1/5 上)，held-out accuracy 69.6 ± 0.9% vs in-distribution 72.4 ± 0.4% — 只掉 3 个百分点，说明 RM 没严重 overfit 到训练 labeler 的个人 idiosyncrasy。这是 paper 一个 understated 但 important 的 generalization claim。

---

## 5. 关键实验结果

### 5.1 API distribution 上的 winrate (Figure 1)

| Model | Winrate vs 175B SFT |
|---|---|
| 175B GPT-3 | ~15% |
| 175B GPT-3 (prompted) | ~25% |
| 175B SFT | 50% (baseline) |
| 175B PPO | ~80% |
| 175B PPO-ptx (InstructGPT) | ~85% |
| **1.3B PPO-ptx** | **~60%** ← 这个数字是 paper 的 punchline |

**1.3B InstructGPT 打败 175B GPT-3** — 这是 paper 最常被引用的数字。同样的 architecture (GPT-3 family)，同样的 tokenizer，唯一区别是 fine-tuning 数据 + RLHF。这从根本上改变了 LLM 的 scaling law 思考方式：**alignment 数据的 ROI 远高于 raw compute**。Paper Section 5.1 给出成本对比：175B SFT 训练只需 4.9 PF-days，175B PPO-ptx 需要 60 PF-days，而 GPT-3 预训练是 3,640 PF-days — alignment 只占预训练成本的 ~1.6%。

### 5.2 TruthfulQA (Figure 6)

在 TruthfulQA 上，175B PPO-ptx 的 "truthful + informative" 比例比 175B GPT-3 高出约 2×。在 "Instruction+QA" prompt 下 (让模型不确定时说 "I have no comment")，PPO 模型更倾向于 "truthful but uninformative" 而非 "confidently wrong"。

注意一个反直觉点：**1.3B PPO-ptx 在 TruthfulQA 上比 1.3B GPT-3 略差**。Paper 自己的解释是 1.3B 模型 capacity 不足以同时 fit instruction-following 和 truthful reasoning 两个 manifold — 这暗示 small model 上 alignment tax 更严重。

### 5.3 Hallucination (Figure 4)

Closed-domain task (summarization, closed QA) 上：
- GPT-3 hallucination rate: ~41%
- PPO model: ~21%

砍半。这是因为 RM 学到了 "answer should be grounded in input" 的偏好。

### 5.4 Toxicity (Figure 7, Figure 39)

在 RealToxicityPrompts 上：
- "respectful prompt" 下 InstructGPT 比 GPT-3 毒性低 ~25%
- "no prompt" 下差不多
- **"biased prompt" 下 InstructGPT 比 GPT-3 毒得多** — 因为 InstructGPT 更听话，让它生成 toxic 它就生成得更 toxic

这是 paper Section 5.3 重点 flag 的 limitation: **InstructGPT 把 helpfulness 放在 harmlessness 之上**。当 user 显式要 harmful content 时，模型会 comply。这是 alignment 的核心 tension — "follow user intent" vs "do what user actually wants" 的冲突。

### 5.5 Bias (Figure 32)

Winogender 和 CrowS-Pairs 上 **InstructGPT 没改善 bias**。Paper 解释是：instructed 模型 entropy 更低 (更确定)，不管方向是否 stereotype，都更 confident。这指向一个深层问题：**RLHF 让模型更 decisive，但不一定更 unbiased**。

### 5.6 Public NLP datasets (Figure 28, 29)

PPO (无 ptx mix) 在 SQuADv2, DROP, HellaSwag, WMT Fr→En 上都 regressed。PPO-ptx 几乎完全恢复，HellaSwag 甚至超过 GPT-3。

| Task (175B, few-shot) | GPT-3 | PPO | PPO-ptx |
|---|---|---|---|
| SQuADv2 F1 | 69.75 | 51.95 | 69.93 |
| DROP F1 | 35.27 | 27.78 | 33.34 |
| HellaSwag acc | 0.781 | 0.743 | 0.807 |
| WMT Fr→En BLEU | 39.93 | 26.58 | 36.76 |

SQuADv2 几乎完全恢复，DROP 和 WMT 还有 gap。Paper 暗示 alignment tax 的完全消除仍是 open problem。

### 5.7 vs FLAN / T0 (Figure 5)

175B GPT-3 在 FLAN / T0 上 fine-tune ~1M examples 后，Likert score 略高于 default GPT-3，与 GPT-3 (prompted) 相当，但**低于 SFT baseline**。InstructGPT vs FLAN winrate 78±4%，vs T0 79±4%。

Paper 给两个原因：
1. Public NLP dataset 任务分布与 API usage 不匹配 (classification/QA 占少数)
2. Public NLP dataset 输入 diversity 不足，不符合真实用户兴趣

这个结论对 instruction-tuning 研究有深远影响 — 它 justify 了 RLHF over pure SFT on instruction data 的价值。

---

## 6. Qualitative 泛化结果 (Section 4.3)

InstructGPT 在 fine-tuning 几乎全是英语的数据上，能 follow 法语 / 瑞典语指令 (Figure 8, 42)，能 summarize / QA code (Figure 45)。**Fine-tuning 数据中非英语 + code 占比 < 4%**，但模型学到了 "follow instruction" 这个 meta-skill，能 zero-shot 迁移到新 domain。

这是 paper 一个 exciting 的 generalization claim — alignment 不需要 per-task supervision。Karpathy 经常在 talk 里提到这种 "capability generalizes beyond supervision distribution" 的现象，InstructGPT 给了最早的实证。

### Simple mistakes (Figure 9)

1. **False premise**: "Why is it important to eat socks after meditating?" — InstructGPT 会编出 theories，而不质疑 premise。Paper 解释是训练数据里 false-premise prompt 太少。
2. **Over-hedging**: "What happens if you fire a cannonball at a pumpkin?" — InstructGPT 给一堆 "depends on factors" 而不直接说 "pumpkin explodes"。Paper 怀疑是 labeler 被 instruct 奖励 "epistemic humility"，结果 over-correct 了。

---

## 7. RLHF 的成本结构与启示 (Section 5.1)

| 项目 | PF-days |
|---|---|
| GPT-3 pretraining | 3,640 |
| 175B SFT | 4.9 |
| 175B PPO-ptx | 60 |
| **总 alignment 成本** | **~65 (占 pretraining 1.8%)** |

对比效果：1.3B InstructGPT > 175B GPT-3 (100× 参数差)。这意味着：
- **Alignment 是 high-leverage 投资**
- Scaling 大模型不如先把 alignment 做好
- 未来 superhuman alignment 的工作中，RLHF 是 building block (Christiano et al. 2018, Irving et al. 2018, Leike et al. 2018 都引用 RLHF 作为 scalable method)

---

## 8. 与后续工作的关联

### ChatGPT (2022.11)
ChatGPT 就是 InstructGPT 的产物化版本，paper 没公开的就是 ChatGPT 用了更大规模的数据、更精细的 labeler 团队、以及更复杂的 prompt distribution (对话式)。InstructGPT paper 是 ChatGPT 的技术底座。Reference: https://openai.com/blog/chatgpt

### Constitutional AI (Anthropic, 2022.12)
Anthropic 用 AI feedback (constitution-based) 替代部分 human feedback，让 RLHF scale 更好。这是对 InstructGPT "labeler 是 bottleneck" 的直接 response。Reference: https://arxiv.org/abs/2212.08073

### LLaMA-2 / Llama-3 RLHF (Meta)
Meta 公开了 RLHF 训练细节 (https://arxiv.org/abs/2307.09288)，其中 rejection sampling + PPO + DPO 三种方法对比，DPO (https://arxiv.org/abs/2305.18290) 因不需要训 RM 而崛起，但 InstructGPT 的 PPO pipeline 仍是 reference baseline。

### Direct Preference Optimization (DPO)
DPO 把 RM 训练 + PPO 一步到位 — 直接用 preference data 训 policy，跳过显式 RM。但 DPO 没有 KL anchor 的 natural mechanism，容易出现 over-optimization，实际部署中常常需要 NPO / IPO / KTO 等 variants (https://arxiv.org/abs/2402.01306)。InstructGPT 的 PPO-ptx 思路 — 在 objective 里混入 pretraining gradient — 在 DPO 里对应 "DPO + SFT regularization"。

### DeepSeek-R1 / o1 系列 (2024-2025)
RL 在 reasoning 上的扩展 (GRPO 等) 实际上是 InstructGPT 的精神继承：用 reward signal 优化 policy，但 reward 从 "human preference" 变成 "verifiable correctness"。Reference: https://arxiv.org/abs/2501.12948

### Reward Hacking / Goodhart's Law 文献
Gao et al. 2022 (https://arxiv.org/abs/2210.10760) 在 InstructGPT 之后量化了 RM over-optimization — KL penalty 的必要性。这是 InstructGPT Eq.2 第二项的理论依据。

---

## 9. Paper 没明说但能读出来的 subtext

### "Reward model 是 OOD 不可靠的"
Paper 只用 6B RM，且 RM 和 policy size 解耦 (所有 policy 共用一个 6B RM)。理由表面是 compute，深层是：**RM 在 policy 推进到的 OOD 区域是 unreliable 的**，6B RM 已经够用 — 训更大的 RM 收益递减。Gao et al. 2022 之后证实了这一点：RM quality 和 KL distance 是 log-linear 关系。

### "PPO-ptx 是 alignment 的妥协"
PPO-ptx 实际上是 "alignment 和 capability 的折中" — paper Section 5.4 承认 PPO-ptx 让 model 没完全摆脱 pretraining distribution 中的 undesirable behavior (toxicity 等)。这是 RLHF 工程化的 inherent tradeoff：完全 alignment 会 lose capability，完全 capability 会 lose alignment。

### "Labeler 不是 user"
Section 5.2 是 paper 最哲学的部分：alignment 的对象不是抽象的 "human values"，而是一个 specific group (40 个 contractor + OpenAI researchers + API customers)。这暗示 alignment 永远是政治问题，不是纯技术问题。后续 Anthropic 的 "Society-plus" (https://arxiv.org/abs/2310.01215) 和 OpenAI 的 Democratic AI (https://arxiv.org/abs/2310.16837) 都在这个方向探索。

### "Public NLP benchmarks 不再是 LLM 的 North Star"
Paper Section 4.1 的 FLAN/T0 对比是 manifesto：在真实 API 分布上，public instruction-tuning dataset 几乎无效。这推动了整个 field 从 "benchmark chasing" 转向 "real-world usage alignment"。

---

## 10. Karpathy 视角的可能联想

你自己 2023 年在 Stanford 的 talk "State of GPT" (https://www.youtube.com/watch?v=bZQun8Y4L2Z) 里专门画过 InstructGPT 的三步流程图，并强调 "SFT is imitation, RLHF is optimization"。这篇 paper 的 Eq.2 是这个论断的数学体现 — SFT 是 behavior cloning (maximum likelihood)，RL 是 reward maximization (different objective entirely)。

你在 nanoGPT (https://github.com/karpathy/nanoGPT) 和 nanoRLHF 思路里能直接复现这套 pipeline — 唯一 caveat 是 RM 训练的 list-wise trick (一个 batch 内所有 $\binom{K}{2}$ comparisons 同 forward) 在小规模复现时容易 OOM，实践里可以退化为 random pair sampling。

你在 2024 年的 "Deep Dive into LLMs like ChatGPT" 系列讲座 (https://www.youtube.com/watch?v=7xTGNNVPypU) 里详细讲过 RLHF 的 PPO 部分，强调了 GAE 和 reward whitening — InstructGPT paper 没明说 reward whitening 但 RM normalization (demonstrations mean=0) 起到类似作用。

---

## 11. 一些 paper 没充分讨论但关键的细节

### RM 的 inference cost
PPO 每个 token 都要 forward RM (计算 reward) 和 reference policy (计算 KL)，所以 PPO 训练 cost ≈ 2× SFT 训练 cost × token 数。Paper 给的 60 PF-days 已经包含这个 cost，但 paper 没拆解 "多少是 RM forward，多少是 policy forward"。

### Prompt filtering 的 PII 处理
训练数据全部 filter 过 PII。这是工程必需但 paper 一笔带过 — 实际 pipeline 里 PII filter 本身就是一个 production-grade 系统 (regex + NER model + LLM-as-judge)。

### Labeler 薪酬与 satisfaction
Table 13 显示 47.4% "strongly agree" paid fairly，42.1% "agree" — 没人 disagree。这在 data-labeling 行业里是非常罕见的满意度。Paper 没给具体时薪数字，但 Scale AI 行业均价 ~$15-25/hr，OpenAI 通常给 contractor 更高 ($30-50/hr 区间)。

### Value function 的 detachment
PPO 的 value function 用 6B RM 初始化，但**所有 size 的 policy 都用同一个 6B value function**。这是个工程妥协 — 175B policy 用 6B value 会 underestimate advantage，但 paper Section C.4 实测不影响最终 quality。这与 DeepSeek-R1 后来的 GRPO (抛弃 value function) 形成对比。

### KL penalty 的 per-token 实现
KL 不是 trajectory-level 的，是 per-token 累加：$\sum_t \log(\pi^{RL}(y_t|y_{<t}, x) / \pi^{SFT}(y_t|y_{<t}, x))$。这意味着短 response 和长 response 的 KL 量级不同，工程上需要按 length normalize 或者用 mean-KL。Paper 没明确说用哪种，从 PPO 实现惯例推测是 sum-KL。

---

## 12. 公式直觉总结

**Eq. 1 (RM loss) 的直觉**：
让 RM 在 winner 和 loser 的 reward 差上做 logistic regression — reward 差越大且方向正确，loss 越小。除以 $\binom{K}{2}$ 是为了 list size 不同时归一化。这个 loss 形式等价于 Bradley-Terry model 的 maximum likelihood，是 pairwise ranking 的标准做法。

**Eq. 2 (PPO-ptx objective) 的直觉**：
三项分别是：
1. **Exploit RM**: 让 policy 采样的 response 拿高 reward
2. **Stay close to SFT**: KL penalty，policy 不能漂离 SFT 太远 — 这是 regularization，防止 reward hacking
3. **Stay close to pretraining distribution**: 混入 pretraining 数据的 next-token likelihood，让 model 在 RL 过程中不遗忘原始 capability

第一项是 exploitation，第二、三项是 exploration-ish 的 regularization。这是 RL 在 LM 上能 work 的核心 trick — 没 KL 和 ptx mix，PPO 会快速 collapse 到 RM 的 adversarial example 上。

---

## 13. 一些延伸阅读

- **Christiano et al. 2017 (Deep RL from Human Preferences)**: RLHF 的开山之作 https://arxiv.org/abs/1706.03741
- **Stiennon et al. 2020 (Learning to Summarize from Human Feedback)**: InstructGPT 的直接前驱 https://arxiv.org/abs/2009.01325
- **Schulman et al. 2017 (PPO)**: https://arxiv.org/abs/1707.06347
- **Gao et al. 2022 (Scaling laws for reward model over-optimization)**: https://arxiv.org/abs/2210.10760
- **Rafailov et al. 2023 (DPO)**: https://arxiv.org/abs/2305.18290
- **Bai et al. 2022 (Constitutional AI)**: https://arxiv.org/abs/2212.08073
- **Touvron et al. 2023 (Llama-2)**: https://arxiv.org/abs/2307.09288
- **Ouyang et al. 2022 (InstructGPT 原文)**: https://arxiv.org/abs/2203.02155
- **Karpathy "State of GPT" talk**: https://www.youtube.com/watch?v=bZQun8Y4L2Z
- **OpenAI ChatGPT announcement**: https://openai.com/blog/chatgpt
- **DeepSeek-R1**: https://arxiv.org/abs/2501.12948
- **Anthropic HH-RLHF dataset**: https://huggingface.co/datasets/Anthropic/hh-rlhf
- **TRL (HuggingFace RLHF library)**: https://huggingface.co/docs/trl/index
- **CarperAI trlx (开源 RLHF)**: https://github.com/CarperAI/trlx

---

## 14. 一句话总结

InstructGPT 的贡献是把 Christiano 2017 的 RLHF 从 Atari / summarization 推广到 general instruction-following domain，用 PPO + KL + pretraining mix 三件套把 alignment tax 压到 ~2% pretraining cost，证明了 **objective design 的 ROI 远高于 capacity scaling**。Eq.1 和 Eq.2 是整个 RLHF pipeline 的两行核心数学 — 一行定义 preference，一行定义 optimization。这两行加上 40 个 labeler，就是 ChatGPT 时代的起点。
