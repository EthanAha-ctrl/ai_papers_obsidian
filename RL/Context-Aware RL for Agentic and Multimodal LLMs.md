---
source_pdf: Context-Aware RL for Agentic and Multimodal LLMs.pdf
paper_sha256: 04449884aaa2e6df9ac0d8a60360ea70acee765cadd4b369d1fb3b2ed0f8bfcb
processed_at: '2026-08-18T04:06:17-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CONTEXTRL

## 一句话概括

Model 经常"答对了但没真正看 context"，作者给 model 出了一道"找证据"的选择题，让它学会真的 grounded 在 context 上，而不是靠 prior 蒙对。

## 问题到底是什么

举几个直觉例子：

**Agentic coding 场景**：你让 agent 改一个 bug，agent 改完之后跑通了 test，但其实它没看你给的 source file 里已经定义了 `i` 这个变量，自己瞎删了，碰巧 test 没覆盖到，于是 resolve rate 涨了点，但这个 fix 实际上是 broken 的。

**VQA 场景**：图里 $g(x)$ 在 $x \to -1$ 时 $y=3$，model 读成 2，然后基于 2 算了一堆，最后答案错了。相关 information 就在图里，model 就是没看对。

作者管这种现象叫 **context unawareness** — information 就在 context 里，但 model 的 prediction 没 grounded 在上面。

为什么这事重要？因为现在的 benchmark 数字会骗人。作者做了个非常 simple 的 probe：给 model 一个 $(Q, A)$ pair 和两个长得很像的 context $C^+$ (支持 $A$) 和 $C^-$ (不支持)，让它选哪个 context 支持 $A$。

结果 Figure 2 非常 striking：
- GPT-5.4, Claude Opus 4.7：80%+ 正确
- Qwen3-8B, Qwen3.5-9B：接近 random choice (50%)

这些 Qwen model 在标准 benchmark 上表现很好，但这个简单的 grounding test 就把它们打回原形。这说明 benchmark 分数和真实 grounding 能力是两回事。

## 作者的 Insight

Standard RL 只告诉 model "你答对了"，但不告诉它"你答对是因为看了 context 还是靠 prior 蒙的"。这两种 path 在 outcome reward 上是 indistinguishable 的。

作者的解法很 elegant：**显式构造一道选择题，让 model 证明自己真的能区分 supporting context 和 confounding context**。

这跟 DPO 的精神一脉相承 — DPO 把"prefer response A over B"做成 implicit reward；CONTEXTRL 把"prefer context $C^+$ over $C^-$ given (Q,A)"做成 auxiliary loss。

关键 trick 是 **decoupling**：把"产出什么 answer"和"哪个 context 支持这个 answer"分开训练。前者用 GRPO，后者用 contrastive loss。

## Contrastive Data 怎么构造

这是 paper 的核心工程贡献。两个 domain 用不同方法。

### Agentic：从 66k trajectory 里挖 1k hard pair

Pipeline 极其 aggressive（保留率 1.5%）：

1. 同 repo + 同 commit（保证 codebase 相同）
2. 改同 file
3. 改同 function/class（只在 small code region 不同）
4. Issue 相关但 distinct
5. **Mask patch content**：把 edit command 里的 patch 用 `<PATCH_MASKED>` 替换，防止 model 直接读 patch 就知道选哪个

然后 GPT-5.4 verifier reject 所有有 superficial shortcut 的 pair（length disparity / formatting difference / patch token leakage）。

关键 insight：**negatives 不是 random samples，是 token-level 几乎 identical 的 hard negatives**。同 repo、同 commit、同 file、同 function，只在 small region 不同。这逼 model 必须 understanding context，不能靠 surface statistics。

### Multimodal：两条 path 凑 7k pair

**Natural images (700 pairs)**：GPT-5.4 提议 edit → Nano Banana 2 执行 edit → GPT-5.4 reject artifact。要求 edit localized 到 answer-relevant region，其他不动。65% rejection rate。

**Structured images (6300 pairs)**：用 Qwen3-VL-Embedding 8B encode image，retrieve cos similarity ≥ 0.85 且 answer 不同的 pair。0.85 是非常 aggressive 的 threshold，几乎是 similarity ceiling。3.1% 保留率。

## Loss Function

给定 contrastive instance $z = (Q, A, C^+, C^-)$：

Model 看到 $Q, A$ 和两个 labeled options ("A" / "B")，$C^+$ 和 $C^-$ 顺序随机化（消除 position bias）。

- $t^+$：$C^+$ 对应的 option letter token（比如 "A"）
- $t^-$：$C^-$ 对应的 option letter token（比如 "B"）
- $\ell_\theta^+(z)$：model 对 $t^+$ 的 next-token logit
- $\ell_\theta^-(z)$：model 对 $t^-$ 的 next-token logit
- **用 teacher forcing 计算，不需要 sampled rollout**

Margin：$\Delta_\theta(z) = \ell_\theta^+(z) - \ell_\theta^-(z)$

Loss：
$$\mathcal{L}_{\mathrm{CA}}(z; \theta) = -\log \sigma\left(\mathrm{clip}\left(\Delta_\theta(z), -c, c\right)\right)$$

- $\sigma$：sigmoid
- $c = 5.0$：clip bound，防止 large margin 主导 training
- $-\log \sigma(\cdot)$：binary cross-entropy 作用在 margin 上

Joint objective：
$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{RL}}}[\mathcal{L}_{\mathrm{GRPO}}(x; \theta)] + \lambda \mathbb{E}_{z \sim \mathcal{D}_{\mathrm{CA}}}[\mathcal{L}_{\mathrm{CA}}(z; \theta)]$$

$\lambda$ 非常小（0.001-0.005），auxiliary loss 只是"轻轻推一下" policy。

### 为什么这个设计 work

**Teacher forcing 的 dense gradient**：DA-RL 的 binary reward (0/1) 极其 sparse — model 必须 actually sample 正确 option 才能拿到 reward，初期 sampling probability 很低，gradient 几乎为 0。CONTEXTRL 通过 teacher forcing，每个 example 都有 meaningful gradient，即便 model 现在很少 sample 正确 context。

**Clip 的 automatic curriculum decay**：
- Early training：$\Delta \approx 0$，gradient 最大，auxiliary signal 最强
- Late training：$\Delta > c$，gradient 变 0，auxiliary signal "退出"，把 budget 让给 primary task

这是 automatic curriculum — auxiliary task 在 model 需要时 active，掌握后 fade out。

**Bounded $\lambda$ 防 collapse**：Table 10 的 ablation 显示 $\lambda = 0.01$ 反而比 baseline 差。Auxiliary signal 太强会 compete with GRPO，破坏 primary task policy。

## 实验结果

### Long-Horizon (Table 1)

5 个 benchmark：SWE-Bench Verified, SWE-Bench Lite (ID), LiveCodeBench v6, LongBench v2, NIAH (OOD)。

| Base model | Avg Δ over RL baseline |
|---|---|
| Klear-AgentForge-8B | +3.2% |
| Qwen3-8B | +1.5% |

几个 striking 点：
- Klear-AgentForge-8B + CONTEXTRL 在 SWE-Bench 上 **outperform Qwen3-32B (4× larger) 和 Qwen3-Coder-30B**
- Standard GRPO 在 NIAH 上 **regress** relative to base，CONTEXTRL 反而超越 base
- LongBench v2 Long subset +4.6%，NIAH +5.8%（Klear base）— 这些是 pure long-context retrieval，跟 agentic coding 完全不同 domain

OOD transfer 比 ID gain 还大，这是最强 evidence：学到的是 general context grounding skill，不是 task-specific shortcut。

### Multimodal (Table 2)

12 个 benchmark 跨 5 个 category。CONTEXTRL 在 **每个** benchmark 上都 outperform RL baseline，没有 trade-off。这说明不是 category-specific tuning，是 underlying skill 提升。

对比 PAPO（perception-aware RL）：CONTEXTRL +2.0% > PAPO +0.8%（Qwen2.5-VL）。Context-selection 的 auxiliary signal 比单纯 perception reward shaping 更有效。

## 最有 Insight 的对照实验

作者用 **完全相同的 contrastive data** 构造两个 baseline：

- **DA-SFT**：先 SFT 学 context selection，再 GRPO
- **DA-RL**：把 contrastive 直接 mix 进 RL stream，binary reward

结果非常 striking：

**Agentic**：
- DA-SFT：Klear 28.0 → 6.4，Qwen3-8B 6.20 → **0.00**（catastrophic collapse）
- DA-RL：几乎无变化
- CONTEXTRL：Klear 30.2，Qwen3-8B 7.00

**Multimodal**：
- DA-SFT：+0.1%（negligible）
- DA-RL：+0.4%（negligible）
- CONTEXTRL：+2.0%

### 为什么 DA-SFT 在 agentic 上 collapse

Agentic coding 的 policy distribution 是高度 specialized 的 — model 需要在 long-horizon, multi-turn, tool-use format 下 maintain 特定行为 pattern（何时 explore file, 何时 edit, 何时 test）。SFT on short selection examples 把 model 推到完全不同的 distribution（single-turn, short-answer），这种 distribution shift 在 long-horizon setting 下是 catastrophic。

Multimodal 没 collapse 是因为它是 single-turn short-answer，format mismatch 不严重。

### 为什么 DA-RL 没用

Binary reward (0/1) 太 sparse。Model 必须 actually sample 正确 option 才能拿到 reward，初期 sampling probability 很低，gradient 几乎为 0。这就是 outcome-based RL 的根本 limitation。

### Mechanism Analysis (Figure 5)

作者把 selection accuracy (x-axis) vs end-task performance (y-axis) 画出来：

1. **RL baseline 和 DA-RL**：stay near base model cluster — outcome reward 学不到 discrimination
2. **DA-SFT**：selection accuracy 最高（85-93%），但 end-task collapse — 学到了 selection 但破坏 policy
3. **CONTEXTRL**：selection accuracy 高 **且** end-task 提升 — 唯一两者兼得的方法

这里有个重要 implication：如果 selection accuracy 是靠 construction artifact 驱动的，那么 selection accuracy 最高的 DA-SFT 应该 transfer 最好。但事实相反 — DA-SFT transfer 最差。这反驳了"CONTEXTRL 只是学到 artifact"的 concern。

## 我的 Intuition Building

### 数据效率的启示

1k agentic + 7k multimodal contrastive pair，加上 standard 7k/38k task data，就拿到 +2% consistent 提升。对比 DeepSWE 和 SWE-RL 这些 scale 到几十万量级的方法，CONTEXTRL 用极少量 high-quality contrastive data + auxiliary objective 就能 beat。

这暗示一个 insight：**很多 capability gap 不是 data 量不够，是 supervision signal 的 form 不对**。Outcome reward 是 dense in quantity 但 sparse in information — 一万个 trajectory 只告诉 model "对/错"。Contrastive pair 哪怕只有一千个，每个 pair 都 dense in information — 显式指出"这两个 context 的差异是决定性的"。

### 跟 PRM 的对比

OpenAI 的 Let's Verify Step by Step 和 Math-Shepherd 训练 process reward model 给 reasoning step 打分。CONTEXTRL 的区别：

- PRM：reward reasoning process 的 internal consistency
- CONTEXTRL：reward reasoning 和 external evidence 的 alignment

两者互补，一个 natural future work 是 combine。

### 跟 VC-STaR, mDPO 的对比

VC-STaR：contrast visually similar VQA pairs to improve VLM reasoning。mDPO：augment DPO with image-side preference term。

关键区别：这些方法都是 **fix context, prefer one response over another**。CONTEXTRL 反过来：**fix (Q,A), prefer one context over another**。这是 orthogonal axis。

这个 flip 很深刻 — 把 grounding 从"output 是否对"变成"input evidence 是否被 identify"。这种 input-side supervision 在 representation learning 里有 CLIP 的精神 — contrast 不同 modality 的 alignment。

### OOD Transfer 的 Root Cause

Table 1 里 LongBench v2 Long +4.6%，NIAH +5.8%（Klear base）— 这些是 pure long-context retrieval，跟 agentic coding 完全不同 domain。为什么能 transfer？

Hypothesis：contrastive context selection 训练的是一种 meta-skill — "scan context, identify supporting evidence"。这个 meta-skill 在任何 long-context task 上都有用，不管 context 是 trajectory 还是 document。

这跟 in-context learning 的 meta-learning 视角一致 — ICL 本质是 model 学会了 "given examples, infer pattern"。CONTEXTRL 教的是 "given context, identify relevant evidence"，这是更 fine-grained 的 ICL skill。

### 跟 Reward Hacking 的关系

Outcome-only reward 容易导致 model 学习 shortcut 而非真实 capability。CONTEXTRL 的 auxiliary loss 本质是 anti-reward-hacking 的 — 它显式 reward grounding behavior，让 shortcut path 拿不到 auxiliary reward。

这跟 Constitutional AI 的 spirit 类似 — 用 explicit principle（这里是 context grounding）来 constrain reward signal。但 CONTEXTRL 更轻量，不需要额外的 critique model。

## Limitations 和 Concerns

Paper 自己承认：只在 <10B model 上验证，主要在 Qwen family。我加几个：

1. **Contrastive pair diversity**：1k agentic + 7k multimodal 虽然高效，但 coverage 可能不够。Agentic 只覆盖 SWE-smith 的 Python repo，其他语言 / 其他 agent task type 如何？

2. **Selection format vs free-form grounding**：CONTEXTRL 训练的是 multiple choice format 的 selection，实际 deployment 是 free-form generation。Paper 证明 transfer 是 work 的，但 mechanism 不完全 clear。

3. **Contrastive pair stability**：GPT-5.4 verifier 本身有 noise。~65% 和 ~97% 的 rejection rate 之后，remaining 里可能还有 false positive。需要 human spot check estimate false positive rate。

4. **Long-term retention**：Auxiliary skill 是否会随 further training 而 forget？这对 deployment 很重要。

5. **Multi-auxiliary composition**：如果同时加 multiple auxiliary losses（coherence, safety, grounding），它们如何 interact？Multi-auxiliary balancing 是 open problem。

## Future Direction Speculation

1. **Multi-context selection**：从 binary (A/B) 扩展到 N-way selection，类似 InfoNCE 的 NCE loss，更强 discrimination signal。

2. **Hierarchical context**：把 context decompose 成 hierarchical chunks（function-level, file-level, repo-level），让 model 学 hierarchical grounding。

3. **Self-generated contrastive pairs**：让 model 自己 generate contrastive pairs，类似 Self-Rewarding LLMs，大大降低 data 构造成本。

4. **Active context selection**：训练 model 不仅选 context，还主动 query 相关 context，bridge agentic tool use 和 context grounding。

5. **Adversarial contrastive**：让另一个 model 主动 generate hard negatives challenge current model，类似 GAN 的 adversarial training，curriculum 难度自动 adjust。

## 最后的 Takeaway

这篇 paper 的核心 contribution 不是某个具体技术 trick，是一个 **conceptual reframe**：把 context grounding 从 implicit outcome 显式化为 verifiable auxiliary task。

它的 elegance 在于 modality-agnostic、data-efficient、compute-efficient、stable、transferable。它揭示了 RL post-training 的 blind spot：outcome reward 能教 model "答对"，但教不了"为什么答对"。Contrastive context selection 显式 supervise 这个"为什么"，把 grounding 从 emergent property 变成 explicit training signal。

这跟 (Karpathy) 我一直强调的 "understand the mechanism, not just the metric" 是同一个 philosophy。Benchmark 数字会骗人，但 controlled contrastive probe 不会 — 它强迫 model 证明自己真的 grounded。

相关 reference：

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [SimCSE](https://arxiv.org/abs/2104.08821)
- [InfoNCE / CPC](https://arxiv.org/abs/1807.03748)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [PAPO](https://arxiv.org/abs/2507.06448)
- [mDPO](https://arxiv.org/abs/2406.11839)
- [VC-STaR](https://arxiv.org/abs/2603.02556)
- [SWE-RL](https://arxiv.org/abs/2502.18449)
- [DeepSWE](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33)
- [LongRLVR](https://arxiv.org/abs/2603.02146)
- [FILM](https://arxiv.org/abs/2404.16811)
- [Turpin et al. on unfaithful CoT](https://arxiv.org/abs/2305.04388)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [GradNorm](https://arxiv.org/abs/1711.02257)
- [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [SWE-smith](https://arxiv.org/abs/2504.21798)
- [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)
- [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923)

---

# Context-Aware RL for Agentic and Multimodal LLMs 深度解析

## 一、问题动机:Context Unawareness 这个 failure mode

Paper的核心观察是: 即便现代 LLM 在标准 benchmark 上表现接近 proprietary model, 它们在 grounding 决策时常常"无视"决定性 evidence. Figure 1 给出两个典型 failure:

- **Agentic coding**: model 有 source file access, 但跨 edit 时丢失 consistency, 把后续被 reference 的 variable `i` 的 definition 删掉, 导致 runtime error
- **Multimodal**: 图中 $g(x)$ 在 $x \to -1$ 时 $y=3$, model 把它读成 2, 给出错误 prediction

作者构造了一个 controlled contrastive context probe 来量化这个问题. 给 model 一个 $(Q, A)$ pair 和两个高度相似的 context $C^+, C^-$ (其中一个支持 $A$), 看 model 能否选出 $C^+$. Figure 2 的结果非常 striking: GPT-5.4 和 Claude Opus 4.7 在 80%+ 附近, 而 Qwen3-(VL)-8B 和 Qwen3.5 9B 都接近 random choice (50%). 这意味着 strong benchmark performance 完全可以掩盖 context grounding 的失败.

这跟我 (Karpathy) 一直思考的 "shortcut learning" 现象高度一致. 当 reward signal 只关注 final answer, model 完全可以学习一个 surface-level heuristic (比如某些 token pattern, position bias) 来 pass benchmark, 同时在 distribution shift 下彻底崩溃. 参考 [Anthropic 的 Sleeper Agents paper](https://arxiv.org/abs/2401.05566) 和 [Turpin et al. on unfaithful CoT](https://arxiv.org/abs/2305.04388), 都指向同一个 root cause.

## 二、核心 Insight:把 Context Grounding 变成 Verifiable Auxiliary Task

作者的 insight 很 elegant: 与其猜测 model 是否"真的" grounded, 不如显式构造一个 contrastive task 让它证明. 这与 [DPO](https://arxiv.org/abs/2305.18290) 的精神一脉相承 — DPO 把"prefer response A over B"做成 implicit reward; CONTEXTRL 把"prefer context $C^+$ over $C^-$ given $(Q, A)$"做成 auxiliary loss.

这里的关键 trick 是 decoupling: **把 "what answer to produce" 和 "which context supports it" 分开训练**. 标准 outcome-based RL 只能告诉 model "你答对了", 但无法区分:
- (a) model grounded 在 context 里某条 evidence
- (b) model 用了 prior / shortcut

CONTEXTRL 通过强制 model 在两个 minimally different context 之间做选择, 显式 reward (a) 这条 path.

## 三、Contrastive Data 构造的两个 Domain

### 3.1 Agentic:Cascade Filtering from SWE-smith Trajectories

Pipeline 极其 aggressive, 66k trajectories → 1k pairs (1.5% 保留率):

1. **Group by repo + commit**: 保证 underlying codebase 相同
2. **Filter by modified file**: 同一文件被改
3. **Filter by target function/class**: 解析 patch hunks, 保证改的是同一 function/class, 只 differ 在 small decisive code region
4. **Filter by issue relation**: issue distinct 但 semantically related (同 API / 同 corner case family)
5. **Mask direct leakage**: patch content inside edit commands 用 `<PATCH_MASKED>` 替换, 防止 model 直接读 edit command 就知道答案

然后 GPT-5.4 verifier 显式 reject 包含 superficial shortcut 的 pairs (length disparity / formatting difference / patch token leakage / masking inconsistency). UNCERTAIN 的 case 升级到 manual review.

这个设计很关键: **negatives 不是 random samples, 而是 token-level 几乎 identical 的 hard negatives**. 同 repo, 同 commit, 同 file, 同 function, 只在 small code region differ. 这逼 model 必须 understanding context 而不能靠 surface statistics.

### 3.2 Multimodal:Generative Editing + Similarity Retrieval

两条 path:

**Natural images (700 pairs from 2k candidates, ~65% rejection rate)**:
- GPT-5.4 propose edit prompt 改变 answer
- Nano Banana 2 应用 edit
- GPT-5.4 reject artifacts (blur, warping, broken boundaries, implausible lighting, texture mismatch, floating objects) 或 global restyling
- 必须 localized 到 answer-relevant region, question-irrelevant content 不变

**Structured images (6300 pairs from >200k candidates, ~3.1% 保留率)**:
- 用 Qwen3-VL-Embedding 8B encode images
- retrieve $(I', Q, A')$ 使得 $\cos(f_I(I), f_I(I')) \geq \alpha_I$, $\alpha_I = 0.85$
- $A \ne A'$ (不同 answer)
- GPT-5.4 filter semantically unrelated pairs 或 annotation artifacts

注意 $\alpha_I = 0.85$ 这个 threshold 的选择非常 aggressive — 几乎是 similarity ceiling. 这是 hard negative mining 的极致形式, 类似 [SimCSE](https://arxiv.org/abs/2104.08821) 里的 hard negative 思路.

## 四、Context-Awareness Loss 详解

### 4.1 公式与变量

给定 contrastive instance $z = (Q, A, C^+, C^-)$:

- $Q$: query (GitHub issue 或 VQA question)
- $A$: anchor answer (reference patch 或 ground-truth answer)
- $C^+$: supporting context (正确 trajectory 或正确 image)
- $C^-$: confounder context (similar 但 inconsistent)

Model 看到的是 $Q, A$ 以及两个 labeled options ("A" / "B"), $C^+$ 和 $C^-$ 的顺序随机化消除 position bias.

- $t^+$: option-letter token assigned to $C^+$ (比如 "A")
- $t^-$: option-letter token assigned to $C^-$ (比如 "B")
- $\ell_\theta^+(z)$: model 的 next-token logit for $t^+$ at answer position
- $\ell_\theta^-(z)$: model 的 next-token logit for $t^-$ at answer position

**关键: 这些 logits 用 teacher forcing 计算, 不是 sampled rollout**. 这是 dense gradient 的来源 — 不需要等 model 真的 sample 出 "A" 或 "B", 每一步都有 gradient.

Margin 定义:
$$\Delta_\theta(z) = \ell_\theta^+(z) - \ell_\theta^-(z)$$

Loss:
$$\mathcal{L}_{\mathrm{CA}}(z; \theta) = -\log \sigma\left(\mathrm{clip}\left(\Delta_\theta(z), -c, c\right)\right) \tag{1}$$

- $\sigma(\cdot)$: sigmoid, 把 margin 压到 $(0, 1)$
- $c > 0$: clip bound, paper 里 $c = 5.0$, 防止 large margin 主导 training
- $-\log \sigma(\cdot)$: 标准 binary cross-entropy 形式, 但作用在 margin 上

### 4.2 与 DPO 的形式对比

DPO 的 loss 是:
$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)$$

CONTEXTRL 的形式简化版:
$$\mathcal{L}_{\mathrm{CA}} = -\log \sigma\left(\mathrm{clip}(\ell_\theta^+ - \ell_\theta^-, -c, c)\right)$$

差异:
- DPO 比较 response log-prob ratio (with reference), CONTEXTRL 比较 single-token logit
- DPO 处理的是 response preference, CONTEXTRL 处理的是 context preference
- CONTEXTRL 加了 clip, DPO 通常用 $\beta$ 作为 temperature 控制 sharpness

这个 single-token logit margin 的 trick 很聪明 — 它把"选 context"这个 task 压缩到 one decision step, 同时利用了 LLM 在 multiple choice format 上的 inductive bias. 参考 [RACE benchmark](https://arxiv.org/abs/1705.02736) 和 [MMLU](https://arxiv.org/abs/2009.03300) 的 multiple choice eval, model 在这种 format 下有天然的 calibration.

### 4.3 Joint Objective

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{RL}}}\left[\mathcal{L}_{\mathrm{GRPO}}(x; \theta)\right] + \lambda \mathbb{E}_{z \sim \mathcal{D}_{\mathrm{CA}}}\left[\mathcal{L}_{\mathrm{CA}}(z; \theta)\right] \tag{2}$$

- $\mathcal{D}_{\mathrm{RL}}$: standard task data (7k agentic / 38k multimodal)
- $\mathcal{D}_{\mathrm{CA}}$: contrastive pairs (1k / 7k)
- $\lambda > 0$: 平衡系数, agentic 用 0.005 (Klear) 或 0.001 (Qwen3-8B), multimodal 用 0.005 (Qwen2.5-VL) 或 0.001 (Qwen3-VL)

$\lambda$ 非常小 (0.001-0.005) 是关键设计 — auxiliary loss 只是"轻轻推一下" policy, 不让它 dominate GRPO 的 primary signal. Table 10 的 ablation 显示 $\lambda = 0.01$ 反而比 baseline 还差, 验证了这个设计.

## 五、实验结果

### 5.1 Long-Horizon (Table 1)

5 个 benchmark: SWE-Bench Verified, SWE-Bench Lite (in-distribution), LiveCodeBench v6, LongBench v2, NIAH (out-of-distribution).

| Base model | Avg Δ over RL baseline |
|---|---|
| Klear-AgentForge-8B | +3.2% |
| Qwen3-8B | +1.5% |

最 striking 的几点:
- Klear-AgentForge-8B + CONTEXTRL 在 SWE-Bench 上 **outperform Qwen3-32B (4× larger) 和 Qwen3-Coder-30B**
- OOD transfer 强: LongBench v2 Long subset +4.6%, NIAH +5.8% (Klear base)
- Standard GRPO 在 NIAH 上 **regress** relative to base, CONTEXTRL 反而超越 base

这个 OOD transfer 是最强 evidence, 证明学到的是 general context grounding skill, 而不是 task-specific shortcut.

### 5.2 Multimodal (Table 2)

12 个 benchmark 跨 5 个 category: Math reasoning, General multimodal, Fine-grained perception, Scientific reasoning, Real-world scene.

| Base model | Avg Δ over RL baseline |
|---|---|
| Qwen2.5-VL-7B | +2.0% |
| Qwen3-VL-8B | +1.6% |

CONTEXTRL 在 **每个** benchmark 上都 outperform RL baseline, 没有 trade-off. 这点很重要 — 如果只是 category-specific tuning, 应该有 trade-off. Breadth of improvement 强烈暗示 underlying skill 的提升.

对比 PAPO (perception-aware RL, [paper](https://arxiv.org/abs/2507.06448)): CONTEXTRL +2.0% > PAPO +0.8% (Qwen2.5-VL). 这个对比虽然不严格 (PAPO 用自己的 dataset 和 RL formulation), 但说明 context-selection 的 auxiliary signal 比单纯的 perception reward shaping 更有效.

### 5.3 关键对照实验:为什么 Data Augmentation 失败 (Section 5)

这是 paper 最有 insight 的部分. 作者用 **完全相同的 contrastive data** 构造两个 baseline:

- **DA-SFT**: 先 SFT 学 context selection, 再 GRPO
- **DA-RL**: 把 contrastive 直接 mix 进 RL stream, binary reward

结果:

**Agentic (Table 3)**:
- DA-SFT: Klear 28.0 → 6.4, Qwen3-8B 6.20 → 0.00 (**catastrophic policy collapse**)
- DA-RL: 几乎无变化 (Klear 28.0 → 27.6, Qwen3-8B 6.20 → 5.60)
- CONTEXTRL: Klear 30.2, Qwen3-8B 7.00

**Multimodal (Table 4)**:
- DA-SFT: +0.1% (negligible)
- DA-RL: +0.4% (negligible)
- CONTEXTRL: +2.0%

### 5.4 为什么会这样?Mechanism Analysis (Figure 5)

作者把 selection accuracy (x-axis) vs end-task performance (y-axis) 画出来, 三个清晰的 pattern:

1. **Outcome-only RL fails to learn context selection**: RL baseline 和 DA-RL 都 stay near base model cluster. Sparse 0/1 reward 学不到 discrimination.

2. **DA-SFT 学到 selection accuracy 但破坏 policy**: DA-SFT 在几乎所有 setting 都达到 highest selection accuracy (85-93%), 但 end-task performance 反而 collapse. 原因: SFT on short selection examples 把 model distribution 推离 long-horizon multi-turn pattern, 在 agentic setting 这种 mismatch 是 catastrophic.

3. **Selection skill alone 不足**: DA-SFT 和 CONTEXTRL 都 push selection accuracy 到 85-93%, 但只有 CONTEXTRL 提升 downstream. Context selection 是 necessary but not sufficient — 必须 acquire 这个 capability **without disrupting policy**.

这给了一个很强的结论: **如果 selection accuracy 是靠 artifact 驱动的, 那么 selection accuracy 最高的 DA-SFT 应该 transfer 最好, 但事实相反**. 这反驳了 "CONTEXTRL 只是学到 construction artifacts" 这个 concern.

## 六、为什么 CONTEXTRL 能避开两个 failure mode

### 6.1 Updates remain constrained

两个机制:

(a) **GRPO 的 importance-ratio clipping**:
$$\mathrm{clip}(\rho_t^{(i)}, 1-\epsilon, 1+\epsilon)$$

- $\rho_t^{(i)} = \frac{\pi_\theta(a_t^{(i)}|s_t^{(i)})}{\pi_{\mathrm{old}}(a_t^{(i)}|s_t^{(i)})}$: importance ratio
- $\epsilon$: clip bound (PPO-style)
- 加上 KL regularization to $\pi_{\mathrm{ref}}$, keep policy close to reference

(b) **CONTEXTRL 的 margin clipping**: $\mathrm{clip}(\Delta_\theta(z), -c, c)$

当 $C^+$ 和 $C^-$ 已经 well separated ($\Delta > c$), gradient 变 0, auxiliary signal 不再 dominate. 这是 dense signal + bounded update 的关键组合.

### 6.2 Auxiliary signal is dense

DA-RL 的 binary reward (0/1) 极其 sparse — model 必须 actually sample correct option 才能拿到 reward, 而 sampling probability 在初期很低. 

CONTEXTRL 的 $\mathcal{L}_{\mathrm{CA}}$ 直接 supervise **relative preference** between $C^+$ 和 $C^-$ on **every** example, 通过 teacher forcing. 即便 policy 现在很少 sample 正确 context, 仍然有 meaningful gradient. 这是 contrastive loss 相对 outcome reward 的本质优势.

这跟 [SimCSE](https://arxiv.org/abs/2104.08821) 和 [InfoNCE](https://arxiv.org/abs/1807.03748) 在 representation learning 里的 dense gradient 优势是同一个道理.

## 七、我的 Intuition Building 和相关联想

### 7.1 与 Process Reward Model 的对比

[OpenAI 的 Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 和 [Math-Shepherd](https://arxiv.org/abs/2312.08935) 训练 process reward model 给 reasoning step 打分. CONTEXTRL 的区别:

- PRM: 给 reasoning process 的每个 step 打 reward
- CONTEXTRL: 不直接 reward reasoning step, 而是 reward "你能不能 identify supporting context"

这两者其实互补 — PRM 关注 reasoning 的 internal consistency, CONTEXTRL 关注 reasoning 和 external evidence 的 alignment. 一个很自然的 future work 是 combine 两者.

### 7.2 与 [VC-STaR](https://arxiv.org/abs/2603.02556), [mDPO](https://arxiv.org/abs/2406.11839) 的对比

VC-STaR: contrast visually similar VQA pairs to improve VLM reasoning. mDPO: augment DPO with image-side preference term. 

关键区别: 这些方法都是 **fix context, prefer one response over another**. CONTEXTRL 反过来: **fix (Q, A), prefer one context over another**. 这是 orthogonal axis — 之前的工作 teach model "given this image, this answer is better than that", CONTEXTRL teach model "given this answer, this image is the supporting one".

这个 flip 很深刻. 它把 grounding 从"output 是否对" 变成 "input evidence 是否被 identify". 这种 input-side supervision 在 representation learning 里有 [CLIP](https://arxiv.org/abs/2103.00020) 的精神 — contrast 不同 modality 的 alignment.

### 7.3 与 [FILM](https://arxiv.org/abs/2404.16811), [LongRLVR](https://arxiv.org/abs/2603.02146) 的对比

FILM: information-intensive supervision for long-context retrieval. LongRLVR: 在 long-context RL 里加 context rewards. 

CONTEXTRL 的差异: 它不直接在 long-context task 上加 reward, 而是用 contrastive auxiliary task 来 implicitly 教 grounding. 这种 indirect supervision 的优势是 modality-agnostic — 同一个 $\mathcal{L}_{\mathrm{CA}}$ 对 trajectory 和 image 都适用.

### 7.4 与 [MemOCR](https://arxiv.org/abs/2601.21468) 的对比

MemOCR 用 memory + layout-aware compression 保留 sparse but decisive evidence. 这是 inference-time 的 context engineering, CONTEXTRL 是 training-time 的 capability shaping. 两者完全可以 combine — CONTEXTRL 训出来的 model 在压缩后的 context 上应该 grounding 更准.

### 7.5 关于 $\lambda$ 的 ablation 的 intuition

Table 10 和 Figure 9 都显示 inverted-U: $\lambda$ 太小 (0.001) signal 不够强, $\lambda$ 太大 (0.01) auxiliary signal 压过 GRPO. 

这个 trade-off 跟 multi-task learning 里的 task balancing 是同一类问题. 参考 [GradNorm](https://arxiv.org/abs/1711.02257) 和 [Uncertainty Weighting](https://arxiv.org/abs/1705.07115), 一个自然的 future direction 是用 dynamic $\lambda$ scheduling 让两个 loss 的 gradient magnitude 自动 balance.

### 7.6 关于 Generative Editing 的潜在 risk

Paper 用 Nano Banana 2 做 generative editing, GPT-5.4 做 verifier. 这里有几个 concern:

1. **Editor 和 verifier 的 distribution gap**: Nano Banana 2 生成的 artifact 可能 GPT-5.4 看不出来 (因为两者 training distribution 不同). 一个 stronger check 是用 human verifier 做 spot check.

2. **Edit 的 semantic plausibility**: localized edit 改了 answer-relevant region, 但周围 context 是否还 consistent? 比如 chart 里改一个 bar 的高度, axis label 是否还 match? Paper 的 criterion (iii) 检查这个, 但实际执行可能有 edge case.

3. **Distribution shift from training to eval**: 12 个 eval benchmark 都是 natural unedited images. 如果 model 学到 "edited image 有特定 artifact", 在 natural image 上可能 transfer 不好. 但 Table 2 的结果证明 transfer 是 work 的, 这间接说明 model 学到的是 content-level grounding 而不是 artifact detection.

### 7.7 关于 Agentic Setting 的 Catastrophic Collapse

DA-SFT 在 Qwen3-8B 上 resolve rate 直接降到 0.00. 这个现象非常 striking. 我的 hypothesis:

Agentic coding 的 policy distribution 是高度 specialized 的 — model 需要在 long-horizon, multi-turn, tool-use 的 format 下 maintain 特定行为 pattern (何时 explore file, 何时 edit, 何时 test). SFT on short selection examples 把 model 推到完全不同的 distribution (single-turn, short-answer), 这种 distribution shift 在 long-horizon setting 下是 catastrophic.

这跟 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 里观察到的 "SFT 冷启动后 RL 才能发挥" 是相反的现象 — R1 是 SFT 提供 cold start, RL refine. 但在已经 specialized 的 agentic model 上, 任何 short-format SFT 都是 destructive.

这也解释了为什么 multimodal setting 下 DA-SFT 没有 collapse — multimodal 是 single-turn short-answer, format mismatch 不严重.

### 7.8 关于 Contrastive Data 的 Efficiency

最 striking 的数字: 1k agentic pairs + 7k multimodal pairs, 加上 standard 7k/38k task data, 就能拿到 consistent +2% 提升. 这是极高的 data efficiency.

对比 [DeepSWE](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33) 和 [SWE-RL](https://arxiv.org/abs/2502.18449), 这些方法 scale RL data 到几十万量级. CONTEXTRL 反其道而行 — 用极少量 high-quality contrastive data + auxiliary objective, 就能 beat larger scale. 

这暗示一个重要 insight: **很多 capability gap 不是 data 量不够, 是 supervision signal 的 form 不对**. Outcome reward 是 dense in quantity but sparse in information — 一万个 trajectory 只告诉 model "对/错". Contrastive pair 哪怕只有一千个, 每个 pair 都 dense in information — 显式指出 "这两个 context 的差异是决定性的".

### 7.9 关于 OOD Transfer 的 Root Cause

Table 1 里 LongBench v2 Long +4.6%, NIAH +5.8% (Klear base) — 这些是 pure long-context retrieval task, 跟 agentic coding 完全不同 domain. 为什么 CONTEXTRL 能 transfer?

我的 hypothesis: contrastive context selection 训练的是一种 meta-skill — "scan context, identify supporting evidence". 这个 meta-skill 在任何 long-context task 上都有用, 不管 context 是 trajectory 还是 document.

这跟 [In-context Learning](https://arxiv.org/abs/2306.15507) 的 meta-learning 视角一致 — ICL 本质是 model 学会了 "given examples, infer pattern". CONTEXTRL 教的是 "given context, identify relevant evidence", 这是一种更 fine-grained 的 ICL skill.

### 7.10 与 RLHF Reward Hacking 的关系

[Skalse et al. on Reward Hacking](https://arxiv.org/abs/2204.06574) 和 [Reward Hacking in RLHF](https://arxiv.org/abs/2402.19416) 都指出: outcome-only reward 容易导致 model 学习 shortcut 而非真实 capability. CONTEXTRL 的 auxiliary loss 本质是 anti-reward-hacking 的 — 它显式 reward grounding behavior, 让 shortcut path 拿不到 auxiliary reward.

这跟 [Constitutional AI](https://arxiv.org/abs/2212.08073) 的 spirit 类似 — 用 explicit principle (这里是 context grounding) 来 constrain reward signal. 但 CONTEXTRL 更轻量, 不需要额外的 critique model.

### 7.11 关于 Future Direction 的一些 Speculation

1. **Multi-context selection**: 现在是 binary (A/B), 可以扩展到 N-way selection. 类似 [InfoNCE](https://arxiv.org/abs/1807.03748) 的 NCE loss, 更强 discrimination signal.

2. **Hierarchical context**: 现在 context 是 flat 的 trajectory 或 image. 可以 decompose 成 hierarchical chunks (function-level, file-level, repo-level), 让 model 学 hierarchical grounding.

3. **Self-generated contrastive pairs**: 现在 data 构造依赖 GPT-5.4 和 Nano Banana 2. 可以让 model 自己 generate contrastive pairs — 类似 [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020). 这会大大降低 data 构造成本.

4. **Active context selection**: 训练 model 不仅选 context, 还主动 query 相关 context — 类似 [ReAct](https://arxiv.org/abs/2210.03629) 但更 grounded. 这能 bridge agentic tool use 和 context grounding.

5. **Adversarial contrastive**: 让另一个 model 主动 generate hard negatives 来 challenge current model — 类似 [GAN](https://arxiv.org/abs/1406.2661) 的 adversarial training. Curriculum 难度自动 adjust.

### 7.12 关于 $\mathcal{L}_{\mathrm{CA}}$ 的 Clip 设计的更深层 Intuition

Clip 操作 $\mathrm{clip}(\Delta_\theta(z), -c, c)$ 看起来只是数值稳定, 但其实有更深的含义.

考虑两种 regime:
- **Early training**: $\Delta$ 接近 0 (model 还没学会区分), gradient 最大, auxiliary signal 最强
- **Late training**: $\Delta$ 远超 $c$ (model 已经 well separate), gradient 变 0, auxiliary signal "退出"

这是一种 **automatic curriculum decay** — auxiliary task 在 model 需要时 active, 在 model 掌握后 fade out, 把 training budget 让给 primary task. 

对比 standard multi-task learning 的 fixed $\lambda$, 这种 clip-based decay 更优雅. 类似 [Layer-wise Adaptive Rate Scaling](https://arxiv.org/abs/1904.05960) 的 adaptive gradient idea.

### 7.13 关于 Inference Cost 的 Implication

CONTEXTRL 的 $\mathcal{L}_{\mathrm{CA}}$ 用 teacher forcing, 不需要 sampled rollout. 这意味着 contrastive data 的 training cost 远低于 task data 的 rollout cost. 

Table 9 显示 multimodal Qwen3-VL-8B 总共 ~288 GPU-hours on 4× H200. 对比 [PAPO](https://arxiv.org/abs/2507.06448) 和 [Vision-R1](https://arxiv.org/abs/2503.06749) 这类纯 RL 方法, CONTEXTRL 的 compute 是非常经济的.

这暗示一个 scaling law: **auxiliary contrastive loss 的 marginal cost 随 model scale 的增长是 sublinear 的** (因为 teacher forcing 不随 model size scale rollout cost), 而 task rollout cost 是 superlinear 的. 所以 model 越大, CONTEXTRL 的 cost advantage 越明显.

## 八、Paper 的 Limitations 和我的 Critique

Paper 自己承认的 limitation: 只在 <10B model 上验证, 主要在 Qwen family. 我额外加几个 concern:

1. **Contrastive pair 的 diversity**: 1k agentic + 7k multimodal 的量虽然 efficient, 但可能 coverage 不够. 比如 agentic 只覆盖 SWE-smith 的 Python repo, 其他语言 / 其他 agent task type (data analysis, web browsing) 如何?

2. **Selection format vs free-form grounding**: CONTEXTRL 训练的是 multiple choice format 的 selection. 实际 deployment 时 model 是 free-form generation, 这两个 format 之间有 gap. Paper 证明 transfer 是 work 的, 但 mechanism 不完全 clear.

3. **Contrastive pair 的 stability**: GPT-5.4 verifier 的判断本身有 noise. Paper 报 ~65% 和 ~97% 的 rejection rate, 但 remaining 35% / 3% 里可能还有 false positive. 一个 stronger validation 是用 human spot check estimate false positive rate.

4. **Long-term retention**: paper 只 report 训练后的 immediate performance. Auxiliary skill 是否会随 further training (e.g., subsequent SFT or RL) 而 forget? 这个 retention property 对 deployment 很重要.

5. **Composition with other auxiliary losses**: CONTEXTRL 只加一个 $\mathcal{L}_{\mathrm{CA}}$. 如果同时加 multiple auxiliary losses (e.g., coherence loss, safety loss, grounding loss), 它们如何 interact? Multi-auxiliary 的 balancing 是 open problem.

## 九、总结

这篇 paper 的核心 contribution 不是某个具体的技术 trick, 而是一个 **conceptual reframe**: 把 context grounding 从 implicit outcome 显式化为 verifiable auxiliary task. 

它的 elegance 在于:
- **Modality-agnostic**: 同一个 $\mathcal{L}_{\mathrm{CA}}$ 对 trajectory 和 image 都适用
- **Data-efficient**: 1k/7k contrastive pairs 就够
- **Compute-efficient**: teacher forcing 避免 expensive rollout
- **Stable**: clip + bounded $\lambda$ 避免 policy collapse
- **Transferable**: OOD 提升 > ID 提升, 说明学到的是 general skill

它揭示了 RL post-training 的一个 blind spot: outcome reward 能教 model "答对", 但教不了 "为什么答对". Contrastive context selection 显式 supervise 这个 "为什么", 把 grounding 从 emergent property 变成 explicit training signal.

这跟 (Karpathy) 我在 [NanoGPT](https://github.com/karpathy/nanoGPT) 和 [Zero to Hero](https://karpathy.ai/zero-to-hero.html) 系列里一直强调的 "understand the mechanism, not just the metric" 是同一个 philosophy. Benchmark 数字会骗人, 但 controlled contrastive probe 不会 — 它强迫 model 证明自己真的 grounded.

希望这些分析对你 build intuition 有帮助. 相关 reference:

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [SimCSE](https://arxiv.org/abs/2104.08821)  
- [InfoNCE / CPC](https://arxiv.org/abs/1807.03748)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Reward Hacking survey](https://arxiv.org/abs/2204.06574)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [PAPO](https://arxiv.org/abs/2507.06448)
- [mDPO](https://arxiv.org/abs/2406.11839)
- [VC-STaR](https://arxiv.org/abs/2603.02556)
- [SWE-RL](https://arxiv.org/abs/2502.18449)
- [DeepSWE](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33)
- [LongRLVR](https://arxiv.org/abs/2603.02146)
- [FILM](https://arxiv.org/abs/2404.16811)
- [Turpin et al. on unfaithful CoT](https://arxiv.org/abs/2305.04388)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [GradNorm](https://arxiv.org/abs/1711.02257)
- [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [SWE-smith](https://arxiv.org/abs/2504.21798)
- [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)
- [Qwen2.5-VL technical report](https://arxiv.org/abs/2502.13923)
- [SkyRL](https://arxiv.org/abs/2504.28839)
- [EasyR1](https://github.com/hiyouga/EasyR1)
- [LMMs-Eval](https://arxiv.org/abs/2407.12772)
- [VLMEvalKit](https://arxiv.org/abs/2407.11691)
