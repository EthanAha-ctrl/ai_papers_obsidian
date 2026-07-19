---
source_pdf: AgentArk Distilling Multi-Agent Intelligence into a Single LLM Agent.pdf
paper_sha256: 4ede1aa3d445b14cb71ae1a853ee90ba4f2186748afdbc2e87c2f157c2a78cf3
processed_at: '2026-07-18T04:48:21-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AgentArk 深度解析：把 Multi-Agent 辩论蒸馏进单个 LLM

这篇 paper 来自 CMU / Georgia Tech / William & Mary / Amazon / UBC 的合作,核心 thesis 很 Karpathy-style:**inference-time compute 可以被 shift forward 成 training-time compute**。Multi-agent debate 在 inference 时产生的 reasoning dynamics(自我批判、迭代纠错、多视角交叉验证),其实可以"内化"进单个 model 的 weights,部署时只需要一次 forward pass。

参考链接:
- Paper repo: https://github.com/AIFrontwardLab/AgentArk
- 同类思想背景 Du et al. 2023 (LLM debate): https://arxiv.org/abs/2305.14325
- GRPO 原文 DeepSeekMath: https://arxiv.org/abs/2402.03300
- PRM "Let's Verify Step by Step": https://arxiv.org/abs/2305.20050

---

## 1. 核心动机：MAS 的 double-edged sword

MAS(Multi-Agent System)通过 debate / critique / consensus 取得了强 reasoning 性能,但有两个结构性问题:

**Computational overhead**:在 densely-connected 网络里,计算量随 agent 数量近似 quadratic 增长。5 个 agent × 3 轮 debate = 至少 15 次 forward pass + 同步开销;20 agents 时,这变成上百次 invocation。对实时部署是灾难。

**Error propagation / vulnerability amplification**:He et al. 2025 和 Nguyen et al. 2025 的工作指出,在高密度交互里,单个 agent 的 hallucination 或 bias 会通过 debate 网络传播并被 amplify,导致 collective failure。MAS 不只是 correct errors,也会 escalate them。

关键问题被 paper 抛出:
> Can a single model internalize the reasoning benefits of MAS without their high inference-time cost and collaborative vulnerabilities?

这让我想到 DeepSeek-R1 的纯 RL reasoning 训练:它们也是把 test-time CoT 的能力固化进 weights。但 AgentArk 的 setup 更具体——它要把 *inter-agent* 的 dialectical dynamics(distillation 的源不是单 agent 的 thinking,而是 agent 之间的 disagreement-and-resolution)压缩进单 model。

---

## 2. 一个关键的 empirical insight:MAS 的真正增益在哪

Paper 引用了 Kim et al. 2025b 和 Ke et al. 2026 的发现,这是整个工作的理论基础:

> Removing or perturbing explicit agent structures leads to only marginal performance degradation... the essential contribution of MAS lies in the reasoning dynamics they induce, rather than in the interaction schema itself.

换句话说,agent topology(谁连谁、几轮、什么角色)只是 *first-class citizen*,真正起作用的是它 *induce* 出来的 reasoning behavior:自我批判、寻找逻辑漏洞、假设修正。这跟 single-model CoT + self-refine 在 surface 上等价,只要 single model 学会这些 *behavior pattern*。

这个 insight 让我想到 AlphaGo 的 self-play:你最终要的是 policy network 这一个 network 的 weights,但训练时多 agent / 自我对弈只是产生 gradient 信号的 *scaffold*。AgentArk 就是把 multi-agent debate 当作 training scaffold,最终 student 是单 model。

---

## 3. Pipeline 三阶段总览

```
[Phase 1: Data Generation]       [Phase 2: Knowledge Extraction]   [Phase 3: Distillation]
   n=5 agents, K=3 rounds  →     correctness filtering  →           RSFT / DA / PAD
   debate logs L_x                diverse trajectory selection       (SFT or PRM+GRPO)
```

### Phase 1: Multi-Agent Debate 数据生成

对每个 input $x$,初始化 $n$ 个 agents $\mathcal{A} = \{a_1, ..., a_n\}$,共享同一 LLM backbone 但独立 context。$K$ 轮交互,round $k$ 里 agent $a_i$ 生成 reasoning trace $\tau_{i,k}$,条件是 problem $x$ 和 peers 的上一轮 traces $\{\tau_{j,k-1}\}_{j \neq i}$。

辩论日志:
$$\mathcal{L}_x = \{\tau_{i,k} \mid i \in [1,n], k \in [1,K]\}$$

这里 $i$ 是 agent index(下标),$k$ 是 round index(下标)。这等价于在 tree-of-thought 上施加横向通信。

### Phase 2: Correctness-First Trajectory Selection

关键设计:**不挑 error-free 的 path,反而优先挑 corrective trajectories**——agent 一开始答错,被 peer critique 后 pivot 到正确答案。用 Qwen2.5-72B-Instruct 作为 verifier,只判答案对错不看推理过程,定义:

$$\mathcal{A}_{correct} = \{a \in \mathcal{A} \mid \text{Answer}(a, x) \text{ verified correct}\}$$

若 $|\mathcal{A}_{correct}| < 2$ 就丢弃这个 problem(没有 diversity 可挖)。

这个选择策略让我想到 Constitutional AI 的 critique-revision,以及 STaR(Star bootstrapping)的 rationalization。差别在于 AgentArk 保留的 *corrective arc* 是真实的 inter-agent disagreement 触发的,比 self-generated revision 更 *epistemically diverse*。

### Phase 3: 三种 hierarchical distillation 策略

下面逐一深入。

---

## 4. Reasoning-Enhanced SFT (RSFT)

公式(1):

$$\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,r,y^*) \sim \mathcal{D}} \big[ \mathcal{L}_{res} + \mathcal{L}_{ans} \big]$$

其中:

- $\theta$:student model 参数
- $x$:input problem
- $r = (r_1, ..., r_n)$:从 debate 提取的 reasoning trace 序列
- $y^*$:gold final answer
- $\mathcal{D}$:distillation dataset

两项拆开:

$$\mathcal{L}_{res} = \sum_{t=1}^{|r|} \log p_\theta(r_t \mid r_{<t}, x)$$

$t$ 是 trace 内 token/step index。这一项是 reasoning likelihood。

$$\mathcal{L}_{ans} = \log p_\theta(y^* \mid r, x)$$

这一项保证答案 grounded 在 reasoning 上。

**Intuition**: vanilla SFT 只学 $\mathcal{L}_{ans}$,容易 overfit 到 task-specific mapping(paper 的 Appendix G.1 实证了这点)。RSFT 把 reasoning trace 当作 *explicit supervision target*,逼 student 复现 multi-agent 的中间推理结构。但缺陷是它只是 *imitation*,没有 reward signal 告诉 student *哪些 step 是关键的*。

---

## 5. Data Augmentation via Diverse Extraction (DA)

公式(2):

$$\mathcal{L}_{Aug}(\theta) = -\frac{1}{k} \sum_{i=1}^{k} \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, r_i, x)$$

变量:
- $k \in \{1,2,3\}$:对同一 problem 提取的 diverse trajectory 数
- $i$:trajectory index(下标)
- $t$:answer token index
- $r_i$:第 $i$ 条 reasoning trajectory
- $y_t$:第 $t$ 个 answer token

用 high-capacity teacher LLM 做 "distiller",从 $\mathcal{A}_{correct}$ 里挑出满足 (1) correct,(2) structurally diverse 的 trace。diversity 的判据是 prompt 里指定的:不同的 mathematical identity、不同的 logical heuristic、不同的 starting assumption。

**Intuition**: 这其实是 *one-to-many distillation*——把同一 answer 的多个解法路径塞进 student,提升 robustness。和 GAN 的 mode coverage、model ensemble 的 variance reduction 同源。但 paper 实验显示 DA 的 gain 不稳定(高 variance),原因可能是 student 容量不够装下过多异质 reasoning style,反而引入噪声。

---

## 6. Process-Aware Distillation (PAD) —— 重头戏

PAD 把 distillation 当作 RL problem,用 Process Reward Model 提供 step-level 监督,再用 GRPO 优化 student policy。这是 paper 最核心、最 effective 的方法。

### 6.1 PRM 训练(两阶段 curriculum)

PRM $R_\phi$ 初始化自 student weights。

**Stage I - Feature Alignment(backbone frozen)**: 只训 last layer + reward head,避免破坏预训练 linguistic features,reward head 学着把现有 representation 映射到 correctness label $z_t \in \{0,1\}$。

**Stage II - Full Specialization(backbone unfrozen)**: 全网络 fine-tune,让 model 发展出针对逻辑谬误检测的 specialized attention pattern。

PRM loss 设计(Appendix B.1,公式 7)是 **contrastive** 而非标准 BCE:

$$\mathcal{L}_{PRM}(\phi) = -\sum_t \log \mathrm{softmax}\left(\frac{\sigma(R_\phi(r_t^+)), \{\sigma(R_\phi(r^-))\}_{r^- \in N_t}}{\tau}\right)$$

变量解释:
- $\sigma(\cdot)$:sigmoid
- $r_t^+$:第 $t$ 步的 positive step(与 consensus 一致的推理步骤)
- $r^-$:从其他 agent trajectory 采样的 negative step
- $N_t$:第 $t$ 步的 negative set
- $\tau$:temperature,控制 contrastive distribution 的 sharpness
- $\phi$:PRM 参数

**Intuition**: contrastive 比 BCE 强在哪——它学的是 *relative correctness* 而不是 *absolute label*。MAS 的 consensus 本身就是相对的(多个 agent 同意),用 contrastive 更符合数据生成机制。这跟 DPO、RLHF 里的 preference loss 异曲同工:避免 reward hacking,因为 reward 必须跟负例拉开差距才有意义。

### 6.2 GRPO Policy Optimization

公式(3)是 GRPO 的目标:

$$\mathcal{I}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{o_i\} \sim \pi_{old}} \left[\frac{1}{G}\sum_{i=1}^G \mathcal{L}_i(\theta) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right]$$

变量:
- $\theta$:student policy 参数
- $\pi_{old}$:更新前的 student snapshot(behavior policy)
- $\pi_{ref}$:固定的 reference policy,用于 KL regularization
- $\beta$:KL penalty 系数
- $G$:group size,从 $\pi_{old}$ 采样的 output 数量
- $o_i$:第 $i$ 个 sampled output
- $\mathcal{L}_i(\theta)$:第 $i$ 个 output 的 clipped surrogate objective

每个 output 的 surrogate(公式 4):

$$\mathcal{L}_i(\theta) = \min\left(\rho_i(\theta)\hat{A}_i, \mathrm{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right)$$

概率比:

$$\rho_i(\theta) = \frac{\pi_\theta(o_i \mid x)}{\pi_{old}(o_i \mid x)}$$

 Advantage(公式 5):

$$\hat{A}_i = \frac{R_\phi(o_i) - \mu_R}{\sigma_R}$$

- $R_\phi(o_i)$:PRM 对 output $o_i$ 的 step-wise 聚合分数
- $\mu_R, \sigma_R$:group 内 $G$ 个 output 的 PRM 分数均值/标准差

**Intuition**: GRPO 相对 PPO 的关键省略是 **去掉 value function**。Advantage 直接用 group 内 reward 的 z-score 估计。这对 reasoning task 特别合适:
- reasoning 的 step space 巨大,value function 难训准确
- group-relative baseline 自动适配每个 problem 的难度(简单的 problem 整组 reward 都高,normalization 后差异主要在 step quality)
- 跟 AlphaGo 的 leave-one-out baseline、or REINFORCE with batch mean baseline 思想一致

### 6.3 PPO vs GRPO 的 ablation

Table 5 显示 PPO 略微占优(53.10 vs 52.61 avg on GSM8K distillation),但 GRPO 大幅省 compute(无 value network)。这个 trade-off 跟 DeepSeekMath 原文结论一致:GRPO 在大规模 reasoning RL 上是更 scalable 的选择。

---

## 7. 实验结果的关键 Insights

### 7.1 主结果

- AgentArk 让 single agent 平均提升 4.8%,仅 slightly worse than vanilla MAS
- ID(in-distribution)提升 30% max / 4-6% avg;OOD 提升 7% max / 1-3% avg
- PAD 在所有 setup 下都稳定有 gain;RSFT 和 DA 不稳定,因 dataset 波动

### 7.2 PRM capacity > student capacity

Table 4 是最 insightful 的 ablation 之一:

| PRM | Policy | GSM8K avg |
|-----|--------|-----------|
| 0.6B | 0.6B | 42.63 |
| 8B  | 0.6B | 42.63 |
| 0.6B | 8B  | 88.52 |
| 8B  | 8B  | 88.59 |

注意:**PRM 大小几乎不影响最终 policy 性能,policy 大小决定一切**。这反直觉——直觉上更强的 PRM 应该提供更好的 step supervision,但实验显示 PRM 即使是 0.6B,只要 policy 是 8B,性能跟 8B PRM 几乎一样。

这暗示 step-level supervision 的 *信号* 在小 PRM 上已经足够 *qualitatively correct*,policy 的 capacity 才是 bottleneck——policy 要有能力 *执行* PRM 引导的 reasoning 行为。这跟 Chinchilla 的 compute-optimal scaling 思路相通:数据/信号 quality 的边际收益很快被 model capacity 限制。

### 7.3 Scaling agents:小 student 不受益于更多 teacher

Figure 4:
- Qwen3-0.6B(student):5 → 10 → 20 agents,性能不升反降
- Qwen3-8B(student):scaling 持续受益但有 diminishing returns

**Intuition**: 小 student 的 representation capacity 不够 absorb 过度 diverse 的 teacher signal。这跟 distillation 经典发现一致——student 容量决定它"消化"多少 teacher 的 distribution。在 KD 文献里这叫 *capacity mismatch*。

### 7.4 Data quality > quantity

Figure 5:RSFT 和 DA 在 data scaling 时高 variance,有时 data 越多反而越差。PAD 跨 data scale 稳定。

这印证一个直觉:multi-agent 生成的 trace 噪声高(部分 agent 推理质量差),简单堆 data 等于堆噪声。PRM 的 *process-level filtering* 起到了 denoising 作用,只让 high-signal 的 step 进入 gradient。

### 7.5 Reasoning quality 的细粒度评估

Table 1 用 InternLM-2.5-20b 做自动评估,四个维度:

| Metric | Single | RSFT | DA | PAD |
|--------|--------|------|----|----|
| Step Decomposition | 2.75 | 3.13 | 3.38 | 3.23 |
| Intermediate Verification | 2.41 | 3.48 | 4.04 | 4.07 |
| Error Localization | 1.97 | 2.19 | 2.91 | 2.78 |
| Reasoning Coherence | 1.88 | 2.25 | 3.07 | 3.96 |

PAD 在 Reasoning Coherence 上碾压(3.96 vs DA 的 3.07)。但有趣的是 DA 在 Step Decomposition / Intermediate Verification 上略高于 PAD——说明 DA 学到的是 *显式结构化* 的 reasoning,pad 学到的是 *隐式连贯* 的 reasoning。

我直觉这反映 DA 是 SFT-based,直接模仿结构化 trace,而 PAD 是 RL-based,学到的 reasoning style 更"内化"——不一定显式标 "Step 1, Step 2",但逻辑流更紧密。

### 7.6 Robustness:TruthfulQA

Table 2:PAD 在 BLEU / ROUGE-1/2/L 上全面提升(ROUGE-2 从 0.5704 → 0.6414),说明 MAS distillation 不仅提升 accuracy,还提升 factual robustness,且不发生 catastrophic forgetting。

### 7.7 Open-ended OOD generalization

Figure 6:仅用 GSM8K(math)训练,在 HotpotQA(multi-hop QA)、QASPER(long-context)、QMSum(summarization)上 OOD 测试。8B 模型显著受益——Qwen3-8B 在 QMSum 上 F1 从 14.94 → 17.82。

这是我最喜欢的结果:它说明 reasoning 是 *modality-agnostic / task-agnostic capability*,可以从 math 迁移到 summarization。这跟 "reasoning as a general skill" 的假说一致——也是 OpenAI o1 / DeepSeek-R1 的核心赌注。

### 7.8 Multimodal 扩展

Figure 7:把 text-only MAS reasoning 蒸馏到 Qwen2.5-VL-3B-Instruct。即使没在 MLLM reasoning 上训过,PAD 仍带来稳定 gain。说明 *reasoning pattern* 是 modality-agnostic 的——一旦内化,可以跨 modality 复用。这跟 LLaVA、Qwen-VL 把 LM reasoning 能力桥接到 vision 的 philosophy 一脉相承。

---

## 8. 与相关工作的脉络

### 8.1 Distillation 谱系
- **传统 KD**(Hinton 2015):softmax temperature matching,soft label
- **CoT distillation**(Magister et al.):把 teacher CoT 蒸到 student
- **Multi-agent distillation**: MAGDI(Chen 2024b,graph-based interaction),Zhou 2025(debate-derived preference)
- **AgentArk 的位置**:在 process level 蒸馏,不依赖 task-specific agent design,framework-agnostic

### 8.2 Reasoning RL 谱系
- **STaR**(Zelikman 2022):self-taught reasoner,rationalization
- **Process Reward Models**(Lightman 2023,Math-Shepherd):step-level verifier
- **Constitutional AI**(Bai 2022):critique-revision
- **DeepSeek-R1 / GRPO**:group-relative policy optimization for reasoning
- **AgentArk**:把 PRM + GRPO 应用到 multi-agent debate 的 corrective traces 上

### 8.3 Test-time compute 谱系
- **Tree of Thoughts**(Yao 2023)
- **Self-Consistency**(Wang 2022)
- **Multi-agent debate**(Du 2023)
- **OpenAI o1 / DeepSeek-R1**:internalized test-time compute
- **AgentArk**:explicitly 把 MAS 的 test-time compute shift 到 training time

---

## 9. Paper 的局限和我的批判

### 局限 1:只测了 debate 一种 MAS
Paper 自己承认这点。Debate 是高度结构化的 MAS,其他范式(如 ReAct + tool use、hierarchical agent、AutoGen 的 conversation pattern)可能 induce 不同的 reasoning dynamics,不一定都能被同样方式蒸馏。

### 局限 2:OOD gain 仍小
OOD 1-3% avg 提升虽然显著但绝对值小。若目标是 general reasoning lift,可能需要更大规模 / 更 diverse 的 training mixture。

### 局限 3:PRM 训练成本
PAD 需要 8×H100 × 20h(Table 6),对比 RSFT 的 1×H100 × 6h 贵很多。对于 resource-constrained lab,RSFT + DA 的组合可能更实用。

### 局限 4:Correctness filtering 的 bias
用 Qwen2.5-72B 做 verifier,verifier 自身的 bias 会传递到蒸馏数据。Paper 没讨论 verifier error 的影响。

### 局限 5:对 reasoning 的定义偏向 math
评估主要在 math / MedMCQA,reasoning 在 creative writing、code debugging、scientific hypothesis formation 上的表现没测。

---

## 10. 我的 Intuition Building

读完这篇 paper,我形成几个 mental model:

**Mental Model 1: MAS as data augmentation scheme, not architecture**
MAS 不是部署时架构,而是训练时的 *trajectory generator*。它的价值在产生 *diverse, corrective, multi-perspective* 的 reasoning trace,这些 trace 是 high-signal supervision source。一旦数据生成完毕,MAS 就可以丢弃。

**Mental Model 2: Process supervision > Outcome supervision**
RSFT(outcome + reasoning trace)不如 PAD(step-level PRM),根本原因是 step-level signal 让 student 学到 *what makes a step good*,而不只是 *what sequence leads to correct answer*。这跟 supervised RL 里 dense reward >> sparse reward 的道理一致。

**Mental Model 3: Capacity decides everything**
小 student 不能 absorb 大 teacher 的全部智慧。这跟 LoRA / quantization 经验一致:在 capacity-constrained regime,数据 quality 和 regularization 比数据 quantity 重要。

**Mental Model 4: Reasoning is a transferable skill**
GSM8K 上训的 reasoning 能迁移到 QMSum summarization,这跟"reasoning 是 general capability"的假说一致。下一步实验应该测 code generation、logic puzzle、planning 等更多 reasoning type。

**Mental Model 5: Debate dynamics ≈ self-critique dynamics**
如果 MAS 的 gain 主要来自 reasoning dynamics 而非 topology,那 single model 只要学会 *self-critique + revision* 就能复现大部分 gain。这预示着 future work 可以用更轻量的 self-generated critique 替代 expensive multi-agent debate 来生成蒸馏数据,类似 Self-Rewarding LM 或 Self-Refine 的思路。

---

## 11. 与 Karpathy 自己工作的关联

Andrej 你会注意到这跟几个你提过的 idea 高度相关:

1. **"Software 2.0"**:MAS 的 explicit agent orchestration 是 Software 1.0(imperative code),蒸馏后变成 Software 2.0(learned weights)。这正是 Software 1.0 → 2.0 迁移的实例。

2. **"Shift test-time compute to training"**:你在播客和 talk 里多次提过这个 idea,AgentArk 是其中一种具体实现路径。

3. **Eureka Labs / education analog**:MAS debate 像班级讨论,蒸馏像把班级讨论的智慧内化到单个学生。Paper 引用 Curşeu 2015 / Navajas 2018 关于 group-to-individual transfer 的认知科学研究,这个认知科学背景很有趣。

4. **"LLM OS" / agentic future**:如果未来是 single-model-as-OS-agent,AgentArk 这种把 multi-agent coordination 内化的路径是关键 enabler——你不能在 inference 时跑 5 个 agent debate 来写一行代码。

---

## 12. 未来方向猜想

基于 AgentArk 暴露的问题,我猜 future work 会往这些方向走:

1. **Adaptive distillation**:根据 problem difficulty 动态选择 distillation strategy。简单题用 RSFT,难题用 PAD。

2. **Hierarchical PRM**:不同 reasoning type(math / code / logic)用不同 PRM head,modular supervision。

3. **Cross-architecture distillation at scale**:把 GPT-4 / Claude debate 蒸馏到 Qwen / Llama。Paper 已经做了 Qwen3-32B → Gemma-7B 跨家族蒸馏,但 scale 还可以更大。

4. **Tool-use / agentic distillation**:MAS 不只 debate,还有 tool use。把 ReAct + tool debate 蒸馏进 single agent 是自然的下一步,也跟 Zhao et al. 2024 "Distill embodied agent into single model" 同向。

5. **Online distillation with self-play**:让 student 自己 generate debate data,持续 self-improve。这跟 AlphaGo self-play、Constitutional AI 的 self-improvement 同源。

6. **Theoretical analysis**:为什么 PRM capacity 不影响最终性能?为什么小 student 不受益于更多 agents?这些 empirical finding 需要理论解释,可能跟 information bottleneck、PAC-Bayes generalization bound 相关。

7. **Multimodal full-stack distillation**:Paper 只测了 text-only reasoning → MLLM 的迁移。下一步应该让 MAS 在 multimodal(图文混合)上 debate,蒸馏 MLLM 的 multimodal reasoning。

---

## 13. 一句话总结

AgentArk 把 multi-agent debate 从 *inference-time architecture* 转化为 *training-time data generation scaffold*,通过 PRM + GRPO 把 inter-agent dialectical dynamics 内化进 single model 的 weights,实现 test-time efficiency 和 reasoning capability 的双赢。它的核心 contribution 不是新算法,而是 *systematic empirical study* + *clear framework* 揭示了 MAS distillation 的关键 trade-off:reasoning quality > quantity,PRM process signal > outcome signal,student capacity 是终极 bottleneck。

参考阅读:
- Du et al. 2023 "Improving Factuality and Reasoning through Multiagent Debate": https://arxiv.org/abs/2305.14325
- Shao et al. 2024 "DeepSeekMath GRPO": https://arxiv.org/abs/2402.03300
- Lightman et al. 2023 "Let's Verify Step by Step": https://arxiv.org/abs/2305.20050
- Chen et al. 2024b "MAGDI": https://arxiv.org/abs/2402.01620
- Liang et al. 2024 "Encouraging Divergent Thinking through Multi-Agent Debate": https://aclanthology.org/2024.emnlp-main.1085/
- Kim et al. 2025b "Towards a Science of Scaling Agent Systems": https://arxiv.org/abs/2512.08296
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Constitutional AI: https://arxiv.org/abs/2212.08073
- STaR: https://arxiv.org/abs/2203.14465

如果你想 dive deeper 到某个具体方面(比如 PRM 的 contrastive loss 设计选择、GRPO 的 variance 分析、或 cross-family distillation 的 representation mismatch),我可以进一步展开。
