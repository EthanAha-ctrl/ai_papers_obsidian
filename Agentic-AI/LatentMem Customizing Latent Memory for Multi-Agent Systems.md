---
source_pdf: LatentMem Customizing Latent Memory for Multi-Agent Systems.pdf
paper_sha256: 33ec4bd63359ac152d75d03e49b3bf44956aac9d9b1126b09dc467850b1fffa2
processed_at: '2026-08-05T12:23:36-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LatentMem 用人话版

嗨 Andrej，我再把这篇 paper 用大白话给你捋一遍，重点放在 **intuition** 和 **为什么这么设计**，技术细节该讲的还是讲，但用更顺的语言。

---

## 一句话版

现有 multi-agent system 的 memory 都是用文字存的，大家共享一份历史记录，所有 agent 看到同样的东西，token 还越堆越多。LatentMem 干的事儿：用一个 small transformer 把历史 trajectory 压成 **8 个 latent vector**，按每个 agent 的角色定制，prepend 到 hidden state 上，再用 GRPO 直接拿 task reward 训这个 small transformer。结果是更省 token、更 robust、还能 transfer 到没见过的 domain 和 MAS framework。

---

## 现有 MAS Memory 为啥不行：两个痛点

### 痛点 1：Memory Homogenization（大家吃同一碗饭）

你想想 MetaGPT、ChatDev、OAgents、G-Memory 这些设计——它们给所有 agent **同一份 memory**。User proxy agent 看到的历史记录跟 Code agent 看到的历史记录是一样的。这就像让产品经理和程序员看同一份 Git log，两个人需要的 abstraction level 完全不同，但 memory system 不管这个。

Cemri et al. 2025 ([link](https://arxiv.org/abs/2503.13657)) 实证说这会导致 **correlated errors**——agent 们因为看了同样有偏的历史，错也错到一块儿去。

### 痛点 2：Information Overload（token 爆炸）

G-Memory 这种 multi-granularity memory 设计，把 trajectory 拆成三层 graph（Query Graph + Insight Graph + Interaction Graph，[link](https://arxiv.org/abs/2506.07398)）。听起来很 fancy，实际 retrieve 多了 context 直接被淹没。Paper 在 Appendix C.3 Figure 10 实测：G-Memory 在 top-K 从 3 涨到 5 时，KodCode 准确率从 72.95% 掉到 70.25%。**retrieve 越多反而越差**——典型的 information overload。

MIRIX ([link](https://arxiv.org/abs/2507.07957))、EvolveR ([link](https://arxiv.org/abs/2510.16079))、Agent KB ([link](https://arxiv.org/abs/2507.06229)) 都是同一类问题：symbolic memory 不可避免地膨胀。

---

## LatentMem 的核心 Idea：把 Memory 搬到 Latent Space

### Architecture 的大白话

三个组件：

**1. Experience Bank $\mathcal{B}$：只存 raw trajectory，啥也不蒸馏**

这是 paper 的一个 philosophical choice。作者明确引用 Sutton 的 "Bitter Lesson" ([link](http://www.incompleteideas.net/IncIdeas/BitterLesson.html))——scalable system 应该靠 general learning mechanism，不靠 hand-crafted knowledge。

每条 trajectory $\tau = \{(\alpha_j, p_j, o_j)\}_{j=1}^{H}$：
- $\alpha_j$：第 $j$ 步是哪个 agent 在动
- $p_j$：那一步的 input prompt
- $o_j$：那一步的 output
- $H$：trajectory 总步数

Retrieval 用 MiniLM embedding + cosine similarity，公式 (3)：

$$\mathcal{T}_q = \text{top-}K_{\tau_i \in \mathcal{B}}(\text{sim}(\mathbf{v}(q), \mathbf{v}(\tau_i)))$$

- $\mathbf{v}(\cdot)$：MiniLM encoder（all-MiniLM-L6-v2，[link](https://arxiv.org/abs/2002.10957)）
- $\text{sim}$：cosine similarity
- $K$：retrieved trajectory 数量，主实验 K=1

为啥 K=1 这么保守？我猜是 efficiency 优先。但 ablation Figure 10 显示 K 涨到 5 还涨点——说明 composer 有能力从冗余 trajectory 里榨信息，不像 G-Memory 那样崩。

**2. Memory Composer $\sigma_\phi$：核心组件**

公式 (5)：

$$m_j = \sigma_\phi(\gamma_{\alpha_j}, \mathcal{T}_q) \in \mathbb{R}^{L' \times D}$$

- $\sigma_\phi$：memory composer，一个 small transformer，**用 backbone LLM 初始化 + LoRA 训练**（[LoRA link](https://arxiv.org/abs/2106.09685)）
- $\gamma_{\alpha_j}$：当前 active agent 的 role profile（文本，比如 "You are a code implementation agent..."）
- $\mathcal{T}_q$：retrieved raw trajectories
- $L'$：latent memory 固定长度，paper 设 $L'=8$
- $D$：backbone LLM 的 hidden dim

**关键**：同一个 trajectory，喂给 user-proxy 的 $\gamma$ 和 assistant 的 $\gamma$，composer 输出的 $m_j$ 不一样。这就是 role-aware。

Paper 用 t-SNE 可视化验证（Figure 4, Appendix C.2 公式 (14)-(15)）：先把 $m_i = (m_{i,1}, \dots, m_{i,L'}) \in \mathbb{R}^{L' \times D}$ mean pooling：

$$\bar{m}_i = \frac{1}{L'} \sum_{l=1}^{L'} m_{i,l}$$

再 t-SNE 降维：

$$y_i = f_{\text{t-SNE}}(\bar{m}_i) \in \mathbb{R}^2$$

结果：user-proxy 和 assistant 的 latent memory 在 in-domain KodCode/AutoGen 上清晰分离两 cluster；OOD BigCodeBench + unseen CAMEL 也分离。**role conditioning 真的学到几何结构了**，而且能 transfer。

**3. Memory Injection：concat 到 hidden state**

公式 (6)：

$$\tilde{\pi}_{\theta_{\alpha_j}}(p_j, m_j) = \pi_{\theta_{\alpha_j}}(\text{concat}(h_j, m_j))$$

- $h_j = (h_j^{(1)}, \dots, h_j^{(L)}) \in \mathbb{R}^{L \times D}$：input prompt $p_j$ 经过 embedding layer 的 hidden states
- $L$：prompt 长度
- $\text{concat}(h_j, m_j) \in \mathbb{R}^{(L+L') \times D}$：latent memory 当 8 个 prefix token
- $\tilde{\pi}$：wrapped policy，对 agent 层完全透明

这跟 prefix-tuning ([link](https://arxiv.org/abs/2104.08691))、soft prompt 思路同源，区别是 prefix 是 **动态从 retrieved trajectory 生成** 的，且通过 RL 训练。

为啥选 concat 到 embedding 而非 prompt token level？因为 **concat 到 hidden state 可微**——RL signal 能穿过这里反传。如果像 RAG 那样把 retrieved text 拼到 prompt 前面，就断了 gradient flow。

---

## LMPO：怎么用 RL 训这个 Composer

### Gradient Flow 的核心 trick

公式 (7)：

$$\mathbb{P}(\tau \mid q, \mathcal{T}_q; \phi, \{\theta_k\}_{k=1}^N) = \prod_{j=1}^{H} \mathbb{P}(o_j \mid p_j, m_j; \theta_{\alpha_j})$$

- $\phi$：composer 参数（要 train）
- $\{\theta_k\}_{k=1}^N$：所有 agent backbone 参数（**frozen**）
- $m_j = \sigma_\phi(\mathcal{T}_q, \gamma_{\alpha_j})$：latent memory，是 differentiable 的

因为 $m_j$ 是 $\sigma_\phi$ 的输出，且 $m_j$ 被 concat 到 hidden state 进入 backbone，所以从 task reward $R(\tau)$ 出发的 gradient 可以 **穿过 backbone forward pass 反传到 $\phi$**，而 $\theta$ 全程 frozen。

这是整个 LMPO 的核心 insight：**不用 retrain backbone，只 train 一个 small composer，让 task reward 信号"借道" latent memory**。

公式 (8) 把 $o_j$ 拆到 token 级别：

$$\mathbb{P}(o_j \mid p_j, m_j; \theta_{\alpha_j}) = \prod_{t=1}^{T} \tilde{\pi}_{\theta_{\alpha_j}}(o_j^{(t)} \mid p_j, o_j^{(<t)}, m_j)$$

- $o_j^{(t)}$：第 $j$ 步 active agent 输出的第 $t$ 个 token
- $o_j^{(<t)}$：autoregressive condition
- $m_j$：作为前缀的 latent memory

### Group-based Advantage（GRPO 风格）

公式 (9)：对每个 query $q$，sample G 条 trajectory：

$$\{\hat{\tau}_i\}_{i=1}^G \sim \mathbb{P}(\cdot \mid q, \mathcal{T}_q; \phi, \{\theta_k\})$$

公式 (10)：组内 normalize 算 advantage：

$$\hat{A}_i = \frac{R(\hat{\tau}_i) - \text{mean}(\{R(\hat{\tau}_i)\}_{i=1}^G)}{\text{std}(\{R(\hat{\tau}_i)\}_{i=1}^G) + \epsilon}$$

- $R(\hat{\tau}_i)$：第 $i$ 条 trajectory 的 reward
- $\epsilon$：防除零

这是 GRPO（[DeepSeekMath link](https://arxiv.org/abs/2402.03300)）的精髓——**不要 critic**。group 内 mean/std 做 baseline 就行。对 memory composer 这种"输出是连续 latent 而非离散 action"的场景非常合适，否则训 critic 得做 latent state value estimation，复杂度爆炸。

### Token-level Clipped Objective

公式 (11)-(13) 是 PPO 风格的 clipped objective，但 token-level：

公式 (12)：

$$\mathcal{L}_{i,j,t}(\phi) = \min\Big(r_{i,j,t}(\phi)\hat{A}_i, \text{clip}(r_{i,j,t}(\phi), 1-\varepsilon, 1+\varepsilon)\hat{A}_i\Big)$$

- $\varepsilon$：clip ratio = 0.2（标准 PPO 值）
- $\hat{A}_i$：来自公式 (10)

公式 (13)：importance sampling ratio

$$r_{i,j,t}(\phi) = \frac{\tilde{\pi}_{\theta}(\ldots \mid \sigma_\phi(\gamma_{\alpha_{i,j}}, \mathcal{T}_q))}{\tilde{\pi}_{\theta}(\ldots \mid \sigma_{\phi_{\text{old}}}(\gamma_{\alpha_{i,j}}, \mathcal{T}_q))}$$

- 分子：当前 composer $\phi$ 生成的 latent memory 算出的 token 概率
- 分母：old composer $\phi_{\text{old}}$（rollout 时版本）算出的 token 概率

**为啥 token-level 而非 trajectory-level**？作者引 DAPO ([link](https://arxiv.org/abs/2503.14476))：trajectory-level 让 long sequence 每个 token 的 gradient 贡献被稀释。MAS trajectory 通常很长（多 agent 多步交互），token-level 才能让 composer 捕捉关键 coordination pattern。

**直觉总结 LMPO**：传统 prefix-tuning 用 LM loss 让 prefix 模仿 demonstration；LMPO 用 task reward 让 latent memory 直接服务最终 task performance。这是把 prefix-tuning 从 imitation 推到 RL 范式。

---

## 实验里最 eye-catching 的数字

### Table 1：跨 6 benchmark × 4 MAS framework

Qwen3-4B backbone：

| Setting | No-Memory | LatentMem | Gain |
|---|---|---|---|
| AutoGen + TriviaQA | 60.31 | 76.51 | **+16.20** |
| AutoGen + PopQA | 38.78 | 52.70 | **+13.92** |
| DyLAN (unseen) + PopQA | 24.89 | 44.25 | **+19.36** |
| AutoGen + PDDL (OOD) | 16.39 | 23.49 | +7.10 |
| CAMEL (unseen) + KodCode | 70.70 | 77.75 | +7.05 |

最 striking 的是 **DyLAN PopQA +19.36%**——debate-style MAS + entity-centric QA 的 long-tail knowledge 任务，latent memory 杠杆最大。

Table 4（Llama-3.1-8B backbone）更强：AutoGen KodCode 从 47.45 拉到 65.90，**+18.45%**。

### vs MARTI（multi-agent fine-tuning baseline）

MARTI ([link](https://github.com/TsinghuaC3I/MARTI)) 是直接用 GRPO fine-tune agent backbone，同 budget、同 data、同 framework：

| MAS + Dataset | MARTI | LatentMem | Diff |
|---|---|---|---|
| AutoGen + TriviaQA | 64.78 | 76.51 | **+11.73** |
| MacNet + KodCode | 73.10 | 78.90 | +5.80 |

关键观察：MARTI 在 AutoGen→MacNet 时 KodCode 掉 1.10（74.20→73.10），说明直接训 backbone 容易 overfit 到特定 MAS topology。LatentMem 反而 +2.10（76.80→78.90）——**memory composer 学到的是 MAS-agnostic 的高阶 pattern**，能 transfer 到不同 topology。

### Cost（Figure 3）

- Token：LatentMem 用 0.01M tokens 比 No-Memory 还少（8 个 latent token ≪ symbolic memory 几千 token）。JoyAgent 多用 1.87M tokens 才涨 2.50%。
- Time：推理时间约 No-Memory 的 2/3，OAgents 的 1/2.16。因为不需要 trajectory summarization / insight extraction 这种 LLM-as-judge 步骤。

### Ablation（Figure 6 Right）

- w/o role（去掉 $\gamma$ in eq. 5）：AutoGen KodCode 掉 2.30，MacNet 掉 6.45。**MAS 越复杂，role conditioning 越关键**。
- w/o experience bank update（disable eq. 4）：KodCode 掉 3.60，**PDDL 掉 7.63**。说明 online accumulation 对 OOD task distribution 至关重要。

### Case Study（Figure 7）

PDDL 任务里 vanilla MacNet 出现 Step Repetition（反复移动 ball2 不看 goal），OAgents 出现 Disobey Task Specification（盲目跟随 retrieved trajectory），都有 Reasoning-Action Mismatch。LatentMem 在某步生成错误 action，下一步立即 "check valid actions" 自我纠正——**latent memory 在 actor-critic 风格 MAS 里能激活自我纠错**。

---

## 我的 Intuition 和几点质疑

### 漂亮的地方

**1. Gradient flow 设计很巧妙**：latent memory concat 到 hidden state 而非 prompt token，保留了 differentiability。这是让 RL signal 能反传到 composer 的关键。RAG 那种 discrete retrieve + discrete concat 就做不到。

**2. Role conditioning 有几何证据**：Figure 4 的 t-SNE 不是 cosmetic visualization，是真正验证 role 信息被编码进 latent memory 的几何结构，且能 transfer 到 unseen MAS。

**3. Train small, freeze large**：composer 是 LoRA + small transformer，整个 LMPO 只动 $\phi$，$\theta$ frozen。这让方法能 plug-in 到任意 LLM 而不需要 full retrain——工程友好性关键。

### 可质疑的点

**1. L'=8 是不是太短**？8 个 latent token 表达复杂 multi-step reasoning history 听起来很 tight。Appendix Figure 6 (Left) 显示 L' 增到 16-32 还有收益，但 trade-off 不明显——可能不同 task 最优 L' 不同，paper 用全局 L'=8 是 compromise。

**2. K=1 retrieval 太保守**：Table 3 说 K=1，ablation Figure 10 显示 K 增到 5 还涨。为啥主实验不直接用 K=3 或 5？可能是 efficiency consideration 没说清楚。

**3. Generalization 归因**：作者把 OOD 性能归功于 latent representation 的 robustness。另一种解释：LMPO 训练只用 in-domain data，composer 在 OOD 上能 generalize 主要因为学的是"如何从 trajectory 提取 task-relevant signal"这种 meta-skill，而非具体 domain 知识。这个归因 paper 没 disentangle。

**4. vs MARTI 的 controlled 比较**：MARTI fine-tune 整个 backbone，LatentMem 只训 composer（LoRA）。LatentMem 赢了，一种解读是"learnable memory > learnable backbone"，另一种是"MARTI 在 MAS 上 fine-tune 容易 overfit topology"。这里其实可以更 controlled——比如固定 MARTI 的 LoRA rank 跟 LatentMem 的 trainable param 数对齐。

**5. Reward sparsity**：PDDL 这类 task 的 reward 是 binary（任务完成/未完成），advantage 的 std 可能很小，公式 (10) 的 normalize 可能数值不稳。Paper 没讨论 reward shaping 或 sparse reward 的 robustness。

---

## 联想延伸

### 1. Latent Reasoning 这条线
- SoftCoT ([link](https://arxiv.org/abs/2502.12134))：soft chain-of-thought，continuous latent 替代 discrete CoT
- MemGen ([link](https://arxiv.org/abs/2509.24704))：generative latent memory for self-evolving agent
- LatentSeek ([link](https://arxiv.org/abs/2505.13308))：test-time policy gradient in latent space
- LatentMAS ([link](https://arxiv.org/abs/2511.20639))：latent communication for MAS

LatentMem 是这条线在 **MAS memory 维度** 的延伸。区别：latent reasoning 通常是 single-agent 内部思考压缩；LatentMem 是 agent 间 shared experience 的角色化编码。

### 2. Composer 可以是 cross-attention 而非 concat
当前公式 (6) 是直接 concat latent memory 到 hidden state，相当于把 latent memory 当 prefix token。如果换成 cross-attention（agent query latent memory 的 key-value），可能表达力更强，且不必固定 L'。这跟 Perceiver ([link](https://arxiv.org/abs/2103.03206))、Memory Networks ([link](https://arxiv.org/abs/1410.3916)) 思路相通。

### 3. Multi-modal extension
如果 trajectory 包含图像/工具调用结果，latent memory 可以自然容纳——这是 latent representation 比文本 memory 的隐性优势。LatentMAS ([link](https://arxiv.org/abs/2511.20639)) 已经在 latent communication 上探索多模态。

### 4. Continual learning 角度
experience bank 在线 append，composer frozen 训完不再更新。如果 composer 也 online update（持续 LMPO），会触到 continual learning 的 catastrophic forgetting 问题。MemEvolve ([link](https://arxiv.org/abs/2512.18746)) 看起来在探索这个方向。

### 5. 和 Test-Time Scaling 的关系
latent memory 本质是把"思考过程"压缩进 prefix，类似 test-time compute 的内化。如果和 OpenAI o1 / DeepSeek-R1 这类 reasoning model 结合，latent memory 可能充当 "reasoning skill 的 cache"，把 test-time 反复推理的 pattern 蒸馏成可复用的 latent prefix。

### 6. 信息论视角
L'=8 token × D dim = 8D 个 float。Qwen3-4B D≈2560 → 20480 float ≈ 80KB 信息容量。比 trajectory 的几千 token（几十 KB 文本 + embedding）是同量级但结构化更强。这个 capacity 是否足够存 multi-step coordination？理论上 8D 能 express 任意 $e^{8D}$ 个状态，应该够用，但实际训练效率是另一回事。

### 7. Game theory angle
MAS 里 agent 间有 cooperative / competitive 关系。LatentMem 把 shared experience role-conditioned 编码，某种程度上是 **decentralized shared memory + role-specific readout**。这跟 MARL 里的 centralized critic + decentralized actor ([MADDPG link](https://arxiv.org/abs/1706.02275)) 哲学相似——shared info, role-specific use。

### 8. 应用到 nanoGPT 的简化版
如果你 Karpathy 想在 nanoGPT 上做 toy experiment：
- 给一个小 transformer 训 LoRA，输入 (agent role text, retrieved past trajectories)，输出 8 个 latent vector
- 把这 8 个 vector 拼到下一个 LLM call 的 input embedding 前面
- 用 GRPO 训这个 LoRA，reward 是 task correctness

整个 pipeline 不需要改 LLM forward，只需要 hook 在 embedding layer 后面。非常 friendly 到 HuggingFace transformers + PEFT 生态。

---

## Paper 的小瑕疵

诚实记录 reading 中发现的 issues：

1. 公式 (3) 的 `topK` 写法有 weird 字符残留，公式排版不太干净
2. Table 1 里 `Auten Hel-n`、`CAEL Helout`、`DLAN` 是 AutoGen / Held-in、CAMEL / Held-out、DyLAN 的 OCR 错误
3. Section 5.6 ablation 里"w/o role"在 AutoGen KodCode 掉 2.30，MacNet 掉 6.45——但 main table 显示 LatentMem 在 MacNet KodCode 比 AutoGen 还高（78.90 vs 76.80），说明 MacNet 的 MAS 结构对 role conditioning 更 sensitive，但 ablation 数字和 main table 数字需要交叉验证
4. K=1 的主实验 setting vs ablation 的 K 扫描——主实验用 K=1 是 efficiency 考虑还是 performance 考虑没说清楚

---

## 最终直觉总结

LatentMem 的核心 insight 三条：

**1. Memory 可以是 continuous latent prefix，不必是离散文本**。这绕开了 symbolic memory 的 token explosion 和 hand-crafted engineering。

**2. Role conditioning 让 memory heterogeneous**，避免 homogenization。一个 shared experience bank，每个 agent 读出不同的 latent memory。

**3. RL signal 可以穿过 latent memory 反传到 composer**，无需 retrain backbone。这让 memory 设计变成 optimization 问题，不再是 design pattern 问题。

对 LLM agent research 来说，这个工作把"agent memory" 从 RAG / reflection / skill library 这类 design pattern 推到了 learnable module，跟 Sutton 的 Bitter Lesson 哲学对齐。后续可能的方向：cross-attention composer、continual LMPO、multi-modal trajectory、和 reasoning model 结合做 test-time compute cache。

---

参考链接汇总：
- Paper GitHub: https://github.com/KANABOON1/LatentMem
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- LoRA: https://arxiv.org/abs/2106.09685
- DAPO: https://arxiv.org/abs/2503.14476
- AutoGen: https://arxiv.org/abs/2308.08155
- MacNet: https://arxiv.org/abs/2406.07155
- CAMEL: https://arxiv.org/abs/2303.17760
- DyLAN: https://arxiv.org/abs/2403.02423
- MetaGPT: https://arxiv.org/abs/2308.00352
- ChatDev: https://arxiv.org/abs/2307.07924
- Voyager: https://arxiv.org/abs/2305.16291
- Generative Agents: https://arxiv.org/abs/2304.03442
- G-Memory: https://arxiv.org/abs/2506.07398
- OAgents: https://arxiv.org/abs/2506.15741
- JoyAgent: https://arxiv.org/abs/2510.00510
- MIRIX: https://arxiv.org/abs/2507.07957
- EvolveR: https://arxiv.org/abs/2510.16079
- Agent KB: https://arxiv.org/abs/2507.06229
- MARTI: https://github.com/TsinghuaC3I/MARTI
- SoftCoT: https://arxiv.org/abs/2502.12134
- MemGen: https://arxiv.org/abs/2509.24704
- LatentSeek: https://arxiv.org/abs/2505.13308
- LatentMAS: https://arxiv.org/abs/2511.20639
- MemEvolve: https://arxiv.org/abs/2512.18746
- MiniLM: https://arxiv.org/abs/2002.10957
- t-SNE: http://www.jmlr.org/papers/v9/vandermaaten08a.html
- TriviaQA: https://aclanthology.org/P17-1147/
- KodCode: https://arxiv.org/abs/2503.02951
- StrategyQA: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100676
- PopQA: https://aclanthology.org/2023.findings-acl.985/
- BigCodeBench: https://arxiv.org/abs/2403.07974
- PDDL (Silver et al.): https://ojs.aaai.org/index.php/AAAI/article/view/30166
- Why MAS Fail (Cemri): https://arxiv.org/abs/2503.13657
- Bitter Lesson (Sutton): http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Prefix-tuning: https://arxiv.org/abs/2104.08691
- P-tuning v2: https://arxiv.org/abs/2110.07602
- Perceiver: https://arxiv.org/abs/2103.03206
- Memory Networks: https://arxiv.org/abs/1410.3916
- MADDPG: https://arxiv.org/abs/1706.02275
- Self-RAG: https://arxiv.org/abs/2310.11511
- REPLUG: https://arxiv.org/abs/2305.05170
- G-Retriever: https://arxiv.org/abs/2402.07630
- AgentBoard: https://arxiv.org/abs/2401.13178
- Memory survey (Hu et al.): https://arxiv.org/abs/2512.13564
- LLM memory survey (Wu et al.): https://arxiv.org/abs/2504.15965

如果你想 deep dive 进任何一个 angle——比如 LMPO 的数值稳定性、role conditioning 的几何分析、或怎么把这个 idea 落到 nanoGPT 上做 toy experiment——我可以继续展开。

---

# LatentMem: 把 Multi-Agent Memory 从"手工符号"搬到"可微分 Latent Space"

嗨 Andrej！这篇 paper 挺有意思的——它把最近 latent reasoning / latent memory 这条线（SoftCoT、MemGen、LatentSeek）从 single-agent 推到了 multi-agent systems (MAS)，核心 motivation 是用一个 learnable 的 memory composer 取代手工设计的 memory 单元（insight graph、skill library、procedural memory 那一类），并且做到 **role-aware + token-efficient**。下面我从 motivation、architecture、LMPO、experiments、与相关工作联系这几个角度深入拆解，并尽量 build 你的 intuition。

---

## 1. Motivation：为什么 Multi-Agent Memory 当前卡住了

Paper 把现有 MAS memory 的瓶颈归纳为两条：

**(i) Memory Homogenization**。现有方法（MetaGPT 的 shared message pool、ChatDev 的 inside-trial log、G-Memory 的三层层级、OAgents 的 long/short-term hierarchical memory）对 **所有 agent 都用同一种 memory 表征**，忽略了 agent 间的功能异质性。一个 user proxy agent 和一个 code-writing agent 拿到同一份 history，没有 role conditioning，会导致 correlated errors——Cemri et al. 2025 的 "Why do multi-agent LLM systems fail?" 实证了这点 ([link](https://arxiv.org/abs/2503.13657))。

**(ii) Information Overload**。Multi-granularity memory（G-Memory 的 Query Graph + Insight Graph + Interaction Graph 三层结构，MIRIX 的 procedural memory）会膨胀 context。当 retrieved trajectory 多了，token 爆炸且 critical signal 被稀释。Paper 在 Figure 10 / Appendix C.3 给出实证：G-Memory 在 top-K > 3 时 KodCode 从 72.95% 掉到 70.25%，而 LatentMem 一直涨。

**核心 idea**：与其在符号空间堆 memory，不如让一个 small neural composer 把 raw trajectory 蒸馏成 **fixed-length latent token**（L'=8 个 token），按 agent role conditioning，再 prepend 到 hidden state 上。这绕开了"语言层面的 memory engineering"，把 memory 设计变成一个 RL 优化问题。

直觉上，这跟 SoftCoT ([link](https://arxiv.org/abs/2502.12134)) 和 MemGen ([link](https://arxiv.org/abs/2509.24704)) 的思路一致：让模型在 latent space 里"思考"或"记忆"，避免离散 token 的 bottleneck。LatentMem 把这个 idea 拓到 multi-agent，并且加上了 role conditioning + RL training。

---

## 2. Architecture：Experience Bank + Memory Composer

### 2.1 系统定义

公式 (1)：

$$\chi = (\mathcal{A}, \mathcal{G}, \mathcal{M})$$

- $\chi$：整个 MAS
- $\mathcal{A} = \{a_1, \dots, a_N\}$：N 个 agent，每个 agent $a_k = (\gamma_k, \pi_{\theta_k})$，$\gamma_k$ 是 role profile（文本描述），$\pi_{\theta_k}$ 是 policy（即 LLM 参数化为 $\theta_k$）
- $\mathcal{G}$：execution graph，可以是 static topology（如 MetaGPT 流水线）或 dynamic regulation（如 AgentNet 那类）
- $\mathcal{M}$：global memory module，本工作就是要 learn 这个 $\mathcal{M}$

优化目标，公式 (2)：

$$\max_{\mathcal{M}} \mathbb{E}_{q \sim \mathcal{D}, \tau \sim X(q)} [R(\tau)]$$

- $q$：从 dataset $\mathcal{D}$ 采样的 query
- $\tau \sim X(q)$：MAS 处理 $q$ 生成的 trajectory
- $R(\tau)$：从 $\tau$ 抽 answer 算 reward（正确性）

这个 formulation 故意 agnostic 到 memory 具体形式，符号的、参数化的都行——这就给 latent memory 留了空间。

### 2.2 Experience Bank $\mathcal{B}$

这是 storage 层，**极简**，只存 raw trajectory，不蒸馏不抽取。这里很关键：作者明确引用 Sutton 的 "Bitter Lesson" ([link](http://www.incompleteideas.net/IncIdeas/BitterLesson.html))，主张 scalable system 应该靠 general learning mechanism，而非 hand-crafted knowledge。

每条 trajectory：
$$\tau = \{(\alpha_j, p_j, o_j)\}_{j=1}^{H}$$

- $\alpha_j$：第 $j$ 步 active agent 的 index
- $p_j$：input prompt
- $o_j$：output
- $H$：trajectory horizon（即总步数）

**Retrieval** 用 cosine similarity + MiniLM embedding（all-MiniLM-L6-v2，[link](https://arxiv.org/abs/2002.10957)）。公式 (3)：

$$\mathcal{T}_q = \text{top-}K_{\tau_i \in \mathcal{B}}(\text{sim}(\mathbf{v}(q), \mathbf{v}(\tau_i))) = \{\tau_i\}_{i=1}^{K}$$

- $\mathbf{v}(\cdot)$：MiniLM encoder 把 query 或 trajectory 投影到 embedding 空间
- $\text{sim}$：cosine similarity
- $K$：取 top-K，paper 里 K=1（ablation 在 Appendix C.3 试 K=1..5）

**Update**：任务完成后把新 $\tau_{\text{new}}$ append 进去，公式 (4)：$\mathcal{B} \leftarrow \mathcal{B} \cup \{\tau_{\text{new}}\}$。这样形成 self-improving loop，**online inference 时也能不断积累**。

### 2.3 Memory Composer $\sigma_\phi$：核心组件

公式 (5)：

$$m_j = \sigma_\phi(\gamma_{\alpha_j}, \mathcal{T}_q) \in \mathbb{R}^{L' \times D}$$

- $\sigma_\phi$：memory composer network，参数为 $\phi$，实例化为一个 small transformer，**初始化自 backbone LLM**，用 LoRA 训练 ([LoRA link](https://arxiv.org/abs/2106.09685))
- $\gamma_{\alpha_j}$：当前 active agent $a_{\alpha_j}$ 的 role profile
- $\mathcal{T}_q$：retrieved raw trajectories
- $L'$：latent memory 的固定长度，paper 设 L'=8
- $D$：backbone LLM 的 hidden dim（Qwen3-4B 是 D=2048 量级，Llama-3.1-8B 是 D=4096 量级）

注意这里 $\gamma_{\alpha_j}$ 是 role conditioning 的关键——**同一个 trajectory，不同 role 会输出不同的 latent memory**。Figure 4 的 t-SNE 验证：在 KodCode/AutoGen（in-domain, seen）和 BigCodeBench/CAMEL（OOD, unseen）上，user-proxy 和 assistant 的 latent memory 都形成清晰分离的两个 cluster。

**注入方式**：在 token embedding 层做 concat，公式 (6)：

$$\tilde{\pi}_{\theta_{\alpha_j}}(p_j, m_j) = \pi_{\theta_{\alpha_j}}(\text{concat}(h_j, m_j))$$

- $h_j = (h_j^{(1)}, \dots, h_j^{(L)}) \in \mathbb{R}^{L \times D}$：input prompt $p_j$ 经过 embedding layer 的 hidden states，L 是 prompt 长度
- $\text{concat}(h_j, m_j) \in \mathbb{R}^{(L+L') \times D}$：把 latent memory 当作额外的 L' 个 "prefix token"，类似 prefix tuning ([link](https://arxiv.org/abs/2104.08691)) 或 soft prompt
- $\tilde{\pi}$：wrapped policy，对 agent 层完全透明，**不需要修改 MAS framework**

这里和 prefix-tuning / soft prompt 的关键差异：(a) prefix 是从 retrieved trajectory 动态生成的，不是固定可学习参数；(b) 是 role-conditioned 的；(c) 通过 LMPO 训练而非 LM head 对齐。

---

## 3. LMPO：Latent Memory Policy Optimization

这是 paper 的技术核心。本质是 GRPO 的变种（GRPO 来自 DeepSeekMath，[link](https://arxiv.org/abs/2402.03300)），但做了几个针对性设计。

### 3.1 Parametric Dependency & Gradient Flow

公式 (7)：

$$\mathbb{P}(\tau \mid q, \mathcal{T}_q; \phi, \{\theta_k\}_{k=1}^N) = \prod_{j=1}^{H} \mathbb{P}(o_j \mid p_j, m_j; \theta_{\alpha_j})$$

- 左边：给定 query $q$ 和 retrieved trajectory $\mathcal{T}_q$，整条 trajectory $\tau$ 的概率
- $\phi$：composer 参数（要 train 的）
- $\{\theta_k\}$：agent backbones（**frozen**，不动）
- 右边：每一步 active agent $a_{\alpha_j}$ 在条件 $(p_j, m_j)$ 下生成 $o_j$ 的概率的乘积

公式 (8)：进一步把 $o_j$ 拆到 token 级别

$$\mathbb{P}(o_j \mid p_j, m_j; \theta_{\alpha_j}) = \prod_{t=1}^{T} \tilde{\pi}_{\theta_{\alpha_j}}(o_j^{(t)} \mid p_j, o_j^{(<t)}, m_j)$$

- $o_j^{(t)}$：第 $j$ 步 active agent 输出的第 $t$ 个 token
- $o_j^{(<t)}$：前面已生成的 tokens（autoregressive condition）
- $m_j$：latent memory，作为前缀条件

**关键 insight**：$m_j = \sigma_\phi(\mathcal{T}_q, \gamma_{\alpha_j})$ 是 differentiable 的，所以从 task reward $R(\tau)$ 出发的 gradient 可以**穿过 agent forward pass 反传到 $\phi$**，而 $\theta$ 保持 frozen。这就是把 RL 信号"借道" latent memory 优化 composer，避免了 retrain backbone 的成本。

### 3.2 Group-based Advantage（来自 GRPO）

公式 (9)：对每个 query $q$，sample G 条 trajectory：

$$\{\hat{\tau}_i\}_{i=1}^G \sim \mathbb{P}(\cdot \mid q, \mathcal{T}_q; \phi, \{\theta_k\})$$

公式 (10)：组内 normalize 算 advantage（GRPO 的精髓，去掉 critic）：

$$\hat{A}_i = \frac{R(\hat{\tau}_i) - \text{mean}(\{R(\hat{\tau}_i)\}_{i=1}^G)}{\text{std}(\{R(\hat{\tau}_i)\}_{i=1}^G) + \epsilon}$$

- $R(\hat{\tau}_i)$：第 $i$ 条 trajectory 的 reward
- $\epsilon$：防止除零的小常数

这个 group-relative advantage 让 GRPO 不需要训 value function，对 memory composer 这种"输出是连续 latent 而非离散 action"的场景很合适——不然 critic 也得做 latent state 的 value estimation，复杂度爆炸。

### 3.3 Token-level Surrogate Objective

公式 (11)–(13) 是 PPO 风格的 clipped objective，但**作用在 token 级别**：

公式 (11)：

$$\mathcal{T}_{\text{LMPO}}(\phi) = \mathbb{E}_{q \sim \mathcal{D}, \{\hat{\tau}_i\}_{i=1}^G \sim \mathbb{P}(\cdot | q, \mathcal{T}_q)} \left[\frac{1}{|\{\hat{\tau}_i\}_{i=1}^G|} \sum_{i,j,t} \mathcal{L}_{i,j,t}(\phi)\right]$$

- $|\{\hat{\tau}_i\}|$：group 内总 token 数
- $\mathcal{L}_{i,j,t}(\phi)$：单 token 的 loss

公式 (12)：

$$\mathcal{L}_{i,j,t}(\phi) = \min\Big(r_{i,j,t}(\phi)\hat{A}_i, \text{clip}(r_{i,j,t}(\phi), 1-\varepsilon, 1+\varepsilon)\hat{A}_i\Big)$$

- $\varepsilon$：clip ratio，paper 设 0.2（标准 PPO 默认值）
- $\hat{A}_i$：来自公式 (10) 的 group advantage

公式 (13)：importance sampling ratio

$$r_{i,j,t}(\phi) = \frac{\tilde{\pi}_{\theta}(\ldots \mid \sigma_\phi(\gamma_{\alpha_{i,j}}, \mathcal{T}_q))}{\tilde{\pi}_{\theta}(\ldots \mid \sigma_{\phi_{\text{old}}}(\gamma_{\alpha_{i,j}}, \mathcal{T}_q))}$$

- 分子：用当前 composer $\phi$ 生成的 latent memory 算出的 token 概率
- 分母：用 old composer $\phi_{\text{old}}$ 生成的 latent memory 算出的 token 概率（rollout 时用的版本）

**为什么 token-level 而非 trajectory-level**？作者引用 Yu et al. 2025 (DAPO, [link](https://arxiv.org/abs/2503.14476))：trajectory-level objective 让 long sequence 中每个 token 的 gradient 贡献被稀释。MAS trajectory 通常很长（多 agent 多步交互），token-level 才能让 composer 捕捉到关键 coordination pattern。

**直觉**：LMPO 是在做"逆向 prefix tuning"——传统 prefix-tuning 是用 LM loss 监督，让 prefix 模仿某些 demonstration；LMPO 是用 task reward 监督，让 latent memory 直接服务于最终 task performance。这种 "policy gradient through a continuous prefix" 思路和 MemGen ([link](https://arxiv.org/abs/2509.24704))、LatentSeek ([link](https://arxiv.org/abs/2505.13308)) 是同源思路。

---

## 4. 实验数据深度解读

### 4.1 Setup 关键点

- **6 benchmarks**：
  - In-domain: TriviaQA ([link](https://aclanthology.org/P17-1147/)), KodCode ([link](https://arxiv.org/abs/2503.02951)), StrategyQA ([link](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100676)), PopQA ([link](https://aclanthology.org/2023.findings-acl.985/))
  - Out-of-domain: BigCodeBench ([link](https://arxiv.org/abs/2403.07974)), PDDL ([link](https://ojs.aaai.org/index.php/AAAI/article/view/30166))

- **4 MAS frameworks**：
  - In-distribution: AutoGen ([link](https://arxiv.org/abs/2308.08155)), MacNet ([link](https://arxiv.org/abs/2406.07155))
  - Unseen: CAMEL ([link](https://arxiv.org/abs/2303.17760)), DyLAN ([link](https://arxiv.org/abs/2403.02423))

- **Backbones**：Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct

- **训练数据**：40,580 trajectories，只用 in-domain datasets + in-distribution MAS 采集

### 4.2 Table 1 关键数字

读 Table 1（Qwen3-4B backbone）：

| MAS | Dataset | No-Memory | 最强 baseline | LatentMem | Gain over No-Memory |
|---|---|---|---|---|---|
| AutoGen | TriviaQA | 60.31 | G-Memory 60.56 | **76.51** | +16.20 |
| AutoGen | KodCode | 68.40 | G-Memory 71.40 | **76.80** | +8.40 |
| AutoGen | PopQA | 38.78 | G-Memory 42.67 | **52.70** | +13.92 |
| AutoGen | PDDL (OOD) | 16.39 | G-Memory 17.06 | **23.49** | +7.10 |
| MacNet | KodCode | 70.50 | G-Memory 72.50 | **78.90** | +8.40 |
| CAMEL (unseen) | KodCode | 70.70 | OAgents 71.40 | **77.75** | +7.05 |
| CAMEL | PDDL | 22.10 | G-Memory 24.56 | **28.12** | +6.02 |
| DyLAN (unseen) | PopQA | 24.89 | G-Memory 42.28 | **44.25** | +19.36 |

几个关键观察：

1. **LatentMem 在所有 setting 都有正向 gain**，从 +1.59 (DyLAN BigCodeBench) 到 +19.36 (DyLAN PopQA)。
2. **大多数 baseline 在 OOD (PDDL) 和 unseen MAS (CAMEL/DyLAN) 上掉点**。例如 MetaGPT 在 AutoGen/PDDL 掉 4.44，Voyager 掉 4.44，OAgents 在 CAMEL/PDDL 暴跌 8.21。LatentMem 反而稳定上涨——这说明 latent representation 比 symbolic memory 更 robust 到 distribution shift。
3. **DyLAN PopQA 19.36% 的巨大 gain** 很 eye-catching，说明在 debate-style MAS + entity-centric QA 这种 long-tail knowledge 任务上，latent memory 的杠杆最大。

Table 4（Llama-3.1-8B-Instruct backbone，Appendix C.3）显示更强：

| MAS | Dataset | No-Memory | LatentMem | Gain |
|---|---|---|---|---|
| AutoGen | KodCode | 47.45 | **65.90** | +18.45 |
| MacNet | KodCode | 48.50 | **65.50** | +17.00 |
| CAMEL | KodCode | 48.95 | **63.85** | +14.90 |
| DyLAN | KodCode | 47.55 | **64.25** | +16.70 |

Llama-3.1-8B 在 KodCode 上 No-Memory 才 ~48%（似乎 baseline 偏低，可能是 framework 配置问题），LatentMem 直接拉到 ~65%，**+18%** 这种夸张涨幅值得注意——可能 KodCode 的 verifiable test 让 reward signal 极干净，LMPO 训练效率很高。

### 4.3 Table 2: vs MARTI（multi-agent fine-tuning）

MARTI ([link](https://github.com/TsinghuaC3I/MARTI)) 是用 GRPO 直接 fine-tune agent backbone。同 budget、同 data、同 framework：

| MAS | Dataset | MARTI | LatentMem | Diff |
|---|---|---|---|---|
| AutoGen | KodCode | 74.20 | 76.80 | +2.60 |
| AutoGen | TriviaQA | 64.78 | 76.51 | **+11.73** |
| MacNet | KodCode | 73.10 | 78.90 | +5.80 |
| MacNet | TriviaQA | 62.31 | 65.98 | +3.67 |

**重要 insight**：MARTI 在 AutoGen→MacNet 时 KodCode 掉 1.10（74.20→73.10），说明直接训 backbone 容易 overfit 到特定 MAS topology。LatentMem 反而 +2.10（76.80→78.90），说明 memory composer 学到的是 **MAS-agnostic 的高阶 pattern**，能 transfer 到不同 topology。

### 4.4 Cost（Figure 3, Figure 8）

- **Token cost**：LatentMem 用 0.01M tokens 比 No-Memory 还少（因为 latent memory L'=8 token ≪ symbolic memory 的几千 token），而 JoyAgent 多用 1.87M tokens 才涨 2.50%。
- **Time cost**：LatentMem 推理时间约 No-Memory 的 2/3，OAgents 的 1/2.16。因为不需要复杂的 trajectory summarization / insight extraction 这种 LLM-as-judge 步骤。

直觉上这跟 latent reasoning 的优势一致：固定长度 prefix 替代 unbounded discrete context，trade 一点 train cost 换大幅 inference efficiency。

### 4.5 Role-aware Memory 可视化（Figure 4, 9）

Appendix C.2 给了方法：对每个 latent memory sequence $m_i = (m_{i,1}, \dots, m_{i,L'}) \in \mathbb{R}^{L' \times D}$，先 mean pooling 成 $\bar{m}_i \in \mathbb{R}^D$，公式 (14)：

$$\bar{m}_i = \frac{1}{L'} \sum_{l=1}^{L'} m_{i,l}$$

再用 t-SNE（公式 (15)）降到 2D。

结果显示 user-proxy 和 assistant 在 in-domain KodCode/AutoGen 上清晰分离，OOD BigCodeBench + unseen CAMEL 也保持分离——说明 role conditioning **是真的学到了**，且能 transfer。这是 vs MetaGPT/ChatDev/OAgents 这种 "shared memory for all agents" 的核心不同。

### 4.6 Ablation（Figure 6 Right, Section 5.6）

- **w/o role**（去掉 $\gamma$ in eq. 5）：AutoGen KodCode 掉 2.30，MacNet 掉 6.45。MAS 越复杂，role conditioning 越关键。
- **w/o experience bank update**（disable eq. 4）：KodCode 掉 3.60，**PDDL 掉 7.63**。说明 online accumulation 对 OOD task distribution 至关重要——只靠静态 bank 不足以应对复杂 symbolic planning。

### 4.7 Case Study（Figure 7, Appendix C.4）

PDDL 任务里 vanilla MacNet 出现 **Step Repetition**（反复在 room a/b 之间移动 ball2），OAgents 出现 **Disobey Task Specification**（盲目跟随 retrieved trajectory 不看 goal state），都有 **Reasoning-Action Mismatch**。LatentMem 在某步生成错误 action "pick ball5 rooma right"，下一步立即用 "check valid actions" 自我纠正。这暗示 latent memory 在 actor-critic 风格的 MAS 里能激活自我纠错机制。

---

## 5. 与相关工作的关系网（联想）

让我把这篇工作放到一个更广的 map 上：

### 5.1 Latent Reasoning 这条线
- **SoftCoT** ([link](https://arxiv.org/abs/2502.12134))：soft chain-of-thought，用 continuous latent 替代 discrete CoT token
- **MemGen** ([link](https://arxiv.org/abs/2509.24704))：用 generative latent memory 给 self-evolving agent
- **LatentSeek** ([link](https://arxiv.org/abs/2505.13308))：test-time instance-level policy gradient in latent space
- **LatentMAS** ([link](https://arxiv.org/abs/2511.20639))：latent communication for MAS（更接近 communication channel compression）

LatentMem 是这条线在 **MAS memory** 维度的延伸。区别：latent reasoning 通常是 single-agent 内部思考压缩；LatentMem 是 **agent 间 shared experience 的角色化编码**。

### 5.2 Multi-Agent Memory 这条线
- **Voyager** ([link](https://arxiv.org/abs/2305.16291))：Minecraft skill library，单 agent，symbolic
- **Generative Agents** ([link](https://arxiv.org/abs/2304.03442))：observation + reflection memory，Park et al. 的经典工作
- **MetaGPT** ([link](https://arxiv.org/abs/2308.00352))：shared message pool，software dev
- **ChatDev** ([link](https://arxiv.org/abs/2307.07924))：inside-trial memory only
- **OAgents** ([link](https://arxiv.org/abs/2506.15741))：multi-granularity memory
- **G-Memory** ([link](https://arxiv.org/abs/2506.07398))：三层 graph（Query + Insight + Interaction）
- **MIRIX** ([link](https://arxiv.org/abs/2507.07957))：procedural memory from user goals
- **EvolveR** ([link](https://arxiv.org/abs/2510.16079))：experience-driven lifecycle
- **Agent KB** ([link](https://arxiv.org/abs/2507.06229))：cross-domain experience

这些都是 **symbolic** memory，token 爆炸且 role-agnostic。LatentMem 用 latent representation 一次解决两个问题。

### 5.3 RL for LLM 这条线
- **GRPO** ([link](https://arxiv.org/abs/2402.03300))：group relative advantage，去 critic
- **DAPO** ([link](https://arxiv.org/abs/2503.14476))：token-level objective，paper 显式引用
- **MARTI** ([link](https://github.com/TsinghuaC3I/MARTI))：直接 GRPO fine-tune MAS backbone
- **MIRIX-style RL for memory**：少见，LatentMem 是少数用 RL train memory module 的工作

### 5.4 Prefix Tuning / Soft Prompt 这条线
- **Prefix-tuning** ([link](https://arxiv.org/abs/2104.08691))：固定可学习 prefix
- **P-tuning v2** ([link](https://arxiv.org/abs/2110.07602))：per-layer prefix
- **Soft prompt** ([link](https://arxiv.org/abs/2104.08691))

LatentMem 可以理解为 **dynamic, role-conditioned, RL-trained prefix**，prefix 内容由 retrieved trajectory 和 role 决定，被 task reward 直接监督。这是把 prefix-tuning 从 imitation 推到 RL 范式。

### 5.5 RAG 这条线
LatentMem 的 experience bank 本质是 **trajectory-level RAG**。但 RAG 通常 retrieve document chunk → 放进 context；LatentMem retrieve trajectory → 通过 composer 编码成 latent prefix。这跟 RAG 的离散 retrieve + 离散 context 拼接不同，是 **discrete retrieve + continuous encoding**。

类似思路有：
- **Self-RAG** ([link](https://arxiv.org/abs/2310.11511))：reflective retrieval
- **REPLUG** ([link](https://arxiv.org/abs/2305.05170))：retrieve-then-generate with black-box adaptation
- **G-Retriever** ([link](https://arxiv.org/abs/2402.07630))：graph retrieval + GNN

---

## 6. 我的 Intuition & 几点思考

### 6.1 这个工作的"漂亮之处"
1. **Gradient flow 设计很巧妙**：通过把 latent memory 注入到 hidden state 而非 prompt token level，保留了 differentiability，让 RL signal 能反传。如果放在 prompt token level（像 RAG 那样拼接 retrieved text），就不可微了。
2. **Role conditioning 的几何证据**：Figure 4 的 t-SNE 不是 cosmetic visualization，是真正验证了"role 信息被编码进了 latent memory 的几何结构"，且能 transfer 到 unseen MAS。
3. **Train small, freeze large**：composer 是 LoRA + small transformer，整个 LMPO 只动 $\phi$，$\theta$ frozen。这让方法可以 plug-in 到任意 LLM 而不需要 full retrain——这是工程友好性的关键。

### 6.2 几个隐忧 / 可质疑点
1. **L'=8 是不是太短了**？8 个 latent token 表达复杂 multi-step reasoning history 听起来很 tight。Appendix Figure 6 (Left) 显示 L' 增到 16-32 还有收益，但 trade-off 不明显——可能不同 task 的最优 L' 不同，paper 用一个全局 L'=8 是 compromise。
2. **K=1 retrieval 是不是太保守**？Table 3 说 K=1，ablation Figure 10 显示 K 增到 5 还涨——为什么不直接用 K=3 或 5 作为主实验 setting？可能 cost consideration。
3. **Generalization 的归因**：作者把 OOD 性能归功于 latent representation 的 robustness。但其实另一个解释是：LMPO 训练分布只用了 in-domain data，composer 在 OOD 上能 generalize 主要因为 LMPO 学的是"如何从 trajectory 提取 task-relevant signal"这种 meta-skill，而非具体 domain 知识。这个归因 paper 没完全 disentangle。
4. **vs MARTI 的比较**：MARTI fine-tune 整个 backbone，LatentMem 只训 composer（LoRA）。LatentMem 赢了 → 一种解读是"learnable memory > learnable backbone"，但另一种解读是"MARTI 在 MAS 上 fine-tune 容易 overfit topology"，paper Section 5.4 倾向后者。这里其实可以做得更 controlled，比如固定 MARTI 的 LoRA rank 跟 LatentMem 的 trainable param 数对齐。
5. **Reward sparsity**：PDDL 这类 task 的 reward 是 binary（任务完成/未完成），advantage 的 std 可能很小，公式 (10) 的 normalize 可能数值不稳。Paper 没讨论 reward shaping 或 sparse reward 的 robustness。

### 6.3 延伸联想
1. **Composer 可以是 cross-attention 而非 concat**：当前公式 (6) 是直接 concat latent memory 到 hidden state，这相当于把 latent memory 当 prefix token。如果换成 cross-attention（agent query latent memory key-value），可能表达力更强，且不必固定 L'。这跟 Perceiver ([link](https://arxiv.org/abs/2103.03206))、Memory Networks ([link](https://arxiv.org/abs/1410.3916)) 思路相通。
2. **Multi-modal extension**：如果 trajectory 包含图像/工具调用结果，latent memory 可以自然容纳——这是 latent representation 比文本 memory 的隐性优势。LatentMAS ([link](https://arxiv.org/abs/2511.20639)) 已经在 latent communication 上探索了多模态。
3. **Continual learning 角度**：experience bank 在线 append，composer frozen 训完不再更新，这是 offline-then-frozen 模式。如果 composer 也 online update（持续 LMPO），会触到 continual learning 的 catastrophic forgetting 问题。MemEvolve ([link](https://arxiv.org/abs/2512.18746)) 看起来在探索这个方向。
4. **和 Test-Time Scaling 的关系**：latent memory 本质是把"思考过程"压缩进 prefix，类似 test-time compute 的内化。如果和 OpenAI o1 / DeepSeek-R1 这类 reasoning model 结合，latent memory 可能充当 "reasoning skill 的 cache"，把 test-time 反复推理的 pattern 蒸馏成可复用的 latent prefix。
5. **信息论视角**：L'=8 token × D dim = 8D 个 float。Qwen3-4B D≈2560 → 20480 float ≈ 80KB 信息容量。比起 trajectory 的几千 token（几十 KB 文本 + embedding）是同量级但结构化更强。这个 capacity 是否足够存 multi-step coordination？理论上 8D 能 express 任意 $e^{8D}$ 个状态，应该够用，但实际训练效率是另一回事。
6. **Game theory angle**：MAS 里 agent 间有 cooperative / competitive 关系。LatentMem 把 shared experience role-conditioned 编码，某种程度上是 **decentralized shared memory + role-specific readout**。这跟 MARL 里的 centralized critic + decentralized actor ([MADDPG link](https://arxiv.org/abs/1706.02275)) 哲学相似——shared info, role-specific use。

### 6.4 最直接的应用直觉
如果你 Karpathy 自己在做 nanoGPT 或类似 small-scale experiment，这个 paper 的方法可以简化成：
- 给一个小 transformer 训 LoRA，输入 (agent role text, retrieved past trajectories)，输出 8 个 latent vector
- 把这 8 个 vector 拼到下一个 LLM call 的 input embedding 前面
- 用 GRPO 训这个 LoRA，reward 是 task correctness

整个 pipeline 不需要改 LLM forward，只需要 hook 在 embedding layer 后面。非常 friendly 到 HuggingFace transformers + PEFT 生态。

---

## 7. Paper 的小瑕疵

诚实记录一些 reading 中发现的 issues：

1. 公式 (3) 的 `topK` 写法有个 weird 字符 `⊲` 残留，公式排版不太干净
2. Table 1 里 `Auten Hel-n`、`CAEL Helout`、`DLAN` 显然是 AutoGen / Held-in、CAMEL / Held-out、DyLAN 的 OCR 错误
3. Section 5.6 ablation 里"w/o role"在 AutoGen KodCode 掉 2.30，MacNet 掉 6.45——但 main table 显示 LatentMem 在 MacNet KodCode 比 AutoGen 还高（78.90 vs 76.80），说明 MacNet 的 MAS 结构对 role conditioning 更 sensitive，但 ablation 数字和 main table 数字需要交叉验证
4. K=1 的主实验 setting vs ablation 的 K 扫描——主实验用 K=1 是 efficiency 考虑还是 performance 考虑没说清楚

---

## 8. 总结：这篇 paper 的核心 contribution

一句话：**LatentMem 把 multi-agent memory 从 symbolic hand-crafted 设计转变为 learnable latent representation，通过 LMPO 用 task reward 直接优化 memory composer，实现了 role-aware + token-efficient + generalizable 的 MAS memory**。

它建立的核心 insight：
1. Memory 可以是 **continuous latent prefix**，不必是离散文本
2. Role conditioning 让 memory **heterogeneous**，避免 homogenization
3. RL signal 可以 **穿过 latent memory 反传** 到 composer，无需 retrain backbone
4. 这种设计天然 transfer 到 unseen domain / unseen MAS

对 LLM agent research 来说，这个工作把"agent memory" 这个方向从 "design patterns"（RAG、reflection、skill library）推到了 "learnable module"，跟 Sutton 的 "Bitter Lesson" 哲学对齐。后续可能的方向：cross-attention composer、continual LMPO、multi-modal trajectory、和 reasoning model 结合做 test-time compute cache。

---

参考链接汇总：
- Paper 本身 GitHub: https://github.com/KANABOON1/LatentMem
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- LoRA: https://arxiv.org/abs/2106.09685
- DAPO: https://arxiv.org/abs/2503.14476
- AutoGen: https://arxiv.org/abs/2308.08155
- MacNet: https://arxiv.org/abs/2406.07155
- CAMEL: https://arxiv.org/abs/2303.17760
- DyLAN: https://arxiv.org/abs/2403.02423
- MetaGPT: https://arxiv.org/abs/2308.00352
- ChatDev: https://arxiv.org/abs/2307.07924
- Voyager: https://arxiv.org/abs/2305.16291
- Generative Agents: https://arxiv.org/abs/2304.03442
- G-Memory: https://arxiv.org/abs/2506.07398
- OAgents: https://arxiv.org/abs/2506.15741
- JoyAgent: https://arxiv.org/abs/2510.00510
- MIRIX: https://arxiv.org/abs/2507.07957
- EvolveR: https://arxiv.org/abs/2510.16079
- Agent KB: https://arxiv.org/abs/2507.06229
- MARTI: https://github.com/TsinghuaC3I/MARTI
- SoftCoT: https://arxiv.org/abs/2502.12134
- MemGen: https://arxiv.org/abs/2509.24704
- LatentSeek: https://arxiv.org/abs/2505.13308
- LatentMAS: https://arxiv.org/abs/2511.20639
- MemEvolve: https://arxiv.org/abs/2512.18746
- MiniLM: https://arxiv.org/abs/2002.10957
- t-SNE: http://www.jmlr.org/papers/v9/vandermaaten08a.html
- TriviaQA: https://aclanthology.org/P17-1147/
- KodCode: https://arxiv.org/abs/2503.02951
- StrategyQA: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100676
- PopQA: https://aclanthology.org/2023.findings-acl.985/
- BigCodeBench: https://arxiv.org/abs/2403.07974
- PDDL (Silver et al.): https://ojs.aaai.org/index.php/AAAI/article/view/30166
- Why MAS Fail (Cemri): https://arxiv.org/abs/2503.13657
- Bitter Lesson (Sutton): http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- Prefix-tuning: https://arxiv.org/abs/2104.08691
- P-tuning v2: https://arxiv.org/abs/2110.07602
- Perceiver: https://arxiv.org/abs/2103.03206
- Memory Networks: https://arxiv.org/abs/1410.3916
- MADDPG: https://arxiv.org/abs/1706.02275
- Self-RAG: https://arxiv.org/abs/2310.11511
- REPLUG: https://arxiv.org/abs/2305.05170
- G-Retriever: https://arxiv.org/abs/2402.07630
- AgentBoard: https://arxiv.org/abs/2401.13178
- Cemri et al. MAS failure: https://arxiv.org/abs/2503.13657
- Memory survey (Hu et al.): https://arxiv.org/abs/2512.13564
- LLM memory survey (Wu et al.): https://arxiv.org/abs/2504.15965

如果你想 deep dive 进任何一个 angle（比如 LMPO 的数值稳定性、role conditioning 的几何分析、或怎么把这个 idea 落到 nanoGPT 上做 toy experiment），我可以继续展开。
