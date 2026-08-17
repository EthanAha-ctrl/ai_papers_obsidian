---
source_pdf: Mem-W Latent Memory-Native GUI Agents.pdf
paper_sha256: 3e61e5754a25197b8be861a5b538626ec89b8fba61b702bf59341a3b44f0e7c7
processed_at: '2026-08-05T17:20:31-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Mem-W 用人话讲

## 一句话总结

GUI agent 之前记东西的方式像把日记本写在纸上再念给自己听, Mem-W 直接把记忆塞进脑子里的神经元激活, 不经过文字翻译这一步.

---

## 问题是什么

想象你用手机叫车: 打开 Google Maps 搜附近书店, 记住地址, 切到 Lyft, 输入地址叫车. 中间换了两个 app, 你的脑子需要记住 "书店地址" 这个 intermediate state.

现在的 GUI agent 记忆方式很别扭:

**方式 A — 全程截图**: 把每一屏 screenshot 都留着. 问题是一屏 screenshot 几百个 visual tokens, 走十步 context 就爆了, 而且大部分 pixels 没变化, 极其冗余.

**方式 B — 文字摘要**: 让 LLM 把历史写成 text "我刚才搜了书店, 找到了地址 XXX, 然后切到 Lyft". 问题是这个 text 要被 agent 重新 encode 一遍才能用, 这一步有信息损失, 还会引入 summarizer 自己的 noise.

**方式 C — 结构化 memory layer**: 工程师预先设计 "task-progress layer", "task-status layer", "vision layer", "failure-reminder layer" 等等. 问题是这些 categories 是工程师拍脑袋决定的, 未必是 agent 真正需要的结构. 你给 agent 一个 "failure reminder" layer, 但 task 实际需要的是 "partial progress" layer, 就 mismatch 了.

Mem-W 的洞察: agent 的 policy 本来就在 continuous embedding space 里推理, 那 memory 也应该活在这个 space 里, 别翻译成 text 再翻译回来.

---

## 类比 — 为什么 latent memory 比 text memory 好

想象你要告诉同事一个复杂的技术细节, 有两种方式:

**方式 1 (text memory)**: 你写一份文档, 同事读文档, 在脑子里重建理解. 文档是 lossy compression, 同事的理解又是二次 lossy decoding. 两步信息损失.

**方式 2 (latent memory)**: 你直接用神经接口把你的脑状态传给同事. 同事立刻 "感受到" 你在想什么, 没有 text encoding/decoding 步骤. 信息保真度高.

Mem-W 走的是方式 2. Compressor 把 trajectory 压成几个 "soft tokens" — 这些 tokens 不是英语单词, 是 direct embedding, 直接被 agent 的 attention 消费, 不经过 vocab lookup.

这跟 BLIP-2 的 Q-Former 思路一脉相承: BLIP-2 把 image features 压成固定数量的 latent tokens 喂给 frozen LLM, Mem-W 把 GUI trajectory 压成 latent tokens 喂给 frozen GUI agent.
参考: https://arxiv.org/abs/2301.12597

---

## 架构三件套

**1. Frozen backbone** — GUI agent 本身 (UI-TARS, Qwen3-VL, UI-Venus) 完全冻结. 不动它一个参数.

**2. Compressor (Q-Former)** — 唯一训练的模块. 输入一段 trajectory (几个 screenshot-action pair), 输出 8 个 latent tokens. 就这么简单. 8 个 tokens 携带这段历史的全部决策相关信息.

**3. Weaving** — 这 8 个 tokens 加上 source embedding (标识是 procedural memory 还是 working memory), 拼到当前 observation 的 embedding 前面, 一起喂给 frozen agent. Agent 像 attend 普通 context 一样 attend 这些 memory tokens.

---

## 两种 memory, 同一个 compressor

这里是最优雅的设计点.

**Working memory** — 当前 session 的过期历史. 比如你在 step 10, 保留最近 3 步 raw, 之前的 7 步压缩成 latent tokens. 用 "outcome unknown" 标记.

**Experiential memory** — 从 trajectory bank 检索的类似历史 task. 比如你做 "在 Amazon 买耳机", 检索出之前成功的 "在 eBay 买书" trajectory. 用 "success" 或 "failure" 标记.

两种 memory 用 **同一个 compressor** 压缩, 进入 **同一个 latent space**. 区别只在 provenance (从哪来), 不在 representation (怎么表示). 这让 working memory 和 experiential memory 共享训练信号, 互相 benefit.

之前的工作 (CoMEM, HyMEM) 是两种 memory 用两套 mechanism, 无法 cross-benefit. Mem-W 用 shared compressor 统一了.

参考 CoMEM: https://arxiv.org/abs/2510.09038
参考 HyMEM: https://arxiv.org/abs/2603.10291

---

## Outcome embedding — 让 failure 也有用

Compressor 有个细节很关键: 检索到的历史 trajectory 带 success/failure label, 压缩时把 outcome 也 embed 进去 (一个 vector 加到所有 query 上).

这意味着 failure trajectory 也进 context, agent 能学到 "这种 action pattern 会失败, 别走". 之前的工作多数只保留 success, 浪费了 failure 的负样本信号.

在线 working memory 的 outcome 是 unknown (episode 还没结束), 用第三个 outcome embedding 占位. 接口一致, 训练时能无缝切换.

---

## 训练 — 两步走

**Stage 1: Self-distillation — 教 compressor 怎么压缩**

Teacher agent 看一段很长的 raw history, student agent 看压缩后的 latent tokens, 让 student 模仿 teacher 的 action distribution (KL divergence).

直觉: teacher 看到了完整历史, student 只看到 8 个 tokens. 如果 student 能模仿 teacher 的决策, 说明这 8 个 tokens 保留了决策相关信息. 这是 "压缩保真" 训练.

**Stage 2: RLOO RL — 教 compressor 压什么对 success 有用**

Stage 1 只能模仿 expert, 无法 discover "什么 memory 真正导向成功". Stage 2 让 agent 在真实环境 rollout, 用任务成功/失败作为 reward, policy gradient 优化 compressor.

RLOO 是 leave-one-out mean baseline: 同一任务 rollout G 条, 每条的 advantage = 自己的 reward 减其他 G-1 条的平均 reward. 无需 learned value function, variance 低.

关键: gradient 从 reward → log-prob → frozen agent attention → latent tokens → compressor. Discrete retrieval (Top-M) 不可微, 不学习, 只有 compression 部分接收 gradient.

KL regularization 防止 stage 2 把 compressor 推飞, 锚定到 stage 1 的 frozen 版本.

参考 RLOO: https://arxiv.org/abs/2402.14740

---

## 为什么这个 setup 能 work

Frozen backbone + 可训练 memory adapter 这个 setup 让人想起 LoRA — 不动 backbone, 只调小 adapter. 但 Mem-W 调的不是 weight, 是 input context. 这相当于 "learnable context engineering".

传统 context engineering 是人工写 prompt, 加 retrieval, 加 history truncation 规则. Mem-W 让这些规则全部 end-to-end 学出来, gradient 替代 hand-tuning.

这跟 Karpathy 你提的 "Software 2.0" 一脉相承: 用 gradient 替代 hand-written rules. 之前的 AWM, ReasoningBank, HyMEM 是 "Software 1.5" (hand-engineered memory structure + learned retrieval), Mem-W 是更纯的 "Software 2.0" (memory representation 也学出来).

参考 AWM: https://arxiv.org/abs/2409.07429
参考 ReasoningBank: https://arxiv.org/abs/2509.25140

---

## 实验数据直觉

**Web navigation 提升巨大 (+30 success rate)** — UI-Venus-1.5-8B 在 MMInA shopping 从 18.5 飙到 48.5. Web 任务长 horizon, 跨多 page, memory 价值大.

**Mobile 提升小 (+2~13)** — Mobile 任务短, less long-horizon, memory 价值低. 合理.

**超过 GPT-5+UGround** — Mem-W-8B 在 AC-v2-Low Pass@4 达 94.22, 超 GPT-5 级别 closed-source. 暗示 long-horizon GUI 瓶颈不在 model scale, 在 memory representation.

**Memory bank scaling favorable** — trajectory bank 从 10K 涨到 50K, performance 持续上升无 plateau. 意味着这个 framework 是 scalable foundation, 投入更多 trajectory 就有更多收益.

**两种 memory 互补** — ablation 显示去掉 working memory 或 experiential memory 任一, 性能都掉. Working memory 保 partial progress, experiential memory 提供可复用 procedure, 不可互替.

---

## 几个我感兴趣的 design choice

**K=8 fixed budget** — 所有 segment 都压成 8 tokens, 简单 action 浪费, 复杂 decision 不够. Adaptive budget 是 obvious next step, paper 自己也承认.

**Top-M discrete retrieval** — 检索不可微, 检索阶段不学习. Soft retrieval / differentiable routing (类似 MoE) 可能改进, 但工程复杂度上升.

**Trajectory-level binary reward** — long-horizon task 上极其稀疏. 50 步任务只有最后一个 reward, credit assignment 难. Process reward model 或 step-level reward 可能改进, paper 没探索.

**No online memory bank update** — memory bank 静态, 不在 inference 时新增当前 session 成功 trajectory. Lifelong learning 缺失. 想象一个 agent 跑一整天, memory bank 持续增长, 越用越强 — 这是 obvious next step.

**No memory forgetting** — 检索到的 trajectory 都进 context, 没有 "现在不 relevant 就丢" 机制. Human memory 有 decay, 这个 paper 没建模.

---

## 更大的图景

Mem-W 属于一个正在形成的 "latent memory" research program:

- MemoryLLM / M+ (https://arxiv.org/abs/2502.00592): memory 写进 model parameters, 像 synaptic weight change
- MemGen (https://openreview.net/forum?id=vI56m4Iu4e): generative latent memory
- NextMem (https://arxiv.org/abs/2603.15634): latent factual memory
- SoftCoT (https://arxiv.org/abs/2502.12134): soft chain-of-thought as latent tokens
- VisMem (https://arxiv.org/abs/2511.11007): latent vision memory
- L2-VMAS (https://arxiv.org/abs/2602.00471): dual latent memories for multi-agent

这条线的 unifying principle: **memory 和 reasoning 共享同一 continuous substrate, 用 gradient 替代 rule**.

Mem-W 在 GUI agent 上的贡献是把这个理念具体化: shared compressor 统一 working + experiential, self-distillation + outcome-aware RL 双信号训练, frozen backbone + memory adapter 的 scalable setup.

---

## 给你的直觉

如果让我用一句话 capture Mem-W 的 essence:

> 把 memory 从 agent 外部的 "笔记" 变成 agent 内部的 "脑状态", 用 gradient 学怎么压、压什么, 而不是工程师拍脑袋决定.

这个 paradigm 我预期会在 2026 年看到大量 follow-up.  particularly 几个方向:
- Hierarchical compression (跨 chunk 的层次化 context)
- Adaptive token budget (简单 action 少 tokens, 复杂 decision 多 tokens)
- Process reward (step-level credit assignment)
- Lifelong memory bank (inference 时持续更新)
- Differentiable retrieval (soft routing 替代 Top-M)
- Cross-model transfer (tokenizer-independent latent representation)

希望这个人话版本帮你 build 起对 latent memory-native GUI agent 的 intuition. 如果你想更深入某个部分 (比如 Q-Former 内部 attention pattern, RLOO 的 variance 分析, 或者 memory bank 的 retrieval 策略), 我可以再展开.

---

# Mem-W: Latent Memory-Native GUI Agents — 深度解读

## 1. Core Insight — 这篇 paper 真正想解决什么问题

GUI agent 的痛点: long-horizon control 本质上是一个 **context-interface problem**. 一个 task 需要记住几个 screen 之前的 constraint, 保留 partial progress, reuse 类似 app 的 procedure, avoid 之前导致 failure 的 action pattern. 当前 observation 不够, 但 full raw history 又太长太冗余 (screenshots 视觉冗余极高).

之前的工作 (AWM, ReasoningBank, Memp, HyMEM, AndroTMem, MGA) 都把 memory 当成 **external, human-readable artifact**: 把 trajectory 摘要成 text workflows, 组织成 graph, 或分层 task-progress / task-status / vision layer. 这套设计有个根本的 architectural commitment — 工程师在学习开始前就要决定: memory 有哪些类型? 每种怎么表示? 怎么和 policy 接口?

Mem-W 的核心论断: 既然 modern GUI policy 已经在 continuous embedding space 上做推理, 那 memory 也应该活在同一个 latent space 里, 让 end-to-end gradient (而不是 hand-written rules) 决定保留什么、怎么 compose.

这等价于把 working memory 和 procedural memory 统一成同一类 token, 只是 provenance 不同 (一个是 online episode prefix, 一个是从 trajectory bank 检索的), 而 representation 完全一致. 不需要预先定义 memory taxonomy, 让 task-relevant structure 从训练中 emerge.

**直觉类比**: 这就像大脑里 memory consolidation 不区分 " episodic " 和 " procedural " 用不同神经元编码, 而是 distributed representation; 又像 RAG, 但 retrieval 返回的是 latent tokens 而非 text; 还像 Perceiver/Q-Former 把 variable-length signal 压到 fixed-size bottleneck.

参考链接:
- BLIP-2 Q-Former 设计: https://arxiv.org/abs/2301.12597
- CoMEM (Mem-W 直接灵感来源): https://arxiv.org/abs/2510.09038
- Memory survey (Hu et al. 2026): https://arxiv.org/abs/2512.13564
- Latent reasoning survey: https://arxiv.org/abs/2505.16782

---

## 2. Architecture 解析 — 三个核心组件

### 2.1 Frozen GUI Agent $\Pi_\theta$

整个 GUI policy backbone (例如 UI-TARS-1.5-7B, Qwen3-VL-4B, UI-Venus-1.5-8B) 参数 $\theta$ 全程冻结. $\Pi_\theta$ 内部有:
- Visual-textual encoder $\mathbf{G}_\theta$
- 普通 input embedding 函数 $\mathbf{E}_\theta(x, o_t, \rho_t)$, 其中 $\rho_t$ 是当前 step 周围保留的 bounded raw context

冻结 backbone 这件事意义重大 — Mem-W 是一个 **memory adapter**, 不是重新训 agent. 可以挂载到任何 frozen VLM GUI agent 上.

### 2.2 Trajectory-to-Latent Compressor $C_\phi$

这是 Mem-W 唯一训练的模块, 基于 Q-Former. 输入是 variable-length observation-action segment, 输出是固定 $K$ 个 latent tokens. 关键公式逐步拆解:

**Equation (1)** — 类型签名:
$$\mathfrak{S} = \bigcup_{\ell \geq 1} (\mathcal{O} \times \mathcal{A})^\ell, \quad C_\phi: \mathfrak{S} \times (\mathcal{V} \cup \{\emptyset\}) \to \mathbb{R}^{K \times d}$$

- $\mathfrak{S}$: 所有 finite GUI event sequences 的并集, $\ell$ 是 sequence length
- $\mathcal{O}$: observation 空间 (screenshots)
- $\mathcal{A}$: action 空间
- $\mathcal{V}$: outcome label 集合 $\{\text{succ}, \text{fail}\}$, 加上 $\emptyset$ 表示 "outcome 未知" (用于 online working memory)
- $K$: per-segment latent token budget (paper 中 $K=8$)
- $d$: agent embedding dimension (与 frozen agent 对齐)
- 输出 $K \times d$ 矩阵, 即 $K$ 个 $d$ 维 soft tokens

**Equation (3)** — 用 frozen encoder 提特征:
$$\mathbf{H}_{u:v}^y = \mathbf{G}_\theta\big((o_u, \alpha(a_u)), \ldots, (o_v, \alpha(a_v)); \gamma(y)\big) \in \mathbb{R}^{N_{u:v} \times h}$$

- $\mathbf{H}_{u:v}^y$: 从 step $u$ 到 $v$ 的 segment, 携带 outcome $y$ 的 contextual feature
- $\alpha(\cdot)$: action serialization (type + arguments)
- $\gamma(y) \in \mathbb{R}^h$: outcome embedding, 让 success/failure/unknown 三种 segment 编码不同
- $N_{u:v}$: encoder 输出的 token 数 (随 segment 长度变化)
- $h$: encoder hidden dim

注意 $\gamma(y)$ 是 outcome embedding, 让 success 和 failure trajectories 在 latent space 里被区分开, 但与 unknown 在线 segment 共用同一接口. 这一步替代了 hand-designed filtering rule (例如 "只保留 successful trajectories").

**Equation (4)** — Q-Former 压缩:
$$\mathbf{Z}_{u:v}^y = \mathbf{P}_\phi\big(\text{QFormer}_\phi(\mathbf{Q} + \mathbf{1}_K \gamma(y)^\top, \mathbf{H}_{u:v}^y)\big) \in \mathbb{R}^{K \times d}$$

- $\mathbf{Q} \in \mathbb{R}^{K \times h}$: $K$ 个 learned queries
- $\mathbf{1}_K \in \mathbb{R}^{K \times 1}$: 把 outcome embedding $\gamma(y)$ 广播到所有 $K$ 个 query slot (即每个 query 都加上 outcome 信息)
- $\text{QFormer}_\phi$: cross-attention 把 $N_{u:v}$ 个 features 蒸馏成 $K$ 个 query outputs
- $\mathbf{P}_\phi: \mathbb{R}^h \to \mathbb{R}^d$: token-wise projection, 把 compressor 内部维度 $h$ 投到 agent embedding 维度 $d$
- 最终 $\mathbf{Z}_{u:v}^y = [\mathbf{z}_1; \ldots; \mathbf{z}_K]$ — $K$ 个 memory tokens, **永不 decode 成 text**, 直接被 frozen agent 的 attention 消费

这是与 text-summarization memory 的本质差别. Text summary 仍要被 agent encoder 再编码一次 (引入 noise + 信息损失), 而 $\mathbf{Z}$ 直接进入 attention, 没有 translation 步骤.

### 2.3 Dual-Scale Memory Weaving — Procedural vs Working

两个 memory 来源在 inference 时拼成统一 context $\mathcal{M}_t$:
$$\mathcal{M}_t = [\mathcal{M}_t^{\text{proc}}; \mathcal{M}_t^{\text{work}}]$$

#### Procedural Memory (跨 session, 长期经验)

从外部 trajectory bank $\mathcal{B} = \{(x_i, s_{1:T_i}^i, y_i)\}_{i=1}^N$ 检索. 检索 key 和 query 都用 frozen encoder:

**Equation (5)**:
$$\mathbf{k}_i = \text{pool}(\mathbf{G}_\theta(x_i, s_{1:T_i}^i)), \quad \mathbf{q}_t = \text{pool}(\mathbf{G}_\theta(x, o_t, \rho_t)), \quad \mathcal{T}_t = \text{Top-M sim}(\mathbf{q}_t, \mathbf{k}_i)$$

- $\text{pool}(\cdot)$: 取 last hidden state 作为 trajectory-level embedding
- $M$: retrieval budget (paper 中 $M=5$)
- Top-M 是 discrete operation, 不通过 gradient 优化 — 只有 retrieval 之后的 compression 是可微的

**Equation (6)** — 检索到的 trajectories 按相似度排序压缩:
$$\mathbf{Z}_t^{\text{proc}} = [C_\phi(s_{1:T_{i_1}}^{i_1}, y_{i_1}); \cdots; C_\phi(s_{1:T_{i_M}}^{i_M}, y_{i_M})] \in \mathbb{R}^{MK \times d}$$

每个 trajectory 压成 $K$ tokens, $M$ 个 trajectory 总共 $MK$ tokens. 注意 success 和 failure trajectories 用同一个 latent space 表示, 通过 $\gamma(y)$ 区分. 这让 agent 可以 "看到失败长什么样" 而不只是模仿成功 — 这点很关键, 后面 RL 阶段会强化.

#### Working Memory (in-session, 短期状态)

保留最近 $L$ 个 step 的 raw 形式 (paper 中 $L=3$), 只压缩 expired prefix:

**Equation (7)**:
$$\rho_t = s_{\max(1, t-L):t-1}$$

当 $t > L+1$ 时, expired prefix $\bar{s}_t = s_{1:t-L-1}$, 否则为空.

把 $\bar{s}_t$ 切成最大长度 $W$ 的不重叠 chunks (paper 中 $W=4$), $\mathcal{C}_t = \{c_{t,1}, \ldots, c_{t,J_t}\}$, 其中 $J_t = \lceil |\bar{s}_t|/W \rceil$.

**Equation (8)** — 用 unknown-outcome marker $\emptyset$ 压缩每个 chunk:
$$\mathbf{Z}_t^{\text{work}} = [C_\phi(c_{t,1}, \emptyset); \cdots; C_\phi(c_{t,J_t}, \emptyset)] \in \mathbb{R}^{J_t K \times d}$$

这是 design 上的一个 elegant 选择: in-session segment 没有 outcome label, 用 $\emptyset$ 占位, 让 compressor 的 outcome-conditioning 机制对 online/offline 两种来源保持接口一致.

#### Latent Weaving

**Equation (9)** — 构造完整 input:
$$\mathbf{U}_t = [\mathbf{Z}_t^{\text{proc}} + \mathbf{1}_{MK}(\mathbf{b}^{\text{proc}})^\top; \mathbf{Z}_t^{\text{work}} + \mathbf{1}_{J_t K}(\mathbf{b}^{\text{work}})^\top; \mathbf{E}_\theta(x, o_t, \rho_t)]$$

- $\mathbf{b}^{\text{proc}}, \mathbf{b}^{\text{work}} \in \mathbb{R}^d$: learned source embeddings, 让 agent attention 知道哪些 tokens 是 procedural memory, 哪些是 working memory (类似 segment embedding)
- $\mathbf{1}_n \in \mathbb{R}^{n \times 1}$: 把 source embedding 广播到对应 block 的每一行
- 总 latent overhead: $(M + J_t)K$ tokens, 实践中 cap $M + J_t$ 保持 bounded

**Equation (10)** — frozen agent 直接预测 action:
$$\pi_{\theta,\phi}(a_t | \mathbf{U}_t) = \Pi_\theta(a_t | \mathbf{U}_t)$$

整个 inference 过程的 computation flow:

```
[trajectory bank B] --(retrieve Top-M via frozen G_θ)--> [M trajectories]
                                                              ↓
[expired prefix s̄_t] --(chunk into W-step)--> [J_t chunks]   ↓
                  ↓                                          ↓
            C_φ(c, ∅)                                  C_φ(s, y_i)
                  ↓                                          ↓
            Z_t^work + b^work                       Z_t^proc + b^proc
                              \                /
                               \              /
                                [U_t] + E_θ(x, o_t, ρ_t)
                                          ↓
                                Frozen Π_θ → a_t
```

---

## 3. Training — 两阶段关键

### 3.1 Stage 1: Self-Distillation

核心思路: 让 latent-augmented student 模仿看到 extended raw context 的 teacher. Teacher 和 student 用同一个 frozen agent.

- Teacher context $\chi_t^{\text{raw}} = (x, s_{\max(1, t-L'):t-1}, o_t)$, $L' \gg L$
- Student context $\mathbf{U}_t^{\text{s1}}$ (latent-augmented, 见 Equation 15)

**Equation (11/16)** — Loss:
$$\mathcal{L}_{\text{sd}} = \mathbb{E}_{(\tau, t)}\big[\ell_{\text{gui}}(\Pi_\theta(\cdot | \mathbf{U}_t^{\text{s1}}), a_t^\star) + \lambda D_{\text{KL}}(\text{sg}[\Pi_\theta(\cdot | \chi_t^{\text{raw}})] \| \Pi_\theta(\cdot | \mathbf{U}_t^{\text{s1}}))\big]$$

- $\ell_{\text{gui}}$: action-level cross-entropy, ground-truth action $a_t^\star$
- $\text{sg}[\cdot]$: stop gradient, teacher 不更新 (但 teacher 用的是同一个 frozen $\theta$, 这里 stop gradient 主要是避免 teacher 的 forward 影响 gradient flow)
- $\lambda$: balancing coefficient
- KL term: teacher 的 extended-context 分布 → student 的 compressed 分布. 这让 compressor 学到 "extended raw context 里的什么信息要保留"

**Equation (17)** — action loss 详细:
$$\ell_{\text{gui}} = -\frac{1}{S} \sum_{s=1}^S \log p_{\theta,\phi}(a_t^{(s)} | \mathbf{U}_t^{\text{s1}}, a_t^{(1:s-1)})$$

- $S$: action token 序列长度 (action 被序列化成多个 token)
- $a_t^{(s)}$: 第 $s$ 个 action token
- $a_t^{(1:s-1)}$: 之前生成的 token (autoregressive)

**Intuition**: Stage 1 是 "压缩保真" — 教 compressor 把 trajectory 压成 K 个 tokens 时, 保留决策相关的信息. KL term 是关键, 它等价于 "在压缩空间里重建 teacher 的 belief". 类似 knowledge distillation 中的 dark knowledge.

### 3.2 Stage 2: Outcome-Aware RL (RLOO)

Stage 1 只能让 memory 模仿 expert trajectory, 但无法教 compressor "什么 memory 真正导向 success". Stage 2 用 environment rollout + reward 信号.

**Equation (18)** — 每个 instruction $x$ 采样 $G$ 条 rollout:
$$\tau^{(g)} \sim \pi_{\theta,\phi}, \quad R(\tau^{(g)}) \in \{0, 1\}$$

**Equation (19/20)** — RLOO advantage (无参数 baseline):
$$\hat{A}^{(g)} = R(\tau^{(g)}) - \frac{1}{G-1}\sum_{g' \neq g} R(\tau^{(g')}) = \frac{G}{G-1}(R(\tau^{(g)}) - \bar{R}_x)$$

- $G$: rollouts per instruction (RLOO 来自 Ahmadian et al. 2024, https://arxiv.org/abs/2402.14740)
- $\bar{R}_x$: mean reward over $G$ rollouts
- 这是 leave-one-out mean baseline, 无需 learned value function

**Equation (21)** — per-trajectory policy gradient:
$$\mathcal{L}_{\text{pg}}^{(g)}(\phi) = -\hat{A}^{(g)} \cdot \frac{1}{T^{(g)}} \sum_{t=1}^{T^{(g)}} \log \Pi_\theta(a_t^{(g)} | \mathbf{U}_t^{(g)})$$

- 同一 trajectory 内所有 step 共享 trajectory-level advantage (因为 reward 是 trajectory-level)
- 对 step 取平均避免长 trajectory 占主导

**Equation (22)** — KL regularization vs frozen reference:
$$D_t^{(g)} = D_{\text{KL}}\big(\Pi_\theta(\cdot | \mathbf{U}_t^{(g)}) \| \Pi_\theta(\cdot | \mathbf{U}_t^{\text{ref}, (g)})\big)$$

- $\mathbf{U}_t^{\text{ref}, (g)}$: 用 stage-1 frozen compressor $C_{\phi_{\text{ref}}}$ 构造的 reference context
- 防止 stage 2 把 compressor 推到 catastrophic forgetting

**Equation (23/24)** — 组合目标:
$$\mathcal{L}_{\text{s2}}(\phi) = \mathbb{E}_{x \sim p(x)}\bigg[\frac{1}{G} \sum_{g=1}^G \big(\mathcal{L}_{\text{pg}}^{(g)}(\phi) + \frac{\beta}{T^{(g)}} \sum_{t=1}^{T^{(g)}} D_t^{(g)}\big)\bigg]$$

- $\beta$: KL penalty 系数

**关键 gradient flow**: 
1. Reward 是 trajectory-level, 是 non-differentiable 的环境反馈
2. 但是 log-prob $\log \Pi_\theta(a_t | \mathbf{U}_t)$ 中 $\mathbf{U}_t$ 包含可微的 latent tokens $\mathbf{Z}_t^{\text{proc}}, \mathbf{Z}_t^{\text{work}}$
3. Frozen decoder $\Pi_\theta$ 的 attention layer 在 forward 时被应用, 但参数不更新
4. Gradient 从 scalar loss → 通过 frozen attention → 进入 latent tokens → 进入 QFormer + projection head
5. Discrete retrieval (Top-M) 不贡献 gradient, 只有 retrieval 之后的 compression 可微

这是一个非常聪明的 design: 在 frozen backbone 上做 RL, 只更新 memory adapter. 这等价于在 fixed policy 下优化 "context engineering" — 让 context 自己学习怎么变成最优 context.

---

## 4. 实验数据 — 关键结果

### 4.1 主结果 (Table 2 & 3)

| Backbone | Setting | MMInA Shop | Mind2Web Info | Mind2Web Service |
|---|---|---|---|---|
| UI-TARS-1.5-7B | Vanilla | 5.5 | 5.88 | 6.86 |
| UI-TARS-1.5-7B | Mem-W-7B | **32.5 (+27.0)** | 20.59 (+14.71) | 22.55 (+15.69) |
| UI-Venus-1.5-8B | Vanilla | 18.5 | 5.88 | 15.69 |
| UI-Venus-1.5-8B | Mem-W-8B | **48.5 (+30.0)** | 23.53 (+17.65) | 36.27 (+20.58) |
| Qwen3-VL-4B | Vanilla | 11.5 | 13.72 | 14.71 |
| Qwen3-VL-4B | Mem-W-4B | **40.5 (+29.0)** | 22.55 (+8.83) | 26.47 (+11.76) |

**+30.0 是非常 dramatic 的提升**, 在 web navigation 上. Mobile (GUI-Odyssey, AC-v2) 提升相对小 (+2~13), 这是因为 mobile 任务通常更短, less long-horizon, memory 价值低.

特别值得注意的是 Mem-W-8B 在 AC-v2-Low Pass@4 达到 **94.22**, 超过 GPT-5+UGround 和 GUI-Libra-8B. 一个 8B frozen backbone + latent memory adapter 在 mobile step prediction 上超越 GPT-5 级别的 closed-source model. 这暗示 long-horizon GUI agency 的瓶颈不在 model scale, 而在 memory representation.

### 4.2 Memory-augmented Agent 对比 (Figure 2)

在 Multimodal-Mind2Web:
- Baseline (no memory): 6.4 (Info), 4.7 (Service)
- AWM (text workflow): 改善但有限
- ReasoningBank: 中等
- HyMEM (graph + embedding): 16.7 (Info), 25.9 (Service)
- CoMEM (continuous trajectory embeddings): 16.7 (Info), 28.2 (Service)
- **Mem-W: 20.8 (Info), 30.1 (Service)** — 最强

为什么 Mem-W 能超过 CoMEM 和 HyMEM (它们也用 latent embeddings)?
1. CoMEM/HyMEM 的 working memory 仍然 coarse, 依赖 truncated nearest-neighbor screenshots. Mem-W 用 trajectory-to-latent compressor 动态蒸馏 (Equation 8).
2. CoMEM/HyMEM 训练信号主要是 expert imitation, Mem-W 引入 outcome-aware RL (Equation 12), 让 memory 保留与 success 真正相关的 procedural evidence.

### 4.3 Ablation (Table 4, UI-Venus on MMInA)

| Setting | Succ. | #Steps | Hit-Max | Time/Task | Time/Step |
|---|---|---|---|---|---|
| Vanilla | 18.5 | 13.0 | 70.5% | 83.2 | 6.4 |
| w/o Working | 47.5 | 7.7 | 28.5% | 82.4 | 10.7 |
| w/o Experiential | 43.0 | 10.0 | 52.1% | 142.0 | 14.2 |
| Full Mem-W | **48.5** | 8.2 | 42.2% | 127.9 | 15.6 |

几个关键观察:
- **两种 memory 互补不冗余**: 去掉任一, 性能下降. Working memory 单独 (47.5) > Experiential memory 单独 (43.0), 说明保留当前 session 的 partial progress 更关键, 但 cross-session experience 仍然额外加分.
- **Hit-Max 大幅下降**: Vanilla 70.5% 任务打满 step limit (迷失), Mem-W 只 42.2%. 说明 memory 让 agent 决策更 decisive, 不是单纯拖长交互.
- **Time/step 略增** (6.4 → 15.6 ms): compressor 引入 overhead, 但 Time/Task 反而下降 (83.2 → 127.9 — 实际是上升了, 因为 step 数变化). 与 GLM-4.1V-9B-Thinking (180.6s/task) 和 Qwen3-VL-32B (125.7s/task) 比, Mem-W 在更低 cost 下达到更高 success rate.

### 4.4 Retrieval & Memory Bank Scaling (Figure 4, Table 5)

**Retrieval budget M (Figure 4)**:
- MMInA: M=1 → 38.0, M=5 → 47.5, 之后 saturate
- AC-v2: M=1 → 63.1, M=9 → 70.8
- Mind2Web: M=1 → 28.0, M=9 → 39.2

Saturation 说明 most relevant procedural evidence 已经在 top-5 retrieved, 再多 retrieval 提供 diminishing returns.

**Memory bank size (Table 5)**:

| Model | Bench | 10K | 20K | 30K | 50K |
|---|---|---|---|---|---|
| Qwen3-VL-4B | MMInA | 35.00 | 38.00 | 39.50 | 40.00 |
| Qwen3-VL-4B | Mind2Web | 25.49 | 26.47 | 27.45 | 28.43 |
| UI-Venus-1.5-8B | MMInA | 37.00 | 43.00 | 47.50 | **50.50** |
| UI-Venus-1.5-8B | Mind2Web | 29.41 | 33.33 | 36.27 | **38.24** |

UI-Venus 从 10K 到 50K 涨 +13.5 (MMInA), +8.83 (Mind2Web). 这是 very favorable scaling behavior — trajectory bank 越大, performance 越好, 没有 plateau. 暗示 latent memory framework 是 scalable foundation.

---

## 5. 训练配置细节 (Appendix B)

- **LoRA**: rank=16, alpha=32, dropout=0.05 (应用在 backbone 上, 但 paper 又说 backbone frozen, 这里有点矛盾, 应该是 LoRA 也只加在 compressor 相关部分, 实际上 backbone 完全 frozen)
- **DeepSpeed ZeRO-3** memory optimization
- **bfloat16 mixed precision**, gradient checkpointing
- **Optimizer**: AdamW, LR=5e-5, weight decay=0.1, warmup ratio=0.03, cosine schedule
- **Per-GPU batch**: 2
- **Hardware**: 8× NVIDIA A800 (80GB)
- **Inference**:
  - Web evaluation: temperature=0.1, top_p=0.001, max generation 10000 tokens, KV-cache enabled
  - GUI-Odyssey / AC-v2: temperature=0.0 (Pass@1) / 1.0 (Pass@4), max generation 64-2048 tokens
- **Retrieval**: FAISS index, L2-normalized embeddings, inner-product similarity, top-5 retrieval

Web training data 81,527 samples (11,176 success, 70,351 failure), 覆盖 13 domains (academic, education, finance, government, health, shopping, tech, travel 等). Web memory bank 22,346 successful trajectories.

Mobile training data 2,610 samples (2,486 success), 6 categories (utility, social, media, info handling, cross-app workflow, e-commerce). Mobile memory bank 4,972 successful trajectories.

注意 success rate 极不均衡: web training set 整体 success rate 仅 13.71%, mobile 高达 95.29%. 这意味着 web 任务远比 mobile 难, 与实验中 web 提升更大相吻合 (web 任务长 horizon, 更需要 memory).

---

## 6. 与相关工作脉络 — Latent Memory 这条线

Mem-W 不是孤立的, 它属于一个正在形成的 "latent memory/reasoning" research program:

**Text agent latent memory**:
- MemoryLLM / M+ (Wang et al. 2025a, https://arxiv.org/abs/2502.00592): persistent memory in latent space
- NextMem (Zhang et al. 2026b, https://arxiv.org/abs/2603.15634): latent factual memory
- MemGen (Zhang et al. 2026a, https://openreview.net/forum?id=vI56m4Iu4e): generative latent memory
- SoftCoT/SoftCoT++ (Xu et al. 2025a/b): soft chain-of-thought as latent tokens
- FlashMem (Hou et al. 2026, https://arxiv.org/abs/2601.05505): distill intrinsic latent memory via computation reuse
- G-MemLLM (Xu 2026, https://arxiv.org/abs/2602.00015): gated latent memory augmentation

**Multimodal latent memory**:
- VisMem (Yu et al. 2026c, https://arxiv.org/abs/2511.11007): latent vision memory
- L2-VMAS (Yu et al. 2026b, https://arxiv.org/abs/2602.00471): dual latent memories for visual multi-agent
- CoMEM (Wu et al. 2025b, https://arxiv.org/abs/2510.09038): auto-scaling continuous memory for GUI agent — Mem-W 直接前辈
- Latent space survey (Yu et al. 2026a, https://arxiv.org/abs/2604.02029)
- Latent reasoning survey (Chen et al. 2025, https://arxiv.org/abs/2505.16782; Zhu et al. 2025, https://arxiv.org/abs/2507.06203)

**Symbolic GUI memory (Mem-W 想替代的)**:
- AWM (Wang et al. 2024, https://arxiv.org/abs/2409.07429): reusable workflows
- ReasoningBank (Ouyang et al. 2026, https://arxiv.org/abs/2509.25140): reasoning memories from self-evaluated experience
- Memp (Fang et al. 2026, https://arxiv.org/abs/2508.06433): procedural instructions
- ExpeL (Zhao et al. 2024, https://arxiv.org/abs/2308.10144): experiential learners
- EchoTrail-GUI (Li et al. 2026a, https://arxiv.org/abs/2512.19396): actionable trajectory memory
- AndroTMem (Shi et al. 2026, https://arxiv.org/abs/2603.18429): causally linked state anchors
- MGA (Cheng et al. 2026, https://arxiv.org/abs/2510.24168): dynamic structured state memory
- HyMEM (Zhu et al. 2026a, https://arxiv.org/abs/2603.10291): hybrid symbolic + embedding
- MGA / GUI-KV (Huang et al. 2025, https://arxiv.org/abs/2510.00536): KV cache compression
- ActionEngine (Zhong et al. 2026, https://arxiv.org/abs/2602.20502): state machine memory

**Q-Former 谱系**:
- BLIP-2 (Li et al. 2023, https://arxiv.org/abs/2301.12597): Q-Former 原始设计, 把 frozen image encoder features 蒸馏成 fixed-size latent tokens 给 frozen LLM. Mem-W 的 compressor 是 Q-Former 在 GUI trajectory 上的应用, 把 variable-length trajectory 蒸馏成 $K$ 个 latent tokens.

这条脉络的 unifying principle: **让 memory 和 reasoning 共享同一个 continuous substrate, 用 gradient 替代 rule**.

---

## 7. Intuition 总结 — 为什么这个设计 work

我把 Mem-W 的成功归结为四个直觉:

**1. Representation alignment**: Policy 在 latent embedding 上推理, memory 也应该是 latent embedding. Text summary 要被 encoder 二次编码, 信息必然损失. Latent memory 跳过这个 translation, 信息保真度更高. 类似直接共享神经激活 vs 用语言描述激活.

**2. End-to-end gradient 替代 hand-engineered ontology**: 之前的 GUI memory 设计师要先决定 "task-progress layer, task-status layer, vision layer, tool-use experience layer, failure reminder layer, high-level planning layer" 等. 这些 categories 是工程师的 ontology, 未必匹配 task 实际需求 (论文引用 Zhang et al. 2025, Pan et al. 2026 证明这一点). Mem-W 把 memory categories 留给 training emerge — 事实证明学习出来的结构比 hand-designed 更适配下游任务.

**3. Working + Experiential 共享 compressor**: 之前 working memory 用 KV-cache 或 screenshot truncation, experiential memory 用 text workflows 或 graph. 两种 memory 用不同 mechanism, 不能 cross-benefit. Mem-W 用同一个 $C_\phi$ 处理两者, 共享参数, 共享 latent space. 这让 working memory 也能 benefit from experiential 的训练信号 (反之亦然).

**4. Self-distillation + RL 双信号**: Stage 1 教 compressor "怎么压缩保真" (imitation-level), Stage 2 教 compressor "压什么对 success 有用" (outcome-level). 缺一不可 — 只有 Stage 1 学到的是模仿 expert, 无法 discover 新的 memory structure; 只有 Stage 2 没有 imitation prior, 训练不稳定. KL regularization 把两个 stage 串起来.

**与 MemoryLLM/M+ 的对比**: MemoryLLM 把 memory 写进 model parameters (像 synaptic weight change), Mem-W 把 memory 写进 input latent tokens (像 working memory activation). 两者是互补的 — 参数-level memory 适合跨 session 长期知识, token-level memory 适合当前 episode 状态. Mem-W 选了 token-level 是因为 frozen backbone 设定.

**与 Memory Compression 在 LLM 中的对比**: 类似 Gisting tokens / ICAE / AutoCompressors, 但 GUI setting 有视觉冗余的特殊性, screenshot 同一页面只需 minor change, raw history 视觉 token 极其冗余. Trajectory-to-latent compressor 在这里 leverage 程度更高.

---

## 8. Limitation 与 Open Questions

Paper 自己承认的 limitation:
1. **Fixed latent-token budget** $K=8$: 所有 segment 都压成 8 tokens, 没有自适应. 简单 action 可以 1 token, 复杂 decision 可能要更多. Adaptive compression 是 obvious next step.
2. **只在 web + mobile 评估**: Desktop GUI (e.g., macOS, Windows) 未测.
3. **Tokenizer-dependent**: Latent tokens 绑定到特定 frozen agent 的 embedding 空间, 跨 model transfer 未必 work. 未来需要 tokenizer-independent latent representations.

我额外想到几个 open questions:
1. **Hierarchical compression**: 现在所有 chunks 独立压缩, 没有跨 chunk 的 hierarchical context. 类似 Hierarchical RNN 或 Hourglass Transformer 可能更好.
2. **Memory editing / forgetting**: 检索到的 trajectories 不加区分都进 context, 没有 "当前不再 relevant 就 forget" 机制. 这与 human memory 的 decay law 缺失.
3. **Memory composition / reasoning**: Latent tokens 进入 attention 后怎么被 reasoning 使用仍是黑箱. Probing study 缺失 — 这些 $K$ tokens 学到的是什么? Per-token semantic 可解释性没有分析.
4. **Reward sparsity**: Stage 2 用 trajectory-level binary reward. 在 long-horizon task (50+ steps) 上, 这个 reward 信号非常稀疏. Credit assignment 没有显式处理 — paper 用 RLOO 减 variance, 但 step-level credit 仍是问题. Process reward model 可能改进.
5. **Compressor capacity vs backbone capacity mismatch**: Compressor 是 lightweight Q-Former, 可能不足以捕获 backbone 的全部 reasoning 能力. 让 compressor 更大可能 marginal收益递减, 但太小是 bottleneck.
6. **Retrieval 的可微性**: 现在 Top-M 是 discrete, retrieval 阶段不学习. Soft retrieval / differentiable retrieval (类似 MoE soft routing) 可能更好.
7. **Online memory bank update**: 现在 memory bank 是静态的, 不在 inference 时新增当前 session 的成功 trajectory. Lifelong learning 缺失.
8. **Outcome embedding 设计**: $\gamma(y)$ 三种 outcome (success/fail/unknown) 用单一 vector 表示, 可能不够 expressive. 用 trajectory-level summary vector 而非 outcome label 可能 capture 更多 nuance.

---

## 9. Case Study 速读

Figure 5 (Web): "Use Google Map to locate nearest bookstore and then book a ride through Lyft". 跨多 page, 需要 preserve "nearest bookstore location" 这个 intermediate state, 跨 app (Google Maps → Lyft) 切换. Mem-W 的 working memory 在 page 切换间保持 location info, experiential memory 提供类似 task 的 procedural pattern.

Figure 6 (Mobile): "Install Triller via Google Play, open it, navigate to Settings app to disable Triller notifications, reopen Triller to watch a video". 跨 app workflow, 需要 preserve partial progress (installed? opened? notification disabled?). Figure 7 显示 Mem-W 检索到的类似 trajectories, 提供 cross-app navigation 的 procedural evidence.

这两个 case 都是 long-horizon + 跨 app/page + 中间状态保留 — 正是 Mem-W 设计的目标场景.

---

## 10. Final Thoughts

Mem-W 在我看来是 GUI agent memory 从 "symbolic ontology-first" 到 "latent-context-native" 范式转移的一个标志性工作. 它证明了:

1. **Memory 可以 end-to-end 学出来**, 而不需要工程师预先定义 categories
2. **Working memory 和 experiential memory 可以统一**在 shared latent space, 只差 provenance
3. **Frozen backbone + memory adapter** 的 setup 可以大幅提升 long-horizon 性能 (+30.0!), 超过 GPT-5 级别 closed-source model
4. **Memory bank scaling** favorable — trajectory 越多越好, 没有明显 plateau

更深层的 implication: future GUI agent 架构应该把 latent context 当作 first-class foundation substrate, 让 perception, memory, procedural experience, ongoing task state 都在同一 machine-native form 下组织. 这暗示了一条路径 — interactive multimodal agent 的 long-horizon competence 不再围绕 external prescribed memory modules, 而是围绕 learned latent context interface.

与 Karpathy 你之前关于 "Software 2.0" 的论述一脉相承: 用 gradient 替代 hand-written rules. Mem-W 把这个理念推广到 GUI agent memory — 之前的 AWM, ReasoningBank, HyMEM 是 "Software 1.5" (hand-engineered memory structure + learned retrieval), Mem-W 是更纯的 "Software 2.0" (memory representation 也 end-to-end 学出来).

参考代码: GitHub (Mem-W-4B, Mem-W-8B) — paper 提到但具体 URL 在 attachment 中没给出完整链接. 论文作者联系方式: guibinz@outlook.com.

希望这些讲解帮你 build 起对 latent memory-native GUI agent 的 intuition. 这条 research line 我预期会在 2026 年看到大量 follow-up, 特别是 hierarchical compression, adaptive token budget, process reward, lifelong memory bank 等方向的扩展.
