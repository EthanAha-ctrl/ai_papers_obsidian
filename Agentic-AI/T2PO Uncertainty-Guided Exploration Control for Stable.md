---
source_pdf: T2PO Uncertainty-Guided Exploration Control for Stable.pdf
paper_sha256: 0f2ab8dd1e0cd772e6937ba86c16f00c351a29cdcf3ff82782595666c8e3e502
processed_at: '2026-08-12T11:57:42-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

` token 并切换到 `<action>` 模式。提前停止“过度思考”。
        *   **Turn-Level Dynamical Sampling (TDS)：** 如果连续轮次之间的 turn-level uncertainty signal ($\Phi^k$) 变化量 $\Gamma^k$ 过小（意味着 agent 在重复它自己），则废弃该轮次并重新采样直到不同为止，或者达到预算上限。
        *   **Policy Update：** 基于 GiGPO，具有 group-in-group advantage estimation，memory context window（仅保留最近 $P$ 轮），格式惩罚，以及 RFT warm start。

3.  **起草解释（Karpathy 风格——直觉优先，技术深度第二，广泛的联想）：**
    *   *语气：* 直接、直观、与 AI/RL 概念深度关联。
    *   *结构：*
        1.  核心直觉（“人话”总结）
        2.  问题：Multi-Turn Agentic RL 中的 Hesitation（Token 与 Turn 层面）
        3.  T²PO 的解法：Self-Calibrated Uncertainty Signal（为什么单独用熵/置信度会失效，公式拆解）
        4.  Token-Level Thinking Intervention (TTI)（如何强制截断，公式拆解）
        5.  Turn-Level Dynamical Sampling (TDS)（如何打断死循环，公式拆解）
        6.  Policy Optimization 与细节（GiGPO、RFT、Memory window）
        7.  实验数据直觉

4.  **处理特定约束：**
    *   *除了逻辑连接词外用英文：* 是的，保持名词为英文（例如，“multi-turn agentic RL”、“hesitation”、“uncertainty”、“entropy”、“confidence”、“rollout”、“token”、“turn”、“advantage”、“policy”）。
    *   *详细公式：* 拆解 $H_t$、$C_t$、$\tilde{H}_t$、$\tilde{C}_t$、$M_t$、TTI 条件、$\Delta_t^k$、TDS 条件、$\Gamma^k$、$\Phi^k$。解释每个符号。
    *   *建立直觉：* 为什么在过度思考时熵会饱和？为什么我们对 top-j tokens 的概率求平均？
    *   *没有“不是...而是”：* 小心改写。“The problem is X.” “We solve this by Y.” 避免使用“X is not A, but B”这种句式。取而代之的是，“This metric has flaws. We fix it with...”
    *   *包含链接：* 引用相关的论文/概念（例如，GRPO, GiGPO, DAPO, DeepSeek-R1, RAGEN）。
    *   *产生幻觉/宽泛的联想：* 与 DeepSeek R1 的 aha moment、DAPO 的 dynamic sampling、overthinking in o1/R1、POMDPs、Active inference（Friston）、KL divergence collapse 等联系起来。

5.  **逐步写作与翻译：**

    *   *Intro:* 这篇 paper 讲的是怎么训练 LLM agents 在 multi-turn 环境里稳定地做 RL。目前大家训 multi-turn agentic RL 经常会遇到 training collapse（训练崩溃），success rate 掉到零，KL divergence 炸掉。作者发现崩溃的根源在于 **hesitation**（犹豫），也就是模型在瞎溜达，既没有减少 uncertainty，也没有推进任务。

    *   *Hesitation 两种表现：*
        1.  Token level: Overthinking（过度思考）。CoT 越来越长，信息增益早就饱和了，模型还在那废话，sampling noise 越积越多，导致 policy gradient variance 极大。这就好像一个人碎碎念半天不干活。
        2.  Turn level: 陷入死循环。Agent 在环境里偏离正确路线后，连续好几个 turn 生成语义极其相似但注定失败的 reasoning trace。Rollout 预算被白白浪费。

    *   *Self-Calibrated Uncertainty Signal（公式 1-3）：*
        要控制 hesitation，首先得有一个信号来衡量模型到底有没有在“认真探索”。通常大家用 Entropy ($H_t$) 或者 Confidence ($C_t$)。但两者都有盲区。
        Entropy $H_t = -\sum p_t^{(i)} \log p_t^{(i)}$。当 vocab 很大（比如 Qwen3 的 152K）时，分布极不均匀（比如 1,0,0...）和（0.5,0.5,0...）的 entropy 差距只有 $\log 2$，在 $[0, \log V]$ 的总尺度上几乎看不出来。
        Confidence $C_t = -\frac{1}{j} \sum_{i=1}^j \log p_t^{(i)}$。这玩意儿只看 top-j 的概率，完全不管后面的尾部概率怎么分布。如果两个分布 top-1 概率一样，但尾部天差地别，Confidence 是一样的。
        作者把这两者融合：先归一化 $\tilde{H}_t, \tilde{C}_t$，然后通过参数 $\alpha$ 加权：$M_t = \alpha \tilde{H}_t + (1-\alpha)(1-\tilde{C}_t)$。
        这个 $M_t$ 就是一个局部的 distributional stability 指标。$M_t$ 越高，说明模型越不确定。

    *   *Token-Level Thinking Intervention (TTI)（公式 4-6）：*
        怎么停止 overthinking？监控 $M_t$ 的变化率。$\Delta_t^k = |M_t^k - M_{t-1}^k|$。如果在过去 $N$ 个 token 的滑动窗口内，平均变化量 $\frac{1}{N+1} \sum \Delta_{t-i}^k < \varepsilon$，说明 predictive distribution 已经收敛了，再生成更多 token 意义不大。
        这时候怎么办？强制干预！把 logits $z_{t^*+1}$ 里面 `` 这个 token 的 logit 设为 $+\infty$，其他全设为 $-\infty$。这样模型百分之百吐出 ``。接着塞入一个固定的 queue `['', '\n', '<action>']`，强行把推理阶段切换到执行阶段。
        这个设计很巧妙，因为它截断在“信息增益停止”的地方，如果截断在 uncertainty 最高峰，可能把关键的任务 token（比如商品名）给切掉了。任务 token 往往是连续一段的，用 sliding window 能平滑掉这种局部的 spike。

    *   *Turn-Level Dynamical Sampling (TDS)（公式 7）：*
        一个 turn 生成完了，怎么判断它有没有在原地打转？把一个 turn 里面所有的 token 的 $M_t$ 做几何平均（Geometric mean），得到 turn-level observation signal $\Phi^k = (\prod_{t=1}^T M_t)^{1/T}$。
        如果当前 turn 的 $\Phi^k$ 和上一个 turn 的 $\Phi^{k-1}$ 差不多，也就是变化量 $\Gamma^k = |\Phi^k - \Phi^{k-1}| < \eta$，说明这个 turn 白做了，模型内部状态没变。
        怎么办？直接丢弃这个 turn，重新 rollout！一直重新生成，直到 $\Gamma^k \ge \eta$ 或者达到最大重采样 budget $B_{\max}$。这保证了 trajectory 的多样性，避免了在死胡同里浪费计算资源。

    *   *Policy Update 细节：*
        基于 GiGPO (Group-in-Group Policy Optimization)。
        Credit assignment 依然很难。用了 discounted return $R(\tau^k) = \sum \beta^{j-k} r^j$，把 terminal reward 往前传。
        Advantage 融合了 trajectory level 和 turn level：$A'(\mathbf{a}_i^k) = A(\tau_i^k) + \omega A^{\text{turn}}(\mathbf{a}_i^k)$。
        Memory context window：只保留最近 $P$ 个 turn 的历史，不把整个 trajectory 拼进去，省 memory 算力。
        RFT cold start：先用 base model 生成数据，过滤掉成功率低的轨迹，SFT 一把，确保模型一开始能输出格式正确的 `<action>...</action>`。
        Format penalty：格式不对直接扣分 $r_i \leftarrow r_i - \lambda_{\text{fmt}}$。

    *   *实验结果直觉：*
        WebShop 和 ALFWorld 上，T²PO 把 Success Rate 提升到了 80%+，而 GiGPO 只有 70%+。
        Token 效率巨高：同样预算下，T²PO 成功的 trajectory 用的 token 更少，而且极少触发 max length clipping。
        Turn 效率也高：完成任务用的 turn 数大概少了 20-25%。
        从消融实验看，去掉 RFT 掉点最严重，说明 cold start 极其重要。去掉 TTI 和 TDS 都会掉点，证明这两层控制缺一不可。
        对比其他 thinking control 方法（比如 hard thinking budget，lengthy reward），T²PO 完胜，因为那些都是全局 heuristic，没有针对 per-token 或者 per-turn 的动态状态做自适应。

    *   *相关联想与 Hallucination：*
        这让我想起 DeepSeek R1 里的 aha moment，模型自己学会了反思。T²PO 算是用外部机制强行阻止了没意义的反思。
        和 DAPO 的 dynamic sampling 区别：DAPO 是在 prompt level 过滤 trivial prompt，T²PO 是在 turn level 过滤 trivial turn。
        和 Active Inference 的联系：Friston 提出的 Active inference 里面，agent 要最小化 surprise（也就是 uncertainty）。当 agent 无法减少 uncertainty 时，就应该改变 action（换策略），这和 T²PO 监控 $\Delta M_t$ 小于阈值就强制停掉是殊途同归的。
        POMDP 视角：LLM 在 multi-turn 里面只看到 partial observation，uncertainty 信号其实是 belief state 的 proxy。

    *   *公式变量细节：*
        一定要拆解！
        $p_t^{(i)}$: 第 $t$ 步生成词表中第 $i$ 个 token 的概率。
        $V$: Vocabulary 大小。
        $j$: Top-j tokens 的数量。
        $\alpha$: 融合系数，WebShop 实验里选了 0.4 最好。
        $L_{\min}$: 最小前缀长度，防止过早截断。
        $N$: 滑动窗口大小。
        $\varepsilon$: token 级容忍度，实验里 1e-4。
        $t^*$: 触发干预的那个 token 位置。
        $\eta$: turn 级容忍度，实验里 1e-3。
        $B_{\max}$: 最大重采样预算。
        $\beta$: turn 级 discount factor。
        $G$: rollout group size。
        $\omega$: 融合 trajectory advantage 和 turn advantage 的权重。

    *   *链接添加：*
        - GRPO: https://arxiv.org/abs/2402.03300
        - GiGPO: https://arxiv.org/abs/2505.10978
        - DAPO: https://arxiv.org/abs/2503.14476
        - DeepSeek R1 (Aha moment): https://arxiv.org/abs/2501.12548
        - verl framework: https://arxiv.org/abs/2409.19256
        - RAGEN: https://arxiv.org/abs/2504.20073
        - WebShop: https://arxiv.org/abs/2207.01206
        - ALFWorld: https://arxiv.org/abs/2010.03768

6.  **对照约束条件审查：**
    - 英文名词/中文逻辑词？是的。
    - 没有“不是...而是”？检查完成。
    - 详细的公式解释？是的。
    - 实验数据表？是的。
    - 建立直觉？是的，使用了类比（例如，“瞎溜达”、“碎碎念”、“死循环”）。
    - 广泛的联想？Active inference, POMDP, DeepSeek R1。
    - 尽可能多的回复？是的，将彻底覆盖所有内容。

7.  **最终润色：** 确保语气尊重且充满热情，直接与 Andrej 对话。确保公式部分的 Markdown 格式正确无误。确保行文流畅，在直觉和技术深度之间平滑过渡。Andrej, 這篇 paper 其實在處理一個非常直觀且痛點明確的問題。我們在訓練 LLM agent 做 multi-turn reinforcement learning 時，經常遇到 training collapse（success rate 突然掉到零，KL divergence 爆炸）。過去的解法大多是在 trajectory level 做 filtering 或是加 process reward，但這篇 paper 發現 collapse 的根本原因是模型在「瞎溜達」，作者稱之為 **hesitation**。

這篇 paper 提出的 **T²PO** (Token- and Turn-level Policy Optimization) 核心思路就是：用模型內部的 uncertainty 信號作為探測器，在 token level 強行打斷「過度思考」，在 turn level 強行重置「死循環」，從而把 exploration 控制在一個高效的範圍內。

以下我為你拆解其中的技術細節、公式變數以及背後的 intuition。

---

### 1. 核心病灶：Hesitation (猶豫)

在 multi-turn 場景中，LLM 的 hesitation 表現在兩個層面，這兩者都會引入巨大的 noise，導致 policy gradient 變異數爆炸：

1.  **Token-level hesitation (Overthinking)**：模型在 `

---

# T²PO: Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic RL — 深度解析

## 1. Paper 的核心问题与动机

这篇 paper 要解决的是 **multi-turn agentic reinforcement learning** 中的 **training collapse** 问题。作者观察到,即使是 SOTA 方法如 GiGPO + DAPO,在不同 environment random seeds 下也会出现 success rate 突然崩塌、KL divergence 与 gradient norm 爆炸的现象(Figure 1)。

作者把这个根因诊断为 **hesitation**(犹豫),这是 exploration–exploitation tradeoff 的系统性违反,具体表现为两种模式:

- **Token-level hesitation (overthinking)**:LLM agent 在 `
