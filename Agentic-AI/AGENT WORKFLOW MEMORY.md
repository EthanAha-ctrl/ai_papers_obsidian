---
source_pdf: AGENT WORKFLOW MEMORY.pdf
paper_sha256: ddc6f6fedec48c88ae743b43a464f7eb4aecaab05acd18c29cfb657ab445ebe4
processed_at: '2026-07-18T04:28:04-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

这篇 paper 提出了 **Agent Workflow Memory (AWM)**，一种能够让 LM-based agents 从过往经验中提取可重用工作流，并且将这些工作流注入到 memory 中以指导未来任务执行的方法。这种机制类似于人类在解决复杂问题时会总结出常规的"套路"或者标准操作程序。所以，当 agent 面对长序列、多步骤的 web navigation 任务时，它可以通过 AWM 积累出诸如 "find a place by its name" 这样的基础 workflow，并且在此基础上进一步组合出 "get the zip code of a place" 这种更复杂的 workflow，从而产生一种滚雪球式的性能提升。

### 核心方法与公式解析

在 web navigation 任务中，agent 拥有一个 language model backbone $L$ 和一个文本化的 memory $M$。为了解决一个由自然语言指令 $q$ 指定的任务，agent 在环境 $T$ 中通过 observe-act 循环进行交互。在每个时间步 $t_i$，环境状态 $s_i$ 产生观察 $o_i$，模型结合指令与记忆生成动作 $a_i$：

$$ L(q, M, o_i) \to a_i $$

这里 $L$ 代表 language model backbone，$q$ 代表 natural language instruction，$M$ 代表包含 built-in actions (如 `CLICK`, `TYPE`) 文档的基础 memory，$o_i$ 是在时间步 $t_i$ 获得的环境 observation，而 $a_i$ 是生成的 action。随后环境发生状态转移 $T(s_i, a_i) \to s_{i+1}$。

AWM 的核心在于引入了一个 induction module $I$。对于一个成功的 experience $e = (q, P^e)$ (其中 $P^e = (p_1^e, \dots, p_n^e)$ 是解决任务的动作轨迹)，AWM 通过 induction module 诱导出一组 workflows $\mathcal{W}$：

$$ I(\bar{\varepsilon}) \to \mathcal{W} = \{w\} = \{(d_j, P_j^d)\} $$

其中 $I$ 是 induction module，$\bar{\varepsilon} = \{e_i\}_{i=1}^m$ 是包含 $m$ 个 experiences 的集合。每一个 workflow $w$ 由两部分组成：$d_j$ 是用自然语言描述的 workflow 目标，$P_j^d$ 是完成该目标的一系列步骤。每个步骤 $p$ 包含三个部分：当前环境状态的 NL 描述，agent 的 reasoning 过程，以及一个可执行的程序化 action。

当 workflows 被诱导出后，它们会被整合到 agent 的 memory 中，形成 augmented memory $M_w = M + \mathcal{W}$。在后续推理时，动作的生成公式变为：

$$ L(q, \dot{M}_w, o) = L(q, \dot{M} + \mathcal{W}, o) \to a $$

这样，agent 在面对新任务时，不仅依赖于内置的 action 文档，还能参考之前总结出的 high-level workflows。

### Offline 与 Online 两种工作模式

AWM 灵活地支持了两种不同的运行场景：

1. **AWM offline**: 在有高质量标注数据的情况下，AWM 首先一次性消费所有 training examples $\mathcal{E}_{train}$，诱导出 workflows $\mathcal{W}_{offline} = I(\mathcal{E}_{train})$。在 test time，agent 使用固定的 augmented memory $M + \mathcal{W}_{offline}$ 来解决每一个 test instruction。
2. **AWM online**: 在缺乏训练数据的 supervision-free 场景下，AWM 以 streaming 的方式处理 test queries。当第 $t$ 个 test instruction $q_t$ 到来时，agent 先用当前的 memory $M^t$ 尝试解决它，生成 trajectory 并形成 experience $e_t$。随后，使用一个 evaluation model $L_{eval}$ 输出二值标签 $L_{eval}(e^t) \in \{0, 1\}$ 判断是否成功。如果成功 (值为 1)，则诱导出新的 workflows $\{w^t\} = I(e^t)$ 并更新 memory $M^{t+1} = M^t + \{w^t\}$，用于处理下一个 instruction。这种模式展现了极强的 on-the-fly 适应能力。

### 实验数据与架构直觉

作者在 **WebArena** 和 **Mind2Web** 两大基准测试上进行了广泛实验。

**WebArena 表现:**
在 Table 1 中，使用 GPT-4 作为 backbone，AWM 在没有使用任何人工编写的 workflows 的情况下，达到了 $35.5\%$ 的 Total Success Rate (SR)。这不仅远超 autonomous agent baseline `BrowserGym_ax-tree` ($15.0\%$，绝对提升 $20.5$ 个百分点，相对提升 $51.1\%$)，甚至击败了使用 14 个 human expert written workflows 的 SteP 方法 ($33.0\%$)。此外，AWM 解决任务所需的平均步数 (# Steps) 降到了 $5.9$ 步，比 baseline 减少了约 $2$ 步，比需要额外评估和细化步骤的 AutoEval ($46.7$ 步) 少了惊人的一截。这证明了 AWM 提取的 workflows 极大提高了 agent 的执行效率。

**跨任务泛化:**
Table 2 展示了在 WebArena 的 cross-template subset 上的表现。为了排除同模板任务的作弊嫌疑，作者从每个 template 中只随机抽取一个 example。在这种严苛的 cross-task 设定下，AWM 依然达到了 $33.2\%$ 的 SR，证明了 workflows 的泛化能力超越了具体的 task template。

**Mind2Web 表现:**
在 Table 3 的 Mind2Web cross-task 测试中，AWM (使用 GPT-4) 达到了 $45.1\%$ 的 Step SR 和 $50.6\%$ 的 Element Accuracy (Elem Acc)。相较于 Synapse 这种检索最相关 concrete examples 的方法，AWM 通过提供抽象的 sub-routines，减少了具体 example 带来的 element selection bias，使得 Elem Acc 提升了 $5.0$ 个百分点。这说明 abstract workflows 比 concrete trajectories 更具灵活性和普适性。

在 Table 4 展示的 cross-website 和 cross-domain 评估中，随着 train-test distribution gap 的扩大 (从 cross-task 到 cross-website，再到 cross-domain)，**AWM online** 的优势越来越明显。在 cross-domain 设定下，AWM online 达到了 $35.5\%$ 的 Step SR，而 AWM offline 仅为 $32.6\%$。因为 AWM online 完全不依赖可能存在分布偏差的 training data，它能够通过自我诱导动态适应全新的 test distribution。

### Action Space 扩展与 Action 形式探索

论文进一步探讨了 workflow 的多种利用方式。在 $\text{AWM}_{AS}$ 变体中，workflow 被包装成 high-level function 直接加入到 agent 的 action space 中，允许 agent 调用诸如 `login(username, password)` 这样的宏操作。虽然 Table 9 显示 $\text{AWM}_{AS}$ 的 Step SR 略有提升 ($46.4\%$)，但作者发现 agent 在仅 $18.5\%$ 的任务中愿意主动调用这些新 action。这揭示了当前 LLM agents 在适应新 action interface 时存在一定的惯性阻力。

Table 7 和 Table 8 探索了 workflow 的表示形式。将 code 形式的 action 转化为 pure text (如 "CLICK the submit button")，性能差异不大，说明无论是 code 还是 text 都能有效 augment memory。然而，如果在 workflow 步骤中加入过滤后的 HTML 表示来强化环境 grounding，性能反而下降 (Table 8，Step SR 从 $34.6\%$ 降到 $32.9\%$)。作者推测这源于 context length 的增加以及过滤 HTML 中高达 $47\%$ 的 missing correct elements 造成的噪声干扰。这给我们的直觉是：在 workflow memory 中，保持 high-level 的 NL abstraction 往往比强行的 environment grounding 更有效。

### 扩展联想与相关研究

AWM 的思想与 program synthesis 领域的 library learning 极为相似。像 DreamCoder 和 LILO 这样的系统也是通过压缩过往的解决路径来构建可重用的 abstraction library。AWM 可以被视为在 web agent 领域实现的一种 lightweight library learning。此外，这与人类的 procedural memory 机制在认知架构上产生了共鸣。人类在第一次订机票时需要逐步思考，但熟练后就简化为了一个单一的 "book flight" 程序块。

考虑到论文最后提到的 $\text{AWM}_{off+on}$ (Table 11) 效果介于两者之间，这暗示了 offline workflows 和 online workflows 之间可能存在冲突或者冗余。未来的系统可能需要一种类似于 attention mechanism 的 workflow retrieval 策略，或者引入更复杂的 meta-learning 机制来动态管理 workflow library 的兼容性。如果结合 Voyager 的自动 curriculum 生成，AWM 有潜力发展成一个完全自主、永不停止进化的 web agent。

### Reference Links
*   **Agent Workflow Memory (GitHub)**: https://github.com/zorazrw/agent-workflow-memory
*   **WebArena (ICLR 2024)**: https://openreview.net/forum?id=oKn9c6ytLx
*   **Mind2Web (NeurIPS 2023)**: https://openreview.net/forum?id=kiYqbO3wqw
*   **Voyager (TMLR)**: https://openreview.net/forum?id=ehfRiF0R3a
*   **DreamCoder**: https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0050
*   **Synapse (ICLR 2024)**: https://openreview.net/forum?id=Pc8AU1aF5e
