---
source_pdf: LLaVA-CoT Let Vision Language Models Reason Step-by-Step.pdf
paper_sha256: a090e8d2b079f7e061ca77d8bb7b7df72d6eac1781086cf55dafdcc5181d6173
processed_at: '2026-08-05T15:19:33-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LLaVA-CoT

Andrej，我上次讲得太技术了，这次咱坐下来用大白话聊聊这篇 paper 到底干了啥。

## 一句话说清楚

**这篇 paper 教一个看图说话的 AI 学会"先想清楚再回答"，而不是一上来就瞎说。**

## 问题出在哪

你给现在的 VLM 一张图和一个问题，它马上就开始输出答案。这就像你问一个学生一道数学题，他不等题目读完就开始写数字。

更糟糕的是，autoregressive model 有个要命的特性：**一旦开头说错了，后面全是在圆谎**。比如模型先说"这个人是哭"，然后整段 reasoning 都在解释"他为什么哭"——哪怕图里的人明明是在笑。token 一个一个生成，前面的 token 是后面的 context，错了就回不了头。

paper 里举了个真实例子：图里停车场上写着数字 31，模型一路推理得挺好，最后突然说答案是 17。为什么？因为它一开始 hallucination 了一下，后面就顺着错的走下去了。

## 他们的办法：分四步走

很简单，就是逼模型按固定顺序思考：

1. **SUMMARY**：这题问的是什么？我打算怎么解？
2. **CAPTION**：图里跟问题相关的部分有什么？
3. **REASONING**：一步步推理
4. **CONCLUSION**：最终答案

每一步用特殊 tag 标出来，比如 `<SUMMARY>...</SUMMARY>`。

这个设计看起来朴素，但有一个关键的心理学依据：**人在解题时也是先理解题目、再看条件、再推理、再下结论**。模型之前的问题是跳过前两步直接下结论，现在被强制按顺序来。

## 数据怎么来的

没有现成的训练数据长这样，所以他们用 GPT-4o 来生成。拿一些 VQA 数据集的题，让 GPT-4o 按四步格式回答，过滤掉格式不对或答案错误的，凑了 10 万条。

然后用这 10 万条去 fine-tune Llama-3.2-11B-Vision-Instruct。8 张 H100，3 个 epoch，完事。

## 关键发现：光靠 prompting 没用

这是 paper 里一个很容易被忽略但很重要的实验。

他们试过：不 fine-tune，直接用同样的四步 prompt 去 inference。

- GPT-4o：从 66.0 涨到 67.6，有效
- Llama-3.2-11B：56.9 到 56.9，**完全没变化**

这说明什么？**弱模型你给它 prompt 它也用不起来**。因为模型内部的 weights 没有形成支持 structured reasoning 的 circuit。必须通过 SFT 把这种 reasoning pattern 真正刻进 weights 里。

这跟你之前在 nanoGPT 讲座里说的道理一样：model 需要见过足够多的 example 才能学会某种 pattern，光靠 prompt 是激活不了的。

## 更有意思的部分：SWIRES

好，fine-tune 完了，模型会按四步走了。但还能不能更好？

这就是 SWIRES 登场的地方。名字很唬人，其实逻辑很简单。

### 先说三种 test-time 策略

**Best-of-N**：让模型生成 N 个完整答案，用 reward model 选最好的。
- 问题：如果一个答案中间某步错了，后面全废，但你得生成完才知道

**Stage-wise Beam Search**：每个 stage 生成几个候选，选最好的进入下一 stage。
- 问题：如果 caption stage 就错了，后面 reasoning 再好也救不回来

**SWIRES**：在 beam search 基础上加一个"回退"机制。如果当前 stage 的候选都不够好，就退回上一 stage 重新生成。
- 这就是核心创新

### 用大白话讲 SWIRES

想象你在考试，写作文。你先写提纲，再写每段内容。

Best-of-N 是：你写 5 篇完整作文，选最好的一篇。
Beam Search 是：你写 5 个提纲选 2 个，每个提纲写 5 段选 2 段……
SWIRES 是：你写到第二段觉得不对劲，**不是硬着头皮往下写，而是回头改提纲**。

就这么简单。但效果很明显。

### 一个数字说明问题

在 MMStar benchmark 上：
- Base model：49.8
- Fine-tuned LLaVA-CoT：57.6
- + SWIRES：62.5

SWIRES 单独贡献了 5 个点。而 SWIRES 没有改任何 weights，纯 inference time 的搜索策略。

### 为什么 SWIRES 比 beam search 强

Figure 5 那张 scaling curve 很能说明问题。

Best-of-N 和 Beam Search 在花了大约 1 万秒后都 plateau 了——再给更多 compute 也没用。但 SWIRES 在 1 万秒后还在继续涨，到 10 万秒量级还在提升。

直觉解释：beam search 只能往前走，一旦某步选错了，后面的 compute 全浪费在错误的分支上。SWIRES 可以回头，所以 compute 能花在"修正错误"上，而不是"在错误道路上越走越远"。

这就像 debug——你发现代码有 bug，是在 bug 处修补快，还是回到前面重新设计快？显然是后者。

## 效果到底有多好

11B 的模型，fine-tune + SWIRES 后：

- 超过了 Llama-3.2-90B-Vision（大 8 倍的模型）
- 超过了 Gemini-1.5-Pro
- 超过了 GPT-4o-mini
- 接近 Claude 3.5 Sonnet

但要注意：这些 benchmark 都是 reasoning-intensive 的。在纯 perception 任务（比如 OCR、物体识别）上提升不大，因为那些任务不需要 structured reasoning，需要的是 vision encoder 够强。

## 一个特别诚实的 ablation

Table 7 有个实验我觉得很关键。

他们把四个 stage 的顺序打乱来训练，比如让 model 先出结论再 reasoning。

结果：基本没提升。

这证明不是"多写点 token"就有用，**必须是正确的因果顺序**。perception 必须在 inference 前面，plan 必须在 execution 前面。这跟人的认知过程是一致的。

## 还有个小细节：threshold 怎么定

SWIRES 需要判断"当前 stage 的候选够不够好"。他们用的公式是：

threshold = mean + 0.25 × std

意思就是设在一个不算太高的 bar——只要 top candidate 比平均水平好一点就行。如果连这个都达不到，就回头重来。

这个 threshold 的设计比较 ad hoc，但 pragmatic。paper 也没说这是最优的，只是 work。

## 局限性：paper 自己承认的

Section J 写得很诚实：

1. 有时候 retracing 会让模型迷失——回头改了半天反而越改越乱
2. 复杂图像可能根本看不懂，再怎么 reasoning 也没用
3. 没有 verifiable reward，open-ended 问题很难判断对错

这些限制其实指向了未来的方向：**需要真正的 RL + 可验证的 reward**。比如数学题可以用代码验证对错，但"描述这张图"就很难自动打分。

## 我的直觉判断

这篇 paper 做的事情其实很朴素：**把人类解题的认知过程显式地编码进 model 的 generation 流程里**。

没有花哨的 architecture 创新，没有新的 loss function，就是：
1. 定义四个 stage
2. 用 tag 标出来
3. 蒸馏数据训练
4. Inference 时可以回退搜索

但正因为朴素，才 work。很多 ML 的突破不是来自更复杂的 model，而是来自更清晰的 thinking structure。

真正的下一步，我赌在 RL 上。现在 SWIRES 是 inference time 的 search，如果能把 search 的 signal 通过 RL 回传到 weights 里，让 model 在 training time 就学会"什么样的 reasoning path 会 lead to 正确答案"，那就是 OpenAI o1 / DeepSeek R1 的完整配方了。

LLaVA-CoT 把 VLM 版的 o1 路线往前推了一大步，但还差 RL 这一脚。

## 参考

- [Paper PDF](https://arxiv.org/abs/2411.10440)
- [GitHub 代码和数据](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- [Test-time scaling 理论基础 - Snell et al.](https://arxiv.org/abs/2408.03314)
- [DeepSeek R1 - RL 路线](https://arxiv.org/abs/2501.12948)
- [CoT 原始 paper](https://arxiv.org/abs/2201.11903)
- [VLMEvalKit 评测工具](https://github.com/open-compass/VLMEvalKit)

---

# LLaVA-CoT 深度解析

Andrej，这篇 paper 我读了之后有几个比较强的直觉想要分享。让我系统性地讲讲。

## 1. 核心问题的诊断

paper 诊断了当前 VLM 的两个核心 pathology：

**Pathology A**: 模型在没有 organize problem 之前就开始输出。这本质上是 System 1 thinking 的表现——直接 reflex 而不是 deliberate。

**Pathology B**: 模型过早下结论，然后试图 justify 它。这是 confirmation bias 在 autoregressive generation 中的体现。一旦某个错误的 conclusion token 被生成，由于 token-by-token 的 conditional generation 性质，后续 token 都会 condition 在这个错误之上，导致 error propagation。这跟你在 NanoGPT 里讲过的 autoregressive model 的"committed path"问题是一回事。

这两个问题在 reasoning-intensive 任务上特别致命。MMStar 上 base model Llama-3.2-11B 只有 49.8，但经过 structured reasoning + SWIRES 后达到 62.5，绝对提升 12.7 个点，相对提升 25%。

## 2. 四阶段 Reasoning 的设计哲学

### 2.1 Stage 分解的 rationale

```
SUMMARY   → meta-cognition (我面对什么问题？)
CAPTION   → perception grounding (图像里有什么相关内容？)
REASONING → logical inference (怎么推理？)
CONCLUSION → answer synthesis (最终答案)
```

这个分解对应了人类解题的认知过程。我联想到几个相关框架：

- **ReAct (Yao et al., 2022)**: Reasoning + Acting 交替，但 LLaVA-CoT 是 linear 的
- **Plan-and-Solve (Wang et al., 2023)**: 先 plan 再 execute，LLaVA-CoT 的 SUMMARY 类似 plan
- **Self-Refine (Madaan et al., 2023)**: 生成 + 反馈 + 修正，但 LLaVA-CoT 在 single forward pass 里完成
- **Neuro-symbolic methods** (如 Neural Module Networks): 把 reasoning 分解成 modules，LLaVA-CoT 用 tags 模拟这种 module boundary

关键 insight：**tags 在 token 序列中引入了 structural inductive bias**。Table 2 显示去掉 tags 后 average 从 62.4 降到 60.9，证明这种 structural scaffolding 起作用。我猜想机制是：tags 让 attention pattern 在训练时学会"在 SUMMARY tag 内做摘要，在 CAPTION tag 内做描述"，形成一种 soft modularization。

### 2.2 为什么顺序重要

Table 7 的 ablation 极其重要：

| Setting | Avg |
|---------|-----|
| LLaVA-CoT (correct order) | 63.1 |
| LLaVA-CoT (reorder stages) | 58.2 |
| LLaVA-CoT (multi-task, no CoT) | 57.7 |

shuffle 顺序后几乎没有提升，证明 stage 顺序对应了 reasoning 的 natural causal chain。这让我想到 causal reasoning 的 literature——reasoning 是有方向性的，perception 必须在 inference 之前，summary 必须在 detail 之前。

## 3. LLaVA-CoT-100k 数据集构造

### 3.1 数据来源

| Dataset | Type | Size |
|---------|------|------|
| ShareGPT4V | General VQA | 31.3k |
| ChartQA | General VQA | 17.2k |
| A-OKVQA | General VQA | 16.1k |
| AI2D | Science VQA | 11.4k |
| GeoQA+ | Science VQA | 11.4k |
| ScienceQA | Science VQA | 5.6k |
| DocVQA | General VQA | 4.0k |
| PISC | General VQA | 1.0k |
| CLEVR | General VQA | 0.5k |
| CLEVR-Math | Science VQA | 0.5k |

总计 ~99k。这是相对小的数据量（对比 LLaVA-1.5 用了 665k），但通过 GPT-4o 蒸馏得到 high-density reasoning supervision。

### 3.2 GPT-4o 蒸馏的 prompt

paper 附录 B 给出了完整的 prompt template。核心是要求 GPT-4o 严格按 SUMMARY/CAPTION/REASONING/CONCLUSION 四段输出，且 CONCLUSION 必须与标准答案精确匹配。

数据过滤逻辑：用另一个 prompt 让 GPT-4o 判断生成内容是否 "valid"（非拒答 + 与标准答案语义一致）。这种 two-stage verification 保证了数据质量。

### 3.3 蒸馏的微妙之处

Table 7 还做了一个有意思的实验：直接用 structured CoT prompt 让 GPT-4o 推理，GPT-4o 在 MMStar-R 上从 66.0 提升到 67.6，说明 structured CoT prompting 本身有效。但同样 prompt 给 Llama-3.2-11B，performance 没有变化（56.9 → 56.9）。

这个对比非常重要：**prompting 在 weak model 上不起作用，必须通过 SFT 内化这种 reasoning pattern**。我猜测原因是 weak model 的 attention/MLP weights 没有形成支持 structured reasoning 的 circuits，光靠 prompt 无法激活。

## 4. SWIRES 算法详解

这是 paper 最有意思的贡献。让我详细讲。

### 4.1 从 Best-of-N 到 Stage-wise Beam Search 到 SWIRES

**Best-of-N**:
- 生成 N 个完整 response
- 用 reward model 选最好的
- 问题：coarse-grained，中间错误无法纠正

**Stage-wise Beam Search**:
- 每个 stage 生成 M 个 candidates
- 选 top N 进入下一 stage
- 每个 selected candidate 生成 M/N 个新 candidates，保持总数 M
- 问题：local optima——如果 caption stage 错了，后续 reasoning 再好也救不回来

**SWIRES (Stage-WIse REtracing Search)**:
- 加入 retracing 机制
- 如果当前 stage 所有 candidates 的 reward 都低于 threshold，retrace 到上一 stage 重新生成
- 最多 retracing C 次

### 4.2 SWIRES 的 pseudocode 解析

```
Require: M, N, C
1: Generate initial summary
2: c ← 0  // backtracking counter
3: Cand ← [], Score ← []
4: repeat
5:   Generate M captions, evaluate, select top N
6:   Generate M reasonings for each of N captions
7:   for each reasoning: evaluate, append to Cand/Score
8:   if reasonings satisfy preset conditions: break
9:   c ← c + 1
10: until c ≥ C
11: Select top N reasonings
12: Generate one conclusion per reasoning
13: Evaluate all conclusions, return best
```

注意几个细节：
- Summary stage 只生成一次（empirical observation：summary 通常 high quality）
- Retracing 从 caption stage 开始
- Preset condition 是 "top candidate score > backtrack_cutoff"

### 4.3 Threshold 公式

$$\text{backtrack\_cutoff} = \text{reward\_mean} + Z \times \text{reward\_std}$$

变量含义：
- `reward_mean` = -0.77：reward model (InternLM-XComposer2.5-Reward) 在 reasoning stage 输出的均值
- `reward_std` = 2.08：标准差
- `Z` = 0.2533：标准正态分布系数

Z=0.2533 对应 $P(Z < 0.2533) \approx 0.6$，所以 $P(X > \text{cutoff}) \approx 0.4$。即阈值设在 60th percentile——要求 top candidate 至少优于 60% 的样本才认为合格。

参数选择：M=4, N=2, C=3。即每 stage 生成 4 个候选，保留 2 个，最多回溯 3 次。

### 4.4 为什么 SWIRES 比 Beam Search 好

Figure 5 的 scaling curve 非常 striking：
- Best-of-N 在 ~10^4 秒后 plateau
- Stage-wise Beam Search 也在 ~10^4 秒 plateau
- SWIRES 在 10^4 秒后继续上升，到 10^5 秒量级还在涨

我的直觉解释：**SWIRES 解决了 credit assignment problem**。在 sequential decision making 中，当 conclusion 错时，错误可能来自 reasoning、caption 或 summary 任一 stage。Beam search 只能 forward 优化，无法 backpropagate error 到前序 stage。SWIRES 的 retracing 机制本质上是一种 heuristic backpropagation——当后续 stage 表现差时，"梯度"传回前序 stage 重新生成。

这跟 MCTS 中的 backpropagation 步骤有相似之处，只不过 SWIRES 用的是 hard retracing 而不是 value 更新。

### 4.5 Retracing 的失败模式

paper 在 Section J 承认局限：有时 retracing 会让 model 迷失，或者为了 reach an answer 而 hallucinate。这让我想到：
- Self-refine 类方法都有这个风险：model 不知道自己错在哪里，反复修正可能越改越错
- 缺少 verifiable reward signal 是根本问题（数学题可以验证，open-ended VQA 很难）

## 5. 实验结果深度分析

### 5.1 Table 5: 与 SOTA 对比

| Model | Size | Avg |
|-------|------|-----|
| GPT-4o-0806 | - | 71.8 |
| GLM-4v-Plus | - | 72.5 |
| Claude3.5-Sonnet | - | 66.7 |
| Gemini-1.5-Pro | - | 63.6 |
| GPT-4o-mini | - | 63.8 |
| Llama-3.2-90B | 90B | 62.3 |
| Deepseek-VL2 | MoE 27B | 66.0 |
| Qwen2-VL-7B | 8B | 65.9 |
| InternVL2-8B | 8B | 64.0 |
| **LLaVA-CoT** | 11B | 63.1 |
| **LLaVA-CoT (w/ scaling)** | 11B | **66.3** |

几个观察：
- 11B 模型 + structured reasoning + SWIRES 超过 90B 的 Llama-3.2-Vision
- 接近 Claude 3.5 Sonnet（66.7 vs 66.3）
- 超过 Gemini-1.5-Pro 和 GPT-4o-mini

但要 caveat：这些是 reasoning-filtered benchmarks（MMStar-R, MMBench-R, MMVet-R），移除了纯 perception/OCR 任务。所以在"原生 reasoning 任务"上 LLaVA-CoT 优势明显，但在 perception-heavy 任务上可能不如大模型。

### 5.2 Table 3: Skill-level breakdown (MMStar)

| Model | CP | FP | IR | LR | Math | Sci&Tech | Avg |
|-------|----|----|----|----|------|----------|-----|
| Base | 66.0 | 46.4 | 57.6 | 50.8 | 45.2 | 32.8 | 49.8 |
| LLaVA-CoT | 68.8 | 46.8 | 63.2 | 58.0 | 64.0 | 44.8 | 57.6 |

关键 insight：
- Coarse Perception (CP) 几乎没变（66.0 → 68.8）
- Fine Perception (FP) 完全没变（46.4 → 46.8）
- Math 暴涨（45.2 → 64.0，+18.8）
- Sci&Tech 暴涨（32.8 → 44.8，+12.0）
- Logical Reasoning 大涨（50.8 → 58.0，+7.2）

这证明 structured reasoning 主要提升的是 reasoning 能力，对 perception 帮助不大。这符合直觉——perception 是 vision encoder 的事，reasoning 是 LLM 的事，LLaVA-CoT 主要作用于后者。

### 5.3 Table 4: Test-time scaling 效果

| Model | MMStar | MMBench | MMVet | MathVista | AI2D | Hallusion | Avg |
|-------|--------|---------|-------|-----------|------|-----------|-----|
| LLaVA-CoT | 57.6 | 75.0 | 60.3 | 54.8 | 78.7 | 47.8 | 62.4 |
| + scaling | 62.5 | 77.6 | 64.9 | 57.7 | 81.0 | 49.1 | 65.5 |

Scaling 带来 +3.1 平均提升。注意 MMStar 上提升最大（+4.9），因为 MMStar 是 reasoning-intensive benchmark，正好命中 SWIRES 的优势场景。

## 6. 训练细节

Table 6 的 hyperparameters：

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| Epochs | 3 |
| Batch size | 4 |
| Context length | 4096 |
| Weight decay | 0.0 |
| Gamma (LR scheduler) | 0.85 |
| Mixed precision | True |
| FSDP | enabled |

8×H100 上 full parameter fine-tuning。lr=1e-5 对 11B 模型 full fine-tuning 算保守，但 3 epochs + 100k 数据可能刚好。

## 7. 更深层的 intuition

### 7.1 为什么 structured reasoning 比 flat CoT 好？

我自己的 hypothesis：**flat CoT 是 unstructured trajectory，model 在生成时没有明确的"我现在在做什么"的 signal**。Structured tags 提供了 explicit role signal，让 attention 可以 specialize。

类比：Transformer 的 multi-head attention 本身就是一种 structural decomposition，让不同 head 学不同 relation。Tags 在 sequence level 提供了类似的 decomposition。

### 7.2 Test-time scaling 的本质

paper 引用了 Snell et al. 2024 的工作 "Scaling LLM test-time compute optimally can be more effective than scaling model parameters"。这是 OpenAI o1 路线的理论基础。

LLaVA-CoT 的贡献是：**在 VLM 上验证了 test-time scaling 的有效性，且提出了比 best-of-N 更 sample-efficient 的方法**。

### 7.3 与 RL 的关系（未来方向）

paper conclusion 提到未来用 RL。我联想：
- SWIRES 已经用了 reward model，相当于 inference-time 的 value-based search
- 可以用 GRPO 或 PPO 在 training time 优化每个 stage 的 policy
- 类似 DeepSeek R1 的思路，但分 stage 做 RL
- Process Reward Model (PRM) 比 Outcome Reward Model (ORM) 更适合这个 setting，因为 SWIRES 天然有 stage-level supervision signal

### 7.4 局限性的诚实评估

paper Section J 承认：retracing 可能迷失，复杂图像仍可能失败。这暗示：
- Reward model 不够强（InternLM-XComposer2.5-Reward 可能不如 GPT-4 级别的 reward）
- 没有真正的 self-correction 能力（retracing 只是重新 sample，不是修正）
- 缺少 verifiable reward signal（不像数学题可以 code verify）

### 7.5 蒸馏 vs 原生 reasoning

一个潜在 concern：LLaVA-CoT 的 reasoning 是从 GPT-4o 蒸馏来的。这限制了 model 的 reasoning ceiling——它不可能超过 GPT-4o 的 reasoning 能力。

真正的突破可能需要：
- Self-play / self-improvement（像 AlphaZero）
- RL with verifiable rewards（像 DeepSeek R1 在数学上做的）
- Test-time search + RL training（像 OpenAI o1 推测的做法）

LLaVA-CoT 是这条路上的一个重要 milestone，但不是终点。

## 8. 与相关工作的 positioning

| Method | Reasoning 结构 | Test-time scaling | Visual grounding |
|--------|---------------|-------------------|------------------|
| CoT prompting (Wei 2022) | Flat chain | No | No |
| Visual CoT (Shao 2024) | Flat chain + bbox | No | Yes |
| Prism (Qiao 2024) | 2-stage (perception+reasoning) | No | Partial |
| MSG (Cesista 2024) | Forced CoT | No | No |
| LLaVA-CoT | 4-stage structured | Yes (SWIRES) | Yes (caption stage) |

LLaVA-CoT 是第一个把 structured reasoning + test-time scaling 结合起来的 VLM。

## 9. 我的整体评价

**Strengths**:
1. Structured reasoning 的设计 elegant 且 effective
2. SWIRES 是 test-time scaling 在 VLM 上的 solid 尝试
3. Ablation 设计严密（reorder, multi-task, w/o tags）
4. 100k 数据就达到 SOTA-level reasoning，data efficiency 高
5. 开源 code/data/weights

**Weaknesses**:
1. 依赖 GPT-4o 蒸馏，reasoning ceiling 受限
2. Reward model 不够强
3. 没有真正的 self-correction（只是 re-sample）
4. 在 perception-heavy 任务上提升有限
5. Retracing 可能迷失，缺少 safeguard

**Future directions 我会建议**:
1. 用 RL (GRPO) 替代 SFT，让 model 自己 discover reasoning pattern
2. 用 stronger reward model 或 verifiable rewards
3. 加入 tool use（如 code execution for math）
4. 把 SWIRES 和 MCTS 结合，做更 sophisticated 的 search
5. 探索 stage 数量的影响（为什么是 4 个？3 个或 6 个如何？）

## 10. 参考资源

- **Paper**: [LLaVA-CoT arXiv](https://arxiv.org/abs/2411.10440)
- **Code**: [GitHub - PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- **Related - Test-time scaling**: [Snell et al., "Scaling LLM test-time compute"](https://arxiv.org/abs/2408.03314)
- **Related - DeepSeek R1**: [DeepSeek-R1 technical report](https://arxiv.org/abs/2501.12948)
- **Related - CoT prompting**: [Wei et al., "Chain-of-Thought Prompting"](https://arxiv.org/abs/2201.11903)
- **Related - Self-Refine**: [Madaan et al., "Self-Refine"](https://arxiv.org/abs/2303.17651)
- **Related - Visual CoT**: [Shao et al., "Visual CoT"](https://arxiv.org/abs/2402.14804)
- **Reward model**: [InternLM-XComposer2.5-Reward](https://arxiv.org/abs/2501.12368)
- **Evaluation toolkit**: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- **Base model**: [Llama 3.2 Vision](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- **Related - Plan-and-Solve**: [Wang et al., "Plan-and-Solve Prompting"](https://arxiv.org/abs/2305.04091)
- **Related - Tree of Thoughts**: [Yao et al., "Tree of Thoughts"](https://arxiv.org/abs/2305.10601)
- **Related - ReAct**: [Yao et al., "ReAct"](https://arxiv.org/abs/2210.03629)
- **Benchmark - MMStar**: [MMStar](https://arxiv.org/abs/2403.20330)
- **Benchmark - MathVista**: [MathVista](https://arxiv.org/abs/2310.02255)

---

Andrej，这篇 paper 让我最 excited 的点是 SWIRES。它本质上是在 autoregressive generation 上加了一个类似 MCTS 的 search structure，但用 stage boundaries 作为 natural branching points。这个 idea 可以推广到很多场景——任何有自然 stage 分解的 reasoning task 都可以用这个方法。

但真正的 breakthrough 还是要等 RL + verifiable rewards + test-time search 的三角组合。LLaVA-CoT 走出了重要的一步，但路还很长。
