---
source_pdf: DeepEyes Incentivizing Thinking with Images.pdf
paper_sha256: fe00cc121bc008ad282c3ff7d9e344c9144fc37f2f7fdcd8aeccd2f8d4b563b8
processed_at: '2026-08-03T18:37:33-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepEyes 人话版：让模型学会 "边想边看"

Andrej，我换个更口语的角度重新讲一遍，抓住核心 story。

---

## 一句话说清楚这篇 paper 在干嘛

现在的 vision-language model 有个尴尬的矛盾：它能看图，但思考的时候基本把眼睛闭上了。模型一次性把整张图 encode 成一堆 tokens，然后剩下的 reasoning 全在文字里进行。遇到图里某个小角落的细节，它就只能靠记忆里那张 "缩略图" 猜。

DeepEyes 干的事：用 RL 训练一个 7B 模型，让它在 reasoning 过程中**主动决定 "我要 zoom in 看看这块区域"**，把 crop 出来的小图塞回 context 继续推理。整个过程 end-to-end，不需要 cold-start SFT，不需要外部 detector，模型自己生成 bbox 坐标调用自己的 zoom-in 工具。

这基本就是把 OpenAI o3 的 "thinking with images" 能力在开源 7B 模型上复现了一遍。项目代码在 [github.com/Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes)。

---

## 为什么这件事重要

### 人类是 "active vision"，不是 camera

人看东西从来不是 passive snapshot。你看一个复杂场景（比如找一个穿红衣服的人），眼球会做 saccadic movement，先扫一遍整体，锁定几个候选区域，再 fixate 到具体人脸确认。这个过程是 sequential 的、goal-driven 的。Najemnik 和 Geisler 在 [Nature 2005](https://www.nature.com/articles/nature03390) 里证明人类的 eye movement strategy 接近 Bayesian optimal。

VLM 现在的做法相当于给你一秒钟看完整张图，然后没收你的眼睛，让你用文字描述推理。这显然不对。

### Text-only CoT 在高分辨率上崩了

这有一个非常直观的原因。Qwen2.5-VL 的 vision encoder 不管输入是 2K 还是 8K，最终都 encode 成固定数量的 tokens（大概 1280 个）。8K 图里的目标物体可能只占 100-200 pixels，downsample 之后基本就糊了，token 里没保留足够信息。再长的 text CoT 也救不回来——信息根本没进来。

Table 4 的 ablation 特别说明问题：用 text-only CoT 做 RL 训练，V* 分数从 71.2 涨到 88.5（很好），但 HR-Bench-8K 反而从 65.3 掉到 60.8（更差了）。因为 8K 图在训练数据里是 OOD，text-only reasoning 没有处理这种 resolution 的 mechanism。

DeepEyes 通过 crop 绕开了这个 bottleneck——你不需要 vision encoder 能处理 8K，你只需要模型能 grounding 到目标位置，crop 出来就是 256x256 的小图，encoder 处理这种分辨率毫无压力。本质上 DeepEyes 是 **adaptive token budget allocation**。

---

## 方法其实挺简洁的

### 模型架构：没改 backbone，只改了 "interaction protocol"

Base model 就是 Qwen2.5-VL-7B，没有任何架构修改。作者做的是在 system prompt 里告诉模型："你有一个 `image_zoom_in_tool`，输入是 bbox 坐标 `[x1, y1, x2, y2]`，输出是 cropped image"。

模型的 grounding 能力本来就在（refCOCO 89.1 的水平），RL 做的事情是 **incentivize 模型在 reasoning trajectory 中主动调用这个能力**。

### MDP 形式化

State $s_t$ 是到目前为止所有 text tokens 和 image observation tokens 的 interleaved 序列：

$$
s_t = \{(X_0, I_0), (X_1, I_1), \ldots, (X_t, I_t)\} = \{\mathbf{X}_{\leq t}; \mathbf{I}_{\leq t}\}
$$

- $X_i$：第 $i$ 步 VLM 自己生成的 text tokens
- $I_i$：第 $i$ 步的 image observation（原始 image 或 crop 出来的 image，经过 vision encoder 后的 tokens）
- $\mathbf{X}_{\leq t}$：累积 text
- $\mathbf{I}_{\leq t}$：累积 image observations

Action $a_t \sim \pi_\theta(a | s_t)$ 就是下一个 token。模型可以继续生成 text，也可以 emit `<tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [...]}}</tool_call>`，environment 执行 crop，把结果 append 到 trajectory。

关键：**observation tokens 不参与 loss 计算**。模型只对自己生成的 text 和 tool call tokens 负责，cropped image 是 environment 的回复，policy gradient 不流过它。这点跟标准 agentic RL 框架（比如 [veRL](https://github.com/volcengine/verl)）的处理一致。

### Reward：三部分，关键是 conditional

$$
R(\tau) = R_{\mathrm{acc}}(\tau) + R_{\mathrm{format}}(\tau) + \mathbb{I}_{R_{\mathrm{acc}}(\tau) > 0} \cdot R_{\mathrm{tool}}(\tau)
$$

- $R_{\mathrm{acc}}$：答案对不对
- $R_{\mathrm{format}}$：格式对不对
- $R_{\mathrm{tool}}$：用没用工具
- $\mathbb{I}_{R_{\mathrm{acc}}(\tau) > 0}$：indicator，**只有答案对的时候 tool reward 才生效**

这个 conditional design 是 paper 的关键 insight。看 Figure 5 旁边的 ablation：

| Reward 设置 | V* | HR-4k | HR-8k |
|---|---|---|---|
| 无 tool reward | 87.4 | 53.4 | 55.4 |
| Unconditional tool reward | 87.4 | 72.1 | 71.8 |
| **Conditional tool reward** | **90.1** | **75.1** | **72.6** |

无 tool reward：模型很快就学会 "工具没用，不用了"，HR-4k 只有 53.4。
Unconditional：模型保持 tool usage 但不优化策略，停在 72 左右。
Conditional：模型必须 "用工具做对题" 才拿 bonus，所以它得学会 **有意义的 tool use**。

人话翻译：你不能奖励 "孩子翻书" 这个动作，你得奖励 "孩子翻书找到了答案"。否则孩子学会的是翻书表演，不是查阅资料。

### 训练算法：GRPO

用的 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)。对每个 prompt 采样 16 个 rollouts，用 group 内 reward 的 mean/std 归一化作为 advantage：

$$
A_i = \frac{R(o_i) - \mathrm{mean}(R)}{\mathrm{std}(R)}
$$

然后标准 PPO-style policy gradient。KL coefficient 设为 0.0——和 DeepSeek-R1 一样，完全 trust policy gradient，不往 reference policy 拉。这通常意味着更多的 exploration，但也更不稳。80 个 iteration，每 batch 256 prompts，max response 20480 tokens，最多 6 次 tool call。

---

## Data Selection 是被低估的关键

早期实验失败了——模型不愿意用工具，即使用了也 crop 得很烂。作者设计了 4 步筛选 47k 训练样本：

1. **难度筛选**：用 Qwen2.5-VL-7B 生成 8 个回答，按 accuracy 估难度。全错（太难）和全对（太简单）的都丢掉，留 "sometimes right sometimes wrong" 的 sweet spot。

2. **格式统一**：全转 open-ended，方便 rule-based verification。

3. **可验证性**：剔除答案错误、图像不可读的。

4. **Tool 信息增益筛选**（最重要）：让模型单轮回答，错的样本再给 ground-truth bbox 的 crop，如果给 crop 就能答对，保留这个样本。相当于只保留 "工具确实能救" 的样本。

数据构成：
- V*（visual search，自然图像细粒度感知）：22k，47%
- ArxivQA（chart 数据，增加视觉元素多样性）：14k，30%
- ThinkLite-VL（reasoning 数据，防止 catastrophic forgetting）：11k，23%

Table 5 的 ablation 说明这个组合的必要性：
- 只用 fine-grained data：reasoning 能力掉（catastrophic forgetting）
- 加 reasoning data：保住 math 能力
- 加 chart data：HR-8k 从 70.5 跳到 74.6，因为 chart 引入了 "多元素关系推理" 的多样性

人话：你训练模型用放大镜，得给它 "不用放大镜看不清，用了能看清" 的样本。给它本来就看得清的样本，它学不到工具的价值。给它就算放大也看不清的样本，它学不到工具的成就感。

---

## 训练动态：三阶段演化（这是最 fascinating 的部分）

Figure 3 在 fine-grained data 上的 training dynamics 显示了非常清晰的三个阶段。我用 IoU 量化 crop 质量。

### Stage 1: Initial Tool Exploration (Steps 0-20)

模型刚开始响应 system prompt 调用工具，但没有 coherent policy。
- Tool call count 上升
- Response length 上升（写很多 verbose 的 image description）
- Grounding IoU 很低（crop 区域不准）
- Step 8-20 之间 response length 突然下降——模型开始 trim 掉冗余描述

**人话**：婴儿在挥动手臂，不知道手能抓东西。系统提示说 "你可以用工具"，它就乱试。

### Stage 2: High-Frequency Tool Usage (Steps 20-45)

模型发现 "调用工具 = reward"，开始 aggressive 使用。
- Tool call count 大幅上升
- Accuracy 和 IoU 都显著上升
- Response length 很长
- "Broad sweep" strategy——不靠 internal reasoning，靠 externalize 到 environment

**人话**： toddler 发现 "敲桌子有声音"，就一直敲。已经在建立 "tool → correct answer" 的因果关联，但还没学会节制。

### Stage 3: Efficient Tool Exploitation (Steps 45-80)

模型学会 selective use。
- Tool call count 下降
- Response length 下降
- Accuracy 和 IoU 保持高
- Implicit planning emerge——模型先 internal 缩小范围，再 selective zoom 确认

**人话**：专家查资料，只在需要时查，查的时候精准定位。模型内化了 "什么时候需要工具" 的判断。

### 为什么这个 dynamic 很有意思

这跟 DeepSeek-R1 报告的 "aha moment" 是同一类现象。R1 训练中模型突然学会 reflection（"等等，让我重新想想"），DeepEyes 训练中模型从盲目 tool use 过渡到 selective tool use。

我怀疑这是 RL + sparse reward 的 universal dynamic：
- 初期高 entropy 探索
- 中期 reward-driven aggressive exploitation
- 后期 entropy 衰减 + policy 精化

AlphaGo 的训练历史也有类似 pattern：早期 rollout policy 乱下，中期 RL policy aggressive 改进，后期 MCTS + value network selective 搜索。

---

## Emerge 的 4 种 Thinking Patterns

论文识别出 4 种自然涌现的 reasoning 模式，都跟人类认知对应。

### Visual Search（Figure 7）

例子：判断一个 wetsuit 是不是湿的。模型先看整体，发现光看不够，第一次 zoom 不准，第二次主动 focus 到 wetsuit 区域找水滴痕迹，再结合环境（湿沙反光）综合判断。

这跟人类 visual search 的 scan-fixation 模式一致。

### Visual Comparison（Figure 8）

例子：判断 4 个 chart section 哪个数据变异性最小。模型依次 zoom in (a)(b)(c)(d)，对比波动幅度，最后选 (c)。

人类做对比任务也是这样：逐个 fixate，建立 mental image，再比较。

### Visual Confirmation（Figure 9）

例子：判断窗户形状。模型初始不确定（"可能是 arch 形"），多次 zoom 后从不确定到确定（"是 arch 形"）。

人类在 uncertain 时也会重新看一眼确认。

### Hallucination Mitigation（Figure 10）

例子：模型初始混淆 pants 和 blazer 的颜色（典型 VLM hallucination），zoom in 后纠正。

这相当于模型的 "self-correction" 机制——通过主动 perception 校正 hallucination。

**这些 patterns 不是 hard-coded 的，是 RL 训练 emerge 出来的**。作者没告诉模型 "你应该 search / compare / confirm"，模型自己从 outcome reward 里学到了这些策略。

---

## 结果有多强

### High-Resolution Benchmarks（Table 1）

DeepEyes 7B 在 V* 上 90.1，HR-8K 上 72.6。

对比：
- Qwen2.5-VL 7B baseline：71.2 / 65.3（DeepEyes 提升 +18.9 / +7.3）
- Qwen2.5-VL 32B：87.9 / 70.4（DeepEyes 7B 超过 32B！）
- GPT-4o：66.0 / 55.5（DeepEyes 高 24 个点）
- ZoomEye（workflow 方法，tree-based exploration）：90.6 / 69.3（DeepEyes 在 HR-8K 上超过它）

7B 模型超过 32B 这件事本身就很说明问题——scaling 被 reasoning 范式超越了。

### Grounding & Hallucination（Table 2）

POPE（hallucination）从 85.9 提升到 87.7，random split 从 87.2 提升到 91.8（+4.6）。

Grounding（refCOCO 系列）小幅提升 0.6-1.0。

人话：模型学会 "answer before verify"，所以 hallucination 减少。

### Reasoning（Table 3）

WeMath 提升 +4.3 最多。MathVerse 略降 -1.9（可能 data distribution 问题）。整体上 iMCoT 对 reasoning 也有正向 transfer。

---

## 失败案例的启示（Appendix D.2）

### Grounding Drift（Figure 11）

模型第一次 zoom 假设 awning 是绿色，第二次 zoom 时 bbox 漂移到错误区域（蓝色），导致模型反转判断，答案错误。

人话：模型的 spatial memory 有问题。前一次 crop 的信息没被 well-integrated 到下一次 grounding 的决策中。这暗示 iMCoT 的 context 利用还有改进空间。

### Reasoning Limitation（Figure 12）

模型准确 zoom 到了 figure (b)，但看不懂曲线趋势，答案还是错。

人话：perception 和 reasoning 还是 partially decoupled。zoom 提供了 information，但 model 不会用。这是 foundation model capability 的 bottleneck。

作者在 Section E 也承认：用 7B base model 能力有限，用 32B 或 72B 可能缓解。

---

## 我的几点直觉

### 1. 这本质上是 test-time compute scaling 的新轴

之前 test-time scaling 只有 "think longer in text" 一个维度。DeepEyes 加了 "look more in image" 这个维度。两个轴可以叠加——模型可以同时 think longer 和 look more。这开启了 visual test-time scaling law 的可能性，跟 [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) 是同一类思想在 multimodal 上的延伸。

### 2. Active perception vs passive perception 是范式转变

传统 VLM 是 passive perception：一次性 encode，然后 reasoning。DeepEyes 是 active perception：模型决定看哪里、看几次。这跟 cognitive science 里的 [active vision](https://www.cambridge.org/core/books/active-vision/EC4A0F9A2C1C5C0F8C0A0A0A0A0) 理论一致——人类视觉本质是 active 的，眼球运动是 cognition 的一部分。

### 3. 模型 "use itself as tool" 这件事很 self-referential

模型的 grounding 能力被封装成 tool 调用，然后 RL 让模型学会 use itself。这跟 [Voyager](https://arxiv.org/abs/2305.16291) 让 agent 学会 use 自己写的 skill library 有点像，但更 recursive。

### 4. Conditional reward 的设计哲学

这跟 RLHF 里 reward hacking 的讨论一脉相承。你不能奖励 surface behavior（"用了工具"），你得奖励 outcome（"用工具解决了问题"）。R1 的发现是 "sparse outcome reward 就够了"，DeepEyes 的发现在此基础上加了 "outcome-conditioned tool bonus"——一个 middle ground：完全 sparse 的话模型学不会 tool use，完全 dense 的话模型 reward hack。

### 5. 为什么不需要 cold-start SFT

这是 paper 最 surprising 的 claim。我的猜测：
- Qwen2.5-VL 本身已有 grounding 能力
- Tool calling format 通过 system prompt 就能 elicit
- RL 只需要 incentivize 已有 capability，不需要 teach 新 capability

这跟 R1-Zero 的发现一致：base model 已经有 reasoning capability，RL 只是 unlock。但 R1-Zero 用 DeepSeek-V3 这种大 base，DeepEyes 在 7B 上 work 更 impressive。

### 6. 跟 System 1 / System 2 的联系

[Visual Agents as Fast and Slow Thinkers](https://arxiv.org/abs/2408.08862) 把 visual reasoning 分成 fast（perception）和 slow（deliberation）。DeepEyes 的 iMCoT 在某种意义上 unify 两者：text CoT 是 slow thinking，zoom-in tool call 是 fast perception 的主动调用，两者 interleave 就是 Kahneman 的 System 1 + System 2 协同。

### 7. 数据效率令人印象深刻

47k samples + 80 iterations RL training（H100 GPUs）就 unlock 这个能力。对比 SFT-based 方法动辄百万级数据，RL 在 reasoning task 上确实更 sample-efficient。这跟 R1 的发现一致。

### 8. 未来方向：Tool Diversity

作者 Section G 提到未来要加更多 tool：search、draw auxiliary lines。这让我想到：
- **Draw auxiliary lines** 对应 geometry reasoning——人类在图上画辅助线解题
- **Search** 对应 external knowledge retrieval
- 可能还可以加 **generate sub-image**（让 model 画草图）——对应 sketch-based reasoning

每个新 tool 都会引入新的 reasoning pattern space，跟 [Toolformer](https://arxiv.org/abs/2302.04761) 的 vision 版本演化方向一致。

---

## 总结

DeepEyes 的核心 contribution 不是什么花哨的架构，是一个简洁有力的 insight：**VLM 已经有 grounding 能力，把它 externalize 成 tool，用 RL 让模型学会在 reasoning 中主动调用，就能 unlock "thinking with images"**。

关键 engineering tricks：
1. Conditional tool reward（防止 reward hacking）
2. Tool-use-oriented data selection（"工具能救" 的样本）
3. Observation tokens 不算 loss（agentic RL 标准做法）
4. KL = 0（充分 exploration）

最 fascinating 的发现：
- 训练三阶段 dynamic（exploration → aggressive use → efficient exploitation）
- 4 种 thinking patterns 自然 emerge（search / compare / confirm / mitigate hallucination）
- 7B 超过 32B 和 workflow 方法

这基本就是 OpenAI o3 "thinking with images" 的开源复现，而且用了一种非常 clean 的 RL 范式实现。代码在 [github.com/Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes)，值得 clone下来玩玩。

Andrej，你觉得这个 RL-incentivized tool use 的范式能 scale 到更复杂的 agentic task 吗？比如让模型学会在 reasoning 中主动调用 web search、code execution、或者自己的 sub-agent？我直觉上觉得这个 paradigm 比 prompt-engineered agent 更有潜力，因为 reward signal 直接 optimize 整个 trajectory，避免了 modular pipeline 的 compounding error。

---

# DeepEyes: 通过 RL 激励 "Thinking with Images" 能力

Andrej，这篇 paper 非常有意思，我觉得它抓住了 multimodal reasoning 的一个关键 missing piece。让我从直觉出发，逐层拆解。

---

## 1. 核心直觉：为什么需要 "Thinking with Images"

当前的 VLM（如 Qwen2.5-VL、LLaVA-OneVision）虽然在 multimodal understanding 上很强，但 reasoning process 基本被锁死在 text modality 里。模型看到一张图，把 image tokens encode 进来，然后纯粹用 text CoT 推理。这跟人类认知差距很大——人类在思考时会有 **saccadic eye movements**，通过 sequential visual fixations 主动从图像中提取信息 [Najemnik & Geisler, Nature 2005](https://www.nature.com/articles/nature03390)。

OpenAI o3 ([thinking with images blog](https://openai.com/index/thinking-with-images/)) 据称已经实现了这个能力，但机制未公开。DeepEyes 的贡献就是用 **end-to-end RL** 把这个能力从 7B 模型里 "incentivize" 出来，而且 **不需要 cold-start SFT**，这很 surprising。

关键 insight：模型本身（Qwen2.5-VL）已经有 grounding 能力（能输出 bounding box 坐标），DeepEyes 把这个能力封装成一个 **image_zoom_in_tool**，让模型在 reasoning 过程中自主决定何时调用、调用几次、zoom 哪个区域。crop 出来的 image 被拼回 trajectory，形成 **interleaved Multimodal Chain-of-Thought (iMCoT)**。

---

## 2. 方法细节

### 2.1 Architecture Overview

参考 [项目主页](https://github.com/Visual-Agent/DeepEyes) 和 Figure 2，整体流程：

```
Input: (Question, Image I₀)
  ↓
[Text CoT step 1] → action: 要么继续 text reasoning, 要么 emit tool_call
  ↓ (if tool_call)
[image_zoom_in_tool(bbox)] → cropped image I_t1
  ↓
[Text CoT step 2] with (I₀, I_t1) in context
  ↓
... (repeat up to 6 times)
  ↓
<answer>...</answer>
```

这个设计有几个重要特点：
- **Native tool calling**：zoom-in 工具的 "执行器" 就是模型自己（它生成 bbox 坐标），不依赖外部 specialized model（比如独立的 detector）
- **Observation tokens 不参与 loss**：crop 出来的 image tokens 作为 environment observation，在 GRPO 的 token-wise loss mask 中被 mask 掉
- **Trajectory-level optimization**：整个 rollout（text + tool calls + observations）一起通过 policy gradient 优化

### 2.2 MDP Formulation

这是我觉得最 elegant 的部分。传统 text-only CoT 的 MDP：
- State $s_t$ = (input prompt + 所有已生成 tokens)
- Action $a_t$ = next token

Agentic RL 扩展引入 **observation tokens**（来自外部 function call，而非模型自身生成）。iMCoT 的 state 定义见 Eq. (1)：

$$
s_t = \{(X_0, I_0), (X_1, I_1), \ldots, (X_t, I_t)\} = \{\mathbf{X}_{\leq t}; \mathbf{I}_{\leq t}\}
$$

变量解释：
- $s_t$：step $t$ 的 state
- $X_i$：第 $i$ 步的 text token 序列（由 VLM 生成）
- $I_i$：第 $i$ 步的 image observation tokens（来自 tool crop 或原始 image）
- $\mathbf{X}_{\leq t} = \{X_1, \ldots, X_t\}$：累积 text tokens
- $\mathbf{I}_{\leq t} = \{\bar{I}_1, \ldots, \bar{I}_t\}$：累积 image observation tokens（$\bar{I}$ 表示经过 vision encoder 处理后的 token 表示）

Action 通过 policy $\pi_\theta(a \mid s_t)$ 采样。Rollout 持续直到生成 answer 或达到 max tool calls（6 次）。

**Intuition**：这个 formulation 本质上把 "看图" 变成了一个 actionable decision，模型在每一步都能选择 "继续纯文本推理" vs "调用视觉工具获取新信息"。这和 DAgger [Ross et al. 2011](https://arxiv.org/abs/1011.0686) 里的 interactive imitation 思想有点像，但这里完全用 outcome reward 驱动，没有 step-level supervision。

### 2.3 Reward Design

Eq. (2) 是论文的核心 reward：

$$
R(\tau) = R_{\mathrm{acc}}(\tau) + R_{\mathrm{format}}(\tau) + \mathbb{I}_{R_{\mathrm{acc}}(\tau) > 0} \cdot R_{\mathrm{tool}}(\tau)
$$

变量解释：
- $\tau$：一条完整的 reasoning trajectory
- $R_{\mathrm{acc}}(\tau)$：accuracy reward，最终答案是否正确
- $R_{\mathrm{format}}(\tau)$：formatting reward，惩罚格式错误的输出
- $R_{\mathrm{tool}}(\tau)$：tool usage bonus
- $\mathbb{I}_{R_{\mathrm{acc}}(\tau) > 0}$：indicator function，仅当 $R_{\mathrm{acc}}(\tau) > 0$（答案正确）时取 1

**关键设计**：tool reward 是 **conditional** 的——只有答案正确且至少调用了一次工具时才给 bonus。这避免了模型学会 "为了调用而调用" 的 degenerate behavior。

从 Table（Figure 5 旁边）能看到这个设计的重要性：

| Method | V* | HR-4k | HR-8k |
|---|---|---|---|
| w/o Tool Reward | 87.4 | 53.4 | 55.4 |
| Unconditional Reward | 87.4 | 72.1 | 71.8 |
| Conditional Reward | 90.1 | 75.1 | 72.6 |

没有 tool reward，模型很快就不再用工具；unconditional reward 让模型保持基础 tool usage 但不优化；只有 conditional reward 让模型持续探索更 sophisticated 的 reasoning strategies。

**Intuition**：这跟 DeepSeek-R1 [Guo et al. 2025](https://arxiv.org/abs/2501.12948) 的 insight 一致——sparse outcome reward 能 emergent 出复杂推理行为，但需要 reward shaping 避免局部最优。这里 "条件性" 就是 shaping 的关键。

### 2.4 Optimization: GRPO

采用 Group Relative Policy Optimization [Shao et al. 2024, DeepSeekMath](https://arxiv.org/abs/2402.03300)，GRPO 的核心是 group-relative advantage estimation：

对每个 prompt $q$，采样 $G$ 个 rollouts $\{o_1, \ldots, o_G\}$，advantage 用 group 内 normalized reward：

$$
A_i = \frac{R(o_i) - \mathrm{mean}(R)}{\mathrm{std}(R)}
$$

然后优化标准 policy gradient loss（带 KL regularization，这里 KL coefficient 设为 0.0，和 R1 一样）。

**Multi-turn 关键**：observation tokens（crop 出来的 image tokens）在 loss 中被 mask 掉。这意味着模型只对自己生成的 text tokens 和 tool call tokens 负责，environment 的 "回复"（cropped image）不计入 policy gradient。这点和 agentic RL 框架如 [veRL](https://github.com/volcengine/verl) 的处理一致。

### 2.5 Data Selection Mechanism（4 步）

这是论文另一个 underappreciated 的贡献。早期的 naive 训练失败了——模型不愿意用工具，即使用了也 crop 得很差。作者提出 4 步筛选：

1. **Managing Difficulties**：用 Qwen2.5-VL-7B 对每个 question 生成 8 个 responses，按 accuracy 估计难度。accuracy = 0（太难）或 = 1（太简单）的样本被剔除。保留 "sweet spot" 的样本。
   
2. **Structuring Question Formats**：转成 open-ended format（适合 RL 验证），剔除无法可靠转换的。

3. **Ensuring Verifiability**：剔除答案错误、图像不可读的样本。

4. **Facilitating Tool Integration**（最关键）：筛选 "单轮做错但用 ground-truth crop 能做对" 的样本。这些样本是 tool usage 信息增益最大的。

具体地，对每个样本：
- 先让模型单轮回答（无 tool）→ 记录错误样本
- 对错误样本，提供 ground-truth bbox 的 cropped image → 如果能答对，保留

**Intuition**：这相当于在 data 层面 enforce "tool 是有用的" 这个 prior。否则模型很难从 sparse reward 里学会 "为什么要用工具"。这跟 RLHF 里 "选好的 demonstration" 的思想类似，但这里是选 "tool 能拯救" 的样本。

最终数据构成（47k samples）：
- V* (Visual Search): 22k samples, 47%
- ArxivQA (Chart): 14k samples, 30%
- ThinkLite-VL (Reason): 11k samples, 23%

---

## 3. 实验结果深度解读

### 3.1 High-Resolution Benchmarks（Table 1）

| Model | Param | V* Overall | HR-4K Overall | HR-8K Overall |
|---|---|---|---|---|
| GPT-4o | - | 66.0 | 59.0 | 55.5 |
| o3 | - | 95.7 | - | - |
| SEAL (workflow) | 7B | 75.4 | - | - |
| ZoomEye (workflow) | 7B | 90.6 | 69.6 | 69.3 |
| LLaVA-OneVision | 7B | 75.4 | 63.0 | 59.8 |
| Qwen2.5-VL | 7B | 71.2 | 68.8 | 65.3 |
| Qwen2.5-VL | 32B | 87.9 | 73.9 | 70.4 |
| **DeepEyes** | **7B** | **90.1** | **75.1** | **72.6** |
| Δ vs Qwen2.5-VL 7B | | **+18.9** | **+6.3** | **+7.3** |

几个关键观察：
- DeepEyes 7B 超过 Qwen2.5-VL 32B（在 V* 上 90.1 vs 87.9）——scaling 被推理范式超越
- 超过 workflow 方法 ZoomEye，而 ZoomEye 用了 tree-based exploration，更复杂
- 比 GPT-4o 高 24 个点（在 V* 上）

**Intuition**：高分辨率 benchmark 的核心挑战是 "object 只占 100-200 pixels in 8K image"。任何把整图 encode 的方法都会因为 downsample 丢失细节。DeepEyes 通过主动 crop 把 relevant region 放大，本质上是 **adaptive resolution allocation**。

### 3.2 Grounding & Hallucination（Table 2）

| Model | refCOCO | refCOCO+ | refCOCOg | ReasonSeg | POPE Overall |
|---|---|---|---|---|---|
| Qwen2.5-VL* | 89.1 | 82.6 | 86.1 | 68.3 | 85.9 |
| DeepEyes | 89.8 | 83.6 | 86.7 | 68.6 | 87.7 |

POPE（hallucination）提升 +1.8，random split 提升 +4.6。Grounding 小幅提升。

**Intuition**：iMCoT 让模型 "verify before answer"，在 POPE 这种 yes/no 任务上，模型可以 zoom in 确认物体存在/不存在，减少 hallucination。这跟 visual confirmation pattern（Section 4.3）一致。

### 3.3 Multimodal Reasoning（Table 3）

| Model | MathVista | MathVerse | MathVision | WeMath | DynaMath | LogicVista |
|---|---|---|---|---|---|---|
| Qwen2.5-VL | 68.2 | 49.2 | 25.1 | 35.2 | - | 44.1 |
| DeepEyes | 70.1 | 47.3 | 26.6 | 38.9 | 55.0 | 47.7 |

WeMath 提升 +4.3 最多，MathVerse 反而略降 -1.9。这说明 reasoning data 的 inclusion 重要（避免 catastrophic forgetting）。

### 3.4 Ablation: iMCoT vs Text-only CoT（Table 4）

| Model | V* | HR-4K | HR-8K |
|---|---|---|---|
| Qwen2.5-VL baseline | 71.2 | 68.8 | 65.3 |
| RL w. Text-only CoT | 88.5 | 75.4 | 60.8 |
| DeepEyes (iMCoT) | 90.1 | 75.1 | 72.6 |

关键发现：text-only CoT 在 HR-8K 上反而下降到 60.8（vs baseline 65.3）！因为 8K 图像在训练数据中 OOD，text-only CoT 没法处理。iMCoT 通过 crop 绕过 resolution limit，达到 72.6。

**Intuition**：这验证了 iMCoT 的核心价值——它不只是 "更好的 CoT"，而是 **bypass 了 vision encoder 的 resolution bottleneck**。Vision encoder 把图像 encode 成固定数量 tokens（比如 256 或 1280），8K 图像信息必然损失。Crop 等于在推理时动态分配 token budget 给 relevant region。

---

## 4. 训练动态：三阶段演化（我觉得最 fascinating 的部分）

参考 Figure 3。在 fine-grained data 上的 training dynamics 显示出清晰的三个阶段：

### Stage 1: Initial Tool Exploration (Steps 0-20)

- Tool call count ↑
- Response length ↑（verbose image descriptions）
- Grounding IoU 低（crop 不准）
- Step 8-20 之间 response length 突然下降——模型开始 trim 冗余描述

**Interpretation**：模型在 "试" 工具，但没有 coherent policy。像 baby 在挥动手臂，不知道手能抓东西。

### Stage 2: High-Frequency Tool Usage (Steps 20-45)

- Tool call count 大幅 ↑
- Accuracy 和 IoU 都显著 ↑
- Response length 长
- "Broad sweep" strategy——不靠 internal reasoning，靠 externalize 到 environment

**Interpretation**：模型发现 "调用工具 = reward"，开始滥用。像 toddler 发现 "敲桌子有声音"，就一直敲。但已经在建立 tool → correct answer 的关联。

### Stage 3: Efficient Tool Exploitation (Steps 45-80)

- Tool call count ↓
- Response length ↓
- Accuracy 和 IoU 保持高
- 内化了 "何时该用工具" 的判断

**Interpretation**：模型学会 selective tool use。像 expert 在需要时才查资料，而非通读全书。**Implicit planning** emerge——模型先 internal 缩小范围，再 selective zoom in 确认。

**这个 3-stage dynamic 让我想到 LLM RL 训练中的 "phase transition"**：
- DeepSeek-R1 paper 也报告了类似的 "aha moment"——模型突然学会 reflection
- 这里是 "tool use aha moment"——从盲目调用到有意义调用
- 都体现了 RL + sparse reward 下 emergent behavior 的非线性

我觉得这背后可能是 **exploration-exploitation tradeoff** 的自然演化：初期高 entropy 探索，中期 reward-driven exploitation，后期 entropy 衰减但 policy 精化。

---

## 5. Thinking Patterns（Section 4.3）

论文识别出 4 种 emerge 的 reasoning patterns（Figure 7-10）：

### 5.1 Visual Search
面对复杂问题，模型 zoom in 不同区域扫描，收集 visual clues，再综合推理。类似人类的 visual search [Najemnik & Geisler 2005](https://www.nature.com/articles/nature03390)。

### 5.2 Visual Comparison
对多个对象，模型逐个 zoom in，close examination 后比较。Figure 8 例子：判断 4 个 chart section 哪个数据变异性最小，模型依次 zoom in (a)(b)(c)(d) 对比。

### 5.3 Visual Confirmation
模型初始不确定，通过 zoom in 收集证据逐步建立 confidence。Figure 9 例子：判断窗户形状，多次 zoom 后从不确定到确定。

### 5.4 Hallucination Mitigation
模型初始可能 hallucinate（比如混淆颜色），通过 zoom in 校正。Figure 10 例子：混淆 pants 和 blazer 颜色，zoom 后纠正。

**Intuition**：这些 patterns 都跟 human visual cognition 对应。人类 eye tracking 研究里也有类似的 fixation patterns：scan（search）、compare、verify、re-check。这暗示 RL 训练和人类认知可能有相似的 inductive biases——sparse reward + grounded perception 自然收敛到这些策略。

---

## 6. 与其他工作的对比

### 6.1 vs Workflow-based Methods

| 方法 | 类型 | 需要 SFT | 需要 external model | 端到端优化 |
|---|---|---|---|---|
| SEAL [Wu & Xie, V*](https://arxiv.org/abs/2412.04467) | Workflow | 是 | 是 | 否 |
| DyFo [Li et al. 2025](https://arxiv.org/abs/2504.14920) | Workflow | 否 | 是 | 否 |
| ZoomEye [Shen et al. 2024](https://arxiv.org/abs/2411.16044) | Workflow | 是 | 是 | 否 |
| **DeepEyes** | **End-to-end RL** | **否** | **否** | **是** |

Workflow 方法（SEAL、DyFo、ZoomEye）依赖 hand-designed pipeline，每个组件单独训练，suboptimal（参考 [Ross et al. DAgger](https://arxiv.org/abs/1011.0686) 的 compounding error 分析）。DeepEyes 通过 end-to-end RL 避免 this。

### 6.2 vs Visual-CoT 类方法

- Visual-CoT [Shao et al. NeurIPS 2024](https://arxiv.org/abs/2405.09118)：依赖大量 SFT data
- Perception Tokens [Bigverdi et al. 2024](https://arxiv.org/abs/2412.03548)：在 latent space 加 perception tokens
- VoT [Li et al. 2025](https://arxiv.org/abs/2501.07542)：visualization-of-thought

DeepEyes 不需要 SFT，直接从 outcome reward 学。

### 6.3 vs Multimodal RL 方法

- MM-Eureka [Meng et al. 2025](https://arxiv.org/abs/2503.07365)
- LMM-R1 [Peng et al. 2025](https://arxiv.org/abs/2503.07536)
- VLM-R1 [Shen et al. 2025](https://arxiv.org/abs/2504.07615)
- Visual-RFT [Liu et al. 2025](https://arxiv.org/abs/2503.01785)

这些主要 extend text-only CoT 到 multimodal tasks，但 reasoning 仍 in text modality。DeepEyes 真正 interleave visual reasoning。

---

## 7. 我的 Intuition & 相关联想

### 7.1 Test-Time Compute Scaling 的新轴

DeepEyes 本质上是 OpenAI o-series 提出的 test-time compute scaling 在 multimodal 上的延伸。原来 test-time scaling 只有 "think longer in text" 一个轴。现在多了 "look more in image" 这个轴。两个轴可以叠加——模型可以同时 think longer 和 look more。这开启了 **visual test-time scaling law** 的可能性。

### 7.2 Active Perception vs Passive Perception

传统 VLM 是 passive perception：一次性 encode 整图，然后 reasoning。DeepEyes 是 active perception：模型决定看哪里、看几次。这跟 cognitive science 里的 active vision [Findlay & Gilchrist, 2003](https://www.cambridge.org/core/books/active-vision/9C4F0F5C0C0A0A2C4D9C9D9C0A0A0A0) 一致——人类视觉本质是 active 的，不是 camera。

这让我想到 [ACT-R](https://en.wikipedia.org/wiki/ACT-R) cognitive architecture 里的 visual buffer——人类 working memory 有 dedicated visual buffer，通过 eye movements 更新。DeepEyes 的 iMCoT trajectory 就类似这个 visual buffer 的计算实现。

### 7.3 Tool Use 作为 Cognitive Prosthesis

模型自己的 grounding 能力被封装成 tool 调用。这很有意思——相当于把模型的一个 submodule "externalize" 成 tool interface，然后通过 RL 让模型学会 use itself。这跟 [Voyager](https://arxiv.org/abs/2305.16291) 里 agent 学会 use自己写的 skill library 有点像，但更 self-referential。

### 7.4 Reward Hacking 风险

Conditional tool reward 虽然防止了 "为调用而调用"，但还有风险：模型可能学会 "先盲猜再调用工具确认" 的 degenerate pattern。从 Figure 3 stage 3 看，模型确实减少了调用，但仍保持高 accuracy——这可能意味着它在 internal reasoning 已经够准时就不调用，但也可能意味着它学会了 "调用一次就够，不需要多次验证"。后者可能 lose robustness。

### 7.5 联系到 AlphaGo 的 Policy Iteration

训练 3-stage dynamic 让我想到 AlphaGo 的演化：
- Stage 1: 早期 policy network 探索（rollout policy）
- Stage 2: reinforcement learning 阶段，aggressive improvement
- Stage 3: MCTS + value network，selective and efficient

DeepEyes 也有类似的从 "exploration-heavy" 到 "exploitation-heavy" 的演化。可能这是 RL 训练的 universal dynamic。

### 7.6 失败案例的启示（Appendix D.2）

Figure 11（grounding limitation）和 Figure 12（reasoning limitation）揭示了当前限制：
- Grounding drift：第二次 zoom 时 bbox 漂移到错误区域——这暗示模型的 spatial memory 有问题，前一次 crop 的信息没被 well-integrated 到下次 grounding 的 context
- Reasoning limitation：zoom 准了但 reasoning 还是错——这说明 perception 和 reasoning 还是 decoupled 的，zoom in 提供了 information 但 model 不会用

这两个 limitation 都指向 **foundation model capability** 的 bottleneck。用 32B 或 72B base model 可能缓解。作者在 Section E 也承认这一点。

### 7.7 Future Direction: Tool Diversity

作者提到未来要加更多 tool（search、draw auxiliary lines）。这让我想到：
- **Draw auxiliary lines** 对应 geometry reasoning——类似人类在图上画辅助线解题
- **Search** 对应 external knowledge retrieval——类似 human consulting reference
- 可能还可以加 **generate sub-image**（让 model 画草图）——对应 sketch-based reasoning

每个新 tool 都会引入新的 reasoning pattern space。这跟 [Toolformer](https://arxiv.org/abs/2302.04761) 的 vision 版本演化方向一致。

### 7.8 联系到 System 1 / System 2 Thinking

[Visual Agents as Fast and Slow Thinkers](https://arxiv.org/abs/2408.08862) 把 visual reasoning 分成 fast（perception）和 slow（deliberation）。DeepEyes 的 iMCoT 在某种意义上 unify 两者：
- Text CoT 是 slow thinking
- Zoom-in tool call 是 fast perception 的主动调用
- 两者 interleave，形成 Kahneman 的 System 1 + System 2 协同

### 7.9 数据效率

47k samples + 80 iterations RL training（H100 GPUs）就能 unlock 这个能力，相对 SFT-based 方法动辄百万级数据，效率很高。这跟 R1 的发现一致——RL 比 SFT 更 sample-efficient for reasoning。

### 7.10 为什么不需要 cold-start SFT？

这是 paper 最 surprising 的 claim。我的猜测：
- Qwen2.5-VL 本身已经有 grounding 能力（从 refCOCO 89.1 看）
- Tool calling 的 format（`<tool_call>{...}</tool_call>`）通过 system prompt 就能 elicit
- RL 只需要 "incentivize" 而非 "teach"——model 已经有 capability，只是没被 aligned 到 "use it in reasoning"

这跟 [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) 的发现一致：base model 已经有 reasoning capability，RL 只是 unlock。但 R1-Zero 用了更强大的 base（DeepSeek-V3），DeepEyes 在 7B 上也能 work，更 impressive。

---

## 8. 实施细节（Implementation Details）

- Base model: Qwen2.5-VL-7B ([technical report](https://arxiv.org/abs/2502.13923))
- Algorithm: GRPO
- Iterations: 80
- Batch size: 256 prompts per batch
- Rollouts per prompt: 16
- Max tool calls: 6
- Max response length: 20480 tokens
- KL coefficient: 0.0（和 R1 一样，trust the policy gradient）
- Hardware: H100 GPUs

KL = 0 这点值得注意——通常 RLHF 加 KL regularization 防止 policy 漂离 reference。R1 和 DeepEyes 都发现 KL = 0 效果更好，可能因为 exploration 更充分。但这也可能导致 training instability，需要 careful learning rate scheduling。

---

## 9. 总结：DeepEyes 的 Positioning

DeepEyes 在我看来是 multimodal reasoning 的一个重要 milestone：
1. **Conceptual**：把 active perception 引入 CoT，bridge vision 和 language reasoning
2. **Methodological**：证明 end-to-end RL 不需要 SFT 就能 incentivize tool-augmented visual reasoning
3. **Empirical**：7B model 超过 32B 和 workflow 方法
4. **Interpretability**：emerge 的 thinking patterns 跟人类认知对应

潜在 impact：
- 启发更多 "active perception" 的 multimodal agent 设计
- 推动 visual test-time scaling law 研究
- 为下一代 o3-style open-source model 提供 blueprint

代码开源于 [github.com/Visual-Agent/DeepEyes](https://github.com/Visual-Agent/DeepEyes)。

---

## 关键 References

- [DeepEyes GitHub](https://github.com/Visual-Agent/DeepEyes)
- [OpenAI o3 Thinking with Images](https://openai.com/index/thinking-with-images/)
- [V* Bench (Wu & Xie, CVPR 2024)](https://arxiv.org/abs/2412.04467)
- [DeepSeek-R1 (RL for reasoning)](https://arxiv.org/abs/2501.12948)
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [ZoomEye](https://arxiv.org/abs/2411.16044)
- [Najemnik & Geisler, Nature 2005 - optimal eye movements](https://www.nature.com/articles/nature03390)
- [DAgger (Ross et al.)](https://arxiv.org/abs/1011.0686)
- [Visual-CoT (Shao et al. NeurIPS 2024)](https://arxiv.org/abs/2405.09118)
- [VLM-R1](https://arxiv.org/abs/2504.07615)

希望这些细节和 intuition 对你有用，Andrej！如果你想 dive deeper into 某个具体部分（比如 GRPO 的 multi-turn loss masking 实现细节、或者 reward shaping 的 ablation），我可以继续展开。
