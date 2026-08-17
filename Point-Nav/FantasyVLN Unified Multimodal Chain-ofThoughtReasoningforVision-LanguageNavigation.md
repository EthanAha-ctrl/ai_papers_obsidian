---
source_pdf: FantasyVLN Unified Multimodal Chain-ofThoughtReasoningforVision-LanguageNavigation.pdf
paper_sha256: d29b211d4422927f4436c5d8581326cc328c7f8d7021173132ac10451033e7c5
processed_at: '2026-08-04T06:38:17-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 FantasyVLN

## 一句话总结

**训练时让模型学会"想"（reasoning），推理时直接"做"（action），通过把"想象出来的画面"压缩到极小的latent space，同时训练4种思考模式并让它们互相校准。**

---

## 问题是什么？

想象你给一个机器人下指令："去厨房拿杯水，然后到客厅的沙发旁边停下"。

这个任务为什么难？因为：

1. **Long-horizon**: 要走很多步，中间一步走错，后面全崩
2. **Multi-stage**: 3个subtask（找厨房→拿水→找沙发），每个都要判断"我到了没"
3. **Semantic-spatial gap**: 语言说"厨房"，视觉看到的是"有个冰箱和灶台的空间"，这俩要对上
4. **Real-time constraint**: 机器人不能想5秒才走一步，要实时响应

现在的CoT方法陷入两难：

- **只生成文字CoT** (NavCoT, Aux-Think): 模型会"自言自语"规划，但看不到"想象的未来画面"，spatial grounding弱
- **生成视觉CoT** (CoT-VLA): 模型每步都要"脑补"未来画面再决定动作，但一张图就是几千tokens，5步就3k-5k tokens，机器人卡死在"想象"上

这就好比：你要么只会默念路线但不会预判转弯后看到什么，要么会在脑中逐像素渲染整条街景——太慢了，走不了路。

---

## 核心Idea：三个巧妙设计

### 设计1：用VAR把"想象"压到30个token

VAR (Visual AutoRegressive, Tian et al. NeurIPS 2024) 的核心idea是 **next-scale prediction** 而非 next-token prediction。

传统VAE/VQ-VAE怎么压缩图像？把256×256的图编码成64×64的latent grid，压缩比1/64。

VAR不一样。它把图像看成 **multi-scale的层级结构**：

```
Scale 1: 1×1    (最粗，全局色调)
Scale 2: 2×2    (粗略结构)
Scale 3: 4×4    (大致形状)
Scale 4: 8×8    (细节)
...
Scale K: 256×256 (原图)
```

每个scale作为一个整体预测，且是 **residual** 的——每个scale只预测前一个scale没编码好的"残差"信息。

所以对于VLN，我们只需要预测前几个coarse scales（论文中scale=4最优），就能捕捉"未来大概是个什么场景"的语义。这30个latent tokens就够了，不用生成像素。

**直觉类比**：你不是在脑中渲染4K高清街景，而是在脑中画个粗略的火柴人草图——"前面应该是个走廊，左边有门"。这个草图足够指导你走路，但生成成本极低。

Table 1的数据很震撼：
- VAE: 压缩比1/64，MSE 0.005
- VAR: 压缩比1/2185，MSE 0.039

VAR牺牲了一点点重建质量（MSE从0.005到0.039），但压缩比提升34倍。在VLN场景下，这个trade-off非常划算——我们不需要看清想象中的画面，只需要"想到"足够指导动作的画面。

### 设计2：一个模型同时训练4种"思考模式"

这是论文最elegant的设计。用两个binary gate $g_\mathcal{T}, g_\mathcal{V} \in \{0,1\}$ 控制：

| $g_\mathcal{T}$ | $g_\mathcal{V}$ | 模式 | 模型做什么 |
|---|---|---|---|
| 0 | 0 | **Non-CoT** | 直接看图→输出action |
| 1 | 0 | **T-CoT** | 先输出文字推理→再输出action |
| 0 | 1 | **V-CoT** | 先输出VAR latent（想象画面）→再输出action |
| 1 | 1 | **MM-CoT** | 同时输出文字+VAR latent→再输出action |

训练时每个batch随机sample一个模式，模型共享所有参数，只是输入前面加不同的special token告诉它"这次用哪种方式思考"。

**为什么这work？** 因为这4种模式本质是在用不同视角学习同一个"导航policy"：
- Non-CoT是"直觉反射"
- T-CoT是"语言规划"
- V-CoT是"视觉预演"
- MM-CoT是"完整的人类式思考"

让一个模型同时掌握这4种，相当于multi-task learning——每种模式都在强化对"navigation problem"本身的理解，互补。

### 设计3：Cross-Mode Alignment——让4种模式不打架

这是论文的killer insight。

直觉上，如果让一个模型同时学4种思考方式，会发生什么？**Mode conflict**。文字模式说"该左转"，视觉模式说"该直走"，non-CoT模式又说"该右转"——模型无所适从。

Table 5的数据触目惊心：

| 有Alignment | SR |
|---|---|
| 没有 | 0 |
| 有 | 2.44 |

没有alignment，SR直接是0！模型完全学不到consistent policy。

**怎么解决？** 用non-CoT模式当"锚点"，强制其他3种CoT模式的action prediction向non-CoT的prediction对齐。

具体做法是个类似knowledge distillation的trick：

1. 先用non-CoT模式forward一次，得到prediction $\widehat{\mathcal{A}}_t$
2. 用GT action更新non-CoT的参数
3. 再forward一次non-CoT，stop-gradient，得到soft target $\widetilde{\mathcal{A}}_t$
4. 3种CoT模式分别forward，它们的action prediction $\widehat{\mathcal{A}}_t^\mathcal{T}, \widehat{\mathcal{A}}_t^\mathcal{V}, \widehat{\mathcal{A}}_t^\mathcal{M}$ 都要向 $\widetilde{\mathcal{A}}_t$ 靠拢

**直觉类比**：你有4个实习生，分别用文字分析、图象分析、综合分析来推荐投资决策。你要求：不管你们怎么分析，最后给出的"买/卖"结论必须一致。这样他们的推理过程会互相校准，形成内在一致的判断力。

---

## 推理时：直接"做"，不"想"

关键来了：训练时4种模式都练，**推理时只用non-CoT**。

为什么？因为VLN要real-time。Table 4显示：
- Explicit CoT (CoT-VLA): 0.19 APS（每秒0.19个action，太慢）
- Implicit (FantasyVLN): 1.03 APS（快5.4倍）

但奇怪的是，Table 6显示推理时不"想"反而比"想"更好：

| 推理模式 | MM-CoT训练 | SR |
|---|---|---|
| Explicit (想出来) | ✓ | 0.98 |
| Implicit (不直接想) | ✓ | **2.44** |

为什么训练了CoT但推理不输出CoT反而更好？论文给出两个理由：

1. **训练数据有限** (LH-VLN只有18k slices)，explicit CoT容易overfit
2. **Explicit CoT会累积误差**：长trajectory中，一步CoT想错了，后面步步受影响

**直觉**：就像你学开车时，教练让你默念"看后视镜→打转向灯→看盲区→转方向盘"。训练时这样练能让你形成正确的肌肉记忆。但真正上路时，你不能边开边默念——你得让这些步骤内化为直觉反应。CoT是"训练辅助轮"，不是"驾驶方式"。

---

## 实验结果的几个Surprising Point

### 1. Visual CoT baselines在LH-VLN上全军覆没

CoT-VLA和WorldVLA的SR都是0。这说明pixel-space V-CoT在long-horizon场景完全无法泛化。为什么？因为pixel reconstruction的梯度信号太弱——模型把所有capacity都花在"重建画面"上，没学会"画面意味着什么"。

Figure 5的training efficiency对比很直观：WorldVLA训练10k iterations还在挣扎，FantasyVLN几千iterations就收敛了。这就是latent space reasoning的优势——优化landscape友好得多。

### 2. MM-CoT单独训练效果最差，但组合起来最好

Table 3里，单用MM-CoT (SR=0.49)比单用V-CoT (SR=1.46)还差。但4种模式组合后达到最佳(SR=2.44)。

这说明MM-CoT本身最难学（要同时处理文字+视觉），但作为multi-task的一部分，它提供了最rich的supervision signal。其他模式"帮"它学，它也"帮"其他模式学。

### 3. Scale=4是sweet spot

Figure 3显示VAR scale=4最优。论文的解释：太小信息不够，太大有冗余。Figure 4的重建对比很直观——scale 1-3的重建模糊得看不清，scale 5+和scale 4差别不大。

---

## 公式深入讲解

### Eq.(4): Joint Training Loss

$$\mathcal{L}_{\text{Joint}} = (\neg g_\mathcal{T} \land \neg g_\mathcal{V}) \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t, \mathcal{A}_t)$$
$$+ (g_\mathcal{T} \land \neg g_\mathcal{V}) \mathcal{L}_{CE}([\widehat{\mathcal{T}}_t, \widehat{\mathcal{A}}_t], [\mathcal{T}_t, \mathcal{A}_t])$$
$$+ (\neg g_\mathcal{T} \land g_\mathcal{V}) \mathcal{L}_{CE}([\widehat{\mathcal{V}}_t, \widehat{\mathcal{A}}_t], [\mathcal{V}_t, \mathcal{A}_t])$$
$$+ (g_\mathcal{T} \land g_\mathcal{V}) \mathcal{L}_{CE}([\widehat{\mathcal{M}}_t, \widehat{\mathcal{A}}_t], [\mathcal{M}_t, \mathcal{A}_t])$$

**变量含义**：
- $g_\mathcal{T}, g_\mathcal{V}$: binary gate, 控制textual/visual CoT是否激活
- $\widehat{\mathcal{A}}_t$: 预测的action序列
- $\mathcal{A}_t$: ground truth action序列
- $\widehat{\mathcal{T}}_t, \mathcal{T}_t$: 预测/真实的textual reasoning
- $\widehat{\mathcal{V}}_t, \mathcal{V}_t$: 预测/真实的visual reasoning (VAR latents)
- $\widehat{\mathcal{M}}_t, \mathcal{M}_t$: 预测/真实的multimodal reasoning = $[\mathcal{T}_t, \mathcal{V}_t]$
- $\mathcal{L}_{CE}$: causal cross-entropy loss (因为是autoregressive生成)
- $\land, \neg$: 逻辑与、逻辑非

**逻辑**：这是一个用布尔逻辑表达的mixture loss。每个batch sample一个 $(g_\mathcal{T}, g_\mathcal{V})$ 组合，只有对应的那一项loss被激活，其他项被乘0。这样4种模式共享参数但轮流监督。

### Eq.(7)-(9): Cross-Mode Alignment

$$\mathcal{L}_{\text{Joint}}^* = \mathcal{L}_{\text{Align}} + \mathcal{L}_{\text{CoT}}$$

$$\mathcal{L}_{\text{Align}} = \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t^\mathcal{T}, \widetilde{\mathcal{A}}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t^\mathcal{V}, \widetilde{\mathcal{A}}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t^\mathcal{M}, \widetilde{\mathcal{A}}_t)$$

$$\mathcal{L}_{\text{CoT}} = \mathcal{L}_{CE}(\widehat{\mathcal{T}}_t, \mathcal{T}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{V}}_t, \mathcal{V}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{M}}_t, \mathcal{M}_t)$$

**变量含义**：
- $\widehat{\mathcal{A}}_t^\mathcal{T}, \widehat{\mathcal{A}}_t^\mathcal{V}, \widehat{\mathcal{A}}_t^\mathcal{M}$: 3种CoT模式各自的action预测
- $\widetilde{\mathcal{A}}_t$: non-CoT模式的soft target（stop-gradient）

**直觉**：$\mathcal{L}_{\text{CoT}}$ 保证CoT reasoning本身的质量（生成的文字/视觉要和GT对齐），$\mathcal{L}_{\text{Align}}$ 保证不同模式的action决策一致。前者管"想得对"，后者管"做得一致"。

### Eq.(10): APS

$$\text{APS} = \frac{N_{\text{act}}}{T_{\text{nav}}}$$

- $N_{\text{act}}$: 总执行action数
- $T_{\text{nav}}$: 总navigation时间(秒)

这是衡量real-time能力的关键metric。1.0 APS意味着每秒1个action，对VLN基本可用。CoT-VLA的0.19 APS意味着5秒才走1步——这机器人没法用。

---

## 几个值得深想的Connection

### 1. 与Dreamer系列的connection

CompV-CoT本质是一种 **latent imagination**。Dreamer (Hafner et al.) 也是在latent space想象未来，再用想象来规划。区别是：
- Dreamer用latent dynamics model预测
- FantasyVLN用VLM直接生成VAR latents

两者都在说同一件事：**embodied agent不需要pixel-perfect的imagination，需要的是latent-level的"未来预演"**。

Reference: [DreamerV3](https://arxiv.org/abs/2301.04104)

### 2. 与System 1/System 2 thinking的connection

Kahneman的System 1 (fast, intuitive) vs System 2 (slow, deliberative)框架在这里很贴切：
- Non-CoT = System 1 (推理时用)
- T/V/MM-CoT = System 2 (训练时用)

FantasyVLN的本质是：**用System 2训练，把System 2的能力蒸馏进System 1**。

这跟Anthropic最近关于"internal reasoning"的讨论、OpenAI的"thinking tokens"想法都相关。只不过FantasyVLN走得更极端——连implicit thinking tokens都不要，直接把reasoning融入weights。

Reference: [Anthropic: Training language models to reason internally](https://www.anthropic.com/research/training-language-models-to-reason-internally)

### 3. 与Aux-Think的关系

Aux-Think (Wang et al., NeurIPS 2025) 是FantasyVLN的直接前作，提出了"train-with-CoT, infer-without-CoT"的paradigm。FantasyVLN在此基础上扩展到multimodal，并发现多模态CoT需要cross-mode alignment才能work——这是Aux-Think没遇到的问题，因为单模态不存在mode conflict。

Reference: [Aux-Think paper](https://arxiv.org/abs/2505.13461)

### 4. 与Mixture-of-Experts的微妙呼应

虽然FantasyVLN不是MoE，但gating机制有相似之处。MoE用gate选expert，FantasyVLN用gate选reasoning mode。区别是：
- MoE: 不同expert处理不同样本，参数不共享
- FantasyVLN: 同一参数处理所有模式，mode通过input conditioning切换

这更像是 **conditional computation on reasoning modality**，可以看作"reasoning mode层面的meta-learning"。

### 5. 与Test-time Scaling的tension

当前LLM社区在讨论test-time scaling（o1, R1等）——推理时多花compute换取更好效果。FantasyVLN反其道而行：训练时多花compute（4种模式），推理时少花compute。

这俩方向看似矛盾，实则互补：
- Language reasoning任务：latency容忍度高，test-time scaling适用
- Embodied任务：real-time硬约束，train-time scaling更合适

这可能是embodied AI和NLP领域方法论分化的一个信号。

---

## 几个Critical的Point

### 1. SR绝对值还是很低

最好的FantasyVLN在LH-VLN上SR只有2.44%。这意味着100个multi-stage任务只成功2-3个。这提醒我们long-horizon VLN远没解决。可能需要：
- 更大的pretraining
- RL fine-tuning
- Hierarchical planning
- Better exploration

### 2. T-CoT标注依赖Qwen-VL-Max

用大模型生成CoT标注是个趋势，但会引入annotation bias。如果Qwen-VL-Max本身的spatial reasoning有偏差，这些偏差会被蒸馏进FantasyVLN。

### 3. Scale=4的empirical性

为什么是scale=4而不是3或5？论文给了empirical解释但缺乏理论。这暗示VAR latent的选择可能需要per-task tuning，不够"plug-and-play"。

### 4. 只在LH-VLN上验证

没有R2R, RxR, VLN-CE等其他benchmark的结果。可能这种方法特别适合long-horizon但在short-horizon上没优势，或者VAR latent对某些环境不适用。Generalization存疑。

### 5. Cross-mode alignment的stop-gradient

Algorithm 1第7行用stop-gradient防止soft target参与backprop。这是个工程trick，但理论上为什么这样work得这么好？是否可以用EMAs、moving averages等更soft的方式？这些都是可以深挖的direction。

---

## 实操层面的几个细节

### Trajectory Slicing的$k=5$

每个训练样本预测未来5个actions。这是个compromise：
- $k$太大：长期预测误差累积，训练信号噪声大
- $k$太小：每步都要决策，real-time压力

$k=5$对应大约2-3秒的navigation（假设每action约0.5秒），这个时长内环境变化可控，预测可信。

### Data Augmentation的哲学

两种augmentation（uniform subsampling和stochastic trimming）都是针对 **history**，不动 **current observation**。这个设计很微妙：
- Current obs是决策的直接依据，不能动
- History是context，可以有"记忆模糊"的robustness

这模拟了真实场景：机器人可能记得"我刚才路过一个厨房"，但记不清具体第3帧是什么。训练时引入这种模糊性，让模型学会依赖"关键信息"而非"完整序列"。

### Special Tokens设计

```
<|forward|>, <|left|>, <|right|>, <|stop|>     # actions
<|1|> ~ <|4096|>                                # VAR latents (vocab size 4096)
<|NAV|>, <|CoT|>, <|/CoT|>                      # system tokens
<textual think>, <no textual think>             # gate signals
<visual think>, <no visual think>               # gate signals
```

VAR latent token vocab size是4096，意味着每个scale的每个position是从4096个codebook entries中选一个。这比pixel prediction简单多了。

---

## 总结：这篇paper的真正贡献

用一句话说：**它证明了embodied agent可以在latent space里"想象"，并把这想象能力内化为直觉**。

用更广的视角看，这是"reasoning representation"探索的一个milestone：
- 显式CoT (text tokens) → latent CoT (compressed visual tokens) → implicit CoT (in weights)
- 每一步都是reasoning representation的"压缩"

这跟"从chain-of-thought到implicit reasoning"的大趋势完全一致，但落地到了embodied场景，并解决了real-time约束这个硬问题。

从Karpathy的视角，这让人想起你 (Andrej) 在 [YouTube讲座](https://www.youtube.com/watch?v=VMj-3S1vk0w) 里讲"software 2.0"——weights本身就是程序。FantasyVLN把reasoning也写进了weights，不只是input-output mapping。这是迈向"software 3.0"——reasoning-aware weights——的一个step。

Future direction的话，我直觉上觉得：
1. **加入RL**：SFT学的是"模仿expert"，但expert数据有限。用RL让agent自己探索，可能突破SR的天花板
2. **Hierarchical CoT**：VAR的multi-scale天然适合hierarchical planning——coarse scale做"去哪个房间"，fine scale做"走哪条路"
3. **World model integration**：把VAR换成learnable world model，让agent在world model里rollout做planning
4. **Real robot**：从sim到real的gap是最终考验

Reference: 
- [FantasyVLN Project](https://fantasy-amap.github.io/fantasy-vln/)
- [VAR original paper](https://arxiv.org/abs/2404.02905)
- [LH-VLN benchmark](https://arxiv.org/abs/2503.14834)
- [Aux-Think](https://arxiv.org/abs/2505.13461)
- [Karpathy: Software 2.0](https://karpathy.medium.com/software-2-0-a6c52ba1c4d6)
- [Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

# FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation 深度解析

## 1. 论文核心问题与动机

这篇paper要解决的核心问题是 **VLN (Vision-and-Language Navigation)** 中的 **long-horizon, multi-stage navigation** 场景。让我先build up the intuition关于为什么这个问题难：

### 1.1 任务本质

VLN要求一个 **embodied agent** $\pi_\theta$ 在连续3D环境 $\mathcal{O}$ 中，根据自然语言指令 $\mathcal{T}$ 和多视角视觉观察 $o_t$ 预测未来动作 $\mathcal{A}_t$。这是一个 **non-Markovian temporal decision problem**：

$$\mathcal{A}_t \sim \pi_\theta(\mathcal{T}, \{o_{\leq t}\})$$

其中：
- $o_{\leq t}$ 表示从初始到当前时刻 $t$ 的所有visual observations历史
- $\mathcal{A}_t = \{a_t, a_{t+1}, \ldots, a_{t+k-1}\}$ 是未来 $k$ 步action序列（论文中 $k=5$）
- action space $\mathcal{U}$ 包含 `<|forward|>`, `<|left|>`, `<|right|>`, `<|stop|>`

### 1.2 现有CoT方法的两难困境

这是论文最精彩的insight所在。让我用一个表格梳理：

| 方法 | CoT模态 | 问题 |
|------|---------|------|
| **NavCoT** (Lin et al., 2025b) | Textual only | 缺乏spatial grounding，overfit训练分布 |
| **NavGPT-2** (Zhou et al., 2024) | Textual only | 同上 |
| **CoT-VLA** (Zhao et al., 2025) | Visual (pixel-space) | Token爆炸：5-7 actions → 3k-5k tokens |
| **OctoNav-R1** (Gao et al., 2025) | Multimodal | 同上，real-time不可行 |
| **Aux-Think** (Wang et al., 2025) | Textual + implicit | 仅textual模态，spatial信息缺失 |

**Token inflation的核心痛点**：multimodal CoT需要iteratively generate + interpret imagined intermediate observations。一个5-7步的reasoning step会膨胀到3k-5k tokens，相比textual CoT的<500 tokens，order of magnitude的膨胀。这导致即使high-end GPUs也无法real-time navigation。

## 2. FantasyVLN的核心设计思想

论文的核心idea可以概括为 **"Train with CoT, Infer without CoT"** 范式，但在两个关键维度上做了创新：

### 2.1 两大核心创新

**Innovation 1: Compact Visual Chain-of-Thought (CompV-CoT)**

将imagine observation tokens从pixel space压缩到 **VAR (Visual AutoRegressive) model**的latent space。VAR采用 **next-scale prediction** paradigm，hierarchically编码视觉信息：

- 256×256 image → 仅需30个visual tokens
- 压缩比 1/2185 (vs VAE的1/64, VQ-VAE的1/64)

**Innovation 2: Unified Multimodal CoT (UM-CoT) with Cross-Mode Alignment**

四个reasoning modes在一个模型中联合训练：
- **Non-CoT** $(g_\mathcal{T}=0, g_\mathcal{V}=0)$: 直接instruction→action
- **T-CoT** $(g_\mathcal{T}=1, g_\mathcal{V}=0)$: 文本推理链
- **V-CoT** $(g_\mathcal{T}=0, g_\mathcal{V}=1)$: VAR latent space视觉推理
- **MM-CoT** $(g_\mathcal{T}=1, g_\mathcal{V}=1)$: 文本+视觉配对推理

### 2.2 整体架构图解析

从Figure 2可以看出架构由以下components构成：

```
Input: [Instruction T, Visual Observations {o_≤t}, Gate (g_T, g_V)]
    ↓
[Qwen2.5-VL Base Model + LoRA]
    ↓
    ├── Mode (a): non-CoT → direct action Â_t
    ├── Mode (b): T-CoT → [T̂_t, Â_t^T]
    ├── Mode (c): V-CoT → [Ĥ_t, Â_t^V] → (VAR decoder) → V̂_t
    └── Mode (d): MM-CoT → [M̂_t = (T̂_t, Ĥ_t), Â_t^M]
    ↓
Cross-Mode Alignment: Â_t^T, Â_t^V, Â_t^M → align to Ã_t (non-CoT soft target)
```

## 3. 方法详解

### 3.1 Problem Setup的数学形式化

Formally, VLN定义为：

$$\pi_\theta: (\mathcal{T}, \{o_{\leq t}\}) \mapsto \mathcal{A}_t \in \mathcal{U}$$

- $\mathcal{T}$: natural language instruction
- $s_0$: initial state (location + orientation)
- $\mathcal{U}$: action space
- $\dot{T}$: maximum step limit

Agent与环境interaction循环直到 `<|stop|>` 或达到 $\dot{T}$。

### 3.2 Compact Visual Chain-of-Thought (CompV-CoT)

这是论文的关键技术贡献。让我深入解析VAR模型的工作机制：

#### 3.2.1 VAR模型原理

VAR (Tian et al., 2024, NeurIPS)采用 **multi-scale residual quantization** 思想：

给定图像 $x \in \mathbb{R}^{256 \times 256 \times 3}$，VAR将其编码为hierarchical latent representations：

$$\mathcal{H} = \{R_1, R_2, \ldots, R_K\}$$

其中：
- $R_k$ 表示第 $k$ 个scale的residual latent
- $K$ 是total scale数
- 每个scale的token数随scale增大而增加（coarse-to-fine）

**Next-scale prediction**:
$$p(\mathcal{H}) = \prod_{k=1}^{K} p(R_k | R_{<k})$$

这与传统autoregressive的 **next-token prediction** $p(x_t | x_{<t})$ 不同，而是预测 **整个scale的所有tokens**。

#### 3.2.2 压缩比对比

Table 1的关键数据：

| Compressor | Compression Ratio | MSE |
|-----------|------------------|-----|
| RAE-DINOv2-B | 1/256 | 0.012 |
| RAE-SigLIP2-B | 1/256 | 0.011 |
| VAE | 1/64 | 0.005 |
| VQ-VAE | 1/64 | 0.007 |
| **VAR** | **1/2185** | 0.039 |

注意VAR的MSE (0.039)略高于VAE (0.005)，但压缩比是VAE的34倍。在VLN场景下，这种trade-off是有利的，因为我们更关心inference speed而非perfect reconstruction。

#### 3.2.3 CompV-CoT的训练流程

CompV-CoT推理模式形式化为：

$$[\widehat{\mathcal{H}}_t, \widehat{\mathcal{A}}_t] \sim \pi_\theta(\mathcal{T}, \{o_{\leq t}\}, g_\mathcal{T}=0, g_\mathcal{V}=1)$$

其中 $\widehat{\mathcal{H}}_t$ 是预测的VAR latent representations。然后通过frozen VAR decoder重建pixel observations：

$$\widehat{\mathcal{V}}_t \sim g(\widehat{\mathcal{H}}_t)$$

这里 $g$ 是VAR的next-scale generation pipeline。

**训练目标**:
$$\arg\min_\theta \mathcal{L}_{CE}(\widehat{\mathcal{V}}_t, \mathcal{V}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t, \mathcal{A}_t)$$

注意：VAR在训练时是frozen的，只有VLM (Qwen2.5-VL)被fine-tuned。

### 3.3 Unified Multimodal CoT (UM-CoT)

#### 3.3.1 Gating机制

两个binary gating signals $g_\mathcal{T}, g_\mathcal{V} \in \{0, 1\}$ 控制reasoning mode：

$$[\widehat{\mathcal{R}}_t, \widehat{\mathcal{A}}_t] = \pi_\theta(\mathcal{T}, \{o_{\leq t}\}, g_\mathcal{T}, g_\mathcal{V})$$

其中reasoning output $\widehat{\mathcal{R}}_t$ 由Eq.(2)决定：

$$\widehat{\mathcal{R}}_t = \begin{cases} 
\text{None}, & \text{if } (g_\mathcal{T}, g_\mathcal{V}) = (0, 0) \\
\widehat{\mathcal{T}}_t, & \text{if } (g_\mathcal{T}, g_\mathcal{V}) = (1, 0) \\
\widehat{\mathcal{V}}_t, & \text{if } (g_\mathcal{T}, g_\mathcal{V}) = (0, 1) \\
\widehat{\mathcal{M}}_t, & \text{if } (g_\mathcal{T}, g_\mathcal{V}) = (1, 1)
\end{cases}$$

#### 3.3.2 数据组织

Expert navigation dataset $\mathcal{D}$ 组织为five-tuples：

$$[\mathcal{T}, \{o_{\leq t}\}, \mathcal{T}_t, \mathcal{V}_t, \mathcal{A}_t] \in \mathcal{D}$$

其中：
- $\mathcal{T}_t$: ground truth textual reasoning steps (由Qwen-VL-Max生成)
- $\mathcal{V}_t$: ground truth CompV-CoT visual reasoning steps (VAR latents)

#### 3.3.3 Joint Training Objective

Eq.(4)的joint loss：

$$\mathcal{L}_{\text{Joint}} = (\neg g_\mathcal{T} \land \neg g_\mathcal{V}) \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t, \mathcal{A}_t)$$
$$+ (g_\mathcal{T} \land \neg g_\mathcal{V}) \mathcal{L}_{CE}([\widehat{\mathcal{T}}_t, \widehat{\mathcal{A}}_t], [\mathcal{T}_t, \mathcal{A}_t])$$
$$+ (\neg g_\mathcal{T} \land g_\mathcal{V}) \mathcal{L}_{CE}([\widehat{\mathcal{V}}_t, \widehat{\mathcal{A}}_t], [\mathcal{V}_t, \mathcal{A}_t])$$
$$+ (g_\mathcal{T} \land g_\mathcal{V}) \mathcal{L}_{CE}([\widehat{\mathcal{M}}_t, \widehat{\mathcal{A}}_t], [\mathcal{M}_t, \mathcal{A}_t])$$

训练时 $(g_\mathcal{T}, g_\mathcal{V})$ 被uniformly sampled，确保模型在所有四种模式上都有监督。

### 3.4 Cross-Mode Alignment Constraint

这是论文的第三个关键技术贡献，解决多模式冲突问题。

#### 3.4.1 核心思想

使用 **non-CoT模式作为anchor**，将所有CoT variants的action prediction对齐到non-CoT的soft target。这类似于知识蒸馏的思想。

#### 3.4.2 数学形式化

**Step 1**: 先优化non-CoT模式：
$$\mathcal{L}_{\text{non-CoT}} = \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t, \mathcal{A}_t)$$
$$\widehat{\mathcal{A}}_t = \pi_\theta(\mathcal{T}, \{o_{\leq t}\}, g_\mathcal{T}=0, g_\mathcal{V}=0)$$

**Step 2**: 重新forward获得soft target $\widetilde{\mathcal{A}}_t$ (stop-gradient)：
$$\widetilde{\mathcal{A}}_t = \text{sg}[\pi_\theta(\mathcal{T}, \{o_{\leq t}\}, g_\mathcal{T}=0, g_\mathcal{V}=0)]$$

**Step 3**: 计算alignment loss：
$$\mathcal{L}_{\text{Align}} = \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t^\mathcal{T}, \widetilde{\mathcal{A}}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t^\mathcal{V}, \widetilde{\mathcal{A}}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{A}}_t^\mathcal{M}, \widetilde{\mathcal{A}}_t)$$

其中 $\text{sg}[\cdot]$ 表示stop-gradient操作。

**Step 4**: Joint aligned objective：
$$\mathcal{L}_{\text{Joint}}^* = \mathcal{L}_{\text{Align}} + \mathcal{L}_{\text{CoT}}$$

其中：
$$\mathcal{L}_{\text{CoT}} = \mathcal{L}_{CE}(\widehat{\mathcal{T}}_t, \mathcal{T}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{V}}_t, \mathcal{V}_t) + \mathcal{L}_{CE}(\widehat{\mathcal{M}}_t, \mathcal{M}_t)$$

#### 3.4.3 Algorithm 1 伪代码解析

```
Algorithm 1: Cross-Mode Aligned Joint Training
1: Input: Dataset D, parameters θ, learning rate η, alignment weight λ_align
2: Output: Trained parameters θ*
3: while not converged do
4:   Sample [T, {o_≤t}, T_t, V_t, A_t] ~ D
5:   Compute non-CoT prediction: Â_t ← π_θ(T, {o_≤t}, g_T=0, g_V=0)
6:   Update θ with non-CoT loss: θ ← θ - η∇_θ L_CE(Â_t, A_t)
7:   Compute soft target: Ã_t ← sg[π_θ(T, {o_≤t}, g_T=0, g_V=0)]
8:   Compute T-CoT prediction: [T̂_t, Â_t^T] ← π_θ(T, {o_≤t}, g_T=1, g_V=0)
9:   Compute V-CoT prediction: [V̂_t, Â_t^V] ← π_θ(T, {o_≤t}, g_T=0, g_V=1)
10:  Compute MM-CoT prediction: [M̂_t, Â_t^M] ← π_θ(T, {o_≤t}, g_T=1, g_V=1)
11:  Compute L_Joint* using Eq.(7)
12:  Update θ with joint loss: θ ← θ - η∇_θ L_Joint*
13: end while
14: θ* ← θ
15: return θ*
```

注意第7行使用 `sg` (stop-gradient)，这非常关键——soft target不参与梯度回传，只作为监督信号。

## 4. 实验深度解析

### 4.1 LH-VLN Benchmark

LH-VLN (Song et al., CVPR 2025) 是专门为 **long-horizon, multi-stage navigation** 设计的benchmark：
- Multi-stage任务要求agent依次到达多个goals
- Longer trajectories放大cumulative errors
- Online evaluation，test set的tasks和scenes都unseen

### 4.2 Metrics详解

- **SR (Success Rate)**: 多stage任务整体成功率
- **ISR (Independent Success Rate)**: 各subtask独立成功率
- **CSR (Conditional Success Rate)**: 按前序subtask成功率加权的ISR
- **CGT (CSR weighted by Ground Truth)**: 按expert trajectory长度加权的CSR
- **APS (Action Per Second)**: $APS = \frac{N_{\text{act}}}{T_{\text{nav}}}$，衡量inference efficiency

### 4.3 Main Results分析

Table 2的核心结果：

| CoT Modal | Methods | SR | ISR | CSR | CGT |
|-----------|---------|-----|-----|-----|-----|
| None/ZS | Random | 0 | 0 | 0 | 0 |
| None/ZS | GLM-4v prompt | 0 | 0 | 0 | 0 |
| None/ZS | GPT-4 + NaviLLM | 0 | 2.19 | 1.45 | 2.61 |
| None/ZS | MGDM | 0 | 2.34 | 1.65 | 2.91 |
| Visual | CoT-VLA | 0 | 0 | 0 | 0 |
| Visual | WorldVLA | 0 | 0 | 0 | 0 |
| Textual | Aux-Think | 0.65 | 3.16 | 2.04 | 1.47 |
| **unified multimodal** | **FANTASYVLN** | **2.44** | **11.01** | **9.64** | **8.99** |

**关键观察**：

1. **所有visual CoT baselines (CoT-VLA, WorldVLA)在LH-VLN上完全失败** (SR=0)。这说明pixel-space V-CoT在long-horizon场景下无法generalize。

2. **FantasyVLN相比Aux-Think提升3.75x** (SR: 0.65→2.44)。这验证了multimodal CoT比纯textual CoT更适合long-horizon navigation。

3. **CGT上的巨大优势** (8.99 vs 1.47)说明FantasyVLN在更长trajectories上表现更好，这正是long-horizon的核心挑战。

### 4.4 Inference Efficiency对比

Table 4的关键数据：

| Reasoning Mode | Methods | Model Size | APS |
|---------------|---------|------------|-----|
| Explicit | CoT-VLA | 7B | 0.19 |
| Implicit | WorldVLA | 7B | 1.02 |
| Implicit | Aux-Think | 8B | 0.97 |
| Implicit | **FANTASYVLN** | 7B | **1.03** |

FantasyVLN达到1.03 APS，比CoT-VLA的0.19快 **5.4倍**。这验证了implicit reasoning的效率优势。

### 4.5 Ablation Studies深度解析

#### 4.5.1 Reasoning Mode组合贡献 (Table 3)

| non-CoT | T-CoT | V-CoT | MM-CoT | SR | ISR | CSR | CGT |
|---------|-------|-------|--------|-----|-----|-----|-----|
| ✓ | | | | 0 | 2.01 | 1.51 | 1.55 |
| ✓ | ✓ | | | 0.98 | 8.26 | 6.60 | 6.15 |
| ✓ | | ✓ | | 1.46 | 11.19 | 9.66 | 8.84 |
| ✓ | | | ✓ | 0.49 | 7.77 | 6.48 | 8.89 |
| ✓ | ✓ | ✓ | ✓ | **2.44** | **11.01** | **9.64** | **8.99** |

**重要insight**：
- V-CoT单独使用时ISR最高(11.19)，但在SR上不如全模式组合
- MM-CoT单独使用效果最差(SR=0.49)，这看似反直觉，但原因是MM-CoT的训练难度最大
- **所有模式组合最佳**，说明各模式提供complementary supervision signals

#### 4.5.2 VAR Scale选择 (Figure 3)

Scale从1到10的ablation：
- Scale=4最佳
- 较小scales缺乏视觉信息
- 较大scales导致redundancy

Figure 4的reconstruction comparison直观展示了不同scale的重建质量。

#### 4.5.3 Cross-Mode Alignment的关键作用 (Table 5)

| Alignment Constraint | SR | ISR | CSR | CGT |
|---------------------|-----|-----|-----|-----|
| ✗ | 0 | 2.39 | 1.19 | 1.28 |
| ✓ | **2.44** | **11.01** | **9.64** | **8.99** |

**没有alignment，SR直接从2.44掉到0**！这是论文最震撼的ablation结果。这说明多模式训练如果没有alignment约束，会导致 **mode conflict**——不同推理模式产生的action prediction相互干扰，最终导致模型无法学到consistent policy。

### 4.6 Explicit vs. Implicit Reasoning对比 (Table 6)

| Metrics | Mode | T-CoT | V-CoT | MM-CoT |
|---------|------|-------|-------|--------|
| SR | explicit | 0.98 | 0.49 | 0.98 |
| SR | implicit | 0.49 | 1.46 | **2.44** |
| ISR | explicit | 8.26 | 7.34 | 8.62 |
| ISR | implicit | 6.06 | 11.19 | **11.01** |

**重要发现**：
- T-CoT上explicit反而更好 (SR: 0.98 > 0.49)
- V-CoT和MM-CoT上implicit明显更好
- MM-CoT + implicit = 最佳组合

论文给出的解释：
1. LH-VLN训练数据有限(仅18k trajectory slices × 5 steps)，explicit CoT容易overfit
2. Explicit reasoning扩大temporal dependencies，misaligned CoT tokens会累积偏差

## 5. 数据准备与实现细节

### 5.1 Trajectory Slicing

每个navigation trajectory $\mathcal{T}_i$ 被切分为non-overlapping slices：

$$\{\mathcal{T}, \{o_{\leq t}\}, \mathcal{A}_t\}_{t \in S_i} \sim \text{Slice}(\mathcal{T}_i)$$

其中：
- $\mathcal{A}_t = \{a_t, a_{t+1}, \ldots, a_{t+k-1}\}$, $k=5$
- $S_i = \{1, 1+k, \ldots, T_i\}$
- $T_i$: trajectory $i$ 的action总数

### 5.2 T-CoT数据标注

使用 **Qwen-VL-Max** 对18,554个navigation slices进行T-CoT标注。标注prompt包含4个steps：

1. **Semantic Planning**: 将mission分解为sub-tasks with clear spatial goals
2. **Visual Description**: 描述historical和current images揭示的信息
3. **Action Decision-Making**: 预测next 5-step action sequence
4. **Visual Imagination**: 描述执行actions后的expected scene

### 5.3 Data Augmentation

两种augmentation策略：

**Uniform Subsampling** (prob=0.5, for N≥10):
$$\{h_1, h_2, \ldots, h_N\} \rightarrow \{h_1, h_3, h_5, \ldots\}$$

**Stochastic History Trimming** (for N≥7):
- 移除前2帧 (prob=0.5): $\{h_3, h_4, \ldots, h_N\}$
- 随机移除2连续帧 (prob=0.5): 移除 $\{h_k, h_{k+1}\}$

### 5.4 实现配置

- **Base model**: Qwen2.5-VL with LoRA on language layers + VL projection
- **Hardware**: 64× H20 GPUs (141GB each)
- **Optimizer**: AdamW, lr=1e-4, weight decay=0.1, cosine schedule with 5% warmup
- **Batch size**: 4 per device, 32 dataloader workers
- **Precision**: bfloat16, gradient checkpointing
- **Distributed**: DeepSpeed ZeRO-2

### 5.5 Special Tokens扩展

通过vocabulary extensibility引入：
- Action tokens: `<|forward|>`, `<|left|>`, `<|right|>`, `<|stop|>`
- VAR latent tokens: `<|1|>`–`<|4096|>` (for CompV-CoT和MM-CoT)
- System tokens: `<|NAV|>`, `<|CoT|>`, `<|/CoT|>`
- Gating tokens: `<textual think>`, `<no textual think>`, `<visual think>`, `<no visual think>`

## 6. 与相关工作的关系

### 6.1 CoT Reasoning谱系

```
Textual CoT (Wei et al., 2022)
    ├── Self-Consistency (Wang et al., 2023)
    ├── Least-to-Most (Zhou et al., 2023)
    └── VLN applications:
        ├── NavGPT (GPT-4 zero-shot)
        ├── NavCoT (disentangled reasoning)
        ├── NavGPT-2 (VLV-based)
        └── Aux-Think (implicit reasoning)

Visual CoT
    ├── CoT-VLA (pixel-space, manipulation)
    ├── DreamVLA (world knowledge)
    └── VISTA (visual imagination)

Multimodal CoT
    ├── Zhang et al. (2024b) - foundational
    ├── OctoNav-R1
    └── FantasyVLN (unified)
```

### 6.2 与Aux-Think的关系

FantasyVLN继承了Aux-Think的 **"train-with-CoT, infer-without-CoT"** 范式，但在两个维度扩展：
1. **Multimodal**: Aux-Think仅textual CoT，FantasyVLN加入visual和multimodal
2. **Cross-mode alignment**: 显式约束不同模式的一致性

## 7. 个人思考与潜在局限

### 7.1 核心创新点评价

**优点**：
1. **VAR压缩是巧妙设计**：解决了pixel-space V-CoT的token爆炸问题
2. **Cross-mode alignment是关键insight**：没有它，多模式训练完全失败
3. **Unified gating机制优雅**：单一模型支持4种模式，参数共享

**潜在问题**：
1. **VAR scale选择需人工调优**：Scale=4是empirical最优，但缺乏理论指导
2. **T-CoT标注依赖Qwen-VL-Max**：可能引入annotation bias
3. **SR绝对值仍然很低** (2.44%)：LH-VLN确实困难，但说明long-horizon VLN远未解决
4. **Evaluation仅在LH-VLN**：缺乏其他benchmark验证generalization

### 7.2 与更广research context的联系

这篇工作触及了几个deep research questions：

1. **Implicit vs. Explicit reasoning trade-off**: 与Anthropic对latent reasoning的研究、DeepMind的reasoning distillation工作呼应

2. **Latent space reasoning**: 与Dreamer系列、world model文献的connection——本质上CompV-CoT是latent imagination的一种形式

3. **Multi-task learning中的mode conflict**: Cross-mode alignment的思想可推广到其他multi-task场景

4. **Embodied AI的real-time约束**: 这篇工作提醒我们，robotics application对latency有硬约束，不能盲目追求复杂reasoning

### 7.3 未来方向联想

基于这篇工作，可以想象几个extension：

1. **Reinforcement Learning integration**: 当前是SFT，如果加入RL fine-tuning可能进一步提升
2. **Dynamic mode switching**: 目前inference固定用non-CoT，可以根据task difficulty动态选择
3. **Hierarchical VAR latents**: 利用VAR的multi-scale特性做hierarchical planning
4. **Cross-embodiment transfer**: 结合NaVILA的cross-embodiment思想
5. **Real robot deployment**: 从simulation到real-world的sim-to-real transfer

## 8. 参考链接

- **Project page**: https://fantasy-amap.github.io/fantasy-vln/
- **VAR paper**: [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)
- **LH-VLN benchmark**: [Towards Long-Horizon Vision-Language Navigation](https://arxiv.org/abs/2503.14834)
- **Aux-Think**: [Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation](https://arxiv.org/abs/2505.13461)
- **CoT-VLA**: [CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](https://arxiv.org/abs/2503.22046)
- **NavCoT**: [NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning](https://arxiv.org/abs/2501.18052)
- **Qwen2.5-VL**: [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- **R2R benchmark**: [Vision-and-Language Navigation](https://arxiv.org/abs/1711.07280)
- **VLN-CE**: [Beyond the Nav-Graph](https://arxiv.org/abs/2007.04545)
- **NavGPT-2**: [NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](https://arxiv.org/abs/2407.07023)

## 总结

FantasyVLN这篇工作的核心贡献在于：它同时解决了multimodal CoT在VLN中的两个fundamental问题——**token爆炸**和**mode conflict**。通过VAR latent compression和cross-mode alignment两个关键技术，实现了implicit multimodal reasoning，在LH-VLN benchmark上取得了显著提升(3.75x over Aux-Think)且保持了real-time inference能力(1.03 APS)。

从更广的视角看，这篇工作代表了 **embodied AI reasoning** 的一个重要方向：如何在保持reasoning能力的同时满足real-time约束。它提醒我们，单纯的explicit CoT scaling在robotics场景下是不可行的，必须寻找更高效的reasoning representation。这种"train with complex reasoning, infer with distilled policy"的思想，很可能成为未来embodied AI的标准范式。
