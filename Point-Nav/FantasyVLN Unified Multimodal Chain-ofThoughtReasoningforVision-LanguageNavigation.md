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

训练时让模型学会"想"（reasoning），推理时直接"做"（action），通过把"想象出来的画面"压缩到极小的latent space，同时训练4种思考模式并让它们互相校准。

想象你给一个机器人下指令："去厨房拿杯水，然后到客厅的沙发旁边停下"：1. Long-horizon; 2. Multi-stage: 3个subtask（找厨房→拿水→找沙发），每个都要判断"我到了没"; 3. Semantic-spatial gap: 语言说"厨房"，视觉看到的是"有个冰箱和灶台的空间"，这俩要对上. 4. Real-time constraint: 机器人不能想5秒才走一步，要实时响应. 5. 生成视觉CoT (CoT-VLA): 模型每步都要"脑补"未来画面再决定动作，但一张图就是几千tokens，5步就3k-5k tokens.

* 用VAR把"想象"压到30个token: VAR (Visual AutoRegressive, Tian et al. NeurIPS 2024) 的核心idea是 next-scale prediction 而非 next-token prediction。传统VAE/VQ-VAE怎么压缩图像？把256×256的图编码成64×64的latent grid，压缩比1/64。VAR不一样。它把图像看成 multi-scale的层级结构：

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

* 模型同时训练4种"思考模式": 用两个binary gate $g_\mathcal{T}, g_\mathcal{V} \in \{0,1\}$ 控制：

| $g_\mathcal{T}$ | $g_\mathcal{V}$ | 模式 | 模型做什么 |
|---|---|---|---|
| 0 | 0 | **Non-CoT** | 直接看图→输出action |
| 1 | 0 | **T-CoT** | 先输出文字推理→再输出action |
| 0 | 1 | **V-CoT** | 先输出VAR latent（想象画面）→再输出action |
| 1 | 1 | **MM-CoT** | 同时输出文字+VAR latent→再输出action |

训练时每个batch随机sample一个模式，模型共享所有参数，只是输入前面加不同的special token告诉它"这次用哪种方式思考"。

4种模式本质是在用不同视角学习同一个"导航policy"：
- Non-CoT是"直觉反射"
- T-CoT是"语言规划"
- V-CoT是"视觉预演"
- MM-CoT是"完整的人类式思考"

让一个模型同时掌握这4种，相当于multi-task learning——每种模式都在强化对"navigation problem"本身的理解，互补。

* Cross-Mode Alignment——让4种模式不打架: 如果让一个模型同时学4种思考方式，会发生什么？Mode conflict。文字模式说"该左转"，视觉模式说"该直走"，non-CoT模式又说"该右转"——模型无所适从。

| 有Alignment | SR   |
| ---------- | ---- |
| 没有         | 0    |
| 有          | 2.44 |

没有alignment，SR直接是0！模型完全学不到consistent policy。

用non-CoT模式当"锚点"，强制其他3种CoT模式的action prediction向non-CoT的prediction对齐。

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

它证明了embodied agent可以在latent space里"想象"，并把这想象能力内化为直觉

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