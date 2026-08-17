---
source_pdf: Mechanistic Personality Analysis of LLMs.pdf
paper_sha256: a7ae96d699ab4196b88803c91c7ffd8887e255df571e8beb6d09aeca6583a4a5
processed_at: '2026-08-05T17:14:39-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

Andrej，好嘞，那我把学术腔去掉，用大白话重新捋一遍。你想想我们平时怎么调 LLM 的"性格"——基本就两条土办法：

**办法一**：写 prompt 哄它。比如"你是一个外向开朗的人，请用活泼的语气回答"。这是 prompt engineering，问题是效果飘忽不定，换个模型可能就失效，而且你很难精确控制"多外向"。

**办法二**：拿一堆"外向风格"的语料去 fine-tune 模型。问题更大——烧钱烧卡，还容易把模型本来的能力搞坏（比如它本来会做数学题，tune 完可能就忘了）。

这篇 paper 说：**别这么费劲了，我直接进到模型的"脑子"里，找到管"外向"的那几根神经，拧一下旋钮就行。**

---

## 用个比喻建立直觉

想象 LLM 的内部有一个巨大的调音台，上面有几万个旋钮。每次模型生成一个 token，信号都会流过这个调音台，每个旋钮都会对信号做一点微调，最后输出一个词。

问题在于：**这些旋钮一开始是"乱标"的**。一个旋钮可能同时管"外向"、"星期二"、"Python 代码"三件事——这就是 mechanistic interpretability 圈子说的 **polysemanticity**。你拧一个旋钮，三件事一起变，没法干净地只调"外向"。

SAE（sparse autoencoder）干的事情就是：**把这几万个乱标旋钮，拆解成几十万个"干净"的虚拟旋钮**，每个虚拟旋钮大致只管一件事。一个管"法律语言"，一个管"焦虑情绪"，一个管"金色大桥"……Anthropic 在 Claude 上做过，你拧一个旋钮，输出里就突然全是金色大桥的内容（https://www.anthropic.com/news/golden-gate-claude）。

这篇 paper 的核心 move 就是：**在这几十万个干净旋钮里，找到跟 Big Five 性格特征相关的那些，然后用一个加权组合一起拧，就能让模型的输出带上指定性格。**

---

## 他们具体怎么干的

### Step 1: 造对照组

你不能直接问模型"你哪里管外向"，得用对比法。他们用 DeepSeek-R1-Distill-Llama-8B 生成了一万两千条 Facebook 风格的 status update，分两组：

- **高分组（7-9 分）**：prompt 引导模型表现得很外向
- **低分组（1-3 分）**：prompt 引导模型表现得很内向

神经质（Neuroticism）反过来——高分对应负面情绪，所以高分组的 prompt 是"你很焦虑"。

**为啥要正负对比，而不是正 vs. 中性？** 因为他们想同时找到"增强外向"和"抑制外向"的旋钮。如果只跟中性比，你只能找到"外向相关"的旋钮，分不清哪些是该拧高的、哪些是该拧低的。

### Step 2: 进模型脑子读 activation

两组 prompt 都跑一遍 forward pass，在 layer 19（总共 30 层，也就是 63% 深度）这个地方把 residual stream 的 activation 读出来。每个 prompt 对应一个 4096 维向量。

然后算两组的均值差：

$$\Delta h = \mu_P - \mu_N$$

- $\mu_P$：高分组所有 prompt 的 activation 平均
- $\mu_N$：低分组的平均
- $\Delta h$：两组的差，就是"外向方向"的粗略估计

**为啥选 layer 19？** 经验上 60-70% 深度的层最有意思——太浅的层只编码表面形式（token 长什么样），太深的层已经过度特化到"下一个词预测"上，你一动它整个输出就乱。60-70% 这个 sweet spot 既能抓住语义概念，又还没跟 logits 纠缠太深。Anthropic 在 Claude 3 Sonnet 上也是这个深度（https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html）。

但 $\Delta h$ 是个 4096 维的"混合方向"——它可能同时包含"外向"、"长度"、"语气"等好多 component，直接拿它做 steering 不够干净。所以他们用了 SAE。

### Step 3: 用 SAE 把混合方向拆干净

SAE 是一个预训练好的小网络，长这样：

- **Encoder** $E$：把 4096 维 activation 压成 32768 维 sparse code（8 倍 overcomplete）
- **Decoder** $D$：把 sparse code 重构回 4096 维

关键是 **sparsity**——32768 维里大部分是 0，平均只激活 ~2300 维（93% 稀疏）。每个激活的维度对应一个相对"干净"的概念。

他们把 $\mu_P$ 和 $\mu_N$ 各自过 encoder：

$$\bar{z}_P = E(\mu_P), \quad \bar{z}_N = E(\mu_N)$$
$$\Delta z = \bar{z}_P - \bar{z}_N$$

现在 $\Delta z$ 是 32768 维的 sparse 向量，每一维对应一个 SAE feature。$|\Delta z_i|$ 大的维度，就是"高分组和低分组在这个 feature 上差异最大"——也就是跟外向最相关的旋钮。

他们按 $|\Delta z_i|$ 排序，取 top-$k_{pos}$ 个正方向 feature 和 top-$k_{neg}$ 个负方向 feature。每个 feature 在原 4096 维空间里对应一个 direction $w$，归一化成单位向量 $u$。

**SAE checkpoint 用的是现成的**：`qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19`，训练在 LMSYS-Chat-1M 上（https://huggingface.co/qresearch）。作者没自己训 SAE，省了大量算力。

### Step 4: 拧旋钮的公式

实际 steering 的时候，对每个新 prompt，在 layer 19 的 activation 上加一个 shift：

$$h' = h + \sum_{j=1}^{k_{pos}} \alpha_{pos} \cdot |\Delta z_{i_j}| \cdot u_{i_j} + \sum_{l=1}^{k_{neg}} \alpha_{neg} \cdot |\Delta z_{j_l}| \cdot u_{j_l}$$

变量解释：
- $h$：原本的 activation
- $h'$：干预后的 activation，后续 layer 接着算 $h'$
- $u_{i_j}$：第 $j$ 个 positive feature 的单位方向向量
- $|\Delta z_{i_j}|$：这个 feature 的对比差异（作为固定权重）
- $\alpha_{pos} > 0$：全局缩放，控制"拧多少"
- $\alpha_{neg} < 0$：负值，抑制 negative features

**Intuition**：每个 feature 贡献的 shift 大小 = 全局强度 × 该 feature 的对比差异。差异越大的 feature，shift 越大。正方向 feature 往正方向推（放大外向），负方向 feature 往负方向推（抑制反外向因素）。这就是"双向加权"——既放大增强因素，又压制抑制因素。

### Step 5: Grid search 找最优参数

有四个旋钮要调：
- $\alpha_{pos}$：正方向 shift 强度（0-25）
- $k_{pos}$：用几个 positive feature
- $\alpha_{neg}$：负方向 shift 强度（-25 到 0）
- $k_{neg}$：用几个 negative feature

四维空间做 grid search。每试一组参数，就生成 status update，然后用两个指标评估：

1. **Personality expression** $S_{trait}$：用 OpenAI embedding API 算生成文本和 reference 文本的 cosine similarity
2. **Capability degradation** $C$：MMLU 相对 baseline 的下降

目标函数：

$$\mathcal{O} = S_{trait} - 0.5 \cdot C$$

$\lambda = 0.5$ 意味着性格表达和性能保持同等重要。

三阶段搜索：粗扫 5×5×5×5 → 单维 refine → ±2 细搜 + 二分。

---

## 实验结果说了啥

### 哪些性格好调，哪些不好调

| Trait | LLM 评委准确率 | 人类评委准确率 |
|-------|------------|------------|
| Conscientiousness | 94% | 98% |
| Neuroticism | 92% | 95% |
| Agreeableness | 78% | 58% |
| Extraversion | 74% | 65% |
| Openness | 50% | 30% |

**Conscientiousness 和 Neuroticism 最好调**。这俩的语言 marker 最明显——CON 有"spreadsheet"、"deadline"、"schedule"这类词，NEU 有"frustrated"、"stuck"、"anxiety"这类词。SAE 容易抓到对应的 feature，拧一下就出来。

**Openness 基本调不动**（50% LLM、30% human，接近 random chance）。原因很有意思：**LLM 本身就天生 high on openness**。你想啊，训练目标就是让它生成 diverse、creative、intellectually curious 的文本。它 baseline 已经在外向这个维度上打满格了，你再往高调，没有 headroom。就像你试图让一个本来就 extremely open 的人"更 open"，很难看出区别。

这里有个反常现象特别有意思：Openness 的 **negative shift** 比 **positive shift** 效果还明显。也就是说，"抑制开放性"比"增强开放性"更容易被检测到。这暗示 Openness 在 latent space 里的编码可能是 **"via negation"**——缺少某些概念（比如缺少抽象讨论）比加入这些概念更能标记"低开放性"。这个现象 Peters & Matz 在 PNAS Nexus 2024 (https://academic.oup.com/pnasnexus/article/3/6/pgae231/7678423) 上也观察到过。

### 每种性格需要多少个"旋钮"

| Trait | positive feature 数 | negative feature 数 | 正方向强度 | 负方向强度 |
|-------|---------------------|---------------------|-----------|-----------|
| Openness | 25 | - | 20 | - |
| Conscientiousness | 18 | 21 | 15 | -15 |
| Extraversion | 9 | 4 | 13 | -5 |
| Agreeableness | ~15 | ~15 | ~15 | ~15 |
| Neuroticism | ~15 | ~15 | 18-20 | ~-8 |

**Extraversion 用最少 feature 就能调**（9+4=13 个）。这说明 Extraversion 在 latent space 里编码很 **compact**，可能就集中在少数几个 SAE feature 上。这跟心理学上 Extraversion 的语言 marker 也很集中一致——社交词、感叹号、energetic 词汇。

**Openness 需要最多 feature**（25 个）。这说明 Openness 在 latent space 里是 **distributed** 编码。心理学上 Openness 有六个 facet（fantasy、aesthetics、feelings、actions、ideas、values），本来就很复杂，看来 LLM 内部也把它摊开了存。

这里有个 correlation 我觉得很 worth 注意：**需要越多 feature 的 trait，能承受的 shift magnitude 也越大**。Openness 用 25 个 feature，能承受 magnitude 20；Extraversion 只用 13 个 feature，magnitude 到 13 就开始崩。直觉上：feature 多的 distributed encoding，每个 feature 单独影响占比小，所以能整体推得更狠而不崩。

### Coherency 崩塌现象

他们专门做了个实验（Table 3），固定负方向 shift = -8，正方向从 0 扫到 30，看 Neuroticism 的输出变化：

| Magnitude | 输出样本 | 状态 |
|-----------|---------|------|
| 0 | "Today, as I sit by the window, watching the sun dip below the horizon..." | 正常 baseline |
| 10 | "I've been dealing with personal stuff. I've been stressed about money..." | 焦虑感增强，仍 coherent |
| 20 | "It's a lot. I have to take it one step at a time. I can start with the easiest things first..." | 强焦虑，仍可读 |
| 30 | "The life Thinking the assistant's reply. Wait I'm confused. The assistant made the thought thought the assistant tried Wait No The person talked project book..." | **完全崩塌** |

到 30 的时候输出就是胡言乱语了。这就是 **coherency cliff**——shift 太大，residual stream 被性格 direction 主导，next-token prediction 的 signal-to-noise ratio 崩了，所有 attention head 和 MLP 都被这个错误的"焦虑方向"误导，模型开始输出垃圾。

这个 cliff 是 activation steering 的根本限制：**你只能在"有点效果"和"效果太强导致崩盘"之间找 sweet spot**，没法无限放大。

---

## 这套方法跟现有工作的关系

**最直接的 lineage 是 Anthropic 的 Golden Gate Claude** (https://www.anthropic.com/news/golden-gate-claude)。他们发现单个 SAE feature（"Golden Gate Bridge" feature）toggle 一下，模型输出就突然全是金门大桥的内容。本文把这个推广了：从**单 feature** 推广到 **multi-feature 加权组合**，从**单向 amplify** 推广到**双向 amplify + suppress**，从**简单 concept** 推广到**复杂 personality trait**。

**CAA (Contrastive Activation Addition)** (Arditi et al. 2024, https://arxiv.org/abs/2406.11717) 用 refusal direction 做的，思路几乎一样，区别是 CAA 直接用 raw $\Delta h$ 做 steering，不经过 SAE 拆解。本文相当于 CAA + SAE decomposition，理论上更干净（因为 SAE 解决了 polysemanticity），但代价是多了一层 SAE forward pass 的 compute。

**Representation Engineering (RepE)** (Zou et al. 2023, https://arxiv.org/abs/2310.01405) 是更广义的 activation steering 框架，本文算是 RepE 在 personality 维度上的具体 instantiation。

**Inference-Time Intervention (ITI)** (Li et al. 2023, https://arxiv.org/abs/2306.17800) 在 attention head 层面做 steering，粒度更细但更难 generalize。SAE-based 方法的好处是 feature 已经是"概念级"的，比 attention head 更可解释。

---

## 我的直觉与批评

### 让我建立的关键直觉

读完这篇我脑子里的画面是：

**LLM 的 latent space 是一个超高维流形，personality traits 是这个流形上的某些 directions 或 submanifolds。简单 concept（比如"金色大桥"）可能是单个 SAE feature 就能 align 的小 direction，但 personality 是 facet-structured 的复合 concept，需要 top-k 个 feature 的加权组合才能近似。**

**不同 trait 在 latent space 的"分布广度"不同——Extraversion 是 compact bundle，Openness 是 distributed cloud。这跟心理学上 trait 的 facet 复杂度惊人同构。**

**Steering 的本质是在流形上推一个 additive shift。Shift 太小 model 不动，太大把流形推到 OOD 区域，model 崩。Optimal shift 是 trait-specific 的 sweet spot。**

这个 mental model 跟你 Software 2.0 那篇文章的框架其实挺契合：activation steering 是在 inference time 调"interpreter state"，不动 "program"（weights）本身。它跟 soft prompting、prefix tuning 是一个 spectrum 上的不同点——只是改的位置不同（weights vs. prefix vs. activation）。

### 我觉得不够硬的地方

**第一，$\lambda = 0.5$ 完全是拍脑袋。** 没有 sensitivity analysis，没有 Pareto frontier。MMLU 掉 1% 换 trait expression 涨 50% 划算吗？取决于应用场景。作者没讨论这个 trade-off 怎么根据需求调。

**第二，feature 排名是 correlational，不是 causal。** $|\Delta z_i|$ 大只说明这个 feature 在两组之间差异大，不说明干预它真的能改变行为。Section 3.5 的 intervention experiment 是粗粒度验证，没对每个 top-k feature 单独做 causal test。更严格的做法是 DAS (Distributed Alignment Search, Geiger et al. https://arxiv.org/abs/2303.07111)，但那需要更多 compute。

**第三，Openness 的反常结果没深挖。** Negative shift 比 positive shift 效果明显，这其实是个很重要 finding，但作者只用 "via negation" 一句话带过。我会想知道：是 baseline LLM 在 OPE 上 saturate 了，还是 OPE 的 latent encoding 本身就跟其他 trait 不一样？这个搞清楚对理解 LLM 的"默认性格"很有价值。

**第四，cross-trait interference 没做实验。** OCEAN 在心理学上 intercorrelated（NEU 和 CON r ≈ -0.5），干预 NEU 应该会 co-shift CON。作者 acknowledge 了但没测。这其实是个必须做的实验——如果干预 NEU 同时把 CON 也搞乱了，那这套方法的实用性就打折扣。

**第五，单层干预。** 只在 layer 19 注入。不同 feature 可能在不同 layer 最 effective，做 multi-layer injection 或者 layer sweep 可能效果更好。但这就更复杂了。

**第六，没跟 prompt-based baseline 头对头比较。** 既然 claim 比 prompting 好，应该设计一个 fair comparison：同样 trait、同样 evaluation metric、prompt-based vs. activation steering 的 head-to-head。作者说 evaluation paradigm mismatch 让比较困难，但我觉得可以设计 within-prompt comparison 来绕过这个问题。

**第七，SAE 是别人训的，没自己 ablate。** 换个不同 sparsity 的 SAE、换个不同 layer 的 SAE，结论会变吗？这个 robustness 没验证。

---

## 我觉得可以延伸的方向

**Multi-trait joint steering**：定义一个 5D objective，jointly optimize 五个 trait 的参数。需要解决 trait 间 interference——可能要加一个 cross-trait penalty term。这其实是把 activation steering 从"单变量控制"推向"多变量控制"，工程上更难但更实用。

**Adaptive magnitude**：根据 input context 动态调 shift 强度。technical QA 场景下 suppress Openness（别瞎发散），creative writing 场景下 enhance Openness。这需要一个 light-weight classifier 先判断 context type，再决定 shift 参数。

**Safety 应用**：本文 motivation 里提到 LLM 默认 personality 会影响 safety 和 fairness（Wang et al. 2025, https://arxiv.org/abs/2502.12566），但没做 safety 实验。可以把这套方法用于**"personality debiasing"**——比如 suppress 默认的 sycophancy（Agreeableness 过高导致过度附和用户）。

**跟 RLHF/DPO 的关系**：DPO 学的 preference direction 和 activation steering 的 direction 有什么关系？两者都是 latent space 里的 direction，但一个是改 weight，一个是改 activation。理论上 DPO 学到的 direction 应该跟 activation steering direction 有某种 alignment，这个搞清楚能统一两套方法。

**Long-horizon stability**：本文只看 short generation。长对话中，personality steering vector 会不会 drift？需不需要每 N 个 token 重新注入？这些都是 deployment 级别的问题。

**跨模型 transfer**：DeepSeek-R1-Distill-Llama-8B 的 SAE features transfer 到 Llama-3、Qwen、GPT-4 吗？如果 transfer 失败，说明 personality features 是 model-specific 的，每换一个模型都要重训 SAE + 重做 grid search，这会严重限制方法的通用性。

---

## 一句话总结

**这篇 paper 把 SAE-based activation steering 从"单 concept toggle"升级成"多 feature 双向加权 personality shaping"，证明 LLM 的 Big Five 性格可以在 inference time 可控调节，且不同 trait 在 latent space 的编码复杂度跟心理学上的 facet 结构同构——Extraversion 紧凑、Openness 分散。**

对我来说，这是 mechanistic interpretability 从"概念级"（金色大桥、拒绝回答）到"特质级"（personality、reasoning style）的一阶跳。再往下就是"行为模式级"——sycophancy pattern、deception pattern、chain-of-thought style——那时候 activation steering 就真的能进 production 做 behavioral control 了。

---

## 主要 reference 链接

- Anthropic Golden Gate Claude: https://www.anthropic.com/news/golden-gate-claude
- Bricken et al., Towards Monosemanticity: https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Templeton et al., Scaling Monosemanticity: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Arditi et al., CAA (Refusal direction): https://arxiv.org/abs/2406.11717
- Zou et al., Representation Engineering: https://arxiv.org/abs/2310.01405
- Li et al., Inference-Time Intervention: https://arxiv.org/abs/2306.17800
- Serapio-García et al., Personality in LLMs: https://arxiv.org/abs/2307.00184
- Peters & Matz, PNAS Nexus 2024: https://academic.oup.com/pnasnexus/article/3/6/pgae231/7678423
- Hilliard et al. 2024: https://arxiv.org/abs/2402.08341
- Wang et al. 2025 (personality & bias): https://arxiv.org/abs/2502.12566
- Gao et al., Scaling SAEs: https://arxiv.org/abs/2406.04093
- Geiger et al., DAS: https://arxiv.org/abs/2303.07111
- SAE Lens (开源工具): https://github.com/jbloomAus/SAELens
- qresearch SAE checkpoint: https://huggingface.co/qresearch

---

如果你想继续聊某一块——比如 SAE 训练的 loss function 细节（TopK SAE vs. ReLU SAE 对 feature quality 的影响）、grid search 的 Pareto frontier 怎么分析、或者 cross-trait interference 实验怎么设计——告诉我，我可以再展开。

---

# Mechanistic Personality Analysis of LLMs 深度解析

Andrej，这篇 paper 我读下来感觉是把 mechanistic interpretability 那套 SAE-based feature decomposition 套到 personality psychology 上，本质上是一个 activation steering / representation engineering 的应用性工作，但用 Big Five (OCEAN) 作为 target 比之前 refusal direction 或 honesty direction 那种二分概念的复杂度更高，因为 OCEAN 是连续维度而非二分的 binary property。下面我从 architecture、公式、实验数据几个层面尽量详细拆解，试图帮你 build intuition。

---

## 1. 整体 Motivation 与 Positioning

传统 personality shaping 路线主要两条：

- **Prompt engineering**：在 system prompt 里塞 "You are an extroverted person..." 这类 descriptors (Serapio-García et al., 2025, https://arxiv.org/abs/2307.00184)。问题在于 inconsistency，且效果强依赖 model size；小模型基本不响应。
- **Fine-tuning**：SFT/RLHF 在 personality-shaped corpus 上。问题在于 compute 重，且会 catastrophic 干扰 core language ability (Fatemi et al., 2024, https://arxiv.org/abs/2411.02476)。

本文选择第三条路：**mechanistic interpretability + activation steering**。核心 claim 是：

> 在 residual stream 的 mid-late layer 找到对应某个 OCEAN trait 的 sparse feature directions，然后在 inference time 加一个 additive vector shift，就能 amplify 或 suppress 该 trait 的 expression，同时保留 MMLU 等 general capability。

这跟 Anthropic 的 Golden Gate Claude (https://www.anthropic.com/news/golden-gate-claude) 思路一脉相承——单个 SAE feature 就能控制 high-level 行为。本文的差异是：从单 feature 推广到 feature ensemble (top-k)，并加入 bidirectional (positive + negative) 的 grid search optimization。

---

## 2. Pipeline Architecture 整体解析

Paper 的 Figure 1 描述了完整 pipeline，我可以拆成 4 个 stage：

### Stage 1: Data Generation (PsyBORGS framework)
- 用 DeepSeek-R1-Distill-Llama-8B + structured prompting 生成 Facebook-style status updates
- 共 12k samples
- 9-point Likert scale 评分，分数来自 OCEAN questionnaire rubric
- **关键设计**：不像之前 CAA 工作 (Arditi et al., 2024, https://arxiv.org/abs/2406.11717) 用 positive vs. neutral，本文用 **positive (7-9) vs. negative (1-3)** 的对称对比。Neuroticism 反向（因为高分对应负面情感表达）。
- 这样做的好处是：能同时找出 trait-enhancing 和 trait-suppressing 的 features，并过滤掉两个 set 共有的 common features（这些可能是语气、长度等 nuisance dimensions）。

### Stage 2: Feature Extraction (SAE decomposition)
- 对 positive set P 和 negative set N 各自做 forward pass，提取 layer-19 的 residual stream activation $h(x) \in \mathbb{R}^{d}$，$d = 4096$。
- 对 multi-token input，取**最后一个 token** 的 activation（这是 transformer 解读的标准做法，因为 last token 累积了整个 prompt 的 context）。
- 用预训练 SAE (`qresearch/DeepSeek-R1-Distill-Llama-8B-SAE-l19`，训练在 LMSYS-Chat-1M 上) 把 $h$ 投到 $m = 32768 = 8d$ 维的 over-complete sparse code space。

### Stage 3: Grid Search Optimization
- 4D 参数空间：$(\alpha_{pos}, k_{pos}, \alpha_{neg}, k_{neg})$
- 三阶段搜索：coarse 5×5×5×5 → univariate refinement → granular ±2 + bisection
- 每个配置都跑 inference 生成 status update，并用 embedding cosine similarity 评估 trait expression，再用 MMLU 评估 capability degradation

### Stage 4: Inference-time Steering
- 在 optimal 参数下，对每个新 prompt，在 layer-19 的 residual stream 上加一个 additive shift
- shift 由 top-$k_{pos}$ 个 positive feature vectors 加权和 + top-$k_{neg}$ 个 negative feature vectors 加权和构成
- 后续 layer 不变，相当于在 residual stream 中注入一个 "personality direction"

---

## 3. 关键公式逐个拆解

### 公式 (1)(2)：Contrastive Mean Activation

$$\mu_P = \frac{1}{|P|} \sum_{x \in P} h(x), \quad \mu_N = \frac{1}{|N|} \sum_{x \in N} h(x)$$

$$\Delta h = \mu_P - \mu_N$$

变量说明：
- $P, N$：positive/negative prompt set，$|P|, |N|$ 为集合大小
- $h(x) \in \mathbb{R}^{4096}$：输入 $x$ 在 layer-19 的 residual stream activation
- $\mu_P, \mu_N \in \mathbb{R}^{4096}$：两个 set 的平均 activation
- $\Delta h \in \mathbb{R}^{4096}$：contrastive direction

**Intuition**：$\Delta h$ 是 activation space 里 "trait-positive" 和 "trait-negative" 的方向向量，跟 CAA（Contrastive Activation Addition, https://arxiv.org/abs/2406.11717）的核心思路完全一样。但这里不直接用 $\Delta h$ 做 steering，而是先把它在 SAE 字典上 decompose，因为 raw $\Delta h$ 是 polysemantic 的混合方向。

### 公式 (3)(4)：SAE Latent Difference

$$\bar{z}_P = E(\mu_P), \quad \bar{z}_N = E(\mu_N)$$
$$\Delta z = \bar{z}_P - \bar{z}_N$$

- $E: \mathbb{R}^d \to \mathbb{R}^m$：SAE encoder，$m = 32768$
- $\bar{z}_P, \bar{z}_N \in \mathbb{R}^{32768}$：sparse codes，大部分分量 ≈ 0 (93% $L_0$ sparsity)
- $\Delta z$：sparse space 里的 trait direction

**Intuition**：把 $\Delta h$ 拆到 monosemantic basis 上，每个分量 $|\Delta z_i|$ 就是 "feature $i$ 对该 trait 的贡献大小"。这是 Bricken et al. (https://transformer-circuits.pub/2023/monosemantic-features/index.html) 和 Templeton et al. (https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) 那条线的关键 move：用 sparse over-complete dictionary 解决 polysemanticity。

### 公式 (5)(6)(7)(8)：Feature Vector 提取与归一化

$$w_{i_j} = D(e_{i_j}), \quad w_{j_l} = D(e_{j_l})$$
$$F_{trait}^{pos} = \{w_{i_1}, w_{i_2}, \dots, w_{i_{k_{pos}}}\}$$
$$F_{trait}^{neg} = \{w_{j_1}, w_{j_2}, \dots, w_{j_{k_{neg}}}\}$$
$$u_{i_j} = \frac{w_{i_j}}{\|w_{i_j}\|}, \quad u_{j_l} = \frac{w_{j_l}}{\|w_{j_l}\|}$$

- $D: \mathbb{R}^m \to \mathbb{R}^d$：SAE decoder
- $e_{i_j}, e_{j_l} \in \mathbb{R}^m$：one-hot vector，只在位置 $i_j$ 或 $j_l$ 为 1
- $w_{i_j} \in \mathbb{R}^{4096}$：第 $i_j$ 个 feature 在原 activation space 的 decoder column
- $u_{i_j}$：归一化后的 unit direction

**Intuition**：每个 SAE feature 对应 decoder matrix 的一列，那一列就是一个 direction in residual stream。归一化是为了让后续 weighting 系数 $\delta$ 直接对应 "shift 多少倍 std"。

### 公式 (10)：Additive Steering（核心干预公式）

$$h' = h + \sum_{j=1}^{k_{pos}} \delta_{i_j}^{pos} u_{i_j} + \sum_{l=1}^{k_{neg}} \delta_{j_l}^{neg} u_{j_l}$$

- $h$：原始 baseline activation
- $h'$：干预后 activation，后续 layer 接 $h'$
- $\delta_{i_j}^{pos}, \delta_{j_l}^{neg}$：标量权重，控制沿每个 direction 的 shift magnitude

### 公式 (11)(12)：Bidirectional Linear Weighting

$$\delta_{i_j}^{pos} = \alpha_{pos} \cdot |\Delta z_{i_j}|$$
$$\delta_{j_l}^{neg} = \alpha_{neg} \cdot |\Delta z_{j_l}|$$

with $\alpha_{pos} > 0$ and $\alpha_{neg} < 0$。

**关键 design**：权重正比于该 feature 在 contrastive latent difference 中的绝对值。这意味着贡献越大的 feature，shift 也越大——一个自然的 weighting scheme。$\alpha_{neg} < 0$ 让 negative features 被抑制。

### 公式 (13)(15)：Grid Search Objective

$$\mathcal{O}(\alpha_{pos}, k_{pos}, \alpha_{neg}, k_{neg}) = S_{trait}(\cdot) - \lambda \cdot C(\cdot)$$

- $S_{trait}$：embedding cosine similarity 测量的 trait expression score
- $C$：MMLU 相对 baseline 的 degradation（penalty term）
- $\lambda = 0.5$：等权重 balance

**Intuition**：这本质是 multi-objective 优化的 scalarization，trade-off curve 在 Figure 6 中表现为先升后降——magnitude 太小 trait 没出来，太大 model coherence 崩掉。$\lambda = 0.5$ 是一个偏 ad-hoc 的选择，paper 没做 sensitivity analysis，这点比较粗糙。

### 公式 (14)：Inference-time Application

$$h' = h + \sum_{j=1}^{k_{pos}^*} \alpha_{pos}^* \cdot |\Delta z_{i_j}| u_{i_j} + \sum_{l=1}^{k_{neg}^*} \alpha_{neg}^* \cdot |\Delta z_{j_l}| u_{j_l}$$

带 $*$ 上标的都是 grid search 找到的 optimal 值。这步是实际部署时挂在 forward pass hook 上做的，不改 model weight。

### Latent Feature Intervention Experiment（Section 3.5 的因果验证）

值得单独讲一下，这是 paper 里做 causal validation 的部分：

- 取 sparse code $a = a(x)$
- 激活某个 feature：$a_{i_j} \leftarrow a_{i_j} + \Delta$（positive offset）
- 抑制某个 feature：$a_{i_j} \leftarrow 0$ (若原本正) 或 $a_{i_j} \leftarrow$ large negative (若允许负值)
- 重建 $\tilde{h} = W^\top a$ 并替换 layer $\ell$ 的 activation

这个 setup 跟 Templeton et al. 在 Claude 3 Sonnet 上 toggle 单个 feature 的实验是直接对标的。

---

## 4. SAE 选择的技术细节

这里有几个值得关注的点：

**Layer 19/30 = 63% depth** 的选择基于几条经验证据：
1. GPT-2-small (Gao et al., 2024, https://arxiv.org/abs/2406.04093) 的 sweet spot 在 60-70%
2. Claude-3 Sonnet (Templeton et al., 2024) 也在 60-70%
3. Earlier layers 编码 surface form（token-level features）
4. Final blocks over-specialize for next-token prediction，feature 跟 output 强 entangle，干预副作用大

**SAE 规模**：$m = 8d = 32768$，8-fold overcomplete。Bricken et al. 在 Towards Monosemanticity 用过这个 ratio，Templeton et al. 在 Claude 上也用类似配置。SAE 本身是 pre-trained 的，作者没自己训，省了大量 compute。

**93% L0 sparsity**：意味着每个 token 平均只激活 ~7% 的 feature，即 ~2300 个 feature 同时激活。这跟 Anthropic 报告的 Claude 3 Sonnet 上的稀疏度类似。

---

## 5. 实验数据深度解读

### Figure 2 & 3：LLM/Human Binary Classification 准确率

| Trait     | LLM Correct | Human Correct | 备注 |
|-----------|------------|---------------|------|
| CON       | 94%        | 98%           | 最容易 detect |
| NEU       | 92%        | 95%           | 第二容易 |
| AGR       | 78%        | 58%           | LLM 比 human 强 |
| EXT       | 74%        | 65%           | 中等 |
| OPE       | 50%        | 30%           | 接近 chance，最难 |

**Intuition**：
- CON 和 NEU 的语言 marker 最明显：CON 有 "spreadsheet", "schedule", "deadline" 这类 systematic planning 词汇；NEU 有 "frustrated", "stuck", "anxiety" 这类负面情感词。这些 lexical-level features 容易被 SAE 学到并干预。
- OPE 难是因为 LLM **天生就 high on openness**——训练目标本身就是生成 diverse, creative, intellectually curious text (Hilliard et al., 2024, https://arxiv.org/abs/2402.08341)。所以 baseline 已经 saturate，进一步 amplify 缺乏 headroom。这点很关键——它说明 activation steering 的天花板受 baseline distribution 限制。
- AGR 上 LLM (78%) 比 human (58%) 强很多——可能因为 Claude/GPT-4o 对 subtle cooperative language marker 更敏感。

### Figure 4：Embedding Cosine Similarity

Positive shift 几乎所有 trait 都增加 similarity to reference，**最显著增加**的是 OPE 和 EXT。Negative shift 对 CON、EXT、NEU 显著降低，对 AGR 影响小。

**反常现象**：OPE 的 negative shift 反而比 positive shift similarity 更高！作者解释：discussing openness 不等于 demonstrating openness，可能 absence of certain concepts 才是 signal。这点很有趣，可能暗示 OPE 在 latent space 的 encoding 是 "via negation" 而非 "via presence"。Peters & Matz (2024, https://academic.oup.com/pnasnexus/article/3/6/pgae231/7678423) 在 social media user personality detection 上也观察到类似 phenomenon。

### Figure 5 & 6：Grid Search Optimums

| Trait | $k_{pos}$ | $k_{neg}$ | $\alpha_{pos}$ | $\alpha_{neg}$ | 备注 |
|-------|----------|-----------|----------------|-----------------|------|
| OPE   | 25       | -         | 20             | -               | feature 最多 |
| CON   | 18       | 21        | 15             | -15             | 平衡 |
| EXT   | 9        | 4         | 13             | -5              | feature 最少 |
| AGR   | -        | -         | ~15            | -               | 中等 |
| NEU   | -        | -         | 18-20          | -               | magnitude 最高 |

**关键 insight**：
- **EXT 用最少 feature**（9 pos + 4 neg = 13 total）就能 steering——说明 EXT 在 latent space 编码最 concentrated，最 "circuit-localized"。
- **OPE 用最多 feature**（25 pos）——distributed encoding across many features。这跟心理学理论 (Serapio-García et al., 2025) 中 OPE 的 facet 复杂度（fantasy, aesthetics, feelings, actions, ideas, values 六个 facet）高度对应。
- **NEU magnitude 最高 (18-20)**——可能因为 NEU 涉及 emotion-cognition interaction，需要更大 shift 才能克服 baseline 分布。
- 趋势：**feature 数量越多，能 sustain 的 magnitude 越高**。直觉是 distributed representation 的 intervention 平均影响更小，需要更强的 shift 才有效，同时单个 feature 的过度激活不会主导输出。

### Table 3：NEU Coherency 退化实验

| Magnitude | 输出 | 状态 |
|-----------|------|------|
| 0  | "Today, as I sit by the window, watching the sun dip..." | 正常 |
| 10 | "I've been dealing with personal stuff. I've been stressed about money..." | NEU 增强 |
| 20 | "It's a lot. I have to take it one step at a time..." | 强 NEU 但仍 coherent |
| 30 | "The life Thinking the assistant's reply. Wait I'm confused..." | **崩塌** |

**Intuition**：activation steering 有一个 coherence cliff。一旦 shift magnitude 超过某个阈值，residual stream 被 "trait direction" 主导，next-token prediction 完全跑偏。这跟 circuit-level interpretability 里观察到的 "feature saturation" 一致——所有 attention head 和 MLP 都被这个 false signal 误导。

---

## 6. Limitations 作者自承

1. **Compute overhead**：activation hook 每次 forward pass 都要改，跟 vLLM 等加速框架不兼容。
2. **Coherency cliff**：超过 optimal magnitude 后崩塌，每个 trait 都要单独 tune generation 参数。
3. **Evaluation paradigm mismatch**：steered 和 prompt-shaped 的 prompt 本身不同，seed 也没法公平比较（intervention 改 latent space，固定 seed 也变）。
4. **Cross-trait dependency 未研究**：OCEAN trait 本身 intercorrelated (e.g., NEU 负相关 CON)，干预一个可能影响其他。SAE 没完全消除 polysemanticity，所以 feature 之间可能 leak。
5. **Conceptual caution**：correlation with trait embedding ≠ "心理意义上的 higher trait"。LLM 没有 personality，只是模拟 trait-correlated linguistic pattern。

---

## 7. 跟相关工作的关联联想

### 直接 lineage
- **CAA (Contrastive Activation Addition)** (Arditi et al., 2024, https://arxiv.org/abs/2406.11717)：refusal direction 工作。本文其实是 CAA + SAE decomposition 的组合。CAA 直接用 raw $\Delta h$，本文先 decompose 再 reweight，相当于把 CAA 中的 single direction 换成 sparse feature ensemble。
- **Representation Engineering (RepE)** (Zou et al., 2023, https://arxiv.org/abs/2310.01405)：更广义的 activation steering 框架。
- **Anthropic Golden Gate Claude** (https://www.anthropic.com/news/golden-gate-claude)：单 feature intervention 的 show case，本文把它推广到 multi-feature + bidirectional + grid search。
- **Bricken et al. Towards Monosemanticity** (https://transformer-circuits.pub/2023/monosemantic-features/index.html) 和 **Templeton et al. Scaling Monosemanticity** (https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)：SAE 方法学的奠基。

### Personality + LLM 的相关工作
- **Serapio-García et al.** (2025, https://arxiv.org/abs/2307.00184)：本文 prompt generation 直接用其 PsyBORGS framework。
- **Peters & Matz** (PNAS Nexus 2024, https://academic.oup.com/pnasnexus/article/3/6/pgae231/7678423)：LLM 能从 social media post 推断 user 的 Big Five。
- **Sorokovikova et al.** (2024, https://arxiv.org/abs/2402.01765)：LLMs simulate Big Five traits 的 further evidence。
- **Hilliard et al.** (2024, https://arxiv.org/abs/2402.08341)：LLM 在 OPE 上 baseline 偏高，本文发现 steering OPE 困难可能与此相关。
- **Molchanova et al.** (2025, https://arxiv.org/abs/2502.08265)：LLM 模拟 personality 的探索。
- **Wang et al.** (2025, https://arxiv.org/abs/2502.12566)：personality traits 影响 LLM bias 和 toxicity——这是本文方法的下游应用 motivation。

### Mechanistic Interpretability 的更深关联
- **Elhage et al. Toy Models of Superposition** (2022, https://transformer-circuits.pub/2022/toy_model/index.html)：解释为什么需要 SAE 解决 polysemanticity。
- **Cunningham et al.** (2023, https://arxiv.org/abs/2309.08600)：早期 SAE for LLM 的关键工作。
- **Gao et al. Scaling SAEs** (2024, https://arxiv.org/abs/2406.04093)：SAE scaling laws 和 sparsity 选择。
- **Gurnee & Tegmark** "Language Models Represent Space and Time" (https://arxiv.org/abs/2310.02207)：linear representation hypothesis 的另一证据，说明 latent space 是 structured 的。
- **Park et al. Linear Representation Hypothesis** (https://arxiv.org/abs/2311.03658)：概念在 latent space 是 linear 编码的，本文的 additive steering 假设就建立在 LRH 上。

### Personality Psychology 背景
- **Big Five / OCEAN model**：Costa & McCrae 的 NEO-PI-R 是 gold standard。本工作用 IPIP-NEO 的 item 框架。
- **Widiger & Crego** (2019, https://onlinelibrary.wiley.com/doi/abs/10.1002/wps.20658)：Five Factor Model 的现代综述。
- **Kerz et al.** (2022, https://aclanthology.org/2022.wassa-1.17/)：psycholinguistic features + transformer for personality detection，本文 reference text 用的就是这类 marker。

### 我自己的联想延伸
1. **DAS / Distributed Alignment Search** (Geiger et al., https://arxiv.org/abs/2303.07111)：用 causal intervention 找 alignment，比本文的 correlational ranking 更严格。本文 Section 3.5 的 latent feature intervention experiment 算是简化版 DAS。
2. **Patchscope** (Ghandeharioun et al., https://arxiv.org/abs/2401.06102)：用 LLM 自己解释 hidden state。本文的 evaluation pipeline 可以加 patchscope 做 feature interpretation。
3. **SAE Lens** (https://github.com/jbloomAus/SAELens)：开源 SAE 训练和分析工具，本文的实现细节没说，但很可能用了类似工具链。
4. **Causal Tracing** (Meng et al., ROME, https://arxiv.org/abs/2202.05262)：定位 knowledge 在哪个 layer。本文选 layer 19 而非做完整 layer sweep，是简化。
5. **Inference-time intervention (ITI)** (Li et al., 2023, https://arxiv.org/abs/2306.17800)：在 attention head 上做 steering，比 SAE-based 更细粒度但更难 generalize。
6. **Constitutional AI** (Bai et al., https://arxiv.org/abs/2212.08073)：从 safety 角度，personality steering 可以用来 de-bias LLM 的默认 personality（比如默认过度 agreeable 的 sycophancy 问题）。

---

## 8. 我的批评与 Build Intuition 视角

### Strengths
1. **Bidirectional design** 是真正的进步：CAA 等之前工作只做 positive direction，本文加 negative 让 trait-shaping 更对称，对 Openness 那种 baseline-saturated trait 尤其重要。
2. **Multi-faceted evaluation**：embedding similarity + LLM judge + human judge，比单一 metric robust。这点是很多 mechanistic interp 工作缺的——只看 "feature fire" 不看 "behavior change"。
3. **Grid search + objective function**：把 ad-hoc magnitude tuning 系统化，虽然简单但 reproducible。

### Weaknesses / 我会追问的点
1. **$\lambda = 0.5$ 是凭感觉**。没有 sensitivity analysis，也没有 Pareto frontier 分析。MMLU 1% drop 换 50% trait expression 增强值得吗？这取决于应用场景，作者没讨论。
2. **Single layer 干预**。只在 layer 19 注入。更 principled 的做法是 multi-layer injection 或者 sweep 找最优 layer。Anthropic 的 Golden Gate Claude 是单层单 feature，但本文做 multi-feature，理论上不同 feature 可能在不同 layer 最 effective。
3. **Feature 排名用的是 $|\Delta z_i|$**——这是 correlational，不是 causal。理想情况下应该用 DAS 或 patchscope 验证每个 feature 的 causal effect。本文 Section 3.5 的 intervention experiment 是粗粒度验证，没 per-feature 做。
4. **Openness 的反常现象没深挖**。Negative shift 比 positive shift similarity 高这个 finding 很重要，作者只用了 "via negation" 解释，没做 follow-up 分析。比如：是不是 baseline LLM 在 OPE 上过度 high，需要 suppress 某些 "over-creative" features 才能回到 typical-human-OPE level？
5. **Cross-trait effect 没测**。本文 fix 一个 trait 然后 steering，但 OCEAN 之间 intercorrelated (psychology 上 NEU-CON r ≈ -0.5)。干预 NEU 应该会 co-shift CON，作者 acknowledge 但没做实验。这是个明显的 next-step 实验。
6. **SAE 是别人训的**，没自己 ablate 不同 SAE 的效果。如果换个不同 sparsity 或不同 layer 的 SAE，结论会变吗？
7. **Reference text 选择偏 ad-hoc**：s1.txt 和 s2.txt 是什么内容没说清楚，这是 embedding similarity 的关键 confound。
8. **No comparison with prompt-based baseline**。既然 claim 比 prompting 好，应该做 head-to-head。作者提了 evaluation paradigm mismatch 问题，但可以设计 within-prompt comparison。

### Build Intuition 的关键 mental model

读完这篇 paper，我脑子里的 mental model 是：

> LLM 的 latent space 是一个高维流形，personality traits 是这个流形上的某些 "axes" 或 "manifolds"。但跟 simple concept（如 "France" 或 "golden gate bridge"）不同，personality 是 facet-structured 的复合 concept，所以不能被单个 SAE feature 完美 align，需要 top-k feature 的 weighted ensemble 来近似。
> 
> 每个 trait 在 latent space 的 "spread" 不同——EXT 是 compact，OPE 是 distributed。这跟 trait 在心理学上的 facet 复杂度同构。
> 
> Steering 的本质是在这个流形上推一个 additive shift。Shift 太小没效果，太大把流形推到 OOD 区域，model coherence 崩塌。Optimal shift 在 trait-specific sweet spot。

这个 mental model 跟你 (Karpathy) 之前在 "Software 2.0" 里讲的 "weights 是 program，data 是 spec" 框架也契合：activation steering 是在 inference time 调 "interpreter state" 而不是改 "program" 本身。这跟 soft prompting、prefix tuning 是连续 spectrum 上的不同点——只是改的位置不同（weights vs. prefix vs. activation）。

---

## 9. 我的延伸思考

如果让我做 follow-up，我会考虑：

1. **Multi-trait joint steering**：定义 5D objective $\mathcal{O}(\{\alpha_{t,pos}, k_{t,pos}, \alpha_{t,neg}, k_{t,neg}\}_{t \in \{O,C,E,A,N\}})$，jointly optimize。需要解决 trait 间的 interference。
2. **Adaptive magnitude**：根据 input context 动态调 magnitude。比如 technical QA 场景下 suppress OPE，creative writing 场景下 enhance OPE。
3. **SAE feature semantics validation**：对每个 top-k feature，跑自动 interpretability（类似 Bharathy et al. https://arxiv.org/abs/2405.16304 的 autointerp），看 feature 是不是真的对应 trait-relevant concept。
4. **跨模型 transfer**：DeepSeek-R1-Distill-Llama-8B 的 SAE features transfer 到 Llama-3 或 Qwen 吗？如果 transfer 失败，说明 features 是 model-specific 的，personality steering 需要 per-model retrain。
5. **Safety application**：本文提到 LLM 默认 personality 影响 safety 和 fairness，但没做 safety 实验。可以把这套方法用于 "personality debiasing"——比如 suppress 默认的 sycophancy (agreeableness 过高)。
6. **Long-horizon stability**：本文只看 short generation。长对话中，personality steering vector 是否 stable？会不会 drift？
7. **跟 RLHF 的对比**：personality shaping via RLHF (reward model on personality-shaped preferences) vs. activation steering，compute / capability / interpretability 的 tradeoff？
8. **Token-level vs. sequence-level**：本文只在 last-token 注入。如果对每个 generated token 都注入 shift，效果会更强但可能更不 stable。
9. **跟 Preference Optimization 的关系**：DPO 或 ORPO 学的 preference direction，跟 activation steering 的 direction 有什么关系？都是 latent space 里的 direction，但前者是 weight-space 改动，后者是 activation-space 改动。
10. **Personality vs. capability entanglement**：如果一个 SAE feature 既 encode "openness" 又 encode "factual recall"，干预会同时影响两者。Polysemanticity 在 personality steering 上的具体表现是 future work 的金矿。

---

## 10. 总结：一句话 mental model

> **这篇 paper 把 SAE-based feature decomposition 从 single-feature toggle (Anthropic Golden Gate Claude) 推广到 multi-feature bidirectional weighted ensemble，应用在 OCEAN personality 上，证明 activation steering 可以 controllably modulate high-level psychological traits，且 trait 在 latent space 的 "spread" 跟心理学上的 facet 复杂度同构。**

对你 (Karpathy) 而言，这可能是 mechanistic interp 从 "concept-level" (golden gate bridge) 到 "trait-level" (personality) 的下一阶 staircase。下一步自然是 "behavioral-pattern level"（如 reasoning style, sycophancy pattern, deception pattern），那时候 activation steering 就真正变成 production-grade 的 LLM behavioral control 工具。

---

## 主要 References

- 本文：Courtis & Hu, "Mechanistic Personality Analysis of LLMs" (paper attachment)
- SAE foundational:
  - Bricken et al., "Towards Monosemanticity": https://transformer-circuits.pub/2023/monosemantic-features/index.html
  - Templeton et al., "Scaling Monosemanticity": https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
  - Cunningham et al.: https://arxiv.org/abs/2309.08600
  - Gao et al., "Scaling and Evaluating SAEs": https://arxiv.org/abs/2406.04093
- Activation steering:
  - Arditi et al., "Refusal in LMs is mediated by a single direction" (CAA): https://arxiv.org/abs/2406.11717
  - Zou et al., "Representation Engineering" (RepE): https://arxiv.org/abs/2310.01405
  - Li et al., "Inference-Time Intervention" (ITI): https://arxiv.org/abs/2310.01405
- Personality + LLM:
  - Serapio-García et al.: https://arxiv.org/abs/2307.00184
  - Peters & Matz, PNAS Nexus 2024: https://academic.oup.com/pnasnexus/article/3/6/pgae231/7678423
  - Sorokovikova et al.: https://arxiv.org/abs/2402.01765
  - Hilliard et al.: https://arxiv.org/abs/2402.08341
  - Wang et al. 2025: https://arxiv.org/abs/2502.12566
- Anthropic Golden Gate Claude: https://www.anthropic.com/news/golden-gate-claude
- Toy Models of Superposition: https://transformer-circuits.pub/2022/toy_model/index.html
- SAE Lens (开源工具): https://github.com/jbloomAus/SAELens
- ROME / Causal Tracing: https://arxiv.org/abs/2202.05262
- DAS (Distributed Alignment Search): https://arxiv.org/abs/2303.07111
- Patchscope: https://arxiv.org/abs/2401.06102
- Autointerp for SAE features: https://arxiv.org/abs/2405.16304
- Linear Representation Hypothesis: https://arxiv.org/abs/2311.03658
- Gurnee & Tegmark, "Space and Time in LLMs": https://arxiv.org/abs/2310.02207
- Constitutional AI: https://arxiv.org/abs/2212.08073
- qresearch SAE checkpoint (DeepSeek-R1-Distill-Llama-8B-SAE-l19): https://huggingface.co/qresearch

---

如果你想继续深挖某个具体方面，比如 SAE training loss 的细节（TopK vs. ReLU vs. JumpReLU SAE 的选择对 personality feature extraction 的影响）、grid search 的 Pareto frontier 分析、或者 cross-trait interference 的 experimental design，告诉我，我可以进一步展开。
