---
source_pdf: ScalingVerificationCanBeMoreEffectivethanScalingPolicyLearning forVision-Language-ActionAlignment.pdf
paper_sha256: 591a67475b1cb2776d5f8f94d98eb3a91c0a4a44f7b44650171a6045e2e62e8a
processed_at: '2026-08-12T03:47:49-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CoVer-VLA 人话版

Andrej，我把这篇 paper 用大白话再过一遍，重点讲 intuition。

---

## 这篇 paper 在说什么？

一句话：**与其花大钱训练一个更强的 robot policy，不如花小钱训练一个"裁判"，让裁判在执行时帮你挑最好的动作。**

类比一下：你有一个员工（VLA policy），他大部分时候活干得还行，但偶尔会理解错你的意思。传统做法是给他报更多培训班（data augmentation），让他见过更多说法。但这很贵，效果还一般。

这篇 paper 说：别折腾员工了，给他配一个助理（verifier）。你下达指令后，助理先把你这句话改写成 8 种说法，让员工针对每种说法各出 5 个方案，然后助理挑一个最靠谱的执行。

---

## 为什么员工会理解错？

VLA 模型的本质是一个 conditional generator：给场景图 $o_t$ 和语言指令 $l$，输出动作 $a \sim \pi(a | o_t, l)$。

问题出在：同一个意思可以有无数种说法。比如 "把红牛罐放到盘子上"，你可以写成：
- "place the energy drink on the plate"
- "put the blue can on the yellow plate"
- "strategically position the energy can on the wide platter"

人听起来都一样，但 VLA policy 对每种说法生成的 action distribution 可能差很远。Paper 里管这个叫 **"intention-action gap"**——意图一样，动作却跑偏了。

更糟的是，VLA 在 fine-tuning 成 action generator 的过程中，VLM backbone 的语言理解能力会被部分破坏（catastrophic forgetting），所以越 train 越 dumb。

---

## 核心发现：改写指令比重复采样更有效

这是 paper 最有意思的 empirical 结果。

假设你有 budget 采样 $k$ 个 action candidates，四种花法：

**花法 1：重复采样**
同一个指令，从 $\pi(a|s,l)$ 里反复 sample $k$ 次。就像让员工把同一道题做 $k$ 遍——他大概率每次都犯类似的错。

**花法 2：Gaussian 扰动**
采一小撮样本，拟个 Gaussian，再从 Gaussian 里抽 $k$ 个。相当于在员工答案上加 noise——还是在同一个 mode 附近晃。

**花法 3：指令改写**
用 VLM 把指令改写成 $k$ 种说法，每个说法采一个 action。相当于让员工从 $k$ 个不同角度理解任务——可能会进入完全不同的 mode。

**花法 4：混合**
每个改写再 fan out 多个采样。最好的方案。

Paper 拟合了 power law：$\log(e) \approx \log(a) + b \cdot \log(k)$
- $e$ 是 action error
- $k$ 是采样数
- $b$ 是斜率，越负越好

结果：**指令改写比重复采样下降快得多**。

Intuition：VLA 是高度 conditional 的模型，改 $l$ 会让 $\pi(\cdot|s,l)$ 进入完全不同的 mode。而同一 $l$ 下的 temperature sampling 只在 mode 附近晃。这跟你 Karpathy 讲 LLM 时说的"temperature sampling 在 mode 附近，CoT/prompt 变化会跳到不同 mode"是一个道理。

参考：[Scaling LLM Test-Time Compute (Snell et al.)](https://arxiv.org/abs/2408.03314)

---

## Verifier 怎么训练？Contrastive Learning 的妙处

最大 challenge：robotic dataset 里只有成功演示，没有失败标签。你想训一个 verifier 来区分"好动作"vs"坏动作"，但没有"坏动作"的样本。

**传统思路的两条死路：**
- 合成假坏动作 → 通常不 realistic，训出来的 verifier 学不到有用的东西
- 人工标注失败 → 太贵

**Contrastive learning 的聪明解法：**
一个 batch 里塞 $B$ 个 $(o_i, l_i, a_i)$ tuples，它们来自不同任务不同轨迹。让模型学：把 $(o_i, l_i)$ 拉到对应的 $a_i$ 附近，远离其他 $a_j$（$j \neq i$）。

这样你不需要显式造负样本——batch 里的其他样本天然就是负样本。这跟 CLIP 学"图像-文本对齐"一个套路，只是把文本换成 action。

### 架构

```
Image (o_t)  ─┐
              ├→ SigLIP2 (frozen) → Text-aware Visual Attention → F_combined ─┐
Text (l)     ─┘                                                              │
                                                                             ├→ cosine sim s(f, a)
Action (h_t, a_t) → Transformer Encoder → A ─────────────────────────────────┘
```

几个关键点：
- **SigLIP2 frozen**：保留 web-scale 知识，不让它在 action training 里被破坏
- **Text-aware visual attention**：让 visual encoder 只提取跟任务相关的视觉特征（受 [Otter](https://arxiv.org/abs/2503.03734) 启发）
- **Action encoder 用 transformer**：处理 (history $h_t$ + future chunk $a_t$)，捕捉时间依赖

### Loss：双向 InfoNCE

$$\mathcal{L}_i^{f \to a} = -\log \frac{\exp(s_{i,i})}{\sum_{j=1}^B \exp(s_{i,j})}$$

$$\mathcal{L}_i^{a \to f} = -\log \frac{\exp(s_{i,i})}{\sum_{j=1}^B \exp(s_{j,i})}$$

$$\mathcal{L}_{InfoNCE} = \frac{1}{2B} \sum_{i=1}^B (\mathcal{L}_i^{f \to a} + \mathcal{L}_i^{a \to f})$$

变量解释：
- $\mathbf{f}_i$: 第 $i$ 个样本的 normalized vision-language embedding
- $\mathbf{a}_j$: 第 $j$ 个样本的 normalized action embedding
- $s_{i,j} = \langle \mathbf{f}_i, \mathbf{a}_j \rangle$: 它们的 cosine similarity
- $s_{i,i}$: positive pair（真的配对）
- $s_{i,j}$ ($j \neq i$): negative pair（batch 内其他 action）
- $B$: batch size，决定 negatives 数量

**双向**的意思：从 vision-language 端 query action，也从 action 端 query vision-language。两个方向都 cycle-consistent，embedding space 学得更稳。

### Rephrase augmentation

训练时把每个 instruction 改写成 $N$ 种说法，配同一个 $(o_t, a_t)$。这样 verifier 见到同一意图的多种 phrasing，学到的是 **semantic equivalence** 而不是表面字面 matching。

Intuition：verifier 学到的不是"这句话对应这个动作"，而是"这堆话表达的意图都对应这个动作"。这样 test-time 见到新说法也不会傻。

---

## Test-Time 怎么用？

两层 hierarchy，paper 叫 "Hierarchical Prompt-Action Optimization"：

### 层 1：Language-level optimization
把用户指令 $l'$ 改写成 $K$ 个 rephrases $\{l_k'\}_{k=1}^K$。

每个 rephrase 用 VLA 采 $M$ 个 action candidates。

对每个 rephrase，算它的 $M$ 个 action 的平均 verifier score：
$$S_k = \frac{1}{M} \sum_{j=1}^M s_{k,j}$$

选 $S_k$ 最大的 rephrase $l_{k^*}'$。这个 rephrase 诱导的 action distribution 最"靠谱"。

### 层 2：Action-level optimization
在选中的 rephrase 下，挑 verifier score 最高的那个 action：
$$j^* = \arg\max_j s_{k^*, j}$$

执行 $a_{k^*, j^*}'$。

### 关键细节

**Verifier 评分时用原始 user instruction $l'$，不用 rephrase $l_k'$**
这点很重要。你在评估"这个 action 是否满足用户真实意图"，所以要拿原始指令比对。Rephrase 只是用来诱导 action distribution 的工具。

**Language-level 用 average score 而不是 max**
因为 rephrase 诱导的是 distribution，平均更鲁棒。Max 容易被 outlier 误导。

**Ensemble 3 个 verifier**
训 3 个不同 seed 的 verifier，inference 时 average embedding 再算 similarity。Variance reduction，跟 LLM 那边 ensemble reward model 一个道理。

---

## Boot-Time Compute：巧妙的工程优化

这是 paper 里容易被忽略但很重要的设计。

**问题**：VLM 生成 rephrase 很慢（11 秒生成 8 个），如果在 control loop 里调，机器人会卡住。

**观察**：一个 episode 里用户意图通常不变，没必要在每步都重新生成 rephrase。

**解法**：把 rephrase generation 移到 deployment 前（"boot time"）：
1. 拿到初始场景图 $o_0$ 和用户指令 $l'$
2. VLM 做 scene reasoning + 生成 K 个 rephrases
3. **Cache rephrase embeddings**
4. Runtime 时 retrieval 几乎零开销

### VLM prompt 设计（Appendix 8.10）

System prompt 定义任务，User prompt 要求 VLM：
1. 先 describe scene in its own words
2. 再 reinterpret instruction 在这个 scene 上下文里
3. 列出可能的 nouns/verbs/adjectives 替换
4. 生成 rephrases

这种中间 reasoning step 类似 chain-of-thought，但用于 lexical variation。关键是 **让 VLM 先 ground 到场景再改写**，避免 hallucination。

### VLM vs LLM 改写质量对比（Table 9）

例子：任务 "put redbull can on plate"
- VLM: "Place the **blue can** on the yellow plate" — 用视觉 ground 出颜色
- LLM: "Place the **energy drink** on the large tray" — 没视觉，用 category 词，引入 semantic drift

VLM 改写质量明显更高，因为它能 ground 到场景里实际颜色和物体。

---

## 实验结果：几个关键发现

### Finding 1：Random rephrase 是危险的

π₀ + 随机选一个 rephrase（不 verify）：
- ID tasks: 42.3%（比 base π₀ 的 41.5% 略升）
- OOD tasks: 28.7%（比 base π₀ 的 29.7% 反而降了！）

不同 task 上 variance 巨大：Eggplant in Basket 78%，Redbull on Plate 1%。

**Insight**：rephrase 是双刃剑。好的 rephrase 能救活 policy，坏的 rephrase 会 catastrophic mislead policy。这恰恰说明了 **verifier 的必要性**——你不能随便挑个 rephrase，必须 verify 哪个 rephrase 诱导的 action 真的符合用户意图。

### Finding 2：RoboMonkey 失败的教训

RoboMonkey 是 prior SOTA 的 test-time verifier 方法，用 7B VLM verifier + action resampling。但在 SIMPLER 上只拿到 7.5% / 24.7%，**比 base π₀ 还差**！

两个原因：
1. RoboMonkey 的 verifier 训练在 OpenVLA action distribution 上，跟 π₀ 的 action distribution mismatch
2. RoboMonkey 是 step-level verification，破坏了 flow-based policy 的 chunk structure

**教训**：test-time verifier 不是 plug-and-play，必须跟 base policy 对齐。CoVer 通过 contrastive learning 学的是 "semantic alignment"，对 action distribution 的具体形式更 robust。

### Finding 3：Training-time augmentation 不如 test-time verification

| 方案 | ID | OOD | Training FLOPs |
|------|-----|-----|----------------|
| π₀ (base) | 41.5 | 29.7 | 1.0× |
| π₀ (rephrase) | 44.0 | 48.7 | 16.0× |
| **π₀ + CoVer** | **57.0** | **61.0** | **3.8×** |

CoVer 用 π₀(rephrase) 的 **24% compute**，获得更高 performance：
- ID: +13%
- OOD: +12.3%

这就是 paper 标题 "Scaling Verification Can Be More Effective than Scaling Policy Learning" 的直接量化证据。

### Finding 4：两者互补

| 方案 | ID | OOD |
|------|-----|-----|
| π₀ (rephrase) + CoVer | **65.5** | **62.0** |

最强的组合是 training augmentation + test-time verification。它们不冲突，反而 synergize。

### Finding 5：在更强 base model 上仍有效

在 PolaRiS 上用 π₀.₅（更强 base）：

| Model | Task Progress | Success Rate |
|-------|---------------|--------------|
| π₀.₅ | 40.0 | 3.8 |
| π₀.₅ + CoVer | **53.9 (+13.9)** | **13.1 (+9.3)** |

即便 base model 已经很强，CoVer 仍能显著提升。

### Finding 6：Real-world 45% 绝对提升

两个 WidowX 任务：
- "pepto bismol on plate": +30% success
- "redbull on plate": +60% success

Base π₀ 在 challenging scenes 下经常**完全不动**（0% success），CoVer 让它至少能启动动作。

---

## Scaling 维度

Paper 测了 5 个维度，全部展示 consistent gains：

| Dimension | Range | Trend |
|-----------|-------|-------|
| Synthetic instructions | 8× → 64× | Top-1 retrieval accuracy 持续提升 |
| Model capacity | 250M → 1B | Steady improvement |
| Batch size | 2048 → 8192 | More in-batch negatives → 更好收敛 |
| Training epochs | 更多 | 更多 negatives exposure |
| Ensemble size | 1 → 8 | Variance reduction |

**重要观察**：500M verifier 比 250M verifier 强，主要因为 text encoder 大了 7×（280M vs 40M）。**Language representation 是 verifier 性能的主要 driver**——这印证了 paper 的核心 thesis：VLA 的 bottleneck 在语言理解，不在 action generation。

---

## Latency 分析（Table 2 & 5）

| Batch Size | π₀.₅ (ms) | CoVer (ms) | π₀.₅ + CoVer (ms) |
|------------|-----------|------------|---------------------|
| 1 | 56 | 7 | 63 |
| 16 | 445 | 8 | 453 |
| 32 | 865 | 8 | 873 |

**关键工程优化**：image-text encoder 跟 π₀.₅ forward pass **并行**（处理同一个 observation 但独立），所以 end-to-end latency 只增加 ~8ms（action encoder 部分）。

Batch=16 时，~2.2 Hz control frequency，对 quasi-static manipulation 够用。

---

## Compute Cost 估算

公式：$C \approx 6ND$
- $N$: 参数量
- $D$: 训练 tokens 数

CoVer 的关键节省：image 和 text encoders frozen during training，所以 backward pass 只算 action encoder 部分（~1.0×10⁹ FLOPs），远低于 forward pass（~3.3×10¹¹ FLOPs）。

| Configuration | Total FLOPs | Relative Cost |
|---------------|-------------|---------------|
| π₀ (Base Policy) | 3.4×10¹⁹ | 1.0× |
| π₀ (rephrase) (16× Data) | 5.4×10²⁰ | 16.0× |
| **CoVer** | **1.3×10²⁰** | **3.8×** |

---

## 我的思考

### 这篇 paper 的深层意义

它给 robotics 提供了一个 LLM 已经走过的路径的验证：**test-time compute 是新的 training compute**。

LLM 这边的发展轨迹：
1. 先 scaling pre-training
2. 然后 SFT/RLHF
3. 然后 test-time scaling（CoT, best-of-N, process reward model）

Robotics 现在正从 stage 1 往 stage 2 走。CoVer 这篇 paper 提前给出了 stage 3 的雏形：用 verifier 做 test-time selection。

### CoVer 作为 implicit reward model

CoVer 训出来的是一个 alignment score function $\mathcal{V}_\theta(o, h, l, a)$。这本质上是一个 **reward model**，只是用 contrastive learning 训出来而不是用 preference data。

这很关键，因为 robotics 的 preference data 标注成本极高。CoVer 给了一条不用 failure data 的路径训 reward model。下一步自然是用这个 reward model 做 RL post-training（类似 LLM 那边的 RLAIF）。

### 类比 LLM 的 process reward model

OpenAI 的 "let's verify step by step"：在 LLM 数学推理中训练 PRM 评估每个 step。CoVer 是 robotics 版本——但 evaluate 的是 action chunk 而不是 reasoning step。

未来可以扩展成 hierarchical PRM：既评估 high-level instruction choice，又评估 low-level action chunk 质量，甚至评估 action chunk 内的每一步。

### 局限性

- **VLM 依赖**：rephrase generation 依赖 GPT-4o 级别 VLM，本体部署困难
- **Quasi-static 限制**：2.2 Hz 对 dynamic manipulation 不够
- **Single-task focus**：每个 episode 单 instruction，long-horizon 多阶段任务未充分验证
- **Contrastive 的 batch 依赖**：需要大 batch (32K)，训练资源门槛高

### 与 RLHF 的关系

CoVer 是 implicit reward model，用 contrastive learning 代替 preference data。这避开了 RLHF 的标注成本。思路类似 LLM 那边的 RLAIF（Constitutional AI）——用某种自动化的方式生成 preference signal。

### 与 Diffusion Policy / Flow Matching 的关系

π₀ 是 flow-based，generate action chunks。CoVer 的 action encoder 用 transformer 处理 chunks，保留 chunk 内 temporal structure。RoboMonkey 失败的部分原因就是 step-level verification 破坏了 chunk structure。

这是 VLA 时代 verifier 与传统 RL verifier 的关键区别：**必须 respect action chunk 的整体结构**，不能拆成独立 step 评估。

---

## 总结

这篇 paper 的 elegance 在于它把多个 independently-developed idea 合成一个 coherent pipeline：
- Test-time scaling（从 LLM 借）
- Prompt optimization（从 LLM 借）
- Contrastive learning（从 CLIP 借）
- Boot-time caching（工程优化）

然后用一个漂亮的 compute-vs-performance 比较论证了核心 thesis：**在 VLA 时代，scaling verification 比 scaling policy 更高效**。

更深层的意义：CoVer 给了 robotics 一个**不用 failure data 的 reward model**。这可能是 robotics 走向 RL self-improvement 的关键 piece——就像 LLM 那边 RLHF 之前需要先有 reward model 一样。

参考链接：
- [Project page](https://cover-vla.github.io)
- [π₀ paper](https://arxiv.org/abs/2410.24164)
- [RoboMonkey](https://arxiv.org/abs/2506.17811)
- [OpenVLA](https://openvla.github.io/)
- [SIMPLER benchmark](https://arxiv.org/abs/2405.05941)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314)
- [Embodied Red Teaming](https://embodied-redteam.github.io/)

---

# CoVer-VLA 深度解析

Andrej，这篇 paper 我读完之后感觉是一个非常 elegant 的工作，它把 LLM 那边已经成熟的 "test-time scaling" 思想系统地移植到 VLA (Vision-Language-Action) 领域，并且做了一个相当漂亮的 empirical claim：**用同样的 compute 预算去训练一个 verifier 比直接 augment policy training 数据更高效**。这个 claim 的意义不亚于 Snell et al. 在 LLM 上的发现 ("Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters")。下面我分模块展开。

---

## 1. 核心问题与 motivation：Intention-Action Gap

VLA 模型 (π₀, OpenVLA, RT-2, GR00T N1, Gemini Robotics) 的失败模式很有意思——它们不是完全不会做 task，而是会做对一部分但语义上 misalign。Paper 给的例子很生动：

> "putting a plastic container into a drawer" → robot 正确 grasp 了 container，但分不清 oven 和 drawer，结果把 container 放进 oven，可能导致火灾。

这种失败模式说明 VLA 的 multimodal understanding（继承自 VLM backbone）在 fine-tuning 成 action generator 后被部分破坏了，即所谓的 **catastrophic forgetting** [10, 14]。传统解决方案是 scaling policy training：
- Augment 训练数据用 rephrased instructions [10]
- 用更大的 VLM backbone [2, 11]

但这些方法：(i) 只 yield incremental gains，(ii) 性能在 simple perturbations 下仍然 degrade severely，(iii) 进一步加剧 catastrophic forgetting。

Paper 的 insight：**与其在 training 时塞进更多 instruction 多样性，不如在 test-time 把 instruction 本身当作 decision variable 来优化**。

---

## 2. Test-Time Scaling Law for Embodied Instruction Following

这是 paper 最核心的 empirical 发现。他们在 Bridge V2 上采样 1000 个 (s, a, I) tuples，比较四种 action candidate sampling 策略：

1. **Repeated sampling**: $a \sim \pi(a | s, I)$ with positive temperature
2. **Gaussian perturbation**: 从少量 samples 拟合 Gaussian，再 draw 大量 samples
3. **Instruction rephrasing**: $a \sim \pi(a | s, l_k')$，每个 rephrase 一个 sample
4. **Hybrid sampling**: 每个 rephrase 上再 fan out 多个 samples

他们 fit 了 power law：

$$\log(e) \approx \log(a) + b \cdot \log(k)$$

变量解释：
- $e$: Normalized Root Mean Squared Error (NRMSE) between ground-truth action $a^*$ 和 sampled actions
- $k$: action candidates 数量（总 inference budget）
- $a$, $b$: 拟合参数，$a$ 反映 baseline error，$b$ 反映 scaling efficiency（斜率越负越好）

**关键发现**：
- Instruction rephrasing > Repeated sampling > Gaussian perturbation
- Hybrid sampling 最有效

**我的 intuition**：这个结果的背后含义是——action space 上的 local perturbation 只能探索 mode 附近的小邻域，而 instruction 上的 perturbation 会让 policy 进入**不同的 mode**。因为 VLA 是 conditional model $\pi(a|s, l)$，不同 $l$ 导致的 action distribution 之间的距离远大于同一 $l$ 下的 noise。这就构造了一个更 "diverse" 的 action proposal distribution，oracle verifier 选择时更容易命中正确 action。

这其实呼应了你 Karpathy 在 LLM 那边讲 "best-of-N" 的 intuition：单纯的 temperature sampling 在 mode 附近，而 CoT/prompt 变化会跳到不同 mode。这里 instruction rephrase 是 robotics 版本的 "prompt diversity"。

参考链接：
- [Best-of-N Scaling for LLMs (Snell et al.)](https://arxiv.org/abs/2408.03314)
- [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling (Brown et al.)](https://arxiv.org/abs/2407.21787)

---

## 3. Hierarchical Prompt-Action Optimization 形式化

这个 formulation 我觉得非常 clean。把 VLA inference 重写成两层优化问题：

### 3.1 Language-level optimization

$$l^* = \arg\max_{l' \in \mathcal{L}_r} \mathbb{E}_{a \sim \pi(\cdot | o_t, l')} [r(o_t, a, l)]$$

变量：
- $\mathcal{L}_r(l') = \{l_1', ..., l_K'\}$：K 个 rephrases
- $r(o_t, a, l)$: conceptual reward function，**不在 test-time 计算**，只用来定义目标
- $l^*$: 选中的 rephrase，其诱导的 action distribution 在期望 reward 下最大

这把 VLA inference 转化成 **language space 上的 optimization**，而不是 parameter space 上的 optimization。这有点像 prompt tuning 的 spirit，但完全在 inference time。

### 3.2 Action-level optimization

$$a_t^* = \arg\max_{j \in [M]} \mathcal{V}_\theta(o_t, h_t, l^*, a_j')$$

变量：
- $\mathcal{V}_\theta$: verifier 参数化为 $\theta$
- $h_t \in \mathcal{A}^W$: recent action history window（过去 $W$ 个 actions），提供 temporal context
- $a_j' \sim \pi(\cdot | o_t, l^*)$: $M$ 个 candidate action chunks

两层 hierarchy 的好处是：先选 rephrase（high-level 决定 action distribution），再在选中的 distribution 内选最佳 action chunk（low-level 决定具体执行）。

---

## 4. CoVer 的训练：Contrastive Learning 的精妙之处

这是 paper 的技术核心。

### 4.1 为什么用 Contrastive Learning？

核心 challenge：robotic datasets 只有 successful demonstrations，**没有 failure labels**。传统 verifier 训练需要正负样本对比。两条 prior route：
- Synthesize incorrect actions → 产生 unrealistic motions
- Manually annotate failures → prohibitively expensive

Contrastive learning [CLIP, InfoNCE] 的妙处：**把 batch 中其他 actions 当作 implicit negatives**，不需要显式 failure labels。这等价于让 verifier 学会"哪些 (instruction, action) 配对是 aligned 的"，剩下的就自然是 misaligned 的。

### 4.2 Architecture

```
Vision (o_t) ──┐
               ├─→ SigLIP2 (frozen) ──→ Text-aware Visual Attention ──→ F_combined ─┐
Language (l) ──┘                                                                    │
                                                                                    ├─→ cosine similarity s(f, a)
Action (h_t, a_t) ─→ Transformer Encoder ─→ A ───────────────────────────────────────┘
```

关键设计：
- SigLIP2 encoders frozen during training 以保留 web-scale knowledge（防止 catastrophic forgetting 再次发生）
- **Text-aware visual attention**: 让 visual encoder 选择性地提取 task-relevant features（受 Otter [16] 启发）
- Action encoder 用 transformer 处理 short-term history $h_t$ + future chunk $a_t$，捕捉 long-range temporal dependencies

### 4.3 InfoNCE Loss 的双向对称设计

给定 minibatch of B tuples $\{(o_i, h_i, l_i, a_i)\}_{i=1}^B$：

归一化：
$$\mathbf{f}_i = \mathbf{F}_i / \|\mathbf{F}_i\|_2, \quad \mathbf{a}_i = \mathbf{A}_i / \|\mathbf{A}_i\|_2$$

Pairwise similarity：
$$s_{i,j} = \langle \mathbf{f}_i, \mathbf{a}_j \rangle$$

双向 InfoNCE：
$$\mathcal{L}_i^{f \to a} = -\log \frac{\exp(s_{i,i})}{\sum_{j=1}^B \exp(s_{i,j})}$$

$$\mathcal{L}_i^{a \to f} = -\log \frac{\exp(s_{i,i})}{\sum_{j=1}^B \exp(s_{j,i})}$$

$$\mathcal{L}_{InfoNCE} = \frac{1}{2B} \sum_{i=1}^B (\mathcal{L}_i^{f \to a} + \mathcal{L}_i^{a \to f})$$

变量解释：
- $\mathbf{f}_i$: 第 $i$ 个样本的 normalized vision-language embedding
- $\mathbf{a}_j$: 第 $j$ 个样本的 normalized action embedding
- $s_{i,i}$: positive pair 的 similarity（aligned）
- $s_{i,j}$: negative pair 的 similarity（cross-batch misaligned）
- $B$: batch size，决定 negatives 数量

**双向的含义**：从 vision-language 端 query action，也从 action 端 query vision-language。这让 embedding space 在两个方向上都是 cycle-consistent 的，更稳定。CLIP 也是这么设计的。

### 4.4 为什么这个能 work？

我的 intuition：当 batch 里的 actions 都来自不同 tasks / different trajectories 时，它们之间天然构成 "语义上不同的 actions"。Contrastive loss 强制模型学到：要把 (o_t, l_t) 拉到与之真正对应的 a_t 附近，远离其他 a_j。这隐含地学到了"什么样的 action 是对应这个 instruction 的语义"。

Rephrase augmentation 的作用：让 verifier 见到同一个 intent 的多种 phrasing，于是 verifier 学到的是 **semantic equivalence**，而不是表面 phrasing matching。这就是为什么训练时的 rephrase augmentation 让 verifier 学到了 "意图" 而非 "字面"。

---

## 5. Boot-Time Compute: 巧妙的工程优化

这是 paper 中容易被忽视但很重要的设计：

### 问题
VLM 在 runtime 调用太慢（11 秒生成 8 个 rephrases），会破坏 control loop 的实时性。

### 解决方案
利用 **episode 内 instruction 通常不变** 这个特点，把 rephrase generation 完全移到 deployment 前 ("boot time")：

1. VLM 拿到初始场景图 $o_0$ 和 user instruction $l'$
2. 做 structured reasoning（识别 objects、spatial relations、task decomposition）
3. 生成 K 个 rephrases $\{l_k'\}_{k=1}^K$
4. **Cache rephrase embeddings** 以便 runtime 时 retrieval 几乎零开销

### VLM prompt 的细节
Paper 在 Appendix 8.10 给了详细的 prompt。System prompt 定义"重写 instruction 保持 intent 不变"，User prompt 要求 VLM：
1. 先 describe scene in its own words
2. 再 reinterpret instruction 在 scene 上下文中
3. Enumerate potential lexical variations (nouns, verbs, adjectives)
4. 生成 rephrases

这种 **intermediate reasoning step** 类似 chain-of-thought，但用于 lexical variation generation，显著减少 instruction drift。

**VLM vs LLM 的 rephrase 对比** (Table 9) 很有意思：
- Task: "put redbull can on plate"
- VLM: "Place the **blue can** on the yellow plate" — 用视觉 grounding 推出颜色
- LLM: "Place the **energy drink** on the large tray" — 没有视觉 grounding，用 category-level terms，引入 semantic drift

这强化了"VLM rephrase > LLM rephrase"的结论，因为 VLM 能 ground 到 scene 中实际的颜色和物体。

---

## 6. Test-Time Verification Pipeline

Algorithm 2 完整流程：

```
Input: π, verifier ensemble V_θ, user instruction l', K, M

Boot-time:
  Generate {l_k'}_{k=1}^K ← VLM(o_0, l')
  Cache embeddings

while episode not finished:
  # 1. Sample action proposals
  for k = 1 to K:
    for j = 1 to M:
      a_{k,j}' ~ π(· | o_t, l_k')
  
  # 2. Score proposals
  s_{k,j} = V_θ(o_t, h_t, l', a_{k,j}')
  
  # 3. Select rephrase (language-level)
  S_k = (1/M) Σ_j s_{k,j}
  k* = argmax_k S_k
  
  # 4. Select action (action-level)
  j* = argmax_j s_{k*,j}
  
  Execute a_{k*,j*}'
  Update (o_{t+Δ}, h_{t+Δ})
```

注意几个关键点：
- Verifier scoring 是用 **原始 user instruction $l'$**，而不是 rephrase $l_k'$，这样 verifier 评估的是 "这个 action 是否满足用户的真实意图"
- Language-level selection 用 average score $S_k$，是因为 rephrase 诱导的是 **distribution**，平均更鲁棒
- 用 ensemble of 3 verifiers 平衡 robustness 和 overhead

---

## 7. Scaling 实验细节

Paper 测试了 5 个 scaling 维度，全部展示 consistent gains：

| Dimension | Range | Trend |
|-----------|-------|-------|
| Synthetic instructions | 8× → 64× | Top-1 retrieval accuracy 持续提升 |
| Model capacity | 250M → 1B | Steady improvement |
| Batch size | 2048 → 8192 | Better convergence (more in-batch negatives) |
| Training epochs | More epochs | More diverse negatives exposure |
| Ensemble size | 1 → 8 | Variance reduction (averaging biases) |

### Model size 配置 (Table 4)
| Verifier Size | Backbone |
|---------------|----------|
| 250M | ViT-B/16-CLIP |
| 500M | ViT-B/16-SigLIP2 |
| 1B | ViT-L/16-SigLIP2 |

注意 500M 用了 7× 更大的 text encoder (280M vs 40M)，**language representation 是 verifier 性能的主要 driver**——这印证了 paper 的核心 thesis：在 VLA 中语言理解是 bottleneck。

### 最终训练配置
- 1B parameter backbone
- Batch size: 32,768
- 20M offline samples (16× synthetic instructions on Bridge V2)
- 2k steps
- 8× NVIDIA H200 GPUs
- Ensemble: 3 verifiers

---

## 8. 实验结果深度分析

### 8.1 SIMPLER Benchmark (Table 3)

| Model | ID Avg | OOD Avg |
|-------|--------|---------|
| π₀ (base) | 41.5 | 29.7 |
| π₀ (rephrase) | 44.0 | 48.7 |
| π₀ + random rephrase | 42.3 | 28.7 |
| RoboMonkey [22] | 7.5 | 24.7 |
| **π₀ + CoVer** | **57.0** | **61.0** |
| **π₀ (rephrase) + CoVer** | **65.5** | **62.0** |

### 几个关键 ablation findings

**(1) Training-time augmentation alone provides modest gains**
- π₀ → π₀(rephrase): ID 41.5→44.0 (+2.5), OOD 29.7→48.7 (+19)
- OOD 上提升明显但 ID 提升微弱——说明 augmentation 主要在 unseen instructions 上 work

**(2) Random rephrases 是危险的**
- π₀ + random rephrase: ID 42.3 (slight up), OOD 28.7 (down)
- 不同 task 上 variance 巨大：Eggplant in Basket 78%, Redbull on Plate 1%
- **Insight**: rephrase 本身是把双刃剑，好的 rephrase 能救活 policy，坏的 rephrase 会 catastrophic mislead。这就是为什么需要 verifier 来 select rephrase。

**(3) RoboMonkey 失败原因分析**
RoboMonkey 在 SIMPLER 上 7.5% / 24.7%，比 base π₀ 还差！Paper 给出两个原因：
- RoboMonkey 的 verifier 训练在 OpenVLA action 分布上，与 π₀ action 分布 mismatch
- RoboMonkey 是 step-level verification，破坏了 flow-based policy 的 chunk structure

这是一个非常重要的负面结论：**test-time verifier 不是 plug-and-play**，需要和 base policy 的 action distribution 对齐。CoVer 通过 contrastive learning 学到的是"semantic alignment"，更通用。

**(4) CoVer 与 training-time augmentation 互补**
π₀(rephrase) + CoVer > π₀(rephrase)，> π₀+CoVer——这说明 rephrase augmentation 扩展了 verifier 见过的语言覆盖，进一步放大 verifier 的能力。

### 8.2 PolaRiS Benchmark (Table 1)

用更强的 base model π₀.₅：

| Model | Task Progress | Success Rate |
|-------|---------------|--------------|
| π₀.₅ | 40.0 ± 6.4 | 3.8 ± 4.9 |
| π₀.₅ + CoVer | **53.9 ± 11.7 (+13.9)** | **13.1 ± 14.1 (+9.3)** |

即便 base model 已经很强，CoVer 仍能提升。注意 success rate 的 baseline (3.8%) 极低，说明这些任务（PanClean, BlockStack, FoodBussing）本身非常难，CoVer 把它抬到 13.1%。

### 8.3 Real-World (Figure 9)

两个 WidowX 任务：
- "pepto bismol on plate": +30% success
- "redbull on plate": +60% success

Base π₀ 在 challenging scenes 下经常 **完全不动** (0% success)，CoVer 让它**至少能 initiate motion**。

---

## 9. Compute Cost 分析：论文的 punchline (Table 7)

| Configuration | Total FLOPs | Relative Cost |
|---------------|-------------|---------------|
| π₀ (Base Policy) | 3.4×10¹⁹ | 1.0× |
| π₀ (rephrase) (16× Data) | 5.4×10²⁰ | 16.0× |
| **CoVer (Ours)** | **1.3×10²⁰** | **3.8×** |

CoVer 用 **24% 的 compute of π₀(rephrase)**，获得了**更高的 performance**：
- ID: 57.0 vs 44.0 (+13%)
- OOD: 61.0 vs 48.7 (+12.3%)

这就是 paper 标题 "Scaling Verification Can Be More Effective than Scaling Policy Learning" 的直接量化证据。

**Compute 估算公式**：$C \approx 6ND$，其中 $N$ = 参数量，$D$ = 训练 tokens 数。
CoVer 的关键节省：image 和 text encoders frozen during training，所以 backward pass 只算 action encoder 部分（~1.0×10⁹ FLOPs），远低于 forward pass（~3.3×10¹¹ FLOPs）。

---

## 10. Latency 优化 (Table 2 & 5)

| Batch Size | π₀.₅ (ms) | CoVer (ms) | π₀.₅ + CoVer (ms) |
|------------|-----------|------------|---------------------|
| 1 | 56 | 7 | 63 |
| 16 | 445 | 8 | 453 |
| 32 | 865 | 8 | 873 |

**关键工程优化**：image-text encoder 可以与 π₀.₅ forward pass **并行**，因为它们处理同一个 observation 但独立。所以 end-to-end latency 只增加 ~8ms (action encoder 部分)。

在 batch=16 时，~2.2 Hz 的 control frequency，对 quasi-static manipulation 任务足够。

---

## 11. 我的思考与联想

### 11.1 类比 LLM 的发展轨迹
这篇 paper 基本上把 LLM 的几个核心 idea 系统地搬到 robotics：
1. **Test-time scaling** (Snell et al., Brown et al.) → 通过 sample 多个 candidates + verify
2. **Prompt optimization** (e.g., DSPy, APE) → 把 instruction 当 decision variable
3. **Contrastive reward model** (RLHF 的 preference model 思想) → 用 InfoNCE 学 semantic alignment
4. **Boot-time reasoning** (类似 system prompt caching) → VLM reasoning 移到 deployment 前

### 11.2 与 RLHF 的关系
CoVer 本质上是一个 **implicit reward model**，但不需要 preference data。Contrastive learning 把 batch 内其他 samples 当作 implicit negatives，等价于学到了"什么是 aligned vs misaligned"的偏好。这避开了 RLHF 的标注成本。

### 11.3 与 process reward model (PRM) 的关系
OpenAI 的 "let's verify step by step" 思路：在 LLM 数学推理中训练 PRM 评估每个 step 的质量。CoVer 是 robotics 版本——但 evaluate 的是 action chunk 而不是 reasoning step。可以考虑未来把 CoVer 扩展为 **hierarchical PRM**，评估 action sequence 的每一步。

### 11.4 与 Diffusion Policy / Flow Matching 的关系
π₀ 是 flow-based，generate action chunks。CoVer 的 action encoder 用 transformer 处理 chunks，保留了 chunk 内的 temporal structure。RoboMonkey 失败的部分原因就是 step-level verification 破坏了 chunk structure。这是 VLA 时代的 verifier 与传统 RL verifier 的关键区别。

### 11.5 潜在的局限
- **VLM 依赖**：rephrase generation 依赖 GPT-4o 级别的 VLM，在机器人本体上部署仍困难
- **Quasi-static 限制**：2.2 Hz 对 dynamic manipulation (e.g., 抛接、快变 contact) 仍不够
- **Single-task focus**：每个 episode 单 instruction，多阶段 long-horizon 任务未充分验证
- **Contrastive 的 batch 依赖**：需要大 batch size (32K) 才能获得足够 negatives，训练资源门槛高

### 11.6 Future directions paper 提到
- Post-training with RL using CoVer as reward
- Runtime monitoring (用 verifier score 检测 failure)
- 更高效的 verifier 架构

参考链接：
- [π₀ paper (Black et al.)](https://arxiv.org/abs/2410.24164)
- [RoboMonkey (Kwok et al.)](https://arxiv.org/abs/2506.17811)
- [OpenVLA (Kim et al.)](https://openvla.github.io/)
- [SIMPLER benchmark (Li et al.)](https://arxiv.org/abs/2405.05941)
- [Bridge V2 (Walke et al.)](https://arxiv.org/abs/2308.12952)
- [Embodied Red Teaming (Karnik et al.)](https://embodied-redteam.github.io/)
- [CLIP (Radford et al.)](https://arxiv.org/abs/2103.00020)
- [SigLIP 2 (Tschannen et al.)](https://arxiv.org/abs/2502.14786)
- [Otter (Huang et al.)](https://arxiv.org/abs/2503.03734)
- [Project page](https://cover-vla.github.io)

---

## 总结

CoVer-VLA 这篇 paper 的 elegance 在于它把多个 independently-developed idea（test-time scaling, prompt optimization, contrastive learning, boot-time caching）合成一个 **coherent pipeline**，并且用一个漂亮的 compute-vs-performance 比较论证了核心 thesis：在 VLA 时代，**scaling verification 比 scaling policy 更高效**。

它给你 Karpathy 在 LLM 那边讲过的 "test-time compute is the new training compute" 提供了一个 robotics 领域的 solid empirical 验证。如果 VLA 模型继续沿 LLM 的轨迹发展，我会预期这条 verification-based test-time scaling 路线会成为 standard deployment pipeline 的一部分——就像 inference-time best-of-N 已经成为 LLM deployment 的标配一样。

CoVer 的 contrastive verifier 思路还有一个深远意义：它给了我们一个**不需要 failure data 的 reward model**，这可能是 robotics 走向 RL self-improvement 的关键 piece（类似 LLM 那边 RLHF 之前需要先有 reward model）。从这个角度看，CoVer 不仅是 test-time 工具，也是 future RL post-training 的基础设施。
