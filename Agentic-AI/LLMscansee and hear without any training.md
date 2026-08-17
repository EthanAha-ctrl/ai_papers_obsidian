---
source_pdf: LLMscansee and hear without any training.pdf
paper_sha256: 4b5b144797244d66ef863f316e222c44ab3e26d6f91492ab6fdc9ea7218a2b22
processed_at: '2026-08-05T15:38:59-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：MILS 在干嘛

## 一句话版本

LLM 本身是个又瞎又聋的语言天才。这篇 paper 说：**给它配一个能看见/听见的"打分员"，让它俩玩"你猜我评"的游戏，猜几轮就能猜对，完全不用训练。**

## 为什么这事有意思

想象一个场景：你有个朋友，文笔极好，能写出漂亮的英语句子，但他生来就看不见。你给他看一张猫的图，让他描述——他当然写不出。

传统做法：收集几百万对"图+描述"，训练一个模型直接从图生成描述。这就是 [BLIP-2](https://arxiv.org/abs/2301.12597)、[LLaVA](https://arxiv.org/abs/2304.08485) 这类 vision-language model 的套路。问题是每换个任务（video、audio、style transfer）就得重新搞数据重新训。

MILS 的做法：**不训练。** 就让这个瞎子朋友随便写 50 个 caption，你拿每个去跟图比对（用 CLIP 算 similarity），告诉他自己写的哪些分数高。他根据反馈再写 50 个，你再打分。来回 10 轮，他就能写出相当准的 caption。

关键是——他从头到尾没"看见"那张图。他只是从分数反馈里推断"哦，这张图大概有猫、有键盘、有显示器"。

## 具体怎么玩

**Round 1**：瞎子朋友凭空写 50 个 caption。因为你没给他任何线索，他可能写："a dog playing in the park"、"a sunset over the ocean"、"a cat sitting on a monitor"……各种乱猜。

**Scorer**（比如 [SigLIP](https://arxiv.org/abs/2303.15343)）把每个 caption 跟图算 cosine similarity，发现 "a cat sitting on a monitor" 得分最高，"a sunset" 得分最低。返回 top-50 排序。

**Round 2**：你把这 50 个 caption + 分数喂回给瞎子朋友，说："上次你写的这些，分数从高到低是这样。再写 50 个，争取更高分。"

瞎子朋友的 LLM 大脑开始 in-context learning：它发现 "cat"、"monitor"、"keyboard" 这些词出现频率高，于是新写的 caption 都围绕这些词组合——"a cat on a keyboard near a monitor"、"a feline sitting on a computer screen"……

**Round 10**：caption 收敛到类似 "A cat sitting on a monitor with a keyboard nearby"。

整个过程 [paper 的 Figure 8](https://arxiv.org/abs/2411.18448) 展示得很清楚，你能看到 caption 从 vague 到 specific 的演化。

## 为什么这比之前的 zero-shot 方法聪明

之前的 [ZeroCap](https://arxiv.org/abs/2206.14106) 走的是另一条路：让 LLM 一个 token 一个 token 地生成，每生成一个 token 就用 CLIP 的 gradient 微调下一个 token 的概率分布。这听起来很"科学"，实际上：

- 一个 token 一个 token 地走，error 会累积
- 必须对 CLIP 求 gradient，换 modality 就得重写
- 对 PickScore 这种要跑 diffusion 的 scorer 完全不可行——没法 backprop through diffusion

MILS 把整个 caption 当成一个原子单位来评估。这让它能用任何 scorer，哪怕 scorer 是个黑盒。**句子级别的 black-box search 比 token 级别的 gradient guidance 更鲁棒、更通用。**

## 同一个套路打六个 task

这是 paper 最骚的地方。同一个 generator-scorer 迭代框架，换换 module 就能做完全不同的事：

| 想干什么 | 让 LLM 生成什么 | 用什么打分 |
|---|---|---|
| Image captioning | caption | SigLIP（图-文相似度）|
| Video captioning | caption | ViCLIP（视频-文相似度）|
| Audio captioning | caption | ImageBind（音频-文相似度）|
| 改进 T2I 生成 | 改写后的 prompt | PickScore（人类偏好预测）|
| Style transfer | edit instruction | Gram matrix（纹理相似度）|
| 跨模态算术 | 组合 caption | 间接，通过下游 T2I |

注意最后两个特别有意思：

**Style transfer**：给一张 content 图和一张 style 图（比如梵高星空），让 LLM 写一句 edit instruction（"make it look like an impressionist painting with swirling brushstrokes"），然后 [Emu Edit](https://arxiv.org/abs/2311.10089) 执行编辑，再用 Gatys 的 [Gram matrix loss](https://arxiv.org/abs/1508.06576) 检查生成图的纹理是否匹配 style 图。LLM 从头到尾两张图都没看到，只看到"我写的 instruction 拿了多少分"。它能通过分数反馈学会描述它看不见的视觉风格——这相当 magic。

**Cross-modal arithmetic**：一张猫的图 + 一段海浪声 → 生成一张"海边的猫"的图。做法是把图 invert 成 text（"a cat on grass"），把音频 invert 成 text（"ocean waves crashing on shore"），让 LLM 把两个 caption 组合成"a cat beside the shore with waves coming"，再喂给 T2I 模型。[ImageBind](https://arxiv.org/abs/2305.05665) 也能做这事，但它只能在 embedding 空间做线性加法，且只能接兼容 CLIP embedding 的 T2I 模型。MILS 把 inversion 落到 text 上，text 是任何模型都能吃的 universal API，且 LLM 的语义组合能力远比 embedding 加法灵活。

## 效果怎么样

几个 highlights：

**MSCOCO image captioning**：METEOR 15.0、SPICE 9.6，跟专门训练的 [MeaCap](https://arxiv.org/abs/2405.13069) 接近（MeaCap SPICE 11.8 更高，但 MeaCap 有 memory module 做 concept retrieval，专门优化词汇匹配）。CIDEr 33.3 输给 MeaCap 的 42.5——CIDEr 看具体词汇匹配，MILS 没学过 MSCOCO 的"captionese"方言。BLEU4 8.0 是所有方法里最高的，说明语法流畅（LLM 语言先验的功劳）。

**Video captioning (MSR-VTT)**：METEOR 14.4，超过在 VideoCC3M 上训练的 baseline（11.3）。在 HowTo100M 上训练的 baseline CIDEr 只有 0.5——基本是垃圾，说明 video captioning 对 training data 质量极敏感。MILS 完全不训练却很有竞争力。

**Audio captioning (Clotho)**：SPICE 7.6 vs [ZerAuCap](https://arxiv.org/abs/2310.03093) 的 5.3，涨了 43%。audio 这个 modality 上 MILS 优势明显。

**T2I 改进**：用 [DrawBench](https://arxiv.org/abs/2205.11487) 200 prompt，Amazon Mechanical Turk 人类评估，MILS 改写后的 prompt 生成的图在 quality 和 faithfulness 上都被偏好。这跟 [DALL-E 3](https://arxiv.org/abs/2307.01952) 的思路一致——prompt rewrite 能显著提升 T2I——但 DALL-E 3 是训练一个 captioner 做数据 curation，MILS 是 inference-time 自动 rewrite。

## 几个关键 ablation 的直觉

**迭代步数**：10-20 步收敛，且 scorer 分数和人类判断高度 correlated——没有"scorer 高分但人类不喜欢"的 reward hacking。句子级搜索空间太大，LLM 难以找到 adversarial 短语。

**Initial set 大小**：30K initial caption 越多越好。这暴露一个隐含依赖：LLM 需要"概念覆盖先验"避免在某些 visual concept 上想不起来。做法是取 ImageNet 1000 个 class label，每个让 LLM 生成 40 个 caption。换 domain（如医学影像）就需要新的 prior set。

**LLM 和 scorer 大小**：两个都越大越好，但 LLM scaling 收益更大——暗示 language prior 是 bottleneck，scorer 已经够用。Llama 3.1 8B > Mistral 7B > Gemma2 9B。

## 跟更大趋势的关系

MILS 属于 "inference-time compute as task-solver" 这条线：

- [O1](https://openai.com/index/introducing-openai-o1-preview/)：RL 训练 LLM 在 test time 多思考
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)：test-time reasoning 的零成本版
- [OPRO](https://arxiv.org/abs/2310.03720)：LLM 自己做黑盒优化
- [Self-Refine](https://arxiv.org/abs/2303.11366)：LLM self-critique + revise

MILS 的独特之处是**引入外部 scorer 作为 grounding**。Self-Refine 的问题是 LLM 自己 critique 自己容易 hallucinate（没 grounding）。MILS 用 CLIP/PickScore 这种 perception-grounded scorer 提供 objective feedback，避免了 self-reinforcement 的幻觉。

激进 hypothesis：**如果 LLM 足够强 + 任意 verifier 足够准，任何 task 都不需要 supervised training，只需要 test-time search。** 这是 [Scaling test-time compute](https://arxiv.org/abs/2408.03314) 论文的极端版本。当然 scorer 本身是 trained 的（CLIP 用了 400M pair），所以严格说是 "no task-specific training"，不是 "no training at all"。但 task-level 的 zero-shot emergence 已经很惊人。

## 一个有意思的类比：离散 diffusion

可以把 MILS 看成 caption 空间上的离散 diffusion：

- Diffusion：从 noise $z_T$ 出发，迭代去噪，逐渐 sharpen 到 data manifold
- MILS：从 random captions 出发，迭代 refine，逐渐 sharpen 到 high-scoring caption

scorer 类比 score function $\log p(x)$，LLM 类比 denoiser。这也解释了为什么 10 步就收敛——跟 diffusion 一样，前几步 coarse alignment，后面 fine-tuning。

## 局限

1. **慢**：10 步 × (LLM 生成 50 candidate + scorer 50 次 forward) ≈ 几十秒到分钟级，比 forward-only VLM 慢得多
2. **Scorer 上限**：style transfer 受 Gram matrix 分辨率限制；audio 受 ImageBind 质量限制
3. **隐含 prior**：30K initial set 依赖 ImageNet 类别，新 domain 要重建
4. **CIDEr 吃亏**：需要精确词汇匹配的 benchmark 上不如专门方法
5. **细粒度 spatial reasoning 可能不行**：scorer 是 global similarity，"红球在蓝方块左边"这种关系 CLIP 分辨不清

## 你可能关心的 relevance

考虑到你过去的工作——[Karpathy & Fei-Fei 2015](https://arxiv.org/abs/1412.2306) 是 MILS 评估用的 MSCOCO split 的来源。当年你用 bidirectional LSTM + CNN alignment 需要 supervised pair，MILS 在同一个 split 上做 zero-shot，历史闭环。

你的 [nanoGPT](https://github.com/karpathy/nanoGPT) / [minGPT](https://github.com/karpathy/minGPT) 精神强调 minimalism，MILS 高度一致——core code 可能 <100 行，比 ZeroCap 的 gradient engineering 简单一个数量级。

你的 "Software 2.0" 框架里，MILS 是个有趣变种：weights 不变，inference-time 用 search 替代 weight update。某种意义上是 "Software 2.0 + 1.5"——用 learned model 做 search operator。

更激进地，如果 LLM 足够强 + verifier 足够准，inference-time search 可能成为 task-solving 的默认范式，task-specific training 退居 fine-tuning 角色。这跟 O1 的哲学遥相呼应，只是 O1 在数学/代码上验证，MILS 在多模态上验证。

---

**核心 reference：**
- [MILS paper](https://arxiv.org/abs/2411.18448)
- [MILS code](https://github.com/facebookresearch/MILS)
- [SigLIP](https://arxiv.org/abs/2303.15343) / [ImageBind](https://arxiv.org/abs/2305.05665) / [PickScore](https://arxiv.org/abs/2305.01569)
- [ZeroCap](https://arxiv.org/abs/2206.14106) / [MeaCap](https://arxiv.org/abs/2405.13069)（对比方法）
- [O1](https://openai.com/index/introducing-openai-o1-preview/) / [Scaling test-time compute](https://arxiv.org/abs/2408.03314)（大趋势）

---

# MILS: Multimodal Iterative LLM Solver — 让 LLM "看见听见" 的 test-time 黑魔法

## 一、核心 insight：把 LLM 当成 proposal distribution，把 CLIP 当成 fitness function

这篇 paper 的精髓可以用一句话概括：**任何多模态任务，只要能表述成 "找一个 text 使得某个 scorer 给出高分"，就能用 LLM + scorer 的迭代搜索来解决，完全不需要 task-specific training。**

形式化地说，给定 test sample $x$（image / video / audio / style pair），目标是找一个 text $c^*$：

$$c^* = \arg\max_{c \in \mathcal{C}} \; S(x, c)$$

其中 $S: \mathcal{X} \times \mathcal{C} \to \mathbb{R}$ 是 scorer（如 CLIP similarity、PickScore、Gram matrix distance），$\mathcal{C}$ 是 text 空间。问题在于 $\mathcal{C}$ 是离散的、巨大、且 $S$ 对 $c$ 不可微（或可微但梯度噪声大）。

MILS 的做法是把 LLM 当成 **learned proposal distribution** $p_{\text{LLM}}(c \mid \text{history})$，迭代地：

1. **Generate**: LLM 根据 history $T_t = \{(c_i, s_i)\}_{i=1}^{K}$ 生成一批新候选 $\{c_j'\}$
2. **Score**: scorer 计算 $s_j' = S(x, c_j')$
3. **Select**: 保留 top-K $\to T_{t+1}$

迭代 $N$ 步（paper 里 $N=10$）直到收敛（候选集相似度稳定）。

这本质上是 **evolution strategy / CMA-ES 的语言版本**：LLM 是 mutation + crossover operator（通过 in-context learning 隐式学会 fitness landscape），scorer 是 fitness function。LLM 通过 seeing 过去的 candidate-score pair，内部近似了 $\nabla_c S$ 的方向，然后 propose 更好的 $c$。

## 二、为什么这比 ZeroCap / MeaCap 等 gradient-based 方法更优雅

之前的 zero-shot captioning 方法（[ZeroCap](https://arxiv.org/abs/2206.14106), [ConZIC](https://arxiv.org/abs/2306.05236), [MeaCap](https://arxiv.org/abs/2405.13069)）走的是 **token-level gradient guidance** 路线：

$$p(y_t \mid y_{<t}, x) \propto p_{\text{LLM}}(y_t \mid y_{<t}) \cdot \exp\left(\frac{1}{\beta} \cdot \frac{\partial S(x, y)}{\partial y_t}\bigg|_{y=y_{<t}}\right)$$

问题：
- **必须 differentiable scorer**：CLIP 可以，但 PickScore（要跑 diffusion）、Gram matrix（要跑 VGG + 编辑模型）、T2I 重写（要 backprop through diffusion）都不可行
- **token-by-token greedy** 容易误差累积，且长句时 gradient 信号衰减
- **无法换 modality**：换 audio 就要重写整个 pipeline

MILS 是 **sentence-level black-box optimization**，整个 caption 作为原子单位。这让它能：
- 用任何 scorer（哪怕是非可微的、黑盒的、人肉评估的）
- 跨 modality 只需换 scorer（SigLIP → ViCLIP → ImageBind）
- 跨 task 只需换 generator 组合（LLM-only / LLM+T2I / LLM+Editor）

直觉上：**LLM 的语言先验已经足够强，能生成语法正确、语义合理的 caption；缺的只是一个"裁判"告诉它哪个更准。** ZeroCap 让 LLM 兼任运动员和裁判（用 gradient 引导 token），MILS 把裁判外包给专门的 multimodal model，让 LLM 专心做语言生成。

## 三、架构详解：GENERATOR × SCORER 的组合矩阵

| Task | GENERATOR | SCORER | 输入 |
|---|---|---|---|
| Image captioning | Llama 3.1 8B | SigLIP ViT-L/14 | image |
| Video captioning | Llama 3.1 8B | ViCLIP ViT-L/14 (8 frames) | video |
| Audio captioning | Llama 3.1 8B | ImageBind | audio |
| T2I 改进 | Llama 3.1 8B → LDM / FLUX.1-schnell | PickScore | text prompt |
| Style transfer | Llama 3.1 8B → Emu Edit | Gram matrix on VGG19 (multi-layer) | content image + style image |
| Cross-modal arithmetic | image→text (MILS) + audio→text (MILS) + LLM combine → T2I | (中间无 scorer，组合后用 PickScore) | image + audio |

关键设计：

### GENERATOR 的 prompt 结构（见 Appendix B）

```
You need to provide a short image description. 
I am providing to you a list of short image descriptions and scores. 
Higher score means that the image description characterizes the image better:
{descriptions}   ← top-K 历史 candidate + score
Generate additional {requested number} short image descriptions 
that you think that will maximize the score and fully capture the image. 
Be concrete and try to find elements that are unique to this image. 
You can introduce new elements, combine unique elements, rephrase, drop, simplify...
```

这个 prompt 让 LLM 扮演一个"看到历史尝试和分数的探索者"。LLM 的 in-context learning 能力让它从 $\{(c_i, s_i)\}$ 中推断出"哪些词组合得分高"，相当于隐式估计了 $\hat{S}(c)$ 的局部曲面。

### SCORER 的 ϵ-greedy 选择

paper 提到可以加 ϵ-greedy（保留一些低分 candidate 维持 diversity），但实验发现 greedy top-K 效果最好。这有点反直觉——通常 evolutionary search 需要 exploration。可能的解释：LLM 自身的 sampling temperature 已经提供了足够的 diversity，不需要 scorer 端刻意保留差解。

### Bootstrap：30K initial captions

这是 paper 一个隐藏的关键设计。对于 captioning，他们用 [Gandelsman et al. 2024](https://arxiv.org/abs/2406.04341) 的方法：取 ImageNet 的 1000 个 class label，每个让 LLM 生成 40 个 caption，得到 ~30K。audio 版本用 AudioSet 527 类 × ~100 caption ≈ 50K。

Ablation（Figure 10）显示 initial set 越大 SPICE 越高，且 CLIP similarity 也越高。这暗示 **LLM 即使语言能力很强，也需要一个"概念覆盖先验"来避免在某些 visual concept 上想不起来**。本质上 initial set 是 $\mathcal{C}$ 上的一个 importance sampling 分布，覆盖了 ImageNet/AudioSet 的常见概念。

## 四、实验数据深度解读

### Image captioning (MSCOCO Karpathy split)

| Method | BLEU4 | CIDEr | METEOR | SPICE |
|---|---|---|---|---|
| ZeroCap | 2.6 | 14.6 | 11.5 | 5.5 |
| ConZIC | 1.3 | 13.3 | 11.2 | 5.0 |
| CLIPRe | 4.6 | 25.6 | 13.3 | 9.2 |
| MeaCap<sub>TF</sub> | 7.1 | 42.5 | 16.6 | 11.8 |
| **MILS** | **8.0** | 33.3 | 15.0 | 9.6 |

注意：
- **CIDEr 输给 MeaCap 不少**（33.3 vs 42.5）。CIDEr 是 n-gram TF-IDF based，对"是否命中 ground truth 的具体词汇"敏感。MeaCap 有 memory module 显式 retrieve 关键 concept，词汇匹配更好。
- **METEOR / SPICE 接近 SOTA**。这两个 metric 对 semantic synonym 友好（METEOR 用 WordNet 同义词，SPICE 解析成 scene graph）。这说明 MILS 生成的 caption 语义对但用词不同——这正是 emergent zero-shot 的特征：模型没学过 MSCOCO 的"captionese"方言。
- **BLEU4 8.0 是所有方法里最高的**，说明 4-gram 精确度高，语法流畅（LLM 的语言先验功劳）。

### Video captioning (MSR-VTT)

| Method | Training data | CIDEr | METEOR |
|---|---|---|---|
| Nagrani et al. | HowTo100M | 0.5 | 8.23 |
| Nagrani et al. | VideoCC3M | 8.2 | 11.3 |
| **MILS** | (none) | 2.3 | **14.4** |

这里有个有意思的对比：在 HowTo100M（噪声大、自动 speech-to-text 对齐）上训练的 baseline CIDEr 只有 0.5——基本是垃圾。换 VideoCC3M（干净）后跳到 8.2。这说明 video captioning 对 training data 质量极敏感。MILS 完全不训练却达到 CIDEr 2.3 / METEOR 14.4，METEOR 还超过 VideoCC3M baseline，很说明问题：**ViCLIP 的 embedding 质量够好，LLM 的语言能力够强，中间不需要 supervised bridge。**

### Audio captioning (Clotho)

| Method | BLEU4 | ROUGE-L | METEOR | SPICE |
|---|---|---|---|---|
| ZerAuCap | 2.9 | 25.4 | 9.4 | 5.3 |
| **MILS** | 2.7 | 23.1 | **12.4** | **7.6** |

SPICE 涨了 43%（5.3→7.6），audio 这个 modality 上 MILS 优势明显。可能因为 audio captioning 的 baseline 本身就弱（ZerAuCap 也是 zero-shot），LLM 的语义先验在这里相对价值更大。

### T2I 改进

用 DrawBench 200 prompt，AMT 三人 majority vote：
- LDM: MILS 在 quality 和 faithfulness 上都被人类偏好（win rate > 50%）
- FLUX.1-schnell: 同样偏好 MILS

这其实印证了 [Imagen / DALL-E 3 paper](https://arxiv.org/abs/2307.01952) 的发现：**T2I 模型对 prompt 极其敏感，LLM rewrite 能显著提升**。DALL-E 3 是训练一个 captioner 生成详细 caption 再训练 diffusion；MILS 是 test-time 自动 rewrite。前者是 data curation，后者是 inference-time search。

## 五、Ablation 的几个关键发现

### 1. Optimization steps（Figure 9）

SCORER score（CLIP sim / PickScore）和 downstream metric（SPICE / human win%）**同步上升且 10-20 步收敛**。这很重要——说明 scorer 和人类判断高度 correlated，没有"scorer 高但人类不喜欢"的 reward hacking。原因可能是 sentence-level 优化空间太大，LLM 难以找到 adversarial 短语。

### 2. Initial set size（Figure 10）

1000 → 30000 initial captions，SPICE 从 ~7 涨到 ~9.6。这是 log-like 曲线，说明有 diminishing return 但确实关键。**这暴露了 MILS 的一个隐含依赖：它需要一个覆盖 visual concept 的 prior set。** ImageNet 1000 类够覆盖 MSCOCO 的常见物体，但如果做医学影像 captioning 就需要新的 prior set。

### 3. Generator / Scorer scaling（Figure 12）

- LLM: Llama 3.1 8B > Mistral 7B > Gemma2 9B（图里 Gemma 反而差，可能 instruction tuning 差异）。scaling 趋势明显。
- Scorer: SigLIP > MetaCLIP > CLIP > DFN（Table 4，SigLIP SPICE 9.7 最高）。

两个 module 都越大越好，且 LLM scaling 收益更大——这暗示 **language prior 是 bottleneck，scorer 已经够用了**。

## 六、Cross-modal arithmetic：text 作为 universal interface

这是 paper 最有想象力的应用。流程：

1. Image $\xrightarrow{\text{MILS + SigLIP}}$ text caption $c_{\text{img}}$
2. Audio $\xrightarrow{\text{MILS + ImageBind}}$ text caption $c_{\text{aud}}$
3. LLM combine: prompt "Image caption: {c_img}. Audio caption: {c_aud}. Generate combined caption" $\to c_{\text{comb}}$
4. $c_{\text{comb}} \xrightarrow{\text{T2I}}$ final image

对比 [ImageBind](https://arxiv.org/abs/2305.05665)：ImageBind 把 audio/image/text 都映射到 CLIP embedding 空间，然后 $\text{emb}_{\text{img}} + \text{emb}_{\text{aud}}$ 直接相加，喂给 DALLE-2 (因为 DALLE-2 用 CLIP embedding)。问题：**只能用兼容 CLIP embedding 的 T2I**。

MILS 的 inversion to text 绕过了这个限制——text 是任何 T2I 模型都能吃的 universal API。而且 LLM 的组合能力远比 embedding 加法灵活：embedding 加法是线性组合，LLM 能做语义级组合（"crane on grass" + "ocean waves" → "crane beside the shore with waves"）。

形式化对比：
- ImageBind: $f_{\text{T2I}}(E_{\text{img}}(x) + E_{\text{aud}}(y))$
- MILS: $f_{\text{T2I}}(\text{LLM}(\text{MILS}^{-1}(E_{\text{img}}, x), \text{MILS}^{-1}(E_{\text{aud}}, y)))$

MILS 多了 inversion 步骤但获得了组合灵活性和 T2I 自由度。

[Kazemi et al. 2024](https://arxiv.org/abs/2403.02580) 也做 CLIP inversion 但到 continuous image space（用 diffusion 直接从 embedding 生成 image）。MILS 到 discrete text 的好处是可解释、可组合、可喂给任何下游模型。

## 七、Style Transfer：non-differentiable scorer 的展示

这里 scorer 是经典 [Gatys 2015](https://arxiv.org/abs/1508.06576) 的 Gram matrix loss：

$$\mathcal{L}_{\text{style}} = \sum_{l \in \mathcal{L}} w_l \cdot \|G(F_l(I_{\text{gen}})) - G(F_l(I_{\text{style}}))\|_2^2$$

$$\mathcal{L}_{\text{content}} = \sum_{l \in \mathcal{L}_{\text{high}}} w_l \cdot \|F_l(I_{\text{gen}}) - F_l(I_{\text{content}})\|_2^2$$

其中 $F_l(I)$ 是 VGG19 第 $l$ 层 feature map，$G(\cdot)$ 是 Gram matrix $G_{ij} = \sum_k F_{ik} F_{jk}$（捕捉 channel 间 correlation，即 texture）。低层 $l$ 管 texture（style），高层管 content。

GENERATOR 是 Llama → Emu Edit（image editing model），LLM 输出 edit instruction（如 "make it look like a Van Gogh painting with swirling brushstrokes"），Emu Edit 执行。

**关键：LLM 从头到尾没看到 style image 也没看到 content image**，它只看到 "instruction + score pair"，通过 scorer 反馈推断"什么样的 edit instruction 能产生匹配 style 的 texture"。这非常 magic——LLM 在 blind 状态下学会了描述它看不到的视觉风格。

paper 也坦承限制：Gram matrix 对 fine-grained texture 不够分辨，LLM 的 style 词汇也有限。这是 scorer 上限决定的。

## 八、更宏观的直觉：inference-time compute as task-solver

把 MILS 放到更大的图景里看，它属于一个 emerging 趋势：**用 inference-time compute 替代 task-specific training**。

- [O1](https://openai.com/index/introducing-openai-o1-preview/): RL 训练 LLM 在 test time 多思考
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903): test-time reasoning 的零成本版本
- [LLM as optimizer (OPRO)](https://arxiv.org/abs/2310.03720): LLM 自己做黑盒优化
- [Self-Refine / Reflexion](https://arxiv.org/abs/2303.11366): LLM self-critique + revise
- **MILS**: LLM + external scorer 的迭代优化

MILS 的独特之处是 **引入外部 scorer 作为 grounding**。Self-Refine 的问题是 LLM 自己 critique 自己容易 hallucinate（没 grounding）。MILS 用 CLIP/PickScore 这种 perception-grounded scorer 提供 objective feedback，避免了 self-reinforcement 的幻觉。

更激进地，这暗示了一个 hypothesis：**如果 LLM 足够强 + 任意 verifier 足够准，那么任何 task 都不需要 supervised training，只需要 test-time search。** 这是 [Scaling test-time compute](https://arxiv.org/abs/2408.03314) 的极端版本。

当然现实里 scorer 本身是 trained 的（CLIP 用了 400M image-text pair），所以严格说是 "no task-specific training"，不是 "no training at all"。但 task-level 的 zero-shot emergence 已经很惊人。

## 九、与 diffusion 的有趣类比

可以把 MILS 看成 **caption 空间上的离散 diffusion**：

- Diffusion: 从 noise $z_T$ 出发，迭代 $z_{t-1} = z_t - \eta \nabla \log p(z_t)$，逐渐 sharpen 到 data manifold
- MILS: 从 random captions $C_0$ 出发，迭代 $C_{t+1} = \text{LLM}(\text{TopK}(C_t, S))$，逐渐 sharpen 到 high-scoring caption

scorer $S$ 类比 score function $\log p(x)$，LLM 类比 denoiser。区别是 MILS 在离散空间、用 black-box scorer、用 LLM 隐式近似 score gradient。

这也解释了为什么 10-20 步就收敛——和 diffusion 一样，前几步是 coarse alignment，后面是 fine-tuning。

## 十、批判性思考

### 优点
1. **概念优雅**：一个框架打天下，6 个 task 换 module 即可
2. **真正 emergent zero-shot**：不是 zero-shot 到新数据分布，是 zero-shot 到新 task 本身
3. **可解释**：每步的 candidate 都是人类可读 text，方便 debug
4. **可组合**：text 作为 universal interface，能接任何下游模型

### 限制
1. **速度**：10 步 × (LLM 生成 50 candidate + scorer 50 次 forward) ≈ 几十秒到分钟级，比 forward-only VLM 慢得多
2. **Scorer 上限**：style transfer 受 Gram matrix 分辨率限制；audio 受 ImageBind 质量限制
3. **隐含 prior**：30K initial set 依赖 ImageNet/AudioSet 类别，对新 domain 不一定 transfer
4. **CIDEr 不如专门方法**：在需要精确词汇匹配的 benchmark 上吃亏
5. **LLM 看不到 image**：对需要细粒度 spatial reasoning 的任务（"红色球在蓝色方块左边"）可能不行，因为 scorer 是 global similarity 而非 spatial

### 我的联想

- **与 AlphaCode**：generate-test-cluster 思路相通。AlphaCode 生成海量 candidate 用 tests 过滤；MILS 生成 candidate 用 scorer 过滤。区别是 AlphaCode 的 tests 是离散 pass/fail，MILS 的 scorer 是连续分数。
- **与 RLHF 的 PPO**：MILS 像是 test-time 的、black-box 版的 PPO。PPO 用 reward model + gradient 更新 policy weight；MILS 用 scorer + LLM in-context update 更新 candidate distribution。
- **与 [Decision Transformer](https://arxiv.org/abs/2106.01345)**：都是把 optimization 转成 sequence modeling。DT 把 RL 转成 next-token prediction；MILS 把 black-box opt 转成 LLM in-context generation。
- **与 [Constitutional AI](https://arxiv.org/abs/2212.08073)**：用 verifier 指导 generator，但 CAI 是训练 stage，MILS 是 inference stage。

### 未来方向猜测

1. **多模态 LLM 作为 generator**：如果 LLM 本身能看图（如 LLaVA / Pixtral / GPT-4o），那 generator 能直接 ground 到 image，scorer 只需要做 fine-tuning 而非 coarse search——速度可能快 10×。但那就不是 "blind LLM" 的精神了。
2. **用 RL 训练 LLM 更好地 propose**：像 O1 那样，训练 LLM 在给定 history 下生成更高 fitness 的 candidate。这能减少迭代步数。
3. **Self-consistency verifier**：多个 scorer 投票，避免单 scorer 的 bias。
4. **3D / spatial tasks**：paper 提到但没做。scorer 可以是 NeRF rendering + CLIP，或 3D shape descriptor。
5. **Active MILS**：LLM 主动 query scorer，类似 active learning，减少 scorer 调用次数。

## 十一、对你（Andrej）的可能 relevance

考虑到你过去的工作：

- **[Karpathy & Fei-Fei 2015 deep visual-semantic alignments](https://arxiv.org/abs/1412.2306)**：这篇是 MILS 评估用的 MSCOCO Karpathy split 的来源。当年你用 bidirectional LSTM + CNN alignment，需要 supervised image-caption pair。MILS 在同一个 split 上做 zero-shot——历史闭环。
- **[char-rnn / minGPT / nanoGPT](https://github.com/karpathy/nanoGPT)**：你的教育项目强调 minimalism。MILS 的精神高度一致——<100 行 core code 可能就能跑，比 ZeroCap 的 gradient engineering 简单一个数量级。
- **你的 "Software 2.0" 文章**：MILS 是 Software 2.0 的有趣变种——weights 不变，inference-time 用 search 替代 weight update。某种意义上是 "Software 2.0 + 1.5"：用 learned model 做 search operator。
- **你的 LLM101 / intro to LLM 教程**：MILS 是绝佳的教学案例，展示 in-context learning 的力量——LLM 通过 seeing examples（candidate + score）隐式学会了 fitness landscape，这是 in-context learning 的极致应用。

## 十二、Implementation 细节速查

- LLM: Llama 3.1 8B（instruct）
- Image scorer: SigLIP ViT-L/14（[WebLI training](https://arxiv.org/abs/2303.15343)）
- Video scorer: ViCLIP ViT-L/14（8 frames uniformly sampled）
- Audio scorer: ImageBind（[FAIR](https://arxiv.org/abs/2305.05665)）
- T2I scorer: [PickScore](https://arxiv.org/abs/2305.01569)
- T2I models: [LDM](https://arxiv.org/abs/2112.10752), [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- Image editor: [Emu Edit](https://arxiv.org/abs/2311.10089)
- Style scorer: VGG19 Gram matrix ([Gatys 2015](https://arxiv.org/abs/1508.06576))
- Iterations: 10 (captioning), 可调
- Top-K: 50
- Initial set: 30K (image/video), 50K (audio)
- Code: [github.com/facebookresearch/MILS](https://github.com/facebookresearch/MILS)

## 总结

MILS 是一个"对的地方戳了一刀"的工作。技术本身不复杂（evolution strategy + LLM + CLIP），但 insight 深刻：**multimodal understanding 的 bottleneck 不在 generation（LLM 已经够强），而在 grounding（scorer 提供），而 grounding 可以外包给现成的 embedding model，不需要 trained bridge。** 

它不是 SOTA on every metric（CIDEr 输给 MeaCap），但它是 **generality × simplicity × zero-shot-ness 的 Pareto frontier 上的一个点**，没有任何方法能在这么多 task/modality 上用同一个 training-free 框架达到这个性能。

更深层的，它预示了一个趋势：**当 LLM 足够强，inference-time search + external verifier 可能成为 task-solving 的默认范式**，task-specific training 退居 fine-tuning 角色。这跟 O1 的哲学遥相呼应，只是 O1 在数学/代码上验证，MILS 在多模态上验证。

---

**Reference links:**
- Paper: [arxiv.org/abs/2411.18448](https://arxiv.org/abs/2411.18448) (MILS)
- Code: [github.com/facebookresearch/MILS](https://github.com/facebookresearch/MILS)
- [ZeroCap (Tewel et al. 2022)](https://arxiv.org/abs/2206.14106)
- [MeaCap (Zeng et al. 2024)](https://arxiv.org/abs/2405.13069)
- [ImageBind (Girdhar et al. 2023)](https://arxiv.org/abs/2305.05665)
- [SigLIP (Zhai et al. 2023)](https://arxiv.org/abs/2303.15343)
- [PickScore (Kirstain et al. 2023)](https://arxiv.org/abs/2305.01569)
- [Gatys neural style (2015)](https://arxiv.org/abs/1508.06576)
- [OPRO: LLM as optimizer (Yang et al. 2023)](https://arxiv.org/abs/2310.03720)
- [Karpathy & Fei-Fei 2015](https://arxiv.org/abs/1412.2306)
- [Gandelsman et al. 2024 (CLIP neuron interpretation)](https://arxiv.org/abs/2406.04341)
- [Kazemi et al. 2024 (CLIP inversion)](https://arxiv.org/abs/2403.02580)
- [OpenAI O1](https://openai.com/index/introducing-openai-o1-preview/)
- [Scaling test-time compute](https://arxiv.org/abs/2408.03314)
