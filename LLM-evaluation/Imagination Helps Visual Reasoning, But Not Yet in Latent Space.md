---
source_pdf: Imagination Helps Visual Reasoning, But Not Yet in Latent Space.pdf
paper_sha256: 6e20f0d6d286a4a9119beed69809d337f134fc4677a614adf28164c62b61fb56
processed_at: '2026-08-05T09:10:11-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

## 一句话版本

**现在这帮做 latent visual reasoning 的paper，表面上看让模型在hidden state里"想象图像"很酷，但实际上那些latent tokens基本是摆设——换输入它们不变，改它们答案也不变，里面的visual信息也probe不出来。作者用text把"想象"显式写出来，反而效果好得多，因果性也强得多。**

---

## 这个领域在干嘛

最近multimodal LLM圈子有一波人想做"视觉想象"。想法很自然：人脑子里想一个红苹果，不需要真的看见红苹果，是在神经元层面replay视觉信号。那MLLM能不能也这样？在hidden state层面"想图像"，不decode出来，直接在latent space里reasoning。

这条路最有代表性的几个工作：
- **Mirage** (https://arxiv.org/abs/2506.17218)：把中间推理图像的visual feature压缩成几个latent tokens做supervision
- **LVR** (https://arxiv.org/abs/2509.24251)：直接用image feature supervise latent tokens
- **Monet** (https://arxiv.org/abs/2511.21395)：用distillation，让gradient只流过latent tokens，保留visual + textual semantics

这些paper都report了不错的结果，大家都觉得"latent visual reasoning是个promising paradigm"。

**但这篇paper的作者说：等等，我们先别急着celebrate，先搞清楚这些latent tokens到底在干嘛。**

---

## 作者怎么查的：Causal Mediation Analysis

借用了Judea Pearl的causal framework（https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf），把整个过程抽象成：

$$X \rightarrow Z \rightarrow Y$$

- $X$ = 输入（图片+问题）
- $Z$ = 中间的latent tokens
- $Y$ = 最终答案

然后做两个intervention：

**Intervention 1: 扰动$X$，看$Z$变不变**
- 如果$Z$真的在"看"输入，那换一张图、换一个问题，$Z$应该显著变化
- 结果：**几乎不变**。跨instance、跨task，同一位置的latent tokens cosine similarity极高。而且越往后生成越collapse，全都长一样。

**Intervention 2: 扰动$Z$，看$Y$变不变**
- 如果$Z$真的在"想"东西影响答案，那把$Z$换成随机噪声、换成固定tensor、甚至清零，答案应该崩
- 结果：**几乎不崩**。V*上甚至反而涨了0.5分。Mirage在最强intervention下才崩，但那是因为模型开始repeat退化，不是因为latent真的carry了信息。

**Probing analysis: $Z$里有visual信息吗？**
- 用同一张图但问不同属性的问题，看latent tokens能不能支持回答
- 结果：**比random guess还差**。latent tokens基本不encode visual semantics。

---

## 三个Finding连起来的故事

**Finding 1**: latent tokens跨instance高度相似 → 它们没在看输入
**Finding 2**: 改latent tokens对答案几乎无影响 → 模型没在用它们
**Finding 3**: latent tokens里visual信息极少 → 它们也没存什么有用的东西

三件事加起来：**latent tokens既不接收输入信息，也不输出信息给答案，自己也不存储信息。它们就是dead weight，是decorative placeholder。**

那模型为什么还能答对？因为模型走的是implicit shortcut，绕开了latent pathway，直接用原始image tokens + text reasoning把答案搞出来。latent tokens只是被attention"礼貌性地扫一眼"然后ignore。

---

## 为什么会这样？我的intuition

这里paper没深挖，但我自己有几个猜测，对你应该能共鸣：

### (1) Autoregressive hidden state回灌的collapse问题

LVR的inference公式是：
$$h_i = \mathcal{M}(E(x); y_{<i})$$

latent mode下，$y_{<i}$里的latent token是上一step的hidden state直接projection回来的，没有经过discrete bottleneck。这就像一个RNN没有reset gate，hidden state反复self-attend，容易塌进attractor state。

回忆一下continuous VAE的posterior collapse问题（https://arxiv.org/abs/1611.00742）：当decoder足够强，latent code就变成degenerate，什么都不encode。这里类似——LLM backbone足够强，latent tokens变成了被bypass的"装饰品"。

### (2) Train-inference mismatch

训练时，latent tokens有teacher model的visual feature做supervision，所以训练阶段它们确实被push去encode visual info。但inference时，这个supervision信号没了，model自生成的latent和训练时的target分布有gap，结果就是模型实际生成的latent和训练时target的latent对不上。

这就像RLHF里reward model在训练时用human pref，但部署时reward信号没了，policy可能drift。

### (3) Attention bypass

Transformer的attention是softmax，如果某些key/value对query的weight本来就很低，那把它们的value换成什么都没用。latent tokens在attention pattern里可能就是被ignore的那些head。作者做的intervention实验恰好印证了这点——改latent几乎不影响output，说明attention根本没在attend它们。

---

## CapImagine：用text做"想象"

既然latent space不行，那把"想象"显式verbalize成text。

### 核心idea

Monet的训练数据本来是interleaved的：
```
[图1] [问题] [text reasoning] [中间图像: zoom-in的region] [继续text reasoning] [答案]
```

CapImagine把中间图像换成它的textual caption：
```
[图1] [问题] [text reasoning] [caption: "zoom into the red cart in the center, which has 4 wheels and a man pushing it"] [继续text reasoning] [答案]
```

让model在text space通过explicit reasoning chain实现"想象"。

### 数据pipeline（Figure 3）

1. **Data Rewriting**: 用Qwen3-VL-4B给中间图像生成caption
   - Visual-CoT/Zebra-CoT子集：caption描述zoom-in region的visual semantics
   - Refocus/CogCoM子集：描述image manipulation前后的visual differences

2. **Global Refinement**: 用MLLM全局refine reasoning chain，让新caption平滑integrate

3. **Data Filtering**: 发现原始Monet-SFT-125K里Visual-CoT subset质量很差（答案和observation冲突、问题ambiguous），filter后只剩17k high-quality instances

### 为什么这个设计work

text tokens有几个latent tokens没有的property：
- **Discrete bottleneck**: 每个token必须从vocab里选，强迫model commit到一个具体语义
- **Causal attention**: 后续token显式attend前面的text，绕不开
- **Pretrained prior**: LLM对text reasoning chain的distribution有强prior，SFT只需要小幅adaptation

---

## 实验结果：text space碾压latent space

### 主表（Table 1）

| Model | V* | HR-Bench-4K | HR-Bench-8K | MME-RW-Lite | BLINK Jigsaw | TableVQA |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B (base) | 76.4 | 68.0 | 63.8 | 45.8 | 62.7 | — |
| LVR | 81.7 | 70.8 | 63.0 | 50.6 | 52.0 | — |
| Monet | 83.3 | 71.0 | 68.0 | 46.9 | 50.0 | 64.8 |
| **CapImagine** | **85.9** | **74.1** | **70.7** | **54.8** | **64.7** | **70.7** |
| Δ vs Monet | +2.6 | +3.1 | +2.7 | +7.9 | +14.7 | +5.9 |

特别亮眼：
- MME-RealWorld-Lite涨7.9分，Monet相对base几乎没涨，CapImagine大幅超越
- BLINK Jigsaw涨14.7分，这是spatial composition reasoning，text imagination对这种abstract reasoning帮助大
- TableVQA涨5.9分

### Causal验证：CapImagine的imagination是真的在用

作者对CapImagine做同样的causal mediation analysis：

**X→Z**: 跨instance的imagination tokens cosine similarity低，说明它们真的在看输入
**Z→Y**: 用Qwen3-32B故意把imagination content改成lead to错误结论的版本，performance从85.9暴跌到22.5（-63.4分）

这个对照特别关键：latent space改Z几乎无影响，text space改Z直接崩盘。**text imagination是causally necessary的，latent imagination是decorative的。**

### Ablation：数据质量 > 复杂pipeline

- **w/o Rewriting**: 把text caption换成单个`<think_image>` token → V*掉3.2分，证明text内容本身重要
- **w/o Filtering**: 用原125k数据直接SFT（中间image换成`<think_image>` token消除train-inference mismatch）→ 持续下降

最有意思的发现：**直接SFT on 125k (with mismatch fixed) 就能match Monet的performance**。Monet额外做了复杂的distillation + policy optimization，结果和简单SFT打平。这严重question了latent reasoning pipeline的必要性。

---

## 这篇paper的深层message

我觉得这篇paper真正在说的是：

### (1) Latent reasoning当前是emperor's new clothes

大家都觉得"在latent space reasoning"很elegant、很cognitive、很接近人类imagination。但实际上现有方法的latent tokens既不encode info也不被使用。整个field可能在做一件看起来漂亮但实际空转的事情。

### (2) Causal analysis应该成为standard evaluation

光看benchmark分数是不够的。Monet在V*上83.3分，看起来很好，但do(Z)后还是83.3分，说明这83.3分和latent tokens无关。如果我们只看benchmark，会被误导。

这就像当年NLP圈发现BERT的很多performance来自spurious correlation而不是真正的linguistic understanding（参考https://arxiv.org/abs/1903.10318）。Causal probing应该成为reasoning method的标准evaluation。

### (3) Text space的ceiling可能比想象高

作者在limitations里承认text granularity < high-dim latent space的理论capacity。但实验表明当前text space已经碾压当前latent space。这说明latent space的"理论优势"还没被任何方法realize。

这让我想起image generation领域：VAE的continuous latent理论capacity远超VQ-VAE的discrete codebook，但实际上VQ-VAE/DALL-E那套discrete token路线反而先work。discrete bottleneck有regularization benefit。

### (4) Data quality is king

从125k filter到17k，performance反而提升。这和LIMA（https://arxiv.org/abs/2305.11206）的"1000 examples就够了"哲学一致。当前MLLM的bottleneck是data quality不是data quantity，也不是training pipeline complexity。

---

## 对未来的implication

### Latent reasoning不是死路，但需要新机制

当前的supervision方式（用teacher visual feature直接supervise latent）显然不够。未来可能需要：
- **Information bottleneck regularization**: 强迫latent encode info，比如mutual information constraint
- **Discrete latent tokens**: 不在continuous space做，而是用VQ把latent quantize成discrete token（类似VQ-VAE）
- **Multi-step latent reasoning with explicit dependency**: 让后续latent必须attend前面latent，不能被bypass

### Causal mediation应该成为reasoning paper的标配

任何提出"reasoning in latent space"的paper，都应该报告do(Z)实验。如果改Z不影响Y，那这个reasoning就是decorative的。

---

## 给你的takeaway

如果你要在这个方向做事：

1. **别盲目信latent reasoning的benchmark分数**。先做causal analysis，看latent tokens是不是真的在用。
2. **Text-space reasoning是被低估的baseline**。简单、可解释、causally faithful。
3. **Data quality >> pipeline complexity**。Monet的distillation + RL pipeline不如简单SFT on filtered data。
4. **如果想做latent reasoning，必须解决posterior collapse**。可以借鉴continuous VAE的literature，比如β-VAE、InfoVAE那套regularization。

这篇paper的核心贡献不是CapImagine这个method（作者自己也承认它只是verification probe），而是**用causal analysis戳破latent reasoning的bubble**。这种"指出皇帝没穿衣服"的paper在当前field里非常needed。

参考：
- Paper: 搜"Imagination Helps Visual Reasoning, But Not Yet in Latent Space"
- Monet: https://arxiv.org/abs/2511.21395
- Mirage: https://arxiv.org/abs/2506.17218
- LVR: https://arxiv.org/abs/2509.24251
- Causal Mediation (Pearl): https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf
- Posterior Collapse in VAE: https://arxiv.org/abs/1611.00742
- LIMA (data quality): https://arxiv.org/abs/2305.11206
- BERT spurious correlation probing: https://arxiv.org/abs/1903.10318

---

这篇paper的核心问题是：**latent visual reasoning (LVR) 这种通过 MLLM hidden states 做"想象"的范式，latent tokens 到底有没有真正承载 visual semantics 和 reasoning 功能？** 作者用 Causal Mediation Analysis 框架做了系统诊断，发现 latent tokens 基本是"死"的（高度 homogenous、对 input 不敏感、对 output 无 causality），然后提出 text-space 的 CapImagine 作为对照，证明 imagination 确实 help visual reasoning，只是 not yet in latent space。

下面我详细拆解。

---

## 1. Problem Setup：为什么要质疑 Latent Visual Reasoning

LVR 的 idea 看起来很 elegant：既然 MLLM 里 image embedding 和 text embedding 已经 align 了，那不如让 model 在 hidden state 层面"想图像"，不 decode 出来，直接在 high-dimensional latent space 里 deliberate。代表工作有 Mirage、LVR、Monet。

但作者质疑：这些 latent tokens 真的 encode 了 visual 信息吗？真的在 reasoning chain 里起 causality 作用吗？还是只是一个 soft prompt / placeholder？

参考 Monet: https://arxiv.org/abs/2511.21395
参考 Mirage: https://arxiv.org/abs/2506.17218
参考 LVR: https://arxiv.org/abs/2509.24251

---

## 2. Causal Mediation Analysis 框架

作者把整个 reasoning 抽象成一条 causal chain：

$$X \rightarrow Z \rightarrow Y$$

- $X$: input，即 $\bar{X} = (\{I_i\}_{i=0}^N, q)$，包含 $N+1$ 张 image $I_i$ 和 question $q$
- $Z$: latent tokens（中间的 mediator）
- $Y$: final answer

然后做两套 intervention：
- $P(Z \mid do(X))$：扰动 input，看 latent 怎么变（测 $X \to Z$）
- $P(Y \mid do(Z))$：扰动 latent，看 answer 怎么变（测 $Z \to Y$）

这是 Pearl 的 Causal Mediation Analysis，参考：https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf

---

## 3. Inference 公式（Section 3.1）

LVR 的 inference process：

$$h_i = \mathcal{M}(E(x); y_{<i}), \quad y_0 = \emptyset \tag{1}$$

- $h_i$: 第 $i$ 步的 last hidden state
- $\mathcal{M}$: 整个 MLLM 的 forward 函数
- $E(x)$: input embedding（image tokens + text tokens）
- $y_{<i}$: 第 $i$ 步之前已经生成的所有 tokens（包括 latent tokens 的 hidden state 直接回灌）
- $y_0 = \emptyset$: 初始输入为空（即只有 input embedding，无 prior generation）

$$y_i = \mathbb{I}(i \in \mathcal{T}_L) \cdot \phi(h_i) + \mathbb{I}(i \notin \mathcal{T}_L) \cdot E(\text{Decode}(h_i)) \tag{2}$$

- $\mathcal{T}_L$: latent tokens 的 index set（哪些 step 是 latent mode）
- $\phi(h_i)$: optional projection layer（把 hidden state 投影成下一步的 input embedding）
- $\text{Decode}(h_i)$: 把 hidden state decode 成离散 text token id
- $E(\cdot)$: embedding lookup
- $\mathbb{I}(\cdot)$: indicator function，决定这一步是 latent mode 还是 text mode

**Intuition**: 在 latent mode 下，hidden state 直接被 $\phi$ 投影后回灌，不经过 vocab unembedding/embedding 的 round-trip；在 text mode 下，正常 decode→lookup。模型在输出 `<|latent_start|>` 后进入 latent mode，输出 `<|latent_end|>` 后退出。

---

## 4. Finding 1: Latent tokens 跨 instance 跨 task 高度相似，且逐步 collapse

### 实验设置
- 三个 baseline: Monet（distillation-based）、LVR（image feature supervision）、Mirage（task-specific，VSP 数据集）
- 从 V*、MME、OCRBench-v2、MME-RealWorld-Lite、TableVQA 各均匀采样 instance，共 100 个
- 两个分析维度：
  - **Inter-instance**: 在 fixed position 上跨 instance 采样 latent tokens，算 cosine similarity
  - **Intra-instance**: 单个 instance 内所有 latent tokens，再 cross instance 平均

### 关键观察

**Inter-instance**: 不同 instance、不同 task 在同一 position 的 latent tokens cosine similarity 极高。意味着 latent tokens 几乎不 encode input image/question 的信息，连 coarse task-level 区分都没有。而且随着 reasoning step 推进，similarity 越来越高，全部 degenerate 成 uniform representation。

对比：text tokens、image tokens、MLLM 在 input 之后的 inner representation 都 exhibit 低 similarity，说明它们 carry distinctive semantics。只有 latent tokens 是"死"的。

**Intra-instance**: 单个 instance 内，latent tokens 随 autoregressive step 推进，逐渐 collapse 进 high-similarity cluster。LLM backbone 对 latent state 的 modification 越来越小，导致 tokens converge。

- LVR 衰退最快，第 2 步就 collapse
- Monet 一开始 semantically rich，第 5 步开始 lose distinctiveness
- Mirage 因为压缩 lengthy visual tokens 进 few latent tokens，全程 distinctiveness 最差

**Intuition**: 这是一种 mode collapse 现象，类似 GAN 训练里的 collapse——hidden state 在自回归迭代下越来越像，最后变成一个常数。这让人想起 continuous space 的 self-attention without discrete bottleneck 容易产生 trivial fixed point。

参考 V*: https://arxiv.org/abs/2310.12971

---

## 5. Finding 2: 对 Z 根本性扰动，对 Y 影响极小

### 实验设置
对 Monet 和 Mirage 做 $do(Z)$：
- Monet: 把所有 latent tokens 在所有 position、所有 instance 强制设成同一个 shared tensor $\tau$
- Mirage: 更激进，包括 4 种 intervention：
  1. $Z_i = \tau$（所有 latent 用同一个固定 tensor）
  2. $Z_i = Z_i + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2)$（注入 Gaussian noise）
  3. $Z_i = \epsilon \sim \mathcal{N}(0, \sigma^2)$（全部替换成 Gaussian noise）
  4. $Z_i = \mu \approx 0$（设成接近 0 的小值）

公式里：
- $\tau$: 固定 tensor
- $\epsilon$: random Gaussian noise
- $\sigma^2$: noise variance
- $\mu$: 接近 0 的小值

### 结果（Table 4 in Appendix B）

**Monet**:

| Model | V* Overall | HR-Bench-4K | MME-RW-Lite |
|---|---|---|---|
| Monet | 82.7 | 71.1 | 46.9 |
| Monet do(Z) | 83.3 (+0.5) | 70.1 (-1.0) | 46.2 (-0.7) |

把所有 latent token 替换成同一个固定 tensor，V* 反而 **涨了 0.5 分**。这种根本性扰动只导致 marginal 变化，说明 latent tokens 对 final answer 几乎无 causal effect。

**Mirage**:

| Intervention | VSP Accuracy | Δ |
|---|---|---|
| Baseline | 64.2 / 77.0 | — |
| $Z_i = \tau$ | 64.0 / 77.2 | -0.2 / +0.2 |
| $Z_i + \epsilon$ | 64.0 / 76.7 | -0.2 / -0.3 |
| $Z_i = \epsilon$ | 64.5 / 76.2 | +0.3 / -0.8 |
| $Z_i = \mu \approx 0$ | 65.0 / 35.5 | +0.8 / -41.5 |

只有 stage-2 variant 设成 $\mu \approx 0$ 时才 dramatic decline（-41.5），但这是因为 intervention 太强导致 repetition degeneration。其他全部 negligible。

**Intuition**: 模型其实没在 attend 这些 latent tokens，它们可能被 attention 几乎 skip 了，或者被 positional 上"忽略"。模型走的是 implicit shortcut，绕开了 latent visual reasoning pathway。

---

## 6. Finding 3: Latent tokens 编码的 visual semantics 极有限

### Probing Analysis
对 Monet 做：
1. 从 V* 采样 question-image pairs $\{(I_i, q_i)\}_{i=0}^N$
2. 收集对应的 latent embeddings $\{Z_i\}_{i=0}^N$
3. 构造 30 个 multiple-choice derived questions $\{(Z_i, \tilde{q}_i)\}_{i=0}^N$，这些 $\tilde{q}_i$ 聚焦同一 image region 但 query 不同 attributes（比如原问 cart 颜色，derived 问 cart 有几个轮子、推 cart 的人头发颜色等，见 Appendix A 的 Figure 6）
4. 假设：如果 latent tokens 真的 encode 了 visual semantics，应该能支持回答 derived questions

### 结果
- 只用 latent tokens 作为输入：performance 比 text-only random guessing baseline 还差
- 提供原始 image：Monet 和 Qwen3-VL-32B 都达到 76.67% accuracy（验证了 derived questions 本身质量 OK）

**Intuition**: latent tokens 既不 carry 可被 probing 的 visual info，也不 carry 可被下游 task 利用的 semantics。它们更像是 soft prompt/placeholder，而不是 active carrier of visual imagination。

---

## 7. CapImagine：在 text space 做 imagination

### 核心动机
既然 latent space 失败，那把 interleaved multimodal reasoning 里的"中间图像"显式 verbalize 成文本，让 model 在 text space 通过 explicit reasoning chain 实现 imagination。

### Dataset Construction Pipeline（Figure 3）

**Step 1: Data Rewriting**

基于 Monet-SFT-125K 数据集，两类子集两种 rewriting：

- **Visual-CoT + Zebra-CoT subsets**（主要是 zoom-in 到 key region）：把 original question + highlighted image region 给 Qwen3-VL-4B，prompt 它生成 concise caption，refo cus highlighted visual semantics
- **Refocus + CogCoM subsets**（image manipulation 如 mark / draw auxiliary lines）：把 original image + manipulated image 都给 Qwen3-VL-4B，让它 describe visual differences，explicitly verbalize key info（如 marked 数值、highlighted 文本实体）

**Step 2: Global Refinement**

直接插入 rewritten text 容易产生 rigid transition 和逻辑不连贯，所以用 MLLM 全局 refine reasoning chain，让新 text description 平滑 integrate 进原 reasoning trajectory。

**Step 3: Data Filtering**

发现 Visual-CoT subset（占 Monet-SFT-125K 的 94.88%）质量很差：
- 原始 final answer 与新生成 visual observation 冲突
- 大量 question 过于 ambiguous 或根本 unanswerable

用 MLLM 做 quality assessment，filter 掉 flawed instance，最后 retain **17k high-quality instances**（从 125k → 17k）。

### 严格对照设计
为了和 Monet 公平比较：
- 同源数据（Monet-SFT-125K）
- 同 backbone（Qwen2.5-VL-7B）
- 同 codebase（Monet codebase）
- 同 hardware（8 × A800-80G，batch=1, grad accum=16）

参考 Qwen2.5-VL: https://arxiv.org/abs/2502.13923
参考 Qwen3-VL: https://arxiv.org/abs/2511.21631

---

## 8. Main Results（Table 1 & 2）

### 高分辨率感知 benchmarks

| Model | V* Overall | HR-Bench-4K | HR-Bench-8K | MME-RW-Lite | BLINK Jigsaw | BLINK MV | TableVQA Overall |
|---|---|---|---|---|---|---|---|
| GPT-4o | 67.5 | 59.0 | 55.5 | 52.0 | 55.3 | 59.4 | — |
| Qwen2.5-VL-7B | 76.4 | 68.0 | 63.8 | 45.8 | 62.7 | 42.9 | — |
| PixelReasoner | 80.6 | 72.9 | 66.9 | 49.7 | — | — | — |
| DeepEyes | 90.0 | 75.1 | 72.6 | 53.2 | — | — | — |
| LVR | 81.7 | 70.8 | 63.0 | 50.6 | 52.0 | 46.6 | — |
| Monet | 83.3 | 71.0 | 68.0 | 46.9 | 50.0 | 47.4 | 64.8 |
| **CapImagine** | **85.9** | **74.1** | **70.7** | **54.8** | **64.7** | **49.6** | **70.7** |

关键 deltas：
- HR-Bench-8K: +4.0% over Monet（70.7 vs 68.0）
- MME-RealWorld-Lite: +4.9% over Monet（54.8 vs 46.9）
- HR-Bench 平均: +3.44%
- V*: +2.6%
- BLINK Jigsaw: +14.7%（52.0 → 64.7），multi-view reasoning +2.2%
- TableVQA: +6.1%（64.8 → 70.7）

### CapImagine vs 工具方法
- CapImagine 大幅超过 PixelReasoner
- 略低于 DeepEyes（DeepEyes 用真实 image replay 有 complementary benefits）

### TableVQA 细分（Table 2）

| Model | VWTQ | VWTQ_syn | VTabFact | Overall |
|---|---|---|---|---|
| Monet | 55.3 | 60.4 | 78.8 | 64.8 |
| CapImagine | 60.9 | 68.0 | 83.2 | 70.7 |

VWTQ (Visual Wikipedia Table QA)、VWTQ_syn (synthetic)、VTabFact 三项全面提升。

---

## 9. Ablation Study（Table 1 末尾）

两个 ablation：

### (1) w/o Rewriting
把 text-space imagination descriptions 替换成单个 `<think_image>` token，相同 setting fine-tune：
- V*: 82.7（vs 85.9，-3.2%）
- HR-Bench-8K: 69.8（vs 70.7）
- 全部 benchmark 一致 degrade

证明：**text-driven imagination 是关键**，单纯的 placeholder token 不够。

### (2) w/o Filtering
直接在原 Monet-SFT-125K 上 fine-tune，把中间 image 替换成 `<think_image>` token（消除 train-inference mismatch）：
- V*: 82.7
- HR-Bench-8K: 69.3
- 持续下降，证明 quality filtering 必要

**关键 observation**: 去掉 train-inference mismatch 后，直接 SFT on Monet-SFT-125K 就能 match Monet 的 performance（Monet 还额外做了 Policy Optimization 阶段）。这进一步 question 了 latent 的作用——复杂 distillation + policy optimization 并没有比简单 SFT 好。

---

## 10. Dependency Analysis：CapImagine 的因果性验证

### X → Z 测试
对 CapImagine 的 text imagination tokens 做同样的 instance-level perturbation：
- Inter-instance cosine similarity: **低**（vs latent 的高）
- Intra-instance: consecutive hidden states diversity **大**

证明 CapImagine 的 imagination tokens 强依赖 input。

### Z → Y 测试
Intervention 协议（参考 Zhang et al. 2025b: https://arxiv.org/abs/2512.21711）：
1. CapImagine 先回答 question
2. 删除生成的 answer
3. 用 Qwen3-32B 故意 alter imagination content，让它 lead to incorrect conclusion
4. 把 corrupted reasoning trace 喂回 CapImagine，让它 complete generation 并输出 final answer

### 结果（Table 3）

| Model | V* Avg | V* Attr | V* Spa | HR-Bench-4K Avg | FSP | FCP |
|---|---|---|---|---|---|---|
| Qwen2.5-VL | 76.4 | 77.4 | 75.0 | 68.0 | 80.3 | 55.8 |
| CapImagine | 85.9 | 87.8 | 82.9 | 74.1 | 88.5 | 59.8 |
| CapImagine do(Z) | 22.5 | 20.0 | 26.3 | 24.0 | 20.0 | 28.0 |
| Δ↓ | **-63.4** | -67.8 | -56.6 | -50.1 | -68.5 | -31.8 |

V* 从 85.9 暴跌到 22.5（-63.4%），HR-Bench-4K 从 74.1 到 24.0（-50.1%）。**修改 imagination content 直接摧毁 performance**。

对比 latent space 的 intervention：几乎无影响。

**Intuition**: text-space imagination 真的是 reasoning chain 的 active carrier，causally 必要。而 latent tokens 是 decorative。

---

## 11. Efficiency Analysis（Figure 5）

在 V* 上测 decoding time：
- Monet: ~baseline
- CapImagine: 与 Monet **comparable**（即使 CapImagine 是更长 text sequence）
- DeepEyes: ~2× CapImagine 的 latency

CapImagine 在 effectiveness-efficiency trade-off 上 sweet spot。

---

## 12. 我的 Critical Reading / 想想 intuition

几个值得 deep dive 的点：

### (a) 为什么 latent tokens collapse？
自回归 hidden state 回灌会 collapse，类似 RNN without reset gate 在长序列上的 attractor 行为。问题可能是：
- 训练时 latent supervision（image feature）只在某一层注入，但 inference 时这个监督信号没了
- $\phi$ projection layer 是 frozen 还是 trainable？文中没明说
- Attention 机制下，如果后续 token 都 attend 同一个 latent，且 latent 自身没有 strong regularization，容易塌成 degenerate fixed point

参考 mode collapse in continuous VAE: https://arxiv.org/abs/1611.00742

### (b) Causal Mediation 的局限
作者用 $do(X)$ 和 $do(Z)$，但实际上：
- $do(Z)$ 直接替换 latent，可能破坏 attention 的 positional / structural 一致性，所以"模型 ignore 它"也可能是模型 ignore 了 strange perturbation，而不是 ignore 正常的 latent
- 不过作者用了 4 种 intervention（包括 minimal noise $+\epsilon$），结果一致，这个 confound 较好控制

### (c) Data filtering 的 confound
CapImagine 17k vs Monet-SFT-125K（125k）数量差 7 倍。虽然作者声称 17k high-quality 更好，但这个 comparison 本身就说明：**可能不是 text vs latent 的胜利，而是 data quality 的胜利**。

不过 ablation 里"w/o Filtering"用 125k 训练，performance 也只是 marginal 比 17k 差，说明质量确实更关键。但和 Monet 比，Monet 用了同样 125k 数据 + policy optimization，CapImagine 用 17k rewritten data，control 还不够严格（作者自己说 "strictly controlled setting"，但 backbone、数据源同，数据量不同，这其实是 partially controlled）。

### (d) Probing analysis 的设计
derived questions 聚焦同一 region 但 query 不同 attribute，这个 setup 不错。但 latent tokens 只在 Monet 的 multi-stage training 下被鼓励 encode visual semantics，而 inference 时这些 latent 是 model 自生成的，可能和 training 时的 target latent 分布有 gap（train-inference mismatch，作者后面也提到了）。所以 latent 不 encode visual info 也可能是 inference-time distribution shift 导致。

### (e) Text-space 的 ceiling
作者在 Limitations 里诚实承认：natural language 的 granularity < high-dim latent space 的理论 information capacity。所以这个 paper 更像 negative result + diagnostic tool，而不是 latent reasoning 的 tombstone。未来如果有人能构造真正 causal 的 latent reasoning chain，可能还是能超越 text-space。但当前 latent methods 都没做到。

参考 VoT (Visualization-of-Thought): https://arxiv.org/abs/2501.07542

---

## 13. 整体 contribution 总结

1. **Negative diagnostic**: 系统地用 Causal Mediation Analysis 揭示 latent visual reasoning 中 latent tokens 的三重失效（input-disconnect、output-disconnect、semantically-empty）
2. **Positive baseline**: CapImagine 证明 text-space imagination 工作良好，且 causally necessary
3. **Methodological insight**: 数据质量 > 复杂 distillation + RL pipeline；simple SFT on filtered data 就能 beat 复杂 pipeline

**最大 takeaway**: 在 visual reasoning 里，"想象"这个 cognitive metaphor 当前用 explicit text 实现比用 latent space 实现更靠谱。Latent space methods 看起来漂亮，但可能只是 sophisticated placeholder，没有真正的 information flow。

参考论文链接：
- Paper 本身（推测 arxiv ID）: 建议搜 "Imagination Helps Visual Reasoning, But Not Yet in Latent Space"
- Monet: https://arxiv.org/abs/2511.21395
- Mirage: https://arxiv.org/abs/2506.17218
- LVR: https://arxiv.org/abs/2509.24251
- Causal Mediation (Pearl): https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf
- V*: https://arxiv.org/abs/2310.12971
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- DeepEyes: https://arxiv.org/abs/2505.14362
- PixelReasoner: https://arxiv.org/abs/2505.15966
- CogCoM: https://arxiv.org/abs/2402.11211
- Visual-CoT: https://arxiv.org/abs/2402.14874
- Zhang et al. latent adversarial: https://arxiv.org/abs/2512.21711
