---
source_pdf: VL-JEPA.pdf
paper_sha256: f3218282df13e288d07056e05a50e1306e4cb2b87cd7292538c9da9daf3d7439
processed_at: '2026-08-13T02:38:44-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VL-JEPA 用人话说

Andrej，我用最 plain 的语言重新讲一遍，但技术细节还是会保留。

---

## 一句话总结

**传统 VLM 是"看到画面 → 一个字一个字吐答案"，VL-JEPA 是"看到画面 → 脑子里形成答案的意思 → 需要说话时才翻译成文字"**。

就这么简单。其余的所有 fancy 东西都是围绕这一个 idea 展开的。

---

## 传统 VLM 的毛病在哪

假设你问模型："这张图里发生了什么？"

正确答案可能是：
- "the lamp is turned off"
- "the room will go dark"
- "the light just got switched off"
- "灯关了"

这四句话**意思完全一样**，但是**字完全不同**。

传统 VLM 训练的时候，loss 是在 token space 算的。也就是说，模型要被强迫去学会：
- 这四句话都是对的
- 但它们在 token space 里互相 orthogonal（因为共享 token 极少）

这就好比让一个学生背同一个知识点的四种不同表述，还要他知道这四种都对。这学生大部分精力花在"背表述"而不是"懂知识"。

更糟的是，模型还要学会 surface-level 的东西：什么时候用 "lamp" 什么时候用 "light"，什么时候说 "turned off" 什么时候说 "got switched off"。这些对**任务正确性完全没影响**，但模型必须建模它们，因为它要 generate token。

LeCun 早就吐槽过这个：**generative model 在 pixel/token space 做预测，是在 modeling 一大堆你根本不关心的东西**。

---

## VL-JEPA 的解法：搬到 embedding space

核心 idea：**别预测 token，预测 embedding**。

Y-Encoder 会把 "the lamp is turned off"、"the room will go dark"、"灯关了" 都映射到 embedding space 里**附近的几个点**（因为它们语义一样）。于是原来 token space 里四个 disjoint 的 high-density regions，在 embedding space 就塌缩成一个 compact 的 mode。

学生现在不用背四种表述了，只需要学到"这个意思"。具体怎么说，交给后面的 Y-Decoder 去处理，而且 Y-Decoder **训练时根本不用**，只有 inference 时需要才调用。

---

## 架构四件套（用比喻）

想象一个翻译场景：

| 组件 | 角色 | 比喻 |
|---|---|---|
| X-Encoder | 把画面变成视觉特征 | 眼睛 + 视觉皮层，看到画面 |
| Predictor | 综合"画面 + 问题"想出答案的意思 | 大脑想"啊，灯关了" |
| Y-Encoder | 把答案文本变成 embedding（训练 target） | 训练时的"标准答案的意思" |
| Y-Decoder | 把预测的 embedding 翻译成文字 | 嘴巴，只在需要说话时才用 |

训练时：X-Encoder + Predictor + Y-Encoder 一起学，Y-Decoder 站旁边看戏。
推理时：X-Encoder + Predictor 输出 embedding stream，**只在需要文字时**才叫 Y-Decoder 出来翻译。

---

## 具体怎么实现

### X-Encoder
- **Frozen V-JEPA 2 ViT-L**（304M params）
- 这是 LeCun 团队自己的 self-supervised video model，本来就是 JEPA 家族
- 输入 16 frames @ 256×256
- 输出 visual token sequence

为什么 frozen？因为 vision encoder 已经很强了，不必再 fine-tune，省 compute。

### Predictor
- 用 **Llama-3.2-1B 的最后 8 层 Transformer**（490M trainable params）
- 关键修改：**关掉 causal mask，改成 bidirectional attention**

为什么要关 causal mask？因为 query token 是 append 在 visual tokens 后面的，如果用 causal mask，visual tokens 就 attend 不到 query。VQA 任务里这是 fatal 的——你问"图里有几个人？"，visual tokens 不看问题怎么回答？

Ablation 显示：保留 causal mask 让 VQA 掉 1.9 points。

为什么用最后 8 层而不是全部 16 层？因为前面层是 low-level token processing，后面层是 high-level reasoning。用最后 8 层是 efficiency sweet spot。Ablation 显示全 16 层只比最后 8 层多 0.8 retrieval / 3.0 VQA，但参数翻倍。

### Y-Encoder
- **EmbeddingGemma-300M**（Google 的 text embedding model）
- 关键：**learning rate multiplier = 0.05**（即 Y-Encoder 学得比 Predictor 慢 20 倍）

为什么 Y-Encoder 要慢慢学？因为训练开始时 Predictor 输出的 embedding 是垃圾，如果 Y-Encoder 学得太快，会被这些 garbage target 带偏。所以让 Predictor 先慢慢稳定下来，Y-Encoder 再跟着 align。

Ablation 这个 multiplier 极其 sensitive：
- ×0.00（frozen）：VQA 41.4
- ×0.01：VQA 41.0
- **×0.05：VQA 42.5**（sweet spot）
- ×0.10：VQA 42.9（也不错）
- ×1.00：VQA 40.7

差几个点全在这。

### Y-Decoder
- Paper 没细说怎么训练的——这是 paper 的一个 gap
- 推测：可能用 Y-Encoder 的某种 distillation 或 reverse mapping
- 只在 inference 时调用，把 Ŝ_Y decode 成 text

---

## 训练 loss：Bidirectional InfoNCE

$$\mathcal{L} = -\frac{1}{2}\left[\log \frac{\exp(\text{sim}(\hat{S}_Y, S_Y)/\tau)}{\sum_{k} \exp(\text{sim}(\hat{S}_Y, S_Y^{(k)})/\tau)} + \log \frac{\exp(\text{sim}(S_Y, \hat{S}_Y)/\tau)}{\sum_{k} \exp(\text{sim}(S_Y, \hat{S}_Y^{(k)})/\tau)}\right]$$

变量解释：
- $\hat{S}_Y$：Predictor 输出，shape 是 [batch, 1536]
- $S_Y$：Y-Encoder 输出，shape 也是 [batch, 1536]
- $S_Y^{(k)}$：batch 里第 k 个 sample 的 Y-Encoder 输出（作为 negative）
- $\hat{S}_Y^{(k)}$：batch 里第 k 个 sample 的 Predictor 输出
- $\tau$：temperature，控制 contrastive 的 hardness
- $\text{sim}$：cosine similarity
- 第一项：让 Predictor 输出靠近对应 Y-Encoder 输出，远离 batch 内其他
- 第二项：反方向，让 Y-Encoder 输出也靠近对应 Predictor 输出

为什么用 InfoNCE 而不是 L2？InfoNCE 自带 anti-collapse：它既让正样本对靠近（alignment），又让 batch 内样本互相远离（uniformity）。如果只用 L2/cosine，模型可以直接 collapse 成输出常量，loss 看起来很低但其实啥都没学。

Ablation 验证了这个：
- InfoNCE：Classification 23.3, Retrieval 30.3
- Cosine loss：Classification 16.5（崩）
- L2 loss：Classification 13.5（更崩）
- L1 loss：Classification 14.8（崩）

但有趣的是 **Cosine loss 在 VQA 上反而比 InfoNCE 高 2.3 points**（46.6 vs 44.3）。Paper 解释：VQA 的 answer space 比 classification 窄，collapse 风险小，所以纯 alignment 反而更直接。但 classification/retrieval 必须 space 大，所以 InfoNCE 整体更优。

---

## 两阶段训练

### Stage 1: Pretraining（建立 vision-language 对齐）
- 数据：DataComp + YFCC-100M（image-text）+ Action100M（video-text）
- **没有 query**，只学"画面 → caption 的 embedding"
- 先 image-only（100k iters，batch 24k）
- 再 video（8 frames × 60k iters + 32 frames × 10k iters）
- 4 weeks on 24 nodes × 8×H200
- 输出模型叫 **VL-JEPA_BASE**

### Stage 2: SFT（加 VQA 能力）
- 数据：PLM mixture（25M VQA + 2.8M captioning + 1.8M classification + downsampled pretraining data）
- 83k steps，batch 3072，~2.5 days
- 输出模型叫 **VL-JEPA_SFT**

为什么 SFT 数据要 mix classification 和 captioning？因为如果只学 VQA，classification 和 retrieval 会 catastrophic forgetting。Ablation Table 7(a) 显示：去掉 pretraining 直接 SFT，classification 掉 21.7 points，retrieval 掉 17.3 points。

---

## 最 important 的实验：Controlled Comparison

Section 4.5 是 paper 的核心证据。

Setup：VL-JEPA vs token-prediction VLM，**唯一区别是 prediction 在 embedding 还是 token space**。

控制变量：
- 同一个 vision encoder：frozen PE ViT-L-14, 336², 16 frames
- 同样 training iterations
- 同样 batch size 128
- 同样 LR schedule
- 同样 data
- VL-JEPA 用 0.5B predictor，VLM 用 1B LLM（VLM 参数还更多）

结果（Fig. 3）：

| 训练样本 | VL-JEPA CIDEr | VLM CIDEr | VL-JEPA Top-5 | VLM Top-5 |
|---|---|---|---|---|
| 500K | 1.23 | 1.35 | 14.9% | 14.0% |
| 5M | 14.7 | 7.1 | 35.3% | 27.2% |
| 15M | 14.8 | 7.1 | 41.0% | 27.2% |

读这张表的方式：
- 500K 时两者差不多（VLM 甚至略好）
- 但很快 VL-JEPA 把 VLM 甩开
- 15M 时 VL-JEPA 用一半参数、同数据，CIDEr 是 VLM 的 2 倍，classification 高 14 points

**这就是 embedding-space supervision 比 token-space supervision 更 efficient 的硬证据**。

intuition：因为 target distribution 简化了，模型不用学 surface language variability，sample efficiency 自然高。

---

## Selective Decoding：streaming 场景的杀手锏

这是 paper 最 practical 的部分。

### 问题
传统 streaming VLM 每帧都得 autoregressive decode，token-by-token，latency 高。比如 smart glasses 一直录像，每秒要输出描述，传统 VLM 每 0.5 秒 decode 一次都很吃力。

### VL-JEPA 的优势
因为输出是 continuous embedding stream，可以一直监控 embedding 的变化。**只有当 embedding 显著变化时才 decode 成文字**。

具体做法：
1. 对 embedding stream 做 average pooling smoothing
2. 用 **agglomerative clustering with temporal connectivity**（一种 hierarchical clustering，要求相邻时间点才聚类）
3. 用 Ward distance 度量 intra-segment 一致性
4. 在每个 segment 的中点 decode 一次

### 实验
EgoExo4D，218 个长视频，平均 6 分钟，每个视频平均 143 个动作标注。

对比：
- **Uniform sampling**：固定间隔 decode（传统方法）
- **Selective decoding**：自适应选 decode 点（VL-JEPA 独有）

频率 sweep：从 2Hz 到 0.01Hz。

结果：selective decoding 在 0.35Hz 匹配 uniform 在 1Hz 的 CIDEr，**减少 ~2.85× decode cost**。

intuition：视频里大部分时间是"同一动作在持续"，比如"切菜切菜切菜"，embedding 没变化，不需要 decode。只有动作切换时（"切菜 → 炒菜"）才需要 decode。

这个对 smart glasses / robotics / live streaming 都是 game changer。

---

## Multitasking：一套架构做四件事

| Task | 怎么做 |
|---|---|
| Captioning | Predict Ŝ_Y → Y-Decoder → text |
| Open-vocab Classification | 把候选 label 用 Y-Encoder 编码，选 cosine 距离最近的 |
| Text-to-video Retrieval | 把候选 video 用 captioning prompt 编码，rank by similarity |
| Discriminative VQA | 把候选 answer 编码，选离 Ŝ_Y 最近的 |

关键：**所有 task 用同一套 weights，不需要改架构**。

这是 embedding space 的天然优势——classification 和 retrieval 本来就是 embedding space 的 native 操作。传统 VLM 做 retrieval 很别扭，因为它是 generative 的，要把"候选 video"decode 成文字才能比；VL-JEPA 直接在 embedding space 比就行。

---

## 实验结果亮点

### Zero-shot Classification + Retrieval（Table 1）

VL-JEPA_BASE 用 3.6B samples、1.6B params：
- Classification avg (8 datasets)：**52.5**（PE-Core-G 是 44.7，用 86B samples、2.3B params）
- Retrieval avg R@1 (8 datasets)：**63.7**（PE-Core-G 是 58.1）

**用 24× 更少 data、1.4× 更少 params，全面超越**。

特别注意 VL-JEPA 在 motion-centric 任务（SSv2, EK-100, EgoExo4D）特别强，因为 frozen V-JEPA 2 vision encoder 是 video-trained。但在 appearance-centric（Kinetics-400）相对弱。

### VQA（Table 2）

VL-JEPA_SFT 1.6B：
- GQA: 61.5（LLaVA-1.5-13B 是 62.0，接近）
- TallyQA: 69.9（PaliGemma-3B 是 76.8，输）
- POPE: 85.7（LLaVA-1.5-13B-HD 是 86.3，接近）
- POPEv2: 86.3（Qwen2-VL-2B 是 91.3，输）

VQA 上 competitive 但不是 SOTA。Paper 诚实承认：VL-JEPA 不是为 reasoning 设计的。

### WorldPrediction-WM（Table 3）—— 最 striking

任务：给初始和最终状态图片，从 4 个 candidate video clips 选正确的 action transition。

| Model | Accuracy |
|---|---|
| GPT-4o | 52.0 |
| Claude-3.5-Sonnet | 53.3 |
| Gemini-2.0 | 55.6 |
| **VL-JEPA_BASE 1.6B** | **63.9** |
| **VL-JEPA_SFT 1.6B** | **65.7** |

VL-JEPA 用 1.6B 参数**击败所有 frontier LLM**。这是 paper 最强的卖点。

intuition：world modeling 这种任务本质是 perception + action understanding，不需要 language reasoning。embedding space alignment 在这里完胜 text reasoning。

### Action Anticipation（Table 4, 5）

EPIC-KITCHENS-100（1s anticipation，Recall@5）：
- V-JEPA2 ViT-g-384px（更大 encoder）：39.7
- V-JEPA2 ViT-L-256px：32.7
- **VL-JEPA ViT-L-256px（同 encoder）：34.2**

用同一个 encoder，VL-JEPA 比 V-JEPA2 高 1.5 points。证明 language supervision 帮了 video representation。

COIN next step forecasting：
- VideoLLM-online 8B: 49.1
- VideoLLM-MoD 8B: 49.7
- ProVideLLM 8B/11+: 53.6
- **VL-JEPA 1.6B: 56.2**（新 SOTA）

用 1/5 的参数超 8B VLMs。

---

## Y-Encoder 评估（Table 6）—— 一个谜团

用 SugarCrepe++ 和 VISLA 测 Y-Encoder 对 hard negatives 的鲁棒性。

| Model | SugarCrepe++ | VISLA |
|---|---|---|
| CLIP ViT-L | 44.5 | 34.5 |
| SigLIP2 ViT-g | 56.5 | 40.4 |
| PE-Core ViT-G | 58.6 | 38.3 |
| **VL-JEPA_BASE** | **63.9** | **42.9** |
| VL-JEPA_SFT | 58.4 | 39.5 |

VL-JEPA_BASE 的 Y-Encoder 比 PE-Core-G 强 5.3 points。**说明 JEPA architecture 不仅提升 Predictor，也提升了 Y-Encoder**。这是 collaborative learning 的好处——Predictor "拉" Y-Encoder，两边互相提升。

但有个谜：**VL-JEPA_SFT 的 Y-Encoder 反而比 BASE 弱 5.5 points**。

Paper 没解释。我的猜测：SFT 数据是 VQA-heavy 的，answer distribution 窄，让 Y-Encoder 变 "narrow"，损失了 general text understanding。可能解法是 multi-stage 解耦，让 Y-Encoder 在 SFT 时 frozen 或慢学。

---

## Ablation 的重要细节

### 为什么 bidirectional attention 这么重要？

如果用 Llama 原始的 causal mask，query tokens 在 visual tokens 后面，visual tokens 看不到 query。VQA 任务里这是 fatal：你问"图里有几个人"，visual tokens 不看问题怎么回答？

Ablation：保留 causal mask 让 VQA 掉 1.9 points。但 interesting 的是，classification 和 retrieval 反而**略升**（+0.6 / +1.0）——因为这两个任务不需要 query。

### Llama-3 initialization 的作用

w/ Llama-3 init：VQA 42.5
w/o Llama-3 init：VQA 40.6（-1.9）

Llama 的语言先验对 VQA 重要，但对 pure vision-language alignment 反而略降（classification -0.8, retrieval -0.2）。说明 Llama 的 LM 先验可能 over-regularize alignment 任务。

---

## 我的几点 critical 思考

### 1. Y-Decoder 怎么训练的？

Paper 说 "not involved during the main training phase"，但 inference 时怎么 decode？是 distillation？reverse mapping？还是 separately trained decoder？这个 gap 在 paper 里很模糊。

如果 Y-Decoder 是 frozen Y-Encoder 的逆，那 generation quality 受限。如果是 separately trained，那训练 cost 没省。

### 2. Long-form generation 怎么办？

所有 generation 实验都是 captioning 或 short VQA answer（几个 token）。如果一个 embedding decode 成 50 token 的长答案，single point embedding 信息够吗？

我猜 VL-JEPA 在 long-form generation 上会 struggle，因为 1536 维 embedding 表达力有限。可能需要 sequence of embeddings 或 hierarchical embedding。

### 3. Reasoning 的天花板

Paper 自己承认：reasoning / tool use / agentic tasks 上 VLMs 仍然 excel。VL-JEPA 是 "perception-friendly"，不是 "reasoning-friendly"。

这跟 LeCun 的 latent reasoning 大方向（COCONUT, Large Concept Models）形成对照：latent reasoning 需要 sequential latent states，而 VL-JEPA 是 single-shot prediction。

下一步可能是：**在 VL-JEPA 基础上加 latent reasoning loop**——让 embedding stream 在 inference 时多步 evolve，而不是 one-pass 输出。

### 4. Scale 没充分探索

1.6B params，3.6B pretraining samples，跟 frontier VLMs 差两个数量级。如果 scale 上去，embedding prediction 是否还能保持 efficiency advantage？还是会遇到 bottleneck？

我猜：embedding prediction 在 perception-heavy 任务上 advantage 会保持，但在 reasoning-heavy 任务上不会自动改善，因为 single-shot embedding 表达力有限。

### 5. Y-Encoder SFT 退化的谜团

VL-JEPA_SFT 的 Y-Encoder 在 hard negative 上比 BASE 弱 5.5 points。这说明 SFT data 让 Y-Encoder 变 narrow 了。

可能的 fix：
- SFT 时 freeze Y-Encoder
- SFT 时 multi-stage：先只 train Predictor，Y-Encoder 冻
- 用 regularizer 保持 Y-Encoder 的 uniformity

---

## 跟 CLIP / VLM 的本质区别

| | CLIP | VLM | VL-JEPA |
|---|---|---|---|
| 怎么学 | 对齐 embedding | 预测 token | **预测 embedding** |
| Generation | 不行 | 行 | 行（需要 decoder） |
| Retrieval | 行 | 不行 | 行 |
| Streaming | 行（只 encode） | 不行（slow decode） | **行（selective decode）** |
| Anti-collapse | Contrastive | 不需要 | InfoNCE / EMA / freeze |
| 推理 cost | 最低 | 最高 | 中等（可选 decode） |

VL-JEPA 的 sweet spot：**兼有 CLIP 的 embedding efficiency 和 VLM 的 generation 能力**。

---

## 这篇 paper 的真正 contribution

不是"超越 SOTA"——它在很多 VQA 上没超。真正的 contribution 是：

**证明 embedding-space supervision 在 perception-heavy vision-language 任务上比 token-space supervision 更 efficient**。

这是一个 strong claim，用 controlled experiment 证明了。如果这个 claim 成立，那么：
- 未来 VLM 训练可能 split 成"embedding-space perception trunk" + "token-space reasoning head"
- Streaming/real-time 应用应该用 embedding-space prediction
- Long-horizon planning/world modeling 可能用 embedding state 更 natural

LeCun 的 JEPA 哲学终于从 vision-only 延伸到 vision-language，这是重要一步。

---

## 相关链接

- V-JEPA 2 (X-Encoder 来源): https://arxiv.org/abs/2506.09985
- I-JEPA (前作): https://arxiv.org/abs/2301.08243
- Perception Encoder (baseline): https://arxiv.org/abs/2504.13181
- EmbeddingGemma (Y-Encoder): https://arxiv.org/abs/2509.20354
- WorldPrediction benchmark: https://arxiv.org/abs/2506.04363
- COCONUT (latent reasoning 对照): https://arxiv.org/abs/2412.06769
- Large Concept Models: https://arxiv.org/abs/2412.08821
- LeCun 2022 (JEPA 哲学源头): https://openreview.net/pdf?id=BZ5a1r-kVsf
- VICReg (alternative regularization): https://arxiv.org/abs/2105.04906
- Wang & Isola 2020 (InfoNCE 分解): https://arxiv.org/abs/2005.10242

---

## TL;DR

VL-JEPA = "别预测 token，预测 embedding"，这个 simple idea 在 controlled experiment 上用一半参数、同数据量，CIDEr 翻倍、classification 高 14 points。代价是 reasoning 能力受限。对 real-time streaming 是 game changer（2.85× decode cost reduction）。对 LeCun 来说，这是 JEPA 哲学在 vision-language 的第一个正经 instantiation。

下一步我赌：**latent reasoning on top of VL-JEPA**，让 embedding 在 inference 时多步 evolve 而不是 single-shot。这是 bridge perception 和 reasoning 的关键。

---

# VL-JEPA: 在 Embedding Space 做 Vision-Language 的尝试

Andrej,这篇 paper 我觉得挺有意思,因为它是 Yann LeCun 长期以来 JEPA 哲学在 vision-language 领域的延伸。我尽量把 intuition 和技术细节都拆解给你看。

---

## 1. 这篇 paper 想解决什么问题

传统 VLMs 在 **token space** 做 autoregressive generation,本质上是在解一个 ill-posed 的问题。Paper Section 2 给了一个非常关键的例子:

> 给定 query "What will happen here if I flip this light switch down?",plausible answers 可以是 "the lamp is turned off" 也可以是 "room will go dark"。在 **one-hot token space** 里,这两个序列几乎 **orthogonal**,因为它们共享的 token 极少。

这就引出了一个根本问题:**在 token space 训练,模型被迫去 fit 多个 disjoint high-density regions**,这些 regions 在 token space 互相距离很远,但语义上其实是同一个 modality。这种 learning signal 很 noisy,而且 model 还要额外 capture surface linguistic variability(paraphrasing、word choice、style),这些都不影响 correctness。

VL-JEPA 的核心 idea:**把 prediction target 从离散 token space 移到 continuous embedding space**。在 embedding space,Y-Encoder 可以把 "the lamp is turned off" 和 "room will go dark" 映射到 nearby points,从而把多个 plausible targets 压成一个 compact unimodal distribution。

---

## 2. 架构设计:四个组件

```
X_V → X-Encoder → S_V ↘
                        → Predictor → Ŝ_Y → Y-Decoder (only when needed) → Ŷ
X_Q (tokenized)  ──────↗                  ↑
                                           Y-Encoder ← Y (training target)
```

### 2.1 X-Encoder
- **Frozen V-JEPA 2 ViT-L**(304M params,self-supervised video model)
- 输入 16 frames @ 256² resolution(单 image 复制 16 份)
- 输出 sequence of visual tokens

为什么用 V-JEPA 2?因为它同时 excel at image and video,而且 V-JEPA 2 本身就是 JEPA 家族的一员,representation 已经自带 latent-space 哲学。

### 2.2 Predictor(核心组件)
- **Llama-3.2-1B 的最后 8 层 Transformer**(490M trainable params)
- Tokenizer 和 token embedding 也来自 Llama-3.2-1B
- **关键修改**:disable causal attention mask,改成 bidirectional attention
  - 因为 query tokens 是 append 在 visual tokens 后面,如果用 causal mask,visual tokens 就 attend 不到 query,严重损害 VQA 性能(ablation 显示 -1.9 VQA accuracy)
- 输入:visual embeddings + tokenized query embeddings(用 [PAD] padding 到 512)
- 输出:对 non-[PAD] tokens 做 average pooling,再 linear projection 到 1,536 维 shared embedding space

为什么用 Llama 的最后 8 层?因为这部分主要是 "high-level reasoning" 的 layers,而前面 0-8 层主要是 "low-level token processing"。Ablation Table 7(d) 显示:Layer 0-16(全 16 层)只比 Layer 8-16 在 retrieval 上好 0.8,在 VQA 上好 3.0,但 parameters 翻倍。所以 last 8 layers 是 efficiency sweet spot。

### 2.3 Y-Encoder
- **EmbeddingGemma-300M** 作为初始化
- Max context length 512
- **LR multiplier = 0.05**(关键 ablation!Table 7(b))
  - ×1.00:VQA -1.8
  - ×0.10:VQA +0.4 (sweet spot 之一)
  - **×0.05:VQA 42.5**(default)
  - ×0.01:VQA -1.5
  - ×0.00(frozen):VQA -1.1

为什么 LR multiplier 这么重要?Paper 给的 intuition:**prediction quality 在 training 开始时是 suboptimal 的**,如果 Y-Encoder 学得太快,会被 noisy prediction signal 带偏。所以让 Y-Encoder 慢慢学,等 Predictor 稳定后再 align。

Linear projection head 把 Predictor 和 Y-Encoder 投到 shared 1,536 维空间,loss 在这里计算。

### 2.4 Y-Decoder
- **训练时完全不参与**!只在 inference 时调用
- 把预测的 Ŝ_Y 解码成 text
- 这是 VL-JEPA efficiency gain 的关键来源

---

## 3. 训练目标:Bidirectional InfoNCE

$$\mathcal{L}_{\text{VL-JEPA}} = D(\hat{S}_Y, S_Y)$$

具体用 **bidirectional InfoNCE**:

$$\mathcal{L} = -\frac{1}{2}\left[\log \frac{\exp(\text{sim}(\hat{S}_Y, S_Y)/\tau)}{\sum_{k} \exp(\text{sim}(\hat{S}_Y, S_Y^{(k)})/\tau)} + \log \frac{\exp(\text{sim}(S_Y, \hat{S}_Y)/\tau)}{\sum_{k} \exp(\text{sim}(S_Y, \hat{S}_Y^{(k)})/\tau)}\right]$$

其中:
- $\hat{S}_Y$ = Predictor 输出
- $S_Y$ = Y-Encoder 输出
- $\tau$ = temperature
- $\text{sim}(\cdot, \cdot)$ = cosine similarity
- 第二项是反方向的 InfoNCE,让 Predictor 和 Y-Encoder 互相学习

**为什么 InfoNCE 而不是 L2/cosine loss?**

InfoNCE 可以拆成两项(Wang & Isola 2020):
1. **Alignment term**:minimize distance between normalized prediction and target embeddings
2. **Uniformity regularization**:push batch 内 embeddings 互相远离,避免 representation collapse

Ablation Table 7(c):
- InfoNCE:Classification 23.3, Retrieval 30.3, VQA 44.3
- Cosine:Classification 16.5 (-6.8), Retrieval 20.2 (-10.1), VQA 46.6 (+2.3)
- L1:Classification 14.8 (-8.5), Retrieval 15.5 (-14.8), VQA 41.9
- L2:Classification 13.5 (-9.8), Retrieval 11.7 (-18.6), VQA 43.7

Interesting 的是 **Cosine loss 在 VQA 上反而比 InfoNCE 好 2.3 points**,但 Classification/Retrieval 惨败。Paper 解释:只有 InfoNCE 有 anti-collapse regularization,可以配合 unfrozen Y-Encoder 用。

---

## 4. 两阶段训练

### Stage 1: Query-free Pretraining
- 数据:DataComp + YFCC-100M (image-text) + Action100M (video-text)
- **Query-free**:只用 caption,没有 question-conditioning
- 先 image-only(1 frame,100k iterations,batch 24k)
  - 2B samples seen
  - ImageNet zero-shot 61.6%(无 prompt ensembling)
- 再 video(8 frames × 60k iters + 32 frames × 10k iters)
- LR = 5e-5 constant
- 4 weeks on 24 nodes × 8×H200

为什么 constant LR 而不是 cosine?Paper 说 "to facilitate extended training"——constant LR 在长训练上更稳。

### Stage 2: Query-conditioned SFT
- 数据:PLM data mixture
  - 25M VQA samples
  - 2.8M captioning samples
  - 1.8M classification samples
  - downsampled pretraining data(避免 catastrophic forgetting)
- 83k steps,batch 3,072,~2.5 days
- Cosine LR annealing

为什么 SFT 数据里要 mix classification 和 captioning?因为如果只学 VQA,classification 和 retrieval 会 catastrophic forget(ablation Table 7(a):drop pretraining 直接 -21.7 classification,-17.3 retrieval)。

---

## 5. 关键实验:Controlled Comparison

这是 paper 最 important 的实验,Section 4.5。

**Setup**:严格 controlled comparison vs token-prediction VLM
- 相同 vision encoder: frozen PE ViT-L-14, 336² resolution, 16 frames
- 相同 training iterations,相同 effective batch size 128
- 相同 LR schedule,相同 pretraining data mixture
- **唯一区别**:VL-JEPA predict embedding(0.5B predictor)vs VLM 做 next-token prediction(1B LLM)

**Results**(Fig. 3):

| Samples Seen | VL-JEPA CIDEr | VLM CIDEr | VL-JEPA Top-5 | VLM Top-5 |
|---|---|---|---|---|
| 500K | 1.23 | 1.35 | 14.9% | 14.0% |
| 5M | 14.7 | 7.1 | 35.3% | 27.2% |
| 15M | 14.8 | 7.1 | 41.0% | 27.2% |

关键 insight:
- 500K 时两者差不多,VLM 甚至略好
- 但很快 VL-JEPA 的 sample efficiency **远超** VLM
- 在 15M 时 VL-JEPA 用 **0.5B predictor** 击败了用 **1B LLM** 的 VLM

这印证了 paper 的核心 hypothesis:**embedding-space supervision 比 token-space supervision 更 efficient**,因为 target distribution 简化了。

---

## 6. Selective Decoding:Real-time Streaming 的杀手锏

这是 paper 最 practical 的 contribution。

### 6.1 问题
传统 streaming VLM 必须 autoregressive decode,token-by-token,每帧都得 decode。VL-JEPA 因为 predict 的是 continuous embedding,可以一直输出 embedding stream,**只在语义发生显著变化时才 decode**。

### 6.2 机制
- 对 embedding stream 做 sliding window smoothing(average pooling)
- 用 **agglomerative clustering with temporal connectivity constraints** 分段
- Ward distance 作为 intra-segment monosemanticity 度量
- 只在 segment midpoint decode 一次
- 可以用 exact embedding 或 average-pooled embedding decode

### 6.3 实验(EgoExo4D,218 videos,平均 6 分钟)
- 对比 uniform sampling(固定间隔 decode)vs adaptive selective decoding
- Frequency sweep:2.0Hz → 0.01Hz

**结果**:
- Selective decoding 在 0.35Hz 匹配 uniform 在 1Hz 的 CIDEr
- **~2.85× decoding cost reduction**,性能保持
- Average pooling 在两种策略上都 consistently 好,因为有 denoising + stabilization 效果

这个 2.85× 数字很 important。意味着在 smart glasses / robotics 这种场景,可以 always-on 监控 embedding stream,只在事件发生时 decode 文字输出。

---

## 7. Multitasking:一个架构做四件事

| Task | How |
|---|---|
| Captioning / Open-ended VQA | Predict Ŝ_Y → Y-Decoder decode → text |
| Open-vocabulary Classification | Encode candidate labels → 比较 cosine distance with Ŝ_Y |
| Text-to-video Retrieval | Encode candidate videos(用 captioning prompt)→ rank by similarity |
| Discriminative VQA | Encode candidate answers → 选 nearest to Ŝ_Y |

关键:**所有这些 task 用同一个 unified architecture,不用改任何 module**。这是 embedding space 设计的天然优势,因为 classification/retrieval 本来就是 embedding space 的 native 操作。

---

## 8. 结果概览

### 8.1 Zero-shot Classification + Retrieval(Table 1)

VL-JEPA_BASE vs 最佳 baseline PE-Core-G:

| | Classification Avg (8 datasets) | Retrieval Avg R@1 (8 datasets) |
|---|---|---|
| PE-Core-G (86B samples, 2.3B params) | 44.7 | 58.1 |
| **VL-JEPA_BASE (3.6B samples, 1.6B params)** | **52.5** | **63.7** |

VL-JEPA 用 **24× 更少 samples** 和 **1.4× 更少 params**,全面超越。Per-dataset 分析显示 VL-JEPA 在 **motion-centric** benchmarks(SSv2, EK-100, EgoExo4D, COIN/CrossTask step recognition)特别强,而在 **appearance-centric**(Kinetics-400, COIN/CrossTask task recognition)相对弱。这是 frozen V-JEPA 2 vision encoder 的偏向性导致。

### 8.2 VQA(Table 2)

| Benchmark | VL-JEPA_SFT (1.6B) | 最佳 baseline |
|---|---|---|
| GQA | 61.5 | LLaVA-1.5-13B (62.0) |
| TallyQA | 69.9 | PaliGemma-3B (76.8) |
| POPE | 85.7 | LLaVA-1.5-13B-HD (86.3) |
| POPEv2 | 86.3 | Qwen2-VL-2B (91.3) |

VL-JEPA_SFT 在大部分 VQA 上 competitive 但不是 SOTA。Paper 诚实承认:VL-JEPA 不是用来替代 VLM 做 reasoning/tool-use 的,而是在 perception-heavy 任务上 efficient alternative。

### 8.3 WorldPrediction-WM(Table 3)—— 最 striking 的结果

WorldPrediction-WM:给 initial 和 final state images,从 4 个 candidate video clips 中选 correct action transition。

| Model | Accuracy |
|---|---|
| GPT-4o | 52.0 |
| Claude-3.5-Sonnet | 53.3 |
| Gemini-2.0 | 55.6 |
| **VL-JEPA_BASE** | **63.9** |
| **VL-JEPA_SFT** | **65.7** |

VL-JEPA 用 **1.6B params** 击败了所有 frontier LLM!这说明在 world modeling 这种 perception + action understanding 的任务上,embedding-space alignment 比 text reasoning 更 efficient。

### 8.4 Action Anticipation(Table 4, 5)

EPIC-KITCHENS-100(Recall@5,1s anticipation):
- V-JEPA2 ViT-g-384px: 39.7
- V-JEPA2 ViT-L-256px: 32.7
- **VL-JEPA ViT-L-256px: 34.2**(用相同 encoder 超 V-JEPA2 +1.5 points)

COIN next step forecasting:
- VideoLLM-online 8B: 49.1
- VideoLLM-MoD 8B: 49.7
- ProVideLLM 8B/11+: 53.6
- **VL-JEPA 1.6B: 56.2**(新 SOTA)

VL-JEPA 在 **action anticipation** 这种语义不确定的 prediction 任务上表现特别强,因为 embedding space 天然支持 distributional prediction。

---

## 9. Y-Encoder 评估:Hard Negatives(Table 6)

测试 Y-Encoder 是否对 semantic vs lexical changes 敏感(SugarCrepe++, VISLA):

| Model | SugarCrepe++ Avg | VISLA Avg |
|---|---|---|
| CLIP ViT-L | 44.5 | 34.5 |
| SigLIP2 ViT-g | 56.5 | 40.4 |
| PE-Core ViT-G | 58.6 | 38.3 |
| **VL-JEPA_BASE** | **63.9** | **42.9** |
| VL-JEPA_SFT | 58.4 | 39.5 |

VL-JEPA_BASE 的 Y-Encoder 比 PE-Core-G 强 5.3 points on SugarCrepe++!这说明 **JEPA architecture 不仅提升 prediction,还提升了 Y-Encoder 本身**。这很 important,因为 Y-Encoder 训练时是被 Predictor "pull" 的,这种 collaborative learning 让两边都受益。

但 interesting:VL-JEPA_SFT 反而比 BASE 弱 5.5 points。Paper 没明说原因,我推测是 SFT 数据偏向 VQA 任务的 narrow answer distribution,损失了 general text understanding 的细粒度。

---

## 10. Ablation 的重要细节

### 10.1 为什么用 bidirectional attention?(Table 7d)

- 用 bidirectional:VQA 42.5
- 用 causal(原始 Llama):VQA 40.6 (-1.9)

如果 query tokens append 在 visual tokens 后面,causal mask 让 visual tokens attend 不到 query,这是 fatal 的。所以 paper disable causal mask。这也意味着 Predictor 不再是严格的 language model,而是一个 **encoder**。

### 10.2 Llama-3 initialization 的作用(Table 7d)

- w/ Llama-3 init:VQA 42.5
- w/o Llama-3 init:VQA 40.6 (-1.9)

但 classification 和 retrieval 反而 random init 更好(Classification +0.8, Retrieval +0.2)。

Insight:**Llama-3 的语言先验对 VQA 这种 query-conditioned reasoning 重要**,但对 pure vision-language alignment 反而可能 over-regularize。

### 10.3 不同 Y-Encoder(Table 7e)

EmbeddingGemma-300M 是 default,但他们试了:
- Qwen3-Embedding-0.6B/4B/8B
- PE-Core-B/L/G

PE-Core-G(539M)最佳:Classification +14.4, Retrieval +7.9(vs EmbeddingGemma-300M)。

但 Paper 仍然用 EmbeddingGemma-300M,我推测是因为 PE-Core 是 visually-aligned encoder,用它会引入 circular dependency(用 PE-Core 训练一个想超越 PE-Core 的 model)。

---

## 11. 与 CLIP 和 VLM 的本质区别

| | CLIP | VLM | VL-JEPA |
|---|---|---|---|
| Architecture | JEA (joint embedding) | Vision encoder + LLM | JEPA (predictive) |
| Loss space | Embedding | Token | **Embedding** |
| Generation | ✗ | ✓ | ✓(with decoder) |
| Retrieval | ✓ | ✗ | ✓ |
| Streaming | ✓(encode only) | ✗(slow autoregressive) | ✓(selective decode) |
| Anti-collapse | Contrastive | N/A | InfoNCE / EMA / freeze |

VL-JEPA 的 sweet spot:**兼有 CLIP 的 embedding-space efficiency 和 VLM 的 generation 能力**。但 trade-off 是 reasoning/tool-use 仍然不如 generative VLM。

---

## 12. 我对这篇 paper 的几点 critical 思考

1. **为什么 embedding prediction 真的 work?**
   Paper 给的 hypothesis(target distribution 简化)很合理,但缺乏更深的 analysis。比如,如果能 visualize embedding space 里 multiple plausible targets 的 distribution,会更说服力。

2. **Y-Decoder 怎么训练的?**
   Paper 说 "is not involved during the main training phrase",但没说 inference 时 decoder 怎么来的。是不是用 Y-Encoder 的逆映射?是 distillation 还是 separately trained?这个 gap 在 paper 里比较 fuzzy。

3. **Generation quality 的 ceiling**
   所有 generation 实验都是 captioning 或 short VQA answer,没有 long-form generation。如果生成需要 long coherent text,embedding space 单点预测可能不够,因为一个 embedding 怎么 decode 成 50 tokens?

4. **Y-Encoder 在 SFT 后退化的谜团**
   VL-JEPA_SFT 的 Y-Encoder 在 hard negative benchmarks 上比 BASE 弱 5.5 points。这说明 SFT data 让 Y-Encoder 变 narrow 了。是否可以 multi-stage 解耦,让 Y-Encoder 保持 general,只 train Predictor?

5. **Reasoning 任务的天花板**
   Paper 自己承认:在 reasoning / tool use / agentic tasks 上 VLMs 仍然 excel。VL-JEPA 的 embedding prediction 是 "perception-friendly" 但不是 "reasoning-friendly"。这跟 LeCun 的 latent-space reasoning 大方向(如 COCONUT)是个有趣的对照——latent reasoning 需要 sequential latent states,而 VL-JEPA 是 single-shot prediction。

6. **Scale 没充分探索**
   Paper 最后说 "did not fully explore scaling"。1.6B params、3.6B pretraining samples,跟 frontier VLMs 比差两个数量级。如果 scale 上去,embedding prediction 是否还能保持 efficiency advantage?这是个 open question。

---

## 13. 相关工作链接

- V-JEPA 2(self-supervised video model,作为 X-Encoder):https://arxiv.org/abs/2506.09985
- I-JEPA(image JEPA,前作):https://arxiv.org/abs/2301.08243
- Perception Encoder(对比 baseline):https://arxiv.org/abs/2504.13181
- PerceptionLM(VLM baseline):https://arxiv.org/abs/2504.13180
- SigLIP 2:https://arxiv.org/abs/2502.14786
- EmbeddingGemma(Y-Encoder):https://arxiv.org/abs/2509.20354
- WorldPrediction benchmark:https://arxiv.org/abs/2506.04363
- COCONUT(latent reasoning 对照):https://arxiv.org/abs/2412.06769
- Large Concept Models(sentence-level latent LLM):https://arxiv.org/abs/2412.08821
- VICReg(regularization alternative):https://arxiv.org/abs/2105.04906
- Wang & Isola 2020(InfoNCE 的 alignment-uniformity decomposition):https://arxiv.org/abs/2005.10242
- LeCun 2022 "A Path Towards Autonomous Machine Intelligence"(JEPA 哲学源头):https://openreview.net/pdf?id=BZ5a1r-kVsf

---

## 总结

VL-JEPA 的核心 bet:**vision-language 任务的主要 difficulty 在 perception 和 semantic alignment,而非 surface-level language generation**。如果这个 bet 成立,那么 embedding-space supervision 比 token-space supervision 更 efficient。

Paper 用 controlled experiments 证明了这个 bet 至少在 perception-heavy 任务上成立(2.85× decode cost reduction,5M samples 击败 1B LLM 的 VLM on retrieval+captioning)。但 paper 也诚实承认:这不是 universal VLM replacement,reasoning/agentic 任务上 token-level generative VLMs 仍然占优。

下一步 interesting direction 可能是:**在 VL-JEPA 基础上加 latent reasoning**(类似 COCONUT),让 embedding stream 在 inference 时多步 evolve 而不是 single-shot。这或许能 bridge perception efficiency 和 reasoning 能力的 gap。
