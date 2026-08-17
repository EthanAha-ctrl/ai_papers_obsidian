---
source_pdf: MMFuser.pdf
paper_sha256: 9ebbc30d9ce5cf9733cf693a5251632c68ffe58d58aa0a013a0d1f54b3c7dc12
processed_at: '2026-08-05T19:17:34-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MMFuser

## 一句话概括

**ViT 有 24 层，你只用最后一层喂给 LLM，前面 23 层全扔了，太浪费了。但简单把浅层和深层拼起来又没用，因为浅层 feature 跟文字没对齐。MMFuser 的办法是：让深层 feature 当"提问者"，去浅层那里"借"细节，借回来的东西天然带着深层的语义锚定，LLM 就能看懂了。**

---

## 为什么这件事值得做

你想想现在的 MLLM pipeline：一张图进 CLIP-ViT，出来一个 576×1024 的 feature map（24×24 patch，每个 1024 维），这个 feature map 经过 MLP projector 喂给 LLM。

但 ViT 是 24 层 Transformer，每一层都在干不同的事。第 1 层在看 edge、texture，第 12 层在看 object part，第 23 层在看 "这是一只猫"。你只取第 23 层，等于把前面 95% 的 computation 扔了。

这就像你花 100 块钱吃自助餐，只吃了一个甜点就走了。

**为什么大家一直这么干？** 因为 CLIP 预训练的 contrastive loss 只作用在最后一层 CLS token 上，只有最后一层是跟 text space 对齐的。浅层 feature 虽然细节丰富，但它在另一个 "representation space" 里，LLM 根本看不懂。

---

## 简单融合为什么失败（这是 paper 最有价值的部分）

作者试了 4 种最直觉的方法：

1. **Concat**：把 5 层 feature 拼一起，channel 维度变 5 倍
2. **Average**：5 层取平均
3. **Weighted Average**：5 层学个权重加权平均
4. **FPN**：用 Feature Pyramid Network 融合（目标检测里的经典做法）

**结果：全都没用，有的还变差了。**

这就很反直觉。在 detection / segmentation 任务里，FPN 融合多层 feature 是 standard practice，效果都很好。为什么到 MLLM 这就不 work？

Root cause：**detection 任务的 head 是从头训的，它能适应任何 feature space。但 MLLM 的 "head" 是 LLM，LLM 是用 text pretrain 出来的，它只认 text-aligned 的 feature。** 你把没对齐的浅层 feature 混进去，对 LLM 来说就是噪声。

这就像你给一个只会英语的人一份中英混杂的文档，他反而看得更费劲，不如只给他英文版。

---

## MMFuser 的核心 idea

既然浅层 feature 的 "对齐性" 是问题，那就别让它直接进 LLM。用深层 feature（对齐好的）当 Query，浅层 feature 当 Key/Value，做 cross-attention。

**直觉类比**：你（deep feature，懂文字）去图书馆找资料。图书馆员（shallow feature，有细节但不懂数字）把所有书都摊开。你说 "我要关于猫的资料"，图书管理员就把相关的细节递给你。你拿到的是 "带着你的语义标签的细节"，而不是 "一堆看不懂的 raw 细节"。

公式上就三步：

**Step 1**：`F_ca = CrossAttn(Q=F_deep, K=V=Concat(F_shallow))`
深层问浅层："在我这个语义位置上，有什么细节？"

**Step 2**：`F_sa = F_ca + γ2 * SelfAttn(F_ca)`
让借来的细节之间再聊一聊（比如 OCR 场景里，patch A 借到 "3"，patch B 借到 "5"，self-attn 让它们组合成 "35"）

**Step 3**：`F_visual = F_deep + γ1 * F_sa`
把整理好的细节加回深层 feature 上

---

## 两个关键的 Engineering Trick

### Trick 1：γ1 和 γ2 初始化为 0

这个非常关键。训练第一步 forward 的时候：
- γ2 = 0 → F_sa = F_ca（self-attn 不起作用）
- γ1 = 0 → F_visual = F_deep（整个 MMFuser 输出 = 原始 deep feature）

也就是说，**训练起点时 MMFuser 完全等价于 LLaVA-1.5 baseline**。梯度可以从 LLM 一路流回到 cross-attn 的 weight，但模型行为没变。

然后随着训练，γ1、γ2 慢慢从 0 长起来，shallow detail 逐渐被注入。这就像给一个已经在跑的引擎装涡轮增压——你得让它先正常跑，再慢慢加压，不能一上来就爆缸。

这个 pattern 在 LoRA、CaiT LayerScale 里都见过，是 "frozen pretrained model + learnable residual" 的标准做法。

### Trick 2：用 Deformable Attention 而不是 Global Attention

Cross-attention 可以用普通的多头注意力，也可以用 deformable attention。作者发现 deformable 明显更好。

**为什么？** Global attention 是每个 query patch 看所有 576 个 key patch，太分散。Deformable attention 是每个 query patch 只采样 4 个可学习位置的 key——模型会学到 "文字通常在这里" "小物体在那里" 这种 spatial prior。

对于 OCR 和 visual grounding 这种需要精确空间定位的任务，deformable 的 "看哪儿" 比全局 "平均看" 高效太多。

而且 deformable 是 linear complexity，576 个 patch 不会爆。

---

## 选哪几层有讲究

作者 ablate 了 5 种 layer 组合：

- 只取浅层 [1,3,5,7]：细节太多，对齐太难，效果一般
- 只取中层 [9,11,13,15]：最差，既不浅也不深，redundant
- 只取深层 [17,19,21,24]：还行，但跟 deep query 太像
- 非均匀 [5,8,11,20]：不错
- **均匀 [3,8,13,18]：最好**

均匀采样覆盖了从 local 到 global 的完整 receptive field spectrum。这跟 ViT 的 attention head 演化规律一致（Raghu 2021）：浅层 local，深层 global，中间过渡。你想要 "既看局部又看全局"，就得跨跨度采样。

---

## 实验结果的人话版

### General benchmarks（12 个）

7B 模型 11/12 项超过 LLaVA-1.5-7B，平均 61.8 vs 60.3。
13B 模型 10/12 项超过 LLaVA-1.5-13B，平均 64.1 vs 63.2。

**最亮眼的提升**：
- VizWiz（盲人 VQA，需要读小字）：+3.8
- MME（综合感知，含 OCR/count/position）：+53.9
- MMBench：+2.2

### OCRBench

7B：297 → 315（+18）
13B：331 → 343（+12）

文字识别基本持平，但 "理解文字" 的任务（scene text VQA、document VQA）大幅提升。说明 MMFuser 不是让模型 "看得更清"，而是让模型 "更懂文字的 context"。

### Visual Grounding（RefCOCO）

7B 模型 REC 任务 +5.7，提升最大。这说明 shallow feature 保留了更精细的 spatial layout，对小物体定位帮助巨大。

### 失败的地方

**HMER（手写数学公式）一直是 0**，加了 MMFuser 还是 0。这类任务可能需要完全不同的 mechanism，不是 feature fusion 能救的。

---

## 跟其他路线对比

### vs. Multi-Encoder（MouSi, DeepSeek-VL）

这些方法用 CLIP + DINOv2 + SAM 多个 encoder，效果确实好，但：
- 计算量翻 2-3 倍
- 参数多几百 M
- 需要 fusion network 处理不同 encoder 的 feature space

MMFuser 只加几 M 参数，FLOPs 几乎不变。代价是 single encoder 的 "complementary information" 不如 multi-encoder。

两者其实是 orthogonal 的，可以叠加。

### vs. Dense Connector（concurrent work）

Dense Connector 也做多层融合，但直接 concat 后送 MLP，没有 "用 deep query 去 select" 的机制。MMFuser 的 cross-attn 相当于一个 learned gating，过滤掉 shallow feature 里的语义噪声。

---

## 我的 Intuition 总结

这篇 paper 给我三个 takeaway：

**1. Feature fusion 的关键是谁当 anchor**
用 semantically aligned 的 feature 当 query，noisy 的 feature 当 key/value，比简单融合有效得多。这个 principle 应该能推广到其他 modality fusion。

**2. Zero-init residual 是 frozen-pretrained-model 时代的标配**
你想在 LLM 上加新 module，就让它一开始等于 identity，然后慢慢学。LoRA、CaiT LayerScale、MMFuser 都是这个思路。

**3. 单个 model 的内部 redundancy 往往被低估**
我们总想着加更多 encoder、更多参数，但 ViT 24 层每层都在做不同 computation，榨干单 model 的潜力可能比 ensemble 更高效。

---

## 一个我自己的反思

这篇 paper 让我想起一件事：CNN 时代大家都用 FPN 融合多层 feature，这个 practice 在 detection / segmentation 里是 standard。但到了 ViT + LLM 时代，大家好像突然忘了这个套路，只取最后一层。

原因可能是 LLaVA 等 baseline 太 "好用" 了，follow pipeline 就能刷分，没人深究 vision encoder 的潜力。MMFuser 重新把这个直觉带回来，还加了一个关键的 alignment-aware 设计，算是给 MLLM 的 vision side 打了一个 "你其实没认真用 ViT" 的提醒。

---

## Reference

- Paper repo: https://github.com/yuecao0119/MMFuser
- LLaVA baseline: https://github.com/haotian-liu/LLaVA
- Deformable DETR: https://arxiv.org/abs/2010.04159
- Dense Connector (concurrent): https://arxiv.org/abs/2405.13800
- Raghu et al. ViT layer analysis: https://arxiv.org/abs/2108.08810
- CaiT LayerScale: https://arxiv.org/abs/2103.17239
- MouSi (multi-encoder 对比): https://arxiv.org/abs/2401.17221

---

# MMFuser 深度解析

## 一、论文核心 Intuition

这篇paper的核心observation非常直观：**ViT不同layer的feature map携带截然不同的信息**，而主流MLLM只用了最后一层，相当于浪费了一个single ViT encoder的潜力。

具体来看，作者做了一个很关键的analysis（Fig. 1）：
- **跨encoder比较**：CLIP-ViT-L / ConvNeXt-XXL / DINOv2-L / EVA02-L / SigLIP-L 之间的cosine similarity差异巨大
- **同encoder跨layer比较**：CLIP-ViT-L不同layer之间的feature map同样差异显著

这个observation直接指向一个hypothesis：**与其ensemble多个encoder（MouSi、DeepSeek-VL、LLaVA-HR的路线），不如先榨干单个ViT的multi-layer信息**。

## 二、关键Motivation：为什么简单融合失败？

这是这篇paper我觉得最valuable的analysis。作者实验了4种naive fusion方法（Table I），**全部fail**：

| Method | VizWiz | MME | MMBench | Avg |
|--------|--------|-----|---------|-----|
| LLaVA-1.5-13B baseline | 53.6 | 1531.3 | 67.7 | 63.0 |
| w/ Concatenation | 52.1 | 1537.5 | 63.7 | 63.5 |
| w/ Average | 54.7 | 1527.9 | 63.6 | 63.0 |
| w/ Weighted Average | 53.7 | 1553.2 | 63.4 | 63.9 |
| w/ FPN | 54.4 | 1532.8 | 62.5 | 63.1 |
| w/ MMFuser | **57.4** | **1585.2** | **69.9** | **64.9** |

**Why naive fusion fails的root cause**：shallow features虽然细节丰富，但与text embedding space**语义不对齐**。这是CLIP预训练的inherent property——contrastive loss只对最后一层CLS token施加约束，中间layer从未被显式对齐到text space。强行concat或average会把"语义噪声"注入到LLM input中，LLM无法解读这些"未对齐的细节"。

这给了一个重要的design principle：**必须保持deep feature的语义对齐作为anchor，把shallow detail"嫁接"到这个anchor上**，而不能反过来让shallow feature主导representation。

## 三、MMFuser架构详解

### 3.1 Overall Pipeline

```
Image → CLIP-ViT-L/336 → [F_1, F_2, ..., F_L] (L=5 layers)
                                ↓
                          MMFuser Module
                                ↓
                         F_visual (N×D)
                                ↓
                       MLP Projector
                                ↓
                       Concat with text embedding
                                ↓
                            LLM (Vicuna)
```

关键design choice：MMFuser位于ViT和MLP projector之间，在semantic alignment发生之前就完成feature fusion。

### 3.2 三步公式解析

**Step 1: Cross-Attention提取细节**

$$F_{ca} = \text{Attention}(\text{norm}(F_L), \text{norm}(X))$$

其中：
- $F_L \in \mathbb{R}^{N \times D}$：第L层（penultimate layer）的deep feature，作为**Query**
- $X = \text{Concat}(F_1, F_2, ..., F_{L-1}) \in \mathbb{R}^{N \times (L-1)D}$：浅层和中间层concat后的feature，作为**Key和Value**
- $N$：ViT patch数量（336px输入，patch size 14，N=24×24=576）
- $D$：embedding dimension（CLIP-ViT-L/14为1024）

**Intuition**：用语义对齐的deep feature作为query去"询问"shallow features："在我这个语义位置上，有什么细节信息被我丢失了？" 这样提取出来的细节天然带着deep feature的语义锚定。

**Step 2: Self-Attention强化交互**

$$F_{sa}' = \text{Attention}(\text{norm}(F_{ca}), \text{norm}(F_{ca}))$$
$$F_{sa} = F_{ca} + \gamma_2 \cdot F_{sa}'$$

其中：
- $\gamma_2 \in \mathbb{R}^D$：learnable vector，**初始化为0**

**关键trick**：$\gamma_2 = 0$ initialization意味着训练开始时 $F_{sa} = F_{ca}$，相当于self-attention是"渐进生效"的。这和LayerScale（CaiT）以及LoRA的zero-init思路一致——保证训练初期模型行为与baseline等价，避免random initialization破坏pretrained feature。

**Step 3: Residual融合到Deep Feature**

$$F_{visual} = F_L + \gamma_1 \cdot F_{sa}$$

其中：
- $\gamma_1 \in \mathbb{R}^D$：learnable vector，同样**初始化为0**
- 当$\gamma_1 = 0$时，$F_{visual} = F_L$，模型退化为原始LLaVA-1.5

**这个zero-init设计极其重要**：它保证了
1. 训练起点等价于baseline，梯度可以稳定流动
2. 模型自主决定"要注入多少shallow detail"——如果某个channel的细节有害，$\gamma_1$可以学成0或负值
3. 不会破坏deep feature已经学好的semantic alignment

### 3.3 Deformable Attention的选择

作者比较了3种attention机制（Table VII）：

| Attention Type | Complexity | VizWiz | MME | Avg |
|----------------|------------|--------|-----|-----|
| Global Attn (Vaswani) | Quadratic | 52.9 | 1566.3 | 64.2 |
| Linear SRA (PVTv2) | Linear | 54.3 | 1581.6 | 64.3 |
| Deformable Attn | Linear | **57.4** | **1585.2** | **65.4** |

Deformable attention胜出的原因：每个query point只sample 4个可学习的reference points，这些points会动态学习到"应该从shallow feature的哪个spatial位置提取细节"。对于OCR这种需要精确定位的任务，deformable的spatial-adaptive特性比global attention的"全局平均"更高效。

**架构细节**：
- Sampling points = 4
- Attention heads = 16
- 复杂度：$O(N \cdot K)$ where $K=4$，远小于global attention的$O(N^2)$

## 四、Layer Selection的Ablation（Table VI）

这个ablation非常enlightening：

| Query Layer | Key/Value Layers | VizWiz | MME | Avg |
|-------------|------------------|--------|-----|-----|
| - (baseline) | - | 53.6 | 1531.3 | 63.5 |
| 23 | [1,3,5,7] (shallow) | 54.3 | 1582.2 | 64.5 |
| 23 | [9,11,13,15] (middle) | 52.4 | 1560.3 | 64.0 |
| 23 | [17,19,21,24] (deep) | 54.7 | 1591.2 | 64.8 |
| 23 | [5,8,11,20] (non-uniform) | 54.7 | 1584.0 | 64.9 |
| 23 | **[3,8,13,18]** (uniform) | **57.4** | **1585.2** | **65.4** |

**关键发现**：
1. Pure shallow ([1,3,5,7]) 不如uniform sampling——shallow detail太多反而难对齐
2. Pure middle ([9,11,13,15]) 最差——这个区间既不够shallow（细节不足）又不够deep（语义不足），最redundant
3. Uniform sampling across [3,8,13,18] 最优——覆盖了shallow/middle/deep的完整receptive field spectrum

这个结果让我想到ViT的attention head receptive field演化（Raghu et al. NeurIPS 2021）：shallow layer的attention local，deep layer的attention global。uniform sampling相当于让模型同时访问local和global的receptive field，类似于FPN在CNN里的作用。

## 五、实验结果详解

### 5.1 12个General Benchmark（Table II）

**7B model关键提升**：
- VizWiz: 50.0 → 53.4 (+3.4)
- MMBench-EN: 64.3 → 67.5 (+3.2)
- MMBench-CN: 58.3 → 60.1 (+1.8)
- SEED-Bench: 58.6 → 60.8 (+2.2)
- LLaVA-Bench-Wild: 63.4 → 65.5 (+2.1)
- MMVet: 30.5 → 32.6 (+2.1)

**13B model关键提升**：
- VizWiz: 53.6 → 57.4 (+3.8)
- MME: 1531.3 → 1585.2 (+53.9)
- MMBench-EN: 67.7 → 69.9 (+2.2)
- POPE: 85.9 → 87.5 (+1.6)

**特别注意MME的+53.9**：MME有14个子任务，包括existence/count/position/color/OCR等fine-grained维度。MMFuser在这类需要细节的任务上提升最大，验证了shallow feature的价值。

### 5.2 OCRBench（Table III）

| Model | Recog. | VQA^S | VQA^D | KIE | HMER | Final |
|-------|--------|-------|-------|-----|------|-------|
| LLaVA-1.5-7B | 160 | 117 | 15 | 5 | 0 | 297 |
| + MMFuser | 159 | **128** | **20** | **8** | 0 | **315** (+18) |
| LLaVA-1.5-13B | 176 | 129 | 19 | 7 | 0 | 331 |
| + MMFuser | 171 | **136** | **25** | **11** | 0 | **343** (+12) |

**有意思的observation**：Text Recognition (Recog.) 略微下降（176→171），但Scene Text VQA和Document VQA大幅提升。这暗示MMFuser不是单纯提升OCR能力，而是让LLM更好地**理解**文字的语义——可能是因为shallow feature提供了text layout和context，帮助LLM推理而不是单纯识别。

### 5.3 Region-Level Tasks（Table IV & V）

**Region Captioning (CIDEr)**：
- 7B: 37.2 → 39.7 (+2.5)
- 13B: 38.9 → 42.8 (+3.9)

**Referring Expression Comprehension (Precision@0.5)**：
- 7B: 51.7 → 57.4 (+5.7)
- 13B: 60.7 → 61.1 (+0.4)

7B的REC提升尤其大（+5.7），这说明shallow feature对spatial grounding帮助巨大——shallow feature保留了更精细的spatial layout信息，而deep feature由于global attention的spatial smearing，定位小物体能力较弱。

### 5.4 Module Ablation（Table VIII）

| Configuration | VizWiz | MME | MMB | Avg |
|---------------|--------|-----|-----|-----|
| LLaVA-1.5-13B | 53.6 | 1531.3 | 67.7 | 63.5 |
| + Cross-Attn only | 54.6 | 1557.1 | 68.2 | 64.4 |
| + Cross-Attn + Self-Attn | **57.4** | **1585.2** | **69.9** | **65.4** |

Cross-Attn单独贡献：+0.9 avg
Self-Attn额外贡献：+1.0 avg

Self-Attn的作用：让提取出来的细节在patch之间交互。比如OCR场景中，一个patch提取了字符"3"，self-attention让它能"看到"相邻patch提取的"5"，组合成"35"——这对数字、单词的holistic识别至关重要。

## 六、与相关工作的对比

### 6.1 vs. Multi-Encoder Ensemble（MouSi, DeepSeek-VL, LLaVA-HR）

| 维度 | Multi-Encoder | MMFuser |
|------|---------------|---------|
| 计算开销 | 2-3x ViT forward | 1x ViT forward + 轻量module |
| 参数量 | 多个ViT (各~300M) | +几M (γ1, γ2, attn weights) |
| Feature alignment | 需要复杂fusion network | 利用deep feature天然对齐 |
| 推理速度 | 显著变慢 | 几乎无影响 |

**Trade-off**：multi-encoder提供了真正的"complementary information"（CLIP vs DINOv2 pretraining objective完全不同），而MMFuser只在single ViT内做intra-layer fusion。两者其实是orthogonal的，可以叠加使用。

### 6.2 vs. Dense Connector（concurrent work, [57]）

Dense Connector也做multi-layer fusion，但思路不同：它直接把多层feature concat后送入MLP，没有显式的semantic alignment mechanism。MMFuser的核心差异是用deep feature作为query去"selective extract"——这是一种attention-based的gating，避免了shallow feature的语义噪声直接进入LLM。

### 6.3 vs. LayerScale (CaiT)

MMFuser的$\gamma_1, \gamma_2$ zero-init设计与CaiT的LayerScale一脉相承，但方向相反：
- CaiT：用LayerScale控制**identity path** vs **transformation path**的比例
- MMFuser：用$\gamma$控制**baseline feature** vs **fused enhancement**的比例

两者都遵循"zero-init residual"原则，确保deep feature不被破坏。

## 七、Critique & 思考

### 7.1 优点
1. **Insight清晰**：semantic alignment是fusion的关键约束，naive fusion失败的原因分析透彻
2. **Lightweight**：只增加几M参数，FLOPs增量小
3. **Plug-and-play**：可以直接加到任何"ViT-MLP-LLM"架构

### 7.2 局限
1. **只测了CLIP-ViT-L/336**：没有验证DINOv2、SigLIP等其他encoder的效果。DINOv2的shallow feature可能语义对齐更差（无language supervision），MMFuser能否work存疑
2. **L=5是magic number**：没有充分ablate L=3/7/10等选择
3. **HMER始终为0**：手写数学公式识别完全fail，说明shallow feature对某些任务无效
4. **13B上REC提升微弱**：13B的LLM本身已经很强，shallow detail的边际价值降低
5. **没有video实验**：paper说"can be readily adapted for video"，但没有实验验证

### 7.3 延伸思考

**关于ViT shallow feature的本质**：ViT的shallow layer其实类似一个"non-linear CNN"——patch embedding后第1-3层主要做local texture/edge detection，没有global information。MMFuser的价值在于让LLM能"看到"这些CNN-like features，而CNN feature对OCR/grounding一直是SOTA（想想TextOCR、DETR系列）。

**关于zero-init residual的通用性**：这个design pattern在LLM时代越来越重要。LoRA、LayerScale、MMFuser都用了这个trick。本质上是"用pretrained model作为strong prior，learnable module作为residual correction"——这与LLM的scaling philosophy一致：base model已经足够强，新module应该"additive improvement"而非"disruptive change"。

**关于deformable attention的spatial inductive bias**：deformable attention的4个sampling points会学到具体的offset，这相当于给模型一个"where to look"的spatial prior。对于OCR这种任务，模型需要知道"文字通常在图像中央"或"小字在背景某处"，deformable attention能encode这种spatial pattern。

## 八、Reference Links

- **Paper**: [MMFuser GitHub](https://github.com/yuecao0119/MMFuser)
- **LLaVA-1.5 baseline**: https://github.com/haotian-liu/LLaVA
- **CLIP-ViT**: https://github.com/openai/CLIP
- **Deformable Attention**: https://github.com/fundamentalvision/Deformable-DETR
- **Dense Connector (concurrent)**: https://arxiv.org/abs/2405.13800
- **MouSi (multi-encoder)**: https://arxiv.org/abs/2401.17221
- **Raghu et al. (ViT layers analysis)**: https://arxiv.org/abs/2108.08810
- **CaiT LayerScale**: https://arxiv.org/abs/2103.17239
- **MME Benchmark**: https://github.com/BradyFU/Awesome-Multimodal-Large-Models
- **OCRBench**: https://github.com/Yuliang-Liu/ocrbench

## 九、结论

MMFuser这篇paper给我的最大启发：**在scaling的时代，我们往往急着加更多encoder、更多参数，却忽略了单个model内部的信息冗余**。ViT有24层，每层都在做不同的computation，只用最后一层等于扔掉了95%的intermediate computation。MMFuser用一个简单的cross-attention把"浪费的computation"回收利用，这比加新encoder高效得多。

从building intuition角度，这篇paper的core lesson是：**feature fusion的关键不是"加什么"，而是"以谁为anchor来加"**。用semantically aligned feature作为query，让noisy feature做key/value，比简单融合有效得多——这个principle应该能推广到其他modality fusion场景（比如audio-text fusion、sensor fusion等）。
