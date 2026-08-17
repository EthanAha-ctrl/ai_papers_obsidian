---
source_pdf: PaddleOCR-VL-1.5 Towards a Multi-Task 0.9B VLM for Robust In-the-Wild
  Document Parsing.pdf
paper_sha256: e76fba456e1bde5683347e7fd665f5692601f9b931b7449fe4eb78f0c1fbe218
processed_at: '2026-08-06T01:50:05-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 PaddleOCR-VL-1.5

## 一句话概括

这帮人用 0.9B 的小模型，在 document parsing 这个任务上把 235B 的 Qwen3-VL 和 Gemini-3 Pro 都按在地上摩擦了，而且专门解决了"手机拍照歪歪扭扭还能识别"这个现实问题。

---

## 这篇 paper 到底在解决什么痛点

你想想 RAG 系统是怎么工作的——用户给一个 PDF，你要把它变成 LLM 能消化的 text。这个过程叫 document parsing。听起来简单，实际上地狱级难度。

之前的 SOTA model 都在 "digital-born" 的干净 PDF 上刷分。所谓 digital-born 就是电脑直接生成的 PDF，文字是 vector 的，layout 是规整的。但现实世界呢？用户拿手机对着合同拍一张，页面是弯的、光是斜的、有 moiré 干扰、还歪了 15 度。现有 model 在这种情况下基本就废了。

PaddleOCR 团队觉得这个 gap 没人填，于是自己造了个 benchmark 叫 Real5-OmniDocBench，包含 5 种真实场景：scanning、warping、screen photography、illumination、skew。然后他们自己的 model 在这个 benchmark 上做到 92.05% overall accuracy，比 235B 的 Qwen3-VL 高 3 个点。

---

## 核心架构直觉：为什么是两阶段

这里有个设计选择值得品味。现在 trend 是 end-to-end VLM，一个 model 吃 image 吐 markdown。DeepSeek-OCR 就是这个路线。PaddleOCR-VL-1.5 偏要搞 two-stage：先 PP-DocLayoutV3 做 layout analysis，再 PaddleOCR-VL-1.5-0.9B 做 element recognition。

为什么？因为 autoregressive VLM 处理整页高分辨率 image 的 latency 太高了。你想想，一页 document 可能 2000×3000 pixel，VLM 要 autoregressive 吐出几千个 token，每页要好几秒。而且 VLM 对 geometric structure 的建模能力弱，它在 pixel level 上没有 inductive bias 去理解 "这两个 text block 哪个在前"。

Two-stage 的好处是分工明确：
- **PP-DocLayoutV3** 是 vision-centric model，一次 forward 就给出所有 element 的位置、形状、reading order
- **PaddleOCR-VL-1.5-0.9B** 只需要处理 rectified 后的小 region，分辨率低，速度快，精度高

代价是会有 cascading error——layout 漏检一个 block，VLM 永远救不回来。但实验数据证明这个 trade-off 是值得的。

---

## PP-DocLayoutV3 最有意思的设计：Reading Order 怎么预测

这是我认为这篇 paper 最 elegant 的部分。

传统做法是什么？用 pointer network，一个一个点过去。或者用 rule-based post-processing，根据坐标 heuristic 排序。这些方法在规整 document 上还能用，在歪斜的 document 上就崩了。

PP-DocLayoutV3 的思路：既然 decoder 已经有 N 个 object query $q_1, q_2, \ldots, q_N$（每个 query 对应一个检测到的 element），为什么不直接让这些 query 互相比较，得出一个 pairwise 的 "谁在谁前面" 的 score？

公式是这样的：

$$S_{i,j} = \frac{f(q_i, q_j) - f(q_j, q_i)}{\sqrt{d_h}}$$

其中 $f(q_i, q_j) = (W_q q_i)^\top (W_k q_j)$

变量解释：
- $q_i, q_j$：decoder 最后一层输出的第 $i$ 个和第 $j$ 个 object query embedding
- $W_q, W_k$：两个 learnable projection matrix，把 query 映射到 relational space
- $d_h$：hidden dimension，用来做 scaling（类似 attention 里的 $\sqrt{d_k}$）
- $S_{i,j}$：第 $i$ 个 element 在第 $j$ 个 element 前面的 score

关键在那个 $f(q_i, q_j) - f(q_j, q_i)$。这个减法强制了 **antisymmetric**：$S_{i,j} = -S_{j,i}$。如果 "A 在 B 前" 的 score 是 +2，那 "B 在 A 前" 的 score 自动是 -2。这编码了 "precedence" 这个概念本身的逻辑——不可能 A 既在 B 前又在 B 后。

然后怎么从 pairwise score 得到全局顺序？用 voting：

$$V_j = \sum_{i \neq j} \sigma(S_{i,j})$$

$\sigma$ 是 sigmoid，$V_j$ 是所有其他 element 认为 "自己在 $j$ 前面" 的概率之和。最后按 $V_j$ 升序排，votes 越少的越靠前。

这个 voting 机制天然 robust 到 cyclic inconsistency。假设 model 误判 A>B、B>C、C>A（循环），voting 也能给出一个合理的排序，因为它是 aggregate 所有 pairwise opinion 的结果。

直觉上：这就像 round-robin tournament 的 ranking 方法。每个人跟所有人比一次，赢的次数多的人排名靠前。pairwise comparison 容易出错，但 aggregate 起来就 robust 了。

---

## 为什么从 bounding box 换成 instance segmentation

传统 object detection 输出 axis-aligned bounding box，就是两个点 (x1, y1) 和 (x2, y2)。在规整 document 上这够了。但在弯曲的页面上，你想象一个 warping 的 page，两个相邻的 text block 在 image 上看是斜的，它们的 axis-aligned box 必然 overlap。

PP-DocLayoutV3 直接预测 pixel-accurate mask。mask 可以贴合物体的真实形状，不管它怎么歪怎么弯。这对后续的 element rectification 很关键——你只有精确知道 element 的边界，才能把它 crop 出来 rectify 成正的给 VLM 识别。

这个改动看似简单，实际收益巨大。Real5-OmniDocBench 的 Skew 场景，PaddleOCR-VL-1.5 从 V1.0 的 77.47% 直接跳到 91.66%，+14.19 个点。这个跳跃主要归功于 mask-based detection + distortion-aware augmentation。

---

## Text Spotting 的 coordinate 表示：为什么用 special token

这是另一个值得品味的设计细节。

Text spotting 任务要求 model 同时输出 text 和位置。位置用什么表示？传统做法是用数字，比如 "DREAM 253 286 346 298 345 339 252 330"。但这里有个坑：数字 "253" 在 BPE tokenizer 里可能被切成 "25" + "3"，因为 "253" 这个 string 不在 vocabulary 里。这种 fragmentation 让 model 很难学到 spatial information 的连续性。

PaddleOCR-VL-1.5 的做法：引入 1001 个 special token `<LOC_0>` 到 `<LOC_1000>`，每个 token 代表一个 normalized coordinate 值。坐标 253 就是 `<LOC_253>`，一个 single token。

这样做的好处：
1. 每个 coordinate 值有独立的 embedding，model 可以学到 "253 和 254 在 spatial 上很近" 这种连续性
2. 避免 tokenization fragmentation
3. 输出 length 固定（8 个 LOC token 对应 4 个点的 x, y），model 容易学

另外他们用 4-point quadrilateral 而不是 2-point bounding box。4 个点 (TL, TR, BR, BL) 能 tightly enclose 倾斜的文字。你想象一个旋转 30 度的 text，axis-aligned box 会包含很多背景，4-point quad 能紧贴文字边界。

---

## UACS：数据选择为什么不能 random sample

Post-training 阶段只有 5.6M 样本的 budget，怎么选？

Uniform random sampling 的问题：model 已经做得很好的 case（比如规整的英文 document）和 model 做不好的 case（比如弯曲的中文古文）各占一定比例。在简单 case 上继续训练是浪费 budget。

UACS (Uncertainty-Aware Cluster Sampling) 的三步：
1. 用 CLIP visual encoder 对所有 candidate image 做 embedding，K-Means 聚类，保证 visual diversity
2. 对每个 cluster，用 Stage 1 model 做 multiple stochastic inference，算 output divergence 作为 uncertainty score $S_i$
3. 按 $(S_i + \alpha)^\beta$ 加权采样，$\alpha=1.0$ 防 $S_i=0$，$\beta=2.0$ 放大 hard cluster 的权重

$\beta=2.0$ 的直觉：如果 cluster A 的 uncertainty 是 cluster B 的 2 倍，A 拿到 4 倍的 sampling budget。这是 polynomial amplification，让 model 把训练资源集中在 hard case 上。

这个思路本质是 hard example mining + curriculum learning 的结合，但用 visual clustering 保证了 diversity，不会只 sample 同一种 hard case。

---

## GRPO 阶段为什么需要

Post-training 后还有一个 RL 阶段，用 GRPO (Group Relative Policy Optimization)。

为什么需要 RL？因为 supervised fine-tuning 学到的是 "模仿 ground truth"，但 ground truth 本身可能有 style 不一致的问题。比如同样是 table，有的 ground truth 用 markdown 格式，有的用 HTML，有的用 LaTeX。SFT model 会学到这种 inconsistency，输出风格飘忽。

GRPO 的做法：对同一个 input，parallel rollout 多个 output，在 group 内计算 relative advantage。reward function 可以设计成 "格式一致性"、"结构正确性" 等。dynamic data screening protocol 会优先选 high reward potential 和 high entropy uncertainty 的样本，让 model 聚焦在 non-trivial case 上。

这跟 DeepSeekMath 的 GRPO 思路一致，只不过这里是用在 document parsing 而不是 math reasoning。

---

## 0.9B 为什么能赢 235B

这是最反直觉的结果。Qwen3-VL-235B 有 260 倍的参数量，在 OmniDocBench v1.5 上只有 89.15%，PaddleOCR-VL-1.5 是 94.50%。在 Real5-OmniDocBench 上差距更大：88.90% vs 92.05%。

为什么？我理解有几个原因：

1. **Task-specific architecture inductive bias**：PP-DocLayoutV3 的 anti-symmetric pairwise scoring 是专门为 reading order 设计的，general VLM 没有这种 inductive bias
2. **Targeted data**：46M pre-training + 5.6M post-training，全部是 document domain data。general VLM 的数据里 document 占比很小
3. **Distortion-aware augmentation**：专门模拟 real-world physical distortion，general VLM 没有这种针对性 augmentation
4. **Special token for coordinate**：4-point coordinate 的 special token 设计让 spatial information 学习更高效
5. **Two-stage 分工**：vision-centric model 处理 geometric structure，VLM 处理 semantic recognition，各司其职

General VLM 的优势在 general visual understanding，但在 domain-specific task 上，specialized small model + targeted data + task-aware architecture 是更高效的路径。

---

## Real5-OmniDocBench 的意义

这个 benchmark 本身就是这篇 paper 的重要贡献。之前没有专门测 real-world physical distortion 的 document parsing benchmark。OmniDocBench v1.5 虽然全面，但都是 clean document。

Real5-OmniDocBench 的 5 个场景都是 handheld mobile photography 常见的：
- **Scanning**：扫描仪，相对简单
- **Warping**：页面弯曲，像书本摊开
- **Screen Photography**：屏幕翻拍，有 moiré pattern
- **Illumination**：光照不均，像户外拍摄
- **Skew**：页面倾斜

除 Scanning 外都是手机手动采集，与原 OmniDocBench 一一对应。这个 benchmark 逼着 community 关注 real-world robustness，而不只是 clean document 上刷分。

---

## 推理性能为什么能优化到 1.43 pages/s

几个关键优化：

1. **异步多线程 pipeline**：PDF rendering、layout analysis、VLM inference 三个 stage 用三个 thread 并行，queue-based buffer 交换数据。这是经典的 pipeline parallelism
2. **Dynamic mini-batching**：VLM inference 的 batch 动态形成，queue 满或 timeout 就 launch。这让 multiple page 的 content block 能 group 进 single inference call
3. **多 backend 支持**：FastDeploy、vLLM、SGLang 都支持，FastDeploy 在 A100 上最快（1.4335 pages/s）

跨硬件测试也很全面：H800、A100、H20、L20、A10、RTX 3060、RTX 4090D 都测了。RTX 3060 这种消费级显卡都能跑 0.5351 pages/s，这对个人开发者和小型 deployment 很友好。

---

## 一些有趣的细节

1. **Seal Recognition**：NED 0.138 vs Qwen3-VL 的 0.382。Seal 是圆形或椭圆形的 curved text，general VLM 很难处理。PaddleOCR-VL-1.5 通过专门的数据和 instruction 训练，在 1/260 的参数量下做到 3 倍精度。

2. **Cross-page table merging**：跨页表格合并。这是 RAG 场景特别需要的功能，一个 table 跨两页，如果分成两个 chunk 检索就断了。

3. **Heading hierarchy refinement**：标题层级识别。给 RAG 提供 document structure metadata，改善 chunking 质量。

4. **支持 111 种语言**：新增了 Tibetan 和 Bengali。对 low-resource language 的支持是 document parsing 的重要方向。

5. **Ancient text 识别**：Text spotting benchmark 上 Ancient category 达到 0.8523，而 Rex-Omni 只有 0.4251。古文识别对文化 digitization 很重要。

---

## 对未来的启示

这篇 paper 给我几个 intuition：

1. **Specialized small model 在 domain task 上有持续优势**。general VLM scale up 的边际收益在递减，specialized model 通过 targeted data + task-aware architecture 能达到更高 efficiency。

2. **Real-world robustness 是下一个 battlefield**。clean benchmark 上的 SOTA 已经接近天花板，real-world physical distortion 是更有价值的研究方向。

3. **Two-stage 在 robustness 上有优势**。end-to-end 看起来 elegant，但 two-stage 的 division of labor 在 distortion 处理上更灵活——layout stage 可以 rectify distortion，recognition stage 在 clean input 上工作。

4. **Data curation 比 data scale 重要**。UACS 用 5.6M 样本就超过很多用几十 M 样本的方法，说明怎么选数据比选多少数据更关键。

5. **Coordinate 的 special token 设计**对 VLM 处理 spatial information 是重要 pattern，可以推广到其他需要 fine-grained localization 的 task。

References:
- PaddleOCR-VL: https://arxiv.org/abs/2510.14528
- OmniDocBench: https://arxiv.org/abs/2503.17213
- RT-DETR: https://arxiv.org/abs/2404.02907
- NaViT: https://arxiv.org/abs/2307.06304
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- CLIP: https://arxiv.org/abs/2103.00020
- PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR HuggingFace: https://huggingface.co/PaddlePaddle

---

# PaddleOCR-VL-1.5: 深度技术解析

## 1. 论文核心动机与定位

这篇 paper 来自 PaddlePaddle Team，核心贡献是在 0.9B 参数的 ultra-compact VLM 架构上，把 document parsing 的 SOTA 推到 OmniDocBench v1.5 的 94.5%，同时新提出 Real5-OmniDocBench 来测试 real-world physical distortion 下的 robustness（92.05% overall accuracy）。

关键 insight 在于：现有 SOTA model（包括 DeepSeek-OCR、MonkeyOCR、MinerU2.5）都主要针对 "digital-born" 或 clean scan 的 document 优化，对于 handheld mobile photography 常见的 skew、warping、screen moiré、illumination variation 这些 physical distortion 处理能力很差。PaddleOCR-VL-1.5 通过升级 layout engine + 扩展 VLM capability 来填补这个 gap。

官方资源：
- Website: https://www.paddleocr.com
- Code: https://github.com/PaddlePaddle/PaddleOCR
- Models: https://huggingface.co/PaddlePaddle

---

## 2. 整体架构：Two-Stage Framework

### 2.1 为什么是 Two-Stage

整体框架是 **PP-DocLayoutV3 (layout analysis) + PaddleOCR-VL-1.5-0.9B (element recognition)** 的级联。这里的设计哲学非常值得品味：

- **Pure VLM 一阶段方案的问题**：autoregressive VLM 处理高分辨率整页 document 时 latency 极高，且对 layout 结构的几何约束建模弱。
- **Pure pipeline 方案的问题**：传统 pipeline（如 PP-StructureV3）各模块独立训练，feature 不共享，cascading error 严重。
- **Two-Stage 折中**：用 vision-centric detector 一次前向给出完整 layout（包括 reading order），再让 VLM 处理 rectified 后的 element，既高效又精确。

对于 **Text Spotting 任务**，框架简化为直接用 VLM 做 end-to-end detection + recognition，跳过 layout stage。

### 2.2 PP-DocLayoutV3：Unified Layout Analysis

这是这篇 paper 最有意思的架构创新。相比 PP-DocLayoutV2 的 decoupled pointer network，V3 把 **detection + instance segmentation + reading order prediction** 三个任务全部塞进一个 RT-DETR-based Transformer decoder。

#### 2.2.1 从 bounding box 到 instance segmentation

传统 axis-aligned bounding box 在 skew/warped document 上会大量 overlap 或捕获背景噪声。V3 用 mask-based detection head 直接输出 pixel-accurate mask。这个改变看似简单，实际上对 non-planar document 的 element isolation 是 critical 的——想象一个弯曲页面上两个相邻 text block，矩形框几乎必然 overlap，但 mask 可以精确贴合物体边界。

#### 2.2.2 Reading Order Prediction 的数学形式化

这是 paper 中的核心公式（公式 1 和 2）。让我详细拆解：

**Global Pointer Mechanism** 把 decoder 的 N 个 object query $Q = \{q_i\}_{i=1}^N \in \mathbb{R}^{N \times d}$ 投影到 shared relational space，计算 pairwise precedence score：

$$S_{i,j} = \frac{f(q_i, q_j) - f(q_j, q_i)}{\sqrt{d_h}}, \quad \text{where } f(q_i, q_j) = (W_q q_i)^\top (W_k q_j)$$

变量含义：
- $q_i, q_j$：第 $i$ 个和第 $j$ 个 object query embedding（decoder 最后一层输出）
- $W_q, W_k \in \mathbb{R}^{d \times d_h}$：learnable projection matrices，把 query 投影到 relational space
- $d_h$：hidden dimension，用于 scaled dot-product attention 的 scaling
- $S_{i,j} \in \mathbb{R}^{N \times N}$：pairwise precedence score matrix

**关键设计**：$f(q_i, q_j) - f(q_j, q_i)$ 这一项强制了 **anti-symmetric** 性质，即 $S_{i,j} = -S_{j,i}$。这个设计非常 elegant——它直接编码了 "precedence" 这个概念本身的 antisymmetry（如果 A 在 B 前，那 B 必然不在 A 前），避免了用 softmax 时 pair (A,B) 和 (B,A) 独立预测的不一致问题。

**Voting-based Ranking**（公式 2）：

$$V_j = \sum_{i=1, i \neq j}^N \sigma(S_{i,j})$$

- $\sigma(\cdot)$：sigmoid function，把 score 转成 probability
- $V_j$：第 $j$ 个 element 的 total "被precede" votes
- 最终 reading order：按 $V_j$ **升序**排序（votes 越少，越靠前）

这个 voting 机制解决了一个经典问题：pairwise comparison 容易出现循环（A>B, B>C, C>A）。通过 summing 所有 incoming votes，相当于做了一种 tournament-style ranking，对噪声和 cyclic inconsistency 有天然 robustness。

#### 2.2.3 Single Forward Pass 的优势

V3 的 multi-head system 并发输出：
- classification labels
- bounding box coordinates
- pixel-accurate segments
- logical reading sequence

对比 V2 的 decoupled pointer network，V3 消除了 redundant post-processing 和 separate feature extraction。这意味着 detection 和 reading order 共享同一份 query representation，geometric localization 和 logical sequencing 在 feature level 就 alignment 了。

### 2.3 PaddleOCR-VL-1.5-0.9B：Element Recognition

继承了 V1.0 的轻量架构：
- **NaViT-style Native Resolution Visual Encoder** [17]：支持任意 aspect ratio 和 resolution，对 document 这种长宽比变化大的场景特别重要
- **Adaptive MLP Connector**：连接 vision encoder 和 LLM
- **ERNIE-4.5-0.3B** [5]：0.3B 的 lightweight language backbone

V1.5 把支持任务从 4 个扩展到 6 个：
1. OCR
2. Formula Recognition
3. Table Recognition
4. Chart Recognition
5. **Seal Recognition**（新）
6. **Text Spotting**（新）

---

## 3. Training Recipe 深度剖析

### 3.1 PP-DocLayoutV3 的训练

#### 3.1.1 End-to-End Joint Optimization

V2 是 two-stage decoupled（先训检测，再训 reading order），V3 改为 **end-to-end joint training**：RT-DETR backbone + Global Pointer head + Mask head 同时优化。这确保 decoder query 同时编码 geometric boundary 和 topological relationship。

#### 3.1.2 Distortion-Aware Data Augmentation

专门设计的 augmentation pipeline 模拟 mobile photography 的 physical deformation。这是 robustness 提升的关键。Standard augmentation（random crop, flip）无法覆盖手持设备的 perspective distortion、curvature、moiré pattern。

#### 3.1.3 训练超参数

- Optimizer: AdamW, weight decay = 0.0001
- Learning rate: constant $2 \times 10^{-4}$
- Epochs: 150
- Batch size: 32
- Initialization: PP-DocLayout_plus-L 预训练权重
- Training data: 38k high-quality samples，25 categories

#### 3.1.4 数据集构建策略

25 个 component categories 包括：Paragraph Title, Image, Text, Number, Abstract, Content, Figure Title, Display Formula, Table, Reference, Doc Title, Footnote, Header, Algorithm, Footer, Seal, Chart, Formula Number, Aside Text, Reference Content, Header Image, Footer Image, Inline Formula, Vertical Text, Vision Footnote。

**Hard-case mining** 用 PP-DocLayoutV2 做 dual-threshold inference：高/低 confidence threshold 下 detection density 差异大的样本标记为 unstable case。这个策略发现了 comics、CAD drawings、high-aspect-ratio screenshots 这些非标准 layout。

### 3.2 PaddleOCR-VL-1.5-0.9B 的训练

三阶段 progressive training paradigm（实际是两阶段，Pre-training + Post-training，但 Post-training 内部有 GRPO）：

| Settings | Pre-training | Post-training |
|----------|--------------|--------------|
| Training Samples | 46M | 5.6M |
| Max Resolution | 1280 × 28 × 28 | 1280 × 28 × 28 |
| Sequence length | 16384 | 16384 |
| Trainable components | All | All |
| Batch sizes | 128 | 128 |
| Data Augmentation | Yes | Yes |
| Maximum LR | $5 \times 10^{-5}$ | $8 \times 10^{-6}$ |
| Epoch | 1 | 1 |

注意 spotting task 的 max resolution 提升到 $2048 \times 28 \times 28$，这是因为 spotting 需要更精细的 spatial localization。

#### 3.2.1 Pre-training: Vision-Language Alignment

数据从 29M 扩到 46M image-text pairs。关键扩展：
- 更多 multilingual documents
- 复杂 real-world scenarios
- **seal recognition 和 text spotting 的大规模 pre-training data**——这里很巧妙：把新任务的 prior 提前注入 alignment phase，让 visual backbone 在 fine-tuning 前就建立对这些 pattern 的 sensitivity

#### 3.2.2 Post-training: Instruction Fine-tuning

继承 V1.0 的 4 个 instruction task（OCR、Table、Formula、Chart），新增 2 个：

**Seal Recognition**：处理 official seals/stamps 的 curved text、blur、background interference。

**Text Spotting** 的 representation 设计非常关键。paper 用 4-point quadrilateral 而不是 2-point bounding box，因为 4-point（TL, TR, BR, BL）能 tightly enclose inclined/irregular text。

**Target sequence 构造**（公式 3）：

$$Y = \text{Text} \oplus <\text{LOC\_}x_{\text{TL}}> <\text{LOC\_}y_{\text{TL}}> \ldots <\text{LOC\_}y_{\text{BL}}>$$

关键设计：引入 special tokens $\{<\text{LOC\_0}>, \ldots, <\text{LOC\_1000}>\}$ 表示 normalized coordinates。这避免了把坐标当 plain text 数字 tokenizing 时的 fragmentation 问题。

具体例子（公式 4）：
```
DREAM <LOC_253> <LOC_286> <LOC_346> <LOC_298> <LOC_345> <LOC_339> <LOC_252> <LOC_330>
```

这里 8 个 LOC token 分别对应 4 个点的 (x, y)，normalized 到 [0, 1000] 范围。每个 LOC token 有独立的 embedding，让 model 学到 spatial information 的 specific representation。

#### 3.2.3 GRPO 阶段

Post-training 后还有一个 **Reinforcement Learning stage using GRPO (Group Relative Policy Optimization)** [20]。

GRPO 的核心：
- 执行 parallel rollouts
- 在每个 group 内计算 relative advantage
- 避免 style inconsistency 问题

配合 **dynamic data screening protocol**：优先选择 high reward potential 和 entropy uncertainty 的 challenging samples。这确保 model 聚焦在 non-trivial、high-value 的 learning case 上，而非 trivial sample 浪费 capacity。

---

## 4. 数据策略：Uncertainty-Aware Cluster Sampling (UACS)

这是 Section 3.2.1 的核心，值得单独细讲。

### 4.1 三步流程

**Step 1: Visual Feature Clustering**
- 用 CLIP [21] visual encoder 提取 high-dimensional semantic embedding
- 对每个 task（OCR、Table、Formula、Chart、Seal、Spotting）独立做 K-Means 聚类
- 得到 K 个 visual clusters $\{C_1, C_2, \ldots, C_K\}$
- 目的：group 视觉结构相似的样本（如 solid line table vs. wireless table）

**Step 2: Uncertainty Estimation**
- 从每个 cluster $C_i$ 随机 sample 子集
- 用 Stage 1 pre-trained model 做 multiple inference passes with stochastic decoding
- 计算 uncertainty score $S_i$：基于 generated outputs 的 divergence
- $S_i$ 越高，model 对该 cluster 越不 confident

**Step 3: Weighted Sampling Plan**（公式 4）：

$$N_i = \min\left(\left|\frac{(S_i + \alpha)^\beta}{\sum_{j=1}^K (S_j + \alpha)^\beta} \times N_{\text{total}}\right|, |C_i|\right)$$

变量含义：
- $S_i$：cluster $C_i$ 的 average uncertainty score
- $|C_i|$：cluster $C_i$ 的总样本数
- $\alpha$：smoothing factor（$\alpha = 1.0$），防止 $S_i = 0$ 时权重为 0
- $\beta$：power factor（$\beta = 2.0$），控制 hard cluster 的 amplification 程度
- $N_{\text{total}}$：total sampling budget
- $N_i$：分配给 cluster $C_i$ 的样本数

$\beta = 2.0$ 的含义：uncertainty 被平方放大。如果 cluster A 的 uncertainty 是 cluster B 的 2 倍，A 拿到的 sampling budget 是 B 的 4 倍。这是 hard example mining 的 polynomial weighting 思想。

$\min(\cdot, |C_i|)$ 的含义：防止分配的 sample 数超过 cluster 实际有的样本数（避免重复采样或下溢）。

### 4.2 UACS 的直觉

传统 uniform random sampling 的问题：简单样本（model 已经做得好的）和困难样本各占一定比例，但 model 在简单样本上已经 saturated，再训练边际收益很低。UACS 通过 visual clustering 保证 diversity（不重复 sample 同类视觉结构），再通过 uncertainty estimation 把 budget 倾斜给 hard case，相当于 curriculum learning + hard example mining 的结合。

---

## 5. 评估结果深度分析

### 5.1 OmniDocBench v1.5（Table 2）

OmniDocBench v1.5 是 v1.0 的扩展，加了 374 个新 document，共 1355 页，中英文更平衡，formula 更多。Evaluation metric 升级：
- Text + Reading Order: Edit Distance
- Table: TEDS (Tree-Edit-Distance-based Similarity)
- Formula: CDM (Character Detection Matching) [23]——比之前的 metric 更 objective

PaddleOCR-VL-1.5 vs 主要 competitor：

| Model | Params | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ | ReadingOrder↓ |
|-------|--------|----------|-----------|-------------|------------|---------------|
| MinerU2.5 | 1.2B | 90.67 | 0.047 | 88.46 | 88.22 | 0.044 |
| PaddleOCR-VL (V1.0) | 0.9B | 92.86 | 0.035 | 91.22 | 90.89 | 0.043 |
| **PaddleOCR-VL-1.5** | **0.9B** | **94.50** | **0.035** | **94.21** | **92.76** | **0.042** |
| Qwen3-VL-235B | 235B | 89.15 | 0.069 | 88.14 | 86.21 | 0.068 |
| Gemini-3 Pro | - | 90.33 | 0.065 | 89.18 | 88.28 | 0.071 |

关键 observation：
- **0.9B 参数 vs 235B Qwen3-VL**：PaddleOCR-VL-1.5 在 Overall 上领先 5.35 个点
- **Formula CDM 提升最显著**：从 91.22 → 94.21（+2.99），说明 distortion-aware augmentation 对 formula recognition 帮助最大
- **Table TEDS**：从 90.89 → 92.76（+1.87），borderless table 和 invoice 的改进
- **Reading Order**：从 0.043 → 0.042，绝对值小但相对提升对 irregular layout 显著

### 5.2 Real5-OmniDocBench（Table 3）——这篇 paper 的真正杀手锏

Real5-OmniDocBench 基于 OmniDocBench v1.5 构造，5 个场景：
- **Scanning**：扫描仪
- **Warping**：页面弯曲
- **Screen Photography**：屏幕翻拍（有 moiré）
- **Illumination**：光照变化
- **Skew**：倾斜

除 Scanning 外都用 handheld mobile device 手动采集，与原 ground truth 一一对应。

| Model | Params | Overall↑ | Scanning | Warping | Screen | Illumination | Skew |
|-------|--------|----------|----------|---------|--------|--------------|------|
| Gemini-3 Pro | - | 89.24 | 89.47 | 88.90 | 88.86 | 89.53 | 89.45 |
| Qwen3-VL-235B | 235B | 88.90 | 89.43 | 89.99 | 89.27 | 89.27 | 86.56 |
| PaddleOCR-VL (V1.0) | 0.9B | 85.54 | 92.11 | 85.97 | 82.54 | 89.61 | 77.47 |
| **PaddleOCR-VL-1.5** | **0.9B** | **92.05** | **93.43** | **91.25** | **91.76** | **92.16** | **91.66** |

最戏剧性的提升在 **Skew 场景**：从 77.47 → 91.66（+14.19 absolute improvement）。这验证了 PP-DocLayoutV3 的 multi-point localization + Distortion-Aware Augmentation 在 extreme geometric distortion 下的有效性。

Appendix B 给出了 5 个子场景的详细 breakdown（Table A2-A6），可以看 PaddleOCR-VL-1.5 在每个子场景的 Formula-CDM 和 Table-TEDS 都是新 SOTA。例如 Skew 场景下（Table A6），Table-TEDS 达到 88.69%，而 Gemini-3 Pro 只有 88.06%，Qwen3-VL-235B 只有 83.41%。

### 5.3 Text Spotting（Table 4）

In-house benchmark 覆盖 9 个维度：Ancient、Blur、Common、Handwrite_ch、Handwrite_en、Printing_ch、Printing_en、Table、Japanese。

| Model | Overall | Ancient | Blur | Common | Handwrite_ch | Handwrite_en | Printing_ch | Printing_en | Table | Japanese |
|-------|---------|---------|------|--------|--------------|--------------|-------------|-------------|-------|----------|
| HunyuanOCR | 0.6290 | 0.6164 | 0.6392 | 0.5222 | 0.7984 | 0.7665 | 0.6213 | 0.5956 | 0.4419 | 0.6593 |
| Rex-Omni | 0.6682 | 0.4251 | 0.6936 | 0.6112 | 0.8147 | 0.7812 | 0.6961 | 0.6088 | 0.7185 | 0.6642 |
| **PaddleOCR-VL-1.5** | **0.8621** | **0.8523** | **0.8422** | **0.7713** | **0.8952** | **0.9163** | **0.8669** | **0.8689** | **0.8993** | **0.8461** |

PaddleOCR-VL-1.5 在所有 9 个维度都领先，特别是 Table（0.8993 vs 0.7185）和 Ancient（0.8523 vs 0.4251）提升巨大。Table 高分得益于 4-point quadrilateral representation 能 tightly enclose table 中倾斜的文字。

### 5.4 Seal Recognition（Table 5）

| Model | Params | NED↓ |
|-------|--------|------|
| Qwen2.5-VL-72B | 72B | 0.396 |
| Qwen3-VL-235B | 235B | 0.382 |
| **PaddleOCR-VL-1.5** | **0.9B** | **0.138** |

NED (Normalized Edit Distance) 越低越好。0.9B model 的 NED 是 235B Qwen3-VL 的约 1/3。这反映了 specialized training data + dedicated instruction 的力量——general VLM 即使参数大 260 倍，在 domain-specific task 上仍输给 specialized small model。

---

## 6. Inference Performance 优化

### 6.1 异步多线程 pipeline

整个 workflow 拆成 3 个 thread：
1. **Input preparation**：PDF → images
2. **Layout analysis**：PP-DocLayoutV3
3. **VLM inference**：PaddleOCR-VL-1.5-0.9B

Thread 之间用 queue-based buffer 交换 intermediate result，形成 pipeline 并行。

### 6.2 Dynamic Mini-batching

VLM inference stage 的 batching 策略：
- Queue size 达到 preset capacity → launch batch
- 或 oldest queued item 等待超过 time limit → launch batch

这个策略让 multiple page 的 content block 能 group 进 single inference call，对 large document collection 的 parallel efficiency 提升显著。

### 6.3 多 backend 部署性能（Table 6）

OmniDocBench v1.5，batch 512，单 A100 GPU：

| Method | Backend | Total Time (s)↓ | Pages/s↑ | Tokens/s↑ |
|--------|---------|-----------------|----------|------------|
| PaddleOCR-VL | FastDeploy v2.3 | 1104.5 | 1.2261 | 1700.5 |
| PaddleOCR-VL | vLLM v0.10.2 | 1325.5 | 1.0216 | 1419.9 |
| PaddleOCR-VL-1.5 | FastDeploy v2.3 | 944.4 | 1.4335 | 2016.6 |
| PaddleOCR-VL-1.5 | vLLM v0.10.2 | 1184.3 | 1.1433 | 1605.6 |
| PaddleOCR-VL-1.5 | SGLang v0.5.2 | 1342.0 | 1.0091 | 1418.9 |
| MinerU2.5 | vLLM v0.10.2 | 1356.5 | 0.9984 | 1415.1 |
| DeepSeek-OCR | vLLM v0.8.5 | 2130.5 | 0.6358 | 897.4 |

PaddleOCR-VL-1.5 + FastDeploy 达到 **1.4335 pages/s** 和 **2016.6 tokens/s**，比 V1.0 + FastDeploy 提升 16.9% 和 18.6%。

### 6.4 跨硬件部署（Table A8）

测试了 H800、A100、H20、L20、A10、RTX 3060、RTX 4090D 等 7 种硬件。例如：
- H800 + FastDeploy: 2.4320 pages/s, 64.8 GB VRAM
- RTX 4090D + vLLM: 1.4667 pages/s, 16.3 GB VRAM
- RTX 3060 + vLLM: 0.5351 pages/s, 11.8 GB VRAM（消费级显卡也能跑）

这种广泛的硬件兼容性对实际部署意义重大。

---

## 7. 关键 Design Pattern 总结

### 7.1 Vision-Centric vs. VLM-Centric 的分工

PP-DocLayoutV3 处理 geometric/topological structure（vision-centric 擅长），VLM 处理 semantic recognition（language modeling 擅长）。这种 division of labor 避免了让 autoregressive VLM 处理高分辨率整页（计算量爆炸），同时保留了 VLM 的 semantic understanding 能力。

### 7.2 Anti-Symmetric Design 的数学美

Reading order prediction 的 $S_{i,j} = -S_{j,i}$ 约束，加上 voting-based ranking，是把 graph ranking 问题转化成 pairwise comparison + aggregation 的经典方法。类似的思路在 tournament ranking、PageRank 等算法中都有体现。

### 7.3 Special Token for Coordinates

LOC_0 到 LOC_1000 的 special token 设计，避免了数字 tokenization 的 fragmentation 问题。例如坐标 "253" 如果用 BPE 可能被切成 "25" + "3"，破坏了 spatial information 的整体性。Special token 让每个 normalized 坐标值有独立 embedding，model 可以学到 spatial 的连续性。

### 7.4 Uncertainty-Aware Data Selection

UACS 把 hard example mining 从 instance level 提升到 cluster level，结合 visual diversity（clustering）和 difficulty（uncertainty），是 data curation 的一种 principled 方法，比纯 random sampling 或纯 hard mining 都更鲁棒。

### 7.5 Distortion-Aware Augmentation 的针对性

General augmentation（ImageNet style）对 document distortion 针对性弱。Distortion-Aware Augmentation 专门模拟 scanning、warping、screen-capture、geometric skewing、illumination variation，让 model 在训练时就见过这些 physical distortion。这是 Real5-OmniDocBench 大幅提升的根本原因。

---

## 8. 与同期工作的对比

### 8.1 vs. DeepSeek-OCR [10]

DeepSeek-OCR 用 optical 2D mapping 做 high-ratio vision-to-text compression，是 end-to-end 思路。PaddleOCR-VL-1.5 选择 two-stage（layout + recognition）。从 Real5-OmniDocBench 看，DeepSeek-OCR Overall 73.99，PaddleOCR-VL-1.5 是 92.05，差距 18 个点。Two-stage 在 robustness 上有优势，因为 layout analysis 能先 rectify distortion。

Paper: https://arxiv.org/abs/2510.18234

### 8.2 vs. MonkeyOCR v1.5 [11]

MonkeyOCR 用 structure-recognition-relation 三阶段范式。PaddleOCR-VL-1.5 用 detection-segmentation-ordering 统一在一个 model 里。MonkeyOCR-pro-3B 在 Real5-OmniDocBench Overall 是 79.49，PaddleOCR-VL-1.5 是 92.05。

Paper: https://arxiv.org/abs/2511.10390

### 8.3 vs. MinerU2.5 [2]

MinerU2.5 是 decoupled vision-language model，1.2B 参数。OmniDocBench v1.5 Overall 90.67，Real5-OmniDocBench Overall 85.61。PaddleOCR-VL-1.5 用更少参数（0.9B）达到更高精度（94.50 和 92.05）。

Paper: https://arxiv.org/abs/2509.22186

### 8.4 vs. HunyuanOCR [12]

HunyuanOCR 统一架构支持 translation 和 extraction。Text spotting benchmark 上 HunyuanOCR Overall 0.6290，PaddleOCR-VL-1.5 是 0.8621。

Paper: https://arxiv.org/abs/2511.19575

---

## 9. 局限与可能改进

虽然 paper 没有明确讨论 limitation，但可以推断：

1. **Two-stage 的 cascading error**：虽然 V3 减少了 decoupled 问题，但 layout 错误仍会传播到 recognition stage。例如 layout 漏检一个 formula block，VLM 永远无法识别它。
2. **Real5-OmniDocBench 的覆盖**：5 个场景可能不够全面。Real-world 还有 watermark、stamp overlap、partial occlusion、mixed language in same page 等场景。
3. **Seal Recognition 的 NED 0.138**：虽然比 Qwen3-VL 好，但绝对值仍然有 13.8% 的 character-level error，对于 legal document 这种高精度场景可能不够。
4. **End-to-end latency**：1.43 pages/s 在大规模 document 处理时仍然有瓶颈，特别是 million-page scale 的 archive digitization。
5. **Multilingual 扩展**：目前支持 111 种语言，新增了 Tibetan 和 Bengali，但 low-resource language 的 accuracy 没有单独 report。

---

## 10. 对 RAG 和 LLM 应用的影响

Paper 反复强调 document parsing 是 RAG 系统 [8] 的基础。高质量的 document parsing 直接决定 RAG 的 ingestion fidelity。PaddleOCR-VL-1.5 的几个特性对 RAG 特别重要：

- **Cross-page table merging**：跨页表格合并，避免 RAG 检索到断裂的 table chunk
- **Heading hierarchy refinement**：标题层级识别，给 RAG 提供 document structure 的 metadata
- **Markdown/JSON 输出**：直接给 LLM 消费的结构化格式
- **Real-world robustness**：手持拍摄的 document（如用户上传的合同照片）也能 high-fidelity 解析

对 LLM 应用而言，0.9B 的 compact size 意味着可以 local deployment，对 privacy-sensitive 场景（如医疗病历、法律合同）特别有价值。

---

## 11. 个人 Takeaway

这篇 paper 的核心贡献是把 document parsing 的 robustness 维度正式提上议程。之前的工作都在 clean document 上刷分，PaddleOCR-VL-1.5 通过 Real5-OmniDocBench 把 research community 的注意力引向 real-world physical distortion。

技术上有几个值得学习的 pattern：
1. **Anti-symmetric pairwise scoring + voting ranking** 处理 reading order 是 elegant 的 graph ranking 解决方案
2. **Special coordinate token** 避免数字 tokenization fragmentation 是 VLM 处理 spatial information 的好实践
3. **UACS** 把 data curation 从 instance level 提升到 cluster level，结合 visual diversity 和 uncertainty
4. **Distortion-Aware Augmentation** 针对性模拟 deployment scenario 的物理 distortion

0.9B 的参数量在 235B Qwen3-VL 和 Gemini-3 Pro 面前能拉开 5+ 个点的差距，再次验证了 specialized small model + targeted data + task-aware architecture 在 domain-specific task 上对 general large model 的优势。这对未来 VLM 设计的启示是：与其无脑 scale up，不如在 data quality、task formulation、architecture inductive bias 上下功夫。

References:
- PaddleOCR-VL: https://arxiv.org/abs/2510.14528
- DeepSeek-OCR: https://arxiv.org/abs/2510.18234
- MonkeyOCR: https://arxiv.org/abs/2506.05218
- MinerU2.5: https://arxiv.org/abs/2509.22186
- Dolphin: https://arxiv.org/abs/2505.14059
- OmniDocBench: https://arxiv.org/abs/2503.17213
- RT-DETR: https://arxiv.org/abs/2404.02907
- NaViT: https://arxiv.org/abs/2307.06304
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- CLIP: https://arxiv.org/abs/2103.00020
- CDM: https://arxiv.org/abs/2504.10479
- PaddleOCR 3.0: https://arxiv.org/abs/2507.05595
- HunyuanOCR: https://arxiv.org/abs/2511.19575
