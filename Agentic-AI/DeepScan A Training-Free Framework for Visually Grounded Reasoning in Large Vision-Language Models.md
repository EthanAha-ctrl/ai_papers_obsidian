---
source_pdf: DeepScan A Training-Free Framework for Visually Grounded Reasoning in
  Large Vision-Language Models.pdf
paper_sha256: 977dcfc6a75fc2ed57ba927e2a3e80673042b608689d0b05d5a7a346ad868562
processed_at: '2026-08-03T18:40:52-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DeepScan

好嘞 Karpathy，我换个频道，抛开公式，用大白话再过一遍这篇 paper。

---

## 这 paper 在干嘛

一句话：**教 LVLM 学会"找东西"再"答问题"**，而且不重新训练模型。

场景是这样的：你给 LVLM 一张超高清图，问 "guard 的手套什么颜色"。图里 guard 站在人群里，手套就一小撮像素。LVLM 直接看整张图回答，大概率会瞎蒙，因为手套的信号在整张图的 token 里被稀释得不成样子。

V* Bench 上目标 object 平均面积 < 0.05% 的图像，Qwen2.5-VL-7B 只有 74% accuracy。这就是痛点。

---

## 为什么直接 grounding 会挂

之前的方法套路都是 **top-down one-shot**：拿整张图丢给 GroundingDINO 之类的 detector，让它直接框出 "guard 的手套"。问题是 detector 在这种 fine-grained 场景下会犯两种经典错误：

1. **Attention sink**：注意力飘到画面里最显眼但根本无关的地方。比如问 "cyclist 的 box 什么颜色"，detector 框到了旁边的 "box sign"——因为 "box" 这个词太显眼了。

2. **Attention drift**：画面里有几个语义相似的物体，detector 分不清是哪个。比如三个穿红衣服的人，它框错了那一个。

Fig 10 里换不同 expert (GroundingDINO → BLIP) 还是犯同样的错。这说明 **top-down 一次性定位本身就有问题**，换工具救不了。

---

## DeepScan 的核心 idea：学人类找差异

paper 的灵感来自人类玩 "找差异" 游戏。你不会一眼看出两张图的差异，而是逐块扫描，发现某个局部有线索，再回到全图验证。

这叫 **bottom-up**：先在小窗口里找 cue，再把 cue 提升到全图 level 去恢复完整 evidence。

具体三步：

### 第一步：Hierarchical Scanning（分层扫描）

把大图切成一堆小 patch（576×576 或 768×768）。每个 patch 单独丢给 BLIP 看一遍，问它 "跟问题相关的区域在哪儿"。BLIP 用 GradCAM 给出一张 heatmap。

然后 Otsu 阈值化这张 heatmap，找出一团一团的 "cue 区域"。每个 cue 选一个点作为 SAM 的 point prompt，让 SAM 回到**原图**上做 segmentation，恢复出完整 evidence 的 mask。

这里有个 trick：选哪个点做 prompt 很关键。naive 选 centroid 对 U 形区域会落在洞里，他们用 "attention 强度 × 到边界距离" 的乘积选点，既保证在 cue 中心又保证在语义最显著的位置。

分割完做形态学后处理（closing 填洞 + dilation 扩一圈），保证 evidence 完整。然后下次扫描时跳过已覆盖区域，避免重复。

还有一个加速 trick：**只评估 top-k 最小的 candidate**。因为大 evidence LVLM 自己能识别出来，小 evidence 才需要显式 grounding。k=1 时保留 96% 性能 + 2× 加速。

### 第二步：Refocusing（重新聚焦）

第一步得到的 evidence view 不一定最优。可能 zoom 太近丢了 context，可能太远带了一堆 distractor。

他们让 LVLM 跟 visual expert 协作，搜一个 "最佳 view"。两个 action：
- **Zoom-In**：用 LangSAM 在当前 view 里 detect 跟问题相关的 object，crop 到 detection union
- **Zoom-Out**：以中心为锚点等比例放大 1.5 倍

naive 会用 MCTS 暴搜，但他们用几个 invariance（zoom-in 幂等、zoom-out 可合成、zoom-out 后 zoom-in 总比反过来差）把搜索空间剪成 **4 个 candidate**，LVLM 给每个 view 打个 binary 分（够不够答问题），再按 "够用 × 尺寸越小越好" 选最佳。

这步只加 0.8s latency 但能涨 4.7% accuracy，性价比巨高。

### 第三步：Evidence-Enhanced Reasoning

最后把 fine-grained evidence（小 crop，看颜色纹理 OCR）和 coarse-grained view（大 crop，看空间关系）一起塞进 LVLM 做 multi-image prompt，让它答题。

直觉分工：小图解决 "这是什么"，大图解决 "它在哪儿、跟谁有关系"。

---

## 为什么 bottom-up 比 top-down 强

paper Tab 7 做了个直接对比：
- Top-down one-shot：83.8%
- Bottom-up (Hierarchical Scanning)：**90.6%**

Fig 10 的可视化特别直观：top-down 的 attention 一直飘到错误的 "box sign" 上，而 bottom-up 因为先在小 patch 里锁定 "cyclist's box" 的 cue，再回到全图 segmentation，attention 准确对齐到正确 object。

直觉上，**在小窗口里做决策，干扰项少**。整图里有 100 个 distractor，单个 patch 里可能就 3 个，SNR 高了一个数量级。这就是 paper 反复强调的核心 insight。

---

## 效果怎么样

V* Bench 上 Qwen2.5-VL-7B baseline 74.3% → DeepScan **90.6%**，涨 16.3 个点。

几个对比：
- 超过 GPT-4o (66.0%)
- 超过 RL-based 的 DeepEyes (90.0%)、PixelReasoner (80.6%)、TreeVGR (85.9%)
- 接近 GPT-o3 (95.0%)

换 72B LVLM 后能到 **94.2%**，说明这方法能随 model scale 涨。

更厉害的是 efficiency：经过 vLLM batching 优化后，**90.1% accuracy / 3.1s latency**。对比 DeepEyes 89.0% / 6.9s——快 2.2 倍，准确率还更高，token cost 省 35%。

为啥？因为 DeepScan 是 **deterministic batched sampling**，而 DeepEyes 那种 agentic multi-turn tool execution 天然串行。这跟 LLM 里的 test-time scaling 思路一致——batched 推理 >> 多轮 agent。

---

## 几个有意思的 ablation

**Fig 5(left)**：evidence 越小，DeepScan 带来的 gain 越大。大 evidence LVLM 自己能搞定，小 evidence 才需要 explicit grounding。

**Fig 11**：grounding 精度（IoU）从 1/10 涨到 1（perfect crop），LVLM 性能反而**下降**。过度 zoom-in 会丢必要 context。这就是 Refocusing 的 reward 为什么加 size regularizer。

**Fig 11 还发现**：大 LVLM 在 perception 任务上随 grounding 精度收敛，但在 spatial reasoning 上 gap 持续存在。暗示可以用小 LVLM 做 evidence judgment，大 LVLM 做最终 reasoning，省 latency。

**Tab 2**：RL-based 方法在 second-order reasoning（Perspective Transform、Ordering）上几乎没 gain，只在 perception 任务上有提升。paper 据此 claim：**RL 只是 bias 了 LVLM 的 perception 行为，没根本强化 visual reasoning**。这判断挺犀利。

---

## 对你（Karpathy）的角度看

这 paper 的核心位置在 **test-time scaling 的视觉版本**。

o1/o3 路线是把 search policy 烤进 LLM weights，通过 RL 训练让模型学会 "内部思考"。DeepScan 走相反路线——**不训练，靠 external expert + structured search 在 inference time 实现 grounding**。

两条路线对比：
- RL-based (DeepEyes 等)：search policy 内化，需训练，latency 高（multi-turn）
- Training-free (DeepScan)：search policy 外化，无训练，能 batch，latency 低

更有意思的是 attention sink 这个现象。你在 nanoGPT 里应该见过 LLM 前几个 token 会变成 attention "黑洞"。DeepScan 在**视觉模态**观察到类似现象——LVLM 把 attention 集中到全局显著但无关的区域。这种 cross-modal 的相似性提示：**attention sink 可能是 transformer 架构本身的 inductive bias，不是语言特有的**。

还有 patchification 的角度。Qwen2-VL 用 mRoPE 做 dynamic resolution，LLaVA-NeXT 做 tile-level 切分。DeepScan 更激进——直接 patch-level 切分 + question-conditional patch size。这是 dynamic patchification 的极致版本。

---

## 几个我会 push 的问题

1. Search expert 为什么用 BLIP 不用 CLIP？Tab 3 显示 BLIP-ITM base/large 性能差不多（90.6 vs 90.1），说明什么？

2. Refocusing 的 reward 是 binary 的（够/不够），换成 soft reward（概率）会不会让 search 更平滑？

3. Patch 之间 overlap 吗？如果不 overlap，object 正好骑在 patch 边界上怎么办？

4. Multi-object reasoning 用 minimal enclosing bbox 是个简单 heuristic，两个 object 在对角时 bbox 里 90% 是 noise。他们提了 generative composition 但没实现。

5. "RL 没根本强化 visual reasoning" 这个 claim 只基于 Tab 2。会不会是 RL 训练数据不够、reward shaping 不够好？也许更精细的 reward 能突破。

6. 能不能 extend 到 video？用 optical flow 做 cue，frame-level evidence extraction。

---

## 一句话总结

**把 visual grounding 从 "一眼定位" 换成 "逐块扫描 + 逐步聚焦"**，因为局部 SNR 比全局 SNR 高得多。配上 expert 协作和 4-state 剪枝搜索，training-free 还能打到 90%+，效率还比 RL-based agentic 方法快 2 倍。

对你来说，这 paper 展示了 visual test-time scaling 的一种可行路径——**不靠 RL 烤进 weights，靠 structured search 在 inference time 实现**。跟 o1/o3 在 language 上的路线形成有趣的对照。

Reference:
- DeepScan (本篇, arXiv 待发布)
- V* Bench: https://arxiv.org/abs/2312.14135
- StreamingLLM (attention sink): https://arxiv.org/abs/2309.17453
- DyFo: https://arxiv.org/abs/2503.02401
- DeepEyes: https://arxiv.org/abs/2505.14362
- ZoomRefine: https://arxiv.org/abs/2506.01663
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- BLIP: https://arxiv.org/abs/2201.12086
- LangSAM: https://github.com/luca-medeiros/lang-segment-anything
- GradCAM: https://arxiv.org/abs/1610.02391
- Otsu's method: https://en.wikipedia.org/wiki/Otsu%27s_method

---

# DeepScan: 训练自由 的 Visually Grounded Reasoning 框架深度解析

Karpathy 你好!这篇 paper 我仔细读完了，挺有意思的一篇工作。让我从你比较熟悉的几个角度切入——attention mechanism、test-time scaling、bottom-up vs top-down perception——来 build your intuition。

---

## 1. Problem Setting: 为什么 LVLM 在 fine-grained visual reasoning 上会挂

这篇 paper 切入的点非常清楚: V* Bench 上 Qwen2.5-VL-7B 只有 74.3% 的 accuracy，而目标 object 平均面积小于图像的 0.05%。这是一个**信号-噪声比 (SNR)** 极端的 setting。当 LVLM 把整张 2246×1582 的图喂进 ViT 后，那个小 guard 的手套在 token 序列里大概只占极小一部分 attention weight。

paper 在 Sec 1 提到两个关键 failure mode:
- **Attention sink** [49] (Xiao et al., StreamingLLM): attention 不自觉地集中到一些 "anchor" token 上，而不是真正的 evidence region
- **Attention drift** [6]: 当图像里有语义相似但 spatially adjacent 的物体时，grounding 会漂到错误的对象上

Fig 10 的 qualitative 分析很说明问题: 用 GroundingDINO 的时候 attention 漂到了 "box sign" 而不是 "cyclist's box";换成 BLIP 之后还是漂到同一个错误区域。这说明 **top-down one-shot localization 本身有 paradigm-level 的 fragility**，跟用哪个 expert 关系不大。

这一点直觉上很合理: top-down 的方法本质是在一个高维、充满 distractor 的空间里做 one-shot 决策。这就像让你在一个 cluttered 房间里，闭着眼睛想一下钥匙在哪儿——大概率你会想到最显眼的位置，但钥匙往往在沙发缝里。

Reference: V* (Wu & Xie, CVPR'24) https://arxiv.org/abs/2312.14135 ; StreamingLLM https://arxiv.org/abs/2309.17453

---

## 2. 核心 Insight: Bottom-up Grounding Paradigm

paper 的核心贡献是把 grounding 从 "top-down one-shot" 换成 "bottom-up hierarchical"。

类比是人类做 spot-the-difference puzzle: 人类不会一眼看出全局差异，而是 scan 局部 patch，发现 cue，再回到 image level verify。这是一个 **locality-first, global-verify-second** 的 routine。

技术实现上，这变成三件事:
1. **Hierarchical Scanning**: patch-wise 找 cue → point-based proxy → image-level evidence recovery
2. **Refocusing**: LVLM × visual expert 协作，对 evidence view 做局部 zoom 校准
3. **Evidence-Enhanced Reasoning**: 用 hybrid evidence memory 把 fine-grained evidence 和 coarse-grained view 同时喂给 LVLM

让我逐个展开。

---

## 3. Hierarchical Scanning: 数学细节

### 3.1 Local Cue Exploration

给定 patch $p \in \mathbb{R}^{h \times w \times 3}$ 和 question $q$，先用 search expert (BLIP-ITM base) + GradCAM [29] 产生 attention map:

$$S_p = \text{SEARCH}(p, q) \in \mathbb{R}^{h \times w}$$

这里 $S_p$ 的每个元素 $S_p(i,j)$ 表示 patch 内位置 $(i,j)$ 相对于 question $q$ 的 relevance。GradCAM 的本质是: 用最后一层 cross-attention 或 classification head 对 input pixel 的 gradient，加权最后一个 feature map，得到 spatial relevance map。

然后用 **Otsu's method** [22] 做自适应阈值化:

$$S_p^+ = \mathbb{I}(S_p \geq T_p^*), \quad T_p^* = \text{OTSU}(S_p)$$

Otsu 是 1979 年的经典图像处理方法，本质是最大化前景/背景两类方差的一个 threshold。这里 $T_p^*$ 是 scalar，$\mathbb{I}(\cdot)$ 是 indicator function，把 continuous attention map 二值化成 cue mask。

接下来在 $S_p^+$ 上找 connected components $\{G_p^k\}_{k=1}^K$——也就是 graph 理论意义上的 4-连通或 8-连通 region。每个 $G_p^k$ 是一个 candidate cue。

### 3.2 Point-based Proxy 的设计

这里有个很精巧的工程细节。对每个 cue $G_p^k$，他们想找一个 "interior point" 作为 SAM 的 point prompt。但 naive 的 centroid 对 U-shape region 会落在洞里。所以公式 5 是个乘积:

$$\mathcal{C}_p = \Big\{ c_p^k \mid c_p^k = \arg\max_{c \in G_p^k} \tilde{S}_p(c) \cdot \tilde{d}(c, \partial G_p^k), \quad |G_p^k| \geq \tau \Big\}$$

变量解释:
- $c \in G_p^k$: cue region 内的一个 pixel 位置
- $\tilde{S}_p(c)$: 在 $c$ 处归一化后的 attention score (semantic 信息)
- $\tilde{d}(c, \partial G_p^k)$: 在 $c$ 处归一化后的到 cue 边界 $\partial G_p^k$ 的距离 (topological 信息)
- $\tau$: 面积阈值，过滤掉噪声小 cue，paper 中设为 50 pixels
- $\arg\max$ 是在 $G_p^k$ 内部搜索使乘积最大的点

$\tilde{S}_p$ 和 $\tilde{d}$ 都是 normalized 的——paper 没明说具体归一化方式，但从 Algorithm 1 看应该是 min-max 之类。

这个设计的直觉: 既要在 cue 内部尽量靠中心 (高 $\tilde{d}$，避免 SAM 把边界切掉)，又要在 semantic 最显著的位置 (高 $\tilde{S}_p$，避免漂到无关子区域)。乘积形式比加权和少了 weight tuning，比较 robust。

Tab 5(left) 的 ablation 验证了这个设计:
- Centroid only (T only): 84.3%
- Chebyshev center only (T): 91.3%  
- Attention peak only (S): 87.8%
- Ours (S × T): **93.0%** on Attribute

### 3.3 Multi-Scale Evidence Extraction

拿到 proxy $c_p'$ (lift 到 image coordinates 后) 之后，用 LangSAM 做 point-prompt segmentation:

$$m = \text{SEGMENT}(I, c_p') \in \{0, 1\}^{H \times W}$$

但 single point prompt 表达力有限，mask 经常有 interior holes 或不完整 context。所以做形态学后处理:

$$m^+ = (m \bullet \mathcal{K}) \oplus \mathcal{S}_r$$

变量解释:
- $\bullet$: morphological closing = dilation followed by erosion，用 5×5 flat kernel $\mathcal{K}$，作用是 seal 内部洞
- $\oplus$: dilation，用 disk kernel $\mathcal{S}_r$ (半径 $r=20$)，作用是向外扩展一圈，保留 surrounding context
- $m^+$: 增强后的 binary mask

然后 crop $m^+$ 的 minimal enclosing bbox $b$ 得到 evidence $e$。

这里有个 deduplication: 任何落在已 visited mask 内的 proxy 被过滤掉 (Eq 8):

$$\mathcal{C}_p' \gets \{ c \in \mathcal{C}_p' : m^+(c) = 0 \}$$

同时全局 image 被更新成 $I \leftarrow I \odot (1 - m^+)$，避免重复计算。

### 3.4 Heuristic Acceleration: top-k smallest

Fig 5(left) 是个很重要的发现: **evidence 越小，gain 越大**。直觉上大 evidence LVLM 自己就能 detect 出来，小 evidence 才需要 explicit grounding。

所以他们保留 top-$k$ smallest candidate，$k=10$。Fig 5(right) 显示 $k=1$ 时保留 96% 性能并 2× speedup——这暗示了 evidence candidate 数量上的 long-tail 分布。

Reference: GradCAM https://arxiv.org/abs/1610.02391 ; Otsu 原始 paper 1979 ; LangSAM https://github.com/luca-medeiros/lang-segment-anything

---

## 4. Refocusing: 搜索空间设计的数学美

这一节我个人觉得是 paper 里最优雅的部分，因为它把一个本来需要 MCTS/A* 的搜索问题，通过几个 invariance 假设压缩成一个 4-state 集合。

### 4.1 状态空间定义

从 Hierarchical Scanning 得到 evidence set $\mathcal{E}$，初始化:

$$V_1 = \text{CROP}(I, b_m)$$

$b_m$ 是所有 evidence bbox 的最小外接矩形。注意 $V_1$ 不一定是最优 view——可能 evidence 之间 spatially adjacent 时，$V_1$ 包含太多 distractor; 也可能 evidence 不完整，需要 zoom out 找 context。

两个 action:
- **Zoom-In**: $\text{IN}(V, q) = \text{CROP}(V, \text{DETECT}(V, q))$，用 LangSAM detection 缩小到 query-conditioned detection 的 union
- **Zoom-Out**: $\text{OUT}(V, s) = \text{CROP}(I, \text{SCALEBOX}(V, s))$，以 $V$ 中心 isotropically 扩展 $s$ 倍

### 4.2 Reward Function

$$R(V) = \mathbb{I}_{V \sim q} \cdot HW/hw$$

变量解释:
- $\mathbb{I}_{V \sim q} \in \{0, 1\}$: LVLM 判断 view $V$ 是否包含足够 answer $q$ 的 evidence
- $H, W$: 原图高宽
- $h, w$: view $V$ 的高宽
- $HW/hw$: size regularizer，鼓励更小的 view (在 evidence complete 的前提下)

这个 reward 的妙处在于: 它 discourage naive "zoom out to full image" 的 cheat solution，因为 $HW/hw$ 会很小。同时又用 $\mathbb{I}_{V \sim q}$ 硬性保证 evidence completeness。

### 4.3 搜索空间剪枝: 四个 invariance

paper 列了三个性质 (Eq 12-16) 用来剪枝:

**Idempotency of IN** (Eq 12):
$$\text{IN}(\text{IN}(V_1, q), q) = \text{IN}(V_1, q)$$

意思是 zoom-in 之后再 zoom-in 是幂等的，因为 detection 是 query-conditioned 的，第二次 detect 还是同样那些 object。

**Compositionality of OUT** (Eq 13):
$$\text{OUT}(\text{OUT}(V_1, s_1), s_2) := \text{OUT}(V_1, s_1 s_2)$$

因为 $V_1$ 占 $I$ 很小一部分，连续 zoom out 等价于一次大 zoom out。

**Dominance of IN** (Eq 14):
如果 $\mathbb{I}_{V_1 \sim q} = 1$ (即 $V_1$ 已包含完整 evidence)，那么:
$$R(\text{OUT}(\text{IN}(V_1, q), s)) \leq R(\text{IN}(V_1, q))$$

直觉: 加 context 不增加 reward，反而降低 size regularizer。

**Monotonicity of OUT-IN** (Eq 16):
如果 zoom-out 能为 zoom-in 后的 view 补 context，那也能直接为 $V_1$ 补 context:
$$R(\text{OUT}(\text{IN}(V_1, q), s)) \leq R(\text{IN}(\text{OUT}(V_1, s), q))$$

推论: $\text{OUT}(\text{IN}(V_1, q), s)$ 这个状态可以剪掉。

### 4.4 4-State 搜索空间

剪枝之后:
$$\mathcal{V} = \{V_1, V_2, V_3, V_4\}$$
$$V_2 = \text{IN}(V_1, q), \quad V_3 = \text{OUT}(V_1, s), \quad V_4 = \text{IN}(V_3, q)$$

$s = 1.5$ via grid search。

Tab 6(right) 给出了对比:
- MCTS: search length 2.24, budget 4
- A*: 3.07, budget 4
- Ours: **1.87, budget 4**

也就是说 Refocusing 平均只需要 ~2 次 expansion 就找到 oracle-optimal state。这是个非常 tight 的 bound。

这一点让我联想到你之前讲过的 test-time compute scaling 的工作——这里本质上是把搜索空间用先验知识剪枝到极简，每次 expansion 都用 LVLM 做 binary 判断 (cost ~一次 forward pass)。可以理解为 "structured search with domain-specific pruning"。

---

## 5. Evidence-Enhanced Reasoning: Hybrid Memory

最后一步是把 fine-grained evidence 和 coarse-grained view 组合成一个 multi-image prompt:

$$\mathcal{H} = \{ e, V^* \mid (b, e) \in \mathcal{E}, V^* = \arg\max_{V \in \mathcal{V}} R(V) \}$$

Materialize 成 $[e_1, ..., V^*]$ 的有序 multi-image prompt，然后:

$$\mathcal{A} = \text{REASON}(\mathcal{H}, q)$$

这里 paper 没有给特别复杂的 fusion 机制，就是 concat images 让 LVLM 自己处理。直觉上:
- Fine-grained evidence $e_i$: 解析 object attributes (颜色、纹理、OCR text)
- Coarse-grained view $V^*$: 推理 spatial relation、second-order reasoning

这是 multi-granularity 的分工——evidence resolution vs context resolution 的 trade-off 用两个 view 一起解决。

---

## 6. 实验结果的关键 takeaway

### 6.1 主表 (Tab 1)

V* Bench:
- Qwen2.5-VL-7B baseline: 74.3%
- DeepScan (k=10): **90.6%**, +16.3% absolute
- DeepScan-72B (k=∞): **94.2%**, 比 72B baseline 84.8% 涨 9.4%
- 超过 DeepEyes (90.0%), PixelReasoner (80.6%), TreeVGR (85.9%)
- 超过 GPT-4o (66.0%), 接近 GPT-o3 (95.0%)

HR-Bench (4K/8K): 平均 +3.6% / +2.6% over DyFo。

Tab 9 显示 scaling 到 72B 后 Spatial subset 从 80.9% 跳到 93.4%——这是一个很显著的 scaling 效应，说明大 LVLM 能更好利用 multi-granular evidence。

### 6.2 Ablation 关键发现

**Tab 4 - 范式对比**: 
- Detection-based grounding (DyFo style): 82.2%
- Hierarchical Scanning w/o post-processing: 87.4%
- Hierarchical Scanning (full): **90.6%**

morphological post-processing 贡献 +3.2% 并且从 32.1s 加速到 24.5s——加速是因为 dedup 减少了 LVLM 判断次数。

**Tab 7 - bottom-up vs one-shot**: 90.6% vs 83.8%, latency 24.5s vs 20.4s。bottom-up 多 4s latency 但涨 6.8% accuracy。值得注意的是 one-shot latency 优势不大——因为 top-k 过滤后 LVLM 判断次数差不多。

**Fig 11 - grounding precision vs reasoning**: 这是个非常重要的 ablation。IoU 从 1/10 提到 1 (perfect crop) 时，性能反而下降——**过度 zoom-in 会丢失必要 context**。这印证了 Refocusing 的 reward 设计为什么需要 size regularizer。

**Fig 11 还揭示**: 大 LVLM 在 spatial reasoning 上的 gap 持续存在，但在 perception 上收敛。这暗示**用小 LVLM 做 evidence judgment、大 LVLM 做 final reasoning** 是一个 latency-friendly 的架构选择。

### 6.3 Performance-Latency (Tab 10)

经过 vLLM backend + batch 优化:
- Qwen2.5-VL-7B baseline: 75.4% / 0.4s
- DeepEyes: 89.0% / 6.9s / 13k tokens
- DeepScan: **90.1% / 3.1s / 8.4k tokens**

DeepScan 比 DeepEyes 快 2.2× 且省 35% tokens。这是 deterministic batched sampling 相对 agentic multi-turn tool execution 的优势。

---

## 7. 与你之前 work 的联想

Karpathy，读这篇 paper 时我忍不住想到几个 connection:

### 7.1 Test-Time Compute Scaling

DeepScan 本质上是 test-time scaling 的一个 instance——不训练 LVLM，而是在 inference time 用 expert + search 来增强。这跟你在 Eureka Labs / 最近 tweet 讨论的 "inference-time compute" 主题一致。但这里的 scaling 是 **structured** 的（基于先验剪枝的搜索空间），不是 free-form CoT 的那种 scaling。

类比: o1/o3 是 LLM 内部的 test-time scaling，DeepScan 是 LLM × tool 的外部 test-time scaling。两者可以叠加——paper 也提到 GPT-o3 在 V* 上有 95.0%，DeepScan 在 72B 上 94.2% 已经接近。

### 7.2 nanoGPT 视角的 attention sink

你在 nanoGPT 里实现 attention 时应该注意到，前几个 token 经常会变成 "sink"。Xiao et al. 2023 的 StreamingLLM paper 把这个形式化了。DeepScan 在视觉模态观察到类似现象: LVLM 把 attention 集中到一些全局显著但无关的区域。这种 cross-modal 的相似性提示 attention sink 可能是 transformer 架构本身的 inductive bias，不是语言特有的。

### 7.3 ViT 的 patchification 

paper 里 patch size 576×576 (single-object) 或 768×768 (multi-object) 的选择让我想到 ViT 的 patchification 问题 (Wang et al. [35], "Scaling laws in patchification")。DeepScan 用 question type 来 condition patch size，相当于一个 dynamic patchification。这跟 LLaVA-NeXT、Qwen2-VL 的 dynamic resolution 路线一脉相承，但更激进——直接 patch-level 切分而不是 tile-level。

### 7.4 BLIP / CLIP as search expert

他们用 BLIP-ITM base 作为 search expert，本质是利用 image-text matching 的 contrastive head 作为 relevance detector。这跟 CLIP-based open vocabulary detection 一脉相承。GradCAM 在 BLIP 上的应用其实是一个 "weakly supervised grounding" 的 hack——但效果出奇地好，因为 BLIP 的 cross-attention 已经 encode 了 image-text alignment。

### 7.5 关于 "think with images" 

你之前对 o3 "thinking with images" 的能力很感兴趣 (e.g., 你的 tweet)。DeepScan 是一个 training-free 的 alternative: 不靠 RL 内化 visual search policy，而是用 expert + search 显式实现。两种路线的对比:
- RL-based (DeepEyes, PixelReasoner): 把 search policy 烤进 LLM weights，需要 RL 训练成本
- Training-free (DeepScan, DyFo): 用 expert 实现 search policy，无训练成本，但 latency 更高

paper Tab 2 显示在 second-order reasoning (Perspective Transform, Ordering) 上 RL 没显著 gain——这暗示 **RL 主要 bias 了 perception 行为，没根本强化 visual reasoning**。这是个挺尖锐的观察。

---

## 8. Limitations 和 Future Directions

paper 在 Supp. A 讨论了两种 failure:

**Grounding failure**: 多个 visually similar objects 在 evidence 邻域内时，expert 可能 propose 错 evidence，LVLM 也可能 misjudge。

**Reasoning failure**: 多 evidence spatially 远离时，minimal enclosing bbox 会引入大量 inter-evidence noise。他们提出未来用 generative composition 重新 layout evidence。

这一点让我想到你之前讲过的 "LLM as world model" 的局限——layout composition 本质上要保留 spatial relation，这跟 diffusion-based image editing 的 object repositioning 类似。

还有 inference latency 问题。Tab 10 显示优化后 3.1s，相对 baseline 0.4s 还是 7.75× 慢。paper 提到未来想做 adaptive patch size——简单 case 不需要细 patch，能省时间。

---

## 9. 几个我想 push 的问题

如果是我审这篇 paper，我会问:

1. **Search expert 选择**: 为什么用 BLIP-ITM 而不是 CLIP? BLIP 的 ITM head 和 CLIP 的 contrastive head 在 GradCAM 上有显著差异吗? Tab 3 显示 BLIP-ITM base/large 性能差不多 (90.6 vs 90.1)，这暗示什么?

2. **Reward 的 binary 形式**: $\mathbb{I}_{V \sim q}$ 是 binary 的，会不会太 hard? Soft reward (e.g., probability) 会不会让 search 更平滑?

3. **Hierarchical Scanning 的 patch overlap**: paper 没提到 patch 是否 overlap。如果不 overlap，跨 patch 的 evidence 会不会丢? 比如一个 object 正好在 patch 边界上。

4. **Multi-object reasoning 的 evidence fusion**: minimal enclosing bbox 是个简单 heuristic。如果两个 object 在 image 对角，bbox 包含 90% noise。paper Supp. A 提了 generative composition 但没实现。

5. **Generalization to video**: bottom-up paradigm 能不能 extend 到 temporal dimension? 比如用 optical flow 作为 cue，然后 frame-level evidence extraction。

6. **关于 RL 路线的判断**: paper 说 "RL does not fundamentally strengthen LVLMs in visual reasoning; rather, it biases LVLMs toward perception behaviors." 这个 claim 的证据是 Tab 2 上 second-order reasoning 没显著 gain。但会不会是因为 RL 训练 data 不够? PixelReasoner, DeepEyes 都是 relatively early 的 RL work，maybe with better reward shaping 能突破。

---

## 10. 总结

DeepScan 的核心贡献是**把 visual grounding 从 top-down one-shot 换成 bottom-up hierarchical**，这个 paradigm shift 在 attention sink/drift 这些 failure mode 上有根本性的优势。技术实现上三个亮点:

1. **Point-based proxy** (公式 5) 用 semantic × topological 乘积避免了 U-shape region 的 centroid 问题
2. **Refocusing 的 4-state search space** 用三个 invariance 把 MCTS 剪枝成 deterministic 4-way choice
3. **Hybrid evidence memory** 把 multi-granularity information 显式分工给 LVLM

加上 vLLM batching 优化后 90.1% / 3.1s 的 performance-latency profile，作为 training-free 方法相当 solid。

对你 (Karpathy) 来说，这篇 paper 可能最有意思的角度是: **它展示了 test-time scaling 的 visual 版本**——不训练 LVLM，而是用 expert + structured search 来 inject grounding ability。这跟 language-only LLM 的 test-time scaling (o1/o3 路线) 形成 cross-modal 对照。也许 visual reasoning 的 inference-time scaling 路线，比 RL-based in-weight policy 更 scalable 且 generalizable。

主要参考链接:
- DeepScan paper (本篇): ECCN 等，arXiv 待发布
- V* Bench: https://arxiv.org/abs/2312.14135
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- BLIP: https://arxiv.org/abs/2201.12086
- LangSAM: https://github.com/luca-medeiros/lang-segment-anything
- GroundingDINO: https://arxiv.org/abs/2303.05499
- StreamingLLM (attention sink): https://arxiv.org/abs/2309.17453
- GradCAM: https://arxiv.org/abs/1610.02391
- DyFo: https://openaccess.thecvf.com/content/CVPR2025/papers/Li_DyFo... (CVPR'25)
- DeepEyes: https://arxiv.org/abs/2505.14362
- ZoomRefine: https://arxiv.org/abs/2506.01663
- PixelReasoner: https://arxiv.org/abs/2505.15966
- TreeVGR: https://arxiv.org/abs/2507.07999
- LLaVA: https://arxiv.org/abs/2304.08485
- BLIP-2: https://arxiv.org/abs/2301.12597
- Scaling laws in patchification: https://arxiv.org/abs/2503.13005 或相关
- Hierarchical Vision Transformer (相关思路): https://arxiv.org/abs/2103.12730

希望这个分析对你 build intuition 有帮助!如果你对某个 component (e.g., Refocusing 的搜索空间剪枝、或 attention sink 在视觉模态的表现) 想更深入聊，欢迎继续追问。
