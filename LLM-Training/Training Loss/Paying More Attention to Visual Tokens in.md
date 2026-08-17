---
source_pdf: Paying More Attention to Visual Tokens in.pdf
paper_sha256: 9edfd90137dff3df16972f2961c84fe5e9ae86620bbbb9ea5915aa9ebed68beb
processed_at: '2026-08-06T02:28:20-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VISE 用人话说

## 1. 故事的起点：一个让人头疼的现象

你知道现在这些 self-evolving LMM 怎么训练吗？让模型自己跟自己玩——一个 role 出题，一个 role 答题，reward 就是"两个 role 的答案一致"。听起来很美，unsupervised，不用人标注。

**但这里有个巨大的漏洞**。

假设我问模型："这张图里滑板停在什么上面？"模型答"ramp surface"。换个完全不同的图，滑板明明停在 metal ledge 上，模型还是答"ramp surface"。为什么？因为"滑板 + ramp"在 training data 里 co-occur 太频繁了，decoder 学会了走 language prior shortcut——它根本没看图，靠统计规律蒙。

问题在于，reward 只检查"答案一致性"，不检查"答案是不是来自 visual evidence"。模型靠 prior 答得一致，reward 照样给高分。于是 visual under-conditioning 被 reinforced，不是被 fixed。

这就像考试只看答案对不对，不查有没有抄袭。学生学会了抄统计答案，而不是真的会做题。

## 2. VISE 的"啊哈"时刻

Paper 作者想明白了一件事：**别再 optimize 答案了，直接 optimize 模型看不看图**。

怎么知道模型看不看图？两个 trick：

### Trick 1: 把图转一下

如果模型真的在看图，我把图旋转 10 度，它预测的 bounding box 应该跟着旋转。如果模型走 prior，它输出的 box 跟原图没区别——因为它根本没 condition 在 pixel 上。

这就是 **Geometric Invariance Reward**。公式：

$$\mathcal{R}_{\mathrm{geo}} = \frac{\mathrm{GIoU}(B_{\mathrm{proj}}, B_{\mathrm{new}}) + 1}{2}$$

- $B_{\mathrm{proj}}$：原图预测的 box 经过数学变换后的"应该在哪里"
- $B_{\mathrm{new}}$：模型在变换后的图上实际预测的 box
- GIoU：两个 box 的重合度，范围 [-1, 1]
- 加 1 除 2：映射到 [0, 1] 作为 reward

如果模型看图了，两个 box 应该高度重合，reward 接近 1。如果模型走 prior，两个 box 不沾边，reward 接近 0。

### Trick 2: 把关键区域糊掉

这个更绝。模型说"这里有个 skateboard"，好，我把模型预测的那个区域用 Gaussian blur 糊掉（σ=25，足够模糊到看不出是什么，但保留 spatial structure）。然后再问模型："这里还有 skateboard 吗？"

如果模型真的在看图，它应该回答"没了"。如果模型走 prior，它还是会说"有"——因为它根本不依赖那个区域的 pixel evidence。

这就是 **Semantic Invariance Reward**：

$$\mathcal{R}_{\mathrm{sem}} = \begin{cases} 1.0 & \text{if } v = 1 \text{ and } \tilde{v} = 0 \\ 0.0 & \text{otherwise} \end{cases}$$

- $v$：模型在原图上说 object visible（=1）
- $\tilde{v}$：模型在 ghosted 图上说 object visible（=0）
- 只有"原来看到、糊掉后看不到"才给满分

这个 reward 直接攻击 evidence binding：模型必须证明它的判断依赖于那个区域的 visual content，而不是 language prior。

## 3. 为什么这招特别狠

之前的 self-evolving 方法有个结构性问题：Proposer-Solver 是个 implicit minimax game。两个 role 目标对立，联合训练不稳定。Proposer 容易 collapse 到 trivial query（保证 Solver 答对），Solver 容易 overfit Proposer 分布（无法 generalize）。

VISE 干脆**单模型自己玩**。一个 well-pretrained LMM 已经有足够的 visual knowledge 来自己 formulate query + predict box。不需要 Proposer，不需要 Solver，不需要 external reward model，不需要 annotation。

训练流程就是：
1. 模型看图，自己问自己"图里有什么 prominent object"→ 生成 query q
2. 预测 bounding box $B_{\mathrm{orig}}$
3. 几何变换图 → 预测新 box → 算 $\mathcal{R}_{\mathrm{geo}}$
4. Ghosting 原图区域 → 判断 visibility → 算 $\mathcal{R}_{\mathrm{sem}}$
5. $\mathcal{R}_t = 0.5 \mathcal{R}_{\mathrm{geo}} + 0.5 \mathcal{R}_{\mathrm{sem}}$
6. REINFORCE + adaptive KL 更新

整个 pipeline 完全 self-supervised，只需要 raw unlabeled images。4000 张图，4000 步，16 小时训完 Qwen3-VL-2B。

## 4. 效果有多炸裂

**COCO Captioning (CIDEr)**：

| 方法 | 2B | 4B | 8B | 32B |
|------|----|----|-----|------|
| Base | 21.54 | 27.35 | 29.01 | 33.45 |
| EvoLMM | 20.84 (-0.70) | 30.53 (+3.18) | 29.84 (+0.83) | 34.01 (+0.56) |
| iReasoner | 20.93 (-0.61) | 30.68 (+3.33) | 33.26 (+4.25) | 37.62 (+4.17) |
| VisionZero-RW | 25.58 (+4.04) | 31.13 (+3.78) | 37.42 (+8.41) | 41.21 (+7.76) |
| **VISE** | **38.39 (+16.85)** | **39.65 (+12.30)** | **38.49 (+9.48)** | **42.17 (+8.72)** |

注意几个事：

1. **EvoLMM 在 2B 上竟然 drop -0.70**。这就是 visual under-conditioning 的铁证——answer agreement reward 反而强化了 language prior，captioning 直接 regress。
2. **VISE 的 gain 是最强 baseline 的 4-7 倍**。+16.85 vs +4.04，不是一个量级的提升。
3. **Gain 随 model size 衰减**：+16.85 → +12.30 → +9.48 → +8.72。这很 intuitive——大模型 pretraining 时已经 consolidate 了 visual conditioning，headroom 小了。

**Hallucination (Chair-I, 越低越好)**：

| 方法 | 2B Chair-I | 2B POPE Acc |
|------|------------|-------------|
| Base | 13.21 | 89.01 |
| EvoLMM | 12.99 (-0.23) | 87.59 (-1.42) |
| iReasoner | 12.98 (-0.23) | 87.70 (-1.31) |
| VisionZero-RW | 10.22 (-2.99) | 88.70 (-0.31) |
| **VISE** | **8.21 (-5.00)** | **90.03 (+1.02)** |

VISE 是唯一同时降 hallucination 又升 POPE accuracy 的方法。EvoLMM/iReasoner 降了 Chair 但 POPE 掉了——sentence-level 幻觉减少，但 binary object-presence 判断反而变差，说明改善不一致。VISE 的 ghosting reward 直接惩罚"模型在 evidence 被移除后仍坚持说有"，所以两个指标一起改善。

## 5. Mechanism：到底改了什么

这是这篇 paper 最精彩的部分——它不只 show 结果，还 show **模型内部发生了什么变化**。

### Generation-time Visual Attention

测量生成每个 token 时，decoder 每一层分配给 image tokens 的 attention 比例。

结果：VISE 在 mid-to-late decoder layers（第 15-25 层）对 image tokens 的 attention 明显增加，mean gain +2.84%（2B），per-sample peak +5.09%。

为什么是 mid-to-late layers？因为早期层做 visual encoding，后期层做 semantic generation decisions。Language prior shortcut 恰恰发生在后期层——模型"决定"生成什么词的时候。VISE 把这个 decision 从 prior-driven shift 到 image-conditioned。

### CKA Similarity

Centered Kernel Alignment 衡量 original view 和 geometrically augmented view 之间的 representation similarity。

- 2B：gains 集中在 final layers（layer 27 peak Δ=+0.069），100% win-rate
- 4B：gains 分布在 layers 19-33（peak Δ=+0.253），win-rate 从 60% 涨到 100%

这说明 VISE 真的让模型在 geometric transformation 下保持 representation consistency，而且发生在生成决策的关键层。

## 6. Ablation 的故事

| 配置 | COCO CIDEr | Chair-I |
|------|------------|---------|
| Base | 21.54 | 13.21 |
| $\mathcal{R}_{\mathrm{geo}}$ only | 26.37 (+4.83) | 11.86 (-1.35) |
| $\mathcal{R}_{\mathrm{sem}}$ only | 35.53 (+13.99) | 9.06 (-4.15) |
| Full | 38.39 (+16.85) | 8.21 (-5.00) |

**$\mathcal{R}_{\mathrm{sem}}$ 贡献了大部分 gain**（+13.99 vs +4.83），这很 intuitive——captioning 和 hallucination 的核心是 evidence binding，ghosting reward 直接攻击这个。

但 $\mathcal{R}_{\mathrm{geo}}$ 不是没用。Full model 比 $\mathcal{R}_{\mathrm{sem}}$ alone 多 +2.86 CIDEr，说明 geometric consistency 提供了 complementary signal：它 catch 的是"semantically 对但 spatially 不稳定"的 case。

两个 reward 覆盖 visual under-conditioning 的两个不同 dimension：
- $\mathcal{R}_{\mathrm{geo}}$：spatial consistency
- $\mathcal{R}_{\mathrm{sem}}$：evidence sensitivity

## 7. 为什么 LoRA 比 Full Fine-tuning 好

这个发现有点反直觉。Table S2：

| 方法 | COCO CIDEr |
|------|------------|
| FFT (COCO train) | 32.80 ± 0.45 |
| LoRA (COCO train) | 38.49 ± 0.67 |

LoRA 全面碾压 FFT。Paper 的解释：unsupervised reward 是 noisy 的，传到 vision encoder 会 destabilize 已经高质量的 representations。冻结 encoder，只更新 projector + decoder，更稳定且 sufficient。

这跟 RLAIF 领域的一个共识吻合：preference signal 越 noisy，越要限制参数更新范围。LoRA 的 low-rank constraint 本身就是 regularizer。

## 8. 我看到的一些联想

### 跟 RLHF 的关系

VISE 的 reward 设计让我想到 RLHF 里 preference model 的作用。传统 RLHF 用 human preference 训 reward model，VISE 用 invariance property 直接构造 reward signal。这跟 Anthropic 的 Constitutional AI 有点像——用规则代替 human feedback，但 VISE 更进一步：用物理/几何不变性作为 self-supervised signal。

### 跀 Contrastive Learning 的关系

Geometric invariance reward 本质上是 contrastive learning 的变种：
- Positive pair: 原图和变换图应该 produce consistent localization
- Negative signal: GIoU 低就 penalize

SimCLR、BYOL、MoCo 都用 augmentation invariance，但它们是 representation-level。VISE 把这个 idea 搬到 generation-level——要求 output behavior 在 transformation 下保持一致。

### 跀 Active Learning 的关系

Ghosting 操作有点像 active learning 里的 ablation test：移除关键 evidence 看模型还信不信自己的判断。这跟 explainable AI 里的 occlusion sensitivity 是同一个 family 的 idea，但 VISE 用它做 training signal 而非 evaluation tool。

### 跀 LLM 的 Chain-of-Thought

VISE 让模型先 predict box，再 verify visibility。这跟 CoT 的"先推理再回答"结构相似。但 VISE 的"推理"是 spatial grounding，不是 verbal reasoning。会不会有 multimodal CoT + invariance reward 的组合？

### 跀 Test-time Compute

VISE 是 training-time method。但 ghosting idea 可以直接用于 test-time：生成 answer 后，ghost 关键 region，看模型还信不信自己。如果 model 不信了，说明 answer 是 image-grounded；如果还信，可能是 hallucination。这可以作为 test-time verification signal。

## 9. 局限性

1. **Scale effect**：32B 上 gain 只有 +8.72 CIDEr，vs 2B 的 +16.85。大模型 headroom 小，但也可能说明 invariance reward 对已经 strong 的模型边际效用递减。

2. **Training data 依赖**：虽然 Table S2 证明 Objects365 也 work，但 4000 张图还是相对小。如果 scale 到 100k 图，gain 会更大还是 saturate？Paper 没答这个问题。

3. **只测了 image**：Video、3D、audio modalities 没碰。Geometric invariance 在 video 里可以扩展到 temporal invariance，semantic invariance 可以扩展到 frame ablation。

4. **Reward sparsity**：$\mathcal{R}_{\mathrm{sem}}$ 是 binary 的（0 或 1），如果模型一开始就 v=0（self-generated query 质量差），这步 reward 直接是 0，没梯度。Paper 说"noisy localization steps 自然被 down-weight"，但没量化有多少样本被浪费。

## 10. 一句话总结

**VISE 告诉我们：与其让模型答得一致，不如逼模型证明它看了图。**

把训练目标从 "answer agreement" 换成 "visual conditioning policy regularization"，用几何不变性和证据敏感性两个 self-supervised signal，就能让 decoder 从 language-prior-driven shift 到 image-conditioned decoding。4000 张无标注图，16 小时，18 个 benchmark 全面提升，没 tradeoff。

这不是 incremental improvement，是 paradigm shift——self-evolving multimodal training 的目标函数该重新定义了。

---

**主要参考链接**：
- Paper project: https://mbzuai-oryx.github.io/VISE/
- GitHub: https://github.com/mbzuai-oryx/VISE  
- HuggingFace: https://huggingface.co/shravvvv/VISE
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- GIoU original: https://arxiv.org/abs/1902.09630
- EvoLMM baseline: https://arxiv.org/abs/2511.16672
- VisionZero baseline: https://openreview.net/forum?id=s00SNXREV6
- CKA similarity: https://arxiv.org/abs/1905.00414
- LoRA: https://openreview.net/forum?id=nZeVKeeFYf9
- lmms-eval framework: https://github.com/EvolvingLMMs-Lab/lmms-eval

---

# VISE: Visual Invariance Self-Evolution 深度讲解

## 1. Paper核心洞察

这篇paper诊断出self-evolving LMM的一个根本性失败模式：**visual under-conditioning**。具体来说，现有的self-evolving方法（EvoLMM、iReasoner、VisPlay、VisionZero）使用Proposer-Solver multi-role self-play + self-consistency reward，结果模型可以靠**statistical language priors**达到answer agreement，decoder对visual tokens的attention不足。论文用Figure 1展示了一个很直观的例子：问"skateboard实际停在哪里"，baseline说"ramp surface"或"concrete ground"（语言先验常见的搭配），VISE能准确说出"metal ledge"。

Paper的Project page: https://mbzuai-oryx.github.io/VISE/
GitHub: https://github.com/mbzuai-oryx/VISE
HuggingFace: https://huggingface.co/shravvvv/VISE

## 2. 失败模式分析：为什么prior self-evolving方法不够

**结构性问题1：Proposer-Solver implicit minimax game**

Proposer和Solver目标对立，联合优化时不稳定。实践中常常一方dominate：Proposer collapse到trivial query保证agreement，或者Solver overfit Proposer分布无法generalize。系统陷入local minima，缺少external intervention难以纠正。

**结构性问题2：answer consistency ≠ visual grounding**

Reward围绕answer correctness隐式假设"self-consistent outputs reflect improved visual understanding"。但decoder如果走language priors shortcut，仅靠statistical co-occurrence就能达到高answer agreement，visual under-conditioning反而被reinforced。

**实验验证**

Table 1显示EvoLMM在Qwen3-VL-2B上COCO CIDEr drop -0.70，NoCaps drop -0.77，Flickr30k drop -0.94。iReasoner同样regress。这无法用distribution mismatch解释，因为answer agreement reward根本不要求模型描述它实际看到的内容。

## 3. VISE方法详解

### 3.1 Problem Formulation

- 输入：无标注图像集合 X = {x}，无query、无bounding box、无category label
- 单模型π，每步执行：
  1. 自问自答：生成natural-language localization query q（描述场景中一个prominent, spatially unambiguous object）
  2. 预测bounding box B = (x_1, y_1, x_2, y_2)定位queried object
- 坐标空间：normalized [0, S]^4, S = 1000
- 像素坐标 c_pix 沿dimension D映射：c̃ = (c_pix / D) · S
  - 变量含义：c_pix是pixel space坐标，D是图像在该维度的size，S是normalization scale (1000)，c̃是normalized coordinate

### 3.2 训练策略

冻结vision encoder，更新multimodal projector + FFN + decoder attention projections。理由：vision encoder已经产生strong visual representations，问题在于decoder如何project和utilize。把noisy unsupervised reward gradient传到encoder会destabilize已有的高质量representations。

### 3.3 Geometric Invariance Reward (R_geo)

**核心思想**：如果decoder真正condition在visual content上，那么对image做已知geometric transformation后，预测的box应该等于原box经过同样transformation的analytic projection。任何偏差就是visual under-conditioning的证据。

**Transformation采样**（均匀从三类中采样）：

1. **Affine**:
   - rotation θ ~ U(-10°, 10°)
   - scale s ~ U(0.9, 1.1)
   - translation (δ_x, δ_y) ~ U(-50, 50)^2

2. **Crop**:
   - ratio ρ ~ U(0.8, 1.0)
   - resize回original resolution

3. **Horizontal flip**

每个transformation由3×3 homogeneous matrix M描述。

**计算流程**：
1. 模型在原图x上预测 B_orig
2. 应用transformation：x' = τ(x)
3. 模型在x'上预测 B_new
4. 计算projected box B_proj：把B_orig的4个corner lift到homogeneous coordinates，应用M:
   c_i' = M · c_i
   - c_i：第i个corner的homogeneous coordinate [x_i, y_i, 1]^T
   - M：3×3 transformation matrix
   - c_i'：transformed corner
5. B_proj = axis-aligned box enclosing all c_i'

**Reward公式**（Eq. 1）：

$$\mathcal{R}_{\mathrm{geo}} = \frac{\mathrm{GIoU}(B_{\mathrm{proj}}, B_{\mathrm{new}}) + 1}{2}$$

- B_proj：原prediction的analytic projection
- B_new：模型在transformed view上的prediction
- GIoU：Generalized IoU
- +1再除2：把GIoU∈[-1,1]映射到[0,1]

**GIoU定义**（Eq. 2）：

$$\mathrm{GIoU}(B_1, B_2) = \mathrm{IoU}(B_1, B_2) - \frac{|\mathcal{C}| - |B_1 \cup B_2|}{|\mathcal{C}|}$$

- B_1, B_2：两个box
- C：enclosing B_1和B_2的最小axis-aligned box
- |·|：面积
- 第一项是标准IoU
- 第二项是惩罚项，当两个box不相交但靠近时为正

**为什么用GIoU而不是IoU**：IoU在两个box不相交时为0，无法提供gradient信号告诉模型box应该往哪个方向移动。GIoU通过enclosing box C，即使不相交也能给出距离信号。

### 3.4 Semantic Invariance Reward (R_sem)

**核心思想**：Geometric consistency是必要条件但非充分。模型可以预测large, spatially stable regions而不真正care内容。需要补充evidence sensitivity：模型conditioned在image上的话，应该recognize：移除predicted region就移除了evidence。

**Ghosting操作**：
1. 给定predicted box B_orig
2. 找到对应的pixel region in original image x
3. 用Gaussian blur替换content，kernel σ = 25.0
4. 得到ghosted image x̃：localized region视觉degraded，周围context完全保留

**Visibility判断**：
- v = vis(x, q) ∈ {0, 1}：模型greedy decoding判断object在原图中是否visible
- ṽ = vis(x̃, q) ∈ {0, 1}：模型在ghosted image上的判断

**Reward公式**（Eq. 3）：

$$\mathcal{R}_{\mathrm{sem}} = \begin{cases} 1.0 & \text{if } v = 1 \text{ and } \tilde{v} = 0 \\ 0.0 & \text{otherwise} \end{cases}$$

- v = 1：模型在original上确实看到object
- ṽ = 0：模型在ghosted上判断object消失
- 这种情况reward为1.0

**关键设计**：
- 如果v = 0（模型在原图都没看到object），说明self-generated query/box本身有问题，这种样本不给正reward
- 如果ṽ = 1（ghosting之后模型还说visible），说明模型不依赖visual evidence走shortcut，penalize

**与R_geo的互补性**：一个prediction如果geometrically consistent但semantically arbitrary（包围的region不含queried object），R_sem = 0，即使R_geo高。两个signal互补，jointly necessary。

### 3.5 Composite Reward与Optimization

**Total reward**：

$$\mathcal{R}_t = \lambda_{\mathrm{geo}} \mathcal{R}_{\mathrm{geo}} + \lambda_{\mathrm{sem}} \mathcal{R}_{\mathrm{sem}}$$

- λ_geo = λ_sem = 0.5

**Baseline**：exponential moving average减少gradient variance

$$b_t \leftarrow 0.9 b_{t-1} + 0.1 \mathcal{R}_t$$

**Advantage**：

$$A_t = \mathcal{R}_t - b_t$$

**KL-like divergence**：

$$\Delta_t = \log p_\theta(y | x, q) - \log p_{\mathrm{ref}}(y | x, q)$$

- p_θ：当前policy
- p_ref：frozen reference policy
- 这个是log-ratio，近似KL divergence

**Loss**（Eq. 4）：

$$\mathcal{L}(\theta) = -A_t \cdot \log p_\theta(y | x, q) + \beta_t \cdot \Delta_t$$

- 第一项：REINFORCE-style policy gradient update
  - A_t：advantage
  - log p_θ(y|x,q)：completion的对数概率
- 第二项：KL regularization，防止policy drift太远
- β_t：adaptive KL coefficient

**Adaptive β**（Eq. 5）：

$$\beta_{t+1} = \begin{cases} \beta_t (1 + \eta) & \text{if } |\Delta_t| > \tau \\ \beta_t / (1 + \eta) & \text{otherwise} \end{cases}$$

- τ = 0.020：target divergence budget
- η = 0.10：adaptation rate
- β_t clipped below at 10^-6

**Intuition**：当policy drift超过budget τ时，tighten regularization（β增大）；当updates保守时，relax regularization（β减小）。这样无需固定regularization strength，自动稳定。

## 4. 架构图解析（Figure 2）

```
Raw unlabeled image x
        │
        ▼
[Model generates query q]
        │
        ├──────────────────────────┐
        ▼                          ▼
[Geometric Branch]         [Semantic Branch]
        │                          │
   Apply τ (affine/crop/flip)  Predict B_orig on x
        │                          │
        ▼                          ▼
   x' = τ(x)                  Ghost B_orig region
        │                    (Gaussian blur σ=25)
        ▼                          │
   Predict B_new on x'              ▼
        │                    Ghosted image x̃
        │                          │
   Compute B_proj via M             ▼
        │                    Predict vis on x and x̃
        ▼                          │
   R_geo = (GIoU(B_proj, B_new)+1)/2 ▼
        │                    R_sem = 1 if v=1, ṽ=0
        │                    R_sem = 0 otherwise
        └──────────┬───────────────┘
                   ▼
              R_t = 0.5·R_geo + 0.5·R_sem
                   ▼
              REINFORCE + adaptive KL
                   ▼
              Update θ (LoRA)
```

## 5. Mechanistic Evidence：为什么VISE有效

**Figure 3：Generation-time visual attention**

测量生成每个token时，每个decoder layer分配给image tokens的attention fraction。VISE vs Base model对比，在Qwen3-VL-2B和4B上：

- Mean gain: +2.84% (2B), +2.56% (4B)
- Per-sample peak: up to +5.09% in layers 15-25
- 中后layer改善最显著——这些layer负责semantic generation

**Figure 4：CKA similarity**

Per-layer Centered Kernel Alignment (CKA) similarity between original和geometrically augmented views的representations，跨100张COCO images：

- Qwen3-VL-2B：gains只在final decoder layers，peak Δ = +0.069 at layer 27，100% win-rate
- Qwen3-VL-4B：gains分布在layers 19-33，peak Δ = +0.253，win-rate从~60% (layer 15)增到100% (beyond layer 25)

**Interpretation**：
- 在2B上，geometric under-conditioning集中在generation决策形成的最后阶段，所以invariance correction直接带来downstream gains
- 在4B上，failure分布更广，CKA advantage分布在更多层

## 6. 实验数据详解

### 6.1 Captioning（Table 1）

**Qwen3-VL-2B-Instruct**:

| Method | COCO (C) | NoCaps (C) | Flickr30k (C) | TextCaps (C) |
|--------|----------|------------|---------------|--------------|
| Base | 21.54 | 19.52 | 26.09 | 22.20 |
| VisPlay | 23.85 (+2.31) | 19.14 (-0.38) | 27.50 (+1.41) | 22.11 (-0.09) |
| EvoLMM | 20.84 (-0.70) | 18.75 (-0.77) | 25.15 (-0.94) | 23.04 (+0.84) |
| iReasoner | 20.93 (-0.61) | 18.81 (-0.71) | 25.23 (-0.86) | 23.14 (+0.94) |
| VisionZero-RW | 25.58 (+4.04) | 22.61 (+3.09) | 29.94 (+3.85) | 25.28 (+3.08) |
| **VISE** | **38.39 (+16.85)** | **34.25 (+14.73)** | **42.64 (+16.55)** | **41.86 (+19.66)** |

**关键观察**：
- VISE gains是4×-7×大于最强baseline
- 没有任何regression across datasets/scales
- Gains随model size衰减：+16.85 (2B) → +12.30 (4B) → +9.48 (8B) → +8.72 (32B)
- 这符合"larger models进入post-training时visual conditioning已经更强"的假设

### 6.2 VQA/Reasoning（Table 2）

**Qwen3-VL-2B-Instruct**:

| Method | GQA | OK-VQA | VQAv2 | AI2D | ChartQA | InfoVQA | ScienceQA | MMMU | CaptionQA | RWQA | ESB | MMBench |
|--------|-----|--------|-------|------|---------|---------|-----------|------|-----------|------|-----|---------|
| Base | 58.25 | 40.76 | 78.37 | 73.67 | 79.16 | 69.02 | 79.42 | 38.92 | 77.04 | 63.41 | 68.54 | 74.48 |
| EvoLMM | 59.01 (+0.76) | 38.03 (-2.73) | 77.86 (-0.51) | 75.78 (+2.11) | 79.80 (+0.64) | 70.69 (+1.67) | 83.01 (+3.59) | 39.08 (+0.16) | 76.73 (-0.31) | 63.78 (+0.37) | 69.32 (+0.78) | 74.62 (+0.14) |
| iReasoner | 59.13 (+0.88) | 38.13 (-2.63) | 77.94 (-0.43) | 75.97 (+2.30) | 79.96 (+0.80) | 70.82 (+1.80) | 83.12 (+3.70) | 39.11 (+0.19) | 76.82 (-0.22) | 63.94 (+0.53) | 69.67 (+1.13) | 74.75 (+0.27) |
| **VISE** | **59.41 (+1.16)** | **41.24 (+0.48)** | **78.54 (+0.17)** | **76.42 (+2.75)** | **80.08 (+0.92)** | **71.43 (+2.41)** | **83.61 (+4.19)** | **40.67 (+1.75)** | **79.16 (+2.12)** | **64.58 (+1.17)** | **70.14 (+1.60)** | **76.72 (+2.24)** |

**关键观察**：
- VISE在2B上12个benchmark全部提升，没有regression
- 而EvoLMM/iReasoner呈现tradeoff：ScienceQA提升+3.59/+3.70但OK-VQA drop -2.73/-2.63
- 4B上MMMU gain达到+3.72，是所有方法和scales中最大的
- 没有structured-vs-open-ended tradeoff

### 6.3 Hallucination（Table 3）

**Qwen3-VL-2B-Instruct**:

| Method | POPE Acc | POPE F1 | Chair-I ↓ | Chair-S ↓ | Cap Recall |
|--------|----------|---------|-----------|-----------|------------|
| Base | 89.01 | 88.37 | 13.21 | 45.96 | 72.09 |
| VisPlay | 89.32 (+0.31) | 88.72 (+0.35) | 13.22 (+0.003) | 46.19 (+0.23) | 71.32 (-0.77) |
| EvoLMM | 87.59 (-1.42) | 88.51 (+0.14) | 12.99 (-0.23) | 44.22 (-1.74) | 70.99 (-1.10) |
| iReasoner | 87.70 (-1.31) | 88.56 (+0.19) | 12.98 (-0.23) | 44.22 (-1.74) | 70.99 (-1.10) |
| VisionZero-RW | 88.70 (-0.31) | 87.84 (-0.53) | 10.22 (-2.99) | 41.91 (-4.05) | 72.10 (+0.005) |
| **VISE** | **90.03 (+1.02)** | **89.22 (+0.85)** | **8.21 (-5.00)** | **40.51 (-5.45)** | **72.31 (+0.22)** |

**关键观察**：
- VISE同时改善POPE (+1.02)和Chair (-5.00, -5.45)
- EvoLMM/iReasoner减少Chair但drop POPE accuracy，inconsistent improvement
- VISE是唯一同时改善两个指标的方法

### 6.4 Backbone Generalization（Table 4）

四种architecturally diverse backbones，同样4000张unlabeled COCO images：

| Backbone | COCO Δ | NoCaps Δ | POPE F1 Δ | Chair-I Δ | ScienceQA Δ |
|----------|--------|----------|-----------|-----------|--------------|
| Qwen3-VL-8B | +9.48 | +10.52 | +0.71 | -0.36 | +1.93 |
| InternVL3-8B | +9.01 | +10.16 | +0.81 | -0.29 | +0.58 |
| Gemma3-12B | +7.65 | +8.67 | +1.04 | -0.29 | +0.48 |
| Llama-3.2-11B | +6.44 | +6.25 | +0.66 | - | - |

Invariance reward是architecture-agnostic的，visual under-conditioning是跨backbone的general phenomenon。

### 6.5 Ablation Study（Table 5）

**Qwen3-VL-2B**:

| Method | COCO Δ | Chair-I Δ | POPE Δ | ScienceQA Δ |
|--------|--------|-----------|--------|--------------|
| R_geo only | +4.83 | -1.35 | +0.28 | +0.76 |
| R_sem only | +13.99 | -4.15 | +0.85 | +2.52 |
| Full | +16.85 | -5.00 | +1.02 | +4.19 |

**关键发现**：
- R_sem贡献大部分gains（captioning +13.99 vs +4.83）
- R_geo提供complementary improvements（Full +2.86 over R_sem alone on COCO）
- 在2B上R_geo占整个captioning gain的~28%
- 两个reward覆盖distinct facets of visual under-conditioning

## 7. 训练Implementation细节

**LoRA配置**：
- 2B/4B: rank r = 16, α = 32
- 8B/32B: rank r = 32, α = 64
- Dropout = 0.05

**Optimizer**：
- AdamW
- Weight decay = 0.01
- Gradient clipping at 1.0
- LR = 10^-6 (smaller models), 1.5×10^-7 (larger models)
- KL target τ = 0.020 (adaptive rate 0.10 smaller, 0.15 larger)

**Training**：
- 4000 steps
- 8× AMD MI250X GPUs
- bfloat16 precision
- 4000张unlabeled COCO images

**Per-step cost**：7 forward passes
1. Query generation
2. Box prediction on original
3. Box prediction on transformed
4. Visibility prediction on original
5. Visibility prediction on ghosted
6. Policy log-prob evaluation
7. Reference log-prob evaluation

Training Qwen3-VL-2B for 4000 steps需16小时，比EvoLMM等multi-role baselines快约2×。

## 8. Validation实验（Supplementary）

### 8.1 Training Domain Robustness（Table S2）

Qwen3-VL-8B上，VISE在COCO训练和Objects365训练都得到consistent gains：
- COCO training: COCO CIDEr 38.49 ± 0.67
- Objects365 training: COCO CIDEr 38.57 ± 0.36

证明gains不来自COCO-specific image exposure。

### 8.2 LoRA vs Full Fine-Tuning（Table S2）

- FFT COCO Training: 32.80 ± 0.45
- LoRA COCO Training: 38.49 ± 0.67

LoRA全面outperform FFT，支持frozen-encoder设计：noisy unsupervised encoder updates会hurt visual conditioning。

### 8.3 Reward Causality（Table S3）

Random reward control（R ~ U(0,1)）：
- 2B: COCO CIDEr 21.38（base 21.54）
- 8B: COCO CIDEr 29.12（base 29.01）

Random reward几乎和base相同，证明gains来自invariance reward本身而非generic fine-tuning。

### 8.4 Transformation/Perturbation Design（Table S4）

**Geometric transformations on 2B (COCO CIDEr)**：
- Affine only: 36.84 (+15.30)
- Crop only: 36.14 (+14.60)
- Flip only: 33.84 (+12.30)
- Small affine (±5°): 35.62 (+14.08)
- Large affine (±20°): 31.84 (+10.30)

**Semantic perturbations on 2B**：
- Default ghosting σ=25: 38.39 / POPE 90.03
- σ=50: 37.12 / 89.78
- σ=80: 35.24 / 89.34
- σ=15: 34.82 / 89.52
- Zero masking: 33.41 / 89.14
- Gaussian noise: 30.41 / 88.94

σ=25最优：足够degraded但保留spatial structure clue。

### 8.5 Hyperparameter Sensitivity（Table S1）

在Qwen3-VL-2B和8B上变化λ ratio和τ：

- λ_geo=0.75/λ_sem=0.25: COCO 38.18 (2B)
- λ_geo=0.50/λ_sem=0.50 (default): COCO 38.39
- λ_geo=0.25/λ_sem=0.75: COCO 38.33

Variance < 0.5 CIDEr，证明VISE对precise hyperparameter choices不sensitive。

## 9. 核心Intuition

**VISE的本质**：把训练目标从"output agreement"换成"evidence-conditioned generation"。

**为什么这样work**：

1. **自监督信号的来源**：单模型已经能formulate meaningful queries和predict boxes，所以self-generated queries就是有效的训练signal source，无需external reward models或multi-role framework。

2. **Geometric invariance如何detect under-conditioning**：一个走language prior的decoder，在image做transformation后，它生成的answer不会跟着transform，因为它根本不look at image。GIoU = 0立即penalize这种情况。

3. **Semantic invariance如何detect under-conditioning**：ghosting操作等于把evidence拿走。如果decoder真正conditioned在visual content上，它应该recognize evidence消失。如果decoder走language prior，它会继续说"visible"，因为它的输出由co-occurrence statistics驱动而非pixel evidence。

4. **为什么R_sem贡献大于R_geo**：captioning和hallucination的核心是evidence binding。R_geo只catch空间不一致，但无法penalize"geometrically consistent但semantically empty"的prediction。R_sem直接攻击这个gap。

5. **Attention层面的解释**：Figure 3显示VISE在mid-to-late decoder layers增加对image tokens的attention。这些layer负责semantic generation decisions，正是language prior shortcut发生的地方。Invariance reward把generation behavior从"prior-driven"shift到"image-conditioned"。

## 10. 与Related Work对比

| 方法 | 类型 | 监督 | 关键问题 |
|------|------|------|----------|
| EvoLMM [28] | Proposer-Solver | Unsupervised | minimax instability, answer agreement reward |
| iReasoner [27] | Trajectory-aware | Unsupervised | 同上 + trajectory reward仍optimizes agreement |
| VisPlay [6] | Multi-role + diversity/difficulty | Unsupervised | diversity reward优先question complexity over visual fidelity |
| VisionZero [29] | Multi-agent self-play | CLEVR label-free, Chart/RW用GPT-4o | domain-specific adaptation, generalization tradeoff |
| C2-Evo [3] | Co-evolutionary data loops | - | training instability |
| DoGe [11] | Role decoupling | - | training stability |
| **VISE** | **Single model** | **Fully unsupervised** | **直接regularize visual conditioning policy** |

VISE是唯一同时满足：single model / no specialist roles / no external reward / no annotation / no tradeoff的方法。

## 11. 局限性与未来方向

**Scale effect**：Gains随model size衰减。Paper归因于"larger models进入post-training时visual conditioning已经consolidated during pretraining"。这暗示VISE在smaller models（2B/4B）上价值最大。

**Domain specificity**：训练只用4000张COCO images，但在18个benchmark上都generalize。Table S2证明用Objects365训练效果类似，暗示reward signal本身是domain-agnostic的。

**Potential extensions**：
- 多modality invariance（音频、视频）
- Video temporal invariance
- 与test-time compute结合

## 12. 关键参考链接

- Project page: https://mbzuai-oryx.github.io/VISE/
- GitHub: https://github.com/mbzuai-oryx/VISE
- HuggingFace: https://huggingface.co/shravvvv/VISE
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- GIoU paper: https://arxiv.org/abs/1902.09630
- EvoLMM: https://arxiv.org/abs/2511.16672
- VisPlay: https://arxiv.org/abs/2507.04079 (CVPR 2026)
- VisionZero: https://openreview.net/forum?id=s00SNXREV6 (ICLR 2026)
- iReasoner: https://arxiv.org/abs/2601.05877
- C2-Evo: https://arxiv.org/abs/2507.16518
- LoRA: https://openreview.net/forum?id=nZeVKeeFYf9
- CKA similarity: https://arxiv.org/abs/1905.00414
- lmms-eval: https://arxiv.org/abs/2407.12772
- Agent0-VL: https://arxiv.org/abs/2511.19900

## 13. 总结

VISE的关键贡献是把self-evolving LMM的训练目标从"answer agreement"shift到"visual conditioning policy regularization"。通过两个互补的invariance reward（geometric + semantic），VISE在18个benchmark上取得consistent gains，无tradeoff，跨4种backbone都有效。Mechanistic evidence显示gains来自mid-to-late decoder layers对visual tokens的attention增加，直接证实了"decoder从language-prior-driven shift到image-conditioned decoding"的假设。

这个工作给self-evolving multimodal training指明了新方向：与其optimize output consistency，不如directly increase attention to visual tokens during decoding。这是both necessary and sufficient for broad, robust gains。
