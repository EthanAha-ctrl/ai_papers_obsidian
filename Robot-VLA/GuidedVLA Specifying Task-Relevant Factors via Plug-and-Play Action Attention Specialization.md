---
source_pdf: GuidedVLA Specifying Task-Relevant Factors via Plug-and-Play Action Attention
  Specialization.pdf
paper_sha256: 6825b4b5218757409aa1ecdd2de2301c3f13b13f4a35d2ac68ce6f49bba2c167
processed_at: '2026-08-04T23:16:19-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 GuidedVLA

## 一句话版本

π0 的 action decoder 训练完，attention 经常乱飘 —— 盯着 background texture、camera artifact 看半天，不盯 task-relevant 的 object。GuidedVLA 的做法很简单：**把 multi-head attention 的某些 heads "强制分配" 给特定任务因子，告诉它们"你这几个 head 负责 locate object，那几个负责识别 skill phase，剩下几个负责 depth 几何"**，然后施加 auxiliary supervision。结果 generalization 涨了 7-20 个百分点。

---

## Motivation：问题出在哪

先把问题讲清楚。VLA 的 pipeline 大概是：

```
Image + Language → VLM (PaliGemma) → rich features → Action Decoder (Gemma-300M) → action chunk
```

VLM backbone 是 pretrained 的，features 质量很好。但 action decoder 是用 flow matching loss $\mathcal{L}_{FM}$ end-to-end 训练的，loss 只监督 final action output。中间的 cross-attention 怎么 attend，完全靠 gradient 隐式决定。

作者做了 probing，发现 π0 的 action token attention：
- 落在 task-relevant object region 的 mass 只有 **26.5%**（random baseline 大概 50%，所以比 random 还差）
- argmax 落对 object 的 accuracy **2.2%**
- Skill classification（linear probe）**48.4%**（4 类，random 25%，基本没学到 temporal structure）

直觉上这意味着：action decoder **并没有真正利用** VLM 提供的 vision-language features。它可能学会了"看到绿色 pixel 就 grab 一下"这种 shortcut，而不是真正理解"这是个 bowl，我要 grab 它的 rim"。

这就是 classic 的 **shortcut learning** 问题（Geirhos et al. [2020], https://www.nature.com/articles/s42256-020-00257-z）—— 模型找了个 easy surface feature 混过去，generalization 就崩了。

---

## Core Idea：把 attention heads 拆成 functional modules

这里有个很深的 insight。Multi-Head Attention 设计的初衷（Vaswani et al. [2017], https://arxiv.org/abs/1706.03762）就是让不同 heads 学不同的 representation subspaces。但实际训练中，谁学什么完全 random —— 有些 head 可能 attend 到 object，有些 attend 到 background，每次跑都不一样。

GuidedVLA 的做法：**把这种 random specialization 变成 explicit assignment**。

具体来说，选三组 heads：
- $\mathcal{H}_{obj}$: object head，负责 visual grounding
- $\mathcal{H}_{skill}$: skill head，负责 temporal phase recognition  
- $\mathcal{H}_{depth}$: depth head，负责 3D geometry

每组 head 施加不同的 supervision signal。

剩下的 heads 不管，让它们自由学 data-driven 的 pattern。

---

## 关键工程细节：ControlNet-style Adapter

这里有个很重要的设计问题：π0 已经 pretrained 了，怎么加这些 supervised heads 才不破坏 pretrained 能力？

直接 fine-tune 整个 model？会 catastrophic forgetting。加新 branch 然后 concat？初始时 random branch 会破坏 pretrained behavior。

作者借鉴了 ControlNet（Zhang et al. [2023], https://arxiv.org/abs/2302.05543）的设计：

$$\text{Attn}(x) = \text{Attn}_{main}(x) + \text{ZeroConv}(\text{Attn}_{specified}(x))$$

- $\text{Attn}_{main}(x)$: 原始 π0 的 attention（权重 freeze 或 slow update）
- $\text{Attn}_{specified}(x)$: 新加的 factor-specific attention branch
- $\text{ZeroConv}$: zero-initialized linear projection

**关键点**：训练 step 0 时，ZeroConv 权重全是 0，所以 $\text{Attn}(x) = \text{Attn}_{main}(x)$，model 行为跟原 π0 完全一样。然后随着 training，gradient 慢慢把 ZeroConv 的 weight 学出来，factor-specific bias 逐渐 inject 进去。

这跟 LoRA 的 zero-init B 矩阵思路类似，但更接近 ControlNet 在 diffusion model 里做 conditional control 的 philosophy —— 你有个 base model，想要 inject condition，用 zero-init 保证训练初期 identity behavior。

---

## 三个 Factor 的具体实现

### 1. Object Head：让 attention 盯着 object

最直觉的一个。给一组 heads $\mathcal{H}_{obj}$，强制它们的 attention mass 落在 task-relevant object region 上。

先对 object heads 做 mean aggregation：

$$\bar{P}_{b,t,k} = \frac{1}{|\mathcal{H}_{obj}|} \sum_{h \in \mathcal{H}_{obj}} P_{b,h,t,k}$$

- $b$: batch index
- $t$: action query index
- $k$: key position（image patch 或 token）
- $h$: head index
- $\bar{P}$: averaged attention probability

Object mass（落在 object mask 内的 attention 总量）：

$$m_{b,t} = \sum_k \bar{P}_{b,t,k} M_{b,k}$$

- $M_{b,k} \in [0,1]$: object region mask（non-object patch 是 0）
- $m_{b,t} \in [0,1]$: action query $t$ 在 object 上的累计 attention

Loss：

$$\mathcal{L}_{object} = -\frac{1}{\sum_b v_b |\mathcal{T}_a|} \sum_b v_b \sum_{t \in \mathcal{T}_a} \log(\max(m_{b,t}, \epsilon))$$

- $v_b$: sample $b$ 是否有可见 labeled object
- $\mathcal{T}_a$: action query 集合
- $\epsilon$: 防 log(0)

**这个 loss 很微妙**：它只惩罚"object region 内 attention mass 不够"，不约束 object region 内 attention 怎么分布。也就是说，模型自己决定 attend object 的 handle 还是 rim 还是 edge，但必须把 mass 放在 object 区域内。

附录 ablation 显示：binary mask supervision (83.33%) > Gaussian soft prior + KL (72.00%)。说明 explicit spatial constraint 比 soft prior 有效。

### 2. Skill Head：识别 temporal phase

Long-horizon task 比如"pick → place → pour → return"，模型容易在 phase transition 时 mode collapse。

Pool skill heads 的 output features：

$$\bar{\mathbf{f}}_b = \frac{1}{|\mathcal{L}_g| |\mathcal{H}_{skill}| |\mathcal{T}_a|} \sum_{\ell \in \mathcal{L}_g} \sum_{h \in \mathcal{H}_{skill}} \sum_{t \in \mathcal{T}_a} \mathbf{f}_{b,\ell,h,t}$$

- $\mathcal{L}_g$: guided layers 集合
- $\bar{\mathbf{f}}_b$: pooled feature

过个 classifier + KL loss：

$$\hat{\mathbf{p}}_b = \text{softmax}(W\bar{\mathbf{f}}_b + \mathbf{b})$$

$$\mathcal{L}_{skill} = \frac{1}{B} \sum_b \sum_k y_{b,k} (\log y_{b,k} - \log \hat{p}_{b,k})$$

- $\hat{\mathbf{p}}_b$: predicted skill distribution
- $\mathbf{y}$: ground-truth soft skill label
- $k$: skill class

Soft label 构造：

$$y_k = \frac{\sum_{t=1}^T \mathbb{I}[s_t = k]}{\sum_j \sum_t \mathbb{I}[s_t = j]}$$

- $s_t$: timestep $t$ 的 skill id
- $T$: action chunk 长度

这个 soft target 处理 transition frames 很关键 —— 一个 action chunk 跨 pick→place 过渡时，soft label 表达 mixed intent。Ablation: soft label (75.00%) > hard one-hot (69.33%)。

### 3. Depth Head：注入 3D 信息

SigLIP 是 2D supervision 训的，没 3D awareness。GuidedVLA 用 frozen Depth Anything 3 (https://arxiv.org/abs/2511.10647) 提 depth features，project 成 K/V，然后 constrain specific heads 只 attend 这些 depth KV：

$$\mathcal{H}_{depth}: \text{softmax}\left(\frac{Q_{act}[\mathcal{H}_{depth}] K_{Depth}^\top}{\sqrt{d_h}}\right) V_{Depth}$$

- $Q_{act}[\mathcal{H}_{depth}]$: action query 在 depth head 上的 projections
- $K_{Depth}, V_{Depth}$: depth features 经过 projector
- $d_h$: head dimension

这个 head 没 loss，纯 architectural constraint。Query 还是从 action decoder 来（model 决定何时用 depth），但 KV 来自 frozen depth encoder（保证 3D 信息准确）。

---

## Annotation Pipeline：工程关键

要让这 method scale，annotation 成本必须低。作者搞了个 pipeline：

1. **Qwen3-VL** (https://arxiv.org/abs/2511.21631) 识别 object，输出 point prompts
2. **SAM2** (https://arxiv.org/abs/2408.00714) propagate mask 到整个 video segment
3. **Human verification**（可选）

结果：**92% episodes 不需要人工修改，50 episodes 标注 4 分钟**（全 manual 要 43.5 分钟）。

Mask 转 patch grid（PaliGemma 用 16×16 image tokens）：

$$m_p = \mathbb{I}[s_p \geq \tau], \quad p \in \mathcal{P}$$

- $s_p$: patch $p$ 的 foreground coverage ratio
- $\tau$: threshold

---

## 实验结果速览

### LIBERO-Plus（Table I）

LIBERO-Plus (https://arxiv.org/abs/2510.13626) 在 7 个 perturbation dimension 上测 robustness：

| Model | Total |
|---|---|
| OpenVLA | 15.6 |
| π0 | 68.2 |
| DreamVLA | 69.9 |
| **GuidedVLA** | **75.4** |

Head-specific 贡献（single-head ablation）：
- Object head 在 **Object suite** 最强（82.5%, +8.4%）
- Skill head 在 **Goal suite** 最强（68.9%, +7.5%）
- Depth head 在 **Spatial suite** 最强（81.4%, +3.7%）

这验证了 factor-task alignment：不同 task 类型受益于不同 factor。

### RoboTwin 2.0（Fig. 5）

8 个 manipulation task，π0 的 77.38% → **90.63%**。

最 dramatic：**Click Bell**（需精确 Z 轴控制）35% → 65%（depth head 贡献）。

### Real-World（Table II）

| Setting | π0 | GuidedVLA |
|---|---|---|
| In-domain | 55.8 | **75.8** |
| Scene | 44.2 | **67.5** |
| Lighting | 57.5 | **79.2** |

Lighting generalization 涨 **+21.7%** 最显著 —— 因为 explicit guidance 减少了 spurious appearance correlation。

---

## 最 Convincing 的分析：Factor Quality ↔ Performance

作者不只是说"with > without"，而是 controlled ablation 看 factor quality 跟 success 的关系（Fig. 7）。

### Object grounding quality

人为控制 attention mass $m$ 到目标值 $\alpha \in \{0.25, 0.5, 0.75, 1.0\}$：

$$\mathcal{L}_{ablation} = \begin{cases} \frac{0.5(m-\alpha)^2}{\beta}, & |m-\alpha| < \beta \\ |m-\alpha| - 0.5\beta, & \text{otherwise} \end{cases}$$

- $\beta = 0.05$: Huber loss smoothing

结果：attention mass 0.25 → 61.3%，1.0 → 74.6%，**单调正相关**。

### Skill recognition quality

控制 linear probe accuracy 到 $\gamma \in \{0.25, 0.5, 0.75, 1.0\}$，用 Smooth L1：

$$\mathcal{L}_{ctrl} = \begin{cases} \frac{0.5(S-\gamma)^2}{\beta}, & |S-\gamma| < \beta \\ |S-\gamma| - 0.5\beta, & \text{otherwise} \end{cases}$$

- $S = \frac{1}{N}\sum_i \hat{p}_i(y_i)$: soft accuracy
- $\gamma$: target

结果：25% → 66.2%，100% → 72.9%，正相关。

### Depth feature quality

通过 noise injection 控制 depth signal ratio：

$$\tilde{\mathbf{f}} = \delta \cdot \mathbf{f}_{depth} + (1-\delta) \cdot \boldsymbol{\epsilon}$$

- $\delta \in [0,1]$: clean depth ratio
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: matching noise

结果：0（纯噪声）→ 15.6%，1.0（clean）→ 76.7%，**massive monotonic gain**。

这三个实验结合非常强：**factor 质量越高，task 表现越好**，说明这是 causal relationship，不只是 correlation。

---

## Specialization vs Mixture：为什么不所有 heads 一起监督所有 factors？

作者对比两种 paradigm：
- **Specialization**: 不同 heads 监督不同 factors（Ours）
- **Mixture**: 所有 heads 监督所有 factors

Mixture 明显 underperform。t-SNE（Fig. 10）显示：
- Specialization: 三个 factor 的 features 形成 **well-separated clusters**
- Mixture: clusters overlap，features entangled

直觉：当一个 head 同时被 object loss、skill loss、depth constraint 拉扯时，gradient 冲突导致 features 啥都学不好。这跟 MoE 的 routing 问题、multi-task learning 的 negative transfer 类似。

---

## Layer 选择：Guidance 施加在哪层？

Table X 测了 all layers 和 4 个 quartiles：

| Layer subset | Total |
|---|---|
| All layers | 74.1 |
| 1st quartile (bottom) | 74.4 |
| 2nd quartile | 74.3 |
| **3rd quartile** | **75.4** |
| 4th quartile (top) | 73.8 |

**3rd quartile 最好** —— mid-to-upper layers。直觉：底层是 low-level features（颜色、纹理），顶层接近 output，中间层是 abstract semantic representation，最适合施加 task-level guidance。

---

## Loss 权重：Auxiliary loss 应该是 regularizer

Table XIII ablation：

| $(w_{obj}, w_{skill})$ | Avg |
|---|---|
| **(0.001, 0.001)** | **87.83** |
| (0.01, 0.01) | 85.77 |
| (0.01, 0.001) | 86.12 |
| (0.001, 0.01) | 85.22 |

Auxiliary loss 权重太大反而干扰主 flow matching loss。这暗示 auxiliary supervision 应该是 **regularizer** 角色，主导 task 还是 action generation。

---

## 我的几点 Take-away

1. **Attention head specialization 是个 general paradigm**。这篇只 instantiate 三个 factor，但 framework 本身 extensible。可以加 force/torque head、affordance head、human pose head。作者在 conclusion 也提到这个方向。

2. **ControlNet-style adapter 的 zero-init 思路很优雅**。既保留了 pretrained capability，又允许 gradual injection。这跟 LoRA 的 zero-init B、adapter 的 zero-init down-projection 都是一脉相承的 philosophy —— 避免 initialization 破坏 pretrained behavior。

3. **Factor quality ↔ performance 的 monotonic correlation 很 powerful**。这不只是"加了 supervision 就好"，而是"supervision 质量 causal 地决定 task 表现"。这给 future work 指明方向：与其堆更多 factor，不如提升每个 factor 的 supervision 质量。

4. **跟 mechanistic interpretability 的连接**。这工作实际上是在做 **attention head role assignment**，跟 Anthropic Circuits (https://transformer-circuits.pub/) 的 induction head、in-context learning head 发现思路相通。区别是：Circuits 是 post-hoc 发现 heads 的 role，GuidedVLA 是 ex-ante 指定 heads 的 role。两者结合可能是个 direction —— 用 Circuits 发现的 role pattern 来 inform factor assignment。

5. **Skill label 自动化是 bottleneck**。Continuous task（比如 continuous pouring）的 automatic skill labeling 还没解决。可以参考 LOTUS (https://arxiv.org/abs/2203.00752) 的 unsupervised skill discovery，或者用 VLM 自动 stage segmentation。

6. **Depth encoder 的选择**。Depth Anything 3 是 relative depth。如果换成 VGGT (https://arxiv.org/abs/2503.11651) 或 DUSt3R (https://arxiv.org/abs/2312.14132) 这种 metric 3D reconstruction，可能对 millimeter-precision task（比如 beaker insertion）效果更好。

7. **跟 RLHF 的关系**。GuidedVLA 用 supervised guidance，RLHF (https://arxiv.org/abs/2203.02155) 用 reward signal。两者可以结合 —— 用 reward signal 自动 discover task-relevant factors，然后用 attention specialization 注入。这能解决 GuidedVLA 需要手动定义 factor 的 limitation。

8. **3D representation 的 orthogonality**。附录提到 depth head 加到 Spatial Forcing 上从 35% → 50%，加到 π0 上从 25% → 45%。这说明 depth-specialized pathway 跟更强的 spatial VLA backbone 是 complementary 的。未来可以探索更 sophisticated 的 3D representation（point cloud、3D Gaussian、NeRF）。

9. **Failure analysis 很 informative**。三类 failure（object grounding, metric geometry, temporal skill collapse）跟三个 factor heads 一一对应，这从反面验证了 factor selection 的合理性。这种 failure-driven 的 method design 值得学习。

10. **Generalization 提升最显著的是 lighting**（+21.7%）。这很有意思 —— 说明 explicit factor guidance 最能帮的场景是 appearance shift 下的 generalization，因为它强迫 model 学 task structure 而非 surface texture。这跟 Geirhos 的 texture-bias 研究（https://arxiv.org/abs/1811.12231）形成呼应。

---

## References

- GuidedVLA project: https://guidedvla.github.io/project_page/
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- LIBERO: https://arxiv.org/abs/2311.07423
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- ControlNet: https://arxiv.org/abs/2302.05543
- SAM2: https://arxiv.org/abs/2408.00714
- Depth Anything 3: https://arxiv.org/abs/2511.10647
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- DreamVLA: https://arxiv.org/abs/2506.15785
- AdaMoE: https://arxiv.org/abs/2510.14300
- Spatial Forcing: https://arxiv.org/abs/2510.08881
- VLA-Adapter: https://arxiv.org/abs/2501.06234
- Shortcut learning (Geirhos): https://www.nature.com/articles/s42256-020-00257-z
- Texture bias (Geirhos): https://arxiv.org/abs/1811.12231
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- LoRA: https://arxiv.org/abs/2106.09685
- Anthropic Circuits: https://transformer-circuits.pub/
- LOTUS skill discovery: https://arxiv.org/abs/2203.00752
- VGGT: https://arxiv.org/abs/2503.11651
- DUSt3R: https://arxiv.org/abs/2312.14132
- RLHF: https://arxiv.org/abs/2203.02155
- ACT (ALOHA): https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RT-2: https://arxiv.org/abs/2307.15818

如果某个 factor 实现、ControlNet adapter 的 gradient flow、或者 factor automation 方向你想深挖，告诉我。

---

# GuidedVLA: 通过 Plug-and-Play Action Attention Specialization 指定 Task-Relevant Factors

这篇来自 Fudan University / HKU OpenDriveLab 的工作很有意思，让我从 intuition、method、experiment 三个层面给你拆解。整体 philosophy 是：**与其让 action decoder 黑盒地隐式学习，不如把 MHA (Multi-Head Attention) 的 heads 拆成 functional modules，每个 module 显式地负责一个 task-relevant factor**。

---

## 1. Motivation：为什么 end-to-end supervision 不够用

作者先做了一个 probing experiment，发现 π0 的 action token attention 经常扩散到无关的 background texture、camera artifact 上。这跟 Geirhos et al. 在 Nature Machine Intelligence [2020] 提出的 **shortcut learning** (https://www.nature.com/articles/s42256-020-00257-z) 是同一个现象 —— CNNs biased toward texture 也是这个问题。

具体来说，VLA 的 action generation 可以写成：
$$a \sim p_\theta(a | v, l)$$

其中 $v$ 是 vision tokens，$l$ 是 language tokens，$a$ 是 action tokens。VLM backbone (PaliGemma) 提供了 rich semantic features，但 action decoder (Gemma-300M expert) 通过 flow matching loss $\mathcal{L}_{FM}$ 训练时，没有任何 explicit signal 告诉它"应该 attend 到哪些 key 上"。结果就是 cross-attention 的 attention pattern 随机分布，stochastic across heads and scenarios。

作者 baseline 的 probing 结果：
- Object attention mass: π0 只有 **26.5%** 落在 task-relevant region
- Argmax-hit accuracy: π0 只有 **2.2%**
- Skill classification (linear probe): π0 只有 **48.4%** (4-class，random 是 25%，所以基本没学到什么 temporal structure)

这就很能说明问题了 —— action decoder 并没有真正 "use" VLM 的 vision-language features。

---

## 2. Core Idea：Attention Head Specialization

这是 paper 最核心的 insight：**把 action decoder 的 attention heads 当成一组 functional modules，而不是 monolithic learner**。

MHA 本来就有天然的 decoupling 特性 —— 不同的 heads 可以学习不同的 subspaces。Vaswani et al. 的 "Attention is All You Need" [2017] (https://arxiv.org/abs/1706.03762) 早就提过 multi-head allows the model to jointly attend to information from different representation subspaces。GuidedVLA 把这个 idea explicit 化了：**手动指定哪些 heads 负责哪个 factor，然后施加不同的 auxiliary supervision**。

### ControlNet-style Residual Adapter

这里他们借鉴了 ControlNet (Zhang et al. [2023], https://arxiv.org/abs/2302.05543) 的设计思路，搞了一个 plug-and-play adapter：

$$\text{Attn}(x) = \text{Attn}_{main}(x) + \text{ZeroConv}(\text{Attn}_{specified}(x))$$

- $\text{Attn}_{main}(x)$: 主分支，保留 pretrained π0 的 attention 权重
- $\text{Attn}_{specified}(x)$: 指定的 factor-specific 分支
- $\text{ZeroConv}$: zero-initialized linear projection

关键 intuition：**因为 ZeroConv 初始化为 0，所以训练开始时 $\text{Attn}(x) = \text{Attn}_{main}(x)$，pretrained behavior 完全保留**。然后随着训练，gradient 慢慢把 ZeroConv 的权重学出来，factor-specific bias 逐渐被 inject 进去。这跟 LoRA (https://arxiv.org/abs/2106.09685) 的 zero-init 有点像，但更接近 ControlNet 在 diffusion model 里的 conditional control 思路。

---

## 3. 三个 Task-Relevant Factors 的具体实现

### 3.1 Object Head (Visual Grounding)

让一组 heads $\mathcal{H}_{obj}$ 显式 attend 到 task-relevant object region。

#### 公式解析

给定 attention probabilities $P$（action queries 到所有 keys 的 softmax），先对 $\mathcal{H}_{obj}$ 做 mean-head aggregation：

$$\bar{P}_{b,t,k} = \frac{1}{|\mathcal{H}_{obj}|} \sum_{h \in \mathcal{H}_{obj}} P_{b,h,t,k}$$

变量含义：
- $b$: batch index
- $t$: action query index（在 $\mathcal{T}_a$ 集合中，即 action queries 的集合）
- $k$: key position（对应 image patch 或其他 token）
- $h$: head index
- $\bar{P}_{b,t,k}$: 对 object heads 平均后，action query $t$ 在 key $k$ 上的 attention probability

Object mass（action query $t$ 在 object region 上的总 attention）：

$$m_{b,t} = \sum_k \bar{P}_{b,t,k} M_{b,k}$$

- $M_{b,k} \in [0,1]$: object-region target mask，non-object patches 和 non-image tokens 都为 0
- $m_{b,t} \in [0,1]$: action query $t$ 在 object region 上的累计 attention mass

Loss 用 negative log object mass：

$$\mathcal{L}_{object} = -\frac{1}{\sum_b v_b |\mathcal{T}_a|} \sum_b v_b \sum_{t \in \mathcal{T}_a} \log\left(\max(m_{b,t}, \epsilon)\right)$$

- $v_b$: binary indicator，sample $b$ 是否有可见的 labeled object patch
- $\epsilon$: small numerical constant 防止 log(0)

**Key insight**：这个 loss 只惩罚"object region 内的 attention mass 不足"，但不约束 object region 内的 distribution。也就是说，模型自己决定 attend 到 object 的哪个部分（handle? rim? edge?），但必须把 attention mass 放在 object 区域内。这给了 model flexibility，但又有 explicit grounding signal。

附录里还有个 ablation 实验 (Table VI)：binary region supervision (83.33%) vs Gaussian prior + KL divergence (72.00%)。Binary mask 比 Gaussian soft prior 强不少，说明 **explicit spatial constraint 比 soft prior 更有效**。

### 3.2 Skill Head (Temporal Logic Intent)

这个 head 处理 long-horizon task 的 temporal structure。

#### Feature pooling

把 selected skill heads $\mathcal{H}_{skill}$ 的 output features 在 guided layers、heads、action queries 上 pool：

$$\bar{\mathbf{f}}_b = \frac{1}{|\mathcal{L}_g| |\mathcal{H}_{skill}| |\mathcal{T}_a|} \sum_{\ell \in \mathcal{L}_g} \sum_{h \in \mathcal{H}_{skill}} \sum_{t \in \mathcal{T}_a} \mathbf{f}_{b,\ell,h,t}$$

- $\mathcal{L}_g$: guided transformer layers 的集合
- $\bar{\mathbf{f}}_b \in \mathbb{R}^d$: pooled feature for batch $b$

#### Classification head + KL loss

$$\hat{\mathbf{p}}_b = \text{softmax}(W\bar{\mathbf{f}}_b + \mathbf{b})$$

$$\mathcal{L}_{skill} = \frac{1}{B} \sum_{b=1}^B \sum_k y_{b,k} (\log y_{b,k} - \log \hat{p}_{b,k})$$

- $W, \mathbf{b}$: classification head 参数
- $\hat{\mathbf{p}}_b$: predicted skill distribution
- $\mathbf{y}$: ground-truth soft skill label（来自 trajectory-level skill distribution）
- $B$: batch size
- $k$: skill class index

这个 loss 实际上是 KL divergence: $D_{KL}(y \| \hat{p})$ 的 batch mean 版本。

**Soft label 的设计**（Eq. 21）：
$$y_k = \frac{\sum_{t=1}^T \mathbb{I}[s_t = k]}{\sum_{j=0}^{K-1} \sum_{t=1}^T \mathbb{I}[s_t = j]}$$

- $s_t$: timestep $t$ 的 skill id
- $T$: action chunk 长度
- $K$: skill class 数量（LIBERO 里 K=4，3 个 task-level skill + 1 个 null/background）

这个 soft target 的好处是处理 transition frames 和 ambiguous segments —— 比如一个 action chunk 跨越 pick → place 的过渡阶段，soft label 就能表达这种 mixed intent。附录 Table VIII 显示 soft label (75.00%) 比 hard one-hot label (69.33%) 好，尤其在 ambiguity 大的 task 上。

### 3.3 Depth Head (3D Structure)

这个 head 不用 loss，用 **architectural constraint**。

SigLIP 这种 vision encoder 是 2D supervision 训练的，缺 3D awareness。作者用 frozen Depth Anything 3 (DA3, Lin et al. [2025], https://arxiv.org/abs/2511.10647) encoder 提取 depth features $F_{Depth}$，project 成 depth-aware keys 和 values，然后 constrain specific heads 只 attend 到这些 depth-derived KV 上：

$$\mathcal{H}_{depth}: \text{softmax}\left(\frac{Q_{act}[\mathcal{H}_{depth}] (K_{Depth})^\top}{\sqrt{d_h}}\right) V_{Depth}$$

- $Q_{act}[\mathcal{H}_{depth}]$: action query 在 depth head 上的 query projections
- $K_{Depth}, V_{Depth}$: depth features 经过 projector 后的 keys/values
- $d_h$: head dimension
- $\sqrt{d_h}$: standard scaling factor (来自 Vaswani et al.)

这个设计很巧妙：query 还是来自 action decoder（所以 model 仍然可以决定"何时、如何"用 depth 信息），但 KV 来自 frozen depth encoder（保证 3D structure 信息准确）。

附录 Table VII 显示 Depth Anything 3-small (83.00%) 反而比 base (69.67%) 和 large (82.00%) 好，说明 **depth encoder 不是越大越好** —— 大 encoder 引入太多 redundant tokens 反而干扰 learning。同时 downsampling depth tokens 也很关键：w/o downsample 掉到 68.00%，因为 depth tokens 太多会稀释 attention。

---

## 4. 总 Loss 和 Integration

最终 mixed loss：

$$\mathcal{L} = \mathcal{L}_{FM} + \lambda_{object} \mathcal{L}_{object} + \lambda_{skill} \mathcal{L}_{skill}$$

- $\mathcal{L}_{FM}$: flow matching loss (π0 的主 loss)
- $\lambda_{object}, \lambda_{skill}$: 都是 0.001（LIBERO）或 0.01（RoboTwin 2.0）

Depth head 没有 loss term，因为它通过 architectural injection (cross-attention 到 depth KV) 直接干预 attention 计算。

Table XIII 的 ablation 显示：
- Final setting: $(w_{obj}, w_{skill}) = (0.001, 0.001)$ → 87.83
- High / High: $(0.01, 0.01)$ → 85.77（auxiliary loss 太强干扰主 loss）
- Asymmetric settings 都比 balanced 低

**Intuition**：auxiliary supervision 应该作为 "regularizer" 而非 "primary objective"，所以权重要小。

---

## 5. Dataset Annotation Pipeline

这是工程上很关键的部分 —— 怎么低成本拿到 object mask 和 skill label。

### Object masks

Pipeline：
1. **Qwen3-VL** (https://arxiv.org/abs/2511.21631) 识别 stage-relevant object，输出 foreground point prompts
2. **SAM2** (https://arxiv.org/abs/2408.00714) propagate mask 到整个 video segment
3. **Human verification** 最后 step

Mask 再 convert 到 16×16 patch grid (PaliGemma 的 image token grid)：

$$m_p = \mathbb{I}[s_p \geq \tau], \quad p \in \mathcal{P}$$

- $s_p$: foreground coverage ratio for patch $p$
- $\tau$: threshold
- $\mathcal{P}$: patch set

### Skill labels

由 Qwen3-VL 从 stage description + predefined skill list 自动生成，再 convert 到 soft target (Eq. 21)。

效率数据：**92% episodes 不需要 human correction，50 episodes 标注只需 4 分钟（manual 是 43.5 分钟）**。这对 scale up VLA training data 很关键。

---

## 6. 实验结果

### 6.1 LIBERO-Plus (Table I)

LIBERO-Plus (https://arxiv.org/abs/2510.13626) 是 LIBERO (https://arxiv.org/abs/2311.07423) 的 robustness benchmark，在 7 个维度做 perturbation：camera, robot, language, light, background, noise, layout。

主要数字：

| Model | Total |
|---|---|
| OpenVLA | 15.6 |
| π0 | 68.2 |
| DreamVLA | 69.9 |
| OpenVLA-OFT | 69.6 |
| **GuidedVLA (all heads)** | **75.4** |

Head-specific gains：
- Object head 在 Object suite 最强（82.5%, +8.4%）
- Skill head 在 Goal suite 最强（68.9%, +7.5%）
- Depth head 在 Spatial suite 最强（81.4%, +3.7%）

这验证了 factor-task alignment 的 intuition —— 不同 factor 对不同 task 类型贡献不同。

### 6.2 RoboTwin 2.0 (Fig. 5)

RoboTwin 2.0 (https://arxiv.org/abs/2506.18088) 在 8 个 manipulation task 上测，full model 从 π0 的 77.38% 提升到 **90.63%**。

最 dramatic 的 case 是 **Click Bell**（需要精确 Z-axis 控制）：35% → 65% (depth head alone) → 65% (full model)。这印证了 depth 对 geometry-heavy task 的关键作用。

### 6.3 Real-World (Table II)

两个 platform：
- **ALOHA AgileX**: 3 个 household task (pick fruits & veggies, stack bowls, clean tabletop)
- **PSI-Bot RealMan**: 3 个 lab task (place beaker in heating mantle, stack beakers, heat beaker)

Generalization 三个 setting：in-domain (positional)、scene (distractor clutter)、lighting。

整体 Avg：
| Setting | Base π0 | GuidedVLA |
|---|---|---|
| In-domain | 55.8 | **75.8** |
| Scene | 44.2 | **67.5** |
| Lighting | 57.5 | **79.2** |

**Lighting generalization 提升最显著 (+21.7%)**，因为 explicit factor guidance 减少了 spurious appearance correlation，model 学到的是 task structure 而非 surface texture。

---

## 7. Sensitivity Analysis：Factor Quality ↔ Performance

这部分是 paper 最 convincing 的分析（Fig. 7）—— 不只问"with/without"，而是问"factor quality 提升是否 correlate with success"。

### Object grounding

人为控制 attention mass $m$ 到 $\alpha \in \{0.25, 0.5, 0.75, 1.0\}$，loss：

$$\mathcal{L}_{ablation} = \begin{cases} \frac{0.5(m-\alpha)^2}{\beta}, & \text{if } |m-\alpha| < \beta \\ |m-\alpha| - 0.5\beta, & \text{otherwise} \end{cases}$$

- $\beta = 0.05$ (Huber loss smoothing)

结果：0.25 → 61.3%, 1.0 → 74.6%，**单调正相关**。

### Skill recognition

Linear probe accuracy 从 25% → 100%，loss 用 Smooth L1：

$$\mathcal{L}_{ctrl} = \begin{cases} \frac{0.5(S-\gamma)^2}{\beta}, & \text{if } |S-\gamma| < \beta \\ |S-\gamma| - 0.5\beta, & \text{otherwise} \end{cases}$$

- $S = \frac{1}{N}\sum_{i=1}^N \hat{p}_i(y_i)$: soft accuracy
- $\gamma$: target accuracy

结果：25% → 66.2%, 100% → 72.9%，正相关。

### Depth feature

通过 noise injection 控制 depth signal strength：

$$\tilde{\mathbf{f}} = \delta \cdot \mathbf{f}_{depth} + (1-\delta) \cdot \boldsymbol{\epsilon}$$

- $\delta \in [0, 1]$: depth feature ratio
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: Gaussian noise matching depth feature statistics

结果：0 (pure noise) → 15.6%, 1.0 (clean depth) → 76.7%，**dramatic monotonic gain**。

这三个实验结合起来非常强 —— 证明了 factor 不只是 "with 比 without 好"，而是 **factor 质量越高，task 表现越好**，符合 causal interpretation。

---

## 8. Specialization vs Mixture (Fig. 9, Fig. 10)

这是另一个重要 analysis：**为什么不所有 heads 一起监督所有 factors？**

对比 two paradigms：
- **Specialization (Ours)**: 不同 heads 监督不同 factors
- **Mixture**: 所有 heads 监督所有 factors

Mixture underperforms 显著。t-SNE visualization (Fig. 10) 给出 intuition：
- Specialization: 三个 factor 的 attention outputs 形成 well-separated clusters
- Mixture: clusters overlap，features entangled

这呼应了 Mixture of Experts (AdaMoE, https://arxiv.org/abs/2510.14300) 的 routing 问题 —— 当多个 objective 共享同一组 parameters 时，gradient conflict 和 feature entanglement 会成为 bottleneck。

---

## 9. Layer-wise Guidance (Table X)

Guidance 应该施加在哪些 transformer layers？作者测了 all layers、4 个 quartiles：

| Layer subset | Total |
|---|---|
| All layers | 74.1 |
| 1st quartile (bottom) | 74.4 |
| 2nd quartile | 74.3 |
| **3rd quartile** | **75.4** |
| 4th quartile (top) | 73.8 |

**3rd quartile 最好** —— mid-to-upper layers。Intuition：底层 captures low-level features (颜色、纹理)，顶层接近 output，中间层是 abstract semantic representation 最适合施加 task-level guidance。

---

## 10. 与其他方法的对比

Table I 比了几个 related approach：

- **DreamVLA** (https://arxiv.org/abs/2506.15785): 用 VLM query 预测 dynamic regions, depth maps, semantic knowledge → 69.9%
- **VLA-Adapter** (https://arxiv.org/abs/2501.06234): Bridge Attention 注入 V-L condition → 59.1%
- **Spatial Forcing** (https://arxiv.org/abs/2510.08881): align VLA visual embeddings with 3D foundation models → 29.1%
- **AdaMoE**: MoE-based action experts → 50.1%

GuidedVLA (75.4%) 全面胜出。优势在于：
1. **Decoupling** —— 每个 factor 有自己的 head，避免 feature entanglement
2. **Plug-and-play** —— ControlNet-style residual adapter 保留 pretrained π0 的能力
3. **Interpretability** —— 可以 inspect 每个 head 学到了什么

---

## 11. Architecture 细节 (Table XI)

完整 architecture：
- **Vision encoder**: SigLIP (12 layers, 768 hidden, 256 tokens per view)
- **Multi-modal projector**: Linear 768 → 2048
- **Language backbone**: PaliGemma / Gemma-2B (18 layers, 2048 hidden)
- **Action expert**: Gemma-300M (18 layers, 1024 hidden, 50-step action chunk)
- **Depth encoder**: Depth Anything 3-small (frozen)
- **DepthKVProjector**: Linear K/V projections × 4 groups
- **Skill head**: Linear 256 → K
- **ControlAttention**: 18 layers ControlAwareAttention

Action chunk size 是 50 steps，inference 时 20Hz 输出 + linear interpolation 到 50Hz control rate。

---

## 12. Failure Analysis (附录 P)

作者分析了 π0 baseline 的三类 failure：

1. **Object grounding failures**: phantom grasps (approach 空气)，grasp offset 导致 slippage。Transparent glassware 因为 refraction 更严重。

2. **Metric geometry failures**: 毫米级 depth/clearance 不够，half-grasp on nested bowls，rim collision during heating mantle insertion。

3. **Temporal skill collapse**: 完成 visually salient subgoal 但 skip 后续阶段，比如 pouring 之后 forget tool return。

这三类 failure 跟三个 specialized heads 一一对应，验证了 factor selection 的合理性。

---

## 13. 我的几点想法

1. **Generalization potential**：paper 只 instantiate 三个 factors (object, skill, depth)，但 framework 本身是 extensible 的。可以加 force/torque head、human pose head、affordance head 等。作者在 Section VI 也提到这个方向。

2. **Skill label 自动化是 bottleneck**：continuous task 的 automatic skill labeling 还没解决。可以参考 LOTUS (https://arxiv.org/abs/2203.00752) 的 unsupervised skill discovery。

3. **跟 mechanistic interpretability 的连接**：这个 work 实际上是在做 **induction circuit specialization** —— 给 attention heads 显式 role assignment。跟 Anthropic 的 Circuits (https://transformer-circuits.pub/) 思路有共通之处。

4. **ZeroConv 的 risk**：训练初期 ZeroConv 是 0，所以 control branch 完全不参与。但训练到一定程度后，ZeroConv 学到 weight 之后，主 branch 和 control branch 的 gradient 会相互干扰。可能需要 gradient surgery 或者 PCGrad (https://arxiv.org/abs/2001.06782) 处理。

5. **Comparison with RLHF-style preference learning**：GuidedVLA 用 supervised guidance，跟 RLHF (https://arxiv.org/abs/2203.02155) 的 reward-based guidance 思路不同。两者结合是个 interesting direction —— 用 reward signal 自动发现 task-relevant factors。

6. **Data efficiency implication**：92% 不需要 human correction 的 annotation pipeline 让我想到 RT-2 (https://arxiv.org/abs/2307.15818) 的 web-scale data training philosophy。VLA 的下一个 milestone 可能就是 semi-automatic factor annotation at scale。

7. **3D representation 的选择**：Depth Anything 3 是 metric depth 还是 relative depth？(我理解是 relative)。如果用 VGGT (https://arxiv.org/abs/2503.11651) 或 DUSt3R (https://arxiv.org/abs/2312.14132) 这种 metric 3D reconstruction，可能效果更好。

8. **Action chunk temporal coherence**：paper 用 skill label 作为 action chunk 级的 supervision (Eq. 21)，但 action chunk 内部的 sub-step level temporal logic 还没显式 modeling。这个跟 ACT (https://arxiv.org/abs/2304.13705) 和 Diffusion Policy (https://arxiv.org/abs/2303.04137) 的 action chunking 思路需要结合。

---

## References

- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- LIBERO: https://arxiv.org/abs/2311.07423
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- ControlNet: https://arxiv.org/abs/2302.05543
- SAM2: https://arxiv.org/abs/2408.00714
- Depth Anything 3: https://arxiv.org/abs/2511.10647
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- DreamVLA: https://arxiv.org/abs/2506.15785
- AdaMoE: https://arxiv.org/abs/2510.14300
- Spatial Forcing: https://arxiv.org/abs/2510.08881
- VLA-Adapter: https://arxiv.org/abs/2501.06234
- RT-2: https://arxiv.org/abs/2307.15818
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT (ALOHA): https://arxiv.org/abs/2304.13705
- LoRA: https://arxiv.org/abs/2106.09685
- Shortcut learning (Geirhos): https://www.nature.com/articles/s42256-020-00257-z
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- Project page: https://guidedvla.github.io/project_page/

如果你对某个 factor (object / skill / depth) 的实现细节、或者 ControlNet-style adapter 的 gradient flow、或者 factor discovery 的 automation 方向感兴趣，我可以再深入展开。
