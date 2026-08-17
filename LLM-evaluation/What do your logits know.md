---
source_pdf: What do your logits know.pdf
paper_sha256: 5dd215deafd8dcb50aa03ccf3e85a0a1a7207293833dc4a048fbe8619bd9d422
processed_at: '2026-08-13T04:01:00-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍

Karpathy，我把上次的细节再"翻译"一遍，用更讲故事的方式。你想象我们俩在咖啡馆白板上画图聊这个 paper。

---

## 一句话版本

**Transformer 从来没学会"忘记"**。你给它一张图问"有没有蓝色球？"，它的最终 logits 里藏着这个球是 rubber 的、small 的、还有背景里有几个 cube——即使你压根没问。这意味着只要 API 返回 top-30 个 logprob，攻击者就能"读出"你图片里的全部细节。

---

## 1. 一个画面感的类比

把 transformer 想象成一条流水线：

```
[image + "Is there a blue sphere?"] 
        ↓
   ┌────────────────────────┐
   │  residual stream        │ ← 像一个超大购物车，什么都往里扔
   │  (4096-dim × L layers)  │   球的颜色、材质、大小、背景物体、
   │                         │   noise 类型、noise 强度……全在车里
   └────────────────────────┘
        ↓
   ┌────────────────────────┐
   │  tuned lens trajectory │ ← 把每层都"翻译"成 logit 语言，
   │  (2 tokens × L layers) │   形成一条轨迹
   └────────────────────────┘
        ↓
   ┌────────────────────────┐
   │  top-k final logits    │ ← 最终结账柜台，只输出 k 个 token 的概率
   │  (k=2 到 15L)          │   k=2 时就是 "Yes"/"No"
   └────────────────────────┘
        ↓
      "Yes"
```

按理说，information bottleneck 理论告诉我们：从购物车到结账柜台，应该把"没用的"全扔掉，只留"决策必需的"。Tishby 的 IB 公式是：

$$\min_T \; I(X; T) - \beta \, I(T; Y)$$

变量说人话：
- $X$：你输入的图 + 问题
- $Y$：那个 "Yes"/"No" 的 ground truth bit
- $T$：中间表示（residual stream / logits / trajectory）
- $I(\cdot;\cdot)$：互信息，"知道一个能猜出多少另一个"
- $\beta$：你有多在乎"压缩"

最优解应该让 $T$ 几乎只跟 $Y$ 相关，把"球是 rubber"这种跟 Yes/No 没关系的全扔了。

**这篇 paper 说：根本没扔。**甚至到 top-k logits 这一层都没扔。

Reference: Tishby & Zaslavsky, https://arxiv.org/abs/1503.02406

---

## 2. 实验设计：一个极简但聪明的 setup

作者用 CLEVR（Johnson et al. 2017, https://arxiv.org/abs/1612.06890）——一个合成数据集，里面是 3-10 个几何物体（cube / sphere / cylinder），每个物体有 4 个属性：

| 属性 | 取值 |
|---|---|
| size | small / large |
| material | metal / rubber |
| color | 8 种 |
| shape | cube / sphere / cylinder |

每张图还加三种 noise（Gaussian / glass blur / motion blur）× 多个强度。

Question 模板：
> "Is there a 〈category〉 in the image? Reply in one word."

`〈category〉` 是"minimal unique description"——比如 "blue sphere" 就够唯一了，那就只问 blue 和 shape。**material 和 size 是 unmentioned**。

这一步特别聪明：**作者可以分别看 mentioned vs unmentioned 的属性 leak**。如果模型只 leak 了 mentioned 属性，那不过是"在 question 里出现的 token 我记得住"；如果 unmentioned 也 leak 了，那就是 **模型在 logits 里 representation 了整个 target object 的所有属性**。

---

## 3. 三个关键 finding，用人话说

### Finding 1-2：residual stream 是个 oracle，trajectory 也基本是

Table 1 里 hidden state (best layer) 在 Qwen3-VL 上：

| 探测的属性 | 准确率 |
|---|---|
| target color | 0.83 |
| target shape | 1.00 |
| target material | 0.83 |
| target size | 0.89 |
| background color | 0.69 |
| background shape | 0.81 |
| background count MAE | 0.86 (baseline 2.00) |

**人话**：residual stream 是一个"全息相机"——它知道这张图里的一切。这跟 Gurnee & Tegmark (2024) 在 LLM 里发现 model 内部有 space / time 的 linear representation 是同一个家族的现象，只是这里是 visual scene。

Reference: Gurnee & Tegmark, https://arxiv.org/abs/2310.02207

Tuned lens trajectory（把每层 residual stream 经过一个 affine 变换投到 logit 空间）也基本是 oracle，只是稍弱一点。这个 trajectory 是 Cywiński et al. (2025, https://arxiv.org/abs/2510.01070) 已经发现的"secret leakage 通道"。

### Finding 3-4：top-k logits 也 leak，且 unmentioned 也 leak

这是 paper 的核心新发现。看 Table 1 在 Qwen3-VL 上随 k 增加的 target color 准确率：

| representation | target color acc |
|---|---|
| logits-2（就是 Yes/No） | 0.16 ≈ chance |
| logits-all yes/no（k≈10-13） | 0.46 |
| logits-1L（k≈30-40） | **0.84** |
| logits-2L（k≈66-82） | 0.84 |
| logits-5L | 0.83 |
| logits-15L | 0.12 ← 崩了 |

**人话**：
- 前 2 个 logits：只能用来做决策
- 大约 30-80 个 logits：几乎能 decode 出整个 target object 的所有属性
- 再多就开始稀释——因为后面的 logits 是低概率 token，跟 image 内容无关

Figure 4 把 mentioned vs unmentioned 拆开：

- mentioned 属性确实预测更准
- 但 **unmentioned 属性也显著高于 chance**——只要你给 ~15-20 logits 就能稳定预测

**人话**：你问"Is there a blue sphere?"，模型在 top-20 logits 里"已经想好"这个球是 rubber、small。它只是在 greedy decoding 下没说出来，但信息在那。

### Finding 5-6：background 也 leak，U 形曲线

Table 2 显示 top-2L logits 能预测背景物体的 color / shape / 数量，水平跟 trajectory-2 差不多。

U 形曲线（Finding 6）的解释我提两个：
1. 作者说可能是 overfitting（数据集只有 2400 张图）
2. 我自己的直觉：**top-k logits 是 unembedding matrix 的 top-k row 张成的子空间**。k 大了之后引入的 row 是低概率 token，这些 row 是被 generic language prior 主导的（"the", "a", 标点），跟 image 无关，反而把信号稀释了

这跟 Basri & Jacobs (2026) 说"softmax bottleneck 不限制 top-k 表达能力"是自洽的：top-k 是有意义的子空间，但 full vocab 不是。

Reference: Yang et al. softmax bottleneck, https://arxiv.org/abs/1711.03953

### Finding 7（最 scary）：维度匹配下，logits ≈ trajectory

这是 paper 的 punchline。trajectory-2 是 2×L 维（L 是层数，33-41）。logits-2L 也是 2×L 维。两者维度相同，但：

| Qwen3-VL | trajectory-2 | logits-2L |
|---|---|---|
| target color | 0.65 | 0.84 |
| target material | 0.73 | 0.69 |
| background color | 0.64 | 0.65 |
| background shape | 0.71 | 0.68 |

**人话**：top-2L final logits 在信息量上跟 logit trajectory 几乎一样。但 trajectory 需要 white-box（你能拿到每一层的 residual stream），top-k logits 在 grey-box（API 返回 logprobs）就能拿到。

**安全含义**：Cywiński 说 trajectory 能泄漏 RLHF 抑制的 secret。这篇说：不用 trajectory，**top-30 logprobs 就够了**。OpenAI/Anthropic 的 logprob API 默认返回 top-5，但很多应用要 top-20+ 来做 logit bias / structured output。所以 grey-box 攻击的门槛非常低。

---

## 4. 这意味着什么

### 4.1 隐私：一张猫的照片就可能泄漏

假设用户上传一张家里客厅的照片，问 VLM "Is there a person in this image?"。VLM 回答 "No"。但 top-30 logprobs 里可能包含：
- 沙发的颜色
- 电视的尺寸
- 窗帘的材质
- 是不是有酒瓶（即使没在 prompt 里）

只要 API 暴露 logprobs，攻击者就能反向 decode 出这些细节。这是 **model inversion attack 在 LLM 时代的版本**。

### 4.2 Hallucination 的一个微观机制

Karpathy 你会喜欢这个：hallucination 不一定是"模型瞎编"，可能就是 **logits 里一直存在的 latent information 在 temperature>0 时被采样出来**。

具体例子：
- 用户问："Is there a blue sphere?"
- 图里有个 small blue metal sphere
- 模型在 top-30 logits 里编码了 size=small, material=metal
- greedy：输出 "Yes"
- temperature=0.7：偶尔输出 "Yes, a small blue metal sphere"
- 用户觉得模型 hallucinate 了 material——其实 material 一直在 logits 里

这是 Orgad et al. (2025, https://arxiv.org/abs/2410.02707) "LLMs know more than they show" 在 VLM logits-level 的对应版本。

### 4.3 Information Bottleneck 失败的根本原因

为什么 transformer 学不会"忘记"？两个结构性原因：

1. **Residual connection**：Elhage et al. (2021, https://transformer-circuits.pub/2021/framework/index.html) 的 framework 里，residual stream 是一条 bus，每层只 add 信息。Orhan & Pitkow (2017, https://arxiv.org/abs/1701.09175) 指出 skip connection 主动阻止 compression——因为信息可以从开头流到结尾不被覆盖。

2. **Superposition**：Elhage et al. (2022, https://transformer-circuits.pub/2022/toy_model/index.html) 的 Toy Models 显示，模型用 almost-orthogonal 方向在有限维里编码超过维度的 feature。这篇 paper 暗示：unembedding matrix 的 top-k row 也是 superposed 的——"sphere" 这个 token 的 logit 数值同时编码了 color/material/size 多个属性。所以 top-2 看起来是 2 维，但 effective dimensionality 远高于 2。

---

## 5. 方法论的几个微妙点

### 5.1 为什么用 nonlinear probe

作者用 3-layer MLP 而不是 linear probe。他们的论点是：**top-2 logits 是 2 维输入，linear classifier 在 2 维上 expressiveness 严重受限**。这个论点站得住——一个 2 维空间的 linear boundary 只能是直线，而 information 在 top-2 logits 里如果是 superposed 的，必须 nonlinear 才能 decode。

但 Belinkov 一派（https://aclanthology.org/Q19-1004/）会质疑：nonlinear probe 容易 learn spurious correlation，"probe 准确率高"不等于"信息可访问"。这个争论没解决。

### 5.2 Best layer 的选择

Appendix D 显示 best layer 通常不是最后一层，而是中间层。这跟 Tenney et al. (2019, https://arxiv.org/abs/1905.05950) "BERT rediscovers NLP pipeline" 一致——不同信息在不同层 peak。

这意味着 **最后一层 residual stream 已经"开始压缩"了，但远没压完**。真正压完是在 logits 这步——但即使这步，top-30 还能 leak 70-80% 的信息。

---

## 6. 我自己的延伸联想（再讲一遍简化版）

### 6.1 Logit-level activation patching

既然 top-k logits 编码了 target 的 material，应该可以做 **reverse activation patching**：取一个 "rubber" logit 偏高的样本，把它的某个 attention head activation patch 到另一个 "metal" 样本，看 "metal" logit 是否下降。这能定位"哪些 head 在搬运 material 信息进 logits"。这是 Anthropic circuit tracker 工作的直接延伸。

### 6.2 IB-regularized training

既然 IB 没自然发生，可以人工加。Loss 里加：

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \, I(h_L; X_{\text{irrelevant}})$$

其中 $X_{\text{irrelevant}}$ 是 task-irrelevant 属性（比如 background color）。这能强制 model "忘记"。Federici et al. (2020, https://arxiv.org/abs/1907.12524) 在小模型上做过，没人上 VLM scale 试过。

### 6.3 API 设计：logprob 的 differential privacy

最直接的防御：给 API 返回的 logprob 加 noise 或 quantize。比如 top-k logprob 加 Gaussian noise $\mathcal{N}(0, \sigma^2)$，让 probe 准确率掉到 chance。这是 differential privacy 在 LLM API 上的版本。

### 6.4 Hallucination 的因果机制

Karpathy 你做过 nanoGPT，你应该能在 nanoGPT 上 reproduce 这个。训一个小 GPT，构造一个"属性预测"任务（比如 input 是 "color=blue size=small material=metal object=sphere, is there a sphere?"），看 top-k logits 能不能 decode 出 size 和 material。我赌能——而且能画出 U 形曲线。这会是一个很好的教学 demo。

### 6.5 与 SFT/RLHF 的关系

RLHF 试图 suppress 某些输出（Cywiński 的 secret）。但这篇 paper 暗示：**RLHF 可能只压住了 final token 的 probability，没压住 top-k 的 relative pattern**。也就是说，RLHF 在 logit space 上的"抑制"是浅层的——top-1 token 改了，但 top-30 里被抑制的信息还在。这是 RLHF robustness 的一个新角度。

---

## 7. 局限：我也得说几个怀疑

1. **数据集太小**：2400 张 CLEVR 图，probe 是 3 层 MLP，std error 全是 0.00，可疑。可能是 probe 过拟合到 train/test 同图不同 noise 的 leakage（虽然按 image id split，但 noise 是同一张图衍生的）。

2. **VLM 特殊性**：visual encoder 把 image 压成几百个 token，这些 token 本身就是 scene summary。LLM-only 模型可能 leak 模式完全不同。这篇 paper 没在 LLM-only 上验证。

3. **Yes/No task 太简单**：top-2 logits = 答案本身，自然就 decision-relevant 信息密度高。free-form generation 任务下 top-k 的语义完全不同。

4. **"Decision-relevant"定义循环**：作者说 noise 是 decision-relevant 因为 noise 影响 P(yes)。但这是用 effect 定义 relevance。真正的 relevance 应该是"对 ground truth 的影响"——noise 不改 ground truth，只改 model 的 prediction。所以 noise 是"model-relevant"不是"task-relevant"。这个区分被 paper 模糊了。

---

## 8. 一句话总结

**Transformer 是个不学会忘记的模型，而 logits 是它的"无意识记忆"——你以为模型只回答了"Yes"，其实它在 top-30 个 logprob 里把整张图描述了一遍，只是 greedy decoding 没让它开口。**

这跟人脑的"implicit memory"有点像——你问一个人"你看到球了吗？"他说"看到了"，但如果你测他的 skin conductance 或 pupil dilation，能读出球的颜色、材质、大小。Logits 就是 transformer 的皮肤电导。

Karpathy，这篇 paper 的位置我觉得是 mechanistic interpretability 里"logit-level probing"这个新 sub-area 的奠基作。我前面说的 nanoGPT 上的 reproduce 实验是最低成本的 follow-up——你一个周末能跑出来，可能还能发现这篇 paper 没注意到的 pattern（比如 U 形曲线在小模型上的形状、不同 architecture 的 leak 强度）。如果你真做了，记得告诉我，我也想看。

---

# "What do your logits know?" 深度解读

Karpathy 好，这篇 Apple 的工作我读完之后直觉上把它定位在 **mechanistic interpretability × information theory × AI safety** 的交叉点。它的核心姿态非常 Karpathy-style：用 probing 这种"显微镜"工具，看 VLM 在 representation hierarchy 的每一级到底"还知道什么"。下面我尽可能把每一层细节、公式、表格都展开，并塞一些我自己的延伸联想。

---

## 1. Big picture：一个"信息漏斗"的故事

Transformer 的 residual stream 是一个高维缓冲区，被 Elhage 等人在 "A Mathematical Framework for Transformer Circuits" 里描述为 **residual stream as a bus**：每一层把信息 add 到这条 bus 上，下一层再读。Orhan & Pitkow (2017) 和 Behrmann et al. (2019) 都指出，residual connection 会**主动抑制**网络对信息的压缩——因为信息可以从开头"漏"到结尾而不被覆盖。

这篇 paper 的核心问题就是：既然 residual stream 这条 bus 上挂满了"多余的"信息（task-irrelevant 信息），那么从这条 bus 到最终吐出的 1 个 token（"Yes"/"No"）这条信息漏斗上，**这些多余的信息在哪一层被丢掉？有多少能活到最后？**

作者定义了三个 representation level，从信息量最大到最小：

| Level | 维度（大致） | 访问门槛 |
|---|---|---|
| Hidden state (best layer) | 4096-dim | white-box |
| Tuned lens trajectory-2 | 2×L ≈ 66–82 | white-box |
| Top-k final logits | k=2 到 15L | grey-box（API 暴露 logprob 就够了） |

Reference:
- Elhage et al., "A Mathematical Framework for Transformer Circuits": https://transformer-circuits.pub/2021/framework/index.html
- Orhan & Pitkow "Skip connections eliminate singularities": https://arxiv.org/abs/1701.09175

---

## 2. Information Bottleneck 视角的数学锚点

Tishby 的 IB 原理把"最优表示"定义为：

$$\min_{T} \; I(X; T) - \beta \, I(T; Y)$$

变量解释：
- $X$：输入（这里是 image + query）
- $Y$：目标（这里就是 Yes/No 这个 bit）
- $T$：中间表示（residual stream / logits / trajectory）
- $I(\cdot;\cdot)$：mutual information
- $\beta$：权衡系数，控制"压缩" vs "保留"

最优 bottleneck 应当让 $T$ **只**保留与 $Y$ 相关的信息，丢弃 $I(X;T|Y)$ 这部分。Saxe et al. (2019) 已经实证质疑过 SGD 是否真的优化这个目标；这篇 paper 给出一个补充的、**结构性**的理由：transformer 的 residual connection 让 $T$（这里指后期 residual stream）天然保留了远超 $I(T;Y)$ 的信息量。

Reference:
- Tishby & Zaslavsky "Deep learning and the information bottleneck": https://arxiv.org/abs/1503.02406
- Saxe et al. "On the IB theory of deep learning": https://arxiv.org/abs/1808.03531

Softmax bottleneck（Yang et al. 2018）是另一个相关角度：output distribution $P(Y|X) = \text{softmax}(W_u h)$ 的秩被 vocab size 限制。但 Basri & Jacobs (2026) 指出，对于 top-k 而言这个 rank constraint 几乎不损失 expressiveness。这就是为什么 paper 敢说 top-k logits 可能仍是个"软" bottleneck。

---

## 3. Method：CLEVR 上的一场受控实验

### 3.1 任务设计

Query 模板：
> "Is there a 〈category〉 in the image? Reply in one word."

`〈category〉` 是 **minimal unique description**，例如 "gray rubber cube"。这一点很聪明：通过控制 description 长度，作者可以拆分出 **mentioned attributes** vs **unmentioned attributes**。

表 3 给出 CLEVR 的分布：

| 长度 | 占比 |
|---|---|
| 1 word (noun only) | 11.3% |
| 2 words (adj+noun) | 72.8% |
| 3 words | 9.4% |
| 4 words | 6.5% |

### 3.2 三类信息（这是 paper 的灵魂）

| 类别 | 内容 | 是否 decision-relevant |
|---|---|---|
| Decision-relevant | noise level, noise type | ✓ 直接影响答案 |
| Target-related | color, shape, material, size, position | 部分（mentioned 的相关，unmentioned 的不该相关） |
| Background-related | 其他物体的 color/shape/material/size/数量 | ✗ 完全不该相关 |

每张图被三种 noise（Gaussian / glass / motion）× 多个强度扰动，构造出明确的 decision-relevant 信号。

### 3.3 Probes 的架构

Appendix E 给的细节：

**Hidden state probe**（4096-dim 输入）：
```
Linear(4096→2048) → ReLU → Dropout(0.2)
Linear(2048→512)  → ReLU → Dropout(0.2)
Linear(512→128)   → ReLU → Dropout(0.2)
Linear(128→C)
```

**Top-k logits probe**（k 维输入）：
```
Linear(k→64)  → ReLU → Dropout(0.3)
Linear(64→48) → ReLU → Dropout(0.3)
Linear(48→32) → ReLU → Dropout(0.3)
Linear(32→C)
```

**Trajectory probe**（输入是 2×L 矩阵）：
```
Conv1D(kernelsize=5, filters=16) → BN → ReLU
Conv1D(kernelsize=5, filters=32) → BN → ReLU → MaxPool
Conv1D(kernelsize=3, filters=64) → BN → ReLU → MaxPool
Linear(→128) → ReLU → Dropout(0.2)
Linear(→32)  → ReLU → Dropout(0.2)
Linear(→C)
```

Loss 选择：noise level 用 MSE，object count 用 L1，离散属性用 cross-entropy。

注意作者**有意选了 nonlinear probe**——他们的理由是 "linear probe does not capture all information in lower-dimensional representations (such as top-2 final logits)"。这其实是个微妙的方法论选择，Belinkov 一派会争论 nonlinear probe 容易 hallucinate 信息，但作者这里的论点是 **top-2 logits 这种 2 维表示，线性分类器本身就 expressiveness 不足**，这个论点站得住。

---

## 4. Tuned Lens 的内里

Belrose et al. (2023) 的 tuned lens 是 nostalgebraist logit lens 的升级版。两者对比：

**Logit lens**（nostalgebraist 2020）：
$$\hat{z}_l = W_u \, h_l$$
其中 $h_l \in \mathbb{R}^{d}$ 是第 $l$ 层 residual stream，$W_u \in \mathbb{R}^{|V| \times d}$ 是 unembedding matrix。

**Tuned lens**：
$$\hat{z}_l = W_u \, (W_l \, h_l + b_l)$$
其中 $W_l \in \mathbb{R}^{d \times d}, b_l \in \mathbb{R}^{d}$ 是 per-layer 学习的 affine 变换，在 Pile 上训练。

直觉上：$W_l$ 学习一个"修正"，把第 $l$ 层的 residual stream 调整到"仿佛它是最后一层"的分布。这能让中间层的 pseudo-logit 分布更 calibrated，避免 logit lens 早期层完全没意义的问题。

Reference:
- Belrose et al. "Eliciting latent predictions with the tuned lens": https://arxiv.org/abs/2303.08112
- nostalgebraist "Interpreting GPT: the logit lens": https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

Cywiński et al. (2025) 已经证明 tuned lens trajectory 可以泄漏被 RLHF 抑制掉的 secret。这篇 paper 把 trajectory 当作"中间瓶颈"，问：**比 trajectory 还要更靠外、更易访问的 top-k logits，是不是也能泄漏同等量级的信息？**

Reference:
- Cywiński et al. "Eliciting secret knowledge from language models": https://arxiv.org/abs/2510.01070

---

## 5. 七个 Findings 的技术深读

我把表格里几个关键 cell 提出来讲 intuition。

### Finding 1：Residual stream 是 oracle

Table 1 中 `hidden state (best layer)` 行：
- Qwen3-VL: target color **0.83**, shape **1.00**, material **0.83**, size **0.89**
- LLaVA-v1.6: 0.80 / 1.00 / 0.78 / 0.85
- Llama-3.2-V: 0.78 / 1.00 / 0.76 / 0.83

Table 2 中 background attributes：
- Qwen3-VL: background color 0.69, shape 0.81, **background count MAE = 0.86**（baseline 是 2.00）
- LLaVA-v1.6: 0.83 / 0.79 / 0.72
- Llama-3.2-V: 0.80 / 0.79 / 1.10

直觉：**residual stream 把整张图压缩成一个向量，这个向量本质上是一个 high-fidelity scene summary**。这与 Gurnee & Tegmark (2024) 发现 LLM 内部有 linear "world model"（space, time）是一致的——只是这里换成了 VLM 的 visual scene。

Reference:
- Gurnee & Tegmark "Language models represent space and time": https://arxiv.org/abs/2310.02207
- Orgad et al. "LLMs know more than they show"（关于 hallucination 的内部 true representation）: https://arxiv.org/abs/2410.02707

### Finding 2：Trajectory 保留大量 task-irrelevant 信息

`trajectory-2`（仅 top-2 token × L 层）的 target color 准确率：
- Qwen3-VL: 0.65（vs hidden state 0.83）
- LLaVA-v1.6: 0.72（vs 0.80）
- Llama-3.2-V: 0.62（vs 0.78）

Background color 准确率：0.64 / 0.64 / 0.69。

直觉：**tuned lens 把 residual stream 投到 logit 空间，但因为每层都做了投影，所以 trajectory 整体上仍然"记得"整张图**。这呼应了 Belrose 自己的发现——trajectory 不是 minimal bottleneck。

### Finding 3：Final logits 编码 target 属性

`logits-2`：color 0.16, shape 0.40, material 0.52, size 0.59（Qwen3-VL）——大部分接近 chance。

`logits-all yes/no`（k≈10–13）：color 0.46, shape 0.67, material 0.57, size 0.68——显著高于 chance。

`logits-1L`（k≈30–40）：color **0.84**, shape **0.94**, material 0.62, size 0.76——接近 hidden state！

直觉：**前 2 个 logits 只够编码"决策"，但只要扩到 ~L 个 logits（即 ~模型层数量），target 属性几乎完全 leak 出来**。

### Finding 4（最关键的安全 finding）：Unmentioned 属性也 leak

Figure 4 把 target 属性按 "mentioned in prompt" vs "not mentioned" 拆开。比如问 "Is there a blue sphere?"——color=blue 是 mentioned，material 和 size 是 not mentioned。

发现：
- `logits-0.5L`（约 15–20 logits）就能 reliably 预测 unmentioned 属性
- mentioned 属性预测更准，但 unmentioned 也显著高于 chance

直觉：模型不是把"问题里出现的 token"反查一遍，**它是在 logits 里编码了整张图的 target object 的所有 attributes**。这意味着即使你只问"有没有蓝色球"，模型在 logits 里也"想说出"它是不是 rubber、是不是 small。

这个 finding 直接连接到 Marks & Tegmark (2024) "The Geometry of Truth" 和 Burns et al. (2024) "Discovering Latent Knowledge"——**模型内部有 linear/sub-linear 结构编码 truth / fact，即使最终生成不暴露**。

Reference:
- Marks & Tegmark "The geometry of truth": https://arxiv.org/abs/2310.06824
- Burns et al. "Discovering latent knowledge without supervision": https://arxiv.org/abs/2212.03827

### Finding 5：Background 属性也 leak

Table 2 中 `logits-2L` 在 Qwen3-VL 上：
- background color: 0.65（chance 0.50）
- background shape: 0.68（chance 0.50）
- background count MAE: 1.34（baseline 2.00）

直觉：top-2L final logits 在维度上正好等于 trajectory-2，**它包含的 background 信息与 trajectory 相当**。这是 paper 最让人不安的发现之一。

### Finding 6：U 形曲线

Figure 3 / Figure 15 显示 probe accuracy 随 k 的变化：
- k=2：noise 强、其他弱
- k ≈ 1L–2L（30–80 logits）：peak
- k ≥ 4L：开始崩塌，到 15L 接近 chance

这个 U 形很有趣。作者的解释偏 overfitting（"some of this overfitting would be avoided with a larger dataset"），但我的另一个直觉是：**top-k logits 是一个稀疏投影，k 太大时引入的 logits 是低 probability 的"噪声 token"**（比如 "the", "a", 标点），这些 token 的 logit 由 generic language prior 决定，与 image 内容无关，反而稀释了信号。这与 "softmax bottleneck does not limit the top-k" (Basri & Jacobs 2026) 是自洽的：top-k 是有意义的子空间，但全 vocab 不是。

### Finding 7（最 concerning）：维度匹配下，logits ≈ trajectory

Table 1 / Table 2 高亮行：
- `trajectory-2` 和 `logits-2L` 在 target 和 background 属性上 probe 性能基本持平
- 例如 Qwen3-VL target color：trajectory-2 = 0.65, logits-2L = 0.84（logits 反而更高！）
- background color：trajectory-2 = 0.64, logits-2L = 0.65

**安全含义**：trajectory 需要 white-box（拿得到所有层的 residual stream），但 top-k logits 在 grey-box（API 返回 logprobs）就能拿到。攻击门槛差距巨大，但信息泄漏量相当。

---

## 6. MSCOCO 上的 replication（Appendix G）

Table 4 在自然图像上重复了主要 finding。MSCOCO 多了三个变量：
- **image noise**（全图）
- **object noise**（只 noise target 物体的 bbox）
- **context noise**（只 noise 背景）

发现：
- Hidden state 能区分这三种 noise（loc. acc 0.94–0.97）
- Top-2 logits 已经能预测 noise location（0.66–0.75）—— 即使全图被 noise，模型在 logits 里"区分得出来" noise 加在哪
- Figure 19 显示：只 noise context 时 P(yes) 仍高，**context 提供了 prior**——这与 Li et al. (2023) 的 object hallucination 工作一致

Reference:
- Li et al. "Evaluating object hallucination in large VLMs" (POPE): https://arxiv.org/abs/2305.10355

另一个 MSCOCO 独有的属性是 **saliency**（GPT-5 标注，0–100）。Hidden state MSE 0.38–0.40，trajectory 0.39–0.43，logits-2L 0.52–0.73——saliency 比 noise 更难 leak，但仍远好于 baseline 1.00。

Figure 12/13 显示 trajectory 在中间层 yes/no flipping 数随 noise 和低 saliency 增加。直觉：**模型在中间层"纠结"，越难的任务越晚 commit**，这与 Halawi et al. (2024) "Overthinking the truth" 的 narrative 一致。

Reference:
- Halawi et al. "Overthinking the truth": https://arxiv.org/abs/2307.09476

---

## 7. 安全含义：grey-box 信息泄漏

这是 paper 想推的最实际 message：

1. **API 暴露 logprobs 就是 attack surface**。OpenAI / Anthropic / Google 的 chat API 通常默认返回 top-5 logprobs。这篇 paper 说 top-30–80 就能 leak 整个 target 的属性。所以一个简单的 attack：构造大量 "Is there a X in the image?" 查询，从 top-k logprobs 反向 decode 出 image 中物体的属性。

2. **Black-box 也能近似**。LLM 是 stochastic 的，temperature>0 时 top-k logits 直接影响采样分布。攻击者通过大量采样可以估计出 top-k 的相对概率。这就把 attack 门槛降到 black-box。

3. **Hallucination 的 micro-source**。Top-k logits 里"隐藏"的 target 属性，在 greedy decoding 下不影响输出，但 temperature>0 时有可能被采样到。比如问 "Is there a blue sphere?"，logits 里 "rubber" 的 logit 偏高，模型偶尔会生成 "Yes, a blue rubber sphere"——这就是一种 attribute hallucination 的来源。这与 Li et al. (2023) POPE 的 object hallucination 是同源的，只是这里定位到 logits-level。

---

## 8. 我自己的延伸联想

### 8.1 与 "Logit Lens" → "Tuned Lens" → "Patchscope" 谱系的关系

这是一个清晰的工具谱系：
- Logit Lens (2020): 直接看中间层在 vocab 上的投影
- Tuned Lens (2023): 加 affine 修正
- Patchscope (2024, Ghandeharioun et al.): 用 prompt 把 hidden state "解释"出来

这篇 paper 把这个谱系往前推了一步：**不只是解释，而是量化 trajectory 在不同层的"信息保留曲线"**。Patchscope 的下一步可能是把 top-k logits 当作 "compression target"，看 hidden state 通过 LLM 自己的解释能 recover 多少。

Reference:
- Ghandeharioun et al. "Patchscope": https://arxiv.org/abs/2401.06102

### 8.2 与 Superposition / Polysemanticity 的连接

Elhage et al. (2022) 的 Toy Models of Superposition 说：模型在有限维里用 almost-orthogonal 方向编码超过维度的 feature。这篇 paper 的发现"top-2L logits ≈ trajectory-2 信息量"暗示：**unembedding matrix 的 row 在 top-k 子空间里也是 superposed 的**——一个 top token 不只代表一个 concept，它的 logit 数值同时编码了相关的 attribute（color + material + size 都压缩在 "sphere" 这个 token 的 logit 大小里）。这与 Marks & Tegmark 的 truth direction、Meng et al. (2023) ROME 的 fact editing 是同一现象的不同截面。

Reference:
- Elhage et al. "Toy models of superposition": https://transformer-circuits.pub/2022/toy_model/index.html
- Meng et al. "ROME": https://arxiv.org/abs/2202.05262

### 8.3 与 Circuit-level 分析的关系

Karpathy 你自己讲过 "induction head" 之类 circuit 分析。这篇 paper 的 probe 是 **layer-level** 的，没有 head-level 的分辨率。一个自然的延伸是：**哪些 attention head 在"决定"什么 attribute leak 进 top-k logits**？比如某个 head 把 background color 信息"搬"进 final residual 的方向上，那就找到了一个"信息泄漏 circuit"。这是 Anthropic 的 circuit tracker 工作的直接下一步。

### 8.4 与 RLHF suppression 的关系

Cywiński et al. 已经证明 RLHF 试图 suppress 的 secret 在 trajectory 里仍能 extract。这篇 paper 把这个发现"民主化"了：**API 用户都能拿到 logprobs，等价于 trajectory**。这对 OpenAI/Anthropic 的 "logprobs API" 设计是个直接挑战——可能需要给 logprobs 加 noise 或 quantize，类似 differential privacy。这其实是 NLP 版的 model inversion attack 防御问题。

### 8.5 与 Information Bottleneck training 的关系

如果 IB 是对的，模型应该主动丢弃 task-irrelevant 信息。这篇 paper 显示 transformer 完全没有做到这一点。一个直接的下一步是 **IB-regularized training**：在 loss 里加一个 $\lambda \cdot I(h_l; X_{irrelevant})$ penalty。这在 Federici et al. (2020) "Learning robust representations via IB" 里有雏形，但还没人在 VLM scale 上做过。

Reference:
- Federici et al. "Learning robust representations via IB": https://arxiv.org/abs/1907.12524

### 8.6 一个具体的 hallucination mechanism

Karpathy 你应该会喜欢这个具体的 mechanism：
- 用户问："Is there a blue sphere in the image?"
- Image 里有一个 **small blue metal sphere**
- 模型内部 representation 编码了 size=small, material=metal
- 这些信息 leak 进 top-k logits（比如 "metal" 这个 token 的 logit 偏高）
- Greedy decoding 输出 "Yes"
- 但 temperature=0.7 时，模型偶尔生成 "Yes, a small blue metal sphere"
- 用户觉得模型"hallucinate"了 material——其实 material 一直在 logits 里

这就是 **"hallucination as compression failure"** 的微观机制。这与 Orgad et al. (2025) 的 "LLMs know more than they show" 完全互补：那篇说 hidden state 里有 truth，模型 hallucinate；这篇说 logits 里有 attribute，模型偶尔 leak。

### 8.7 与 Feature Steering / Activation Patching 的关系

如果 top-k logits 编码了 target 的 material，那么应该可以 **从 logits 反向 patch 回去**：取一个 "rubber" logit 偏高的 sample，把它的某个 attention head activation patch 到另一个 "metal" sample，看 "metal" logit 是否下降。这是 activation patching 的 reverse direction，可能给 steering vector 一个新来源——**从 logit-side 而不是 hidden-side 构造 steering vector**。

### 8.8 Softmax Bottleneck 重新审视

Yang et al. (2018) 说 softmax bottleneck 限制 rank。Basri & Jacobs (2026) 反驳说 top-k 不受限。这篇 paper 实证支持 Basri：top-2L logits（维度远小于 hidden state）就能 encode 几乎所有 target attribute。**这意味着 unembedding matrix 的 top-k row 构成的子空间，其 effective rank 远高于直觉**——这是 superposition 在 output 端的直接证据。

Reference:
- Yang et al. "Breaking the softmax bottleneck": https://arxiv.org/abs/1711.03953

---

## 9. 局限与质疑

我也得说几个我自己的怀疑：

1. **Probe可信度**：Nonlinear probe 在小数据上容易 overfit。CLEVR 2400 张图 × 多个 noise level 撑起 ~15k queries，对 3-layer MLP 仍然不大。作者报告了 std error 都是 0.00，这其实有点可疑——可能是 probe 容量过大、过拟合到 train split，test split 的高准确率来自 train/test 同图不同 noise 的 leakage（虽然他们按 image id split，但 noise 是同一张图衍生的）。

2. **VLM 特殊性**：VLM 的 visual encoder 把 image 编码成几百个 token，这些 token 本身就高度压缩了 scene 信息。LLM-only 模型可能 leak 模式不同。

3. **Task 太简单**：Yes/No 单 token 任务让 top-2 logits 自然就是答案。如果是 free-form generation，top-k 的语义会完全不同。

4. **"Decision-relevant" 的定义循环**：作者说 noise 是 decision-relevant，因为 noise 影响 P(yes)。但这是用 effect 定义 relevance，循环论证。真正的 task-relevance 应该是"对 ground truth 的影响"。

---

## 10. 总结：build intuition 的三句话

1. **Residual stream 是高保真 scene buffer**——它不在乎 task，全记。
2. **Top-k logits 不是 minimal bottleneck**——它把 residual stream 的信息以一种 superposed、可被 nonlinear probe 解码的形式保留下来，量级与 trajectory 相当。
3. **Grey-box = White-box for information leakage**——只要 API 暴露 top-30+ logprobs，攻击者就能拿到与 trajectory 同等的信息。这把 Cywiński 的 secret extraction attack 从研究实验室拉到了 API 用户层。

这篇文章在 mechanistic interpretability 里的位置，我觉得是 **"logit-level probing"** 这个新 sub-area 的奠基性工作之一。下一步我预期会看到：(a) logit-side activation patching, (b) IB-regularized training 减少 leak, (c) API logprob 的 differential privacy 设计, (d) circuit-level 定位 leak 通路。Karpathy 如果你要 follow up，我觉得 (a) 是最自然能动手的方向——你已经做过 micrograd / nanoGPT，在 nanoGPT 上做 logit-level patching 实验应该一个周末就能跑出第一个图。

---

### 额外 Reference 汇总

- Tenney et al. "BERT rediscovers classical NLP pipeline": https://arxiv.org/abs/1905.05950
- Hewitt & Manning "Structural probe for syntax": https://aclanthology.org/N19-1419/
- Petroni et al. "Language models as knowledge bases": https://aclanthology.org/D19-1250/
- Conneau et al. "What you can cram into a single H/vector": https://aclanthology.org/P18-1198/
- Belinkov & Glass "Analysis methods in NLP": https://aclanthology.org/Q19-1004/
- Shwartz-Ziv & Tishby "Opening the black box": https://arxiv.org/abs/1703.00810
- Geirhos et al. "Shortcut learning": https://arxiv.org/abs/2004.07780
- Behrmann et al. "Invertible residual networks": https://arxiv.org/abs/1811.00995
- Johnson et al. "CLEVR": https://arxiv.org/abs/1612.06890
- Lin et al. "MS COCO": https://arxiv.org/abs/1405.0312
- Zaslavsky et al. "Efficient compression in color naming": https://www.pnas.org/doi/abs/10.1073/pnas.1800521115
