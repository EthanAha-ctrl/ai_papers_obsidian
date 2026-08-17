---
source_pdf: Embed-RL.pdf
paper_sha256: a1fc7b181aca706b6dcaa83c1af7918e603c4b44fba810b463dd7421bcc0a1b0
processed_at: '2026-08-04T03:21:33-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Embed-RL

Andrej，咱们抛开公式，用最直白的话聊聊这篇 paper 到底在干嘛。

---

## 一句话版本

让一个 AI 先"想清楚"要找啥（生成带证据的推理链），再让另一个 AI 拿这个推理链去算 embedding，然后**用检索结果反过来教训第一个 AI"你刚才想得不对，重新想"**。

---

## 为什么需要这玩意

先说背景。假设你要做图片搜索——输入一句话，找最匹配的图。

传统做法（CLIP 那套）：把文字和图片各塞进一个 encoder，出来两个向量，算 cosine similarity。简单粗暴，但遇到复杂 query 就拉胯，因为 text encoder 不够聪明。

后来大家发现：**MLLM（像 Qwen-VL 这种多模态大模型）本身就很会理解图文，为啥不直接拿它当 encoder？** 于是有了 VLM2Vec、GME 这些工作——取 MLLM 最后一层 hidden state 当 embedding。效果确实好了一截。

再后来有人想：**MLLM 不是会"思考"吗？能不能让它先推理一下再出 embedding？** 这就是 generative embedding 的思路。这里头有两个代表：

### UME-R1 的做法（有问题）
一个模型干两件事：边生成 CoT 推理，边输出 embedding。听着美好，但**两个目标打架**——

- contrastive loss 说："把正样本向量拉近，负样本拉远"
- next-token prediction 说："下一个 token 要预测准"

这俩在模型内部争抢同一批参数的梯度方向，像一个人同时被喊"往左"和"往右"，结果原地打转。

### TTE 的做法（也有问题）
那就拆开呗——一个 Reasoner 生成 CoT，一个 Embedder 用 CoT 算 embedding。

但问题在于 Reasoner 是**预训练好的，没跟 Embedder 联合训练**。它生成的 CoT 可能跟检索任务半毛钱关系没有，甚至胡说八道（hallucination），反而污染了 Embedder 的输入。

而且更关键：**TTE 的 CoT 只有文字**。你说"帮我找一张猫的图"，它就输出一段文字描述猫长啥样。但 MLLM 明明能看图啊！为啥不直接告诉它"猫在图片左下角这个位置"？

---

## Embed-RL 怎么解决

核心就三招：

### 第一招：拆开，但让 Embedder 当"老师"教训 Reasoner

跟 TTE 一样拆成 Reasoner + Embedder 两个模块。但区别在于：

1. 先把 Embedder 用对比学习训好，然后**冻住不动**
2. 让 Reasoner 生成 CoT，喂给 Embedder 算 embedding
3. 看 embedding 检索效果好不好，**好就奖励 Reasoner，差就惩罚**
4. 用 GRPO（DeepSeek-R1 那套）更新 Reasoner 的 policy

这就相当于：**Embedder 训好之后变成了一个"检索裁判"**。Reasoner 生成的 CoT 好不好，不看它文字写得多漂亮，就看它能不能帮 Embedder 把正样本排在第一名。

这跟数学推理里"答案对不对可以验证"的思路一模一样——数学题有标准答案，检索任务有正样本，都是 **verifiable reward**。RL 在有 verifiable reward 的地方才好使。

### 第二招：CoT 不只是文字，要带"视觉证据"

这是最 elegant 的点。Reasoner 生成的 CoT 长这样：

```
<thinking>
我需要找一只猫。文字关键词：{"text_keywords": ["cat", "sitting"]}
图片里猫在左下角：{"bbox_2d": [100, 600, 300, 850]}
</thinking>

<rethink>
关键区域是左下角的猫，特征是坐姿、橘色
</rethink>

<answer>
找一只坐着橘猫的图，位置在左下角
</answer>
```

然后 Embedder 不光读这段文字，还会**把 bbox 指定的区域 crop 出来**，单独当一张图喂进去。Video 同理——Reasoner 说"第 3、7、12 帧最关键"，Embedder 就把这几帧单独抽出来看。

为啥这招有效？因为 MLLM 处理图片时，一张图被切成几百个 patch token，attention 会被大量冗余区域稀释。你直接告诉它"重点看左下角"，等于给它做了个 **learnable RoI pooling**。视频更明显——几百帧里大部分是废话，你告诉它哪几帧关键，它就不用在海量帧里瞎找。

这能 decouple 的原因是：Reasoner 只管生成，不用管 contrastive loss，所以可以自由输出 bbox 坐标、帧索引这种"非语言"结构。UME-R1 那种 joint training 做不到——NTP loss 会把 bbox 坐标退化成"预测下一个数字"的无聊任务。

### 第三招：双重 reward 防作弊

光看检索结果给 reward 有个漏洞——Reasoner 可能学到捷径，比如直接把 query 里的词抄一遍当 CoT，让 Embedder 靠 token overlap 拿高分，但根本没真推理。

所以加了第二个 reward：**找一个独立的 VLM（裁判），让它判断 query 的 CoT 和 target 的 CoT 是不是真的对得上**。对得上给 1 分，对不上给 0 分。

这相当于说：你不光要检索准，你的推理过程本身还得"说得通"。这是给 reasoning trajectory 加了可解释性约束。

---

## 实验结果说明了啥

最硬核的数字：**4B 参数的 Embed-RL 打过 7B 的 UME-R1 3.6 分**。用更小模型更好效果，说明这套 decoupled + multimodal CoT 的设计比单纯 scale up 有效得多。

几个有意思的细节：

**VisDoc OOD 暴涨 30 分**（67.1 vs 37.6）。OOD 就是训练时没见过的文档布局。bbox 机制让模型学会了"找关键区域"这个能力，换一种文档 layout 它照样能定位关键信息，而不是死记某一种排版。

**Video spatial fine-grained 拿第一**（89.9 vs GVE-7B 的 84.6）。bbox 对"找特定物体/外观"这种任务简直是降维打击——直接框出来喂给 Embedder，比让它在整张图里找高效多了。

**去掉 T-CoT 直接崩 6.6 分**，video 崩 8.4 分。这是最狠的 ablation——证明 T-CoT 不是锦上添花，是 embedding quality 的地基。没有推理做中间桥梁，MLLM 的 hidden state 直接当 embedding 在复杂任务上严重退化。

**RL 前后 bbox/keyframe 数量变化方向相反**：image 变多（多抓证据提 recall），video 变少（focus 关键帧提精度）。模型自己学会了"不同模态用不同策略"，这挺聪明的。

---

## 我觉得最 cool 的点

这篇 paper 真正的 insight 不是某个具体技巧，而是**把 "reasoning model" 和 "embedding model" 这两个看似不相关的方向统一了**。

以前大家觉得：reasoning 是给 VQA、数学题这种"有答案"的任务用的，embedding 是检索这种"没标准答案"的任务用的。

但这篇 paper 说：**检索也有 verifiable reward——正样本排第几就是答案**。所以你可以用 RL 训 Reasoner 生成对检索有用的推理链，就像用 RL 训数学模型生成对解题有用的推理链一样。

这个 framing 很 powerful。它意味着未来所有"有 ground truth pair"的任务（检索、匹配、排序）都可以套这套 "Reasoner + frozen judge + RL" 的框架。

而且 multimodal CoT 这一步打开了想象空间——Reasoner 可以输出任意结构化证据（bbox、keyframe、未来的 segmentation mask、audio segment...），只要 Embedder 能用。这是 "thinking with images" 在 embedding 领域的落地。

---

## 简单比喻

如果还不清楚，这么想：

传统 embedding 像让一个人看一眼图片就写个关键词标签，靠标签匹配找图。简单但粗糙。

generative embedding with text CoT 像让这个人先写段文字描述图片，再根据描述匹配。好一点，但描述可能跑题。

Embed-RL 像让一个人拿着放大镜（bbox）先仔细看重点区域，挑出关键帧，总结成带证据的推理笔记，再交给另一个专门做匹配的人。而且第一个人的笔记质量会被第二个的匹配结果**反馈纠正**——笔记没用就挨骂，下次改写法。

这俩人分工明确：一个负责"看懂、定位、总结"，一个负责"算相似度"。谁也不干扰谁。

---

# Embed-RL 深度技术讲解

Andrej，这篇 paper 的核心思路非常 elegant，下面我从 motivation、方法、公式、实验四个层次系统拆解，尽量给你 build 出完整的 intuition。

---

## 1. 背景与 motivation：为什么 generative embedding 会"打架"

Universal Multimodal Embedding (UME) 的目标：给定一个 query $q$（text/image/video 或它们的交错形式），从 candidate 集合 $\Omega=\{c_n\}_{n=1}^N$ 里检索最相关的 target。这本质是学一个 encoder 把任意模态输入压成一个向量，让向量相似度反映语义相关性。

主流路线演化分三个阶段：

**Stage 1: Dual-encoder 对比学习**（CLIP [43], BLIP [34], SigLIP [68]）。缺点是 text encoder 容量有限，难以处理 interleaved 输入。
- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP: https://arxiv.org/abs/2303.15343

**Stage 2: MLLM 当 encoder**（VLM2Vec [26], GME [72], UniME [18], CAFe [64]）。直接取 MLLM 最后一层 hidden state 作 embedding，但这是 **discriminative** 用法，没充分利用 MLLM 的生成和推理能力。
- VLM2Vec: https://arxiv.org/abs/2410.05160
- GME: https://arxiv.org/abs/2412.16855
- UniME: https://arxiv.org/abs/2504.17432
- CAFe: https://arxiv.org/abs/2503.19900

**Stage 3: Generative embedding with CoT**。这里有两个代表：
- **UME-R1 [30]**：在同一个 MLLM 上同时优化 contrastive loss 和 next-token prediction (NTP)，让模型边生成 CoT 边输出 embedding。问题：两个 loss 的 gradient 冲突（参考文献 [8, 30]）。直觉上，contrastive loss 想把 last token 的 hidden state 拉向正样本远离负样本，NTP 想让 token 分布拟合 ground-truth next token，两者在 hidden state 几何上往往指向不同方向，导致 suboptimal。
- **TTE [13]**：decoupled 架构——一个 Reasoner 生成 CoT，一个 Embedder 用 CoT 当 context 算 embedding。但 Reasoner 是预训练好的 MLLM 直接生成 CoT，**没和 Embedder 联合训练**，所以 CoT 可能和 retrieval 任务无关，甚至引入 hallucination 噪声；而且 CoT 只用文本，没用到 multimodal cues。

UME-R1: https://arxiv.org/abs/2511.00405
TTE: https://arxiv.org/abs/2510.05014

Embed-RL 的核心 thesis：**把 TTE 的 decoupled 思路保留，但用 RL 让 Reasoner 学会生成"对 Embedder 有用"的 CoT，并且 CoT 要 multimodal（带 bbox/keyframe/keyword 的可追溯证据）**。

---

## 2. 整体架构：Embedder-Guided RL (EG-RL)

架构图（Fig. 2b）的核心信息流：

```
Multimodal Query q
      │
      ▼
┌─────────────────┐
│  Reasoner (MLLM) │  ← RL 优化（GRPO）
│  生成 T-CoT      │
└─────────────────┘
      │ T-CoT = <thinking> bbox/keyframe/keyword </thinking>
      │         <rethink> ... </rethink>
      │         <answer> ... </answer>
      ▼
┌─────────────────────────────────────┐
│  Embedder (frozen MLLM)             │  ← contrastive 预训练后冻结
│  Input = [x_text, x_img, x_vid,     │
│          T-CoT, <emb>]              │
│  Output = hidden state of <emb>     │
└─────────────────────────────────────┘
      │
      ▼
  Embedding e_q
      │
      ▼
  Reward R_total → 反传给 Reasoner
```

关键设计：
1. **Embedder 完全 freeze**，提供稳定 reward signal。这是 RL 训练稳定性的关键——如果 Embedder 在变，reward 就是非平稳的，policy 很难收敛。
2. **T-CoT 拼接到原始输入**（Eq. 2）：$\mathcal{T}=[x_{\text{text}}, x_{\text{img}}, x_{\text{vid}}, \text{T-CoT}(x), \text{<emb>}]$。`<emb>` 是一个 special token，它的 last-layer hidden state 就是最终 embedding。
3. **多模态 evidence 回灌**：bbox 区域会被 crop 出来，keyframe 会被重新 extract，作为额外的 visual input 喂给 Embedder。这是 "thinking with images" 的关键，参考 GRIT [15]、DeepEyes [76] 的思路。
- GRIT: https://arxiv.org/abs/2505.15879
- DeepEyes: https://arxiv.org/abs/2505.14362

---

## 3. 数据构建（Section 3.2）

数据管线 "sampling-annotation-filtering-splitting"：

**Sources**：
- Image: MMEB-train [26] (classification/QA/retrieval/grounding)
- Video: LLaVA-Hound [71] (caption/QA/retrieval)
- VisDoc: ViDoRe [16] + VisRAG [66]
- ViDoRe: https://arxiv.org/abs/2407.01449
- VisRAG: https://arxiv.org/abs/2410.10594

**T-CoT 标注**：用 Qwen3-VL-8B 给所有 query-positive pair 标注，格式严格三段：
1. `<thinking>` 抽取 modality-specific cue：text 用 `text_keywords`（JSON 列表），image 用 `bbox_2d`（[x1,y1,x2,y2]），video 用 `key_frames`（1-based 帧索引）
2. `<rethink>` 基于 thinking 内容做 retrieval-relevant 的逻辑精炼
3. `<answer>` 总结核心 retrieval 信息

**Filtering**：用 Qwen3-VL-8B 判断 query_cot 和 pos_cot 是否"明显无关或冲突"，只保留 "No" 的样本。原始 2.22M → 保留 1.83M（retention 82.21%）。被滤掉的 20% 作为 RL hard examples（约 19K 样本，Table 6）。

直觉：这个过滤很关键，因为 contrastive learning 对噪声 pair 非常敏感——一个 false positive 会把整个 batch 的 InfoNCE 拉错方向。而保留的 hard examples 喂给 RL，是因为 RL 探索需要"有挑战性"的样本，太简单的样本无法区分不同 T-CoT 的质量。

---

## 4. 核心公式逐个拆解

### 4.1 InfoNCE loss (Eq. 1)

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\cos(h_{q_i}, h_{t_i^+})/\tau)}{\exp(\cos(h_{q_i}, h_{t_i^+})/\tau) + \sum_{t^-\in\mathcal{T}^-}\exp(\cos(h_{q_i}, h_{t^-})/\tau)}$$

变量含义：
- $N$：batch 内 query 数
- $h_{q_i}$：query $q_i$ 的 embedding（MLLM 最后一层 last token hidden state）
- $h_{t_i^+}$：正样本 embedding
- $h_{t^-}$：负样本 embedding（in-batch negatives）
- $\cos(\cdot,\cdot)$：cosine similarity
- $\tau$：temperature，控制分布锐度。$\tau$ 小则模型对 hard negative 更敏感，但训练不稳定；$\tau$ 大则平滑但区分度弱。

直觉：这是把"正样本应该比所有负样本都更相似"转成一个 softmax 分类问题——把正样本从 in-batch negatives 里"挑出来"。$-\log$ 的形式让正样本相似度越高、负样本相似度越低，loss 越小。

InfoNCE 原始 paper: https://arxiv.org/abs/1807.03748

### 4.2 Outcome Reward $\mathcal{R}_{\text{outcome}}$ (Eq. 3)

$$\mathcal{R}_{\text{outcome}}(o_i^q) = \text{Acc}_k(e_{q_i}, t_i^+) \cdot \Big(\text{sim}(e_q, e_{t_i^+}) - \mathbb{E}_\tau[\text{sim}(e_{q_i}, e_{t_j^-})]\Big)$$

变量：
- $o_i^q$：query $q_i$ 的某次 T-CoT rollout
- $e_{q_i} = \pi_e(q_i, o_i^q)$：Embedder 在 (query + T-CoT) 上算出的 embedding
- $e_{t_j} = \pi_e(t_j, o_j^t)$：target 的 embedding
- $\text{Acc}_k(e_{q_i}, t_i^+)$：top-k retrieval accuracy，正样本是否在 top-k（paper 里 $k=8$）
- $\mathbb{E}_\tau[\cdot]$：用 softmax 加权的 negative similarity 平均（Eq. 7）
- $\tau=0.5$（实验设置）

softmax 加权公式 (Eq. 7)：
$$\mathbb{E}_\tau[\text{sim}(e_{q_i}, e_{t_j^-})] = \frac{\sum_{j\neq i}\exp(\text{sim}/\tau)\cdot\text{sim}}{\sum_{j\neq i}\exp(\text{sim}/\tau)}$$

设计 intuition：
- **Acc_k 是门控**：如果正样本连 top-k 都进不去，reward 直接归零。这避免模型通过"把所有 similarity 整体拉高"来骗 margin。
- **margin 项是细化**：在 Acc_k=1 的前提下，进一步拉开正样本和 hard negative 的相似度差。softmax 加权让"最难的 negative"主导 reward——这是 hard negative mining 的 implicit 实现。
- **对称计算**：同样的公式反向算一遍（target 当 anchor，query 当 positive），强制双向对齐。这避免了 asymmetric embedding space 的退化。

### 4.3 Process Reward $\mathcal{R}_{\text{process}}$ (Eq. 4)

$$\mathcal{R}_{\text{process}}(o_i) = \begin{cases}1, & \text{if } \mathcal{D}(q_{\text{cot}}, \{c_{\text{cot}}^j\}_{j=1}^m) \in \mathcal{P} \\ 0, & \text{otherwise}\end{cases}$$

变量：
- $\mathcal{D}$：独立的预训练 VLM Discriminator（Qwen3-VL-8B）
- $q_{\text{cot}}$：query 的 T-CoT 输出
- $\{c_{\text{cot}}^j\}_{j=1}^m$：$m$ 个候选 target 的 T-CoT（正样本来自 query 的多次 rollout，负样本来自 in-batch）
- $\mathcal{P}$：shuffle 后 ground-truth positive 的索引集合
- $\mathcal{D}(\cdot,\cdot)$：输出最匹配 $q_{\text{cot}}$ 的候选索引

直觉：让一个独立的"裁判 VLM"做 listwise 选择——把 query 的 CoT 和一堆候选 target 的 CoT 放一起，问裁判"哪个 target CoT 和 query CoT 最匹配"。如果裁判挑中了真正的正样本，reward=1；否则 0。

为什么需要 process reward？因为 outcome reward 只看最终 embedding 相似度，**CoT 可以通过非 reasoning 的捷径"骗"过 embedding**（比如直接抄某些 token）。Process reward 强制 CoT 本身要对齐，让 reasoning trajectory 可追溯、可解释。这呼应了 TreeVGR [47] 的"joint localization + reasoning supervision"思路。
- TreeVGR: https://arxiv.org/abs/2507.07999

### 4.4 Total Reward (Eq. 5)

$$\mathcal{R}_{\text{total}} = \alpha\mathcal{R}_{\text{format}} + \beta\mathcal{R}_{\text{process}} + \gamma\mathcal{R}_{\text{outcome}}$$

权重：$\alpha=0.05, \beta=0.8, \gamma=0.2$（Sup. B）。

**有意思的观察**：process 权重（0.8）远大于 outcome（0.2）。直觉是——outcome reward 噪声大（依赖 in-batch sampling，rollout 间差异远大于 reward 自身增长，见 Sup. H.1），而 process reward 由独立 VLM 给出，更稳定。所以 paper 把"信号"押在 process 上，outcome 当辅助。这也解释了 ablation 里 outcome 只贡献 1.0 分，process 贡献 0.8 分但 video 上影响最大（52.1 → 51.3）。

### 4.5 GRPO objective (Eq. 6)

$$\mathcal{L}_{\text{grpo}} = \mathbb{E}_{q\sim S, \{o_i\}\sim\pi_{\theta_{\text{old}}}}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\bigg(\min(r_\theta(o_i)A_i, \text{clip}(r_\theta(o_i), 1-\epsilon, 1+\epsilon)A_i) - \beta\mathbb{D}_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}})\bigg)\Bigg]$$

变量：
- $S$：训练 sample 集合
- $G=8$：每个 query 采样的 rollout 数（group size）
- $\pi_{\theta_{\text{old}}}$：旧 policy（用于 importance sampling）
- $r_\theta(o_i) = \pi_\theta(o_i|q)/\pi_{\theta_{\text{old}}}(o_i|q)$：importance ratio
- $\epsilon=0.2$：clip 阈值（PPO 标配）
- $A_i = (r_i - \mu_r)/\sigma_r$：group-relative advantage，$\mu_r, \sigma_r$ 是 group 内 reward 的均值和标准差
- $\beta=0.01$：KL 系数
- $\pi_{\text{ref}}$：参考 policy（训练前 frozen）

直觉：GRPO [19] 相比 PPO 的核心简化是**用 group-relative baseline 替代 value function**。对每个 query 采 $G=8$ 个 T-CoT，reward 归一化成 advantage。这避免了训 critic 网络（省一半显存），同时 group 内 baseline 自然适应 query 难度。clip 项防 policy 跑太远，KL 项拉回 reference 防 collapse。
- DeepSeek-R1 / GRPO: https://arxiv.org/abs/2501.12948

---

## 5. T-CoT 的多模态结构（核心创新）

T-CoT 的三段式设计是这篇 paper 的灵魂。和传统纯文本 CoT 相比，它显式注入三种 modality-specific evidence：

| Modality | Evidence type | 输出形式 | 作用 |
|----------|--------------|---------|------|
| Text | keywords | `{"text_keywords": ["...", ...]}` | 锚定核心语义 |
| Image | bounding box | `{"bbox_2d": [x1,y1,x2,y2]}` 或 list | RoI 定位，crop 后回灌 |
| Video | keyframe indices | `{"key_frames": [1,5,9]}` (1-based) | 时序定位，重新 extract 帧 |

**为什么这对 embedding 有效？** 直觉有三层：

1. **Attention focusing**：MLLM 的 visual token 很多（一张图几百 token，一个视频上千 token），全局 attention 会被冗余区域稀释。bbox crop 把"重要区域"拎出来单独看，等于显式做了 RoI pooling 的可学习版本。

2. **Cross-modal alignment 的中间桥梁**：传统 embedding 直接从 raw input 跳到 single vector，跨度太大。T-CoT 把这个跳跃分解成 (raw → structured evidence → refined reasoning → final embedding)，每一步都有显式监督（process reward 监督 evidence alignment）。

3. **Decoupling 解锁 multimodal CoT**：UME-R1 不能加 bbox——因为 NTP loss 会让 bbox 坐标变成"预测下一个数字"的退化任务，和 contrastive loss 严重冲突。EG-RL 的 decoupling 让 Reasoner 自由生成 multimodal output，Embedder 只管用。

参考文献：Ground-R1 [6] 用 dual reward 做 grounded reasoning，BRPO [11] 用 IoU reward + visual token 机制，都启发了这种"reasoning with visual evidence"范式。
- Ground-R1: https://arxiv.org/abs/2505.20272
- BRPO: https://arxiv.org/abs/2505.23558

---

## 6. 实验结果深度分析

### 6.1 MMEB-V2 主结果（Table 1）

MMEB-V2 [41] 包含 78 个 task，覆盖 image (36)、video (18)、visdoc (24) 三大模态，9 个 meta-task。
- MMEB-V2: https://arxiv.org/abs/2507.04590

关键数据点（overall Hit@1 / NDCG@5）：

| Model | Image | Video | VisDoc | All |
|-------|-------|-------|--------|-----|
| UME-R1-7B | 71.3 | 47.5 | 67.1 | 64.5 |
| VLM2Vec-V2-7B | 68.1 | 36.4 | 69.3 | 60.6 |
| CAFe-7B | 67.6 | 42.4 | 63.9 | 60.1 |
| **Embed-RL-2B** | 69.2 | 52.1 | 74.1 | **66.8** |
| **Embed-RL-4B** | 70.1 | 53.0 | 74.7 | **68.1** |

观察：
1. **Embed-RL-4B 用 4B 参数打过 7B baseline 3.6 分**。这是 parameter efficiency 的强证据——decoupled + multimodal CoT 比 scale up 更有效。
2. **VisDoc OOD 暴涨**：Embed-RL-4B 在 OOD visdoc 上达 67.1，而 UME-R1-7B 只有 37.6，差 30 分。直觉：T-CoT 的 bbox 让模型对 unseen document layout 的鲁棒性大幅提升——它学会了"找关键区域"而不是记住特定 layout。
3. **Video 全线领先**：52.1/53.0 vs UME-R1 的 47.5。keyframe 机制让模型显式建模时序，而不是把视频当成 "image bag" 平均处理。

### 6.2 UVRB 视频检索（Table 2, 10, 13）

UVRB [20] 有 16 个数据集，从三个正交维度评估：
- **Tasks**: TXT (text→video), CMP (composed), VIS (visual→video)
- **Domains**: CG (coarse), FG (fine), LC (long-context)
- **Sub-domains**: S (spatial), T (temporal), PR (partially relevant)

- UVRB / GVE: https://arxiv.org/abs/2510.27571

Embed-RL-4B 的数据（Table 13）：
- AVG: 58.5
- CG: 60.7（第一），FG: 55.6（第一），LC: 86.1（第二）
- Sub-domain S: 87.9（第一）——bbox 对 spatial fine-grained 极其有效
- Sub-domain T: 46.0（第二）
- Sub-domain PR: 40.6

关键 insight：**bbox 让 spatial fine-grained（87.9）暴涨**。比如 CRB-S (spatial retrieval) 上 Embed-RL-4B 达到 89.9，比 GVE-7B (84.6) 高 5 分。这直接验证了"bbox crop 回灌"机制在 object/appearance 级检索上的威力。

但 **MS-TI/MS-TV（composed retrieval）上 Embed-RL 反而弱**（15.8/21.0，倒数）。直觉：composed query 需要"用 image 当 query 去 retrieve video"，这种 image-video 直接对齐任务，T-CoT 的 text keyword 提取帮不上忙，反而可能因为 reasoning overhead 干扰。

### 6.3 Ablation: Reward 组件（Table 3）

| 变体 | Image | Video | VisDoc | All | Δ |
|-----|-------|-------|--------|-----|---|
| Full | 69.2 | 52.1 | 74.1 | 66.8 | — |
| w/o EG-RL | 68.0 | 50.1 | 72.7 | 65.3 | -1.5 |
| w/o weighted negative | 68.9 | 51.7 | 73.9 | 66.5 | -0.3 |
| w/o process reward | 68.3 | 51.3 | 73.5 | 66.0 | -0.8 |
| w/o outcome reward | 68.1 | 51.2 | 73.1 | 65.8 | -1.0 |

观察：
1. **EG-RL 整体贡献 1.5 分**——RL 优化 Reasoner 的净收益。
2. **Process reward 对 video 影响最大**（52.1 → 51.3，-0.8）。Video 任务依赖 step-by-step 时序推理，process reward 强制 CoT 的逻辑链对齐，对长时序理解特别关键。
3. **Outcome reward 贡献 1.0 分**，主要在 visdoc（74.1 → 73.1）。VisDoc 的检索粒度细（paragraph/table/figure），margin reward 让模型精确区分 hard negative。
4. **Weighted negative 只贡献 0.3**，说明 softmax 加权 negative 的边际收益最小，但稳定性贡献大（防 reward hacking）。

### 6.4 Ablation: T-CoT 结构（Table 4）

| 变体 | All | Δ |
|-----|-----|---|
| Full | 66.8 | — |
| w/o reasoning（只保留 answer） | 65.5 | -1.3 |
| w/o multimodal cues（去掉 bbox/keyframe） | 65.8 | -1.0 |
| w/ raw input（完全不用 T-CoT） | 60.2 | **-6.6** |

**w/ raw input 直接崩 6.6 分**，其中 video 崩 8.4 分（52.1 → 43.7）。这是最强证据：**T-CoT 不是锦上添花，而是 embedding quality 的基石**。没有 reasoning 的 bridge，MLLM 的 hidden state 直接当 embedding 在复杂检索任务上严重退化。

### 6.5 判别能力分析（Fig. 4）

Fig. 4 量化了 RL 前后模型的"相似度区分度"：定义 $\Delta s = \text{sim}(\text{query}, \text{top1}) - \text{sim}(\text{query}, \text{top2})$，即正样本和最相似 hard negative 的相似度差。

RL 后的 radar plot 完全包住 RL 前——三模态所有数据集 $\Delta s$ 都变大。直觉：**RL 让模型把"正样本相似度"和"最像的负样本相似度"之间的 gap 拉开了**，这正是 embedding 区分能力的本质。

### 6.6 Traceable evidence count vs retrieval metric（Fig. 5）

非常有趣的现象：
- **Image/VisDoc**：RL 后 bbox 数量变多，检索指标上升。模型学会"多抓几个证据"提升 recall。
- **Video**：RL 后 keyframe 数量变少，检索指标上升。模型学会"focus 关键帧"而非"撒网"。

这反映不同模态的最优策略不同：image 是空间冗余的，多 bbox 提升覆盖；video 是时序冗余的，少 keyframe 提升精度。

---

## 7. 训练细节关键点（Sup. B）

**Contrastive 阶段**：
- Qwen3-VL-2B/4B-Instruct 作为 Embedder
- LoRA [21]：2B 用 r=64, α=128；4B 用 r=96, α=192
- Batch 512（2B）/256（4B），sub-batch 256/128（同数据集采样）
- LR 1e-4, cosine schedule, 2 epochs

对比：UME-R1 用 batch 1024，TTE 用 8192——Embed-RL 的训练 scale 明显更小，但效果更好。这说明 decoupled + RL 的 sample efficiency 远高于 joint training。

- LoRA: https://arxiv.org/abs/2106.09685

**RL 阶段**：
- Qwen3-VL-8B-Instruct 当 Reasoner（注意：Reasoner 比 Embedder 大！这是为了让 Reasoner 有足够容量生成高质量 T-CoT）
- GRPO: G=8, ε=0.2, β_KL=0.01
- Batch 256, LR 3e-6, 1 epoch
- Discriminator: 独立的 Qwen3-VL-8B（process reward 用）
- Acc_k 的 k=8（8 个 rollout 全部进 top-8 才给 outcome reward）

**Vision processing**：
- Image: min 128×32×32, max 768×32×32 pixels
- Video: min 128×32×32, max 300×32×32, total 300×32×32×8
- Frame sampling: FPS=2.0, min/max frames=8, frame_factor=2
- Bbox 坐标缩放到 0-1000 范围（相对坐标），crop 时转回原图绝对坐标

---

## 8. 关键 intuition 总结

1. **Gradient conflict 的根因**：contrastive loss 在 hidden state 几何上做"分类拉开"，NTP 做"分布拟合"，两者在 last-layer 表征上的最优解不重合。Decoupling 让两个 objective 在不同 module 上各司其职。

2. **Embedder 当 reward model 是"可验证 reward" 的迁移**：和 math reasoning 的 verifiable reward 类比——math 有 ground-truth answer 可验证，retrieval 有"正样本是否排第一"可验证。Embedder 训好后就是 retrieval 的"答案验证器"。

3. **Multimodal CoT 是 decoupling 的红利**：Reasoner 不受 NTP 约束，所以可以自由输出 bbox/keyframe 这种"非语言"结构。这开启了 "thinking with images" 的 embedding 版本。

4. **Process reward 防 reward hacking**：纯 outcome reward 会让 Reasoner 学捷径（比如直接输出 query 的某些 token 当 CoT，让 Embedder 通过 token overlap 拿高分）。Process reward 强制 CoT 本身要对齐，相当于给 reasoning trajectory 加了"可解释性约束"。

5. **Parameter asymmetry**：Reasoner (8B) > Embedder (2B/4B)。直觉：Reasoner 需要"理解力"生成结构化证据，Embedder 只需要"判别力"做相似度映射。生成比判别更难，所以 Reasoner 更大。

---

## 9. Limitations & 我的看法

Paper 自己承认：
1. Reward 权重 $\alpha, \beta, \gamma$ 经验设定，没自适应机制
2. 排除了 3 个 classification 数据集（HatefulMemes, N24News, VOC2007），因为 sub-batch contrastive 在小类别数下 false negative 严重——这是 contrastive learning 的经典坑
3. 没 hard negative mining、没 curriculum learning

我的额外观察：
1. **Reasoner 8B + Embedder 2B/4B 的"非对称"** 在工业部署上是双倍成本。Paper Sup. I 说 T-CoT 可以 offline 一次性生成缓存，但只对 target 侧成立——query 侧每次都要 Reasoner 推理，latency 仍然高。
2. **Process reward 依赖独立 VLM Discriminator**——又是一个 8B 模型，训练时显存压力大。可以用更小的 discriminator 或蒸馏替代。
3. **MS-TI/MS-TV 上的退化**提示 T-CoT 设计偏向 text-dominant 任务，对纯 image/video query 的 composed retrieval 不友好。可以设计 image-query 专属的 T-CoT template。

---

## 10. 相关工作脉络（供你按图索骥）

**Embedding 基础**：
- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP: https://arxiv.org/abs/2303.15343
- InfoNCE/CPC: https://arxiv.org/abs/1807.03748

**MLLM-based embedding**：
- VLM2Vec: https://arxiv.org/abs/2410.05160
- VLM2Vec-V2: https://arxiv.org/abs/2507.04590
- MM-Embed: https://arxiv.org/abs/2411.02571
- GME: https://arxiv.org/abs/2412.16855
- E5-V: https://arxiv.org/abs/2407.12580
- UniME: https://arxiv.org/abs/2504.17432
- CAFe: https://arxiv.org/abs/2503.19900
- LamRA: https://arxiv.org/abs/2406.12747 (近似)

**Generative embedding with reasoning**：
- UME-R1: https://arxiv.org/abs/2511.00405
- TTE: https://arxiv.org/abs/2510.05014

**RL for reasoning**：
- DeepSeek-R1 / GRPO: https://arxiv.org/abs/2501.12948
- GRIT (thinking with images): https://arxiv.org/abs/2505.15879
- Ground-R1: https://arxiv.org/abs/2505.20272
- DeepEyes: https://arxiv.org/abs/2505.14362
- TreeVGR: https://arxiv.org/abs/2507.07999

**Benchmark**：
- MMEB: https://arxiv.org/abs/2410.05160
- MMEB-V2: https://arxiv.org/abs/2507.04590
- UVRB / GVE: https://arxiv.org/abs/2510.27571
- ViDoRe: https://arxiv.org/abs/2407.01449
- VisRAG: https://arxiv.org/abs/2410.10594

---

这篇 paper 真正的 contribution 在于把 "verifiable reward" 的思路从 math/reasoning 迁移到 retrieval/embedding——Embedder 训好之后就是个"答案验证器"，CoT 质量好不好直接看 retrieval 表现。这个 framing 让 RL 在 embedding 任务上变得 tractable，而 decoupling 让 multimodal CoT（bbox/keyframe）首次成为可能。值得在你的 nanoGPT 后续系列或者 Eureka Labs 课程里作为一个 case study 讲——它把"reasoning model"和"embedding model"两个看似不相关的方向统一了。
