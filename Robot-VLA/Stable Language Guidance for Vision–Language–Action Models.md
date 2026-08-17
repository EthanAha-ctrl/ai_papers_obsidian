---
source_pdf: Stable Language Guidance for Vision–Language–Action Models.pdf
paper_sha256: 44203e4848d9f82213cbd83803a3ba1e150282d6d2723016313428d80d2aa7eb
processed_at: '2026-08-12T10:28:53-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话版本

**现在的 VLA 模型其实根本没在看 language instruction，它主要靠视觉"蒙"答案。这篇 paper 做了两件事：训练时让模型多见各种说法，推理时把视觉的"蒙"的部分减掉，只留 language 真正贡献的部分。**

---

## 现状：模型根本没在听你说话

想象你训练了一个 robot，你说 "put the wine bottle on top of the cabinet"，它做对了。你觉得它听懂了。但实际上呢？

你把 instruction 换成空白，什么都不说，它还是做对了。

你把 "wine bottle" 换成 "sealed container typically used for pouring grape-based drinks"，它就懵了。

这说明什么？**模型根本没在 parse 你的 language，它在靠视觉场景猜**。柜子上方有放东西的空间，酒瓶在桌上，"抓起来放上去" 这个 action 在训练数据里出现频率最高，所以模型 default 就做这个。Language 只是一个 noise，模型学会了 ignore 它。

这就是 paper 里说的 **"instruction blindness"** 和 **"modality collapse"**。

---

## 为什么会这样

两个原因，都很好理解：

### 原因一：视觉信号太"厚"了，language 太"薄"了

一张图进来，变成 256 个 token，每个 token 都携带边缘、颜色、深度、物体形状这些 dense 信息。一句 "put the wine bottle on top of the cabinet" 只有 10 个 token，其中 "the", "on top of" 这些 function word 不 carry 任何 task-relevant 信息。

训练时 backprop 算 gradient，视觉那边贡献了绝大部分 gradient magnitude，language 那边几乎被淹没了。结果就是模型的权重 $W_v$（视觉投影）长得很大，$W_l$（语言投影）长得很小。

你想想，如果训练中 99% 的 gradient signal 来自视觉，1% 来自语言，那模型当然会 optimize 成"主要靠视觉，language 随便看看"的样子。

### 原因二：训练数据里的 language 表述太单一了

一个 task "把酒瓶放柜顶上"，训练集里可能就一种说法："Put the wine bottle on top of the cabinet"。模型把这个具体 string 记住了，但没学到背后的 intent。

这就好比你只见过 "open the door" 这个说法，突然有人说 "could you please get that door open for me"，你就不知道该怎么办了。你 memorize 了 surface form，没 generalize 到 semantic intent。

---

## Paper 的解法：RSS 框架

两个 component，一个治训练，一个治推理。

### 治训练：MCSI（Monte Carlo Syntactic Integration）

思路特别直白：**既然训练数据里 language 表述太单一，那就让 LLM 帮你多写几个版本。**

你有一个 instruction "Put the wine bottle on top of the cabinet"。你让 Qwen2.5-VL 当 teacher，生成 $K$ 个 paraphrase：

- "Move the wine bottle onto the cabinet top"
- "Place the wine bottle on the upper surface of the cabinet"
- "Transport the wine bottle to the cabinet's top shelf"
- "Set down the wine bottle on top of the standing storage furniture"
- ...

然后训练的时候，对于同一个 $(o, a)$ pair，你让模型在这 $K$ 个 instruction 上都做 MLE loss，取平均。

**效果**：模型被迫学会"不管你怎么说这个 task，只要 intent 一样，action 就应该一样"。这就把 surface form 的 noise marginalized 掉了。

直觉上，这等价于让模型的 language encoder 把 "wine bottle"、"sealed container"、"grape-based drinks container" 这些不同的 string 都 map 到 embedding space 里相近的区域。模型不再 memorize 具体 wording，而是学到 semantic intent。

这个 trick 在 NLP 领域不新鲜（就是 data augmentation 嘛），但 paper 的 contribution 在于用 **multimodal LLM** 当 teacher —— 它能保证 paraphrase 在当前 visual scene 下是 grounding 的，不会生成语义漂移的 rewriting。

### 治推理：RAS（Residual Affordance Steering）

这个是 paper 的核心 contribution，也是最 clever 的部分。

#### CFG 回顾

Diffusion model 领域有个经典 trick 叫 Classifier-Free Guidance (CFG, https://arxiv.org/abs/2207.12598)。公式是：

$$\tilde{\epsilon} = (1+\omega) \cdot \epsilon_\theta(x_t, c) - \omega \cdot \epsilon_\theta(x_t, \emptyset)$$

意思是：conditional prediction 减去 unconditional prediction，把差值放大 $\omega$ 倍。这样让生成结果更贴近 condition $c$。代价是 diversity 下降。

#### RAS 的 reinterpretation

Paper 说：在 VLA 里，unconditional pass $s(a|o, \emptyset)$ 其实就是 **"纯靠视觉猜的 action"**，也就是 robot 的 "visual instinct" / "affordance prior"。

conditional pass $s(a|o, l)$ 是 "看了视觉又看了 language 的 action"。

两者相减：

$$\Delta_{\text{sem}} = s(a|o, l) - s(a|o, \emptyset)$$

这个 residual 是什么？是 **language 单独贡献的部分**，visual bias 被减掉了。

然后 RAS 的公式：

$$\tilde{\pi}(a|o,l) \propto \exp\left(s(a|o,\emptyset) + \gamma \cdot \Delta_{\text{sem}}\right)$$

把 language 贡献放大 $\gamma$ 倍（$\gamma > 1$），visual 部分保持原样。

#### 为什么这有效

paper 在 Appendix A 给了个 first-order 分析。假设 logit 可以线性分解：

$$S(a|o,l) = W_v^\top \phi(o) + W_l^\top \psi(l)$$

视觉权重 $W_v$ 很大，语言权重 $W_l$ 很小（因为训练时 gradient 被视觉主导）。

当你做 $S(a|o,l) - S(a|o,\emptyset)$ 时，$W_v^\top \phi(o)$ 被消掉了（因为 $S(a|o,\emptyset) \approx W_v^\top \phi(o)$，即 null-text 时 language embedding ≈ 0），剩下来的就是 $W_l^\top \psi(l)$，即纯语言贡献。

然后你把它放大 $\gamma$ 倍，等价于人工把 $W_l$ 变成 $\gamma \cdot W_l$。原本 $\|W_v\| \gg \|W_l\|$，现在 $\|W_v\| \approx \gamma \|W_l\|$，modality balance 被恢复了。

#### 与 CFG 的关键区别

CFG 在 diffusion model 里是 "quality booster" —— 让生成更逼真、更贴近 prompt，代价是多样性下降。

RAS 在 VLA 里是 "bias suppressor" —— 把 visual 模式的 shortcut 给压制住，让模型不得不认真听 language。

在 generative model 里，"diversity" 是好的（你想要不同的生成结果）。在 control 里，"diversity" 不重要（一个 state 对应一个正确 action），重要的是 "accuracy to intent"。所以 RAS 可以放心地 amplify language signal 而不用担心副作用——除了一个：当 language 本身是 garbage 的时候（比如 mask 80%），amplify garbage 会让事情更糟。这就是 paper Limitations 部分提到的 trade-off。

---

## 实验结果讲了什么

### Table 1：Destructive perturbation

把 instruction 各种破坏：空白、乱序、mask 80% 等等。

$\pi_0$ baseline 在 Blank（完全没 instruction）时 success rate 只有 25.2%。加 RAS 后变 63.4%。为什么？因为 RAS 显式建模了 $s(a|o,\emptyset)$ 这个 affordance prior，即使没 instruction，模型也能靠视觉本能做一个 reasonable action，而不是 random jitter。

Mask 80% 时 baseline 只有 7.8%，加 MCSI 后变 52.8%，加 RAS+MCSI 变 69.9%。MCSI 的作用是训练时见过各种 paraphrase，所以即使 80% 的 token 被 mask，剩余 20% 的 token 依然能 trigger 正确的 intent representation。

### Table 2：Obfuscated reinterpretation

保持 intent 但换说法。最难的是 R3（reasoning chain）："Move the bottle over the cabinet, then release it once it is stable on top."

Baseline 只有 28.4%，加 RAS+MCSI 后变 80.0%。这种 multi-step reasoning instruction 对 autoregressive VLA 是很 challenging 的，因为模型需要 parse "先 move over，再 release" 的顺序关系。MCSI 训练时如果见过类似 reasoning chain 的 paraphrase，encoder 就能学到这种结构。

### Table 4：OOD transfer

训练时没见过的 object-goal composition。Baseline 在 10-shot 下只有 27%，而且这 27% 还是 overfit 到一个 task 上，另一个完全失败。RAS+MCSI 在两个 task 上都成功。

这说明 RSS 改善的是 **generalization** 而非 **memorization**。模型不再靠 "见过这个 object 就记住这个 action"，而是 "这个 object 的 visual embedding + 这个 goal 的 language embedding → compose 出 action"。

### Table 11：Steering coefficient ablation

$\gamma$ 太大（3.0）时，当 instruction 是 clean 的，Origin SR 反而从 92 跌到 82。因为过强 steering 让模型 over-condition 在 language 上，但 language 是 clean 的时候，正常 inference 已经够好了，over-amplify 反而引入 numerical instability。

$\gamma$ 太大时，Mask 80% 的 instruction（大部分是 garbage）被 amplify 5 倍，模型执行 random action，SR 跌到 1.9%。

所以 $\gamma$ 的 optimal 值在 1.25-1.5 之间，是一个 "enough to suppress visual bias, not too much to amplify language noise" 的 sweet spot。

---

## 整个 Paper 的 Big Picture

VLA 模型的根本问题是 **modality imbalance**：视觉太强，语言太弱，训练时 gradient 被视觉主导，推理时模型默认走视觉 shortcut。

RSS 的解法分两层：

**训练层（MCSI）**：用 LLM 生成 dense paraphrase，扩充 language 的 manifold coverage，让 $W_l$ 在训练时收到更多 gradient，长得更大。

**推理层（RAS）**：把 visual affordance prior 显式减掉，只留 language 的纯贡献，再放大 $\gamma$ 倍，人工恢复 modality balance。

两层攻击同一个问题的不同侧面：MCSI 让 model 学得好，RAS 让 model 用得好。

这个 idea 的 beauty 在于它 **不需要 retrain from scratch**，不需要改 architecture，只需要在 fine-tuning 时加 paraphrase、在 inference 时改 logit 计算。工程上极简，理论上 principled。

---

## 我的直觉解读

这篇 paper 让我想到一个类比：VLA 模型就像一个学生，视觉是数学（练得多、信号强），language 是英语（练得少、信号弱）。考试的时候，这个学生主要靠数学拉分，英语部分基本靠蒙。

MCSI 是给他做英语真题集，让他多见各种题型。
RAS 是考试时告诉他："先把数学答案算出来，然后看你英语题目里的关键词，如果关键词和数学答案方向一致，就放大英语的权重。"

最终效果：这个学生英语部分从蒙变成真答题了，总成绩显著提升。

更深层地看，这篇 paper 点出了一个所有 multi-modal foundation model 都会面临的问题：**dense modality 会 dominate sparse modality，不是在 capacity 层面，而是在 optimization dynamics 层面**。Gradient flow 是 proportional to signal density 的，所以只要 visual token 数量 >> language token 数量，这个 imbalance 就会存在。

RAS 的妙处在于它把 CFG 这个 generative model 的 trick 重新 frame 成了一个 **causal intervention**：$s(a|o,\emptyset)$ 是 counterfactual（"如果没有 language 会怎样"），$\Delta_{\text{sem}}$ 是 Average Treatment Effect of language。这在 causal inference literature (Pearl, 2009, https://en.wikipedia.org/wiki/Judea_Pearl) 里是标准做法，只是被重新 apply 到了 VLA 的 logit space。

---

## References

- Paper 本身: 你给我看的这个
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- CFG (Ho & Salimans): https://arxiv.org/abs/2207.12598
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus audit: https://arxiv.org/abs/2510.13626
- LIBERO-Pro audit: https://arxiv.org/abs/2510.03827
- OpenVLA: https://arxiv.org/abs/2406.09246
- RDT-1B (alternating attention solution): https://arxiv.org/abs/2410.07864
- Qwen2.5-VL (teacher): https://arxiv.org/abs/2502.13923
- $\mathcal{E}_0$ (same group, continuized discrete diffusion): https://arxiv.org/abs/2511.21542
- Causal inference (Pearl): https://en.wikipedia.org/wiki/The_Book_of_Why

---

# Stable Language Guidance for Vision-Language-Action Models 深度解析

## 一、Paper 的核心问题：Modality Collapse 的根因

Karpathy 你这个 paper 抓住了一个非常重要且未被充分讨论的问题：VLA 模型在结构上就**注定**会被视觉信号主导。让我从信息论和优化几何两个角度 build intuition。

### 1.1 信号密度的不对称

视觉输入 $o$ 由 $\{I_t^1, \ldots, I_t^n\}$（多视角图像）和 proprioceptive state $q_t$ 组成。一张 224×224 RGB 图像经过 ViT/SigLIP tokenizer 后产生 ~196-256 个 dense tokens，每个 token 携带 edge、texture、depth、object geometry 等 high-frequency 信息。而语言指令 $l$ 通常只有 5-20 个 tokens，且大部分是 function words（"the", "on top of"）。

在反向传播时，gradient flow $\nabla_\theta \mathcal{L}$ 通过 cross-attention 注入 transformer backbone。视觉 token 的 Jacobian 矩阵的 Frobenius norm $\|\partial \mathcal{L} / \partial \phi(o)\|_F$ 在数量级上压倒 $\|\partial \mathcal{L} / \partial \psi(l)\|_F$。这就是 paper 中 Assumption 1 的物理基础：$\|W_v\| \gg \|W_l\|$。

这种 asymmetric gradient flow 导致的后果是 **mode collapse to visual prior** —— 模型学到的本质上是 $\pi(a|o)$，而把 $l$ 当作无关变量。这在 LIBERO-Plus audit (Fei et al., 2025, https://arxiv.org/abs/2510.13626) 和 LIBERO-Pro audit (Zhou et al., 2025, https://arxiv.org/abs/2510.03827) 中被独立验证。

### 1.2 Manifold Sparsity 的几何视角

训练数据 $\mathcal{D}$ 覆盖的 $p(l|z)$ support 极其稀疏。对于 intent $z=$ "把酒瓶放柜顶上"，训练集可能只有一两个具体表述："Put the wine bottle on top of the cabinet"。这相当于在 $\mathbb{R}^d$ 的语言 embedding manifold 上只采样了几个 isolated points。

MLE 目标 $\max_\theta \mathbb{E}[\log \pi_\theta(a|o,l)]$ 在这些 isolated points 上 overfit 表面 syntax 而非 latent intent $z$。当测试时遇到 $l_j = $ "Move the wine bottle onto the cabinet top"，虽然 $p(l_i|z) > 0$ 和 $p(l_j|z) > 0$，但模型在 $l_j$ 处的 likelihood 接近 0。

## 二、方法 RSS 详解

### 2.1 Monte Carlo Syntactic Integration (MCSI)

#### Intuition
真正要学的是 $p(a|o,z)$，但 $z$ 是 latent variable，无法直接 observe。可观测的是 $l \sim p(l|z)$。根据 marginalization:

$$p(a|o,z) = \int p(a|o,l) \, p(l|z) \, dl \quad \text{(Eq 2)}$$

这个积分 intractable，因为 $p(l|z)$ 不可解析。**关键 insight**：用一个 high-capacity LLM (Qwen2.5-VL, https://arxiv.org/abs/2502.13923) 作为 Oracle Teacher 来近似 $p(l|z)$ 的 sampling。

具体做法：以原 instruction $l_{\text{orig}}$ 为 seed，让 LLM 生成 $K$ 个 paraphrases $\{l_1, \ldots, l_K\} = \mathcal{N}(\bar{l}_{\text{orig}})$，这些 paraphrases 覆盖同义词替换、句式重写、冗余描述、推理链表达等不同 surface form。

#### Expected Semantic Loss

$$\mathcal{L}_{\text{RSS}} = \mathbb{E}_{(o,a) \sim \mathcal{D}} \left[ \frac{1}{K} \sum_{k=1}^{K} -\log \pi_\theta(a | o, l_k) \right] \quad \text{(Eq 3)}$$

**变量解析**：
- $(o, a) \sim \mathcal{D}$：从 demonstration dataset 采样的 observation-action pair
- $K$：paraphrase 数量，下标 $k \in \{1, \ldots, K\}$ 索引
- $l_k$：第 $k$ 个 paraphrase
- $\pi_\theta$：参数为 $\theta$ 的 policy

这个 loss 的本质：迫使 encoder 把 $\{l_1, \ldots, l_K\}$ 这 $K$ 个表面不同但 intent 相同的 instruction 映射到 latent space 的同一区域。这等价于最小化 conditional entropy $H(A|L)$ 在 syntactic variation 下的期望，即 $\min H(A | L_{\text{syntax}})$ 同时 $\max I(A; Z | O)$。

#### 与 Data Augmentation 的区别
标准 NLP data augmentation (back-translation, EDA) 是 domain-agnostic 的。MCSI 用 multimodal LLM 作为 teacher，能生成视觉-grounded 的 paraphrase —— 例如对 "pick up the red mug" 生成 "grasp the crimson cup" 而不是语义漂移的 "obtain the ruby container"。Qwen2.5-VL 的视觉 grounding 能力保证了 paraphrase 在 visual scene 下依然 interpretable。

#### 训练开销
论文用 batch size 32，30K steps，每个样本生成 $K$ 个 paraphrase（推断 $K=4-8$）。这意味着 effective batch size 在语言维度上 ×$K$。Qwen2.5-VL 的 inference 是一次性 offline 完成的，所以训练时只是 lookup 预生成的 paraphrase set。

### 2.2 Residual Affordance Steering (RAS)

#### 与 CFG 的本质区别

CFG (Ho & Salimans, 2022, https://arxiv.org/abs/2207.12598) 在 diffusion model 中公式为:
$$\tilde{\epsilon}_\theta(x_t, c) = (1+\omega)\epsilon_\theta(x_t, c) - \omega \epsilon_\theta(x_t, \emptyset)$$

其中 $\omega$ 是 guidance scale，$c$ 是 condition。CFG 的目标是 **enhance fidelity** —— 让生成更靠近 condition，代价是 diversity 下降。

RSS 在 VLA 中的公式:
$$\tilde{\pi}(a|o,l) \propto \exp\left(s(a|o,\emptyset) + \gamma \cdot \Delta_{\text{sem}}(a,o,l)\right) \quad \text{(Eq 5)}$$

其中:
$$\Delta_{\text{sem}}(a,o,l) = s(a|o,l) - s(a|o,\emptyset) \quad \text{(Eq 4)}$$

**变量解析**:
- $s(a|o,l) \in \mathbb{R}$：conditional logit score，给定 observation $o$ 和 instruction $l$ 时 action $a$ 的 logit
- $s(a|o,\emptyset)$：unconditional logit score，instruction 被 mask 掉
- $\gamma > 1$：steering coefficient，控制 semantic signal 的放大倍数
- $\Delta_{\text{sem}}$：纯语义残差，visual bias 被减去后的部分

**核心 insight**：$s(a|o,\emptyset)$ 在 paper 中被 reinterpret 为 **Base Affordance Distribution** —— 即视觉本能，比如"抓最近的物体"。CFG 在生成模型里把 null condition 当作质量基线，RSS 把它当作 **visual bias 的显式建模**。

#### 为什么减法有效：Proposition 1 证明解读

paper 在 Appendix A 给出了 Proposition 1 的证明。让我重新 build intuition。

假设 logit 可线性近似:
$$S(a|o,l) = W_v^\top \phi(o) + W_l^\top \psi(l) + \epsilon \quad \text{(Eq 7)}$$

**变量**:
- $\phi(o) \in \mathbb{R}^d$：视觉 embedding
- $\psi(l) \in \mathbb{R}^d$：语言 embedding
- $W_v, W_l \in \mathbb{R}^d$：modality-specific projection weights
- $\epsilon$：高阶交互项 + bias，first-order 分析忽略

**Assumption 1** (Visual Dominance): $\|W_v\| \gg \|W_l\|$，因为视觉梯度主导训练。

**Assumption 2** (Null-Text Baseline): $\psi(\emptyset) \approx \mathbf{0}$，因此:
$$S(a|o,\emptyset) \approx W_v^\top \phi(o) \quad \text{(Eq 8)}$$

Residual:
$$\Delta_{\text{sem}} = S(a|o,l) - S(a|o,\emptyset) \approx W_l^\top \psi(l) \quad \text{(Eq 14)}$$

视觉项 $W_v^\top \phi(o)$ 在减法中**完全 cancelled out**。

Steered logit:
$$\tilde{S}(a) = S(a|o,\emptyset) + \gamma \cdot \Delta_{\text{sem}} \approx W_v^\top \phi(o) + \gamma W_l^\top \psi(l) \quad \text{(Eq 16)}$$

**Signal-to-Noise Ratio**:
$$\text{SNR}_{\text{std}} = \frac{|W_l^\top \psi(l)|}{|W_v^\top \phi(o)|} \approx 0 \quad \text{(Eq 17)}$$

$$\text{SNR}_{\text{rss}} = \gamma \cdot \text{SNR}_{\text{std}} \quad \text{(Eq 18)}$$

通过设 $\gamma \gg 1$，等价于 synthesizing $\tilde{W}_l = \gamma W_l$，**artificially restore modality balance**。这是一个非常优雅的 inference-time fix —— 不需要 retrain，只是改 inference 时 logit 的 linear combination。

#### 为什么 RAS 是 Bias Suppressor 而非 Quality Booster

在 generative diffusion 中，$\omega > 1$ 让 sample 更靠近 condition，但同时降低 sample diversity，可能产生 mode collapse。在 VLA 控制里，"diversity" 不是 virtue —— 一个 $(o,l)$ 对应一个正确的 $a$。但 "over-conditioning" 是问题：如果 $l$ 是 noisy 的（mask 80% 的 token），过强的 condition 会让模型执行无意义 action。

RAS 的 $\gamma$ 起到一个 **gate** 的作用：当 $\Delta_{\text{sem}}$ 大（语言信号强且 unambiguous），放大它；当 $\Delta_{\text{sem}}$ 小（语言被破坏），模型自动 fallback 到 $s(a|o,\emptyset)$ 这个 visual affordance。这正是 paper Limitations 部分提到的 conservative behavior —— 当 instruction 极其模糊时，RAS 反而会 hesitate。

## 三、实验数据深度解读

### 3.1 Destructive Instruction Overwriting (Table 1)

针对 $\pi_0$ baseline (Black et al., 2024, https://arxiv.org/abs/2410.24164):

| Setting | Origin | Blank | M4 | M8 | Simple | Avg | Δ |
|---|---|---|---|---|---|---|---|
| base | 94.15 | 25.20 | 42.50 | 7.80 | 26.25 | 52.37 | - |
| +RAS | 90.65 | 63.40 | 55.05 | 17.95 | 62.50 | 64.46 | +12.09 |
| +MCSI | 94.55 | 41.85 | 77.75 | 52.80 | 39.85 | 71.77 | +19.40 |
| +RAS&MCSI | 93.35 | 69.65 | 85.85 | 69.90 | 64.90 | 82.22 | +29.85 |

**关键观察**:
1. **Blank SR**: base 25.20 → +RAS 63.40 → +RAS&MCSI 69.65。Blank 意味着 instruction 完全为空，base 模型完全靠 visual prior，频繁失败。+RAS 把 $s(a|o,\emptyset)$ 显式建模为 affordance，因此即使 instruction 为空，模型依然能产生 reasonable action。这就是 RAS 的 bias suppression 在极端条件下的表现。

2. **M8 (mask rate=0.8)**: base 7.80 → +MCSI 52.80 → +RAS&MCSI 69.90。这是 9× 的提升。MCSI 在这里起决定性作用 —— 训练时见过大量 paraphrase，使模型对部分 mask 的 instruction 依然能提取 intent。

3. **Simple ("do something")**: base 26.25 → +RAS 62.50。RAS 单独就能从 26 提到 62，说明 visual affordance 本身能 cover 大部分 LIBERO 任务。这暴露了 LIBERO 的一个特点：任务多样性低，视觉 prior 本身就接近正确答案。

4. **Origin SR (无 perturbation)**: +RAS 90.65 < base 94.15，**轻微 regression**。这是 bias suppression 的代价 —— 当语言完全清晰时，减去 visual prior 反而损失了一些 effective information。这个 trade-off 在 paper 中没有充分讨论，但 Table 11 ablation 显示 SC=1.25 时 Origin SR 恢复到 92.45，说明 $\gamma$ 可以调节这个 trade-off。

### 3.2 Obfuscated Instruction Reinterpretation (Table 2)

针对 $\pi_0$:

| Variant | R0 | R1 | R2 | R3 | R4 | Avg | Δ |
|---|---|---|---|---|---|---|---|
| base | 91.4 | 55.8 | 7.4 | 28.4 | 42.4 | 45.08 | - |
| +RAS | 90.0 | 59.0 | 11.4 | 40.2 | 46.0 | 49.32 | +4.24 |
| +MCSI | 92.6 | 83.6 | 28.0 | 71.2 | 58.6 | 66.80 | +21.72 |
| +RAS&MCSI | 88.4 | 85.4 | 26.8 | 80.0 | 47.0 | 65.52 | +20.44 |

**关键观察**:
1. **R2 (Common Sense)**: base 7.4 → +MCSI 28.0。R2 把 "wine bottle" 换成 "sealed container typically used for pouring grape-based drinks"，这种 abstraction 对 base 模型是灾难性的。MCSI 训练时见过类似 paraphrase，因此能 ground 这种描述。但即使 +MCSI 也只有 28，说明 commonsense reasoning 仍是 bottleneck。

2. **R3 (Reasoning Chain)**: base 28.4 → +RAS&MCSI 80.0。R3 的例子："Move the bottle over the cabinet, then release it once it is stable on top." 这种 multi-step reasoning chain 对 autoregressive VLA 是 challenging 的。+RAS&MCSI 达到 80，说明 RSS 框架对 reasoning-style instruction 显著 robust。

3. **+MCSI 在 R3 上单独 71.2 vs +RAS&MCSI 80.0**：RAS 在 reasoning chain 上额外提供 +8.8 的 boost。reasoning chain 通常包含 visual landmarks ("over the cabinet", "stable on top")，RAS 的 visual prior subtraction 帮助模型 focus 在这些 landmark 上。

### 3.3 OOD Semantic Transfer (Table 4)

针对 $\pi_{0.5}$ (Intelligence et al., 2025, https://arxiv.org/abs/2504.16054):

| Setting | 10-shot | 100-shot | 1000-shot | Avg |
|---|---|---|---|---|
| base | 27.0 | 31.0 | 91.0 | 49.67 |
| +RAS | 17.0 | 29.0 | 98.0 | 48.00 |
| +MCSI | 28.0 | 42.0 | 97.0 | 55.67 |
| +RAS&MCSI | 31.0 | 31.0 | 97.0 | 53.00 |

**关键观察**:
1. **10-shot +RAS 反而下降 27→17**：这验证了 Limitations 部分 —— RAS 在 instruction 信号弱且 OOD 时，bias suppression 让模型 over-conservative。base 模型会 hallucinate 一个 action，RAS 抑制了这种 hallucination 但没有替代方案。

2. **1000-shot base 91.0 vs +RAS&MCSI 97.0**：高 shot 下 base 已经接近 ceiling。但 paper 指出 base 的 91 来自 overfit 到一个 held-out task，另一个完全失败；+RAS&MCSI 在两个 task 上都成功。这说明 RSS 改善的是 **generalization** 而非 memorization。

3. **MCSI 在 OOD 上主导**：+MCSI 55.67 vs +RAS&MCSI 53.00。MCSI 通过 dense syntactic coverage 让 model 学到 object-goal composition 而非 rote pattern，这在 OOD 上至关重要。

### 3.4 Ablation: Steering Coefficient (Table 11)

针对 $\pi_0$+RAS:

| SC | Origin | Blank | M8 | Simple | Avg |
|---|---|---|---|---|---|
| 1.25 | 92.45 | 63.25 | 24.15 | 63.40 | 67.35 |
| 1.5 | 90.65 | 63.40 | 17.95 | 62.50 | 64.46 |
| 2.0 | 89.50 | 66.15 | 9.00 | 62.05 | 60.20 |
| 3.0 | 81.95 | 64.00 | 1.90 | 58.80 | 51.08 |

**关键观察**:
1. **SC=3.0 时 Origin 从 92.45 跌到 81.95**：过强 steering 让模型过度依赖语言信号，但语言本身是 clean 的，过度 amplify 反而引入 numerical instability 或 saturation。

2. **M8 (mask 80%) 在 SC=3.0 时只有 1.90**：mask 80% 的 instruction 是 noisy 的，SC=3.0 放大这个 noise 5×，模型执行 random action。

3. **Blank SR 在 SC=2.0 反而升到 66.15**：Blank 时 $\Delta_{\text{sem}} = s(a|o,l_{\text{blank}}) - s(a|o,\emptyset) \approx 0$，所以 $\gamma$ 的影响理论上 neutral。实际微小波动可能是 numerical effect。

**Optimal SC**: $\pi_0$ 上 1.25，$\pi_{0.5}$ 上 1.25-1.5。$\pi_{0.5}$ 比 $\pi_0$ 更 robust，因为它的 base language grounding 更强，不需要太强的 steering。

### 3.5 Denoising Steps Ablation (Table 12, 13)

针对 $\pi_0$+RAS (SC=1.5):

| Steps | Origin | Blank | M8 | Avg |
|---|---|---|---|---|
| 5 | 93.55 | 66.10 | 18.65 | 66.59 |
| 10 | 90.65 | 63.40 | 17.95 | 64.46 |
| 15 | 90.60 | 63.50 | 16.90 | 63.57 |

针对 $\pi_{0.5}$+RAS (SC=1.25):

| Steps | Origin | Blank | Avg |
|---|---|---|---|
| 5 | 96.90 | 69.90 | 84.47 |
| 15 | 97.40 | 69.35 | 84.37 |
| 20 | 96.85 | 70.30 | 84.32 |

**关键观察**:
1. **Steps 对 $\pi_0$ 影响显著但非 monotonic**：5 steps 反而比 10/15 steps 好。Diffusion-based policy 的 denoising steps 主要 refine low-level action trajectory，对 semantic grounding 贡献有限。更多 steps 可能让 action distribution sharpen 到 visual mode，反而损害 language steering。

2. **$\pi_{0.5}$ 几乎不敏感**：5/15/20 steps 的 Avg SR 在 84.3-84.5 之间。这证实 paper 的观察：stronger base model 对 inference hyperparameter 更 robust。

## 四、与 Related Work 的 Positioning

### 4.1 与 RDT-1B 的对比

RDT-1B (Liu et al., 2024, https://arxiv.org/abs/2410.07864) 通过 **alternating cross-attention** 在 successive layers 交替注入 image 和 text tokens，从 architecture 上防止 text overshadowing。这是 **structural solution**。

RSS 是 **algorithmic solution** —— 不改 architecture，通过 inference-time logit manipulation 实现 modality balance。两者可以叠加：RDT-1B 的 alternating attention 减轻 training 时的 gradient imbalance，RSS 在 inference 时进一步 suppress visual bias。

### 4.2 与 OpenVLA-OFT 的对比

OpenVLA-OFT (Kim et al., 2025, https://arxiv.org/abs/2502.19645) 优化 continuous action tuning，关注 **action representation** 而非 language grounding。Orthogonal direction。

### 4.3 与 CoT-VLA / OneTwoVLA 的对比

CoT-VLA (Zhao et al., 2025, https://arxiv.org/abs/2503.22030) 和 OneTwoVLA (Lin et al., 2025, https://arxiv.org/abs/2505.11917) 通过 chain-of-thought reasoning 增强 planning。它们解决的是 **long-horizon planning**，RSS 解决的是 **instruction grounding**。CoT-VLA 的 reasoning chain 可以被 R3 perturbation 破坏，RSS 能 robustify 它。

### 4.4 与 $\mathcal{E}_0$ 的对比

$\mathcal{E}_0$ (Zhan et al., 2025, 同一作者组, https://arxiv.org/abs/2511.21542) 用 continuized discrete diffusion 增强 generalization 和 fine-grained control。RSS 与 $\mathcal{E}_0$ 的结合点：$\mathcal{E}_0$ 改进 action generation 的 continuous dynamics，RSS 改进 condition steering。两者在 diffusion formulation 上兼容。

## 五、Limitations 的深度分析

Paper 提到的 limitation：**vague instruction 时的 conservative behavior**。这是 bias suppression 的必然代价。

**Intuition**: Base VLA 在 vague instruction 下会 hallucinate 一个 frequent action（"grab nearest object"）。这是 unsafe 但 functional。RSS 在 vague instruction 下 $\Delta_{\text{sem}} \approx 0$，$\tilde{\pi}(a|o,l) \approx \pi(a|o,\emptyset)$，模型会 wait 或执行 affordance-driven action。这更 safe 但可能被 perceive 为 "stuck"。

**Real-world implication**: 在 human-robot interaction 中，如果用户说 "do something"，base 模型会行动（可能错误），RSS 模型会 wait for clarification。前者在 demo 中 impressive，后者在 deployment 中 safer。

## 六、Missing Discussion 与潜在 Extension

### 6.1 Multi-modal Steering
Paper 只 steering 语言 vs 视觉。但 VLA 还有 proprioception $q_t$。理论上可以:
$$\Delta_{\text{proprio}} = s(a|o,l) - s(a|o_{\setminus q}, l)$$

提取 proprioception 的纯贡献。这对 joint-limit-aware control 可能有帮助。

### 6.2 Negative Steering for Adversarial Defense
当前 $\gamma > 1$ amplify semantic signal。如果检测到 instruction 是 adversarial（例如 ChatGPT-5.2 生成的 distractor），可以用 $\gamma < 1$ **suppress** semantic signal，让模型 fallback 到 visual affordance。这需要 instruction quality estimator。

### 6.3 MCSI 的 Teacher Choice
Paper 用 Qwen2.5-VL 作为 teacher。如果 teacher 本身有 bias（例如偏好某种句式），MCSI 会 amplify 这个 bias。Future work 可以用多个 teacher ensemble，或用 retrieval-augmented paraphrase generation 从 web corpus 采样真实人类表述。

### 6.4 Theory 的高阶项
Proposition 1 忽略 $\epsilon$（高阶交互项）。实际上 $W_{vl}^\top (\phi(o) \otimes \psi(l))$ 这种 bilinear term 在 transformer attention 中是 dominant 的。RSS 对 bilinear term 的影响需要更精细的分析。如果 $\Delta_{\text{sem}}$ 包含 visual-linguistic interaction，减法可能 over-correct。

### 6.5 与 RLHF 的结合
RSS 是 imitation learning framework。如果扩展到 RL fine-tuning，reward signal 可以直接 shape $\Delta_{\text{sem}}$ —— reward 高时 amplify，reward 低时 suppress。这相当于 **language-conditioned RLHF**。

### 6.6 Causal Inference 视角
RSS 本质是 **counterfactual reasoning**: "如果没有 instruction，action 会是什么？" $s(a|o,\emptyset)$ 是 counterfactual baseline。$\Delta_{\text{sem}}$ 是 Average Treatment Effect (ATE) of language。这连接到 causal inference literature (Pearl, 2009)。Paper 可以更明确地 frame 成 causal estimation problem。

## 七、Reference 链接汇总

**Core methods**:
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- $\mathcal{E}_0$ (same group): https://arxiv.org/abs/2511.21542
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- $\pi_0$ FAST: https://arxiv.org/abs/2501.09747
- RDT-1B: https://arxiv.org/abs/2410.07864
- CogACT: https://arxiv.org/abs/2411.19650
- SpatialVLA: https://arxiv.org/abs/2501.15830
- CoT-VLA: https://arxiv.org/abs/2503.22030
- GR-1: https://arxiv.org/abs/2312.13139
- GR00T N1: https://arxiv.org/abs/2503.14734
- OneTwoVLA: https://arxiv.org/abs/2505.11917
- DITA: https://arxiv.org/abs/2503.19757
- TraceVLA: https://arxiv.org/abs/2412.10345
- MDT: https://arxiv.org/abs/2407.05996

**Foundational**:
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- CFG: https://arxiv.org/abs/2207.12598
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Octo: https://arxiv.org/abs/2405.12213

**Benchmarks**:
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- LIBERO-Pro: https://arxiv.org/abs/2510.03827

**Backbones**:
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Gemma: https://arxiv.org/abs/2403.08295

## 八、Summary Intuition

整个 RSS 框架的 intuition 可以浓缩为一句话：**VLA 模型的 failure 不是因为没看到 language，而是因为 visual gradient 在 training 时淹没了 language gradient，导致 $\|W_l\| \ll \|W_v\|$。RSS 通过两个 mechanism 修复这个 imbalance**:

1. **MCSI** (training-time): 用 LLM 生成 syntactic neighborhood，dense coverage of $p(l|z)$，让 $W_l$ 在更多 data points 上 receive gradient，扩大 $\|W_l\|$。

2. **RAS** (inference-time): 显式 subtract visual affordance $s(a|o,\emptyset)$，提取 pure semantic residual $\Delta_{\text{sem}} \approx W_l^\top \psi(l)$，再用 $\gamma > 1$ 人工 amplify 这个 residual，相当于 synthesizing $\tilde{W}_l = \gamma W_l$。

两者结合，MCSI 让 $W_l$ 学得更好，RAS 让 $W_l$ 用得更强。在 LIBERO 上 +29.85% 的 robustness gain 验证了这一 framework 的有效性。

这个工作对 embodied AI 的 broader implication：**modality imbalance 是 multi-modal foundation model 的通病，RSS 的 bias suppression 范式可以推广到 video-language generation、audio-visual learning 等任何 dense-vs-sparse modality 的场景**。
