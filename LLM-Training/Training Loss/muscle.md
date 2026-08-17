---
source_pdf: muscle.pdf
paper_sha256: 94f4e4d8ef728577b50f3980e3017cda95677fe4212a1bd30355ad7425db4adf
processed_at: '2026-08-05T21:47:10-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，Karpathy。咱们抛开那些学术包装，用最直白的话来拆解这篇 paper。

核心就一件事：**你升级了 LLM，整体 benchmark 涨了点，但用户发现以前好用的 prompt 突然抽风了。** 

用户用你的模型，慢慢摸出了一套 "mental model"（知道怎么哄它、怎么写 prompt 能得到好结果）。结果你一发新版，很多以前能做对的 case 现在做错了。这叫 **Negative Flip**。用户心智崩溃，投诉雪片般飞来。Apple 这篇 paper 就是来解决这个工程地狱的。

---

### 1. 为什么 Average Accuracy 是个谎言

假设你从 Llama 1 升级到 Llama 2。
- 100 道题，Llama 1 做对 75 道。
- Llama 2 做对了 80 道。
你看着报表说："Accuracy 提升了 5 个点，发版！"

但如果你拆开看 instance 层面：
- Llama 1 做对的 75 道题里，有 10 道 Llama 2 做错了。**（这 10 道就是 Negative Flips）**
- Llama 1 做错的 25 道题里，有 15 道 Llama 2 做对了。**（这 15 道是 Positive Flips）**

整体涨了 5 个点，掩盖了底层 10% 的内容发生了 "churn"（翻烧饼）。对用户来说，他根本不关心 average accuracy，他只关心："我上周写的那个代码生成 prompt，怎么今天突然语法就错了？"

这篇 paper 提出一个核心指标 **NFR (Negative Flip Rate)**，专门盯住这部分"退化的 case"：
$$ \mathrm{NFR} \triangleq \frac{1}{N}\sum_{i}^N \mathbf{1} [\mathrm{NF}(x_i)] $$
解释一下变量：$N$ 是测试集总数，$x_i$ 是第 $i$ 个样本，$\mathbf{1}$ 是指示函数（条件成立返回1，否则返回0）。简单说，NFR 就是**"以前对的现在错了"的样本比例**。

---

### 2. 生成式任务怎么算 Negative Flip？

如果是分类任务，对就是对，错就是错。但如果是 SAMsum 这种对话摘要任务呢？
Llama 1 生成的摘要是 A，Llama 2 生成的摘要是 B，Ground truth 是 C。
A 和 C 的 ROUGE-1 score 是 40，B 和 C 的 ROUGE-1 score 是 35。
B 虽然也是个还不错的摘要，但它的 ROUGE 分数比 A 低了 5 分。这就构成了一个 generative negative flip。

Paper 里定义了一个差值 $D(x_i)$：
$$ D(x_i) \triangleq S(\mathcal{M}_{v2}(x_i), y_i) - S(\mathcal{M}_{v1}(x_i), y_i) $$
变量解释：$S$ 是某种 similarity metric（比如 ROUGE-1），$\mathcal{M}_{v2}(x_i)$ 是新模型对样本 $x_i$ 的输出，$\mathcal{M}_{v1}(x_i)$ 是老模型输出，$y_i$ 是 ground truth。
如果 $D(x_i) < 0$，说明新模型在这个样本上退化了一点点。这就把离散的"对/错"变成了连续的"好/坏"度量。基于此定义了 $\widetilde{\mathrm{NFR}}$，也就是 ROUGE 下降的样本比例。

---

### 3. MUSCLE 的核心 trick：看脸色行事的 Knowledge Distillation

以前的解法是 knowledge distillation (KD)：把老模型当 teacher，新模型当 student，让新模型的 logits 分布尽量去模仿老模型。
问题在于：如果新模型本来在某道题上已经比老模型牛了，你强行让它向老模型对齐，那新模型的性能就掉下去了。

MUSCLE 的做法极度务实。它引入了一个新的 **Compatibility Adapter**（记作 $\mathcal{M}_{v2}^C$），初始化为新模型的 adapter $\mathcal{M}_{v2}$，然后开始微调它。
微调的时候，同时把老模型 $\mathcal{M}_{v1}$ 和新模型 $\mathcal{M}_{v2}$ 都当成 teacher。对 sequence 里的每一个 token $i$，它做了一个 **Masking 判断**：

$$ m_i = \mathbf{1}[\mathrm{argmax}\ \sigma(z_{\mathcal{M}_{v2}^C, i}) \neq y_i] $$

变量解析：$z_{\mathcal{M}_{v2}^C, i}$ 是当前 compatibility adapter 在位置 $i$ 输出的 logits，$\sigma$ 是 softmax，$y_i$ 是这个 token 的 ground truth。
翻译成人话：**如果当前的 compatibility adapter 自己预测错了（和 ground truth 不一致），mask $m_i$ 就设为 1；如果预测对了，mask $m_i$ 就是 0。**

然后，这个 mask 决定了当前 token 应该向哪个 teacher 学习：

$$ \mathcal{L}_{comp}^m = \frac{1}{n} \sum_{i=1}^n m_i \cdot a_{\mathcal{M}_{v1}} + (1 - m_i) \cdot a_{\mathcal{M}_{v2}} $$

变量解析：$n$ 是 sequence 的 token 总数。$a_{\mathcal{M}_{v1}}$ 是向老模型做 KL divergence 蒸馏的 loss，$a_{\mathcal{M}_{v2}}$ 是向新模型做 KL 蒸馏的 loss。
$$ a_{\mathcal{M}_{v1}} = KL\big(\sigma(z_{\mathcal{M}_{v1}, i}/T) \|\sigma(z_{\mathcal{M}_{v2}^C, i}/T)\big) $$
这里的 $T$ 是 temperature（论文里设为 2，让 softmax 软一点），$z$ 是 logits。

**直觉建立**：
- 当 $m_i = 1$（当前 student 预测错了）：loss 变成 $1 \cdot a_{\mathcal{M}_{v1}} + 0 \cdot a_{\mathcal{M}_{v2}}$。此时 student 拼命向老模型 $\mathcal{M}_{v1}$ 学习。因为既然你做错了，那就去抄老模型的作业，老模型在这些 case 上大概率是对的，所以能捡回被新模型丢失的能力。
- 当 $m_i = 0$（当前 student 预测对了）：loss 变成 $0 \cdot a_{\mathcal{M}_{v1}} + 1 \cdot a_{\mathcal{M}_{v2}}$。此时 student 向新模型 $\mathcal{M}_{v2}$ 学习。既然你做对了，那就保持新模型的行为，吸收新模型带来的性能提升。

这个设计的巧妙之处在于：它天然处理了 "Q3 诡异区间"（即两个模型都错的情况）。如果新模型错了，它就无脑向老模型对齐，哪怕老模型也错。为什么？因为 **consistency**。用户宁愿你一直犯同样的错（这样他们能写 workaround），也不愿你每次升级换一种错法。

---

### 4. 实验数据里的反直觉发现

他们测了 Llama 1 -> Llama 2, Phi 1.5 -> Phi 2 等各种组合。结果非常有意思：

**发现 1：性能差距越小，Negative Flip 越多。**
Llama 1 到 Llama 2 在 HellaSwag 上，accuracy 几乎没变（72.74 -> 72.91），但 NFR 高达 10.27%。这就像换了一版稍微不同的随机种子，虽然平均分没变，但具体对错的题完全翻了一批。对用户的打击极重。

**发现 2：MUSCLE 训练后，Accuracy 居然还涨了。**
Llama 1 -> Llama 2 用 MUSCLE 训练后，NFR 降了 40%，同时 accuracy 暴涨 6.62 个点（72.74 -> 79.53）。
这非常反直觉。按理说向老模型妥协会损失性能，但实际却涨了。Paper 解释这是 "ensemble knowledge effect"——相当于 student 把 v1 和 v2 两个 teacher 的知识给综合了。

**发现 3：老模型太烂时，MUSCLE 会反噬。**
Phi 1 在 GSM8k 数学任务上准确率只有 3.4%，Phi 1.5 涨到 30%。用 MUSCLE 强行向只有 3.4% 准确率的 Phi 1 对齐，结果新 adapter 的 accuracy 掉了 3 个点。
这符合直觉：如果老模型是个智障，你抄它的作业只会把自己变蠢。

---

### 5. 工程落地的直觉

如果你在生产环境里用 LoRA 微调部署 LLM：
1. 你平时肯定是用同一个 base model，更新数据来 retrain LoRA。这种情况 NFR 极高，最适合用 MUSCLE。
2. 部署 MUSCLE 时，你不需要在 inference 时同时加载老模型和新模型。你只是训出了一个**全新的 LoRA adapter**，它的形状和参数量跟普通 task adapter 完全一样。inference 成本零增加。
3. 唯一的成本是 training 时显存翻倍，因为你要把 $\mathcal{M}_{v1}$ 和 $\mathcal{M}_{v2}$ 都放进显存里当 teacher。他们用了 r=128 的 LoRA rank（比平时常用的 r=16/32 大得多），因为 compatibility 需要更高的 capacity 来容纳两代模型的知识冲突。

**总结一句**：MUSCLE 的精神就是 "只抄你不会的题"。会做的题，你按新模型的路子走，保持进步；不会做的题，去翻老模型的底牌，保住用户的 mental model。

### 参考链接
- Paper ArXiv: https://arxiv.org/abs/2402.01838
- ACL Anthology: https://aclanthology.org/2024.acl-long.345/
- LoRA 原论文: https://arxiv.org/abs/2106.09685
- Negative Flip Rate 概念起源 (Yan et al. 2021): https://arxiv.org/abs/2105.03048

---

# MUSCLE: 模型更新时的兼容性策略 - 深度解析

## 1. 核心问题与 Motivation

这篇论文 (Apple + UC San Diego, ACL 2024) 抓住了一个工程上极常见但学术界长期被忽视的痛点: **base model 更新后,下游 task adapter 即便用完全相同的训练流程重训,也会出现大量 negative flip —— 旧版本能做对的样本新版本做错了**。这直接破坏 Bansal et al. 2019 提出的 "user mental model",用户不知道这次升级到底什么能用什么不能用,信任崩塌。

举个直觉例子: 你用 Llama 1 + LoRA 微调了一个 summarizer,业务跑得好好的。Meta 升级到 Llama 2,你用同样的 SAMSum 数据集、同样的 hyperparameter 重训一个 LoRA,overall ROUGE 提升了 0.2 个点, 但用户报告说**之前那个特定风格的对话突然总结错了**。这就是 instance regression, paper Figure 1 给的就是这种真实例子。

关键 insight: **整体指标 (accuracy / ROUGE) 的平均提升掩盖了 instance 层面的剧烈 churn**。Yan et al. 2021 在 vision classification 上首次系统化这个现象叫 NFR (Negative Flip Rate),但 LLM 时代把这件事 formalize 出来的,MUSCLE 是第一篇,而且扩展到了 generative 任务。

## 2. 四象限分类法 — 这是一切讨论的基础

Figure 2 的四象限划分是全文的概念核心,理解它之后后面所有公式都顺理成章:

| Quadrant | $\mathcal{M}_{v1}$ | $\mathcal{M}_{v2}$ | 说明 |
|----------|-------------------|-------------------|------|
| Q1 | ✓ 正确 | ✓ 正确 | 理想状态,但即便都正确,生成式任务里也可能"略有差异" |
| Q2 | ✗ 错误 | ✓ 正确 | Positive flip — 升级该有的样子 |
| Q3 | ✗ 错误 | ✗ 错误 | **Träuble 2021 假设这里 cost=0,MUSCLE 反对这个** |
| Q4 | ✓ 正确 | ✗ 错误 | Negative flip — 真正的 regression |

为什么 Q3 重要? Karpathy 你可以想象 ChatGPT 用户的心理: 即使 GPT-4o 在某个 edge case 上也答错, 但如果每次错的方式都不一样, 用户根本没法 build a coping strategy ("我加这段 prompt 它就会避开这个错"), 这种 inconsistency 比"还是错"更让人崩溃。这是这篇 paper 的社会学层 insight。

## 3. 评估指标体系

### 3.1 经典 NFR (Yan et al. 2021)

$$
\mathrm{NF}(x_i) \triangleq [\mathcal{M}_{v1}(x_i) = y_i] \wedge [\mathcal{M}_{v2}(x_i) \neq y_i]
$$

$$
\mathrm{NFR} \triangleq \frac{1}{N}\sum_i^N \mathbf{1}[\mathrm{NF}(x_i)]
$$

变量解释: $x_i$ 是第 $i$ 个输入样本, $y_i$ 是 ground truth, $\mathcal{M}_{v1}, \mathcal{M}_{v2}$ 分别是 v1 / v2 task-adapted model, $N$ 是测试集大小, $\mathbf{1}[\cdot]$ 是 indicator function (条件为真返回1, 否则0)。NFR 即"之前对的现在错的"占比。

### 3.2 NFR_mc — Multiple Choice 扩展,捕捉 Q3

$$
\mathrm{NF}_{mc}(x_i) \triangleq [\mathcal{M}_{v2}(x_i) \neq y_i] \wedge [\mathcal{M}_{v1}(x_i) \neq \mathcal{M}_{v2}(x_i)]
$$

$$
\mathrm{NFR}_{mc} \triangleq \frac{1}{N}\sum_i^N \mathbf{1}[\mathrm{NF}_{mc}(x_i)]
$$

这里巧妙之处: 把判定条件从 "$\mathcal{M}_{v1}$ 对、$\mathcal{M}_{v2}$ 错"放宽到 "$\mathcal{M}_{v2}$ 错且两个模型预测不一致"。这样 Q3 里那种"两个都错但错得不一样"也被计入 inconsistency。HellaSwag / PIQA 这种 multi-choice benchmark 直接套用。

### 3.3 连续指标 — 生成任务的关键创新

对于 summarization 这类没有"正确答案枚举"的任务, paper 引入一个 similarity metric $S$ (可以是 ROUGE / BLEU / BERTScore / LLM-as-judge):

$$
D(x_i) \triangleq S(\mathcal{M}_{v2}(x_i), y_i) - S(\mathcal{M}_{v1}(x_i), y_i)
$$

$D(x_i) > 0$ 表示新模型对样本 $i$ 更接近 ground truth (gain), $D(x_i) < 0$ 表示 regression。基于 $D$ 定义:

$$
\widetilde{\mathrm{PFR}} \triangleq \frac{1}{N}\sum_i^N \mathbf{1}[D(x_i) > 0], \quad \widetilde{\mathrm{NFR}} \triangleq \frac{1}{N}\sum_i^N \mathbf{1}[D(x_i) < 0]
$$

更进一步, 用 $m_g, m_r$ 量化 magnitude:

$$
m_g \triangleq \frac{1}{N \cdot \widetilde{\mathrm{PFR}}}\sum_i^N D(x_i)\mathbf{1}[D(x_i)>0]
$$

$$
m_r \triangleq \frac{1}{N \cdot \widetilde{\mathrm{NFR}}}\sum_i^N |D(x_i)|\mathbf{1}[D(x_i)<0]
$$

变量: $m_g$ 是所有 positive flip 样本的平均 gain 幅度, $m_r$ 是所有 negative flip 样本的平均 regression 幅度。这两个量很重要 — 只看 NFR 会把 "掉0.1 ROUGE" 和 "掉5 ROUGE" 一视同仁, $m_r$ 把这种 severity 暴露出来。

## 4. MUSCLE 方法 — Compatibility Adapter 训练

### 4.1 Setup

记号系统 (Section 3):

- $\mathcal{M}_i^{\mathrm{base}}$: 第 $i$ 版 base LLM,参数 $\theta_i$
- $\mathcal{A}_i^{\mathcal{T}}$: 任务 $\tau$ 上的 LoRA adapter
- $\mathcal{M}_i^{\mathcal{T}}$: task-adapted model, 参数 $\theta_i^{\mathcal{T}} = \theta_i + \Delta_i^{\mathcal{T}}$, 其中 $\Delta_i^{\mathcal{T}}$ 是 LoRA 增量

当 base 从 $\mathcal{M}_{v1}^{\mathrm{base}}$ 升级到 $\mathcal{M}_{v2}^{\mathrm{base}}$, 通常会**重新训 LoRA** 得到 $\mathcal{M}_{v2}$。MUSCLE 要训的是第三个模型 $\mathcal{M}_{v2}^C$, 它有 $\mathcal{M}_{v2}$ 的能力同时具备 $\mathcal{M}_{v1}$ 的兼容性。

### 4.2 Architecture (Figure 3 解析)

训练时有**三个模型**同台:

1. $\mathcal{M}_{v1}$ — frozen teacher 1 (旧版, 用作"保持兼容"信号源)
2. $\mathcal{M}_{v2}$ — frozen teacher 2 (新版, 用作"性能提升"信号源)
3. $\mathcal{M}_{v2}^C$ — trainable student, 即 compatibility adapter, **初始化为 $\mathcal{M}_{v2}$ 的 task adapter**

这点很关键: 初始化用 $\mathcal{M}_{v2}$ adapter 而非随机,意味着 student 起点 = 新模型, 然后用 distillation 损失把它**轻微拉向** $\mathcal{M}_{v1}$, 在保持性能的同时恢复对旧版正确样本的覆盖。这本质上是 **continued fine-tuning**, 而不是 from-scratch 重训, 训练成本低。

### 4.3 Masking 策略 + KL 蒸馏损失 (Eq. 1)

核心公式 (Section 4.1, Eq. 1):

$$
m_i = \mathbf{1}[\mathrm{argmax}\ \sigma(z_{\mathcal{M}_{v2}^C,i}) \neq y_i]
$$

$$
a_{\mathcal{M}_{v1}} = KL\big(\sigma(z_{\mathcal{M}_{v1},i}/T)\ \|\ \sigma(z_{\mathcal{M}_{v2}^C,i}/T)\big)
$$

$$
a_{\mathcal{M}_{v2}} = KL\big(\sigma(z_{\mathcal{M}_{v2},i}/T)\ \|\ \sigma(z_{\mathcal{M}_{v2}^C,i}/T)\big)
$$

$$
\mathcal{L}_{comp}^m = \frac{1}{n}\sum_{i=1}^n m_i \cdot a_{\mathcal{M}_{v1}} + (1-m_i) \cdot a_{\mathcal{M}_{v2}}
$$

变量逐项解释:

- $z_{\mathcal{M},i}$: 模型 $\mathcal{M}$ 在位置 $i$ 的 logits (vocabulary 维度向量)
- $\sigma(\cdot)$: softmax 函数
- $T$: temperature, paper 中设 $T=2$, 软化分布让 KL 更平滑
- $y_i$: 第 $i$ 个 token 的 ground truth
- $m_i \in \{0, 1\}$: mask, **如果 student $\mathcal{M}_{v2}^C$ 当前 argmax 预测不等于 ground truth, 设为 1**
- $a_{\mathcal{M}_{v1}}$: student 向 v1 teacher 蒸馏的 KL 损失
- $a_{\mathcal{M}_{v2}}$: student 向 v2 teacher 蒸馏的 KL 损失
- $n$: sequence 总 token 数

**Intuition**: 这个 mask 是个"路由器"。如果 student 已经预测对, 拉它向 v2 (强化新版知识); 如果 student 预测错, 拉它向 v1 (从旧版里捞回那些被新版丢失的样本)。这相当于**在 token 层面动态地决定向谁学习**, 比单纯 KD 灵活得多。

注意 KL 方向: $KL(\text{teacher} \| \text{student})$ 是 forward KL, 等价于以 teacher 分布为 target 的 cross-entropy, 这保证 student 试图覆盖 teacher 的 high-probability 区域。

### 4.4 为什么这个设计比朴素 KD 好

朴素 KD (Shen 2020, Yan 2021) 把 student 完整拉向 v1 teacher, 这会丢失 v2 的能力提升。MUSCLE 的 mask 把 v2 正确的部分**冻结** (因为初始就是 $\mathcal{M}_{v2}$, $a_{\mathcal{M}_{v2}}$ 的梯度方向也是 v2 自己,几乎不动), 只对错误的部分用 v1 修正。Section 6.5 Table 7 的 ablation 直接验证:

| 方法 | $\Delta\mathrm{NFR}_c$ | $\Delta\%\mathrm{NFR}_c$ | $\Delta\mathrm{PFR}_c$ | $\Delta\mathrm{acc}_c$ |
|------|------------------------|---------------------------|------------------------|------------------------|
| $\mathcal{L}_{comp}^{\mathcal{M}_{v2}^C \neq y}$ (MUSCLE) | **-3.97** | **-34.25** | **+0.54** | **+4.51** |
| $\mathcal{L}_{comp}^{\mathcal{M}_{v1} = y}$ (只在 v1 对时拉 v1) | -2.29 | -19.76 | -0.60 | +1.68 |
| $a_{\mathcal{M}_{v1}}$ only (无 mask, 全员拉 v1) | -2.99 | -25.80 | -0.82 | +2.17 |
| CE + $\mathcal{L}_{comp}^{\mathcal{M}_{v2}^C \neq y}$ | -3.10 | -26.75 | +0.38 | +3.48 |

注意 MUSCLE 还**提升了 accuracy +4.51**, 这是 "ensemble knowledge effect" — student 被两个 teacher 同时"指导", 知识聚合, 类似 Jaeckle 2023 (FastFill) 在 vision retrieval 上观察到的现象。

## 5. 实验设置详解

### 5.1 模型对 (Table 2)

| $\mathcal{M}_{v1}^{\mathrm{base}}$ | $\mathcal{M}_{v2}^{\mathrm{base}}$ | 更新类型 |
|---|---|---|
| Phi 1 (1.3B) | Phi 1.5 (1.3B) | Synthetic data + data selection (Gunasekar 2023) |
| Phi 1.5 (1.3B) | Phi 2 (2.7B) | Synthetic data + 参数翻倍 (Javaheripi 2023) |
| Llama 1 (7B) | Llama 2 (7B) | 数据更多 + context 加长 |
| Vicuna 1.3 (7B) | Vicuna 1.5 (7B) | Llama 1→2 + instruction tuning |

这覆盖了4种典型更新场景: 数据更新、参数扩大、架构改进、base 替换, 覆盖面比较周全。

### 5.2 任务对 (Table 3)

| Dataset | Task | Metric |
|---|---|---|
| HellaSwag | 语言理解 (4选1) | Log-Likelihood Accuracy |
| PIQA | 物理常识 (2选1) | Log-Likelihood Accuracy |
| GSM8k | 小学数学 | Exact Match |
| SAMsum | 对话摘要 | ROUGE-1 |

包含 multi-choice (分类式评估) + generation 两大类, 后者是 paper 主推的贡献点。

### 5.3 训练超参 (Table 8)

| Hyperparameter | Value |
|---|---|
| Epochs | 10 |
| Learning Rate | 1e-4 |
| Gradient accumulation | 8 |
| LoRA α | 256 |
| LoRA rank | 128 |
| Dropout | 0.0 |
| Adapter layers | All linear |
| Warmup Steps | 500 |
| KL temperature T | 2 |

注意 LoRA rank 用到 128, 比典型 16/32 高很多, paper 解释说 compatibility adapter 比 task adapter 需要更高容量 — 因为它要同时编码两个模型的知识。Dropout=0 也合理,因为这是 continued training,加正则反而干扰 distillation 信号。

训练成本 (A.2): 720×8 GPUh ≈ $14,400, A100+H100。Inference cost 跟普通 LoRA 一致, 因为 compatibility adapter 跟 task adapter 结构相同。

## 6. 关键实验结果

### 6.1 Negative Flip 在所有更新中都存在 (Figure 4)

最重要的观察: **当 v1 和 v2 整体性能差距小时, NFR 反而更高**。直觉: 升级跨度小, v1 学对的样本大部分被 v2 也学对了,但 v2 又新增了一些它自己更自信的"错误偏好",结果被它误杀了一批 v1 对的样本。升级跨度大 (Phi 1 EM=3.4% → Phi 1.5 EM=30%), v1 本来就不太对, flip 概率上限就低。

这对工程实践的启示: **频繁小步迭代比一次大升级更容易触发用户感知到的 regression**, 这正好是工业界常见的发布模式 (周更/双周更), 所以 MUSCLE 这种方法实际意义重大。

### 6.2 Multi-choice 结果 (Table 4)

Llama 1 → Llama 2, HellaSwag:
- $\mathrm{acc}_{v1} = 72.74$, $\mathrm{acc}_{v2} = 72.91$ (几乎没提升)
- $\mathrm{acc}_c = 79.53$ (**+6.62**)
- NFR 从 10.27 降到 6.10 (**-40.6%**)

注意 acc 居然涨了 6.62 个点, 这远超预期。Paper 在 6.5 节解释这是 ensemble knowledge effect — student 从 v1 拿到了 v2 没学到的部分知识, 反而综合精度更高。这给方法学带来一个意外惊喜: **compatibility training 可能本身就是个 regularizer / 集成方法**。

Vicuna 1.3 → Vicuna 1.5 也类似: NFR -38.74%, acc +6.56.

Phi 1.5 → Phi 2 (性能跳跃大): NFR 仅降 7.77%, acc 反而 -0.16。验证 Section 6.6 的洞察: 性能差距大时 MUSCLE 价值有限。

### 6.3 GSM8k 数学 (Table 5)

| Update | $\mathrm{EM}_{v1}$ | $\mathrm{EM}_{v2}$ | $\mathrm{EM}_c$ | $\Delta\mathrm{EM}_c$ | NFR | $\Delta\%\mathrm{NFR}_c$ |
|---|---|---|---|---|---|---|
| Llama 1→2 | 24.45 | 33.09 | 36.66 | +3.57 | 8.49 | -10.72 |
| Phi 1.5→2 | 30.02 | 48.18 | 50.68 | +2.50 | 5.88 | **-29.08** |
| Phi 1→1.5 | 3.41 | 30.02 | 26.99 | -3.03 | 2.01 | -1.99 |
| Vicuna 1.3→1.5 | 26.72 | 29.91 | 31.84 | +1.93 | 11.60 | -8.53 |

Phi 1→1.5 这个 case 警示: 当 v1 准确率只有 3.4% 时, 强行向 v1 蒸馏是有害的 (-3.03 EM), 这时 compatibility 的价值远小于 performance 损失。Paper 在 Limitation 里直说: 是否值得用 compatibility 换 performance drop, 需要业务判断。

### 6.4 SAMsum 摘要 (Table 6) — 生成任务核心结果

| Update | $\mathrm{R1}_{v1}$ | $\mathrm{R1}_{v2}$ | $\mathrm{R1}_c$ | $\Delta\mathrm{R1}_c$ | $\widetilde{\mathrm{NFR}}$ | $\Delta\%\widetilde{\mathrm{NFR}}_c$ | $\Delta m_g$ | $\Delta m_r$ |
|---|---|---|---|---|---|---|---|---|
| Llama 1→2 | 32.06 | 32.28 | 34.79 | +2.51 | 48.96 | **-15.95** | +0.64 | -1.13 |
| Phi 1.5→2 | 37.53 | 36.15 | 40.69 | +4.54 | 54.70 | **-27.46** | +0.24 | -3.39 |
| Phi 1→1.5 | 30.92 | 37.53 | 38.76 | +1.23 | 32.60 | **-17.61** | -0.38 | -0.59 |
| Vicuna 1.3→1.5 | 30.32 | 30.88 | 34.08 | +3.20 | 49.69 | **-22.10** | -0.02 | -1.85 |

注意 SAMsum 的 $\widetilde{\mathrm{NFR}}$ 都在 30-55% 这个量级,远高于 multi-choice 任务的 10% 左右 — 因为 ROUGE-1 这种连续 metric 对生成文本极其敏感, 几乎任何 wording 变化都可能让 $D(x_i) < 0$。这恰恰说明用 NFR_mc 单独评估 multi-choice 不够, 生成任务必须用 $\widetilde{\mathrm{NFR}}$。

$m_r$ 全部为负, 说明 MUSCLE 不只减少 flip 数量, 也降低了 flip 的严重程度 — 平均 regression 幅度更小, 对用户的"惊讶度"也更低。

## 7. Ablation 中的 Masking 策略对比

Section 6.5 Table 7 测试了 6 种 mask 设计:

1. **$\mathcal{L}_{comp}^{\mathcal{M}_{v2}^C \neq y}$** (MUSCLE, 当前 student 错才拉 v1): 最佳
2. $\mathcal{L}_{comp}^{\mathcal{M}_{v1} = y}$ (只在 v1 对时拉 v1): 次之
3. $a_{\mathcal{M}_{v1}}$ (无 mask, 全部拉 v1): 中
4. CE + MUSCLE: 加 cross-entropy 反而稍差
5. $\mathrm{CE} + \mathcal{L}_{comp}^{LL_L}$ (token-level likelihood mask): 见 Eq. 2
6. $\mathrm{CE} + \mathcal{L}_{comp}^{LL_S}$ (sequence-level likelihood mask): 见 Eq. 3

Likelihood-based mask 公式:

$$
m_i = LL_L = \mathbf{1}[\sigma(z_{\mathcal{M}_{v2}^C,i}) < \sigma(z_{\mathcal{M}_{v1},i})] \quad (\text{Eq. 2})
$$

意思是当前 student 在 ground-truth token 上的概率若低于 v1, 才拉 v1。这是更细粒度的判断, 但需要 cross-entropy 辅助损失稳定训练 (A.4):

$$
\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N\sum_{k=1}^K y_{i,k}\log(\sigma(z_{\mathcal{M}_{v2}^C,i,k})) \quad (\text{Eq. 4})
$$

$$
\mathcal{L} = \lambda\mathcal{L}_{comp}^m + (1-\lambda)\mathcal{L}_{CE} \quad (\text{Eq. 5})
$$

变量: $K$ 是 vocab 大小, $y_{i,k}$ 是 one-hot ground truth, $\lambda \in [0,1]$ 是损失权重。Paper 解释 likelihood-based 方法在 large gap 情况下需要 $\mathcal{L}_{CE}$ 因为 likelihood 不能保证整体分布合理。

最终结论: $\mathcal{L}_{comp}^{\mathcal{M}_{v2}^C \neq y}$ 最优, 因为它**直接以 ground-truth 为锚**, 而非以模型间相对概率比较, 鲁棒性更好, 还天然处理 Q3 (inconsistency) 情况 — 因为只要 v2 错就拉 v1, 不管 v1 对不对。

## 8. 我的几点 Intuition 与延伸思考

### 8.1 与 Forward-Backward Compatibility 的关系

经典 backward compatibility (Srivastava 2020) 只要求"新版对的地方 ⊇ 旧版对的地方",这其实是 set inclusion, MUSCLE 把它扩展成分布对齐 (KL on logits),信息利用更充分。在 vision retrieval 领域, Ramanujan 2022 的 Forward Compatible Training 强调新模型 embedding 空间要兼容旧 query, 思路同源, 但 LLM 自回归特性让 MUSCLE 必须做到 **token-level** 对齐, 这是难度差异点。

### 8.2 类比 Continual Learning

MUSCLE 本质是 continual learning 的一个变体, $\mathcal{M}_{v1}$ 是"旧任务", $\mathcal{M}_{v2}$ 是"新任务", compatibility adapter 是要 avoid catastrophic forgetting 的 adapter。但与经典 CL 不同的是, "旧任务"和"新任务"用的是**同一份数据**, 区别只在 base model — 这就简化了数据访问问题, 不需要 replay buffer, 不存在 data privacy barrier。

### 8.3 Mask 选择的更深一层 intuition

为什么"student 错才拉 v1"比"v1 对才拉 v1"好? 设想两种情形:

- Student 错, v1 也错: 朴素的 $\mathcal{L}_{comp}^{\mathcal{M}_{v1} = y}$ 不拉, $\mathcal{L}_{comp}^{\mathcal{M}_{v2}^C \neq y}$ 拉。
- 后者让 student 至少**和 v1 错得一样**, 用户保持 coping strategy 不变。

这就是 Section 4 中 Q3 处理的关键技术点, 它**在数学上等价于"在错误子空间内, student 应当模仿 v1"**, 这是 paper 没明说但实际编码的逻辑。这点用 NFR_mc 指标验证 (Figure 5) 就看到效果显著。

### 8.4 LoRA rank=128 的暗示

普通 task LoRA r=16 就够, 但 compatibility 需要 r=128, 这暗示"兼容性知识"在参数空间是更高 rank 的。可能因为 v1 的"已被记住的正确样本"散布在很多 subspace, 要把它们重新激活, low-rank projection 不够。这点可以作为后续研究的切入口 — 是否能用 dynamic rank 或 task-specific rank allocation 进一步优化?

### 8.5 与 Phi 1→1.5 失败 case 的连接

Phi 1 EM=3.4%, Phi 1.5 EM=30%, MUSCLE 训出来 EM=27% (掉3个点)。这本质是 teacher quality 决定 student 上限 — 当 v1 teacher 本身就极差, KL 损失反而把 student 拉向 v1 的错解。Paper 提议 instance-based loss weighting (Limitation), 我直觉上认为可以引入 **confidence-weighted distillation**: 用 v1 在 sample 上的 log-likelihood 作为权重, v1 低置信度时 mask=0 不拉。这跟 focal loss 思路一致。

### 8.6 推广到 Tokenizer 变化

Paper Limitation 提到不能处理 Llama 2→Llama 3 (tokenizer 改了, vocab 从 32k 变 128k)。这是个真问题: logit 空间维度都不同, KL 直接算不了。可能的解法是学一个 **vocabulary projection matrix** $P \in \mathbb{R}^{|V_2| \times |V_1|}$, 把 v1 logits 映射到 v2 vocab 再做 KL。这等价于 cross-lingual word embedding alignment 的扩展, 应该可行。

### 8.7 与 RLHF / DPO 的潜在联动

如果 v2 不是 base 升级而是 RLHF 后的 chat model, MUSCLE 应该照样可用 — 把 v1 chat model 作 teacher, 训 v2 chat model 的 compatibility adapter。这给 ChatGPT/GPT-4 这种频繁小版本迭代提供了正式的"温和升级"训练框架, 应用空间巨大。

### 8.8 训练成本的边际

A.2 提到训练时两个 teacher 都要驻留 GPU memory。对 7B 模型 + r=128 LoRA, 用 8×A100 应该可行。但若是 70B 级别, 需要更精细的 memory optimization, 比如 teacher 用 8-bit 量化, 或异步 KD (teacher 先离线 cache logits)。这条工程线值得专门 paper。

## 9. 局限性与开放问题

Paper 自己列出:

1. **不支持 tokenizer/vocabulary 变化** (Llama 2→3 不可用)
2. **大性能差距时失效** (Phi 1→1.5 反而掉点)
3. **潜在 bias transfer** — v1 的 bias 会被传到 v2, 没有研究
4. **Performance gain 来源不明** — 是 ensemble effect 还是 regularization?

我额外想加的开放问题:

5. **Multi-turn dialogue 任务未涵盖** — SAMsum 是 single-turn summarization, 但 ChatGPT 用户最痛的是 multi-turn 体验一致性, 这里 MUSCLE 还没碰。
6. **Reward hacking 风险** — KL 到 v1 可能让 student 学到 v1 的 spurious correlation (e.g. 特定模板), 而非真正的"对的样本", 这种 surface alignment 短期降 NFR, 长期可能损害 generalization。
7. **理论分析缺失** — 为什么 mask-based KD 比 vanilla KD 在 NFR reduction 上有质变? 信息论分析可以补强。
8. **Sample efficiency** — 跟 task adapter 一样需要全量训练数据, 是否能 few-shot / zero-shot compatibility training? 这对 adapter as a service 业务模型很关键。

## 10. 总结性 intuition

MUSCLE 的核心 trick 用一句话讲: **让新版 student 在"它能做对"的 token 上保持自我, 在"它做错"的 token 上跪舔旧版**, 用 argmax + ground-truth 这一对简单判定做路由。简单但有效 — 因为它把"什么算对"这个高频重复信号锚定住了, 而把"在哪学"这件事做了条件化, 避免了朴素 KD 把所有 token 一视同仁向 v1 拉的性能损失。

对 LLM 工程实践者, 这篇 paper 给的 takeaway: 不要只看 aggregate metric 升级模型, 一定要在 release pipeline 里加 NFR / $\widetilde{\mathrm{NFR}}$ regression test, 然后用 MUSCLE 这种 compatibility adapter 做最后一道"温和化"训练, 7B 模型在 8×A100 上一天能跑完一轮, 收益是用户对升级的"无感化"。

## 参考链接

- Paper PDF (ACL 2024): https://aclanthology.org/2024.acl-long.345/
- ArXiv version: https://arxiv.org/abs/2402.01838
- Author page (Jessica Echterhoff, UCSD): https://j-echterhoff.github.io/
- Author page (Hadi Pouransari, Apple): https://hadipoulansari.github.io/
- LoRA paper (Hu et al. 2021): https://arxiv.org/abs/2106.09685
- Yan et al. 2021 (NFR origin): https://arxiv.org/abs/2105.03048
- Bansal et al. 2019 (mental models in human-AI teams): https://ojs.aaai.org/index.php/HCOMP/article/view/5271
- Träuble et al. 2021 (Q3 cost assumption): https://arxiv.org/abs/2111.10418
- Jaeckle et al. 2023 (FastFill, similar finding in vision): https://arxiv.org/abs/2303.04766
- Llama 2 tech report: https://arxiv.org/abs/2307.09288
- Phi-2 technical report (Javaheripi et al. 2023): https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/
- LM Evaluation Harness (Gao et al. 2023): https://github.com/EleutherAI/lm-evaluation-harness
- HellaSwag dataset: https://arxiv.org/abs/1905.07830
- PIQA dataset: https://arxiv.org/abs/1911.11641
- GSM8k dataset: https://arxiv.org/abs/2110.14168
- SAMsum dataset: https://arxiv.org/abs/1911.12237
