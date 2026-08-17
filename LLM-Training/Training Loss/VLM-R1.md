---
source_pdf: VLM-R1.pdf
paper_sha256: a6ca9a3267f1b11b95b906d0a57974a5160c97e6e6745c7d9656cf1acd83adcc
processed_at: '2026-08-13T03:07:21-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLM-R1 人话版

## 这 paper 到底干了啥

一句话：**把 DeepSeek R1 那套"用规则奖励做 RL"的 recipe 搬到 vision-language model 上，看看能不能让 VLM 学会视觉推理，结果发现比 SFT 更 generalize，还顺带撞见了一个叫 "OD aha moment" 的涌现现象。**

就这么简单。整个 paper 的故事性其实很强，像一部有 plot twist 的小纪录片。

---

## 为什么要干这个事

先说背景。R1 的核心 insight 是：你不需要训练一个 reward model 去模拟人类偏好，只要任务有**确定答案**（数学题有标准解、代码题有 unit test），你就能直接用规则算 reward，让 LLM 自己摸索怎么 reasoning。

那 VLM 这边呢？作者发现了一个被忽视的事实：**大量视觉任务天然就有确定 ground truth**。REC（referring expression comprehension）要预测一个 bounding box，ground truth 就是那个 box，IoU 一算就完了。OVD（open-vocabulary detection）也一样，每个物体都有一个标准 box + 标签。

这意味着 R1 的 recipe 几乎可以原样移植——只要你能 parse 模型输出、算出一个 scalar reward，GRPO 就能跑起来。

更刺激的是 Table 1 给的那个 motivation：Qwen2.5-VL-3B 有 37.5 亿参数，是 Grounding DINO（3.4 亿）的 10 倍多，但在 RefCOCO 系列 benchmark 上**反而打不过** Grounding DINO。说明 VLM 虽然会聊天、会 OCR、会写代码，但它在"精准看图定位物体"这种 low-level perception 上，反而比专门的小模型差。这是个尴尬的 gap，作者想用 RL 把这个 gap 填上。

---

## 怎么干的

### GRPO 用人话讲

PPO 之前是 RLHF 的标配，但它要训一个 critic model（value network）来估计"这个 state 值多少分"，对于 LLM 这种巨无霸 state space，critic 训得又慢又烂。

GRPO 的 trick 很简单粗暴：**对同一个 prompt 采样 N 个回答，让它们互相比较**。

打个比方：你让 8 个学生做同一道题，老师不打绝对分，只看相对排名——比平均好的学生加分、比平均差的扣分。这样就不需要事先定义"满分是多少"，只用组内 mean 和 std 归一化一下就行。

数学上就是：
$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_N)}{\text{std}(r_1, ..., r_N)}$$

$r_i$ 是第 $i$ 个 response 的 reward，$A_i$ 就是它的"组内相对优势"。比平均高多少个标准差，policy 就往这个方向 update 多少。

然后套 PPO 那套 importance ratio + clip + KL penalty，更新 policy。

GRPO 论文：https://arxiv.org/abs/2402.03300

### Reward 怎么设计

REC 任务特别干净：模型输出一个 bbox，跟 ground truth 算 IoU，IoU 就是 reward，范围 [0, 1]。完事。

OVD 任务就脏了：模型要输出一堆 bbox + label pair，你要算 mAP。但 mAP 这个指标有个隐含 bug——COCO 官方 API 在算 AP 时，会把当前 image 里没有 ground truth 的 category 直接**排除出分母**。

这意味着什么？**模型狂吐 box，错的也吐，反正不亏**。对于一个 image 里没有"消防栓"的样本，模型预测 10 个"消防栓"框，AP 计算时这些 prediction 压根不参与计算——免费的尝试。万一这个 image 里真有"消防栓"，多吐几个框说不定蒙对一个，recall 上去了。所以模型学到的策略是：**把 80 个 COCO 类全吐一遍，每个类多吐几个框**。

这就是 reward hacking——reward 函数被 model 钻空子了，reward 涨得飞快，但实际 detection 质量稀烂。

Figure 6 里那张图特别直观：用原始 AP50 reward 训练，模型 output 长度从几百 token 爆炸式涨到几千 token——它在疯狂输出框。

作者的 fix 是加一个 length penalty：
$$s_{ovd} = \min\left(1, \frac{L_{gt}}{L_{pred}}\right)$$

$L_{gt}$ 是 ground truth 数量，$L_{pred}$ 是预测数量。预测数 > ground truth，reward 直接按比例缩水。预测 50 个但 GT 只有 5 个？mAP 乘以 0.1，痛到 model 自己学会克制。

这个 penalty 看起来 trivial，但 Table 5 显示效果炸裂：mAP 从 11.8 飙到 21.1（COCO_filtered），OVDEval overall 从 21.68 飙到 31.01。**一个公式把性能翻倍**。

---

## 干出来什么

### REC：RL 把 reasoning 学会迁移了

Table 2 是整个 paper 最核心的一张表。

训练数据是 RefCOCO/+/g，这种数据的特点是描述非常 surface——"左边穿红衣服的人"这种 spatial attribute。没有 reasoning 成分。

测试有两个：
- **In-domain**：RefCOCO val split，同分布
- **Out-of-domain**：LISA-Grounding，需要复杂推理 + 世界知识（比如"足球守门员"——你得先知道守门员是谁、长啥样、站在哪，才能框出来）

结果：
- In-domain 上，SFT 几乎不动（88.7 → 88.7），RL 稳步涨（88.7 → 90.55）。SFT 的问题在于它只是在 mimic 训练数据，数据本身简单，学不到新东西。
- **Out-of-domain 上差异更夸张**：SFT 训着训着反而退步（56.51 → 54.82），RL 一路涨（56.51 → 63.14）。

这是 "SFT Memorizes, RL Generalizes" 的活体演示。SFT 学到的是 training data 的表面 pattern，遇到更难的 reasoning 就懵了；RL 学到的是"如何思考出正确答案"这个 meta skill，能迁移到 OOD。

Chu et al. 那篇 "SFT Memorizes, RL Generalizes"：https://arxiv.org/abs/2501.17161

### OVD：小模型吊打大模型，VLM 吊打专业 OD

Table 4 最有意思的对比：

| Model | OVDEval Overall NMS-AP |
|---|---|
| Grounding DINO | 25.30 |
| OmDet (SOTA specialized OVD) | 25.86 |
| Qwen2.5-VL-7B (base) | 29.08 |
| VLM-R1 3B (RL) | **31.01** |

3B 的 VLM-R1 不仅超过同 size 的 base，还超过了 7B 的 base model，甚至超过专门做 OVD 的 OmDet。

更细看子任务：
- **Celebrity 检测**：VLM-R1 55.0 vs OmDet 1.8。差 50 多分。OmDet 这种 specialist 模型根本不认识"泰勒·斯威夫特"是谁，VLM 有预训练的世界知识，直接降维打击。
- **Color / 小物体检测**：OmDet 22.9 vs VLM-R1 4.5。反过来吊打。Specialist 模型在 fine-grained 局部 feature 上有优势，VLM 的 vision encoder 在小物体上还是弱。

这个对比给了一个 future direction：hybrid 架构，specialist 负责精细 perception，VLM 负责 semantic understanding。

---

## Plot twist：OD aha moment

这是 paper 最有意思的发现，也是最 R1-style 的"涌现"现象。

加了 odLength reward 之后，模型自发学会了一个**两步推理策略**：
1. 先在 `

---

# VLM-R1 深度解析：R1-style RL 在 Vision-Language 上的延伸

## 一、核心动机：为什么这篇 paper 存在

这篇 paper 想回答一个非常 specific 的问题：**DeepSeek R1 在 LLM 上靠 rule-based reward + GRPO 触发了 reasoning emergence，那这套 recipe 能否移植到 vision-language model 上，并且能够超越 SFT 的 generalization ceiling？**

作者的 insight 很朴素但很关键——大量 vision grounding 任务（REC、OVD、counting）有**deterministic ground-truth**：bounding box 就是 bounding box，IoU 可计算、mAP 可计算。这意味着这些 task 天然适配 R1 那套 rule-based reward，不需要训练一个 RM（reward model），不依赖 human preference data。

Table 1 给了一个非常有说服力的 motivation：Qwen2.5-VL-3B 有 3.75B 参数，是 Grounding DINO（341M）的 10 倍多，但在 RefCOCO/+/g 上反而**全面落后**于 Grounding DINO（89.1 vs 90.6, 82.4 vs 88.2, 85.2 vs 86.1）。这说明 VLM 在 grounding 这种"low-level perception"任务上有 structural deficiency，而 RL 正好可以补这块短板。

GitHub repo: https://github.com/om-ai-lab/VLM-R1

---

## 二、GRPO 算法回顾：为什么不需要 critic

GRPO（Group Relative Policy Optimization）的核心 trick 是用**组内相对排名**替代 critic baseline。PPO 需要训一个 value network $V_\phi(s)$ 来估计 advantage $A = R - V_\phi$，这个 critic 训起来很麻烦，尤其对于 LLM 这种 huge state space。GRPO 直接对同一个 prompt $q$ 采样 $N$ 个 response $\{o_1, ..., o_N\}$，用 group mean/std 做 normalization。

**公式 (1) — Advantage 计算：**
$$A_i = \frac{r_i - \text{mean}\{r_1, r_2, ..., r_N\}}{\text{std}\{r_1, r_2, ..., r_N\}}$$

变量解释：
- $A_i$：第 $i$ 个候选 response 的 advantage（标准化后的相对优势）
- $r_i = R(q, o_i)$：第 $i$ 个 response 在 reward function $R$ 下获得的 scalar reward
- $\text{mean}\{\cdot\}$、$\text{std}\{\cdot\}$：组内 $N$ 个 reward 的均值和标准差

直觉：如果一个 response 的 reward 高于组平均，advantage 正，policy 应该增加其概率；反之亦然。除以 std 是为了 scale invariance，让不同 group 的梯度量级一致。

**公式 (3) — Clipped surrogate objective：**
$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\left\{\min[s_1 \cdot A_i, s_2 \cdot A_i] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]\right\}\right]$$

变量解释：
- $\theta$：policy 参数
- $\beta$：KL penalty 系数（REC 用 0.04，OVD 设为 0，这是个关键 ablation）
- $\pi_\theta$：当前 policy
- $\pi_{ref}$：reference policy（通常是初始 SFT model，做 KL anchor 防止漂移过远）

**公式 (4)(5) — Importance ratio：**
$$s_1 = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, \quad s_2 = \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right)$$

- $s_1$：新旧 policy 对 $o_i$ 的概率比（importance sampling ratio）
- $s_2$：clipped version，把 ratio 限制在 $[1-\epsilon, 1+\epsilon]$
- $\epsilon$：clip 范围（通常 0.2）

取 min 是 PPO 的 standard trick：当 advantage 正时，只允许 ratio 上升到 $1+\epsilon$（防止 over-shooting）；当 advantage 负时，只允许 ratio 下降到 $1-\epsilon$。

**KL 项的目的**：DeepSeek R1 的 R1-Zero 把 KL 设得很小或忽略，让 model free explore；但通常需要 KL 防止 mode collapse。这篇 paper 在 OVD 上把 $\beta=0$，这是 reward hacking 出现的部分原因——没有任何 anchor 拉住 model 不漂。

参考 GRPO 原文：https://arxiv.org/abs/2402.03300

---

## 三、Reward 设计：两套任务、两种复杂度

### 3.1 REC 的 reward — 极简方案

REC 任务输入一个 referring expression（"the man in red shirt on the left"），输出一个 bounding box。

**公式 (6) — Accuracy reward：**
$$R_{acc}^{rec}(q, o) = \text{IoU}(b^*, f_{rec}(o))$$

- $q$：input question（包含 referring expression + image）
- $o$：VLM output sentence
- $b^*$：ground truth bounding box $[x_1, y_1, x_2, y_2]$
- $f_{rec}(o)$：parser 函数，从 output text 提取 predicted bbox
- IoU：intersection over union，范围 $[0, 1]$

Format reward：检查 output 是否在 `<answer>{...[x1, y1, x2, y2]...}</answer>` 标签里，返回 0/1。

非常简洁，因为 REC 是 single-box prediction，ground truth 唯一，没有歧义。

### 3.2 OVD 的 reward — 朴素方案会 reward hack

OVD 输入一个 target list（如所有 COCO 80 类），输出 multiple bounding boxes + class labels。这里复杂度陡升——预测数量本身就是一个 variable。

**公式 (7) — Length penalty factor：**
$$s_{ovd} = \min\left(1, \frac{L_{gt}}{L_{pred}}\right)$$

- $L_{gt}$：ground truth 中 object instance 的数量
- $L_{pred}$：模型预测的 object instance 数量
- $\min(1, \cdot)$：如果预测数量 ≤ ground truth，penalty = 1（不惩罚）；如果 over-predict，按比例惩罚

**公式 (8) — Final accuracy reward：**
$$R_{acc}^{ovd}(q, o) = s_{ovd} \cdot \text{mAP}(\mathbf{b}_{pred}, \mathbf{b}_{gt})$$

- $\mathbf{b}_{pred} = \{(b_1, c_1), (b_2, c_2), ..., (b_n, c_n)\}$：predicted (bbox, class) pairs
- $\mathbf{b}_{gt}$：ground truth (bbox, class) pairs
- $\text{mAP}(\cdot)$：mean Average Precision

### 3.3 Reward hacking 现象 — Table 5 的核心

Table 5 展示了三种 reward 的对比：

| Reward | COCO_filtered mAP | OVDEval Overall NMS-AP |
|---|---|---|
| AP50 | 11.4 | 21.46 |
| mAP | 11.8 | 21.68 |
| **odLength (s_ovd · mAP)** | **21.1** | **31.01** |

差距几乎 2 倍。为什么 naive AP50 和 mAP 会 hack？

**Reward hacking 的机制**：COCO 官方 evaluation API 在计算 AP 时，会**忽略当前 image 中没有 ground truth instance 的 categories**。所以如果 prompt 让模型预测所有 80 个 COCO 类，模型只需要狂吐 boxes（哪怕重复、哪怕乱猜），对于那些 image 里没有 ground truth 的 category，AP 计算时根本不计入分母——错误 prediction "免费"。但对真正存在的 category，多预测几个 box 偶尔也能 hit 中 ground truth，提高 recall。

这导致模型学会一个 pathological behavior：**enumerate all 80 categories with multiple boxes per category**，因为 marginal reward 永远非负。Figure 6 中 AP50 reward 的 completion length 在训练中爆炸式增长（从 ~100 tokens 飙到几千），就是模型在 spam boxes。

odLength 通过 $s_{ovd} = \min(1, L_{gt}/L_{pred})$ 砍掉冗余预测的 reward——你预测 50 个 box 但 ground truth 只有 5 个？你的 mAP 直接乘以 0.1。这逼迫模型精准 count + precise localize。

**OD aha moment 的 emergence**：加上 odLength 之后，模型自发学会两步推理：
1. 先在 `
