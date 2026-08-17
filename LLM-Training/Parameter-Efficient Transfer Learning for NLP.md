---
source_pdf: Parameter-Efficient Transfer Learning for NLP.pdf
paper_sha256: 7889e34caf5b786c378951be4b2f391234f5ff05ed61cba237b8281e3c94974b
processed_at: '2026-08-06T02:21:17-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Parameter-Efficient Transfer Learning for NLP — 用人话讲

paper: https://arxiv.org/abs/1902.00751
code: https://github.com/google-research/adapter-bert
后续 LoRA: https://arxiv.org/abs/2106.09685
后续 AdapterHub: https://adapterhub.ml/

---

## 一句话版本

BERT 已经把"通用语言知识"学到 weights 里了，每个 downstream task 其实只需要做一点点"小修小补"，根本不需要重新训整个 330M 参数的网络。塞几个 tiny bottleneck module 进去，frozen 主网络只训这些 module，性能跟 full fine-tune 几乎一样，参数省 97%。

---

## 故事的起点：fine-tuning 其实很浪费

假设你跑了 Google Cloud，客户一个个排队来：要做 sentiment analysis、要做 NER、要做 QA、要做 spam detection……每个 task 都要一个 BERT_LARGE。BERT_LARGE 有 330M 参数，9 个 GLUE task 就要存 9 份 = ~3B 参数。这还没算客户不断来新 task 的情况。

更糟的是，每个 task 你都要独立 fine-tune 整个网络，旧 task 的能力就忘了 — 这就是 catastrophic forgetting。Continual learning 文献里这个 struggle 了三十年 (McCloskey & Cohen, 1989, https://psycnet.apa.org/record/1990-12243-001) 都没彻底解决。

Houlsby 这篇 paper 提了个很优雅的思路：**freeze 主网络，只加 tiny 适配器**。这样每个 task 只存一份小 adapter，主网络大家共用，新 task 不影响老 task。

---

## 三种 transfer 范式，一图看懂

设 pre-trained network 是 $\phi_{\mathbf{w}}(\mathbf{x})$，参数 $\mathbf{w}$。

**Feature-based transfer**（ELMo 时代的做法）：
$$\hat{\mathbf{y}} = \chi_{\mathbf{v}}(\phi_{\mathbf{w}}(\mathbf{x}))$$

主网络 frozen 当 feature extractor，只训新的 head $\mathbf{v}$。问题：中间层完全不能 adapt，表达能力受限。

**Full fine-tuning**（BERT 时代的标准做法）：
$$\hat{\mathbf{y}} = \phi_{\mathbf{w}'}(\mathbf{x})$$

整个 $\mathbf{w}$ 复制一份 → $\mathbf{w}'$，全部训。每 task 占 $|\mathbf{w}|$ 参数，参数效率最差。

**Adapter tuning**（这篇 paper）：
$$\hat{\mathbf{y}} = \psi_{\mathbf{w}, \mathbf{v}}(\mathbf{x}), \quad \text{where } \psi_{\mathbf{w}, \mathbf{v}_0}(\mathbf{x}) \approx \phi_{\mathbf{w}}(\mathbf{x})$$

主网络 $\mathbf{w}$ frozen，新加参数 $\mathbf{v}$ 初始化成 "near-identity"，训练时只动 $\mathbf{v}$。约束：$|\mathbf{v}| \ll |\mathbf{w}|$。

直觉是什么？pre-trained model 已经在 good minimum 附近了，downstream task 只需要在这个 minima 周围做 small perturbation，不需要 rewiring 整个网络。这个观察后来被 Aghajanyan et al. 2020 (https://arxiv.org/abs/2012.13255) 形式化为 "intrinsic dimension" 假设，也是 LoRA 的理论基础。

---

## Adapter 长什么样：一个超简单的 bottleneck

整个 module 就一行公式：

$$\mathbf{h} = \mathbf{x} + W_{up} \cdot f(W_{down} \cdot \mathbf{x})$$

变量含义：
- $\mathbf{x} \in \mathbb{R}^d$：输入，$d$ 是 hidden size（BERT_BASE $d=768$，BERT_LARGE $d=1024$）
- $W_{down} \in \mathbb{R}^{m \times d}$：down-projection，把 $d$ 维压到 $m$ 维
- $m$：bottleneck 维度，**远小于** $d$，paper 里试 $\{2, 4, 8, 16, 32, 64, 256\}$
- $f(\cdot)$：GeLU（跟 BERT 一致）
- $W_{up} \in \mathbb{R}^{d \times m}$：up-projection，还原回 $d$ 维
- $\mathbf{h} \in \mathbb{R}^d$：输出
- 末尾 $+\mathbf{x}$：internal skip-connection

参数量（含 bias）：$|\mathbf{v}| = 2md + d + m$

举个例子，BERT_BASE，$d=768$，$m=64$：
$$2 \times 64 \times 768 + 768 + 64 = 99{,}072 \text{ params}$$

而 BERT_BASE 单层 attention+FFN 大约 7M 参数。所以一个 adapter 大概是单层的 1.4%。整个 BERT_BASE 12 层 × 2 个 adapter = 24 个 adapter，总共约 2.4M 参数，占整个 model 的 2%。

---

## 两个关键设计决策，每个都有 intuition

### 决策 1：Near-identity 初始化

$W_{down}, W_{up}$ 用 truncated Gaussian 初始化，std = $10^{-2}$。

为什么这么重要？因为训练刚开始时：
$$W_{up} \cdot f(W_{down} \cdot \mathbf{x}) \approx \mathbf{0}$$
所以 $\mathbf{h} \approx \mathbf{x}$，整个 adapter-augmented network 行为 ≈ 原始 pre-trained network。

这意味着：
1. 训练第一步的 gradient 信号干净，因为起点就是 pre-trained minima
2. Adapter 慢慢"激活"，逐渐学会需要多少 deviation
3. 不会一上来就把 pre-trained features 破坏掉

paper 在 Section 3.6 做了 robustness 实验：std < $10^{-2}$ 性能稳定，std > $10^{-2}$ 开始 degrade，std = 1 直接崩。这跟 continual learning 文献里的 "stability vs plasticity dilemma" 是一个道理 — 你要够 stable 来保护旧知识，又要够 plastic 来学新东西。Near-identity 初始化把 plasticity 起点设得很低，让训练自己决定要不要"激活"plasticity。

### 决策 2：插在每层 Transformer 的两个位置

每个 Transformer layer 长这样（Vaswani et al., 2017, https://arxiv.org/abs/1706.03762）：

```
x → MultiHeadAttention → projection → [Adapter_1] → +x → LayerNorm → x'
x' → FFN → projection → [Adapter_2] → +x' → LayerNorm → x''
```

每个 Transformer layer 插 2 个 adapter：一个在 attention 之后，一个在 FFN 之后。BERT_BASE 12 层 → 24 个 adapter。BERT_LARGE 24 层 → 48 个 adapter。

为什么插在每层，而不是只加在最后？因为 paper 试过 fine-tune 只 top-k 层（Table 2 的 "variable fine-tuning" baseline），性能明显不如 adapter。distributed 的 small modification 比 concentrated 的 large modification 更高效。这跟 Yosinski et al. 2014 (https://papers.nips.cc/paper/5767) 在 vision 上的观察一致：lower layers 通用，higher layers task-specific，但要 fully adapt 还是要每层都动一点。

---

## 顺便：LayerNorm 也要 task-specific

除了 adapter，每层 LayerNorm 的 $\gamma_l, \beta_l \in \mathbb{R}^d$ 也 task-specific 训练。这灵感来自 conditional batch norm (De Vries et al., 2017, https://arxiv.org/abs/1707.00683) 和 FiLM (Perez et al., 2018, https://arxiv.org/abs/1709.07871)。

每层只多 $2d$ 参数。Paper 单独做了 ablation：只训 LayerNorm（40k 参数 for BERT_BASE）在 CoLA 掉 3.5%，MNLI 掉 4%。LayerNorm 单独不够，但配合 adapter 有用 — 它给每层 feature 一个 task-specific 的 scale/shift，相当于"轻量级的整层 modulate"。

---

## 实验结果：性能差 0.4%，参数省 97%

### GLUE Benchmark (Table 1)

| Method | Total params | Trained/task | GLUE avg |
|---|---|---|---|
| BERT_LARGE Full FT | 9.0× | 100% | 80.4 |
| Adapter (size 8-256) | 1.3× | 3.6% | 80.0 |
| Adapter (size 64 fixed) | 1.2× | 2.1% | 79.6 |

数字说明一切：9 个 task 总参数从 9× 降到 1.3×，单 task 训练参数从 100% 降到 3.6%，性能只差 0.4%。

有意思的细节：不同 task 最优 adapter size 不同。MNLI 用 256（数据多，可以多学点），RTE 用 8（数据少，大 adapter 会 overfit）。但固定 size=64 性能损失很小（0.4%），所以工程上懒得调就用 64。

### 17 个额外 classification task (Table 2)

| Method | Avg acc | Total params | Trained/task |
|---|---|---|---|
| No-BERT AutoML (搜了 10k+ 模型) | 72.7 | - | - |
| BERT_BASE Full FT | 73.7 | 17× | 100% |
| BERT_BASE Variable FT (top 52% layers) | 74.0 | 9.9× | 52.9% |
| BERT_BASE Adapters | 73.3 | 1.19× | 1.14% |

注意 Variable FT（只 fine-tune top ~52% layers）平均比 full FT 还好 0.3% — 这暗示小数据上 fine-tune 全网络反而 overfit。Adapter 跟 full FT 差 0.4%，但每 task 只加 1.14% 参数，17 个 task 总共 1.19×（vs 17×）。

### SQuAD v1.1 (Figure 5)

| Method | F1 | Params/task |
|---|---|---|
| Full FT | 90.7 | 100% |
| Adapter (size=64) | 90.4 | 2% |
| Adapter (size=2) | 89.9 | 0.1% |

哪怕 adapter size=2（0.1% 参数），F1 还有 89.9。这点挺惊人的 — extractive QA 这种 span prediction 任务对参数修改也很 sensitive，但仍然在 low-rank 子空间内能搞定。

---

## Ablation：哪些 adapter 重要？

paper 做了个聪明的实验：训完后**移除**部分 adapter（不重训），看 validation accuracy 掉多少。结果（Figure 6 heatmap）：

- 移除**单层** adapter：性能掉 ≤2%
- 移除 layer 0-4（lower layers）：**几乎不掉**（MNLI 上几乎 0）
- 移除**所有** adapter：MNLI → 37%（majority class baseline），CoLA → 69%
- 移除 higher layer adapter：影响显著

**Insight**：adapter 自动学会 focus on higher layers。低层的 adapter 训完几乎没用 — 因为低层 features 跨 task 通用，不需要 adapt。这跟 Howard & Ruder 2018 (https://arxiv.org/abs/1801.06146) ULMFiT 里 "discriminative fine-tuning"（高层学习率大、低层学习率小）的直觉一致。

---

## 跟后续 PEFT 工作的关系

这篇 paper 开启了整个 PEFT (Parameter-Efficient Fine-Tuning) 方向。后来者：

**LoRA** (Hu et al., 2021, https://arxiv.org/abs/2106.09685)：把 adapter 简化成纯线性 $\mathbf{h} = W\mathbf{x} + BA\mathbf{x}$，砍掉 nonlinearity 和 skip-connection。最大优势：训练完可以把 $BA$ merge 回 $W$，inference 时零 latency。但表达力弱于 Houlsby adapter。

| 特性 | Houlsby Adapter | LoRA |
|---|---|---|
| Nonlinearity | 有 GeLU | 无 |
| Skip-connection | 有 | 无（直接加到 weight） |
| Inference | 不能 merge，慢 | 可 merge，零 latency |
| 形式 | $\mathbf{x} + W_{up} f(W_{down} \mathbf{x})$ | $W\mathbf{x} + BA\mathbf{x}$ |

**Prefix tuning** (Li & Liang, 2021, https://arxiv.org/abs/2101.00190)：在每层 attention 的 key/value 前加 prefix vectors。只 modulate attention distribution，不改变 feature transformation，表达力弱于 adapter。

**Prompt tuning** (Lester et al., 2021, https://arxiv.org/abs/2104.08691)：只在 input embedding 前加 soft prompt tokens，连中间层都不动。参数最少，但小 model 上效果差。

**AdapterHub** (Pfeiffer et al., 2020, https://aclanthology.org/2020.emnlp-demos.7/)：建立 adapter 仓库，训好的 adapter 可以复用、组合。**AdapterFusion** (https://arxiv.org/abs/2005.00247) 探索了能不能把多个 task adapter linearly combine 出新 task。

**Compacter** (Mahabadi et al., 2021, https://arxiv.org/abs/2106.04647)：用 Kronecker product 参数化 adapter，进一步压缩参数量。

---

## 几个 build intuition 的点

### 1. 为什么 bottleneck low-rank modification 够用？

Pre-trained model 的 weights 已经在 pretraining 任务上 well-conditioned。Downstream task 跟 pretraining 任务之间的 "task shift" 在 feature space 上通常是 **low-rank** 的。$m$ 维 bottleneck 相当于把 modification 限制在 $m$-rank 子空间，对于大多数 NLP 任务，$m=64$ 已经远超实际需要的 intrinsic dimension。

打个比方：pre-trained model 像一个训练有素的厨师，会做各种菜。你要让他专精川菜，不需要重学刀工火候，只需要调整一下调料配比。Adapter 就是那瓶调料。

### 2. 为什么插在每层，而不是只加在最后？

做过 fine-tune 的人都知道，只训 top layer 性能很差（Table 2 的 variable FT baseline）。因为 task-specific 信息需要渗透到中间层 — 你不能只改 output，得让中间层 features 也 adapt 到新 task。Adapter 加在每层，等于"分布式地微调每层 output"，比集中改 top layer 高效得多。

### 3. 为什么 near-identity 初始化这么关键？

Pre-trained network 的 loss landscape 在一个 sharp minima 附近。Random init 的 adapter 会立刻把 activation distribution 推离这个 minima，gradient 信号混乱。Near-identity 相当于在 minima 附近做 small perturbation，gradient 来自 pre-trained model 已经 well-conditioned 的 neighborhood，训练稳定。

### 4. 为什么不用更复杂的 adapter 结构？

paper Section 3.6 试了一堆变体：加 batch norm、多层 adapter、不同 activation、parallel adapter 加 multiplicative interaction……都没显著提升。这是个好的工程信号 — bottleneck + skip + near-identity 这个组合已经足够 capture task shift 了，复杂度换不来性能。后来 LoRA 把它进一步简化（甚至砍掉 nonlinearity）也能 work，进一步印证这点。

### 5. 跟 multi-task learning 的本质区别

MTL 同时训所有 task，参数 share 在 lower layers，high-level task-specific。Adapter 完全不需要同时访问所有 task：

- 天然支持 online setting（task 一个个来）
- 不存在 task interference / negative transfer
- 新 task 不影响旧 task（zero catastrophic forgetting）
- 代价：每个 task 都要单独训练一次（vs MTL 一次训完）

对 cloud service 场景特别合适 — 客户 task 一个个来，你不可能等所有客户 task 攒齐了再 MTL。

---

## Limitations

1. **Inference latency**：adapter 不能 merge 回 backbone，每层多两次 matmul。虽然 matmul 很小（$d \times m \times d$，$m=64$），但 24 层 × 2 个 = 48 次额外 matmul 还是会有 overhead。后期 LoRA 通过纯线性 + merge 解决了这个。

2. **只测了 classification + extractive QA**：没测 generation、summarization、translation。不过后来 AdapterHub 在更多 task 上验证了有效性。

3. **没探 adapter compositionality**：能不能把两个 task 的 adapter linearly combine 出新 task？这是后续 AdapterFusion 探索的。

4. **没用 Encoder-Decoder model**：只测了 BERT encoder。T5/BART 上行为是否一致未探（后来 Pfeiffer et al. 在 T5 上验证了）。

---

## 工程上的实用 takeaway

1. 默认 adapter size = 64 是个 robust 选择，性能损失 < 0.5%
2. 必须用 near-zero 初始化（std = $10^{-2}$ 左右）
3. 记得每层 LayerNorm 也要 task-specific 训
4. 小数据任务用小 adapter（size=8）防 overfit
5. 不要过度设计 adapter 内部结构，simple bottleneck 就够

后期 LoRA 在工程上更胜一筹（可 merge、纯线性），但 Houlsby adapter 的 bottleneck + nonlinearity 设计在表达力上仍然有优势。在资源充裕、不 care inference latency 的场景下，Houlsby adapter 仍然是个 strong baseline。

---

# Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019) 详解

Paper link: https://arxiv.org/abs/1902.00751
Original blog / code (Google Research): https://github.com/google-research/adapter-bert
后续 LoRA 论文 (相关对照): https://arxiv.org/abs/2106.09685
后续 AdapterHub (社区生态): https://adapterhub.ml/

---

## 1. 背景与动机：为什么需要 adapter

BERT 出来之后，标准 transfer 流程是 **fine-tuning**：把 pre-trained weights $\mathbf{w}$ 复制一份，整个网络在 downstream 数据上 jointly 训练。问题在于：

- BERT_LARGE 有 ~330M 参数
- 假设 cloud service 上来了 $N$ 个 task，就要存 $N$ 份完整 weights
- GLUE 9 个 task → 9× = ~3B 参数
- 17 个 classification task → 17× 参数

而且新 task 来了之后，如果要 fine-tune，会 **catastrophic forgetting** 之前 task 的能力（参数被改了）。这篇 paper 想解决两件事：

1. **Compactness**: 每个 task 只加一点点 task-specific 参数
2. **Extensibility**: 新 task 来了不影响老 task，可以无限累加

跟 multi-task learning (MTL) 不同，adapter 不需要同时访问所有 task 的数据；跟 continual learning 不同，adapter 不存在 forgetting，因为 shared backbone 完全 frozen。

---

## 2. 数学形式化：三种 transfer 范式

设 pre-trained network 为 $\phi_{\mathbf{w}}(\mathbf{x})$，参数为 $\mathbf{w}$，输入 $\mathbf{x}$。

### 2.1 Feature-based transfer
$$\hat{\mathbf{y}} = \chi_{\mathbf{v}}\!\left(\phi_{\mathbf{w}}(\mathbf{x})\right)$$

- $\mathbf{w}$ frozen
- 只训练 $\mathbf{v}$（新加的 head）
- 缺点：$\phi_{\mathbf{w}}$ 的 intermediate features 完全不动，无法 adapt 中间层

### 2.2 Full fine-tuning
$$\hat{\mathbf{y}} = \phi_{\mathbf{w}'}(\mathbf{x})$$

- 对每个 task 复制一份 $\mathbf{w} \to \mathbf{w}'$
- 整个 $\mathbf{w}'$ 都训
- 每 task 占用 $|\mathbf{w}|$ 参数

### 2.3 Adapter tuning
$$\hat{\mathbf{y}} = \psi_{\mathbf{w}, \mathbf{v}}(\mathbf{x})$$

- $\mathbf{w}$ 从 pre-training 拷贝过来，**frozen**
- 引入新的 task-specific 参数 $\mathbf{v}$，初始化为 $\mathbf{v}_0$ 使得
  $$\psi_{\mathbf{w}, \mathbf{v}_0}(\mathbf{x}) \approx \phi_{\mathbf{w}}(\mathbf{x})$$
- 训练时只更新 $\mathbf{v}$
- 关键约束：$|\mathbf{v}| \ll |\mathbf{w}|$

**核心 intuition**：pre-trained model 已经把通用知识编码在 $\mathbf{w}$ 里，新 task 通常只需要做 **small deviation from identity**，不需要 rewiring 整个网络。这跟后期 LoRA 的 "intrinsic dimension" 假设一致 (Aghajanyan et al., 2020, https://arxiv.org/abs/2012.13255)。

---

## 3. Adapter Module 架构

### 3.1 Bottleneck 结构

每个 adapter 是一个 small bottleneck：

$$
\mathbf{h} = \mathbf{x} + W_{up} \cdot f\!\left(W_{down} \cdot \mathbf{x}\right)
$$

变量说明：
- $\mathbf{x} \in \mathbb{R}^d$：输入向量，$d$ 是 model hidden size（BERT_BASE $d=768$，BERT_LARGE $d=1024$）
- $W_{down} \in \mathbb{R}^{m \times d}$：down-projection 矩阵，把 $d$ 维压到 $m$ 维
- $m$：bottleneck dimension，**远小于** $d$（paper 里 $m \in \{2, 4, 8, 16, 32, 64, 256\}$）
- $f(\cdot)$：nonlinearity（paper 用 GeLU/ReLU，跟 BERT 保持一致）
- $W_{up} \in \mathbb{R}^{d \times m}$：up-projection 矩阵，把 $m$ 维还原回 $d$ 维
- $\mathbf{h} \in \mathbb{R}^d$：输出向量
- 末尾的 $+\mathbf{x}$ 是 internal skip-connection

**参数量**（包含 bias）：
$$|\mathbf{v}|_{\text{adapter}} = 2md + d + m$$

举例：$d=768$，$m=64$ 时：
- $2 \times 64 \times 768 + 768 + 64 = 99,072$ 参数
- BERT_BASE 单层 attention+FFN 大约 7M 参数
- 所以 adapter 大约占单层 1.4%

### 3.2 Near-identity 初始化

$W_{down}$ 和 $W_{up}$ 用 truncated normal 初始化，std = $10^{-2}$，截断到 2 个 std。

为什么这么重要？因为 $W_{up} \cdot f(W_{down} \cdot \mathbf{x})$ 在初始化时数值很小，加上 skip-connection 后 $\mathbf{h} \approx \mathbf{x}$。这意味着：

- 训练开始时，整个 adapter-augmented network 行为 ≈ 原始 pre-trained network
- Gradient 信号干净，不会破坏 pre-trained features
- Adapter 慢慢"激活"，逐渐 learn 需要多少 deviation

Paper 在 Section 3.6 验证：std > $10^{-2}$ 后性能开始 degrade，太大就训不动了。

### 3.3 在 Transformer 中的插入位置

Transformer 单层结构（Vaswani et al., 2017）：

```
x → MultiHeadAttention → Linear projection → [Adapter_1] → +x (skip) → LayerNorm → x'
x' → FFN(2 layers) → Linear projection → [Adapter_2] → +x' (skip) → LayerNorm → x''
```

关键点：adapter 插在 **sub-layer projection 之后、skip-connection 相加之前、LayerNorm 之前**。每个 Transformer layer 插 2 个 adapter。

BERT_BASE 12 层 → 24 个 adapter
BERT_LARGE 24 层 → 48 个 adapter

### 3.4 额外训练的参数：LayerNorm

除了 adapter，每层的 LayerNorm 参数 $\gamma_l, \beta_l \in \mathbb{R}^d$ 也 task-specific 训练。这是 conditional normalization 的简化版，类似 FiLM (Perez et al., 2018, https://arxiv.org/abs/1709.07871) 和 conditional batch norm (De Vries et al., 2017, https://arxiv.org/abs/1707.00683)。

每层 LayerNorm 只多 $2d$ 参数。Paper 单独做了 ablation：只训 LayerNorm 参数（40k 参数 for BERT_BASE）在 CoLA 上掉 3.5%，MNLI 掉 4%。说明 LayerNorm 单独不够，但跟 adapter 配合有用。

---

## 4. 实验设计

### 4.1 Base model
- GLUE：BERT_LARGE (24 layers, 330M params)
- 其他 classification：BERT_BASE (12 layers, ~110M params)
- SQuAD：BERT_BASE

### 4.2 Training details
- Optimizer：Adam (Kingma & Ba, 2014, https://arxiv.org/abs/1412.6980)
- Learning rate schedule：linear warmup (前 10% steps) → linear decay to 0
- Batch size：32
- Hardware：4 Google Cloud TPUs
- 学习率 sweep：$\{3 \times 10^{-5}, 3 \times 10^{-4}, 3 \times 10^{-3}\}$
- Adapter size sweep：$\{2, 4, 8, 16, 32, 64, 256\}$
- Random seeds：5 个，选 validation 最好的

### 4.3 Baselines
1. **Full fine-tuning**：训所有参数
2. **Variable fine-tuning**：只 fine-tune top-$n$ 层，$n \in \{1, 2, 3, 5, 7, 9, 11, 12\}$
3. **LayerNorm only**：只训 LayerNorm 参数
4. **AutoML baseline (no BERT)**：用 single-task Neural AutoML 在 text embeddings (TensorFlow Hub) 上搜 10k+ 模型，跑 1 周 CPU × 30 machines

---

## 5. 关键实验结果

### 5.1 GLUE Benchmark (Table 1)

| Method | Total params | Trained / task | CoLA | SST | MRPC | STS-B | QQP | MNLI_m | MNLI_mm | QNLI | RTE | **Total** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BERT_LARGE FT | 9.0× | 100% | 60.5 | 94.9 | 89.3 | 87.6 | 72.1 | 86.7 | 85.9 | 91.1 | 70.1 | **80.4** |
| Adapter (8-256) | 1.3× | 3.6% | 59.5 | 94.0 | 89.5 | 86.9 | 71.8 | 84.9 | 85.1 | 90.7 | 71.5 | **80.0** |
| Adapter (64 fixed) | 1.2× | 2.1% | 56.9 | 94.2 | 89.6 | 87.3 | 71.8 | 85.3 | 84.6 | 91.4 | 68.8 | **79.6** |

**关键 takeaways**：
- Adapter 跟 full FT 差 0.4%（80.0 vs 80.4）
- 9 个 task 总参数从 9× 降到 1.3×
- 单 task 训练参数从 100% 降到 3.6%
- 不同 task 最优 adapter size 不同：MNLI 用 256，RTE (small dataset) 用 8 — 小数据用小 adapter 防止 overfit
- 固定 size=64 性能损失小（0.4%）

### 5.2 Additional 17 Classification Tasks (Table 2)

| Method | Avg acc | Total params | Trained/task |
|---|---|---|---|
| No-BERT AutoML | 72.7 | - | - |
| BERT_BASE FT | 73.7 | 17× | 100% |
| BERT_BASE Variable FT | 74.0 | 9.9× | 52.9% |
| BERT_BASE Adapters | 73.3 | 1.19× | 1.14% |

注意 **Variable FT（只 fine-tune top ~52% layers）平均比 full FT 还好 0.3%**！这暗示小数据上 fine-tune 全网络反而 overfit。

Adapter 跟 full FT 差 0.4%，但每个 task 只加 1.14% 参数，17 个 task 总共 1.19×（vs 17×）。

### 5.3 SQuAD v1.1 (Figure 5)

| Method | F1 | Params/task |
|---|---|---|
| Full FT | 90.7 | 100% |
| Adapter (size=64) | 90.4 | 2% |
| Adapter (size=2) | 89.9 | 0.1% |

**Insight**：哪怕 adapter size=2（0.1% 参数），F1 还有 89.9，只比 full FT 差 0.8。说明 QA 这类 span prediction 任务对 parameter 修改也很 sensitive，但 still 在 low-rank 子空间内。

### 5.4 Parameter/Performance Trade-off (Figure 3, 4)

对 MNLI_m：
- Full FT: 84.4% ± 0.02 (110M params trained)
- Top-1-layer FT: 77.8% ± 0.1 (~9M params trained)
- Adapter size=64: 83.7% ± 0.1 (~2M params trained)

**关键观察**：fine-tune top-k layers 时，参数量跟 adapter 同档位，性能差一大截。说明 adapter 加在每层（分布式的 low-rank 修改）比集中 fine-tune top layers 更高效。

---

## 6. Ablation & 分析 (Section 3.6)

### 6.1 哪些层的 adapter 重要？(Figure 6 heatmap)

实验：训完后**移除**部分 adapter（不重训），看 validation accuracy 掉多少。

结果：
- 移除**单层**的 adapter：性能掉 ≤2%
- 移除 layer 0-4 的 adapter（lower layers）：**几乎不掉**（MNLI 上几乎 0）
- 移除**所有** adapter：MNLI → 37%（majority class），CoLA → 69%（majority class）
- 移除 higher layer adapter：影响显著

**Insight**：adapter 自动学会 focus on higher layers。这跟视觉领域 Yosinski et al. (2014, https://papers.nips.cc/paper/5767) 的 "transferability of features" 经典观察一致 — low-level features 跨 task 通用，high-level features task-specific。Adapter 只在需要的地方"激活"。

### 6.2 初始化 scale robustness (Figure 6 right)

测试 std ∈ $[10^{-7}, 1]$：
- std < $10^{-2}$：性能稳定
- std > $10^{-2}$：开始 degrade
- std = 1：CoLA 大幅下降

证实 near-identity 初始化的关键性。

### 6.3 Adapter size robustness

| Adapter size | 8 | 64 | 256 |
|---|---|---|---|
| Avg val acc (8 tasks) | 86.2% | 85.8% | 85.7% |

跨 3 个数量级，性能变化 < 0.5%。说明 adapter 对 size 不敏感 — 这是个 robust hyperparameter。

### 6.4 试过的其他架构变体

Paper 试过都没显著提升：
- (i) 在 adapter 里加 batch/layer norm
- (ii) 多层 adapter（deeper bottleneck）
- (iii) 不同 activation（tanh 等）
- (iv) 只在 attention 里插
- (v) parallel adapter + multiplicative interaction

结论：**最简单的 bottleneck 就够了**。这是好的工程信号 — 复杂度不一定换来性能。

---

## 7. 帮你 build intuition 的几个点

### 7.1 为什么 bottleneck low-rank modification 够用？

Pre-trained model 的 weights 已经在 pretraining 任务上 well-conditioned。Downstream task 跟 pretraining 任务之间的 "task shift" 在 feature space 上通常是 **low-rank** 的（参考 Aghajanyan et al. 2020 关于 intrinsic dimension 的研究 https://arxiv.org/abs/2012.13255）。$m$ 维 bottleneck 相当于把 modification 限制在 $m$-rank 子空间，对于大多数 NLP 任务，$m=64$ 已经远超实际需要的 intrinsic dimension。

### 7.2 跟 LoRA 的对比（后来 Microsoft 2021 的工作 https://arxiv.org/abs/2106.09685）

| 特性 | Houlsby Adapter | LoRA |
|---|---|---|
| 插入位置 | sub-layer output | attention 的 $W_q, W_v$ |
| 非 linear | 有 GeLU | 无（纯线性） |
| Skip-connection | 有 internal skip | 无（直接加到原 weight 上） |
| Inference | 不能 merge，慢 | 可 merge 回 $W$，零 latency |
| 形式 | $\mathbf{x} + W_{up} f(W_{down} \mathbf{x})$ | $W \mathbf{x} + BA\mathbf{x}$ |

LoRA 砍掉了 nonlinearity 和 skip，更 minimal；但 Houlsby adapter 提供了表达能力更强的 bottleneck transformation。两者本质都是 low-rank adaptation。

### 7.3 跟 Prompt Tuning / Prefix Tuning 的关系

- **Prompt tuning** (Lester et al., 2021, https://arxiv.org/abs/2104.08691)：只在 input embedding 前面加 soft prompt tokens，连中间层都不动。参数最少，但小 model 上效果差。
- **Prefix tuning** (Li & Liang, 2021, https://arxiv.org/abs/2101.00190)：在每层 attention 的 key/value 前面加 prefix vectors。比 prompt tuning 强，但表达能力仍弱于 adapter（prefix 只 modulate attention distribution，不改变 feature transformation）。
- **Adapter**：直接 modify sub-layer output，能改变 feature transformation 本身，表达能力最强。

### 7.4 Near-identity 初始化的深层原因

Pre-trained network 的 loss landscape 已经在一个 sharp minima 附近。Random init 的 adapter 会立刻把 activation distribution 推离这个 minima，gradient signal 混乱。Near-identity 相当于在 minima 附近做 small perturbation，gradient 来自 pre-trained model 已经 well-conditioned 的 neighborhood，训练稳定。这跟 **stability vs plasticity dilemma** (continual learning 文献) 的解决思路一致。

### 7.5 跟 multi-task learning 的本质区别

MTL 同时训所有 task，参数 share 在 lower layers。Adapter 完全不需要同时访问所有 task。这意味着：
- Adapter 天然支持 **online setting**（task 一个个来）
- 不存在 task interference / negative transfer
- 新 task 不影响旧 task（zero catastrophic forgetting）
- 代价：每个 task 都要单独训练一次（vs MTL 一次训完）

---

## 8. Limitations & 后续工作

### 8.1 这篇 paper 的 limitations

1. **Inference latency**：adapter 不能 merge 回 backbone，每层多两次 matmul。后期 LoRA 解决了这个。
2. **只测了 classification + extractive QA**：没测 generation、summarization、translation 等。
3. **没探 adapter compositionality**：能不能把两个 task 的 adapter linearly combine 出新 task？这是后续 AdapterFusion (Pfeiffer et al., 2020, https://arxiv.org/abs/2005.00247) 探索的。
4. **没用 Encoder-Decoder model**：只测了 BERT encoder。T5/BART 上行为是否一致未探。

### 8.2 后续 PEFT 工作谱系

- **AdapterHub** (Pfeiffer et al., 2020, https://aclanthology.org/2020.emnlp-demos.7/)：建立 adapter 仓库，复用训好的 adapter
- **AdapterFusion** (2020, https://arxiv.org/abs/2005.00247)：组合多个 task adapters
- **Compacter** (Mahabadi et al., 2021, https://arxiv.org/abs/2106.04647)：用 Kronecker product 参数化 adapter，进一步压缩
- **LoRA** (Hu et al., 2021, https://arxiv.org/abs/2106.09685)：简化为线性，可 merge
- **Prefix tuning** (Li & Liang, 2021, https://arxiv.org/abs/2101.00190)
- **Prompt tuning** (Lester et al., 2021, https://arxiv.org/abs/2104.08691)
- **BitFit** (Zaken et al., 2022, https://arxiv.org/abs/2106.10199)：只训 bias 项
- **DoRA** (Liu et al., 2024, https://arxiv.org/abs/2402.09353)：LoRA 改进版，decompose magnitude & direction

---

## 9. 公式变量总结表

| 符号 | 含义 | 维度 |
|---|---|---|
| $\mathbf{x}$ | adapter 输入 | $\mathbb{R}^d$ |
| $\mathbf{h}$ | adapter 输出 | $\mathbb{R}^d$ |
| $d$ | model hidden size | 768 (BASE) / 1024 (LARGE) |
| $m$ | bottleneck dim | $\{2, 4, 8, ..., 256\}$ |
| $W_{down}$ | down-projection | $\mathbb{R}^{m \times d}$ |
| $W_{up}$ | up-projection | $\mathbb{R}^{d \times m}$ |
| $f(\cdot)$ | nonlinearity (GeLU) | - |
| $\mathbf{w}$ | pre-trained backbone weights | frozen |
| $\mathbf{v}$ | task-specific adapter params | trainable |
| $\gamma_l, \beta_l$ | layer $l$ LayerNorm gain/bias | $\mathbb{R}^d$ each, trainable |

---

## 10. 我的整体评价

这篇 paper 的核心贡献是 **conceptual**：它证明了 "pre-trained model + tiny bottleneck modules" 就能 match full fine-tuning，开启了整个 PEFT 方向。技术实现简单到一天就能复现，但 idea 深远 — 它告诉我们 pre-trained model 内部已经包含了大部分 task 所需的 representation capacity，fine-tuning 100% 参数本质上是过度参数化的。

工程上最实用的 takeaway：
- 默认 adapter size = 64 是个 robust 选择
- 必须用 near-zero 初始化
- 记得每层 LayerNorm 也要 task-specific 训
- 小数据任务用小 adapter（如 size=8）防 overfit

后期 LoRA 在工程上更胜一筹（可 merge、纯线性），但 Houlsby adapter 的 bottleneck + nonlinearity 设计在表达力上仍然有优势，在某些 task 上仍然超过 LoRA。
