---
source_pdf: RETHINKING INVARIANCE IN IN-CONTEXT LEARNING.pdf
paper_sha256: a273059a2fde671bb6a7026e0c7edfc5cbaac8894e496854776c8ce81a324729
processed_at: '2026-08-11T23:08:30-07:00'
target_folder: LLM-from-scratch/Attention
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

## 问题是什么

你给LLM几个few-shot examples,让它预测一个新input。按道理这些examples是i.i.d.的,你打乱顺序喂进去,结果应该一样。但实际上GPT-3在SST-2上,换个顺序accuracy能从90%掉到50%。这很荒谬——**数据本身没变,只是顺序变了**。

根本原因:Transformer的causal mask让每个example只能看前面的,后面的看不见前面。所以同样5个examples,放在第1位和第5位,模型看到的context完全不同,预测自然不同。

## 三个想要的性质

作者说,好的invariant ICL要同时满足三条:

**1. Permutation invariance**: 打乱context examples顺序,预测结果不变。这是最直接的需求。

**2. Information non-leakage**: 当模型预测第i个example的label时,不能偷看第i个example的ground-truth label。听起来像废话,但Prefix ICL就违反了这条——所有examples互相看到对方的label,预测变成copy-paste,训练信号废了。

**3. Context interdependence**: 每个example的encoding应该能利用其他examples的信息。比如看到3个"正面情感"的例子,模型对第4个例子的encoding应该带有"这大概是情感分类任务"的context信息。BoE ICL就违反了这条——每个example独立encode,互不看对方,像bag-of-words。

## 现有方法为什么不行

**AR ICL**(标准GPT): 1和3满足(部分),2满足,但1不满足——causal mask天然破坏invariance。

**Prefix ICL**(T5风格): 1和3满足,但2不满足——所有context examples之间full attention,label互相可见,leakage。模型学shortcut直接抄答案。

**BoE ICL**(PCW/BatchICL): 1和2满足,但3不满足——每个example单独encode,没有interaction。像把每道例题单独看,不知道彼此的关系。

**没人同时满足三条**。这就是paper的出发点。

## 为什么单步attention做不到

作者用graph message passing的视角证明了一个挺漂亮的结论:

把self-attention看成graph上的message passing,mask $\mathbf{M}$决定哪些node能互相传消息。

- 要permutation invariance,$\mathbf{M}$必须满足 $\mathbf{T}\mathbf{M}\mathbf{T}^{-1} = \mathbf{M}$ 对任意permutation $\mathbf{T}$。这把 $\mathbf{M}$ 限制到三种:全0(full attention)、diagonal(只看自己)、off-diagonal(只看别人不看自己)。
- 要non-leakage,message-passing graph必须是DAG(有向无环图),否则信息会绕一圈传回来。DAG能topological sort成lower triangular。
- 两个条件取交集:只能选diagonal mask。而diagonal mask = BoE = 没interdependence。

**结论**: 单层message passing里,invariance + non-leakage必然force出BoE,force掉interdependence。这是表达力的硬限制。要破局,必须用multi-stage。

## InvICL的核心trick — Leave-One-Out

思路很直觉: **让每个example先独立encode一遍(防leakage),再用其他examples的encoding作为context来re-encode自己(获得interdependence)**。

具体两步:

**Stage 1**: 每个example单独encode,只用自己当attention target。得到 $\bar{\mathbf{h}}_i$。这一步保证non-leakage,但没interdependence。

**Stage 2**: 对第i个example,用**所有其他**examples的Stage 1 encoding(即 $\bar{\mathbf{h}}_j, j \neq i$)作为context,再encode自己一次,得到 $\mathbf{h}_i$。这步获得了interdependence——$\mathbf{h}_i$现在带着"其他examples告诉我这是什么task"的信息。

为什么叫leave-one-out?因为encode $\mathbf{x}_i$时,把 $\mathbf{y}_i$排除在外(用其他examples的label信息但不直接看自己的label)。这跟统计学里的jackknife一模一样——评估样本i时,用除i外的所有样本估task。

**Test example**: 最后 $\mathbf{x}_t$ attend所有Stage 2的 $\mathbf{h}_i$,aggregate出预测。

## 为什么三条都满足

- **Invariance**: Stage 1独立,Stage 2用"除自己外所有"这个集合(无论怎么permutation,集合内容不变),Stage 3 aggregate所有。全程对permutation不变。
- **Non-leakage**: $\mathbf{h}_i$的Stage 2 encoding只用了其他examples的 $\bar{\mathbf{h}}_j$($j \neq i$),这些 $\bar{\mathbf{h}}_j$来自Stage 1的独立encoding,不包含 $\mathbf{y}_i$信息。$\mathbf{x}_i$自己只用自己的input部分。
- **Interdependence**: Stage 2里每个example都吸收了其他examples的信息。

## 工程实现 — Unrolling trick

朴素实现要对每个example跑一次LOO forward pass,共 $n+1$次forward,太慢。

**Trick**: 把input sequence复制一遍,变成 $(\tilde{\mathbf{x}}_1, \ldots, \tilde{\mathbf{x}}_n, \tilde{\mathbf{x}}_1, \ldots, \tilde{\mathbf{x}}_n, \mathbf{x}_t)$,长度 $2n+1$。然后设计一个特殊的 $(2n+1) \times (2n+1)$ attention mask:

- **前n行**: diagonal mask,实现Stage 1(每个example只看自己)
- **中间n行**: LOO mask——第 $n+i$行(对应 $\tilde{\mathbf{x}}_i$的第二次出现)能attend到所有 $\bar{\mathbf{h}}_j$($j \neq i$)+ 自己,但**不能attend到第 $i$行**(防leakage)
- **最后一行**: test example attend所有Stage 2的 $\mathbf{h}_i$

一个forward pass搞定,complexity $O(n^2)$,和Prefix/AR同阶。实测inference time 22ms,跟AR的21.9ms几乎没差别。

这个unrolling思想跟chain-of-thought的思路类似——把sequential computation unroll成single forward pass,用mask控制信息流向。

## 实验亮点

**Synthetic linear regression**: 训练40 examples,测试10-100 examples。InvICL在整个区间都接近least squares optimal,AR/Prefix/BoE在length > 40时严重退化。InvICL收敛也快——50k epochs时只有InvICL学到,其他都还没学明白。

**MetaICL (GPT-2 Large)**: 142个NLP task上short fine-tuning。
- All target tasks: InvICL 42.4 vs AR 41.9 (4/7 task胜)
- **OOD (unseen domains): InvICL 48.4 vs AR 43.6,7/7 task全部胜**——这是最亮眼的,invariance prior在distribution shift下特别管用

**Ablation关键数据**: 只加symmetric PE反而下降5个点(因为破坏pretraining的locality prior),但加inv mask + sym PE后,sensitivity从0.25降到0.00——完美invariance,性能还提升。

**不同base model**: GPT-2(trainable PE)、GPT-Neo(Alibi)、Pythia(Rotary)上都work,说明不依赖specific PE类型。

## 理论bonus — ICL as Gradient Descent

在linear regression + 特殊parametrization下,作者证明InvICL每层近似执行一次gradient descent update:

$$\mathbf{w}_\ell = \mathbf{w}_{\ell-1} - \eta \mathbf{X}^\top(\mathbf{X}\mathbf{w}_{\ell-1} - \mathbf{y}) + \eta^2 \Delta \mathbf{w}_{\ell-1}$$

第一项是standard GD,第二项是LOO机制产生的二阶修正(来自off-diagonal Gram matrix)。当 $\eta$足够小时,二阶项被dominant,InvICL收敛到least squares解。

对比:
- AR ICL ≈ online GD(constant learning rate,不保证收敛)
- Prefix ICL ≈ standard GD,但leakage让训练信号废了
- BoE ICL: gradient永远在初始点算,不converge
- **InvICL**: 近似standard GD,无leakage,有interdependence——三者最优组合

## Intuition总结

1. **Permutation invariance不是装饰,是inductive bias**。数据有symmetry时,architecture respect这个symmetry能降低sample complexity,OOD generalization bound更紧。这解释为什么InvICL在OOD特别强。

2. **LOO = task从别人学,data从自己取**。这是概念上的separation——task信息来自其他examples告诉你"这是什么问题",data信息来自当前example的input。跟人类few-shot learning"先看例题找规律,再解当前题"的过程对应。

3. **Single-pass attention有表达力极限**。Prop 3.6是硬限制——invariance + non-leakage必然force出diagonal mask = BoE = 无interdependence。要破局必须multi-stage。

4. **Unrolling = architectural chain-of-thought**。把sequential computation unroll到single forward pass,用mask控制信息流。这暗示未来可能有更深unrolling实现更复杂算法。

5. **Dense learning signal很重要**。causal mask不只是为了autoregressive生成,它提供dense supervision——每个token位置都是prediction task。破坏这个(像Prefix)会让训练信号稀疏,即使architecture理论上能实现GD,实际训不出来。

一句话: 这篇paper用graph message passing的framework,精确定位了invariant ICL设计上的硬约束(单层做不到),然后用LOO pre-encoding巧妙破局,unrolling trick保持效率,理论上是GD的invariant approximation,实验上OOD generalization显著超越AR。是architectural inductive bias设计的好例子。

---

# Rethinking Invariance in In-Context Learning - 深度解析

## 1. 核心问题的motivation

ICL有一个反直觉的现象: 给LLM的context examples $D = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$ 虽然在统计上i.i.d. (理论上应该与permutation无关), 但由于Transformer的causal mask $\mathbf{M}$, 把sequence reorder一下accuracy就能从90%掉到50% (Lu et al. 2022, SST-2上的实验)。

这违反了一个基本的inductive bias: 当数据具有某种symmetry时, model architecture如果respect这个symmetry, generalization error bound会变紧 (Sokolic et al. 2016, Bietti et al. 2021, Tahmasebi & Jegelka 2023)。DeepSets / Set Transformer / PointNet这些work也都基于这个思想。所以问题变成: 能不能设计一个architecture让ICL在permutation下invariant, 同时不损失performance?

**Reference**: 原始paper [github.com/PKU-ML/InvICL](https://github.com/PKU-ML/InvICL), Garg et al. 2022 (synthetic ICL benchmark) [arxiv.org/abs/2208.01066](https://arxiv.org/abs/2208.01066), Lu et al. 2022 [arxiv.org/abs/2107.08615](https://arxiv.org/abs/2107.08615)。

---

## 2. 三个desiderata的形式化

作者claim好的invariant ICL必须同时满足三条性质, 这三条性质其实是分析现有方法缺什么的锐利工具:

### (1) Permutation Invariance (Definition 3.1)

$$f_t(\tilde{\mathbf{x}}_1, \ldots, \tilde{\mathbf{x}}_n, \mathbf{x}_t) = f_t(\tilde{\mathbf{x}}_{i_1}, \ldots, \tilde{\mathbf{x}}_{i_n}, \mathbf{x}_t)$$

其中 $\tilde{\mathbf{x}}_i := (\mathbf{x}_i, \mathbf{y}_i)$ 是第 $i$ 个context example, $(i_1, \ldots, i_n) \in S_n$ 是 $[n]$ 的任意permutation。注意只需要 $f_t$ (对test example的prediction) invariant, 不需要对intermediate prediction invariant。

### (2) Information Non-leakage (Definition 3.2)

$$f(\ldots, \mathbf{x}_i, \mathbf{y}_i, \ldots)_i = f(\ldots, \mathbf{x}_i, \mathbf{y}_i', \ldots)_i, \quad \forall \mathbf{y}_i, \mathbf{y}_i' \in \mathcal{Y}$$

直觉: 当model预测 $\mathbf{x}_i$ 的label $\hat{\mathbf{y}}_i$ 时, 它不能access到 ground-truth $\mathbf{y}_i$ 自己。这是AR LLM训练dense learning signal的关键: 因为每个token位置都是一个prediction task, 一个query的answer必须被mask掉, 否则model学会shortcut $f(\mathbf{x}) = \mathbf{y}$, 没有学到task。

### (3) Context Interdependence (Definition 3.3)

$$f(\ldots, \mathbf{x}_i, \mathbf{y}_i, \ldots, \mathbf{x}_j, \mathbf{y}_j, \ldots)_i \neq f(\ldots, \mathbf{x}_i, \mathbf{y}_i, \ldots, \mathbf{x}_j', \mathbf{y}_j', \ldots)_i$$

直觉: $\mathbf{x}_i$ 的prediction应该依赖于其他context example。这允许每个example的encoding带有其他examples作为context的信息, 像message passing / GNN那样。

**Table 1 解读**:

| ICL Type | Invariance | Non-leakage | Interdependence | Performance |
|---|---|---|---|---|
| AR | ✗ | ✓ | ✓ (partial) | A |
| Prefix | ✓ | ✗ | ✓ | A⁻ |
| BoE | ✓ | ✓ | ✗ | A⁻ |
| InvICL | ✓ | ✓ | ✓ | A/A+ |

这是paper最关键的insight: 没有任何现有方法同时满足三条。下面看为什么。

---

## 3. 现有三种ICL aggregation scheme的公式化

设 $\mathbf{h}_{\mathbf{x}_i}, \mathbf{h}_{\mathbf{y}_i}$ 是 $\mathbf{x}_i, \mathbf{y}_i$ 的encodings, $\text{aggr}\{\cdot\}$ 是某种aggregation (这里是Transformer的self-attention)。

**AR ICL** (Eq. 4): 
$$\mathbf{h}_{\mathbf{x}_k} \leftarrow \text{aggr}\{(\mathbf{h}_{\mathbf{x}_i}, \mathbf{h}_{\mathbf{y}_i})_{i=1}^{k-1}, \mathbf{h}_{\mathbf{x}_k}\}, \quad k \in [n+1]$$

第 $k$ 个example只attends到前 $k-1$ 个examples。等价于causal mask $\mathbf{M}_{ij} = -\infty$ 当 $j > i$ (Eq. 3)。这给出sequential order, 后面的examples因为有更长的context, 准确率更高 (Liu et al. 2022; Wu et al. 2022)。Partial interdependence是因为前面的examples看不到后面的, 所以是partial。

**Prefix ICL** (Eq. 5): 
$$\mathbf{h}_{\mathbf{x}_k} \leftarrow \text{aggr}\{(\mathbf{h}_{\mathbf{x}_i}, \mathbf{h}_{\mathbf{y}_i})_{i=1}^n\}, \forall k \in [n]$$
$$\mathbf{h}_{\mathbf{x}_t} \leftarrow \text{aggr}\{(\mathbf{h}_{\mathbf{x}_i}, \mathbf{h}_{\mathbf{y}_i})_{i=1}^n, \mathbf{h}_{\mathbf{x}_t}\}$$

Context examples之间用full attention (所有context都互相看到), test example用causal attention。这是T5的设计 (Raffel et al. 2020, [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683))。**Information leakage**: 当model预测 $\hat{\mathbf{y}}_k$ 时, $\mathbf{y}_k$ 已经在context里, 可以直接copy。所以训练时prediction task trivial化, learning signal变稀疏, performance下降。

**Bag-of-Examples ICL** (Eq. 6, PCW/SAICL/BatchICL):
$$[\mathbf{h}_{\mathbf{x}_k}, \mathbf{h}_{\mathbf{y}_k}] \leftarrow \text{aggr}\{(\mathbf{h}_{\mathbf{x}_k}, \mathbf{h}_{\mathbf{y}_k})\}, \forall k \in [n]$$
$$\mathbf{h}_{\mathbf{x}_t} \leftarrow \text{aggr}\{(\mathbf{h}_{\mathbf{x}_i}, \mathbf{h}_{\mathbf{y}_i})_{i=1}^n, \mathbf{h}_{\mathbf{x}_t}\}$$

每个example单独encode, 像bag-of-words。然后test example再aggregate所有independent encodings。没有cross-example attention, 所以没有interdependence, performance差。代表方法: PCW (Ratner et al. 2022, [arxiv.org/abs/2212.07442](https://arxiv.org/abs/2212.07442)), SAICL (Cai et al. 2023), BatchICL (Zhang et al. 2024)。

---

## 4. 三个Proposition的详细证明 (这部分是paper的精髓)

把self-attention看作graph上的message passing:

$$\mathbf{A} = \text{softmax}(\mathbf{H}\mathbf{W}_q(\mathbf{H}\mathbf{W}_k)^\top + \mathbf{M})$$

$\mathbf{A}_{ij}$ 是从node $j$ 到node $i$ 的message weight。$\mathbf{M} \in \{0, -\infty\}^{n \times n}$ 是mask, $-\infty$ 表示不允许message传递。我们只需要控制 $\mathbf{M}$。

### Proposition 3.4 — 三选一的mask

要实现permutation invariance, $\mathbf{M}$ 必须属于 $\mathcal{M} = \{\mathbf{M}_1, \mathbf{M}_2, \mathbf{0}\}$:

$$\mathbf{M}_1 = \begin{pmatrix} 0 & -\infty & \cdots & -\infty \\ -\infty & 0 & \cdots & -\infty \\ \vdots & & \ddots & \vdots \\ -\infty & \cdots & -\infty & 0 \end{pmatrix}, \quad \mathbf{M}_2 = \begin{pmatrix} -\infty & 0 & \cdots & 0 \\ 0 & -\infty & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & 0 & -\infty \end{pmatrix}$$

证明思路 (Lemma D.1): 对任意permutation matrix $\mathbf{T}$, permutation equivariance要求 $\mathbf{T}\,\text{SA}(\mathbf{H}) = \text{SA}(\mathbf{T}\mathbf{H})$, 展开后等价于 $\mathbf{T}\mathbf{M}\mathbf{T}^{-1} = \mathbf{M}$ (Eq. 22), 即 $\mathbf{M}$ 与 $\mathbf{T}$ commute。

- 由 $\mathbf{T} = \mathbf{T}(i,j)$ (交换 $i,j$), $\mathbf{M}_{ii} = \mathbf{M}_{jj}$, 所有对角元相等, 设为 $c_1$。
- 由 $\mathbf{T}(i,k)$ 和 $\mathbf{T}(j,k)$, $\mathbf{M}_{ij} = c_2$ 对所有 $i \neq j$。
- 所以 $\mathbf{M} = c_1 \mathbf{I} + c_2(\mathbf{1}\mathbf{1}^\top - \mathbf{I})$, 只能取 $c_1, c_2 \in \{0, -\infty\}$, 排除全 $-\infty$ (无意义), 得到三种。

- $\mathbf{M}_1$: BoE ICL (每个node只attention自己)
- $\mathbf{0}$: Prefix ICL (full attention)
- $\mathbf{M}_2$: 只允许cross-example attention, 不允许self-attention。$\mathbf{M}_2$ 在attention score上是 $\mathbf{M}_1$ 和 $\mathbf{0}$ 的linear combination。

Lemma D.2 把"context examples的embedding是permutation equivariant"与"test embedding $f_t$ 是permutation invariant"等价起来。证明用反证: 如果 $\mathbf{M}_{ii} \neq \mathbf{M}_{jj}$, 构造 $\mathbf{h}_1 = \mathbf{e}_1, \mathbf{h}_2 = \mathbf{e}_2$, 交换它们后第一层update就不equivariant, 进而第二层update $f_t$ 改变。

**Intuition**: permutation invariance其实是graph symmetry。$\mathbf{M}$ 必须是equivariant under permutation group $S_n$ 作用, 这强烈限制 $\mathbf{M}$ 只能是这三个。

### Proposition 3.5 — DAG → lower triangular

Information non-leakage要求message-passing graph无cycle (除了self-loop), 等价于存在topological ordering使 $\mathbf{M}$ 在该ordering下是lower triangular。证明 (Appendix D.3): DAG (有向无环图)一定可以topological sort, 在sort后的ordering下所有edge $i \to j$ 满足 $i > j$, 即 $\mathbf{A}_{ij} = 0$ 当 $i \leq j$, $\mathbf{A}$ 是strictly lower triangular。加上self-loop, $\mathbf{M}$ 是lower triangular。

**Intuition**: 这相当于graph coloring的acyclicity constraint。任何cycle都会让信息能从 $\mathbf{y}_i$ 通过其他examples传回 $\hat{\mathbf{y}}_i$, 导致leakage。

### Proposition 3.6 — 必须是diagonal

要同时满足Prop 3.4 (三选一)和Prop 3.5 (lower triangular), 唯一交点是 $\mathbf{M} = \mathbf{M}_1$ (diagonal)。因为:
- $\mathbf{0}$ (Prefix): 全是0, 不是lower triangular。
- $\mathbf{M}_2$ (off-diagonal 0): 严格非lower triangular。
- $\mathbf{M}_1$ (diagonal 0, off-diagonal $-\infty$): 既是lower triangular又属于 $\mathcal{M}$。

**关键结论**: 单步self-attention propagation中, 同时实现invariance + non-leakage **必须**退化到BoE, 而BoE没有interdependence。这就是paper title "Rethinking"的核心: 在single-pass message passing的framework下, 这三个性质不能同时满足。必须引入multi-stage / pre-encoding。

---

## 5. InvICL的设计 — Leave-One-Out Pre-encoding

### 两阶段设计 (Algorithm 1)

**Stage 1 (independent encoding)**: 对每个context example, 用BoE独立编码:

$$(\bar{\mathbf{h}}_{\mathbf{x}_i}^{(k)}, \bar{\mathbf{h}}_{\mathbf{y}_i}^{(k)}) = \text{aggr}\{(\bar{\mathbf{h}}_{\mathbf{x}_i}^{(k-1)}, \bar{\mathbf{h}}_{\mathbf{y}_i}^{(k-1)})\}$$

每个example只attention自己 (diagonal mask $\mathbf{M}_1$), 防止leakage。

**Stage 2 (LOO pre-encoding)**: 对每个context example, 用所有**其他**examples的independent encodings作为context, encode它自己:

$$(\mathbf{h}_{\mathbf{x}_i}^{(k)}, \mathbf{h}_{\mathbf{y}_i}^{(k)}) = \text{aggr}\{(\bar{\mathbf{h}}_{\mathbf{x}_j}^{(k-1)}, \bar{\mathbf{h}}_{\mathbf{y}_j}^{(k-1)})_{j \neq i}, \mathbf{h}_{\mathbf{x}_i}^{(k-1)}\}$$

注意 $j \neq i$ — leave-one-out! 第 $i$ 个example的encoding不包含 $\mathbf{y}_i$ 信息, 因为其他examples的 $\bar{\mathbf{h}}$ 来自Stage 1的independent encoding, 而 $\mathbf{h}_{\mathbf{x}_i}$ 只用 $\mathbf{x}_i$ (不直接看 $\mathbf{y}_i$)。

**Stage 3 (test prediction)**: test example $\mathbf{x}_t$ aggregate所有LOO encodings:

$$\mathbf{h}_{\mathbf{x}_t}^{(k)} = \text{aggr}\{(\mathbf{h}_{\mathbf{x}_i}^{(k-1)}, \mathbf{h}_{\mathbf{y}_i}^{(k-1)})_{i=1}^n\}$$

### 为什么三个性质都满足

- **Invariance**: Stage 1每个example独立, Stage 2用leave-one-out集合 (无论permutation, 这个集合内容一样), Stage 3aggregate所有, 所以对permutation不变。
- **Non-leakage**: $\hat{\mathbf{y}}_i$ 由 $\mathbf{h}_{\mathbf{x}_i}^{(k)}$ 预测, 而 $\mathbf{h}_{\mathbf{x}_i}^{(k)}$ 由 $(\bar{\mathbf{h}}_{\mathbf{x}_j}, \bar{\mathbf{h}}_{\mathbf{y}_j})_{j \neq i} + \mathbf{h}_{\mathbf{x}_i}$ 计算, 没有 $\mathbf{y}_i$ 的直接路径。
- **Interdependence**: Stage 2每个example的encoding包含所有其他examples的信息。

### Intuition — 这其实就是Jackknife / LOO Cross-Validation思想

统计学中jackknife估计某个样本 $i$ 的"没有它的影响": 用所有除 $i$ 之外的样本估计, 再加回 $i$ 的信息。这里把每个context example的"task-relevant信息"分离出来, 用其他examples告诉model "task是什么", 然后给 $\mathbf{x}_i$ 自己一个fair prediction机会。

类似思想: 
- **Cross-validation**: 评估一个example在没看到它时模型怎么predict。
- **Influence functions**: 估计去掉一个样本后的model output变化。
- **Boosting / bagging**: 用subset训练再aggregate。
- **Dropout 1-of-n**: 随机mask掉一个样本看其余怎么aggregate, 这里是deterministic version。

### Symmetric Positional Encoding (Appendix A.1)

为了不破坏invariance, positional encoding也要symmetric: 每个example都从position 1开始独立编码, test example从 $\ell_{\text{max}}$ 开始 (避免与context examples的position冲突)。Figure 5给出示意图。这与NoPE (Kazemnejad et al. 2024, [arxiv.org/abs/2305.16843](https://arxiv.org/abs/2305.16843))对比, 显示symmetric PE比无PE更好。

---

## 6. Parallel Implementation — The Unrolling Trick (Section 3.3)

### 朴素实现: $n+1$ 次 forward pass

对每个 $i \in [n]$, 需要一次LOO forward pass pre-encode $\mathbf{x}_i$, 加一次final prediction forward pass, 共 $n+1$ 次。Computationally prohibitive。

### Parallel实现: 输入unrolling + LOO mask

把input复制一次:

$$\mathbf{Z} = (\tilde{\mathbf{x}}_1, \ldots, \tilde{\mathbf{x}}_n, \tilde{\mathbf{x}}_1, \ldots, \tilde{\mathbf{x}}_n, \mathbf{x}_t)$$

总长度 $2n+1$。构造一个 $(2n+1) \times (2n+1)$ 的mask (Figure 2(d)), 分三个block:

**Block 1 (前 $n$ 行, 前 $n$ 列)**: $\mathbf{M}_1$ (diagonal), 实现Stage 1 independent encoding。结果: $\bar{\mathbf{h}}_i$。

**Block 2 (第 $n+1$ 到 $2n$ 行, 前 $2n$ 列)**: LOO mask — 第 $n+i$ 行 (对应 $\tilde{\mathbf{x}}_i$ 的第二次出现) 允许attend到:
- Block 1的所有 $\bar{\mathbf{h}}_j$ 其中 $j \neq i$ (用Stage 1的结果作为context)
- Block 2自身的 $\mathbf{h}_i$ (即第 $n+i$ 行第 $n+i$ 列的对角)
- **禁止** attend到第 $i$ 行 (即Block 1的 $\bar{\mathbf{h}}_i$), 这保证leakage不会发生 — $\tilde{\mathbf{x}}_i$ 的Stage 2 encoding不能用到自己的Stage 1 encoding (其中包含 $\mathbf{y}_i$ 信息)

具体说: 第 $n+i$ 行的mask在列 $j \in [n]$ 上设 $-\infty$ 当 $j = i$, 设 0 当 $j \neq i$; 在列 $n+j \in [n+1, 2n]$ 上设 0 当 $j = i$, 设 $-\infty$ 当 $j \neq i$。

这给出一个精巧的"twisted diagonal" pattern。结果: $\mathbf{h}_i$ (Stage 2 LOO encoding)。

**Block 3 (第 $2n+1$ 行, 第 $n+1$ 到 $2n$ 列)**: test example $\mathbf{x}_t$ attend到所有Stage 2 encodings $\mathbf{h}_i, i \in [n]$。

### 复杂度分析

设 $n$ 个context examples, 1个test example, attention计算次数 (mask里0元素个数):
1. Stage 1 self-encoding (Block 1): $n$ 个self-attention
2. Stage 2 LOO pre-encoding (Block 2): $n^2$ 次 (每个 $\mathbf{h}_i$ 看其他 $n-1$ + 自己1 = $n$)
3. Test aggregation (Block 3): $n+1$ 次

总计 $n^2 + 2n + 1$, 与Prefix ICL ($n^2 + 1$)同阶, 约为AR ICL ($n^2/2 + 3n/2 + 1$)的2倍。但同一forward pass, 实测inference time几乎一样 (Table 3): AR 21.9ms, Prefix 22.0ms, PCW 21.7ms, InvICL 22.0ms。Memory overhead: GPT-2 Large输入从512到1024, GPU memory增14% (4.2GB → 4.8GB)。

### Positional Encoding for unrolled input

Figure 5(b): 第二次出现的example用与第一次相同的PE, 维持对称性。

---

## 7. 实验数据深度解读

### 7.1 Synthetic linear regression (Section 4.1)

Setup: $g(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$, $\mathbf{w} \sim \mathcal{N}(0, \mathbf{I}_d)$, $d=20$, $k=40$ context examples, 12-layer GPT decoder, MSE loss。Optimal baseline: least squares。

**Figure 3的关键观察**:
- 50k epochs: 只有InvICL接近optimal, 其他所有方法 (AR, Prefix, BoE, NoPE) 还没学到。
- 200k epochs: InvICL ≥ AR > Prefix ≈ NoPE > BoE, 顺序明确。
- Length extrapolation: 训练用 $k=40$, 测试 $k$ 从10到100。InvICL在整个区间都接近optimal, 其他方法在 $k > 40$ 时严重退化。

**Intuition**: 这个MSE task直接对应least squares, 而least squares是permutation invariant的 — 重排 $\{(\mathbf{x}_i, y_i)\}$ 不改 $(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$ 的值。所以invariant architecture的inductive bias完全匹配problem structure。

**Out-of-distribution setup (Figure 7, Appendix B.1)**: 测试时改分布:
1. 加高斯noise $b \sim \mathcal{N}(0, 1)$
2. Scale data $\mathcal{D}_\mathcal{X} = \mathcal{N}(0, 9\mathbf{I})$
3. 从10维subspace采样 (原20维)

InvICL在每个OOD setup都outperform AR ICL, 显示更强的generalization — 这是permutation invariance + non-leakage + interdependence三者结合的效果。

**Sparse linear regression & decision tree (Figure 6)**: 
- Sparse (3/20 non-zero): optimal是Lasso, InvICL快速converge, AR不converge。
- Decision tree (depth 4): 短sequence时AR略好, 长sequence时InvICL胜出, 暗示InvICL的extrapolation来自invariance而非简单memorize。

**Linear probing (Figure 10, Appendix B.5)**: 冻结pretrained model, 在layer 3, 6, 9, 12的hidden states上训练一个linear probe。
- InvICL: 4层都converge到pretrained model的水平, 表示任务features在早期layer就被encode。
- AR ICL: 只有layer 12 converge。
- 这与context interdependence一致: InvICL能更早layer就propagate task信息。

### 7.2 Real-world (MetaICL, Section 4.2)

Setup: GPT-2 Large (762M), short fine-tuning on 142 tasks (classification, QA, NLI, paraphrase), 8 context examples per prompt, 7 evaluation settings: HR→LR, Class→Class, Non-Class→Class, QA→QA, Non-QA→QA, Non-NLI→NLI, Non-Para→Para。也测OOD (target tasks在unseen domains)。

**Table 2 解读 (GPT-2 Large)**:

"All target tasks" (average):
- AR ICL: 41.9 ± 1.15 (baseline)
- NoPE: 37.0 ± 0.81 (差)
- PCW (BoE): 38.1 ± 0.98 (差, 没interdependence)
- SAICL (BoE): 41.4 ± 1.03 (略差于AR)
- BatchICL (BoE): 30.7 ± 0.45 (最差, 因为aggregation layer选择粗糙)
- Prefix ICL: 39.4 ± 1.11 (leakage拉低)
- **InvICL: 42.4 ± 0.87** (最好, 4/7 task胜出)

"Target tasks in unseen domains" (OOD):
- AR ICL: 43.6 ± 1.65
- **InvICL: 48.4 ± 1.72** (7/7 task全部胜出!)

这个OOD泛化优势非常显著, 比AR ICL高4.8个点, 说明invariance在分布shift下提供更稳定的inductive bias。这与Sokolic et al. 2016的group invariant classifier理论一致: 在数据有symmetry时, invariant model的sample complexity降低, OOD generalization bound更紧。

### 7.3 Length Generalization (Figure 4)

HR→LR setting, meta-train用8 examples, test用1到12 examples。InvICL从1-shot到12-shot都保持稳定性能, AR ICL在test length < train length时性能下降明显。这暗示InvICL学到的"aggregation algorithm"对长度敏感度低, AR ICL的per-position信息泄漏让它overfit到train length。

### 7.4 Ablation (Table 4)

在HR→LR setting:
- AR ICL: 43.4, sensitivity 0.25 (sensitivity = 不同permutation下accuracy的variance)
- AR + SymPE: 38.4 (下降5.0!), sensitivity 0.30 (上升0.05)
- AR + Inv mask: 44.8 (+1.4), sensitivity 0.10 (-0.15)
- AR + Both (=InvICL): 45.1 (+1.7), sensitivity 0.00 (完全invariant!)

注意只加SymPE反而下降, 因为GPT-2的absolute PE和causal mask是coupled的, 改PE破坏了pretraining的locality prior。但加Inv mask + SymPE后两者协同, 不仅invariant, 性能也提升。Sensitivity从0.25降到0.00是完美的permutation invariance证据。

### 7.5 Doubled input ablation (Table 8, Appendix B.3)

质疑者可能说: InvICL胜只是因为input duplicated。把AR/PCW/Prefix也用duplicated input训练, 结果:
- AR ICL doubled: 43.8 (vs original 43.4)
- PCW doubled: 40.6 (vs 39.7)
- Prefix doubled: 41.7 (vs 40.3)
- InvICL: 45.1

InvICL仍胜出。说明性能优势来自架构本身 (mask pattern), 不是单纯的双倍输入。

### 7.6 Different base models (Tables 6, 7)

- GPT-Neo 2.7B (Alibi PE): InvICL avg 41.3 vs AR 41.2 (all), OOD 42.7 vs 42.1
- Pythia-2.8B (Rotary PE): InvICL avg 31.9 vs AR 30.8 (all), OOD 32.4 vs 30.5

显示InvICL对PE类型 (trainable, Alibi, Rotary) 都robust, 不依赖specific PE。

---

## 8. 理论分析 — InvICL ≈ Gradient Descent (Theorem C.1, Appendix C)

这是paper最theoretical的部分, 与Von Oswald et al. 2023 ([arxiv.org/abs/2212.07677](https://arxiv.org/abs/2212.07677))和Dai et al. 2022 ([arxiv.org/abs/2212.10559](https://arxiv.org/abs/2212.10559))的"ICL as implicit gradient descent"工作衔接。

### Setup

Linear regression: $\mathcal{L}(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$, $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{y} \in \mathbb{R}^n$。

Standard GD:
$$\mathbf{w}_\ell = \mathbf{w}_{\ell-1} - \eta \mathbf{X}^\top(\mathbf{X}\mathbf{w}_{\ell-1} - \mathbf{y}) \tag{11}$$

其中 $\ell$ 是iteration step, $\eta$ 是step size。

### 特殊parametrization (Eq. 12)

$$\mathbf{W}_k = \mathbf{W}_q = \begin{pmatrix} \mathbf{I}_{d \times d} & \mathbf{0} \\ \mathbf{0} & 0 \end{pmatrix}, \quad \mathbf{W}_v = \begin{pmatrix} \mathbf{0}_{d \times d} & \mathbf{0} \\ \mathbf{w}_0 & -1 \end{pmatrix}, \quad \mathbf{P} = \eta \mathbf{I}$$

变量解释:
- $\mathbf{W}_k, \mathbf{W}_q$ 用 $\mathbf{I}_d$ 部分意味着只有 $\mathbf{x}$ 维度进入key/query inner product (attention score取决于 $\mathbf{x}_i^\top \mathbf{x}_j$)。
- $\mathbf{W}_v$ 的最后一行 $(\mathbf{w}_0, -1)$ 表示value的最后一维 (label维)由当前weight估计 $\mathbf{w}_0$ 减去label $y_j$ 之差给出 (residual)。
- $\mathbf{P} = \eta \mathbf{I}$ 是projection, scale成 $\eta$ 控制update步长。

### Theorem C.1结果

在InvICL architecture下, 第 $\ell$ 层最后一个token (test example)输出:

$$\mathbf{z}_t^{(\ell)} = \binom{\mathbf{x}_t}{\mathbf{x}_t^\top \mathbf{w}_\ell}$$

其中 $\mathbf{w}_\ell$ 遵循update rule:

$$\mathbf{w}_\ell = \mathbf{w}_{\ell-1} - \eta \mathbf{X}^\top(\mathbf{X}\mathbf{w}_{\ell-1} - \mathbf{y}) + \eta^2 \Delta \mathbf{w}_{\ell-1} \tag{13}$$

$$\Delta \mathbf{w}_\ell = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top - \text{diag}(\mathbf{X}\mathbf{X}^\top))(\mathbf{X}\mathbf{w}_\ell - \mathbf{y})$$

变量含义:
- $\eta \mathbf{X}^\top(\mathbf{X}\mathbf{w}_{\ell-1} - \mathbf{y})$: standard GD update (Eq. 11)
- $\text{diag}(\mathbf{X}\mathbf{X}^\top)$: Gram matrix的对角, 即 $\|\mathbf{x}_i\|^2$ 组成的对角阵
- $\mathbf{X}\mathbf{X}^\top - \text{diag}(\mathbf{X}\mathbf{X}^\top)$: off-diagonal Gram matrix, 表示cross-example interactions
- $\eta^2 \Delta \mathbf{w}_{\ell-1}$: 二阶修正项, 来自LOO机制

证明sketch (Appendix D.1): 三个update rule (Eq. 14):
- $\mathbf{z}_j$ (Stage 1 self-encoding, $j \in [n]$): 对自身self-attention, 用 $\mathbf{W}_v$ 产生residual $\mathbf{w}_0 - y_j$。
- $\mathbf{z}_{n+j}$ (Stage 2 LOO, $j \in [n]$): sum over $i \neq j$, 等价于一次GD step on dataset minus第 $j$ 个example。
- $\mathbf{z}_{2n+1}$ (test): sum over $n$ 个LOO encodings。

展开后 (Eq. 17-19) 看到LOO机制自然产生的修正项 $\eta[\mathbf{x}_i^\top \mathbf{x}_i (\mathbf{x}_i^\top \mathbf{w}_{\ell-1} - y_i)]_{i=1}^n$, 与 $\eta \mathbf{X}\mathbf{X}^\top$ 的对角元素相消, 留下off-diagonal二阶项 $\eta^2 \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top - \text{diag}(\mathbf{X}\mathbf{X}^\top))(\mathbf{X}\mathbf{w}_\ell - \mathbf{y})$。

### Convergence条件

GD收敛要求 $\eta < 1/\lambda_{\max}(\mathbf{X}\mathbf{X}^\top)$。在这个条件下, 二阶项 $\eta^2 \|\Delta \mathbf{w}\| = O(\eta^2)$ 被一阶项 $O(\eta)$ dominant, InvICL仍然converge到least squares解 $\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$。

### 对比其他ICL方法 (Section C, Discussion)

- **AR ICL**: 对应online GD with constant learning rate (Ding et al. 2023), 不保证收敛 — 每个example只看前面的, gradient估计有偏。
- **Prefix ICL**: 严格实现standard GD (Von Oswald et al. 2023), 但leakage使prediction task trivial, 实际训练信号弱。
- **BoE ICL**: 只能更新test example的weight, context examples不动。Gradient永远在初始点 $\mathbf{w}_0$ 计算: $\mathbf{w}_\ell = \mathbf{w}_{\ell-1} - g(\mathbf{w}_0, \{(\mathbf{x}_i, y_i)\})$, 所以不能converge到optimal。
- **InvICL**: 近似standard GD (有二阶修正), 无leakage, 有interdependence — 综合最优。

---

## 9. 关键insight总结 (build intuition)

1. **Symmetry作为architecture prior**: permutation invariance不是nicety, 是一个inductive bias, 在数据有symmetry时降低sample complexity (Bietti et al. 2021, [arxiv.org/abs/2107.09549](https://arxiv.org/abs/2107.09549))。这解释为什么InvICL在OOD length generalization上特别强。

2. **Information leakage vs. dense learning signal**: causal mask不只是为了autoregressive生成, 它提供了dense supervision — 每个token位置都是prediction task。破坏这个 (像Prefix ICL) 会让训练信号稀疏。

3. **Single-pass message passing的表达力极限**: Proposition 3.6告诉我们, 单层graph propagation里, invariance + non-leakage必然force diagonal mask, 即BoE, 即无interdependence。这是message passing表达能力上的硬限制。Multi-stage / pre-encoding是必须的。

4. **LOO = "task from other examples" + "data from this example"**: InvICL的LOO设计是一种concept separation — task信息从其他examples获取, data信息从当前example获取。这与人类few-shot learning的"先看其他例题找规律, 再解当前题"的过程对应。

5. **Unrolling作为parallelization trick**: 这是LLM chain-of-thought (Wei et al. 2022, [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903))的architectural analog — 把sequential computation unroll成single forward pass, 通过mask控制信息流。这暗示未来可能有multi-stage unrolling实现更复杂算法。

6. **ICL ≈ GD的family of theories**: Von Oswald et al. 2023, Dai et al. 2022, Ahn et al. 2023 ([arxiv.org/abs/2306.13892](https://arxiv.org/abs/2306.13892)), Bai et al. 2023 ([arxiv.org/abs/2306.00297](https://arxiv.org/abs/2306.00297))。InvICL paper在这个family里加入了invariance约束下的GD分析, 显示GD的invariant approximation可以实现。

7. **对称的PE**: 在invariance要求下, PE本身也要symmetric。这与rotary/alibi PE的设计哲学不矛盾 — 关键是每个example独立编号, 不引入global position。NoPE (Kazemnejad et al. 2024, [arxiv.org/abs/2305.16843](https://arxiv.org/abs/2305.16843))显示有时连PE都可以省去。

---

## 10. 一些可能的extension / open questions

1. **更高阶interdependence**: 当前InvICL实现Stage 2 LOO用 $n-1$ 个其他examples。如果做 $n-2$ LOO (jackknife-2) 或者更深unrolling, 能否实现更细的task/data separation?

2. **Multi-task / multi-modal**: 当context examples来自不同task时, invariance可能不再合适 — 应该有hierarchical structure。能否把InvICL generalize到equivariant (preserve ordering in some structured way)?

3. **Real-LLM scale**: 现有实验在GPT-2 762M / GPT-Neo 2.7B / Pythia-2.8B上。对GPT-4 / Llama-70B规模, InvICL的invariance prior是否还重要? 还是说大模型自己学到了隐式invariance?

4. **Connection to Mamba / state space models**: SSMs的selective scan也是sequential, 也有order sensitivity。InvICL的mask design能否迁移到SSM?

5. **Generation tasks**: 现有evaluation是classification。对generation (QA等), 怎么定义并实现invariance? Test input的generation位置如何mask?

6. **Pre-training vs. fine-tuning**: 现有work是short fine-tuning。如果InvICL architecture用于pre-training, 能否获得更好的ICL emergent ability?

7. **Implicit batch normalization**: 注意Stage 1每个example独立encode, Stage 2用所有其他 — 这其实是一个deterministic ensemble。能否与batch normalization / group norm类比, 看成implicit cross-example normalization?

---

## 11. Reference汇总

核心paper:
- [github.com/PKU-ML/InvICL](https://github.com/PKU-ML/InvICL)

ICL基础:
- Brown et al. 2020 (GPT-3, ICL emergence) [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
- Min et al. 2022b (MetaICL) [arxiv.org/abs/2110.15943](https://arxiv.org/abs/2110.15943)

Order sensitivity:
- Lu et al. 2022 (fantastically ordered prompts) [arxiv.org/abs/2107.08615](https://arxiv.org/abs/2107.08615)
- Zhao et al. 2021 (calibrate) [arxiv.org/abs/2102.09690](https://arxiv.org/abs/2102.09690)

Invariant ICL variants:
- Raffel et al. 2020 (T5/Prefix LM) [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
- Ratner et al. 2022 (PCW) [arxiv.org/abs/2212.07442](https://arxiv.org/abs/2212.07442)
- Zhang et al. 2024 (BatchICL)

ICL as GD theory:
- Von Oswald et al. 2023 [arxiv.org/abs/2212.07677](https://arxiv.org/abs/2212.07677)
- Dai et al. 2022 [arxiv.org/abs/2212.10559](https://arxiv.org/abs/2212.10559)
- Ahn et al. 2023 [arxiv.org/abs/2306.13892](https://arxiv.org/abs/2306.13892)
- Bai et al. 2023 [arxiv.org/abs/2306.00297](https://arxiv.org/abs/2306.00297)
- Akyürek et al. 2022 [arxiv.org/abs/2211.08360](https://arxiv.org/abs/2211.08360)

Symmetry & generalization:
- Sokolic et al. 2016 [arxiv.org/abs/1611.00738](https://arxiv.org/abs/1611.00738)
- Bietti et al. 2021 [arxiv.org/abs/2107.09549](https://arxiv.org/abs/2107.09549)
- Tahmasebi & Jegelka 2023

Synthetic ICL benchmarks:
- Garg et al. 2022 [arxiv.org/abs/2208.01066](https://arxiv.org/abs/2208.01066)

PE studies:
- Kazemnejad et al. 2024 (NoPE) [arxiv.org/abs/2305.16843](https://arxiv.org/abs/2305.16843)

Related architectural ideas:
- Set Transformer / DeepSets (permutation invariant set encoding)
- GNN message passing (graph invariance)
- Jackknife / cross-validation (statistics)
- Influence functions (Koh & Liang 2017)

---

**总结**: 这篇paper的核心贡献是用一个清晰的mathematical framework (graph message passing + mask design) 识别出invariant ICL的两个关键性质 (non-leakage + interdependence), 并通过Prop 3.4-3.6证明single-pass无法同时满足, 然后用LOO pre-encoding巧妙地实现三者。unrolling trick让它保持 O(n²) 复杂度。理论分析显示这是standard GD的approximation。实验在synthetic和real-world都验证, 尤其OOD generalization显著超越AR ICL。这是一个architectural inductive bias设计的好例子, 在symmetry / dense learning / interaction三者中找到平衡。
