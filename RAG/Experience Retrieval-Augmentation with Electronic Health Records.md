---
source_pdf: Experience Retrieval-Augmentation with Electronic Health Records.pdf
paper_sha256: ac9bb38154f36a0993dad0f6c504df8fd4bbd119e4612c535fd138dcf8416eb0
processed_at: '2026-08-04T06:11:57-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ExPRAG 这篇 paper

## 一句话说清楚

医生看病靠的是经验，不是背教科书。这篇 paper 就是让 LLM 也学会"翻病例档案"——遇到一个病人，先去数据库里找几个长得最像的病人，看看他们当时是怎么治的，然后照葫芦画瓢。

就这么简单。没有什么花里胡哨的 architecture innovation，核心 trick 就是：**用 EHR 里的 ICD code 做精准匹配，再用 text retriever 提取细节**。

---

## 为什么 traditional RAG 在医疗场景下不好使

假设你问 LLM："这个房颤 + CKD stage 3 + 之前有 GI bleed 的老太太，warfarin 该开多少？"

Traditional RAG 的做法：去 Wikipedia 检索 "warfarin standard dosage"，拿回来一段说 "一般 5mg/day starting dose"。

这完全没用。因为标准剂量谁都查得到，真正难的是 **这个特定组合的 comorbidity 下该怎么调**。

有经验的医生会回忆："我上个月那个病人也是房颤 + CKD + bleed history，最后我们给的是 2.5mg，INR 维持在 2.0 左右"。这种 **case-based experiential knowledge** 才是临床决策的核心。

ExPRAG 就是把这个过程自动化：用 EHR codes 找到"上个月那个病人"，把他的 discharge report 拉出来给 LLM 当 reference。

Reference: 这种 case-based reasoning 的认知科学基础可以看 [Schmidt et al. on medical expertise](https://www.cambridge.org/core/books/medical-cognition/...)，以及经典的 [Aamodt & Plaza CBR survey](https://www.researchgate.net/publication/222830493)。

---

## 两步检索的逻辑

### 第一步：用 ICD code 找"双胞胎病人"

MIMIC-IV 里有 32 万份 discharge summary，你不可能全部扔给 LLM。得先 filter。

怎么 filter？ExPRAG 的做法特别朴素：把每个病人的 diagnosis codes、medication codes、procedure codes 分别拎出来，跟 target patient 算 Jaccard similarity。

公式长这样：

$$\tau = \lambda_1 \tau_{\mathrm{Diag}} + \lambda_2 \tau_{\mathrm{Med}} + \lambda_3 \tau_{\mathrm{Proc}}$$

人话翻译：
- $\tau$ 就是两个病人的"像不像程度"总分
- $\tau_{\mathrm{Diag}}$ 是 diagnosis code 的重合度，用 Jaccard 算
- $\tau_{\mathrm{Med}}$ 是 medication code 的重合度
- $\tau_{\mathrm{Proc}}$ 是 procedure code 的重合度
- $\lambda_1, \lambda_2, \lambda_3$ 是三个权重，默认各 1/3

Jaccard 就是交集除以并集：

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

比如病人 A 有 codes `{diabetes, hypertension}`，病人 B 有 `{diabetes, hypertension, CKD}`，那 J(A,B) = 2/3 ≈ 0.67。

这个计算超级快，因为 ICD codes 是 discrete 的，可以 hash 成 binary vector，用 [Faiss](https://github.com/facebookresearch/faiss) 做 batch dot product，32 万病人秒级搞完。

### 第二步：在 top-15 病人的 discharge report 里找相关段落

第一步把 32 万砍到 15 个，第二步就在这 15 份 report 里做 text retrieval。

Paper 试了好几种 retriever，auto-merging 效果最好（79.5%），BM25 最差（68%）。Auto-merging 的优势是它把相关的小 chunk 合并成大 chunk，保留了上下文连贯性。医疗文本最怕切碎——一个诊断的依据可能跨越好几段。

---

## 为什么不用 text embedding 直接做检索

这是 paper 最关键的实验（Table 3, 4）。

他们对比了三种 text embedding ranker（bge-small-en, all-MiniLM-L6, paraphrase-MiniLM-L3）和 EHR-based ranker：

| Ranker | Pearson | Spearman |
|--------|---------|----------|
| bge-small-en | 0.639 | 0.623 |
| all-MiniLM-L6 | 0.640 | 0.618 |
| paraphrase-L3 | 0.478 | 0.481 |
| **ExPRAG EHR** | **0.669** | **0.648** |

EHR-based 赢了。为什么？

因为 **discharge summary 的文本风格差异太大**。不同医生的写作习惯不同——有人详细有人简略，有人用缩写有人全拼，有人按时间线写有人按系统写。Dense embedding 很容易被这些 stylistic noise 主导，而不是真正的 clinical similarity。

而 ICD-10 codes 是人为 curate 的标准化标签。`I48.91` 就是 atrial fibrillation，不管哪个医生写报告，code 都一样。所以 structured code matching 的 signal-to-noise ratio 远高于 text embedding。

这个 insight 其实很深：**在 domain-specific retrieval 里，structured metadata 往往比 raw text embedding 更 faithful**。类似的现象在 [legal document retrieval](https://arxiv.org/abs/2305.15062) 和 [product search](https://arxiv.org/abs/2304.14373) 里也出现过。

---

## DISCHARGEQA：他们造的数据集

### 三个 task 的直觉

1. **Diagnosis Inference**：给病人入院信息，问该下什么诊断。模拟医生从症状反推疾病。
2. **Medication Inference**：给入院信息 + 住院经过，问出院该开什么药。模拟医生做 discharge prescription。
3. **Instruction Inference**：给前两个信息，问出院医嘱怎么写。模拟最后的 discharge planning。

### 最狠的设计：Multi-select

Diagnosis 和 Medication 都是 multi-select，而且 strict scoring——必须全选对才算对。

比如 medication task 给你 10 个选项，其中 3 个是对的，你必须精确选对那 3 个。选多了错，选少了也错。

Random baseline 大概是 $1/2^{10} = 0.1\%$。GPT-4o 在这个 task 上只有 9.68% accuracy。这说明 task 确实 hard，不是随便就能蒙对的。

### 防止 label leakage 的 trick

Discharge summary 本身包含 final diagnosis 和 discharge medication。如果直接把整个 summary 给 LLM，那就是作弊。

Paper 把 summary 切成三段：
- **Clinical profile**：入院信息
- **In-hospital progress**：住院经过
- **Discharge plan**：出院计划（包含 ground truth）

对 diagnosis task 只给第一段；对 medication task 给前两段；对 instruction task 给前两段。这模拟了医生在 decision point 时实际能看到的信息。

### Distractor 的生成方式很 tricky

对 diagnosis task，distractor 来自 **同一个病人的 EHR 表里那些没进最终 discharge diagnosis 的 codes**。

这意味着 distractor 是 patient-specific 的 plausible 选项。LLM 不能靠 "这个诊断跟这个病人没关系" 来排除，因为 distractor 确实出现在这个病人的记录里过，只是最终没被列为 discharge diagnosis。

这比从 random patients 抽 distractor 难多了。

---

## 实验结果里几个有意思的点

### GPT-4o + ExPRAG 在 medication 上 2.6x 提升

从 3.65% → 9.68%。这是整个 paper 最亮眼的数字。

说明 medication task 最依赖 experiential knowledge。drug 的 generic knowledge 大家都会查，但 "这种 comorbidity 组合下实际开了什么药" 只能从 similar cases 学。

### Small models 反而退化

Mistral-7B 在 diagnosis 上 Direct-Ask 16.03% → ExPRAG 13.79%，不升反降。

这说明 retrieval 不是 free lunch。Small model 的 context utilization 能力不够，retrieved context 反而成了 noise。跟 [Self-RAG paper](https://arxiv.org/abs/2310.11511) 里的发现一致：小模型需要专门训练才能有效使用 retrieved context。

### Complementary weighting 反直觉地赢

Table 5 里，medication task 用 complementary weighting（$\lambda_1=0, \lambda_2=1, \lambda_3=1$，即不看 medication similarity 只看 diagnosis + procedure）反而比 uniform 好。

直觉上 medication task 应该看 medication similarity，但实际不是。

因为 **medication choice 是被 diagnosis 和 procedure 决定的**。两个 diagnosis + procedure 很像的病人，即使用药历史不完全一样，他们的 discharge medication 也会很像——因为治疗指南是 standard 的。反过来，两个用药历史像但 diagnosis 不同的病人，discharge medication 可能天差地别。

这说明 clinical variables 之间有 latent causal structure，naive task-aligned retrieval 会 miss 这个结构。

### k 的 ablation 有 sweet spot

Instruction task 喜欢 k 大（25 最好，82.25%），medication task 在 k=20 后开始退化。

直觉解释：instruction 是 high-recall task，多给点 context 不怕；medication 是 high-precision task，多一个 noisy patient 就可能误导用药决策。

---

## 这篇 paper 的 limitations 和我想吐槽的

### Paper 自己承认的

1. 只用了 3 种 EHR entity，没利用 lab results（MIMIC-IV 有 labevents 表，包含 blood test、vital signs time series）
2. 只有 multi-option，没法 evaluate open-ended generation

### 我觉得可以改进的

**Jaccard 的问题**：它是 symmetric 的，但 clinical similarity 应该有方向性。一个只有 `{diabetes}` 的病人 A，和一个有 `{diabetes, CKD, CHF}` 的病人 B，Jaccard = 0.33。但 B 其实是 A 的 "superset"，B 的经验对 A 可能不完全 applicable（B 更复杂），而 A 的经验对 B 的 diabetes 部分可能很有参考价值。

可以考虑 overlap coefficient：$\frac{|A \cap B|}{\min(|A|, |B|)}$，这样 A ⊂ B 时 similarity = 1。

**ICD code 的 hierarchy 没利用**：`M50.12`（cervical disc disorder with radiculopathy, cervical region）和 `M50.11`（same but high cervical）在 Jaccard 里是完全不同的元素，但临床上是同一个病。可以用 [ICD-10 的 tree structure](https://icd.who.int/browse10/) 做 tree-edit-distance，或者用 [Clinical Classification Software (CCS)](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) 把细码映射到大类。

**Temporal information 丢失**：Jaccard 把所有 codes flatten 成 set，但病人 A 先 sepsis 后 AKI，和病人 B 先 AKI 后 sepsis，clinical meaning 完全不同。前者是 sepsis-induced AKI，后者可能是 drug-induced AKI 然后继发感染。Jaccard 给两者同样分数。

**Cohort bias**：MIMIC-IV 是 Beth Israel Deaconess 的 ICU 数据，elderly critical care population 严重偏多。拿到 outpatient 或 pediatric 场景可能完全不 work。

**GPT-4o-mini 当 annotator 的 reliability**：Table 4 用 GPT-4o-mini 判断 patient similarity 作为 "ground truth" 来算 correlation。但 LLM 在 clinical similarity 判断上的 reliability 本身就存疑。应该用 physician annotation，虽然 cost 高。

---

## 对 Karpathy 的 intuition building

几个我觉得值得 take away 的 deep intuitions：

**1. Retrieval 的 modality 决定 ceiling**

Text-to-text retrieval 在 domain-specific 场景容易被 stylistic noise 限制。Structured-to-structured（EHR codes）更 faithful 但 less expressive。Hybrid 的正确姿势是：structured 做 filtering，unstructured 做 generation。

**2. "Experience" 和 "knowledge" 是两种东西**

Medical reasoning 的 bottleneck 不是缺 facts，而是缺 contextualized application。GPT-4o 知道所有 drug 的 mechanism of action，但不知道 "这个特定 comorbidity 组合下实际该怎么调"。这跟软件工程里 stack overflow 的 case-based answers vs official docs 的 fact-based answers 是一回事。

**3. Multi-select strict accuracy 是 brutal metric**

5 对 5 错的 multi-select，random 是 0.1%，GPT-4o 也就 9.68%。这暴露了 LLM 在 combinatorial reasoning 上的 fundamental weakness。相关的工作可以看 [BIG-Bench Hard 的 multi-hop reasoning](https://github.com/suzgunmirac/BIG-Bench-Hard)。

**4. Small model + RAG 是 anti-pattern**

Mistral-7B 用了 ExPRAG 反而变差。Retrieval augmentation 需要 model 有足够 capacity 去 integrate context。这跟 instruction tuning 前的小模型不能有效使用 few-shot examples 是同一个现象。

**5. Task-aligned retrieval 不一定最优**

Medication task 被 diagnosis similarity 帮助最大。Clinical variables 之间有 latent causal structure，naive task-aligned retrieval 会 miss 这个结构。这跟 recommender system 里 "用 purchase history 推荐不如用 browsing history 推荐" 的现象类似——proxy signal 有时比 direct signal 更 informative。

---

## 这篇 paper 的 broader implication

ExPRAG 的思路其实可以泛化到任何 **有 structured metadata + unstructured narrative** 的领域：

- **法律**：用 case category + jurisdiction + party type 做 filter，然后从 case opinion 里 retrieve reasoning
- **客服**：用 product category + issue type + customer tier 做 filter，然后从 resolution notes 里 retrieve solution
- **教育**：用 subject + difficulty + student profile 做 filter，然后从 past student solutions 里 retrieve learning path

核心 pattern 是：**structured metadata 负责 precision，unstructured text 负责 expressiveness，两者 division of labor**。

这跟 [ColBERT 的 late interaction](https://arxiv.org/abs/2004.12832) 思想异曲同工——都是把 retrieval 拆成 coarse filter + fine refinement，只不过 ColBERT 是 token-level，ExPRAG 是 patient-level。

Reference:
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [MIRAGE: Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.371/)
- [EHRNoteQA: An LLM Benchmark for Real-World Clinical Practice](https://proceedings.neurips.cc/paper_files/paper/2024)
- [MIMIC-IV Database](https://physionet.org/content/mimiciv/2.2/)
- [Faiss: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [LlamaIndex Auto-merging Retriever](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizedAutoMerging/)
- [ICD-10-CM Official Guidelines](https://www.cdc.gov/nchs/icd/icd-10-cm.htm)
- [Clinical Classification Software (CCS)](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)
- [Case-Based Reasoning: Foundational Issues](https://www.researchgate.net/publication/222830493)

---

# ExPRAG: Experience Retrieval-Augmentation with EHR 深度解析

## 1. Paper 的核心 Intuition

这篇 paper 抓住了一个非常 intuitive 的 clinical insight：**experienced clinicians 在做 decision-making 时，mental reference 的对象是过去看过的 similar cases，而 textbook facts 仅仅是 baseline 参考**。传统 RAG 把 Wikipedia 上关于 warfarin 的 standard dosage 喂给 LLM，但这个 information 在具体 patient 场景下基本是 useless 的——因为真实 clinical decision 需要的是 "像我之前那个房颤 + CKD + 之前 GI bleed 的病人，最后我们 dose 调到 2.5mg" 这种 **contextualized experiential knowledge**。

ExPRAG 把这个 intuition engineering 化：用 EHR 的 structured codes（ICD-10, NDC）去 find "twin patients"，然后从他们的 discharge reports 里 extract 段落作为 in-context experience，feed 给 LLM。

Reference: 这与 case-based reasoning (CBR) 的经典 AI 传统一脉相承，参见 [Aamodt & Plaza 1994](https://www.researchgate.net/publication/222830493_Case-Based_Reasoning_Foundational_Issues_Methodological_Variations_and_System_Approaches)。Schmidt 等人在 medical cognition 领域也长期研究 physicians 如何通过 illness scripts 检索 past cases，参见 [Medical Cognition & Error](https://www.cambridge.org/core/books/medical-cognition/...)。

---

## 2. Framework 架构解析

### 2.1 为什么需要 Coarse-to-Fine 两步检索

MIMIC-IV 有 ~430k patients、320k 有 discharge summary、文本 corpus >4GB。如果直接对全 corpus 做 dense retrieval：
- **Inefficiency**: chunk-level embedding + ANN search 在 4GB text 上是 O(N) 的代价
- **Loss of longitudinal context**: chunking 把一个 patient 的 history 撕碎
- **Scattered information**: 同一个 medical decision 的依据散落在 multiple notes 里

ExPRAG 的 trick 在于：**先用 structured EHR codes 做一个 O(1) 的 hash-lookup-style filter，把 candidate pool 从 N=320k 砍到 N'=15**，再在这个小 pool 上跑 expensive text retriever。这是典型的 filter-then-refine 范式，类似 [ColBERT 的 late interaction](https://arxiv.org/abs/2004.12832) 思想但在 patient level。

### 2.2 Formulation 详细解读

**公式 (1)**: $d_* = f_{\mathrm{EXPRAG}}(p, q, \mathcal{D})$

- $p$: target patient（query patient），包含其 structured EHR codes
- $q$: medical query（如 "Which diagnoses should be documented?"）
- $\mathcal{D} = \{D_i\}_{N}$: discharge report cohort，每个 $D_i$ 是一份完整的 discharge summary
- $D_i = \{d_j\}_{M}$: 第 $i$ 份 report 被切成 $M$ 个 paragraphs
- $d_* $: 最终被 retrieved 出来作为 in-context experience 的最佳 paragraph（或 paragraph 集合）

**公式 (2)**: $\mathcal{D}' = f_{\mathrm{Ranker}}(p, \mathcal{D})$，其中 $\mathcal{D}' = \{D_i\}_{N' \ll N}$

- 这是 coarse step：N' = 15 (paper 默认 k=15)
- $f_{\mathrm{Ranker}}$ 是 EHR-based，不是 text-based

**公式 (3)**: $d_* = f_{\mathrm{Retriever}}(q, \mathcal{D}')$

- 这是 fine step：在 15 份 summary 里做 query-aware 的 paragraph extraction
- $f_{\mathrm{Retriever}}$ 可以是 auto-merging / BM25 / sentence-window 等

### 2.3 EHR-Based Ranker 的 Similarity 计算

**公式 (4)-(6)** 三个 modality 的 set similarity：

$$\tau_{\mathrm{Diag}} = f_{\mathrm{similarity}}(E_p^{\mathrm{Diag}}, E_{p'}^{\mathrm{Diag}})$$

- $E_p^{\mathrm{Diag}}$: patient $p$ 的 diagnosis code set，元素是 ICD-10 codes（如 `M50.12` = cervical disc disorder with radiculopathy, cervical region）
- $E_p^{\mathrm{Med}}$: medication code set，元素是 NDC codes（National Drug Code，FDA 标准 11-digit 编码）
- $E_p^{\mathrm{Proc}}$: procedure code set，ICD-10-PCS codes

$f_{\mathrm{similarity}}$ 选用 **Jaccard Index**:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Jaccard 的 intuition：两个 patient 共享的 codes 比例越高越相似。它对 set cardinality 不敏感（不像 cosine 在 sparse binary 上会偏向大 set），且对 medical code 的 "co-occurrence" 语义很 natural——两个都同时有 diabetes + CKD + warfarin 的 patient，比一个只有 diabetes 的 patient 更像。

**公式 (7)**: $\tau = \lambda_1 \tau_{\mathrm{Diag}} + \lambda_2 \tau_{\mathrm{Med}} + \lambda_3 \tau_{\mathrm{Proc}}$

- $\lambda_1, \lambda_2, \lambda_3$: 三个 modality 的 balance weights，默认 $\lambda_1 = \lambda_2 = \lambda_3 = 1/3$
- Paper 实验了 "task-focused"（如 diagnosis task 时 $\lambda_1=1, \lambda_2=\lambda_3=0$）和 "complementary"（$\lambda_1=0, \lambda_2=\lambda_3=1$）两种极端，发现 **complementary 优于 task-focused**，这是个有意思的发现——意味着 cross-modality 信号比单一 modality 更 informative

### 2.4 Efficiency Analysis

Jaccard Index 在 binary set 上可以用 **MinHash** 或直接 sparse matrix multiplication 加速，paper 提到用 [Faiss](https://github.com/facebookresearch/faiss) 和 [NumPy](https://numpy.org/) 实现。每个 patient 的 code set 可以 hash 成 fixed-size binary vector，然后 similarity computation 退化为 batched dot product。对 320k patients 来说，每次 query 只需要一次 matrix-vector product（320k × code_dim），在 GPU 上 sub-second 级别。

---

## 3. DISCHARGEQA Dataset 构建详解

### 3.1 数据筛选 Pipeline

| 阶段 | Patient 数 | 过滤规则 |
|------|----------|---------|
| MIMIC-IV 原始 | 430k | - |
| 有 discharge summary | 320k | 移除无 note 的 |
| Code 数量合理 | 28k | diagnosis/medication/procedure 每个 ∈ [3, 40] |
| Instruction 可用 | <28k | GPT-4o 过滤短 summary |

3-40 的 bound 是 trade-off：太少 codes 的 patient 是 outlier（可能数据缺失），太多 codes 的 patient 是 complex 多病共存 case，会污染 similarity 信号。

### 3.2 三类 Task 的 Design Intuition

| Task | Query 类型 | Background | Option Source | Realism 模拟 |
|------|----------|----------|--------------|-------------|
| Diagnosis Inference | Multi-select | Clinical profile | EHR + Discharge | 医生从症状反推诊断 |
| Medication Inference | Multi-select | Clinical profile + In-hospital | EHR + Discharge | 医生决定出院带药 |
| Instruction Inference | Single-select | Clinical profile + In-hospital | Discharge + AI permutation | 医生写出院医嘱 |

**Multi-select 是关键 design choice**：临床现实中诊断从来不是 single-label，一个 elderly patient 可能同时有 CHF、AF、CKD、diabetes。Multi-select 的 strict accuracy（必须全对才算对）让 task difficulty 暴涨——Table 2 显示 GPT-4o 在 medication 上也只有 9.68% accuracy，这证明 dataset 确实 hard。

### 3.3 Label Leakage 防护

Paper 把 discharge summary 切成 3 个 temporal phase:
- **Clinical profile**: admission info（demographics, presenting condition, initial assessment）
- **In-hospital progress**: treatment plan, therapies
- **Discharge plan summary**: final diagnosis, discharge meds, instructions

对 diagnosis task 只给 clinical profile；对 medication task 给前两个；对 instruction task 给前两个。这模拟了 clinician 在 decision point 时能看到的 information，避免直接 leak discharge summary 里的 ground truth。

### 3.4 Distractor Generation 的 Trick

对 diagnosis/medication task，distractors 来自 **同一 patient 自己的 EHR 表**——即 patient 在住院期间有过但 final discharge 没有的 diagnosis/meds。这比从 random patients 抽 distractors 难得多，因为 distractor 是 patient-specific plausible 选项，LLM 不能靠 "这个 diagnosis 听起来不像这个 patient" 来排除。

---

## 4. 实验结果深度分析

### 4.1 Main Results (Table 2) 关键 Observations

| Model | Task | Direct-Ask | bge-small | ExPRAG EHR | 相对改进 |
|-------|------|-----------|-----------|------------|---------|
| GPT-4o | Medication Acc | 3.65 | 4.95 | **9.68** | +165% |
| GPT-4o | Diagnosis F1 | 0.510 | 0.493 | **0.530** | +3.9% |
| GPT-4o | Instruction Acc | 90.0 | 90.3 | **91.3** | +1.4% |
| Qwen3-30B | Instruction | 90.8 | 93.8 | **95.3** | +4.9% |

几个有意思的 patterns:

1. **GPT-4o + ExPRAG 在 medication 上飞跃**：从 3.65% → 9.68%，几乎是 2.6x。这说明 medication task 是最依赖 experiential knowledge 的——generic drug knowledge 没用，必须看 similar patient 实际被 prescribe 了什么。

2. **Small models 反而退化**：Mistral-7B 在 diagnosis 上 Direct-Ask 16.03% → ExPRAG 13.79%。Hypothesis: small model 无法有效利用 retrieved context，反而被 noise 干扰。这与 RAG 文献中 "small models struggle with retrieval" 的发现一致，参见 [Asai et al. 2023 on Self-RAG](https://arxiv.org/abs/2310.11511)。

3. **Instruction task improvement 小**：因为 single-select + accuracy 已经 90%+，ceiling effect。

### 4.2 Ranker Comparison (Table 3, 4)

EHR-based ranker 在 retrieval correlation 上也最强：

| Ranker | Pearson | Spearman |
|--------|---------|----------|
| bge-small-en | 0.639 | 0.623 |
| all-MiniLM-L6 | 0.640 | 0.618 |
| paraphrase-L3 | 0.478 | 0.481 |
| **ExPRAG EHR** | **0.669** | **0.648** |

Pearson 提升 ~3-5%，Spearman 提升 ~4-5%。这意味着 EHR codes 提供的 similarity 信号比 320k discharge summary 的 dense embedding 更 faithful——这并不意外，因为 discharge summary 的 text 是 clinician 的 narrative，style variation 巨大，dense embedding 很容易被 stylistic noise 主导。而 ICD-10 codes 是人为 curated 的 diagnosis label，semantic 噪声小得多。

### 4.3 k 的 Ablation (Table 6)

| k | Instruction Acc | Diagnosis F1 | Medication F1 |
|---|----------------|-------------|--------------|
| 5 | 80.00 | 0.511 | 0.366 |
| 10 | 78.75 | 0.511 | 0.371 |
| 15 | 79.50 | 0.504 | 0.371 |
| 20 | 80.75 | **0.524** | **0.377** |
| 25 | **82.25** | 0.515 | 0.352 |

Instruction 喜欢大 k（more context helps），但 medication 在 k=25 退化（noise 主导）。这暗示不同 task 对 retrieved experience 的 signal-to-noise ratio 敏感度不同——medication 是 high-precision task（错了就 dangerous），instruction 是 high-recall task（多一些 context 不怕）。

### 4.4 Retriever Ablation (Table 7)

Auto-merging (79.5%) > Sentence-window = Flare (74.5%) > Contriever = BM25+ (69%) > BM25 (68%)。

Auto-merging 的优势在于它 **hierarchically merge 相关 chunks into parent node**，保留了 patient-level context。这印证了 paper 的 core thesis：clinical reasoning 需要 patient-level coherence，不能 chunk-level 撕碎。

### 4.5 Weighting Strategy (Table 5)

Complementary weighting（task-irrelevant modalities 权重 1，task-relevant 权重 0）在 medication 上意外地最强（Acc 2.73% vs Uniform 1.8%）。这反直觉但合理：medication 的 best "experience" 来源可能不是开同样药的 patient，而是有类似 diagnosis + procedure 的 patient——因为 medication choice 是被 diagnosis/procedure 决定的，similar diagnosis 自带 similar medication pattern。

---

## 5. 与相关工作的 Positioning

### 5.1 vs ClinicalRAG / MIRAGE

[ClinicalRAG (Lu et al. 2024)](https://aclanthology.org/2024.knowllm-1.9/) 和 [MIRAGE (Xiong et al. 2024)](https://aclanthology.org/2024.findings-acl.371/) 都做 medical RAG，但它们 retrieve 的是 **medical knowledge**（drug descriptions, guidelines），不是 **patient cases**。ExPRAG 的 differentiation 在于 retrieval target 的性质：factual knowledge vs experiential cases。

### 5.2 vs EHRNoteQA

[EHRNoteQA (Kweon et al. 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/...) 是 NeurIPS 2024 D&B 的 benchmark，但它的 task 是 "extract from full discharge note"——本质上 reading comprehension，没有 reasoning over multiple patients。DISCHARGEQA 强制 model 跨 patient reasoning。

### 5.3 vs Patient Similarity Literature

Patient similarity computing 在 clinical informatics 是老话题，[Ghosn et al. 2023](https://www.nature.com/articles/s41598-023-...) 用 graph-based patient similarity，[Zhang et al. 2020](https://www.jbiomedsem.com/content/...) 用 embedding。ExPRAG 用的 Jaccard on ICD codes 是最 simple baseline，但 paper 证明了 simple + structured > complex + unstructured 在这个 task 上。

---

## 6. 局限与 Future Directions

Paper 自己承认的 limitations：
1. **只用 3 种 EHR entity**：没利用 lab results（MIMIC-IV 有 labevents 表，包含 blood test、vital signs 等 time series）
2. **Multi-option format**：不能 evaluate open-ended generation

我额外想到的几个 open questions：

1. **Jaccard 的 symmetric assumption 有问题**：patient A 有 [diabetes] 和 patient B 有 [diabetes, CKD, CHF] 的 Jaccard = 0.33，但 B 显然是 A 的 "superset"，应该比 A 和另一个只有 [diabetes] 的 patient 更 informative。可以考虑 asymmetric similarity 如 overlap coefficient $= |A \cap B| / \min(|A|, |B|)$。

2. **Code hierarchy 没利用**：ICD-10 是 hierarchical 的（`M50.12` vs `M50.1` vs `M50`），完全不同的 code 可能是同一个 disease 的不同 specificity。可以用 [ICD-10 的 tree structure](https://icd.who.int/browse10/) 做 tree-edit-distance similarity 而非 exact match。

3. **Temporal dynamics 丢失**：Jaccard 把 patient 的所有 codes flatten 成 set，丢失了 admission → discharge 的 temporal evolution。一个先有 sepsis 后有 AKI 的 patient，和先有 AKI 后有 sepsis 的 patient，clinical meaning 完全不同，但 Jaccard 给同样分数。可以引入 temporal set matching 或 [Dynamic Time Warping on code trajectories](https://ieeexplore.ieee.org/document/...)。

4. **Cohort bias**：MIMIC-IV 是 Beth Israel Deaconess 的 ICU 数据，patient population 严重偏 elderly critical care。ExPRAG retrieve 的 "experience" 可能对 outpatient / pediatric scenario 无效。

5. **Retrieval evaluation 缺少 clinical ground truth**：Table 4 用 GPT-4o-mini 当 annotator，但 LLM annotator 本身在 medical similarity 判断上 reliability 存疑。理想情况下应该用 physician annotation，但 cost 高。

6. **与 in-context learning scaling 的关系**：没有实验显示 retrieved experience 数量 vs LLM size 的 scaling law。如果 LLM 足够大（如 GPT-5），是否还能从 retrieval 获益？还是会被 parametric memory 主导？

---

## 7. Case Study 解读 (Patient 20453584)

Target patient: bilateral ulnar paresthesias + neck pain。Retrieved similar patients 通过 ICD code matching 找到：
- 25633130: C3-C4 disc herniation
- 29378221: C6-C7 herniation + radiculopathy
- 28817667: spinal stenosis + foraminal narrowing

这些 similar patients 的 discharge reports 提供了 **diagnostic pattern**：cervical disc + spinal stenosis → upper extremity neurological symptoms。LLM 据此 confirm 了 target patient 的 diagnosis 是 "C5-6 disc osteophyte complex with spinal canal narrowing"，对应 options A + G。

这个 case 完美展示了 ExPRAG 的 mechanism：**structured code matching 找到 phenotypic twins，然后从 twins 的 narrative discharge reports 里 extract diagnostic reasoning pattern**。结构化 code 负责 precision（找对人），unstructured text 负责 recall（提取推理逻辑）。

---

## 8. 对 Karpathy 的 Intuition Building

如果你要从这篇 paper 里 take away 几个 deep intuitions：

1. **Retrieval target 的 modality 决定 RAG 的 ceiling**：Text-to-text retrieval 受 stylistic noise 限制；structured-to-structured retrieval (EHR codes) 更 faithful 但 less expressive；ExPRAG 的 hybrid 把 structured 用于 filtering、unstructured 用于 generation，是好的 division of labor。

2. **"Experience" 是 case-based 而非 fact-based**：Medical reasoning 的 bottleneck 不是缺少 facts（GPT-4o 知道所有 drug facts），而是缺少 contextualized application。这与软件工程里 stack overflow 的 case-based answers vs official docs 的 fact-based answers 类似。

3. **Multi-select strict accuracy 是 brutal metric**：5 个正确选项 + 5 个 distractors，全选对才算对，random baseline 是 $1/2^{10} = 0.1\%$。Table 2 的 9.68% 看起来低，但其实已经远超 random。这暴露了 LLM 在 combinatorial reasoning 上的 weakness。

4. **Small models 与 RAG 的 tension**：Mistral-7B 在 ExPRAG 下退化的现象提示，retrieval augmentation 不是 free lunch——它需要 model 有足够的 capacity 去 integrate retrieved context。这与 instruction tuning 前 small models 不能有效 use few-shot examples 的现象同源。

5. **Complementary weighting 的反直觉胜利**：medication task 反而被 diagnosis + procedure similarity 帮助最多，说明 clinical variables 之间有 deep correlation structure，naive task-aligned retrieval 会 miss 这种 latent structure。

---

## Reference Links

- [MIMIC-IV Database](https://physionet.org/content/mimiciv/2.2/)
- [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/)
- [Faiss Library](https://github.com/facebookresearch/faiss)
- [LlamaIndex Auto-merging](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/OptimizedAutoMerging/)
- [BGE Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [ICD-10 Coding](https://www.cdc.gov/nchs/icd/icd-10-cm.htm)
- [NDC Drug Codes](https://www.fda.gov/drugs/national-drug-code-directory)
- [ColBERT Late Interaction](https://arxiv.org/abs/2004.12832)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [MIRAGE Medical RAG Benchmark](https://aclanthology.org/2024.findings-acl.371/)
- [EHRNoteQA](https://proceedings.neurips.cc/paper_files/paper/2024)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [Case-Based Reasoning Foundations](https://www.researchgate.net/publication/222830493)
- [Patient Similarity for Clinical Decision Support](https://www.jbiomedsem.com/content/11/1/12)

如果你想 dive deeper 进某个具体 direction（比如 temporal-aware patient similarity、或 multi-select LLM evaluation 的 metric design、或 EHR code hierarchy 的 embedding methods），我可以继续展开。
