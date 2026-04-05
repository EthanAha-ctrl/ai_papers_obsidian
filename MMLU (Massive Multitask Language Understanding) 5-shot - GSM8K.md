# MMLU (Massive Multitask Language Understanding) 5-shot 详解

## 1. MMLU Benchmark 概述

MMLU 是由 Hendrycks et al. (2021) 提出的一个大规模多任务语言理解 benchmark。其核心设计理念是测试 LLM 在 **knowledge-intensive** 和 **reasoning-intensive** 任务上的表现。

### 1.1 Dataset 架构

```
MMLU Dataset Structure
├── 57 Subjects (subjects)
│   ├── STEM (17 subjects)
│   │   ├── abstract_algebra
│   │   ├── anatomy
│   │   ├── astronomy
│   │   ├── college_physics
│   │   └── ...
│   ├── Humanities (13 subjects)
│   │   ├── philosophy
│   │   ├── world_religions
│   │   └── ...
│   ├── Social Sciences (16 subjects)
│   │   ├── economics
│   │   ├── psychology
│   │   └── ...
│   └── Other (11 subjects)
│       ├── business_ethics
│       ├── management
│       └── ...
├── Question Format: Multiple Choice (A, B, C, D)
├── Total Questions: ~16,000
└── Split: dev (few-shot examples) + test
```

### 1.2 Question 示例

```
Subject: college_physics
Question: A particle moves in a circle of radius r with angular velocity ω. 
          The centripetal acceleration is:

A. ω²r inward
B. ω²r outward  
C. ωr inward
D. ωr² inward

Answer: A
```

---

## 2. 5-shot Setting 深度解析

### 2.1 什么是 5-shot Learning

5-shot 属于 **In-Context Learning (ICL)** 的范畴，是一种 **prompt-based evaluation** 方法。

**核心公式：**

$$P(y \mid x, \mathcal{D}_{shot}) = P(y \mid x, (x_1, y_1), (x_2, y_2), \ldots, (x_5, y_5))$$

其中：
- $x$ 是当前待回答的 question
- $y$ 是预测的 answer
- $\mathcal{D}_{shot} = \{(x_i, y_i)\}_{i=1}^{5}$ 是 5 个 in-context examples
- $(x_i, y_i)$ 是第 $i$ 个 example 的 question-answer pair

### 2.2 Prompt Template 结构

```
┌─────────────────────────────────────────────────────────────┐
│                    5-shot Prompt Template                    │
├─────────────────────────────────────────────────────────────┤
│  The following are multiple choice questions (with answers) │
│  about [SUBJECT_NAME].                                       │
│                                                              │
│  Question: [EXAMPLE_1_QUESTION]                              │
│  A. [OPTION_A]                                               │
│  B. [OPTION_B]                                               │
│  C. [OPTION_C]                                               │
│  D. [OPTION_D]                                               │
│  Answer: [ANSWER_1]                                          │
│                                                              │
│  Question: [EXAMPLE_2_QUESTION]                              │
│  ...                                                         │
│  Answer: [ANSWER_5]                                          │
│                                                              │
│  Question: [TARGET_QUESTION]                                 │
│  A. [OPTION_A]                                               │
│  B. [OPTION_B]                                               │
│  C. [OPTION_C]                                               │
│  D. [OPTION_D]                                               │
│  Answer:                                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 为什么是 5-shot 而非其他数量？

这是经过大量实验验证的选择，关键考量因素包括：

**Context Window 限制：**

$$L_{prompt} = 5 \times L_{example} + L_{target} \leq L_{max}$$

其中：
- $L_{prompt}$ 是 prompt 总长度
- $L_{example}$ 是单个 example 的平均 token 长度
- $L_{target}$ 是 target question 的长度
- $L_{max}$ 是 model 的 context window 上限

**Performance vs Shot 数量的关系：**

```
Accuracy
    │
    │           ┌──────────────────────
    │          ╱
    │         ╱
    │        ╱
    │       ╱
    │      ╱
    │     ╱
    │    ╱
    │   ╱
    │  ╱
    │ ╱
    │╱
    └───────────────────────────────────
         0    1    2    3    4    5    6    7    8
                    Number of Shots
```

---

## 3. 技术细节：评估机制

### 3.1 Log-Likelihood Based Scoring

对于每个 option，model 计算其 log-likelihood：

$$\mathcal{L}(o_k) = \sum_{t=1}^{T_k} \log P(token_t^{(k)} \mid \text{context})$$

其中：
- $o_k$ 是第 $k$ 个 option（A, B, C, D），$k \in \{1, 2, 3, 4\}$
- $T_k$ 是 option $o_k$ 的 token 数量
- $token_t^{(k)}$ 是第 $k$ 个 option 的第 $t$ 个 token

**Normalized Log-Likelihood：**

$$\mathcal{L}_{norm}(o_k) = \frac{\mathcal{L}(o_k)}{T_k}$$

这样做是为了避免 length bias。

### 3.2 Prediction Selection

选择 normalized log-likelihood 最高的 option：

$$\hat{y} = \arg\max_{k \in \{A,B,C,D\}} \mathcal{L}_{norm}(o_k)$$

### 3.3 Accuracy 计算

**Per-subject accuracy：**

$$Acc_s = \frac{1}{N_s} \sum_{i=1}^{N_s} \mathbb{1}[\hat{y}_i = y_i^{*}]$$

其中：
- $Acc_s$ 是 subject $s$ 的 accuracy
- $N_s$ 是 subject $s$ 的 question 数量
- $\mathbb{1}[\cdot]$ 是 indicator function
- $\hat{y}_i$ 是第 $i$ 个 question 的预测
- $y_i^{*}$ 是第 $i$ 个 question 的 ground truth

**Overall accuracy (Macro-average)：**

$$Acc_{overall} = \frac{1}{S} \sum_{s=1}^{S} Acc_s$$

其中 $S = 57$ 是 subject 总数。

---

## 4. 实验数据与 SOTA 表现

### 4.1 各模型在 MMLU 5-shot 上的表现

| Model | Parameters | MMLU 5-shot Acc | Date |
|-------|------------|-----------------|------|
| GPT-3 (davinci) | 175B | 43.9% | 2020 |
| GPT-3.5 (text-davinci-003) | 175B | 53.6% | 2022 |
| GPT-4 | ~1.8T (MoE) | 86.4% | 2023 |
| GPT-4o | ~1.8T | 88.7% | 2024 |
| Claude 3 Opus | ~400B | 86.8% | 2024 |
| Claude 3.5 Sonnet | ~175B | 88.7% | 2024 |
| Llama 2 70B | 70B | 69.8% | 2023 |
| Llama 3 70B | 70B | 82.0% | 2024 |
| Llama 3.1 405B | 405B | 85.9% | 2024 |
| Mistral Large | 123B | 80.8% | 2024 |
| Gemini Ultra | ~1T | 90.0% | 2024 |
| DeepSeek-V3 | 671B (MoE) | 88.5% | 2024 |

### 4.2 Per-category Performance 分析

以 GPT-4 为例的详细分解：

| Category | Subjects | GPT-4 Acc | Random Baseline |
|----------|----------|-----------|-----------------|
| STEM | 17 | 84.2% | 25% |
| Humanities | 13 | 86.7% | 25% |
| Social Sciences | 16 | 88.1% | 25% |
| Other | 11 | 87.3% | 25% |

### 4.3 Difficulty Analysis

不同难度 question 的表现：

```
Difficulty Level Definition:
- Easy: >70% of humans correct
- Medium: 40-70% of humans correct  
- Hard: <40% of humans correct

Performance by Difficulty (GPT-4):
┌────────────────────────────────────────────────┐
│ Difficulty │ Model Acc │ Human Expert Acc      │
├────────────────────────────────────────────────┤
│ Easy       │ 95.2%     │ 89.3%                 │
│ Medium     │ 86.7%     │ 71.5%                 │
│ Hard       │ 75.3%     │ 45.8%                 │
└────────────────────────────────────────────────┘
```

---

## 5. In-Context Learning 的理论解释

### 5.1 ICL 作为 Implicit Bayesian Inference

**核心理论** (Xie et al., 2022)：

ICL 可以理解为 implicit Bayesian inference，model 在 inference 时隐式地进行：

$$P(y \mid x, \mathcal{D}_{shot}) = \int P(y \mid x, \theta) P(\theta \mid \mathcal{D}_{shot}) d\theta$$

其中：
- $\theta$ 是假设的任务概念 (task concept)
- $P(\theta \mid \mathcal{D}_{shot})$ 是从 few-shot examples 推断出的 posterior
- model 隐式地 marginalize over possible $\theta$

### 5.2 Induction Head 机制

**Layer-wise contribution：**

$$\text{Attention}_{\text{induction}}(q_i, k_j, v_j) = \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) v_j$$

其中：
- $q_i$ 是当前 token 的 query
- $k_j$ 是之前出现过的相同 token 的 key
- $v_j$ 是该 token 下一个 token 的 value

这形成了 "copy pattern"：如果 context 中有 "A → B"，当再次看到 "A" 时会倾向输出 "B"。

### 5.3 Shot 数量与 Performance 的理论关系

**Scaling law for ICL：**

$$Acc(n) \approx Acc_{\infty} - \frac{C}{n^{\alpha}}$$

其中：
- $n$ 是 shot 数量
- $Acc_{\infty}$ 是 shot 数量趋向无穷时的 asymptotic accuracy
- $C$ 和 $\alpha$ 是 task-specific 常数，通常 $\alpha \approx 0.5 - 0.8$

---

## 6. Implementation 细节

### 6.1 Prompt Engineering 要点

**关键设计选择：**

```python
# 标准化的 prompt format
def format_mmlu_prompt(subject, dev_examples, test_question):
    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    
    for ex in dev_examples[:5]:  # 5-shot
        prompt += f"Question: {ex['question']}\n"
        prompt += f"A. {ex['choices'][0]}\n"
        prompt += f"B. {ex['choices'][1]}\n"
        prompt += f"C. {ex['choices'][2]}\n"
        prompt += f"D. {ex['choices'][3]}\n"
        prompt += f"Answer: {ex['answer']}\n\n"
    
    prompt += f"Question: {test_question['question']}\n"
    prompt += f"A. {test_question['choices'][0]}\n"
    # ... 
    
    return prompt
```

### 6.2 Example Selection 策略

**Random vs Stratified Sampling：**

```
Random Sampling:
- 简单，reproducibility 好
- 可能导致 topic distribution 不均

Stratified Sampling:
- 确保每个 sub-topic 都有代表
- 更 robust 但实现复杂

实践中大多数 evaluation 使用 dev set 的前 5 个 examples
```

### 6.3 代码实现框架

```python
import torch
import numpy as np

def evaluate_mmlu_5shot(model, tokenizer, dataset):
    """
    MMLU 5-shot evaluation pipeline
    """
    results = {}
    
    for subject in dataset.subjects:
        subject_acc = []
        dev_examples = dataset.get_dev(subject, n=5)
        
        for question in dataset.get_test(subject):
            # 构造 prompt
            prompt = format_prompt(subject, dev_examples, question)
            
            # 计算 each choice 的 log-likelihood
            choice_scores = []
            for choice in ['A', 'B', 'C', 'D']:
                prompt_with_choice = prompt + f"\nAnswer: {choice}"
                inputs = tokenizer(prompt_with_choice, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                # 计算 normalized log-likelihood
                log_likelihood = compute_log_likelihood(logits, inputs)
                norm_ll = log_likelihood / inputs['input_ids'].shape[1]
                choice_scores.append(norm_ll)
            
            # 选择 score 最高的 choice
            predicted = ['A', 'B', 'C', 'D'][np.argmax(choice_scores)]
            subject_acc.append(predicted == question['answer'])
        
        results[subject] = np.mean(subject_acc)
    
    return {
        'overall': np.mean(list(results.values())),
        'by_subject': results
    }
```

---

## 7. MMLU 的局限性与批评

### 7.1 Data Contamination 问题

**核心担忧：**

$$P(\text{contamination}) = P(\text{question} \in \mathcal{D}_{\text{train}})$$

如果 test questions 出现在 model 的 training data 中，evaluation 就不再是 out-of-distribution。

**检测方法：**

```
N-gram Overlap Detection:
- 计算 test questions 与 training corpus 的 n-gram overlap
- 高 overlap 可能 indicate contamination

Membership Inference:
- 训练 classifier 判断某个 sample 是否在 training set 中
```

### 7.2 Multiple Choice 的局限性

**Limitation：**

```
Multiple choice 只测试 discrimination ability，不测试 generation ability

理想情况下应该测试:
- Open-ended generation
- Explanation generation  
- Multi-step reasoning chain
```

### 7.3 Western-centric Bias

**Dataset 来源分析：**

```
Source Distribution:
├── US Standardized Tests: ~60%
├── US Textbooks: ~25%
├── International Sources: ~10%
└── Non-English Translations: ~5%

这导致 bias towards:
- Western knowledge systems
- English-language concepts
- US-specific contexts (e.g., US law, US history)
```

---

## 8. MMLU Variants 与后续工作

### 8.1 MMLU-Pro

**改进点：**
- 更难的问题（增加 distractor options 从 4 个到 10 个）
- 更复杂的 reasoning 要求
- 减少 memorization 的影响

**Performance comparison：**

| Model | MMLU | MMLU-Pro |
|-------|------|----------|
| GPT-4o | 88.7% | 72.6% |
| Claude 3.5 Sonnet | 88.7% | 78.3% |
| Llama 3.1 405B | 85.9% | 63.5% |

### 8.2 MMMLU (Multilingual MMLU)

**扩展到多语言：**

$$Acc_{multilingual} = \frac{1}{|L|} \sum_{l \in L} Acc_l$$

其中 $L$ 是语言集合。

**覆盖语言：** 包含 100+ 语言版本

### 8.3 MMLU-Redux

**目的：** 重新标注和清洗原始 MMLU 数据集

**发现的问题：**
- 约 2-3% 的 questions 有 annotation errors
- 部分 questions 有 ambiguous answers
- 部分于 outdated 信息

---

## 9. 与其他 Benchmarks 的关系

### 9.1 Benchmark Ecosystem

```
                    ┌─────────────────┐
                    │  General LLM    │
                    │  Benchmarks     │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │  MMLU   │         │  HellaSwag│        │  ARC   │
   │(Knowledge)│       │(Reasoning)│        │(Science)│
   └─────────┘         └──────────┘         └────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Aggregate     │
                    │   Benchmarks    │
                    │ (HELM, Open LLM)│
                    └─────────────────┘
```

### 9.2 与 HellaSwAG 的对比

| Aspect | MMLU | HellaSwAG |
|--------|------|-----------|
| Task Type | Multiple Choice QA | Sentence Completion |
| Knowledge Required | High | Medium |
| Reasoning Required | High | High |
| Domain | Academic/Professional | Commonsense |
| Average Length | Longer | Shorter |

### 9.3 ARC (AI2 Reasoning Challenge)

**相似点：** 都是 multiple choice science questions

**不同点：**
- ARC 分为 Easy Set 和 Challenge Set
- ARC 更专注于 grade-school science
- MMLU 覆盖更广的 subject range

---

## 10. 实践应用指南

### 10.1 如何在自己的模型上评估 MMLU 5-shot

**Step-by-step:**

```bash
# 使用 lm-evaluation-harness
pip install lm-eval

# 评估 local model
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-70b-hf \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 8

# 评估 API model
lm_eval --model openai \
    --model_args model=gpt-4 \
    --tasks mmlu \
    --num_fewshot 5
```

### 10.2 结果解读

**Performance Tier Classification：**

```
Elite (85%+):    State-of-the-art models, expert-level performance
Strong (70-85%): Competitive performance, suitable for many applications
Moderate (55-70%): Useful but with notable limitations  
Weak (<55%):      Significant improvement needed
Random (25%):     No meaningful capability
```

### 10.3 Debugging 低 Performance

**常见问题与解决方案：**

```python
# 问题 1: Context window overflow
# 解决: 减少 shot 数量或截断 examples

# 问题 2: Tokenization issues  
# 解决: 确保 tokenizer 正确处理 special tokens

# 问题 3: Answer format mismatch
# 解决: 统一使用 model 的 expected format

def debug_mmlu_predictions(model, tokenizer, sample_questions):
    """分析 model 的预测模式"""
    for q in sample_questions:
        prompt = format_prompt(q)
        output = model.generate(prompt)
        
        # 分析 attention pattern
        attention = model.get_attention_weights(prompt)
        
        # 检查 model 是否正确 attending to few-shot examples
        example_attention = attention[:, :num_example_tokens].mean()
        question_attention = attention[:, num_example_tokens:].mean()
        
        print(f"Example attention: {example_attention:.3f}")
        print(f"Question attention: {question_attention:.3f}")
```

---

## 11. 前沿研究方向

### 11.1 Better Few-shot Selection

**Research question:** 如何选择最优的 5 个 examples？

**方法：**

$$\mathcal{D}_{shot}^{*} = \arg\max_{\mathcal{D}_{shot} \subset \mathcal{D}_{dev}, |\mathcal{D}_{shot}|=5} \mathbb{E}_{q \sim \mathcal{D}_{test}}[P(correct \mid q, \mathcal{D}_{shot})]$$

**Approaches:**
- **Diversity-based selection:** 最大化 selected examples 的 diversity
- **Similarity-based selection:** 选择与 test question 最相似的 examples
- **Uncertainty-based selection:** 选择 model 最 uncertain 的 examples

### 11.2 Chain-of-Thought in MMLU

**CoT Prompting for MMLU：**

```
Question: What is the capital of France?
A. London
B. Paris
C. Berlin
D. Madrid

Let's think step by step:
1. France is a country in Western Europe
2. The capital of France is Paris
3. Therefore, the answer is B

Answer: B
```

**Performance boost:** CoT 通常能提升 2-5% 的 accuracy

### 11.3 Retrieval-Augmented MMLU

**RAG for MMLU：**

$$P(y \mid x) = P(y \mid x, \text{retrieve}(x, \mathcal{K}))$$

其中 $\mathcal{K}$ 是 external knowledge base。

**效果：** 可以显著提升 knowledge-intensive subjects 的 performance

---

## 12. 总结与 Intuition Building

### 12.1 核心直觉

**MMLU 5-shot 测试的是：**

```
┌────────────────────────────────────────────────────────────┐
│                    MMLU 5-shot 本质                         │
├────────────────────────────────────────────────────────────┤
│  1. World Knowledge: 知道 facts across domains             │
│  2. Reasoning: 能 apply knowledge to new problems          │
│  3. In-Context Learning: 能从 5 个 examples 学习 pattern    │
│  4. Calibration: 知道自己知道什么和不知道什么               │
└────────────────────────────────────────────────────────────┘
```

### 12.2 5-shot vs Zero-shot 的 Trade-off

```
          Zero-shot                    5-shot
          ─────────                    ──────
Pros:     - No example bias           - Better performance
          - Pure capability test       - More stable evaluation
          - Simpler setup              - Tests ICL ability
          
Cons:     - Lower performance         - Potential example selection bias
          - Higher variance           - More context tokens needed
          - Less task-specific        - Possible contamination
```

### 12.3 何时使用 MMLU 5-shot

**适用场景：**
- Model capability comparison
- Training progress monitoring
- Domain-specific evaluation

**不适用场景：**
- Testing generation ability
- Multilingual capability
- Real-world task performance

---

## 参考文献

1. **Original MMLU Paper:**
   - Hendrycks, D., et al. (2021). "Measuring Massive Multitask Language Understanding." ICLR 2021.
   - Link: https://arxiv.org/abs/2009.03300

2. **GPT-4 Technical Report:**
   - OpenAI (2023). "GPT-4 Technical Report."
   - Link: https://arxiv.org/abs/2303.08774

3. **In-Context Learning Theory:**
   - Xie, S.M., et al. (2022). "An Explanation of In-Context Learning as Implicit Bayesian Inference." ICLR 2022.
   - Link: https://arxiv.org/abs/2111.02080

4. **MMLU-Pro:**
   - Wang, Y., et al. (2024). "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark."
   - Link: https://arxiv.org/abs/2406.01574

5. **Induction Heads:**
   - Olah, C., et al. (2022). "In-Context Learning and Induction Heads." Transformer Circuits Thread.
   - Link: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

6. **lm-evaluation-harness:**
   - Gao, L., et al. (2021). "A framework for few-shot language model evaluation."
   - Link: https://github.com/EleutherAI/lm-evaluation-harness

7. **Chain-of-Thought Prompting:**
   - Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
   - Link: https://arxiv.org/abs/2201.11903

8. **Data Contamination:**
   - Dodge, J., et al. (2021). "Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus."
   - Link: https://arxiv.org/abs/2104.08758

# GSM8K (Grade School Math 8K) Benchmark 详解

## 1. GSM8K 概述

GSM8K 是由 OpenAI 发布的一个**小学数学应用题数据集**，全称是 **Grade School Math 8K**。其核心目的是测试 LLM 的**多步推理能力**。

### 1.1 Dataset 基本信息架构

```
GSM8K Dataset Structure
├── Training Set: 7,473 questions
├── Test Set: 1,319 questions
├── Question Type: Arithmetic word problems
├── Grade Level: Elementary school (grades 1-6)
├── Solution Format: Step-by-step reasoning + Final answer
└── Answer Type: Integer or decimal number
```

### 1.2 Question 示例

```
Question: Natalia sold clips to 48 of her friends in April, and then she 
sold half as many clips in May. How many clips did Natalia sell altogether 
in April and May?

Solution (Chain-of-Thought):
Natalia sold 48 clips in April.
In May, she sold half as many, so 48 / 2 = 24 clips.
Altogether, she sold 48 + 24 = 72 clips.
Answer: 72

Question: A robe takes 2 bolts of blue fiber and half that amount of 
white fiber. How many bolts in all does it take to make 2 robes?

Solution:
For 1 robe: 2 bolts blue + (2/2) = 1 bolt white = 3 bolts total
For 2 robes: 3 × 2 = 6 bolts
Answer: 6
```

### 1.3 Dataset 统计特性

```
┌─────────────────────────────────────────────────────────────┐
│                    GSM8K Statistics                          │
├─────────────────────────────────────────────────────────────┤
│  Average solution steps: 2-8 steps                          │
│  Average question length: ~50 words                         │
│  Answer range: Integers, simple decimals                    │
│  Difficulty: Elementary but requires multi-step reasoning  │
│  Language: English only                                     │
│  Problem domains: Money, time, geometry, rates, proportions │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 8-shot Setting 深度解析

### 2.1 为什么需要 Few-shot + CoT

GSM8K 的核心挑战在于**多步推理**，而非简单的 knowledge retrieval。

**Zero-shot 的局限：**

$$P(y \mid x) = P(y \mid \text{question})$$

Zero-shot 时，model 需要：
1. 理解问题
2. 规划解题步骤
3. 执行计算
4. 输出答案

这要求极高的 reasoning capability。

**Few-shot with CoT 的增强：**

$$P(y \mid x, \mathcal{D}_{shot}) = P(y \mid x, \text{examples with reasoning chains})$$

通过 examples，model 学习：
1. How to break down problems
2. Step-by-step reasoning format
3. Calculation verification

### 2.2 标准 8-shot Prompt 结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GSM8K 8-shot Prompt Template                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Question: [EXAMPLE_1_QUESTION]                                     │
│  Answer:                                                            │
│  [STEP_1_REASONING]                                                 │
│  [STEP_2_REASONING]                                                 │
│  ...                                                                │
│  The answer is [FINAL_ANSWER_1].                                    │
│                                                                      │
│  Question: [EXAMPLE_2_QUESTION]                                     │
│  Answer:                                                            │
│  [REASONING_CHAIN_2]                                                │
│  The answer is [FINAL_ANSWER_2].                                    │
│                                                                      │
│  ... (repeat for 8 examples)                                        │
│                                                                      │
│  Question: [TARGET_QUESTION]                                        │
│  Answer:                                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Few-shot Examples 的来源与选择

**标准 practice：** 使用 GSM8K training set 的前 8 个 examples

**选择标准：**
```
Criterion for Example Selection:
├── Diversity: 覆盖不同类型的数学问题
│   ├── Addition/Subtraction
│   ├── Multiplication/Division  
│   ├── Multi-step reasoning
│   └── Real-world contexts
├── Length: 适中长度的 reasoning chain
│   ├── Too short: 不提供足够的 pattern
│   └── Too long: 浪费 context window
└── Correctness: 确保所有 examples 都是正确的
```

### 2.4 8-shot 的 Empirical 理由

**Shot 数量 vs Accuracy 的实验数据：**

| Shots | GPT-4 Accuracy | Llama-2 70B Accuracy |
|-------|----------------|----------------------|
| 0 | 78.2% | 29.6% |
| 1 | 85.6% | 42.3% |
| 2 | 88.4% | 48.7% |
| 4 | 90.1% | 53.2% |
| 8 | 92.0% | 56.8% |
| 16 | 92.3% | 58.1% |

**边际收益递减：**

$$\Delta Acc(n \to n+1) \approx \frac{C}{n^{\alpha}}, \quad \alpha \approx 0.6$$

从 8 到 16 shots 的 gain 已经很小，而 context cost 翻倍。

---

## 3. Chain-of-Thought (CoT) 的核心技术

### 3.1 CoT 的理论基础

**核心思想：** 将复杂问题分解为 intermediate reasoning steps

**形式化表示：**

设 question 为 $Q$，final answer 为 $A$，则：

$$P(A \mid Q) = \sum_{r_1, r_2, \ldots, r_k} P(A, r_1, r_2, \ldots, r_k \mid Q)$$

其中 $r_1, r_2, \ldots, r_k$ 是 reasoning chain 的 intermediate steps。

**通过 CoT 分解：**

$$P(A, r_1, \ldots, r_k \mid Q) = P(r_1 \mid Q) \times P(r_2 \mid Q, r_1) \times \cdots \times P(A \mid Q, r_1, \ldots, r_k)$$

每个 conditional probability 都比直接预测 $P(A \mid Q)$ 更容易学习。

### 3.2 CoT vs Direct Answer 的数学分析

**Direct Prediction：**

$$P(A \mid Q) \quad \text{(single prediction step)}$$

难度：需要 model 内部完成所有 reasoning，然后一次性输出。

**Chain-of-Thought：**

$$P(r_1 \mid Q) \times P(r_2 \mid Q, r_1) \times \cdots \times P(A \mid Q, r_1, \ldots, r_k)$$

难度分解：每一步只需要做一小部分 reasoning。

**Error Propagation 分析：**

假设每步的正确率为 $p$，共 $k$ 步，则：

$$P(\text{correct final answer}) \approx p^k$$

但 CoT 的优势在于：
- 每步的 $p$ 更高（因为 sub-task 更简单）
- 错误可以被后续步骤"纠正"

### 3.3 Reasoning Chain 的质量评估

**关键指标：**

$$\text{CoT Quality} = f(\text{correctness}, \text{completeness}, \text{conciseness})$$

**具体度量：**

```python
def evaluate_reasoning_chain(chain, ground_truth_solution):
    """
    评估 reasoning chain 的质量
    """
    metrics = {}
    
    # 1. Correctness: 每一步是否正确
    step_correctness = []
    for step, gt_step in zip(chain.steps, ground_truth_solution.steps):
        step_correctness.append(step.is_mathematically_correct(gt_step))
    metrics['step_accuracy'] = mean(step_correctness)
    
    # 2. Completeness: 是否遗漏必要步骤
    required_steps = extract_required_steps(ground_truth_solution)
    covered_steps = [s for s in required_steps if s in chain]
    metrics['completeness'] = len(covered_steps) / len(required_steps)
    
    # 3. Conciseness: 是否有多余步骤
    metrics['redundancy'] = len(chain) - len(ground_truth_solution)
    
    # 4. Final Answer Accuracy
    metrics['final_accuracy'] = (chain.final_answer == ground_truth_solution.answer)
    
    return metrics
```

---

## 4. 评估机制详解

### 4.1 Answer Extraction

GSM8K 的最终答案通常是数字，需要从 reasoning chain 中提取。

**标准格式：**

```
The answer is 72.
#### 72
```

**Extraction 算法：**

```python
import re

def extract_answer(generated_text):
    """
    从 generated text 中提取最终答案
    """
    # 方法 1: 提取 "#### " 后的数字
    match = re.search(r'####\s*([\d,\.]+)', generated_text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    # 方法 2: 提取 "The answer is" 后的数字
    match = re.search(r'[Tt]he answer is\s*([\d,\.]+)', generated_text)
    if match:
        return float(match.group(1).replace(',', ''))
    
    # 方法 3: 提取最后一个数字
    numbers = re.findall(r'([\d,\.]+)', generated_text)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    
    return None
```

### 4.2 Accuracy 计算

**Exact Match Accuracy：**

$$Acc = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{a}_i = a_i^*]$$

其中：
- $N$ 是 test set 大小（1319）
- $\hat{a}_i$ 是 model 对第 $i$ 个问题的预测答案
- $a_i^*$ 是 ground truth

**Relaxed Accuracy（容错）：**

$$Acc_{relaxed} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[|\hat{a}_i - a_i^*| < \epsilon]$$

通常 $\epsilon = 10^{-6}$ 或使用 relative error。

### 4.3 Syntactic vs Semantic Correctness

**Syntactic Correctness：** Final answer 是否正确

**Semantic Correctness：** Reasoning chain 是否正确

```
Example:
Question: John has 5 apples. He gives 2 to Mary. How many does he have left?

Correct reasoning:
John starts with 5 apples.
He gives away 2.
5 - 2 = 3
Answer: 3

Incorrect reasoning with correct answer:
John has 5 apples.
He gives away 2.
5 - 2 = 4  [ERROR]
But wait, let me recalculate: 5 - 2 = 3
Answer: 3

第一个是 syntactically and semantically correct
第二个是 syntactically correct but semantically partially incorrect
```

---

## 5. SOTA 模型表现

### 5.1 主要模型在 GSM8K 上的性能

| Model | Parameters | GSM8K 8-shot CoT | Date |
|-------|------------|------------------|------|
| GPT-3 (davinci) | 175B | 40.2% | 2021 |
| GPT-3.5 (text-davinci-002) | 175B | 57.5% | 2022 |
| GPT-3.5 (code-davinci-002) | 175B | 65.4% | 2022 |
| GPT-4 | ~1.8T | 92.0% | 2023 |
| GPT-4o | ~1.8T | 95.3% | 2024 |
| Claude 3 Opus | ~400B | 95.0% | 2024 |
| Claude 3.5 Sonnet | ~175B | 96.4% | 2024 |
| Llama 2 7B | 7B | 14.6% | 2023 |
| Llama 2 70B | 70B | 56.8% | 2023 |
| Llama 3 8B | 8B | 77.0% | 2024 |
| Llama 3 70B | 70B | 93.0% | 2024 |
| Llama 3.1 405B | 405B | 96.8% | 2024 |
| Mistral 7B | 7B | 37.8% | 2023 |
| Mixtral 8x7B | 47B (MoE) | 74.4% | 2024 |
| DeepSeek-V3 | 671B (MoE) | 95.2% | 2024 |
| Gemini Ultra | ~1T | 94.4% | 2024 |
| Qwen2.5-Math-72B | 72B | 96.8% | 2024 |

### 5.2 Scaling Laws for GSM8K

**Model Size vs Accuracy：**

```
Accuracy
    │
    │                           ┌───────
    │                         ╱
    │                       ╱
    │                     ╱
    │                   ╱
    │                 ╱
    │               ╱
    │             ╱
    │           ╱
    │         ╱
    │       ╱
    │     ╱
    │   ╱
    │ ╱
    └──────────────────────────────────────────
         7B    13B   34B   70B   175B   405B
                    Model Size
```

**Empirical Scaling Law：**

$$Acc(M) \approx Acc_{\infty} - \frac{C}{M^{\alpha}}$$

其中：
- $M$ 是 model size（参数数量）
- $Acc_{\infty}$ 是 asymptotic accuracy（约 97-98%）
- $\alpha \approx 0.3 - 0.4$

### 5.3 Training Data 的 Influence

**关键发现：** Code training 提升 math reasoning

| Training Data | GSM8K Accuracy |
|---------------|----------------|
| Text only | 50.2% |
| Text + Code | 65.4% |
| Text + Code + Math | 78.5% |

这表明 **"reasoning as program execution"** 的 hypothesis。

### 5.4 Error Analysis

**常见错误类型：**

```
Error Category Distribution on GSM8K:
┌─────────────────────────────────────────────────────────────┐
│ Error Type                    │ Percentage │ Example Issue │
├─────────────────────────────────────────────────────────────┤
│ Calculation errors            │ 35%        │ Arithmetic mistakes │
│ Reasoning errors              │ 30%        │ Wrong logic flow │
│ Misunderstanding questions    │ 20%        │ Wrong interpretation │
│ Unit/conversion errors        │ 10%        │ Missing unit conversion │
│ Format errors                 │ 5%         │ Incorrect output format │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 高级技术方法

### 6.1 Self-Consistency

**核心思想：** 生成多个 reasoning paths，然后投票

**算法：**

```python
def self_consistency(question, model, n_samples=40):
    """
    Self-Consistency 方法
    """
    answers = []
    reasoning_chains = []
    
    for _ in range(n_samples):
        # 生成完整的 reasoning chain
        chain = model.generate(
            question, 
            temperature=0.7,  # 引入 randomness
            max_tokens=512
        )
        answer = extract_answer(chain)
        answers.append(answer)
        reasoning_chains.append(chain)
    
    # Voting: 选择最 frequent 的答案
    from collections import Counter
    answer_counts = Counter(answers)
    final_answer = answer_counts.most_common(1)[0][0]
    
    return final_answer, reasoning_chains
```

**数学表示：**

$$\hat{a} = \arg\max_{a} \sum_{i=1}^{n} \mathbb{1}[a_i = a]$$

**效果提升：**

| Method | GSM8K Accuracy |
|--------|----------------|
| CoT (single path) | 92.0% |
| Self-Consistency (40 paths) | 94.5% |

### 6.2 Least-to-Most Prompting

**思想：** 将复杂问题分解为一系列简单问题

**示例：**

```
Original Question: 
"Ethan has 3 boxes. Each box contains 15 apples. He gives away 20 apples. 
How many apples does he have left?"

Decomposition:
Step 1: How many apples does Ethan have initially?
        3 boxes × 15 apples/box = 45 apples
        
Step 2: How many apples after giving away 20?
        45 - 20 = 25 apples

Final Answer: 25
```

**数学表示：**

设复杂问题 $Q$ 可分解为子问题 $q_1, q_2, \ldots, q_k$，则：

$$P(A \mid Q) = \prod_{i=1}^{k} P(a_i \mid q_i, a_1, \ldots, a_{i-1})$$

### 6.3 Program-of-Thoughts (PoT)

**核心思想：** 用 Python 程序来执行 reasoning

**Prompt 示例：**

```
Question: John buys 3 shirts for $15 each and 2 pants for $25 each. 
How much does he spend in total?

Answer:
```python
# Cost of shirts
shirt_cost = 3 * 15

# Cost of pants
pants_cost = 2 * 25

# Total cost
total = shirt_cost + pants_cost
print(total)
```
Output: 75
Answer: 75
```

**优势：**
- 精确的计算（避免 arithmetic errors）
- 可以处理更复杂的 operations
- 可执行验证

**效果：**

| Method | GSM8K Accuracy |
|--------|----------------|
| Standard CoT | 92.0% |
| Program-of-Thoughts | 93.8% |

### 6.4 Verification and Self-Correction

**方法：** Model 自我检查和纠正

```
Question: [QUESTION]
Answer: [INITIAL_REASONING]

Wait, let me verify:
- Step 1 check: ✓
- Step 2 check: ✗ (error found)
Let me reconsider step 2...

Corrected Answer: [CORRECTED_REASONING]
The answer is [FINAL_ANSWER].
```

**算法：**

```python
def verify_and_correct(question, initial_answer, model):
    """
    Verification-based self-correction
    """
    # Step 1: 生成初始答案
    initial_response = model.generate(question)
    initial_answer = extract_answer(initial_response)
    
    # Step 2: 让 model 验证
    verification_prompt = f"""
    Question: {question}
    Proposed Answer: {initial_response}
    Is this answer correct? If not, explain the error and provide the correct solution.
    """
    verification = model.generate(verification_prompt)
    
    # Step 3: 如果发现错误，重新生成
    if "error" in verification.lower():
        correction_prompt = f"""
        Question: {question}
        Previous incorrect reasoning: {initial_response}
        Error identified: {verification}
        Please provide the correct solution.
        """
        corrected_response = model.generate(correction_prompt)
        return extract_answer(corrected_response)
    
    return initial_answer
```

---

## 7. Implementation 实践指南

### 7.1 标准 Evaluation Pipeline

```python
import json
import re
from tqdm import tqdm

class GSM8KEvaluator:
    def __init__(self, model, tokenizer, n_shot=8):
        self.model = model
        self.tokenizer = tokenizer
        self.n_shot = n_shot
        self.few_shot_examples = self.load_few_shot_examples()
    
    def load_few_shot_examples(self):
        """加载 few-shot examples"""
        with open('gsm8k_train.json', 'r') as f:
            train_data = json.load(f)
        return train_data[:self.n_shot]
    
    def format_prompt(self, question):
        """构造 8-shot prompt"""
        prompt = ""
        for ex in self.few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        prompt += f"Question: {question}\nAnswer:"
        return prompt
    
    def evaluate(self, test_data):
        """评估整个 test set"""
        correct = 0
        results = []
        
        for sample in tqdm(test_data):
            prompt = self.format_prompt(sample['question'])
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,  # Greedy decoding
                do_sample=False
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            predicted_answer = self.extract_answer(response)
            ground_truth = self.extract_answer(sample['answer'])
            
            is_correct = (predicted_answer == ground_truth)
            correct += is_correct
            
            results.append({
                'question': sample['question'],
                'predicted': predicted_answer,
                'ground_truth': ground_truth,
                'correct': is_correct,
                'full_response': response
            })
        
        accuracy = correct / len(test_data)
        return accuracy, results
    
    def extract_answer(self, text):
        """提取数字答案"""
        # 尝试匹配 #### 格式
        match = re.search(r'####\s*(-?[\d,\.]+)', text)
        if match:
            return float(match.group(1).replace(',', ''))
        
        # 尝试匹配 "The answer is" 格式
        match = re.search(r'[Tt]he answer is\s*(-?[\d,\.]+)', text)
        if match:
            return float(match.group(1).replace(',', ''))
        
        # 提取最后一个数字
        numbers = re.findall(r'(-?[\d,\.]+)', text)
        if numbers:
            return float(numbers[-1].replace(',', ''))
        
        return None
```

### 7.2 使用 lm-evaluation-harness

```bash
# 安装
pip install lm-eval

# 评估 local model
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-70b-hf \
    --tasks gsm8k \
    --num_fewshot 8 \
    --batch_size 4

# 评估 API model
lm_eval --model openai \
    --model_args model=gpt-4 \
    --tasks gsm8k \
    --num_fewshot 8
```

### 7.3 CoT Prompt 模板最佳实践

```python
# 推荐的 CoT prompt format
COT_PROMPT_TEMPLATE = """
Solve the following math problem step by step.

{examples}

Question: {question}
Let's think step by step.
"""

# Example format
EXAMPLE_FORMAT = """
Question: {question}
Answer: {solution}
"""

# 改进版：加入 verification
VERIFIED_COT_TEMPLATE = """
Solve the following math problem step by step. After providing your answer, 
verify your reasoning.

{examples}

Question: {question}
Let's think step by step.
"""
```

---

## 8. GSM8K 与其他 Math Benchmarks 的对比

### 8.1 Math Benchmark Ecosystem

```
                    ┌──────────────────────┐
                    │   Math Benchmarks    │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐          ┌────▼─────┐          ┌────▼─────┐
   │  GSM8K   │          │  MATH    │          │  AQUA    │
   │(Elementary)│        │(Competition)│       │(Multiple Choice)│
   └──────────┘          └──────────┘          └──────────┘
        │                      │                      │
        │                      │                      │
   Difficulty:            Difficulty:           Difficulty:
   Low-Medium             High                  Medium
        │                      │                      │
   ┌────▼─────┐          ┌────▼─────┐          ┌────▼─────┐
   │  SVAMP   │          │  MATH-   │          │  Proof   │
   │(Simple)  │          │  500     │          │  Writer  │
   └──────────┘          └──────────┘          └──────────┘
```

### 8.2 GSM8K vs MATH Dataset

| Aspect | GSM8K | MATH |
|--------|-------|------|
| Difficulty | Elementary school | High school competition |
| Questions | ~8,500 | ~12,500 |
| Solution Type | Step-by-step arithmetic | Formal mathematical proofs |
| Topics | Basic arithmetic, word problems | Algebra, geometry, number theory |
| Avg. Steps | 2-8 | 5-15+ |
| SOTA Accuracy | ~97% | ~70% |

### 8.3 Difficulty Level 分析

```
Difficulty Distribution:
            GSM8K           MATH
            ──────          ────
Level 1:    45%             15%     (Basic arithmetic)
Level 2:    35%             20%     (Multi-step)
Level 3:    15%             25%     (Complex reasoning)
Level 4:    5%              25%     (Advanced concepts)
Level 5:    <1%             15%     (Competition level)
```

### 8.4 SVAMP (Simple Variants)

**目的：** 测试 model 是否真正理解 vs pattern matching

**Modification types：**

```python
# Original GSM8K question
original = "John has 5 apples. He gives 2 to Mary. How many left?"

# SVAMP variants
variant_1_addition = "John has 5 apples. He buys 2 more. How many total?"
variant_2_irrelevant = "John has 5 apples. He gives 2 to Mary. Mary likes red apples. How many does John have?"
variant_3_reordering = "He gives 2 to Mary. John has 5 apples. How many does John have left?"
```

**测试 model robustness：** 如果 model 只是 pattern matching，variants 的 accuracy 会大幅下降。

---

## 9. 深入理解：Why CoT Works

### 9.1 Emergent Ability 假说

**核心观察：** CoT ability 在 model 达到一定 size 后突然出现

```
Accuracy on GSM8K
    │
    │                              ┌───────────
    │                            ╱
    │                          ╱
    │                        ╱
    │                      ╱
    │                    ╱
    │                 ╱
    │              ╱
    │           ╱
    │        ╱
    │     ╱
    │  ╱
    │╱
    └─────────────────────────────────────────────
         1B   7B   13B   30B   70B   175B   500B
                    Model Size
```

**Emergence threshold：** 约 10B parameters

### 9.2 分步 Reasoning 的计算优势

**Direct computation complexity：**

$$C_{direct} = f(\text{reasoning complexity})$$

**Step-by-step complexity：**

$$C_{step} = \sum_{i=1}^{k} f(\text{step}_i \text{ complexity})$$

通常 $C_{step} < C_{direct}$，因为：
- 每步是更简单的 sub-problem
- 错误可以在中间步骤被发现
- Attention 可以聚焦于 relevant information

### 9.3 Attention Pattern 分析

**研究方法：** 分析 model 在 CoT 时的 attention weights

```python
def analyze_attention_pattern(model, question_with_cot):
    """
    分析 CoT reasoning 时的 attention pattern
    """
    # 获取每层的 attention weights
    attentions = model.get_attention_weights(question_with_cot)
    
    # 分析每步的 attention focus
    steps = split_into_steps(question_with_cot)
    
    for i, step in enumerate(steps):
        # 当前步骤主要关注哪些 tokens？
        step_attention = attentions[-1][i]  # 最后一层，第 i 步
        
        # 统计 attention 分布
        top_attended_tokens = get_top_attended(step_attention, k=5)
        print(f"Step {i}: focuses on {top_attended_tokens}")
```

**发现：** Model 会 "look back" 到问题陈述和之前的 intermediate steps。

### 9.4 Internal Computation Hypothesis

**假说：** CoT 允许 model 在 output space 进行 computation

```
Standard forward pass:
input → [hidden layers] → output (single token)
        (internal computation)

CoT forward pass:
input → [hidden layers] → step_1 → [hidden layers] → step_2 → ... → answer
        (externalized computation in output space)
```

**数学表示：**

Standard: $\mathbf{h}_{final} = f_{\theta}(\mathbf{x})$

CoT: $\mathbf{h}_{final} = f_{\theta}(\mathbf{x}, \text{CoT}_1, \text{CoT}_2, \ldots, \text{CoT}_k)$

---

## 10. GSM8K 的变体与扩展

### 10.1 GSM8K-Hard

**构造方法：** 原始问题的复杂变体

```
Original: "John has 5 apples. He gives 2 to Mary. How many left?"
Hard variant: "John has 5 apples. He gives 2 to Mary, then Mary returns 
1 because it was rotten. John then buys 3 more. How many does John have?"
```

### 10.2 GSM8K-Augment

**数据增强方法：**

```python
def augment_question(question):
    """
    生成问题的变体
    """
    # 数值替换
    augmented = replace_numbers(question, random_values())
    
    # 句子重排
    augmented = reorder_sentences(augmented)
    
    # 添加无关信息
    augmented = add_irrelevant_info(augmented)
    
    return augmented
```

### 10.3 Multi-lingual GSM8K

**扩展到其他语言：**

| Language | Dataset Size | GPT-4 Accuracy |
|----------|--------------|----------------|
| English | 8,500 | 92.0% |
| Chinese | 8,500 (translated) | 88.5% |
| Spanish | 8,500 (translated) | 89.2% |
| Japanese | 8,500 (translated) | 87.8% |

**Cross-lingual transfer：** 在 English few-shot 下测试其他语言的 performance。

---

## 11. 前沿研究方向

### 11.1 Curriculum Learning for Math

**思想：** 从简单问题开始，逐步增加难度

```python
def curriculum_training(model, dataset):
    """
    按难度顺序训练
    """
    # 按难度排序
    sorted_data = sort_by_difficulty(dataset)
    
    for epoch in range(num_epochs):
        for difficulty_level in ['easy', 'medium', 'hard']:
            batch = get_batch(sorted_data, difficulty_level)
            loss = model.train(batch)
```

### 11.2 Process-based Reward Models (PRM)

**传统方法：** 只奖励正确答案

$$R_{outcome} = \mathbb{1}[\text{answer is correct}]$$

**Process-based：** 奖励正确的 reasoning steps

$$R_{process} = \sum_{i=1}^{k} \mathbb{1}[\text{step}_i \text{ is correct}]$$

**效果对比：**

| Reward Type | GSM8K Accuracy |
|-------------|----------------|
| Outcome only | 78.5% |
| Process-based | 85.2% |

### 11.3 Tool-use for Math

**思想：** 让 model 使用外部 calculator

```python
TOOL_USE_PROMPT = """
You have access to a calculator. Use it for complex calculations.

Question: {question}

Think step by step. When you need to calculate, write:
CALCULATE: [expression]

For example:
CALCULATE: 234 * 567
CALCULATOR_RESULT: 132678

Continue with your reasoning.
"""
```

### 11.4 Formal Verification

**目标：** 数学地证明 solution 的正确性

```python
def formal_verification(question, solution):
    """
    将 solution 转换为形式化表示并验证
    """
    # 转换为 symbolic representation
    symbolic = convert_to_symbolic(solution)
    
    # 使用 theorem prover 验证
    is_valid = theorem_prover.verify(symbolic)
    
    return is_valid
```

---

## 12. 总结与 Intuition Building

### 12.1 GSM8K 测试的核心能力

```
┌─────────────────────────────────────────────────────────────────┐
│                    GSM8K Core Capabilities                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Multi-step Reasoning: 分解复杂问题为简单步骤              │
│  2. Arithmetic Accuracy: 精确执行数学运算                     │
│  3. Language Understanding: 理解自然语言描述的问题            │
│  4. Planning: 规划解决步骤的顺序                              │
│  5. Verification: 检查中间结果的合理性                        │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 关键 Takeaways

**1. Few-shot + CoT 是关键组合：**
- Few-shot 提供 task format 和 reasoning pattern
- CoT externalizes internal computation

**2. Error 主要来源：**
- Arithmetic calculation errors（~35%）
- Reasoning logic errors（~30%）
- Question misunderstanding（~20%）

**3. Scaling 效应显著：**
- 小 model (<10B) 几乎无法解决
- 大 model (>100B) 可以达到 expert level

**4. Code training 有助于 math：**
- Program execution 和 math reasoning 有共同结构
- Code-trained model 在 GSM8K 上表现更好

### 12.3 评估 Model 的建议

**推荐 protocol：**

```python
def comprehensive_gsm8k_eval(model):
    """
    全面的 GSM8K 评估
    """
    results = {}
    
    # 1. Standard 8-shot CoT
    results['8shot_cot'] = evaluate_standard(model, n_shot=8)
    
    # 2. Zero-shot CoT
    results['0shot_cot'] = evaluate_zero_shot(model)
    
    # 3. Self-consistency (robustness check)
    results['self_consistency'] = evaluate_self_consistency(model, n_paths=40)
    
    # 4. Error analysis
    results['error_analysis'] = analyze_errors(model)
    
    # 5. Difficulty breakdown
    results['by_difficulty'] = evaluate_by_difficulty(model)
    
    return results
```

### 12.4 GSM8K 的局限性

```
Limitations:
├── Language: Only English
├── Math level: Elementary only
├── Question type: Word problems (not formal math)
├── Coverage: Limited to arithmetic reasoning
└── Memorization risk: Some questions may be in pretraining data
```

---

## 参考文献

1. **Original GSM8K Paper:**
   - Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168.
   - Link: https://arxiv.org/abs/2110.14168

2. **Chain-of-Thought Paper:**
   - Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.
   - Link: https://arxiv.org/abs/2201.11903

3. **Self-Consistency:**
   - Wang, X., et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023.
   - Link: https://arxiv.org/abs/2203.11171

4. **Program-of-Thoughts:**
   - Chen, W., et al. (2022). "Program of Thoughts Prompting: Disambiguating Arithmetic Word Problems." arXiv:2211.12561.
   - Link: https://arxiv.org/abs/2211.12561

5. **MATH Dataset:**
   - Hendrycks, D., et al. (2021). "Measuring Mathematical Problem Solving With the MATH Dataset." NeurIPS 2021.
   - Link: https://arxiv.org/abs/2103.03874

6. **Process Reward Models:**
   - Lightman, H., et al. (2023). "Let's Verify Step by Step." arXiv:2305.20050.
   - Link: https://arxiv.org/abs/2305.20050

7. **Least-to-Most Prompting:**
   - Zhou, D., et al. (2022). "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models." ICLR 2023.
   - Link: https://arxiv.org/abs/2205.10625

8. **SVAMP Dataset:**
   - Patel, A., et al. (2021). "Are NLP Models Really Able to Solve Simple Math Word Problems?" NAACL 2021.
   - Link: https://arxiv.org/abs/2103.07191

9. **lm-evaluation-harness:**
   - Gao, L., et al. (2021). "A framework for few-shot language model evaluation."
   - Link: https://github.com/EleutherAI/lm-evaluation-harness

10. **GPT-4 Technical Report:**
    - OpenAI (2023). "GPT-4 Technical Report."
    - Link: https://arxiv.org/abs/2303.08774