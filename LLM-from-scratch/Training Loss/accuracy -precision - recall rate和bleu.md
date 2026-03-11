我来详细解释ML评估中这几个重要的评估指标：

## 1. Accuracy（准确率）

**定义和公式：**

Accuracy是所有预测正确的样本占总样本的比例：

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

其中：
- TP (True Positive)：正确预测为正类的样本数
- TN (True Negative)：正确预测为负类的样本数
- FP (False Positive)：错误预测为正类的样本数（第一类错误）
- FN (False Negative)：错误预测为负类的样本数（第二类错误）

**Confusion Matrix示例：**

```
                  Predicted
                Positive   Negative
Actual  Positive    TP        FN
        Negative    FP        TN
```

**适用场景和局限性：**

适用于类别平衡的分类任务。当数据集类别不平衡时，Accuracy可能会产生误导性结果。

**示例计算：**

假设我们有1000个样本的Binary Classification：

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | 400 | 50 |
| Actual Negative | 30 | 520 |

Accuracy = (400 + 520) / 1000 = 0.92 = 92%

**代码示例（Scikit-learn）：**

```python
from sklearn.metrics import accuracy_score
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
accuracy = accuracy_score(y_true, y_pred)  # 0.8
```

[参考链接](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)

## 2. Precision（精确率/查准率）

**定义和公式：**

Precision衡量的是预测为正类的样本中，真正为正类的比例：

```
Precision = TP / (TP + FP)
```

**含义：**
- Precision越高，表示False Positive越少
- 适用于需要避免False Positive的场景（如垃圾邮件检测）

**示例计算：**

使用上面的Confusion Matrix：
Precision = 400 / (400 + 30) = 400/430 ≈ 0.9302 = 93.02%

**代码示例：**

```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred, average='binary')
```

[参考链接](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

## 3. Recall Rate（召回率/查全率）

**定义和公式：**

Recall衡量的是实际为正类的样本中，被正确预测为正类的比例：

```
Recall = TP / (TP + FN)
```

也称为Sensitivity（敏感性）或True Positive Rate（TPR）。

**含义：**
- Recall越高，表示False Negative越少
- 适用于需要避免False Negative的场景（如疾病诊断、欺诈检测）

**示例计算：**
Recall = 400 / (400 + 50) = 400/450 ≈ 0.8889 = 88.89%

**与Precision的权衡：**

Precision和Recall之间存在trade-off关系，通过调整分类阈值可以影响这两个指标：

```
Threshold ↑ → Precision ↑, Recall ↓
Threshold ↓ → Precision ↓, Recall ↑
```

**F1-Score（调和平均）：**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**代码示例：**

```python
from sklearn.metrics import recall_score, f1_score
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

[参考链接](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)

## 4. BLEU（Bilingual Evaluation Understudy）

**定义：**

BLEU是机器翻译和文本生成任务的评估指标，通过计算生成文本与参考文本的n-gram匹配度来评估质量。

**核心公式：**

```
BLEU = BP × exp(∑(w_n × log(p_n)))
```

其中：
- BP (Brevity Penalty) = 惩罚过短输出的因子
- p_n = n-gram的精确率
- w_n = n-gram的权重（通常设为1/N）

**Modified Precision for n-grams：**

```
p_n = (∑C∈{Candidates} ∑n-gram∈C Count_clip(n-gram)) /
      (∑C∈{Candidates} ∑n-gram∈C Count(n-gram))
```

其中Count_clip的计算：
```
Count_clip(n-gram) = min(Count(n-gram), Max_Ref_Count(n-gram))
```

**Brevity Penalty公式：**

```
BP = {
    1, if c > r
    exp(1 - r/c), if c ≤ r
}
```

其中：
- c = candidate长度
- r = 最接近的reference长度

**计算步骤示例：**

假设：
- Candidate: "the cat is on the mat"
- Reference: "the cat is on the mat"

**Unigram (1-gram)计算：**

| Word | Candidate Count | Reference Count | Clipped Count |
|------|-----------------|-----------------|---------------|
| the | 2 | 2 | 2 |
| cat | 1 | 1 | 1 |
| is | 1 | 1 | 1 |
| on | 1 | 1 | 1 |
| mat | 1 | 1 | 1 |

p1 = (2+1+1+1+1) / 6 = 6/6 = 1.0

**Bigram (2-gram)计算：**

| Bigram | Count | Clipped Count |
|--------|-------|---------------|
| the cat | 1 | 1 |
| cat is | 1 | 1 |
| is on | 1 | 1 |
| on the | 1 | 1 |
| the mat | 1 | 1 |

p2 = 5/5 = 1.0

**BLEU-4计算（常用）：**

```python
import nltk
from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['the', 'cat', 'is', 'on', 'the', 'mat']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
```

**BLEU变体：**
- **BLEU-1**：只考虑unigram
- **BLEU-2**：考虑unigram和bigram
- **BLEU-4**：考虑unigram到4-gram（最常用）
- **BLEU-5, BLEU-6**：更高阶的n-gram

**限制和改进：**
- 不考虑语义相似性，只基于n-gram重叠
- 对参考翻译质量敏感
- 改进版本包括Smoothed BLEU, Sentence BLEU等

[参考链接](https://www.nltk.org/api/nltk.translate.bleu_score.html)
[论文链接](https://www.aclweb.org/anthology/P02-1040.pdf)

## 5. 指标关系和综合使用

**Precision-Recall曲线：**

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
```

**ROC曲线和AUC：**

```
TPR = Recall = TP / (TP + FN)
FPR = FP / (FP + TN)
```

AUC (Area Under Curve) 衡量分类器在不同阈值下的整体性能。

**综合评估表：**

| 场景 | 推荐指标 | 说明 |
|------|---------|------|
| 平衡分类 | Accuracy | 基础评估 |
| 不平衡分类 | F1-Score, Precision, Recall | 细化评估 |
| 医疗诊断 | Recall (高) | 避免漏诊 |
| 垃圾邮件检测 | Precision (高) | 避免误判 |
| 机器翻译 | BLEU, METEOR, ROUGE | 序列生成评估 |
| 信息检索 | Precision@k, MAP, NDCG | 排序质量评估 |

[综合参考](https://en.wikipedia.org/wiki/Precision_and_recall)

希望这些详细的技术讲解能够帮助你理解这些重要的ML评估指标！