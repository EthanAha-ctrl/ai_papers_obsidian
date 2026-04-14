## 一、核心概念

**Greedy Decoding（贪婪解码）** 是语言模型文本生成中最基础、最直观的解码策略。其核心思想是：**在每一个时间步，选择当前概率最高的token作为输出**。在每一步$t$，greedy decoding选择：

$$w_t^* = \arg\max_{w_t \in V} P(w_t | w_1^*, w_2^*, \ldots, w_{t-1}^*)$$

其中：
- $V$ 是词汇表
- $w_i^*$ 表示在第$i$步选择的最优token
- $\arg\max$ 返回使概率最大化的token
## 三、算法实现

greedy search的实现代码如下：

```python
def greedy_search(input_ids, node, length=5):
    if length == 0:
        return input_ids
    
    outputs = model(input_ids)
    predictions = outputs.logits
    
    # 获取预测的下一个token（使用argmax选择最高概率）
    logits = predictions[0, -1, :]  # 取最后一个位置的logits
    token_id = torch.argmax(logits).unsqueeze(0)  # 关键：选择最大概率token
    
    # 计算该token的对数概率
    token_score = get_log_prob(logits, token_id)
    
    # 将预测的token添加到输入序列
    new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=-1)
    
    # 递归调用生成下一个token
    input_ids = greedy_search(new_input_ids, current_node, length-1)
    
    return input_ids

def get_log_prob(logits, token_id):
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities)
    token_log_probability = log_probabilities[token_id].item()
    return token_log_probability
```

### 5.3 反例说明

假设有两棵搜索树：

```
        输入: "The"
       /          \
   "cat"(0.4)    "dog"(0.3)     ← Greedy选择"cat"
     /    \
 "sat"(0.1)  "ran"(0.2)
 
vs.

        输入: "The"  
       /          \
   "cat"(0.4)    "dog"(0.3)
                  /    \
            "ran"(0.5)  "sat"(0.3)
```

Greedy会选择"The cat sat"（概率：$0.4 \times 0.1 = 0.04$），但实际上"The dog ran"（概率：$0.3 \times 0.5 = 0.15$）的整体概率更高！
## 六、与其他解码策略对比

### 6.1 与Beam Search对比

| 维度 | Greedy Decoding | Beam Search |
|------|-----------------|-------------|
| **搜索空间** | 每步保留1个最优token | 每步保留$n$个最优候选序列（$n$ = beam width） |
| **时间复杂度** | $O(t \times |V|)$ | $O(t \times n \times |V|)$ |
| **空间复杂度** | $O(t)$ | $O(t \times n)$ |
| **序列质量** | 可能局部最优 | 更接近全局最优 |
| **多样性** | 低（确定性） | 低（仍是确定性） |

从文件中的beam search实验结果可见：
- Greedy输出："I have a dream of being a doctor."（序列分数：-1.16）
- Beam Search输出："I have a dream. I have a dream"（序列分数：-0.69）

Beam search找到了更高分数的序列，证明greedy的短视性导致次优结果。

### 6.2 与Sampling方法对比

| 方法 | 策略 | 多样性 | 适用场景 |
|------|------|--------|----------|
| **Greedy** | 选最高概率token | 无 | 翻译、摘要（需精确） |
| **Top-k Sampling** | 从前k个高概率token随机采样 | 中 | 创意写作 |
| **Nucleus (Top-p) Sampling** | 从累积概率达$p$的token集合中采样 | 高 | 开放式生成 |
| **Beam Search** | 保留多条候选路径 | 无 | 机器翻译 |

文件中的实验结果：
- Top-k sampling (k=20): "I have a dream job and I want to"（更自然）
- Nucleus sampling (p=0.5): "I have a dream. I'm going to"（更有创意）

---

## 七、为什么Greedy会失败？

### 7.1 概率链式分解的特性

$$P(w_1, w_2, \ldots, w_t) = \prod_{i=1}^{t} P(w_i | w_{<i})$$

最大化整体概率$P(w_1, \ldots, w_t)$ **不等于** 最大化每一步的条件概率！

### 7.2 数学解释

设两个序列：
- 序列A: $w_1^A, w_2^A$，其中每步都是最优
- 序列B: $w_1^B, w_2^B$，其中第一步稍差但整体更优

Greedy选择：
$$w_1^A = \arg\max_{w_1} P(w_1)$$

但可能：
$$P(w_1^B) \times P(w_2^B | w_1^B) > P(w_1^A) \times P(w_2^A | w_1^A)$$
## 八、实际应用场景

### 8.1 适合使用Greedy的场景

| 场景 | 原因 |
|------|------|
| **机器翻译** | 需要精确、一致的输出 |
| **文本摘要** | 要求信息准确性 |
| **代码生成** | 语法必须正确，不允许随机性 |
| **问答系统** | 答案应该确定且可复现 |
| **快速原型** | 调试和测试阶段 |

### 8.2 不适合使用Greedy的场景

| 场景 | 原因 | 推荐替代方案 |
|------|------|--------------|
| **创意写作** | 输出过于单调、重复 | Top-k/Nucleus sampling |
| **对话生成** | 缺乏多样性，用户体验差 | Temperature + Sampling |
| **故事续写** | 结果可预测，缺乏惊喜 | Nucleus sampling |
| **头脑风暴** | 需要多样化想法 | 高温度采样 |
## 九、Temperature的影响

虽然greedy decoding本身不使用temperature，但理解temperature对理解解码策略很重要：

$$\text{softmax}(x_i) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$$

其中$T$是温度参数：

| Temperature值 | 效果 | 适用场景 |
|---------------|------|----------|
| $T \to 0$ | 分布趋向one-hot，接近greedy | 精确任务 |
| $T = 1$ | 原始softmax分布 | 默认设置 |
| $T > 1$ | 分布更平坦，更多样性 | 创意生成 |

---

## 十、可视化理解：搜索树

Greedy search只沿着一条路径走：

```
          Root
           |
        Token A (最高概率)
           |
        Token B (最高概率)
           |
        Token C (最高概率)
           |
          END
```

而Beam search会探索多条路径：

```
              Root
            /   |   \
         A(0.4) B(0.3) C(0.2)  ← 保留top-3
          |     |     |
         ...   ...   ...
```

---

## 十一、代码实现细节

### 11.1 核心操作：Argmax

```python
# Greedy的核心：选择概率最大的token
token_id = torch.argmax(logits)  # 返回最大值索引
```

### 11.2 使用Hugging Face Transformers

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Greedy decoding是默认行为（num_beams=1, do_sample=False）
outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=1,      # Greedy search
    do_sample=False,  # 不采样
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 十二、研究进展与改进

### 12.1 Beam Search变体

| 变体 | 改进点 |
|------|--------|
| **Diverse Beam Search** | 引入多样性惩罚，鼓励不同beam探索不同路径 |
| **Beam Search with Length Penalty** | 避免偏好过短或过长序列 |
| **Constrained Beam Search** | 强制包含/排除特定token |

### 12.2 混合策略

现代LLM常结合多种策略：

```python
outputs = model.generate(
    input_ids,
    num_beams=5,        # 使用beam search
    do_sample=True,     # 启用采样
    top_k=50,           # Top-k采样
    top_p=0.95,         # Nucleus采样
    temperature=0.7,    # 温度调节
)
```

---

## 十三、总结

| 方面        | Greedy Decoding |     |     |
| --------- | --------------- | --- | --- |
| **核心思想**  | 每步选概率最高的token   |     |     |
| **优点**    | 快速、简单、确定性       |     |     |
| **缺点**    | 短视、可能错过全局最优     |     |     |
| **时间复杂度** | $O(t \times     | V   | )$  |
| **空间复杂度** | $O(t)$          |     |     |
| **适用任务**  | 需要精确输出的任务       |     |     |
| **不适用任务** | 需要多样性的创意生成      |     |     |
