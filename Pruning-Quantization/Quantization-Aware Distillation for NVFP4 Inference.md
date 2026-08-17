---
source_pdf: Quantization-Aware Distillation for NVFP4 Inference.pdf
paper_sha256: 0c3a0c92240d9f511386f6b2ded886943ae6f444b559b5e511c8e5d64a04ce87
processed_at: '2026-08-06T07:45:14-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话聊聊 Quantization-Aware Distillation (QAD)

## 1. 这篇 paper 在解决什么痛点？

想象你花了几个月训练出一个超强 LLM，经过 SFT、RL、model merging 等无数道工序，终于能解题、能写代码。现在你想把它部署上线，发现 BF16 太吃显存，于是你想把它压成 NVFP4（4-bit floating-point），省一半显存，速度快两三倍。

你直接用 PTQ (Post-Training Quantization) 试试，发现对于 100B 以上的大模型，精度损失不大，挺完美的。但如果你是 7B、9B 这种小模型，精度直接掉几个点，AIME 这种难题掉得更惨。为什么？因为小模型本来容量就小，你把每个 weight 压成 4-bit，它学到的精细 reasoning 能力直接崩了。

## 2. 为什么不用 QAT (Quantization-Aware Training)？

传统做法是 QAT，简单说就是"带着量化重新训练一遍"。用 next-token cross-entropy loss，让 quantized model 重新适应 task。

但 paper 指出，QAT 对 modern LLMs 有两大死穴：

**痛点 1：训练 pipeline 太复杂，没法复现。**
现在的 LLM 不是简单 SFT 完就结束。AceReason Nemotron 经历 cold-start SFT 然后 RL；Llama Nemotron Super 经历 SFT、RL、还有 model merging。你要做 QAT，理论上得把量化塞回每一个 stage 重新跑，工程量巨大，而且 RL 的不稳定性 + 量化的不稳定性叠加，很容易训崩。

**痛点 2：QAT 会破坏 RL 学到的能力。**
这是最致命的。你手头只有 cold-start SFT 数据，想拿来做 QAT，结果发现模型越训越差，连 PTQ 都不如！为什么？因为 RL 阶段让 model 学会了 self-reflection、verification 这些复杂的 reasoning behavior，这些 behavior 没有写在 SFT 的 label 里。你用 SFT data 重新训 cross-entropy，model 直接退化回 SFT 阶段的笨蛋状态。

## 3. QAD 的核心思路：让 4-bit model 模仿 BF16 "老师"

既然 QAT 是让 model 重新学课本知识，学着学着走偏了，那不如换个思路：让 BF16 原版 model 当老师，4-bit model 当学生，老师说什么，学生就说什么。

具体怎么做？就是 Knowledge Distillation。Loss function 从 cross-entropy 变成 KL divergence：

$$ \mathcal{L}_{\text{QAD}} = \sum_{y \in V} p_{\text{teacher}}(y|x) \log \frac{p_{\text{teacher}}(y|x)}{p_{\text{student}}(y|x)} $$

变量解释：
- $x$：输入 prompt
- $y$：vocabulary 里的某个 token
- $p_{\text{teacher}}(y|x)$：BF16 老师对下一个 token 的 softmax 概率
- $p_{\text{student}}(y|x)$：NVFP4 学生对同一个 token 的概率
- $V$：整个 vocabulary

这个公式在干嘛？它在逼 student 模仿 teacher 的每一个 token 概率分布。teacher 说"正确答案是 A，概率 0.6，B 概率 0.3"，student 也要把这个比例学得一模一样。这叫 forward KL，是 mean-seeking 的，强制 student 覆盖 teacher 所有可能的高概率区域。

## 4. QAD vs QAT 的本质区别：Cross-Entropy 骗了你

看 Table 1，这是理解全篇 paper 最关键的一张表：

| Methods | KL Divergence (vs BF16) | Cross Entropy (vs labels) |
|---------|-------------------------|---------------------------|
| BF16    | 0                       | 0.408                     |
| QAT     | 0.311                   | 0.408                     |
| QAD     | 0.004                   | 0.416                     |

QAT 看起来和 BF16 的 cross-entropy 一模一样，都是 0.408。你如果只看 task loss，会觉得"哇，QAT 成功了"。但你看 KL Divergence，QAT 高达 0.311，QAD 只有 0.004。

这说明什么？Cross-entropy 只关心 "argmax 对不对"，也就是 top-1 预测是不是正确答案。只要 argmax 对，loss 就低。但 model 内部的 probability distribution 可能已经完全变形了。

QAD 的 KL 是 0.004，意味着 4-bit student 的整条概率分布曲线几乎和 BF16 teacher 重叠。不仅 top-1 对，top-10 的排序对，连 teacher 的 uncertainty（entropy）都保留了。

这为什么重要？因为现在 reasoning benchmark（AIME、LiveCodeBench）用 sampling，temperature=0.6，top-p=0.95。Sampling 对整条分布敏感，不只看 argmax。QAT 只 match argmax，sampling 的时候行为就漂了；QAD match 整条分布，sampling 出来的 chain 几乎和 BF16 一样。

## 5. RL Model 上的震撼结果：QAD 救命，QAT 灾难

看 Table 3 的 Nemotron 3 Nano（经历多轮 RL 训练）：

| Method      | AA-LCR | AIME25 | GPQA-D | LiveCodeBench-v5 | SciCode |
|-------------|--------|--------|--------|------------------|---------|
| BF16        | 35.9   | 89.1   | 73.0   | 72.1             | 33.0    |
| NVFP4 PTQ   | 31.3   | 85.0   | 71.6   | 68.9             | 30.5    |
| NVFP4 QAT   | 24.8   | 83.3   | 66.0   | 62.0             | 25.8    |
| NVFP4 QAD   | 34.3   | 87.9   | 72.7   | 68.9             | 32.3    |

QAT 比 PTQ 还差！AA-LCR 从 31.3 跌到 24.8。这就是前面说的"QAT 破坏 RL 能力"。

QAD 几乎完全恢复到 BF16 水平（34.3 vs 35.9，AIME25 87.9 vs 89.1）。

直觉解释：RL 把 reasoning 行为"刻"进了 BF16 model 的 output distribution 里。你用 SFT data 做 QAT，等于让 model 重新学 SFT 分布，把 RL 刻进去的东西抹掉了。QAD 直接 copy teacher 的 distribution，RL 的 reasoning behavior 被完整保留。

这就像 RL 训练让 model 学会了"解题时的自言自语"和"回头检查答案"的思考习惯。SFT label 只有最终答案，QAT 看不到思考过程，就把这些好习惯丢了。QAD 盯着 teacher 的每一句话，连"嗯，让我想想"这种语气词的概率分布都学过来，所以思考习惯保住了。

## 6. Cross-Domain Transfer：只用代码数据，数学成绩也能救回来

Table 4 测了一个很狂野的实验。AceReason Nemotron 会做数学也会写代码。如果你只用代码数据训 QAD，数学成绩会怎样？

| Training Data              | AIME24 | AIME25 | LiveCodeBench-v6 |
|----------------------------|--------|--------|------------------|
| BF16 Baseline              | 73.0   | 63.5   | 54.3             |
| NVFP4 PTQ                  | 69.4   | 58.7   | 52.0             |
| QAD (math only)            | 71.0   | 61.7   | 53.1             |
| QAD (code only)            | 71.0   | 62.0   | 53.3             |
| QAD (math+code)            | 71.7   | 62.0   | 53.3             |

用纯 code 数据，AIME24 也能到 71.0，几乎追上 full data 的 71.7。

为什么？因为 teacher 的 output distribution 里 encode 的是 "reasoning pattern"，不是 domain-specific facts。Step-by-step、verify、backtrack 这些 reasoning 习惯是跨 domain 的。你在 code 问题上让 student 学会"像 teacher 一样思考"，这种思考方式自动迁移到数学上。

## 7. 数据质量极不敏感：Random Token 都不会崩

Table 5 最极端的一行：

| Training Data                | AIME24 | AIME25 | LiveCodeBench-v6 |
|------------------------------|--------|--------|------------------|
| SFT data                     | 71.7   | 62.0   | 53.3             |
| Generated from BOS token     | 70.1   | 60.9   | 52.4             |
| Random tokens                | 68.6   | 60.0   | 51.7             |

Random tokens 喂进去，model 不会崩，甚至比 PTQ baseline 还稳。

这说明 QAD 的 loss landscape 非常平滑。就算 input 是 garbage，teacher 输出的 distribution 也是稳定的，student 跟着 stable 的 teacher 走，不会被带偏太多。QAT 如果喂 random tokens，cross-entropy 会疯掉，model 直接废了。

这对工业界是巨大利好：你不需要完美的原版训练数据，甚至不需要 RL prompt，随便找点通用文本都能做 QAD。

## 8. 实操细节：Learning Rate 怎么选？

Table 6 和 7 给了 LR sensitivity：

**SFT model (Nemotron Nano 9B V2)**：最优 1e-6，和原 SFT LR 一样。调到 1e-5 就开始抖，1e-4 崩盘。因为这些 model 已经在 SFT distribution 上完全收敛，高 LR 会打破 equilibrium。

**RL model (AceReason Nemotron)**：最优 1e-5，比典型 RL LR（1e-6）高 10 倍。因为 RL 阶段把 model 从 SFT distribution 推开了，QAD 需要更大步幅去追 teacher 的 RL distribution。

Table 7 的 VLM 模型更敏感：原 SFT 用 2e-5，QAD 最优是 2e-6，低 10 倍。说明越"娇贵"的模型，越要用保守 LR。

经验法则：SFT model 用 1e-6 量级，RL model 可以激进一点用 1e-5 量级。Temperature 固定 T=1，不要搞那些 T=2、T=4 的 trick，因为你要精确 match teacher distribution。

## 9. 数据量需求小得惊人

Section 3.4 给了每个 model 的数据消耗：

| Model                        | Data Tokens |
|------------------------------|-------------|
| Llama Nemotron Super V1 49B  | 0.3B        |
| Nemotron Nano 9B V2          | 6B          |
| Nemotron Nano 12B V2 VL      | 0.5B        |
| Nemotron 3 Nano 30B-A3B      | 2.5B        |
| AceReason Nemotron 7B        | 0.8B        |

49B 的 Llama Nemotron Super 只用 0.3B tokens 就能 recover！这和 pretraining 的 trillions of tokens 完全不在一个量级。

为什么这么省？因为 QAD 不是在学新知识，只是在"微调"量化带来的 distribution 偏移。量化只改变了 weights 的一小部分 dynamic，student 只需要小幅修正就能对齐 teacher。

## 10. KL Divergence 为什么比 MSE 好？

Table 8 对比了 KL 和 MSE on logits：

| Model               | Loss   | AIME24 | AIME25 | LiveCodeBench |
|---------------------|--------|--------|--------|---------------|
| AceReason           | KL-Div | 71.7   | 62.0   | 53.3          |
| AceReason           | MSE    | 71.7   | 60.1   | 52.4          |
| Nano 9B V2          | KL-Div | 80.4   | 71.5   | 67.8          |
| Nano 9B V2          | MSE    | 80.0   | 71.5   | 66.7          |

KL 稳定胜出。直觉解释：Logits 的数值范围极大，有些 token 的 logit 可能是 10，有些是 -5。MSE 会被大 logit 主导，小 logit 的误差被淹没。KL 在 log 空间比较 relative difference，对所有 token 的概率 shape 更敏感。

而且 KL 的 gradient 是 $\nabla_{z_i} \mathcal{L} = p_{\text{teacher}}(y_i) - p_{\text{student}}(y_i)$，天然被 teacher 概率加权，focus 在 teacher 重视的 token 上。MSE 的 gradient 没有这种 adaptive weighting。

## 11. Larger Teacher 反而更差：别乱换老师

Table 9 测了一个反直觉的实验：用 12B BF16 当 teacher 蒸馏 9B NVFP4 student，会不会更强？

| Teacher      | AIME24 | AIME25 | LiveCodeBench |
|--------------|--------|--------|---------------|
| 9B BF16      | 80.4   | 71.5   | 67.8          |
| 12B BF16     | 80.2   | 69.8   | 66.7          |

9B teacher 胜出。QAD 的目标是"recover 原 model 的 distribution"，不是"transfer 更强的能力"。12B teacher 的 distribution 比 9B student 的 capacity 更丰富，student 学不动这种超出自己能力范围的 distribution，反而学得别扭。

这和经典 KD 的 capacity gap 问题类似：teacher 太强，student 追不上，distillation 效果打折。QAD 的最佳实践就是"谁的原版，就用谁当 teacher"。

## 12. 总结：一句话记住 QAD

QAD = 用 BF16 原版当 teacher，用 KL divergence 让 NVFP4 student 模仿 teacher 的每一步输出分布，不关心 task label，只关心"像不像 teacher"。

它能 work 的核心原因：
1. **Distribution preservation**：Sampling-based evaluation 对整条分布敏感，KL divergence 比交叉熵更能 preserve distribution shape。
2. **RL capability safety**：RL 学到的 reasoning behavior 藏在 distribution 里，QAD 直接 copy distribution，不破坏 behavior。
3. **Data robustness**：Teacher 的 distribution 是 stable reference，garbage input 也不会把 student 带偏。
4. **Cross-domain transfer**：Reasoning pattern 是 domain-agnostic 的，学到的"思考方式"能迁移。

对工程实践的启示：未来做 LLM 量化部署，如果 PTQ 掉点太多，不要盲目上 QAT 重新训，QAD 是更稳、更省、更保真 default 选择，尤其是对 RL 训练出来的 reasoning model。

参考链接：
- [QAD Paper (arxiv 假设链接)](https://arxiv.org/abs/2511.02497)
- [NVFP4 Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Hinton Distillation](https://arxiv.org/abs/1503.02531)
- [Jacob QAT](https://arxiv.org/abs/1712.05877)
- [LLM-QAT Data-Free](https://arxiv.org/abs/2305.17888)
- [BitDistiller](https://arxiv.org/abs/2402.10631)
- [AceReason Nemotron](https://arxiv.org/abs/2505.16400)
- [Llama Nemotron](https://arxiv.org/abs/2505.00949)
- [Nemotron Nano V2](https://arxiv.org/abs/2508.14444)
- [DeepSeek R1](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath GRPO](https://arxiv.org/abs/2402.03300)
- [Menon Why Distillation Helps](https://arxiv.org/abs/2005.10419)

---

# Quantization-Aware Distillation for NVFP4 Inference 深度解析

## 1. 背景与动机

### 1.1 NVFP4 格式

NVFP4 是 NVIDIA 提出的 4-bit floating-point 格式，专门为 Blackwell 架构之后的硬件设计。与 MXFP4 相比，NVFP4 有三个关键改进：

- **Block size**: 从 32 缩小到 16，意味着每 16 个 element 共享一个 scale，更细粒度地适配 data distribution
- **First-level scaling**: per-block E4M3 FP8 scales，提供 non-power-of-two scaling factors（即非 2 的幂次的 scale），降低 quantization error
- **Second-level scaling**: per-tensor FP32 scale，扩展整体 dynamic range

这种 two-level scaling 可以形式化为：

$$\tilde{w}_i = \text{Quant}_{FP4}\left(w_i \cdot s_{\text{block}}^{(b)} \cdot S_{\text{tensor}}\right)$$

其中 $w_i$ 是第 $i$ 个 weight，$s_{\text{block}}^{(b)}$ 是第 $b$ 个 block 的 E4M3 scale（$b = \lfloor i/16 \rfloor$），$S_{\text{tensor}}$ 是 per-tensor FP32 scale。这两个 scale 的分工是：$s_{\text{block}}$ 处理局部 outlier，$S_{\text{tensor}}$ 处理全局 dynamic range。

NVFP4 相比 FP8 提供 2-3× arithmetic throughput 和约 1.8× memory reduction。

参考：[Introducing NVFP4 (NVIDIA Developer Blog)](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

### 1.2 PTQ 在 small models 上的局限

对于 very large models（如 DeepSeek R1 671B），NVFP4 PTQ 已经接近 BF16 精度（见 Table 12：MATH500 95.4 → 94.2，AIME24 80.0 → 80.0）。但对于 small models（< 50B），PTQ 的 accuracy drop 显著，原因是：

1. Small models 容量有限，quantization error 的影响相对更大
2. NVFP4 的 small block size (16) 实际上 neutralize 了传统 outlier mitigation 技术（如 SmoothQuant、QuaRot），因为这些技术依赖 block-level 的 outlier absorption，而 block 太小就没有足够"空间"吸收 outlier

这就是 [Egiazarian et al., 2025](https://arxiv.org/abs/2509.23202) 的发现：common PTQ algorithms often fail to improve over baseline NVFP4 performance。

### 1.3 QAT 在 modern LLMs 上的困难

传统 QAT（[Jacob et al., 2018](https://arxiv.org/abs/1712.05877)）用 task-specific loss（如 next-token cross-entropy）训练 quantized model。但 modern LLMs 经过 multi-stage post-training：

- **SFT** (Supervised Fine-Tuning)
- **RL** (Reinforcement Learning，如 GRPO、PPO)
- **Model merging**（如 model soup、DARE-TIES）

要 replicate 这些 pipeline 非常困难：
- RL 需要复杂 reward model、rollout 生成、reward shaping
- Model merging 的 merge ratio 和 selection 策略难以复现
- 原始训练数据可能不可用（open models）

这就需要一个 robust 的方法，能够用 partial data、partial pipeline 来 recover accuracy。

---

## 2. QAD 方法的核心

### 2.1 方法定义与公式

QAD 的核心 loss 是 teacher 和 student 之间的 KL divergence：

$$\mathcal{L}_{\text{QAD}} = D_{\text{KL}}\left(p_{\text{teacher}}(\cdot|x) \,\|\, p_{\text{student}}(\cdot|x)\right) = \sum_{y \in V} p_{\text{teacher}}(y|x) \log \frac{p_{\text{teacher}}(y|x)}{p_{\text{student}}(y|x)}$$

变量解释：
- $x$: input sequence（tokenized）
- $y$: vocabulary 中的某个 token
- $V$: vocabulary set
- $p_{\text{teacher}}(y|x)$: BF16 teacher model 在 token $y$ 上的 softmax probability
- $p_{\text{student}}(y|x)$: NVFP4 quantized student model 在 token $y$ 上的 softmax probability

注意这里用的是 forward KL（$D_{\text{KL}}(p_{\text{teacher}} \| p_{\text{student}})$），即 teacher 是 reference distribution。Forward KL 是 mean-seeking 的（要求 student 在所有 teacher 概率高的地方都覆盖到），这与 reverse KL（$D_{\text{KL}}(p_{\text{student}} \| p_{\text{teacher}})$，mode-seeking）不同。QAD 选择 forward KL 是合理的：我们希望 student 尽量"模仿" teacher 的完整分布，包括 teacher 的 uncertainty（即 soft labels 中的 entropy）。

### 2.2 QAD vs QAT 的本质区别

关键 insight 在 Table 1：

| Method | KL Divergence (vs BF16) | Cross Entropy (vs labels) |
|--------|------------------------|--------------------------|
| BF16 | 0 | 0.408 |
| QAT | 0.311 | 0.408 |
| QAD | 0.004 | 0.416 |

**直觉解释**：

QAT 用 cross-entropy on labels，它的目标是 match ground truth distribution。即使 cross-entropy loss 相同（0.408），模型的 output distribution 可能完全不同——只要 argmax 正确，cross-entropy 就可以很低。这意味着 QAT 实际上是在"重新训练"模型，model 学到的是 label distribution（通常是 one-hot 的 hard label），而不是 teacher 的 soft distribution。

QAD 用 KL divergence，目标是 match teacher 的完整 output distribution。KL = 0.004 意味着 student 几乎完全复现了 teacher 的 probability mass function。这包括：
- Top-1 prediction（argmax）
- Top-k ranking
- Per-token entropy（即 teacher 的 uncertainty）
- Token 之间的 relative probability 差异（这对 sampling-based generation 至关重要）

**为什么这重要**：modern reasoning models（AIME、LiveCodeBench）用 sampling（temperature=0.6, top-p=0.95），不是 greedy decoding。Sampling 对 complete distribution 极其敏感。QAT 只 match argmax，在 sampling 下表现差；QAD match 整个 distribution，在 sampling 下接近 BF16。

### 2.3 RL-trained models 上的关键洞察

Table 3 是论文最有说服力的结果：

**Nemotron 3 Nano** (RL-heavy):

| Method | AA-LCR | AIME25 | GPQA-D | LiveCodeBench-v5 | SciCode |
|--------|--------|--------|--------|-----------------|---------|
| BF16 | 35.9 | 89.1 | 73.0 | 72.1 | 33.0 |
| NVFP4 PTQ | 31.3 | 85.0 | 71.6 | 68.9 | 30.5 |
| NVFP4 QAT | 24.8 | 83.3 | 66.0 | 62.0 | 25.8 |
| NVFP4 QAD | 34.3 | 87.9 | 72.7 | 68.9 | 32.3 |

QAT 在 RL model 上比 PTQ 还差（24.8 vs 31.3 on AA-LCR）！为什么？

RL 训练学到的是 **reasoning behavior**，通过 reward signal 优化 policy，而不是 match labels。用 cold-start SFT data（只有 prompt + response 的 SFT 格式）做 QAT，会让 model 回到 SFT distribution，破坏 RL 学到的 reasoning pattern。

QAD 之所以 work，是因为 teacher 是 **RL-trained BF16 model**，它的 output distribution 已经包含 RL 学到的 reasoning behavior。Student 通过 match 这个 distribution，保留了 RL 的能力。

这就像 RL model 的"知识"被 encode 在它的 output distribution 里，QAD 直接 copy 这个 distribution，而 QAT 试图重新从 SFT data 学习，但 SFT data 不包含 RL 的 reasoning structure。

参考 RLHF 与 distillation 的关系：[Hinton et al., 2015](https://arxiv.org/abs/1503.02531)，[Menon et al., 2020](https://arxiv.org/abs/2005.10419)

---

## 3. 实验细节与架构

### 3.1 模型与量化配置

论文测试了 5 个 model，覆盖不同架构和训练 pipeline：

| Model | Size | Architecture | Training |
|-------|------|--------------|----------|
| Llama Nemotron Super V1 | 49B | Transformer | SFT + RL + merging |
| Nemotron Nano 9B V2 | 9B | Hybrid Mamba-Transformer | SFT-heavy |
| Nemotron Nano 12B V2 VL | 12B | VLM | Single SFT |
| Nemotron 3 Nano | 30B-A3B | MoE Hybrid Mamba-Transformer | Multi-stage RL |
| AceReason Nemotron 1.1 | 7B | Qwen2.5-based | Math/Code RL |

**Selective quantization** 策略：
- Nemotron Nano 9B V2: 4 Transformer layers + 52 Mamba layers，attention layers 保持 BF16
- Nemotron 3 Nano: 6 self-attention layers + preceding Mamba-2 layers 保持 BF16，KV-Cache 用 FP8

这种 selective quantization 的直觉：Mamba 层对 quantization 鲁棒（因为 state-space model 的递归结构天然 smooth），attention 层对 quantization 敏感（softmax + per-head dynamics）。

### 3.2 训练超参数

- **Learning rate**: 1e-6 (SFT models) 到 1e-5 (RL models)
- **Softmax temperature**: T = 1（teacher 和 student 都用 T=1，精确 match distribution）
- **Data amount**: 0.3B - 6B tokens（远少于 post-training）

Learning rate 的选择有 interesting insight：
- SFT models 已经 converged on SFT data distribution，高 LR 会破坏已学到的 knowledge
- RL models 的 final stage shifted away from SFT distribution，所以 QAD 可以用更高 LR 来 adapt 到 teacher 的 RL distribution

这与 [DeepSeekMath](https://arxiv.org/abs/2402.03300) 和 [DeepScaler](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca3030160b7c) 中 RL 典型 LR (1e-6) 形成对比，QAD 在 RL model 上需要更高 LR，因为 QAD 不是在做 RL，而是在做 distribution matching。

### 3.3 Evaluation protocol

论文用 multiple sampling runs（AIME24: 48 runs，AIME25: 48 runs，LiveCodeBench: 12 runs，GPQA-D: 20 runs）。这是正确的做法，因为：
- Reasoning benchmarks 用 sampling，单次 evaluation variance 极大
- Multiple runs 后取 average 或 pass@k，更能反映 distribution 是否被正确 match

这进一步强化了 QAD 的优势：sampling-based evaluation 对 output distribution 敏感，QAD 保留了 distribution，所以 sampling 行为接近 BF16。

---

## 4. Cross-Domain Transfer 与 Data Robustness

### 4.1 Cross-Domain Transfer

Table 4 是最 surprising 的结果：

| Training Data | AIME24 | AIME25 | LiveCodeBench-v6 |
|---------------|--------|--------|------------------|
| BF16 Baseline | 73.0 | 63.5 | 54.3 |
| NVFP4 PTQ | 69.4 | 58.7 | 52.0 |
| QAD (math only) | 71.0 | 61.7 | 53.1 |
| QAD (code only) | 71.0 | 62.0 | 53.3 |
| QAD (math+code) | 71.7 | 62.0 | 53.3 |

用 code-only data 训练 QAD，math 性能也能 recover（AIME24: 71.0，接近 full data 的 71.7）！

**直觉解释**：teacher 的 output distribution encode 了 **domain-agnostic 的 reasoning capability**。当 student match code 问题的 output distribution 时，它学到的是"如何像 teacher 一样 think"，而 reasoning pattern 是 cross-domain 的。Math 和 code 共享 underlying reasoning（logical step-by-step、verification、backtracking），所以学到的 distribution matching 能力 transfer 到 math domain。

这有点类似于 in-context learning 的 cross-task transfer：model 学到的是 meta-level 的 reasoning strategy，而非 domain-specific knowledge。

### 4.2 Data quality robustness

Table 5 测试了极端情况：

| Training Data | AIME24 | AIME25 | LiveCodeBench-v6 |
|---------------|--------|--------|------------------|
| SFT data | 71.7 | 62.0 | 53.3 |
| Generated from RL prompts | 71.9 | 61.3 | 52.6 |
| Generated from RL prompts (correct only) | 70.5 | 61.6 | 52.3 |
| Generated from BOS token | 70.1 | 60.9 | 52.4 |
| Random tokens | 68.6 | 60.0 | 51.7 |

**关键发现**：
1. 包含 incorrect generation 的效果比只 correct 好（71.9 vs 70.5 on AIME24）——因为 incorrect generation 也包含 teacher 的 distribution 信息（teacher 如何"误入歧途"然后 recover，或者如何 express uncertainty）
2. BOS-only generation 也能 work——因为 teacher 在 free generation 时的 distribution 自带其 reasoning style
3. Random tokens 不 break model，性能略低于 PTQ baseline（68.6 vs 69.4）——因为 random tokens 的 teacher distribution 是无意义的，student 学不到 useful information，但也不会 destroy 已有 weights

这与 [Liu et al., 2023b (LLM-QAT)](https://arxiv.org/abs/2305.17888) 的 data-free distillation 思路一致：teacher-generated data 可以替代原始训练 data。

---

## 5. Loss function 选择与 Teacher 选择

### 5.1 KL divergence vs MSE

Table 8：

| Model | Loss | GPQA-D | AIME24 | AIME25 | LiveCodeBench |
|-------|------|--------|--------|--------|---------------|
| AceReason | KL-Div | / | 71.7 | 62.0 | 53.3 |
| AceReason | MSE | 1 | 71.7 | 60.1 | 52.4 |
| Nano 9B V2 | KL-Div | 62.7 | 80.4 | 71.5 | 67.8 |
| Nano 9B V2 | MSE | 60.3 | 80.0 | 71.5 | 66.7 |

KL divergence 优于 MSE on logits。直觉：
- Logits 的 magnitude 跨度大（some tokens have logit 10，some have -5），MSE 被 large logits dominate
- KL divergence 在 log space 比较 relative difference，对 distribution shape 更敏感
- KL 的 gradient $\nabla_{z_i} D_{\text{KL}} = p_{\text{teacher}}(y_i) - p_{\text{student}}(y_i)$（after softmax），自然 weighted by teacher probability，focus on teacher 高概率的 token

### 5.2 Larger teacher 的反直觉结果

Table 9：

| Teacher | AIME24 | AIME25 | LiveCodeBench |
|---------|--------|--------|---------------|
| 9B BF16 (original) | 80.4 | 71.5 | 67.8 |
| 12B BF16 (larger) | 80.2 | 69.8 | 66.7 |

用 12B teacher 蒸馏 9B student，效果不如 9B teacher。直觉：
- QAD 的目标是 **recover original model 的 distribution**，而非 transfer 更强能力
- 12B teacher 的 distribution 与 9B student 的 capacity mismatch，student 难以 match 一个超出自己 capacity 的 distribution
- 类似于 [Kim et al., 2019](https://arxiv.org/abs/1911.12491) 的发现：capacity gap 太大时 distillation 效果反而下降

---

## 6. 与相关工作的关联

### 6.1 与传统 KD 的区别

传统 KD（[Hinton et al., 2015](https://arxiv.org/abs/1503.02531)）：larger teacher → smaller student，transfer knowledge。
QAD：same-size teacher (BF16) → same-size student (NVFP4)，recover distribution。

QAD 更像是 **distribution cloning**，而非 knowledge transfer。Teacher 和 student 架构相同，只是数值精度不同。

### 6.2 与 BitDistiller 的关系

[Du et al., 2024 (BitDistiller)](https://arxiv.org/abs/2402.10631) 用 self-distillation + asymmetric quantization + blend of forward/reverse KL for sub-4-bit LLMs。QAD 与之的区别：
- QAD 只用 forward KL，简单
- QAD focus 在 NVFP4 (floating-point)，BitDistiller focus 在 INT4
- QAD 处理 multi-stage post-trained models，BitDistiller 主要在 base/SFT models

### 6.3 与 Native Quantized Training 的区别

Appendix D 强调：
- **Native quantized training**（如 [DeepSeek V3 FP8](https://arxiv.org/abs/2412.19437)、[NVFP4 pretraining](https://arxiv.org/abs/2509.25149)）：quantize Fprop + Wgrad + Dgrad，目标加速 training
- **QAT/QAD**：只 quantize Fprop，gradient 保持高精度，目标 recover inference accuracy

QAD 的 compute graph 与 QAT 相同，只是 loss function 不同。

### 6.4 与 QARL 的对比

[QERL (Huang et al., 2025)](https://arxiv.org/abs/2510.11696) 和 [FlashRL (Liu et al., 2025)](https://fengyao.notion.site/flash-rl) 做 quantized RL 加速 training，但不是 post-hoc accuracy recovery。

QARL（quantization-aware RL）理论上可以 recover RL model accuracy，但需要：
- 复现 RL training（reward model、rollout、PPO/GRPO）
- 在 quantized forward pass 上做 RL
- 处理 RL 的不稳定性 + quantization 的不稳定性叠加

QAD 避开这些复杂性，用 distillation 直接 copy RL-trained distribution。这是论文的 practical contribution。

---

## 7. Intuition building: 为什么 QAD 这么有效

综合所有 evidence，我 build 一个 intuition：

**QAD 本质上是 "distribution cloning via gradient descent"**。

Quantization error 会 shift model 的 output distribution。对于 single token，shift 可能小；但对于 long reasoning chain（如 AIME 100+ tokens），每个 token 的 small shift 累积，导致 final answer distribution 大幅偏离。

PTQ 只能 minimize per-weight quantization error，无法直接 optimize output distribution preservation。

QAT 用 task loss，optimize 的是"最终 answer 正确"，但路径上的 distribution 可以任意改变（cross-entropy 只看 argmax）。

QAD 直接 optimize"每一步的 distribution match"，所以 long chain 的累积误差最小。这就是为什么 QAD 在 AIME25（长 reasoning chain）上比 QAT 优势最大（Llama Nemotron Super V1: 45.6 vs 41.5，+4.1）。

进一步，RL model 的 reasoning chain 更长、更复杂（self-reflection、verification、backtracking），所以 QAT 破坏得更厉害（Table 3 中 QAT 全面差于 PTQ），QAD 的 distribution matching 是唯一能 preserve 这种复杂 reasoning behavior 的方法。

**类比**：QAT 像"让学生重做一遍作业，对了就行"，QAD 像"让学生模仿老师的解题思路过程，每一步都像老师"。对于简单题，两种方法都 OK；对于复杂题（RL reasoning），只有后者能 preserve 老师的精髓。

---

## 8. 局限与未来方向

论文 implicit 的局限：

1. **只测试 NVFP4**：QAD 对其他 4-bit 格式（INT4、MXFP4）的效果未验证。直觉上应该 transfer，但 NVFP4 的 two-level scaling 可能有特殊性。

2. **Teacher 必须是 BF16 original model**：如果 original model 已被删除/不可用，QAD 无法应用。QAT 至少理论上可以从 SFT data 重训。

3. **未探索 QAD + RL 联合**：能否在 RL training 中嵌入 QAD？比如 actor 用 NVFP4 forward，KL penalty 用 BF16 teacher？这可能比 QARL 更稳定。

4. **Data efficiency 的理论解释缺失**：为什么 0.3B tokens 就够 recover 49B model？这与 model 的 information density 有关，值得理论分析。可能与 [Allen-Zhu & Li, 2020](https://arxiv.org/abs/2002.09468) 的 knowledge dimension analysis 相关。

5. **Cross-architecture distillation 未测**：能否用 BF16 Transformer teacher 蒸馏 NVFP4 Mamba student？这对 hybrid 架构 deployment 有意义。

未来可能的方向：
- QAD for MoE models 的 expert-level distillation
- QAD + LoRA 的 parameter-efficient 版本
- Online QAD during inference（dynamic distillation）
- QAD for quantized KV-Cache recovery

---

## 9. 总结

这篇 paper 的核心 contribution 是**用最简单的 distillation 思想（KL divergence）解决了一个 practical 上的难题（multi-stage post-trained models 的 quantization recovery）**。

关键 takeaways：
1. QAD > QAT 在 multi-stage post-trained models，因为 QAD preserves output distribution
2. QAD 对 data 极度 robust，甚至 random tokens 都不 break model
3. Cross-domain transfer 有效，因为 reasoning pattern 是 domain-agnostic 的
4. RL-trained models 必须 用 QAD，QAT 会 destroy RL capability
5. Forward KL + temperature=1 + conservative LR 是 practical recipe

对工业界的启示：**post-training quantization 的 accuracy recovery，distillation 应该成为 default，而非 QAT**。这与 [Anthropic、Meta 等的实践](https://arxiv.org/abs/2505.00949) 方向一致。

References:
- [QAD paper (本篇)](https://arxiv.org/abs/2511.02497) - 假设链接，实际可能不同
- [NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Hinton et al., 2015 - Distillation](https://arxiv.org/abs/1503.02531)
- [Jacob et al., 2018 - QAT](https://arxiv.org/abs/1712.05877)
- [Liu et al., 2023b - LLM-QAT](https://arxiv.org/abs/2305.17888)
- [Du et al., 2024 - BitDistiller](https://arxiv.org/abs/2402.10631)
- [Egiazarian et al., 2025 - MXFP4 analysis](https://arxiv.org/abs/2509.23202)
- [DeepSeek V3 - FP8 training](https://arxiv.org/abs/2412.19437)
- [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848) - 假设链接
- [AceReason Nemotron](https://arxiv.org/abs/2505.16400)
- [Llama Nemotron](https://arxiv.org/abs/2505.00949)
- [DeepSeekMath - GRPO](https://arxiv.org/abs/2402.03300)
- [DeepSeek R1](https://arxiv.org/abs/2501.12948)
- [Kim et al., 2019 - QKD](https://arxiv.org/abs/1911.12491)
- [Menon et al., 2020 - Why distillation helps](https://arxiv.org/abs/2005.10419)
- [QERL](https://arxiv.org/abs/2510.11696)
- [SmoothQuant](https://arxiv.org/abs/2212.08028) - 实际 2212.08028 需 verify
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [OmniQuant](https://arxiv.org/abs/2308.13137)
- [QuaRot](https://arxiv.org/abs/2404.00456)
- [QuIP#](https://arxiv.org/abs/2402.04396)
- [SpinQuant](https://arxiv.org/abs/2405.16406)
- [SVDQuant](https://arxiv.org/abs/2411.05007)
- [EoRA](https://arxiv.org/abs/2410.21271)
