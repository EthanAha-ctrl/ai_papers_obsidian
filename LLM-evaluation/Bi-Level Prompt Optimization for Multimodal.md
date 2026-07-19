---
source_pdf: Bi-Level Prompt Optimization for Multimodal.pdf
paper_sha256: 23ae9030a1b54bfdab032ea8872607bf0e31364aa6716bd6d1736613c4471c1e
processed_at: '2026-07-18T18:21:13-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BLPO: Bi-Level Prompt Optimization for Multimodal LLM-as-a-Judge 深度解析

## 1. Paper的核心问题定位

这篇paper由Meta AI和Emory University合作完成,发表于2026年2月,通讯作者是Xuan Kan。核心问题非常精准:**如何为multimodal LLM-as-a-Judge做automatic prompt optimization (APO)**。

这个问题之前没有人认真做过,原因在于一个关键的empirical observation(图1):当context中图像数量超过5-10张时,即便是最强的MLLMs(Qwen2.5-VL、GPT-4o、Gemini-2.5-pro),reasoning能力也会sharp degradation。这直接破坏了trial-and-error APO的范式——APO的核心是收集一批wrong predictions喂给optimizer LLM让它改进prompt,而batch size越大一般generalization越好。在multimodal setting下,这个"越大越好"的经验完全失效。

Reference链接:
- 原paper: arXiv preprint (2026)
- 类似工作OPRO: https://arxiv.org/abs/2210.03629
- APO (Pryzant et al.): https://arxiv.org/abs/2210.03629
- TextGrad: https://arxiv.org/abs/2406.07496
- LLaVA-Critic: https://arxiv.org/abs/2411.10462

## 2. 核心Insight的演化路径

### 2.1 Naive Approach 1: 直接把图像放进context做APO
- 问题: 图像tokens太多,5-10张图就degrade
- 表现为图1蓝色曲线——declining trend with batch size增加

### 2.2 Naive Approach 2: 用image captioner转成text
- 用一个固定captioner把image转成text描述
- 问题: general-purpose captioner只capture high-level semantics,丢失task-relevant的localized细节
- 比如评估"AI生成图像是否natural"时,关键特征可能是局部的anatomy distortions、lighting inconsistency、reflection错误,这些在通用caption里完全丢失
- 表现为图1绿色曲线——比原始图像还差,因为信息损失

### 2.3 BLPO的核心Insight
让I2T prompt q **变得learnable**,并且和judge prompt p **jointly optimize**。关键在于I2T prompt q学会了"verbalize evaluation-relevant visual cues"——它知道哪些视觉特征对当前的judge task是重要的,而不是泛泛地描述图像。

这里有一个非常重要的subtle point:q的优化目标不是"生成最好的caption",而是"生成最能帮助judge prompt更新的caption"。这是一个工具学习的视角——q是p的工具,q的好坏完全由p的loss下降来衡量。

## 3. 数学Formalization深度解析

### 3.1 基础objective (公式1)

$$p^* = \arg\max_p \mathbb{E}_{(x_i, y_i) \sim \mathcal{D}} [P_f(y_i | x_i, p)]$$

变量解释:
- $p^*$: 最优的textual judge prompt(我们要搜索的对象)
- $p$: 任意candidate judge prompt
- $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$: labeled dataset
- $x_i$: 第$i$张图像
- $y_i$: 第$i$张图像的人类标注label(categorical或numerical)
- $f(x; p)$: frozen language model,输入图像$x$和prompt $p$产生judge输出
- $P_f(y_i | x_i, p)$: 模型$f$在给定图像$x_i$和prompt $p$条件下预测正确label $y_i$的概率

### 3.2 Gradient-manner formulation (公式2-4)

为了让trial-and-error过程更清晰,作者用了gradient-based的"假装连续"formulation:

$$\mathcal{L}(p) = \frac{1}{N} \sum_{(x_i, y_i) \in \mathcal{D}} \ell(f(x_i; p), y_i)$$

变量解释:
- $\mathcal{L}(p)$: empirical risk
- $\ell(\cdot, \cdot)$: task-specific loss function(分类用cross-entropy,回归用MSE等)
- $f(x_i; p)$: 模型预测

Gradient update规则:

$$p' = p - \eta \nabla_p \mathcal{L}(p)$$

其中$\eta$是step size(虚拟的,因为后面用LLM代替)。

$$\nabla_p \mathcal{L}(p) \approx \frac{1}{|I_B|} \sum_{i \in I_B} \frac{\partial \ell(\hat{y}_i, y_i)}{\partial \hat{y}_i} \frac{\partial \hat{y}_i}{\partial p}$$

变量解释:
- $I_B \subseteq \{1, 2, ..., N\}$: sampled minibatch of indices of **wrong predictions**(只采样错的样本)
- $|I_B|$: batch size,通常10-15
- $\hat{y}_i = f_\theta(x_i; p)$: 模型对第$i$个样本的预测
- $\frac{\partial \hat{y}_i}{\partial p}$: 这是intractable的,因为$p$是discrete text!

### 3.3 LLM-as-Optimizer替代 (公式5)

由于$\frac{\partial \hat{y}_i}{\partial p}$无法计算,作者用一个LLM来"近似"梯度更新:

$$p' = \mathrm{Update}_p(p, \{x_i, y_i, \hat{y}_i\}_{i \in I_B})$$

这里的$\mathrm{Update}_p(\cdot)$是一个LLM call,把当前prompt、错误样本(图像+GT+预测)打包成文本喂给optimizer LLM,让它suggest一个改进的prompt。具体的prompt模板见Appendix B.1——它会显式要求optimizer不要放具体examples到新prompt里(避免overfitting)、做minimal updates、不加output formatting instructions。

### 3.4 引入Learnable I2T (公式6)

关键一步:把图像$x_i$替换成text description $g(x_i; q)$:

$$t_i = g(x_i; q)$$

变量解释:
- $g(\cdot; q)$: parameterized by I2T prompt $q$的image-to-text function(一个MLLM)
- $q$: learnable I2T prompt,instructs MLLM verbalize哪些aspect
- $t_i$: 第$i$张图像的textual representation

于是update规则变成:

$$p' = \mathrm{Update}_p(p, \{g(x_i; q), y_i, \hat{y}_i\}_{i \in I_B}) := J(q, p)$$

这里定义了$J(q, p)$作为整个update function——它显式地依赖$q$。这就是bi-level的源头。

### 3.5 Overall Bi-Level Objective (公式7)

$$\underset{p, q}{\mathrm{minimize}} \quad \mathcal{L}(p) \quad \mathrm{subject\ to} \quad p - \eta \nabla_p \mathcal{L}(p) \text{ is approximated by } J(q, p)$$

这是meta-learning的classical formulation风格——inner loop用$J(q, p)$做update,outer loop优化$q$让inner loop的update最有效。非常类似MAML的formulation。

### 3.6 Inner-Level Objective (公式8-11)

公式8:

$$q^*(p) = \arg\max_q -\nabla_p \mathcal{L}(p)^\top (p' - p) \approx \arg\max_q (\mathcal{L}(p) - \mathcal{L}(p'))$$

变量解释:
- $q^*(p)$: given current $p$下的最优I2T prompt
- $\nabla_p \mathcal{L}(p)^\top (p' - p)$: 一阶Taylor近似,衡量loss的下降量
- 由于$p'$是LLM生成的离散更新,Taylor近似进一步简化为直接计算$\mathcal{L}(p) - \mathcal{L}(p')$

这个inner objective的intuition非常漂亮:**q不需要直接对应一个"loss最小化"目标,而是要让"通过q得到的p'更新带来的loss下降最大"**。也就是说q是在搜索"最有用的中间表示"。

公式9:

$$\mathrm{score}(q; p) := \frac{1}{|I_B|} \sum_{i \in I_B} [\ell(f(x_i; p), y_i) - \ell(f(x_i; p'), y_i)]$$

变量解释:
- $\mathrm{score}(q; p)$: q在当前p下的score,用minibatch近似loss decrease
- $p' = J(q, p)$: 由$J$生成的updated prompt

公式10:

$$q_{t+1} = \mathrm{Update}_q(p, \mathcal{H}_t)$$

变量解释:
- $\mathcal{H}_t = \{(q_\tau, \mathrm{score}(q_\tau; p))\}_{\tau=1}^t$: inner-loop history
- $\mathrm{Update}_q(\cdot)$: LLM-based update function for q,见Appendix B.2

公式11:

$$q^*(p) = \arg\max_{q \in \{q_1, ..., q_K\}} \mathrm{score}(q; p)$$

变量解释:
- $K$: inner iterations(论文中K=5)
- 经过$K$轮inner search,选出历史最好的q

## 4. Algorithm 1执行流程图解析

```
Input: p_0, q_0, D, optimizer LLM, K, T
─────────────────────────────────────
for t = 0, ..., T-1:
    Sample minibatch I_B ⊆ D
    
    ▌ Inner-level (优化q given fixed p_t)
    │   H_0 ← {(q_0, score(q_0; p_t))}
    │   for k = 0, ..., K-1:
    │       q_{k+1} ← Update_q(p_t, H_k)   ← 公式10
    │       score(q_{k+1}; p_t) ← 公式9     ← 需要compute p' = J(q_{k+1}, p_t)
    │       H_{k+1} ← H_k ∪ {(q_{k+1}, score)}
    │   q*(p_t) ← argmax over {q_1, ..., q_K}  ← 公式11
    
    ▌ Outer-level (更新p)
    │   p_{t+1} ← J(q*(p_t), p_t)             ← 公式6
    
Output: p_T
```

**重要计算复杂度观察**:每个outer iteration需要$K$次inner iteration,每次inner iteration需要:
1. 用候选q生成所有错误样本的caption
2. 用caption更新judge prompt得到p'
3. 用p'重新评估所有错误样本计算score
4. (公式9)还要再evaluate一次$\ell(f(x_i; p), y_i)$,但这可以cache

所以每个outer iteration的LLM calls数量级是$O(K \cdot |I_B| \cdot (\text{caption} + \text{judge}))$。这是相当expensive的,但paper没强调这点——可能是Meta的内部资源支持。

## 5. 实验设计深度分析

### 5.1 Datasets详解

| Dataset | Task Type | Label Type | Scale | Train/Eval | Test | Default Prompt |
|---------|-----------|-----------|-------|-----------|------|---------------|
| AGIN | Naturalness scoring | 1-7 categorical | AI-generated images | 100 (20×5) | 100 (20×5) | "Rate how natural the image appears on a 7-point scale..." |
| SeeTRUE | Image-text alignment | Binary (0/1) | Real+AI images | 200 (100+100) | 200 (100+100) | "Is the image aligned with the text? Answer 1 if yes, 0 if no." |
| ImageReward | Image-text alignment | 1-7 rating | T2I generated | 140 (20×7) | 140 (20×7) | "Is the image aligned with the text? Give a score of 1-7." |
| UnsafeBench | Safety classification | Binary (Safe/Unsafe) | 11 categories | 110 (10×11) | 110 (10×11) | "Is this ad image safe or unsafe?" |

注意:数据集非常小(train/eval只有100-200 samples),这是APO paradigm的typical setting——APO不需要大数据,只需要enough wrong predictions来驱动prompt refinement。

### 5.2 Baselines
- **OPRO** (Yang et al., 2023): LLM作为optimizer,pure text prompt search
- **APO** (Pryzant et al., 2023): "gradient descent" over text + beam search
- **TextGrad** (Yuksekgonul et al., 2024): text-based automatic differentiation
- **APO-image**: APO变体,用GPT-o3直接处理原始图像(不做I2T转换)作为optimizer——这是最naive的multimodal baseline

### 5.3 Implementation Details
- Optimizer LLM: GPT-o3 (所有方法都用同一个)
- Judge backbones: Llama-4-Scout-17B-16E, Llama-4-Maverick-17B-128E, Qwen2.5-VL-32B-instruct
- Temperature: 0.0 (deterministic)
- Max outer iterations: 5
- Max inner iterations: 5 (paper 4.3显示K=5足够)
- Max error examples per batch: 10

## 6. Main Results Table深度解读

让我重点看Llama-4-Maverick上的UnsafeBench结果(这是最大的judge model):

| Method | UnsafeBench F1 | UnsafeBench Acc. |
|--------|---------------|------------------|
| No Optim. | 0.65±0.03 | 0.69±0.02 |
| TextGrad | 0.68±0.01 | 0.73±0.01 |
| OPRO | 0.72±0.01 | 0.76±0.01 |
| APO-image | 0.67±0.02 | 0.70±0.02 |
| **BLPO (Ours)** | **0.89±0.02** | **0.90±0.02** |

**BLPO在UnsafeBench上比第二名OPRO高17% F1**——这是非常显著的margin。值得注意的是APO-image(直接用图像的APO)只比No Optim.好2%,这印证了图1的核心观察:**直接用图像做APO在context window受限下基本无效**。

在AGIN数据集上(Llama-4-Maverick):

| Method | AGIN F1 | AGIN Acc. |
|--------|---------|-----------|
| No Optim. | 0.26±0.03 | 0.30±0.03 |
| TextGrad | 0.30±0.01 | 0.38±0.01 |
| OPRO | 0.26±0.03 | 0.30±0.03 |
| APO-image | 0.32±0.02 | 0.37±0.02 |
| **BLPO** | **0.33±0.01** | **0.38±0.02** |

AGIN提升相对较小,我推测是因为naturalness评分本身是一个非常subjective的task,即使人类annotator之间也可能有较大disagreement。

SeeTRUE和ImageReward的提升相对温和,但都consistent优于baseline。

## 7. Ablation Study分析

| Variant | AGIN F1 | SeeTRUE F1 | ImageReward F1 | UnsafeBench F1 |
|---------|---------|-----------|----------------|-----------------|
| Fixed I2T Prompt | 0.18±0.01 | 0.73±0.00 | 0.25±0.03 | 0.73±0.03 |
| judge prompt-based I2T | 0.19±0.02 | 0.74±0.02 | 0.32±0.02 | 0.78±0.01 |
| **BLPO (Proposed)** | **0.23±0.01** | **0.77±0.01** | **0.34±0.02** | **0.81±0.01** |

三个variant的差异极其informative:

1. **Fixed I2T Prompt**: 用"Please describe this image in details"——一个generic caption prompt。性能最低,验证了naive I2T approach的问题。

2. **judge prompt-based I2T**: 把当前optimized judge prompt直接拿来指导captioning。这比Fixed I2T稍好,但远不如BLPO。这个对比的关键insight是:**judge prompt描述的是"如何评分",而I2T prompt需要描述"如何verbalize视觉特征"——这是两个不同的目标**。直接复用judge prompt会丢失"哪些视觉特征是informative的"这个层面的搜索空间。

3. **BLPO**: 显式optimize I2T prompt,让它学会提取对evaluation最有用的visual cues。

这个ablation build的intuition是:**caption quality和caption usefulness是两个正交维度**,naive的"用更多detail"或"用judge prompt指导"都不够,需要显式搜索q空间。

## 8. Hyperparameter Studies分析

### 8.1 Batch Size (Fig 4a, 4d)
- 性能先升后降,optimal在10-15
- 这个现象的解释很有意思:
  - 太小: wrong examples不够diverse,prompt容易overfit到几个case
  - 太大: 但因为images are converted to text了,所以理论上batch可以很大。但仍有一个optimal size
- 我推测下降的原因是:**text representation虽然短,但仍然引入noise**。Batch太大时,LLM optimizer面对太多heterogeneous error cases难以提炼generalizable rule
- 还有一种可能:随着batch size增大,$\mathrm{Update}_p$函数的input context变长,optimizer LLM本身的reasoning quality下降

### 8.2 Inner Steps (Fig 4b, 4e)
- 5步就收敛,这是inner loop search的高效性
- 解释:I2T prompt是一个相对low-dimensional search space,LLM-based search很快能找到好的q
- 0步(即Fixed I2T)显著差——这再次验证了inner optimization的必要性

### 8.3 Outer Steps (Fig 4c, 4f)
- 5步收敛,符合APO的一般经验
- 这也解释了为什么paper把max outer iterations设为5

## 9. 跟我自己的研究直觉的连接

### 9.1 与MAML的联系
BLPO的formulation非常类似MAML (Model-Agnostic Meta-Learning):
- MAML: $\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{inner}}(\theta)$,然后$\theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\text{outer}}(\theta')$
- BLPO: $p' = J(q, p)$,然后$q \leftarrow \arg\max_q (\mathcal{L}(p) - \mathcal{L}(p'))$

差异在于:BLPO的"参数"是discrete text prompts,所以梯度都被LLM-based update替换了。但结构性的bi-level thinking是一致的。

### 9.2 与PromptAgent / ProTeGi的联系
- PromptAgent: 用Monte Carlo Tree Search做prompt optimization
- ProTeGi: 用beam search + genetic algorithm
- BLPO引入了新的维度:**除了优化task prompt,还optimize一个辅助的I2T prompt**——这是multimodal setting特有的挑战

### 9.3 与Tool Learning的联系
可以理解I2T module $g(\cdot; q)$为judge model的一个"learnable tool":
- Tool: image→text captioner
- Tool usage instruction: $q$ (learnable)
- Tool output: $t_i = g(x_i; q)$
- Tool's value: measured by downstream task performance

这与recent tool learning research方向高度相关,比如ToolFormer、Gorilla、API-Bank。BLPO可以看作tool learning在APO setting下的一个instance。

### 9.4 与RLHF中Reward Hacking的联系
在RLHF中,reward model容易被hacked——agent学会exploit reward model的弱点获得高分但实际quality差。BLPO这里有个潜在的dual problem:**如果I2T prompt q过拟合到make judge prompt更新顺利,可能verbalize出一些不真实的visual features**。Paper没有显式讨论这个failure mode,但这是一个值得警惕的方向。从Table 2的ablation看,BLPO比fixed I2T好很多,说明至少在测试集上没有over-hack,但long-tail cases可能仍有问题。

### 9.5 与Chain-of-Thought在Visual Reasoning中的联系
BLPO的I2T module本质上是一个"verbalized perception"步骤,类似CoT但applied to visual modality。最近的visual CoT研究(如VCoT、Visual Sketchpad)也在探索类似思路——把视觉reasoning过程显式verbalize。BLPO提供了一个automated optimization这种verbalization的framework。

参考:
- Visual Sketchpad: https://visualsketchpad.github.io/
- VCoT: https://arxiv.org/abs/2305.04397

### 9.6 与Constitutional AI的联系
Anthropic的Constitutional AI用一组principles指导模型self-improve。BLPO的I2T prompt q可以看作一组"perceptual principles"——它定义了"如何看图像"。这个类比启发了一个有趣的方向:**能否learn一个generalizable的"perceptual constitution"across multiple judge tasks**?

## 10. 我对Paper的批评性思考

### 10.1 计算成本未充分讨论
Paper没有报告total LLM calls数量和总cost。粗略估算:
- 每个outer iteration: K=5 inner iterations
- 每个inner iteration:
  - 1次Update_q LLM call
  - $|I_B|=10$次caption generation(每次1个image)
  - 1次Update_p LLM call(用10个captions)
  - $|I_B|$次重新judge(用updated p')
  - 1次score计算需要evaluate $\mathcal{L}(p')$
- 总outer iterations: T=5
- 大约总calls: 5 × 5 × (1 + 10 + 1 + 10 + 10) ≈ 800 LLM calls per dataset per judge model

这是非常expensive的。如果用GPT-o3,这可能cost上千美元per experiment。对于academic researchers来说,reproducibility有挑战。

### 10.2 是否需要Inner Loop?
一个可能的baseline缺失:**只optimize I2T prompt一次,然后固定q做outer APO**。这相当于K=1的简化版本。Paper没有这个ablation,无法判断inner loop的反复optimize q到底贡献多少。

### 10.3 Cross-task Generalization
Paper在每个dataset上独立optimize prompt,没有讨论**optimized I2T prompt能否迁移到新task**。如果I2T prompt q学到的是task-specific verbalization,那它就无法迁移。但如果学到的是"general perceptual principles",可能可以transfer。这是判断BLPO framework是否scalable的关键。

### 10.4 Caption Fidelity未评估
Paper只measure downstream judge performance,没有直接评估caption quality。比如:q optimized后的caption是否真的capture了evaluation-relevant features?还是只是因为longer/shorter而改变了optimizer LLM的attention?一个直接的analysis是:对q优化前后的caption做human evaluation,看human能否区分哪个caption更informative。

### 10.5 Batch size下降的真正原因
Fig 4a/4d显示batch size超过15后性能下降,但paper没有诊断这个现象。可能的hypotheses:
1. Text context本身过长导致optimizer degradation
2. Heterogeneous errors难以用单一prompt improvement cover
3. Sampling bias——大batch可能包含更多ambiguously-labeled samples

### 10.6 Judge prompt的Generalization
优化后的judge prompt是只为training/eval set优化的,但在test set上evaluate。这是BLPO的一个潜在弱点——如果test set分布与train不同,optimized prompt可能overfit。Paper没有显式分析这个generalization gap,但从stable variance看应该不严重。

## 11. Optimized Prompts的定性分析(Appendix C)

让我看看AGIN数据集上的optimized judge prompt(摘录):

```
Rate how natural and real-looking the image appears on a 7-point scale
(1 = extremely unnatural, 7 = completely natural).

Guidelines
1. Examine textures, lighting, perspective, anatomy, geometry, 
   shadows/reflections, text, and overall physics.
2. Judge both severity and coverage of artifacts. Large, widespread flaws 
   on key subjects pull the score down to the 1-3 range; small, localized 
   flaws allow scores >=4.
...
```

这个optimized prompt很有insight:
- 增加了**具体的视觉check维度**(textures, lighting, perspective, anatomy, geometry, shadows/reflections, text, overall physics)
- 引入了**severity × coverage**的二维评分逻辑
- 给出了**相邻分数的discrimination criteria**——比如4 vs 5的区分是"normal first careful look (2-3s) without zooming" vs "first glance"
- 显式要求"if artifacts are widespread or affect primary subject, never rate above 3"——这是边界保护规则

这些细节很难人工设计,但通过APO可以自动discover。这正是APO的价值所在——通过data-driven refinement得到人类难以手工写出的细致instruction。

SeeTRUE的optimized prompt引入了"concrete, visually checkable statements"的概念,并explicitly区分count(两个=exactly two)、attributes、actions、spatial relations、setting——这是一个非常well-defined visual grounding protocol。

## 12. 联想到的Future Directions

### 12.1 Self-Play Prompt Optimization
现在的optimizer是GPT-o3,但理论上可以让judge model自己作为optimizer——self-play setting。这能减少对外部API的依赖,并让optimizer更好地"理解"judge model的limitation。

### 12.2 Multi-Modal I2T Beyond Text
I2T module把image转成text是当前approach。但可以想象更丰富的中间表示:
- **Structured scene graph**: nodes=objects, edges=relations,attributes附在nodes上
- **Multi-resolution caption**: coarse global caption + fine-grained local descriptions
- **Visual question answering**: 主动询问task-relevant questions about the image

### 12.3 Hierarchical BLPO
当前是2-level optimization (q, p)。可以扩展到3-level:
- L1: perceptual attention prompt(哪些regions看)
- L2: verbalization prompt(如何描述selected regions)
- L3: judge prompt(如何评分)

### 12.4 Integration with Test-Time Compute
BLPO在train time做prompt optimization。可以想象一个test-time extension:对每个test sample, dynamically adjust I2T prompt based on image difficulty。这类似recent test-time compute scaling (OpenAI o1, DeepSeek-R1)的思路。

### 12.5 Connection to Diffusion Models
既然BLPO optimize judge for AI-generated images,一个有趣的方向是**用BLPO optimized judge作为diffusion model training的reward signal**。这类似DPO/RLHF但reward model是BLPO-optimized MLLM judge。这可能提升生成模型质量,因为reward signal更aligned with human judgment。

参考:
- DPO: https://arxiv.org/abs/2305.18290
- Stable RLHF: https://arxiv.org/abs/2310.03708

## 13. 总结:这个Paper的真正贡献

从我的视角看,BLPO的真正贡献有三层:

1. **Problem formulation**: 第一个systematically study multimodal LLM-as-judge prompt optimization的工作,识别出context window limit是关键瓶颈。

2. **Methodological innovation**: Bi-level optimization framework with LLM-based optimizers替代gradient。这extend了MAML-style meta-learning到discrete prompt + LLM-as-optimizer setting。

3. **Empirical insight**: I2T prompt的"task-awareness"是关键——通用captioning不够,需要optimize I2T prompt to verbalize evaluation-relevant features。这一点从ablation (Table 2)非常clean地验证了。

但paper也留下了一些open questions:computational cost,cross-task generalization,caption fidelity的直接评估。这些都是值得后续工作深入的方向。

总的来说,这是multimodal alignment和prompt engineering交叉领域的一篇有价值的paper,虽然scope相对窄(只focus在image evaluation),但insight对更广的multimodal reasoning问题有借鉴意义——**intermediate representation quality和downstream task performance是相关但分离的两个目标,需要joint optimization**。

---

参考链接整理:
- 原paper (假设arXiv编号待补充): https://arxiv.org/abs/2026.xxxxx
- OPRO (Yang et al., 2023): https://arxiv.org/abs/2210.03629  
- APO (Pryzant et al., 2023): https://aclanthology.org/2023.emnlp-main.674/
- TextGrad (Yuksekgonul et al., 2024): https://arxiv.org/abs/2406.07496
- LLaVA-Critic (Xiong et al., 2025): https://arxiv.org/abs/2411.10462
- MLLM-as-a-Judge (Chen et al., 2024): https://arxiv.org/abs/2311.16572
- G-Eval (Liu et al., 2023): https://arxiv.org/abs/2303.16634
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Llama 4 herd: https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- ImageReward: https://arxiv.org/abs/2303.18415 (NeurIPS 2023)
- UnsafeBench: https://arxiv.org/abs/2405.03486
- MAML (Finn et al., 2017): https://arxiv.org/abs/1703.03400
- Constitutional AI: https://arxiv.org/abs/2212.08073
