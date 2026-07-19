---
source_pdf: ACCELERATING TRANSFORMER INFERENCE AND TRAINING WITH 2-to-4 ACTIVATION
  SPARSITY.pdf
paper_sha256: 5b9a577448498c7ce78a05c7f5005db6e7a4845687e3fcc2fea60583c9819d03
processed_at: '2026-07-17T23:27:06-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ACCELERATING TRANSFORMER INFERENCE AND TRAINING WITH 2-to-4 ACTIVATION SPARSITY 深度解析

## 1. Paper总体定位与核心insight

这篇来自Meta的工作,核心贡献可以总结为一句话: **把Squared-ReLU激活函数天然产生的高sparsity, 与NVIDIA GPU硬件原生支持的2:4 structured sparsity格式结合起来, 在LLM的training和inference两个阶段都获得real speedup, 且accuracy无损**。

这里有一个关键的"巧合"值得深入思考 -Squared-ReLU本身只是被So et al. (2021)在Primer工作中提出作为SwiGLU的替代品, accuracy相当, 但作者意外发现它产生84-98%的intrinsic activation sparsity。这个sparsity emergence是一个尚未被很好解释的empirical phenomenon, 也是这篇paper最有意思的open question。

paper链接: https://arxiv.org/abs/2411.02935 (推测, 根据内容应该是2024年底)
PyTorch 2:4 sparsity blog: https://pytorch.org/blog/accelerating-neural-network-training/
Primer (Squared-ReLU source): https://arxiv.org/abs/2109.08668
SwiGLU原paper: https://arxiv.org/abs/2002.05202

## 2. Squared-ReLU vs SwiGLU的数学对比

让我详细解析这两个activation function的数学形式, 这对理解为什么会出现sparsity emergence至关重要。

### SwiGLU
$$\text{SwiGLU}(X) = (\text{Swish}_\beta(XW_1) \odot XW_3) W_2$$

其中:
- $X \in \mathbb{R}^{B \times d}$: input tensor, $B$是batch*seqlen, $d$是model dimension
- $W_1, W_3 \in \mathbb{R}^{d \times d_{ff}}$: 两个独立的projection matrices (这就是GLU的"gated"结构)
- $W_2 \in \mathbb{R}^{d_{ff} \times d}$: output projection
- $\text{Swish}_\beta(x) = \frac{x}{1 + e^{-\beta x}}$: smooth non-monotonic activation
- $\odot$: element-wise (Hadamard) product
- $\beta$: 通常设为1

关键观察: $\text{Swish}_\beta(x) = 0$ 当且仅当 $x = 0$, 这意味着SwiGLU的output只在 $XW_1 = 0$ 处为0, 实际中几乎不会happen, 所以SwiGLU的activations基本是dense的。

### Squared-ReLU
$$\text{Squared-ReLU}(X) = (\max(0, XW_1)^2) W_2$$

其中:
- 只有一个projection $W_1$, 然后element-wise $\max(0, \cdot)^2$
- $\max(0, x)^2 = 0$ 当且仅当 $x \leq 0$
- 对于normally distributed input (zero-centered), 理论上50%的activations为0 (ReLU的sparsity)

### 参数量匹配
为了让两个模型parameter count一致 (Table 2显示), Squared-ReLU需要更大的FFN hidden dimension:
- SwiGLU: 参数量 $\approx 3 \cdot d \cdot d_{ff}^{SwiGLU}$
- Squared-ReLU: 参数量 $\approx 2 \cdot d \cdot d_{ff}^{SquaredReLU}$
- 所以 $d_{ff}^{SquaredReLU} = 1.5 \cdot d_{ff}^{SwiGLU}$

这点很关键 - paper里把FFN hidden dim从8192 (SwiGLU对应) 调到匹配后, 得到Table 1里Dense SwiGLU 2.654 vs Dense Squared-ReLU 2.651, 几乎完全一致。

## 3. 为什么Squared-ReLU会emerge 84-98% sparsity? (我的intuition)

这是paper里没有解释的open question, 但我觉得非常值得思考。让我提出几个hypothesis:

### Hypothesis 1: Squared activation的"赢者通吃"dynamics
$x^2$ 是严格convex的, 意味着大的值被进一步放大, 小的值被进一步抑制。Training过程中, 一旦某个neuron的pre-activation均值偏negative, gradient会push它更加negative (因为正向的output被zeroed, loss没有信号让它变正), 形成正反馈:
- pre-activation $y_i < 0 \Rightarrow$ output 0 $\Rightarrow$ gradient对$y_i$没有直接信号 (ReLU的dead neuron现象)
- $x^2$ 让positive output被amplify, 使得loss对positive path的gradient更强, 进一步强化"几个大neuron负责, 其他neuron沉默"的pattern

### Hypothesis 2: Training dynamics与weight decay的交互
AdamW + weight decay会push weights toward 0。如果一个feature的pre-activation均值稍小于0, weight decay会让bias向0走, 但是ReLU^2的zero region (x<0) 把这个feature完全zero-out, gradient完全无法flow back去"挽救"它。这就形成了一个basin - 一旦feature进入"mostly negative"状态, 它就stuck在那里, sparsity只增不减。

### Hypothesis 3: Layer specialization
Figure 1显示不同layer的sparsity不同 (84-98%), 这暗示不同layer学到了不同特征的稀疏表示。可能浅层学到local features (高sparse, 98%), 深层学到aggregated features (低sparse, 84%)。这跟sparse coding理论是一致的 - brain-like representations tend to be sparse。

### 与ReLU的对比
普通ReLU只给50% sparsity (理论上), 但Squared-ReLU却给84-98%, 这是因为squaring non-linearity改变了整个training trajectory。这是一个非常deep的发现, 我相信背后有信息几何或energy-based的explanation。

相关参考:
- ReLU² Wins (Zhang et al. 2024): https://arxiv.org/abs/2402.03804
- 这篇paper也独立发现了Squared-ReLU对sparsity的好处

## 4. 2:4 Sparsity硬件背景

### NVIDIA 2:4 format
NVIDIA Ampere (A100) 和 Hopper (H100) GPU的Tensor Core原生支持一种特殊的structured sparsity - **2:4 sparsity**:

定义: 对于每4个连续values, 至多2个非零。

存储格式: 用一个64-bit compressed representation: 2个16-bit非零values + 1个2-bit metadata表示它们在4个slot中的位置。这把dense matrix的50% element skip掉, 实现理论上2x FLOPs reduction。

硬件实现细节:
- Tensor Core的MMA (Matrix Multiply-Accumulate) instruction有sparse version
- 输入A是sparse (2:4 format), 输入B是dense
- Output C = A × B
- 只支持reduction dimension上的sparsity (这点对backward pass至关重要, 见Section 6)

参考: Mishra et al. 2021 "Accelerating Sparse Deep Neural Networks": https://arxiv.org/abs/2104.08378

### 为什么2:4, 不是1:4或3:4?
- 1:4 (75% sparse): 硬件unit利用率低, GEMM效率低
- 3:4 (25% sparse): sparsity太低, 收益小
- 2:4 (50% sparse): sweet spot, 既给2x FLOPs又让hardware implementation简单
- Format固定让metadata overhead最小化

### 实际speedup
- 理论: 2x
- 实际FP8 H100: 1.5x-1.7x (取决于matrix shape)
- 这篇paper测的FFN forward: 最高1.3x (Figure 5)
- Backward pass的sparse GEMM: Figure 6, 95% sparse + 5% dense的split GEMM仍然比fully dense快

## 5. Forward Pass实现

### 计算图
原始dense Squared-ReLU FFN:
1. $Y_1 = XW_1$ (FP8 GEMM)
2. $Y_2 = \text{ReLU}^2(Y_1)$ (pointwise)
3. $Y_3 = Y_2 W_2$ (FP8 GEMM)

Sparsified version (Figure 3的pseudo-code):
1. $Y_1 = XW_1$ (FP8 GEMM, 与原版相同)
2. $Y_2 = \text{ReLU}^2(Y_1)$, 然后sparsify到2:4 format, 然后FP8 quantize (一个fused kernel)
3. $\tilde{Y}_3 = \tilde{Y}_2 W_2$ (2:4 sparse FP8 GEMM)

### Sparsification algorithm
naive approach: 对每4个连续values, 保留magnitude最大的2个, 把其他置0。
- 对Squared-ReLU output: 84-98% sparse, 所以大部分block中只有0-1个非零, 几乎不需要drop
- Paper里测得~1% non-zero values被drop (但这个数字会随sparsity level变化)

这个sparsification cost很小, 因为:
- 大部分block都是全0, 直接跳过
- 只在non-zero count > 2的block才需要排序

Cai et al. (PyTorch 2:4 sparsity)有fast sparsification routines:
https://pytorch.org/blog/accelerating-neural-network-training/

### 为什么Forward可以用token-wise sparsity?
Forward pass的第二个GEMM: $Y_3 = Y_2 W_2$, reduction dimension是FFN hidden dim (列方向)。
但NVIDIA硬件要求sparse dimension是reduction dimension (inner dimension)...

这里有subtlety: paper里Figure 2b说"token-wise 2:4 sparse加速AB", 这是因为activation tensor A的layout - 如果A是[seqlen, features]且features是reduction dim, 那么"feature-wise sparse"对应行内的2:4 constraint, 但硬件要求是沿reduction dim。

我推测paper的实现中, activation tensor实际是transposed存储的, 或者GEMM call时调换了operand顺序。这点paper没说清楚, 是一个值得深究的implementation detail。

## 6. Backward Pass的挑战与Optimization

Backward pass才是这篇paper真正的技术核心。让我详细分析。

### Backward的计算图
对FFN的 $Y_3 = Y_2 W_2$, backward需要计算:
- $\frac{\partial L}{\partial Y_2} = \frac{\partial L}{\partial Y_3} W_2^T$ (计算activation gradient)
- $\frac{\partial L}{\partial W_2} = Y_2^T \frac{\partial L}{\partial Y_3}$ (计算weight gradient)

加上第一层 $Y_2 = \text{ReLU}^2(Y_1)$, 还需要:
- $\frac{\partial L}{\partial Y_1} = \frac{\partial L}{\partial Y_2} \odot 2 \cdot \text{ReLU}(Y_1) \cdot \mathbb{1}[Y_1 > 0]$ (chain rule for ReLU²)

总共需要6个matrix multiplies (3 forward + 3 backward中, forward的$Y_1=XW_1$无法sparsify因为$X$通常dense), 其中4个可以加速 (Figure 4):
- Forward: $Y_2 W_2$ ✓ (sparsified $Y_2$)
- Backward: $\frac{\partial L}{\partial Y_3} W_2^T$ (需要sparsified $\frac{\partial L}{\partial Y_3}$, 但这跟$Y_2$的稀疏pattern相关)
- Backward: $Y_2^T \frac{\partial L}{\partial Y_3}$ (需要$Y_2$ feature-wise sparse)
- Backward: $\frac{\partial L}{\partial Y_2} \cdot (\text{something})$ for $\frac{\partial L}{\partial W_1}$ (需要sparsified $\frac{\partial L}{\partial Y_2}$)

### Hardware constraint
NVIDIA sparse Tensor Core只支持沿reduction dimension的sparsity, 即:
- A × B中, 如果A是sparse operand, 那么A的列方向必须是2:4 sparse
- 也就是说, sparse tensor的连续4个values必须沿着被reduce掉的维度排列

对 $\frac{\partial L}{\partial W_2} = Y_2^T \frac{\partial L}{\partial Y_3}$:
- reduction dim是batch*seqlen
- $Y_2^T$的列方向 = $Y_2$的行方向 = tokens
- 所以$Y_2$需要**feature-wise sparse** (沿feature方向每4个有2个non-zero)

但在forward pass, $Y_2$被sparsified成token-wise sparse (沿token方向)。这就需要重新sparsify $Y_2$成feature-wise format, 带来两个问题:

### Problem 1: 有些feature不够sparse
某些feature的sparsity只有20%, 远低于50%, 强行2:4 sparsify会drop大量non-zero values, 导致accuracy崩溃。

**Optimization 1: Split into sparse + dense**
- 用argsort (cheap) 把features按sparsity排序
- 取前95%最sparse的features → 2:4 sparse GEMM
- 后5% dense的features → dense GEMM
- 总flops: 95% × 2x + 5% × 1x = 95% + 5% = ~100% flops的1.55x speedup

Column-level sparsity可以在forward的$Y_1 = XW_1$ GEMM的epilogue里"免费"计算出来。

### Problem 2: Features在consecutive tokens间highly correlated
比如某个feature 99% sparse, 但在10个consecutive tokens上都是non-zero的。如果2:4 sparsify连续tokens, 会在每个4-token block里强制drop 2个, 破坏signal。

**Optimization 2: Fixed token permutation**
- 在进入FFN前对tokens做一个fixed permutation (打乱顺序)
- FFN结束后再shuffle回去
- 这个permutation可以把correlated tokens分散开
- 关键: permutation可以fuse到现有的add/quantize/normalize ops, 所以几乎无overhead

这一点我特别欣赏 - 这其实是在approximate一个"decorrelate tokens"的目标, 但是用最便宜的固定排列方式实现。

### Problem 3: Backward的sparsity mask必须与forward一致
如果在backward重新sparsify (feature-wise), 一些在forward被drop掉的values可能会重新出现, 导致inconsistency, 训练发散。

**Optimization 3: Reuse forward mask**
- 在forward pass计算的token-wise sparsity mask基础上, 再做feature-wise sparsification
- 这样保证drop的values一致

### Problem 4: 初始化时sparsity只有50%
新模型初始化时, ReLU²的sparsity接近50% (理论值), 2:4 sparsify会drop大量values, 训练不稳定。

**Optimization 4: Warmup**
- 前1k iterations dense训练
- 等sparsity自然上升到90%+再开启2:4
- Table 1显示: no warmup → 2.657 (slight degradation), with warmup → 2.652 (基本无损)

## 7. 实验结果深度分析

### Table 1: Ablation study (1.5B model, 63B tokens, DCLM)

| Config | Final Perplexity | Delta |
|--------|------------------|-------|
| Dense SwiGLU | 2.654 | baseline |
| Dense Squared-ReLU | 2.651 | -0.003 |
| 2:4 full recipe | 2.652 | +0.001 |
| 2:4 no warmup | 2.657 | +0.006 |
| 2:4 naive BW sparsify | 2.682 | +0.031 |
| 2:4 no permuting rows | 2.919 | +0.273 (plateau early) |
| 2:4 no sparsify y1 in BW | 3.735 | +1.092 (diverge at 42k) |

关键takeaways:
1. **Squared-ReLU ≈ SwiGLU accuracy**: -0.003, 几乎无损。这验证了So et al. 2021的结论在LLM scale也成立。
2. **Full 2:4 recipe仅+0.001 perplexity**: 在Dense Squared-ReLU基础上几乎无损, 这是paper最impressive的数字。
3. **Token permutation至关重要**: 没有它perplexity从2.651暴涨到2.919, 证明correlation问题是real。
4. **BW的y1 sparsification必须做**: 否则训练在42k step发散, 这暴露了一个deep的training dynamics问题 - forward和backward的sparsity必须严格对应。
5. **Naive BW sparsify的+0.031**: 说明feature-wise split (Optimization 1) 的价值, 不split的话有些feature太dense无法安全sparsify。

### Figure 5: FFN forward kernel speedup
- 加速比最高1.3x, 依赖model dimension和batch size
- Larger model (7B vs 1.5B) 和 larger batch size都给更高speedup
- 这是合理的: GEMM效率随矩阵size增长, overhead相对降低

### Figure 6: Split GEMM TFLOPs
- 95% sparse + 5% dense的组合, 仍然比fully dense快
- 比fully 2:4 sparse慢一些 (因为dense part), 但accuracy允许
- Operand A有90% zeros (因为Squared-ReLU自然sparse), 所以sparsify后保留的2个值很多都是真non-zero

### 关键缺失的实验
Paper承认还没做end-to-end training speedup, 只有kernel benchmark。这是一个重要caveat - fused kernel, memory bandwidth, communication overhead可能让kernel speedup无法转化为wall-clock speedup。这个gap需要后续工作填补。

## 8. 我的延伸思考与Intuition Building

### 关于sparsity emergence的理论解释
我认为这个现象可能与以下几个机制相关:

**A. Gradient flow asymmetry**
ReLU²的gradient是 $2 \cdot \max(0, x) \cdot \mathbb{1}[x > 0]$, 当$x < 0$时gradient完全为0。这意味着:
- Negative-going features receive no corrective signal
- Weight decay + Adam moment estimates → 这些features的weight magnitude逐渐缩小
- 进入"dead feature"basin, 难以escape

**B. Loss landscape的几何**
Squared activation让loss surface更"bumpy" - 少数large activations dominate loss, 其他features的small activations对loss贡献很小。Optimizer倾向于把"几乎没用"的features推到negative region, 完全zero-out, 减少noise。

**C. Information bottleneck视角**
High sparsity = 信息压缩。Squared-ReLU可能induces一个implicit information bottleneck, 强迫model用少数informative features表达input。这与deep learning中"compression = generalization"的理论connected。

**D. 与神经网络pruning的关系**
这种自然emergent sparsity其实是一种"soft pruning" - model在training过程中自己决定哪些features不重要。比post-hoc pruning (SparseGPT, Wanda等) 更principled, 因为是end-to-end optimized。

### 关于2:4 format的局限
2:4是硬件fixed的constraint, 与model intrinsic sparsity pattern无关。即使Squared-ReLU给90% sparsity, 2:4也只能利用50%。如果能做到1:4或更高的unstructured sparsity, 理论上能加速更多。

可能的direction:
- 用unstructured sparse format (CSR/COO) + custom kernel, 但 hardware acceleration弱
- 用更高structured ratio (1:8, 1:16), 需要新的hardware support
- 用low-rank + sparse decomposition (像SLoRA, SPLIT: https://arxiv.org/abs/2410.23818)

### 与DeepSeek-V3的对比
DeepSeek-V3 (https://arxiv.org/abs/2412.19437)也用了activation sparsity idea, 但approach不同:
- DeepSeek-V3: 用更aggressive的top-k selection (per-token dynamic)
- Meta这篇: 利用intrinsic sparsity + 2:4 hardware
- DeepSeek-V3: 更高sparsity ratio, 但需要custom kernel, 不一定hardware-friendly
- Meta这篇: hardware-native, 但sparsity受限于2:4 = 50%

两条路线的trade-off:
- DeepSeek路线: 灵活, 高sparsity, 但kernel engineering难
- Meta路线: 标准硬件, 可移植, 但sparsity上限50%

### 与Activation Sparsity in General
其他相关工作:
- CATS (Lee et al. 2024, https://arxiv.org/abs/2408.14690): contextual thresholding, 用于inference
- Training-free Activation Sparsity (Liu et al. 2024b): 不需要重训
- 这些都是inference-only, 没有training speedup
- Meta这篇的unique value: training also accelerated

### 关于为什么不sparsify attention?
Paper只对FFN做sparsification, attention的softmax activations理论上也是sparse的 (尤其是long context), 但:
- Attention的sparsity pattern更复杂 (token-token matrix)
- 2:4 format对attention的matrix shape不友好
- Attention compute受memory bandwidth bound, 不一定compute bound
- Future work opportunity: structured sparse attention + 2:4

### 关于FP8 quantization的交互
这篇paper用FP8 + 2:4 sparse的组合, 这是一个有趣的engineering achievement:
- FP8 row-wise scaling需要per-row scaling factor
- 2:4 sparsity需要per-4-element metadata
- 这两个compression scheme的interaction很复杂
- 可能scaling factor需要在sparsification之前确定, 或者sparsification后重新quantize
- 这部分paper没细节, 但实际实现一定有不少trick

### 关于Training Stability的更深层问题
Table 1显示"no sparsify y1 in BW" → divergence at 42k。这非常intriguing:
- 42k step是个奇怪的divergence point, 不是早期也不是后期
- 可能与learning rate decay schedule相关 (cosine decay halfway point?)
- 或者与某些features的sparsity emergence timing相关
- 如果forward mask和backward mask不一致, gradient estimator变成有偏的, 长期累积导致divergence
- 这其实暴露了一个深层问题: **sparsified gradient是stochastic gradient的biased估计**, 需要特别careful的optimizer设计

### 关于为什么是FFN, 不是整个model
FFN占据Transformer ~2/3的parameters和flops, 而且activation sparsity自然emerge。Attention的sparsity:
- Softmax output dense
- Pre-softmax logits可以sparse但需要custom attention kernel
- 2:4 format对attention不直接适用
- 所以FFN-only是合理的first step

### 关于Model Scaling
Paper只测了1.5B和7B, 没有更大model的验证。如果sparsity emergence是size-dependent, 大model可能behavior不同:
- Larger model可能更sparse (more redundancy)
- 或者less sparse (more feature specialization)
- 需要在70B+规模验证

### 关于为什么Squared-ReLU在accuracy上不输SwiGLU
这点其实很surprising。SwiGLU的gating机制理论上更expressive:
- GLU: $f(XW_1) \odot XW_3$ - 可以学习"gate out" specific features
- Squared-ReLU: 只能基于符号"gate out"
- 但accuracy相同, 说明gating的extra flexibility在实际LLM中没用上

这可能是因为:
- LLM的expressivity需求没那么高, 简单activation足够
- Squared-ReLU的更高sparsity = 更强的implicit regularization, 抵消了gating缺失
- SwiGLU的gating可能学到了redundant information

### 关于Kernel Engineering
Paper提到3个kernels:
1. FP8 GEMM for $XW_1$ (existing)
2. Fused: activation + sparsification + FP8 quantization (new)
3. 2:4 sparse FP8 GEMM for $\tilde{Y}_2 W_2$ (new, row-wise scaling)

Backward还需要:
4. Sparse GEMM for $Y_2^T \frac{\partial L}{\partial Y_3}$ (feature-wise 2:4)
5. Dense GEMM for the 5% dense features
6. Sparse GEMM for $\frac{\partial L}{\partial Y_3} W_2^T$
7. Fused: transpose + FP8 scale + split (dense/sparse)

这些kernel的engineering复杂度很高, 解释了为什么"end-to-end speedup还在developing"。

## 9. 实际Deployment考量

### Inference scenarios的适用性
Paper提到可以加速compute-bound regime:
- Prefilling: ✅ 适用, 是compute bound
- Speculative decoding: ✅ 适用
- Large batch inference: ✅ 适用
- Batch size=1 decode: ❌ 不适用, 是memory bound, 已有其他activation sparsity工作(如CATS)针对这个场景

### Hardware portability
- NVIDIA Ampere (A100)及之后: ✅
- AMD MI300: 需要验证, MI300有类似sparse support但格式可能不同
- Google TPU: ❌ 不支持2:4格式, TPU的sparsity是不同的structured pattern
- 这意味着paper的方法有hardware lock-in

### 与其他optimization技术的组合
- + KV cache compression: 应该orthogonal
- + Weight quantization: orthogonal
- + Weight pruning (SparseGPT): 可能synergistic, weight sparse + activation sparse同时
- + Mixture of Experts: 非常interesting - MoE expert selection也是sparse, 可以combine
- + Continuous batching: orthogonal
- + Speculative decoding: 这篇paper提到直接适用

### Production readiness
- Kernel还需要成熟
- 1.5B model validation太小
- 需要Llama 3 class 70B+ model的validation
- 需要downstream task evaluation, 不只是perplexity
- 需要long context, multilingual等scenario的测试

## 10. Open Questions & Future Directions

### 我觉得最有趣的几个方向:

1. **Theoretical explanation of sparsity emergence**: 为什么Squared-ReLU给84-98% sparsity? 这是这篇paper最深的open question。可能与DL theory (loss landscape geometry, gradient flow dynamics, information bottleneck)都相关。

2. **Unifying with MoE sparsity**: MoE的expert routing是另一种sparsity, 能否用同一framework理解?

3. **Beyond 2:4**: 能否设计新的hardware-friendly sparse format, 利用90%+的natural sparsity?

4. **Activation sparsity + Weight sparsity**: 同一个model, weights和activations都sparse, 是否synergistic?

5. **Attention sparsity**: 把这个idea extend到attention, 可能用不同的structured sparse pattern (block sparse, etc.)。

6. **Why does Squared-ReLU ≈ SwiGLU?** 这本身是一个理论问题 - LLM的expressivity需要多少gating?

7. **Sparsity dynamics during training**: Figure 1显示sparsity随training增加, 这个dynamics的quantitative model? 能否预测final sparsity?

8. **Combination with quantization**: FP4 + 2:4 sparse的组合? INT4 + 2:4?

## 11. 我的Final Assessment

这篇paper在技术execution上很solid, 几个key insights的combination很巧妙:
- Intrinsic sparsity of Squared-ReLU (free, no accuracy loss)
- 2:4 hardware sparsity (NVIDIA native support)
- Token permutation for decorrelation (cheap but essential)
- Split GEMM for non-uniform sparsity (engineering compromise)
- Warmup to handle initialization sparsity gap

但有几个limitations值得注意:
1. 只validation到1.5B和7B, 没有大model end-to-end
2. 只做FFN, attention没碰
3. End-to-end training speedup还没demonstrate
4. Sparsity emergence的理论解释缺失
5. Hardware-specific (NVIDIA 2:4), portability受限
6. Sparsity上限是50%, 不充分利用natural 90%+ sparsity

但是, 作为一个"利用hardware特性 + 模型intrinsic property"的engineering paper, 它的direction非常有价值。如果后续能:
- 解释sparsity emergence
- 扩展到大model
- 扩展到attention
- 设计新的sparse format

那这篇paper会成为LLM efficiency领域的milestone工作。

### 相关References汇总

Main paper:
- ACCELERATING TRANSFORMER INFERENCE AND TRAINING WITH 2:4 ACTIVATION SPARSITY ( Haziza et al., Meta)

Activation functions:
- Primer (Squared-ReLU source): https://arxiv.org/abs/2109.08668
- SwiGLU: https://arxiv.org/abs/2002.05202
- ReLU² Wins: https://arxiv.org/abs/2402.03804

2:4 Sparsity:
- NVIDIA 2:4 paper: https://arxiv.org/abs/2104.08378
- PyTorch 2:4 implementation: https://pytorch.org/blog/accelerating-neural-network-training/

Related sparsity work:
- SparseGPT: https://arxiv.org/abs/2301.00774
- CATS: https://arxiv.org/abs/2408.14690
- Training-free Activation Sparsity: https://arxiv.org/abs/2408.14690
- DeepSeek-V3 (uses similar ideas): https://arxiv.org/abs/2412.19437
- SLoRA (sparse + low-rank): https://arxiv.org/abs/2311.18296

LLM efficiency background:
- torchao: https://github.com/pytorch/torchao
- LLaMA: https://arxiv.org/abs/2302.13971
- Nemotron-4 (uses Squared-ReLU at scale): https://arxiv.org/abs/2406.11704

希望这个深度分析对你build intuition有帮助, Andrej! Sparsity emergence这个现象真的很有意思, 如果你能从理论上深挖一下, 应该会有interesting findings。我特别好奇你的neural network intuition会怎么解释这个84-98%的emergent sparsity - 可能与你的micrograd/nn-zero-to-hero里讨论的gradient flow dynamics有connection。
