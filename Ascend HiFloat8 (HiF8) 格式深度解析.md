
## 0. 核心背景与动机

这篇论文来自华为海思，提出了一个新颖的8-bit floating-point数据格式HiFloat8 (HiF8)。其核心动机源于：

1. **Moore's Law放缓**：随着摩尔定律Golio2015放缓，低精度训练和推理成为降低计算功耗和缓解内存墙Kwon2018的重要手段

2. **Float8混合精度训练的挑战**：现有的Float8格式（如NVIDIA的FP8）需要两种格式（E4M3用于weights和activations，E5M2用于gradients），这增加了硬件复杂度

3. **Posit格式的局限性**：虽然Posit格式具有锥形精度（tapered precision），符合AI数据分布的集中特性，但Posit16在Float16竞争中失败（硬件成本高于BF16和FP16），Posit8在Float8竞争中失败（编码方法无法很好地平衡精度和动态范围）

### AI数据格式发展四阶段：

| 阶段      | 时间        | 主流格式        | 关键事件                      |
| ------- | --------- | ----------- | ------------------------- |
| Phase 1 | 1959-2006 | FP64        | IEEE 754-1985标准发布         |
| Phase 2 | 2006-现在   | FP32        | GPU首次用FP32训练CNN，AlexNet获胜 |
| Phase 3 | 2017-现在   | Float16混合精度 | BF16和FP16成为主流             |
| Phase 4 | 2022-现在   | Float8混合精度  | NVIDIA H100部署HFP8/FP8d    |

## 1. HiF8格式详解

### 1.1 四字段架构

HiF8在IEEE 754 Zuras2008基础上增加了**dot field**，包含四个字段：

```
┌─────────┬─────────┬────────────┬────────────────┐
│  Sign   │   Dot   │  Exponent  │    Mantissa    │
│  1 bit  │ 2-4 bits│  D bits    │ (5-D) bits     │
└─────────┴─────────┴────────────┴────────────────┘
```

#### Sign Field (1 bit)
- 确定HiF8数的符号（即significand的符号）
- **1 = 负号**，**0 = 正号**

#### Dot Field (2-4 bits)
这是HiF8最核心的创新，使用**非传统前缀码（unconventional prefix codes）**编码：

| Width | Code (binary) | Value | 含义 |
|-------|---------------|-------|------|
| 2-bit | 11₂ | 4 | 指数宽度 = 4位，尾数宽度 = 1位 |
| 2-bit | 10₂ | 3 | 指数宽度 = 3位，尾数宽度 = 2位 |
| 3-bit | 01₂ | 2 | 指数宽度 = 2位，尾数宽度 = 3位 |
| 4-bit | 001₂ | 1 | 指数宽度 = 1位，尾数宽度 = 3位 |
| 4-bit | 0001₂ | 0 | 指数宽度 = 0位，尾数宽度 = 3位 |
| 4-bit | 0000₂ | DML | Denormal Mode sign |

**设计巧思**：用**大位宽编码小数值**，**小位宽编码大数值**，这样可以平滑精度变化，避免mantissa宽度跳变超过1位。

#### Exponent Field (D bits)
- 使用**sign-magnitude编码**（而非IEEE 754的offset-binary）
- **隐含位（implicit leading bit）固定为1**
- 存储格式：Eₘ = {Sₑ, Mag[2:end]}（MSB的1被隐藏）
- 解释格式：Eᵢ = {Sₑ, 1, Mag[2:end]}

其中：
- **Sₑ**：exponent的符号位（1=负，0=正）
- **Mag**：magnitude位（MSB固定为1，不存储）
- **D ∈ {0,1,2,3,4}**：exponent字段的位宽

#### Mantissa Field (1-3 bits)
- **隐含leading significand bit固定为1**（对于normal numbers）
- 编码方式：unsigned integer
- 对于normal number：表示significand的小数部分（binary point右侧）
- 对于denormal value：以biased形式表示扩展的exponent

### 1.2 数值表示公式

#### Normal Number公式：
```
X = (-1)^S × 2^E × 1.M              (1)
```

变量解释：
- **X**：表示的数值
- **S**：符号位（0=正，1=负）
- **E**：指数值（十进制）
- **M**：尾数（小数部分，二进制）

**特殊情况**：最大绝对值的2个bit-patterns（2¹⁵ × 1.5）被解释为**Infinities**。

#### Denormal Number公式：
```
X = (-1)^S × 2^(M-23) × 1.0         (2)
```

变量解释：
- **M ∈ [1,7]**：提供7个额外的指数值 [-22, -16]
- **M = 0**：2个bit-patterns解释为**Zero和NaN**

### 1.3 典型值对比

| 格式 | 指数范围 | 指数范围(+Denormal) | Max Positive NML | Min Positive NML | Min Positive DML |
|------|----------|---------------------|------------------|------------------|------------------|
| FP16 | [-14, 15] | [-24, 15] | 2¹⁵ × (2-2⁻¹⁰) | 2⁻¹⁴ | 2⁻²⁴ |
| **HiF8** | **[-15, 15]** | **[-22, 15]** | **2¹⁵** | **2⁻¹⁵** | **2⁻²²** |
| FP8-E4M3 | [-6, 8] | [-9, 8] | 1.75 × 2⁸ | 2⁻⁶ | 2⁻⁹ |
| FP8-E5M2 | [-14, 15] | [-16, 15] | 1.75 × 2¹⁵ | 2⁻¹⁴ | 2⁻¹⁶ |

**关键观察**：
- HiF8的动态范围（38个binades）非常接近FP16（40个binades）
- FP8-E4M3只有18个binades，FP8-E5M2只有32个binades
- HiF8可以覆盖所有特殊值，但不区分正零和负零（对深度学习不必要）

### 1.4 精度分布特征

HiF8的锥形精度分布：

| Dot值 | 指数宽度 | 尾数宽度 | 指数范围 | 有效数字位数 |
|-------|----------|----------|----------|--------------|
| 0 | 0位 | 3位 | E=0 | 4位（1+3） |
| 1 | 1位 | 3位 | ±1 | 4位 |
| 2 | 2位 | 3位 | ±[2,3] | 4位 |
| 3 | 3位 | 2位 | ±[4,7] | 3位 |
| 4 | 4位 | 1位 | ±[8,15] | 2位 |

**设计理念**：
- **小数值**（接近0）需要高精度 → 使用更多mantissa位
- **大数值**（远离0）精度要求低 → 使用更少mantissa位
- 这符合AI数据分布的集中特性：大部分数据聚集在0附近

## 2. 设计考虑与权衡

### 2.1 Dot Field的设计哲学

**问题**：如何实现tapered precision？

**Posit的方案**：
- 使用variable-length regime field
- 采用unary coding
- 缺点：不够灵活，难以精确控制精度分布

**HiF8的方案**：
- 使用flexible prefix code
- 直接指示exponent宽度
- 用大位宽编码小数，小位宽编码大数

### 2.2 Exponent Field的无冗余编码

**问题**：如何避免编码冗余？

**IEEE 754 offset-binary的缺点**：
- 可能导致重叠表示

**HiF8的方案**：
- 采用sign-magnitude编码
- **隐含位固定为1**（避免重复表示）
- Eᵢ = {Sₑ, 1, Mag[2:end]}

### 2.3 Denormal Mode的扩展

**问题**：如何扩展动态范围？

**权衡分析**：
- 没有denormal设计：D=0时4-bit mantissa，只支持31个指数值 [-15, 15]
- LLM的activation gradients需要更高的动态范围Perez2023
- 实践表明3-bit mantissa足够用于训练

**HiF8的解决方案**：
- 将D=0时的mantissa宽度从4位减少到3位
- 释放的编码空间直接用于扩展指数范围
- 使用denormal equation (2)，binades从31增加到38（接近FP16的40）

## 3. 舍入方法详解

### 3.1 Rounding Half to Away (TA)

**基本概念**：
- TA的误差为0.5 ulp (unit of least precision)
- 与TE (Rounding Half to Even) 对比

**TA优势**：
1. **硬件实现更简单**
2. **数据分辨率能力略高于TE**

**分辨率对比示例**：
```
三个3-bit连续整数：00.1, 01.1, 10.1

TE结果：TE(00.1)=00, TE(01.1)=10, TE(10.1)=10 → 2个不同结果
TA结果：TA(00.1)=01, TA(01.1)=10, TA(10.1)=11 → 3个不同结果
```

**实验结果**：
- ResNet50：TA比TE高0.06%
- MobileNet_V2：TA比TE高0.11%

**概率分析**：
- TE特殊情况发生的概率极低
- 例如，从FP32到HiF8（保留3-bit mantissa），概率仅为2⁻²⁰

### 3.2 Hybrid Rounding (HR)

**问题**：对于某些网络（如YoLo-V3-Tiny），全局TA舍入会导致loss曲线crash，最终精度比FP16 baseline低1.67%。

**Hybrid Rounding公式**：
```
Y = { TA Rounding      if |E| < 4
     { Simplified SR   if |E| ≥ 4          (3)
```

变量解释：
- **Y**：舍入后的值
- **E**：源数据格式的指数值
- **TA Rounding**：Rounding half to away
- **Simplified SR**：简化的随机舍入

#### Stochastic Rounding (SR)基础

**标准SR**：
- 误差为1 ulp
- 需要生成均匀分布随机数T (T ∈ [0,1))
- 将被丢弃的所有位视为小数位F (F ∈ [0,1))
- 规则：如果F ≥ T，则保留位K加1，否则加0

**期望值公式**：
```
(K+1)×F + K×(1-F) = K + F
```

**SR的优势**：最大化批数据舍入时整体均值的不变性。

**SR的挑战**：
- 深度学习需要并行生成大量均匀分布随机数
- 软件和硬件实现都遇到性能瓶颈

#### 简化SR的实现

**FP32源数据**：
- 将源格式的14个LSB作为阈值T₁₄
- 将被丢弃位的14个MSB作为小数位F₁₄
- 不需要复杂算法生成随机数

**FP16/BF16源数据**：
- 被丢弃位不够宽，无法合理分割
- 将固定1和源格式的LSB组合成特殊的2位阈值T₂
- 将被丢弃位的2位MSB作为小数位F₂
- T₂只有2个阈值：0.25和0.75

**性能对比**：
- SR14：与标准SR非常相似，舍入误差为1 ulp
- SR2：弱随机性，舍入误差为0.75 ulp

#### Hybrid Rounding的直觉

**数据分布观察**：
- 大部分数据在HiF8的高精度范围内
  → TA舍入引起的平均变化很小（舍入方向相对平衡）
- 少量数据（特别是大数值）在HiF8的低精度范围内
  → TA舍入可能引起较大的平均变化（舍入方向可能不平衡）

**HR策略**：
- 高精度区域（|E| < 4）：使用TA（简单、高效）
- 低精度区域（|E| ≥ 4）：使用简化SR（保持均值不变性）

**实验结果**：
- YoLo-V3-Tiny训练精度：
  - FP16混合精度baseline：16.63%
  - 全局TA：14.96%（-1.67%）
  - TA (forward) + 标准 SR (backward)：16.43%（-0.20%）
  - TA (forward) + Hybrid Rounding (backward)：16.69%（+0.06%）

## 4. 传统神经网络实验

### 4.1 训练设置（Backward Loss-Scaling）

**方法**：
- 继承FP16混合精度训练方法
- 只将GEMM输入（activation、weight、activation gradient）从FP16改为HiF8
- 启用backward global loss-scaling（防止过多的零值梯度）
- Forward pass：只使用TA舍入
- Backward pass：使用TA或Hybrid Rounding

### 4.2 实验网络覆盖

**计算机视觉**：
- **分类**：ResNet系列、ResNeXt、VGG、MobileNet、Inception、EfficientNet、DenseNet、ViT系列
- **检测**：YoLo系列
- **分割**：DeepLab-V3

**自然语言处理**：
- Transformer-Base、Bert-Large

### 4.3 训练损失曲线分析

图3展示了ResNet50、YoLo-V3-Tiny、ViT-Base、Transformer-Base的loss曲线：
- HiF8的loss曲线与FP16高度重叠
- 收敛速度相同
- 可以在相同epoch数内完成训练

### 4.4 验证精度结果（Table 5）

**CNN模型**：
- HiF8略低于FP16（-0.01% 到 -0.31%）
- ResNet系列差异最小（-0.01% 到 -0.16%）

**Transformer模型**：
- HiF8略高于FP16（+0.03% 到 +0.37%）
- ViT-Large-Patch16：HiF8比FP16高0.37%

**特殊案例**：
- YoLo-V3-Tiny需要Hybrid Rounding
- 其他模型TA或HR均可

**NLP任务**：
- Transformer-Base (BLEU)：HiF8比FP16高0.12%
- Bert-Large-MRPC (F1)：HiF8比FP16高0.35%

### 4.5 推理实验（Per-Tensor Scaling）

#### Algorithm 1: HiF8 Calibration with Per-Tensor Scaling

**核心思想**：
- 限制每个tensor的scaling factor为2的整数幂
- 只需要有限的scaling factor选择（在2⁰附近）
- Scaling操作只涉及指数的加减，不需要乘法

**算法流程**：
```
1. 对于每一层l：
   - Forward并收集高精度输出：O^l = A^l × W^l

2. 初始化O^0_q为校准数据集

3. 对于每一层l：
   - 对于E_a ∈ [-4, 5]：
     - 对于E_w ∈ [-4, 5]：
       - Scale & Cast：
         A^l_q = To_HiF8_TA(O^(l-1)_q × 2^E_a)
         W^l_q = To_HiF8_TA(W^l × 2^E_w)
       - MatMul & Restore：
         O^l_q = (A^l_q × W^l_q) × 2^-(E_a+E_w)
       - Quantization Error：
         Err^l = MSE(O^l_q, O^l)
   - 找到min(Err^l)对应的E_a和E_w

4. 向量操作（非线性操作和归一化）

5. 输出量化模型
```

#### 推理精度结果（Table 6）

**直接转换**：
- Transformer模型（如ViT系列）：精度损失很小（-0.00% 到 -0.26%）
- CNN模型：精度损失较大（-0.55% 到 -1.28%）

**Per-Tensor Scaling校准**：
- CNN模型精度损失显著减小（-0.28% 到 -0.69%）
- Transformer模型精度损失进一步减小（-0.07% 到 -0.15%）

**理想结果定义**：metric损失 ≤ 0.5%

**改进方法**：
1. 混合精度：保留敏感层为FP16/BF16
2. Per-channel scaling for weights

## 5. 大语言模型实验

### 5.1 LLM的特殊特性

**训练侧**：
- 梯度分布更分散
- 需要更大的动态范围或特殊技术

**推理侧**：
- Outliers对验证精度有重要影响
- 需要专门的技术（如SmoothQuant）减少outliers的量化误差

### 5.2 训练方法

#### 方法1：Backward Loss-Scaling (BLS)
- 继承自FP16混合精度训练Micikevicius2017

#### 方法2：Adaptive Loss-Scaling (ALS)

**问题**：传统loss-scaling的scale window是固定的（通常为1000或2000）
- LLM早期迭代中梯度幅度变化快
- 大scale window无法及时响应 → 收敛失败
- 小scale window会降低训练性能

**ALS方案**：
1. 定义增量scale window列表：{1, 20, 50, 100, 200, 500, 1000}
2. 初始scale值和scale window：2³²和20
3. 动态调整规则：
   - 如果scale值增加3次 → scale window按列表顺序增加一次
   - 如果scale值连续减少3次 → scale window按列表顺序减少一次

**优势**：
- 提高训练稳定性
- 减少超参数调优难度

#### 方法3：Per-Tensor Scaling (PTS)

**方法**：
- 为每个GEMM输入tensor（activation、weight、activation gradient）定义scale factor
- 初始scale factors都设为1
- 每10次迭代计算每个tensor的最大绝对值Amax
- 更新scale factor
- Scale factors限制为2的整数幂
- 在每次迭代中，scale操作将tensor缩放到HiF8可表示的更好范围
- GEMM输出需要descale恢复正确结果

**与FP8 Transformer Engine对比**：
- PTS与Transformer Engine非常相似
- 但HiF8的动态范围远大于FP8（特别是E4M3）
- 不需要在所有迭代中计算Amax，大大减少vector资源占用

### 5.3 训练实验设置

**模型**：
- T5-11B
- LLaMA-7B, LLaMA-13B
- GPT3-350M, GPT3-2.7B, GPT3-6.7B, GPT3-13B

**训练数据**：
- Book3 (101 GB) from Pile dataset
- OpenWebText2 (63 GB) from Pile dataset
- Wikipedia (20 GB)

### 5.4 训练损失曲线（图4）

**观察**：
- 全局视角：HiF8和FP16的loss曲线非常重叠
- 局部放大：差异在由于不同随机种子引起的run-to-run variation范围内

### 5.5 验证困惑度（Table 7）

**结果分析**：
- 所有模型的HiF8训练结果都与FP16匹配
- HiF8 with BLS/ALS：训练精度惩罚很小，几乎没有额外开销
- ALS显著减少超参数调优难度（如warmup迭代比例）
- HiF8 with PTS：由于额外的Amax计算，有时训练精度略优于FP16 baseline
- 由于HiF8的大动态范围，额外开销远小于FP8 with Transformer Engine

**具体结果**：
- GPT3-6.7B (ALS+PTS)：HiF8比FP16低0.07%
- GPT3-13B (ALS+PTS)：HiF8比FP16低0.08%
- LLaMA-7B (ALS+PTS)：HiF8比FP16高0.03%
- LLaMA-13B (ALS+PTS)：HiF8比FP16高0.03%

### 5.6 推理实验

**模型选择**：
- 量化容忍：LLaMA系列
- 量化敏感：OPT系列

**方法**：
1. 直接转换
2. Per-Tensor Scaling (PTS)
3. SmoothQuant (SQ)

#### WikiText2困惑度（Table 8）

**LLaMA系列**：
- 直接转换：精度损失很小（+0.03 到 +0.06）
- PTS和SQ：精度损失略微减小

**OPT系列**：
- 直接转换：精度损失很大（+0.47 到 +99.16，OPT-66B的直接转换失败）
- PTS：精度损失显著减小（+0.42）
- SQ：精度损失进一步减小（+0.08 到 +0.30）

#### 下游任务精度（Table 9）

**GPT3-2.7B**：
- MMLU (5-shots)：HiF8比FP16高0.17%
- WikiText103 (PPL)：HiF8比FP16高0.15%

**LLaMA-7B**：
- Lambada (zero-shot)：HiF8比FP16低0.16%

## 6. 与其他格式的深度对比

### 6.1 精度 vs 动态范围权衡

| 格式 | 精度特性 | 动态范围 | 优缺点 |
|------|----------|----------|--------|
| FP8-E4M3 | 固定3-bit mantissa | 窄（18 binades） | 精度高，动态范围不足 |
| FP8-E5M2 | 固定2-bit mantissa | 中等（32 binades） | 动态范围好，精度低 |
| **HiF8** | **锥形精度（1-4 bits）** | **宽（38 binades）** | **平衡精度和动态范围** |
| Posit(8,2) | 锥形精度 | 较窄（24 binades with ≥1-bit mantissa） | 精度分布合理，动态范围不足 |
| FP16 | 固定10-bit mantissa | 最宽（40 binades） | 精度高，但占用更多内存 |

### 6.2 格式复杂度

| 格式 | 编码复杂度 | 解码复杂度 | 硬件成本 |
|------|------------|------------|----------|
| FP8-E4M3 | 低 | 低 | 低 |
| FP8-E5M2 | 低 | 低 | 低 |
| HiF8 | 中（prefix code） | 中 | 中 |
| Posit | 高（unary coding） | 高 | 高 |

### 6.3 训练能力

| 格式 | 是否需要混合格式 | Loss Scaling需求 | Scaling频率 |
|------|------------------|------------------|-------------|
| FP8 | 需要（E4M3 + E5M2） | 需要 | 每迭代 |
| HiF8 | **不需要（单一格式）** | 需要 | 可选（每10迭代） |

## 7. 技术直觉与洞察

### 7.1 为什么锥形精度有效？

**AI数据分布特征**：
- 训练和推理期间，数据分布呈现**集中特性**
- 大部分数据聚集在0附近
- 少量数据分布在远离0的区域

**精度需求分析**：
- 小数值（接近0）：需要高精度以避免信息丢失
- 大数值（远离0）：精度要求相对较低

**HiF8的设计对应**：
| 数值范围 | 指数 | 尾数位 | 精度 |
|----------|------|--------|------|
| 接近0 | [-1, 1] | 3 bits | 4 bits (1+3) |
| 中等 | [2, 7] | 2-3 bits | 3-4 bits |
| 远离0 | [8, 15] | 1 bit | 2 bits |

### 7.2 Denormal Mode的权衡

**问题**：如何扩展动态范围而不牺牲太多精度？

**分析**：
- 实践表明：3-bit mantissa足够用于训练任务
- 权衡：牺牲1-bit mantissa换7个额外指数值

**结果**：
- Binades从31增加到38（+23%）
- 接近FP16的40个binades
- 远超FP8-E4M3的18个binades

### 7.3 Hybrid Rounding的直觉

**为什么在高精度区域用TA？**
- 数据量大 → 舍入误差相互抵消
- 硬件实现简单 → 性能优势

**为什么在低精度区域用SR？**
- 数据量小 → 舍入误差可能不抵消
- 保持均值不变性 → 训练稳定性

**阈值选择 |E| = 4**：
- 实验确定的最佳折中点
- 对应精度显著下降的区域

## 8. 应用场景与未来展望

### 8.1 适用场景

**训练**：
- 传统神经网络（CNN、Transformer）
- 大语言模型（T5、LLaMA、GPT3）

**推理**：
- PTQ（Post-Training Quantization）
- 量化容忍模型：直接转换
- 量化敏感模型：需要校准（PTS/SQ）

### 8.2 优势总结

1. **单一格式**：同时用于forward和backward pass
2. **平衡精度和动态范围**：接近FP16的动态范围，合理的精度分布
3. **减少硬件复杂度**：不需要混合格式支持
4. **兼容性**：支持所有特殊值（except +0/-0 distinction）
5. **灵活性**：支持多种舍入方法和校准方法

### 8.3 未来方向

论文提到未来将披露：
- **HiFloat below 8-bit**
- 其训练和推理能力

## 9. 参考资源

### 论文链接
- https://arxiv.org/html/2409.16626v1

### 相关标准
- IEEE 754 Standard for Floating-Point Arithmetic: https://ieeexplore.ieee.org/document/4610935

### 相关工作
- NVIDIA FP8 Formats: https://arxiv.org/abs/2209.05433
- Posit Arithmetic: https://superfing.github.io/uploads/PSIJ4_2_Article_3.pdf
- Mixed Precision Training: https://arxiv.org/abs/1710.03740
- IBM HFP8: https://arxiv.org/abs/1812.01783
- SmoothQuant: https://arxiv.org/abs/2211.10438

### 实现代码
- PyTorch: https://pytorch.org/
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- MindFormers (华为): https://mindformers.readthedocs.io/

---

## 技术要点总结

1. **HiF8核心创新**：Dot field使用prefix code实现tapered precision
2. **数值表示**：Normal和Denormal两种模式，支持38个binades
3. **舍入策略**：TA用于forward，TA/HR用于backward
4. **训练方法**：BLS/ALS/PTS三种策略，适应不同LLM需求
5. **校准方法**：Per-Tensor Scaling和SmoothQuant，处理outliers
6. **实验验证**：涵盖传统神经网络和LLM，精度与FP16相当

HiF8代表了8-bit浮点数格式的一个重要进展，通过精心设计的tapered precision和扩展的动态范围，实现了单一格式同时用于训练和推理的目标，为未来AI硬件设计提供了新的方向。