---
source_pdf: MambaVision.pdf
paper_sha256: dd89de18b35caae38f8cb13fa98f5b84211f58108ca423c6c2e6b24f3285afbc
processed_at: '2026-08-05T16:15:56-07:00'
target_folder: Automata
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MambaVision 人话版

## 一句话总结

Mamba在vision里直接用效果一般，NVIDIA这帮人发现**前面用CNN、中间用Mamba、最后用Transformer**，三种工具各司其职，效果好还跑得快。

---

## 为啥要搞这个？

Mamba出来的时候NLP圈炸了——linear complexity，还能selectively关注不同input，看起来比Transformer强。Vision圈马上跟风：Vim、VMamba、EfficientVMamba一堆paper冒出来。

但结果尴尬：**纯Mamba vision model跑不过Swin、ConvNeXt这些老架构**。原因说白了：

Mamba设计的时候假设"token有顺序"——语言确实有顺序，"我爱你"和"你爱我"意思不同。但image pixel呢？左上角的pixel和右下角的pixel，谁先谁后？没意义。

所以Mamba处理image的时候，得人为给pixel排个序（比如raster scan，从左到右从上到下）。这个排序是arbitrary的，导致信息flow不自然。而且autoregressive意味着必须一个个pixel处理，global context很难capture——你得看到最后一个pixel才能理解整张图。

Vim的solution是双向scan（正向+反向），VMamba是四向scan。但这都是"打补丁"——compute翻倍翻四倍，latency飙升，效果还没好多少。

NVIDIA的insight很简单：**别在Mamba内部硬搞global context了，直接用self-attention补这个能力**。

---

## 架构怎么设计的？

先看整体结构，其实就是一个典型的hierarchical pyramid，跟Swin、ConvNeXt一个套路：

```
Image 224×224
    ↓ Stem
Stage 1: 56×56, 纯CNN
    ↓ 
Stage 2: 28×28, 纯CNN  
    ↓
Stage 3: 14×14, 一半Mamba + 一半Attention
    ↓
Stage 4:  7×7,  一半Mamba + 一半Attention
```

为啥这么分？直觉很清楚：

**高分辨率阶段用CNN**：56×56有3136个token，如果用attention，$3136^2 \approx 10M$次计算。CNN天然处理high-resolution spatial data，inductive bias强，还快。SSM在这阶段也不合适——sequence太长，scan慢。

**低分辨率阶段才hybrid**：14×14只有196个token，7×7只有49个token。这时候attention的quadratic cost可以接受，SSM也能高效处理。而且这时候feature已经抽象化了，需要global reasoning。

**最关键的发现：attention放最后，不放前面**

这个ablation特别有意思。作者试了6种排列方式：

| 排列 | 效果 |
|------|------|
| 随机乱放 | 81.3% |
| 前半段attention | 81.5% |
| 交替Mamba/Attention | 81.4% |
| 后1/4用attention | 81.9% |
| **后1/2用attention** | **82.3%** |

人话翻译：attention越往后放越好。为什么？

Early layers的任务是"提取局部特征"——边缘、纹理、角点。这种local pattern用CNN/Mamba够了，attention在这里是overkill还浪费compute。

Late layers的任务是"理解全局"——这个物体是啥、跟环境什么关系、scene的语义。这时候需要"看到全图"，attention的global receptive field最合适。

这跟人脑直觉一致：视觉皮层early areas处理局部feature，higher areas做object recognition和scene understanding。

---

## MambaVision Mixer改了啥？

原版Mamba mixer大概长这样：

```
Input → Linear → split(x, z)
x → causal_conv → SSM_scan → y_ssm
output = z * SiLU(y_ssm)  # gating
```

MambaVision改了三处：

### 改动1: Causal conv换成regular conv

原版用causal conv是因为要保持autoregressive——当前token只能看过去的token。但image不需要这个约束，pixel之间是spatial关系不是temporal关系。换成regular conv，信息双向流动。

### 改动2: 加了一条symmetric branch

原版只有SSM这一条path。MambaVision加了一条平行的path：只有conv + activation，没有SSM。

为啥？SSM本质是sequential的，会损失spatial information。加一条non-SSM branch，相当于保留了一份"原始spatial features"的copy，让network自己决定怎么混合。

### 改动3: Gating换成Concat

原版用$y = z \cdot \text{SiLU}(y_{ssm})$——multiplicative gating。问题是如果$y_{ssm}$接近0，整个output就没了，information bottleneck。

MambaVision改成concat两个branch然后linear projection。这样两个branch的信息都保留，让下游layer学习怎么混合。

Ablation数据：
- 原版Mamba直接搬：80.5%
- + regular conv：80.9%
- + symmetric branch：81.3%  
- + concat：82.3%

每一步都有效，incremental improvement累积1.8%。

---

## 实验结果咋样？

### ImageNet-1K

跟同期模型比，MambaVision建立了新的Pareto front——同样accuracy下throughput高得多。

拿MambaVision-B (84.2%) 举例：
- vs VMamba-B (83.9%): accuracy高0.3%，throughput快**5.7倍**（3670 vs 645 img/s）
- vs Swin-B (84.6%): accuracy低0.4%，throughput快**6.9倍**
- vs ConvNeXt-B (83.8%): accuracy高0.4%，throughput快**2.5倍**

5.7倍throughput提升是啥概念？原来训练一天的model现在四小时跑完。

### 下游任务更明显

Detection (COCO)：
- MambaVision-B: AP 52.8
- ConvNeXt-B: AP 52.7
- Swin-B: AP 51.9

Segmentation (ADE20K)：
- MambaVision-T: 46.0 mIoU
- Swin-T: 44.5 mIoU
- 提升1.5个点，相当显著

Downstream task提升比classification大，说明features更generalizable。Hybrid design同时有local和global能力，transfer到dense prediction任务时优势明显。

### Scaling到ImageNet-21K

这是第一次有Mamba-based vision model scale到21K。739M参数的L3变体在512分辨率达到88.1% Top-1，跟同期SOTA的ViT-Large一个量级。

之前Vim、VMamba都没敢report 21K结果，可能是因为pure Mamba架构scalability不好。MambaVision的hybrid design让它继承了ViT的scaling property。

---

## 为啥这么快？

throughput提升的来源：

1. **高分辨率用CNN**：避免了ViT/Swin在early stage处理大量token的问题
2. **SSM是linear complexity**：比attention便宜
3. **只有最后几层用attention**：quadratic cost在低分辨率token少的时候可接受
4. **Single forward pass**：不像Vim要双向scan，compute直接减半
5. **Hardware-aware**：Mamba的CUDA kernel针对A100优化过

---

## Attention可视化说明啥？

作者画了attention map，发现最后几层的self-attention确实学到了semantic regions：
- 飞机图：attention覆盖整个plane body
- 鸟图：集中在head和tail（fine-grained features）
- 人拿东西：同时关注人和object

这说明self-attention确实在做"global reasoning"的工作，验证了架构设计的假设。

---

## 跟其他架构的关系

**vs Vim**: Vim用双向SSM试图capture global context，compute翻倍但效果不如MambaVision的single pass + attention。说明"硬改SSM内部"不如"用attention补足"。

**vs VMamba**: VMamba用四向scan，更expensive。MambaVision用regular conv（双向）+ attention，更simple更effective。

**vs Swin**: Swin全程window attention，early stage也在做attention（虽然windowed）。MambaVision early stage直接CNN，更便宜。

**vs ConvNeXt**: ConvNeXt纯CNN，large kernel但终究是local。MambaVision有global receptive field。

---

## 我的intuition总结

1. **没有银弹**：每种module有各自的inductive bias，适合不同的abstraction level。CNN适合local spatial pattern，SSM适合sequential dependency，attention适合global reasoning。硬要用一种module搞定所有level是suboptimal的。

2. **Order matters**: 不是把几种module堆在一起就行，**谁在前谁在后**很关键。Early cheap + late expensive这个pattern在FasterViT、Swin、MambaVision都反复出现。

3. **SSM在vision的根本问题是inductive bias mismatch**，不是complexity问题。Mamba的causal/sequential假设对language是inductive bias，对image是bias。Fix方法要么改SSM本身（难且效果一般），要么用其他module补足（简单且有效）。

4. **Information preservation**: Concat > gating。Gating是lossy的，concat是lossless的。让network自己学习如何混合信息，比hard-coded gating好。

5. **Hybrid才是practical winner**: 纯CNN、纯ViT、纯Mamba都有各自的limitation。工业界最终deploy的model往往是hybrid——FasterViT、EfficientFormer、MaxViT都是这个思路。MambaVision把这个pattern又验证了一遍。

---

## References

- MambaVision paper: https://arxiv.org/abs/2407.08083
- GitHub: https://github.com/NVlabs/MambaVision
- Mamba original: https://arxiv.org/abs/2312.00752  
- Vim: https://arxiv.org/abs/2401.09417
- VMamba: https://arxiv.org/abs/2401.10166
- Swin Transformer: https://arxiv.org/abs/2103.14030
- ConvNeXt: https://arxiv.org/abs/2201.03545
- FasterViT: https://arxiv.org/abs/2306.06189
- ViT: https://arxiv.org/abs/2010.11929
- ImageNet: https://www.image-net.org/
- MS COCO: https://cocodataset.org/
- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/

---

# MambaVision 详解：Hybrid Mamba-Transformer Vision Backbone

## 1. 核心动机与问题设定

这篇paper来自NVIDIA的Ali Hatamizadeh和Jan Kautz，本质上回答一个问题：**Mamba这种为NLP设计的SSM架构，到底能不能直接搬到vision里用？答案是no，但是hybrid可以work得更好**。

### 1.1 Mamba在vision上的根本问题

Mamba原本设计是autoregressive的，也就是token-by-token处理。对language这是天然的——语言有sequential order。但是image pixels并不具备这种sequential dependency，spatial relationships更多是local + parallel的。

具体两个pitfall:
1. **Inefficiency for spatial data**: pixel之间没有强烈的sequential依赖，autoregressive处理浪费了并行性
2. **Limited global context in single pass**: autoregressive model必须逐步处理，无法一次性看到全局

之前的solutions（Vim的bidirectional SSM, VMamba的Cross-Scan Module）都试图通过multi-directional scan来fix这个问题，但代价是**显著增加latency**——必须等整个序列处理完才能预测。

### 1.2 作者的核心insight

与其在Mamba内部想办法模拟global context，**不如直接用Transformer的self-attention来补这个短板**。关键发现是：self-attention放在**最后几层**最有效。这个insight非常符合直觉——early layers学local features（CNN/Mamba擅长），late layers学global reasoning（attention擅长）。

参考文献:
- Mamba原paper: https://arxiv.org/abs/2312.00752
- Vim: https://arxiv.org/abs/2401.09417
- VMamba: https://arxiv.org/abs/2401.10166
- Swin Transformer: https://arxiv.org/abs/2103.14030
- ConvNeXt: https://arxiv.org/abs/2201.03545

---

## 2. Mamba Preliminaries 深度解析

要理解MambaVision，必须先吃透原版Mamba的数学结构。

### 2.1 Continuous-time SSM (Eq. 2)

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$

变量含义：
- $t$: continuous time variable
- $x(t) \in \mathbb{R}$: scalar input signal at time $t$
- $h(t) \in \mathbb{R}^M$: hidden state vector, $M$是state dimension（hyperparameter，paper里d_state=16）
- $A \in \mathbb{R}^{M \times M}$: state transition matrix，控制hidden state如何evolve
- $B \in \mathbb{R}^{M \times 1}$: input projection matrix，把scalar input投影到state space
- $C \in \mathbb{R}^{1 \times M}$: output projection matrix，把hidden state读出来变成output
- $y(t) \in \mathbb{R}$: scalar output

Intuition: 这是一个linear ODE。$A$决定了系统的dynamics——是衰减、振荡还是发散。$B$和$C$分别控制input和output的接口。

### 2.2 Discretization (Eq. 3)

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot (\Delta B), \quad \bar{C} = C$$

变量：
- $\Delta$: timescale parameter，连续到离散的step size，**learnable**，这是Mamba的selectivity来源之一
- $\bar{A}, \bar{B}, \bar{C}$: 离散化后的参数
- $I$: identity matrix
- $\exp(\cdot)$: matrix exponential

Intuition: Zero-Order Hold (ZOH) discretization。$\Delta$大意味着"看得远"（更多历史信息混合），$\Delta$小意味着"看得近"（更关注当前）。**这个$\Delta$是input-dependent的，所以model能selectively关注不同时间尺度**。

### 2.3 Discrete recurrence (Eq. 4)

$$h(t) = \bar{A}h(t-1) + \bar{B}x(t), \quad y(t) = \bar{C}h(t)$$

现在$h(t)$是离散hidden state，$t$是discrete time step。

### 2.4 Global Convolution view (Eq. 5)

$$\bar{K} = (\bar{C}\bar{B}, \bar{C}\bar{A}\bar{B}, ..., \bar{C}\bar{A}^{T-1}\bar{B}), \quad y = x * \bar{K}$$

变量：
- $\bar{K}$: 卷积kernel，长度为$T$
- $T$: sequence length
- $*$: convolution operator
- $\bar{A}^{T-1}$: $\bar{A}$的$(T-1)$次幂

Intuition: Linear recurrence可以展开成convolution！这意味着可以parallelize训练（FFT convolution）。但是注意——这是**causal** convolution，因为$\bar{K}$的第$k$个元素只依赖$x(t), x(t-1), ..., x(t-k+1)$。这就是为什么Mamba是autoregressive的根因。

### 2.5 Selectivity机制

原版Mamba让$B, C, \Delta$变成input-dependent：
$$B = \text{Linear}(x), \quad C = \text{Linear}(x), \quad \Delta = \text{softplus}(\text{Linear}(x))$$

这允许model "selectively" attend to different parts of input，类似attention的灵活性但是linear complexity。

参考: S4 paper (https://arxiv.org/abs/2111.00396), S6/Mamba (https://arxiv.org/abs/2312.00752)

---

## 3. MambaVision架构深度解析

### 3.1 Macro Architecture (Figure 2)

Hierarchical 4-stage design，类似Swin/ConvNeXt：

```
Input: H×W×3
    ↓ Stem (2× Conv3x3 stride 2)
Stage 1: H/4 × W/4, CNN ResBlocks (高分辨率)
    ↓ Downsample (Conv3x3 stride 2)
Stage 2: H/8 × W/8, CNN ResBlocks
    ↓ Downsample
Stage 3: H/16 × W/16, [MambaVision × N/2] + [Self-Attention × N/2]
    ↓ Downsample
Stage 4: H/32 × W/32, [MambaVision × N/2] + [Self-Attention × N/2]
```

**关键设计选择**：
- Stage 1, 2用CNN：高分辨率阶段，CNN在spatial locality上更高效，避免token数量爆炸
- Stage 3, 4用hybrid：低分辨率阶段，token数量少，可以afford SSM/attention的全局建模

这与EfficientVMamba的策略**相反**——EfficientVMamba用SSM处理高分辨率、CNN处理低分辨率。MambaVision的approach更好，因为high-resolution阶段用CNN避免了SSM的sequence length问题。

### 3.2 CNN ResBlock (Eq. 1)

$$\hat{z} = \text{GELU}(\text{BN}(\text{Conv}_{3×3}(z))), \quad z = \text{BN}(\text{Conv}_{3×3}(\hat{z})) + z$$

变量：
- $z$: input feature
- $\hat{z}$: intermediate feature after first conv
- BN: Batch Normalization
- GELU: Gaussian Error Linear Unit activation

标准的ResNet bottleneck design，没什么fancy。

### 3.3 Layer Architecture (Eq. 6)

$$\hat{X}^n = \text{Mixer}(\text{Norm}(X^{n-1})) + X^{n-1}, \quad X^n = \text{MLP}(\text{Norm}(\hat{X}^n)) + \hat{X}^n$$

变量：
- $X^{n-1}$: input to layer $n$
- $\hat{X}^n$: output after token mixing
- $X^n$: final output after MLP
- Norm: LayerNorm
- Mixer: either MambaVision mixer or Self-Attention

Pre-norm design + residual connection，标准ViT pattern。

### 3.4 MambaVision Mixer 核心创新 (Eq. 7, Figure 3)

这是paper最核心的贡献。原版Mamba mixer被re-design：

$$X_1 = \text{Scan}(\sigma(\text{Conv}(\text{Linear}(C, C/2)(X_{in}))))$$
$$X_2 = \sigma(\text{Conv}(\text{Linear}(C, C/2)(X_{in})))$$
$$X_{out} = \text{Linear}(C/2, C)(\text{Concat}(X_1, X_2))$$

变量：
- $X_{in} \in \mathbb{R}^{T \times C}$: input，$T$是sequence length，$C$是embedding dimension
- $X_1$: SSM branch output, $\mathbb{R}^{T \times C/2}$
- $X_2$: non-SSM symmetric branch output, $\mathbb{R}^{T \times C/2}$
- $\sigma$: SiLU activation
- Conv: 1D depthwise convolution
- Scan: selective scan operation (Mamba的核心)
- Concat: channel-wise concatenation

**三个关键改动**：

#### 改动1: 去掉causal convolution，换成regular convolution
原版Mamba用causal conv是为了保持autoregressive property。但vision不需要sequential constraint，regular conv让信息双向流动。这是**根本性的vision-friendly改造**。

#### 改动2: 加symmetric branch (no SSM)
这个branch只有Conv + SiLU，没有SSM。它的作用是**compensate SSM丢失的spatial信息**。SSM本质是sequential的，会损失spatial structure；这个parallel branch保留了原始spatial features。

#### 改动3: 用Concat替代gating
原版Mamba用$y = z \cdot \text{SiLU}(y_{ssm})$这种multiplicative gating。MambaVision改成concat + linear projection。

Intuition: gating是**information bottleneck**——如果一个branch输出接近0，整个输出就没了。Concat保留了两个branch的完整信息，让下游layer自己决定如何混合。

### 3.5 Algorithm 1 PyTorch Pseudocode分析

```python
class MambaVisionMixer(nn.Module):
    def __init__(self, dim, d_state=16, kernel_size=3):
        self.dt_rank = math.ceil(dim / 16)  # Δ的低秩投影维度
        self.in_proj = nn.Linear(dim, dim)  # 实际是dim → dim（不是dim → dim*2）
        # 注意：这里代码有subtlety，xz chunk(2)后是dim/2每个
        self.x_proj = nn.Linear(dim//2, self.dt_rank + self.d_state * 2)
        # x_proj输出: Δ (dt_rank) + B (d_state) + C (d_state)
        self.conv1d_x = nn.Conv1d(dim//2, dim//2, kernel_size, 
                                  padding='same', groups=dim//2)  # depthwise
        self.conv1d_z = nn.Conv1d(dim//2, dim//2, kernel_size, 
                                  padding='same', groups=dim//2)
        self.dt_proj = nn.Linear(self.dt_rank, dim//2)
        # A是log-space参数化，保证负值（衰减系统）
        A_log = torch.log(repeat(torch.arange(1, d_state+1), ...))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(dim//2))  # skip connection
```

Forward pass的关键步骤：
1. `in_proj`: $X_{in} \to [x; z]$，split成两个half
2. 两个branch各自做conv + activation
3. SSM branch: `x_proj`生成input-dependent的$\Delta, B, C$
4. `selective_scan_fn`: 实际的SSM forward，CUDA kernel实现
5. Concat两个branch + out_proj

**代码细节注意**：
- `groups=dim//2`: depthwise conv，每个channel独立conv
- `A = -torch.exp(self.A_log)`: 保证$A$是负的（系统stable，decay）
- `D`: residual skip connection，类似S4的D parameter

---

## 4. Hybrid Pattern的关键发现

Table 5的ablation study是paper最有价值的部分之一：

| Pattern | Layout | Top-1 |
|---------|--------|-------|
| Random | 随机 | 81.3% |
| First N/2 SA | SSSSMMMM | 81.5% |
| Mixed-1 | SMSMSMSM | 81.4% |
| Mixed-2 | MSMSMSMS | 81.6% |
| Last N/4 SA | MMMMMMSS | 81.9% |
| **Last N/2 SA** | **MMMMSSSS** | **82.3%** |

**核心insight**: Self-attention放在**最后**效果最好，放在前面或交替都更差。

为什么？我的intuition：
1. **Early layers需要local feature extraction**: SSM/Mamba的inductive bias适合学local patterns
2. **Late layers需要global reasoning**: 当spatial resolution已经downsampled很多（14×14或7×7），token数量少，self-attention的quadratic cost可以接受
3. **Information flow**: SSM先把local features aggregate好，self-attention再在已经compressed的representation上做global mixing

这跟FasterViT的Hierarchical Attention、Swin的shifted window思路有异曲同工之妙——都是"early cheap, late expensive"。

参考: FasterViT (https://arxiv.org/abs/2306.06189)

---

## 5. Token Mixer设计的Ablation (Table 4)

| Config | ImageNet | COCO AP^box | ADE20K mIoU |
|--------|----------|-------------|-------------|
| causal conv1, w/o conv2 | 80.5% | 44.8 | 44.2 |
| regular conv1, w/o conv2 | 80.9% | 45.0 | 44.7 |
| conv1 + conv2, w/o concat (gating) | 81.3% | 45.3 | 45.7 |
| **conv1 + conv2 + concat** | **82.3%** | **46.4** | **46.0** |

四个步骤，每一步都有意义：
1. **Causal → Regular conv**: +0.4% Top-1，vision不需要causal constraint
2. **Add symmetric branch**: +0.4% Top-1，compensate SSM信息损失
3. **Gating → Concat**: +1.0% Top-1，避免information bottleneck

这是一个非常clean的incremental design study，每一步都justified by实验数据。

---

## 6. 实验结果深度分析

### 6.1 ImageNet-1K Classification (Table 1)

Pareto front分析（accuracy vs throughput）：

| Model | Params (M) | FLOPs (G) | Throughput (img/s) | Top-1 |
|-------|-----------|-----------|---------------------|-------|
| MambaVision-T | 31.8 | 4.4 | 6298 | 82.3% |
| MambaVision-B | 97.7 | 15.0 | 3670 | 84.2% |
| MambaVision-L | 227.9 | 34.9 | 2190 | 85.0% |
| VMamba-B | 89.0 | 15.4 | 645 | 83.9% |
| Swin-B | 87.9 | 15.1 | 535 | 84.6% |
| ConvNeXt-B | 88.6 | 15.4 | 1485 | 83.8% |

**惊人发现**：
- MambaVision-B vs VMamba-B: accuracy +0.3%, throughput **5.7×** faster (3670 vs 645)
- MambaVision-B vs Swin-B: accuracy -0.4%, throughput **6.9×** faster
- MambaVision-B vs ConvNeXt-B: accuracy +0.4%, throughput **2.5×** faster

为什么MambaVision这么快？几个原因：
1. Hybrid design: SSM是linear complexity，只有最后几层用attention
2. CNN处理高分辨率: 避免了ViT在early stages的O(N²)问题
3. Single forward pass: 不像Vim的bidirectional需要两次scan
4. Hardware-aware: SSM的CUDA kernel optimized for A100

### 6.2 Object Detection on COCO (Table 2)

| Backbone | AP^box | AP^mask |
|----------|--------|---------|
| Swin-T | 50.4 | 43.7 |
| ConvNeXt-T | 50.4 | 43.7 |
| **MambaVision-T** | **51.1** | **44.3** |
| Swin-B | 51.9 | 45.0 |
| ConvNeXt-B | 52.7 | 45.6 |
| **MambaVision-B** | **52.8** | **45.7** |

Downstream tasks的improvement比classification更明显，说明MambaVision学到的features更generalizable。这可能是因为hybrid design同时保留了local（CNN）和global（SSM+Attention）的信息。

### 6.3 Semantic Segmentation on ADE20K (Table 3)

| Backbone | mIoU |
|----------|------|
| Swin-T | 44.5 |
| **MambaVision-T** | **46.0** (+1.5) |
| Swin-B | 48.1 |
| **MambaVision-B** | **49.1** (+1.0) |

Segmentation对global context要求高，MambaVision的self-attention在最后阶段帮助capture long-range dependencies。

### 6.4 ImageNet-21K Pretraining Scaling (Figure 4)

| Model | 224px Top-1 | 256px | 512px |
|-------|-------------|-------|-------|
| MambaVision-B (1K) | 84.2% | - | - |
| MambaVision-B (21K) | 84.9% | - | - |
| MambaVision-L (1K) | 85.0% | - | - |
| MambaVision-L (21K) | 86.1% | - | - |
| MambaVision-L3 (21K) | - | 87.3% | 88.1% |

**这是第一个成功scale到ImageNet-21K的Mamba-based vision model**。之前的Vim/VMamba都没有report 21K results。739.6M参数的L3在512分辨率达到88.1% Top-1，接近SOTA。

参考: ImageNet-21K (https://www.image-net.org/)

---

## 7. Window Size Ablation (Table S.1)

| Window Size (Stage3, Stage4) | Throughput | Top-1 | AP^box |
|------------------------------|------------|-------|--------|
| 7, 7 | 6318 | 82.2% | 46.4 |
| **14, 7** | 6298 | **82.3%** | 46.4 |

Stage 3用window size 14（即覆盖整个14×14 feature map），Stage 4用7（覆盖整个7×7）。Throughput只损失0.3%，accuracy微升。

Intuition: Stage 3的14×14 feature map如果用window 7，每个window只能看到一半的spatial extent；用window 14就能看到全局。Stage 4的7×7本身就在一个window内，所以7就够。

---

## 8. Attention Map可视化 (Figure 5, S.1)

可视化显示self-attention确实学到了semantic regions：
- Aircraft: attention覆盖整个plane body
- Bird: 集中在head和tail（fine-grained features）
- Human-object interaction: 同时关注subject和object

这validate了"self-attention在最后阶段capture global context"的设计假设。

---

## 9. 与其他架构的深度对比

### 9.1 vs Vim (Bidirectional SSM)
- Vim: 两个direction各scan一次，2× compute
- MambaVision: single forward + self-attention补全局，faster且better

### 9.2 vs VMamba (Cross-Scan Module)
- VMamba: 4-way scan (左上→右下, 右下→左上, 左下→右上, 右上→左下)，4× compute
- MambaVision: regular conv (双向) + self-attention，更efficient

### 9.3 vs Swin Transformer
- Swin: 全程window attention + shifted window
- MambaVision: early CNN + middle SSM + late attention，分工更明确

### 9.4 vs ConvNeXt
- ConvNeXt: 纯CNN，large kernel但仍是local
- MambaVision: hybrid，有global receptive field

### 9.5 vs FasterViT
- FasterViT: hierarchical attention，CNN + window attention + global attention
- MambaVision: 用SSM替代部分attention，更efficient

参考: 
- Vim: https://arxiv.org/abs/2401.09417
- VMamba: https://arxiv.org/abs/2401.10166
- Swin: https://arxiv.org/abs/2103.14030
- FasterViT: https://arxiv.org/abs/2306.06189

---

## 10. 局限性与未来方向

虽然paper没有explicitly讨论limitation，但可以推断：

1. **SSM的causal assumption依然存在**: 虽然用regular conv缓解了，但scan本身还是ordered的。Image的"natural order"（raster scan）并不semantic。

2. **Hybrid design的complexity**: 两种不同的mixer增加了实现复杂度，deployment可能需要不同的CUDA kernels。

3. **Window attention的限制**: Stage 3/4用windowed attention，如果输入分辨率大于window，依然只能看到local。虽然比pure ViT好，但不如global attention灵活。

4. **Pre-training scaling**: 虽然scale到21K，但相比ViT在更大dataset（如JFT-300M, LAION-5B）上的scaling，MambaVision还没验证。

5. **3D vision tasks**: Paper只验证了2D vision，3D medical imaging (如 segmentation)或video tasks的efficacy未知。

未来方向可能包括：
- Video MambaVision: temporal dimension用SSM天然适合
- Multi-modal MambaVision: vision-language，SSM处理长文本
- MambaVision + Diffusion: 作为diffusion model的backbone

---

## 11. 总结：Build Intuition

MambaVision的核心takeaway：

**Architecture design的本质是"right tool for right job"**:
- High-resolution, local features → CNN (cheap, inductive bias强)
- Medium-resolution, sequential patterns → SSM (linear complexity, captures dependencies)
- Low-resolution, global reasoning → Self-attention (quadratic but powerful, token数少)

**Hybrid的关键是placement**: 不是random混合，而是**ordered by abstraction level**。Early layers学concrete features用cheap modules，late layers学abstract reasoning用expensive modules。

**SSM在vision的问题不是complexity，是inductive bias**: Mamba的causal/sequential假设对vision是错的。Fix方法要么改SSM本身（Vim/VMamba），要么用其他module补足（MambaVision的approach）。后者更简单且更effective。

**Information preservation**: Concat > gating。Gating是lossy的，concat是lossless的。让下游layer决定如何混合信息，比upstream hard decision好。

这些insights不仅适用于Mamba，对任何hybrid architecture design都有指导意义。

---

## References

- MambaVision paper: https://arxiv.org/abs/2407.08083
- GitHub: https://github.com/NVlabs/MambaVision
- Mamba: https://arxiv.org/abs/2312.00752
- S4: https://arxiv.org/abs/2111.00396
- Vim: https://arxiv.org/abs/2401.09417
- VMamba: https://arxiv.org/abs/2401.10166
- Swin Transformer: https://arxiv.org/abs/2103.14030
- ConvNeXt: https://arxiv.org/abs/2201.03545
- ViT: https://arxiv.org/abs/2010.11929
- FasterViT: https://arxiv.org/abs/2306.06189
- ImageNet-1K: https://www.image-net.org/
- MS COCO: https://cocodataset.org/
- ADE20K: https://groups.csail.mit.edu/vision/datasets/ADE20K/
- Cascade Mask R-CNN: https://arxiv.org/abs/1906.09756
- UperNet: https://arxiv.org/abs/1807.10221
