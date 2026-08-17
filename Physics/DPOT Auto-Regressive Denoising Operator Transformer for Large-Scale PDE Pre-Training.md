---
source_pdf: DPOT Auto-Regressive Denoising Operator Transformer for Large-Scale PDE
  Pre-Training.pdf
paper_sha256: 8eb0ffbf131613e0c1d0770de2d8d1095ae16412af20822e26167460f62845b4
processed_at: '2026-08-03T23:13:21-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 DPOT

---

## 这论文在干啥

想象你是个天气预报员，但你想偷懒——不写复杂的物理方程,而是让 AI 看一堆历史数据学会自己推未来天气。这叫 "neural operator",就是把 PDE 解算子用神经网络替代掉。

问题来了:训一个能算流体、热扩散、浅水方程的 AI 模型,每种都要好几千条模拟数据,每条数据要跑传统求解器好几个小时甚至好几天才能生成。数据贵得要命。

NLP 和 CV 早就给出答案了:pre-training。先在巨量数据上学个通用的,再 fine-tune 到具体任务。GPT 不就是这么来的吗?

但这事在 PDE 上没人做成过。DPOT 就是第一个做到差不多规模的——10+ 数据集,100k+ trajectories,最大 1B 参数,跨各种 PDE 任务都 work。

---

## 为什么之前没人做成

PDE 数据集长得五花八门,根本不像图片那么统一:

- 有的 2D 有的 3D
- 有的是 32×32 分辨率,有的是 1024×1024
- 有的只有速度场一个量,有的有速度+压力+密度三五个量
- 有的是规则网格,有的是奇怪的不规则 mesh
- 有的跑 10 步就结束,有的跑 500 步
- 数值范围差好几个数量级

你没法像 CV 那样统一 resize 成 224×224 就喂进 ViT。

更糟的是,自回归训 PDE 有个臭名昭著的问题:训练时给 clean input,但推理时模型自己输出的下一帧会带误差,再喂回去当 input,误差越滚越大。这叫 exposure bias,NLP 里早就知道,PDE 里更严重——因为物理系统对初始扰动敏感。

之前的 MPP 工作用 vision transformer 自回归预训练,但 long trajectory 不稳,也没法跨 shape 迁移。

---

## DPOT 怎么搞定的

三个核心 trick:

### Trick 1: 训练时给 input 加噪声

这个简单到让人怀疑人生。他们就是在自回归训练时给输入加一点 Gaussian noise,让模型在训练时见到"已经扰动的状态",强迫它学会从 perturbed state 恢复到 clean next state。

加多少噪声?ablation 显示 $\epsilon = 5 \times 10^{-5}$ 是甜点。不加噪声 test error 0.0178,加一点降到 0.0152。加太多(5e-2)就废了,test error 飙到 0.753。

这本质是个 regularization。exposure bias 问题的根源是"训练见 clean、推理见 noisy",加噪声直接消除了这个 distribution gap。

为什么比 pushforward trick 简单:pushforward 要 multi-step rollout 才能暴露累积误差,计算开销大。加噪声只要改一行代码。

### Trick 2: Fourier Attention——不用标准 attention,在频率域做 MLP

标准 self-attention 是 $O(N^2)$,对 PDE 这种 128×128 网格的 16k 个 token 根本算不动。

他们的做法:把 feature 做 FFT 变到频率域,在频率域过一个小 MLP(2 层),再 IFFT 回来。

为什么这 work:PDE 的解算子在均匀介质上 translation equivariant,数学上等价于在 Fourier domain 做对角化(element-wise 乘法)。直接在频率域学一个非线性变换,就是学了 PDE 的 spectral response。

这跟 FNO 的核心 insight 是一致的——但 DPOT 用 MLP 替代了 per-mode 的 $R_\phi(k)$,参数量从 $m \cdot d_z^2$ 降到 $2 d_z^2$。所有 frequency mode 共享同一组权重,相当于一个 "frequency-equivariant" 设计。

跟 AFNO 的关键区别:AFNO 在 MLP 后加 soft-thresholding 强行稀疏化。DPOT 去掉了——因为 PDE 解有多尺度 spectral 结构,turbulence 的能量谱是 $k^{-5/3}$ power law,强行稀疏化就把小尺度但能量可观的 mode 砍掉了。Table 8 显示去掉 sparsity 让 test error 从 0.121 降到 0.0174,差 7 倍。

### Trick 3: 时间维用 Fourier basis 聚合

PDE 的时间演化本质是 $e^{\lambda t}$ 形式——衰减或振荡。用 $e^{-i\gamma t}$ 这种复数 Fourier basis 把 T 个时间步的 embedding 加权求和,天然契合 PDE 的时间结构。模型能学习 $\gamma$ 来捕捉每个 PDE 的固有时间频率。

这是 DPOT 比 AFNO 多出来的关键设计——AFNO 是给单帧图片用的,没有时间维。DPOT 要处理 trajectory,得有专门的时间编码。

---

## 还有一些工程细节

**数据统一**:所有数据集 resize 到 128×128,channel padding 到最多通道数,irregular geometry 加个 mask channel。简单粗暴但 work。

**Balanced sampling**:小数据集采样概率反比于其大小。否则大数据集 dominate 梯度,小数据集根本学不到。Appendix 显示 equal sampling 让 FNO-1e-5 的 L2RE 是 0.177,balanced 降到 0.0976,差一倍。

**Multi-head**:在 channel 维切分,每个 head 用独立小 MLP。8 heads 最佳。跟 multi-head attention 一个思路——不同 head 学不同 frequency subspace。

---

## 效果

**主表 12 个数据集**:DPOT 在 9 个上 SOTA。相比 FNO-m 在 FNO-1e-4 上从 0.0922 降到 0.0442,降了 52%。

**Zero-shot**:7M 模型就比 30M 的 MPP AViT 强,说明 Fourier attention 比 vision transformer 参数效率高。

**Scaling**:7M → 500M,L2RE 从 0.06 降到 0.028。1B 模型又大幅提升。Scaling law 在 PDE 上也成立。

**Downstream 迁移**:

- 2D pre-train 拿去 fine-tune 3D NS:L2RE 从 FNO 的 0.410 降到 0.226。这很惊艳——因为 Fourier attention 的权重跟空间维度解耦,2D FFT 换成 3D FFT 就行,CNN 就做不到。
- Pre-train 拿去 fine-tune steady-state CNO:0.0357 → 0.0230。
- Long trajectory 500 步:vanilla DPOT 误差 0.0912,pre-trained 0.0385。时间越长 pre-training 优势越大。
- Kolmogorov turbulence 这种 chaotic 系统:FNO full trajectory error 104%(完全没学会),DPOT-pretrained 33.5%。Pre-training 让模型学到 "chaotic 演化的通用 pattern"。
- Super-resolution 128→1024:超过 EDSR 这种专门做超分的 SOTA。

---

## 一句话总结

他们用"加噪声自回归 + Fourier 域 MLP attention + 时间维 Fourier 聚合 + balanced sampling"这套组合拳,把 PDE pre-training 第一次推到 1B 规模,证明 PDE 数据也有 scaling law,pre-trained 权重能迁移到 3D、steady-state、chaotic、super-resolution 等各种没见过的任务上。

核心 takeaway:PDE 解算子的 spectral structure 是个极强的 inductive bias,在 Fourier domain 做非线性变换既高效又表达力够。加噪声自回归是解决 exposure bias 的最便宜方案。Foundation model 范式在 PDE 上也能复现 NLP/CV 的成功。

---

# DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training 深度解析

这篇论文做了一件 NLP/CV 早已做过、但 scientific ML 一直没做成的事：**用 1B 参数的 transformer 在 10+ PDE datasets（100k+ trajectories）上 pre-train 一个 foundation model，迁移到各种 downstream PDE 任务时获得 SOTA**。让我从 motivation、method、experiment、theory 四个层面把直觉建立起来。

---

## 1. Motivation: 为什么 PDE pre-training 这么难

### 1.1 PDE 数据的"丑陋多样性"

NLP 有 token vocabulary 统一，CV 有 224×224 这样的标准 input shape。PDE 数据集合集 $\mathcal{D} = \cup_{k=1}^{K} \mathcal{D}_k$ 每个子集都长得不一样：

- **维度**: 1D / 2D / 3D 空间
- **时间长度**: trajectory $T$ 从几十到几千
- **分辨率**: 32×32 到 1024×1024
- **通道数**: 单变量 vorticity 到多变量
- **几何**: regular grid / irregular mesh / point cloud
- **数值范围**: 不同 PDE 的物理量量级差好几个数量级

任何想用单一 transformer 处理这些数据的设计，都必须解决 shape unification + capacity 两个问题。

### 1.2 既有 pre-training 工作的局限

- **Mialon et al. 2023**: contrastive learning，需要专门设计 augmentation（PDE 的 symmetry 不好定义）
- **MPP (McCabe et al. 2023)**: 用 vision transformer auto-regressive 在 PDEBench 上预训练。问题在 long trajectory 不稳定 + 无法迁移到不同 shape
- **Subramanian et al. 2023**: 仅限 3 个简单 steady-state PDE

DPOT 想做的就是把这些局限全部突破。

---

## 2. Method 详解

### 2.1 PDE 的一般形式（公式 1）

$$\frac{\partial \pmb{u}}{\partial t} - \mathcal{F}[\pmb{u};\theta](\pmb{x},t) = 0, \quad (\pmb{x},t) \in \Omega \times T$$

变量含义：
- $\pmb{u}(\pmb{x},t) \in \mathbb{R}^m$：解向量，m 是物理通道数（速度、压力、密度等）
- $\Omega \subset \mathbb{R}^d$：空间域，d 是空间维度
- $\mathcal{F}[\pmb{u};\theta]$：微分算子，里面包含 $\partial_x \pmb{u}, \partial_{xx} \pmb{u}$ 等空间导数项
- $\theta \in \Theta$：决定 PDE 类型和系数的参数（如 viscosity $\nu$）
- $\pmb{u}^0(\pmb{x})$：初始条件
- $\mathcal{B}[\pmb{u}]$：边界条件

**关键 insight**：实际场景（气候、航空）我们只有 trajectory 数据 $\{\pmb{u}^1, \dots, \pmb{u}^T\}$，θ 是 implicit 的。模型必须从相邻 T 帧反推 θ 然后外推下一帧。

### 2.2 Auto-Regressive Denoising Pre-training（核心创新之一）

#### 基础形式

神经算子 $\mathcal{G}_w$ 自回归地预测下一帧：

$$\pmb{u}^T = \mathcal{G}_w(\pmb{u}^0, \dots, \pmb{u}^{T-1})$$

直接监督 one-step loss 在测试时累积误差大。NLP 里叫 exposure bias（Bengio 2015），PDE 里 Brandstetter 2022 用 pushforward trick 解决但训练复杂度高，不适合 pre-training。

#### Denoising 注入（公式 3）

$$\min_w \mathcal{L} = \mathbb{E}_{\pmb{u} \sim p(\mathcal{D})} \sum_{1 \le t \le T} \|\mathcal{G}_w(\pmb{u}^{<t} + \pmb{\varepsilon}) - \pmb{u}^t\|_2^2$$

其中噪声 $\pmb{\varepsilon} \sim \mathcal{N}(0, \epsilon \|\pmb{u}^{<t}\| \pmb{I})$。

**变量解析**：
- $\epsilon$：控制噪声强度的超参（ablation 显示甜点在 5e-5）
- 噪声 std 与 $\|\pmb{u}^{<t}\|$ 成比例，意味着对量级大的物理量加更多噪声，对量级小的加更少，保持 SNR 一致

**为什么这个 work（intuition）**：

训练时 input 是 clean 的，但 inference 时模型自回归输出会累积误差，下一帧的 input 实际是 "clean + perturbation"。注入 Gaussian noise 等于让模型在训练时看到 "已经被扰动的 input"，强迫它学习从 noisy state 恢复到 clean next state。

这本质上是一种 data augmentation + regularization，但比 pushforward trick 简单太多——只需要在 input 上加噪声即可，无需 multi-step rollout 训练。

### 2.3 Data Preprocessing 和 Balanced Sampling

#### Shape unification

- 固定 resolution $H = 128$：低分辨率 interpolation 上采样，高分辨率 random crop / interpolation 下采样
- Channel padding：所有数据集 pad 到最大 channel 数（用 1 填充）
- Irregular geometry：额外引入一个 mask channel 编码几何

#### Balanced sampling（公式 4）

$$p_k = \frac{w_k}{K |\mathcal{D}_k| \cdot \sum_k w_k}$$

**变量**：
- $w_k$：第 k 个数据集的重要性 weight（long trajectory 数据集如 PDB-DR、PDB-SWE 设为 3，其余为 1）
- $|\mathcal{D}_k|$：第 k 个数据集大小
- $K$：数据集总数
- $p_k$：从第 k 个数据集采样的概率

**关键**：$p_k$ 反比于 $|\mathcal{D}_k|$。这意味着小数据集被 oversample，避免大数据集主导梯度。Appendix B.8 对比显示，equal sampling 会让 FNO-1e-5、PDB-DR、PDB-SWE 等收敛极差（FNO-1e-5 L2RE 0.177 vs balanced 0.0976）。

### 2.4 Model Architecture

整体 pipeline：patchify → temporal aggregation → Fourier attention layers × L → output

#### 2.4.1 Patchification 和 Positional Encoding（公式 5）

$$Z_p^t = \mathcal{P}(\pmb{u}^t + \pmb{p}^t), \quad t = 1, \dots, T$$

- $\mathcal{P}$：convolutional layer，stride = patch size $P$
- $\pmb{p}^t$：learnable positional encoding
- $W_p \in \mathbb{R}^{n \times 3}$：positional encoding 参数，n 是 mode 数，3 对应 (x, y, t) 三个坐标

#### 2.4.2 Temporal Aggregation（公式 6）—— 这是 DPOT 区别于 AFNO 的关键

$$z_{\text{agg}} = \sum_t W_t \cdot z_p^t \cdot e^{-i\gamma t}$$

**变量**：
- $W_t$：每个时间步的 learnable transform
- $\gamma \in \mathbb{R}^C$：可学习 Fourier features（频率）
- $e^{-i\gamma t}$：复数 Fourier basis

**直觉**：把 T 个时间步的 embedding 用复数 Fourier basis 加权求和。复数 $e^{-i\gamma t} = \cos(\gamma t) - i\sin(\gamma t)$ 提供 (cos, sin) 两个正交分量，相当于在时间维度做 spectral decomposition。

PDE 的时间演化本质是 $u(t) = e^{\lambda t}$ 形式的指数衰减/振荡，用 Fourier basis 编码时间天然契合。模型可以学习 $\gamma$ 来捕捉 PDE 的固有时间频率。

#### 2.4.3 Fourier Attention Layer（核心架构创新）

##### 一般 kernel integral form（公式 7）

$$(\mathcal{K}_\phi z^l)(\pmb{x}) = \int_\Omega \kappa(\pmb{x}, \pmb{y}; \phi) z^l(\pmb{y}) \, d\pmb{y}$$

这是 attention 的连续形式，$\kappa(\pmb{x}, \pmb{y}; \phi)$ 是 neural network 参数化的 kernel。Quadratic complexity 不实用。

##### Translation-invariant kernel（公式 8）

假设 $\kappa(\pmb{x}, \pmb{y}; \phi) = \kappa(\pmb{x} - \pmb{y}; \phi)$，这就变成 global convolution：

$$(\mathcal{K}_\phi z^l)(\pmb{x}) = \mathcal{F}^{-1}[R_\phi \cdot \mathcal{F}[z^l]]$$

- $R_\phi(k) \in \mathbb{C}^{d_z \times d_z}$：frequency domain 上的 learnable 权重
- 保留 m 个 modes 时：$R_\phi \in \mathbb{C}^{m \times d_z \times d_z}$，参数量和 m × d_z² 成正比，内存爆炸

**为什么 translation-invariant 是合理 prior**：很多 PDE（Navier-Stokes、diffusion、wave）的解算子本身在平移不变的几何上就是 translation equivariant。在 Fourier domain 上，translation equivariant 算子退化为 element-wise 乘法（对角化）。

##### Weight-sharing MLP approximation（公式 9 和 10）

$$\hat{z}(k) = W_2 \cdot \sigma(W_1 \cdot \mathcal{F}[z^l](k) + b_1) + b_2$$

$$z^{l+1}(\pmb{x}) = \mathcal{F}^{-1}[W_2 \cdot \sigma(W_1 \cdot \mathcal{F}[z^l] + b_1) + b_2](\pmb{x})$$

**变量**：
- $W_1, W_2 \in \mathbb{R}^{d_z \times d_z}$：在所有 frequency modes 之间共享的权重
- $b_1, b_2 \in \mathbb{R}^{d_v}$：biases
- $\sigma$：activation（GELU）

参数量从 $m \cdot d_z^2$ 降到 $2 d_z^2$，但仍然 expressivity 足够（见 Theorem 3.1）。

**与 AFNO 的关键区别（Appendix B.3）**：AFNO 在 MLP 后加 soft-thresholding 操作做 sparsity，DPOT 去掉了。理由是 PDE 解往往有多尺度 spectral 特性，强制 sparse 会损失信息。Table 8 显示：AFNO train loss 0.0664、test 0.121 vs DPOT train 0.0127、test 0.0174。差距巨大。

#### 2.4.4 Multi-head Structure（公式 11）

$$z_i^{l+1}(\pmb{x}) = \mathcal{F}^{-1}[W_{2,i} \cdot \sigma(W_{1,i} \cdot \mathcal{F}[z_i^l] + b_{1,i}) + b_{2,i}](\pmb{x})$$

- $z^l = \text{Concat}(z_1^l, z_2^l, \dots, z_h^l)$，沿 channel 切分
- 每个 head $i$ 有独立小 MLP，维度 $d_z / h$
- 计算 $O(d_z^2 / h)$，fully parallelizable

不同 head 可以专注于不同 frequency subspace，类似 multi-head attention 的"不同 representation subspace"理念。

#### 2.4.5 完整 Layer 结构

每层包含：
1. Fourier attention (上面公式 11)
2. Group normalization（Wu & He 2018）
3. Feedforward network（channel-wise MLP）
4. Residual connection

---

### 2.5 Theoretical Analysis: Universal Approximation

**Theorem 3.1**：对于 Sobolev 空间之间的连续算子 $\mathcal{G}: H^s(\mathbb{T}^d; \mathbb{R}^{d_{in}}) \to H^{s'}(\mathbb{T}^d; \mathbb{R}^{d_{out}})$ 和紧集 $K \subset H^s$，对任意 $\varepsilon > 0$，存在 Fourier attention layers $N$ 使得：

$$\sup_{v \in K} \|\mathcal{G}(v) - \mathcal{N}(v)\|_{L^2} \le \varepsilon$$

**证明思路（Appendix C）三步**：

**Step 1**: 构造 equivariant proxy target（公式 35）

$$f([(x_1, \pi(1)), \dots, (x_N, \pi(N))]) := \pi^{-1} \circ \mathcal{G}([x_{\pi^{-1}(1)}, \dots, x_{\pi^{-1}(N)}])$$

把 permutation $\pi$ concat 到 input，输出再 inverse permutation。这样 $f$ 对 input 顺序 equivariant（Lemma C.8）。

**Step 2**: 用 Sumformer (Alberti et al. 2023) 的 universal approximation 定理 C.7

Sumformer 是 $S([x_1, \dots, x_n]) := [\psi(x_1, \Sigma), \dots, \psi(x_n, \Sigma)]$，$\Sigma = \sum_k \phi(x_k)$。定理 C.7 说任何 equivariant 函数都能被 Sumformer 近似。

**Step 3**: 构造 DPOT 来拟合 Sumformer（Lemma C.9）

- **Lifting operator** $\mathcal{R}$（公式 40）：编码 $\phi(x_k)$ 和 positional encoding $P(j) = \mathcal{F}^{-1}(\{1, \dots, N\})(j)$
- **第一层 Fourier attention**（公式 42-44）：用特定 kernel $K_1$ 让只有 DC component（k=1）保留 $\frac{1}{\sqrt{N}} \sum_k \phi(x_k)$，其他 frequency 被过滤。经过 IFFT 后每个位置都拿到 $\Sigma / N$
- **Skip connection + $M_1$**（公式 45-46）：把 $\Sigma$ 复制到每个位置
- **第二层 Fourier attention**：kernel 设为 identity，直接 $M_2$ 拟合 $\psi(\Sigma, x_k)$（公式 47）

**直觉**：DPOT 的 Fourier 操作天然能实现 "全位置求和然后复制回每个位置"——因为 DC component (k=0) 经过 IFFT 后是常数广播。这就是 attention 中 "每个 token 看到 global context" 的 Fourier 实现。

---

## 3. 实验详解

### 3.1 Setup

- 12 个数据集来自 4 个 source：FNO、PDEBench、PDEArena、CFDBench
- AdamW，lr=1e-3，One-cycle schedule，1000 epochs（200 warm-up）
- 8×A800 80GB，batch size 160
- Patch size P=8，T=10 时间步输入预测下一帧
- Long trajectory 数据集（PDB-DR、PDB-SWE）weight $w=3$

### 3.2 Model Configurations（Table 5）

| Size | Attn dim | MLP dim | Layers | Heads | Params |
|------|----------|---------|--------|-------|--------|
| Tiny | 512 | 512 | 4 | 4 | 7M |
| Small | 1024 | 1024 | 6 | 8 | 30M |
| Medium | 1024 | 4096 | 12 | 8 | 122M |
| Large | 1536 | 6144 | 24 | 16 | 509M |
| Huge | 2048 | 8092 | 27 | 8 | 1.03B |

### 3.3 Main Results（Table 1）

分三部分：

**Part 1: Small models (≤30M)**
- DPOT-S 在 9/12 数据集 SOTA
- FNO-1e-4：从 FNO-m 的 0.0922 → DPOT-S 的 0.0442（52% improvement）
- 对比 MPP：MPP 只支持部分数据集，DPOT 在大多数子集上更好

**Part 2: Pre-trained large models**
- DPOT-H (1B) 几乎全面 SOTA
- DPOT-L (500M) vs MPP-L (400M)：DPOT 在 10 个数据集上更好
- CFDBench 上 MPP 略胜，留有改进空间

**Part 3: Fine-tuned DPOT**
- DPOT-L-500 在 9/12 任务 SOTA
- 相比所有 zero-shot 方法，8/12 任务 L2RE 降低 >50%
- Fine-tuning 越久通常越好，存在 cost-performance trade-off

### 3.4 Downstream Tasks（Table 2）

| L2RE/L1 | Turbulence | 3D PDB | Steady CNO |
|---------|-----------|--------|------------|
| (Geo-)FNO | 0.193 | 0.410 | 0.0357 |
| MPP-FT | 0.152 | - | - |
| DPOT-Vanilla | 0.167 | 0.262 | 0.0331 |
| DPOT-FT | 0.135 | 0.226 | 0.0230 |

亮点：
1. **High-res turbulence**：DPOT-FT 0.135 < MPP-FT 0.152
2. **3D NS**：FNO 0.410 → DPOT-FT 0.226，**仅用 2D pre-training 数据迁移到 3D 仍 work**！这是 Fourier attention 的优势——FFT 直接换成 3D FFT 即可
3. **Steady CNO**：从 0.0357 → 0.0230，证明对 steady-state（非 time-dependent）PDE 也可迁移

### 3.5 Scaling Experiments（Figure 3）

- 7M → 500M：zero-shot L2RE 从 0.06 → 0.028，遵循 scaling law
- 7M DPOT 比 30M AViT 性能更好——DPOT 参数效率高
- 不同 layer 数都享受 scaling，但 DPOT 整体曲线更优

### 3.6 Ablations

#### Number of heads（Table 3 left）

| $N_h$ | Train loss | Test L2RE |
|-------|-----------|-----------|
| 1 | 0.0139 | 0.0214 |
| 4 | 0.0129 | 0.0186 |
| **8** | 0.0127 | **0.0174** |
| 16 | 0.0138 | 0.0188 |

8 heads 最优。太少 expressivity 不够，太多参数稀释。

#### Patch size（Table 3 right）

| $P$ | Train loss | Test L2RE |
|-----|-----------|-----------|
| 2 | 0.0242 | 0.369 |
| 4 | 0.0114 | 0.0275 |
| **8** | 0.0127 | **0.0174** |
| 16 | 0.0186 | 0.0278 |

$P=8$ 最佳。$P=2$ 灾难性——可能因为分辨率太高导致 Fourier mode 数过多，soft 信息丢失；$P=16$ 太粗糙丢失空间细节。

#### Noise level $\epsilon$（Table 4）

| $\epsilon$ | Train loss | Test L2RE |
|-----------|-----------|-----------|
| 0 | 0.00735 | 0.0178 |
| **5E-5** | 0.00769 | **0.0152** |
| 5E-4 | 0.0133 | 0.0156 |
| 5E-3 | 0.0411 | 0.0672 |
| 5E-2 | 0.166 | 0.753 |

甜点在 5E-5：训练 loss 略升但 test loss 下降——典型 regularization 现象。超过 5E-3 噪声过强导致欠拟合。

### 3.7 关键 Supplementary 实验

#### Long trajectories（Table 10，Appendix B.5）

| Steps | 20 | 50 | 100 | 200 | 500 |
|-------|----|----|-----|-----|-----|
| DPOT | 0.00188 | 0.00192 | 0.00592 | 0.0241 | 0.0912 |
| DPOT-pretrained | 0.00148 | 0.00186 | 0.00335 | 0.0110 | 0.0385 |

短 trajectory pre-training 帮助小，长 trajectory 优势爆发——500 步时 pretrained 误差是 vanilla 的 42%。Pre-training 让模型学到 "PDE 通用的长时间稳定性 prior"。

#### Kolmogorov turbulence（Table 12，Appendix B.7）

| Model | Train loss | Test (full trajectory) |
|-------|-----------|------------------------|
| FNO | 0.0931 | 1.04 |
| DPOT | 0.0485 | 0.822 |
| DPOT-pretrained | 0.0296 | 0.335 |

Kolmogorov flow 有显著 chaotic 行为。FNO full trajectory error 104% = 完全没学到。DPOT-pretrained 33.5% = 学到了部分 chaotic 演化 pattern。Pre-training 对 chaotic 系统 help 巨大。

#### Self-attention vs Fourier mixer（Table 14）

| Mixer | #Params | FLOPs | Train | Test |
|-------|---------|-------|-------|------|
| Self-attention | 52.1M | 88.45G | 0.0183 | 0.0238 |
| Fourier mixer | 30.8M | 75.44G | 0.0127 | 0.0174 |

Fourier mixer 参数少 40%、FLOPs 少 15%、test error 低 25%。PDE 数据的 spectral structure 让 Fourier 操作天然契合。

#### Super-resolution（Table 11，Appendix B.6）

Navier-Stokes Kraichnan Turbulence (Re=16000) 8× upsampling (128→1024)：

| Model | RFNE (%) |
|-------|----------|
| SwinIR | 0.80 |
| WDSR | 0.72 |
| EDSR | 0.57 |
| DPOT-vanilla | 0.59 |
| DPOT-pretrained | 0.46 |

Pre-trained DPOT 显著超过 EDSR（current SOTA）。证明 PDE foundation model 对 super-resolution downstream 也有效。

#### Resolution generalization（Table 9）

| Resolution | Test L2RE |
|-----------|-----------|
| 32 | 0.0181 |
| 48 | 0.0182 |
| 72 | 0.0187 |
| 96 | 0.0187 |
| 128 | 0.0190 |

不同分辨率下性能非常稳定——Neural operator 的 key 优势。需要对 patchification layer 做 CNO 风格修改。

#### AFNO 对比（Table 8）

| Model | Train | Test |
|-------|-------|------|
| AFNO (soft thresholding) | 0.0664 | 0.121 |
| DPOT (no sparsity) | 0.0127 | 0.0174 |

去掉 sparsity constraint 让 train loss 降 5×、test loss 降 7×。Soft thresholding 对 PDE 多尺度特性有害。

---

## 4. Intuition 总结：为什么 DPOT work

1. **Auto-regressive denoising = 解决 exposure bias 的廉价方案**。注入 noise 让模型在训练时就见到 "已扰动 input"，强迫它学习从 perturbed state 回到 clean state。比 pushforward trick 简单，比 scheduled sampling 更 principled。

2. **Fourier attention = PDE 解的天然 basis**。很多 PDE 解在 frequency domain sparse 或 compressible（turbulence 的 energy cascade 就是 k^(-5/3) law）。在 frequency domain 做 mixing 等价于学习一个 global convolution operator，complexity $O(N \log N)$ 而非 $O(N^2)$。

3. **Translation-invariant kernel 是 PDE 的物理 prior**。Homogeneous media 上的 PDE 解算子 translation equivariant，Fourier multiplier 是其最自然表达。

4. **Temporal aggregation 用 $e^{-i\gamma t}$ basis**。PDE 时间演化 $u(t) = e^{\lambda t} u(0)$ 形式，Fourier basis 编码周期性，复数还能表达衰减+振荡。

5. **No sparsity**：PDE 多尺度 spectral 特性意味着 frequency spectrum 不该被强行 sparse，soft thresholding 反而破坏信息。

6. **Weight sharing across frequencies**：参数量从 $m \cdot d_z^2$ 降到 $2 d_z^2$，但提供 inductive bias——所有 frequency modes 由同一组 MLP 处理，类似一种 "frequency-equivariant" 设计。

7. **Balanced sampling**：小数据集反比采样，避免大数据集 dominate 梯度。Long trajectory 数据集额外加权。

8. **Scaling law 在 PDE 也成立**：7M → 1B 参数持续提升，证明 PDE pre-training 跟 NLP/CV 一样可以 scale。

9. **Pre-training 学到 "PDE universal prior"**：从 12 个数据集学到的 representation 能迁移到 3D、high-res、steady-state、chaotic 等完全 OOD 任务，说明模型学到的不是某条 PDE 的解，而是 "PDE 算子家族" 的通用结构。

10. **2D pre-training 能迁移到 3D**：因为 Fourier attention 的参数与 spatial dimension 解耦——只需把 2D FFT 换成 3D FFT，权重直接复用。这是 transformer + spectral operation 相比 CNN 的优势。

---

## 5. 局限与开放问题

- CFDBench 上 MPP 仍胜，DPOT 在 irregular geometry 上可能需要更强设计
- Long trajectory 上虽然 pretrained 优势大，但绝对误差仍较大（500 步 9%）
- 1B 参数虽然大，但相比 LLM 还小得多——PDE scaling law 的极限尚未探明
- Pre-training 数据多样性 vs 模型容量的 scaling 关系未深入研究
- 与 PINN-style physics-informed loss 结合的可能性未探索

---

**Reference links**:
- Paper PDF: https://arxiv.org/abs/2403.02612
- Official code: https://github.com/thu-ml/DPOT
- PDEBench: https://github.com/pdebench/PDEBench
- PDEArena: https://github.com/microsoft/pdearena
- AFNO paper (Guibas et al. 2021): https://arxiv.org/abs/2111.13587
- FNO original (Li et al. 2020): https://arxiv.org/abs/2010.08895
- MPP (McCabe et al. 2023): https://arxiv.org/abs/2310.02994
- Sumformer (Alberti et al. 2023) used in proof: https://arxiv.org/abs/2310.11524
- ViT (Dosovitskiy et al. 2020): https://arxiv.org/abs/2010.11929
- FourcastNet (Pathak et al. 2022): https://arxiv.org/abs/2202.11214
- GNOT (Hao et al. 2023): https://arxiv.org/abs/2302.14376
- CNO (Raonic et al. 2023): https://arxiv.org/abs/2302.01178
- CFDBench: https://arxiv.org/abs/2310.05963
