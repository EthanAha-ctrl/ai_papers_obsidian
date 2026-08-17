---
source_pdf: DiT4Edit.pdf
paper_sha256: 74fe38d812aaeba181875307d6277cdcc9de5c29641bf08a4e59bc64085243e3
processed_at: '2026-08-03T22:39:52-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DiT4Edit

## 一句话总结

之前所有 image editing 方法都用 UNet，这篇 paper 第一次说："我们换用 DiT (Diffusion Transformer) 来做 editing，效果更好，而且能编辑大图"。

就这么简单。核心 contribution 就是这个 backbone 切换。

---

## 为什么换 backbone 这么重要？

想象 UNet 是一个**近视眼**。它的核心操作是 convolution，conv 只能看周围一小圈邻居。要让它理解"图左上角的猫和右下角的椅子有什么关系"，得堆很多层 conv，让信息一层层传过去。对 512×512 的小图还行，图像一大，信息传递就跟不上。

DiT 是个**千里眼**。它的 self-attention 让图里每个 patch 第一层就能看到所有其他 patch。不管图多大，long-range relationship 都能直接 capture。

这对 image editing 意味着什么？

Editing 任务本质是"改 A 但保持 B 不变"。比如"把马变成船，但背景湖水不动"。要做到这个，模型得理解马和湖的空间关系、马的形状怎么和湖面互动。这种 long-range reasoning 恰好是 DiT 的强项、UNet 的弱项。

Paper 里 Figure 3 很直观：PixArt-α 浅层的 self-attention query 模糊一片，深层却清晰刻画 object boundary 和 layout。这说明 DiT 深层确实在做 layout reasoning，而 UNet 的 self-attention 只在很低分辨率（16×16 latent）才出现，能力受限。

参考：https://arxiv.org/abs/2212.09748 (DiT 原始 paper)

---

## 三个技术组件，分别解决三个问题

### 问题 1：Inversion 慢

**背景**：editing 真实图片，得先把图片"反向加噪"变成纯噪声，然后再用新 prompt 生成。这个反向过程叫 inversion。

**老方法**：DDIM inversion，得跑 50 步才能拿到好的 noise map。慢。

**DiT4Edit 做法**：用 DPM-Solver++(2M) 做 inversion，30 步就够。

**直觉解释**：DDIM 是一阶 solver，每步误差 $O(h^2)$。DPM-Solver++(2M) 是二阶 multistep solver，每步误差 $O(h^3)$。同样步数下高阶 solver 精度高得多，所以能少跑步数。

**数学一点**：DDIM 更新公式是
$$x_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} x_{t_{i-1}} - \alpha_{t_i}(e^{-h_i}-1) x_\theta(x_{t_{i-1}}, t_{i-1})$$

只用了当前一步的 model prediction $x_\theta$。

DPM-Solver++(2M) 更新公式是
$$x_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} x_{t_{i-1}} - \alpha_{t_i}(e^{-h_i}-1) \left[(1+\frac{1}{2r_i}) x_\theta(x_{t_{i-1}}, t_{i-1}) - \frac{1}{2r_i} x_\theta(x_{t_{i-2}}, t_{i-2})\right]$$

用了前两步的 model prediction，相当于做了 linear extrapolation，二阶精度。

变量含义：
- $x_{t_i}$: 时间步 $t_i$ 的 latent
- $\sigma_{t_i}$: 时间步 $t_i$ 的噪声标准差
- $\alpha_{t_i}$: 时间步 $t_i$ 的信号保留系数
- $h_i = \lambda_{t_i} - \lambda_{t_{i-1}}$: 在 $\lambda = \log(\alpha/\sigma)$ 空间的步长
- $r_i = \frac{\lambda_{t_{i-1}} - \lambda_{t_{i-2}}}{\lambda_{t_i} - \lambda_{t_{i-1}}}$: 相邻步长比
- $x_\theta$: 模型预测的 clean image

**Inversion 的难点**：sampling 是正向 $x_{t_{i-1}} \to x_{t_i}$，inversion 是反向 $x_{t_i} \to x_{t_{i-1}}$，但公式里 $x_\theta(x_{t_i}, t_i)$ 依赖未知的 $x_{t_i}$，得迭代解。

DiT4Edit 借用了 Hong et al. 2024 的 backward Euler trick，还发现可以省掉辅助的 DDIM inversion 估计，直接用已有的两个历史点近似二阶项。这个 empirical 简化很实用。

参考：https://arxiv.org/abs/2211.01095 (DPM-Solver++)

---

### 问题 2：怎么控制编辑？

**背景**：P2P (Prompt-to-Prompt) 发现 cross-attention map 携带语义信息，替换 cross-attention map 可以做 editing。MasaCtrl 进一步发现 self-attention 携带 layout 信息，mutual self-attention（用 target 的 Q 查 source 的 K, V）能做 non-rigid editing。

**MasaCtrl 的问题**：它全程用 target 的 $Q_{tar}$，当 source 和 target layout 差异大时，$Q_{tar}$ 会把 attention 带偏，背景破坏。

**DiT4Edit 的修正**：加一个阈值 $S$，前半段完全用 source 的 attention，后半段才用 target 的 Q 查 source 的 K, V。

公式：
$$\text{Attention} = \begin{cases} \text{Attn}(Q_{src}, K_{src}, V_{src}) & \text{if } t > S \\ \text{Attn}(Q_{tar}, K_{src}, V_{src}) & \text{if } t \leq S \end{cases}$$

**直觉**：diffusion 早期（t 大，噪声多）决定 coarse layout，晚期（t 小，噪声少）决定 detail。
- 早期用纯 source attention：把背景和 layout 锚定住，不让 target prompt 乱跑
- 晚期切换到 target Q：让新 object 在锚定的 layout 槽位里生成

这就是"unified attention control"：cross-attention 管语义替换（"马"变"船"），self-attention 管 layout 约束（船在湖里，湖不变）。

参考：https://arxiv.org/abs/2304.08465 (MasaCtrl)

---

### 问题 3：DiT 计算太慢

**背景**：DiT 的 self-attention 复杂度 $O(N^2)$，N 是 patch 数。512×512 图 N≈4096，1024×2048 图 N 会上万。attention 矩阵爆炸大。

**DiT4Edit 做法**：插入 Token Merging (ToMe)。在每层 attention 前，计算 patch 间的 similarity，贪心合并最相似的 patch pair，保留 80% patches 做 attention，attention 完了再 unmerge 恢复数量。

**直觉**：图里大量 patch 是 redundant 的——天空、墙面、水面这些大面积均匀区域，合并它们不会丢信息。ToMe 在 ViT classification 上证明几乎无损，DiT4Edit 借用到 editing 上也 work。

**消融结果**：
- 512×512：6.01s → 9.13s（省 3s，34% 加速）
- 1024×1024：34.27s → 41.52s（省 7s，17% 加速）
- 1024×2048：99.39s → 122.41s（省 23s，19% 加速）

大图上绝对收益更大，因为 redundancy 更多。

参考：https://arxiv.org/abs/2210.09461 (Token Merging)

---

## 实验数据怎么看

Table 1 最有说服力的几个数字：

**FID（越低越好，衡量生成质量）**：
- 1024×1024 上，DiT4Edit 62.45，最好的 UNet 方法 PnPInversion 85.33
- 差距 23 个点，这是巨大优势

**PSNR（越高越好，衡量背景保持）**：
- 1024×1024 上，DiT4Edit 29.75，最好的 UNet 方法 PnPInversion 23.42
- 差 6dB，相当于 background preservation 有数量级提升

**1024×2048（UNet 方法基本崩掉）**：
- SDEdit FID 143.87，MasaCtrl FID 236.49，Pix2Pix-Zero FID 158.29
- DiT4Edit FID 75.43，虽然比 1024² 退步，但远好于 UNet 方法

**Inference time**：
- 512×512 上 5.15s，和 InfEdit (5.08s) 持平
- InfEdit 是 inversion-free 方法，本该最快。DiT4Edit 带 inversion 还能追平，说明 DPM-Solver + patches merging 的组合效率很高

---

## 我的几点 intuition

### 1. 这篇 paper 的真正价值

不是某个算法特别创新，而是**第一次证明 DiT 在 editing 任务上系统性优于 UNet**。三个组件（DPM inversion, unified attention, patches merging）都是借用已有技术，但组合起来在 DiT 上 work，本身就是有价值的 empirical 发现。

### 2. 为什么 self-attention 控制在 DiT 上更重要

UNet 的 self-attention 只在 16×16 / 32×32 低分辨率层出现，且是局部的。DiT 全层都是 global self-attention，每层都做 layout reasoning。所以 MasaCtrl 式的 mutual self-attention 在 DiT 上效果放大——因为 self-attention 本身就更强大，控制它的影响力也更大。

### 3. 阈值 $S$ 是个 hack

Eq. 11 的 hard threshold $S$ 其实很粗糙。理想情况下应该有 adaptive 切换——根据 source/target 的 layout 相似度自动决定何时切换 $Q$。Paper 没做这个，是未来改进空间。

### 4. Patches merging 与 attention control 的潜在冲突

如果 source branch 和 target branch 各自独立做 patches merging，merge pattern 不一样，那 attention sharing 时 token 对不上。Paper 没说怎么解决，可能是 silent bug，限制了复杂编辑场景的效果。

### 5. 局限很真实

- T5 tokenizer 的分词问题：PixArt-α 用 T5-XXL 做 text encoder，sentencepiece 对复合词切分不规则，导致 cross-attention 替换时 token 对不齐
- Color inconsistency：DPM-Solver inversion 不做 per-step optimization（不像 Null-Text Inversion），inverted latent 与真实 noise 有 residual error

---

## 一句话评价

这篇 paper 是 DiT editing 的开山之作，方法上不算惊艳，但 empirical 价值很高——它告诉我们 DiT 不只在 generation 上比 UNet 强，在 editing 这种需要精细控制的任务上也能系统性碾压。后续 Flux editing、SLED 等工作都在这条线上继续走。

如果你想自己试，建议：
- Backbone 换成 PixArt-Σ 或 Flux，效果会更好
- 阈值 $S$ 按 task 调：object replacement 用大 $S$ 保守，style transfer 用小 $S$ 激进
- Patches merging ratio 0.8 是安全值，background-heavy 图可以更激进

Paper 的开源代码应该在这里：https://github.com/PixArt-alpha/PixArt-alpha (基础 backbone)
DPM-Solver 实现：https://github.com/LuCheng-ThU/DPM-Solver

---

# DiT4Edit 深度技术解析

## 1. Paper 核心定位

DiT4Edit 是第一个基于 **Diffusion Transformer (DiT)** 的 tuning-free image editing framework。此前 image editing 领域（PnPInversion, MasaCtrl, InfEdit, Pix2Pix-Zero）几乎全部基于 UNet architecture (Stable Diffusion 系列)，受限于 UNet 的 local inductive bias 和 fixed resolution。DiT4Edit 把编辑能力迁移到 PixArt-α backbone 上，利用 global self-attention 的 long-range 优势，支持 512×512 到 1024×2048 arbitrary aspect ratio 的编辑。

**关键参考链接**：
- Paper arXiv: https://arxiv.org/abs/2410.01419 (推断)
- PixArt-α: https://arxiv.org/abs/2310.00426
- DPM-Solver++: https://arxiv.org/abs/2211.01095
- MasaCtrl: https://arxiv.org/abs/2304.08465
- Token Merging (ToMe): https://arxiv.org/abs/2210.09461
- PnPInversion: https://arxiv.org/abs/2303.17563

---

## 2. 方法论三大支柱

### 2.1 DPM-Solver++ Inversion (替代 DDIM Inversion)

**动机**：传统 Null-Text Inversion / PnPInversion 使用 DDIM inversion，需要 ~50 steps 才能拿到好的 inverted noise map。DiT4Edit 采用 high-order DPM-Solver++ 的 exact inversion（Hong et al. 2024 的工作），把 inversion steps 降到 30。

#### 数学推导细节

Diffusion 前向 (Eq. 2)：
$$q(x_t | x_0) = \mathcal{N}(x_t | \alpha_t x_0, \sigma_t^2 I)$$

变量解释：
- $x_0$: clean image latent
- $x_t$: 加噪到时间 $t$ 的样本
- $\alpha_t$: 信号系数，随 $t$ 单调递减
- $\sigma_t$: 噪声标准差，随 $t$ 单调递增
- $\alpha_t^2/\sigma_t^2$: Signal-to-Noise Ratio (SNR)，$t$ 的严格递减函数

Diffusion ODE (Eq. 3)：
$$\frac{dx_t}{dt} = \left(f(t) + \frac{g^2(t)}{2\sigma_t^2}\right) x_t - \frac{\alpha_t g^2(t)}{2\sigma_t^2} x_\theta(x_t, t)$$

其中：
- $f(t) = \frac{d \log \alpha_t}{dt}$: drift coefficient
- $g(t) = \sqrt{\frac{d\alpha_t^2}{dt} - 2 f(t) \alpha_t^2}$: diffusion coefficient
- $x_\theta(x_t, t)$: 模型预测的 clean image $x_0$

**关键 insight**：通过 exponential integrator 在 $\lambda$-空间求解，$\lambda_t = \log(\alpha_t/\sigma_t)$，比在 $t$-空间直接 Euler 积分收敛快得多。

通用解 (Eq. 4)：
$$x_t = \frac{\alpha_t}{\alpha_s} x_s - \alpha_t \int_{\lambda_s}^{\lambda_t} e^{-\lambda} x_\theta(\hat{x}_\lambda, \lambda) d\lambda$$

#### DPM-Solver++ 通式 (Eq. 5)

$$x_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} x_{t_{i-1}} + \sigma_{t_i} \sum_{n=0}^{k-1} \underbrace{x_\theta^{(n)}(x_{\lambda_{t_{i-1}}}, \lambda_{t_{i-1}})}_{\text{estimated}} \underbrace{\int_{\lambda_{t_{i-1}}}^{\lambda_{t_i}} e^{\lambda} \frac{(\lambda - \lambda_{t_{i-1}})^n}{n!} d\lambda}_{\text{analytically computed}} + \underbrace{O(h_i^{k+1})}_{\text{omitted}}$$

变量逐项解析：
- $t_i$: 第 $i$ 个离散时间点
- $h_i = \lambda_{t_i} - \lambda_{t_{i-1}}$: 在 $\lambda$ 空间的步长（注意 $\lambda$ 递减，所以 $h_i < 0$）
- $x_\theta^{(n)}$: 模型预测的 $n$ 阶导数（$n=0$ 即 $\hat{x}_0$ 预测，$n=1$ 即 score）
- 上标 $(n)$: 表示对 $\lambda$ 的 $n$ 阶导数，不是 power
- 下标 $t_i$, $t_{i-1}$: 时间步索引

#### 退化情况 $k=1$ → DDIM (Eq. 6)

$$x_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} x_{t_{i-1}} - \alpha_{t_i}(e^{-h_i} - 1) x_\theta(x_{t_{i-1}}, t_{i-1})$$

这里 $x_\theta$ 是 $\hat{x}_0$ prediction。DDIM 本质就是 DPM-Solver++ 的 1 阶特例。

#### 实用版本 $k=2$ → DPM-Solver++(2M) (Eq. 7)

$$x_{t_i} = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} x_{t_{i-1}} - \alpha_{t_i}(e^{-h_i} - 1) \cdot \left[\left(1 + \frac{1}{2r_i}\right) x_\theta(x_{t_{i-1}}, t_{i-1}) - \frac{1}{2r_i} x_\theta(x_{t_{i-2}}, t_{i-2})\right]$$

其中 $r_i = \frac{\lambda_{t_{i-1}} - \lambda_{t_{i-2}}}{\lambda_{t_i} - \lambda_{t_{i-1}}}$ 是相邻步长比，"2M" 表示 2nd-order Multi-step，需要保存前两步的模型预测。

#### Inversion 困难点

Sampling 是 $x_{t_{i-1}} \to x_{t_i}$（noise 到 image）。Inversion 反向：从 $x_0$ 推回 $x_T$ 需要解 implicit equation，因为 $x_\theta(x_{t_i}, t_i)$ 依赖 $x_{t_i}$ 自身。

Hong et al. 2024 (On Exact Inversion of DPM-Solvers) 的 trick (Eq. 9)：

$$d_i' = z_\theta(\hat{z}_{t_{i-1}}, t_{i-1}) + \frac{z_\theta(\hat{y}_{t_{i-1}}, t_{i-1}) - z_\theta(\hat{y}_{t_{i-2}}, t_{i-2})}{2r_i}$$

变量：
- $\hat{z}_{t_{i-1}}$: 当前已知的 inversion latent
- $\hat{y}_{t_{i-1}}, \hat{y}_{t_{i-2}}$: 用 DDIM inversion 单独估算的辅助量（用于估计二阶导数项）
- $d_i'$: 修正后的方向估计

最终 inversion 更新 (Eq. 10)：
$$\hat{z}_{t_{i-1}} = \hat{z}_{t_{i-1}} - \rho(z_{t_i}' - \hat{z}_{t_i})$$

其中 $z_{t_i}' = \frac{\sigma_{t_i}}{\sigma_{t_{i-1}}} \hat{z}_{t_{i-1}} - \alpha_{t_i}(e^{-h_i} - 1) d_i'$，$\rho$ 是 relaxation factor。

**DiT4Edit 的重要 empirical 发现**：作者提到"we observe that we can still obtain a good inversion latent map without using DDIM inversion to calculate the values of $\hat{y}$"。这是个很实用的简化——意味着可以跳过 Eq. 9 中 $\hat{y}$ 的 DDIM inversion，直接用 $\hat{z}$ 自身的两个历史点估计二阶项，大幅节省计算。

### 2.2 Unified Attention Control

#### 出发点：Figure 3 的观察

PixArt-α 不同深度的 self-attention query feature 可视化：
- **浅层**（左）：query 模糊，layout 信息不显著
- **深层**（右）：query 清晰刻画 object boundary 和 semantic layout

这个观察是 unified attention control 的核心 intuition——深层 self-attention 携带 geometric/layout 信息，而 cross-attention 携带 text semantic 信息。两者协同才能做好 non-rigid editing。

#### 与 MasaCtrl 的对比

MasaCtrl (Cao et al. 2023) 原始设计：
- Early steps: 用 $Q_{tar}, K_{tar}, V_{tar}$ 自由生成目标 layout
- Later steps: 用 $K_{src}, V_{src}$ 注入源 image 的结构信息
- 全程使用 $Q_{tar}$

**MasaCtrl 的失败模式**（Xu et al. 2024 / InfEdit 指出）：当 source 和 target 的 layout 差异较大时，全程用 $Q_{tar}$ 会导致 layout drift，背景破坏。

#### DiT4Edit 的修正 (Eq. 11)

$$\text{Mutual Edit} = \begin{cases} \text{Attention}\{Q_{src}, K_{src}, V_{src}\}, & \text{if } t > S \\ \text{Attention}\{Q_{tar}, K_{src}, V_{src}\}, & \text{otherwise} \end{cases}$$

变量含义：
- $Q_{src}, K_{src}, V_{src}$: source branch（reconstruction path）的 self-attention 投影
- $Q_{tar}, K_{tar}, V_{tar}$: target branch（editing path）的 self-attention 投影
- $S$: 步骤切换阈值（hyperparameter）
- $t$: 当前 diffusion step

**Intuition**：
- 当 $t > S$（早期，noise 大）：完全用 source 的 attention，保持 background 不动，相当于 "anchor" 阶段
- 当 $t \leq S$（后期，detail 阶段）：用 target 的 $Q$ 去查询 source 的 $K, V$，让新 prompt 引导的 object 在 source 的结构 slot 中生成

这其实就是 cross-attention 替换（P2P 思想）在 self-attention 域的扩展，只是切换逻辑变成了 hard threshold $S$。

#### 还需要 cross-attention 控制

DiT4Edit 同时使用 cross-attention replacement（标准 P2P）处理 token-level replacement，例如 "a horse" → "a boat"。这两层 attention 协同形成 "unified control"：
- Cross-attention: 语义替换（what）
- Self-attention: 结构/位置约束（where）

### 2.3 Patches Merging

#### 动机

DiT 的 self-attention 复杂度 $O(N^2)$，$N$ 是 patch 数。对 1024×2048 图像，$N$ 可达数千，远超 UNet 中 attention 的 token 数（通常 64×64=4096 latent token）。直接计算 attention 极慢。

#### 方法（来自 Token Merging, Bolya et al. 2023）

流程（Figure 4）：
1. 对 feature map 计算所有 patch 间的 cosine similarity
2. 贪心合并 similarity 最高的 patch pair（类似 agglomerative clustering）
3. 合并到目标 ratio（paper 用 0.8，即保留 80% patches）
4. 在 attention 计算后 unmerge，恢复原始 patch 数量供下一层使用

#### 为什么这对 DiT 编辑有效

- 编辑任务对 spatial precision 有要求，但 attention 内部其实存在大量 redundant patches（背景天空、平整墙面）
- Token Merging 证明对 ViT classification / generation 几乎无损
- DiT4Edit 实证：merging ratio 0.8 在质量上几乎无差异（Figure 6），但时间显著下降

---

## 3. 实验数据深度解读

### 3.1 Table 1 主结果

| Model | Backbone | FID 1024² | PSNR 1024² | CLIP 1024² | Time 512² |
|---|---|---|---|---|---|
| SDEdit | UNet | 88.56 | 20.76 | 21.39 | 15.62s |
| IP2P | UNet | 98.73 | 19.73 | 19.36 | 12.43s |
| Pix2Pix-Zero | UNet | 101.32 | 16.27 | 17.85 | 31.45s |
| MasaCtrl | UNet | 176.15 | 20.51 | 21.96 | 19.76s |
| InfEdit | UNet | 87.42 | 22.36 | 23.74 | 5.08s |
| PnPInversion | UNet | 85.33 | 23.42 | 20.76 | 30.48s |
| **DiT4Edit** | **DiT** | **62.45** | **29.75** | **26.97** | **5.15s** |

**关键观察**：
1. FID 在 1024² 上 DiT4Edit (62.45) 显著低于所有 UNet baseline (85+)，说明生成质量优势在大图上放大
2. PSNR 29.75 是 UNet 方法中最佳 (PnPInversion 23.42) 的近 6dB 提升——意味着 background preservation 有数量级提升
3. Inference time 5.15s 与 InfEdit (inversion-free, 5.08s) 持平，但 InfEdit 质量明显差。DiT4Edit 既快又好。
4. **1024×2048 行**：所有 UNet 方法都做不到（要么 FID 爆炸如 Pix2Pix-Zero 158，要么 PSNR 极低如 SDEdit 16.35）。DiT4Edit 是唯一能在该尺寸保持 FID 75.43 / PSNR 27.46 的方法。

### 3.2 Table 2 Patches Merging 消融

| Image Size | Merging | Time |
|---|---|---|
| 512² | ✓ | 6.01s |
| 512² | ✗ | 9.13s |
| 1024² | ✓ | 34.27s |
| 1024² | ✗ | 41.52s |
| 1024×2048 | ✓ | 99.39s |
| 1024×2048 | ✗ | 122.41s |

**Intuition**：加速比与 patch 数近似线性。512² 只省 3s（绝对值小），1024×2048 省 23s（相对 19%）。这暗示 patches merging 在大图上的收益递增，因为 redundancy 随图像增大而增多。

### 3.3 Figure 7 DPM-Solver vs DDIM 消融

同 $T=30$ steps 下，DPM-Solver 的 editing 质量明显优于 DDIM。这验证了 high-order solver 在 inversion 上的优势——同样步数下，DPM-Solver++(2M) 的 truncation error 是 $O(h^3)$，而 DDIM 是 $O(h^2)$。

---

## 4. Architecture 深度解析（PixArt-α Backbone）

PixArt-α 的核心 DiT block：

```
Input: latent z_t (B, N, d), timestep t, text embedding y, class c
   │
   ├── AdaLN-single: 由 t 和 c 共同生成 scale/shift γ, β (单一 MLP，复用)
   │
   ├── Self-Attention:
   │     - Q = W_q * z_t, K = W_k * z_t, V = W_v * z_t
   │     - Attn = softmax(QK^T/√d) V
   │     - [DiT4Edit intervention point 1: self-attention control]
   │
   ├── Cross-Attention:
   │     - Q = z_t, K = W_k * y, V = W_v * y
   │     - [DiT4Edit intervention point 2: cross-attention replacement]
   │
   ├── MLP: 2-layer with GELU
   │
   └── Output: modulated residual + z_t
```

**与 SD UNet 的关键区别**：
1. **AdaLN-single**: PixArt-α 把 timestep 和 class condition 共享一个 AdaLN modulation，而 SD 用 separate AdaLN。这减少参数但需要 careful design。
2. **Pure global attention**: SD UNet 只有在 16×16 / 32×32 spatial resolution 才有 self-attention；其他层是 convolution。DiT 全层都是 global self-attention，这是大图编辑优势的来源。
3. **No skip connections**: UNet 的 encoder-decoder skip 是其核心 inductive bias。DiT 没有 skip，靠纯 transformer 处理所有 scale 信息，这意味着对图像内容的"理解"是 holistic 的。

---

## 5. 隐含的 Intuition 与 Open Questions

### 5.1 为什么 DiT 适合 image editing？

UNet 的 convolution 是 translation-equivariant，意味着对 patch-level long-range dependency 的捕获有限。当编辑需要 "把图左上角的猫变成狗，且狗的姿态与右下角的椅子保持构图关系" 这种 long-range reasoning，UNet 的 local conv 必须堆叠多层才能传递信息。DiT 的 global attention 让所有 patch 在第一层就能互相看到，layout reasoning 直接发生。

### 5.2 Eq. 11 的阈值 $S$ 该怎么设？

Paper 没明确给出 $S$ 的具体数值。从 MasaCtrl 经验推断，$S$ 应在总步数的 30%-50% 之间。Intuition：$t > S$ 阶段是 high-noise regime，主要决定 coarse layout；$t \leq S$ 是 low-noise regime，决定 detail。把 $Q$ 切换点放在这个 boundary 上能让 layout 既不被 source 过度锚定，又不被 target 完全带跑。

### 5.3 Patches merging 与 attention control 的交互

一个被 paper 忽略的问题：patches merging 后，self-attention 的 $K, V$ 来自 merged tokens，但 $Q$ 也 merged 了。那 source/target 之间的 attention sharing 如何对齐？如果 source branch 和 target branch 各自独立 merge，merge pattern 不同就会导致 token mismatch。这个 implementation 细节论文没说清楚，可能是性能瓶颈的潜在来源。

### 5.4 与后续工作的潜在连接

- **SLED (NeurIPS 2024)**: 同样探索 DiT editing，但用 exact inversion
- **LEDITS++**: 在 PixArt 上做 editing，但是更早
- **InfEdit (CVPR 2024)**: inversion-free 路线，可作为 DiT4Edit 的对照
- **Flux editing**: Flux 是更新的 DiT model，原理上 DiT4Edit 可以迁移

---

## 6. 局限与 Failure Modes

Paper 明确承认：
1. **T5 tokenizer 分词问题**：PixArt-α 用 T5-XXL 作为 text encoder。T5 的 sentencepiece tokenizer 对某些词切分不规则，导致 cross-attention map 对齐失败。例如"ice cream"可能被切成"ice" + "cream"，替换 token 时定位不准。
2. **Color inconsistency**: 由于 DPM-Solver inversion 不像 Null-Text Inversion 那样做 per-step optimization，inverted latent 与真实 noise map 之间有 residual error，导致颜色漂移。

未明确但可能存在的：
3. **Background leakage in object insertion**: 当 target prompt 引入全新 object（如"添加一只船"），Eq. 11 的 $K_{src}, V_{src}$ 仍带 source 全部信息，可能让新 object 渲染时携带 source 局部纹理。
4. **Attention control 与 patches merging 的冲突**: 如前述，merge pattern 不对齐可能破坏 attention sharing 的语义对应关系。

---

## 7. 实用建议

如果想复现或改进 DiT4Edit：

1. **Backbone 选择**: PixArt-α 是较弱的 DiT backbone。考虑用 PixArt-Σ 或 Flux 替换，注意 attention control 需要适配 dual-stream / single-stream transformer 架构。
2. **阈值 $S$ 调参**: 不可仅看 FID，要看 task-specific。Object replacement 用较大 $S$（保守），style transfer 用较小 $S$（激进）。
3. **Patches merging ratio**: 0.8 是经验值。对 background-heavy 图像可以更激进（0.6），对 foreground-detail 重要图像要保守（0.9）。
4. **Inversion accuracy 检查**: 跑 inversion 后做 reconstruction（用 source prompt 重建），PSNR 应 > 30，否则后续编辑会失真。

---

## 8. 总结与个人 Intuition

DiT4Edit 的真正价值不在于单一组件创新（DPM-Solver inversion、unified attention、patches merging 都借鉴了已有工作），而在于**第一次系统证明 DiT 在 image editing 上的优越性**，并在大图 / arbitrary aspect ratio 上打开新空间。

从 first principles 看：
- Image editing 需要 disentangle "what" (semantic) 和 "where" (layout)
- Cross-attention 控制 what
- Self-attention 控制 where
- DiT 的 global self-attention 让 where 控制在大图上仍然有效
- UNet 的 local self-attention 在大图上 where-reasoning 衰减

这就是为什么 paper 在 1024×2048 上展现出压倒性优势——不是单一算法更好，而是 backbone 的 inductive bias 与 editing 任务的 long-range reasoning 需求天然对齐。

**核心参考资源**：
- PixArt-α 官方 repo: https://github.com/PixArt-alpha/PixArt-alpha
- DPM-Solver 官方实现: https://github.com/LuCheng-ThU/DPM-Solver
- ToMe 实现: https://github.com/facebookresearch/ToMe
- DiT 原始 paper: https://arxiv.org/abs/2212.09748
- Stable Diffusion: https://arxiv.org/abs/2112.10752
