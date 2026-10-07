---
source_pdf: Diffusion Model Attribution via Spectral Coupling of.pdf
paper_sha256: 4698b2a2d0fc8e4184854082d911d040985c2046145babbae63ff74cd30e3919
processed_at: '2026-08-03T21:42:50-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话说清楚

**给一张 AI 生成的图，能不能猜出是哪个模型生成的？** 这篇 paper 说：别看图本身，看模型"擦掉噪声"的手法——每个模型擦噪声时，在不同频率段之间搬运能量的方式不一样，这就是它的指纹。

---

## 为什么看图不行

这个问题的根源在于 diffusion model 的本质。Diffusion model 学的是 score function $\nabla_{z_t} \log p_t(z_t)$，本质上是在学数据分布的几何结构。Kadkhodaie et al. 2023 那篇 [paper](https://arxiv.org/abs/2310.02557) 已经证明：即使在不同数据上训练，diffusion model 也会收敛到几乎一样的 score function，生成的图像在视觉上 indistinguishable。

举个具体例子：SD v1.4 和 SD v1.5 共享几乎所有的 weights，同一个 UNet-860M backbone，同一个 KL-f8 VAE。你用同一个 prompt、同一个 seed 让它们各生成 500 张图，然后用 pixel-space feature（786,000 维）或者 FFT power spectrum（262,000 维）去分类——不管你用什么 classifier，accuracy 都在 chance level 附近（12.5% 是 8 个 model 的 random baseline）。Output space 里根本没有 signal。

---

## 现有方法为什么也失败

**Invasive 方法**（watermarking）：需要在训练时或者 fine-tune 时把 watermark 嵌进去，会改 model 本身，而且 [Hu et al. 2024](https://arxiv.org/abs/2405.07145) 已经证明 stable signature 这种 watermark 可以被 attack 掉。

**Non-invasive 方法**里，classifier-based 方法（[Xu et al. 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Detecting_Origin_Attribution_for_Text-to-Image_Diffusion_Models_WACV_2025_paper.pdf), [Corvi et al. 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_CVPR_2023_paper.pdf)）纯粹在 output image 上训分类器，碰到架构相似的模型就完蛋。

Inversion-based 方法更聪明一点。[RONAN](https://proceedings.neurips.cc/paper_files/2023/hash/446a8ee2c8b5c7b7b7b7b7b7b7b7b7b7-Abstract-Datasets_and_Benchmarks.html) 和 [LATENTTRACER](https://arxiv.org/abs/2405.13360) 会把 image invert 回 latent space，测 reconstruction loss。问题在于：当几个 candidate model 共享同一个 VAE 时，不管你 query 哪个 model，reconstruction loss 都一样——signal 直接 collapse。Table 3 里 LATENTTRACER 在五个共享 VAE 的 1.x model 上全部是 20-27%，跟 random baseline 一样。

---

## 核心 insight：看 denoiser 怎么擦噪声

### 为什么是 denoiser

Latent diffusion 的 forward process 是给 latent $z_0$ 加 Gaussian noise：

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \eta$$

这里 $\bar{\alpha}_t$ 是 noise schedule（t=0 时为 1，t=T 时为 0），$\eta \sim \mathcal{N}(0, I)$ 是标准 Gaussian noise。

Denoiser $\hat{\epsilon}_\theta(z_t, t)$ 训练来预测 noise $\eta$，这等价于估计 score function $\nabla_{z_t} \log p_t(z_t) \propto -\hat{\epsilon}_\theta(z_t, t)$。Score function 编码了数据分布的完整统计几何，包括不同频率段之间的 covariance 结构。

关键点：**score function 是 model-intrinsic 的，input-independent 的**。所有 generation 都共享同一个 denoiser。你看到的 output image 反而被 prompt 和 noise seed 混杂了，但 denoiser 本身直接反映 model 学到了什么。

### Spectral geometry 是什么

Score function 的 Hessian $\nabla^2_{z_t} \log p_t(z_t)$ 表征 local covariance 结构。在频率域里，它的 off-diagonal block 告诉你：一个频率段的 variation 如何 couple 到另一个频率段。

不同训练数据会 imprint 不同的 covariance 结构。不同架构会 impose 不同的 inductive bias。这些都体现在 denoiser 的 Jacobian 上：

$$J_\theta(z_t, t) = \frac{\partial \hat{\epsilon}_\theta(z_t, t)}{\partial z_t}$$

**paper 的核心 insight：model identity 编码在 $J_\theta$ 的 spectral 结构里。**

---

## 怎么 probe 这个 Jacobian

直接计算 full Jacobian 不可行（64×64 latent 就是 4096×4096 矩阵）。paper 的做法很巧妙：**在频率域里 probe Jacobian 的 block 结构**。

### Step 1: 把频率平面切成 ring

把 $H \times W$ 的频率平面切成 $K$ 个同心 ring。频率 $(u, v)$ 的归一化半径：

$$\rho(u, v) = \frac{\sqrt{u^2 + v^2}}{\sqrt{H^2 + W^2}}$$

ring $k$ 包含 $\rho \in [k/K, (k+1)/K)$ 的频率。ring 0 是 DC 和最低频（global structure），ring $K-1$ 是最高频（fine texture）。paper 用 $K=8$。

### Step 2: 注入 band-limited noise

取 Gaussian noise $\eta$，在 Fourier 域里 mask 掉除了 ring $k$ 之外的所有频率，再 inverse FFT 回 spatial domain，然后 renormalize 保持能量：

$$\eta^{(k)} = \mathcal{F}^{-1}\big(\mathcal{F}(\eta) \odot M_k\big) \cdot \frac{\sigma(\eta)}{\sigma\big(\mathcal{F}^{-1}(\mathcal{F}(\eta) \odot M_k)\big) + \delta}$$

这里 $\mathcal{F}$ 是 2D FFT，$\odot$ 是 element-wise 乘法，$M_k$ 是 ring $k$ 的 binary mask，$\sigma(\cdot)$ 是标准差，$\delta=10^{-8}$ 防止除零。这个 renormalization 保证每个 ring 被以相同 signal level 探测。

再乘以 amplitude $s$（paper 用 $S=3$ 个 amplitude：$\{1.0, 1.25, 1.5\}$），得到 $\eta^{(k, s)} = s \cdot \eta^{(k)}$。

### Step 3: 看 denoiser 怎么响应

构造 noisy latent：

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \eta^{(k, s)}$$

Query denoiser（用 empty text embedding，去掉 prompt 依赖）：

$$\hat{\eta} = \hat{\epsilon}_\theta(z_t, t, \emptyset)$$

算 residual：$\mathbf{r} = \hat{\eta} - \eta^{(k, s)}$

如果 denoiser 是完美的 ring-$k$ noise estimator，residual 应该是零。任何非零的 residual 都告诉你 denoiser 把能量 map 到了哪里。

### Step 4: 算 coupling matrix

测量 residual 在每个 ring $j$ 上的能量，重复 $R$ 次取平均（paper 用 $R=10$）：

$$C_{s, t}[k, j] = \frac{1}{R} \sum_{i=1}^{R} \big\langle |\mathcal{F}(\mathbf{r}_i^{(k, s, t)})|^2, M_j \big\rangle$$

$|\mathcal{F}(\mathbf{r})|^2$ 是 residual 的 power spectrum，$\langle \cdot, M_j \rangle$ 是在 ring $j$ 内求和。$C_{s,t}[k, j]$ 就是"在 timestep $t$、amplitude $s$ 下，把能量注入 ring $k$，denoiser 在 ring $j$ 输出多少能量"。

然后做 row normalization：

$$C_{s, t}[k, \cdot] \gets \frac{C_{s, t}[k, \cdot]}{\|C_{s, t}[k, \cdot]\|_1 + \delta}$$

这一步非常关键。它把绝对能量转成分布，去掉了 image brightness 这个 confounding variable。这是后面 cross-domain generalization 的核心。

### Step 5: 跨 timestep 和 amplitude 聚合

不同 timestep probe 不同的东西：高噪声时 Jacobian 编码 low-frequency global structure，低噪声时编码 high-frequency texture。所以均匀采样 $T_n=5$ 个 timestep 在 $[0.2T, T]$ 内。

最终 signature 是把所有 $C_{s,t}$ 拼成一个向量：

$$\Phi(M, z_0) = \text{vec}(\mathbf{A}) \in \mathbb{R}^{S \cdot T_n \cdot K^2}$$

paper 配置下是 $3 \times 5 \times 64 = 960$ 维。

---

## 理论上为什么这是 Jacobian 的投影

Appendix C 给了 derivation。对 band-limited perturbation 做一阶 Taylor 展开：

$$\hat{\epsilon}_\theta(z_t, t, \emptyset) \approx \hat{\epsilon}_\theta(z_0, t, \emptyset) + J_\theta(z_0, t) \sqrt{1 - \bar{\alpha}_t} \eta^{(k, s)}$$

如果 $\eta^{(k, s)}$ 在 ring $k$ 内 isotropic，coupling entry 近似等于 Jacobian 在 frequency block 上的 squared Frobenius norm：

$$C[k, j] \propto \left\| P_j J_\theta(z_0, t) P_k \right\|_F^2$$

$P_k$ 是投影到 ring $k$ 的算子，$P_j J_\theta P_k$ 是 Jacobian 从 ring $k$ 到 ring $j$ 的 block。

所以 coupling matrix 就是 Jacobian 在频率 ring pair 上的 Monte Carlo 估计。因为 $J_\theta$ 由 learned score function 决定，expected coupling 是 model-intrinsic 的。

---

## 怎么用 signature 做 attribution

### Prototype 构建

对每个 candidate model $M$，用 $N=50$ 张 reference image 算 signature 取平均：

$$\Psi(M) = \frac{1}{N} \sum_{i=1}^{N} \Phi\big(M, E(x_0^{(i)})\big)$$

平均掉 content-dependent noise，恢复 model 在 signature space 的 centroid。因为 row normalization 已经去掉 content scale，$N=50$ 就够。

### 识别

两种方式：
1. **Zero-training argmin**：test image 的 signature 跟每个 prototype 算 cosine distance，assign 到最近的。
2. **LinearSVC**：在有 label 的 signature 上训线性分类器。

实验发现 LinearSVC 比 argmin 好很多，尤其在 model weight overlap 高时。这是因为 signal 结构是 global mean-shift：class centroid 线性可分，但 individual sample 在局部 overlap。Linear classifier 在 centroid 上 operate 所以 work，kNN 在 local neighborhood 上 operate 所以 fail。

---

## 实验结果

### 8 个 model 的 closed-set

故意选了覆盖三个 variation 维度的 model：

**Training data variation**（共享 1.x 架构）：SD v1.4, v1.5, Dreamshaper-8, Realistic Vision v5, SD v2.1

**Architecture variation**：SDXL (UNet-2.6B), PixArt-α (DiT transformer)

**Training procedure variation**：SDXL-Turbo（adversarial distillation 训练）

LinearSVC 在 $T_n=5$ 上达到 ≈100% accuracy。唯一错误发生在 SD v1.4 ↔ SD v1.5（weight overlap 最大），LinearSVC 完全消除这个 confusion。

### Cross-domain generalization

Prototype 用 SD-Prompts 数据集建，直接在 MS-COCO 上测，不 retrain：

| Classifier | SD-Prompts | MS-COCO | Drop |
|---|---|---|---|
| Argmin | 78.33% | 78.40% | +0.07 |
| LinearSVC | 99.98% | 96.20% | -3.80 |
| Random Forest | 96.23% | 83.60% | -12.63 |

Argmin 几乎完全不受影响（+0.07pp），因为 row normalization 和 unconditional conditioning 把 content-scale variation 去掉了，centroid 在 distribution shift 下不 move。Classifier-based 掉更多，因为 decision boundary 部分是 distribution-specific 的。LinearSVC 最 robust 因为它利用 global mean-shift 结构。

### Distortion robustness

6 种 strong distortion：

| Distortion | SD-Prompts (LinearSVC) | MS-COCO (LinearSVC) |
|---|---|---|
| Rotation (75°) | 91.9% | 90.6% |
| JPEG (Q=25) | 93.1% | 85.0% |
| Blur (σ=8) | 95.6% | 99.9% |
| Noise (σ=0.1) | 93.8% | 92.5% |

Rotation、JPEG、crop、brightness 基本不影响，因为保留 latent structure。Noise 和 blur 让 argmin 大幅下降，但 LinearSVC 反而更高——因为 strong distortion 加 isotropic noise，blurs individual-sample distance 但保留 class centroid，LinearSVC 在 centroid 上 operate。

**fundamental takeaway**：SDS 测的是 denoiser spectral routing，不是 pixel statistics，所以 pixel-space distortion 擦不掉 fingerprint。

### Baseline 对比

| Model | RONAN | LATENTTRACER | SDS |
|---|---|---|---|
| SD v1.4 | 21.4% | 22.1% | 100.0% |
| SD v1.5 | 20.2% | 19.7% | 100.0% |
| SD v2.1 | 22.2% | 26.6% | 100.0% |
| Dreamshaper-8 | 21.2% | 22.1% | 99.8% |
| Realistic V. v5 | 20.4% | 20.9% | 100.0% |
| SDXL | 48.4% | 59.2% | 100.0% |
| PixArt-α | 55.4% | 79.0% | 100.0% |
| SDXL-Turbo | 35.0% | 67.0% | 100.0% |
| **Overall** | **30.5%** | **44.8%** | **99.9%** |

两个 baseline 在共享 VAE 的五个 model 上全部 collapse 到 20-27%（random baseline 是 20%）。SDS 比 LATENTTRACER 高 55pp，比 RONAN 高 70pp，整个 gap 都在最难的 near-clone case 上。

---

## 我觉得最有意思的几个点

### 1. Mean-shift 结构

Section 3 的发现：class centroid 线性可分，individual sample 局部 overlap。这告诉你 signature 不是在做 tight cluster，是在做 centroid shift。因为 Jacobian 依赖 $z_0$（image content），同一 model 不同 image 的 Jacobian 会变，所以 individual signature 有 content-dependent 偏移。但 model 的 inductive bias（反映在 Jacobian 的频率结构上）是 image-independent 的，所以 centroid 稳定。

这跟 [FINGERINV](https://openaccess.thecvf.com/content/CVPR2025/papers/Teng_Fingerprinting_Denoising_Diffusion_Probabilistic_Models_CVPR_2025_paper.pdf) 的关键区别：FINGERINV 的 signature 高度 tied 到 chosen image，必须先 enroll；SDS 因为 row normalization 和 averaging，把 content 抽掉，得到 model-intrinsic centroid，可以 generalize 到任意 generated image。

### 2. Row normalization 的作用

Cross-domain generalization 和 distortion robustness 几乎完全来自 row normalization。去掉它的话，absolute energy 在 image brightness 上 heavily dependent，signature 会随 image content 变，centroid 在 distribution shift 下 move 大。

Row normalization 把"绝对能量"换成"能量分布"，把 content scale 这个 confounding variable 干掉。这跟 batch normalization 在 supervised learning 里的作用有哲学相似性——都是 normalize 掉 nuisance variation 让 underlying structure 显现。

### 3. 为什么 frequency domain

直接估计 full Jacobian 不可行。通过 frequency ring 投影把 Jacobian 压缩成 $K \times K$，既降维又保留 cross-band coupling 这个最 discriminative 的结构。

而且 spectral basis 跟 diffusion model 学习的 structure 天然对齐——[Kadkhodaie et al. 2023](https://arxiv.org/abs/2310.02557) 证明 diffusion model 学到 geometry-adaptive harmonic basis，所以 frequency-domain projection 跟 model 内部 representation 同构，不会 throw away discriminative signal。

### 4. Evasion 难度

要 evade SDS，adversary 需要 modify image 的 latent representation 来 mimic 另一个 model 的 denoising geometry。这比 pixel-level post-processing 难得多。对比 [Hu et al. 2024](https://arxiv.org/abs/2405.07145) 对 stable signature 的 attack——stable signature 可以通过 modify decoder 被 remove，但 SDS probe denoiser 本身，attacker 即使修改 output 也 erase 不了 denoiser 内部的 spectral geometry。

### 5. 跟 Stein's identity 的联系

Score function 的 Hessian $\nabla^2 \log p(z)$ 在 score-based model 理论里是核心对象。Stein's identity 告诉我们：

$$\mathbb{E}_{z \sim p}\left[\nabla^2 \log p(z) + \nabla \log p(z) \nabla \log p(z)^\top\right] = 0$$

Hessian 跟 data distribution 的 covariance 直接挂钩。SDS 的 coupling matrix 本质是 Hessian（通过 Jacobian）在频率 ring 上的 projection，捕获的是 distribution 的 cross-frequency covariance——这是 model-specific 的。

---

## Limitation 和 Future Direction

paper 自己列了三个限制：
1. **Closed-set**：source model 必须在 registry 里。Open-set attribution 是 open problem。
2. **White-box access**：需要 candidate model weights，对 proprietary API 不适用。
3. **不 detect tampering**：只能回答"哪个 model 生成的"，不能回答"image 是否被 post-modified"。

Natural extension：
- Open-set 可以用 distance threshold 或 outlier detection 在 signature space 做
- API setting 可以考虑 query-based fingerprinting
- Tamper detection 可以跟 [DetectGPT](https://arxiv.org/abs/2301.11305) 类似思路结合

paper 在 conclusion 里还提了一个 broader 想法：spectral coupling signature 可能 generalize 到其他 model analysis 任务——detecting fine-tuning, quantization artifacts, distillation。直觉上 fine-tuning 改变 score function 的 high-frequency 细节，quantization 改变 Jacobian 的 numerical structure，distillation 改变 denoising trajectory 的 geometry，这些都应该在 spectral coupling 上 leave detectable imprint。

---

## 最后

paper 的 [code](https://github.com/Pragati-Meshram/SGS) 开源了。整个方法只需要 forward pass，不需要 inversion、optimization、model modification，960 维 signature 在 8 个 model 上 99.9% accuracy，cross-domain 掉 3.8pp，对 6 种 strong distortion 鲁棒。核心 message 是：**generative model 的 identity 编码在 score function 的 spectral geometry 里**。这跟 Kadkhodaie et al. 关于 diffusion model generalization 来自 geometry-adaptive harmonic representation 的工作形成互补——diffusion model 学到 harmonic basis，而不同 model 在这个 basis 上的 cross-band coupling 是 model-specific 的。

---

# Diffusion Model Attribution via Spectral Coupling of Denoiser Responses 深度解读

Andrej，这篇 paper 处理一个在 generative AI provenance 里很根本的问题：**给定一张生成的图像，能不能可靠地判断它是哪个 diffusion model 生成的？** 作者 Pragati Meshram 和 Varun Chandrasekaran（UIUC）提出 Spectral Denoising Signatures（SDS），核心 insight 是 model identity 不编码在 generated outputs 里，而是编码在 **denoiser 的 spectral geometry** 里——也就是 denoiser 在 denoising 过程中如何跨 frequency band 重新分配能量。代码在 https://github.com/Pragati-Meshram/SGS 。

---

## 1. 为什么这个问题真正难

### 1.1 Output space 的根本 ambiguity

Kadkhodaie et al. 2023 的 [Generalization in diffusion models arises from geometry-adaptive harmonic representations](https://arxiv.org/abs/2310.02557) 已经证明：在不同数据子集上训练的 diffusion model 会收敛到几乎 identical 的 score function，产生视觉上 indistinguishable 的输出。这意味着在 output space 做 attribution 本质上是 ambiguous 的——你看到的图像本身没有足够 discriminative signal。

### 1.2 现有方法的两个 family 及其失败模式

**Invasive methods**：watermarking 需要 retraining 或 fine-tuning，比如 [Stable Signature](https://arxiv.org/abs/2303.10137)、[A recipe for watermarking diffusion models](https://arxiv.org/abs/2303.10137)。问题是会 degrade generation quality，而且 [Hu et al. 2024](https://arxiv.org/abs/2405.07145) 已经证明 stable signature 可以被 attack 掉。

**Non-invasive methods** 分两类：
- **Classifier-based**（[Xu et al. 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Detecting_Origin_Attribution_for_Text-to-Image_Diffusion_Models_WACV_2025_paper.pdf), [Corvi et al. 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_CVPR_2023_paper.pdf)）：纯粹在 generated images 上训分类器，但碰到 architecturally similar variants（SD v1.4 vs v1.5，共享 UNet-860M backbone 和 KL-f8 VAE）就完全 fail。
- **Inversion-based**（[RONAN](https://proceedings.neurips.cc/paper_files/2023/hash/446a8ee2c8b5c7b7b7b7b7b7b7b7b7b7-Abstract-Datasets_and_Benchmarks.html), [LATENTTRACER](https://arxiv.org/abs/2405.13360)）：通过 invert VAE decoder 或 gradient-based latent inversion 来测 reconstruction loss。每个 candidate 都要跑一次优化，慢；更致命的是当 candidate 共享同一个 autoencoder 时，reconstruction loss 对所有 model 都一样，discriminative signal 直接 collapse。

paper 用一个关键的实验把这个 structural failure 暴露出来：SD v1.4 和 SD v1.5 共享 VAE，所以任何 decoder-level signal 都失效。

### 1.3 Gap

paper 要回答的是：**能不能直接 probe denoiser 本身，既不需要 inversion，也不依赖 output-space signal，又能区分连架构都几乎一样的 model？**

---

## 2. 核心 insight：Model identity 在 denoiser 的 spectral geometry 里

### 2.1 为什么是 denoiser？

Latent diffusion model 的 forward process 是：

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \eta$$

其中 $\bar{\alpha}_t$ 是 cumulative noise schedule（随 t 单调递减，从 1 到 0），$\eta \sim \mathcal{N}(0, I)$ 是 Gaussian noise，$z_0 = E(x_0)$ 是 VAE encoder 输出的 latent。

Denoiser $\hat{\epsilon}_\theta(z_t, t)$ 训练来估计 noise component，等价于估计 score function：

$$\nabla_{z_t} \log p_t(z_t) \propto -\hat{\epsilon}_\theta(z_t, t)$$

这是 [Song et al. 2020](https://arxiv.org/abs/2011.13456) 的经典结论。关键点是：score function 编码的是数据分布的**完整统计几何**，包括 cross-frequency covariance 结构。任何由训练数据、architecture、fine-tuning 引入的 inductive bias 都必须通过这个 score function 表现出来。

跟 output image 比，denoiser 是**intrinsic 的、input-independent 的对象**：它被所有 generations 共享，直接反映 model 学到了什么。Output image 反而是被 specific prompt 和 noise seed mediates 的，signal 混杂。

### 2.2 Spectral geometry 是什么

Score function 的 Hessian $\nabla^2_{z_t} \log p_t(z_t)$ 表征 local covariance 结构：在频率域里，它的 off-diagonal block 反映一个 frequency band 的 variation 如何 couple 到另一个 band。不同训练数据 imprint 不同 covariance 结构，不同 architecture impose 不同 inductive bias（参考 [Kadkhodaie et al. 2023](https://arxiv.org/abs/2310.02557) 关于 geometry-adaptive harmonic basis 的工作）。

这些差异都体现在 denoiser 的 Jacobian：

$$J_\theta(z_t, t) = \frac{\partial \hat{\epsilon}_\theta(z_t, t)}{\partial z_t}$$

### 2.3 Paper 用三个 probe 验证这个假设（Section 3）

他们在 SD v1.4 vs SD v1.5 上做对比实验，1000 张 image（每 model 500 张），512×512，50 DDIM steps，CFG=7.5。比较三类 feature：
- Pixel-space statistics（786,000 维）
- FFT power spectrum（262,000 维）
- Spectral coupling signature（192 维，single timestep）

结果（Figure 2）：
- Pixel/FFT feature 完全 overlap，LinearSVC/PCA-LR/cosine-kNN 全部 near-chance
- Spectral coupling signature 即使只用 192 维，LinearSVC 也达到 98.5%

**Takeaway 1**：共享 autoencoder 的 model，output-space feature 无论维度多高都没有 discriminative signal。
**Takeaway 2**：192 维的 single-timestep spectral response 就能线性分开两个几乎一样的 model。signal 来自 denoiser Jacobian，是 model-intrinsic 的。
**Takeaway 3**：扩展到 $T_n=5$ timestep，LinearSVC 达到 99.9%，因为不同 timestep probe 正交的方面——高噪声时 Jacobian 编码 low-frequency global structure，低噪声时编码 high-frequency texture。

这个 mean-shift 结构特别有意思：class centroids 线性可分，但 individual sample 局部 overlap，所以 linear classifier work 而 kNN 失败（cosine-kNN 只有 56.1%）。

---

## 3. 方法：Spectral Denoising Signatures（SDS）

SDS 是一个 4-step pipeline（Figure 1）。

### 3.1 Problem formulation

**Closed-set attribution**：给定生成图像 $x_0$ 和 candidate model registry $\mathcal{M} = \{M_1, \ldots, M_n\}$，判断哪个 model 生成了 $x_0$。

**Verifier access**：white-box access 到 candidate model 的 denoiser forward pass $\hat{\epsilon}_\theta(z_t, t, \emptyset)$，其中 $\emptyset$ 是 empty text embedding（unconditional denoising）。**只需 forward pass，不需 gradient、optimization、generation-time knowledge**。

### 3.2 标准化条件

为隔离 model-specific 行为，所有 candidate model 共享：
- 一个 VAE encoder（paper 用 SD v1.5 的 KL-f8 VAE，512×512 input → 64×64 latent）
- 一个 scheduler（DDIM, T=50）
- Empty text embedding $\emptyset$（消除 prompt-dependent signal）

这保证 signature 反映的是 model 的**intrinsic denoising geometry**，不是 mismatch 的 latent space 或 sampling parameter 的 artifact。

### 3.3 Step 1: Partition（频率环划分）

把 $H \times W$ 的频率平面划分成 K 个同心 radial ring。频率 $(u, v)$ 的 normalized radial coordinate：

$$\rho(u, v) = \frac{\sqrt{u^2 + v^2}}{\sqrt{H^2 + W^2}}$$

ring $k$（$k \in \{0, \ldots, K-1\}$）包含所有 $\rho \in [k/K, (k+1)/K)$ 的频率，对应 mask $M_k \in \{0, 1\}^{H \times W}$。

- ring 0 = DC + lowest spatial frequency（global structure）
- ring K-1 = highest frequency（fine texture）

paper 用 $K=8$。

### 3.4 Step 2: Probe（注入 band-limited noise 测 response）

#### Band-limited probe noise

给定 i.i.d. Gaussian noise $\eta \sim \mathcal{N}(0, I)$，把它限制在 ring k：

$$\eta^{(k)} = \mathcal{F}^{-1}\big(\mathcal{F}(\eta) \odot M_k\big) \cdot \frac{\sigma(\eta)}{\sigma\big(\mathcal{F}^{-1}(\mathcal{F}(\eta) \odot M_k)\big) + \delta}$$

变量解释：
- $\mathcal{F}, \mathcal{F}^{-1}$：2D FFT 和 inverse FFT
- $\odot$：element-wise 乘法
- $M_k$：ring k 的 binary mask
- $\sigma(\cdot)$：标准差
- $\delta = 10^{-8}$：防止除零

这个 renormalization 保证所有 ring 被以 comparable signal level 探测，不会因为 high-frequency ring 能量天然低而造成 scale 失衡。

Scaled version with amplitude $s$：$\eta^{(k, s)} = s \cdot \eta^{(k)}$。paper 用 $S=3$ amplitudes $\{1.0, 1.25, 1.5\}$。

#### Denoiser query

构造 noisy latent 并 query denoiser：

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \eta^{(k, s)}$$

$$\hat{\eta} = \hat{\epsilon}_\theta(z_t, t, \emptyset)$$

#### Residual 和 coupling matrix

Residual $\mathbf{r} = \hat{\eta} - \eta^{(k, s)}$ 测量 model 的 predicted noise 偏离 injected perturbation 多少。如果 model 是 perfect noise estimator for ring k，residual 应该是零；任何 nonzero residual 都揭示 denoiser 把能量 map 进或 map 出哪些频率。

Coupling matrix entry：

$$C_{s, t}[k, j] = \frac{1}{R} \sum_{i=1}^{R} \big\langle |\mathcal{F}(\mathbf{r}_i^{(k, s, t)})|^2, M_j \big\rangle$$

变量解释：
- $R$：independently drawn noise samples 数量（paper 用 $R=10$，取平均以稳定 stochastic response）
- $|\mathcal{F}(\mathbf{r})|^2$：residual 的 power spectrum
- $M_j$：ring j 的 mask
- $\langle \cdot, \cdot \rangle$：内积（在这里就是 mask 内 power 的 sum）

$C_{s,t}[k, \cdot]$ 这一行就编码了当 ring k 被 perturbed 时，denoiser 把能量 route 到哪些 output frequency band。$C_{s,t}$ 是 $K \times K$ 矩阵。

#### Row normalization

$$C_{s, t}[k, \cdot] \gets \frac{C_{s, t}[k, \cdot]}{\|C_{s, t}[k, \cdot]\|_1 + \delta}$$

把绝对能量转成 output band 上的分布，去除 content-dependent scale variation（不同 image 亮度不同），让 coupling pattern 反映 model 的 learned behavior，不是 image brightness。这是 SDS 跨 prompt distribution 泛化的关键 design choice。

### 3.5 Step 3: Aggregate（跨 timestep 和 amplitude 聚合）

#### Timestep selection

均匀采样 $T_n$ 个 timestep 在 $[0.2T, T]$ 内（paper 用 $T_n = 5$）。动机是 Takeaway 3：高噪声 probe low-frequency global structure，低噪声 probe high-frequency texture，timestep 之间是 complementary 的。

#### Multi-scale probing

$S=3$ perturbation amplitudes，spanning 不同 noise magnitude 但仍在 model operational noise regime 内。不同 amplitude probe denoiser 在不同 operating point 上的 response，提供 complementary discriminative signal。

#### Signature tensor 和 final vector

完整测量：

$$\mathbf{A} \in \mathbb{R}^{S \times T_n \times K \times K}, \quad \mathbf{A}[s, t, k, j] = C_{s, t}[k, j]$$

SDS signature 是 flattened tensor：

$$\Phi(M, z_0) = \text{vec}(\mathbf{A}) \in \mathbb{R}^{S \cdot T_n \cdot K^2}$$

paper 配置下：$S \cdot T_n \cdot K^2 = 3 \cdot 5 \cdot 64 = 960$ 维。

每个 dimension 回答一个具体问题：在 timestep $t$、amplitude $s$ 下，如果 ring $k$ 注入能量，denoiser residual 有多少比例出现在 ring $j$？

### 3.6 Step 4: Identify（识别 source model）

#### Prototype construction

给定 N 张 reference images $\{x_0^{(i)}\}$ 由 model $M$ 生成，prototype 是 mean signature：

$$\Psi(M) = \frac{1}{N} \sum_{i=1}^{N} \Phi\big(M, E(x_0^{(i)})\big)$$

Averaging 抑制 content-dependent noise，恢复 model 在 signature space 的 intrinsic centroid。paper 用 $N=50$，因为 row normalization 已经去掉了 content-scale variation，小 N 就够。

#### Zero-training identification（argmin）

test image 来自 unknown model $M^*$，assign 到最近的 prototype：

$$\hat{M} = \underset{M \in \mathcal{M}}{\arg\min} d_{\text{cos}}\big(\Phi(M^*, E(x)), \Psi(M)\big)$$

其中 $d_{\text{cos}}(u, v) = \frac{1 - u^\top v}{\|u\| \|v\|}$。无需训练 classifier，prototype 算完就能用。

#### Classifier-based identification

当有 labeled signatures 可用时，LinearSVC 在 raw signature 上训练，比 argmin 显著更好，尤其在 model weight overlap 高时。Linear classifier 足够反映了 Section 3 的 global mean-shift 结构——model classes 在 signature space 是线性可分的，nonlinear classifier 不必要。

---

## 4. 理论 grounding：Coupling matrix 是 Jacobian 的频率域投影

paper 在 Appendix C 给了一个 clean 的 derivation，把 SDS 跟 denoiser Jacobian 正式联系起来。

### 4.1 First-order expansion

对 band-limited perturbation $\eta^{(k, s)}$（confined to ring k）做一阶 Taylor 展开：

$$\hat{\epsilon}_\theta(z_t, t, \emptyset) \approx \hat{\epsilon}_\theta(z_0, t, \emptyset) + J_\theta(z_0, t) \sqrt{1 - \bar{\alpha}_t} \eta^{(k, s)}$$

其中 $z_t = z_0 + \sqrt{1 - \bar{\alpha}_t} \eta^{(k, s)}$（这里略去 $\sqrt{\bar{\alpha}_t} z_0$ 的依赖，因为 probe noise 是叠加的）。

Output residual $\delta\hat{\eta} = \hat{\epsilon}_\theta(z_t, t, \emptyset) - \hat{\epsilon}_\theta(z_0, t, \emptyset)$ 反映 $J_\theta$ 如何把 ring-k input map 到所有 ring 的 output。

### 4.2 Coupling entry as Jacobian projection

如果 $\eta^{(k, s)}$ 在 ring k 内 isotropic，那么：

$$C[k, j] \propto \left\| P_j J_\theta(z_0, t) P_k \right\|_F^2$$

变量解释：
- $P_k$：到 ring k 的投影算子
- $P_j J_\theta P_k$：Jacobian 在 frequency domain 的 block，从 ring k 映射到 ring j
- $\|\cdot\|_F$：Frobenius norm
- $C[k, j]$：这个 block 的 squared Frobenius norm

这非常 elegant——coupling matrix 就是 Jacobian 在频率 ring pair 上的 Monte Carlo 估计。因为 $J_\theta$ 由 learned score function 决定，expected coupling $C[k, j]$ 是 model-intrinsic 的，averaging over images 后基本独立于 image content。

### 4.3 为什么 spectral 而不是 spatial

这是一个重要 intuition。直接估计 full Jacobian $J_\theta \in \mathbb{R}^{HW \times HW}$（对 64×64 latent 来说是 $4096 \times 4096$）计算上不可行。通过 frequency ring 投影把 Jacobian 压缩成 $K \times K$（这里 8×8=64），既降低了维度，又保留了 cross-band coupling 这个最有判别力的结构。

而且 spectral basis 跟 diffusion model 学习的 structure 天然对齐——[Kadkhodaie et al. 2023](https://arxiv.org/abs/2310.02557) 证明 diffusion model 学到 geometry-adaptive harmonic basis，所以 frequency-domain projection 跟 model 内部 representation 同构，不会 throw away discriminative signal。

---

## 5. 实验设置

### 5.1 八个 model 构成的 challenging closed-set

paper 故意选了三个 variation 维度交叉的 8 个 model：

**Training data variation**（fix "1.x" latent architecture）：
- SD v1.4, v1.5（共享 UNet-860M, KL-f8 VAE）
- Dreamshaper-8, Realistic Vision v5（在 1.x backbone 上 domain-specific fine-tune）
- SD v2.1（更新 data curation + OpenCLIP text encoder）

**Architecture variation**：
- SDXL（UNet-2.6B + dual CLIP encoders，参考 [Podell et al. 2023](https://arxiv.org/abs/2307.01952)）
- PixArt-α（DiT transformer + T5-XXL，参考 [Chen et al. 2023](https://arxiv.org/abs/2310.00426)）

**Training procedure variation**：
- SDXL-Turbo（共享 SDXL 架构，但用 adversarial distillation 训练，参考 [Sauer et al. 2024](https://arxiv.org/abs/2311.17042)）

这八 model 涵盖了所有 attribution 难点的组合。

### 5.2 Data

- 每个 model 用 $N=50$ images 建 prototype，测试 $N_{\text{test}}=500$ held-out images from SD prompt dataset
- Cross-domain 测试：prototype 不变，在 MS-COCO prompts 生成的 image 上测
- 6 种 strong distortion：rotation 75°, JPEG Q=25, center crop 75%, brightness ×1.5, Gaussian blur σ=8, additive noise σ=0.1

### 5.3 SDS 配置

- VAE：SD v1.5 KL-f8（512×512 input → 64×64 latent）
- Scheduler：DDIM, T=50
- $K=8$ rings, $S=3$ amplitudes $\{1.0, 1.25, 1.5\}$, $R=10$ noise repeats, $T_n=5$ timesteps
- Signature dimension: $3 \times 5 \times 64 = 960$
- LinearSVC 训在 raw signature 上，5-fold cross-validation

### 5.4 Baselines

- **LATENTTRACER**（[Wang et al. 2024](https://arxiv.org/abs/2405.13360)）：通过 invert VAE decoder 测 reconstruction fidelity
- **RONAN**（[Wang et al. 2023](https://proceedings.neurips.cc/paper_files/2023/hash/446a8ee2c8b5c7b7b7b7b7b7b7b7b7b7-Abstract-Datasets_and_Benchmarks.html)）：gradient-based input inversion，每个 candidate 跑一次优化

两者都依赖 decoder reconstruction 作为 discriminative signal。

---

## 6. 结果

### 6.1 Attribution accuracy（Section 6.1）

Figure 3 的 confusion matrix 显示：
- LinearSVC 在 $T_n=5$ 上达到 ≈100% overall accuracy
- SDXL, PixArt-α, SDXL-Turbo 完美 attribution（所有 setting）
- SD v2.1, Dreamshaper-8, Realistic Vision v5 也完美
- 唯一错误发生在 SD v1.4 ↔ SD v1.5（weight overlap 最大），LinearSVC 完全消除这个 confusion
- PCA-LR 在这个 pair 上有 ~1.5% residual confusion（PCA 压缩扔掉 marginal discriminative signal）

### 6.2 Cross-domain generalization（Section 6.2）

Table 1 的关键数据：

| Classifier | SD-Prompts | MS-COCO | Drop (pp) |
|---|---|---|---|
| Argmin† | 78.33% | 78.40% | +0.07 |
| LinearSVC | 99.98% | 96.20% | -3.80 |
| PCA-LR | 99.62% | 92.50% | -7.12 |
| Random Forest | 96.23% | 83.60% | -12.63 |
| PCA-SVM (RBF) | 91.83% | 80.30% | -11.53 |
| PCA-kNN (k=5) | 82.37% | 74.20% | -8.17 |

两个 pattern：
- **Argmin 几乎不受影响**（+0.07pp），因为 row normalization 和 unconditional conditioning 把 content-scale variation 完全去掉，prototype centroid 在 distribution change 下不 shift。这是 SDS 的核心 robustness 机制。
- **Classifier-based methods drop 更大**，LinearSVC 掉 3.78pp，其他掉 7-13pp。原因是 classifier 在 SD-Prompts signature 上学的 decision boundary 部分是 distribution-specific 的，跨 distribution 时 boundary 微 shift。LinearSVC 最 robust 因为 linear decision surface 利用 global mean-shift 结构，比 Random Forest 或 SVM-RBF 的 nonlinear boundary 稳定。

LinearSVC 在 MS-COCO 上无 retraining 达到 96.20%，证明 underlying signature 是 substantially model-intrinsic 的。

### 6.3 Robustness to image distortions（Section 6.3）

Table 2 是我最感兴趣的部分。SDS 在 6 种 strong distortion 下：

| Distortion | SD-Prompts (LinearSVC) | MS-COCO (LinearSVC) |
|---|---|---|
| Rotation (75°) | 91.9% | 90.6% |
| JPEG (Q=25) | 93.1% | 85.0% |
| Crop (75%) + resize | 90.6% | 87.5% |
| Brightness (×1.5) | 88.8% | 85.6% |
| Noise (σ=0.1) | 93.8% | 92.5% |
| Blur (σ=8) | 95.6% | 99.9% |

两个行为模式：
- **Rotation, JPEG, cropping, brightness**：基本不 degrade（≤1.3pp argmin drop，LinearSVC ≥88%），因为这些 distortion 保留 latent structure
- **Noise 和 blur**：argmin 大幅下降（61.9% 和 51.9%），但 classifier accuracy 反而更高。LinearSVC 在 MS-COCO blur 下达到 99.9%！这看似 paradox，实际是因为 strong distortion 加 isotropic noise，blurs individual-sample distance 但保留 class centroid。LinearSVC 在 centroid 上 operate，所以反而受益。

**fundamental takeaway**：SDS 测的是 denoiser spectral routing，不是 pixel statistics，所以 pixel-space distortion 不论强度都 erase 不了 fingerprint。这是 SDS 相比 output-based method 的核心优势。

### 6.4 Baseline comparison（Section 6.4）

Table 3 是 paper 的高潮：

| Model | RONAN | LatentTracer | SDS (ours) |
|---|---|---|---|
| SD v1.4 | 21.4% | 22.1% | 100.0% |
| SD v1.5 | 20.2% | 19.7% | 100.0% |
| SD v2.1 | 22.2% | 26.6% | 100.0% |
| Dreamshaper-8 | 21.2% | 22.1% | 99.8% |
| Realistic V. v5 | 20.4% | 20.9% | 100.0% |
| SDXL | 48.4% | 59.2% | 100.0% |
| PixArt-α | 55.4% | 79.0% | 100.0% |
| SDXL-Turbo | 35.0% | 67.0% | 100.0% |
| **Overall** | **30.5%** | **44.8%** | **99.9%** |

两个 baseline 在共享 VAE 的五个 1.x model 上全部 collapse 到 20-27%（random baseline 是 20%），因为 shared decoder 产生 identical reconstruction loss。它们在 autoencoder 不同的 model（SDXL, PixArt-α, SDXL-Turbo）上 work（35-79%），但在最关键的 near-clone case 上完全失效。

SDS 比 LATENTTRACER 高 55pp，比 RONAN 高 70pp，整个 gap 都在最 matter 的 model pair 上。这证明了一个 general principle：**attribution signal 应该在最 varies across models 的组件上找（denoiser），不是在 shares 跨 model 的组件上找（decoder）**。

### 6.5 Ablation

**K 和 S 的 ablation**（Table 4，$T_n=1$）：
- $K=2$：即使 $S=3$ 也只有 77.4%（near chance for 8-way），coarse 频率 partition 抓不到 cross-band routing 差异
- $K=4, S=3$：97.2%
- $K=6, S=3$：99.6%（saturation）
- $K=8, S=3$（baseline）：99.9%

$S$ 的 effect 一致：每个 $K$ 下，加 amplitude 提升 10-25pp，$S=2$ 到 $S=3$ 有 diminishing return，$S=3$ 是 practical optimum。

**N 的 ablation**（Figure 5）：
- $N=8$：LinearSVC 在 SD-Prompts 84.0%，MS-COCO 80.7%（已经远超 12.5% random）
- $N=25$：SD-Prompts 94.8%，MS-COCO 92.4%
- $N=50$：SD-Prompts 99.9%，MS-COCO 96.2%

SD-Prompts 和 MS-COCO 的 gap 在所有 N 下稳定 ~3-4pp，确认 cross-domain drop 是 fixed distribution shift，不是 amplified estimation error。

---

## 7. 更深层的 intuition 和联想

### 7.1 Mean-shift 结构的意义

Section 3 的 mean-shift 发现很有意思：class centroids 线性可分，但 individual samples 在 local neighborhood 重叠。这告诉我们 spectral coupling signature 不是在做"tight cluster"——它在做"centroid shift"。

为什么是这样？因为 signature 是 Jacobian 结构的 Monte Carlo 估计，而 Jacobian 依赖 $z_0$（image content）。同一 model 不同 image 的 Jacobian 会变，所以 individual signature 有 content-dependent 偏移。但 model 的 inductive bias（反映在 Jacobian 的频率结构上）是 image-independent 的，所以 centroid 稳定。

这跟 [Teng et al. 2025 FINGERINV](https://openaccess.thecvf.com/content/CVPR2025/papers/Teng_Fingerprinting_Denoising_Diffusion_Probabilistic_Models_CVPR_2025_paper.pdf) 的关键区别就在这里：FINGERINV 的 signature 高度 tied 到 chosen image，必须先 enroll；SDS 因为 row normalization 和 averaging，把 content 抽掉，得到 model-intrinsic centroid，可以 generalize 到任意 generated image。

### 7.2 SDS 跟 Stein's identity 的潜在联系

Score function $\nabla \log p(z)$ 的 Hessian $\nabla^2 \log p(z)$ 在 score-based model 理论里是核心对象。Stein's identity 告诉我们：

$$\mathbb{E}_{z \sim p}\left[\nabla^2 \log p(z) + \nabla \log p(z) \nabla \log p(z)^\top\right] = 0$$

Hessian 的结构跟 data distribution 的 covariance 直接挂钩。SDS 的 coupling matrix 本质是 Hessian（通过 Jacobian）在频率 ring 上的 projection，所以它捕获的正是 distribution 的 cross-frequency covariance——这是 model-specific 的。

### 7.3 跟神经网络 fingerprinting 的关系

传统分类 model 的 fingerprinting 工作（[IPGuard](https://arxiv.org/abs/2104.04066), [conferrable adversarial examples](https://arxiv.org/abs/1912.00888), [universal adversarial perturbations](https://arxiv.org/abs/2204.07707)）利用 decision boundary geometry 或 adversarial example 构造 model-specific signature。SDS 把这个 idea 推广到 generative model——但 generative model 没有 decision boundary，所以要从 score function 的 geometry 替代。

[Quan et al. 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Quan_Fingerprinting_Deep_Image_Restoration_Models_ICCV_2023_paper.pdf) 在 image restoration model 上用 critical points near performance boundary 做 fingerprinting，跟 SDS 的精神类似——都是找 model 对 controlled perturbation 的 distinctive response。

### 7.4 为什么 row normalization 这么关键

paper 的 cross-domain generalization 和 distortion robustness 几乎完全来自 row normalization（Eq. 5）。把它去掉会怎样？粗略想：absolute energy 在 image brightness 上 heavily dependent，所以 signature 会随 image content 变，prototype centroid 在 distribution shift 下 shift 大。

Row normalization 把"绝对能量"换成"能量分布"，把 content scale 这个 confounding variable 干掉。这跟 batch normalization 在 supervised learning 里的作用有哲学相似性——都是 normalize 掉 nuisance variation 让 underlying structure 显现。

### 7.5 SDS 的 evasion 难度

paper 在 conclusion 里提了一个点：要 evade SDS，adversary 需要 modify image 的 latent representation 来 mimic 另一个 model 的 denoising geometry。这比 pixel-level post-processing 难得多——你需要在 frequency domain 上 fine-grained 地调整 denoiser 对所有 frequency ring 的 response pattern。

这可能跟 [Hu et al. 2024](https://arxiv.org/abs/2405.07145) 对 stable signature 的 attack 形成有趣对比——stable signature 可以通过 modify decoder 被 remove，但 SDS 因为 probe denoiser 本身，attacker 即使修改 output 也 erase 不了 denoiser 内部的 spectral geometry。

### 7.6 限制和未解问题

paper 自己列了几个：
1. **Closed-set**：source model 必须在 registry 里，open-set attribution 是 open problem
2. **White-box access**：需要 candidate model weights，对 proprietary API 不适用
3. **不 detect tampering**：只能回答"哪个 model 生成的"，不能回答"image 是否被 post-modified"

这些都指向自然的 extension：
- Open-set 可以用 distance threshold 或 outlier detection 在 signature space 做
- API setting 可以考虑 query-based fingerprinting（用特定 query pattern 探测 API response）
- Tamper detection 可以跟 [DetectGPT](https://arxiv.org/abs/2301.11305) 类似思路结合——用 perturbation response 的 consistency 检测修改

### 7.7 跟 model interpretability 的联系

SDS 的 mean-shift 结构提示一个 broader 想法：spectral coupling signature 可能 generalize 到其他 model analysis 任务——detecting fine-tuning, quantization artifacts, distillation。paper 在 conclusion 里提到这个。

直觉上，fine-tuning 改变 score function 的 high-frequency 细节（因为 fine-tune 通常在小 data 上，改变 texture-level inductive bias）；quantization 改变 Jacobian 的 numerical structure；distillation（SDXL-Turbo 用 adversarial distillation）改变 denoising trajectory 的 geometry。这些都应该在 spectral coupling 上 leave detectable imprint。

---

## 8. 总结：SDS 在 landscape 里的位置

SDS 的核心贡献是把 diffusion model attribution 从"看输出"转到"看 denoiser 内部"，并且具体到"看 denoiser 在频率域如何 redistribute energy"。

几个关键 design decision 让它 work：
1. **Probe denoiser 而不是 decoder**——避开 shared VAE 的 collapse 问题
2. **Spectral domain 而不是 spatial domain**——跟 diffusion model 学习的 harmonic basis 对齐
3. **Band-limited noise 而不是 full-spectrum noise**——隔离 cross-band coupling 这个最 discriminative 的结构
4. **Row normalization**——去除 content-dependent scale，enable cross-domain generalization
5. **Multi-timestep + multi-amplitude aggregation**——probe orthogonal aspect of spectral geometry
6. **Empty text conditioning**——去除 prompt-dependent signal，保留 intrinsic geometry

实验上 99.9% accuracy 在 8-model closed-set，96.2% cross-domain，对 6 种 distortion 鲁棒，比基线高 55-70pp，整个 gap 在最难的 near-clone case 上。

paper 的 broader message 是：**generative model 的 identity 编码在 score function 的 spectral geometry 里**。这跟 Kadkhodaie et al. 关于 diffusion model generalization 来自 geometry-adaptive harmonic representation 的工作形成互补——diffusion model 学到 harmonic basis，而不同 model 在这个 basis 上的 cross-band coupling 是 model-specific 的。

future work 的 natural direction：open-set attribution、query-based API attribution、跟 tamper detection 结合、扩展到其他 generative model family（autoregressive, VAE, flow-based）、研究 adversarial attack 和 defense。paper 提供的 [code](https://github.com/Pragati-Meshram/SGS) 是进一步探索的好起点。

---

## References 关键链接

- [SDS paper code](https://github.com/Pragati-Meshram/SGS)
- [Kadkhodaie et al. 2023 - Generalization in diffusion models](https://arxiv.org/abs/2310.02557)
- [Song et al. 2020 - Score-based generative modeling via SDE](https://arxiv.org/abs/2011.13456)
- [Song et al. 2020 - DDIM](https://arxiv.org/abs/2010.02502)
- [Rombach et al. 2022 - Stable Diffusion / Latent Diffusion](https://arxiv.org/abs/2112.10752)
- [Podell et al. 2023 - SDXL](https://arxiv.org/abs/2307.01952)
- [Chen et al. 2023 - PixArt-α](https://arxiv.org/abs/2310.00426)
- [Sauer et al. 2024 - SDXL-Turbo / Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042)
- [Wang et al. 2024 - LATENTTRACER](https://arxiv.org/abs/2405.13360)
- [Wang et al. 2023 - RONAN](https://proceedings.neurips.cc/paper_files/2023/hash/446a8ee2c8b5c7b7b7b7b7b7b7b7b7b7-Abstract-Datasets_and_Benchmarks.html)
- [Teng et al. 2025 - FINGERINV](https://openaccess.thecvf.com/content/CVPR2025/papers/Teng_Fingerprinting_Denoising_Diffusion_Probabilistic_Models_CVPR_2025_paper.pdf)
- [Fernandez et al. 2023 - Stable Signature](https://arxiv.org/abs/2303.10137)
- [Hu et al. 2024 - Stable Signature attack](https://arxiv.org/abs/2405.07145)
- [Cao et al. 2021 - IPGuard](https://arxiv.org/abs/2104.04066)
- [Lukas et al. 2019 - Conferrable adversarial examples](https://arxiv.org/abs/1912.00888)
- [Peng et al. 2022 - Universal adversarial perturbations for fingerprinting](https://arxiv.org/abs/2204.07707)
- [Quan et al. 2023 - Fingerprinting image restoration models](https://openaccess.thecvf.com/content/ICCV2023/papers/Quan_Fingerprinting_Deep_Image_Restoration_Models_ICCV_2023_paper.pdf)
- [Corvi et al. 2023 - Intriguing properties of synthetic images](https://openaccess.thecvf.com/content/CVPR2023/papers/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_CVPR_2023_paper.pdf)
- [Xu et al. 2025 - Detecting origin attribution](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Detecting_Origin_Attribution_for_Text-to-Image_Diffusion_Models_WACV_2025_paper.pdf)
