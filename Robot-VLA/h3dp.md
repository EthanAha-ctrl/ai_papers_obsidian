---
source_pdf: h3dp.pdf
paper_sha256: 001c23bf688c29c3fd1827dc67efc643f907683cf0544962c5a2470ccb382e34
processed_at: '2026-08-04T23:18:58-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# H³DP 大白话版

## 一句话概括

机器人看着东西动手这件事，现有方法都是"眼睛归眼睛，手归手"，两边各干各的。这篇 paper 说：**眼睛看东西的方式和手动的节奏应该对上**——粗看的时候决定大方向，细看的时候精修小动作。

## 问题出在哪

Diffusion Policy 是现在最主流的机器人学习方法。它干两件事：先用一个 CNN 把图片变成一串 feature，再用一个 diffusion model 从噪声里"去噪"出机器人该怎么动。这两个模块之间只有一个很松的连接——CNN 输出直接 concat 进 diffusion 网络，至于"哪个 feature 该在 denoising 的哪一步起作用"，没人管。

结果就是：网络可能用一张 96×96 的 feature map 同时指导"先把手伸到杯子附近"（粗规划）和"手指精确捏住杯把"（细动作）这两个完全不同粒度的决策。feature 是混在一起的，diffusion 也只能囫囵吞枣。

人类不是这样工作的。你伸手拿杯子的时候，眼睛先扫一眼全局"杯子在桌上左边"，然后视线聚焦"杯把朝右"，手指才精确调整。**视觉的 coarse→fine 和 motor 的 coarse→fine 是天然对齐的**。H³DP 就是把这个对齐显式建出来。

## 三个层次的设计

paper 标题里的 "Triply-Hierarchical" 就是三层 hierarchy，我一层层说。

### 第一层：把 RGB-D 图片按深度切片

RGB-D 相机给你一张彩色图加一张深度图。传统做法是把深度当第四个 channel 拼到 RGB 后面，变成 4-channel image 喂给 CNN。问题是 CNN 不知道"第四个 channel 是几何信息而不是颜色"，往往学得很差——paper 里 DP with depth 的性能甚至跟普通 DP 差不多（Table 1: 52.8 vs 48.1，提升微乎其微）。

H³DP 的做法：**按深度把图片切成 N 层**。近处的物体一层，远处背景一层，中间再分几层。每一层都是一张纯 RGB 图片，只是只保留对应深度范围的像素，其他位置补零。

公式 Eq.(1) 看着吓人，其实意思很简单：近处分得细，远处分得粗。因为机器人操作的 workspace 通常在前面 0.3-1 米，这个范围内 10cm 的深度差很重要；背景 2 米和 3 米的差异其实无所谓。所以用一个二次方程让近处 layer 边界窄、远处边界宽。

切完之后，每一层 RGB 单独喂一个 encoder。网络现在不用猜"哪里是前景哪里是背景"了——layer 0 就是前景，layer N 就是背景，结构是显式的。

这一层解决的是"输入结构化"。

### 第二层：每一层都抽出多分辨率的 feature

光把图片分层还不够。H³DP 对每一层用 VQ-VAE 抽出 K 个不同 resolution 的 feature map。比如 MetaWorld 用 K=4，分辨率是 1×1、3×3、5×5、7×7。

1×1 那个相当于一个 global CLS token，编码"整个场景是个什么情况"。7×7 那个保留 spatial detail，编码"物体具体在哪个位置"。中间两个是过渡。

具体做法是先 encode 到最高分辨率，然后 interpolate 到低分辨率，再用 VQ-VAE 的 codebook 量化（强制使用离散 token，类似把 feature "翻译"成有限 vocabulary 里的词），最后再 interpolate 回高分辨率加 detail recovery CNN。这是个 Laplacian pyramid 风格的分解——粗 scale 抓低频，细 scale 抓高频。

这一层解决的是"特征多尺度"。

### 第三层：denoising 分阶段，每阶段用对应尺度的 feature

这是 paper 的核心。Diffusion 一共 T 步去噪，H³DP 把 T 步切成 K 个阶段。比如 T=50, K=4, boundaries 是 {0, 20, 30, 40, 50}：

- 第 40-50 步（最 noisy）：用 1×1 的 global feature，决定 action chunk 的粗轮廓
- 第 30-40 步：用 3×3 feature，确定大致 spatial layout
- 第 20-30 步：用 5×5 feature
- 第 0-20 步（接近 clean）：用 7×7 fine feature，精修细节

为什么这么做？因为 diffusion 本质是从低频到高频的"频谱自回归"——噪声里先长出来的是低频成分（action chunk 的总体趋势），高频成分（每一步的细微调整）是后面才补上的。Sander Dieleman 2024 年那篇博客把这事讲得很透。H³DP 在 action 上实测了这个现象（Figure 3 的 DFT 分析），确认 action 跟 image 一样有这个 inductive bias。

既然 diffusion 自己也是 coarse→fine 走的，你给它 condition 也按 coarse→fine 提供，两边节奏对上，自然比一直塞同一个 feature 强。

训练时有个 tricky 的地方：loss 只在最细 scale $\hat{f}_K$ 上算 diffusion loss，靠 consistency loss 把其他 scale 拉齐。这样训练效率高，不用给每个 stage 单独 sample 训练数据。

## 三个层次怎么协同

depth layering 让输入有结构 → multi-scale encoding 让特征有结构 → hierarchical denoising 让 action 生成也有结构。三个结构都对齐到同一个 hierarchy axis 上：**从 coarse 到 fine，从低频到高频，从全局到局部**。

去掉任何一层，性能都掉 10+ 个点（Table 3）。三个一起上，相对 DP 提升平均 27.5%。

## 跟其他方法的差别

- **DP** [1]：visual encoder + diffusion，单 scale feature，松连接。H³DP 的直接 baseline。
- **DP3** [4]：用 point cloud 替代 image，3D 表示更强，但需要高质量 depth sensor 和理想 segmentation。真实世界 segmentation 一崩就完蛋。
- **CARP** [16]：action side 做 multi-scale（用 VQ-VAE 量化 action 成多尺度 token，autoregressive 生成）。H³DP 是 perception side 做 multi-scale，action 仍是连续 diffusion。前者有量化损失，后者保精度。
- **VAR** [48]：image generation 的 multi-scale autoregressive。H³DP 借鉴了 VAR 的 multi-scale 思想，但搬到了 visuomotor policy 的 visual encoding 上。

## 实验亮点

- 44 个 simulation task 平均提升 27.5%，variance 也显著降低（说明更鲁棒）
- ManiSkill Deformable task（可变形物体操作）提升特别大：59.3 vs DP 22.3，因为可变形物体需要精细 force/shape tracking，multi-scale feature 帮助很大
- Real-world 4 个 task（含 2 个 long-horizon bimanual）平均 +32.3%
- Long-horizon task（Pour Juice 要 4 个 subtask）上 +41%，因为 multi-scale 天然能 disambiguate "现在该执行哪个 subtask"
- Inference speed 24 FPS（asynchronous 设计），DP 是 12 FPS

## 工程细节

real-world 部署有几个实用 trick：
- **Asynchronous inference**：推理和 action execution 并行，predicted action 进 queue
- **Temporal ensembling**：借鉴 ACT，对 overlapping action chunks 做 weighted average 减少 jitter
- **p-masking**：训练时随机 mask 掉 proprioception，强制网络早期依赖 vision。因为 proprioception MLP 容易学，CNN 难学，不 mask 的话网络会 shortcut 到 proprioception

## 我的直觉

这篇 paper 的核心 insight 其实很简洁：**diffusion 是从低频到高频的频谱自回归，所以 condition 也应该按频谱层次提供**。这个 insight 一旦说破就觉得很自然，但之前没人把它在 visuomotor policy 上完整 wire 起来。

三个 hierarchy 设计看似各自独立，其实都是同一个 inductive bias 的不同侧面：
- Depth layering 是 spatial 维度的结构化（前景/背景）
- Multi-scale representation 是 feature 维度的结构化（低频/高频）
- Hierarchical denoising 是 action 维度的结构化（粗略/精细）

三者对齐到同一个 frequency axis 上，让 perception 和 action 在 frequency domain 显式 align。这就是 paper 标题 "Triply-Hierarchical" 的真正含义。

## Limitations

- Diffusion 本身慢，asynchronous 24 FPS 对高频率控制仍不够，可能需要 distill 成 consistency model
- ZED 相机 depth 质量有限，限制了 layering 效果
- N（层数）是手调的，没有 principled 选择方法
- Codebook 大小和利用率没分析
- Real-world 只有 4 个 task，validation 偏少
- 训练时只用最细 scale 算 loss，训练-推理 mismatch 可能存在

## 跟大趋势的关系

2024-2025 robot learning 有几个并行 trend：3D representation（DP3 系）、hierarchical action（CARP 系）、diffusion 加速（Consistency Policy 系）、VLA（π0/OpenVLA 系）。H³DP 没站任何一个队的极端，而是选了一个被忽视的 angle——**perception 和 action 在 frequency domain 的对齐**。这个 angle 在 long-horizon 和 deformable task 上显示出独特优势，可能跟 VLA 结合也很有想象空间：VLM 的 token hierarchy 跟 action hierarchy 对齐，说不定是下一个 paper。

---

# H³DP: Triply-Hierarchical Diffusion Policy 深度解读

## 1. 核心论点与 positioning

H³DP 由 Tsinghua IIIS / Shanghai Qi Zhi / Shanghai AI Lab 的团队提出（第一作者 Yiyang Lu, Yufeng Tian, Zhecheng Yuan），核心 thesis 非常清晰：**visuomotor policy 的 perception 和 action generation 之间存在 coupling 不足的问题**。现有方法（Diffusion Policy [1], DP3 [4], Consistency Policy [6], CARP [16]）要么单独 refine perception，要么单独 refine action generation，却忽视了让 visual feature 在 frequency / granularity 维度上与 action 的生成阶段对齐。

H³DP 把这个 coupling 用一个三层 hierarchy 显式地 wired up：
- **Input level**: depth-aware layering（RGB-D 分层）
- **Representation level**: multi-scale VQ-VAE 风格的 visual features
- **Action level**: 分 stage 的 denoising，每个 stage condition 在对应 scale 的 visual feature 上

Project page: https://lyy-iiis.github.io/h3dp/

这个想法的来源 paper 在 Introduction 里写得很直白——人类 visual cortex 是 hierarchical processing 的（引用了 Hubel & Wiesel 1962, Lee & Mumford 2003, Bill et al. 2022 的工作 [11-14]），机器人也应该模拟这种 perception→motor 的 hierarchical 处理。

我的 intuition：这篇 paper 的关键 contribution 不在于单点的 architectural trick，而在于把一个**已经在 image diffusion（U-Net multi-scale）和 image autoregression（VAR [48]）里被证明 work 的 inductive bias**——即"从 coarse 到 fine、从 low-frequency 到 high-frequency"——首次完整地迁移到 visuomotor action 生成上，并且把 visual encoding 也对齐到同一个 hierarchy 上。这是 DP 系列（[1, 4, 6, 7, 8]）和 CARP [16] / Dense Policy [15] / ARP [50] 这些"只 hierarchy action"或"只 refine perception"的工作之间真正的空白。

## 2. Depth-Aware Layering：把 RGB-D 从"concat"升级为"显式结构"

### 2.1 公式与变量

Equation (1) 把每个 pixel 的 depth $d$ 映射到 layer index $m$：

$$
m = \left\lfloor -0.5 + 0.5\sqrt{1 + 4(N+1)(N+2)\frac{d - d_{\min}}{d_{\max} - d_{\min} + \epsilon}} \right\rfloor
$$

变量含义：
- $m$：分配的 layer index，整数，取值范围 $[0, N]$
- $N$：超参，layer 总数（ablation Table 4 显示 $N=3$ 或 $N=4$ 最优，$N=6$ 反而退化）
- $d$：当前 pixel 的 depth 值（相机坐标系下的距离）
- $d_{\min}, d_{\max}$：scene 的 depth 最小/最大值（动态计算 per frame）
- $\epsilon$：小常数（防除零）
- $\lfloor\cdot\rfloor$：floor operation

### 2.2 为什么是这种 discretization

这个公式来源于 Zhang et al. 的 MonoDETR [52]（https://arxiv.org/abs/2305.19529）。它的形式来自解二次方程 $m(m+1)/2 = (N+1)(N+2) \cdot \text{normalized\_depth}$，等价于让**layer 边界随 depth 线性增加而变宽**。

直觉：在 robotic manipulation 场景下，workspace 通常在前方 0.3m–1.0m，物体之间 depth 差异很小但很重要；背景（>2m）depth 差异大但语义无关。Linear-increasing discretization 让**近处分得细、远处分得粗**，对齐了机器人的 task-relevant region 分布。

对比方案：paper 在 Appendix E.3 用 GMM-based layering 做了对比（H³DP-GMM），结果 Table 12 显示 GMM 把 H³DP 从 64.8 拉到 47.2，基本退化到 N=1（50.3）的水平。这说明**显式的 geometric inductive bias 比 data-driven 的 soft clustering 更好**，因为 robotic data 中 depth 分布不均衡，GMM 容易把大部分 pixel 塞进同一个 mode。

### 2.3 这一步的本质

把 RGB-D 从"4-channel image"转换为"(N+1) × 3-channel image stack"，本质上是一个 **structural tokenization**：让网络不必从 RGB+D 的 raw correlation 里发现"前景 vs 背景"的概念，而是显式被告知"这是 layer 0（最近），这是 layer N（最远）"。

这一点跟 DP3 [4] 的 philosophy 形成有趣对比。DP3 用 point cloud 直接保留 3D 结构，但代价是：
1. 需要 high-fidelity depth sensor（RealSense L515 级别，ZED 在 paper 的 real-world 实验里质量不够，见 Appendix E.8 Table 17）
2. 需要 ideal segmentation 移除 task-irrelevant points（见 Appendix E.6 Table 15，DP3 去 segmentation 后从 82.8 掉到 54.3）

H³DP 用 depth-aware layering 取得了类似的"foreground-background 分离"效果，但**不需要 segmentation**，直接吃 raw RGB-D。这是工程上很重要的 win——在真实部署时，seg 失败是 DP3 类方法的最大 failure mode。

参考链接：
- DP3: https://dp3.cs.columbia.edu/
- DP3 paper: https://arxiv.org/abs/2403.03954
- 关于 RGB-D concat 为何 work 不好：[21] https://arxiv.org/abs/2402.02500

## 3. Multi-Scale Visual Representation：用 VQ-VAE 做 hierarchical feature pyramid

### 3.1 Encoding 流程（Algorithm 1）

对每个 layer $I_m$（$m=0,\dots,N-1$），用独立的 encoder $\mathcal{E}_m$（VQGAN [63] 架构）编码到最高 resolution feature $f_m \in \mathbb{R}^{h_K \times w_K \times C}$。然后对每个 scale $k=1,\dots,K$：

1. Interpolate $f_m$ 到 $(h_k, w_k)$ resolution 得到 $f_{m,k}$
2. VQ-VAE 量化（Equation 2）：
$$
f_{m,k}^{(i,j)} \gets \arg\min_{z \in \mathcal{Z}_m} \|z - f_{m,k}^{(i,j)}\|_2
$$
   - $f_{m,k}^{(i,j)} \in \mathbb{R}^C$：位置 $(i,j)$ 的 feature vector
   - $\mathcal{Z}_m \in \mathbb{R}^{V \times C}$：layer m 的 codebook，V 是 codebook size
   - $\arg\min$：欧氏距离最近邻
3. Interpolate 回 $(h_K, w_K)$ 最高 resolution
4. 过一个轻量 CNN $\phi_{m,k}$ 恢复 fine detail
5. 累加得到 $\hat{f}_{m,k} = \sum_{k' \le k} f_{m,k'}$
6. 减掉已用部分 $f_m \gets f_m - f_{m,k}$（这是 LAP / residual pyramid 的思想）

最终输出 $F = \{\hat{f}_k = \{\hat{f}_{m,k}\}_{m=0}^{N-1}\}_{k=1}^K$。

### 3.2 Resolution 配置

Table 5–8 给出各 benchmark 的具体配置。例如 MetaWorld 用 $K=4$，resolutions $\{(1,1), (3,3), (5,5), (7,7)\}$，stage boundaries $\{0, 0.4, 0.6, 0.8, 1.0\} \cdot T$。也就是说：
- $k=1$：1×1 spatial，全局 scene context（类似 CLS token）
- $k=2$：3×3，coarse spatial layout
- $k=3$：5×5，medium
- $k=4$：7×7，fine detail

Stage boundaries 是**非线性分配**：早期 40% timesteps 用 $k=1$ 的全局特征，后面每个 stage 各占 20%。这跟 image diffusion 里"低频信号先恢复"的频率分布匹配——大部分 structural 信息集中在低频，需要更多 denoising steps 去 commit。

### 3.3 Consistency Loss（Equation 3）

$$
\mathcal{L}_{\text{consistency}} = \sum_{m=0}^{N-1} \sum_{k=1}^{K} \left( \|\hat{f}_{m,k} - \text{sg}(f_m)\|_2^2 + \beta \|f_m - \text{sg}(\hat{f}_{m,k})\|_2^2 \right)
$$

变量：
- $\hat{f}_{m,k}$：经过 quantization + interpolation 的 multi-scale representation
- $f_m = \mathcal{E}_m(I_m)$：encoder 直接输出的最高 resolution feature
- $\text{sg}(\cdot)$：stop-gradient operator
- $\beta$：超参，balance 两个方向的 gradient flow

这是 VQ-VAE [34, 37] 的标准 commitment + codebook loss 的 multi-scale 扩展形式：
- 第一项让 quantized feature 逼近 encoder output（commitment）
- 第二项让 encoder output 逼近 quantized feature（codebook update）
- $\text{sg}$ 防止双向 gradient 互相抵消

Intuition：这个 loss 强制每个 scale 的 representation 都与原 feature 保持信息一致，从而**避免低 resolution scale 丢失过多信息**。没有这个 loss，$\hat{f}_{m,1}$（1×1）可能 collapse 成 trivial vector，无法为早期 denoising 提供有用 condition。

### 3.4 与 VAR [48] / CARP [16] 的对比

VAR（Visual AutoRegressive, NeurIPS 2024, https://arxiv.org/abs/2404.02905）做 image generation 的核心 insight 是：autoregressive 应该按 **scale** 而不是按 **token** 生成。VAR 用 multi-scale VQ-VAE 把 image tokenize 成多 resolution 的 discrete tokens，然后 GPT-style autoregressive 生成。

CARP [16]（https://arxiv.org/abs/2412.06782）把 VAR 这个 idea 搬到 action 上：用 multi-scale VQ-VAE 编码 action sequence，autoregressive 生成。

H³DP 的 multi-scale representation 设计明显借鉴了 VAR/CARP，但有一个 critical difference：**H³DP 是 multi-scale 在 visual side，不是 action side**。Action 仍然是连续 diffusion 生成的（Equation 5 是 standard DDPM/DDIM），multi-scale 体现在 visual condition 上。这避免了 VQ-VAE 量化 action 的 information loss（对精细 control 不友好），同时保留了 hierarchy 的 inductive bias。

Table 13 直接对比 CARP：H³DP 在 7 个 MetaWorld 任务上 86.3 vs CARP 67.4，平均提升 18.9%。这个 gap 说明：把 hierarchy 放在 perception 而非 action 上，对 visuomotor 任务可能更优。

## 4. Hierarchical Action Generation：diffusion 的 spectral autoregressive 视角

### 4.1 Inference（Equation 4–5）

总 denoising timesteps $T$ 分成 $K$ 个 stages，$\cup_{k=1}^K (\tau_{k-1}, \tau_k]$。当 $t \in (\tau_{k-1}, \tau_k]$ 时：

$$
\epsilon^t = \epsilon_\theta^{(t)}(a^t \mid \hat{f}_k, q)
$$

变量：
- $a^t$：timestep t 的 noisy action chunk
- $\hat{f}_k$：scale k 的 multi-scale visual feature（condition）
- $q$：robot proprioception（pose 等）
- $\epsilon_\theta^{(t)}$：denoising network（参数共享，但 condition 随 stage 变化）

Reverse step：

$$
a^{t-1} = \sqrt{\alpha_{t-1}}\left(\frac{a^t - \sqrt{1-\alpha_t}\cdot\epsilon^t}{\sqrt{\alpha_t}}\right) + \sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\epsilon^t + \sigma_t\tilde{\epsilon}^t
$$

变量：
- $\alpha_t$：noise scheduler 参数（cosine schedule, Equation 7: $\alpha_t = f(t)/f(0)$, $f(t) = \cos^2(\frac{\pi}{2}\frac{t/T + s}{1+s})$）
- $\sigma_t$：stochasticity 参数
  - DDIM (MetaWorld/Adroit/DexArt)：$\sigma_t = 0$，deterministic ODE
  - DDPM (ManiSkill/RoboTwin)：$\sigma_t = \sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}}$，VP SDE
- $\tilde{\epsilon}^t \sim \mathcal{N}(0, \mathbf{I})$：random Gaussian noise

### 4.2 Training Loss（Equation 6）—— 这里有个巧思

$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{a^0, \epsilon, t}\left[\gamma_t \|\epsilon_\theta^{(t)}(\sqrt{\alpha_t}a^0 + \sqrt{1-\alpha_t}\epsilon \mid \hat{f}_K, q) - \epsilon\|^2\right]
$$

注意：**训练时只用最高 resolution 的 $\hat{f}_K$ 作为 condition**！所有 scale 都通过 gradient 从 $\hat{f}_K$ 流回 encoder + codebook。

为什么这样 work？
1. Consistency loss 已经强制各 scale 与原始 $f_m$ 一致，所以 $\hat{f}_k$ 之间是 correlated 的
2. 推理时不同 scale 的 feature 对应不同 frequency 的 action component，但训练时不必显式监督各 scale
3. 训练效率高——不需要 sample 不同 $t$ stage 分别训，每次 forward 都过整个 hierarchy

$\gamma_t = 1$（Equation 10）—— 简化 weighting，因为 $\gamma_t$ 不影响最优 denoising network $\epsilon_{\theta^*}$。

### 4.3 Spectral analysis：核心 evidence

Section 4.1.3 和 Figure 3 是这篇 paper 最 compelling 的实验。作者对 action chunk 在 denoising 过程的中间结果做 DFT，可视化其频谱随 $t$ 演化。

观察：
- $t = \tau_4$（最 noisy）：频谱几乎 flat（Gaussian noise）
- $t = \tau_3 \to \tau_2$：低频成分先 emerge
- $t = \tau_1 \to \tau_0$：高频成分逐步补充

这印证了 Sander Dieleman 在 "Diffusion is spectral autoregression"（https://sander.ai/2024/09/02/spectral-autoregression.html）中提出的观点：**diffusion process 本质上是从低频到高频的 autoregressive 生成**。Image diffusion 早就观察到这个现象，paper [27-29] 都有讨论。

H³DP 的 contribution 在于：**这个 bias 在 action 上同样存在**，并且可以被 visual condition 的 multi-scale structure 显式利用。Coarse visual feature（low resolution）自然 align 低频 action component，fine visual feature align 高频 action component，二者在对应 stage 互相 reinforce。

我的 intuition：这是 paper 的 conceptual core。其他两个 hierarchy（depth layering, multi-scale representation）单独看都不算全新——depth layering 借鉴 MonoDETR，multi-scale VQ-VAE 借鉴 VAR——但**把它们和 diffusion 的 spectral inductive bias 显式 align**，是这个 paper 的真正 novel insight。

## 5. 整体架构图解读（Figure 2）

Figure 2 的 pipeline：

```
RGB-D image I
    ↓ depth-aware layering (Eq 1)
{I_0, I_1, ..., I_N} (N+1 layers)
    ↓ 各自独立 encoder E_m (VQGAN)
{f_0, f_1, ..., f_{N-1}} (highest-res features)
    ↓ interpolate + VQ-VAE + CNN refine
Multi-scale features {f_{m,k}}_{m,k}
    ↓ 累加 pyramid
{\hat{f}_k = {\hat{f}_{m,k}}_m}_{k=1}^K
    ↓ 作为 condition 喂入 diffusion
Diffusion denoising (T steps, K stages)
    ↓ t ∈ (τ_{k-1}, τ_k] uses \hat{f}_k
Action chunk a^0
```

关键设计点：
1. **Encoder per layer**：每个 depth layer $I_m$ 有独立 encoder $\mathcal{E}_m$ 和独立 codebook $\mathcal{Z}_m$。这避免了不同 depth 层 feature 在 codebook 里互相 interference。
2. **Codebook size V**：在 VQ-VAE 工作里通常 V=1024 或 8192，paper 没明说具体值（应该在 0.7M 参数 budget 内推算，结合 DINOv2 对比 Table 14，DINOv2 ViT-S 是 21M，H³DP encoder < 0.7M）。
3. **End-to-end training**：encoder + codebook + CNN refine + denoising network 联合训，loss = $\mathcal{L}_{\text{diffusion}} + \alpha \mathcal{L}_{\text{consistency}}$（Equation 11）。

## 6. 实验数据深入分析

### 6.1 主表（Table 1）

| Method | Avg over 44 tasks |
|---|---|
| H³DP | 75.6 ± 18.6 |
| DP3 | 59.3 ± 24.9 |
| DP (w/ depth) | 52.8 ± 22.2 |
| DP | 48.1 ± 23.1 |

H³DP vs DP3 = +27.5% relative improvement。注意 std：H³DP 18.6 < DP3 24.9 < DP 23.1。**H³DP 不仅 mean 高，variance 也显著降低**。这表明 hierarchical structure 提升的不是"平均性能"，而是"鲁棒性"——在不同 task 难度上都有稳定表现。

具体 task 上有几个值得注意：
- MetaWorld Hard++ (5 tasks)：H³DP 95.8 vs DP3 88.4 vs DP 58.0。Hard++ 是 MetaWorld 中最难的子集，H³DP 几乎 saturate。
- ManiSkill Deformable (4 tasks)：H³DP 59.3 vs DP3 26.5 vs DP 22.3。Deformable object（如 hang pour, fill excavate）需要精细 force control 和 shape tracking，H³DP 的 multi-scale visual feature 在这里特别 effective。
- RoboTwin (8 tasks)：H³DP 57.4 vs DP3 45.9。Bimanual 任务需要两个 arm 的协调，action chunk 维度更高，frequency spectrum 更复杂，hierarchical conditioning 优势明显。

### 6.2 Ablation（Table 3）

去掉每个 hierarchy 组件后的 average performance：
- Full H³DP: 59.6
- w/o depth layering: 46.5（−13.1）
- w/o hierarchical action: 49.0（−10.6）
- w/o multi-scale representation: 48.7（−10.9）
- DP (w/ depth) baseline: 42.1

三个组件 contribution 比较均衡（10–13 points each），**没有冗余设计**。Combined effect（59.6 - 42.1 = 17.5）小于单独 sum（13.1+10.6+10.9 = 34.6），说明三者**部分功能 overlap**——都从不同角度强化 perception-action coupling，所以不是 fully additive。

### 6.3 Layer number N（Table 4）

- N=1: 46.5（等价于无 layering，只有 multi-scale + hierarchical action）
- N=2: 50.2
- N=3: 59.6 ← optimum
- N=4: 59.5
- N=5: 54.6
- N=6: 49.0

Sweet spot 在 3-4。N 过小无法充分分离 foreground/background，N 过大导致每个 layer 信息 sparse，codebook 利用率下降。

### 6.4 Real-world experiments

四个 task：Clean Fridge, Pour Juice, Place Bottle, Sweep Trash。Figure 4 显示 H³DP 全胜 DP，平均 +32.3%。

特别关注 Pour Juice 和 Sweep Trash 这两个 long-horizon task：
- PJ: 4 subtasks（place cup → scoop powder → fill water → insert straw），随机化包括 cup position (7×7 cm²), juice powder color, dispenser position
- ST: pick broom → sweep debris → empty to trash bin，trash randomized over 40×40 cm²

H³DP 在 PJ 上 +41% over DP，在 ST 上 long-horizon 表现也 strong。Long-horizon 关键在于 visual feature 能 disambiguate 当前 stage——multi-scale representation 在这里很 natural：粗 scale 判断"现在该执行哪个 subtask"，细 scale 提供 local 控制。

Instance generalization（Table 2）：在 Place Bottle / Sweep Trash 上换 instance（coke → sprite → can，64cm³ → 216cm³ trash），H³DP 66.2 vs DP 50.8，+15.4%。这印证 multi-scale 的泛化能力——粗 scale 抓 object category，细 scale 抓具体 geometry。

### 6.5 Inference speed（Table 9, 18）

Real-world asynchronous: H³DP 24.2 FPS vs DP 12.4 / DP3 12.7。Asynchronous 设计（Section D.1）让 inference 和 action execution 并行，predicted action 进 queue 以固定 12Hz 执行。

Simulation（Table 18）：DP 11.1, DP3 12.2, H³DP 12.0 FPS。**H³DP 的额外 overhead 几乎可忽略**——multi-scale encoding 和 depth layering 都是轻量操作，瓶颈仍是 diffusion 本身。

## 7. 关键 engineering tricks（Appendix D）

### 7.1 Temporal ensembling

借鉴 ACT [2]（https://tonyzhaozh.github.io/aloha/）。Asynchronous inference 产生 overlapping action chunks，对同一 timestep 的多个 prediction 做 weighted average，减少 jitter。

### 7.2 p-masking

训练时 stochastic mask 所有 proprioception input，概率 $p(t) = 1 - t/T$（linear decay）。

Intuition：proprioception MLP 简单易优化，CNN 处理 RGB-D 复杂难优化。Without masking，网络会 shortcut 到 proprioception，忽视 vision。p-masking 强制网络早期训阶段依赖 vision，建立 visual grounding，后期再允许用 proprioception。

这个 trick 适用于所有 DP-based method，是 real-world deployment 的实用经验。

### 7.3 Pre-trained ResNet18 for RGB

Long-horizon real-world task 用预训 ResNet18 encoder 处理 RGB modality（Section 4.2.1）。这跟 simulation 用 from-scratch encoder 不同。预训 encoder 提供 generic visual prior，提升 real-world 泛化。

## 8. Limitations 和 future directions

Paper Section 6 自承：
1. Diffusion inference 慢——可考虑 distill 成 consistency model [6] 或 shortcut model [8]
2. ZED camera depth 质量限制——更高 quality depth sensor 可进一步提升

我的额外观察：
1. **Codebook collapse 风险**：VQ-VAE 训练中 codebook 利用率不均是个常见问题，paper 没讨论 codebook usage metric。可考虑 EMA codebook update [34] 或 entropy regularization。
2. **N=3 是固定值**：不同 task 的最优 N 可能不同，adaptive N 选择是个潜在改进方向。
3. **Multi-scale 只在 spatial 维度**：可扩展到 temporal scale——长 horizon action chunk 的 multi-scale temporal representation，可能进一步提升 long-horizon task 性能。
4. **跟 VLA 的结合**：π0 [3]（https://arxiv.org/abs/2410.24164）和 OpenVLA 用 VLM 做 action prediction。H³DP 的 hierarchical conditioning 思想可以注入 VLA，让 VLM 的 token-level hierarchy 跟 action hierarchy align。
5. **Depth sensing**：单目 depth estimation（如 Depth Anything, https://arxiv.org/abs/2401.10891）可能让 H³DP 在只有 RGB 的 setting 下也 work，扩大适用范围。
6. **跟 3D-Actor [32] / 3D Diffuser Actor [32] 的对比**：这些方法用 3D scene representation 做 policy diffusion，跟 H³DP 的 RGB-D multi-scale 路线不同。Bridging 两者是 future work。

## 9. 我的整体评价

### 9.1 Strengths

1. **Conceptual novelty**：把"diffusion 是 spectral autoregression"这个 insight 从 image 严格迁移到 action，并用 multi-scale visual condition 显式 exploit，这是真正的新 idea。
2. **Engineering integration**：三个 hierarchy 组件协同设计，从 input → representation → output 全链路 align，paper 的 Figure 2 把这个 pipeline 画得很清楚。
3. **Extensive validation**：44 simulation tasks + 4 real-world tasks + 充分 ablation + spectral analysis + GMM 对比 + DP3 segmentation 对比 + DINOv2 对比 + CARP 对比——evidence 非常 thorough。
4. **No segmentation needed**：相比 DP3 需要理想 segmentation，H³DP 直接吃 raw RGB-D，工程上更 deployable。

### 9.2 Weaknesses / Open questions

1. **N 的选择是 hand-tuned**：paper 给出 N=3 或 4 最优，但没给 principled 选择方法。不同 workspace depth 分布应该需要不同 N。
2. **Codebook size V 未明确**：VQ-VAE 的 codebook 大小对性能影响大，paper Appendix 没详述。
3. **Codebook utilization 未分析**：dead code 是 VQ-VAE 常见问题，paper 没给 codebook usage 数据。
4. **Real-world 只有 4 个 task**：相对 simulation 的 44 task 规模，real-world validation 偏少。需要更多 long-horizon 和 dexterous task 验证。
5. **Inference speed limitation**：asynchronous 24 FPS 仍不及 consistency policy [6] 或 one-step diffusion [7] 的水平，对高频率 control 仍不足。
6. **Hierarchical action training 只用 $\hat{f}_K$**：训练-推理 mismatch 是 potential issue。虽然 consistency loss 缓解，但若 explicit multi-scale training（per-stage supervise）可能进一步提升。
7. **跟 ACT 的 temporal ensemble 依赖**：asynchronous + ensemble 是工程 patch，没从 architecture 上解决 jitter。

### 9.3 更广的 positioning

H³DP 在 2024-2025 这个时间点出现，处于几个 trend 的交汇：
- **3D representation for policy**：DP3, 3D Diffuser Actor, RVT-2 都在用 3D token 替代 2D image token
- **Hierarchical action generation**：CARP, Dense Policy, ARP 都在试 hierarchical action，但都局限在 action side
- **Diffusion acceleration**：Consistency Policy, ManiCM, One-Step DP, Shortcut Models 都在做 diffusion 加速
- **VLA**：π0, OpenVLA, OpenVLA-OFT 把 LLM/VLM 范式带入 robot control

H³DP 选了一个不同的 angle——**不追求 3D representation 的极致，也不追求 action hierarchy 的极致，而是强化 perception 和 action 在 frequency domain 的 alignment**。这个 angle 在 long-horizon 和 deformable task 上显示出独特优势。

## 10. 关键参考链接汇总

- **H³DP project page**: https://lyy-iiis.github.io/h3dp/
- **Diffusion Policy (Chi et al. 2023)**: https://diffusion-policy.cs.columbia.edu/
- **DP3 (Ze et al. RSS 2024)**: https://dp3.cs.columbia.edu/ , https://arxiv.org/abs/2403.03954
- **CARP (Gong et al. 2024)**: https://arxiv.org/abs/2412.06782
- **VAR (Tian et al. NeurIPS 2024)**: https://arxiv.org/abs/2404.02905
- **VQ-VAE (van den Oord et al. 2017)**: https://arxiv.org/abs/1711.00937
- **VQ-VAE-2 (Razavi et al. 2019)**: https://arxiv.org/abs/1906.00446
- **VQGAN (Esser et al. 2021)**: https://arxiv.org/abs/2012.09841
- **MonoDETR (Zhang et al. 2023)**: https://arxiv.org/abs/2305.19529
- **ACT (Zhao et al. 2023)**: https://tonyzhaozh.github.io/aloha/ , https://arxiv.org/abs/2304.13705
- **Consistency Policy (Prasad et al. 2024)**: https://arxiv.org/abs/2405.07503
- **π0 (Black et al. 2024)**: https://arxiv.org/abs/2410.24164
- **Dense Policy (Su et al. 2025)**: https://arxiv.org/abs/2503.13217
- **Diffusion is spectral autoregression (Sander Dieleman blog)**: https://sander.ai/2024/09/02/spectral-autoregression.html
- **DINOv2 (Oquab et al. 2023)**: https://arxiv.org/abs/2304.07193
- **RoboTwin (Mu et al. 2024)**: https://arxiv.org/abs/2409.02920
- **RoboVerse (Geng et al. 2025)**: https://github.com/RoboVerseOrg/RoboVerse
- **ManiSkill2 (Gu et al. 2023)**: https://arxiv.org/abs/2302.04659
- **MetaWorld (Yu et al. 2020)**: https://arxiv.org/abs/1910.10897
- **DexArt (Bao et al. CVPR 2023)**: https://arxiv.org/abs/2304.05098
- **DDPM (Ho et al. 2020)**: https://arxiv.org/abs/2006.11239
- **DDIM (Song et al. 2021)**: https://arxiv.org/abs/2010.02502
- **Score SDE (Song et al. 2021)**: https://arxiv.org/abs/2011.13456

---

如果你想 build 进一步 intuition，我建议你重点玩味 Figure 3 的 spectral analysis——这是整篇 paper 的 conceptual keystone。理解了"diffusion 是从低频到高频的 spectral autoregression"，三个 hierarchy 的设计就都 naturally 推出来了：depth layering 提供 spatial 结构的多 layer 输入，multi-scale VQ 提供 frequency-aware visual condition，hierarchical denoising 让 condition 和 action frequency 演化对齐。这篇 paper 的 elegant 之处在于：**三个看似独立的设计，背后是同一个 inductive bias 的多面体现**。
