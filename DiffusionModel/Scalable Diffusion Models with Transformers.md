---
source_pdf: Scalable Diffusion Models with Transformers.pdf
paper_sha256: 5fbb0ec35e3ae76826240171ca63ed4c27de6cb43180367453019e9d28a8b076
processed_at: '2026-08-12T03:13:56-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiT 用人话说一遍

## 故事的开头

2022年底，Peebles 和 Xie 这俩人盯着一件事发愣：**全世界做 diffusion model 的人都在用 U-Net，但好像没人认真问过"凭什么是 U-Net？"**

你看 NLP 早就被 transformer 统一了，vision recognition 也被 ViT 拿下了，连 RL、meta-learning 都在用 transformer。唯独 image generation 这个圈子，大家还在死守着 U-Net 不放。这个 U-Net 是 2020 年 Ho et al. 做 DDPM 的时候从 PixelCNN++ 那里顺手抄过来的，中间夹杂了 ResNet block、塞了几层 self-attention、用 GroupNorm 注入 condition。后来 ADM（Dhariwal & Nichol）虽然做了大量 ablation，但 high-level 结构动都没动。

这俩人就琢磨：**U-Net 的那些 inductive bias——局部性、平移等变性、U 形 long skip connection——到底对 diffusion 有多大帮助？换个最朴素的标准 transformer 行不行？**

答案出乎意料地干净：行，而且效果更好。这篇 paper 就是来证明这件事的。

paper: <https://arxiv.org/abs/2212.09748>
code: <https://github.com/facebookresearch/DiT>

---

## 先理一下 diffusion 在干嘛

### 前向：往图片里不断加噪声

给定一张干净图 $x_0$，在 timestep $t$ 时它变成什么样，有 closed-form：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon_t$$

- $x_0$：原始 clean image
- $x_t$：第 $t$ 步的 noisy 版本
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$：累积保留信号比例，单调递减，$t$ 越大信号越弱
- $\epsilon_t \sim \mathcal{N}(0, \mathbf{I})$：标准高斯噪声

**直觉**：$t=0$ 是原图，$t=t_{\max}$ 是纯噪声。中间任意一步都可以一步采出来，不用一步步跑马尔可夫链。

### 反向：训练一个网络预测噪声

反向过程就是学一个网络 $\epsilon_\theta(x_t, t)$，输入 noisy 图和当前 timestep，输出"我猜加进去的噪声是什么"。loss 简化成：

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}\Big[ \big\| \epsilon_\theta(x_t, t) - \epsilon_t \big\|_2^2 \Big]$$

**关键 insight**：网络不需要直接生成图像，只要预测"减去多少噪声"。这是个回归任务，和用什么 backbone 没关系——这就是 DiT 能换掉 U-Net 的理论前提。

### Classifier-free guidance：推理时把条件方向"放大"

条件模型训 $p_\theta(x_{t-1} \mid x_t, c)$，$c$ 是 class label。训练时随机把 $c$ 替换成 null token $\emptyset$。推理时：

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \big( \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \big)$$

- $s$：guidance scale，$s=1$ 是标准采样，$s>1$ 把"有条件 vs 无条件"的差异放大
- 直觉：$\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x)$，guidance 就是把后一项加权

### Latent Diffusion：别在 pixel space 折腾

$256 \times 256 \times 3$ 直接训 diffusion 太贵。LDM 先训一个 VAE：$z = E(x)$，把图片压成 $32 \times 32 \times 4$ 的 latent（downsample 8×），然后在 $z$-space 训 diffusion。最后 $x = D(z)$ 解码回 pixel。

**DiT 就是接在 frozen VAE 后面的那个 diffusion backbone**，VAE 完全不动。所以整个 pipeline 是 convolutional VAE + transformer DDPM 的混合体——卷积的"遗产"还在 VAE 里，但所有"生成智能"都交给了 transformer。

---

## DiT 长什么样

整个 forward pass 一句话讲完：

```
noised latent z [32×32×4]
  → 切 patch → 一串 token
  → 加位置编码
  → N 个 DiT block（每个 block 都注入 t 和 c）
  → 线性 decode 回 [32×32×8]（4 通道噪声 + 4 通道 log-variance）
```

这就是个标准 ViT，几乎没动过。

### Patchify：最关键的"旋钮"

把 $32 \times 32 \times 4$ 的 latent 按 patch size $p$ 切成 $T = (32/p)^2$ 个 token：

| patch size $p$ | token 数 $T$ | 含义 |
|---|---|---|
| 8 | 16 | token 少，便宜 |
| 4 | 64 | 中等 |
| 2 | 256 | token 多，贵但效果好 |

**这里有个反直觉的发现**：减小 $p$ 不会增加 parameter（patch embed 就是个线性层），但 Gflops 会涨——因为 self-attention 是 $O(T^2 d)$。所以你可以**不动参数量，光靠减小 patch size 把 Gflops 推上去**。论文后面会证明这件事对 sample quality 至关重要。

位置编码用的是 ViT 最朴素的 sine-cosine，没有 learned PE、没有 RoPE、没有 conv stem。Karpathy 你应该 appreciate 这种"少即是多"——为了保留 transformer 的 scaling property，尽量别乱改。

### 四种 condition 注入方式（paper 最有教学意义的部分）

diffusion model 必须把 timestep $t$ 和 class label $c$ 喂给网络。他们试了四种方案：

**方案一：in-context conditioning**

把 $t$ 和 $c$ 各自 embed 成一个 token，append 到 image token 序列前面。这就跟 ViT 的 `[CLS]` token 一个思路。
- 优点：标准 ViT block 一行代码不用改
- 缺点：condition 只能通过 self-attention 间接影响所有 token，路径太长

**方案二：cross-attention**

把 $t, c$ 当另一条长度 2 的序列，每个 block 在 self-attention 后面再加一个 cross-attention 让 image tokens attend 到 condition。
- 优点：表达力强，LDM 原版就这么干
- 缺点：额外 15% Gflops

**方案三：adaptive LayerNorm (adaLN)**

标准 LayerNorm 是 $\text{LN}(h) = \gamma \odot \frac{h - \mu}{\sigma} + \beta$，其中 $\gamma, \beta$ 是 learnable per-channel 参数。

adaLN 把 $\gamma, \beta$ 变成 condition 的函数：
$$\gamma, \beta = \text{MLP}\big(\text{embed}(t) + \text{embed}(c)\big)$$

每个 token 在每个 block 都会根据 $(t, c)$ 整体做 affine 变换。
- 优点：Gflops 几乎可以忽略不计，condition 直接作用到每个 token 的每个 channel
- 这其实是 StyleGAN、BigGAN 用过的 FiLM 范式在 diffusion 上的移植

**方案四：adaLN-Zero（最终采用的方案）**

在 adaLN 基础上，再回归一个 per-channel scaling $\alpha_1, \alpha_2$，分别乘到 attention 输出和 MLP 输出进入 residual 之前：

$$h \leftarrow h + \alpha_1 \cdot \text{Attn}(\text{adaLN}(h))$$
$$h \leftarrow h + \alpha_2 \cdot \text{MLP}(\text{adaLN}(h))$$

**关键初始化**：产生 $\alpha_1, \alpha_2$ 的 MLP 最后一层权重全部置零，这样整个 block 在训练初期就是 identity function。

这个 trick 是从 ResNet 的 "zero-init final BN scale" 和 diffusion U-Net 的 "zero-init final conv in each block" 偷来的。让 residual block 初始化为 identity，训练初期梯度直接 short-circuit 回去，loss 不会爆炸。

**实验结果（Figure 5，400K iter，DiT-XL/2）**：

| Block 设计 | Gflops | FID |
|---|---|---|
| in-context | 119.4 | ~68 |
| cross-attention | 137.6 | ~50 |
| adaLN | 118.6 | ~35 |
| **adaLN-Zero** | **118.6** | **~19.5** |

adaLN-Zero 同时是 Gflops 最少、FID 最低、训练最稳的方案。**这是整篇 paper 最 actionable 的发现**：conditioning mechanism 比模型大小重要得多——in-context 与 adaLN-Zero 用同样的 backbone 容量，FID 差了 3 倍多。

### 模型大小：照搬 ViT config

| Model | Layers $N$ | Hidden $d$ | Heads | Gflops ($p=4$) | Params |
|---|---|---|---|---|---|
| DiT-S | 12 | 384 | 6 | 1.4 | 33M |
| DiT-B | 12 | 768 | 12 | 5.6 | 130M |
| DiT-L | 24 | 1024 | 16 | 19.7 | 458M |
| DiT-XL | 28 | 1152 | 16 | 29.1 | 675M |

最终 SOTA 用 DiT-XL/2：$p=2$ 把 Gflops 推到 118.6。

---

## 核心发现：scaling law

### 发现一：Gflops 和 FID 强相关，parameter count 不是好代理

他们训了 12 个模型（4 个 size × 3 个 patch size），都在 400K iter 处测 FID。看几个典型对比：

| Model | Params | Gflops | FID-50K @400K |
|---|---|---|---|
| DiT-S/2 | 33M | 6.06 | 68.40 |
| DiT-B/4 | 130M | 5.56 | 68.38 |
| DiT-L/2 | 458M | 80.71 | 23.33 |
| DiT-XL/4 | 675M | 29.05 | 43.01 |
| **DiT-XL/2** | **675M** | **118.64** | **19.47** |

**关键观察**：DiT-S/2（33M 参数，6 Gflops）和 DiT-B/4（130M 参数，5.6 Gflops）FID 几乎一样。DiT-L/2（458M 参数，80 Gflops）比 DiT-XL/4（675M 参数，29 Gflops）FID 好得多——**参数多 50% 的模型反而更差，因为 Gflops 少**。

直觉解释：扩散 backbone 的 sample quality 由 forward pass 的计算量决定，参数量只是"计算量的容器"之一。同样 Gflops 下，小模型多跑几次 forward 和大模型少跑几次 forward，效果差不多。

### 发现二：大模型 compute-efficient

把 FID 作为总训练 compute（model Gflops × batch × steps × 3）的函数画出来，大模型在每个 compute 预算下都更优。小模型训再久也追不上大模型——这跟 Chinchilla 的结论一致。

### 发现三：sampling compute 不能替代 model compute

Diffusion 有个独特"漏洞"：推理时可以多跑几步。他们测了 16/32/64/128/256/1000 步。

**结论**：DiT-L/2 用 1000 步（80.7 Tflops/sample）的 FID，仍然劣于 DiT-XL/2 用 128 步（15.2 Tflops/sample，5× 更少 compute）。

直觉：每步去噪的"质量上限"由 backbone 决定。迭代次数再多也无法突破这个上限。**部署时宁愿训大模型少跑步，也不愿训小模型多跑步**。

---

## SOTA 结果

### ImageNet 256×256（训 7M steps）

| Model | FID↓ | IS↑ | Precision↑ | Recall↑ |
|---|---|---|---|---|
| BigGAN-deep | 6.95 | 171.4 | 0.87 | 0.28 |
| StyleGAN-XL | 2.30 | 265.12 | 0.78 | 0.53 |
| ADM-G + ADM-U | 3.94 | 215.84 | 0.83 | 0.53 |
| LDM-4-G (cfg=1.50) | 3.60 | 247.67 | 0.87 | 0.48 |
| **DiT-XL/2-G (cfg=1.50)** | **2.27** | **278.24** | 0.83 | **0.57** |

DiT-XL/2 同时拿到最低 FID 和最高 Recall。这是 GAN vs Diffusion 长期对决的分水岭——diffusion 在所有指标上全面碾压 GAN。

### ImageNet 512×512（训 3M steps）

| Model | FID↓ | Gflops |
|---|---|---|
| ADM-G + ADM-U | 3.85 | 2813 |
| **DiT-XL/2-G (cfg=1.50)** | **3.04** | **524.6** |

512 分辨率下 DiT 用 1/5 的 Gflops 打败 ADM。compute 效率提升非常夸张。

---

## 训练 recipe 里几个反直觉的细节

- AdamW，lr=$10^{-4}$，**constant schedule，no warmup，no weight decay**
- batch size 256，没有 dropout，没有 Mixup，没有 CutMix
- 唯一 data augmentation：horizontal flip
- EMA decay 0.9999
- **训练全程没看到 loss spike**，高度稳定

Karpathy 你应该 appreciate：早期 ViT 训练需要大量 augmentation + warmup + LARS 才稳，这里什么都没用。**diffusion 的 per-step noise injection 本身就是极强的 regularization**，让 transformer 训练变得出奇地容易。这点很反直觉，但后来 SD3、Sora 都验证了。

---

## 这件事为什么重要：DiT 的"辐射"

DiT paper 本身只做 ImageNet class-conditional，看起来像个工程论文，但它改变了整个生成模型社区的认知。后续几乎所有重要工作都站在它肩膀上：

**Sora (OpenAI, 2024)**：把 DiT 用在 video 上，spacetime patches 作为 token，adaLN-Zero 注入 text + timestep。OpenAI 在 technical report 里明确说这是 DiT 的扩展。
ref: <https://openai.com/research/video-generation-models-as-world-simulators>

**Stable Diffusion 3 (2024)**：backbone 从 U-Net 换成 MM-DiT，text 和 image token 在同一个 transformer 里 self-attention，依然用 adaLN-Zero 和零初始化。
ref: <https://arxiv.org/abs/2403.03206>

**PixArt-α / PixArt-Σ**：高效 text-to-image DiT，不过 condition 是文本所以用 cross-attention 而不是 adaLN。
ref: <https://arxiv.org/abs/2310.00626>

**Latte**：video DiT，时空 patch 化。
ref: <https://arxiv.org/abs/2401.03048>

**U-ViT**：清华团队的 concurrent work，思路类似，arXiv 时间还更早一点。
ref: <https://arxiv.org/abs/2209.12152>

**MDT (Masked Diffusion Transformer)**：在 DiT 上加 masked modeling 加速训练。
ref: <https://arxiv.org/abs/2306.14075>

**SiT (同一作者后续)**：把 DiT 推广到更一般的 interpolant 框架，连接 diffusion 和 flow matching。
ref: <https://arxiv.org/abs/2401.01622>

**EDM2 (Karras, 2024)**：在 DiT 之后做了更精细的 scaling law 分析，显示 fixed Gflops 下 parameter 和 token 数有更微妙的平衡。
ref: <https://arxiv.org/abs/2406.07596>

---

## 几个你可能想质疑的点

**第一，Gflops 真的是 universal scaling axis 吗？**

DiT 只在 ImageNet class-conditional 上验证。当 condition 变成自然语言（SD3、Sora），cross-attention 反而比 adaLN 更合适——因为 condition 是变长序列而不是单个 label。所以 adaLN-Zero 不是 universally best，只是在 class-conditional 设定下 best。

**第二，patchify 没有 free lunch**

减小 $p$ 增加 Gflops 不增加 parameter 听起来很划算，但 self-attention 是 $O(T^2)$。$p=2$ 在 $256 \times 256$ 上 $T=256$ 还好，到 $512 \times 512$ 是 $T=1024$，到 video 就爆炸。后续工作（Sora）需要 window attention、spatiotemporal factorization 来缓解。DiT paper 没讨论这个，因为 ImageNet 256/512 还撑得住。

**第三，VAE 的卷积遗产**

DiT 在 latent space 操作，VAE 仍然是卷积的。严格来说"U-Net inductive bias 不必要"只适用于 diffusion backbone 部分，整个生成 pipeline 里卷积还在。如果直接在 pixel space 训 DiT（像 ADM 那样），Gflops 会高到无法承受，但论文没做这个对照。

**第四，scaling law 缺 power-law 拟合**

DiT 没有像 Chinchilla 那样给出 $\text{FID} \propto \text{Gflops}^{-\alpha}$ 的具体指数。后续 EDM2 补上了这个分析。

---

## 最该带走的 intuition

1. **架构统一在生成模型里也成立**——diffusion 的 backbone 可以脱掉 U-Net 这个历史包袱，拥抱 transformer 的 scaling property。

2. **Conditioning mechanism 是 critical design choice**，比增加 parameter 重要得多。adaLN-Zero 同时拿到最低 Gflops 和最低 FID，是罕见的"免费午餐"。

3. **Zero-init residual block 是训深 transformer 的通用 trick**——把网络初始成 identity function，让训练初期梯度直通。

4. **Forward pass Gflops 是扩散 backbone 的关键 scaling axis**，比 parameter count 更相关。推理 compute 和训练 compute 在 DiT 里通过 Gflops 统一了。

5. **Sampling compute 不能替代 model compute**——大模型少步数 > 小模型多步数。部署性价比的判断就靠这条。

6. **Diffusion 的 noise injection 自带 regularization**，让 transformer 训练异常稳定，省掉 ViT 常用的 warmup/augmentation。很反直觉，但很真实。

7. **Patchify 是 DiT 的 tokenization 入口**，$p$ 是最重要 hyperparameter。后续所有 video DiT 工作都沿用这个 abstraction。

---

## 一句话总结

DiT 表面上是"换个 backbone"的工程论文，实际上改变了社区对扩散模型架构的根本认知：**transformer 在生成模型里同样可以 scale，U-Net 的时代就此结束**。从 Sora 到 SD3，整个 2024–2026 生成模型架构统一的浪潮，起点就是这篇 paper。

---

# Scalable Diffusion Models with Transformers (DiT) 深度解读

## 一、Motivation 与历史背景

这篇论文的核心问题非常简单且尖锐：**扩散模型为什么一定要用 U-Net？** 

自 Ho et al. 2020 的 DDPM 以来，整个 diffusion 社区几乎默认把 convolutional U-Net 当作 backbone。这个 U-Net 的设计其实是从 PixelCNN++ 那里"继承"过来的——其中夹杂了 ResNet block、在低分辨率插入少量 spatial self-attention、用 adaptive GroupNorm 注入 condition。Dhariwal & Nichol (ADM, 2021) 对这个 U-Net 做了大量消融，但 high-level 结构没动过。

Karpathy 你一定记得，transformer 已经把 NLP、ViT、RL、meta-learning 都"统一"了，但 image-level **生成** 模型（diffusion、GAN）几乎都还在用卷积。Peebles & Xie 想验证一个 hypothesis：**U-Net 的 inductive bias（局部性、平移等变性、long skip connection 的 U 形拓扑）对扩散模型性能不是必要的**，可以被一个标准的、几乎没改过的 transformer 替换，并继承 transformer 的 scalability。

这个工作发表在 ICCV 2023，arXiv 2212.09748。后续它直接成为 **Stable Diffusion 3、Sora、PixArt-Σ、OpenAI Sora、Facebook Chameleon 部分组件** 的 backbone——可以说 DiT 是 2023–2026 生成模型架构统一的真正起点。

Paper link: <https://arxiv.org/abs/2212.09748>
Project page: <https://www.wpeebles.com/DiT>
Code (official): <https://github.com/facebookresearch/DiT>

---

## 二、Diffusion 数学准备（公式逐项拆解）

### 2.1 Forward process

DDPM 假设一个加噪马尔可夫链，给定真实数据 $x_0$：

$$q(x_t \mid x_0) = \mathcal{N}\big(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\, (1 - \bar{\alpha}_t)\, \mathbf{I}\big)$$

- $x_0 \in \mathbb{R}^{H \times W \times 3}$：原始 clean image
- $x_t$：在 timestep $t$ 下的 noisy version
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$：累积 noise schedule 系数，单调递减
- $\alpha_s$：每步保留信号的比例，工程上常用 linear 或 cosine schedule

重参数化得到 closed-form 采样：

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})$$

这里 $\epsilon_t$ 是标准高斯噪声。**关键 insight**：因为 $q$ 是高斯，整个前向过程无需神经网络，可以一步直接生成任意 $t$ 的 noisy sample。

### 2.2 Reverse process 与训练目标

反向过程建模：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(\mu_\theta(x_t), \Sigma_\theta(x_t)\big)$$

通过 ELBO 推导，最终训练目标简化为：

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon_t}\Big[ \big\| \epsilon_\theta(x_t, t) - \epsilon_t \big\|_2^2 \Big]$$

- $\epsilon_\theta$：神经网络，输入 noisy $x_t$ 和 timestep $t$，预测加进去的噪声
- 论文同时学 covariance $\Sigma_\theta$，用完整 KL 项训练它（沿用 Nichol & Dhariwal 2021 的方法）

**Intuition**：网络不需要直接生成图像，只需要预测"减去多少噪声"——这是一个相对容易的回归任务。$\epsilon_\theta$ 就是一个条件去噪器，与 backbone 选择无关，这是 DiT 能替换 U-Net 的理论前提。

### 2.3 Classifier-free guidance

条件扩散模型 $p_\theta(x_{t-1} \mid x_t, c)$，其中 $c$ 是 class label。训练时随机 dropout $c$（用 learnable null embedding $\emptyset$ 替代），推理时用：

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \big( \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \big)$$

- $s > 1$：guidance scale，把条件方向"放大"
- $s = 1$：标准采样
- 直观理解：在 score function $\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x)$ 中，把 $\log p(c \mid x)$ 项加权

这篇论文在 appendix 里有一个很妙的 trick：**只对 latent 的前 3 个 channel 做 guidance**（不对第 4 个 channel 做）。三通道 guidance scale $1+x$ 近似于四通道 guidance scale $1 + \frac{3}{4}x$。这是个未被解释的有趣现象。

### 2.4 Latent Diffusion

直接在 pixel space 训 diffusion 在 $256 \times 256$ 以上几乎不可承受。LDM (Rombach et al., CVPR 2022) 用两阶段：

1. 训练 VAE：$z = E(x)$，$x = D(z)$
2. 在 $z$-space 训 diffusion

Stable Diffusion 的 VAE 对 $256 \times 256 \times 3$ 输入，输出 $32 \times 32 \times 4$ 的 latent（downsample factor 8）。这个 latent space 把 Gflops 压缩了 64 倍，是 DiT 能"scale"起来的关键。

DiT 直接接在预训练 VAE 后面，VAE 完全 frozen，只训 transformer。所以整套系统是 **convolutional VAE + transformer DDPM** 的混合体——这部分保留了卷积的局部性，但所有"生成智能"都给了 transformer。

---

## 三、DiT 架构详解

### 3.1 整体 forward pass

输入：noised latent $z \in \mathbb{R}^{32 \times 32 \times 4}$（对应 $256 \times 256 \times 3$ image）。

整个 pipeline：

```
z [32×32×4]
  → Patchify (patch size p) → tokens [T, d] where T=(32/p)²
  → + sine-cosine positional embedding
  → N × DiT block (with conditioning t, c)
  → final adaLN + Linear decoder
  → reshape → output [32×32×8]  (4 channels noise + 4 channels log-variance)
```

### 3.2 Patchify（关键的"tokenization"）

给定 patch size $p$，将 $I \times I \times C$ 的 latent 切成 $T = (I/p)^2$ 个 patch，每个 patch 通过线性层映射到 hidden dimension $d$。

| $p$ | $T$ (for $I=32$) | 注释 |
|-----|-----|-----|
| 8 | 16 | 最少 token |
| 4 | 64 | 中等 |
| 2 | 256 | 最多 token |

**核心 insight**：减小 $p$ 不会改变 parameter count（patch embed 只是线性层），但 Gflops 会 $(1/p)^2$ 增长——因为 self-attention 是 $O(T^2 d)$。这是论文发现的最重要 scaling 轴之一：**通过 token 数量增 Gflops，比通过 depth/width 增 Gflops 更高效**。

Patchify 后加的是 ViT 标准的 sine-cosine positional embedding，没有用 learned position embedding，没有用 RoPE，没有用 conv stem。Karpathy 你应该 appreciate 这种"少即是多"的设计——为了保留 transformer 的 scaling property，尽量不动架构。

### 3.3 四种 conditioning block 设计（论文最有 teaching value 的部分）

DiT 必须把 timestep $t$ 和 class label $c$ 注入 transformer。论文系统比较了四种方案：

#### (a) In-context conditioning
把 $t$ 和 $c$ 各自 embed 成一个 token，append 到 image token 序列前。这相当于 ViT 的 `[CLS]` token 思路。**优点**：标准 ViT block 不需任何改动。**缺点**：condition 只能通过 self-attention 间接影响所有 token，inductive bias 太弱。

#### (b) Cross-attention block
把 $t, c$ 拼成另一条长度为 2 的序列，在标准 self-attention 后面加一个 multi-head cross-attention layer 让 image tokens attend 到 condition。**优点**：表达力强，是 LDM 原版用的方案。**缺点**：额外引入约 15% Gflops。

#### (c) Adaptive Layer Norm (adaLN)

标准 LayerNorm：
$$\text{LN}(h) = \gamma \odot \frac{h - \mu}{\sigma} + \beta$$
其中 $\gamma, \beta$ 是 learnable per-channel 参数。

adaLN 把 $\gamma, \beta$ 变成 **condition 的函数**：
$$\gamma, \beta = \text{MLP}\big(\text{embed}(t) + \text{embed}(c)\big)$$

这样每个 token 在每个 block 都会根据 $(t, c)$ 整体 affine 变换。**优点**：Gflops 几乎可忽略（只是 per-channel scale/shift）；condition 直接作用到每个 token 的每个 channel。这是 GAN 里 StyleGAN、BigGAN 用过的 FiLM 范式在 diffusion 上的迁移。

#### (d) adaLN-Zero（最终方案）

在 adaLN 基础上，**再回归一个 per-channel scaling $\alpha_1, \alpha_2$**，分别乘到 attention 输出和 MLP 输出进入 residual 之前：

$$h \leftarrow h + \alpha_1 \cdot \text{Attn}(\text{adaLN}(h));\quad h \leftarrow h + \alpha_2 \cdot \text{MLP}(\text{adaLN}(h))$$

**关键初始化**：把产生 $\alpha_1, \alpha_2$ 的 MLP 最后一层权重置零，使整个 block 在训练初期是 **identity function**。

这是从 ResNet 的"zero-init final BN scale"和 diffusion U-Net 的"zero-init final conv in each block"传统迁移来的。Karpathy 你应该熟悉这种 trick：让 residual block 初始化为 identity，可以让超深网络初期 loss 不爆炸，梯度直接 short-circuit 回去。

**实验对比（Figure 5, 400K iter, DiT-XL/2）**：

| Block design | Gflops | FID |
|---|---|---|
| in-context | 119.4 | ~68 |
| cross-attention | 137.6 | ~50 |
| adaLN | 118.6 | ~35 |
| **adaLN-Zero** | **118.6** | **~19.5** |

adaLN-Zero 同时是 Gflops 最少、FID 最低、训练最稳的方案。**这是 paper 里最 actionable 的发现**：conditioning mechanism 比模型大小更重要——in-context 与 adaLN-Zero 同样的 backbone 容量，FID 差了 3 倍。

### 3.4 模型配置

完全照搬 ViT 的 config，加一个 XL：

| Model | Layers $N$ | Hidden $d$ | Heads | Gflops ($I=32, p=4$) | Params |
|---|---|---|---|---|---|
| DiT-S | 12 | 384 | 6 | 1.4 | 33M |
| DiT-B | 12 | 768 | 12 | 5.6 | 130M |
| DiT-L | 24 | 1024 | 16 | 19.7 | 458M |
| DiT-XL | 28 | 1152 | 16 | 29.1 | 675M |

最终 SOTA 用 DiT-XL/2：$p=2$ 把 Gflops 推到 **118.6 Gflops**。

### 3.5 Decoder

最后一个 DiT block 后：final adaLN → Linear 把每个 token 映射到 $p \times p \times 2C$（$C=4$ 通道 latent，输出 4 通道 $\epsilon$ + 4 通道 $\log\sigma^2$）→ reshape 回 $32 \times 32 \times 8$。这步非常 ViT-style，没有用任何 deconvolutional decoder。

---

## 四、Scaling Law 的核心发现

这是 paper 的"灵魂"部分。

### 4.1 Gflops vs FID 强相关

论文训练了 12 个 DiT 变体（4 个 size × 3 个 patch size），都在 400K iter 处测 FID-50K。**关键观察**：

> 不同 config 但 Gflops 相近的模型 FID 也相近。例如 DiT-S/2（6.06 Gflops）与 DiT-B/4（5.56 Gflops）FID 接近，DiT-B/2（23.0 Gflops）与 DiT-L/4（19.7 Gflops）接近。

这意味着：**对于扩散 backbone，决定样本质量的是 forward pass Gflops，不是 parameter count**。这与语言模型的 scaling law 有所不同——后者通常把 loss 与 parameter、data、compute 三者联合 power-law 拟合。DiT 显示，在 parameter 持平但 Gflops 不同的设置下，质量仍按 Gflops 改进。

具体数据点（Table 4 + Figure 8）：

| Model | Params | Gflops | FID-50K @400K (no guidance) |
|---|---|---|---|
| DiT-S/8 | 33M | 0.36 | 153.60 |
| DiT-S/4 | 33M | 1.41 | 100.41 |
| DiT-S/2 | 33M | 6.06 | 68.40 |
| DiT-B/8 | 131M | 1.42 | 122.74 |
| DiT-B/4 | 130M | 5.56 | 68.38 |
| DiT-B/2 | 130M | 23.01 | 43.47 |
| DiT-L/8 | 459M | 5.01 | 118.87 |
| DiT-L/4 | 458M | 19.70 | 45.64 |
| DiT-L/2 | 458M | 80.71 | 23.33 |
| DiT-XL/8 | 676M | 7.39 | 106.41 |
| DiT-XL/4 | 675M | 29.05 | 43.01 |
| **DiT-XL/2** | **675M** | **118.64** | **19.47** |

注意 DiT-L/2（80 Gflops, 458M params, FID 23.33）vs DiT-XL/4（29 Gflops, 675M params, FID 43.01）：**参数多的模型反而更差，因为 Gflops 少**。这是 parameter count 不是 quality 代理的有力反证。

### 4.2 Larger models are more compute-efficient (Figure 9)

把 FID 作为总训练 compute（model Gflops × batch × steps × 3）的函数画出来，发现 **大模型在每个 compute 预算下都更优**——小模型训练再久也追不上。这与 Kaplan & Henighan 的 Chinchilla-style 观察一致。

### 4.3 Sampling compute 不能替代 model compute (Figure 10)

Diffusion 的一个独特"漏洞"是推理时可以多采样步数。论文测试 16/32/64/128/256/1000 步。

实验结论：**DiT-L/2 用 1000 步（80.7 Tflops/sample）的 FID 仍劣于 DiT-XL/2 用 128 步（15.2 Tflops/sample，5× 更少 compute）**。

Intuition：每步去噪的"质量上限"由 backbone 决定。再多次迭代也无法超越这个上限。这对实际部署很重要——你宁愿训一个大模型然后只跑 50 步，也不愿训一个小模型跑 1000 步。

---

## 五、SOTA 结果

### 5.1 ImageNet 256×256

DiT-XL/2 训 7M steps：

| Model | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|---|---|---|---|---|---|
| BigGAN-deep | 6.95 | 7.36 | 171.4 | 0.87 | 0.28 |
| StyleGAN-XL | 2.30 | 4.02 | 265.12 | 0.78 | 0.53 |
| ADM-G, ADM-U | 3.94 | 6.14 | 215.84 | 0.83 | 0.53 |
| LDM-4-G (cfg=1.50) | 3.60 | – | 247.67 | 0.87 | 0.48 |
| **DiT-XL/2-G (cfg=1.50)** | **2.27** | **4.60** | **278.24** | 0.83 | **0.57** |

DiT-XL/2 同时拿到了最低 FID 和最高 Recall，这在 GAN vs Diffusion 长期对决里是分水岭。

### 5.2 ImageNet 512×512

512 模型训 3M steps，input latent 变成 $64 \times 64 \times 4$，patchify 后 1024 个 token，Gflops 飙到 524.6：

| Model | FID↓ | sFID↓ | IS↑ |
|---|---|---|---|
| ADM-G, ADM-U | 3.85 | 5.86 | 221.72 |
| **DiT-XL/2-G (cfg=1.50)** | **3.04** | **5.02** | **240.82** |

注意 ADM 用 1983 Gflops，ADM-U 用 2813 Gflops，DiT-XL/2 只用 524.6 Gflops——**4–5× compute 效率提升**。

---

## 六、训练 Recipe（值得 highlight 的细节）

- Optimizer: AdamW，lr=$10^{-4}$，**constant schedule，no warmup，no weight decay**
- Batch size: 256
- EMA decay: 0.9999
- 没有 dropout、no Mixup、no CutMix、no RandAugment
- 唯一 data augmentation: horizontal flip
- 没看到 loss spike，training 高度稳定

Karpathy 你应该 appreciate：很多 ViT 早期训练需要大量 augmentation + warmup + LARS，但这里只用了最朴素的 AdamW 配置。原因是 diffusion 的 per-step noise 注入本身已经提供了极强的 stochasticity / regularization，使得 transformer 训练变成"easy"。

---

## 七、为什么这件事重要——后续的"DiT 谱系"

DiT 论文本身只做 ImageNet class-conditional，但它打开了几个方向：

### 7.1 Sora (OpenAI, 2024)
Sora 把 DiT 用在 video 生成上——spacetime patches 作为 token，adaLN-Zero 注入 text condition + timestep。OpenAI 在 technical report 里明确指出这是 DiT 的扩展。
Ref: <https://openai.com/research/video-generation-models-as-world-simulators>

### 7.2 Stable Diffusion 3 (Stability AI, 2024)
SD3 把 backbone 从 U-Net 切换到 MM-DiT（multimodal DiT），text 和 image token 在同一 transformer 里做 self-attention，使用 adaLN-Zero 和零初始化。
Ref: <https://arxiv.org/abs/2403.03206>

### 7.3 PixArt-α / PixArt-Σ
高效 text-to-image DiT，用 cross-attention 而非 adaLN-Zero（因为 condition 是文本不是 class）。
Ref: <https://arxiv.org/abs/2310.00626>

### 7.4 Latte (Video DiT)
Video generation 的 DiT，时空 patch 化。
Ref: <https://arxiv.org/abs/2401.03048>

### 7.5 MDT (Masked Diffusion Transformer)
在 DiT 基础上加 masked modeling 提速训练。
Ref: <https://arxiv.org/abs/2306.14075>

### 7.6 U-ViT
与 DiT 同期的 concurrent work，思路类似但更早出现在 arXiv 上，由清华团队完成。
Ref: <https://arxiv.org/abs/2209.12152>

---

## 八、Critique / 你可能想质疑的点

Karpathy 你看到这个工作大概率会有几个自然的疑问：

### 8.1 "Gflops 是不是真正的 scaling axis？"

论文只做了 ImageNet class-conditional（数据集很局限）。后续工作（SD3、Sora）显示，**当 condition 变成自然语言时，cross-attention 反而比 adaLN 更合适**——因为 condition 是变长序列而不是单个 label。所以 adaLN-Zero 不是 universally best，只是在 class-conditional 设定下 best。

### 8.2 Patchify 的"无 free lunch"

减小 $p$ 增加 Gflops 没有增加 parameter，听起来"免费"。但实际部署时，self-attention 是 $O(T^2)$，$p=2$ 在 $256 \times 256$ 上 $T=256$ 还好，到 $512 \times 512$ $T=1024$，到 video 就爆炸。后续工作（如 Sora）需要 window attention、spatiotemporal factorization 来缓解。DiT 论文没讨论这个，因为 ImageNet 256/512 还撑得住。

### 8.3 VAE 的"卷积遗产"

DiT 在 latent space 操作，VAE 仍然是卷积的。所以严格来说"U-Net inductive bias 不必要"的论断只适用于 **diffusion backbone 部分**，整个生成 pipeline 里仍存在卷积。如果直接在 pixel space 训 DiT（如 ADM 那样），Gflops 会高到无法承受，但论文没做这个对照。

### 8.4 Scaling law 的 power-law 形式

DiT 没有像 Chinchilla 那样给出 $\text{FID} \propto \text{Gflops}^{-\alpha}$ 的具体指数拟合。后续 U-ViT 与一些 scaling 论文（如 Karras 的 EDM2, 2024）做了更精细的 scaling law 分析。EDM2 进一步表明，在固定 Gflops 下，parameter 与 token 数有更微妙的平衡。
Ref EDM2: <https://arxiv.org/abs/2406.07596>

---

## 九、最该带走的 Intuition

1. **Architecture unification works for generative models too**——扩散模型的 backbone 可以"脱掉" U-Net 这个历史包袱，拥抱 transformer 的 scaling law。
2. **Conditioning mechanism 是 critical design choice**，比增加 parameter 更重要。adaLN-Zero 同时拿到最低 Gflops 与最低 FID，是 rare 的"免费午餐"。
3. **Zero-init residual block 是训练深 transformer 的通用 trick**，把网络初始成 identity function 让训练初期梯度直通。
4. **Forward pass Gflops 是扩散 backbone 的关键 scaling axis**，比 parameter count 更相关——这意味着推理时的 compute budget 与训练时的 compute budget 在 DiT 里通过 Gflops 统一。
5. **Sampling compute 不能替代 model compute**——大模型少步数 > 小模型多步数。这对部署性价比的判断很重要。
6. **Diffusion 的 noise injection 自带 regularization**，让 transformer 训练变得异常稳定，省掉 ViT 常用的 warmup/augmentation——这点很反直觉，值得记住。
7. **Patchify 是 DiT 的 tokenization 入口**，$p$ 是最重要 hyperparameter。后续所有 video DiT 工作都沿用这个 abstraction。

---

## 十、如果想自己复现/扩展

- 官方 code: <https://github.com/facebookresearch/DiT>——非常 clean 的 JAX/Flax 实现
- PyTorch port: <https://github.com/williamberman/DiT-pytorch>
- HuggingFace diffusers 集成: <https://huggingface.co/docs/diffusers/api/pipelines/dit>
- Sora technical report: <https://openai.com/research/sora>
- SD3 paper: <https://arxiv.org/abs/2403.03206>
- EDM2 (Karras scaling law 后续): <https://arxiv.org/abs/2406.07596>
- PixArt-α: <https://arxiv.org/abs/2310.00626>
- U-ViT (concurrent): <https://arxiv.org/abs/2209.12152>
- Latte (video): <https://arxiv.org/abs/2401.03048>
- MDT: <https://arxiv.org/abs/2306.14075>
- Original LDM: <https://arxiv.org/abs/2112.10752>
- ADM (Dhariwal & Nichol): <https://arxiv.org/abs/2105.05233>
- Classifier-free guidance (Ho & Salimans): <https://arxiv.org/abs/2207.12598>
- ViT (Dosovitskiy et al.): <https://arxiv.org/abs/2010.11929>
- Peebles & Xie follow-up (SiT, score-based interpolation transformer): <https://arxiv.org/abs/2401.01622>

最后一个 link 是同一作者在 DiT 之后做的 SiT (Scalable Interpolant Transformers)，把 DiT 推广到更一般的 interpolant 框架，对理解 diffusion 与 flow matching 的统一视角很有帮助——这也是 2024–2026 生成模型社区的重要方向（rectified flow、flow matching 都在 similar space 里）。

希望这能 build 你对 DiT 的 intuition——它表面上是个"换 backbone"的工程论文，但真正改变了社区对扩散模型架构的认知：**transformer 在生成模型里同样可以 scaling，U-Net 的时代就此结束。**
