---
source_pdf: The Ingredients for Robotic Diffusion Transformers.pdf
paper_sha256: 34afc7d45b47c11d8fa6220f10ac3bc0fea71878c99d645c0cac897b2698a949
processed_at: '2026-08-12T14:19:50-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 这篇 paper 在干嘛

一群人想给 robot 教动作。教法很直接——让人类 tele-op 示范，robot 照着模仿，术语叫 imitation learning / behavioral cloning。

问题来了。你想让 robot 学得厉害，得给它喂很多数据。数据一多，麻烦也多起来：

同一个任务，不同人 demo 的做法可能完全不一样。比如"把杯子放到碗里"，张三先抓杯子再挪到碗上方松手，李四先推杯子滚到桌边再一把抓起来丢进去。这两种都对。你让 neural network 去拟合，它很老实，取个平均——结果输出一个"半抓半推"的动作，啥也干不成。这就是 **multi-modal action distribution** 问题。

还有个问题。robot 干精细活儿要毫米级精度。切寿司这种任务要 1500 步，每步差一点，到最后寿司已经飞出砧板了。所以你得预测得很准。

## 怎么解决 multi-modal

用 **diffusion model**。这个东西最近几年在 image generation 火得一塌糊涂（Stable Diffusion 之类）。核心思想很简单：

给你一张纯噪声图，你训练一个 network 去猜"这张图里加了什么噪声"，然后把猜到的噪声减掉。反复做 100 步，噪声越来越少，图就显现出来了。

把这套搬到 robot policy 上：你把"正确 action"当 ground truth，往里加噪声，让 network 学着预测噪声。推理时从纯随机噪声出发，一步步 denoise，最后蹦出一个合理 action。

为啥这能解决 multi-modal？因为 diffusion 是个**生成模型**，它学的是 action 的**分布**，不是回归出一个平均值。从同一个噪声起点出发，只要噪声采样不同，就能生成不同的合理 action——抓也好、推也好，都是分布里的合法样本。https://arxiv.org/abs/2303.04137

## 然后他们撞墙了

原版 diffusion policy paper 给了两个 backbone 选择：

- **U-Net**：好调，能 work，但有个毛病——它假设 action signal 是平滑的。切寿司这种"刀落下去停一下再落"的断续动作，U-Net 天然不擅长。
- **Transformer**：容量大、能 scale、理论上是未来。但这帮作者自己都说"extremely difficult to train"。后来一堆人跟着踩坑，基本放弃了。

所以现在大家用 diffusion policy 都还在用 U-Net，transformer 版本没人敢碰。

## 这篇 paper 的核心发现

**Transformer diffusion policy 训不动不是 transformer 本身的问题，是你往里喂 condition 的方式不对。**

原版怎么喂 condition 的？用 **cross-attention**。decoder 那边拿着带噪 action 当 query，去 encoder 出来的 observation feature 里 attend。

听起来很自然，但实际训练时 cross-attention 的梯度路径太陡。带噪 action 这个 query 早期是随机的，它 attend 到哪些 observation token 全凭运气。每一步 denoise 的条件信号没被稳住，模型学不到 consistent 的 mapping。推理时你想少跑几步加速，10 步 DDIM，数值误差一累积就崩了。

## 解法：抄 image generation 的作业

**DiT**（Scalable Diffusion Models with Transformers, https://arxiv.org/abs/2212.09748）这篇 paper 去年在 image generation 领域解决了完全一样的问题。他们把 condition 注入方式从 cross-attention 换成 **adaptive LayerNorm (adaLN)**，再配一个 zero-initialization trick，训练立马稳了。

这篇 robot paper 就是把这套搬过来。

### adaLN 是什么

普通 LayerNorm 就是把 feature 减均值除标准差，再乘个固定 $\gamma$ 加个固定 $\beta$：

$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

adaptive LayerNorm 把 $\gamma$ 和 $\beta$ 换成**由 condition 信号动态生成**的：

$$x' = a(c) \odot \text{LN}(x) + b(c)$$

- $x$：当前 block 的隐状态
- $c$：condition 信号，这里是 observation embedding 均值 + diffusion step $k$ 的 embedding
- $a(c), b(c)$：两个小 MLP，输入 $c$，输出和 $x$ 同维度的 scale 向量和 shift 向量
- $\odot$：逐元素乘

直觉：condition 不再是通过 attention "去拿"，而是通过 LayerNorm 的 affine 参数"去塑形"。每一层、每个 channel 都被 condition 调控。梯度路径平滑很多。

### adaLN-Zero：最关键的 trick

光 adaLN 还不够。DiT 还干了一件事：**把生成 $a(c)$ 和 $b(c)$ 的那个 MLP 的最后一层权重初始化成 0**。

这意味着训练刚开始时 $a(c) = 0, b(c) = 0$，于是：

$$x' = 0 \cdot \text{LN}(x) + 0 = 0$$

整个 block 输出是 0，加上 residual connection：

$$x_{\text{out}} = x + \text{block}(x') = x + 0 = x$$

网络初始化成 **identity**。

这事儿在 diffusion 里特别对路。diffusion noise prediction 的本质是学一个**残差**：从带噪 action $x^k$ 里减掉噪声 $\epsilon^k$。如果初始网络啥也不做，输出 $\epsilon_\theta = 0$，那就相当于"这一步不去噪"——完全安全，不会把 $x^k$ 搞飞。

cross-attention 初始化是随机的，一开始就往 $x^k$ 里注入随机扰动，高精度 robot 任务直接崩。zero-init 让 condition 的影响**渐进打开**，训练早期信息量低，慢慢加进来，模型才扛得住。

这跟 Goyal et al. 的 skip-init 思路一脉相承：https://arxiv.org/abs/1706.02677

## 完整架构长什么样

```
Input: 4 路 camera + proprio + language goal + diffusion step k
        │
        ▼
[4 个独立 ResNet-26 encoder，每个吃一路 camera]
        │  + FiLM 层把 language 注进 ResNet 每一层
        ▼
[observation dropout 对 proprio 做 per-dim 随机 mask]
        │
        ▼
[加 learned positional encoding]
        │
        ▼
[Octo Block-Attention Transformer Encoder]
        │
        ▼  e^(1), e^(2), ..., e^(L)  ← 每层一组 embedding
        
        同时：
        [x^k (带噪 action chunk)] + [k embedding (sinusoidal + MLP)]
                    │
                    ▼
        [Transformer Decoder with adaLN-Zero blocks]
                    │
                    ▼
              ε_θ (预测的噪声)
```

几个细节：

**为什么用独立 ResNet 而不是共享 encoder 或 ViT？** 实验里他们试了把 ResNet 换成 patchify + 更大 transformer，参数加到 150M vs 115M，结果 Pick Place 从 50% 掉到 13%。robot 数据才几小时量级，CNN 的 spatial prior 是免费的午餐。https://arxiv.org/abs/2106.14881

**FiLM 是什么？** Feature-wise Linear Modulation。language embedding 经 MLP 生成 per-channel 的 scale $\gamma$ 和 shift $\beta$，在 ResNet 每层对 feature map 做 $\gamma \odot x + \beta$。让 language 在 vision backbone 早期就塑形 attention。https://arxiv.org/abs/1707.09835

**observation dropout 是什么？** proprio 信号（关节角度）是 shortcut，模型容易只靠它偷懒不看 camera。训练时按维度随机把 proprio 置零，强制用 vision。https://arxiv.org/abs/2310.00498

## 训练目标

就是个标准 DDPM 的 noise prediction MSE：

$$\mathcal{L} = \left\| \epsilon^k - \epsilon_\theta(a_t + \epsilon^k, k, o_t, g) \right\|_2^2$$

- $a_t$：ground truth action chunk（连续 100 步 action，$H=100$）
- $\epsilon^k \sim \mathcal{N}(0, I)$：随机采的高斯噪声
- $k \in \{1, \dots, 100\}$：diffusion step，训练时随机采样
- $o_t$：当前 observation
- $g$：language goal

训练用 $K=100$ 步，cosine noise schedule。推理用 DDIM 压到 10 步。https://arxiv.org/abs/2010.02502

## 实验结果有多炸

### 主对比（ALOHA 双臂，3 个任务）

| Method | Pick Place | Pen Uncap | Sushi Cut | 平均 |
|---|---|---|---|---|
| **DiT-Block Policy** | **50%** | **100%** | **29%** | **60%** |
| ACT (原 ALOHA paper) | 37.5% | 40% | 21% | 33% |
| Diffusion Policy U-Net | 31.3% | 90% | 4% | 42% |
| Diffusion Policy Transformer | 0% | 0% | 0% | 0% |

原版 transformer diffusion policy 直接 0 分，全崩。U-Net 在 Pen Uncap 很强（90%）但 Sushi Cut 几乎零——切寿司是断续 action，U-Net 的平滑 prior 反成负担。

DiT-Block Policy 三项都不差，平均领先 20%。

### Attention Block 消融（这篇 paper 最关键的实验）

| Method | DDIM 步数 | Pick Place | Pen Uncap |
|---|---|---|---|
| **adaLN-Zero (本文)** | 10 | **50%** | **100%** |
| adaLN 不 zero-init | 10 | 38% | 80% |
| Cross-Attention | 10 | 0% | 0% |
| Cross-Attention | 100 | 38% | 70% |
| In-Context | 100 | 0% | 0% |

cross-attention 要 100 步才能勉强跑起来，10 步直接全崩。adaLN-Zero 10 步就 100%。zero-init 这个 trick 单独贡献 12-20%。

### Encoder 消融

| Method | 参数量 | Pick Place | Pen Uncap |
|---|---|---|---|
| **ResNet-26 (本文)** | 115M | **50%** | **100%** |
| 无 ResNet (小) | 85M | 0% | 0% |
| 无 ResNet (加大 transformer) | 150M | 13% | 20% |

多给 35M 参数也补不回来。CNN prior 在小数据量级上完胜纯 transformer。

### 跨 robot 泛化

在 Franka 单臂上（不同 action space、不同 camera 设置）也跑赢 baseline 20-35%。说明这套架构不是 ALOHA 专属。

## 一句话总结

**Transformer diffusion policy 训不动是因为 condition 注入方式太猛，换成 adaLN-Zero 让 condition 软注入 + 网络初始化成 identity，训练就稳了。** 配上独立 ResNet tokenizer + FiLM 语言注入 + observation dropout，就是让 robot diffusion transformer work 的最小配方。

核心 intuition：diffusion 是残差任务，condition 应该渐进塑形而非硬路由，网络初始最好啥也不做。

---

# The Ingredients for Robotic Diffusion Transformers —— 详解

paper 来自 CMU + Berkeley 团队（Sudeep Dasari、Oier Mees、Sergey Levine 等），项目主页 https://dit-policy.github.io 。核心是给出一个让 **Diffusion Transformer Policy** 稳定训练并 SOTA 的最小配方，命名为 **DiT-Block Policy**。下面拆解动机、架构、公式、实验，并尽量 build 出 intuition。

---

## 1. 为什么需要这篇 paper

Robot imitation learning 长期被三个 tension 撕扯：

1. **高维 multi-camera observation**：ALOHA 这种双臂系统有 top camera + 双 wrist camera + 本体感知（proprioception），naive 的 shared encoder 会偷懒只用 proprio 或 top camera，导致 wrist occlusion 时崩盘。
2. **毫米级长 horizon 精度**：Sushi Cut 这种任务 1500+ step，单点 error 会沿着轨迹放大，到最后切不到四块。
3. **Multi-modal action distribution**：10 小时多专家 demo，同一个 scene 可能存在多种合理 solution；用 L2/MSE 回归 action 会取平均得到"非决定性"的中间动作直接 fail。

**Diffusion Policy (Chi et al., RSS 2023, https://arxiv.org/abs/2303.04137)** 用 generative diffusion 解决 multi-modality，但其原版有两个 backbone：
- **U-Net 版**：好调，但要求 action signal 足够 smooth，对硬件 setup 敏感，长 horizon 灵巧任务精度受限。
- **Transformer 版**：容量大、可 scale，但作者自己都说"extremely difficult to train"。后续 Consistency Policy (https://arxiv.org/abs/2405.07503) 等也回避了 Transformer backbone。

本文的 thesis：**Transformer 版 diffusion policy 训不稳不是 fundamental 问题，而是 conditioning 机制选错了。** 把 image generation 里 DiT (Peebles & Xie, https://arxiv.org/abs/2212.09748) 的 adaLN-Zero 思路搬过来即可。

---

## 2. 问题形式化

学一个 goal-conditioned 策略 $\pi_\theta(a_t \mid o_t, g)$，其中

- $o_t \in \mathcal{S}$：观测，这里就是 multi-camera images + proprio
- $a_t \in \mathcal{A}$：动作 chunk（连续 H 步的 joint 或 Cartesian state）
- $g$：language instruction，e.g. "pick up the corn and place it in the bowl"

通过 BC 在 expert dataset $\mathcal{D} = \{\tau_i\}$ 上监督学习，每条轨迹 $\tau_i = \{g, o_0, a_0, o_1, \dots\}$。

### 2.1 Diffusion 训练目标

把策略写成 **DDPM** (Ho et al., https://arxiv.org/abs/2006.11239)。给定初始高斯噪声 $x^K$（注意 paper 用上标 $k$ 表示 diffusion step，下文统一用 $k$），noise prediction network $\epsilon_\theta(x^k, k, o_t, g)$，denoise 一步：

$$x^{k-1} = \alpha \left( x^k - \gamma \, \epsilon_\theta(x^k, k, o_t, g) \right) + \mathcal{N}(0, \sigma^2 I)$$

变量含义：
- $x^k$：第 $k$ 步带噪 action chunk（从 $a_t$ 出发加噪得到）
- $k$：diffusion time index，训练时 $k \in \{1, \dots, K\}$，$K=100$
- $\alpha, \gamma, \sigma$：cosine noise schedule (https://arxiv.org/abs/2102.09772) 决定的常数，控制每步去噪量与剩余噪声量
- $\epsilon_\theta$：神经网络预测的 noise；采样到 $x^0 \approx a_t$ 即得到 action

监督目标就是预测 noise 的 MSE：

$$\mathcal{L} = \left\| \epsilon^k - \epsilon_\theta(a_t + \epsilon^k, k, o_t, g) \right\|_2^2$$

- $\epsilon^k \sim \mathcal{N}(0, I)$：训练时随机采样的 Gaussian noise
- $a_t + \epsilon^k$：对 ground-truth action 加噪后作为输入
- 推理时用 DDIM (https://arxiv.org/abs/2010.02502) 把 100 step 压到 10 step，减少 latency，但越少 step 对模型稳定性要求越高，这是后面 ablation 的关键。

---

## 3. DiT-Block Policy 架构

整体是 **encoder-decoder Transformer**：

```
[Image top, Image wrist-L, Image wrist-R, proprio] 
     --(各自 ResNet-26 tokenizer + FiLM(text))-->  
     --(learned PE)--> 
     --(Block-Attention Encoder, from Octo)-->  e^(1), ..., e^(L)
                                                        | (per-layer feed to decoder)
[k-embed (sinusoidal + MLP)]  +  [x^k (noised action chunk)] 
     --(Decoder with adaLN-Zero blocks)-->  epsilon^k
```

### 3.1 Image tokenization: ResNet-26 + FiLM

每个 camera 独立走一个 **ResNet-26** (He et al., https://arxiv.org/abs/1512.03385) 编码到 49 个 spatial token。text goal 用 **DistilBERT** (https://arxiv.org/abs/1910.01108) 编码后，通过 **FiLM** (Perez et al., https://arxiv.org/abs/1707.09835) 在 ResNet 每层注入：

$$\hat{x}_{c,h,w} = \gamma_c^{(\ell)}(g) \odot x_{c,h,w}^{(\ell)} + \beta_c^{(\ell)}(g)$$

- $x_{c,h,w}^{(\ell)}$：第 $\ell$ 层、channel $c$、位置 $(h,w)$ 的特征
- $\gamma_c^{(\ell)}(g), \beta_c^{(\ell)}(g)$：由 text embedding 经 MLP 推出的 per-channel scale/shift
- $\odot$：逐元素乘

这让 language 在 vision backbone 早期就塑形 attention，避免后期 cross-attention 难对齐。

proprio 走 **observation dropout** (Srivastava et al., https://jmlr.org/papers/v15/srivastava14a.html；Dasari et al., https://arxiv.org/abs/2310.00498) ：训练时按维度随机 mask 本体感知，强制模型利用 vision，避免 overfit 到 proprio shortcut。

### 3.2 Encoder：Block Attention

直接复用 **Octo** (https://octo-models.github.io) 的 Block Attention encoder：把 4 路 camera + proprio 的 token 一起进 transformer，但 attention 限制在 block 内部（类似 attention with hierarchical masking），既保持 spatial prior 又允许 cross-modal 信息混入。输出 $e^{(1)}, \dots, e^{(L)}$（每层一组 embedding）。

### 3.3 Decoder：核心 ingredient —— adaLN-Zero

这是 paper 的灵魂。原版 Diffusion Policy Transformer 用 **standard cross-attention** 把 encoder embedding 喂给 decoder：

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V, \quad Q = x^k W^Q, \; K = e^{(i)} W^K, \; V = e^{(i)} W^V$$

paper 的发现：这种 hard attention 在 diffusion 训练里梯度路径太陡，$x^k$（被噪化的 action chunk）经常 attend 到无关 token，造成训练发散；推理时若想减少 DDIM step（10 步），数值误差累积导致 unsafe action。

**DiT (Peebles & Xie)** 在 image generation 里的解法是 **adaptive LayerNorm (adaLN)**，本文把它移植到 policy：

$$x' = a(e^{(i)}, k) \odot \text{LN}(x) + b(e^{(i)}, k)$$

- $x$：当前 decoder block 的隐状态
- $\text{LN}(\cdot)$：标准 LayerNorm（无 affine）
- $a(e^{(i)}, k), b(e^{(i)}, k)$：scale 与 shift 向量，由 condition signal 推出
- $\odot$：逐 channel 缩放

具体实现：

$$a(e^{(i)}, k) = \text{Dense}\left( \text{tokenmean}(e^{(i)}) + \text{MLP}(k) \right)$$

- $\text{tokenmean}(e^{(i)})$：把第 $i$ 层 encoder embedding 沿 token 维做平均得到一个 condition vector
- $\text{MLP}(k)$：diffusion step $k$ 经 sinusoidal embedding + MLP 得到 time condition
- 两者相加再过 Dense，得到 per-channel 的 scale/shift

**关键 trick —— adaLN-Zero**：把最后一层 projection（输出 scale $a$ 和 shift $b$ 的 dense layer）**初始化为 0**。即训练刚开始时 $a = 0, b = 0$，于是 $x' = 0 \cdot \text{LN}(x) + 0 = 0$，残差分支输出 0，整个 block 退化为 identity：

$$x_{\text{out}} = x + \text{Attn}(x') + \text{FFN}(x') \;\xrightarrow{\text{init}}\; x + 0 + 0 = x$$

直觉（这是 Karpathy 你应该最爱的部分）：
- Diffusion noise prediction 是个**纯残差任务**：模型要学的就是从带噪 $x^k$ 减掉噪声分量 $\epsilon^k$。如果初始网络就是 identity，那输出的 $\epsilon_\theta \approx 0$，去噪一步相当于不动 $x^k$——这是个**安全的初始化**，不会让 $x^k$ 突然跳到奇怪位置。
- Cross-attention 的初始化是随机的，初期会把 $x^k$ 投到随机 attention 分布上，造成 non-trivial 扰动；在高 precision robotics 任务里，这种扰动会放大成 unsafe action。
- adaLN-Zero 把 conditioning 信号"软注入"到 LayerNorm 的 affine，相比 cross-attention 的 hard routing，梯度更平滑；加上 zero-init 让 condition 影响渐进打开，避免训练早期信息过载。

这跟 Goyal et al. 的 large-batch training trick (https://arxiv.org/abs/1706.02677) 里 "skip-init" / "ReZero" 的思想一脉相承。

---

## 4. BiPlay 数据集

为测试 scaling 行为，作者构建了 **BiPlay**：326 个 randomized scene，7023 个 3.5-min play clip，约 9.7 小时，每个 clip 配 language annotation。Mix 上再加 ALOHA 静态 task demos、YaY (https://arxiv.org/abs/2403.12910) 的 correction rollouts、DROID (https://droid-dataset.github.io) 等开源数据做 regularizer。

| Dataset | Clips | Scenes | Tasks | Hours |
|---|---|---|---|---|
| BiPlay | 7023 | 326 | 200+ | 9.7 |
| ALOHA | 855 | 15 | 16 | 2.9 |
| YaY | 4k | 3 | 3 | 15.4 |
| Pen Uncap | 100 | 1 | 1 | 0.3 |
| Sushi Cut | 256 | 1 | 1 | 1.4 |
| Pick Place | 863 | 1 | 1 | 1.8 |
| Dough Cut | 150 | 1 | 1 | 2.7 |
| Open Drawer | 115 | 1 | 1 | 2.7 |

训练用 **AdamW** (https://arxiv.org/abs/1711.05101)、cosine LR schedule (https://arxiv.org/abs/1608.03983)、250K iter；预测 **H=100 的 action chunk**，配合 **temporal ensembling** (https://arxiv.org/abs/2304.13705) 在执行时多步预测加权平均，进一步降低 jitter。

---

## 5. 实验：Bi-manual ALOHA

5 个任务，前 3 个在 ALOHA：

1. **Pick Place**：3D 文字指令 ground 到两个物体 + 两个 receptacle 中正确组合
2. **Pen Uncap**：一手抓 sharpie，另一手拔 cap——精度 + 双臂协同
3. **Sushi Cut**：1500+ step，把 sushi 放砧板 → 抓刀 → 换手 → 切 4 块。partial credit 按 1/3 递减。

### 5.1 Baseline 对比 (Table II)

| Method | Pick Place (BiPlay) | Pen Uncap | Sushi Cut | Avg (BiPlay) |
|---|---|---|---|---|
| **DiT-Block Policy** | 50% | 100% | 29% | **60% ± 9%** |
| ACT (https://arxiv.org/abs/2304.13705) | 37.5% | 40% | 21% | 33% ± 8% |
| D.P. U-Net | 31.3% | 90% | 4% | 42% ± 9% |
| D.P. Transformer | 0% | 0% | 0% | 0% ± 0% |

观察：
- D.P. Transformer（原版 cross-attention）**在所有任务上 0%**，验证了"transformer diffusion policy 训不稳"的原 paper 痛点。
- D.P. U-Net 在 Pen Uncap 强（90%），Sushi Cut 弱（4%），证明 U-Net 的 smoothness assumption 在长 horizon 切割任务上失灵——切割是断续、非 smooth action，U-Net 的 inductive bias 反成负担。
- DiT-Block Policy 在三个任务上都不差，没明显短板。

### 5.2 Attention Block Ablation (Table III)

| Method | DDIM Iters | Pick Place | Pen Uncap |
|---|---|---|---|
| **Ours (adaLN-Zero)** | 10 | 50% | 100% |
| No Zero-Init | 10 | 38% | 80% |
| Cross Attn | 10 | 0% | 0% |
| Cross Attn | 100 | 38% | 70% |
| In-Context | 100 | 0% | 0% |

- Cross-attention 即使加到 100 step 也只能勉强跑起来（38%/70%），且推理慢到 trajectory jerky。
- **Zero-init 贡献 12%–20%**：去掉 zero-init，10 step 推理下从 50% 掉到 38%。证明 zero-init 不只是稳定 trick，而是直接决定少步采样的可行性。
- In-context（把 encoder embedding 直接 prepend 到 decoder）彻底崩盘，说明 condition 注入方式比 condition 内容本身更关键。

### 5.3 Encoder Ablation (Table IV)

| Method | Params | Pick Place | Pen Uncap |
|---|---|---|---|
| **Ours (ResNet-26)** | 115M | 50% | 100% |
| No ResNet (small) | 85M | 0% | 0% |
| No ResNet (scaled) | 150M | 13% | 20% |

把 ResNet 替换成 ViT-style patchify + 更大 transformer，即使**多 35M 参数**也远不如 ResNet。理由：robotics 数据相对少，CNN 的 translation equivariance 与 spatial prior 是 strong inductive bias，纯 ViT 需要更大数据才能学好空间结构。这跟 "Early Convolutions Help Transformers See Better" (https://arxiv.org/abs/2106.14881) 结论一致。

---

## 6. 单臂 Franka 迁移

为测试 morphology 泛化，作者在 Franka FR3 上跑两个任务（single camera + Cartesian velocity action space）：

- **Toasting**：抓物体 → 放入 toaster → 关盖
- **Wiping**：定位 sponge → 抓 → 把 debris 推进 dustpan

DiT-Block Policy 相比 ACT 平均 +20%，相比 D.P. U-Net +35%。说明 DiT-Block Policy 对 action space (joint → Cartesian velocity) 与 observation space (multi-cam → single-cam) 的迁移 robust，而 D.P. U-Net 的 smoothness prior 在 Cartesian velocity 上失灵。

---

## 7. Robomimic Sim 评测 (Table V)

| Task | DiT-Block Policy | D.P. Transformer |
|---|---|---|
| Lift | 100% | 100% |
| Can | 98% | 100% |
| Square | 84% | 100% |
| Tool Hang | 72% | 76% |
| **Avg (Sim)** | 88.5% | 94% |
| **Avg (ALOHA Real)** | 60% | 0% |

在 sim 上 DiT-Block Policy 略低 5.5%，但**几乎零 task-specific tuning**；D.P. Transformer 在 sim 上能调到 94%，但搬到 ALOHA real 立刻崩到 0%。这印证了 sim-to-real gap 主要来自 architecture 的 tuning sensitivity，而非 raw capacity。

---

## 8. 串联直觉：为什么这套配方 work

把所有 ingredient 串起来：

1. **Diffusion 是 residual learning** → 模型本质上学的是"从 $x^k$ 减掉噪声"，所以初始化成 identity（zero-init adaLN）天然对齐了 task 的 inductive bias。
2. **Diffusion step $k$ 是 strong condition** → 必须以 scale/shift 形式注入到每一层（adaLN），让所有 layer 都知道当前在哪个 denoise step，否则 cross-attention 只在某几层看到 $k$ embedding，信号衰减。
3. **Multi-camera 高维 obs 容易让模型走 shortcut** → 用独立 ResNet + observation dropout 强制利用所有 modality。
4. **Long-horizon + dexterous 需要高精度 action distribution** → diffusion 解决 multi-modality，adaLN-Zero 让少步 DDIM（10 步）也能稳定，降低 latency 让 1500 step 任务跑得动。
5. **Vision backbone 用 CNN 仍优于 ViT 在 robot data regime** → 因为 data 量级远小于 imageNet，CNN prior 省下大量 sample efficiency。

可以联想到的相关工作：
- **Stable Diffusion 3 / Flux** (https://arxiv.org/abs/2403.03206) 也用 rectified flow + adaLN-Zero 路线，验证了这套 conditioning 在高分辨率生成上的扩展性。
- **ALOHA Unleashed** (https://arxiv.org/abs/2410.13126) 用 cross-attention transformer + 26k episode 才 work，说明 cross-attn 不是不行，而是需要海量数据 + 调参；DiT-Block Policy 在 7k 数据就能 work，sample efficiency 显著优。
- **Consistency Policy** (https://arxiv.org/abs/2405.07503) 用 distillation 加速 diffusion policy，但 backbone 仍是 U-Net，可与本文组合（用 DiT-Block 作为 teacher）。
- **$\pi_0$ (Physical Intelligence, https://www.physicalintelligence.company/blog/pi0)** 同样把 flow matching + VLM encoder 组合做 generalist policy，思路与本文一致，但 scale 大几个量级。

---

## 9. 局限与潜在改进

paper 没明说但能推断的：
- BiPlay 仍只有 10 小时，要验证 next-order scaling 需 100h+。
- ResNet-26 仍是 fixed encoder，未做 vision-language pretrained backbone（如 SigLIP, https://arxiv.org/abs/2303.15343）的对比。
- adaLN 把 encoder embedding 做了 token mean，损失了 spatial token-level 信息；高 dexterous 任务可能需要 per-token adaLN（参考 PixArt-alpha, https://arxiv.org/abs/2310.00460）。
- 100 step DDIM 训练、10 step 推理，仍比 ACT (1 step decoder) 慢一个数量级，real-time 控制频率受限于 action chunk size（H=100）的 amortization。

---

## 10. 一句话总结

DiT-Block Policy 的核心 contribution 是把 image generation 领域已经验证的 **adaLN-Zero conditioning** 搬到 robotic diffusion policy，配上 **独立 ResNet tokenizer + FiLM 语言注入 + observation dropout**，让 transformer diffusion policy 从"训不动"变成"零调参 SOTA"。其直觉是：diffusion noise prediction 是 residual 任务，conditioning 应该是软注入 + 零初始化，避免 cross-attention 的 hard routing 在 low-data + high-precision regime 失灵。

参考链接汇总：
- 项目主页：https://dit-policy.github.io
- DiT (Peebles & Xie)：https://arxiv.org/abs/2212.09748
- Diffusion Policy (Chi et al.)：https://arxiv.org/abs/2303.04137
- ACT / ALOHA (Zhao et al.)：https://arxiv.org/abs/2304.13705
- ALOHA Unleashed：https://arxiv.org/abs/2410.13126
- Octo：https://octo-models.github.io
- DROID：https://droid-dataset.github.io
- DDPM：https://arxiv.org/abs/2006.11239
- DDIM：https://arxiv.org/abs/2010.02502
- FiLM：https://arxiv.org/abs/1707.09835
- DistilBERT：https://arxiv.org/abs/1910.01108
- ResNet：https://arxiv.org/abs/1512.03385
- Early Convs Help Transformers：https://arxiv.org/abs/2106.14881
- YaY：https://arxiv.org/abs/2403.12910
- PixArt-alpha (per-token adaLN)：https://arxiv.org/abs/2310.00460
- Flux / SD3：https://arxiv.org/abs/2403.03206
