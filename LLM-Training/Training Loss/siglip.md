---
source_pdf: siglip.pdf
paper_sha256: 2efade8de3baeffe9eba759aab9f510ea8f8ac3c365f02f0da75d9baf9ffa2db
processed_at: '2026-08-12T06:07:24-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SigLIP 用人话讲

## 一句话说清楚

CLIP 训练的时候，loss function 用的是 softmax，意思就是"给你一张图，从 batch 里 N 个 caption 里挑出正确那一个"。SigLIP 说：别这么搞，换个更简单的——"给一个 image-text pair，直接判断它匹配不匹配"。就这一个改动，连带解锁了一堆好处。

paper: https://arxiv.org/abs/2303.15343

---

## 原来 CLIP 的 loss 长什么样，为啥别扭

假设 batch 里有 4 对 (image, text)：

```
(I1, T1)  (I2, T2)  (I3, T3)  (I4, T4)
```

CLIP 的 softmax loss 干的事：

对 image $I_1$，算它跟 $T_1, T_2, T_3, T_4$ 的相似度，然后做 softmax，希望 $T_1$ 的概率最大。公式：

$$
\mathcal{L}_{\text{img→txt}} = -\log\frac{e^{t\,\mathbf{x}_i\cdot\mathbf{y}_i}}{\sum_{j=1}^{N}e^{t\,\mathbf{x}_i\cdot\mathbf{y}_j}}
$$

变量翻译成人话：
- $\mathbf{x}_i$：第 $i$ 张图过 image encoder 后的 embedding，L2 normalize 到单位球面
- $\mathbf{y}_j$：第 $j$ 个文本过 text encoder 后的 embedding，同样 normalize
- $\mathbf{x}_i\cdot\mathbf{y}_j$：cosine similarity，范围 $[-1, 1]$
- $t$：temperature，控制 softmax 分布锐度，$t=\exp(t')$，$t'$ 是 learnable
- $N$：batch size
- 分母 $\sum_j$：把 batch 里所有 text 都过一遍，做归一化

然后反方向再算一遍 text→image，两个方向加起来除 2。

**别扭在哪？**

1. **分母把整个 batch 绑死了**。要算 $I_1$ 的 loss，必须知道 $T_1, T_2, T_3, T_4$ 全部的 embedding。分布式训练时，batch 散在多个 TPU/GPU 上，你必须 all-gather 把所有 embedding 拉到每个卡上，然后 materialize 一个 $N\times N$ 的相似度矩阵。$N=32k$ 时这个矩阵是 $32k\times 32k \approx 10^9$ 个 float，约 4GB。$N=1M$ 时 $10^{12}$ 个 float，4TB，根本放不下。

2. **任务难度跟 batch size 耦合**。batch=4 时是"4 选 1"，batch=32k 时是"32000 选 1"。同样的数据，batch 大了任务就难了，batch 小了任务就简单。这意味着 batch size 不只是个优化超参，它直接改变了"你在学什么"。

3. **数值上要 subtract-max**。softmax 直接算 $e^{x}$ 会溢出，必须先减去最大值。这要求额外遍历一遍 batch 找 max，又是全局操作。

4. **不对称**。image→text 和 text→image 是两次独立 softmax，行归一化一次、列归一化一次，概念上就是两次不同的 categorization task，硬拼在一起。

---

## SigLIP 的改法：把"多选一"换成"逐对判断"

核心想法：别从 N 个里挑 1 个了。直接把所有 pair 组合都拿出来，每个 pair 独立判断"这是不是一对"。

batch=4 时，有 $4\times 4=16$ 个 pair：
- 4 个正样本：$(I_1,T_1), (I_2,T_2), (I_3,T_3), (I_4,T_4)$，label $z=+1$
- 12 个负样本：$(I_1,T_2), (I_1,T_3), \dots$，label $z=-1$

Loss 就是标准 binary cross-entropy，每个 pair 独立算：

$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N}\log\frac{1}{1+e^{z_{ij}(-t\,\mathbf{x}_i\cdot\mathbf{y}_j + b)}}
$$

逐个符号说人话：
- $N$：batch size
- $i, j$：遍历所有 image × text 组合
- $z_{ij}$：pair 的标签，正样本 $+1$，负样本 $-1$
- $\mathbf{x}_i\cdot\mathbf{y}_j$：cosine similarity
- $t=\exp(t')$：temperature，和 CLIP 一样 learnable，初始 $t'= \log 10$，即 $t=10$
- $b$：bias，learnable，初始 $b=-10$。这个是关键，下面单独讲
- $-t\,\mathbf{x}_i\cdot\mathbf{y}_j + b$：这是 logit 的负数形式。正样本 $z=+1$ 时希望 $t\,\mathbf{x}\cdot\mathbf{y} - b$ 大；负样本 $z=-1$ 时希望 $t\,\mathbf{x}\cdot\mathbf{y} - b$ 小
- $\log\frac{1}{1+e^{\cdot}}$：就是 $-\log\sigma(\cdot)$ 的另一种写法，$\sigma$ 是 sigmoid

**和 softmax 的本质区别**：

softmax 是"从 N 个里选 1 个"，分母把所有 N 个耦合在一起。sigmoid 是"给每个 pair 打 yes/no"，每个 pair 独立。没有分母，没有归一化，没有全局耦合。

---

## 为什么必须有 bias $b=-10$？这是全篇最关键的 trick

想想 batch=16k 的情况。正样本 16k 个，负样本 $16k\times 16k - 16k \approx 268M$ 个。比例 1:16384。

刚初始化时，image encoder 和 text encoder 都是随机的，$\mathbf{x}\cdot\mathbf{y}\approx 0$（单位球面上两随机向量期望 cos 相似度为 0）。

**情况一：$b=0$**

logit $\approx t\cdot 0 - 0 = 0$，sigmoid(0) = 0.5。

每个 pair 的 loss = $-\log(0.5) \approx 0.69$。

总 loss $\approx 16k \times 0.69 + 268M \times 0.69 \approx 185M$。负样本贡献占 99.99%。

梯度被负样本完全主导，第一步 update 就把参数往"让所有东西都判成负"的方向猛推。训练直接炸。

**情况二：$b=-10$，$t=10$**

logit $\approx 10\cdot 0 - (-10) = 10$。等等，公式里是 $-t\,\mathbf{x}\cdot\mathbf{y} + b$，所以 logit（sigmoid 输入）$\approx -10\cdot 0 + (-10) = -10$。sigmoid(-10) $\approx 4.5\times 10^{-5}$。

注意 $1/16384 \approx 6.1\times 10^{-5}$，和 $4.5\times 10^{-5}$ 非常接近！

意思就是：**初始时模型"默认认为每个 pair 大概率是负样本"，这个"大概率"恰好等于真实正样本比例**。

此时：
- 正样本 loss：$-\log(4.5\times 10^{-5}) \approx 10$
- 负样本 loss：$-\log(1 - 4.5\times 10^{-5}) \approx 4.5\times 10^{-5}$
- 单条正样本贡献 / 单条负样本贡献 $\approx 2.2\times 10^5$
- 正样本总数 / 负样本总数 $\approx 1/16384$
- 两者乘起来大致平衡，loss 不会爆

Table 4 的 ablation 数据说话：

| $b$ 初始 | $t'$ 初始 | ImageNet 0-shot | Pet 0-shot | C100 0-shot |
|---|---|---|---|---|
| 无 bias | $\log 10$ | 62.0 | 81.8 | 59.9 |
| **-10** | $\log 10$ | **63.0** | **82.4** | **61.0** |
| -10 | $\log 1$ | 61.0 | 80.0 | 60.4 |
| 0 | $\log 10$ | 61.7 | 79.9 | 59.0 |
| 0 | $\log 1$ | 53.7 | 73.2 | 53.8 |

有 bias 和没 bias 差 1.3%，bias 初始 0 vs -10 差 1.3%。尤其当 $t$ 也小（$\log 1$，$t=1$）时，没 bias 直接崩到 53.7%，有 bias 还有 61.0%。

**人话总结**：极端不平衡的二分类里，模型初始得有个"先验偏置"编码"大多数都是负样本"这个事实。否则梯度被多数类淹没，训练早期就飞了。这个 trick 适用于任何极端不平衡的 binary classification 场景。

---

## 工程红利：Chunked Implementation，让 1M batch 成为可能

sigmoid loss 因为是"加性"的（每个 pair 独立相加），可以拆开算。softmax 有除法分母，拆不开。

假设 4 个 device，每个 device 持有 $b$ 个 image 和 $b$ 个 text，全局 batch $N=4b$。

### softmax 怎么算

1. all-gather：所有 device 把自己的 image embedding 拉到所有 device，每个 device 都有完整的 $[N, d]$
2. all-gather：text 同样
3. 在每个 device 上算 $[N, N]$ 相似度矩阵
4. 行 softmax + 列 softmax，各自需要 max-reduce 和 sum-reduce（更多通信）

内存峰值：$O(N^2)$，通信 2 次 all-gather + 2 次 all-reduce。

### sigmoid 怎么算（chunked）

1. 每个 device 先算自己本地的 $[b, b]$ block（$b$ 个正样本 + $b(b-1)$ 个负样本），累加 local loss
2. **collective permute**：device 1 的 text 流到 device 2，device 2 流到 device 3，环状交换。每个 device 现在 image 还是自己的，text 是邻居的
3. 算新的 $[b, b]$ block loss，累加
4. 重复 D 次（D=device 数），每个 device 已经看过"自己的 image × 所有 device 的 text"
5. 最后一次 cross-device sum 把所有 local 累加器加起来

对比：

| 维度 | Softmax | Chunked Sigmoid |
|---|---|---|
| 通信 | 2 次 all-gather + 2 次 all-reduce | D 次 collective permute |
| 内存峰值 | $O(N^2)$ | $O(b^2)$，$b=N/D$ |
| 数值稳定 | 需 subtract-max | log-sigmoid 天然稳定 |
| 对称性 | 两次 softmax | 一次计算 |

$N=1M$，$D=1024$ 时，$b=1024$。每个 device 只需 materialize $1024\times 1024$ 的矩阵，约 4MB。softmax 要 materialize $10^6\times 10^6$，4TB，根本不可能。

而且 D 次小通信通常比 2 次大 all-gather 快——all-gather 延迟随数据量线性增长，collective permute 每次只传 $b$ 个 embedding，固定小包，ring 拓扑下效率高。

**人话**：sigmoid 因为没有"除法归一化"，每个 pair 独立可加，所以能"分块算、最后加"。softmax 必须看到全貌才能算分母，没法拆。这一个数学性质决定了工程上能不能 scale。

---

## Batch Size 的真相：32k 就够了，再大反而 hurt

这是 paper 最反直觉的发现。大家都以为 batch 越大越好，CLIP 原文都在强调大 batch 的重要性。SigLIP 把 batch 推到 1M，结论是：

### SigLiT 数据（frozen ViT-g image tower，只训 text，18B examples seen）

| Batch | sigmoid | softmax |
|---|---|---|
| 512 | 72.5 | 69.5 |
| 1k | 75.5 | 73.6 |
| 4k | 79.2 | 78.3 |
| 16k | 81.2 | 81.2 |
| 32k | 81.9 | 81.4 |
| 64k | 81.6 | 81.6 |
| 128k | 80.5 | 80.0 |
| 256k | 72.8 | 72.2 |
| 1024k | - | - |

3B examples 时 256k 直接崩到 72.8%。18B examples 时 1024k sigmoid 才到 84.7%。

### SigLIP 数据（from scratch B/16，9B examples）

| Batch | sigmoid | softmax |
|---|---|---|
| 4k | 68.4 | 66.6 |
| 8k | 70.6 | 69.4 |
| 16k | 72.3 | 71.7 |
| **32k** | **73.4** | 72.9 |
| 98k | 73.0 | 73.2 |
| 307k | 71.6 | 72.6 |

sigmoid peak 在 32k，softmax peak 在 98k。再大反而掉点。

### mSigLIP（多语言 100+ 语言，30B examples）

| Batch | INet-0 | XM3600 avg |
|---|---|---|
| 16k | 71.6 | 34.8 |
| **32k** | **73.2** | **34.9** |
| 64k | 73.2 | 34.4 |
| 128k | 73.2 | 33.6 |
| 240k | 73.1 | 32.7 |

多语言 retrieval 在 32k 以后明显下降。

**为什么 32k 是 sweet spot？**

直觉解释：

1. **Hard negatives 饱和**。batch 里有足够多"语义相近但不同"的负样本后，再加更多 easy negatives 没用。paper Figure 6 的实验证明：去掉 easy negatives 不掉点，去掉 hard negatives 崩盘。
2. **优化步数减少**。同样 examples seen，batch 越大 step 越少。256k batch 训 3B examples 只有 ~12k step，32k batch 有 ~94k step。大 batch 需要更长 schedule 才能收敛（Figure 3 证实）。
3. **多语言场景下密度稀释**。batch=240k，100 种语言，平均每种语言 2400 个样本。batch=32k 时平均每种语言 320 个。但多语言 hard negative 是"同语言内语义近但不同"，batch 太大反而让同语言样本密度稀释（被其他语言样本挤占），hard negative 比例下降。

**人话**：32k 已经有 32000 个负样本可选，hard negatives 够用了。再往上堆 batch，省下的"更多负样本"都是 easy 的，对学习没贡献，反而因为 step 数减少让优化不充分。

---

## 小资源训练：4 块 TPUv4 跑出 79.7% ImageNet 0-shot

Table 1 最震撼的一行。用 SigLiT（frozen image tower + 只训 text tower）：

- Image tower：公开的 ViT-AugReg-B/8 (https://arxiv.org/abs/2106.10270)，冻住，预计算 embeddings
- Text tower：12 层 Large Transformer（不是 24 层，省一半）
- Optimizer：Lion (https://arxiv.org/abs/2305.10817)，lr peak $10^{-4}$，weight decay $10^{-7}$
- Schedule：6.5k 步 linear warmup → 65k 步 cosine decay 到 0
- Batch：32k
- 硬件：4 块 TPUv4
- 时间：**1 天**
- 结果：**79.7% ImageNet 0-shot**

换 ViT-g/14 frozen image tower：4 块 TPUv4，2 天，84.5%。

对比：原 CLIP 要 256 块 TPUv3 跑 10 天才 76.2%。FLIP 要 256 块 TPUv3 跑 5 天。

**为什么 4 块卡能跑 32k batch？** 因为 chunked sigmoid 实现内存只需 $O(b^2)$，$b=32k/4=8k$，每个 device materialize $8k\times 8k$ 矩阵约 256MB，完全放得下。softmax 要 all-gather 全部 32k，每个 device 都有 $32k\times 32k$ 矩阵 4GB，4 块卡上放不下。

从头训的 SigLIP：B/16 image + B text，32 块 TPUv4，2 天，72.1%；5 天 73.4%。

---

## 两个工程小 trick

### Trick 1：β2 = 0.95 稳定大 batch 训练

Figure 5 展示的现象：大 batch 训 ViT 时偶发 gradient spike，loss 突然飙升，gradient norm 爆炸，参数被推飞。

Adam/Adafactor 的 update rule：

$$
m_t = \beta_1\,g_{t-1} + (1-\beta_1)g_t
$$
$$
v_t = \beta_2\,v_{t-1} + (1-\beta_2)g_t^2
$$
$$
\theta_t = \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

变量：
- $g_t$：第 $t$ 步 gradient
- $m_t$：一阶动量 EMA
- $v_t$：二阶动量 EMA（gradient 平方的 EMA）
- $\beta_1, \beta_2$：EMA 衰减率
- $\eta$：learning rate
- $\epsilon$：数值稳定小常数

默认 $\beta_2=0.999$，$v_t$ 半衰期约 693 步。当 $g_t$ 突然 spike，$v_t$ 反应慢（被历史平滑了），分母 $\sqrt{v_t}$ 还是小值，结果 $\frac{m_t}{\sqrt{v_t}}$ 爆炸，参数被推飞。

改成 $\beta_2=0.95$，$v_t$ 半衰期约 13.5 步。spike 后 $v_t$ 快速跟上 $g_t$ 量级，分母 $\sqrt{v_t}$ 变大，$\frac{m_t}{\sqrt{v_t}}$ 保持稳定。

**人话**：$\beta_2$ 是"gradient 方差的记忆长度"。$\beta_2=0.999$ 记太久，gradient 突然变大时分母还没反应过来，update 就爆了。$\beta_2=0.95$ 记得短，分母快速跟上 gradient 量级，update 稳定。这个 trick 来自 Kaiming He 的 MAE paper (https://arxiv.org/abs/2111.11394)。

### Trick 2：Fine-tune 时关掉 pretrained 权重的 weight decay

Figure 4 的发现。用 pretrained ViT 初始化 image tower 并 fine-tune 时：

- 默认（所有权重都 weight decay）：fine-tune 效果和 from-scratch 差不多，ImageNet 10-shot linear probe 表征甚至退化
- **关掉 pretrained backbone 的 weight decay**（只对随机初始化的 text tower 用 wd）：71% ImageNet 0-shot，显著提升

直觉：weight decay 把权重往 0 拉。pretrained 视觉表征是有用的非零结构，继续 decay 会慢慢"漂白"它。新初始化的 text tower 需要 wd 正则，pretrained image tower 应该被"信任并保留"。

这个思想和 BiT (https://arxiv.org/abs/1912.11370) 的"BatchNorm 层不衰减"、ViT AugReg (https://arxiv.org/abs/2106.10270) 的 fine-tune 配方一脉相承：**pretrained 的东西别乱动，随机初始化的部分才需要正则**。

---

## 负样本比例实验：Hard negatives 才是真正干活的

Figure 6 的实验。16k batch，正负比 1:16384。用不同策略 mask 负样本：

| 策略 | 效果 |
|---|---|
| Random masking 负样本 | 性能退化 |
| Keep easiest negatives | **完全废掉** |
| Keep hardest negatives | 几乎不掉点 |
| Hard + match total pairs seen（mask 后训更久保持总 pair 数）| 甚至略升 |

观察 bias $b$ 终值：负样本越少，$b$ 越偏正（logit 偏移让正样本更容易"过线"）。只剩 hard negatives 时，正样本平均 logit 几乎不变——说明 hard negatives 主要在"挤压"负样本 logit，不是"推"正样本。

**人话**：contrastive learning 的信号 99% 来自 hard negatives。Easy negatives 是白噪音，去掉不影响，留着只是浪费算力。这和 self-supervised learning 里 hard negative mining 的经典直觉完全一致。paper 说"future work 是找高效的 mining 方法"——这是个开放问题。

---

## 对 Label Noise 鲁棒

Figure 7。人为破坏训练数据：image 换成随机噪声、text 换成随机 tokens、shuffle batch alignment。随破坏概率 $p$ 增加：

- sigmoid loss 始终优于 softmax
- 差距随 $p$ 增大而扩大

直觉：

- sigmoid：错标签只污染那一个 pair 的梯度。其他 pair 不受影响。
- softmax：错配样本进入分母 $\sum_j e^{t\,\mathbf{x}_i\cdot\mathbf{y}_j}$，变成"伪 positive"，把其他正常样本的归一化概率拉低。一个错标签污染整个 batch 的 gradient。

这个性质来自 Beyer et al. 2020 (https://arxiv.org/abs/2006.07159)，他们在 supervised classification 上就发现 sigmoid 对噪声鲁棒。SigLIP 把这个结论迁移到 contrastive setting。

**人话**：web 上爬的 image-text pair 本来就噪声大。sigmoid 让错标签的"污染范围"限制在单个 pair，softmax 让错标签污染整个 batch 的归一化。数据越脏，sigmoid 优势越大。

---

## 多语言 SigLIP 的工程细节：Bottlenecked Token Embedding

100+ 语言需要大 vocab，比如 250k tokens。标准做法 $N\times W$ embedding matrix，$N=250k$，$W=768$，约 192M 参数。

**Bottleneck**：$N\times K$ lookup + $K\times W$ projection。$K=96$。
- 参数：$250k\times 96 + 96\times 768 \approx 24M$
- 省 8 倍
- 质量损失 ~0.5% ImageNet 0-shot

直觉：不同语言 token 间有语义结构相似性。低维 bottleneck 强制 token embedding 学紧凑"语义码"，再 projection 到 transformer 隐空间。本质是 low-rank factorization。

mSigLIP 在 XM3600 (https://arxiv.org/abs/2209.00190) 多语言 retrieval 上：Base 模型 34.9% text-to-image R@1，比之前 4B 参数 ViT-e 的 LiT 28.5% 高 6 个多点。Scaled-up 版（overtraining）42.6% image R@1 / 54.1% text R@1。

---

## 最终大模型结果（Table 3）

32k batch，40B examples "over-training"（参考 LLaMA 的 overtraining 思路 https://arxiv.org/abs/2302.13971）：

| 模型 | Patches | INet-1k | INet-v2 | ReaL | ObjectNet | COCO I→T | COCO T→I |
|---|---|---|---|---|---|---|---|
| SigLIP B/16 | 196 | 76.2 | 69.6 | 82.8 | 70.7 | 64.4 | 47.2 |
| SigLIP B/16 | 1024 | 79.2 | 73.0 | 84.9 | 74.7 | 67.6 | 50.4 |
| SigLIP L/16 | 576 | 82.1 | 75.9 | 87.0 | 81.0 | 70.6 | 52.7 |
| **SigLIP SoViT-400M** | 729 | **83.2** | 77.2 | 87.5 | 82.9 | 70.2 | 52.0 |
| EVA-CLIP-18B (E/14, 5B 参数) | 256 | 82.0 | 75.7 | - | 79.6 | 68.8 | 51.1 |
| OpenCLIP G/14 (2B 参数) | 256 | 80.1 | 73.6 | - | 73.0 | 67.3 | 51.4 |

SoViT-400M 仅 400M 参数，全面碾压 5B 参数的 EVA-CLIP。SoViT 来自 Alabdulmohsin et al. (https://arxiv.org/abs/2305.13460)，是 compute-optimal ViT 形状（patch size、depth、width 比例优化）。

**人话**：loss 改对了 + shape 设计对了 + overtraining，400M 参数干翻 5B 参数。这验证了"训练效率比堆参数重要"的直觉。

---

## Pseudo-code 逐行说人话

```python
# img_emb: [n, dim] — n 张图的 embedding
# txt_emb: [n, dim] — n 个文本的 embedding
# t_prime, b: learnable, 初始 t_prime=log(10), b=-10
# n: batch size

t = exp(t_prime)                    # 保证 t>0, 初始 t=10
zimg = l2_normalize(img_emb)        # 单位球面上
ztxt = l2_normalize(txt_emb)
logits = dot(zimg, ztxt.T) * t + b  # [n,n], 每个元素是 cos*t+b
labels = 2 * eye(n) - ones(n)       # 对角线 1, 其他 -1
l = -sum(log_sigmoid(labels * logits)) / n
```

`labels * logits` 这个技巧：正样本时 label=1，logit 不变，算 $-\log\sigma(\text{logit})$；负样本时 label=-1，logit 取负，算 $-\log\sigma(-\text{logit})$。一行代码统一两种 case。

---

## 整体直觉，用人话最后总结一遍

1. **CLIP 的 softmax loss 把 batch size 和任务难度绑死了**。batch 小任务就简单，学习信号弱。sigmoid 解耦了这个绑定，每个 pair 独立判断，batch 小也有强信号。

2. **bias $b=-10$ 是对极端不平衡的先验补偿**。正负比 1:16384，没 bias 的话初始梯度被负样本淹没，训练直接飞。bias 让初始 sigmoid 输出接近真实正样本比例，训练从一个"合理起点"出发。这个 trick 适用于任何极端不平衡 binary classification。

3. **sigmoid 是加性的，softmax 是除法归一化的**。这个数学性质决定了 sigmoid 能 chunked 实现，内存从 $O(N^2)$ 降到 $O(N^2/D^2)$，通信从 all-gather 变成 collective permute。1M batch 因此可能。

4. **32k batch 就够了**。hard negatives 在 32k 里已经足够，再加都是 easy negatives，没用。而且 batch 大 step 少，优化不充分。多语言场景下超大 batch 还稀释同语言 hard negative 密度。这个发现对 practitioner 极具指导意义。

5. **β2=0.95 稳定大 batch 训练**。transformer 训练偶发 gradient spike，$\beta_2=0.999$ 反应慢导致 update 爆，$\beta_2=0.95$ 让二阶动量快速跟上，update 稳定。

6. **Fine-tune 时关掉 pretrained 权重的 weight decay**。pretrained 表征是有用非零结构，继续 decay 会漂白它。只对随机初始化的部分用 wd。

7. **Hard negatives 才是干活的**。Easy negatives 是白噪音，去掉不影响，留着浪费算力。Hard negative mining 是开放问题。

8. **对 label noise 天然鲁棒**。错标签在 sigmoid 里只污染单个 pair，在 softmax 里污染整个 batch 归一化分母。数据越脏 sigmoid 优势越大。

9. **SoViT-400M + SigLIP loss + overtraining = 83.2% ImageNet 0-shot**，400M 参数干翻 5B 参数的 EVA-CLIP。训练效率比堆参数重要。

**一句话哲学**：CLIP 之所以 work 不是因为 softmax，是因为"image-text 互信息"这个 inductive bias。softmax 只是实现手段，sigmoid 实现得更简洁、更省内存、更鲁棒、和 batch size 解耦。把 loss 想清楚比堆 trick 重要。

相关链接：
- SigLIP paper: https://arxiv.org/abs/2303.15343
- CLIP: https://arxiv.org/abs/2103.00020
- LiT: https://arxiv.org/abs/2111.07991
- InfoNCE/CPC: https://arxiv.org/abs/1807.03748
- big_vision 代码: https://github.com/google-research/big_vision
- OpenCLIP: https://github.com/mlfoundations/open_clip
- EVA-CLIP: https://arxiv.org/abs/2303.15389
- FLIP: https://arxiv.org/abs/2212.00794
- MAE (β2=0.95): https://arxiv.org/abs/2111.11394
- Lion optimizer: https://arxiv.org/abs/2305.10817
- SoViT: https://arxiv.org/abs/2305.13460
- WebLI/PaLI: https://arxiv.org/abs/2209.06794
- XM3600: https://arxiv.org/abs/2209.00190
- Beyer et al. sigmoid 鲁棒性: https://arxiv.org/abs/2006.07159
- AugReg ViT: https://arxiv.org/abs/2106.10270
- BiT: https://arxiv.org/abs/1912.11370

---

# SigLIP: Sigmoid Loss for Language-Image Pre-Training 深度解析

## 1. 一句话直觉

把 CLIP 的 InfoNCE softmax contrastive loss 换成 **pairwise sigmoid binary classification**。每个 image-text pair 独立打分，不再做 batch 内的全局归一化。这个看似简单的改动让 loss 与 batch size **decouple**，由此带来：内存效率、更小 batch 也 work、更大 batch 可达 1M、对噪声更鲁棒、并且实现上不需要 all-gather。

paper 链接: https://arxiv.org/abs/2303.15343
代码: https://github.com/google-research/big_vision

---

## 2. 标准 Softmax Contrastive Loss 回顾（找出它的"病"）

给定一个 mini-batch $\mathcal{B} = \{(I_1, T_1), \dots, (I_n, T_n)\}$，CLIP 优化目标：

$$
-\frac{1}{2|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\left(
\underbrace{\log\frac{e^{t\,\mathbf{x}_i\cdot\mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|}e^{t\,\mathbf{x}_i\cdot\mathbf{y}_j}}}_{\text{image→text softmax}}
+
\underbrace{\log\frac{e^{t\,\mathbf{x}_i\cdot\mathbf{y}_i}}{\sum_{j=1}^{|\mathcal{B}|}e^{t\,\mathbf{x}_j\cdot\mathbf{y}_i}}}_{\text{text→image softmax}}
\right)
$$

变量含义：
- $\mathbf{x}_i = f(I_i)/\|f(I_i)\|_2$：image embedding，L2-normalize 后落在单位球面
- $\mathbf{y}_i = g(T_i)/\|g(T_i)\|_2$：text embedding，同样 normalize
- $t = \exp(t')$：temperature，$t'$ 是 learnable 的标量，初始常取 $\log(10)$，即 $t=10$
- 系数 $1/2$：因为 image→text 和 text→image 两个方向各算一遍，每个样本算了两次

**这里的"病"在哪？**

1. **需要全局归一化**：分母 $\sum_j e^{t\,\mathbf{x}_i\cdot\mathbf{y}_j}$ 要求每个样本"看到"整个 batch。
2. **数值稳定性需要 subtract-max**：标准 softmax 数值上溢，必须先 max-subtraction，这又是一次遍历整个 batch。
3. **分布式实现要 all-gather**：所有 device 上 image 和 text embeddings 都要全部聚合，然后 materialize $|\mathcal{B}|\times|\mathcal{B}|$ 的相似度矩阵，内存 $O(|\mathcal{B}|^2)$。
4. **不对称性**：image→text 和 text→image 是两次独立 softmax（先按行归一化，再按列归一化），概念上别扭。
5. **batch size 与"任务定义"耦合**：softmax 把任务定义为"$N$-way classification，从 $N$ 个候选里挑 1 个对的"，这本质上是 categorization，而 $N$ 就是 batch size。改 batch size 就改了任务难度。

直觉：softmax 让 loss 隐式说"我看到的正样本要在所有我见过的样本里最像"。这种"我见过的样本集合"就是 batch 本身。batch 小，任务简单（"挑 1 个出 8 个"）；batch 大，任务难（"挑 1 个出 32k 个"）。这导致小 batch 训练信号弱。

---

## 3. Sigmoid Loss 的核心思想

把"从 $N$ 个里挑 1 个"重新 cast 成"给每个 pair 打 binary 标签"。即，把 contrastive learning 视作**所有 pair 组合上的二分类**：

- $|\mathcal{B}|$ 个正样本 pair：$(I_i, T_i)$，label $z_{ij}=+1$
- $|\mathcal{B}|^2 - |\mathcal{B}|$ 个负样本 pair：$(I_i, T_{j\neq i})$，label $z_{ij}=-1$

Loss：

$$
\mathcal{L} = -\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\sum_{j=1}^{|\mathcal{B}|}\log\frac{1}{1+e^{z_{ij}(-t\,\mathbf{x}_i\cdot\mathbf{y}_j + b)}}
$$

逐项拆解：

- **外层 $1/|\mathcal{B}|$**：除以 image 数（正样本数），不除以 pair 数。这让 loss 量级不随 $|\mathcal{B}|^2$ 爆炸，但负样本对总贡献仍远大于正样本——这正是为什么需要 bias $b$。
- **内层 $\sum_{j}$**：第 $i$ 个 image 和 batch 内所有 text 配对，每个 pair 独立算一个 binary cross-entropy。
- **$z_{ij}\in\{+1,-1\}$**：pair 的标签。
- **logit = $t\,\mathbf{x}_i\cdot\mathbf{y}_j - b$**：cosine 相似度乘以 temperature 再减去 bias。注意符号，$z_{ij}\cdot(-t\,\mathbf{x}_i\cdot\mathbf{y}_j+b)$，正样本时 $z=+1$，希望 $t\,\mathbf{x}\cdot\mathbf{y}-b$ 大；负样本时 $z=-1$，希望 $t\,\mathbf{x}\cdot\mathbf{y}-b$ 小。
- **$t = \exp(t')$**：和 CLIP 一样，learnable temperature，初始 $t'=\log 10$，即 $t=10$。
- **$b$**：learnable bias，初始 $b=-10$。**这是这篇 paper 的关键技术点**，下面详细解释。

### 3.1 为什么必须要有 bias $b$？

正负样本极度不平衡。$|\mathcal{B}|=16k$ 时正样本只有 16k 个，负样本有 $16k^2 - 16k \approx 268M$ 个，比例 ~1:16384。

考虑随机初始化时 $\mathbf{x}\cdot\mathbf{y}\approx 0$（球面上随机两向量期望 cos=0），logit $\approx -b$。

- 如果 $b=0$：所有 pair 的 logit $\approx 0$，sigmoid $\approx 0.5$。
  - 正样本 loss: $-\log(0.5) \approx 0.69$
  - 负样本 loss: $-\log(1-0.5) \approx 0.69$
  - 每个样本总 loss = $|\mathcal{B}|\cdot 0.69$（正）+ $(|\mathcal{B}|-1)\cdot |\mathcal{B}|\cdot 0.69$（负）
  - 负样本贡献占绝对主导，loss 极大，导致早期梯度巨大、过度修正。
- 如果 $b=-10$ 且 $t=10$：logit $\approx -10$，sigmoid $\approx 4.5\times 10^{-5}$。
  - 这正好接近先验概率 $1/|\mathcal{B}| = 1/16384 \approx 6.1\times 10^{-5}$
  - 意思：训练初始时模型"默认认为每个 pair 大概率是负样本"，这恰好与真实先验匹配。
  - 此时正样本 loss: $-\log(4.5\times 10^{-5})\approx 10$；负样本 loss: $-\log(1-4.5\times 10^{-5})\approx 4.5\times 10^{-5}$
  - 正负样本的"单条贡献"量级差异约 $10/4.5\times 10^{-5}\approx 2.2\times 10^5$，与负样本数 $|\mathcal{B}|-1\approx 16384$ 的乘积大致平衡——loss 不会爆。

直觉：$b$ 起到一个"对样本不平衡的先验补偿"作用，让训练从一个合理的初始状态出发，避免 SGD/Adam 在前几个 step 把参数拉飞。Table 4 的 ablation 完美证实了这点：$b=-10$ vs $b=0$ 在 ImageNet 0-shot 上差了 1.3%。

> 这是一个非常重要的设计直觉：在极端不平衡的二分类里，要给模型一个能 encode 先验的"偏置项"，否则梯度会被多数类淹没。

---

## 4. Chunked Implementation: 内存与通信的精彩拆解

这是 sigmoid loss 的"工程红利"，也是它能扩展到 1M batch 的根本原因。

### 4.1 Softmax 实现的痛点

D 个 device，每个 device 持有 $b=|\mathcal{B}|/D$ 个 image 和 text embeddings。要算 softmax，每个 device 必须：
1. all-gather 所有 image embeddings → 形状 $[|\mathcal{B}|, d]$ 在每个 device 都有副本
2. all-gather 所有 text embeddings → 同上
3. 计算 $|\mathcal{B}|\times|\mathcal{B}|$ 的相似度矩阵 → 内存 $O(|\mathcal{B}|^2)$
4. 对行 softmax 和列 softmax → 各需要一次 max-reduce（额外通信）

$|\mathcal{B}|=1M$ 时相似度矩阵 $10^{12}$ 个 float，绝对无法 materialize。

### 4.2 Sigmoid 的 chunked 思路

Loss 可以重写为：

$$
\mathcal{L} = -\frac{1}{|\mathcal{B}|}\underbrace{\sum_{d_i=1}^{D}}_{A:\,\text{sum across devices}}\underbrace{\sum_{d_j=1}^{D}}_{B:\,\text{swap negatives across devices}}\underbrace{\sum_{i=bd_i}^{bd_i+b-1}\sum_{j=bd_j}^{bd_j+b-1}\mathcal{L}_{ij}}_{C:\,\text{local computation on device } d_i}
$$

关键：sigmoid loss 每个 pair 是独立项，可以"哪里算、哪里累加"。

Figure 1 的 mock（3 device，全局 batch 12）演示流程：

- (a) 每个 device 起始持有 4 个 image + 4 个 text。
- (b) 每个 device 先算自己本地的 $4\times 4$ block（包括 4 个 positive + 12 个 negative），累加 local loss。
- (c) 用 **collective permute**（不是 all-gather！）让 device 1 的 text 流到 device 2，device 2 流到 device 3，环状交换。每个 device 现在 image 还是自己的，text 是邻居的。
- (d) 再算一个新的 $4\times 4$ block loss，累加到之前的累加器。
- 重复 D 次后，每个 device 已经以"自己的 image × 所有 device 的 text"视角看全了一轮。最后一次 cross-device sum 把所有 local 累加器加起来。

### 4.3 这个设计的精妙之处

| 维度 | Softmax | Chunked Sigmoid |
|---|---|---|
| 通信原语 | 2 次 all-gather（image, text）+ 2 次 all-reduce（row/col max + sum） | D 次 collective permute |
| 内存峰值 | $O(|\mathcal{B}|^2)$（materialize 全矩阵） | $O(b^2)$（仅 $b\times b$ 局部 block） |
| 数值稳定 | 需 subtract-max | log-sigmoid 天然稳定 |
| 对称性 | 两次 softmax（不对称） | 一次计算，对称 |
| 多 device 性能 | all-gather 在大 batch 时延迟高 | D 次小通信通常比 2 次大通信快 |

> 直觉：sigmoid 因为是"加性"的，可以拆解；softmax 因为有"除法"（归一化），分母耦合整个 batch，没法拆。这是这篇 paper 在工程上最漂亮的点。

---

## 5. 实验全景：batch size 怎么影响？

### 5.1 SigLiT（locked-image，仅训 text tower）

使用预训练的 ViT-g vision encoder，冻住，只训 text encoder。Figure 2 (左) 是核心：

- batch 512 时：sigmoid ~75%，softmax ~70%，差 5%
- batch 8k 时：sigmoid ~80.8%，softmax ~79.7%
- batch 16k 时：sigmoid ~81.2%，softmax ~81.2%（持平）
- batch 32k 时：sigmoid ~81.9%，softmax ~81.4%
- batch 256k 时：sigmoid ~72.8%，softmax ~72.2%（**都崩了**！）
- batch 1024k 时：sigmoid ~84.7%（在 18B examples 下）

注意表 8 (Appendix) 更完整。3B/18B examples 下，1024k 的 sigmoid 达到 84.7%。但短 schedule 下大 batch 反而差（Figure 3）——梯度更新次数太少。

### 5.2 SigLIP（从头训）

Table 5 (Appendix) 完整数据。9B examples：

| Batch | sigmoid | softmax |
|---|---|---|
| 4k | 68.4 | 66.6 |
| 8k | 70.6 | 69.4 |
| 16k | 72.3 | 71.7 |
| 32k | **73.4** | 72.9 |
| 98k | 73.0 | 73.2 |
| 307k | 71.6 | 72.6 |

sigmoid 的 peak 在 32k，softmax 的 peak 在 98k。**sigmoid 用 1/3 的 batch size 达到比 softmax 还略高的 peak**。再往上 batch 反而 hurt。

### 5.3 mSigLIP（多语言，100+ 种语言）

Table 2：

| BS | INet-0 | XM avg |
|---|---|---|
| 16k | 71.6 | 34.8 |
| 32k | **73.2** | **34.9** |
| 64k | 73.2 | 34.4 |
| 128k | 73.2 | 33.6 |
| 240k | 73.1 | 32.7 |

> 关键发现：32k 就够了。再大反而 hurt（特别是多语言 retrieval）。推翻了"batch 越大越好"的迷思。

直觉：batch 大到一定程度后，"难负样本"已经足够多，再增加只是边际正样本稀释，且大 batch 的训练步数变少（同样 examples seen 下），优化不足。多语言场景下超大 batch 反而让每种语言在同 batch 内的"同类 hard negative"密度降低。

---

## 6. 训练稳定性：β2 = 0.95

Figure 5 揭示了一个有趣现象：大 batch 训练 ViT 时会偶发 gradient spike，导致参数大幅 update，loss 飙升甚至发散。

**原因**：transformer 训练本质不稳定，spike 偶尔出现。Adam/Adafactor 默认 $\beta_2=0.999$ 时，二阶动量 $v_t$ 累积很久的历史，对当前 spike 的"分母"反应慢——结果 spike 当下 $\frac{m_t}{\sqrt{v_t}}$ 变巨大，参数被推飞。

**修复**：$\beta_2=0.95$（来自 He et al. MAE paper 的建议）。$v_t$ 衰减快，spike 后 $v_t$ 快速跟上 $g_t$ 量级，恢复 $\frac{m_t}{\sqrt{v_t}}$ 的稳定性。

直觉上 $\beta_2$ 是"二阶矩的 EMA 半衰期"。$\beta_2=0.999$ 半衰期 ~693 步，$\beta_2=0.95$ 半衰期仅 ~13.5 步。后者让 optimizer 更"活在当下"，spike 后能快速校准分母。

---

## 7. 工程配方：在 4 块 TPUv4 上跑出 79.7% ImageNet 0-shot

Table 1 第一行最震撼。**4 块 TPUv4，1 天，79.7% ImageNet 0-shot**。这把 CLIP 训练从"几百卡"打到"单机 4 卡"。配方：

- Image tower：frozen 公开 ViT-AugReg-B/8（https://arxiv.org/abs/2106.10270），预计算 embeddings 加速
- Text tower：12 层 Large Transformer（不是 24 层）
- Optimizer：Lion（https://arxiv.org/abs/2305.10817），lr 峰值 $10^{-4}$，weight decay $10^{-7}$
- Schedule：6.5k 步 linear warmup → 65k 步 cosine decay 到 0
- Batch size：32k（4 块卡上跑 32k batch 在 softmax loss 下绝无可能）

升级到 ViT-g/14 frozen image tower：**2 天 4 卡 84.5% ImageNet 0-shot**。

从头训的 SigLIP：B/16 + B text，16 卡 3 天 71.0%；32 卡 2 天 72.1%；32 卡 5 天 73.4%。

---

## 8. Locked-image Tuning 的反直觉发现：关掉预训练权重的 weight decay

Figure 4 是另一处"看似微小但极重要"的工程发现。

当用 pretrained ViT 初始化 image tower 并 fine-tune（unlocked fine-tuning，区别于 LiT 的 frozen）时：

- **默认配方**（weight decay 应用到所有权重）：fine-tuning 几乎不比 from-scratch 好。ImageNet 10-shot linear probe 表征甚至退化。
- **关掉 pretrained backbone 的 weight decay**（只对随机初始化的 text tower 应用 wd）：71% ImageNet 0-shot，显著提升。

直觉：weight decay 本质是"把权重往 0 拉"，但 pretrained 的视觉表征是有用的非零结构。继续衰减会慢慢"漂白"它，破坏学到的特征。新初始化的部分（text tower）需要 wd 正则，但 pretrained 部分应该被"信任并保留"。这与 ViT fine-tuning 配方 (https://arxiv.org/abs/2106.10270) 和 BiT (https://arxiv.org/abs/1912.11370) 中的"BatchNorm 不衰减"思想一脉相承。

---

## 9. 负样本比例实验：什么真正在驱动学习？

Figure 6 把 16k batch 中的 1:16384 正负比 mask 成不同比例：

- **Random masking 负样本**：性能退化。说明"随机稀释负样本"会丢失信息。
- **Keep easiest negatives**：完全废掉。说明 easy negatives 对学习几乎没贡献。
- **Keep hardest negatives**：几乎不掉点。说明 hard negatives 才是真正驱动学习的关键。
- **Hard + match total pairs seen**（mask 后训练更久以保持总 pair 数）：甚至略升。

> 直觉：contrastive learning 的"信号"主要来自 hard negatives。Easy negatives 是"白噪音"，把它们去掉省下的计算量可以投入到更多 hard negatives 上。这和 hard negative mining 的经典直觉吻合，但 paper 没给出有效的 mining 方法（"future work"）。

观察 bias $b$ 的终值：负样本越少，$b$ 越偏正（logit 偏移让正样本更容易"过线"）。当只剩 hard negatives 时，正样本的平均 logit 几乎不变——说明 hard negatives 主要在"挤压"负样本 logit 而非"推"正样本。

---

## 10. 对 Label Noise 的鲁棒性

Figure 7：人为破坏训练数据（替换 image 为噪声、替换 text 为随机 tokens、shuffle batch alignment），随破坏概率 $p$ 增加，sigmoid loss 始终优于 softmax baseline，且差距随 $p$ 增大而扩大。

直觉来源：sigmoid loss 把每个 pair 当独立 binary 分类，错标签只污染那一个 pair 的梯度。softmax 的分母耦合整个 batch，错标签会"污染"所有正常样本的归一化分布——一个错配的样本会变成"伪 positive"，把其他正常样本的相似度拉低。

参考 Beyer et al. 2020 (https://arxiv.org/abs/2006.07159)：sigmoid 在 supervised classification 上对噪声本就鲁棒，本文把这个结论迁移到 contrastive setting。

---

## 11. 大模型 scaling 结果（Table 3）

在 32k batch、40B examples seen 的"over-training"配方下（参考 Touvron et al. LLaMA "overtraining" 思路 https://arxiv.org/abs/2302.13971）：

| 模型 | Patches | INet-1k | INet-v2 | ReaL | ObjectNet | COCO I→T | COCO T→I |
|---|---|---|---|---|---|---|---|
| SigLIP B/16 | 196 | 76.2 | 69.6 | 82.8 | 70.7 | 64.4 | 47.2 |
| SigLIP B/16 | 1024 | **79.2** | 73.0 | 84.9 | 74.7 | 67.6 | 50.4 |
| SigLIP L/16 | 576 | **82.1** | 75.9 | 87.0 | 81.0 | 70.6 | 52.7 |
| SigLIP SoViT-400M | 729 | **83.2** | 77.2 | 87.5 | 82.9 | 70.2 | 52.0 |
| EVA-CLIP-18B (E/14) | 256 | 82.0 | 75.7 | - | 79.6 | 68.8 | 51.1 |

SoViT-400M 仅 400M 参数 + 729 patches，**全面碾压 5B 参数的 EVA-CLIP**。SoViT 来自 Alabdulmohsin et al. (https://arxiv.org/abs/2305.13460)，是 compute-optimal ViT 形状（patch size、depth、width 比例优化）。

---

## 12. 多语言 SigLIP 的工程细节：Bottlenecked Token Embedding

100+ 语言需要大 vocab，比如 250k tokens。标准做法是 $N\times W$ embedding matrix，$N=250k$，$W=768$。这是 $250k\times 768\approx 192M$ 参数，且 gradient 优化时是稀疏更新但内存占用大。

**Bottleneck**：用 $N\times K$ lookup + $K\times W$ projection。$K=96$，$W=768$。
- 参数量：$250k\times 96 + 96\times 768 \approx 24M + 0.07M \approx 24M$
- 相比 $192M$ 节省 8 倍
- 质量损失 ~0.5% ImageNet 0-shot（vs full vocab）

直觉：不同语言 token 之间有结构相似性（语义/词源），低维 bottleneck 强制 token embedding 学习紧凑的"语义码"，再用 projection 映射到 transformer 隐空间。本质是个 low-rank factorization。

mSigLIP 在 XM3600（https://arxiv.org/abs/2209.00190）上的 SOTA：Base 模型 34.9% text-to-image R@1，比之前用 4B 参数 ViT-e 的 LiT 28.5% 高出 6 个多点。scaled-up 版（"overtraining"）达到 42.6%/54.1%。

---

## 13. Algorithm 1 伪代码逐行解析

```python
# img_emb: [n, dim] — image embeddings, n=batch size
# txt_emb: [n, dim] — text embeddings
# t_prime, b: learnable temperature log-param and bias
# n: mini-batch size

5  t = exp(t_prime)             # t > 0 始终, 防止符号翻转
6  zimg = l2_normalize(img_emb) # 单位球面
7  ztxt = l2_normalize(txt_emb)
8  logits = dot(zimg, ztxt.T) * t + b
                               # [n, n] 矩阵, logit = t·cos + b
9  labels = 2 * eye(n) - ones(n)
                               # 对角线 1, 其他 -1
10 l = -sum(log_sigmoid(labels * logits)) / n
                               # log_sigmoid(x) = log(1/(1+e^-x)) = -softplus(-x)
                               # labels * logits: 正样本 logit 不变, 负样本 logit 取负
                               # 也就是: 正样本希望 logits 大, 负样本希望 logits 小
```

注意 line 9 的 `labels * logits` 这个技巧：它把 positive case 的 $-\log\sigma(\text{logit})$ 和 negative case 的 $-\log\sigma(-\text{logit})$ 统一成 $-\log\sigma(z\cdot\text{logit})$。数学上等价，工程上一行实现。

---

## 14. 整体直觉总结

把这篇 paper 提炼成几个"为什么 work"：

1. **任务重新定义**：从 "$N$-way categorization"（耦合 batch size）变成 "pairwise binary classification"（与 batch size 解耦）。这让小 batch 也有强学习信号——不再"小 batch 就是简单任务所以信号弱"。
2. **Bias term 编码先验**：$b=-10$ 让初始 sigmoid 输出接近真实正样本比例 $1/|\mathcal{B}|$，避免初始化阶段被负样本主导导致参数飞。
3. **加性可拆解**：sigmoid loss 没有"除法归一化"，每个 pair 独立可加，从而能做 chunked ring-communication，内存从 $O(|B|^2)$ 降到 $O(b^2)$，使 1M batch 可达。
4. **没有"看到 batch 全貌"的硬约束**：softmax 的数值稳定要 max-subtract（额外 pass），sigmoid 用 log-sigmoid 天然稳定。
5. **对噪声天然鲁棒**：错标签只污染单个 pair 的梯度，不污染整个 batch 的归一化分母。
6. **饱和点在 32k**：实证发现 batch 大到 32k 后收益骤减，再大反而 hurt（优化步数不够 + hard negatives 密度稀释）。这对 practitioner 极具指导意义——不再盲目追求大 batch。
7. **β2 = 0.95** + **关闭 pretrained wd**：两个工程配方细节，分别稳定大 batch 训练、保住预训练表征。

**paper 的哲学**：与其在 InfoNCE 上堆 trick，不如换一个更简洁的 loss，让"任务定义"和"硬件规模"真正解耦。这与简化 loss surface 的研究风格一致——CLIP 之所以 work 不在于 softmax，而在于"image-text 互信息"这个 inductive bias；softmax 只是实现手段，sigmoid 实现得更优雅。

更多相关 reading：
- 原 paper: https://arxiv.org/abs/2303.15343
- CLIP: https://arxiv.org/abs/2103.00020
- LiT: https://arxiv.org/abs/2111.07991
- InfoNCE / CPC: https://arxiv.org/abs/1807.03748
- OpenCLIP 复现: https://github.com/mlfoundations/open_clip
- big_vision 代码: https://github.com/google-research/big_vision
- EVA-CLIP: https://arxiv.org/abs/2303.15389
- FLIP: https://arxiv.org/abs/2212.00794
- MAE (β2=0.95 来源): https://arxiv.org/abs/2111.11394
- Lion optimizer: https://arxiv.org/abs/2305.10817
- SoViT (shape-optimal ViT): https://arxiv.org/abs/2305.13460
- WebLI / PaLI: https://arxiv.org/abs/2209.06794
- XM3600: https://arxiv.org/abs/2209.00190
- Supervised sigmoid loss 对噪声鲁棒性 (Beyer et al.): https://arxiv.org/abs/2006.07159
- AugReg ViT (used as frozen image tower): https://arxiv.org/abs/2106.10270
- BiT (no weight decay on pretrained): https://arxiv.org/abs/1912.11370
