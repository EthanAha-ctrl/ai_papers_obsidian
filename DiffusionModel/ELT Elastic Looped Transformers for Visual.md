---
source_pdf: ELT Elastic Looped Transformers for Visual.pdf
paper_sha256: cbeaf98d44a013d84c37c978425db92971fa0ef5e7f01edc3210021e5e930381
processed_at: '2026-08-04T03:09:20-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ELT

---

## 核心就一句话

**同一坨 transformer 层，反复跑好几遍，参数量省了，图照样画得好。**

---

## 为什么这事不 trivial

你本能会想：一个 block 跑 8 遍跟 8 个不同 block 跑一遍，不都是 8 层深度吗？有啥区别？

区别在 **参数量**。8 个不同层要存 8 份权重，1 个层跑 8 遍只存 1 份。参数少 → memory 少 → 推理快（权重能塞进 SRAM 不用从 HBM 反复搬）。

这个 idea 本身不新，2018 年 Universal Transformers 就提了。问题是 **7 年来没人把它在视觉生成上用 work**。

---

## 为什么之前用不起来

你训一个 looped transformer，训练时固定跑 8 遍（L=8），loss 只监督第 8 遍的输出。

中间第 1、2、3...7 遍的 hidden state？**没人管它们**。模型爱把它们变成啥样就变成啥样，反正最后一步能修正回来就行。

结果就是：推理时你想省算力只跑 4 遍，发现第 4 遍的输出是坨垃圾——因为训练时第 4 遍从来没被要求"画得像样"。

Figure 1 右边那些狗图就是这意思：只有 L=8（跟训练对齐）清晰，L=2/4/6/10 全是糊的。

---

## ELT 怎么解决的

训练时加一个 trick：**每次随机挑一个中间遍数，也监督它的输出。**

具体说，训练时同时算两个 loss：
- 第 8 遍的输出要像 ground truth（正常 loss）
- 随机挑的第 3 遍（比如）的输出**也要**像 ground truth，**同时还要 mimic 第 8 遍的输出

后者就是 paper 叫的 **Intra-Loop Self Distillation（ILSD）**——loop 内部的自蒸馏。

---

## 为啥叫"自"蒸馏

因为 teacher 和 student 是**同一个 model、同一份权重**。student 就是 teacher 的"前半段"。

- Teacher = 跑完 8 遍的完整 trajectory
- Student = 只跑到第 3 遍的 prefix trajectory

不是两个 model，是同一个 forward 的不同截断点。所以叫"自"蒸馏。

---

## 最聪明的地方：几乎不花额外算力

普通蒸馏要 forward 两次（student 一次、teacher 一次）。

ILSD 不用。你 forward 到第 3 遍的时候顺手存一下 hidden state，然后继续 forward 到第 8 遍。两次输出都拿到了，但 forward 只跑了一次。

额外开销：**存一个 tensor，多算一个 loss**。基本免费。

---

## 训练完的好处

你拿到一个 model，推理时 L 想填几就填几：
- 算力紧张？L=2，出图快但稍微糙
- 算力充裕？L=10，出图慢但精细

**一次训练，得到一整个 model family**。不用为"云上版""手机版""手表版"分别训三个 model。

这就是 paper 说的 "Any-Time inference"——随时可以叫停，叫停时给当前最好的结果。

---

## 效果有多好

ImageNet 256×256 图像生成：
- MaskGIT-XL：446M 参数，FID 2.0
- ELT-XL：**111M 参数**，FID 2.0

参数量 1/4，质量一样。

UCF-101 视频：
- MAGVIT：306M，FVD 76
- ELT：**76M**，FVD 72.8

参数量 1/4，质量还更好。

---

## 还有个隐藏福利

参数少不等于跑得快（memory bound 才是）。但 ELT 因为参数小到能塞进加速器的 SRAM，不用反复从 HBM 搬权重。

实测在 TPU 上吞吐量 3.5×（H scale）。

**这就是 weight sharing 的物理意义**：不是省参数本身，是让权重能住进最快的存储层。

---

## 有啥毛病

1. **N 太小不行**：1 个 unique layer 跑 32 遍，FID 直接 10.30（崩了）。说明一个层再怎么循环也学不出复杂变换，至少得 4-8 个 unique layers 凑一个 block。
2. **L 超过训练时的 L_max 太多也会崩**：训了 8 遍，推理跑 20 遍，会 drift。不过在 UCF-101 上小幅 extrapolate（训 4 遍推理 6 遍）居然还行，ILSD 给了点泛化能力。
3. **训练不省内存**：forward + backward 还是要 unfold 全部 L 遍的 activation。参数省了，activation memory 没省。

---

## 一句话总结 intuition

**让 looped transformer 的每一遍都"有用"，而不只是最后一遍才有用。**

方法是在训练时随机抽查中间遍数的输出，逼它也要画得像样。这样推理时不管跑几遍，每一遍都是"能交差的半成品"。

像写作文：传统 looped transformer 是"写完 8 稿才看，前面 7 稿随便涂鸦"。ELT 是"随机抽查第 3 稿也得能交"。于是每稿都在认真写，随时能交。

---

# ELT: Elastic Looped Transformers for Visual Generation 深度解析

嘿 Andrej,这篇 paper 我觉得正好戳中你这些年关心的几个点: **weight tying / recurrence, test-time compute scaling, parameter efficiency, anytime inference**。下面我尽量展开来讲,把每一层技术细节和背后的 intuition 都拆开。

paper link: https://arxiv.org/abs/2502.05171 (实际编号待确认, 跟 Geiping 的 latent reasoning 在同一个 arxiv batch)

---

## 1. TL;DR

ELT 把一个深度 transformer (比如 48 层) 拆成 **N 个 unique layers 组成的 block,然后把这个 block 重复执行 L 次**,得到 effective depth D = N×L。参数量只跟 N 有关,计算量跟 N×L 有关 — 这就把 parameter count 和 FLOPs 解耦了。

vanilla looped transformer 的致命问题: 训练时固定 L=L_max,中间那些 loop 的输出是"黑箱",只在最终步被监督,所以 inference 时把 L 改成别的值 (比如想省算力),生成质量会塌掉 (Figure 1 右边那些狗图)。

ELT 的关键贡献 **Intra-Loop Self Distillation (ILSD)**: 训练时 dual-path,teacher path 跑 L_max loops,student path 跑 L_int loops (随机采样),让 student 的输出同时 mimic teacher 的输出和 ground truth。因为 student trajectory 是 teacher trajectory 的 strict prefix,所以 forward 一次就把两个都算出来了,几乎零额外开销。

结果: 一次训练得到一整个 model family,inference 时根据硬件预算动态选 L,从 cloud 的高质量到 edge 的低延迟全部 cover。ImageNet 256×256 FID 2.0 (跟 MaskGIT-XL 平手),但参数量 1/4 (111M vs 446M)。

---

## 2. Background: Looped Transformer 是什么

### 2.1 公式定义

设 N 个 unique transformer layers,每个 layer 参数是 $\theta_i$,组成 composite block:

$$g_\Theta(\mathbf{x}) = f_{\theta_N}\big( f_{\theta_{N-1}}(\cdots f_{\theta_1}(\mathbf{x})\cdots)\big)$$

变量含义:
- $f_{\theta_i}$: 第 $i$ 个 transformer layer (attention + MLP), 参数 $\theta_i$
- $g_\Theta$: composite block, 参数集合 $\Theta = \{\theta_1, \theta_2, \dots, \theta_N\}$
- $\mathbf{x}$: 输入 token sequence 的 hidden state

Looped 应用 L 次:

$$F_{(N,L)}(\mathbf{x}) = \underbrace{g_\Theta\big(g_\Theta(\cdots g_\Theta(\mathbf{x}))\big)}_{L \text{ loops}} \equiv g_\Theta^L(\mathbf{x})$$

上标 $L$ 表示 self-composition 次数。下标 $(N, L)$ 表示 "N 个 unique layers × L 次 loop",effective depth = N×L。

### 2.2 跟 standard transformer 的对比

| 维度 | Standard Transformer (N×L depth) | Looped Transformer (N layers, L loops) |
|------|-----------------------------------|---------------------------------------|
| 参数量 | ∝ N×L (每层 unique) | ∝ N (复用同一组) |
| FLOPs | ∝ N×L | ∝ N×L |
| Memory footprint | 大 (要存 N×L 组权重) | 小 (只存 N 组) |
| Expressivity | 每层独立 | 共享参数 + recurrence |

这里有一个微妙的点: 你直觉上会问 "复用同一组参数 L 次, expressivity 真的够吗?" 答案是 Universal Transformers 早就证明过 looped transformer 是 Turing complete 的 (给定足够多 iterations),而且 Saunshi et al. 2025 在 reasoning tasks 上展示 looped transformer 能实现 multi-step gradient descent in-context。所以 recurrence 本身的 expressivity 上限不是瓶颈,真正的瓶颈是 **训练时如何让中间步骤学得有意义**。

参考:
- Universal Transformers: https://arxiv.org/abs/1807.03819
- Saunshi et al. "Reasoning with latent thoughts: On the power of looped transformers": https://arxiv.org/abs/2502.17416
- Gatmiry et al. "Can looped transformers learn to implement multi-step gradient descent": https://arxiv.org/abs/2410.08292
- Geiping et al. "Scaling up test-time compute with latent reasoning": https://arxiv.org/abs/2502.05171

---

## 3. Vanilla Looped Transformer 为什么 Fragile

Figure 1 右边非常直观: 一个用 L_max = 8 训练的 vanilla looped transformer,在 inference 时 L = 2、4、6、10 全部崩坏,只有 L = 8 (跟训练时一致) 才能生成 coherent 图像。

为什么? 因为训练 objective 只监督 $F_{(N, L_{max})}(\mathbf{x})$,中间的 hidden state $g_\Theta^k(\mathbf{x})$ for $k < L_{max}$ 从来没被监督过。模型自由地把这些中间状态塑造成"任何对最终输出有帮助的中间 representation",这些中间 representation **本身不一定要能直接 decode 出合理图像**。

这跟 deep networks 的 "layer-by-layer abstraction" 直觉相悖: 我们一般想象浅层学 edge、中层学 part、深层学 object,但 vanilla looped transformer 的中间状态更像 "经过 k 步迭代后的某个不可解释 latent",直到最后一步才被"修正"到 solution space。

Figure 2 把这个画得很好: vanilla 的 trajectory 只在最后一步 $X_k^{max}$ 进入 solution space (target 图像所在的 manifold),中间的 $X_k^{int}$ 都还在外面飘。ELT + ILSD 的 trajectory 在每一步都有一个"指向 solution space"的"分身"。

这个 failure mode 让我想到你 NanoGPT 那个 nanoGPT-speedrun — 训练时跟 inference 时 compute allocation 不一致,就会出问题。这里也是同一个 idea: **训练 distribution 和 inference distribution 要 match**,不然就 OOD。

---

## 4. Intra-Loop Self Distillation (ILSD) 详解

### 4.1 总损失

每次 training step,先从 uniform distribution 采样 student 的 loop 数:

$$L_{int} \sim \mathcal{U}(L_{min}, L_{max})$$

- $L_{min}$: student loop 的下限 (constrain student 分布,防止太极端的浅 student)
- $L_{max}$: 训练时的最大 loop 数,teacher 永远跑这个
- $\mathcal{U}$: 离散均匀分布

然后 dual-path forward (其实是同一个 forward 的 prefix):

$$\mathcal{L}_\Theta^{ILSD} = \underbrace{\mathcal{L}^{GT}\big(F_{(N, L_{max})}(\mathbf{x}), \mathbf{y}\big)}_{\text{(1) Teacher ground-truth}} + \underbrace{\lambda \cdot \mathcal{L}^{GT}\big(F_{(N, L_{int})}(\mathbf{x}), \mathbf{y}\big)}_{\text{(2) Student ground-truth}} + \underbrace{(1-\lambda) \cdot \mathcal{L}^{dist}\big(F_{(N, L_{int})}(\mathbf{x}), sg(F_{(N, L_{max})}(\mathbf{x}))\big)}_{\text{(3) Intra-Loop Self Distillation}}$$

变量含义:
- $\mathbf{y}$: ground truth (mask 的真实 token,或 diffusion 的 $\mathbf{x}_0$)
- $\lambda$: curriculum weight,训练过程中从 1 linearly decay 到 0
- $sg(\cdot)$: stop-gradient, teacher 不接收来自 distillation 的梯度 (online teacher, 没 EMA)

### 4.2 Curriculum for λ (关键设计)

$\lambda$ 从 1 → 0 线性 decay,这个 schedule 的 intuition:
- **训练初期** $\lambda \approx 1$: student 主要监督 ground truth,因为 teacher 还没训练好,它给的 distillation signal 是噪声。这时 student 用 GT 锚住,避免被烂 teacher 误导。
- **训练后期** $\lambda \approx 0$: student 主要做 distillation,因为 teacher 已经成熟,能给 low-variance 的 target。GT 反而有噪声 (尤其是 diffusion 的 noisy latent),distillation 提供平滑的 target。

paper 说他们对 decay rate 不敏感,"as long as the transition is gradual"。这点让我想到 mean teacher (Tarvainen & Valpola 2017) 里 teacher EMA momentum 也是类似的 curriculum — 早期 teacher 不可靠,后期 teacher 才有意义。这里没用 EMA,直接用同一次 forward 的 L_max 输出当 teacher,因为没有 latency 上的额外成本。

### 4.3 Masked Generative Model 的具体 loss

对 MaskGIT (discrete tokens),用 cross-entropy:

$$\mathcal{L}^{GT} = -\sum_{i \in Mask} \log P_{(N, L_{int})}\big(y_i \mid \mathbf{x}_{mask}\big)$$

$$\mathcal{L}^{dist} = -\sum_{i \in Mask} \sum_{\nu \in \mathcal{V}} P_{(N, L_{max})}\big(\nu \mid \mathbf{x}_{mask}\big) \log P_{(N, L_{int})}\big(\nu \mid \mathbf{x}_{mask}\big)$$

变量:
- $i \in Mask$: 被 mask 的 position index
- $y_i$: position $i$ 的真实 token id
- $\mathbf{x}_{mask}$: 输入的 masked image tokens
- $\mathcal{V}$: tokenizer 的 vocabulary (codebook size = 1024)
- $\nu$: vocabulary 里的某个 token
- $P_{(N, L)}(\cdot \mid \cdot)$: 用 $N \times L$ 配置的 model 给出的 softmax 概率

第二个公式就是标准 KL divergence 的 cross-entropy 形式,teacher 的 soft label 给 student。

### 4.4 Diffusion Model 的具体 loss

对 DiT,用 sigmoid-weighted MSE (来自 Simpler Diffusion, Hoogeboom et al. 2025):

$$\mathcal{L}^{GT} = w(t) \Big\| F_{(N, L)}(\mathbf{x}_t) - \mathbf{x}_0 \Big\|_2^2$$

$$\mathcal{L}^{dist} = w(t) \Big\| F_{(N, L_{max})}(\mathbf{x}_t) - F_{(N, L_{int})}(\mathbf{x}_t) \Big\|_2^2$$

变量:
- $\mathbf{x}_t$: 加噪后的 latent (time step $t$)
- $\mathbf{x}_0$: clean latent (ground truth)
- $w(t)$: time-dependent sigmoid weighting,用于 reweight 不同 noise level 的 loss (min-SNR 思想)
- $L$: 这里 teacher 用 $L_{max}$, student 用 $L_{int}$

注意 teacher 用 v-prediction, 但损失形式这里写成 x_0 prediction 是简化表达。

### 4.5 零额外 forward 开销的妙处

这是 ILSD 最聪明的地方。对比传统 distillation:

```
传统 distillation:
  forward student network  -> student output
  forward teacher network  -> teacher output
  total: 2 forward passes

ILSD:
  forward block 1次 -> student output (at L_int)
  forward block 再 L_max - L_int 次 -> teacher output (at L_max)
  total: 1 forward pass of length L_max (跟 vanilla looped 一样!)
```

所以 ILSD 的额外开销只有一个: **保存 L_int 时刻的 hidden state** (一个 tensor copy),还有多算两个 loss (一次 softmax + reduction,几乎免费)。

这跟 BYOL / DINO 用 EMA teacher 的设计哲学相反: BYOL 要维护一个 momentum teacher (额外 forward + EMA update),换来 student 学到 invariant representation。ILSD 直接借用 forward 的中间状态作为 teacher,牺牲一点 "teacher 的 maturity" (因为 teacher 和 student 同步训练),换来了零额外 forward。

参考:
- Simpler Diffusion: https://arxiv.org/abs/2410.01131 (Hoogeboom et al. 2025)
- Mean teacher: https://arxiv.org/abs/1703.01780
- BYOL: https://arxiv.org/abs/2006.07733
- DINO: https://arxiv.org/abs/2104.14294

---

## 5. Algorithm 解析 (伪代码细读)

### 5.1 Algorithm 1: Training

```python
# 1. 采样 student loop 数 (uniform)
L_int = random.randint(L_min, L_max)

# 2. Forward pass (用 jax.lax.fori_loop, 单次 sequential execution)
F_curr = x
F_int = x  # placeholder
for step in range(L_max):
    F_curr = g_Θ(F_curr)        # 共享参数 block 走一步
    if step == L_int - 1:        # 0-indexed
        F_int = F_curr           # 在 L_int 步时 cache 一下
return F_curr (== F_max), F_int

# 3. Losses
L_max^GT = L^GT(head(F_max), y)
L_int^GT = λ * L^GT(head(F_int), y)
L^distill = (1-λ) * L^dist(F_int, stop_gradient(F_max))

# 4. Total
Loss = L_max^GT + L_int^GT + L^distill
```

几个 implementation 细节值得注意:

**(a) 共享 head**: student 和 teacher 用同一个 MLM head (或者 diffusion 的 prediction head),这进一步强化了"same parameter, varying compute"的哲学。Head 不 loop,只 block loop。这个设计很关键 — 如果 student 和 teacher 用不同 head,distillation 学的是 head 的 alignment,而不是 block 的 alignment。

**(b) jax.lax.cond 的 lazy evaluation**: 代码里用 `jax.lax.cond` 来做"只在 step == L_int - 1 时保存 F_curr",这是 JAX 的 functional programming 写法,等价于一个 if 但可 trace。普通 PyTorch 里直接 if 就行。

**(c) stop_gradient 的位置**: `jax.lax.stop_gradient(F_max)` 在 distillation loss 里包住 teacher,确保 teacher 只被它自己的 GT loss 训练,student 通过 distillation 跟随 teacher。如果忘掉这个,会变成一个互相 drift 的"双星系统"。

### 5.2 Algorithm 2: Any-Time Inference

```python
F = x
for step in range(L):  # L 是 inference 时动态决定的
    F = g_Θ(F)
out = head(F)
return out
```

就这么简单。一个 trained model,任意 L 都能用。这就是"Any-Time"的精髓 — 来自 Zilberstein 1996 的 anytime algorithm 概念 (AI magazine 的老文章): 算法随时可以"被打断"输出,质量随时间单调提升。

参考: Zilberstein "Using anytime algorithms in intelligent systems": https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/1239

---

## 6. 架构图解析 (Figure 3)

Figure 3 左边是 training:
- 输入 $\mathbf{x}_k$ (第 $k$ 个 sampling step 的输入)
- 进入共享 block $g_\Theta$ (橙色方块)
- 循环 L 次,中间第 $L_{int}$ 次的输出 $F_{int}$ 被缓存
- 第 $L_{max}$ 次的输出 $F_{max}$ 是 teacher
- 两个输出都过 shared MLM head
- 三个 loss (teacher GT, student GT, distillation)

右边是 Any-Time inference:
- 同样进入 block
- 但循环次数 $L$ 由 inference 时的算力预算决定
- 任意时刻 exit,过 head 输出 $\mathbf{X}_{k+1}$
- 输出图像假设是 last sampling step ($k = K$)

这里有个细节 paper 没说太清楚: inference 时的 $L$ 是 per sampling step 的,跟 sampling step $K$ 是两个不同的轴:
- $K$: sampling steps (diffusion 的 denoising steps,或者 MaskGIT 的 parallel decoding steps)
- $L$: 每个 sampling step 内的 loop 数

所以总 compute = $K \times L \times N \times (\text{per-layer FLOPs})$。ELT 调的是 $L$ 这一个 axis (intra-step),跟 consistency models 调 $K$ (inter-step) **orthogonal**。这俩可以叠加。

---

## 7. 实验结果深度分析

### 7.1 ImageNet 256×256 主结果 (Table 1)

| Model | FID↓ | IS↑ | # params | # steps | # GFLOPs |
|-------|------|-----|----------|---------|----------|
| MaskGIT-L8 | 2.1 | 270.1 | 303M | 24 | 3.7k |
| MaskGIT-XL8 | 2.0 | 294.8 | 446M | 24 | 3.9k |
| **ELT-L (8N × 3L)** | 2.2 | 254.3 | **101M** | 24 | 3.7k |
| **ELT-L (12N × 2L)** | 2.1 | 281.8 | **152M** | 24 | 3.7k |
| **ELT-XL (7N × 4L)** | 2.0 | 266.1 | **111M** | 24 | 3.9k |

关键观察:
1. ELT-XL (111M) 跟 MaskGIT-XL (446M) FID 同样是 2.0,但参数量 1/4
2. iso-inference-compute 设置下 GFLOPs 几乎一样 (3.7k vs 3.9k)
3. ELT-XL 用 7N × 4L (effective depth = 28,跟 MaskGIT-XL 的 28 layers 一样),所以 depth 也对齐
4. 唯一的差异就是 weight sharing: ELT 7 个 unique layers 重复 4 次,MaskGIT-XL 28 个 unique layers

这个对比设计非常 fair: depth、FLOPs、effective capacity 都对齐,只差 weight sharing。结论很硬: **在 visual generation 任务上,weight sharing 几乎不损失质量**。

### 7.2 DiT 上的结果 (Table 2)

| Model | FID↓ | # params | D |
|-------|------|----------|---|
| DiT - 32 layers | 3.43 | 2.1B | 32 |
| ELT (1N × 32L) | 10.30 | 69M | 32 |
| ELT (4N × 8L) | 3.96 | 271M | 32 |
| ELT (8N × 4L) | 3.16 | 539M | 32 |
| ELT (16N × 2L) | 2.83 | 1.1B | 32 |

非常 interesting 的 ablation:
1. **1N × 32L = 10.30 FID**: 单个 unique layer 重复 32 次,**参数量虽少但表达力严重不足**。这说明 weight sharing 是有极限的 — N 太小时,recurrence 不能完全 compensate。
2. **16N × 2L = 2.83**: 16 个 unique layers 重复 2 次,超过 baseline 的 32 unique layers (FID 3.43)。说明在 iso-depth 下,适当的 weight sharing 反而**有助于** (类似 regularization 效果)。
3. **8N × 4L = 3.16**: 比 baseline 还好,4× 参数 reduction。

这里有一个隐藏的 insight: **N 和 L 的 trade-off**。N 越大,expressivity 越强但参数大; L 越大,compute 越多但参数不变。最优 (N, L) 配比跟 model width $d$ 有关 — Figure 5 显示 d=1536 (G scale) 用 8 unique layers + 适当 loops 就能逼近 48 unique layers 的 baseline。

### 7.3 Pareto Front (Figure 4)

paper 拟合了一条 pareto front:

$$\text{FID} = 1922.5 \cdot G^{-0.95} + 1.48$$

变量:
- $G$: inference GFLOPs
- 指数 $-0.95$: 接近 -1, 说明 FID 跟 GFLOPs 近似 inversely proportional
- 截距 $1.48$: 渐近下限,代表这个 model family 在 ImageNet 256×256 上的 intrinsic FID floor

这个 power-law 形式让我想到 Chinchilla / Kaplan 的 scaling laws — 同样是 power-law,但 axis 不同: 这里是 inference compute (而不是 training compute)。Hoffmann et al. 的 compute-optimal scaling 关注 training,ELT 关注 inference。这是一个值得系统研究的方向: **inference-time scaling laws for generative models**。

参考:
- Chinchilla: https://arxiv.org/abs/2203.15556
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361

paper 还说: "transitioning to the next architecture scale becomes more performant than over-looping smaller models"。换句话说, 沿着 L axis 缩放的边际效益会饱和,需要切到更大 N (更大 d) 才能继续 gain。这跟 LLM scaling 里 "compute better spent on params than context length" 的发现类似。

### 7.4 Throughput (Table 3) — 真正的 engineering win

| ELT config | d_model | Throughput Ratio (vs baseline) |
|------------|---------|-------------------------------|
| 6N × 2L (B) | 768 | 1.0× |
| 8N × 3L (L) | 1024 | 2.9× |
| 7N × 4L (XL) | 1152 | 3.3× |
| 8N × 4L (H) | 1280 | 3.5× |

3.5× 的 throughput gain 是怎么来的? paper 解释: **shared parameters 全部 fit 在 on-chip SRAM**, 避免了 HBM ↔ SRAM 之间的反复 weight transfer。

这个其实戳到 modern accelerators 的核心痛点: 标准 transformer 推理时,weights 从 HBM 加载到 SRAM 的延迟主导 (memory wall, 见 Pinneapple / inference efficiency literature)。Looped transformer 因为参数小,可以**永久驻留在 SRAM**, 算 L 次循环只是反复读 SRAM (1-2 cycle),不需要重新从 HBM fetch。

注意 B scale (768 d_model) 没有 speedup,因为 baseline MaskGIT-B 本身就小到能装进 SRAM。从 L scale 开始 (1024 d_model) 才出现 speedup — 也就是说 ELT 的 speedup 主要在"模型大到装不进 SRAM 但 looped 版本能装"这个 sweet spot。

TPU v6e (Trillium) 的 specs: https://cloud.google.com/tpu/docs/v6e

### 7.5 Video Generation (Table 4, UCF-101)

| Method | FVD↓ | # params | # steps | # GFlops |
|--------|------|----------|---------|----------|
| MAGVIT-L | 76 ± 2 | 306M | 12 | ~4.3k |
| **ELT (6N × 4L)** | **72.8 ± 2.5** | **76M** | 12 | ~4.3k |
| **ELT (6N × 6L)** | **60.8 ± 2.7** | 76M | 24 | ~13k |

video 上 ELT 也 work,而且**参数量 1/4 还更优**。UCF-101 是 data-constrained 的设置 (~13.7M training tokens),paper 说 looped transformer 在这种 regime 下"exhibit robustness against overfitting",起到了 regularization 作用。

这个观察跟你之前提过 "深度模型在小数据上容易过拟合,recurrence 因为 weight sharing 自带 regularization" 的直觉一致。其实跟 dropout / weight decay 一样,looped 是另一种 implicit regularizer。

### 7.6 训练收敛速度 (Figure 6)

ELT 在 diffusion framework 下收敛速度:
- 16N × 2L_max: 2× speedup vs N=32 DiT baseline
- 8N × 4L_max: 1.4× speedup

iso-inference-compute (effective depth D=32 一样),但训练时 ELT 收敛更快。可能的解释:
1. **weight sharing 的 implicit regularization** 减少过拟合,每个参数见更多 "effective gradient steps"
2. **共享参数在 L_max 次 loop 中被 update L_max 次**,相当于每次 step 给同一组参数更多 gradient signal
3. 类似 "data recycling" — 把 model capacity 在不同 loop 间复用,可能加速 representation 学习

这个 finding 跟我之前看过的某些 weight-tied LLM 训练结论一致: weight tying 在 training compute 上也是 efficient 的,只是大家通常只关注 inference。

---

## 8. 跟其他方法的对比

### 8.1 vs Deep Equilibrium Models (DEQ)

DEQ (Bai et al. 2019) 也用 weight-tied iteration,但把 output 定义为 fixed point $x^* = f_\Theta(x^*)$,用 black-box solver (Broyden / Anderson) 找 fixed point。

| 维度 | DEQ | ELT |
|------|-----|-----|
| Iteration count | Implicit (until convergence) | Explicit fixed L |
| Solver | Black-box root finding | Direct unrolling |
| Early exit | Hard (fixed point) | Native (any L) |
| Training | Implicit differentiation | Standard backprop through time |
| Robustness | Sensitive to solver init | More stable |

DEQ 的 elegance 在于"算到收敛",但实际上 inference budget 不固定。ELT 显式控制 L,适合"serving with SLO"。另外 DEQ 的 implicit differentiation 在 training 时 memory 省但 unstable,ELT 直接 BPTT 反而 simpler。

参考:
- DEQ: https://arxiv.org/abs/1909.01377
- DEQ for diffusion: https://arxiv.org/abs/2210.12867

### 8.2 vs Consistency Models

Consistency Models (Song et al. 2023) 是 **inter-step** acceleration: 把 N 步 diffusion 蒸馏到 1-2 步。ELT 是 **intra-step** acceleration: 在单个 sampling step 内减少 loop。两个 axis 正交,可以叠加。

ELT 论文还提到: "ELT is particularly compelling for one-step generative paradigms, where the loop count L becomes the sole lever for controlling the compute-quality trade-off at inference time." 这点很重要: 如果用 consistency model / drifting model 走到 one-step regime,那传统的"减 step 数"手段就失效了,只能靠 L 这种 intra-step lever。ELT 在 one-step world 里非常 strategic。

参考:
- Consistency Models: https://arxiv.org/abs/2303.01469
- Drifting models (Deng et al. 2026): https://arxiv.org/abs/2602.04770

### 8.3 vs Matryoshka / Matformer

Matryoshka Representation Learning (Kusupati et al. 2022): 学一个 embedding,前 k 维就是"低分辨率" embedding,可以从一个 model 切出任意粒度。Matformer (Devvrit et al. 2024): nested transformer,每层有"小 model 嵌在大 model 里"的结构,可以 elastic depth。

ELT 跟它们的哲学一脉相承: **训练一次,得到一个 elastic family**。Matryoshka 是 elastic width,Matformer 是 elastic depth via nested layers,ELT 是 elastic depth via loop count。

我直觉上觉得 ELT 的 form 最 elegant: 一个 model,一个 axis (L),连续可选。Matformer 的 nested 是离散的几个 sizes,Matryoshka 是连续但只对 embedding 层。ELT 把"weight-shared recurrence + 自蒸馏"结合起来,真正做到"inference 时再决定 depth"。

参考:
- Matryoshka: https://arxiv.org/abs/2205.13147
- Matformer: https://arxiv.org/abs/2310.07720

### 8.4 vs Early Exiting

传统 early exit (e.g., BranchyNet, Bert cascading): 在网络中间加 classifier,根据 confidence 决定是否提前 exit。问题是要训每个 exit 的 classifier,而且 confidence 不总是 reliable。

ELT 的"early exit"本质上是"early stop looping",但 **exit 的 head 是同一个 shared head**,而且 ILSD 已经在训练时强制让每个 L 的输出都"可用",不需要额外的 confidence 判断。

LoopViT (Shu et al. 2026) 在 visual reasoning 任务上结合了 looped transformer + parameter-free dynamic exit,跟 ELT 的"显式选 L"是不同的 exit mechanism。LoopViT 的 dynamic exit 不需要 user 指定 L,根据 uncertainty 自动停。

参考:
- LoopViT: https://arxiv.org/abs/2602.02156
- BranchyNet: https://arxiv.org/abs/1705.02431

### 8.5 vs E-DiT (Elastic Diffusion Transformer)

E-DiT (Wang et al. 2026) 做的是 "adaptive block skipping + MLP width reduction",通过 architecture-level skipping 实现 elastic inference,需要专门训练。

ELT 不依赖 skipping,直接通过 ILSD 正则化 looped process。E-DiT 的"跳过某层"是离散操作,ELT 的"减 L"是连续 (相对而言) 操作。两者其实可以叠加 — 在 ELT 内部再做 block skipping。

参考: E-DiT: https://arxiv.org/abs/2602.13993

---

## 9. 关键 Limitations

paper 在 Appendix B 里很诚实地列了:

1. **N 太小时崩坏**: 1N × 32L 在 DiT 上 FID 10.30,说明 recurrence 不能完全 substitute architectural diversity。Minimum N 大概在 4-8 (跟 model width 相关)。
2. **L 远超 L_max 时也崩**: 因为 shared block 在超出训练 regime 时会 over-iterate,输出 drift。不过 ILSD 在 UCF-101 上能 extrapolate 到 L=6 (训练 L_max=4,峰值 FVD=69.20 at L=6),说明 ILSD 的 regularization 给了一定的 extrapolation 能力。但这是个 brittle 的 extrapolation,paper 说"warrants further investigation"。
3. **Deployment 需要调 (N, L)**: 不同 hardware 上最优 operating point 不同,需要 per-tier calibration。

我额外想到的 limitation paper 没明说:
- **训练时 cost 不一定省**: 虽然 inference 省参数,training 时 forward + backward 的 L_max 次循环都要 unfold,activation memory 跟 D = N×L 成正比。Training memory 跟 vanilla transformer 一样大,只是 parameters 少。
- **Sequential dependency 强**: L 次循环必须串行 (下一次循环依赖上一次),并行化不如 vanilla deep stack。在单卡 throughput 上是劣势,但 paper 说 SRAM residency 弥补了这一点。
- **批次内不同 sample 用不同 L 困难**: 现在 inference 时 batch 内所有 sample 用同一 L,如果不同 sample 想用不同 L (类似 adaptive computation),需要 special handling。Mixture-of-Recursions (Bae et al. 2025) 在 LLM 里做了这个,但 ELT 没探索。

参考: Mixture-of-Recursions: https://arxiv.org/abs/2507.10524

---

## 10. 我的延伸联想 (跟你的兴趣点对应)

### 10.1 跟 latent reasoning / hidden thinking tokens 的关系

你近期肯定在 follow Geiping 的 "Scaling up test-time compute with latent reasoning" (https://arxiv.org/abs/2502.05171) — 同一时间段发的同一类工作。Geiping 在 LLM 上用 recurrent depth (looped transformer) 实现 "thinking in latent space",不用 explicit chain-of-thought tokens,而是让模型在 hidden state 里反复循环 refine。

ELT 是 visual generation 上的对应: 每个 sampling step 内的 L 次 loop 就是 "latent thinking steps",把 image 生成从 "single forward pass" 变成 "iterative latent refinement"。两者本质上是同一个 idea 在不同 modality 的 instantiate:
- LLM latent reasoning: hidden state recurrent → 输出 answer
- ELT visual generation: hidden state recurrent → 输出 image

未来如果这两个 paradigm 合流 (multimodal model 用 latent reasoning 同时做 text + visual),ELT 的 ILSD 训练方法可以直接迁移过来。

### 10.2 跟 multi-token prediction / speculative decoding 的关系

你最近推的 multi-token prediction (DeepMind 那篇, Meta 也有 follow-up) 是 **inter-token** parallelism: 一次预测多个 token。ELT 是 **intra-forward** parallelism (loop count 调)。两者其实可以叠加:
- 用 multi-token prediction 减 token-level 的 sequential dependency
- 用 ELT 减 layer-level 的 sequential dependency
- 双 axis 上同时加速

这俩 axis 在 LLM 上还没人系统组合过,值得探索。

### 10.3 Inference scaling laws

ELT 揭示了一个 deep question: 我们能不能为 generative model 建立系统的 **inference-time scaling laws**? 现在 LLM 上的 inference scaling 主要靠 chain-of-thought (token-level),vision 上靠 diffusion steps (step-level) 和 ELT loops (intra-step)。这些 axis 的最优 allocation 是什么? 

paper 给了一个 empirical fit: FID = 1922.5 · G^(-0.95) + 1.48。这跟 SNR 的 scaling laws 形式接近,但 exponent (-0.95) 比 training scaling 的 exponent (~-0.1 to -0.5) 大很多。说明 inference compute 的"边际效益衰减"在 visual generation 上比 training compute 慢 — 也就是说,多花 inference compute 是相对划算的。

如果这个 scaling law 在更大 model / 更高 resolution 上还成立,ELT 类方法的 strategic value 就更大了。

### 10.4 跟 "thinking" 视觉系统的类比

paper 一开始引用了 Kar & DiCarlo (Nature Neuroscience 2019) 和 Kietzmann et al. (PNAS 2019) 的 neuroscience 工作,说 biological visual system 也靠 recurrence。这其实是你和 DiCarlo 团队一直关心的方向: brain 用 feedforward + recurrent 混合,recurrence 用来 resolve hard recognition tasks。

ELT 的 implementation 在某种意义上是"机械版的 cortical recurrence": 同一组 weights 反复用,直到 representation converge 到 solution space。ILSD 让每一步 intermediate state 都有意义,这跟 brain 里 "early visual cortex 已经能给 rough 估计,后续 recurrence refine" 的现象 align。

参考:
- Kar et al. 2019: https://www.nature.com/articles/s41593-019-0392-5
- Kietzmann et al. 2019: https://www.pnas.org/doi/10.1073/pnas.1909063116

### 10.5 ILSD 跟你能想到的所有"自蒸馏"variants 的 family tree

ILSD 在自蒸馏的 family tree 里:

```
Self-Distillation
├── Online + Same Model (BYOL, DINO)
│   └── EMA teacher
├── Self-Training (no teacher, 用 model 自己的预测)
├── Progressive (e.g., deep supervision)
└── **Intra-Network Self-Distillation** (新家族)
    ├── Layer-wise (e.g., FitNets, MiniLM)
    └── **Intra-Loop (ILSD)** ← ELT
        └── Student 是 teacher trajectory 的 prefix
        └── 零额外 forward 成本
        └── Online (无 EMA)
```

ILSD 最 novel 的地方是 "prefix trajectory" 性质 — student 不是另一个 model,也不是同一 model 的不同 view,而是 **forward propagation 的中间态**。这个结构性约束让 distillation 几乎免费。

### 10.6 跟 test-time RL / verifier-based reasoning 的潜在结合

现在 LLM reasoning 大热的方向是 test-time RL (e.g., OpenAI o1 / DeepSeek R1),用 verifier 给 reasoning trace 打分。在 visual generation 上,类似 paradigm 是 "test-time search" (EvoSearch, He et al. 2025)。

ELT + verifier 想象: inference 时一个 sample 跑 L 次 loop,每步用 verifier 评估 hidden state 的"图像质量" (类似 VLM scoring),根据 verifier 信号决定继续 loop 还是 exit。这跟 LoopViT 的 uncertainty-based exit 类似,但用外部 verifier 替代 internal uncertainty。这个方向值得做。

参考: EvoSearch: https://arxiv.org/abs/2505.17618

---

## 11. 总结

这篇 paper 做的事情可以一句话概括: **把 weight-shared recurrence (Universal Transformer 老 idea) + intra-loop self distillation (新 training trick) 应用到 visual generation,得到一个 elastic model family**,test-time 可以根据算力预算动态调整 L,quality 平稳过渡。

真正 novel 的不是 looped transformer 本身 (这个 idea 有 7 年了),而是 **ILSD 这个 training trick 让 looped transformer 终于在 visual generation 上 work**,而且 work 得很好 (4× param reduction,3.5× throughput)。

ILSD 的核心 insight 我觉得是: **如果 forward propagation 是一个 trajectory,中间步骤也应该被监督到 "可用"状态**,而不是任由它们 uninterpretable。这跟 deep supervision 思想相通,但通过"trajectory prefix self-distillation"实现得异常优雅,几乎零额外开销。

从更宏观的角度看,ELT 是"inference-time scaling for generative models"这个新兴方向的一个 data point,跟 latent reasoning (Geiping)、test-time RL、consistency models、EvoSearch 等等一起,正在形成一个 **"training 一次,inference 时弹性扩展 compute"** 的新 paradigm。这跟以前 "training cost = inference cost" 的对称模型不同,把 compute 这个轴 decoupling 出来了。

对于你做的 NanoGPT / 教学场景,ILSD 其实也很容易拿来当教学 example: 几行代码加上 vanilla transformer 训练就能演示 self-distillation + elastic inference,比 BERT early exit 教材还直观。

一些可能的 follow-up 方向你可能会感兴趣:
1. **ILSD 在 LLM 上**: 不只在 visual,LLM 的 looped transformer 加 ILSD 能不能实现 "test-time elastic reasoning"?
2. **N=1 时为什么 fail**: 理论分析单 layer 重复的 expressivity limit,跟 Turing machine simulation 的关系
3. **Training memory 优化**: 用 gradient checkpointing 让 ELT 训练 memory 也按 N (而非 N×L) 缩放
4. **Cross-loop MoE**: 不同 loop 用不同 expert,介于 fully shared 和 fully unique 之间
5. **Verifier-guided dynamic L**: 用外部 verifier 在 inference 时动态决定 exit timing
6. **Inference scaling laws**: 系统化研究 K (steps) × L (loops) × N (layers) × d (width) 的 pareto frontier

paper link (推测, 准确 arxiv id 需要搜): https://arxiv.org/abs/2502.05171 (跟 Geiping 一同 batch)

如果哪部分你想我深挖 (比如 ILSD 跟 BYOL EMA teacher 的理论对比、为什么 N=1 fail、或者在 nanoGPT 上怎么 50 行实现 ILSD),尽管问。
