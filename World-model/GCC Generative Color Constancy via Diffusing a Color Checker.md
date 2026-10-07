---
source_pdf: GCC Generative Color Constancy via Diffusing a Color Checker.pdf
paper_sha256: a841f9caf2548f02f7ee37225293ae24edbf337113c10bd1b0b6144175f08f72
processed_at: '2026-08-19T08:48:47-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GCC 用人话版

好嘢 Karpathy，那我换个频道，完全口语地讲一遍。先把那句最核心的话说在前面——

**这篇 paper 在干一件事：让 Stable Diffusion 当"测光表"用。**

不是让它画图，是让它读色温。这个 repurposing 非常 6。

项目地址：https://chenwei891213.github.io/GCC/

---

## 1. 这问题到底是什么

人眼有个本事叫 color constancy——白纸在黄灯光下，你看它还是白的。相机不行，相机只会忠实地记录"黄光下的白纸 = 偏黄"。要让相机也"看起来正常"，得先估计当时那个光源是什么颜色（RGB triplet），然后除掉这个偏色。这个估计光源颜色的任务就叫 computational color constancy。

听起来简单，实际上有个巨坑：**cross-camera generalization**。

同一个场景，Canon 5D 拍出来光源 raw RGB 是 $(0.51, 0.52, 0.48)$ 这种值，Sony 拍出来可能是 $(0.42, 0.55, 0.50)$。不同 sensor 的光谱响应曲线（spectral sensitivity function）不同，导致同一个光在 raw 域的 RGB 值完全不一样。

所以你拿 NUS-8 数据集（8 个 camera）训练一个 CNN，去测 Gehler 数据集（2 个 camera）——performance 暴跌。因为学到的 mapping 是 camera-specific 的，换个 camera 就失效。

过去十年大家想了一堆办法：metric learning（IGTN）、quasi-unsupervised（Bianco）、contrastive learning（CLCC）……都是试图学 camera-invariant feature。有效但天花板有限。

---

## 2. 天降灵感：Diffusion Model 是个见过世面的家伙

LAION-5B 上预训练的 Stable Diffusion 看过 50 亿张图。它对"室内暖光长什么样"、"阴天偏蓝长什么样"、"夕阳下的红橙调子"、"霓虹灯的紫绿混合"——这种 **scene-level lighting prior** 是有 internalized knowledge 的。

而且这个知识是 camera-agnostic 的——SD 学的是图像层面的视觉规律，不管你 sensor 怎么响应，最后呈现在 sRGB 图像里的"暖光室内场景"它都认识。

那能不能直接问 SD："这张图光源是什么 RGB？" 

**不行**。直接 regression RGB 三元组太难监督，输出空间太自由，diffusion 训练不好收敛。

GCC 的核心 idea：**让它画一个 color checker 进去**。

color checker 就是那种摄影师用来校色的小卡片（Macbeth chart），有一排灰块。灰块的好处是——它本身应该反射所有波段的均匀光，所以灰块呈现的颜色 = 光源颜色。把灰块画对了，光源 RGB 直接读出来。

这个 idea 借鉴自 DiffusionLight（https://pakkapon.net/papers/DiffusionLight.html），人家是 inpaint chrome ball 做 HDR lighting estimation。GCC 把它搬到 color constancy 上，载体从 chrome ball 换成 color checker——因为 color checker 更结构化、更易读色、更适合 sRGB 域。

---

## 3. 怎么干：三个骚操作

### Trick 1：单步推理，把 diffusion 用成 CNN

传统 diffusion 推理要 25-50 步去噪，慢且随机。同一张图跑两次结果不一样，做 color constancy 这种 deterministic task 是灾难。

GCC 借鉴 Garcia et al. 的发现（https://arxiv.org/abs/2409.11355）——对于低频任务，fine-tune 之后 diffusion 一步就够。Lin et al. 还专门写过一篇文章（https://arxiv.org/abs/2401.02041）说社区主流的 noise schedule 实现其实是有 bug 的，误导大家以为 multi-step 是必须的。

具体公式长这样：

$$z_T = \sqrt{\bar{\alpha}_T} \cdot z_h + \sqrt{1 - \bar{\alpha}_T} \cdot \epsilon$$

变量含义：
- $z_T$: timestep $T$ 时的 noised latent
- $\bar{\alpha}_T = \prod_{i=1}^{T} \alpha_i$: 累积保留系数，控制原信号占多少
- $z_h$: 经过 Laplacian 分解后的高频 latent
- $\epsilon$: 噪声项——**直接设成 0**

然后去噪一步走完：

$$\hat{z}_0 = \sqrt{\bar{\alpha}_T} \cdot z_T - \sqrt{1 - \bar{\alpha}_T} \cdot \epsilon_\theta(z_{combined}, T, c)$$

变量含义：
- $\hat{z}_0$: 预测的 clean latent
- $\epsilon_\theta(\cdot)$: fine-tuned U-Net
- $z_{combined} = [z_T, M', z_{masked}]$: channel 维度拼接
- $M'$: 下采样 8 倍的 mask
- $c$: text embedding（这里其实没用真 prompt）

跑一次 0.18 秒，比传统 25 步 diffusion 快 100 倍。本质上 diffusion 退化成了一个 conditional image-to-image CNN。但保留了 pre-trained 的 prior。

### Trick 2：Laplacian 分解，逼模型看外面

这个 trick 极其关键。

设想一下——你往图里贴个 neutral color checker，然后让 U-Net inpaint 它。问题是：color checker 自己就有颜色（中性灰），latent encode 之后这些颜色信息就在 $z$ 里了。模型完全可以偷懒：直接 reconstruct input latent，把外面的颜色照样写进去，根本不去管场景光照。

GCC 用 Laplacian pyramid 把 input latent 分解，**只保留高频分量 $z_h$**，喂给 U-Net。

color checker 的"高频"是什么？是方块的边缘、网格的 layout、每个 patch 的形状。
color checker 的"低频"是什么？是每个 patch 的颜色——这正是我们想预测的东西！

把低频 channel 关掉后，模型输入里就只剩"checker 的几何结构"，颜色信息完全 blank。模型必须从 scene context 推断颜色，不能 cheating。

算法是经典 Laplacian pyramid：

```
for l = 0 to L-1:
    z_blur = Gaussian(z_curr)
    z_high = z_curr - z_blur     # 高频 = 原图 - 低通
    z_h += z_high
    z_curr = AvgPool(z_blur)     # 下采样进下一层
```

实验上 $L=2$ 最好。$L=1$ 留了太多低频，$L=3$ 把结构也削掉。

| Level | Mean | Median | Worst-25% |
|-------|------|--------|-----------|
| L=1 | 3.53 | 3.27 | 6.03 |
| **L=2** | **2.35** | **2.02** | **4.57** |
| L=3 | 3.16 | 2.83 | 5.62 |

这个 insight 我认为可以推广到很多任务——任何"结构由 input 决定、内容由 context 决定"的 generation 任务（harmonization、relighting、shadow removal）都可以用 frequency decomposition 切掉 input 的低频。

### Trick 3：Mask Color Jitter，处理标注烂的问题

NUS-8 和 Gehler 这俩数据集只给 color checker 的 bounding box，不给精确角点。所以没法用 homography 把 standard checker 模板贴上去——pixel-level misalignment 严重。

GCC 的方案看似反直觉：直接把 mask 区域的颜色 random jitter 一下，亮度随机 $[0.8, 2.0]$、对比度 $[0.8, 1.4]$、饱和度 $[0.8, 1.4]$。

公式：

$$I_{aug} = (1-M) \odot I + M \odot \tau(I)$$

变量：
- $I$: 输入图
- $M$: binary mask
- $\odot$: element-wise 乘
- $\tau(\cdot)$: color jitter 函数

**关键反直觉点**：训练时 input 是 jittered checker，但 ground truth 是**原始未 jitter** 的 checker。模型必须学会"无视 mask 区域的颜色，从外面推断"。

这就像 BERT 的 masked language modeling——把 token 替换成 [MASK]，逼模型从上下文猜。这里是把 mask 区域颜色打乱，逼模型从 scene context 猜光照。

配合 Laplacian decomposition 是必须的——光 jitter 不行，因为低频颜色还是会从 VAE encoder 漏出来。Laplacian 把这条路也堵死。

---

## 4. 训练损失非常朴素

就是 pixel-level MSE：

$$\mathcal{L} = \frac{1}{HW} \sum_{i,j} (I^*_{i,j} - \hat{I}_{i,j})^2$$

变量：
- $(i,j)$: pixel 坐标
- $I^*$: ground truth image（带原始 color checker）
- $\hat{I}$: 模型输出
- $H,W$: 高宽

**没有用 illuminant label 监督**！这是非常重要的一点。整个 pipeline 只需要图像里有 color checker 存在，不需要 ground truth 光源 RGB。极大降低数据标注需求。

Loss 是对整张图算的，不只是 mask 区域——保证背景不动，inference 时背景必须原样保留，不然后续 white balance 会被污染。

---

## 5. 推理流程一步一步

1. 拿一张输入图，gamma 校正到 sRGB（因为 VAE 在 sRGB 上训练过）
2. 在指定位置贴一个固定大小的 neutral color checker（grey = 128）
3. VAE encode 到 latent
4. Laplacian 分解取高频
5. 加上 noise schedule 系数，构造 $z_T$（$\epsilon = 0$）
6. 单步 U-Net forward → 得到 $\hat{z}_0$
7. VAE decode 回 image
8. 逆 gamma 校正回 linear RGB
9. 把生成的 color checker 映射到标准 grid，按 patch 取色
10. 从 achromatic patches 平均得到 illuminant RGB $\hat{y}$

完事。Angular error 评估：

$$\theta = \arccos\left(\frac{\hat{y} \cdot y}{|\hat{y}| |y|}\right)$$

变量：
- $\hat{y}$: 预测光源向量
- $y$: ground truth 光源向量
- $\theta$: angular error，单位度数，越小越好

---

## 6. 实验数据看哪些

### Cross-Camera 主战场（Table 1）

| 方向 | Metric | GCC | 最强 baseline |
|------|--------|-----|--------------|
| NUS→Gehler | Mean | **2.35** | 2.73 (C⁴-FC4) |
| NUS→Gehler | Worst-25% | **4.57** | 5.69 |
| Gehler→NUS | Mean | 2.38 | 2.28 (C⁴-FC4) |
| Gehler→NUS | Worst-25% | **4.58** | 4.60 |

**Worst-25% 提升明显**——这是 cross-camera 真正考验。NUS→Gehler 方向，GCC 把最差 25% 的 case 从 5.69 降到 4.57，相对改进 20%。说明 diffusion prior 在难场景上的鲁棒性远超纯 CNN。

但 Gehler→NUS 上 GCC 2.38 略输给 C⁴ 的 2.28。原因是 Gehler 只有 568 张图，fine-tune diffusion 容易过拟合，类似 DreamBooth 的小数据问题。作者在 Limitations 里也坦白说了。

### 计算效率（Table 7）

| Method | Steps | Time | Mean |
|--------|-------|------|------|
| SDXL+SDEdit+LoRA | 25 | 17.98s | 4.47 |
| **GCC** | 1 | **0.18s** | **2.35** |

100x 加速，性能反而更好。single-step 在低频任务上是正确选择。

### Spatially Varying Illumination（Table 6, LSMI 数据集）

这是隐藏武器。传统 color constancy 假设全局单一光源，GCC 天然能处理 multi-illuminant——把图分成 4×4 网格，每个 cell 贴一个 color checker，得到 16 个局部光源，然后 interpolate 成 illuminant map。

在 LSMI 上 zero-shot 评估（没专门 fine-tune 过 multi-light 数据），Galaxy Single 场景 mean **2.05**，比专门训练的 LSMI-H 的 2.85 强很多。说明 diffusion prior 对局部光照的理解也是 implicit 学到的。

### Ablation（Table 8）

最有趣的 ablation 是"直接 predict RGB vs inpaint color checker"：

| 方案 | Mean | Median | Worst-25% |
|------|------|--------|-----------|
| 直接 predict RGB (无 inpaint) | 2.98 | 2.53 | 6.14 |
| **Inpaint color checker (full)** | **2.35** | **2.02** | **4.57** |

直接 predict RGB 不差！2.98 也是 SOTA 级别。但 inpaint color checker 能再降 0.63 度。这 0.63 度的改进全部来自"structured output representation"——把自由度高的 regression 问题转成结构化的 image generation 问题，diffusion model 更在行。

---

## 7. 为什么这思路 work

我看完之后总结三点：

**第一，diffusion prior 是真 prior。** SD 在 LAION 上学的不是"Canon 5D 拍室内暖光 raw RGB 长什么样"，它学的是"sRGB 域里室内暖光场景的颜色统计长什么样"。后者天然 camera-agnostic。

**第二，structured output 比自由 regression 好做。** 让 diffusion 吐一个 RGB 三元组，输出空间自由度高，loss landscape 不好优化。让它 inpaint 一张图，输出空间是 image manifold，diffusion 在这个 manifold 上有强 prior，可以平滑地"slide"到正确答案。

**第三，frequency decomposition 是 control 信号。** 通过 Laplacian 切掉 input 的低频，等于告诉模型"结构是给你的，颜色你要从外面猜"。这种"给什么不给什么"的明确划分，让模型没办法偷懒。

---

## 8. 我脑子里冒出来的联想

### 8.1 跟 Marigold 是亲兄弟

Marigold（https://arxiv.org/abs/2312.02145）做 monocular depth estimation，思路几乎一模一样：fine-tune Stable Diffusion + single-step inference + synthetic data 训练。GCC 和 Marigold 共同验证一个 thesis——**pre-trained diffusion 是个通用的 inverse problem solver**，只要任务输出是 image-like 的。

### 8.2 "Generative model as measurement device" 这个 paradigm

diffusion 本来是生图用的，这里被改造成测量工具。这个 repurposing 思路可以推广到很多地方：

- **Material estimation**：inpaint 一个 BRDF chart
- **Camera response function**：inpaint 一组 grey ramp
- **HDR from LDR**：inpaint 一个 exposure bracket
- **Skin tone calibration**：inpaint skin tone chart（医疗摄影、电影级 color grading）
- **Atmospheric scattering estimation**：inpatch 一个 known distance target

凡是"用已知物理 reference 推断未知环境参数"的任务，都可以套这个 template。

### 8.3 DreamBooth-like 过拟合是真问题

作者自己承认小数据集 fine-tune 会有 DreamBooth 那种 catastrophic forgetting。Gehler→NUS 那一组实验 mean 2.38 略输 C⁴ 的 2.28 就是证据。

可能的解法：
- LoRA（https://arxiv.org/abs/2106.09685）只调低秩权重，缓解 forgetting
- Orthogonal fine-tuning（https://arxiv.org/abs/2306.07280）保持原 weight space 的正交性
- 或者干脆不要 fine-tune，想办法做 zero-shot——但 zero-shot 估计很难达到 SOTA

### 8.4 Text condition 是个没开发的 channel

U-Net 接 text embedding $c$，但 GCC 没用真 prompt，相当于 unconditional。

完全可以做 text-conditioned color constancy——给 prompt "warm indoor sunset" 引导模型倾向暖光估计。或者反过来，让模型先输出它的"光照语义描述"，再把这个描述 ground 成 RGB。这就把 color constancy 变成 vision-language 任务了，非常有意思的 follow-up 方向。

### 8.5 跟 CLCC 的哲学对比

CLCC（https://arxiv.org/abs/2104.07691）走 contrastive learning 路线，试图学一个 camera-invariant feature space。
GCC 走 generative prior 路线，试图学一个 camera-agnostic image manifold。

两个路线哲学不同：
- CLCC: 对齐不同 camera 的 feature
- GCC: 绕过 camera，直接用图像层面 prior

后者更 robust——因为它不需要"对齐"，直接跳出 sensor 问题。这也是为什么 GCC 在 worst-25% 上完胜——难场景难在 sensor 不匹配 + 光照异常，CLCC 只解决前者，GCC 同时解决两者。

### 8.6 VAE sRGB 假设的小 hack

pre-trained VAE 是 sRGB 域训练的，但 color constancy 数据是 linear RGB。GCC 通过 $\gamma = 1/2.2$ 桥接两个 domain。这是个工程 hack，理论上不"正确"——更严格的做法是 fine-tune VAE 或者训练 linear-aware encoder。但 hack 有效，工程实用主义胜利。说明 pre-trained VAE 的 representation 足够 robust，小幅 domain gap 不影响功能。

### 8.7 关于 multi-step diffusion 的反思

GCC 证明了 color constancy 这种低频任务，single-step 就够。这跟 Lin et al.（https://arxiv.org/abs/2401.02041）的发现吻合——主流 DDIM 实现有 bug，让大家误以为 multi-step 是必须的。

我觉得这是个被低估的发现。如果 single-step 在低频任务上够用，那 diffusion 在很多 inverse problem 上的计算成本可以从 100x 降到 1x。Marigold、DiffusionLight、GCC 都印证了这一点。未来我们应该会看到更多 single-step diffusion 的应用。

---

## 9. 总结一句

这篇 paper 的核心贡献是个 paradigm demonstration：**把 pre-trained diffusion model 当 inverse problem solver 用，输出形式是 structured image（color checker），通过 frequency decomposition 控制信息流，通过 single-step 实现高效推理**。

技术细节（Laplacian、mask jitter、deterministic inference）都是 supporting cast，真正的 star 是那个 core idea——让 generative model 当 measurement device。

公式 $\hat{z}_0 = \sqrt{\bar{\alpha}_T} z_T - \sqrt{1-\bar{\alpha}_T} \epsilon_\theta$ 看着平平无奇，实际上重新定义了 diffusion 的使用方式。这种"扭曲用法"会在 computational photography、medical imaging、scientific imaging 上有大量后续工作涌现。

我个人非常喜欢这篇 paper，因为它做了我一直想验证但没动手的事——**证明 generative model 的内部 representation 比专门训练的 discriminative model 更 transferable**。这是 deep learning 时代一个反复出现的主题，而 GCC 给了它在 low-level vision 领域的又一个证据。

---

# GCC: Generative Color Constancy via Diffusing a Color Checker 深度解析

Karpathy 老兄，这篇论文我读完之后相当兴奋，因为它的核心 idea 非常漂亮——用 pre-trained diffusion model 的 prior 来做 color constancy，本质上是在问一个极有想象力的问题：**能不能让 Stable Diffusion 凭它的"世界知识"直接告诉我们这个 scene 的光源 RGB 是什么？** 答案是可以，且做法出乎意料的优雅——不去直接 regress 一个 RGB 三元组，而是让模型 inpaint 一个 MacBeth color checker 进去，然后我们从灰块里读色温。这是一个 representation 的胜利，而不是 architecture 的胜利。

项目主页：https://chenwei891213.github.io/GCC/

---

## 1. Intuition：为什么用 Diffusion 来做 Color Constancy 是个好主意

Color constancy 的核心难题是 **cross-camera generalization**。不同 camera sensor 的 spectral sensitivity function 不一样，同一个场景，Canon 5D 拍出来 raw RGB 是 $(R_1, G_1, B_1)$，Sony 拍出来是 $(R_2, G_2, B_2)$，两个向量甚至不在同一个空间里。所以传统 CNN 学到的"raw RGB → illuminant RGB"映射，过 camera 就废。

但是！如果你去问一个在 LAION-5B 上训练过的 Stable Diffusion——它见过几十亿张 sRGB 图像，对"室内暖光长什么样"、"阴天冷光长什么样"、"霓虹灯场景的颜色分布如何"这种 **semantic + perceptual prior** 是有强记忆的。这种 prior 是 **camera-agnostic** 的，因为它学的是图像层面的统计规律，而不是 sensor 层面的物理映射。

GCC 的洞察在于：与其让 diffusion 直接吐一个 RGB 三元组（输出空间太自由，模型容易 collapse），不如给它一个结构化的"载体"——color checker。Color checker 的灰块在 D65 光下应该是中性灰，在暖光下应该偏红黄，在冷光下偏蓝。这就把"predict 一个 scalar triplet"问题变成了"predict 一张结构受约束的图，从图里读色温"问题，后者对 diffusion 来说几乎是 native 任务（inpainting）。

参考 DiffusionLight 的 chrome ball 思路：https://pakkapon.net/papers/DiffusionLight.html —— GCC 是它的 color constancy 版本。

---

## 2. Pipeline 全景图

我用文字画一下整个数据流，把它当 architecture diagram 来理解：

### 2.1 Training Pipeline (Figure 2)

```
Input Image I  ──┬──► VAE Encoder ──► z = E(I)
                 │                          │
Mask M (binary) │                          ▼
                 │                  Laplacian Decomposition
                 │                  (提取高频 z_h)
                 │                          │
                 ▼                          ▼
I_aug = (1-M)⊙I + M⊙τ(I)  ──► VAE Enc ──► z* = E(I_aug)
                 │                          
                 │    [对 masked 区域做 color jitter]
                 │                          │
                 ▼                          ▼
            z_masked = E(I⊙(1-M))      z_T = √(ᾱ_T)·z_h + √(1-ᾱ_T)·ε   (ε=0!)
                 │                          │
                 └──────────┬───────────────┘
                            ▼
            concat → z_combined = [z_T, M', z_masked] ∈ R^{h×w×(2d+1)}
                            │
                            ▼
                  SD Inpainting U-Net ε_θ
                  (input: z_combined, t=T, text c)
                            │
                            ▼
                   ẑ_0 = √(ᾱ_T)·z_T - √(1-ᾱ_T)·ε_θ(z_combined, T, c)
                            │
                            ▼
                   Î = D(ẑ_0)  (VAE decode)
                            │
                            ▼
                   L = ||I* - Î||²  (MSE on pixels)
```

### 2.2 Inference Pipeline (Figure 3)

```
Input Image I ──► gamma correction (γ=2.2) → sRGB domain
                          │
                          ▼
          Paste neutral color checker at mask region
                          │
                          ▼
                  VAE Encode → z*
                          │
                          ▼
              Laplacian Decomposition → z_h
                          │
              z_T = √(ᾱ_T)·z_h  (no noise term!)
                          │
                          ▼
          z_combined = [z_T, M', z_masked]
                          │
                          ▼
                Single-step U-Net forward (t=T)
                          │
                          ▼
                     ẑ_0 → VAE Decode → Î
                          │
                          ▼
              inverse gamma correction → linear RGB
                          │
                          ▼
        Map generated checker to standard grid → sample patches
                          │
                          ▼
            Extract achromatic patches → illuminant RGB ŷ
```

这里的关键 trick 在于训练和推理的 **不对称设计**：训练时输入带 jitter 的 $I_{aug}$，推理时输入贴了 neutral color checker 的图像。Laplacian decomposition 保证两边都只走 high-frequency 通道，所以 jitter 产生的 color 偏移不会泄漏到模型输入里——模型只看见 checker 的几何结构，颜色必须从 scene context 推断。

---

## 3. 关键技术细节深入

### 3.1 单步确定性推理：为什么不要传统 multi-step diffusion

传统 diffusion 推理是 $z_T \to z_{T-1} \to \dots \to z_0$，每步加噪去噪，stochastic 性很强。对 color constancy 这是灾难——同一张图跑两次得到不同 illuminant，无法做 evaluation。

GCC 借鉴了 Garcia et al. 的发现（https://arxiv.org/abs/2409.11355）：对于低频任务（如 depth、color constancy），fine-tune 后 single-step 就够了。原因在 DDIM scheduler 的 trailing 设计——但很多开源实现有 bug（参考 https://arxiv.org/abs/2401.02041 揭示的"flawed noise schedules"问题），导致大家误以为 multi-step 是必须的。

具体公式：

$$z_T = \sqrt{\bar{\alpha}_T} \cdot z_h + \sqrt{1 - \bar{\alpha}_T} \cdot \epsilon$$

- $z_T$: timestep T 时刻的 noisy latent
- $\bar{\alpha}_T = \prod_{i=1}^{T} \alpha_i$: 累积 noise schedule coefficient，控制 signal 保留比例
- $z_h$: 经 Laplacian decomposition 后的高频 latent
- $\epsilon$: 噪声项——**GCC 直接设为 0**

然后去噪一步到位：

$$\hat{z}_0 = \sqrt{\bar{\alpha}_T} \cdot z_T - \sqrt{1 - \bar{\alpha}_T} \cdot \epsilon_\theta(z_{combined}, T, c)$$

变量含义：
- $\hat{z}_0$: 预测的 clean latent
- $\epsilon_\theta(\cdot)$: U-Net 预测的噪声
- $z_{combined} = [z_T, M', z_{masked}]$: channel-wise 拼接，$M'$ 是降采样 8 倍的 mask
- $c$: text embedding（这里其实是个 dummy，没有真实 text prompt）

这个公式表面是 DDPM 标准的 reverse process 第一步，但因为 $\epsilon=0$ 且只走一步，整个 pipeline 退化成一个**带 noise schedule 加权的 conditional image-to-image network**。diffusion 退化成 deterministic CNN。

### 3.2 Laplacian Decomposition：高频保结构，低频让位给光照

这是论文最巧的 design 之一。直觉上是这样的——color checker 由两部分组成：

- **高频**：方块的边界、网格 layout、每个 patch 的形状
- **低频**：每个 patch 的颜色（这正是我们想要的输出！）

如果直接把 neutral color checker 的 latent 喂给 U-Net，模型会偷懒——直接 reconstruct 输入，因为 input 已经有 patch 的颜色信息了，何必去问 scene context？

Laplacian decomposition 的算法（Algorithm 1）：

```
Input: z ∈ R^{B×C×H×W}, pyramid levels L
Output: z_h (high-frequency only)

for each channel c:
    z_curr ← z[c]
    for l = 0 to L-1:
        z_blur ← Gaussian(z_curr)   # 3×3 kernel
        z_high ← z_curr - z_blur    # high-freq = orig - lowpass
        if l == 0:
            z_h[c] ← z_high
        else:
            z_h[c] += Upsample(z_high)
        z_curr ← AvgPool(z_blur)    # downsample for next level
```

实验上 $L=2$ 最优（Table 9）：

| Level | Mean | Median | Best-25% | Worst-25% |
|--------|------|--------|-----------|------------|
| L=1 | 3.53 | 3.27 | 1.48 | 6.03 |
| **L=2** | **2.35** | **2.02** | **0.78** | **4.57** |
| L=3 | 3.16 | 2.83 | 1.25 | 5.62 |

L=1 保留太多低频（因为只做了一层 high-pass），L=3 又过度激进把 checker 结构也削掉了。L=2 是 sweet spot。

**这个 insight 可以推广**：任何"结构由 input 决定，内容由 context 决定"的 generation 任务，都可以用 frequency decomposition 把 input 的低频 channel 关掉。比如 image harmonization、shadow removal、relighting——这些任务的本质都是"保持几何，改写外观"。

### 3.3 Mask Color Jittering：解决标注噪声的骚操作

这一段非常聪明。问题是 NUS-8 和 Gehler 数据集只给 color checker 的 bounding box，不给精确 corner。三种方案对比（Figure 4）：

**(a) 直接 inpainting** → 模型生成的 checker 轮廓乱七八糟，因为没告诉它 checker 应该长什么样
**(b) Homography overlay** → 即使做了 perspective transform，bounding box 不准导致 pixel-level misalignment
**(c) Mask color jittering** → 把 checker 区域的颜色随机扰动，强迫模型从外面重建

公式：

$$I_{aug} = (1-M) \odot I + M \odot \tau(I)$$

- $I$: input image
- $M$: binary mask
- $\odot$: element-wise multiplication
- $\tau(\cdot)$: color jittering function，随机调 brightness $[0.8, 2.0]$、contrast $[0.8, 1.4]$、saturation $[0.8, 1.4]$

**这里有个反直觉的设计**：训练时输入是 jittered 的 color checker，但 ground truth 是原始未 jitter 的 color checker。模型必须学会 "ignore the mask region's color, infer from context"。这就像做 masked language modeling——把 token 替换成 [MASK]，让模型从上下文预测原 token。

而且这个 jitter 配合 Laplacian decomposition 是必须的——单 jitter 还不够，因为低频 color 信息会从 VAE encoder 漏出来。Laplacian 把这一层堵死后，模型完全无法从 input latent 推断 mask 区域的原始颜色，必须真的去理解场景。

---

## 4. 损失函数与训练细节

Loss 非常简单，pixel-level MSE：

$$\mathcal{L} = \frac{1}{HW} \sum_{i,j} (I^*_{i,j} - \hat{I}_{i,j})^2$$

- $(i, j)$: pixel 坐标
- $I^*$: ground truth image（带原始 color checker）
- $\hat{I}$: model output
- $H, W$: image 高宽

注意这里 loss 是对**整张图**算的，不是只对 mask 区域。这样背景区域也参与监督，U-Net 不会乱改非 mask 区域（这点很重要，因为 inference 时背景必须保持原样，否则会污染后续 white balance）。

训练配置：
- Base model: stable-diffusion-2-inpainting
- Optimizer: Adam, lr = 5e-5, exponential decay after 150 warmup steps
- Iterations: 20k
- Image size: 512×512
- Batch size: 8（NUS→Gehler 用 gradient accumulation 到 16）
- Hardware: 单卡 NVIDIA RTX 4090 / A6000

**没有用 illuminant label 监督**，纯 pixel-level reconstruction loss。这一点很关键——这意味着这个方法不需要 ground truth illuminant 颜色，只需要图像里有 color checker 存在。这极大降低了数据需求。

---

## 5. 实验结果深入解读

### 5.1 Camera-Agnostic Evaluation (Table 1) — 最关键的 cross-camera 实验

| 方向 | Metric | GCC | 最强 baseline (C⁴squeeze-FC4) | 次强 (C⁵) |
|------|--------|-----|------------------------------|-----------|
| NUS→Gehler | Mean | **2.35** | 2.73 | 3.34 |
| NUS→Gehler | Worst-25% | **4.57** | 5.69 | 7.39 |
| Gehler→NUS | Mean | **2.38** | 2.28 | 2.65 |
| Gehler→NUS | Worst-25% | **4.58** | 4.60 | 5.72 |

重点看 **Worst-25%**——这是 cross-camera generalization 的真实考验。GCC 在 NUS→Gehler 上把 worst-case 从 5.69 降到 4.57，相对改进 20%。这说明 diffusion prior 在"难场景"上的鲁棒性远超纯 CNN 方法。

但是看 Gehler→NUS 的 mean，GCC 2.38 略输给 C⁴ 的 2.28。这里的小差距可能因为 Gehler 只有 568 张图，fine-tune diffusion model 数据量不足——作者在 Limitations 里也提到了 DreamBooth-like 过拟合问题。

### 5.2 Leave-One-Out (Tables 2, 3)

NUS-8 上 mean 2.03，**SOTA**（之前最好的 SIIE 是 2.05）。Gehler 上 mean 2.80，输给 C⁵ 的 2.50——这再次印证了"小数据集 fine-tune diffusion 容易过拟合"的判断。

### 5.3 Spatially Varying Illumination (Table 6, LSMI Dataset)

这是 GCC 的隐藏武器。传统 color constancy 假设全局单一光源，GCC 可以天然处理 multi-illuminant——把图像分成 4×4 grid，每个 cell inpaint 一个 color checker，得到 16 个局部 illuminant，然后 interpolate。

在 LSMI 上 zero-shot 评估（没专门 fine-tune multi-light 数据），Galaxy Single 场景 mean **2.05** vs LSMI-H 的 2.85——大幅领先！这个结果说明 diffusion prior 对"局部光照"的理解也是 implicit 学到的。

### 5.4 Computational Efficiency (Table 7)

| Method | Steps | Ensemble | Time | Mean |
|--------|-------|----------|------|------|
| SDXL+SDEdit+LoRA | 25 | 10 | 17.98s | 4.47 |
| **GCC (full)** | 1 | 1 | **0.18s** | **2.35** |

100x 加速 + 性能更好。这是 single-step deterministic inference 的胜利，也证明 multi-step 在 color constancy 这种低频任务上是 overkill。

### 5.5 Ablation (Table 8)

| Config | Mean | Median | Worst-25% |
|--------|------|--------|------------|
| No Laplacian, w/ inpaint, w/ mask DA | 3.71 | 2.86 | 7.68 |
| w/ Laplacian, w/ inpaint, no mask DA | 3.52 | 2.76 | 6.78 |
| w/ Laplacian, no inpaint (直接 predict RGB), no mask DA | 2.98 | 2.53 | 6.14 |
| **Full (all on)** | **2.35** | **2.02** | **4.57** |

有意思的是"直接 predict RGB"那一行 mean 是 2.98——并不差！这说明 diffusion prior 本身就够强。但 inpaint color checker 加上后能到 2.35，**0.63 度的改进**全部来自"structured output via inpainting"——这印证了我开头说的 representation 胜利。

---

## 6. 推广与联想

### 6.1 这个 paradigm 可以套到哪些任务

GCC 的本质是 "用 diffusion model 的 prior 求解一个 inverse problem，输出形式是 structured image 而非 scalar"。可以推广到：

- **Illumination estimation for 3D rendering**：inpaint 一个 chrome ball（DiffusionLight 已经做了）
- **Material estimation**：inpaint 一个 BRDF chart
- **Camera response function estimation**：inpatch 一组 grey ramp
- **HDR estimation from LDR**：inpaint 一个 exposure bracket
- **Skin tone calibration**：inpatch 一个 skin tone chart（医疗摄影、电影级 color grading）

### 6.2 跟 Marigold 的对照

Marigold（https://arxiv.org/abs/2312.02145）做 monocular depth，也是 fine-tune Stable Diffusion + single-step inference + synthetic data 训练。GCC 和 Marigold 几乎是 sibling work——都验证了一个 thesis：**pre-trained diffusion model 是个通用的 inverse problem solver，只要任务输出是 image-like 的**。

### 6.3 与 DreamBooth 的相似性

作者明确提到数据集太小会有 DreamBooth-like 过拟合（https://arxiv.org/abs/2208.12242）。这是 personalization 类方法的通病——prior 太强时，少样本会引发 catastrophic forgetting 或 concept distortion。GCC 通过 mask jittering 缓解，但根本问题没解决。可能 LoRA + orthogonal fine-tuning（https://arxiv.org/abs/2306.07280）会更好。

### 6.4 与 CLCC 的对照

CLCC（https://arxiv.org/abs/2104.07691）用 contrastive learning 提升 cross-camera 特征。GCC 用 generative prior 替代了 discriminative learning—— philosophically 是两种不同的泛化路径：
- CLCC: 学一个 camera-invariant feature space
- GCC: 学一个 camera-agnostic image manifold

后者明显更 robust，因为它不需要"对齐"不同 camera 的特征——直接绕过了 sensor 的问题。

### 6.5 VAE 的 sRGB 假设

一个值得关注的细节：pre-trained VAE 是在 sRGB 图上训练的，但 color constancy 数据是 linear RGB。GCC 通过 gamma correction $\gamma=1/2.2$ 桥接两个 domain。这其实是个 hack——更"正确"的做法是 fine-tune VAE 或者用一个 linear-aware encoder。但 hack 有效就行，工程上的实用主义。

### 6.6 关于 text condition c

U-Net 接收 text embedding $c$，但 GCC 没用真实 prompt。这意味着 text channel 实际上是个 unconditional signal。可以想象——如果给个 prompt "warm indoor light" 或 "cool outdoor shadow"，能不能做 controllable color constancy？这会是个有趣的 follow-up，把 color constancy 变成 text-conditioned 任务。

---

## 7. 失败模式与局限性（Figure 9）

作者很诚实地报告：当 inpaint 进去的 color checker 的"假设光照"和场景 ambient light 严重 mismatch 时，模型会试图调和两者，产生错误的颜色。这暴露了 diffusion model 的本质局限——它优化的是"visual plausibility"，不是"physical accuracy"。

这个 limitation 在 multi-illuminant 强对比场景（比如一边日光一边霓虹灯）最明显。GCC 的解决方案是 spatial grid sampling，但这只是 workaround，没有解决 fundamental 问题。

---

## 8. 我的 Take

这篇论文的核心 contribution 不是某个具体技术，而是一个 **paradigm demonstration**：diffusion model 是个极其强大的 prior engine，对 low-level vision 任务的 generalization 有奇效。比起 CNN 在 specific dataset 上 over-fit，diffusion 在 LAION 上学到的"what does natural image look like under various lighting"是个 far more transferable 的知识。

公式 $\hat{z}_0 = \sqrt{\bar{\alpha}_T} z_T - \sqrt{1-\bar{\alpha}_T} \epsilon_\theta$ 看起来轻描淡写，但它实际上重新定义了 diffusion 的使用方式——从 **generative sampler** 退化成 **deterministic conditional regressor**。这种"扭曲"用法会在很多 inverse problem 上反复出现。

我个人觉得这篇工作的真正价值在于它**示范了如何让 diffusion 说它不该说的话**——diffusion 本来是生图的，这里被改造成测量工具。这种"generative model as measurement device"的思路会在 computational photography、medical imaging、scientific imaging 上有大量后续工作。

---

## 参考链接

- 项目主页: https://chenwei891213.github.io/GCC/
- Stable Diffusion (Rombach et al.): https://arxiv.org/abs/2112.10752
- DiffusionLight (chrome ball inpainting): https://pakkapon.net/papers/DiffusionLight.html
- Garcia et al. (fine-tuning is easier than you think): https://arxiv.org/abs/2409.11355
- Marigold (depth estimation via diffusion): https://arxiv.org/abs/2312.02145
- DreamBooth: https://arxiv.org/abs/2208.12242
- LoRA: https://arxiv.org/abs/2106.09685
- CLCC (contrastive learning for CC): https://arxiv.org/abs/2104.07691
- NUS-8 dataset: https://www.comp.nus.edu.sg/~whitebal/illuminant/
- Gehler dataset: https://www5.informatik.uni-erlangen.de/research/data/
- LSMI dataset: https://github.com/sMBRTN/LSMI
- C4 / C5 (cross-camera CC): https://arxiv.org/abs/2104.07691
- Flawed noise schedules (Lin et al.): https://arxiv.org/abs/2401.02041
- Orthogonal fine-tuning: https://arxiv.org/abs/2306.07280
- FC4 (Hu et al.): https://arxiv.org/abs/1707.06284
- SDEdit: https://arxiv.org/abs/2108.01073
