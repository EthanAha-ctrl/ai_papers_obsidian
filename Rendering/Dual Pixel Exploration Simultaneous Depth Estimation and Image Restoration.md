---
source_pdf: Dual Pixel Exploration Simultaneous Depth Estimation and Image Restoration.pdf
paper_sha256: d38dc9a1c523b39a5181572f6258ebe3f5ea84e539e57c5f23cdc19c1cf8b311
processed_at: '2026-08-04T00:23:28-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Dual-Pixel 这篇paper到底干了啥

## 一句话总结

手机/单反里的 Dual-Pixel 传感器拍一张照片会顺带输出两张"微微错位"的图，大家以前要么拿它做立体匹配求深度（被模糊搞死），要么直接去模糊（浪费了模糊里藏的深度信息）。这篇paper说：**模糊和错位本质上是同一个量**，干脆一起解，还顺手发明了一个"把答案糊回去跟原图对"的loss，连ground truth都不用就能继续train。

paper: https://arxiv.org/abs/2012.08002
code: https://github.com/pan-liny/Dual-Pixel-Exploration

---

## Dual-Pixel 到底是什么（从物理讲起）

普通相机传感器每个像素就像一个"测光桶"，镜头把光聚焦进来，桶接到多少光就记多少。

**Dual-Pixel 的脑洞**：把这个桶劈成左右两半。左半桶只接经过镜头左半边的光，右半桶只接经过镜头右半边的光。一次曝光，同时得到两张图 $\mathbf{I}_L$ 和 $\mathbf{I}_R$。

因为镜头左半边和右半边在光瞳上有横向偏移，这两张图其实就是一个**baseline极短的立体对**（stereo pair）。Canon 5D IV、Google Pixel、Samsung Galaxy 这些都用它做自动对焦——只要看左右两张图错位多少，就知道有没有对上焦。

**关键现象**：
- 在焦平面上的物体：左右两张图里都清晰，而且完全对齐（disparity = 0）
- 偏离焦平面的物体：左右两张图里都模糊，同时还错位

blur 和 disparity **同时**出现，paper的核心insight就一句话：**它俩本来就是同一个物理量**，由"偏离焦平面程度"这一个变量决定，只差一个常数比例。

---

## 数学上为什么是一回事（核心intuition）

用薄透镜几何推一下。设：
- $f$ = focal length（镜头本身的焦距，固定属性）
- $F$ = sensor到lens的距离（可以调，调这个就是对焦）
- $d$ = 物体depth（离镜头多远）
- $d'$ = virtual image 距离（由 $d$ 决定，满足 $\frac{1}{d}+\frac{1}{d'}=\frac{1}{f}$）

物体经过aperture上一个点 $\mathbf{C}$ 在sensor上的落点（论文 Eq.5）：

$$T(\mathbf{X}, \mathbf{C}) = \underbrace{\frac{d'-F}{d'}\mathbf{C}}_{\text{blur/disparity的来源}} + \underbrace{F(\frac{y}{f}, \frac{z}{f})}_{\text{in-focus时的base位置}}$$

看这一项 $\frac{d'-F}{d'}\mathbf{C}$：
- 当 $d' = F$（点恰好成像在sensor上，即in focus），这一项消失 → 零disparity，零blur
- 当 $d' \neq F$，这一项随aperture位置 $\mathbf{C}$ 变化：
  - 对aperture左边和右边两个**端点** $\mathbf{C}_L, \mathbf{C}_R$ 取差 → **disparity**，正比于 $\frac{d'-F}{d'} \cdot (\text{左右瞳距离})$
  - 对aperture里**所有点**积分 → **blur disk**，半径正比于 $\frac{d'-F}{d'} \cdot (\text{aperture直径})$

disparity 和 blur radius 共享 $\frac{d'-F}{d'}$ 这一项，只差一个跟光瞳几何有关的常数。**所以测了blur就等于测了disparity**，反过来也一样。

这就是为什么paper能搞joint model——你不需要把blur当噪声去掉，blur本身就是depth的observation。

---

## 网络怎么搭的

```
B_{L,R} ──► DepthNet ──► 粗略 inverse depth D̂_c
                                   │
                                   ▼
B_{L,R} + D̂_c ──► DeblurNet ──► { Î, D̂ }  (清晰图 + 精细depth)
                                   │
                                   ▼
Î + D̂ ──► DP Simulator (可微) ──► B̂_{L,R}  (重新糊回去)
                                   │
                                   ▼
                          Reblur Loss: ‖B - B̂‖
```

直觉上：
- **DepthNet** 先粗估深度，输出 inverse depth（用 $1/d$ 是因为近处depth变化大，远处变化小，inverse更数值友好）
- **DeblurNet** 把粗depth和模糊DP pair一起塞进去，同时输出清晰图和refined depth
- **DP Simulator** 是个可微的"糊回去"函数：给定清晰图和深度，按forward model重新合成DP pair
- **Reblur Loss** 比较合成的DP pair和真实DP pair，强迫"答案"落在一个物理自洽的manifold上

这跟可微渲染、NeRF的思路完全一样：**inverse problem ill-posed？那就把forward model写成可微的，让它当regularizer**。

---

## Reblur Loss 为什么重要

这是整篇paper最聪明的一招。考虑两种场景：

**场景A：有ground truth时**
- 普通：$\mathcal{L}_{\text{res}}$ 让 $\hat{\mathbf{I}} \to \mathbf{I}$，$\mathcal{L}_d$ 让 $\hat{\mathbf{D}} \to \mathbf{D}$
- 加上reblur loss：额外要求"用 $\hat{\mathbf{I}}, \hat{\mathbf{D}}$ 糊回去要等于真实 $\mathbf{B}$"
- 直觉上：GT告诉网络"答案长这样"，reblur loss告诉网络"你的答案跟物理要一致"。两者协同，相当于一个data-driven + physics-driven的混合监督

**场景B：没有ground truth时（self-supervised fine-tune）**
- 完全没有 $\mathbf{I}$ 和 $\mathbf{D}$ 的GT
- 只有真实DP pair $\mathbf{B}$（手机拍的）
- 你就可以拿在合成数据上预训练的model，用reblur loss在真实数据上继续fine-tune
- paper里就这么干的：合成数据训练 → DPD-disp测试集上用reblur loss self-supervised fine-tune，性能从AI(1)=0.0906 → 0.0609，逼近专门为这个数据集设计的DPdisp方法的0.0481

**为什么这个work**：reblur loss本质上是一种**物理cycle consistency**。它的"锚"不是identity mapping（像CycleGAN那样），而是物理forward model。这种锚比identity更informative，因为forward model里编码了"depth → blur"的因果关系，反向约束depth就必须合理。

这跟 Image-to-Image translation 里的cycle loss、跟 SSL 里的mask-and-reconstruct、跟 NeRF 里"render然后compare"是一脉相承的研究范式。

---

## DP Simulator 为什么必要

DP领域的一个老大难问题：**没数据**。
- Canon 5D IV 有DP输出，但贵，且没法同时拿到ground truth depth
- Google Pixel 有DP，但光圈固定不能调，且同样没depth GT
- 之前所有DP learning方法都被这个数据瓶颈卡住

paper的做法：**写一个可微的DP图像合成器**。给定任意RGB-D数据（NYU Depth v2就有），按forward model把每个pixel的intensity spread到对应的blur disk里。

工程上的trick：naive实现要4层循环（每个像素、每个blur disk里的每个像素），太慢。作者用**integral image**（也叫summed area table，Crow 1984[https://en.wikipedia.org/wiki/Summed-area_table]那个老graphics trick）：
- 把blur disk近似成矩形（4个corner坐标由公式算）
- 在4个corner上写入差分mask
- 最后做一次integral image pass

复杂度从 $O(n \cdot |\mathbf{R}|)$ 降到 $O(n)$，跟blur disk大小无关。这个simulator让DP领域第一次有了大规模"合成+GT"训练数据，是这个工作的重要side contribution。

---

## 关键实验数字

**Deblurring (Table 1, DPD-blur dataset)**：

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| DPDNet (前SOTA) | 25.53 | 0.826 |
| Ours without reblur | 26.15 | 0.827 |
| Ours with reblur | **26.76** | **0.842** |

reblur loss带来约0.6dB提升，相对MSE提升~10%。在更难的Our-real数据集上差距更大。

**Depth (Table 2, DPD-disp dataset)**：

这个数据集只有test set，所以作者展示了两种能力：
- 直接用合成数据训练好的model测试（zero-shot transfer）：AI(1) = 0.0906
- 用reblur loss在test set上self-supervised fine-tune（**不用GT depth**）：AI(1) = 0.0609，逼近专门设计的DPdisp方法的0.0481

这个fine-tune实验是reblur loss最有说服力的证据。

**Ablation (Table 3)**：

| 阶段 | abs_rel↓ | rmse↓ |
|------|----------|-------|
| DepthNet only | 0.149 | 1.222 |
| + DeblurNet (joint) | 0.091 | 0.599 |
| + reblur loss | 0.083 | 0.461 |

每一阶段都在改进，说明joint optimization和物理约束各自贡献独立。

**Sim-to-real transfer**：
- 合成数据训好的model直接test on真实DPD-blur：20.28 dB
- 用真实数据fine-tune后：26.92 dB，**比从头在DPD-blur上训（26.76 dB）还高**
- 用一半真实数据fine-tune：26.52 dB

说明simulator合成数据有真实信息含量，可以做pretrain。

---

## 这套思路在更大图景里

**Forward model约束inverse problem的范式**：
- NeRF：volume rendering forward → 约束radiance field
- Differentiable rendering：约束3D shape/material
- Phase retrieval：Fourier transform forward → 约束object
- 本文：DP imaging forward → 约束

每个case都是把inverse problem的解空间用物理约束收紧。

**跟其他领域的直觉联系**：
- 跟plénoptic camera (Lytro) 的关系：plénoptic牺牲spatial resolution换angular resolution，DP不牺牲spatial但angular只采两个view。本质DP是"2-view light field"的极端版本。可以扩展到4-view (quad-pixel, Sony已有), 6-view, etc.
- 跟event camera的联系：event sensor的blur model跟defocus blur在频域有结构相似性，DP的left/right差分类似某种"spatial contrast"，跟event的"temporal contrast"对偶
- 跟generative model的联系：可以把forward DP model当condition放进diffusion model里做conditional deblurring。reblur loss会变成物理informative的score matching约束
- 跟大模型时代：DDDNet很小（Titan XP能训），如果用DINOv2/MAE这种预训练vision encoder当backbone，DP pair作为input做representation learning，reblur loss作为pretrain task，应该能更强

---

## 局限 & 可以再挖的坑

1. **Aperture shape近似**：simulator把半圆光瞳近似成矩形，blur disk形状会失真。真实blur kernel是半椭圆，边缘会跟矩形对不上。
2. **Thin lens假设**：忽略了aberration、衍射、色差。真实PSF有ring结构，可以学一个residual PSF项。
3. **Textureless region**：depth-from-defocus和stereo都怕无纹理区域，paper没专门处理。可以引入edge-aware smoothness prior（像Monodepth2那种）。
4. **Reblur loss的歧义性**：reblur loss只约束到"能糊出真实B"的manifold，但这个manifold有多解——不同 $(\mathbf{I}, \mathbf{D})$ pair可能糊出同一 $\mathbf{B}$。所以paper仍需GT supervision做主信号，reblur是regularizer。要完全self-supervised还需要额外prior。
5. **Scale ambiguity**：$\frac{d'-F}{d'} \cdot \text{aperture size}$ 这个乘积有尺度耦合，所以DP-based depth天生affine invariant（Garg 2019也指出过），必须靠scene prior或supervision定scale。
6. **Macro区域**：$d < f$ 时virtual image在lens左边（$d'<0$），blur disk翻转。simulator公式处理得了，但网络学这个case可能困难，real macro DP数据稀缺。
7. **可以扩到更多view**：quad-pixel (Sony)、6-pixel等更multi-view的硬件能更好解aberration和depth ambiguity。
8. **可以做成完全self-supervised**：加上natural image prior（比如pretrained diffusion model的score）和depth smoothness prior，理论上可以扔掉GT supervision，完全靠real DP pair + 物理一致性训练。

---

## 总结成几个intuition

1. **DP sensor的blur和disparity是同一物理量**的两个observation，jointly model比单独处理更优，因为它们互相提供redundant information
2. **Forward model可微化**让物理约束以loss形式融入网络，这种"物理cycle consistency"比identity-based cycle更informative
3. **Depth和deblur互相帮助**形成正向循环：deblur提供clean input给depth estimation，depth提供blur kernel形状给deblur
4. **Simulator**让DP领域突破数据瓶颈，从任意RGB-D数据合成训练样本，开启sim-to-real transfer和self-supervised adaptation
5. 这套范式（forward model + inverse problem + 可微physics loss）跟NeRF、可微渲染、inverse graphics是同一研究范式，paper把这套范式干净地apply到了computational photography的一个具体问题

对Karpathy你来说最可能有意思的点：
- Forward model当可微layer做inverse problem的clean formulation
- Self-supervised adaptation via physical consistency loss（reblur loss等价于一种physics-grounded cycle consistency）
- 极小baseline stereo与传统depth-from-defocus的统一
- 这种"小模型+物理prior"的研究范式 vs "大模型end-to-end"的范式对比，在data-scarce domain里前者往往更efficient

参考链接：
- Paper: https://arxiv.org/abs/2012.08002
- Code: https://github.com/pan-liny/Dual-Pixel-Exploration
- DPD dataset: https://www.eecs.yorku.ca/~abuolaim/dpd_dataset/
- DPdisp: https://www.eecs.yorku.ca/~abuolaim/dpd_disp/
- Du²Net: https://arxiv.org/abs/2003.14299
- Summed area tables: https://en.wikipedia.org/wiki/Summed-area_table
- Monodepth2: https://github.com/godardfar/monodepth2
- BTS: https://github.com/cogaplex-bts/bts
- NYU Depth v2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

---

# Dual Pixel Exploration: Simultaneous Depth Estimation and Image Restoration 深度解析

这篇paper来自ANU的Liyuan Pan等人，发表于ECCV 2020。核心思想非常优雅：把Dual-Pixel (DP) sensor的**disparity cue**和**defocus blur cue**当作同一个物理过程的两个表现，jointly做depth estimation和image deblurring，同时用forward model构造reblur loss做regularization。

论文链接：https://arxiv.org/abs/2012.08002
项目代码：https://github.com/pan-liny/Dual-Pixel-Exploration
DPD dataset（baseline用的）: https://www.eecs.yorku.ca/~abuolaim/dpd_dataset/

---

## 1. 背景：Dual-Pixel Sensor 是什么

DP sensor把每个pixel的光敏面劈成左右两半（physically split photodiode），所以单次曝光能同时输出两张图：
- $\mathbf{I}_L$: 只接收经过aperture左半部分 $\mathbf{A}_L$ 的光线
- $\mathbf{I}_R$: 只接收经过aperture右半部分 $\mathbf{A}_R$ 的光线

由于 $\mathbf{A}_L$ 与 $\mathbf{A}_R$ 在光瞳处有横向offset，两张图构成一个**baseline极短**的stereo pair（sub-aperture stereo）。Canon 5D IV、Google Pixel、Samsung Galaxy等都用DP做autofocus。

**两个关键观察**：
1. 在focal plane上的点，在 $\mathbf{I}_L$ 与 $\mathbf{I}_R$ 中**既sharp也对齐**（disparity=0）。
2. 偏离focal plane的点会**同时**产生defocus blur和disparity。
   
这两件事是同一回事——blur的尺度与disparity的幅度都由"depth偏离focal plane的程度"决定。前人要么纯当stereo做matching（被blur搞砸），要么直接把blur当噪声deblur掉（浪费了blur里的depth信息）。这篇paper的核心 motivation 是**这两个cue应该jointly model**。

---

## 2. DP Image Formation 数学模型（Section 3 的精华）

### 2.1 薄透镜 + 虚拟世界 (virtual world) 几何

坐标系约定：
- Lens 平面：$X=0$，lens center在origin。
- World 在lens左侧：$X < -f$，点 $\mathbf{X}=(X,Y,Z)$。
- Virtual world $\mathbf{W}'$ 在lens右侧：$X' > f$，点 $\mathbf{X}'=(X',Y',Z')$。
- Sensor (focal plane) 在 $X=F$，**注意 $F$ 是sensor到lens的距离，跟focal length $f$ 不是一回事**。
- $f$ = focal length；$F$ = lens-sensor distance；$d$ = object distance (depth)；$d'$ = virtual image距离。

**关键observation**（论文里有formal proof）：

> 从 $\mathbf{A}_L$（或 $\mathbf{A}_R$）形成的image $\mathbf{I}_L$，等价于以 $\mathbf{C}_L \in \mathbf{A}_L$ 为pinhole center、对virtual world $\mathbf{W}'$ 做pinhole projection得到的image。

这是整篇paper的几何基石。它意味着：
- 如果 $\mathbf{A}_L$、$\mathbf{A}_R$ 退化为单点 $\mathbf{C}_L$、$\mathbf{C}_R$，$\mathbf{I}_L$、$\mathbf{I}_R$ 就是 $\mathbf{W}'$ 的一个标准stereo pair。
- 真实情况下 $\mathbf{A}_L$、$\mathbf{A}_R$ 是半圆 aperture，每个 $\mathbf{C} \in \mathbf{A}_L$ 都贡献一张 $\mathbf{W}'$ 的pinhole image，**superpose起来就blur**了。这就是DP defocus blur的成因。

### 2.2 从RGB-D合成DP image的核心公式

**Eq. (1)**：world point $\mathbf{X}=(X,Y,Z)$ 在virtual world里的位置
$$\mathbf{X}' = \frac{f}{f+X}(X,Y,Z)$$
- $f$：focal length
- $X$：object的X坐标（depth方向，负值）
- 这是thin-lens公式 $1/d + 1/d' = 1/f$ 的向量形式

**Eq. (3)**：纯depth形式
$$d' = \frac{fd}{d-f}$$
- $d$：object depth（绝对值，正值）
- $d'$：virtual image距离（正值）
- 注意 $d>f$ 时 $d'>0$（real image），$d<f$ 时 $d'<0$（virtual image，常见macro情况）

**Eq. (4) / Eq. (5)**：把virtual point $\mathbf{X}'$ 通过aperture上的点 $\mathbf{C}=(0,Y_0,Z_0)$ 投影到sensor plane $X=F$：
$$T(\mathbf{X}, \mathbf{C}) = \frac{d'-F}{d'}\mathbf{C} + \frac{F}{d'}\mathbf{X}' = \frac{d'-F}{d'}\mathbf{C} + F\left(\frac{y}{f},\frac{z}{f}\right)$$
- 第一项 $\frac{d'-F}{d'}\mathbf{C}$：由于aperture位置 $\mathbf{C}$ 引起的offset，**这就是blur和disparity的来源**。
- 第二项 $F(\frac{y}{f},\frac{z}{f})$：base projection，跟 $\mathbf{C}$ 无关，是in-focus时的位置。
- 关键：当 $d'=F$（点恰好在sensor上，即in focus），第一项消失 → **零disparity，零blur**。完美对应前面的观察。
- 当 $d'\neq F$，$\frac{d'-F}{d'}\mathbf{C}$ 给出 $\mathbf{C}$ 偏离lens center造成的位移，对 $\mathbf{C}_L$、$\mathbf{C}_R$ 不同 → **disparity**；对aperture内连续 $\mathbf{C}$ 集合积分 → **blur disk**。

scale factor $s = F/d'$：
- $s>0$：sensor在virtual image前面，blur disk与aperture同向
- $s<0$：sensor在virtual image后面，blur disk倒置（这个会出现在depth极小或 $d<f$ 的macro区域，对应real DP image里"反过来的散斑"）

### 2.3 Intuition：为什么blur和disparity是同一个量

从Eq. (5)能直接读出来：
$$\text{disparity}(\mathbf{X}) \propto \frac{d'-F}{d'} \cdot \text{(aperture separation)}$$
$$\text{blur radius}(\mathbf{X}) \propto \frac{d'-F}{d'} \cdot \text{(aperture size)}$$

两者**只差一个常数比例**（aperture separation vs aperture size）。这就是为什么paper能"用blur监督disparity"——它们线性相关。

---

## 3. DP Simulator（Section 3.3，工程上的小聪明）

Naive合成：对每个RGB-D像素 $(y,z)$，把它的intensity $\mathbf{I}(y,z)$ 均匀spread到 $\mathbf{R}_L$（left view中的blur disk）内所有像素。复杂度 $O(n \cdot |\mathbf{R}|)$，blur disk大就慢。

**Integral image trick**：把left/right aperture近似成rectangle，blur disk也是rectangle（4个corner由Eq.5算出）。定义differential mask $\mathcal{T}_L$：
$$\mathcal{T}_L(\mathbf{p}_{tl}) = +\frac{\mathbf{I}(y,z)}{|\mathbf{R}_L|},\quad \mathcal{T}_L(\mathbf{p}_{tr}) = -\frac{\mathbf{I}(y,z)}{|\mathbf{R}_L|}$$
$$\mathcal{T}_L(\mathbf{p}_{bl}) = -\frac{\mathbf{I}(y,z)}{|\mathbf{R}_L|},\quad \mathcal{T}_L(\mathbf{p}_{br}) = +\frac{\mathbf{I}(y,z)}{|\mathbf{R}_L|}$$
- $\mathbf{p}_{tl}, \mathbf{p}_{tr}, \mathbf{p}_{bl}, \mathbf{p}_{br}$：blur disk的top-left/top-right/bottom-left/bottom-right四个corner坐标
- $|\mathbf{R}_L|$：blur disk面积
- 这四点构成一个"box filter"的差分表示

然后积分：
$$\hat{\mathbf{B}}_{\{L,R\}} = \tau(\mathcal{T}_{\{L,R\}})$$
- $\tau(\cdot)$：integral image操作（summed area table [Crow 1984]）

复杂度变成 $O(n)$，跟blur disk大小无关。这个simulator是这篇paper另一大贡献，因为Canon/Google的real DP数据很难拿到depth GT。

---

## 4. DDDNet 架构（Section 4.1）

```
                  ┌────────────────────┐
B_{L,R} ─────────►│  DepthNet g(·)      │──► D̂_c (coarse inverse depth)
                  └────────────────────┘            │
                                                    ▼
                  ┌────────────────────────────────────┐
B_{L,R}, D̂_c ───►│  DeblurNet f(·) (encoder-decoder)  │──► {Î, D̂}
                  └────────────────────────────────────┘
                                                    │
                                                    ▼
                  ┌────────────────────────────────────┐
Î, D̂ ────────────►│  DP Simulator (differentiable)     │──► B̂_{L,R}
                  └────────────────────────────────────┘
                                                    │
                                                    ▼
                                            Reblur Loss vs B_{L,R}
```

- **DepthNet** $g(\cdot; \vec{\mathcal{G}})$：基于Hierarchical NAS stereo [Cheng et al. 2020]，输入DP pair，输出coarse inverse depth $\hat{\mathbf{D}}_c$。这里用inverse depth $1/d$ 是因为近处深度变化大、远处变化小，inverse depth在数值上更friendly。
- **DeblurNet** $f(\cdot; \vec{\mathcal{F}})$：基于multi-patch network [Zhang et al. 2019]，encoder-decoder结构，输入是DP pair与coarse inverse depth的concat，同时输出deblurred image $\hat{\mathbf{I}}$ 和refined inverse depth $\hat{\mathbf{D}}$。
- **DP Simulator**：可微，用 $\hat{\mathbf{I}}$ 与 $\hat{\mathbf{D}}$ 重新合成DP pair $\hat{\mathbf{B}}_{L,R}$，跟input $\mathbf{B}_{L,R}$ 算reblur loss。这是**让forward model 反过来约束inverse problem**的关键，跟NeRF、phase retrieval、可微渲染的思路一脉相承。

---

## 5. Loss 函数（Section 4.2）

总loss：
$$\mathcal{L} = \mathcal{L}_{\text{res}} + \mathcal{L}_d + \mathcal{L}_{\text{reb}}$$

**Image restoration loss**（Eq. 9）：
$$\mathcal{L}_{\text{res}} = \frac{1}{N}\sum_{y,z}\|\mathbf{I}(y,z) - \hat{\mathbf{I}}(y,z)\|$$
- $N$：pixel总数
- $\|\cdot\|$：$\ell_2$ norm
- $\mathbf{I}$：ground truth sharp image
- $\hat{\mathbf{I}}$：网络deblur输出

**Depth loss**（Eq. 10）：
$$\mathcal{L}_d = \frac{1}{N}\sum_{y,z} S(\mathbf{D}(y,z) - \hat{\mathbf{D}}(y,z))$$
- $S(\cdot)$：smooth $\ell_1$ loss（Fast R-CNN [Girshick 2015]里那种，对小残差用 $\ell_2$，对大残差用 $\ell_1$，对outlier鲁棒）
- $\mathbf{D}$：GT inverse depth
- $\hat{\mathbf{D}}$：网络输出inverse depth

**Reblur loss**（Eq. 11，paper的核心创新）：
$$\mathcal{L}_{\text{reb}} = \frac{1}{N}\sum_{y,z}\|\mathbf{B}_{\{L,R\}}(y,z) - \hat{\mathbf{B}}_{\{L,R\}}(y,z)\|$$
- $\mathbf{B}_{\{L,R\}}$：input DP pair（真实blur图）
- $\hat{\mathbf{B}}_{\{L,R\}}$：用网络输出的 $\hat{\mathbf{I}}$ 与 $\hat{\mathbf{D}}$ 通过DP simulator重新blur得到的DP pair
- 直觉：如果 $\hat{\mathbf{I}}$ 和 $\hat{\mathbf{D}}$ 都对，那用forward model blur回去应该等于真实blur。这个loss等价于说"$\hat{\mathbf{I}}, \hat{\mathbf{D}}$ 必须落在DP model约束的manifold上"，是一种基于物理的自监督，可以在没有GT depth时fine-tune（后面实验就这么用）。

---

## 6. 实验数据与结果解析

### 6.1 Datasets
| Dataset | Type | 用途 |
|---------|------|------|
| DPD-blur [Abuolaim 2020] | Real, Canon 5D IV | deblurring测试 |
| DPD-disp [Punnappurath 2020] | Real, Canon | depth测试，只有test set |
| Our-syn | Synthetic，用NYU Depth v2 + DP simulator | 训练 |
| Our-real | Real, Canon 150 scenes, f/4~f/22 | 测试，更diverse |

### 6.2 Deblurring定量结果（Table 1）

| Dataset | Method | PSNR↑ | SSIM↑ | MSE_rel↓ |
|---------|--------|-------|-------|----------|
| DPD-blur | EBDB | 24.82 | 0.801 | 5.74 |
| DPD-blur | DMENet | 23.93 | 0.812 | 6.36 |
| DPD-blur | DPDNet | 25.53 | 0.826 | 5.29 |
| DPD-blur | Ours_wb | 26.15 | 0.827 | 4.93 |
| DPD-blur | **Ours_reb** | **26.76** | **0.842** | **4.59** |
| Our-syn | Ours_reb | 33.21 | 0.956 | 2.17 |
| Our-real | Ours_reb | 24.03 | 0.850 | 6.13 |

关键takeaway：
1. **Reblur loss带来约0.6dB提升**（Ours_wb → Ours_reb），且SSIM也涨。MSE_rel相对提升~10%。
2. 在Our-real（更难、更diverse的real data）上优势更明显，说明forward model regularization对domain shift更鲁棒。

### 6.3 Depth定量结果（Table 2，DPD-disp dataset）

这个dataset只有test set，论文做了两个实验：
- **Ours**：直接用Our-syn训练的model在DPD-disp上test（zero-shot transfer），AI(1)=0.0906
- **Ours_ft**：用reblur loss在DPD-disp上self-supervised fine-tune（**不用GT depth**），AI(1)=0.0609，逼近SOTA DPdisp的0.0481

这是reblur loss的关键证据：**只要有real DP pair，没有GT也能self-supervise**。这种能力对real-world deployment至关重要。

### 6.4 Ablation（Table 3，Our-syn）

| Stage | abs_rel↓ | rmse↓ | δ<1.25²↑ | PSNR↑ |
|-------|----------|-------|----------|-------|
| Ours_b (DepthNet only) | 0.149 | 1.222 | 0.930 | – |
| Ours_wb (+DeblurNet) | 0.091 | 0.599 | 0.993 | 32.17 |
| Ours_reb (+reblur loss) | **0.083** | **0.461** | **0.998** | **33.22** |

三个stage逐渐变好，说明：
1. Joint depth+deblur 比 depth-only 显著好（abs_rel从0.149→0.091，deblur Net提供了更clean的input到depth head）
2. Reblur loss 进一步refine（0.091→0.083），物理一致性约束有效

### 6.5 Sim-to-Real Transfer

一个非常有意思的实验：用Our-syn训好的model直接test on DPD-blur，得PSNR=20.28dB；用DPD-blur fine-tune后得26.92dB，**比从头训DPD-blur（26.76dB）还高**。说明simulator合成数据有真实信息含量，可以用作pretrain。用一半数据fine-tune也达到26.52dB，simulator学到的representation是useful的。

---

## 7. 这篇paper在更大图景里的位置

**正向可微forward model约束inverse problem**的思路这几年很常见：
- NeRF：volume rendering forward model约束radiance field
- Differentiable rendering：约束3D shape/material
- Phase retrieval：forward Fourier transform约束object
- 本文：DP imaging forward model约束 (sharp image, depth) pair

每个case都是：inverse problem通常ill-posed，加上forward model的物理约束就让解空间更紧致。本文的reblur loss本质上是把DP simulator当作"$\hat{\mathbf{I}}, \hat{\mathbf{D}} \to \mathbf{B}$"的可微函数，构成consistency loss。

**跟后续工作的联系**：
- Abuolaim et al.后续做"Continuous Dual-Pixel"(CVPR 2021)，把aperture split做得更连续
- DP在smartphone上做portrait mode（Google Pixel的Synthetic Depth-of-Field [Wadhwa 2018]）是工业应用代表
- Defocus deblur + depth一脉到DMENet、Multi-depth defocus map等
- 最近(2023-2024) Apple iPhone的计算摄影也用类似DP cues做depth

---

## 8. 可能的不足与联想

1. **Aperture shape矩形近似**：simulator把半圆aperture近似成rectangle，blur disk形状会不准。Real aperture是圆/半圆，blur kernel应该是半椭圆而不是矩形，边缘会失真。可以用Gaussian近似或真正的convolution实现，但会慢。
2. **Thin lens假设**：忽略了lens aberration、衍射、色差。Real DP sensor的blur kernel不是完美的geometric disk，有ring结构。可以学一个residual PSF。
3. **Textureless region**：depth-from-defocus和stereo都怕textureless区域。paper没专门处理，可以引入edge-aware smoothness prior（像Monodepth2那种）。
4. **Reblur loss与GT depth的协同**：reblur loss只约束到 $\mathbf{B}$ 的manifold，但manifold有歧义——不同 $(\mathbf{I}, \mathbf{D})$ 可能blur出同一 $\mathbf{B}$（多解）。所以paper仍需GT supervision做主信号，reblur是regularizer。要完全self-supervised需要额外prior（如sharp image的natural image prior、depth的smoothness prior）。
5. **Aperture size的尺度ambiguity**：$\frac{d'-F}{d'}$ 与aperture size乘积决定blur，二者有尺度耦合。所以DP-based depth通常是affine invariant [Garg 2019]，必须靠scene prior或supervision定scale。
6. **Macro区域**：$d < f$ 时virtual image在lens左边（$d' < 0$），blur disk翻转。Simulator Eq.(5)的 $s=F/d'$ 此时为负，几何上对应blur disk倒置（符合物理），但神经网络学到这个case可能困难，real macro DP数据稀缺。
7. **跟plenoptic camera的对比**：Lytro等light field camera也做depth+refocus，但需要microlens array牺牲spatial resolution。DP sensor没有spatial resolution损失，只有angular两view，是light field的极端下采样版。理论上DP是2-view light field，可以从这里扩展到4-view（quad-pixel，Sony已经有）、6-view等，更多view能更好解aberration和depth ambiguity。
8. **跟event camera / neuromorphic sensor的联系**：异步sensor的blur model跟defocus blur在频域有相似性，DP的left/right差分也能看成某种"spatial contrast"，跟event的temporal contrast对偶。
9. **大模型时代**：今天回头看，DDDNet很小（Titan XP训练），如果用large-scale pretrain vision encoder（DINOv2、MAE）做backbone，DP pair作为input可能能提取更强representation，reblur loss可以做为pretrain task。
10. **Generative model角度**：可以把forward DP model当作某种"condition"放到diffusion model里，让model生成 sharp image给定DP pair——本质上是conditional diffusion deblurring。Reblur loss会变成diffusion training里的score matching constraint的一种物理informative版本。

---

## 9. 总结

这篇paper的几个关键intuition：
1. **DP sensor的blur与disparity是同一物理量**的两个表现，jointly model比分别处理更优。
2. **Forward model可微化**（DP simulator）让物理约束以loss形式融入网络训练，reblur loss既能在有GT时辅助训练，也能在无GT时self-supervised fine-tune。
3. **Depth和deblur互相帮助**：deblur提供更clean的input给depth估计；depth提供blur kernel形状给deblur。两者形成正向循环。
4. **Simulator**让DP数据获取不再受限于Canon/Google，可以从任何RGB-D dataset合成，开启了DP领域的大规模学习。

对于你（Karpathy）可能最感兴趣的点：
- Forward model作为可微layer做inverse problem的clean formulation
- Self-supervised adaptation via physical consistency loss（reblur loss等价于cycle consistency，但锚定在物理模型而非identity）
- 极小baseline stereo与传统depth-from-defocus的统一

这套思路在image formation model + inverse problem这个大框架里非常优雅，跟可微渲染、NeRF、inverse graphics是一脉相承的研究范式。

### 相关参考链接
- Paper PDF: https://arxiv.org/pdf/2012.08002
- GitHub: https://github.com/pan-liny/Dual-Pixel-Exploration
- DPD dataset (Abuolaim): https://www.eecs.yorku.ca/~abuolaim/dpd_dataset/
- DPdisp (Punnappurath, ICCP 2020): https://www.eecs.yorku.ca/~abuolaim/dpd_disp/
- Du²Net (Zhang et al.): https://arxiv.org/abs/2003.14299
- Monodepth2 (Godard et al.): https://github.com/godardfar/monodepth2
- BTS (Lee et al.): https://github.com/cogaplex-bts/bts
- NYU Depth v2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- Summed area tables (Crow 1984): https://en.wikipedia.org/wiki/Summed-area_table
- Hierarchical NAS stereo (Cheng et al.): https://arxiv.org/abs/2010.13501
- Multi-patch deblur (Zhang et al. CVPR 2019): https://github.com/HongguangZhang/Deep-Stacked-Hierarchical-Multi-patch-Network
