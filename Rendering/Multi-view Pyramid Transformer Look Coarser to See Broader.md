---
source_pdf: Multi-view Pyramid Transformer Look Coarser to See Broader.pdf
paper_sha256: 82e6debb52062e95eae2ded2a88a5ed62ecab95804382ee156dac96b397f11c6
processed_at: '2026-08-05T21:16:55-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲MVP这篇paper

## 一句话概括

这篇paper说白了就是: **你看远处的山的时候,眼睛不会去抠每片树叶的细节,而是大致扫一眼轮廓和走势;只有看近处的时候,你才会去关注纹理和细节**。MVP把这个道理用到了multi-view 3D reconstruction上,让transformer学会"该粗的时候粗,该细的时候细"。

## 为什么要搞这个东西?

你想象一下这个场景: 你有一堆照片,比如128张,拍的是同一个scene的不同角度。你想用一个feed-forward network一次性把它们变成3D model。

Standard transformer的做法是什么? 把所有照片patchify成token,扔进一个giant self-attention,让所有token互相attend。听起来很直接,但问题来了:

**Problem 1: 算不动**

128张960×540的图片,patch size=8,那就是 $128 \times (960/8) \times (540/8) = 128 \times 8100 = 1,036,800$ 个token。Self-attention的complexity是 $O(n^2)$,也就是 $10^{12}$ 量级的attention matrix。H100有80GB memory也扛不住。

**Problem 2: attention会变蠢**

这个更隐蔽也更致命。你想想,token数量从16 views增加到256 views,attention要覆盖的范围大了16倍。softmax之后,attention的probability distribution被spread得越来越flat。本来query应该强烈attend到几个corresponding token上,现在被稀释到一堆noise token上,signal-to-noise ratio下降。

这就是paper里说的**attention dilution**。iLRM和Long-LRM的实验数据就证明了这一点: 给它们更多views,PSNR反而掉下来。按理说更多input应该reconstruction更好,结果反而变差。

这就像你让一个人同时听100个人说话,他谁都听不清;只让他听4个人说话,他反而能抓住每个人说的内容。

## MVP的核心idea: 两个axis同时做hierarchy

### Axis 1: Inter-view hierarchy (views之间的关系)

这个axis是说: **views之间的attention范围要逐渐扩大**。

- **Stage 1 (Frame-wise)**: 每张图自己玩自己的,只做frame内部的self-attention。这就像每张图先自己消化一下,提取local spatial features。
  
- **Stage 2 (Group-wise)**: 把views分成小组,比如每4张一组。组内做cross-view attention,但组与组之间不通信。这就像把128个人分成32个小组,小组内讨论,但小组之间暂时不交流。

- **Stage 3 (Global)**: 所有views一起做attention,此时token已经是coarse representation了,数量不大,可以handle。

为什么这样work? 因为geometric correspondence通常是**local的** — 相邻几张图的overlap区域最大,correspondence最明显。先在local scale建立这些correspondence,再往global推,比一上来就full global attention要stable得多。

### Axis 2: Intra-view hierarchy (每张图内部spatial)

这个axis是说: **每张图内部的token resolution要逐渐降低**。

- **Stage 1**: patch size=8, 每张图有 $8100$ 个token, dim=256
- **Stage 2**: spatial downsample 2×, patch size等效=16, 每张图 $2025$ 个token, dim=512  
- **Stage 3**: 再downsample 2×, patch size等效=32, 每张图 $506$ 个token, dim=1024

每个token覆盖的spatial区域越来越大,信息越来越"浓缩"。这就像CNN里的feature pyramid: 早期layer看细节,后期layer看global structure。

### 两个axis为什么complementary?

关键insight: **两个axis是反方向增长的,这恰好控制了token budget**。

你看,如果只做inter-view hierarchy不做intra-view hierarchy,那到Stage 3 global attention时,还是得处理 $N \times 8100$ 个token,一样爆炸。

如果只做intra-view hierarchy不做inter-view hierarchy,那所有views从一开始就full attention,即使spatial downsampling了,Stage 1的cost也是 $O(N^2 \times hw)$。

两个hierarchy同时做: **spatial变coarse的时候,view attention变global**。Stage 3做global attention时,每个view只有506个token,128 views也就 $64768$ 个token,完全handleable。而Stage 1做fine attention时,只做frame-wise,token数量是 $8100$ per view,也不爆炸。

这就是paper标题"Look Coarser to See Broader"的精髓: **你把spatial看粗了,才能afford把view看广**。

## 架构具体怎么走的

### Input encoding

给N张图 $I_i \in \mathbb{R}^{H \times W \times 3}$:

1. 每张图配一个9D Plücker ray map $P_i \in \mathbb{R}^{H \times W \times 9}$,把camera pose编进去
   - 9D = origin (3D) + direction (3D) + cross product (3D)
   - 这个ray map告诉network: 这个pixel是从哪个camera位置、朝哪个方向拍到的
2. Image和ray map concat: $\tilde{I}_i \in \mathbb{R}^{H \times W \times 12}$
3. Patchify (patch size=8) + linear projection
4. 加4个register tokens per view (吸收artifacts,从[Darcet et al.](https://arxiv.org/abs/2309.16588)借鉴)

### Three stages具体配置

```
Stage 1: Frame-wise attention (2 blocks)
  - patch size 8, dim 256
  - 每张图独立做self-attention
  
Stage 2: Group-wise attention (4 blocks)
  - patch size 16 (等效), dim 512
  - 4 views per group
  - 先frame-wise,再group-wise
  
Stage 3: Global attention (8 blocks)
  - patch size 32 (等效), dim 1024
  - 所有views一起做attention
```

Stage之间用conv layer做spatial downsampling (2×) + channel up-projection (2×)。

### Group-wise attention的数学

$$G \gets \text{group}(T)$$
$$G_{i,j} \gets \text{self-attn}(G_{i,j}) \quad \text{(frame-wise)}$$
$$T_i \gets \text{self-attn}(G_i) \quad \text{(group-wise)}$$

变量:
- $T \in \mathbb{R}^{Nhw \times d}$: 所有input tokens
- $G_{i,j} \in \mathbb{R}^{hw \times d}$: 第$i$个group第$j$张图的tokens
- $G_i \in \mathbb{R}^{Mhw \times d}$: 第$i$个group的所有tokens
- $M=4$: group size

就是一个两步attention: 先组内每张图自己attend,再组内所有图互相attend。简单粗暴但有效。

### Pyramidal Feature Aggregation (PFA)

三个stage各自输出feature map,怎么merge? 用top-down refinement:

$$F = \text{fuse}(\text{up}(\text{fuse}(\text{up}(F^{(3)}) + F^{(2)})) + F^{(1)})$$

- 从最coarse的 $F^{(3)}$ 开始
- Upsample,与 $F^{(2)}$ residual fuse
- 再upsample,与 $F^{(1)}$ residual fuse
- 得到既有global context又有local detail的unified feature

这个设计灵感来自[DPT](https://arxiv.org/abs/2103.13413)。Ablation显示没有PFA,PSNR从22.79跌到21.58,LPIPS从0.235恶化到0.340。说明**coarse和fine信息都需要,缺一不可**。

### Output decoding

每个pixel预测一个3D Gaussian:
- $\mu_j$: 位置 (3D)
- $s_j$: scale (3D)
- $q_j$: rotation quaternion (4D)
- $\alpha_j$: opacity (scalar, 用spherical harmonics degree=2)
- $c_j$: color (spherical harmonics degree=1)

Opacity也用spherical harmonics是借鉴[VOD-3DGS](https://arxiv.org/abs/2501.17978),防止opacity随viewpoint突变产生artifacts。

### Loss function

$$\mathcal{L}_{\text{img}} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} (\mathcal{L}_{\text{MSE}}(\hat{I}_i, I_i) + 0.2 \cdot \mathcal{L}_{\text{percept}}(\hat{I}_i, I_i))$$

$$\mathcal{R}_\alpha = \frac{1}{N_\mathscr{G}} \sum_{j=1}^{N_\mathscr{G}} |\sigma(\alpha_j \cdot \omega_j)|$$

$$\mathcal{L} = \mathcal{L}_{\text{img}} + 0.001 \cdot \mathcal{R}_\alpha$$

变量:
- $\mathcal{T}$: target view indices
- $\hat{I}_i, I_i$: rendered vs ground-truth
- $N_\mathscr{G}$: Gaussian总数
- $\sigma$: Sigmoid
- $\alpha_j$: opacity
- $\omega_j$: random view direction的spherical harmonic basis

Opacity regularization非常小 ($\gamma=0.001$),只是防止view-dependent opacity artifacts。

## 实验数据有多impressive

### DL3DV主战场

| Setting | MVP | iLRM | Long-LRM | 3D-GS (30k) |
|---------|-----|------|----------|-------------|
| 32 views PSNR | **25.96** | 24.30 | 23.97 | 24.43 |
| 32 views Time | **0.17s** | 0.53s | 0.84s | 8min |
| 128 views PSNR | **29.02** | 22.98 | 21.24 | 29.43 |
| 256 views PSNR | **29.67** | 20.63 | OOM | 30.75 |
| 256 views Time | **1.84s** | 20.92s | OOM | 8min |

关键观察:
1. **32 views下MVP已经beat optimization-based 3D-GS** (25.96 vs 24.43),这是feed-forward method第一次做到
2. **256 views下baselines全部崩盘**:
   - Long-LRM直接OOM
   - iLRM PSNR从24.30 (32 views)跌到20.63 (256 views) — 这就是attention dilution
   - MVP从25.96一路涨到29.67,没有dilution
3. **Speed碾压**: MVP在256 views下1.84秒,iLRM要20.92秒

### Zero-shot generalization (Tanks&Temples, Mip-NeRF360)

128 views setting:
- Tanks&Temples: MVP 22.36, iLRM 19.22, Long-LRM 18.47
- Mip-NeRF360: MVP 25.12, iLRM 21.32, Long-LRM 19.82

注意baselines在128 views下比32 views还差:
- Long-LRM Tanks&Temples: 32 views 18.59 → 128 views 18.47 (下降!)
- Long-LRM Mip-NeRF360: 32 views 21.08 → 128 views 19.82 (下降!)

这是attention dilution的smoking gun。给baselines更多views,它们反而用不好。MVP通过hierarchy design让更多views持续带来benefit。

### Longer context extrapolation

只用32 views训练,test时给32/40/48 views:

| Method | 32v PSNR | 40v PSNR (gain) | 48v PSNR (gain) |
|--------|----------|-----------------|------------------|
| Long-LRM | 23.97 | 24.18 (+0.21) | 24.30 (+0.33) |
| iLRM | 24.30 | 24.54 (+0.24) | 24.78 (+0.48) |
| MVP | 25.88 | 26.36 (+0.48) | 27.06 (+1.18) |

MVP在longer context下的gain是baselines的2-3倍,而且inference time增长缓慢 (0.17s→0.26s,只涨53%),baselines快速增长 (0.84s→1.38s,涨64%)。

这证明hierarchical design不仅解决efficiency问题,更本质地解决了**long context利用能力**问题。

## Ablation揭示了什么

### Group-wise attention值不值得?

Stage 2换成全frame-wise: PSNR 22.53 (差0.26)
Stage 2换成全global: PSNR 22.94 (好0.15)
Stage 2用group-wise (baseline): PSNR 22.79

性能上group-wise比global只差0.15dB,但cost上随views增加差距越来越大。Group=4时cost是 $O(N \cdot M^2)$ scale,global是 $O(N^2)$ scale。这就是group-wise的sweet spot: **用很小的accuracy代价换取巨大的efficiency gain**。

### Dual hierarchy缺一不可

最震撼的ablation:
- 去掉inter-view hierarchy (全用global): 256 views直接OOM,64 views时比MVP慢6×
- 去掉intra-view hierarchy (patch size固定8): 256 views OOM,64 views时慢50×
- 两个都去掉: 64 views时慢80×

这就是为什么必须dual hierarchy。单个axis不够,必须两个axis同时control token budget。

### Reversed hierarchy是个反例

把order反过来: global→group→frame,coarse→fine。PSNR从22.79暴跌到18.95。

为什么? 因为**信息丢失是单向的**。你先做coarse processing丢了fine details,后面再想recover fine details是不可能的。先global attention建了coarse correspondence,后面再做local attention没有新的information可以补充。

这就像你看一张模糊图片先建立了大概印象,然后给你看清晰版,你能利用清晰信息。反过来,先看清晰版记住了所有细节,然后给你看模糊版,你反而会confused。

## 为什么这个design本质上work

### Attention dilution的数学直觉

Standard global attention的problem: softmax的distribution entropy随token数量增加而增加。

假设每个query真正strongly attend的token数量是 $k$ (correspondence的数量),总token数量是 $n$。当 $n \gg k$ 时,softmax会把probability mass spread到大量noise tokens上,真正important的 $k$ 个token得到的attention mass被dilute。

MVP的hierarchy design让每层attention的 $n$ 保持bounded:
- Stage 1: $n \approx hw$ (frame-wise, fine)
- Stage 2: $n \approx M \cdot hw$ (group-wise, medium)
- Stage 3: $n \approx N \cdot h'w'$ (global, coarse, 但 $h'w'$ 已经很小)

这就像你让那个听100个人说话的人,先4人一组讨论,每组总结成1个观点,然后再100个组的代表一起讨论。这样每个人在任何阶段都只需要handle小规模信息,不会被drown out。

### 为什么fine-to-coarse而不是coarse-to-fine?

你可能会问: 为什么不先做coarse global attention建立大致correspondence,再做fine local attention refine细节?

Ablation告诉你: 这样做PSNR从22.79跌到18.95。原因:

1. **Information loss is irreversible**: Coarse processing把fine details丢掉了,后面refine不回来
2. **Local processing需要global guidance**: 但global guidance建在coarse representation上,不够precise
3. **Optimization instability**: 先coarse后fine相当于先easy后hard,gradient signal混乱

Fine-to-coarse的好处: 先在fine scale建立precise local correspondence,这些correspondence作为"anchor"传到coarse scale,coarse scale的global attention基于这些reliable anchors做更稳定的long-range integration。

### Group size为什么是4?

Table 11的spatial cognition task很illuminating:
- Group=2: 83.6% (context太少)
- Group=4: 96.4% (sweet spot)
- Group=8: 97.1% (marginal gain)

Group=2时两个views的overlap可能不够establish reliable correspondence。Group=4时4个views形成一个reasonable local batch,有足够overlap做几何推理。Group=8时多出来的views提供redundant信息,不带来新information但增加cost。

这也符合intuition: 相邻4张照片的overlap通常足够大,correspondence清晰;再往后加的views可能overlap减少,marginal value递减。

## 这个paper对field的启示

### 1. Hierarchy design > 换sequence model架构

Long-LRM用Mamba降低complexity,但expressive power受限。MVP用pure transformer + hierarchy就beat了Mamba-based方法。这暗示:

**Complexity问题不一定需要换architecture,通过hierarchical design让standard attention保持expressive power的同时控制cost,可能更general**。

### 2. Attention dilution是real problem

之前大家以为是computational bottleneck限制了multi-view transformer的scaling。这篇paper指出attention dilution是更本质的问题 — 即使你afford得起computation,full global attention在long context下也会degenerate。

### 3. Feed-forward method可以beat optimization-based

32 views下MVP PSNR 25.96 > 3D-GS 30k iter 24.43。这是feed-forward method第一次在fair comparison下beat optimization-based per-scene optimization。这意义重大: 说明learned prior可以compensate优化不足。

### 4. Multi-view reasoning有inductive bias

不是所有correspondence都需要full global attention。Local-to-global的progression更符合multi-view geometry的本质: 相邻views overlap大,correspondence强;远距离views可能没overlap,只能通过intermediate views间接建立关系。MVP的hierarchy恰好mimic这个过程。

## 训练策略也有讲究

Three-stage training schedule:

**Stage 1 (warmup)**: 低分辨率480×256,固定32 input views,100k iter。让模型先学会基本multi-view reasoning,computational cost可控。

**Stage 2 (high-res)**: 切到960×540,还是32 views但减少target views到6 (memory constraint),50k iter。让模型适应high resolution。

**Stage 3 (variable views)**: 保持960×540,用不同数量的input views训练,30k iter。**冻结frame-wise和group-wise modules,只train global modules**。

Stage 3的freezing很关键: 前两个stage学到的local和group representation已经很robust,不需要再update。只需要让global attention module适应不同view counts,这样fine-tuning cost低且不会catastrophic forgetting。

整个训练用32 H100 GPUs,9天 (4+3+2)。虽然不便宜,但考虑model capacity和task complexity,算是reasonable。

## 局限性和我的猜测

Paper没讨论但我觉得重要的limitation:

1. **Known camera pose assumption**: 需要预先用COLMAP算好pose。如果能end-to-end joint estimate pose和reconstruct,实用价值更高。VGGT已经证明可行,MVP可能可以extend。

2. **Static scene only**: 现在只做static reconstruction。Extend到dynamic scene (4D)需要temporal modeling,[4D-GS](https://arxiv.org/abs/2403.12742)提供了可能的技术path。

3. **Photometric loss only**: 没用geometry supervision。Table 12显示已经beat baseline,但如果加depth/point cloud supervision ([DUSt3R](https://arxiv.org/abs/2312.14132), [VGGT](https://arxiv.org/abs/2502.02066) style),几何精度可能进一步提升。

4. **为什么不用DINO encoder?** Ablation里"w/o Dual Hierarchy"是modified VGGT用linear patch embedding替代DINO。这是deliberate choice还是limitation? 如果用[DINOv2](https://arxiv.org/abs/2304.07193) features作为input,hierarchy design是否还能work? 我猜测hierarchy attention本身已经提供了足够inductive bias,pretrained features可能redundant甚至conflict。

5. **Group partitioning strategy**: 现在按frame index locality分组。如果views是unordered或非sequential (比如internet photos),怎么定义locality? 用visual overlap或camera distance作为grouping criterion可能更general。

## 给Karpathy的额外联想

Andrej,你之前在[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)里讲过nanoGPT的simplicity。这篇paper某种程度上呼应了你的philosophy: **standard transformer架构如果design得当,就能解决看起来需要复杂架构的问题**。

MVP用的就是standard self-attention,没有Mamba,没有linear attention,没有sparse attention。它只是:
1. 控制attention scope的hierarchy (inter-view)
2. 控制token resolution的hierarchy (intra-view)  
3. Feature pyramid aggregation

三个经典idea的组合,但用得非常precise,而且两者complementary的设计让1+1>2。

这让我想到你说过的一句话: "The best ideas are simple once you understand them." MVP就是这个感觉 — 看完之后觉得"obvious",但想到这个complementary dual hierarchy design并不obvious。

另一个联想: 这篇paper的attention dilution诊断让我想到你在[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)里讲的attention temperature问题。Softmax在长context下的distribution flattening是general problem,MVP通过hierarchy control token budget来缓解,而不是通过temperature scaling或sparse attention。这是一个orthogonal的设计direction,可能对LLM的long context problem也有启发。

---

**References**:
- [MVP Project Page](https://gynjn.github.io/MVP/)
- [Long-LRM](https://arxiv.org/abs/2503.04652)
- [iLRM](https://arxiv.org/abs/2507.23277)
- [VGGT](https://arxiv.org/abs/2502.02066)
- [DUSt3R](https://arxiv.org/abs/2312.14132)
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.14737)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [DPT](https://arxiv.org/abs/2103.13413)
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- [PRoPE](https://arxiv.org/abs/2503.04652)
- [FlashAttention-3](https://arxiv.org/abs/2407.11062)
- [DL3DV-10K](https://arxiv.org/abs/2403.15037)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Lost in the Middle (attention dilution in LLMs)](https://arxiv.org/abs/2307.03172)
- [Token Merging](https://arxiv.org/abs/2210.09461)
- [Video Swin Transformer](https://arxiv.org/abs/2106.13290)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [4D Gaussian Splatting](https://arxiv.org/abs/2403.12742)

---

# Multi-view Pyramid Transformer (MVP) 深度技术解析

## 1. Paper核心动机与定位

这篇paper要解决的核心问题是: **如何让multi-view transformer在处理几十到上百张高分辨率图像时,既能保持quality又能保持scalability**。

当前landscape的关键痛点:
- **Long-LRM** [arxiv](https://arxiv.org/abs/2503.04652): 用Mamba替代部分attention降低complexity,但expressive capacity受限
- **iLRM** [arxiv](https://arxiv.org/abs/2507.23277): 用compact scene representation实现full attention,但views数量增加时computational bottleneck严重,而且出现attention dilution
- **LVT** [arxiv](https://arxiv.org/abs/2509.25001): local-view attention,但global 3D consistency只能通过多层local interaction间接实现,且依赖known camera poses

**关键insight**: attention dilution问题。当input views增加时,attention distribution会被dilute变得不稳定,导致correspondence learning退化。这不是computational问题,而是generalization问题。这是这篇paper最重要的诊断。

## 2. 核心设计哲学:Dual Attention Hierarchy

作者借鉴了两个经典design pattern:
- **CNN的fine-to-coarse** ([ResNet](https://arxiv.org/abs/1512.03385), [VGG](https://arxiv.org/abs/1409.1556)): early layers处理fine spatial features, later layers处理coarse semantically rich features
- **Swin Transformer** ([arxiv](https://arxiv.org/abs/2103.14030)): progressively reduce spatial/temporal resolutions

但是这篇paper的关键创新在于: **两个hierarchy同时且互补地运作**,形成"dual axis":

| Hierarchy | 方向 | 作用 |
|-----------|------|------|
| Inter-view | local → global | views之间: frame-wise → group-wise → global |
| Intra-view | fine → coarse | 单帧内: spatial tokens逐渐merge |

这种complementary design的好处在于: 在early layers,模型在narrow view window + fine tokens上操作,提取local geometric details;在later layers,模型在wider view window + coarse tokens上操作,integrate broader context。两个hierarchy互相"补偿",使得参与attention的token数量不会跨层excessive增长。

## 3. 架构详细解析

### 3.1 Input Encoding

给定N张input images $\{I_i\}_{i=1}^{N}$,其中 $I_i \in \mathbb{R}^{H \times W \times 3}$:

1. **Camera encoding**: 将camera pose编码为9D Plücker ray map $P_i \in \mathbb{R}^{H \times W \times 9}$
   - 每条ray由三部分concatenate: origin (3D) + direction (3D) + cross product (3D)
   - 这种representation比纯intrinsics更geometry-aware

2. **Posed image tensor**: $\tilde{I}_i = \text{concat}(I_i, P_i) \in \mathbb{R}^{H \times W \times 12}$

3. **Patchification**: patch size $p$ (paper中初始为8),通过linear projection
4. **Register tokens**: 每个view添加4个register tokens ([Darcet et al.](https://arxiv.org/abs/2309.16588)),total tokens = $N(HW/p^2 + 4)$

Register tokens的作用是吸收high-norm artifacts,避免attention maps被污染。这是从VGGT ([Wang et al.](https://arxiv.org/abs/2502.02066))继承的设计。

### 3.2 Inter-view Attention Hierarchy

这是paper最核心的formulation。Three stages:

**Stage 1: Frame-wise attention (2 blocks)**
- 每个view独立做self-attention,提取spatial features
- 这一步实际上等价于standard ViT

**Stage 2: Group-wise attention (4 blocks)**
- 关键创新。引入grouping operator:

$$\text{group}(\cdot): \mathbb{R}^{Nhw \times d} \rightarrow \mathbb{R}^{\frac{N}{M} \times Mhw \times d}$$

其中:
- $N$ = total views
- $M$ = views per group (paper中default = 4)
- $h, w$ = downsampled token resolution
- $d$ = embedding dimension

Grouping operator简单按frame index的locality将$N$个views分成$\frac{N}{M}$个连续groups。

然后**两步attention**:

$$G \gets \text{group}(T)$$
$$G_{i,j} \gets \text{self-attn}(G_{i,j}) \quad \forall i, j \quad \text{(frame-wise attention)}$$
$$T_i \gets \text{self-attn}(G_i) \quad \forall i \quad \text{(group-wise attention)}$$

变量解释:
- $T \in \mathbb{R}^{Nhw \times d}$: 所有input tokens
- $G_{i,j} \in \mathbb{R}^{hw \times d}$: 第$i$个group中第$j$个image的tokens
- $G_i \in \mathbb{R}^{Mhw \times d}$: 第$i$个group中所有tokens
- $i \in [1, \frac{N}{M}]$, $j \in [1, M]$

**Intuition**: group-wise attention是local (frame-wise)和global (full attention)之间的中间态。它既保留了大部分cross-view correspondence reasoning的能力,又限制了computational cost为 $O(\frac{N}{M} \cdot M^2 h^2 w^2)$ 而非 $O(N^2 h^2 w^2)$。

**Stage 3: Global attention (8 blocks)**
- 此时$M = N$,所有views属于同一个group
- 这是完整的global self-attention,实现full cross-view信息整合
- 这一步对应Alternating-Attention module ([VGGT](https://arxiv.org/abs/2502.02066))

### 3.3 Intra-view Attention Hierarchy

通过spatial downsampling实现fine-to-coarse:

- **Stage 1**: patch size 8, embedding dim 256 (paper中ablation用的更小)
- **Stage 2**: patch size 16 (等效), embedding dim 512
- **Stage 3**: patch size 32 (等效), embedding dim 1024

实现方式: 单个conv layer同时做spatial downsampling和channel up-projection。每stage token数量减少4倍 ($h \to h/2$, $w \to w/2$),embedding dim翻倍。

**Intuition**: 这相当于让每个token"看到的"spatial区域越来越大,receptive field扩大,feature capacity增加。同时因为token变少,后面的group-wise/global attention的computational cost得到有效控制。

### 3.4 Three-stage具体配置

| Stage | Blocks | Attention Type | Patch Size | Hidden Dim |
|-------|--------|-----------------|------------|------------|
| 1 | 2 | Frame-wise | 8 | 256 (实际ablation用128) |
| 2 | 4 | Group-wise (M=4) | 16 | 512 (ablation用256) |
| 3 | 8 | Global | 32 | 1024 (ablation用512) |

### 3.5 Pyramidal Feature Aggregation (PFA)

灵感来自Dense Prediction Transformer ([Ranftl et al.](https://arxiv.org/abs/2103.13413))的progressive fusion:

$$F = \text{fuse}(\text{up}(\text{fuse}(\text{up}(F^{(3)}) + F^{(2)})) + F^{(1)})$$

变量解释:
- $F^{(1)}, F^{(2)}, F^{(3)}$: 三个stage的reshaped feature maps
- $\text{up}(\cdot)$: upsampling layer (恢复spatial resolution)
- $\text{fuse}(\cdot)$: residual convolutional fusion block

这是一个**top-down refinement**过程:
1. 从最coarse的 $F^{(3)}$ 开始
2. Upsample后与 $F^{(2)}$ fuse
3. 再upsample后与 $F^{(1)}$ fuse
4. 最终得到既有global context又有local detail的feature

ablation显示 (Table 6): 没有PFA时PSNR从22.79跌到21.58,LPIPS从0.235恶化到0.340。说明multi-scale fusion对fine-grained prediction至关重要。

### 3.6 Output Decoding

每pixel参数化一个3D Gaussian primitive,attributes:
- $\mu_j$: position (3D)
- $s_j$: scale (3D)
- $q_j$: rotation quaternion (4D)
- $\alpha_j$: opacity (scalar)
- $c_j$: color (spherical harmonics, degree=1)

opacity也用spherical harmonics (degree=2),这是为了处理view-dependent opacity artifacts,借鉴自[VOD-3DGS](https://arxiv.org/abs/2501.17978)。

### 3.7 Training Objective

**Image loss**:

$$\mathcal{L}_{\text{img}} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} (\mathcal{L}_{\text{MSE}}(\hat{I}_i, I_i) + \lambda \mathcal{L}_{\text{percept}}(\hat{I}_i, I_i))$$

变量:
- $\mathcal{T}$: target view indices集合
- $\hat{I}_i, I_i$: rendered vs ground-truth target view
- $\lambda = 0.2$: perceptual loss权重

**Opacity regularization**:

$$\mathcal{R}_\alpha = \frac{1}{N_\mathscr{G}} \sum_{j=1}^{N_\mathscr{G}} |\sigma(\alpha_j \cdot \omega_j)|$$

变量:
- $N_\mathscr{G} = NHW$: predicted Gaussian primitives总数
- $\sigma(\cdot)$: Sigmoid function
- $\alpha_j$: predicted opacity
- $\omega_j$: spherical harmonic basis from随机sampled per-pixel view direction

这个regularization的目的: 防止opacity随viewpoint变化产生artifacts,即一个Gaussian在某些view下突然透明/不透明。

**Total loss**: $\mathcal{L} = \mathcal{L}_{\text{img}} + \gamma \mathcal{R}_\alpha$,其中 $\gamma = 0.001$ (非常小,只是regularization)

## 4. 实验结果分析

### 4.1 DL3DV主结果 (Table 1, 9, 10)

关键数据点对比 (32 views setting):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time (s) ↓ |
|--------|--------|--------|---------|------------|
| 3D-GS (30k iter) | 24.43 | 0.827 | 0.191 | 480 (8min) |
| Long-LRM | 23.97 | 0.778 | 0.267 | 0.84 |
| iLRM | 24.30 | 0.803 | 0.256 | 0.53 |
| **MVP** | **25.96** | **0.847** | **0.187** | **0.17** |

MVP同时实现了: **比optimization-based方法更好的quality** + **比feed-forward baselines快5-10倍**。

**Scalability关键观察**:
- 256 views: MVP 1.84s, PSNR 29.67; Long-LRM直接OOM; iLRM 20.92s且PSNR退化到20.63 (attention dilution现象!)
- 128 views: MVP 0.77s, iLRM 5.61s, Long-LRM 6.39s
- MVP在256 views下距3D-GS-30k只差0.7dB PSNR,但快250×

这个结果直接验证了paper的核心论断: **hierarchical design有效避免了attention dilution,使模型在long context下仍然能从更多views受益**。

### 4.2 Zero-shot Generalization (Table 3)

在Tanks&Temples和Mip-NeRF360上,128 views setting下:

Tanks&Temples:
- Long-LRM: 18.47 PSNR (从32 views的18.59退化!)
- iLRM: 19.22 PSNR
- MVP: **22.36 PSNR** (+2.77 over iLRM)

Mip-NeRF360:
- Long-LRM: 19.82 PSNR (退化!)
- iLRM: 21.32 PSNR
- MVP: **25.12 PSNR** (+3.8 over iLRM)

这组数据非常关键: baselines在views数量增加时PSNR反而下降,这是attention dilution的典型表现。MVP通过hierarchical design让更多views继续提供benefit。

### 4.3 Longer Context Generalization (Table 5)

所有模型只用32 views训练,test时使用32/40/48 views:

| Method | 32-view PSNR | 40-view PSNR (gain) | 48-view PSNR (gain) |
|--------|--------------|---------------------|----------------------|
| Long-LRM | 23.97 | 24.18 (+0.21) | 24.30 (+0.33) |
| iLRM | 24.30 | 24.54 (+0.24) | 24.78 (+0.48) |
| MVP | 25.88 | 26.36 (+0.48) | 27.06 (+1.18) |

MVP在longer context下的improvement rate明显更高,且inference time增长缓慢(从0.17s到0.26s),而baselines快速增长(0.84→1.38s)。这证明hierarchical design的本质优势不仅在于efficiency,更在于更好地利用了long context information。

### 4.4 RE10K Low-res (Table 4)

256×256分辨率,4/8 views:

| Method | 4-view PSNR | 8-view PSNR |
|--------|-------------|-------------|
| CLiFT | 30.13 | 29.68 |
| iLRM | 30.37 | 28.90 (32-view trained: 31.57) |
| Ours-coarse | 30.56 | 31.78 |
| **Ours-fine** | **32.12** | **33.40** |

MVP的fine variant (patch size 4,8,16)在4 views下就达到32.12 PSNR,说明architecture在sparse view setting下也有优势。

## 5. Ablation深度分析 (Table 6)

### 5.1 Group-wise Attention作用

对比三种Stage 2设置:
- Frame(4) (替代group): PSNR 22.53
- Group(4) (baseline): PSNR 22.79
- Global(4): PSNR 22.94

Group-wise与Global只差0.15dB,但computational cost差异巨大 (Figure 6a显示随views增加差距越来越大)。Group size=4时,group attention cost是 $O(N \cdot M^2)$ scale,而global是 $O(N^2)$ scale。

### 5.2 Dual Hierarchy贡献

最关键的ablation:
- **w/o Inter-view Hierarchy** (all global): 在256 views下OOM,latency比full model高6×
- **w/o Intra-view Hierarchy** (patch size固定8): 256 views直接OOM,64 views时慢50×
- **w/o Dual Hierarchy (p=8)**: 类似modified VGGT,在64 views时慢80×
- **w/o Dual Hierarchy (p=16)**: PSNR跌到21.80 (粗patch但无hierarchy,信息丢失严重)

这组对比强烈说明: 单纯用coarse patch不能替代hierarchical design。hierarchy的本质在于让coarse representation有fine representation作为基础。

### 5.3 Reversed Hierarchy的反例

将inter-view改为global→group→frame,intra-view改为coarse→fine:
- PSNR: 18.95 (相比baseline的22.79,跌3.84dB)
- LPIPS: 0.555 (恶化严重)

这反向验证了设计: **先global再local,先coarse再fine都不work**,因为fine-grained local information丢失后无法恢复。

### 5.4 Group Size选择 (Table 11)

| Group Size | Novel view PSNR | Spatial cognition (32 views) |
|------------|-----------------|------------------------------|
| 2 | 22.94 | 83.6% |
| 4 (baseline) | 23.18 | 96.4% |
| 8 | 23.18 | 97.1% |

Group=2 context太少,Group=8 marginal gain但cost高。Group=4是sweet spot。spatial cognition任务([PRoPE](https://arxiv.org/abs/2503.04652)提出的)的差距更明显,说明group size对multi-view awareness影响显著。

## 6. Point Map Estimation (Table 12)

在NRGBD和ETH3D上评估几何精度(虽然只用photometric loss训练):

| Method | Views | NRGBD CD↓ | ETH3D CD↓ |
|--------|-------|-----------|-----------|
| Long-LRM | 16 | 0.53 | 2.75 |
| MVP | 16 | 0.18 | 1.74 |
| Long-LRM | 32 | 0.43 | 2.69 |
| MVP | 32 | 0.14 | 2.22 |

MVP在geometric accuracy上也大幅领先,即使baselines额外用[DepthAnything](https://arxiv.org/abs/2401.10891)做regularization。这说明architecture design本身已经induces了更好的3D geometry understanding。

## 7. Training策略细节

Three-stage training schedule非常关键:

**Stage 1** (low-res warmup):
- 480×256, 32 input + 12 target views
- 100k iter, lr=2e-4
- batch size 256 (32 GPUs × 8)
- frame interval均匀采样64-128之间

**Stage 2** (high-res):
- 960×540, 32 input + 6 target views
- 50k iter, lr=2e-5
- batch size 64 (32 GPUs × 2)
- 全frame range采样 + intrinsic augmentation

**Stage 3** (variable views fine-tuning):
- 960×540, variable input views
- 30k iter, lr=2e-5
- **冻结frame-wise和group-wise modules**,只训练global modules

Stage 3的freezing策略很关键: 既保留了stage 1/2学到的local和group representation,又能efficient地fine-tune global模块适应不同view counts。

**其他tricks**:
- PRoPE ([Li et al.](https://arxiv.org/abs/2503.04652))作为relative positional encoding
- Plücker rays + PRoPE (相比Cam rays + PRoPE效果更好)
- FlashAttention3 ([Shah et al.](https://arxiv.org/abs/2407.11062))
- Gradient checkpointing ([Chen et al.](https://arxiv.org/abs/1604.06174))
- bfloat16 mixed precision
- EMA,no gradient clipping
- AdamW,β1=0.9, β2=0.95,weight decay=0.05

## 8. Attention Visualization分析 (Figure 5, 8)

选取reference view上的3个query patches (red, yellow, green),visualize其在other views的top-3 attended tokens。

**Stage 2 (Group-wise)**: attention集中在group内的spatially corresponding区域,蓝色overlay清晰显示local correspondence。

**Stage 3 (Global)**: attention不仅覆盖group内,还能跨group attend到semantically和geometrically consistent的远距离区域,绿色overlay展示了这种long-range correspondence。

这组visualization直观证明了: group-wise attention学会了local geometric correspondence,而global attention在此基础上进一步建立了scene-level的一致性。

## 9. Intuition Building: 为什么这个design work?

### 9.1 Attention Dilution的本质

Standard global attention的问题: 当token数量指数级增长时,softmax attention的distribution会变得"扁平化"。每个query要 attend到所有tokens,但大部分tokens是irrelevant的,attention mass被spread到大量noise tokens上,导致真正important的correspondence被淹没。

MVP的解决方案是: 通过hierarchical design,**让每层attention参与的tokens数量保持bounded**:
- Stage 1: $O(hw)$ per view, frame-wise
- Stage 2: $O(M \cdot hw)$ per group, group-wise  
- Stage 3: $O(N \cdot h'w')$ globally, 但此时$h'w'$已经是coarse resolution

这种"token budget control"让attention始终能focus到relevant tokens。

### 9.2 Fine-to-coarse的生物学启发

Paper标题"Look Coarser to See Broader"其实是对人类视觉系统的隐喻:
- Fovea (central vision)处理fine details,但范围小
- Peripheral vision处理coarse information,但范围广
- 两个系统协同工作,既看细节又看全局

MVP的intra-view hierarchy对应fovea→peripheral的progression,inter-view hierarchy对应visual field的expansion。

### 9.3 与其他hierarchical architectures的关系

- **FPN** ([Lin et al.](https://arxiv.org/abs/1612.03144)): 只在spatial domain做hierarchy
- **Swin Transformer**: shifted window attention,但不是progressive
- **Video Swin** ([Liu et al.](https://arxiv.org/abs/2106.13290)): temporal downsampling,但multi-view setting下不能直接做temporal downsampling
- **VGGT** ([Wang et al.](https://arxiv.org/abs/2502.02066)): alternating attention但没有hierarchical structure

MVP的独特之处: 在view和spatial两个axes上同时做hierarchy,且两者complementary运作。

## 10. Limitations和Future Work

Paper最后提到了几个extensions:
1. **Dynamic scenes**: 当前只做static reconstruction,可扩展到4D ([4D-GS](https://arxiv.org/abs/2403.12742))
2. **Geometry-supervised training**: 当前只用photometric loss,可加入depth/point cloud supervision ([DUSt3R](https://arxiv.org/abs/2312.14132), [VGGT](https://arxiv.org/abs/2502.02066), [π3](https://arxiv.org/abs/2503.06322))
3. **Camera pose estimation**: 当前需要known poses,可扩展到joint pose estimation

## 11. 我的额外思考

### 11.1 为什么group=4是sweet spot?

从Table 11看,group=2时spatial cognition task表现差(83.6% vs 96.4%),说明group太小导致cross-view信息不足。group=8时novel view synthesis没有提升(23.18 vs 23.18),但cost上升。这可能是因为M=4时,一个group刚好覆盖一个小区域(locality),M=8时已经包含太多redundant views。

### 11.2 与Mamba的关系

Long-LRM用Mamba替代attention降低complexity,但paper显示MVP用pure transformer + hierarchy就能beat Mamba-based方法。这暗示: **complexity问题不一定需要更换sequence model架构,通过hierarchical design也能解决**。

### 11.3 关于inference speed

MVP在128 views下0.77s,这个速度已经接近real-time applications。如果结合[gsplat](https://arxiv.org/abs/2503.15114)的fast rendering,可能实现real-time large-scale 3D reconstruction。这对AR/VR应用意义重大。

### 11.4 关于训练cost

32 H100 GPUs训练9天 (4+3+2)。这个规模相当大,但考虑到模型要处理上百views的高分辨率图像,算是reasonable。Stage 3的freezing策略也降低了fine-tuning cost。

### 11.5 与Foundation Models的关系

Paper提到了[DINOv2](https://arxiv.org/abs/2304.07193)和[DINOv3](https://arxiv.org/abs/2508.10104),但作者没有用DINO作为encoder (ablation中提到"w/o Dual Hierarchy"是modified VGGT,用linear patch embedding替代DINO encoder)。这是一个有趣的选择: 是不是hierarchical attention本身已经提供了足够的inductive bias,不需要pretrained features?

### 11.6 关于PRoPE的使用

[PRoPE](https://arxiv.org/abs/2503.04652) (Pose-Relative Positional Encoding)是把camera pose作为relative positional signal注入attention。Paper发现"Plücker rays + PRoPE"比"Cam rays + PRoPE"更好,这暗示extrinsics information比intrinsics更关键(可能因为intrinsics在dataset内variation较小)。

## 12. 总结

MVP的核心贡献是**将经典的CNN/ViT hierarchical design哲学成功移植到multi-view setting**,通过dual axes hierarchy同时解决:
1. Computational scalability (views增加时cost线性而非二次增长)
2. Attention dilution (避免long context下attention退化)
3. Multi-scale representation (fine细节+coarse context)

实验数据very convincing: 在DL3DV, Tanks&Temples, Mip-NeRF360, RE10K四个dataset上全面SOTA,且inference speed比baselines快5-250×。

**关键take-away**: 当面对long context问题时,与其更换sequence model架构(Mamba, linear attention等),不如思考如何通过hierarchical design让standard attention保持expressive power的同时控制computational cost。这是一个orthogonal且可能更general的设计方向。

---

**Reference links**:
- Paper project page: https://gynjn.github.io/MVP/
- [Long-LRM](https://arxiv.org/abs/2503.04652)
- [iLRM](https://arxiv.org/abs/2507.23277)
- [VGGT](https://arxiv.org/abs/2502.02066)
- [DUSt3R](https://arxiv.org/abs/2312.14132)
- [3D Gaussian Splatting](https://arxiv.org/abs/2308.14737)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [DPT (Dense Prediction Transformer)](https://arxiv.org/abs/2103.13413)
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- [PRoPE - Cameras as Relative Positional Encoding](https://arxiv.org/abs/2503.04652)
- [FlashAttention-3](https://arxiv.org/abs/2407.11062)
- [DL3DV-10K Dataset](https://arxiv.org/abs/2403.15037)
- [Depth Anything](https://arxiv.org/abs/2401.10891)
- [Token Merging](https://arxiv.org/abs/2210.09461)
- [DINOv3](https://arxiv.org/abs/2508.10104)
