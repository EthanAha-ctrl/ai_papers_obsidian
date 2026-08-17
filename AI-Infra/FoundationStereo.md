---
source_pdf: FoundationStereo.pdf
paper_sha256: 24891ba61e728544529c95609869964a9fb603d6cfca869cda6c4c9dc9912ba9
processed_at: '2026-08-04T10:13:44-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FoundationStereo

## 一句话总结

NVIDIA 把 stereo matching 这个老任务硬生生推到了 foundation model 时代——你拿一对没见过的真实场景左右图扔进去，它直接吐出靠谱的 disparity map，不用 fine-tuning。Middlebury 误差从之前的 7-13% 直接干到 1%，差不多一个数量级的跳跃。

## 为什么这事以前做不到

Stereo matching 听起来简单——找左右图的对应像素，算 disparity。但实际有几个老大难：

**数据问题**：你要训 network，得有 GT disparity。GT 怎么来？要么 LiDAR 扫（稀疏、贵），要么 structured light（只能 indoor、近距离），要么干脆合成。公开数据 Scene Flow 才 40K 对，KITTI 几百对，real-world dense GT 几乎没有。你看 monocular depth 那边 Depth Anything 直接爬 internet 几百万张图训，stereo 这边根本玩不了这个游戏。

**架构问题**：主流方法分两派。一派是 cost volume + 3D CNN filter，memory 吃得厉害，high resolution 直接爆显存；另一派是 RAFT-Stereo 那种 iterative GRU，省 memory 但靠局部 recurrent update，缺 long-range context。两派在 benchmark 上都很强，但都依赖 fine-tuning，换个 domain 就拉胯。

所以 stereo 一直停在「per-domain SOTA」阶段，离 foundation model 还差得远。

## 他们怎么搞的

三件事一起做：造数据 + 借 prior + 改架构。

### 第一件：造一个真正大的合成数据集

他们用 NVIDIA Omniverse（自家 RTX path-tracing 渲染器）造了 100 万对 stereo image。听起来「合成数据」不新鲜，Scene Flow 也是合成的。但关键区别：

- **以前**：固定 baseline、固定 focal length、固定 camera pose，场景就那几个
- **这次**：baseline 随机、focal length 随机、camera 角度随机、光照随机、物体组合随机

为什么要这么 random？sim-to-real 的核心经验：与其让合成数据「看起来像真实」，不如让合成数据「覆盖足够广的 distribution」，real data 总落在 distribution 里的某个点。

但 random 也有代价——会生成出一些「无解」的样本，比如纯反射物体在弱光下变成纯色块、重复纹理的 flying objects。这些样本没有可学习的对应关系，反而会污染训练。

他们的解法很聪明：**self-curation**。先训一版 model，让 model 在自己训练集上跑，把 BP-2 error 大于 60% 的样本判为「ambiguous」踢掉，重新生成。循环两次。结果 Middlebury zero-shot 从 1.27 降到 1.15。

这个思路本质上类似 RLHF：用 model 的 prior 来 detect data distribution 的 tail，再 fix tail。

### 第二件：借 monocular foundation model 的 prior

这是最关键的一招。

Sim-to-real gap 怎么破？以前的思路是「学习 domain-invariant feature」（DSMNet 之类），效果一般。这篇直接换思路：**我已经有 DepthAnythingV2 这个在 internet-scale 真实图上训过的 monocular depth foundation model，它的 feature 里全是 real-world 的 semantic 和 geometric prior，我直接拿来用不就完了**。

但有几个坑要避开：
- 不能直接用 monocular depth 当 GT——monocular depth 是 relative scale，stereo 是 metric
- 不能 fine-tune ViT——catastrophic forgetting，几步就把 prior 毁了
- 不能只用 ViT feature 不要 CNN feature——ViT 缺高频细节

他们的解法叫 **STA (Side-Tuning Adapter)**，简单到有点意外：
1. 把 DepthAnythingV2 freeze 住，一根毛都不动
2. 旁边加一个轻量 CNN（EdgeNeXt-S）提 multi-scale feature
3. 在 1/4 scale 把 ViT 的中间 feature（注意是 output head **之前** 的 feature，不是最终 depth）和 CNN feature concat 起来
4. 这个 concat 后的 hybrid feature 作为 stereo 网络的 unary input

为什么这么简单的设计比 ViT-Adapter 那种复杂双向交互还好？因为 foundation model 的 prior 是宝贵的，最稳定的 transfer 方式是「additive」而非「distortive」——你给它加个 side branch 用 stereo signal 训，让它学怎么用 ViT 的 prior，但不动 ViT 本身。这个 pattern 在 LoRA、linear probe 里都见过，这里是 stereo matching 的具体应用。

Ablation 很说明问题：freeze ViT 1.97，unfreeze 掉到 3.94，差一倍。

### 第三件：把 cost volume filtering 做对

有了好的 unary feature，还得构造 cost volume 然后过滤。这一步以前被 transformer 时代忽略了——大家把 transformer 用在 unary feature extraction（CroCo 那种），cost filtering 还是老 3D CNN。

他们发现 cost volume filtering 其实是个被低估的环节，做了两件事：

**APC (Axial-Planar Convolution)**：3D conv 3×3×3 在 disparity 维度感受野只有 3，但 disparity 概率分布有时需要看 17+ 的范围才能定 peak。直接换 5×5×5 在 80GB GPU 上 OOM。他们把 3D conv 拆成两个 axial conv：一个做 spatial 平面（$K_s \times K_s \times 1$），一个专门做 disparity 维（$1 \times 1 \times K_d$）。disparity kernel 设到 17 最优。

这个拆解的 intuition：cost volume 里 spatial 维和 disparity 维语义完全不同。spatial 是图像平面上邻域的平滑性，disparity 是「这一列里哪个 d 是真正的 peak」。分开处理比耦合在一起更合理。

**Disparity Transformer (DT)**：APC 的 17 还是 local。条纹衬衫、thin structure 这种场景需要 global context。他们加了一个 transformer，但有个反直觉的发现：**只在 disparity 维度做 self-attention 比对整个 4D volume 做 attention 好**（1.97 vs 2.25）。

为什么？因为 cost volume 的 4 个维度 $(C, D, H, W)$ 里，spatial $(H, W)$ 的 smoothing 已经被 APC hourglass 做了，channel $(C)$ attention 价值有限，真正稀缺的是沿 disparity 维的全局 reasoning——「这一列所有 d 的 cost，哪个是真的 peak」。在 4D token 序列上做 full attention 引入过多无意义 token，优化反而难。

## 整个 pipeline 跑起来什么样

输入左右图 →

STA 提取 unary feature（CNN + frozen DepthAnythingV2 的 hybrid feature） →

构造 hybrid cost volume（group-wise correlation + concat 保留 monocular prior） →

AHCF 过滤：APC hourglass 和 Disparity Transformer 并行跑，输出相加 →

soft-argmin 出初始 disparity →

GRU 迭代 32 步 refine，每步同时 lookup hybrid cost volume 和 correlation volume →

输出 full resolution disparity

训练 loss 就是 initial disparity 的 smooth L1 + 各 iteration 的 L1，后期 iteration 权重指数增大（$\gamma=0.9$），让模型重点优化最终输出。

## 结果有多强

Middlebury zero-shot BP-2：之前 SOTA 大约 7-8%，他们 1.1%，差不多 7× 提升。

ETH3D 更夸张：他们 zero-shot 推理 0.5 BP-1，比人家 fine-tuned 的 SOTA（Selective-IGEV 1.23）还低。换句话说，他们不 fine-tune 就比所有 fine-tune 的方法强。

Booster（透明反射物体）：EPE 2.2 vs IGEV 22.7，10× 差距。monocular prior 在这种 correspondence 完全 fail 的场景下发挥了关键作用。

Limitation 也明显：375×1242 上 0.7 秒一张，比 IGEV 慢 3 倍。作者自己说未来需要 distillation/pruning。

## 我的几个 takeaway

**Foundation model prior 是 sim-to-real 的新银弹**。以前 sim-to-real 的方法是「学习不变特征」，效果天花板有限。现在直接 leverage 已有的 foundation model，把它的 real-world knowledge 借过来用，效果直接飞跃。STA 这种 freeze + side branch 模式应该可以推广到 optical flow、cross-view matching 等所有 dense correspondence 任务。

**Cost volume filtering 没过时**。RAFT 那种纯 iterative 方法流行之后，cost volume filtering 显得有点老派。但这篇证明，把 cost volume filtering 做对（APC + DT），加上后续 GRU refine，两者结合比纯 iterative 强很多。

**Self-curation 是合成数据时代必备**。数据可以无限造，但 random 会引入 ambiguity。用 model 自己 detect 并 fix tail，这个 loop 值得标配化。

**Foundation model 时代的 stereo 终于来了**。之前 stereo matching 一直停留在 per-domain SOTA 竞赛，离 foundation model 还远。这篇应该是第一个真正能 off-the-shelf 用的 stereo foundation model。

---

# FoundationStereo 技术深度讲解

## 1. 核心 motivation 与定位

Stereo matching 是 CV 经典任务，给定一对左右 calibrated images，目标是 dense pixel-level correspondence → disparity map。Past 5 年深度学习方法在 benchmark 上几乎 saturate，但代价是 per-domain fine-tuning。其他 vision tasks（CLIP, SAM, Depth Anything）已经有 foundation model 级别的 zero-shot 能力，stereo matching 却迟迟没跟上。这篇 NVIDIA 的工作就是把 stereo matching 推到 foundation model 时代。

核心难点来自两边的矛盾：
- **数据侧**：Scene Flow 只有 40K 对，Sintel 1K 对，real-world dataset 稀疏且少。stereo 与 monocular depth 不同，stereo GT 是需要 LiDAR 或 structured light 这种昂贵设备采集的，internet-scale 真实数据基本不可能。
- **架构侧**：cost volume + 3D CNN 难 scale，iterative GRU 缺 long-range context，纯 transformer unary feature 又丢失 stereo 的归纳偏置。

FoundationStereo 的策略 = 合成数据扩张（1M pairs）+ Vision foundation model priors 注入 + Hybrid cost volume 架构 + Self-curation 数据清洗。

项目主页：https://nvlabs.github.io/FoundationStereo/

---

## 2. FSD 数据集

### 2.1 数据规模对比

| Dataset | Stereo Pairs | Scenarios | Sim | Resolution |
|---|---|---|---|---|
| Sintel | ~1K | Movie | Blender | 1024×436 |
| Scene Flow | 40K | Flying/Driving/Monkaa | Blender | 960×540 |
| CREStereo | 200K | Flying | Blender | 1920×1080 |
| TartanAir | 306K† | Outdoor | UE | 640×480 |
| Spring | 6K† | Movie | Blender | 1920×1080 |
| **FSD (Ours)** | **1000K** | Flying/Indoor/Outdoor/Driving/Movie | Omniverse | 1280×720 |

FSD 是迄今最大规模的 stereo 训练集，且 baseline、focal length、camera pose 都做了 randomization（其他 dataset 都是 constant）。

### 2.2 数据生成 pipeline

- **渲染引擎**：NVIDIA Omniverse，RTX path-tracing，32–128 SPP（samples per pixel）
- **资产**：5K+ 3D assets，包含 artist design + 3D scanning；12 个大型场景模型（factory/hospital/office/grocery/warehouse 等）；150+ materials；400+ textures（real-world photos + procedural patterns）
- **硬件**：48 块 NVIDIA A40 GPU × 10 天
- **Domain randomization**：
  - 随机 stereo baseline（其他 dataset 通常 constant）
  - 随机 focal length → 覆盖不同 FOV 与 disparity distribution
  - 随机 camera perspective（平行/俯视/俯瞰）
  - 随机 lighting（global illumination + directed sky rays + 烘焙 lights + light spheres，颜色/强度/方向都 randomize）
  - 随机 object placement（两种 mode：camera-first spawn 物体；object-first spawn 相机）

### 2.3 场景类型划分

realistic-style 数据下分 3 个 sub-type：
- **Navigation**：camera 平行地面，物体较远，free-standing walls/furniture/digital humans 概率高
- **Driving**：camera 平行地面高处，vehicles/poles/signs/speed bumps 概率高
- **Manipulation**：camera 朝前/朝下（ego-centric），物体近距离，household items/containers/robotic arms 概率高

混合 chaotic（flying distractors + skybox）与 realistic 两种 layout，这是 sim-to-real 论文 [62] 中证明有效的策略。

### 2.4 Iterative Self-Curation

合成数据生成在 flying objects 场景会引入 ambiguity，比如：
- 严重纹理重复
- 反射场景但周围 context 不足
- 不当 lighting 下纯色物体

策略：
1. 用 FSD 训练初始版本 FoundationStereo
2. 在 FSD 上 evaluate，BP-2 > 60% 的 sample 判为 ambiguous
3. 替换为重新生成的新样本
4. 训练与 curation 交替进行（做 2 次）

**效果**：Table 8 显示，with self-curation BP-2 = 1.15，without self-curation BP-2 = 1.27。注意这里 evaluation 是 Middlebury，但 curation 是在合成数据上做的，说明 curation 不只是「去掉难样本」而是去掉了真正有 ambiguity 的样本，否则在 training 上去掉「难样本」通常 zero-shot 性能下降。

### 2.5 FSD 对其他方法也有帮助

Table 9 显示，把 IGEV 和 Selective-IGEV 从 Scene Flow 训练换到 FSD 训练：
- IGEV Middlebury 8.8 → 7.8
- IGEV KITTI-12 5.2 → 3.2
- Selective-IGEV Middlebury 9.2 → 7.9
- Selective-IGEV KITTI-12 4.5 → 3.0

这说明 FSD 的优势 transferable 到其他架构，不只是 FoundationStereo 的特殊得益。

---

## 3. 网络架构

整体 pipeline：
1. **STA** 提取 unary features（含 monocular priors）
2. **Hybrid Cost Volume** 构造
3. **AHCF** = APC Hourglass + DT 联合 filter
4. Soft-argmin → 初始 disparity
5. **Iterative GRU refinement**

### 3.1 STA (Side-Tuning Adapter) — 解决 sim-to-real gap

#### Motivation

训练数据基本是合成的（FSD + Scene Flow + Sintel + CREStereo + FallingThings + InStereo2K + Virtual KITTI 2），sim-to-real gap 是 zero-shot 泛化的最大障碍。但 DepthAnythingV2 在 internet-scale 真实 monocular image 上训练过，蕴含丰富的 semantic + geometric prior。直接把 monocular depth 当 prior 用有 scale ambiguity 问题（monocular depth 是 relative 的，stereo 是 metric 的）。

解法：用 DepthAnythingV2 的 latent feature 作为 prior，通过 cost volume filtering 在 stereo pair 上做 explicit comparison。

#### 设计选择对比（Fig 3 left）

**(a) DPT head only**：直接用 frozen DepthAnythingV2 DPT head 的 feature pyramids
**(b) ViT-Adapter style**：CNN 与 ViT 双向 feature exchange
**(c) Side-tuning**：
- 用 4×4 stride-4 conv downscale 输入到 DepthAnythingV2 final output head 之前
- 把这个 ViT feature 与同 level CNN feature concat → 1/4 scale hybrid feature
- side CNN 学习把 ViT features adapt 到 stereo task

#### 为什么 (c) 最好

**Intuition**：(a) 丢失了 CNN 的高频细节，(b) feature exchange 引入了复杂的双向耦合容易不稳定，(c) 是最简单的「加性」side network。Side-Tuning 是 Zhang et al. ECCV 2020 [83] 提出的 baseline pattern，假设是大 model 已经学好了 powerful representations，adaptation 应该是「additive」而非「distortive」。

(c) 的另一个关键点是取 DepthAnythingV2 final output head **之前** 的 feature。Table 5 中 (c) BP-2 = 1.97，(a) 6.48，(b) 2.22。最后的 1×1 head 把高维 features 压成单 channel depth，丢失了大部分语义信息，所以必须取 head 前的 feature。

#### 公式化表达

给定左右图像 $I_l, I_r \in \mathbb{R}^{H \times W \times 3}$：

STA 输出 4 个 level 的 pyramid features：
$$f_l^{(i)}, f_r^{(i)} \in \mathbb{R}^{C_i \times \frac{H}{i} \times \frac{W}{i}}, \quad i \in \{4, 8, 16, 32\}$$

CNN backbone 用 EdgeNeXt-S（比更大的 CNN backbone 没有额外收益，但 memory 效率高）。

DepthAnythingV2 输入前 resize 到 14 的倍数（patch size=14）。

STA weights 在 $I_l, I_r$ 之间共享（参数对称性）。

Context feature 也是用 STA 但 CNN 部分换成 residual blocks + down-sampling：
$$f_c^{(i)} \in \mathbb{R}^{C_i \times \frac{H}{i} \times \frac{W}{i}}, \quad i \in \{4, 8, 16\}$$

#### Ablation 关键点

| 选项 | BP-2 |
|---|---|
| DINOv2-L | 2.46 |
| DepthAnythingV2-S | 2.22 |
| DepthAnythingV2-B | 2.11 |
| **DepthAnythingV2-L** | **1.97** |
| Unfreeze ViT | 3.94 |
| **Freeze ViT** | **1.97** |

**Freeze ViT 至关重要**：unfreeze 会让 ViT priors 被 stereo task-specific signal 腐蚀（catastrophic forgetting），性能从 1.97 掉到 3.94。这与 LoRA、linear probe 等 parameter-efficient adaptation 的发现一致——freeze foundation model + train small adapter 是最稳定的 transfer pattern。

DINOv2-L 比 DepthAnythingV2-S 还差（2.46 vs 2.22），尽管 DINOv2 在 correspondence matching 任务上 [19] 表现好。原因：DINOv2 是 self-supervised 通用特征，与 depth task 相关性弱，分辨率有限难以做 pixel-level 高精度匹配。

参考：DepthAnythingV2 https://depth-anything-v2.github.io/

---

### 3.2 Hybrid Cost Volume Construction (Eq.1)

给定 1/4 scale unary features $f_l^{(4)}, f_r^{(4)}$，cost volume $V_C \in \mathbb{R}^{C \times \frac{D}{4} \times \frac{H}{4} \times \frac{W}{4}}$，包含两部分：

#### Group-wise Correlation $V_{gwc}$

$$V_{gwc}(g, d, h, w) = \langle \widehat{f}_{l,g}^{(4)}(h, w), \widehat{f}_{r,g}^{(4)}(h, w-d) \rangle$$

- $\widehat{f}$ 表示 L2-normalized feature（训练稳定性）
- $g \in \{1, 2, ..., G\}$，G=8（feature 维度均匀分 8 组）
- $d \in \{1, 2, ..., \frac{D}{4}\}$ disparity index
- $\langle \cdot, \cdot \rangle$ dot product

Group-wise correlation 思想来自 GwcNet [24]：分 group 做点积而非整 channel 点积，得到 G 个 similarity score，比 scalar 更 expressive。

#### Concatenation $V_{cat}$

$$V_{cat}(d, h, w) = [\text{Conv}(f_l^{(4)})(h, w), \text{Conv}(f_r^{(4)})(h, w-d)]$$

- 1×1 Conv 把 channel 减到 14 节省 memory
- $f_l^{(4)}, f_r^{(4)}$ 共享 conv weights

#### 合并

$$V_C(d, h, w) = [V_{gwc}(d, h, w), V_{cat}(d, h, w)]$$

$V_{cat}$ 的关键作用是 **保留 monocular priors**：concatenation 让左右特征共存于 volume，cost filtering 时 ViT priors 可以被聚合使用。如果只做 correlation 会丢失大量 unary 信息。

---

### 3.3 APC (Axial-Planar Convolution)

#### 问题

标准 3D CNN kernel $3 \times 3 \times 3$ 在 cost volume 上 receptive field 小。当训练 high resolution image 时 disparity 范围大（416 pixels），3×3×3 kernel 在 disparity 维度的感受野只有 3，难以捕捉 disparity distribution 的 context。试 5×5×5 在 80GB GPU 上 OOM。

#### 解法：解耦 3D conv 为两个 axial convs

$$\text{APC}(V) = \text{Conv}_{1 \times 1 \times K_d}(\text{Conv}_{K_s \times K_s \times 1}(V))$$

- Spatial branch：$K_s \times K_s \times 1$（在 H-W 平面）
- Disparity branch：$1 \times 1 \times K_d$（沿 D 维）
- 每个 conv 后跟 BatchNorm + ReLU

**Intuition**：这相当于 3D 版本的 Separable Conv [16]。但与 depthwise separable 不同，**不分组 channel**（保留 channel 间交互）。只分 spatial 和 disparity 两个维度，因为这两个维度语义不同：
- spatial 维：图像 plane 上相邻 pixel 是同一物体或邻近物体的 cost，应该平滑
- disparity 维：cost volume 在 d 上是一维概率分布，需要 long-range context 判断哪个 d 是真正 peak

#### 复杂度分析

3D conv 3×3×3 on $C \times D \times H \times W$ 复杂度 $\approx 27 \cdot C^2 \cdot D \cdot H \cdot W$。

APC 复杂度 $\approx (9 + K_d^2) \cdot C^2 \cdot D \cdot H \cdot W$。

$K_d = 17$ 时（最优）大约 $9 + 17 = 26$，与 3×3×3 的 27 相当，但 disparity 维度的 receptive field 从 3 → 17，提升巨大。

#### Ablation

| APC kernel | BP-2 |
|---|---|
| (3,3,1), (1,1,5) | 2.10 |
| (3,3,1), (1,1,9) | 2.06 |
| (3,3,1), (1,1,13) | 2.01 |
| **(3,3,1), (1,1,17)** | **1.97** |
| (3,3,1), (1,1,21) | 1.98 |
| (7,7,1), (1,1,17) | 1.99 |

$K_d = 17$ 达到饱和，再增大没有收益；spatial $K_s = 7$ 反而稍差（可能因为 cost volume 的空间 smoothing 已经被后续的 hourglass 下采样做了）。

#### 实现细节

在 hourglass network 里（3 个 down block + 3 个 up block + residual connections），**除 down/up sampling 层外都用 APC**。

---

### 3.4 Disparity Transformer (DT)

#### Motivation

APC 的 disparity kernel 即便 17，也是 local。stereo matching 的一些场景（thin structures、repetitive texture）需要 global disparity context。比如条纹衬衫，local context 都长得一样，必须看全 disparity 范围才能确定正确 disparity。

之前 transformer-based stereo 工作 [35, 68] 只把 transformer 用在 unary feature extraction，cost filtering 还是 CNN。这里把 transformer 引入到 4D cost volume filtering。

#### 公式

输入 $V_C$，先 3D conv 4×4×4 stride 4 downsize，reshape 成 token sequences：

$$Q_0 = \text{PE}(\text{R}(\text{Conv}_{4 \times 4 \times 4}(V_C))) \in \mathbb{R}^{(\frac{H}{16} \times \frac{W}{16}) \times C \times \frac{D}{16}}$$

- $R(\cdot)$ reshape：把 spatial 维 $(\frac{H}{16} \times \frac{W}{16})$ 展开成 batch，每个 sample 是一个长度 $\frac{D}{16}$ 的 token sequence
- $PE(\cdot)$ position encoding：cosine 最好（RoPE 反而差）

Multi-head self-attention（h=4 heads）：

$$\text{MultiHead}(Q, K, V) = [\text{head}_1, ..., \text{head}_h] W_O$$
$$\text{head}_i = \text{FlashAttention}(Q_i, K_i, V_i)$$

FlashAttention [18] 是 IO-aware 的 exact attention，避免 materializing attention matrix，节省 memory。

Transformer encoder block：

$$Q_1 = \text{Norm}(\text{MultiHead}(Q_0, Q_0, Q_0) + Q_0)$$
$$Q_2 = \text{Norm}(\text{FFN}(Q_1) + Q_1)$$

4 个 transformer encoder blocks stacked。

最后 trilinear upsample 回 $V_C$ 大小，与 hourglass output 相加。

#### Ablation 关键设计

| 选项 | BP-2 |
|---|---|
| RoPE position encoding | 2.19 |
| **Cosine position encoding** | **1.97** |
| 1/32 scale | 2.06 |
| **1/16 scale** | **1.97** |
| Full volume attention | 2.25 |
| **Disparity-only attention** | **1.97** |
| Pre-hourglass | 2.06 |
| Post-hourglass | 2.20 |
| **Parallel** | **1.97** |

**重要 finding**：full 4D attention 比 disparity-only attention 差（2.25 vs 1.97）。

**Intuition**：cost volume 的 4 个维度 $(C, D, H, W)$ 中，$(H, W)$ 是空间，沿空间 attention 等价于在 cost volume 上做 spatial smoothing，但 APC hourglass 已经做了；$(C)$ 沿 channel attention 类似 SE block，相对弱；$(D)$ 沿 disparity attention 是直接看 disparity 概率分布的全局，这是真正稀缺的能力。对 4D token 序列做 self-attention 还会引入过多 token（$D/16 \times H/16 \times W/16$ 个 token），在 4D volume 上做 self-attention 信息复杂度高、优化难。

**RoPE 不如 cosine**：因为 disparity size 是固定的（4D cost volume 大小固定），RoPE 的相对位置优势在固定长度场景不显著，反而增加训练难度。

**DT 与 hourglass 平行最好**：pre-hourglass 让 transformer 输出再经过 CNN smoothing 可能丢信息；post-hourglass 让 transformer 接收已经被 CNN 过滤的 cost volume，限制 transformer 的能力；parallel 让两者各展所长。

参考：FlashAttention https://arxiv.org/abs/2205.14135

---

### 3.5 Initial Disparity Prediction (Eq.2)

soft-argmin [30]：

$$d_0 = \sum_{d=0}^{\frac{D}{4}-1} d \cdot \text{Softmax}(V_C')(d)$$

- $V_C'$ 是 AHCF filtered cost volume
- Softmax 在 disparity 维度上 → 概率分布
- 期望值作为初始 disparity

soft-argmin 可微，相比 hard argmin 允许梯度 backprop 到 cost volume。但 soft-argmin 在 multi-modal distribution 上会过度平滑（取 peak 之间的均值），所以需要后续 GRU refinement。

输出 $d_0$ 在 1/4 scale。

---

### 3.6 Iterative GRU Refinement (Eq.3-10)

基于 RAFT [57] / RAFT-Stereo [36] 框架。

#### 第 k 步输入

**Correlation volume lookup**（pair-wise correlation，与 hybrid cost volume 不同）：

$$V_{corr}(w', h, w) = \langle f_l^{(4)}(h, w), f_r^{(4)}(h, w') \rangle$$

注意 $V_{corr} \in \mathbb{R}^{\frac{W}{4} \times \frac{H}{4} \times \frac{W}{4}}$，是 4D 全 correlation（不是 hybrid volume 那种 disparity index 限制的）。这个全 correlation 提供 GRU refinement 时任意 w' 的相似度信息。

**Volume feature lookup**：

$$F_V(h, w) = [V_C'(d_k, h, w), V_{corr}(w - d_k, h, w)]$$

用当前 disparity $d_k$ 同时从 hybrid cost volume 和全 correlation volume 取特征。这是 FoundationStereo 相对纯 RAFT-Stereo 的关键不同——RAFT-Stereo 只有 correlation volume lookup，FoundationStereo 多了 filtered hybrid cost volume 的 prior。

**GRU input**：

$$x_k = [\text{Conv}_v(F_V), \text{Conv}_d(d_k), d_k, c]$$

- $\text{Conv}_v(F_V)$: volume feature projection
- $\text{Conv}_d(d_k)$: 当前 disparity projection（让 GRU 知道当前在哪）
- $d_k$: 直接拼接 current disparity
- $c = \text{ReLU}(f_c)$: context feature（含 STA-adapted monocular prior），guides refinement

**GRU update**：

$$z_k = \sigma(\text{Conv}_z([h_{k-1}, x_k])) \quad \text{(update gate)}$$
$$r_k = \sigma(\text{Conv}_r([h_{k-1}, x_k])) \quad \text{(reset gate)}$$
$$\hat{h}_k = \tanh(\text{Conv}_h([r_k \odot h_{k-1}, x_k])) \quad \text{(candidate)}$$
$$h_k = (1 - z_k) \odot h_{k-1} + z_k \odot \hat{h}_k \quad \text{(new hidden state)}$$

- $\sigma$ sigmoid, $\odot$ element-wise product
- ConvGRU 是标准 GRU 的 conv 版本（参数在 spatial 上共享）

**Disparity update**：

$$d_{k+1} = d_k + \text{Conv}_\Delta(h_k)$$

residual update pattern。

#### 多尺度 GRU

3 个 level coarse-to-fine（1/4, 1/8, 1/16），每个 level 1 个 GRU block。

$$h_0^{(i)} = \tanh(f_c^{(i)}), \quad i \in \{4, 8, 16\}$$

context feature 初始化 hidden state。

每个 level 用 attention-based selection mechanism [67] 来捕获不同频率信息。

**Convex upsampling** [57] 把 1/4 scale disparity upsample 到 full resolution。

#### 训练 vs 推理

- 训练 22 iterations
- 推理 32 iterations（更多 iteration 更精确，但推理时间增加）

参考：RAFT-Stereo https://github.com/princeton-vl/RAFT-Stereo

---

### 3.7 Loss Function (Eq.11)

$$\mathcal{L} = |d_0 - \bar{d}|_{smooth} + \sum_{k=1}^{K} \gamma^{K-k} \|d_k - \bar{d}\|_1$$

- $\bar{d}$ ground-truth disparity
- $|\cdot|_{smooth}$ smooth L1 (在 initial disparity $d_0$ 上)
- $k$ iteration index
- $\gamma = 0.9$ exponential increasing weight
- 后期 iteration 权重更大（$\gamma^{K-k}$ 当 k 接近 K 时接近 1）

这种 exponentially increasing weight 来自 RAFT [36]：让模型重点优化最终输出，但前期 iteration 也得到监督信号，避免 divergence。

---

## 4. 训练 details

- **Framework**: PyTorch
- **Optimizer**: AdamW [39]
- **Iterations**: 200K steps
- **Batch size**: 128（32 NVIDIA A100 GPUs 均分）
- **Learning rate**: 1e-4, decay ×0.1 at 80% training
- **Crop size**: 320×736
- **Augmentation**: similar to RAFT-Stereo [36]
- **Training GRU iterations**: 22
- **Inference GRU iterations**: 32
- **Max disparity**: 416
- **Train datasets 混合**：FSD + Scene Flow + Sintel + CREStereo + FallingThings + InStereo2K + Virtual KITTI 2

---

## 5. 实验结果

### 5.1 Zero-shot Generalization (Table 2)

| Method | Train | Middlebury BP-2 ↓ | ETH3D BP-1 ↓ | KITTI-12 D1 ↓ | KITTI-15 D1 ↓ |
|---|---|---|---|---|---|
| CREStereo++ | Scene Flow | 14.8 | 4.4 | 4.7 | 5.2 |
| RAFT-Stereo | Scene Flow | 12.6 | 3.3 | 4.7 | 5.5 |
| IGEV | Scene Flow | 8.8 | 4.0 | 5.2 | 5.7 |
| IGEV++ | mixed | 7.8 | 4.1 | 5.1 | 5.9 |
| Selective-IGEV* | mixed | 7.5 | 3.4 | 3.2 | 4.5 |
| **Ours (Scene Flow only)** | Scene Flow | 5.5 | 1.8 | 3.2 | 4.9 |
| **Ours** | mixed | **1.1** | **0.5** | **2.3** | **2.8** |

**重要观察**：
1. 即使只在 Scene Flow 训练（同样训练数据条件），FoundationStereo 已经超越所有方法 → 说明 architecture 起作用
2. 充分训练后 Middlebury BP-2 从次优 7.5 降到 1.1（7× 减少），这是数量级提升
3. ETH3D BP-1 从 3.0 → 0.5

### 5.2 Scene Flow In-Domain (Table 3)

| Method | EPE ↓ |
|---|---|
| LEAStereo | 0.78 |
| GANet | 0.84 |
| IGEV-Stereo | 0.47 |
| MoCha-Stereo | 0.41 |
| Selective-IGEV | 0.44 |
| **Ours** | **0.33** |

In-domain 不是这篇工作焦点，但仍 20% lead。说明架构本身更强。

### 5.3 ETH3D Leaderboard (Table 4)

| Method | Fine-tuned | BP-0.5 ↓ | BP-1.0 ↓ | EPE ↓ |
|---|---|---|---|---|
| Selective-IGEV | ✓ | 3.06 | 1.23 | 0.12 |
| CroCo-Stereo | ✓ | 3.27 | 0.99 | 0.14 |
| MoCha-Stereo | ✓ | 3.20 | 1.41 | 0.13 |
| CREStereo | ✓ | 3.58 | 0.98 | 0.13 |
| HITNet | ✓ | 7.83 | 2.79 | 0.20 |
| **Ours (fine-tuned)** | ✓ | **1.26** | **0.26** | **0.09** |
| **Ours (zero-shot)** | ✗ | 2.31 | 1.52 | 0.13 |

**Ours zero-shot 性能与 fine-tuned SOTA 持平甚至更好**：零样本 BP-0.5 2.31，已经超过 fine-tuned 的 MoCha-Stereo（3.20）、CREStereo（3.58）、Selective-IGEV（3.06）。

Fine-tuned 版本排名第一（提交时）。

### 5.4 Translucent Objects (Booster, Table in Sec.9)

| Method | BP-1 ↓ | BP-2 ↓ | BP-3 ↓ | EPE ↓ |
|---|---|---|---|---|
| Selective-IGEV | 23.8 | 15.0 | 12.0 | 6.6 |
| IGEV | 30.8 | 22.3 | 19.0 | 22.7 |
| **Ours** | **19.0** | **9.6** | **6.7** | **2.2** |

透明/反射物体大幅领先，monocular prior 起到关键作用。

### 5.5 Module Effects (Table 7 left)

| STA | APC | DT | BP-2 |
|---|---|---|---|
| ✗ | ✗ | ✗ | 2.48 |
| ✓ | ✗ | ✗ | 2.21 |
| ✓ | ✓ | ✗ | 2.16 |
| ✓ | ✗ | ✓ | 2.05 |
| ✓ | ✓ | ✓ | **1.97** |

每个 module 都贡献增益。STA 贡献最大（2.48 → 2.21），说明 monocular prior 是 zero-shot 泛化最关键的。DT + APC 都增益时收益叠加（2.21 → 1.97）。

### 5.6 FSD Effects (Table 7 right)

| FSD | BP-2 |
|---|---|
| ✗ | 2.34 |
| ✓ | **1.15** |

注意 Table 7 right 用的是更小 training subset（100K from FSD），所以数字和左半边不同。**包含 FSD 训练的模型 Middlebury BP-2 从 2.34 → 1.15**，差异巨大。这验证了大规模合成数据的核心价值。

---

## 6. 速度与 memory (Sec.10)

Middlebury 不同分辨率：

| Resolution | BP-2 | Peak Mem (G) | Time (s) |
|---|---|---|---|
| Full (ours) | 4.8 | 18.5 | 8.14 |
| Half (ours) | 1.1 | 10.5 | 2.97 |
| Quarter (ours) | 1.3 | 2.3 | 0.55 |
| Selective-IGEV Full | 12.9 | 6.9 | 2.52 |
| IGEV++ Full | 12.7 | 13.1 | 2.12 |
| NMRF Full | 35.3 | 8.1 | 0.95 |

FoundationStereo 比 Selective-IGEV 在 Full 分辨率上慢 3.2×，但 Middlebury BP-2 减少 2.7×。Trade-off 明显，速度是当前 limitation。

Full 分辨率时 peak memory 在 DT module（attention 在大 cost volume 上贵）；half/quarter 分辨率时 peak memory 在 STA module（ViT forward）。

Limitations：375×1242 上 0.7s on A100，未来需要 distillation/pruning 优化（参考 FastSAM [87] 对 SAM 的压缩）。

---

## 7. 我的整体 intuition 总结

这篇 work 给我几个 takeaway：

**1. Foundation model priors 是 sim-to-real 的银弹**

之前 stereo matching 的 sim-to-real 工作大多尝试学习 domain-invariant features（DSMNet, Mask-CFNet），效果有限。这篇直接 leverage 已经在 internet-scale 真实数据上训练好的 monocular foundation model（DepthAnythingV2），通过 side-tuning 把 priors 「借」过来。这个 trick 比 domain generalization 方法有效得多，因为 monocular prior 包含的 semantic+geometric 信息远超 stereo model 能从 40K Scene Flow 学到的。**STA (c) 这种 side-tuning 模式（freeze ViT + additive CNN adapter）应该是一个 general recipe**，可以推广到 optical flow、cross-view matching 等其他 dense correspondence 任务。

**2. Cost volume filtering 是被低估的环节**

之前 transformer-based stereo 把 transformer 用在 unary feature（如 CroCo [68]），cost filtering 还是 CNN。这篇发现 cost filtering 这个 step 也能 benefit from transformer。但注意：不是 full volume attention，而是只沿 disparity 维度 attention。这暗示 stereo matching 的「long-range context」实际是 disparity 上的 long-range，不是空间上的（空间平滑 CNN 已经做得很好）。

**3. APC 是 elegant 工程设计**

3D conv 5×5×5 在 80GB GPU 上 OOM，但 3D conv 3×3×3 受限 disparity kernel=3。把 3D conv 分解成 (spatial 2D conv) + (disparity 1D conv) 既保留表达力（不分组 channel）又显著降低 memory。这是 insight-driven engineering，比简单换 kernel size 更聪明。**这个分解可以推广到所有 cost-volume filtering task**。

**4. Self-curation 是 synthetic data 必备**

合成数据规模可以无限扩，但 domain randomization 会引入 ambiguity。比如 reflective surface 在光照不够时变成纯色，flying objects 重复 texture。Self-curation 思路：用训出来的 model 自己 detect bad samples（BP-2 > 60%），重新生成。这个 loop 是 self-improving 的。本质上是用 model 的 prior 来 detect data distribution 的 tail，再 fix tail。这是 RLHF 之类方法的简化版在数据生成上的应用。

**5. 训练数据组合**：FSD + 6 个公开数据集，混合训练。FSD 是主体（1M pairs），其他作为 diversity 补充。这暗示 synthetic data 还需要公开数据「调味」，因为 FSD 即便 1M pairs 仍有 distribution gap 与 real-world。

参考：

- 项目主页：https://nvlabs.github.io/FoundationStereo/
- DepthAnythingV2：https://depth-anything-v2.github.io/
- RAFT-Stereo：https://github.com/princeton-vl/RAFT-Stereo
- IGEV-Stereo：https://github.com/gangweix/IGEV-Stereo
- NVIDIA Omniverse：https://www.nvidia.com/en-us/omniverse/
- FlashAttention：https://arxiv.org/abs/2205.14135
- Side-Tuning paper：https://arxiv.org/abs/1912.13503
- EdgeNeXt：https://github.com/mmaaz60/EdgeNeXt
- DINOv2：https://github.com/facebookresearch/dinov2
- GwcNet (Group-wise Correlation)：https://github.com/xy-guo/GwcNet
- CroCo v2：https://github.com/naver/croco
- RAFT (optical flow 原始)：https://arxiv.org/abs/2003.12039
- IGEV++：https://arxiv.org/abs/2409.00638
- SAM（fast distillation 参考）：https://github.com/CASIA-IVA-Lab/FastSAM
- Selective-Stereo：https://github.com/RAGHILALI/Selective-Stereo
