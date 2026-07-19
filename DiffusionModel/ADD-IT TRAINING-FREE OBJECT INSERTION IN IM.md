---
source_pdf: ADD-IT TRAINING-FREE OBJECT INSERTION IN IM.pdf
paper_sha256: ebf5b48d0aaa19aaa02fd23cbb295d8dbf1fa3a54fb79582c9b4a4b434eea004
processed_at: '2026-07-18T02:23:18-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Add-it 论文深度解析

Andrej 你好，这篇 paper 我读得很仔细，因为我觉得它触及了一个非常 fundamental 的问题：**如何激活 pretrained diffusion model 里已经存在、但默认不会被使用的 affordance knowledge**。下面我尽量把直觉、公式和工程细节都铺开讲。

---

## 1. Task 设定：Image Additing

任务形式上很简洁：
- Input: source image $X_{source}$ + text prompt $P_{target}$（描述要 add 什么）
- Output: 一张新图，包含原 scene 的大部分内容，并在合理位置出现新 object

paper 识别出三种 failure mode，这个 taxonomy 非常 useful：

| Failure mode | 含义 | 谁负责评估 |
|---|---|---|
| **Neglect** | object 根本没被加上 | CLIP-based / Grounding-DINO inclusion |
| **Appearance** | 加了但视觉 artifact | CLIP_out, human eval |
| **Affordance** | 加了但位置不对（比如 basket 漂在空中） | 论文新提出的 Affordance Benchmark |

Affordance 这个失败模式特别 tricky，因为现有 automatic metric 基本捕捉不到。InstructPix2Pix 可能生成了一张"看起来对"的图（CLIP_dir 高），但 object 放在物理上不合理的位置。这是 paper 推动 benchmark 的核心动机。

参考：[Emu Edit paper](https://arxiv.org/abs/2311.10089)、[EraseDraw](https://arxiv.org/abs/2404.01598)

---

## 2. 为什么已有方法都不行

### 2.1 Supervised 方法（InstructPix2Pix / MagicBrush / EraseDraw）

这些方法在 (image_before, image_after, instruction) 三元组上 fine-tune。问题在：
- **训练分布窄**：合成数据 bias 严重，比如 EraseDraw 用 inpainting model 生成训练对，inpainting model 本身就有 placement prior
- **Affordance 学不到**：训练目标主要是 reconstruction + text alignment，没有显式 supervision 关于"放在哪里合理"
- 表 1 显示 InstructPix2Pix affordance 只有 0.276，几乎随机

### 2.2 Training-free 方法（Prompt-to-Prompt, SDEdit）

- **Prompt-to-Prompt** ([Hertz et al. 2022](https://arxiv.org/abs/2208.01626))：通过把 source caption 的 attention map inject 到 target caption 来 preserve structure。问题：它假设 source 和 target caption 几乎一致，只差几个 token。但 additing 任务里新 object 的 attention map 在 source 里**根本不存在**，inject 无从下手
- **SDEdit** ([Meng et al. 2022](https://arxiv.org/abs/2108.01073))：给 source image 加部分噪声再 denoise。问题：噪声太低 → 改不动；噪声太高 → 全图重新生成，丢失 source

### 2.3 关键 insight

FLUX / SD3 这种 T2I foundation model **本来就懂 affordance**——它在 billion-scale 数据上学过"狗通常坐地上不漂浮"、"basket 放桌上不放空中"。问题是这些 knowledge 默认激活的 trigger 是 **text prompt + 随机噪声**，而 additing 任务需要的是 **text prompt + 既有 scene structure**。

所以 Add-it 的本质 idea：**改造 attention 机制，让 generation 过程同时被三个 signal 路由**——source image（提供 scene structure）、target prompt（提供要加什么）、target image 自己（提供正在生成的内容）。这就是 paper title 里"weighted extended attention"的含义。

---

## 3. Preliminaries：MM-DiT Attention

这部分很重要，因为整个方法建立在 FLUX 的 attention 结构上。

FLUX（[Black Forest Labs, 2024](https://github.com/black-forest-labs/flux)）基于 SD3 ([Esser et al. 2024](https://arxiv.org/abs/2403.03206)) 的 MM-DiT 架构。两种 block：

**Multi-stream blocks**：text 和 image 用**独立的** projection matrices $W_K, W_V, W_Q$
**Single-stream blocks**：text 和 image **共享** projection matrices

两种 block 都在**拼接后的 token 序列**上做 self-attention。公式 (1)：

$$A = \text{softmax}\left([Q_p, Q_{img}][K_p, K_{img}]^\top / \sqrt{d_k}\right), \quad h = A \cdot [V_p, V_{img}]$$

变量含义：
- $Q_p, K_p, V_p$：text prompt token 的 query/key/value，下标 $p$ = prompt
- $Q_{img}, K_{img}, V_{img}$：image patch token 的 query/key/value
- $d_k$：key 维度，用于 scaled dot-product 的 normalization
- $[\cdot, \cdot]$：token 维度上的 concatenation

关键直觉：**text 和 image token 在同一个 attention space 里互相 attend**。这意味着 prompt token 可以"指挥"image token 怎么生成，反之亦然。这是 extended attention 能 work 的物理基础。

---

## 4. Method 核心组件

### 4.1 Weighted Extended Self-Attention

#### 4.1.1 Naive extension 的问题

最直接的扩展：把 source image 的 K, V 也拼进去（公式 2）：

$$A = \text{softmax}\left([Q_p, Q_{target}][K_{source}, K_p, K_{target}]^\top / \sqrt{d_k}\right)$$
$$h = A \cdot [V_{source}, V_p, V_{target}]$$

直觉上这应该 work——target image 既能 attend 到 source image（保 structure），又能 attend 到 prompt（加 object）。

**但实验上失败了**。Paper section 5 给出诊断：source image 的 token 数量远多于 prompt token（image patch 几百上千个 vs text 几十个），softmax 之后 source 主导 attention 分布，target image 直接 copy source，object 没被加上。

#### 4.1.2 加权方案

公式 (3) 引入三个权重：

$$A = \text{softmax}\left([Q_p, Q_{target}][\gamma_s K_{source}, \gamma_p K_p, \gamma_t K_{target}]^\top / \sqrt{d_k}\right)$$
$$h = A \cdot [V_{source}, V_p, V_{target}]$$

变量：
- $\gamma_s$：source image 的 key scale，下标 $s$ = source
- $\gamma_p$：prompt 的 key scale
- $\gamma_t$：target image 自己的 key scale

注意：**只 scale keys，不 scale values**。这是个微妙但重要的设计——values 决定"信息内容"，keys 决定"被 attend 的概率"。只调 keys 等于在调 routing 权重，不污染信息本身。

#### 4.1.3 自动确定 γ

设两个量：
$$A_{source} = \frac{\exp(Q_p \cdot K_{source})}{Z}$$
$$A_{target} = \frac{\exp(\gamma \cdot Q_p \cdot K_{target})}{Z}$$

其中 $Z$ 是 softmax 的 normalization constant（所有 exp 项之和）。$A_{source}$ 表示 prompt token 对 source image 的 attention mass，$A_{target}$ 是 prompt token 对 target image 自己的 attention mass。

定义 $f(\gamma) = A_{source} - A_{target}$，用 root-finder 解 $f(\gamma) = 0$。

直觉：让 prompt 的注意力在 source 和 target 之间**均分**。这样 source 提供 scene context，target 有足够"自主权"去 incorporate 新 object。

实践中发现 $\gamma_p = \gamma_t = \gamma$ 就够了，最终值 $\gamma = 1.05$。注意这个值非常接近 1，说明只需要轻微 boost target 就能反转主导关系——这本身就是一个关于 attention scaling 敏感性的有趣发现。

#### 4.1.4 Analysis 的精髓

Figure 7(B) 是 paper 最 informative 的图之一。它可视化了 prompt token 的 attention 在三个 source 上的分布：
- $\gamma = 1.0$：source（紫色）压倒 target（橙色）→ object 不出现
- $\gamma = 1.2$：target 压倒 source → 偏离 source structure
- $\gamma = \text{Auto}$：两者平衡 → object 出现且位置合理

这个分析揭示了一个**杠杆效应**：attention 分布对 key scaling 极其敏感，微小变化就能翻转主导关系。这也解释了为什么 naive extended attention 不行——它本质上把所有 keys 等权处理。

---

### 4.2 Structure Transfer

#### 4.2.1 问题诊断

即便 attention 平衡了，还有一个隐藏问题：**FLUX 的 random seed 决定 global structure**。

举个例子：你给 source 是"狗坐椅子上"，prompt 是"加个篮子"。不同 random seed 生成的 target 可能是"狗坐地上"、"狗站着"等等。即便 attention 把 source 信息拉进来，seed 决定的 structural prior 太强，attention 拉不回来。

Figure 8 的 ablation 证实这一点：相同 seed 下，有/无 extended attention 的输出 structure 相似。

#### 4.2.2 方案

基于 rectified flow 的 noising 公式：
$$X_t = (1 - \sigma_t) x_0 + \sigma_t \epsilon$$

变量：
- $x_0$：clean image latent
- $\sigma_t$：timestep $t$ 对应的 noise schedule 值，$\sigma_0 = 0$（clean），$\sigma_T = 1$（pure noise）
- $\epsilon \sim \mathcal{N}(0, I)$：高斯噪声
- $X_t$：timestep $t$ 的 noisy latent

方案：把 source latent 加噪到 $t_{struct}$（很高但不到 $T$），用这个 noisy latent 作为 target 的起始点。

实验值：
- 生成图像：$t_{struct} = 933$（FLUX 用 1000 步 schedule）
- 真实图像：$t_{struct} = 867$

直觉：在 $\sigma_{t_{struct}} \approx 0.93$ 时，latent 保留了 source 的 low-frequency global structure（layout、object 大致位置），但 high-frequency detail 被 noise 抹掉，留出空间让 prompt 注入新 object。

#### 4.2.3 Ablation 的双刃剑

Figure 8 显示：
- $t_{struct}$ 太早（比如 800）：structure 没传够，target 偏离 source
- $t_{struct}$ 太晚（比如 980）：noise 太多，object 被"洗掉"（inclusion 下降）
- $t = 933$：平衡点

这个 trade-off 本质上反映 rectified flow schedule 的信息分层——低 noise level 编码 high-frequency detail，高 noise level 编码 low-frequency structure。

---

### 4.3 Subject-Guided Latent Blending

#### 4.3.1 为什么需要这一步

前两步保证了：
1. Object 会出现
2. Global structure 和 source 一致

但**fine detail**（纹理、小物体）仍可能 drift。原因：diffusion 是迭代 denoise，每步都有微小 perturbation，累积起来 background 会变。

#### 4.3.2 挑战

朴素方案：搞个 mask 分离 object 和 background，background 直接用 source。两个问题：
1. **Mask 不准导致 artifact**：边缘错位会很明显
2. **丢失 collateral effects**：新 object 投下的 shadow、reflection 也会被 mask 掉

#### 4.3.3 流程详解

**Step 1: Extract rough mask from attention**

利用 self-attention map 定位 object。具体：

$$M_{rough}(i) = \text{aggregate}\left(Q_{target}^{(i)} \cdot k_{object}\right)$$

其中：
- $Q_{target}^{(i)}$：target image 第 $i$ 个 patch 的 query
- $k_{object}$：object token（prompt 里 "dog" 这种词）的 key
- 点积衡量 patch $i$ 和 object token 的 semantic alignment

聚合在特定 layer 和 timestep 上做。Paper appendix A.1 给出经验最优 layer：
- Multi-stream: `transformer_blocks.13, 14, 18`
- Single-stream: `single_transformer_blocks.23, 33`

直觉：这些 layer 处于"semantic ↔ spatial"的交界，attention map 既反映 object identity 又反映位置。

**Step 2: Otsu thresholding**

[Otsu 1979](https://ieeexplore.ieee.org/document/4310076) 经典方法，自动找使类间方差最大的阈值，把 continuous attention map 二值化。优势：无需手调阈值。

**Step 3: Refine with SAM-2**

SAM-2 ([Ravi et al. 2024](https://arxiv.org/abs/2408.00714)) 需要 image input + localization prompt（points / box / mask）。

- Image 怎么来？从 velocity prediction $v_\theta$ 估计 $X_0$：
  $$X_0 = X_{T_{blend}} + (\sigma_{T_{blend}+1} - \sigma_{T_{blend}}) \cdot v_\theta$$
  
  这是 rectified flow 的 $X_0$ 预测公式，$v_\theta$ 是 model 预测的 velocity field
- Points 怎么来？从 attention map 迭代采样 local maxima：
  1. 选 attention 最大的点
  2. Mask 掉周围区域
  3. 选下一个最大点
  4. 重复直到 4 个点，或最大值低于 $0.35 \cdot p_{max}$（$p_{max}$ 是初始最大值）

**Step 4: Latent blending at $T_{blend}$**

$$Z_{target} = M \odot Z_{target} + (1 - M) \odot Z_{source}$$

- $M$：refined mask，1 表示 object 区域
- $\odot$：element-wise product
- $Z_{target}, Z_{source}$：timestep $T_{blend}$ 的 noisy latents

关键设计：**blend 在中间 timestep $T_{blend} = 500$ 做，不在最后做**。这样：
- $t < T_{blend}$：完全由 target generation 决定，object 和 background 自然融合
- $t > T_{blend}$：background 用 source 的 noisy latent，但 denoise 过程还会演化（保留 shadow、reflection 等 collateral effect）

这个设计非常聪明——它把"hard pixel copy"转化成了"latent initialization"，让后续 denoise step 平滑过渡。

---

### 4.4 Real Image 处理

#### 4.4.1 Inversion 的失败

标准做法：用 [DDIM inversion](https://arxiv.org/abs/2010.02502) 把 real image 反推到 noise。但 paper 报告这在 FLUX 上不 work——重建质量差。

可能原因（我的猜测）：
- FLUX 是 rectified flow 不是 DDPM，inversion 算法不直接适用
- FLUX 的 text conditioning 很强，inversion 时 caption 不准会累积误差
- 28 step 推理对于精确 inversion 太少

#### 4.4.2 简单替代方案

每个 denoise step $t$，重新构造 noisy source：

$$X_{source}^t = (1 - \sigma_t) X_{source} + \sigma_t \epsilon$$

其中 $\epsilon$ 是固定的随机噪声（一次采样全程用同一个）。

这个 trick 利用了 rectified flow 的线性性：在 $\sigma_0 = 0$ 时 $X_{source}^0 = X_{source}$，**保证完美重建**。在中间 timestep，$X_{source}^t$ 提供了 source 的结构 hint，但不强制 target 必须等于 source。

工程上很优雅，但 limitation 部分承认 real image 效果仍不如 generated image——可能因为这种"伪 inversion"不能提供真正等价于 noise space 的 representation。

---

## 5. Positional Encoding 的隐藏角色

Appendix A.3 的实验我觉得是 paper 最有意思的 finding 之一，但被放在了 appendix。

实验：把 source image 的 positional encoding 向右下 shift，然后跑 Add-it。结果：**生成的 object 位置也 shift 了**。

含义：
- Extended attention 不是基于 content similarity 来 transfer feature 的
- 它高度依赖 positional encoding 的对应关系
- 即便 target image 在"head"位置其实是 laptop 的特征，模型还是会把 headphone 放在那里，**因为 source 的 head 在那个位置**

这暗示 DiT 模型的 attention 机制比想象中更"位置驱动"。这点对未来的工作有启发：
- 也许应该研究 content-aware positional encoding
- 也许 cross-image attention 需要 explicit 的 correspondence cue
- 这也解释了为什么 structure transfer 必要——它保证了 source 和 target 的 positional encoding 在 layout 上对齐

---

## 6. Affordance Benchmark 设计

这部分工程价值很高。

### 6.1 数据构造

用 [GPT-4](https://openai.com/research/gpt-4) 生成 300 组 (source_prompt, target_prompt, instruction, subject_token)。Prompt 模板见 Figure 18，关键约束：
- "Only generate examples where there is clearly only one possible place for the object to be added"
- 排除 negative example（"a man wearing no hat"）

用 FLUX 生成 source image，人工 filter 到 200 张，再人工标 bounding box 标记 plausible placement area。

### 6.2 Evaluation Protocol

用 [Grounding-DINO](https://arxiv.org/abs/2303.05499) 检测 output image 中新 object 的位置，计算：

$$\text{Affordance} = \frac{\text{IoU(detected\_box, GT\_box)} \geq 0.5 \text{ 的比例}}{\text{总样本数}}$$

更精确说：检测到的 object box 至少 50% 面积落在 GT box 内才算正确。

### 6.3 Result 解读

Table 1:
| Method | Affordance |
|---|---|
| InstructPix2Pix | 0.276 |
| EraseDraw | 0.341 |
| MagicBrush | 0.418 |
| SDEdit+P2P | 0.474 |
| **Add-it** | **0.828** |

差距惊人。Supervised 方法（IP2P、MagicBrush、EraseDraw）反而比 training-free 的 SDEdit+P2P 还差。这强烈暗示：**affordance 不是用 supervised reconstruction loss 能学到的**，它需要 foundation model 自身的 world knowledge。Add-it 的成功在于它找到了正确"打开"这个 knowledge 的方式。

---

## 7. 实验 Result 细读

### 7.1 Emu-Edit Benchmark (real images)

Table 2:
- CLIP_dir (方向一致性): Add-it 0.101 > EraseDraw 0.088
- CLIP_out (output 与 caption 相似度): Add-it 0.322 最高
- CLIP_im (与 source 相似度): EraseDraw 0.941 > Add-it 0.929
- Inclusion: Add-it 81% vs EraseDraw 65%

Paper 指出 EraseDraw 的 CLIP_im 高是 misleading——35% 的情况它根本没加 object，自然 image similarity 高。这暴露了 automatic metric 的陷阱。

### 7.2 Additing Benchmark (generated images)

- CLIP_dir: Add-it 0.200，远超所有 baseline
- Inclusion: 93%
- 但 CLIP_out 和 Inclusion 上 Prompt-to-Prompt 略高，因为它**完全重画**，所以和 prompt 对齐好但 CLIP_im 只有 0.850（vs Add-it 0.968）

### 7.3 Human Eval

- vs InstructPix2Pix: 80% preferred
- vs EraseDraw: 81%
- vs MagicBrush: 82%
- vs SDEdit: 81%
- vs Prompt-to-Prompt: 90%

90% vs P2P 这个数字特别高，因为 P2P 在 additing 上确实结构性不擅长。

---

## 8. Limitations

Paper 自己承认：
1. **继承 pretrained model bias**：复杂 scene 仍可能失败
2. **Prompt 需要更详细**："A dog" 不会加第二条狗，需要 "Two dogs sitting on the grass"。这点和 InstructPix2Pix 的 instruction-following 范式相比是退步
3. **Real image 弱于 generated image**：归因于 FLUX inversion 不好

Figure 10 的 failure case 很有教学意义：让加"another dog"，Add-it 倾向于 reproduce 原狗而不是生成新狗。这反映了 attention 机制的本质——它 transfer existing feature 比 synthesize new instance 容易。

---

## 9. 我的 Intuition 和联想

### 9.1 为什么 attention scaling 这么敏感

公式 (3) 里 $\gamma = 1.05$ 就能翻转主导关系，这个微小数值背后是 softmax 的指数放大效应。考虑：

$$\frac{\exp(\gamma \cdot q \cdot k_1)}{\exp(\gamma \cdot q \cdot k_2)} = \exp(\gamma \cdot q \cdot (k_1 - k_2))$$

如果 $q \cdot (k_1 - k_2)$ 本来是 $O(1)$ 量级，$\gamma$ 从 1.0 变到 1.05 等于把 logit gap 放大 5%，但因为 softmax 是 winner-take-all 的，这个 5% 可能把二阶位置的 token 顶到第一位。

这点对推 general attention-based editing method 很重要——**naive 拼接 token 永远会被 dominant source 接管**，必须显式 reweight。

### 9.2 联系到 StyleAligned / MasaCtrl 系列工作

[MasaCtrl](https://arxiv.org/abs/2303.01548) (Cao et al. 2023) 和 [StyleAligned](https://arxiv.org/abs/2312.02133) 都用类似的 extended self-attention 做 appearance/style transfer。Add-it 的区别：
- 那些方法假设 source 和 target 是**同类内容**（同一只狗不同 pose，或同风格不同 content）
- Add-it 处理**异类内容**——source 是"狗在椅子"，target 是"狗在椅子+篮子"，篮子在 source 里根本不存在

这就是为什么 Add-it 需要 weighting 而之前的方法不需要——之前的 extended attention 默认 source 主导是 feature，不是 bug。Add-it 把它当 bug 来 fix。

### 9.3 与 ControlNet 的对比

[ControlNet](https://arxiv.org/abs/2302.05543) 用结构条件（depth、canny）来约束 generation。Add-it 的 structure transfer 类似，但**用 noise 而不是显式 condition**。优势：
- 不需要额外 encoder
- 不需要训练
- 保留 FLUX 原生 distribution

劣势：
- 控制度不如 ControlNet 精确
- 依赖 schedule 的 information 分层假设

### 9.4 关于 Rectified Flow 的角色

整个 method 高度依赖 rectified flow 的**线性 noising**：
$$X_t = (1 - \sigma_t) x_0 + \sigma_t \epsilon$$

这使得：
1. Structure transfer 可行——线性插值意味着 $\sigma_t$ 高时 $X_t$ 主要由 $\epsilon$ 决定，但 $(1-\sigma_t)x_0$ 项保留 source 弱信号
2. Real image 的 "fake inversion" 可行——直接用线性公式构造 $X_{source}^t$
3. $X_0$ 预测简单——线性外推就行

如果换成 DDPM 的非线性的 forward process，这些 trick 都不直接适用。这间接说明**架构选型决定 method design space**。

### 9.5 与 RAG 的类比

我觉得 Add-it 在 concept 上类似 Retrieval-Augmented Generation：
- "Retrieval"：从 source image 提取 K, V
- "Augmented generation"：target image 生成时 attend 到这些 retrieved token
- "Weighting"：类似 RAG 里的 relevance threshold

这个类比也许能 inspire 更多 unification——比如把 multiple reference images 的 K, V 都拼进去，做 "retrieval-augmented image editing"。

### 9.6 关于 Positional Encoding 实验的更深含义

Appendix A.3 的实验让我想到 [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) 那篇 paper 的发现——ViT 会在 high-norm token 上 dump 全局信息。Add-it 的 positional encoding 实验暗示：**DiT 用 positional encoding 作为 cross-image correspondence 的 implicit cue**。

这可能解释为什么 DiT 在 image-to-image task 上表现和 UNet 很不同。未来工作可以探索：
- 显式 correspondence token
- Relative positional encoding 跨 image
- Content-conditioned positional encoding

### 9.7 Failure mode 的哲学含义

Figure 10 的"无法加 another dog"暴露了一个深层问题：**attention 机制本质是 feature transfer，不是 instance synthesis**。要让模型生成"另一只不同的狗"，可能需要：
- 显式 instance token（类似 Slot Attention）
- Object-centric latent decomposition
- 或者用更 explicit 的 instruction tuning

这和 [Object-Centric Representation Learning](https://arxiv.org/abs/2202.02905) 方向的工作有连接。

---

## 10. 总结性 Intuition

Add-it 的核心贡献，在我看来，是**把 diffusion model 的 attention 重新理解为多源信号路由器**：

1. **Naive 视角**：attention 是 token 间的 information flow
2. **Add-it 视角**：attention 是 source image / prompt / target image 三个 information source 之间的 dynamic router，scaling 控制了路由权重

这个视角的 power 在于：
- 它把"加 object"这个 semantic 操作转化成了"调 attention 分布"这个 mechanical 操作
- 它不需要 retrain，因为 routing 机制本来就在
- 它暴露了 foundation model 内在的 affordance knowledge

但它的 limitation 也很本质：
- Attention 是 feature-level operation，做不了 instance-level reasoning
- Real image inversion 还是瓶颈
- Prompt 表达力比 instruction 弱

未来方向我能想到的：
- 把 instruction → prompt translation 用 LLM 自动做
- 探索 multi-reference extended attention（多个 source image）
- 结合 object-centric 方法突破 instance synthesis 瓶颈
- 研究专门为 additing 设计的 inversion 算法

---

## Reference Links

- [Add-it project page](https://research.nvidia.com/labs/par/addit/)
- [FLUX](https://github.com/black-forest-labs/flux)
- [SD3 / MMDiT paper](https://arxiv.org/abs/2403.03206)
- [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)
- [SDEdit](https://arxiv.org/abs/2108.01073)
- [InstructPix2Pix](https://arxiv.org/abs/2211.09800)
- [MagicBrush](https://arxiv.org/abs/2211.09769)
- [Emu Edit](https://arxiv.org/abs/2311.10089)
- [EraseDraw](https://arxiv.org/abs/2404.01598)
- [MasaCtrl](https://arxiv.org/abs/2303.01548)
- [StyleAligned](https://arxiv.org/abs/2312.02133)
- [SAM-2](https://arxiv.org/abs/2408.00714)
- [Grounding DINO](https://arxiv.org/abs/2303.05499)
- [DDIM Inversion](https://arxiv.org/abs/2010.02502)
- [Otsu method (原始 paper)](https://ieeexplore.ieee.org/document/4310076)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- [Object-Centric Learning survey](https://arxiv.org/abs/2202.02905)

希望这个解析对 build intuition 有帮助，Andrej。如果你想 drill into 某个具体 component（比如 attention scaling 的理论分析，或 structure transfer 的 frequency-domain 直觉），我们可以继续展开。
