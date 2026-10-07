---
source_pdf: GAUSSIANANYTHING.pdf
paper_sha256: 906be7ed8567b0d43638b6265d492241a2f9fb60200959156c5b6b581db0ef66
processed_at: '2026-08-04T12:33:33-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GAUSSIANANYTHING

Andrej，咱们抛开 paper 那套学术腔，坐下来聊这玩意儿到底在干嘛。

---

## 一句话总结

**他们想让 3D generation 拥有 Stable Diffusion 在 2D 上那种 "在 latent space 里 drag 一下就能编辑" 的能力，而 key insight 是——3D 的 "pixel grid" 就是 point cloud 本身。**

---

## 为什么要做这件事

你去用 Stable Diffusion 画图，想做 inpainting、想做 drag editing、想做 ControlNet，之所以能 work，是因为 VAE 的 latent code 跟原图 pixel grid 是 spatially 一一对应的。你在 latent 上某一块做手脚，回去就是图像上对应那一块发生变化。这个 correspondence 是 2D editing 全部的基础。

3D 一直缺这个。之前大家做 3D native diffusion（Point-E、Shape-E、LION、CLAY、LN3Diff），latent space 设计都有毛病：

- **LION / CLAY 那一派**把 3D 物体 encode 成一个 set of tokens，类似 transformer 里的 $N$ 个 slot。问题是这些 token 是 permutation-invariant 的，token #47 跟 3D 物体的哪个部位对应？鬼才知道。你想 drag 一下鼻子，根本不知道该改哪个 token。
- **LN3Diff 那一派**用 triplane latent，三个 axis-aligned plane 加起来表征 3D feature volume。但 triplane 的一个 cell 对应 3D 空间里的一整条射线，你想编辑物体左侧某个点，triplane 一改，右侧同一条射线上所有点都跟着乱动。Editing 是糊的。

GAUSSIANANYTHING 的作者看了一眼说：**等等，3D 里最自然、最 spatially-corresponded 的 representation 就是 point cloud 啊。** 每个 point 有自己的 $(x,y,z)$ 坐标，你改哪个 point 就是改哪个 spatial 位置。为啥不直接让 latent code 长成 point cloud 的样子？

这就是 paper 标题 "Interactive Point Cloud Flow Matching" 的来历。

---

## 整个 pipeline 在干啥

我分三步给你讲清楚。

### Step 1: 3D VAE — 把物体压成 "point cloud shaped latent"

**输入**：一个 3D 物体的 8 个视角的 RGB-D-N 渲染图。每张图自带 RGB、depth、normal map、camera pose。

为啥不直接用 dense point cloud 当输入？因为 point cloud 是 sparse signal，CNN 处理起来别扭，而且 texture 这种 high-frequency 信息在 sample 到 16k points 时就丢差不多了。用 multi-view rendering 就灵活多了——任何 3D source 都能 render 出来，dataset 可扩展性强很多。

每个 view 的每个像素都被显式编码了 15 维信息：
- RGB (3)
- 像素在世界坐标的 3D 位置 (3，depth unprojection)
- Normal (3)
- Plücker ray embedding (6，camera ray 的位置 + 方向)

也就是说**每个像素都"知道"自己在 3D 空间里对应哪条 ray、看到啥 normal、看到啥 color**。Transformer 不需要再隐式地 infer geometry，所有 3D 信息都是显式喂进去的。

**Encoder**：先 CNN 把每张图 downsample 成 patch tokens，然后一个 multi-view transformer 让 8 个 view 的 tokens 全 attend 起来。产出 32768 个 tokens 的 set latent $\mathbf{z}_z$。

**问题来了**：32768 个 token 做 diffusion 太贵了，attention 是 $O(N^2)$，扛不住。而且这些 tokens 还是 view-indexed，不是 native 3D，会有 multi-view inconsistency。

**Solution**：用 cross attention 把这 32768 个 view tokens "投影" 到 768 个 3D point 上。具体来说，从 input 3D shape 表面用 FPS (Farthest Point Sampling) 采 768 个点，每个点作为 query，去 attend 那 32768 个 view tokens，问它们："你们看到了我附近的什么信息？" 这样每个 3D 点就 gather 了周围多视角的信息。

结果就是一个 point-cloud structured latent：
$$\mathbf{z} = [\mathbf{z}_x \oplus \mathbf{z}_h] \in \mathbb{R}^{13 \times 768}$$
- $\mathbf{z}_x$ (768×3): point cloud 坐标，代表 geometry layout
- $\mathbf{z}_h$ (768×10): 每个 point 上的 feature，代表 texture / appearance proxy

这个 latent 同时具备了三个性质：
1. **Compact**：768 个 token，比 32768 小 43 倍
2. **3D-native**：每个 token 自带 3D 坐标
3. **Editable**：改 $\mathbf{z}_x$ 就是改 geometry，改 $\mathbf{z}_h$ 就是改 texture，天然 disentangle

### Step 2: Cascaded Flow Matching — 两阶段 diffusion

这里有个关键 design choice：不把 $\mathbf{z}_x$ 和 $\mathbf{z}_h$ 联合建模，而是 cascade 两阶段。

**Stage-1: Point Cloud Diffusion**
只对 $\mathbf{z}_x$ (768×3) 做 flow matching，condition 在 text/image prompt 上。生成 coarse geometry layout。

$$\mathcal{L}_w^x(x_0) = -\frac{1}{2} \mathbb{E}_{t, \epsilon} \left[ w_t^{\text{FM}} \lambda_t' \|\epsilon_\Theta^x(\mathbf{z}_{x,t}, t, c) - \epsilon\|^2 \right]$$

**Stage-2: Point Feature Diffusion**
对 $\mathbf{z}_h$ (768×10) 做 flow matching，**额外 condition 在 Stage-1 输出的 clean point cloud $\mathbf{z}_{x,0}$ 上**（不是 noisy 版本，是 final 输出）。

$$\mathcal{L}_w^h(x_0) = -\frac{1}{2} \mathbb{E}_{t, \epsilon} \left[ w_t^{\text{FM}} \lambda_t' \|\epsilon_\Theta^h(\mathbf{z}_{h,t}, \mathbf{z}_{x}, t, c) - \epsilon\|^2 \right]$$

这跟 Imagen 的 cascaded diffusion 思路类似，但 cascade 的不只是 resolution，而是 **representation modality**——先出 geometry，再 condition 在 geometry 上出 texture。

**为这么做**：geometry 和 texture 是不同 DOF 的信息，联合建模会互相 leak noise。Cascaded 之后：
- 改 $\mathbf{z}_x$（geometry）能 re-generate 对应的 $\mathbf{z}_h$（texture），保持 texture 合理
- 同一个 $\mathbf{z}_x$ 配不同 random seed 出不同 texture
- 两者 disentangle，3D editing 才有可能

**Flow matching** 而不是 DDPM：path 是直的 $z_t = (1-t)x_0 + t\epsilon$，ODE 几步就能 sample，效率高。实现用了 SiT (Scalable Interpolant Transformers) 的 recipe — pred-v objective + GVP schedule + uniform t sampling。

**DiT backbone**: 24 层, 16 heads, 1024 hidden dim, 458M params。AdaLN-single (PixArt-α 风格) 注入 timestep, QK-Norm 稳训练, cross-attention 注入 condition。

**Conditioning**:
- Text: CLIP penultimate tokens
- Image: DINOv2 global + patch features
- CFG: 10% drop, sampling CFG=4, 250 ODE steps

### Step 3: VAE Decoder — Latent 变回 surfel Gaussians

接到 $\mathbf{z}_0 = [\mathbf{z}_{x,0} \oplus \mathbf{z}_{h,0}]$ 后，要 decode 成 dense 3D Gaussians 用于渲染。

结构：DiT transformer refine → 3 级 cascaded upsampler

每级 upsampler 把 $N$ 个 token 裂变成 $f_u \cdot N$ 个。具体做法是每个 input token 前面 prepend $f_u$ 个 learnable sub-tokens，transformer 让它们互相 attend 决定怎么"分裂"。

级联三次 $f_u = 8, 4, 3$：
$$768 \xrightarrow{\times 8} 6144 \xrightarrow{\times 4} 24576 \xrightarrow{\times 3} 73728 \text{ Gaussians}$$

每个 output token 是一个 13 维的 surfel Gaussian (2D Gaussian Splatting, Huang et al. 2024a)：center、两个 tangent vector、scale、opacity、view-dependent color。

为啥用 surfel Gaussian 而不是 3DGS 或 triplane？
- vs 3DGS: surfel 是 flat 的，更 align 实际物体表面，reconstruct 出来的 surface 更干净
- vs triplane: triplane + volumetric rendering 在 high resolution 时 memory 爆炸，surfel 是 surface-based，快很多

**一个 nice 副产品**：三级 upsampler 自然形成 Level-of-Detail，手机预览用 LoD=1 (768 Gaussians)，桌面高质量用 LoD=3 (74k Gaussians)。

---

## 训练 loss

VAE training:
$$\mathcal{L} = \mathcal{L}_{\text{render}} + \mathcal{L}_{\text{geo}} + \lambda_{\text{kl}}\mathcal{L}_{\text{KL}} + \lambda_{\text{GAN}}\mathcal{L}_{\text{GAN}}$$

- Render: $\mathcal{L}_1$ + VGG perceptual loss，input view 和 random novel view 都监督
- Geometry: Mip-NeRF 的 depth distortion loss + normal consistency loss
- KL: 让 latent 接近 $\mathcal{N}(0, I)$，方便后面 flow matching
- GAN: discriminator 提 perceptual fidelity

权重 $\lambda_{\text{kl}}=2e{-6}, \lambda_d=1000, \lambda_n=0.2, \lambda_{\text{GAN}}=0.1$。$\lambda_d$ 很大，说明 depth distortion 是强约束。

每次 iteration 随机选一个 LoD 监督，省 compute。

---

## 实验关键数据

**Text-to-3D** (Table 1): 所有 metrics 都领先 Point-E / Shape-E / LN3Diff / 3DTopia。Q-Align (接近 human judgment 的 LMM-based scorer) 3.13 vs LN3Diff 2.22，差距显著。

**Image-to-3D** (Table 2): 最 striking 的对比是跟 LGM:
- LGM FID 19.93 < Ours 24.21（LGM 视觉质量略好）
- LGM P-FID 32.37 vs Ours 8.72（Ours 3D shape quality 好 4 倍）

这证实了 paper 的论点：multi-view → 3D cascaded pipeline (LGM 用 MVDream 出 4 views 再 reconstruction) 视觉质量好但 3D geometry 烂。Native 3D diffusion 在 3D 一致性上完胜。

**Gaussian Utilization** (Table 4): Ours 96.84% vs LGM 52.63% vs Splatter Image 17.14%。Pixel-aligned 方法把一半 Gaussian 浪费在空白背景和 multi-view overlap 上，Ours 是 surface-aligned，几乎全有效。实际渲染速度和内存占用都有 ~2x 优势。

**Ablation** (Table 3):
- Dense PCD input LPIPS 0.174 → Multi-view RGB-D 0.163 → + Normal 0.157 → + Upsampler 0.095 → + 3× Upsampler 0.067
- 每个 design choice 都有 measurable gain，upsampler 提升 最大

**Cascaded ablation** (Fig 6a): 单 stage joint training 的 texture 明显 worse，3D shape 也有 artifacts。Cascade 是必要的。

---

## 3D Editing — 这才是 payoff

Fig 5 和 Fig 6b 展示了 latent structure 设计的真正价值：

**Texture variation**: 用同一个 stage-1 输出 $\mathbf{z}_{x,0}$，给 stage-2 不同 random seed，得到同 geometry 不同 texture 的 3D 物体。

**Geometry editing**: 直接修改 $\mathbf{z}_{x,0}$（比如把消防栓盖子挪开），然后 stage-2 用 same Gaussian noise 重新生成，得到结构变了但 texture 合理的结果。

对比直接编辑 dense 3D Gaussians：会出 tearing artifacts，因为 Gaussian 之间没有 semantic binding，硬拉就撕裂。在 latent space 编辑则 holistic——VAE decoder 会重新 decode 出 consistent 的 Gaussian set。

这跟 2D 里的 DragGAN / DragonDiffusion 是同一个 idea：**在 generative manifold 上 edit，让 prior 帮你 fill in the blanks**。

---

## 我的几个 takeaways

**1. Latent space design 决定上限。** 2D LDM 的成功很大程度是 VAE latent 跟 pixel grid 一一对应这个红利。3D 一直缺这个，GAUSSIANANYTHING 用 point cloud 作为 latent 是非常 natural 的选择——point cloud 就是 3D 的"pixel grid"。

**2. Cross attention 是 representation converter。** $\mathbf{z}_z$ (unstructured set) → $\mathbf{z}_h$ (point-structured) 这个 cross attention 操作，其实是个 general pattern。CLAY / 3DShape2VecSet 也用 cross attention 把 set 投到 learnable query 上，区别是它们的 query 是 arbitrary learnable tokens，这里 query 是有空间意义的 point cloud。

**3. Cascade for disentanglement。** 两阶段 cascade 不只是为 quality，更为了让 $\mathbf{z}_x$ 和 $\mathbf{z}_h$ 在生成时 disentangle。这种 disentanglement 让 editing 成为可能。类似 Imagen 的 cascaded diffusion，但这里 cascade 的是 representation modality 而不是 resolution。

**4. Multi-view RGB-D-N > Dense point cloud as VAE input。** 两点好处：信息更 comprehensive（texture + geometry + normal），dataset 可扩展性更好（任何 3D source 都能 render）。

**5. Surfel Gaussian > Triplane。** LN3Diff 用 triplane + volumetric rendering，高分辨率时 memory 爆炸。2DGS 是 surface-based，渲染效率和 surface quality 都更好。配 cascaded upsampler 还能做 LoD。

---

## 同期工作对比

- **Trellis** (Xiang et al. 2024): 用 sparse voxel 替代 point cloud 作为 latent，类似 cascaded flow matching 框架。Sparse voxel 有 octree structure，可能比 flat point cloud 更适合复杂 topology 的 hierarchical 编辑。这条 path 很 promising。
- **AtlasGaussians** (Yang et al. 2025): 也是 native 3D GS diffusion，但没 explicit latent space，不能做 interactive editing。
- **Direct3D** (Wu et al. 2024): 也是 3D native latent diffusion，但用 triplane latent，editing 能力受限。
- **Geometric Distribution** (Zhang et al. 2024a): 只做 point cloud diffusion 不做 texture，效率问题待解决。

---

## Weaknesses 和未来方向

Paper 自己承认：
1. Texture 质量还比不过 LGM 这种 multi-view cascaded pipeline（LGM 用了 MVDream 2D prior，Ours 纯 3D 训练）
2. 没用 2D pretraining prior，generalization 受限于 3D dataset size
3. 768 个 latent point 对复杂物体可能不够
4. DiT-B/2 decoder 比 DiT-XL 弱（VRAM constraint）

未来方向：
1. 增大 latent point 数量到几千
2. Diffusion training 加 rendering loss（类似 DMV3D）
3. 用 2D pretraining prior（类似 LGM 用 MVDream）
4. 引入 PBR rendering（3DGRT, Moenne-Loccoz et al. 2024）
5. 扩 dataset 到 Objaverse-XL (10M+) 和 MVImageNet

---

## 最后

我个人觉得这篇 paper 的真正贡献是把 2D LDM 的"editable latent"哲学完整 transfer 到 3D。找到 3D 的 spatially-corresponded latent（point cloud）→ cascaded diffusion disentangle geometry/texture → surfel Gaussian 实现 high-quality rendering → multi-view RGB-D-N input 让 dataset 可扩展。

这四点加起来，让 3D native diffusion 第一次在 quality、editability、scalability 三个维度同时达到可用水平。如果未来 3D generation 有 "Stable Diffusion moment"，这条 path 是最有希望的候选之一。

Point cloud vs sparse voxel 谁能成为 3D 的 "pixel grid"，还是 open question。Point cloud 更轻量、更接近 artist mental model；sparse voxel 有 octree structure，更适合 hierarchical 编辑。我个人赌 sparse voxel 会赢，但 paper 这条 point cloud path 已经证明 concept 是 work 的。

---

**Reference**:
- Project: https://nirvanalan.github.io/projects/GA/
- Stable Diffusion: https://arxiv.org/abs/2112.10752
- LION: https://arxiv.org/abs/2210.10695
- CLAY: https://arxiv.org/abs/2406.04459
- LN3Diff: https://arxiv.org/abs/2403.07957
- 2DGS: https://arxiv.org/abs/2403.17888
- Trellis: https://arxiv.org/abs/2412.01506
- Flow Matching: https://arxiv.org/abs/2210.02747
- SiT: https://arxiv.org/abs/2401.08740
- PixArt-α: https://arxiv.org/abs/2310.00426
- DragGAN: https://arxiv.org/abs/2305.10974
- DragonDiffusion: https://arxiv.org/abs/2307.02421
- LGM: https://arxiv.org/abs/2402.05063
- MVDream: https://arxiv.org/abs/2308.16512
- DMV3D: https://arxiv.org/abs/2311.13292
- Direct3D: https://arxiv.org/abs/2407.02244
- AtlasGaussians: https://arxiv.org/abs/2412.02009

---

# GAUSSIANANYTHING 深度解读

Andrej，这篇 paper 我觉得非常值得拆解，因为它在 3D generation 这个领域做了一件类似 Stable Diffusion 当年为 2D generation 做的事情：**为 3D 找到一个 editable 的 latent manifold**。让我从 motivation 一路推到公式，重点放在 build intuition 上。

参考链接：
- Project page: https://nirvanalan.github.io/projects/GA/
- Stable Diffusion (Rombach et al. 2022): https://arxiv.org/abs/2112.10752
- LION (Zeng et al. 2022): https://arxiv.org/abs/2210.10695
- CLAY (Zhang et al. 2024b): https://arxiv.org/abs/2406.04459
- 3DShape2VecSet (Zhang et al. 2023): https://arxiv.org/abs/2305.18339
- LN3Diff (Lan et al. 2024a): https://arxiv.org/abs/2403.07957
- 2D Gaussian Splatting (Huang et al. 2024a): https://arxiv.org/abs/2403.17888
- Flow Matching (Lipman et al. 2023): https://arxiv.org/abs/2210.02747
- SiT (Ma et al. 2024): https://arxiv.org/abs/2401.08740
- PixArt-α (Chen et al. 2023): https://arxiv.org/abs/2310.00426
- DiT (Peebles & Xie 2023): https://arxiv.org/abs/2212.09748
- Trellis (Xiang et al. 2024): https://arxiv.org/abs/2412.01506

---

## 1. The Core Problem: 为什么 3D Latent Diffusion 这么难做

2D LDM (Stable Diffusion) 的成功有一个被低估的关键点：**VAE latent 与 input pixel grid 一一对应**。这带来一个直接红利——你在 latent 上做 drag、做 inpainting、做 ControlNet，都能 spatially align 回原图。Mou et al. 的 DragonDiffusion、Pan et al. 的 DragGAN 都是建立在这个 correspondence 之上的。

3D 领域一直缺这个。现有两种主流 latent space design 都不理想：

**Set latent** (LION, 3DShape2VecSet, CLAY)：把 3D object 编成一个 permutation-invariant 的 token set $\{z_i\}_{i=1}^N$。优点是 flexible、可以处理任意 topology。致命问题是 tokens 没有 spatial meaning——你不知道第 47 个 token 对应 3D 物体的哪个部位。所以做 3D drag-style editing 几乎不可能。

**Triplane latent** (LN3Diff, Direct3D, GS-LRM 风格)：把 3D feature volume 投影到三个 axis-aligned plane。问题是 triplane 的一个 cell 会聚合一整条射线上的信息，编辑某个 plane 区域时，反向映射回 3D 空间是模糊的（同一条射线上的所有点都被影响），所以 triplane 编辑也是不精确的。

GAUSSIANANYTHING 的 insight 是：**3D 中最 natural 的 spatially-corresponded representation 就是 point cloud 本身**。如果我们让 latent code **直接长成 point cloud 的形状**——每个 latent token 都绑定一个 3D 坐标——那 2D 编辑的直觉就能直接 transfer 过来。

---

## 2. Pipeline 全景

整个框架分三块：

**Stage A: 3D VAE** — 把 multi-view RGB-D-N 渲染 encode 成 point-cloud structured latent $\mathbf{z} = [\mathbf{z}_x \oplus \mathbf{z}_h] \in \mathbb{R}^{(3+C_h)\times N}$
- $\mathbf{z}_x \in \mathbb{R}^{3\times N}$: sparse point cloud 坐标（geometry skeleton）
- $\mathbf{z}_h \in \mathbb{R}^{C_h\times N}$: 每个 point 上的 feature（texture/appearance proxy）
- $N=768, C_h=10$

**Stage B: Cascaded Flow Matching** — 在 latent 上做两阶段 diffusion：
- Stage-1: 生成 $\mathbf{z}_{x,0}$ (point cloud layout)
- Stage-2: condition 在 $\mathbf{z}_{x,0}$ 上生成 $\mathbf{z}_{h,0}$ (point features)

**Stage C: Decode** — 用 VAE decoder 把 $\mathbf{z}_0$ 解成 dense surfel Gaussians，可微渲染监督

这个 cascade 是关键 design choice，下面会详细讲为什么。

---

## 3. 3D VAE 详解：怎么把 latent "concretize" 成 point cloud

### 3.1 Versatile Input: 为什么不用 dense point cloud 而用 multi-view RGB-D-N

之前 LION、Shape-E、CLAY 直接拿 dense colored point cloud (16k-50k points) 喂 VAE。问题有两个：
- Point cloud 是 sparse signal，CNN 难处理，而且 high-frequency texture 信息在 sample 到 16k points 之后会大量丢失
- 训练数据限制在 artist-created mesh，而 multi-view rendering 可以从任何 source（scan, NeRF, 4D capture）rendering 出来，dataset scalability 好很多

GAUSSIANANYTHING 改用 multi-view RGB-D-N rendering 作为 input。每个 view 渲染 $R = (I, \Delta, N, \pi)$：
- $I \in \mathbb{R}^{H\times W\times 3}$: RGB
- $\Delta \in \mathbb{R}^{H\times W}$: depth
- $N \in \mathbb{R}^{H\times W\times 3}$: normal map
- $\pi$: camera pose

然后做了两个聪明的变换：

1. **Plücker embedding** 把 camera pose $\pi$ 变成 dense tensor: $\mathbf{p}_{i} = (\mathbf{o}\times\mathbf{d}_{u,v}, \mathbf{d}_{u,v}) \in \mathbb{R}^6$
   - $\mathbf{o}$: camera origin (3D)
   - $\mathbf{d}_{u,v}$: 每个像素的 ray direction (3D)
   - $\times$: cross product
   - 结果 $\mathbf{P} \in \mathbb{R}^{H\times W\times 6}$: 每个像素的 ray 都被显式编码

2. **Depth unprojection** 把 $\Delta$ 变成世界坐标 $\mathbf{X} \in \mathbb{R}^{H\times W\times 3}$

最终 channel-wise concat:
$$\tilde{R} = [I \oplus \mathbf{X} \oplus N \oplus \mathbf{P}] \in \mathbb{R}^{H\times W\times 15}$$

直觉：每张图每个像素都"自带"了它的 3D 位置、法向、以及射线的 Plücker 编码。后面 transformer 处理时不需要再隐式地 infer 几何，所有 3D 信息都是显式输入。这是 MCC (Wu et al. 2023a) 的 trick，这里被复用得很自然。

### 3.2 Transformer Encoder: SRT-style

Encoder 是两段式：

$$\mathbf{z}_z = \mathcal{E}_\phi^{TX}(\mathcal{E}_\phi^{CNN}(\{\tilde{R}\}))$$

- $\mathcal{E}_\phi^{CNN}$: 一个 shared CNN backbone (类似 LDM VAE，downsample factor $f=8$)，每个 view 独立 downsample 成 patch tokens
- $\mathcal{E}_\phi^{TX}$: 5 层 multi-view transformer，所有 views 的 tokens 全部 attend（full 3D attention）

这一步产出 unstructured set latent $\mathbf{z}_z \in \mathbb{R}^{V \times (H/f) \times (W/f) \times C}$。对于 $V=8, H=W=512, f=8$，得到 $8 \times 64 \times 64 = 32768$ 个 tokens。

**这里有一个关键 observation**：32768 这个数太大，直接做 flow matching 的话 attention 成本 $\mathcal{O}(N^2) \approx 10^9$，扛不住。而且这些 tokens 还是 view-indexed，不是 native 3D，会有 multi-view inconsistency 问题。所以必须再压一次。

### 3.3 Point Cloud-structured Latent: 把 set "concretize" 成 point cloud

这是论文的核心 trick。用 cross attention 把 $\mathbf{z}_z$ 投影到一个 sparse point cloud 上：

$$\mathbf{z}_h := \text{CrossAttn}(\text{PE}(\mathbf{z}_x), \mathbf{z}_z, \mathbf{z}_z)$$

变量含义：
- $\mathbf{z}_x \in \mathbb{R}^{3\times N}$: 用 Farthest Point Sampling (FPS) 从 input 3D shape 表面采样的 sparse point cloud，$N=768$
- $\text{PE}(\cdot)$: positional embedding (Fourier features, Tancik et al. 2020)，把 3D 坐标编码成高维频率特征
- Query $Q = \text{PE}(\mathbf{z}_x)$: 768 个 query point 的位置编码
- Key, Value: 都是 $\mathbf{z}_z$ (来自 multi-view encoder 的 set latent)
- Output $\mathbf{z}_h \in \mathbb{R}^{C_h \times N}$, $C_h=10$: 每个 3D 点上聚合的 feature

直觉：这是一个 **"read" cross attention**——768 个 3D point "询问" 32768 个 view tokens："你们看到了我附近的什么信息？" 每个点 gather 周围多视角的信息。本质上是从 multi-view observation 回到 3D world 的 inverse projection。

最终 latent code：
$$\mathbf{z} = [\mathbf{z}_x \oplus \mathbf{z}_h] \in \mathbb{R}^{(3+C_h)\times N} = \mathbb{R}^{13\times 768}$$

这个 $\mathbf{z}$ 同时具备三个性质：
1. **Compact** (768 个 token，比 32768 小 43x)
2. **3D-native** (每个 token 有显式 3D 坐标)
3. **Editable** (改 $\mathbf{z}_x$ 就是改 geometry，改 $\mathbf{z}_h$ 就是改 texture，二者 disentangle)

这就是 paper 标题里 "Interactive Point Cloud Flow Matching" 的来源。

### 3.4 Decoder: DiT + Cascaded Upsampler → Surfel Gaussians

Decoder 接到 $\mathbf{z}$ 之后要把它变回 dense 3D Gaussians 用于渲染。结构：

**Step 1: DiT transformer** (Eq 3)
$$\tilde{\mathbf{z}} := \mathcal{D}_T(\text{MLP}(\mathbf{z}))$$

MLP 先把 13-dim 投影到 working dimension，然后 DiT-B/2 (Peebles & Xie 2023 风格) refine。这一步保持 token 数 N=768 不变。

**Step 2: Cascaded Upsampler** (Eq 4) — 这一步让 point 数量从 768 涨到 ~74k

每个 upsampler block 接收当前 $N$ 个 tokens，输出 $f_u \cdot N$ 个 tokens。具体做法：

$$\mathbf{z}_i^{(k+1)} := \mathcal{D}_U^k([\mathbf{z}_u \oplus \tilde{\mathbf{z}}_i])$$

变量含义：
- $\mathbf{z}_u \in \mathbb{R}^{f_u \times C}$: learnable embedding，是 upsampling ratio 个可学习 token
- $\tilde{\mathbf{z}}_i$: 第 $i$ 个 input token
- $[\mathbf{z}_u \oplus \tilde{\mathbf{z}}_i] \in \mathbb{R}^{(f_u+1)\times C}$: 把 learnable tokens prepend 到 input token 前
- $\mathcal{D}_U^k$: 2 个 transformer blocks
- Output $\mathbf{z}_i^{(k+1)} \in \mathbb{R}^{f_u \times C}$: 一个 input token 被裂变成 $f_u$ 个 output tokens

直觉：这其实是 **token-level "split and refine"**。每个 input point 上挂 $f_u$ 个 learnable sub-tokens，transformer 让这些 sub-tokens 互相 attend（也包括 attend 邻居 input token 的 sub-tokens），从而决定怎么"分裂"成 $f_u$ 个新 point。

论文 cascade 三次，$f_u = 8, 4, 3$：
$$768 \xrightarrow{\times 8} 6144 \xrightarrow{\times 4} 24576 \xrightarrow{\times 3} 73728 \text{ Gaussians}$$

每个 Gaussian 是 13-dim surfel Gaussian attribute (Huang et al. 2024a)：center $\mathbf{p}_k$, tangents $\mathbf{t}_u, \mathbf{t}_v$, scales $s_u, s_v$, opacity $\alpha$, view-dependent color $\mathbf{c}$。

**一个很 nice 的副产品**：这三个 LoD (Level of Detail) 层级自然形成 coarse-to-fine 渲染。手机端可以用 LoD=1 快速预览，桌面端可以用 LoD=3 高质量渲染。

### 3.5 VAE Training Objective (Eq 5)

$$\mathcal{L}(\phi, \psi) = \mathcal{L}_{\text{render}} + \mathcal{L}_{\text{geo}} + \lambda_{\text{kl}}\mathcal{L}_{\text{KL}} + \lambda_{\text{GAN}}\mathcal{L}_{\text{GAN}}$$

- $\mathcal{L}_{\text{render}} = \mathcal{L}_1 + \text{VGG loss}$: 渲染图 vs GT 渲染图
- $\mathcal{L}_{\text{geo}} = \lambda_d \mathcal{L}_d + \lambda_n \mathcal{L}_n$: 几何正则
- $\mathcal{L}_{\text{KL}}$: KL-reg (Kingma 2013)，让 latent 接近 $\mathcal{N}(0, I)$，便于后面 flow matching
- $\mathcal{L}_{\text{GAN}}$: discriminator 提升 perceptual fidelity

Geometry loss 详情：

**Depth distortion** (Eq 8, from Mip-NeRF):
$$\mathcal{L}_d = \sum_{i,j} \omega_i \omega_j |d_i - d_j|$$
- $\omega_i$: 第 $i$ 个 Gaussian 与 ray 交点的 blending weight（基于 alpha compositing）
- $d_i$: 第 $i$ 个交点的 depth
- 直觉：让沿 ray 的 weight distribution 集中、不要散开

**Normal consistency** (Eq 9):
$$\mathcal{L}_n = \sum_i \omega_i (1 - \hat{N}_i^T N)$$
- $\hat{N}_i$: 第 $i$ 个 surfel 的 normal
- $N$: GT normal map
- 直觉：让 surfel 法向跟 GT 表面法向对齐

权重: $\lambda_{\text{kl}}=2e{-6}, \lambda_d=1000, \lambda_n=0.2, \lambda_{\text{GAN}}=0.1$。注意 $\lambda_d$ 很大，说明 depth distortion 是强约束。

每个 iteration 随机选一个 LoD 监督（节省 compute），同时 input views 和 random novel views 都监督。

---

## 4. Cascaded Flow Matching: 为什么 disentangle 这么重要

### 4.1 Flow Matching Background (Appendix A.3)

Flow matching (Lipman et al. 2023, Liu et al. 2023 rectified flow, Albergo et al. 2023 stochastic interpolants) 可以看作是 DDPM 的一个 special case，用一个 **straight-line** path 连接 data 和 noise：

$$z_t = (1-t) x_0 + t \epsilon, \quad t \in [0, 1]$$

相比 DDPM 的 curved Markov chain，flow matching 的 path 是直的，ODE 求解器（如 Euler）走几步就能 sample，效率高。

General weighted objective (Eq 13/27):
$$\mathcal{L}_w(x_0) = -\frac{1}{2} \mathbb{E}_{t\sim\mathcal{U}(t), \epsilon\sim\mathcal{N}(0,I)} \left[ w_t \lambda_t' \|\epsilon_\Theta(z_t, t) - \epsilon\|^2 \right]$$

变量含义：
- $t$: timestep，从 $\mathcal{U}(0,1)$ 均匀采样
- $\epsilon \sim \mathcal{N}(0,I)$: 标准 normal noise
- $z_t = a_t x_0 + b_t \epsilon$: forward noised latent
- $\lambda_t = \log(a_t^2 / b_t^2)$: signal-to-noise ratio (SNR)
- $\lambda_t' = 2(a_t'/a_t - b_t'/b_t)$: SNR 对 $t$ 的导数
- $w_t$: 时间步权重，flow matching 选 $w_t^{\text{FM}} = -\frac{1}{2}\lambda_t' b_t^2$
- $\epsilon_\Theta$: 神经网络预测的 noise

推导路径（Eq 14-26）就是从 conditional flow matching $\mathcal{L}_{\text{CFM}} = \mathbb{E}\|v_\Theta - u_t\|^2$ 走到 noise-prediction form，最后等价于 weighted diffusion loss。这跟 Karras et al. 2022 (EDM) 的 unified framework 是一致的。

实现细节：用了 SiT (Ma et al. 2024) 的 training recipe — **pred-v objective, GVP schedule, uniform t sampling**。GVP 是 Generalized Variance Preserving，是 SNR 调度的选择。

### 4.2 为什么 Cascade: Geometry-Texture Disentanglement

如果直接联合建模 $\mathbf{z} = [\mathbf{z}_x \oplus \mathbf{z}_h]$，模型要把 geometry 和 texture 一起生成。问题：
- Geometry 是 low-DOF 的结构信息，texture 是 high-DOF 的 appearance
- 联合建模时 texture 的 noise 会 leak 进 geometry，反之亦然
- 无法做 "保持 geometry 换 texture" 或 "改 geometry 但 texture consistent" 这种编辑

GAUSSIANANYTHING 的解决方案：**两个独立的 flow matching model，cascade 起来**。

**Stage-1: Point Cloud Diffusion** (Eq 6)
$$\mathcal{L}_w^x(x_0) = -\frac{1}{2} \mathbb{E}_{t, \epsilon} \left[ w_t^{\text{FM}} \lambda_t' \|\epsilon_\Theta^x(\mathbf{z}_{x,t}, t, c) - \epsilon\|^2 \right]$$

只对 $\mathbf{z}_x$ (768×3) 做 diffusion，condition 在 text/image prompt $c$ 上。生成 coarse geometry layout。

**Stage-2: Point Feature Diffusion** (Eq 7)
$$\mathcal{L}_w^h(x_0) = -\frac{1}{2} \mathbb{E}_{t, \epsilon} \left[ w_t^{\text{FM}} \lambda_t' \|\epsilon_\Theta^h(\mathbf{z}_{h,t}, \mathbf{z}_{x}, t, c) - \epsilon\|^2 \right]$$

对 $\mathbf{z}_h$ (768×10) 做 diffusion，**额外 condition 在 clean point cloud $\mathbf{z}_x$ 上**（不是 noisy 的 $\mathbf{z}_{x,t}$，而是 stage-1 输出的 $\mathbf{z}_{x,0}$）。

cascade 方式（Section 3.2 末段）：把 stage-1 输出 $\mathbf{z}_{x,0}$ 经过 PE 编码后，**add 到 stage-2 第一层 feature 上**。这保证 stage-2 每个 token 都 "知道" 自己绑定的 3D 坐标。

直觉类比：这跟 Imagen (Ho et al. 2021) 的 cascaded diffusion 思想类似——先低分辨率 base model 出大致结构，再 super-resolution model 加细节。但这里 cascade 的不只是分辨率，而是 **representation modality**：先出 geometry，再 condition 在 geometry 上出 texture。

Ablation (Fig 6a) 直接证实：单 stage joint training 的 texture 明显 worse，3D shape 也有 artifacts。

### 4.3 Conditioning: Text 和 Image 不同位置

DiT block 设计 (Fig 2)：
- **AdaLN-single** (PixArt-α, Chen et al. 2023): timestep embedding 通过 single adaptive layer norm 注入
- **QK-Norm** (Dehghani et al. 2023, Esser et al. 2021): attention 的 Q 和 K 各自做 LayerNorm，稳定训练
- **Cross-attention** 注入 condition

Text condition: CLIP (Radford 2021) penultimate tokens
Image condition: DINOv2 (Oquab 2023) global + patch features

区别在于 cross-attention 注入位置：
- Text: cross attention 在 DiT block 中间
- Image: cross attention 位置不同（看 Fig 2a vs 2b）

CFG (Ho & Salimans 2021): 10% drop condition 概率，sampling 时 CFG=4，250 ODE steps。

DiT 配置：24 layers, 16 heads, 1024 hidden dim, 458M params。BF16 + FlashAttention (Dao 2024)。

---

## 5. 实验：关键数据解读

### 5.1 Text-to-3D (Table 1)

| Method | CLIP ViT-B/32 ↑ | ViT-L/14 ↑ | MUSIQ-AVA ↑ | Q-Align ↑ |
|---|---|---|---|---|
| Point-E | 26.35 | 21.40 | 4.08 | 1.21 |
| Shape-E | 27.84 | 25.84 | 3.69 | 1.56 |
| LN3Diff | 29.12 | 27.80 | 4.16 | 2.22 |
| 3DTopia | 30.10 | 28.11 | 3.31 | 1.42 |
| **Ours** | **31.80** | **29.38** | **4.99** | **3.13** |

GAUSSIANANYTHING 在所有指标上都领先。Q-Align 是 LMM-based visual scoring，更接近 human judgment，3.13 vs LN3Diff 2.22 是显著差距。

### 5.2 Image-to-3D (Table 2)

最 striking 的对比：

| Method | CLIP-I ↑ | FID ↓ | P-FID ↓ | P-KID ↓ |
|---|---|---|---|---|
| LGM (V=4) | 87.99 | **19.93** | 32.37 | 12.44 |
| LN3Diff | 87.24 | 29.08 | 27.17 | 10.02 |
| **Ours** | **89.06** | 24.21 | **8.72** | **3.22** |

注意 LGM 在 FID 上领先（19.93 vs 24.21），但 P-FID (3D shape quality) **差 4 倍**（32.37 vs 8.72）。这证实了 paper 的论点：multi-view → 3D 的 cascaded pipeline (LGM 依赖 MVDream) 视觉质量好但 3D geometry 烂，native 3D diffusion 在 3D 一致性上完胜。

### 5.3 Gaussian Utilization Ratio (Table 4)

| Method | High-opacity Gaussians (%) |
|---|---|
| Splatter Image | 17.14 |
| LGM | 52.63 |
| **Ours** | **96.84** |

这是一个被低估的指标。Pixel-aligned Gaussian 预测方法（每个像素 spawn 一个 Gaussian）大量浪费 Gaussians 在空白背景和多视角重叠区域。GAUSSIANANYTHING 因为是 surface-aligned 的 point cloud → Gaussian，几乎 97% 都有效。这意味着实际 rendering 速度和内存占用都有 ~2x 优势。

### 5.4 Ablation (Table 3)

| Design | LPIPS@100K |
|---|---|
| Dense PCD as Input | 0.174 |
| Multi-view RGB-D as Input | 0.163 |
| + Normal Map | 0.157 |
| + Gaussian SR Module | 0.095 |
| + 3× Gaussian SR Module | 0.067 |

每个 design choice 都有 measurable gain。最 dramatic 的是 upsampler module (0.157 → 0.067)。

---

## 6. 3D Editing: 这是 latent structure 设计的 payoff

Fig 5 和 Fig 6b 展示了 3D editing 能力：

**Texture editing**: 用同一个 $\mathbf{z}_{x,0}$ (stage-1 输出)，给 stage-2 不同的 random seed，得到同样 geometry 但不同 texture 的 3D object。

**Geometry editing**: 直接修改 $\mathbf{z}_{x,0}$（比如把消防栓盖子 disjoint 一段距离），然后 re-run stage-2 with same Gaussian noise，得到结构修改但 texture reasonable 的结果。

对比：直接编辑 dense 3D Gaussians 会导致 tearing artifacts（Fig 6b 上排），因为 Gaussian 之间没有 semantic binding。在 latent space 编辑则 holistic——VAE decoder 会重新 decode 出 consistent 的 Gaussian set。

这跟 2D 中的 DragGAN/DragonDiffusion 是同一个 idea：**在 latent manifold 上 edit，让 generative prior 帮你 "fill in the blanks"**。

---

## 7. 关键 Insights 和我的思考

**Insight 1: Latent space design 决定 generative model 的能力上限。** 2D LDM 的成功很大程度归功于 VAE latent 的 spatial correspondence。3D 一直缺这个，GAUSSIANANYTHING 用 point cloud 作为 latent 是一个非常 natural 的选择——它就是 3D 的"pixel grid"。

**Insight 2: Cross attention 是 representation converter。** $\mathbf{z}_z$ (unstructured set) → $\mathbf{z}_h$ (point-structured) 是用 cross attention 完成的。这其实是一个 general pattern——CLAY/3DShape2VecSet 也用 cross attention 把 set 投到 learnable query 上。区别在于 GAUSSIANANYTHING 的 query 是 **有空间意义的 point cloud**，而不是 arbitrary learnable tokens。

**Insight 3: Cascaded diffusion for disentanglement。** 这是 paper 最 valuable 的 design choice。把 geometry 和 texture 分两阶段建模，不仅仅是为了 quality，更为了让 latent $\mathbf{z}_x$ 和 $\mathbf{z}_h$ 在生成时 disentangle。这种 disentanglement 让 editing 成为可能。

**Insight 4: Surfel Gaussian > Triplane for rendering。** LN3Diff 用 triplane + volumetric rendering，高分辨率时 memory 爆炸。2DGS (surfel) 是 surface-based，渲染效率和 surface quality 都更好。配上 cascaded upsampler，可以做 LoD trade-off。

**Insight 5: Multi-view RGB-D-N > Dense point cloud as VAE input。** 这个选择有两个 benefit：(a) 信息更 comprehensive（texture + geometry + normal）；(b) dataset 可扩展性更好（任何 3D source 都能渲染）。

**潜在 weakness** (paper 自己也承认):
1. Texture 质量还比不过 LGM 这种 multi-view cascaded pipeline
2. 没用 2D pretraining prior，所以 generalization 受限于 3D dataset size
3. 768 points 对于复杂 object 可能不够（fine geometry detail 丢失）
4. DiT-B/2 decoder 比 full DiT-XL 弱（VRAM constraint）

**Concurrent work 对比**:
- **Trellis** (Xiang et al. 2024): 用 sparse voxel 替代 point cloud 作为 latent，类似 cascaded flow matching 框架。Sparse voxel 在编辑上也很 intuitive，可能比 point cloud 更适合 complex topology。
- **AtlasGaussians** (Yang et al. 2025): 也是 native 3D GS diffusion，但没有 explicit latent space，不能做 interactive editing。

---

## 8. Architecture 图解析 (Fig 1 & Fig 2)

**Fig 1 (VAE pipeline)** 从左到右：
1. Multi-view RGB-D-N renderings $\mathcal{R}$ (8 views)
2. CNN backbone (per-view) → tokens
3. Multi-view Transformer (full 3D attention) → unstructured set latent $\mathbf{z}_z$
4. Cross attention with FPS-sampled sparse point cloud $\mathbf{z}_x$ → point-structured latent $\mathbf{z} = [\mathbf{z}_x \oplus \mathbf{z}_h]$
5. DiT transformer decoder $\mathcal{D}_T$
6. Cascaded upsampler $\mathcal{D}_U^1, \mathcal{D}_U^2, \mathcal{D}_U^3$ → dense surfel Gaussians
7. Differentiable rasterization → renderings
8. Loss: render + geo + KL + GAN

**Fig 2 (Diffusion training)**:
- (a) Text condition: CLIP tokens → cross attention → DiT block (AdaLN-single + QK-Norm)
- (b) Image condition: DINOv2 global + patch features → cross attention at different position
- (c) Two-stage cascade: Stage-1 generates $\mathbf{z}_{x,0}$, Stage-2 condition on $\mathbf{z}_{x,0}$ generates $\mathbf{z}_{h,0}$, concat → VAE decoder → 3D object

---

## 9. 总结：这篇 paper 的真正贡献

在我看来，GAUSSIANANYTHING 的核心贡献是把 2D LDM 的"editable latent" philosophy 完整 transfer 到 3D：

1. **找到 3D 的 "spatially-corresponded latent"**: point cloud-structured latent
2. **解决 disentanglement**: cascaded flow matching 把 geometry 和 texture 分开
3. **解决 rendering quality**: surfel Gaussian + cascaded upsampler
4. **解决 scalability**: multi-view RGB-D-N input 让 dataset 可扩展

这四点加起来，让 3D native diffusion 第一次在 quality、editability、scalability 三个维度同时达到可用水平。

值得 follow-up 的方向（paper Section B 自己列了）：
- 增大 latent point 数量 (768 → 几千)
- Diffusion training 加 rendering loss (类似 DMV3D)
- 用 2D pretraining prior (类似 LGM 用 MVDream)
- 引入 PBR rendering (3DGRT, Moenne-Loccoz et al. 2024)
- 用 Objaverse-XL (10M+) 扩 dataset

我个人觉得最 promising 的方向是 **用 Trellis 的 sparse voxel 替代 point cloud**——sparse voxel 有 octree structure，可以 hierarchical 编辑，可能比 flat point cloud 更 expressive。但 point cloud 的好处是更轻量、更接近 artist 的 mental model。这两种 representation 谁能成为 3D 的 "pixel grid"，还是一个 open question。

---

## Reference 链接汇总

- **GAUSSIANANYTHING**: https://nirvanalan.github.io/projects/GA/
- **Stable Diffusion / LDM**: https://arxiv.org/abs/2112.10752
- **LION (latent point diffusion)**: https://arxiv.org/abs/2210.10695
- **CLAY**: https://arxiv.org/abs/2406.04459
- **3DShape2VecSet**: https://arxiv.org/abs/2305.18339
- **LN3Diff**: https://arxiv.org/abs/2403.07957
- **2D Gaussian Splatting**: https://arxiv.org/abs/2403.17888
- **3D Gaussian Splatting**: https://arxiv.org/abs/2308.14737
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **Rectified Flow**: https://arxiv.org/abs/2209.03003
- **Stochastic Interpolants**: https://arxiv.org/abs/2303.08797
- **SiT**: https://arxiv.org/abs/2401.08740
- **Stable Diffusion 3 / Rectified Flow Transformers**: https://arxiv.org/abs/2403.03206
- **PixArt-α**: https://arxiv.org/abs/2310.00426
- **DiT**: https://arxiv.org/abs/2212.09748
- **Trellis**: https://arxiv.org/abs/2412.01506
- **Point-E**: https://arxiv.org/abs/2212.08751
- **Shape-E**: https://arxiv.org/abs/2305.02463
- **SRT (Scene Representation Transformer)**: https://arxiv.org/abs/2111.11260
- **RUST**: https://arxiv.org/abs/2303.12279
- **MCC (Multi-view Compressive Coding)**: https://arxiv.org/abs/2301.08247
- **LRM**: https://arxiv.org/abs/2311.04400
- **LGM**: https://arxiv.org/abs/2402.05063
- **MVDream**: https://arxiv.org/abs/2308.16512
- **CLIP**: https://arxiv.org/abs/2103.00020
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **Mip-NeRF 360 (distortion loss)**: https://arxiv.org/abs/2111.12021
- **EDM (Karras 2022)**: https://arxiv.org/abs/2206.00364
- **Plücker coordinates (Light Field Networks)**: https://arxiv.org/abs/2106.02620
- **Fourier Features (Tancik et al.)**: https://arxiv.org/abs/2006.10739
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **QK-Norm (Scaling ViT to 22B)**: https://arxiv.org/abs/2302.05442
- **Pre-norm Transformer (Xiong et al.)**: https://arxiv.org/abs/2002.04745
- **PointNet++**: https://arxiv.org/abs/1706.02413
- **G-Objaverse**: https://aigc3d.github.io/gobjaverse/
- **Objaverse**: https://objaverse.allenai.org/
- **Cap3D**: https://arxiv.org/abs/2306.07279
- **GSO (Google Scanned Objects)**: https://arxiv.org/abs/1904.03167
- **Q-Align**: https://arxiv.org/abs/2312.17090
- **MUSIQ**: https://arxiv.org/abs/2108.05952
- **DragonDiffusion**: https://arxiv.org/abs/2307.02421
- **DragGAN**: https://arxiv.org/abs/2305.10974
- **Cascaded Diffusion Models (Ho et al.)**: https://arxiv.org/abs/2106.15282
- **Classifier-free Guidance**: https://arxiv.org/abs/2207.12598
- **Direct3D**: https://arxiv.org/abs/2407.02244
- **AtlasGaussians (ICLR 2025)**: https://arxiv.org/abs/2412.02009
- **Geometric Distributions (Zhang et al.)**: https://arxiv.org/abs/2411.16076

希望这些拆解能帮你在脑中 build 起这个 framework 的 intuition。整个 paper 的 narrative 很清晰：找到 3D 的 editable latent manifold → 用 cascaded diffusion disentangle geometry/texture → 用 surfel Gaussian 实现 high-quality rendering。如果未来 3D generation 有 "Stable Diffusion moment"，我觉得这条 path 是最有可能性的候选之一。
