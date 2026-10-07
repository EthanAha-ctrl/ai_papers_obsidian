---
source_pdf: Seed3D 1.0.pdf
paper_sha256: bd6802be8665af2acfc88549b4eaf5f1d5b1d2da5c610e1cc066d0dc1d4cd8ab
processed_at: '2026-08-12T04:27:33-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Seed3D 1.0 用人话说

## 一句话总结

给一张图，给你一个能直接塞进 Isaac Sim 跑物理仿真的 3D 物体 —— watertight geometry + PBR material + UV texture，robotic manipulation 直接能跑。

---

## 为什么搞这个

embodied AI 卡在 data 上。RL 在 coding domain 已经证明 "interactive environment 能解决 data scarcity"（DeepSeek-R1 那套），但 robotic 想复刻这个 paradigm，得先有 high-fidelity simulation environment。

问题来了：
- **Video world model**（Cosmos, Genie-3）：生成内容 diverse，但没 3D consistency，没法给 embodied agent 提供真实的 spatial feedback
- **Physics engine**（IsaacGym）：dynamics 准，但 asset 全靠人工做，太贵，diversity 上不去

所以缺一个东西：能 scalable 生成又能进 physics engine 的 3D content generator。Seed3D 就来填这个坑。

---

## 整体怎么干

一条流水线，5 步，每步一个 sub-model：

```
图 → geometry → multi-view RGB → PBR material → UV 补全 → 完整 asset
```

为什么分这么多步？因为 end-to-end 训练 gradient 信号太乱。Decoupling 让每个 stage 解决一个明确 sub-problem，每步都好 debug、好 scale、好 curate training data。

---

## 第 1 步：生成几何 — Seed3D-VAE + Seed3D-DiT

### VAE 干嘛的

把 mesh 压成一堆 latent tokens，decoder 能从任意 3D 点 query 出它的 TSDF 值。

**关键 trick 1：用 TSDF 不用 SDF**

SDF 在远离 surface 的地方值趋于无穷，模型大部分 capacity 浪费在拟合那些 trivial 的远距离值。TSDF 把距离截断，只在 surface 附近有意义，模型 focus 在真正重要的地方。

**关键 trick 2：set-based latent**

latent tokens 没有 positional encoding 绑定，是 permutation-invariant 的 set。好处：训练时随机 sample token 数量 256~4096，inference 时简单 shape 用少 token 省算力，复杂 shape 用多 token 保细节。同一个 decoder 都能处理。

**关键 trick 3：hybrid point sampling**

uniform 点采 global shape，edge 点采 sharp features。纯 uniform 在 sharp edges 上点太稀疏，纯 edge 丢失 overall shape。两个都要。

Encoder 把 point cloud 压成 fixed-length latent set，decoder 把任意 query point 映射到 TSDF value。continuous field decoder，没有 voxel resolution bottleneck。

### DiT 干嘛的

在 latent space 上跑 rectified flow，从 noise 到 structured latent，conditional on input image。

**关键 trick 4：dual image encoder**

DINOv2 给 semantic features，RADIO 给 geometric features（depth / normal 能力，通过 distill 多个 vision foundation model 来的）。single image 有 inherent depth ambiguity，加 RADIO 帮 DiT 做更合理的 3D inference。

**关键 trick 5：FLUX hybrid DiT**

double-stream blocks 让 shape token 和 image token 用各自参数处理（保留 modality inductive bias），attention 时 concatenate 做 cross-modal interaction。single-stream blocks 再 fuse。

直觉：早期各 modality 有自己的 structure，分开处理；中间 attention 让 information flow；后期统一 refine。

**关键 trick 6：length-aware timestep shift**

长 latent sequence（4096 tokens）信息容量大，同样 noise level 可能不足以 disrupt 所有 structure。所以 noise schedule 根据 sequence length scale，长序列加更多 noise。

**训练 3 阶段**：PT（256 tokens, full data）→ CT（4096 tokens, full data, enhanced aug）→ SFT（curated high-quality subset, low LR）。典型 coarse-to-fine + quality boost。

---

## 第 2 步：生成 multi-view RGB — Seed3D-MV

给 generated mesh + reference image，生成多视角一致的 RGB 图像。

**为什么不用 prior work**

- Multi-view attention methods 需要额外 ControlNet / MVAdapter，parameter overhead 大
- UniTEX 直接 fine-tune pretrained DiT 但 base 没设计成 multi-view，in-the-wild image 效果差

**Seed3D 的做法**

基于 MMDiT（SD3 架构），用 **in-context multi-modal conditioning** —— noisy target tokens 和 clean condition tokens（geometry / reference image / text）沿 sequence dimension concatenate，让 attention 自然 learn correspondence。

这个比 ControlNet 的 side-branch + zero-conv 注入优雅多了。ControlNet 是"硬连接"，in-context 是让 attention 自己 learn 该 attend 哪里。

**关键 trick 7：cross-modal RoPE**

token sequence 组织成 `[multi-view noisy, geometry, reference image, text]`。

发现：noisy tokens 和 geometry tokens 用 **separate** spatial positions 比 shared 效果好。虽然同一视角的 noisy image 和 geometry image 在 pixel grid 上对齐，但语义维度不同（RGB vs normal/CCM），分开 spatial encoding 让 RoPE 更精细表达"同位置但不同 modality"。

**关键 trick 8：resolution-aware shift-SNR**

multi-view 大幅增加 sequence length（6 views × 4096 tokens = 24K tokens），高分辨率时高频率 noise 占更多 signal energy，传统 SNR schedule 会 over/under-noise。Shift-SNR 让 schedule 适应。

---

## 第 3 步：分解 PBR material — Seed3D-PBR

multi-view RGB → albedo + metallic + roughness。

**为什么 estimation 不用 generation**

high-quality PBR training data 稀缺（互联网上 PBR-labeled 3D asset 远少于 RGB-labeled）。Estimation 从 multi-view RGB 反推（inverse rendering 思路），RGB data 多，且分解任务比生成更 constrained。

**关键 trick 9：two-stream network**

albedo 和 MR 物理 / 视觉特性差异极大：
- albedo：连续 RGB，diffuse，low-frequency
- metallic：近似 binary，surface reflectance 性质突变
- roughness：连续 scalar，控制 specular lobe 宽度

完全 shared network 会让各 modality 特征在 attention 中互相 wash out。完全 separate network parameter 翻倍且 multi-view consistency 难保证。

折中：每个 DiT block 内 Q/K/V projection 分 modality instantiate（让各 modality 提取自己关心的 pattern），attention 和 FFN 共享（省 parameter + 保证 consistency）。加 learnable modality embeddings 区分。

**dual-level conditioning**：
- Global：CLIP feature 替换 text embedding，控制整体 appearance
- Local：reference image VAE latent 和 noise latent channel concat，只 feed 到 first DiT block，pixel-level 对齐

---

## 第 4 步：补全 UV texture — Seed3D-UV

multi-view bake 到 UV 空间会有 holes —— limited view coverage + self-occlusion，某些 mesh 区域在所有 view 下都不可见。

**关键 trick 10：coordinate conditioning**

UV 空间的"距离"≠ mesh surface 的"距离"。UV 上相邻 pixel 可能对应 mesh 上相隔很远的 surface point（UV seam 处）。Naive image inpainting 把 UV map 当普通 2D image 会乱填。

把 UV coordinate maps 作为 positional tokens 加入 DiT，模型知道每个 UV pixel 对应的 3D surface coordinate，生成 spatially coherent completion。UV boundary 处 transition 更 sharp，和 mesh geometry alignment 更好。

---

## 第 5 步：inference 加速 — hierarchical extraction

Dual Marching Cubes 要在 dense grid 上 query SDF。全 grid full-precision 太慢（512³ = 134M points）。

先 bfloat16 coarse evaluation 找出"可能含 zero-crossing"的 active cells，inactive cells 直接 prune。只对 active cells 做 float32 full precision。compute 大幅减少，mesh fidelity 保留。

DMC vertex placement 需要 SDF gradient（法向量）。用 auto-diff 直接 backprop through VAE decoder 拿 exact gradient，thin features 保留得比 numerical gradient 好。

---

## Data pipeline 是重头戏

3D data 处理比 2D 难一个数量级。格式乱（OBJ/FBX/GLTF/PLY/proprietary），coordinate system 不一致，质量参差不齐，大量 corrupted geometry。

7 步预处理：
1. **Sourcing**：公共 repo + licensed marketplace + synthetic
2. **Format standardization**：全转 GLB
3. **Geometric dedup**：4-view render + DINOv2 feature + FAISS similarity search，dual-threshold filtering
4. **Orientation canonization**：4-view rendering → orientation classifier → 预测 canonical pose → apply transform
5. **Quality filtering**：2-stage，aesthetic scoring + VLM (Qwen2.5-VL) 评估 quality / category / data type。排除 real-world scanned（噪声多 topology messy）和 scene-level（多物体组合）
6. **Multi-view rendering**：Blender Cycles PBR，随机 viewpoint + HDRI lighting
7. **Remeshing**：CUDA-based 4-stage——voxelization + signed distance floodfill + thin-structure preservation + Dual Marching Cubes，把任意 mesh 转 watertight

watertight 是 physics simulation 的 prerequisite。Non-watertight mesh 在 physics engine 里 collision detection 失败，mass calculation 错误。

---

## Training infra 也不含糊

- **HSDP**：node 内 DP + node 间 FSDP，最小化 cross-node communication（跨 NVLink/IB 比 node 内 NVLink 慢得多）
- **MLAC**：selective activation checkpointing + CPU offload + async prefetch，比 full checkpointing 省 memory 且 recompute overhead 小
- **Kernel fusion**：torch.compile + custom CUDA，profiling 发现 memory-bound ops 是 bottleneck，fuse element-wise ops
- **Fault tolerance**：pre-launch health check + NCCL flight recorder + centralized monitoring (ETTR)

---

## 结果怎么样

### Geometry

1.5B 参数，ULIP-T/I 和 Uni3D-T/I 4 个 metric 全 SOTA，超过 3B 的 Hunyuan3D-2.1。

### Texture

Multi-view generation 和 PBR estimation 都 SOTA。Seed3D 1.0*（用 GT multi-view images 做 PBR input）显示 upper bound——证明 Seed3D-PBR 本身强，但被 multi-view generation error bottleneck。

### Application

直接进 Isaac Sim 做 robotic manipulation。VLM 估 scale 调到 real-world dimensions，Isaac Sim 自动从 watertight geometry 生成 collision mesh + default friction，无需 manual tuning。

Scene generation 用 factorized approach：VLM 做 layout planning（擅长），Seed3D 做 per-object generation（擅长），然后 geometric placement。把 end-to-end scene generation 拆成几个 sub-problem，每个都有成熟 tool。

---

## 为什么这工作重要

从 Karpathy 视角看几个有意思的点：

**1. VecSet latent 很 elegant**

permutation-invariant + length-agnostic，inference 时能灵活 allocate compute，类似 LLM 的 variable-length sequence。这个 representation 以后可能成 3D 生成标配。

**2. In-context conditioning 是 trend**

Flux.1 Kontext 范式——condition tokens 和 noisy tokens 沿 sequence concat，让 attention 自然 learn correspondence。比 ControlNet 的 side-branch + zero-conv 注入优雅。Diffusion model conditioning 的 future 方向大概率往这走。

**3. Two-stream PBR 是 parameter-efficient multi-modal design 典范**

该分开的地方分开（Q/K/V），该共享的地方共享（attention core, FFN）。类比 MoE 但更轻量。这个 pattern 在其他 multi-modal 任务里也能用。

**4. Estimation over generation for PBR**

RGB data 多，inverse rendering 比 direct generation 更 constrained。这个 insight 在 data-scarce domain 通用——先想清楚你有什么 data，再选 paradigm。

**5. Production-grade 全栈工程化**

从 data pipeline 到 training infra 到 inference 加速，每环都有工程细节。Data preprocessing 7-stage pipeline、Ray Data 异构 compute scheduling、HSDP + MLAC、hierarchical extraction + auto-diff gradient——没有哪个 stage 是随便搞搞。这是真正能跑 production 的 3D generation system。

---

## 一句话 takeaway

Seed3D 1.0 把 3D generation 从"好看就行"推进到"能进 physics engine 跑 sim"。配合 VLM 做 scale estimation + scene layout，从 single object 一路 scale 到 full scene。embodied AI 缺的 scalable simulation-ready content creation，这工作给了一个相当完整的 solution。

---

# Seed3D 1.0 — From Images to Simulation-Ready 3D Assets

ByteDance Seed 出的工作，核心定位：single image → simulation-ready 3D asset (watertight geometry + PBR materials + UV textures)，可以直接 drop 进 Isaac Sim 做 robotic manipulation。这事的根本动机是 embodied AI 的 data scarcity —— internet data 偏 text/2D，缺乏 spatial-physical 信息；同时 physics-based simulators (IsaacGym) 有准确 dynamics 但 manual asset creation 太贵。Seed3D 试图 bridge 这个 gap。

Official page: https://seed.bytedance.com/seed3d

---

## 1. 整体架构 — 5-stage sequential pipeline

Pipeline 设计的核心 insight 是 **解耦 (decoupling)**：把 3D 生成分成 geometry、multi-view appearance、PBR decomposition、UV completion 四个独立 stage，每个 stage 解决一个明确 sub-problem，避免 end-to-end 训练时 gradient 信号混乱。

```
Input Image
   ↓
[1] Seed3D-DiT + VAE decoder  → watertight mesh
   ↓
[2] Seed3D-MV  → multi-view RGB (consistent across viewpoints)
   ↓
[3] Seed3D-PBR → multi-view albedo + metallic-roughness
   ↓
[4] Seed3D-UV → complete UV texture maps (inpaint occluded regions)
   ↓
[5] Final integration → OBJ/GLB with PBR materials
```

为什么这样分？因为 geometry error 会 propagate 到 texture，texture 在 wrong geometry 上 bake 会失真；而 PBR 需要干净的 multi-view RGB 作 input；UV completion 必须在 mesh 的 UV parameterization 之上做。每个 stage 都有清晰的 input/output contract，便于独立 debug、独立 scale、独立 curate training data。

---

## 2. Geometry — Seed3D-VAE + Seed3D-DiT

### 2.1 Seed3D-VAE: Latent 3D shape representation

借鉴 3DShape2VecSet 和 Dora，核心 idea 是把 mesh 压成一个 **set of latent tokens** (permutation-invariant, length-agnostic)，然后 decoder 把任意 query point $x$ 映射到 continuous TSDF field。

**为什么用 TSDF 而非 SDF / occupancy？**

- **SDF** (full signed distance function)：在整个 $\mathbb{R}^3$ 上定义，$|d|$ 在远离 surface 处趋于无穷，regression loss 被远点 dominate，模型容量浪费在拟合 trivial 的距离值上。
- **Occupancy** (0/1 binary)：信息量太 coarse，丢掉 surface 附近 distance gradient，无法做 smooth mesh extraction。
- **TSDF** (truncated SDF)：只在 surface 邻域 $\{x : |d(x)| < \tau\}$ 内有意义，超出则 clip。Regression range bounded，模型 focus 在 surface 附近的 fine details 上。这对 mesh extraction 至关重要，因为 Dual Marching Cubes 只 care zero-crossing 附近的 gradient。

**Point sampling strategy**

输入 mesh 上采两类点：
- $P_u$：uniformly sampled points，capture global shape
- $P_s$：salient edge points，capture sharp features (creases, corners, boundary edges)

合并 $P = P_u \cup P_s$，每个点附带 surface normal $n$。

为什么 hybrid？纯 uniform sampling 在 sharp edges 上点密度太低，VAE 学不出 high-frequency edge features；纯 edge sampling 又丢失 overall shape。Hybrid 既保 global 又保 detail。

**Encoder (Equation 1)**

$$
\mathbf{Z}_0 = \text{CrossAttn}(\text{PE}(P), n_P), \quad \mathbf{Z}_i = \text{SelfAttn}(\mathbf{Z}_{i-1}), \quad i = 1, \ldots, L_e
$$

- $\text{PE}(P)$: Fourier positional encoding of point coordinates，把 $\mathbb{R}^3$ 坐标映射到 high-frequency spectral basis，解决 MLP 对 low-dim input 的 spectral bias 问题（Tancik et al. NeurIPS 2020）
- $n_P$: normals at sampled points
- $\text{CrossAttn}$: learnable query set $\{\mathbf{z}_m\}_{m=1}^M$ attend to point features，把 variable-length point cloud 压到 fixed-length latent set $\mathbf{Z}_0 \in \mathbb{R}^{M \times d}$
  - $M$: number of latent tokens (e.g., 256 / 512 / ... / 4096)
  - $d$: token embedding dimension
- $\text{SelfAttn}^{(i)}$, $i=1, \ldots, L_e$: $L_e$ self-attention layers 让 latent tokens 互相 exchange global context

**Decoder (Equation 2)**

$$
\hat{d}(x) = \text{MLP}\Big( \text{CrossAttn}\big( \text{SelfAttn}^{(j)}(\text{PE}(x)), \mathbf{Z} \big) \Big), \quad j = 1, \ldots, L_k
$$

- $x \in \mathbb{R}^3$: 任意 query point
- $\text{PE}(x)$: query point 的 Fourier encoding
- $\text{SelfAttn}^{(j)}$, $j=1, \ldots, L_k$: $L_k$ 层 self-attention refine query embedding
- $\text{CrossAttn}(\cdot, \mathbf{Z})$: refined query attends to latent set $\mathbf{Z} \in \mathbb{R}^{M \times d}$
- MLP: regression head 输出 predicted TSDF value $\hat{d}(x) \in \mathbb{R}$

关键 intuition：decoder 是 **continuous field decoder** —— 不在 fixed grid 上预测，而是任意 $x$ 都能 query。这避免了 voxel resolution bottleneck，做高分辨率 mesh extraction 时直接密集 sampling 就行。

**VAE Training (Equation 3)**

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}}
$$

- $\mathcal{L}_{\text{recon}}$: TSDF reconstruction loss，在 query points 上算 L1/L2 between $\hat{d}(x)$ 和 GT $d(x)$
- $\mathcal{L}_{\text{KL}}$: KL divergence between learned posterior $q(\mathbf{Z}|P)$ 和 prior $\mathcal{N}(0, \mathbf{I})$
- $\lambda_{\text{KL}} = 10^{-4}$: 很小，因为 priority 是 reconstruction fidelity，KL 只是 regularization

**KL warm-up**：$\lambda_{\text{KL}}$ 从小值 ramp up 到 target。为什么？如果一开始 $\lambda_{\text{KL}}$ 就大，posterior 会立即 collapse 到 prior，latent tokens 不携带 information (KL collapse 现象)，整个 VAE 退化成 unconditional auto-encoder。Warm-up 给 encoder 时间先学会 encode shape，再逐渐约束 latent distribution。

**Multi-scale training**：训练时随机 sample $M \in \{256, 512, \ldots, 4096\}$。关键 enabler 是 latent tokens 没有 positional encoding 绑定，是 permutation-invariant 的 set，所以同一个 decoder 处理任意长度 $M$ 都行。Inference 时简单 shape 用 256 tokens (省 compute)，复杂 shape 用 4096 (保 detail)。

---

### 2.2 Seed3D-DiT: Rectified flow on latent space

VAE 给了 latent space，DiT 在 latent space 上做 conditional generation。

**Image conditioning — dual encoder**

- **DINOv2** (Oquab et al.): self-supervised vision foundation model, strong semantic features
- **RADIO** (Ranzinger et al. CVPR 2024): "Agglomerative vision foundation model — Reduce All Domains Into One"，distill 多个 vision foundation models 到一个 network，提供 geometric understanding (depth, normal 等能力)

DINOv2 和 RADIO features 在 channel dimension concatenate 作为 conditioning。为什么需要 RADIO？因为 single image 有 inherent depth ambiguity —— 一张 2D 图无法唯一确定 3D。RADIO 通过蒸馏 depth estimator 等 multi-domain model，给 DiT 更强的几何先验，让它做更合理的 3D inference，同时提升 training stability (减少 mode collapse 到 degenerate flat shapes)。

**Transformer architecture — FLUX hybrid**

FLUX (Black Forest Labs) 的 hybrid design：
- **Double-stream blocks**: shape tokens 和 image tokens 用 **modality-specific parameters** (separate LayerNorm, QKV projection, MLP) 处理，但在 attention 时把两 modality 的 tokens concatenate 做 cross-modal interaction。
- **Single-stream blocks**: refined shape tokens 再过若干层 transformer layer，最后通过 VAE decoder 解码。

直觉：早期 modality 各自有 inductive bias (shape 有几何结构，image 有 spatial layout)，分开处理保留各自特征；中间 attention 让 cross-modal information flow；后期 single-stream 把 cross-modal info fuse 到 shape latent 上做 final refinement。

**Diffusion scheduling — rectified flow + length-aware shift**

- Flow matching framework with **velocity field prediction**：模型预测 velocity $v_\theta(\mathbf{Z}_t, t) = \frac{d\mathbf{Z}_t}{dt}$，从 noise $\mathbf{Z}_1$ 到 data $\mathbf{Z}_0$ 的 ODE trajectory 是 straight line (rectified flow, Lipman et al. ICLR 2023)。
- **Logit-normal timestep sampling**：$t \sim \text{LogitNormal}(\mu, \sigma)$，让中间 timesteps 采样更频繁。直觉：极端 t (very noisy / very clean) 的 velocity 容易学，中间 t 是难点，多采中间样本。
- **Length-aware timestep shift**：长 latent sequences (e.g., 4096 tokens) 信息容量大，同样的 noise level 可能不足以 disrupt 所有 structure。所以 noise schedule 根据 sequence length scale，长序列需要更高 noise level 才能 fully randomize。

Inference 时用 deterministic sampling 通过 learned velocity field，conditional on image。

**3-stage training**: PT (256 tokens, full dataset) → CT (4096 tokens, full dataset, enhanced augmentation) → SFT (curated high-quality subset, low LR)。典型的 coarse-to-fine + quality boosting 范式。

---

## 3. Texture Pipeline

### 3.1 Seed3D-MV: Multi-view consistent image generation

**Problem with prior work**

- Multi-view attention methods (MVDiffusion, ImageDream) 需要额外 ControlNet / MVAdapter，parameter overhead 大，且和 base diffusion model 的 coupling 不够 tight。
- UniTEX 直接 fine-tune pretrained DiT with concatenated multi-view + cross-view attention，但 base DiT 没设计成 multi-view，对 in-the-wild image 效果 suboptimal。

**Seed3D-MV 的设计**

基于 MMDiT (Multi-Modal Diffusion Transformer, SD3 architecture)，用 in-context multi-modal conditioning + 专门设计的 positional encoding。

**Target distribution (Equation 4)**

$$
p(x | g, i, c)
$$

- $x$: target multi-view images (要生成的)
- $g$: spatially aligned multi-view geometry images = normal maps + canonical coordinate maps (CCM)，从 generated mesh 渲染
- $i$: reference image (user输入)
- $c$: optional text prompt

注意 $g$ 是 "spatially aligned" —— multi-view normal maps 和 CCMs 共享 camera coordinate frame，所以 $g$ 提供了 multi-view 一致的几何 anchor。

**In-context multi-modal conditioning** (借鉴 Flux.1 Kontext)

Noisy input tokens 和 clean condition tokens (geometry, reference image, text) 沿 sequence dimension concatenate。Geometry 和 reference image 通过 frozen VAE encode 到 latent，text 通过 pretrained LM 处理。Training 时 random drop conditional tokens 启用 CFG (classifier-free guidance, Ho & Salimans 2021)。

直觉：in-context conditioning 让 attention 自然地 learn 到 condition 和 target 之间的 correspondence，避免 ControlNet 那种 side-branch + zero-conv 注入的"硬连接"。

**Positional encoding — cross-modal RoPE**

Token sequence 组织：`[multi-view noisy tokens, geometry image tokens, reference image tokens, text tokens]`

修改标准 RoPE 处理两类 token：
- spatially aligned tokens (multi-view noisy + geometry)：用 separate spatial positions，因为它们在不同视角不同空间位置
- non-aligned tokens (reference image, text)：用 non-spatial positions

实验发现：noisy tokens 和 geometry tokens 用 **separate** spatial positions 比 shared spatial positions 效果好。直觉：虽然同一视角的 noisy image 和 geometry image 在像素 grid 上对齐，但它们语义维度不同 (RGB vs normal/CCM)，分开 spatial encoding 让 RoPE 更精细地表达"对应空间位置但不同 modality"的关系。

**Timestep sampling — resolution-aware shift-SNR**

Multi-view 大幅增加 sequence length (e.g., 6 views × 4096 tokens = 24K tokens)，挑战 model capacity。用 resolution-aware timestep sampling (SD3, Esser et al. ICML 2024) with shift-SNR distribution，根据 noisy token sequence length 动态调整 SNR shift。

直觉：高分辨率 (long sequence) 时，高频率 noise 占据更多 signal energy，传统 SNR schedule 会 over-noise 或 under-noise。Shift-SNR 让 schedule 适应分辨率。

---

### 3.2 Seed3D-PBR: Multi-view material decomposition

**Estimation vs generation**

PBR = albedo (base color) + metallic (0/1 ish) + roughness (scalar)。两条技术路线：
- **Generation** (MaterialMVP, Intrinsix)：从 reference image + geometry 直接 synthesize PBR maps。受限于 high-quality PBR training data 稀缺 (互联网上 PBR-labeled 3D asset 远少于 RGB-labeled)。
- **Estimation** (IDArb, Neural LightRig)：从 multi-view RGB images 反推 PBR components (inverse rendering 思路)。Multi-view RGB data 多 (任意 mesh 都能 render)，且分解任务比生成任务更 constrained。

Seed3D 选 estimation paradigm。

**Two-stream network architecture**

核心 insight：albedo 和 MR 物理 / 视觉特性差异极大。
- Albedo: 连续 RGB 值，diffuse 反射，频域偏 low-frequency
- Metallic: 近似 binary (metal or dielectric)，surface reflectance 性质突变
- Roughness: 连续 scalar，控制 specular lobe 宽度

如果完全 shared network (输出 head 区分)：modality-specific 特征在 attention 中互相 wash out。
如果完全 separate networks：parameter 翻倍，且 multi-view consistency 难保证。

Two-stream 折中方案：
- 每个 DiT block 内，为 albedo 和 MR **分别** instantiate Q/K/V projection layers (parameter-efficient)
- 把两个 modality 的 latent vectors 在 attention dim 上 concatenate，加上 global image conditioning
- 通过 **shared full-attention module** 处理
- Feed-forward network 等 other components 保持 shared
- Learnable modality embeddings 加到 positional embeddings 区分 modality
- 两个 decoder heads 分别输出 albedo 和 MR

直觉：Q/K/V projection 是 "modality-specific feature extraction" 的入口，分开让各 modality 提取自己关心的 pattern；attention 和 FFN 是 "cross-modal interaction + nonlinear transformation" 的核心，shared 既省 parameter 又保证 multi-view consistency。

**Dual-level conditioning**

- **Global control**：CLIP vision encoder 提取 reference image 的 global embedding，**替换** text embedding (作为 cross-attention 的 condition)。提供 high-level appearance guidance (overall color tone, style)。
- **Local control** (借鉴 ImageDream)：reference image 的 VAE-encoded latent 和 noise latent 在 channel dim concatenate，作为 DiT block 输入。Multi-view conditioning latents 直接加到 initial noise latents，**只 feed 到 first DiT block** 作为 initial guidance (减少 computational overhead)。

为什么 global + local 双重？Global 控制"整体长什么样"，local 控制"pixel-level 怎么对齐 reference"。CLIP feature 是 semantic-level，丢了 pixel-level detail；VAE latent 是 pixel-level，丢了 global context。互补。

---

### 3.3 Seed3D-UV: UV texture completion

**Problem**

Multi-view baking 步骤：
1. 对每个 multi-view image，用 camera projection matrix back-project 到 mesh surface
2. 每个 visible surface point，根据 visibility 和 surface normal alignment 找 contributing pixels
3. Weighted averaging based on viewing angles (normal alignment 越好 weight 越高)
4. Aggregated surface colors baked 到 2D UV texture map (用 mesh 的 predefined UV parameterization)

结果：UV map 有 holes / seams —— 因为 limited view coverage 和 self-occlusion，某些 mesh 区域在所有 view 下都不可见 (e.g., 物体底部贴桌面、内部凹槽)。

**Coordinate-conditioned UV DiT**

输入：partial UV texture + UV coordinate maps。

UV coordinate maps 作为 **positional tokens** 加入 DiT visual stream。这个 geometric conditioning 引导模型 respect UV parameterization，生成 completion 和 mesh boundary + existing texture content 对齐。

为什么需要 coordinate conditioning？Naive image inpainting 把 UV map 当普通 2D image，但 UV map 是 mesh surface 经过 unwrap 的 2D 展开，UV 空间的"距离"和 mesh surface 的"距离"不一致。UV 上相邻 pixel 可能对应 mesh 上相隔很远的 surface point (UV seam 处)。Coordinate conditioning 让模型知道每个 UV pixel 对应的 3D surface coordinate，从而生成 spatially coherent completion。

实验发现：coordinate-guided conditioning 在 UV boundary 处 texture transition 更 sharp，和 mesh geometry alignment 更好。

---

## 4. Data Pipeline

3D data 处理比 2D 难得多 —— heterogeneity (OBJ/FBX/GLTF/PLY/proprietary), coordinate system 不一致, 质量参差不齐, 大量 corrupted geometry。

### 4.1 7-stage preprocessing pipeline

1. **Diversity-oriented sourcing**：公共 repo + licensed marketplace + synthetic generation。Coverage: geometric complexity, topology, categories (characters/vehicles/furniture/architecture), styles, materials, surface details。

2. **Format standardization**：所有格式 → GLB (compact binary, wide compatibility)。提取 geometry + material，normalize coordinate system。

3. **Geometric deduplication** (重要！)
   - 4 canonical viewpoints 渲染 RGB + normal maps
   - DINOv2 提取 features，所有 views concatenate 形成 mesh representation
   - FAISS (Johnson et al. IEEE Trans. Big Data 2019) 做 billion-scale similarity search
   - Dual-threshold filtering：cosine similarity + L2 distance，平衡 duplicate removal 和 legitimate geometric variations 保留
   
   直觉：3D asset repo 里同一 model 不同版本 (LOD、不同 topology) 大量存在，dedup 避免训练时 model over-fit 到 duplicate subset。

4. **Mesh orientation canonization**
   - 4-view renderings feed 到 trained orientation classifier
   - 预测 canonical orientation，apply transformation
   
   为什么重要？3D model 来源不同，朝向不一致 (有的 +Y up，有的 +Z up；有的前向 +Z，有的前向 -Z)。如果不 canonize，model 要 learn 同一 shape 的所有朝向 variant，浪费 capacity；且 inference 时 generation 朝向不可控。

5. **Quality filtering — 2-stage**
   - Stage 1: Aesthetic scoring (Schuhmann's improved-aesthetic-predictor, open-source) — visual appeal threshold
   - Stage 2: VLM-based (Qwen2.5-VL) assessment：
     - Quality classification: unusable / usable / high-quality
     - Category identification: characters / vehicles / furniture / etc.
     - Data type detection: synthetic / real-world scanned / scene-level
   
   只保留：aesthetic score 达标 + usable-or-higher + **非 real-world scanned + 非 scene-level**。
   
   为什么排除 scanned 和 scene-level？
   - Real-world scanned: 噪声多，topology messy，incomplete surfaces，不适合 VAE training (需要 watertight)
   - Scene-level: 多物体组合，单物体 training 时引入 background noise

6. **Multi-view image rendering** (Blender Cycles, PBR engine)
   - Geometry training: 随机 viewpoint (elevation [-30°, 70°]) + stochastic illumination (30% point lights / 70% HDR env)
   - MV + PBR training: 随机 HDRI + orthogonal viewpoints, render RGB + normal + CCM。PBR 额外渲染 albedo + MR + 1 个 fully-lit reference view
   - UV: xatlas unwrap + bake albedo + CCM

7. **Mesh remeshing** (CUDA-based) — 关键 step
   把 arbitrary raw mesh 转 watertight representation，4 stages：
   1. **Voxelization**: fast raster-like kernels (Schwarz & Seidel TOG 2010) + boundary marking
   2. **Signed distance floodfill**: classify interior / exterior voxels
   3. **Mesh extraction**: threshold $\epsilon$ preserve thin structures (避免把薄壳当 noise 删掉)
   4. **Dual Marching Cubes** (Schaefer & Warren 2002): 在 dual grid 上做 primal contouring，生成 final mesh。Reference 原 mesh 取 zero-crossing normals (保留 sharp feature normals)

   为什么 watertight 重要？Non-watertight mesh 有 holes, non-manifold edges，physics engine 处理会出 numerical issues (collision detection 失败, mass calculation 错误)。Watertight manifold geometry 是 simulation-ready 的 prerequisite。

### 4.2 Data engineering infrastructure

3 个 component：

1. **Data management & indexing**: MongoDB 存 metadata (source, format, processing status, storage paths)。Custom ORM layer 暴露 standardized API (asset registration, metadata update, query)，把 preprocessing logic 和 backend storage 解耦。

2. **Storage & visualization platform**: 
   - Object storage 存 raw + intermediate files (rendered images, VLM annotations)
   - Web-based data platform：filtering / tagging / thumbnail browsing / WebGL 3D viewer
   - Training data packing module：curate/export structured datasets (按 category / quality / processing stage)
   - Processed assets (SDF samples, VAE latents) 打包成 training-ready bundles 存 HDFS

3. **Distributed processing infrastructure** (Ray Data, Moritz et al. OSDI 2018):
   - 关键 challenge：异构 compute requirements (rendering 要 CPU, remeshing 要 GPU)
   - Custom Kubernetes operator: 为每个 processing stage launch CPU / GPU pods with appropriate resource allocation
   - Ray Data 的 elasticity + fault tolerance：用 preemptible resources from cluster idle capacity
   - Preemptible 被 reclaim 时自动 launch replacement pods + reschedule tasks
   - Strategic checkpointing after each major stage：从中间点重启，避免 full reprocessing

---

## 5. Training Infrastructure

### 5.1 Kernel fusion
- torch.compile + custom CUDA kernels
- Profiling 发现 memory-bound ops 是 bottleneck (而非 compute-bound)
- Fuse consecutive element-wise ops → 减少 memory access, 提升 arithmetic intensity
- FlashAttention (Dao et al. NeurIPS 2022) + Apex fused optimizers

### 5.2 Parallelism — HSDP (Hybrid Sharded Data Parallelism)
- Data Parallelism (DP) within nodes + Fully Sharded Data Parallelism (FSDP) across nodes
- Hierarchical 设计：节点内 DP (low communication cost) + 节点间 FSDP (memory efficient sharding)
- 比纯 FSDP 好：跨节点 communication (跨 NVLink / InfiniBand) 比 node 内 (NVLink) 慢得多，HSDP 最小化 cross-node communication

### 5.3 MLAC (Multi-Level Activation Checkpointing)
- Full gradient checkpointing (Chen et al. ICML 2016) 省 GPU memory 但 backprop 时 recompute overhead 大
- MLAC (from Seaweed-7B, Yang et al. 2025)：selectively checkpoint based on recompute cost
- High-cost tensors offload 到 CPU memory + 异步 prefetching overlap memory transfer 和 compute
- 比 full checkpointing：significant memory savings + minimal performance impact

### 5.4 Stability & fault tolerance
- Pre-launch machine health checks (排除 faulty nodes + stragglers)
- Flight recorder：track NCCL communication patterns，failure 时 identify problematic machines
- Centralized monitoring：ETTR (Effective Training Time Ratio) + communication patterns + GPU utilization

---

## 6. Inference Pipeline

```
[1] Geometry generation
    - Input image → Seed3D-DiT → latent → Seed3D-VAE decoder
    - Dual Marching Cubes (DMC) for iso-surface extraction
    - Hierarchical extraction: coarse SDF with bfloat16 → identify candidate zero-crossing cells → 
      inactive cells prune → active cells full-precision float32 evaluation
    - Gradient estimation: analytical gradients from VAE SDF decoder via auto-diff
    - Retopology + UV unwrapping (xatlas)

[2] Multi-view generation + initial texturing
    - Generated mesh + input image → Seed3D-MV → multi-view RGB
    - Back-project to mesh surface → bake to UV space (partial texture)

[3] Material estimation
    - Seed3D-PBR decompose multi-view images → albedo + MR
    - Bake PBR maps to UV space

[4] Texture completion
    - Seed3D-UV inpaint incomplete UV regions

[5] Final integration
    - Completed UV maps + mesh → OBJ/GLB asset
```

**Hierarchical extraction 的直觉**：在所有 voxel 上做 full-precision SDF evaluation 太慢 (e.g., 512³ grid = 134M points)。先用 bfloat16 coarse evaluation 找出"可能含 zero-crossing"的 cells (active cells)，剩下 inactive cells 直接 prune。只对 active cells 做 float32 full precision evaluation，compute 量大幅减少但 mesh fidelity 保留。

**Analytical gradients via auto-diff**：DMC vertex placement 需要 SDF gradient (法向量)。传统数值 gradient (finite difference) 不准确 (尤其 thin features)。Auto-diff 直接 backprop through VAE decoder 得到 exact gradient，thin features 保留得更好。

---

## 7. Performance

### 7.1 Geometry — 1.5B 参数 SOTA

对比 baseline：TRELLIS, TripoSG, Step1X-3D, Direct3D-S2, Hunyuan3D-2.1 (3B params)

| Metric | 含义 |
|---|---|
| ULIP-T | ULIP text-mesh similarity (Xue et al. CVPR 2024) |
| ULIP-I | ULIP image-mesh similarity |
| Uni3D-T | Uni3D text-mesh similarity (Zhou et al. ICLR 2024) |
| Uni3D-I | Uni3D image-mesh similarity |

Test set: 1000 images，多 category + 多 style。每个 mesh sample 8192 surface points，用 Qwen2.5-VL generated captions 作 text conditioning。

Seed3D 1.0 在所有 4 个 metric 上 SOTA，且 1.5B 参数 > Hunyuan3D-2.1 的 3B 参数 → architecture 和 training strategy 更 efficient。

直觉：dual encoder (DINOv2 + RADIO) 让 conditioning 信号更丰富；length-aware timestep shift 让 long-sequence generation 训练更稳定；multi-scale VAE 让 latent representation 更 robust。

### 7.2 Texture — SOTA on multi-view + PBR

对比：MVPainter, Hunyuan3D-Paint, UniTEX, MV-Adapter, Pandora3d, Hunyuan3D 2.1

Multi-view generation metrics: CLIP-FID, CMMD (CLIP Maximum-Mean Discrepancy), CLIP-I (CLIP image similarity), LPIPS

Seed3D-MV SOTA。Seed3D-PBR SOTA。Seed3D 1.0* (用 GT multi-view images) 显示 upper bound —— decouple multi-view generation error 后，PBR estimation quality 进一步提升，证明 Seed3D-PBR 本身强，但被 multi-view generation 的 error bottleneck。

---

## 8. Application

### 8.1 Simulation-ready generation

Pipeline：
1. VLM (Qwen2.5-VL) 估计 asset scale → 调到 real-world dimensions (e.g., 一个生成的椅子缩放到 0.5m 高)
2. Isaac Sim 自动从 watertight geometry 生成 collision meshes (convex decomposition)
3. Apply default material properties (friction coefficient 等) —— 无需 manual tuning
4. Robotic manipulation experiment: grasping + multi-object interaction
5. Physics engine 提供实时 contact force, object dynamics, manipulation outcomes

三个 benefit for embodied AI:
- **Scalable training data**: diverse manipulation scenarios 自动生成
- **Interactive learning**: physics feedback on action consequences
- **Multi-view multi-modal observation**: comprehensive evaluation benchmarks for VLA (vision-language-action) models

### 8.2 Scene generation — factorized approach

1. VLM identify objects + infer spatial relationships → layout maps (scales, positions, orientations)
2. 对每个 object 单独 generate geometry + texture (用 Seed3D 1.0 pipeline)
3. 按 layout assemble 成 complete scene

为什么 factorized？End-to-end scene generation 难度极大 —— scene 包含多 object + 复杂 spatial relationship + lighting interaction。Factorized 把 problem 拆成 (a) VLM 做 semantic layout planning (LLM 擅长) + (b) per-object generation (Seed3D 擅长) + (c) object placement (geometric operation)。每个 sub-problem 都有成熟 tool。

---

## 9. 关键 design intuitions 总结

1. **Decoupled pipeline**: geometry → multi-view → PBR → UV，每 stage 清晰 contract，便于 debug 和 scale。

2. **TSDF over SDF/occupancy**: bounded regression range + surface-localized information，模型 capacity 集中在 fine details。

3. **Set-based latent (VecSet)**: permutation-invariant + length-agnostic，支持 multi-scale training 和 inference-time compute allocation。

4. **Dual image encoder (DINOv2 + RADIO)**: semantic + geometric 互补，缓解 single-view depth ambiguity。

5. **FLUX hybrid DiT**: early modality-specific processing + middle cross-modal attention + late single-stream refinement。

6. **Length-aware timestep shift**: 长 latent sequence 需要更高 noise level disrupt，schedule 自适应 sequence length。

7. **In-context multi-modal conditioning** (Flux.1 Kontext inspired): sequence concatenation 让 attention 自然 learn cross-modal correspondence，避免 ControlNet 硬连接。

8. **Two-stream PBR architecture**: Q/K/V 分 modality + shared attention/FFN，parameter-efficient + modality-specific feature extraction + multi-view consistency。

9. **Estimation over generation for PBR**: RGB data 多，inverse rendering 比 direct generation 更 constrained。

10. **Coordinate-conditioned UV inpainting**: UV space ≠ mesh surface space，coordinate conditioning 让模型 respect mesh geometry。

11. **Watertight remeshing pipeline**: voxelization + floodfill + thin-structure preservation + Dual Marching Cubes，为 physics simulation 提供 valid geometry。

12. **HSDP + MLAC**: node-内 DP + node-间 FSDP，最小化 cross-node communication；selective activation checkpointing + CPU offload，平衡 memory 和 recompute cost。

---

## 10. 可能的 follow-up direction (hallucination / speculation)

- **Joint geometry + texture diffusion**: 当前 sequential pipeline 有 error propagation (geometry error → texture on wrong geometry)。Future: end-to-end latent space 同时 encode geometry + UV + PBR，joint diffusion。
- **Dynamic / articulated assets**: 当前生成 static mesh。Robotics 需要 articulated objects (drawer, door, scissors)。Extending latent representation 加 joint parameters。
- **Physics-aware generation**: 当前生成 geometry 后 physics engine 估算 mass / friction。Future: 生成时直接预测 physical properties (density, Young's modulus)。
- **Interaction-aware scene**: 当前 scene 是 object placement。Future: 生成 scene 时考虑 object 之间 interaction (book on shelf, knife in drawer)。
- **Generative world simulator**: 把 Seed3D 接入 RL training loop，agent interaction 时按需 generate 新 asset，实现 open-ended environment。

---

## 11. References & 进一步阅读

- Seed3D Official: https://seed.bytedance.com/seed3d
- 3DShape2VecSet (Zhang et al. TOG 2023): https://arxiv.org/abs/2312.06721
- Dora (Chen et al. CVPR 2025): https://arxiv.org/abs2412.18790
- TRELLIS (Xiang et al. CVPR 2025): https://arxiv.org/abs/2412.01506
- Hunyuan3D 2.1: https://arxiv.org/abs/2506.15442
- TripoSG (Li et al.): https://arxiv.org/abs/2502.06608
- Step1X-3D: https://arxiv.org/abs/2505.07747
- Direct3D-S2 (Wu et al.): https://arxiv.org/abs/2505.17412
- FLUX (Black Forest Labs): https://github.com/black-forest-labs/flux
- Flux.1 Kontext: https://arxiv.org/abs/2506.15742
- SD3 / MMDiT (Esser et al. ICML 2024): https://arxiv.org/abs/2403.03206
- Rectified flow (Liu et al. ICLR 2023): https://arxiv.org/abs/2209.03003
- Flow matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- DINOv2 (Oquab et al.): https://arxiv.org/abs/2304.07193
- RADIO (Ranzinger et al. CVPR 2024): https://arxiv.org/abs/2312.06709
- UniTEX (Liang et al.): https://arxiv.org/abs/2505.23253
- ImageDream (Wang & Shi): https://arxiv.org/abs/2312.02201
- MV-Adapter (Huang et al.): https://arxiv.org/abs/2412.03632
- ControlNet (Zhang et al. ICCV 2023): https://arxiv.org/abs/2302.05543
- Classifier-free guidance (Ho & Salimans): https://arxiv.org/abs/2202.08271
- RoPE / RoFormer (Su et al.): https://arxiv.org/abs/2104.09864
- Fourier features (Tancik et al. NeurIPS 2020): https://arxiv.org/abs/2006.10739
- FlashAttention (Dao et al. NeurIPS 2022): https://arxiv.org/abs/2205.14135
- VAE (Kingma & Welling): https://arxiv.org/abs/1312.6114
- CLIP (Radford et al. ICML 2021): https://arxiv.org/abs/2103.00020
- FAISS (Johnson et al.): https://arxiv.org/abs/1702.08734
- Ray (Moritz et al. OSDI 2018): https://arxiv.org/abs/1712.05889
- PyTorch FSDP (Zhao et al.): https://arxiv.org/abs/2304.11277
- IsaacGym: https://arxiv.org/abs/2108.10470
- Cosmos World Foundation Model: https://arxiv.org/abs/2501.03575
- Genie 3: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- Qwen2.5-VL (Bai et al.): https://arxiv.org/abs/2502.13923
- xatlas (UV unwrapping): https://github.com/jpcy/xatlas
- Dual Marching Cubes (Schaefer & Warren): https://faculty.cs.tamu.edu/schaefer/research/dmc.pdf
- Blender: https://www.blender.org/
- Seaweed-7B (MLAC): https://arxiv.org/abs/2504.08685
- Mixed precision training (Micikevicius et al.): https://arxiv.org/abs/1710.03740
- Disney PBR (Burley): https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
- Improved aesthetic predictor (Schuhmann): https://github.com/christophschuhmann/improved-aesthetic-predictor
- IDArb (Li et al. ICLR 2025): https://openreview.net/forum?id=uuef1HP6X7
- Neural LightRig (He et al. CVPR 2025): https://arxiv.org/abs/2502.03407
- MaterialMVP (He et al.): https://arxiv.org/abs/2503.10289
- Pandora3D (Yang et al.): https://arxiv.org/abs/2502.14247
- MVPainter (Shao et al.): https://arxiv.org/abs/2505.12635
- Hunyuan3D-Paint (Zhao et al.): https://arxiv.org/abs/2501.12202
- CraftMan3D (Li et al.): https://arxiv.org/abs/2405.14979
- CLAY (Zhang et al. TOG 2024): https://arxiv.org/abs/2405.04841
- ULIP-2 (Xue et al. CVPR 2024): https://arxiv.org/abs/2402.08749
- Uni3D (Zhou et al. ICLR 2024): https://arxiv.org/abs/2310.17288

---

## 12. 与 Karpathy 视角的对齐思考

这篇工作的核心 thesis 是 **"3D asset 是 embodied AI 的 bottleneck"**，而 LMM 已经证明 "interactive environments 能克服 data scarcity through structured feedback" (DeepSeek-R1 在 coding domain 的成功)。把 paradigm 从 coding 移到 embodied AI，需要 high-fidelity simulation environments 提供有意义 spatial reasoning 和 physical manipulation feedback。

Seed3D 1.0 把 3D generation 从 "好看就行" 推进到 "能进 physics engine 跑 sim"，这正好是 missing piece。配合 VLM 做 scale estimation + scene layout，pipeline 可以从 single object 一路 scale 到 full scene，逼近真正的 "scalable simulation-ready content creation"。

值得关注的几个 design choice，从教学角度：
1. **VecSet latent** 是很 elegant 的 representation —— set 的 permutation invariance 配合 multi-scale training，让 inference 时能灵活 allocate compute，类似 LLM 的 variable-length sequence。
2. **Two-stream PBR** 是 parameter-efficient multi-modal design 的典范 —— 在该分开的地方分开 (Q/K/V)，该共享的地方共享 (attention core, FFN)，类比 MoE 但更轻量。
3. **In-context conditioning** (Flux.1 Kontext 范式) 可能是 diffusion model conditioning 的 future 方向 —— 比 ControlNet 的 side-branch 注入更 elegant，让 attention 自然 learn correspondence。

如果你在 Stanford 教 3D generation / embodied AI，这篇 paper 几乎每个 component 都值得拆开讲一节课：VAE design, rectified flow scheduling, multi-modal conditioning, PBR decomposition, UV completion, data pipeline, training infra。Production-grade 3D generation 的全栈工程化，这篇是当前最完整的 reference 之一。
