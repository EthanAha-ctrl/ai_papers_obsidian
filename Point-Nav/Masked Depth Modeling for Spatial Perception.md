---
source_pdf: Masked Depth Modeling for Spatial Perception.pdf
paper_sha256: 480937fc369c02535a6eac4a69ccb47566e394bc898e8911cdbc63cbeeb30896
processed_at: '2026-08-05T16:32:39-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们换个轻松点的聊法，我把这篇 paper 揉碎了用大白话给你讲讲，顺便把里面的技术细节再串一串，帮你 build 更深的 intuition。

### 1. RGB-D 相机的“阿喀琉斯之踵”

如果你玩过 Intel RealSense 或者 Orbbec 这类 RGB-D camera，你会发现它们测距确实准，能给你一张每个 pixel 都对应真实米数的 depth map。这在 robotic manipulation 或者 autonomous driving 里简直是刚需。

但是，这玩意儿有个致命弱点：一碰到 glass、mirror、specular metal 或者 texture-less 的白墙，它就“瞎”了。投射出去的 infrared structured light 或者 speckle pattern，要么被反射跑了，要么直接透过去了，要么 stereo matching 根本找不到 correspondence。于是 depth map 上就出现一堆大黑窟窿（holes）。

过去大家怎么处理？要么用 filtering 抹一抹，要么干脆把这些 invalid points 当 noise 扔了。

### 2. 破局点：把黑窟窿当“考题”

这篇 paper 的 idea 非常精妙：这些 holes 并非 random noise 产生的，它们恰恰代表了 scene 里面“最难看懂”的地方（比如 specular reflection 说明 material 特殊，透光说明是 glass）。如果把这当成一种天然的 mask，那这简直就是物理世界馈赠的 MAE (Masked Autoencoder) training set。

标准的 MAE 是怎么干的？把一张 RGB image 随机挖掉 75%，让 model 猜。但 random masking 太简单了，model 往往靠 local interpolation 就能蒙混过关，学不到什么 high-level 的 semantic。

MDM (Masked Depth Modeling) 怎么干？它让 RGB image 全开，depth map 把 sensor 产生的 holes 挖掉，让 model 猜。因为 RGB 是全亮的 condition，model 必须学会 cross-modal reasoning——看懂了 RGB 里的 texture、material、boundary，才能推断出 depth 里缺的那块到底有多深。这其实是把 sensor 的 physical failure 变成了 importance-weighted hard negative mining。

### 3. 网络架构：让 RGB 和 Depth 互相“聊天”

架构上，输入被拆成两路：RGB 走 RGB 的 patch embedding，Depth 走 Depth 的。假设 patch size 是 $p=14$，那么 token 数量就是 $N = H \cdot W / p^2$。

$$ \mathbf{c}_i \in \mathbb{R}^n $$ 是第 $i$ 个 RGB token，$$ \mathbf{d}_i \in \mathbb{R}^n $$ 是第 $i$ 个 Depth token。上标 $n$ 是 hidden dimension（ViT-Large 里是 1024）。它们加上 spatial 和 modality 两个 positional embedding 后，被扔进一个 24 层的 ViT encoder 里。

ViT 的核心是 self-attention，在这里变成了 RGB token 和 Depth token 互相 attend。Depth token 会去问 RGB token：“哎，我这儿空了，你那边看起来像个 cup 的边缘，你觉得我这该填多深？”这就强迫网络学到了 appearance 和 geometry 之间的 fine-grained correspondence。

到了 decoder 阶段，作者放弃了 vanilla MAE 的 shallow transformer decoder，换成了 MoGe 的 ConvStack（一堆 convolutions）。为什么？因为预测 depth 是个 dense geometric regression 任务，convolution 那种 spatial translation invariance 和 local smoothness 天生适合画 depth map，而 transformer token 容易画出 blocky artifacts。最后预测的 loss 就是简单粗暴的 L1 loss：

$$ \mathcal{L} = \frac{1}{|\mathcal{V}_{\text{GT}}|} \sum_{p \in \mathcal{V}_{\text{GT}}} | \hat{D}_p - D^{\text{GT}}_p |_1 $$

其中 $\hat{D}_p$ 是预测的 depth，$D^{\text{GT}}_p$ 是 ground truth，$\mathcal{V}_{\text{GT}}$ 是 GT 里 valid pixel 的集合，下标 $p$ 是 pixel 索引。

### 4. 一个模型，两种用法

这套设计有个极大的好处：通过控制 mask ratio，model 能无缝切换任务。

如果把 Depth 全部 mask 掉（mask ratio = 100%），model 只能靠 RGB 猜，这就变成了 Monocular Depth Estimation (MDE)。
如果只把 sensor 瞎掉的地方 mask 掉，model 就把剩下的 valid depth 和 RGB 融合，补全整张图，这就变成了 Depth Completion (DC)。

其实训练时的 mask 策略是个混合体：
- 如果一个 patch 完全没瞎，100% mask；
- 如果一个 patch 半瞎，75% 概率 mask；
- 如果总 mask ratio 还不够 60%~90%，就随机抓点全好的 patch 来 mask 凑数。

这保证了 model 既学到了 hard cases 的 reconstruction，又学到了 global geometry 的 reasoning。

### 5. 数据是硬通货

要训练大 ViT，数据少了不行。作者搞了两套 data curation pipeline：
一套是合成的（LingBot-Depth-S，1M samples）。用 Blender 建了 442 个 indoor scenes，不仅渲染完美的 depth，还渲染带 speckle 的 infrared stereo pair，然后用 SGM (Semi-Global Matching) 算法去算这个假的 sensor depth。这完美模拟了真实 active camera 的“瞎眼”过程。
一套是真实的（LingBot-Depth-R，2M samples）。拿着 3D printed 的 fixture，装上各种 RealSense、Orbbec、ZED，去 residential、commercial、public spaces 一通狂拍。但 real data 没 GT depth 啊？作者就用 FoundationStereo 算法去算 stereo matching，得到 pseudo depth label，再做一下 left-right consistency check 过滤掉不靠谱的点。

加上开源的 7M samples，总共凑了 10M training data。用 128 张 GPU，batch size 1024，AdamW optimizer，差分学习率（encoder $10^{-5}$，decoder $10^{-4}$），训了 7.5 天。

### 6. 实验和应用：大杀四方

在 depth completion 上，不管是随机的 block-wise masking 还是极度稀疏的 SfM 点输入，都是 SOTA。比如在 NYUv2 的 extreme 遮挡下，RMSE 比之前最强的 PromptDA 降了 40% 多。

更神的是，把训练好的 model 里的 Depth branch 直接砍掉，只留 RGB encoder 去做 MDE，居然比大名鼎鼎的 DINOv2 初始化的效果还要好！这说明在 RGB-D joint training 中，RGB encoder 偷偷学会了“geometric intuition”，这比起 unimodal pretrain 更适合 dense geometric task。这也让它成了 FoundationStereo 更好的 backbone prior，收敛更快更稳。

应用上更接地气：
直接往 video 里一帧帧跑，虽然没有加任何 temporal module，出来的 depth video 却异常 smooth，temporal consistency 极好。
接给 SpatialTrackerV2 做 3D point tracking，轨迹丝滑。
最猛的是接给 Rokae 机械臂做 dexterous grasping。面对 transparent storage box，原相机的 depth 烂到根本没法抓（0% 成功率），用这个 model 补全后，居然有 50% 的成功率抓起来了。

**总结一句**：这篇 paper 把 RGB-D camera 的物理缺陷变成了 self-supervised learning 的信号，用 cross-modal masking 统一了 MDE 和 DC 两个任务，通过大规模数据训练出了一个具有 strong geometric prior 的 foundation model。

**References:**
- Paper GitHub: https://github.com/robbyant/lingbot-depth
- MAE: https://arxiv.org/abs/2111.06377
- DINOv2: https://arxiv.org/abs/2304.07193
- MoGe: https://arxiv.org/abs/2410.19115
- FoundationStereo: https://arxiv.org/abs/2501.09862
- SpatialTrackerV2: https://arxiv.org/abs/2507.12562
- PromptDA: https://arxiv.org/abs/2412.14054

---

# Masked Depth Modeling for Spatial Perception 深度讲解

你好 Andrej，很高兴跟你聊这篇 paper。这是 Robbyant 团队的工作，核心 idea 非常优雅——把 RGB-D sensor 的 "missing depth" 这种 "缺陷" 重新解读为天然的 masking signal，让 ViT 在 RGB-D 联合 embedding 中学习 metric depth prior。下面我从 motivation、架构、数据、训练、实验、应用六个层面来 build your intuition。

---

## 1. Motivation 的深层 reasoning

### 1.1 RGB-D camera 的三难困境

paper 开篇就指出：3D perception 在 physical world（autonomous driving、robotic manipulation）需要三个硬指标——**absolute metric scale**、**pixel-aligned dense geometry**、**real-time acquisition**。

- Multi-view geometry (COLMAP, SfM, SLAM) 缺第三条：computational expensive
- Monocular depth estimation 缺第一条：up-to-scale ambiguity（除非 metric supervised）
- Active sensors (LiDAR/ToF) sparse：缺第二条
- RGB-D camera (structured light / active stereo / passive stereo) 是唯一同时满足三条的

但 RGB-D 在 specular reflection、texture-less surface、glass、mirror 上彻底 fail。商业 sensor (Orbbec Gemini 335, Intel RealSense, ZED) 在这些场景输出的 depth map 出现大块 "holes"——这被传统视角当作 noise 来 discard 或 interpolate。

### 1.2 关键 insight：holes 不是 random dropout

Karpathy 你应该会喜欢这个点：**holes 是 appearance ambiguity 的指示器**。Specular reflection 让 structured-light projector 的 speckle pattern 看不见；texture-less surface 让 stereo matching 没有 correspondence 可对；玻璃让 IR 光直接透射或折射。

这跟 MAE 的 random masking 有本质区别。MAE 的 random mask 假设 reconstruction difficulty 是均匀分布的；natural mask 偏向 "hardest region"——恰恰是模型最需要学习 prior 的地方。所以这其实是 **importance-weighted hard negative mining** 的一个 self-supervised 变体。

可以参考：
- MAE: https://arxiv.org/abs/2111.06377
- iBOT: https://arxiv.org/abs/2111.07832
- BEiT: https://arxiv.org/abs/2106.08254

### 1.3 Unifying MDE 和 DC via masking ratio

一个非常漂亮的 design choice：**masking ratio 决定 task identity**。

- mask ratio = 100%：纯 Monocular Depth Estimation
- mask ratio = invalid region only：纯 Depth Completion
- mask ratio ∈ [60%, 90%] mixed：generalist regime

这意味着不需要 multi-task head 或 task token，single Transformer 在 inference 时通过 masking configuration 就能切换任务。这跟 Unified-IO、Florence、Depth Anything 系列的 "single backbone multi-task" 思路一致，但更轻——不需要 task embedding，mask pattern 即 task indicator。

---

## 2. Architecture Deep Dive

### 2.1 Patch Embedding 的分离设计

输入 RGB image $I \in \mathbb{R}^{H \times W \times 3}$ 和 depth map $D \in \mathbb{R}^{H \times W \times 1}$，patch size $p=14$（遵循 DINOv2 ViT-L/14 约定）。token 数量：

$$N = \frac{H \cdot W}{p^2} = \frac{H \cdot W}{196}$$

每个 modality 独立 patch embedding：
- RGB patch embedding：$\mathbf{c}_i = \text{Conv2d}_{3 \to n}(I)_{\text{patch}_i}$，输出 $\mathbf{c}_i \in \mathbb{R}^n$
- Depth patch embedding：$\mathbf{d}_i = \text{Conv2d}_{1 \to n}(D)_{\text{patch}_i}$，输出 $\mathbf{d}_i \in \mathbb{R}^n$

其中 $n = 1024$（ViT-Large hidden dim）。

**为什么 separated 而非 concat input (4-channel)**？我推测：
1. RGB 和 depth 的 statistical distribution 差异大（RGB 是 0-255，depth 是 metric 米数），共享 conv kernel 容易互相拖累
2. 分离后可以 freeze RGB embedding 用 DINOv2 init weights，只新增 depth embedding（paper 没明说但符合 transfer learning 直觉）
3. Masking 时只 mask depth tokens，RGB 全部保留——分离 token 才能 selective drop

### 2.2 Positional Embedding 双层结构

每个 token 加两个 PE：

$$\text{PE}(\mathbf{c}_i) = \text{SPE}_{2D}(i) + \text{ME}_{\text{rgb}}$$
$$\text{PE}(\mathbf{d}_i) = \text{SPE}_{2D}(i) + \text{ME}_{\text{depth}}$$

其中：
- $\text{SPE}_{2D} \in \mathbb{R}^n$：shared learnable 2D spatial positional embedding，对 RGB 和 depth **同一个** lookup table
- $\text{ME}_{\text{rgb}} = \mathbf{e}_1$，$\text{ME}_{\text{depth}} = \mathbf{e}_2$：modality embedding，paper 里写 "set to 1 for RGB, 2 for depth"——这应该指的是 modality id，实际是 learnable embedding $\mathbf{e}_1, \mathbf{e}_2 \in \mathbb{R}^n$ 的 lookup key

**Intuition**：shared SPE 让 RGB 和 depth 在同一位置的 token 有相同 spatial prior，self-attention 在跨 modality attend 时能利用 "同位置" 这个 strong inductive bias；ME 让模型能区分 token modality，避免 modality collapse。

### 2.3 RGB-D Token 序列构造

masking 后保留 $\{ \mathbf{c}_i \}_{i=1}^{N}$ (全部 RGB tokens) + $\{ \mathbf{d}_j \}_{j \in \mathcal{V}}$ (valid depth tokens, $\mathcal{V}$ 是 unmask set)。再加 [cls] token：

$$\mathbf{Z}^{(0)} = [\text{cls}; \mathbf{c}_1; \dots; \mathbf{c}_N; \mathbf{d}_{j_1}; \dots; \mathbf{d}_{j_{|\mathcal{V}|}}] \in \mathbb{R}^{(1 + N + |\mathcal{V}|) \times n}$$

送进 24 层 ViT encoder。这跟 MAE 的 "only masked tokens + mask tokens in decoder" 不同——这里 RGB 完整保留在 encoder 中作为 condition，depth 只有 valid 部分进 encoder。Masked depth tokens 在 encoder 后直接丢弃（不补 mask token），由 decoder 从 contextual tokens 重建。

### 2.4 ConvStack Decoder 替代 MAE Shallow Transformer Decoder

这是个关键架构选择。vanilla MAE decoder 是 8-layer shallow transformer，输出 $\text{patch\_size}^2 \times 3$ 像素值。Paper 放弃这个，改用 MoGe [28,29] 的 **ConvStack**——pyramid conv decoder。

**为什么**？Karpathy 你能直觉到：
- Depth 是 dense geometric prediction，需要 spatial smoothness + sharp boundary 的混合特性
- Transformer decoder 的 token-wise 重建缺乏 inductive bias for spatial coherence（除了 positional embedding），容易产生 blocky artifacts
- Conv decoder 有 translation invariance prior，适合 dense regression
- Pyramid 结构（residual blocks + transposed conv stride 2）progressive upsample $h \times w \to 16h \times 16w$，再 bilinear 到 full resolution

Decoder 结构（从 paper 重建）：
```
encoder output: [B, 1+N+|V|, n]
  → discard latent depth tokens
  → broadcast [cls] to all contextual tokens (element-wise add)
  → reshape to [B, h, w, n]
ConvStack:
  for s in [1..4]:  # 4 upsampling stages, ×2 each
    x = residual_blocks(x)
    x = transposed_conv2d(x, kernel=2, stride=2)
    x = inject_UV_PE(x)  # circular mapping of image coordinates
  → multi-scale feature pyramid
  → task-specific heads
  → bilinear upsample to (H, W)
  → predicted depth \hat{D}
```

**UV positional encoding 的细节**：用 circular mapping $\phi: (u, v) \to (\sin(2\pi u/W), \cos(2\pi u/W), \sin(2\pi v/H), \cos(2\pi v/H))$ 这样的 Fourier feature，每个 scale 注入对应分辨率下的 UV encoding。这跟 NeRF 的 PE 思想类似——让 conv 能 "感知" 绝对坐标，否则纯 conv 只知道相对位置。

参考：
- MoGe: https://arxiv.org/abs/2410.19115
- MoGe-2: NeurIPS 2025
- NeRF PE: https://arxiv.org/abs/2003.08934

### 2.5 Multi-layer Aggregation 的弃用

DepthAnythingV2 和 MoGe 都从 ViT 的 layer 6/12/18/24 抽 feature 做 hierarchical decoder，类似 UNet skip connection。Paper 这里 **只用 layer 24 输出**。

**Hypothesis**：因为 RGB tokens 全部保留在 encoder 中，且 DINOv2 init 已经提供 strong appearance representation，layer 24 的 single-layer feature 已经包含足够信息。Multi-layer aggregation 的好处在 pure RGB MDE 是 "RGB feature 在深层会丢失 fine-grained 细节"，但这里 depth valid tokens 一直在 encoder 里跟 RGB interaction，fine-grained depth cue 直接通过 cross-modal attention 保留在 final layer。这是个值得 ablation 的点，paper 没给。

### 2.6 Attention Visualization 验证 cross-modal learning

Fig. 3 用 Orbbec Gemini-335 在 aquarium 和 indoor shelf 上做 multi-query depth-to-RGB attention 可视化。选 Q1/Q2/Q3 三个 depth query patch，看它们 attend 哪些 RGB tokens。

**Expected behavior**：如果学到了 trivial attention（比如所有 query 都 attend [cls] 或者均匀分布），那 cross-modal learning 失败。

**Observed**：不同 query attend 不同的、spatially corresponding RGB regions——证明 attention 学到了 fine-grained position-aware geometric-appearance correspondence。这跟 DINOv2 attention rollout 揭示的 "object discovery" 类似，但这里是 cross-modal。

参考：
- Attention rollout: https://arxiv.org/abs/2005.00928
- DINO attention: https://arxiv.org/abs/2104.14294

---

## 3. Masking Strategy 数学化

### 3.1 Patch-level Validity 定义

对一个 patch $P_k \subset \{1, \dots, H \cdot W\}$，定义 valid set $V_k = \{p \in P_k : D_p \text{ valid}\}$。

mask decision $m_k \in \{0, 1\}$（1 = mask）：

$$m_k = \begin{cases} 
1 & \text{if } |V_k| = 0 \quad \text{(完全 missing)}\\
\text{Bernoulli}(0.75) & \text{if } 0 < |V_k| < |P_k| \quad \text{(混合)}\\
\text{Bernoulli}(p_{\text{rand}}) & \text{if } |V_k| = |P_k| \quad \text{(完全 valid, 用于补足)}
\end{cases}$$

总 mask ratio target $\in [60\%, 90\%]$。$p_{\text{rand}}$ 动态调整以满足 target。

### 3.2 设计意图

- 完全 missing 强制 mask：这些 patch 是最 informative 的 hard example，必须让模型学
- 混合 patch 75% mask：保留 25% 让模型有 "桥接" valid 和 invalid 的中间信号，类似 SLN (sparse-to-dense) supervision
- 完全 valid patch 随机补：避免模型 collapse 成 "只学 missing 区域 reconstruction" 的 shortcut，强制学全局 geometry

这跟 CompletionNets、CFL (coarse-to-fine) 系列 depth completion 的 sparse-to-dense 思路有精神上的呼应，但用 patch token 实现得更优雅。

### 3.3 与 Standard MAE 的对比

| 方面 | MAE | MDM |
|------|-----|-----|
| Mask target | RGB patches | Depth patches |
| Mask source | Uniform random | Sensor-induced natural + random supplement |
| Condition | None (masked tokens only to encoder) | Full RGB image as condition |
| Reconstruction | RGB pixels | Depth values |
| Decoder | Shallow Transformer | ConvStack |
| Mask ratio | 75% | 60-90% |

---

## 4. Training Hyperparameters 完整解读

### 4.1 Optimizer

- AdamW, $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $0.05$
- **Differential learning rate**：
  - Encoder backbone (DINOv2-pretrained): $\eta_{\text{enc}} = 10^{-5}$
  - Decoder (random init): $\eta_{\text{dec}} = 10^{-4}$
  - 参数分组用 name match：`\*backbone\*` pattern → low LR group

**Intuition**：DINOv2 已经有 strong representation，大 LR 会 catastrophic forgetting；decoder 是 random init，需要大 LR 快速学到 task-specific 信号。这是 transfer learning 的 standard practice，但 paper 明确写出来说明团队重视这个细节。

### 4.2 LR Schedule

- Warmup: 前 2000 iter, $\eta_{\text{enc}}$ 线性 0 → $10^{-5}$, $\eta_{\text{dec}}$ 直接 $10^{-4}$
- Step decay: 每 25000 iter × 0.5
- Total: 250000 iter
- Batch size: 1024 (128 GPUs × 8)
- 训练时间: ~7.5 天

**LR at iter t**: $\eta(t) = \eta_{\text{base}} \cdot 0.5^{\lfloor t / 25000 \rfloor}$ for $t > 2000$.

参考 AdamW: https://arxiv.org/abs/1711.05101

### 4.3 Data Augmentation

- Random resized crop
- Horizontal flip
- **Synthetic degradations**:
  - Color jittering
  - JPEG compression artifacts
  - Motion blur
  - Shot noise

这些 degradation 是为了 robustness——paper 想让模型对 real-world image quality variation 不敏感。

### 4.4 Loss Function

L1 on valid GT pixels only:

$$\mathcal{L} = \frac{1}{|\mathcal{V}_{\text{GT}}|} \sum_{p \in \mathcal{V}_{\text{GT}}} | \hat{D}_p - D^{\text{GT}}_p |_1$$

其中 $\mathcal{V}_{\text{GT}}$ 是 GT depth 的 valid set。L1 比 L2 更 robust to outliers，且对 depth 这种 heavy-tailed distribution 更友好。

### 4.5 稳定性技巧

- Gradient clip max norm = 1.0
- BF16 mixed precision
- ViT-Large 24 layers, hidden 1024, heads 16, MLP 4096（standard ViT-L config）

### 4.6 Initialization

- Encoder: DINOv2 ViT-L/14 official checkpoint
- Decoder: random init
- Depth patch embedding: 应该是 random（paper 没说，但 RGB embedding 应该来自 DINOv2 patch embed layer）

---

## 5. Data Curation Pipeline 详解

### 5.1 Synthetic Branch (LingBot-Depth-S, 1M samples)

**Pipeline**:
```
Blender + 3D assets (442 indoor scenes)
  ↓
Render simultaneously:
  - RGB image (960×1280) from left camera
  - Perfect depth map (960×1280)
  - Stereo IR pair with speckle patterns (720×960)
  - GT disparity (960×1280)
  ↓
Stereo random config:
  - baseline ∈ U(0.05, 0.2) meters
  - focal length ∈ U(16, 28) mm
  ↓
SGM (Semi-Global Matching) on stereo pair → sensor-like depth
  ↓
Nearest-neighbor upsample 720×960 → 960×1280
  ↓
Final sample: {RGB, perfect depth, stereo pair, GT disp, simulated sensor depth}
```

**Key insight**: 用 SGM 处理 stereo IR pair 来模拟 active RGB-D camera 的 artifacts——specular reflection 让 speckle 不可见 → SGM 失败 → hole。这比直接 random dropout 更接近真实 sensor failure mode。

参考:
- SGM (Hirschmüller): https://ieeexplore.ieee.org/document/4359315
- DREDS: https://arxiv.org/abs/2207.09004 (Dai et al., 类似思路但 scale 小)

**Scale comparison**: 
- HSSD-IsaacSIM-STD: 10k stereo pairs
- DREDS: 130k stereo pairs
- LingBot-Depth-S: 1M (10× DREDS, 100× HSSD)

### 5.2 Real Branch (LingBot-Depth-R, 2M samples)

**Hardware setup**:
- 3D-printed mounting fixture
- 后置: 多种 RGB-D camera (Intel RealSense, Orbbec Gemini, ZED)
- 前置: portable PC + touchscreen
- 统一 SDK 数据接口

**Scene distribution** (Table 1):
- Residential: 30.5% (3 子类 × 10.16%)
- Work/Study: 23.8% (7 子类)
- Commercial/Service: 16.1% (5 子类)
- Public: 13.6% (4 子类)
- Special-Function: 16.9% (5 子类)
- Outdoor: 10.16%

设计覆盖广，避免 dataset bias。

**Pseudo-label generation**:
- Real captures 没有 perfect GT
- 用 FoundationStereo [33] 处理 left-right IR stereo pairs → pseudo depth
- Left-right consistency check 过滤 inconsistent pixels
- 类似 self-supervised stereo + filtering

参考:
- FoundationStereo: https://arxiv.org/abs/2501.09862
- Left-right consistency (Hirschmüller SGM 经典做法)

### 5.3 Open-source Data 补充 (7 个 dataset, ~7M samples)

- ARKitScenes [4]: iOS RGB-D indoor
- Aria Digital Twin [19]
- Hypersim [21]: photorealistic synthetic indoor
- ClearGrasp [22]: transparent objects
- ScanNet++ [36]: high-fidelity indoor
- Taskonomy [37]: 4-task dataset
- TartanAir [30]: synthetic SLAM dataset

Total training data: **10M samples** (3M self-curated + 7M open-source)。

Fig. 8 给出 data composition breakdown。

对 open-source datasets：
- Synthetic (无 missing): random mask 60-90%
- Real (相对 complete): 也以 random mask 为主

---

## 6. Experimental Results 深度解析

### 6.1 Depth Completion (Sec 4.1)

**Protocol 1**: Block-wise depth masking with 4 difficulty levels (easy/medium/hard/extreme) on iBims, NYUv2, DIODE-Indoor, DIODE-Outdoor。

Metrics: RMSE↓, REL↓ (relative error)

Table 2a 关键数字（RMSE, NYUv2 extreme）:
- OMNI-DC: 1.937
- OMNI-DC-DA: 1.937 (重复?)
- PromptDA: 0.607
- PriorDA: 0.845
- **Ours: 0.345** (vs PromptDA 最好的 0.607, **43% relative improvement**)

Table 2b (Protocol 2, sparse SfM input on ETH3D):
- Indoor RMSE: Ours 0.192 vs PriorDA 0.360 (**47% reduction**)
- Outdoor RMSE: Ours 0.664 vs OMNI-DC-DA 1.093 (**39% reduction**)

Protocol 2 比 Protocol 1 难得多——sparse SfM points 极稀疏，相当于 mask ratio ~99%。这说明 MDM 学到了非常 strong 的 monocular depth prior，能在 near-zero depth observation 下靠 RGB context 推理。

参考:
- PromptDA: https://arxiv.org/abs/2412.14054
- PriorDA: https://arxiv.org/abs/2505.10565
- OMNI-DC: https://arxiv.org/abs/2409.17965
- iBims-1: https://arxiv.org/abs/1904.03092
- NYUv2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- DIODE: https://arxiv.org/abs/1908.00463
- ETH3D: https://arxiv.org/abs/1702.01313

### 6.2 Monocular Depth Estimation (Sec 4.2)

用 LingBot-Depth encoder 替换 DINOv2 作为 MoGe 的初始化。**Training**: 仅用 TartanAir（节省计算，paper 说）。

**Evaluation** on 10 datasets: NYUv2, KITTI, ETH3D, iBims-1, GSO, Sintel, DDAD, Spring, DIODE, HAMMER。

Three metric families:
- Affine-invariant (Aff-inv): depth 抓 up-to-shift + up-to-scale
- Scale-invariant (Scl-inv): up-to-scale
- Disparity-invariant (Disp-inv): up-to-affine on disparity

Table 3 关键数字 (Scl-inv δ1↑):
- NYUv2: DINOv2 0.957, Ours 0.971 (+1.4 pp)
- KITTI: DINOv2 0.544, Ours 0.556
- iBims-1: DINOv2 0.947, Ours 0.962
- Sintel: DINOv2 0.559, Ours 0.573

**Key finding**: 在 inference 时 depth branch 完全移除（remove depth embedding + ConvStack decoder），只用 RGB-only encoder。但 MDM pretrain 让 encoder 内化了 3D geometric knowledge，使得 monocular depth 也提升。

这是个非常重要的 transfer learning insight——**通过 cross-modal pretrain 学到的 representation 比 unimodal (DINOv2) 更适合 dense geometric task**。这跟 ImageNet pretrain → downstream、CLIP pretrain → classification 的 spirit 一致，但换成 RGB-D joint embedding → MDE。

参考:
- MoGe: https://arxiv.org/abs/2410.19115
- TartanAir: https://arxiv.org/abs/2003.14338
- KITTI: http://www.cvlibs.net/datasets/kitti/
- Sintel: https://arxiv.org/abs/1204.2436
- GSO: https://arxiv.org/abs/2104.00783
- DDAD: https://arxiv.org/abs/2011.03412
- HAMMER: https://arxiv.org/abs/2205.04565

### 6.3 FoundationStereo with MDM Prior (Sec 4.3)

替换 FoundationStereo 的 DepthAnythingV2 backbone 为 MDM-pretrained encoder。Training on FSD dataset, 15 epochs, 完全相同 hyperparams。

**Three comparison runs**:
- Vanilla FoundationStereo (DepthAnythingV2 prior)
- FoundationStereo + MoGe (DepthAnythingV2 alternative)
- FoundationStereo + LingBot-Depth encoder (ours)

Fig. 10 epoch-wise comparison (EPE 和 BP-1.0 在 epochs 5, 10, 15):

**Faster convergence** (epoch 5):
- HAMMER EPE: vanilla 0.46, MoGe 2.53, **Ours 0.27** (41% better than vanilla, MoGe 严重 unstable)
- Booster EPE: vanilla 1.00, MoGe 2.84, **Ours 0.86**

**Training stability**: MoGe variant 在 early epoch 出现 catastrophic high error，说明 MoGe init 的 representation 不适合 stereo matching task；MDM init 稳定。

**Final performance** (epoch 15):
- Middlebury EPE: 0.75 (best)
- HAMMER EPE: 0.17 (best)
- FSD EPE: 0.40 (best)

这证明 MDM 学到的 RGB representation 比 DINOv2 和 MoGe 更 "geometric-aware"，作为 stereo backbone init 更好。

### 6.4 Ablation 缺失

Paper 没有显式 ablation table，但可以从实验对比中提取：
- DINOv2 init vs MDM init (Sec 4.2)
- Natural mask vs random mask (隐含在 Protocol 1 vs 2 性能差异)
- ConvStack vs MAE decoder (没 ablate)
- Multi-layer aggregation vs single-layer (没 ablate)

这是 paper 的一个 limitation——希望后续 work 能补 ablation。

---

## 7. Extensions & Applications

### 7.1 Video Depth Completion (zero-shot temporal consistency)

尽管只在 static image 上 train，LingBot-Depth 在 30 FPS 640×480 video 上展现 **temporal consistency without explicit temporal modeling**。

四个 challenging scenarios (Fig. 11, 12):
- (a) Glass Lobby: 透明玻璃墙
- (b) Rowing Machine: 窗户反射
- (c) Gym: 镜子
- (d) Aquarium Tunnel: 折射玻璃

Comparison: Orbbec raw depth (大块 holes) vs ZED-mini stereo depth (also fails on glass) vs LingBot-Depth (complete, temporally stable)。

**Why temporal consistency emerges**? Karpathy 你会直觉到：
1. RGB-D 数据本身有 implicit temporal continuity（相邻帧 appearance 相似）
2. DINOv2 init 的 patch representation 在邻帧间 stable
3. 没有 temporal smoothing 反而避免 over-smoothing 引入的 lag

但 paper 没量化 temporal flickering metric（如 TFS, temporal consistency ratio）。这是个可以做 follow-up 的点。

### 7.2 Online 3D Point Tracking with SpatialTrackerV2

把 LingBot-Depth 作为 SpatialTrackerV2 [34] 的 RGB-D 前端，替换 VGGT [27]。

**Modification**:
- 用 online SpatialTrackerV2 (不依赖 VGGT for initial pose/depth)
- frame-wise extrinsics 用 SE(3) identity init
- Bundle Adjustment 只在 tracked VO points 上，no global BA
- 不 finetune SpatialTrackerV2

**Camera tracking** (Fig. 13a): glass-heavy 室内场景，raw depth 严重 drift，refined depth 平滑准确 trajectory。

**Object motion tracking** (Fig. 13b): 4 dynamic scenarios，query points on moving objects, 3D trajectories 用 rainbow color 显示。Trajectories coherent，证明 depth 几何精度足够支持 dynamic tracking。

参考:
- SpatialTrackerV2: https://arxiv.org/abs/2507.12562
- VGGT: https://arxiv.org/abs/2503.11651

### 7.3 Robotic Dexterous Grasping (最 impressive 应用)

**Hardware**:
- Rokae XMate-SR5 robotic arm
- X Hand-1 dexterous hand (22 DOF)
- Orbbec Gemini 335 RGB-D camera

**Pipeline**:
```
RGB-D observation
  → LingBot-Depth → completed depth → point cloud
  → RGB features via DINOv2 ViT-L/14
  → point cloud features via Point Transformer (DP3-style [38])
  → Diffusion Policy predicts N × 22 hand pose
```

**Training**: HOI4D dataset [14]，把 human hand-object interaction retarget 到 dexterous hand 通过 3D keypoint correspondence。

**Test objects** (4 challenging):
- Stainless steel cup (reflective)
- Transparent cup
- Toy car
- Transparent storage box

**Results** (Table 4, 20 trials each):
| Object | LingBot-Depth | Raw depth |
|--------|---------------|-----------|
| Steel cup | 17/20 | 13/20 |
| Transparent cup | 16/20 | 12/20 |
| Toy car | 16/20 | 9/20 |
| Transparent box | 10/20 | **N/A** (完全 ungraspable) |

**Key win**: transparent storage box 在 raw depth 上完全不可抓（sensor 全 fail），LingBot-Depth 给出 geometrically plausible 估计，50% 成功率。这是 zero-shot generalization 到 train distribution 之外的 impressive 案例。

参考:
- DP3 (3D Diffusion Policy): https://arxiv.org/abs/2403.03954
- HOI4D: https://arxiv.org/abs/2105.08560
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Point Transformer: https://arxiv.org/abs/2012.09164

---

## 8. 跟相关工作放一起看

### 8.1 在 Masked Modeling 谱系中的位置

| Method | Modality | Mask target | Reconstruction | Year |
|--------|---------|-------------|----------------|------|
| MAE | RGB | Random RGB patches | RGB pixels | 2021 |
| BEiT | RGB | Predicted tokens | VQ tokens | 2021 |
| iBOT | RGB | Random + self-distill | RGB + cls | 2021 |
| data2vec | Multi | Random | Teacher targets | 2022 |
| MultiMAE | Multi-modal | Random per modality | Multi-modal pixels | 2022 |
| **MDM (this paper)** | RGB-D | Sensor-induced depth | Depth values | 2025 |

MDM 的独特性：**mask 来自 sensor 的物理 failure，而非 random sampling**。这跟 synthetic 3D-aware data augmentation (e.g., DREDS, D3 Roma) 思路类似但用 self-supervised pretraining 形式表达。

参考:
- MultiMAE: https://arxiv.org/abs/2204.09144
- data2vec: https://arxiv.org/abs/2202.03555
- D3 Roma: https://arxiv.org/abs/2410.24219

### 8.2 在 Depth Completion 谱系中的位置

传统 DC 方法:
- Sparse-to-dense (CSPN, GuideNet, CompletionFormer)
- Pseudo-LiDAR 系列
- Image-guided (e.g., GuideNet, NLSPN)

学习范式:
- Supervised on synthetic + synthetic-to-real transfer
- Self-supervised via photometric consistency
- MDM (this paper): self-supervised via masked reconstruction

MDM 的优势：**不需要 paired perfect GT** for pretraining，只需要 sensor 自己输出的 (corrupted depth, RGB) pair 就能学。这是 scalable 的关键。

### 8.3 在 Monocular Depth Estimation 谱系中的位置

- MiDaS: affine-invariant, dataset mixture
- DPT: ViT + decoder
- DepthAnything V1/V2: DINOv2 + massive data
- Metric3D: camera-intrinsic-aware
- MoGe: point map, optimal supervision
- **MDM-as-MDE-init**: cross-modal pretrain → unimodal inference

MDM 给 MDE 提供了一种新 pretrain paradigm：让 RGB encoder 通过 joint embedding with depth 学到 "depth-awareness"，再剥离 depth branch。

参考:
- MiDaS: https://arxiv.org/abs/1907.01341
- DPT: https://arxiv.org/abs/2103.13413
- DepthAnything: https://arxiv.org/abs/2401.10891
- Metric3D: https://arxiv.org/abs/2307.10984

---

## 9. 一些 Open Questions & 思考方向

Karpathy 你应该会想问：

1. **Natural mask 的 distribution shift**：不同 sensor (RealSense vs Orbbec vs ZED) 的 missing pattern 不一样。Pretraining 期间混在一起，模型学到的是 "universal depth prior" 还是 "specific sensor prior"？Fig. 5 的 mask ratio distribution 显示 synthetic 和 real 有显著差异。如果 domain-specific evaluation，会有什么结果？

2. **Mask ratio 60-90% 的 ablation**：为什么是这个 range？更低 (40%) 会怎样？更高 (95%) 会怎样？我猜 60% 是因为 sensor hole 本身平均 ratio 较低，90% 上限是为了保留 enough valid tokens 做 cross-modal attention anchor。

3. **Cross-modal attention 的 entropy**：Fig. 3 的 attention 可视化显示 spatially-corresponding attention。但如果 attention entropy 在不同 layer 怎么变？是不是 layer 6/12/18/24 不同层 attend 不同 thing？这关系到 multi-layer aggregation 的 ablation。

4. **Modality embedding 的 ablation**：如果完全去掉 ME，让 RGB 和 depth token 共享所有 embedding，会怎样？我猜 attention 会 collapse 到 trivial pattern。

5. **Zero-shot metric scale**：MDM 学到的是 metric depth（有 absolute scale）还是 affine-invariant？Table 3 的 "Scl-inv" 和 "Aff-inv" 都有数字，但 paper 没强调 metric accuracy 在 MDE task 上。这是 MDE 跟 DC 的关键区分——DC 必然 metric (input depth 有 scale)，MDE 可能 lose scale。

6. **Failure cases**：paper 没显式展示 failure modes。Transparent storage box 50% 成功率说明 still far from perfect。完全 transparent 表面（无 specular cue）应该是 fundamental limit——RGB context 本身没有几何信息。

7. **Foundation model 可能性**：如果训练 scale 再大 10× (100M RGB-D pairs)，能不能成 RGB-D foundation model？10M 已经是 SOTA 级别，scale up 是 obvious next step。

8. **跟 VGGT 的关系**：VGGT 是 visual geometry grounded transformer，做 multi-view geometry in one forward pass。MDM 单帧 RGB-D，VGGT 多帧 RGB。如果结合 MD-MDM with multi-view RGB-D？

---

## 10. 我的整体 Intuition 总结

这篇 paper 的 elegance 在于：
1. **问题转化**：sensor failure → learning signal
2. **任务统一**：masking ratio 控制 task identity
3. **架构 minimalism**：standard ViT + ConvStack，无花哨 head
4. **Data scalability**：self-curated pipeline 可规模化（10M 已经是 SOTA scale）
5. **Downstream utility**：3 个真实世界 application（video、tracking、grasping）证明实用价值

**Limitations**:
- 缺 ablation table
- Failure analysis 不充分
- Temporal consistency 没量化
- 与 VGGT/SpatialTracker 范式的关系没讨论

**For your intuition, Andrej**：把 RGB-D 看成 "two modality streams"，自然 mask 是 sensor 告诉模型 "这里 appearance ambiguous, ask RGB for help"。模型通过 self-attention 学到 RGB-D cross-modal correspondence，这 correspondence generalize 到：
- Inference 时 remove depth branch → pure MDE (因为 RGB encoder 已经内化 depth reasoning)
- Inference 时给 sparse depth → DC (因为 cross-modal reasoning 学过 sparse-to-dense)
- Inference 时给 video sequence → temporal consistency emerges (因为 per-frame reasoning 已 smooth)

整个 paper 像是把 MAE 的思想在 multi-modal + physically-grounded setting 下重做，结果证明这个方向 productive。后续 work 应该会沿这个方向 scale up，可能成为 RGB-D foundation model 的雏形。

---

## References & Web Links

主要参考：
- Paper: https://github.com/robbyant/lingbot-depth
- HuggingFace checkpoint: https://huggingface.co/robbyant/lingbot-depth
- Project page: https://technology.robbyant.com/lingbot-depth

方法相关：
- MAE: https://arxiv.org/abs/2111.06377
- DINOv2: https://arxiv.org/abs/2304.07193
- iBOT: https://arxiv.org/abs/2111.07832
- MoGe: https://arxiv.org/abs/2410.19115
- FoundationStereo: https://arxiv.org/abs/2501.09862
- DepthAnythingV2: https://arxiv.org/abs/2406.09414
- VGGT: https://arxiv.org/abs/2503.11651
- SpatialTrackerV2: https://arxiv.org/abs/2507.12562

应用相关：
- DP3: https://arxiv.org/abs/2403.03954
- HOI4D: https://arxiv.org/abs/2105.08560
- Diffusion Policy: https://arxiv.org/abs/2303.04137

数据相关：
- SGM (Hirschmüller): https://ieeexplore.ieee.org/document/4359315
- DREDS: https://arxiv.org/abs/2207.09004
- TartanAir: https://arxiv.org/abs/2003.14338
- NYUv2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
- ETH3D: https://arxiv.org/abs/1702.01313
- iBims-1: https://arxiv.org/abs/1904.03092

Baseline DC:
- PromptDA: https://arxiv.org/abs/2412.14054
- PriorDA: https://arxiv.org/abs/2505.10565
- OMNI-DC: https://arxiv.org/abs/2409.17965

希望这个 walkthrough 帮你 build intuition。如果对某个细节（比如 ConvStack decoder 的具体 conv kernel size、或 attention visualization 的 quantitative metric）想深挖，告诉我，我可以再展开。
