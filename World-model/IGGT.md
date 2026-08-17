---
source_pdf: IGGT.pdf
paper_sha256: f7f0e78fa4c9887f947eb2989c706f9c762a970e0bb7a177047f9dd367704ef7
processed_at: '2026-08-05T08:58:38-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 IGGT

好，咱们坐下来喝杯咖啡聊聊这个事，不整那些花里胡哨的。

---

## 一、这个 paper 在解决什么"蛋疼"的问题

想象你看一张照片，里面有个椅子。你脑子里同时干了两件事：
- **这是啥？** —— 椅子
- **它在哪？离我多远？** —— 三米远，左边

这两件事对你来说是同时发生的、一体的。但在 AI 里，历史上一直被拆成两个独立任务做：

- **Geometry 这帮人**（COLMAP, DUSt3R, VGGT 那一挂）只管重建出 3D 点云、camera pose，输出一堆数字，完全不知道"这是椅子"；
- **Semantic 这帮人**（LangSplat, Feature-3DGS, LSeg 那一挂）只管贴 label，但要依赖别人先把 3D 重建好，自己再做 per-scene optimization。

中间还有一些"撮合派"（LSM, Uni3R）——他们想着，那我把 3D model 和 LSeg 这种 language model 直接 align 一下不就行了？结果发现三个 bug：

1. **几何被磨平了**：3D 几何是 high-frequency 信号（边缘、纹理细节），强行和"椅子"这种粗语义对齐，细节就糊了；
2. **被锁死了**：你 align 了 LSeg 就只能用 LSeg，CLIP、SigLIP、DINOv3 出来了你眼巴巴看着用不上；
3. **分不清同类**：LSeg 只知道"这是椅子"，不知道"这是椅子 A，那是椅子 B"，两个椅子混在一起。

IGGT 的作者就琢磨：**凭啥非要硬 align 啊？我能不能让一个 transformer 同时学好几何和"实例身份"两件事，然后把"具体叫啥类别"丢给现成的 VLM 去做？**

这就是整个 paper 的 intuition。

---

## 二、核心 idea：把"语义"拆成两半

这是 paper 最聪明的地方。

之前的语义 3D 重建把"语义"当成一个不可分割的整体——一个 pixel 既要有 instance identity，又要有 category label。

IGGT 把它拆开：
- **"这是 instance A 还是 instance B"** —— 我自己学（因为区分 instance 需要 cross-view 3D 一致性，2D VLM 干不了这个）；
- **"这个 instance A 是啥类别"** —— 丢给 VLM（因为 VLM 见过的 image-text pair 多，category 知识比我自己学强）。

两者之间的"通信协议"就是 **instance mask**——一张二值图，告诉你"这块像素属于哪个物体"。任何 VLM / LMM 都能吃这个接口，plug-and-play。

这就是为什么 paper 标题叫 **Instance-Grounded** ——以 instance 为锚，把 geometry 和 semantic 串起来。

---

## 三、具体怎么做的

### 1. 数据从哪来

这是个蛋疼事。要让模型同时学 geometry 和 instance，你得有标注：每张图 RGB + depth + pose + **跨视角 ID 一致的 instance mask**。

社区里没有现成的。所以作者自己造了 **InsScene-15K**，三类来源：

- **合成的**（Aria, Infinigen）：直接渲染，mask 是免费送的，准得很；
- **RE10K 视频**：用 SAM2 做 temporal propagation，一个 keyframe 出 proposals，往后传，丢了再补 keyframe；
- **ScanNet++ 真实扫描**：有 coarse 3D annotation，投影到 2D 形状糊。用 SAM2 出 sharp proposals，再用 coarse GT 给它 assign ID，迭代 refine。

这里有意思的点是：**SAM2 不是用来做最终 segmentation 的，是用来做数据标注的**。SAM2 擅长出 shape-accurate mask，但不保证 cross-view ID 一致；ScanNet++ 的 3D annotation 保证 cross-view ID 但形状糊。两者一结合，互补。

### 2. 模型长啥样

整体就是 **VGGT（1B 参数）+ 加两个 head**：

```
N 张图 → DINOv2 patch tokenize
       → 每张图前面拼一个 learnable camera token
       → 24 层交替做 intra-view self-attn + global cross-view attn
       → 出 unified tokens
       → 分两路：
           ├── Geometry Head (DPT) → pose, depth, point map
           └── Instance Head (DPT) → 8 维 instance feature
                ↑
                这里有个 Cross-Modal Fusion Block：
                instance feature 当 Query，geometry feature 当 K/V，
                做 sliding-window cross-attention
```

**关键设计：Cross-Modal Fusion Block**

为啥要这个？因为 Instance Head 如果只从 unified tokens 解码，它不知道物体边界在哪——而 geometry feature 里正好有 boundary 信息（depth discontinuity 通常对应物体边界）。

所以让 instance feature 去 query geometry feature："嘿，这块区域的几何长啥样？" 几何回答："这里有个深度跳变，应该是物体边界。"instance feature 就被 sharpen 了。

用 sliding window 而不是 global attention，是因为 global attention 太贵（pixel 数多），window 又能保持局部细节。这是 Swin Transformer 的套路。

### 3. 怎么训练

四个 loss 加一起：

$$\mathcal{L} = \mathcal{L}_{pose} + \mathcal{L}_{depth} + \mathcal{L}_{pmap} + \mathcal{L}_{mvc}$$

前三个是 VGGT 原有的几何 loss，第四个是新加的 **multi-view contrastive loss**：

- **Pull term**：同一 3D instance 在不同视角的 pixel feature 要尽量靠近；
- **Push term**：不同 instance 的 pixel feature 要至少推开一个 margin。

公式：

$$\mathcal{L}_{mvc} = \lambda_{pull} \sum_{\text{同 ID}} d(f_i, f_j) + \lambda_{push} \sum_{\text{不同 ID}} \max(0, M - d(f_i, f_j))$$

直觉：把 8 维 feature space 想象成一个房间，每个 instance 是一群人聚在一起的一团，不同团之间要离得够远。HDBSCAN 之后就能轻松把团分开。

**为啥是 8 维？** 因为场景里 instance 数一般几十个，8 维空间足够线性可分。低维还有个好处：存储和算力便宜，不像 CLIP 那种 512 维要扛一堆。

**训练 schedule 也很有讲究**：backbone LR = $10^{-6}$，head LR = $10^{-5}$，差 10 倍。因为 backbone 是 VGGT 预训练好的 1B 参数，不能大 LR 把它搞坏；head 是新的，可以猛一点。

### 4. 推理时怎么用

这部分最妙。IGGT 训练完输出的是 8 维 latent feature，**不知道是椅子还是桌子**。怎么用？

**第一步：Instance Spatial Tracking**

把所有视角的 8 维 feature 拼一起，跑 HDBSCAN 聚类。HDBSCAN 比 K-Means 好的地方是不用指定 K，对变密度鲁棒（大物体 feature 密、小物体稀）。聚出来的每个 cluster = 一个 3D object instance。然后 re-project 回 2D，得到跨视角 ID 一致的 mask。

这就是 spatial tracking——为啥能在大相机运动下不丢目标？因为 feature 是 3D-grounded 的，相机怎么动，同一 3D 点的 feature 天然接近。

**第二步：Open-Vocabulary Segmentation**

把 IGGT 出的 mask 喂给 VLM（比如 OpenSeg）。OpenSeg 给每张图出 language feature；用 IGGT 的 mask 做 average pooling，每个 instance 得到一个 compact feature；再和 text embedding 做 dot product，就出 category 了。

mask pooling 有两个好处：
- 给 VLM 提供了 sharp boundary（VLM 自己出的 feature 经常糊）；
- Instance-level 决策比 pixel-level 鲁棒（pixel 噪声被 average 掉）。

**第三步：QA Scene Grounding**

更狠。把 mask 直接当 prompt 喂给 GPT-4o / Qwen2.5-VL，在图像上用红框高亮同一 instance，问 LMM："这几张图里红框是不是同一个东西？" LMM 答 yes/no，聚合 yes 的 mask 就完成 grounding。

这里 LMM 提供 reasoning，IGGT 提供 pixel-precise mask，两者互补。LMM 单独干这事不行（pixel 精度差），IGGT 单独干也不行（不知道类别），合起来才行。

---

## 四、实验结果说明了啥

我挑几个最关键的数：

**Spatial Tracking（Tab. 1, 2）**：
- IGGT: T-SR **98.66% / 98.90%**（几乎不丢目标）
- SAM2\*: 71.25% / 57.89%（大视角变化经常丢）
- SpaTracker+SAM: 38.57% / 23.68%（基本不能用）

这说明：**2D temporal tracking 在大相机运动下是死路**。必须有 3D prior。

**Reconstruction（vs VGGT）**：
- ScanNet 上 IGGT 比 VGGT 略差（1.90 vs 1.84）；
- ScanNet++ 上 IGGT 反超 VGGT（2.61 vs 2.75）。

为啥 ScanNet++ 反超？因为 ScanNet++ 是真实扫描，object boundary 信号更明显，instance branch 的监督反过来帮 geometry 收敛。这就是 paper 说的"mutual enhancement"——**joint training 不光不拖后腿，还能反过来帮 geometry**。

**Open-Vocab 2D Segmentation（ScanNet++）**：
- LSeg: 22.61 mIoU
- Feature-3DGS: 22.47
- LSM: 17.88
- IGGT + OpenSeg: **31.31**

为啥 IGGT 这么强？因为 IGGT 的 mask 给了 VLM sharp boundary，而 LSeg/LSM 这种 pixel-level alignment 边界糊。

**Plug-and-Play 验证（Tab. 3）**：
- LSeg: 22.72 mIoU
- CLIP: 21.52
- OpenSeg: **31.31**

不同 VLM 在不同场景有不同强项。OpenSeg 在 ScanNet++ 这种常见室内类别上强，CLIP 在 brand-name 类别上强（比如 'DALL-E' 牌子）。这正说明 plug-and-play 的价值——**没有一个 VLM 通用最强，按需切换才是正解**。

---

## 五、一句话总结

**IGGT 把"语义"拆成"instance identity"（自己学）和"category label"（VLM 来）两件事，用 instance mask 当通信协议，让 geometry 和 instance 在一个 transformer 里 joint train，达到 mutual enhancement，同时把 category 灵活性留给下游 VLM/LMM。**

为啥这思路 work？因为：
- Instance 区分需要 3D cross-view consistency → 2D VLM 做不到 → 必须自己学；
- Category 识别需要海量 image-text pair → 自己学不过 VLM → 交给 VLM；
- 两者接口（mask）简单通用 → 任何 VLM 都能接 → 灵活；
- Joint train 让 geometry 和 instance 互相 regularize → 1+1>2。

---

## 六、个人的吐槽和联想

**1. HDBSCAN 的边界**

paper 自己承认，HDBSCAN 聚类出的 mask 边界不如 SAM2 准。这是 unsupervised clustering 的通病。未来如果想再上一个台阶，应该接一个 Mask2Former-style 的 query-based head，直接预测 mask，不再走 feature clustering 这条路。但那样就不再是 "pure latent + post-hoc clustering" 的优雅设计了。

**2. 8 维 feature 够不够**

8 维在几十个 instance 的场景够用。但要是超市货架几百瓶洗发水，8 维可能不够线性可分。可以考虑 hierarchical feature 或 dynamic dimension。

**3. 没用 language supervision 训**

IGGT 训练时完全没用 text。这意味着它没法内化任何 category prior。如果哪天 VLM 失灵了（比如训了个新 category），IGGT 也跟着废。理论上可以做 self-distillation，把 VLM 的 category prior 蒸进 IGGT 自己，但那样就破坏了"解耦"的优雅。

**4. 跟 SpatialVLA 的关系**

SpatialVLA 是把 spatial representation 喂给 VLA（机器人 action model）。IGGT 的输出（point map + instance mask）正好是 SpatialVLA 想要的 input。所以 IGGT + SpatialVLA 是天然的 pipeline——机器人看一眼，IGGT 给它 3D 几何 + 实例分组，SpatialVLA 据此规划动作。

**5. 跟 SAM2 的对比**

SAM2 是 2D video segmentation 的王者，但在大相机运动下会丢。IGGT 用 3D prior 解决了这个问题。但 SAM2 的优势是 mask 边界极其 sharp，IGGT 在这一点上还差。所以一个理想的 future work 是：**用 IGGT 的 8 维 feature 做 coarse tracking + SAM2 做 mask refine**。粗定位靠 3D，精修靠 2D。

**6. 跟 3DGS 的关系**

IGGT 现在只输出 point map + feature。如果想做 novel view synthesis with instance control，需要再加 3DGS head。Anysplat 已经做了 feed-forward 3DGS，把 IGGT 的 instance feature 接进去应该挺自然。这能做出 "feed-forward 3DGS + instance-aware editing" 的 demo，应该很酷。

**7. Open-vocabulary 的根本困境**

IGGT 把 category 留给 VLM。但 VLM 在 3D 场景下也有局限——比如 VLM 没见过的小众类别就识别不了。最理想的状态是 IGGT 内化一些常见 category prior，对没见过的再 fallback 到 VLM。但这就破坏了 paper 的 clean 设计。research 总是在 elegance 和 capability 之间 trade-off。

---

## 七、最大启发

IGGT 让我最受启发的不是技术细节，而是这个 **"problem factorization" 思路**：

> 不要试图让一个模型干所有事。先把问题拆成几个 sub-problem，每个 sub-problem 由最适合它的 module 负责，然后用一个 minimal interface 把它们连起来。

具体到这个 paper：
- Geometry → IGGT 自己（需要 pixel-aligned 3D 信号）
- Instance identity → IGGT 自己（需要 cross-view 3D 一致性）
- Category → VLM（需要海量 image-text 知识）
- Reasoning → LMM（需要复杂推理能力）

**Interface = instance mask**，简单到任何模型都能吃。

这个思路其实非常 general。比如做 robot foundation model：
- Perception 模块（看场景）
- Language 模块（理解指令）
- Action 模块（输出动作）
- Memory 模块（记住历史）

中间用啥通信？可能就是 point cloud + object mask + trajectory 这种 minimal structured representation。

不必硬搞一个大一统的 end-to-end model，把所有东西塞进一个 loss 里。**Modular design with clean interfaces** 在工程上更 scalable，在科学上更 interpretable。

---

希望这次"人话版"讲清楚了。简单说就是：**与其硬把几何和语言捏到一起，不如让几何和实例身份自己学好，然后拿 mask 当 API，需要啥语言能力就接啥 VLM。** 干净、灵活、还 work。

要是哪点还想细聊（比如为啥 HDBSCAN 不用 spectral clustering、为啥 backbone LR 差 10 倍、为啥不直接用 SAM2 出 mask 而要绕一圈 latent feature），随时叫我。

---

# IGGT: Instance-Grounded Geometry Transformer — 深度技术讲解

你好 Andrej！很高兴为你梳理这篇 IGGT (Instance-Grounded Geometry Transformer) 的工作。我会尽量把每个细节掰开揉碎，把直觉 (intuition) 和工程实现 (engineering) 都讲透，并尽量多联想相关工作。

---

## 1. 核心问题与 motivation

这篇 paper 想解决一个长期存在的 **二分法困境 (dichotomy)**：

- **Geometry-first 方法** (COLMAP, DUSt3R, VGGT, MASt3R) 只关心 low-level 几何重建 (camera poses, depth maps, point maps)，不输出 semantic understanding；
- **Semantic-first 方法** (LangSplat, LangSurf, Feature-3DGS, LSeg, OpenSeg) 关心 high-level semantic features，但通常依赖已经重建好的 3D 表征 (3DGS, NeRF)，做 per-scene optimization。

中间还有一些 **强行 align 的方法** (LSM, Uni3R, Panst3R)，它们把 3D spatial model 和某个 specific VLM (比如 LSeg) 做 feature alignment。但这带来三个问题：

1. **Low-frequency over-smoothing**：3D geometry 含 fine-grained high-frequency 信号 (边缘、纹理)，强行和 language 这种 high-level concept 对齐会把几何细节平滑掉；
2. **架构耦合 (architectural lock-in)**：被绑到 LSeg 上，就享受不到 CLIP / SigLIP / SigLIP 2 / DINOv3 等更强 VLM；
3. **Instance-level blindness**：2D image–text 训练的 VLM 只能区分 semantic category (chair vs. table)，不能区分同一类的不同 instance (chair A vs. chair B)。

IGGT 的 insight：**与其强行把 geometry 和 language feature 对齐，不如把 geometry 和 instance-level clustering feature 在同一个 transformer 里 joint train，让模型自己学到 "几何结构如何帮助实例区分，实例区分又如何反过来约束几何"**。然后通过 instance masks 作为通用接口 (universal API)，让任何 VLM / LMM 都能 plug-and-play 接进来。

这其实是非常 elegant 的设计——**把"语义"分解成"instance 分组" + "每个 instance 的类别"两件事**，前者由 IGGT 学，后者由现成 VLM 提供。

参考链接：
- VGGT: https://wangjianno1.github.io/VGGT/
- DUSt3R: https://dust3r.eu/
- MASt3R: https://naver.github.io/mast3r/
- LSM: https://largespatialmodel.github.io/
- LangSplat: https://langsplat.github.io/
- Feature-3DGS: https://xnihao.github.io/feature-3dgs/
- Panst3R: https://vincent1bonhomme.github.io/PAnSt3R/
- Uni3R: https://arxiv.org/abs/2508.03643

---

## 2. InsScene-15K 数据集

数据是这种 instance-level 3D-consistent task 的命脉。社区里很难找到一个数据集同时满足：**多视角 RGB + camera pose + dense depth + cross-view ID-consistent instance masks**。

IGGT 团队构建 InsScene-15K 用了三类来源：

| 来源类型 | 数据集 | 性质 |
|---|---|---|
| Synthetic | Aria Digital Twin, Infinigen Indoors | 渲染器直接生成 mask GT，无需 post-processing |
| Video captured | RE10K (RealEstate10K) | 真实视频，需要 SAM2 时序传播 |
| RGBD scanned | ScanNet++ | 真实扫描，已有 3D coarse mask 但粗糙 |

### 2.1 SAM2-driven 数据引擎

对 RE10K (图2a)，他们设计了一套**迭代式 SAM2 视频传播流程**：

1. 在初始 frame 上跑 SAM (注意是 SAM 不是 SAM2) 生成 dense proposals；
2. 把这些 proposals 当 prompt 喂给 SAM2 做时序传播 (temporal propagation)；
3. 当未分割区域增大时，触发 new keyframe 检测，重新跑 SAM 找新物体；
4. 处理完整个视频后做 **bidirectional propagation** pass，保证时间一致性。

这是 SAM2 video object segmentation 的标准 trick，但关键在于 unsegmented area threshold 的设计——太敏感会引入噪声，太松会漏物体。

对 ScanNet++ (图2b)，他们解决的是另一个问题：3D annotation 投影到 2D 后 mask 形状粗糙。所以：

1. 先把 3D 实例投影到 2D 得到 coarse mask (保证 ID 跨视角一致)；
2. 跑 SAM2 生成 shape-accurate 但无 ID 的 proposals；
3. 用 coarse GT mask 给 proposals assign ID；
4. 同 ID 合并，迭代直到覆盖全部图像区域。

图 3 给的对比非常直观：原 ScanNet++ GT mask 形状糊，IGGT refined mask 既准又跨视角 ID 一致。

这部分让我想到 SAMPart3D (https://arxiv.org/abs/2504.11451) 和 PartField 的数据生成思路，也是用 SAM 系列做粗粒度标注的 refine。

参考：
- SAM2: https://arxiv.org/abs/2408.00714
- ScanNet: https://github.com/ScanNet/ScanNet
- ScanNet++: https://scan-net.github.io/scannetpp/
- Aria Digital Twin: https://www.projectaria.com/
- Infinigen Indoors: https://infinigen.org/

---

## 3. IGGT 架构

整体结构 = **VGGT backbone (1B params)** + **Geometry Head** + **Instance Head** + **Cross-Modal Fusion Block**。

### 3.1 输入输出形式化

给定 N 张图像 $\{I^i \in \mathbb{R}^{H \times W \times 3}\}_{i=1}^N$，模型 $\mathcal{F}$ 一次性 forward 输出：

$$
\mathcal{F}: \{I_i\}_{i=1}^N \mapsto (t_i, D_i, P_i, S_i)_{i=1}^N \tag{1}
$$

变量解释：
- $t_i$：第 i 个视角的 camera parameters (extrinsics + intrinsics)；
- $D_i \in \mathbb{R}^{H \times W}$：dense depth map；
- $P_i \in \mathbb{R}^{H \times W \times 3}$：point map (世界坐标系下的 3D 点云)；
- $S_i$：3D-consistent instance-level feature map。

注意这里关键：**P_i 直接是 point map，不是 depth + pose 重建**。这是 DUSt3R / VGGT 范式的好处——避免了显式 pose → depth → point map 的误差级联，直接回归 3D 坐标。

### 3.2 Large Unified Transformer

继承 VGGT 设计，1B 参数。三步：

**Step 1: DINOv2 patch token 提取**
用预训练 DINOv2 (https://arxiv.org/abs/2304.07193) 把每张图编码成 patch-level tokens。

**Step 2: Camera token concatenation**
为了支持任意数量多视角输入且保持 permutation equivariance (输入顺序不影响输出)，每个视角的 token sequence 前面 concatenate 一个 learnable camera token。这个 token 后面被 Geometry Head 的 camera predictor 读取来回归 pose。

**Step 3: 24 blocks 交替 attention**
24 个 block 交替做：
- **Intra-view self-attention**：同一视角内 patch 之间互相 attend，捕获 local image context；
- **Global-view cross-attention**：所有视角的 patch tokens 互相 attend，捕获 cross-view 几何一致性 (这是关键，让不同视角看到同一 3D 点的 token 能 communicate)。

输出 unified tokens $\{\mathbf{T}_i \in \mathbb{R}^{M \times D}\}_{i=1}^N$，其中 $M$ 是每张图的 token 数 (一般 $= H/14 \times W/14$，DINOv2 用 patch size 14)，$D$ 是 token 维度。

这个设计直接来自 VGGT，paper 没改 backbone，只是加 head。这是一个工程上很务实的选择——VGGT 已经训好 1B 参数，从它热启动比从头训容易得多。

### 3.3 Geometry Head 和 Instance Head

两个 head 都是 DPT-like architecture (Ranftl et al., 2021)，做 progressive upsampling + multi-scale fusion：

$$
\{F_i^{pt}\} = \Phi_{pt}(\{\mathbf{T}_i\}), \quad \{F_i^{ins}\} = \Phi_{ins}(\{\mathbf{T}_i\}) \tag{2}
$$

变量解释：
- $F_i^{pt} = \{F_{i,(l)}^{pt}\}_{l=1}^4$：geometry feature pyramid，4 个尺度 (DPT 标准)；
- $F_i^{ins} = \{F_{i,(l)}^{ins}\}_{l=1}^4$：instance feature pyramid，4 个尺度；
- $\Phi_{pt}, \Phi_{ins}$：两个 DPT-like decoder，参数独立。

Geometry Head 进一步分三个子模块：
- **Camera predictor**：从 camera-specific token 回归 extrinsics + intrinsics；
- **Depth predictor**：DPT 解码出 dense depth；
- **Point predictor**：DPT 解码出 dense 3D point map。

### 3.4 Cross-Modal Fusion Block (关键创新)

这是 paper 的核心创新点之一。问题：Instance Head 默认只从 unified tokens 解码，**没有显式用到 pixel-level 几何细节**，对 object boundary 和 spatial layout 不敏感。

解决方案：在每一层 pyramid 都注入 geometry feature 到 instance feature：

$$
\hat{F}_{i,(l)}^{ins} = F_{i,(l)}^{ins} + \mathscr{F}_{win}(Q = F_{i,(l)}^{ins}, K = F_{i,(l)}^{pt}, V = F_{i,(l)}^{pt}) \tag{3}
$$

变量解释：
- $\hat{F}_{i,(l)}^{ins}$：fusion 后的 instance feature，第 i 个视角第 l 层；
- $Q, K, V$：cross-attention 的 query/key/value；
- $\mathscr{F}_{win}$：window-shifted cross attention。

直觉：**Query 来自 instance branch (问 "我应该属于哪个 instance")，Key/Value 来自 geometry branch (回答 "这是 3D 空间里这块区域的几何特征")**。这样 instance feature 被几何结构 sharpen，对物体边界更敏感。

为什么用 window-shifted 而不是 global？避免 $O(N^2)$ 全图 attention 复杂度，同时 window-shifted (类似 Swin Transformer, Liu et al., 2021) 能 cross-window 信息交流。这里也直接呼应了 Swin 的设计哲学。

图 11 的 ablation 显示：没有这个 fusion block，instance head 训练 loss 收敛更慢，PCA 可视化里椅子边缘更糊。

最后所有 $\{\hat{F}_{i,(l)}^{ins}\}$ 拼起来过一个 3×3 conv，映射到 8 维 instance feature：
$$O_{ins} \in \mathbb{R}^{N \times 8 \times H \times W}$$

**为什么是 8 维？** 这是一个 hyperparameter 选择。8 维足以区分场景里的不同 instance (一般场景几十个 instance，8 维 feature space 足够 linearly separable)，同时降低存储和计算开销。这跟 LSeg 用 CLIP 的 512 维、SAMPart3D 用更高维 feature 形成对比——IGGT 选 8 维是因为它不打算直接和 language 对齐，只是做 instance clustering 的 latent space。

参考：
- DPT: https://arxiv.org/abs/2103.13413
- Swin Transformer: https://arxiv.org/abs/2103.14030
- SAMPart3D: https://arxiv.org/abs/2504.11451

### 3.5 3D-Consistent Contrastive Supervision

这是另一个关键创新。要保证 instance feature 在多视角下一致 (同一 3D 物体的 feature 跨视角应该 close，不同物体应该 apart)，用 multi-view contrastive loss：

$$
\mathcal{L}_{mvc} = \lambda_{pull} \cdot \sum_{\substack{p_i, p_j \in \mathcal{P} \\ m(p_i) = m(p_j)}} d(f_{p_i}, f_{p_j}) + \lambda_{push} \cdot \sum_{\substack{p_i, p_j \in \mathcal{P} \\ m(p_i) \neq m(p_j)}} \max(0, M - d(f_{p_i}, f_{p_j})) \tag{4}
$$

变量逐项解释：
- $\mathcal{P}$：采样像素集合 (从所有视角的所有像素中 batch sample)；
- $p_i, p_j$：两个采样像素；
- $m(p_i)$：像素 $p_i$ 所属的 instance ID (来自 InsScene-15K 的 GT mask)；
- $f_{p_i} \in \mathbb{R}^8$：像素 $p_i$ 处的 instance feature (即 $O_{ins}[i, :, p_i.y, p_i.x]$)；
- $d(\cdot, \cdot)$：L2 distance，作用于 normalized features；
- $M = 1.0$：margin 超参，控制不同 instance 之间至少要推开多远；
- $\lambda_{pull} = 2.0, \lambda_{push} = 1.0$：两个 term 的权重，pull 比 push 重，说明更强调"同一 instance 跨视角要紧靠"。

第一项 (pull)：所有同 ID 的 pixel pair，feature 距离要尽量小；
第二项 (push)：所有不同 ID 的 pixel pair，距离至少要大于 margin M，否则有 loss。

这就是经典的 **InfoNCE / Contrastive Loss** 变种，在 segmentation 监督里也常见 (类似 DenseCLIP, Mask2Former 的 contrastive query loss)。

总训练 loss：

$$
\mathcal{L}_{overall} = \mathcal{L}_{pose} + \mathcal{L}_{depth} + \mathcal{L}_{pmap} + \mathcal{L}_{mvc} \tag{5}
$$

前三项 follow VGGT 训练 paradigm，第四项是 IGGT 新增。

**直觉理解**：joint training 让 backbone 同时被几何信号和 instance 信号驱动。Geometry 提供了 cross-view 的硬约束 (两个视角的同一 3D 点必须回归到同一世界坐标)，而 instance contrastive 提供了软的语义约束 (同一 instance 的像素 feature 必须接近)。两种信号在 backbone 里融合，互相 regularize。

这与 OpenScene (https://arxiv.org/abs/2212.08858)、LERF (https://lerf.io/) 的 philosophy 不同——那些方法把 CLIP feature 直接蒸馏到 3D，IGGT 只学 instance identity，把 category 留给下游 VLM。

---

## 4. Instance-Grounded Scene Understanding (推理时的 plug-and-play)

这部分最 cool。训练完 IGGT 后，instance feature $O_{ins}$ 是个 8 维 latent，没法直接读出 "chair" 或 "table"。怎么用？

### 4.1 Instance Spatial Tracking

对 $O_{ins}$ 跑 **HDBSCAN** (https://arxiv.org/abs/1911.02226) 做密度聚类，把所有视角的 2D instance feature 聚成 K 个 cluster，每个 cluster 对应一个 unique 3D object instance。

HDBSCAN 优势：不需要预先指定 K (相比 K-Means)，对噪声鲁棒，能处理变密度 cluster (这正是 3D 场景里 instance feature 分布的特点——大物体 feature 密集，小物体稀疏)。

聚完类后，把 cluster label re-project 到对应 pixel locations，得到 3D-consistent 2D instance masks $\{M_{i,k}^{ins}\}_{k=1}^K$。

**为什么这能实现 spatial tracking？** 传统 SAM2 之类 tracker 依赖 2D appearance temporal continuity，相机大幅运动时会丢目标。IGGT 的 instance feature 是 3D-grounded 的 (从 unified tokens + geometry fusion 来的)，所以同一 3D 物体跨视角 feature 自然接近，HDBSCAN 一聚就聚到一起。

Tab. 1, 2 的 T-SR 几乎 100%，对比 SAM2\* 大幅掉点，就是直接证明。

### 4.2 Open-Vocabulary Semantic Segmentation

到这里 IGGT 只知道"有 K 个 instance"，但不知道每个 instance 是什么类别。这就引入外部 VLM：

以 OpenSeg 为例：
1. OpenSeg 对每张图产生 image-wise language features $\{F_i^{lang} \in \mathbb{R}^{D \times H \times W}\}_{i=1}^N$；
2. 对每个 instance mask $M_{i,k}^{ins}$ 做 **mask average pooling**：$\mathbf{f}_k^{lang} = \frac{1}{|M_k|} \sum_{p \in M_k} F^{lang}[p]$；
3. 得到每个 instance 的 compact representation $\{\mathbf{f}_k^{lang} \in \mathbb{R}^D\}_{k=1}^K$；
4. 再用 text embeddings 做 dot product 分类。

mask pooling 的好处：
- 把 mask prior 注入 visual-language space (相当于给 VLM 提供了 sharp boundary)；
- Average pooling 平滑了 VLM 在 single pixel 上的预测噪声；
- Instance-level decision 比 pixel-level decision 更鲁棒。

Tab. 3 的 ablation 比较了 LSeg / CLIP / OpenSeg 三种 VLM 接入，发现：
- **LSeg / OpenSeg**：global context 强，对 background class (cabinet) 准；
- **CLIP**：text alignment 强，对 fine-grained category (e.g., 'DALL-E' 这种 brand-name) 准。

这正是 paper 强调的 "plug-and-play" 价值——不同 VLM 有不同强项，按需切换。

### 4.3 QA Scene Grounding

更进一步，把 instance masks 喂给 LMM (GPT-4o, Qwen2.5-VL)：
1. 给 LMM 看 N 个视角的图像；
2. 在每个视角上 highlight 同一 instance k 的 mask region (red 高亮)；
3. 问 LMM "is this the same object across all these views?" yes/no；
4. 聚合所有 yes 响应，对应 mask 拼成最终 segmentation。

图 9 给的 Teatime 场景例子，对比 vanilla Gemini 2.5 Pro，IGGT + Qwen-VL 的多视角一致性显著更好。

直觉：LMM 的 reasoning 能力很强但缺乏 pixel-level precision；IGGT 提供 pixel-precise 的 instance masks 作为 prompt，让 LMM 在已经 segmented 的 region 上做 reasoning，两者完美互补。

参考：
- OpenSeg: https://arxiv.org/abs/2112.12143
- LSeg: https://arxiv.org/abs/2201.03546
- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP 2: https://arxiv.org/abs/2502.14786
- DINOv3: https://arxiv.org/abs/2508.10104
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Gemini 2.5: https://arxiv.org/abs/2507.06261

---

## 5. 实验结果分析

### 5.1 Instance Spatial Tracking (Tab. 1, 2)

ScanNet:
- IGGT: T-mIoU **69.41**, T-SR **98.66**
- SAM2\*: T-mIoU 53.74, T-SR 71.25
- SpaTracker+SAM: T-mIoU 26.43, T-SR 38.57

ScanNet++:
- IGGT: T-mIoU **73.02**, T-SR **98.90**
- SAM2\*: 44.16 / 57.89
- SpaTracker+SAM: 16.15 / 23.68

**关键观察**：T-SR 几乎 100% 说明 IGGT 几乎不丢目标，而 SAM2\* 在大视角变化下经常丢。这背后的本质是——SAM2\* 是 2D temporal model，相机视角剧烈变化时 appearance 差异巨大，2D propagation 失败；IGGT 用 unified tokens 做 cross-view attention，几何 prior 让同一 3D 物体的 token 必然互相 attend，instance feature 跨视角一致。

### 5.2 Reconstruction Quality (Tab. 1, 2)

ScanNet:
- VGGT: Abs. Rel 1.84, $\tau$ 83.60
- IGGT: Abs. Rel **1.90**, $\tau$ **83.71**

ScanNet++:
- VGGT: 2.75 / 85.41
- IGGT: **2.61 / 85.66**

直觉：IGGT 在 ScanNet 上 Abs. Rel 略差 (1.90 vs 1.84)，但在 ScanNet++ 上反而更好 (2.61 vs 2.75)。这说明 **joint training instance head 在 ScanNet++ 这种真实扫描数据上对 geometry 有 mutual enhancement**——可能因为 ScanNet++ 的 instance mask 提供了 object boundary 信号，sharpen 了 depth 估计。

但 ScanNet 上略差，可能是因为 InsScene-15K 训练时 instance branch 抢了一些 capacity，对纯 geometry 略有 negative transfer。

### 5.3 2D / 3D Open-Vocabulary Segmentation (Tab. 1, 2)

ScanNet 2D mIoU:
- LSeg: 58.11
- OpenSeg: 42.33
- LSM (multi): 53.40
- Feature-3DGS: 57.69
- IGGT: **60.46**

ScanNet++ 3D mIoU:
- LSM: 15.17
- Feature-3DGS: 10.59
- IGGT: **20.14**

直觉：3D mIoU 上 IGGT 领先 LSM 5 个点、领先 Feature-3DGS 10 个点，原因有两层：
1. IGGT 的 instance mask 提供 sharp object boundary，而 LSM 的 pixel-level LSeg feature 是糊的；
2. IGGT 的 point map 质量比 Feature-3DGS (依赖 per-scene 3DGS optimization 但输入视角稀疏) 更准。

### 5.4 Cross-Modal Fusion Ablation (Fig. 11)

训练 loss 曲线对比：去掉 fusion block 后 loss 更高、收敛更慢。PCA 可视化中椅子边缘明显糊。这印证了 "geometry 信息对 instance segmentation 至关重要" 这个 intuition。

### 5.5 VLM Plug-and-Play Ablation (Tab. 3)

ScanNet++ 上：
- LSeg: mIoU 22.72, mAcc 63.56
- CLIP: mIoU 21.52, mAcc 61.36
- OpenSeg: mIoU **31.31**, mAcc **70.78**

OpenSeg 在 ScanNet++ 上显著优于 CLIP，因为 ScanNet++ 的类别大多是 common indoor object (cabinet, chair, floor)，OpenSeg 的 global context 更适合。但 CLIP 在 brand-name 类别上更强。这正说明 "no single VLM wins all"——plug-and-play 价值就在这里。

---

## 6. 评估指标 (Metrics)

### 6.1 Instance Spatial Tracking

$$
\mathrm{T-mIoU}(o) = \frac{1}{T} \sum_{t=1}^T \frac{|\hat{M}_t^o \cap M_t^o|}{|\hat{M}_t^o \cup M_t^o|}
$$

变量：$o$ 是某个 object，$T$ 是总视角数，$\hat{M}_t^o, M_t^o$ 是第 $t$ 个视角的预测 mask 和 GT mask。

$$
\mathrm{T-SR}(o) = \mathbb{1}\left[\forall t \in \{1,...,T\}, |\hat{M}_t^o| > 0\right]
$$

变量：$\mathbb{1}[\cdot]$ 是 indicator function。T-SR 衡量 object 是否在所有视角都被成功 track (非空 mask)。

### 6.2 Reconstruction

- **Abs. Rel** = $\frac{1}{|P|}\sum_{p \in P} \frac{|D_p - \hat{D}_p|}{D_p}$：相对深度误差，越小越好；
- **$\tau$ (Inlier Ratio)** = 阈值 1.03 下的 inlier 比例，越大越好。

### 6.3 3D Semantic Segmentation mIoU

流程 (Fig. 13)：
1. 从 per-image point maps 获取 RGB 3D points；
2. 用 2D open-vocab segmentation 结果给 3D points assign semantic label；
3. 3D points 做 voxelization；
4. 在 voxel 上算 mIoU。

直觉：voxel-based 3D mIoU 比直接在 point 上算更稳定，避免离群点污染。

---

## 7. 训练细节

- 8 × NVIDIA A800 GPU，2 天；
- AdamW optimizer；
- Backbone LR = $1 \times 10^{-6}$ (因为已经 VGGT 预训练好，不能太大 LR 破坏)；
- Geometry Head LR = Instance Head LR = $1 \times 10^{-5}$；
- 每个 batch 随机选 1-12 个 scene，每个 scene 24 张图；
- $\lambda_{pull} = 2.0, \lambda_{push} = 1.0, M = 1.0$。

注意到 backbone LR 比 head LR 小 10 倍——这是经典的 fine-tuning 范式，避免预训练 1B 模型被新任务破坏。

---

## 8. 与相关工作的深层联系

### 8.1 vs. VGGT

VGGT (Wang et al., CVPR 2025) 是 IGGT 的直接 predecessor。VGGT 已经做到 "feed-forward N 张图 → pose + depth + pointmap + tracks"，但完全没有 semantic。IGGT 在它上面加 instance head + cross-modal fusion + contrastive loss，把 semantic 理解塞进同一个 transformer。

值得注意：IGGT 没有改 VGGT 的 backbone 结构，只加 head。这暗示 VGGT 的 unified tokens 已经隐含了某些 instance-discriminative 信息，只是没有被 decode 出来。IGGT 的 contrastive loss 等于在 backbone 里 carve 出一个 instance-aware subspace。

### 8.2 vs. LSM (Large Spatial Model)

LSM (Fan et al., NeurIPS 2024) 是最先尝试 unify geometry + semantics 的 feed-forward model。但 LSM 把 3D spatial feature 和 LSeg feature 直接 align，所以：
- 必须用 LSeg，不能换 VLM；
- 只能 category-level，不能 instance-level；
- 几何被 language alignment over-smooth。

IGGT 通过 "instance mask 作为通用接口" 完美解决了这三点。

### 8.3 vs. Panst3R / MASt3R+

Panst3R (Zust et al., 2025) 是 DUSt3R + panoptic segmentation 的组合。但它 freeze geometry module，semantic 单独跑，没有 mutual benefit。IGGT 用 joint training，geometry 和 instance 互相 enhance。

### 8.4 vs. SAMPart3D / PartField

SAMPart3D 和 PartField 学 3D feature field 做 part segmentation，也用 contrastive loss + HDBSCAN 聚类。但它们是 per-scene optimization (3D feature field)，IGGT 是 feed-forward across scenes。

IGGT 的 8 维 feature + HDBSCAN 这套 pipeline 明显借鉴 SAMPart3D 的设计。

### 8.5 vs. Thinking in Space (Yang et al., CVPR 2025)

Fei-Fei Li 组的 "Thinking in Space" (https://arxiv.org/abs/2504.14091) 也研究 MLLM 怎么理解 3D 空间，但走的是 benchmark + reasoning 路线，不训新模型。IGGT 是 model + method 路线，把 3D 理解能力直接 bake 进 transformer。

### 8.6 vs. SpatialVLA

SpatialVLA (Qu et al., 2025) 把 spatial representation 用到 VLA (vision-language-action) model，用于机器人。IGGT 的输出 (instance masks + point maps) 可以作为 SpatialVLA 之类的下游 input，pipeline 上互补。

### 8.7 vs. Anysplat / GGRt

Anysplat (https://arxiv.org/abs/2505.23716) 和 GGRt (https://arxiv.org/abs/2410.10642) 都是 feed-forward 3DGS 重建。它们输出 3D Gaussians 而非 point map + instance feature。如果 IGGT 的 instance feature 加到 3DGS 表征里，可能能做到 "feed-forward 3DGS + instance tracking"。

### 8.8 vs. Feature-3DGS / Feature Splatting

Feature-3DGS (Zhou et al., CVPR 2024) 把 CLIP feature 蒸到 3DGS。但它是 per-scene optimization + 不能跨 scene 泛化。IGGT feed-forward 跨 scene 工作，更适合机器人等 online 应用。

---

## 9. Limitations 和未来方向

Paper 自己指出 (Sec. A.7)：
- HDBSCAN 的 mask 边界还不能比 SAM2 准；
- 未来可能加 DETR-style instance head (Mask2Former)。

我的额外思考：
1. **8 维 feature 可能不够复杂场景**：如果场景有几百个 instance (比如超市货架)，8 维 feature space 可能不够 linearly separable。可以做成可学习维度或者 hierarchical clustering；
2. **No language supervision in training**：IGGT 完全没用 text 训练，依赖下游 VLM 做 categorization。如果 joint train 时加 text supervision (像 OpenScene)，可能 instance + category 在一个 model 里都解决；
3. **No dynamic scene support**：IGGT 假设静态场景。视频里物体移动时 unified tokens 的一致性假设会破坏。这跟 SAM2 的 video segmentation 设计哲学有冲突；
4. **No explicit 3DGS output**：现在只有 point map + instance feature，没有 3D Gaussian。如果想做 novel view synthesis with instance control，需要再加 3DGS head；
5. **Camera token 的 equivariance**：现在 permutation equivariance 是通过 camera token 实现的，但当 N 很大 (e.g., 100+) 时 global attention 仍然 quadratic。可以替换成 sparse attention 或 linear attention (像 Linformer / Performer)；
6. **Open-vocabulary 仍依赖 VLM**：这意味着 VLM 失败时 IGGT 也失败。未来可以做 self-distillation，让 IGGT 内化部分 VLM 的 category prior。

---

## 10. 直觉总结 (Final Intuition)

IGGT 给我的最大启发是：

**"Don't fuse representations by force; instead, factor the problem so each module does what it's best at, then connect them with a clean interface."**

具体到这个 paper：
- **3D Geometry**：IGGT 自己学 (因为它需要 fine-grained pixel-aligned 3D signal)；
- **Instance Identity**：IGGT 自己学 (因为它需要 cross-view 3D consistency，2D VLM 做不到)；
- **Semantic Category**：交给 VLM (因为这是 VLM 的强项，且 VLM 在快速进化)；
- **Complex Reasoning**：交给 LMM (因为 reasoning 是 LMM 的强项)。

**Interface** 就是 instance masks——一个简单的 binary mask，任何 VLM / LMM 都能接收。

这个思路其实非常 general：未来在做 multimodal foundation model 时，与其强行 align 不同 modality 的 representation，不如设计一个 minimal interface (mask, point, box 等) 让不同 expert model 通信。这跟 Flakey AI 的 modular design philosophy 也有点像。

另一个 takeaway：**joint training 是 mutually beneficial 的**——Instance head 不光从 geometry 拿信息 (通过 cross-modal fusion)，也反过来 sharpen geometry (ScanNet++ 上 Abs. Rel 反超 VGGT)。这说明两个 supervision signal 在 backbone 里互相 regularize，是一种 implicit multi-task learning benefit。

参考总览：
- IGGT GitHub: https://github.com/lifuguan/IGGT_official
- VGGT: https://wangjianno1.github.io/VGGT/
- DUSt3R: https://dust3r.eu/
- MASt3R: https://naver.github.io/mast3r/
- SAM2: https://arxiv.org/abs/2408.00714
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- LSeg: https://arxiv.org/abs/2201.03546
- OpenSeg: https://arxiv.org/abs/2112.12143
- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP 2: https://arxiv.org/abs/2502.14786
- LSM: https://largespatialmodel.github.io/
- Panst3R: https://vincent1bonhomme.github.io/PAnSt3R/
- Uni3R: https://arxiv.org/abs/2508.03643
- Feature-3DGS: https://xnihao.github.io/feature-3dgs/
- LangSplat: https://langsplat.github.io/
- LangSurf: https://arxiv.org/abs/2412.17635
- SAMPart3D: https://arxiv.org/abs/2504.11451
- HDBSCAN: https://arxiv.org/abs/1911.02226
- DPT: https://arxiv.org/abs/2103.13413
- ScanNet: https://github.com/ScanNet/ScanNet
- ScanNet++: https://scan-net.github.io/scannetpp/
- RE10K: https://arxiv.org/abs/1805.09817
- Aria: https://www.projectaria.com/
- Infinigen: https://infinigen.org/
- Thinking in Space: https://arxiv.org/abs/2504.14091
- SpatialVLA: https://arxiv.org/abs/2501.15830
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- Anysplat: https://arxiv.org/abs/2505.23716
- GGRt: https://arxiv.org/abs/2410.10642
- Swin Transformer: https://arxiv.org/abs/2103.14030
- OpenScene: https://arxiv.org/abs/2212.08858
- LERF: https://lerf.io/

希望这个拆解能帮到你建立 IGGT 的完整 intuition，Andrej！如果想深入讨论任何 sub-design 决策 (比如为什么 8 维 feature、为什么 HDBSCAN 而非 spectral clustering、为什么不用 Mask2Former-style query-based head)，可以继续聊。
