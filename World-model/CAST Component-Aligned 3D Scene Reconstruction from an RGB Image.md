---
source_pdf: CAST Component-Aligned 3D Scene Reconstruction from an RGB Image.pdf
paper_sha256: 9619b7b9caf03dbffcaaab17238838316d1c0ec047f04794344e8dc36220773d
processed_at: '2026-08-03T15:05:12-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CAST 用人话说

## 一句话概括

给一张照片，CAST 把里面每个东西单独生成一个 3D model，然后一个个摆回正确的位置，最后再用物理规则把位置调对，让整个 scene 不穿模、不悬浮、看起来合理。

## 为什么这件事难

你拿一张厨房照片，想重建 3D。乍看好像就是把每个东西 3D 化然后拼起来，但仔细想全是坑：

**坑 1：你只看得到物体的一面**
照片里的椅子你只能看到前面，后面被自己挡住了。如果让 AI 直接生成，它可能给你生成一个只有前半截的椅子。这是 occlusion 问题。

**坑 2：你不知道物体在 3D 里怎么摆的**
AI 生成的椅子默认是"标准朝向"——正面朝某个方向。但照片里的椅子可能侧着、斜着、甚至倒着。你得把它旋转到跟照片里一样的角度。这是 pose estimation 问题。

**坑 3：物体之间的关系**
就算每个椅子都摆对了，你把吉他放进冰箱里，看起来就怪。或者 surfboard 浮在面包车上面 30 厘米，也怪。这些关系照片里是隐含的，但 AI 生成时不会自动考虑。这是 physical constraint 问题。

CAST 就是把这三个坑分别用三个模块去填。

## 整体流程像什么

想象你是个 3D 美术，拿到一张参考图要做 3D scene。你的工作流大概是：

1. 先看图，识别里面有哪些东西（椅子、桌子、杯子...）
2. 每个东西单独建一个 3D model
3. 把每个 model 摆到图里对应的位置
4. 检查一下有没有穿模、悬浮，微调一下

CAST 就是把这个人类工作流自动化了。每个步骤对应一个模块。

## 模块逐个讲

### 模块 0：Preprocessing——"看图认东西"

这一步用一堆现成的 vision model：

- **Florence-2**：识别图里有啥东西，给每个东西画个框
- **GPT-4v**：过滤一下，把 Florence-2 误识别的东西去掉（比如把墙上的污渍认成画）
- **GroundedSAMv2**：给每个东西画精细的轮廓 mask
- **MoGe**：估计 depth，把照片"立起来"变成一个 rough 的 3D point cloud

这一步的产出是：
- 每个物体的 image patch + segmentation mask
- 每个物体的 partial point cloud（从全局 point cloud 切出来的）
- 全局 camera 参数

intuition：先把 scene 拆成 per-object 信息，后面才能独立处理每个物体。

### 模块 1：ObjectGen——"建单个 3D model"

这是核心模块，做的事是：给一张椅子的照片，生成一个完整的 3D 椅子 mesh。

backbone 是 CLAY 那套架构——先有个 VAE 把 3D shape 压成 latent code，再有个 diffusion model 在 latent 空间生成。这部分是预训练好的，在 Objaverse（50 万个 3D model）上学过。

关键创新在两个 conditioning：

**Occlusion handling 用 DINOv2 的 MAE**

DINOv2 训练时被随机 mask 掉 75% 的 patch，逼着它学会从局部推断整体。CAST 利用这一点：把 segmentation mask 之外的区域（即被这个物体"挡住"或不属于这个物体的区域）在 patch token 层面替换成 [mask] token，DINOv2 就能 infer 出完整物体的 feature。

为什么不直接做 2D inpainting？因为 inpainting 是在像素层面 hallucinate，错一步后面全错。在 feature 层面 infer 更鲁棒——feature 是 high-level 语义，本来就容错。

**Partial point cloud conditioning**

光看 image feature 不够，因为 DINOv2 feature 是 high-level 的，没有 pixel-level alignment。生成出来的椅子可能 scale 不对、形状细节不对。

所以加一个 partial point cloud 当额外 condition。但有个鸡生蛋问题：ObjectGen 要的是 canonical space（标准朝向）的 point cloud，而你手里只有 scene space（相机视角）的 point cloud。怎么办？先不用，让 ObjectGen 只靠 image 生成一个粗的，再用 AlignGen 把 scene-space point cloud 变到 canonical space，再喂回 ObjectGen。这就是后面要说的 iterative loop。

训练时怎么模拟 partial point cloud？从 Objaverse 的 3D asset 渲染多视角，用 MoGe / Metric3D 估计 depth（带噪声），再 unproject 成 point cloud。还用 $\alpha$ 插值 GT depth 和 estimated depth，让 model 见过从干净到噪声的各种情况。还随机 mask 掉一些 primitive shape（圆、矩形）模拟 occlusion。

关键设计选择：**partial point cloud 必须在 canonical space**，不做 random rotation/scale/translation augmentation。这样 model 学到的是"point cloud 长这样，geometry 就长这样"的直接对应关系，不会被 augmentation 扰动。

### 模块 2：AlignGen——"摆到正确的位置"

ObjectGen 出来的 mesh 在 canonical space，朝向是标准的。但照片里的椅子可能朝任意方向。怎么把它摆回去？

传统方法用 ICP（Iterative Closest Point），但 ICP 是 local optimizer，初始位置不对就卡在 local minima。对称物体（圆柱杯）更糟，ICP 可能转 180 度反过来。

CAST 的 trick：**生成一个 canonical-space point cloud，再闭式算 transform**。

具体来说，AlignGen 也是一个 diffusion model，输入是：
- scene-space partial point cloud $\mathbf{q}$（你手里有的）
- ObjectGen 输出的 geometry latent $Z$（表示 canonical-space 的完整 geometry）

输出是一个 canonical-space 的 partial point cloud $\mathbf{P}$，跟 $\mathbf{q}$ 是 pointwise corresponded（同样的 N 个点，只是坐标系不同）。

有了 $\mathbf{P}$ 和 $\mathbf{q}$ 的对应关系，直接套 Umeyama algorithm（1991 年的经典闭式解）算出 rotation、translation、scale。

为什么这样设计？

1. **Multi-modal pose**：对称物体有多个等价 pose，diffusion 可以 sample 多次取最 confident 的，不会像 regression 那样平均成 invalid pose。
2. **避免 ICP 局部极小**：diffusion 用 image feature 当 condition，能跨过 local minima。
3. **避免 differentiable rendering 的 occlusion 问题**：DR 在 RGB 有 occlusion 时 loss 乱掉，point cloud + geometry latent 对 occlusion 不敏感。

intuition：把 pose estimation 重新 cast 成 generation problem，用 diffusion 的 multi-modal 能力处理 pose 的歧义性。

### 模块 3：Iterative loop——"互相磨"

这是 paper 最巧妙的地方。问题是这样：

- ObjectGen 要 canonical-space point cloud 当 condition
- 但你一开始只有 scene-space point cloud
- AlignGen 要 geometry latent 当 condition
- 但 geometry latent 要 ObjectGen 先生成

鸡生蛋。解法是 iterative：

**第 0 轮**：
- ObjectGen 只用 image feature 生成一个粗 geometry（不用 point cloud condition，$\beta=0$）
- AlignGen 用这个粗 geometry + scene-space point cloud，算出粗 transform

**第 1 轮**：
- 把 AlignGen 输出的 canonical-space point cloud 当 condition 喂回 ObjectGen，$\beta$ 调大一点
- ObjectGen 这次有 geometric prior，geometry 更准
- AlignGen 用更准的 geometry 重新算 transform

**第 2、3 轮**：继续磨，$\beta$ 越来越大，最后 $\beta=1$，point cloud condition 完全起作用，geometry 和 transform 都收敛。

这就像 EM 算法：E-step 估 latent（transform），M-step 优化 parameter（geometry），来回迭代。$\beta$ 从 0 到 1 是 annealing，从 prior（image-only）逐步过渡到 likelihood（point cloud condition）。

paper 说实际跑 2-3 次就收敛了，因为 ObjectGen 7 秒、AlignGen 1 秒，整个 loop 十几秒搞定。

### 模块 4：Physics-aware correction——"调到物理合理"

经过前面 pipeline，每个物体都摆好了，但还是会出问题：surfboard 浮在 van 上面 20 厘米，guitar 穿进 cooler 里。

为什么不直接用现成的物理引擎（PyBullet、MuJoCo）跑一下让它 settle？paper 给三个 reason：

1. **Scene 不完整**：照片里看不到的东西（比如支撑 guitar 的架子）没重建。物理引擎跑的话 guitar 会掉下来，但照片里 guitar 明明是 stable 的。
2. **Geometry 不完美**：generator 出的 mesh 不是 perfect convex，要 convex decomposition。太细了碰撞面崎岖物体乱跳，太粗了 visual mesh 和 collision mesh 不一致看起来浮空。
3. **Initial penetration**：物理引擎不擅长处理初始就 deeply penetrate 的状态。

所以 CAST 自己写了个"静态物理"：不模拟 dynamics，只确保当前 timestep 满足 pairwise constraint。

具体做法：

1. **用 GPT-4v 抽 relation graph**：用 Set-of-Mark prompting，给图里每个物体标上彩色数字，让 GPT-4v 输出它们之间的 6 种细粒度关系：
   - Stack（A 在 B 上面）
   - Lean（A 倚着 B）
   - Hang（A 从 B 上面挂着）
   - Clamped（B 从两侧夹住 A）
   - Contained（A 在 B 里面）
   - Edge/Point（最小接触）
   
   再映射成 Support（单向）或 Contact（双向）。用 ensemble（多次 prompt + 投票）减少 VLM hallucination。

2. **定义 cost function**：基于 SDF（Signed Distance Field）。
   - $D_i(\mathbf{p}) = 0$：点在物体表面
   - $D_i(\mathbf{p}) < 0$：点在物体内（penetration）
   - $D_i(\mathbf{p}) > 0$：点在物体外
   
   **Contact cost**（双向）：惩罚穿透 + 确保至少一个接触点
   **Support cost**（单向）：被支撑物体应该贴在支撑物体上，既不悬空也不穿进
   
3. **Optimize**：只调 rotation + translation（scale 固定），用 PyTorch autodiff 做 gradient descent。SDF 用 Open3D 预计算。

intuition：把"物理合理"形式化成 pairwise cost function，让 gradient descent 把物体推到合理位置。比物理引擎简单可控，因为只关心静态合理性，不关心 dynamics。

## 为什么这么设计——核心 design philosophy

CAST 的核心 thesis：**modular generation > end-to-end generation** for scene reconstruction。

理由：

1. **Decoupling enables best-of-breed**：object generator 可以独立升级（CLAY → Trellis → 下一个），physics module 可以独立 swap，relation extractor 可以独立升级。Native 3D generator 进步了，CAST 自动受益。

2. **Decoupling enables editability**：每个 object mesh 独立可编辑，每个 pose 独立可调，每个 relation 独立可改。美工可以拿 CAST 的输出继续手动改。End-to-end 黑盒做不到。

3. **Decoupling matches 3D production pipeline**：3D 美术工作流本来就是 modular 的——asset → layout → animation → lighting。CAST 输出直接可导入 Unreal / Unity。

代价是 pipeline 长、hand-engineered。如果未来 native 3D generator 能直接吃 image 输出 scene-level 3D（类似 Sora 直接生成 video），可能简化。但目前 native 3D generator 还在 object-level，scene-level 还有难度，所以 CAST 这种 modular 方法是当下最佳实践。

## 关键 intuition 提炼

1. **Occlusion 不该在 2D 层面解决**：不要 inpainting，在 feature 层面用 MAE infer 更鲁棒。错误传播链短。

2. **Pose 不该 regression，该 generation**：symmetric object 有 multi-modal pose，diffusion sample 多次取最 confident 比 regression 平均好。

3. **鸡生蛋用 iterative EM 解决**：ObjectGen 和 AlignGen 互相 condition，β annealing 让 prior 逐步让位给 likelihood。

4. **VLM commonsense 形式化成 SDF cost**：GPT-4v 抽 relation graph，再用 SDF 写成可微 cost function，让 gradient descent 自动调 pose。比物理引擎简单可控。

5. **Modular pipeline 让 foundation model 各司其职**：Florence-2 检测、SAM 分割、MoGe depth、DINOv2 feature、CLAY geometry、GPT-4v relation——每个都是该 task 的 SOTA，组合起来比 end-to-end 训一个 monster model 更好。

## 跟其他工作的关系

- **Gen3DSR**：也是 generation-based，但用 2D inpainting 处理 occlusion，错一步后面全错。CAST 在 feature 层面用 MAE，跳过 2D 错误传播。
- **ACDC**：retrieval-based，从 dataset 检索相似 object。超出 dataset domain 就 fail。CAST 是 generative，open-vocabulary 任意 image 都行。
- **Midi**：用 multi-instance diffusion 学 spatial relation，但需要 GT 3D + annotation 训练。CAST 不需要 scene-level supervision，relation 用 VLM zero-shot。
- **PhyRecon**：用 differentiable rendering + physical simulation，但需要 multi-view。CAST 单 image，只调 pose 不动 geometry。

## 一句话总结 CAST 的贡献

CAST 把 single-image scene reconstruction 重新定义为一个 **generation + alignment + constraint** 的 modular problem，每个模块用 SOTA foundation model，用 iterative refinement 把它们耦合，用 VLM commonsense 抽 physical constraint 形式化成 SDF cost。

这种 design pattern 在 robotics、LLM agent、video generation 里都会反复出现——foundation model 当 tool，reasoning loop 当 glue，physical / semantic constraint 当 verification。值得作为 mental model 内化。

## 主要参考

**Method components**:
- Florence-2: https://arxiv.org/abs/2311.06242
- GroundingDINO: https://arxiv.org/abs/2303.05499
- SAM 2: https://arxiv.org/abs/2408.00714
- GroundedSAM: https://arxiv.org/abs/2401.14159
- DINOv2 (MAE): https://arxiv.org/abs/2304.07193
- MoGe: https://arxiv.org/abs/2410.19115
- Metric3D: https://arxiv.org/abs/2307.10952
- 3DShape2VecSet: https://arxiv.org/abs/2303.07005
- CLAY: https://arxiv.org/abs/2406.13797
- Trellis: https://arxiv.org/abs/2412.01506
- Objaverse: https://objaverse.allenai.org/
- Set-of-Mark: https://arxiv.org/abs/2310.11441
- Umeyama algorithm: https://ieeexplore.ieee.org/document/88573
- Open3D: https://arxiv.org/abs/1801.09847

**Baselines / related scene reconstruction**:
- Gen3DSR: https://arxiv.org/abs/2404.03421
- ACDC (Digital Cousins): https://arxiv.org/abs/2410.07408
- Midi: https://arxiv.org/abs/2412.03558
- InstPIFu: https://arxiv.org/abs/2207.08256
- 3D-Front dataset: https://arxiv.org/abs/2011.09192

**Physics-aware & related**:
- PhyRecon: https://arxiv.org/abs/2411.18777
- Physcene: https://arxiv.org/abs/2405.02478
- Atlas3D: https://arxiv.org/abs/2405.18515
- PhysGaussian: https://arxiv.org/abs/2311.12127
- SayPlan: https://arxiv.org/abs/2312.14125
- Real-to-sim-to-real: https://arxiv.org/abs/2403.03949

**Other 3D generation baselines**:
- DreamGaussian: https://arxiv.org/abs/2309.16653
- LRM: https://arxiv.org/abs/2311.04400
- Wonder3D: https://arxiv.org/abs/2310.21208
- SyncDreamer: https://arxiv.org/abs/2309.03453
- CAT3D: https://arxiv.org/abs/2405.10314
- Flash3D: https://arxiv.org/abs/2406.04343
- Hyper3D OmniCraft (HDR env): https://hyper3d.ai/omnicraft/hdri

---

# CAST: Component-Aligned 3D Scene Reconstruction 深度解析

## 1. 问题本质：为什么 single-image scene reconstruction 很难

Single-image scene reconstruction 的核心难点不在"单个 object 怎么生成"——这部分已经被 3DShape2VecSet、CLAY、Trellis 这类 native 3D generator 解决得差不多了。真正的难点在于**scene 的 combinatorial structure**：N 个 object、N×(N-1)/2 个 pairwise relation、3N 个 pose 参数（每个 object 6 DoF + scale）共同决定了 scene 的 plausibility。这就是 paper 在 Introduction 里反复强调的 "objects exist within networks of relations" 的 Latour-style 社会学隐喻。

具体来说有三个互相纠缠的 sub-problem：

**(a) Per-object geometry 在 occlusion 下如何完整**：一张 RGB image 里看到的 chair 通常只有前两条腿可见，后两条腿被 seat 遮住。如果直接把 image feature 喂给 generator，generator 会 hallucinate 出一个只有前腿的 chair。这就是 paper Section 4.1 要解决的核心问题。

**(b) Canonical-space object 怎么摆回 scene-space**：generator 输出的 mesh 在 canonical [-1,1]³ 朝向是"标准"的（chair 朝 +z，杯子开口朝 +y），但 image 里的 chair 可能朝任意方向，且 scale 未知。传统方法要么假设 view-aligned（直接放在 image plane），要么用 ICP。前者不 robust，后者对 outlier 和 symmetric geometry 极敏感。

**(c) 物体之间的物理关系**：即便每个 object pose 都对，把一个 guitar mesh 放进 cooler mesh 内部，scene 就 physically implausible。这种 constraint 是 image 隐含的，但很难从 pixel 里直接读出来。

CAST 的核心 contribution 就是把这三个 sub-problem 解耦成一个 iterative 的 generation pipeline，再用一个 VLM-driven 的 constraint graph 把它们缝合。

## 2. Pipeline Overview（对应 Fig. 2）

整个 pipeline 可以拆成四个 stage：

```
RGB image
   ↓
[Preprocessing] Florence-2 + GPT-4v filter → GroundedSAMv2 → MoGe depth
   ↓
[Per-object generation] ObjectGen ⟷ AlignGen (iterative)
   ↓
[Texture generation] (off-the-shelf, 类似 CLAY)
   ↓
[Physics-aware correction] GPT-4v relation graph → SDF-based optimization
   ↓
Final scene mesh
```

关键 insight 是：**不要 end-to-end 直接生成整个 scene**，而是把 scene 拆成 per-object asset + per-object similarity transform + pairwise constraint，这样既保留了 native 3D generator 的 high-fidelity geometry，又能 edit / simulate / animate。

## 3. Preprocessing：scene analysis

Preprocessing 阶段用一串 foundation model 抽出三个信息流：

1. **Object detection + caption**：Florence-2（unified vision model，同时做 detection、grounding、caption）输出 bounding boxes + per-object description。Florence-2 选用是因为它一个模型就覆盖多种 vision task，避免了拼装多个 model 的开销。然后 GPT-4v 过滤 spurious detection——这一步的 intuition 是：open-vocabulary detector 会在复杂 scene 里 over-detect（比如把墙上的一块颜色识别成"画"），GPT-4v 的 commonsense 能 prune 这些。

2. **Segmentation mask**：GroundedSAMv2 给每个 object 精细 mask。这个 mask 后面会作为 occlusion mask 喂给 DINOv2 MAE——即 mask 之外的区域被当作"被该 object 遮住的区域"。

3. **Scene-space point cloud + camera**：MoGe（"Monocular Geometry"，来自同一个 ShanghaiTech/Deemos 团队）输出 metric-aligned pixel-wise 3D point cloud。每个 object 的 mask 把 global point cloud 切成 per-object partial point cloud `q_i ∈ ℝ^{N×3}`，记在 camera/scene coordinate system 里。

这里有个隐藏的设计选择：MoGe 给的 depth 是 metric 还是 relative？paper 说 "pixel-aligned point cloud + global camera parameter"，意思是 MoGe 给的是 canonical scale 的 metric depth，因此不同 image 之间可以比较。这一点对后面 AlignGen 的 Umeyama 是关键——如果只有 relative depth，没法恢复 scale。

## 4. Perceptive 3D Instance Generation：核心模块

### 4.1 ObjectGen：occlusion-aware 3D object generator

ObjectGen 的 backbone 沿用了 3DShape2VecSet / CLAY 的 architecture：

**Geometry VAE**（公式 1）：
$$Z = \mathcal{E}(X), \quad \mathcal{D}(Z, \mathbf{p}) = \mathrm{SDF}(\mathbf{p})$$

- $X$：从 object surface 均匀采样的 point cloud（一般 4096 点）
- $Z$：VAE encoder $\mathcal{E}$ 输出的 unordered latent code，长度固定（CLAY 用 2048 个 latents，每个 8 维）
- $\mathbf{p}$：任意 3D query point
- $\mathrm{SDF}(\mathbf{p})$：query point 到 surface 的 signed distance
- $\mathcal{D}$：VAE decoder，输入 (Z, p)，输出 SDF 值
- 用 Marching Cubes 从 SDF 场提取 mesh

VecSet 表示比 triplane / voxel 优势在于：unordered，对 permutation invariant，且容易做 cross-attention conditioning。

**Geometry LDM**（公式 2）：
$$\epsilon_{\mathrm{obj}}(Z_t; t, c) \rightarrow Z$$

- $Z_t$：在 timestep $t$ 加噪后的 latent
- $t$：diffusion timestep，从 $T=1000$ 退化到 0
- $c$：DINOv2 编码的 image feature（patch tokens）
- $\epsilon_{\mathrm{obj}}$：24-layer DiT，约 1.5B 参数，在 Objaverse（约 500K assets）上预训练

**Occlusion handling via MAE**（公式 3）：
$$\mathbf{c}_m = \mathcal{E}_{\mathrm{DINOv2}}(\mathbf{I} \odot \mathbf{M})$$

- $\mathbf{I}$：cropped object image patch
- $\mathbf{M}$：binary mask，1 表示该 patch token 是 occluded 区域（即 segmentation mask 之外），会被替换成 DINOv2 的 [MASK] token
- $\odot$：element-wise，这里是概念性的，实际是在 patch token 层面 mask

DINOv2 训练时用了 random mask augmentation（典型的 75% mask ratio），所以 inference 时即使把 chair 的下半部分 mask 掉，feature 仍然能 infer 出完整 chair 的 semantic。这是 paper 的关键 insight：**不要做 2D inpainting**（Gen3DSR 那样），因为 inpainting 会引入 2D hallucination，再 lift 到 3D 会放大误差。**直接在 feature 层面 infer** 更鲁棒。

**Canonical-space partial point cloud conditioning**（公式 4）：
$$\epsilon(Z_t; t, c, \mathbf{P}_{\mathrm{disturb}}) \rightarrow Z$$

其中 $\mathbf{P}_{\mathrm{disturb}}$ 是关键：
$$\mathbf{P}_{\mathrm{disturb}} = \alpha \cdot \mathbf{P}_{\mathrm{gt}} + (1-\alpha) \cdot \mathbf{P}_{\mathrm{est}}, \quad \alpha \sim U[0,1]$$

- $\mathbf{P}_{\mathrm{gt}}$：从 GT depth map unproject 的 partial point cloud（训练时来自 3D asset 的 multi-view render + GT depth）
- $\mathbf{P}_{\mathrm{est}}$：用 MoGe / Metric3D 估计的 depth，再 align 到 GT depth（用 median + median absolute deviation 做 scale/shift）——这模拟 inference 时的 noisy depth
- $\alpha$：每个 training sample 随机采样，让 model 同时见过 clean 和 noisy 两种情况

这个 interpolation trick 的 intuition 是：训练时 distribution 覆盖从 perfect 到 estimated depth 的全谱，inference 时无论 MoGe 估计得多准多不准，model 都在训练 distribution 内。

还有一个关键设计：**partial point cloud 必须在 canonical space**。paper 说："we maintain the alignment of partial point clouds with the geometry in our training data set. Unlike methods that apply random scaling, translation, or rotation to augmented point clouds, our aligned partial point clouds ensure that the generative model can more effectively conform to the input point clouds' inherent structure."

这意味着：训练时，partial point cloud 是 object 在 canonical pose 下的 multi-view render 的 depth unprojection——即 point cloud 本身就在 canonical space。但 inference 时，从 input image 拿到的 scene-space point cloud 是 camera 坐标系的，不在 canonical space。这就引出了 AlignGen 的必要性。

为了进一步模拟 occlusion，还做了 random primitive mask（circle/rectangle）on depth map。

Conditioning 机制：partial point cloud 用 FPS 采样到 2048 个点，encode 成 512 维 feature token，通过 cross-attention 注入 DiT。

### 4.2 AlignGen：generative pose alignment

AlignGen 的核心 idea：**不要直接回归 similarity transform 参数，而是 generate 一个 transformed canonical-space point cloud**，再用 Umeyama algorithm 算 transform。

**公式 5**：
$$\epsilon_{\mathrm{align}}(\mathbf{P}_t; t, \mathbf{q}, Z) \rightarrow \mathbf{P}$$

- $\mathbf{q} \in \mathbb{R}^{N \times 3}$：scene-space partial point cloud（从 MoGe + mask 拿到）
- $Z$：ObjectGen 输出的 geometry latent（表示 canonical-space 完整 geometry）
- $\mathbf{P} \in \mathbb{R}^{N \times 3}$：要 generate 的，**canonical space** 的 partial point cloud，且对应 $\mathbf{q}$ 的每一点
- $\mathbf{P}_t$：加噪版本
- $\epsilon_{\mathrm{align}}$：24-layer point cloud diffusion transformer，150M 参数

然后 $\mathbf{P}$ 和 $\mathbf{q}$ 是 pointwise corresponded（同样的 N），所以可以套 Umeyama：
$$\arg\min_{R, \mathbf{t}, s} \sum_i \| \mathbf{P}_i - (s R \mathbf{q}_i + \mathbf{t}) \|^2$$

- $R$：rotation（SO(3)）
- $\mathbf{t}$：translation
- $s$：scale
- 闭式解，数值稳定

这个 design 的妙处：

1. **Multi-modal pose**：对称物体（如圆柱杯、四方盒子）有多个等价 valid pose。Direct regression 会把它们平均，得到 invalid pose。Diffusion 可以 sample 多次，paper 说"sample 多个 noise realization，aggregate transforms，select most confident"。

2. **避免 ICP 局部极小**：ICP 是 local optimizer，初始 pose 不对就 stuck。Diffusion 用 image feature 当 conditioning，能跨过 local minima。

3. **Avoid differentiable rendering 的 occlusion 问题**：DR 在 RGB image 有 occlusion 时 loss landscape 很乱（occluded pixel 没有监督），AlignGen 用 point cloud + geometry latent，对 occlusion 不敏感。

Conditioning 细节：
- $\mathbf{q}$：concat 到 noisy $\mathbf{P}_t$ 的 feature channel——transformer 能直接学 correspondence
- $Z$：cross-attention 注入

### 4.3 Iterative procedure（Section 4.3，公式 6、7）

这是 paper 最巧妙的部分。问题是：ObjectGen 需要 canonical-space point cloud 当 conditioning，但 inference 开始时只有 scene-space point cloud，没法直接用。

solution 是 iterative：

**Step 1 - ObjectGen**（公式 6）：
$$z^{(k)} = \mathrm{ObjectGen}(c, \mathbf{p}^{(k)} \otimes \beta^{(k)})$$

- $\mathbf{p}^{(0)} = \mathbf{q}$（第一次直接用 scene-space point cloud）
- $\beta^{(k)} \in [0,1]$：point cloud conditioning scale，第一次为 0（即不用 point cloud conditioning，只靠 image），迭代中逐步增大到 1
- $\otimes$：conditioning scale，即 classifier-free guidance 的 scale

**Step 2 - AlignGen**（公式 7）：
$$\mathbf{p}^{(k+1)} = \mathrm{AlignGen}(\mathbf{q}, z^{(k)})$$

- 用新生成的 geometry latent $z^{(k)}$ 当 condition，把 scene-space $\mathbf{q}$ map 到 canonical space 的 $\mathbf{p}^{(k+1)}$

**Step 3 - Refinement**：把 $\mathbf{p}^{(k+1)}$ 反馈给 ObjectGen，下次迭代用它当 canonical-space conditioning，$\beta$ 增大。

直觉：

- 第 0 次迭代：只靠 image 生成一个粗 object，AlignGen 算粗 transform。这个粗 object 可能 scale 不对（DINOv2 feature 是 high-level 的，没有 pixel supervision）。
- 第 1 次迭代：把 AlignGen 输出的 canonical-space point cloud 当 conditioning 给 ObjectGen，β 较小。ObjectGen 这次有 geometric prior，scale 对一些。
- 第 2、3 次迭代：β 继续增大，canonical-space point cloud 越来越准，ObjectGen 输出越来越对齐 image。
- Convergence：transform 变化小于阈值或达到 max iter。

这是经典的 EM-style 优化（geometry 和 alignment 互相 refine），用 generative model 替代了 traditional EM 的 M-step。

paper 说单次 object generation ~7s，texture ~10s，alignment ~1s，所以 iter 2-3 次完全可承受。

## 5. Physics-aware Correction：让 scene 物理合理

经过上面 pipeline，每个 object 都在 scene-space 摆好了，但还是会出问题：surfboard 浮在 van 上面，guitar 穿进 cooler 里。这是因为 ObjectGen 和 AlignGen 没有考虑 inter-object constraint。

### 5.1 为什么不用现成 rigid-body simulator

paper Section 5.1 给了三个 reason：

1. **Partial scene**：image 里看不到的 object（比如支撑 guitar 的架子）没重建。物理 simulator 模拟缺失 object 时，guitar 会掉下来——但这是不对的，因为 image 里的 guitar 明显是 stable 的。
2. **Imperfect geometry**：generator 输出的 mesh 不是 perfect convex，需要 convex decomposition。Decomposition 太细，碰撞面崎岖，物体乱跳；太粗，视觉 mesh 和 collision mesh 不一致，物体看起来浮空。
3. **Initial penetration**：标准 solver 不擅长处理初始就 deeply penetrating 的状态。

所以 paper 提出 customized "static" 物理：不模拟 dynamics，只确保当前 timestep 静态稳定 + 满足 pairwise constraint。

### 5.2 Problem formulation（公式 8）

$$\min_{\mathcal{T} = \{T_1, T_2, ..., T_N\}} \sum_{i,j} C(T_i, T_j; \mathbf{o}_i, \mathbf{o}_j)$$

- $N$：object 数量
- $T_i$：第 i 个 object 的 rigid transform（只 optimize rotation + translation，scale 固定）
- $\mathbf{o}_i$：第 i 个 object 的 mesh
- $C$：cost function，依赖两个 object 之间的 relation 类型

注意这里 optimize 的是 $T_i$（pose），而 $\mathbf{o}_i$（geometry）是固定的——这一步只调位姿，不调形状。

### 5.3 Contact constraint（公式 9）

设 $D_i(\mathbf{p})$ 为 object $\mathbf{o}_i$ 在 point $\mathbf{p}$ 处的 SDF 值。

- $D_i(\mathbf{p}) = 0$：$\mathbf{p}$ 在 $\mathbf{o}_i$ surface 上
- $D_i(\mathbf{p}) < 0$：$\mathbf{p}$ 在 $\mathbf{o}_i$ 内部（penetration）
- $D_i(\mathbf{p}) > 0$：$\mathbf{p}$ 在 $\mathbf{o}_i$ 外部

Contact 是双向的，cost function 写成：

$$C(T_i, T_j; \mathbf{o}_i \to \mathbf{o}_j) = -\frac{\sum_{\mathbf{p} \in \partial \mathbf{o}_j} D_i(\tilde{\mathbf{p}}(T_j)) \mathbb{I}(D_i(\tilde{\mathbf{p}}(T_j)) < 0)}{\sum_{\mathbf{p} \in \partial \mathbf{o}_j} \mathbb{I}(D_i(\tilde{\mathbf{p}}(T_j)) < 0)} + \max(\min_{\tilde{\mathbf{p}} \in \partial \mathbf{o}_j} D_i(\tilde{\mathbf{p}}(T_j)), 0)$$

- $\partial \mathbf{o}_j$：$\mathbf{o}_j$ 的 surface（实际均匀采样的固定点集）
- $\tilde{\mathbf{p}}(T_j)$：$\mathbf{p}$ 经过 $T_j$ 变换后的位置
- $\mathbb{I}(\cdot)$：indicator function
- 第一项：$\mathbf{o}_j$ surface 上 penetrating 点的平均 penetration depth（取负，所以越深 cost 越大；分母是 penetrating 点数，避免点数变多时 cost 失衡）
- 第二项：所有 surface 点中 SDF 最小值的 max(·, 0)——如果存在任何一点还在 $\mathbf{o}_i$ 外（min SDF > 0），不惩罚；如果所有点都 penetrate（min SDF < 0），惩罚为 0（因为 max with 0）。这一项确保至少有一个 contact point。

双向：
$$\hat{C}(T_i, T_j) = C(T_i, T_j; \mathbf{o}_i \to \mathbf{o}_j) + C(T_i, T_j; \mathbf{o}_j \to \mathbf{o}_i)$$

intuition：既要避免 $\mathbf{o}_j$ 穿进 $\mathbf{o}_i$，也要避免 $\mathbf{o}_i$ 穿进 $\mathbf{o}_j$，且两个 surface 至少要"贴上"一点。

### 5.4 Support constraint（公式 10）

单向：$\mathbf{o}_i$ supports $\mathbf{o}_j$，固定 $T_i$，只 optimize $T_j$。

$$C(T_i, T_j) = |\min_{\mathbf{p} \in \partial \mathbf{o}_j} D_i(\mathbf{p}(T_j))|, \quad \text{if } \mathbf{o}_i \text{ supports } \mathbf{o}_j$$

- 只看 $\mathbf{o}_j$ surface 上离 $\mathbf{o}_i$ 最近的点的距离
- 取绝对值：无论是 penetrate（min < 0）还是 separated（min > 0），都惩罚

intuition：support 关系下，被支撑物体应该贴在支撑物体上——既不悬空，也不穿进去。绝对值让 cost 在 0 处取最小，自然把 $\mathbf{o}_j$ 拉到 $\mathbf{o}_i$ 表面。

### 5.5 Flat surface regularization（公式 11）

针对部分重建的 support 物体（如 van 只有 visible 部分，下表面不完整）：

$$C(T_i, T_j) = \frac{\sum_{\mathbf{p} \in \partial \mathbf{o}_j} D_i(\mathbf{p}(T_j)) \mathbb{I}(0 < D_i(\mathbf{p}) < \sigma)}{\sum_{\mathbf{p} \in \partial \mathbf{o}_j} \mathbb{I}(0 < D_i(\mathbf{p}) < \sigma)}$$

- $\sigma$：threshold，只考虑"贴近 surface 但不 penetrate"的点
- 这个 cost 是这些点的平均 SDF 距离，把它降到 0 就让 $\mathbf{o}_j$ 贴紧 $\mathbf{o}_i$ 的平面区域

intuition：van 下表面（重建不全）如果用 Contact cost 会被惩罚 surface 不平整，但用这个 regularizer 把 surfboard "压"到 van 顶面那块平的可见区域。

### 5.6 Scene relation graph via GPT-4v（Section 5.3）

这一步把上面的 cost function 实例化，关键是从 image 抽出 pairwise relation。

**Set-of-Mark prompting**：先给 image 里每个 object 标上彩色数字标号，再让 GPT-4v 输出 relations。

**六种细粒度 relation**（paper Appendix 给了完整 prompt）：
1. Stack: Object 1 on top of Object 2
2. Lean: Object 1 leaning against Object 2
3. Hang: Object 1 hanging from Object 2
4. Clamped: Object 2 grips Object 1 on multiple sides
5. Contained: Object 1 inside Object 2
6. Edge/Point: minimal contact

然后映射到 Support / Contact：
- 双向 edge（两个 object 互相 support）→ Contact
- 单向 edge → Support

为什么用 6 种细粒度而不是直接 Support/Contact？paper 说："Prompting GPT-4v with these nuanced relations helps eliminate potential ambiguity in binary relation classification and facilitates more accurate reasoning by GPT-4v." 即 GPT-4v 在细粒度任务上 reasoning 更清晰，再映射回抽象类别更可靠。

**Ensemble**：多次 prompt + random colorization + numeric ordering，只保留出现次数超过一半的 relation，减少 VLM hallucination。

**Directed graph**：node 是 object，edge 是 relation。Contact = bidirectional edge，Support = directed edge。

### 5.7 Optimization implementation

- Surface point：rest pose 上均匀采样固定数量点
- SDF：用 Open3D 计算
- Autodiff：PyTorch
- 只 optimize rotation + translation（scale 固定）
- 因为 cost function 是 SDF-based 可微的，可以直接 gradient descent

注意：SDF 不是深度学习的可微，而是离散几何 SDF——预先从 mesh 算好 SDF grid，然后 query。cost function 在 transform 后的 point 上 query SDF，autodiff 通过 transform 的 $T$ 参数传梯度。

## 6. 实验数据分析

### 6.1 Open-vocabulary 评测（Table 1）

| Method | CLIP↑ | GPT-4↓ | VQ↑ | PP↑ |
|--------|-------|--------|-----|-----|
| ACDC | 69.77 | 2.7 | 5.58% | 22.86% |
| Gen3DSR | 79.84 | 2.175 | 6.35% | 5.72% |
| CAST | 85.77 | 1.125 | 88.07% | 71.42% |

- CLIP score：渲染图和 input image 的 CLIP embedding 相似度（背景去除后比较）
- GPT-4 ranking：让 GPT-4 给每个方法排名，越小越好
- VQ（Visual Quality）：user study 中视觉质量获胜率
- PP（Physical Plausibility）：user study 中物理合理性获胜率

CAST 在 PP 上 71.42%，Gen3DSR 只有 5.72%——这印证了 paper 的论点：纯生成方法（不做物理约束）会出大量 penetration / floating。

ACDC 的 PP 反而比 Gen3DSR 高，因为 ACDC 是 retrieval-based，retrieved object 本身是 dataset 里的完整 model，bounding box 不重叠就没事——但 VQ 低，因为 retrieved object 跟 input 不像。

### 6.2 3D-Front 量化对比（Table 2）

| Method | CD-S↓ | FS-S↑ | CD-O↓ | FS-O↑ | IoU-B↑ |
|--------|-------|-------|-------|-------|--------|
| ACDC | 0.104 | 39.46 | 0.072 | 41.99 | 0.541 |
| InstPIFu | 0.092 | 39.12 | 0.103 | 38.29 | 0.436 |
| Gen3DSR | 0.083 | 38.95 | 0.071 | 39.13 | 0.459 |
| CAST | 0.052 | 56.18 | 0.057 | 56.50 | 0.603 |

- CD-S：scene-level Chamfer Distance
- FS-S：scene-level F-Score
- CD-O：object-level Chamfer Distance
- FS-O：object-level F-Score
- IoU-B：scene-level bounding box IoU

CAST 在所有指标上都显著领先，尤其 FS 提升明显（从 39 跳到 56），说明 geometry 质量提升很大。IoU-B 从 0.459 到 0.603 说明 layout 也更准。

公平性：所有方法都用 GT mask 替换 segmentation 模块，所以差异纯来自 reconstruction 能力。

### 6.3 Ablation study（Table 3）

| Method | CD-S↓ | FS-S↑ | CD-O↓ | FS-O↑ | IoU-B↑ |
|--------|-------|-------|-------|-------|--------|
| Vanilla | 0.079 | 53.38 | 0.069 | 52.83 | 0.515 |
| +MAE | 0.064 | 53.79 | 0.066 | 54.32 | 0.548 |
| +PCD | 0.056 | 53.91 | 0.060 | 54.60 | 0.582 |
| +iter | 0.052 | 56.18 | 0.057 | 56.50 | 0.603 |

观察：
- **MAE** 主要帮助 CD-S（occlusion 下 geometry 更完整，scene-level 提升 23%）
- **PCD**（partial point cloud conditioning）主要帮助 IoU-B（layout 准确性大幅提升）
- **Iter** 主要帮助 FS（细节 geometry 更精细）

每个模块贡献不同维度，组合起来全面提升。

## 7. 关联研究脉络与我的 intuition

### 7.1 与 Gen3DSR / ACDC 的关系

- **Gen3DSR** (Dogaru et al. 2024)：用 DreamGaussian 做 per-object generation，但用 2D inpainting 处理 occlusion，再 SDF 后处理。问题是 inpainting 错误会传播。CAST 直接在 feature 层面用 MAE infer occlusion，跳过了 2D 错误传播这一环。
  - 参考：https://arxiv.org/abs/2404.03421

- **ACDC** (Dai et al. 2024)：retrieval-based，从 3D asset dataset 检索相似 object 替换。问题是 dataset 限制——超出 dataset domain 就 fail。CAST 是 generative，open-vocabulary 任意 image 都行。
  - 参考：https://arxiv.org/abs/2410.07408

### 7.2 与 Midi (Huang et al. 2024) 的关系

Midi 用 multi-instance diffusion 学 spatial relations，但需要 ground truth 3D + annotation 训练。CAST 不需要 scene-level supervision——只用了 Objaverse 的 object-level data，scene-level relation 是用 VLM zero-shot 推理的。
- 参考：https://arxiv.org/abs/2412.03558

### 7.3 与 PhyRecon (Ni et al. 2024) 的区别

PhyRecon 用 differentiable rendering + physical simulation 优化 implicit representation，但需要 multi-view image 输入。CAST 单 image 即可，且只 optimize pose，不动 geometry。
- 参考：https://arxiv.org/abs/2411.18777 (PhyRecon)

### 7.4 与 Physcene (Yang et al. 2024a) 的区别

Physcene 做物理交互的 scene synthesis，但限定室内。CAST 是 open-vocabulary，可处理 outdoor。
- 参考：https://arxiv.org/abs/2405.02478

### 7.5 与 Set-of-Mark prompting 的关系

SoM 是 GPT-4V 视觉 grounding 的关键技术。CAST 把它用在了 relation extraction 上，配合 6 种细粒度 relation 分类 + ensemble，是 VLM commonsense reasoning 应用于 3D 几何约束的好案例。
- 参考：https://arxiv.org/abs/2310.11441

### 7.6 与 Umeyama algorithm 的关系

Umeyama (1991) 是经典 least-squares point set alignment 闭式解。给定两组 pointwise corresponded points $\mathbf{P}$ 和 $\mathbf{Q}$，求 $R, t, s$ 让 $\sum \|P_i - (sRQ_i + t)\|^2$ 最小。CAST 把它放在 generative pipeline 末端，作为 diffusion 输出到 transform 参数的桥梁——这个 design 让 diffusion 不必直接学 6D pose（multi-modal 难学），只学一个 point cloud（multimodal 用 sample 多次解决）。
- 参考：https://ieeexplore.ieee.org/document/88573

### 7.7 与 3DShape2VecSet / CLAY 的关系

CAST 直接复用 CLAY 的 VAE + DiT architecture（1.5B params），新增的只是：
- Partial point cloud conditioning（用 cross-attention 注入）
- AlignGen 模块（新模块，150M params）
- Physics-aware correction（后处理，不 train）

这意味着 CAST 可以快速跟进 native 3D generator 的进步——CLAY 升级到 Trellis，CAST 也能升级。
- 参考：https://arxiv.org/abs/2303.07005, https://arxiv.org/abs/2406.13797, https://arxiv.org/abs/2412.01506 (Trellis)

### 7.8 关于 Iterative refinement 的联想

Iterative EM-style refinement 在很多 CV task 里有先例：
- Bundle adjustment (SfM): camera pose 和 3D point 互相 refine
- NeRF: 权重和 density 互相 refine
- Iterative shape-from-silhouette: silhouette 和 shape 互相 refine

CAST 这个 pattern 等价于：geometry 和 alignment 互相 refine，β 增大是 annealing（从 prior 走向 likelihood）。这跟 alpha-go zero 的 MCTS 有点类似——先用 prior network 探索，再用 value network 修正。

### 7.9 关于 VLM + physical constraint 的联想

用 VLM 抽 commonsense constraint 是当前 trend。相关项目：
- SayPlan: LLM + 3D scene graph 做机器人 task planning
- SPATIA: LLM 调整 3D scene layout 满足语言指令
- SCENE-EDIT: LLM-guided scene editing

CAST 是这一脉在"reconstruction"任务上的应用。

### 7.10 关于 GPT-4v relation extraction 的可靠性

Paper 用 ensemble (多次 sampling + 投票)，类似 self-consistency prompting。但 relation extraction 的 ground truth 没法大规模验证（没有 dataset），只能 user study 间接验证。这是一个 limitation——未来可能用专门的 scene-graph model（如 3D scene graph）替代。

- 参考：https://arxiv.org/abs/2312.14125 (SayPlan), https://vis-www.cs.umass.edu/3DsceneGraph/ (3D scene graph)

## 8. Limitations 的解读

Paper 提到三个 limitation：

1. **Object generator 本身的质量**：CAST 依赖底层 3D generator，如果 generator 对透明 / 纺织品表现不好，CAST 也救不了。这是 modular pipeline 的通病——下游的 ceiling 由上游决定。
2. **No lighting estimation**：CAST 没建模 HDR environment map，所以用 off-the-shelf panoramic HDR tool (Hyper3D OmniCraft) + 手动 lighting。这是 scene reconstruction 跟 intrinsic image decomposition 的交界——未来 work 可以加 NeILF / PILoT 那种 joint albedo-lighting estimation。
3. **No background modeling**：只重建 discrete object，不重建 wall、floor、ceiling。这意味着放到 Unreal / Unity 里需要手动加 environment。
- 参考：https://hyper3d.ai/omnicraft/hdri, https://arxiv.org/abs/2205.08967 (NeILF)

## 9. 一些可能的研究方向联想

1. **Joint object-relation generation**：当前 pipeline 是先 object 再 relation，可以试着 end-to-end 生成 scene graph + per-object latent，类似 Midi 但用 native 3D generator。
2. **Diffusion-based physics**：现在 physics 是 gradient descent on SDF cost，可以用 score-based diffusion 直接 sample physically plausible pose distribution。
3. **Multi-view extension**：CAST 单 image，可以扩到 multi-view。Iterative alignment 部分可以替换成 SLAM-style pose graph optimization。
4. **Articulated object support**：current 假设 rigid body。Articulated object（如 drawer、door）需要 joint constraint。
5. **Video input + temporal consistency**：从 video 重建 scene 可以加 temporal constraint 防止 flicker。
6. **Real-to-sim for robotics**：CAST 是 robotics sim2real pipeline 的重要一环。配合 MuJoCo / Isaac Sim 可以做 digital twin。

- 参考：https://arxiv.org/abs/2403.03949 (Real-to-sim-to-real), https://arxiv.org/abs/2405.05941 (real-world policy eval in sim)

## 10. 总结：CAST 的核心 design philosophy

CAST 的核心 thesis 是：**modular generation > end-to-end generation** for scene reconstruction。理由：

1. **Decoupling enables best-of-breed**：object generator 可以独立迭代，physics module 可以独立 swap，relation extractor 可以独立升级。
2. **Decoupling enables interpretability & editability**：每个 object mesh 独立可编辑，每个 pose 独立可调，每个 relation 独立可改。
3. **Decoupling matches 3D production pipeline**：3D 美术工作流本来就是 modular 的（asset → layout → animation），CAST 输出直接可导入 Unreal / Unity。

代价是 pipeline 长、hand-engineered。如果未来 native 3D generator 能直接吃 image 输出 scene-level 3D（类似 Sora 那种 video diffusion 直接生成），可能简化。但目前 native 3D generator 还在 object-level，所以 CAST 这种 modular 方法是当下最佳实践。

这种 modular philosophy 也可以类比到 LLM agent——把 complex task 拆成 tool use + verification loop，而不是 end-to-end 黑盒。CAST 的 iterative refinement + VLM constraint extraction 本质上就是 reasoning loop 的 3D 版本。

---

### 主要 reference 链接

**Method components**:
- Florence-2: https://arxiv.org/abs/2311.06242
- GroundingDINO: https://arxiv.org/abs/2303.05499
- SAM 2: https://arxiv.org/abs/2408.00714
- GroundedSAM: https://arxiv.org/abs/2401.14159
- DINOv2 (MAE): https://arxiv.org/abs/2304.07193
- MoGe: https://arxiv.org/abs/2410.19115
- Metric3D: https://arxiv.org/abs/2307.10952
- 3DShape2VecSet: https://arxiv.org/abs/2303.07005
- CLAY: https://arxiv.org/abs/2406.13797
- Trellis: https://arxiv.org/abs/2412.01506
- Objaverse: https://objaverse.allenai.org/
- Set-of-Mark: https://arxiv.org/abs/2310.11441
- Umeyama algorithm: https://ieeexplore.ieee.org/document/88573
- Open3D: https://arxiv.org/abs/1801.09847

**Baselines / related scene reconstruction**:
- Gen3DSR: https://arxiv.org/abs/2404.03421
- ACDC (Digital Cousins): https://arxiv.org/abs/2410.07408
- Midi: https://arxiv.org/abs/2412.03558
- InstPIFu (single-view indoor): https://arxiv.org/abs/2207.08256
- 3D-Front dataset: https://arxiv.org/abs/2011.09192

**Physics-aware & related**:
- PhyRecon: https://arxiv.org/abs/2411.18777 (related work)
- Physcene: https://arxiv.org/abs/2405.02478
- Atlas3D: https://arxiv.org/abs/2405.18515
- PhysGaussian: https://arxiv.org/abs/2311.12127 (related dynamics)
- SayPlan: https://arxiv.org/abs/2312.14125 (LLM + 3D scene graph)
- Real-to-sim-to-real: https://arxiv.org/abs/2403.03949

**Other 3D generation baselines**:
- DreamGaussian: https://arxiv.org/abs/2309.16653
- LRM: https://arxiv.org/abs/2311.04400
- Wonder3D: https://arxiv.org/abs/2310.21208
- SyncDreamer: https://arxiv.org/abs/2309.03453
- CAT3D: https://arxiv.org/abs/2405.10314
- Flash3D: https://arxiv.org/abs/2406.04343
- Hyper3D OmniCraft (HDR env): https://hyper3d.ai/omnicraft/hdri

CAST 这个 pipeline 的精髓是把 single-image scene reconstruction 重新定义为一个 generation + alignment + constraint 的 modular problem，每个模块都用了 SOTA foundation model，并且巧妙地用 iterative refinement 把它们 coupled 起来。这种 design pattern 在 robotics、LLM agent、video generation 里都会反复出现——值得作为 mental model 内化。
