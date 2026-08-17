---
source_pdf: DGGT Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed
  Images.pdf
paper_sha256: 108f7ddb1ffce9b16a812e3f92a04413996a128fba0f902c334694fdb0e8104e
processed_at: '2026-08-03T21:05:22-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DGGT 人话版：用最直白的方式讲清楚这篇 paper

## 0. 一句话概括

DGGT 就是**给你几张没标过 camera pose 的开车视频帧，0.4 秒内吐出整个 4D 场景（3D Gaussian + 物体运动 + camera pose），还能让你随便编辑场景里的车和人**。

就这么简单。剩下全是细节。

- Project page: https://github.com/xiaomi-research/dggt

---

## 1. 这个问题为啥难：先讲 pain point

你想想 driving scene reconstruction 这个事，理想很丰满——拿到 sensor log，重建出 3D 场景，想渲染哪个视角就渲染哪个视角，想加车加车想删人删人，autonomous driving 团队就能随便生成 corner case 训练数据。

现实很骨感。现有方法几乎全是 per-scene optimization：**一个 scene 跑几十分钟到几小时**，一堆 hyperparameter 要调，camera pose 还得事先标定好。你以为 reconstruction 是 preprocessor，结果它成了整个 pipeline 的 bottleneck。

小米做 autonomous driving 的人肯定深有体会：你 fleet 一天采集几百 TB 数据，按 20 分钟一个 scene 重建，算到下辈子也算不完。所以这条 line 的核心 motivation 就一个字：**快**。要在秒级搞定。

再加一个字：**泛**。Waymo 标定跟 nuScenes 不一样，跟 Argoverse2 也不一样。你 model 在 Waymo 上 train 完，去 nuScenes 上能不能直接跑？现有方法几乎全崩。

DGGT 的目标：**0.4 秒 + 跨 dataset zero-shot**。这个目标定下来，剩下的技术选择都是围着这个转。

---

## 2. 现有方法为啥不行：三个 pain point

### Pain point 1：pose 当 input 的诅咒

几乎所有 reconstruction 方法都把 camera pose 当作 input。听起来合理——你得知道相机在哪才能重建嘛。

但这个设计有个隐藏的诅咒：**network 会偷偷把 pose 数值当 prior 学进去**。Waymo 的 camera rig 跟 nuScenes 不一样，extrinsics 数值范围不同，intrinsics 也不同。你 model 一看 "哦 pose 长这样就是 Waymo"，换 dataset 它就懵了。

DGGT 把 pose 从 input 挪到 output。**让 model 自己从 image 推 pose**。这相当于把 pose 当 latent variable，跟 scene geometry 联合估计。好处是 network 不再依赖 pose 的数值分布，只依赖 image evidence。

这个 reformulation 的回报：Waymo train 的 model 直接 zero-shot 去 nuScenes，能拿 25.31 PSNR。比 STORM 在 nuScenes 上 in-domain 训练的 24.54 还高。这个数字读出来我真的愣了一下，一般 zero-shot 比 in-domain 差 5-10 个点很正常，DGGT 居然反超，说明 pose-free formulation 的 domain gap 抑制能力非常强。

### Pain point 2：dynamic 怎么搞

Static scene 好办，3D Gaussian Splatting 这套已经成熟了。但 driving 是 dynamic 的——车在动、人在动、自行车在动。

naive 的做法是把每帧重建出来的 Gaussian 全部 union 起来当 4D scene。结果一辆车在 t1 在左边、t2 在右边，union 完渲染出来就是**双影鬼影**。

STORM 是这条 line 的前 SOTA，它的做法是给每个 dynamic Gaussian 预测一个 constant velocity，然后按时间积分位置。问题：**driving 里大部分运动都不是匀速直线**——加速、减速、转弯、pedestrian 突停。constant velocity 模型在长 sequence 上会严重 diverge，所以 STORM 只能用 short window（4 帧左右），再多就崩。

DGGT 的做法：**给每个 dynamic pixel 学一条完整的 3D 轨迹**，不是速度，是 trajectory。这样能拟合任意非线性运动。再用 TAPIP3D 预训练的 weight 做 init，免得从 driving 数据上从头学 3D tracking。

### Pain point 3：static 也会"变"

这个点容易忽略。Static background 听起来应该永远不变对吧？但实际上 driving 里 static 区域的 **appearance 是会变的**：
- 光照随时间渐变
- shadow 因为太阳角度变化在移动
- wet road 反光在变
- 远处的 tree 因为 ego-motion 产生 parallax-driven appearance shift

STORM 假设 static Gaussian 永远 opacity 不变，结果这些 subtle appearance change 全丢了，重建出来 static 区域 flicker。

DGGT 给每个 Gaussian 加一个 **lifespan 参数 $\sigma$**，本质是让 Gaussian 在时间维上也是个高斯分布——在自己 anchor 时间点 opacity 最强，远离 anchor 时间点 opacity 衰减。appearance 真不变的就让 $\sigma \to \infty$，appearance 变的就让 $\sigma$ 小一点，让其他 frame 的 Gaussian 接管。

这个 design 非常优雅，相当于用一根连续的标量参数化了一个本来是离散的 "static vs transient" 区分。

---

## 3. DGGT 的关键 trick 逐个人话讲

### Trick 1：DINO + Alternating Attention 的 backbone

backbone 是 ViT，但 feature extractor 用 **DINO 预训练权重 frozen**。DINO 在海量 image 上自监督训练，feature 自带 cross-view correspondence 能力。对 pose-free 设定特别关键，因为 pose 估计本质就是找 2D-2D correspondence。

attention 是 **alternating**：先在 view 内做 self-attention（处理空间信息），再在 frame 间做 cross-attention（处理时间 correspondence）。这比 flat 的 global attention 高效得多，也是 DGGT 能 scale 到 16 个 view 不崩的原因。

一个细节：作者发现 $F_{attn}$ 经过多层 attention 后 spatial detail 流失严重，对 per-pixel Gaussian 预测是致命的。所以又把原始 $F_{dino}$ skip 融合回来给 Gaussian head 用。**$F_{dino}$ 负责 where，$F_{attn}$ 负责 what**，跟 U-Net skip connection 一个思路。

### Trick 2：Per-Pixel Gaussian Map

每帧生成一个 $H \times W \times 15$ 的 tensor，每个 pixel 对应一个 Gaussian，参数包括：
- RGB color (3)
- 3D mean position $\mu$ (3)
- rotation quaternion (4)
- scale (3)
- opacity (1)
- lifespan $\sigma$ (1)

合计 15 维。

好处是 resolution 跟 image 对齐，监督清晰，instance editing 直接 mask 掉某些 pixel 就行。坏处是 sky 这种 unbounded 区域不好搞，所以单独有 Sky head。

### Trick 3：Lifespan 公式（Eq.1）

$$
o^{t'} = o^t \cdot \exp\left(-\frac{1}{2} \cdot \frac{(t'-t)^2}{\sigma^t}\right)
$$

人话翻译：**Gaussian 的"可见度"在时间维上是个高斯钟形曲线，以它 anchor 的 timestamp $t$ 为中心，宽度由 $\sigma$ 决定**。

- $\sigma$ 大 → Gaussian 在很多帧都还可见 → 对应 static appearance 稳定的区域
- $\sigma$ 小 → Gaussian 很快淡出 → 对应 appearance 在变化的区域，下一帧换别的 Gaussian 接管

正则项 $\mathcal{L}_{lifespan} = \|1/\sigma\|_1$ 鼓励 $\sigma$ 大（prior 是大部分场景 static），渲染 loss 反过来 push $\sigma$ 小（当 appearance 真变的时候）。两者拔河达到平衡。

### Trick 4：Dynamic/Static Decomposition（Eq.3）

$$
\hat{G}^t = \left(\bigcup_{t'=1}^{N} G_s^{t'}\right) \cup G_d^t \cup G_{sky}
$$

人话：**当前帧的完整场景 = 所有帧 static 部分 union + 当前帧 dynamic 部分 + sky**。

为什么 static 要 union 所有帧？因为 single view coverage 太 sparse，多帧 union 才能把 background 拼完整。

为什么 dynamic 只取当前帧？因为 union 起来就会鬼影——一辆车在 t1 和 t2 位置不一样，两个位置的 Gaussian 都保留就 double image 了。

### Trick 5：Motion Head 用 TAPIP3D 预训练

Motion head 是个 Transformer，输入是 image feature + 从 dynamic Gaussian 反投影出来的 3D point。它在 $t_a$ 和 $t_b$ 两个 timestamp 之间预测每个 query point 的 3D displacement。

**关键 design**：用 TAPIP3D [https://arxiv.org/abs/2504.14717] 的预训练权重 init。TAPIP3D 在大规模 4D point tracking 数据上训练过，已经具备 generic 3D motion correspondence 能力。DGGT 把它 specialize 到 driving domain 上 finetune。

这避免了在 driving 数据上从 scratch 学 3D motion 的数据饥渴。driving 数据虽然多，但 3D motion annotation 几乎没有，photometric supervision 又弱。pretrained init 相当于把别处学好的"物体怎么动"的 prior 拿过来用。

### Trick 6：Motion Interpolation（Eq.6）

中间 timestamp $t_i$ 的 dynamic Gaussian mean：

$$
\mu_d^{t_i} = \mu_d^{t_a} + \omega^{t_i} \cdot F(t_a, t_b), \quad \omega^{t_i} = \frac{t_i - t_a}{t_b - t_a}
$$

人话：**在 $t_a$ 和 $t_b$ 之间做线性插值，权重是时间比例**。

注意只有 $\mu$ 被插值，rotation / scale / opacity / color 都 inherit 自 $t_a$。这是个 simplification——假设动态物体短时间窗口内 shape 不变。对 rigid body（车）成立，对 deformable（行人肢体）会有 artifact，是个已知 limitation。

camera pose 也插值：translation linear，rotation 用 SLERP（spherical linear interpolation on quaternions），避免 linear 插值破坏 quaternion 的 unit-norm 约束。

### Trick 7：Sky Head

Per-pixel Gaussian 建模 sky 不友好——sky 在 image space 占很多 pixel，但 geometry 上对应一个球面，per-pixel 表达冗余且不准。

DGGT 在 fixed-radius hemisphere 上均匀采样一组 sky Gaussian，固定 rotation 和 opacity（让它们永远朝内、永远全透），只通过 MLP 微调 color 和 scale。radius $r_{sky}$ fixed 是为了把 sky 推到 effectively at infinity，避免和 scene geometry 耦合。

### Trick 8：Diffusion Refinement

3DGS 在 sparse view + 大 rotation/translation 下会产生 ghosting、disocclusion gap、texture blur。这些是 representation 的 intrinsic limitation——primitives 之间没有 prior 来填补 unseen area。

DGGT 加一个 **single-step diffusion** 后处理。把渲染图 $\hat{I}^{t_i}$ 和从 input sequence 随机采样的 reference image $I_{ref}$ 拼起来，encode 进 VAE latent，UNet 一步去噪，decode 出 refined 图。

人话：**3DGS 渲染出来的图有 artifact，diffusion 来 polish 一下。reference image 告诉 diffusion "场景长这样"，diffusion 只做 inpainting-like refinement，不 hallucinate 新物体**。

用 single-step 而不是 multi-step 是因为 inference speed 第一。Adversarial Diffusion Distillation [https://arxiv.org/abs/2311.17042] 这套技术让 UNet 学会"一步从 corrupted latent 跳到 clean latent"，inference 时间几乎可以忽略。

---

## 4. Loss 全貌人话版

训练时一次采 $N \in [4, 8]$ input frames + $2N$ target frames。input sparse、target dense，强制 model 学 interpolation。target frame 之间的 intermediate timestamp 就是 motion head 和 diffusion refinement 的 supervision 来源。

总 loss：

$$
\mathcal{L}_{feedforward} = \mathcal{L}_{rgb} + \lambda_{opacity}\mathcal{L}_{opacity} + \lambda_{dynamic}\mathcal{L}_{dynamic} + \lambda_{lifespan}\mathcal{L}_{lifespan}
$$

- $\mathcal{L}_{rgb}$：渲染图 vs GT 的 $\ell_2$ + LPIPS，主要 supervision
- $\mathcal{L}_{opacity}$：sky mask 的 BCE loss
- $\mathcal{L}_{dynamic}$：dynamic mask 的 BCE loss，dynamic mask 来自 LiDAR bbox + SAM2 [https://arxiv.org/abs/2408.00714] propagation
- $\mathcal{L}_{lifespan} = \|1/\sigma\|_1$：lifespan sparsity prior

Diffusion 单独训练——先把 feedforward model 训好，再用它的 output（带 artifact 的 interpolation）作为 input，GT 作为 target，训练 diffusion 学 artifact→clean 的映射。两阶段而不是 joint，原因可能是 diffusion 需要稳定的 artifact pattern，joint 早期 feedforward output 不稳定会让 diffusion 学歪。

---

## 5. 实验结果讲个故事

### Waymo 主表的故事

DGGT 拿 27.41 PSNR / 0.39s inference，同时 ✓ Dynamic + ✓ Pose-free。表里所有 method 只有 DGGT 一个同时打两个 ✓。

更关键的是 **D-RMSE 3.47 全表最低**。PSNR 高可能只是 appearance overfit，但 depth error 低意味着 geometry 真的准。STORM D-RMSE 5.48，比 DGGT 高一半多，说明 STORM 的 geometry 其实不太行，靠 appearance 弥补。

Inference time 0.39s vs STORM 0.18s。DGGT 慢一倍多，但比 per-scene optimization 快 4 个数量级。trade-off 完全 OK。

NoPoSplat 23.22s 的原因：它虽然 pose-free 但 inference 时还要 iterative refine pose，慢得离谱。DGGT 完全 single-pass，model 再重也快。

VGGT++ 22.50 PSNR——VGGT 是 DGGT 的 backbone 来源，但加上 Gaussian head 直接做 rendering 表现很差。作者归因 attention feature 保留 RGB detail 不足。这正好反推 DGGT 用 $F_{dino}$ skip fusion 的必要性。

### Cross-dataset 的故事

**DGGT zero-shot on nuScenes 25.31 > STORM in-domain trained 24.54**。

这一行结果太关键了。zero-shot 一般比 in-domain 差 5-10 个点很正常，DGGT 反超。这强烈说明 pose-free formulation 的 generalization 优势——把 pose 当 input 真的会让 model 学到 dataset-specific prior，换 dataset 就崩。

Argoverse2 也是一样：DGGT zero-shot 26.34，STORM in-domain 24.97。

### 3D Motion 的故事

EPE3D 0.183m vs STORM 0.276m，下降 33%。Angular error 0.328 vs STORM 0.658，下降 50%。

angular error 砍半特别重要——motion vector 不只是 magnitude 对了，direction 也对了。对 downstream tracking 这是关键。

### View scalability 的故事

STORM 在 view 数 4 → 8 → 16 时 PSNR 严重退化（26.55 → 25.11 → 23.69）。DGGT 几乎稳定（30.54 → 31.41 → 30.66）。

root cause：STORM 把所有 view 同时塞进 fixed-size transformer，token 数 linear 增长，attention 的 quadratic 复杂度爆炸。DGGT 用 alternating attention 把 view 内和 view 间解耦，复杂度接近 linear。

---

## 6. Scene Editing 这个 application

DGGT 的 representation 因为是 object-level decomposed Gaussian，scene editing 几乎是 free lunch：

- **删车**：把 dynamic Gaussian 里 car 对应的 pixel mask 掉，rerender
- **移车**：给 dynamic Gaussian 的 $\mu$ 加 offset，rerender
- **加新车**：从另一个 scene 重建一辆车的 dynamic Gaussian，union 进当前 scene，rerender

Gaussian manipulation 会留下 holes / artifacts，diffusion refinement 来 fill in。

这个 workflow 的工业价值：autonomous driving 的 corner case generation、data augmentation、what-if simulation 都需要这种 editing 能力。STORM 没有 explicit object decomposition，做不了这个。

---

## 7. 与相关工作的关系

### DGGT vs STORM

STORM 是这条 line 的前 SOTA。差异：

1. STORM 需 pose，DGGT 不需要——formulation 级差异
2. STORM 用 constant velocity，DGGT 用完整 3D trajectory——表达能力差异
3. STORM 没 lifespan，static appearance 变化时崩——temporal modeling 差异
4. STORM 没 diffusion refinement——artifact 处理差异
5. STORM 长序列 diverge，DGGT 长序列稳定——scalability 差异

可以说 DGGT 是 STORM 的全面升级版。

### DGGT vs VGGT

VGGT [https://arxiv.org/abs/2503.11651] 是 DGGT 的 backbone 来源，做 static pose-free multi-view reconstruction。DGGT 在它基础上加 dynamic head / motion head / lifespan head / sky head / diffusion refinement。

DGGT = VGGT + dynamic extension + diffusion post-processing，可以这样理解。

### DGGT vs NoPoSplat

NoPoSplat [https://arxiv.org/abs/2410.24207] 也是 pose-free feedforward Gaussian splatting，但只做 static。inference 时需要 iterative refine pose，所以慢（23.22s）。DGGT 完全 single-pass，更快。

### DGGT vs 4D-GS

4D-GS [https://arxiv.org/abs/2311.11235] 是 per-scene optimization 的 4D Gaussian Splatting。DGGT 把 4D-GS 的核心 idea（time-conditioned Gaussian）转成 feedforward 形式，并用 lifespan 参数化代替 explicit time embedding，更 compact 且 disentangled。

---

## 8. 我看到的 limitation 和未来方向

Paper 自己提到：
1. Dynamic mask 不准时崩
2. Heavily occluded motion 时 tracking 失败

我额外想到的：

1. **Motion interpolation 假设 shape 不变**：对行人肢体、自行车手会 artifact。可以加 shape interpolation 或 per-Gaussian deformation field。

2. **Lifespan 是标量**：可以扩展成 anisotropic temporal Gaussian，让不同时间方向有不同 decay，比如一个 Gaussian 在过去很快淡出但未来还能持续。

3. **Sky hemisphere fixed radius**：对 aerial view 或大幅 pitch 变化不够 flexible。可以做成 learnable radius 或 cubemap。

4. **Diffusion single-step**：hard artifact 可能 miss。可以做成 adaptive step，根据 artifact 严重程度动态决定步数。

5. **没 explicit material / lighting decomposition**：对 relighting 不够，未来可以加 BRDF head。

6. **Motion head 没利用 vehicle rigid body prior**：可以用 bbox annotation 监督 vehicle pose，比 generic 3D tracking 更准。

7. **没处理 reflective surface**（玻璃、湿路面）：driving 这块很多，DGGT 没特别处理，会被 lifespan + opacity 简单 modulate 掉。

8. **Long sequence 实际 transformer attention 还是 O(N²)**：可以用 sparse attention 或 chunked processing 进一步优化。

---

## 9. 延伸联想

### 与 World Model 的连接

最近 driving world model 很热（GAIA-2 [https://arxiv.org/abs/2506.01425]、DriveDreamer [https://arxiv.org/abs/2310.09777]、Copilot4D [https://arxiv.org/abs/2311.01017]）。DGGT 这种 feedforward 4D Gaussian reconstruction 可以作为 world model 的 **deterministic initialization**——先用 DGGT reconstruct 当前帧 4D state，再让 generative model 在此基础上 rollout。比 pure video diffusion world model 更 controllable。

### 与 Robotics SLAM 的连接

Robotics SLAM 也在用 Gaussian Splatting 做 dense reconstruction（SplaTAM [https://arxiv.org/abs/2403.02751]、MonoGS [https://arxiv.org/abs/2404.16742]）。DGGT 的 pose-free formulation 对 SLAM 也有启发——SLAM 核心难点就是 pose estimation + dense reconstruction 联合优化，DGGT 把它们 feedforward 出来，可以当 SLAM 的 dense prior 或 initialization。

### 与 L4GM 的对比

L4GM [https://arxiv.org/abs/2410.21274] 是第一个 4D large reconstruction model，主要做 object-level。DGGT 是 scene-level + driving domain。思路相似（feedforward 4D），但 DGGT 处理 unbounded scene + dynamic decomposition，技术难度高一个量级。

### 与 DUSt3R / MASt3R / VGGT 的 lineage

DUSt3R [https://arxiv.org/abs/2312.14132] → MASt3R [https://arxiv.org/abs/2406.09681] → VGGT [https://arxiv.org/abs/2503.11651] 这条 line 是 pose-free 3D 的 backbone evolution。DGGT 选 VGGT 作起点，因为 VGGT 已经把 alternating attention + DINO feature + multi-head prediction 这套打通了。DGGT 的贡献是在 VGGT 之上加 dynamic-aware heads。

### 工业应用直觉

小米 EV 内部大概率已经在 production pipeline 里用 DGGT 或者它的变体。driving data generation pipeline 一般是：sensor log → reconstruction → editing → simulation → new sensor log。DGGT 直接 fit 在第二步和第三步——fast reconstruction 让 pipeline 真能 scale，editing 能力让 what-if 分析可行。

---

## 10. TL;DR

如果让我用三句话给 Andrej 讲清楚 DGGT：

1. **把 camera pose 从 input 挪到 output**，让 model 从 image 自己推 pose，跨 dataset generalization 直接起飞（zero-shot 反超 in-domain STORM）。

2. **用 per-pixel Gaussian + lifespan 参数 + 3D motion trajectory 三件套搞定 dynamic**：lifespan 让 static appearance 能渐变，3D motion trajectory 让 dynamic object 能非线性运动，加起来支持长序列和任意 view 数。

3. **Diffusion 做 single-step 后处理 polish artifact**，3DGS 渲染的 ghosting / disocclusion / blur 由 diffusion 用 natural image prior 来修，速度还很快。

这就是 DGGT。一个让 driving 4D reconstruction 真正能当 preprocessor 用的 work。

---

**References**
- DGGT: https://github.com/xiaomi-research/dggt
- STORM: https://arxiv.org/abs/2501.00602
- VGGT: https://arxiv.org/abs/2503.11651
- NoPoSplat: https://arxiv.org/abs/2410.24207
- MVSplat: https://arxiv.org/abs/2403.11351
- DepthSplat: https://arxiv.org/abs/2410.13862
- EmerNeRF: https://arxiv.org/abs/2311.02077
- L4GM: https://arxiv.org/abs/2410.21274
- TAPIP3D: https://arxiv.org/abs/2504.14717
- ADD: https://arxiv.org/abs/2311.17042
- DINOv2: https://arxiv.org/abs/2304.07193
- SAM2: https://arxiv.org/abs/2408.00714
- DUSt3R: https://arxiv.org/abs/2312.14132
- MASt3R: https://arxiv.org/abs/2406.09681
- 3DGS: https://arxiv.org/abs/2308.04079
- 4D-GS: https://arxiv.org/abs/2311.11235
- Copilot4D: https://arxiv.org/abs/2311.01017
- SplaTAM: https://arxiv.org/abs/2403.02751
- MonoGS: https://arxiv.org/abs/2404.16742
- DriveDreamer: https://arxiv.org/abs/2310.09777
- Waymo Open Dataset: https://arxiv.org/abs/1912.07212
- nuScenes: https://arxiv.org/abs/1903.11006
- Argoverse2: https://arxiv.org/abs/2301.00493

---

# DGGT: Feedforward 4D Reconstruction of Dynamic Driving Scenes using Unposed Images 深度讲解

## 1. 核心动机与定位

这篇 paper 的核心 take-away 在于把 **driving scene 4D reconstruction** 从一个 *per-scene optimization* 任务彻底转成一个 *single-pass feedforward inference* 任务，并且把这个 formulation 里的 **camera pose 从 input 端搬到 output 端**。

为什么这个搬动很关键？我个人的 intuition 是这样的：driving dataset 之间的 domain gap 很大一部分来自 **camera intrinsics / extrinsics 的标定差异**。Waymo 的 camera rig 跟 nuScenes 的 camera rig 完全不一样，pose 的数值分布、坐标系定义、相机间距都不一样。一旦你把 pose 当 input 喂进 network，network 就很容易把 pose 的数值分布当一种 prior 学进去，结果在跨 dataset 时崩掉。把 pose 当 output，让 model 自己从 image evidence 推出来，相当于把 pose 当一个 *latent variable*，让 model 把它和 scene geometry 绑在一起联合估计，从而减弱对 dataset-specific camera configuration 的依赖。

所以 DGGT 的 cross-dataset generalization 表现（Waymo train → nuScenes zero-shot 25.31 PSNR，Argoverse2 zero-shot 26.34 PSNR）来自这个 formulation，from architecture 的次之。这一点在 Sec.4.2 末尾作者自己也 explicit remark 了。

- Paper: https://arxiv.org/abs/2506.05603 (DGGT)
- Code: https://github.com/xiaomi-research/dggt
- 竞品 STORM: https://arxiv.org/abs/2501.00602
- VGGT (backbone 沿用): https://arxiv.org/abs/2503.11651

## 2. 整体架构解析

### 2.1 Pipeline 一图流

```
{I^t}_{t=1..N}  (unposed images, dynamic scene)
        │
        ▼
   [ViT patch tokenizer]
        │
        ▼
   [DINO frozen feature extractor] → F_dino
        │
        ▼
   [Alternating-attention layers]  → F_attn  (with F_dino skip-fusion)
        │
        ├──→ H_cam     → Π^t            (camera extrinsics + intrinsics per frame)
        ├──→ H_gs      → G^t ∈ R^{H×W×15}  (per-pixel Gaussian map)
        ├──→ H_life    → σ^t             (lifespan per Gaussian)
        ├──→ H_dy      → M_d^t           (dynamic mask)
        └──→ H_sky     → G_sky           (sky Gaussians on hemisphere)

        │  (after Gaussian maps available)
        ▼
   [H_motion (Transformer, pretrained by TAPIP3D)]
        │   queries Q = pixels where M_d = 1
        │   F(t_a,t_b) = per-query 3D displacement
        ▼
   [Static / Dynamic decomposition (Eq.2)]
   Ĝ^t = ∪_{t'} G_s^{t'} ∪ G_d^t ∪ G_sky      (Eq.3)
        │
        ▼
   [Differentiable 3DGS Renderer]  → Î^t
        │
        ▼
   [Single-step diffusion refinement f_diffusion(Î^t, I_ref)]
        │
        ▼
   refined Î̃^t
```

### 2.2 为什么用 DINO + alternating attention 而不是从 scratch 训 ViT

DINO v2 [https://arxiv.org/abs/2304.07193] 提供的 feature 本身就编码了 *geometric correspondences* 和 *semantic saliency*，这对 *pose-free* 设定特别重要——因为 camera pose 的预测本质上是 *2D-2D correspondences* + *2D-3D lifting*，需要 feature 有 cross-view 一致性。作者 freeze DINO feature extractor 和 camera head，只 finetune 后面的 head。这是一个很典型的 *foundation model transfer* 策略，省显存、省时间、还能利用 DINO 在海量数据上学到的 cross-image correspondence 结构。

**Alternating attention** 来自 VGGT 的设计：在 *view-wise self-attention*（同 view 内 token 之间交互）和 *frame-wise cross-attention*（跨 view / 跨 time token 交互）之间交替进行。这种 design 比 flat 的 global attention 更 sample efficient，并且隐式 encode 了 *spatial-temporal 的 factorization*——空间维度信息（appearance / depth）主要在 view-wise 里精修，时间维度信息（motion / correspondence）主要在 frame-wise 里传播。

### 2.3 特征融合：F_dino + F_attn 的 skip fusion

作者在 paper 里特别强调了一个 *failure mode*：**`F_attn` 经过多层 attention 后 semantic 信息富集但 spatial detail 流失**，这对 pixel-aligned Gaussian map 是致命的，因为 Gaussian 的位置 $\mu$ 是 per-pixel 预测的，对 spatial fidelity 要求极高。

所以 Gaussian head 的输入不是 $F_{attn}$ 单独，而是 $F_{attn} \oplus F_{dino}$ 的某种融合（在 Fig.2 里画成 arrow）。Intuition：$F_{dino}$ 提供 *where*（spatial），$F_{attn}$ 提供 *what*（语义/几何聚合后的 high-level 信息）。这和 U-Net 的 skip connection 思想完全一致，只是放在了 token feature 层面。

## 3. Scene Representation：Per-Pixel Gaussian + Lifespan

### 3.1 每个像素一个 Gaussian 的优缺点

**Pro**：resolution 直接对应 pixel grid，预测监督清晰，no point cloud 抽样偏差，容易做 instance editing（直接 mask 掉某些 pixel 即可）。

**Con**：representation redundancy 高（一个真实 surface point 可能由多个 pixel 重复描述），unbounded region（sky）建模不友好（因为 sky 在 image space 占的 pixel 多，但 geometry 上对应球面，难以用 per-pixel 准确编码）。

DGGT 处理 Con 的方式是单独引入 **Sky head**，在 fixed-radius hemisphere 上均匀采样点作为 sky Gaussians，固定其 rotation 和 opacity，只通过 MLP $\mathcal{H}_{sky}$ 微调 color 和 scale。radius $r_{sky}$ 选 fixed 是为了把 sky 推到 *effectively at infinity*，避免和场景几何耦合。

### 3.2 Lifespan 参数 $\sigma^t$：解决 STORM 的退化

STORM 把所有 static Gaussian 简单累加起来，认为 static = 永远可见。但 driving 里 static 区域的 *appearance 是随时间变化的*——比如光照渐变、shadow 漂移、wet road 的反光变化、distant tree 因为 ego-motion 产生的 parallax-driven appearance shift。这些都不是真正"dynamic object"，但用单一 opacity 描述是不够的。

DGGT 给每个 Gaussian 加一个标量 lifespan $\sigma^t \in \mathbb{R}^+$，作用是 modulate opacity 跨时间的衰减：

$$
o^{t'} = o^t \cdot \exp\left(-\frac{1}{2} \cdot \frac{(t'-t)^2}{\sigma^t}\right)
$$

变量含义：
- $o^t$：Gaussian 在其 anchor timestamp $t$ 处的原始 opacity（由 $\mathcal{H}_{gs}$ 预测）
- $o^{t'}$：在 timestamp $t'$ 处经过 lifespan modulation 后的 effective opacity
- $\sigma^t$：lifespan 参数，控制 Gaussian 在时间维上的"宽度"
- $(t'-t)^2$：时间偏移的平方

注意这是一个 *time-Gaussian* 形式，即把 temporal influence 建模成以 $t$ 为中心、宽度 $\sigma^t$ 的高斯。$\sigma^t \to \infty$ 时退化成 STORM 的恒定 opacity；$\sigma^t \to 0$ 时退化成只在 $t$ 帧可见的瞬时 Gaussian。

这个 design 的优雅之处在于：**用一个连续标量参数化离散的 "static vs transient" 区分**。传统方法要么 hard binary 分（导致边界 artifact），要么纯静态（导致外观漂移失败）。Lifespan 是 soft 的中间态。

Lifespan 的 supervision：作者用 $\ell_1$ regularization $\mathcal{L}_{lifespan} = \|\frac{1}{\sigma}\|_1$，倾向于 push $\sigma$ 往大走（鼓励 long lifespan），因为 prior 是 "most of the scene is static"。但渲染 loss 会反过来 push $\sigma$ 往小走，当 appearance 确实变化时。这两者形成 equilibrium。

## 4. Dynamic Decomposition 与 3D Motion Estimation

### 4.1 Decomposition 公式

每个 frame 的完整 Gaussian 集合 $\hat{G}^t$（Eq.3）由三部分 union 组成：

$$
\hat{G}^t = \left(\bigcup_{t'=1}^{N} G_s^{t'}\right) \cup G_d^t \cup G_{sky}
$$

其中：
- $\bigcup_{t'} G_s^{t'}$：所有 frame 的 static Gaussian 的并集（时间累积的"background point cloud"）
- $G_d^t = G^t \odot M_d^t$：当前 frame t 的 dynamic Gaussian，由 dynamic mask $M_d^t$ 提取
- $G_{sky}$：固定的 sky Gaussian set

为什么 static Gaussian 要 *union across all frames* 而不是只取当前 frame？因为 single-view coverage 是 sparse 的，ego-motion 又在不同 frame 从不同视角看到 background 的不同部分。Union 起来才能在 current view 形成相对完整的 background coverage。

为什么 dynamic Gaussian 只取当前 frame？因为 union 起来会形成 ghosting——一辆车在 $t_1$ 和 $t_2$ 位置不同，两个位置的 Gaussian 都被并集保留，渲染出来就是双影。

### 4.2 Motion Head：超越 STORM 的速度场

STORM 只预测 Gaussian 的 *constant velocity*（一阶线性运动模型），在 driving 这种有加速 / 转弯 / pedestrian 突停场景里会严重 diverge。

DGGT 改成 **per-query 3D displacement trajectory**，并且 query 是从 dynamic mask 区域 backproject 出来的 3D point。这本质上是 *point tracking* 而不是 *flow estimation*。

Motion head 公式（Eq.5）：

$$
F(t_a, t_b) = \mathcal{H}_{motion}\left(\mathcal{Q} \mid G^{t_a}, G^{t_b}, I^{t_a}, I^{t_b}\right)
$$

变量：
- $\mathcal{Q} \in \mathbb{R}^{q \times 2}$：query 像素坐标集合，$q$ 是 query 数量
- $G^{t_a}, G^{t_b}$：两个 frame 的 Gaussian map
- $I^{t_a}, I^{t_b}$：两个 frame 的 RGB 图像
- $F(t_a, t_b) \in \mathbb{R}^{q \times 3}$：每个 query 的 3D displacement 向量

**Architecture 细节**（paper 没全展开但可以推断 + 从 TAPIP3D [https://arxiv.org/abs/2504.14717] 继承）：motion head 是一个 Transformer，输入有两条流：
1. **Image stream**：多尺度 image features（来自 image encoder，跟 backbone 共享或独立）
2. **Point cloud stream**：从 $G^{t_a}$ 提取的 3D point + 其 feature（color, scale, opacity）

每个 query point 在 $t_a$ 处 init 为对应 3D 位置，通过 **neighborhood-to-neighborhood attention** iteratively refine 它在 $t_b$ 处的位置。这个 "neighborhood" 既包括 spatial 邻近的 point，也包括 temporal 对应的 point。最后 query 在 $t_b$ 处的位置减去 $t_a$ 处的位置就是 $F(t_a, t_b)$。

**Pretraining 策略**：motion head 用 TAPIP3D 预训练权重 init，然后用 photometric loss 在 interpolated frames 上 finetune。这是一个非常聪明的 design——TAPIP3D 本身在大规模 4D point tracking 数据上训练过，已经具备 *generic 3D motion correspondence* 能力，DGGT 只是把它 specialize 到 driving domain。这避免了在 driving 数据上从头学 3D motion 的数据饥渴问题。

### 4.3 Motion Interpolation 公式（Eq.6）

给定相邻 timestamp $t_a, t_b$ 和中间 timestamp $t_i \in [t_a, t_b]$，dynamic Gaussian 的 mean 坐标线性插值：

$$
\mu_d^{t_i} = \mu_d^{t_a} + \omega^{t_i} \cdot F(t_a, t_b), \quad \omega^{t_i} = \frac{t_i - t_a}{t_b - t_a}
$$

变量：
- $\mu_d^{t_a}$：dynamic Gaussian 在 $t_a$ 处的 3D mean
- $F(t_a, t_b)$：motion head 预测的 3D displacement
- $\omega^{t_i} \in [0, 1]$：线性插值权重

注意这里 *只有 mean 被 interpolate*，rotation / scale / opacity / color 是 inherit 自 $t_a$ 处的值。这是一个 simplification——假设动态物体在短时间窗口内 shape / color 不变，只有 position 在变。对 driving 大部分场景成立，但对 deformable object（如行人肢体）会有偏差。

Camera pose 的插值：translation 用 linear，rotation 用 **SLERP on quaternions**，这是为了避免 linear quaternion 插值破坏 unit-norm 约束导致 rotation 矩阵奇异。

## 5. Diffusion Refinement：把 3DGS 渲染结果当作 condition

### 5.1 Motivation

3DGS 在 sparse view + 大 rotation / translation 时会产生三类 artifact：
1. **Ghosting**：motion 估计不准时 double-image
2. **Disocclusion gaps**：intermediate frame 中出现的区域没有任何 Gaussian 覆盖
3. **Texture blur**：Gaussian 在 appearance 不连续处会 splat 出模糊带

这些 artifact 都是 3DGS representation 的 *intrinsic limitation*——primitives 之间没有 learnable prior 来填补 unseen area。Diffusion model 训练在大规模 image 上，天然具备 *natural image manifold* 的 prior，可以作为后处理 inject 进来。

### 5.2 单步 diffusion 设计

DGGT 用 **Adversarial Diffusion Distillation (ADD)** [https://arxiv.org/abs/2311.17042] 风格的 single-step diffusion。具体网络：

- **Frozen VAE encoder**：把 rendered image $\hat{I}^{t_i}$ 和 reference image $I_{ref}$（从 input sequence 随机采样）拼起来 encode 成 latent
- **UNet denoiser**：在 latent space 做一步去噪
- **LoRA fine-tuned decoder**：decode 成 refined RGB $\tilde{I}^{t_i}$

公式（Eq.11）：

$$
\tilde{I}^{t_i} = f_{diffusion}(\hat{I}^{t_i}, I_{ref})
$$

注意没有显式 noise schedule，没有多步 sampling。这是 ADD 的核心 trick：用 adversarial loss + distillation 让 UNet 学会"一步从 corrupted latent 跳到 clean latent"。对 driving 应用来说，**inference speed 比 fidelity 极限更重要**，0.39s 的 total inference time 里 diffusion 部分必须只占很小比例。

为什么 ref image 重要？因为 diffusion 需要一个 *identity anchor* 来避免 hallucinate 出场景里不存在的物体。Ref image 从 input sequence 随机采样，告诉 diffusion "scene 长这样"，diffusion 只需要做 *inpainting-like refinement*，而 NOT generate from scratch。

### 5.3 Diffusion Loss

$$
\mathcal{L}_{diffusion} = \mathcal{L}_{Recon} + \mathcal{L}_{LPIPS} + \lambda_{Gram} \mathcal{L}_{Gram}
$$

- $\mathcal{L}_{Recon}$：$\ell_2$ between $\tilde{I}^{t_i}$ 和 ground truth $I^{t_i}$
- $\mathcal{L}_{LPIPS}$：perceptual loss
- $\mathcal{L}_{Gram}$：VGG-16 features 的 Gram matrix loss，专门 push sharpness 和 fine detail

Gram loss 的 intuition 来自 neural style transfer [https://arxiv.org/abs/1508.06576]：Gram matrix 编码 feature map 的 *channel-wise correlation*，对应 texture 信息。优化 Gram matrix 距离就能 transfer texture without affecting content structure。

数据：~2000 clips from 798 Waymo training scenes。input = interpolated frames with artifacts，output = GT frames。这是一种 *paired image-to-image translation* 的 setup，diffusion 学的是 artifact → clean 的映射。

## 6. Training Objective 全貌

完整 training loss（Eq.10）：

$$
\mathcal{L}_{feedforward} = \mathcal{L}_{rgb} + \lambda_{opacity}\mathcal{L}_{opacity} + \lambda_{dynamic}\mathcal{L}_{dynamic} + \lambda_{lifespan}\mathcal{L}_{lifespan}
$$

其中：

$$
\mathcal{L}_{rgb} = \mathcal{L}_{\ell_2} + \lambda_{LPIPS}\mathcal{L}_{LPIPS}
$$

- $\mathcal{L}_{\ell_2}$：渲染图和 GT 的 pixel-wise MSE
- $\mathcal{L}_{LPIPS}$：perceptual loss
- $\mathcal{L}_{opacity} = \text{BCE}(M_{sky}, \hat{M}_{sky})$：sky mask 监督
- $\mathcal{L}_{dynamic} = \text{BCE}(M_d, \hat{M}_{dynamic})$：dynamic mask 监督（来自 LiDAR bbox + SAM2 propagation）
- $\mathcal{L}_{lifespan} = \|1/\sigma\|_1$：lifespan sparsity prior

**重要细节**：training 采 $N \in [4, 8]$ input frames + $2N$ GT target frames。Input frames sparse，target frames dense，强制 model 学会 *interpolation*。这就是 motion head 和 diffusion refinement 的 supervision 来源——target frame 之间的 intermediate timestamp 需要被插值出来，再和 GT 比。

Diffusion 单独训练（end-to-end feedforward model 先训好，再用其 output 训 diffusion）。两阶段训练而不是 joint training 的原因我推测是：(1) diffusion 需要稳定的 artifact pattern，而早期 feedforward model output 不稳定；(2) joint training 显存吃不消。

## 7. 实验结果深度解读

### 7.1 Waymo 主表（Tab.1）

| Method | PSNR ↑ | SSIM ↑ | D-RMSE ↓ | Inference time | Dynamic | Pose-free |
|---|---|---|---|---|---|---|
| EmerNeRF | 24.51 | 0.738 | 33.99 | 14 min | ✓ | ✗ |
| 3DGS | 25.13 | 0.741 | 19.68 | 23 min | ✗ | ✗ |
| PVG | 22.38 | 0.661 | 13.01 | 27 min | ✓ | ✗ |
| DeformableGS | 25.29 | 0.761 | 14.79 | 29 min | ✓ | ✗ |
| LGM | 18.53 | 0.447 | 9.07 | 0.06 s | ✗ | ✗ |
| GS-LRM | 25.18 | 0.753 | 7.94 | 0.02 s | ✗ | ✗ |
| MVSplat | 20.56 | 0.697 | 10.13 | 0.08 s | ✗ | ✗ |
| NoPoSplat | 24.31 | 0.751 | 9.08 | 23.22 s | ✗ | ✓ |
| DepthSplat | 23.26 | 0.696 | 10.05 | 0.11 s | ✗ | ✗ |
| STORM | 26.38 | 0.794 | 5.48 | 0.18 s | ✓ | ✗ |
| VGGT++ | 22.50 | 0.749 | 3.80 | 0.24 s | ✗ | ✓ |
| **DGGT (Ours)** | **27.41** | **0.846** | **3.47** | 0.39 s | ✓ | ✓ |

关键观察：
- **DGGT 是唯一同时 ✓ Dynamic + ✓ Pose-free 的方法**，且 PSNR 在 27+ 这个一档。
- **D-RMSE 3.47 是全表最低**——说明 DGGT 不仅 appearance 好还 geometry 好。这一点很关键，因为 PSNR 高可能只是 appearance overfit，但 depth error 低意味着 geometry 真的准。
- **Inference 0.39 s**：比 STORM 慢一倍多（0.18s），但比 per-scene optimization 快 4 个数量级。这个 trade-off 完全 acceptable。
- **NoPoSplat 23.22s**：非常慢。原因是它 inference 时还需要做 camera pose 的 iterative refinement。DGGT 完全 feedforward，所以即使 model 重一点也更快。
- **VGGT++ 表现差**：作者归因为 attention feature 保留 RGB detail 不足。这印证了 DGGT 用 $F_{dino}$ skip fusion 的必要性。

### 7.2 Cross-dataset generalization（Tab.2）

Zero-shot on nuScenes / Argoverse2：

| Method | nuScenes PSNR | Argoverse2 PSNR |
|---|---|---|
| MVSplat | 17.84 | 18.67 |
| NoPoSplat | 19.75 | 22.00 |
| DepthSplat | 19.52 | 22.05 |
| STORM | 17.77 | 20.83 |
| **DGGT (zero-shot)** | **25.31** | **26.34** |
| **DGGT (trained on target)** | **26.63** | **26.96** |

**DGGT zero-shot on nuScenes 25.31 vs STORM trained on nuScenes 24.54**。这是 paper 里最强的一行结果——zero-shot 居然超过 in-domain 训练的 STORM。这强烈支持 pose-free formulation 的 generalization 优势。

### 7.3 3D Motion Estimation（Tab.5）

| Method | EPE3D (m) ↓ | Acc5 (%) ↑ | Acc10 (%) ↑ | θ (rad) ↓ |
|---|---|---|---|---|
| NSFP | 0.698 | 42.17 | 54.26 | 0.919 |
| NSFP++ | 0.711 | 53.10 | 63.02 | 0.989 |
| STORM | 0.276 | 81.12 | 85.61 | 0.658 |
| **DGGT** | **0.183** | **85.42** | **90.42** | **0.328** |

EPE3D（end-point-error）从 0.276 → 0.183，下降 33%。Angular error 从 0.658 → 0.328，下降 50%。这个 angular error 的下降特别重要——它说明 motion vector 不只是 magnitude 对了，**direction 也对了**。对 downstream tracking 任务这是关键。

### 7.4 View 数 scalability（Tab.3）

STORM 在 view 数从 4 → 8 → 16 时 PSNR 严重退化（26.55 → 25.11 → 23.69），DGGT 几乎稳定甚至小幅上升（30.54 → 31.41 → 30.66）。这个 difference 的 root cause：STORM 把所有 view 同时塞进一个固定-size transformer，view 多了 token 数线性增长，attention 的 quadratic 复杂度爆炸；DGGT 用 alternating attention 把 view 内和 view 间解耦，复杂度更接近 linear scaling。

## 8. Scene Editing 应用

DGGT 的 representation 因为是 *object-level decomposed Gaussian*，scene editing 是 free lunch：

- **Remove car**：把 dynamic Gaussian 里 car 对应的 pixel 直接 mask 掉，rerender
- **Shift car**：把 dynamic Gaussian 的 $\mu$ 加一个 offset，rerender
- **Insert new object**：从另一个 scene 重建一辆车 / 行人的 dynamic Gaussian，union 进当前 scene 的 $\hat{G}^t$，rerender

Diffusion refinement 在 editing 后特别有用：因为 Gaussian manipulation 会留下 holes / artifacts（见 Fig.5 红框），diffusion 能 fill in。

这个 workflow 的工业价值很高：autonomous driving 的 corner case generation、data augmentation、what-if simulation 都需要这种 editing 能力。STORM 因为没有 explicit object decomposition，做不了这种 editing。

## 9. 与相关工作的连接

### 9.1 与 STORM 的本质差异

STORM [https://arxiv.org/abs/2501.00602] 是这个 line 最近的 SOTA，差异：
1. **STORM 需要 pose，DGGT 不需要**——这是 formulation 级别差异
2. **STORM 用 constant velocity motion model，DGGT 用 per-query 3D trajectory**——表达能力差异
3. **STORM 没有 lifespan，static appearance 变化时崩**——temporal modeling 差异
4. **STORM 没有 diffusion refinement**——artifact 处理差异
5. **STORM 长序列会 diverge，DGGT 长序列稳定**——scalability 差异

### 9.2 与 VGGT 的关系

VGGT [https://arxiv.org/abs/2503.11651] 是 DGGT 的 backbone 来源。VGGT 做的是 *static* pose-free multi-view reconstruction，DGGT 把它扩展到 *dynamic*。扩展方式：
- 加 dynamic head
- 加 motion head
- 加 lifespan head
- 加 sky head
- 加 diffusion refinement

可以说 DGGT = VGGT + dynamic extension + diffusion post-processing。

### 9.3 与 NoPoSplat 的关系

NoPoSplat [https://arxiv.org/abs/2410.24207] 也是 pose-free feedforward Gaussian splatting，但只做 static。它需要 inference 时 refine pose，所以慢（23.22s）。DGGT 完全 single-pass，不需要 refinement。

### 9.4 与 4D Gaussian Splatting line 的关系

4D-GS [https://arxiv.org/abs/2311.11235] 是 per-scene optimization 路线。DGGT 把 4D-GS 的核心 idea（time-conditioned Gaussian）转成 feedforward 形式，并用 lifespan 参数化代替 explicit time embedding，更 compact 且 disentangled。

### 9.5 与 EmerNeRF 的关系

EmerNeRF [https://arxiv.org/abs/2311.02077] 提出 *emergent decomposition*：让 NeRF 自己 emergent 出 static 和 dynamic 的 decomposition，不需要 bbox annotation。DGGT 沿用这个 idea 但用 Gaussian 代替 NeRF，并且加 explicit dynamic mask supervision（来自 LiDAR bbox + SAM2）作为 auxiliary，让 decomposition 更稳定。

## 10. Limitations 与 Potential Improvements

Paper explicit 提到的 limitation：
1. Dynamic mask 不准时崩
2. Heavily occluded motion 时 tracking 失败

我能想到的 additional limitations 和改进方向：

1. **Motion interpolation 假设 shape 不变**：对 deformable object（行人肢体、自行车手）会有 artifact。可以加 shape interpolation 或者 per-Gaussian deformation field。
2. **Lifespan 是 global 标量 per Gaussian**：可以扩展成 anisotropic temporal Gaussian，让不同时间方向有不同 decay。
3. **Sky hemisphere 是 fixed radius**：对 aerial view 或者大幅 pitch 角度变化时会不够 flexible。可以做成 learnable radius 或者 cubemap。
4. **Diffusion refinement 单步**：单步 ADD 可能 miss 一些 hard artifact。可以做成 *adaptive step*，根据 artifact 严重程度动态决定步数。
5. **没有 explicit material / lighting decomposition**：对 relighting 应用不够，未来可以加 BRDF head。
6. **Motion head 用 TAPIP3D 预训练，但 driving 特有的 motion pattern（如车辆 rigid body motion）没显式利用**：可以加 vehicle pose head 用 bbox 类 annotation 监督。
7. **没有处理 reflective surface**（玻璃、湿路面）：driving 场景这块很多，DGGT 没特别处理，会被 lifespan + opacity 简单 modulate 掉导致 artifact。
8. **Long sequence scalability 虽然 paper 说支持 arbitrary length，但实际 transformer 的 attention memory 还是 O(N²)**。可以用 sparse attention 或 chunked processing。

## 11. 一些延伸的思考与联想

### 11.1 与 World Model 的连接

最近 driving world model 这条线很热（如 GAIA-2 [https://arxiv.org/abs/2503.02350]、DriveDreamer [https://arxiv.org/abs/2310.09777]、Copilot4D [https://arxiv.org/abs/2311.01017]）。DGGT 这种 *feedforward 4D Gaussian reconstruction* 可以作为 world model 的 *deterministic initialization*——先用 DGGT reconstruct 出当前帧的 4D state，再让 generative model 在此基础上 rollout。这比 pure video diffusion world model 更 controllable。

### 11.2 与 NeRF Marine / Gaussian Splatting Robotics 的连接

Robotics SLAM line 也在用 Gaussian Splatting 做 dense reconstruction（如 SplaTAM [https://arxiv.org/abs/2403.02751]、MonoGS [https://arxiv.org/abs/2404.16742]）。DGGT 的 pose-free formulation 对 SLAM 也有启发——SLAM 的核心难点就是 pose estimation + dense reconstruction 联合优化，DGGT 把它们一起 feedforward 出来，可以作为 SLAM 的 *dense prior* 或者 *initialization*。

### 11.3 与 L4GM 的对比

L4GM [https://arxiv.org/abs/2410.21274] 是第一个 4D large reconstruction model，主要做 object-level。DGGT 是 scene-level + driving domain。两者思路相似（feedforward 4D），但 DGGT 处理 unbounded scene + dynamic decomposition，技术难度更高。

### 11.4 与 DUSt3R / MASt3R / VGGT 的 lineage

DUSt3R [https://arxiv.org/abs/2312.14132] → MASt3R [https://arxiv.org/abs/2406.09681] → VGGT [https://arxiv.org/abs/2503.11651] 这条 line 是 *pose-free 3D* 的 backbone evolution。DGGT 选了 VGGT 作为起点，因为 VGGT 已经把 alternating attention + DINO feature + multi-head prediction 这套打通了。DGGT 的贡献是在 VGGT 之上加 *dynamic-aware heads*。

### 11.5 与 SAM2 的协作

DGGT 用 SAM2 [https://arxiv.org/abs/2408.00714] 做 dynamic mask propagation（Appendix A.1）。这是 *pretrained vision foundation model* 的典型应用——SAM2 提供 temporally consistent mask，DGGT 把它作为 supervision 而非 input。这条 path 未来可能 reverse：SAM2 进化后可以直接 integrate 进 DGGT 的 backbone，做 fully end-to-end mask prediction。

### 11.6 与 Driving Data Generation Pipeline 的对接

工业上 driving data generation pipeline 一般是：sensor log → reconstruction → editing → simulation → new sensor log。DGGT 直接 fit 在第二、三步——fast reconstruction 让 pipeline 真的能 scale，editing 能力让 what-if 分析可行。可以预计小米 EV 内部已经在 production pipeline 里用这个或者它的变体。

## 12. 结论与 takeaway

DGGT 的核心贡献可以浓缩成三句话：
1. **Reformulate pose from input to output**——获得 cross-dataset generalization。
2. **Disentangle static/dynamic with explicit motion head + lifespan**——获得 stable long-sequence dynamic reconstruction。
3. **Diffusion post-processing injects natural image prior**——获得 artifact-free high-fidelity rendering。

这三个加起来让 DGGT 在 0.39s inference time 下达到 27.41 PSNR，同时支持任意 view 数和 long sequence，且 zero-shot 跨 dataset 都能跑。对 autonomous driving 的 *reconstruction-as-preprocessing* vision 来说，这是一个 milestone。

从 research direction 看，DGGT 标志着 *feedforward 4D reconstruction* 这个 sub-field 正式成熟。接下来几年的发展方向我推测会是：(1)更大规模 pretraining 把 DGGT 推向 foundation model；(2) joint training with downstream task（planning, prediction）；(3) incorporation with lidar / radar multi-modal input；(4) explicit physics / dynamics prior 注入 motion head。

---

**References**
- DGGT paper: https://github.com/xiaomi-research/dggt
- STORM: https://arxiv.org/abs/2501.00602
- VGGT: https://arxiv.org/abs/2503.11651
- NoPoSplat: https://arxiv.org/abs/2410.24207
- MVSplat: https://arxiv.org/abs/2403.11351
- DepthSplat: https://arxiv.org/abs/2410.13862
- EmerNeRF: https://arxiv.org/abs/2311.02077
- L4GM: https://arxiv.org/abs/2410.21274
- TAPIP3D: https://arxiv.org/abs/2504.14717
- ADD (Adversarial Diffusion Distillation): https://arxiv.org/abs/2311.17042
- DINOv2: https://arxiv.org/abs/2304.07193
- SAM2: https://arxiv.org/abs/2408.00714
- DUSt3R: https://arxiv.org/abs/2312.14132
- MASt3R: https://arxiv.org/abs/2406.09681
- 3DGS: https://arxiv.org/abs/2308.04079
- 4D-GS: https://arxiv.org/abs/2311.11235
- Copilot4D: https://arxiv.org/abs/2311.01017
- Waymo Open Dataset: https://arxiv.org/abs/1912.07212
- nuScenes: https://arxiv.org/abs/1903.11006
- Argoverse2: https://arxiv.org/abs/2301.00493
- Neural Style Transfer (Gram loss): https://arxiv.org/abs/1508.06576
- SplaTAM: https://arxiv.org/abs/2403.02751
- MonoGS: https://arxiv.org/abs/2404.16742
