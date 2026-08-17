---
source_pdf: Sparse-view Pose Estimation and Reconstruction via Analysis by Generative
  Synthesis.pdf
paper_sha256: b0e29b21de88719a64640bef1c6c8d6e4b6a78a473712a18137c8e434a0d9670
processed_at: '2026-08-12T08:54:37-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SparseAGS

Andrej, 我换个画风，抛开公式，直接讲 intuition。

---

## 这篇 paper 在解决什么问题？

想象你拍了 6 张一个机器人的照片，从不同角度。你想重建出这个机器人的 3D 模型。

问题来了——你不知道每张照片是从哪个角度拍的。这就尴尬了：

- 要重建 3D，你得知道每张照片的 camera pose
- 要猜 camera pose，你又得先有个 3D 模型做 reference

这就是个鸡生蛋蛋生鸡的死循环。心理学里叫 mental rotation，人脑也面临同样困境，但我们大脑 somehow 能解。computational 上怎么解？这就是这篇 paper 的事。

---

## 为什么之前的路子不行？

有三派人试过，各有各的坑：

**第一派：SfM**。传统方法，靠 image 之间的 feature matching 算 camera pose。但 sparse-view 下 overlap 太少，matching 失败，整个崩盘。

**第二派：直接学 pose**。用 deep learning 从 images 直接 predict camera poses（比如 RelPose++, Ray Diffusion, DUSt3R）。这些方法挺强，但偶尔会犯大错，而且不建模 3D，refine 不动。

**第三派：假设 pose 已知，重建 3D**。比如 SparseFusion, DreamSparse。结果很好，但现实中谁给你精确 pose？没法用。

这三派都只解了半边问题。SparseAGS 的野心是**把两边拼起来**——用 off-the-shelf pose estimator 给个初始猜测，然后同时 refine pose 和 重建 3D。

---

## 经典思路为什么不 work？

经典思路叫 analysis by synthesis，说白了就是：猜一个 3D 模型，渲染出来，和真实照片比，不对就调，直到对为止。

$$\min_{\theta, \Pi} \sum_i ||I_i - f_\theta(\pi_i)||^2$$

这个思路在 dense view（很多照片）下 work，因为信息充足。但 sparse-view 下有两个致命问题：

**问题一：3D 会 overfit**。只有 6 张照片，3D representation 又有 huge capacity（比如 NeRF 几百万参数），它会直接 overfit 到这 6 张照片上，input view 看着 OK，novel view 一塌糊涂。相当于死记硬背了 6 张图，没学到真正的 3D structure。

**问题二：pose 误差大时优化卡住**。photometric loss 在 pose space 上是非凸的，初始 pose 错得厉害，gradient descent 直接 stuck 在 local optima。

---

## SparseAGS 的核心 idea

一句话总结：**给 analysis-by-synthesis 加上 generative prior，让 3D 不光 fit 照片，还要"看起来合理"**。

具体说，除了让渲染结果 match input photos，还从 random novel views 渲染图，要求这些图落在 diffusion model 学到的 natural image distribution 上。diffusion model 见过 millions 张图，知道什么样的物体长什么样，充当了一个 learned common sense prior。

这相当于从 pure observation（只看照片）升级到 observation + imagination（看照片 + 用常识补全）。sparse-view 的根本痛点是 information 不足，generative prior 就是补 information 的。

---

## 怎么实现的？分三步

### Step 1: 改造 diffusion model 让它能处理 real-world camera

作者用 DreamGaussian 作 base，DreamGaussian 用 Zero-1-to-3 这个 diffusion model 做 prior。但 Zero-1-to-3 有个硬伤：它只接受 3-DoF camera 参数（azimuth, elevation, radius），假设 camera 永远朝向 object 中心、没有 in-plane rotation。

这对 synthetic data（Objaverse 渲染的图）OK，但 real-world 照片哪有这么规矩？手机横拍竖拍有 rotation，focal length 不同有 scale 变化，camera 不一定看向 object 中心。

作者把 3-DoF 升级成 6-DoF，编码成 18-dim vector：把 $4 \times 4$ relative extrinsic matrix flatten 成 16 维，加上 2 维 focal length ratio 的 log。然后 finetune Zero-1-to-3，只更新 camera conditioning 相关的层（linear projection + cross-attention），冻结 U-Net 主体。这样既保留了 image prior，又让 camera conditioning 适配 real world。

### Step 2: 让 SDS 用上多张 input image

原始 DreamGaussian 只用单张 reference image 做 SDS。SparseAGS 修改成：从 N 张 input image 都做 noise prediction，然后简单 averaging。

$$\overline{\epsilon}_\phi = \frac{1}{N} \sum_{i=1}^N \epsilon_\phi(\mathbf{z}_t; t, I_i, \Delta\pi_i)$$

为什么简单 averaging？几个 intuition：

- **Variance reduction**：SDS 的 gradient variance 很高，averaging N 个 prediction 降方差
- **Ensemble effect**：每个 view 提供一种 "novel view 该长啥样" 的 hypothesis，averaging 相当于 ensemble
- **不按 closeness 加权**：因为初始 poses 不可靠，不知道哪个 view 真的"近"，简单平均更 robust

### Step 3: 处理 outlier poses

off-the-shelf pose estimator 偶尔会给个离谱的 pose。naive 把所有 image 塞进去重建 3D，一个 outlier 就能把整体 3D 带偏。

作者的 insight 很妙：一个 outlier 不仅自己 reprojection error 高，更关键的是**把它加入训练会拖累其他 views 的重建质量**。这是 negative transfer。

检测方法：leave-one-out。把 image i 拿掉重建一次 3D，和加入 i 重建的 3D 比，如果拿掉后其他 views 的 error 显著降低，那 i 就是 outlier。

检测到 outlier 后，用 inliers 重建的 3D 做 reference，在 sphere 上密集采样 pose candidates，render 出来和 outlier image 比（MSE + LPIPS 联合 rank），找最匹配的 pose 做纠正，再 gradient 微调。

---

## 迭代框架长这样

```
For k = 1 to K:
    1. 用当前 poses 重建 3D（θ_k = MV-DG(I, Π_{k-1})）
       - 3D 从头重建，不继承上一次的
    2. 用重建的 3D refine poses（Π_k = GD(I, θ_k, Π_{k-1})）
       - poses 累积优化，不 reset
    3. (k > 1) 先 filter outliers，只用 inliers 重建
    4. outlier 用 render-and-compare 纠正 pose
    5. 纠正完再用所有 image 重建一次 final 3D
```

**关键设计：3D reset，poses 累积**。为什么不对称？

- 3D 表示 capacity 高（millions of parameters），容易 overfit 历史，每次 reset 让它从更准的 poses 重新 fit，摆脱之前的 bias
- Poses 是低维（6-DoF × N），信息量少，需要 refinement 累积，且 pose space 优化相对稳定

这类似 EM 算法 E-step / M-step 的 alternation，但加了 reset 防止 3D overfitting。

---

## 效果如何？

### Pose accuracy

在 NAVI 数据集上，用 8 张 image：

- RelPose++ 初始 Rot.@5° 只有 10.9%，SparseAGS 提到 42.1%（+31.2）
- Ray Diffusion 初始 13.5%，SparseAGS 提到 60.3%（+46.8）
- DUSt3R 初始 52.3%，SparseAGS 提到 83.7%（+31.4）

对比 SPARF（之前的 sparse-view pose-NeRF co-optimization）：
- SPARF 有时反而让 pose 变差（RelPose++ Rot.@15° 从 56.4 降到 51.9）
- 因为 SPARF 依赖 dense correspondences，object-centric 下 viewpoint 变化大，false matches 误导
- SPARF 训练要 10 小时，SparseAGS 只要 5-10 分钟

### 3D reconstruction (NVS)

对比 LEAP 和 UpFusion（不建模 pose 的 feedforward 方法）：

- LEAP 在 N=6,8,10 下 PSNR 基本不变（12.84 → 12.98），完全 saturate
- SparseAGS 持续提升（15.56 → 17.03 → 18.03）
- 因为 LEAP training 时用 5 views，超出 capacity 就饱和了
- SparseAGS 是 optimization-based，flexible w.r.t. N，加 image 就变好

---

## Ablation 说了什么？

用 Ray Diffusion (N=8) 做消融：

| 配置 | Rot.@5° | PSNR | F1@0.01 |
|------|---------|------|---------|
| 只有 pose-3D co-opt（无 SDS） | 28.4 | 12.72 | 46.3 |
| 加 vanilla Zero123 SDS | 30.2 | 13.04 | 49.9 |
| 换成 6-DoF Zero123 SDS | 34.6 | 13.44 | 57.2 |
| 再加 outlier handling | **60.3** | **15.30** | **68.2** |

三个 takeaway：

**1. Generative prior 用错了反而有害**。vanilla Zero123 用 3-DoF camera，在 real image 上 conditioning 错了，diffusion 给的 noise prediction 是错的，Rot.@15° 和 CC@0.1 反而下降。Prior 要 match task 才有用。

**2. 6-DoF conditioning 是 real-world 的门槛**。换成 6-DoF 后全面提升，说明不能表达 real camera 的 prior 根本不能用。

**3. Outlier handling 贡献最大**。从 (3) 到 (4)，Rot.@5° 跳了 +25.7，远超 generative prior 的贡献。说明 sparse-view 下 robustness 是第一位的。

---

## 一句话总结

SparseAGS 的故事可以这么讲：

> sparse-view 3D 重建的核心痛点是 information 不足和 initial pose 误差大。用 generative prior 补 information，用 outlier handling 对付大误差，用 iterative pose-3D co-optimization 让两者 bootstrap 起来。三件事缺一不可。

更深一层的 intuition：

- **Prior 是 learned common sense**：diffusion model 见过 millions 张图，知道物体该长什么样，比 hand-crafted regularizer 强太多
- **Robustness 来自 explicit modeling**：outlier 不是噪声是 structural failure，要 leave-one-out 检测 + render-and-compare 纠正
- **Off-the-shelf + refine > end-to-end**：optimization-based 方法能 scale with input，feedforward 方法受 training 限制会 saturate
- **Prior quality 要 match task**：3-DoF prior 在 real world 有害，6-DoF 才 work

---

## 这篇 paper 对社区的启示

1. **Generative prior 是 sparse-view 的 game changer**。从 pure observation 到 observation + imagination，这是范式转变。
2. **6-DoF conditioning 是 real-world deployment 的 must-have**。任何用 synthetic 训练的 view-conditioned model 想上 real image 都得过这关。
3. **Robustness > precision**。real-world data 总有 outlier，principled 处理比追求 sub-degree precision 重要。
4. **Asymmetric update design 有道理**。3D reset + pose accumulate 反映了不同 representation 的 capacity 和 stability 差异，这个设计 pattern 可能通用。

---

## 局限性

paper 自陈三个：
1. 初始 pose 不能全错，大部分错就没救了
2. 不处理 truncation / occlusion / close-up
3. 只 object-centric，没扩展到 scene

我补充几点：
- Zero-1-to-3 在 Objaverse 上训练，对常见 category prior 强，罕见物体可能弱
- SDS 本身有 mode seeking 问题，multi-view averaging 可能加剧，novel view 可能偏模糊
- Outlier detection 的 LPIPS 阈值 0.05 是经验性的，没做 sensitivity analysis
- Render-and-compare 的 sphere sampling 密度没明确，影响精度和速度的 trade-off

---

## References

- **Project page**: https://qitaozhao.github.io/SparseAGS/
- **DreamGaussian**: https://dreamgaussian.github.io/
- **Zero-1-to-3**: https://zero123.cs.columbia.edu/
- **DUSt3R**: https://dust3r.europe.naverlabs.com/
- **Ray Diffusion**: https://cameraray.github.io/
- **SPARF**: https://sparf.github.io/
- **LEAP**: https://github.com/hjxwhy/LEAP
- **NAVI dataset**: https://navi5d.github.io/
- **3D Gaussian Splatting**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **DreamFusion (SDS)**: https://dreamfusion3d.github.io/

---

希望这版讲得更 intuitive 了，Andrej。核心就是：**sparse-view 下 information 不够，用 diffusion prior 补；initial pose 不准，用 outlier handling 兜底；两者通过 iterative co-optimization 互相 bootstrap**。Engineering 上很扎实，intuition 上也说得通。

---

# SparseAGS: Analysis by Generative Synthesis 深度解析

Andrej, 这篇 paper 触及了一个非常 fundamental 的 inverse graphics 问题——sparse-view setting 下 pose 和 3D 的联合推断。让我从底层 intuition 出发，把方法拆解得足够细。

---

## 1. Core Problem: Co-dependency 的鸡生蛋困境

paper 开篇用 mental rotation 实验 [Shepard & Metzler 1971] 类比, 这个类比相当深刻。人类认知 3D 需要 viewpoint, 理解 viewpoint 又需要 mental 3D model。computational 上这个 co-dependency 体现为:

- accurate 3D reconstruction 需要 precise camera poses (否则 photometric loss 落到错误几何上)
- 预测 camera poses 又依赖对 underlying 3D 的隐式或显式建模 (否则没有 reference frame 来 anchor relative poses)

经典 SfM [Schonberger & Frahm 2016] 用 sparse correspondences 联合解算, 但 sparse-view 下 overlap 不足直接失效。learning-based 方法分成两派: 一派 sparse-view reconstruction [SparseFusion, DreamSparse] 假设 poses known, 一派 pose estimation [RelPose++, Ray Diffusion, DUSt3R] 不建模 3D。两派都只解了半边问题。

---

## 2. Analysis by Synthesis 的经典框架与失效原因

经典 formulation 非常优雅, 给定 input images $\mathbb{I} = \{I_i\}_{i=1}^N$:

$$\min_{\theta, \Pi} \sum_{i=1}^N ||I_i - f_\theta(\pi_i)||^2 \quad (1)$$

变量含义:
- $\theta$: 3D representation (NeRF, 3D Gaussians, triplane 等)
- $\Pi = \{\pi_i\}_{i=1}^N$: N 个 camera poses 集合
- $f_\theta(\cdot)$: differentiable rendering function
- $I_i \in \mathbb{R}^{H \times W \times 3}$: 第 i 张 input image
- $\pi_i$: 第 i 张 image 对应的 6-DoF camera pose (extrinsic + intrinsic)

只要 $f_\theta$ differentiable, 就能 gradient descent 联合优化 [BARF, NeRF--]。但在 sparse-view (N 小) 下这套体系崩塌, 原因有二:

**Overfitting**: 当 N 小, photometric loss 提供的监督信号稀疏, 3D 表示有 huge capacity (尤其 NeRF), 会 overfit 到 input pixels 而不形成 plausible geometry。结果是 input views 看起来 OK, novel views 一塌糊涂, pose refinement 也被误导。

**Local optima**: pose 初始化如果误差大, gradient descent 容易陷入局部极小。photometric loss 在 pose space 上非凸, 大误差下 gradient 方向不可靠。

---

## 3. SparseAGS 的核心 idea: 加 Generative Prior

paper 的命名 "Analysis by **Generative** Synthesis" 抓住了精髓——在经典 analysis-by-synthesis 基础上注入 generative prior。形式化:

$$\min_\theta \mathbb{E}_\pi -\log p_\phi(f_\theta(\pi) | \pi, \mathbb{I}, \Pi) \quad (2)$$

变量含义:
- $p_\phi(\cdot)$: 由 diffusion model $\phi$ 建模的 conditional likelihood
- $\pi$: 从某个分布随机采样的 novel view 的 camera pose
- $f_\theta(\pi)$: 当前 3D 表示 $\theta$ 在 pose $\pi$ 下渲染出的 image
- $\mathbb{I}, \Pi$: 已知 input images 和对应 poses 作为 condition

直觉上, 这相当于要求 3D 表示渲染出来的 novel views 不仅 match input, 还要落在 diffusion model 学到的 natural image manifold 上。diffusion model 充当了 learned natural image prior, 比 hand-crafted smoothness / sparsity regularizer 强得多。

gradient 通过 Score Distillation Sampling (SDS) [DreamFusion, Poole et al. 2023] 获得, 不需要 diffusion model 本身可微, 只需要它的 score function $\nabla \log p_\phi$。

---

## 4. Preliminaries: DreamGaussian 的 two-stage 设计

SparseAGS 选 DreamGaussian [Tang et al. 2024] 作为 base, 因为它在 speed 和 fidelity 间取得了好 trade-off。two-stage 设计:

### Stage 1: 3D Gaussians optimization

用 3D Gaussians [Kerbl et al. 2023] 参数化 $\theta$, 联合优化 photometric loss + SDS loss with Zero-1-to-3 [Liu et al. 2023]。SDS gradient:

$$\nabla_\theta \mathcal{L}_{SDS} = \lambda_{SDS} \mathbb{E}_{t, \pi, \epsilon} \left[ w(t) \left( \epsilon_\phi(\mathbf{z}_t; t, I_1, \Delta\pi) - \epsilon \right) \frac{\partial f_\theta(\pi)}{\partial \theta} \right] \quad (3)$$

变量逐项解析:
- $\lambda_{SDS} \in \mathbb{R}^+$: SDS loss 的总体系数
- $t \sim \mathcal{U}(0, T)$: diffusion 的 timestep, $T$ 是最大步数
- $\pi$: 随机采样的 novel view pose
- $\epsilon \sim \mathcal{N}(0, I)$: 添加的 Gaussian noise
- $w(t)$: 关于 timestep $t$ 的 weighting function (通常与 noise schedule 相关)
- $\mathbf{z}_t = \alpha_t \cdot \text{Enc}(f_\theta(\pi)) + \sigma_t \cdot \epsilon$: noisy latent, 其中 $\alpha_t, \sigma_t$ 是 noise schedule 系数, $\text{Enc}(\cdot)$ 是 VAE encoder
- $\epsilon_\phi(\cdot)$: U-Net 预测的 noise
- $I_1$: 单张 reference input image
- $\Delta\pi = \pi \ominus \pi_1$: relative camera pose from reference 到 novel view
- $\frac{\partial f_\theta(\pi)}{\partial \theta}$: 渲染 image w.r.t. 3D 参数的 Jacobian, 通过 differentiable rendering 获得

直观理解: SDS gradient 推 $\theta$ 让 rendering 的 latent 在 noise prediction 上更接近"不加噪声"的状态, 也就是让 rendering 更像 diffusion model 认为的 plausible image。

Stage 1: 500 steps, ~1 min on single GPU。

### Stage 2: Texture refinement on mesh

3D Gaussians → textured mesh via Marching Cubes [Lorensen & Cline 1987], 只优化 texture。50 steps, ~30 sec。

---

## 5. DreamGaussian 的两个 Key Limitations

paper 指出 DreamGaussian 直接用不了, 有两个核心问题:

### Limitation 1: 3-DoF Camera Parameterization

Zero-1-to-3 用 (radius, elevation, azimuth) 表示 camera, 假设:
- 没有 in-plane rotation (camera up vector 永远 world up)
- 所有 camera 朝向共同 origin (object-centric 强假设)

这套对 synthetic data (e.g., Objaverse 渲染) OK, 但 real-world image 可能:
- 手机横拍竖拍有 in-plane rotation
- focal length 不同导致 scale 变化
- camera 不一定看向 object center

3-DoF 表示无法 encode 这些, diffusion model 在 conditioning 上 confused, 给出错误的 noise prediction。

### Limitation 2: Single Input Image

DreamGaussian 的 SDS 只用单张 $I_1$ 作 condition, 无法 aggregate 多 view 的 visual cues。sparse-view reconstruction 需要让 3D faithful to 多张 image 的细节。

---

## 6. 6-DoF Camera Conditioning: 18-Dim Vector

SparseAGS 的解决方案是把 3-DoF 替换为 6-DoF, 编码成 18-dim vector:

$$[\text{Flatten}(\pi_{rel}), \log(f_{rel}^x), \log(f_{rel}^y)] \quad (4)$$

逐项解析:
- $\pi_{rel} \in \mathbb{R}^{4 \times 4}$: source view 到 target view 的 relative extrinsic matrix, 包含 rotation ($3 \times 3$) 和 translation ($3 \times 1$), 以及 homogeneous row
- $\text{Flatten}(\pi_{rel}) \in \mathbb{R}^{16}$: 把 $4 \times 4$ matrix 展平
- $f_{rel}^x = f_{target}^x / f_{source}^x$: x 轴 focal length ratio
- $f_{rel}^y = f_{target}^y / f_{source}^y$: y 轴 focal length ratio
- $\log(\cdot)$: 取对数, 把 multiplicative scale 变成 additive, 训练更稳定, 也对应 Gaussian 分布假设
- 总维度: 16 + 1 + 1 = 18

为什么包含 focal length ratio? 因为 real-world image 经常被 crop, focal length 变化导致 object scale 变化。如果不 encode, diffusion model 会把 scale mismatch 误判为 viewpoint 变化。

### Finetuning 策略

- Initialize from Zero123-XL checkpoint [Objaverse-XL 训练版本]
- **只 finetune camera conditioning 相关层**: linear projection layer + cross-attention layers
- 冻结其他所有层 (U-Net 主体)
- 用 CO3D [Reizenstein et al. 2021] + Objaverse renderings 训练, 缓解 synthetic data bias
- 8 V100 GPUs, batch 36/GPU, gradient accumulation 6, 23,500 iters, ~4 days

这个部分 finetuning 策略很聪明——既保留 Zero-1-to-3 学到的 image prior, 又让 camera conditioning 适应 6-DoF。

注意 ZeroNVS [Sargent et al. 2024] 也讨论过 3-DoF 问题, 提出 "6-DoF+1" for scene-level, 但 trained on complex backgrounds + depth priors, 不适用 object-centric setting。

---

## 7. Multi-view SDS: Averaging Noise Predictions

让 SDS aware 多 view 信息, 修改 Eq. 3:

$$\nabla_\theta \mathcal{L}_{\text{Multi-view SDS}} = \lambda_{SDS} \mathbb{E}_{t, \pi, \epsilon} \left[ w(t) \left( \overline{\epsilon}_\phi - \epsilon \right) \frac{\partial f_\theta(\pi)}{\partial \theta} \right] \quad (5)$$

其中 averaged noise prediction:

$$\overline{\epsilon}_\phi = \frac{1}{N} \sum_{i=1}^N \epsilon_\phi(\mathbf{z}_t; t, I_i, \Delta\pi_i) \quad (6)$$

变量含义:
- $N$: input views 总数
- $I_i$: 第 $i$ 张 input image
- $\Delta\pi_i = \pi \ominus \pi_i$: 第 $i$ 张 input image w.r.t. sampled novel view $\pi$ 的 relative pose
- $\overline{\epsilon}_\phi$: 所有 input views 在**相同 timestep $t$** 下 noise prediction 的算术平均

### 为什么简单 averaging 有效?

这里有个深层 intuition 值得深挖:

1. **Variance reduction**: SDS 用单次 noise sample 估计 gradient, variance 很高 (这是 SDS 著名的问题)。averaging N 个 noise predictions, 假设它们近似独立, variance 降低 ~N 倍, 训练更稳定。

2. **Implicit hypothesis ensemble**: 每个 input view 通过 conditioning 提供一种 "novel view 应该长什么样" 的 hypothesis。averaging 相当于 ensemble, 降低单 view 的 bias, 形成更鲁棒的 prior。

3. **不按 closeness 加权的设计选择**: Stable-Dreamfusion [Tang 2022] 实现里会按 relative pose 的 closeness 加权 (close views 权重大)。SparseAGS 故意不做加权, 因为 sparse-view 下 camera poses 不可靠, 过分依赖 "close" views 会引入 conflict。这是 robustness 考虑。

paper 把这套整体称作 **MV-DreamGaussian**, 记作 $\theta = MV\text{-}DG(\mathbb{I}, \Pi)$。

---

## 8. Pose Refinement: Custom CUDA + Photometric Backprop

在 MV-DreamGaussian 重建 3D 过程中, photometric loss 的 gradient 不仅更新 $\theta$, 还 backprop 到 camera poses $\Pi$。这需要 custom CUDA kernels 来实现 differentiable rendering w.r.t. camera parameters。

记 $\Pi' = GD(\mathbb{I}, \theta, \Pi)$ 为给定 images, 3D, 初始 poses 优化后的 poses。整体迭代框架:

$$\text{For } k = 1 \cdots K: \quad \theta_k = MV\text{-}DG(\mathbb{I}, \Pi_{k-1}); \quad \Pi_k = GD(\mathbb{I}, \theta_k, \Pi_{k-1}) \quad (7)$$

变量含义:
- $K$: 总 outer iteration 数
- $k$: 当前 iteration index
- $\theta_k$: 第 $k$ 次迭代的 3D 表示
- $\Pi_k$: 第 $k$ 次迭代后的 camera poses
- $\Pi_{k-1}$: 上一次迭代的 poses, 作为本次初始化
- $\theta_k$ **每次都 reset 从头重建**, 不累积
- $\Pi_k$ **累积优化**, 不 reset

### 为什么 3D reset 但 poses 累积?

这个不对称设计很关键, intuition:
- 3D 表示 capacity 高 (3D Gaussians 有 millions of parameters), 容易 overfit 历史 information, 累积会带 historical bias, 反复 reset 让它从更准的 poses 重新拟合, 摆脱之前的局部结构
- Poses 是低维 (6-DoF × N), 信息量少, 需要 refinement 累积, 而且 pose space 优化相对稳定, 不容易 stuck 在 historical bias

这种 asymmetric update schedule 类似 EM 算法中 E-step 和 M-step 的 alternation, 但加上 reset 防止 3D overfitting。

---

## 9. Outlier Identification: Leave-One-Out 思想

迭代框架 susceptible to local optima, 不 robust to 大初始 pose error。paper 提出 outlier detection + correction。

### Key Insight

一个 outlier image 不仅自己 reprojection error 高, 更重要的是**把它纳入 training 会 degrade 整体 3D 质量**, 导致其他 views 的 reconstruction 也变差。这是 negative transfer 现象。

### Formalization: Leave-One-Out

用 $\mathbb{I}^{-i}$ 表示去掉第 $i$ 张 image 后的 set, $\mathcal{E}(\theta, \mathbb{I}, \Pi)$ 表示 average reprojection error。image $i$ 是 outlier 当且仅当:

$$\mathcal{E}(MV\text{-}DG(\mathbb{I}, \Pi), \mathbb{I}^{-i}, \Pi^{-i}) >> \mathcal{E}(MV\text{-}DG(\mathbb{I}^{-i}, \Pi^{-i}), \mathbb{I}^{-i}, \Pi^{-i}) \quad (8)$$

变量含义:
- $\mathcal{E}(\theta, \mathbb{I}, \Pi) = \frac{1}{|\mathbb{I}|} \sum_{I \in \mathbb{I}} \text{LPIPS}(I, f_\theta(\pi_I))$: average reprojection error (paper 用 LPIPS 作 metric, 阈值 0.05)
- $MV\text{-}DG(\mathbb{I}, \Pi)$: 用**所有** images + 当前 poses 重建的 3D
- $\mathbb{I}^{-i}, \Pi^{-i}$: 去掉第 $i$ 个 image 和 pose 后的子集
- **左边**: 用所有 images 重建的 3D, 在 $\mathbb{I}^{-i}$ 上测 error (即去掉 $i$ 后剩下的 views 上的表现)
- **右边**: 不用 $i$ 重建的 3D, 在 $\mathbb{I}^{-i}$ 上测 error
- 如果左边 >> 右边, 说明加入 $i$ 拖累了其他 views 的 reconstruction, $i$ 是 outlier

这本质是 **leave-one-out cross-validation** + **influence function** 的思想。在 robust statistics 里, 这等价于测量每个 training point 对 held-out performance 的影响。

### Efficiency Optimization

不做 exhaustive search over all images, 而是**按 reprojection error 降序遍历 candidates**。error 高的更可能是 outlier, 优先检测。

每次迭代 (除 $k=1$) 先 filter outliers:

$$\mathbb{I}_{k-1}^{inlier}, \Pi_{k-1}^{inlier} \equiv \text{filter-outliers}(\mathbb{I}, \theta_{k-1}, \Pi_{k-1}) \quad (9)$$

然后只用 inliers 重建 3D: $\theta_k = MV\text{-}DG(\mathbb{I}_{k-1}^{inlier}, \Pi_{k-1}^{inlier})$

终止条件:
- outlier candidate 不再满足 Eq. 8 (即被判定为 inlier)
- 或 inliers 数量低于阈值: N=6,8 时 4 个; N=10 时 6 个; N=16 时 12 个

---

## 10. Outlier Correction: Render-and-Compare

只 identify outliers 会丢失 image 细节, 所以要"纠正" outlier poses。用当前 inliers 重建的 3D 作 reference, 对每个 outlier 做 render-and-compare (类似 MegaPose [Labbé et al. 2022] 思路):

1. **Discrete search**: 在 sphere 上密集采样 pose candidates $\{\pi^{(j)}\}_{j=1}^M$
2. **Render**: 从当前 3D $\theta$ 渲染每个 candidate pose 的 image: $\{f_\theta(\pi^{(j)})\}_{j=1}^M$
3. **Rank**: 对每个 candidate 计算两个 metric:
   - Pixel-space error: $\text{MSE}(I_{outlier}, f_\theta(\pi^{(j)}))$
   - Perception error: $\text{LPIPS}(I_{outlier}, f_\theta(\pi^{(j)}))$
4. **Select**: 按 cumulative rank (两个 metric 排名之和) 选最高的 candidate 作为初始 corrected pose
5. **Continuous optimization**: 用 gradient descent 微调

### 为什么 MSE + LPIPS 联合 rank?

- MSE: pixel-level, 精确但容易被 illumination / texture difference 误导
- LPIPS: perception-level, 鲁棒但可能 miss fine details (e.g., 高频纹理)
- Cumulative rank: 取两者都好的 candidate, 避免 single metric 的 failure mode

所有 outliers 纠正后, 再做一次 reconstruction 形成包含所有 images 的 consistent 3D。

### Inference Time

- 单 RTX A5000 GPU
- 8 images 输入: 一次 MV-DreamGaussian ~2 min
- 每个 outlier render-and-compare ~1 min
- RayDiffusion init 在 NAVI 上平均 0.94 outliers/sequence
- 完整 pipeline ~9 min
- SPARF [Truong et al. 2023] 训练 ~10 hours

效率优势 ~60×。

---

## 11. 实验数据深度分析

### Table 1: Pose Accuracy vs SPARF (N=8)

| Baseline | 方法 | Rot.@5° ↑ | Rot.@15° ↑ | CC@0.1 ↑ |
|----------|------|-----------|------------|----------|
| RelPose++ | baseline | 10.9 | 56.4 | 26.0 |
| | w/ SPARF | 28.6 (+17.7) | 51.9 (-4.5) | 37.9 (+11.9) |
| | w/ SparseAGS | **42.1 (+31.2)** | **67.6 (+11.2)** | **53.3 (+27.3)** |
| Ray Diff. | baseline | 13.5 | 73.5 | 38.3 |
| | w/ SPARF | 46.0 (+32.5) | 76.1 (+2.6) | 65.8 (+27.5) |
| | w/ SparseAGS | **60.3 (+46.8)** | **88.2 (+14.7)** | **80.4 (+42.1)** |
| DUSt3R | baseline | 52.3 | 93.8 | 82.2 |
| | w/ SPARF | 59.7 (+7.4) | 87.8 (-6.0) | 81.9 (-0.3) |
| | w/ SparseAGS | **83.7 (+31.4)** | **96.2 (+2.4)** | **93.5 (+11.3)** |

关键观察:
1. SparseAGS 在所有 baseline × 所有 metric 上都改进
2. SPARF 在某些 metric 上**变差** (red numbers, e.g., RelPose++ Rot.@15° -4.5, DUSt3R Rot.@15° -6.0)。归因于 SPARF 依赖 dense correspondences, NAVI 的 object-centric viewpoint 变化大, false matches (e.g., symmetric patterns) 误导优化
3. 改进幅度: Ray Diff. 上 Rot.@5° +46.8 是最大改进, 因为 Ray Diff. 初始 pose 不太准但也不太离谱, 给了 SparseAGS 足够的 working space
4. DUSt3R 已经很强 (52.3), SparseAGS 还能 +31.4 到 83.7, 说明 generative prior 提供了 orthogonal 信号

### Table 2: Varying N

SparseAGS 的改进随 N 增加而增大 (e.g., DUSt3R Rot.@5°: N=6 时 +31.6, N=10 时 +57.1, N=16 时 +69.7)。这反映了 multi-view SDS 的 ensemble 效应——更多 views 提供更强的 prior, 更稳的优化。

### Table 3: NVS vs LEAP, UpFusion

| Method | Init | N=6 PSNR | N=8 PSNR | N=10 PSNR |
|--------|------|----------|----------|-----------|
| LEAP | X | 12.84 | 12.93 | 12.98 |
| UpFusion | X | 13.30 | 13.27 | / |
| SparseAGS | Ray Diff. | 13.63 | 15.30 | 16.80 |
| SparseAGS | DUSt3R | 15.56 | 17.03 | 18.03 |

关键观察:
1. LEAP 和 UpFusion **性能 saturate** with more images (12.84 → 12.98 for LEAP)
2. SparseAGS **持续提升** (15.56 → 18.03)
3. 原因: LEAP trained with 5 views, UpFusion trained max 6 images, 超出 training capacity 就 saturated。SparseAGS 是 optimization-based, flexible w.r.t. N
4. 这是 optimization-based vs feedforward 的本质区别——optimization 方法能 scale with data

---

## 12. Ablation Study 深度解读

Table 4 用 Ray Diffusion (N=8) 作 ablation:

| Config | Rot.@5° | Rot.@15° | CC@0.1 | PSNR | LPIPS | F1@0.01 |
|--------|---------|----------|--------|------|-------|---------|
| (1) Pose-3D co-opt (w/o SDS) | 28.4 | 79.9 | 57.7 | 12.72 | 0.3100 | 46.3 |
| (2) + SDS (vanilla Zero123) | 30.2 | 78.3 | 57.3 | 13.04 | 0.2999 | 49.9 |
| (3) + SDS (Our 6-DoF Zero123) | 34.6 | 83.1 | 65.3 | 13.44 | 0.2793 | 57.2 |
| (4) + Outlier Removal & Correction | **60.3** | **88.2** | **80.4** | **15.30** | **0.2304** | **68.2** |

逐项 intuition:

**(1) → (2): 加 vanilla Zero123 SDS**
- PSNR: 12.72 → 13.04 (轻微提升)
- 但 Rot.@15° 和 CC@0.1 **下降** (79.9→78.3, 57.7→57.3)
- 原因: vanilla Zero123 用 3-DoF camera, 不能表达 real-world 6-DoF, diffusion 给的 noise prediction 是错的, 误导 3D 优化, 进而 pose 优化也错。**Prior 错了反而有害**。

**(2) → (3): 换成 6-DoF Zero123**
- 全面提升 (Rot.@5° 30.2→34.6, F1 49.9→57.2)
- 说明 6-DoF conditioning 是 generative prior 在 real-world 起作用的前提
- F1 提升 7.3 说明几何质量明显改善

**(3) → (4): 加 outlier handling**
- Rot.@5°: 34.6 → 60.3 (**+25.7**)
- PSNR: 13.44 → 15.30 (+1.86)
- F1: 57.2 → 68.2 (+11.0)
- 这是最大单项贡献
- 说明 sparse-view 下 outlier 是主要 bottleneck, 处理好它能解锁巨大改进

### Ablation 的核心启示

1. **Generative prior 不是免费午餐**: 用错了 (vanilla Zero123) 反而有害。Prior quality 与 task match 很关键。
2. **Outlier handling > generative prior**: 在 ablation 中, outlier 的贡献 (+25.7 Rot.@5°) 远大于 generative prior (+4.4 from (2) to (3))。这说明 sparse-view 下 robustness 比 prior quality 更重要。
3. **两个组件协同**: 仅有 outlier handling 没有 generative prior (类似 SPARF 的 setup) 也不行, 因为 3D 表示 overfit 后 outlier detection 都不准。

---

## 13. 为什么这套设计 Work: 综合 Intuition

让我把几个深层 intuition 串起来:

### Intuition 1: Generative Prior 作为 Implicit Regularizer

sparse-view 下 photometric loss 提供的监督信号不足以约束高容量 3D 表示。SDS 通过 diffusion model 学到的 image manifold 提供额外约束, 相当于一个 learned natural image prior。这个 prior 不是 hand-crafted smoothness, 而是从 web-scale data 学到的真实 image distribution, 远比传统 regularizer 强大。

### Intuition 2: Pose-3D Co-optimization 的 Bootstrap

经典 BARF 假设初始 pose 接近正确, 小幅 refinement。SparseAGS 处理大误差的诀窍是 bootstrap: generative prior 让 3D 即使在 wrong poses 下也能形成 plausible structure, 这个 plausible structure 又反过来提供更好的 photometric signal 修正 poses。这是一个 positive feedback loop, SDS 是打破初始 dead-lock 的关键。

### Intuition 3: Reset 防止 Path Dependence

每次 iteration reset 3D 但累积 poses, 防止 3D 表示 path dependence (前一次的错误结构影响后一次)。这是 EM 算法中"完全重新 fit"的思路, 而不是 incremental update。

### Intuition 4: Outlier Detection 通过 Negative Transfer

leave-one-out 检测本质是测量 negative transfer——某个 image 加入后是否让其他 views 变差。这比单纯测量 self-reprojection error 更鲁棒, 因为 outlier 可能自己 reprojection 不算差 (overfit 到自己), 但会全局 degrade。

### Intuition 5: Render-and-Compare 利用 3D 作 Pose Anchor

outlier correction 用 inliers 重建的 3D 作 reference, 通过 render-and-compare 在 sphere 上 search。这其实把 pose estimation 问题转化成 pose verification 问题——验证比 prediction 容易, 因为有 3D 结构作 strong prior。

---

## 14. Limitations 与 Future Directions

paper 自陈三个 limitation:
1. 依赖 reasonable off-the-shelf pose estimates, 大部分预测都错时失效
2. 不处理 truncation / occlusion / close-up
3. 只 object-centric, 未扩展到 scene-level

我的额外思考:

### Limitation 4: Generative Prior 的 Category Bias

Zero-1-to-3 在 Objaverse 上训练, 对某些 category (e.g., vehicles, furniture) prior 强, 对罕见 category 可能 weak。这可能解释为什么在某些 NAVI objects 上效果一般。

### Limitation 5: SDS 的 Mode Seeking 行为

SDS 著名的问题是 mode seeking, 倾向于生成 "average" plausible image, 可能 lose fine details。Multi-view SDS 的 averaging 加剧这个问题。虽然 paper 用 photometric loss 补偿, 但在 input view 之间的 novel views 上仍可能模糊。

### Future Direction 1: Video Diffusion as Prior

如果用 video diffusion model 作 prior, 不仅能约束单 frame, 还能约束 frame 间的 temporal coherence, 对应 multi-view consistency。这可能比独立 SDS per view 更强。

### Future Direction 2: Triplane / Hash Grid 加速

3D Gaussians 已经很快, 但如果用 triplane [Efficient-GAN] 或 hash grid [Instant-NGP] 可能更快, 允许更多 outer iterations。

### Future Direction 3: Joint Training of Pose Estimator + Generative Prior

paper 用 off-the-shelf pose estimator 作 init, 但如果 end-to-end 训练 pose estimator 和 generative prior, 可能学到更好的耦合。FORGE [Jiang et al. 2024] 已经在这个方向探索。

### Future Direction 4: Diffusion-based Bundle Adjustment

PoseDiffusion [Wang et al. 2023] 用 diffusion 做 bundle adjustment。如果把 SparseAGS 的 iterative framework 替换成 diffusion-based sampling, 可能更 principled。

---

## 15. 与相关工作的 Positioning

| 方法 | Pose 处理 | 3D 处理 | Prior | Limit |
|------|-----------|---------|-------|-------|
| SfM [Schonberger 2016] | correspondences | sparse points | N/A | sparse-view 失效 |
| BARF [Lin 2021] | gradient refine | NeRF | photometric | 小误差 only |
| NeRF-- [Wang 2021] | gradient refine | NeRF | photometric | 小误差 only |
| SPARF [Truong 2023] | gradient refine | NeRF | photometric + correspondence | correspondence 易失败 |
| SparseFusion [Zhou 2023] | assumed known | NeRF distillation | view-conditioned diffusion | 需要 known poses |
| DreamSparse [Yoo 2023] | assumed known | NeRF distillation | Stable Diffusion | 需要 known poses |
| LEAP [Jiang 2024] | bypass | feedforward triplane | learned | saturates with N |
| UpFusion [Kani 2024] | bypass | SDS distillation | view-conditioned diffusion | max 6 images |
| FORGE [Jiang 2024] | feedforward | feedforward | learned | quality limited |
| **SparseAGS** | **off-the-shelf + refine** | **3D Gaussians + SDS** | **6-DoF view-conditioned diffusion** | **object-centric only** |

SparseAGS 的独特定位: 把 off-the-shelf pose estimator 和 generative prior 结合, 既利用了 pose estimator 的 scalability, 又用 generative prior 弥补 sparse-view 的 information 缺失, 还用 outlier handling 保证 robustness。三件事缺一不可。

---

## 16. 实现细节的工程考量

### Custom CUDA Kernels for Pose Gradients

paper 提到实现 custom CUDA kernels 来 enable photometric loss w.r.t. camera poses 的 gradient。3D Gaussian Splatting 原始实现里 camera pose 是 input, 不参与 gradient。SparseAGS 修改了 rendering pipeline, 让 camera extrinsic (rotation + translation) 参与 backward pass。

具体来说, 3D Gaussian splatting 把每个 Gaussian 从 object space transform 到 camera space:
$$\mathbf{p}_{cam} = R_{cam} \mathbf{p}_{obj} + t_{cam}$$
要对 $R_{cam}, t_{cam}$ 求 gradient, 需要 custom backward kernel 在 rasterization 过程中累积 gradient。这是工程上 tricky 的部分, 因为 splatting 涉及 alpha blending 和 sorting, gradient 计算 complex。

### LPIPS 阈值的 Sensitivity

outlier detection 用 LPIPS 阈值 0.05。这个值是经验性的:
- 太低: 正常 image 也被判 outlier, false positive 高
- 太高: 真 outlier 漏检, false negative 高

paper 没做 sensitivity analysis, 但这是实际部署时需要调的 hyperparameter。

### Sphere Sampling Density for Render-and-Compare

outlier correction 在 sphere 上"密集"采样 pose candidates, 但没说具体密度。这影响:
- 计算成本 (每个 candidate 要 render 一次)
- Pose 精度 (采样越密越准但越慢)

推测用 fibonacci sphere sampling 或类似均匀采样, 大概几百到几千 candidates。

---

## 17. 对 3D Vision 社区的启示

### 启示 1: Generative Prior 是 Sparse-view 的 Game Changer

sparse-view 的根本问题是 information 不足。Generative prior 通过 leveraging web-scale data 提供 "imagined" information, 这是从 pure observation 到 observation + imagination 的范式转变。

### 启示 2: Off-the-shelf + Refinement 比 End-to-end 更实用

end-to-end 训练 (e.g., LEAP, FORGE) 受 training data 限制, 不能 scale with N。SparseAGS 的 off-the-shelf + iterative refinement 模式更灵活, 能利用 best available pose estimator, 随着底层 estimator 进步自动获益。

### 启示 3: Robustness > Precision in Real-world

ablation 显示 outlier handling 贡献最大。real-world data 总有 outlier, robust 处理它们比追求 sub-degree precision 更重要。这对未来 method 设计有指导意义。

### 启示 4: 6-DoF Conditioning 是 Real-world 的 Must-have

3-DoF camera 假设在 synthetic 上 work, 但 real-world 必然失败。任何想 deploy 到 real images 的方法都必须 6-DoF, 这个 insight 对 novel view synthesis 社区尤其重要。

---

## References & Web Links

- **Project page**: https://qitaozhao.github.io/SparseAGS/
- **NeRF** [Mildenhall et al. 2020]: https://arxiv.org/abs/2003.08934
- **3D Gaussian Splatting** [Kerbl et al. 2023]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **DreamGaussian** [Tang et al. 2024]: https://dreamgaussian.github.io/
- **DreamFusion (SDS)** [Poole et al. 2023]: https://dreamfusion3d.github.io/
- **Zero-1-to-3** [Liu et al. 2023]: https://zero123.cs.columbia.edu/
- **BARF** [Lin et al. 2021]: https://chenhsuanlin.gitlab.io/bundle-adjusting-NeRF/
- **NeRF--** [Wang et al. 2021]: https://arxiv.org/abs/2102.07064
- **SPARF** [Truong et al. 2023]: https://sparf.github.io/
- **DUSt3R** [Wang et al. 2024]: https://dust3r.europe.naverlabs.com/
- **MASt3R** [Leroy et al. 2024]: https://arxiv.org/abs/2406.09756
- **Ray Diffusion** [Zhang et al. 2024]: https://cameraray.github.io/
- **RelPose++** [Lin et al. 2024]: https://amywwang.com/RelPose/
- **RelPose** [Zhang et al. 2022]: https://noeguess.github.io/
- **LEAP** [Jiang et al. 2024]: https://github.com/hjxwhy/LEAP
- **UpFusion** [Kani et al. 2024]: https://upfusion.github.io/
- **NAVI dataset** [Jampani et al. 2023]: https://navi5d.github.io/
- **SparseFusion** [Zhou & Tulsiani 2023]: https://sparsefusion.github.io/
- **DreamSparse** [Yoo et al. 2023]: https://yoojiyo.github.io/dreamsparse/
- **FORGE** [Jiang et al. 2024]: https://fywww.github.io/FORGE/
- **PoseDiffusion** [Wang et al. 2023]: https://posediffusion.github.io/
- **PF-LRM** [Wang et al. 2024]: https://postech-cvlab.github.io/posefreelrm/
- **ZeroNVS** [Sargent et al. 2024]: https://kylesargent.github.io/zeronvs/
- **ID-Pose** [Cheng et al. 2023]: https://arxiv.org/abs/2306.17140
- **Stable-Dreamfusion** [Tang 2022]: https://github.com/ashawkey/stable-dreamfusion
- **CO3D** [Reizenstein et al. 2021]: https://github.com/facebookresearch/co3d
- **Objaverse** [Deitke et al. 2023]: https://objaverse.allenai.org/
- **Objaverse-XL** [Deitke et al. 2023]: https://objaverse.allenai.org/objaverse-xl/
- **MegaPose** [Labbé et al. 2022]: https://megapose6d.github.io/
- **Marching Cubes** [Lorensen & Cline 1987]: 经典算法, 提取 isosurface
- **Mental Rotation** [Shepard & Metzler 1971]: 认知科学经典实验
- **LPIPS** [Zhang et al. 2018]: https://richzhang.github.io/PerceptualSimilarity/
- **DDPM** [Ho et al. 2020]: https://arxiv.org/abs/2006.11239
- **DDIM** [Song et al. 2021]: https://arxiv.org/abs/2010.02502
- **Structure-from-Motion** [Schonberger & Frahm 2016]: COLMAP, https://colmap.github.io/
- **Photo Tourism** [Snavely et al. 2006]: https://phototour.cs.washington.edu/

---

## 总结

SparseAGS 是一个 engineering 与 intuition 都很扎实的 work。核心 contribution 在于把 generative prior (6-DoF view-conditioned diffusion) 和 robust outlier handling 整合进 analysis-by-synthesis 框架, 解决了 sparse-view 下 pose-3D co-optimization 的两个核心难题: information 不足和 initial error 大。

从 build intuition 的角度, 这篇 paper 教会我们几件事:

1. **Prior quality 要 match task**: vanilla Zero123 在 real-world 上有害, 6-DoF 版本才有用。Prior 不是越多越好, 而是越准越好。
2. **Robustness 来自 explicit handling**: outlier 不是噪声, 是结构性的 failure mode。Leave-one-out detection + render-and-compare correction 是 principled 的解决方案。
3. **Asymmetric update is feature**: 3D reset + pose accumulate 的不对称设计反映了不同 representation 的 capacity 和 stability 差异。
4. **Off-the-shelf + refine > end-to-end in real-world**: flexibility 比 training-time optimality 更重要。

期待看到这套 framework 扩展到 scene-level, 用 video diffusion 作 prior, 或者和大型 reconstruction model (e.g., LRM, Instant3D) 结合。
