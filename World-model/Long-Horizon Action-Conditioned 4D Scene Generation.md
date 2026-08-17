---
source_pdf: Long-Horizon Action-Conditioned 4D Scene Generation.pdf
paper_sha256: 9562d4a1f635e645d32b2c499ef5eb4fb049a42f81b3bd06cc54a0cf25800a98
processed_at: '2026-08-05T15:49:26-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PerpetualWonder 人话版

好,Andrej,我用最直白的方式讲一遍,顺便把技术细节和直觉都给你铺开。

---

## 一句话概括

给一张图 + 一串动作,生成一段 4D 场景,而且做完一轮动作还能接着做下一轮,中间物体不会"魂飞魄散"。

---

## 为什么这件事难

先说 baseline WonderPlay [21] (https://WonderPlay.github.io/) 的做法,它已经踩出了 hybrid generative simulator 的第一步:

1. 用 physics simulator (Genesis, https://github.com/Genesis-Embodied-AI/Genesis) 算出物体应该怎么动, 得到一堆 physics particles 的轨迹
2. 把这些 particles 当成"骨架",挂上 3D Gaussians [18] (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 当"皮"
3. 渲染成粗糙视频, 丢给 video diffusion model (CogVideoX [50]) 做精修, 让画面变逼真

听起来挺好, 但致命伤在 **信息流是单向的**: physics particles 驱动 gaussians, video model 修的只是 gaussians, **particles 从头到尾不动**。

打个比方: 你用骨骼动画做了一个小人, 然后用 AI 把皮肤画得更逼真, 但骨骼永远是初始那副骨架。下一轮让小人做新动作, 骨架还是第一轮开始时的样子, 皮肤却被 AI 改过一轮 → 皮肤和骨骼对不上 → 撕裂。

Figure 5 里那个 shovel 就是典型: 第一轮甩到空中旋转, gaussians 跟着 video model 走, 但 particles 还在地面附近。第二轮 shovel 插 sand castle 时, 直接碎成乱码。

这个问题的本质是 **没有 closed loop**。你有一个强大的 "observation model" (video diffusion) 在告诉你"真实世界应该长这样", 但你只允许它修 appearance, 不允许它回头修 dynamics。在 long-horizon 下, 这个 gap 会累积, 系统必然 drift 到崩溃。

---

## PerpetualWonder 的核心 idea: 把骨头和皮绑在一起

他们提出一个叫 **VPP (Visual-Physical aligned Particle)** 的表示, 让 physics particles 和 visual gaussians **双向绑定**。

### 公式 1: 位置绑定

$$\mu_{j,k} = p_j + \tanh(\tilde{p}_{j,k}) \cdot \delta$$

逐个讲变量:
- $p_j \in \mathbb{R}^3$: 第 $j$ 个 physics particle 的 3D 位置, 由 physics simulator 更新
- $\tilde{p}_{j,k} \in \mathbb{R}^3$: 第 $j$ 个 particle 锚定的第 $k$ 个 gaussian 的 **learnable offset**, 是网络要优化的参数
- $\delta$: physics particle 的采样尺寸, 在 simulator 里固定, 大概是 $10^{-2}$ 量级
- $\mu_{j,k}$: 最终 gaussian 在 3D 空间的中心位置
- 下标 $j$ 指 particle index (一共 $J$ 个), $k$ 指 gaussian index (每个 particle 挂 $K$ 个)

$\tanh$ 是关键。它把 $\tilde{p}_{j,k}$ 压缩到 $(-1, 1)$, 乘 $\delta$ 后, gaussian 离 particle 最远只能漂一个 particle size。

直觉: 想象 particle 是一个 "锚点", gaussian 是系在锚点上的风筝。tanh 就是那根绳子, 风筝可以在锚点周围一个球状范围内乱飞, 但飞不走。video model 可以局部调整 gaussian 的分布(改变形状细节、次级效果), 但整体不能漂离 physics anchor。

### 公式 2: 时间维度上的"出现-消失"

$$o_t(t) = \exp\left(-\frac{1}{2} \left(\frac{t - \mu_t}{s_d}\right)^2\right)$$

- $o_t(t)$: 时刻 $t$ 的 temporal opacity
- $\mu_t$: 这个 gaussian "出现"的中心时刻 (learnable)
- $s_d$: 持续时长 (learnable)
- 形状是一个 Gaussian 函数 over time

最终 opacity $o(t) = o_s \cdot o_t(t)$, 其中 $o_s$ 是标准 3DGS 的 spatial opacity。

为什么需要这个: smoke, splash 这类次级视觉效果, 粒子是"短暂存在"的。如果每个 gaussian 在所有时刻都参与渲染, 烟雾就会变成一团永远不散的雾。借自 FreeTimeGS [42] (https://arxiv.org/abs/2502.24608) 的想法, 让 gaussian 在时间轴上能"出生"和"死亡", video model 通过 refinement 学到"你只在 t=15 附近出现一下"。

### K 的自适应配置 (Supp. D)

根据 material 类型调整每个 particle 挂几个 gaussian:

| Material | K | Gaussian scale |
|---|---|---|
| Rigid body, Cloth (surface) | 1 | = $\delta$ |
| Gas, Liquid, Sand, Snow, Elastic (volumetric) | 20 | = $0.5\delta$ |

直觉: 表面材料严格一对一, 防止大变形时 ghosting。体积材料一个 particle 要"撑起"一片体积, 所以挂 20 个小 gaussian, scale 缩到 $0.5\delta$ 来表达细粒度体积细节。这个 material-aware 配置比 universal K 简洁很多。

---

## Closed loop 怎么跑

整个系统像一个 **Bayesian filter**, 在 prediction 和 update 之间循环:

### Forward pass (prediction)

$$\hat{\mathcal{S}}_{t+1} = \Phi_p(\hat{\mathcal{S}}_t, \mathcal{A}_t)$$

- $\Phi_p$: physics operator, 用 Genesis 里各种 solver (MPM [17] for fluid/sand, PBD [25] for cloth, Shape Matching [24] for rigid)
- $\mathcal{A}_t$: 时刻 $t$ 的 user action, 包括 global force field (gravity, wind) 和 local force (push, poke)
- $\hat{\mathcal{S}}_t$: 时刻 $t$ 的 coarse scene state

一个 time window 跑 $T = 392$ 个 physics step, 每 8 步采样一帧给 video model, 得到 49 帧的粗糙视频。

Physics 更新 $p_j$, 通过公式 1 自动把 $\mu_{j,k}$ 也带走 (因为 $\tilde{p}_{j,k}$ 是相对 offset, 不变)。**皮自动跟着骨头动**。

### Backward pass (update)

把粗糙 4D scene 渲染成 RGB + optical flow, 丢给 video model (CogVideoX + Go-with-the-flow [5] warped noise control, https://go-with-the-flow.github.io/) 做精修, 得到 refined video $\mathbf{V}_t$。

然后优化 VPP 参数让渲染结果对齐 $\mathbf{V}_t$。这里有两个关键设计:

### 公式 3: 总 loss

$$\mathcal{L} = \mathcal{L}_p(\text{Render}(\mathcal{B}_t) \odot (1-\mathbf{M}), \mathbf{V}_t \odot (1-\mathbf{M})) + \mathcal{L}_p(\text{Render}(\mathcal{G}_t), \mathbf{V}_t \odot \mathbf{M}) + \lambda_{\text{sim}} \mathcal{L}_{\text{sim}}$$

- $\mathbf{M}$: foreground binary mask
- $\text{Render}(\cdot)$: differentiable gaussian splatting
- $\mathbf{V}_t$: video model 在时刻 $t$ 给的 refined frame
- $\mathcal{L}_p$: photometric loss = L1 + SSIM
- $\odot$: element-wise masking

三项含义:
1. Background photometric: 处理 shadow, lighting 这类次级效果 (background gaussians 的 position 不动, 但 opacity, color 可学)
2. Foreground photometric: 主物体的 appearance 对齐
3. $\mathcal{L}_{\text{sim}}$: 下面专门讲

### 公式 4: Simulation consistency loss

$$\mathcal{L}_{\text{sim}} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| p_{j,t} - \frac{1}{K} \sum_{k=1}^{K} \mu_{j,k,t} \right\|_2^2$$

- $T$: time window 长度
- $J$: particle 总数
- $p_{j,t}$: 时刻 $t$ 第 $j$ 个 particle 的位置 (由 simulator 给, **不优化**, 是 anchor)
- $\mu_{j,k,t}$: 时刻 $t$ 第 $j$ 个 particle 锚定的第 $k$ 个 gaussian 的位置 (优化目标)
- $\frac{1}{K}\sum_k \mu_{j,k,t}$: K 个 gaussian 的质心

直觉: 这个 loss 强制 K 个 gaussian 的质心留在 particle $p_j$ 附近。注意 $p_j$ 本身不在 loss 里被优化, 它是 anchor reference。

配合公式 1 的 tanh bound, 这给 video model 留了"局部调整空间" (gaussian 可以围绕 particle 重新分布, 改变形状细节), 同时阻止"全局漂移" (整体不能跑离 physics 预测的位置)。

这两个公式合起来就是 VPP 的几何基础: **软绑定**。比起 PhysGaussian [48] (https://physgaussian.github.io/) 的 rigid binding (gaussian 直接等于 particle), VPP 允许 K > 1 和可学 offset, 给 video model 足够 visual expressiveness, 但物理 anchor 不丢。

### Loop closure (核心创新)

一个 time window 跑完, 要进入下一个 window 时:

```
Time window 0 ends at S_T
  ↓
Update p_j ← (1/K) Σ_k μ_{j,k} at time T
  ↓
v_j ← original v_j at time T (直接继承, 不修)
  ↓
S_T becomes S_0 for next window
  ↓
Apply new action A_0, run Φ_p again
```

把 G 的平均位置写回 P, 这就完成了 "video model 反向修正 physics" 的闭环。下一轮 physics 从修正过的 state 出发, drift 不会累积。

为什么 velocity 可以直接继承不修: 因为 $\mathcal{L}_{\text{sim}}$ 限制了 particle 位置更新范围很小, 所以 "velocity at T" 近似等于 "refined scene at T 的 velocity"。这是工程简化, 完整的 velocity 修正需要可微 physics, 这里用 position 反推 + velocity 直传避开。

---

## 为什么单视角优化不行

2D 视频 supervision 对应到 3D scene 是 **一对多** 的: 同一组 2D 像素可以对应无穷多个 3D 配置。单视角优化会 overfit 到那个视角, 换视角就崩。

WonderPlay 的 Imaging metric 只有 36.80 (Table 1), 就是这个问题的直接体现。

### 解决方案: 用 GEN3C [34] (https://research.nvidia.com/labs/toronto-ai/GEN3C/) 重建完整 3D scene

WonderPlay 用单视角 depth estimation + inpainting, 只能 narrow baseline 渲染。PerpetualWonder 改用 GEN3C 生成 dense surrounding views:

- GEN3C 原生要求从 input view 开始生成, 直接 180° 轨迹会 consistency 退化
- 拆成两条 90° 轨迹 ("arc left" + "arc right"), 都从 input view 出发
- 聚合 → 242 dense views

然后:
1. COLMAP [35] 算点云 → 初始化 3D Gaussians $\{G_i\}_{i=1}^N$
2. SAM2 [31] (https://github.com/facebookresearch/sam2) 分割 + Gaussian Grouping [52] (https://github.com/lkeab/gaussian-grouping) 分离 background / foreground
3. Foreground 用 TSDFusion [55] 转 mesh → 采样 physics particles $\mathcal{P}_0$
4. Rigid body 特殊处理: TSDFusion 背面质量差, 用 Hunyuan3D [61] (https://github.com/Tencent/Hunyuan3D-2) 单独生成 mesh, 再用 6-DoF pose + scale 优化放回 scene

### Progressive multi-view optimization

即便有了多视角, video model 在不同视角生成的视频本身 **互不一致** (它不知道这是同一场景的不同视角)。直接一起优化 → 表示层只能"妥协"到模糊解 → Figure 7 底部的 blurry texture + flicker。

三阶段策略:

1. **Stage 1**: 只从 input view 渲染 + refine → 优化 (单视角, 无冲突)
2. **Stage 2**: 从其他视角渲染, video model refine, 但 **smaller control weight** (弱监督, 不让冲突信息破坏已建立的表示)
3. **Stage 3**: 所有视角一起优化 → 此时 representation 已经有 stable baseline, multi-view 冲突的相对影响减小

直觉: warm-up 策略。先建一个 robust 的 single-view 表示作为 anchor, 再逐步引入 multi-view 信息。避免了在表示完全随机时直接面对 multi-view 冲突, 让优化陷入 local minima。

实验用 3 个 key views: frontal, left-side, right-side。

---

## 实验关键数据

### Table 1: World-Score [8] (https://worldscore.github.io/) 量化指标

| Method | Camera Ctrl | 3D Consist | Imaging |
|---|---|---|---|
| Wan2.2 [40] | 59.73 | 65.35 | 67.03 |
| GEN3C [34] | 80.29 | 61.69 | 66.25 |
| WonderPlay [21] | 75.95 | 63.93 | **36.80** ← 崩 |
| Veo3.1 [44] | 60.61 | 73.93 | 67.82 |
| **PerpetualWonder** | **93.26** | **80.41** | 66.98 |

观察:
- WonderPlay 的 Imaging 36.80 是 single-view optimization 导致 novel view artifacts 的直接证据
- PerpetualWonder 在 Camera Ctrl 和 3D Consist 上大幅领先 (>12 分 vs 第二名)
- Imaging 上没赢 Veo3.1 — 合理, video generator 在单帧质量上有 data 规模优势, 3D 表示渲染会损失一些 photorealism

### Table 2: 2AFC human study (350 人)

| 对比对象 | Physics Plausibility | Motion Fidelity |
|---|---|---|
| over Wan2.2 | 74.1% | 71.8% |
| over GEN3C | 93.5% | 83.5% |
| over WonderPlay | 80.8% | 86.3% |
| over Veo3.1 | 62.0% | 70.8% |

- 70-90% 用户偏好 PerpetualWonder
- vs GEN3C 93.5% 最极端, 因为 GEN3C 完全 ignore text prompt 里的 action 描述
- vs Veo3.1 优势最小, Veo3.1 (https://deepmind.google/models/veo/) 生成质量极高, 物理细节也学了不少

### Ablation 亮点

**VPP vs standard 3DGS (Figure 6)**: 同样 multi-view optimization, 用 VPP 物体随 physics 动且视觉一致; 用 standard 3DGS, gaussian 自由优化只 minimize photometric loss → chaotic dynamics, 视觉 artifacts。没有 VPP 的 anchor, gaussian 就是一群"无物理的彩色点"。

**Progressive vs Direct multi-view (Figure 7)**: Progressive 苹果纹理清晰时间稳定; Direct 出现 blurry texture + flicker。这是 multi-view 冲突的代价 — 一起优化时表示层只能妥协到模糊解。

**Isotropic vs Anisotropic (Figure S1)**: Isotropic (球形) gaussian 在 novel view 下更鲁棒, anisotropic 容易过拟合 input view 的特定形状, 在新视角下产生 stretch artifacts。

**Particle radius (Figure S2)**: $[0.25\delta, 4\delta]$ 范围 robust; 太小 ($\leq 0.01\delta$) 表示能力不足; 太大 ($\geq 100\delta$) 优化不稳定。说明 VPP 对 $\delta$ 具体值不敏感, 但需要在合理量级。

### Runtime (Table S2)

| Stage | Init | Forward | Backward | Total (1st loop) |
|---|---|---|---|---|
| Time | ~8 min | <1 min | ~7 min | ~16 min |

非实时, backward optimization 是瓶颈 (3 视角 × video inference + gaussian optimization)。

---

## 跟相关工作的关系

### Hybrid generative simulator 谱系

```
PhysMotion [38] (https://arxiv.org/abs/2411.17189)
  → PhysDreamer [58] (https://physdreamer.github.io/)
    → PhysGaussian [48] (https://physgaussian.github.io/)
      → WonderPlay [21] (https://WonderPlay.github.io/)
        → PerpetualWonder (本文)
```

WonderPlay 是直接前作, 同一作者组。PhysGaussian 也用 gaussian + physics, 但 rigid binding, 无 video refinement feedback, 无 long-horizon。

### Pure video generation 路线

- **Sora [4]**, **Veo3.1 [44]**, **Wan [40]**, **CogVideoX [50]**, **Open-Sora [62]**: 纯 video model, 无 explicit 3D / physics 结构。Long-horizon 下也会 drift, 因为没有 explicit state correction
- **Force Prompting [11]** (https://force-prompting.github.io/): 2D force vector 控制 video gen, 无 explicit 3D rep

PerpetualWonder 的路线是 hybrid: physics 提供 structure, video model 提供 realism, closed-loop 让两者互相纠正。

### Scene reconstruction 组件

- **GEN3C [34]**: NVIDIA, 3D-informed video gen with camera control
- **WonderWorld [54]** (https://wonderworld-2024.github.io/), **WonderJourney [53]**: 同组 scene 生成前作
- **Voyager [15]** (https://voyager-video.github.io/): 同组 long-range video diffusion for explorable 3D scene
- **World-Score [8]** (https://worldscore.github.io/): 同组 evaluation benchmark

---

## 深层直觉

### 为什么 closed-loop 在 4D generation 中是 fundamental 的

类比 RL: state → action → next state, 你必须维护一致的 state 表示。如果只在 state 上做 supervised learning 而 action 后的 next state 不反馈到 state representation, 你的 policy 就会 drift。

4D generation 的 long-horizon 同理。Video model 是一个强大的 "observation model", 给的是 noisy 但信息丰富的 measurement。你需要 **Bayesian filter-like** 结构: prediction (physics) → update (video refinement) → prediction → update...

VPP 就是这个结构下的 state representation, 支持:
- **Prediction step**: physics solver 推 P → 通过公式 1 推 G
- **Update step**: video model 提供 measurement → optimize G → 通过 $\mathcal{L}_{\text{sim}}$ 间接 constrain P → loop closure 把 G 平均位置写回 P

"first true closed-loop system" 这句话的分量在于: 之前的 hybrid system 只有 prediction, 没有 update feedback 到 state。

### VPP 与 PhysGaussian 的本质区别

PhysGaussian 把 gaussian 直接等于 particle, rigid binding。VPP 是 **soft binding**:
- 允许 K > 1 (一个 particle 挂多个 gaussian)
- 允许 offset $\tilde{p}_{j,k}$ 可学 (gaussian 可在 particle 邻域微调)
- 允许 temporal opacity (gaussian 可在时间上"出现-消失")

这让 video model 有足够 visual expressiveness, 同时物理 anchor 不丢。这个 K-offset-opacity 的组合是 VPP 的真正贡献。

### Progressive multi-view 的 Bayesian 诠释

直接 multi-view optimization 失败的本质: video model 在不同视角给的 prediction 是 marginal distribution 而非 joint distribution。它不知道正面和侧面是同一物体的两面, 给的两个 video 是 inconsistent samples。

Progressive strategy 的本质: 先用 single-view 建立 prior over 3D scene, 再用其他视角作为 likelihood 更新。避免了从零先验开始时 multi-view 冲突的"先有鸡还是先有蛋"问题。

### Background 建模的巧妙

Background $\mathcal{B}_t$ 也用 spatial + temporal opacity, 但 position 不变 (不参与 physics)。允许 shadow, lighting 随时间变化 (物体移动产生的 shadow 移动), 通过 $\mathcal{L}_p$ 在 background mask 外对齐到 video model 输出。

轻巧的设计: 不需要显式 shadow modeling, 让 video model 通过 refinement 学到 shadow, 再通过 background gaussian 的 temporal opacity 表达出来。

---

## 局限

### Paper 明确提到

- **Runtime**: 16 min 首 loop, 非实时
- **Unseen geometry** (Figure S3): hockey stick 从画面外进入, 背面 unseen geometry 补不全, Hunyuan3D 也无能为力。单视角 reconstruction 的本质限制

### 推测的潜在局限

- **Action representation 粒度**: global force field 和 local force 是显式 given 的。从抽象 instruction ("轻轻拍一下苹果") 转换到 force vector 需要 VLM (GPT-4o [16]) 协助, 粒度可能不准
- **Material 参数估计**: GPT-4o 估 Young's modulus, Poisson ratio 误差可能很大, 实际可能需要 trial-and-error
- **Velocity 直传简化**: 高速运动场景 (物体反弹) 可能累积误差。完整方案需要可微 physics 或从 video optical flow 反推 velocity
- **Progressive 策略启发式**: 3 个 view 是 fixed, control weight 递减是 hand-tuned。更原则性的方案是用 video model 的 uncertainty 加权不同 view 的 supervision
- **Foreground 分割依赖 SAM2 + Gaussian Grouping**: 遮挡、复杂 material 边界 (fluid 与 background 颜色相近) 时可能失败

### 与未来 world model 的关系

Sora [4] 提出的 "video models as world simulators" 概念, 纯 video model 缺 explicit 3D / physics 结构, long-horizon 下也会 drift。PerpetualWonder 的 hybrid 路线 + closed-loop 思想可以推广: 任何 long-horizon generative simulator 都需要一个 state representation 能被 observation (video model) 反向更新。

VPP 给出的 pattern — 软绑定 anchor + consistency loss + progressive multi-view update — 可能在其他 modality (audio, tactile) 的 closed-loop 生成中也适用。

---

## Reference links

- 项目主页: https://johnzhan2023.github.io/PerpetualWonder/
- World-Score: https://worldscore.github.io/
- GEN3C: https://research.nvidia.com/labs/toronto-ai/GEN3C/
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- SAM2: https://github.com/facebookresearch/sam2
- Hunyuan3D: https://github.com/Tencent/Hunyuan3D-2
- Genesis physics engine: https://github.com/Genesis-Embodied-AI/Genesis
- Go-with-the-flow: https://go-with-the-flow.github.io/
- WonderPlay (前作): https://WonderPlay.github.io/
- WonderWorld: https://wonderworld-2024.github.io/
- PhysGaussian: https://physgaussian.github.io/
- PhysDreamer: https://physdreamer.github.io/
- PhysGen3D: https://physgen3d.github.io/
- Force Prompting: https://force-prompting.github.io/
- Voyager: https://voyager-video.github.io/
- Gaussian Grouping: https://github.com/lkeab/gaussian-grouping
- FreeTimeGS: https://arxiv.org/abs/2502.24608
- Veo3.1: https://deepmind.google/models/veo/

---

# PerpetualWonder 深度解析: Long-Horizon Action-Conditioned 4D Scene Generation

这是一篇来自 Stanford (Jiahao Zhan, Zizhang Li, Hong-Xing Yu, Jiajun Wu) 的工作, 直接延续了同一组之前的 WonderPlay (CVPR 2025) [21] 的路线。核心要解决的问题是: **如何让 hybrid generative simulator (物理仿真器 + video diffusion model) 支持长时序、多步的 action-conditioned 交互**。项目主页: https://johnzhan2023.github.io/PerpetualWonder/

---

## 1. 问题动机: 为什么 WonderPlay 在 long-horizon 必然崩溃

### 1.1 Task formulation

输入: 单张图像 $I$ + 一串用户动作 $\{\mathcal{A}_t\}_{t=0}^{T-1}$
动作形式:
- global force field $\mathbf{f}(x, y, z, t)$ (gravity, wind)
- local force $\mathbf{f}(t)$ (推、戳等)

输出: 动态 4D scene 序列 $\{\mathcal{S}_t\}_{t=0}^{T}$, 每个时刻 $\mathcal{S}_t = (\mathcal{B}_t, \mathcal{F}_t)$ (background + foreground objects)

关键: $\{\mathcal{A}_t\}$ 是 sequential 的, 即第一个 window 跑完后再执行第二个 window 的动作, 这要求系统必须**记住上一个 window 结束时的物理状态**。

### 1.2 WonderPlay 的 decoupled representation 致命缺陷

WonderPlay 用 physics particles $\mathcal{P}$ 驱动 visual primitives (3D Gaussians) $\mathcal{G}$, 但 **信息流是单向的**:

```
Physics solver → particles P → drives → Gaussians G
                                          ↓
                          video model refines G (only G)
                                          ↓
                                    refined G
                          (P stays frozen, never updated!)
```

下一轮动作来了, P 仍然是上一轮物理仿真结束时的"原始"P, 而 G 已经被 video model 改过了。于是出现 **drift**: 比如论文 Figure 5 的 shovel, 第一轮被甩到空中旋转, G 跟着 video refinement 走, 但 P 还在原始位置附近 → 第二轮 shovel 插 sand castle 时, G 严重撕裂, 物理上完全失效。

这个问题的本质: **video diffusion model 是一个先验概率修正器, 它会把 appearance 和 dynamics 一起改, 但你只允许它改 appearance (G), 不允许它改 dynamics (P)**。在 long-horizon 下, 这个 discrepancy 累积 → 系统崩溃。

PerpetualWonder 的核心洞察: 要做 closed-loop, 必须让 video model 的 refinement 同时能反向更新 P。这就需要一个 unified representation 把 P 和 G **bidirectionally** 绑定起来。

---

## 2. 核心创新 1: Visual-Physical Aligned Particle (VPP)

### 2.1 Representation 结构

每个 foreground object 由 $\mathcal{F} = \{\mathcal{P}^o, \mathcal{V}^o, \mathcal{G}^o\}_{o=1}^{O}$ 表示:
- $\mathcal{P} = \{p_j\}_{j=1}^{J}$: J 个 physics particles 的位置
- $\mathcal{V} = \{v_j\}_{j=1}^{J}$: 对应速度
- $\mathcal{G} = \bigcup_{j=1}^{J} \{g_{j,k}\}_{k=1}^{K}$: 每个 particle $p_j$ 锚定 K 个 gaussians

VPP 的关键: gaussians 不是自由漂浮的, 它们的位置必须**依附**于 particle $p_j$。

### 2.2 Gaussian 参数化 (公式 1, 2 详解)

**Position (Eq. 1):**
$$\mu_{j,k} = p_j + \tanh(\tilde{p}_{j,k}) \cdot \delta$$

变量解释:
- $\mu_{j,k}$: 第 $j$ 个 particle 锚定的第 $k$ 个 gaussian 的 3D 中心位置
- $p_j \in \mathbb{R}^3$: 第 $j$ 个 physics particle 的位置 (由 simulator 更新)
- $\tilde{p}_{j,k} \in \mathbb{R}^3$: **learnable** position offset, 是网络要优化的参数
- $\delta$: physics particle 的采样尺寸 (在 simulator 中固定)
- $\tanh(\cdot)$: 把 offset 限制在 $(-1, 1)$ 范围, 乘 $\delta$ 后 offset 范围是 $(-\delta, \delta)$, 即 gaussian 不能离开 particle 太远

**Intuition**: 这个设计非常重要。tanh 是一个 saturating function, 它确保 gaussian 即使被 video model 优化, 也只能在一个 particle-size 的邻域内微调位置。这就给了 $\mathcal{L}_{\text{sim}}$ 一个软约束的几何基础 — 你可以乱漂, 但漂不远。

**Temporal opacity (Eq. 2):**
$$o_t(t) = \exp\left(-\frac{1}{2} \left(\frac{t - \mu_t}{s_d}\right)^2\right)$$

变量解释:
- $o_t(t)$: 时刻 $t$ 的 temporal opacity
- $\mu_t$: gaussian "出现"的中心时刻 (learnable)
- $s_d$: temporal duration (learnable)
- 形式是一个 Gaussian 函数 over time

**最终 opacity**: $o(t) = o_s \cdot o_t(t)$, $o_s$ 是标准 3DGS 的 spatial opacity。

**Intuition**: 这个设计借自 FreeTimeGS [42]。它允许一个 gaussian 在时间轴上"出现-消失", 而不是永远存在。对 smoke, splash 这类次级视觉效果特别关键 — 你不希望每一帧所有 particle 的所有 gaussian 都参与渲染, 否则烟雾会变成一团永远不散的雾。$\mu_t$ 和 $s_d$ 可学, 意味着 video model 可以通过 refinement 告诉某些 gaussian "你只在 t=15 那一帧附近出现一下"。

**Scale**: isotropic (球状), $\leq \delta$。这一点 Figure S1 有 ablation: isotropic 比 anisotropic 在 novel view 下更鲁棒, 因为 anisotropic 容易过拟合 input view 的形状。

### 2.3 VPP 的 bidirectional bridge 工作机制

**Forward pass (physics → visuals)**:
- Physics solver 更新 $p_j \to p_j'$
- 由 Eq. 1, 所有 $\mu_{j,k}$ 自动跟随平移 (因为 $\tilde{p}_{j,k}$ 是相对 offset, 不变)
- Visual appearance 被 physics 驱动 ✓

**Backward pass (visuals → physics)**:
- Video model 提供 multi-view supervision
- 优化 $\tilde{p}_{j,k}, o_s, \mu_t, s_d, q, c$ 等 gaussian 属性
- 通过 $\mathcal{L}_{\text{sim}}$ 约束 gaussian 中心 $\mu_{j,k}$ 不能偏离 $p_j$ 太远
- Loop closure 时: $p_j^{new} = \frac{1}{K}\sum_k \mu_{j,k}$ (平均 K 个 gaussian 位置更新 particle 位置)

**这就是真正的 closed-loop**: physics 告诉 visual 该怎么动, visual 告诉 physics 该怎么修正。

### 2.4 K 值的 adaptive 配置 (Supp. D)

这是一个很有工程美感的细节, paper 的 Supp. D 给了具体规则:

| Material | K | Gaussian scale |
|---|---|---|
| Rigid body, Cloth (surface) | 1 | = $\delta$ |
| Gas, Liquid, Sand, Snow, Elastic (volumetric) | 20 | = $0.5\delta$ |

**Intuition**: 
- Surface material: 严格一对一, 防止 large deformation 时的 ghosting/detachment
- Volumetric material: 一个 particle 需要"撑起"一片体积, 所以挂 20 个小 gaussian, scale 缩小到 $0.5\delta$ 来表达细粒度 volumetric detail

这种 material-aware 配置比 universal K 简洁很多, 也避免了在 rigid body 上挂 20 个 gaussian 时的优化不稳定。

---

## 3. 核心创新 2: Multi-View Optimization

### 3.1 为什么单视角优化不行

WonderPlay 用单视角 video refinement 直接优化 3D scene。问题: video model 给出的 2D supervision 是单视角的, 而 3D scene 的参数是高维的 — **同一个 2D 像素值可以对应无穷多个 3D 配置**。

举例: 一片 sand 被 wind 吹起, 正面看是一个 dust cloud, 但 sand particles 的具体 3D 分布是 ambiguous 的。单视角优化会 overfit 到正面那个 appearance, 一旦换视角就撕裂。

### 3.2 3D Scene Initialization (替代 WonderPlay 的单视角 depth unprojection)

WonderPlay 用 depth estimation + inpainting 从单视角生成 scene, 导致只能在一个 narrow baseline 内渲染新视角。PerpetualWonder 改用 **GEN3C [34]** 生成 dense surrounding views:

策略 (Supp. A):
- GEN3C 原生要求从 input view 开始生成, 直接生成 180° 轨迹会出现 consistency 退化
- 解决: 拆成两条 90° 轨迹 ("arc left" + "arc right"), 都从 input view 出发, 各转 90°
- 聚合所有帧 → 242 dense views (实验细节)

然后:
1. COLMAP [35] 算点云 → 初始化 3D Gaussians $\{G_i\}_{i=1}^{N}$
2. SAM2 [31] 在 dense views 上分割 → 监督 Gaussian Grouping [52] 的 learnable feature $g_i$
3. 分离 background $B_0$ + foreground objects
4. Foreground 用 **TSDFusion [55]** 转 mesh → 采样 physics particles $\mathcal{P}_0$
5. 用 GEN3C 帧再次优化 foreground VPP gaussians

**对于 rigid body 的特殊处理 (Supp. B)**: TSDFusion 在不可见区域生成质量差 → 用 Hunyuan3D [61] 单独生成 object mesh, 再通过 6-DoF pose + scale 优化 (minimize projection error vs. dense views) 放回 scene。这是为什么 Figure S3 的 hockey stick (unseen long geometry) 仍然失败 — Hunyuan3D 也只能从 input view 推断, 看不见的部分还是补不全。

### 3.3 Loss function (公式 3, 4 详解)

**总体 loss (Eq. 3):**
$$\mathcal{L} = \mathcal{L}_p(\text{Render}(\mathcal{B}_t) \odot (1-\mathbf{M}), \mathbf{V}_t \odot (1-\mathbf{M})) + \mathcal{L}_p(\text{Render}(\mathcal{G}_t), \mathbf{V}_t \odot \mathbf{M}) + \lambda_{\text{sim}} \mathcal{L}_{\text{sim}}$$

变量解释:
- $\mathbf{M}$: foreground binary mask
- $\text{Render}(\cdot)$: differentiable gaussian splatting
- $\mathbf{V}_t$: refined video from video model at time $t$
- $\mathcal{L}_p$: photometric loss = L1 + SSIM
- $\odot$: element-wise masking

三项含义:
1. **Background photometric**: background 渲染图 (在 mask 外) 与 video 背景部分对齐 — 处理 secondary effects 如 shadow
2. **Foreground photometric**: foreground VPP 渲染图 (在 mask 内) 与 video 前景对齐
3. **$\mathcal{L}_{\text{sim}}$**: simulation consistency

Background 也用 spatial + temporal opacity 建模 (除 position 外都可学), 这样 shadow 能随时间变化。

**Simulation consistency loss (Eq. 4):**
$$\mathcal{L}_{\text{sim}} = \frac{1}{T \cdot J} \sum_{t=1}^{T} \sum_{j=1}^{J} \left\| p_{j,t} - \frac{1}{K} \sum_{k=1}^{K} \mu_{j,k,t} \right\|_2^2$$

变量解释:
- $T$: time window 长度
- $J$: physics particle 数量
- $p_{j,t}$: 时刻 $t$, particle $j$ 的位置 (由 simulator 给出, **不是直接优化目标**, 而是作为 reference)
- $\mu_{j,k,t}$: 时刻 $t$, particle $j$ 锚定的第 $k$ 个 gaussian 的位置 (可优化)
- $\frac{1}{K}\sum_k \mu_{j,k,t}$: 该 particle 锚定的 K 个 gaussian 的平均中心

**Intuition**: 这个 loss 把 K 个 gaussian 的质心拉回 particle $p_j$。注意 $p_j$ 本身不在 loss 内被优化, 它是 anchor。这意味着:
- Forward pass 时, $p_j$ 被 physics 更新 → $\mu_{j,k}$ 跟着移动
- Backward pass 时, video model 试图把 $\mu_{j,k}$ 拉到能产生好图像的位置, 但 $\mathcal{L}_{\text{sim}}$ 强制 $\mu_{j,k}$ 质心留在 $p_j$ 附近
- 所以 gaussian 可以围绕 particle 重新分布 (改变 offset), 但整体不能漂走

这是 VPP 的几何基础: 它给 video model 留了"局部调整空间", 同时阻止"全局漂移"。

### 3.4 Progressive multi-view optimization

直接同时用 3 个视角的 video 监督会冲突 — video model 在不同视角生成的视频互不一致 (例如正面看到苹果有反光斑, 侧面没有)。直接优化 → 表示 corrupt → blurry texture + flicker (Figure 7 bottom)。

PerpetualWonder 的三阶段策略:

**Stage 1**: 只从 input view 渲染 + refine → 优化 (单视角, 没有冲突)
**Stage 2**: 从其他视角渲染, 用 video model refine, 但 **smaller control weight** (弱监督, 不让冲突信息破坏已建立的表示)
**Stage 3**: 用所有视角的 refined video 一起优化 → 此时 representation 已经有了一个 stable baseline, multi-view 冲突的相对影响减小

**Intuition**: 这是一个 warm-up 策略。先建一个 robust 的 single-view 表示作为 anchor, 再逐步引入 multi-view 信息。这避免了在表示完全随机时直接面对 multi-view 冲突, 让优化陷入 local minima。

实验用 3 个 key views: frontal, left-side, right-side (Sec. 4 实现细节)。

---

## 4. Simulation Loop: 三个 stage 的闭环

### 4.1 Forward physics pass $\Phi_p$

$$\hat{\mathcal{S}}_{t+1} = \Phi_p(\hat{\mathcal{S}}_t, \mathcal{A}_t), \quad t = 0, \ldots, T-1$$

- 用 Genesis [19] 作为 physics simulator (注意: ref 19 的 "Genesis" 引文写的是 NLP 论文, 应该是另一个 Genesis 物理仿真器, 这是引用错误, 实际他们用的应该是 Genesis — a generative and general-purpose physics engine, https://github.com/Genesis-Embodied-AI/Genesis)
- 多种 solver: MPM (Material Point Method, [17]), PBD (Position Based Dynamics, [25]), Shape Matching [24] 等
- 每个 window: 392 physics steps, 每 8 步采样 1 帧到 video → 49 frames 给 video model

### 4.2 Backward optimization pass $\Psi_n$

输入: coarse 序列 $\{\hat{\mathcal{S}}_t\}$
- Render 成 RGB + optical flow (bimodal control 给 video model)
- Video model (CogVideoX [50] with warped-noise control [5]) 生成 refined video $\mathbf{V}_t$
- 用 Sec. 3.3 的 progressive multi-view optimization 优化 VPP
- 输出: refined 序列 $\{\mathcal{S}_t\}$

### 4.3 Loop closure (核心创新)

```
Time window 0 ends at S_T
  ↓
Update P_T from optimized G_T:  p_j ← (1/K) Σ_k μ_{j,k} at time T
  ↓
v_j ← original v_j at time T (directly inherited)
  ↓
S_T becomes S_0 for next window
  ↓
Apply new action A_0, run Φ_p again
```

**为什么 velocity 可以直接继承而不被 video refinement 更新?** Paper 解释: 因为 $\mathcal{L}_{\text{sim}}$ 限制了 particle 位置更新的范围很小, 所以"velocity at T" 近似于"refined scene at T 的 velocity"。这是一个工程上的折中 — 完整的 velocity 修正需要更复杂的可微物理, 这里用 position 反推 + velocity 直传的简化策略。

### 4.4 物理 simulation 参数 (Supp. C, Table S1)

通用:
- Step time: $1e^{-3}$ s
- Sub-steps: 10
- Particle size: $1e^{-2}$

各 material:
- Gravity: $(0, 0, -9.8)$
- Friction coefficient: 0.1
- Grid density (MPM): 64
- Elastic: Young $3e^5$, Poisson 0.2
- Liquid: Young $1e^7$, Poisson 0.2
- Granular: Young $1e^6$, Poisson 0.2, Friction angle 0.2

这些参数由 Vision-Language Model (GPT-4o [16]) 从 input image 估计, 可选 manual fine-tuning。这延续了 WonderPlay 的做法。

---

## 5. 实验详解

### 5.1 Dataset & Metrics

- **10 scenes**: 涵盖 cloth, rigid body, elastic, liquid, gas, granular
- **Scene quality**: World-Score [8] (https://worldscore.github.io/) 的三个子指标
  - Camera Ctrl: 摄像机轨迹可控性
  - 3D Consist: 3D 一致性
  - Imaging: 单帧图像质量
- **Physical dynamics**: 350 人 2AFC human study
  - Physics Plausibility: 动作响应正确性
  - Motion Fidelity: 动作自然度

### 5.2 Baselines

两类:
- **Conditional video generators**: Wan2.2 [40], Wan2.6, Veo3.1 [44], Tora [59], DaS [12], GEN3C [34]
- **Hybrid generative simulator**: WonderPlay [21], **WonderPlay++** (作者构造的强基线: 用 PerpetualWonder 的 multi-view 3D reconstruction + WonderPlay 的 decoupled representation + single-view optimization)

WonderPlay++ 这个 baseline 设计很关键 — 它隔离了 "multi-view reconstruction" 和 "VPP + closed-loop" 两个因素的贡献。Table 1 显示 PerpetualWonder > WonderPlay++, 证明 closed-loop VPP 才是真正的核心。

### 5.3 Quantitative results (Table 1)

| Method | Camera Ctrl | 3D Consist | Imaging |
|---|---|---|---|
| Wan2.2 | 59.73 | 65.35 | 67.03 |
| GEN3C | 80.29 | 61.69 | 66.25 |
| WonderPlay | 75.95 | 63.93 | **36.80** ← 崩 |
| Veo3.1 | 60.61 | 73.93 | 67.82 |
| **PerpetualWonder** | **93.26** | **80.41** | 66.98 |

关键观察:
- WonderPlay 的 Imaging 只有 36.80 — 这反映了 single-view optimization 导致的 novel view artifacts
- PerpetualWonder 在 Camera Ctrl 和 3D Consist 上大幅领先 (>12 分 vs 第二名)
- Imaging 上没有赢 Veo3.1 — 这是合理的, 因为 video generator 本身在单帧质量上有 data 规模优势, 而 PerpetualWonder 通过 3D 表示渲染会损失一些 photorealism

### 5.4 Human study (Table 2)

| 对比对象 | Physics Plausibility | Motion Fidelity |
|---|---|---|
| over Wan2.2 | 74.1% | 71.8% |
| over GEN3C | 93.5% | 83.5% |
| over WonderPlay | 80.8% | 86.3% |
| over Veo3.1 | 62.0% | 70.8% |
| over Wan2.6 | 68.5% | 77.3% |
| over Tora | 83.5% | 85.3% |
| over DaS | 80.9% | 81.9% |

- 70-90% 的用户偏好 PerpetualWonder
- vs GEN3C 的 93.5% 最极端, 因为 GEN3C 完全 ignore actions (text prompt 描述的 action 对它无效)
- vs Veo3.1 优势最小 (62%, 70.8%), 因为 Veo3.1 (https://deepmind.google/models/veo/) 生成质量极高, 物理细节也学习了不少

### 5.5 Long-horizon 对比 (Figure 5)

城堡场景四轮交互:
- 第一轮: shovel 旋转
- 第二轮: shovel 插入城堡
- ...

WonderPlay 在第二轮就出现严重撕裂 (shovel 形状破坏, 因为 G 被 refine 了但 P 还在原位置), 而 PerpetualWonder 在四轮后仍保持形状完整。这是 closed-loop 的直接体现。

### 5.6 Ablation: VPP vs standard 3DGS (Figure 6)

用相同 multi-view optimization, 分别用:
- VPP (top): 物体随 physics 动, 视觉一致
- standard 3DGS (bottom): gaussian 自由优化, 只 minimize photometric loss → chaotic dynamics, 视觉 artifacts

**Intuition**: 没有 VPP 的 anchor, gaussian 就是一群"无物理的彩色点", 视频监督告诉它们怎么动就怎么动, 完全无视物理。这验证了 VPP 的 anchor 机制是必要的。

### 5.7 Ablation: Progressive vs Direct multi-view optimization (Figure 7)

- Progressive (top): 苹果纹理清晰, 时间稳定
- Direct (bottom): 苹果出现 blurry texture + flicker

**Intuition**: 这就是 multi-view 冲突的代价。Video model 在不同视角生成的视频本身不一致 (它不知道这是同一场景的不同视角), 一起优化时表示层只能"妥协"到模糊解。Progressive 让单视角先建立稳定表示, 再逐步加 multi-view, 让冲突的相对能量降低。

### 5.8 Isotropic vs Anisotropic primitives (Figure S1)

Isotropic (球形) gaussian 在 novel view 下更鲁棒 — anisotropic 容易过拟合 input view 的特定形状, 在新视角下产生 stretch artifacts。

### 5.9 Particle radius ablation (Figure S2)

- 合理范围 $[0.25\delta, 4\delta]$: robust
- 太小 ($\leq 0.01\delta$): 表示能力不足
- 太大 ($\geq 100\delta$): 优化不稳定

这说明 VPP 对 $\delta$ 的具体值不敏感, 但需要在一个合理量级。

### 5.10 Runtime (Table S2)

| Stage | Initialization | Forward Pass | Backward Opt. | Total (1st Loop) |
|---|---|---|---|---|
| Time | ~8 min | <1 min | ~7 min | ~16 min |

非实时。后续 loop 应该更短 (跳过 initialization), 但 backward optimization 仍是瓶颈。

### 5.11 Failure case (Figure S3)

Hockey stick 从画面外进入, 在中间帧时由于背面 unseen geometry 补不全, stick 显得短。Hunyuan3D 也只能从 input view 推断, 这是单视角 reconstruction 的本质限制。

---

## 6. 与相关工作的关系图谱

### 6.1 Hybrid generative simulator 谱系

```
PhysMotion [38] (single image → physics-grounded dynamics, video prior)
    ↓
PhysDreamer [58] (physics-based interaction with 3D objects via video gen)
    ↓
PhysGaussian [48] (physics-integrated 3D gaussians)
    ↓
WonderPlay [21] (single image + actions → dynamic 3D scene, decoupled rep)
    ↓
PerpetualWonder (this work: closed-loop, long-horizon, VPP unified rep)
```

### 6.2 单一表示路线对比

- **PhysGaussian [48]** (https://physgaussian.github.io/): 也用 gaussian + physics, 但只做 single interaction, 不支持 video refinement feedback 到 physics
- **4D Gaussian Splatting [45, 49, 51]**: 纯 reconstruction, 不支持 action
- **DreamPhysics [14]**: 用 video diffusion prior 学 physics-based 3D dynamics, 但没有 explicit action conditioning
- **PhysGen3D [6]** (https://physgen3d.github.io/): 从 single image 造 miniature interactive world, 也是纯 physics 路线

### 6.3 Video generation 控制方法

- **Camera trajectory control**: RecamMaster [3], Cinemaster [41], Stable Virtual Camera [63], GEN3C [34], CAT4D [46]
- **2D motion/trajectory control**: MotionPrompting [10], DragAnything [47], MoFA-Video [26], Motion-I2V [36], Tora [59]
- **Force conditioning**: Force Prompting [11] (https://force-prompting.github.io/) — 2D force vector, 无 explicit 3D rep

PerpetualWonder 的区别: 在 **完整 3D 表示** 上操作, 而非 2D pixel-space。这让 multi-view supervision 成为可能。

### 6.4 Long-range video generation

- **Voyager [15]** (https://voyager-video.github.io/): long-range world-consistent video diffusion for explorable 3D scene generation — 同一作者组, 但 focus 在 scene exploration 而非 action interaction
- **HunyuanWorld 1.0 [39]**: 从 text/pixels 生成 immersive 3D worlds
- **Sora [4]**, **Veo3.1 [44]**, **Wan [40]**, **CogVideoX [50]**, **Step-Video [13]**, **Open-Sora [62]**: pure video generation, 无 explicit 3D

### 6.5 评估

- **World-Score [8]** (https://worldscore.github.io/): unified evaluation benchmark for world generation, 由同一 Stanford 组提出, 包含 camera controllability, 3D consistency, imaging 等子指标
- **WonderWorld [54]** / **WonderJourney [53]**: 单视角 3D scene generation 的前作, 同一作者组的 scene reconstruction 路线

### 6.6 Reconstruction 组件

- **GEN3C [34]** (https://research.nvidia.com/labs/toronto-ai/GEN3C/): NVIDIA 的 3D-informed video generation with precise camera control
- **SAM2 [31]** (https://github.com/facebookresearch/sam2): Meta 的 video segmentation
- **Gaussian Grouping [52]** (https://github.com/lkeab/gaussian-grouping): 3D scene 分割 + 编辑
- **TSDFusion [55]**: 经典 TSDF fusion (Curless & Levoy 1996)
- **Hunyuan3D 2.0 [61]** (https://github.com/Tencent/Hunyuan3D-2): Tencent 的 image-to-3D
- **COLMAP [35]**: SfM 经典
- **3DGS [18]** (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/): INRIA 的 gaussian splatting
- **FreeTimeGS [42]**: temporal opacity gaussian 的来源
- **Genesis [19]**: 物理仿真器 (https://github.com/Genesis-Embodied-AI/Genesis)
- **Go-with-the-flow [5]** (https://go-with-the-flow.github.io/): warped noise video control, CogVideoX 的 conditioning 方法
- **GPT-4o [16]**: 物理参数估计

---

## 7. 方法论的深层直觉

### 7.1 为什么 closed-loop 在 4D generation 中是 fundamental 的

类比 RL: 在 RL 中, state → action → next state, 你必须维护一个一致的 state 表示。如果你只在 state 上做 supervised learning 而 action 后的 next state 不反馈到 state representation, 你的 policy 就会 drift。

4D generation 的 long-horizon 也是同样问题。Video model 是一个强大的"observation model", 它给的是 noisy 但信息丰富的 measurement。你需要一个 **Bayesian filter-like 的结构**: prediction (physics) → update (video refinement) → prediction → update...

VPP 就是这个结构下的 state representation, 它支持:
- **Prediction step**: physics solver 推 P → 通过 Eq. 1 推 G
- **Update step**: video model 提供 measurement → optimize G → 通过 $\mathcal{L}_{\text{sim}}$ 间接 constrain P → loop closure 把 G 的平均位置写回 P

这就是为什么 paper 说 "first true closed-loop system" — 之前的 hybrid system 只有 prediction, 没有 update feedback 到 state。

### 7.2 VPP 与 PhysGaussian 的对比

PhysGaussian 也把 gaussian 绑到 physics particle, 但它的绑定是 rigid 的 (gaussian 直接等于 particle)。PerpetualWonder 的 VPP 是 **soft binding**:
- 允许 K > 1 (一个 particle 挂多个 gaussian)
- 允许 offset $\tilde{p}_{j,k}$ 可学 (gaussian 可在 particle 邻域微调)
- 允许 temporal opacity (gaussian 可在时间上"出现-消失")

这让 video model 有足够的 visual expressiveness, 同时物理 anchor 不丢。这个 K-offset-opacity 的组合是 VPP 的真正贡献。

### 7.3 Progressive multi-view 与 Bayesian 优化的类比

直接 multi-view optimization 失败的本质: video model 在不同视角给出的 prediction 是 marginal distribution 而非 joint distribution。它不知道正面和侧面是同一物体的两面, 所以给出的两个 video 是 inconsistent samples。

Progressive strategy 的本质: 先用 single-view建立一个 prior over 3D scene, 再用其他视角作为 likelihood 更新。这避免了从零先验开始时 multi-view 冲突的"先有鸡还是先有蛋"问题。

### 7.4 Bimodal control (RGB + optical flow)

Paper 在 Sec. 3.2 提到 "bimodal control scheme"。这是 WonderPlay 已有的设计: video model 同时接收 RGB 和 optical flow 两路 conditioning。RGB 提供 appearance prior, optical flow 提供 motion prior。PerpetualWonder 沿用这个, 用 Go-with-the-flow [5] 的 warped noise 实现 — noise field 在 sampling 过程中被 optical flow warp, 自然产生 follow-motion 的视频。

### 7.5 Background 建模的细节

Background $\mathcal{B}_t$ 也用 spatial + temporal opacity, 但 position 不变 (因为 background 不参与 physics)。这允许 background 的 shadow, lighting 随时间变化 (例如物体移动产生的 shadow 移动), 通过 $\mathcal{L}_p$ 在 background mask 外对齐到 video model 的输出。

这是一个轻巧的设计: 不需要显式 shadow modeling, 让 video model 通过 refinement 学到 shadow, 再通过 background gaussian 的 temporal opacity 表达出来。

---

## 8. 局限性与未来方向

### 8.1 Paper 明确提到

- **Runtime**: 16 分钟首 loop, 非实时。Backward optimization 是瓶颈 — 3 个视角 × video model inference + gaussian optimization。
- **Unseen geometry**: Figure S3 的 hockey stick 案例, 单视角无法 reconstruct 背面, Hunyuan3D 也无能为力。

### 8.2 推测的潜在局限

- **Action representation 粒度**: global force field 和 local force 是显式 given 的。如何从更抽象的 instruction (e.g., "轻轻拍一下苹果") 转换到 force vector, paper 没有讨论, 实际需要 VLM 协助。
- **Material 参数估计**: GPT-4o 估 Young's modulus, Poisson ratio 误差可能很大。Paper 说 "optional manual fine-tuning", 实际生产中可能需要 trial-and-error。
- **Velocity 直传的简化**: loop closure 时 velocity 不被 video refinement 修正, 这在高速运动场景 (e.g., 物体反弹) 可能积累误差。更完整的方案需要可微 physics 或者从 video optical flow 反推 velocity。
- **Multi-view 冲突的 progressive 策略是启发式的**: 3 个 view 是 fixed (frontal, left, right), control weight 的递减也是 hand-tuned。一个更原则性的方案是用 video model 的 uncertainty 来加权不同 view 的 supervision。
- **Foreground 分割依赖 SAM2 + Gaussian Grouping**: 在遮挡、复杂 material 边界 (e.g., fluid 与 background 颜色相近) 时可能失败, 进而影响 VPP 初始化。

### 8.3 与未来 world model 的关系

Sora [4] 提出的 "video models as world simulators" 概念, 但纯 video model 缺乏 explicit 3D / physics 结构。PerpetualWonder 的路线是 hybrid: physics 提供 structure, video model 提供 realism, closed-loop 让两者互相纠正。

这其实指向一个更宏观的方向: **structured world models**。纯 neural 的 world model (Sora, Genie, Genie 2) 在 long-horizon 下也会 drift, 因为没有 explicit state correction。PerpetualWonder 的 closed-loop 思想可以推广: 任何 long-horizon generative simulator 都需要一个 state representation 能被 observation (video model) 反向更新。

VPP 给出的一个具体实现: 软绑定的 anchor + consistency loss + progressive multi-view update。这个 pattern 可能在其他 modality (audio, tactile) 的 closed-loop 生成中也适用。

---

## 9. 总结

PerpetualWonder 的贡献集中在三点:

1. **VPP representation**: $\mu_{j,k} = p_j + \tanh(\tilde{p}_{j,k}) \cdot \delta$ — 用一个 saturating offset 把 K 个 gaussian 软绑定到一个 physics particle, 加上 temporal opacity 让 gaussian 可在时间上"出没"。这是 closed-loop 的几何基础。

2. **Multi-view progressive optimization**: 用 GEN3C 生成 dense surrounding views 初始化完整 3D scene, 再用 progressive 三阶段策略 (single-view → weak multi-view → full multi-view) 解决 video model 跨视角不一致问题。

3. **Loop closure**: 上一 window 结束时把 G 平均写回 P, 下一 window 用更新过的 P 作为初始 state, 实现 long-horizon sequential interaction。

实验显示 camera controllability 93.26, 3D consistency 80.41, 大幅超越 WonderPlay (75.95, 63.93) 和所有 video generator baseline, 在 human study 上有 62-93% 的 preference rate。Ablation 证明 VPP 和 progressive optimization 各自不可或缺。

Reference links:
- 项目主页: https://johnzhan2023.github.io/PerpetualWonder/
- World-Score: https://worldscore.github.io/
- GEN3C: https://research.nvidia.com/labs/toronto-ai/GEN3C/
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- SAM2: https://github.com/facebookresearch/sam2
- Hunyuan3D: https://github.com/Tencent/Hunyuan3D-2
- Genesis physics engine: https://github.com/Genesis-Embodied-AI/Genesis
- Go-with-the-flow: https://go-with-the-flow.github.io/
- WonderPlay (前作): https://WonderPlay.github.io/
- WonderWorld (相关 scene 生成): https://wonderworld-2024.github.io/
- PhysGaussian: https://physgaussian.github.io/
- PhysDreamer: https://physdreamer.github.io/
- PhysGen3D: https://physgen3d.github.io/
- Force Prompting: https://force-prompting.github.io/
- Voyager: https://voyager-video.github.io/
- Gaussian Grouping: https://github.com/lkeab/gaussian-grouping
- FreeTimeGS: https://arxiv.org/abs/2502.24608
