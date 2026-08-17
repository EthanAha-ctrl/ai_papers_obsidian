---
source_pdf: ManiGaussian Dynamic Gaussian Splatting for Multi-task Robotic Manipulation.pdf
paper_sha256: b38311f0229d7589fbebd08629ab0ae6d37e652283476a0f506ef729fb290f5d
processed_at: '2026-08-05T16:19:25-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ManiGaussian 用人话讲

## 一句话总结

ManiGaussian就是一个机器人, 它在动手之前会先"想象"一下: "如果我这样抓, 那个block会飞到哪里去?" 这个想象的过程用了3D Gaussian Splatting作为载体, 让想象既有几何结构, 又有物理意义。

## 为什么要做这件事

你让机器人"把红色方块推到黄色目标位置", 现有的方法分两派:

**Perceptive派**(PerAct这些人): 给机器人一个voxel grid, 用transformer直接预测action。问题是它只看"现在长什么样", 不懂"动了之后会怎样"。就像你只给一个人看一张照片, 让他猜怎么拼乐高, 他能模仿动作但不懂因果。

**Generative派**(GNFactor这些人): 用NeRF重建3D scene, 学到geometry prior再预测action。比第一派强, 因为至少知道物体的3D shape。但是它重建的是**静态**scene, 完全不建模时间维度。它知道"red block在这里", 不知道"如果我推它, 它会去哪里"。

ManiGaussian的作者就发现了这个gap: 机器人操作的核心难点, 不在于看清楚, 在于**理解物理交互**。你抓一个东西, 东西会动; 你推一个东西, 东西会滑。这个"动了会怎样"的信息, 现有方法完全没编码进representation里。

## 核心idea: Dynamic Gaussian Splatting + World Model

### Gaussian Splatting是啥

先回忆下vanilla 3D Gaussian Splatting (https://arxiv.org/abs/2208.08220)。它的思路是: 用一堆3D Gaussian椭圆球来表示scene, 每个Gaussian有:
- 位置 μ (在哪)
- 颜色 c (什么色)
- 旋转 r (朝哪)
- 缩放 s (多大)
- 不透明度 σ (多透)

渲染的时候, 把这些3D Gaussian按相机视角project到2D平面, 用alpha-blending叠加, 公式是:

$$C(\mathbf{p}) = \sum_{i=1}^{N} \alpha_i c_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

翻译成人话: pixel p的颜色 = 所有覆盖它的Gaussian按深度排序后, 各自贡献颜色乘以"我前面没人挡"的概率。$\alpha_i$是这个Gaussian在pixel p处的有效不透明度, $\prod_{j=1}^{i-1}(1-\alpha_j)$是前面那些Gaussian"透过去"的累积概率。

### ManiGaussian的关键改动: 给Gaussian加时间维度

vanilla Gaussian Splatting是static的, 一个scene一套参数。ManiGaussian说: 咱们让Gaussian的参数随时间变化吧。

具体怎么做? 他们先做了一个关键假设: **机器人操作中的物体都是rigid body**。这意味着:
- 颜色 $c_i$ 不变 (方块不会变色)
- 缩放 $s_i$ 不变 (方块不会变形)
- 不透明度 $\sigma_i$ 不变
- 语义特征 $f_i$ 不变 (red block的语义一直是"red block")

**只有位置 μ 和旋转 r 会变**。这就像你抓一个杯子, 杯子还是那个杯子, 只是位置和朝向变了。

所以时间传播规则很简单:

$$(\mu_i^{(t+1)}, r_i^{(t+1)}) = (\mu_i^{(t)} + \Delta\mu_i^{(t)}, r_i^{(t)} + \Delta r_i^{(t)})$$

第i个Gaussian在下一时刻的位置 = 当前位置 + 一个small offset。旋转同理。这个 $\Delta\mu_i^{(t)}$ 和 $\Delta r_i^{(t)}$ 由一个神经网络(deformation predictor)预测出来, 输入是当前Gaussian参数 + 当前action。

### World Model: 怎么训练这个deformation predictor?

这是整个paper的精髓。问题是: 怎么让网络学到"如果我执行action a, block会怎么动"?

答案: **future scene reconstruction**。你有expert demonstration, 它包含 $(o^{(t)}, a^{(t)}, o^{(t+1)})$ 三元组。你让网络:
1. 从 $o^{(t)}$ 预测当前Gaussian参数 $\theta^{(t)}$
2. 用 $\theta^{(t)}$ 和 $a^{(t)}$ 通过deformation predictor得到 $\theta^{(t+1)}$
3. 用Gaussian renderer把 $\theta^{(t+1)}$ 渲染成RGB图像 $\hat{C}^{(t+1)}$
4. 算 $\hat{C}^{(t+1)}$ 和真实 $C^{(t+1)}$ 的L2 loss

这个loss就是公式(8)的 $\mathcal{L}_{\text{Dyna}}$:

$$\mathcal{L}_{\text{Dyna}} = \|\hat{\mathbf{C}}^{(t+1)}(a^{(t)}, o^{(t)}) - \mathbf{C}^{(t+1)}\|_2^2$$

这个loss为什么牛逼? 因为它强迫representation $v^{(t)}$ 必须编码"哪些物体是rigid body"、"哪些物体会被gripper接触"、"接触之后力怎么传递"这些物理信息。如果你只预测action, 网络可以学到shortcut(比如"看到red block就输出某个固定action"); 但如果要预测future scene, 这个shortcut就不够了, 必须真的理解物理。

## 整个pipeline

```
当前RGB-D观察 o^(t)
        ↓
   [3D UNet q_φ]  ← 来自GNFactor的representation
        ↓
   voxel feature v^(t) ∈ R^(100³×128)
        ↓
   [Gaussian Regressor g_φ]  ← multi-head预测μ, c, r, s, σ, f
        ↓
   当前Gaussian参数 θ^(t) = {μ_i, c_i, r_i, s_i, σ_i, f_i}^16384
        ↓                              ↓
        ↓                    [Deformation Predictor p_φ]
        ↓                         ↑
        ↓                    action a^(t) (expert)
        ↓                         ↓
        ↓                    Δθ^(t) → θ^(t+1)
        ↓                         ↓
        ↓                    [Gaussian Renderer R]
        ↓                         ↓
        ↓                    predicted future RGB Ĉ^(t+1)
        ↓                         ↓
        ↓                    L_Dyna (vs 真实C^(t+1))
        ↓
   [PerceiverIO action decoder]
        ↓
   predicted action â^(t)
        ↓
   L_Act (vs expert action a^(t))
```

另外还有两个辅助loss:
- $\mathcal{L}_{\text{Geo}}$: 当前scene的重建loss, 让Gaussian参数能准确重建当前观察
- $\mathcal{L}_{\text{Sem}}$: 把Stable Diffusion的feature蒸馏到Gaussian的semantic head里

总loss:

$$\mathcal{L} = \mathcal{L}_{\text{Act}} + 0.01 \cdot \mathcal{L}_{\text{Geo}} + 0.0001 \cdot \mathcal{L}_{\text{Sem}} + 0.001 \cdot \mathcal{L}_{\text{Dyna}}$$

action loss权重是1, 其他三个是auxiliary。这个权重比例很重要, Table 5显示 $\lambda_{\text{Sem}}=0.001$ 时性能反而下降到37.6%, 因为太强的semantic prior会绑架网络。

## 训练的trick: warm-up

前3000 iterations冻结deformation predictor, 只训representation和Gaussian regressor。这就像DreamerV3里的staged training: 先让Gaussian参数稳定下来, 再让deformation predictor学习。如果一上来就joint train, deformation predictor会收到garbage gradient(因为Gaussian参数本身还在乱跳), 学到noise。

## 实验结果: 为什么这么强

### 主结果

| 方法 | 平均成功率 |
|------|-----------|
| PerAct | 20.4% |
| PerAct (4 cameras) | 22.7% |
| GNFactor | 31.7% |
| **ManiGaussian** | **44.8%** |

比SOTA高13.1%, 相对提升41.3%。

### 哪些任务提升最大

- **drag stick**: 37.3% → 92.0% (+146%)。这个任务要用stick推cube, 必须理解"stick接触cube后, cube怎么动"。GNFactor完全不行, 因为它不懂tool-object interaction。
- **sweep to dustpan**: 28.0% → 64.0% (+129%)。dustpan和dirt的接触关系。
- **stack blocks**: 4.0% → 12.0% (+200%)。长时序任务, 要预测多步堆积。
- **put in drawer**: 0.0% → 16.0% (从零突破)。drawer开闭 + item放入, 双物体运动。

### 哪些任务提升不大

- **close jar**: 25.3% → 28.0% (+2.7%)。单一rigid body的拧紧, 不需要复杂物理推理。
- **turn tap**: 50.7% → 56.0% (+5.3%)。简单旋转操作。

这告诉我们: dynamic modeling对**tool use**和**multi-object interaction**类任务帮助最大, 对single rigid body的简单操作帮助有限, 因为后者靠keyframe模仿就能解决。

### Ablation揭示的insight

| 几何 | 语义 | 动态 | 平均 |
|-----|------|------|------|
| ✗ | ✗ | ✗ | 23.6% |
| ✓ | ✗ | ✗ | 39.2% (+15.6) |
| ✓ | ✓ | ✗ | 41.6% (+2.4) |
| ✓ | ✗ | ✓ | 43.6% (+4.4) |
| ✓ | ✓ | ✓ | 44.8% (+1.2) |

几个关键观察:

1. **Geometry是地基**: +15.6%, 3D Gaussian Splatting本身就很强, 因为显式建模空间结构, 对occlusion和tool use特别有效。

2. **Dynamics比Semantic更值钱**: 加semantic只+2.4%, 加dynamics +4.4%。在RLBench这种已经discrete化的任务里, 语言instruction已经decoupled到action decoder里, semantic的作用被部分替代了; 而dynamics是其他模块完全无法提供的。

3. **Long-horizon任务特别依赖Dynamics**: Long任务从8.0(只有semantic)→14.0(加dynamic), 提升75%。这很符合直觉: 长时序任务必须预测未来才能planning。

4. **Occlusion任务也依赖Dynamics**: 56.0 → 76.0, 提升35.7%。因为被遮挡物体未来的运动可以通过dynamics预测, 弥补当前观察的缺失。

### 训练速度

ManiGaussian比GNFactor快2.29×。原因:
- Gaussian是explicit point-based representation, 渲染时直接alpha-blending
- NeRF要volume rendering, 每条ray采64-128个点, 每个点过MLP, 计算开销大10-100倍
- Gaussian的gradient是closed-form, NeRF要backprop through implicit function
- Gaussian的tile-based rasterization在GPU上cache友好

## 跟相关工作的关系

### vs. GNFactor

GNFactor用generalizable NeRF重建当前scene, 但完全static。ManiGaussian不仅重建当前, 还预测未来。相当于从"3D understanding"升级到"3D + 4D understanding"。GNFactor的失败case很好说明: 它"看到"green base是稳定的, 但不知道"如果gripper这么动, block会怎样掉下来"。

### vs. UniPi (https://arxiv.org/abs/2302.03081)

UniPi用video diffusion生成未来视频, 再用inverse dynamics model提取action。但video在pixel space, 训练diffusion需要海量数据, 在RLBench这种小数据集上infeasible。

ManiGaussian选择在**Gaussian embedding space**预测未来, 这是sweet spot:
- 比pixel space更compact, 数据需求小
- 比text space更几何, 表达力强
- 与3D task天然对齐(action就是3D pose)

### vs. DreamerV3

Dreamer在latent space做imagination, 用RSSM预测未来latent。问题: latent space的disentanglement不可控, 几何和语义纠缠在一起。ManiGaussian的Gaussian space天然disentangled: μ就是position, r就是orientation, f就是semantics, 每个维度都有明确物理含义。

### vs. PhysGaussian (https://arxiv.org/abs/2311.12198)

PhysGaussian把物理simulator(MPM/FEM)嵌入Gaussian, 用物理方程驱动deformation。但需要mesh和material properties, 在RLBench这种没有物理参数标注的环境里用不了。ManiGaussian用data-driven的deformation predictor绕开了对physics prior的需求, 用supervision让网络自己学到physics-like dynamics。

## Limitations

1. **需要multi-view supervision**: 训练时需要20个相机视角的GT RGB做loss, real-world部署要camera calibration。这是Gaussian Splatting所有方法的通病。

2. **Rigid body assumption**: 假设物体颜色、scale、opacity在操作中不变。对deformable objects(cloth, rope)失败。如果要扩展到soft body, 需要让 $c_i^{(t)}, s_i^{(t)}$ 也时间相关, deformation predictor的输出从7维升到22维per Gaussian。

3. **单步future prediction**: 只预测 $t+1$, 长horizon靠iterative rollout, 误差累积。可以做multi-step prediction或Monte Carlo tree search。

4. **Action space discretization**: 100³ translation bin + 5度 rotation bin, 精度受限。精细装配任务需要continuous action head或coarse-to-fine refinement。

5. **Appearance-based physics shortcut**: 当dynamic supervision只有RGB时, 网络可能学到shortcut: "如果gripper在block右边, 就把block的μ向左移一点"这种correlational pattern, 而不是真正的contact force。在OOD分布(不同摩擦系数)上会失败。如果要让world model真正理解physics, 需要额外的contact、force、material监督。

## 我的核心take-away

**1. World model的representation choice决定了它的天花板。** Pixel-space world model(UiPi)需要海量数据; latent-space world model(Dreamer)可解释性差; Gaussian-space world model在这两者间找到了平衡。Gaussian的显式参数化让action prediction可以"读"到position, "读"到orientation, "读"到scale, 这是latent vector做不到的。

**2. Rigid body assumption是"作弊"也是"神来之笔"。** 作弊: 对soft body完全失效; 神来之笔: 大幅简化了deformation predictor的搜索空间, 让网络在 $10^6$ 级别的动作空间里只需要预测7维per Gaussian的偏移, 这就是为啥能用16384个Gaussian而训练100k步就收敛。

**3. Action prediction和world model是mutually beneficial的。** Action supervision告诉Gaussian参数"哪些信息有用"; world model监督告诉Gaussian参数"物理交互是什么样"。如果只训action, Gaussian会塌缩成noise; 如果只训world model, Gaussian不知道哪些feature对action重要。

**4. Gaussian Splatting在robotics的价值不在rendering, 在structural inductive bias。** ManiGaussian渲染出来的图像质量未必比NeRF好, 但Gaussian的explicit structure让robot agent可以直接用Gaussian参数做action inference, 而NeRF要先query MLP再decode, 不利于online decision-making。

**5. 这个paradigm的极限是physical accuracy。** 当dynamic supervision只有RGB时, 网络学到的"physics"是appearance-based的。如果要让world model真正理解physics, 需要额外的contact、force、material监督信号。这是ManiGaussian下一步的可能方向。

## 相关链接

- Paper: https://arxiv.org/abs/2406.11777
- Code: https://github.com/GuanxingLu/ManiGaussian
- 3D Gaussian Splatting原始论文: https://arxiv.org/abs/2208.08220
- GNFactor: https://arxiv.org/abs/2305.08448
- PerAct: https://arxiv.org/abs/2209.05451
- RLBench: https://arxiv.org/abs/1909.12271
- DreamerV3: https://arxiv.org/abs/2301.04104
- UniPi: https://arxiv.org/abs/2302.03081
- Dynalang: https://arxiv.org/abs/2308.01399
- PhysGaussian: https://arxiv.org/abs/2311.12198
- LangSplat: https://arxiv.org/abs/2312.16084
- Act3D: https://arxiv.org/abs/2403.19655
- GaussianGrasper: https://arxiv.org/abs/2405.19306

总之, ManiGaussian这篇paper给我的intuition是: **robot learning的下一个突破点不在bigger policy network, 而在更好的scene + dynamics representation**。Gaussian Splatting作为一个显式的、物理可解释的、可微分的3D representation, 是一个非常有前途的载体。未来的方向可能是: 把Gaussian world model和LLM结合(用language指导imagination), 或者用更强的physics prior(MPM, differentiable simulation)替换data-driven的deformation predictor。

---

# ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation 深度讲解

这篇paper的核心问题非常清晰: language-conditioned robotic manipulation中, 现有方法(无论是perceptive methods还是generative methods)都忽略了**scene-level spatiotemporal dynamics**(场景级时空动态), 即物体间物理交互的建模。GNFactor虽然用NeRF做了generative的3D understanding, 但是它只能重建static scene, 无法预测"如果gripper这么动, block会怎么走"这种因果链。ManiGaussian的关键insight是: 把Gaussian Splatting变成一个dynamic, action-conditioned的world model, 让representation本身被迫编码物理交互信息, 从而产生更好的action prediction。

参考链接:
- Paper: https://arxiv.org/abs/2406.11777
- 3D Gaussian Splatting原始论文: https://arxiv.org/abs/2208.08220
- GNFactor: https://arxiv.org/abs/2305.08448
- RLBench: https://arxiv.org/abs/1909.12271
- PerAct: https://arxiv.org/abs/2209.05451

## 1. 整体架构直觉

整个pipeline可以拆成四个 cascaded modules, 全部end-to-end可微:

```
RGB-D观察 o^(t)
   ↓
[Representation q_φ] → voxel feature v^(t) ∈ R^(100³×128)
   ↓
[Gaussian Regressor g_φ] → 当前Gaussian参数 θ^(t) = {μ, c, r, s, σ, f}^N
   ↓                                       ↓
   ↓                                  [Deformation Predictor p_φ] ← action a^(t)
   ↓                                       ↓
   ↓                                  Δθ^(t) → θ^(t+1)
   ↓                                       ↓
   ↓                                  [Gaussian Renderer R] → future RGB
   ↓                                       ↓
   ↓                                  L_Dyna监督
   ↓
[PerceiverIO action decoder] ← (language + Gaussian embedding)
   ↓
action a^(t) = (translation, rotation, openness, collision)
```

关键insight: Gaussian embedding space既被action decoder用来输出action, 又被renderer约束重建current和future scene。这种dual supervision让Gaussian参数本身就承载了"几何 + 语义 + 物理"三者。

## 2. Dynamic Gaussian Splatting的数学构造

### 2.1 Vanilla Gaussian Splatting回顾

公式(1)的alpha-blending rendering:

$$C(\mathbf{p}) = \sum_{i=1}^{N} \alpha_i c_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

其中:
- $\mathbf{p}$: 待渲染的2D pixel坐标
- $N$: tile-based排序后覆盖该pixel的Gaussian数量
- $c_i$: 第i个Gaussian的color(实际用spherical harmonics基的系数, 维度12)
- $\alpha_i$: 该Gaussian在pixel $\mathbf{p}$ 处的2D density, 由3D opacity $\sigma_i$ 经过投影衰减得到

$$\alpha_i = \sigma_i \exp\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1} (\mathbf{p} - \boldsymbol{\mu}_i)\right)$$

变量含义:
- $\sigma_i$: opacity, 学习得到的不透明度, ∈ [0, 1]
- $\boldsymbol{\mu}_i$: 该Gaussian在2D投影后的中心(由3D中心投影得到)
- $\boldsymbol{\Sigma}_i$: covariance matrix, 由rotation quaternion $r_i$和scaling vector $s_i$参数化: $\boldsymbol{\Sigma}_i = R_i S_i S_i^\top R_i^\top$, 这里$R_i$是quaternion $r_i$对应的rotation matrix, $S_i = \text{diag}(s_i)$。这种参数化保证$\Sigma$半正定

- $\prod_{j=1}^{i-1}(1-\alpha_j)$: 前景遮挡项, 按深度排序, 让更靠前的Gaussian优先贡献颜色

### 2.2 关键修改: 时间维度扩展

公式(2)给Gaussian primitive加上时间下标:

$$\boldsymbol{\theta}_i^{(t)} = (\mu_i^{(t)}, c_i^{(t)}, r_i^{(t)}, s_i^{(t)}, \sigma_i^{(t)}, f_i^{(t)})$$

上标$(t)$表示第t步的值。$f_i^{(t)} \in \mathbb{R}^3$是从Stable Diffusion visual encoder蒸馏得到的语义特征(经过降维)。

**关键刚性假设(rigid body assumption)**: 在机器人操作中, 物体被当作rigid body, 因此:
- $c_i^{(t)}, s_i^{(t)}, \sigma_i^{(t)}, f_i^{(t)}$ **时间无关**(在操作过程中保持不变)
- 只有 $\mu_i^{(t)}$ (位置)和 $r_i^{(t)}$ (旋转)随时间变化

这个假设大幅减少了deformation predictor需要预测的维度: 从6维(μ 3 + r 3, rotation用quaternion是4维, 但实际预测增量)降到只预测Δμ(3维) + Δr(4维) = 7维。如果Gaussian数量是16384(见Table 4), 那么deformation predictor的输出是 $16384 \times 7 = 114688$维。这比预测所有参数(每个Gaussian约22维)减少了近3倍。

公式(3)的传播规则:

$$(\mu_i^{(t+1)}, r_i^{(t+1)}) = (\mu_i^{(t)} + \Delta\mu_i^{(t)}, r_i^{(t)} + \Delta r_i^{(t)})$$

注意rotation用quaternion相加是approximation, 严格应该用quaternion multiplication $r^{(t+1)} = \Delta r^{(t)} \otimes r^{(t)}$, 但paper用加法作为简化(可能quaternion被normalized后approximation可接受)。如果严格的话需要用Lie algebra se(3)的exponential map, 但简化为additive对RLBench的短时序任务够用。

## 3. Gaussian World Model的内部结构

公式(4)是world model的四步流水线:

$$\begin{cases}
\mathbf{v}^{(t)} = q_\phi(o^{(t)}) & \text{(Representation)} \\
\boldsymbol{\theta}^{(t)} = g_\phi(\mathbf{v}^{(t)}) & \text{(Gaussian regressor)} \\
\Delta\boldsymbol{\theta}^{(t)} = p_\phi(\boldsymbol{\theta}^{(t)}, \mathbf{a}^{(t)}) & \text{(Deformation predictor)} \\
o^{(t+1)} = \mathcal{R}(\boldsymbol{\theta}^{(t+1)}, w) & \text{(Gaussian renderer)}
\end{cases}$$

各模块的设计细节(参考Appendix B):

### 3.1 Representation model $q_\phi$
继承自GNFactor, 是一个shallow 3D UNet。输入是voxel $\in \mathbb{R}^{100^3 \times 10}$(RGB features + coordinates + indices + occupancy), 输出是 $\mathbf{v}^{(t)} \in \mathbb{R}^{100^3 \times 128}$。100³的voxel分辨率意味着每个voxel大约对应1cm³(假设workbench是1m³)。

### 3.2 Gaussian regressor $g_\phi$
这是一个multi-head的轻量级网络。每个head负责预测一个Gaussian参数:
1. **Position offset head**: 输出每个voxel位置的3D center offset ∈ $\mathbb{R}^3$
2. **Color head**: 输出spherical harmonic基的系数 ∈ $\mathbb{R}^{12}$
3. **Rotation head**: 输出normalized quaternion ∈ $\mathbb{R}^4$
4. **Scaling head**: 用exponential activation输出scaling ∈ $\mathbb{R}^3$(保证正定)
5. **Opacity head**: 用sigmoid输出opacity ∈ $\mathbb{R}^1$
6. **Semantic head**: 输出语义特征 ∈ $\mathbb{R}^3$

每个voxel中心对应一个Gaussian primitive, 16384个Gaussian对应 $16384 / 100^3 \approx 0.16\%$ 的voxel占用率, 说明只有前景的voxel被激活。

### 3.3 Deformation predictor $p_\phi$
一个fully-connected network + residual connections。输入是当前Gaussian参数 $\theta^{(t)}$ 和action $\mathbf{a}^{(t)}$, 输出 $\Delta\mu_i^{(t)} \in \mathbb{R}^3$ 和 $\Delta r_i^{(t)} \in \mathbb{R}^4$。Residual connection很重要: 它让网络学习"偏移量"而非绝对值, 数值范围小, 训练更稳定。这与PhysGaussian (https://arxiv.org/abs/2311.12198) 中物理驱动deformation的思路类似, 但ManiGaussian是data-driven而非physics-driven。

## 4. 损失函数的直觉

四个loss加权组合:

$$\mathcal{L} = \mathcal{L}_{\text{Act}} + \lambda_{\text{Geo}}\mathcal{L}_{\text{Geo}} + \lambda_{\text{Sem}}\mathcal{L}_{\text{Sem}} + \lambda_{\text{Dyna}}\mathcal{L}_{\text{Dyna}}$$

权重设置(从Table 4):
- $\lambda_{\text{Geo}} = 0.01$
- $\lambda_{\text{Sem}} = 0.0001$  
- $\lambda_{\text{Dyna}} = 0.001$

这个权重大小说明: action prediction是主任务, 几何重建是次任务, semantic和dynamic是auxiliary。Sem权重最小是因为Stable Diffusion feature本身就很大, 直接L2/similarity loss数值范围大, 需要scale down。

### 4.1 Current Scene Consistency Loss (公式5)

$$\mathcal{L}_{\text{Geo}} = \|\mathbf{C}^{(t)} - \hat{\mathbf{C}}^{(t)}\|_2^2$$

$\mathbf{C}^{(t)}$: 来自20个相机视角的ground truth RGB; $\hat{\mathbf{C}}^{(t)}$: renderer用当前Gaussian参数投影到相同视角的渲染结果。20个视角覆盖整个workbench, 提供multi-view supervision。

### 4.2 Semantic Feature Consistency Loss (公式6)

$$\mathcal{L}_{\text{Sem}} = 1 - \sigma_{\text{cos}}(\mathbf{F}^{(t)}, \hat{\mathbf{F}}^{(t)})$$

$\mathbf{F}^{(t)}$: Stable Diffusion encoder在RGB上的输出feature map; $\hat{\mathbf{F}}^{(t)}$: ManiGaussian的semantic head输出的f_i经过renderer投影的feature map; $\sigma_{\text{cos}}$: cosine similarity。这是典型的CLIP-style feature distillation, 但应用在Gaussian space上(类似LangSplat: https://arxiv.org/abs/2312.16084)。

### 4.3 Action Prediction Loss (公式7)

$$\mathcal{L}_{\text{Act}} = \text{CE}(p_{\text{trans}}, p_{\text{rot}}, p_{\text{open}}, p_{\text{col}})$$

这是一个**分类**任务而非回归:
- $p_{\text{trans}}$: 在 $100^3 = 10^6$ 个voxel位置上的分类概率
- $p_{\text{rot}}$: 在 $(360/5) \times 3 = 216$ 个离散旋转上的分类概率(每5度一个bin, 3轴)
- $p_{\text{open}}$: 二分类(gripper open/close)
- $p_{\text{col}}$: 二分类(collision/no collision)

总分类维度: $10^6 \times 216 \times 2 \times 2 \approx 8.6 \times 10^8$, 这就是PerAct风格的"next keyframe prediction"formulation。PerceiverIO通过cross-attention把这个高维空间压缩到latent query tokens上来做分类。

### 4.4 Future Scene Consistency Loss (公式8)

$$\mathcal{L}_{\text{Dyna}} = \|\hat{\mathbf{C}}^{(t+1)}(\mathbf{a}^{(t)}, o^{(t)}) - \mathbf{C}^{(t+1)}\|_2^2$$

$\hat{\mathbf{C}}^{(t+1)}(\mathbf{a}^{(t)}, o^{(t)})$: 用当前observation和expert action通过deformation predictor预测的future scene rendering; $\mathbf{C}^{(t+1)}$: 真实的next step observation。这个loss是整个paper的精髓, 它迫使representation $\mathbf{v}^{(t)}$ 不仅编码"当前长什么样", 还要编码"如果gripper这样动, 物体会怎样响应"。

### 4.5 训练schedule
前3k iterations冻结deformation predictor, 只训练representation和Gaussian regressor, 让Gaussian参数先稳定下来。之后再joint training。这个warm-up类似DreamerV3中的staged training, 防止deformation predictor在Gaussian还不准时学到garbage gradient。

## 5. 实验数据深度解析

### 5.1 主实验(Table 1)

| Method | Avg Success Rate |
|--------|------------------|
| PerAct | 20.4% |
| PerAct (4 cameras) | 22.7% |
| GNFactor | 31.7% |
| **ManiGaussian** | **44.8%** |

相对GNFactor的提升: $(44.8 - 31.7) / 31.7 = 41.3\%$。绝对提升13.1%。

按task分解的亮点:
- **drag stick**: 92.0% (GNFactor 37.3%) → 提升146%。这是tool use任务, 需要理解stick如何传递力到cube, dynamic modeling在这里最关键
- **sweep to dustpan**: 64.0% (GNFactor 28.0%) → 提升129%。dustpan和dirt的接触关系
- **stack blocks**: 12.0% (GNFactor 4.0%) → 提升200%。长时序任务, 需要预测多步堆积
- **put in drawer**: 16.0% (GNFactor 0.0%) → GNFactor完全失败, ManiGaussian从0到16。drawer的开闭需要预测drawer和item的双物体运动

但有些任务提升不大:
- **close jar**: 28.0% (GNFactor 25.3%) → 仅+2.7%
- **turn tap**: 56.0% (GNFactor 50.7%) → +5.3%

这种差异说明: dynamic modeling对tool use和multi-object interaction类任务帮助最大, 而对single rigid body的简单操作(如turn tap)帮助有限, 因为后者本身就靠keyframe模仿能解决。

### 5.2 Ablation Study (Table 2)

| Geo | Sem | Dyna | Avg |
|-----|-----|------|-----|
| ✗ | ✗ | ✗ | 23.6% |
| ✓ | ✗ | ✗ | 39.2% (+15.6) |
| ✓ | ✓ | ✗ | 41.6% (+2.4) |
| ✓ | ✗ | ✓ | 43.6% (+4.4) |
| ✓ | ✓ | ✓ | 44.8% (+1.2) |

关键观察:
1. **Geometry是地基**: 从23.6 → 39.2, +15.6%, 表明3D Gaussian Splatting本身就比vanilla 2D/voxel representation强很多。这是因为Gaussian显式建模空间结构, 对occlusion和tool use任务尤其有效
2. **Dynamics比Semantic更值钱**: 加Sem只+2.4%, 加Dyna+4.4%。这说明在RLBench这种已经discrete化的任务里, 语言instruction本身已经decoupled到action decoder里, semantic的作用被部分替代; 而dynamics是其他模块完全无法提供的
3. **Long-horizon任务特别依赖Dyna**: Long任务从8.0(只有Sem)→14.0(加Dyna), 提升75%。这很符合直觉, 长时序任务必须预测未来才能planning
4. **Occlusion任务也依赖Dyna**: 56.0 → 76.0, 提升35.7%。原因: 遮挡物体未来的运动可以通过dynamics预测, 弥补当前观察缺失

### 5.3 Balance Hyperparameters (Table 5)

这是一个很有意思的sensitivity analysis:

| $\lambda_{\text{Geo}}$ | $\lambda_{\text{Sem}}$ | $\lambda_{\text{Dyna}}$ | Avg |
|---|---|---|---|
| 0.01 | 0 | 0.0001 | 42.4 |
| 0.01 | 0 | 0.001 | 43.6 |
| 0.01 | 0.0001 | 0 | 41.6 |
| 0.01 | 0.001 | 0 | 37.6 |
| 0.01 | 0.0001 | 0.001 | **44.8** |

关键发现: $\lambda_{\text{Sem}} = 0.001$时性能反而下降到37.6%, 说明semantic feature distillation权重过大时, 网络会被pretrained feature绑架, 牺牲了task-specific的几何/动态信息。这呼应了Radford等人的观察: foundation model的feature虽然是好的prior, 但太强的prior会损害下游任务。

### 5.4 学习曲线(Figure 3)

ManiGaussian在性能和速度上都优于GNFactor:
- 性能: 1.18×更好
- 速度: 2.29×更快

为什么Gaussian比NeRF快这么多? 三个原因:
1. **Explicit vs Implicit**: Gaussian是explicit point-based representation, 渲染时直接alpha-blending, 而NeRF要volume rendering(每条ray采64-128个点, 每个点过MLP), 计算开销大10-100倍
2. **Gradient友好**: Gaussian的参数是closed-form gradient, NeRF的MLP需要backprop through implicit function
3. **Memory访问pattern**: Gaussian的tile-based rasterization在GPU上cache友好, NeRF的随机sample对cache不友好

## 6. 与相关工作的对比直觉

### 6.1 vs. GNFactor
GNFactor用generalizable NeRF做scene reconstruction, 但只重建**当前**scene。ManiGaussian不仅重建当前, 还预测**未来**。这相当于从"3D understanding"升级到"3D + 4D understanding"。GNFactor的失败case(Figure 1)很好地说明了: 它"看到"green base是稳定的, 但不知道"如果gripper这么动, block会怎样掉下来"。

### 6.2 vs. UniPi / Dynalang
UniPi (https://arxiv.org/abs/2302.03081) 用video diffusion生成未来视频, 然后用inverse dynamics model提取action。但video是在pixel space, 训练diffusion需要海量数据, 在RLBench这种小数据集上infeasible。Dynalang (https://arxiv.org/abs/2308.01399)在text space预测, 但text无法表达精细几何。

ManiGaussian选择在**Gaussian embedding space**预测未来, 这是一个sweet spot:
- 比pixel space更compact, 数据需求小
- 比text space更几何, 表达力强
- 与3D task天然对齐(action就是3D pose)

### 6.3 vs. DayDreamer / DreamerV3
Dreamer系列在latent space做imagination, 用RSSM预测未来latent。Latent space的问题是disentanglement不可控, 几何和语义纠缠在一起。ManiGaussian的Gaussian space天然disentangled: μ就是position, r就是orientation, f就是semantics, 每个维度都有明确物理含义。

### 6.4 vs. PhysGaussian
PhysGaussian (https://arxiv.org/abs/2311.12198) 把物理simulator嵌入Gaussian, 用MPM或FEM驱动deformation。但需要mesh和material properties, 在RLBench这种没有物理参数标注的sim环境里用不了。ManiGaussian用data-driven的deformation predictor绕开了对physics prior的需求, 用supervision信号让网络自己学到physics-like dynamics。

## 7. 关键Limitations与未来方向

1. **Multi-view supervision依赖**: 训练时需要20个相机视角的GT RGB做$L_{\text{Geo}}$和$L_{\text{Dyna}}$监督, real-world部署需要camera calibration。这是Gaussian Splatting所有方法的通病。

2. **Rigid body assumption**: 假设物体颜色、scale、opacity在操作中不变。这对deformable objects(cloth, rope)失败。如果要扩展到soft body manipulation, 需要让$c_i^{(t)}, s_i^{(t)}$也时间相关, deformation predictor的输出维度从7维升到22维。

3. **单步future prediction**: 公式(8)只预测$t+1$, 长horizon planning靠iterative rollout。Rollout误差会累积, 类似Dreamer中horizon过长latent imagination漂移的问题。可以做multi-step prediction或 Monte Carlo tree search结合。

4. **Action space的discretization**: 100³的translation bin + 5度rotation bin, 精度受限。如果要做精细装配任务(<1mm精度), 需要continuous action head或coarse-to-fine refinement(类似Act3D: https://arxiv.org/abs/2403.19655)。

5. **Voxel vs. Gaussian的对齐**: $100^3$的voxel映射到16384个Gaussian, 平均每个Gaussian对应610个voxel。这个稀疏化会丢失几何细节。可以用importance sampling或adaptive density控制动态分配Gaussian数量。

## 8. 我的整体intuition

读完这篇paper后, 我对它有几个核心take-away:

**第一, world model的representation choice决定了它的天花板。** Pixel-space world model(UiPi)需要海量数据; latent-space world model(Dreamer)可解释性差; Gaussian-space world model在这两者间找到了平衡。Gaussian的显式参数化让action prediction可以"读"到position, "读"到orientation, "读"到scale, 这是latent vector做不到的。

**第二, rigid body assumption是这个paper的"作弊"也是"神来之笔"。** 作弊在于它对soft body完全失效; 神来之笔在于它大幅简化了deformation predictor的搜索空间, 让网络在$10^6$级别的动作空间里只需要预测7维per Gaussian的偏移, 这就是为啥能用16384个Gaussian而训练100k步就收敛。

**第三, action prediction这个主任务和world model这个auxiliary task是mutually beneficial的。** Action supervision告诉Gaussian参数"哪些信息有用"; world model监督告诉Gaussian参数"物理交互是什么样"。如果只训action, Gaussian会塌缩成noise; 如果只训world model, Gaussian不知道哪些feature对action重要。Table 2的ablation正印证了这点: 几何+语义+动态三者叠加, 性能从23.6飞到44.8。

**第四, Gaussian Splatting在robotics的价值不在rendering, 而在structural inductive bias。** ManiGaussian渲染出来的图像质量未必比NeRF好, 但Gaussian的explicit structure让robot agent可以直接用Gaussian参数做action inference, 而NeRF要先query MLP再decode, 不利于online decision-making。

**第五, 这个paradigm的极限是physical accuracy。** 当dynamic supervision只有RGB时, 网络学到的"physics"是appearance-based的, 可能学到shortcut。比如block被推的时候, 网络可能不学contact force, 而是学"如果gripper在block右边, 就把block的μ向左移一点"这种correlational pattern。在OOD分布(比如不同摩擦系数)上会失败。如果要让world model真正理解physics, 需要额外的contact、force、material监督信号。

最后, 这篇paper的方向(用3D Gaussian作为robot learning的backbone)非常promising, 我会关注它在real-world robot部署、bimanual manipulation、以及与LLM结合(LangGaussian-style)上的后续工作。值得关注的follow-ups: GaussianGrasper (https://arxiv.org/abs/2405.19306), SpatioNet (https://arxiv.org/abs/2404.03825), 还有与diffusion policy结合的探索方向。
