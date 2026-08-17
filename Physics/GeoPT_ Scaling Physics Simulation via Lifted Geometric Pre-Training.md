---
source_pdf: GeoPT_ Scaling Physics Simulation via Lifted Geometric Pre-Training.pdf
paper_sha256: 038e00cd012325eea1591be34c1cf5f65016bec6b5e5637bcba7ecc4feebc2be
processed_at: '2026-08-04T21:32:55-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GeoPT 人话版：用"假动作"白嫖物理先验

好，咱们把这篇paper揉碎了，用大白话讲一遍。我尽量像在coffee chat里跟你聊那样。

---

## 一句话总结

**物理仿真数据太贵，咱们手头只有一堆免费3D模型。直接拿3D模型做pre-training反而有害。GeoPT的trick是给每个3D模型配上随机假速度场，让模型学"粒子在任意velocity下如何沿geometry演化"，这个universal prior能transfer到真实物理仿真。**

---

## 1. 为什么这事难：label是物理ML的命门

你train一个neural simulator去预测汽车周围的气流， supervised loss就是：

$$\mathcal{L}^{\mathrm{physics}} = \mathbb{E}_{\mathcal{D}}\left[ \| \mathcal{F}_{\theta}(\mathbf{x}; G, S) - \mathbf{u}(\mathbf{x}) \|_2^2 \right]$$

变量翻一下：
- $\mathbf{x}$：query点位置（你要在哪预测物理量）
- $G$：geometry（车的mesh）
- $S$：simulation settings（来流速度、攻角这些）
- $\mathbf{u}(\mathbf{x})$：solver算出来的ground truth物理场（压力、速度）
- $\mathcal{F}_{\theta}$：你的neural net

问题在哪？生成一个$\mathbf{u}(\mathbf{x})$要跑一次CFD solver，DrivAerML里**单个sample要6.1×10^4 CPU小时**。你想想，LLM pre-training动不动几B sample，物理这边100个sample就让你烧不起。

vision有web image，language有web text，物理没有web physics。这是根本瓶颈。

---

## 2. Naive的想法为啥挂了

自然idea：ShapeNet有海量3D mesh，拿来做self-supervised pre-training呗。最naive的loss：

$$\mathcal{L}_{\mathrm{native}}^{\mathrm{pre}} = \mathbb{E}_{\mathbf{x}, G}\left[ \| \mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G) - \pmb{h}_G(\mathbf{x}) \|_2^2 \right]$$

这里$h_G(\mathbf{x})$是geometry-derived feature，比如SDF（signed distance field）或者vector distance。你train一个model去predict"给定点到geometry表面的距离"。

听起来挺合理对吧？**实测结果：比from scratch还差。** 见Fig.1。

**为啥？** 作者做了个非常漂亮的visualization（Fig.3a），把Transolver学到的aggregation weight画出来：

- **Geometry-only pre-training**：模型把车头和车尾归到同一个token，左右pattern不对称 → 完全按static shape cue分组
- **Physics supervision**：前后不对称、左右对称 → 这才对应真实的气动流场结构

**根本原因**：下游任务需要$\mathcal{G} \times \mathcal{S} \to \mathcal{U}$这个mapping，但pre-training只有$\mathcal{G} \to \mathcal{H}$。**pre-training空间严格小于downstream空间**，缺的那个维度（dynamics $\mathcal{S}$）在pre-training里完全是random的，导致model学到的correlation跟物理无关，甚至起反作用。

这跟MAE完全不一样。MAE做image→masked image重建，input space和downstream（recognition）space对齐。GeoPT面对的是**input严格低维**的新场景，SSL的native-space assumption挂了。

---

## 3. 核心trick：用合成dynamics把空间"撑大"

作者的idea很妙。既然真实dynamics $\pmb{v}_S$要跑solver才得到，那我**瞎编一个random velocity**行不行？

考虑粒子运动：

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{v} \cdot \mathbb{1}_G(\mathbf{x}_t), \quad \mathbf{x}_0 = \mathbf{x}, \quad \mathbf{v} \sim \mathrm{Unif}(\mathbb{B}^C)$$

翻译一下：
- $\mathbf{x}_t$：粒子在time $t$的位置
- $\mathbf{v}$：每个粒子i.i.d.采样的随机速度，从一个半径$v_{\max}=2$的ball里采
- $\mathbb{1}_G(\mathbf{x}_t)$：indicator函数，**粒子在geometry外面返回1，碰到表面返回0**——粒子撞墙就停下

这个设计有两个关键：
1. **随机速度让空间远点耦合**：不同initial position的粒子trajectories可能intersect，迫使他们share信息
2. **sticking boundary**：模拟真实物理的boundary interaction（气动压力作用在车表面，crash接触力在接触面）

然后监督信号就是**沿这条trajectory的geometry feature序列**：

$$h_G(\mathbf{x}_{0:\tau}) = \{h_G(\mathbf{x}_t)\}_{t=0}^{\tau}$$

- $\tau=2$（3个time step包括t=0）
- $h_G$用vector distance（比SDF好，因为带方向信息）

完整pre-training loss：

$$\mathcal{L}_{\mathrm{lifted}}^{\mathrm{pre}} = \mathbb{E}_{\mathbf{x}, G, V}\left[ \left\| \mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G, V) - \pmb{h}_G(\mathbf{x}_{0:\tau}) \right\|_2^2 \right]$$

三个随机变量一起averaged：
- $G$：从ShapeNet采geometry
- $\mathbf{x}$：从volume和surface同时采tracking point
- $V$：per-point random velocity

---

## 4. 这个"lifting"到底干了啥

Paper里那张inset图特别清楚：

```
(G, V) ──lifted pretrain──> H_traj   ← 新空间，包含dynamics
  ↑ lifting                  ↓ slice
  G    ──native pretrain──>  H       ← 老空间，纯geometry
```

**关键insight**：downstream simulation也活在$(\mathcal{G}, \mathcal{V})$这个joint space里！你fine-tune时把random velocity换成real $V_S$（来流速度、impact方向这些），model直接就能用。

这跟ControlNet的思路有点神似——你有一个universal backbone，conditioning作为"prompt"激活对应能力。GeoPT的dynamics condition就是prompt。

---

## 5. Fine-tuning：用真dynamics当prompt

Fine-tune时换成task-specific velocity：

$$\mathcal{L}^{\mathrm{fine}} = \mathbb{E}_{\mathbf{x}, G, V_S}\left[ \left\| \mathcal{F}_{\theta}(\mathbf{x}; G, V_S) - \mathbf{u}(\mathbf{x}) \right\|_2^2 \right]$$

不同任务配置$V_S$的方式：

| Task | Velocity direction | Velocity magnitude |
|------|-------------------|-------------------|
| Aerodynamics (DrivAerML, NASA-CRM, AirCraft) | 来流方向 | freestream speed |
| Hydrodynamics (DTCHull) | 船速方向，水/气两相分别配 | 船速 |
| Crash (Car-Crash) | impact方向 | 从collision point空间衰减 |
| Radiosity (bonus) | 光传播方向 | 光强 |

**一个pre-trained model，靠reconfigure velocity input就能适配5种物理**。这个unified interface是paper的亮点。

---

## 6. 为啥这个prior能transfer：Transport equation视角

Appendix B的证明很有意思。作者说，sampling random particle trajectories等价于解一个phase-space transport equation：

$$\partial_t f(x, v, t) + v \cdot \nabla_x f(x, v, t) = 0$$

变量解释：
- $f(x, v, t)$：phase-space density（位置$x$、速度$v$、时间$t$处的粒子密度）
- $\partial_t$：对时间偏导
- $v \cdot \nabla_x$：速度和空间梯度的内积，就是advection operator

加上sticking boundary：

$$\partial_t f_G(x, v, t) = (v \cdot n(x))_+ f(x, v, t) dv, \quad x \in G$$

- $f_G$：boundary上的粒子密度
- $n(x)$：outward normal
- $(v \cdot n)_+$：只有$v \cdot n > 0$（向边界外移动）的粒子能stick

**Proposition B.1说这个系统mass守恒**：内部粒子被transfer到boundary，总量不变。

**为啥能transfer**（Remark B.3）：transport equation是**所有PDE的common骨架**。Boltzmann加collision operator，Navier-Stokes加momentum和constitutive relation。GeoPT学的不是某个具体物理，是**"沿characteristics传+boundary交互"这个universal structure**。

所以我理解GeoPT的本质是：**用最弱的common prior pre-train，让model学会所有PDE共享的skeleton，然后fine-tune时具体填充某个physics的flesh**。

---

## 7. 数据生成效率

这部分数字让人惊叹：

- Tracking 36,864 points in一个sample：**0.2秒**（80 CPU cores）
- 工业CFD一个sample：6.1×10^4 CPU小时 ≈ 2.2×10^8 CPU秒
- **加速比约10^7倍**

所以生成1,346,300个pre-training samples（约5TB）只要3天80 cores。这跟工业CFD完全是两个量级的世界。

**Algorithm 1核心**：

```python
for G in ShapeNet:
    Normalize(G)  # rotate, shift, scale to unit
    Build FCPW scene  # ray-triangle intersection加速
    {x} ← Sample(Ω_G ∪ ∂G)  # 32K volume + 4K surface
    {x} ← Outside({x}, G)  # 移除inside points
    
    for i in 1..100:  # 每个geometry配100个random dynamics
        {v} ← Sample(B^C)  # per-point random velocity
        for t in 0..2:  # τ = 2
            h_G({x_t}) ← VectorDistance({x_t}, G)
            {x_{t+1}} ← x_t + v·I_G(x_t)  # evolve
```

---

## 8. 实验结果：数字说话

5个工业benchmark：

| Benchmark | Mesh size | Variables | Train/Test |
|-----------|-----------|-----------|------------|
| DrivAerML | ~160M | Geometry | 100/20 |
| NASA-CRM | ~450K | Geo, Speed, AoA | 105/44 |
| AirCraft | ~330K | Geo, Speed, AoA, Slip | 100/50 |
| DTCHull | ~240K | Geo, Yaw angle | 100/20 |
| Car-Crash | ~1M | Impact angle | 100/30 |

**DrivAerML（200 epochs）的关键数据**：

| Train samples | From scratch | GeoPT | 提升 |
|---------------|--------------|-------|------|
| 20 | 0.206 | 0.148 | 28% |
| 60 | 0.112 | 0.091 | 19% |
| 100 | 0.093 | 0.075 | 19% |

**100个sample的scratch水平，GeoPT用20个就达到了**。相当于**5× data efficiency**。

Paper还claim：20-60% data reduction，2× convergence speedup。

---

## 9. 几个关键ablation给我build的intuition

### 9.1 Geometry-only conditioning 没用（Fig. 8a）

作者拿Hunyuan3D的VAE encoder提取geometry tokens，作为auxiliary feature塞进Transolver。Hunyuan3D能精准重建geometry（Fig.20），tokens质量很高。

**结果：几乎没帮助。**

**Intuition**：frozen geometry encoder学到的是static representation，缺dynamics awareness。**必须直接pre-train physics backbone，光conditioning不够**。这跟vision里frozen CLIP feature对dense prediction任务效果差不多的道理。

### 9.2 Diversity比quality重要（Fig. 10a）

ShapeNet-V1（13,463 shapes，low quality）vs V2（9,515 shapes，high quality）。

**V1多数情况更好。**

这跟LLM pre-training的intuition一致：**quantity和diversity通常胜过精心筛选的quality**。Physics这边可能更明显，因为geometry的intrinsic diversity本身有限。

### 9.3 Step number $\tau$的sweet spot（Fig. 10b）

- $\tau=0$：退化到static，无效
- $\tau=1$：已经显著提升
- $\tau=2$：默认，平衡
- $\tau=3,4$：累积discretization error，部分benchmark下降

**Intuition**：trajectory太短抓不到dynamics信息，太长则数值error累积。$\tau=2$是Goldilocks zone。

### 9.4 Dynamics-dependent correlations（Fig. 7c）

这组visualization超棒。同一个pre-trained model，喂不同的$V_S$会激活完全不同的correlation pattern：
- 零速度：退化到static geometry
- 60° crosswind：inclined correlation
- 高速：concentrated correlation

这本质上是**pre-trained model把$V_S$当prompt来retrieve对应的物理prior**。跟in-context learning的机制有异曲同工之妙。

---

## 10. Scalability：让physics ML终于能"scale up"

### Model size（Fig. 6a）

Transolver from scratch：8→16→32 layers出现bottleneck，因为limited data下overfit。
GeoPT：consistent improvement with model size。

**Pre-training起了regularization作用**，把hypothesis space约束到physics-aligned manifold，避免overfit。

### Data diversity（Fig. 6b）

- Geometry diversity比dynamics diversity重要
- 固定来流任务（DrivAerML）：6% dynamics就够
- 多variable任务（AirCraft）：更多dynamics trajectories显著帮助

**Intuition**：dynamics分布越广，下游任务越能被pre-training分布覆盖。

---

## 11. 跟我熟悉的工作的联系

### vs. MAE

MAE在input-aligned space工作，GeoPT在input严格低维空间工作。MAE的mask reconstruction学到的是input本身的structure，GeoPT需要学到"missing dimension如何与input交互"。这是新的SSL paradigm。

### vs. Flow Matching / Diffusion

GeoPT的trajectory supervision跟flow matching有共鸣——都在velocity field下evolve particles。但方向不同：
- Flow matching：**learn** velocity field来transport分布
- GeoPT：**given** random velocity，**predict** resulting geometry feature trajectory

某种意义上，GeoPT学的是"**在任意dynamics下geometry feature如何演化**"的universal function。这恰好是PDE solver要做的。

### vs. PINN

PINN把PDE residual作为loss，需要知道governing equation。GeoPT完全反过来，**不假设任何具体PDE**，只假设最弱的mass conservation + boundary interaction。结果反而更generalizable。这让我想到LLM里"少即是多"的现象——弱prior + 大数据，往往比强prior + 小数据更scale。

### vs. Physics Foundation Models (Poseidon, DPOT, P3D)

这些都在expensive simulation data上scale。GeoPT是**第一个用off-the-shelf geometry做scaling的**，真正解除了label bottleneck。这是physics ML向LLM路线靠拢的关键一步。

---

## 12. 我对limitations的看法

Paper Section H提到几个limitation，我加几个自己的思考：

### 作者提的
1. **Material property parameterization**：crash里elastic和strength两种property压到单一speed
2. **Regular grid simulation**（如3D turbulence）：没complex geometry boundary

### 我加的
3. **Time-dependent simulation**：当前只steady-state。Transient需要更长trajectory，可能$\tau$要大很多，discretization error会更严重
4. **Multi-physics coupling**：fluid-structure interaction这种，单一velocity field可能不够，需要multi-field dynamics
5. **Pre-training vs downstream geometry gap**：AirCraft的distraction问题说明，pre-training分布和downstream分布差距太大时会hurt。可能需要curriculum或domain-adaptive pre-training
6. **Velocity field的表达能力**：随机i.i.d. per-point velocity比较简单，复杂physics（如vortex shedding）可能需要structured velocity field（比如divergence-free）
7. **$\tau$的长度限制**：长trajectory的discretization error说明这个framework难scale到long-horizon dynamics

---

## 13. 可能的extension

1. **Latent dynamics pre-training**：用VAE学到latent dynamics，避免显式trajectory
2. **Multi-resolution geometry**：coarse-to-fine hierarchical pre-training
3. **Active sampling of velocity**：用uncertainty sampling选informative velocity fields
4. **结合Physics-Informed loss**：pre-training后期加入weak physics constraint
5. **Cross-modal transfer**：从geometry到general 3D field (temperature, EM, acoustic)
6. **Divergence-free velocity sampling**：让随机velocity更物理（不可压缩流）
7. **Velocity field的hierarchical structure**：从simple translation到rotation到deformation，curriculum学习

---

## 14. 最后的大图景

GeoPT让我看到physics ML终于摸到了LLM的scaling paradigm边：

| 维度 | LLM | 传统Physics ML | GeoPT |
|------|-----|---------------|-------|
| 数据 | web text（免费） | solver-generated（昂贵） | off-the-shelf geometry（免费） |
| Pre-training目标 | next-token | task-specific | lifted dynamics-aware |
| Scaling | data + model | 受限于solver | data + model（geometry驱动） |
| Downstream | fine-tune | from scratch | fine-tune with dynamics prompt |

**核心insight**：当你的pre-training data空间严格小于downstream task空间时，必须**主动augment pre-training数据到richer space**，否则SSL会collapse。这个principle可能适用于其他modality mismatch场景，比如video→audio、text→action、image→depth。

GeoPT的"lifting"思想，可能是一个新的SSL paradigm的开端：**pre-training不再受限于native space reconstruction，可以通过合成"假condition"把数据空间撑大到与downstream对齐**。

这个idea对generative model、world model、具身智能pre-training都可能inspirational。

### References

- GeoPT paper: [arXiv:2509.25788](https://arxiv.org/abs/2509.25788)
- Code: [github.com/Physics-Scaling/GeoPT](https://github.com/Physics-Scaling/GeoPT)
- Transolver backbone: [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)
- DrivAerML dataset: [arXiv:2408.11969](https://arxiv.org/abs/2408.11969)
- ShapeNet: [arXiv:1512.03012](https://arxiv.org/abs/1512.03012)
- MAE: [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Hunyuan3D: [huggingface.co/tencent/Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2)
- Transport equation theory (Evans PDE book): [AMS](https://www.ams.org/books/gsm/019/)
- OpenFOAM: [openfoam.com](https://www.openfoam.com/)
- OpenRadioss: [openradioss.org](https://www.openradioss.org/)
- FCPW (ray-triangle): [github.com/rohan-sawhney/fcpw](https://github.com/rohan-sawhney/fcpw)
- Vector distance functions: [Faugeras & Gomes 2000](https://link.springer.com/chapter/10.1007/978-1-4471-0499-8_17)

要继续聊哪个部分都行——比如Transolver的内部架构、某个benchmark的具体物理意义、或者transport equation的严格proof。

---

# GeoPT: 用 Lifted Geometric Pre-Training 缩放物理仿真

这篇paper由MIT和清华大学团队联合完成（Wu, Guo, Li, Dou, Long, He, Matusik），核心思想是**用纯geometry数据通过合成dynamics做self-supervised pre-training，从而bridge geometry-physics gap**，让neural simulator能scale。下面我从intuition到implementation一层层拆解。

---

## 1. 核心问题：为什么 geometry-only pre-training 会失败

### 1.1 工业仿真数据的瓶颈

工业级CFD仿真（如DrivAerML汽车外气动数据集）单个样本的生成成本是 **6.1×10^4 CPU-hours**，这是个天文数字。对比vision/language的web-scale数据，neural simulator受限于supervised learning on solver-generated data。作者发现geometry本身是廉价的、abundant的（ShapeNet, Objaverse），所以自然想法是：**先用geometry做pre-training，再用少量physics labels fine-tune**。

### 1.2 Native pre-training 的失败现象

最naive的approach是直接在geometry空间做self-supervision：

$$\mathcal{L}_{\mathrm{native}}^{\mathrm{pre}} = \mathbb{E}_{\mathbf{x}, G}\left[ \| \mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G) - \pmb{h}_G(\mathbf{x}) \|_2^2 \right]$$

这里：
- $\mathbf{x}$：query position，$\mathbb{R}^C$中的mesh点
- $G$：geometry，从空间$\mathcal{G}$采样
- $h_G(\mathbf{x})$：geometry-derived feature，如SDF或vector distance
- $\mathcal{F}_{\widehat{\theta}}$：待pre-train的neural simulator

**实验结果反直觉**：这种native pre-training不仅没帮到下游任务，反而**degrade**了performance（Fig.1）。

### 1.3 失败的根源：geometry-physics gap

作者通过visualization来解释（Fig.3a）。他们可视化Transolver学到的aggregation weights，发现：

- **Geometry-only supervision**：把车的前后volume分到同一个token，左右asymmetric的pattern → 模型按static shape cue分组
- **Physics supervision**：前后asymmetric、左右symmetric → 对应气动流场结构

**Intuition**：下游physics任务的解空间$\mathcal{G} \times \mathcal{S} \to \mathcal{U}$比pre-training的$\mathcal{G} \to \mathcal{H}$严格richer。当pre-training data space与downstream task space不对齐时，native-space SSL方法会collapse，因为uncovered factor（dynamics）在native space里完全random。

这点跟MAE/contrastive learning的根本差异在于：MAE做的是input-aligned task（图像→图像），而这里是 **input-strictly-lower-dimensional**的任务（geometry→physics）。这一点我觉得是paper最关键的insight之一。

---

## 2. 核心方法：Lifted Geometric Pre-Training

### 2.1 关键idea：用合成dynamics把geometry"提升"

作者的解决方案很巧妙——既然真实dynamics $v_S$ 太贵，那就**随机合成dynamics**作为proxy。考虑粒子运动：

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \mathbf{v} \cdot \mathbb{1}_G(\mathbf{x}_t), \quad \mathbf{x}_0 = \mathbf{x}, \quad \mathbf{v} \sim \mathrm{Unif}(\mathbb{B}^C)$$

变量解释：
- $\mathbf{x}_t$：粒子在time $t$ 的位置
- $\mathbf{v}$：每个粒子i.i.d.采样的速度向量，从bounded ball $\mathbb{B}^C = \{\mathbf{v} \in \mathbb{R}^C : \|\mathbf{v}\|_2 \leq v_{\max}\}$
- $\mathbb{1}_G(\cdot)$：indicator function，**inside或on boundary返回0，outside返回1** —— 这个设计让粒子hit到geometry surface时stuck住，模拟boundary interaction
- $v_{\max}$：速度上限，论文里设为2

这个formulation编码两个关键结构：
1. **Velocity field $v_S$ 耦合空间远点**：不同initial position的trajectories可能intersect，造成这些点share correlated physical responses
2. **Indicator $\mathbb{1}_G$ 在boundary halt**：physical response由boundary interaction塑造（气动中的表面压力、crash中的接触力、radiosity中的光传输）

### 2.2 Lifted self-supervision target

监督信号变成trajectory of geometric features：

$$h_G(\mathbf{x}_{0:\tau}) = \{h_G(\mathbf{x}_t)\}_{t=0}^{\tau} \in \mathcal{H}_{\mathrm{traj}}$$

- $\tau$：time horizon，论文默认$\tau=2$（3 steps包括$t=0$）
- $h_G(\cdot)$：vector distance field（比SDF更好，因为同时编码距离和方向）

完整pre-training loss：

$$\boxed{\mathcal{L}_{\mathrm{lifted}}^{\mathrm{pre}} = \mathbb{E}_{\mathbf{x}, G, V}\left[ \left\| \mathcal{F}_{\widehat{\theta}}(\mathbf{x}; G, V) - \pmb{h}_G(\mathbf{x}_{0:\tau}) \right\|_2^2 \right]}$$

期望对三个随机变量：
- $G$：category-balanced sampling的geometry
- $\mathbf{x}$：从surrounding volume $\Omega_G$ 和geometry boundary $G$同时采样
- $V$：per-point velocity，uniform from $\mathbb{B}^C$

### 2.3 Lifting的几何视角

Paper里那张inset diagram很优雅：

```
(G, V) ──lifted pretrain──> H_traj
  ↑ lifting                  ↓ slice
  G    ──native pretrain──>  H
```

- 纵向$\mathcal{G} \to (\mathcal{G}, \mathcal{V})$：lifting操作（augment with random velocity）
- 上箭头$(\mathcal{G}, \mathcal{V}) \to \mathcal{H}_{\mathrm{traj}}$：lifted pre-training task
- 下箭头$\mathcal{G} \to \mathcal{H}$：native pre-training（degenerate case）
- 虚线$\mathcal{H}_{\mathrm{traj}} \to \mathcal{H}$：slicing（取$t=0$）

**关键observation**：downstream simulation也在joint space $(\mathcal{G}, \mathcal{V})$上操作，所以lifted representation直接transfer。

### 2.4 Fine-tuning：用real dynamics"prompt"模型

Fine-tuning时把随机velocity换成task-specific $V_S$：

$$\mathcal{L}^{\mathrm{fine}} = \mathbb{E}_{\mathbf{x}, G, V_S}\left[ \left\| \mathcal{F}_{\theta}(\mathbf{x}; G, V_S) - \mathbf{u}(\mathbf{x}) \right\|_2^2 \right]$$

不同physics task配置$V_S$：
- **Aerodynamics**（DrivAerML, NASA-CRM, AirCraft）：direction对齐incoming flow，magnitude=freestream speed
- **Hydrodynamics**（DTCHull）：water/air两相分别配$V_S$，反映两相流
- **Crash simulation**（Car-Crash）：direction对齐impact，spatially decaying magnitude from collision point
- **Radiosity**（Appendix A扩展）：direction=light propagation

这种统一接口让**一个pre-trained model适应diverse physics**，仅通过reconfigure velocity input。这点我觉得跟ControlNet的conditioning思想有异曲同工之妙。

---

## 3. 理论解释：Transport Equation视角

### 3.1 Phase-space transport equation

Paper的Appendix B证明，tracking moving particles等价于解一个**带sticking boundary的collisionless transport equation**：

$$\partial_t f(x, v, t) + v \cdot \nabla_x f(x, v, t) = 0, \quad (x, v) \in \Omega \times \mathbb{R}^C$$

- $f(x, v, t)$：phase-space density，位置$x$、速度$v$、时间$t$
- $\partial_t$：对时间偏导
- $v \cdot \nabla_x$：速度$v$与空间梯度$\nabla_x$的内积，即advection operator

特征曲线$\dot{\mathbf{x}}(t) = \mathbf{v}, \dot{\mathbf{v}}(t) = 0$对应particle trajectories。

### 3.2 Sticking boundary

Boundary accumulation：

$$\partial_t f_G(x, v, t) = (v \cdot n(x))_+ f(x, v, t) dv, \quad x \in G$$

- $f_G$：boundary-supported phase-space density
- $n(x)$：outward unit normal
- $(v \cdot n(x))_+$：只有$v \cdot n > 0$的粒子能hit boundary

### 3.3 Mass conservation

**Proposition B.1**：

$$\frac{d}{dt}\left( \int_{\Omega}\int_{\mathbb{R}^d} f(x,v,t)dvdx + \int_G \int_{\mathbb{R}^d} f_G(x,v,t)dv dS(x) \right) = 0$$

总质量守恒——内部粒子只是被transfer到boundary。

### 3.4 为什么这个prior有generalizability

**Remark B.3**的核心intuition：transport equation捕获**canonical conservation-law structure**——mass沿characteristics advection + boundary flux accounting。Boltzmann方程加collision operator，Navier-Stokes加momentum和constitutive relations。所以pre-training signal虽然简化，但**inductive bias toward characteristic-driven correlations和boundary interactions**是broad continuum/kinetic systems共享的。

这个理论解释对build intuition非常重要：**GeoPT学的不是某个具体物理，而是所有PDE共享的"传输+边界"骨架**。

---

## 4. 实现细节

### 4.1 Backbone

用**Transolver**作为default backbone，architecture-agnostic。三个model size：
- Base：8 layers, 3M params
- Large：16 layers, 7M params
- Huge：32 layers, 15M params
- 都用256 hidden channels, 32 state tokens

注意15M params在neural simulator领域已经算大model了。

### 4.2 Pre-training data

- ShapeNet-V1（cars 4,045 + airplanes 1,939 + watercraft，约13,463 shapes）
- 每个geometry采32,768 volume points + 4,096 surface points
- Per-point velocity从radius=2的ball采样
- 每个geometry生成100 random dynamics fields
- **总计1,346,300 samples，约5TB**
- 80 CPU cores 3天完成

### 4.3 计算效率

Tracking 36,864 points in one sample约0.2s（80 CPU cores），比工业CFD快 **10^7倍**。这数据生成cost几乎可忽略，这是paper的key scalability argument。

### 4.4 Algorithm 1 数据生成

```python
for G in G:
    Normalize(G)  # rotate, shift, scale to unit
    Build FCPW scene  # ray-triangle intersection加速
    {x} ← Sample(Ω_G ∪ ∂G)  # 32K volume + 4K surface
    {x} ← Outside({x}, G)  # 移除inside points
    for i in 1..N_dyn:  # N_dyn = 100
        {v} ← Sample(B^C)  # per-point i.i.d.
        for t in 0..τ:  # τ = 2
            h_G({x_t}) ← VectorDistance({x_t}, G)
            Append h_G({x_t}) to trajectory
            {x_{t+1}} ← Evolve({x_t}, {v}, G)
                # x_{t+1} = x_t + v·I_G(x_t)
```

### 4.5 训练配置

- Pre-training：AdamW + cosine annealing，200 epochs，lr=1e-3
- Fine-tuning：AdamW + OneCycleLR，200 epochs
- Pre-training GPU hours：144 (B) / 360 (L) / 576 (H)
- Fine-tuning GPU hours：3 (B) / 6 (L) / 10 (H)

---

## 5. 实验结果分析

### 5.1 五个工业级benchmark

| Task | Benchmark | #Mesh | Variables | Size |
|------|-----------|-------|-----------|------|
| Aerodynamics | DrivAerML | ~160M | Geometry | ~6TB |
| Aerodynamics | NASA-CRM | ~450K | Geo, Speed, AoA | ~3GB |
| Aerodynamics | AirCraft | ~330K | Geo, Speed, AoA, Slip | ~7GB |
| Hydrodynamics | DTCHull | ~240K | Geo, Yaw angle | ~2GB |
| Crash | Car-Crash | ~1M | Impact angle | ~8GB |

每个benchmark用~100 training samples（mimic工业实践），test 20-50 samples。

### 5.2 主要发现

**Fig. 5的三个observations**：

**(i) 减少data需求20-60%**
- DTCHull：减少60%（geometric variability大，pre-training收益最大）
- NASA-CRM：改善适度（geometry variations只是aileron angle局部变化）
- AirCraft：显著改善（4 variables，最复杂）

**(ii) 改善geometry generalization**
Geometry diversity越高的任务，GeoPT收益越大。这跟pre-training时见过的geometry distribution覆盖有关。

**(iii) 支持surface-only simulation**
Car-Crash虽然是surface-only，但通过配置decayed velocity field仍能adapt，证明stochastic pre-training dynamics的robustness。

### 5.3 Scalability

**Fig. 6a Model size scaling**：
- Transolver from scratch：从8→32 layers出现**scaling bottleneck**，因为limited data下overfitting
- GeoPT：consistent improvement with model size，pre-training起到regularization作用

**Fig. 6b Data diversity**：
- Geometry diversity比dynamics diversity更重要
- DrivAerML固定incoming flow：6% dynamics就够
- AirCraft多variables：更多dynamics trajectories显著提升

### 5.4 关键ablation：Geometry usage（Fig. 8a）

四个对比：
1. From scratch（最差）
2. Geometry-only conditioning（Hunyuan3D VAE tokens作为auxiliary feature）—— **几乎没用**
3. Geometry-only pre-training（vector distance）—— 比scratch略差或持平
4. GeoPT lifted pre-training —— 最好

**关键insight**：frozen geometry encoder（Hunyuan3D）虽然能精准重建geometry（Fig.20），但**static representation不能help physics learning**。必须直接pre-train physics backbone本身，而且必须用lifted dynamics。

### 5.5 Backbone comparison（Fig. 8b）

GeoPT在Galerkin Transformer, GNOT, UPT, Transolver, Transolver++上都consistent improvement，证明architecture-agnostic。Transolver本身在多数benchmark上最强，所以选它作为default。

### 5.6 Dynamics-dependent correlations（Fig. 7c）

零speed时模型degenerate到static geometry correlation。不同$V_S$激活不同pattern：
- Crosswind（60° shifted）：inclined correlation
- High speed：concentrated correlation
- 这本质上是个**"prompting"机制**——dynamics condition像prompt一样激活pre-trained knowledge

### 5.7 Radiosity generalization（Appendix A）

在Cornell box + Stanford bunny上fine-tune，MAE 9.0×10^-2 vs scratch 9.7×10^-2。**重点是pre-training时既没见过Cornell box geometry也没见过light transport physics**，仍然transfer成功。这验证了Remark B.3的理论——transport prior确实generalize。

### 5.8 Quantitative results（Tables 6-10）

我highlight几个最impressive的数字（DrivAerML, 200 epochs）：

| Data | Scratch | GeoPT | Improvement |
|------|---------|-------|-------------|
| 20 | 0.206 | 0.148 | 28% |
| 60 | 0.112 | 0.091 | 19% |
| 100 | 0.093 | 0.075 | 19% |

100 samples时20%数据就能match scratch 100 samples的性能，意味着数据效率5×。

---

## 6. Ablation Studies深度分析

### 6.1 Single vs Mixed geometry dataset（Fig. 10a）

- Mixed（car+plane+watercraft）多数benchmark更好
- AirCraft例外：single-subset更好，因为training geometry和pre-training geometry差异大造成distraction
- **启示**：扩大pre-training geometry diversity能缓解distraction

### 6.2 ShapeNet-V1 vs V2（Fig. 10a）

V1：13,463 shapes，low quality但high diversity
V2：9,515 shapes，high quality但low diversity

**结果**：V1多数情况更好，**diversity比quality更重要**。这跟LLM pre-training的发现一致——data quantity和diversity通常胜过精心筛选。

### 6.3 Step number $\tau$（Fig. 10b）

- $\tau=0$：degenerate到static，无效
- $\tau=1$：显著提升
- $\tau=2$：default，平衡
- $\tau=3, 4$：discretization error累积，部分benchmark下降

**Intuition**：太短的trajectory抓不到足够dynamics information，太长则accumulated error。

### 6.4 Vector distance vs SDF（Fig. 10c）

Vector distance更好，因为同时包含**距离和方向**信息。SDF是标量，丢失了"closest point在哪"的信息。

### 6.5 Fine-tuning dynamics configuration（Fig. 12, 13）

**Direction shift**：直接用incoming flow/impact direction最好。Shift越大drop越严重，high-speed任务（NASA-CRM, AirCraft）对direction shift更sensitive。

**Speed configuration**：
- Low-speed（车、船）：norm 0.3左右
- High-speed（飞机）：[1.0, 2.0]区间

**Recipe**：低速度任务norm在[0.1, 1.0]，高速度在[1.0, 2.0]。

---

## 7. 个人intuition building & 联想

### 7.1 跟MAE的本质区别

MAE在input-aligned space（图像→图像重建），所以masked reconstruction能学到有用representation。GeoPT面对的是**input-strictly-lower-dimensional**任务，native reconstruction学到的是reduced representation，反而有害。这跟vision-language pre-training里CLIP的双encoder对齐也是不同问题——CLIP是两个rich space对齐，GeoPT是poor spacerich space。

### 7.2 跟Diffusion/Flow Matching的联系

GeoPT的trajectory supervision跟flow matching的核心思想有共鸣——都是在velocity field下evolve particles。区别在于：
- Flow matching：learn velocity field to transport simple→complex distribution
- GeoPT：**given** random velocity, predict resulting trajectory of geometry features

某种意义上，GeoPT学的是"**在任意dynamics下geometry feature如何演化**"的universal function，这恰是PDE solver要做的。

### 7.3 跟Physics-Informed Neural Networks (PINN)的对比

PINN把PDE residual作为loss，需要知道governing equation。GeoPT完全反过来：**不假设任何具体PDE**，只假设mass conservation + boundary interaction这种**最弱common prior**。这样反而更generalizable，能跨physics domain transfer。

### 7.4 跟Foundation Model的scaling思路对比

现有physics foundation models（Poseidon, DPOT, Unisolver, P3D）都靠**expensive simulation data**做scaling。GeoPT是第一个用**off-the-shelf geometry**做scaling的，真正解除了label bottleneck。这跟LLM用web text做next-token prediction的策略类似——**用免费数据学习universal prior**。

### 7.5 Limitations思考

Paper Section H提到的limitation：
1. **Material property parameterization**：crash里elastic和strength两种property压到单一speed，loss distinguishability
2. **Regular grid simulation**（如3D turbulence）：没有complex geometry boundary，目前方法不直接适用

我的额外思考：
3. **Time-dependent simulation**：当前只考虑steady-state。Transient simulation需要更长trajectory，可能$\tau$要大很多
4. **Multi-physics coupling**：如fluid-structure interaction，单一velocity field可能不够
5. **Pre-training geometry与downstream gap**：如AirCraft的distraction问题，需要curriculum或domain-adaptive pre-training

### 7.6 跟我熟悉的工作的联系

我之前在multiple LLM pre-training相关work里讨论过scaling law的瓶颈。GeoPT的发现"geometry diversity比quality更重要"跟Chinchilla之后大家发现data quality scaling不一定linear的发现形成有趣对比。可能因为geometry的"intrinsic diversity"本身有限，所以quantity更关键。

跟Transolver的aggregation weight visualization让我想到attention map interpretability问题——可视化physical states learned是验证inductive bias的好工具，应该在更多physics ML work里推广。

### 7.7 可能的extension

1. **Latent dynamics pre-training**：不用显式trajectory，用VAE-encoder学到latent dynamics
2. **Multi-resolution geometry**：coarse-to-fine hierarchical pre-training
3. **Active sampling of velocity**：用uncertainty sampling选informative velocity fields
4. **结合Physics-Informed loss**：pre-training后期加入weak physics constraint
5. **Cross-modal transfer**：从geometry到general 3D field prediction (temperature, EM, etc.)

---

## 8. 总结

GeoPT的核心贡献：
1. **识别geometry-physics gap**：native-space SSL在input-strictly-lower-dimensional任务上会collapse
2. **Lifted geometric pre-training**：用synthetic dynamics把pre-training空间从$\mathcal{G}$提升到$(\mathcal{G}, \mathcal{V})$
3. **理论解释**：tracking particles ⟺ solving collisionless transport equation with sticking boundary，这个prior对broad PDE class共享
4. **工业级验证**：5个benchmark，20-60% data reduction，2× convergence speedup
5. **Scalability demonstration**：model size和数据diversity都scale

**最重要的intuition**：pre-training space必须cover downstream task space的所有variation source，否则uncovered factor会造成negative transfer。这个principle可能适用于其他modality mismatch的pre-training场景。

### Reference Links

- Paper arXiv: [https://arxiv.org/abs/2509.25788](https://arxiv.org/abs/2509.25788) (注：实际arXiv number需确认，paper里Zhang et al. 2025提到此号)
- Code: [https://github.com/Physics-Scaling/GeoPT](https://github.com/Physics-Scaling/GeoPT)
- Transolver (backbone): [Wu et al. 2024 ICML](https://arxiv.org/abs/2402.02366)
- DrivAerML: [Ashton et al. 2024](https://arxiv.org/abs/2408.11969)
- ShapeNet: [Chang et al. 2015](https://arxiv.org/abs/1512.03012)
- Hunyuan3D: [Tencent 2025](https://huggingface.co/tencent/Hunyuan3D-2)
- MAE (related SSL): [He et al. 2022 CVPR](https://arxiv.org/abs/2111.06377)
- FCPW (ray-triangle intersection): [Sawhney 2021](https://github.com/rohan-sawhney/fcpw)
- Vector distance functions: [Faugeras & Gomes 2000](https://link.springer.com/chapter/10.1007/978-1-4471-0499-8_17)
- OpenFOAM: [Jasak 2009](https://www.openfoam.com/)
- OpenRadioss: [Altair Engineering](https://www.openradioss.org/)

如果你想深入某个具体方面（比如transolver架构细节、某个benchmark的具体设置、或theoretical analysis的严格proof），我可以继续展开。
