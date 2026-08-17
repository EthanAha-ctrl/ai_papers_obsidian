---
source_pdf: Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene.pdf
paper_sha256: f5708e8ffec2ce8243542fc7aa0922a468b0c845950b0554863e4c61ba70b854
processed_at: '2026-08-03T19:12:10-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话总结

3D-GS本来只能拍静态照片,这群人给它加了个"时间机器",让它能拍动态视频,而且画质吊打之前所有方法,还能实时渲染。

---

## 为什么要做这个事

先说背景。3D-GS是2023年SIGGRAPH上炸场子的工作,把一堆3D Gaussian椭球往图像上"splat"(甩上去),通过alpha blending拼出最终图像。速度飞快,100+ FPS,质量还超过Mip-NeRF。大家一看就疯了,这是neural rendering的holy grail啊。

但3D-GS有个尴尬的地方:它只能处理静态场景。你给它一堆照片,它重建出一个时刻的3D世界。你问它"这个人站起来之前是什么样子",它答不上来。

所以很自然的想法是:能不能把3D-GS扩展到dynamic scene,既能做novel view synthesis,又能做time interpolation(时间维度的插值,比如两个关键帧之间生成中间帧)?

这就是这篇paper要干的事。

---

## 最naive的思路为什么不行

你可能会想,这还不简单?每个时间点单独训一组3D-GS不就完了?

确实可以,但只适用于MVS(multi-view stereo)那种采集方式——每个时刻你都有多台相机从不同角度拍。问题是大多数real-world场景是monocular capture,就是一个相机在动,同时场景也在动。每个时刻你只有一个视角的信息,根本没法单独重建那个时刻的3D世界。

另一个naive思路是给每个Gaussian加个时间参数t,让它自己学。但这样就把时间和空间couple在一起了,丢失了"同一个物体在不同时刻的运动是相关的"这个prior。而且3D-GS的CUDA rasterizer是精心设计过的,你给每个Gaussian塞一堆time-dependent参数,既破坏了pipeline的物理含义,又容易overfit。

---

## 他们的核心idea

非常优雅的decoupling:

**把世界分成两层**

1. **Canonical space(参考空间)**:这里有一组静态的3D Gaussians,代表场景的"基础形态"。这组Gaussians不依赖时间,就是一堆位置、大小、朝向、颜色、透明度。

2. **Deformation field(形变场)**:一个小MLP网络,输入是(某个Gaussian的位置x, 时间t),输出是(x的位移δx, 朝向变化δr, 大小变化δs)。

然后你拿canonical space的Gaussian,加上deformation field告诉你的offset,得到"在这个时刻t,这个Gaussian应该在什么位置、什么形状",再送去rasterize成图像。

这个idea的intuition是:场景里大多数运动是structured的。一个人站起来,他的身体各部分运动是有correlation的,不是随机noise。用一个MLP去建模(x, t)→(δx, δr, δs)的映射,MLP的smoothness天然就给了你motion的连续性和结构化prior。

---

## 几个关键trick

### Stop-gradient的妙用

公式里有个细节特别重要:deformation MLP的输入是 $\gamma(\text{sg}(\mathbf{x}))$,这个sg是stop-gradient。

意思是:Gaussian的位置x喂给MLP的时候,梯度不从这条路回流。MLP只能"看"Gaussian在哪,不能通过反传梯度去"改"Gaussian的canonical position。

为什么这么设计?如果允许MLP改canonical position,它就会偷懒——比如把canonical点全挪到时间t=0的position,然后deformation field输出0,啥也不用学。或者更微妙的degenerate solution,deformation field和canonical Gaussians互相"作弊",一起找到一个local minimum但语义完全错误。

stop-gradient强制一种role分工:canonical Gaussians是"事实",由rasterization loss直接优化;deformation MLP是"解释器",只能读canonical事实,把不同时刻的观测解释成deformation。这种asymmetry让两个组件各司其职。

### Warm-up

前3000 iteration只训Gaussians,不训deformation field。

intuition是:如果一开始就joint optimize,Gaussians还在到处乱跑找自己的位置,deformation MLP去学一个"moving target",很容易卡在bad local minimum。让Gaussians先稳定下来,deformation MLP再去学怎么map,效率高很多。

### 为什么用pure MLP而不用grid

paper里反复强调一个rank argument。static scene的rank低,所以K-Planes、HexPlane这种low-rank decomposition工作得很好。但dynamic scene的rank高——每个空间点在不同时刻有不同状态。而explicit point-based rendering又进一步elevate了rank,因为每个Gaussian是完全独立的。

如果你用grid-based deformation field,low-rank assumption会限制你能表达的运动复杂度上限。MLP虽然慢,但能express任意high-rank function。这是他们做design choice的核心理由。

---

## AST——最clever的设计

### 解决什么问题

real-world dataset有个通病:COLMAP估计的camera pose不准确。在dynamic scene里尤其严重,因为场景在动,SfM的feature matching更容易出错。

implicit representation(MLP)有inherent smoothness,小偏差被自然平滑掉,不太影响结果。但explicit point-based rendering是"硬"的,每个Gaussian对应一个具体位置,pose差一点点,Gaussian就会被放到错误位置,在time interpolation任务中表现为jitter(抖动)。

注意:novel view synthesis在固定时刻t不受影响,因为同一时刻的spatial consistency由图像本身保证。问题只出在time维度——不同时刻的scene稍有misalignment,deformation field被这种misalignment带偏。

### 怎么做的

给时间编码 $\gamma(t)$ 加Gaussian noise,noise的amplitude随训练iteration线性衰减:

$$\mathcal{X}(i) = \mathcal{N}(0,1) \cdot \beta \cdot \Delta t \cdot (1 - i/\tau)$$

训练早期noise大,deformation MLP看到的时间编码被扰动,无法精确overfit到某个特定frame的pose error,被强制学习时间维度上的smooth mapping。训练后期noise衰减到0,模型可以refine到真实时间细节。

### 为什么比smoothness loss好

之前的D-NeRF和Tensor4D用smoothness loss,就是额外forward pass算时间维度的consistency loss,有computational overhead。AST只在input端加noise,forward pass完全不变,**zero额外计算开销**。

而且smoothness loss是硬约束,容易把dynamic detail也smooth掉。AST的annealing机制让早期平滑、后期refine,既能抑制pose noise的影响,又能保留high-frequency motion detail。

---

## 实验结果怎么说

### Synthetic data (D-NeRF dataset)

PSNR从~32提到~40,这是huge jump。主要因为explicit representation的capacity远超implicit,3D-GS在static scene本来就有这个优势,现在延伸到dynamic。

### Real-world data (NeRF-DS)

提升幅度小很多,PSNR从23.60到24.11。因为real-world pose noise对explicit方法不友好。AST在这里贡献明显,从23.97到24.11。

HyperNeRF dataset他们干脆不做quantitative comparison了,因为pose太差,blurry output反而PSNR更高,clear output PSNR低但visually更好。这是evaluation metric的缺陷。

### Speed

Gaussians少于250k时30+ FPS实时渲染。HyperNeRF的Gaussians爆炸到1.3M导致只有6 FPS,反过来说明pose准确性对efficiency极其重要——pose不准,系统需要更多Gaussians去"补偿"错误。

---

## Ablation里的有意思发现

### 不输出opacity offset

deformation MLP不输出 $\delta\sigma$,opacity在canonical space保持不变。实验证明这是最优的。intuition是opacity是个"离散"属性——物体要么存在要么不存在,不应该随时间平滑变化。而position、rotation、scaling是连续geometry属性,deformation合理。

### 给MLP输入rotation和scaling反而更差

counter-intuitive:让deformation MLP看到Gaussian的rotation和scaling,结果更差。直觉上更多信息应该更好。但加r和s会让MLP更容易overfit到specific Gaussians,破坏了stop-gradient设计的isolation意图。canonical position x已经通过空间位置间接编码了Gaussian的identity,不需要额外的identity信息。

### SE(3) field不划算

参考Nerfies,试了用SE(3) field约束position transformation。synthetic data上微弱提升,real-world反而下降,而且训练时间增加50%,FPS下降20%。trade-off不划算。real-world的拓扑变化和非刚性motion比SE(3)能表达的更复杂。

---

## 局限性

1. **Inaccurate pose致命**:pose不准时Gaussians数量爆炸,无法real-time。这是explicit方法的intrinsic limitation。

2. **Few viewpoints overfit**:viewpoint diversity比绝对图像数更重要。100张图但只有4个视角,严重overfit;swap train/test后(100张图但视角多样)结果大幅改善。

3. **复杂人体motion未验证**:subtle facial expression这类high-frequency motion可能超出MLP的capacity。

---

## 我的take

这篇paper的position很巧妙:它抓住了3D-GS刚出来大家最想做的事(dynamic extension),用了一个clean的formulation(canonical + deformation),还加了个clever的engineering trick(AST)解决real-world的pose noise问题。

stop-gradient和AST这两个design choice显示作者对optimization dynamics有很深的理解。pure MLP的选择基于rank argument,虽然slow但principled。

作为开山之作,这套canonical space + deformation MLP + AST成了后续大量4D Gaussian splatting工作的baseline范式。后续工作多在deformation field上做文章——用grid代替MLP提速,加物理约束,做human-specific extension等等。

核心insight就是:**把动态场景的complexity分解为静态structure + 时间deformation,用explicit representation保capacity,用MLP保smoothness,用noise injection保robustness**。这个recipe在工程上work得很好,也为后续工作奠定了pattern。

---

# Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction 深度讲解

这篇paper由Zhejiang University和ByteDance Inc.的团队完成,发表于2023年,是将3D-GS(Kerbl et al., SIGGRAPH 2023)从静态场景扩展到monocular dynamic scene reconstruction的早期工作。核心idea非常清晰:在canonical space学习3D Gaussians,用一个pure MLP deformation field建模Gaussians随时间的形变。

项目主页与代码:https://github.com/ingra14m/Deformable-3D-Gaussians

---

## 1. 问题背景与动机

### 1.1 3D-GS的扩展难题

原始3D-GS [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]通过CUDA实现的高效differentiable Gaussian rasterization达到了real-time rendering(>100 FPS),但是其CUDA pipeline高度customized,scalability较差,直接套用到dynamic scene非trivial。

直接的naive思路有两个:
- **Entangled approach**: 把时间t作为Gaussian的额外可学习参数 → 违背rasterization pipeline的物理含义,且丢失spatiotemporal continuity
- **Per-frame training**: 每个时间步单独训练一组3D-GS,然后做post-hoc interpolation → 只适用于MVS捕获,对连续monocular capture无效

### 1.2 Dynamic scene的rank问题

paper明确指出了一个key insight:dynamic scenes比static scenes的rank更高。这就是为什么K-Planes [https://arxiv.org/abs/2301.10241]、HexPlane [https://arxiv.org/abs/2301.09602]、Tensor4D [https://arxiv.org/abs/2306.05392]等grid/plane-based方法虽然能加速,但在动态场景中存在quality ceiling。而explicit point-based rendering进一步**elevates the rank of the scene**。这就解释了为什么作者选择pure MLP而不是hybrid grid结构。

---

## 2. 方法核心:Canonical Space + Deformation Field

### 2.1 整体pipeline

pipeline分为三部分:
1. **Initialization**: 从COLMAP SfM得到的sparse points初始化3D Gaussians $G(\mathbf{x}, \mathbf{r}, \mathbf{s}, \sigma)$,其中x是center position,r是quaternion表示rotation,s是scaling vector,σ是opacity。appearance通过spherical harmonics (SH)表示。
2. **Deformation**: 给定time $t$和Gaussian center $\mathbf{x}$,deformation MLP $\mathcal{F}_\theta$输出offsets $(\delta\mathbf{x}, \delta\mathbf{r}, \delta\mathbf{s})$,得到deformed Gaussians $G(\mathbf{x}+\delta\mathbf{x}, \mathbf{r}+\delta\mathbf{r}, \mathbf{s}+\delta\mathbf{s}, \sigma)$。
3. **Rasterization**: deformed Gaussians送入differentiable Gaussian rasterization pipeline,通过α-blending得到2D image,与GT比较计算loss反传。

### 2.2 渲染公式解析

**公式(1) 投影2D covariance**:
$$\Sigma' = J V \Sigma V^T J^T$$
- $\Sigma'$: 投影到image plane后的2D covariance matrix
- $J$: projective transformation的affine近似下的Jacobian
- $V$: view matrix,从world坐标变换到camera坐标
- $\Sigma$: 3D空间中的covariance matrix

这个公式来自EWA splatting [Zwicker et al., 2001, https://dl.acm.org/doi/10.1109/VISUAL.2001.964382]。

**公式(2) 3D covariance分解**:
$$\Sigma = R S S^T R^T$$
- $R$: 从quaternion $\mathbf{r}$ 变换得到的rotation matrix
- $S$: 从scaling vector $\mathbf{s}$ 构造的diagonal scaling matrix

这种分解保证$\Sigma$是positive semi-definite,同时让网络只学习3+4=7个参数(r是4维quaternion,s是3维scaling)而不是6个独立covariance entries。

**公式(3) 体渲染公式**:
$$C(\mathbf{p}) = \sum_{i \in N} T_i \alpha_i c_i$$
$$\alpha_i = \sigma_i \exp\left(-\frac{1}{2}(\mathbf{p} - \mu_i)^T \Sigma'^{-1} (\mathbf{p} - \mu_i)\right)$$
- $C(\mathbf{p})$: pixel $\mathbf{p}$的颜色
- $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$: 前方Gaussians的transmittance
- $\alpha_i$: 第i个Gaussian在pixel $\mathbf{p}$处的opacity
- $c_i$: 第i个Gaussian的color(由SH计算)
- $\mu_i$: 3D Gaussian投影到2D的uv坐标

### 2.3 Deformation Field的数学表达

**公式(4)**:
$$(\delta\mathbf{x}, \delta\mathbf{r}, \delta\mathbf{s}) = \mathcal{F}_\theta(\gamma(\text{sg}(\mathbf{x})), \gamma(t))$$

关键设计点:
- **stop-gradient $\text{sg}(\cdot)$**: 这是这篇paper一个关键trick。对Gaussian position x做stop-gradient,意味着deformation MLP不能通过梯度反过来修改Gaussians的canonical position,只能修改自己的weights。这避免了deformation field和3D Gaussians相互"作弊"——例如deformation MLP直接把所有点平移到别处,而让canonical Gaussians待在错误位置上。stop-gradient强制canonical Gaussians稳定下来承担geometry,deformation field只负责motion。
- **Positional Encoding (PE)**: $\gamma(\cdot)$ 给input加上高频信息。
- **不输出 $\delta\sigma$**: opacity在canonical space保持不变,论文ablation证明了这个设计最优(详见后文)。

**公式(5) Positional Encoding**:
$$\gamma(p) = (\sin(2^k \pi p), \cos(2^k \pi p))_{k=0}^{L-1}$$
- $L=10$ for $\mathbf{x}$, $L=6$ for $t$ in synthetic scenes
- $L=10$ for both $\mathbf{x}$ and $t$ in real scenes

synthetic scene时间编码频率更低,因为合成场景时间步数有限且分布均匀;real scene需要更高频率时间编码来capture fast motion和长duration。

### 2.4 Deformation MLP架构

网络结构(附录B):
- 输入: $\gamma(\text{sg}(\mathbf{x}))$ 和 $\gamma(t)$ 拼接
- 8层全连接,ReLU激活,hidden dimension 256
- 第4层concatenate input features(类似NeRF [https://arxiv.org/abs/2003.08934])
- 输出256维feature
- 3个额外FC层(无激活),分别输出 $\delta\mathbf{x}$(3维), $\delta\mathbf{r}$(4维), $\delta\mathbf{s}$(3维)

存储开销:仅2MB,相比3D Gaussians本身可忽略。

### 2.5 Adaptive Density Control

跟随原3D-GS:
- **Pruning**: 删除opacity $\sigma$过小的Gaussians
- **Clone**: 在position gradient大且Gaussian小的区域clone
- **Split**: 在position gradient大且Gaussian大的区域split,新scale除以 $\xi=1.6$
- **Threshold**: $t_{pos} = 0.0002$

这是3D-GS能capture细节的关键。paper强调:在dynamic scene中,deformation field的梯度能传回Gaussians,在dynamic区域帮助Gaussians做更智能的densification。

---

## 3. Annealing Smooth Training (AST) - 关键创新

### 3.1 解决的问题

real-world dataset的pose估计(COLMAP)经常不准确,在dynamic scene中尤其严重。implicit representation的MLP inherent smoothness能掩盖这种小偏差,但explicit point-based rendering会**放大**这个effect,导致time interpolation任务中出现jitter。

这是一个微妙问题:在固定时间t做novel-view synthesis不受影响,因为同一时刻的spatial consistency由图像自身保证;但interpolating time时,不同时刻的scene稍有misalignment,deformation field会被这种misalignment带跑偏,在测试集上表现出jitter。

### 3.2 公式(6) AST机制

$$\Delta = \mathcal{F}_\theta(\gamma(\text{sg}(\mathbf{x})), \gamma(t) + \mathcal{X}(i))$$
$$\mathcal{X}(i) = \mathcal{N}(0,1) \cdot \beta \cdot \Delta t \cdot (1 - i/\tau)$$

- $\mathcal{X}(i)$: 第i次iteration加在 $\gamma(t)$上的Gaussian noise
- $\mathcal{N}(0,1)$: 标准正态分布sample
- $\beta = 0.1$: 缩放因子
- $\Delta t$: 平均时间间隔(数据集相关)
- $\tau = 20k$: annealing的threshold iteration
- $i$: 当前iteration

**直觉**: 
- 训练早期($i$小),noise大 → deformation field看到的时间编码被扰动,无法精确overfit到某个特定frame的pose error,被强制学习时间维度上的平滑映射
- 训练后期($i \to \tau$),noise线性衰减到0 → 模型可以refine到具体时间细节,preserve high-frequency motion
- 注意noise是加在 $\gamma(t)$上的(PE之后or之前?看公式是加在 $\gamma(t)$输出端,即PE feature上),所以扰动的是time的高频分量

**对比传统smoothness loss**: D-NeRF [https://arxiv.org/abs/2011.13961]和Tensor4D使用smoothness loss(在时间维度上对deformation加约束),需要额外forward pass和computation。AST只在input端加noise,**zero额外计算开销**。

---

## 4. Training策略与超参

### 4.1 两阶段训练

1. **Warm-up phase**: 前3k iterations只训练3D Gaussians,不优化deformation field。这让canonical Gaussians先有一个相对稳定的shape和position,避免deformation MLP在Gaussians还在剧烈变化时被误导。
2. **Joint training**: 3k-40k iterations联合训练deformation MLP和3D Gaussians,同时进行adaptive density control。

paper观察到30k iteration后,Gaussians的shape和canonical space都stabilize,间接证明decoupling设计的有效性。

### 4.2 Optimization

- Optimizer: 单个Adam,但不同component不同learning rate
  - 3D Gaussians: 和原3D-GS完全相同
  - Deformation network: exponential decay从 $8 \times 10^{-4}$ 到 $1.6 \times 10^{-6}$
- Adam betas: $(0.9, 0.999)$
- 总iterations: 40k
- Hardware: 单张NVIDIA RTX 3090

### 4.3 Loss函数

**公式(7)**:
$$\mathcal{L} = (1-\lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$
- $\lambda = 0.2$
- $\mathcal{L}_1$: pixel-wise L1 loss,主要driver
- $\mathcal{L}_{\text{D-SSIM}}$: differentiable SSIM loss,补充structural信息

这个组合在3D-GS原始paper中已经验证有效,这里直接复用。

---

## 5. 实验结果解析

### 5.1 Synthetic Data (D-NeRF dataset) - Table 1

| Method | Hell Warrior PSNR | Mutant PSNR | Hook PSNR | Bouncing PSNR | Lego PSNR | T-Rex PSNR | Stand Up PSNR | Jumping PSNR |
|---|---|---|---|---|---|---|---|---|
| 3D-GS | 29.89 | 24.53 | 21.71 | 23.20 | 22.10 | 21.93 | 21.91 | 20.64 |
| D-NeRF | 24.06 | 30.31 | 29.02 | 38.17 | 25.56 | 30.61 | 33.13 | 32.70 |
| TiNeuVox | 27.10 | 31.87 | 30.61 | 40.23 | 26.64 | 31.25 | 34.61 | 33.49 |
| Tensor4D | 31.26 | 29.11 | 28.63 | 24.47 | 23.24 | 23.86 | 30.56 | 24.20 |
| K-Planes | 24.58 | 32.50 | 28.12 | 40.05 | 28.91 | 30.43 | 33.10 | 31.11 |
| **Ours** | **41.54** | **42.63** | **37.42** | **41.01** | **33.07** | **38.10** | **44.62** | **37.72** |

Mean PSNR达到 **39.51**,远超baseline。这种巨大提升是3D-GS在static scene就有的优势延伸到dynamic scene的直接结果——explicit representation的capacity远大于implicit。

**Lego scene的特殊说明**: paper指出了一个有趣的细节——D-NeRF的Lego场景训练集和测试集存在flip angle不一致(可以通过观察Lego shovel的角度验证),为了meaningful comparison,作者用Lego的validation set作为test set。

### 5.2 Real-world Data (NeRF-DS) - Table 2 & 3

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| 3D-GS | 20.29 | 0.7816 | 0.2920 |
| TiNeuVox | 21.61 | 0.8234 | 0.2766 |
| HyperNeRF | 23.45 | 0.8488 | 0.1990 |
| NeRF-DS | 23.60 | 0.8494 | 0.1816 |
| Ours (w/o AST) | 23.97 | 0.8346 | 0.2037 |
| **Ours** | **24.11** | **0.8525** | **0.1769** |

在real-world上提升明显小于synthetic,这反映了pose inaccuracy对explicit方法的影响。AST机制贡献了PSNR +0.14、LPIPS -0.027的改进,在real scene上尤其有效。

**HyperNeRF dataset未做quantitative comparison**: 作者解释,HyperNeRF的pose非常不准确,而PSNR这类metric对小offset比对blur更敏感。换言之,blurry output反而PSNR更高,clear output PSNR低但visually更好。这是evaluation metric的缺陷,在Nerfies和HyperNeRF的paper中都讨论过。

### 5.3 Rendering Speed - Table 4

| Scene | Num Gaussians (k) | FPS |
|---|---|---|
| Lego (D-NeRF) | 300 | 24 |
| Jump (D-NeRF) | 90 | 85 |
| Bouncing | 170 | 38 |
| Espresso (HyperNeRF) | 620 | 15 |
| Americano (HyperNeRF) | 1300 | 6 |

经验法则:**Gaussians数量 < 250k时,30+ FPS实时渲染**。

HyperNeRF场景Gaussians数量爆炸(1.3M)反映其pose极不准确,需要更多Gaussians去fit错误的pose。这反过来说明pose准确性对方法efficiency至关重要。

---

## 6. Ablation Studies深入解析

### 6.1 Network architecture ablation - Table 7

paper对比了三种变体:
- **w/o $\delta\mathbf{s}$**: 不输出scaling offset,只输出position和rotation → 平均PSNR 38.97
- **w/o $\delta\mathbf{r}$**: 不输出rotation offset → 平均PSNR 39.27
- **w r&s**: MLP输入除了x和t,还加入Gaussian的r和s → 平均PSNR 38.55
- **Full (ours)**: 平均PSNR 39.51

观察:
1. 去掉 $\delta\mathbf{s}$ 和 $\delta\mathbf{r}$ 影响都不大,但是同时保留是最佳。说明deformation确实需要capture shape变化
2. **"w r&s"反而更差**这非常counter-intuitive。直觉上,让deformation MLP知道当前Gaussian的rotation和scaling应该更有信息量。但实际更差。我的interpretation:canonical space本身已经通过 $\mathbf{x}$ 间接编码了Gaussian的identity,加入r和s会让MLP更容易overfit到specific Gaussians,失去generalization能力,且破坏了stop-gradient隔离design的初衷。

### 6.2 AST ablation - Figure 4

对比了三种configuration在real dataset上的效果:
- (a) 6pe w/o AST: 时间PE的 $L=6$,无AST
- (b) 10pe w/o AST: 时间PE的 $L=10$,无AST  
- (c) ours: 10pe + AST

观察到:
- (b) 比 (a) 高频细节更多(因为更高频率PE),但也暴露了更多pose inaccuracy带来的jitter
- (c) 在保持(b)高频细节的同时,通过AST消除了jitter

这个ablation非常重要,因为它说明:单纯提高PE的 $L$ 不能解决问题,需要AST的annealing机制。

### 6.3 SE(3) Field ablation - Table 5, 6

参考Nerfies [https://arxiv.org/abs/2108.07381],作者尝试用6-DOF SE(3) field约束position transformation:
- D-NeRF dataset: ours-SE(3) PSNR 39.58 vs ours 39.51 → 微弱提升
- NeRF-DS dataset: ours-SE(3) PSNR 23.95 vs ours 24.11 → 下降

trade-off: SE(3)给刚性场景(如D-NeRF合成)带来inductive bias,但real-world的拓扑变化和非刚性motion更复杂,SE(3)约束过强。且SE(3)额外开销50%训练时间,20% FPS下降,得不偿失。

### 6.4 Background color - Table 8

在D-NeRF数据集上测试black vs white背景:
- 平均: black 39.51, white 38.10 → black更好
- 例外: bouncing (43.52 white vs 41.01 black), trex (38.57 white vs 38.10 black) → white更好
- warrior场景对white特别敏感(32.75 vs 41.54),可能因为warrior本身有深色区域与white背景形成高对比,加大optimization难度

---

## 7. 失败模式与局限

### 7.1 Inaccurate poses

HyperNeRF dataset pose过差 → Gaussians数量爆炸到1.3M → 无法real-time。这是explicit point-based方法的intrinsic limitation:每个Gaussian对应一个具体位置,pose错一点都需要更多Gaussian去"补偿"。

### 7.2 Few training viewpoints

DeVRF dataset [https://arxiv.org/abs/2211.00517]的某场景:100张训练图但只有4个viewpoint → 严重overfit。Swap train/test后(100张图但viewpoint多样)结果大幅改善。

这说明:**viewpoint diversity比绝对图像数更重要**。这和NeRF的multi-view consistency要求一致——deformation MLP需要看到同一时刻的不同视角才能学习3D geometry。

### 7.3 复杂人体motion

paper承认对subtle facial expression的handling能力尚未验证。这暗示deformation MLP的capacity可能不足以capture extreme high-frequency motion(如微表情),后续工作如GaussianAvatars [https://arxiv.org/abs/2309.11129]、SplattingAvatar [https://arxiv.org/abs/2309.13295]等专门针对human的paper做了改进。

---

## 8. Intuition总结

让我提炼几个关键insight:

### 8.1 Canonical Space + Deformation的inductive bias

把dynamic scene的learning分解为两部分:
- **Canonical space**: 学习"参考态"的3D structure,无时间依赖
- **Deformation field**: 学习时间如何warp canonical space到具体时刻

这种decoupling引入了**geometric prior**:物体在不同时刻的motion是structured的,而非random。一个Gaussian在t1的position和t2的position是相关的,这种相关性通过MLP的smoothness和PE的结构被显式建模。

### 8.2 Stop-Gradient的角色分工

$\text{sg}(\mathbf{x})$ 强制deformation MLP只能"读"Gaussian position而不能"写"。这创造了一个non-symmetric信息流:
- Gaussians的canonical position是"事实",由rasterization gradient直接优化
- Deformation MLP是"解释器",把canonical facts映射到具体时刻

这种asymmetry避免了两个组件互相适应导致degenerate solution(例如deformation MLP把canonical点推到错误位置)。

### 8.3 Pure MLP vs Grid的rank argument

paper反复强调dynamic scenes的rank问题。让我用intuition解释:

- Static scene:每个3D点有一个固定color和density → low rank
- Dynamic scene:每个3D点在不同时间有不同color/geometry → higher rank
- Explicit point-based:每个Gaussian是独立的,rank最高,但capacity也最大
- Grid/Plane:共享grid features,强制low-rank decomposition → 在dynamic场景quality上限被rank假设限制

这就是为什么这篇paper坚持用pure MLP——dynamic场景需要高rank representation,MLP虽然慢但能express任意high-rank function。

### 8.4 AST的时间维度正则

AST的intuition可以类比成training data augmentation。给时间编码加噪声,等价于让模型见到"slightly off"的t。如果deformation field对pose error敏感(因为overfit),稍微扰动t就会导致不同canonical mapping,模型被强制学习robust mapping。annealing确保最后阶段fine-tune到真实t分布。

这和noise injection在GAN training [https://arxiv.org/abs/1701.00169]、DAgger [https://arxiv.org/abs/1011.0686]中的思路一致——给input加扰动是一种cheap regularizer。

### 8.5 Warm-up的必要性

前3k iterations不训练deformation field。如果一开始就joint optimize,Gaussians还在剧烈调整position和shape,deformation MLP试图学一个"moving target",容易卡在local minimum。Warm-up让canonical space先建立起来,deformation MLP再去学mapping,效率更高。

---

## 9. 与同期及后续工作的关联

- **同期 Dynamic 3D Gaussians [Luiten et al., 2023, https://arxiv.org/abs/2306.13837]**: tracking-by-synthesis思路,per-frame tracking Gaussians,不做canonical space,适合object tracking但不适合time interpolation。
- **4D Gaussians [Wu et al., 2024, https://arxiv.org/abs/2402.07385]**: 用4D splatting (3D + 1D time)直接建模,无需deformation field,但需要更复杂rasterizer。
- **SC-GS [Huang et al., 2024, https://arxiv.org/abs/2312.14937]**: sparse control points驱动Gaussian deformation,减少MLP开销。
- **Deformable-GS variants**: 后续大量工作改进deformation field,如用grid代替MLP [https://arxiv.org/abs/2311.16776],或加入物理约束 [https://arxiv.org/abs/2311.16006]。

---

## 10. 我的评价

这篇paper的工程价值很高:第一个把3D-GS扩展到dynamic scene,在D-NeRF合成数据上PSNR提升幅度巨大(从~32到~40)。AST机制是clever且elegant的设计,解决了explicit representation在real-world pose noise下的robustness问题。

不过也存在几个问题:
1. **MLP开销**: deformation MLP对每帧每个Gaussian都要forward pass,在大场景中costly。后续工作用grid-based deformation field缓解。
2. **Per-Gaussian deformation**: 没有利用Gaussian之间的structure(如rigid body assumption),无法约束physically plausible motion。
3. **Long sequence**: 对于长时间序列,MLP的global representation可能出现forgetting或drift,grid-based methods如K-Planes在长序列上可能更优。

但是作为开山之作,这套canonical space + deformation MLP + AST的设计成为后续大量4D Gaussian splatting工作的baseline范式。

---

## 参考链接

- Project & code: https://github.com/ingra14m/Deformable-3D-Gaussians
- 3D-GS original: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- D-NeRF dataset: https://github.com/albertpumarola/D-NeRF
- HyperNeRF: https://hypernerf.github.io/
- NeRF-DS: https://github.com/ JeffreyWaight/NeRF-DS
- Nerfies (SE(3) field): https://nerfies.github.io/
- K-Planes: https://arxiv.org/abs/2301.10241
- Tensor4D: https://arxiv.org/abs/2306.05392
- TiNeuVox: https://arxiv.org/abs/2210.10590
- EWA Splatting: https://dl.acm.org/doi/10.1109/VISUAL.2001.964382
- COLMAP SfM: https://colmap.github.io/
- NeRF original: https://arxiv.org/abs/2003.08934
