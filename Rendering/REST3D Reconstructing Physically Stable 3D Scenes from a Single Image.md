---
source_pdf: REST3D Reconstructing Physically Stable 3D Scenes from a Single Image.pdf
paper_sha256: 886d4ff68fc2a413bc6f0cdade498cfa53fc1ce31db46748bec58b98f8ee61dd
processed_at: '2026-08-11T23:02:18-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# REST3D 用人话讲讲

## 核心问题，一句话版本

你拍一张照片，想让AI还原出3D场景，然后扔进physics simulator里不会塌——就这个。

听起来简单，实际上巨难。因为现有的image-to-3D方法（比如SAM3D [6]）重建出来的scene，视觉上看着挺好，但一放进simulator就垮了：table悬空、plant插进table里、chair翻倒。像Figure 3那样，跑60步simulation之后objects散了一地。

**why会这样？** 因为这些方法只optimize visual loss，根本不知道"桌子应该站在地上""花盆应该放在桌上"这种physical constraint。它每个object单独reconstruct得很好，但拼到一起就乱了。

那scene generation方法（SAGE [40]、Holodeck [42]）呢？它们用LLM agent生成plausible layout，物理上stable，但跟输入image对不上——你给它一张卧室照片，它给你生成一个"合理但不一致的"卧室。

**REST3D要解决的就是这个gap**：既要faithful to input image，又要physically stable ready for simulation。

---

## 三阶段Pipeline，用人话走一遍

### Stage 1: 让VLM看图说话，建一棵support tree

你给VLM（Gemini 3 Flash）一张图，让它做三件事：

**第一件：列出所有objects**

不是简单说"有个plant"，要说"plant on dining table"——用position attribute区分同类的不同instances。还手动加一个floor作为ground reference。

**第二件：给每个object切mask**

这里有个clever的agent loop。直接把object description喂给SAM3 [4]效果不好（occlusion、复杂场景下mask不准），所以设计了两个agent打配合：
- Segmentation agent $A^{\text{seg}}$：写prompt给SAM3
- Verifier agent $A^{\text{ver}}$：看mask对不对

Loop最多10轮：$A^{\text{seg}}$ 写prompt → SAM3出mask → overlay到image → $A^{\text{ver}}$ 判断 → 不对就反馈，$A^{\text{seg}}$ 改prompt再来。这个iterative refinement比one-shot靠谱多了。

**第三件：建scene tree $T$**

这是关键。Tree有四个root：`ground`、`wall`、`ceiling`、`ground-wall`（复合支撑，比如radiator既靠地又靠墙）。

每个object挂到它的support parent下面，relation标注为`on`/`hanging`/`attached to`/`inside`。

举例：table的parent是ground，relation是on；plant的parent是table，relation也是on；curtain的parent是wall，relation是hanging。

**这棵tree干嘛用？** 它是后续optimization的structural prior。告诉optimizer：plant的垂直位置要settle在table表面，不是随便飘在空中。而且它定义了divide-and-conquer的分组方式——table和它上面的plant组成一个local group，一起optimize。

### Stage 2: 重建+对齐gravity

对每个object用SAM3D [6] reconstruct mesh，用SAM3D估计的pose拼成scene $S^{\text{raw}}$。

$S^{\text{raw}}$ 的问题：
- 全局朝向歪了——up direction跟gravity不对齐
- Objects互相penetrate
- Table飘在空中，plant插进table里

**Canonicalization两步修：**

第一步，估计dominant vertical direction $Y'$（从ground-supported objects和large furniture综合估计），把整个scene旋转让 $Y'$ 对齐世界 $Y$-axis。

第二步，traverse scene tree，按parent-child关系把每个child沿垂直方向平移到parent的support surface上，解决垂直penetration。

得到 $S^{\text{cano}}$。但horizontal penetration、复杂intersection还在，simulator里还是会崩（Figure 4证明 $S^{\text{cano}}$ 跑simulation最终collapse）。所以需要Stage 3。

### Stage 3: Physics-constrained Optimization——这篇paper的核心

**High-level idea**：用physics simulator当oracle，通过stochastic optimization找一组object pose adjustments，使scene在simulator里settle到一个stable state，同时不偏离canonicalized layout太远。

**为什么不直接用gradient descent？** 因为physics simulator不可微（Isaac Gym [20]是black-box）。CEM (Cross-Entropy Method) [11] 是model-free的population-based方法，只需要forward evaluate，不需要gradient。

**Divide-and-conquer为什么必要？**

Global optimization（所有objects一起优化）在Table 2(c)里Fail Rate 10%。原因：一个scene可能有20-30个objects，每个6-DoF，search space是120-180维。CEM在这个维度上maintain Gaussian distribution很难converge。

所以用scene tree做hierarchical decomposition：
- **Local group optimization**：post-order traverse tree，对每个有children的node $g$，把 $g$ 和它的direct children组成一个group，单独跑CEM。Post-order保证叶子先优化，parent group优化时children已经settle了。
- **Global layout optimization**：把每个optimized group当rigid unit，跟ground-supported objects一起再做一轮CEM。
- **Wall/ceiling objects**：不参与CEM，用heuristic贴到拟合的wall surface上。

**CEM细节，讲讲公式：**

第 $t$ 轮iteration，maintain一个Gaussian分布 $\mathcal{N}(\mu_t, \Sigma_t)$ over pose adjustments $\Delta\mathbf{P}$。

$\mu_t \in \mathbb{R}^{N_g \times 6}$：$N_g$ 个objects，每个object 6维（3 translation + 3 rotation，用axis-angle或euler表示）

$\Sigma_t$：diagonal covariance，每个维度独立

**Sample**（公式1）：采 $K = 2048$ 个candidates
$$\Delta\mathbf{P}_t^{(k)} \sim \mathcal{N}(\mu_t, \Sigma_t)$$

**Apply**：$\mathbf{P}_t^{(k)} = \mathbf{P}^{\text{cano}} \oplus \Delta\mathbf{P}_t^{(k)}$，就是把adjustment加到canonicalized pose上

**Simulate**：每个candidate扔进Isaac Gym跑 $L = 60$ 步forward simulation，记录每个object的final pose和intermediate velocity

**Score**：算energy $E(\mathbf{P}_t^{(k)})$

**Select elite**：取energy最低的top $\rho = 0.025$（约51个）

**Update**（moment matching）：
$$\mu_{t+1} = \frac{1}{|\mathcal{E}_t|} \sum_{k \in \mathcal{E}_t} \Delta\mathbf{P}_t^{(k)}$$
$$\Sigma_{t+1} = \frac{1}{|\mathcal{E}_t|} \sum_{k \in \mathcal{E}_t} (\Delta\mathbf{P}_t^{(k)} - \mu_{t+1})^2$$

跑 $T = 15$ 轮，最后取所有iteration里energy最低的那个candidate。

**Energy function是灵魂，四个term各有分工：**

$$E = \lambda_{\text{stab}} E_{\text{stab}} + \lambda_{\text{vel}} E_{\text{vel}} + \lambda_{\text{pen}} E_{\text{pen}} + \lambda_{\text{layout}} E_{\text{layout}}$$

**$E_{\text{stab}}$（公式3）——settle后drift多少**

$$E_{\text{stab}} = \sum_i \left( \|t_i^{(k),L} - t_i^{(k)}\| + d(R_i^{(k),L}, R_i^{(k)}) \right)$$

- $t_i^{(k)}$：object $i$ 的initial placement translation
- $t_i^{(k),L}$：跑 $L$ 步simulation后的translation
- $R_i^{(k)}, R_i^{(k),L}$：对应rotation
- $d(\cdot,\cdot)$：rotation的geodesic distance，用quaternion distance算

Intuition：simulation前后object几乎没动 → 处于stable equilibrium → energy低。动了 → 不stable → energy高。

**$E_{\text{vel}}$（公式4）——中间step的速度**

$$E_{\text{vel}} = \sum_i \|v_i\|$$

$v_i$：object $i$ 在intermediate step $\tau = 15$ 的linear velocity。

Why需要这个？因为有的candidate可能"假stable"——当前step静止，但下一秒会collapse。$\tau = 15$ 时如果还有高velocity，说明object还在dynamic interaction中（碰撞、滑动），没真正settle。这是early warning signal。

**$E_{\text{pen}}$（公式5）——penetration**

$$E_{\text{pen}} = E_{\text{pen}}^{(k)} + E_{\text{pen}}^{(k),L}$$

- $E_{\text{pen}}^{(k)}$：placement时刻pairwise convex hull intersection数量，用GJK算法 [12] 计算
- $E_{\text{pen}}^{(k),L}$：settle后的intersection数量

两个时刻都查：placement时penetrate不行，settle后还penetrate也不行。

**$E_{\text{layout}}$（公式6）——别跑偏**

$$E_{\text{layout}} = \sum_i \left( \lambda_{\text{pos}} \|t_i^{(k),L} - t_i^{\text{cano}}\| + d(R_i^{(k),L}, R_i^{\text{cano}}) \right)$$

$\lambda_{\text{pos}} = 6$ 权重很大，强制settle后的pose接近canonicalized pose $S^{\text{cano}}$。

Why重要？没有这个term，optimizer可能找到trivial stable solution——比如所有objects都推到ground上散开，物理stable但跟原图差太远。$E_{\text{layout}}$ 是anchor，保证visual consistency。

**超参数sensitivity**（Table 6）：

Elite fraction $\rho$ 控制exploration-exploitation。$\rho = 0.025$ 最优。太小（0.01）elite太少update不稳定，太大（0.5）selectivity降低变成random search。

---

## 实验结果，讲故事

### 主实验Table 1——Custom dataset最说明问题

Custom是25张Internet casual images，覆盖real/synthetic/cartoon/Gaussian-rendered，最能测real-world generalization。

| Method | Fail Rate | Stable Rate | Pos. Drift | Lin. Vel. |
|--------|-----------|-------------|------------|-----------|
| DigitalCousins [9] | 48.0% | 50.2% | 0.911m | 2.196 m/s |
| Gen3DSR [1] | 40.0% | 7.2% | 2.756m | 6.493 m/s |
| SceneGen [22] | 16.0% | 2.9% | 15.626m | 27.475 m/s |
| SAM3D† [6] | 0.0% | 16.0% | 1.189m | 1.929 m/s |
| **REST3D** | **0.0%** | **95.5%** | **0.017m** | **0.140 m/s** |

几个观察：

**Failure Rate**：DigitalCousins 48%直接挂了，因为retrieval-based，database里没合适asset就fail。REST3D和SAM3D†用generative image-to-3D，0% failure。

**Stable Rate**：REST3D 95.5%，SAM3D†只有16%——差了79.5个百分点（abstract里说83-point improvement是Replica dataset上的数字）。SceneGen更惨，2.9%，Pos. Drift 15.626m——objects在simulation里飞出视野了。

**Lin. Vel.**：SceneGen 27.475 m/s，相当于objects以100km/h速度乱飞。REST3D 0.140 m/s，基本静止。

### 几何指标的"悖论"（Section 4.3, Figure 9）

SAM3D†在CD/F-Score/B-IoU上反而略好于REST3D。Why？

因为CD（Chamfer Distance）只算ICP alignment后的point-to-point distance，**不管objects是否penetrate**。SAM3D†重建的chair虽然插进table里，但每个chair的geometry本身更接近GT，所以CD更低。

REST3D为了physical plausibility把penetrating objects分开，引入了small geometric deviation，CD反而高了。

**这是paper的一个insight**：现有geometric metrics有盲区，不反映physical validity。社区需要physical-aware metrics。这个观点跟最近simulation-based evaluation的工作 [32] 方向一致。

### Ablation讲清楚每个component的价值

**Table 2——canonicalization和divide-and-conquer**

| 配置 | Fail Rate | Stable Rate |
|------|-----------|-------------|
| SAM3D† raw | 0.0% | 12.0% |
| + Canonicalization | 0.0% | 45.8% |
| + Global-only optimization | 10.0% | 89.4% |
| + Full (local + global) | **0.0%** | **94.9%** |

Canonicalization alone把12%提到45.8%——coarse alignment有用但不够。Global-only optimization把stable rate提到89.4%但10% fail——大scene下CEM不converge。Full pipeline 94.9% + 0% fail，divide-and-conquer是必须的。

**Table 3——四个energy term**

去掉任何一个都变差，但有意思的pattern：

去掉 $E_{\text{pen}}$ → geometric metrics反而最好（CD=0.016），因为不penalty penetration，objects保持原位geometry接近GT，但physical stability降。再次证明geometric metrics盲区。

去掉 $E_{\text{layout}}$ → CD变差到0.024，因为optimizer把objects散开求stable。这个term是visual consistency的anchor。

去掉 $E_{\text{stab}}$ → 5% fail，找不到stable solution。

去掉 $E_{\text{vel}}$ → stable rate 92.8%，略降，少了early warning。

**Table 5——scene tree本身的价值**

光给SAM3D†加scene tree $T$（仅在simulation时固定wall/ceiling objects），stable rate从12%提到36.6%。加到canonicalized scene上，提到70.6%。但都远不如full pipeline的94.9%——scene tree是structural prior，但光靠它不够，必须配physics optimization。

---

## VR Interaction——证明scene真的simulation-ready

Paper不只停在metrics，还搭了个VR系统让你用Meta Quest Pro抓virtual objects。

技术细节（Supp. Sec. C.3）：
- Wrist控制：PD force-torque，position gain 200 N/m，force clamp 35 N
- Finger控制：IK retargeting [28]映射fingertip到joint
- Soft-grip spring：两个finger links在0.12m内激活，stiffness 60 N/m
- Timestep 1/30s，4 substeps
- Mass按semantic category分：heavy 60-200kg, medium 3-7kg, light 2.5-4kg

30 FPS real-time interaction，说明reconstructed scene不只是"看起来stable"，而是真的能在simulator里被抓取、移动、碰撞。

---

## Limitations，honest地讲

1. **VLM会miss objects**：Figure 13里漏检了wall-mounted shelf，导致layout偏差。这是open-vocabulary detection的固有困难。

2. **Rigid objects only**：不处理deformable（curtain、cushion）和articulated objects（cabinet door、drawer）。Paper里明确说future work。

3. **Wall modeling缺失**：optimization里不explicitly model wall geometry，导致wall-supported objects（如bench上的pillows）被relocate到ground求lower energy。Figure 13红圈展示了这个问题。

4. **CEM sample efficiency**：K=2048 per iteration × 15 iterations × 2 episodes × 2 stages = 61440次simulation。虽然GPU并行，但10.4分钟还是慢。高维scene（30 objects）可能不够。

---

## 联想与broader context

### 跟Karpathy关心的方向的连接

**1. VLM as world parser**

REST3D的Stage 1本质是用VLM做"visual world parsing"——从image提取structured scene representation。这跟VIGA [44]（Vision-as-Inverse-Graphics Agent）思路类似，都是用VLM做inverse graphics。区别是REST3D加了agent loop（seg + verify）提升robustness。

这个方向会越来越重要：VLM不只是生成text，而是作为structured scene understanding的backbone，输出graph、tree、program等structured representation供downstream reasoning用。

**2. Simulation as differentiable-ish oracle**

REST3D用Isaac Gym当black-box oracle，CEM当gradient-free optimizer。这跟differentiable physics [23]路线不同。Differentiable physics需要可微simulator，建模限制多，但gradient效率高。CEM + black-box simulator灵活但sample效率低。

未来方向：hybrid——用differentiable physics做local refinement，用CEM做global search。或者用learned dynamics model（world model）替代physics simulator，加速evaluation。

**3. Simulation-ready assets是embodied AI的瓶颈**

Robotics和embodied AI需要大量simulation environments训练policy。手动建scene成本高，从image自动生成simulation-ready scene能大幅降低成本。REST3D是这个pipeline的关键一环。

相关工作：URDFormer [7]从image构建articulated simulation environments，SceneFoundry [5]生成infinite 3D worlds，PhyReCon [23]做physically plausible reconstruction但需multi-view input。

**4. Physical-aware evaluation metrics**

Paper指出CD/F-Score的盲区是个重要insight。社区需要physical-aware metrics：
- Simulation stability score（rest后drift）
- Support relation accuracy（support parent预测对不对）
- Contact plausibility（object是否contact在合理surface上）

相关工作：Reconciling Reality through Simulation [32]提出simulation-based evaluation for manipulation。这个方向值得follow。

**5. 从single image到video**

Paper只处理single image。Video input（Holoscene [38]）能利用temporal consistency和multi-view信息，physical constraints更容易infer。但casual image的practical value更高——任何一张照片都能转成3D asset。

### 技术细节的intuition深化

**CEM为什么对这个问题work？**

CEM本质是estimation of distribution algorithm。它不直接optimize solution，而是optimize一个分布，让分布逐渐集中在good solution区域。对于physics-in-the-loop这种black-box、non-convex、noisy的objective，CEM比gradient methods鲁棒，比random search高效。

关键超参数：
- $K = 2048$：population size。太小variance大，太大compute贵。2048在GPU上并行Isaac Gym可handle。
- $\rho = 0.025$：elite fraction。太小overfitting局部，太大selectivity弱。
- $T = 15$：iteration数。够distribution converge，再多diminishing return。
- Initial $\Sigma$：per-DoF standard deviation。translation $\sigma_0^{\text{trans}} = (0.05, 0.005, 0.05)$ m，rotation $\sigma_0^{\text{rot}} = (0.005, 0.05, 0.005)$ rad。Horizontal displacement和pitch需要更大search range（因为canonicalization后vertical和roll/yaw已经大致对了，但水平位置和pitch可能还偏）。

**Scene tree的post-order traversal为什么重要？**

Post-order保证leaf-first。Plant（leaf）先optimize，settle在table上。然后table group（table + plant）再optimize，plant已经stable，table找自己的stable position。如果是pre-order，table先优化时plant位置还没定，优化结果可能suboptimal。

这跟compilation里的post-order evaluation、graph neural network里的message passing方向类似——信息从leaf流向root。

**Energy function的设计哲学**

四个term代表四种constraint：
- $E_{\text{stab}}$：physics constraint（simulator说stable才算）
- $E_{\text{vel}}$：temporal constraint（不只是final state stable，intermediate也要stable）
- $E_{\text{pen}}$：geometric constraint（不能penetrate）
- $E_{\text{layout}}$：semantic constraint（跟input一致）

这是multi-objective optimization的经典formulation。权重 $\lambda$ 决定trade-off。$\lambda_{\text{pos}} = 6$ 比 $\lambda_{\text{pen}} = 0.5$ 大很多，说明layout consistency比penetration更优先——宁愿有微小penetration也要保持layout。这个choice跟goal有关：REST3D要faithful reconstruction + physical stability，不是纯generation。

### 跟我（Karpathy）过去讲的connection

我在[Building multimodal AI systems]讲过，未来AI system需要grounded understanding——不只是生成pixel，而是理解physical world structure。REST3D是这方面的实践：用VLM提取structured scene understanding，用physics simulator验证consistency。

我在讲nanoGPT时强调过，好的工程是simple and works。REST3D的CEM + simulator组合不novel（CEM 1997年就有 [29]），但把VLM scene understanding + image-to-3D + physics optimization串起来解决实际问题，这个system design是valuable的。

---

## References

- REST3D project: https://shirleymaxx.github.io/REST3D/
- SAM3D [6]: https://arxiv.org/abs/2511.16624
- SAM3 [4]: ICLR 2026
- DigitalCousins [9]: https://arxiv.org/abs/2406.08476 (CoRL 2024)
- Gen3DSR [1]: 3DV 2025
- SceneGen [22]: 3DV 2026
- SAGE [40]: CVPR 2026
- Isaac Gym [20]: https://arxiv.org/abs/2108.10470 (NeurIPS 2021)
- CEM tutorial [11]: https://link.springer.com/article/10.1007/s10479-005-5723-z
- GJK algorithm [12]: https://ieeexplore.ieee.org/document/2086
- Replica [31]: https://arxiv.org/abs/1906.05797
- ScanNet++ [43]: https://arxiv.org/abs/2308.11491 (ICCV 2023)
- Open-TeleVision [8]: CoRL 2024
- VIGA [44]: https://arxiv.org/abs/2601.11109
- PhyReCon [23]: https://arxiv.org/abs/2404.00476
- Holoscene [38]: NeurIPS 2025
- URDFormer [7]: RSS 2024
- SceneFoundry [5]: https://arxiv.org/abs/2601.05810
- Reconciling Reality through Simulation [32]: RSS 2024
- World Labs Marble [36]: https://www.worldlabs.ai/blog/marble-world-model
- V-HACD: https://github.com/kmammou/v-hacd

---

## 最后一句总结

REST3D的价值在于：用VLM scene understanding建structural prior，用physics simulator当oracle做black-box optimization，在single-image reconstruction的fidelity和scene generation的plausibility之间找到了sweet spot。83-point stability improvement说明physical constraints被现有方法严重忽视了，这是一个值得社区follow的方向——不只是重建看起来对，而是重建出来的东西能在world里run。

---

# REST3D: 从单张图像重建物理稳定的3D场景

这篇paper来自CMU的团队，解决的核心问题是：从单张casual RGB image重建出**physics-simulation-ready**的3D场景。核心intuition在于：现有的single-image 3D reconstruction方法（如SAM3D、Gen3DSR）虽然在视觉上看起来合理，但重建出的场景在physics simulator中会"塌掉"——objects float、penetrate、topple over。而scene generation方法虽然物理合理，却不能faithfully reproduce输入图像。REST3D的目标是同时满足visual consistency和physical plausibility。

---

## 1. 核心问题与动机

### 1.1 问题定义

给定单张RGB image $I$，重建一个3D scene $S$，包含：
- 每个object的mesh $\{M_i\}$
- 每个object在世界坐标系下的6-DoF pose $(R_i, t_i)$ 和scale
- 整个scene需要ready for physics simulation，即导入simulator后不会因为floating/penetration导致unstable dynamics

### 1.2 现有方法的痛点

| Method类型 | 代表方法 | 问题 |
|-----------|---------|------|
| Single-image reconstruction | SAM3D [6], Gen3DSR [1], SceneGen [22] | 视觉合理但物理invalid，objects float/penetrate |
| Retrieval-based | DigitalCousins [9] | 受限于asset database coverage，retrieve的objects与input mismatch |
| Scene generation | SAGE [40], SceneFoundry [5] | 依赖strong scene priors，plausible但不accurate，与input image不match |

Table 1中的数据很说明问题：在Custom dataset上，DigitalCousins的Failure Rate高达48.0%，SceneGen的Pos. Drift达到15.626m，Lin. Vel.达到27.475 m/s——这意味着objects在simulation中飞得到处都是。

---

## 2. 方法架构：三阶段Pipeline

整体pipeline分三阶段（Figure 2）：

### Stage 1: Scene-Tree Construction

**目标**：从image中推断一个hierarchical scene tree $T$，编码object的physical states和inter-object relationships（从gravity-support perspective）。

#### 2.1.1 Open-vocabulary Object List Analysis

使用VLM（Gemini 3 Flash）生成structured object list $\mathcal{O} = \{\mathcal{O}_i\}$。关键trick是：用disambiguating attributes区分同类instances，例如"plant on dining table"而不是笼统的"plant"。还手动加入floor作为canonical ground-plane reference。

#### 2.1.2 Agentic Instance Segmentation

两个agent的iterative loop：
- **Segmentation agent** $A^{\text{seg}}$：调用SAM3 [4]生成mask
- **Verifier agent** $A^{\text{ver}}$：评估mask质量，给出feedback

Loop：$A^{\text{seg}}$ refine prompt → SAM3生成candidate mask → overlay到image → $A^{\text{ver}}$判断 → 若不对则$A^{\text{seg}}$再refine。最多10次iterations。最终得到mask集合 $\mathcal{M} = \{\mathcal{M}_i\}$。

#### 2.1.3 Scene Tree via Spatial Reasoning

Scene tree $T$ 有**四个canonical root nodes**：
- `ground`（地面支撑）
- `wall`（墙面附着）
- `ceiling`（天花板悬挂）
- `ground-wall`（复合支撑，如radiator同时接触地面和墙）

每种support relation有四种type：`on`, `hanging`, `attached to`, `inside`。还标注object是否movable。

VLM分析mask-overlaid object pairs $(\mathcal{M}_i, \mathcal{M}_j)$，从gravity-aware perspective判断support parent和relation type。

**Intuition**：这个scene tree是后续physics-constrained optimization的structural prior。它告诉optimizer："这个plant的parent是table，relation是on"，这样optimizer就知道调整plant的vertical position时要让它settle在tabletop上。

### Stage 2: Scene Initialization and Canonicalization

#### 2.2.1 Object Reconstruction

对每个object用image-to-3D model（SAM3D [6]）从其mask重建mesh $\{M_i\}$，并用SAM3D估计的rotation/translation/scale初始化scene $S^{\text{raw}}$。

但 $S^{\text{raw}}$ 的问题：
- Global orientation可能misaligned（up direction与gravity不aligned）
- Inter-object collisions频繁
- Table可能floating，plant可能penetrate tabletop

#### 2.2.2 Scene Canonicalization

两步coarse correction：

**Global orientation correction**：从ground-supported objects和large furniture anchors估计dominant vertical direction $Y'$，将整个scene旋转使 $Y'$ 对齐世界坐标系 $Y$-axis（opposite to gravity）。

**Hierarchical support enforcement**：traverse scene tree $T$，根据parent-child relations调整object positions——将每个child沿vertical direction平移到其supporting surface上，解决vertical penetrations。

得到 $S^{\text{cano}}$，但仍有residual inconsistencies（如inter-object intersections），所以需要Stage 3。

### Stage 3: Physics-Constrained Optimization

这是paper的核心技术贡献。采用**divide-and-conquer**策略，由scene tree $T$引导。

#### 2.3.1 Local Group Optimization

**Group构造**：post-order traversal $T$，对每个有至少一个child的non-root node $g$，定义local group = $\{g\} \cup \text{children of } g$。

Post-order保证：当node $g$ 出现在其parent的group时，$g$ 自己的group已经optimized过了。

**Formulation**：

每个object的pose用6-DoF表示。Group $g$ 有 $N_g$ 个children，初始layout：
$$\mathbf{P}^{\text{cano}} = \{(R_i^{\text{cano}}, t_i^{\text{cano}})\}_{i=1}^{N_g}$$

其中 $R_i^{\text{cano}} \in SO(3)$ 是rotation matrix，$t_i^{\text{cano}} \in \mathbb{R}^3$ 是translation。

目标：找到small pose adjustments使group达到physically stable state且preserve original layout。

**Cross-Entropy Method (CEM)**：

CEM是population-based stochastic optimization。维护parametric Gaussian distribution over pose adjustments，iteratively refine toward high-quality solutions。

第 $t$ 次iteration：
- **Distribution**：$\mathcal{N}(\mu_t, \Sigma_t)$，其中 $\mu_t \in \mathbb{R}^{N_g \times 6}$，$\Sigma_t$ 是diagonal covariance
- **Sampling**（公式1）：
$$\Delta \mathbf{P}_t^{(k)} \sim \mathcal{N}(\mu_t, \Sigma_t), \quad k = 1, \ldots, K$$
其中 $K = 2048$ 是population size，$\Delta \mathbf{P}_t^{(k)}$ 是第 $k$ 个sample的pose adjustment（6维向量 per object）

- **Candidate layouts**：$\mathbf{P}_t^{(k)} = \mathbf{P}^{\text{cano}} \oplus \Delta \mathbf{P}_t^{(k)}$

- **Simulation evaluation**：把每个candidate导入Isaac Gym [20]，forward simulate $L = 60$ steps，得到post-simulation states

- **Energy computation**：$E(\mathbf{P}_t^{(k)})$ for each candidate

- **Elite selection**：选top $\lceil \rho K \rceil$ lowest-energy samples作为elite set $\mathcal{E}_t$，其中 $\rho = 0.025$（即top 2.5%，约51个samples）

- **Moment matching**（更新distribution）：
$$\mu_{t+1} = \frac{1}{|\mathcal{E}_t|} \sum_{k \in \mathcal{E}_t} \Delta \mathbf{P}_t^{(k)}$$
$$\Sigma_{t+1} = \frac{1}{|\mathcal{E}_t|} \sum_{k \in \mathcal{E}_t} (\Delta \mathbf{P}_t^{(k)} - \mu_{t+1})^2$$

- **Final selection**：
$$\mathbf{P}^\star = \arg\min_{t,k} E(\mathbf{P}_t^{(k)})$$

总共 $T = 15$ iterations，2个episodes。

#### Energy Function（公式2）

$$E = \lambda_{\text{stab}} E_{\text{stab}} + \lambda_{\text{vel}} E_{\text{vel}} + \lambda_{\text{pen}} E_{\text{pen}} + \lambda_{\text{layout}} E_{\text{layout}}$$

超参数：$\lambda_{\text{stab}} = \lambda_{\text{vel}} = \lambda_{\text{layout}} = 1$，$\lambda_{\text{pen}} = 0.5$，$\lambda_{\text{pos}} = 6$（用于layout energy内部）。

**1. Stability energy $E_{\text{stab}}$（公式3）**：

衡量objects在forward simulation后drift了多少。
$$E_{\text{stab}} = \sum_i \left( \|t_i^{(k),L} - t_i^{(k)}\| + d(R_i^{(k),L}, R_i^{(k)})\right)$$

变量含义：
- $t_i^{(k)}$：object $i$ 的initial placement translation
- $t_i^{(k),L}$：object $i$ 在 $L = 60$ steps simulation后的translation
- $R_i^{(k)}, R_i^{(k),L}$：对应的rotation matrices
- $d(\cdot, \cdot)$：rotation之间的geodesic distance，用quaternion distance实现

Intuition：如果object在simulation后基本没动，说明它处于stable equilibrium。

**2. Velocity energy $E_{\text{vel}}$（公式4）**：

在intermediate step $\tau = 15$ 测量，作为instability的early indicator。
$$E_{\text{vel}} = \sum_i \|v_i\|$$

$v_i$：object $i$ 在step $\tau$ 的linear velocity。High velocity说明object还在undergoing dynamic interactions（如collisions），未settle。

**3. Penetration energy $E_{\text{pen}}$（公式5）**：

在placement和settlement两个时刻都penalize geometric overlap。
$$E_{\text{pen}} = E_{\text{pen}}^{(k)} + E_{\text{pen}}^{(k),L}$$

- $E_{\text{pen}}^{(k)}$：placement时刻的pairwise convex-hull intersections，用GJK算法 [12] 计算
- $E_{\text{pen}}^{(k),L}$：$L$-step settlement后的intersections

**4. Layout energy $E_{\text{layout}}$（公式6）**：

防止optimizer为了physical stability而把objects分散到偏离原layout的位置。
$$E_{\text{layout}} = \sum_i \left( \lambda_{\text{pos}} \|t_i^{(k),L} - t_i^{\text{cano}}\| + d(R_i^{(k),L}, R_i^{\text{cano}}) \right)$$

$\lambda_{\text{pos}} = 6$ 这个较大的权重确保settlement后的scene与canonicalized scene $S^{\text{cano}}$ 在spatial上一致。

#### 2.3.2 Global Layout Optimization

Local groups优化完后，每个group作为single rigid unit。Global stage优化：
- 所有parent为`ground`或`ground-wall`的objects
- 所有optimized groups的roots

用相同的CEM formulation + 同样的energy function $E$。Group的children通过compose updated root pose与local group pose $\mathbf{P}^\star$ 重建。

#### 2.3.3 Wall & Ceiling Object Placement

Parent为`wall`或`ceiling`的objects不参与CEM optimization。Heuristic post-processing：
1. 从已settle objects的axis-aligned bounding box拟合三面墙
2. 每个wall/ceiling object assign到最近的fitted surface
3. 沿surface normal direction平移直到所有intersections resolved

---

## 3. 实验数据深度解析

### 3.1 主实验（Table 1）

在三个dataset上评估：
- **Replica** [31]：synthetic，有GT mesh
- **ScanNet++** [43]：real-world，有GT mesh
- **Custom**：25张Internet images + prior work images，涵盖real/synthetic/cartoon/Gaussian-rendered，无GT

**关键数字**（Custom dataset，最能反映真实场景）：

| Method | Fail. Rate | Coll. Rate | Stable Rate | Pos. Drift | Lin. Vel. | Ang. Vel. |
|--------|-----------|------------|-------------|------------|-----------|-----------|
| DigitalCousins [9] | 48.0 | 25.0 | 50.2 | 0.911 | 2.196 | 2.158 |
| Gen3DSR [1] | 40.0 | 68.2 | 7.2 | 2.756 | 6.493 | 9.009 |
| SceneGen [22] | 16.0 | 78.4 | 2.9 | 15.626 | 27.475 | 29.499 |
| SAM3D† [6] | 0.0 | 45.4 | 16.0 | 1.189 | 1.929 | 11.801 |
| **Ours** | **0.0** | **1.2** | **95.5** | **0.017** | **0.140** | **0.468** |

REST3D的Stable Rate达到95.5%，相比SAM3D†的16.0%提升了约**83 percentage points**——这就是abstract里说的"83%-point improvement"。Pos. Drift从1.189m降到0.017m（降低70倍），Lin. Vel.从1.929降到0.140 m/s。

**几何指标的悖论**：SAM3D在CD/F-Score/B-IoU上略优于REST3D。原因在Section 4.3和Figure 9分析：CD等metric只计算ICP alignment后的bidirectional distance，**不考虑inter-object collisions或physical constraints**。SAM3D虽然objects严重penetrate，但每个object的geometry本身更接近GT，所以CD更低。REST3D为了physical plausibility把penetrating objects分开，反而引入了small geometric deviation。

### 3.2 Ablation Study

#### 3.2.1 Scene Canonicalization（Table 2）

| Ablation | Fail. Rate | Coll. Rate | Stable Rate | Pos. Drift |
|----------|-----------|------------|-------------|------------|
| (a) SAM3D† (S_raw) | 0.0 | 45.1 | 12.0 | 1.111 |
| (b) Canon. (S_cano) | 0.0 | 26.3 | 45.8 | 0.767 |
| (c) Global-only | 10.0 | 4.3 | 89.4 | 0.116 |
| **Ours (S)** | **0.0** | **2.7** | **94.9** | **0.072** |

Observations：
- Canonicalization alone把Stable Rate从12.0%提到45.8%，但还不够（Figure 4显示S_cano仍会collapse）
- Global-only optimization（无local group stage）Fail Rate 10%——大scene下joint optimization不converge
- 完整的local + global divide-and-conquer策略最优

#### 3.2.2 Energy Terms（Table 3）

| Ablation | Stable Rate | Pos. Drift | CD | F-Score |
|----------|-------------|------------|-----|---------|
| w/o $E_{\text{stab}}$ | 90.8 (5% fail) | 0.136 | 0.018 | 0.820 |
| w/o $E_{\text{vel}}$ | 92.8 | 0.109 | 0.017 | 0.823 |
| w/o $E_{\text{pen}}$ | 92.2 | 0.103 | **0.016** | **0.828** |
| w/o $E_{\text{layout}}$ | 91.8 | 0.075 | 0.024 | 0.726 |
| **Ours** | **94.9** | **0.072** | 0.017 | 0.824 |

有意思的发现：
- 去掉 $E_{\text{pen}}$ 反而geometric metrics最好（CD=0.016, F-Score=0.828）——因为不penalize penetration，objects保持原位，geometric similarity高，但physical stability下降。这再次印证了geometric metrics的局限性。
- 去掉 $E_{\text{layout}}$，CD变差到0.024，F-Score降到0.726——optimizer为了stability把objects分散得太远。
- 去掉 $E_{\text{stab}}$ 导致5% failure——找不到stable solution。

#### 3.2.3 CEM Elite Fraction（Table 6）

| Elite Frac $\rho$ | Stable Rate | Pos. Drift | Lin. Vel. |
|-------------------|-------------|------------|-----------|
| 0.01 | 92.0 | 0.081 | 0.204 |
| **0.025 (Ours)** | **94.9** | **0.072** | **0.138** |
| 0.05 | 92.9 | 0.107 | 0.199 |
| 0.1 | 91.2 | 0.060 | 0.192 |
| 0.5 | 83.0 | 0.124 | 0.276 |

$\rho = 0.025$ 是exploration-exploitation的最佳平衡。太小（0.01）elite diversity不足，太大（0.5）selectivity降低。

### 3.3 Scene Tree的作用（Table 5）

把scene tree $T$ 加到SAM3D†的 $S^{\text{raw}}$ 上（仅在simulation时用 $T$ 固定wall/ceiling objects）：

| Method | Coll. Rate | Stable Rate | Pos. Drift |
|--------|------------|-------------|------------|
| SAM3D† (S_raw) | 45.1 | 12.0 | 1.111 |
| SAM3D† (S_raw) + $\mathcal{T}$ | 45.1 | 36.6 | 0.714 |
| Canon. (S_cano) | 26.3 | 45.8 | 0.767 |
| Canon. (S_cano) + $\mathcal{T}$ | 26.3 | 70.6 | 0.334 |
| **Ours (S)** | **2.7** | **94.9** | **0.072** |

仅加 $T$ 能提升stability（12.0→36.6），但Coll. Rate不变（45.1），因为geometric penetration仍在。加上canonicalization后Stable Rate到70.6，但仍不如完整pipeline的94.9%。

### 3.4 效率与成本（Table 4）

| Method | min/Scene | min/Obj | Cost/Scene |
|--------|-----------|---------|------------|
| DigitalCousins [9] | 32.5 | 2.5 | $1.25 |
| Gen3DSR [1] | 39.3 | 2.1 | — |
| SceneGen [22] | 27.1 | 2.1 | — |
| SAM3D† [6] | 10.6 | 0.8 | $0.22 |
| **Ours (Total)** | **25.8** | **2.0** | **$0.47** |
| Ours (Stage 1) | 10.6 | 0.8 | $0.22 |
| Ours (Stage 2) | 4.8 | 0.3 | $0.25 |
| Ours (Stage 3) | 10.4 | 0.9 | — |

REST3D总时间25.8 min/scene，比Gen3DSR和DigitalCousins快。Stage 3（CEM optimization）占10.4 min，但无API cost。用Gemini 3 Flash比GPT-4o便宜（$0.47 vs $1.25）。

---

## 4. VR Interaction System（Section 4.5, C.3）

paper还实现了一个real-time VR interaction system（30 FPS）：
- **Hardware**：Meta Quest Pro headset
- **Hand tracking**：参考Open-TeleVision [8]，two-level control
  - Wrist：PD force-torque actuation（position gain 200 N/m，force clamp 35 N；orientation gain 10 Nm/rad，torque clamp 5 Nm）
  - Fingers：IK-based retargeting [28]映射fingertip positions到joint configurations
- **Soft-grip spring**：至少两个finger links在0.12m内有object时激活，stiffness 60 N/m，damping 8 N·s/m
- **Simulator**：Isaac Gym GPU PhysX，timestep 1/30s，4 substeps
- **Mass assignment**：按semantic category分三档
  - Heavy（sofa/desk/bed/cabinet）：[60, 200] kg
  - Medium（chair/lamp/TV）：[3, 7] kg
  - Light（cup/book/vase）：[2.5, 4] kg

---

## 5. Limitations（Section 5, C.4）

1. **VLM robustness**：open-vocabulary detection可能miss objects（Figure 13中漏检wall-mounted shelf），导致layout与input image不一致
2. **Rigid objects only**：不处理deformable/articulated objects
3. **Wall modeling缺失**：optimization中不explicitly model walls as support structures，导致wall-supported objects（如bench上的pillows）被relocate到ground以achieve lower energy state

---

## 6. 关键Intuition总结

### 6.1 为什么divide-and-conquer优于global-only？

Global-only optimization在objects多的大scene中Fail Rate 10%（Table 2c）。原因是CEM的search space随object数量指数增长，$N_g \times 6$ 维Gaussian的moment matching变得unstable。Divide-and-conquer通过scene tree的hierarchical structure把大问题分解成小subproblems，每个local group的search space小，CEM更容易converge。

### 6.2 为什么需要四个energy terms协同？

- $E_{\text{stab}}$：确保settlement后stable（必要条件）
- $E_{\text{vel}}$：early indicator，避免"假stable"（object暂时静止但稍后会collapse）
- $E_{\text{pen}}$：解决geometric overlap（避免objects卡在彼此内部）
- $E_{\text{layout}}$：anchor到canonicalized scene，防止optimizer找到trivial stable solution（如所有objects散落到ground上）

去掉任何一个都会degrade，Table 3充分验证。

### 6.3 Scene Tree的structural prior价值

Scene tree不只是spatial layout描述，更是physics optimization的**约束骨架**。它告诉CEM：
- 哪些objects是support关系（local group内需要jointly optimize）
- 哪些objects是wall/ceiling attached（不参与dynamics，用heuristic placement）
- Post-order traversal保证leaf-to-root的优化顺序，parent group优化时children已settle

### 6.4 Geometric Metrics的盲区

这是paper一个重要insight（Section 4.3, Figure 9）：传统的CD/F-Score/B-IoU在ICP alignment后计算bidirectional distance，**完全ignore inter-object physical relationships**。一个physically invalid的reconstruction（objects penetrate）可能因为每个object的geometry更准确而获得更好的CD score。这提示community需要设计physical-aware的evaluation metrics。

---

## 7. 与相关工作的关系

- **SAM3D [6]**：REST3D的Stage 2基于它做single-object reconstruction，但SAM3D本身不处理scene-level physical consistency
- **Gen3DSR [1]**：divide-and-conquer from single view，但focus visual quality
- **DigitalCousins [9]**：retrieval-based，受database coverage限制
- **SAGE [40]** / **SceneFoundry [5]**：fully agentic generation，controllability差
- **PhyReCon [23]**：physically plausible neural scene reconstruction，但用multi-view input
- **PAT3D [18]**：physics-augmented text-to-3D，但focus tabletop scenes

REST3D的positioning：在single-image reconstruction的fidelity和scene generation的physical plausibility之间找到平衡，通过agent-assisted（非fully agentic）pipeline + physics-constrained optimization实现。

---

## Reference Links

- Project page: https://shirleymaxx.github.io/REST3D/
- SAM3D [6]: arXiv:2511.16624
- SAM3 [4]: ICLR 2026
- DigitalCousins [9]: CoRL 2024
- Gen3DSR [1]: 3DV 2025
- SceneGen [22]: 3DV 2026
- Isaac Gym [20]: NeurIPS 2021 Datasets and Benchmarks
- Replica [31]: arXiv:1906.05797
- ScanNet++ [43]: ICCV 2023
- CEM tutorial [11]: Annals of Operations Research
- GJK algorithm [12]: IEEE Journal of Robotics and Automation 1988
- Open-TeleVision [8]: CoRL 2024
- SAGE [40]: CVPR 2026
- World Labs Marble [36]: https://www.worldlabs.ai/blog/marble-world-model
- V-HACD convex decomposition: 用于object collision geometry

---

## 8. 个人思考与延伸方向

从Karpathy的视角，这篇paper体现了几个值得关注的trends：

1. **VLM as scene parser**：Gemini 3 Flash不只是生成text，而是作为structured scene understanding的agent，通过iterative verification loop（$A^{\text{seg}}$ + $A^{\text{ver}}$）实现robust segmentation。这比直接prompt VLM更reliable。

2. **Physics-in-the-loop optimization**：CEM + physics simulator的组合是model-free的black-box optimization，不需要differentiable physics。代价是sample efficiency低（K=2048 per iteration），但并行化在GPU上可行。

3. **Simulation-ready assets**：从vision reconstruction到simulation-ready的gap主要在physical consistency，这需要explicit physics constraints而非learned priors。

4. **Future directions**：
   - Differentiable physics for gradient-based optimization（替代CEM）
   - Articulated objects（如cabinet doors, drawers）
   - Explicit wall/ceiling geometry reconstruction
   - Video input利用temporal consistency
   - 与world model（如Marble [36]）的结合

5. **Scaling concerns**：CEM的population size K=2048对大scene（30 objects = 180 DoF）可能不够。可以考虑learned initialization或hierarchical CEM。

6. **Evaluation metric改进**：paper指出现有geometric metrics的盲区，社区需要physical-aware metrics，如simulation-based stability score、support relation accuracy等。

这篇paper的core contribution不在于novel architecture，而在于把physical scene understanding（via VLM agents）和physics-constrained optimization（via CEM + simulator）有机结合，解决了single-image reconstruction中长期被忽视的physical plausibility问题。83 percentage point的stability提升是一个convincing的demonstration。
