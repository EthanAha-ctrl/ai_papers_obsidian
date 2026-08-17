---
source_pdf: MiCADangelo Fine-Grained Reconstruction of Constrained CAD Models from
  3D Scans.pdf
paper_sha256: a6aac2b0139a12c503482b6ca549523e071654ad1dff7d9c6b73a0457ff9c300
processed_at: '2026-08-05T18:08:43-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 MiCADangelo

## 0. 一句话讲清这论文在干嘛

你拿一个真实物体(比如一个零件)扫成 3D mesh,想把它变回 SolidWorks 里那套"先画 sketch 再 extrude"的可编辑设计文件。以前的方法要么生成出来的东西没法编辑,要么细节丢光。MiCADangelo 模仿人类设计师的工序,先切片看截面、再把截面描成带约束的 sketch、最后 extrude 拉伸——而且**第一次**在 reverse engineering 里把 sketch 上的 constraint(垂直、相切、等长这些)也恢复出来。

参考: https://www.freecad.org (constraint solver 就是这套东西)

---

## 1. 为什么这个任务难——用一个类比

想象有人给你一栋建好的房子,让你还原出建筑师的施工图纸,而且要求图纸"还能改"——比如你想把二楼窗户挪一下,图纸得能让施工队按它继续干活。

3D scan 的 mesh 就像这栋房子的外墙扫描:你只知道表面长啥样,不知道哪面墙是承重墙、哪个洞是门窗、哪根梁和哪根梁应该平行。mesh 是"焊死"的几何,你改一个点其他地方不会跟着动。

真正的 CAD 模型 = sketch + constraint + extrude sequence。constraint 就是施工图纸上的"这根梁和那根梁垂直"、"这个圆弧和这条线相切"这种关系标注。改一个点,所有有约束关系的几何会自动按 design intent 一起动。

所以 reverse engineering 不只是复制几何形状,而是**反推设计师脑子里的设计意图**。

---

## 2. 以前的方法为什么不行

两大流派,各有硬伤:

**Bottom-up 派**(Point2Cyl [8], ExtrudeNet [7]):从 point cloud 直接预测一堆 cylinder 参数,拼起来近似 mesh。好处是几何贴得紧,坏处是输出不是 sketch-extrude 序列,导到 SolidWorks 里就是一堆死参数,没有 design history。而且没有 constraint 这个概念。

**Top-down 派**(CAD-SIGNet [17], CAD-Diffuser [9]):把 sketch-extrude 序列当成一段"语言",用 transformer 自回归一个 token 一个 token 生成。好处是输出完全 parametric。坏处有两个:一是 autoregressive 的 error accumulation——前面错一个 token,后面全错;二是 transformer 的 attention 容易被大结构主导,**小孔、小倒角、薄壁这种 fine-grained detail 经常被 attention mask 掉**。

参考 CAD-SIGNet: https://arxiv.org/abs/2405.20094

---

## 3. MiCADangelo 的核心 insight:人就怎么干,机器就怎么干

一个有经验的设计师拿到 scan,不会盯着 point cloud 脑补 CAD 命令流。他会做三件事:

**第一步:切几个截面看看**。在 SolidWorks 里沿 x/y/z 三个方向切几十刀,看看哪个切面上的轮廓"长得像 sketch"——通常是模型突然变粗/变细的过渡面。这些切面就是候选 sketch plane。

**第二步:在选定的 sketch plane 上描线**。把截面轮廓用 line、arc、circle 描出来,顺手加上 constraint:这两条线垂直,这个圆弧和这条线相切,这两个圆同心。这些 constraint 是设计师脑里的 design intent。

**第三步:extrude 拉伸**。把 sketch 沿平面法线方向拉成 3D solid,拉伸长度根据 mesh 实际几何决定。内层 loop 当 cut,外层当 new。

MiCADangelo 把这三步显式建模成三个 network。这跟 prior work 的根本区别:prior work 想用一个大模型 end-to-end 黑盒搞定,MiCADangelo 拆成人类工序的三个 well-posed 子任务。

---

## 4. Stage 1:Sketch Plane Detection——机器怎么"挑切面"

沿 x/y/z 各切 40 刀,共 120 个 slice。每个 slice 渲染成 $128 \times 128$ 的 binary image。然后:

- ResNet34 encoder 把每个 slice image 编码成 256 维向量 $\mathbf{f}_i$
- 加上 contextual embedding:这个 slice 是第几个(slice index)、属于哪个 axis、normalization 的 translation 和 scale
- 4 层 transformer encoder 让 120 个 slice 互相 attention
- 每个 slice 过一个 sigmoid 分类器,输出"我是不是 sketch plane"的概率
- 阈值 0.5 过滤,留下 key slice

为什么要 contextual embedding?打个比方:你看一个 cube 的 40 个 z-axis 切面,中间 38 个都是一样的正方形,只有最顶和最底是 sketch plane。光看一张图根本分不出来。必须告诉网络"你是第几个 slice",它才能基于"切片序列里形状突然变化的位置"来判断。

公式 1 那三个 embedding:
$$
\omega_i^{\text{pos}} = \mathbf{W}_{\text{pos}} \mathbf{y}_{\sigma_i}, \quad \omega_i^{\text{axis}} = \mathbf{W}_{\text{axis}} \mathbf{y}_{a_i}, \quad \omega_i^{\text{norm}} = \mathbf{W}_{\text{norm}} \boldsymbol{\eta}_i
$$

变量解释:
- $\sigma_i$:slice 在所属 axis 里的序号,0 到 39
- $\mathbf{y}_{\sigma_i}$:slice 序号的 one-hot
- $a_i \in \{0,1,2\}$:这个 slice 属于 x/y/z 哪个 axis
- $\boldsymbol{\eta}_i = (t_i^x, t_i^y, s_i)$:这个 slice 在 normalize 到 unit box 时用了多少 translation 和 scale,作用是让网络知道这个 slice 在 3D 空间里的真实尺寸

Table 4 的 ablation 很说明问题:没有 contextual embedding,F1 只有 0.296;加上之后 F1 = 0.870。**单看图像根本不够,position information 是决定性的**。

参考 ResNet: https://arxiv.org/abs/1512.03385

---

## 5. Stage 2:Constrained Sketch Parameterization——机器怎么"描线加约束"

每个 key slice 会自然分解成几个 closed loop(截面轮廓就是封闭折线)。每个 loop 单独渲染成 $128 \times 128$ binary image,然后:

- 共享的 ResNet34 encoder 把 loop image 编码成 256 维向量
- transformer encoder-decoder 输出 $n_p$ 个 primitive 的 embedding 集合 $\zeta_j \in \mathbb{R}^{d_e \times n_p}$
- 两个 head 分别预测 primitive 参数 $\mathcal{P}_j$ 和 constraint $\mathcal{C}_j$

这里的直觉是:**把"从 3D point cloud 预测 2D parametric curve"这个极难的问题,降维成"从 2D image 预测 2D parametric curve"**。后者已经有 DaVinci [37]、Vitruvion [4] 这些成熟工作。image domain 对 scan noise 天然 robust——mesh 上的 holes 和 misoriented normals 在 image 上就是些像素扰动,卷积网络不在乎。

但有个大问题:**DeepCAD 没有 constraint 标注**。怎么办?

解决方案很聪明:
1. 先在 **SketchGraphs [34]** 上 pretrain——这是 Onshape 真实 CAD sketch 的大数据集,带完整 constraint
2. 然后 fine-tune 在 **augmented SketchGraphs** 上:50% 概率用合成 random loop(Algorithm 1:随机 polygon,把一些 edge 换成 arc,加 coincident constraint 闭合),50% 用原数据。另外加 image-space noise(Algorithm 2:random resize 到 64/128/256 再 resize 回 128、Gaussian blur、foreground 附近加随机 noise pixel)

为什么要这么 augment?因为 SketchGraphs 是设计师手画的干净线稿,cross-section 是从 noisy mesh 切出来的毛边折线,domain gap 巨大。augment 就是把 SketchGraphs 的分布往 cross-section 那边拉。

Table 6 显示这个 trick 让 SCD(Sketch Chamfer Distance)从 DaVinci 的 0.827 降到 0.283——**三倍提升**,完全靠 data augmentation。

参考 SketchGraphs: https://github.com/PrincetonLabs/SketchGraphs
参考 DaVinci: https://arxiv.org/abs/2410.18900

---

## 6. Stage 3:Differentiable Extrusion——机器怎么"拉伸"

extrusion $\mathbf{e}_j = (\pi_j, t_j, \mathbf{v}_j, h_j)$ 四个量:
- $\pi_j$:sketch plane,Stage 1 已经定了
- $\mathbf{v}_j$:extrusion 方向,取 $\pi_j$ 的 normal(这是 limitation,非 axis-aligned extrude 会失败)
- $t_j \in \{new, cut\}$:由 loop nesting 决定。最外层是 new,内一层是 cut,交替下去。cut loop 当 infinite cut 处理,不优化长度
- $h_j$:**唯一需要优化的量**

优化过程:
1. 在每个 sketch $\mathcal{K}_j$ 的 loop boundary 上 sample $n_r$ 个 3D anchor point $\mathbf{r}_k$
2. 每个 anchor point 配一条 extrusion vector:$\boldsymbol{\rho}_k = \mathbf{r}_k + h_j \mathbf{v}_j$
3. 整个 loop 共享一个 $h_j$——因为 extrude 是平面整体平移,所有点位移相同
4. 从 mesh 上 sample $n_M$ 个点 $\mathbf{q}_l$
5. 每个 mesh point 找最近的 extrusion vector: $\boldsymbol{\rho}_{\min} = \arg\min_u d(\mathbf{q}_l, \boldsymbol{\rho}_u)$
6. Loss:

$$
\mathcal{L}_{\text{extr}} = \frac{1}{n_M} \sum_{l=1}^{n_M} d(\mathbf{q}_l, \boldsymbol{\rho}_{\min})^2 + \lambda \sum_{i,j} h_j^2
$$

变量解释:
- 第一项:所有 mesh 点到最近 extrusion vector 的平均距离平方,推动 extrusion vector 贴在 mesh 表面上
- 第二项:L2 正则,$h_j$ 是每个 loop 的拉伸长度,防止 trivial solution(全 $h_j=0$)
- $\lambda$:正则系数
- $i$ 索引 key slice plane,$j$ 索引 loop

为什么用 point-to-vector 而非 point-to-surface?因为 differentiable surface rendering(marching cube、SDF 之类)计算贵且梯度噪声大。这里把 extrusion 看成"一堆从 sketch boundary 长出来的射线",这些射线应该贴在 mesh 表面。射线 fit 好了,拉伸长度就对了。

AdamW 优化 200 iter,lr=2e-4。Figure 6 显示 $n_r$ 从 1 到 8 改善明显,超过 8 收益递减。

---

## 7. 实验结果——人话解读

### 7.1 主结果(Table 1)

在 DeepCAD 上,Chamfer Distance 从 CAD-SIGNet 的 0.28 降到 0.20,IoU 从 77.6 提到 80.6,ECD 从 0.74 降到 0.46。

ECD(Edge Chamfer Distance)是最能说明问题的指标——它衡量 sharp edge 保留。MiCADangelo 在 ECD 上明显赢,说明 cross-section 方法保留了 sharp edge,而 point-cloud autoregressive 方法的 edge 被 transformer smooth 掉了。

### 7.2 复杂模型(Table 2)

复杂模型(≥4 loops 或 >2 extrusions)上,MiCADangelo 的 IoU 比 CAD-SIGNet 高 20 个点。Supp Table 8 看 >8 loops 的极复杂模型:Median CD 三倍提升,IoU 从 41 提到 64。

为什么 top-down autoregressive 在复杂模型上崩?两个原因:sequence 越长 error accumulation 越严重;transformer attention 被大结构主导,小 loop 被 mask 掉。MiCADangelo 每个 loop 独立处理,没有 attention dilution 问题。

### 7.3 Constraint 的价值——deformation robustness(Table 3)

这是 paper 最 interesting 的实验。问题:重建出来的 CAD 模型被"修改"后还能不能保持结构一致?

实验:从 SketchGraphs 取 1000 个 closed-loop sketch,extrude 成 3D solid。对每个 sketch 的一个点施加 random displacement。对 ground truth,FreeCAD 的 constraint solver 会按 design intent 自动 propagate 修改。对 CAD-SIGNet 输出,只有 closed-loop 隐含的 coincident constraint,修改后几何可能崩。对 MiCADangelo 输出,有完整 constraint,FreeCAD solver 按 design intent 传播。

结果:CAD-SIGNet 修改后 CD 2.89,ECD 20.43;MiCADangelo 修改后 CD 0.38,ECD 1.29。**ECD 差 15 倍**。

Figure 5 视觉对比:CAD-SIGNet 修改后几何严重扭曲(线段不再共面、圆弧变椭圆),MiCADangelo 修改后保持 design intent。

**这是 reverse engineering 的根本目的**——拿到模型是为了进一步改设计,不是为了看一眼。没 constraint 的 CAD 模型改一个点几何就崩,本质还是 mesh。有 constraint 的 CAD 模型才是真正的 CAD。

### 7.4 真实扫描数据 CC3D(Table 7)

CC3D 是真实 3D scan,有 holes、misoriented normals、noise。训练在 DeepCAD,测试在 CC3D。

MiCADangelo CD 降 42%,IoU 提 8 个点,IR 减半,ECD 降 32%。image-based representation 对 noise 天然 robust,point-cloud 方法对 missing point 和 noise 敏感。

参考 SHARP / CC3D: https://sharp3d.cc/

---

## 8. 几个值得 internalize 的 intuition

**Intuition 1:Cross-section 是 3D-to-2D 的"自然投影"**。CAD 模型本身就是从 2D sketch extrude 出来的,cross-section 切到 sketch plane 时自然回到 2D sketch domain。比把整个 3D point cloud 喂给 transformer 让它"脑补"2D sketch 信息密度高得多。

**Intuition 2:Image 是 robust representation for noisy input**。3D scan 的 noise、holes、misoriented normals 在 image 上是轻微像素扰动,卷积网络 robust。在 point cloud 上每个 noise 都是 3D 坐标偏移,影响几何参数预测。

**Intuition 3:Decouple parametric recovery from geometric fitting**。Sketch primitive + constraint 是离散+连续混合预测,extrusion length 是纯连续优化。把它们解耦到不同 stage,各自用最适合的算法——网络做离散决策,优化做连续拟合——比一个 autoregressive model 同时承担两种任务更稳定。

**Intuition 4:Constraint 是 reverse engineering 的真正目的**。光重建几何不够,要重建"可编辑的设计意图"。没 constraint 改一个点几何崩,有 constraint 改一个点 design intent 自动 propagate。这是从"几何复制"升级到"设计理解"的关键。

**Intuition 5:Stage-level robustness > end-to-end optimization**。Supp Table 9 显示即使 plane detection 不准,后续 stage 能补救。autoregressive 一旦前面 token 错后面全错,MiCADangelo 即使 plane detection 漏一个,sketch parameterization 和 extrusion 在已检测 plane 上仍能正确工作。

---

## 9. 限制

1. 只支持 extrusion,不支持 revolution、sweep、loft、fillet。这是整个领域的限制。
2. Extrusion 方向固定为 sketch plane normal,非 axis-aligned 的 extrude 会失败。
3. Sketch primitive 只支持 line/arc/circle,不支持 B-spline,做不了 freeform 设计。
4. Invisible cut extrusion 处理不了——sketch loop 在最终几何上不可见的 cut(比如完全切除的内部 cavity)无法从 cross-section 反推。
5. Failure case:extrusion 高度估错、arc 被简化成 line segments、sketch plane 漏检。

---

## 10. 我的读后感

这篇 paper 的核心贡献与其说是某个网络结构创新,不如说是**任务分解方式的重新设计**。把"point cloud → CAD 命令流"这个 ill-posed end-to-end 问题,拆成"image → sketch plane"、"image → sketch + constraint"、"optimization → extrusion length"三个 well-posed 子问题,每个都用该方向最成熟的技术。

最有想象力的是 constraint 的引入。reverse engineering 一直被当成"几何重建"任务,这篇把它升级成"design intent 重建"任务。这跟 LLM 领域从"surface form generation"到"semantic understanding"的演进很像——光生成像不像不够,要生成得"懂"。

如果后续工作能解决 B-spline、revolution、sweep 这些限制,加上 constraint solver in-the-loop 训练,reverse engineering 可能真的能从实验室走向工业级 CAD 软件插件。

参考 FreeCAD API: https://www.freecadweb.org/wiki/Python
参考 Onshape: https://www.onshape.com
参考 SolidWorks: https://www.solidworks.com

---

# MiCADangelo: 把 3D Scan 反向工程成带约束的参数化 CAD 模型

## 1. 大背景与 Motivation:为什么这个任务是硬骨头

CAD reverse engineering 的目标是把 3D scan(通常是 mesh)转回 parametric CAD model——也就是设计师在 SolidWorks/Onshape/FreeCAD 里那一连串 sketch + extrude 的"设计历史"。mesh 只是一个表面采样,丢失了 sketch plane 在哪、哪条线是 line 哪条是 arc、哪些 primitive 之间应该 tangent/perpendicular、extrude 多深这些**设计意图 (design intent)**。没有这些信息,下游想改一个圆角、想拉伸一个面就完全没法做,等于把模型"焊死"了。

现有的 deep learning 方法分成两派,各有硬伤:

- **Bottom-up (geometry-driven)**:代表是 Point2Cyl [8], ExtrudeNet [7], SecAD-Net [20]。它们从 point cloud 直接预测每个 extrusion cylinder 的参数。优点是 local geometric fidelity 好,因为每一步都贴着 scan 的几何走。缺点是输出不是真正的 sketch-extrude sequence,不能直接导到 CAD 软件里编辑,而且没有 constraint 这个概念。
- **Top-down (language-based)**:代表是 CAD-SIGNet [17], CAD-Diffuser [9], Hierarchical Neural Coding [19]。把 sketch-extrude sequence 当成一种"语言"自回归生成,像 transformer 翻译一样。优点是输出完全 parametric、有 design history。缺点是 autoregressive 容易 error accumulation,而且 cross-attention 经常被大结构主导,**fine-grained 的几何细节(小孔、小倒角、薄壁)经常被丢掉**。

MiCADangelo 的核心 insight 很朴素:**人是怎么做的就怎么做**。一个有经验的设计师拿到 scan,不会从 point cloud 直接脑补一段 CAD 命令流。他会做三件事:(i) 在软件里切几个 cross-section 看看哪几个 slice 对应原始 sketch plane;(ii) 在选定的 sketch plane 上用 line/arc/circle 把 contour 描出来,顺手加上 coincident/tangent/perpendicular 等约束;(iii) 用 extrude 把 sketch 拉成 3D solid。MiCADangelo 把这三步显式建模成三个 network,而且**第一次**把 sketch constraints 放进了 3D reverse engineering 的 reconstruction pipeline 里。

参考链接:
- DeepCAD dataset: https://deepcad3d.github.io/
- Fusion 360 Gallery: https://autodesk.github.io/fusion360gallery/dataset.html
- SketchGraphs: https://github.com/PrincetonLabs/SketchGraphs
- CAD-SIGNet (main baseline): https://arxiv.org/abs/2405.20094

---

## 2. 形式化定义(Section 3):把 CAD 模型数学化

先把所有要恢复的东西定义清楚,这样后面公式才有 anchor。

- **Sketch primitive** $\mathbf{p}$:三种基本 2D 实体
  - Line: 起止点 $(x_s, y_s), (x_e, y_e) \in \mathbb{R}^2$
  - Circle: center $(x_c, y_c)$ + radius $r$
  - Arc: start $(x_s,y_s)$ + mid $(x_m,y_m)$ + end $(x_e,y_e)$,三点共圆
- **Sketch constraint** $\mathbf{c} = (\mathbf{p}_i, \mathbf{p}_j, c_t)$:作用在两个 primitive 上的关系 $c_t \in \{$coincident, concentric, equal, fix, horizontal, midpoint, normal, offset, parallel, perpendicular, quadrant, tangent, vertical$\}$。其中 vertical/horizontal 作用在单个 primitive 上。
- **Constrained sketch** $\mathcal{K} = (\mathcal{P}, \mathcal{C})$:primitive 集合 + constraint 集合。这是整个 pipeline 的核心输出对象。
- **Sketch plane** $\pi = (\mathbf{o}, \mathbf{n})$:origin $\mathbf{o} \in \mathbb{R}^3$ + normal $\mathbf{n} \in \mathbb{R}^3$。
- **Cross-section slice** $S$:3D mesh $\mathbf{M}$ 和 slicing plane $\pi$ 相交所得到的 line segments 集合。自然形成多个 connected components。
- **Closed loop** $\mathbf{L}$:首尾相接、不自交的封闭折线。CAD 的 cross-section slice 通常天然就是 closed loops。
- **Extrusion** $\mathbf{e} = (\pi, t, \mathbf{v}, h)$:plane $\pi$ + type $t \in \{new, cut\}$ + direction $\mathbf{v} \in \mathbb{R}^3$ + length $h \in \mathbb{R}$。
- **CAD model** $\mathbf{C} = \{(\mathcal{K}_j, \mathbf{e}_j)\}_{j=1}^{n_s}$:就是 sketch-extrude 序列。

Problem statement 就是:给 mesh $\mathbf{M}$,恢复 $\mathbf{C}$。

为什么这种形式化重要:因为它显式把 constraint 当成一等公民,而不像 prior work 只用 closed-loop 隐含的 coincident。后续 Table 3 那个 deformation robustness 实验就是基于这套定义做的——它证明有 explicit constraint 的 sketch 在被"扰动一个点"之后,FreeCAD 的 constraint solver 会按 design intent 自动传播修改,而没有 constraint 的 sketch 会几何崩溃。

---

## 3. Pipeline 总览(Section 4.1)

三个 stage 串起来,Figure 2 把架构画得很清楚:

```
3D Mesh M
   │
   ├─ sample N=120 slicing planes (40 per axis along x,y,z)
   │     → cross-section slices {S_i}_{i=1}^N
   │     → rasterize to binary images X_i ∈ {0,1}^{128×128}
   ▼
[Stage 1: Sketch Plane Detection Network]
   - ResNet34 encoder f_enc → f_i ∈ R^d
   - contextual embedding ω_i = pos + axis + norm
   - 4-layer 4-head transformer encoder f_plane
   - binary classifier f_key → ŷ_i^key ∈ [0,1]
   - threshold τ=0.5 → S_key = {S_i | ŷ_i^key ≥ τ}
   ▼
[Stage 2: Constrained Sketch Parameterization Network]
   - 每个 key slice 分解成 closed loops {L_j}
   - 每个 loop rasterize → X_j ∈ {0,1}^{128×128}
   - 共享 ResNet34 encoder f_enc → f_j ∈ R^d
   - transformer encoder-decoder f_loop → ζ_j ∈ R^{d_e × n_p}
   - two heads: f_prim → P_j, f_ctr → C_j
   ▼
[Stage 3: Differentiable Extrusion Optimization]
   - 对每个 sketch K_j,sample n_r 个 anchor points r_k 沿 loop boundary
   - 每个 loop 共享一个 learnable h_j
   - extrusion vector ρ_k = r_k + h_j * v_j
   - sample n_M 个点 q_l 从 mesh M
   - L_extr = mean min distance(q_l, ρ_u)^2 + λ Σ h_j^2
   - gradient descent 200 iters, lr=2e-4
   ▼
Assemble → final parametric CAD model C
```

关键设计直觉:把"从 point cloud 直接 autoregressive 出 CAD 命令"这个非常 ill-posed 的 task,**分解成三个各自 well-posed 的子任务**。每个子任务都有强 inductive bias:plane detection 用 transformer 在 slice 序列上做 contextual reasoning;sketch parameterization 用 image 作为输入,绕开了 point cloud 上做 fine-grained 2D curve fitting 这个难题;extrusion 用 differentiable optimization 直接拟合 mesh 几何,避免 autoregressive 的 error accumulation。

---

## 4. Stage 1: Sketch Plane Detection Network(Section 4.2)

### 4.1 输入设计

沿着 x/y/z 三个 axis 各采样 40 个等间距 slicing plane,共 N=120 个 slice。每个 slice 投影到 2D,normalize 到 unit bounding box,render 成 $128 \times 128$ 的 binary image $\mathbf{X}_i \in \{0,1\}^{H \times W}$。

为什么用 raster image 而不是 line segments 的 parametric 表示?因为后面要复用 sketch parameterization network,它本来就是 image-to-sketch 的,而且 image 对 noise(扫描 holes、misoriented normals)更 robust。

### 4.2 Contextual Embedding(公式 1, 2)

光看单个 slice 的图像无法判断它是不是 sketch plane,因为同一个 slice 在序列里的"位置"很重要——比如一个 cube 的 40 个 z-axis slice,只有最顶和最底两个是 sketch plane,中间 38 个都是一样的正方形。所以必须把 slice 在序列中的 spatial context 编进去。

$$
\omega_i^{\text{pos}} = \mathbf{W}_{\text{pos}} \mathbf{y}_{\sigma_i}, \quad \omega_i^{\text{axis}} = \mathbf{W}_{\text{axis}} \mathbf{y}_{a_i}, \quad \omega_i^{\text{norm}} = \mathbf{W}_{\text{norm}} \boldsymbol{\eta}_i
$$

变量解释:
- $\sigma_i \in \{0, \ldots, N-1\}$:slice 在所属 axis 内的 index
- $\mathbf{y}_{\sigma_i} \in \{0,1\}^N$:slice index 的 one-hot
- $a_i \in \{0,1,2\}$:axis identifier (x/y/z)
- $\mathbf{y}_{a_i} \in \{0,1\}^3$:axis 的 one-hot
- $\boldsymbol{\eta}_i = (t_i^x, t_i^y, s_i) \in \mathbb{R}^3$:normalization 参数——translate x/y + scale,这是 reverse 回 original 3D 坐标用的事实载体
- $\mathbf{W}_{\text{pos}} \in \mathbb{R}^{d \times N}$, $\mathbf{W}_{\text{axis}} \in \mathbb{R}^{d \times 3}$, $\mathbf{W}_{\text{norm}} \in \mathbb{R}^{d \times 3}$:三个 learnable embedding 矩阵

最终的 contextual embedding 是三者相加:

$$
\mathbf{z}_i = \mathbf{f}_i + \omega_i, \quad \omega_i = \omega_i^{\text{pos}} + \omega_i^{\text{axis}} + \omega_i^{\text{norm}}
$$

这里 $\mathbf{f}_i = f_{\text{enc}}(\mathbf{X}_i) \in \mathbb{R}^d$ 是 ResNet34 encoder 出来的视觉特征,$d=256$。

Intuition:这相当于 ViT 里的 positional encoding,但更丰富——除了"我在第几个位置",还有"我属于哪个 axis"(因为 x/y/z 三个 axis 上的 slice 序列语义不一样)、还有"我在 3D 空间里被 normalize 了多大尺度"。norm embedding 让 transformer 知道 slice 在 3D 中真实几何尺度,不然两个相同 pattern 的 slice 一个是外层 box 一个是内层小孔就分不清了。

### 4.3 Transformer Encoder + Binary Classifier(公式 3, 4)

$$
(\mathbf{h}_1, \ldots, \mathbf{h}_N) = f_{\text{plane}}(\mathbf{z}_1, \ldots, \mathbf{z}_N)
$$

$f_{\text{plane}}$ 是 4 层 4 头 transformer encoder,embedding dim 256。它让 120 个 slice 之间互相 attention,这样 transformer 能基于"slice 之间形状如何变化"来判断哪个是 sketch plane。比如一个 slice 形状突然从"小矩形"跳到"大矩形",那中间那个 slice 大概率是 sketch plane——这是 cross-section 切到 extrude 起始面的特征。

$$
\hat{y}_i^{\text{key}} = f_{\text{key}}(\mathbf{h}_i)
$$

$f_{\text{key}}$ 是 linear + sigmoid,输出该 slice 是 key sketch plane 的概率。用 BCE loss 训练,supervision 来自 DeepCAD 里的真实 sketch plane(见 supp 7.1.1,具体做法是把每个 ground-truth plane 投影到最近的 slicing plane 上做 label)。

阈值 $\tau = 0.5$,最终 $S_{\text{key}} = \{S_i \mid \hat{y}_i^{\text{key}} \geq \tau\}$。

### 4.4 Plane Detection 性能

Table 4 的 ablation 显示 contextual embedding 的威力:Precision 从 0.317 飙到 0.894,F1 从 0.296 到 0.870。没有 contextual embedding,单看 image 几乎无法判断哪个 slice 是 sketch plane——因为同 axis 内大部分 slice 长得一样。

Table 5 显示在 Fusion360 和 CC3D 这种 unseen 分布上,F1 还有 0.820 和 0.777,泛化性相当不错。

Supp Table 11 把 plane detection 和 CAD-SIGNet 比较:CAD-SIGNet 在它自己的 design-plane detection 任务上 F1 = 0.686,MiCADangelo 在 cross-section plane detection 上 F1 = 0.870。注意两者任务定义不同——CAD-SIGNet 是在 DeepCAD 的原始 design plane 上做的,而 MiCADangelo 是在预处理后的 canonical cross-section plane 上做的(见 supp 7.1.1,把 extent_type 处理掉、normal flip 到 positive axis)。

---

## 5. Stage 2: Constrained Sketch Parameterization Network(Section 4.3)

### 5.1 为什么从 loop image 入手

每个 key slice $S_i$ 自然分解成多个 closed loops $\{\mathbf{L}_j\}_{j=1}^L$。每个 loop 单独 rasterize 成 $128 \times 128$ binary image $\mathbf{X}_j$。这是非常关键的设计:**把"从 3D point cloud 预测 2D parametric curve"这个难题,降维成"从 2D image 预测 2D parametric curve"**。后者已经有 DaVinci [37], Vitruvion [4], Picasso [36], SketchGen [35] 等成熟工作,而且 image domain 对 scan noise 天然 robust。

### 5.2 网络结构(公式 5, 6)

共享同一个 ResNet34 encoder $f_{\text{enc}}$(和 Stage 1 共享,先在 SketchGraphs 上 pretrain,frozen 后再 fine-tune Stage 1):

$$
\mathbf{f}_j = f_{\text{enc}}(\mathbf{X}_j) \in \mathbb{R}^d
$$

然后过 transformer encoder-decoder $f_{\text{loop}}$:

$$
\zeta_j = f_{\text{loop}}(\mathbf{f}_j) \in \mathbb{R}^{d_e \times n_p}
$$

$\zeta_j$ 是 $n_p$ 个 primitive 的 embedding 集合,每个 primitive 一个 $d_e$ 维向量。架构借鉴 DaVinci [37]——single-stage,而非 autoregressive,避免了 autoregressive 那种 error accumulation。

两个 head 分别处理:

$$
\mathcal{P}_j = f_{\text{prim}}(\zeta_j), \quad \mathcal{C}_j = f_{\text{ctr}}(\zeta_j)
$$

- $f_{\text{prim}}: \mathbb{R}^{d_e \times n_p} \to \mathbb{Q}^{n_p}$:parameterization head,输出 quantized primitive 参数。$\mathbb{Q}$ 是 quantized space,DeepCAD 里坐标是 8-bit 量化(0-255)。
- $f_{\text{ctr}}: \mathbb{R}^{d_e \times n_p} \to \mathbb{Q}^{n_c}$:constraint prediction head,输出 quantized constraint。

注意 $n_c$(constraint 数量)和 primitive 数量关系:每两个 primitive 之间可能有 constraint,所以 $n_c$ 通常是 $O(n_p^2)$ 的。

### 5.3 训练数据策略(关键直觉)

这是这个 stage 最巧妙的地方:**DeepCAD 没有 constraint 标注**,所以不能直接在 DeepCAD 上训 sketch parameterization。

解决方案:
1. 先在 **SketchGraphs [34]** 上 pretrain——SketchGraphs 是 Onshape 真实 CAD sketch 的大规模数据集,带完整 constraint 标注。
2. 然后 **fine-tune 在 augmented SketchGraphs** 上:50% 用合成 random loop sketch(Algorithm 1,生成随机 polygon,把一些 edge 替换成 arc,加 coincident constraint 闭合 loop),50% 用原 SketchGraphs。另外加 image-space noise(Algorithm 2,Algorithm 3:resize 到 random resolution 再 resize 回 128,加 Gaussian blur,在 foreground 像素附近加 noise)。

为什么这么 augment?因为 SketchGraphs 的 sketch 是"设计师手画的干净线稿",而 cross-section slice 是"从 noisy mesh 切出来的有毛边的折线"。domain gap 巨大。augment 就是把 SketchGraphs 的 distribution 朝 cross-section 那边拉。Table 6 显示这个 trick 让 SCD 从 0.827(DaVinci)降到 0.283。

### 5.4 输出形式

每个 key slice 上的每个 loop 输出一个 constrained sketch $\mathcal{K}_j = (\mathcal{P}_j, \mathcal{C}_j)$。这就是直接可以喂给 FreeCAD / Onshape 的 sketch 对象,完全 parametric、可编辑,而且 constraint 让后续修改自动传播。

---

## 6. Stage 3: Differentiable Extrusion Optimization(Section 4.4)

### 6.1 Extrusion 的确定与待定部分

extrusion $\mathbf{e}_j = (\pi_j, t_j, \mathbf{v}_j, h_j)$:
- $\pi_j$:sketch plane,由 Stage 1 选出来的 key slice 决定
- $\mathbf{v}_j$:extrusion direction,取 $\pi_j$ 的 normal 方向(这是 limitation——非 axis-aligned 的 extrude 会失败)
- $t_j \in \{new, cut\}$:由 loop nesting 决定。最外层是 new,内一层是 cut,再内一层是 new,交替下去。cut loop 当作 infinite cut(不优化 length)
- $h_j$:**唯一需要优化的量**

### 6.2 Anchor Points 与 Extrusion Vector(公式 7)

对每个 sketch $\mathcal{K}_j$,沿 loop boundary sample $n_r$ 个 3D anchor point $\{\mathbf{r}_k\}_{k=1}^{n_r} \in \mathbb{R}^3$。每个 anchor point 配一个 extrusion vector:

$$
\mathbf{F}: \mathbb{R}^3 \times \mathbb{R}^3 \to \mathbb{R}^3, \quad \mathbf{F}(\mathbf{r}_k, h_j \mathbf{v}_j) := \mathbf{r}_k + h_j \mathbf{v}_j = \boldsymbol{\rho}_k
$$

变量:
- $\mathbf{r}_k$:loop boundary 上第 k 个 anchor point(3D 坐标)
- $h_j$:这个 loop 共享的 learnable extrusion length(scalar)
- $\mathbf{v}_j$:这个 loop 所在 sketch plane 的 normal(unit vector)
- $\boldsymbol{\rho}_k$:从 $\mathbf{r}_k$ 出发沿 $\mathbf{v}_j$ 方向延伸 $h_j$ 长度的终点

注意 $h_j$ 是**整个 loop 共享一个**,而不是每个 anchor point 独立——因为 extrude 是 sketch plane 整体平移,所有点位移相同。这样参数量小,优化稳定。

### 6.3 Point-to-Vector Distance 与 Loss(公式 8)

从 mesh M 上 sample $n_M$ 个点 $\mathcal{Q} = \{\mathbf{q}_l\}_{l=1}^{n_M} \in \mathbb{R}^3$。考虑所有 loop 和所有 key slice 的 extrusion vector 集合 $\{\boldsymbol{\rho}_u\}_{u=1}^{n_r + n_L + n_{\text{key}}}$。

对每个 mesh point $\mathbf{q}_l$,找最近的 extrusion vector:

$$
\boldsymbol{\rho}_{\min} := \arg\min_u d(\mathbf{q}_l, \boldsymbol{\rho}_u)
$$

这里 $d(\mathbf{q}_l, \boldsymbol{\rho}_u)$ 是 point 到 line segment 的距离(点到 $\boldsymbol{\rho}_u$ 这条从 $\mathbf{r}_u$ 到 $\mathbf{r}_u + h_u \mathbf{v}_u$ 的线段的最短距离)。

总 loss:

$$
\mathcal{L}_{\text{extr}} = \frac{1}{n_M} \sum_{l=1}^{n_M} d(\mathbf{q}_l, \boldsymbol{\rho}_{\min})^2 + \lambda \sum_{i,j} h_j^2
$$

变量:
- 第一项:mean squared point-to-vector distance,推动 extrusion vector 覆盖 mesh 表面
- 第二项:L2 regularization on $h_j$,防止 trivial solution $h_j = 0$(所有 extrusion 长度为 0,extrusion vector 退化为 anchor point,虽然 fit 不好但避免一些 pathological case)。$\lambda$ 是 scaling factor
- $i$ 索引 key slice plane,$j$ 索引该 slice 内的 loop

为什么 point-to-vector 而非 point-to-surface?因为 differentiable surface rendering(像 differentiable marching cube / SDF)计算昂贵且梯度噪声大。point-to-vector 是把 extrusion 看成"一堆从 sketch boundary 长出来的射线",这些射线应该贴在 mesh 表面上。射线 fit 好了,extrusion 长度自然就对了。

### 6.4 优化

joint optimization:所有 key slice 上所有 loop 的 $h_j$ 一起 gradient descent 200 iterations,AdamW,lr=2e-4。

Figure 6 ablation:extrusion vector 数量从 1 增到 8,CD 和 ECD 都明显改善;超过 8 收益递减。inference time 几乎不受 vector 数量影响(因为 loss 计算复杂度主要在 mesh sampling point 数,不在 vector 数)。

---

## 7. 实验结果解读

### 7.1 主结果(Table 1)

| Method | DeepCAD Med.CD↓ | IoU↑ | IR↓ | ECD↓ | Fusion360 Med.CD↓ | IoU↑ | IR↓ | ECD↓ |
|---|---|---|---|---|---|---|---|---|
| DeepCAD [6] | 9.64 | 46.7 | 7.1 | - | 89.2 | 39.9 | 25.2 | - |
| Point2Cyl [8] | 4.27 | 73.8 | 3.9 | - | 4.18 | 67.5 | 3.2 | - |
| CAD-Diffuser [9] | 3.02 | 74.3 | 1.5 | - | 3.85 | 63.2 | 1.7 | - |
| CAD-SIGNet [17] | 0.28 | 77.6 | 0.9 | 0.74 | 0.56 | 65.6 | 1.6 | 4.14 |
| **Ours** | **0.20** | **80.6** | 2.6 | **0.46** | **0.48** | **68.7** | 3.2 | **2.66** |

读法:
- **Med.CD**(median Chamfer Distance):surface 几何相似度,MiCADangelo 在两个 dataset 上都最好。CD 越低越好。
- **IoU**:体积重叠,越高越好。MiCADangelo 80.6 / 68.7。
- **IR**(Invalidity Ratio):导出 mesh 失败的比例。注意 CAD-SIGNet 在这里很低(0.9 / 1.6)是因为它用了 **test-time sampling**——生成多个 candidate 选最好的,不用 sampling 时 IR 是 4.4 / 9.3。MiCADangelo 不用 sampling,IR 2.6 / 3.2,虽然比 CAD-SIGNet-with-sampling 高,但比 CAD-SIGNet-without-sampling 低。
- **ECD**(Edge Chamfer Distance):这是衡量 sharp edge 保留的关键指标。MiCADangelo 0.46 vs CAD-SIGNet 0.74 on DeepCAD,2.66 vs 4.14 on Fusion360。**这正是 fine-grained detail 的体现**——cross-section 方法保留了 sharp edge,而 point-cloud autoregressive 方法 edge 被 smooth 掉了。

### 7.2 复杂模型上的优势(Table 2)

| Method | ≥4 Loops CD↓ | IoU↑ | >2 Extrusions CD↓ | IoU↑ |
|---|---|---|---|---|
| CAD-SIGNet | 1.34 | 49.2 | 3.95 | 40.6 |
| **Ours** | **0.37** | **68.3** | **0.46** | **64.8** |

读法:复杂模型(≥4 loops 或 >2 extrusions)上 MiCADangelo 大幅领先。IoU 提升接近 20 个点。Supp Table 8 进一步看 >8 loops 的极复杂模型:Mean CD 5.47 vs 6.48,Median CD 0.45 vs 1.64(三倍提升),IoU 64.30 vs 41.07。**这印证了 paper 的核心 claim——cross-section 方法在 fine-grained detail 上有结构性优势**。

为什么 top-down autoregressive 方法在复杂模型上崩?因为 sequence 越长 error accumulation 越严重,而且 transformer 的 attention 容易被 dominant structure 主导,小 loop 被 mask 掉。MiCADangelo 把每个 loop 独立处理,不存在 attention dilution 问题。

### 7.3 Constraint 的价值——deformation robustness(Table 3, Section 5.2)

这是 paper 最 interesting 的实验。问题:重建出来的 CAD 模型在被"修改"后还能不能保持结构一致?

实验设计:
1. 从 SketchGraphs 取 1000 个 closed-loop sketch,extrude 成 3D solid(用 FreeCAD API [3])
2. 对每个 sketch 的一个点施加 random displacement
3. 对 ground truth:FreeCAD 的 constraint solver 会按 design intent 自动 propagate 修改(比如一个点动了,tangent 的另一条线也跟着调整保持 tangent)
4. 对 CAD-SIGNet 的输出:它只有 closed loop 隐含的 coincident constraint,修改后几何可能崩
5. 对 MiCADangelo 的输出:有完整 constraint,FreeCAD solver 按 design intent 传播
6. 度量修改后的 solid 与 ground-truth 修改后 solid 的差异

结果:

| Method | Med.CD↓ | IoU↑ | IR↓ | ECD↓ |
|---|---|---|---|---|
| CAD-SIGNet | 2.89 | 57.4 | 3.5 | 20.43 |
| **Ours** | **0.38** | **81.1** | 4.3 | **1.29** |

CD 差 7.6 倍,ECD 差 15.8 倍。Figure 5 视觉上看:CAD-SIGNet 的修改后模型几何严重扭曲(线段不再共面、圆弧变椭圆),MiCADangelo 的修改后模型保持 design intent,几何一致。

这是 reverse engineering 的根本目的——拿到模型不只是看一眼,是为了进一步改设计。没 constraint 的 CAD 模型本质上还是 mesh,改不动。有 constraint 的 CAD 模型才是真正的 CAD。

### 7.4 真实扫描数据 CC3D(Table 7, Figure 7)

CC3D [41, 42] 包含真实 3D scan,有 holes、misoriented normals、noise。cross-dataset evaluation:训练在 DeepCAD,测试在 CC3D。

| Method | Med.CD↓ | IoU↑ | IR↓ | ECD↓ |
|---|---|---|---|---|
| CAD-SIGNet | 2.90 | 42.6 | 4.4 | 8.68 |
| **Ours** | **1.69** | **50.8** | 2.2 | **5.93** |

CD 降 42%,IoU 提 8 个点,IR 减半,ECD 降 32%。image-based representation 对 noise 天然 robust,而 point-cloud-based 方法对 missing point 和 noise 敏感。Figure 7 视觉对比,MiCADangelo 输出明显更干净。

### 7.5 N (slice 数量) Ablation(Supp Table 10)

| N | Mean CD↓ | Median CD↓ | IR↓ | IoU↑ | ECD↓ |
|---|---|---|---|---|---|
| 10 | 3.20 | 0.26 | 4.5 | 78.2 | 0.54 |
| 20 | 3.17 | 0.25 | 4.3 | 78.4 | 0.53 |
| 40 | 2.27 | 0.20 | 2.6 | 80.6 | 0.46 |

N=40 是 accuracy 和 compute 的 sweet spot。注意即使是 N=10,median CD 也有 0.26——pipeline 对 slice 密度不是非常敏感,这跟 point cloud 方法依赖 dense sampling 形成对比。

### 7.6 Contextual Embedding 对整体 CAD reconstruction 的影响(Supp Table 9)

| Contextual Emb. | Mean CD↓ | Median CD↓ | IR↓ | IoU↑ | ECD↓ |
|---|---|---|---|---|---|
| ✗ | 8.80 | 0.39 | 3.5 | 69.4 | 2.25 |
| ✓ | 2.27 | 0.20 | 2.6 | 80.6 | 0.46 |

Mean CD 从 8.80 降到 2.27,这是接近 4 倍。但有意思的是 median CD 只从 0.39 到 0.20,说明:**大部分简单模型即使没 contextual embedding 也能 reconstruct 得不错**(因为后续 stage 能补救),但**少数复杂模型严重失败**(拉高 mean)。contextual embedding 主要救的是复杂 case。

### 7.7 各 stage 独立评估(Supp Table 11)——error accumulation 分析

| Stage | Compared with | MiCADangelo | Baseline |
|---|---|---|---|
| Plane Detection (F1) | CAD-SIGNet | **0.870** | 0.686 |
| Sketch Param. (SCD) | DaVinci / Vitruvion | **0.283** | 0.827 / 1.236 |
| Extrusion (CD) | Point2Cyl | **10.1** | 27.9 |

每个 stage 都比对应 prior work 好。这说明 MiCADangelo 的优势不仅来自某个单点突破,而是**三个 stage 各自都比 prior work 更 well-posed**——cross-section 让 plane detection 更容易;image 让 sketch parameterization 更容易;differentiable optimization 让 extrusion 不依赖 autoregressive。

---

## 8. 与 Cad-SIGNet 的根本区别

CAD-SIGNet [17] 是这个 paper 的主要 baseline,也是 current SOTA。它的 pipeline:
- Point cloud → PointNet++ encoder
- Transformer decoder 自回归生成 sketch-extrude sequence 的 token(类似 DeepCAD 的 command representation)
- 用 sketch instance guided attention 改善精度

为什么 MiCADangelo 能赢?根本原因是**任务分解方式不同**:

| 维度 | CAD-SIGNet | MiCADangelo |
|---|---|---|
| 输入 | Point cloud (8K points) | Multi-plane cross-section images |
| Sketch 来源 | Autoregressive token prediction | Image-to-sketch 转换 |
| Constraint | 无(只有 closed-loop 隐含 coincident) | 显式从 image 预测 |
| Extrusion | Token 的一部分,autoregressive | Differentiable optimization |
| Fine detail | 受 transformer attention dilution 影响 | 每个 loop 独立处理 |
| Error accumulation | 严重(autoregressive) | 弱(各 stage 相对独立) |

CAD-SIGNet 的优势是 IR 低(用 sampling)。MiCADangelo 的优势是 CD/ECD/IoU 全面好,尤其是 fine-grained detail 上。

---

## 9. Limitations 与未来方向

1. **只支持 extrusion**:不支持 revolution、sweep、loft、fillet。这是所有近期 sketch-extrude 方法 [17, 20, 8] 的共同限制。
2. **Extrusion direction 固定为 sketch plane normal**:非 axis-aligned 的 extrude 会失败。Supp 提到 future work 会优化 $\mathbf{v}_j$ 而非固定。
3. **不支持 B-spline**:sketch primitive 只有 line/arc/circle,无法处理 freeform 设计。
4. **Invisible cut extrusion 处理不了**:visible cut(loop 在 cross-section 可见)能处理,但 invisible cut(sketch loop 在最终几何上不可见)不能。比如一个被完全切除的内部 cavity,从 cross-section 看不到原 sketch。
5. **Failure cases**(Supp Figure 11):extrusion 高度估错、arc 被简化成 line segments、sketch plane 漏检。

---

## 10. Build Intuition:为什么这个 approach 工作

我觉得这篇 paper 最值得 internalize 的几个 intuition:

**Intuition 1:Cross-section 是 3D-to-2D 的"自然投影"**。CAD 模型本身就是从 2D sketch extrude 出来的,所以 cross-section 切到 sketch plane 时自然回到 2D sketch domain。这比把整个 3D point cloud 喂给 transformer 让它"脑补"2D sketch 信息密度高得多。

**Intuition 2:Image 是 robust representation for noisy input**。3D scan 有 noise、holes、misoriented normals。在 image domain 处理 cross-section,这些 noise 变成 image 上的轻微像素扰动,卷积网络天然 robust。而 point cloud 上每个 noise 都是 3D 坐标偏移,影响几何参数预测。

**Intuition 3:Decouple parametric recovery from geometric fitting**。Sketch primitive + constraint 是离散+连续混合的预测任务,extrusion length 是纯连续优化任务。把它们解耦到不同 stage,各自用最适合的算法——网络做离散决策,优化做连续拟合——比一个 autoregressive model 同时承担两种任务更稳定。

**Intuition 4:Constraint 是 reverse engineering 的真正目的**。光重建几何形状不够,要重建"可编辑的设计意图"。没 constraint 的 CAD 模型改一个点几何就崩,有 constraint 的 CAD 模型改一个点 design intent 自动 propagate。这是 reverse engineering 从"几何复制"升级到"设计理解"的关键。

**Intuition 5:Stage-level robustness > end-to-end optimization**。Supp Table 9 显示即使 plane detection 不准,后续 stage 能补救。这种"各 stage 都有 fallback"的设计比端到端 autoregressive 更 robust。autoregressive 一旦前面 token 错,后面全错;MiCADangelo 即使 plane detection 漏一个,sketch parameterization 和 extrusion 在已检测的 plane 上仍能正确工作。

---

## 11. 相关工作索引

如果想深入这个领域,推荐按顺序读:

1. **DeepCAD [6]** — sketch-extrude 序列数据集 + 生成模型。基础。
   - https://deepcad3d.github.io/
2. **Fusion 360 Gallery [18]** — 另一个 CAD 设计序列数据集。
   - https://autodesk.github.io/fusion360gallery/
3. **SketchGraphs [34]** — 大规模 constrained sketch 数据集,Stage 2 训练用。
   - https://github.com/PrincetonLabs/SketchGraphs
4. **Vitruvion [4]** — 早期 constrained sketch 生成,autoregressive。
   - https://arxiv.org/abs/2104.05568
5. **SketchGen [35]** — constrained sketch 生成,VAE-based。
6. **DaVinci [37]** — single-stage constrained sketch inference,MiCADangelo 的 Stage 2 直接借鉴。
   - https://arxiv.org/abs/2410.18900
7. **Point2Cyl [8]** — bottom-up 代表,baseline。
   - https://arxiv.org/abs/2110.08981
8. **CAD-SIGNet [17]** — top-down 代表,主要 baseline。
   - https://arxiv.org/abs/2405.20094
9. **CAD-Diffuser [9]** — diffusion-based CAD 生成。
10. **SecAD-Net [20]** — self-supervised sketch-extrude 学习。
11. **CC3D [41, 42]** — 真实扫描数据集 + SHARP Challenge。
    - https://sharp3d.cc/
12. **CAD-Recode [11]** — ICCV 2025,后续工作,point cloud → CAD code。
13. **CAD-Assistant [12]** — ICCV 2025,tool-augmented VLM for CAD tasks。
14. **Picasso [36]** — WACV 2025,rendering self-supervision 的 parametric inference。
15. **CadVLM [38]** — ECCV 2024,VLM 跨模态 CAD sketch 生成。
16. **Aligning constraint generation with design intent [40]** — 2025 年新工作,constraint 与 design intent 对齐。
    - https://arxiv.org/abs/2504.13178

---

## 12. 总结

MiCADangelo 把 CAD reverse engineering 重新 cast 成三个 stage 的 cross-section pipeline,核心贡献:
1. 用 multi-plane cross-section 模仿人类设计师,捕捉 fine-grained detail(对应 ECD 大幅下降)
2. **第一次**把 sketch constraints 纳入 3D reverse engineering pipeline,让重建的 CAD 模型真正"可编辑"
3. Differentiable extrusion optimization 避免 autoregressive error accumulation
4. Image-based sketch parameterization 对 scan noise robust
5. 在 DeepCAD、Fusion360、CC3D 上都 SOTA,复杂模型和 real-world scan 上优势尤其明显

限制主要集中在 CAD operation 类型(只 extrude)和 primitive 类型(只 line/arc/circle),但这是整个领域的限制,MiCADangelo 在自己的 scope 内做到了极致。

参考链接(整理):
- Paper: https://arxiv.org/abs/2506.16548 (推测 arxiv ID,实际需查)
- 项目页: 应该在 SnT Luxembourg 团队 page
- FreeCAD(用来做 constraint evaluation): https://www.freecad.org
- SolidWorks: https://www.solidworks.com
- Onshape: https://www.onshape.com

如果想真正复现,关键卡点在 SketchGraphs 的 augment 策略(Algorithm 1/2/3)和 plane label 的预处理(Supp 7.1.1 把 extent_type 处理成 canonical plane 的那段)——这两个细节没在主文里,但决定了能不能训出 paper 报告的数字。
