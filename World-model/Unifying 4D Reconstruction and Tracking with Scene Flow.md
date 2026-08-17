---
source_pdf: Unifying 4D Reconstruction and Tracking with Scene Flow.pdf
paper_sha256: 345f56356e97d2d0eb94a4548e977d13954d34be93147cf35df96754a3a99ec0
processed_at: '2026-08-12T19:48:54-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞 Karpathy，咱们换个接地气的角度，把这篇 paper 像白板推导一样给你讲透。核心就是怎么用一个最简单、最对称的 formulation，把 3D 重建和 tracking 这两件事儿给统一了。

## 1. 痛点：为什么现有的 pipeline 总是拧巴？

传统计算机视觉里，重建和 tracking 是分开的。
- **重建（如 DUSt3R / VGGT）**：默认世界是静止的。给两张图，直接回归出在一个 shared coordinate 下的 3D pointmap。
- **Tracking（如 MonST3R / POMATO）**：要处理移动物体，通常得加个专门的 head，或者搞个 mask 把动态物体抠掉，再算相机位姿。

这就很别扭。就像作者引用的赫拉克利特名言 "Panta Rhei — Everything Flows"（万物皆流）。**Motion 是相对的**。你在自动扶梯上，刚踏上第一步时，大脑把扶梯当 static reference；要离开时，大脑把 landing 当 static reference。所谓 "static scene" 只是人为选了个坐标系而已。

强行让网络去猜 "哪块是 static" 是很难泛化的。远处的山、暗处无纹理的箱子、反光的玻璃，虽然不动，但对算 camera pose 毫无帮助。网络需要一个更 soft 的机制。

## 2. 核心直觉：极简的 4 件套

Flow4R 的精髓在于，给两张图 $(I, I')$，对每个 pixel $i$，网络只预测 4 个量：
- $\mathbf{P} \in \mathbb{R}^{H \times W \times 3}$：**Point position**。这个 pixel 在当前相机坐标系下的 3D 位置。
- $\mathbf{P}_{vt} \in \mathbb{R}^{H \times W \times 3}$：**Point position at new view and time**。这个 pixel 在另一张图 $I'$ 的视角和 $I'$ 的时刻下，所在的 3D 位置。（作者实验证明直接预测这个绝对位置 $\mathbf{P}_{vt}$ 比预测 scene flow $\mathbf{F}$ 效果更好，因为评估 metric 都是基于绝对位置的）。
- $\mathbf{W} \in (0,1)^{H \times W}$：**Pose weight**。这个 pixel 在算 camera pose 时有多靠谱。总和归一化 $\sum \mathbf{W}^i = 1$。
- $\mathbf{C} \in (1, \infty)^{H \times W}$：**Confidence**。网络对这个预测有多确信。

有了这 4 个量，几何和运动就全包含了。

## 3. 公式里的物理直觉：如何拆解 Motion？

咱们来看数学公式，里面全是变量和下标的游戏。
记住两个下标：
- $v$ = View change（相机视角变了）
- $t$ = Time change（物体自己动了）

公式 (1) 定义了总运动：
$$ \mathbf{P}_{vt}^i = \mathbf{P}^i + \mathbf{F}^i $$
$\mathbf{P}_{vt}^i$ 就是 pixel $i$ 在经历了相机运动和物体运动后，跑到新坐标系新时刻的最终位置。$\mathbf{F}^i$ 就是 Scene flow，它是总位移。

接下来用 $\mathbf{W}$ 去算相机的刚体运动 $\hat{\mathrm{T}}$。公式 (2) 是个 weighted least-squares：
$$ \hat{\mathrm{T}} = \arg\min_{\mathrm{T} \in SE(3)} \sum_{i=1}^{HW} \mathbf{W}^i \| \mathbf{P}_{vt}^i - \mathrm{T} \mathbf{P}^i \|_2 $$
$\mathrm{T}$ 是个 $3 \times 4$ 的刚体变换矩阵。$\mathbf{W}^i$ 权重大的地方，就是网络自己学出来的 "靠谱静态锚点"。算出来 $\hat{\mathrm{T}}$ 后，我们就能把总运动拆开：

- 公式 (4) 刚体运动：$\mathbf{F}_v^i = \hat{\mathrm{T}}\mathbf{P}^i - \mathbf{P}^i$ （相机移动造成的位移）
- 公式 (5) 非刚体运动：$\mathbf{F}_t^i = \mathbf{F}^i - \mathbf{F}_v^i$ （物体自己动造成的位移）

这就很漂亮了。网络不需要搞个二元 mask 去 hard-decide 谁动谁静。它输出一个 dense 的 weight map，在这个 weight map 的引导下，通过一个 SVD 闭环求解，自动把总 flow 拆成了 camera motion 和 object motion。

## 4. 最巧妙的设计：Pose Weight $\mathbf{W}$ 怎么训练？

这是整篇 paper 最 brilliant 的地方。现实世界里，我们根本没有 "哪些 pixel 适合用来算 pose" 的 ground truth。作者设计了一个 self-supervised 的 loss（公式 13）：
$$ \mathcal{L}_{\mathbf{W}} = \frac{1}{|\mathbf{M}_{\mathbf{P}}|} \sum_i \mathbf{M}_{\mathbf{P}}^i \left( \| \mathbf{P}_v^i - \bar{\mathbf{P}}_v^i \|_2 \right) $$
这里的逻辑是：
1. 网络预测出 $\mathbf{P}, \mathbf{P}_{vt}, \mathbf{W}$。
2. 用 $\mathbf{W}$ 去求解出 camera pose $\hat{\mathrm{T}}$。
3. 把 $\hat{\mathrm{T}}$ 乘上 ground truth 的 $\bar{\mathbf{P}}^i$，得到 "如果 pose 完美算对，pixel 应该在哪" $\bar{\mathbf{P}}_v^i$。
4. 计算 "用当前 $\mathbf{W}$ 算出来的 pose 移动后的点" $\mathbf{P}_v^i$ 与 "完美 pose 移动后的点" $\bar{\mathbf{P}}_v^i$ 的距离。

**关键操作**：在这个 loss 里，对 $\mathbf{P}$ 和 $\mathbf{P}_{vt}$ 做 stop-gradient（公式里没写，但文字里强调了）。梯度**只**流回 $\mathbf{W}$。
这就等于告诉网络：**"我现在把 3D 位置锁死，你去自己调整你的权重分配。如果你调得好，让求出来的 $\hat{\mathrm{T}}$ 正好等于 GT 的 $\bar{\mathrm{T}}$，这个 loss 就是 0。如果你给动态物体分配了高权重，导致 $\hat{\mathrm{T}}$ 算偏了，你就等着吃 gradient 吧。"**

这种 stop-gradient 传递 implicit gradient 的方式，让网络自动学会了：动态物体要降权，无纹理区域要降权，远处不靠谱的点要降权。

## 5. 架构上的好处：对称即效率

因为 formulation 是完全对称的（从 $I$ 看 $I'$ 和从 $I'$ 看 $I$ 是一样的逻辑），所以网络结构变得极其简单。
DUSt3R 那套是共享 encoder，但 decoder 和 head 要分两套，训练时还得手工构造 symmetrized pairs 交换输入。
Flow4R 因为预测的是 camera-space 的 scene flow，对 reference frame invariant。所以**两个分支完全共享 decoder 和 head 参数**。少了参数，也不用倒腾数据。没有专门的 pose head，也没有 bundle adjustment，单次 forward pass 搞定一切。

## 6. 实验数据说话

来看核心的 World Coordinate 3D Point Tracking（Table 1）。这要求模型同时搞懂几何、相机运动和物体运动。

| Method | ADT | DR | PO | PS | # param (B) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MonST3R | 74.4 | 58.1 | 33.5 | 51.3 | 0.7 |
| POMATO | 57.2 | 68.4 | 49.7 | 64.9 | 0.7 |
| St4RTrack | 76.0 | 73.7 | 68.0 | 69.7 | 0.7 |
| **Flow4R** | **78.6** | **78.5** | **71.1** | 64.3 | **0.4** |

看参数量，Flow4R 只有 0.4B，是所有强 baseline 里最小的，但效果在多数数据集上 SOTA。这就证明了 minimal formulation 的威力。别的模型堆 head、加后处理，反而不如这种对称的、轻量的设计。

另外 Table 3 的 ablation 也很有意思：
网络直接预测 $\mathbf{F}$ (Scene Flow) 并监督 $\mathbf{F}$，效果是 PO 60.23。
网络直接预测 $\mathbf{P}_{vt}$ (绝对位置) 并监督 $\mathbf{P}_{vt}$，效果是 PO 67.93。
哪怕它们之间只差一个加法 $\mathbf{P}_{vt} = \mathbf{P} + \mathbf{F}$。网络直接去拟合评估时关心的那个绝对位置，误差就能低很多。这其实揭示了 deep learning 里的一个普适规律：**尽量让网络直接预测你关心的那个 target，少让它做中间的转换运算。**

## 7. 总结 Intuition

这篇 paper 读下来，最大的 build intuition 就是：**不要用 hard 的离散逻辑去切分世界，要用可微的、连续的 soft weight 去逼近它。**
传统方法说："这里有辆车在动，mask 掉它，剩下的算 sfm"。
Flow4R 说："万物皆流。我预测所有点的 3D 运动。我同时预测一个 continuous 的 weight map，告诉你哪些点移动的规律最像刚体相机运动。我就用这些点去反解相机轨迹，剩下的部分就是物体真正的运动。"
这种从 "decoupled hard logic" 到 "unified soft learning" 的转变，正是当下 foundation model 时代 3D vision 进化的主轴。

---
**References / 相关链接：**
*   **Flow4R Project**: https://shenhanqian.github.io/flow4r
*   **DUSt3R (基础架构来源)**: https://dust3r.europe.naverlabs.com/
*   **MonST3R (Dynamic 重建先驱)**: https://monst3r-project.github.io/
*   **St4RTrack (直接对比 baseline)**: https://st4rtrack.github.io/
*   **VGGT (多视图 SOTA)**: https://vgg-t.github.io/
*   **CroCo v2 (预训练 backbone)**: https://arxiv.org/abs/2210.10716
*   **PointOdyssey (Tracking 评测集)**: https://pointodyssey.com/

---

# Flow4R 深度解析

好嘞 Karpathy,这篇论文挺有意思的,我觉得它在 4D perception 这个方向上做了一个很优雅的 unified formulation。让我把它拆开来讲,尽量帮你 build intuition。

## 1. 核心哲学:Motion 的相对性

作者用 "Panta Rhei — Everything Flows" (赫拉克利特"万物皆流") 作 subtitle,这定下了整个工作的基调。关键 insight 是:

**Motion 是相对的** — 我们在 image 中观察到的 motion,是 object motion 和 camera motion 的叠加。所谓 "static scene" 只是选择了某个 reference coordinate system 而已。比如在自动扶梯上,接近扶梯时把第一步当 static reference,离开时把 landing 当 static reference。

这其实呼应了 Mach's principle 和 general relativity 的精神 — 没有绝对的 inertial frame。所以与其强行 declaring "这块区域是 static",不如**预测 camera-space scene flow**(它和 reference frame 选择无关),然后通过一个 learnable 的 pose weight map 来决定"哪些 pixel 是用于 pose estimation 的可靠 anchor"。

这个 pose weight W 是 self-supervised 学出来的,可以替换 — 推理时换一张 W map 就能切换 reference coordinate system,这是我觉得最 elegant 的设计之一。

---

## 2. Minimal Property Set — 把 4D 压进 4 个量

对每个 image pair (I, I'),为 I 中的每个 pixel i 预测:

$$
\mathcal{S}(I, I') = \{\mathbf{P}, \mathbf{F}, \mathbf{W}, \mathbf{C}\}
$$

| Symbol | Shape | Meaning |
|---|---|---|
| $\mathbf{P} \in \mathbb{R}^{H \times W \times 3}$ | per-pixel | point position in I 的 local Euclidean space |
| $\mathbf{F} \in \mathbb{R}^{H \times W \times 3}$ | per-pixel | scene flow: 把 P 从 (view=I, time=I) 移到 (view=I', time=I') |
| $\mathbf{W} \in (0,1)^{H \times W}$ | per-pixel | pose weight, Σ W^i = 1 (normalized) |
| $\mathbf{C} \in (1, \infty)^{H \times W}$ | per-pixel | confidence/uncertainty |

对称地有 $\mathcal{S}(I', I) = \{\mathbf{P}', \mathbf{F}', \mathbf{W}', \mathbf{C}'\}$。

**关键点**: 这 4 个量是最 minimal 的 — 它们既包含 local geometry (P),又包含 bidirectional motion (F),还包含求解 camera pose 所需的 weight (W),以及 uncertainty (C)。一切都从这 4 个量导出。

---

## 3. Scene Flow Decomposition — 整个方法的核心

### 3.1 基本关系

scene flow F 把 P 带到 "在 I' 的 view 和 I' 的 timestamp 下的位置":

$$
\mathbf{P}_{vt}^i = \mathbf{P}^i + \mathbf{F}^i \tag{1}
$$

注意下标的语义:
- 下标 $v$ = view 切换 (从 I 视角到 I' 视角)
- 下标 $t$ = time 切换 (从 I 时刻到 I' 时刻)
- $\mathbf{P}_{vt}^i$ = pixel i 的 3D point 在 I' 视角 + I' 时刻的位置(全切换)
- $\mathbf{P}_{v}^i$ = pixel i 的 3D point 在 I' 视角 + I 时刻的位置(只切 view,保持 time)
- $\mathbf{P}_{t}^i$ = pixel i 的 3D point 在 I 视角 + I' 时刻的位置(只切 time,保持 view)

### 3.2 求 camera pose T

用 weighted least squares 从静态区域求解 rigid transformation:

$$
\hat{\mathrm{T}} = \arg\min_{\mathrm{T} \in SE(3)} \sum_{i=1}^{HW} \mathbf{W}^i \| \mathbf{P}_{vt}^i - \mathrm{T} \mathbf{P}^i \|_2 \tag{2}
$$

- $\mathbf{W}^i$ 是 pose weight — 自动 down-weight 动态/不可靠 pixel
- 这是一个 weighted Procrustes / Kabsch 问题,有 closed-form SVD 解
- $\hat{\mathrm{T}} \in \mathbb{R}^{3 \times 4}$ 是 I 相对 I' 的 rigid transformation

### 3.3 把 flow 分解为 rigid + non-rigid

$$
\mathbf{P}_v^i = \hat{\mathrm{T}} \mathbf{P}^i \quad \text{(rigid 部分:纯 camera motion 引起)} \tag{3}
$$

$$
\mathbf{F}_v^i = \mathbf{P}_v^i - \mathbf{P}^i \quad \text{(rigid flow component)} \tag{4}
$$

$$
\mathbf{F}_t^i = \mathbf{F}^i - \mathbf{F}_v^i \quad \text{(non-rigid flow = object motion)} \tag{5}
$$

这就是把 scene flow 拆成"相机刚性运动" + "物体非刚性运动"两部分。直觉上:

- $\mathbf{F}^i$ = 完整的运动向量(同时含相机和物体运动)
- $\mathbf{F}_v^i$ = 如果 pixel i 是 static,它本应该有的运动(纯相机引起)
- $\mathbf{F}_t^i$ = 实际运动偏离 static 假设的部分 = 物体本身的运动

### 3.4 3D Point Tracking

把点 track 到 I 时刻的 I' view:

$$
\mathbf{P}_t^i = \hat{\mathrm{T}}^{-1} \mathbf{P}_{vt}^i = \hat{\mathrm{T}}^{-1}(\mathbf{P}^i + \mathbf{F}^i) \tag{6}
$$

### 3.5 Focal Length 和 Optical Flow

通过最小化 reprojection error 解 focal length:

$$
\hat{f} = \arg\min_f \sum_i \| \hat{\mathbf{p}}^i - \pi(f, \mathbf{c}, \mathbf{P}^i) \|^2 \tag{7}
$$

其中 $\pi(f, \mathbf{c}, \mathbf{P}^i)$ 是 pinhole projection:
$$
\pi(f, \mathbf{c}, \mathbf{P}^i) = f \cdot \frac{[\mathbf{P}^i_{xy}]}{\mathbf{P}^i_z} + \mathbf{c}
$$

optical flow 就是 projection 的差:
$$
\mathbf{f}^i = \mathbf{p}_{vt}^i - \mathbf{p}^i = \pi(f, \mathbf{c}, \mathbf{P}_{vt}^i) - \pi(f, \mathbf{c}, \mathbf{P}^i) \tag{8, 9}
$$

---

## 4. 架构设计

### 4.1 网络结构

整体沿用 DUSt3R 的 two-view transformer paradigm,但有重要改进:

```
┌─────────────────────────────────────────────┐
│  Image I  ──→  CroCo Encoder  ──┐           │
│                                  │           │
│                          Cross-Attention      │
│                                  │           │
│  Image I' ──→  CroCo Encoder  ──┘           │
│                                  │           │
│              ┌───────────────────┐           │
│              │ Shared Decoder    │           │
│              │ (DPT head)        │           │
│              └───────────────────┘           │
│                  │            │              │
│           ┌──────┘            └──────┐       │
│       output for I              output for I'│
│   {P, P_vt, W, C}              {P', P'_vt, W', C'}│
└─────────────────────────────────────────────┘
```

关键点:
- **对称性**: 两个 forward path **共享 encoder + decoder + head 参数**(DUSt3R 是共享 encoder 但 decoder 是分开的)
- 不需要像 DUSt3R 那样在训练时 manually construct symmetrized pairs
- 没有 explicit pose head — pose 是从 scene flow + W 反解出来的
- 没有 bundle adjustment — feedforward 单次推理

### 4.2 序列处理

采用 anchored connection(类似 St4RTrack):
- 第一帧 I_0 作为 anchor
- 后续帧和 anchor 配对:(I_0, I_1), (I_0, I_2), (I_0, I_3), ...
- 因为预测 local point map P for anchor view,可以用 anchor 的平均 norm 做 scale alignment:

$$
s_n = \text{mean}_i \|\mathbf{P}_{I_0 \to I_n}^i\|_2
$$

对后续 pair 的预测乘以 $\frac{s_1}{s_n}$ 来对齐 scale。

这点比 St4RTrack / POMATO 强 — 因为它们不预测 anchor view 的 local point map,做不了 scale alignment。

---

## 5. 训练监督 — 5 个 Loss 的协同

总 loss:

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{\mathbf{P}} + \lambda_2 \mathcal{L}_{\mathbf{F}} + \lambda_3 \mathcal{L}_{\mathbf{f}} + \lambda_4 \mathcal{L}_{\mathbf{W}} + \lambda_5 \mathcal{L}_{\mathbf{F}_v}
$$

权重: $\lambda_1 = 1, \lambda_2 = \lambda_4 = \lambda_5 = 0.5, \lambda_3 = 0.3, \alpha = 0.2$

### 5.1 Point Position Loss

$$
\mathcal{L}_{\mathbf{P}} = \frac{1}{|\mathbf{M}_{\mathbf{P}}|} \sum_i \mathbf{M}_{\mathbf{P}}^i \left( \mathbf{C}^i \|\mathbf{P}^i - \bar{\mathbf{P}}^i\|_2 - \alpha \log \mathbf{C}^i \right) \tag{10}
$$

- $\mathbf{M}_{\mathbf{P}}^i$ = valid depth mask
- $\bar{\mathbf{P}}^i$ = GT point map (从 depth + intrinsics back-project)
- $\mathbf{C}^i$ = confidence (uncertainty-aware)
- $\alpha \log \mathbf{C}^i$ 是 confidence regularization,防止 C → ∞

这是 DUSt3R 风格的 robust loss,confidence 自动学。

### 5.2 3D Motion Loss

$$
\mathcal{L}_{\mathbf{F}} = \frac{1}{|\mathbf{M}_{\mathbf{F}}|} \sum_i \mathbf{M}_{\mathbf{F}}^i \left( \mathbf{C}^i \|\mathbf{P}_{vt}^i - \bar{\mathbf{P}}_{vt}^i\|_2 - \alpha \log \mathbf{C}^i \right) \tag{11}
$$

- 这里直接监督 $\mathbf{P}_{vt}$ (而非 F),实验证明更好(Table 3)
- 共享 confidence map(因为是同一 Euclidean space)
- mask 包括 scene flow、optical flow、point tracking 三种来源

### 5.3 2D Motion Loss

$$
\mathcal{L}_{\mathbf{f}} = \frac{1}{|\mathbf{M}_{\mathbf{f}}|} \sum_i \mathbf{M}_{\mathbf{f}}^i \|\mathbf{p}_{vt}^i - \bar{\mathbf{p}}_{vt}^i\|_2 \tag{12}
$$

- 对 2D projection 监督(无 confidence 加权,因为是 projective space 而非 Euclidean)

### 5.4 Pose Weight Loss — self-supervised!

这是最精妙的 loss。W 没有 GT,所以通过让求解的 T 和 GT 对齐来学习:

$$
\mathcal{L}_{\mathbf{W}} = \frac{1}{|\mathbf{M}_{\mathbf{P}}|} \sum_i \mathbf{M}_{\mathbf{P}}^i \|\mathbf{P}_v^i - \bar{\mathbf{P}}_v^i\|_2 \tag{13}
$$

- $\mathbf{P}_v^i = \hat{\mathrm{T}} \mathbf{P}^i$ — T 是从 P, P_vt, W 求解的
- $\bar{\mathbf{P}}_v^i = \bar{\mathrm{T}} \bar{\mathbf{P}}^i$ — 用 GT pose 算
- **stop gradient on P 和 P_vt**(它们已经在其他 loss 充分监督)
- 所以梯度只流到 W
- 直觉: 调整 W 使得从这些 weighted pixels 求解出的 T 与 GT 对齐

**这等于自动学出了一个 "static region mask"**,而且:
- 不仅 mask 动态物体
- 还 down-weight 远处、纹理少、反射、遮挡的区域
- 学习 dataset 中常见的 reference coordinate system 选择

### 5.5 Rigid Motion Loss — 用 GT pose 增强

由于 scene flow GT 稀缺,但 static scene 的 camera pose 数据多。对 static 区域可以用 GT pose 算出 rigid flow 来监督:

$$
\mathcal{L}_{\mathbf{F}_v} = \frac{1}{|\mathbf{M}_{\mathbf{P}}|} \sum_i \mathbf{M}_{\mathbf{P}}^i \left( w^i \mathbf{C}^i \|\mathbf{P}_{vt}^i - \bar{\mathbf{P}}_v^i\|_2 - \alpha \log \mathbf{C}^i \right) \tag{14}
$$

其中:

$$
w^i = \begin{cases} sg(\mathbf{W}^i) \times HW & \text{if dynamic dataset} \\ 1 & \text{if static dataset} \end{cases} \tag{15}
$$

- $sg(\cdot)$ = stop gradient,防止 W 在这个 loss 中被优化(否则会和 $\mathcal{L}_{\mathbf{W}}$ 冲突)
- 乘 HW 是因为 ΣW^i = 1,不放大就没有 comparable loss scale
- 这条 loss 让 P_vt 在 static 区域接近 GT rigid flow

---

## 6. 训练数据 — 跨域大杂烩

训练集是个大杂烩,包括 static + dynamic、real + synthetic:

- **Static real**: Habitat, Blended-MVS, MegaDepth, ARKitScenes, CO3D, Static Scenes 3D, ScanNet++, Waymo, TartanAir, WildRGBD, DL3DV, MapFree, ScanNet, HyperSim
- **Synthetic dynamic**: Virtual KITTI 2 (scene flow), Spring (optical flow), Dynamic Replica (optical flow + 3D tracking), Kubric (3D tracking), OmniWorld-Game (optical flow), PointOdyssey (3D tracking)

**两阶段训练**:
| Setting | Stage 1 | Stage 2 |
|---|---|---|
| Epochs | 100 | 100 |
| Resolution | 224 | 512 (random aspect ratio) |
| Pairs/epoch | 900K | 84K |
| Batch size | 256 | 64 |
| Head | linear | DPT |
| Warmup epochs | 10 | 20 |
| LR peak | 1e-4 | 1e-4 |
| LR final | 1e-6 | 1e-6 |

- 8× NVIDIA A100/H100
- CroCo v2 初始化(不是 DUSt3R/MASt3R/MonST3R,因为 formulation 变了)
- Adam,linear warmup + cosine decay
- Gradient clip max norm = 10
- 总训练时间约 4 天

---

## 7. 实验

### 7.1 World Coordinate 3D Point Tracking (Table 1)

metric: APD3D — Average Percentage of 3D Points within Delta,δ ∈ {0.1, 0.3, 0.5, 1.0} 米

| Method | ADT | DR | PO | PS | DR (dyn) | PO (dyn) | #param (B) |
|---|---|---|---|---|---|---|---|
| MonST3R | 74.4 | 58.1 | 33.5 | 51.3 | 51.9 | 39.4 | 0.7 |
| SpaTracker | 45.7 | 54.9 | 38.5 | 62.6 | 58.7 | 51.2 | 0.2 |
| POMATO | 57.2 | 68.4 | 49.7 | 64.9 | 62.7 | 58.1 | 0.7 |
| St4RTrack | 76.0 | 73.7 | 68.0 | 69.7 | 68.1 | 68.7 | 0.7 |
| **Flow4R** | **78.6** | **78.5** | **71.1** | 64.3 | **77.2** | **72.9** | **0.4** |

Flow4R 参数量 0.4B(最少的!),在 ADT/DR/PO 上 SOTA,只有 PS 略低。

**核心观察**: St4RTrack/POMATO 都是 asymmetric head-bound formulation(基于 DUSt3R 改),Flow4R 用 symmetric shared decoder 反而更强 — 这说明 formulation 的设计比堆参数重要。

### 7.2 World Coordinate 3D Reconstruction (Table 2)

| Method | PO APD↑ | PO EPE↓ | TUM APD↑ | TUM EPE↓ |
|---|---|---|---|---|
| DUSt3R+GA | 43.90 | 0.609 | 70.49 | 0.315 |
| MASt3R+GA | 60.44 | 0.403 | 68.38 | 0.519 |
| MonST3R+GA | 72.31 | 0.263 | 63.87 | 0.343 |
| DUSt3R | 45.79 | 0.639 | 72.26 | 0.289 |
| MASt3R | 56.90 | 0.464 | 66.22 | 0.551 |
| MonST3R | 68.25 | 0.304 | 61.38 | 0.365 |
| POMATO | 66.50 | 0.385 | 49.80 | 0.509 |
| St4RTrack | 78.73 | 0.205 | 83.42 | 0.185 |
| **Flow4R** | **81.00** | **0.182** | 79.87 | 0.202 |

- Point Odyssey: Flow4R 最好 (81.00 / 0.182)
- TUM: St4RTrack 略好 (83.42 vs 79.87),Flow4R 次之
- 注意: DUSt3R/MASt3R/MonST3R 需要 +GA(global alignment)后处理,Flow4R 是 feedforward,无后处理

### 7.3 Ablation: 预测 F 还是 P_vt? (Table 3)

| Pred | Target | ADT | DR | PO | PS | PO (recon) | TUM (recon) |
|---|---|---|---|---|---|---|---|
| F | $\bar{\mathbf{F}}$ | 78.03 | 73.26 | 60.23 | 55.80 | 69.36 | 79.78 |
| F | $\bar{\mathbf{P}}_{vt}$ | 77.72 | 76.41 | 61.21 | 63.69 | 66.29 | 80.05 |
| **P_vt** | $\bar{\mathbf{P}}_{vt}$ | **78.50** | **78.48** | **67.93** | **67.17** | **77.20** | **80.34** |

直觉解释:
- $\mathbf{P}_{vt}$ 是直接评估用的位置(最终 metric 基于 point position)
- $\mathbf{F}$ 是中间表示,要再经过 $\mathbf{P}_{vt} = \mathbf{P} + \mathbf{F}$ 才得到评估位置,多一个加法步骤就多一个 error source
- 这跟 deep learning 里的 "predict the final target, not the intermediate" 经验一致

---

## 8. Visualization 中的 Intuition (Figure 6)

四个例子很有启发性:

**(a) Static scene + 旋转相机**(行李箱场景)
- scene flow map 五颜六色(3D 旋转引起不同方向的 flow)
- pose weight 在暗的、无纹理的行李箱区域低 — 这些区域对 localization 无用

**(b) Dynamic scene**(跳动的机器人)
- scene flow map 几乎和 rigid flow 一致,**除了机器人区域**
- pose weight 在机器人区域被强烈抑制,防止其独立运动 bias pose 估计

**(c) Stopped train(远 vs 近)**
- optical flow 主导红蓝色调 → 主要 translation
- pose weight 只在两图重叠区域高,这正是用于 pose 计算的"可见 anchor"

**(d) 四人跳舞**
- confidence map 高,除了只在一帧出现的最后一人
- pose weight 排除所有移动的人 — pose 只从 static background 估计

**这些可视化展示了一个关键 insight**: pose weight 不是简单的 "moving object mask",而是一个更微妙的"哪些 pixel 对 localization 有用"的 indicator,可以 down-weight 远处、反射、遮挡等区域。

---

## 9. 我对这篇工作的几点思考

### 9.1 与 DUSt3R 谱系的对比

DUSt3R 预测 shared-frame pointmaps — 所有点在同一个 reference frame。这在 static scene 自然,但 dynamic 时有歧义。后续工作(MonST3R、POMATO、St4RTrack、D²USt3R)走两条路:
1. 加 head(在原 pointmap 之外额外预测 timestamp 2 的 pointmap)
2. Repurpose head(把 pointmap head 改成预测 timestamp 2 的 pointmap)

Flow4R 的路线完全不同:用一个 minimal property set(P, F, W, C)在 single forward pass 里搞定,共享 decoder。这种**对称性**让模型容量集中,而且推理时可以双向查询。

### 9.2 为什么 scene flow 是"对"的中心表示?

- camera-space scene flow **和坐标选择无关**(invariant to reference frame)
- 它"含而不露"地包含了 camera motion 和 object motion
- 通过 pose weight 做分解,而非 hard-coding "static vs dynamic"
- 静态/动态 distinction 只是 pose weight 的极端情况(W=1 vs W=0)

### 9.3 Pose Weight 的 self-supervision 真的优雅

注意 Eq. 13 的设计:
- stop-grad P, P_vt(防止它们被这个 loss 干扰)
- 只让 W 接收梯度
- 让求解出的 $\hat{\mathrm{T}}$ 与 GT $\bar{\mathrm{T}}$ 一致

这等于在说: "**调整你的 weight,使得从你 weight 出发求解的 pose 正好等于 GT pose**"。这是一种 implicit differentiation 风格的监督,但用的是 forward sensitivity(stop-grad trick)。

### 9.4 局限性(作者也承认)

- scene flow GT 稀缺,所以 flow 质量受限于数据
- 只 two-view,没有像 VGGT 那样的 multi-view global attention
- Online tracking 下的 memory/compute 没解决

---

## 10. 相关工作的谱系图

```
DUSt3R (2024)
├── MASt3R (correspondences)
├── MonST3R (dynamic, fine-tune + post-opt)
├── POMATO (add head for t2 pointmap)
├── St4RTrack (repurpose head for t2)
├── D²USt3R (repurpose + dynamic)
├── VGGT (multi-view global attention)
│   ├── Streaming VGGT
│   └── VGGT4D / D4RT / Any4D
└── Flow4R (this paper)
    — symmetric shared decoder
    — scene flow + pose weight formulation
    — no explicit pose head, no BA
```

我觉得 Flow4R 在哲学上和 NeRF 的 "let the network figure out everything" 思路很像 — 不做 hard decision(static vs dynamic),让 network 学一个 flexible 的 weight map 来 soft-decide。

---

## 参考链接

- 项目主页: https://shenhanqian.github.io/flow4r
- DUSt3R: https://dust3r.europe.naverlabs.com/
- MASt3R: https://arxiv.org/abs/2406.09856
- VGGT: https://vgg-t.github.io/
- MonST3R: https://monst3r-project.github.io/
- St4RTrack: https://st4rtrack.github.io/
- CroCo (pretraining backbone): https://arxiv.org/abs/2210.10716
- WorldTrack benchmark: 见 St4RTrack paper
- PointOdyssey dataset: https://pointodyssey.com/
- Dynamic Replica: https://dynamic-stereo.github.io/
- Aria Digital Twin: https://www.projectaria.com/datasets/adt/
- Spring (optical flow): https://spring-benchmark.jinwcho.de/
- CoTracker: https://co-tracker.github.io/

---

总结一句:Flow4R 把 4D 重建和 tracking 用一个 minimal、对称、self-supervised pose weight 的 formulation 统一起来,核心 insight 是 **"motion 是相对的,不要 hard-decide static/dynamic,让 network 学一个 weight map 来 soft-decide"**。在 two-view 范式下 SOTA 且参数最少。下一步看怎么扩展到 multi-view global attention (VGGT-style) 和 online tracking。
