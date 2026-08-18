---
source_pdf: Continuous Cost Aggregation for Dual-Pixel Disparity Extraction.pdf
paper_sha256: 28fba7905cc89cf731f75521fa97d6cae2b3da2285ea113361224d365d60105e
processed_at: '2026-08-18T04:10:51-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CCA

Andrej，我换个讲法，尽量把数学符号扔掉，用直觉和类比来说。

---

## 一、Dual-Pixel 到底是个什么东西

你手机摄像头里每个像素，其实物理上被劈成了两半——左半边和右半边。左半边只接收镜头左半边的光，右半边只接收镜头右半边的光。所以一次拍照，你其实拿到了两张图：一张"从左边看"，一张"从右边看"。

这就是个 mini stereo camera。两只眼睛靠得特别近——近到 baseline 只有 pupil diameter 那么点。所以 disparity 范围特别小。普通 stereo 是几十甚至上百 pixel 的 disparity，DP 可能就几个 pixel，phone 上甚至不到 1 个 pixel。

一句话：**DP 就是 baseline 极小、disparity 极小的 stereo，且两个 view 的 blur 形状还不一样。**

参考：Wadhwa et al., SIGGRAPH 2018 https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/

---

## 二、为什么传统 SGM 在这里崩了

SGM 的工作方式是：对每个像素，对每个 candidate disparity 算个 cost，堆成一个 3D 的 cost volume $(x, y, d)$，然后沿路径做 aggregation，最后 winner-take-all 选一个 integer disparity，再用 parabola fitting 抠出 sub-pixel。

这套流程在普通 stereo 上很好用，但在 DP 上有三个问题：

**问题 1：disparity 太小，integer 精度根本不够用。**
如果真实 disparity 是 0.3，你只能在 0 和 1 里二选一，然后 parabola fitting 补救。但 cost curve 在这么窄的范围内可能根本没有干净的 U 形，因为——

**问题 2：两个 view 的 PSF 不一样。**
普通 stereo 假设 $I_R(x) = I_L(x-d)$，是 pure shift。但 DP 两个 view 的 blur kernel 形状不同（一个偏左一个偏右），所以即使在真实 disparity 处，两个 view 的 patch 也不是简单 shift 关系，matching cost 不干净。

**问题 3：cost volume 需要存 $WHD$，对 phone 这种边缘设备吃不消。**

SGM 是"先在 integer grid 上选最优，再 sub-pixel refine"的两步式。在 disparity 大的场景下，quantization error 占比小，refine 够用。但在 DP 这种 disparity 本身就 sub-pixel 量级的场景，第一步的 quantization 就已经把信息丢了，refine 救不回来。

---

## 三、CCA 的核心 idea，用人话说

**别在 integer grid 上做决定了，从一开始就住在 continuous 空间里。**

具体怎么做？

### Step 1：每个像素画一条抛物线

对每个像素 $p$，你算完 discrete cost 之后，找到 cost 最低的那个 integer disparity $d^0$。然后取 $d^0$ 周围三个点 $(d^0-1, d^0, d^0+1)$ 的 cost 值，拟合一条 parabola。

这条 parabola 的最低点，就是你这个像素的 local best guess——而且是 sub-pixel 的，不用 refine。

parabola 有三个系数，但只有两个有用：
- **开口的宽窄**（curvature，对应 $\alpha$）：窄=自信，宽=不自信
- **最低点的位置**（对应 $-\beta/(2\alpha)$）：你猜的 disparity

### Step 2：沿路径传话

现在你有一个 2D 的 parabola field——每个像素一条 parabola。但每个像素的 parabola 都是 local 的，没考虑邻居。

SGM 的做法是沿路径传播 cost，让邻居互相影响。CCA 也传播，但传播的不是整个 cost volume，而是 parabola 的两个系数。

怎么传播？对每个像素，把"自己的 local parabola"加上"被邻居拉一把的 penalty"：

$$L_p(d) = \underbrace{C_p(d)}_{\text{自己的 parabola}} + \underbrace{P \cdot (d - m_{p-1})^2}_{\text{被邻居拉的 penalty}}$$

这里 $m_{p-1}$ 是前一个像素已经聚合过的 parabola 的最低点。意思是：**"我倾向于让我的 disparity 离前一个像素的 disparity 近一点。"**

**关键的数学魔法**：自己的 parabola 是 quadratic，penalty 也是 quadratic，加起来还是 parabola！所以聚合后的 cost 仍然是 parabola，仍然有 closed-form 的最低点。

这就意味着：你只需要更新两个 scalar（curvature 和 slope），就能完成一次 propagation。不需要存 3D cost volume，不需要 DP 递归，就是简单的标量加法。

### Step 3：多路径求和

SGM 沿 8 个方向（或 4 个）做 path propagation。CCA 一样。但 CCA 的好处是：parabola 加 parabola 还是 parabola。所以你把 8 条路径的 parabola 系数加起来，得到一个总 parabola，它的最低点就是最终 disparity：

$$\text{disparity}(p) = -\frac{\sum_r \mathrm{B}_p^r}{2 \sum_r \mathrm{A}_p^r}$$

完事。没有 WTA，没有 parabola fitting 后处理，disparity 直接从聚合结果里出来。

---

## 四、为什么 penalty 要设计成 $P \cdot \mathrm{A}_{p-1} \cdot \exp(\ldots)$

这个公式三部分，每部分有直觉：

1. **$P$**：整体 smoothness 强度，用户调参用
2. **$\mathrm{A}_{p-1}$**：前一个像素的 curvature（confidence）。**只有自信的像素才有资格拉别人。** 如果前一个像素自己都不确定（parabola 很平），它就不该影响别人。这相当于 "confidence-weighted propagation"。
3. **$\exp(-(I_p - I_{p-1})^2/\sigma^2)$**：image edge detector。如果两个像素的图像值差异大（可能是物体边界），penalty 衰减，允许 disparity 跳变。这跟 SGM 里 $P_1, P_2$ 的思路一样，只是形式变成了 exponential decay。

---

## 五、多尺度怎么融合

粗尺度算完，怎么帮细尺度？

朴素做法：把粗尺度的 disparity 上采样，当成细尺度的初始 guess，只在其附近搜。问题是：粗尺度错了就全错了，没救。

CCA 的做法更软：把粗尺度的 parabola 系数（A, B）上采样，**加到**细尺度的 local parabola 系数上：

$$\alpha^{\text{with prior}} = \alpha^{\text{local}} + w \cdot \mathrm{A}^{\text{prior}}$$

这是个 additive prior，不是 hard constraint。结果：
- 如果细尺度 local cost 很自信（$\alpha^{\text{local}}$ 大），prior 影响小，按 local 走
- 如果细尺度 local cost 不自信（blur 区域、textureless 区域），prior 主导
- 如果细尺度有强证据反对 prior，细尺度可以纠正粗尺度的错误

这个"加法"而不是"约束"的设计，让多尺度变成了"建议"而非"命令"。粗尺度给个方向，细尺度有证据可以反驳。

**为什么 B 要乘 scale factor F 而 A 不用？** 因为 minimizer $m = -B/(2A)$。图像放大 F 倍，disparity 也放大 F 倍。所以 m 要乘 F，即 B 要乘 F，A 不变。这是个细节但很重要——如果你直接把 A 和 B 都上采样，prior 的 disparity 会偏。

---

## 六、Confidence score 的 trick

简单 curvature 不够，因为 parabola 只能表示一个 minimum。如果 cost curve 其实有两个差不多低的 minimum（repetitive pattern），parabola 拟合可能选错。

CCA 的处理：看 second-lowest cost。如果 second-lowest 离 minimum 不远（两个 cost 接近），说明这个像素不可信，把 parabola 整体"压扁"——同时缩小 $\alpha$ 和 $\beta$，但保持 minimizer 不变。

$$S_{\text{confidence}} = \max\left(\min\left(\frac{1-q}{1-T_q}, 1\right), \epsilon\right)^2$$

其中 $q = C_{\min}/C_{\text{second-min}}$。$q$ 接近 1（两个 cost 接近）→ $S$ 接近 0 → parabola 被压扁。

**为什么平方？** 因为 minimizer 是 $-\beta/(2\alpha)$。如果你把 $\alpha$ 和 $\beta$ 同时乘 $S$，minimizer 不变，但 curvature 变小（变扁）。这样 confidence 降低了，但 minimum 位置不动。很聪明。

---

## 七、和 SGM 的根本区别，一张表说清

| 维度 | SGM | CCA |
|---|---|---|
| Cost 表示 | discrete cost per integer disparity | parabola 三个系数 |
| Sub-pixel | 最后 parabola fitting 补救 | 一开始就在 continuous 空间 |
| Penalty | L1 ($P_1, P_2$ piecewise) | L2 ($(d-m)^2$) |
| Aggregation 空间 | 3D $(x,y,d)$ | 2D $(x,y)$，只传 2 个 scalar |
| 空间复杂度 | $O(WHD)$ | $O(WH)$ |
| 时间复杂度 | $O(WHDR)$ | $O(WHD + WHR)$ |
| Edge handling | $P_1, P_2$ + image gradient | exponential factor in penalty |
| Confidence | 要单独算 | 内建在 curvature 里 |

**L2 vs L1 是核心 trade-off**：L2 保持 closed-form，但 over-smooth sharp edges；L1 edge-preserving，但破坏 closed-form，必须 discrete DP。在 DP 这种小 disparity 场景，L2 的 over-smoothing 不严重，所以 trade-off 划算。在 Middlebury 大 disparity 场景，CCA 在 sharp edges 上略差于 SGM，正是 L2 的代价。

---

## 八、整个流程的 intuition，一句话版

> **每个像素先自己画一条抛物线猜 disparity，然后沿路径把抛物线"加"起来，加到最后还是抛物线，最低点就是答案。**

"加"这个动作之所以 work，是因为：
1. Parabola + parabola = parabola
2. Parabola + quadratic penalty = parabola  
3. Parabola × scalar = parabola
4. 这三条性质让整个 pipeline 从头到尾都是 parabola，最后直接拿最低点，没有任何离散搜索

这就是为什么 CCA 能在 2D 空间完成 SGM 需要 3D 空间做的事。

---

## 九、更宏观的 intuition

这篇 paper 给我最大的启发：**当你发现一个问题需要在一个高维离散空间里搜索（比如 SGM 的 $(x,y,d)$ cost volume），看看能不能把每个点的 cost 用一个对加法封闭的低维参数化表示（比如 parabola），这样聚合就退化成参数传播，复杂度降一个量级。**

这个 pattern 在其他地方也见过：
- **Optical flow** 用 polynomial trajectory 表示，多帧聚合变参数聚合
- **Lucas-Kanade** 用 inverse compositional 把 alignment 变成参数更新
- **Kalman filter** 本质也是"用 Gaussian 表示 belief，加法下封闭，所以只传参数"

CCA 是这个 pattern 在 stereo cost aggregation 上的一个漂亮实例。Parabola 是 "just expressive enough" 的 sweet spot——比 constant expressive（能表示 sub-pixel minimum），比 higher-order polynomial 简单（保持 convex 和 closed-form）。

---

## 十、局限和未来方向

1. **Parabola 只能表示一个 minimum**。repetitive pattern 上会选错。未来可以用 Gaussian Mixture 或 spline，但失去 closed-form。
2. **L2 penalty over-smooth sharp edges**。在大 disparity 场景不如 SGM。Paper 在 Middlebury 上确实略差。
3. **Cost function 还是 hand-crafted (SAD/NCC)**。如果用 learned feature，parabola 拟合会更干净。
4. **Multi-scale prior 是 additive 的**。更复杂的 prior（比如 learned prior network）可能更好，但破坏 closed-form。

---

## References

- 原文 https://arxiv.org/abs/2403.18460  
- SGM (Hirschmuller 2008) https://ieeexplore.ieee.org/document/4359315  
- Wadhwa SIGGRAPH 2018 https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/  
- DPdisp (Punnappurath ICCP 2020) https://sites.google.com/view/dualpixeldepth/home  
- Bilateral Solver (Barron & Poole ECCV 2016) https://arxiv.org/abs/1601.02588

---

最后一句人话总结：**CCA 就是"用抛物线代替 cost volume，用加法代替 DP 搜索"，靠的是 quadratic function 在加法下的封闭性，把 SGM 从 3D 拍扁到 2D，顺带把 sub-pixel 精度内置进去了。**

---

# Continuous Cost Aggregation for Dual-Pixel Disparity Extraction 深度讲解

Andrej 你好，这篇 paper 我读了之后觉得特别 elegant。它的核心 trick 在数学上非常简洁——利用 parabola 在加法下的封闭性，把 SGM 的 3D cost volume aggregation 压缩成 2D parabola coefficient propagation，同时天然拿到 sub-pixel disparity。下面我从 motivation、数学、intuition、实验四个层面展开。

---

## 1. 背景：为什么 DP sensors 让传统 stereo 算法失效

### 1.1 Dual-Pixel 是什么

DP sensor 原本是给 autofocus 用的：每个 pixel 在硬件上被劈成左右两半 (left/right sub-pixels)，分别收集来自 opposite sides of the aperture 的光线。这等价于两个 sub-aperture views，baseline 极小（≈ pupil diameter × pixel pitch），且天然 rectified，所以看起来像一个 tiny-baseline stereo pair。

参考：Wadhwa et al., SIGGRAPH 2018 - "Synthetic depth-of-field with a single-camera mobile phone" 
https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/

### 1.2 与传统 stereo 的两个本质区别

**Difference 1：Different PSF per view.** 

传统 stereo 假设两视图之间是 pure shift：$I_R(x) = I_L(x - d)$。但 DP 不是。在 out-of-focus 区域，scene point 投影到 left/right sub-pixel 时被两片不同的 lens aperture 区域成像，因此两个 view 各自的 PSF 形状不同（Fig. 2 中画出的两个 2D kernel）。这意味着匹配 cost 在 disparity 维度上不再是干净的 "shift-and-subtract"，SGM 的 implicit model 被违反。

**Difference 2：极小的 disparity range。**

DSLR 数据上 disparity ∈ [-12, 6]，phone (Pixel 2/3) 上甚至只有 [-1.3, 0.5]。这意味着：
- integer disparity 几乎没有几个候选值；
- sub-pixel 精度不再是 "锦上添花"，而是 "硬性前置条件"；
- SGM 那种 "先 winner-take-all 选 integer disparity，再 parabola fitting 做 sub-pixel refinement" 的两步式 pipeline 在这里会失败，因为 integer selection 阶段的 quantization error 占比极大。

paper 中 Fig. 1 对比了 SGM 与 CCA 的输出：SGM 给出非常 noisy 的 disparity map，CCA 给出平滑连续的 disparity。这就是 motivation 的直观证据。

---

## 2. CCA 的核心 idea：用 parabola 表示 continuous cost，且保持聚合后仍是 parabola

### 2.1 关键 insight

SGM 之所以复杂度是 $O(WHDR)$，是因为它在 3D cost volume $(x, y, d)$ 上沿 paths 做 aggregation。如果我们能把每个像素的 1D cost slice $C_p(d)$ 压缩成少量参数，并且这种压缩在加法下封闭（即两个 compressed cost 相加仍是同 family 的 compressed cost），那 aggregation 就退化成 2D 参数传播，复杂度 $O(WHR)$。

CCA 选了 quadratic function (parabola) 作为这个压缩 family，因为：
1. Parabola is convex → unique minimum → 直接给出 sub-pixel disparity；
2. Parabolas are closed under addition → $\sum_i (a_i d^2 + b_i d + c_i)$ 仍是 parabola；
3. Parabola closed under multiplication by positive scalar → 可以做 confidence weighting；
4. Parabola closed under composition with quadratic regularizer → smoothness term $(d - m)^2$ 与 parabola 相加仍是 parabola。

这四条性质共同保证了：从 initial parabola → path aggregation → multi-path summation → multi-scale fusion，全流程每一步的 cost 都是 parabola，最终 minimization 是 closed-form。

### 2.2 为什么 parabola 而不是 higher-order polynomial？

Paper 在 conclusion 里提到 "higher-order polynomials for continuous cost representation" 作为 future work。我猜他们试过，但 cubic 不再 convex、不再 closed under addition-with-quadratic-regularizer（cubic + quadratic = cubic，但 cubic 可能没有 unique minimum）。Quadratic 是 "刚好够用" 的 sweet spot。

---

## 3. 数学细节与公式拆解

### 3.1 Initial continuous cost (Sec 3.1, Eq. 1-2)

对每个像素 $p$，先算 discrete cost $C_{int}(p, d)$ 对所有 integer $d$（用 SAD / NCC 等都可以）。设最小 cost 对应的 integer disparity 为 $d^0$。在 $d^0$ 周围取三点 $\{d^0-1, d^0, d^0+1\}$ 拟合 parabola：

$$C_{p, d^0}(\Delta d) = a_p \Delta d^2 + b_p \Delta d + c_p$$

变量说明：
- $\Delta d$：相对于 $d^0$ 的 continuous offset（sub-pixel 增量）
- $a_p$：二次项系数 = $\frac{C(d^0+1) + C(d^0-1) - 2C(d^0)}{2}$，即**二阶中心差分的一半**，几何上等于 cost curve 在 $d^0$ 处的曲率
- $b_p$：一次项系数 = $\frac{C(d^0+1) - C(d^0-1)}{2}$，即**一阶中心差分的一半**，等于 cost curve 在 $d^0$ 处的斜率
- $c_p$：常数项 = $C(d^0)$，即最小 cost 值本身

注意：$a_p$ 必须为正（否则不是 convex），这也是为什么后面用 threshold $T_a$ 把 $a_p < T_a$ 的 parabola invalidate。

**Reparametrize 到 absolute disparity $d = d^0 + \Delta d$**（Eq. 2）：

$$C_p(d) = \alpha_p d^2 + \beta_p d + \gamma_p$$

代入推导：
$$
\begin{aligned}
C_p(d) &= a_p(d - d^0)^2 + b_p(d - d^0) + c_p \\
&= a_p d^2 - 2 a_p d^0 \cdot d + a_p (d^0)^2 + b_p d - b_p d^0 + c_p \\
&= \underbrace{a_p}_{\alpha_p} d^2 + \underbrace{(b_p - 2 a_p d^0)}_{\beta_p} d + \underbrace{(c_p + a_p (d^0)^2 - b_p d^0)}_{\gamma_p}
\end{aligned}
$$

为什么要 reparametrize？因为后续 aggregation 时不同像素的 $d^0$ 不同，统一在 absolute $d$ 空间下相加才有意义。在 $\Delta d$ 空间下相加是错的，因为它们 reference 不同的 origin。

**Local optimum 与 confidence**：
$$d_p^{\text{optimal}} = -\frac{\beta_p}{2\alpha_p}$$

这就是 parabola vertex 的标准公式。Confidence 由 $\alpha_p$（curvature）表示：curvature 越大 → parabola 越 "窄尖" → 越 confident。$\gamma_p$ 不影响 minimizer，可跳过计算 — 这是实现上一个小优化。

### 3.2 Confidence score 修正 (Sec 3.2, Eq. 3)

朴素的 $\alpha_p$ 只考虑单一 minimum。在 repetitive pattern / noisy 区域，真实 disparity 可能落在 "次小 cost" 处。Paper 假设：真实 disparity 即使不是 minimum，也应离 minimum 不远，且 second-lowest cost 离 minimum 越近，越不可信。

设 $d^1$ 是 second-lowest cost 对应的 integer disparity，且 $|d^1 - d^0| > 1$（不相邻，避免 trivial case）。定义 ratio：

$$q = \frac{C_{int}(p, d^0)}{C_{int}(p, d^1)}$$

$q \in (0, 1]$，$q$ 越接近 1（两个 cost 越接近）→ confidence 越低。Scale factor：

$$S_{\text{confidence}} = \max\left(\min\left(\frac{1 - q}{1 - T_q}, 1\right), \epsilon\right)^2$$

变量说明：
- $T_q$：ratio threshold，当 $q < T_q$（即 minimum 显著小于 second-minimum）时 $\frac{1-q}{1-T_q} \geq 1$，被 $\min(\cdot, 1)$ 截到 1，相当于不 scale
- $\epsilon$：下限，避免 confidence 归零
- 平方：使 scale 对 $\alpha_p$ 与 $\beta_p$ 同时作用时，minimizer $-\beta_p/(2\alpha_p)$ 保持不变（因为分子分母同乘 $S$），只降低 curvature → parabola 变宽 → 在 aggregation 中权重降低

这个 trick 很聪明：**通过同向 scale $\alpha$ 与 $\beta$ 实现 "降 confidence 但不偏移 minimum"**。

### 3.3 Cost aggregation (Sec 3.3, Eq. 4-5) — 全文的数学核心

这是 paper 最 elegant 的部分。对单条 path $r$，定义 aggregated cost：

$$L_p(d) = C_p(d) + P_{\text{adapt}} \cdot (d - m_{p-1})^2$$

变量说明：
- $C_p(d)$：当前像素的 local continuous cost (parabola)
- $m_{p-1} = \arg\min_d L_{p-1}(d)$：path 上前一个像素 aggregated parabola 的 minimizer
- $P_{\text{adapt}}$：adaptive smoothness penalty

**关键观察**：因为 smoothness term 是关于 $d$ 的 quadratic（$(d - m_{p-1})^2$），且 $C_p(d)$ 也是 quadratic，所以 $L_p(d)$ 仍是 parabola！这是整个方法成立的根本原因。如果 smoothness term 是 L1（如 SGM 中 $P_1, P_2$ 那种 piecewise linear penalty），那么 $L_p$ 就不再是 parabola，整套 closed-form 推导就崩了。

展开 $L_p(d)$：

$$
\begin{aligned}
L_p(d) &= \alpha_p d^2 + \beta_p d + \gamma_p + P_{\text{adapt}}(d^2 - 2 m_{p-1} d + m_{p-1}^2) \\
&= (\alpha_p + P_{\text{adapt}}) d^2 + (\beta_p - 2 P_{\text{adapt}} m_{p-1}) d + (\gamma_p + P_{\text{adapt}} m_{p-1}^2)
\end{aligned}
$$

所以新的 aggregated 系数（大写希腊字母）：

$$
\begin{aligned}
\mathrm{A}_p &= \alpha_p + P_{\text{adapt}} \\
\mathrm{B}_p &= \beta_p - 2 P_{\text{adapt}} m_{p-1} \\
\Gamma_p &= \gamma_p + P_{\text{adapt}} m_{p-1}^2
\end{aligned}
$$

由于 $m_{p-1} = -\mathrm{B}_{p-1} / (2\mathrm{A}_{p-1})$，可以替换 $-2 m_{p-1} = \mathrm{B}_{p-1} / \mathrm{A}_{p-1}$，得到 paper 中的形式：

$$
\begin{aligned}
\mathrm{A}_p &= \alpha_p + P_{\text{adapt}} \\
\mathrm{B}_p &= \beta_p + P_{\text{adapt}} \cdot \frac{\mathrm{B}_{p-1}}{\mathrm{A}_{p-1}} \\
\Gamma_p &= \gamma_p + P_{\text{adapt}} \cdot \left(\frac{\mathrm{B}_{p-1}}{2\mathrm{A}_{p-1}}\right)^2
\end{aligned}
$$

**Intuition 解读**：
- $\mathrm{A}_p$ (new curvature) = local curvature + smoothness penalty。Smoothness 越强，curvature 越大 → 越自信 → 后续像素更倾向跟随这个 parabola。
- $\mathrm{B}_p$ (new slope) = local slope + penalty × previous slope-direction。这一项是把 "previous minimizer" 的 "拉力" 加进来：如果 previous minimizer 大，那么 $\mathrm{B}_{p-1}/\mathrm{A}_{p-1}$ 大（注意 $\mathrm{B}_{p-1}$ 与 minimizer 异号，因为 $m = -\mathrm{B}/(2\mathrm{A})$），$\mathrm{B}_p$ 被拉向 previous minimizer 方向。
- $\Gamma_p$ 不影响 minimizer，所以实际只需 propagate $\mathrm{A}$ 和 $\mathrm{B}$。

**Adaptive penalty**：

$$P_{\text{adapt}} = P \cdot \mathrm{A}_{p-1} \cdot \exp\left(-\frac{(I_p - I_{p-1})^2}{\sigma^2}\right)$$

三部分：
1. $P$：用户参数，控制整体 smoothness 强度
2. $\mathrm{A}_{p-1}$：previous pixel 的 confidence。**Confidence 越高的 previous pixel 对当前 pixel 拉力越大** — 这是一种 confidence-weighted propagation
3. $\exp(\cdot)$：image-gradient based edge detector。Pixel 值差异大（疑似 edge）时，penalty 衰减，允许 disparity 跨 edge 跳变

这种 $P \cdot \mathrm{A}_{p-1}$ 的设计非常聪明：confidence 与 penalty 耦合，相当于 "我自己 confident 才去影响别人"。

**Multi-path 聚合 + 最终 minimization**：

总 cost 是所有 paths 的和：

$$S_p(d) = \sum_r L_p^r(d) = \left(\sum_r \mathrm{A}_p^r\right) d^2 + \left(\sum_r \mathrm{B}_p^r\right) d + \text{const}$$

仍是 parabola。最终 disparity：

$$\boxed{\quad \text{disparity}(p) = -\frac{\sum_r \mathrm{B}_p^r}{2 \sum_r \mathrm{A}_p^r} \quad}$$

这就是 Eq. 5。注意分母 $\sum_r \mathrm{A}_p^r$ 通常远大于 0（多 path 求和），数值稳定。但 paper 仍加了一道保险：如果 $\alpha_p < T_a$，把初始系数设为 $\{\epsilon, 0, 0\}$，意味着该像素 local cost 完全不 informative，minimizer 完全由 path propagation 决定。

### 3.4 Multi-scale fusion (Sec 3.4)

Coarse scale $s$ 的 aggregated 系数 $\{\mathrm{A}_{p,s}, \mathrm{B}_{p,s}\}$ 上采到 fine scale $s-1$：

$$
\begin{aligned}
\mathrm{A}_{p,s-1}^{\text{prior}} &= \text{upsample}(\mathrm{A}_{p,s}) \\
\mathrm{B}_{p,s-1}^{\text{prior}} &= \text{upsample}(\mathrm{B}_{p,s} \cdot F)
\end{aligned}
$$

变量 $F$：scale factor（如 $F=2$ 表示 fine scale 是 coarse scale 的 2 倍分辨率）。**为什么要给 $\mathrm{B}$ 乘 $F$ 而不给 $\mathrm{A}$ 乘？**

因为 minimizer $m = -\mathrm{B}/(2\mathrm{A})$。在 fine scale，disparity 范围是 coarse scale 的 $F$ 倍（因为图像放大了，相同 scene 的 pixel disparity 按比例放大）。所以 $m$ 应乘 $F$，即 $\mathrm{B}$ 应乘 $F$（$\mathrm{A}$ 不变）。

然后 fine scale 的 effective local cost 系数被改写为 prior + local 的加权和：

$$
\begin{aligned}
\alpha_{p,s}^{\text{with prior}} &= \alpha_{p,s} + w \cdot \mathrm{A}_{p,s-1}^{\text{prior}} \\
\beta_{p,s}^{\text{with prior}} &= \beta_{p,s} + w \cdot \mathrm{B}_{p,s-1}^{\text{prior}}
\end{aligned}
$$

$w$ 是 fusion weight。

**关键 intuition**：这是一个 additive prior，而非 hard constraint。在 high-confidence 区域（$\alpha_{p,s}$ 大），local cost 主导；在 low-confidence / blurred 区域（$\alpha_{p,s}$ 小），prior 主导。这与 "coarse-to-fine with hard constraint" 不同 — 后者一旦 coarse level 错了就 propagation 错到底，而 CCA 的 additive 形式允许 fine level 修正 coarse level 的错误（如果 local evidence 足够强）。

Paper 还提到一个 acceleration trick：根据 coarse scale 的 disparity range $[d_{\min}, d_{\max}]$，fine scale 只计算 $[d_{\min}-1, d_{\max}+1]$ 范围内的 cost volume，不计算全部。但这是 global range，不是 per-pixel range，避免过早 hard decision。

### 3.5 Complexity (Sec 3.5)

| 项 | CCA | SGM |
|---|---|---|
| Cost volume 初始化 | $O(WHD)$ | $O(WHD)$ |
| Aggregation | $O(WHR)$ | $O(WHDR)$ |
| 空间 | $O(WH)$ | $O(WHD)$ |

CCA 在空间上完全不依赖 $D$！这是因为只存 2 个 scalar（$\mathrm{A}, \mathrm{B}$）per pixel per path，不像 SGM 要存整个 cost volume。对 DP 这种 $D$ 小的情况时间优势不显著，但空间优势意味着可以在 edge device 上跑；对 stereo 这种 $D$ 大的情况时间优势显著。

---

## 4. 与 SGM 的更深对比

### 4.1 SGM 复习

SGM 的 path cost：
$$L_r(p, d) = C(p, d) + \min\left(\begin{array}{l} L_r(p-r, d) \\ L_r(p-r, d\pm1) + P_1 \\ L_r(p-r, d\pm k) + P_2 \end{array}\right)$$

这是 DP-style 递归，但 penalty 是 piecewise linear，不保持任何 closed form family。Aggregation 后还要 WTA + parabola fitting 做 sub-pixel。

### 4.2 CCA 的根本性区别

| 维度 | SGM | CCA |
|---|---|---|
| Continuous cost | 后处理 parabola fitting | 内置在 aggregation 中 |
| Penalty 形式 | L1-like ($P_1, P_2$) | L2 ($P_{\text{adapt}} (d-m)^2$) |
| Aggregation 空间 | 3D $(x, y, d)$ | 2D $(x, y)$ |
| Sub-pixel 精度 | 仅在 final WTA 后 | 在每一步 propagation 中 |
| Edge handling | $P_1, P_2$ + image gradient | $P_{\text{adapt}}$ 中 exp factor |
| Confidence | 需额外计算 | 内建在 $\mathrm{A}$ 中 |

L2 penalty 的代价：无法很好处理 sharp depth discontinuities（L2 倾向 over-smoothing）。但 DP disparity range 本身很小（最多十几个 pixel），discontinuities 也不剧烈，所以 L2 是合理的。Paper 在 Middlebury 上测试也确实发现 CCA 在 sharp edges / textureless walls 上略差于 SGM（Table 3）。

---

## 5. 实验数据分析

### 5.1 DSLR 数据集 (Table 1)

| Method | AI(1) | AI(2) | $1-\|\rho_s\|$ | Geometric Mean |
|---|---|---|---|---|
| SDoF [43] | 0.087 | 0.129 | 0.291 | 0.144 |
| DPdisp [33] | 0.047 | 0.074 | 0.082 | 0.065 |
| DPE [31] | 0.061 | 0.098 | 0.103 | 0.110 |
| **CCA** | 0.041 | 0.068 | 0.061 | 0.053 |
| **CCA + filter** | **0.036** | **0.061** | **0.049** | **0.048** |

CCA 在所有 metrics 上都最好。值得注意：
- AI(1), AI(2) 是 affine-invariant error（depth up to affine ambiguity，paper [13] 提出此问题）
- $1 - |\rho_s|$：Spearman correlation based metric
- Geometric Mean：所有 metric 的几何平均，避免单一 metric 主导

CCA 比 DPdisp（一种 explicit PSF modeling 方法）还好，这点很 surprising — 因为 DPdisp 显式建模了不同 PSF，理论上更 "正确"。我猜测原因是 DPdisp 的 patch-wise optimization 在 textureless / occlusion boundary 上失败，而 CCA 的 semi-global aggregation 通过 multi-scale 弥补了 PSF mismatch。

### 5.2 Phone 数据集 (Table 2)

CCA 略优于 SDoF，但作者坦诚 [13] 的 DNN 方法更好（但 [13] 的网络是 device-specific，generalization 差）。Phone 数据挑战：
- Only green channel（单色）
- Spherical aberration 引起 radial disparity distortion（Fig. 8 可见边缘误差大）
- 不同 left/right sub-pixel 的 photometric properties 不同
- 作者用 LPF-based vignetting compensation + bilateral filter photometric correction 来预处理

### 5.3 Middlebury stereo (Table 3)

CCA 在 RMSE 上显著优于 SGM (5.2 vs 9.9 before filter)，但 bad-pixel rate 略差。RMSE 优势来自 sub-pixel 精度（CCA 内置 continuous）；bad-pixel 劣势来自 L2 penalty 在 sharp edge 处的 over-smoothing。这印证了 §4.2 的分析。

### 5.4 Ablation (Tables 4-7)

- **Iterations** (Table 4)：从 1 → 3 iterations，Geometric Mean 从 0.075 → 0.055，效果显著。Multiple iterations 是为了消除 "streaking artifacts"（沿 path 的伪影），思路类似 SGM 的 8-direction 但更彻底。
- **Scales** (Table 5)：1 → 3 scales，Geometric Mean 0.069 → 0.055。Multi-scale 对 DP 至关重要，因为不同 resolution 的 PSF 不同。
- **Cost function** (Table 6)：SAD 最好，SSD 次之，NCC 最差。这与 [20] 的结论一致。
- **Sub-pixel initialization** (Table 7)：ENCC [32] 最好。ENCC = Enhanced Normalized Cross Correlation，是 sub-pixel disparity estimation 的经典方法，参考 https://ieeexplore.ieee.org/document/1041749

---

## 6. 一些 meta-level 的思考

### 6.1 为什么这个方法之前没人想到？

我猜原因是 SGM 社区长期习惯于 L1 penalty（$P_1, P_2$），因为 L1 能更好保持 depth discontinuities。但 L1 不保持 closed-form family，所以必须做 discrete DP。CCA 的关键 trade-off 是：**放弃 L1 的 edge-preserving 性质，换取 L2 的 closed-form 与 sub-pixel 内建**。在 DP 这种小 disparity range 场景，这个 trade-off 是净正收益。

### 6.2 Parabola assumption 的局限

Parabola 只能 capture 单一 minimum。在 repetitive pattern 区域，cost curve 有 multiple minima，parabola 拟合可能选错 minimum。Paper 在 Sec 3.2 用 second-lowest cost 来 downscale confidence，缓解但没解决。如果用 Gaussian Mixture 之类多模态 representation 会更准，但失去 closed-form。

### 6.3 与现代 deep stereo 的关系

CCA 是 learning-free 的，但 paper 提到 "improved cost calculation (e.g. deep features)" 作为 future work。可以想象：用 learned feature 提取 cost volume，再用 CCA 做 aggregation。这会比 MCNet / PSVNet 之类 memory 占用低很多。

### 6.4 与 Bilateral Solver / Optimization 的联系

CCA 的 multi-scale prior 注入方式让我想起 Barron & Poole 的 Bilateral Solver (https://arxiv.org/abs/1601.02588) — 都是 "smoothness penalty + per-pixel confidence"，但 Bilateral Solver 是 linear system，CCA 是 closed-form parabola。CCA 更轻量但表达力弱。

### 6.5 在你的 micrograd / tinygrad 直觉里

如果用 micrograd 实现 CCA，forward pass 就是几个 matrix op：
1. Cost volume: $O(WHD)$ — 一次 conv-like operation
2. Initial parabola: elementwise 三点拟合，$O(WH)$
3. Aggregation per path: 顺序扫描，每个 pixel 2 个 scalar update，$O(WH)$
4. Multi-path sum + argmin: $O(WH)$

整体非常 cache-friendly，没有 large tensor。Backward pass 也简单，因为每一步都是 polynomial arithmetic。这种算法在 GPU 上反而可能不如在 SIMD CPU 上高效（因为 path aggregation 是 sequential scan），但非常适合 mobile NPU / DSP。

---

## 7. References

- 原文（Open Access 镜像）: https://arxiv.org/abs/2403.18460
- SGM 原文 (Hirschmuller 2008): https://ieeexplore.ieee.org/document/4359315
- DP depth estimation (Google SIGGRAPH 2018): https://research.google/pubs/synthetic-depth-of-field-with-a-single-camera-mobile-phone/
- DPdisp (Punnappurath ICCP 2020): https://sites.google.com/view/dualpixeldepth/home
- DPE (Pan CVPR 2021): https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_Dual_Pixel_Exploration_Simultaneous_Depth_Estimation_and_Image_Restoration_CVPR_2021_paper.pdf
- ENCC (Psarakis & Evangelidis ICCV 2005): https://ieeexplore.ieee.org/document/1041749
- Bilateral Solver (Barron & Poole ECCV 2016): https://arxiv.org/abs/1601.02588
- SGM implementation (用于对比): https://github.com/kobybibas/semi_global_matching

---

## 8. 一句话总结

**CCA = "用 parabola 参数化 continuous cost + L2 smoothness 保持 closed-form + 2D propagation 代替 3D aggregation + multi-scale additive prior"。** 整个方法的 elegance 来自于 quadratic function 在 addition、scalar multiplication、composition with quadratic penalty 下的封闭性 — 这使得 SGM 的 semi-global 思想可以在 sub-pixel regime 下高效运行。

读完这篇 paper 我最大的 takeaway 是：**算法设计时如果能找到一个对加法封闭的参数化 family，往往能把 3D 问题压成 2D，complexity 降一个量级。** 这种思路在 optical flow（trajectories as polynomials）、image alignment (LK 的 inverse compositional) 中都有类似 instance，CCA 是在 stereo aggregation 上的一个漂亮实例。
