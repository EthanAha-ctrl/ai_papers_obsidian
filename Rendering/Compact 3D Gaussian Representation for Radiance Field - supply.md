---
source_pdf: Compact 3D Gaussian Representation for Radiance Field - supply.pdf
paper_sha256: b97f3fddb19298aa8155d7f616e2fbeaceba6998919b0c1419cb52d503302776
processed_at: '2026-08-03T16:35:46-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Compact 3D Gaussian Splatting 的人话版

## 0. 先讲清楚 3DGS 在干啥

想象你要把一个真实房间"复制"到电脑里，让人能从任意角度看清它。3DGS 的做法特别朴素：往房间里撒一堆发光的小椭球（Gaussian），每个小椭球记 5 件事——

- 我在哪儿（position，3个数）
- 我多透明（opacity，1个数）
- 我多大、啥形状（scale，3个数 + rotation，4个数）
- 从不同角度看，我啥颜色（SH coefficients，48个数）

为啥要 48 个数表示颜色？因为一面金属从正面看是亮的，从侧面看是暗的，这是 view-dependent effect，得用 Spherical Harmonics (SH) 拟合球面上的函数。

然后 rendering 就是把这些椭球"投影"到屏幕上，按深度排序，一个一个 alpha-blend 出来。整个 pipeline 极快，因为避开了 NeRF 那种沿 ray 反复 sample 的笨办法。

**问题**：一个 Mip-NeRF 360 场景，3DGS 要 734 MB。734 MB 是啥概念？一张 iPhone 照片 3 MB，这个场景相当于 244 张 iPhone 照片的数据量，可它本身只是从几十张照片重建出来的，这显然有巨大的 redundancy。

参考链接：
- 3DGS 原版论文：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- 3DGS 项目主页：https://graphics.cs.uni-tuebingen.de/research/3DGS/

---

## 1. Redundancy 到底在哪儿

作者发现 redundancy 有两种，要分别打掉：

### 1.1 Gaussian 数量冗余：很多球根本没用

3DGS 训练时有一个叫 densification 的过程：发现某个区域 gradient 大（说明细节多），就 clone 或 split 一堆 Gaussian 过去。这导致 Gaussian 数量疯狂膨胀，Bonsai 场景最终能到 125 万个。

但作者做了一个简单实验：把这些 Gaussian 砍到 1/2.42，渲染出来的图 PSNR 几乎没变。这说明大概 60% 的 Gaussian 是 dead weight。

为啥会这样？因为 3DGS 的 pruning 策略只看 opacity——把 opacity 设很小，过一阵子还没涨回来就删。它完全不管一个 Gaussian 的 **volume**（也就是 scale 决定的体积）。一个 scale 极小的 Gaussian，就算 opacity 满 1.0，它在屏幕上覆盖的 pixel 数也是个位数，删了对画面完全没影响。

### 1.2 Attribute 冗余：每个球记的东西太多

每个 Gaussian 有 59 个 float，其中 SH 占 48 个（81%）。这 48 个 SH 系数是 per-Gaussian independent 的——一堵白墙上 1000 个 Gaussian，每个都独立记 48 个数，但它们其实应该几乎一模一样。

这里浪费的就是 **spatial redundancy**：相邻 Gaussian 的颜色高度 correlated，但 3DGS 完全不 exploit 这一点。

---

## 2. 三个核心 idea：用三招分别打两种 redundancy

### 2.1 招式一：Learnable Mask（打数量冗余）

**核心思路**：给每个 Gaussian 配一个 learnable 参数 $m_n$，训练时让 $m_n$ 学一个 binary decision——我该不该留。

直接做 binary decision 不可微，所以用 Straight-Through Estimator (STE)：

$$
M_n = \text{sg}\left(\mathbb{1}[\sigma(m_n) > \epsilon] - \sigma(m_n)\right) + \sigma(m_n) \tag{5}
$$

逐字拆解：
- $m_n$：第 $n$ 个 Gaussian 的 mask logit，是个标量，可学习
- $\sigma(\cdot)$：sigmoid，把任意实数压到 (0,1)
- $\epsilon$：threshold，论文里用 0.5
- $\mathbb{1}[\cdot]$：满足条件输出 1，否则 0
- $\text{sg}(\cdot)$：stop gradient，前向透传、反向截断 gradient

这个公式翻译成人话：**前向时 $M_n$ 是 0 或 1 的硬判断，反向时假装它是 $\sigma(m_n)$ 这个连续值，让 gradient 能流到 $m_n$**。这是 deep learning 处理 discrete decision 的经典 hack，Bengio 2013 提出，详见 https://arxiv.org/abs/1308.3432。

**关键设计：mask 同时作用于 scale 和 opacity**

$$
\hat{\Sigma}_n = R(r_n) S(M_n s_n) S(M_n s_n)^T R(r_n)^T \tag{6}
$$

$$
\hat{\alpha}_n(x) = M_n o_n \exp(\ldots) \tag{7}
$$

变量解释：
- $\hat{\Sigma}_n$：masked 后的 3D covariance（决定 Gaussian 在 3D 空间里的形状）
- $R(r_n)$：从 quaternion 构造的 rotation matrix
- $S(M_n s_n)$：被 mask 后的 scale 形成的对角矩阵
- $\hat{\alpha}_n(x)$：masked 后的最终 pixel opacity

**为啥要同时 mask scale 和 opacity**？因为只 mask opacity，Gaussian 还会参与 rasterization 的 sorting 和 tile assignment，浪费计算。同时 mask scale 等于把 Gaussian 的体积直接归零，它在 pipeline 里彻底消失。

**Mask loss**：

$$
L_m = \frac{1}{N} \sum_{n=1}^{N} \sigma(m_n) \tag{8}
$$

这个 loss 鼓励所有 mask 都趋向 0（即所有 Gaussian 都想被删），但 rendering loss 又要求画面好看，所以最终只有那些删了不影响画面的 Gaussian 会被删掉。$\lambda_m$ 控制 trade-off 强度。

**Intuition 总结**：3DGS 的 densification 是 local greedy（看 gradient 大就 split），完全不知道哪些 Gaussian 是真正 redundant 的。Learnable mask 是 global optimization 视角——让每个 Gaussian 在 gradient descent 中自己争抢"贡献度"，没贡献的自动被压到 0。这跟 sparse coding / L1 regularization 的精神一脉相承。

**Ablation 数据**（Table VI，Playroom 场景）：
- 3DGS 原版：2.34M Gaussians，553 MB，PSNR 29.87
- 加 mask：967K Gaussians（**减少 59%**），228 MB，PSNR 29.91（**反而升高**）

PSNR 反而升高的现象值得 build intuition：redundant Gaussian 不仅没用，还可能轻微 hurt 质量，因为它们会增加 rasterization 时的 sorting 噪声、产生小 floaters。

### 2.2 招式二：Grid Neural Field 替代 SH（打 color 冗余）

**核心思路**：一堵白墙上所有 Gaussian 共享同一个 color representation，而不是每个 Gaussian 各记一份 SH。

用 Instant NGP [4] 的 multiresolution hash grid，输入 position 输出 color：

$$
c_n(d) = f(\text{contract}(p_n), d; \theta) \tag{12}
$$

变量：
- $c_n(d)$：第 $n$ 个 Gaussian 在 view direction $d$ 下的颜色
- $f(\cdot; \theta)$：hash grid + tiny MLP，参数 $\theta$
- $p_n$：Gaussian 的 3D position
- $d$：view direction

**Contract function**（处理 unbounded outdoor scenes）：

$$
\text{contract}(p_n) = \begin{cases} p_n & \|p_n\| \leq 1 \\ \left(2 - \frac{1}{\|p_n\|}\right) \frac{p_n}{\|p_n\|} & \|p_n\| > 1 \end{cases} \tag{13}
$$

contract 把整个 $\mathbb{R}^3$ 压缩到半径 2 的球内。单位球内不变，单位球外把无穷远映射到半径 2 的球面上。这是 Mip-NeRF 360 [23] 的设计，详见 https://jonbarron.info/mipnerf360/。

**架构细节**：
- 16 个 resolution levels，从 $2^4 = 16$ 到 $2^{12} = 4096$
- 每 level 2 个 channel
- Hash map size：real scenes $2^{19}$, synthetic $2^{16}$
- Tiny MLP：2 layer × 64 channel，输入 hash feature + view direction

**Storage 对比**（Table VIII，Mip-NeRF 360）：
- 3DGS 的 SH 部分：606.9 MB（占总 746 MB 的 81%）
- 本方法的 hash grid + MLP：25.2 MB + 0.016 MB

**80× 压缩**，因为 color 不再随 Gaussian 数量 $N$ 线性增长，hash grid 大小固定。

**Intuition**：这是把 inductive bias 从 "per-element independent" 换成 "spatially smooth"。卷积网络共享 weights、attention 共享 KV cache 都是同一个 insight——explicit representation 之所以 parameter-inefficient，因为它假设每个 element 互相独立。一旦引入 spatial sharing，参数量就降一个数量级。

### 2.3 招式三：R-VQ Codebook（打 geometry 冗余）

**核心观察**：大量小 Gaussian 在一个 scene 里只有少数几种"形状模式"。一面墙上几百个 Gaussian 的 scale 和 rotation 几乎一样。

**Idea**：学一个 codebook $\mathcal{Z} \in \mathbb{R}^{C \times d}$，每个 Gaussian 只存 codebook 里的 index。

naive VQ 的问题：要精确表示所有 vector，codebook 得很大。所以用 Residual VQ (R-VQ) [80]，级联 $L$ 个 stage，每个 stage 量化前一个 stage 的 residual：

$$
\hat{r}_n^l = \sum_{j=1}^{l} \mathcal{Z}^j[i_n^j] \tag{9}
$$

$$
i_n^l = \arg\min_k \left\| \mathcal{Z}^l[k] - (r_n - \hat{r}_n^{l-1}) \right\|_2^2 \tag{10}
$$

变量：
- $r_n \in \mathbb{R}^4$：原始 rotation quaternion
- $\hat{r}_n^l$：经过 $l$ 个 stage 后的 quantized rotation
- $\mathcal{Z}^l$：第 $l$ 个 stage 的 codebook（size $C$，每条 entry 4 维）
- $i_n^l$：第 $n$ 个 Gaussian 在第 $l$ 个 stage 选中的 index

**人话解释**：
- Stage 1：在 codebook 1 里找最像 $r_n$ 的 code
- Stage 2：算 residual（差距），在 codebook 2 里找最像 residual 的 code，加回去
- 重复 $L$ 次，越往后 residual 越小，codebook 越 fine-grained

每个 stage 只存 1 个 index（$\log_2 C$ bits），$L$ 个 stage 共 $L \log_2 C$ bits per Gaussian。比如 $L=6, C=64$ → 36 bits ≈ 4.5 bytes，相比原版 16 bytes (4 × 32-bit float) 压缩 3.5×。

**训练 loss**（VQ-VAE 风格）：

$$
L_r = \frac{1}{NC} \sum_{k=1}^{L} \sum_{n=1}^{N} \left\| \text{sg}[r_n - \hat{r}_n^{k-1}] - \mathcal{Z}^k[i_n^k] \right\|_2^2 \tag{11}
$$

stop gradient 防止训练不稳，gradient 只更新被选中的 code。

**工程细节**：
- codebook 用 K-means 初始化（避免 random 起步时大部分 code 没人选中——codebook collapse 问题）
- R-VQ 只在最后 1K iterations 启用，之前用原始连续 attributes 训练

**Fig. 9 的可视化很有 intuition**：Stump 场景各 stage 的 codebook indices 分布——Stage 1 的 norm 0.9752、indices 均匀分布（抓 coarse pattern），Stage 6 的 norm 0.0074、indices 高度集中（finer residual）。这印证了 R-VQ 多 stage 设计是合理的：前几个 stage 抓 dominant pattern，后面 stage 精修。

参考 R-VQ 原始来源 SoundStream：https://arxiv.org/abs/2107.03312

---

## 3. Dynamic Scenes 怎么扩展

Baseline 是 STG [19]（Spacetime Gaussian），项目主页 https://zju3dv.github.io/st-gaussian/。

### 3.1 STG 的核心设计

每个 Gaussian 有个"出场时间" $\mu_n$——它在这个时间点最 prominent。opacity 随时间衰减：

$$
o_n(t) = so_n \exp\left(-\xi_n |t - \mu_n|^2\right) \tag{19}
$$

变量：
- $so_n$：spatial opacity（time-independent）
- $\xi_n$：temporal scale，越大 effective duration 越短
- $|t - \mu_n|^2$：时间距离的平方

运动用 polynomial basis：

$$
p_n(t) = sp_n + \sum_{k=1}^{no_p} u_{n,k}(t - \mu_n)^k \tag{15}
$$

$$
r_n(t) = sr_n + \sum_{k=1}^{no_r} v_{n,k}(t - \mu_n)^k \tag{16}
$$

变量：
- $sp_n, sr_n$：canonical position/rotation（$t = \mu_n$ 时刻）
- $u_{n,k}, v_{n,k}$：polynomial coefficients
- $no_p = 3, no_r = 1$：polynomial order

**人话**：每个 Gaussian 不是每个时间点都存一份 attribute，而是用一个 polynomial 拟合它的运动轨迹，只存 polynomial coefficients。

### 3.2 三个 idea 怎么搬过来

**(1) Space-Time Mask**：把 Eq. 5 的 binary mask 应用到时间相关的 covariance 和 opacity：

$$
\hat{\Sigma}_n(t) = R(r_n(t)) S(M_n s_n) S(M_n s_n)^T R(r_n(t))^T \tag{23}
$$

$$
\hat{o}_n(t) = M_n o_n \exp\left(-\xi_n |t - \mu_n|^2\right) \tag{24}
$$

**这里有个 subtle 的 advantage**：dynamic 场景下 post-hoc pruning 需要评估每个 Gaussian 在所有时间点 $t$ 的 importance，计算量爆炸。Learnable mask 通过 gradient descent 自动学到跨整个时间维度的 importance——mask parameter 是所有 timestamps 的 rendering loss 共同 update 的，等效于自动评估 temporal importance。

**(2) Color neural field**：STG 把 color 表示为 9 维 feature（6 维 spatial+view, 3 维 temporal）。本文用 hash grid 替代前 6 维：

$$
c_n(t) = \text{stack}\left(f(\text{contract}(sp_n); \theta), (t - \mu_n) \hat{sc}_n\right) \tag{25}
$$

变量：
- $sp_n$：canonical position
- $\hat{sc}_n$：R-VQ 量化后的 temporal color feature
- $\mu_n$：temporal center

**(3) R-VQ**：应用到 time-invariant 几何属性 $s_n, sr_n$（scale + canonical rotation），以及 temporal 属性 $v_{n,k}$（rotation polynomial coefficients）和 $\hat{sc}_{n,7:9}$（temporal color feature）。

**为啥不压缩 $u_{n,k}$（position polynomial coefficients）**？position 需要高精度，且 polynomial basis 已经够 compact，再压缩会 hurt 质量。

**Dynamic 场景 ablation**（Table VII，Painter 场景）：
- STG：553K Gaussians, 84.1 MB, PSNR 36.21
- 加 mask：145K Gaussians（**减少 74%**），22.0 MB, PSNR 36.29
- 全套：132K Gaussians, 6.56 MB（**最终 12.4× 压缩**），PSNR 36.35

---

## 4. Post-processing 还能再压一刀

虽然 end-to-end 训练已经大幅压缩，作者还做了 "+PP" 变体，叠加几个标准压缩技巧：

1. **8-bit min-max quantization**：hash grid 参数 + scalar attributes (opacity, $\mu_n$, $\xi_n$)
2. **Pruning hash grid**：删掉绝对值 < 0.1 的参数
3. **Morton order sorting** [22]：把 Gaussian 按 Morton curve 排序，让空间相邻的 Gaussian 在 storage 中也相邻，提升后续 entropy coding 效率
4. **Huffman encoding** [84]：对 8-bit values 与 R-VQ indices 做 entropy coding
5. **DE-FLATE** [85]：最后通用无损压缩

参考 Compressed 3DGS：https://github.com/graphdeco-inria/compressed-3dgs

**Storage breakdown**（Table VIII，Mip-NeRF 360，单位 MB）：

| Component | 3DGS | Ours (FP) | Ours+PP |
|---|---|---|---|
| Position | 37.9 | 8.3 | 8.3 |
| Opacity | 12.6 | 2.8 | 1.2 |
| Scale | 37.9 | 6.3 (R-VQ) | 5.9 |
| Rotation | 50.6 | 6.3 (R-VQ) | 6.2 |
| Color | 606.9 | 25.2 (Hash) | 7.4 |
| MLP | - | 0.016 | 0.016 |
| **Total** | **746** | **48.8** | **26.2** |

**最终 28× 压缩**，质量几乎不变。Position 保持 16-bit float 不动，因为几何精度对视觉质量最敏感。

---

## 5. 实验结果一览

### Static scenes

**Mip-NeRF 360**（Table I）：
- 3DGS*：PSNR 27.46, 120 FPS, 746 MB
- Ours+PP：PSNR 27.03, -, 26.2 MB（**28× 压缩，PSNR 只降 0.43 dB**）

**Deep Blending**（Table II）：
- 3DGS*：PSNR 29.46, 132 FPS, 663 MB
- Ours+PP：PSNR 29.73, -, 21.6 MB（**31× 压缩，PSNR 反而升 0.27 dB**）

Deep Blending 这里 PSNR 升高的 intuition：hash grid 的 spatial smoothness 起到 regularizer 作用，减少了 3DGS 的 floaters。

**NeRF-Synthetic**（Table III）：
- 3DGS：PSNR 33.32, 359 FPS, 68.1 MB
- Ours+PP：PSNR 32.88, -, 2.47 MB（**28× 压缩**）

### Dynamic scenes

**DyNeRF**（Table IV）：
- STG*：PSNR 31.94, 181 FPS, 197 MB
- Ours+PP：PSNR 31.69, -, 15.4 MB（**12.8× 压缩**）

**Technicolor**（Table V）：
- STG*：PSNR 33.5, 105 FPS, 1.3 MB/frame
- Ours+PP：PSNR 33.1, 116 FPS, 0.16 MB/frame（**8× 压缩**）

---

## 6. 跟 concurrent works 怎么比

同一时期有好几个想做 storage-efficient 3DGS 的工作：

- **LightGaussian** [21]：post-hoc pruning + SH quantization + distillation。https://lightgaussian.github.io/
- **Compressed 3DGS** [22]：post-hoc pruning + quantization + entropy coding。https://github.com/graphdeco-inria/compressed-3dgs
- **Compact3D** [54]：post-hoc VQ for Gaussian attributes。https://maincold2.github.io/compact3d/
- **EAGLES** [56]：end-to-end，但只是调整 densification schedule 控制数量。https://github.com/ExplainableML/EAGLES

本文独特性有三点：

1. **唯一在 training 过程中 mask Gaussians 的 end-to-end 方法**。EAGLES 虽然也 end-to-end，但它只是粗暴调整 densification schedule（控制 split 频率），是 suboptimal 的 indirect control。本文是直接 learn mask，让 gradient 告诉模型哪个 Gaussian 该删。
2. **唯一扩展到 dynamic scenes**。其他几个都只针对 static。
3. **Color 用 neural field 替代 SH**。其他都是 quantize SH 本身（比如把 48 维 SH 量化到 8-bit），本质还是 per-Gaussian independent。

---

## 7. 几个值得 build intuition 的点

### 7.1 为什么 learnable mask 比 post-hoc pruning 更好

Post-hoc pruning（LightGaussian, Compressed 3DGS 的做法）需要先训练完，再用某个 criterion（比如 opacity × volume 评分）删掉不重要的 Gaussian。问题：训练时模型已经"适应"了有这些 Gaussian 的状态，删掉后 rendering quality 会掉一点，需要 fine-tune 恢复。

Learnable mask 是 end-to-end 的：mask 在整个训练过程中都参与，模型一直在"适应"Gaussian 会被删的状态，所以最终删掉时 quality 几乎不掉。

更深一层：这跟 knowledge distillation 里"teacher forcing" vs "free-running"的对比有点像。Post-hoc 是先训完再剪枝，类似 teacher forcing；learnable mask 是边训边剪，类似 scheduled sampling，模型对剪枝后的状态更 robust。

### 7.2 为什么 R-VQ 比 naive VQ 好

Naive VQ 要精确表示所有 vector，codebook 得很大，计算和 GPU memory 成本高。R-VQ 用 $L$ 个小 codebook 级联，每个 codebook 只需 $C$ 个 entry，总参数 $L \cdot C \cdot d$。但因为有 residual 累积，精度能逼近 naive VQ 用 $C^L$ 个 entry 的效果。

举例：$L=6, C=64$ → 等效 $64^6 \approx 6.8 \times 10^{10}$ 个 entry，但实际只存 $6 \times 64 = 384$ 个 entry。这是指数级的 effective codebook size 提升。

参考 SoundStream 原始 R-VQ 设计：https://arxiv.org/abs/2107.03312

### 7.3 为什么 hash grid 比 SH compact

SH 假设 per-Gaussian independent，所以 cost 是 $O(N \cdot 48)$，随 Gaussian 数量线性增长。

Hash grid 是 fixed-size data structure，cost 是 $O(\text{hash map size})$，跟 Gaussian 数量无关。一个 Mip-NeRF 360 场景，3DGS 有 ~100 万 Gaussian × 48 SH = 4800 万 float = 190 MB（half precision）。Hash grid $2^{19}$ entries × 2 channel × 16 levels = 600 万 float = 25 MB，固定大小。

**Intuition**：这是把 representation 从 "随元素数量线性增长" 换成 "固定大小 + spatial sharing"。本质上是把 Gaussian 数量和 color 参数量 decouple。

### 7.4 Contract function 为啥这么设计

$$
\text{contract}(p) = \begin{cases} p & \|p\| \leq 1 \\ \left(2 - \frac{1}{\|p\|}\right) \frac{p}{\|p\|} & \|p\| > 1 \end{cases}
$$

人话解释：
- 单位球内：identity，啥也不做
- 单位球外：沿着原方向把点压到半径 $2 - 1/\|p\|$ 的球面上

取极限 $\|p\| \to \infty$：半径趋向 2，所以整个 $\mathbb{R}^3$ 被压到半径 2 的球内。

为啥要这么干？outdoor scenes (Mip-NeRF 360) 有 unbounded depth——天空、远处山脉理论上在无穷远。Hash grid 没法覆盖无穷大空间，所以得把无穷远压到有限区域。

这种 contraction 是 $C^0$ 连续且可微的，Jacobian 在边界处不奇异，能正常 backprop。Mip-NeRF 360 [23] 证明这对 unbounded scene 的 anti-aliasing 至关重要。

---

## 8. 潜在 limitations 和改进方向

论文没专门写 limitations section，但读的时候能感觉到几个点：

1. **Training time 增加**：Mip-NeRF 360 上 33m vs 3DGS 的 24m，hash grid 训练比直接优化 SH 慢。
2. **LPIPS 略升**：mask + R-VQ 引入的 quantization artifact 在 perceptual metric 上有体现。Mip-NeRF 360 上 LPIPS 从 0.214 升到 0.247，虽然小但存在。
3. **Hash collision**：大场景下 hash grid 会有 collision（不同 position 映射到同一个 hash entry），导致高频细节损失。Triplane 或 factorized grid 可能更稳。
4. **Mask threshold $\epsilon$ hard-coded**：不同场景可能需要不同 threshold，做成 learnable 或 adaptive 更优雅。
5. **没跟最新 dynamic methods 比**：只跟 STG，而 4DGS [17]（https://guanjunwu.github.io/4dgs/）和 4K4D [73]（https://zju3dv.github.io/4K4D/）可能在 quality 上更好。

**潜在改进方向**：
- **Adaptive mask threshold**：用 Gumbel-softmax temperature annealing 替代 fixed threshold
- **End-to-end entropy coding**：把 bitrate 直接作为 loss 项，类似 neural compression 文献里的 rate-distortion optimization
- **Triplane 替代 hash grid**：triplane 在大场景下 hash collision 更少，可以 factorize
- **Combined with 4DGS**：用 4DGS 的 hexplane 替代 STG 的 polynomial basis

---

## 9. 一句话 intuition 总结

3DGS 的 storage bloat 来自两个独立 axis：**哪里的 Gaussian 应该存在**（数量冗余），**每个 Gaussian 应该长什么样**（attribute 冗余）。这篇 paper 用三招分别打这两类冗余——

- **Learnable mask** 打数量冗余：用 STE 让每个 Gaussian 自己学一个 binary decision，gradient 全局优化告诉你哪个该删。
- **Hash grid neural field** 打 color 冗余：把 per-Gaussian independent 的 SH 换成 spatially shared 的 grid representation，color 参数量与 Gaussian 数量 decouple。
- **R-VQ codebook** 打 geometry 冗余：利用 Gaussian 在一个 scene 里只有少数几种"形状模式"的 clustering structure，每个 Gaussian 只存 codebook index。

三招合起来在 Mip-NeRF 360 上达到 28× 压缩，质量几乎不掉，渲染速度反而快 1.4×。

更深一层的 intuition：explicit representation 之所以 parameter-inefficient，因为它假设每个 element 是 independent 的。一旦引入 spatial / temporal / structural sharing 机制（这是 CNN、attention、VQ-VAE 共享的核心 insight），压缩率就能提升一个数量级。这篇 paper 本质上是把 neural compression 的经典工具（VQ、entropy coding）和 neural rendering 的 explicit representation（3DGS）做了一个 clean 的 end-to-end 融合。

项目主页：https://maincold2.github.io/c3dgs/

---

如果你想 dive deeper 到某个具体组件，比如 hash grid 的 forward 过程、R-VQ 的 EMA codebook update 细节、contract function 的 Jacobian 推导、或者 4DGS 的 hexplane 跟 STG 的 polynomial basis 的本质区别，我可以继续展开。

---

# Compact 3D Gaussian Splatting for Static and Dynamic Radiance Fields 深度解析

## 1. Motivation 与 Problem Setup

让我先从最高层 motivation 讲起，帮你 build intuition。

3D Gaussian Splatting (3DGS) [1] 的核心 insight 是：放弃 NeRF 沿 ray dense sampling 的 volumetric rendering pipeline，改用 explicit 的 3D Gaussian primitives 配合 tile-based rasterization，从而在保持 photo-realistic 质量的同时获得 real-time rendering speed。但其代价是 **memory bloat**：

- 一个 Mip-NeRF 360 场景 3DGS 要 ~734 MB
- 一个 Deep Blending 场景要 ~676 MB
- 一个 DyNeRF 动态场景 STG 要 ~200 MB

为什么会膨胀？两个 source of redundancy：

1. **Number of Gaussians 冗余**：densification (clone + split) 阶段会盲目放大 Gaussian 数量，但其中很多 Gaussian 的 volume 或 opacity 极小，对最终 rendering 几乎没有贡献（Fig. 3 显示 ×2.42 的 Gaussians 与精简后效果类似）。
2. **Per-Gaussian attribute 冗余**：每个 Gaussian 自带 59 个 float (position 3 + opacity 1 + scale 3 + rotation 4 + SH 48)。其中 SH 48 维占 81% 的参数，且相邻 Gaussian 在 spatial 上具有强相关性，但 3DGS 完全忽略这种 spatial redundancy。

这篇 paper 的核心 thesis 就是：**同时从这两个 axis 压缩，并且 end-to-end training**。最终在 Mip-NeRF 360 上达到 **28× compression**，在 DyNeRF 上达到 **12× compression**，渲染速度反而提升 1.4× 左右。

Project page: https://maincold2.github.io/c3dgs/
3DGS 原版: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 2. 三个核心贡献的技术细节

### 2.1 Learnable Volume Mask (减少 Gaussian 数量)

#### 2.1.1 问题：opacity-based pruning 不够

原版 3DGS 的 pruning：每 ~100 iterations 把所有 opacity 设为很小值，再删除那些仍未"恢复"opacity 的 Gaussian。这个策略主要 kill 的是 **floaters**（飘在空中的孤立 Gaussian），但忽略了一个关键事实：

**一个 Gaussian 即使 opacity 高，如果它的 volume (scale) 极小，它在 rasterization 时覆盖的 pixel 数极少，对 PSNR 贡献可以忽略。** Paper 在 Fig. 3 中用 Bonsai 场景证明：把 Gaussian 数量降到 1/2.42，性能几乎不变。

这就是 "volume-aware masking" 的 motivation。

#### 2.1.2 Straight-Through Estimator (STE) 公式解析

定义一个 per-Gaussian learnable mask parameter $m_n \in \mathbb{R}$。我们想要一个 binary mask $M_n \in \{0, 1\}$，但 binary 不可微，所以用 Bengio 2013 [78] 的 straight-through estimator：

$$
M_n = \text{sg}\left(\mathbb{1}[\sigma(m_n) > \epsilon] - \sigma(m_n)\right) + \sigma(m_n) \tag{5}
$$

变量含义：
- $m_n$：第 $n$ 个 Gaussian 的 learnable mask logit（标量）
- $\sigma(\cdot)$：sigmoid function，把 logit 映射到 $(0, 1)$
- $\epsilon$：masking threshold（通常 0.5）
- $\mathbb{1}[\cdot]$：indicator function，条件满足为 1，否则为 0
- $\text{sg}(\cdot)$：stop-gradient operator，前向时 identity，反向时 gradient 为 0

intuition：
- **Forward pass**：$M_n = \mathbb{1}[\sigma(m_n) > \epsilon]$，即 hard binary decision
- **Backward pass**：gradient 从 $\sigma(m_n)$ 直接流过，相当于假装 $M_n = \sigma(m_n)$ 是连续的

这是 deep learning 里处理 discrete decision 的经典 trick（REINFORCE 之外更便宜的选择，相关讨论见 https://arxiv.org/abs/1308.3432）。

#### 2.1.3 Mask 同时作用于 scale 和 opacity

把 mask 应用到 covariance 和 opacity：

$$
\hat{\Sigma}_n = R(r_n) S(M_n s_n) S(M_n s_n)^T R(r_n)^T \tag{6}
$$

$$
\hat{\alpha}_n(x) = M_n o_n \exp\left(-\frac{1}{2}(x - p'_n)^T \hat{\Sigma}'_n{}^{-1} (x - p'_n)\right) \tag{7}
$$

变量含义：
- $\hat{\Sigma}_n$：masked 3D covariance
- $R(r_n)$：从 quaternion $r_n$ 构造的 3×3 rotation matrix
- $S(M_n s_n)$：被 mask 后的 scale 形成的 diagonal matrix
- $\hat{\alpha}_n(x)$：masked final opacity at pixel $x$
- $o_n$：原始 opacity
- $\hat{\Sigma}'_n$：经过 viewing transformation 投影后的 2D covariance（即 Eq. 2 的 masked 版本）

**关键 intuition**：mask 同时乘到 scale 和 opacity 上。如果一个 Gaussian 的 $M_n \to 0$，那么它的 volume（决定它在 image plane 上覆盖的 pixel 数）和 opacity 都归零，rendering 时彻底消失。这种 joint design 比单独 mask opacity 更干净——单纯把 opacity 设 0，Gaussian 仍然占用 rasterization 的 sorting 与 tile assignment 计算资源。

#### 2.1.4 Mask loss 与端到端训练

$$
L_m = \frac{1}{N} \sum_{n=1}^{N} \sigma(m_n) \tag{8}
$$

这个 loss 鼓励所有 $\sigma(m_n) \to 0$，即 push 所有的 mask logit 趋向负无穷。但因为 rendering loss $L_{ren}$ 需要 informative 的 Gaussian 保留下来，所以最终只有对 rendering 影响小的 Gaussian 会被压到 0。

$\lambda_m$ 是 hyper-parameter：太大会把所有 Gaussian 全 mask 掉（图像变黑），太小则没有压缩效果。Paper 中 real scenes 用 $5 \times 10^{-4}$，synthetic 用 $4 \times 10^{-3}$（synthetic 场景更简单，可以容忍更激进的 pruning）。

注意 paper 强调一个细节：**与原版 3DGS 在 15K iterations 后 stop densification 不同，本文全程持续 densify + mask**（Fig. 3）。这保证了训练全程的 GPU memory 都处于受控状态——这是大型场景训练时的工程优势。

训练完成后 mask parameter 不需要存储（直接物理删除被 mask 的 Gaussian），所以 mask 本身不带来额外 storage cost。

---

### 2.2 Grid-based Neural Field for View-dependent Color (压缩 SH)

#### 2.2.1 SH 的浪费

3DGS 每个 Gaussian 用 48 个 SH 系数（max 3 degrees）表示 view-dependent color。48 个参数相对于总共 59 个参数，占 81.4%。但相邻 Gaussian 在空间上颜色高度 correlated——一面墙上的 1000 个 Gaussian，它们的 SH 系数应该几乎一样。

这就是用 **grid-based neural field** 替代 per-Gaussian SH 的 motivation。Paper 选 Instant NGP [4] 的 multiresolution hash grid，因为它本身就是为了 compact representation 设计的。

#### 2.2.2 Contract function (处理 unbounded scenes)

对于 outdoor scenes (Mip-NeRF 360)，position 是 unbounded 的，hash grid 无法直接覆盖。Paper 用了 Mip-NeRF 360 [23] 的 contract function：

$$
\text{contract}(p_n) = \begin{cases} 
p_n & \|p_n\| \leq 1 \\
\left(2 - \frac{1}{\|p_n\|}\right)\left(\frac{p_n}{\|p_n\|}\right) & \|p_n\| > 1 
\end{cases} \tag{13}
$$

变量含义：
- $p_n \in \mathbb{R}^3$：第 $n$ 个 Gaussian 的 center position
- $\|p_n\|$：欧氏范数

intuition：
- 单位球内（$\|p_n\| \leq 1$）：identity mapping，不变形
- 单位球外：把所有无穷远点都映射到半径 2 的球面上（l'Hôpital 取极限 $\|p_n\| \to \infty$ 时 contract 值趋向 $2 \cdot \hat{p}_n$）

这种 contraction 是 $C^0$ 连续的、可微的，且把 $\mathbb{R}^3$ 压缩到半径 2 的球内。Mip-NeRF 360 证明这对 unbounded scene 的 anti-aliasing 至关重要。

#### 2.2.3 Neural field 公式

$$
c_n(d) = f(\text{contract}(p_n), d; \theta) \tag{12}
$$

变量含义：
- $c_n(d)$：第 $n$ 个 Gaussian 在 view direction $d$ 下的 color
- $f(\cdot; \theta)$：hash grid + tiny MLP 构成的 neural field，参数 $\theta$
- $d \in \mathbb{R}^3$：从 camera center 到 Gaussian 的 view direction

架构细节：
- Hash grid：16 个 resolution levels，从 $2^4=16$ 到 $2^{12}=4096$
- 每 level 2 channels feature
- 总 hash map size：real scenes $2^{19}$, synthetic $2^{16}$
- Tiny MLP：2 layer, 64 channel, 接收 concatenated feature + view direction $d$

**关键 insight**：原本要存 48 个 SH × $N$ 个 Gaussian，现在只需存 hash grid (固定大小 ~25 MB for real) + 小 MLP (~0.016 MB)。Gaussian 数量 $N$ 不再影响 color 的 storage cost。

注意一个 subtle point：paper 表示的是 **0-degree SH**（即 view-independent 部分的 RGB base color）+ view-dependent part 通过 neural field 学，而不是直接输出 RGB。他们发现这比直接输出 RGB 略微好一些。

#### 2.2.4 在 dynamic scenes 中的扩展

STG [19] 把每个 Gaussian 的 color 表示为 9 维 feature $sc_n$：6 维 spatial+view (static) + 3 维 temporal。本文用 neural field 替换前 6 维：

$$
c_n(t) = \text{stack}\left(f(\text{contract}(sp_n); \theta), (t - \mu_n) \hat{sc}_n\right) \tag{25}
$$

变量含义：
- $sp_n$：canonical position (在 $t = \mu_n$ 时刻)
- $\hat{sc}_n \in \mathbb{R}^3$：R-VQ 量化后的 temporal color feature
- $\mu_n$：temporal center（Gaussian 最 prominent 的时间点）

所以 dynamic 场景下 color 完全不需要 per-Gaussian 存储 6 维 spatial+view color，只需存 3 维 temporal feature + neural field 参数共享。

---

### 2.3 Residual Vector Quantization (R-VQ) Codebook (压缩 geometry 与 temporal attributes)

#### 2.3.1 为什么 geometry 适合 codebook？

观察：在大量小 Gaussian 拼成的场景中，**大多数 Gaussian 的 scale 与 rotation 极度相似**。一面墙上几百个 Gaussian 的 scale 几乎一样，rotation 也几乎一样（都贴墙面）。这意味着 scale 和 rotation 在 N 个 Gaussian 之间存在 strong low-rank / clustering structure。

VQ [79] 的核心 idea：学一个 codebook $\mathcal{Z} \in \mathbb{R}^{C \times d}$，每个原始 vector 只需存一个 index $i \in \{0, ..., C-1\}$。当 $C \ll N$ 时压缩率很高。

但 naive VQ 问题：codebook 要很大才能精确表示所有 vector，compute 与 GPU memory 成本高 [80]。

#### 2.3.2 R-VQ 公式详解

R-VQ [80] 的 idea：级联 $L$ 个 VQ stage，每个 stage 量化前一个 stage 的 residual。

$$
\hat{r}_n^l = \sum_{j=1}^{l} \mathcal{Z}^j[i_n^j], \quad l \in \{1, ..., L\} \tag{9}
$$

$$
i_n^l = \arg\min_k \left\| \mathcal{Z}^l[k] - (r_n - \hat{r}_n^{l-1}) \right\|_2^2, \quad \hat{r}_n^0 = \vec{0} \tag{10}
$$

变量含义：
- $r_n \in \mathbb{R}^4$：第 $n$ 个 Gaussian 的原始 rotation quaternion
- $\hat{r}_n^l$：经过 $l$ 个 stage 后的 quantized rotation
- $\mathcal{Z}^l \in \mathbb{R}^{C \times 4}$：第 $l$ 个 stage 的 codebook（size $C$，vector dim 4 for quaternion）
- $i_n^l$：第 $n$ 个 Gaussian 在第 $l$ 个 stage 选中的 codebook index
- $\hat{r}_n^0 = \vec{0}$：initial residual 即原始 vector 本身

intuition：
- **Stage 1**：在 $\mathcal{Z}^1$ 中找最接近 $r_n$ 的 code，得到 $\hat{r}_n^1 = \mathcal{Z}^1[i_n^1]$
- **Stage 2**：residual 是 $r_n - \hat{r}_n^1$，在 $\mathcal{Z}^2$ 中找最接近此 residual 的 code，累加：$\hat{r}_n^2 = \mathcal{Z}^1[i_n^1] + \mathcal{Z}^2[i_n^2]$
- **Stage L**：累积所有 stage 的 code

每个 stage 只需存 1 个 index ($\log_2 C$ bits)，所以总 cost 是 $L \cdot \log_2 C$ bits per Gaussian。比如 $L=6, C=64$ → $6 \times 6 = 36$ bits ≈ 4.5 bytes per Gaussian for rotation，相比原版 4×32 bit = 16 bytes，节省 3.5×。

#### 2.3.3 R-VQ 训练 loss

$$
L_r = \frac{1}{NC} \sum_{k=1}^{L} \sum_{n=1}^{N} \left\| \text{sg}[r_n - \hat{r}_n^{k-1}] - \mathcal{Z}^k[i_n^k] \right\|_2^2 \tag{11}
$$

变量含义：
- $\text{sg}[\cdot]$：stop gradient
- $r_n - \hat{r}_n^{k-1}$：第 $k$ stage 的 input（即上一 stage 的 residual）

这是 VQ-VAE 风格的 commit loss + codebook loss 合并写法。stop gradient 保证：
- gradient 流向 codebook $\mathcal{Z}^k[i_n^k]$（更新被选中的 code，让它更接近 residual）
- gradient 不流回前面 stages 的累加值（避免训练不稳定）

scale $s$ 用类似的 loss $L_s$。

#### 2.3.4 K-means initialization 与 last 1K iterations

工程细节：
- codebook 用 K-means 在已训练好的 attributes 上初始化（避免从随机起步时大部分 code 不被选中——codebook collapse 问题）
- R-VQ 只在 **最后 1K iterations** 启用，之前用原始连续 attributes 训练。这避免了 R-VQ 早期训练时的 instability，同时保证最终模型确实是 quantized 的。

#### 2.3.5 Fig. 9 中的 stage 分布观察

Paper 在 Stump 场景上可视化各 stage 的 codebook indices 分布：
- Stage 1: Norm 0.9752，indices 分布均匀 → coarse geometry pattern
- Stage 2: Norm 0.1893 → finer residual
- Stage 3-6: Norm 急剧衰减 (0.0765 → 0.0074) → 表示残差越来越小

这印证了 R-VQ 的多 stage 设计是合理的：前几个 stage 抓住 dominant pattern，后面 stage 精修。

#### 2.3.6 在 dynamic scenes 中的应用

STG 用 polynomial basis 表示 temporal motion：
- Position: $p_n(t) = sp_n + \sum_{k=1}^{no_p} u_{n,k}(t - \mu_n)^k$ (Eq. 15)，$no_p = 3$
- Rotation: $r_n(t) = sr_n + \sum_{k=1}^{no_r} v_{n,k}(t - \mu_n)^k$ (Eq. 16)，$no_r = 1$

R-VQ 应用于：
- $s_n$ (scale)：static attribute，spatial redundancy 大
- $sr_n$ (canonical rotation)：static attribute
- $v_{n,k}$ (rotation polynomial coefficients)：temporal redundancy
- $\hat{sc}_{n,7:9}$ (temporal color feature)：temporal redundancy

不压缩 $u_{n,k}$（position polynomial coefficients），因为 position 需要高精度，且 polynomial basis 本身已经 compact。

DyNeRF: $C=256$, geometry stages $L=4$, temporal stages $L=3$
Technicolor: $C=256$, geometry stages $L=5$, temporal stages $L=4$

---

## 3. STG Baseline 与 Dynamic Scene Formulation 详解

由于 dynamic extension 建立在 STG [19] 之上，我把它核心公式梳理清楚。

STG paper: https://zju3dv.github.io/st-gaussian/

### 3.1 Temporal center $\mu_n$ 与 radial basis opacity

每个 Gaussian 有一个最 prominent 的时间点 $\mu_n$，以及 effective duration $\xi_n$：

$$
o_n(t) = so_n \exp\left(-\xi_n |t - \mu_n|^2\right) \tag{19}
$$

变量含义：
- $so_n \in [0, 1]$：spatial opacity（time-independent）
- $\xi_n \in \mathbb{R}$：temporal scale，越大表示 effective duration 越短
- $|t - \mu_n|^2$：L2 时间距离

intuition：这是一个 Gaussian-shaped temporal window，Gaussian 在 $t = \mu_n$ 时最不透明，远离这个时间点时 opacity 衰减。这避免了每个时间戳都存一组 Gaussian attributes，而是用 temporal basis 控制 Gaussian 的"出场时机"。

### 3.2 Final splatted feature 与 color MLP

STG 把 splatting 推广到 feature space：

$$
F(x, t) = \sum_{k=1}^{\mathcal{N}(x, t)} c_k(t) \alpha_k(x, t) \prod_{j=1}^{k-1}(1 - \alpha_j(x, t)) \tag{21}
$$

$$
C(x, t) = F(x, t)_{1:3} + \phi(F(x, t)_{4:6}, F(x, t)_{7:9}, d) \tag{22}
$$

变量含义：
- $F(x, t) \in \mathbb{R}^9$：splatted feature，包含 spatial+view+temporal 三段
- $F_{1:3}$：spatial color (direct RGB)
- $F_{4:6}$：view-dependent color feature
- $F_{7:9}$：temporal color feature
- $\phi(\cdot, \cdot, \cdot)$：MLP，融合 view-dependent 与 temporal feature

intuition：STG 的 design 让 spatial color 直接是 splatting 结果，而 view-dependent 与 temporal 通过小 MLP 后处理，这样大部分场景（静态、view-independent 区域）不需要经过 MLP。

本文的改进是把 $F_{1:6}$ 部分换成 hash grid neural field 输出（Eq. 25），$F_{7:9}$ 用 R-VQ 量化。

### 3.3 Space-Time Mask

把 Eq. 5 的 binary mask 应用到 dynamic covariance 和 opacity：

$$
\hat{\Sigma}_n(t) = R(r_n(t)) S(M_n s_n) S(M_n s_n)^T R(r_n(t))^T \tag{23}
$$

$$
\hat{o}_n(t) = M_n o_n \exp\left(-\xi_n |t - \mu_n|^2\right) \tag{24}
$$

关键 advantage：dynamic 场景下 post-hoc pruning 需要评估每个 Gaussian 在所有 $t$ 上的 importance，非常 expensive。本文的 learnable mask 通过 gradient descent 自动学到每个 Gaussian 的 importance 跨整个时间维度——mask parameter 是在所有 timestamps 的 rendering loss 共同作用下更新的，等效于 automatic 评估 temporal importance。

---

## 4. Post-processing 技术细节

### 4.1 Pipeline

虽然 end-to-end 训练已经大幅压缩，paper 还做了一个 "+PP" (post-processing) 变体：

1. **8-bit min-max quantization**：hash grid 参数 + scalar attributes (opacity, temporal center, temporal scale)
2. **Pruning hash grid**：删掉绝对值 < 0.1 的参数（小参数对最终 feature 贡献小）
3. **Morton order sorting** [22]：把 Gaussian 按 Morton curve 重新排序，让 spatially adjacent 的 Gaussian 在 storage 中也 adjacent，提升后续 entropy coding 效率
4. **Huffman encoding** [84]：对 8-bit quantized values 与 R-VQ indices 做 entropy coding
5. **DE-FLATE** [85]：最后再做通用无损压缩

### 4.2 Storage breakdown (Table VIII)

Mip-NeRF 360 平均 storage 分解（单位 MB）：

| Component | 3DGS | Ours (FP) | Ours+PP |
|---|---|---|---|
| Position | 37.9 | 8.3 (16f) | 8.3 (16f) |
| Opacity | 12.6 | 2.8 (16f) | 1.2 (8b+H) |
| Scale | 37.9 | 6.3 (R-VQ) | 5.9 (+H) |
| Rotation | 50.6 | 6.3 (R-VQ) | 6.2 (+H) |
| Color (SH/Hash) | 606.9 | 25.2 (Hash) | 7.4 (+8b+P+H) |
| MLP | - | 0.016 | 0.016 |
| **Total** | **746** | **48.8** | **26.2** (after DE-FLATE) |

Key takeaways：
- Color 部分从 606.9 MB 降到 7.4 MB（80× 压缩）
- Rotation 从 50.6 → 6.2 MB（8× 压缩）
- Position 保持 16f 不动，因为 position 精度对几何保真度至关重要

参考 LightGaussian [21] 与 Compressed 3DGS [22] 也采用类似 quantization + entropy coding pipeline，但它们是 post-hoc compression（先训练完再压缩），本文是 end-to-end trainable。

---

## 5. 实验结果深度分析

### 5.1 Static scenes (Table I, II, III)

#### Mip-NeRF 360 (9 个真实 unbounded 场景)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Train | FPS ↑ | Storage ↓ |
|---|---|---|---|---|---|---|
| 3DGS | 27.21 | 0.815 | 0.214 | 41m33s | 134 | 734 MB |
| 3DGS* | 27.46 | 0.812 | 0.222 | 24m07s | 120 | 746 MB |
| Ours | 27.08 | 0.798 | 0.247 | 33m06s | 128 | 48.8 MB |
| Ours+PP | 27.03 | 0.797 | 0.247 | - | - | 26.2 MB |

Observations:
- **PSNR 仅降 0.38 dB**（27.46 → 27.08），几乎不可感知
- **SSIM 降 0.014**，**LPIPS 升 0.025**，质量损失极小
- **Storage 压缩 28×**（746 → 26.2 MB）
- **FPS 几乎不变**（120 → 128），但 training 时间多 9 分钟（因为 hash grid 训练慢）

注意 3DGS* 是 paper 作者用相同硬件 (A100) 重新跑的 3DGS baseline，确保比较公平。

#### Deep Blending

| Method | PSNR | SSIM | FPS | Storage |
|---|---|---|---|---|
| 3DGS* | 29.46 | 0.900 | 132 | 663 MB |
| Ours | **29.79** | 0.901 | 181 | 43.2 MB |
| Ours+PP | 29.73 | 0.900 | - | 21.6 MB |

**这里本文 PSNR 反而更高**！作者的解释是：hash grid 的 spatial smoothness 起到了 regularizer 作用，减少了 3DGS 中 floaters 造成的 artifacts。FPS 提升 40%（132 → 181）。

#### NeRF-Synthetic

| Method | PSNR | Storage | Train | FPS |
|---|---|---|---|---|
| 3DGS | 33.32 | 68.1 MB | 6m14s | 359 |
| Ours | 33.33 | 5.55 MB (×0.08) | 8m04s | 545 |
| Ours+PP | 32.88 | 2.47 MB | - | - |

Synthetic 场景更简单，压缩比相对小（68.1 MB 本身就不大），但 FPS 提升 1.52×。

### 5.2 Dynamic scenes (Table IV, V)

#### DyNeRF

| Method | PSNR | SSIM | LPIPS | FPS | Storage |
|---|---|---|---|---|---|
| STG | 32.05 | 0.946 | 0.044 | 140 | 200 MB |
| STG* | 31.94 | 0.948 | 0.046 | 181 | 197 MB |
| Ours | 31.73 | 0.945 | 0.053 | 186 | 21.8 MB |
| Ours+PP | 31.69 | 0.945 | 0.054 | - | 15.4 MB |

PSNR 降 0.21 dB，storage 压缩 12.4× (197 → 15.4 MB)，FPS 178 → 186 (微涨)。

#### Technicolor

| Method | PSNR | FPS | Storage/Fr |
|---|---|---|---|
| STG* | 33.5 | 105 | 1.3 MB |
| Ours+PP | 33.1 | 116 | 0.16 MB |

Per-frame storage 压缩 8×。

### 5.3 Ablation study (Table VI, VII)

#### Static scene ablation (Playroom 与 Bonsai)

| Mask | Col | Geo | Half | Post | PSNR (Playroom) | #Gauss | Storage | FPS |
|---|---|---|---|---|---|---|---|---|
| - | - | 3DGS | - | - | 29.87 | 2.34M | 553 MB | 154 |
| ✓ | - | - | - | - | 29.91 | 967K | 228 MB | 254 |
| ✓ | ✓ | - | - | - | 30.33 | 770K | 59 MB | 210 |
| ✓ | ✓ | ✓ | - | - | 30.33 | 761K | 44 MB | 204 |
| ✓ | ✓ | ✓ | ✓ | - | 30.32 | 778K | 38 MB | 206 |
| ✓ | ✓ | ✓ | ✓ | ✓ | 30.30 | - | 17 MB | - |

Insights:
- **Mask 单独**: 2.34M → 967K Gaussians (减少 59%)，storage 553 → 228 MB，**PSNR 反而升 0.04**，FPS 升 65%。这是 paper 最 striking 的结果——redundant Gaussians 不仅没用，还可能轻微 hurt quality。
- **Color neural field**: 770K → 770K (微减)，storage 228 → 59 MB (3.86× 压缩)，PSNR +0.42。training time 显著增加 (17m → 24m)
- **Geometry codebook**: 44 MB → 38 MB，质量不变
- **Half tensor**: 38 → 35 MB
- **Post-processing**: 35 → 15 MB

#### Dynamic scene ablation (Painter 与 Cut Roasted Beef)

| Mask | Color | Time | Geo | Half | Post | PSNR | #Gauss | Storage |
|---|---|---|---|---|---|---|---|---|
| - | - | STG | - | - | - | 36.21 | 553K | 84.1 MB |
| ✓ | - | - | - | - | - | 36.29 | 145K | 22.0 MB |
| ✓ | ✓ | - | - | - | - | 36.45 | 121K | 19.2 MB |
| ✓ | ✓ | ✓ | - | - | - | 36.28 | 132K | 16.4 MB |
| ✓ | ✓ | ✓ | ✓ | - | - | 36.22 | 132K | 14.0 MB |
| ✓ | ✓ | ✓ | ✓ | ✓ | - | 36.35 | 132K | 10.2 MB |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 36.35 | - | 6.56 MB |

Dynamic 场景下 mask 把 553K → 145K (减少 74%)，效果同样显著。

---

## 6. 与 concurrent works 的对比

paper 在 Section II-A.3 提到几个 concurrent works：

- **LightGaussian** [21]: post-hoc pruning + SH quantization + distillation
- **Compressed 3DGS** [22]: post-hoc pruning + quantization + entropy coding
- **Compact3D** [54]: VQ for Gaussian attributes (post-hoc)
- **Self-organizing Gaussian grids** [55]: reorganize Gaussian storage order for entropy coding
- **EAGLES** [56]: end-to-end，但只是调整 densification schedule 控制数量

本文独特性：
1. **唯一成功在 training 过程中 mask Gaussians 的 end-to-end 方法**（EAGLES 只是粗暴调整 densification schedule，suboptimal）
2. **唯一扩展到 dynamic scenes 的方法**（其他都只针对 static）
3. **Color 用 neural field 替代 SH**（其他都是 quantize SH 本身）

LightGaussian: https://lightgaussian.github.io/
Compressed 3DGS: https://github.com/graphdeco-inria/compressed-3dgs
Compact3D: https://maincold2.github.io/compact3d/
EAGLES: https://github.com/ExplainableML/EAGLES

---

## 7. Limitations 与 Potential Improvements

虽然 paper 没有专门 limitations section，从阅读中可以推断：

1. **Training time 增加**：33m vs 24m (3DGS*) for Mip-NeRF 360，hash grid 训练比直接优化 SH 慢
2. **LPIPS 略有上升**：mask + R-VQ 引入的 quantization artifact 在 perceptual metric 上有体现
3. **Color neural field 的 hash collision**：大场景下 hash grid 会有 collision，导致高频细节损失
4. **Mask threshold $\epsilon$ 是 hard-coded**：可能不同场景需要不同 threshold，可以做成 learnable 或 adaptive
5. **未与最新 dynamic methods (4DGS [17], 4K4D [73]) 比较**：只跟 STG 比较，而 4DGS 等可能在 quality 上更好

潜在研究方向：
- **Adaptive mask threshold**：用 Gumbel-softmax temperature annealing 替代 fixed threshold
- **Per-Gaussian codebook size**：高细节区域用更大 codebook，平坦区域用更小
- **End-to-end entropy coding**：把 entropy coding 的 bitrate 直接作为 loss，类似 neural compression 文献中的 rate-distortion optimization
- **Replace hash grid with triplane**：triplane 在大场景下 hash collision 更少，且可以 factorize
- **Combined with 4DGS**：用 4DGS 的 hexplane 替代 STG 的 polynomial basis，可能更 compact

---

## 8. 我的 Intuition 总结

让我把这篇 paper 的核心 insights 用一句话总结：

> **3DGS 的冗余主要来自两个 axis：哪里的 Gaussian 应该存在、每个 Gaussian 应该长什么样，前者用 learnable mask 通过 gradient 自动学会，后者用 spatial continuity (neural field) 与 clustering structure (R-VQ) 分别 exploit。**

更深的 intuition：

- 3DGS 的 densification 是 **local greedy** 的（看 gradient 大就 split），它不知道哪些 Gaussian 是 truly redundant。Learnable mask 是 **global optimization** 视角，让每个 Gaussian 自己争抢"贡献度"。
- SH 假设了 per-Gaussian independence，但真实场景的 color 是 spatial smooth 的。Neural field 强迫相邻 Gaussian 共享 representation，这是 inductive bias 的胜利。
- Geometry 的 R-VQ 利用了 clustering structure——大多数 Gaussian 在一个 scene 里只有少数几种"形状模式"。这与 image patch dictionary learning 是同源 idea。

更深一层，这篇 paper 体现了 representation learning 中一个反复出现的 pattern：**explicit representation 之所以 parameter-inefficient，是因为它假设每个 element 是 independent 的；只要引入 spatial / temporal / structural 共享机制，压缩率就能提升一个数量级**。这与 convolution 共享 weights、attention 共享 KV cache 是同一个 insight。

如果你熟悉 VQ-VAE [Van den Oord 2017] 或者 SoundStream [80] 的 R-VQ，paper 的 codebook 训练逻辑其实非常 standard。novelty 在于把 R-VQ 这种 audio compression 的工具应用到 3D Gaussian attributes 上，并且证明在几何领域也工作得很好。

---

## References

- [1] 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- [4] Instant NGP: https://nvlabs.github.io/instant-ngp/
- [19] STG: https://zju3dv.github.io/st-gaussian/
- [20] 本文 CVPR 2024 version: https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Compact_3D_Gaussian_Representation_for_Radiance_Field_CVPR_2024_paper.html
- [21] LightGaussian: https://lightgaussian.github.io/
- [22] Compressed 3DGS: https://graphdeco-inria.github.io/compressed-3dgs/
- [23] Mip-NeRF 360: https://jonbarron.info/mipnerf360/
- [54] Compact3D: https://maincold2.github.io/compact3d/
- [56] EAGLES: https://github.com/ExplainableML/EAGLES
- [78] STE (Bengio 2013): https://arxiv.org/abs/1308.3432
- [79] VQ (Gray 1984): https://ieeexplore.ieee.org/document/1165262
- [80] SoundStream (R-VQ): https://arxiv.org/abs/2107.03312
- [17] 4DGS: https://guanjunwu.github.io/4dgs/
- [73] 4K4D: https://zju3dv.github.io/4K4D/

Project page (本文): https://maincold2.github.io/c3dgs/

如果你对某个具体组件（比如 hash grid 内部如何 forward、R-VQ 与 EMA codebook update 的区别、contract function 的 Jacobian 性质）想要更深的 dive，我可以继续展开。
