---
source_pdf: ET-SEED.pdf
paper_sha256: f449918e0cf081c60abf68b97eb196eecabd9a3863531f0baf09277c0168e06c
processed_at: '2026-08-04T05:11:56-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ET-SEED

好，我用大白话把这篇 paper 重新捋一遍，少摆公式，多讲故事，但关键的技术骨架还是会拎出来。

---

## 1. 这篇 paper 在解决什么"人话版"问题

假设你教机器人开瓶盖。你给它看了 50 段人类示范，瓶子都在桌面正中央竖着。训练完，你把瓶子横过来放到桌角，机器人就懵了——它学到的"动作"是死记硬背的"先到位置 A，再往下伸手"，而不是"跟着瓶子走"。

这就是 imitation learning 最烦的地方：**示范分布一旦被打破，policy 就废**。

那怎么办？朴素办法是 data augmentation：训练时把瓶子随机旋转、随机平移，让网络自己学到"不管瓶子在哪我都该这么抓"。DP3+Aug 就是这么干。但问题在于，你要 augment 到覆盖整个 SE(3)，需要海量数据，而且网络也不一定能从 augment 里真正悟出对称性——它可能只是记住了一堆旋转版本。

**另一个思路**：直接告诉网络"这个任务有空间对称性，输入怎么转，输出就怎么转"。这个性质数学上叫 **SE(3) equivariance**。如果网络架构本身就保证这个性质，那你只要给 20 段示范，网络就能自动泛化到瓶子的任意 6D 位姿。

ET-SEED 走的就是这条路。它的卖点：**用极少 demo，泛化到任意空间位姿，比前人的 equivariant 方法训练快得多，还能搞定复杂长 horizon 任务**。

---

## 2. 前人方案为什么"卡"

在 ET-SEED 之前，有两个代表工作：Equivariant Diffusion Policy（Wang et al.）和 EquiBot（Yang et al.）。它们都做 equivariant diffusion policy，思路是：

> Diffusion 是个 Markov chain，从纯噪声一步步 denoise 到 action。如果我要保证最终输出 equivariant，那就让**每一步 denoise 都 equivariant**。

听起来合理——函数复合的 equivariance 要求每一层都 equivariant。所以它们每一步都用 SE(3) equivariant network（比如 SE(3) Transformer）。

**问题**：训练 equivariant network 比训练 invariant network 难得多。Paper 在 Appendix E 做了个对照实验：三个网络结构几乎一样，只差 input/output 的 feature type。

| Network | 任务 | 训练后 loss |
|---|---|---|
| P1Net（输出 invariant） | 输出一个固定的 identity matrix | 0.0002 |
| P2Net（输出 equivariant） | 输出 input 点云的位姿 | 0.25 |
| P3Net（输出 equivariant，输入也 equivariant） | 同上 | 0.27 |

loss 差了三个数量级。换句话说，让网络学会"输入转一下，输出也跟着转"这件事，比让网络学会"输入转不转，输出都一样"难 1000 倍量级。

所以 EquiBot 在简单任务上还行，遇到 Calligraphy（机器人写字）或者 Fold Garment（叠衣服）这种长 horizon 复杂任务，它就训不动了——每个 denoising step 都得 equivariant，梯度信号太弱。

---

## 3. ET-SEED 的核心 insight：其实只要一步 equivariant 就够

这是 paper 最关键的发现。我用人话讲：

**旧观念**：要保证最终输出 equivariant，每一层都得 equivariant。这是从 function composition 的直觉来的。

**ET-SEED 发现**：在 Markov process 的概率分布意义下，equivariance 是 marginal distribution 的性质，不要求每一步都是 equivariant transition。具体说，只要满足：

1. 初始噪声分布跟 observation 无关（invariant）
2. 前 K-1 步 denoise 是"invariant"的——也就是说，不管 observation 怎么转，中间这些步的输出都不变
3. **最后一步** 是 "equivariant" 的——这一步根据 observation 把 invariant 的中间结果"lift"成 equivariant 输出

那最终 marginal distribution 就还是 equivariant 的。

这就是 Proposition 1。它把 EquiBot 的强假设（每步 equivariant）放宽到"只要一步 equivariant"。

### 直觉类比

想象一个工厂流水线生产瓶子。你想保证最后出货的瓶子不管运到哪里都贴着正确的标签。

- **旧方案**：流水线上每一道工序都得能跟着瓶子位置移动——每一道都要"对位置敏感"
- **ET-SEED 方案**：前 K-1 道工序都在一个固定的"标准工位"上做（瓶子被预先归一化到标准位置），最后一道工序才把成品"搬"到实际位置

前面所有步都在"标准坐标"里干活，不用管瓶子实际在哪——这就便宜了。最后一跳负责把"标准坐标的结果"映射到"实际坐标"。这一跳虽然贵，但只跳一次。

这个思想其实在很多地方都出现过：
- **Canonicalization**：先把输入归一化到 canonical frame，做完操作再 lift 回去
- **物体识别**：在 canonical view 上做 recognition，再根据观察位姿反变换
- **群论里的 Reynolds operator**：把任意函数投影到 invariant 子空间

ET-SEED 本质上是把这个思想塞进了 diffusion 的 Markov chain 里。

---

## 4. 为什么要在 SE(3) manifold 上做 diffusion

这一点 paper 讲得比较隐晦，但其实很重要。

普通 diffusion policy（DP3）把 action 定义在 $\mathbb{R}^d$ 上，比如把 6DoF 位姿拍扁成 6 个数字 $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$，加 Gaussian noise，DDPM 一套流程。

问题：旋转不是 Euclidean 空间。两个旋转矩阵插值不等于对每个元素插值；旋转矩阵加 noise 后不再是合法旋转（不满足 $R^T R = I$）。所以你要么用 Euler angle（有 gimbal lock），要么用 quaternion（单位长度约束），都很别扭。

ET-SEED 直接把 action 定义在 **SE(3) manifold** 上，每个 action 是一个 4×4 矩阵 $\mathbf{H} \in SE(3)$。噪声加在 Lie algebra $\mathfrak{se}(3)$（一个 6D 向量空间）上，然后通过 exponential map $\mathrm{Exp}: \mathfrak{se}(3) \to SE(3)$ 把噪声"卷"回 manifold。

forward process 长这样：

$$
\mathbf{H}^k = \underbrace{\mathrm{Exp}(\gamma\sqrt{1-\bar\alpha_t}\,\varepsilon)}_{\text{Lie algebra 上的噪声}} \cdot \underbrace{\mathcal{F}(\sqrt{\bar\alpha_t}; \mathbf{H}^0, \mathbb{H})}_{\text{SE(3) 上的插值}}
$$

变量说人话：
- $\mathbf{H}^0$ 是 ground truth 位姿
- $\mathbb{H}$ 是 identity matrix（纯噪声的中心）
- $\bar\alpha_t$ 是 DDPM 的 noise schedule，从 1 到 0
- $\varepsilon \in \mathbb{R}^6$ 是 se(3) 上的 Gaussian 噪声
- $\mathcal{F}$ 是 SE(3) 上的 geodesic 插值：当 $\bar\alpha_t \to 1$ 时 $\mathcal{F} \to \mathbf{H}^0$，当 $\bar\alpha_t \to 0$ 时 $\mathcal{F} \to \mathbb{H}$

这样做的好处：
- 噪声样本始终在 SE(3) 上，是合法位姿
- 旋转和平移的"耦合关系"被 group 结构天然尊重
- 对 multimodal 分布（比如瓶子可以从左边抓也可以从右边抓）表示能力更强

类比一下：DP3 是在 $\mathbb{R}^6$ 上做扩散，相当于把球面拍扁到平面再加噪声，样本经常跑出球面；ET-SEED 是在球面（SE(3) manifold）上做扩散，样本始终在球面上。manifold diffusion 在分子生成、pose estimation 等领域已被验证更好，ET-SEED 把它搬到 manipulation。

---

## 5. 网络架构的巧思

ET-SEED 用 SE(3) Transformer（Fuchs et al. 2020）作为基础，但做了两个改造。

### 5.1 两个 backbone

- **$\mathcal{E}_{inv}$**：前 K-1 步用，输出 SE(3) invariant 的中间结果
- **$\mathcal{E}_{equiv}$**：最后一步用，输出 SE(3) equivariant 的最终 action

两个网络结构很像，都是 SE(3) Transformer 加 mean pooling，但 output feature type 不同：
- $\mathcal{E}_{inv}$ 输出 type-0 feature（标量，对 SE(3) 变换不变）
- $\mathcal{E}_{equiv}$ 输出 type-1 feature（向量，对 SE(3) 变换等变）

### 5.2 关键 trick：怎么让 equivariant 网络输出任意位置的平移

SE(3) Transformer 有个限制：它的 type-1 输出只能落在 input 点云的凸包内。但 robot action 可能要伸到点云外面（比如抓桌上的瓶子，手要去到瓶子上方）。

ET-SEED 的解法（Eq. 33）：

$$
t(\mathcal{X}) = \mathcal{M} + R \cdot \text{offset}
$$

- $\mathcal{M}$：输入点云的质心（输入被旋转平移，质心也跟着转）
- $R$：由两个 type-1 feature 经 Gram-Schmidt 得到的旋转矩阵（equivariant）
- $\text{offset}$：三个 type-0 feature（invariant 标量）

组合起来：质心提供"基础位置"（跟着 input 等），offset 提供相对偏移（不变），R 把 offset 旋转到正确朝向。整个 $t$ 既跟着 input 等变，又能伸到点云外面。

人话讲：**用点云质心当锚点，再加一个学到的相对偏移**。这就像你学开车，先定位车在哪（质心），再决定要往哪个方向走多远（offset），整个坐标系跟着车走，但偏移本身是车坐标系下的不变量。

---

## 6. 整个 pipeline 跑一遍

我用一个 inference 流程串起来：

1. **输入**：colored point cloud $O$，加一个随机噪声 action sequence $A^K = [\mathbf{H}_0^K, \mathbf{H}_1^K, \dots, \mathbf{H}_{T_p}^K]$（每个 $\mathbf{H}_i^K$ 是从以 identity 为中心的 SE(3) Gaussian 采样）

2. **前 K-1 步 denoise**（invariant backbone $\mathcal{E}_{inv}$）：
   - 网络输入 $(O, \mathbf{H}_i^k, k, i)$，预测相对变换 $\hat{\mathbf{H}}_i^{k\to 0}$
   - 用 Eq. 8 更新：$\mathbf{H}_i^{k-1} = \mathrm{Exp}(\lambda_0 \log(\hat{\mathbf{H}}_i^{k\to 0} \mathbf{H}_i^k) + \lambda_1 \log(\mathbf{H}_i^k))$
   - 这步的结果跟 $O$ 的位姿无关——不管 $O$ 在哪，$\mathbf{H}_i^{k-1}$ 都一样

3. **最后 1 步 denoise**（equivariant backbone $\mathcal{E}_{equiv}$）：
   - 网络输入 $(O, \mathbf{H}_i^1, 1, i)$，预测 $\hat{\mathbf{H}}_i^{1\to 0}$
   - 直接乘：$\mathbf{H}_i^0 = \hat{\mathbf{H}}_i^{1\to 0} \mathbf{H}_i^1$
   - 这一步 $\hat{\mathbf{H}}_i^{1\to 0}$ 跟 $O$ 等 变，所以最终 $\mathbf{H}_i^0$ 跟 $O$ 等变

4. **输出**：$A^0 = [\mathbf{H}_0^0, \dots, \mathbf{H}_{T_p}^0]$，一条 SE(3) 等变的 trajectory

**为什么这能保证 equivariant**：因为前 K-1 步输出跟 $O$ 无关（invariant），所以中间结果 $\mathbf{H}_i^1$ 在 $O \to TO$ 下不变。最后一步用 equivariant 网络，它接收 $O$ 和 $\mathbf{H}_i^1$，输出会跟 $O$ 等变。乘法组合保持等变性。这就是 Proposition 2。

---

## 7. 实验数字讲讲

### 7.1 Simulation（Table 2）

挑几个有代表性的数字：

**Open Bottle Cap（简单任务）**：
- DP3：训练位姿 76% → 新位姿 14%（崩盘）
- DP3+Aug：训练 44% → 新位姿 46%（augmentation 有点用）
- EquiBot：训练 73% → 新位姿 77%（equivariance 有效）
- ET-SEED：训练 81% → 新位姿 82%（equivariance + 训练容易，两全）

**Calligraphy（长 horizon 复杂任务）**：
- DP3：50% → 3%
- DP3+Aug：21% → 0%
- EquiBot：43% → 40%（训不太动）
- ET-SEED：55% → 54%（训练效率优势明显）

观察：**简单任务上 EquiBot 和 ET-SEED 接近，但任务越复杂 ET-SEED 优势越大**。这印证了核心论点：每步 equivariant 的训练 cost 在长 horizon 上爆掉，而 ET-SEED 只需一步 equivariant，训练轻松。

### 7.2 Geodesic Distance（Table 3）

这个指标测 trajectory 的几何质量。公式：

$$
\mathcal{D}_{geo}(\mathbf{T}, \hat{\mathbf{T}}) = \sqrt{\|\log(\mathbf{R}^\top \hat{\mathbf{R}})\|^2 + \|\hat{\mathbf{t}} - \mathbf{t}\|^2}
$$

人话：旋转部分的测地距离 + 平移部分的欧式距离。

Calligraphy NP 场景：
- DP3：4.662（完全跑偏）
- ET-SEED：0.089（几乎贴合）

差了 50 倍。这意味着 ET-SEED 不只是"成功率"高，它生成的整个 trajectory 在几何上都是对的——这对你后面要执行精细任务（写字、叠衣服）非常重要。

### 7.3 Real-world（Table 4）

只给 20 段 demo：

| 方法 | Open Bottle Cap | Open Door | Calligraphy | Fold Garment |
|---|---|---|---|---|
| DP3 | 0.2 | 0.2 | 0.0 | 0.1 |
| EquiBot | 0.6 | 0.5 | 0.0 | 0.3 |
| **ET-SEED** | **0.8** | **0.6** | **0.4** | **0.6** |

DP3 和 EquiBot 在写字任务上 0% 成功，ET-SEED 40%。20 段 demo 学会写字，我觉得这个本身就很说明问题。

### 7.4 Ablation（Table 5）

去掉 SE(3) equivariant backbone（换成 PointNet++）：成功率从 76% 掉到 24%。
去掉 equivariant diffusion（用普通 DDIM）：从 76% 掉到 57%。

两个 component 都重要，但 equivariant backbone 更关键。这告诉我：**equivariance 的 inductive bias 比 diffusion framework 本身更 fundamental**。Diffusion 是个不错的 generative model，但真正起作用的是 equivariance 给的 sample efficiency。

---

## 8. 这篇 paper 让我想到的更深的东西

### 8.1 "Inductive bias 分摊"思想

ET-SEED 的核心思想其实是一种 **asymmetric computation allocation**：

> 大部分 computation 在 cheap space（invariant）做，关键的 expensive operation（equivariant lift）只做一次。

这种思想在别的地方也出现：
- **MoE**：大部分 token 走 cheap expert，少数走 expensive expert
- **Mamba / SSM**：linear scan 是 cheap 的，selective gate 是 expensive 的，但 gate 只控制信息流
- **Sparse Transformer**：大部分 attention 被稀疏化，只在关键位置做 full attention
- **Deep equilibrium model**：大部分 layer 是"迭代求解器"，真正的 expressive power 在 fixed point

ET-SEED 在 diffusion policy 上实现了类似的设计——**K-1 步 invariant reasoning + 1 步 equivariant lift**。这种"asymmetric inductive bias"可能是个更普适的架构设计原则。

### 8.2 Equivariance 不一定要 per-layer

经典深度学习 wisdom：要保证网络 equivariant，每层都得 equivariant。这是从 group representation theory 的 closure property 来的。

ET-SEED 提供了另一个角度：在 Markov process / distribution 层面，equivariance 是 marginal 的性质，不是 pointwise 的性质。所以可以在"分布层面"做 equivariance，在"每步层面"做 invariant。这有点像：

- **Conservation law**：整体守恒不要求局部每点守恒，只要边界 flux 正确
- **Noether theorem**：symmetry 对应 conservation，但 conservation 是 global 性质
- **Renormalization**：物理量在不同 scale 下行为不同，但整体 RG flow 保持对称性

这种"全局性质 vs 局部操作"的区分，在 ML 架构设计里可能还有很多空间可挖。

### 8.3 与 "canonicalization" 的关系

ET-SEED 的前 K-1 步本质上是在 **canonical frame** 里做 reasoning。invariant feature 就是 canonical frame 下的 representation。最后一步做 frame transformation，把 canonical frame 下的结果 lift 到 actual frame。

这跟 recent works 比如 Octo、π0 的思路有点像——先学一个 task-agnostic 的 canonical representation，再做 task-specific 的 head。但 ET-SEED 是在 SE(3) 空间里做这件事，更几何化。

### 8.4 我觉得这篇 paper 局限在哪

1. **Partial equivariance 没理论保证**：Fold Garment 不是严格 SE(3) equivariant（衣服每次变形不同）。Paper 说"还是 work"，但理论上 Proposition 1 的假设被违反。实际 work 是靠 invariant backbone 学到了 robust feature，但没有理论 guarantee。

2. **K-1 步 invariant 可能丢信息**：中间所有步都在 quotient space 里 reason，fine-grained spatial 信息被压扁。对于精细 manipulation（手术、装配），可能需要更多 equivariant step。Proposition 1 已经支持这种 trade-off（取 $n > 2$），但 paper 没探索。

3. **依赖 SAM2 做 point cloud segmentation**：Real-world 实验里 SAM2 是个 silent bottleneck。如果 segmentation 失败，整个 pipeline 崩。未来应该 explore 端到端学习或者更 robust 的 perception。

4. **Scale 不变**：用 SE(3) 不含 scale。不同尺寸的同类物体（大瓶子 vs 小瓶子）可能不 work。扩展到 SIM(3) 或 affine group 是个方向。

5. **Dual-arm equivariance 结构简化**：Fling Garment 是双臂任务，paper 把两个 end-effector 的 trajectory concat 起来。但严格说双臂的 equivariance 结构更复杂（每个 arm 有自己的 frame），这个 simplification 可能限制 generalization。

---

## 9. 谁应该读这篇 paper

- **做 robot learning 的人**：这是 equivariant diffusion policy 的 SOTA，值得直接拿来用或 compare
- **做 equivariant ML 的人**：Proposition 1 是真 theoretical contribution，可以推广到其他 Markov process 类生成模型
- **做 diffusion model 的人**：manifold diffusion + equivariance 的组合是个有意思的 design pattern
- **做 representation learning 的人**：invariant + equivariant 双 backbone 的"分摊"思想可以 inspire 其他架构

---

## 10. 关键链接

- **项目主页**: https://et-seed.github.io/
- **Diffusion Policy (Chi et al., 2023)**: https://arxiv.org/abs/2303.04137
- **EquiBot (Yang et al., 2024)**: https://arxiv.org/abs/2407.01479
- **SE(3) Transformer (Fuchs et al., 2020)**: https://arxiv.org/abs/2006.10503
- **GeoDiff (Xu et al., 2022)**: https://arxiv.org/abs/2203.02923
- **SE(3)-Diffusion Fields (Urain et al., 2023)**: https://arxiv.org/abs/2209.10640
- **SE(3) Diffusion for pose estimation (Jiang et al., 2024)**: https://arxiv.org/abs/2403.19460
- **DP3 (Ze et al., 2024)**: https://arxiv.org/abs/2403.03954
- **DDPM (Ho et al., 2020)**: https://arxiv.org/abs/2006.11239
- **Vector Neurons (Deng et al., 2021)**: https://arxiv.org/abs/2104.12229
- **Equiformer (Liao & Smidt, 2022)**: https://arxiv.org/abs/2206.11990
- **SAM2 (Ravi et al., 2024)**: https://arxiv.org/abs/2408.00714

---

## 一句话总结

ET-SEED 证明了一个反直觉的事情：**要保证 diffusion policy 的输出在空间上等变，不用让每一步都等变，只要最后一步等变就够了**。这把训练成本降了一个量级，同时保留了等变带来的 sample efficiency 和空间泛化。20 段 demo 就能让机器人学会在新位姿下开瓶盖、开门、写字、叠衣服。

---

# ET-SEED: Efficient Trajectory-Level SE(3) Equivariant Diffusion Policy 深度解析

Andrej，这篇 paper 在我看来是把 equivariant diffusion policy 从"理论上正确但训练困难"推进到"实用且 sample efficient"的关键一步。让我把里面的技术细节、数学结构和 intuition 拆开来讲。

---

## 1. 核心问题与 motivation

Imitation learning 在 robot manipulation 里的核心痛点：**demonstration 分布一旦被 spatial perturbation 突破，policy 就崩了**。比如训练时 bottle 总在桌面中央，测试时 bottle 旋转 90° 放到桌角，DP3 这种 Euclidean diffusion policy 直接失败（Table 2 里 NP 场景 success rate 从 76 掉到 14）。

Symmetry 的物理直觉：如果任务结构在 SE(3) 变换下保持不变（比如开瓶盖的动作轨迹会随瓶子的位姿等变地"贴"过去），那么把这种 inductive bias 显式编码进网络，等效于把 observation space 折叠成 quotient space $\mathcal{O}/SE(3)$，每个等价类只需一个 sample。

但前人工作（EquiBot [14], Equivariant Diffusion Policy [13]）的做法是：**让 diffusion 的每一步 Markov transition 都 equivariant**。这导致每个 denoising step 都得用 SE(3)-equivariant network（如 SE(3) Transformer），训练 cost 高、收敛慢、长 horizon 任务表现差。

ET-SEED 的关键 insight：**equivariance 不需要 per-step，只需要 "至少一步" equivariant 就够了**。这是一个相当 surprising 的理论结果。

---

## 2. Equivariant Markov Process 的理论扩展（核心 contribution）

### 2.1 三类 Markov transition 的形式化

Paper 定义了三类 transition kernel（Eq. 1），变量含义：
- $x^k$: 第 $k$ 步的 noisy latent（这里是 SE(3) 元素）
- $c$: condition（observation $O$）
- $T \in SE(3)$: 任意 rigid transformation
- $p_1, p_2, p_3$: 三种 transition 密度

$$
p_1(x^{k-1} | x^k, c) = p_1(x^{k-1} | x^k, Tc) \quad \text{(condition-invariant)}
$$
$$
p_2(x^{k-1} | x^k, c) = p_2(Tx^{k-1} | x^k, Tc) \quad \text{(output-equivariant, input-invariant)}
$$
$$
p_3(x^{k-1} | x^k, c) = p_3(Tx^{k-1} | Tx^k, Tc) \quad \text{(fully equivariant)}
$$

**Intuition 解读**：
- $p_1$：condition 怎么变都不影响 transition —— 输出对 $c$ "无感"
- $p_2$：latent $x^k$ 不动，但当 $c$ 被 $T$ 变换，输出 $x^{k-1}$ 也被 $T$ 变换 —— 这是"半 equivariant"
- $p_3$：完全 equivariant，前人 EquiBot 假设的全是这个

### 2.2 Proposition 1（关键定理）

如果 Markov chain $x^{K:0}$ 满足：
- 初始分布 invariant：$p(x^K | c) = p(x^K | Tc)$
- 前 $K-n+1$ 步是 $p_1$ 类型
- 中间 1 步是 $p_2$ 类型  
- 后 $n-2$ 步是 $p_3$ 类型

则 marginal：$p(x^0 | c) = p(Tx^0 | Tc)$（equivariant）。

ET-SEED 取 $n=2$：$K-1$ 步 $p_1$（invariant）+ 1 步 $p_2$（equivariant）+ 0 步 $p_3$。

**为什么这是大事**：原来需要 $K$ 个 equivariant step，现在只需要 1 个。Appendix E 的实验（Fig. 4）显示 P1Net 几个 gradient step 就收敛到 loss 0.0002，而 P2Net/P3Net 训练到 loss 0.25 还卡着 —— invariant task 比 equivariant task 学习难度低 1000 倍量级。

### 2.3 我的理解：为什么这个理论成立

回头看 proof（Appendix C 的 Eq. 24）。本质上：
1. 前 $K-1$ 步 $p_1$-like 把 $x^{K}$ 逐步 refine 成 $x^1$，且这个 $x^1$ 与 $c$ 无关（invariant）
2. 最后一步 $p_2$ 把 "invariant latent" 转成 "equivariant output" —— 这一跳承担全部 equivariance 责任

这就像一个信息瓶颈：前面所有步都在 quotient space $SE(3)\backslash \mathcal{X}$ 里 reasoning（invariant feature），最后一跳把 invariant feature "lift" 回 SE(3) 并 attach 到 condition 的 frame 上。这种设计思想和 "canonicalization" 很像 —— 先归一化到一个 canonical frame，再做最后一步 equivariant mapping。

---

## 3. SE(3) Manifold 上的 Diffusion Process

### 3.1 Forward process（Eq. 4）

$$
\mathbf{H}^k = \underbrace{\mathrm{Exp}(\gamma\sqrt{1-\bar{\alpha}_t}\,\varepsilon)}_{\text{Perturbation}} \cdot \underbrace{\mathcal{F}(\sqrt{\bar{\alpha}_t}; \mathbf{H}^0, \mathbb{H})}_{\text{Interpolation}}
$$

变量解释：
- $\mathbf{H}^k \in SE(3)$: 第 $k$ 步 noisy action（4×4 matrix）
- $\mathbf{H}^0 \in SE(3)$: ground truth action
- $\mathbb{H} \in SE(3)$: identity transformation（noise 中心）
- $\varepsilon \in \mathbb{R}^6$: se(3) Lie algebra 上的 Gaussian noise
- $\bar{\alpha}_t \in [0,1]$: noise schedule（DDPM 的标准 schedule）
- $\gamma$: noise scale hyperparameter
- $\mathrm{Exp}: \mathfrak{se}(3) \to SE(3)$: exponential map
- $\mathcal{F}$: SE(3) manifold 上的 geodesic interpolation

**Interpolation 函数**（Eq. 36）：
$$
\mathcal{F}(\sqrt{\bar{\alpha}_t}; \mathbf{H}^0, \mathbb{H}) = \mathrm{Exp}\left((1-\sqrt{\bar{\alpha}_t}) \cdot \log(\mathbb{H}\mathbf{H}^{0,-1})\right) \mathbf{H}^0
$$

这相当于：在 Lie algebra $\mathfrak{se}(3)$ 上做线性插值，再 $\mathrm{Exp}$ 回 SE(3)。

**Intuition**：当 $\bar{\alpha}_t \to 1$（小噪声），$\mathcal{F} \to \mathbf{H}^0$；当 $\bar{\alpha}_t \to 0$（大噪声），$\mathcal{F} \to \mathbb{H}$（identity）。整个 forward process 把任意 action 逐步扰动到以 identity 为中心的 SE(3) Gaussian。

### 3.2 与 DDPM 的类比（Eq. 5）

$$
x_t = \bar{\alpha}_t x_0 + \bar{\beta}_t \varepsilon
$$

DDPM 在 Euclidean 上线性混合 signal 和 noise；ET-SEED 在 SE(3) manifold 上用 group multiplication 替代加法，用 Exp/Log 替代线性映射。这保证 noisy sample 始终在 SE(3) 上，不会跑出 manifold。

为什么 SE(3) manifold 比 Euclidean 好处理 6DoF action？因为 SE(3) 是 non-commutative group，Euclidean 上的 Gaussian 会把旋转和平移混淆，而 manifold formulation 自然尊重 group 结构（Urain et al. [15] 的 SE(3)-Diffusion Fields 已经验证这点）。

### 3.3 Reverse denoising step（Eq. 8 与 Eq. 9）

前 $K-1$ 步（invariant）：
$$
\mathbf{H}_i^{k-1} = \mathrm{Exp}\left(\lambda_0 \log(\hat{\mathbf{H}}_i^{k\to 0} \mathbf{H}_i^k) + \lambda_1 \log(\mathbf{H}_i^k)\right)
$$

变量：
- $\hat{\mathbf{H}}_i^{k\to 0}$: 网络预测的从 $k$ 到 0 的相对变换
- $\lambda_0, \lambda_1$: scheduler 系数，控制 prediction 和 current state 的混合
- $\log: SE(3) \to \mathfrak{se}(3)$

最后 1 步（equivariant）：
$$
\mathbf{H}_i^0 = \hat{\mathbf{H}}_i^{1\to 0} \mathbf{H}_i^k
$$

直接乘上 predicted transformation，不再插值。这一步是纯 group operation，自然保持 equivariance。

---

## 4. Backbone 设计（Appendix F）

ET-SEED 用两个 SE(3) Transformer 衍生的 backbone：$\mathcal{E}_{inv}$ 和 $\mathcal{E}_{equiv}$。

### 4.1 Invariant module $\mathcal{E}_{inv}$

- **Rotation 分支**：输出 6 个 type-0 features（标量），通过 Gram-Schmidt 正交化得到 rotation matrix $R$。type-0 在 SE(3) 下不变，所以 $R$ 不变。
- **Translation 分支**：输出 3 个 type-0 features，直接是平移 $t$。
- 组合成 $\begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix}$，整体对 SE(3) input 不变。

### 4.2 Equivariant module $\mathcal{E}_{equiv}$（Eq. 33-34）

这里有个精巧的设计。SE(3) Transformer 的 type-1 feature 是 vector feature，equivariant 但 bounded 在 input point 的凸包内 —— 不能直接输出任意位置的 translation。

ET-SEED 的 trick（Eq. 33）：
$$
t(\mathcal{X}) = \mathcal{M} + R \cdot \text{offset}
$$

其中：
- $\mathcal{M} = \frac{1}{N}\sum_i x_i$: input point cloud 的质心
- $R$: 由 2 个 type-1 features Gram-Schmidt 得到的 rotation matrix
- $\text{offset}$: 3 个 type-0 features（标量）

**Equivariance 验证**（Eq. 34）：input 被 $T = (R_{data}, t_{data})$ 变换后：
$$
t'(\mathcal{X}) = (R_{data}\mathcal{M} + t_{data}) + R_{data} R \cdot \text{offset} = R_{data}(\mathcal{M} + R \cdot \text{offset}) + t_{data} = R_{data} t(\mathcal{X}) + t_{data}
$$

完全 equivariant。这个 trick 把"质心 anchor + equivariant offset"组合，绕开了 SE(3) Transformer 只能输出凸包内 vector 的限制。我觉得这是一个相当 elegant 的工程实现，类似 NeuS 里把 SDF 表达成"base + residual"的思路。

### 4.3 网络选择方程（Eq. 7）

$$
s_\theta(O, \mathbf{H}_i^k; k, i) = 
\begin{cases} 
\mathcal{E}_{inv}(O, \mathbf{H}_i^k; k, i), & k > 1 \\
\mathcal{E}_{equiv}(O, \mathbf{H}_i^k; k, i), & k = 1
\end{cases}
$$

这里 $k$ 是 diffusion step index，$i$ 是 trajectory horizon index。前 $K-1$ 步用 invariant backbone，最后一步切换到 equivariant backbone。**两个 backbone 不共享权重**，因为是不同的 mapping。

---

## 5. End-to-End Equivariance Proof（Proposition 2）

我重新整理一下 Appendix D 的逻辑链：

1. **初始**：$A^K \sim \mathcal{N}_{SE(3)}(\mathbb{H}, \Sigma)$，与 $O$ 独立，所以 $p(\mathbf{H}_i^K | O) = p(\mathbf{H}_i^K | TO)$（Eq. 31）—— invariant
2. **前 $K-1$ 步**：因为 $\mathcal{E}_{inv}$ 输出 invariant，所以 $\hat{\mathbf{H}}_i^{k\to 0}$ 不随 $O \to TO$ 改变（Eq. 27）；带入 Eq. 8 的 update rule，$\mathbf{H}_i^{k-1}$ 也不变；归纳得 $\mathbf{H}_i^1$ invariant（Eq. 28）—— $p_1$-like
3. **最后一步**：$\mathcal{E}_{equiv}$ 输出 equivariant，$T\hat{\mathbf{H}}_i^{1\to 0} = \mathcal{E}_{equiv}(TO, \mathbf{H}_i^1; 1, i)$（Eq. 29）；带入 Eq. 9 直接乘法，$T\mathbf{H}_i^0 = T\hat{\mathbf{H}}_i^{1\to 0} \mathbf{H}_i^1$（Eq. 30）—— $p_2$-like
4. **组合**：用 Proposition 1（$n=2$ 情形），得 $p(A^0 | O) = p(TA^0 | TO)$

整个 trajectory $A^0 = \bigcup_i \mathbf{H}_i^0$ 在 SE(3) 下 equivariant。

---

## 6. 实验数据深度分析

### 6.1 Simulation 结果（Table 2）

| 任务 | 方法 | T(50 demos) | NP(50 demos) | 性能保持率 |
|---|---|---|---|---|
| Open Bottle Cap | DP3 | 76±5.5 | 14±6.5 | 18% |
| Open Bottle Cap | EquiBot | 73±2.74 | 77±7.58 | 105% |
| Open Bottle Cap | **ET-SEED** | **81±2.24** | **82±2.74** | **101%** |
| Calligraphy | EquiBot | 43±8.37 | 40±10.61 | 93% |
| Calligraphy | **ET-SEED** | **55±3.54** | **54±8.22** | **98%** |
| Fold Garment | EquiBot | 58±2.74 | 60±7.90 | 103% |
| Fold Garment | **ET-SEED** | **67±2.74** | **69±4.18** | **103%** |

关键观察：
- DP3 在 T 上表现不错，但 NP 场景 success rate 暴跌到 0-15% —— 完全没有 spatial generalization
- DP3+Aug 通过数据增强（三轴 0°-90° 旋转 + 10% workspace Gaussian 噪声）部分缓解，但仍然显著低于 equivariant 方法
- EquiBot 在简单任务（Open Bottle Cap, Open Door）上和 ET-SEED 接近，但在长 horizon 复杂任务（Calligraphy, Fold/Fling Garment）上明显落后 —— 因为它要求每步 equivariant，训练困难
- **ET-SEED 在 NP 场景下性能几乎不下降**，证明 equivariance 的理论保证转化为实际 generalization

### 6.2 Geodesic Distance（Table 3）

$$
\mathcal{D}_{geo}(\mathbf{T}, \hat{\mathbf{T}}) = \sqrt{\|\log(\mathbf{R}^\top \hat{\mathbf{R}})\|^2 + \|\hat{\mathbf{t}} - \mathbf{t}\|^2}
$$

变量：$\mathbf{R}, \mathbf{t}$ 是 ground truth 旋转和平移；$\hat{\mathbf{R}}, \hat{\mathbf{t}}$ 是预测。这项度量整个 trajectory 的几何质量。

在 Calligraphy NP 场景：DP3 是 4.662，ET-SEED 是 0.089 —— **52 倍** 差距。DP3 完全无法 extrapolate 到新位姿，而 ET-SEED 的轨迹几何上几乎贴合 ground truth。

### 6.3 Real-world（Table 4）

只给 20 demonstrations：

| Method | Open Bottle Cap | Open Door | Calligraphy | Fold Garment |
|---|---|---|---|---|
| DP3 | 0.2 | 0.2 | 0.0 | 0.1 |
| EquiBot | 0.6 | 0.5 | 0.0 | 0.3 |
| **ET-SEED** | **0.8** | **0.6** | **0.4** | **0.6** |

Calligraphy 上 DP3/EquiBot 都 0% 成功，ET-SEED 40% —— 在 20 demos 下能学会写汉字，这件事本身就挺惊人的。

### 6.4 Ablation（Table 5）

- Ours w/o SE(3)（用 PointNet++ 替代 equivariant backbone）：24%
- Ours w/o Eqv-Diff（用 DDIM 不做 equivariant diffusion）：57%
- Full ET-SEED：76%

两个 component 都重要，但 SE(3) equivariant backbone 更关键（去掉直接掉 52 个百分点）。这说明 equivariance 的 inductive bias 比 diffusion framework 本身更 fundamental。

---

## 7. 与相关工作的 positioning

### 7.1 与 EquiBot [14] 的对比

EquiBot 用 SIM(3)（包含 scale），并要求每步 equivariant。ET-SEED 用 SE(3)（不含 scale，因为 manipulation task 的 scale 通常固定），且只要求 1 步 equivariant。理论上 ET-SEED 是 EquiBot 的严格 superset：Proposition 1 取 $n=K$ 退化到 EquiBot 假设。

### 7.2 与 Equivariant Diffusion Policy [13] 的对比

Wang et al. 做 SO(2) equivariance（桌面绕 z 轴旋转对称），ET-SEED 做完整 SE(3)。SO(2) 是 SE(3) 的 subgroup，所以 ET-SEED 更 general。

### 7.3 与 GeoDiff [28] 的对比

GeoDiff 在分子构象生成里证明 SE(3) equivariant Markov process，ET-SEED 把这个理论扩展到 condition $c$ 存在的 setting，并放宽到 Proposition 1。

### 7.4 与 SE(3)-Diffusion Fields [15] / Jiang et al. [16] 的对比

这两个工作定义 SE(3) manifold diffusion 但没有 equivariance；ET-SEED 把 manifold diffusion 和 equivariance 结合，得到双重 benefit。

---

## 8. 我看到的 limitations 和未来方向

1. **Partial equivariance 的处理**：Fold/Fling Garment 任务并不是严格 SE(3) equivariant（衣服每次变形不同），paper 说"即便如此也 work"，但理论上这违反了 Proposition 1 的假设。实际 work 是因为 invariant backbone 学到了 robust geometric feature，但这个 gap 没有理论 guarantee。未来工作可以探索 approximate equivariance 或 learned symmetry group。

2. **K-1 步 invariant 的信息损失**：前面所有步都在 quotient space 里 reason，意味着 fine-grained spatial information 在中间步骤被"压扁"。对于需要精细 spatial reasoning 的任务（比如手术机器人），可能需要更多 equivariant step，即 $n > 2$。Proposition 1 已经支持这种 trade-off。

3. **Dual-arm 协作的 equivariance**：Fling Garment 是双臂任务，paper 把两个 end-effector 的 trajectory concat 起来处理。但严格说双臂的 equivariance 结构更复杂（两个 arm 可能有自己的 frame），可能需要更细致的 group representation。

4. **Scale equivariance**：用 SE(3) 而非 SIM(3) 意味着对 scale 变化不 robust。对于 deformable object 或不同尺寸的同类 object，可能需要扩展到 SIM(3) 或更一般的 affine group。

5. **Point cloud 输入的依赖**：需要 SAM2 做分割投影到 point cloud。这部分在 real-world 实验里是个 silent bottleneck —— 如果 segmentation 失败整个 pipeline 崩。未来可以探索 end-to-end 学习或更 robust 的 perception。

---

## 9. 更深层的 intuition（build your intuition 的部分）

我觉得这篇 paper 最 deep 的 insight 是关于 **"equivariance 何时必要"** 的重新思考。

经典视角：要保证输出 equivariant，整个 pipeline 每一步都要 equivariant（function composition 的 equivariance closure）。

ET-SEED 视角：在 Markov process 框架里，equivariance 是关于 marginal distribution 的性质，不是关于每一步的 pointwise 性质。可以"积累 invariant reasoning，最后一跳 equivariant lift"。

这个思想其实在其他领域也有 echo：
- **Physics**：在 conservation law 框架下，整体守恒不要求局部每点守恒，只要边界 flux 正确
- **Neural ODE**：连续动力学可以离散化为任意 steps，只要整体 flow map 满足性质
- **Normalizing flow**：coupling layer 的 invertibility 不要求每层 bijective，只要组合 bijective
- **Invariant theory**：Reynolds operator 把任意函数投影到 invariant subs间，是一个"最后一步 lift"的思想

这种 "**大部分 cheap computation + 关键一步 expensive 但 expressive operation**" 的设计哲学，在 deep learning architecture design 里会越来越重要。类似的例子有：
- Mixture of Experts：大部分 token 走 cheap expert
- Mamba / SSM：linear scan + selective gate
- Sparse transformer：大部分 attention 被稀疏化

ET-SEED 在 diffusion policy 里实现了类似的"computation allocation asymmetry"。

---

## 10. 参考链接

- **项目主页**: https://et-seed.github.io/  
- **Diffusion Policy (Chi et al.)**: https://arxiv.org/abs/2303.04137  
- **EquiBot (Yang et al.)**: https://arxiv.org/abs/2407.01479  
- **Equivariant Diffusion Policy (Wang et al.)**: https://arxiv.org/abs/2407.01479 (相同 ID 可能 conflict，建议搜 "Equivariant Diffusion Policy Wang Platt")  
- **SE(3) Transformer (Fuchs et al.)**: https://arxiv.org/abs/2006.10503  
- **GeoDiff (Xu et al.)**: https://arxiv.org/abs/2203.02923  
- **SE(3)-Diffusion Fields (Urain et al.)**: https://arxiv.org/abs/2209.10640  
- **SE(3) Diffusion for pose estimation (Jiang et al.)**: https://arxiv.org/abs/2403.19460  
- **DP3 (Ze et al.)**: https://arxiv.org/abs/2403.03954  
- **SAM2 (Ravi et al.)**: https://arxiv.org/abs/2408.00714  
- **DDPM (Ho et al.)**: https://arxiv.org/abs/2006.11239  
- **Vector Neurons (Deng et al.)**: https://arxiv.org/abs/2104.12229  
- **Equiformer (Liao & Smidt)**: https://arxiv.org/abs/2206.11990  
- **RiemannMan (Gao et al.)**: https://arxiv.org/abs/2403.19460  

---

## 11. 总结

ET-SEED 的核心贡献是三层叠加：
1. **理论层**：Proposition 1 把 equivariant Markov process 的条件从 "每步 equivariant" 放宽到 "至少一步 equivariant"，这是一个真正的数学 insight
2. **算法层**：把这个理论落到 SE(3) manifold diffusion 上，结合 SE(3) Transformer 的 modified backbone（invariant + equivariant 双 backbone）
3. **系统层**：在 6 个仿真任务 + 4 个 real-world 任务上验证 sample efficiency 和 spatial generalization，只 20 demos 就能 work

我觉得这个工作对 robot learning 的更大意义是：**inductive bias 的形式可以比传统理解更灵活**。Equivariance 不是 all-or-nothing，而是可以"分摊"在 computation graph 的不同位置。这个 insight 可能会启发更多 "asymmetric inductive bias" 设计，比如 temporal equivariance、causal equivariance、hierarchical equivariance 等等。

如果你（Andrej）想进一步思考这个方向，我建议关注三个 angle：(1) Proposition 1 是否能推广到非 Markov 的 diffusion（如 flow matching）；(2) 当 task 不是严格 equivariant 时，能否学一个 "soft" equivariance trade-off；(3) 这种 "K-1 invariant + 1 equivariant" 的思想能否用到其他 generative model family（VAE, autoregressive, MoE）。
