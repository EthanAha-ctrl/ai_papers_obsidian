---
source_pdf: Dynamic Rank Adjustment in Diffusion Policies for.pdf
paper_sha256: d4f0bee611571d94071e0c3243cb2bd4172a0ebcca9aab54a40ec6c31b0307ee
processed_at: '2026-08-04T00:38:03-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 DRIFT

## 一句话版本

Diffusion policy 太大太慢，想 online 跟人互动训练根本等不起。DRIFT 的办法是：用 SVD 把 weight matrix 切成 "重要的" 和 "不重要的" 两块，训练早期全开，后期把不重要的冻住只更新重要的，训练就快了。

---

## 为什么这事儿重要

先说 background。Diffusion policy 现在在 robot manipulation 里很火，Chi et al. 2023 那篇 paper（https://arxiv.org/abs/2303.04137）用 DDPM 做 visuomotor control，效果很好。问题是这玩意儿参数量大——tens of millions 级别。

参数大本身不是问题，问题出在 **online interactive imitation learning** 这种场景。

什么叫 online interactive IL？就是 DAgger 那一套（https://arxiv.org/abs/1011.0686）：robot 自己跑，快出问题的时候 human expert 接管纠正，收集 corrective data，马上 update policy，再跑。这个 loop 的节奏很关键——human 在旁边等着呢，你 update 一下要 10 分钟，人早跑了。

以前 DAgger 用 small MLP 或 LSTM，parameter 几十万，batch update 秒级完成，互动很丝滑。换成 diffusion policy 后 batch update 变分钟级，这个 loop 就废了。

LoRA 本来能救场——只训 small adapter 嘛。但 LoRA 是给 pre-trained model fine-tuning 用的，robotics 里没 foundation model，每个 task 都 train from scratch。你拿 LoRA 从头训，backbone 本身没学好，low-rank adapter 表达力不够，效果差。而且动态调 rank 还要反复 merge + re-inject 新 adapter，把训练搞崩。

DRIFT 就是来解决这个尴尬的。

---

## 核心思想：训练过程中 rank 该怎么变

Intuition 很朴素：

**训练早期**：network 还在学 general behavior pattern，weight matrix 各个方向都在变，需要 full rank 来探索。这时候砍 rank 就等于把 baby 的腿绑住让它学走路。

**训练后期**：principal directions 已经稳定了，weight 的更新主要集中在少数几个 dominant singular directions 上，剩下的方向贡献很小。这时候还全 rank 训练就是浪费。

这其实跟 ML community 的 intrinsic dimensionality 发现一致（https://arxiv.org/abs/2012.13255）——overparameterized model 实际只在低维子空间里学习。只不过 DRIFT 说：这个低维子空间是 **训出来** 的，不是一开始就在的，所以要动态调。

---

## 怎么切 weight matrix：Rank Modulation

这是 paper 最核心的技术 contribution。讲细一点。

### LoRA 怎么做的

给定 conv layer weight $W_{conv} \in \mathbb{R}^{C_{out} \times C_{in} \times k}$，LoRA 不动 $W_{conv}$，新加两个小矩阵：

$$
\text{Conv}_{LoRA}(x) = W_{conv} \circledast x + \alpha \left( (W_{up} \times W_{down}) \circledast x \right)
$$

变量含义：
- $W_{down} \in \mathbb{R}^{r \times C_{in} \times k}$：把 input 降维到 rank-$r$
- $W_{up} \in \mathbb{R}^{C_{out} \times r \times k}$：再升维回 $C_{out}$
- $\alpha$：scaling factor
- $r \ll C_{in}$：bottleneck
- $\circledast$：卷积

只有 $W_{up}, W_{down}$ 接 gradient。Forward pass 多了一次卷积，cost 是 $O(C_{out} \cdot C_{in} \cdot r \cdot k)$，跟 $r$ 线性增长。

Fine-tuning 时 $r$ 很小（4 左右），多出来的 cost 可以忽略。但 from-scratch 训练需要 $r \sim 256$，这个 cost 就显著了。更糟的是 dynamic 调 rank 时要 re-inject 新初始化的 adapter，gradient 又得重新 align，训练就崩了（paper Table IV 的 D(LR) 行 success rate 掉到 0.58）。

### DRIFT 怎么做的

不用 LoRA 那套加 adapter，直接对 $W$ 做 SVD：

$$
W = U \Sigma V^T
$$

变量：
- $U \in \mathbb{R}^{m \times m}$：left singular vectors，orthonormal
- $\Sigma \in \mathbb{R}^{m \times n}$：对角矩阵，singular values $\sigma_1 \geq \sigma_2 \geq \dots$ 降序排
- $V \in \mathbb{R}^{n \times n}$：right singular vectors，orthonormal
- $m = C_{out} \cdot k$，$n = C_{in}$（reshape 后）

在 rank $r$ 处切：

$$
U = [U_{train} \mid U_{frozen}], \quad
\Sigma = \begin{bmatrix} \Sigma_{train} & 0 \\ 0 & \Sigma_{frozen} \end{bmatrix}, \quad
V = [V_{train} \mid V_{frozen}]
$$

变量：
- $U_{train} \in \mathbb{R}^{m \times r}$：前 $r$ 个 left singular vectors
- $\Sigma_{train} \in \mathbb{R}^{r \times r}$：top-$r$ singular values（**能量最大的**）
- $V_{train} \in \mathbb{R}^{n \times r}$：前 $r$ 个 right singular vectors
- 带下标 frozen 的是剩下的，冻住

然后：

$$
W_{train} = U_{train} \Sigma_{train} V_{train}^T
$$
$$
W_{frozen} = U_{frozen} \Sigma_{frozen} V_{frozen}^T
$$

Forward 时把两部分加回去 reshape 回 conv tensor：

$$
W_{conv} = \text{reshape}(W_{train} + W_{frozen}, (C_{out}, C_{in}, k))
$$

**Forward 还是单次卷积，cost 跟原 conv 一样**。Backward 只对 $U_{train}, \Sigma_{train}, V_{train}$ 算 gradient，参数量从 $m \times n$ 降到 $r \times (m + n + 1)$。

### 为什么这样切比 LoRA 好

几个关键 intuition：

**1. Top-$r$ subspace 是数据驱动的，不是随机的**

LoRA 的 $W_{up}, W_{down}$ 随机初始化，trainable subspace 是 random 的，要靠训练慢慢 align 到 principal directions。DRIFT 的 trainable subspace 直接就是当前 weight 的 top-$r$ singular directions——按定义就是能量最大的方向。

从优化角度看：loss landscape 的 Hessian 最大 eigenvalue 方向通常是 loss 变化最敏感的方向，weight 在这些方向上的更新幅度最大。把 trainable subspace 选成 top singular directions，相当于直接对齐到最重要的更新方向。

**2. 冻结的是小 singular value 方向**

$\Sigma$ 的对角元素降序排列，$\Sigma_{frozen}$ 持有的是最小的那些 singular values。冻住它们相当于说 "这些方向本来贡献就小，不动它们对 weight 的 Frobenius norm 影响最小"。

LoRA 冻住的是整个 backbone $W_{conv}$，包括它所有方向，无论重要与否。

**3. 不引入新参数**

调 rank 时 DRIFT 重新做一次 SVD，把现有 weight 重新切，没有新参数进来。LoRA 调 rank 要 merge 旧 adapter + re-inject 新 adapter，新 adapter 初始输出近似 0，等于突然把学到的 low-rank update 清零，training destabilize。

---

## Rank Scheduler：rank 怎么随时间衰减

有了 rank modulation 能切 rank，还得决定什么时候切到什么 rank。这就是 rank scheduler。

四种 decay function：

### Linear

$$
r_{linear} = \left\lfloor r_{max} - (r_{max} - r_{min}) \cdot \frac{i}{T} \right\rfloor
$$

变量：
- $i$：当前 epoch
- $T$：总 epoch 数
- $r_{max}$：初始 rank
- $r_{min}$：terminal rank
- $\lfloor \cdot \rfloor$：floor function

均匀衰减。简单但 early stage rank 降得太快，late stage 还在慢慢降。实验里 success rate 0.96，MBT 0.25s。

### Cosine

$$
r_{cosine} = \left\lfloor r_{min} + 0.5 \cdot (r_{max} - r_{min}) \cdot \left(1 + \cos\left(\pi \cdot \frac{i}{T}\right)\right) \right\rfloor
$$

变量：
- $\cos(\pi \cdot i/T)$：从 1 到 -1
- $1 + \cos(\pi \cdot i/T)$：从 2 到 0
- 整体：从 $r_{max}$ smooth 衰减到 $r_{min}$

前半段慢降，后半段快降。Success rate 1.0，MBT 0.25s。

### Sigmoid

$$
r_{sig} = \left\lfloor r_{max} - \frac{r_{max} - r_{min}}{1 + e^{-\tau \cdot (i - t_m)}} \right\rfloor
$$

变量：
- $t_m$：midpoint
- $\tau$：steepness
- $e^{-\tau(i - t_m)}$：sigmoid 的指数部分

前半段保持高 rank，midpoint 附近快速下降，后半段保持低 rank。$\tau = 0.5$ 时 success rate 1.0，MBT 0.24s——**最佳平衡**。

### Exponential

$$
r_{exp} = \left\lfloor r_{min} + (r_{max} - r_{min}) \cdot e^{-\tau \cdot i} \right\rfloor
$$

一开始就猛降。$\tau = 0.1$ success rate 0.88，$\tau = 0.5$ 掉到 0.72。**太激进，rank 还没学好就降到很低**。

### 为什么 sigmoid 最好

从 loss 曲线看（Figure 3），sigmoid $\tau=0.5$ 在 early stage 下降最快——因为前半段保持高 rank，充分享受 overparameterization 带来的 optimization landscape 优势。Mid-training 快速降 rank，late training 已经在 low-rank refinement 阶段，rank 小不影响性能。

Exponential 的问题：early 就开始降，rank 还没学好 general pattern 就被砍，underfitting。
Linear 的问题：衰减太均匀，前半段降太快（应该保持高 rank），后半段降太慢（应该快速降 rank 提效率）。

Sigmoid schedule 思路类似 SGDR 里的 cosine annealing（https://arxiv.org/abs/1608.03983），都是前期保持大学习率后期快速降。

---

## DRIFT-DAgger：把 DRIFT 塞进 interactive IL

Algorithm 1 伪代码简化版：

```
1. Offline bootstrapping:
   for epoch i = 1 to T:
     train π on D_B (offline demo dataset)
     r_i = DecayFunction(i, r_min, r_max)
     π = RankReduction(r_i, π)  # SVD 切一下
   
2. Online adaptation:
   D ← D_B
   for iteration j = 1 to I:
     learner rollout in env
     if expert detects deviation:
       expert takes control, provide corrective action
       collect (obs, action) into D_j
     D ← D ∪ D_j
     train π on D  # rank 固定在 r_min
```

两个 stage，offline 用 rank scheduler 从 $r_{max}$ 衰减到 $r_{min}$，online 保持 $r_{min}$。

Gating function 用 HG-DAgger 那套（https://arxiv.org/abs/1810.10006）：计算 learner action 和 expert action 的 cosine similarity，低于 threshold 就让 expert 接管。

---

## 实验关键数据

### Terminal rank $r_{min}$ 怎么选

PnP task（Table II）：

| $r_{min}$ | Success Rate | MBT (All) |
|-----------|--------------|-----------|
| 64 | 0.78 | 0.22 |
| 128 | 0.98 | 0.23 |
| 256 | 1.0 | 0.24 |
| 512 | 1.0 | 0.26 |

$r_{min} = 256$ 是 sweet spot。低于 128 表达力不够，高于 256 没额外收益。

**对比 LoRA fine-tuning 的 $r = 4$**：差 64 倍。这说明 from-scratch 训练 representation learning 阶段确实需要远高于 fine-tuning 的 rank。LoRA 在 from-scratch 上 suboptimal 的根本原因就在这里。

### 主实验对比（Table IV PnP）

| Method | SR | NEL ($\times 10^4$) | CT (hrs) |
|--------|-----|------|---------|
| Expert | 0.92 | - | - |
| BC | 1.00 | 4.86 | 3.76 |
| HG-DAgger | 1.00 | 3.26 | 3.30 |
| D(L) LoRA 一次 inject | 0.98 | 3.14 | 3.01 |
| D(LR) LoRA + scheduler | 0.58 | 3.49 | 3.10 |
| D(RR) Rank Mod + scheduler | 1.00 | 3.21 | 2.91 |

看点：

1. **D(LR) 崩了**（0.58）：LoRA + dynamic rank scheduler 反复 merge + re-inject，把训练搞崩。Paper 反复强调这点。
2. **D(RR) 最优**：success rate 1.0 跟 HG 持平，training time 2.91h 比 HG 3.30h 降 12%，expert label 3.21 比 HG 3.26 略低。
3. **BC 最费 expert**：NEL 4.86 远高于 interactive 方法的 ~3.2，证明 interactive IL 在 sample efficiency 上的优势在 diffusion policy 上依然成立。

### Batch training time（Table V）

| Method | SR | MBT (Offline) | MBT (Online) | MBT (All) |
|--------|-----|---------------|--------------|-----------|
| HG (full rank) | 1.0 | 0.27 | 0.27 | 0.27 |
| D(L) | 1.0 | 0.27 | 0.23 | 0.26 |
| D(LR) | 0.56 | 0.27 | 0.23 | 0.26 |
| D(RR) | 1.0 | 0.26 | 0.22 | 0.24 |

D(RR) 在 online stage 降 18% MBT，all stage 降 11%。绝对值看似不大（0.27 → 0.22），但对 online interactive loop 的 UX 至关重要——0.05s 乘上几百次 update 就是几分钟的人机等待差异。

### Additional baselines（Table VII）

| Method | SR | MBT |
|--------|-----|------|
| D (DRIFT-DAgger) | 1.0 | 0.24 |
| QR per step | 1.0 | 0.47 |
| FPMO (full offline, RM online) | 1.0 | 0.26 |
| MPLO (RM offline, LoRA online) | 1.0 | 0.24 |
| DSR (last 10% denoising) | 0.0 | 0.24 |
| QLoRA | 0.48 | 0.25 |

几个 takeaway：

1. **QR per step**：success rate 没提升但 MBT 翻倍。说明 strict orthonormality 不重要，能量分桶才重要。
2. **FPMO**：offline 全 rank online 才用 rank modulation，性能 1.0 但效率比 D(RR) 差。说明 offline 也该用 rank scheduler。
3. **DSR**：只在最后 10% denoising steps fine-tune，完全 collapse。说明 diffusion policy 没 pre-trained backbone，部分 fine-tune 在 from-scratch 场景行不通。
4. **QLoRA**：0.48，quantization + LoRA 在 from-scratch 上不稳。

### Real-world（Table VI）

Drawer Interaction task：

| Method | SR | MSD (min) | NEL | CT (hrs) |
|--------|-----|----------|------|---------|
| BC | 0.83 | 0.94±0.11 | 4.82 | 3.99 |
| HG | 0.90 | 0.85±0.12 | 3.94 | 3.87 |
| D(L) | 0.87 | 0.89±0.11 | 4.04 | 3.73 |
| D(LR) | 0.37 | 1.09±0.18 | 4.53 | 3.79 |
| D(RR) | 0.93 | 0.84±0.10 | 4.08 | 3.59 |

D(RR) 在 real-world 也是最优：success rate 最高 0.93，task duration 最短最稳，training time 最低 3.59h。

Long-horizon task（Drawer Assembling，~1.7 分钟）interactive IL 优势更明显：sample efficiency 提升 14.82%-17.14%。Short task（Block Stacking，~0.45 分钟）只提升 11.32%-13.17%。任务越长 interactive 收益越大——因为 long task 里 BC 失败的 trajectory 更多，expert correction 价值更高。

---

## 我觉得哪些地方可以挑刺

### 1. SVD 本身的 cost

每个 epoch 开始做 full SVD of $W \in \mathbb{R}^{m \times n}$，cost 是 $O(\min(m^2 n, m n^2))$。对于大 conv layer 这非 trivial。Paper 没在 ablation 里单独 report SVD cost 占总训练时间多少。

不过从 Table V 看 D(RR) 的 MBT 0.24s < HG 0.27s，说明 SVD overhead 被 backward 减少的 cost 盖过了。但如果 rank 衰减得很慢（比如 late stage rank 还很高），SVD cost 可能吃掉收益。

### 2. Stale SVD 问题

SVD 只在 epoch 开始做，epoch 内 weight 更新会让 $U, V$ 不再精确 orthonormal，trainable subspace 是 "stale" 的。Early training weight 变化大，stale 严重；late training weight 变化小，stale 影响小。

Paper 测试了 QR per step 严格保持 orthonormality，结果性能没提升但 MBT 翻倍（Table VII）。这说明 stale SVD 在实践中可以接受——可能因为 trainable subspace 即使 stale 也大致对齐 principal directions，小幅偏离不影响 gradient 更新方向。

### 3. Monotonic decay 的局限

Rank 只能降不能升。如果 online adaptation 遇到 OOD 数据需要新 representation，low rank 可能不够。Paper 在 Limitations 里提到这点，但没做实验验证。

可能的解法：用 loss spike 做 trigger，loss 突然涨就临时升 rank。或者做 cyclic schedule，rank 周期性升降。

### 4. Uniform rank across blocks

所有 conv block 用同一个 rank。但 U-Net 不同层 $r_{max}$ 不同，early layer 提取 low-level feature 可能需要更高 rank，late layer 做 high-level reasoning 可能 rank 可以更低。Adaptive per-block rank 是明显改进方向。

### 5. Online adaptation 的 data balance

Interactive IL 收集的 corrective data 集中在 hard states，跟 offline uniform data 混合训练有 distribution shift。Paper 没讨论这点。实际中可能需要 importance weighting 或 separate buffer。

---

## 更大的 context

### 跟 ML community intrinsic rank 研究的关系

Aghajanyan et al. 2020（https://arxiv.org/abs/2012.13255）发现 pre-trained language model fine-tuning 实际只在 ~100 维子空间进行。Li et al. 2018（https://arxiv.org/abs/1804.08838）发现 loss landscape intrinsic dimension 远低于参数量。

这些工作都是在 fine-tuning 场景下。DRIFT 说：from-scratch 训练也有 intrinsic low-rank structure，但这个 structure 是训出来的，所以需要动态 schedule——早期高 rank 探索，后期低 rank refine。这是对 intrinsic rank 思想从 fine-tuning 到 from-scratch 的扩展。

### 跟 Lottery Ticket 的关系

Lottery Ticket Hypothesis（https://arxiv.org/abs/1803.03635）说 dense network 里存在 sparse subnetwork 达到相近性能。DRIFT 类似：dense weight 里有 low-rank subspace 承载主要学习。

区别：Lottery Ticket 做 pruning（永久删参数），DRIFT 做 freezing（暂时停更新），可逆。Freezing 更灵活，万一发现冻错了可以解冻。

### 跟 DoRA / DyLoRA 的区别

DyLoRA（https://arxiv.org/abs/2210.07558）动态搜 rank，但仍是 LoRA-based，需要 pre-trained backbone。

DoRA（https://arxiv.org/abs/2402.09353）把 weight 分 magnitude 和 direction 分别 LoRA。仍 fine-tuning 导向。

DRIFT 独特之处：from-scratch + monotonic decay + SVD-based energy-preserving partition。这是 robotics 场景特殊需求——没 foundation model，每个 task train from scratch。

### Diffusion policy 加速方向

现在主流是 inference 加速：
- Consistency Policy（https://arxiv.org/abs/2405.07503）：distillation 少步推理
- One-Step Diffusion Policy（https://arxiv.org/abs/2410.21257）：单步 distillation

DRIFT 是少数关注 **training efficiency** 的工作。这两个方向正交——可以同时用 DRIFT 训练加速 + Consistency Policy 推理加速。

### Online RL 的可能性

Paper 没探索 online RL，但 PPO + LoRA 已有先例（https://arxiv.org/abs/2307.01852）。把 DRIFT 用到 PPO 的 policy network 上很 natural。Online RL sample efficiency 比 IL 更受限，reduced rank training 的 speed-up 价值更大。可能需要重新设计 rank scheduler——RL 里 reward signal 跟 IL 的 supervised loss 不一样，rank 衰减节奏可能要跟 reward curve 挂钩。

---

## 我的核心 takeaway

1. **Top-$r$ singular subspace 比 random subspace 好用**：这是 DRIFT vs LoRA 在 from-scratch 训练上差异的根本原因。数据驱动的 subspace 选择 > 随机初始化。

2. **Dynamic rank 是 fine-tuning 到 from-scratch 的桥梁**：intrinsic rank 思想在 fine-tuning 已被验证，DRIFT 把它扩展到 from-scratch 场景，关键 insight 是 rank 需要动态——前期高后期低。

3. **SVD-based partition 天然 stability**：不引入新参数这个性质太重要了。LoRA 反复 re-inject 是它 dynamic rank 不稳定的根源，DRIFT 用 SVD 巧妙绕开。

4. **Interactive IL 在 diffusion policy 时代重新 practical**：这是 robotics 的实际收益。HG-DAgger 这类方法在 small network 时代 work，diffusion policy 时代差点废掉，DRIFT 救回来。

5. **$r_{min} = 256$ vs LoRA $r = 4$ 的 64 倍 gap**：很 striking 的数字。From-scratch 训练需要的 rank 比 fine-tuning 高两个数量级。这对未来 robotics PEFT 方法设计有指导意义。

6. **Sigmoid decay 优于 exponential / linear**：这个 empirical 发现跟 learning rate schedule 的最佳实践一致——都是前期保持大、后期快速降。可能 rank 和 learning rate 在某种意义下有对偶关系。

---

## Reference

- DRIFT project page: https://apollo-lab-yale.github.io/25-RSS-DRIFT-website/
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- LoRA: https://arxiv.org/abs/2106.09685
- DyLoRA: https://arxiv.org/abs/2210.07558
- DoRA: https://arxiv.org/abs/2402.09353
- QLoRA: https://arxiv.org/abs/2305.14314
- HG-DAgger: https://arxiv.org/abs/1810.10006
- DAgger: https://arxiv.org/abs/1011.0686
- Mega-DAgger: https://arxiv.org/abs/2303.00638
- Intrinsic Dimensionality: https://arxiv.org/abs/2012.13255
- Loss Landscape Intrinsic Dimension: https://arxiv.org/abs/1804.08838
- Lottery Ticket: https://arxiv.org/abs/1803.03635
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Consistency Policy: https://arxiv.org/abs/2405.07503
- One-Step Diffusion Policy: https://arxiv.org/abs/2410.21257
- Diff-DAgger: https://arxiv.org/abs/2410.14868
- Diffusion meets DAgger: https://arxiv.org/abs/2402.17768
- Overparameterization Survey: https://arxiv.org/abs/2109.02355
- Dynamical Low-Rank Approximation: https://epubs.siam.org/doi/10.1137/060660723
- MVS Tasks: https://arxiv.org/abs/2409.14615
- Robosuite: https://arxiv.org/abs/2009.12293
- SGDR: https://arxiv.org/abs/1608.03983
- PPO+LoRA: https://arxiv.org/abs/2307.01852
- Double Descent: https://arxiv.org/abs/1912.02292
- PyTorch: https://arxiv.org/abs/1912.01703

---

# DRIFT: Dynamic Rank Adjustment in Diffusion Policies

这篇 paper 来自 Yale University 的 Apollo Lab（Daniel Rakita 组）和 University of Pennsylvania，发表于 RSS 2025。核心贡献是提出 DRIFT framework，利用 SVD-based rank modulation + rank scheduler 让 diffusion policy 在 from-scratch 训练过程中动态调整 trainable ranks，从而让 online interactive imitation learning（如 DAgger）在 large diffusion policy 上变得 practical。

项目主页：https://apollo-lab-yale.github.io/25-RSS-DRIFT-website/

---

## 1. Motivation：为什么 diffusion policy 需要动态 rank 调整

Diffusion policy [Chi et al. 2023] 用 DDPM 框架做 visuomotor control，把 robot action 当作 $x_0$，把 observation 作为 condition，用 U-Net 或 transformer 预测 noise。这类 policy 通常有 tens of millions 参数（overparameterization），带来了 strong representation，但也带来两个问题：

1. **Offline BC 训练慢**：每个 batch 的 forward+backward 耗时长。
2. **Online interactive IL 几乎不可行**：DAgger / HG-DAgger [Kelly et al. 2019] 这种在每轮 rollout 后立刻更新 policy 的方法，原本用 small MLP/LSTM，batch training time 很短；换成 diffusion policy 后，单次 update 时间从秒级变成分钟级，online loop 的交互节奏被破坏。

LoRA [Hu et al. 2022] 这类 PEFT 方法虽然能降参数，但设计目标是 fine-tuning pre-trained models，**直接搬到 from-scratch robotics 训练有两个根本问题**：

- LoRA 要求 backbone 已经 well-trained，否则 low-rank adapter 的表达力受限，无法享受 overparameterization 的好处。
- LoRA 在动态调 rank 时需要 merge + re-inject 新初始化的 adapter，这会 destabilize 训练（paper 中实验验证，见 Table IV 的 D(LR) 列），并且 forward pass 多了一次 $O(C_{out} \times C_{in} \times r \times k)$ 的卷积开销。

DRIFT 的 insight 是：**overparameterized network 训练过程中存在 intrinsic low-rank structure，训练早期需要高 rank 抓住 general pattern，后期只需 low rank 做 incremental refinement**。所以理想的 schedule 是 rank 从高到低动态衰减，既保住 early-stage 表示力，又省下 late-stage 的训练成本。

LoRA 论文：https://arxiv.org/abs/2106.09685
Diffusion Policy 论文：https://arxiv.org/abs/2303.04137
HG-DAgger 论文：https://arxiv.org/abs/1810.10006

---

## 2. Background：Diffusion Policy 与 rank

### 2.1 DDPM 数学回顾

Forward process（加噪）：

$$
q(x_t \mid x_{t-1}) := \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\right)
$$

变量含义：
- $x_0 \sim p(x_0)$：原始数据（robot action sequence）
- $t \in \{1, \dots, T\}$：diffusion timestep
- $\beta_t$：noise schedule，控制第 $t$ 步加噪的方差
- $\sqrt{1-\beta_t}$：mean 的 scaling，保证方差守恒
- $I$：单位矩阵，维度与 action space 对齐

Reverse process（去噪，由网络 $\pi_\theta$ 参数化）：

$$
x_{t-1} \sim p_\theta(x_{t-1} \mid x_t) := \mathcal{N}\left(x_{t-1};\, \mu_k(x_t, \pi_\theta(x_t, t)),\, \sigma_t^2 I\right)
$$

变量含义：
- $\pi_\theta(x_t, t)$：noise prediction network，输入 noisy action $x_t$ 和 timestep $t$（同时 condition 在 observation 上）
- $\mu_k(\cdot)$：根据预测 noise 反推 $x_{t-1}$ 的 mean（DDPM 用 closed-form，DDIM 用 deterministic）
- $\sigma_t^2$：fixed variance（通常取 $\beta_t$ 或 $\tilde{\beta_t}$）

### 2.2 卷积权重矩阵的 rank

对于 1D conv block，weight tensor $W_{conv} \in \mathbb{R}^{C_{out} \times C_{in} \times k}$，通过 reshape 变成矩阵 $W \in \mathbb{R}^{m \times n}$，其中 $m = C_{out} \cdot k$，$n = C_{in}$（或等价 view）。

$$
r_{max} \leq \min(m, n)
$$

对于 diffusion policy 的 U-Net，最大 rank 通常在 2048 量级（paper 中 $r_{max}=2048$）。

**Problem statement**：对每个 conv block 的 weight $W$，希望在整个 training 过程中能动态调整 trainable segment 的 rank $r \in [1, r_{max}]$，且：
- 不引入新参数
- 不破坏训练稳定性
- forward cost 不增（只降 backward cost）

---

## 3. DRIFT Framework 核心：Rank Modulation

### 3.1 SVD 分解与 partition

给定 weight matrix $W \in \mathbb{R}^{m \times n}$，做 full SVD：

$$
W = U \Sigma V^T
$$

变量含义：
- $U \in \mathbb{R}^{m \times m}$：left singular vectors，orthonormal，代表 row space 的 rotation/reflection
- $\Sigma \in \mathbb{R}^{m \times n}$：对角矩阵，对角元素 $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_{r_{max}} > 0$ 是 singular values（按降序排列）
- $V \in \mathbb{R}^{n \times n}$：right singular vectors，orthonormal，代表 column space 的 rotation/reflection
- $V^T$：$V$ 的转置

在指定 rank $r$ 处把三个矩阵 split：

$$
U = [U_{train} \mid U_{frozen}]
$$
$$
\Sigma = \begin{bmatrix} \Sigma_{train} & 0_{r \times (n-r)} \\ 0_{(m-r) \times r} & \Sigma_{frozen} \end{bmatrix}
$$
$$
V = [V_{train} \mid V_{frozen}]
$$

变量含义：
- $U_{train} \in \mathbb{R}^{m \times r}$：前 $r$ 个 left singular vectors（对应最大的 $r$ 个 singular values）
- $U_{frozen} \in \mathbb{R}^{m \times (m-r)}$：剩余的 left singular vectors
- $\Sigma_{train} \in \mathbb{R}^{r \times r}$：top-$r$ singular values 组成的对角矩阵（**能量最大的部分**）
- $\Sigma_{frozen}$：剩余较小的 singular values
- $V_{train} \in \mathbb{R}^{n \times r}$：前 $r$ 个 right singular vectors

然后重组成 trainable 和 frozen 两部分 weight：

$$
W_{train} = U_{train} \Sigma_{train} V_{train}^T
$$
$$
W_{frozen} = U_{frozen} \Sigma_{frozen} V_{frozen}^T
$$

**关键设计点**：
- 训练时只有 $\{U_{train}, \Sigma_{train}, V_{train}\}$ 接收 gradient，这三块对应 rank-$r$ subspace（top singular directions）
- $\{U_{frozen}, \Sigma_{frozen}, V_{frozen}\}$ 保持 frozen
- 因为 $\Sigma_{train}$ 持有 top singular values，trainable 部分承载了 weight matrix 的主要 "能量"（Frobenius norm 的主要贡献），这与 LoRA 把 adapter 加在整个 $W$ 上的逻辑不同
- 当 $r$ 减小时，重新做 SVD（在每个 epoch 开始时，仅当 rank 变化时），新的 $W_{train}$ 是更新后 weight 的 top-$r$ 投影，**不需要 inject 新参数**

### 3.2 Forward pass 的 cost

把 $W_{train} + W_{frozen}$ 加回去再 reshape 回 conv tensor：

$$
W_{conv} = \text{reshape}(W_{train} + W_{frozen}, (C_{out}, C_{in}, k))
$$

forward 仍是单次卷积，time complexity $O(C_{out} \cdot C_{in} \cdot k)$，与原 conv 相同。**没有 LoRA 那个 $O(C_{out} \cdot C_{in} \cdot r \cdot k)$ 的额外开销**。

### 3.3 与 LoRA 的对比公式

LoRA 的卷积形式：

$$
\text{Conv}_{LoRA}(x) = W_{conv} \circledast x + \alpha \left((W_{up} \times W_{down}) \circledast x\right)
$$

变量含义：
- $W_{down} \in \mathbb{R}^{r \times C_{in} \times k}$：down-projection（降维到 rank-$r$）
- $W_{up} \in \mathbb{R}^{C_{out} \times r \times k}$：up-projection（升维回 $C_{out}$）
- $\alpha$：scaling factor，控制 adapter 输出幅度
- $r \ll C_{in}$：rank bottleneck
- $\circledast$：卷积运算

forward 增加的 cost：$O(C_{out} \cdot C_{in} \cdot r \cdot k)$，与 $r$ 线性增长。Fine-tuning 时 $r \leq 4$ 影响小；from-scratch 训练需要 $r \sim 256$，开销就显著了。

**DRIFT vs LoRA 的本质区别**：
| 特性 | LoRA | DRIFT (Rank Modulation) |
|------|------|------------------------|
| 新参数 | 是（$W_{up}, W_{down}$） | 否 |
| Forward cost | 增加 $O(r)$ 倍 | 不变 |
| Re-init on rank change | 需要（destabilize） | 不需要 |
| 依赖 pre-trained backbone | 强 | 弱 |
| Trainable subspace | 任意 low-rank subspace | top-$r$ singular subspace（能量集中） |

### 3.4 Reorthonormalization 的策略

SVD 之后 $U, V$ 是 orthonormal 的，但训练若干步后 gradient 更新会破坏 orthonormality。Paper 测试了两种方案：

1. **每个 gradient step 后做 QR decomposition**：保证 $U, V$ 始终 orthonormal，但 overhead 大。实验中 offline MBT 从 0.26s 升到 0.55s（见 Table VII 的 QR 行），online MBT 从 0.22s 升到 0.33s，性能无明显提升。
2. **每个 epoch 开始时（仅 rank 变化时）做 full SVD**：默认方案，overhead 小，性能与 QR 相当。

这个 trade-off 提示：**strict orthonormality 不是性能关键，能量分桶（top-$r$ vs bottom）才是关键**。

---

## 4. Rank Scheduler：四种 Decay Function

Rank scheduler 类比 diffusion 的 noise scheduler，用 decay function 在 epoch $i$ 计算 trainable rank $r_i$。

设 $r_{max}$ 为初始 rank，$r_{min}$ 为 terminal rank，$T$ 为 total epochs，$i$ 为 current epoch，$t_m$ 为 midpoint，$\tau$ 为 steepness，$\lfloor \cdot \rfloor$ 为 floor function。

### 4.1 Linear decay

$$
r_{linear} = \left\lfloor r_{max} - (r_{max} - r_{min}) \cdot \frac{i}{T} \right\rfloor
$$

变量含义：
- $\frac{i}{T}$：training 进度，从 0 到 1
- $(r_{max} - r_{min})$：rank 总衰减幅度

特点：恒定衰减速率。实验中 success rate 0.96（接近 perfect），MBT 0.25s（比 HG 0.27s 略快），但训练时间在所有 decay function 中最高。

### 4.2 Cosine decay

$$
r_{cosine} = \left\lfloor r_{min} + 0.5 \cdot (r_{max} - r_{min}) \cdot \left(1 + \cos\left(\pi \cdot \frac{i}{T}\right)\right) \right\rfloor
$$

变量含义：
- $\cos(\pi \cdot i/T)$：从 1 衰减到 -1
- $1 + \cos(\pi \cdot i/T)$：从 2 衰减到 0
- $0.5 \cdot (1 + \cos(\cdot))$：从 1 衰减到 0 的 smooth 函数

特点：early slow decay, late fast decay（与 warmup cosine 相反）。Success rate 1.0，MBT 0.25s。

### 4.3 Sigmoid decay

$$
r_{sig} = \left\lfloor r_{max} - \frac{r_{max} - r_{min}}{1 + e^{-\tau \cdot (i - t_m)}} \right\rfloor
$$

变量含义：
- $t_m$：sigmoid 中心点（midpoint）
- $\tau$：steepness，控制 sigmoid 的陡峭度
- $e^{-\tau(i - t_m)}$：从 $+\infty$（$i \ll t_m$）衰减到 0（$i \gg t_m$）
- $1 + e^{-\tau(i - t_m)}$：从 $\infty$ 到 1
- 整个分式：从 0 到 $(r_{max} - r_{min})$
- $r_{max} -$ 整个分式：从 $r_{max}$ 到 $r_{min}$

特点：early 保持高 rank，mid-training 快速衰减，late 保持低 rank。$\tau=0.5$ 时 success rate 1.0，MBT 0.24s（最佳平衡）。$\tau=0.1$ 时 success rate 0.98（衰减太平缓，rank 没充分降下来）。

### 4.4 Exponential decay

$$
r_{exp} = \left\lfloor r_{min} + (r_{max} - r_{min}) \cdot e^{-\tau \cdot i} \right\rfloor
$$

变量含义：
- $e^{-\tau i}$：从 1 指数衰减到 0
- $(r_{max} - r_{min}) \cdot e^{-\tau i}$：rank 的衰减部分
- $r_{min} +$ 上述：确保不低于 $r_{min}$

特点：early 极速衰减。$\tau=0.1$ 时 success rate 0.88，$\tau=0.5$ 时 success rate 0.72。**衰减太激进，rank 在还没学好 general pattern 时就降到很低**，导致 underfitting。

### 4.5 Decay function 选择 intuition

Table I 总结：

| Function | Success Rate | MBT (Offline) | MBT (Online) | MBT (All) |
|----------|--------------|---------------|--------------|-----------|
| HG (full rank) | 1.0 | 0.27 | 0.27 | 0.27 |
| Linear | 0.96 | 0.26 | 0.22 | 0.25 |
| Cosine | 1.0 | 0.26 | 0.22 | 0.25 |
| Exp 0.1 | 0.88 | 0.23 | 0.23 | 0.23 |
| Exp 0.5 | 0.72 | 0.22 | 0.22 | 0.22 |
| Sig 0.1 | 0.98 | 0.25 | 0.23 | 0.24 |
| Sig 0.5 | 1.0 | 0.26 | 0.22 | 0.24 |

Intuition：**sigmoid with $\tau=0.5$ 最优**，因为它在前半段保持高 rank 充分利用 overparameterization（loss 下降快），在后半段快速降到 $r_{min}$ 加速训练。Exponential 太激进；linear 太保守（rank 持续慢降，效率提升不够）。

Sigmoid 学习率 schedule 参考 SGDR：https://arxiv.org/abs/1608.03983

---

## 5. DRIFT-DAgger Algorithm

DRIFT-DAgger 把 DRIFT framework 嵌入到 interactive IL 流程中：

```
Algorithm 1: DRIFT-DAgger
1. procedure DRIFT-DAgger(π_exp, π_N0, D_B)
2.   for offline epoch i = 1, ..., T do
3.     train π_Ni on offline dataset D_B
4.     if use rank scheduler:
5.       r_i = DecayFunction(i, r_min, r_max)
6.       π_Ni = RankReduction(r_i, π_Ni)
7.   if not use rank scheduler:
8.     π_NT = RankReduction(r_min, π_NT)
9.   D ← D_B
10.  for online iteration j = 1, ..., I do
11.    for timestep t ∈ T of online rollout j:
12.      if π_exp takes control:
13.        observation ← rollout_j^t
14.        action ← π_exp(observation)
15.        D_j ← (observation, action)
16.    D ← D ∪ D_j
17.    Train π_(N_T+j) on D
18.  return π_(N_T+I)
```

### 5.1 Two-stage 流程

**Stage 1: Offline Bootstrapping**
- 用 BC 在离线 demonstration 数据集 $D_B$ 上训练
- 配合 rank scheduler，rank 从 $r_{max}$ 衰减到 $r_{min}$
- 这个阶段建立 general behavior representation

**Stage 2: Online Adaptation**
- Learner 在环境中 rollout
- Expert 监控，发现 deviation 时接管并提供 corrective action
- 收集到 $D_j$，并入 global dataset $D$
- 用 expanded $D$ 训练 $\pi_{N_{T+j}}$，rank 固定在 $r_{min}$（或者继续用 rank scheduler）

### 5.2 Gating function

Paper 采用 expert-gated 策略（HG-DAgger 风格）：用 cosine similarity 判断 expert 何时接管。

$$
\text{sim}(\pi_{exp}(o), \pi_{N}(o)) < \text{threshold} \implies \text{expert takes control}
$$

Threshold 设置见 Table III：
- Robosuite-Lift: 0.94
- Robosuite-Can: 0.95
- MVS-Microwave: 0.99
- MVS-PnP: 0.99

Threshold 越高，expert 越早干预，sample efficiency 越高但 expert 负担越重。

HG-DAgger 论文：https://arxiv.org/abs/1810.10006
Mega-DAgger（同作者前作）：https://arxiv.org/abs/2303.00638

---

## 6. Simulation Experiments 详细分析

### 6.1 实验环境

四个仿真环境：
1. **Robosuite-Lift**：Panda arm 抓红色 cube 并举起
2. **Robosuite-Can**：Panda arm 把 can 放进指定 bin
3. **MVS-Microwave**：双 xArm7，一只手拿 gripper，一只手拿 camera，开微波炉
4. **MVS-PnP**：双臂抓绿色 cube 放红色区域

MVS 任务的特殊之处：**active perception**——一只机械臂专门负责调整相机视角，另一只执行 manipulation，两者协同。

### 6.2 Terminal rank $r_{min}$ 的 ablation

Table II（PnP task）：

| $r_{min}$ | Success Rate | MBT (Offline) | MBT (Online) | MBT (All) |
|-----------|--------------|---------------|--------------|-----------|
| 64 | 0.78 | 0.24 | 0.19 | 0.22 |
| 128 | 0.98 | 0.25 | 0.21 | 0.23 |
| 256 | 1.0 | 0.26 | 0.22 | 0.24 |
| 512 | 1.0 | 0.27 | 0.24 | 0.26 |

关键观察：
- $r_{min}=64$：太低，表达力不足，success rate 掉到 0.78
- $r_{min}=128$：接近 optimal，0.98
- $r_{min}=256$：达到 perfect，是 sweet spot
- $r_{min}=512$：性能不再提升，但训练时间增加，无收益

**与 LoRA fine-tuning 的对比**：LoRA fine-tuning 通常 $r=4$ 就够，因为 backbone 已经 pre-trained，只需 small adjustment。但 from-scratch 训练需要 $r \sim 256$，差距 ~64 倍。这说明 from-scratch 训练的 representation learning 阶段确实需要 higher rank subspace。

Table VIII（Microwave task）也证实 $r_{min}=256$ 是普适 sweet spot：
| $r_{min}$ | Success Rate | MBT (All) |
|-----------|--------------|-----------|
| 64 | 0.78 | 0.21 |
| 128 | 1.0 | 0.22 |
| 256 | 1.0 | 0.23 |
| 512 | 1.0 | 0.25 |

### 6.3 Benchmark 对比（Table IV）

四种方法对比：BC, HG-DAgger, DRIFT-DAgger with LoRA (D(L)), DRIFT-DAgger with LoRA + rank scheduler (D(LR)), DRIFT-DAgger with rank modulation + rank scheduler (D(RR))。

PnP 任务关键数据：
| Method | SR | MSD | NEL | CT (hours) |
|--------|-----|------|------|-----|
| Expert | 0.92 | 3.03±0.69 | - | - |
| BC | 1.00 | 2.61±0.23 | 4.86×10⁴ | 3.76 |
| HG | 1.00 | 2.54±0.23 | 3.26×10⁴ | 3.30 |
| D(L) | 0.98 | 2.60±0.40 | 3.14×10⁴ | 3.01 |
| D(LR) | 0.58 | 3.54±0.84 | 3.49×10⁴ | 3.10 |
| D(RR) | 1.00 | 2.73±0.50 | 3.21×10⁴ | 2.91 |

关键发现：
1. **D(LR) 崩了**（success rate 0.58）：因为 LoRA 在 rank scheduler 触发时 merge + re-inject 新初始化 adapter，破坏训练。这是 paper 反复强调的 LoRA 不适合 from-scratch dynamic rank 的核心证据。
2. **D(RR) 最优**：success rate 1.0，与 HG 持平；NEL 3.21×10⁴ 比 HG 的 3.26×10⁴ 略低（sample efficiency 更好）；CT 2.91h 比 HG 的 3.30h 降低 ~12%。
3. **D(L) 表现也不错**（success rate 0.98），说明 LoRA 如果只在 offline→online 转换时 inject 一次（不动态调 rank）也能工作，但失去 rank scheduler 的训练加速。
4. **BC vs HG vs D(RR)**：BC 的 NEL 4.86×10⁴ 远高于 interactive methods，说明 interactive IL 在 sample efficiency 上的优势在 diffusion policy 上仍然成立。
5. **MSD（mean & std of task duration）**：D(RR) 的 std 0.50 比 BC 的 0.23 大，说明 online adaptation 阶段引入的 variance，但 mean 2.73 接近 BC 的 2.61，仍可接受。

### 6.4 Batch Training Time 对比（Table V）

| Method | Success Rate | MBT (Offline) | MBT (Online) | MBT (All) |
|--------|--------------|---------------|--------------|-----------|
| HG | 1.0 | 0.27 | 0.27 | 0.27 |
| D(L) | 1.0 | 0.27 | 0.23 | 0.26 |
| D(LR) | 0.56 | 0.27 | 0.23 | 0.26 |
| D(RR) | 1.0 | 0.26 | 0.22 | 0.24 |

D(RR) 在 online stage 减少 18% batch time，all-stage 减少 11%。这个 reduction 看似不大，但对 online interactive loop 至关重要——它把每个 update cycle 从 0.27s 降到 0.22s，在 expert-in-the-loop 场景下大幅提升 UX。

### 6.5 Additional baselines（Table VII）

| Method | Success Rate | MBT (All) |
|--------|--------------|-----------|
| D (DRIFT-DAgger) | 1.0 | 0.24 |
| QR (QR per step) | 1.0 | 0.47 |
| FPMO (full-rank offline, RM online) | 1.0 | 0.26 |
| MPLO (RM offline, LoRA online) | 1.0 | 0.24 |
| DSR (last 10% denoising steps) | 0.0 | 0.24 |
| QLoRA | 0.48 | 0.25 |

关键发现：
1. **QR per step**：success rate 不变（1.0），但 MBT 翻倍（0.47），证明 strict orthonormality 不是性能关键。
2. **FPMO**：offline 用 full rank, online 才用 rank modulation。性能 1.0 但 MBT 0.26，比 D(RR) 的 0.24 差，说明 offline rank scheduler 也有价值。
3. **MPLO**：offline 用 rank modulation, online 用 LoRA。性能与 D(RR) 相同，说明 rank modulation 在 offline bootstrapping 阶段就足够好，online 用什么都行（只要不 re-inject）。
4. **DSR**：只在最后 10% denoising steps 上 fine-tune。**完全 collapse**（success rate 0.0），因为 diffusion policy 没有像 image generation 那样的 well-pretrained backbone。
5. **QLoRA**：success rate 0.48，quantization + LoRA 组合在 from-scratch 训练中不稳。

QLoRA 论文：https://arxiv.org/abs/2305.14314

---

## 7. Real-World Experiments

三个真实任务（双 xArm7 + linear motor，17-DOF system）：
1. **Block Stacking**：非长方体 block 堆叠
2. **Drawer Assembling**：两个 drawer box 插入 drawer container
3. **Drawer Interaction**：移开遮挡物 → 抓 red cube → 放进 drawer → 关 drawer（long-horizon）

Table VI 关键数据：

| Task | Method | SR | MSD (min) | NEL | CT (hrs) |
|------|--------|-----|----------|------|---------|
| Block Stacking | BC | 0.97 | 0.41±0.04 | 4.86 | 4.45 |
| | HG | 1.00 | 0.42±0.04 | 4.22 | 4.35 |
| | D(L) | 1.00 | 0.43±0.05 | 4.31 | 4.19 |
| | D(LR) | 0.53 | 0.58±0.05 | 4.57 | 4.23 |
| | D(RR) | 1.00 | 0.43±0.05 | 4.25 | 4.03 |
| Drawer Assembling | BC | 0.40 | 1.76±0.21 | 12.08 | 14.50 |
| | HG | 0.77 | 1.52±0.18 | 10.21 | 14.21 |
| | D(RR) | 0.73 | 1.58±0.15 | 10.01 | 13.28 |
| Drawer Interaction | BC | 0.83 | 0.94±0.11 | 4.82 | 3.99 |
| | HG | 0.90 | 0.85±0.12 | 3.94 | 3.87 |
| | D(RR) | 0.93 | 0.84±0.10 | 4.08 | 3.59 |

关键观察：
1. **Long-horizon task（Drawer Assembling）interactive IL 优势更明显**：sample efficiency 提升 14.82%-17.14%（NEL 12.08→10.01）。Short task（Block Stacking）只有 11.32%-13.17% 提升。
2. **D(RR) 在 Drawer Interaction 上最佳**：SR 0.93 > HG 0.90 > BC 0.83；MSD 0.84±0.10 最低且最稳定；CT 3.59h 最低。
3. **D(LR) 在 real-world 也崩**：Block Stacking SR 0.53，Drawer Assembling SR 0.20，Drawer Interaction SR 0.37。再次验证 LoRA + dynamic rank scheduler 的不稳定性。
4. **Drawer Assembling 难度大**：BC SR 只有 0.40，HG 0.77，D(RR) 0.73。即使 interactive IL 也很难达到 perfect，说明 task 本身 challenge 大。

---

## 8. Intuition Building：为什么 DRIFT work

### 8.1 Intrinsic dimensionality 的视角

Aghajanyan et al. 2020 的 intrinsic dimensionality 研究表明，pre-trained language model fine-tuning 实际只在低维子空间进行。Li et al. 2018 发现 loss landscape 的 intrinsic dimension 远低于参数量。

DRIFT 的 implicit 假设：**diffusion policy from-scratch 训练也存在 intrinsic low-rank structure**，但这个 structure 不是从一开始就 present，而是在训练过程中逐渐 emerge。

- **Early training**：loss landscape 还在剧烈变化，需要 full rank 探索
- **Mid training**：principal directions 开始稳定，可以截断 small singular values
- **Late training**：weight 主要在 top-$r$ subspace 内做 fine refinement

这与 dynamical low-rank approximation [Koch & Lubich 2007] 的思想相通，但 DRIFT 不需要 target matrix 可微（deep learning 的 weight matrix 是隐式的）。

Intrinsic dimension 论文：https://arxiv.org/abs/2012.13255
Loss landscape intrinsic dimension：https://arxiv.org/abs/1804.08838
Dynamical low-rank approximation：https://epubs.siam.org/doi/10.1137/060660723

### 8.2 Top-r vs random-r subspace 的选择

DRIFT 把 trainable subspace 选为 top-$r$ singular directions（对应最大 singular values），这是关键设计。理由：

- Top singular directions 持有 weight matrix 的主要 Frobenius norm 能量
- 训练 gradient 在 top singular directions 上的投影通常更大（因为 Hessian 的最大 eigenvalue 对应的方向是 loss 变化最敏感的方向）
- 把 small singular values 冻结相当于 prior: "这些方向不重要，不动它们"

对比 LoRA：LoRA 的 $W_{up}, W_{down}$ 是随机初始化的，初始 trainable subspace 是随机的，需要训练才能 align 到 principal directions。这就是为什么 LoRA 在 fine-tuning 上 work（已经 aligned）但在 from-scratch 训练上 suboptimal。

### 8.3 SVD 的稳定性 vs LoRA 的不稳定性

LoRA + dynamic rank 的失败原因（D(LR) success rate ~0.5）：
1. 新初始化的 $W_{up}^{(new)}, W_{down}^{(new)}$ 引入随机噪声
2. Merge 旧 adapter 后 weight 是 $W_{conv} + \alpha W_{up}^{(old)} W_{down}^{(old)}$，新 adapter 初始输出近似 0
3. gradient 在新 adapter 上的更新需要重新 align 到 principal directions
4. 频繁 re-inject 等于频繁 perturb 已学好的 representation

DRIFT 的 SVD-based rank modulation 的稳定性原因：
1. 不引入新参数，只是把现有 weight 分成两部分
2. Top-$r$ subspace 是数据驱动的（由当前 weight 的 SVD 决定），不是随机的
3. Rank 减小时，只是把 bottom singular directions 冻结，它们本来贡献就小
4. Rank 增加时（虽然 paper 不做这个方向），可以解冻 bottom directions，自然延续

### 8.4 与 DoRA / DyLoRA 的关系

DyLoRA [Valipour et al. 2022]：在 fine-tuning 时动态搜索 rank，但仍然是 LoRA-based，需要 pre-trained backbone。

DoRA [Liu et al. 2024]：把 weight 分解为 magnitude 和 direction，分别 LoRA 适配。仍是 fine-tuning 导向。

DRIFT 的核心区别：**from-scratch 训练 + monotonic rank decay + SVD-based energy-preserving partition**。这是 robotics 场景的特殊需求——没有 foundation model 可用，每个 task 都要 train from scratch。

DyLoRA 论文：https://arxiv.org/abs/2210.07558
DoRA 论文：https://arxiv.org/abs/2402.09353

### 8.5 与 Lottery Ticket Hypothesis 的联系

Lottery Ticket Hypothesis [Frankle & Carbin 2019]：dense network 中存在 sparse subnetwork 能达到相近性能。DRIFT 的视角类似：dense weight 中存在 low-rank subspace 能承载主要学习。

但 DRIFT 不做 pruning（永久删除参数），而是 freezing（暂时停止更新）。这是 reversible 的，比 pruning 更灵活。

Lottery Ticket 论文：https://arxiv.org/abs/1803.03635

---

## 9. Architecture 图解析

### 9.1 Figure 1：动机示意

Top: 全 rank BC 训练的 policy 尝试插 upper drawer box，撞 container 和 lower drawer box（失败）。
Bottom: 经过 efficient online adaptation with reduced trainable ranks 后，成功完成 task。

**关键 message**：reduced rank training 不损失 performance，反而因为 online adaptation 提升了 task performance。

### 9.2 Figure 2：DRIFT-DAgger pipeline

两部分：
- **Offline Bootstrapping**：用 demonstration dataset $D_B$ 训练，rank scheduler 从 $r_{max}$ 衰减到 $r_{min}$
- **Online Adaptation**：learner rollout + expert intervention + corrective demonstration + update with reduced rank

**Gating function**：图中标注为 expert 介入的判断节点，对应 HG-DAgger 的 human-in-the-loop 设计。

### 9.3 Figure 3 & 4：Loss / Success Rate / MBT 曲线

- Loss 曲线：sigmoid 0.5 在 early stage 下降最快（高 rank 的好处），后期收敛到与 full rank 相近
- Success rate 曲线：sigmoid 0.5 最早达到 1.0
- MBT 曲线：随着 rank 减小，MBT 单调下降

### 9.4 Figure 5：四个 simulation 环境的 success rate vs NEL 曲线

四种方法在四个 task 上的 sample efficiency 对比。**D(RR) 曲线在大多数 task 上最高最早达到 plateau**，说明 reduced rank + interactive IL 的协同效应。

---

## 10. Limitations 与 Future Directions

Paper 自己承认的局限：
1. **未探索 online RL**：DRIFT 只在 IL 上测试，PPO/SAC 这类 on-policy RL 也可以受益于 dynamic rank
2. **Monotonic decay**：rank 只能单调下降，不能增回去。实际中如果 online adaptation 遇到 OOD 数据，可能需要 rank increase
3. **Uniform rank across blocks**：所有 conv block 用同样的 rank。不同 block 的 $r_{max}$ 不同，应该自适应
4. **Task-dependent decay function**：sigmoid 0.5 是 empirical sweet spot，但理论上应该 task-dependent

我（Karpathy 视角）会额外指出几个潜在问题：
1. **SVD cost**：每个 epoch 开始做 full SVD of $W \in \mathbb{R}^{m \times n}$ 是 $O(\min(m^2 n, m n^2))$。对于大 conv layer 这个 cost 非 trivial，paper 没详细讨论这个 overhead 在总训练时间中的占比
2. **Stale SVD**：epoch 内不更新 SVD，意味着 trainable subspace 是 "stale" 的。对于 fine-tuning 阶段 weight 变化小，stale 影响小；对于 from-scratch 早期 weight 变化大，stale 可能导致 trainable subspace 与实际 principal directions 偏离
3. **Online adaptation 的 dataset balance**：interactive IL 收集的 corrective data 通常 distribution skewed（集中在 hard states），与 offline uniform data 混合训练时可能有 distribution shift issue，paper 没深入讨论
4. **Continuous rank 的可能性**：当前用 floor function 离散化 rank，理论上可以做 soft rank（用 sigmoid gating 每个 singular direction），可能更 smooth

---

## 11. 与 broader ML context 的连接

### 11.1 Diffusion model 加速方向

Paper 提到现有 diffusion policy 加速主要在 inference：
- **Consistency Policy** [Prasad et al. 2024]：distillation 加速 inference
- **One-Step Diffusion Policy** [Wang et al. 2024]：single-step distillation

DRIFT 是少数关注 **training efficiency** 的工作。

Consistency Policy：https://arxiv.org/abs/2405.07503
One-Step Diffusion Policy：https://arxiv.org/abs/2410.21257

### 11.2 Diffusion + DAgger 系列

- **Diff-DAgger** [Lee & Kuo 2024]：用 diffusion 做 uncertainty estimation，robot-gated 而非 expert-gated，无 real-world 实验
- **Diffusion meets DAgger** [Zhang et al. 2024]：用 diffusion 做 data augmentation，主要创新在 data 层面

DRIFT-DAgger 的独特定位：**训练效率**层面的创新，让 interactive IL 在 large diffusion policy 上 practical。

Diff-DAgger：https://arxiv.org/abs/2410.14868
Diffusion meets DAgger：https://arxiv.org/abs/2402.17768

### 11.3 Overparameterization 与 generalization

Dar et al. 2021 的 "A Farewell to the Bias-Variance Tradeoff" 综述指出，overparameterized regime 下 classical bias-variance tradeoff 被 double descent 取代。DRIFT 在 dynamic rank training 中实际上是在 underparameterized 和 overparameterized regime 之间滑动——early training 充分 overparameterized 享受 good optimization landscape，late training 收敛到 low-rank underparameterized 区域享受 efficiency。

Double descent 论文：https://arxiv.org/abs/1912.02292
Overparameterization 综述：https://arxiv.org/abs/2109.02355

### 11.4 Robotic RL 中类似的 low-rank 思想

虽然 paper 没探索 RL，但 PPO + LoRA 已经有相关工作（如 https://arxiv.org/abs/2307.01852）。把 DRIFT 用到 PPO 的 policy network 上是 natural extension。online RL 的 sample efficiency 限制比 IL 更严，reduced rank training 的 speed-up 价值可能更大。

---

## 12. 总结：DRIFT 的核心 contribution

**Conceptual contribution**：把 PEFT 思想从 fine-tuning 扩展到 from-scratch training，并提出 monotonic rank decay 的 training paradigm。

**Technical contribution**：SVD-based rank modulation 避免 LoRA 的 re-injection 不稳定性，保持 forward cost 恒定，同时让 backward cost 随 rank 线性下降。

**Practical contribution**：让 HG-DAgger 这类 interactive IL 方法重新在 diffusion policy 时代变得 practical，real-world cumulative training time 从 3.99h 降到 3.59h（Drawer Interaction task）。

**Methodological insight**：top-$r$ singular subspace（数据驱动）远优于 random subspace（LoRA 风格），特别是在 from-scratch 训练中。Energy-preserving decomposition 是 stability 的关键。

**Empirical insight**：$r_{min}=256$ 是 diffusion policy 的 robust sweet spot，远高于 fine-tuning 场景的 $r=4$；sigmoid decay with $\tau=0.5$ 是 balance training efficiency 与 representation power 的 best practice。

这篇 paper 的价值在于把 ML community 的 intrinsic rank insight 与 robotics 的 from-scratch training 现实需求桥接起来，为 large model 在 robotics 中的 practical deployment 提供了 actionable 的 efficiency 方案。

---

## 参考文献（关键论文）

1. **DRIFT paper**: https://apollo-lab-yale.github.io/25-RSS-DRIFT-website/
2. **Diffusion Policy (Chi et al. 2023)**: https://arxiv.org/abs/2303.04137
3. **DDPM (Ho et al. 2020)**: https://arxiv.org/abs/2006.11239
4. **DDIM (Song et al. 2021)**: https://arxiv.org/abs/2010.02502
5. **LoRA (Hu et al. 2022)**: https://arxiv.org/abs/2106.09685
6. **DyLoRA (Valipour et al. 2022)**: https://arxiv.org/abs/2210.07558
7. **DoRA (Liu et al. 2024)**: https://arxiv.org/abs/2402.09353
8. **QLoRA (Dettmers et al. 2023)**: https://arxiv.org/abs/2305.14314
9. **HG-DAgger (Kelly et al. 2019)**: https://arxiv.org/abs/1810.10006
10. **DAgger (Ross et al. 2011)**: https://arxiv.org/abs/1011.0686
11. **Intrinsic Dimensionality (Aghajanyan et al. 2020)**: https://arxiv.org/abs/2012.13255
12. **Loss Landscape Intrinsic Dimension (Li et al. 2018)**: https://arxiv.org/abs/1804.08838
13. **Lottery Ticket Hypothesis (Frankle & Carbin 2019)**: https://arxiv.org/abs/1803.03635
14. **Mega-DAgger (Sun et al. 2023)**: https://arxiv.org/abs/2303.00638
15. **Diff-DAgger (Lee & Kuo 2024)**: https://arxiv.org/abs/2410.14868
16. **Diffusion meets DAgger (Zhang et al. 2024)**: https://arxiv.org/abs/2402.17768
17. **Consistency Policy (Prasad et al. 2024)**: https://arxiv.org/abs/2405.07503
18. **One-Step Diffusion Policy (Wang et al. 2024)**: https://arxiv.org/abs/2410.21257
19. **Overparameterization Survey (Dar et al. 2021)**: https://arxiv.org/abs/2109.02355
20. **Dynamical Low-Rank Approximation (Koch & Lubich 2007)**: https://epubs.siam.org/doi/10.1137/060660723
21. **MVS Tasks (Sun et al. 2024)**: https://arxiv.org/abs/2409.14615
22. **Robosuite (Zhu et al. 2020)**: https://arxiv.org/abs/2009.12293
23. **PyTorch (Paszke et al. 2019)**: https://arxiv.org/abs/1912.01703
24. **SGDR (Loshchilov & Hutter 2017)**: https://arxiv.org/abs/1608.03983
25. **Double Descent (Belkin et al. 2019)**: https://arxiv.org/abs/1912.02292
