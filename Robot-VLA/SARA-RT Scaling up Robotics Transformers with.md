---
source_pdf: SARA-RT Scaling up Robotics Transformers with.pdf
paper_sha256: b9aec586c915b42251651f928a202f1d8b21febc36b5ad4cdd2e45746a422bdd
processed_at: '2026-08-12T03:04:27-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SARA-RT 人话版

## 一句话说清楚

**已经 train 好的 robot transformer 跑得太慢？把它的 attention 模块换成 linear attention，fine-tune 几百步，速度翻倍精度不掉。**

就这么简单。剩下的全是工程细节和数学证明。

---

## 为什么 robot 上 attention 是个麻烦

Transformer 的 attention 本质上就是算"每个 token 要看其他所有 token 多少"。N 个 token 互相看，就是 $N \times N$ 次计算。N 小的时候无所谓，N 大了就炸。

Robot 上这个矛盾特别尖锐：

- RT-1 才 35M 参数，已经只能跑 3Hz
- RT-2 是 5B / 55B，PaLI-X backbone，attention 序列 200 左右，TPU 上一个 forward 53ms
- Point Cloud Transformer 处理点云，序列长度轻松 1000-4000，quadratic 直接爆

Robot control 要实时，10Hz 起步才有手感，50Hz 以上才丝滑。你给它套个 5B 的 VLM，它在想"我要不要抓这个 coke can"的时候，机械臂已经等了 100ms。这就是 deployment 的核心痛点。

参考：RT-2 的 real-time deployment 讨论 https://arxiv.org/abs/2307.15818

---

## Linear attention 这件事本身不新

把 $O(N^2)$ 降到 $O(N)$ 的 trick 早就有了。核心 idea 就是：**别一个个 token 互相算 similarity，先把所有 keys 压成一个 summary matrix，然后每个 query 去查这个 summary。**

数学上就是找到一种方式把 $K(\mathbf{q}, \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k})$ 拆成 $\phi(\mathbf{q})^\top \phi(\mathbf{k})$ 的形式。拆开了就能先算 $\sum_j \phi(\mathbf{k}_j) \phi(\mathbf{k}_j)^\top$，存起来，每个 query $O(m)$ 查一次就完事。

这个 trick 叫 kernelization，在 SVM 时代就是教科书内容。问题是：**softmax kernel 不好拆**，因为 $\exp(\mathbf{q}^\top \mathbf{k})$ 没法写成两个独立函数的乘积。

现有的两条路：

**路线 A：Random features (Performers)**
用随机 Gaussian matrix $\mathbf{G}$ 把 $\mathbf{q}, \mathbf{k}$ 投影到高维空间，然后 element-wise 取 $\exp$。数学上可以证明这是 $\exp(\mathbf{q}^\top \mathbf{k})$ 的无偏估计。但 variance 大，要 $m \approx 2048$ 维才够准。所以只对 $N > 4096$ 的超长序列才划算，对 robot 上 $N=200$ 这种反而更慢。

**路线 B：简单函数 (ReLU / exp element-wise)**
直接 $\phi(\mathbf{z}) = \text{ReLU}(\mathbf{z})$。极快，$m = d$，但 attention pattern 变得很 flat，完全丢失 softmax 那种 "尖锐聚焦" 的特性。Paper 里 Fig. 2 用 VR navigation 形象展示了 —— ReLU variant 直接让 agent 撞墙，exp variant 被路边不相关物体分心。

参考：Performers paper https://arxiv.org/abs/2009.14794

---

## SARA 的核心 insight

**那个 Gaussian matrix $\mathbf{G}$ 凭什么必须是固定的？**

Performers 用 fixed random Gaussian，是因为它们追求 "无偏估计" 这个数学性质。无偏估计意味着对**任何输入分布**都成立，代价就是 variance 大、需要大 $m$。

但 robot policy 的 attention 输入分布是**固定的** —— 就是这个 specific task 的 image patches / point clouds / text tokens。在这个固定分布上，我们根本不需要无偏性，只需要**在这个分布上近似得好**。

所以：让 $\mathbf{G}$ learnable，用 task data fine-tune 它，让它 specialize 到当前 attention pattern。specialize 之后 $m = d$ 就够了，因为不再是在整个函数空间上做 universal approximation，而是在一个很窄的 manifold 上做 local approximation。

这就是 SARA 的全部核心：

$$\phi^{\text{SARA}}(\mathbf{z}) = \mathbf{v} \odot f(\mathbf{G}\mathbf{z})$$

- $\mathbf{G}$ 从 Gaussian init，然后 learnable
- $\mathbf{v}$ 是 learnable scaling，element-wise 调每一维的重要性
- $f$ 用 ReLU 就行（最简单最快）
- $m = d$，参数量和原始 $\mathbf{W}_Q, \mathbf{W}_K$ 一样

直觉类比：Performers 像是"用 Monte Carlo 模拟一个物理过程，要很多样本才准"；SARA 像是"用少量数据训练一个 surrogate model 去拟合这个物理过程的输出"。后者在 ML 里我们每天都在做，效果通常比 Monte Carlo 好得多。

---

## Up-training 到底是什么操作

这个名字起得好，因为它确实就是 "往上" fine-tune 一次：

1. 拿一个已经 train 好的 transformer policy（比如 RT-2 5B，已经在 robotic data 上 fine-tune 过）
2. 把里面所有 attention 模块的 softmax 换成 SARA-attention
3. $\mathbf{G}_Q, \mathbf{G}_K$ 用 Gaussian 随机初始化，$\mathbf{v}$ 用全 1
4. 在**同一个** robotic data 上继续 fine-tune 几百到几千步
5. 收工

就这么简单。没有重训，没有蒸馏，没有量化，没有剪枝。就是换个 attention kernel 然后让它 adapt 几步。

为什么能这么快 adapt？因为 SARA 的参数空间里**存在**一组参数能很好近似原来的 softmax attention（Theorem 3.3 证明的 existence）。Gradient descent 只要找到这附近就行，不需要从头学起。有点像把一个已经收敛的 model 的某个模块 "swap 成等价但更高效的实现"，然后让它微调一下消化这个 swap。

Paper 里 Fig. 4 显示 PCT 上 up-training 几乎是 immediate 的 —— 一开始 reward 就接近 original，然后稳住。这印证了 "SARA 参数空间和 softmax 参数空间很接近" 这个直觉。

---

## 实验结果用人话讲

### PCT (Point Cloud Transformer)

**任务**：用 Kuka 机械臂抓桌子上的东西。训练只见过 5 个物体（coke can, water bottle, eraser, banana, octopus toy），测试用没见过的 shape。

**结果**：200 次 AB-test，regular PCT 成功率 64%，SARA-PCT 成功率 75%。

**速度**：SARA-PCT 不论点云多大都稳定 ~100ms，regular PCT 随点云 quadratic 增长。

**值得注意**：SARA 这里**比原来还准**。这挺 surprising 的。我的猜测是 linear attention 本身有 regularization 效果，过滤掉了一些 overfitting 到特定物体的 spurious attention pattern。Paper 没解释清楚这个现象，算是个 open question。

### RT-2 (PaLI-5B VLA)

**任务**：7 类 manipulation（pick, knock, open/close drawer, drawer place, upright, move, diverse pick）。Diverse pick 是泛化测试。

**结果**（Table I）：

| 配置 | 平均准确率 | diverse pick |
|---|---|---|
| RT-2 原版 | 65.8% | 33% |
| SARA-RT-2 (只换 attention) | 65.1% | 48% |
| SARA-RT-2 + history + vector action | 76.4% | 67% |

**速度**：TPU 上 53ms → 45ms，14% speedup。

**值得注意**：
- 单纯换 SARA，平均精度几乎不变（65.8 → 65.1），但泛化能力涨了 15 个百分点（33 → 48）。这是一个很 clean 的 signal，说明 linear attention 在 generalization 上有结构性优势。
- 加上 history 和 vector action representation 后，整体涨 10.6%，pick task 满分。这里 SARA 贡献一部分，history + vector action 贡献一部分，paper 没 ablation 清楚两者各占多少。
- 14% speedup 看起来不大，但要知道 RT-2 的 bottleneck 不全在 attention。真正的 win 在 high-resolution image 和 long history，这些场景 regular RT-2 根本跑不了，SARA 可以。

参考：RT-2 详情 https://robotics-transformer2.github.io/

---

## 为什么 SARA 在泛化上更强

这是 paper 里没讲透但我觉得最有意思的点。

Softmax attention 有个特性：**它很容易产生极端 spiky 的 attention pattern**，一两个 key 拿走 90% 的 attention mass。这种 pattern 在 training distribution 上很 sharp，但 distribution 一变就容易崩 —— 因为那几个 "winner" keys 可能就不相关了。

Linear attention 的 attention distribution 天然更 flat，不会让单一 key 完全主导。这其实是一种 implicit smoothing / regularization。在 in-distribution 上可能损失一点精度，但在 out-of-distribution（diverse pick 的 novel objects）上反而更 robust，因为 attention 不会 all-in 到某个可能错误的 key 上。

这让我想起 attention head 上常见的 phenomenon：softmax attention 学到的 spiky pattern 经常是 spurious correlation，而 linear attention 的 flat pattern 更接近 "平均用好多个 hint" 的 ensemble 行为，泛化自然更好。

这个方向值得深挖。如果 SARA 在 LLM 上也有类似泛化优势，那价值远超 robotics。

---

## 这个工作真正在做什么 (meta 层面)

跳出来看，SARA 代表的是一种 transformer deployment 的新范式：

传统部署优化路线：
- **Quantization** (FP16 → INT8 → INT4)：减小数值精度
- **Distillation** (大模型 → 小模型)：训一个 student 模仿 teacher
- **Pruning** (去掉不重要的 head / weight)：稀疏化
- **Speculative decoding** (用小模型 draft 大模型 verify)：推理加速

SARA 开辟的路线：
- **Kernel structure swap**：保持模型架构和参数量，把计算 kernel 从 quadratic 换成 linear，然后 lightweight fine-tune 让它 adapt

这个范式的优点：
- 模型容量几乎不变（参数量一样）
- 只需要少量 fine-tune data（不需要重训）
- 速度提升是 structural 的（不是 approximate 的）
- 可以和其他优化（quantization, distillation）叠加

缺点：
- 目前只在 attention 上做，MLP 还是 $O(N \cdot d^2)$
- 精度提升依赖 fine-tune data 质量
- 理论上是 existence result，实际上能不能找到好参数要看 optimization landscape

如果这个范式能推到 LLM 上 —— 比如把 GPT-4 的 attention 换成 SARA 然后 fine-tune 几 billion token，得到一个 inference 快 2-3 倍但能力几乎不变的模型 —— 那价值是巨大的。目前没人证明这不行，只是没人试过。

---

## 几个值得深挖的联想

1. **SARA + FlashAttention 能叠加吗？** FlashAttention 是 hardware-aware 的 softmax attention 优化，不改变数学只改变 memory access pattern。SARA 是数学层面的 linearization。理论上两者作用层面不同，可以叠加，但实际实现上 SARA 已经把 attention 的 quadratic memory 拿掉了，FlashAttention 的 memory savings 就没意义了。所以两者是替代关系，不是叠加。

2. **SARA 能用在 long context LLM 上吗？** Long context (100K+ token) 的核心瓶颈就是 attention 的 quadratic。如果 SARA 能 fine-tune Llama-3-100K 的 attention 到 linear，保持 perplexity 几乎不变，那基本就是 long context 的 silver bullet。从 SARA-RT 的 evidence 看，技术上是可行的，up-training 几千步应该够。但 LLM 的 attention pattern 比 robot policy 复杂得多，$m = d$ 能不能 hold 住是个 open question。

3. **SARA 和 Mamba / state space models 的关系**。Mamba 这类 SSM 本质上也是一种 linear recurrence，可以看成 linear attention 的特殊形式。SARA 提供了一个 "把 softmax attention 转成 linear form 并保持精度" 的方法，这其实给了一条 "从 transformer 迁移到 SSM" 的路径 —— 用 SARA 做 intermediate step，然后进一步压缩到 SSM 架构。

4. **SARA 的数学和 neural tangent kernel 的联系**。Theorem 3.3 的证明思路是构造性存在性证明，这种 "在某个参数化空间里存在一组参数能近似目标函数" 的论述，和 NTK 理论里 "wide network 可以近似 kernel" 有结构上的相似。可能可以在 NTK framework 下重新解释 SARA 为什么 work。

参考：
- Mamba paper https://arxiv.org/abs/2312.00752
- FlashAttention https://arxiv.org/abs/2205.14135
- Linear attention survey https://arxiv.org/abs/2305.18748

---

## 一句话直觉

**把 softmax attention 想成一道需要现做的复杂菜，Performers 是用很多食材现做这道菜的 Monte Carlo 模拟（准但慢），简单 ReLU linear attention 是用速食版代替（快但难吃），SARA 是用少量样本 fit 一个菜谱让机器照着做（快且接近原味）。**

这就是 SARA-RT。剩下全是细节。

---

# SARA-RT 深度解读

## I. 核心问题与动机 (Motivation)

Robotics Transformers (RT) 这条路线最近几年爆发出三个里程碑：RT-1 (35M, ~3Hz) → RT-2 (1B+, VLA backbone 是 PaLI-X) → 各种 PC-based policies。这里隐藏着一个 deployment 层面的张力：**模型语义推理能力越强，通常意味着越深越宽的 Transformer，而 quadratic softmax attention 在 robot 上就是奢侈品**。35M 的 RT-1 跑到 3Hz 已经是 35M 量级，到了 RT-2 这种 PaLI-X 5B / 55B 的 scale，on-robot inference 几乎不可行。

SARA-RT 的本质问题：**能不能在不重训的情况下，把一个已经 train 好的 quadratic attention policy 改成 linear attention，而且不掉精度？** 这就是 "up-training" 这个名字的由来 —— pre-trained model 不是从零训，而是 "往上" fine-tune 一次把 attention 模块换掉。

参考链接：
- RT-2 paper: https://arxiv.org/abs/2307.15818
- RT-1 paper: https://arxiv.org/abs/2212.06817
- Performers (相近 idea 的 random feature baseline): https://arxiv.org/abs/2009.14794

---

## II. 直觉构建：从 zero-shot VR navigation 切入

作者很聪明地用一个 "macroscopic" 的 toy example 来 build intuition —— Matterport 3D 环境 + CLIP embeddings + 离散动作（点击图像 patch）。这不是 robotics manipulation 本身，但 attention 的本质问题在这里被放大可视化了。

### A. Standard attention 控制器

给定目标 $t_i$（image 或 text），agent 采取动作：

$$\mathbf{a}_i = \sum_{j=1}^{N} s(i,j) \bar{\mathbf{a}}_j, \quad s(i,j) = \frac{K(\mathbf{q}_i, \mathbf{k}_j)}{\sum_{l=1}^{N} K(\mathbf{q}_i, \mathbf{k}_l)} \tag{1}$$

变量解释：
- $N$ = 输入 image patchify 后的 patch 数（CLIP ViT-B/16 一般 196 patches for 224×224）
- $\mathbf{k}_j \in \mathbb{R}^{d_{QK}}$ = 第 j 个 patch 的 key embedding，CLIP 里 $d_{QK}=512$
- $\mathbf{q}_i \in \mathbb{R}^{d_{QK}}$ = target image/text 的 query embedding
- $K: \mathbb{R}^{d_{QK}} \times \mathbb{R}^{d_{QK}} \to \mathbb{R}$ = kernel (similarity function)
- $\bar{\mathbf{a}}_j$ = base action（左转/右转/前进/后退这种离散动作的 embedding）
- $s(i,j)$ = attention score，构成 categorical distribution

CLIP 用的是 softmax-kernel $K(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^\top \mathbf{y})$，**不需要任何 fine-tuning**就能 zero-shot 导航到目标（Fig. 2 黑色块）。这就是 paper 里那种 "spiky attention pattern" 的来源 —— softmax 的指数特性天然会让最相关 patch 拿到压倒性 attention。

### B. 计算瓶颈

如果目标数量 $M$ 很大（接近 $N$），整个 $M \times N$ attention matrix 就是 $O(MN)$ 的 space/time 复杂度。线性 attention 的核心 idea：如果 kernel 能写成

$$K(\mathbf{x}, \mathbf{y}) = \mathbb{E}[\phi(\mathbf{x})^\top \phi(\mathbf{y})] \tag{2}$$

那么 action 可以重写成：

$$\tilde{\mathbf{a}}_i = \frac{\Psi \phi(\mathbf{q}_i)}{\Gamma \phi(\mathbf{q}_i)}, \quad \Psi = \sum_{j=1}^{N} \bar{\mathbf{a}}_j \phi^\top(\mathbf{k}_j), \quad \Gamma = \sum_{j=1}^{N} \phi^\top(\mathbf{k}_j) \tag{3}$$

关键 insight：$\Psi$ 和 $\Gamma$ 都和 $i$ 无关，可以**一次性**预计算好（线性扫一遍 keys），然后每个 query 只需要 $O(m)$ 时间（$m$ 是 $\phi$ 输出维度）。整体复杂度掉到 $O((M+N) \cdot m)$，即 linear。

### C. 现有 linear attention 的两难

paper 这里做了一个非常干净的 taxonomy：

**A. 无偏估计路线 (Performers / random features)**
- $\phi$ = random Gaussian projection 后 exp
- 优点：unbiased estimate of softmax-kernel
- 缺点：$m$ 需要很大（~2048+）才有低 variance，所以只在 $M, N \geq 4K$ 时划算
- 缺点：通常仍有 perf gap

**B. 简单函数路线 ($\phi_f$, $f \in \{\text{ReLU, exp}\}$)**
- $\phi_f(\mathbf{z}) = (f(z_1), \ldots, f(z_{d_{QK}}))^\top$
- 优点：极快，$M, N$ 小到 128 就有 speedup
- 缺点：Fig. 2 粉框（ReLU）和绿框 直接证明 —— ReLU 撞墙、exp 被分心，attention score 分布是 flat 的，没有 softmax 的 "spike"

### D. Randomized preprocessing：用一个简单的 trick 救场

$$\phi_f^{\text{rand}}(\mathbf{z}) = f(\mathbf{G}\mathbf{z}), \quad \mathbf{G} \in \mathbb{R}^{m \times d_{QK}}, \quad G_{ij} \sim \mathcal{N}(0,1) \text{ i.i.d.}$$

蓝/棕色块 —— ReLU 和 exp variant 加 Gaussian matrix preprocessing 后都能正确导航。这是 paper 的关键 "motivating experiment"。直觉上：**raw embedding 维度之间是相关/耦合的，element-wise $f$ 在 raw space 里没办法 pick 出 softmax 的 diagonal-cumulative 行为；用随机 projection 把 embedding "spread" 到 $m$ 维独立 axis 上后，element-wise nonlinearity 就能 reassemble softmax 的 spiky 行为**。这其实就是 Performers 的 random feature 理论在 macro-scale 上的视觉化。

---

## III. SARA: 把 Gaussian matrix 变成 learnable

### A. SARA mapping 的定义

$$\phi_{f,1}^{\text{SARA}}(\mathbf{z}) = \mathbf{v} \odot f(\mathbf{G}_Q \mathbf{z}), \quad \phi_{f,2}^{\text{SARA}}(\mathbf{z}) = \mathbf{v} \odot f(\mathbf{G}_K \mathbf{z}) \tag{4}$$

变量解释：
- $\mathbf{z}$ = raw embedding (不是 query/key！)，维度 $d$
- $\mathbf{v} \in \mathbb{R}^m$ = learnable scaling vector，element-wise 乘（Hadamard product $\odot$）
- $\mathbf{G}_Q, \mathbf{G}_K \in \mathbb{R}^{m \times d}$ = 两个不同的 learnable matrices，分别对应 query 和 key 流
- $f$ = element-wise nonlinearity，可以是 ReLU / exp / sqrt
- 注意这里没有 $\mathbf{W}_Q, \mathbf{W}_K$ —— SARA 把它们吸收进 $\mathbf{G}_Q, \mathbf{G}_K$ 里了，$\mathbf{G}_Q \mathbf{z}$ 直接产出 $m$ 维 query feature

为什么有两个 mapping $\phi_1, \phi_2$ 而不是一个？因为 standard Transformer 里 query 和 key 用不同 projection $\mathbf{W}_Q, \mathbf{W}_K$。SARA 把这套搬到 random feature 框架下：query 走 $\mathbf{G}_Q$，key 走 $\mathbf{G}_K$，kernel 值 $K(\mathbf{x}_i, \mathbf{y}_j) \approx \phi_1^{\text{SARA}}(\mathbf{x}_i)^\top \phi_2^{\text{SARA}}(\mathbf{y}_j)$。

### B. 关键工程 trick: $m = d_{QK}$

Random Gaussian variant 需要很大的 $m$（~2048）才有低 variance，但 SARA 让 $\mathbf{G}$ learnable 后，$m$ 可以**等于** input dimension $d_{QK}$（RT-2 实验里就是 $m = d = 512$）。

直觉：**random Gaussian matrix 是 universal approximator，但它要在 distribution 上平均无偏；learnable matrix 可以 specialize 到这个具体 task 的 attention matrix 上，所以需要的 dimensionality 大幅下降**。这其实和 "over-parameterization 不是必须的，只要参数化的 basis 选对了" 这个 general principle 一致。

### C. Up-training 流程

具体步骤：
1. Start from already-trained Transformer policy（e.g. RT-2 5B 已经 fine-tuned on robotic data）
2. Replace softmax attention with SARA-attention (initialize $\mathbf{G}_Q, \mathbf{G}_K$ from Gaussian, $\mathbf{v}$ from all-ones)
3. Fine-tune on the same downstream robotic data
4. 几百到几千 steps 后就 converge（Fig. 4 显示几乎 immediate adaptation）

直觉：因为 softmax-kernel 和 SARA-kernel 在 $m=d$ 下存在一组参数能很好近似，up-training 的 fine-tuning 主要是在做 **distillation-like optimization** —— 把预训练 model 的 attention pattern 蒸到 linear attention 的参数空间里。这正是为什么 "up-training" 名字合适：参数量几乎不变，但计算结构升级了。

---

## IV. 数学分析：为什么 SARA 能近似 softmax attention

### A. Lemma 3.1: random exp variant 是 unbiased 的

对于 L-2 normalized 输入 $\|\mathbf{x}\| = \|\mathbf{y}\| = r$：

$$m \cdot \exp(r^2) \cdot K(\mathbf{x}, \mathbf{y}) = \mathbb{E}\left[(\phi_{\text{exp}}^{\text{random}}(\mathbf{x}))^\top \phi_{\text{exp}}^{\text{random}}(\mathbf{y})\right] \tag{5}$$

变量：
- $m$ = random feature 维度
- $r$ = 输入向量的 L-2 范数
- $K(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^\top \mathbf{y})$ = softmax kernel

证明利用 Performers paper 的 positive random features：

$$\phi^+(\mathbf{z}) = \frac{1}{\sqrt{m}} \exp\left(-\frac{\|\mathbf{z}\|^2}{2}\right) \exp(\mathbf{G}\mathbf{z}) \tag{6}$$

直觉：$\exp(-\|\mathbf{z}\|^2/2)$ 这一项是 "normalization factor"，用来平衡 $\exp(\mathbf{G}\mathbf{z})$ 的指数增长，让 $\phi^+(\mathbf{x})^\top \phi^+(\mathbf{y})$ 的期望就是 $\exp(\mathbf{x}^\top \mathbf{y})$。当输入是 L-2 normalized 时，$\exp(-\|\mathbf{z}\|^2/2) = \exp(-r^2/2)$ 是常数，可以从期望里提出来。

### B. Lemma 3.2: concentration bound

$$\mathbb{P}\left[\left|\frac{(\phi_{\text{exp}}^{\text{random}}(\mathbf{x}))^\top \phi_{\text{exp}}^{\text{random}}(\mathbf{y})}{m \exp(r^2)} - K(\mathbf{x}, \mathbf{y})\right| > g_{r,t}^m\right] \leq \frac{1}{t^2} \tag{7}$$

其中 $g_{r,t}^m = \frac{t}{\sqrt{m}} \exp(r^2(2\cos\theta+1)) \sqrt{1 - \exp(-2r^2(1+\cos\theta))}$，$\theta$ 是 $\mathbf{x}, \mathbf{y}$ 之间的夹角。

直觉：
- $\theta = 0$（同向）：$\cos\theta = 1$，kernel 值最大，估计也最 stable
- $\theta = \pi$（反向）：$\cos\theta = -1$，kernel 值最小
- $1/t^2$ 是 Chebyshev bound 的标准形式

### C. Theorem 3.3: SARA 的核心近似定理

这是 paper 的 theoretical centerpiece。给定 normalized attention layer，存在 $\mathbf{v}, \mathbf{G}_1, \mathbf{G}_2, f$ 使得 approximate attention matrix $\hat{\mathbf{A}}$ 满足：

$$\|\mathbf{A} - \hat{\mathbf{A}}\|_\infty \leq \delta \tag{8}$$

且参数数量（$m \cdot d$ 量级）只需要 logarithmic in $MN$。

具体 $m$ 的取值：

$$m = \left\lceil \frac{2\rho^2}{\delta^2 \tau^2} \log(2MN) \exp\left(-\frac{r^2}{A}\right) \right\rceil + 1, \quad A < 0, \delta > 0 \tag{9}$$

变量解释：
- $\tau = \min_{i,j} K(\mathbf{q}_i, \mathbf{k}_j)$ = attention matrix 最小 kernel 值
- $\rho = \max_{i,j} K(\mathbf{q}_i, \mathbf{k}_j)$ = 最大 kernel 值
- $r$ = query/key 的 L-2 范数
- $A$ = 一个负的 scaling 参数（控制 normalization 强度）
- $\delta$ = 误差容限

证明关键点：构造性地给出参数：

$$\mathbf{G}_1 = \sqrt{1-4A}\, \mathbf{G}\mathbf{W}_Q, \quad \mathbf{G}_2 = \sqrt{1-4A}\, \mathbf{G}\mathbf{W}_K$$

$$\mathbf{v} = (1-4A)^{d_{QK}/4} (\exp(A\|\mathbf{g}_1\|^2), \ldots, \exp(A\|\mathbf{g}_m\|^2))^\top$$

其中 $\mathbf{G} \in \mathbb{R}^{m \times d_{QK}}$ 是标准 Gaussian matrix，$\mathbf{g}_1, \ldots, \mathbf{g}_m$ 是其 row。直觉：**SARA 把 Performers 的 random Gaussian matrix 和 Transformer 的 learnable $\mathbf{W}_Q, \mathbf{W}_K$ 合二为一，加上 learnable scaling vector $\mathbf{v}$，使得整套机制可以 approximate softmax attention，而且参数量只和 $\log(MN)$ 相关**。

为什么 $\log(MN)$？因为 Chebyshev + union bound 给出 $p_\epsilon \leq MN \cdot r_{\tau\epsilon}$，要让 $p_\epsilon < 1$，需要 $r_{\tau\epsilon} < 1/MN$，即 $\exp(-m\tau^2\epsilon^2/2 \cdot \exp(r^2/A)) < 1/MN$，开对数得 $m \gtrsim \log(MN)$。

---

## V. Experiments: 两类 robot policy

### A. Point Cloud Transformer (PCT)

#### 1. Setting 细节

- 输入：RealSense camera → pass-through filter → hierarchical clustering → 单物体 PC
- Observation: $(N \times 3 \text{ cloud}, (x,y,z) \text{ center}, (x,y,z) \text{ major axis})$
- Action: fingertip position + approach direction + wrist roll
- Hardware: Kuka IIWA arm + Weiss gripper
- Training: blackbox optimization (BGS variant)，50 perturbation directions, $\sigma=0.02$, $\eta=0.02$, top 30% directions
- 只见到 5 个物体训练：coke can, water bottle, chalkboard eraser, banana, octopus soft toy
- 测试用训练时没见过的 shape

#### 2. Up-training 结果 (Fig. 4)

三种 $f$ 都试了：$\{\exp, \text{ReLU}, \text{sqrt}: x \to x^2\}$。注意 sqrt 这里其实是平方，paper 写法可能有 typo 或者从不同视角命名。所有 variant 几乎 immediate adapt 到 high-reward 区域，**up-training 起始 checkpoint 就已经接近 regular PCT 的最终 performance**。

#### 3. AB-test on real robot (Fig. 5)

200 个随机 object configurations，每 config 随机选 SARA-PCT 或 regular PCT 执行：

| Policy | Average reward |
|---|---|
| Regular PCT | $r_{\text{ave}}^{\text{reg}} = 0.64$ |
| SARA-PCT (ReLU) | $r_{\text{ave}}^{\text{SARA}} = 0.75$ |

**SARA 不仅不降低，反而提高了 11 个百分点**。这其实蛮 surprising，作者也没给出明确解释。我的推测：
1. Linear attention 本质上是 implicit regularization，过滤掉了一些过拟合到 specific objects 的 attention pattern
2. Up-training 的额外 fine-tuning step 等价于在已收敛 model 上做了一次额外的 regularization pass
3. 200 个 trial 的 sample size 还不算巨大，0.64 vs 0.75 的差距可能有一定 noise

#### 4. Speed test (Fig. 6)

- SARA-PCT: $\sim 100$ ms near-constant，regardless of PC size
- Regular PCT: 随 PC size quadratic 增长

这是 linear attention 的 expected 行为 —— kernelization 把 attention bottleneck 拿掉了，剩下的 MLP 和 IO 成本是 dominant 且 near-constant 的。

### B. RT-2 (PaLI-5B VLA)

#### 1. Architecture overview (Fig. 7)

PaLI-5B backbone 包括：
- **ViT encoder** (sViT for SARA variant): 把 image patchify + self-attention
- **Text Transformer (TT)**: 处理 text instruction
- **Fuser**: concatenate ViT 和 TT 输出，再做 self-attention

paper 注：他们只把 SARA 注入 ViT，留 fuser 给 future work。这其实是个 reasonable choice —— ViT 处理 $L=196$ tokens（$14 \times 14$ patches），fuser 处理 concat 后的 sequence 也只有 200 量级，ViT 更 long。

#### 2. Action representation

两种 representation 对比：

**a. Action tokens (RT-2 原版)**
- 每个连续 dimension 量化为 256 bins
- 7 tokens 串行（6-DoF + gripper + terminate）
- 用 VLM 自带 tokenizer tokenize

**b. Vector representation (SARA-RT-2 新提)**
- 每个 dimension round 到 4 decimal places
- 拼成字符串，用 text tokenizer 处理
- 实验上更准确（见 Table I 第三行 vs 第一行）

直觉：256 bins 量化损失精度，且 7 个 token 顺序建模有 dependency 问题；vector representation 让 VLM 直接看到数字字面值，更像它 pre-training 时见到的 text distribution。

#### 3. Table I 关键数据

| Variant | pick | knock | open/close drawer | drawer place | upright | move | diverse pick | mean |
|---|---|---|---|---|---|---|---|---|
| RT-2 (no hist + action tokens) | 81% | 86% | 67% | 39% | 57% | 98% | 33% | 65.8% |
| SARA-RT-2 (no hist + action tokens) | 83% | 91% | 78% | 31% | 46% | 79% | 48% | 65.1% |
| SARA-RT-2 (H=3 + vector rep) | 100% | 91% | 89% | 56% | 51% | 81% | 67% | 76.4% |

观察：
- **Row 1 vs Row 2**: SARA 单独换 attention 后，平均只掉 0.7%，但 diverse pick（泛化任务）涨了 15%
- **Row 3 vs Row 1**: 加 history + vector representation 后，平均涨 10.6%，pick task 满分 100%
- diverse pick 是泛化能力的关键指标，SARA 在泛化上始终更强

直觉上：linear attention 在泛化上的优势可能源于它对 attention pattern 的 implicit smoothing —— 不会让单一 key 主导整个 attention distribution，从而对 training distribution 之外的输入更 robust。

#### 4. Speed test (Fig. 8)

- TPU: regular RT-2 53.2 ms → SARA 45.7 ms（14% speedup）
- CPU 上 ViT 不同 resolution: SARA 在 high-resolution 仍然 feasible，regular 直接 blow up

14% 看起来不大，但要注意 RT-2 已经是 PaLI-5B，整个 inference 时间里 attention 只是 ViT 的一部分，且 $L=196$ 并不算特别大。**真正 win 在 high-resolution image 和 large history 上** —— regular RT-2 根本无法 scale 到这些 regime，而 SARA 可以。

---

## VI. 直觉性总结：SARA 到底做了什么

把 SARA 放到 linear attention 的 taxonomy 里看：

| Method | $\phi$ form | Learnable | Unbiased? | $m$ required | Quality |
|---|---|---|---|---|---|
| Performers | $\exp(\mathbf{G}\mathbf{z})$ | No (Gaussian) | Yes | Large (2048+) | Moderate gap |
| Linear Attention (ReLU) | $\text{ReLU}(\mathbf{z})$ | No | No | $d$ | Significant gap |
| **SARA** | $\mathbf{v} \odot f(\mathbf{G}_Q \mathbf{z})$ | **Yes** | Approximate | **$d$** | **接近 softmax** |

SARA 的核心 contribution 可以理解为：**把 random Gaussian matrix 从 "用于 unbiased estimation" 这个用途，重新定位为 "作为 learnable basis" 的初始化**。Random matrix 给了良好的 inductive bias（和高斯分布相关的 JL lemma 等性质），但 fine-tuning 让它 specialize 到当前 task 的 attention pattern 上，从而用 $m = d$ 达到 $m = 2048$ 的 random version 的效果。

更深层 intuition：softmax attention 之所以强，是因为 $\exp$ 在 dot product 上的 combinatorial spiking 行为。直接用 $\exp$ element-wise 缺少了 "对 dot product 敏感" 这个性质（因为 element-wise 操作看不到 cross-dimension）。Random projection $\mathbf{G}\mathbf{z}$ 把 cross-dimension 信息 "spreading" 到不同 axis，然后 element-wise $\exp$ 就能 reconstruct 出类似 dot-product-spike 的 behavior。Learnable $\mathbf{G}$ 进一步 sharpen 这个 reconstruction。

---

## VII. 个人几点 critical thought

1. **Theoretical analysis 和实际实现有 gap**：Theorem 3.3 是 existence result，构造的 $\mathbf{v}, \mathbf{G}_1, \mathbf{G}_2$ 用了 $\sqrt{1-4A}$ 这种特殊形式，但实际 up-training 用的是 random init + gradient descent。Theory 主要证明 "代表性存在"，不是 "可达性"。

2. **Fuser 没换**：Fig. 7 明确说 fuser 也是 attention bottleneck 的候选，但实际只换了 ViT。Fuser 的 sequence length 在 3-frame history 下大约 $196 \times 3 + \text{text tokens} \approx 600$，确实比 ViT 内部 self-attention 长。future work 可能能再榨一些 speedup。

3. **PCT 的 0.75 vs 0.64 解释模糊**：作者只是陈述事实，没解释为什么 linear attention 更强。可能是 regularization effect，也可能只是 AB-test 200 sample 的 noise。

4. **没有和 Performers 直接对比**：paper 里 mathematical section 大量引用 Performers，但实验上只比了 regular PCT 和 regular RT-2。如果 random Performers $m=2048$ 能达到什么程度？没数据。这其实挺关键的 —— 因为 SARA 的 selling point 之一是 $m=d$ 比 $m=2048$ 划算，需要 empirical evidence。

5. **Generalization 在 diverse pick 上的 +15% 是真信号**：这个不是 noise 能解释的，且 trend 一致（SARA 在 row 2 和 row 3 都比 row 1 强）。这暗示 linear attention 在泛化上有结构性优势，值得 future work 深挖。

6. **Action representation 的 vector format 是隐藏 contribution**：row 1 → row 3 的 10.6% 提升，SARA 和 vector rep 各占一部分，paper 没有很好的 ablation 分离两者。但从 row 1 vs row 2 看，SARA 只贡献 0.7%（mean）但 15%（diverse pick），剩下大部分 mean 提升来自 history + vector rep。

---

## VIII. 相关延伸阅读

- **Performers (FAVOR+)**: https://arxiv.org/abs/2009.14794
  SARA 的理论基础，random feature view of softmax kernel

- **Performer# / Chefs' random tables**: https://arxiv.org/abs/2302.00787, https://arxiv.org/abs/2205.15317
  Choromanski 团队后续 non-trigonometric random features 工作

- **cosFormer**: https://arxiv.org/abs/2110.04978
  另一个 linear attention 变体，用 cosine similarity 重写 softmax

- **Linear Transformers are Secretly Fast Weight Programmers**: https://arxiv.org/abs/2102.11174
  Schlag et al. 对 linear attention 的 RNN 视角解释

- **RT-2 (Vision-Language-Action)**: https://arxiv.org/abs/2307.15818
  SARA 直接加速的模型

- **PaLI-X**: https://arxiv.org/abs/2305.18565
  RT-2 5B 的 VLM backbone

- **RT-1**: https://arxiv.org/abs/2212.06817
  Robotics Transformer 的开端

- **PCT (Point Cloud Transformer)**: https://arxiv.org/abs/2012.09188
  另一个被加速的目标

- **Habitat-Matterport 3D**: https://arxiv.org/abs/2109.08238
  Zero-shot navigation 实验用到的 environment

- **Random Features for Large-Scale Kernel Machines** (Rahimi & Recht, 2007): https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
  Random feature methods 的奠基工作

---

## IX. 一个具体的 mental model

如果让我把 SARA 用一句话解释给一个学生：**"SARA 是把 softmax attention 的 random feature approximation 中那个固定 Gaussian matrix 替换为 learnable matrix，并用 up-training fine-tune 到下游 task 上，从而让 linear attention 在保持 softmax attention 行为的同时把 $O(N^2)$ 复杂度降到 $O(N)$"**。

进一步 mental model：把 attention 想成一个 black-box function $A(\mathbf{Q}, \mathbf{K}, \mathbf{V})$，random feature 方法是 "用 Monte Carlo 估计这个 function"，linear attention with ReLU 是 "用一个 crude approximation 代替这个 function"，SARA 是 "用少量 data fit 一个 learnable surrogate function"。最后一个在 ML 里最 common 也最 powerful，所以 SARA 工作得好也不奇怪。

最重要的 takeaway 其实是 paper 没明说的：**pre-trained transformer 里的 attention module 本身就是 task-specific 的，可以无缝 swap 为 linear attention 然后做 lightweight fine-tuning**。这给 transformer deployment 一个新的范式 —— 不是量化、不是蒸馏，而是 kernel structure swap。如果 future work 能把这个推到 LLM 上，影响会大得多。
