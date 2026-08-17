---
source_pdf: Equilibrium Reasoners Learning Attractors Enables Scalable Reasoning.pdf
paper_sha256: 986b4a6a150b08a42d2e42fc3e0f5965ff8d654e001dd18f216bf85f959783c6
processed_at: '2026-08-04T04:44:36-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话说清楚

模型做推理的时候，反复琢磨同一道题，到底有没有用？这篇paper说：**有用，但前提是模型得学会往正确的"坑"里滚**。

---

## 一个直觉比喻

想象你往一个起伏的山坡上扔弹珠。弹珠会滚啊滚，最后停在某个坑底。

- 坑 = attractor（吸引子）
- 滚的过程 = iteration（反复更新）
- 停在哪个坑 = 最终答案

传统模型（feedforward）相当于扔一次珠子看它落哪，不管落得对不对就这么定了。

Iterative model 相当于让珠子多滚一会儿，给它更多时间找到坑。

但问题来了：**山坡上有很多坑，有的坑对应正确答案，有的坑对应错误答案**。珠子多滚一会儿，可能滚到对的坑，也可能滚到错的坑里更深处，越滚越错。

所以"想更多"本身不解决问题。关键是**山坡的形状得对**——正确答案的坑得够大、够好找，错误答案的坑得尽量少或者尽量浅。

这就是这篇paper全部的核心。

---

## 为什么传统模型不行

Table 1 里有个特别扎眼的数字：feedforward model 哪怕堆到64层，Sudoku准确率只有2.6%。

但training accuracy有93.8%。

这说明什么？**模型把训练集背下来了，但没学会做题的逻辑**。就像一个学生把答案全背了，考试遇到新题就傻了。

换成weight-tied模型（同样的block反复用21次，参数量从105M降到5M），准确率从2.6%跳到32.6%。

为什么反复用同一个block比堆不同层好？因为反复用同一个block相当于逼模型学一个**"怎么改答案"的规则**，而不是学一个"从题到答案"的映射。前者能泛化，后者只能背。

---

## 但是光反复用还不够

你可能会想：那把iteration从21次加到336次（16倍），肯定更好吧？

结果：直接backprop会OOM（内存爆炸），换成detached carry（只监督最后一步），准确率51.8%，比21次的51.3%几乎没提升。

为什么？因为你只告诉模型"最后答案要对"，但中间21到336步全没人管。模型不知道中间那些步该干嘛。

这就像让学生做一道很长的题，只看最终答案对不对，中间过程完全不管。学生可能猜对最终答案，但没学会中间的推理过程。

---

## SOT：分段监督，边走边改

作者的解法叫 Segmented Online Training（SOT），思路特别朴素：

**把长轨迹切成段，每段做完就检查答案、更新模型，然后从当前位置继续下一段。**

对比三种做法：
- Vanilla：走完336步，看最后答案，改模型 → 51.8%
- Trajectory supervision：走完336步，中间好几个点都看答案，但最后才一起改模型 → 47.1%（反而更差！）
- SOT：每走一段就改模型，再用新模型走下一段 → 74.7%

为什么 trajectory supervision 反而更差？因为模型参数在走的过程中没变，中间那些点是在"旧模型"下生成的，但最后更新用的是"新模型"的gradient。相当于你让学生在旧教材上做题，然后用新教材的标准答案批改，对学生很confusing。

SOT好在哪？模型走一段，改一下自己，再走下一段。每一段都是在当前最新模型下走的，监督和更新是匹配的。

---

## 两个trick让"坑"更好找

即使SOT把准确率提到84.8%，还有很大提升空间。问题是：正确答案的坑可能太小、太难找，或者错误答案的坑太多。

作者用了两个task-agnostic的trick：

### Trick 1: Randomized Initialization (RI)

以前模型每次都从同一个起点出发。这就像每次都从山坡同一个位置扔珠子——只能探索附近那一小块区域。

RI改成每次从随机位置出发。训练时就这样做，逼模型学会"不管从哪开始，都能滚到对的坑"。

效果：Maze准确率从44.9%直接跳到68.6%。因为Maze的landscape对起始点特别敏感，RI大幅增加了正确basin的reachable范围。

### Trick 2: Noise Injection (NI)

在每一步update里加一点小噪音：

$$
\mathbf{z}_{k+1} = \mathbf{z}_k + (1-\lambda)(f_\theta(\mathbf{z}_k; \mathbf{x}) - \mathbf{z}_k) + \beta \varepsilon_k
$$

用话说：珠子滚的时候地面不是完全光滑的，有点小颠簸。

这有什么用？如果珠子快滚进一个错误的浅坑，小颠簸能把它弹出来，让它继续找更深的坑。如果珠子已经在正确的深坑里，小颠簸晃不动它。

效果：Sudoku从86.0%（只加RI）到86.4%（加RI+NI），Maze从68.6%到82.2%。Maze提升特别大，说明Maze的landscape里有很多spurious shallow attractor需要跳出。

---

## 推理时怎么花compute

训练好了之后，推理时有两条路花更多compute：

**Depth scaling**：一条珠子滚更久。如果珠子已经进了对的坑，多滚一会儿能让它更稳。如果珠子在坑边晃，多滚能帮它settle。

**Breadth scaling**：同时扔B颗珠子，从不同随机起点出发，最后看哪个滚得最稳（residual最小），信它的答案。

关键发现：breadth scaling在depth太小时没用。因为珠子还没滚到任何坑就停了，扔再多颗也白扔。得depth够大（约4次以上iteration），breadth才开始有用。

最终结果：Sudoku 99.8%，Maze 93.0%。而feedforward model只有2.6%。

---

## Convergence能当selection signal吗

扔了B颗珠子，每颗都说自己滚到一个坑了。信谁？

两种策略：
- Majority vote：看大多数珠子的答案
- Top-1 Converged：看哪颗珠子滚得最稳（residual最小），信它

关键发现：**Top-1 Converged在EqR上比majority vote好，但在baseline TRM上不如majority vote**。

为什么？因为TRM的landscape没塑造好，residual小可能只是滚进了一个错的深坑。EqR的landscape被RI+NI塑造过，residual小真的意味着滚进了对的坑。

所以convergence signal不是万能的——它是learned proxy，可靠性取决于landscape shape。这就像考试时"写满整张卷子"不一定分数高，但如果这个学生平时训练得好，写满通常意味着答得全。

---

## 难题多想，简单题少想

ACT（Adaptive Computation Time）：学一个小head预测"我是不是已经做完了"。如果模型觉得自己已经settle了，就提前停。

效果：D=1024时，平均NFE从1024降到58.7（省17倍compute），准确率只降0.8%。

用话说：大部分题其实很简单，几步就做完了。只有一小部分难题需要反复想很久。模型学会了判断"我还要不要继续想"。

---

## 一个重要教训：数据集本身要well-posed

Maze-1k数据集每个maze有多条最短路径，但只标注了一条。这导致模型学到一堆互相竞争的shallow attractor——每条valid path都是一个"对"的坑，但loss只认可其中一个。

结果：depth scaling不work，RI和NI反而counterproductive。因为扩大任何一个坑的reachable范围都是在帮"错"的坑（从loss角度看）。

作者构造了Maze-Unique（每题只有一条最短路径），问题立刻消失了。

教训：**如果你的任务有多个正确答案，但监督信号只指定一个，模型会学到一个misaligned的landscape**。这不是模型capacity的问题，是数据集定义的问题。

---

## 整篇paper的takeaway

1. **反复想 ≠ 越想越好**。得让模型学会往正确的方向想。
2. **训练时要分段监督+边走边改**（SOT），不能只在最后看答案。
3. **随机起点+小噪音**（RI+NI）让模型学会"不管从哪开始都能找到对的路"。
4. **推理时depth和breadth互补**：depth让单条trajectory settle，breadth增加coverage。但breadth要depth够大才有效。
5. **Convergence是learned proxy**，不是universal certificate。能不能用取决于landscape塑造得好不好。
6. **数据集得well-posed**。多解任务配单解监督，landscape就废了。

一句话：**reasoning不是想更多，是想得更稳。稳不稳取决于山坡的形状，形状是训练塑造的**。

---

# Equilibrium Reasoners: 详细技术讲解

## 1. Core Insight: 从 dynamical systems 视角理解 iterative reasoning

这篇 paper 的核心 contribution 是把 iterative reasoning models 重新 interpret 成一个 **task-conditioned attractor dynamical system**。作者的关键 insight 是：test-time scaling 之所以有效，是因为模型内部学习到的 attractor landscape 与 task-metric landscape 对齐了。

直觉上，你可以把 iterative reasoning 想象成在一个 latent space 中"下山"的过程。传统的 feedforward model 是一次性给出答案，而 iterative model 是反复更新 latent state z，就像一个 ball 在一个 energy landscape 上滚动，最终停在某一个 basin 里。如果这个 basin decode 出来正好是正确的 solution，那 test-time compute 就有意义；如果停在一个 spurious basin 里，那再多 iterations 也救不了。

Reference: 原始 paper 在 https://github.com/locuslab/EqR ；相关 DEQ 工作在 https://arxiv.org/abs/1909.01377 。

---

## 2. 形式化框架：Attractor 的数学定义

### 2.1 基本更新规则

EqR 的基本 update operator 是：

$$
\mathbf{z}_{k+1} = f_{\theta}(\mathbf{z}_k ; \mathbf{x})
$$

变量解释：
- $\mathbf{z}_k \in \mathbb{R}^n$：第 $k$ 步的 latent state，n 是 latent dimension
- $\mathbf{x} \in \mathcal{X}$：input condition（例如 Sudoku 的 puzzle）
- $f_{\theta}$：参数为 $\theta$ 的 update operator（通常是一个 weight-tied block）
- $k$：iteration index

这个 formulation 直接来源于 DEQ-style perspective。区别在于 DEQ 要求严格收敛到 fixed point $\mathbf{z}^* = f_\theta(\mathbf{z}^*; \mathbf{x})$，而 EqR relax 了这个要求——只要求 trajectory 收敛到一个 **attractor**，也就是 stable long-run outcome，可以是 fixed point 也可以是 small recurrent set。

### 2.2 Attractor landscape

作者定义 $\mathcal{Z}_\theta^*(\mathbf{x})$ 为从 initialization $\mathbf{z}_0$ 出发、通过反复迭代 $f_\theta$ 得到的 stable long-run outcomes 的集合。这个集合就是 model 的 **attractor landscape**。

这个 landscape 有两个关键属性：

**(i) Task alignment**：reached attractor decode 出来是不是正确答案？如果是 spurious attractor，即使 residual 很小，prediction 也是错的。

**(ii) Reachability**：从不同 initialization 出发，trajectory 能多可靠地到达 correct attractor？

这两个属性直接 map 到两个 test-time scaling levers：
- **Depth scaling (D)**：增加单条 trajectory 的 iteration 数，让 trajectory 有更多机会 refine 到它已经进入的 basin
- **Breadth scaling (B)**：从 B 个 independent initialization 出发，aggregate 它们的输出，增加覆盖不同 basin 的机会

总 compute budget 用 NFE = D · B 衡量。

### 2.3 Residual 作为 convergence diagnostic

Fixed-point residual 定义为：

$$
R_\theta(\mathbf{z}; \mathbf{x}) = \mathbf{z} - f_\theta(\mathbf{z}; \mathbf{x})
$$

它的范数 $\|R_\theta(\mathbf{z}; \mathbf{x})\|$ 衡量当前 state 离 fixed point 有多近。

关键定理（Appendix A.1）：假设 basin 内存在 fixed point $\mathbf{z}^* = f_\theta(\mathbf{z}^*; \mathbf{x})$，且 $f_\theta$ 在该 basin 内是 L-Lipschitz 的（$L < 1$），那么：

$$
\|\mathbf{z} - \mathbf{z}^*\| \leq \|R_\theta(\mathbf{z}; \mathbf{x})\| + L\|\mathbf{z} - \mathbf{z}^*\|
$$

整理得：

$$
\|\mathbf{z} - \mathbf{z}^*\| \leq \frac{\|R_\theta(\mathbf{z}; \mathbf{x})\|}{1 - L}
$$

变量解释：
- $\|\mathbf{z} - \mathbf{z}^*\|$：当前 state 到 attractor 的距离
- $L$：Lipschitz 常数，控制 contraction rate
- $(1-L)$ 在分母上，说明 L 越接近 1，同样的 residual 对应的实际距离越大（系统越"软"）

这个 bound 的 intuition 是：在 contraction basin 内，small residual ⟹ close to attractor。但 residual 小 ≠ 答案对——你还需要 task alignment。

### 2.4 Margin condition 把 residual 连到 correctness

为了让 residual 真正能作为 correctness 的 proxy，还需要 output margin。定义 minimum margin：

$$
\gamma(\mathbf{z}^*) = \min_i \left[ s_{\theta, i, y_i}(\mathbf{z}^*; \mathbf{x}) - \max_{a \neq y_i} s_{\theta, i, a}(\mathbf{z}^*; \mathbf{x}) \right]
$$

变量解释：
- $s_{\theta, i, a}$：output location $i$ 处 label 为 $a$ 的 logit
- $y_i$：正确 label
- $\gamma(\mathbf{z}^*)$：在 attractor 处，正确 label 与最强竞争 label 的最小 logit gap

如果 logits 是 $G_{gap}$-Lipschitz 的，那么 sufficient condition for correctness 是：

$$
\|R_\theta(\mathbf{z}; \mathbf{x})\| < (1-L) \frac{\gamma(\mathbf{z}^*)}{G_{gap}} \quad \Longrightarrow \quad \hat{\mathbf{y}}_\theta(\mathbf{z}; \mathbf{x}) = \mathbf{y}
$$

这个公式很重要，它告诉你：**residual 是 correctness proxy 的条件是 (a) local stability, (b) correct attractor, (c) positive output margin**。三个条件缺一不可。低 residual 在 spurious attractor 附近只是 certify convergence，不是 certify correctness。

---

## 3. Bilevel Optimization 视角

EqR 的 training 目标可以写成 bilevel optimization：

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, \mathbf{y}), \mathbf{z}_0} \left[ \ell_\theta(\mathbf{z}_\theta^*(\mathbf{x}, \mathbf{z}_0); \mathbf{x}, \mathbf{y}) \right]
$$

$$
\text{s.t.} \quad \mathbf{z}_\theta^*(\mathbf{x}, \mathbf{z}_0) := \text{Solve}_\theta^{(D)}(\mathbf{z}_0; \mathbf{x}), \quad \|\mathbf{z}_\theta^* - f_\theta(\mathbf{z}_\theta^*; \mathbf{x})\|_2 \leq \varepsilon_{\text{res}}
$$

变量解释：
- $\ell_\theta$：supervised loss（例如 cross-entropy）
- $\text{Solve}_\theta^{(D)}$：D 步的 lower-level rollout，即反复 apply $f_\theta$ D 次
- $\varepsilon_{\text{res}}$：residual tolerance

这是一个经典 bilevel 结构：upper level 优化 $\theta$ 让 reached state decode 出正确答案，lower level 是 forward rollout。

### 3.1 Implicit gradient 的 conditioning 问题

在 exact fixed point 处微分 $R_\theta(\mathbf{z}^*; \mathbf{x}) = 0$，得到：

$$
(I - J_{\mathbf{z}}) d\mathbf{z}^* = J_\theta d\theta
$$

$$
\frac{d\mathbf{z}^*}{d\theta} = (I - J_{\mathbf{z}})^{-1} J_\theta
$$

变量解释：
- $J_{\mathbf{z}} = \partial_{\mathbf{z}} f_\theta(\mathbf{z}^*; \mathbf{x})$：state-to-state Jacobian
- $J_\theta = \partial_\theta f_\theta(\mathbf{z}^*; \mathbf{x})$：parameter-to-state Jacobian
- $(I - J_{\mathbf{z}})^{-1}$：resolvent operator

如果这个 resolvent poorly conditioned，small parameter change 会导致 large attractor shift，且 lower-level solve 的 approximation error 会被 implicit gradient 放大。这就是为什么作者需要 truncated gradient 和 SOT——不仅仅是省 memory，更是在 latent trajectory 跟随 changing attractor landscape 的过程中保持 optimization local。

参考 TorchDEQ 实现：https://github.com/locuslab/torchdeq 。

---

## 4. Construction Path: 从 Feedforward 到 Iterative

Table 2 给出了一个 clean 的 ablation chain，每一步加一个 ingredient：

| Method | Blocks | Param (M) | NLE | Train Acc. | Eval Acc. |
|---|---|---|---|---|---|
| vanilla feedforward | 42 | 105.6 | 42 | 93.8 | 2.6 |
| + weight-tied | 2 | 5.03 | 42 | 94.5 | 32.6 |
| + SOT + depth×16 | 2 | 5.03 | 672 | 94.9 | 74.7 |
| + hierarchical recurrence | 2 | 5.03 | 672 | 99.3 | 76.5 |
| + ACT training | 2 | 5.03 | 672 | 82.2 | 84.8 |

### 4.1 Weight-tying 的作用

Weight-tied model 用 2 个 block 反复 21 次，参数从 105.6M 降到 5.03M，eval accuracy 从 2.6% 升到 32.6%。这个 gap 很 striking——同样的 layer-evaluation budget 下，weight-tied 远好于 distinct layers。

直觉解释：weight-tied 把 "depth" 变成了 "iteration"，这相当于让 model 学习一个 dynamics 而非一个 static mapping。Feedforward model 在 42 层时可以 fit training set（93.8%），但 evaluation 极差（2.6%），这是典型的 memorization without generalization。Weight-tied 强制 model 学一个能反复 refine 的 operator，这种 inductive bias 更适合 reasoning 任务。

### 4.2 Depth scaling 的 limit

把 iteration depth 翻倍（2× → 16×）有两个问题：
1. Memory explosion：full backprop through long trajectory
2. Recurrent Jacobian product can explode or vanish

简单 detach carry（terminal loss only）能解决 memory 问题，但只能到 51.8%，相比 2× 的 51.3% 几乎没提升。这说明仅仅把 trajectory 拉长，但只在末尾做 supervision，intermediate states 没人管，model 学不到有用的 intermediate refinement。

### 4.3 Segmented Online Training (SOT) 是关键

SOT 的核心 idea 是把 long trajectory 切成 segments，每个 segment 末端做 loss + optimizer step，然后从 detached carry 继续下一段。公式化（Eq. 10）：

$$
\tilde{\mathbf{z}}_{s+1}(\theta) = f_\theta^{(h)}(\mathbf{z}_s; \mathbf{x})
$$
$$
g_s = \nabla_\theta \ell_\theta(\tilde{\mathbf{z}}_{s+1}(\theta); \mathbf{x}, \mathbf{y}) \big|_{\theta = \theta_s}
$$
$$
\theta_{s+1} = \theta_s - \eta g_s, \quad \mathbf{z}_{s+1} = \text{stopgrad}(\tilde{\mathbf{z}}_{s+1}(\theta_s))
$$

变量解释：
- $\mathbf{z}_s$：上一段 detached 的 carry state
- $h$：segment 长度
- $f_\theta^{(h)}$：h 步 rollout
- $g_s$：local gradient
- $\eta$：learning rate

关键 difference vs. trajectory supervision：trajectory supervision 在 trajectory 末端做一次 update，但所有 anchors 是在 stale parameters 下生成的；SOT 在每段后立即 update parameters，下一段在 updated parameters 下生成。这是 alternating approximation：latent updates seek reachable low-residual state under current operator，parameter updates reshape operator。

#### SOT 的 tracking error bound

Eq. 11 给出：

$$
e_{s+1} \lesssim \rho e_s + \kappa_\theta \|\theta_{s+1} - \theta_s\|
$$

变量解释：
- $e_s = \|\mathbf{z}_s - \mathbf{z}_{\theta_s}^*\|$：carried state 与当前 operator 的 attractor 的距离
- $\rho < 1$：local contraction rate
- $\kappa_\theta \approx \|(I - J_{\mathbf{z}, s})^{-1} J_{\theta, s}\|$：attractor map 对 parameter 的 sensitivity

第一项是 latent correction（contract），第二项是 parameter update 引起的 attractor shift。SOT 在两者之间做 alternating。

实验结果：terminal loss 51.8% → trajectory supervision 47.1%（反而更差） → SOT 74.7%。Trajectory supervision 之所以差，是因为 stale trajectory 与 updated operator 不匹配，相当于让 transient states 都去 match 一个 unreachable target。

### 4.4 Late anchors 比 full anchors 好

Table 7 的 ablation 很有意思：

| Supervision | Anchors | Acc. |
|---|---|---|
| terminal loss | 16 | 51.80 |
| full anchors | 1:16 | 47.10 |
| late anchors | 8:16 | 51.36 |
| late anchors | 12:16 | 57.50 |

Intuition：trajectory 早期是 transient state，supervise 它们会与 dynamics 冲突；trajectory 后期已经进入 basin，supervise 后期相当于在 reliable gradient 上做更大 effective step size。

### 4.5 Hierarchical iterations 的 interaction effect

Hierarchical iterations（HRM/TRM 的核心）有两个 latent state $\mathbf{z}_H$ 和 $\mathbf{z}_L$，在不同时间尺度上更新。但 Table 8(d) 显示它的 effect 很难 decouple：

| Latents | w/ grad trunc | Acc. |
|---|---|---|
| z | 21 | 74.7 |
| z + trunc. | 7 | 67.2 |
| (z_L, z_H) | 21 | 69.8 |
| (z_L, z_H) + trunc. | 7 | 75.4 |

Single latent 在无 truncation 时强，hierarchical 在有 truncation 时强。这说明 hierarchy 不是 standalone switch，它依赖于 surrounding training recipe。

---

## 5. Landscape Shaping 的两个 intervention

### 5.1 Randomized State Initialization (RI)

HRM 和 TRM 默认用 fixed initial state $\mathbf{z}_0$。EqR 改为 sample：

$$
\mathbf{z}_0 \sim \mu_0(\cdot \mid \mathbf{x}) = \mathcal{N}(0, \sigma_0 I)
$$

两个 benefit：

**(i) Coverage**：fixed initializer 只 shape 一个小 neighborhood；random $\mathbf{z}_0$ 扩大 explored region，增加 correct attractor 在 inference 时 reachable 的概率。

**(ii) Path independence**：同一个 $(\mathbf{x}, \mathbf{y})$ 在多个 initialization 下被观察到，divergent prediction 被惩罚，这 encourages path independence（参考 https://arxiv.org/abs/2205.13587 ）。

Table 11 显示 path independence $\Delta_{PI}$ 从 TRM 的 3.58% (Sudoku) 降到 +RI 的 0.10%，再降到 +RI+NI 的 0.13%。

关于 noise scale 的 ablation：best setting 是 $\sigma_H = 1, \sigma_L = 8$，达到 87.30%。这说明 randomized initialization 在 moderate noise 下最有用，overly large perturbation 反而会略微降低 accuracy。

Learnable initializer（$\mathbf{z}_0 = g_\phi(\mathbf{x})$）在 Table 10 中显示**没有**提升，83.99% vs. RI 的 86.03% vs. baseline 的 84.06%。这是一个 negative result，作者诚实地承认了。可能与 diffusion model 中的 golden noise（https://arxiv.org/abs/2411.03070 ）形成对比——diffusion 中 learnable init 有用，但 reasoning task 中 simple Gaussian 反而更好。可能原因：reasoning 的 attractor landscape 对 init 的 sensitivity 不像 diffusion 那样可被 learnable prior 捕捉。

### 5.2 Path Stochasticity via Noise Injection (NI)

带 damping 和 noise 的 update：

$$
\mathbf{z}_{k+1} = \mathbf{z}_k + (1 - \lambda) r_\theta(\mathbf{z}_k; \mathbf{x}) + \beta \varepsilon_k
$$

变量解释：
- $r_\theta(\mathbf{z}_k; \mathbf{x}) = f_\theta(\mathbf{z}_k; \mathbf{x}) - \mathbf{z}_k$：residual update
- $\lambda \in [0, 1)$：damping coefficient，控制 update step 大小（$\lambda=0$ 时无 damping，$\lambda \to 1$ 时 update 趋于 0）
- $\beta \geq 0$：noise magnitude
- $\varepsilon_k \sim \mathcal{N}(0, I)$：isotropic Gaussian noise

$\lambda = 0.05$（mild damping）+ $\beta = 0.01$（small noise）效果最好。

NI 的作用机制：mild noise 可以帮 trajectory 跳出 premature trapping，进入更好的 basin。这直接对应 Sec. 4.2 中的 mode (b) "correct and spurious attractors coexist" 和 mode (c) "correct but hard to reach"。

Connection to SAM (Sharpness-Aware Minimization, https://arxiv.org/abs/2010.01412 )：作者在 Appendix E 明确指出这个类比。SAM 在 parameter space 找 flat minima（对 weight perturbation 鲁棒），EqR 在 state space 找 robust attractor（对 latent perturbation 鲁棒）。Noise injection 类似 SAM 的 perturbation step，force model 学习 smooth attractor landscape。

这是一个很漂亮的 conceptual bridge：把 parameter-space robustness 的 intuition 迁移到 state-space dynamics。

---

## 6. 两轴 Test-Time Scaling

### 6.1 Depth vs. Breadth 的 interaction

Figure 3 的 Pareto heatmap 是 paper 的 key figure 之一。关键观察：

- Breadth scaling 在 $D \lesssim 4$（约 168 layers）时几乎无效
- Breadth scaling 在 large D 时 consistently 减少 residual 和 prediction error
- 整个 grid 上 lower prediction error 与 smaller residual 强相关

Intuition：breadth 通过 restart 增加不同 basin 的 coverage，但 restart 后 trajectory 需要足够的 depth 才能"探索并 settle"。如果 depth 太小，所有 restart 都来不及进入任何 basin，breadth 就只是浪费 compute。

### 6.2 Convergence-based selection vs. Majority vote

在 breadth scaling 下有两种 aggregation 策略：

**Majority vote**：从 B 个 restart 中取 mode
**Top-1 Converged**：选 residual 最小的那个 restart 的 prediction

Table 4 显示 EqR + depth scaling + breadth scaling 在 Sudoku 上达到 99.8%，Maze 上 93.0%。

关键发现：Top-1 Converged 在 EqR 上 outperform majority vote（Figure 8），但在 baseline TRM 上不如 majority vote。原因：TRM 的 residual 不与 correctness 对齐（可能 converge 到 spurious attractor），而 EqR 的 landscape 被塑造得使 residual 与 correctness 对齐。

Top-1 Converged 的 selection rule 用最后 L=3 步的平均 residual：

$$
r_{T,L}^{(i)}(\mathbf{x}) = \frac{1}{L} \sum_{t=T-L+1}^{T} \|f_\theta(\mathbf{z}_t^{(i)}; \mathbf{x}) - \mathbf{z}_t^{(i)}\|
$$

选 $i^* = \arg\min_i r_{T,L}^{(i)}(\mathbf{x})$，然后 report $\hat{\mathbf{y}}^{(i^*)}$。

### 6.3 ACT 实现预算弹性

Table 5 显示 ACT 在 D=1024 时把 Avg. NFE 从 1024 降到 58.7（17.4× 减少），accuracy 只从 96.1% 降到 95.3%。这说明大部分 instance 早早就 converge 了，只有一小部分需要长跑。

Eq. 7 的 margin condition 给出理论解释：简单 instance 的 $\gamma(\mathbf{z}^*)$ 大，少量 iteration 就能让 residual 低于 threshold；难 instance 的 margin 小，需要更多 iteration 才能 reduce residual 到 correctness zone。

ACT 的 halting score 是 $\hat{q}_k = f_\phi(\mathbf{z}_k)$，halt condition 是 $\tau = \min\{k \leq K : \hat{q}_k > \delta\}$。

---

## 7. 四种 Attractor Landscape Modes

Figure 6 的四种 mode 是 paper 的 conceptual core：

| Mode | 描述 | Failure source | Effective lever |
|---|---|---|---|
| (a) | No correct attractor | Task misalignment | Neither |
| (b) | Correct + spurious coexist | Basin selection | Breadth |
| (c) | Correct but narrow basin | Reachability | Breadth + weak depth |
| (d) | Well-aligned | (No failure) | Depth |

直觉 mapping：
- Mode (a)：landscape 根本没 shape 对，再多 compute 也救不了
- Mode (b)：multiple attractors 争夺，需要 restart 增加命中正确 basin 的概率
- Mode (c)：correct basin 太窄，需要 restart 增加进入概率，depth 帮 weak trajectory settle
- Mode (d)：ideal case，单条 trajectory 沿 basin 下滑即可

RI 和 NI 主要 mitigate mode (b) 和 (c)：RI 扩大 correct basin 的 reachable region，NI 帮 trajectory 跳出 premature trapping。

---

## 8. Maze-Unique 的 ill-posedness 教训

Appendix C.2 是一个 important lesson。Maze-1k 数据集每个 maze 的 shortest path 不唯一，但 dataset 只提供一个 target path。这导致：

- Task 有 multiple correct attractors
- Loss 把其中一个 arbitrarily 指定为唯一 target
- 其他 valid path 被惩罚
- Learned landscape 有多个 shallow competing attractors
- Depth scaling 不稳定，RI/NI 反而 counterproductive

作者构造 Maze-Unique（perfect maze，unique shortest path），recover stable attractor dynamics。

这是一个 meta-level insight：**数据集的 well-posedness 决定了 attractor landscape 能否被学习**。当 task admit 多 valid solution 但 supervision 只指定一个时，问题是 ill-posed 的，attractor landscape 无法稳定形成。

---

## 9. Cost Accounting

Table 9 给出详细的 cost 比较。关键 symbol：

- $c_\ell$：one local loss/head backward
- $c_B$：segment parameter-backward
- $c_B^{\text{trunc}}$：truncated segment parameter-backward
- $c_J$：temporal state-backward through segment
- $c_\theta$：extra shared-gradient accumulation
- $c_u$：parameter update for one recurrent block
- $a_f$：activation memory for one segment
- $P$：parameter-side memory

Full gradient training through T-step trajectory：

$$
C_{\text{full}}(T) = c_\ell + T c_B + (T-1) c_J + (T-1) c_\theta + c_u
$$
$$
M_{\text{full}}(T) = T a_f + a_\ell + P
$$

Detached carry 移除 temporal state-backward chain：

$$
C_{\text{det}}(T) = c_\ell + c_B + c_u
$$
$$
M_{\text{det}}(T) = a_f^{\text{det}} + a_\ell + P
$$

SOT 把 optimizer interval 从 full T-step trajectory 变成 single outer-loop step：

$$
C_{\text{SOT}} = c_\ell + c_B + c_u, \quad M_{\text{SOT}} = a_f^{\text{det}} + a_\ell + P
$$

这里的关键观察：detached carry 的 cost 与 trajectory 长度 T 无关，这就是为什么 16× depth 在 memory 上 feasible。

---

## 10. 与相关工作的关系

### 10.1 与 DEQ 的区别

DEQ（https://arxiv.org/abs/1909.01377 ）把 representation 定义为 fixed point，主要用 convergence 作为 representation learning 和 training device。EqR 借用 fixed-point vocabulary 但问不同的问题：**learned latent dynamics 是否能让 convergence 对 solving task 可靠？**

DEQ 中 reaching any fixed point 就够了；EqR 中 reaching a fixed point 不够——必须 shape 一个 landscape，其中 large stable basins 对应 correct solutions 而非 spurious attractors。

### 10.2 与 HRM/TRM/URM 的关系

- HRM（https://arxiv.org/abs/2503.08186 估计）：Hierarchical Reasoning Model，nested latent updates
- TRM（https://arxiv.org/abs/2505.18787 估计）：Tiny Recursive Model，简化版
- URM：Universal Reasoning Model

EqR 在 TRM backbone 上加 RI + NI + 两轴 scaling，不改 architecture。Table 3 显示从 TRM 84.8% → EqR 86.4%（Sudoku），TRM 44.9% → EqR 82.2%（Maze）。Maze 的巨大提升（+37.3%）主要来自 RI，这符合 coverage hypothesis。

### 10.3 与 SAM 的类比

如前所述，EqR 把 flat minima 的 intuition 从 parameter space 迁移到 state space。SAM 找对 weight perturbation 鲁棒的 minima，EqR 找对 latent perturbation 鲁棒的 attractor。Noise injection 类似 SAM 的 perturbation step。

### 10.4 与 Universal Transformer 的区别

Universal Transformer（https://arxiv.org/abs/1807.03819 ）用 input token embeddings 初始化 recurrent state，shared block 在该 state 上 iterate。EqR 的 update 是 input-conditioned solver $f_\theta(\mathbf{z}; \mathbf{x})$，problem data $\mathbf{x}$ 在每一步都作为 external condition 可用，更像 DEQ 和 recurrent-depth models（https://arxiv.org/abs/2502.05129 估计）。

### 10.5 与 Coconut / SoftCoT / PonderLM 的区别

这些是 horizontal latent reasoning——在 visible output 前插入额外 latent positions。EqR 是 vertical fixed-state dynamics，研究 extra updates 何时有用，而不是把 CoT 压缩到 latent tokens。

---

## 11. 我的直觉总结

这篇 paper 的 deep insight 在于：**reasoning 不是关于"想更多"，而是关于"想得更稳"**。Test-time scaling 的价值不来自额外 compute 本身，而来自额外 compute 是否能让 trajectory 更接近正确 attractor。

几个关键 takeaways：

1. **Weight-tying 创造 iterative capacity，但 capacity 的 realization 需要训练策略**。简单地加深 iteration 没用，必须在 SOT 这种 alternating scheme 下才能让 intermediate states 学到有用的 refinement。

2. **Residual 是 diagnostic，不是 certificate**。低 residual 只在 (a) local stability, (b) correct attractor, (c) positive margin 三条件下才 imply correctness。这让 convergence-based selection 成为 learned proxy，其 reliability 依赖 landscape shape。

3. **Attractor landscape 的两个 axis（alignment, reachability）对应两个 scaling lever（depth, breadth）**。这给出了一个 task-agnostic 的 diagnostic framework：如果你观察到 depth scaling 不 work，可能 landscape 是 mode (a) 或 (b)；如果 breadth 不 work，可能 basin 已经够宽（mode d）或 trajectory 不够深（depth 太小）。

4. **数据集 well-posedness 决定 landscape 能否被学习**。Maze-1k 的 ill-posedness 直接破坏了 attractor dynamics。这是一个 meta-level lesson：在研究 reasoning model 时，数据集本身的 structure 比 model capacity 更基础。

5. **State-space robustness 是 parameter-space robustness 的 dual**。SAM 在 weight space 找 flat minima，EqR 在 latent space 找 robust attractor。这个 bridge 是 conceptual contribution，可能启发新的 regularization 方法。

潜在 future direction：
- Learnable initializer 在 reasoning task 上的失败提示 init prior 设计需要新思路，可能需要 task-structure-aware 的 parameterization
- Multi-attractor landscape 的 well-posed task（如 Maze-1k）如何处理——可能需要 distributional supervision 而非 single-target
- Attractor landscape 的几何性质（basin volume, basin boundary fractal dimension）的量化诊断
- 与 implicit differentiation 的更深 connection——EqR 的 SOT 是一种 approximate implicit gradient，可能可以与 JFB（https://ojs.aaai.org/index.php/AAAI/article/view/20619 ）等 Jacobian-free 方法结合

Reference:
- 原始 paper: https://github.com/locuslab/EqR
- DEQ: https://arxiv.org/abs/1909.01377
- Path-independent equilibrium: https://arxiv.org/abs/2205.13587
- Universal Transformer: https://arxiv.org/abs/1807.03819
- SAM: https://arxiv.org/abs/2010.01412
- TorchDEQ: https://github.com/locuslab/torchdeq
- HRM: https://arxiv.org/abs/2503.08186 (approximate)
- TRM: https://arxiv.org/abs/2505.18787 (approximate)
- Recurrent depth (Geiping et al.): https://arxiv.org/abs/2502.05129 (approximate)
- Golden noise: https://arxiv.org/abs/2411.03070
- JFB: https://ojs.aaai.org/index.php/AAAI/article/view/20619
- Flat minima (Hochreiter & Schmidhuber): https://direct.mit.edu/neco/article-abstract/9/1/1/5510/Flat-Minima

希望这个讲解能帮到你 build intuition。如果某个部分想更深入（比如 SOT 的 implementation detail、attractor landscape 的 visualization 方法、或者 ACT halting head 的训练），可以继续问。
