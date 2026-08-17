---
source_pdf: GeoWorld Geometric World Models.pdf
paper_sha256: 37e7ed1b195d28356bdb92033d03c0242f63f629022164464435e2b8bb6e7f9a
processed_at: '2026-08-04T21:37:19-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GeoWorld 用人话说

Andrej, 咱们抛开公式，用一个 thought experiment 把 GeoWorld 讲透。

---

## 一句话概括

V-JEPA 2 把"预测未来"这件事做对了——它不生成像素，而是学一个 energy landscape。但 V-JEPA 2 的 latent space 是 flat 的 Euclidean 几何，跟现实世界的 hierarchical 结构不匹配。GeoWorld 说：**等等，世界状态转移本质上是一棵指数分支的树，我们干脆把 latent space 搬到 hyperbolic space 里去，因为 hyperbolic space 天生就是用来装树结构的。**

---

## 为什么要换几何？

想象你站在厨房里，状态是"要做饭"。你可以做炒饭、做汤、做沙拉... 然后做炒饭又分: 先打蛋、先切菜、先热锅... 每一步又有不同细节。

从 "开始做饭" 这个状态往前看 5 步，可能的状态数是 action space 的指数——就像一棵不断分叉的树。

**关键问题**: 这种树结构塞进 Euclidean space 会怎样？

Euclidean space 的体积随半径多项式增长 ($\sim r^n$)，而树有 $B^d$ 个叶子。要嵌入树，越往深层节点越多，Euclidean space 根本"放不下"，必然 distortion。你在 Euclidean latent space 里看到的"距离"已经不是真实的层级关系。

**Hyperbolic space 的魔法**: 它的体积随半径指数增长 $\sim e^{(n-1)r}$，跟树的指数分支天然匹配。Nickel & Kiela 2017 年的 Poincaré Embeddings 就是为了 embed WordNet hierarchy 才搞出来的 ([参考](https://arxiv.org/abs/1705.08039))。

所以 GeoWorld 的第一直觉: **与其在 flat space 里硬编码 hierarchy，不如换一个 native 容纳 hierarchy 的几何空间。**

---

## 换几何具体怎么换？

V-JEPA 2 encoder 吐出来一个向量 $s \in \mathbb{R}^n$，在 Euclidean space 里。GeoWorld 把它当成 tangent vector（切空间里的向量），用一个叫 exponential map 的操作把它"投影"到 Poincaré ball 上。

Poincaré ball 是什么？一个开单位球（半径 $1/\sqrt{c}$，其中 $c$ 是 curvature 参数）。球心是 hierarchy 的 root（最抽象的 coarse state），球边界是 hierarchy 的 leaves（最细节的 fine state）。

这个 projection 的核心是 $\tanh$：

$$s_{\mathbb{H}} = \tanh(\sqrt{c}\|s\|) \cdot \frac{s}{\sqrt{c}\|s\|}$$

- $\tanh$ 把向量的长度压缩到 $[0, 1)$
- 越靠近球边界，等量的 Euclidean 移动对应越大的 hyperbolic distance（指数爆炸）
- 所以 hierarchy 的浅层节点（root）在球心附近，彼此 hyperbolic 距离小；深层节点（leaves）在球边界附近，彼此 hyperbolic 距离大

直觉地讲：**$\tanh$ 就像把一张平坦的地图拉伸到球面上——中心几乎不变，但边缘被指数级放大。这个"放大"恰好对应 hierarchy 深层节点的指数级增多。**

---

## 为什么 hyperbolic distance 不同于 Euclidean distance？

这是 paper 的数学心脏。Poincaré ball 上两点的 hyperbolic distance：

$$d_\mathbb{H}(u, v) = \frac{1}{\sqrt{c}} \text{arcosh}\Bigl(1 + 2c \frac{\|u-v\|^2}{(1 - c\|u\|^2)(1 - c\|v\|^2)}\Bigr)$$

- $\|u\|, \|v\|$ 是 Euclidean norm（必须 $< 1/\sqrt{c}$，即在 ball 内）
- 分母 $(1 - c\|u\|^2)$ 当 $\|u\| \to 1/\sqrt{c}$ 时趋零
- 所以当点接近边界时，hyperbolic distance 趋无穷——边界对应"无穷远"
- 这就是 hierarchy：越深的节点彼此越远

**Euclidean space 的距离有上限**（$2/\sqrt{c}$），**hyperbolic space 的距离无上限**。这就是为什么 hyperbolic 能装下任意深的 hierarchy，而 Euclidean 装不下。

---

## 训练在干什么？

V-JEPA 2 训 predictor：给当前 latent 和 action，预测 next latent。Loss 是 Euclidean $L_1$ 距离。

GeoWorld 把 loss 换成 hyperbolic distance：

- **Teacher forcing loss**: 让 predictor 一步预测的 latent 和 ground truth 在 hyperbolic space 里尽量近
- **Rollout loss**: 让两步 rollout 的预测也尽量近，防止 autoregressive 误差累积

加上一个 loss 权重 $\lambda$ 平衡，$\lambda = 0.5$ 效果最好。

**直觉**: 在 hyperbolic space 里最小化 distance 等价于让预测轨迹沿 geodesic（最短路径）走，而不是 Euclidean 空间里乱漂。

---

## Geometric Reinforcement Learning (GRL) 是什么？

这是 paper 第二个核心创新。问题：SFT 训完 predictor 仍可能在长 horizon 上崩——误差累积。怎么 fix？

**RL 的思路**: 把"未来 cumulative reward"定义成 value function，optimize policy 让 value 最大。

**GRL 的 trick**: 不需要单独的 policy 和 reward model。直接说：
- **Reward = negative hyperbolic distance** (预测离目标越近，reward 越高)
- **Value function = cumulative discounted reward over T steps**
- **Policy 就是 predictor 自己**（参数 $\phi$）

$$V^* = \max_\phi \mathbb{E}\Bigl[\sum_{t=1}^T \gamma^{t-1} r_t\Bigr] = \min_\phi \mathbb{E}\Bigl[\sum_{t=1}^T \gamma^{t-1} d_\mathbb{H}(\hat{s}_{t+1}, s_{t+1})\Bigr]$$

最大化 reward 等价于最小化累积 hyperbolic distance——让 trajectory 沿 geodesic 走。

**再加一个 triangle inequality regularization**：hyperbolic distance 满足三角不等式，所以连续三步 $t, t+1, t+2$ 应该满足 $d(t, t+2) \le d(t, t+1) + d(t+1, t+2)$。如果 trajectory 真的沿 geodesic，等号成立。GRL 用这个作为 soft penalty，push trajectory 越来越接近 geodesic。

**直觉总结**: GRL = 在 hyperbolic latent space 上做的 model-based RL，reward 来自 energy landscape 自身，policy 和 world model 共享参数。这跟 LeCun 2022 年那篇 "Path Towards Autonomous Machine Intelligence" 设想的 energy-based model 既是 world model 又是 value function 是同一个哲学 ([参考](https://openreview.net/pdf?id=BZ5a1r-kVsf))。

---

## Inference 时怎么 plan？

训练完后，给定起始 observation $x_1$ 和目标 $x_{1+T}$：

1. 用 encoder + $\exp_0$ 把它们投到 hyperbolic latent：$s_{1,\mathbb{H}}$ 和 $s_{1+T,\mathbb{H}}$
2. 用 **Cross-Entropy Method (CEM)** 搜 action sequence
3. CEM 是 zero-order optimization：从高斯分布采 800 个 action sequence，用 predictor 想象每个 sequence 的未来 latent，按到目标的 hyperbolic distance 排序，取 top-80 elite，更新高斯分布参数，迭代 10 次
4. 最后取最优 action sequence 的第一个 action 执行，然后 receding horizon 重新规划

这跟 V-JEPA 2 inference pipeline 一模一样，**唯一区别是把 Euclidean $L_1$ 换成 hyperbolic distance $d_\mathbb{H}$**。所以 GeoWorld 是 minimal modification 但几何上 principled 的改进。

---

## 实验上发生了什么？

**Procedural Planning (image → image)** on CrossTask, T=3:

- V-JEPA 2 ViT-g384: SR 45.58, mIoU 69.42
- GeoWorld ViT-g384: SR 47.47, mIoU 86.55

**注意 mIoU 提升 +17.13**——比 SR 提升 (+1.89) 大得多。这说明 GeoWorld 不仅仅预测正确的 action 顺序，还更好捕捉到 procedure 整体结构。

**Visual Planning with Videos (video → video)** on CrossTask, T=3:

- GPT-5 (zero-shot): SR 50.03
- Gemini 2.5 Pro: SR 48.91
- V-JEPA 2 ViT-g384: SR 50.16
- **GeoWorld ViT-g384: SR 51.71**

**GeoWorld 超过 GPT-5**——一个 frozen encoder + 300M predictor beat frontier-scale VLM。这是 paper 最 sharp 的结果。

**Long-Horizon Planning (CrossTask videos)**:

| Horizon | V-JEPA 2 | SFT (Hyp) | GRL (Euc) | GRL (Hyp) | SFT+GRL |
|---|---|---|---|---|---|
| T=3 | 50.16 | 50.42 | 50.26 | 51.04 | 51.71 |
| T=6 | 16.88 | 16.97 | 17.03 | 17.82 | 18.26 |
| T=8 | 4.95 | 11.51 | 12.74 | 13.10 | 13.81 |

**关键观察**:
- T=3 时所有方法都接近 ~50%
- T=8 时 V-JEPA 2 崩到 4.95%，GeoWorld 还维持 13.81%——**2.79× 优势**
- Hyperbolic geometry 单独 (SFT only) 已经能把 T=7 从 8.26 拉到 14.88
- GRL 单独（在 Euclidean）也能拉到 15.12
- 两者结合最强 (16.09)

**这说明 hyperbolic geometry 和 GRL 是两个正交的 fix**：一个改 geometry 约束 error 怎么 propagate，一个改训练信号直接消除 error。

---

## Energy Landscape 可视化为什么重要？

Paper Figure 2 是 intuition building 的关键。在 COIN 的 "Replace Memory Chip" 任务初始 step，扫两个 tangent-space 方向画 energy surface：

- **V-JEPA 2 (Euclidean)**: 几乎对称的 smooth paraboloid，弱 directional structure——CEM 在上面走几乎像 random search
- **GeoWorld (Hyperbolic)**: sharper, curvature-aware basin，方向性更强——CEM 沿 steep gradient 快速收敛到 geodesic

**为什么这重要**: CEM 是 zero-order planner，它沿 energy gradient 走。Flat landscape 上 CEM 慢且容易陷入 local minima；structured landscape 上 CEM 自然 follow geodesic，planning 又快又准。

---

## Curvature 学到了什么？

Curvature $c$ 从 $c=1$ (最 hyperbolic) 逐渐降到 $c \approx 0.3$ 并稳定。

**直觉**: CrossTask/COIN 的 task hierarchy 没那么极端，适度 curvature 就够。太 sharp 的 hyperbolic 反而 distort representation。这说明 **learnable curvature 让模型自己决定需要多强的 hierarchical inductive bias**——这是 Chami et al. NeurIPS 2019 HGNN 的 trick ([参考](https://arxiv.org/abs/1910.12923))。

---

## 跟 V-JEPA 2 比到底改了什么？

**改动极小但 principled**:
1. Encoder 输出加一个 $\exp_0$ projection layer（lightweight，可学习 curvature）
2. Loss 从 Euclidean $L_1$ 换成 hyperbolic distance $d_\mathbb{H}$
3. 加一个 GRL 阶段，把 energy 当 reward 做 RL 优化
4. Inference 时 CEM 用 hyperbolic distance 评价

**没有改的**:
- Encoder 架构（仍用 V-JEPA 2 frozen ViT）
- Predictor 架构（300M transformer）
- CEM 算法本身
- 数据 pipeline

所以 GeoWorld 是在 V-JEPA 2 之上做一个几何层面的 minimal intervention，但获得了长 horizon planning 的大幅提升。

---

## 这篇 paper 在拼图中的位置

LeCun 路线: **JEPA → I-JEPA → V-JEPA → V-JEPA 2 → GeoWorld**

- **JEPA** (LeCun 2022) 是 vision: latent space 预测代替 pixel generation ([参考](https://openreview.net/pdf?id=BZ5a1r-kVsf))
- **V-JEPA 2** (Meta 2025) 把它 scale 到 million-hour video，并加 action-conditioning 做 planning ([参考](https://arxiv.org/abs/2506.09985))
- **GeoWorld** 给 latent space 加 geometric structure，让 planning 在长 horizon 上稳定

下一步逻辑是什么？把这套推到 embodied setting——用 Droid dataset ([参考](https://droid-dataset.github.io/))，在 robot manipulation 任务上做 hierarchical planning。Paper Section 7 明确说这是 future work。如果 curvature 在 embodied task 上学到更 sharp 的值（因为 manipulation hierarchy 更 pronounced），那就真正证明 hyperbolic geometry 的 power。

---

## 我对这篇 paper 的整体看法

**优点**:
- 问题 motivation 干净——predictive world model 的 Euclidean bias
- Solution principled——hyperbolic geometry 是 hierarchy 的 native 容器
- Ablation 详尽——SFT vs GRL、Euclidean vs Hyperbolic、curvature 动态都拆开看
- Long horizon T=8 上 2.79× 优势是硬核结果
- Beat GPT-5 这个数字虽然 setting 不完全公平，但很 striking

**遗憾**:
- Predictor transformer 内部还是 Euclidean operation，没用 fully hyperbolic neural network (像 Möbius linear)——只在 input/output boundary 做 $\exp_0 / \log_0$ projection。所以 hyperbolic geometry 的 inductive bias 没完全贯穿到 model 内部
- Hierarchy 的 intuition 来自 multi-step future expansion（$B^d$），不是 explicit sub-task hierarchy（high-level task → mid action → low end-effector）。这跟原始 LeCun JEPA 设想的 multi-level hierarchy 不同
- 只在 visual planning 验证，没在真正 robotics 上跑

**如果让我改进**: 把 predictor 内部换成 fully hyperbolic transformer (用 Möbius addition 代替 Euclidean addition, Fréchet mean 代替 arithmetic mean)，然后在 Droid 上做 embodied planning experiment。这才是 GeoWorld 这个 program 的 fully realized version。

---

## 参考链接

**核心 paper**:
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2302.14202

**Hyperbolic representation**:
- Poincaré Embeddings: https://arxiv.org/abs/1705.08039
- Hyperbolic Neural Networks: https://arxiv.org/abs/1805.09112
- HGNN: https://arxiv.org/abs/1910.12923

**Planning & baselines**:
- CrossTask: https://arxiv.org/abs/1812.00818
- COIN: https://arxiv.org/abs/1903.02875
- DDN: https://arxiv.org/abs/2007.14030
- VideoWorld: https://arxiv.org/abs/2505.01140
- CEM tutorial: https://link.springer.com/article/10.1007/s10479-005-5723-z

**Future direction (embodied)**:
- π0: https://arxiv.org/abs/2410.24164
- Droid dataset: https://droid-dataset.github.io/

希望这个"人话版"让你对 GeoWorld 有了 cleaner 的 intuition。想继续深挖哪块（比如 fully hyperbolic transformer 怎么实现，或者 GRL 跟 DPO/RLHF 的关系），尽管问。

---

# GeoWorld: Geometric World Models 深度解析

你好 Karpathy! 这篇 paper 非常对你的胃口——它把 LeCun 的 JEPA 范式和 hyperbolic representation learning 缝合到一起，想要回答一个很本质的问题：**为什么 predictive world models 在 Euclidean latent space 里做长 horizon planning 会退化？** 这篇 work 的核心 intuition 是——world state transitions 本质上构成一棵指数分支的树，而 hyperbolic space 恰好是树结构 "native" 的几何容器。

---

## 1. Paper 的问题动机 (Motivation)

### 1.1 Predictive world models 的两难

先回顾 V-JEPA 2 (Meta FAIR, 2025) 的核心思路：它不生成 pixel，直接在 latent space 学习一个 energy landscape $F(s_t, s_{t+1:T})$，planning 等价于

$$\text{Plan} = \arg\min_{\text{actions}} F(s_t, s_{t+1:T})$$

低 energy = plausible future。CEM (Cross-Entropy Method) 这种 sampling-based planner 在这个 landscape 上搜索 action sequence。这跟 generative world models (Sora-style, VideoWorld) 的范式对立——后者必须 decode 像素，且依赖 inverse dynamics model (IDM) 只能做 one-step reactive control，没法看到 trajectory 全局结构。

V-JEPA 2 确实能做 multi-step hierarchical planning，但 paper 指出两个 critical issue：

- **Geometric neglect**: latent $s_t^x \in \mathbb{R}^n$，Euclidean metric 假设了各向同性、flat geometry。但真实的 world state transitions 构成 hierarchical structure（task → subtask → action），Euclidean metric 无法 express hierarchy 的 exponential volume growth。
- **Multi-step shortcoming**: 训练数据多来自 one-step transitions (Kinetics, HowTo100M, Something-Something)，rollout horizon 一拉长，error 累积，SR 从 T=3 的 50.16 掉到 T=8 的 4.95（CrossTask，V-JEPA 2 ViT-g384）。

### 1.2 为什么是 Hyperbolic space

这是这篇 paper 最值得 build intuition 的点。考虑 state $s_t$ 后预测 $d$ 步 future，action space size = $B$，则 future 数量

$$N_d = B^d$$

这恰好是 tree 的叶子数，volume 随 depth 指数增长。Euclidean space $\mathbb{R}^n$ 的 volume 随 radius 多项式增长 ($\sim r^n$)，要把 tree 塞进去 embedding distortion 极大。而 hyperbolic space $\mathbb{H}^n$ 的 volume 随 radius 指数增长

$$\text{Vol}(B_\mathbb{H}(r)) \sim e^{(n-1)r}$$

刚好匹配 tree 的 exponential branching。这是 Nickel & Kiela 在 NeurIPS 2017 Poincaré embeddings 里的核心观察 ([参考](https://arxiv.org/abs/1705.08039))，也是 Ganea et al. NeurIPS 2018 Hyperbolic Neural Networks ([参考](https://arxiv.org/abs/1805.09112)) 的基础。

用 paper Section 2 的话说：**"world state transitions (from video observations) naturally form a hierarchical structure that is suitably represented in hyperbolic space."** 这就给 hyperbolic latent 一个 first-principles 的 motivation，比单纯"hyperbolic 适合 hierarchy"的口号强很多。

---

## 2. 方法核心：Hyperbolic JEPA (H-JEPA)

### 2.1 从 Euclidean 到 Hyperbolic 的 mapping

Encoder $E_\theta(\cdot)$ 是 V-JEPA 2 frozen encoder，输出 Euclidean latent

$$s_t^x = E_\theta(x_t) \in \mathbb{R}^n \quad (\text{Eq. 1})$$

H-JEPA 把 $s_t^x$ 解释为 origin 处的 tangent vector，通过 **exponential map at origin** 投到 Poincaré ball $\mathbb{B}_c^n$：

$$s_{t,\mathbb{H}}^x = \exp_0(s_t^x) = \tanh\!\bigl(\sqrt{c}\,\|s_t^x\|\bigr)\,\frac{s_t^x}{\sqrt{c}\,\|s_t^x\|} \quad (\text{Eq. 2})$$

**变量含义解读**:
- $s_t^x$: 时间 $t$ 的 Euclidean latent embedding (encoder 输出)
- $s_{t,\mathbb{H}}^x$: 投影后落在 Poincaré ball 上的 hyperbolic latent
- $c > 0$: curvature 参数（curvature $K = -c$），可学习，初始化 $c=1$，最终收敛到 $0.3$
- $\tanh(\cdot)$: 把 vector 的 norm 压到 $[0,1)$ 区间内，保证 $\|s_{t,\mathbb{H}}^x\| < 1/\sqrt{c}$（落在 ball 内）
- $\frac{s_t^x}{\|s_t^x\|}$: 只保留方向，norm 用 $\tanh$ 重新赋值

**Intuition**: $\tanh$ 这个 squash 是 Poincaré ball 的灵魂——origin 邻域近似 Euclidean (tanh 在 0 附近线性)，但越往 ball 边界走，等量 Euclidean displacement 对应的 hyperbolic distance 越大（呈指数 blow-up）。所以 hierarchy 的 root (coarse abstraction) 在 origin 附近，leaves (fine details) 沿 radial 方向被推到边界，自动产生 hierarchy。

**注意**: paper 把 $\exp_0$ 实现为 **differentiable hyperbolic projection layer**，curvature $c$ 通过优化 $\log c$ 学习（保证 $c>0$），并 clamp 到 $[0.1, 10.0]$ 防止数值不稳。这是 Chami et al. NeurIPS 2019 HGNN 的 trick ([参考](https://arxiv.org/abs/1910.12923))。

### 2.2 Action-conditioned predictor 在 hyperbolic 空间

Predictor $P_\phi$ 是 ~300M param transformer (24 layers, 16 heads, 1024 hidden, GELU)，吃 hyperbolic latent sequence + action sequence：

$$\bigl(\hat{s}_{t+1,\mathbb{H}}^x\bigr)_{t=1}^T = P_\phi\!\bigl((s_{t,\mathbb{H}}^x, a_t)_{t=1}^T\bigr) \quad (\text{Eq. 3})$$

$\theta$ = encoder weights (frozen), $\phi$ = predictor weights (trainable)。

**这里有个微妙点**: predictor 内部的 transformer operation 是在 Euclidean coordinate 还是 hyperbolic operation？paper 没有显式说做 Möbius linear layer，从实现细节看应该是 Euclidean transformer，只在 input/output boundary 做 $\exp_0$ / $\log_0$ 投影。真正的 hyperbolic neural network operations (Möbius addition, Fréchet mean) 在 transformer 内部并没有完全引入——这有点遗憾，但工程上更稳。

### 2.3 Poincaré ball geodesic distance

这是所有 loss 的基础。给定 $u, v \in \mathbb{B}_c^n$：

$$d_\mathbb{H}(u, v) = \frac{1}{\sqrt{c}}\,\text{arcosh}\!\Bigl(1 + 2c\,\frac{\|u - v\|^2}{(1 - c\|u\|^2)(1 - c\|v\|^2)}\Bigr) \quad (\text{Eq. 68})$$

**变量含义**:
- $u, v$: Poincaré ball 上的两个点
- $\|u\|, \|v\|$: 它们的 Euclidean norm（必须 $< 1/\sqrt{c}$）
- $\|u - v\|^2$: Euclidean 距离平方
- $c$: curvature
- $\text{arcosh}$: inverse hyperbolic cosine，$\text{arcosh}(x) = \ln(x + \sqrt{x^2 - 1})$
- 前缀 $\frac{1}{\sqrt{c}}$: 把 unit-curvature distance 缩放到 general curvature

**关键 intuition**: 当 $\|u\| \to 1/\sqrt{c}$ (approach boundary)，分母 $(1 - c\|u\|^2) \to 0$，整个 fraction 趋于无穷——这意味着 ball 边界对应"无穷远"，hierarchical leaves 在那里被指数级 spread out。Euclidean space 里两点距离有上限，hyperbolic ball 内 Euclidean 距离上限是 $2/\sqrt{c}$，但 hyperbolic distance 无上限。这就是 hierarchy embedding 的几何 power。

### 2.4 Teacher Forcing + Rollout Loss

**Teacher Forcing** (one-step prediction)：

$$\mathcal{L}_{\text{Trf}}(\theta, \phi) = \frac{1}{T}\sum_{t=1}^T d_\mathbb{H}\bigl(P_\phi(\exp_0(E_\theta(x_t)), a_t),\, \exp_0(E_\theta(x_{t+1}))\bigr) \quad (\text{Eq. 4})$$

简化记号：

$$\mathcal{L}_{\text{Trf}} = \frac{1}{T}\sum_{t=1}^T d_\mathbb{H}(\hat{s}_{t+1,\mathbb{H}}^x, s_{t+1,\mathbb{H}}^x) \quad (\text{Eq. 5})$$

展开成显式 Poincaré distance：

$$\mathcal{L}_{\text{Trf}} = \frac{1}{T}\sum_{t=1}^T \frac{1}{\sqrt{c}}\,\text{arcosh}\!\Bigl(1 + 2c\,\frac{\|\hat{s}_{t+1,\mathbb{H}}^x - s_{t+1,\mathbb{H}}^x\|^2}{(1 - c\|\hat{s}_{t+1,\mathbb{H}}^x\|^2)(1 - c\|s_{t+1,\mathbb{H}}^x\|^2)}\Bigr) \quad (\text{Eq. 6})$$

**Rollout loss** (two-step)：

$$\mathcal{L}_{\text{rollout}} = \frac{1}{T}\sum_{t=1}^T d_\mathbb{H}\bigl(P_\phi(\exp_0(E_\theta(x_t)), a_t, a_{t+1}),\, \exp_0(E_\theta(x_{t+2}))\bigr) \quad (\text{Eq. 7})$$

也就是把 predictor 自己的输出 feed back 当 input，强制 temporal consistency across two steps。

**Total SFT loss**：

$$\mathcal{L}_{\text{SFT}} = \lambda\,\mathcal{L}_{\text{Trf}} + (1-\lambda)\,\mathcal{L}_{\text{rollout}} \quad (\text{Eq. 10})$$

Ablation Table 3 显示 $\lambda = 0.5$ 最优 (T=4 时 SR 35.92 vs $\lambda=1$ 的 34.65)，说明 one-step 和 two-step 监督要平衡。

**为什么 two-step rollout 够用？** 因为 longer rollout 在 supervised 训练里 variance 太大，且 GRL 阶段会专门做 multi-step refinement。Two-step 是个 sweet spot——既有 temporal signal 又不会 gradient 衰减太严重。

---

## 3. Geometric Reinforcement Learning (GRL)

这是 paper 第二个核心创新，也是最容易让 reader 困惑的部分。Karpathy 你一定会喜欢这个 formulation——它把 RL 套在 energy landscape 上，但又不需要训单独的 policy network 或 reward model。

### 3.1 Energy cost = negative reward

给定 frozen encoder $E$ 和 trainable predictor $P_\phi$：

$$c_t(s_{t,\mathbb{H}}^x, s_{t+1,\mathbb{H}}^x) = d_\mathbb{H}\bigl(P_\phi(\exp_0(E(x_t)), a_t),\, \exp_0(E(x_{t+1}))\bigr) = d_\mathbb{H}(\hat{s}_{t+1,\mathbb{H}}^x, s_{t+1,\mathbb{H}}^x) \quad (\text{Eq. 11, 12})$$

$$r_t(s_{t,\mathbb{H}}^x, a_t, s_{t+1,\mathbb{H}}^x) = -c_t(s_{t,\mathbb{H}}^x, s_{t+1,\mathbb{H}}^x) \quad (\text{Eq. 13})$$

**关键 insight**: reward 直接定义为 negative hyperbolic distance，省掉了 external reward model。这跟 Du et al. ICML 2022 "Learning iterative reasoning through energy minimization" ([参考](https://arxiv.org/abs/2206.15426)) 思路相通——energy 本身就是 reward signal。

### 3.2 Path value function

$$V(s_{1,\mathbb{H}}^x, s_{1+T,\mathbb{H}}^x) = \mathbb{E}_{a_{1:T}\sim\phi}\Bigl[\sum_{t=1}^T \gamma^{t-1}\,r_t(s_{t,\mathbb{H}}^x, a_t, s_{t+1,\mathbb{H}}^x)\Bigr] \quad (\text{Eq. 14})$$

**变量含义**:
- $s_{1,\mathbb{H}}^x$: 当前 (start) latent state
- $s_{1+T,\mathbb{H}}^x$: target (goal) latent state
- $a_{1:T}$: action sequence，由 predictor 参数化 $\phi$ 决定
- $\gamma \in [0,1)$: discount factor，paper 取 $\gamma = 0.99$
- $r_t$: 即时 reward (negative hyperbolic distance)

注意 $a_{1:T} \sim \phi$ 这个 notation 暗示 predictor 本身就是 policy。这是一个 **model-based RL** 的特殊形式：world model 和 policy 共享参数 $\phi$，没有 actor-critic 分离。

### 3.3 Optimal value function 和 Bellman 意义

$$V^*(s_{1,\mathbb{H}}^x, s_{1+T,\mathbb{H}}^x) = \max_\phi\,\mathbb{E}_{a_{1:T}\sim\phi}\Bigl[\sum_{t=1}^T \gamma^{t-1}\,r_t\Bigr] = \min_\phi\,\mathbb{E}_{a_{1:T}\sim\phi}\Bigl[\sum_{t=1}^T \gamma^{t-1}\,d_\mathbb{H}(\hat{s}_{t+1,\mathbb{H}}^x, s_{t+1,\mathbb{H}}^x)\Bigr] \quad (\text{Eq. 15})$$

最后这一步把 max reward 翻译成 min distance 是核心 trick——**maximize cumulative reward ⟺ minimize cumulative hyperbolic geodesic distance between predicted and true trajectory**。

这里参考了 Bellman 1957 dynamic programming ([参考](https://www.science.org/doi/10.1126/science.153.3731.34)) 和 Sutton & Barto RL 教科书 ([参考](http://incompleteideas.net/book/RLbook2020.pdf))。

### 3.4 Triangle inequality regularization

Hyperbolic distance 满足 triangle inequality。对 predictor rollout 任意连续三元组：

$$d_\mathbb{H}(\hat{s}_{t,\mathbb{H}}^x, \hat{s}_{t+2,\mathbb{H}}^x) \le d_\mathbb{H}(\hat{s}_{t,\mathbb{H}}^x, \hat{s}_{t+1,\mathbb{H}}^x) + d_\mathbb{H}(\hat{s}_{t+1,\mathbb{H}}^x, \hat{s}_{t+2,\mathbb{H}}^x) \quad (\text{Eq. 16})$$

**直觉**: 如果 predictor 学到的 trajectory 真的沿 geodesic 走，那从 $t$ 直接到 $t+2$ 的距离应该等于走两步之和（取等号）——任何 "shortcut" 都应该被惩罚。这是把 metric 的内在约束变成 regularizer。

Paper 写的 regularization 是

$$\mathcal{L}_\Delta = \frac{1}{T-2}\sum_{t=1}^{T-2}\Bigl[d_\mathbb{H}(\hat{s}_t, \hat{s}_{t+2}) - d_\mathbb{H}(\hat{s}_t, \hat{s}_{t+1}) - d_\mathbb{H}(\hat{s}_{t+1}, \hat{s}_{t+2})\Bigr] \quad (\text{Eq. 17})$$

(原 paper 公式 17 末尾被截断了，从 context 补全。)

注意这个差值 $\le 0$，所以 $\mathcal{L}_\Delta$ 通常为负——它作为一个 soft penalty push 这个差值接近 0（即让 trajectory 越来越接近 geodesic）。

### 3.5 GRL total loss

$$\mathcal{L}_{\text{GRL}}(\phi) = \mathbb{E}_{a_{1:T}\sim\phi}\Bigl[\sum_{t=1}^T \gamma^{t-1}\,d_\mathbb{H}(\hat{s}_{t+1,\mathbb{H}}^x, s_{t+1,\mathbb{H}}^x)\Bigr] + \beta\,\mathcal{L}_\Delta \quad (\text{Eq. 18})$$

$\beta = 0.1$ 是 regularization weight。

**这里跟 PPO/SAC 的本质区别**: PPO 训 policy $\pi_\theta(a|s)$，critic 估 $V_\psi(s)$，要训两个 network。GRL 只调 predictor $\phi$，把 energy function 当 value function 用。这种"self-referential"的 RL 形式跟 LeCun 2022 "A Path Towards Autonomous Machine Intelligence" ([参考](https://openreview.net/pdf?id=BZ5a1r-kVsf)) 里设想的 energy-based model 当 world model + value function 是同一个哲学。

### 3.6 GRL 的训练 hyperparams

- Optimizer: AdamW, weight decay 0.04
- LR: linear warmup $5\times 10^{-5} \to 2\times 10^{-4}$ over 2k iter, hold 18k iter, decay to 0 over final 5k iter
- Batch size: 128 (比 SFT 阶段的 256 小，因为 RL objective variance 高)
- $\gamma = 0.99, \beta = 0.1$

Ablation Table 4 显示：
- $\beta = 0$ vs $\beta = 0.1$ 在 T=4 上 SR 差 0.97 (36.07 → 37.04)，证明 triangle inequality 起作用
- $\gamma = 0.99$ vs $\gamma = 0.90$ 在 T=4 上 SR 差 0.62，证明长 horizon discount 重要
- 最佳组合 $\beta = 0.1, \gamma = 0.99$

---

## 4. Energy-Based Planning with CEM

训练完后，做 inference 时用 **Cross-Entropy Method** ([参考](https://link.springer.com/article/10.1007/s10479-005-5723-z)) 在 hyperbolic latent space 搜 action sequence：

**Step 1: encode start 和 goal**
$$s_{1,\mathbb{H}}^x = \exp_0(E(x_1)), \quad s_{1+T,\mathbb{H}}^x = \exp_0(E(x_{1+T})) \quad (\text{Eq. 19})$$

**Step 2: 定义 energy cost function**
$$C((\hat{a}_t)_{t=1}^T;\, s_{1,\mathbb{H}}^x, s_{1+T,\mathbb{H}}^x) = d_\mathbb{H}\bigl(P((\hat{a}_t)_{t=1}^T;\, s_{1,\mathbb{H}}^x),\, s_{1+T,\mathbb{H}}^x\bigr) \quad (\text{Eq. 20})$$

**Step 3: 最小化**
$$(a_t^*)_{t=1}^T = \arg\min_{(\hat{a}_t)_{t=1}^T}\, d_\mathbb{H}\bigl(P((\hat{a}_t)_{t=1}^T;\, s_{1,\mathbb{H}}^x),\, s_{1+T,\mathbb{H}}^x\bigr) \quad (\text{Eq. 21})$$

**CEM 参数**: $N = 800$ samples, $K = 80$ elites, $I = 10$ iterations。Algorithm 1 给了完整伪代码（在 Appendix 1.3.3）。

**Intuition**: CEM 是 zero-order optimization，从 $\mathcal{N}(\mu_0, \Sigma_0)$ 采样 800 个 action sequence，用 predictor 想象 future latent，按 hyperbolic distance to goal 排序，选 top-80 elite，更新 Gaussian 参数，迭代 10 次，最后取最优 sequence 的第一个 action 执行，然后 receding horizon 重新规划。

跟 V-JEPA 2-AC 的 inference pipeline 完全一样，唯一区别是把 $L_1$ Euclidean distance 换成 $d_\mathbb{H}$ hyperbolic distance。

---

## 5. Energy Landscape 可视化 (Appendix 4)

Paper Figure 2 是 intuition building 的关键证据。在 COIN dataset "Replace Memory Chip" 任务的初始 step，选一个 reference latent state $s_t$，扫两个 orthonormal tangent-space direction $(\Delta x, \Delta y)$，画 energy surface。

**V-JEPA 2 (Euclidean)**: 几乎对称的 smooth paraboloid，weak directional structure——即各方向 perturbation 几乎等价对待。
**GeoWorld (Hyperbolic)**: sharper, curvature-aware basin，more pronounced directional variation——沿 hierarchical 方向的 gradient 更陡，反映 H-JEPA 编码了 hierarchical structure。

**为什么这个可视化重要？** 因为 CEM 沿 energy gradient 走，更陡的 gradient = 更快收敛到 geodesic。在 Euclidean flat landscape 里 CEM 等于 random search 退化；在 hyperbolic structured landscape 里 CEM 自动 follow geodesic。

---

## 6. Gromov δ-Hyperbolicity 验证 (Appendix 5)

这是 paper 用数学指标量化 "latent space 到底有多 tree-like" 的方法。Gromov δ-hyperbolicity 用四点条件定义：对任意四个点 $x, y, z, w$，三个 pair-sum

$$d(x,y) + d(z,w),\quad d(x,z) + d(y,w),\quad d(x,w) + d(y,z)$$

排序后取最大的两个之差 = $\delta$。$\delta \to 0$ 表示空间是 tree-like ($\delta = 0$ 是真正的 tree metric)，$\delta$ 越大越不 hyperbolic。

Appendix Figure 1 显示 GeoWorld 在 CrossTask 上的 latent quadruples 上 $\delta$ 分布更集中在 0 附近，证明它的 latent space 确实比 V-JEPA 2 Euclidean latent 更 tree-like。这个 sanity check 很重要，光说"hyperbolic 适合 hierarchy"没用，得证明学出来的 latent 真的 hyperbolic 了。

参考 Gromov 1987 "Hyperbolic Groups" 原始定义 ([参考](https://link.springer.com/chapter/10.1007/978-3-642-58158-8))。

---

## 7. 实验结果深度分析

### 7.1 Dataset

- **CrossTask** ([参考](https://arxiv.org/abs/1912.01901)): 4.7K videos, 83 tasks, 105 actions, avg 8 actions/video, 375h total
- **COIN** ([参考](https://arxiv.org/abs/1903.02875)): 11,287 videos, 180 tasks, 778 actions, avg 3.9 actions/video, 476h total

### 7.2 Metrics

- **SR (Success Rate)**: predicted action sequence exact match ground truth
- **mAcc (Mean Accuracy)**: per-step action accuracy averaged
- **mIoU (Mean IoU)**: overlap between predicted procedure 和 ground truth

### 7.3 Procedural Planning (image observation + image goal)

Table 1 关键数字（CrossTask T=3）：

| Method | SR | mAcc | mIoU |
|---|---|---|---|
| V-JEPA 2 ViT-g384 | 45.58 | 72.74 | 69.42 |
| GeoWorld ViT-g384 | 47.47 | 73.69 | 86.55 |

**注意 mIoU 大幅提升** (69.42 → 86.55, +17.13%)，远超 SR 的 +1.89%。这说明 GeoWorld 不仅预测正确的 action 顺序，还更好捕捉到 procedure 整体结构（多步 overlap 比单步 success 信号更结构化）。

COIN T=3 SR 提升 +0.77 (34.08 → 34.85)，mIoU 提升 +5.31 (64.53 → 89.88)。

### 7.4 Visual Planning with Videos (video observation + video goal)

Table 2 关键数字（CrossTask T=3）：

| Method | SR | mAcc | mIoU |
|---|---|---|---|
| GPT-5 | 50.03 | 72.38 | 91.18 |
| Gemini 2.5 Pro | 48.91 | 73.82 | 90.30 |
| V-JEPA 2 ViT-g384 | 50.16 | 74.86 | 91.73 |
| GeoWorld ViT-g384 | 51.71 | 77.30 | 92.95 |

**这是 paper 最 sharp 的数字**——GeoWorld ViT-g384 在 SR 上超过 GPT-5 (51.71 vs 50.03)！虽然 GPT-5 是 zero-shot VLM 没用 task-specific training，但仍然 striking，因为 V-JEPA 2 系列用 frozen encoder + lightweight predictor 就能 beat 一个 frontier-scale VLM。Video setup 比 image setup (procedural planning) 更能展示 GeoWorld 优势——video 有更丰富的 temporal dynamics，hyperbolic structure 更明显。

### 7.5 Long-Horizon Planning (Table 3 + Appendix Table 5)

Appendix Table 5 是 paper 最有说服力的 ablation：

| Method | T=3 | T=4 | T=5 | T=6 | T=7 | T=8 |
|---|---|---|---|---|---|---|
| V-JEPA 2 ViT-g384 | 50.16 | 35.01 | 23.17 | 16.88 | 8.26 | 4.95 |
| SFT (Hyperbolic) | 50.42 | 35.92 | 23.64 | 16.97 | 14.88 | 11.51 |
| GRL (Euclidean) | 50.26 | 35.47 | 23.85 | 17.03 | 15.12 | 12.74 |
| GRL (Hyperbolic) | 51.04 | 36.33 | 24.05 | 17.82 | 15.54 | 13.10 |
| SFT + GRL | 51.71 | 37.04 | 24.83 | 18.26 | 16.09 | 13.81 |

**关键 insight**:
- T=3 时所有方法都接近，~50% SR
- T=8 时 V-JEPA 2 掉到 4.95%，SFT+GRL 还维持 13.81%——**2.79× 优势**
- SFT (Hyperbolic) alone vs V-JEPA 2 Euclidean：T=7 时 14.88 vs 8.26，**证明 hyperbolic geometry 本身就能稳 long horizon**
- GRL (Euclidean) vs V-JEPA 2：T=8 时 12.74 vs 4.95，**证明 GRL 即使在 Euclidean 也能帮助**
- GRL (Hyperbolic) > GRL (Euclidean) > SFT (Hyperbolic) > V-JEPA 2 baseline——**两个 innovation 正交且互补**

这就是 paper Section 6 说的："error accumulation 和 geometric drift 是 related but distinct——hierarchy 限制 error 怎么 propagate，rollout loss + GRL 帮 eliminate error"。

---

## 8. Curvature 学习动态 (Appendix Figure 2)

Curvature $c$ 从 $c=1$ (unit curvature，最 hyperbolic) 逐渐降到 $c \approx 0.3$ 并稳定。这符合直觉——cross-task video 数据的 hierarchy 不像 pure tree 那么极端，适度 curvature 就够，太 sharp 的 hyperbolic 反而 distort representation。

Figure 2(a-c) 展示 curvature 对 geodesic 的影响：
- $K = -1$ (sharp): geodesic 强烈弯向 origin
- $K \to 0$ (flat): geodesic 趋近 straight line
- 学到的 $c = 0.3$ 在中间，moderate hierarchy

这个 learnable curvature 跟 Chami et al. HGNN 思路一致——curvature 是 data-dependent hyperparameter，应该学。

---

## 9. Frozen Encoder vs Full Fine-Tuning (Appendix Table 1)

Frozen encoder + lightweight $\exp_0$ projection layer 训练 已经能拿到大部分性能。Full fine-tuning encoder 在 ViT-g384 上 SR 只多 +0.33 (51.71 → 52.04)，但 trainable params 和 compute 大幅增加。这跟 LeCun "big pretrain + small adaptation" 哲学一致，也跟 V-JEPA 2 paper 的发现吻合——self-supervised pretrain 学到的 visual representation 已经足够 task-agnostic。

---

## 10. Limitations

Paper Section 7 很诚实地承认：
1. Hierarchy 的 intuition 来自 multi-step future expansion ($B^d$)，**不是** sub-task hierarchy (high-level task → mid action → low end-effector)。这跟原始 LeCun JEPA paper 的 hierarchical intuition 不同——JEPA 是 explicit 的 multi-level，GeoWorld 是 implicit 的 single-level + horizon-induced hierarchy。
2. 只在 visual planning 任务上验证，embodied planning (robotics) 是 future work。

---

## 11. 跟相关工作脉络的联系

### 11.1 JEPA 家族
- **I-JEPA** (Assran et al. CVPR 2023) ([参考](https://arxiviv.org/abs/2301.08243)): image-level masked prediction
- **V-JEPA** (Bardes et al. TMLR 2024) ([参考](https://arxiv.org/abs/2302.14202)): video extension
- **V-JEPA 2** (Assran et al. 2025) ([参考](https://arxiv.org/abs/2506.09985)): scaled-up, action-conditioned, planning capability
- **GeoWorld**: 把 V-JEPA 2 的 latent space 搬到 hyperbolic manifold

### 11.2 Hyperbolic representation learning
- **Poincaré Embeddings** (Nickel & Kiela NeurIPS 2017) ([参考](https://arxiv.org/abs/1705.08039)): first hyperbolic word embedding for hierarchy
- **Hyperbolic Neural Networks** (Ganea et al. NeurIPS 2018) ([参考](https://arxiv.org/abs/1805.09112)): Möbius operations
- **HGNN** (Chami et al. NeurIPS 2019) ([参考](https://arxiv.org/abs/1910.12923)): hyperbolic GCN with learnable curvature
- **Hyperbolic VLM** (Desai et al. ICML 2023, Compositional entailment learning, Pal et al. 2024) ([参考](https://arxiv.org/abs/2410.06912)): hyperbolic vision-language

### 11.3 Energy-based models & planning
- **EBM** (LeCun et al., Du & Mordatch ICML 2022) ([参考](https://arxiv.org/abs/2206.15426)): energy as compatibility
- **CEM** (De Boer et al. 2005) ([参考](https://link.springer.com/article/10.1007/s10479-005-5723-z)): zero-order trajectory optimization
- **π0, π0.5** (Physical Intelligence 2024-2025) ([参考](https://arxiv.org/abs/2410.24164)): generative VLA, 不同范式但相关

### 11.4 Goal-conditioned visual planning
- **DDN** (Chang et al. ECCV 2020): dual dynamics network, procedural planning 先驱 ([参考](https://arxiv.org/abs/2007.14030))
- **P3IV** (Zhao et al. CVPR 2022) ([参考](https://arxiv.org/abs/2203.13230))
- **PDPP** (Wang et al. CVPR 2023) ([参考](https://arxiv.org/abs/2303.01564)): diffusion-based procedure planning
- **KEPP** (Nagasinghe et al. CVPR 2024) ([参考](https://arxiv.org/abs/2312.01992)): knowledge-enhanced
- **ActionDiffusion** (Shi et al. WACV 2025) ([参考](https://arxiv.org/abs/2410.12259))
- **VideoWorld** (Ren et al. CVPR 2025) ([参考](https://arxiv.org/abs/2505.01140)): generative video world model, 直接对比 baseline

---

## 12. 我的几个 critique 和 open questions

**12.1 Predictor 内部还是 Euclidean operation**

Paper 没明确说 predictor transformer 内部用 Möbius linear / Fréchet mean 这些 hyperbolic operations。看起来是 Euclidean transformer + input/output 用 $\exp_0 / \log_0$ 做 boundary projection。这是工程务实但理论不彻底——真正 hyperbolic transformer (像 HNN, HGCN) 内部所有 operation 都在 manifold 上。如果只在 boundary 投影，中间 transformer operation 可能 break hyperbolic geometry 的 inductive bias。Appendix 4 的 energy landscape 显示 hyperbolic 确实有 structure，但多少归功于 $\exp_0$ projection 本身、多少归功于 hyperbolic distance $d_\mathbb{H}$ 在 loss 里、多少归功于 transformer 内部 operation——这三者贡献没 ablation 区分。

**12.2 Curvature 学习到 $c \approx 0.3$ 暗示什么**

$c = 0.3$ 意味着学到的 ball radius $1/\sqrt{c} \approx 1.83$，curvature $K = -0.3$——比 unit curvature ($K=-1$) flat 很多。这说明 CrossTask/COIN 的 task hierarchy 不像 tree 那么极端。问题是：**对于真正的 embodied planning（Droid dataset 这种 manipulation task，hierarchy 可能更 pronounced），curvature 会学得更 sharp 吗？** 这跟 paper Section 7 提的 future work 直接相关。

**12.3 GRL 跟 RLHF / DPO 的关系**

GRL 用 negative energy 当 reward，optimize predictor 参数。这跟 RLHF (Ouyang et al. NeurIPS 2022) ([参考](https://arxiv.org/abs/2203.02155)) 训 LLM 的 PPO 思路很像——只是 reward 来源不同 (RLHF 用 reward model, GRL 用 hyperbolic distance 直接定义)。如果改成 DPO-style (直接用偏好对 optimize policy) 会怎样？这跟 Du et al. ICML 2022 energy-based iterative reasoning 路线一致。

**12.4 Long horizon T=6, 8 上的 gain 来源分解**

Appendix Table 5 已经分得很清楚——hyperbolic geometry 在 $T \ge 7$ 时 gain 突然变大（T=7: +6.62 vs T=6: +0.09），GRL 在 $T \ge 7$ 时持续稳定加成。这暗示 hyperbolic 的优势在 horizon 超过某个 threshold 后才显现，可能跟 task hierarchy depth 有关。如果能 plot SR vs T 的 derivative 会更直观。

**12.5 SR 提升 ~3% 绝对值是否 impressive**

Paper Abstract 说"~3% SR improvement in 3-step planning, ~2% in 4-step"。在 V-JEPA 2 baseline 已经 50.16% 的情况下，绝对 +3% 是相对 +6%，对于 frozen encoder + lightweight change 算不错。但跟 GPT-5 (50.03) 几乎打平的 setting 下，提升空间是不是已经 saturated？真正 large-scale gain 可能在 embodied setting 才显现。

---

## 13. 总结：这篇 paper 在拼图中的位置

GeoWorld 在 V-JEPA 2 的 Euclidean predictive world model 之上，加了两层 structural inductive bias：
1. **Hyperbolic latent geometry**：match world state transition 的 tree-like branching
2. **Energy-based RL with triangle inequality**：把 planning 重新 formalize 为 hyperbolic value function optimization

这跟整个 LeCun 路线图 (JEPA → V-JEPA → V-JEPA 2 → GeoWorld) 一脉相承，**朝着 "world model as value function in structured latent space"** 这个方向走。它不是简单的 ablation 增量，而是给 predictive world model 一个 principled geometric foundation。

跟你 (Karpathy) 在 "micrograd", "nanoGPT", "zero to hero" 系列一直强调的 "build intuition from first principles" 哲学一致——**不要满足于"模型能 work"，要问"为什么能 work，几何上意味着什么"**。GeoWorld 的 energy landscape 可视化 (Figure 2) 就是这种 intuition building 的典范——你看一眼就知道 hyperbolic basin 比 Euclidean paraboloid 信息密度高。

如果接下来有人把这套推到 embodied setting (Droid + RL)，并且真正用 HNN-style fully hyperbolic transformer，那才是这个 program 的 fully realized version。Paper 的工作是先 validate 这个 idea 能在 vision planning 上 beat SOTA，已经做到了。

---

## 参考链接汇总

**核心 paper 引用**:
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA (original): https://arxiv.org/abs/2302.14202

**Hyperbolic geometry**:
- Poincaré Embeddings (Nickel & Kiela 2017): https://arxiv.org/abs/1705.08039
- Hyperbolic Neural Networks (Ganea 2018): https://arxiv.org/abs/1805.09112
- HGNN (Chami 2019): https://arxiv.org/abs/1910.12923
- Hyperbolic Image-Text (Desai 2023): https://arxiv.org/abs/2301.08260
- Compositional entailment VLM (Pal 2024): https://arxiv.org/abs/2410.06912

**Energy-based & planning**:
- Du & Mordatch ICML 2022: https://arxiv.org/abs/2206.15426
- CEM tutorial (De Boer 2005): https://link.springer.com/article/10.1007/s10479-005-5723-z
- Sutton & Barto RL: http://incompleteideas.net/book/RLbook2020.pdf

**Goal-conditioned planning baselines**:
- CrossTask (Zhukov 2019): https://arxiv.org/abs/1812.00818
- COIN (Tang 2019): https://arxiv.org/abs/1903.02875
- DDN (Chang 2020): https://arxiv.org/abs/2007.14030
- PDPP: https://arxiv.org/abs/2303.01564
- KEPP: https://arxiv.org/abs/2312.01992
- ActionDiffusion: https://arxiv.org/abs/2410.12259
- VideoWorld: https://arxiv.org/abs/2505.01140

**VLA & embodied (future direction)**:
- π0 (Physical Intelligence): https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- Droid dataset: https://droid-dataset.github.io/

希望这个 deep dive 帮你 build 起对 GeoWorld 的 intuition！如果要进一步聊 H-JEPA 跟 HNN 内部 operation 区别、或者 GRL 跟 RLHF/DPO 的关系、或者怎么把这套推到 embodied setting，我都乐意展开。
