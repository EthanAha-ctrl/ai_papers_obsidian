---
source_pdf: Delta Forcing Trust Region Steering for Interactive.pdf
paper_sha256: df2412c87ae317e3491e616636c533147f5dd54facbf5297ac4b8eff17803b89
processed_at: '2026-08-18T05:04:56-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Delta Forcing 人话版

## 一句话

你拿一个 frozen 大模型（teacher）蒸馏出一个小模型（student）做实时视频流，过程中 prompt 会切换。前人都说 drift 是 student 自回归误差累积，paper 说不对，drift 是 teacher 自己在 prompt 切换时给的 guidance 就偏了，因为 teacher 训练时只见过短 clip + 固定 prompt，根本不会 condition on student 已经生成的 history。修复办法：用 student 自己生成的 fake trajectory 当作"无偏参照系"，在线检测 teacher 的 chunk-to-chunk delta 与 student delta 在 DINO feature 空间的距离，距离大就说明 teacher 在 bias、就 fall back 到 student 的 momentum 上。

就这么个事。下面把直觉讲透。

---

## 场景：为什么这是一个真问题

interactive streaming video generation 不是"生成 10 秒视频"。它是"生成 60 秒视频，中间 prompt 会切，要求 transition smooth，不允许 hard cut，还要 real-time"。

具体例子（paper Appendix B 给的 benchmark）：

```
00:00-00:10  park ranger walking in national park
00:10-00:20  park ranger discovers orphaned fawn at base of redwood
00:20-00:30  park ranger kneels down, speaks softly
00:30-00:40  park ranger scoops up fawn into arms
00:40-00:50  park ranger carries fawn back to station
00:50-01:00  park ranger walks with purpose carrying fawn
```

这个 setting 在工程上要求：
- causal attention（只能用 past）
- few-step generation（蒸馏）
- 跨 prompt 的 identity / scene / layout consistency

第三点是真正的瓶颈。前人 LongLive、MemFlow 都聚焦在前两点的基础上加 Stage 2 streaming long tuning 解决第三点，但是 60s 跑下来还是漂。

Reference: 
- LongLive: https://arxiv.org/abs/2510.01783
- MemFlow: https://arxiv.org/abs/2512.05457

---

## 失败现象：drift 的具体表现

paper 里反复出现的失败例子：

```
00:00-00:10  grandfather in dark blue clothes enters
00:10-00:20  grandfather talks back   ← 这里 drift 发生
00:20-00:30  ...继续生成...
```

到 00:10-00:20，grandfather 的衣服颜色从 dark blue 漂成 white。或者 object identity 变、layout 偏、scene composition 漂。

这种 drift 不是生成质量的随机噪声。它是 systematic 的、方向一致的、跨 event 边界反复出现的。每个 event switch 都会推一次，越推越远。

---

## 之前人怎么归因 vs 这篇怎么归因

**之前 LongLive / MemFlow / Self-Forcing / Causal Forcing 的归因**：autoregressive generation error accumulation。Student 自回归，每步误差累积，rollout 越长漂得越远。

**这篇的归因**：drift 不是 student 的错，是 teacher 的错。

paper 在 Section 3 做了一个很 sharp 的实验：同时 decode teacher 的 predicted output 和 student 的 generation 到 pixel space，比较它们 trajectory。发现 student 的漂移方向**和 teacher 的漂移方向完全一致**。teacher 漂什么 student 就漂什么。

这推翻了"误差累积"的解释。如果是误差累积，应该是无方向、随机放大。但实测是有方向、有结构、与 teacher 同步。

---

## 为什么 teacher 会漂：核心诊断

这是 paper 最 sharp 的部分。我重新用工程师语言讲一遍。

pretrained video teacher（Wan2.1-14B、HunyuanVideo）是这么训的：短 clip（5 秒）、single static prompt、bidirectional attention。它没见过：
1. 长序列
2. prompt 切换
3. 需要条件在"之前已经生成了什么"上

但是 Stage 2 streaming long tuning 时，student 每次会把已经生成的 history（通过 KV cache 或者 memory mechanism）喂给 teacher，让 teacher 给 guidance。teacher 不知道怎么办 —— 它从来没被训过 condition on 一个 incoming history。

**形式化**：理想情况下，event e 时的 score 应该是 history-conditional 的：

$$s^*(x, t \mid h_{e-1}, c_e) = \nabla_x \log p_t^*(x \mid h_{e-1}, c_e)$$

变量：
- $s^*$：teacher score function
- $x$：noisy latent sample
- $t$：noise level
- $h_{e-1}$：event e 之前累积的历史
- $c_e$：event e 的新 condition

但是 teacher 实际能给的是 history-marginalized 的 score：

$$\bar{s}^*(x, t \mid c_e) = \mathbb{E}_{h_{e-1} \sim p(\cdot \mid c_e)}[s^*(x, t \mid h_{e-1}, c_e)]$$

含义：teacher 把所有可能的 incoming history 按 $p(\cdot \mid c_e)$ 加权平均掉，只给你一个只依赖 $c_e$ 的 marginalized score。

定义 per-event bias：

$$b(x, t; e) \triangleq s^*(x, t \mid h_{e-1}, c_e) - \bar{s}^*(x, t \mid c_e)$$

DMD gradient 在 event 切换时实际变成：

$$\nabla_\theta \mathcal{L}_{DMD}^{biased} = -\mathbb{E}[(\bar{s}^* - s_{fake}) \frac{\partial G_\theta}{\partial \theta}] - \mathbb{E}[b(x, t; e) \frac{\partial G_\theta}{\partial \theta}]$$

第一项是 nominal DMD update，第二项是 conditional bias 引入的 spurious direction。

**直觉翻译**：student 已经生成了 dark blue 衣服的 grandfather。prompt 切到 "talk back"。teacher 看不到 dark blue 这个 history，它只知道 "talk back" 的 marginalized 分布。这个 marginalized 分布里，grandfather 的衣服颜色可能是平均后的某个颜色（不是 dark blue）。teacher 给的 score 就把 student 从 dark blue 拉到那个平均颜色。下一次又来一次。漂移就这么积累。

DMD reference: https://arxiv.org/abs/2312.04261
Improved DMD: https://arxiv.org/abs/2405.14867

---

## 解决思路的灵感来源：TRPO

TRPO 的核心 insight（Schulman 2015）：policy gradient 的 advantage estimate 只在 policy 不动太远的 region 内可信，因此用 KL constraint 把 policy update 限制在 trust region 内。TRPO 用 parameter space 的 KL ball 定义 trust region，需要二阶信息（Fisher information matrix）+ conjugate gradient。

paper 把这个思想搬过来：teacher guidance 在 evolving condition 下不可 uniform 信任，需要 trust region。

但是有两个改造：

1. **trust region 不在 parameter space**：搬到了 observable latent trajectory space。这样不用算二阶 gradient，online 用一个 feature distance 就能定 trust region radius。计算便宜、可复现。

2. **fall back 的对象不同**：TRPO 没 fall back，只限制步长。Delta Forcing 在 teacher 不可信时 fall back 到 student 自己的 momentum。这是利用了 DMD 里 fake critic 与 real teacher 的一个不对称性，下面专门讲。

TRPO reference: https://arxiv.org/abs/1502.05477

---

## 关键 insight：fake critic 是 bias-free 的，real teacher 不是

DMD 里有两种 score：

- **real score $s_{real}$**：frozen teacher。一旦 biased，persistent 地把 student 拉离 trajectory。
- **fake score $s_{fake}$**：learnable critic。在 student 自己生成的 sample 上更新，所以**自带 student 的 generation history momentum**。

这个不对称是 Delta Forcing 整个 design 的支点。

为什么 fake 是 bias-free 的？因为它从来不见 external guidance，它只见 student 自己生成的 fake samples。所以它反映的就是"student 实际生成的 trajectory 走到哪了"，而不是"teacher 认为 student 应该走到哪"。

这就给出了一个自然的设计：teacher 可信时听 teacher（full DMD），teacher 不可信时 fall back 到 fake trajectory 的 momentum。问题就变成：**online 怎么判别 teacher 这次可不可信？**

---

## 怎么判别：观察 teacher chunk-level evolution 的 smoothness

empirical 观察：conditional bias 不表现为 uniform noise，而是 teacher chunk-to-chunk evolution 中的 sharp discontinuity。

意思是：在 stable event 内，teacher 的输出在 DINO feature 空间是平滑过渡的；prompt 切换时，teacher 的输出突然 jump 到一个新 mode。这个 jump 就是 bias 在 surface 上的表征。

所以判别 teacher 可信度，不需要从训练动态里推断 hidden reliability，只需要在线检测 teacher 的 chunk-level evolution 是不是 smooth。

具体计算（用同一 frozen DINO $\Phi$ 编码）：

$$\delta_k^{fake} = \Phi(\hat{x}_{fake}^{(k)}) - \Phi(\hat{x}_{fake}^{(k-1)})$$

$$\delta_k^{real} = \Phi(\hat{x}_{real}^{(k)}) - \Phi(\hat{x}_{real}^{(k-1)})$$

变量：
- $\hat{x}_{fake}^{(k)}$ / $\hat{x}_{real}^{(k)}$：student / teacher 在 chunk k 的 denoised estimate
- $\delta_k^{fake}$：fake trajectory 的 chunk-level evolution
- $\delta_k^{real}$：real trajectory 的 chunk-level evolution

为啥用 DINO 不用 pixel distance？因为 pixel-level noise 会污染 delta，而 DINO feature 对 pixel noise 鲁棒、对 semantic jump 敏感。这是个 well-known 的选择，DINOv2 / DINOv3 都行。

DINOv2: https://arxiv.org/abs/2304.07193
DINOv3: https://arxiv.org/abs/2508.10104

---

## Trust region gating：一个 sigmoid 把它全包起来

定义 delta discrepancy：

$$\rho_k = \|\delta_k^{real} - \delta_k^{fake}\|_2$$

含义：teacher 在 chunk k 的 evolution 与 student 在 chunk k 的 evolution 偏离多少。

定义 trust region weight：

$$w_k = \sigma(-(\rho_k - \mu) \cdot s)$$

变量：
- $\mu$：detection threshold
- $s$：sigmoid sharpness
- $\sigma$：sigmoid

行为：

| 情形 | $\rho_k$ | $w_k$ | 训练行为 |
|---|---|---|---|
| teacher 与 student 同步演化 | 小 | → 1 | full DMD，听 teacher |
| teacher 突然 jump（bias active） | 大 | → 0 | fall back 到 $\mathcal{L}_{cont}$ |

注意 trust region radius 是 dynamic 的、online 的、对每个 chunk 重新计算的。这就是 paper 标题里的 "Trust Region Steering"。

---

## Fall back 分支：continuity loss

fake trajectory momentum 怎么变成可用 supervision signal？用 continuity loss：

$$\mathcal{L}_{cont} = \|f_k^{fake} - f_{k-1}^{fake}\|_2^2, \quad f_k^{fake} = \Phi(x_{fake}^{(k)})$$

变量：
- $f_k^{fake}$：fake chunk k 的 DINO descriptor
- $\mathcal{L}_{cont}$：penalize 偏离 prior evolution 的距离

关键属性：这个 loss **既不涉及 $s_{real}$ 也不涉及 $\hat{x}_{real}$**，所以 by construction 免疫于 $b(x, t; e)$。它不规定 fake trajectory 应该去哪，只保持它已有的 momentum。这就是 fall back 分支。

这个设计的妙处：它不要求另一个 oracle。student 自己的 fake trajectory 就是 oracle。zero extra supervision cost。

---

## 总 loss

$$\mathcal{L} = w_k \mathcal{L}_{DMD} + (1 - w_k) \mathcal{L}_{cont}$$

两个 branch 协同：
- $w_k \to 1$：纯 DMD，听 teacher
- $w_k \to 0$：纯 continuity，保 momentum

paper 在 ablation 里证明：两个 branch 缺一不可。

---

## 算法流程走一遍

Algorithm 1 简化版：

```python
while not converged:
    # 1. 准备 event schedule
    C = []  # KV cache
    l = 0
    p, p_next = sample_prompt_pair()
    tau = sample_switch_idx() * l_chunk
    
    # 2. 决定当前激活哪个 prompt
    p_active = p if l < tau else p_next
    
    # 3. event switch 时 recache KV（借 LongLive 机制）
    if l == tau:
        C = recache(G_theta, C, p_active)
    
    # 4. student 生成下一 chunk + teacher 给 DMD loss
    x_k = generate_next_chunk(G_theta, C, p_active)
    x_real_hat_k, L_DMD = DMD_loss(G_theta, x_k, p_active)
    
    # 5. 算 DINO feature
    f_k_fake = Phi(x_fake_k)
    f_k_real = Phi(x_real_hat_k)
    
    # 6. 算 delta
    delta_k_fake = f_k_fake - f_{k-1}_fake
    delta_k_real = f_k_real - f_{k-1}_real
    
    # 7. 算 trust region weight
    rho_k = ||delta_k_real - delta_k_fake||_2
    w_k = sigmoid(-(rho_k - mu) * s)
    
    # 8. 算 continuity loss
    L_cont = ||f_k_fake - f_{k-1}_fake||^2
    
    # 9. 总 loss + backward
    L = w_k * L_DMD + (1 - w_k) * L_cont
    L.backward()
    update_theta()
    
    l += l_chunk
```

几个工程师视角的注意点：

1. **每 chunk 一次 gradient step**：跟 streaming long tuning 一样
2. **teacher 每 chunk 被 invoke 一次**：开销主要在 teacher forward，但是 teacher frozen，没 backward
3. **DINO 是 single forward pass**：相对 teacher 开销可以忽略
4. **KV cache recache 是借的 LongLive**：paper 没在这里创新，直接复用

---

## 实验结果解读

paper 跑了 4 类评测，我挑要点讲：

### VBench（Table 1）

| Model | Subject Cons. | Background Cons. | Motion Smooth. |
|---|---|---|---|
| LongLive | 94.97 | 93.37 | 98.37 |
| MemFlow | 93.19 | 92.03 | 97.17 |
| Reward Forcing | 95.55 | 93.40 | 98.51 |
| **Delta Forcing** | **96.60** | **94.63** | **98.78** |

Delta Forcing 在 Subject / Background Consistency 上明显领先，其他维度（Aesthetic / Imaging / Dynamic Degree）与 baseline 持平。

**这个结果 pattern 是 paper thesis 的最好印证**：改进来自 cross-event transition stability 而不是 perceptual axis 的整体 shift。如果方法是"全面提升 visual quality"，所有维度都该提升。如果方法是"专门修 transition drift"，consistency 维度提升、其他维度持平。实测是后者，与 thesis 完全吻合。

### VideoAlign（Table 2）

Total 7.55（最佳），其中 TA（text alignment）0.61（远超 Reward Forcing 的 0.04）。

这个 TA 高特别有意思 —— 因为 Delta Forcing 在 teacher 不可信时 fall back 到 student momentum，**理论上应该牺牲一些 prompt following**。但实际 TA 反而更好。可能的解释：teacher 的 bias 让 student 漂到完全无关的 mode，反而损害 prompt following；Delta Forcing 把 trajectory 钉住后，prompt 反而更容易被遵守。

### User Study（Table 3）

Average rank 1.96（最佳），其中 Multi-Event Naturalness 1.78（差距最大）。

Multi-Event Naturalness 是这个 setting 最核心的指标，Delta Forcing 在这里拉开最大差距，与 design goal 完全对齐。

---

## Ablation 的 insight

这是 paper 最有信息量的部分之一，因为它证明两个 branch 缺一不可。

### 去掉 continuity loss（保留 adaptive weight）

结果：object appearance / local motion 还 consistent，但 **global scene layout 逐渐 drift**。

解释：trust region 单独能 suppress 部分 unreliable guidance，但没有 explicit anchor 把当前 prediction 钉回 trajectory。所以 condition-aligned 但 trajectory-agnostic 的 teacher supervision 仍能漏过来。

### 去掉 adaptive weight（保留 continuity loss，用原 DMD）

结果：camera 持续向上 pan，**mode-seeking** failure。

解释：biased DMD gradient 没 normalized reliability 控制，某些 motion direction 被反复放大直到 dominate 整条 trajectory。Reward Forcing 在 multi-event setting 也有类似 failure（因为 dynamics-oriented reward scaling 过度强化 motion gradient）。

**这两个 ablation 一起证实 paper 的核心 thesis**：两个 design 协同。trust region gating 负责"什么时候信"，continuity loss 负责"不信的时候 fall back 到哪"。缺一个就回到 baseline failure mode。

Reward Forcing: https://arxiv.org/abs/2512.17620

---

## Latent trajectory 可视化（Appendix A，被低估的部分）

这部分是 paper 的 hidden gem。方法：

1. 提每 frame 的 denoised latent $z_t \in \mathbb{R}^d$（VAE decode 前）
2. 收集整条序列 $\mathcal{Z} = \{z_1, \dots, z_T\}$
3. PCA 投 2D：$\tilde{z}_t = W^\top(z_t - \mu)$，$W \in \mathbb{R}^{d \times 2}$
4. 按时间顺序连成 trajectory，不同 prompt segment 用不同颜色

为啥用 PCA 不用 t-SNE / UMAP？三个原因：
- PCA 保 global geometry，t-SNE/UMAP 保 local neighborhood 但 distort global arrangement
- PCA 如实反映 transition magnitude / smoothness
- PCA deterministic、reproducible

定义好的 trajectory 的两个 criteria：
- **Within-interaction aggregation**：同一 prompt 内 frame 形成 compact cluster
- **Smooth cross-interaction displacement**：prompt switch 时 displacement 够大反映 semantic shift 但不 abrupt

Fig. 5 三个 baseline 的三种失败模式：
- **Under-reactive**：displacement 太小，continuity 保留但 adaptation 不足
- **Unstructured drift**：scatter + abrupt jump，没 temporal structure
- **Mode-seeking**：不管 event 是什么，trajectory 一直按某种 motion（尤其 camera motion）走，within-event 和 cross-event transition 分不开

Delta Forcing 在 Fig. 6 对比里展示了符合 criteria 的 trajectory：intra-state 紧凑、cross-state 平滑过渡。

**这个 visualization protocol 本身是个可复用的 diagnostic 工具**。给任何 streaming video model 都能跑一遍，看 trajectory shape 判断 failure mode。我认为这个 protocol 比 paper 主结果本身更有 long-term value，因为它是个 generalizable 的诊断方法。

t-SNE: http://jmlr.org/papers/v9/vandermaaten08a.html
UMAP: https://arxiv.org/abs/1802.03426

---

## 工程师视角的 critique

下面这些是 paper 自己没说但值得追问的点：

### 1. $\mu$ 和 $s$ 是 hyperparameter

trust region 的 threshold 和 sharpness 都要调。paper 没给 sensitivity analysis。如果换 teacher（比如 HunyuanVideo）或者换 benchmark，这俩值能不能 transfer？这是方法泛化性的关键问题。

### 2. DINO feature space 的选择是 ad hoc

DINOv2/v3 是 image-level self-supervised feature。视频有时序 drift 是 DINO 看不出来的（比如 camera 慢慢 pan，每帧 DINO feature 变化很小但 trajectory 实际在漂）。理论上应该用 video-specific feature extractor，或者至少做一个 ablation 看 DINO vs 视频专用 feature 的差异。

### 3. Bias detection 是 reactive 的

$\rho_k$ 大才能 detect，但此时 teacher 已经给出 biased guidance 了。理论上应该有 predictive 版本（chunk k 跑 teacher 前先 estimate reliability）。不过 reactive 已经足够好，因为 fall back 在 chunk 内发生，损失就是一个 chunk。

### 4. $\mathcal{L}_{cont}$ 形式可能太简单

L2 in DINO feature space 可能在某些 case 下 over-regularize，让 trajectory 过于平滑、丢掉应有的 semantic shift。一个可能改进：用 velocity matching（match $\delta_k^{fake}$ 到某个 reference velocity），而不是绝对位置 matching。

### 5. Memory-less generator 的限制

Delta Forcing build on LongLive（memoryless）。MemFlow 有 memory 机制，理论上和 Delta Forcing 正交，组合应该效果更好。但 paper 没做这个组合实验。这是个明显的 next step。

### 6. 60 秒上限

paper 评测到 60s。Rolling Forcing 已经能跑 minute-scale，Delta Forcing 在更长 horizon 的 $w_k$ 行为没验。如果 fake trajectory 自己也开始漂（长 horizon 下 student 自回归误差最终还是会累积），那 $\mathcal{L}_{cont}$ 的 bias-free 假设就弱化了。

Rolling Forcing: https://arxiv.org/abs/2509.19890

---

## 在更大的 landscape 里这个 work 站在哪

### Forcing 系列的演进

AR-DiT 蒸馏这一系列 forcing 工作，每篇都修一个 specific mismatch：

| Work | 修的 mismatch |
|---|---|
| Diffusion Forcing | train-test gap on per-token noise level |
| Self-Forcing | train-test gap at frame level |
| Causal Forcing | bidirectional → causal 的 frame-level ODE injectivity violation |
| Context Forcing | student-teacher context mismatch |
| Rolling Forcing | 长时序 KV anchor |
| Reward Forcing | diminished motion dynamics |
| **Delta Forcing** | **conditional bias under evolving event** |

Delta Forcing 修的是这个系列里**最后没被解决的一个 mismatch**，而且是个比较 subtle 的 —— 不是 train-test gap、不是架构问题、不是 reward 问题，而是 supervision reliability 问题。

### 与 RLHF 的连接

teacher / reward model 在 OOD state 上不可信是 RLHF 里反复出现的 theme。最近 "Distribution Matching Distillation meets Reinforcement Learning"（arXiv:2511.13649）和 "Optimizing few-step generation with adaptive matching distillation"（arXiv:2602.07345）都意识到 DMD gradient 不能被 uniform 应用。

Delta Forcing 给了一个非常 cheap 的 reliability probe（DINO delta distance）。比起 RLHF 里需要训 separate reliability model 的方案优雅很多。这个 idea 可能 transfer 到其他 distillation 场景。

### 与 process reward / verifier 的连接

DINO feature 在这里其实扮演了 verifier 角色：判别 teacher 这次 jump 是不是与 student trajectory 兼容。这和 process reward model（PRM）思路接近，只不过 PRM 训一个专门 model，Delta Forcing 直接用现成 self-supervised feature。如果 future work 想更精细，可以用专门 fine-tune 的 verifier 替换 DINO。

### World model 视角

interactive streaming video generation 是 world model 的一个 instance（OpenAI Sora blog 就把 video model 当 world simulator）。conditional bias 在 world model 视角下就是：**transition dynamics 在 condition 切换时被 spurious 拉到 prior mode**。

这和 model-based RL 里 transition model over-fit 到 reward signal 的现象有 conceptual parallel。Trust region 在 model-based RL 里也是经典 topic。Delta Forcing 的 trust region 在 observation space 而非 parameter space，可能给 model-based RL 提供新思路。

Sora as world simulator: https://openai.com/research/video-generation-models-as-world-simulators

### 与 consistency model / rectified flow distillation 的连接

rectified flow / consistency model 的 distillation 也有类似 reliability 问题：teacher 在某些区域给出 noisy target。Delta Forcing 的 trust region 思想或许可以 transfer 过去，定义 latent trajectory consistency 作为 reliability proxy。

---

## 给 Karpathy 这种 reader 的几个 actionable 思考点

1. **Conditional bias 形式化可以在 toy distribution 上验证**：构造一个 2D Gaussian mixture，让 teacher 在两个 mode 之间 marginalized，看 student 是不是真的被拉到 average mode。这种 controlled experiment 比 paper 现有实验更能 isolate 这个现象。

2. **$\mu, s$ sensitivity 是方法泛化性的关键**：可以跑 grid search，看 $w_k$ 行为如何变化、最佳点是不是 stable。如果在不同 teacher / benchmark 上最佳点差很多，说明方法的"鲁棒性"是 marketing 语言而非工程现实。

3. **Latent trajectory PCA protocol 是个 generalizable diagnostic 工具**：这个 protocol 在你自己任何 streaming model 上都能跑。判断任何 future improvement 是真的 fix 了 trajectory 还是只是在 pixel space 修修补补。这个 protocol 比 paper 主结果本身更值得复用。

4. **DINO feature vs video-specific feature 的 ablation**：DINO 是 image-level，理论上 video-level feature（比如 VideoMAE 或 TimeSformer 的 feature）会更敏感于时序 drift。这个 ablation paper 没做。

5. **Delta Forcing + MemFlow 组合**：paper build on LongLive（memoryless），但 Delta Forcing 与 memory mechanism 正交。组合实验应该效果更好。这是个 obvious next paper。

6. **$\mathcal{L}_{cont}$ 形式探索**：L2 in feature space 是最简版本。可以试 velocity matching、动量 conservation、甚至基于 optical flow 的 continuity。这块有 design space 没被 explored。

---

## Reference 汇总

主线工作：
- Delta Forcing project: https://delta-forcing-website.vercel.app/
- Wan2.1: https://arxiv.org/abs/2503.20314
- LongLive: https://arxiv.org/abs/2510.01783
- MemFlow: https://arxiv.org/abs/2512.05457
- Causal Forcing: https://arxiv.org/abs/2602.06123

Forcing 系列：
- Diffusion Forcing: https://arxiv.org/abs/2411.17686
- Self-Forcing: https://arxiv.org/abs/2509.21494
- Self-Forcing++: https://arxiv.org/abs/2510.02283
- Context Forcing: https://arxiv.org/abs/2602.06028
- Rolling Forcing: https://arxiv.org/abs/2509.19890
- Reward Forcing: https://arxiv.org/abs/2512.17620
- Anchor Forcing: https://arxiv.org/abs/2603.13405
- Grounded Forcing: https://arxiv.org/abs/2604.06939
- HiAR: https://arxiv.org/abs/2603.08703
- Diagonal Distillation: https://arxiv.org/abs/2603.09488

蒸馏 / reliability 相关：
- DMD: https://arxiv.org/abs/2312.04261
- Improved DMD: https://arxiv.org/abs/2405.14867
- DMD meets RL: https://arxiv.org/abs/2511.13649
- Adaptive matching distillation: https://arxiv.org/abs/2602.07345
- TRPO: https://arxiv.org/abs/1502.05477

Feature / 可视化：
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- t-SNE: http://jmlr.org/papers/v9/vandermaaten08a.html
- UMAP: https://arxiv.org/abs/1802.03426

Benchmark / evaluation：
- VBench: https://arxiv.org/abs/2311.13582
- VBench-2.0: https://arxiv.org/abs/2503.21755
- Long-CLIP: https://arxiv.org/abs/2403.15378
- VideoAlign: https://arxiv.org/abs/2501.13918

Context：
- Sora as world simulator: https://openai.com/research/video-generation-models-as-world-simulators
- HunyuanVideo 1.5: https://arxiv.org/abs/2511.18870

---

如果你接下来想下钻，我推荐三个最有价值方向：
1. 在 toy 2D Gaussian mixture 上 visualize conditional bias
2. $\mu, s$ sensitivity grid 看 $w_k$ 行为
3. Delta Forcing + MemFlow 组合实验

挑一个我都可以展开。

---

# Delta Forcing：Trust Region Steering for Interactive Autoregressive Video Generation

这篇 paper 来自 KAIST / Visko Platform / Texas A&M，第一作者 Yuheng Wu（KAIST 实习生），项目 lead 是 Zhengzhong Tu。它解决的是 interactive streaming video generation 里一个非常具体但被前人忽略的失败模式：**conditional bias** —— frozen teacher 在 event switch 时给出的 guidance 是 condition-consistent 但 trajectory-inconsistent 的，导致 student 长时序漂移。核心解法借鉴 TRPO 的 trust region 思想：用 fake trajectory 自身的 momentum 作为 bias-free fallback，根据 teacher 与 student 在 DINO feature 空间的 delta discrepancy 在线决定何时信 teacher、何时退回 momentum。

我下面按 motivation → formal diagnosis → design → formula breakdown → algorithm → experiments → ablation → broader context 的顺序讲，目的都是 build intuition。

---

## 1. Setting：interactive streaming video generation 的特殊性

Offline generic video generation（Wan2.1、HunyuanVideo、Sora、Kling-Omni 等）的 setting 是给一个 prompt，生成一段 clip，结束。Interactive streaming 的 setting 完全不同：

- **Causal**：只能用 past → future，没法用 bidirectional attention
- **Real-time**：要求 few-step inference（AR-DiT 蒸馏）
- **Multi-event evolving condition**：prompt 在生成过程中动态切换（"grandfather enters..." → "talk back..."），且要求 smooth transition，不允许 hard cut

这给了一个根本性 tension：**reactivity（对新 event 反应快）vs stability（长时序保持 identity / layout / scene 一致）**。

现有 pipeline（LongLive、MemFlow 等都是这个范式）：

- **Stage 1**：把 bidirectional diffusion teacher（如 Wan2.1-14B-T2V）通过 DMD 蒸馏成 causal、few-step AR-DiT student（如 Wan2.1-1.3B-T2V），用 Self-Forcing / Causal Forcing / Diffusion Forcing 等 forcing-based approach
- **Stage 2**：Streaming Long Tuning —— 在长序列上 fine-tune，序列中嵌入 event switch，让 student 在自己的 rollout 上学习

这个 pipeline 在 5-10 秒短 clip 上效果不错，但是当序列变长（paper 评测是 60 秒 = 6 个 10s event）就出现 drift：object identity 变（深蓝衣服 → 白衣服）、layout 偏、scene 漂。

Reference: 
- LongLive: https://arxiv.org/abs/2510.01783
- MemFlow: https://arxiv.org/abs/2512.05457
- Causal Forcing: https://arxiv.org/abs/2602.06123
- Wan2.1: https://arxiv.org/abs/2503.20314

---

## 2. 之前归因 vs Delta Forcing 归因

之前 LongLive / MemFlow / Self-Forcing / Causal Forcing 几乎都把 drift 归因为 **autoregressive generation error accumulation**，即 student 自回归误差一步步累积放大。

Delta Forcing 通过 decode teacher 的 output 与 student generation 在 pixel space 比较，发现 drift 不是随机累积，而是 **从 teacher supervision signal 继承来的**。teacher 自身在 event switch 时就 shift 到一个新的 semantic mode，student 随后跟随 drift 到同一个方向。Fig. 1 right 直观展示这点。

---

## 3. Conditional Bias 的形式化诊断（这是 paper 最 sharp 的贡献）

理想的 generative target 同时依赖 history $h_{e-1}$ 和新 condition $c_e$：

$$p^*(x | h_{e-1}, c_e), \quad s^*(x, t | h_{e-1}, c_e) = \nabla_x \log p_t^*(x | h_{e-1}, c_e)$$

变量：
- $e \in \{1, 2, ...\}$: event index
- $c_e$: event e 的 condition（prompt / action / control signal）
- $h_{e-1}$: event e 之前累积的 cached history
- $x$: noisy latent sample
- $t$: noise level
- $s^*$: teacher score function
- $p_t^*$: teacher 的 marginal distribution at noise level t

但问题是：pretrained bidirectional video teacher（Wan、Hunyuan）训练在 short clip + single static condition 上，**根本没有跨 event 条件切换机制**。当 student 在 stage 2 把历史 $h_{e-1}$ 喂给 teacher 时，teacher 没法 condition 在这个具体 history 上，于是 student 实际 match 的 score 是 history-marginalized 的：

$$\bar{s}^*(x, t | c_e) = \mathbb{E}_{h_{e-1} \sim p(\cdot | c_e)}[s^*(x, t | h_{e-1}, c_e)]$$

含义：把所有可能的 incoming history 按 $p(\cdot | c_e)$ 加权平均掉，得到一个只取决于 $c_e$ 的 marginalized score。Trajectory-dependent information 被 average 掉。

定义 per-event bias：

$$b(x, t; e) \triangleq s^*(x, t | h_{e-1}, c_e) - \bar{s}^*(x, t | c_e)$$

把 DMD 的 gradient 展开（注意原文公式 (5)）：

$$\nabla_\theta \mathcal{L}_{DMD}^{biased} = -\mathbb{E}[(\bar{s}^* - s_{fake}) \frac{\partial G_\theta}{\partial \theta}] - \mathbb{E}[b(x, t; e) \frac{\partial G_\theta}{\partial \theta}]$$

变量：
- $\theta$: generator 参数
- $s_{fake}$: fake critic 学到的 score（DMD 里的 fake model，learnable）
- $G_\theta$: generator
- 第一项：nominal DMD update（对 marginalized teacher 的）
- 第二项（标 conditional bias）：history–condition mismatch 引入的 spurious direction

Intuition：teacher 的 score 在 event switch 时被 new condition 拉到一个新的 mode，但是这个 mode 与 student 已生成的 history 不兼容（e.g., student 已经生成了深蓝衣服的老人，teacher 在 "talk back" 条件下输出白衣服的老人，因为 teacher 不知道 student 之前生成了深蓝）。**这种 mismatch 通过 DMD 的 score-difference 形式被反复注入 student**，造成 drift。这就是 conditional bias。

---

## 4. TRPO 类比：trust region for unreliable teachers

Delta Forcing 借 TRPO 的 trust region 思想。TRPO 在 RL 里的核心是：**advantage estimate 只在 policy 不动太远的 region 内可信**，因此用 KL constraint 把 policy update 限制在 trust region 内。

Delta Forcing 把这个思想迁移：teacher guidance 不应被 uniform trust。但 TRPO 的 trust region 在 parameter space、需要二阶信息（Fisher matrix）来定义 KL ball。**Delta Forcing 把 trust region 移到 observable latent trajectory space**，只需要一个 feature-space distance，online 就能算。

TRPO reference: https://arxiv.org/abs/1502.05477

---

## 5. 关键观察：fake critic 与 real teacher 的不对称性

这是 Delta Forcing 设计上的 critical insight。DMD 里有两个 score：

- $s_{real}$（teacher）：**frozen**，一旦 biased 就 persistently pull $G_\theta$ off-trajectory
- $s_{fake}$（fake critic）：**dynamic**，在 $G_\theta$ 自己生成的 sample 上更新，因此会**累积反映实际生成 history 的 momentum**

也就是说 fake critic by construction 是 history-grounded、bias-free 的（因为 fake 只见 student 自己生成的数据）。这就给出了一个两段式设计：用 reliability weight 在 real score 之间调制，reliability 低时 fall back 到 fake trajectory 的 momentum。

DMD reference: https://arxiv.org/abs/2312.04261
Improved DMD: https://arxiv.org/abs/2405.14867

---

## 6. Delta Forcing objective 与两个 design

总目标函数（原文公式 (6)）：

$$\mathcal{L} = w_k \mathcal{L}_{DMD} + (1 - w_k) \mathcal{L}_{cont}$$

变量：
- $w_k \in (0, 1)$: 在 chunk $k$ 的 online reliability weight，**实现 trust region radius**
- $\mathcal{L}_{cont}$: 基于 fake trajectory momentum 的 bias-immune continuity loss

注意它是 chunk-level（不是 step-level），与 LongLive 的 chunk-wise autoregressive 范式匹配。

### Design 1: Continuity loss on fake trajectory

generator 生成 fake latent chunk：

$$x_{fake}^{(k)} \sim G_\theta(C_k, c_{e(k)})$$

变量：
- $C_k$: 累积 KV cache，编码 history $h_{e(k)-1}$
- $e(k)$: chunk $k$ 所在的 event index

continuity loss：

$$\mathcal{L}_{cont} = \|f_k^{fake} - f_{k-1}^{fake}\|_2^2, \quad f_k^{fake} = \Phi(x_{fake}^{(k)})$$

变量：
- $\Phi$: **frozen DINO**（DINOv2 / DINOv3）feature extractor
- $f_k^{fake}$: fake chunk k 的 semantic descriptor

关键属性：$\mathcal{L}_{cont}$ **不涉及 $s_{real}$ 也不涉及 $\hat{x}_{real}^{(k)}$**，因此免疫于 $b(x,t;e)$。它不规定 fake trajectory 应该去哪里，只是 penalize 偏离 prior evolution，保持 trajectory 自身的 momentum。

DINOv2: https://arxiv.org/abs/2304.07193
DINOv3: https://arxiv.org/abs/2508.10104

### Design 2: Adaptive trust region from latent-delta discrepancy

Empirical 观察：conditional bias 不表现为 uniform noise，而是 **teacher chunk-to-chunk evolution 中的 sharp discontinuity**。在 stable event 内 teacher 演化 smooth；event switch 时 teacher 突然 jump 到新 mode。这就把 reliability estimation 从"训练动态里推断 hidden quantity"变成"在线观测 teacher chunk-level evolution 的 smoothness"。

具体地，在 DINO feature 空间计算两个 delta：

$$\delta_k^{fake} = \Phi(\hat{x}_{fake}^{(k)}) - \Phi(\hat{x}_{fake}^{(k-1)})$$

$$\delta_k^{real} = \Phi(\hat{x}_{real}^{(k)}) - \Phi(\hat{x}_{real}^{(k-1)})$$

变量：
- $\hat{x}_{fake}^{(k)}$ / $\hat{x}_{real}^{(k)}$: student / teacher 的 denoised estimate at chunk k
- $\delta_k^{fake}$: fake trajectory 的 chunk-level evolution
- $\delta_k^{real}$: real trajectory 的 chunk-level evolution

注意这两个 delta 是用同一个 $\Phi$ 编码再做差，所以反映的是 semantic-level shift 而不是 pixel-level noise。DINO space 对 pixel-level noise 鲁棒，对 semantic jump 敏感，正好满足需求。

Trust region gating（原文公式 (11)）：

$$\rho_k = \|\delta_k^{real} - \delta_k^{fake}\|_2$$

$$w_k = \sigma(-(\rho_k - \mu) \cdot s)$$

变量：
- $\rho_k$: **delta discrepancy**，量化 teacher evolution 与 student evolution 的偏离
- $\mu$: **detection threshold**，决定什么时候算 bias active
- $s$: **sharpness** of sigmoid transition
- $\sigma$: sigmoid function

行为：
- $\rho_k$ 小 → teacher 与 student 同步演化 → trust region 宽 → $w_k \to 1$ → full DMD
- $\rho_k$ 大 → teacher 突然 jump（bias active） → trust region 收紧 → $w_k \to 0$ → fall back 到 $\mathcal{L}_{cont}$

这个 trust region 与经典 trust region 不同：它直接 live 在 observable latent trajectory space，**不需要二阶 gradient 信息**，online 可从单个 feature distance 算出。

---

## 7. Algorithm 1 详解

Algorithm 1 的整体流程：

```
while not converged:
  initialize KV cache C = []
  l = 0  # current video length
  sample (p, p_next) from prompt set  # 当前 event + 下一个 event
  sample switch index τ
  τ = τ * l_chunk  # 把 τ 转成帧数

  if l >= l_video:
    reset C, l, resample

  p_active = p if l < τ else p_next  # 决定当前激活哪个 prompt

  if l == τ:  # event switch
    C_recache(G_θ, ∇v, C, p_active)  # KV cache recache，借鉴 LongLive

  x^(k) = generate_next_chunk(G_θ, C, p_active)
  (x̂_real^(k), L_DMD) = DMD_Loss(G_θ, x^(k), p_active)

  # 用 DINO 提 feature
  f_k^fake = Φ(x_fake^(k))
  f_k^real = Φ(x̂_real^(k))

  # 算 delta
  δ_k^fake = f_k^fake - f_{k-1}^fake
  δ_k^real = f_k^real - f_{k-1}^real

  # 算 trust region weight
  ρ_k = ||δ_k^real - δ_k^fake||_2
  w_k = σ(-(ρ_k - μ) · s)

  # 总 loss
  L_cont = ||f_k^fake - f_{k-1}^fake||_2^2
  L = w_k * L_DMD + (1 - w_k) * L_cont

  .backward()
  update θ
  l += l_chunk
```

几个直觉点：

1. **每 chunk 一次梯度更新**：与 streaming long tuning 一致，每 chunk 拿一个 gradient step
2. **KV cache recache 在 event boundary**：直接用 LongLive 的机制，在 prompt switch 时 reset 部分 stale context 但保留 visual continuity
3. **teacher 在 chunk 内被 invoke 一次**：DMD Loss 内部会跑 teacher 得到 $\hat{x}_{real}^{(k)}$，然后才提 DINO feature
4. **DINO feature 是 single forward pass**：开销小

---

## 8. 实验细节

**Implementation**：
- Generator: Wan2.1-1.3B-T2V
- Teacher: Wan2.1-14B-T2V（native 5s clip @ 16 FPS, 832×480）
- Stage 1: 替换原 Self-Forcing init 为 Causal Forcing init（针对 context window + frame sink），700 steps，lr $2 \times 10^{-6}$ (gen) / $4 \times 10^{-7}$ ($s_{fake}$)
- Stage 2: Delta Forcing + Streaming Long Tuning，3000 steps，lr $1 \times 10^{-5}$ (gen) / $2 \times 10^{-6}$ ($s_{fake}$)
- 硬件：Nvidia H100

**Baselines**：
- Diffusion Forcing Causal 组：SkyReels-V2、MAGI-1
- Distilled Causal 组：LongLive、MemFlow、Reward Forcing（都从 Causal Forcing init，且 Reward Forcing 额外加 Streaming Long Tuning 公平对比）

**Evaluation 4 个 axis**：
1. VBench：6 个维度（Subject Consistency / Background Consistency / Motion Smoothness / Aesthetic Quality / Imaging Quality / Dynamic Degree）
2. Long-CLIP：测 instruction-following，每个 10s chunk 都打分（0-10s、10-20s、...、50-60s）
3. VideoAlign：reward model from human feedback，评 VQ / MQ / TA；评测时把每个 chunk 与前一 chunk 最后 1s 拼起来打分，强调 boundary continuity
4. User study：20 个 participant，每个 3 trial，4 个 video 并排 rank，3 个 criterion（Aesthetic / Dynamic / Multi-Event Naturalness）

**Quantitative highlights**：

VBench（Table 1）：
- Delta Forcing Subject Consistency 96.60（vs Reward Forcing 95.55、LongLive 94.97、MemFlow 93.19）
- Background Consistency 94.63（最佳）
- Motion Smoothness 98.78（最佳）
- 其他维度与 baseline 持平，说明改进来自 transition stability 而非 shift 到某个 single perceptual axis

Long-CLIP（Table 2）：Avg 26.07（最佳，与 LongLive 26.05 接近，但早期 0-10s 是 28.13，明显高于所有 baseline 的 27.x），说明 reactivity 没丢。

VideoAlign（Table 2）：Total 7.55（最佳），MQ 4.18（与 Reward Forcing 并列第一），TA 0.61（最佳，远超 Reward Forcing 的 0.04）。

User Study（Table 3）：Average rank 1.96（最佳），其中 Multi-Event Naturalness 1.78（差距最大，与 Delta Forcing 的 design goal 完全吻合）。

---

## 9. Ablation 的 insight（这部分很有信息量）

**Without continuity loss（去掉 $\mathcal{L}_{cont}$，只保留 adaptive modulation）**：
- 结果：object appearance / local motion 还算 consistent，但 **global scene layout 逐渐 drift**
- 含义：trust region 单独能 suppress 部分 unreliable teacher guidance，但是没有 explicit anchor 把当前 prediction 钉回 trajectory。所以 condition-aligned but trajectory-agnostic 的 teacher supervision 仍能漏过来

**Without adaptive trust region（去掉 $w_k$，只用原 DMD loss）**：
- 结果：camera 持续向上 pan，出现典型 AR-DiT distillation 的 **mode-seeking** failure
- 含义：biased DMD gradient 没 normalized reliability 控制，某些 motion direction 被放大直到 dominate 整条 trajectory。Reward Forcing 在 multi-event setting 也有类似 failure（因为 dynamics-oriented reward scaling 过度强化 motion gradient）
- 引申：原 DMD objective 在 evolving condition 下不应被 uniform 鼓励

这两个 ablation 一起证实 paper 的核心 thesis：**两个 design 是协同的，缺一个都会回归到 baseline failure mode**。

Reference: 
- Reward Forcing: https://arxiv.org/abs/2512.17620
- DMD meets RL: https://arxiv.org/abs/2511.13649
- Adaptive matching distillation: https://arxiv.org/abs/2602.07345

---

## 10. Latent trajectory 可视化（Appendix A，非常值得读）

这部分独立于主结果，但其实是 build intuition 的金矿。方法：

1. 提每个 frame 的 denoised latent $z_t \in \mathbb{R}^d$（VAE decode 之前）
2. 收集整条序列 $\mathcal{Z} = \{z_1, ..., z_T\}$
3. PCA 投到 2D：$\tilde{z}_t = W^\top(z_t - \mu)$，$W \in \mathbb{R}^{d \times 2}$ 是 top-2 principal direction
4. 按时间顺序连成 trajectory，不同 prompt segment 用不同颜色

为什么用 PCA 而非 t-SNE / UMAP？三个原因：
- PCA 保 global geometry，t-SNE/UMAP 保 local neighborhood 但 distort global arrangement
- PCA 如实反映 transition 的 magnitude 和 smoothness
- PCA deterministic、reproducible；t-SNE/UMAP 对 perplexity、seed 敏感

定义 good trajectory 的两个 criteria：
- **Within-interaction aggregation**：同一 prompt 内的 frame 形成 compact cluster
- **Smooth cross-interaction displacement**：prompt switch 时 displacement 要够大反映 semantic change 但不 abrupt

Fig. 5 三个 baseline 的失败模式：
- ❶ Under-reactive：displacement 太小，continuity 保留但 adaptation 不足
- ❷ Unstructured drift：scatter + abrupt jump，没有 temporal structure（典型 baseline failure）
- ❸ Mode-seeking：不管 event 是什么，trajectory 一直按某种 motion（尤其 camera motion）走，within-event 和 cross-event transition 分不开

Delta Forcing 在 Fig. 6 的对比里展示了符合 criteria 的 trajectory：intra-state 紧凑、cross-state 平滑过渡。

这个 visualization protocol 本身是个可复用的诊断工具。给任何 streaming video model 都能跑一遍，看看 trajectory shape。

t-SNE: http://jmlr.org/papers/v9/vandermaaten08a.html
UMAP: https://arxiv.org/abs/1802.03426

---

## 11. 联系到更广的 research context

### 11.1 与 RLHF / DPO 里的 reliability 问题同构

teacher / reward model 在 out-of-distribution state 上不可信是 RLHF 里反复出现的 theme。最近 "Distribution Matching Distillation meets Reinforcement Learning"（arXiv:2511.13649）和 "Optimizing few-step generation with adaptive matching distillation"（arXiv:2602.07345）都意识到 DMD gradient 不能被 uniform 应用。Delta Forcing 给了一个非常 cheap 的 reliability probe（DINO delta distance），比起 RLHF 里需要训 separate reliability model 的方案优雅很多。

### 11.2 与 self-forcing 系列的关系

self-forcing 的核心 insight 是 close train-test gap，让 student 在训练时就 condition on 自己生成的 rollout。Delta Forcing 进一步：不仅 condition on 自己的 rollout，**还要把 teacher guidance 限制在与自己 rollout 一致的 region 内**。这是个递进关系：

- Self-Forcing: close train-test gap
- Causal Forcing: fix bidirectional→causal distillation 的 frame-level ODE injectivity violation
- Context Forcing: 修 student-teacher context mismatch
- Rolling Forcing: 长时序 KV anchor
- Reward Forcing: fix diminished motion dynamics
- Delta Forcing: fix conditional bias

这一系列 forcing 工作（[Self-Forcing++](https://arxiv.org/abs/2510.02283)、[Anchor Forcing](https://arxiv.org/abs/2603.13405)、[Grounded Forcing](https://arxiv.org/abs/2604.06939)、[HiAR](https://arxiv.org/abs/2603.08703)、[Diagonal Distillation](https://arxiv.org/abs/2603.09488)）每一篇都解决一个具体 mismatch，Delta Forcing 解决的是 teacher reliability under evolving condition。

### 11.3 与 process reward / verifier 的关系

DINO feature 这里扮演了一个 verifier 的角色。它判别 "teacher 这次的 jump 是不是与 student trajectory 兼容"。这和 process reward model（PRM）思路接近，只不过 PRM 训一个专门 model，Delta Forcing 直接用现成 self-supervised feature。如果 future work 想更精细，可能可以用专门 fine-tune 的 verifier 替换 DINO。

### 11.4 与 consistency models / rectified flow 的潜在连接

rectified flow / consistency model 的 distillation 也有类似 reliability 问题：teacher 在某些区域给出 noisy target。Delta Forcing 的 trust region 思想或许可以 transfer 过去，定义 latent trajectory consistency 作为 reliability proxy。

### 11.5 World model 的角度

interactive streaming video generation 是 world model 的一个 instance（OpenAI Sora blog 就把 video model 当 world simulator）。Conditional bias 在 world model 视角下就是：**transition dynamics 在 condition 切换时被 spurious 拉到 prior mode**。这和 model-based RL 里 transition model over-fit 到 reward signal 的现象有 conceptual parallel。Trust region 在 model-based RL 里也是经典 topic。

Sora as world simulator: https://openai.com/research/video-generation-models-as-world-simulators

---

## 12. 局限与未解决的问题

paper 自己没明说，但从结果和方法看可以推出几个潜在问题：

1. **$\mu$ 和 $s$ 是 hyperparameter**：trust region 的 threshold 和 sharpness 需要调。paper 给了具体值但没给 sensitivity analysis。如果不同 benchmark / 不同 teacher 需要不同值，迁移成本不小。
2. **DINO feature space 假设**：DINOv2/v3 是 image-level self-supervised feature，对视频时序不一定最优。可能出现 DINO 看不出来的 jump（temporal-only drift）。
3. **bias detection 是 reactive 的**：只在 jump 已经发生（$\rho_k$ 已经大）时才 detect，没有预测性。理论上可以在 chunk k 跑 teacher 前先 estimate reliability。
4. **$\mathcal{L}_{cont}$ 是 L2 in feature space**：可能 over-regularize，让 trajectory 过于平滑、丢掉应有的 semantic shift。paper 没 ablate 不同 continuity loss form。
5. **Memory-less generator**：Delta Forcing build on LongLive（memoryless），但 MemFlow 有 memory。理论上把 Delta Forcing 加到 MemFlow 上效果应该更好，但 paper 没做这个组合实验。
6. **60 秒上限**：评测到 60s，scaling 到 minute-scale 时 $w_k$ 长程行为没验。

---

## 13. Take-away intuition

总结几个直觉：

1. **Frozen teacher under evolving condition 是 biased 的**：teacher 训练在 short clip + static condition，不知道 history，给 marginalized score，造成 conditional bias。
2. **Fake critic by construction 是 history-grounded 的**：因为它只在 student 自己生成的 sample 上更新，自然带 student 的 trajectory momentum。这个不对称是 Delta Forcing 设计的支点。
3. **Bias 在 feature space 表现为 sharp jump**：所以用 DINO feature 算 chunk-to-chunk delta 就能 detect，比从训练动态推断 reliability 简单太多。
4. **Trust region 可以在 observable latent space 定义**：不需要二阶 gradient、不需要参数 space KL，只需要一个 feature distance + sigmoid gating。
5. **两个 design 必须同时存在**：去掉 continuity loss → global layout drift；去掉 adaptive weight → mode-seeking camera panning。两个 branch 协同。
6. **改进来自 transition stability 而非 perceptual axis shift**：VBench 各维度与 baseline 持平、只有 Subject/Background Consistency 大幅提升，VideoAlign 在 TA（text alignment）上也提升，user study 在 Multi-Event Naturalness 上拉开最大差距。这些都印证 thesis。

---

## 14. 进一步阅读

- Project page: https://delta-forcing-website.vercel.app/
- Wan2.1 technical report: https://arxiv.org/abs/2503.20314
- DMD original: https://arxiv.org/abs/2312.04261
- Improved DMD: https://arxiv.org/abs/2405.14867
- LongLive: https://arxiv.org/abs/2510.01783
- MemFlow: https://arxiv.org/abs/2512.05457
- Self-Forcing: https://arxiv.org/abs/2509.21494
- Self-Forcing++: https://arxiv.org/abs/2510.02283
- Causal Forcing: https://arxiv.org/abs/2602.06123
- Context Forcing: https://arxiv.org/abs/2602.06028
- Rolling Forcing: https://arxiv.org/abs/2509.19890
- Reward Forcing: https://arxiv.org/abs/2512.17620
- TRPO: https://arxiv.org/abs/1502.05477
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- VBench: https://arxiv.org/abs/2311.13582
- VBench-2.0: https://arxiv.org/abs/2503.21755
- Long-CLIP: https://arxiv.org/abs/2403.15378
- VideoAlign: https://arxiv.org/abs/2501.13918
- DMD meets RL: https://arxiv.org/abs/2511.13649
- Adaptive matching distillation: https://arxiv.org/abs/2602.07345
- t-SNE: http://jmlr.org/papers/v9/vandermaaten08a.html
- UMAP: https://arxiv.org/abs/1802.03426
- Sora world simulator: https://openai.com/research/video-generation-models-as-world-simulators

---

如果你想下钻某一块，我建议三个最有价值的方向：

1. **Conditional bias 的形式化**：公式 (4)-(5) 是 paper 的核心诊断，可以把 history marginalization 在 toy distribution 上可视化（比如 2D Gaussian mixture），看 marginalized score 怎么偏离 trajectory-conditional score。
2. **Trust region gating 的 sensitivity**：可以跑一组 $\mu, s$ 的 grid，看 $w_k$ 行为如何变化、最佳点是不是 stable。这关系到方法在不同 teacher / benchmark 上迁移的鲁棒性。
3. **Latent trajectory PCA 可视化协议**：这个 protocol 在你自己任何 streaming model 上都能跑，作为 diagnostic 工具的复用价值很高，可以判断任何 future improvement 是真的 fix 了 trajectory 还只是在 pixel space 修修补补。
