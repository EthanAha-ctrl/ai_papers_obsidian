---
source_pdf: JOINT-EMBEDDING PREDICTIVE WORLD MODELS.pdf
paper_sha256: eed9c8e1439a5cdd95300db69492dcfb37f6964b49f633eb16894f180f4d455e
processed_at: '2026-08-05T10:55:41-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 JEPA-WM paper

## 先一句话概括故事

这帮 Meta FAIR 的人想搞清楚一件事：在 latent space 里做 world model + planning 这条路线上，到底哪些 design choice 真的 matters？于是他们把两个 baseline（DINO-WM 和 V-JEPA-2-AC）拉到一个统一框架下，逐个 component 做 ablation，最后凑出一个 best config 把 baseline 都 beat 了。

听起来很朴素，但有意思的地方在于：他们发现一堆 "obvious" 的假设其实是错的，或者至少是 task-dependent 的。

## 背景：为啥要搞 world model

LeCun 老爷子从 2022 年开始就一直推 JEPA 这个 idea（参考他的 position paper: https://openreview.net/forum?id=BZ5a1r-kVsf）。核心论点很简洁：

- 在 pixel space 预测未来是浪费的，背景、光照、纹理这些细节对 planning 没用
- 应该在 abstract representation space 预测
- Encoder + Predictor 联合训，让 representation 自然变成 predictive useful 的形态

听起来很 clean，但落地到 robotics 有很多 engineering choice 要做。DINO-WM (https://arxiv.org/abs/2411.04983) 和 V-JEPA-2-AC (https://arxiv.org/abs/2506.07909) 是两个早期尝试，但谁都没系统 ablate 过各个 component。

paper 的 contribution 就是把这个 gap 填上。

## JEPA-WM 是个啥：用大白话

想象一个 robot 在干活，每一步它看到一张 image $o_t$，执行一个 action $a_t$，然后世界变成下一帧 $o_{t+1}$。

JEPA-WM 干的事：

1. 把 image 喂给一个 frozen visual encoder $E_\phi^{vis}$（比如 DINOv2），拿到一个 embedding $z_t = E_\phi^{vis}(o_t)$
2. 把 action 喂给一个 action encoder $A_\theta(a_t)$
3. 把 $z_t$ 和 $A_\theta(a_t)$ 喂给一个 predictor $P_\theta$，预测下一帧的 embedding $\hat{z}_{t+1} = P_\theta(z_t, A_\theta(a_t))$
4. loss 就是 $\|\hat{z}_{t+1} - z_{t+1}\|^2$，其中 $z_{t+1}$ 是用 ground truth 下一帧算的 embedding

公式版本（Eq 1）：

$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} L[P_\theta(E_{\phi,\theta}(o_{t-w:t}^b), A_\theta(a_{t-w:t}^b)), E_{\phi,\theta}(o_{t+1}^b)]$$

变量说一下：
- $B$ 是 batch size
- $o_{t-w:t}$ 是从 $t-w$ 到 $t$ 的 observation 窗口，长度 $w+1$
- $E_{\phi,\theta} = (E_\phi^{vis}, E_\theta^{prop})$ 是 visual encoder (frozen) + 可选的 proprioception encoder
- $A_\theta$ 是 action encoder
- $L$ 是 MSE，pairwise 在 visual 和 proprio modality 上算

Planning 的时候：给定 initial state $o_t$ 和 goal $o_g$，在 action space $\mathbb{R}^{H \times A}$ 里搜，找到让 unrolled final embedding 接近 goal embedding 的 action sequence。

规划目标（Eq 2）：

$$L_\alpha^p(o_t, a_{t:t+H-1}, o_g) = (L_{vis} + \alpha L_{prop})(G_{\phi,\theta}(o_t, a_{t:t+H-1}), E_{\phi,\theta}(o_g))$$

- $H$ 是 planning horizon
- $\alpha$ 是 proprioception 权重，平衡两个 modality
- $G_{\phi,\theta}$ 是 unrolling 函数，把 action sequence 递归喂给 predictor 得到 final embedding

直觉上特别像 model predictive control (MPC)，只不过 dynamics 是 learned 的，且在 latent space 而非 state space 操作。

## 他们 ablate 了啥

按照影响范围排序（planning-time 优先，因为影响所有 evaluation）：

### 1. Planner 选择

比了 4 种：
- **CEM** (Cross-Entropy Method)：sample N 个 action trajectory，unroll 算 cost，选 top K 拟合 Gaussian，迭代
- **NG** (Nevergrad)：用 NGOpt meta-optimizer 自动选 diagonal CMA-ES，hyperparameter 更少
- **GD**：直接 gradient descent on action sequence
- **Adam**：用 Adam 优化 action sequence

结果挺有意思（看 Figure 3）：

- CEM $L_2$ 整体最好
- **GD/Adam 在 Metaworld 上很强**，因为 Metaworld 是 greedy reachable task（goal 就在那，直接 reach 过去），cost landscape smooth，gradient 工作
- **GD/Adam 在 2D navigation (Wall, Maze) 上灾难性失败**（基本 0%），因为 cost landscape 有 local minima，agent 撞墙或者走到图像边界找到虚假 minimum
- **NG 在 DROID/Robocasa 上和 CEM 持平**，且不用 tune CEM 的 top-K、初始 $\sigma$ 这些敏感 hyperparameter

paper 里有个很好的图（Figure S5.1）展示 GD 在 Wall 上的两种 failure mode：一种是 agent 撞墙后找不到门；另一种是 agent 起始位置接近图像边缘时，走到边缘找到一个 "假装到达 goal" 的 local minimum（其实只是 visual 上离 goal 像而已）。

直觉：sampling-based 方法能 escape local minima，gradient-based 在 non-convex landscape 上脆弱。这跟 RL/optimization 文献里的常识一致，但 paper 用具体 failure case 生动展示了。

另外 $L_2$ cost 一致优于 $L_1$ cost。这点其实有点反直觉，因为 $L_1$ 对 outlier 更 robust 通常。但在 high-dim embedding space 里，$L_2$ 的 gradient 更 smooth，CEM 的 Gaussian fitting 也更稳定。

### 2. Multistep Rollout Loss

训练时除了 1-step teacher forcing loss，还加 k-step rollout loss：让 predictor 用自己的 prediction 当 context 继续预测。

$$\mathcal{L}_k = \frac{1}{B} \sum_{b=1}^B L[P_\theta(\hat{z}_{t-w:t+k-1}^b, A_\theta(a_{t-w:t+k-1}^b)), E_{\phi,\theta}(o_{t+k}^b)]$$

- $\hat{z}_{t+k-1}$ 是之前 prediction 递归 unroll 出来的 embedding
- $k=1$ 退化为 teacher forcing
- 用 TBPTT，detach gradient，只 backprop 最后一步

为啥要这个？因为 test time 是 planning，predictor 要在自己 prediction 的基础上继续 unroll。如果训练只用 teacher forcing（context 永远是 ground truth embedding），test time 拿 prediction 当 context 就 OOD 了。

结果：
- 1-step → 2-step：consistent 提升
- 3-step 之后：simulation 上反而下降，DROID 上 6-step 最优

paper 的解释：planning context $W^p = 2$，rollout step 太长会让 model 偏离 test-time distribution。但 DROID 上 dynamics 复杂，需要更长 rollout 才能 capture real-world arm + object 的 long-range dependency。

这暗示一个 trade-off：rollout step 要匹配 planning 时实际用的 context + horizon，但又要足够长来 capture 真实 dynamics 的 temporal dependency。Simulation 简单 dynamics → 短 rollout 够；real world 复杂 dynamics → 长更好。

paper 还比较了两种 rollout strategy（Figure S2.1）：
- "Last-gradient only"：每步只在最后 prediction 上算 loss，context 是 ground truth prefix + latest prediction
- "All-gradients"：所有中间步都算 loss

结论：Last-gradient only + random initial context 最好。All-gradients 虽然有更多 gradient signal，但没帮助。关键是要让 predictor 训练时看到的 input distribution 和 test time 一致。

### 3. Proprioception

加入 proprioception（关节位置、速度等低维 state 信息）一致提升 performance，特别是 Metaworld。原因：

Metaworld 大部分失败 episode 是 arm 到达 goal 后震荡，无法稳定停下。Proprioception 给了 precise "我现在离 goal 多远" 的信息，避免这种震荡。

2D navigation 上 proprioception 也帮助：agent 能 propose 更精确的 plan。

但 DROID → Robocasa zero-shot transfer 时不能用 proprio，因为两个 robot embodiment 的 proprio space 不对齐。所以 DROID/Robocasa 的 best config 反而是 no-prop。

直觉：proprioception 是 cheap signal（low-dim、dense），对 short-horizon precision 有用。但 cross-embodiment transfer 时反而是负担。这暗示未来可以做 embodiment-invariant proprio representation（类似 RT-X 的思路: https://arxiv.org/abs/2310.08864）。

### 4. Context Size $W$

predictor 训练时看多长的历史？

| $W$ | 含义 | 结果 |
|-----|------|------|
| 1 | 单帧，只能看 position | 差 |
| 2 | 两帧，能 infer velocity | 大提升 |
| 3 | 三帧，能 infer acceleration | sim 上最优 |
| 5 | - | DROID 上最优 |
| 7 | - | 普遍下降 |

直觉很物理：predictor 要从 frame sequence 推 dynamics，velocity 需要两帧，acceleration 需要三帧。$W=1$ 时只能从单帧 spatial pattern 猜 dynamics，相当于无 motion 信息。

$W$ 太大的坏处：相同 compute budget 下，trajectory slice 长度 $W+1$，unique slice 数变少，gradient step 变少。所以 $W=7$ 在 DROID 上反而下降（DROID trajectory 本来就短，20-50 帧，$W=7$ 切完没几片）。

关键 constraint：$W^p \leq W$。Planning 时 context 不能超过训练时的，否则 predictor 见到没 train 过的 context length，prediction 急剧 degrade。这点其实很 basic，但 paper 强调是因为很多人 experimentally 没注意到。

### 5. Encoder Type

比了 4 种 frozen visual encoder，都是 ViT-L：
- DINOv2 (https://arxiv.org/abs/2304.07193)
- DINOv3 (https://arxiv.org/abs/2508.10104)
- V-JEPA (https://arxiv.org/abs/2404.08039)
- V-JEPA-2 (https://arxiv.org/abs/2506.07909)

结果：
- DINO 系列明显优于 V-JEPA 系列
- DINOv3 在 photorealistic 环境 (DROID, Robocasa) 优于 DINOv2
- DINOv2 在 Maze/Wall 这种 synthetic 环境反而更好

这个发现挺反直觉的。V-JEPA 系列是 video encoder，理论上应该更懂 temporal dynamics。但实际 planning task 上反而输给 image encoder。

paper 的假设：DINO 有更好的 fine-grained object segmentation，对 manipulation 和 navigation 中的 precise localization 关键。V-JEPA 的 masked prediction 目标更关注 global motion，patch token 可能 mix 了不同 spatial location 的信息。

我自己的猜测：DINO 的 contrastive + self-distillation 训练目标自然产生 object-centric dense representation，每个 patch token 对应一个 spatial location 的 semantic content。V-JEPA 训的是 "predict masked region"，可能更关注 spatial averaging后的 global pattern，对 precise localization 不利。

video encoder 还有个 trick：把每一帧 duplicate 成 2 帧，组成 mini-video，再独立编码。这避免了 frame-causal mask 的复杂性，又能利用 video encoder 的 temporal pattern。但即使有这个 trick，V-JEPA 还是输给 DINO。

### 6. Predictor Architecture（Action Conditioning）

predictor 是 ViT，action 怎么进 ViT？paper 比了 4 种：

**(a) Feature conditioning + sincos**（DINO-WM 默认）：
- action embedding $A_\theta(a) \in \mathbb{R}^{f_a}$ 和 visual feature $E_\theta(o) \in \mathbb{R}^D$ 在 embedding dim 上 concat
- Hidden dim $D \to D + f_a$
- 加 3D sincos positional embedding
- Action ratio: $\frac{f_a}{D + f_a}$，比较大

**(b) Sequence conditioning + RoPE**（V-JEPA-2-AC 默认）：
- action 编码为单独 token，和 visual tokens 在 sequence dim 上 concat
- Hidden dim 保持 $D$
- RoPE 加在每 block
- Action ratio: $\frac{1}{hw+1} = \frac{1}{257}$（$16 \times 16$ patch grid），非常小

**(c) Feature conditioning + RoPE**：混合

**(d) AdaLN + RoPE**（paper 发现平均最好）：
- action embedding 通过 Adaptive LayerNorm 调制每 block 的 LayerNorm scale/shift
- 类似 DiT (https://arxiv.org/abs/2212.09748) 的 class conditional 设计
- Action 信息每层都 re-inject

结果：AdaLN 平均最好，但 task-dependent。Metaworld 上 sincos + feature cond 反而最好。

paper 给的 intuition：
- Feature conditioning 只在 input 注入 action，靠 self-attention 传播，深层可能稀释
- Sequence conditioning action token 只占 1/257 attention，容易被 visual token 淹没
- AdaLN 每层强制 re-inject，action 信号一致传播

paper 还做了 equalized action ratio 的 ablation（Table S5.1 之后）：把 image 降到 128×128 让 feature cond 和 sequence cond 的 action ratio 接近。结果 task-dependent pattern 依然存在：sequence cond 在 manipulation (3D) 好，feature cond 在 navigation (2D) 好。

我的 hypothesis：feature cond 的 action-to-token 直接通路对 spatially simple task 好；sequence cond 的 attention-based routing 对 spatially complex reasoning 好。但 paper 也承认没找到确切 mechanism。

### 7. Model Scaling

ViT-S → ViT-B → ViT-L，predictor depth 3 → 12。

结果：
- Simulation 上 scaling 无效甚至有害
- DROID 上 scaling 明显有用（depth 12 最好，ViT-L 最好）

paper 的解释：bigger embedding space 让 planning cost landscape 更 flat，optimizer 更难区分 nearby states。Figure S5.9 显示 ViT-L 的 relative embedding distance 比 ViT-S 小 10x，验证了这个 hypothesis。

这个发现非常重要，因为 community 普遍信仰 "bigger is better"（LLM 的 scaling law 影响太广）。但 physical world model 有 task complexity 阈值，简单 dynamics 不需要 big model，big model 反而让 planning 优化更难。

直觉上，这是 representation capacity 和 optimization difficulty 的 trade-off。Big model 能 encode 更多信息，但 latent space 也更 "拥挤"，planner 难以 find good gradient direction。

## Final Best Config 和主结果

Simulation (Maze, Wall, Push-T, Metaworld)：
- DINOv2 ViT-S encoder
- ViT-S predictor, depth 6, AdaLN, RoPE
- proprioception, 2-step rollout, $W=3$
- CEM $L_2$ planner

Real-world (DROID, Robocasa)：
- DINOv3 ViT-L encoder
- ViT-L predictor, depth 12, AdaLN, RoPE
- no proprioception (for zero-shot transfer)
- CEM $L_2$

Table 1 结果：

| Model | Maze | Wall | Push-T | MW-R | MW-RW | Rc-R | Rc-Pl | DROID |
|-------|------|------|--------|------|-------|------|-------|-------|
| DINO-WM | 81.6 | 64.1 | 66.0 | 44.8 | 35.1 | 19.1 | 21.7 | 39.4 |
| V-JEPA-2-AC | - | - | - | - | - | 16.2 | 33.1 | 42.9 |
| **Ours** | **83.9** | **78.8** | **70.2** | **58.2** | **41.6** | **25.4** | 30.7 | **48.2** |

Ours 在大部分 task 上都 best，只在 Robocasa Place 上输给 V-JEPA-2-AC（30.7 vs 33.1）。

## 几个我觉得最 interesting 的发现

### 1. World model quality ≠ Planning performance

paper 反复强调：好的 predictor 不等于好的 plan。即使 predictor 能 faithful unroll 6 步，planning 仍可能失败。

paper 的 Spearman correlation 分析（Table S5.2-S5.5）显示：
- Vis Emb $L_1$ at $H=2$ 是最好的 success rate proxy，correlation ~0.85 在 Push-T 上
- Training loss (H=1) correlation 最低
- 在 Metaworld 这种 long-horizon task 上，$H>1$ metric 比 $H=1$ 明显更相关

但 correlation 也只有 0.4-0.9，远不是 1.0。原因：
- Planner 可能 sample OOD actions，破坏 prediction quality 与 plan quality 的关联
- Cost landscape 有 local minima，好 world model 救不了 bad planner
- Goal embedding 可能 unreachable，但 planner 不知道

这暗示 future work 应该 jointly optimize world model 和 planner，让 latent space 自然适合 planning optimization。Universal Planning Networks (Srinivas et al. 2018, https://arxiv.org/abs/1804.00677) 是早期尝试。

### 2. V-JEPA-2-AC 有 bug

paper reproduce V-JEPA-2-AC 时发现 official code 有 bug：2-step rollout loss 实际计算的是 $\|P_\phi(a_{1:T}, s_1, z_1) - z_T\|_1$，意思是 model 拿 ground truth $z_1$ 和 prediction $\hat{z}_2$ 作为 input，被训练去 output $\hat{z}_2$（identity function！）。

Fix bug 重训后，planning 性能反而更好。但 official checkpoint 的 visual decoding 看起来也 OK，能通过 counterfactual test。

这非常深刻地说明：**visual decoding quality 与 planning quality 完全解耦**。一个能生成漂亮 future frame 的 world model 可能在 planning 上完全 broken，因为 latent space 里的 dynamics 不对，但 decoder 能 "fix up" visual 缺陷。

这对 generative world model (Sora, Genie 2, Cosmos) 是一个警示：visually impressive 不代表 useful for planning。

### 3. Scaling law 不 universal

simulation 上 bigger model 无益甚至有害，real-world data 上 bigger model 明显有用。

这与 LLM scaling law 完全不同。Physical world model 有 task complexity threshold，简单 dynamics 不需要 big model，big model 让 planning optimization 更难。

我直觉上认为，这是因为 LLM 的 output space 是 discrete token，bigger model 让 distribution 更 sharp，sampling 更准。World model 的 output 是 continuous embedding，bigger model 让 latent space 更 "crowded"，distance metric 失去 resolution。

这可能是 latent space planning 这条路线的 fundamental limitation。

### 4. Action Conditioning 是非平凡的

action 怎么进 predictor 这件事，居然有 4 种完全不同的方案，且 best choice 是 task-dependent。

AdaLN 平均最好，但 Metaworld 上 feature conditioning 反而最好。Equalized action ratio 后 task-dependent pattern 依然存在，说明不只是 capacity 问题。

paper 没给出确切的 mechanism 解释，承认 "we cannot provide a precise explanation"。这种诚实在 paper 里很少见。

我的直觉：action 是控制信号，应该像 diffusion model 的 condition 一样 globally influence network。AdaLN 实现了这一点。但 specific task 的 cost landscape 几何可能让 local conditioning 也 work，task-dependent 就出来了。

### 5. Object Manipulation Hallucination

Figure S5.7 展示一个 Metaworld bin-picking 的失败：world model 的 visual decoding 显示 object 被 picked up，但 simulator 中 object 没动。

这意味着 model 学到 "arm 接近 object → object 跟着移动" 的虚假 correlation。本质上 contact physics 没被 world model 真正理解，只是从 visual co-occurrence 学了个 shortcut。

paper 建议需要 separate optimization 处理 gripper closure dimension。但我觉得这暴露了 JEPA-WM 的 fundamental limitation：latent space prediction 学到的是 visual correlation，不是 causal physics。

未来可能需要：
- 显式建模 contact（tactile sensor 输入）
- Object-centric representation（每个 object 单独 embedding）
- Structural inductive bias（rigid body physics prior）

## 一些 open questions 和 future direction

1. **Learnable encoder**：frozen encoder 是 trade-off，小 robot data 上 fine-tune encoder 末端几层可能有用
2. **Multi-view fusion**：DROID 有 left/right/wrist 3 个 camera，paper 只用 1 个。Multi-view 应该提升 3D understanding
3. **Action representation learning**：paper 用 linear action encoder 太简单。LAPA (https://arxiv.org/abs/2410.11758) 用 VQ-VAE 学 latent action from action-free video，能利用更多 data
4. **Uncertainty modeling**：MSE loss 没 uncertainty，OOD action 上 overconfident。Bayesian world model 可能更 robust
5. **Inverse dynamics**：paper 只学 forward dynamics。Inverse dynamics 可能帮 planning
6. **Hierarchical planning**：long-horizon 是 open challenge，需要 subgoal generation
7. **Differentiable planning**：jointly optimize world model 和 planner，让 latent space 适合 planning

## 与其他方向的关系

### vs. Generative World Models (Sora, Genie 2, Cosmos)

这些方法生成 photorealistic video，visually impressive。但 paper 的核心 insight 是：generative quality ≠ planning quality。Cosmos 漂亮的 visual 是 diffusion decoder 的功劳，underlying latent world model 可能并不 accurate for planning。

JEPA-WM 反其道而行：完全 skip pixel generation，专注 latent dynamics。Computationally efficient，但失去 visual interpretability。

### vs. VLA Models (RT-2, $\pi_0$)

VLA 直接 learn policy，没有 explicit world model。End-to-end，data hungry。

JEPA-WM 是 model-based planning，sample efficient（reward-free training），但 inference 慢（CEM 300 trajectory × 15 iteration）。

两者是 complementary：VLA 适合常见 task 的 fast reactive control，JEPA-WM 适合 novel task 的 deliberate planning。未来可能 combine：VLA 提供 prior action distribution 给 CEM 采样。

### vs. DreamerV3, TD-MPC2

这些是 model-based RL，需要 reward。JEPA-WM 是 reward-free，更适合 offline unlabeled data。

但 DreamerV3 在有 reward 时可能更强，因为 reward signal 帮助 shape representation。Paper 没直接比较，DINO-WM paper 之前比较过，reward-free setting 下 JEPA-WM 更好。

## 最后几点 take-away

如果只记几件事：

1. **CEM $L_2$ 是 robust default planner**。Gradient-based 只在 smooth cost landscape 上 work，navigation 这种 non-greedy task 上完全失败。

2. **Multistep rollout loss 重要**，但 sweet spot task-dependent。Simulation 用 2-step，real-world 用 6-step。匹配 test-time context + horizon 是关键。

3. **DINO > V-JEPA on planning**。Dense object-centric representation 比 temporal video representation 更 important for physical planning。

4. **AdaLN 是 strong default action conditioning**，但 task-dependent。Sequence conditioning 在 manipulation 上可能更好，feature conditioning 在 navigation 上可能更好。

5. **Simulation 不要盲目 scale**，real-world data 才需要 big model。Bigger embedding 让 planning optimization 更难。

6. **World model ≠ Planning performance**。Visual decoding quality 与 planning quality 解耦。Future work 应 jointly optimize world model 和 planner。

7. **Long-horizon 是 open challenge**。当前 JEPA-WM 只解决 short-horizon task，需要 hierarchical planning 或 subgoal generation。

8. **Proprioception 帮 short-horizon precision**，但 cross-embodiment transfer 时是负担。

希望这个 "人话版" 帮你 build intuition 了！这篇 paper 的工作很 systematic，把 JEPA-WM 这个方向的 design choice space 基本摸清了，剩下的是更 fundamental 的 limitation（contact physics、long-horizon、world model-planner coupling）需要突破。

主要 references：
- Paper itself: https://arxiv.org/abs/2506.07909 (V-JEPA-2 paper, 同作者群)
- DINO-WM: https://arxiv.org/abs/2411.04983
- LeCun JEPA position paper: https://openreview.net/forum?id=BZ5a1r-kVsf
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- DiT (AdaLN reference): https://arxiv.org/abs/2212.09748
- RoPE: https://arxiv.org/abs/2104.09864
- Code: https://github.com/facebookresearch/jepa-wms

---

# 深入解读 "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?"

## 1. Big Picture: 为什么要研究 JEPA-WM

这篇 paper 的核心 motivation 在于回答一个看似简单的问题：在 latent space planning 这条路线上，到底哪些 design choices 真正 work？作者们把 LeCun 提出的 JEPA 思想落地到 physical planning，构建了一个统一的 JEPA-WM family，然后系统性地 ablate 每一个组件。

JEPA 的核心 philosophy（参考 LeCun 2022 的 "A Path Towards Autonomous Machine Intelligence" 论文：https://openreview.net/forum?id=BZ5a1r-kVsf）：
- 不要在 pixel space 预测未来，要在 abstract representation space 预测
- 这样可以丢弃 irrelevant details（比如背景纹理、光照），只保留 task-relevant 信息
- 通过 prediction + representation learning 联合训练，让 representation 自然地变成 predictive useful 的形式

传统的 world model 比如 DreamerV3 (Hafner et al. 2024, https://arxiv.org/abs/2401.05077) 在 latent space 重构 pixel，TD-MPC2 (Hansen et al. 2024, https://arxiv.org/abs/2310.16828) 用 reward-augmented Q-learning。这些方法依赖 reward signal，限制了在 reward-free offline data 上的应用。而 JEPA-WM 完全 reward-free，只用 self-supervised prediction loss 训练，再用 planning 在 test time 解决 task。

paper 的 baseline 是 DINO-WM (Zhou et al. 2024, https://arxiv.org/abs/2411.04983) 和 V-JEPA-2-AC (Assran et al. 2025, https://arxiv.org/abs/2506.07909)，这篇工作把这两个方法统一到一个 framework 里，找出每个 component 的最优配置。

## 2. Formal Framework 详解

### 2.1 Encoder 结构

State encoder 由两部分组成：
$$E_{\phi,\theta} = (E_\phi^{vis}, E_\theta^{prop})$$

- $E_\phi^{vis}$：frozen visual encoder（DINOv2/DINOv3/V-JEPA/V-JEPA-2 的 ViT）
- $\phi$：frozen 参数，不更新
- $E_\theta^{prop}$：shallow proprioceptive encoder（线性层或小 MLP）
- $\theta$：可训练参数
- $A_\theta$：action encoder（线性层）

关键 design：visual encoder frozen，只训练 predictor 和 proprio/action encoder。这与 DINO-WM 和 V-JEPA-2-AC 一致，目的是利用大规模预训练 visual representation，避免在小规模 robot data 上 overfitting visual encoder。

### 2.2 Training Objective

主 loss（Eq 1）：
$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} L[P_\theta(E_{\phi,\theta}(o_{t-w:t}^b), A_\theta(a_{t-w:t}^b)), E_{\phi,\theta}(o_{t+1}^b)]$$

变量含义：
- $B$：batch size（Metaworld/Push-T 用 256，Maze/Wall/DROID 用 128）
- $o_{t-w:t} := (o_{t-w}, \ldots, o_t)$：长度为 $w+1$ 的 observation 窗口
- $o_t$：包含 visual frame 和可选 proprioceptive vector
- $a_{t-w:t}$：对应的 action 序列
- $w$：训练时的 context window size（默认 $W=3$）
- $L$：pairwise loss，对 visual 和 proprio modality 分别计算，paper 中选 MSE
- 上标 $b$：batch 内的 sample index

这个 loss 实质上是 teacher-forcing：predictor 拿到 ground truth 历史 embedding 和 action，预测下一步 embedding。

### 2.3 Multistep Rollout Loss

为了让 predictor 在 test time 能 faithful unroll，paper 引入 k-step rollout loss（Eq 5）：
$$\mathcal{L}_k = \frac{1}{B} \sum_{b=1}^{B} L[P_\theta(\hat{z}_{t-w:t+k-1}^b, A_\theta(a_{t-w:t+k-1}^b)), E_{\phi,\theta}(o_{t+k}^b)]$$

其中：
$$\hat{z}_{t+k-1}^b = F_{\phi,\theta}(o_t, a_{t-w:t+k-2})$$

- $\hat{z}_i$：第 $i$ 步的 predicted embedding（不是 ground truth）
- $k$：rollout 步数，$\mathcal{L}_1 = \mathcal{L}$（即 1-step = teacher forcing）
- 训练时用 truncated backpropagation through time (TBPTT)，只 backprop 最后一个 prediction 的 gradient，discard 中间 accumulated gradient

paper 比较 "Last-gradient only" 和 "All-gradients" 两种 variant：
- Last-gradient only：只对最后一步 prediction 算 loss，但 context 包含之前的 prediction
- All-gradients：对所有中间步都算 loss

实验结论：2-step "Last-gradient only" 最好，更多 step 在 simulation 上反而下降，因为 planning context $W^p=2$，过多 rollout 步数让 model 偏离 test-time task distribution。但 DROID 上 6-step 最优，因为真实世界 dynamics 更复杂。

### 2.4 Unrolling Function $F_{\phi,\theta}$

递归定义（Eq 3-4）：
$$F_{\phi,\theta}: (o_t, a_{t-w:t+k-1}) \mapsto \hat{z}_{t+k}$$
$$\hat{z}_{i+1} = P_\theta(\hat{z}_{i-w:i}, A_\theta(a_{i-w:i})), \quad i = t, \ldots, t+k-1, \quad z_t = E_{\phi,\theta}(o_t)$$

- $w$：planning 时的 context window（$W^p$，paper 固定 $W^p = 2$）
- 每一步 unroll 只取 predictor 输出的最后一个 timestep，concatenate 到 context 用于下一步
- sliding window 保持长度 $W^p$

### 2.5 Planning Objective

$$L_\alpha^p(o_t, a_{t:t+H-1}, o_g) = (L_{vis} + \alpha L_{prop})(G_{\phi,\theta}(o_t, a_{t:t+H-1}), E_{\phi,\theta}(o_g))$$

变量：
- $H$：planning horizon（Push-T/Maze/Wall: 6，Metaworld: 6，Robocasa/DROID: 3）
- $o_g$：goal observation
- $\alpha$：proprioception 权重（默认 0.1，DROID/Robocasa 上设 0 因为 proprio space 不对齐）
- $G_{\phi,\theta}$：可以是 $F_{\phi,\theta}$（只看最后一步），也可以是所有中间步的函数

optimization 在 $\mathbb{R}^{H \times A}$ 上进行，$A$ 是 action dimension（Metaworld: 20，Push-T/Maze/Wall: 10，DROID: 7）。

## 3. Architecture 详解

### 3.1 Predictor Conditioning 方式

paper 比较了四种 action conditioning：

**Feature conditioning with sincos**（DINO-WM 默认）：
- Action embedding $A_\theta(a) \in \mathbb{R}^{f_a}$ 与 visual feature $E_\theta(o) \in \mathbb{R}^D$ 在 embedding dimension 上 concatenate
- Hidden dim 从 $D$ 增加到 $D + f_a$
- 加 3D sincos positional embedding
- Action ratio: $\frac{f_a}{D + f_a}$

**Sequence conditioning with RoPE**（V-JEPA-2-AC 默认）：
- Action 编码为 separate token，与 visual tokens 在 sequence dimension 上 concatenate
- Hidden dim 保持 $D$
- RoPE (Su et al. 2024, https://arxiv.org/abs/2104.09864) 加在每一个 block
- Action ratio: $\frac{1}{hw+1} = \frac{1}{257}$（对 $16 \times 16$ patch grid）

**Feature conditioning with RoPE**：混合方案

**AdaLN with RoPE**（paper 发现平均最好）：
- Action embedding 通过 Adaptive LayerNorm 调制每个 transformer block 的 LayerNorm 的 scale 和 shift
- 类似 DiT (Peebles & Xie 2023, https://arxiv.org/abs/2212.09748) 的设计
- Action 信息影响所有 layer，避免在深层消失
- 计算上更 efficient，不增加 feature 或 sequence dim

直觉：AdaLN 让 action 信号在每个 block 都"重新注入"，类似 conditional diffusion model 中的 class conditional。Feature conditioning 只在 input 注入，靠 self-attention 传播；Sequence conditioning 通过 attention 传播。AdaLN 提供了最一致的 conditioning pathway。

### 3.2 Frame-Causal Attention Mask

Predictor 用 frame-causal attention mask，这意味着 timestep $t$ 只能看到 $t' \leq t$ 的 frames。这有两个好处：
1. 训练时可以 parallel 预测所有 timesteps 的下一帧（一个 forward pass 搞定）
2. 自然支持任意 context length $w \in [0, W-1]$，因为 mask 是从 full mask 中 sub-sample 出来的

### 3.3 Encoder 比较

paper 比较了 4 种 encoder（都是 ViT-L）：

| Encoder | Type | Patch size | Resolution | Pos Embed | Frame Duplication |
|---------|------|-----------|-----------|-----------|-------------------|
| DINOv2 | Image | 14 | 224 | sincos | No |
| DINOv3 | Image | 16 | 256 | RoPE | No |
| V-JEPA | Video | 16 | 256 | sincos | Yes (each frame duplicated) |
| V-JEPA-2 | Video | 16 | 256 | RoPE | Yes |

关键 trick：对 video encoder，把每一帧 duplicate 成 2 帧组成 mini-video，然后独立编码每一对。这避免了 frame-causal mask 在 encoder 中的复杂性，同时利用 video encoder 学到的 temporal pattern。

DINO 系列明显优于 V-JEPA 系列，paper 假设是因为 DINO 有更好的 fine-grained object segmentation，对 manipulation 和 navigation 的精确位置感知更重要。DINOv3 在 photorealistic 环境（DROID, Robocasa）上明显优于 DINOv2，因为预训练数据更接近 photorealistic images。

## 4. Design Choices 系统性 Ablation

### 4.1 Planner 选择

四种 planner：
- **CEM** (Cross-Entropy Method, Algorithm 1)：sampling-based，迭代拟合 diagonal Gaussian
- **NG** (Nevergrad, Algorithm 2)：用 NGOpt meta-optimizer，自动选择 diagonal CMA-ES
- **GD** (Gradient Descent)：直接 backprop action
- **Adam**：用 Adam 优化 action sequence

实验结果（Figure 3）：
- CEM $L_2$ 整体最好
- GD/Adam 在 Metaworld 上很强（cost landscape smooth），但在 2D navigation 上灾难性失败（local minima）
- NG 与 CEM 在 DROID/Robocasa 持平，但 hyperparameter 更少

paper 中一个有意思的观察（Figure S4.1）：NG 收敛更慢，意味着更多 exploration，对 Push-T 这种需要精确 action 的任务不利，但对 multimodal cost landscape 友好。

GD 失败案例分析（Figure S5.1）：
1. 2D Wall 任务中，agent 卡在墙边无法穿过门
2. 起始位置接近图像边界时，agent 走到边界找到 local cost minimum

直觉：sampling-based 方法能 escape local minima，gradient-based 在 non-convex cost landscape 上脆弱。Metaworld 是 greedy reachable 的 task（goal 可以直接 reach），所以 gradient 有效；2D navigation 需要 non-greedy planning（绕墙），gradient 完全失效。

### 4.2 Multistep Rollout

从 1-step 到 6-step：
- Simulation 环境：2-step 最优，3-step 开始下降
- DROID：6-step 最优

paper 解释：planning context $W^p=2$，训练 rollout 步数远超 $W^p$ 会让 model 偏离 test-time task。但 DROID 的 dynamics 复杂，需要更长 rollout 才能 capture 真实 robot arm 和 object 的 dynamics。

这里有个深层 insight：JEPA-WM 训练时和 planning 时是两个不同 distribution 的 prediction task。训练用 ground truth prefix，planning 用 predicted prefix（除了第一帧）。Multistep rollout loss 强制让 model 在 "predicted prefix" 上也 work，弥合 train-test gap。

### 4.3 Proprioception

加入 proprioception（关节位置、速度等）一致提升 performance。Metaworld 上提升尤其明显，因为：
- 大部分失败 episode 是 arm 到达 goal 后震荡
- proprioception 提供 precise distance to goal 信息

但 DROID 到 Robocasa 的 zero-shot transfer 不能用 proprioception，因为 proprio space 不对齐（不同 robot embodiment）。

### 4.4 Context Size $W$

| $W$ | Metaworld | Push-T | DROID |
|-----|-----------|--------|-------|
| 1 | low | low | low |
| 2 | high | high | mid |
| 3 | optimal | optimal | high |
| 5 | - | - | optimal |
| 7 | drops | drops | drops |

直觉：
- $W=1$：predictor 看不到 velocity，只能从单帧推断 dynamics，差
- $W=2$：可以推断 velocity（两帧差分），大幅提升
- $W=3$：可以推断 acceleration，对 simulation 足够
- $W=5$：DROID 上最优，真实 dynamics 需要更长 temporal context
- $W=7$：性能下降，因为相同 compute budget 下 unique trajectories 数量减少（slice 长度 $W+1$），gradient steps 减少

关键 constraint：$W^p \leq W$，否则 predictor 在 test time 看到 train 时没见过的 context length，prediction 急剧 degrade。

### 4.5 Predictor Depth

- Depth 3：对简单 2D navigation (Wall, Maze) 足够
- Depth 6：大多数 simulation 最优
- Depth 12：DROID 上最优

simulation 在低 capacity 就 saturate，但 real-world data 持续受益于更深 predictor。

### 4.6 Model Scaling

- ViT-S → ViT-B → ViT-L：simulation 上无提升，DROID 上明显提升
- 大 model 在 simulation 上可能反而有害：embedding space 更大，planning 时相邻 states 距离更小（Figure S5.9），更难区分

这个发现有点反直觉：通常我们假设 bigger is better，但 JEPA-WM 在 simulation 上有 sweet spot。Paper 解释：planning 是在 embedding space 中 optimize，bigger embedding 让 cost landscape 更 flat，optimizer 更难找到 good direction。

## 5. 最终最佳配置

### Simulation 环境
- Encoder: DINOv2 ViT-S
- Predictor: ViT-S, depth 6, AdaLN conditioning, RoPE
- Training: proprioception, 2-step rollout, $W=3$
- Planner: CEM $L_2$

### DROID/Robocasa
- Encoder: DINOv3 ViT-L
- Predictor: ViT-L, depth 12, AdaLN, RoPE
- Training: no proprioception（用于 zero-shot transfer）
- Planner: CEM $L_2$

主结果（Table 1）：

| Model | Maze | Wall | Push-T | MW-R | MW-RW | Rc-R | Rc-Pl | DROID |
|-------|------|------|--------|------|-------|------|-------|-------|
| DINO-WM | 81.6 | 64.1 | 66.0 | 44.8 | 35.1 | 19.1 | 21.7 | 39.4 |
| V-JEPA-2-AC | - | - | - | - | - | 16.2 | 33.1 | 42.9 |
| Ours | **83.9** | **78.8** | **70.2** | **58.2** | **41.6** | **25.4** | **30.7** | **48.2** |

## 6. Evaluation Metrics 的相关性分析

paper 做了一个非常细致的分析：哪些 validation metric 与 success rate 相关性最强（Table S5.2-S5.5）？

测试的 metric：
- **Vis Emb $L_1$/$L_2$**：embedding space prediction error at horizon $H \in \{1,2,3\}$
- **Proprio dec**：proprioceptive decoding error
- **Visual decoding LPIPS**：decode predicted embedding 回 pixel，与 ground truth future frame 比 LPIPS

Spearman correlation 结论：
- **Vis Emb $L_1$ at $H=2$** 与 success rate 相关性最强
- Proprioceptive decoding 相关性中等
- Training loss (Vis Emb $H=1$) 相关性最低

为什么 success rate 与这些 metric 不完全 align？
1. Validation prediction task 是 supervised regression，planning 是 heuristic optimization，两者 task 不同
2. Planner 可能 sample OOD actions，破坏 prediction quality 与 plan quality 的关联（Figure S5.10）
3. Cost landscape 可能有 local minima，好的 world model 也救不了 bad planner

paper 还发现：在 Metaworld 这种需要 long-horizon unrolling 的任务上，$H>1$ 的 metric 比 $H=1$ 更相关；Wall 这种 short-horizon 任务上，$H=1$ 已经足够 informative。

## 7. 失败案例和 Limitations

### 7.1 Object Manipulation Hallucination

Figure S5.7 展示了一个典型失败：Metaworld bin-picking task。world model 的 visual decoding 显示 object 被 picked up，但 simulator 中 object 没动。这意味着 model 学到了"arm 接近 object → object 跟着移动"的虚假 correlation，没真正理解 contact physics。

paper 建议需要 separate optimization procedure 处理 gripper closure dimension。

### 7.2 Camera/Action Calibration Shift

DROID → Robocasa zero-shot transfer 时，model 预测的 state 系统性向左偏移（Figure S5.3）。这是 camera viewpoint 微小差异导致的，说明 JEPA-WM 对 visual distribution shift 敏感。

### 7.3 Long-Horizon Tasks

Robocasa 标准 pick-and-place 太 long-horizon，paper 不得不定义 easier 版本（起始位置更近）。这暴露了 JEPA-WM + MPC 在 long-horizon task 上的局限。

### 7.4 Action Error on DROID

DROID 上 action error 主要来自 gripper closure 和 orientation dim，而不是 end-effector position（Figure S5.10）。这意味着 model 对 gripper state 的 prediction 不准，但论文只用前 3 维 action error 作为 metric，回避了这个问题。

### 7.5 V-JEPA-2-AC Reproduction Bug

paper 在 reproduction 时发现 V-JEPA-2-AC official code 有 bug：2-step rollout loss 实际计算的是 $\|P_\phi(a_{1:T}, s_1, z_1) - z_T\|_1$，意思是 model 拿到 ground truth $z_1$ 和 prediction $\hat{z}_2$ 作为输入，被训练去输出 $\hat{z}_2$（identity function！）。Fix bug 后重新训练，planning 性能反而比 official checkpoint 更好，但 official checkpoint 的 visual decoding 居然也看起来不错。这说明 visual decoding quality 与 planning quality 解耦，验证了 paper 的核心 thesis：好的 world model ≠ 好的 planning。

## 8. 与 Concurrent 工作的关系

### 8.1 vs. DINO-WM

DINO-WM 是 JEPA-WM family 的开山之作，证明了 frozen DINOv2 + latent dynamics predictor 可以 zero-shot planning。但这篇 paper 发现 DINO-WM 的多个 suboptimal choices：
- Feature conditioning with sincos：不如 AdaLN with RoPE
- 1-step loss：不如 2-step rollout loss
- 没有 system ablation，不知道每个 component 的贡献

### 8.2 vs. V-JEPA-2-AC

V-JEPA-2-AC 用 video encoder (V-JEPA-2) 替代 image encoder，声称能 capture temporal dynamics。但 paper 发现：
- V-JEPA 系列 encoder 反而比 DINO 系列差
- 因为 V-JEPA 训练目标是 masked prediction，更关注 global motion，而 DINO 的 dense feature 更适合 precise object localization
- V-JEPA-2-AC 的 2-step loss 实现有 bug

### 8.3 vs. DreamerV3, TD-MPC2

这些方法依赖 reward，JEPA-WM reward-free。Paper 没直接比较（因为 setting 不同），但 DINO-WM paper 已经证明在 reward-free setting 下 JEPA-WM > DreamerV3/TD-MPC2。

### 8.4 vs. Generative World Models

Genie 2 (Parker-Holder et al. 2024, https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)、Sora (Brooks et al. 2024, https://openai.com/research/video-generation-models-as-world-simulators)、Cosmos (Agarwal et al. 2025, https://arxiv.org/abs/2501.03575) 这些生成式 world model 生成 photorealistic video，但 paper 指出：generative model 的 visual quality 不代表 planning quality。Cosmos 的漂亮 visual 是 diffusion decoder 的功劳，underlying latent world model 可能并不 accurate。

### 8.5 vs. VLA Models

RT-2 (Zitkovich et al. 2023, https://arxiv.org/abs/2307.15818)、$\pi_0$ (Black et al. 2024, https://arxiv.org/abs/2410.24164) 等 VLA model 直接 learn policy，没有 explicit world model。V-JEPA-2-AC paper 已经证明 JEPA-WM planning 在 greedy manipulation 上能 beat VLA baselines like Octo (https://arxiv.org/abs/2405.12213)。这篇 paper 进一步强化了这个结论。

## 9. 个人 Intuition 和 Open Questions

### 9.1 为什么 DINO > V-JEPA on planning？

我猜测根本原因是 DINO 的 contrastive + self-distillation 目标自然产生 object-centric representation，每个 patch token 对应一个 spatial location 的 semantic content。这种 dense spatial representation 对 navigation 和 manipulation 是必需的。

V-JEPA 的 masked prediction 目标更关注 global motion pattern，patch token 可能 mix 了不同 spatial location 的信息，不利于 precise localization。

未来方向可能是：用 DINOv3 的 dense feature + V-JEPA-2 的 temporal understanding，组合一个 video encoder 既保留 spatial precision 又有 temporal reasoning。

### 9.2 Multistep Rollout 的 sweet spot

为什么 2-step 在 simulation 上最优，6-step 在 DROID 上最优？

我的 hypothesis：simulation 的 dynamics 简单，short rollout 已经能 capture，长 rollout 让 model overfit 到 specific trajectory pattern。真实世界 dynamics 有 long-range temporal dependency（比如 arm 的 momentum、object 的 friction），需要 longer rollout 才能 capture。

但更深层：planning context $W^p$ 决定了 test-time prediction 的 dependency 长度，training rollout step 应该匹配 $W^p$ 加上 planning horizon $H$。Simulation $H=6$, $W^p=2$，所以 2-step rollout 大致覆盖；DROID $H=3$, $W^p=2$，按这个 logic 应该 2-3 step 就够，但实验显示 6-step 最好，说明这个 hypothesis 不完全对。

可能是 DROID 的 trajectory 本身就有 long-range dependency（一个 action 影响后续多帧），需要 longer rollout 才能 capture action 的 delayed effect。

### 9.3 Action Conditioning 的本质

AdaLN 之所以好，我直觉是因为 action 在 JEPA-WM 中是"控制信号"，应该像 diffusion model 中的 condition 一样影响整个 network，而不仅是 input token。

Feature conditioning 的问题：action 只在 input 层与 visual feature concat，靠后续 self-attention 传播。但 predictor depth 不深时，action 信号可能"被稀释"。

Sequence conditioning 的问题：action token 只占 sequence 的 1/257，attention weight 分配给 action token 的比例很小，action 信号容易被 visual token 淹没。

AdaLN 像"全局广播"，每个 block 都强制 re-inject action 信息，避免信号衰减。

### 9.4 Planning 与 World Model 的解耦

paper 最深刻的 insight 是：好的 world model 不等于好的 planning。即使 predictor 能 faithful unroll 6 步，planning 仍可能失败，因为：
1. Planner 在 cost landscape 中 stuck
2. Action distribution OOD for predictor
3. Goal embedding 与 reachable states 距离 mismatch

这暗示未来工作应该 jointly optimize world model 和 planner，比如用 differentiable planner 训练 world model，让 latent space 自然适合 planning。Srinivas et al. 2018 的 Universal Planning Networks (https://arxiv.org/abs/1804.00677) 是这个方向的早期尝试。

### 9.5 Long-Horizon Planning 的根本难题

paper 不得不简化 Robocasa task，因为 long-horizon planning 在 JEPA-WM 框架下很难。原因：
1. Error accumulation：每步 prediction 误差累积，$H$ 大时最终 embedding 偏离 ground truth 太远
2. Cost landscape 多 modality：long-horizon action space 是 $\mathbb{R}^{H \times A}$，维度高，optimizer 难以 explore
3. Subgoal generation：paper 用 ground truth goal frame，但 realistic setting 需要自动 generate intermediate subgoals

未来可能的方向：
- Hierarchical planning：high-level planner 生成 subgoal，low-level planner reach subgoal
- Latent space trajectory optimization：在 latent space 而非 action space 优化
- Diffusion-based planner (Janner et al. 2022, https://arxiv.org/abs/2205.09991; Zhou et al. 2024, https://arxiv.org/abs/2410.05364)：用 diffusion model 直接生成 trajectory distribution，escape local minima

### 9.6 与 LLM-based Reasoning 的结合

当前 JEPA-WM 完全 bottom-up，从 visual prediction 学 dynamics。但人类 planning 时会用 top-down reasoning（"我要先把 cup 拿起来，然后放到桌子上"）。

未来可以 combine：VLM 做 high-level reasoning 生成 subgoal sequence，JEPA-WM 做 low-level planning reach 每个 subgoal。这类似于 ReAct (Yao et al. 2022) 在 physical agent 上的 extension。

### 9.7 评测 Benchmark 的局限

paper 用 success rate 作为主 metric，但 success rate 高度 noisy、sparse。paper 的 Spearman correlation 分析显示，Vis Emb $L_1$ at $H=2$ 是最好的 proxy，但 correlation 也只有 0.4-0.9。

更好的 metric 可能是：
- Trajectory quality（与 expert trajectory 的 DTW distance）
- Sample efficiency（达到 success 所需 planning iterations）
- Robustness（对 perturbation 的稳定性）

### 9.8 Real-World Transfer

paper 在真实 Franka arm 上测试了 16 个 video（DROID evaluation），但只是 open-loop action error，没有 closed-loop control。真正 deploy 到 real robot 需要：
- Real-time planning（CEM 300 trajectories × 15 iterations 太慢）
- Robustness to observation noise
- Safety constraint

V-JEPA-2-AC paper 报告了 real robot closed-loop 实验，这篇 paper 没做，是明显 limitation。

## 10. 一些可能的 Follow-up 方向

### 10.1 Learnable Encoder

paper 假设 encoder frozen，但可能在小规模 robot data 上 fine-tune encoder 末端几层会有帮助。DINOv3 的 register token (Darcet et al. 2024, https://arxiv.org/abs/2309.16588) 处理了 ViT 的 artifact token，但 robot data 的 distribution 可能与 ImageNet 不同。

### 10.2 Action Representation Learning

paper 用 linear action encoder，简单。LAPA (Ye et al. 2024, https://arxiv.org/abs/2410.11758) 用 VQ-VAE 学 discrete latent action from video without action label，可能能利用 action-free video data pretrain action representation。

### 10.3 Multi-View Fusion

DROID 有 3 个 camera (left, right, wrist)，paper 只用 one view。Multi-view fusion 可能提升 3D understanding，特别是对 occlusion 和 depth 的感知。

### 10.4 Active Learning

paper 用 random trajectory training，但 active learning 选 informative trajectory（high prediction error 的）可能 sample efficient。

### 10.5 World Model Uncertainty

paper 用 MSE loss，没建模 uncertainty。在 OOD action 上 predictor 可能 overconfident。Bayesian world model（ensemble 或 MC dropout）可能更 robust，也能用于 exploration。

### 10.6 Inverse Dynamics Model

paper 只学 forward dynamics，但 inverse dynamics（从 state transition 推 action）可能帮助 planning。Latent action model (LAPA) 是这个方向。

## 11. 公式汇总与 Intuition

让我再总结一下关键公式的 intuition：

**Eq 1 (Training Loss)**：
$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} L[P_\theta(E_{\phi,\theta}(o_{t-w:t}^b), A_\theta(a_{t-w:t}^b)), E_{\phi,\theta}(o_{t+1}^b)]$$

Intuition：给定历史 observation embedding 和 action embedding，预测下一帧 embedding。这是 self-supervised prediction task，没有 reward，没有 reconstruction，完全在 latent space 操作。

**Eq 2 (Planning Objective)**：
$$L_\alpha^p(o_t, a_{t:t+H-1}, o_g) = (L_{vis} + \alpha L_{prop})(G_{\phi,\theta}(o_t, a_{t:t:H-1}), E_{\phi,\theta}(o_g))$$

Intuition：在 action space 优化，找到让 unrolled final embedding 接近 goal embedding 的 action sequence。$\alpha$ 控制 proprioception 权重，平衡 visual 和 proprioceptive modality。

**Eq 5 (Multistep Rollout Loss)**：
$$\mathcal{L}_k = \frac{1}{B} \sum_{b=1}^{B} L[P_\theta(\hat{z}_{t-w:t+k-1}^b, A_\theta(a_{t-w:t+k-1}^b)), E_{\phi,\theta}(o_{t+k}^b)]$$

Intuition：让 predictor 在自己之前的 prediction 上继续 prediction，模拟 test-time unrolling。$k$ 越大，越接近 planning task，但 error 也累积越多，找到 sweet spot 是关键。

**Eq 3-4 (Unrolling)**：
$$\hat{z}_{i+1} = P_\theta(\hat{z}_{i-w:i}, A_\theta(a_{i-w:i}))$$

Intuition：递归调用 predictor，每次只在 context 末尾添加新 prediction，保持 sliding window $w$。这是 planning 时的 inference 过程。

## 12. 实验数据表的关键 Reading

### 12.1 Table 1 主结果

Ours 在所有 8 个 task 上都最好或并列最好。最显著的提升在 Wall（+14.7%）和 MW-R（+13.4%）。Wall 提升来自 AdaLN + 2-step rollout；MW-R 提升来自 proprioception + 2-step rollout。

### 12.2 Table S5.1 全 Planner Comparison

这张表展示了 Ours 在所有 6 种 planner 配置下的表现。值得注意的是：
- Ours 在 CEM $L_1$ 上也明显优于 baseline，说明提升不只是 $L_2$-specific
- Gradient-based planner (Adam/GD) 在 DROID 上完全失败（0%），即使 Ours 也不行，说明 cost landscape 极其 non-convex
- Ours 在 NG planner 上也提升，说明提升不是 CEM-specific

### 12.3 Table S5.2-S5.5 Spearman Correlation

最 informative 的发现：
- Vis Emb $L_1$ at $H=2$ 是 universal 最好的 success rate proxy
- Metaworld (long-horizon) 上 $H>1$ metric 明显优于 $H=1$，Wall (short-horizon) 上差别小
- Proprioceptive decoding correlation 在 0.2-0.8 之间，不如 visual embedding metric

### 12.4 Table S2.3 Training Time

训练成本：
- 1-step on Metaworld: 23 min/epoch on 32 H100
- 6-step on Metaworld: 30 min/epoch (+30%)
- W=7 on DROID: 13 min/10 epoch vs W=3 的 7 min/10 epoch
- 总训练时间：Metaworld 50 epoch ≈ 19 hours，DROID 315 epoch ≈ 36 hours

## 13. Paper 的 Methodological 贡献

除了具体 design choice findings，paper 在 methodology 上的贡献：

1. **统一 framework**：把 DINO-WM 和 V-JEPA-2-AC 统一到一个 formalism，便于 fair comparison
2. **Systematic ablation**：每个 component 独立 vary，isolate effect
3. **Multiple metric tracking**：不只看 success rate，看 embedding error, proprio decoding, LPIPS, action error，提供更全面 model 评估
4. **Spearman correlation analysis**：定量分析 metric 与 success rate 的相关性，对未来 work 选 metric 有指导意义
5. **Bug 发现**：reproduce V-JEPA-2-AC 时发现 official code bug，说明 reproducibility 重要

## 14. Code 和 Resource

paper 代码、data、checkpoint 在 https://github.com/facebookresearch/jepa-wms 开源。

这个 codebase 可能有用的 component：
- Unified JEPA-WM training pipeline
- 4 种 planner implementation (CEM, NG, GD, Adam)
- 6 个 environment 的 evaluation harness (Metaworld, Push-T, Maze, Wall, DROID, Robocasa)
- Visual decoder training（用于 qualitative inspection）
- State decoder training（用于 proprioceptive metric）

## 15. 总结：Build Intuition

读完这篇 paper，我对 JEPA-WM 的 intuition：

1. **Latent space prediction 是 physical planning 的 viable path**：不生成 pixel，直接在 pretrained visual encoder 的 latent space 预测 dynamics，computationally efficient 且 semantic meaningful。

2. **Frozen visual encoder 是 double-edged sword**：利用大规模 pretrain 知识，但限制了对 task-specific visual feature 的 adaptivity。DINOv3 > DINOv2 > V-JEPA-2，证明 image encoder 比 video encoder 更适合 planning task。

3. **Predictor 是核心**：决定 JEPA-WM 质量。AdaLN + RoPE + depth 6 (sim) / 12 (real) 是当前最优配置。Action conditioning 方式直接决定 action 信号在 predictor 中的传播。

4. **Multistep rollout 是关键 trick**：让 predictor 适应 test-time unrolling distribution。但 sweet spot task-dependent，需要 careful tuning。

5. **Planner 选 CEM $L_2$**：sampling-based 在 navigation 上 robust，gradient-based 只在 smooth cost landscape 上 work。L2 metric 通常优于 L1。

6. **World model ≠ Planning performance**：好的 prediction 不等于好的 plan。Cost landscape 的 non-convexity、action OOD、goal reachability 都是 bottleneck。Future work 应该 jointly optimize world model 和 planner。

7. **Scaling law 不 universal**：simulation 上 model scaling 无效甚至有害，real-world data 上有效。这与 LLM 的 scaling law 不同，physical world model 有 task complexity 阈值。

8. **Long-horizon 是 open challenge**：当前 JEPA-WM 只能解决 short-horizon task，long-horizon 需要 hierarchical planning 或 subgoal generation。

希望这些 intuition 对你（Karpathy）思考 next-gen world model 有帮助！这篇 paper 的工作很 solid，systematic ablation 让我们对 JEPA-WM 的每个 component 有了 quantitative understanding，而不只是"it works"。

进一步阅读建议：
- DINO-WM: https://arxiv.org/abs/2411.04983
- V-JEPA-2: https://arxiv.org/abs/2506.07909
- LeCun JEPA original: https://openreview.net/forum?id=BZ5a1r-kVsf
- DINOv3: https://arxiv.org/abs/2508.10104
- DiT (AdaLN reference): https://arxiv.org/abs/2212.09748
- RoPE: https://arxiv.org/abs/2104.09864
- CEM tutorial: https://arxiv.org/abs/2309.13799
- Nevergrad: https://arxiv.org/abs/2104.09565
