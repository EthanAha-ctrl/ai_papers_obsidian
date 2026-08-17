---
source_pdf: WHAT DRIVES SUCCESS IN PHYSICAL PLANNING WITH JOINT-EMBEDDING PREDICTIVE
  WORLD MODELS.pdf
paper_sha256: dea31537d59eaa9bd72a3e2588eb37b7d3bde77a443ad95fa9b3921cd91c4f59
processed_at: '2026-08-13T04:02:12-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 Paper

## 先讲清楚一件事：这篇 paper 在干嘛

这篇 paper 没有发明新算法。它做的是一件很朴素的事：**问"为什么 work"**。

现在有一类方法（DINO-WM、V-JEPA-2-AC、PLDM）都在做同一件事——拿一个 frozen 的 visual encoder（比如 DINOv2），在它的 feature space 上训一个小 predictor 预测未来，然后用这个 predictor 做 planning。这家人都叫 **JEPA-WM**。

但这家人内部有很多技术选择没讲清楚：
- predictor 怎么吃 action？
- 要不要 proprioception？
- 训练时 unroll 几步？
- 用什么 planner？
- encoder 该用 image 的还是 video 的？

这篇 paper 就是把这些"没讲清楚"的东西一个个 ablation，找出最优组合，最后拼出一个比两个 baseline 都强的模型。

代码在 [github.com/facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms)。

---

## JEPA-WM 的直觉

想象你在开车。你脑子里有个"世界模型"——你打方向盘，你能预判车子会往哪走。你不需要在脑子里渲染出完整的高清画面，你只需要预测"车会到那个位置"这种抽象信息。

JEPA-WM 就是这个思路：
1. 拿一个已经预训练好的 visual encoder（DINOv2），它把图像变成一堆 feature vector。这个 encoder **冻住不动**。
2. 在这堆 feature 上面训一个小 ViT（叫 predictor），输入"当前 feature + action"，输出"下一步 feature"。
3. Planning 时：给一个 goal 图像，encoder 把它变成 goal feature。然后 predictor 从当前 feature 开始 unroll，用 planner 找一串 action 让预测的 final feature 靠近 goal feature。

公式上就是：
$$\hat{z}_{t+1} = P_\theta(E_\phi(o_t), A_\theta(a_t))$$

这里 $E_\phi$ 是 frozen encoder，$P_\theta$ 是 trainable predictor，$A_\theta$ 是 action encoder。

为什么这样做？因为：
- **像素级重建太贵且没必要**——你 planning 时不需要知道每个像素长啥样，只需要知道"物体到没到目标位置"
- **frozen encoder 让 representation learning 和 dynamics learning 解耦**——你不用同时学两个难题
- **feature space 更紧凑**——planning 在小空间里搜索比在像素空间里搜索容易得多

---

## 7 个 Design Choice，逐个讲

### 1. Planner：用什么优化器找 action？

Planning 的本质是：找一串 action $a_{t:t+H-1}$，让 predictor unroll 出来的 final feature 离 goal feature 最近。这是个优化问题。

测了 4 种 optimizer：
- **CEM**（Cross-Entropy Method）：采样一堆 action 序列，算 cost，保留最好的 top-K，用它们更新高斯分布的均值和方差，再采样。重复几轮。
- **NG**（NeverGrad）：自动选 optimizer 的 meta-optimizer，这里它选了 diagonal CMA-ES。
- **Adam / GD**：直接对 action 序列做梯度下降——因为 predictor 是可微的，可以把 cost 对 action 的梯度算出来。

**结论**：CEM + $L_2$ 距离总体最好。但有意思的是：

- **Gradient-based 在 Metaworld 上很强**——因为 Metaworld 的任务是"reach a goal position"，cost landscape 平滑，goal 可以贪心 reachable。梯度能直接指向 goal。
- **Gradient-based 在 2D navigation 上完全崩溃**——Wall 任务需要先穿过一道门才能到另一个房间。GD 会陷在"撞墙"的 local minima 里出不来。Figure S5.1 里有两个典型 failure case：要么撞墙，要么走到图像边缘卡住。
- **DROID 上 gradient-based 也烂**——real-world manipulation 的 cost landscape 是多模态的（抓物体的姿势有很多种），梯度会被困住。
- **NG 和 CEM 在 DROID/Robocasa 上持平**，但 NG 不需要调 hyperparameter。这对 transfer 到新数据集很实用。

直觉上：**采样-based 方法（CEM/NG）比梯度-based 方法更 robust**，因为它们不依赖 cost landscape 的平滑性。梯度方法只在"贪心可解"的任务上 work。

### 2. Multistep Rollout Training：训练时 unroll 几步？

最简单的训练是 **teacher forcing**：给 groundtruth 当前 state，预测下一步 state，用 groundtruth 下一步做监督。这就是 1-step loss。

但 planning 时你不会一直有 groundtruth——你会用 predictor 自己的输出作为下一步输入。这叫 **exposure bias**：训练时只见 groundtruth，测试时只能吃自己的预测。

解决办法：训练时也 unroll 几步，让 predictor 习惯吃自己的输出。加 k-step rollout loss：

$$\mathcal{L}_k = \frac{1}{B}\sum_b L\left[ P_\theta(\hat{z}_{t-w:t+k-1}^b, A_\theta(a_{t-w:t+k-1}^b)), E_{\phi,\theta}(o_{t+k}^b) \right]$$

这里 $\hat{z}$ 是 predictor 自己的预测，不是 groundtruth。

**关键实现细节**：用 **TBPTT（Truncated Backprop Through Time）**——每次 unroll 后 detach gradient，只 backprop 最后一步的 loss。这样省内存，而且效果反而更好。

测试了多种 rollout 策略（Figure S2.1）：
- "Last-gradient only"：只把最新预测拼到 context
- "All-gradients"：算所有中间 loss
- "Equal-order"：每步用完整前一步输出

**结论**：
- **2-step "Last-gradient only" 在模拟环境最优**
- **6-step 在 DROID 上最优**
- $k > 3$ 在模拟环境反而下降

为什么 2-step 最好？因为 planning 时用的 context 是 $W^p = 2$。你训练时 unroll 太长，模型会 over-specialize 到训练时的 unrolling pattern，跟 test-time 不一致。

为什么 DROID 要 6-step？因为 real-world dynamics 更复杂，更长的 unroll 让 predictor 学到更长期的物理约束。

**直觉**：训练 unroll 长度应该和 test-time planning 的 unroll 长度匹配。不是越长越好。

### 3. Proprioception：要不要机器人本体感觉？

Proprioception 是机器人的"本体感觉"——end-effector 的位置、速度、gripper 开合度等。

对比加 / 不加：
- **加 proprioception 一致更好**（Figure 4a）
- 在 Metaworld 上，不加 proprioception 的失败模式是：arm 到达 goal 后在 goal 周围震荡——因为视觉 feature 对"精确距离"不敏感
- 加了 proprioception 后，planner 能精确知道"我离 goal 还有多远"

**但有个例外**：DROID → Robocasa 是 zero-shot transfer。DROID 和 Robocasa 的 proprioception space 不对齐（不同的 gripper、不同的坐标系），所以这两个任务**不用 proprioception**（设 $\alpha = 0$）。

**直觉**：proprioception 在同 embodiment 训练-评估时有帮助（提供精确 metric 信息），但在 cross-embodiment transfer 时反而有害（coordinate 不对齐）。

### 4. Training Context Size $W$：predictor 看几帧历史？

predictor 输入不只是当前帧，而是一个 sliding window。

测试 $W = 1$ 到 $W = 7$：
- **$W=1 \to W=2$ 有大幅提升**——因为 predictor 需要 2 帧才能推断 velocity
- **3 帧可以推断 acceleration**——提升不大但有
- **模拟环境最优 $W = 3$**
- **DROID 最优 $W = 5$**——real-world dynamics 更复杂，需要更长 context

**硬约束**：planning 时的 context $W^p$ 必须 $\leq$ 训练时的 $W$。否则你让模型做训练时没见过的预测任务，预测会快速退化。

**直觉**：predictor 需要足够的历史推断速度和加速度，但太长的历史会让训练切片变少（每个 trajectory 切成更少的 $W+1$ 长度的 slice），gradient step 减少。

### 5. Encoder Type：用 image encoder 还是 video encoder？

测了 4 种 frozen encoder：
- DINOv2（image）
- DINOv3（image，更新版本，[Siméoni et al., 2025](https://arxiv.org/abs/2508.10104)）
- V-JEPA（video，[Bardes et al., 2024](https://arxiv.org/abs/2406.07679)）
- V-JEPA-2（video，[Assran et al., 2025](https://arxiv.org/abs/2506.07627)）

**关键发现**：**DINO 系列全面优于 V-JEPA 系列**（Figure 4b）。

为什么？DINO 有更强的 **fine-grained object segmentation**。在 manipulation 任务里，你需要精确知道"物体在哪、机械臂末端在哪"。DINO 的 feature 对这些 spatial detail 编码更好，V-JEPA 的 feature 更偏向 motion 和 temporal pattern。

DINOv3 在 photorealistic 环境（DROID, Robocasa）上显著优于 DINOv2，但在 Maze, Wall 这种简单 2D 环境反而更慢且最终 success rate 低——可能因为 DINOv3 的预训练数据更偏 real-world image。

**Encoding 技巧**：用 video encoder 时，最佳做法是**把每帧 duplicate 一份**组成 2-frame video，独立 encode 每一对。这样等价于 image encoder。直接用 video encoder 的时序依赖反而更差——而且要加 frame-causal mask 防 future 信息 leakage 到 past。

**直觉**：你不需要 encoder 理解时序，你需要 encoder 理解空间。时序由 predictor 来学。

### 6. Predictor Architecture：action 怎么注入？

这是最有意思的 ablation。问题是：action 信息怎么进入 predictor？

测了 4 种 conditioning 方式（Figure 5a）：

**a) Feature conditioning + sincos**（DINO-WM 用）：
- action embedding 沿 feature 维度 concat 到 visual feature
- hidden dim 从 $D$ 增到 $D + f_a$
- action 信息只在 input 层进入

**b) Sequence conditioning + RoPE**（V-JEPA-2 用）：
- action 作为一个独立 token，沿 sequence 维度 concat
- hidden dim 保持 $D$
- action token 从第一层就参与 self-attention

**c) Feature conditioning + RoPE**：混合

**d) AdaLN + RoPE**（[Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748) DiT 用的）：
- action embedding 通过一个 MLP 投影成 scale 和 shift
- 在每个 transformer block 的 LayerNorm 后调制 feature
- action 信息**在每层都注入**

**Action ratio 分析**：
- Feature conditioning：$\frac{f_a}{D+f_a}$，比较高
- Sequence conditioning：$\frac{1}{hw+1} = \frac{1}{257}$，很低

**结论**：**AdaLN + RoPE 平均最好**，但 task-dependent（Metaworld 上 sincos+feature 最好）。

为什么 AdaLN 好？直觉是：**action 信息在深层 transformer 中容易 vanishing**。Feature conditioning 只在 input 注入，经过多层 self-attention 后 action 信号可能被稀释。AdaLN 在每层都重新注入 action，保证 action 信号一直 strong。

这也和 DiT 在 diffusion model 里的发现一致——timestep conditioning 用 AdaLN 比用 cross-attention 好。

还测了 **AdaLN-zero**（初始化时 conditioning MLP 输出 0，让 predictor 开始时像 unconditional ViT）。DiT paper 里 AdaLN-zero 更好，但这里 AdaLN-zero 在 DROID, Push-T, Maze 这些信号最可靠的环境上反而不如 AdaLN。可能因为这里 task 不像 image generation 那么难，zero-init 的 regularization 没必要。

### 7. Model Scaling：encoder 和 predictor 多大？

测了 encoder ViT-S → ViT-B → ViT-L，predictor depth 3 → 12。

**关键发现**（Figure 6）：
- **模拟环境 scaling 不 work**——更大模型饱和甚至有害
- **DROID 上 scaling 显著 work**——encoder size 和 predictor depth 都正相关
- 模拟环境最优 predictor depth 通常 6，2D navigation 甚至 3 就够

为什么模拟环境 scaling 反而有害？Figure S5.9 给出线索：ViT-L 的 embedding space 中，两个 state 之间的相对距离比 ViT-S 小约 10 倍。这意味着 embedding space 更"挤"，planner 更难区分 nearby states。Planning 是在 embedding space 上做优化，space 太大反而让 optimization 更难。

**直觉**：capacity 要匹配 task complexity。模拟环境 dynamics 简单，小模型就够；real-world dynamics 复杂，需要大模型。盲目 scaling 不 work。

---

## 最终的 Recipe 和结果

模拟环境：
- DINOv2 ViT-S encoder
- ViT-S predictor, depth 6, AdaLN + RoPE
- 加 proprioception
- 2-step rollout loss
- $W = 3$
- CEM $L_2$ planner

DROID / Robocasa：
- DINOv3 ViT-L encoder
- ViT-L predictor, depth 12, AdaLN + RoPE
- 不加 proprioception
- 6-step rollout loss
- $W = 5$
- CEM $L_2$ planner

**结果**（Table 1）：

| Model | Maze | Wall | Push-T | MW-R | MW-RW | Rc-R | Rc-Pl | DROID |
|---|---|---|---|---|---|---|---|---|
| DINO-WM | 81.6 | 64.1 | 66.0 | 44.8 | 35.1 | 19.1 | 21.7 | 39.4 |
| V-JEPA-2-AC | - | - | - | - | - | 16.2 | 33.1 | 42.9 |
| **Ours** | **83.9** | **78.8** | **70.2** | **58.2** | **41.6** | **25.4** | **30.7** | **48.2** |

在所有 8 个环境上几乎全面胜出。

---

## 几个有意思的"反直觉"发现

### 1. 好的 world model ≠ 好的 planning

Section E.3 做了 Spearman correlation 分析，发现 success rate 和 validation prediction loss 的相关性**没那么强**（Metaworld 上最高才 0.47）。

V-JEPA-2-AC 的 public checkpoint 有个 bug（2-step rollout loss 算错了），但它的 image decoding 看起来正常，counterfactual test 也通过——只是 planning 性能差。这说明：**latent rollout 看起来对不等于 planning 对**。

直觉：planning 依赖 cost landscape 的几何结构，不只是 prediction accuracy。Planner 可能卡 local minima，或者 sample OOD action 让 predictor 输出垃圾。

### 2. Sim 和 Real 的 scaling behavior 完全相反

模拟环境大模型饱和，real-world 大模型 work。这暗示 sim-to-real gap 部分是 capacity 问题，部分是 representation 问题。你不能用模拟环境的 scaling law 直接推 real-world。

### 3. Proprioception 的双面性

同 embodiment 时加 proprioception 一定更好（提供精确 metric 信息）。但 cross-embodiment zero-shot transfer 时反而有害（coordinate 不对齐）。这是个工程上很重要的 insight——你 transfer 时要知道哪些 modality 是 embodiment-specific 的。

### 4. Gradient-based planner 在"贪心可解"任务上意外地强

Adam 在 Metaworld 上 beat 所有 sampling-based 方法。但一旦任务需要 non-greedy planning（Wall 要绕门）或 cost landscape 多模态（DROID），梯度方法就崩。这告诉你：**选择 planner 之前先想清楚你的 task 是不是贪心可解**。

### 5. Counterfactual 实验很漂亮

Figure 2 是个很巧的 qualitative 评估：固定初始状态，hardcode 两种 action（"open and move up" vs "close and move up"），看 model 能否区分"抓起杯子"和"没抓起杯子"的差异。Ours 能正确预测 grab cup 的效果，DINO-WM 和 V-JEPA-2-AC 都不能。

这是个很 cheap 的 sanity check——你不需要跑完整 planning pipeline，只需要看 model 是否理解 action 的因果效果。

### 6. DROID 上的 Action Error 陷阱

Figure S5.10 揭示：DROID 上 7 维 action 中，前 3 维（end-effector position）误差随训练下降，但后 4 维（orientation + gripper）反而上升。如果你用全部 7 维算 Action Error，会得到"训练越久越差"的错觉——其实是 gripper dimension 在 OOD action 上乱预测。

所以 paper 只用前 3 维算 Action Score。这告诉你：**评估 metric 的设计要懂 task 结构**，不然会被误导。

### 7. Training rollout 长度要匹配 test-time

2-step 在模拟环境最好，因为 planning 时 $W^p = 2$。6-step 在 DROID 最好，因为 real dynamics 更复杂。训练时 unroll 太长会让模型 over-specialize 到训练时的 unrolling pattern，与 test-time 不一致。

**直觉**：你的训练任务应该尽量接近 test-time 任务。这是 scheduled sampling 和 DAgger 的核心思想，在 JEPA-WM 上同样成立。

---

## 对做 physical agent 的人的 practical advice

1. **Encoder 选 DINO 系列**，不要选 video encoder。除非你有特别强的理由。
2. **Predictor 用 AdaLN conditioning**，不要用 feature concat 或 sequence concat。
3. **Planner 默认 CEM + $L_2$**。如果 task 贪心可解，可以试 Adam。如果要 transfer 到新数据集又不想调参，用 NG。
4. **加 proprioception**，除非你做 cross-embodiment transfer。
5. **Training rollout 2-3 步**，不要太多。Real-world 数据可以更长。
6. **Context size 2-3 帧**，需要 velocity。Real-world 可以 5 帧。
7. **不要盲目 scaling**。模拟环境小模型就够，real-world 才需要大模型。
8. **评估时小心 metric 设计**。Success rate 噪声大，用多个 metric 交叉验证。
9. **TBPTT 是你的朋友**。unroll 时 detach gradient，省内存且效果更好。
10. **V-JEPA-2-AC 的 public checkpoint 有 bug**，复现时注意检查 2-step loss 的计算。

---

## Open Questions

Paper 自己提到的：
- Metaworld object manipulation 上 model 会 hallucinate grasping object（Figure S5.7）——gripper action dimension 需要单独优化
- Robocasa "Pick" task 失败率高——稍微 misestimate end-effector 位置对小物体就致命
- Sim-to-real camera calibration shift（Figure S5.3）——模型在 Robocasa 上预测始终左偏

我觉得还有几个值得探索的：
- **Action space 的结构**：现在 action 是 flat vector，但 gripper open/close 是 discrete 的，position 是 continuous 的。混合 action space 怎么处理？
- **Long-horizon planning**：现在 horizon 才 3-6，real task 可能需要 60+。Hierarchical planning 怎么和 JEPA-WM 结合？
- **Reward-free pretraining + reward fine-tuning**：JEPA-WM 是 reward-free 的，但 real task 通常有 reward。怎么高效 incorporate reward？
- **Multi-view**：DROID 有 3 个 camera，现在只用 1 个。Multi-view fusion 怎么做？
- **Closed-loop planning**：现在是 open-loop unroll + MPC，但 predictor 的 error 会累积。怎么做 closed-loop correction in latent space？

这些都是在 LeCun 的 [AMI 路线图](https://openreview.net/pdf?id=BZ5a1r-kVsf)里还没完全解决的问题。

---

## 最后的 takeaway

这篇 paper 的价值不在于发明新东西，而在于**把一个正在 fast-moving 的领域的 design choice 讲清楚了**。这对后续工作很有指导意义——你不用再一个个试，照着 recipe 来就行，然后在你关心的维度上继续 push。

而且它揭示了几个 deep insight：
- **Representation 和 dynamics 解耦是有效的**，但 encoder 的 inductive bias（DINO vs V-JEPA）会 strongly 影响 downstream planning
- **Planning 和 world model 解耦是 partial 的**——好的 prediction 不等于好的 planning，cost landscape 的几何结构更重要
- **Scaling 不是万能的**，capacity 要匹配 task complexity
- **Train-test consistency 很重要**，rollout length、context length 都要 match

这些 insight 对任何做 model-based RL / world model / planning 的人都有用，不只是 JEPA-WM 这家人。

---

# 深入解读：What Drives Success in Physical Planning with JEPA-WMs

## 1. Paper 的核心定位

这是 Meta FAIR 团队（包括 Yann LeCun, Adrien Bardes, Basile Terver 等）的一篇**系统性 empirical study**，目标是回答一个具体问题：**在预训练 visual encoder 的 embedding space 上学习 dynamics model 用于 robotic planning，哪些 design choice 真正起作用？**

这篇 paper 本身不提出全新算法，而是把 PLDM ([Sobal et al., 2025](https://arxiv.org/abs/2502.09164))、DINO-WM ([Zhou et al., 2024a](https://arxiv.org/abs/2411.04983))、V-JEPA-2-AC ([Assran et al., 2025](https://arxiv.org/abs/2506.07627)) 这类方法统一形式化为 **JEPA-WM** 家族，然后对 7 个关键组件做 ablation。最终组合出比两个 baseline 都更强的 model。

代码已开源：[github.com/facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms)

---

## 2. JEPA-WM 的统一形式化

### 2.1 训练阶段公式

核心训练 loss（Eq. 1）：

$$\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} L\left[ P_\theta\left( E_{\phi,\theta}(o_{t-w:t}^b), A_\theta(a_{t-w:t}^b) \right), E_{\phi,\theta}(o_{t+1}^b) \right]$$

变量含义：
- $B$：batch size
- $o_{t-w:t} := (o_{t-w}, \ldots, o_t)$：长度为 $w$ 的过去 observation 窗口（包括视觉和可选的 proprioception）
- $a_{t-w:t}$：对应的 action 窗口
- $E_{\phi,\theta} = (E_\phi^{vis}, E_\theta^{prop})$：global state encoder，由 **frozen visual encoder** $E_\phi^{vis}$ 和可训练的 shallow proprioceptive encoder $E_\theta^{prop}$ 组成
- $A_\theta$：action encoder（通常是 linear layer）
- $P_\theta$：predictor（这里是 ViT，带 frame-causal attention mask）
- $L$：pairwise loss，本文用 MSE

**关键设计**：visual encoder frozen，只有 predictor、action encoder、proprioceptive encoder 是 trainable。这与 DINO-WM 和 V-JEPA-2-AC 一致。

### 2.2 Planning 阶段公式

Planning objective（Eq. 2）：

$$L_\alpha^p(o_t, a_{t:t+H-1}, o_g) = (L_{vis} + \alpha L_{prop})\left( G_{\phi,\theta}(o_t, a_{t:t+H-1}), E_{\phi,\theta}(o_g) \right)$$

变量含义：
- $o_t$：当前 observation
- $a_{t:t+H-1} := (a_t, \ldots, a_{t+H-1})$：horizon 为 $H$ 的 action 轨迹，每个 action 维度为 $A$
- $o_g$：goal observation
- $L_{vis}$、$L_{prop}$：分别在视觉和 proprioceptive embedding 上的 dissimilarity（$L_1$、$L_2$ 或负 cosine similarity）
- $\alpha \geq 0$：proprioception 的权重
- $G_{\phi,\theta}$：unrolling 函数

### 2.3 Unrolling 函数

递归定义（Eq. 3, 4）：

$$\hat{z}_{i+1} = P_\theta\left( \hat{z}_{i-w:i}, A_\theta(a_{i-w:i}) \right), \quad i = t, \ldots, t+k-1$$
$$z_t = E_{\phi,\theta}(o_t)$$

这里 $\hat{z}_{i-w:i}$ 是一个 sliding context window，最大长度 $W^p$（planning 时使用）。Planning 时 predictor 像一个 RNN 一样递归 unroll，每次只取输出的最后 timestep 拼到 context 里继续预测。

---

## 3. 研究的 7 个 Design Choices

Paper 按**影响范围优先级**排序：先 fix planning-time 选择，再做 training/architecture ablation，最后 scaling 验证。

### 3.1 Planner 选择

测试 4 种 optimizer：
- **CEM**（Cross-Entropy Method）：CMA-ES 家族的简化版，对角协方差
- **NG**（NeverGrad / NGOpt）：meta-optimizer，自动选择 diagonal CMA-ES
- **Adam**：梯度下降
- **GD**：vanilla gradient descent

每个 planner 通用 hyperparameter：
- $H$：planning horizon
- $m \leq H$：实际 step 到环境的 action 数
- $W^p$：planning 时的 sliding context 长度
- $N$：并行评估的 candidate trajectories 数
- $J$：optimizer iteration 数

**核心发现**（Figure 3a）：
- **CEM + $L_2$** 总体最好
- Gradient-based（Adam, GD）在 Metaworld 这种平滑 cost landscape 上很强，但在 2D navigation（Wall, Push-T, Maze）上完全失败——会陷 local minima
- NG 在 DROID/Robocasa 上与 CEM 持平，**但不需要 hyperparameter tuning**，这对 transfer 到新数据集很实用
- $L_2$ cost 始终优于 $L_1$ cost

CEM 算法（Algorithm 1）的核心更新：
$$\mu^{j+1} = \frac{1}{K}\sum_{k=1}^K (a^{(k)}_{t:t+H-1})$$
$$\sigma^{j+1} = \sqrt{\frac{1}{K-1}\sum_{k=1}^K\left[(a^{(k)}) - \mu^{j+1}\right]^2}$$

其中 $K$ 是 top-K 最低 cost 的 trajectories。

### 3.2 Multistep Rollout Training

除 teacher-forcing loss 外，加上 k-step rollout loss（Eq. 5）：

$$\mathcal{L}_k = \frac{1}{B}\sum_{b=1}^B L\left[ P_\theta\left( \hat{z}_{t-w:t+k-1}^b, A_\theta(a_{t-w:t+k-1}^b) \right), E_{\phi,\theta}(o_{t+k}^b) \right]$$

其中 $\hat{z}_{t+k-1}^b = F_{\phi,\theta}(o_t, a_{t-w:t+k-2})$。

**关键实现**：用 **Truncated Backpropagation Through Time (TBPTT)**，每次 unroll 后 detach gradient，只 backprop 最后一步的 prediction error。

测试了多种 rollout 策略（Figure S2.1）：
- **Increasing order**：prediction order 随 timestep 递增
  - "Last-gradient only"：只把最新预测拼到 context
  - "All-gradients"：计算所有可用 loss terms
- **Equal-order**：每步用完整前一步输出作为输入
- 加 scheduled sampling（Bengio et al., 2015）

**结论**：
- 2-step "Last-gradient only" + random initial context 最好
- 模拟环境最优是 2-step，DROID 上是 6-step
- $k > 3$ 在模拟环境反而下降——因为 planning 时用 $W^p = 2$，太长的 rollout loss 让模型对 test-time 任务 over-specialize

### 3.3 Proprioception

对比加 / 不加 proprioceptive input：
- **加 proprioception 一致更好**（Figure 4a）
- 在 Metaworld 上，不加 proprioception 失败的主要原因是 arm 到达 goal 后在 goal 周围震荡
- 2D navigation 也受益——proprioception 让 plan 更精确
- DROID → Robocasa 是 zero-shot transfer，proprioception space 不对齐，所以这两个任务**不用 proprioception**（$\alpha = 0$）

### 3.4 Training Context Size $W$

测试 $W = 1$ 到 $W = 7$：
- $W=1 \to W=2$ 有大幅提升——predictor 需要 2 帧推断 velocity
- 3 帧可以推断 acceleration
- 模拟环境最优 $W = 3$，DROID 最优 $W = 5$（real-world dynamics 更复杂）
- **硬约束**：$W^p \leq W$，否则 planning 时让模型做训练时没见过的预测任务，预测快速退化

### 3.5 Encoder Type

测试 4 种 frozen encoder（Figure 4b）：
- **DINOv2**（image encoder）
- **DINOv3** ([Siméoni et al., 2025](https://arxiv.org/abs/2508.10104))
- **V-JEPA** ([Bardes et al., 2024](https://arxiv.org/abs/2406.07679))
- **V-JEPA-2** ([Assran et al., 2025](https://arxiv.org/abs/2506.07627))

**关键发现**：
- **DINO 系列全面优于 V-JEPA 系列**——因为 DINO 有更强的 fine-grained object segmentation，这对 manipulation/navigation 至关重要
- DINOv3 在 photorealistic 环境（DROID, Robocasa）上显著优于 DINOv2
- 在 Maze, Wall 上 DINOv3 反而比 DINOv2 慢且最终 success rate 低

**Encoding 技巧**：用 video encoder 时，最佳做法是**把每帧 duplicate 一份**组成 2-frame video，独立 encode 每一对。这样和 image encoder 等价。直接用 video encoder 的时序依赖反而更差（且要 frame-causal mask 防 leakage）。

### 3.6 Predictor Architecture

测试 4 种 conditioning 方式（Figure 5a）：
1. **Feature conditioning + sincos**（DINO-WM 用）：action embedding 沿 embedding 维度 concat 到 visual feature，hidden dim 从 $D$ 增到 $D + f_a$
2. **Sequence conditioning + RoPE**（V-JEPA-2 用）：action 作为独立 token，沿 sequence 维度 concat
3. **Feature conditioning + RoPE**：混合
4. **AdaLN + RoPE**（[Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)）：action embedding 通过 Adaptive LayerNorm 在每个 block 调制 scale 和 shift
5. **AdaLN-zero**：初始化时 conditioning MLP 输出 0，让 predictor 开始时像 unconditional ViT

**Action ratio** 分析：
- Feature conditioning：$\frac{f_a}{D + f_a}$
- Sequence conditioning：$\frac{1}{hw+1} = \frac{1}{257}$（h=w=16）

**结论**：
- **AdaLN + RoPE 平均最好**——action 信息在每层都注入，避免在深层 vanishing
- 结果 task-dependent：Metaworld 上 sincos+feature 最好
- AdaLN-zero 在 DROID, Push-T, Maze 这些信号最可靠的环境上反而不如 AdaLN

### 3.7 Model Scaling

测试 encoder ViT-S → ViT-B → ViT-L，predictor depth 3 → 12：

**关键发现**（Figure 6）：
- **模拟环境 scaling 不 work**——更大模型反而可能有害（embedding space 太大让 planning 优化更难区分 nearby states，见 Figure S5.9）
- **DROID 上 scaling 显著 work**——encoder size 和 predictor depth 都正相关
- 模拟环境最优 predictor depth 通常 6，2D navigation 甚至 3 就够
- Real-world dynamics 需要更高 capacity

---

## 4. 实验设置细节

### 4.1 Datasets

| Dataset | 规模 | Traj. Length | 用途 |
|---|---|---|---|
| PointMaze | 2000 | 100 | 2D navigation |
| Push-T | 18500 | 100-300 | 2D push task |
| Wall | 1920 | 50 | 2D navigation with door |
| Metaworld | 12600 | 100 | 3D manipulation |
| DROID | 8000 | 20-50 | Real-world Franka |
| Robocasa | 16 teleop traj | - | Zero-shot 评估 |

DROID 用的是 [Khazatsky et al., 2024](https://arxiv.org/abs/2403.12945) 的 dataloader，actions 定义为 end-effector position delta。Robocasa 评估需要自定义简化的 pick-and-place task，因为原任务 horizon 太长。

### 4.2 Metrics

主指标：**Success Rate**

辅助指标（用于诊断，不依赖 planner）：
- Embedding space error throughout unrolling
- Proprioceptive decoding error
- Visual decoding LPIPS（[Zhang et al., 2018](https://arxiv.org/abs/1801.03924)）
- **Action Error**（DROID 上用，因为不 step 到真实 robot）：$800(0.1 - E)$ if $E < 0.1$ else 0，其中 $E$ 是 $L_1$ action error

### 4.3 统计可靠性

- 每个 final model 训练 3 个 seeds
- 每 epoch 跑 $e = 96$ 个 episode（DROID $e=64$，Robocasa $e=32$ 因为每个 episode 要 replan 12 次）
- Aggregate score 取最后 $n=10$ 个 epoch 平均（DROID $n=100$）

---

## 5. 最终的 Optimum 模型

| 组件 | Simulated Env | DROID / Robocasa |
|---|---|---|
| Encoder | DINOv2 ViT-S | DINOv3 ViT-L |
| Predictor | ViT-S, depth 6 | ViT-L, depth 12 |
| Conditioning | AdaLN + RoPE | AdaLN + RoPE |
| Proprioception | Yes | No |
| Rollout loss | 2-step | 6-step |
| Context $W$ | 3 | 5 |
| Planner | CEM $L_2$ | CEM $L_2$ |

**结果对比**（Table 1）：

| Model | Maze | Wall | Push-T | MW-R | MW-RW | Rc-R | Rc-Pl | DROID |
|---|---|---|---|---|---|---|---|---|
| DINO-WM | 81.6 | 64.1 | 66.0 | 44.8 | 35.1 | 19.1 | 21.7 | 39.4 |
| V-JEPA-2-AC | - | - | - | - | - | 16.2 | 33.1 | 42.9 |
| **Ours** | **83.9** | **78.8** | **70.2** | **58.2** | **41.6** | **25.4** | **30.7** | **48.2** |

在所有 8 个环境上几乎全面胜出。

---

## 6. 几个值得深挖的技术细节

### 6.1 V-JEPA-2-AC 的 Bug 修复

Paper Section B 提到一个有趣发现：V-JEPA-2-AC 原代码里 2-step rollout loss 算错了。实际计算的是 $\|P_\phi(a_{1:T}, s_1, z_1) - z_T\|_1$，意思是当输入 groundtruth $z_1$ 和 prediction $\hat{z}_2$ 时，模型被训练输出 $\hat{z}_2$。作者修复这个 bug 后重新训练。但有意思的是：**原版有 bug 的 public checkpoint 在 image decoding 和 counterfactual test 上看起来正常**，只是 planning 性能差很多——这说明 latent rollout 质量好不等于 planning 好。

### 6.2 成功率与 validation metric 的相关性

Section E.3 做了 Spearman correlation 分析（Tables S5.3-S5.6）：
- 在 Wall 上 visual embedding $L_1$ loss 与 success rate 相关性最高（0.81）
- 在 Metaworld 上 correlation 整体较弱（0.47 最高），因为 long-horizon unrolling 更难
- **重要观察**：更好的 world model 不等于更好的 planning——可能因为 planner 会卡 local minima，或者 sample OOD action

### 6.3 Counterfactual 实验

Figure 2 是一个很漂亮的 qualitative 评估：固定初始状态，hardcode 两种 action（"open and move up" vs "close and move up"），看 model 能否区分抓与不抓物体的差异。Ours 能正确预测 grab cup 的效果，DINO-WM 和 V-JEPA-2-AC 都不能。

### 6.4 DROID 上的 Action Error 诊断

Figure S5.10 揭示了一个陷阱：DROID 上 7 维 action 中，前 3 维（end-effector position）误差随训练下降，但后 4 维（orientation + gripper）反而上升。所以 paper 只用前 3 维算 Action Score。如果不 clip action magnitude（像 V-JEPA-2-AC 那样），总 error 会更大——因为 planner 会 sample OOD action。

### 6.5 CEM vs NG 的收敛行为

Figure S4.1 显示：在同一 Metaworld 失败 episode 上，NG planner 收敛更慢、更 exploratory；CEM 更快 collapse 到 tight distribution。在需要 precise action 的任务（如 Push-T）上，CEM 更优；在多模态 cost landscape（DROID）上，NG 不需要调参就能匹配 CEM。

---

## 7. 对 Karpathy 直觉的几点启发

1. **Encoder frozen 是关键简化**：把 representation learning 和 dynamics learning 解耦，让 dynamics model 只需要在 fixed feature space 上学预测。这也让 scaling predictor 更纯粹。

2. **Planning 和 world model 解耦**：很 surprising 的是，latent rollout 看起来对（decoding 漂亮）但 planning 不 work。这说明 cost landscape 的几何结构比 prediction accuracy 更重要。Section E.3 的 correlation 分析揭示了这点。

3. **Action conditioning 的信息流**：AdaLN 在每层注入 action，避免了 action 信息在深层 transformer 中 vanishing。这和 Diffusion Transformer ([DiT, Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)) 的发现一致。

4. **Scaling 的 task dependence**：模拟环境在小模型上就饱和，real-world 数据需要大模型——这暗示 sim-to-real gap 部分是 capacity 问题，部分是 representation 问题。

5. **Proprioception 的角色**：在 zero-shot transfer 场景下 proprioception 反而有害（因为 embodiment 不对齐），但在同 embodiment 训练-评估时 proprioception 提供 precise distance to goal 信息，避免 greedy planning 的震荡。

6. **Multistep rollout 的 sweet spot**：2-step 在模拟环境最优，6-step 在 DROID 最优。这和 test-time 用的 $W^p=2$ 有关——train-time rollout 太长会让模型 overfit 到 train-time 的 unrolling pattern，与 test-time 不一致。

---

## 8. 一些不足和开放问题

Paper 自己提到：
- 在 Metaworld object manipulation 上，model 会 hallucinate grasping object（Figure S5.7）——gripper action dimension 需要单独优化
- Robocasa "Pick" task 失败率高，因为稍微 misestimate end-effector 位置对小物体就致命
- Sim-to-real camera calibration shift（Figure S5.3）——模型在 Robocasa 上预测始终左偏
- Equal-order rollout 不 work，但作者没完全解释为什么 attention-based routing 在 manipulation 上更好

可以延伸阅读：
- [LeCun, 2022 - A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [DINO-WM](https://arxiv.org/abs/2411.04983)
- [V-JEPA-2 technical report](https://arxiv.org/abs/2506.07627)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [π₀ (Physical Intelligence)](https://arxiv.org/abs/2410.24164)
- [DiT - Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

整体看，这篇 paper 是 LeCun 路线（JEPA + planning in latent space）的一次重要 empirical clarification。它不发明新东西，但通过非常仔细的 ablation 把"什么 work、什么不 work、在什么条件下 work"讲清楚了，对后续 building physical agent 的工作很有指导价值。
