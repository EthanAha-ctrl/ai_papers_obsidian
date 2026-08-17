---
source_pdf: World-VLA-Loop Closed-Loop Learning of Video World Model and VLA Policy.pdf
paper_sha256: 0a3c7fa596aaf70b562c27b1502036f9fc6ecbd55a0b8e1002f5f2611ef708b3
processed_at: '2026-08-13T05:27:56-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 World-VLA-Loop

好，那我把刚才那堆术语全压成大白话再讲一遍。但有几个数学骨架我还是得留，因为光讲故事你 build 不了真正的 intuition——光知道"joint training 有用"不够，你得知道**为什么 reward head 反过来能让 video 生成变准**，这个机制才是这篇 paper 的灵魂。

---

## 一句话版

想教机器人 RL，但在真机上跑几千次 rollout 太贵也太危险。于是用 video world model 当虚拟考场来跑 rollout，可现在的 video world model 有个毛病——你给它一个明显错误的 action，它还是给你 hallucinate 出"成功"的视频。这篇 paper 的解法是：**让 world model 和 policy 互相纠错、一起进化**，world model 当考场，policy 当考生，考生找出考场的漏洞，把漏洞填上，再让下一版考生来考。

paper: https://showlab.github.io/World-VLA-Loop/

---

## 痛点到底是什么，用人话讲

想象你训练了一个机器人 policy，想用 RL 让它变得更强。RL 需要什么呢？需要它自己试错几千次，每次试完告诉你成功还是失败，拿到 reward 信号再更新参数。

问题是，真机上试错一次要几秒钟，加上失败后还得人去把杯子扶起来 reset，再来一遍。想跑 500 次 rollout，你博士毕业了都还没跑完。π0.6 (https://arxiv.org/abs/2511.14759) 那种真机 RL 是有钱有人能搞，普通 lab 玩不起。

所以大家都想做"虚拟考场"——simulator。simulator 老三条路：

1. **手搭 digital twin**：你自己写一个物理引擎 + 资产，看起来就不像真的，sim-to-real gap 大。
2. **3D 重建**（Gaussian Splatting 那类，如 https://arxiv.org/abs/2509.00086）：精度高但只能拍过的场景能用，机器人一旦探索到没见过的角落就崩，RL 根本不敢探索。
3. **Video world model**（Cosmos-Predict 2 这类，https://arxiv.org/abs/2511.00062）：用 diffusion 生成视频，泛化好，但有个致命问题。

致命问题看 Figure 2 就一目了然：你给 Cosmos-Predict 2 一个 action，说"机械臂往左偏一点去抓杯子"——这个 action 是错的，明显抓不到。但 video model 还是给你生成了一个"成功抓住杯子"的视频。它为什么会这样？因为它在大规模 video 数据上预训练过，视觉先验告诉它"机械臂靠近杯子 → 抓住 → 拿起来"这种 sequence 很常见，它就脑补了一条成功 trajectory，完全无视你给的 action 是错的。

**用这种 world model 当 RL simulator，policy 直接 reward hacking**——它会学会"做什么 action 都行反正都成功"，最后学出一堆废动作。

---

## 三个核心 trick

这篇 paper 用了三招解决这个 hallucination 问题。每一招都不复杂，但合起来效果很猛。

### Trick 1: Near-success 数据（SANS dataset）

SANS = Success And Near-Success。除了成功 trajectory，他们还专门收集"差一点点就成功"的 trajectory——比如机械臂擦着杯子边缘过去没抓住，或者抓的位置偏了一两厘米。

为什么这个特别关键？你想想，纯成功数据训练出来的 world model，它见过的 action 全是"对的动作"。当你给它一个"几乎对但差一点点"的 action 时，它不知道这其实会失败，就用 visual prior 脑补成成功。

但如果你把大量"几乎对但失败"的 case 喂给它，它就被迫去学那个 fine-grained 的空间关系：gripper 中心距离杯把 2 厘米 → 成功；距离 4 厘米 → 失败。这就是 hard negative mining 的思路——你拿最难区分的 case 去逼模型学细的东西。

数据怎么来？ManiSkill（https://arxiv.org/abs/2410.00425）里他们用一个简单 controller 收 success，然后扰动 pose 造 failure，或者直接跑 policy 收它自然失败。LIBERO 和 real-world 里就用 OpenVLA-OFT（https://arxiv.org/abs/2502.19645）自己跑失败 case，加上人 teleoperation 造一些 plausible 的失败。Real-world 10Hz，每 task 大概 50 success + 50 near-success。

这里其实有个跟你之前一直强调的 "data quality > data quantity" 完全一致——100 条精心挑选的 near-success 比几千条 success 信息量大得多，因为它们定义了决策边界在哪。

### Trick 2: Reward prediction head（joint training）

这是个非常巧的设计。Cosmos-Predict 2 原本只是个 video generator——输入历史帧 + action，输出未来帧。World-VLA-Loop 在 DiT 输出的 latent 上加了一个轻量 MLP，让它额外预测一个 scalar reward：

$$\hat{r}_t = \phi(z_t)$$

这里 $z_t$ 是 DiT 在 video timestep $t$ 上的 denoised latent，$\phi$ 是个 MLP，$\hat{r}_t$ 是预测的 reward（binary 0/1，表示这步是不是 success state）。

训练 loss 是：

$$\mathcal{L} = \mathcal{L}_{flow} + \lambda \sum_{t=1}^{T} \|\hat{\mathbf{r}}_t - \mathbf{r}_t\|^2$$

- $\mathcal{L}_{flow}$：标准的 flow matching loss，让 generator 学会生成视频。Flow matching (https://arxiv.org/abs/2210.02747) 是 diffusion 系的另一条路，学一个 velocity field 把 Gaussian 推到数据分布。
- $\|\hat{\mathbf{r}}_t - \mathbf{r}_t\|^2$：reward MSE，让 reward head 学会从 latent 里读出 success/fail。
- $\lambda$：随 noise level 调的权重，遵循 EDM (https://arxiv.org/abs/2206.00389)。

为什么 $\lambda$ 要随 noise level 变？这个细节很关键。diffusion 生成过程中，早期 latent 几乎是纯噪声。这时如果 $\lambda$ 大，reward loss 会逼着 generator 把 latent 在高噪声下也"调成"能 decode 出 reward——这会把生成质量毁掉。EDM 的做法是 high noise 时降权、low noise 时加权，让 reward head 主要在 latent 接近 clean 时起作用。换句话说，**reward supervision 是个 "clean latent 上的 auxiliary task"**。

这个 joint training 是双向收益的，而且收益是反直觉的：

**收益 A（预期内）**：reward 比 VLM 当 judge 准。你看 Table 4，Qwen3-VL 当 reward judge 只有 50-55% alignment，比 random 好不了多少。为什么？VLM 看的是 world model 生成的 frame，frame 本身已经被 visual prior 污染成"看起来成功"，VLM 又相信它看到的——双重 hallucination 叠加。而 reward head 直接从 latent 读，绕过了 pixel level 的污染。

**收益 B（反直觉，最关键）**：reward head 让 **video 生成本身变准了**。Table 4 / Table 5 显示，去掉 reward head，visual alignment 从 85-95% 掉到 60-70%，掉 30%。

为什么 reward head 能反向帮 generator？这是这篇 paper 最深的 insight。直觉是这样：reward head 是接在 latent $z_t$ 上的一个线性 probe，它要能从 $z_t$ 读出 success/fail，那 $z_t$ 必须在 success/fail 这个维度上**线性可分**。generator 为了让 latent 在这个维度上 linearly separable，必须 encode "这个 action 会成功还是失败"的信息进去——这个信息只能来自 action conditioning。所以 reward head 实质上在"逼" generator 把 action 信号真正吃进去，不能偷懒用 visual prior。

你以前讲 representation learning 经常强调 "probe task shapes representation"——这就是个完美例子。reward head 是个 probe，它塑造了 generator 的 latent space，让 latent 在 task-relevant 维度上更 structured。

### Trick 3: Closed-loop co-evolution

前面两个 trick 都还是"一次训练"的范畴。这个 trick 才是 paper 的标题"Loop"的来源。

想象 Step 0：你用 SANS dataset 训了 world model v1，然后让 policy 在 world model v1 里跑 RL。policy 学会了在 world model v1 覆盖的 failure mode 上不犯错。但 policy 会探索出**新的 failure mode**——比如它学会抓杯子的背面（因为 world model v1 没见过背面抓取，给 reward 1，policy 就 reward hacking 了）。

Step 1：你把 RL 后的 policy 拿到真实环境 rollout，收集它的新失败 case（背面抓取失败的视频），加回 SANS dataset。重新 fine-tune world model 得到 v2，v2 现在懂"背面抓取会失败"。再让 policy 在 v2 里跑 RL，这次 policy 就没法 hack 这个 reward 了，得改抓正面。

Real-world 数字：SFT base 13.3% → iter 1: 36.7% → iter 2: 50.0%。每轮都有显著提升。

这个动态跟 LLM RLHF 里"用新 policy 的 generation 重新训 reward model"是同构的。policy 是 reward model 的对抗者，reward model 是 policy 的 verifier，两者一起演化。理论上多轮迭代应该收敛到 policy 找不到 world model 的盲点为止。

---

## 框架怎么转起来（Figure 3 的人话版）

四个 phase，循环往复：

1. **收数据**：ManiSkill / LIBERO / real-world 都收 success + near-success，带 action 和 sparse reward。ManiSkill 大规模 35k pairs / 23 tasks 用来 pretrain。
2. **训 world model**：在 ManiSkill SANS 上 pretrain Cosmos-Predict 2 + reward head，学到 Franka 机器人 + action 的基础物理关系。然后每个新 task 用 < 100 条 fine-tune。
3. **跑 RL**：把 world model 当 simulator，OpenVLA-OFT 在里面 rollout，用 GRPO 更新 policy。reward 从 world model 的 reward head 取，threshold 0.9 二值化。
4. **闭环**：RL 后的 policy 在真实环境 rollout，新失败 case 加回 SANS，重训 world model，回到 phase 3。

实现细节里有点意思的是 request-response 架构：world model 跑在 backend server 上，policy 生成 action chunk 发请求，server 分配 worker 生成下一帧 + reward。单卡 H100 生成 24 帧 batch 大概 7 秒，一个 task RL 50 步收敛，总共 30 小时。对比真机 RL 50 步根本不可能——这是本质上的效率差异。

---

## 结果有多好

### World model 生成质量（Table 1）

| Scenario | SSIM | PSNR | LPIPS | MSE |
|----------|------|------|-------|-----|
| LIBERO | 0.90 | 26.57 | 0.031 | 0.0024 |
| Real-World | 0.91 | 29.61 | 0.059 | 0.0019 |

SSIM 0.9+ 是结构几乎完美，LPIPS 0.05 以下人眼分不出。Real-world 比 LIBERO 还略好，反直觉但说得通——real-world texture 更丰富，pretrained prior 更适配。

### Alignment（Table 2）

visual alignment 平均 87.9%，reward alignment 平均 86.4%。两者高度一致——证明 reward head 和 generator 学到了一致的 success/failure 表示。Real-world 的 reward alignment 95% 比 visual 90% 还高，说明 reward head 可能比 pixel-level 判断更鲁棒，学到了某种 abstracted success signal。

### Policy RL 提升（Table 3）

最 striking 的数字：real-world SFT base 13.3% → RL 后 36.7%，绝对提升 23.4%，相对提升 176%。LIBERO 上各 task 提升 6-24%。LIBERO-100（长 horizon >200 帧）没做——因为 autoregressive video model 200 帧后 quality drift，这是公开 limitation。

### Ablation（Table 4）

去掉 near-success data：visual alignment 85-95% → 60-65%。去掉 reward head：85-95% → 60-70%。用 Qwen3-VL 当 judge：50-55%，**比 random 还差**——证明外部 VLM reward 在 video world model 上根本不靠谱。

---

## 跟你 Tesla 时期工作的直接关联

你之前在 Tesla 推 world model for autonomous driving。这篇 paper 是 manipulation 版的同一思路。最核心的 insight 一模一样：**纯生成不够，必须 grounded 到 task reward 才能当 simulator**。

在 driving 里，reward 可能是 collision / lane deviation / comfort。这篇 paper 用 binary task success，driving 里换 continuous risk metric 就行。架构上完全可以 transfer——把 Cosmos-Predict 2 换成 driving video model，把 reward head 换成 driving metric predictor，把 action 换成 steering/acceleration。如果 Tesla 的 world model 还在迭代，joint training + closed-loop SANS augmentation 是个非常直接可借鉴的方案。

特别值得注意的一点：driving 里"差一点点就出事"的 near-miss 数据本身就极有价值，跟 SANS 的 near-success 概念完全对应。Tesla 拥有车队数据天然就有海量 near-miss——这个数据怎么用进 world model 训练，这篇 paper 给了一个非常具体的 recipe。

---

## 局限和延伸思考

paper 自己承认的：
1. **Long-horizon**：autoregressive video >200 帧质量 drift。需要更长 context video backbone（Mamba / hierarchical latent）。
2. **Sparse reward**：只有终态 binary。改成 dense per-step reward 收敛更快，但标注成本高。

我觉得还没说但重要的：
3. **Reward head 只输出 binary**，没法表达 continuous cost（耗时、能耗）。多 head reward 能支持更复杂 preference RL。
4. **Action 6-DoF + binary gripper**，bimanual 或 dexterous hand 22-DoF 没验证。Additive action injection 在高维 action 上可能不够，需要 cross-attention。
5. **Closed-loop 只跑 2 iter**，5-10 iter 能不能持续提升？plateau 在哪？这跟 LLM RLHF 多轮表现应该很像。
6. **Stochastic world model 的 reward variance**：DiT 生成是 stochastic 的，同一 action 多次 rollout reward 可能不一样。这对 RL exploration 是好事但对 reproducibility 是 challenge，paper 没讨论。
7. **没 uncertainty quantification**：reward head 给 point estimate，没 calibrated probability。能加 confidence 的话可以做 curiosity-driven exploration，sample efficiency 应该更高。
8. **跟 active learning 结合**：closed-loop 数据采集本质是 active learning，但现在是被动收集 failure。如果 policy 主动 probe world model 的 uncertain region，效率会高很多。

---

## 一句话总结直觉

如果你只想记住一件事：

**video world model 当 RL simulator 失败的根因，是 success-only 数据让 generator 学到了 visual prior 而非 action grounding；解法是用 near-success 数据 sharpen 决策边界 + 用 reward head joint training 让 latent 编码 success 信息 + 用 closed-loop 让 world model 跟 policy 共同进化覆盖新 failure mode**。

整套 framework 数学上没新东西——flow matching + MSE + GRPO 都是现成的——但 insight 是把这三者通过 SANS dataset 这个数据层串成一个 self-improving loop。你那句 "simple idea, well executed" 放这非常贴切。

要不要我把 reward head 那个"为什么反方向能帮 generator"的机制再深挖一层？那里其实跟 contrastive learning、跟 probe shaping representation 都有更深的联系，可以再展开讲讲。

---

# World-VLA-Loop: 深入讲解

Hi Andrej，这篇 paper 是你会在机器人学习与 video world model 交叉领域里非常感兴趣的那种工作。它本质上回答了一个你之前在 Eureka Labs / Tesla / OpenAI 一直在思考的问题：**能不能不依赖真实物理 rollout，而是在一个 learned video world model 内部完成 VLA policy 的 RL post-training，并且让 world model 和 policy 互相 bootstrap**。下面我尽量把每个技术细节拆开，帮你 build intuition。

Paper 链接: https://showlab.github.io/World-VLA-Loop/
arXiv（相关 Cosomos-Predict 2）: https://arxiv.org/abs/2511.00062
SimpleVLA-RL: https://arxiv.org/abs/2509.09674
OpenVLA-OFT: https://arxiv.org/abs/2502.19645
GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
Flow Matching: https://arxiv.org/abs/2210.02747
EDM (Karras et al.): https://arxiv.org/abs/2206.00389
LIBERO: https://arxiv.org/abs/2306.03310
ManiSkill3: https://arxiv.org/abs/2410.00425

---

## 1. 核心问题与 motivation：为什么需要 closed-loop

VLA 的训练目前主流还是 imitation learning（SFT 在 demonstration 上）。Imitation learning 有两个根本病：(1) compounding error 在 test time 会指数级累积；(2) 数据稀缺且分布窄。RL post-training 理论上能修复这两个问题，但 real-world RL 在机器人上几乎不可行——样本效率低、需要几千次物理 rollout、每次失败都需要人 reset、还有安全风险。π0.6 (https://arxiv.org/abs/2511.14759) 这种工作证明了 real-world RL 是可行的，但成本极高。

那退路是用 simulator 做 RL。但 simulator 分三类，每一类都有问题：

1. **Handcrafted digital twins** (如 https://arxiv.org/abs/2506.18088 RobotTwin 2.0)：photorealism 和 physics fidelity 不够，sim-to-real gap 大。
2. **3D reconstruction** (Gaussian Splatting 类如 https://arxiv.org/abs/2509.00086 GWM, https://arxiv.org/abs/2506.17644 Drawer)：精度高但泛化差，OOD 区域（相机没拍到的地方、复杂物理交互）直接崩，没法支撑 RL 里的 stochastic exploration。
3. **Action-conditioned video world models** (https://arxiv.org/abs/2511.00062 Cosmos-Predict 2, https://arxiv.org/abs/2503.00200 UVAM, https://arxiv.org/abs/2510.00406 VLA-RFT)：泛化好但有**致命问题——action-following precision 差**。Figure 2 里展示了一个非常关键的现象：当你给 Cosmos-Predict 2 一个明显错误的 action（gripper 轨迹完全偏掉），它还是会 hallucinate 出一个"成功"的视频。这意味着 video world model 用了它的 visual prior，忽略了 action conditioning——这种 simulator 拿来做 RL，policy 直接 reward hacking。

所以 World-VLA-Loop 想解决的就是：**video world model 不能 follow action，导致没法当 RL simulator**。它的解法是双向的：算法层做 co-evolution，数据层引入 near-success trajectories。

---

## 2. 框架总览（Figure 3 的四个阶段）

四个 phase 形成一个闭环：

**Phase 1 — SANS dataset curation**：在 ManiSkill、LIBERO、real-world 三类环境里收集 success + near-success trajectories，每条都带 action 和 sparse reward。

**Phase 2 — World model pretraining**：在 ManiSkill SANS（35k video-action pairs, 23 tasks）上 fine-tune Cosmos-Predict 2 的 action-conditioned 版本，加 reward prediction head。

**Phase 3 — VLA RL post-training**：把 world model 当 simulator，OpenVLA-OFT 在里面 rollout，用 GRPO 更新 policy。reward 来自 world model 的 reward head（threshold 0.9）。

**Phase 4 — Closed-loop augmentation**：RL 后的 policy 在真实环境里 rollout，把 success / near-success 的新轨迹加回 SANS，重新 fine-tune world model，再进 Phase 3。

这个闭环的关键直觉是：**VLA policy 在训练过程中分布会漂移，world model 必须跟着 policy 的 failure mode 一起演化，否则 world model 永远在 policy 已经不存在的失败模式上做监督**。这一点跟 DAgger 的 motivation 是同源的——你在 SFT 上训练的 world model 和你 RL 后的 policy 之间有 covariate shift。

---

## 3. SANS dataset：为什么 near-success 是关键

SANS = Success And Near-Success。Near-success 的定义是：**轨迹几乎完成了任务，但因为 end-effector 定位的小误差失败了**（比如抓杯子擦着边缘过去没抓住）。

为什么这种数据特别重要，paper 给了两个理由：

(1) **强迫模型关注 fine-grained spatial dynamics**：near-success 和 success 在视觉上极其相似，world model 必须学到 gripper 和 object 之间几厘米级别的空间关系，否则区分不开。这相当于给 world model 一个 hard negative mining。

(2) **覆盖 policy 的实际 failure mode**：robot policy 在 rollout 时大概率就是 near-success 失败（擦边失败），而不是完全离谱的失败。如果 world model 只见过完全成功的轨迹，它遇到 near-success 时就会用 visual prior "脑补"成成功——这就是 Figure 2 里的 hallucination。SANS 直接把这种 near-success 喂给模型，让模型学到"这种 action 会失败"。

数据收集策略分两类：
- **ManiSkill**: 用 ground-truth object pose 写一个简单 controller 收 success，然后 perturb pose 生成 failure；或者直接从 policy rollout 里抓 failure。
- **LIBERO / real-world**: 用 OpenVLA-OFT 自己 rollout 抓 failure，加上人 teleoperation 造一些 plausible failure。Real-world 频率 10Hz，每 task 约 50 success + 50 near-success。

这个思路和你之前讲过的 "data quality > data quantity" 完全一致——100 条精心挑选的 near-success 比 1000 条 success 有用得多，因为它们定义了决策边界。

参考 RoboFAC (https://arxiv.org/abs/2505.12224) 和 AHA (https://arxiv.org/abs/2410.00371) 也探索 failure data，但它们是 QA-style reasoning，没 action annotation，也没法做 world model 训练。

---

## 4. State-aware video world model：架构细节

Base 是 Cosmos-Predict 2 的 action-conditioned version。Cosmos-Predict 2 本身是 DiT (Diffusion Transformer) backbone，autoregressive 预测 video chunk。

### 4.1 输入输出形式

给定：
- 历史 $h$ 帧 observation：$x_0, x_1, \ldots, x_{h-1}$
- 未来 $T$ 步 action：$a_1, \ldots, a_T \in \mathbb{R}^6 \cup \{0, 1\}$，其中 $\mathbb{R}^6$ 是 6-DoF end-effector pose (3 translation + 3 rotation, 通常用 axis-angle 或 quaternion 表示)，$\{0, 1\}$ 是 gripper open/close 二值状态。

输出：
- 未来 $T$ 帧 observation：$x_h, \ldots, x_{h+T-1}$

### 4.2 Action 注入机制

action embedder MLP 把每个 7-DoF action 映射成一个 latent tensor，然后**直接加到 diffusion timestep embedding 上**注入 DiT module。这是个比较朴素的设计——比起 cross-attention 注入，additive injection 假设 action 和 timestep 在同一个 latent space 里可加。好处是参数少、训练稳；坏处是 action 信号容易被 timestep 噪声淹没，这其实也部分解释了为什么原始 Cosmos-Predict 2 action-following 差。

### 4.3 Reward prediction head（核心创新）

DiT 输出 denoised latent $z_t$（注意这里的下标 $t$ 是 video timestep，不是 diffusion timestep——这点 paper 写得有点歧义，要小心），然后一个轻量 MLP $\phi$ 把 $z_t$ 映射成 scalar reward：

$$\hat{r}_t = \phi(z_t)$$

训练 loss 是 flow matching + reward MSE 的联合：

$$\mathcal{L} = \mathcal{L}_{flow} + \lambda \sum_{t=1}^{T} \|\hat{\mathbf{r}}_t - \mathbf{r}_t\|^2$$

变量解释：
- $\mathcal{L}_{flow}$：flow matching loss（Lipman et al., 2022），是 diffusion/flow 模型的标准生成 loss。flow matching 跟 DDPM 不同之处在于它学的是一个 vector field 把一个简单分布（如 Gaussian）push 到数据分布，loss 是 $\|v_\theta(z_t, t) - (z_1 - z_0)\|^2$ 这种形式，这里 $z_0$ 是 noise，$z_1$ 是 data，$v_\theta$ 是学的 velocity field。
- $\hat{\mathbf{r}}_t$：第 $t$ 步预测的 reward（标量，但写成 vector 形式可能是因为 chunk 里多个 step 一起算）。
- $\mathbf{r}_t$：第 $t$ 步 ground-truth reward，是 binary 0/1（success state indicator）。
- $\lambda$：weighting factor，**根据 sampled noise level 调制**，遵循 EDM framework (Karras et al., 2022)。

### 4.4 $\lambda$ 的 noise-level modulation（重要细节）

为什么 $\lambda$ 要随 noise level 变？直觉是：denoising 早期 latent $z_t$ 接近纯噪声，reward head 从噪声里读 reward 几乎不可能，如果这时 $\lambda$ 大，reward loss 会把 generator 拉向"让 latent 在高噪声下也能 decode 出 reward"——这会破坏生成质量。EDM 的做法是 high noise 时降权，low noise 时加权，让 reward head 主要在 latent 接近 clean 时起作用。这是个很优雅的设计——把 reward supervision 当作"clean latent 上的 auxiliary task"。

具体 EDM 的 noise schedule (Karras et al. 2022, https://arxiv.org/abs/2206.00389) 是基于 noise-to-signal ratio $\sigma$ 的，weighting $c_{\text{loss}}(\sigma)$ 通常形式是 $\sqrt{\sigma^2 + \sigma_{\text{data}}^2}/\sigma$。paper 这里没给精确公式，但你应该能 intuition 到：denoising 过程被建模成一个 PF-ODE，每一 step 的 latent 是 $z_\sigma$，reward head 看到的应该是 $z_{\sigma \to 0}$ 附近的 latent。

### 4.5 Joint training 的双向收益

这个 design 带来的两个收益：

**收益 A: Reward 可靠性**——reward 是从 generated video latent 里读出来的，intrinsic aligned with 视觉结果，比 external VLM judge（如 Qwen3-VL, https://arxiv.org/abs/2511.21631）或者 heuristic proxy 都准。Table 4 的 ablation 直接对比：Qwen3-VL 当 judge 只有 50-55% alignment，而 internal reward head 有 75-90%。VLM hallucinate 是因为它看视频 frame 判断 success/fail，但 video frame 本身是 world model 生成的，world model 又有 visual prior 偏向成功——VLM 就被骗了。

**收益 B: Video quality 提升**——joint training 强迫 generator 在 latent 里编码 success/fail 信息，这相当于一个 auxiliary regularizer。Table 4 / Table 5 显示去掉 reward head 后 visual alignment 从 85-95% 掉到 60-70%，掉了 30%。这是个相当强的耦合——reward supervision 实质上在"教" generator 区分细粒度的 action outcome。

直觉上你可以这么理解：reward head 像 a "success probe" 接在 generator 的 latent 上，它逼着 generator 的 latent space 在 success/failure 这个维度上线性可分。这个 linear separability 又反过来让 video generation 在这个维度上更精确，因为 generator 必须把"会失败"和"会成功"的 action 区分开来 encode。

### 4.6 Pretraining → fine-tuning 的 transfer

- **Pretraining**: 35k ManiSkill SANS pairs / 23 tasks 上训练，让 model 学到 Franka 机器人 + action 的基础物理关系。
- **Fine-tuning**: 用 LIBERO 或 real-world 的 ~100 条 trajectories（每个 task 80-100 条 success+near-success），全参数 fine-tune，小 learning rate。

这个 two-stage 设计很关键——ManiSkill pretrained model 已经"懂"了 robot kinematics 和 action grounding，到新环境只需要学新 object / texture / task-specific physics，所以 < 100 条就够。这跟 LLM 的 pretrain-finetune paradigm 完全平行。

---

## 5. World simulator 用于 GRPO

### 5.1 整体 RL 框架

base policy 是 OpenVLA-OFT (https://arxiv.org/abs/2502.19645)，RL 框架用 SimpleVLA-RL (https://arxiv.org/abs/2509.09674)，policy update 用 GRPO (https://arxiv.org/abs/2402.03300)。

GRPO 的核心是 group relative advantage：对同一个 state 采样 $G$ 个 rollout，计算 advantage：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

这里 $r_i$ 是第 $i$ 个 rollout 的 return（在 sparse reward setting 下就是终态的 0/1 reward）。GRPO 不需要 critic，只用 group 内的 mean/std 做 baseline，比 PPO 简单很多，特别适合 VLA 这种大 policy。

### 5.2 World model 替换物理 simulator

在 SimpleVLA-RL 原版里，rollout 是在真实物理引擎里跑的。World-VLA-Loop 把它替换成：
- **初始 frame** 来自原始 dataset
- **后续每一 step 的 observation** 由 world model autoregressive 生成，conditioned on policy 输出的 action chunk
- **reward** 由 world model 的 reward head 给出，threshold 0.9 二值化作为 GRPO 用的 binary success

policy 输出 chunk size 24（一次输出 24 步 action），world model 生成 24 帧 video 对应。chunk-based rollout 是 OpenVLA-OFT 的设计——比 step-by-step 更稳，因为 policy 不需要每步重新 attend。

### 5.3 Request-response 架构（实现细节）

GRPO 需要 batch optimization（一个 group 里多个 rollout 并行），所以 policy 和 world model 之间用 request-response：
- World model 跑在 backend server 上，监听 request
- Policy 生成一个 action chunk → 传给 server → server 分配给空闲 worker → worker 跑 24 帧生成 → 返回 observation + reward

效率数据：在单卡 H100 上，24 帧 batch 生成 ~7 秒。SimpleVLA-RL 通常 50 步收敛，所以单 task RL 训练 ~30 小时。这个数字你可以对比一下 real-world RL——real-world 每次物理 rollout 至少几秒 + reset 几秒，还要人盯着，30 小时根本不可能跑完 50 个 GRPO step。

### 5.4 Closed-loop 迭代（Step 0 → Step 1 → ...）

这是 paper 最核心的 selling point：

- **Step 0**: SANS = 手动 teleoperation + SFT policy rollout 的 success/near-success。World model 在这个 SANS 上 fine-tune。Policy 在这个 world model 上做 RL，得到 RL-checkpoint-1。
- **Step 1**: 把 RL-checkpoint-1 在真实环境 rollout，收集新的 success/near-success（重点是新的 failure mode，因为 RL 后 policy 的失败模式跟 SFT 不同），加进 SANS。World model 重新 fine-tune（仍然从 ManiSkill pretrained 初始化）。Policy 又从 SFT base 开始在新 world model 上 RL，得到 RL-checkpoint-2。

Real-world 数据：SFT base 13.3% → RL iter 1: 36.7% (+23.4%) → RL iter 2: 50.0% (+13.3% vs iter 1)。

直觉：iter 1 的 world model 还在覆盖 SFT 时代的 failure mode，所以 RL 后 policy 学会了"在 SFT failure mode 上不犯错"，但会探索出**新的 failure mode**（reward hacking：抓杯子背面）。这些新 failure 没在 world model 里，所以 world model 对这些 action 给错 reward。Step 1 把这些新 failure 收集进 SANS，world model 学会"抓背面会失败"，下一轮 RL 就不会再 hack 这个 reward。

这是个非常漂亮的 **adversarial co-training** 动态——policy 不断 probe world model 的盲点，world model 不断填盲点。理论上多轮迭代会收敛到 policy 找不到 world model 的盲点为止。

---

## 6. 实验结果深度解析

### 6.1 World model 生成质量 (Table 1)

| Scenario | SSIM ↑ | PSNR ↑ | LPIPS ↓ | MSE ↓ |
|----------|--------|--------|---------|-------|
| LIBERO | 0.90 | 26.57 | 0.031 | 0.0024 |
| Real-World | 0.91 | 29.61 | 0.059 | 0.0019 |

- **SSIM** (Structural Similarity Index): 0.9+ 表明结构上几乎完美。SSIM 范围 [-1, 1]，0.9 是非常高的。
- **PSNR** (Peak Signal-to-Noise Ratio, dB): 26-30 dB 是高质量 video generation 的水平（30 dB 约等于 noise/signal 比 1/31）。
- **LPIPS** (Learned Perceptual Image Patch Similarity): 0.03-0.06，越低越像，0.05 以下基本人眼分不出。
- **MSE**: 0.002 级别，pixel-level loss 极小。

关键发现：real-world 比 LIBERO 还略好（PSNR 29.61 vs 26.57）——这有点反直觉，但说明 real-world 的 visual texture 更丰富，diT 的 pretrained prior 更适配。

### 6.2 Alignment metrics (Table 2)

Visual Alignment vs Reward Alignment：

| Task | Vis Align | Rew Align |
|------|-----------|-----------|
| LIBERO-Object T1 | 85% | 75% |
| LIBERO-Object T2 | 95% | 90% |
| LIBERO-Goal T1 | 90% | 85% |
| LIBERO-Goal T2 | 75% | 75% |
| LIBERO-Spatial T1 | 85% | 90% |
| LIBERO-Spatial T2 | 95% | 95% |
| Real-World | 90% | 95% |

平均 visual 87.9%，reward 86.4%，两者一致性高——证明 reward head 和 generator 学到了一致的 success/failure 表示。Reward alignment 在 Real-World 上反而比 visual 高（95% vs 90%），说明 reward head 比 pixel-level 判断更鲁棒——这是个 surprising 的发现，意味着 reward head 可能学到了一些"高于 pixel"的 abstracted success signal。

### 6.3 Policy RL post-training 结果 (Table 3)

| Task | SFT Base | RL Post | Δ |
|------|----------|---------|---|
| LIBERO-Object T1 | 73.9% | 97.9% | +24.0% |
| LIBERO-Object T2 | 73.9% | 91.9% | +18.0% |
| LIBERO-Goal T1 | 91.9% | 100% | +8.1% |
| LIBERO-Goal T2 | 86.1% | 96.2% | +10.1% |
| LIBERO-Spatial T1 | 83.9% | 93.9% | +10.0% |
| LIBERO-Spatial T2 | 87.9% | 94.0% | +6.1% |
| Real-World | 13.3% | 36.7% | +23.4% |

观察：
- **Base 越弱，RL 收益越大**：Real-World base 只有 13.3%，RL 拉到 36.7%——相对提升 176%。LIBERO-Goal T1 已经 91.9%，RL 只能再挤 8.1%。
- **Real-world 的绝对提升 23.4% 是非常 striking 的**——意味着 world model simulator 真的 transfer 到真实物理了。这不是 sim-to-real，是 "learned-sim to real"。
- LIBERO-100（长 horizon，>200 frame）没做，因为 autoregressive video model 200 帧后 quality drift 严重——这是个公开的 limitation，跟你之前讲的 long-horizon video generation 的瓶颈完全一致。

### 6.4 Ablation (Table 4, 5)

| 配置 | Vis Align (T1/T2) |
|------|------------------|
| w/o near-success data | 60% / 65% |
| w/o reward head | 60% / 70% |
| Qwen3-VL as reward | 50% / 55% |
| **Ours** | **85% / 95%** |
| **Ours (reward)** | **75% / 90%** |

两个核心 ablation：
- **Near-success 数据**：去掉后 visual alignment 从 85-95% 跌到 60-65%。证明 near-success 是 action-following 精度的关键。没有 near-success，world model 又会退回 visual prior hallucination。
- **Reward head**：去掉后 alignment 从 85-95% 跌到 60-70%，证明 joint supervision 是 generator 精度的核心。
- **External VLM reward**：Qwen3-VL 当 judge 只有 50-55%，**比 random 还差**（注意：random 是 50%，55% 也基本没信息量）。VLM judge 看的是 world model 生成的 frame，frame 本身就被 visual prior 污染了，VLM 又信任 frame——双重 hallucination 叠加。这给你一个直觉：**reward signal 必须在 latent level，不能在 pixel level**——pixel 已经被 prior 污染，但 latent 在 reward head 的监督下保留了 grounding。

### 6.5 Qualitative findings (Figure 5, 6)

Figure 5 展示了 reward hacking 的具体 case：iter 1 的 RL policy 学会抓杯子的背面（因为 iter 0 的 world model 没见过背面抓取的失败，给 reward 1）。iter 2 把这些失败加进 SANS，world model 学会"背面抓取会失败"，policy 改抓正面。这是个非常具体的 closed-loop benefit example。

Figure 6 是 OOD action sequence 的生成：plate 初始化在 mug 前方，gripper 右移后回到上方，然后 retract 到 neutral pose——这些 action 序列在 fine-tune SANS 里没见过，但 world model 仍然能精确 follow。这证明 ManiSkill pretraining 学到的是**robot kinematics 通用映射**，而不是 task-specific memory。

---

## 7. 跟相关工作的对比和联想

### 7.1 跟其他 world model 方向对比

- **Cosmos Policy** (https://arxiv.org/abs/2601.16163)：unified 预测 action + video，但定位是 imitation learning / zero-shot execution，没设计成 RL 用的 interactive env。World-VLA-Loop 的 reward head 是关键差异。
- **GWM (Gaussian World Model)** (https://arxiv.org/abs/2509.00086)：3D reconstruction 路线，精度高但泛化差，不支撑 stochastic exploration。
- **VLA-RFT** (https://arxiv.org/abs/2510.00406)：在 world simulator 里做 RFT，但 world model 是外部的，没 closed-loop co-evolution。
- **WMPO** (https://arxiv.org/abs/2511.09515)：World Model-based Policy Optimization，跟 World-VLA-Loop 思路类似但更偏 policy 侧优化。
- **World-Env** (https://arxiv.org/abs/2509.24948)：用 world model 当 VLA post-training env，但同样没 closed-loop。

### 7.2 跟 RL for LLM 的类比

你之前对 LLM RLHF/RLHF/GRPO 思考很深。这个 paper 里有几个非常直接的类比：

- **World model = verifier / reward model**：在 LLM RL 里，verifier 给 token-level reward；在 robot 里，world model 的 reward head 给 step-level reward。但 world model 多一个能力——它还能"仿真"未来 state，相当于 verifier 也能预演回答的后果。这在 LLM 里没有完全对应物，最接近的是 process reward model + test-time search。
- **Closed-loop co-evolution = iterative DAgger + RLHF**：policy 漂移 → world model 更新数据 → policy 再训。这跟 LLM RLHF 里 "用新 policy 的 generation 重新训 reward model" 完全同构。
- **Reward hacking on iter 1 → fixed on iter 2**：跟 LLM RLHF 里 reward model over-optimization 现象一模一样，解法也一模一样——更新 reward model (这里就是 world model) 跟上 policy。

### 7.3 跟 diffusion model / video generation 的关联

- **Flow Matching** (Lipman et al., 2022) vs DDPM：Cosmos 用 flow matching，更连续、训练更稳。
- **EDM** (Karras et al., 2022) 的 noise schedule 和 weighting 是关键：reward head 的 $\lambda$ 调制就是这个 framework 的直接应用。如果你之前熟悉 EDM 的 "preconditioning" 和 $c_{\text{loss}}$，这个 paper 的 $\lambda$ design 是 straightforward 的延伸。
- **Autoregressive video chunk**：DiT 在 latent space 里 autoregressive，chunk size 24。这个跟 Sora / WAN (https://arxiv.org/abs/2503.20314) 等 video model 的 spatiotemporal patch 设计同源。

### 7.4 跟 model-based RL 的经典关联

这个 paper 本质上是 **Dreamer (https://arxiv.org/abs/1912.01603) 系列的 video 版**。Dreamer 在 latent state space model 里 rollout 做 actor-critic。World-VLA-Loop 把 state space 换成 video latent，把 reward model 换成 reward head on latent。区别：
- Dreamer 的 world model 是 reconstruction-based，pure latent；
- World-VLA-Loop 的 world model 是 video generation，pixel + latent 双重。
- Dreamer 用 actor-critic，World-VLA-Loop 用 GRPO（更适合 large pretrained policy）。

### 7.5 跟你思考过的 "next token prediction for everything" 的关系

你之前讲过：如果一切都可以 formulate 成 next-token / next-frame prediction，那一个 model 就够。这个 paper 给了一个**反例**：纯 video prediction（Cosmos-Predict 2 原版）action-following 差，必须加 reward head 联合训练才能 grounded。这说明纯生成 target 不足以学到 action grounding，需要一个 task-relevant auxiliary signal 把 generator 推向 action-sensitive 的方向。这跟你讲 "data 决定 model 学什么" 是一致的——success-only data 决定 model 学 visual prior，near-success + reward 决定 model 学 action grounding。

---

## 8. Limitations 和我的延伸思考

### Paper 自己承认的：

1. **Long-horizon**：autoregressive video >200 帧质量 drift，LIBERO-100 没做。这跟所有当前 video model 一样，需要 longer context video backbone（可能是 Mamba-based, 或者 hierarchical latent）。
2. **Sparse reward**：只有终态 binary reward，没用 step-wise sub-goal。如果改成 dense reward（per-step success probability），GRPO 收敛应该更快，但 dense reward 在 robot 上很难标。

### 我觉得还有几个隐含 limitation / interesting extension：

3. **Reward head 只能预测 success/fail，不能预测 continuous cost**（比如耗时、能耗）。在更 general 的 RL 里，reward 是 multi-dimensional 的。如果 reward head 输出 multi-head，能支持更复杂的 preference RL。
4. **Action space 是 6-DoF + binary gripper**，对 bimanual 或 dexterous hand（22-DoF）没验证。action 注入是 additive on timestep embedding，更高维 action 可能需要 cross-attention。
5. **Closed-loop 只跑了 2 iteration**。如果能跑 5-10 iteration，能不能持续提升？还是会有 plateau？这跟 LLM RLHF 的 multi-round 表现应该很像。
6. **Stochastic world model**：DiT 生成是 stochastic 的，每次 rollout 可能不同。这对 RL 是好事（增加 exploration），但对 reproducibility 是 challenge。paper 里没讨论同一个 action sequence 多次 rollout 的 reward variance。
7. **World model 的 uncertainty quantification**：reward head 给的是 point estimate，没给 calibrated probability。如果能在 OOD action 上给低 confidence reward，可以做 curiosity-driven exploration。
8. **跟 active learning 的结合**：closed-loop 数据采集本质是 active learning——policy 选 action，world model 收集 failure。如果 policy 能主动 probe world model 的 uncertain region，sample efficiency 应该更高。

### 跟你 Tesla 时期的工作关联：

你之前在 Tesla 推动 world model for autonomous driving。这个 paper 是 manipulation 版的同样思路。关键 insight 是一样的：**纯生成不够，必须 grounded 到 task reward 才能当 simulator**。在 driving 里，reward 可能是 collision / lane deviation / comfort。这个 paper 用 binary task success，driving 里可以用 continuous risk metric。架构上完全可以 transfer——把 Cosmos-Predict 2 换成 driving video model，把 reward head 换成 driving metric predictor，把 action 换成 steering/acceleration。如果 Tesla 的 world model 还在迭代，这个 paper 的 joint training + closed-loop SANS augmentation 是个非常直接的可借鉴方案。

---

## 9. 总结直觉

如果你只想记住一件事，记住这个：**video world model 当 RL simulator 失败的根本原因，是 success-only 数据让 generator 学到了 visual prior 而非 action grounding；解法是 (1) 用 near-success 数据把决策边界 sharpen 出来，(2) 用 reward head joint training 让 latent 编码 success 信息，(3) 用 closed-loop 让 world model 跟 policy 共同进化覆盖新 failure mode**。

整套 framework 在数学上没什么新东西——flow matching + MSE + GRPO 都是现成的——但 insight 是把这三者通过 SANS dataset 这个数据层串成一个 self-improving loop。这跟你一贯主张的 "simple idea, well executed" 完全契合。

要不要我接着展开哪一块？比如 GRPO 在 chunk action 上的具体 advantage 计算，或者 EDM noise schedule 对 reward head 的精确影响，或者 closed-loop 收敛性的理论分析？
