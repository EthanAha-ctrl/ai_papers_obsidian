---
source_pdf: Learning Real-World Action-Video Dynamics with.pdf
paper_sha256: ae0754b55ded61caa5ee4a806f45bf7f3e1538c7dfeeb33efb24dda75d6c908f
processed_at: '2026-08-05T13:40:37-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 HMA

Karpathy 你让我"用人话说"，那我就抛开 paper 的学术腔，像在 Tesla 食堂跟你吃饭时那样聊。

---

## 一句话总结

**把 Sora 那套 video diffusion 换成 GPT 那套 autoregression，加一个 "机器人动作接口"，你就能在笔记本 4080 上 22 帧每秒跑一个机器人世界模型——可以当 simulator 用、当 policy 用、当数据生成器用。**

---

## 为什么这事儿难（intuition）

你想象一下你想训练一个机器人 policy，但没数据。怎么办？传统路子是去真机采数据——一百万美金的 Franka arm 配一个 PhD 学生，一天采 50 条 trajectory，累死累活采 1000 条。

另一个路子是**自己造一个"会演的导演"**：给它看几百万段机器人视频，它学会了"在这个状态下，机械臂往下走 5 厘米，画面里这块红色物体就会被推到左边"。然后你让它演给你看——**这就是 world model**。

Sora 证明这事在通用视频上能 work，但 Sora 有两个问题：
1. **它不听话**——你没法说"机械臂往左 3 厘米"，只能描述"一只猫在草地上"
2. **它太慢**——生成一帧要 100 步 denoise，整个 sequence 要全过一遍，real-time 不可达

HMA 解决的就是这两件事：**把"动作"作为第一公民塞进去，并且把 diffusion 换成 autoregression**。

---

## 关键 idea 1：把"动作"当第一公民

你想想 GPT 是怎么 work 的：给一段 token 历史，预测下一个 token。

HMA 说：**机器人轨迹不就是 token 序列吗？只不过有两种 token——视觉 token 和动作 token**。

那我可以把它 flatten 成：

```
[o_1, a_1, o_2, a_2, o_3, a_3, ...]
```

然后用 GPT 的方式训 next-token prediction。模型学会的 distribution 就是：

$$
p(o_1, a_1, o_2, a_2, ...) = \prod_k p(X_k | X_1, ..., X_{k-1})
$$

这里 $X_k$ 既可以是 observation token $o$，也可以是 action token $a$。下标 $k$ 是序列位置。

**精髓在于这个 unification**：训完之后你想怎么用就怎么用：
- 给 $(o_1..o_t, a_1..a_{t-1})$，预测 $a_t$ → **这是 policy**
- 给 $(o_1..o_t, a_1..a_t)$，预测 $o_{t+1}$ → **这是 world model / forward dynamics**
- 给 $(o_1..o_t)$，预测 $o_{t+1}..$ → **这是 passive video prediction**（没 action）
- 给 $(o_1..o_{t+1})$，反推 $a_t$ → **这是 inverse dynamics，可以给无 label 视频打 action label**

一个模型，四种用法，跟你 Tesla AI Day 那个 "one network" 的思路一模一样。

---

## 关键 idea 2：Masked Autoregression 替代 Diffusion

这是这 paper 最聪明的地方。让我详细讲讲，因为这是 15× 加速的根源。

### Diffusion 的问题

Sora、IRASim 这套 diffusion 做法：要生成一帧图，先搞一个 pure noise tensor，然后跑 100 步去噪。每一步都要过整个 transformer 一次。生成 16 帧视频，大概要 $16 \times 100 = 1600$ 次完整 transformer forward。

### GPT autoregression 的问题

纯 GPT 的 next-token 是一个一个 token 吐。生成一张 256 token 的图要 256 步——**比 diffusion 还慢**。

### Masked Autoregression (MAR)：两边的好处都要

MAR 是 Kaiming He 组 2024 年的工作（https://arxiv.org/abs/2406.11838），核心 trick：

1. **训练时**：把一张图随机 mask 掉一半 token，让 transformer 预测被 mask 的部分（类似 BERT/MaskGIT，https://arxiv.org/abs/2202.04200）
2. **推理时**：先全部 mask，过一次 transformer 拿到所有 latent，然后用一个**小 diffusion head**（3 层 MLP，不是整个 transformer）去 decode 每个 token

关键洞察：**transformer 只过一次！denoise 只在小 MLP 上做**。

为什么 work？因为 transformer 的 attention 已经把所有 token 的 context 融合好了——它知道"这个位置应该是红色物体的边缘"，剩下的只是把这个"应该是"变成具体的 pixel value，这事儿一个小 diffusion MLP 就能搞定。

数学上：

$$
\mathcal{L}_{soft} = \|\epsilon_o - f(o | t, z)\|^2
$$

- $\epsilon_o$：加在 token 上的高斯噪声，$\epsilon_o \sim \mathcal{N}(0, I)$
- $t$：diffusion 时间步，从 1000 训练 / 100 推理
- $z$：transformer 输出的 latent（"soft token"）
- $f(\cdot | t, z)$：3 层 MLP diffusion head，给定噪声等级 $t$ 和 latent $z$，预测应该减掉的噪声

这跟 Stable Diffusion 的 latent diffusion 概念上一致，但**prior 不是 U-Net 而是 transformer**，而且 transformer 是 autoregressive 的。

### 实测速度对比

| Model | 方法 | 参数 | FPS |
|---|---|---|---|
| IRASim-XL | Diffusion (DiT) | 679M | 0.28 |
| IRASim-XL amortized | Diffusion | 679M | 0.58 |
| HMA-Base MaskGIT | 离散 VQ | 44M | **22.72** |
| HMA-Base MAR | 软 token + diffusion head | 96M | 4.44 |
| HMA-XL MAR | 软 token + diffusion head | 741M | 2.01 |

22.72 FPS 意味着可以**实时闭环控制机器人**。一般机器人 control loop 10-20Hz 就够了。

---

## 关键 idea 3：Heterogeneous Pre-training（"异构"是个啥）

这是这 paper 第二聪明的点。

### 问题

你手上有 40 个机器人数据集，每个数据集的 action 都长得不一样：

- **Franka arm**：6 维 end-effector (位置 + 旋转)，5Hz
- **ALOHA 双臂**：14 维关节角，50Hz
- **Humanoid**：28 维关节角，30Hz
- **人手视频**（Epic-Kitchens）：2D 手部检测点，2 维，15Hz

如果你简单把它们 concat 起来训，模型会疯掉——它不知道 "0.5" 在 Franka 里是 5 厘米位移，在 humanoid 里是关节角度。

### 解法：Stem-Trunk-Head 架构（抄 HPT，https://arxiv.org/abs/2409.20537）

把网络拆成三段：

```
[Franka action 6D] ──→ [Franka Stem MLP] ──┐
[ALOHA action 14D] ─→ [ALOHA Stem MLP] ───┤
[Humanoid 28D] ────→ [Humanoid Stem MLP] ──┤
                                            │
                                  [Shared Trunk: 大 Transformer]
                                            │
[Franka Head] ──→ predict 6D action         │
[ALOHA Head] ──→ predict 14D action  ←──────┘
[Humanoid Head] → predict 28D action
```

- **Stem**：每个 embodiment 一个小 MLP，把任意维 action 映射到统一 latent（256 或 768 维）
- **Trunk**：所有 embodiment 共享的大 transformer，学"物理世界的通用 dynamics"
- **Head**：每个 embodiment 一个小 MLP，把 latent 映射回原 action space

训练时，每个 batch sample 根据 embodiment 激活对应的 stem 和 head，trunk 始终共享。

**好处**：
1. 添加新机器人只需训新 stem + head，trunk 冻结——迁移到新硬件成本极低
2. 不同 embodiment 之间 transfer knowledge：humanoid 学到的 "重力会让东西掉" 可以帮 Franka
3. 训练数据规模直接 ×40

**init 小 trick**：stem 用 Xavier init with gain=0.1。为什么？gain 太大会让 action signal 主导视频 token，transformer 早期就被 action "绑架"，学不到 visual dynamics。这是工程经验。

---

## 关键 idea 4：Modulation 让动作"渗透"每一层

这是另一个细节但很重要。

怎么把 action 信息 inject 到 video generation？最 naive 的做法是 token concatenation——把 action token 拼到 video token 后面。

Paper 做了 ablation：

| 方法 | 效果 |
|---|---|
| **Modulation (DiT-style adaLN)** | **最好** |
| Token concat | 差 |
| Feature addition | 差 |
| Cross-attention | 中等 |

为什么 modulation 赢？因为 token concat 只在 sequence 维度提供 action info，而 modulation 是**在每个 transformer block 的每一层 LayerNorm 里**用 action embedding 调制 scale 和 shift：

$$
\text{LN}(x) \cdot (1 + \gamma(a)) + \beta(a)
$$

- $x$：layer 输入特征
- $a$：action embedding
- $\gamma, \beta$：从 action 学到的 scale/shift 参数
- $1 + \gamma$：保证 identity init 时不会爆炸

这相当于 "action-conditioned normalization"——每一层都重新告诉网络"现在我们在执行这个动作"，而不是只在 input 层告诉一次。

跟你 Eureka Labs 讲的 "conditioning matters more than architecture" 完全对齐。

---

## 训练数据有多野

3 million trajectories，2.5 billion frames，来自 40 个 dataset：

- Open-X Embodiment（35 个真机数据集）
- Epic-Kitchens, Ego-Exo4D（人第一人称视角视频，用 2D 手部检测当 action proxy）
- Robomimic, MetaWorld（仿真基准）

采样策略：**inverse exponential probability**——大数据集被采到的概率反而不高，避免大数据集 dominate batch。这是长尾数据集训练的常用 trick。

所有数据统一到 **2Hz**：10Hz 的数据 stride=5，50Hz 的 stride=25。Context length 12 帧 = 6 秒 wall-clock，其中 4 帧 prompt + 8 帧 prediction。

---

## 推理时的"三层套娃"

推理时是三层 nested loop：

```python
for t in range(T):  # 视频时间维度，T=8
    for m in range(M):  # unmasking steps，M=2 就够
        for n in range(N):  # diffusion steps，N=100
            denoise current batch of soft tokens
        unmask a subset of tokens
    append predicted frame to context
```

关键 insight：**外层 T 必须迭代**（要 roll out 未来），但**每次只 denoise 当前帧的 token**，整个 context 不重新 denoise。

这就是为什么 HMA 比 IRASim 快 15×：IRASim 每生成一帧都要把整个 sequence 重新 denoise 100 次，HMA 只 denoise 当前帧。

另一个有意思的事：训练时只见过 12 帧 context，推理时能稳定 rollout **100+ 帧（30+ 秒）**。这暗示 transformer 学到的是 frame-to-frame transition 而不是 sequence pattern——跟 nanoGPT 在 text 上看到的 generalization 行为一致。

---

## 实验数据直觉解读

### Pre-training 真的有用

Language Table fine-tune 对比：

| | PSNR↑ | Perplexity↓ | ∆PSNR↑ |
|---|---|---|---|
| Scratch | 21.01 | 305.87 | 0.01 |
| Pre-trained | 22.04 | 189.83 | **0.06** |

∆PSNR 是 controllability metric：用真实 action vs 随机 action 预测出来的视频 PSNR 差多少。差越大说明模型越"听话"。

Pre-training 让 ∆PSNR 从 0.01 → 0.06，**6× 提升**。这说明 pre-training 不只是提升视觉质量，更关键是让模型真正"理解"action 和 video 的因果关系。

### Robustness 测试（Robomimic）

加 action perturbation 测：

| | Perplexity*↓ |
|---|---|
| Scratch | 1193.70 |
| Pre-trained | **103.01** |

**12× 更 robust**。Pre-trained 模型对 action 抖动不敏感，说明学到的是真实 dynamics 而非 memorize。

### 合成数据能训 policy

10 真实轨迹 + 90 合成轨迹：

| 配比 | Robomimic 成功率 |
|---|---|
| 10 real + 0 synthetic | 82% |
| 10 real + 50 synthetic | 96% |
| 10 real + 90 synthetic | **100%** |
| 100 real (baseline) | 100% |

**90% 合成数据能 match 100% 真实数据**。对 robotics 来说这是大事——意味着你可以"造数据"。

### Policy Evaluation 相关性

用 HMA 当 simulator evaluate 4 个 policy，跟 Mujoco 真值对比：

| | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| Mujoco | 0.38 | 0.52 | 0.70 | 1.00 |
| HMA | 0.43 | 0.56 | 0.66 | 0.73 |

Pearson correlation = **0.95**。这意味着 HMA simulator 可以用来 rank policy——做 RL、做 model-based planning 都打开了门。

---

## 这 paper 没解决什么（你应该会戳的点）

1. **2Hz 太慢**：真机闭环 control 一般要 20Hz+。虽然 inference FPS 22 够快，但 model 内部频率是 2Hz，需要 action chunk interpolation 才能上 real robot。
2. **256×256 分辨率**：fine-grained manipulation（插孔、拧螺丝）需要更高分辨率。
3. **没在真机闭环**：只 evaluate 了 video generation 和 offline policy eval，没 deploy 到真机做 closed-loop。
4. **Policy 性能一般**：直接当 policy 用效果不如 Diffusion Policy。作者自己承认可能是 VQ token 不适合 action space——action 应该用 diffusion，video 用 AR，二者 loss form 不统一。
5. **FID 反而比 IRASim 差**：33.56 vs 23.22。说明 pixel-level 更 sharp 但 distribution 多样性略低——AR 模型的通病。

---

## 你大概率会问的问题

**Q: 这跟 Decision Transformer (https://arxiv.org/abs/2106.01345) 啥区别？**

A: Decision Transformer 也是 sequence modeling，但只 predict action，不 predict observation——它是个 policy，不是 world model。HMA 同时 predict 两者，能当 simulator 用。

**Q: 跟 Sora 比呢？**

A: Sora 是纯 video generation，没 action conditioning。Sora 也用 diffusion（spatial-temporal DiT），不是 AR。Sora 不能当机器人 simulator 用——你没法告诉 Sora "机械臂往左 3 厘米"。

**Q: 跟 Genie (https://arxiv.org/abs/2401.14891) 比？**

A: Genie 也是 world model，但用 latent action（从视频里反推出来的 implicit action），不是真实低层 action。Genie 不能直接接受真机 action 输入。

**Q: 能不能在 YouTube 视频上 pre-train？**

A: 理论上可以，但 YouTube 视频没 action label。需要用 inverse dynamics 模式（从 $o_t, o_{t+1}$ 反推 $a_t$）或 latent action pretraining（LAPA, https://arxiv.org/abs/2410.11758）来利用无 label 视频。这是 obvious next step。

**Q: 跟你 Tesla AI Day 那套有啥相似？**

A: 你当时讲的是 "one neural network produces all outputs"——jointly predict lane, object, depth, etc. HMA 是同样思想：一个网络 jointly predict observation 和 action。区别是 Tesla 那套是 discriminative（输出 label），HMA 是 generative（输出 future）。

---

## 我的 take（如果你问我的话）

这 paper 最大的贡献**不是**某个具体技术（MAR、stem-trunk-head、modulation 都不是新东西），而是**把这些技术拼起来证明了一个 thesis**：

> **Robotics world model 可以用 autoregressive next-token prediction 做，可以跨 embodiment 共享 trunk，可以 real-time 跑，可以 generate synthetic data 直接训 policy。**

这跟你常说 "scale wins" 是一致的。Figure 6 的 scaling plot 显示三个 axis（数据集数、轨迹数、模型大小）都有 positive scaling 没 saturation——这是 foundation model 的特征。

如果让我下注，我会赌：
1. **半年内**会有人把 resolution 拉到 512，frequency 拉到 10Hz，真机闭环跑通
2. **一年内**会有人接 GPT-4V 当 auto-reward labeler，做 full closed-loop RL
3. **两年内**会有 "robotics GPT"——一个 trillion parameter 的 HMA 在 YouTube + Open-X + 所有仿真数据上 pre-train，stem/head 即插即用，下载就能用

你之前在 Eureka Labs 讲 "build a system, not just a model"，HMA 朝这个方向走了一大步。

---

## 想深入哪个方向？

1. **MAR 的 diffusion head 具体怎么实现**（3 层 MLP 怎么接 transformer latent）
2. **μP 怎么 transfer hyperparameter**（从小 model 调到大 model）
3. **Stem/Head 的 mixture-of-experts 思路**（跟 sparse MoE 的关系）
4. **为什么 forward dynamics 比 full dynamics 在 VQ setting 下表现更好**（Table 5a 那个反直觉结果）
5. **inverse dynamics 怎么给无 label 视频打 action label**（scale 到 YouTube 的关键）

Reference 链接都在上一条回复里，要我展开哪个方向都可以。

---

# HMA: Heterogeneous Masked Autoregression for Action-Video Dynamics

Karpathy 好，这篇 paper 我读完之后觉得非常对你的胃口——它本质上是把你之前在 nanoGPT 和 Eureka Labs 里讲的 "autoregressive next-token prediction" 思想，从 language domain 搬到 robotics 的 video+action dynamics 上。但它做得更妙：把 world model, policy, forward dynamics, inverse dynamics 全部塞进一个统一的 sequence modeling 框架里，用 masked autoregression (MAR) 同时获得 diffusion 的质量 和 autoregression 的速度。下面我尽可能把里面的技术细节拆开讲。

---

## 1. The Core Problem & Intuition

Robotics 有两个 long-standing bottleneck：

1. **Data scarcity**：NLP 有 internet，CV 有 internet，robotics 没有——每条 trajectory 都要真机采集。
2. **Evaluation cost**：要 evaluate policy 必须部署到真实机器人上，cost 巨大、不可批量。

一个非常自然的想法：**learn a generative world model**——给你一段 observation history 和 action sequence，它能 roll out future observations。这就同时解决了 data generation (infinite in-distribution data) 和 policy evaluation (跑 simulator 而不是真机)。

但问题是现有的 robotic video generation 方法（IRASim, Genie, GameNGen, DIFFUSER 等等）都用 **full-sequence diffusion**，每一步生成都要 iterate 整个 sequence 多次，real-time 不可达。IRASim 是 0.28 FPS，离实时差一个数量级。

HMA 的核心 insight：**用 masked autoregression 替代 full-sequence diffusion**。每个 frame/action token 只需要被预测一次（最多 unmask M=2 次），而 diffusion 需要 N=100 步去 denoise 整个 sequence。这就是 15× 加速的来源。

---

## 2. The Unification: Dynamics as Sequence Modeling

这篇 paper 最 elegant 的部分是把所有 robotics 问题统一成一个 sequence modeling objective。

定义 observation history $\mathcal{O}_{history} = \{o_{t-N_{past}}, ..., o_{t-1}\}$ 和 action history $\mathcal{A}_{history} = \{a_{t-N_{past}}, ..., a_{t-1}\}$，目标是预测 $\mathcal{O}_{future} = \{o_t, ..., o_{t+N_{future}}\}$ 和 $\mathcal{A}_{future} = \{a_t, ..., a_{t+N_{future}}\}$。

其中 $N_{past}$ 和 $N_{future}$ 是 hyper-parameters（paper 里设的是 2Hz 下 12 frame context = 6 seconds wall-clock，4 frame prompt + 8 frame prediction）。

这个 formulation 直接 generalize 出四种 robotics 问题（Figure 2）：

| Setting | Input → Output | 用途 |
|---|---|---|
| Full dynamics | $(\mathcal{O}_{hist}, \mathcal{A}_{hist}) \to (\mathcal{O}_{fut}, \mathcal{A}_{fut})$ | World model + policy 联合 |
| Forward dynamics | $(\mathcal{O}_{hist}, \mathcal{A}_{hist}) \to \mathcal{O}_{fut}$ | Video simulator / world model |
| Passive dynamics | $\mathcal{O}_{hist} \to \mathcal{O}_{fut}$ | 纯视频预测 (no action) |
| Policy | $(\mathcal{O}_{hist}, \mathcal{A}_{hist}) \to \mathcal{A}_{fut}$ | Imitation policy |
| Inverse dynamics | $(\mathcal{O}_{hist}, \mathcal{O}_{fut}) \to \mathcal{A}_{fut}$ | Relabeling |

这个 unification 的好处是巨大的：**一个模型一个 loss 一个 training loop，pre-training 之后通过不同 inference 路径就能 serve 不同 task**。这跟你之前在 Tesla AI Day 上讲的 "one neural network to rule them all" 思路是一致的。

---

## 3. Masked Autoregression: The Math

### 3.1 Joint Distribution 分解

把 $(\mathcal{O}_{history}, \mathcal{O}_{future}, \mathcal{A}_{history}, \mathcal{A}_{future})$ flatten 成 token sequence $X_1, ..., X_K$，autoregressive 分解为：

$$
p(X_1, ..., X_K) = \prod_{k=1}^{K} p(X_k | X_1, ..., X_{k-1})
$$

变量解释：
- $K$ 是总 token 数（video token + action token，paper 里每帧 256 video tokens + 64 repeated action tokens，所以一个 12-frame sequence 总共约 $(256+64)\times 12 \approx 3840$ tokens，patch size 2 之后约 960 tokens）
- $X_k$ 是任意 **causally valid masked set**——也就是 $X_k$ 可以是 observation token 也可以是 action token，关键是 order 可以随机（不是严格的 temporal causal），这就是 masked autoregression vs causal autoregression 的关键区别

### 3.2 两种 loss

**Discrete VQ variant (MaskGIT 风格)**：

$$
\mathcal{L}_{VQ}(X; \theta) = \text{MSE}(a, \hat{a}) + \text{CE}(o, \hat{o})
$$

变量：
- $a$ 是 action ground truth (continuous vector，比如 6-DoF end-effector 或 28-DoF humanoid)
- $\hat{a}$ 是模型预测的 action (回归出来)
- $o$ 是 observation 经过 VQ tokenizer (1XGPT / Open-MAGVIT2, 16×16 downsample) 得到的离散 token id
- $\hat{o}$ 是 transformer 输出的 logits over codebook
- $\theta$ 是模型参数

**Continuous soft-token variant (MAR 风格, Li et al. 2024)**：

$$
\mathcal{L}_{soft}(X; \theta) = \|\epsilon_a - f(a|t, z)\|^2 + \|\epsilon_o - f(o|t, z)\|^2
$$

变量：
- $\epsilon_a, \epsilon_o \sim \mathcal{N}(0, I)$ 是 Gaussian 噪声向量
- $t$ 是 diffusion 的 timestep（训练时 $N_{train}=1000$，推理时 $N_{test}=100$，用 DDIM）
- $z$ 是 continuous latent (所谓的 "soft token")，由 transformer trunk 产出
- $f(\cdot | t, z)$ 是 diffusion head (3-layer MLP)，给定 timestep $t$ 和 latent $z$，预测噪声
- action 和 video 的 $z, t$ 在实践中是分开的 (不共享)

这个公式其实就是 **score matching / DDPM** 形式，但关键是 $z$ 来自 transformer 的 latent 而不是 input image，所以这里等价于 **latent diffusion with autoregressive prior**。这跟 MAR paper (Tianhong Li & Kaiming He, https://arxiv.org/abs/2406.11838) 的 idea 一样：用 transformer 当 prior，用 diffusion head 当 decoder，避免 VQ bottleneck。

### 3.3 为什么比 diffusion 快

Full-sequence diffusion (IRASim, Sora, Stable Video Diffusion) 是：每生成一个 frame，整个 sequence 都要 denoise N=100 次，每次都要过整个 transformer。

MAR 是：unmasking steps M=2 (论文说 M=2 就够了)，每次 unmask 只需要 denoise 当前要预测的那一小批 tokens，而且 transformer forward 一次就能拿到所有 latent。

数学上：full diffusion 的 forward pass 数 $\propto N \times T$，MAR 是 $\propto M \times T \times (\text{head compute})$。Head 是 3-layer MLP，比 32-layer transformer 便宜得多。这就是 22.72 FPS (HMA-Base MaskGIT) vs 0.28 FPS (IRASim-XL) 的来源。

---

## 4. Architecture Deep Dive

### 4.1 整体设计 (Figure 3)

```
[Action a_i (embodiment-specific)] ─┐
[Video o_i (256 tokens per frame)] ─┤
                                    │
              stem (per-embodiment  │  ← Modulation
              MLP action encoder)   │
                                    │
                       ┌────────────▼────────────┐
                       │   Shared Trunk:         │
                       │   Spatial-Temporal      │
                       │   Transformer           │
                       │   (32 layers, d=256/768)│
                       └────────────┬────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │                                              │
        video tokens                                  action tokens
              │                                              │
       (VQ head or                            (3-layer diffusion MLP
       diffusion head)                              head, per-embodiment)
              │                                              │
              ▼                                              ▼
         predicted o_t                               predicted a_t
```

### 4.2 Stem/Trunk/Head 设计 (借鉴 HPT, https://arxiv.org/abs/2409.20537)

这是 Heterogeneous Pre-training 的精髓。不同 embodiment 的 action space 完全不同：
- Franka Arm: 6-DoF end-effector, 5Hz
- Humanoid: 28-DoF joints, 30Hz
- Aloha bimanual: 14-DoF, 10Hz
- 2D hand detection proxy (EgoCentric video): 2-DoF

如果直接 concatenate 所有 data 训练，action space heterogeneity 会 destroy 学习。

解决方法：
- **Stem**：每个 embodiment 一个 small MLP，把 normalized action 映射到 shared latent space (d=256/768)
- **Trunk**：所有 embodiment 共享的 spatial-temporal transformer
- **Head**：每个 embodiment 一个 3-layer diffusion MLP，把 latent 映射回 action space

这样：
- Pre-training 时只用 batch 内对应 embodiment 的 stem/head 激活
- Adding new embodiment 只需要 train 新的 stem + head，trunk freeze
- Stem init：Xavier with gain 0.1（这个很关键，gain 太大会让 action signal 主导 video signal，训练不稳）

### 4.3 Spatial-Temporal Attention

- **Spatial attention**：bidirectional (token 在空间维度全连接)，处理同一帧内 masked + unmasked tokens 的 interaction
- **Temporal attention**：causal，处理跨帧时间依赖，保证 future token 只看 past + current frame 的 spatial output

这就是 1xGPT (https://www.1x.tech/discover/1x-world-model-challenge) 用的设计，叫 "bidirectional spatial + causal temporal"。

### 4.4 Action Conditioning：为什么 Modulation 赢

Figure 5(b) 的 ablation 比较了四种 action-to-video fusion 方法：

| Method | 描述 | 表现 |
|---|---|---|
| **Modulation** | DiT-style FiLM/adaLN，action embedding 调制 LayerNorm 的 scale/shift | **最好** |
| Token concatenation | action token 直接拼到 video token 后面 | 较差 |
| Feature addition | action feature 直接加到 video feature 上 | 较差 |
| Token cross-attention | video token 用 action token 做 key/value | 中等 |

Intuition：token concat 只在 sequence 维度提供 information，per-layer modulation 是在每个 transformer block 的每一层都 inject action signal，相当于 "action-conditioned normalization"。

这跟你之前在 Eureka Labs 里强调的 "conditioning matters more than architecture" 完全一致——picking the right injection mechanism is half the battle.

---

## 5. Training & Inference Details

### 5.1 Training

- **Video tokenizer**：两个并行
  - 1XGPT (Open-MAGVIT2 fine-tuned), 16×16 spatial downsample → 用于 discrete VQ loss
  - Stable Video Diffusion VAE, 8×8 downsample → 用于 continuous soft token (隐式 latent)
  - Resolution: 256×256
- **Action frequency 统一**：所有 dataset 重采样到 2Hz (10Hz 数据 stride=5)
- **Masking schedule**：cosine schedule，越靠 future 越多 mask（因为 future 预测更难）
- **μP (Maximal Update Parametrization)**：用于 scaling，从 small model transfer hyperparameters 到 large model (https://arxiv.org/abs/2203.03466)
- **Diffusion head**：DDIM with per-step clipping, $N_{train}=1000, N_{test}=100$
- **Patch size 2**：2×2 patch token 合并成 1，context length 减半
- **Hardware**：8× V100, batch=64, 60k iters, 2 epochs；larger model 用 64 GPUs

### 5.2 Inference: 三层 nested autoregression

```
for t in T (video time dimension):
    for m in M (unmasking steps, M=2):
        for n in N (diffusion steps, N=100):
            denoise current batch of soft tokens
        unmask a subset of tokens (random order)
    append predicted frame to context
```

注意：
- $M=2$ 就够（论文 ablate 过），因为 transformer 已经把所有 context 融合好了
- $T$ 的迭代是必须的（要 roll out future），但**不需要 iterate 整个 sequence**——只 denoise 当前帧
- $N=100$ diffusion step 只对 soft token variant 需要；discrete VQ variant 完全跳过这一层

最大 rollout：**100+ frames (30+ seconds)**，而训练只见过 12-frame context。这个 generalization 暗示 transformer 学到的是 frame-to-frame transition dynamics 而不是 sequence-level pattern。这跟你之前反复讲的 "RNN-like generalization through autoregression" 一致。

---

## 6. Datasets & Scaling

### 6.1 Pre-training 数据

| Category | Datasets | 数量 |
|---|---|---|
| Real robot | Open-X Embodiment (https://arxiv.org/abs/2310.08864) | 35 datasets |
| Human egocentric video | EPIC-Kitchens, Ego-Exo4D, etc. | 3 datasets |
| Simulation | Robomimic, MetaWorld | 2 datasets |
| Total | | **40 datasets, 3M trajectories, 2.5B frames** |

Action space 范围：**2-DoF (2D hand) 到 28-DoF (humanoid)**，heterogeneity 巨大。

Sampling：inverse exponential probability，避免大数据集 dominate batch。

### 6.2 Scaling Laws (Figure 6)

三个 axis 都测了：

1. **Scaling #datasets** (5 → 40): perplexity 持续下降，∆PSNR 持续上升。**重要发现：增加 dataset heterogeneity 不会 degrade 单 dataset 表现**——transfer learning 是 positive 的。
2. **Scaling #trajectories per dataset** (10 → $10^6$): 10^5 之后 plateau。原因：数据不均衡，多出来的 trajectory 全来自几个大数据集。
3. **Scaling model size** (3M → 400M, hidden dim 256 → 768): perplexity 和 ∆PSNR 持续改善，没看到 saturation。

这跟 Chinchilla 风格的 scaling law 不太一样——这里没有看到 compute-optimal frontier，说明还能 scale。

---

## 7. 实验结果详解

### 7.1 Speed Comparison (Table 1)

| Model | Method | Params (M) | FPS |
|---|---|---|---|
| IRASim-XL | DiT | 679 | 0.28 |
| IRASim-XL, amortized | DiT | 679 | 0.58 |
| **HMA-Base** | **MaskGIT** | **44** | **22.72** |
| HMA-XL | MaskGIT | 679 | 4.38 |
| HMA-Base | MAR | 96 | 4.44 |
| HMA-XL | MAR | 741 | 2.01 |

关键观察：
- HMA-Base MaskGIT (44M) 在 22.72 FPS，**完全 real-time** (一般 robot control loop 10-20Hz)
- 同等参数下 (XL, 679M)，HMA 是 IRASim amortized 的 ~7× 快
- MAR (soft token + diffusion head) 比 MaskGIT 慢因为多了 N=100 diffusion step，但 visual quality 更好 (Figure 7)

### 7.2 Quality Comparison on Language Table (Table 2)

| | PSNR↑ | SSIM↑ | ∆PSNR↑ | LPIPS↓ | FID↓ | FVD↓ |
|---|---|---|---|---|---|---|
| IRASim | 25.41 | 0.82 | 5.78 | 0.08 | 23.22 | 152.20 |
| **HMA** | **28.19** | **0.83** | **6.06** | **0.07** | 33.56 | **111.52** |

注意 HMA 的 FID 反而更差 (33.56 vs 23.22)，但 PSNR/SSIM/LPIPS/FVD 都更好。FID 衡量 distribution-level quality，PSNR 是 pixel-level fidelity。我猜测 HMA 在 pixel reconstruction 上更 sharp 但 distribution 多样性略低——这是 autoregressive model 的常见 trade-off (vs diffusion 的 mode coverage)。

### 7.3 Pre-training Ablation (Table 3, 4)

**Real-world finetune (Language Table)**:

| | PSNR↑ | Perplexity↓ | ∆PSNR↑ | LPIPS↓ |
|---|---|---|---|---|
| HMA (scratch) | 21.01 | 305.87 | 0.01 | 0.19 |
| HMA+ (pretrained) | 22.04 | 189.83 | 0.06 | 0.17 |

Pre-training 帮 perplexity 改善 ~38%，∆PSNR 从 0.01 → 0.06 (6× controllability 提升！)。

**Simulation transfer (Robomimic)**:

| | PSNR↑ | Perplexity↓ | PSNR*↑ | Perplexity*↓ |
|---|---|---|---|---|
| HMA (scratch) | 24.17 | 20.69 | 19.19 | 1193.70 |
| HMA+ (pretrained) | 25.11 | 11.82 | 20.20 | **103.01** |

注意 PSNR* 和 Perplexity* 是加 action perturbation 测的 robustness。HMA+ 的 Perplexity* 是 103 vs scratch 的 1193——**12× 更 robust to action perturbation**。这暗示 pre-training 学到了真正的 action-conditioned dynamics 而不是 memorize trajectory。

### 7.4 Policy Evaluation (Table 5)

用 HMA 作 simulator evaluate 4 个不同 convergence level 的 Diffusion Policy：

| Evaluator | Policy 1 | Policy 2 | Policy 3 | Policy 4 |
|---|---|---|---|---|
| Ground Truth (Mujoco) | 0.38 | 0.52 | 0.70 | 1.00 |
| HMA Simulator | 0.43 | 0.56 | 0.66 | 0.73 |

**Pearson correlation = 0.95**。这其实非常强——意味着 HMA simulator 可以用来 rank policy，做 RL，做 model-based planning。

### 7.5 Synthetic Data Generation (Table 6)

10 真实 trajectory + 不同数量 HMA 生成 trajectory：

| | +0 | +10 | +50 | +90 | Full (100 real) |
|---|---|---|---|---|---|
| Robomimic success | 82% | 90% | 96% | **100%** | 100% |
| Language Table loss | 1.72 | 1.16 | 1.09 | **0.88** | 0.87 |

10 real + 90 synthetic = 100% success，几乎 match 100 real trajectory 的 baseline。这是 robotics synthetic data 的 milestone。

---

## 8. 关键 Limitations (Paper 自己承认的)

1. **Action prediction 比 video prediction 差**：当直接用 HMA 作 policy 时效果一般。Paper 假设是 VQ token 不适合 action prediction——visual token 和 action token 的 objective 不统一。这跟 Diffusion Policy 在 action 上 work 得很好的对比暗示：**action space 的 modeling 还是要 diffusion，video 可以 AR**。
2. **Controllability 仍然不完美**：100+ frame rollout 之后会有 compounding error (Figure 8)。
3. **Manual reward labeling**：policy evaluation 还要人去看视频标 success/fail。Future work 应该用 VLM (e.g., GPT-4V) 自动 label。

---

## 9. Related Work & 个人联想

这篇 paper 其实站在几个 trend 的交叉点：

1. **Video generation as world model**: Sora (https://openai.com/research/video-generation-models-as-world-simulators), Genie (https://arxiv.org/abs/2401.14891), GameNGen (https://arxiv.org/abs/2408.14837) — 但这些都没 handle low-level action conditioning。
2. **Robotics foundation model**: RT-X (https://arxiv.org/abs/2310.08864), Octo (https://arxiv.org/abs/2405.12213), HPT — 这些 focus 在 policy，没 model dynamics。
3. **Masked autoregression for vision**: MaskGIT (https://arxiv.org/abs/2202.04200), VAR (https://arxiv.org/abs/2404.02905), MAR (https://arxiv.org/abs/2406.11838), 1xGPT — HMA 是把这些方法首次系统 apply 到 robotics。
4. **Diffusion Policy** (https://arxiv.org/abs/2303.04137) — action 用 diffusion 是 SOTA，但 HMA 用 AR 也行，这块有意思。
5. **Decision Transformer** (https://arxiv.org/abs/2106.01345) — 把 RL 变成 sequence modeling，HMA 是它的 generative 版本。

如果让我推测 next steps：
- **VLM as reward labeler**: 把 GPT-4V 接到 HMA rollout 上做 auto-evaluation，就 close the loop on RL.
- **Latent action pretraining (LAP)**: https://arxiv.org/abs/2410.11758 在无 action label 视频上学 latent action，可以用来 scale pre-training 到 YouTube 数据。
- **Long-horizon planning**: 用 HMA 做 model predictive control (MPC)，rollout K 条 trajectory 取最优 action sequence。
- **Real robot deployment**: 作者在 limitation 里说没在 real robot 上 validate 当 policy 用，但 22.72 FPS 完全够 real-time。
- **Action tokenization**: 像 ALOHA 一样把 action 也 tokenize，统一 video 和 action 的 loss form，可能解决 Table 6 提到的 "policy 性能一般" 问题。

---

## 10. Build Intuition: 跟你之前的工作对比

Karpathy 你应该会喜欢这 paper 的几个地方：

1. **"One model to rule them all"**: Tesla AI Day 的 idea，HMA 在 academic scale 上验证了——world model + policy + simulator 共享一个 trunk。
2. **Autoregression as the universal primitive**: nanoGPT 里讲 "everything is next token prediction"，HMA 把它扩到 video + action。
3. **Modular conditioning**: 你在 Eureka Labs 强调过 conditioning 的重要性，HMA 的 modulation vs token concat ablation 验证这点。
4. **Scale + heterogeneity**: 跟你常说 "scale wins"，HMA 的 scaling plot (Figure 6) 显示三个 axis 都有 positive scaling，没 saturation——这是 foundation model 的特征。

潜在 criticism 你可能会 raise：
- 2Hz 太慢了，real robot control 至少要 20Hz (HMA real-time 是 22Hz FPS 但 model 是 2Hz 内部 frequency，需要 action chunk interpolation)
- 256×256 分辨率对 manipulation 不够 (fine-grained 3D 操作可能需要 512+)
- 没在 real robot 上 evaluate closed-loop policy (只是 video generation 和 offline policy eval)
- VQ tokenizer (16×16 downsample) 会丢 fine-grained 信息——你之前在 ImageGPT 时代就讨论过这个问题 (https://arxiv.org/abs/2012.09841)

---

## Reference Links

- Paper page: https://liruiw.github.io/hma
- HPT (predecessor): https://arxiv.org/abs/2409.20537
- MAR (Kaiming He): https://arxiv.org/abs/2406.11838
- 1XGPT: https://www.1x.tech/discover/1x-world-model-challenge
- Open-MAGVIT2 tokenizer: https://arxiv.org/abs/2409.04410
- Stable Video Diffusion VAE: https://arxiv.org/abs/2311.15127
- IRASim (baseline): https://arxiv.org/abs/2406.14540
- Genie (DeepMind): https://arxiv.org/abs/2401.14891
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Open-X Embodiment: https://arxiv.org/abs/2310.08864
- MaskGIT: https://arxiv.org/abs/2202.04200
- VAR (next-scale prediction): https://arxiv.org/abs/2404.02905
- Decision Transformer: https://arxiv.org/abs/2106.01345
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- μP (Maximal Update Parametrization): https://arxiv.org/abs/2203.03466
- Sora blog: https://openai.com/research/video-generation-models-as-world-simulators
- GameNGen: https://arxiv.org/abs/2408.14837
- Robomimic: https://arxiv.org/abs/2108.03298
- Language Table: https://arxiv.org/abs/2210.01711
- LAPA (Latent Action Pretraining): https://arxiv.org/abs/2410.11758
- Sora discussion by Karpathy: https://www.youtube.com/watch?v=gIkBFRBM0HI (relevant context)

---

要我 drill deep 到某个具体子方向吗？比如：
1. MAR vs MaskGIT 的 math 差异 (diffusion head 的设计)
2. Modulation layer 的实现细节 (adaLN 在 DiT 里怎么 work)
3. μP 在 HMA 里具体怎么 transfer hyperparameters
4. Pre-training → finetune 的 stem/head 选择机制
5. 为什么 forward dynamics > full dynamics (Table 5a) 在 VQ setting 下成立
