---
source_pdf: FromExpertstoaGeneralist.pdf
paper_sha256: e8174008815d00c53c87e25c06e538d581738bae527404352cfc7b22d3efdea1
processed_at: '2026-08-04T11:12:01-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 BumbleBee

好嘞 Andrej，我换个角度，用更直观的方式讲。

## 一句话总结

**一个人脑子学不会所有动作，那就先分科训练专家，再让一个通才把所有专家的本事学过来。**

---

## 问题在哪

想象你训练一个机器人跳舞，它要会 jump、walk、boxing、Charleston dance 这些完全不同的动作。你把这些动作全扔给一个 policy 学，结果就是：

- Jump 要猛踩油门（高 torque）
- Standing 要稳如老狗（保持平衡）
- Walk 要流畅转身

这三种要求**互相矛盾**。policy 在 walk 上学得平稳了，jump 就跳不起来；在 jump 上学得凶猛了，walk 就不稳。gradient 之间打架，谁都学不好。

这就像让一个人同时学短跑、太极拳、举重，三者的发力方式、呼吸节奏、肌肉记忆完全不同，硬塞在一起会四不像。

---

## BumbleBee 的解法：三步走

### 第一步：给动作分类

不是随便分，是先用 autoencoder 学每个动作的"特征指纹"，再聚类。

**为什么不用 joint angles 直接分？**
joint angles 看不出动态。比如快走和慢走，腿的关节角度可能差不多，但脚的 speed 差很多。所以 paper 用 forward kinematics 把 joint angles 转成 world frame 的 3D position，再加上 foot velocity——这样 jump（脚离地快）和 stand（脚不动）就能分开。

**为什么要加 text？**
光看 motion pattern 不够。比如走路可以直走也可以绕圈走，kinematic 上差别大，但语义都是"walking"。引入 HumanML3D 的 text label，把 "a person walks" 这种语义信息和 motion embedding 对齐，就能让语义相似的 motion 在 latent space 靠近。

最后用 K-means 聚出 6 类，Elbow Method 确认 K=6 是甜点：

| Cluster | 代表动作 | 特征 |
|---------|----------|------|
| Jump | 跳跃 | Z 方向位移大 (9.21mm) |
| Walk-slow | 慢走/小跑 | 速度 0.353 m/s |
| Walk-fast | 快走 | 速度 0.429 m/s |
| Stand-up | 上半身动作 | 主要动手 |
| Stand-mid | 中等站立 | 介于上下 |
| Stand-low | 蹲/低位站立 | 主要动腿 |

---

### 第二步：每类训一个专家

先在 full dataset 上训一个 general base model（保证 expert 不会忘掉其他动作），然后每个 expert 从 base model 出发，在自己 cluster 上 fine-tune。

**关键 trick**：专家不从头训，而是继承 base model 的"常识"。这样 jump 专家虽然主攻 jump，遇到 walk 时也不会完全懵——它保留了 general 的 tracking 能力，只是更擅长 jump。

**Sim-to-Real 的处理**：
ASAP 的思路是学一个 delta action model 来补 sim 和 real 的 gap。BB 的改进是**每个 cluster 训一个 delta model**。

为什么？看 Table 6 的数据：

| Cluster | Ankle Yaw Delta |
|---------|-----------------|
| Walk-fast | 0.3399 |
| Stand-up | 0.1556 |

Walk-fast 的 delta 是 Stand-up 的两倍多。如果用统一 delta model，它会被两种 distribution 拉扯——Table 5 显示在 Jump cluster 上，general delta model 反而让性能下降了（59.64% → 50.71%），而 cluster-specific delta 大幅提升（59.64% → 68.92%）。

**Iterative refinement**：训一轮 delta，改 sim，再 fine-tune tracking policy，再 deploy 收集更好的 data，再训更好的 delta…… Table 4 显示 SR 从 51.49% → 60.33% → 70.37%，每轮都在涨。

---

### 第三步：蒸馏成通才

6 个专家训好了，怎么合成一个？

用 DAgger 做 knowledge distillation：

$$\mathcal{L}_{\text{distil}} = \mathbb{E}_{s \sim \mathcal{D}}\left[\text{KL}\left(p_{\text{general}}(a|s) \| p_{\text{expert}, k(s)}(a|s)\right)\right]$$

人话翻译：
- $s$：当前 state
- $k(s)$：这个 state 属于哪个 cluster（路由到哪个专家）
- $p_{\text{expert}, k(s)}(a|s)$：对应专家会怎么动
- $p_{\text{general}}(a|s)$：通才会怎么动
- Loss 目标：让通才的行为概率分布逼近对应专家

**为什么不用 MLP？**
3 层 MLP 容量不够装下 6 个专家的知识。用 Transformer（Gated Transformer-XL，10 步历史，6 个 attention head），因为 self-attention 天然适合"根据当前 state 找到该用哪个专家的知识"——这和 LLM 里 MoE + Transformer 的组合是同一个道理。

---

## 一个有意思的发现

Figure 4 显示：**最终 generalist 在 Jump 和 Walk-slow 上比原专家还好**。

按理说 distillation 有 loss，通才应该比专家差才对。但 paper 发现通才从多个专家继承了 stable control behavior，cross-cluster 知识反而让它在 challenging motion 上更鲁棒。

这就像一个武术家学了拳击、跆拳道、柔道，虽然单项不如专精选手，但综合应变能力更强——遇到复杂场景时能融合多种技巧。

---

## 为什么这个思路 work

Table 3 的 ablation 说了大实话：

| 方案 | MuJoCo SR |
|------|-----------|
| 不分专家直接训 | 33.01% |
| 随机分 6 类训专家再蒸馏 | 35.36% |
| **AE 聚类训专家再蒸馏** | **66.84%** |

随机分类几乎没用（33% → 35%），AE 聚类翻倍（33% → 66%）。**分类质量决定一切**——如果 cluster 内还有矛盾动作，专家也学不好，蒸馏也没用。

---

## 实际部署效果

Figure 1 的 demo：Unitree G1 机器人连续做 boxing + Charleston dance，总共 135 秒，稳定完成。这在以前需要切换两个不同 policy 才能做到，现在一个 generalist policy 搞定。

Figure 5 的 stand-low 可视化更直观：
- Iter 0：脚落地不稳，直接摔倒
- Iter 1：能落地但脚抖
- Iter 2：平滑稳定跟踪

---

## 对你 Andrej 的延伸联想

这个 paper 让我想到几个更广的话题：

**(a) Hierarchical RL 的复兴**：早期 hierarchical RL（options framework, MAXQ）想手动设计 hierarchy，效果一般。BB 用 representation learning 自动发现 hierarchy（通过 AE clustering），比手动设计 task decomposition 更 data-driven。这和 LLM 里 emergent capability 的思路一致——让模型自己发现结构。

**(b) 和 LLM distillation 的平行**：BB 的 expert-to-generalist 和 LLM 里 teacher-student distillation 是同构问题。OpenAI 的 process reward model、Anthropic 的 constitutional AI 都在用类似的"先分专家再融合"思路。robot control 的特殊性在于 action space 是 continuous 的，distillation loss 要用 KL over Gaussian policy（如果用 stochastic policy）或 MSE over deterministic action。

**(c) Sim-to-Real as Domain Adaptation**：delta action model 本质是 learned domain adaptation。传统 domain randomization 是"让 policy 对所有 domain 鲁棒"，delta action 是"学习 domain 之间的差异并补偿"。BB 把这个思路推到 extreme——每个 motion cluster 有自己的 domain shift pattern，所以要 cluster-specific delta model。这和 computer vision 里的 per-domain adaptation 是一个思路。

**(d) 为什么不用 online MoE？**：BB 是 offline clustering + distillation，而不是 online routing-based MoE（每步决定用哪个 expert）。原因可能是 continuous action space 的 router 很难训——discrete token 上 routing 有 well-defined softmax，continuous action 上 router 的 gradient 不稳定。这是一个 open problem，值得探索。

参考链接：
- ASAP: https://arxiv.org/abs/2502.01143
- BumbleBee 项目页: https://beingbeyond.github.io/BumbleBee/
- HumanML3D: https://github.com/EricGuo5513/HumanML3D
- DAgger: https://arxiv.org/abs/1011.0686
- Gated Transformer-XL repo: https://github.com/datvodinh/ppo-transformer

---

## 最最简化版

1. **问题**：一个 policy 学不会所有动作，因为动作之间互相矛盾
2. **思路**：先把动作分类（用 AE + text），每类训专家（cluster-specific delta 解决 sim-to-real），再把专家蒸馏成通才（Transformer 容量大）
3. **关键**：分类要合理（AE clustering 比随机分好 30%+），delta 要分 cluster（统一 delta 在 jump 上反而变差），蒸馏要够容量的 model（MLP 不够用 Transformer）
4. **结果**：MuJoCo SR 从 baseline 最高 50% 提到 66.84%，real robot 上能连续做 135 秒混合动作

核心 philosophy：**与其让一个大脑硬学所有矛盾，不如先分而治之，再融合智慧**。

---

# BumbleBee: 从 Experts 到 Generalist 的人形机器人全身控制

你好 Andrej，这篇 paper 来自 Peking University 和 BeingBeyond，解决了一个人形机器人 control 领域非常实际的问题：**如何在单一 policy 中处理高度多样化的 whole-body motion**。让我用尽量详细的技术视角来讲。

## 1. 核心问题的 Intuition

现有的 humanoid whole-body control 框架（如 ASAP [9], OmniH2O [26], HumanPlus [28], Hover [29]）在单个 motion 上能训到极好的 performance，但当我们想让一个 policy 同时处理 jump、walk、stand、dance 这类差异巨大的动作时，会遇到两个根本性的问题：

**(a) Mismatched Data Distributions**: jump 这种动作要求高 torque 精确控制、剧烈的 COM 轨迹，而 in-place standing 要求平滑稳定。把它们放在同一个 replay buffer 里训练，gradient 会互相打架——也就是 paper 里说的 **conflicting gradients**。

**(b) Sim-to-Real Gap 因 motion 而异**: 一个统一的 delta action model 难以同时适配 jump（大 amplitude、接触变化剧烈）和 stand（基本静态）。Table 6 显示 ankle joint 的 delta magnitude 在不同 cluster 上差很多：Walk-fast 的 yaw delta = 0.3399，而 Stand-up 只有 0.1556。

**核心 insight**：Mixture of Experts (MoE) [15, 16] 在 LLM 里证明有效，因为不同 token 路由到不同 expert 可以避免知识干扰。在 robot control 里这个思路同样适用——先把 motion 聚类，在每个 cluster 上训 expert，再 distill 成 generalist。这是 paper 的主线 motivation。

参考链接:
- ASAP paper: https://arxiv.org/abs/2502.01143
- OmniH2O: https://humanoidhumanoid.com/
- Hover: https://hover-versatile-humanoid.github.io/
- MoE 原始 paper: https://arxiv.org/abs/1701.06538

---

## 2. Pipeline 总览（Figure 2 解析）

整个 BumbleBee framework 分四个阶段：

```
[AMASS Motions] 
    ↓ Motion Retargeting (SMPL → robot joint angles)
    ↓ PHC Filtering → 8179 高质量 trajectories
[Filtered Dataset]
    ↓
[AE Clustering] ← 输入: motion kinematic features + text annotations
    ↓ 输出: 6 个 cluster (Jump, Walk-slow, Walk-fast, Stand-up, Stand-mid, Stand-low)
[General Tracking Policy (Base Model)] ← 在 full dataset 上训练
    ↓ Fine-tune 到每个 cluster
[6 Expert Tracking Policies]
    ↓ 部署到 real robot，collect trajectories
[6 Expert Delta Action Models] ← 每个 cluster 一个
    ↓ Iterative refinement (deploy → train delta → fine-tune tracking)
[Refined Experts]
    ↓ DAgger Knowledge Distillation
[Generalist Policy (Transformer backbone)]
```

---

## 3. AE Clustering 的技术细节

这一步是整个 framework 的基础，clustering 的质量直接决定 expert 的 specialization 效果。

### 3.1 输入表示的选择

paper 不直接用 SMPL 的 joint angles 做 clustering，而是先做 forward kinematics 转成 world frame 的 3D coordinates，并额外加入 foot velocity。原因很直觉：SMPL 只包含 joint angles 和 root transformation，**丢失了 kinematic dynamics**。比如 fast walking 和 slow walking 的 joint angle 序列可能在统计上接近，但 foot 在 world frame 下的 velocity 差异显著。

motion sequence 输入：
$$\mathcal{M}_{\text{full}} = \{\mathbf{p}_t, \mathbf{r}_t, \dot{\mathbf{r}}_t, \mathbf{c}_t, \mathbf{v}_t^{\text{feet}}\}_{t=1}^{T}$$

- $\mathbf{p}_t \in \mathbb{R}^{N \times 3}$: $N$ 个 joint 的 3D position（world frame）
- $\mathbf{r}_t \in \mathbb{R}^3$: root 的 3D translation
- $\dot{\mathbf{r}}_t \in \mathbb{R}^3$: root 的 3D velocity
- $\mathbf{c}_t \in \{0, 1\}^F$: $F$ 只脚的 binary contact state（0/1）
- $\mathbf{v}_t^{\text{feet}} \in \mathbb{R}^{F \times 3}$: $F$ 只脚的 3D velocity

加入 $\mathbf{v}_t^{\text{feet}}$ 是关键设计——leg dynamics 是区分 jump / walk / stand 的关键 factor。这一点在 Table 2 的统计上有明显体现：Jump cluster 的 Z-Move = 9.21mm，远高于其他 cluster。

### 3.2 双模态 Autoencoder 架构

paper 借鉴了 TMR [39] 和 TEMOS [40] 的思路，把 motion 和 text 对齐到同一个 latent space：

**Motion branch**:
- Transformer encoder 处理 motion sequence $\mathcal{M}_{\text{full}}$ → latent $z^m$
- 注意：encoder 的输入维度比 decoder 输出大，因为输入包含所有 joint 的 $\mathbf{p}_t$、root 信息、contact、foot velocity，而 decoder 只重建 key joints（head, pelvis, hands, feet）

**Text branch**:
- Text label $l$ 先经过预训练的 BERT [41] 序列化
- 再过一个 Transformer → latent $z^l$
- $z^l$ 和 $z^m$ 维度对齐

**Loss function**:
$$\mathcal{L}_{\text{cluster}} = \mathcal{L}_{\text{InfoNCE}}(z^l, z^m) + \mathcal{L}_2(z^l, z^m) + \mathcal{L}_{\text{huber}}(\hat{M}^l, M) + \mathcal{L}_{\text{huber}}(\hat{M}^m, M)$$

逐项解释：
- $\mathcal{L}_{\text{InfoNCE}}(z^l, z^m)$: InfoNCE contrastive loss，对齐 text 和 motion 的 latent space。形式为 $-\log \frac{\exp(z^l \cdot z^m / \tau)}{\sum_{j} \exp(z^l \cdot z^m_j / \tau)}$，其中 $\tau$ 是 temperature。这让同一 motion 的 text 和 motion embedding 在 latent space 接近，不同 motion 的远离。
- $\mathcal{L}_2(z^l, z^m)$: 直接的 L2 距离，加强 alignment
- $\mathcal{L}_{\text{huber}}(\hat{M}^l, M)$: Huber loss 重建 key joints features，从 text latent $z^l$ 解码
- $\mathcal{L}_{\text{huber}}(\hat{M}^m, M)$: 同上，从 motion latent $z^m$ 解码

Huber loss 形式：$\mathcal{L}_{\text{huber}}(x, y) = \begin{cases} \frac{1}{2}(x-y)^2 & |x-y| \leq \delta \\ \delta(|x-y| - \frac{1}{2}\delta) & |x-y| > \delta \end{cases}$，比 L2 对 outlier 鲁棒。

### 3.3 为什么需要 text alignment？

一个直觉的解释：walking 可能有 linear path 和 circular path 两种形态，纯 motion alignment 难以发现它们语义相同。引入 text alignment（来自 HumanML3D [38]）后，"a person walks" 的 text embedding 会拉拢两种 walking pattern 在 latent space 接近。

### 3.4 K-means + Elbow Method

训练完 autoencoder，用 motion encoder 输出所有 motion 的 latent $z^m$，再用 K-means 聚类。Elbow Method 看的 metric 是 within-cluster sum of squares (WCSS):

$$\text{WCSS}(K) = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

其中 $\mu_k$ 是 cluster $C_k$ 的中心。Figure 3 显示 K=6 是拐点。最终 6 个 cluster 的统计（Table 2）非常 reasonable：

| Cluster | Disp (m) | Z-Move (mm) | Speed (m/s) | Keywords |
|---------|----------|-------------|-------------|----------|
| Jump | 2.32 | 9.21 | 0.329 | jumps, jumping |
| Walk-slow | 3.42 | 1.60 | 0.353 | jogs, runs |
| Walk-fast | 3.24 | 5.63 | 0.429 | walks, forward |
| Stand-up | 0.89 | 0.68 | 0.061 | something, hand |
| Stand-mid | 1.33 | 0.82 | 0.119 | arms, hand |
| Stand-low | 1.84 | 1.52 | 0.148 | foot, leg |

kinematic features 和 semantic keywords 高度一致，说明 clustering 既在 motion space 有意义，又在 semantic space 有意义。

参考链接:
- TMR: https://arxiv.org/abs/2303.12305
- TEMOS: https://arxiv.org/abs/2204.09397
- HumanML3D: https://github.com/EricGuo5513/HumanML3D
- InfoNCE 原始 paper (CPC): https://arxiv.org/abs/1807.03748

---

## 4. Expert Policy 训练

### 4.1 Base Model + Fine-tuning Strategy

关键设计：**不从头训练每个 expert**。先在 full dataset 上训一个 general tracking policy 作为 base model，然后每个 expert 从 base model fine-tune 到对应 cluster。

intuition：base model 学到了通用的 motion tracking 能力，expert 在它基础上 specialization，既获得 cluster-specific 性能，又保留对其他 motion 的 generalization（防止 catastrophic forgetting）。Figure 4 验证了这一点——expert 在自己 cluster 上 SR 远高于 General Init，但在其他 cluster 上仍有一定 SR。

### 4.2 Observation 设计

paper 在 Appendix A.1 给了详细说明：

**Privileged observation (teacher)**:
- Proprioception: linear velocity, angular velocity, joint position, joint velocity, last action
- Task-relevant: target joint positions, target keypoint positions, target root translations, target root rotations（global coordinates）
- 5 timesteps 历史

**Student observation**:
- Proprioception（去掉 linear velocity）
- Task-relevant: target joint positions, root translation, root rotations（local coordinates）
- 10 timesteps 历史

这种 teacher-student asymmetric observation 是 sim-to-real 的标准做法——teacher 用 privileged info 训练，student 用 onboard sensor 能测的 info。

### 4.3 Reward Design (Table 7)

reward 分三类：

**Penalty**（权重 -10）:
- Torque limits: $\mathbf{1}(\tau_t \notin [\tau_{\min}, \tau_{\max}])$
- DoF position limits: $\mathbf{1}(d_t \notin [q_{\min}, q_{\max}])$
- DoF velocity limits: $\mathbf{1}(\dot{d}_t \notin [\dot{q}_{\min}, \dot{q}_{\max}])$

**Regularization**（防止 action jitter）:
- DoF acceleration: $\|\ddot{d}_t\|_2^2$ (weight $-3 \times 10^{-8}$)
- Action rate: $\|a_t - a_{t-1}\|_2^2$ (weight -2)
- Action smoothness: $\|\dot{a}_t - \dot{a}_{t-1}\|_2^2$ (weight -2)
- Torque: $\|\tau_t\|$ (weight -0.0001)
- Stumble: $\mathbf{1}(F_{\text{feet}}^{xy} > 5 \times F_{\text{feet}^{z})$ (weight -0.00125)
- Feet orientation: $\sum_{\text{feet}} \|\text{gravity}_{xy}\|$ (weight -2.0)

**Task reward**（核心，全部用 $\exp(-4 \cdot \|e\|)$ 形式）:
- Body position: $\exp(-4 \cdot \|\hat{p}_{\text{body}} - p_{\text{ref}}\|)$, weight 1.0
- Root rotation: $\exp(-4 \cdot \|q_{\text{root}} - q_{\text{ref}}\|)$, weight 0.5
- Root angular velocity: $\exp(-4 \cdot \|\omega_{\text{body}} - \omega_{\text{ref}}\|)$, weight 0.5
- Root velocity: $\exp(-4 \cdot \|v_{\text{root}} - v_{\text{ref}}\|)$, weight 0.5
- DoF position: $\exp(-4 \cdot \|d - d_{\text{ref}}\|)$, weight 0.5
- DoF velocity: $\exp(-4 \cdot \|\dot{d} - \dot{d}_{\text{ref}}\|)$, weight 0.5

用 $\exp(-4 \cdot \|e\|)$ 而不是直接 L2 的 intuition：exponential kernel 给 small error 高 reward，large error 迅速衰减到 0，避免 policy 在 large error 时无差别梯度。这是 humanoid RL 的标准 trick，最早在 DeepMimic 里见到。

### 4.4 PD Control Action

policy 输出的是 PD controller 的 target joint position：

$$\tau_t = K_p (a_t - q_t) - K_d \dot{q}_t$$

其中 $a_t$ 是 policy 输出，$q_t$ 是当前 joint position，$K_p, K_d$ 是 PD gain（在 domain randomization 中也会被随机化）。低层 200Hz PD 控制，policy 50Hz 输出。

---

## 5. Multi-Stage Delta Action Training

这是 paper 的核心创新点之一，在 ASAP 基础上做了 cluster-specific 改进。

### 5.1 Delta Action 的数学形式

ASAP [9] 的核心 idea：simulator 的 transition function $f^{\text{sim}}(s_t, a_t)$ 和 real-world $f^{\text{real}}(s_t, a_t)$ 之间有 gap。与其精确建模 gap，不如学一个 residual action $\pi^\Delta(s_t, a_t)$ 加到 action 上：

$$s_{t+1} = f^{\text{sim}}(s_t, a_t + \pi^\Delta(s_t, a_t))$$

intuition：如果 sim 预测的下一状态比 real 滞后，$\pi^\Delta$ 学会"提前"加 action；如果 sim 的 contact 摩擦力太小，$\pi^\Delta$ 学会"补偿"额外 friction。这是一种 model-based residual learning 的思路，类似于 control literature 里的 additive control。

### 5.2 Real-world Data Collection

对每个 expert：
- 随机采样 20 个 deployable motions
- 每个 motion 在 real robot 上执行 8 次 rollout
- 平均每个 motion 8 秒
- 每个 cluster 收集到上百条 trajectory

数据记录：
- $v_t^{\text{base}} \in \mathbb{R}^3$: base linear velocity
- $\alpha_t^{\text{base}} \in \mathbb{R}^4$: base orientation (quaternion)
- $\omega_t^{\text{base}} \in \mathbb{R}^3$: base angular velocity
- $q_t \in \mathbb{R}^{23}$: joint positions
- $\dot{q}_t \in \mathbb{R}^{23}$: joint velocities

注意 paper 不用 motion capture 系统，依赖 odometry，所以 reward 用 root translation 而非所有 joint positions（Table 8）。

### 5.3 Cluster-Specific Delta Model 的优势

Table 5 的 ablation 直接对比：
- General delta model（用所有 real data 训一个）vs Cluster-specific delta model

| Cluster | Expert Init | Expert Gen Final | Expert Final |
|---------|-------------|------------------|---------------|
| Jump | 59.64% | 50.71% | 68.92% |
| Stand-up | 64.11% | 75.85% | 77.32% |
| Walk-slow | 15.71% | 42.32% | 56.50% |

Jump cluster 上 general delta model 反而**降低**了性能（59.64% → 50.71%），而 cluster-specific 大幅提升（59.64% → 68.92%）。

为什么？Table 6 给出答案：不同 cluster 的 delta magnitude 差异巨大。Walk-fast 的 ankle yaw delta = 0.3399，Stand-up 只有 0.1556。一个统一 model 试图同时拟合两种 distribution，会被分布偏移的数据"拉扯"。

### 5.4 Iterative Refinement

paper 做了 2 轮迭代：

```
Iter 0: Expert tracking policy → Deploy → Collect data → Train delta
        ↓ Use delta to modify sim → Fine-tune tracking policy
Iter 1: Better tracking policy → Deploy → Collect better data → Better delta
        ↓ 
Iter 2: Even better tracking → ...
```

intuition 是 bootstrapping：更好的 tracking policy 能 collect 到更高质量、覆盖更广的 real-world trajectory，反过来训练出更准的 delta model，再反过来 fine-tune tracking policy。

Table 4 的 mean SR 印证了这个 positive feedback loop：
- Iter 0: 51.49%
- Iter 1: 60.33%
- Iter 2: 70.37%

Figure 5 的 stand-low 可视化更直观：Iter 0 落地不稳，Iter 1 改善但脚抖，Iter 2 平滑稳定。

---

## 6. Generalist Distillation

### 6.1 DAgger-based Distillation

paper 用 DAgger [42] 把多个 expert 蒸馏成一个 generalist。核心 loss：

$$\mathcal{L}_{\text{distil}} = \mathbb{E}_{s \sim \mathcal{D}}\left[\text{KL}\left(p_{\text{general}}(a|s) \| p_{\text{expert}, k(s)}(a|s)\right)\right]$$

- $s$: 从 dataset $\mathcal{D}$ 采样的 state
- $k(s)$: state $s$ 对应的 expert index（通过 cluster assignment 确定）
- $p_{\text{expert}, k(s)}(a|s)$: 对应 expert 的 action distribution
- $p_{\text{general}}(a|s)$: generalist 要学的 distribution

KL divergence: $\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$

注意这里 KL 方向是 $p_{\text{general}} \| p_{\text{expert}}$（forward KL），即让 generalist 模仿 expert 的 mode-covering behavior。

### 6.2 为什么用 Transformer 而非 MLP?

paper 明确指出 3-layer MLP（hidden sizes 1024-1024-512, ELU activation）容量不足以融合多 expert 行为。他们用 Gated Transformer-XL：

- Input: 10 个 consecutive observations
- 1 个 Transformer block
- 6 个 attention heads
- Hidden size 128
- Embedding dimension 204
- Memory length 10

intuition：MoE 的融合需要 model 有 capacity 同时存储多种 "mode" 的行为。Transformer 的 self-attention 天然适合——不同 observation token 可以 attend 到不同 expert 的"知识区域"。这和 LLM 里 mixture of experts + Transformer 的组合是同源思路。

参考链接:
- DAgger: https://arxiv.org/abs/1011.0686
- Gated Transformer-XL: https://github.com/datvodinh/ppo-transformer
- Transformer-XL: https://arxiv.org/abs/1901.02860

---

## 7. 实验结果分析

### 7.1 主实验（Table 1）

BB 在 IsaacGym 和 MuJoCo 上的对比：

| Method | IsaacGym SR | MuJoCo SR | MuJoCo MPJPE | MuJoCo MPKPE |
|--------|-------------|-----------|--------------|--------------|
| OmniH2O | 85.65% | 15.64% | 0.4601 | 360.96 |
| Exbody2 | 86.63% | 50.19% | 0.3576 | 272.42 |
| Hover | 63.21% | 16.12% | 0.3428 | 323.08 |
| **BB** | **89.58%** | **66.84%** | **0.2356** | **294.27** |

关键观察：**IsaacGym 上 BB 优势小，MuJoCo 上优势巨大**。

intuition：IsaacGym 是训练环境，policy 对它 overfit。MuJoCo 作为 unseen simulator，测试的是真正的 generalization。BB 在 MuJoCo 上 66.84% vs 第二名 Exbody2 的 50.19%，差距 16.65%，说明 expert-to-generalist pipeline 学到的是更 robust 的 control 策略，而非单纯 overfit 训练环境。

### 7.2 Clustering 价值（Table 3）

| Setting | IsaacGym | MuJoCo |
|---------|----------|--------|
| General Init (无 expert) | 88.69% | 33.01% |
| Random (随机聚类) | 86.25% | 35.36% |
| **BB (AE clustering)** | **89.58%** | **66.84%** |

Random clustering 在 MuJoCo 上几乎没提升（33.01% → 35.36%），但 AE clustering 提升一倍（33.01% → 66.84%）。说明**聚类质量是关键**，单纯数据 partitioning 没用——如果 cluster 内仍有 conflicting motions，expert 还是学不好。

### 7.3 Expert vs Generalist（Figure 4）

一个有趣发现：**最终 generalist 在某些 cluster 上比原 expert 还好**！

比如 Jump 和 Walk-slow cluster，generalist SR > expert SR。

paper 的解释：generalist 从多个 expert 继承了 stable control behavior，cross-cluster knowledge 让它在 challenging motion 上更鲁棒。这和 LLM 里 distillation + multi-task learning 提升 single-task 性能的现象一致。

### 7.4 Statistical Significance（Table 12）

paper 提供了 confidence interval：
- BB MuJoCo SR: 66.84% ± 1.262%
- Exbody2 MuJoCo SR: 50.19% ± 1.517%

不重叠，统计显著。

---

## 8. Limitations 和我的思考

paper 自己提到的 limitation：
1. 没有 GPS 或 VIO [44]，缺少 global positioning，可能引入 reference alignment bias
2. Pipeline 复杂度高，scalability 受限

我自己的额外思考：

**(a) Cluster 数量是固定的**: K=6 是用 Elbow Method 在当前数据集上确定的，但 AMASS 数据可能在更细粒度上还有 sub-cluster（比如 dance 内部有不同 style）。一个可能改进是用 hierarchical clustering 或 soft MoE，让 cluster 边界更柔。

**(b) Text annotation 依赖**: clustering 依赖 HumanML3D 的 text label，但不是所有 AMASS motion 都有 text。如果扩展到没标注的 motion dataset（比如 lab mocap data），AE clustering 可能退化。

**(c) Real-world iteration 成本**: Iter 0 → Iter 1 → Iter 2 需要在 real robot 上多次 deploy，每次都要 collect 几百条 trajectory。硬件磨损、操作风险都很高。能否用 learned dynamics model 替代部分 real-world iteration？

**(d) Generalist 在 Out-of-Distribution motion 上表现如何**: paper 评估的都是 AMASS 内的 motion，如果让 generalist 跟踪一个 training 时没见过的全新 action（比如 martial arts kick），泛化性如何？这是真正 "general" WBC 的终极 test。

**(e) 和 Humanoid Locomotion as Next Token Prediction [13] 的关系**: Radosavovic 等人的工作用 Transformer 直接 predict next action，把 locomotion 当 autoregressive problem。BB 的 generalist 也是 Transformer backbone，但 input/output 设计完全不同。两者能否结合？比如用 next-token prediction 作为 pretraining，再用 BB 的 expert-to-generalist pipeline 做 fine-tune。

参考链接:
- Humanoid Locomotion as Next Token Prediction: https://arxiv.org/abs/2411.14426
- AMASS: https://amass.is.tue.mpg.de/
- Exbody2: https://github.com/chengxuxin/Exbody2

---

## 9. 总结：BB 的核心贡献

1. **AE-based motion clustering** that combines kinematic features 和 text semantics，让 cluster 在两个维度都有意义
2. **Cluster-specific delta action model** 解决 sim-to-real gap 因 motion 类型而异的问题
3. **Iterative refinement** 用 positive feedback loop 不断提升 tracking policy 和 delta model
4. **Expert-to-generalist distillation** 用 Transformer backbone 融合多 expert 知识

整个 pipeline 的 philosophy 是 "**Decompose the complexity**"——把 heterogeneous data distribution 拆成 homogeneous cluster，每个 cluster 单独学习，再融合。这和 directly 用更大 model（如 Diffusion Policy）"硬拟合" heterogeneous distribution 是互补的两条路。

对你 Andrej 来说，这个 paper 的一个有意思的 angle 是：**MoE 在 robot control 里的迁移**。LLM 里 MoE 是 routing-based（每层 router 决定 token 走哪个 expert），而 BB 是 **clustering-based MoE**（用 AE 做离线 routing，再 distill 成 single model）。后者更像经典 hierarchical RL 的思路，但用了 representation learning 做 task decomposition。这种 hybrid 思路在 robot learning 里可能会越来越流行，因为 online routing 在 continuous action space 上很难训稳。

希望这个讲解能 build 你对 humanoid WBC 当前 state 的 intuition。如果有具体某个部分想深入，比如 PPO 训练细节、delta action 的 gradient flow、或 Transformer generalist 的 attention pattern，我可以再展开。
