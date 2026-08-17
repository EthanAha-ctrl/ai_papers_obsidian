---
source_pdf: UniGraspTransformer Simplified Policy Distillation for Scalable Dexterous
  Robotic Grasping.pdf
paper_sha256: b0dafb524088072493d4005e62d6bbd1acf4ea38755de6cc0ef24487d2889853
processed_at: '2026-08-12T19:52:18-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 UniGraspTransformer

好,我换个更 intuitive 的角度讲,不堆术语,讲清楚**为什么这么做**和**为什么 work**。

---

## 一句话总结

这篇 paper 干的事: 给每个 object 训一个"专家" RL policy,然后把这些专家的"成功录像"全部录下来,扔给一个大 Transformer 让它"看录像学会"。就这么简单。

---

## 为什么 prior work 不 work?

UniDexGrasp++ 之前的做法: 训一个大 generalist policy 直接学 3200 个 object 抓取,发现训不动。于是搞了一套 complicated pipeline——先聚类别、训 specialist、再 distill 回 generalist、反复迭代。这套路有两个根本问题:

1. **Student model 太小**: online distillation 必须让 student 跟 teacher 同时跑 RL,student 一大就训不收敛。所以 student 只能是 4-layer MLP,容量不够,3200 个 object 的 grasping strategy 全挤在小网络里,result 是 collapse 到几种单调 pose。
2. **Pipeline 太复杂**: curriculum + clustering + iterative GSL,一层套一层,scaling 到更多 object 几乎不可能。

核心 bottleneck 是 **capacity**,不是 algorithm。

---

## UniGraspTransformer 的核心 insight

把 RL distillation 从 "online" 拆成 "offline" 两段:

```
Phase A: 训 3200 个 per-object expert (RL)
         ↓
Phase B: 用 expert 生成 3.2M 条成功 trajectory
         ↓
Phase C: 拿这些 trajectory 当 supervised data,训一个大 Transformer
```

Phase A 完成后,expert 的"知识"全部 encoded 在它的 trajectory distribution 里。Phase C 就是 supervised regression,$f_\theta(S_t) \to A_t$,完全没有 RL 的不稳定问题。Student 想多大就多大——12 层 self-attention 随便堆,batch size 800 trajectory 也跑得动。

这相当于把一个 "RL problem" 转成一个 "imitation learning problem",把 "policy optimization" 转成 "function fitting"。L2 loss 一上去,model 大就能学得动。

Reference:
- UniGraspTransformer: https://dexhand.github.io/UniGraspTransformer/
- UniDexGrasp++: https://arxiv.org/abs/2304.00464

---

## Phase A: Per-object expert 怎么训的?

每个 object 用 PPO 在 Isaac Gym 里 roll out 1000 个并行 env,训 10K iter。Reward 是关键,reward 设计决定了 trajectory 质量。

### Reward 分两个 phase

$$R = R_d + (1-f_c) R_o + f_c (R_l + R_g + R_s)$$

- $R_d$: hand 36 个 surface 点到 object point cloud 的平均 Chamfer distance,鼓励 hand 贴近 object
- $f_c$: contact flag,平均距离小于阈值 $\lambda_c=0.06$ 时变成 1
- $R_o$: pre-contact phase 起作用,惩罚 hand 偏离 pre-defined opening pose $q_{open}$,防止 hand 还没贴到 object 就握拳
- $R_l, R_g, R_s$: post-contact phase 才激活,lift、goal distance、success bonus

这个设计有个 intuitive 解释: 抓取是**两阶段动作**——先"张开手贴近 object",再"握紧提起来"。Reward 用 contact flag 切换阶段,避免 hand 在空中乱握。

### Ablation 证明这套设计 work

Table 10:
- 只用 object center 算 $R_d$,无 $R_o$: 90.3%
- 用 point cloud 算 $R_d$,无 $R_o$: 92.9%
- 完整版: 94.1%

用 point cloud 而非 center 是关键——球这种对称物体 center 够用,但剪刀、瓶子这种细长物体,center 无法告诉 hand "贴哪个面"。

Reference:
- PPO: https://arxiv.org/abs/1707.06347
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- DexGraspNet ($R_o$ 灵感来源): https://arxiv.org/abs/2304.11840

---

## Phase B: Trajectory 数据怎么生成?

每个 object 用对应 expert policy 生成 1000 条 successful trajectory。失败 trajectory 丢弃。最终:

$$|\mathcal{D}| = 3200 \text{ obj} \times 1000 \text{ traj} = 3.2\text{M trajectories}$$

每条 trajectory $\mathcal{T} = \{(S_1, A_1), ..., (S_T, A_T)\}$,长度 $T=200$。

每条 trajectory 同时存两份 point cloud:
- Complete cloud (from object mesh) — 喂给 state-based model
- Partial cloud (from 5-camera reconstruction) — 喂给 vision-based model

这是后面 vision-based adaptation 的伏笔——同一 trajectory 可以训两个 setting。

---

## Phase C: Transformer 怎么吃下这些 trajectory?

### Input 设计

6 个 modality 各自 encode 成 256-d token:

| Modality | Dim | 直觉 |
|---|---|---|
| Proprioception | 167 | Hand 自己在哪、关节角度、力 |
| Previous Action | 24 | 上一步做了什么 |
| Object State | 16 / 12 | Object 位置、朝向、速度 |
| Object Feature | 128 | PointNet 编出来的 geometry feature |
| Hand-Object Distance | 36 | Hand 36 点到 object cloud 的距离 |
| Time | 29 | 当前 step + sin/cos encoding |

**为什么用 6 token 而非 concat 成大 vector?** 直觉上,attention 让 modality 之间 cross-talk——proprioception token 可以 attend 到 object feature token,model 自己学"我的手现在离 object 哪个面最近"。这是 transformer 比 MLP 强的地方: **cross-modal reasoning**。

### Architecture

```
6 tokens (each 256-d)
   ↓
12 self-attention blocks
   ↓
6 refined tokens (each 256-d) concat → 1536-d
   ↓
4-layer MLP head → 24-d action
```

### Loss

$$\mathcal{L} = \|A_t - P_t\|_2$$

就一个 L2 loss。简单到让人怀疑——但 L2 对 continuous action BC 完全 OK,因为 teacher trajectory 是 single-mode deterministic(PPO 输出 mean),不存在 multimodal 问题。

### Ablation 证明 capacity 真的 matters

Table 4:
- 0 self-attention block (纯 MLP): 85.5%
- 6 blocks: 89.7%
- 12 blocks: 91.2%

每加 6 层 +3%。这说明 task 复杂度还没被 12 层 transformer 填满,继续加深可能还能涨。prior work 用 4-layer MLP 是 self-imposed limit,不是 task 的 limit。

Reference:
- Transformer: https://arxiv.org/abs/1706.03762
- PointNet: https://arxiv.org/abs/1612.00593
- Perceiver (modality token 思想): https://arxiv.org/abs/2103.03206

---

## Vision-Based Setting 怎么办?

State-based setting 假设你能直接读到 object 完整位置/朝向。Real world 里不行,只有 5 个 RGBD camera 拍出来的 partial point cloud。

### 两个 workaround

1. **Object position**: 用 partial cloud 的 center 替代 oracle position
2. **Object rotation**: 用 partial cloud 的 PCA 前三个主轴替代 oracle quaternion

### V-Encoder distillation

V-Encoder 见 partial cloud,S-Encoder 见 complete cloud。让 V-Encoder 的 latent feature 对齐 S-Encoder 的 latent:

$$\mathcal{L}_{distill} = \|z_t^S - z_t^V\|_2$$

直觉: S-Encoder 知道完整 shape,把它的 latent 当 soft target 让 V-Encoder 学,等价于隐式 shape completion。V-Encoder 见 partial cloud 时,被迫 infer 出 complete cloud 的 representation。

Table 9 ablation:
- 无 distillation: 86.7%
- 有 distillation: 88.9% (+2.2%)

### Object state ablation

Table 8:
- 啥估计都不用: 83.2%
- 只用 center: 86.4%
- center + PCA: 88.9%

PCA 提供 axis-aligned 近似 rotation,对 rigid body 够用。对 scissors 这种长条形 object 主轴方向有歧义,但这只是 cheap workaround,explains 为什么 vision 比 state 还是低 2-3%。

Reference:
- Privileged learning 思想: https://arxiv.org/abs/2010.05171
- Chamfer Distance: https://arxiv.org/abs/1712.01534

---

## 实验: 数字怎么读?

### 主表关键 take-away

| | State Seen | State Unseen-Cat | Vision Seen | Vision Unseen-Cat |
|---|---|---|---|---|
| UniDexGrasp++ | 87.9 | 83.1 | 85.4 | 76.7 |
| UniGraspTransformer | 91.2 | 88.3 | 88.9 | 86.8 |
| Gap | +3.3 | +5.2 | +3.5 | +10.1 |

**最戏剧性的是 vision unseen-unseen-cat +10.1%**。这说明:
- 大 Transformer 学到的是 category-agnostic 的 grasp strategy,不依赖 specific object geometry memory
- Prior work 的 small MLP 见 unseen category 直接 collapse,因为容量不够泛化

### Generalization gap 几乎消失

UniDexGrasp++: 85.4 (seen) → 76.7 (unseen cat) = **-8.7% drop**
UniGraspTransformer: 88.9 (seen) → 86.8 (unseen cat) = **-2.1% drop**

这个 gap 缩小是 capacity bottleneck 被解开的直接证据。

### Teacher-student gap

Teacher (per-object expert): 94.1%
Student (UniGraspTransformer): 91.2% state-based / 88.9% vision

差 2.9% / 5.2% 是 BC 的 inherent loss——student 只见过 teacher 的成功轨迹,没见过 teacher 探索过的失败。这部分 loss 是不可恢复的,除非 student 自己再 finetune RL。

---

## 几个值得思考的设计 choice

### 为什么 L2 loss 不 mode collapse?

Behavior cloning with L2 loss 经典问题: 如果数据 multi-modal,L2 把 modes 平均掉,model 输出 garbage。

但这里没 mode collapse,因为:
1. Teacher policy 是 deterministic PPO (输出 mean),trajectory 本身 single-mode
2. State $S_t$ 区分度高(object pose + hand pose + time),每个 $S_t$ 在数据里基本只对应一个 $A_t$
3. Model 够大,可以 memorize 这种 deterministic mapping

**但**: 如果同一个 $S_t$ 在不同 trajectory 里对应不同 $A_t$ (比如同一 object 不同 grasp pose),L2 还是会平均。Paper Figure 5 显示 student 保持了 diversity,说明 $S_t$ 区分度够强,没真正 collapse。

更彻底的改进: 换成 diffusion policy head。Diffusion policy 天然处理 multimodality,缺点是 inference 慢 50-100 step。

Reference:
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- BC mode collapse 讨论: https://arxiv.org/abs/1909.11296

### 为什么需要 Time embedding?

Teacher PPO 是 Markov policy,理论上 $A_t = \pi(S_t)$ 不依赖 $t$。但 BC 学 trajectory distribution 时,$S_t$ 的 marginal 随 $t$ 变(pre-contact vs post-contact),time embedding 充当 phase indicator。

更好的替代: return-to-go conditioning (Decision Transformer)。但这里 trajectory 全成功,return 没区分度,所以用 time。

Reference:
- Decision Transformer: https://arxiv.org/abs/2106.01345

### 为什么 6 token 而非 flatten?

每个 modality 独立 project 成 token,让 attention 做 cross-modal reasoning。如果 flatten 成一个大 vector 进 MLP,modality 之间 coupling 只能靠 hidden layer 学,效果差。Table 7 ablation 证明加 modality 都有用,但没单独 ablate token vs flatten,这个对比缺失。

---

## 整体 landscape: 这篇的位置

```
DexGraspNet (2023)         → grasp pose dataset
   ↓
UniDexGrasp (2023)         → 两阶段: static pose + goal-conditioned RL
   ↓
UniDexGrasp++ (2023)       → GSL + clustering + curriculum
   ↓
UniGraspTransformer (2024) → per-object RL + offline distillation + 大 Transformer
```

每一代都在**简化 pipeline + 放大 model**。UniGraspTransformer 把"online RL distillation"换成"offline BC",这一步 unlock 了 model capacity,直接让 12 层 Transformer 上场,效果立竿见影。

类比 LLM 领域: 这就像从 "multi-task RLHF with small model" 转到 "SFT on large model with curated data"。Pipeline 变简单,model 变大,效果变好。

---

## 个人 take

这篇 paper 给我的 intuition 是: **RL distillation 的 bottleneck 不在 algorithm,在 model capacity**。一旦把 RL 转成 supervised learning,model scaling law 就重新生效。Transformer 在 NLP 和 vision 都证明了"大就是好",在 manipulation 里被 online RL 训练稳定性卡了几年,UniGraspTransformer 用 offline BC 绕过去。

潜在方向:
1. **Diffusion policy student**: 处理 multimodality,看 diversity 是否进一步提升
2. **Real-world scaling**: 现在 real demo 只有 12 object,跟 model capacity 不匹配,需要大规模 real eval
3. **Cross-hand transfer**: 现在 Shadow Hand 18 DOF 训练,Inspire Hand 6 DOF deploy,这个 gap 怎么系统性 bridge?
4. **Trajectory 数据继续 scale**: Table 3 显示 1K trajectory/object 还没饱和,加到 5K 或 10K 可能再榨 +1-2%

总体看,这篇胜在 simplicity 和 scaling——没有 fancy trick,就是"per-object expert + offline distillation + 大 Transformer",但每个环节都做对了。

Reference:
- UniGraspTransformer project: https://dexhand.github.io/UniGraspTransformer/
- UniDexGrasp 系列: https://github.com/PKU-EPIC
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Shadow Hand: https://www.shadowrobot.com/dexterous-hand-series/
- Inspire Hand: https://inspire-robots.store/

---

# UniGraspTransformer: Simplified Policy Distillation for Scalable Dexterous Robotic Grasping 深度讲解

非常精彩的paper,我来帮你 build 一些 intuition。这篇文章核心思想其实非常 elegant:用 per-object RL teacher 生成海量 successful trajectories,然后通过 offline imitation learning / behavior cloning 蒸馏到一个大容量的 Transformer student。这等于把 generalist-specialist learning 从 online RL distillation 转成了 offline supervised regression。下面我分层讲一下。

---

## 1. Motivation 与核心 insight

UniDexGrasp++ (CVPR 2023, PKU-EPIC lab) 用 iterative generalist-specialist learning (GSL) + geometry-aware clustering + curriculum + online DAgger distillation 完成了一千多 object 的灵巧抓取。但其 online distillation 范式有本质缺陷:

- Online distillation 要求 student 和 teacher 在同一 RL loop 中交替 roll out,teacher 必须是个可执行策略,而 student 一般只能用浅 MLP(否则 RL 训练不收敛 + GPU memory 爆炸)。
- 一旦 student 容量小,几千 object 的 task space 就会被 collapse 到一组**重复、单调**的 grasping pose——这是 capacity bottleneck 的直接体现,而不是 exploration 问题。
- Curriculum + clustering 的复杂 pipeline 把 scaling 卡死在 ~1k object 量级。

UniGraspTransformer 的核心 insight 是: **teacher policy 一旦训完,它对世界的"知识"全部压缩在它的 trajectory distribution $\pi^*_t(A_t | S_t)$ 里**。只要采样足够多的 trajectories(3200 object × 1000 traj × 200 step ≈ 640M transitions),我们就把它从 RL 任务变成一个**纯 supervised regression**:学一个 $f_\theta(S_t) \to A_t$ 的 mapping。这样 student 的 architecture 可以放飞——直接堆 12 层 self-attention block,batch size 800 trajectories 也跑得动。

Reference:
- UniDexGrasp++ paper: https://arxiv.org/abs/2304.00464
- UniDexGrasp++ code: https://github.com/PKU-EPIC/UniDexGrasp-NF
- UniGraspTransformer project page: https://dexhand.github.io/UniGraspTransformer/
- Generalist-Specialist Learning (GSL) 原文: https://arxiv.org/abs/2206.07982
- DAgger 原文: https://arxiv.org/abs/1011.0686

---

## 2. Pipeline 三段式拆解

### Stage 1: Dedicated Policy Network Training

每个 object 训一个独立的 RL policy(3200 个 object → 3200 个独立 policy)。

- Simulator: **Isaac Gym 3.0** (NVIDIA),GPU-accelerated parallel env,1 object × 1000 envs 并行 roll out。
- Algo: **PPO** (Schulman et al., 2017),on-policy,model-free。
- Network: 4-layer MLP,hidden dim {1024, 1024, 512, 512},输出 24-d action(18 finger DOFs + 6 wrist DOFs,均 normalized to [-1, 1])。
- Training schedule: 16-step rollout buffer,10K iterations,lr=3e-4,~3 hours/object on a single V100。
- 总开销: 3200 × 3h / 16 GPUs ≈ 80 GPU hours(Paper 报告)。

这里有个小细节: dedicated policy 训练时**不包含 128-d object visual feature**(paper Appendix A.1 写了:"the 128-d object visual feature is excluded to enhance training efficiency")。也就是说 teacher 只需要 proprioception + previous action + object state(含完整位置和旋转)+ hand-object distance + time,271-d 输入即可,不需要先经过 PointNet 编码。这是合理的设计——teacher 是 RL,感知越轻训练越稳。

Reference:
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- PPO: https://arxiv.org/abs/1707.06347
- Isaac Gym docs: https://isaacgym.readthedocs.io/

### Stage 2: Grasp Trajectory Generation

每个 object 用其 dedicated policy 生成 **M=1000** 条 successful trajectories。平均 success rate 是 **94.1%**,意味着失败 trajectory 被丢弃。每个 trajectory $\mathcal{T} = \{(S_1,A_1), \ldots, (S_T,A_T)\}$,长度 $T=200$。

最终数据集规模:
$$|\mathcal{D}| = 3200 \times 1000 = 3.2\text{M trajectories} = 6.4\times 10^8 \text{ transitions}$$

每条 trajectory 在存储时除了常规的 $(S_t, A_t)$ 之外,还存了:
- Complete object point cloud ($1024\times 3$)——喂给 S-Encoder
- Partial object point cloud ($1024\times 3$)——喂给 V-Encoder

这是后面 vision-based distillation 的关键: 同一个 trajectory,state-based setting 用 complete cloud,vision-based setting 用 partial cloud。

### Stage 3: UniGraspTransformer Supervised Training

这是论文真正的核心。Loss function 极简:
$$\mathcal{L} = \|A_t - P_t\|_2$$
其中:
- $A_t \in \mathbb{R}^{24}$: ground-truth action at step-$t$ 来自 teacher trajectory
- $P_t = f_\theta(S_t) \in \mathbb{R}^{24}$: UniGraspTransformer 预测

Note: 这里用 L2 而不是 BC 常用的 NLL/Gaussian log-likelihood,本质上是 Gaussian BC with fixed variance。L2 简单粗暴,但对 continuous action space 而言和 MSE-equivalent Gaussian BC 没什么区别。我觉得这里可以改进的方向是加个 multi-modal head(比如 diffusion policy 或 MDN),因为同一个 $S_t$ 在不同 trajectory 里对应的 $A_t$ 可能 multimodal。但作者发现 L2 已经达到 94% teacher 性能,可能因为他们的 trajectory 数据本身由 single-mode deterministic policy(PPO 输出 mean)生成,所以不会 mode collapse。

Reference:
- Behavior cloning 综述: https://arxiv.org/abs/2205.10312
- Diffusion Policy (Toyota Research Institute): https://diffusion-policy.cs.columbia.edu/

---

## 3. Architecture 详解 (Figure 2c 解析)

### Input pipeline

UniGraspTransformer 把 6 个不同的 modality 各自 project 成 256-d token,然后丢进 Transformer:

| Modality | Dim | 含义 |
|---|---|---|
| Proprioception | 167 | Hand 本体感觉:wrist pos+rot(6) + finger joint angle/vel/force (66) + fingertip pos/quat/lin_vel/ang_vel/force/torque (5×16=80) + wrist force/torque 残余 |
| Previous Action | 24 | $A_{t-1}$,含 wrist force/torque (6) + finger joint angles (18) |
| Object State | 16 (state-based) / 12 (vision-based) | object center (3) + quaternion (4) + lin_vel (3) + ang_vel (3) + obj-goal dist (3); vision 用 PCA 三轴(9) 代替 quaternion |
| Object Feature | 128 | S-Encoder / V-Encoder 输出 |
| Hand-Object Distance | 36 | 36 个 hand 表面点到 object point cloud 的 Chamfer 距离 |
| Time | 29 | 当前时间 $t$ (1) + sin/cos time embedding (28),类似 transformer positional encoding 但当成 input modality |

Total: 6 tokens × 256-d。

注意每个 modality 用**独立的 single-layer MLP** 做 projection,这其实是个 mini-Perceiver/Flamingo 风格的 modality embedding,而不是 ViT 那种 patch flatten。

### Backbone

- **12 个 self-attention block**(原 transformer encoder block,pre-LN 或 post-LN paper 没细说,默认 transformer encoder 应是 post-LN)
- 输入 token 数 = 6(sequence length = 6),极短,所以 attention 是 $O(6^2)$=常数,完全不是 bottleneck。bottleneck 是 batch size 和 trajectory 长度。
- 输出: 6 个 refined token 各 256-d,**concat 成 1536-d**,然后接 4-layer MLP head 输出 24-d action。

这里有个有意思的设计选择:他们用 6 个 token 而非把所有 input concat 成一个大 vector 再处理。直觉上这相当于让 attention 在 modality 之间做 cross-talk,proprioception token 可以 attend 到 object feature token,这对耦合 hand-object 的几何关系有帮助。

Ablation Table 4 显示:
- No self-attention (纯 MLP): 85.5%
- 6 blocks: 89.7%
- 12 blocks: 91.2%

性能随 depth 增长 5.7 个点,说明 capacity 是真瓶颈。这点很有趣——传统 RL+transformer 实验(Decision Transformer, Trajectory Transformer)通常 6 层就饱和,这里到 12 层还没饱和,说明 task 本身 complexity 还没被 model 容量填满。

Reference:
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Trajectory Transformer: https://arxiv.org/abs/2106.08262
- Perceiver: https://arxiv.org/abs/2103.03206
- Flamingo: https://arxiv.org/abs/2204.14198

### Point Cloud Encoder (Figure 3 解析)

**S-Encoder (state-based)**:
- 输入: 1024×3 sampled points from complete object mesh
- 结构: PointNet 风格。每点 MLP (3→64→128→256) + max pool 全局 + 接两个 FC 段 输出 128-d global feature
- Decoder: 反过来用 MLP 把 128-d feature 拆回 1024×3
- Loss: **Chamfer Distance** 双向:
$$\mathcal{L}_{CD} = \text{ChamferDistance}(\hat{P}_j, \tilde{P}_j) = \frac{1}{|\hat{P}|}\sum_{x\in\hat{P}} \min_{y\in\tilde{P}}\|x-y\|^2 + \frac{1}{|\tilde{P}|}\sum_{y\in\tilde{P}} \min_{x\in\hat{P}}\|x-y\|^2$$
  - $\hat{P}_j = R_j(P_j - \mathbf{c}_j)$: 经过 random rotation $R_j$ 和 centroid 减法的 augmented point cloud
  - $\tilde{P}_j$: decoder 重建出来的
  - $P_j$: 原始 canonical point cloud
- 训练 800K iter,batch size 100,1× A100,40 hours

**V-Encoder (vision-based)**:
- 输入: partial point cloud 1024×3(从 5 个 RGBD 相机 fusion 后分割出 object 部分)
- 同样的 PointNet 结构,但 latent feature 用 **distillation loss** 对齐到 S-Encoder:
$$\mathcal{L}_{distill} = \|z_t^S - z_t^V\|_2$$
- Total loss:
$$\mathcal{L} = \omega_{CD} \mathcal{L}_{CD} + \omega_{distill}\mathcal{L}_{distill}, \quad \omega_{CD}=1.0, \omega_{distill}=0.1$$
- Ablation Table 9 显示 distillation loss 贡献 2.2% (86.7% → 88.9%)

这是个经典的 privileged learning / teacher-forcing 技巧——state-based encoder 见过 complete geometry,把它的 latent 作为 soft target 让 vision-based encoder 学,这等价于隐式 shape completion。

Reference:
- PointNet: https://arxiv.org/abs/1612.00593, code: https://github.com/charlesq34/pointnet
- Chamfer Distance: https://arxiv.org/abs/1712.01534
- Privileged learning / asymmetry actor-critic: https://arxiv.org/abs/2010.05171

---

## 4. Reward Function 详解 (Eq. 1)

$$R = R_d + (1-f_c) R_o + f_c (R_l + R_g + R_s)$$

变量解释:
- $R_d$: **distance reward**,负向惩罚 hand 36 个表面点到 object point cloud 的平均 Chamfer distance
  $$R_d = -\omega_d \cdot \frac{1}{36}\sum_{i=1}^{36}\text{ChamferDistance}(H_i, P_{obj}), \quad \omega_d=1.0$$
  - $H_i$: 第 $i$ 个 hand 表面点 (Figure 7c 标出 36 个)
  - $P_{obj}$: 完整 object point cloud
- $f_c \in \{0,1\}$: **contact flag**
  $$f_c = \mathbb{1}\left[\frac{1}{36}\sum_{i=1}^{36}\text{ChamferDistance}(H_i, P_{obj}) < \lambda_c\right], \quad \lambda_c = 0.06$$
- $R_o$: **opening reward**,pre-contact phase 惩罚 hand pose $q$ 偏离 predefined opening pose $q_{open}$
  $$R_o = -\omega_o \|q - q_{open}\|_2, \quad \omega_o = 0.1$$
- $R_l$: **lift reward**,鼓励沿 z 轴 lift
  $$R_l = \omega_l (1+a_z), \quad \omega_l = 0.1$$
- $R_g$: **goal reward**,object 到 goal 欧氏距离惩罚
  $$R_g = -\omega_g \|x_{obj} - x_{goal}\|_2, \quad \omega_g = 2.0$$
- $R_s$: **success reward**,sparse bonus
  $$R_s = \omega_s \mathbb{1}[\|x_{obj} - x_{goal}\|_2 < \lambda_g], \quad \lambda_g = 0.05, \omega_s=1.0$$

设计逻辑:
- Phase 1 (pre-contact, $f_c=0$): 只有 $R_d + R_o$ 起作用,要求 hand 张开($R_o$) 同时贴近 object($R_d$)。$R_o$ 的设计借鉴 DexGraspNet,防止 hand 还没贴上就握紧导致 grasp slip。
- Phase 2 (post-contact, $f_c=1$): $R_o$ 关掉,$R_d$ 继续维持 contact,加上 $R_l$ 把 object 提起来,$R_g$ 引导向 goal,$R_s$ 给最后 bonus。

Ablation Table 10 显示:
- 仅用 $R_d$ w/ center (无 $R_o$,无 $R_d$ w/ point cloud): 90.3%
- 用 $R_d$ w/ point cloud (无 $R_o$): 92.9%
- 完整版 (含 $R_o$ + point cloud-based $R_d$): 94.1%

这里关键 ablation 是把 $R_d$ 从"36 点到 object center"升级成"36 点到 object point cloud"——前者对对称物体(球、立方体)还行,对复杂形 object(剪刀、瓶子)几何 ill-posed,后者用完整几何指导 hand 贴合形状。

Reference:
- DexGraspNet: https://github.com/PKU-EPIC/DexGraspNet, paper: https://arxiv.org/abs/2304.11840
- Reward shaping: https://arxiv.org/abs/1606.06667

---

## 5. Vision-Based Adaptation 细节

### Object state 适配 (Table 8)

Vision setting 没有 oracle 位置/旋转。他们用:
- **Object position**: partial point cloud 的 center(3-d)
- **Object rotation**: partial point cloud 的 PCA 前三个主轴(9-d,即 3×3 矩阵展平)

Ablation 显示:
- 都不用 (baseline): 83.2%
- 只用 center: 86.4%
- center + PCA: 88.9%

PCA 提供的是**轴对齐**的近似 rotation,对近似 rigid object 还能用,对 non-rigid 或长条形 object 主轴方向有歧义性,所以这其实是 cheap workaround。更严谨的做法是 explicit pose estimation(比如 implicit-PDF, 或 PoseCNN),但工程量更大。

Reference:
- Implicit-PDF (Eppner et al.): https://arxiv.org/abs/2104.00957
- PoseCNN: https://arxiv.org/abs/1711.00199

### 5-Camera Setup

- 桌中心 (0,0,0) 为原点,相机位姿:
  - 顶部: (0, 0, 0.55) 
  - 四周: (±0.5, 0, 0.15), (0, ±0.5, 0.15)
- Focal point 都对准 (0, 0, 0.15)
- 拍摄 RGBD,fuse 出 scene point cloud,分割 object 部分,但**会被 Shadow Hand 部分遮挡** → partial cloud

### Vision setting 的 Object State 维度变化

state-based: 16-d(center+quat+lin_vel+ang_vel+goal_dist)
vision-based: 12-d(center(3) + 3 PCA axes(9))

省略 quaternion(4) 因为没 oracle orientation,但 PCA 是 9-d 不是 4-d,所以反而多 5-d。同时 velocity 在 vision 里直接砍掉(无法估计)。

---

## 6. 实验数据深度解读

### 主表 (Table 2)

| Method | State Seen | State Unseen-SeenCat | State Unseen-UnseenCat | Vision Seen | Vision Unseen-SeenCat | Vision Unseen-UnseenCat |
|---|---|---|---|---|---|---|
| PPO† | 24.3 | 20.9 | 17.2 | 20.6 | 17.2 | 15.0 |
| DAPG† | 20.8 | 15.3 | 11.1 | 17.9 | 15.2 | 13.9 |
| ILAD† | 31.9 | 26.4 | 23.1 | 27.6 | 23.2 | 20.0 |
| GSL† | 57.3 | 54.1 | 50.9 | 54.1 | 50.2 | 44.8 |
| UniDexGrasp | 79.4 | 74.3 | 70.8 | 73.7 | 68.6 | 65.1 |
| UniDexGrasp++ | 87.9 | 84.3 | 83.1 | 85.4 | 79.6 | 76.7 |
| **Ours** | **91.2** | **89.2** | **88.3** | **88.9** | **87.3** | **86.8** |

观察:
1. **Generalization gap 缩小**: UniDexGrasp++ 在 vision setting 从 85.4→76.7 是 -8.7%,UniGraspTransformer 只从 88.9→86.8 是 -2.1%。这是很大的改进——意味着模型在 unseen category 上几乎不掉点,说明 Transformer 的容量让它在 feature space 里学到了真正 category-agnostic 的 grasp strategy。
2. **Vision drop 也变小**: state-based 91.2 vs vision 88.9 仅 -2.3%,说明 V-Encoder 的 distillation 设计很成功,partial cloud 信息几乎填平。
3. **Teacher-student gap**: teacher (dedicated policy) 平均 94.1% → student 91.2% state-based,-2.9% 是 supervised regression 的信息损失。这是 BC 的 inherent limitation——student 只见 trajectory,不见 teacher 见过的"失败探索"。
4. **vs PPO baseline**: 91.2 vs 24.3 = **+66.9% absolute**。这数字其实有意义——它说明 "RL from scratch on 3200 objects with single policy" 几乎不可行,而 per-object RL + distillation 这条路 truly scales。

### Trajectory Number (Table 3)

| M (trajectories/object) | 0.2K | 0.5K | 1K |
|---|---|---|---|
| Success Rate | 87.2 | 89.3 | 91.2 |

从 0.2K → 1K: +4%,log-like 增长。理论上还可继续加,但 1000 traj × 3200 obj × 200 step = 640M transitions 存储成本不低。看起来 1K 还没完全饱和,继续加到 2K 或 5K 可能再榨 +1%。

### Object Number (Table 5)

| Objects | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|
| Success Rate | 92.5 | 91.8 | 91.3 | 91.2 |

注意: 这里 evaluation 是对应 object set (seen)。Success rate 几乎**保持不变甚至略降**,说明模型 capacity 还有富余,object 数量还没到饱和点。这是 Transformer 大模型的福利——可以继续扩 object 而不掉点。

### Online vs Offline (Table 6)

400 object 子集:
- DAgger (online): 88.2%
- UniGraspTransformer (offline): 92.5%

**+4.3%** 的提升,直接来自 (a) 更大 model 容量 (12 transformer blocks vs MLP),(b) offline 不用 RL,稳定性更高,(c) 见过更多 trajectory。这说明把 RL distillation 转成纯 supervised regression 是真的有信息增益的。

Reference:
- DAgger: https://arxiv.org/abs/1011.0686
- DAPG: https://arxiv.org/abs/1809.02052

### Input Components Ablation (Table 7)

逐个加 input:
- baseline (proprioception + prev_action + obj_state): 78.4%
- + obj feature (128-d from S-Encoder): 86.6% (+8.2%)
- + hand-obj distance: 89.9% (+3.3%)
- + time embedding: 91.2% (+1.3%)

Object feature 贡献最大(+8.2%),因为这是唯一携带几何 shape 信息的 input。Hand-obj distance 提供 spatial relation。Time embedding 帮助 model 知道"现在 trajectory 进行到哪了"——这其实类似 Decision Transformer 的 return-to-go conditioning,但这里是用绝对时间步而非 return。

---

## 7. Grasp Pose Diversity (Figure 4, 5)

UniDexGrasp++ 因为 capacity bottleneck,生成 grasp pose 在不同 object 上**趋同**(monotonous);UniGraspTransformer 加深 model 后保持 teacher policy 的多样性。Figure 5 的可视化显示同一 object 同一 initial pose 下 UniGraspTransformer 能生成两种**显著不同**的 grasp pose,而 UniDexGrasp++ 几乎只能给一种。

这个现象背后的直觉: Behavior cloning with L2 loss 会**回归到 mean**,如果数据 multi-modal,L2 把 modes 平均掉变成 mode collapse。但这里 model 够大,且 state $S_t$ (尤其 object pose + hand pose + time) 区分度高,所以 $f_\theta(S_t) \to A_t$ 这个 function 几乎 deterministic mapping,每个 $S_t$ 对应一个 teacher 产生的 $A_t$,所以 BC 学到的是 teacher 的具体行为而非 mean。这与 L2 不会 mode collapse 的关键条件: **state-conditioned** behavior cloning,而非 unconditional marginal matching。

Reference:
- 关于 BC mode collapse 的讨论: https://arxiv.org/abs/1909.11296
- Diffusion policy for multimodality: https://diffusion-policy.cs.columbia.edu/

---

## 8. 与相关工作的脉络联系

让我帮你把这篇放在整个 landscape 里看:

### DexGraspNet/UniDexGrasp 谱系
- DexGraspNet (CVPR 2023, PKU-EPIC): 大规模 grasp dataset via motion planning + force closure
- UniDexGrasp (ICCV 2023, PKU-EPIC): 两阶段 - 静态 grasp pose 生成(IPDF)+ goal-conditioned RL + curriculum
- UniDexGrasp++ (CVPR 2023): GSL + geometry clustering + iterative generalist-specialist
- **UniGraspTransformer (2024+)**: 同一 lab,把 pipeline 大幅简化为 offline distillation

Reference:
- UniDexGrasp: https://arxiv.org/abs/2303.00809
- DexGraspNet: https://arxiv.org/abs/2304.11840
- IPDF: https://arxiv.org/abs/2104.00957

### 类比 LLM 的"先 specialize 后 generalize"

这篇 pipeline 有个有意思的类比:类似 mixture-of-experts 或者 specialist-then-generalist 的 LLM 训练(专家细分 + ensemble merge)。
- Teacher 是 per-object specialist (3200 个 expert)
- Student 是 generalist Transformer
- Distillation 是 offline knowledge merging

类比 SpecTra / MOLE / Branch-Train-Merge 等 LLM ensemble work。

Reference:
- Branch-Train-Merge: https://arxiv.org/abs/2208.03306
- SpecTra / specialist distillation: https://arxiv.org/abs/2203.11855

### RL Distillation 谱系

- DAgger (Ross et al. 2011): online iterative BC
- Policy Distillation (Rusu et al. 2016): multi-task distillation via KL on policy output
- Actor-Mimic (Parisotto et al. 2016): DQN-based policy distillation
- Distral (Teh et al. 2017): multi-task RL with shared + task-specific components
- GSL (Jia et al. 2022): generalist-specialist iterative
- **UniGraspTransformer**: 把 RL distillation 完全转成 offline BC,显著简化

Reference:
- Policy Distillation: https://arxiv.org/abs/1511.06295
- Actor-Mimic: https://arxiv.org/abs/1511.04379
- Distral: https://arxiv.org/abs/1707.04175

---

## 9. 一些我觉得可以深入探讨/改进的点

### 9.1 Action 的 representation 选择

Paper 直接 regression 24-d continuous action。但有几点:
- Wrist 6 DOF 是 force/torque 控制,finger 18 DOF 是 position 控制。两套物理量混在一个 vector 里 L2 loss,但量纲不一致。是否需要 normalize 或加权?
- 没考虑 action 的 multimodality——同一 $S_t$ 不同 trajectory 可能对应不同 $A_t$(grasp pose 多解)。L2 把它们平均掉。一个改进: 用 diffusion policy 或 MDN 来 capture multimodality。

### 9.2 Time embedding 的角色

为什么需要 time embedding? Teacher PPO 是 Markov policy,理论上 $A_t = \pi(S_t)$ 不依赖 $t$。但 BC 学 trajectory distribution 时 $S_t$ 的 marginal 分布随 $t$ 变化(pre-contact phase vs post-contact phase),所以 time 是一种 phase indicator。其实更好的替代是给 model 加上 return-to-go (Decision Transformer) 或 progress label,但这里 teacher 都成功,return 没区分度,所以用 time。

### 9.3 Partial observability 处理

Vision-based setting 用 PCA 和 center 做 cheap 替代。更好的做法是 explicit pose estimation model + 不确定性传播(比如 Implicit-PDF 那种 distributional pose)。或者直接学习 end-to-end vision policy 不显式用 object state。

### 9.4 Real-world deployment

Appendix 提到 Inspire Hand (6 finger DOF,远比 Shadow Hand 18 DOF 简单) 的 real-world demo。这暗示:
- Policy 容易迁移到不同 hand
- 但 demo 只有 12 object,规模不大
- 没报告 real-world success rate 数据,只有 video,这点对 sim-to-real 验证较弱

Reference:
- Inspire Hand: https://inspire-robots.store/
- Real-world sim-to-real dexterous (DexPoint): https://arxiv.org/abs/2211.01860
- OpenAI Rubik's cube hand: https://arxiv.org/abs/1910.07113

### 9.5 与 Diffusion Policy 的对比

Diffusion Policy (Toyota Research Institute, RSS 2023 best paper) 在 manipulation 任务上展示 BC + diffusion 的 multimodal 优势。UniGraspTransformer 是 deterministic Transformer + L2 loss。两者都没在统一 setup 上对打过,但理论上:
- Diffusion policy 更好处理 multimodality,但 inference 慢(50-100 步去噪)
- UniGraspTransformer 推理快(一次 forward pass),但可能 mode collapse

Reference:
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- 3D Diffusion Policy (DP3): https://arxiv.org/abs/2403.03954

---

## 10. 关键 take-away 总结

1. **核心 insight**: RL policy 一旦训好,其"知识"全部在 trajectory distribution 里。把 RL distillation 转成 offline supervised regression,可以释放 student model 容量(MLP → 12 层 Transformer)。
2. **Teacher 设计**: per-object PPO,reward 用 contact flag 切 phase,$R_d$ 用 36 个 hand 点到 object point cloud 的 Chamfer 距离,$R_o$ 强制 pre-contact 张开。Average success 94.1%。
3. **Student 设计**: 6 modality tokens → 12 self-attention blocks → 4-layer MLP head,L2 loss。
4. **Vision adaptation**: S-Encoder (complete cloud) → V-Encoder (partial cloud) distillation + PCA 替代 orientation。
5. **实验**: 91.2% state-based / 88.9% vision-based,比 SOTA UniDexGrasp++ 提 +3.3/+3.5/+4.9/+5.2/+7.7/+10.1 个点。最 dramatic 是 unseen-unseen category 在 vision 上 +10.1%。
6. **Generalization gap 几乎消失**: vision-setting unseen category 86.8%,只比 seen object 88.9% 低 2.1%,说明 Transformer 容量 + 多样 trajectory 让 model 学到了 category-agnostic grasp strategy。

总体看,这篇 paper 工程上很扎实,insight 上很干净: capacity bottleneck 是 prior work 痛点,offline distillation 是 unlock capacity 的钥匙,Transformer 是 unlock 后的合理 architecture choice。整个 pipeline 没有任何"神奇"的 trick,胜在 simplicity 和 scaling。

Reference:
- Paper project page: https://dexhand.github.io/UniGraspTransformer/
- UniDexGrasp++ (前作): https://github.com/PKU-EPIC/UniDexGrasp-NF
- UniDexGrasp (前前作): https://github.com/PKU-EPIC/UniDexGrasp
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- PPO: https://arxiv.org/abs/1707.06347
- Shadow Hand: https://www.shadowrobot.com/dexterous-hand-series/
- Inspire Hand: https://inspire-robots.store/
- PointNet: https://github.com/charlesq34/pointnet
- Transformer: https://arxiv.org/abs/1706.03762
- DAgger: https://arxiv.org/abs/1011.0686
- DAPG: https://arxiv.org/abs/1809.02052
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DexGraspNet: https://github.com/PKU-EPIC/DexGraspNet
- GSL: https://arxiv.org/abs/2206.07982
- Chamfer Distance: https://arxiv.org/abs/1712.01534

如果你接下来想 build 在这上面,我会推荐三个方向:
- **Diffusion policy student**: 把 Transformer student 换成 diffusion policy head,看 multimodality 是否进一步提升 diversity 和 robustness。
- **Multi-finger vs gripper transfer**: 把 dedicated policy 的 reward 范式转到 parallel gripper / soft hand 上,看 pipeline 是否 hardware-agnostic。
- **Real-world scaling**: paper 的 real-world demo 只有 12 object,跟 model 容量不匹配。需要更大规模 real eval 才能验证 sim-to-real 真的 robust。
