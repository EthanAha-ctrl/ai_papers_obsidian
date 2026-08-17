---
source_pdf: RL from Physical Feedback.pdf
paper_sha256: 0d0cc2f23ea913dd88f1c06ab28406200a5b953f2ed3d38515f13c118f7ede8c
processed_at: '2026-08-11T23:53:00-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RLPF 人话版

Andrej，我把之前那堆公式拆掉，用最直白的话再讲一遍。

---

## 核心问题一句话

**现在的text-to-motion模型生成的动作，机器人根本做不了。**

为什么？因为这些模型是graphics社区训出来的，它们的objective是"看起来像人动的"，完全不管物理。结果生成出来的动作有三个致命问题：

- 脚在地上滑（foot sliding）——违反摩擦力
- 身体穿进地面（ground penetration）——违反刚体约束
- 质心轨迹飘——机器人按这个走直接摔倒

你在simulation里跑这种动作，humanoid瞬间趴下。更别说real robot了。

---

## RLPF的核心思路

**把physics simulator当成reward function，用RL把motion generator往"机器人能执行"的方向调。**

类比一下：
- **RLHF**（LLM）：人给偏好 → 训reward model → PPO调policy
- **RLPF**（motion）：simulator跑动作 → 成功给1失败给0 → GRPO调policy

本质就是：你生成了一个动作，我让机器人在sim里试一遍，成功了reward=1，摔了reward=0。然后用这个信号去调你的生成模型，让它下次生成更可能成功的动作。

---

## 三个组件各自在干嘛

### 1. Motion Generator（生成动作的LLM）

LLaMA2-7B，把motion当作一种"foreign language"。文本进去，motion token序列出来，再通过VQ decoder还原成连续动作。

pre-training阶段就是在MotionX这个大数据集（81K条motion-text pair）上做标准next-token prediction。这个部分没有什么新东西，就是MotionGPT那套。

### 2. Tracking Policy（判断动作能不能执行的裁判）

这是整个pipeline的key。它的工作是：给一个reference motion，让humanoid在simulator里跟着做，看能不能做下来。

训练分两步（teacher-student distillation）：

**Teacher**：在IsaacGym里训练，看得到所有privileged信息（真实friction、真实root velocity、global joint positions等），用PPO训出一个oracle policy。

**Student**：要deploy到real robot的版本，只能看proprioception（自己的joint state）+ motion target + 历史observation。用DAgger从teacher蒸馏过来。

为什么这么搞？因为real robot看不到friction这种simulator-only的量，但teacher知道。蒸馏让student学会从历史observation推断这些latent变量。长observation history是关键——student得从过去几十帧推断当前系统的物理状态。

这个student policy训好之后，**它的成功/失败就是RLPF的reward信号**。

### 3. Alignment Verification（防止reward hacking）

这个模块是救命的。如果你只用tracking reward，会发生什么？

Policy发现：站着不动最容易track成功。于是它生成的motion全变成"standing"——成功率高到0.99，但完全没有semantic content。用户说"跳舞"，它生成"站着"；用户说"走路"，它还是"站着"。

Paper的Figure 3直接展示了这个failure mode——**RLPF-w/o align的FID从3.61飙到32.53，R@1从0.26塌到0.09**。这是教科书级别的reward hacking。

所以加了一个alignment reward：用预训练的text encoder和motion encoder，把生成motion和input text embed到shared space，算它们的距离。距离越近reward越高。

两个reward一起用：tracking reward负责physics，alignment reward负责semantic。权重5:1（tracking:align），因为physics更难达到。

---

## 整个pipeline串起来

1. **Text prompt**进来，LLM生成一组motion token
2. Motion token经过VQ decoder还原成连续motion
3. Motion经过retargeting（SMPL → Unitree G1的形态），因为人和robot的骨架不一样
4. Retargeted motion送进simulator，student policy尝试track
5. 成功了reward=1，摔了reward=0（再乘以weight 10）
6. 同时用contrastive encoder算alignment reward（weight 2）
7. 两个reward加起来，送进GRPO更新LLM参数

GRPO为什么work？同一text prompt采样20个motion（group size=20），reward在group内归一化。某个motion比group平均好，advantage正，policy往那边走；比平均差，advantage负，policy远离。

这消除了absolute reward scale的影响，也不需要训critic network。binary sparse reward在这种group-relative框架下能work，是因为只要group内有variation，就有learning signal。

---

## 实验告诉我们什么

### Q1: RLPF比SFT强多少？

IsaacGym Succ：
- Base Model: 0.43
- SFT: 0.36（反而变差——reinforce了infeasible motion）
- SFT-Filter（只用可track的数据SFT）: 0.57
- RLPF: 0.95-0.97

**结论**：filter数据SFT只能小幅提升，因为分布没变。RL直接shift distribution，效果碾压。

### Q2: Tracking reward多重要？

去掉tracking reward，只留alignment reward：
- IsaacGym Succ从0.95掉到0.32
- 比Base Model还差

**结论**：没有物理信号，光靠semantic alignment完全无法提升feasibility。这个reward是不可或缺的。

### Q3: Alignment reward多重要？

去掉alignment reward，只留tracking reward：
- Succ达到0.99（physics满分）
- 但FID从3.61飙到32.53，R@1从0.26塌到0.09
- 生成的motion全是standing，semantic完全丢失

**结论**：两个reward缺一不可。tracking推动physics，alignment防止collapse。

### Sim-to-real gap

IsaacGym Succ 0.92 → MuJoCo Succ 0.63，drop 30%。

这说明IsaacGym的dynamics比real world乐观。所以paper的evaluation protocol是三阶段：IsaacGym训练 → MuJoCo zero-shot验证 → Real Unitree G1部署。MuJoCo更接近real，能筛出真正robust的policy。

---

## 为什么这个工作重要

1. **Generative model + Physical reward是scalable pattern**。不只是motion，任何需要物理feasibility的生成任务（grasp、trajectory、manipulation）都能用这个pattern。

2. **Binary sparse reward在GRPO下能work**。因为group-relative advantage不需要dense reward。这对real robot RL意义重大——binary success最容易measure，dense reward往往要手工设计，容易hacking。

3. **Alignment reward防止reward hacking是必须的**。robotics RL里这点常被忽视。纯physical reward会collapse到degenerate behavior。这跟LLM的RLHF一样——纯human preference会sycophancy，要加truthfulness/ usefulness reward。

4. **Frozen tracking policy是limitation**。如果生成OOD motion，tracking policy失败 → reward=0 → 学不到。未来需要co-training，让tracking policy能handle novel motion。

---

## 一句话总结

**把simulator的binary success当成RL reward，用GRPO调motion generator，同时用contrastive alignment reward防止semantic collapse。**

就这么简单。优雅，且work。

---

# RLPF: RL from Physical Feedback 深度解析

Andrej，这篇paper的核心idea非常优雅——把RLHF范式从language domain迁移到motion domain，但reward signal不再是human preference，而是来自physics simulator的execution success。这是一个sim-to-real闭环的精彩实例。让我从intuition层面到implementation层面给你彻底拆解。

---

## 1. 核心问题与动机

### 1.1 为什么T2M模型生成的motion不能直接deploy到humanoid

现有的text-to-motion模型（T2M-GPT, MotionGPT, MDM, MotionDiffuse等）源自computer graphics社区，它们的训练objective是**semantic alignment + visual realism**，但完全没有physical constraint的supervision。这导致生成motion存在三类failure mode：

- **Foot sliding**: 接触地面的脚在水平方向滑动，违反friction constraint
- **Ground penetration**: 身体部位穿入地面
- **Dynamic instability**: 质心轨迹不可执行，机器人会直接摔倒

Paper的Figure 1直观展示了这个问题：传统T2M生成的motion在simulation里robot会立刻摔倒，而RLPF生成的motion在real Unitree G1上能稳定执行。

### 1.2 为什么不直接做motion filtering或retargeting fix

这是一个值得深思的设计选择。Naive方案是：
1. 用T2M生成motion
2. 用tracking policy尝试执行
3. Filter掉失败的，只保留成功的

这正是SFT-Filter baseline做的事。但问题在于——T2M模型本身分布是fixed的，filter只能从分布里采到很少的feasible sample，无法systematically shift distribution。RLPF的关键洞察是：**通过policy gradient直接优化motion generator的参数，让整个分布向physically feasible region偏移**，同时用alignment reward防止semantic drift。

---

## 2. 架构总览

RLPF的pipeline可以分成三个phase：

### Phase 1: Pre-training (Large Motion Model)
```
Text → LLM (LLaMA2-7B) → Motion Tokens → VQ Decoder → Motion Sequence
```

### Phase 2: Physical Feedback Pipeline
```
Motion Sequence → SMPL Retargeting → Robot Motion → Tracking Policy (Exbody2 student) → Success/Fail → Reward
```

### Phase 3: RL Fine-tuning (GRPO)
```
LLM Policy → sample G motions → compute rewards (tracking + alignment) → GRPO update
```

---

## 3. Motion Tokenizer与Large Motion Model

### 3.1 VQ-VAE Motion Tokenizer

Motion sequence $\vec{m}_{1:M}$ 包含M帧，每帧是joint positions + rotations的表示。Tokenizer的结构：

$$\vec{z}_{1:L} = \mathcal{E}(\vec{m}_{1:M}), \quad \vec{m}_{1:M} = \mathcal{D}(\vec{z}_{1:L})$$

- $\mathcal{E}$: encoder，将连续motion压缩为discrete latent tokens
- $\mathcal{D}$: decoder，从tokens重建motion
- $\vec{z}_{1:L}$: L个discrete tokens，每个来自codebook $\mathcal{C}$
- M: 帧数，L: token数（L < M，存在temporal compression）

VQ-VAE的核心commitment loss让encoder output接近codebook entry，codebook loss让codebook entry接近encoder output。这里codebook size是K（paper中未明确，但MotionGPT系列一般用512-2048）。

### 3.2 LLM as Motion Generator

LLaMA2-7B作为decoder-only causal transformer，通过扩展vocabulary加入K个motion tokens。Pre-training objective是标准next-token prediction：

$$\mathcal{L}(\Theta) = -\sum_{j=1}^{T} \log P_\Theta(z_j | l, \vec{z}_{1:j-1})$$

变量解释：
- $\Theta$: LLM参数
- $z_j$: 第j个target motion token
- $l$: text instruction
- $\vec{z}_{1:j-1}$: 已生成的motion token prefix
- $T$: target sequence长度

这就是standard cross-entropy loss，让LLM学会在给定text条件下生成motion token序列。预训练数据是MotionX（81K sequences），这是相对大规模的motion-language pair。

**Intuition**: 把motion当作"foreign language"，LLM的language modeling能力可以transfer到motion modeling，因为两者都是sequential discrete tokens with structure。

---

## 4. RL Fine-tuning: GRPO

### 4.1 为什么选GRPO而不是PPO

GRPO来自DeepSeek的DeepSeekMath（[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)）。相比PPO，它的优势是：

- **无需critic network**: PPO需要训练value function $V_\phi(s)$来估计advantage，这需要额外的network和训练成本
- **Group-relative baseline**: 直接从同一prompt采样G个samples，用group mean/std归一化reward作为advantage

对于motion generation这种single-step task（生成完整个sequence再评估），GRPO特别合适，因为没有temporal credit assignment问题。

### 4.2 GRPO Objective详解

公式(2)完整展开：

$$\mathcal{L}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(r_i A_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)A_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\right)\right]$$

变量解析：
- $r_i = \frac{\pi_\theta(\vec{z}^i|l)}{\pi_{\theta_{old}}(\vec{z}^i|l)}$: importance sampling ratio，新policy vs 旧policy在sample $i$ 上的概率比
- $A_i$: advantage，由公式(4)计算
- $\epsilon$: PPO clip range，限制policy update幅度，防止破坏性更新（典型值0.1-0.2）
- $\beta$: KL penalty系数，paper Table 9中设为1.0
- $\pi_{ref}$: reference policy（通常是SFT后的initial policy），防止policy偏离太远

### 4.3 Advantage计算

公式(4)的group-relative advantage：

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

- $r_i$: 第i个sample的total reward（tracking reward + alignment reward）
- $G$: group size，paper Table 9中Num Generations = 20
- mean, std: 在group内计算

**Intuition**: 同一text prompt采样20个motion，reward相对group mean的高低决定advantage方向。如果某个motion比group平均更feasible且aligned，它的advantage为正，policy会增加它的生成概率；反之则降低。这消除了absolute reward scale的影响，只在group内做相对比较。

### 4.4 KL Divergence估计

公式(3)使用了k3 estimator（参考[GRPO implementation](https://arxiv.org/abs/2402.03300)）：

$$\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(\vec{m}^i|l)}{\pi_\theta(\vec{m}^i|l)} - \log\frac{\pi_{ref}(\vec{m}^i|l)}{\pi_\theta(\vec{m}^i|l)} - 1$$

这是一个unbiased estimator。设 $x = \frac{\pi_{ref}}{\pi_\theta}$，则 estimator = $x - \log x - 1$。当 $\pi_\theta = \pi_{ref}$ 时 $x=1$，estimator=0，这是正确的最小值。

注意这里使用了 $\vec{m}^i$ 而不是 $\vec{z}^i$，因为motion token经过decoder后才是实际评估的motion。

---

## 5. Motion Tracking Reward

### 5.1 Motion Retargeting (SMPL → Robot)

Humanoid和human的morphology不同（limb length, DoF配置），需要retargeting。Paper采用H2O（[arXiv:2406.10454](https://arxiv.org/abs/2406.10454)）的optimization-based方法：

**Step 1: Shape alignment**
优化SMPL的shape parameter $\beta$：
$$\min_\beta \sum_{i=1}^{14} \|L_{human}^i(\beta) - L_{robot}^i\|^2$$

其中 $L^i$ 是第i个link的长度，paper Table 8列出了14个对应link（pelvis, hip, knee, ankle, shoulder, elbow, hand, head左右两侧）。

**Step 2: Pose alignment**
固定 $\beta$，优化pose parameter $\theta$（SMPL的joint rotations）+ translation $p$：
$$\min_{\theta, p} \sum_{i} \|J_{human}^i(\theta, p) - J_{robot}^i\|^2 + \lambda \mathcal{R}_{smooth}$$

其中 $J^i$ 是keypoint position，$\mathcal{R}_{smooth}$ 是smoothness regularization防止jerky motion。

**Intuition**: 这是inverse kinematics的优化形式，因为humanoid DoF数和human DoF数不匹配（humanoid G1有23 DoF，human SMPL有更多），需要找到closest pose projection。Decoupling shape和pose优化避免ill-conditioning。

### 5.2 Teacher-Student Tracking Policy

基于Exbody2（[arXiv:2412.13196](https://arxiv.org/abs/2412.13196)）的two-stage distillation：

**Stage 1: Teacher Policy (Oracle)**

Teacher $\hat{\pi}$ 用PPO在IsaacGym训练，input包含：
- $o_t$: proprioceptive state（joint position, velocity, IMU等）
- $g_t$: motion tracking target
- $p_t$: privileged information（root velocity, global joint positions, friction, motor strength等simulator-only info）

Output: $\hat{a}_t \in \mathbb{R}^{23}$（23个joint target positions for PD controller）

Reward设计（参考Exbody2）通常包含：
- Joint position tracking reward
- Body posture reward
- Smoothness penalty
- Energy penalty

**Stage 2: Student Policy (Deployable)**

Student $\pi$ 只用real-world observable input：
- $O_{t-H:t}$: 历史observation window（长度H）
- $g_t$: motion target

通过DAgger（[arXiv:1011.0686](https://arxiv.org/abs/1011.0686)）distillation：
1. Student在sim里rollout收集states
2. 用teacher在这些states上生成oracle actions $\hat{a}_t$
3. 训练student最小化MSE：$\hat{l} = \|a_t - \hat{a}_t\|^2$
4. 迭代直到convergence

**Intuition**: Teacher看到ground truth物理量（如真实friction）能做出optimal action；student只看history和target，需要学会从部分observation推断latent physical state。长history window是关键，因为student需要从过去的observation sequence推断当前系统动力学。

### 5.3 Tracking Reward计算

公式(5)定义offline evaluation的reward：

$$R_{tracking}^{m_i} = \mathbb{I}(Succ(\pi, m_i))$$

- $\pi$: 训练好的student policy
- $m_i$: 第i个retargeted motion
- $Succ(\cdot)$: 二值success flag
- $\mathbb{I}(\cdot)$: indicator function

Success criteria:
- 任意时刻average deviation > 0.5m → fail
- root pitch angle超过threshold → fail（robot摔倒）

**Critical insight**: 这是一个**binary sparse reward**。Table 9显示Reward Weight Tracking = 10，意味着这个binary信号被放大10倍来驱动learning。这种sparse reward在GRPO框架下之所以能work，是因为group内相对比较——只要不是所有sample都成功或都失败，advantage就有信号。

---

## 6. Alignment Verification Module

### 6.1 为什么需要这个模块

如果只用tracking reward，RLPF会collapse到"最容易track的motion分布"——paper Figure 3显示，RLPF-w/o align会生成大量standing motion，因为standing motion最physically stable，但完全丢失semantic content。这是经典的reward hacking。

### 6.2 Contrastive Pre-training

公式(6)是standard contrastive loss（类似CLIP）：

$$\mathcal{L}_{CL} = (1-y)(\|\mathbf{f_t} - \mathbf{f_m}\|)^2 + y \cdot \max(0, m - \|\mathbf{f_t} - \mathbf{f_m}\|)^2$$

变量：
- $y \in \{0, 1\}$: matching label（1 = positive pair, 0 = negative pair）
- $m$: margin threshold
- $\mathbf{f_t} = E_t(\mathbf{t})$: text embedding
- $\mathbf{f_m} = E_m(\mathbf{m})$: motion embedding

**Loss形式解析**：
- Positive pair (y=1): loss = $\max(0, m - \|\mathbf{f_t} - \mathbf{f_m}\|)^2$，pull embeddings together until distance < m
- Negative pair (y=0): loss = $\|\mathbf{f_t} - \mathbf{f_m}\|^2$，push embeddings apart

这是Contrastive Loss原版（Yann LeCun 2006）形式，不同于InfoNCE的softmax形式。它在shared embedding space里enforce matching pair距离小，non-matching pair距离大。

### 6.3 两种Alignment Reward

公式(7)定义两个verification reward：

$$R_{TA}^{m_i} = \|E_t(t) - E_m(m_{pred})\|^2$$
$$R_{MA}^{m_i} = \|E_m(m) - E_m(m_{pred})\|^2$$

- $R_{TA}$: Text Alignment——generated motion embedding与input text embedding的距离
- $R_{MA}$: Motion Alignment——generated motion embedding与ground truth motion embedding的距离
- $m_{pred}$: 生成的motion
- $m$: ground truth motion
- $E_t, E_m$: 预训练的text/motion encoder

注意这两个reward是**distance**，越小越好，但在RL中需要reward越大越好。实际使用时会取负值或者用 $1/(1+d)$ 转换。

**两个reward的语义**:
- $R_{TA}$: 确保生成motion与input text语义对应（这是用户的真实intent）
- $R_{MA}$: 确保生成motion与ground truth motion的kinematic结构相似（防止semantic drift但保留分布内diversity）

### 6.4 MA vs TA的选择

Table 1显示：
- RLPF-MA在FID上更好（CMU: 3.61 vs 8.23，AMASS: 3.34 vs 4.47）
- RLPF-TA在R@1上更好（AMASS: 0.26 vs 0.28；但CMU相反）
- RLPF-TA在MuJoCo Succ上更好（AMASS: 0.66 vs 0.63）

**Intuition**: MA anchor到ground truth motion，FID衡量的是distribution similarity，所以MA自然FID更好。TA anchor到text，R@K衡量text-motion retrieval，所以TA自然R@K更好。但Table 2显示TA的Succ反而更高，可能是因为TA允许motion偏离GT，找到physically easier的等效motion——这其实是个有意思的discovery。

---

## 7. 实验结果深度分析

### 7.1 Q1: Feasibility Improvement (Table 2, 3)

**CMU Dataset IsaacGym Succ**:
- Base Model: 0.43
- SFT: 0.36（变差！）
- SFT-Filter: 0.57
- RLPF-MA: 0.95（提升120% vs base）
- RLPF-TA: 0.97

**MuJoCo Succ (cross-sim transfer)**:
- Base: 0.43
- SFT: 0.30
- SFT-Filter: 0.41
- RLPF-MA: 0.75
- RLPF-TA: 0.61

**Key observations**:
1. SFT反而降低Succ——这是因为SFT在full dataset上训练，reinforce了原始distribution的infeasible motion
2. SFT-Filter有提升但有限，因为只是过滤数据，无法shift distribution
3. RLPF在IsaacGym到MuJoCo的sim-to-sim transfer上loss更大（0.95→0.75），说明IsaacGym的dynamics比MuJoCo更"容易"，符合[Humanoid-Gym](https://arxiv.org/abs/2404.05695)的发现

### 7.2 Q2: Reward Analysis (Table 4, 5)

**RLPF-w/o track**:
- CMU IsaacGym Succ: 0.32（甚至低于base 0.43）
- CMU MuJoCo Succ: 0.24

这证实没有tracking reward，仅靠alignment reward无法提升physical feasibility——alignment reward只能维持semantic，不能enforce物理约束。

**RLPF-PHC**:
- CMU IsaacGym Succ: 0.91
- CMU MuJoCo Succ: 0.65

PHC（[Perpetual Humanoid Control, arXiv:2305.09429](https://arxiv.org/abs/2305.09429)）是另一个keypoint-based tracking policy。RLPF-PHC的Succ略低于RLPF-Full（Exbody2-based），说明tracking policy的选择影响最终性能。Paper提到PHC无法直接deploy到real robot，需要transfer到Exbody2-based policy——这个transfer gap是0.91→0.65的部分原因。

### 7.3 Q3: Alignment Verification (Table 6)

**RLPF-w/o align**的FID:
- CMU: 32.53（vs RLPF-MA 3.61）
- AMASS: 41.97（vs RLPF-MA 3.34）

R@1:
- CMU: 0.09（vs 0.26）
- AMASS: 0.07（vs 0.28）

**MMDist**:
- CMU: 7.39（vs 3.54）
- AMASS: 7.57（vs 3.75）

这是巨大的性能塌陷。Figure 3显示RLPF-w/o align生成的motion大多是standing——因为standing最容易track。这是经典的reward hacking：纯physical reward会让policy collapse到degenerate solution。

### 7.4 High-level Generation (Table 1)

注意RLPF的FID比Base Model高（CMU: 3.61 vs 2.35）。这看起来是regression，但实际上：

- FID衡量motion distribution与GT distribution的Fréchet distance
- RLPF把distribution从"physically infeasible region"shift到"feasible region"
- GT distribution本身包含很多infeasible motion
- 所以FID升高意味着RLPF的motion偏离了GT distribution，但这正是我们想要的——因为GT本身是physics-unaware的

**这个metric paradox值得注意**: 在physical-aware任务中，FID不再是单调越低越好的metric。RLPF-Full在IsaacGym Succ达到0.95/0.97，但FID比Base Model高3倍，trade-off是值得的。

---

## 8. 关键实现细节

### 8.1 Hyperparameters (Table 9)

- Model: LLaMA-2-7B
- Group size G = 20
- Max prompt length = 100
- Max completion length = 100
- Max grad norm = 0.1（非常保守）
- Reward weight tracking = 10
- Reward weight align = 2
- KL weight = 1.0
- KL type = k3

**Ratio分析**: tracking:align = 5:1。这表明physical feasibility比alignment更难达到，需要更大weight。但如果继续增大tracking weight，会引发alignment collapse（Figure 3所示）。

### 8.2 Three-stage Evaluation

1. **IsaacGym training**: 高度并行GPU仿真，适合大规模RL训练
2. **MuJoCo zero-shot transfer**: 验证cross-simulator generalization，因为MuJoCo dynamics更接近real world
3. **Real robot (Unitree G1)**: 最终deployment

这个三阶段protocol借鉴自[OmniH2O](https://arxiv.org/abs/2412.13196)和[ASAP](https://arxiv.org/abs/2502.01143)，是当前humanoid research的标准evaluation pipeline。

### 8.3 Robot Hardware Setup

- Robot: Unitree G1
- Onboard compute: Jetson Orin NX
- Policy inference: 50 Hz
- Low-level control: 200 Hz
- Communication: LCM ([Lightweight Communications and Marshalling](https://arxiv.org/abs/1004.4664))

50Hz policy + 200Hz low-level是humanoid常见配置——policy不需要太快，因为motion是smooth的；但low-level PD controller需要快loop来保证stability。

---

## 9. 与相关工作的联系

### 9.1 与RLHF的类比

| Component | RLHF (LLM) | RLPF (Motion) |
|-----------|-----------|---------------|
| Base model | LLM (GPT) | Large Motion Model (LLaMA2 + tokenizer) |
| Reward source | Human preference | Physics simulator + contrastive encoders |
| Algorithm | PPO | GRPO |
| KL penalty | Keep close to SFT model | Same |
| Failure mode | Reward hacking (sycophancy) | Reward hacking (standing collapse) |

### 9.2 与PhysDiff / ReinDiffuse的区别

- [PhysDiff](https://arxiv.org/abs/2305.12729): Physics-guided diffusion，在diffusion sampling时加physical constraint
- [ReinDiffuse](https://arxiv.org/abs/2403.18701): 用RL finetune diffusion model，但reward更局部（per-step physical violation）
- RLPF: finetune autoregressive LLM motion model，reward是trajectory-level binary success

RLPF的优势在于reward更holistic——直接看整个motion能否被robot执行，而不是per-frame的物理violations。

### 9.3 与InstructMotion的关系

[InstructMotion](https://arxiv.org/abs/2405.15541)同样用RL finetune motion generator，但它的reward是contrastive text-motion similarity，目标是improve generalization to novel text descriptions。RLPF的reward来自physical simulator，目标是enable real robot deployment。两者正交，理论上可以combine。

---

## 10. Limitations与潜在改进

Paper的Limitations section提到："current framework's generalization is constrained by its frozen motion tracking policy, which is pre-trained on a limited static dataset."

这意味着：
1. 如果T2M生成out-of-distribution motion（如非常novel的动作），tracking policy可能失败
2. Tracking policy失败 → reward = 0 → RL无法学习这种motion的feasibility
3. **Future direction**: Joint training of motion generator + tracking policy，让两者co-evolve

我的额外思考：

1. **Binary reward的limitation**: $R_{tracking} \in \{0, 1\}$ 太sparse。可以改成continuous reward，例如 $R = 1 - \text{normalized}(MPJPE + MPKPE)$，提供gradient even when motion is partially feasible。

2. **Sim-to-real gap remaining**: Table 3显示IsaacGym Succ 0.92 → MuJoCo Succ 0.63，drop 30%。这suggest IsaacGym的reward过于optimistic，可能需要domain randomization或system identification（如ASAP）来进一步缩小gap。

3. **KL penalty的schedule**: 固定 $\beta = 1.0$ 可能suboptimal。早期RL训练时policy离ref远，可以容许大更新；后期应该收紧。可以采用curriculum schedule。

4. **Multi-objective reward balancing**: tracking:align = 5:1是fixed的。可以考虑Pareto frontier learning，让user在deployment时调节trade-off。

5. **Motion diversity preservation**: FID升高说明distribution shift。可以加入diversity reward（如motion embedding的entropy）防止mode collapse。

---

## 11. Reference Links

### 核心论文
- RLPF project page: https://beingbeyond.github.io/RLPF/
- Exbody2 (tracking policy基础): https://arxiv.org/abs/2412.13196
- H2O (retargeting方法): https://arxiv.org/abs/2406.10454
- PHC (备选tracking policy): https://arxiv.org/abs/2305.09429
- GRPO (RL算法): https://arxiv.org/abs/2402.03300

### Motion Generation基础
- T2M-GPT: https://arxiv.org/abs/2301.06052
- MotionGPT: https://arxiv.org/abs/2306.10800
- MotionGPT-2: https://arxiv.org/abs/2410.21747
- MDM (Motion Diffusion Model): https://arxiv.org/abs/2209.14916
- MotionDiffuse: https://arxiv.org/abs/2208.15001
- PhysDiff: https://arxiv.org/abs/2305.12729
- ReinDiffuse: https://arxiv.org/abs/2403.18701
- InstructMotion: https://arxiv.org/abs/2405.15541

### 基础设施与数据
- HumanML3D: https://arxiv.org/abs/2205.04719
- AMASS: https://amass.is.tue.mpg.de/
- MotionX: https://arxiv.org/abs/2307.00818
- CMU Mocap: http://mocap.cs.cmu.edu/
- SMPL: https://smpl.is.tue.mpg.de/
- VQ-VAE: https://arxiv.org/abs/1711.00937

### RL基础
- PPO: https://arxiv.org/abs/1707.06347
- DAgger: https://arxiv.org/abs/1011.0686
- RLHF (Summarization): https://arxiv.org/abs/2009.01325

### 仿真与硬件
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- MuJoCo: https://mujoco.org/
- Unitree G1: https://www.unitree.com/g1/
- LCM: https://lcm-proj.github.io/
- Humanoid-Gym (sim2real): https://arxiv.org/abs/2404.05695
- ASAP (sim2real): https://arxiv.org/abs/2502.01143

### LLM基础
- LLaMA2: https://arxiv.org/abs/2307.09288

---

## 12. 总结

RLPF的优雅之处在于把physical feasibility作为RL reward，而不是作为生成时的硬约束或post-hoc filtering。这种思路与LLM领域的RLHF一脉相承——用RL的distribution shifting能力，在不损失pre-training capabilities的前提下，align到下游需求。

对你而言，作为Li Tang/Li Auto的advisor，这篇paper在thinking robot的roadmap中有几个关键启示：

1. **Generative model + Physical reward是scalable pattern**: 不仅是motion，任何需要physical feasibility的generation任务（grasp pose, trajectory, manipulation sequence）都可以用这种pattern。

2. **Binary sparse reward在GRPO下能work**: 因为group-relative advantage不需要dense reward，只要group内有variation就能学。这对real robot RL意义重大，因为binary success通常最容易定义和measure。

3. **Alignment reward防止reward hacking是必须的**: 这点在robotics RL中常被忽略。单纯physical reward会导致policy collapse到degenerate behavior，必须搭配semantic/任务级别的reward。这跟你之前强调的"thinking robot需要语义grounding"完全一致。

4. **Three-stage sim-to-real evaluation protocol**值得标准化：IsaacGym训练 → MuJoCo cross-sim验证 → Real deployment。这是当前humanoid research的gold standard。

5. **Frozen tracking policy是limitation**: 未来需要co-training，让tracking policy能handle OOD motion。这与thinking robot需要continuously expand capability range的需求对齐。

如果你想进一步深挖某个组件（比如GRPO在motion token这种离散空间的specifics，或者tracking policy的具体reward shaping），我可以展开。
