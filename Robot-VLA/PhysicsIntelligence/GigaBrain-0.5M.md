---
source_pdf: GigaBrain-0.5M.pdf
paper_sha256: a2debbb90f494616b511ffa9dcd0fcbbfc2260fcb1972f174953f21daa68b4b3
processed_at: '2026-08-04T21:42:20-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲GigaBrain-0.5M*

Andrej, 我换个方式讲, 像咱在NeurIPS走廊里白板上画图那种感觉。

---

## 1. 这篇paper在解决什么问题?

想象你教一个robot做咖啡。传统的VLA模型(像π₀, OpenVLA这些)的工作方式是:

> "看到现在的画面 + 听到指令 → 直接输出接下来10个动作"

问题在哪? Robot执行到第7步的时候, 咖啡机state已经drift了, 但模型还在用第1步的"plan"硬执行。这就像你开车只看后视镜, 不看前面。每个action的小error累积起来, long-horizon task必崩。

**根本原因**: Policy的conditioning信息太薄了。它只看到now, 没看到future。它不知道"如果我执行这个action, 5秒后世界会变成什么样"。

Video world model(像Wan2.2, Cosmos这种在web-scale视频上pretrain的)恰好会predict future——给它当前画面, 它能"想象"出未来几秒的画面。GigaBrain团队的想法就是:

> **让world model当policy的"望远镜", policy每次decision之前先看看WM预测的未来, 再决定现在该怎么动。**

这就是RAMP (Reinforcement leArning via world Model-conditioned Policy)。

参考: 
- World model for robot learning: https://arxiv.org/abs/2505.12705
- π₀ VLA: https://arxiv.org/abs/2410.24164

---

## 2. 三个版本的演进: 0 → 0.5 → 0.5M*

为了讲清楚, 先理清三个version的关系:

### GigaBrain-0 (foundation)
基础VLA架构: PaliGemma-2 vision-language encoder + DiT action head + flow matching。Pretrain在10K+小时robot data上。Project: https://arxiv.org/abs/2510.19430

### GigaBrain-0.5 (加CoT)
在0的基础上加了**Embodied Chain-of-Thought**: 输出action之前, 模型先生成一段reasoning——subgoal language ("现在要去拿杯子") + discrete action tokens + 2D trajectory keypoints (10个关键点)。这就像LLM的CoT, 但embodied在空间维度上。

Loss function (paper公式1) 有三项:
$$\mathcal{L} = \underbrace{-\sum_j M_{\text{CoT},j} \log p_\theta(x_{j+1} | x_{1:j})}_{\text{CoT language loss}} + \underbrace{\|\epsilon - a_{\text{chunk}} - f_\theta(a_{\text{chunk}}^{\tau,\epsilon})\|^2}_{\text{Flow matching action loss}} + \underbrace{\lambda \|\text{GRU}(\hat{\mathbf{t}}_{1:10}) - \mathbf{t}_{1:10}\|^2}_{\text{2D trajectory loss}}$$

- $M_{\text{CoT},j}$: mask, 只在CoT tokens上算language loss
- $\tau \in [0,1]$: flow matching的noise level
- $a_{\text{chunk}}^{\tau,\epsilon} = \tau \cdot a_{\text{chunk}} + (1-\tau)\epsilon$: linear interpolation between clean action和noise
- $\hat{\mathbf{t}}_{1:10}, \mathbf{t}_{1:10}$: predicted和ground-truth的2D trajectory keypoints

Knowledge Insulation机制防止language loss和action loss在shared backbone上gradient打架。Ref: https://arxiv.org/abs/2505.23705

### GigaBrain-0.5M* (加RAMP)
在0.5的基础上加入world model-conditioned RL。这是本文的核心贡献。

---

## 3. RAMP的核心思想: 从"reactive"到"prospective"

### 3.1 关键比较: RAMP vs RECAP

RECAP是Physical Intelligence在 $\pi_{0.6}^*$ (https://arxiv.org/abs/2511.14759)里提出的方法。它的做法:

> Policy的input除了observation, 还加一个binary advantage signal $I \in \{0, 1\}$: "这个action是好是坏"。

这就像告诉robot: "你刚才那个动作, 不错" 或 "不行, 重来"。问题是这个信号太稀疏了——只有1个bit的信息量。

RAMP的做法:

> Policy的input除了observation和binary advantage, **还加上world model预测的future state latent $\mathbf{z}$**。

这就像告诉robot: "你刚才那个动作不错, 而且5秒后咖啡机会变成这样(显示一张'想象的画面'), 你现在的action应该往这个target对齐"。

数学上 (paper公式4):
$$\pi_{\text{RECAP}}(a | \mathbf{o}, I) = \int_z \pi_{\text{RAMP}}(a | \mathbf{o}, \mathbf{z}, I) p(\mathbf{z} | \mathbf{o}, I) d\mathbf{z}$$

**人话**: RECAP是把所有可能的future平均掉了 (marginalization), 变成"对未来瞎猜的平均策略"。RAMP显式conditioning on具体预测的future, 变成"针对具体未来的精确规划"。

Information theory上:
$$H(a | \mathbf{o}, \mathbf{z}, I) \leq H(a | \mathbf{o}, I)$$

加了 $\mathbf{z}$之后, action generation的不确定性严格降低。因为 $\mathbf{z}$注入了dense的几何结构和physical dynamics priors, 而 $I$ 只有1 bit。

### 3.2 四阶段循环

RAMP不是一次性训练, 是四个stage循环迭代:

```
Stage 1: Pretrain world model (能预测future state + value)
    ↓
Stage 2: 用WM的预测conditioning policy, SFT policy
    ↓
Stage 3: 部署policy到real robot, 人在旁边看着, 失败时干预, 收集rollout data
    ↓
Stage 4: 用新数据jointly update WM + policy
    ↓
回到Stage 3, 循环
```

这是**self-improving closed loop**: policy越强, autonomous rollout数据质量越高, 下次update更有效, policy更强。

---

## 4. Stage 1: World Model怎么训的?

### 4.1 Reward设计 (公式5)

$$r_t = \begin{cases} 0 & t = T \text{ and success} \\ -C_{\text{fail}} & t = T \text{ and fail} \\ -1 & \text{otherwise} \end{cases}$$

- $T$: episode终点
- $C_{\text{fail}}$: 大正数, 让失败episode的cumulative reward远低于成功
- 中间每步reward = -1, 鼓励policy尽快完成

Value function本质就是 **"negative expected steps-to-completion"**——越接近成功的state, value越高。

### 4.2 Latent Frame Injection (公式6)

$$\mathbf{s}_t = [\mathbf{z}_t; \Psi(v_t); \Psi(\mathbf{p}_t)]$$

- $\mathbf{z}_t \in \mathbb{R}^{H' \times W' \times C'}$: 未来4帧 $\{o_{t+12}, o_{t+24}, o_{t+36}, o_{t+48}\}$ 经VAE编码的visual latent
- $v_t \in \mathbb{R}$: scalar value estimate
- $\mathbf{p}_t \in \mathbb{R}^d$: proprioception (joint angles等)
- $\Psi(\cdot)$: spatial tiling, 把scalar复制broadcast到 $H' \times W'$空间维度

**人话**: 把value和proprioception这两个低维信号"伪装"成image latent的样子, 通过channel-wise concat和visual latents拼一起, 送进DiT。WM的spatiotemporal self-attention就能统一reasoning "画面+value+robot状态"。

### 4.3 WM backbone和loss

Backbone: Wan2.2 (https://arxiv.org/abs/2503.20314), 用flow matching训练。

Loss (公式7):
$$\mathcal{L}_{\text{WM}} = \mathbb{E}_{\mathcal{D}, \tau, \epsilon} \left[ \|\mathcal{W}_\phi(\mathbf{s}_{\text{future}}^{\tau,\epsilon}) - (\mathbf{s}_{\text{future}} - \epsilon)\|^2 \right]$$

- $\mathbf{s}_{\text{future}}^{\tau,\epsilon} = \tau \mathbf{s}_{\text{future}} + (1-\tau)\epsilon$: noised future latent
- $\tau \sim \mathcal{U}(0,1)$, $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
- Target是 $(\mathbf{s}_{\text{future}} - \epsilon)$, optimal transport路径上的constant velocity

训练数据: 4K小时real robot manipulation。

---

## 5. Stage 2: Policy怎么用WM的预测?

Policy接收两个auxiliary signals from WM:
1. **Future state tokens** $\mathbf{z}_{\text{future}}$: 通过MLP对齐dimension
2. **Value estimates** $v_t$: 转成advantage

### 5.1 N-step TD Advantage (公式8)

$$A(\mathbf{s}_t, a_t) = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n v_{t+n} - v_t$$

- $\gamma$: discount factor
- $n$: TD步数
- $v_t, v_{t+n}$: WM预测的两个时刻value
- $r_{t+k}$: 真实reward (来自公式5)

这是标准n-step TD。用 $n$ 步真实reward + bootstrap term $\gamma^n v_{t+n}$, 减去baseline $v_t$。比1-step TD bias小, 比Monte Carlo variance小。

### 5.2 Binary discretization

$$I = \mathbb{1}(A(\mathbf{s}_t, a_t) > \epsilon)$$

把advantage二值化, 简化conditioning。

### 5.3 Training objective (公式3)

$$\mathcal{L}(\theta) = \mathbb{E}_\mathcal{D} \left[ -\log \pi_\theta(a | \mathbf{o}, \mathbf{z}, l) - \alpha \log \pi_\theta(a | I, \mathbf{o}, \mathbf{z}_t, l) \right]$$

- 第一项: 无条件policy fitting (类似BC, 但conditioning在augmented state上)
- 第二项: 条件policy fitting, 在 $I=1$的improvement events上学习preference
- $\alpha$: balancing weight

### 5.4 Stochastic Masking: 关键trick

Training时以 $p=0.2$的概率随机把WM tokens mask掉。这有两个作用:

1. **防over-reliance**: 防止policy完全依赖WM的synthetic signal, 强制它保持用observation alone也能work的能力
2. **Flexible deployment**: 推理时可以bypass WM (efficient mode) 或激活WM (standard mode)

这是Dropout思想在conditioning signal上的应用, 让policy更robust。

---

## 6. Stage 3: Human-in-the-Loop Rollout

### 6.1 为什么不直接teleop?

传统teleop的问题: human demonstration的action distribution和policy自己的action distribution有gap。Policy学习模仿human, 但自己生成action时分布对不上, 容易OOD。

HILR的做法:

> Policy先自己跑, 失败的时候human intervene。这样大部分trajectory是policy自己生成的 (native distribution), 只有critical moment是human correction。

### 6.2 数据有两种信号

1. **Autonomous成功执行**: 强化policy的好行为
2. **Expert correction**: 教policy如何从failure中recover——这对long-horizon task极其重要, 因为compounding error最终会推policy进入OOD state

### 6.3 Temporal smoothing

HILR软件自动检测intervention boundaries, 移除transitional artifacts, 保证trajectory时间连贯性。这样policy update时不会因为discontinuity而unstable。

---

## 7. Stage 4: Joint Update防止Advantage Collapse

**关键insight**: 必须jointly update WM和policy, 不能只update policy。

如果只update policy, policy快速适应新数据分布, 但WM还停留在旧distribution。WM的value prediction和policy当前behavior的advantage会collapse到0附近, preference signal消失, RL就废了。

所以WM和policy一起用HILR数据 + base数据jointly train。Masking probability $p=0.2$同时应用于advantage indicator $I$和future latent $\mathbf{z}_{\text{future}}$, 保持training consistency。

---

## 8. Inference: 两种模式

部署时固定 $I=1$ (optimistic control)。对 $\mathbf{z}$, 由于stochastic masking的architectural decoupling:

### Efficient Mode
- Bypass world model
- Attention mask让policy看不见future latent tokens
- 最大化inference frequency, 适合简单task

### Standard Mode  
- WM active生成 $\mathbf{z}$
- Policy看见全部future latent tokens
- 适合复杂long-horizon task, 用prospective context做planning

这种设计很elegant: 训练时20% dropout让policy学会双模式, 推理时按需切换。

---

## 9. 实验数据解读

### 9.1 Pretraining规模

| Item | Value |
|---|---|
| Total data | 10,000+ hours |
| World model-generated data | 6,000+ hours |
| Real-robot data | ~4,000 hours |
| Batch size | 3,072 |
| Training steps | 100,000 |
| Optimization | FSDP v2 (selective sharding) |

FSDP v2 selective sharding: 只对SiglipEncoderLayer和Gemma2DecoderLayerWithExpert的前16层sharding, 平衡memory和通信。

### 9.2 Foundation Model Performance

GigaBrain-0.5在8个internal task上SOTA:
- **Juice Preparation**: 100% (vs GigaBrain-0 90%, +10%)
- **Box Packing**: +10% over $\pi_{0.5}$
- **Espresso Preparation**: +20% over $\pi_{0.5}$
- **Paper Towel Prep / Laundry Folding / Laundry Collection**: 80%+, 分别 +15%/+5%/+10% over $\pi_{0.5}$

RoboChallenge benchmark (https://robochallenge.ai/home): 中间版本GigaBrain-0.1排行第一, avg success rate 51.67%, 比 $\pi_{0.5}$ (42.67%) 高9%。RoboChallenge有20个physical robots, 4平台 (UR5, Franka, ARX5, ALOHA), 30个standardized tasks, 736GB dataset。

### 9.3 Value Prediction比较 (Table 1)

| Model | Inference (s) | MAE↓ | Kendall↑ |
|---|---|---|---|
| VLM-based | 0.32 | 0.0683 | 0.7972 |
| WM (value only) | 0.11 | 0.0838 | 0.7288 |
| WM (state+value) | 0.25 | 0.0621 | 0.8018 |

**三个发现**:
1. VLM-based最慢 (0.32s), SigLIP encoder开销大
2. WM value-only最快 (0.11s) 但精度最差 (Kendall=0.7288), 说明纯value modeling没充分利用WM的future prediction能力
3. **Joint state+value是最优balance**: Kendall 0.8018最高, MAE 0.0621最低, 速度0.25s可接受

**核心insight**: Value本质是"未来reward的discounted sum", 必须知道future state才能准确估计value。所以future state prediction给value estimation提供grounding context, 两者必须jointly predict。

### 9.4 Multi-task Generalization (Figure 14)

四个task (Table Bussing, Laundry Folding, Paper Towel Prep, Box Packing):
- Single-task: 20K steps, batch 256
- Multi-task: 60K steps, batch 256, uniform混合

WM condition在两种setting都超过baseline。特别multi-task, gap随training扩大, Box Packing在step 20K达到~30%提升。说明WM conditioning促进cross-task knowledge transfer。

### 9.5 RAMP vs RL Baselines (Figure 15)

三个baseline:
1. **GigaBrain-0.5 + AWR** (https://arxiv.org/abs/1910.00177): offline RL with weighted imitation
2. **GigaBrain-0.5 + RECAP** (https://arxiv.org/abs/2511.14759): advantage-conditioned offline RL
3. **GigaBrain-0.5 + RAMP**: full method

RAMP在Box Packing, Espresso Preparation, Laundry Folding三个challenging task上达到near-perfect success rate, 比RECAP高约30% points。在Box Packing和Espresso Preparation上improvement最显著。

---

## 10. 整体Intuition: 为什么这个approach work?

### 10.1 从reactive到prospective

传统VLA: $\pi(a | o, l)$——只看now, 像开车只看后视镜。

RAMP: $\pi(a | o, \mathbf{z}, l)$——看now + WM预测的future, 像开车看前面+后视镜。

### 10.2 从sparse到dense conditioning

RECAP: 只有1-bit advantage signal, "好/坏"。
RAMP: 1-bit advantage + dense future state representation $\mathbf{z}$。从"coarse credit assignment"到"precise planning target"。

### 10.3 Self-improving closed loop

```
policy强 → rollout数据质量高 → WM + policy update更有效 → policy更强
```

这和AlphaGo的self-play思想一致, 只是加了HILR的human correction作为bootstrapping。

### 10.4 Stochastic masking的妙处

$p=0.2$的masking让policy学会两种模式:
- 有WM时: 用future prediction做precise planning
- 没WM时: 用observation alone做reactive control

推理时按task复杂度切换, 既保real-time性能, 又保long-horizon capability。

---

## 11. 我的几个观察

### 11.1 WM Fidelity的天花板

RAMP的policy quality受限于WM的prediction质量。论文用single denoising step降overhead, 但可能牺牲了prediction精度。OOD state下WM预测失准, policy的conditioning就会带noise。这是model-based RL的经典问题。

### 11.2 HILR的scaling bottleneck

虽然HILR比纯teleop成本低, 但还需要human expert intervention。真正self-evolution需要减少human involvement。论文Conclusion里提到future direction是autonomous data curation, 说明他们自己也意识到这个limitation。

### 11.3 Advantage Collapse的sensitivity

Stage 4的joint training防止advantage collapse, 但论文没量化分析这个问题的severity和hyperparameter sensitivity。实际部署中, 这个balance很微妙。

### 11.4 Long-horizon的定义

Paper claim "long-horizon execution without failure", 但internal evaluation的8个task最长可能是minutes级别。真正的long-horizon (hours级别, 比如做一桌子菜)还没验证。

### 11.5 Inference speed的real-time concern

WM (state+value) inference 0.25s, 对real-time control (通常要求10-30Hz)还是瓶颈。Efficient mode bypass WM可以高速, 但失去foresight。这个trade-off需要更细的task-dependent scheduling。

---

## 12. 可能的extension方向

1. **Hierarchical WM**: 当前预测fixed horizon (12/24/36/48 frames), 可以multi-scale: 短期精确+长期抽象
2. **Counterfactual reasoning**: 用WM做"如果action A vs B, 未来怎样不同", 提供更rich的preference signal
3. **Active WM query**: Policy主动决定何时query WM (类似Active Inference), 简单action bypass, 复杂action query
4. **Cross-embodiment RAMP**: 当前在PiPER arm和G1 humanoid上, 能否scale到更多morphology?
5. **Autoregressive WM**: 替换diffusion-based Wan2.2为autoregressive token-based WM, 可能加速inference

---

## 13. 总结一句话

**GigaBrain-0.5M\*的核心insight**: 把VLA从reactive control提升到prospective planning, 关键在于让policy condition在world model预测的future state representation上, 同时用stochastic masking保持policy在with/without WM两种模式下的robustness。RAMP的四阶段iterative pipeline + HILR data collection实现了self-improving closed loop。数学上证明RECAP是其special case, 信息论上证明引入 $\mathbf{z}$降低action generation的conditional entropy。实验在30个RoboChallenge task上SOTA (51.67%), 在3个challenging long-horizon task上比RECAP高约30% points。

整体是个很elegant的工作, 把model-based RL和VLA做了deep integration, 为embodied AI的long-horizon reasoning提供了concrete path。

---

## References

- Project page: https://gigabrain05m.github.io
- GigaBrain-0: https://arxiv.org/abs/2510.19430
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- $\pi_{0.6}^*$ RECAP: https://arxiv.org/abs/2511.14759
- Flow Matching: https://arxiv.org/abs/2210.02747
- DreamGen: https://arxiv.org/abs/2505.12705
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- Wan2.2: https://arxiv.org/abs/2503.20314
- Knowledge Insulation: https://arxiv.org/abs/2505.23705
- AWR: https://arxiv.org/abs/1910.00177
- DAgger: https://arxiv.org/abs/1106.0245
- RoboChallenge: https://robochallenge.ai/home
- PaliGemma-2: https://arxiv.org/abs/2412.03555
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- V-JEPA 2: https://arxiv.org/abs/2506.09985

---

# GigaBrain-0.5M* 深度技术解析

Andrej, 这篇paper来自GigaAI团队，是GigaBrain-0.5的延伸工作。核心贡献在于将**world model-based reinforcement learning**整合进VLA训练，提出RAMP (Reinforcement leArning via world Model-conditioned Policy)框架。我会从intuition、architecture、math、experiment四个维度逐一拆解。

---

## 1. Motivation: 主流VLA的根本缺陷

当前主流VLA模型(π₀, OpenVLA, GR-3, CogACT, RDT-1B等)的架构本质上是一种**reactive control system**: 给定当前observation $o$ + language instruction $l$, 直接预测action chunk $a_{\text{chunk}}$。这种架构在long-horizon procedural task(比如做一杯咖啡需要12个sequential subtask)上会暴露出**compounding error**: 每一步的小偏差累积起来最终导致任务失败。

更深层的issue在于: VLA模型预测action时conditioning information过于稀薄——只有current frame + text, 没有"未来会发生什么"的prospective context。理论上要让agent具备foresight, 至少需要它对future state distribution有显式建模。

Video world model(像Wan2.2, Cosmos, V-JEPA 2等)恰好提供这种能力: 在web-scale视频上pretrain后, 它们已经内化了physical dynamics priors和spatiotemporal reasoning。RAMP的核心想法就是:**用world model预测的future latent state + value作为policy的conditioning signal, 把"reactive"变成"prospective"**。

参考: 
- VLA架构综述 https://arxiv.org/abs/2410.24164 (π₀)
- World models for robotics https://arxiv.org/abs/2505.12705 (DreamGen)

---

## 2. GigaBrain-0.5 基础架构

GigaBrain-0.5是GigaBrain-0的升级版, 架构继承自GigaBrain-0 (https://arxiv.org/abs/2510.19430), 设计上采用**Mixture-of-Transformers** (MoT) backbone:

### 2.1 模块组成

| Module | Implementation | 作用 |
|---|---|---|
| Vision-Language Encoder | PaliGemma-2 (https://arxiv.org/abs/2412.03555) | 编码image + text |
| Action Head | DiT + Flow Matching (https://arxiv.org/abs/2210.02747) | 生成continuous action chunk |
| 2D Trajectory Decoder | lightweight GRU | 回归2D keypoint trajectory $\mathbf{t}_{1:10}$ |
| CoT Stream | autoregressive VLM head | 输出subgoal language + discrete action tokens |

### 2.2 Embodied Chain-of-Thought (Embodied CoT)

GigaBrain-0.5在生成action之前先生成一个**Embodied CoT**: 包含autoregressive subgoal language、discrete action tokens (来自FAST tokenizer, https://arxiv.org/abs/2501.09747)、以及2D manipulation trajectory keypoints。这三者构成policy的中间reasoning layer, 类比于LLM的CoT, 但是embodied在spatial-temporal维度上。

### 2.3 Unified Loss (公式1详解)

$$\mathcal{L} = \mathbb{E}_{\mathcal{D}, \tau, \epsilon} \left[ -\sum_{j=1}^{n-1} M_{\mathrm{CoT},j} \log p_\theta(x_{j+1} \mid x_{1:j}) + \|\epsilon - a_{\mathrm{chunk}} - f_\theta(a_{\mathrm{chunk}}^{\tau,\epsilon})\|^2 + \lambda \|\mathrm{GRU}(\hat{\mathbf{t}}_{1:10}) - \mathbf{t}_{1:10}\|^2 \right]$$

**变量含义**:
- $\mathcal{D}$: training dataset (10K+ hours real robot data)
- $\tau \in [0,1]$: flow-matching timestep, 控制noise level
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: Gaussian noise vector
- $M_{\mathrm{CoT},j} \in \{0,1\}$: per-token mask, 表示position $j$是否属于CoT reasoning stream (subgoal language或discrete action)
- $a_{\mathrm{chunk}}^{\tau,\epsilon} = \tau \cdot a_{\mathrm{chunk}} + (1-\tau) \cdot \epsilon$: noised action chunk, 线性插值
- $\hat{\mathbf{t}}_{1:10}, \mathbf{t}_{1:10}$: predicted和ground-truth的2D trajectory keypoints (10个关键点)
- $\lambda$: hyperparameter balancing trajectory regression loss

**第一项**: CoT reasoning stream的autoregressive language modeling loss (类似LLM的next-token prediction, 但只在CoT tokens上计算)

**第二项**: Flow matching的velocity prediction loss. $f_\theta$预测从noise到clean action的velocity field, target是 $(a_{\mathrm{chunk}} - \epsilon)$, 沿optimal transport路径的constant velocity. 这与DDPM不同, flow matching用linear interpolation避免stochastic noise scheduling.

**第三项**: GRU decoder回归2D trajectory keypoints的MSE loss.

**Knowledge Insulation机制** (https://arxiv.org/abs/2505.23705): 防止language modeling loss和action prediction loss在shared backbone上发生gradient interference, 通过architectural isolation让两条pathway独立优化.

---

## 3. RAMP框架: 核心贡献

RAMP的全称是Reinforcement leArning via world Model-conditioned Policy, 整个pipeline分四阶段循环迭代, 见paper Figure 2.

### 3.1 数学推导: 从KL-regularized RL到training objective

#### 3.1.1 Augmented State Space

传统RL的state是 $s \in \mathcal{S}$. RAMP把state扩展到augmented space:
$$\mathbf{S} = (\mathbf{o}, \mathbf{z}, l)$$

其中:
- $\mathbf{o}$: current observation (RGB frame)
- $\mathbf{z}$: world model预测的future latent state (spatiotemporal encoding)
- $l$: language instruction

**Intuition**: 传统VLA的policy $\pi(a | \mathbf{o}, l)$ 只看now, RAMP的policy $\pi(a | \mathbf{o}, \mathbf{z}, l)$ 同时看now和predicted future.

#### 3.1.2 KL-Regularized Optimal Policy (公式2)

$$\hat{\pi}(a \mid \mathbf{S}) \propto \pi_{\mathrm{ref}}(a \mid \mathbf{S}) \exp\left(\frac{A^{\pi_{\mathrm{ref}}}(\mathbf{S}, a)}{\beta}\right)$$

**变量含义**:
- $\pi_{\mathrm{ref}}$: reference policy (pretrained GigaBrain-0.5)
- $A^{\pi_{\mathrm{ref}}}(\mathbf{S}, a)$: advantage function w.r.t. $\pi_{\mathrm{ref}}$, 衡量action $a$在state $\mathbf{S}$下相对reference policy的好坏
- $\beta$: temperature/KL penalty coefficient, 控制policy偏离 $\pi_{\mathrm{ref}}$的程度

这个公式来自regularized RL的理论 (类似TRL/RLHF的closed-form solution). 当 $\beta \to 0$时policy完全greedy地最大化advantage, 当 $\beta \to \infty$时policy完全退化为reference.

#### 3.1.3 Binary Improvement Indicator + Bayesian Reformulation

直接估计 $\exp(A^{\pi_{\mathrm{ref}}}(\mathbf{S},a)/\beta)$ 数值上不稳定. 作者引入binary improvement indicator $I \in \{0,1\}$, 假设:
$$p(I \mid a, \mathbf{S}) \propto \exp\left(\frac{A^{\pi_{\mathrm{ref}}}(\mathbf{S}, a)}{\beta}\right)$$

应用Bayes' theorem, 把intractable的exponential advantage term转化为condition probability ratio:
$$\exp\left(\frac{A^{\pi_{\mathrm{ref}}}(\mathbf{S},a)}{\beta}\right) \propto \frac{\pi_{\mathrm{ref}}(a \mid I, \mathbf{S})}{\pi_{\mathrm{ref}}(a \mid \mathbf{S})}$$

#### 3.1.4 Training Objective (公式3)

$$\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_\mathcal{D} \left[ -\log \pi_\boldsymbol{\theta}(a \mid \mathbf{o}, \mathbf{z}, l) - \alpha \log \pi_\boldsymbol{\theta}(a \mid I, \mathbf{o}, \mathbf{z}_t, l) \right]$$

其中 $I = \mathbb{1}[A(\mathbf{o}, \mathbf{z}, l, a) > \epsilon]$ 是improvement signal (advantage超过threshold $\epsilon$).

**两项含义**:
- 第一项 $-\log \pi_\theta(a|\mathbf{o},\mathbf{z},l)$: unconditional policy fitting (类似behavior cloning, 但是conditioning在augmented state上)
- 第二项 $-\alpha \log \pi_\theta(a|I, \mathbf{o}, \mathbf{z}_t, l)$: conditional policy fitting, 只在improvement events $I=1$上学习preference结构

$\alpha$ 是balancing weight, 控制preference signal的影响.

### 3.2 RAMP vs RECAP: 理论关系 (公式4)

RECAP (https://arxiv.org/abs/2511.14759, 在 $\pi_{0.6}^*$中使用)是RAMP的degenerate special case:

$$\pi_{\mathrm{RECAP}}(a \mid \mathbf{o}, I) = \int_z \pi_{\mathrm{RAMP}}(a \mid \mathbf{o}, \mathbf{z}, I) p(\mathbf{z} \mid \mathbf{o}, I) \, d\mathbf{z}$$

**Intuition**: RECAP把 $\mathbf{z}$ marginalize掉了, 变成对所有possible futures的"平均猜测". RAMP显式conditioning on $\mathbf{z}$, 把问题从"对未来的average guess"变成"针对具体物理状态的precise planning".

**Information-theoretic论证**:
$$H(a \mid \mathbf{o}, \mathbf{z}, I) \leq H(a \mid \mathbf{o}, I)$$

引入 $\mathbf{z}$后action generation的conditional entropy严格降低. RECAP只靠sparse binary advantage $I \in \{0,1\}$做coarse credit assignment, RAMP用 $\mathbf{z}$注入dense geometric structure和physical dynamics priors.

这点非常符合intuition: 想象做咖啡, RECAP只能告诉你"这个action是好是坏", RAMP还能告诉你"5秒后咖啡机会变成这个样子, 你现在的action应该向那个target对齐".

---

## 4. Stage 1: World Model Pre-training

### 4.1 Reward Design (公式5)

$$r_t = \begin{cases} 0 & \text{if } t = T \text{ and episode succeeds} \\ -C_{\mathrm{fail}} & \text{if } t = T \text{ and episode fails} \\ -1 & \text{otherwise} \end{cases}$$

**变量含义**:
- $T$: terminal timestep
- $C_{\mathrm{fail}}$: large positive constant, 确保失败episode累计reward远低于成功episode

这种sparse reward设计鼓励policy: (1) 优先完成任务, (2) 最小化execution time. Value function对应negative expected steps-to-completion.

### 4.2 Latent Frame Injection Strategy

作者借鉴Kim et al. 2026 (https://arxiv.org/abs/2601.16163, Cosmos Policy)的latent frame injection, 把value signal作为额外latent frame拼接进visual latent, 不需要修改DiT backbone架构.

#### 4.2.1 Latent State Construction (公式6)

$$\mathbf{s}_t = [\mathbf{z}_t; \Psi(v_t); \Psi(\mathbf{p}_t)]$$

**变量含义**:
- $\mathbf{z}_t \in \mathbb{R}^{H' \times W' \times C'}$: visual latents, 由pre-trained VAE编码future visual observations $\{\mathbf{o}_{t+i}\}_{i \in \{12,24,36,48\}}$得到 (4个未来帧, 间隔12 frames)
- $v_t \in \mathbb{R}$: 当前value estimate (scalar)
- $\mathbf{p}_t \in \mathbb{R}^d$: proprioceptive state (低维, robot joint angles等)
- $\Psi(\cdot)$: spatial tiling projection, 把低维向量复制broadcast到spatial dimension $\mathbb{R}^{H' \times W'}$匹配visual latents shape
- $[\cdot;\cdot]$: channel-wise concatenation

**Intuition**: 通过spatial tiling, scalar value和proprioception都被broadcast成spatial feature map, 可以直接和visual latents在channel维度concat, 然后输入DiT, 利用其spatiotemporal self-attention统一reasoning.

### 4.3 World Model Backbone: Wan2.2

World model $\mathcal{W}_\phi$使用Wan2.2 (https://arxiv.org/abs/2503.20314)作为backbone, 训练用flow matching.

### 4.4 World Model Loss (公式7)

$$\mathcal{L}_{\mathrm{WM}} = \mathbb{E}_{\mathcal{D}, \tau, \epsilon} \left[ \left\| \mathcal{W}_\phi(\mathbf{s}_{\mathrm{future}}^{\tau,\epsilon}) - (\mathbf{s}_{\mathrm{future}} - \epsilon) \right\|^2 \right]$$

**变量含义**:
- $\mathbf{s}_{\mathrm{future}}^{\tau,\epsilon} = \tau \mathbf{s}_{\mathrm{future}} + (1-\tau) \epsilon$: noised latent state sequence, linear interpolation between noise和ground-truth future latent
- $\tau \sim \mathcal{U}(0, 1)$: uniform采样timestep
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: Gaussian noise
- $(\mathbf{s}_{\mathrm{future}} - \epsilon)$: constant-velocity vector field沿optimal transport path from noise to data

**Intuition**: 这是标准的flow matching (rectified flow)目标. $\mathcal{W}_\phi$学习一个velocity field, 给定noised state和time $\tau$, 预测从noise到data的方向. Training用4K小时real robot manipulation data.

---

## 5. Stage 2: Policy Training with World Model Conditioning

Policy接收两个auxiliary signals:
1. **Future state tokens** $\mathbf{z}_{\mathrm{future}}$: 通过lightweight MLP对齐dimension
2. **Value estimates** $v_t$: 转化为advantage

### 5.1 N-step TD Advantage (公式8)

$$A(\mathbf{s}_t, a_t) = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n v_{t+n} - v_t$$

**变量含义**:
- $\gamma$: discount factor
- $n$: TD steps (n-step return)
- $v_t, v_{t+n}$: world model预测的两个时刻的value
- $r_{t+k}$: 真实reward (来自公式5)

**Intuition**: 这是标准的n-step temporal difference. 用 $n$ 步真实reward加bootstrap term $\gamma^n v_{t+n}$, 减去baseline $v_t$, 得到advantage. 这种估计比1-step TD variance大但bias小, 比Monte Carlo variance小但bias大, 是trade-off.

### 5.2 Binary Discretization

Advantage被离散化为binary indicator:
$$I = \mathbb{1}(A(\mathbf{s}_t, a_t) > \epsilon)$$

threshold $\epsilon$是个超参数. 这种离散化简化了conditioning并保持preference structure.

### 5.3 Stochastic Attention Masking (Key Trick)

Training时随机以probability $p = 0.2$ suppress world model tokens. 这有两个目的:
1. **Prevent over-reliance on synthetic WM signals**: 防止policy完全依赖world model
2. **Enable flexible deployment**: 训练时部分样本"看不见"WM conditioning, 推理时可以bypass WM

---

## 6. Stage 3: Human-in-the-Loop Rollout (HILR) Data Collection

### 6.1 为什么需要HILR?

Imitation learning的根本问题是distribution shift (Ross et al. 2011, https://arxiv.org/abs/1106.0245): training data来自expert demonstration, 但policy自己跑出来的state distribution会drift. DAgger-style (https://arxiv.org/abs/2106.06045) intervention可以缓解, 但传统teleoperation有action distribution gap.

### 6.2 HILR的设计要点

- **Autonomous execution + expert intervention hybrid**: Policy先自主执行, 失败时expert intervene
- **Reduced action distribution gap**: 因为policy生成action在自己native distribution内, 比模仿human demonstration的teleop更"贴近"policy的真正学习目标
- **Temporal smoothing**: HILR软件自动检测intervention boundaries, 移除transitional artifacts, 保证trajectory时间连贯性

### 6.3 关键insight

这个stage产生的数据包含两种信号:
1. **Autonomous成功执行**: 提供policy自己的成功轨迹, 强化good behavior
2. **Expert correction**: 提供从failure到recovery的corrective signal, 教policy如何从错误中recover

第二种信号对long-horizon task极其重要, 因为compounding error最终会导致policy进入OOD state, corrective data教它如何recover.

---

## 7. Stage 4: Continual Training

### 7.1 Joint Training of World Model + Policy

关键点: world model $\mathcal{W}_\phi$必须和policy一起用HILR数据 + base data联合训练, 否则会出现**advantage collapse toward zero**: $A(\mathbf{s}_t, a_t) \approx 0$.

**Intuition**: 如果只更新policy不更新WM, policy会快速适应新数据分布, 但WM还停留在旧distribution, 它的value prediction和policy当前behavior的advantage就会collapse到0附近, 失去preference信号.

### 7.2 Masking Consistency

Stochastic attention masking $p = 0.2$同时应用于:
- Advantage indicator $I$
- Future latent tokens $\mathbf{z}_{\mathrm{future}}$

保持pretraining和fine-tuning的architectural consistency, 避免inference时的distributional shift.

### 7.3 Self-improving Closed Loop

$$\text{policy improves} \to \text{autonomous rollouts cover harder behaviors} \to \text{higher-quality training data} \to \text{policy further improves}$$

这是经典的self-play / self-improvement loop, 类似AlphaGo的training paradigm.

---

## 8. Inference: 两种模式

部署时固定advantage indicator $I = 1$ (optimistic control). 对latent condition $\mathbf{z}$, 由于stochastic masking的architectural decoupling, 有两种执行模式:

### 8.1 Efficient Mode
- Bypass world model
- Attention mask配置成让policy看不见future latent tokens
- 最大化inference frequency
- 适用: simple task, real-time要求高的场景

### 8.2 Standard Mode
- World model active生成 $\mathbf{z}$
- Policy看见全部future latent tokens
- 利用prospective context做long-horizon planning
- 适用: complex task需要foresight

这种设计很巧妙: 训练时20% dropout让policy学会不依赖WM也能工作, 推理时根据任务复杂度动态切换.

---

## 9. 实验数据分析

### 9.1 Pre-training Setup

| Item | Value |
|---|---|
| Total data | 10,000+ hours |
| World model-generated data | 6,000+ hours |
| Real-robot data | ~4,000 hours |
| Batch size | 3,072 |
| Training steps | 100,000 |
| World model training data | 4K hours real robot data |
| Optimization | FSDP v2 (selective sharding) |

FSDP v2 selective sharding: 只对SiglipEncoderLayer模块和Gemma2DecoderLayerWithExpert的前16层做sharding, 是为了在per-GPU memory和通信开销之间找balance.

### 9.2 Foundation Model Performance (Internal 8 Tasks)

GigaBrain-0.5 vs baselines (Figure 4):
- **Juice Preparation**: GigaBrain-0.5 100% vs GigaBrain-0 90% (+10%)
- **Box Packing**: +10% over $\pi_{0.5}$
- **Espresso Preparation**: +20% over $\pi_{0.5}$
- **Paper Towel Preparation, Laundry Folding, Laundry Collection**: 80%+ success rate, +15%/+5%/+10% over $\pi_{0.5}$

GigaBrain-0.5在所有8个task上SOTA.

### 9.3 RoboChallenge Benchmark

RoboChallenge (https://robochallenge.ai/home) 是首个大规模real-robot evaluation platform:
- 20 physical robots
- 4 major platforms: UR5, Franka, ARX5, ALOHA
- 30 standardized manipulation tasks
- 736 GB dataset

GigaBrain-0.1中间版本排行第一, average success rate 51.67%, 比 $\pi_{0.5}$ (42.67%)高9%.

### 9.4 Value Prediction Comparison (Table 1)

| Model | Inference Time (s) | MAE↓ | MSE↓ | RMSE↓ | Kendall↑ |
|---|---|---|---|---|---|
| VLM-based | 0.32 | 0.0683 | 0.0106 | 0.1029 | 0.7972 |
| WM-based (value only) | 0.11 | 0.0838 | 0.0236 | 0.1433 | 0.7288 |
| WM-based (state+value) | 0.25 | 0.0621 | 0.0099 | 0.0989 | 0.8018 |

**关键发现**:
1. VLM-based最慢 (0.32s), 因为SigLIP visual encoder开销大
2. WM-based value-only最快 (0.11s) 但精度最差 (MAE=0.0838, Kendall=0.7288), 说明纯value modeling没有充分利用WM的future prediction能力
3. **Joint prediction (state+value) 是最优balance**: Kendall's tau 0.8018最高, MAE 0.0621最低, 推理速度 0.25s可接受

这个ablation说明: future state prediction提供"grounding context"对value estimation至关重要, value不能脱离future state独立预测好. 这背后intuition是: value本质上是"未来reward的discounted sum", 必须知道future state才能准确估计value.

### 9.5 World Model Conditioning for Multi-task Generalization (Figure 14)

实验设置: 四个task (Table Bussing, Laundry Folding, Paper Towel Preparation, Box Packing)
- Single-task: 20000 steps, batch 256
- Multi-task: 60000 steps, batch 256, uniform混合四任务

结果: WM condition在单任务和多任务场景都超过baseline. 特别在multi-task setting, 性能gap随training step逐渐扩大, 在step 20000时Box Packing上达到~30%的success rate提升. 这说明WM conditioning促进cross-task knowledge transfer.

### 9.6 RAMP vs RL Baselines (Figure 15)

三个baseline比较:
1. **GigaBrain-0.5 + AWR** (https://arxiv.org/abs/1910.00177): offline RL with weighted imitation learning
2. **GigaBrain-0.5 + RECAP** (https://arxiv.org/abs/2511.14759): advantage-conditioned offline RL, 是RAMP的ablated variant (没有state prediction)
3. **GigaBrain-0.5 + RAMP** (GigaBrain-0.5M*): full method

RAMP在Box Packing, Espresso Preparation, Laundry Folding三个challenging task上达到near-perfect success rate, 比RECAP baseline高约30% points. 特别在Box Packing和Espresso Preparation上, RAMP的improvement最为显著.

---

## 10. 整体Intuition总结

### 10.1 为什么RAMP比RECAP强?

RECAP的 $\pi(a|o,I)$ 只有binary advantage signal, 是"粗粒度的credit assignment". 想象训练agent做咖啡, RECAP告诉你"这个action好"或"这个action坏", 但不告诉你"5秒后咖啡机state会变成什么样, 你需要往那个方向去".

RAMP的 $\pi(a|o,z,I)$ 加入了future latent state $\mathbf{z}$, 提供了**dense geometric structure + physical dynamics prior**. Policy不需要implicit地"猜测未来"了, 它explicit地看到future state representation, 然后select action朝那个target对齐.

数学上, RECAP是对所有possible futures的marginalization (平均猜测), RAMP是conditioning on specific predicted future (precise planning).

### 10.2 为什么需要四阶段循环?

- Stage 1: WM必须先pretrain好, 才能给policy提供有意义的 $\mathbf{z}$和 $v$
- Stage 2: Policy需要先学会利用WM conditioning, 才能在real world有意义地rollout
- Stage 3: HILR收集的数据填补training distribution和deployment distribution之间的gap
- Stage 4: 用新数据jointly update WM和policy, 保持advantage signal不collapse

这个loop是**self-improving closed loop**: policy越强, rollout数据质量越高, WM和policy update越有效, policy更强.

### 10.3 Stochastic Masking的双重作用

$p=0.2$的stochastic masking看似小trick, 实际是critical design:
1. 训练时20%样本让policy"看不见"WM, 强制它保持用observation alone也能work的能力
2. 推理时可以bypass WM做efficient mode, 也可以激活WM做standard mode
3. 防止policy overfit到WM的synthetic signal, 失去对真实observation的sensitivity

这本质上是**Dropout-style regularization在conditioning signal上的应用**, 让policy更robust.

---

## 11. 与相关工作的对比

### 11.1 vs $\pi_{0.6}^*$ RECAP
- RECAP用VLM做value prediction, RAMP用WM做value prediction
- RECAP只conditioning on binary advantage, RAMP额外conditioning on future latent state
- 实验上RAMP比RECAP高约30% points

### 11.2 vs DreamGen (https://arxiv.org/abs/2505.12705)
- DreamGen: WM生成video -> IDM提取action, 两阶段pipeline
- RAMP: WM和policy joint training, 单一end-to-end framework
- RAMP避免了生成video fidelity对最终policy的影响

### 11.3 vs Cosmos Policy (https://arxiv.org/abs/2601.16163)
- Cosmos Policy: 直接从WM prediction映射到action sequence, 完全bypass policy network
- RAMP: WM作为policy的conditioning, policy仍是主架构
- RAMP的stochastic masking让policy可以在with/without WM两种模式间切换, 更flexible

### 11.4 vs GigaBrain-0 (https://arxiv.org/abs/2510.19430)
- GigaBrain-0是foundation, GigaBrain-0.5在architecture上加了Embodied CoT, GigaBrain-0.5M*加了RAMP
- GigaBrain-0.5在所有8个internal task上超越GigaBrain-0
- Juice Preparation: 100% vs 90%

---

## 12. 我的思考: Limitations和Potential Directions

### 12.1 Limitations

1. **World Model Fidelity天花板**: RAMP的policy quality受限于WM预测的future state质量. 如果WM对某些OOD state预测失准, policy的conditioning就会错. 论文用single denoising step降低overhead, 但可能牺牲了prediction质量.

2. **HILR Cost**: 虽然HILR的autonomous execution比纯teleop成本低, 但仍然需要human expert干预, scaling性受限. 真正self-evolution需要减少human involvement.

3. **Advantage Collapse处理**: Stage 4的joint WM+policy training是为了防止advantage collapse, 但论文没有量化分析这个问题的severity和sensitivity to hyperparameters.

4. **Long-horizon Evaluation**: 论文claim "long-horizon execution without failure", 但internal evaluation的8个task最长horizon可能是minutes级别, 真正的long-horizon (hours级别, 比如做一桌子菜)还未被验证.

### 12.2 Potential Extensions

1. **Hierarchical WM**: 当前WM预测fixed horizon (12/24/36/48 frames). Hierarchical design可以让WM在不同temporal scale上预测, 短期精确+长期抽象, 类似WaveNet的multi-scale.

2. **Counterfactual Reasoning**: 用WM做counterfactual rollout, "如果采取action A vs B, 未来会怎样不同", 提供更rich的preference signal给policy.

3. **Active WM Query**: Policy主动决定何时query WM (类似Active Inference), 简单action bypass WM节省compute, 复杂action query WM做planning.

4. **Cross-embodiment RAMP**: 当前实验在PiPER arm和G1 humanoid上, RAMP能否scale到更多morphology? WM的future state prediction是否embodiment-agnostic?

5. **VLM-based WM vs Diffusion-based WM**: Wan2.2是diffusion-based, 比较慢. 是否可以用autoregressive token-based WM (像Cosmos)加速inference? Table 1显示WM-based (state+value) inference是0.25s, 这对real-time control还是瓶颈.

---

## 13. References

- Paper project page: https://gigabrain05m.github.io
- RoboChallenge: https://robochallenge.ai/home
- GigaBrain-0: https://arxiv.org/abs/2510.19430
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- $\pi_{0.6}^*$ RECAP: https://arxiv.org/abs/2511.14759
- PaliGemma-2: https://arxiv.org/abs/2412.03555
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- AWR: https://arxiv.org/abs/1910.00177
- DAgger (Ross et al.): https://arxiv.org/abs/1106.0245
- DreamGen: https://arxiv.org/abs/2505.12705
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- Wan2.2: https://arxiv.org/abs/2503.20314
- Knowledge Insulation: https://arxiv.org/abs/2505.23705
- Mixture-of-Transformers: https://arxiv.org/abs/2411.04996
- FAST action tokenizer: https://arxiv.org/abs/2501.09747
- V-JEPA 2: https://arxiv.org/abs/2506.09985

---

这篇paper的核心insight可以概括为: **把VLA从reactive control提升到prospective planning, 关键在于让policy condition在world model预测的future state representation上, 同时保持policy自身的robustness**. RAMP的四阶段iterative pipeline结合HILR data collection, 实现了self-improving closed loop. 数学上证明RECAP是其special case, 信息论上证明引入 $\mathbf{z}$降低action generation的conditional entropy. 实验在30个RoboChallenge task上取得51.67%的SOTA, 在3个challenging long-horizon task上比RECAP高约30% points.

整体非常elegant的工作, 把model-based RL和VLA这两个之前相对独立的方向做了deep integration, 为embodied AI的long-horizon reasoning提供了一条concrete path.
