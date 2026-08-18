---
source_pdf: FLaRe Achieving Masterful and Adaptive Robot Policies with.pdf
paper_sha256: 9fe4ebf83f4b9246f12898cd643edb07d4d93de54f7959388a5a27cf0315b6a6
processed_at: '2026-08-18T13:04:51-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FLaRe 人话版

---

## TL;DR

机器人社区这两年主流套路是拿大 transformer 跑海量行为克隆（BC），训出 generalist policy。这些模型在 demo 数据覆盖的状态里挺好用，一出分布就崩——BC 的本质是"抄专家答案"，没见过的情况就懵。

RL 倒是直接优化"任务完成"，但 from scratch 太慢、还得手写 dense reward。FLaRe 的招数很直接：**先 BC 学个 prior，再用 sparse-reward RL 把这个 prior 从"模仿专家"对齐到"真正完成任务"**。这跟 LLM 里 SFT→RLHF 是同一套配方。

关键技术贡献是 4 个 anti-forgetting tricks，每一个都是为了不让 RL 梯度把 BC 学到的 prior 给毁掉：on-policy PPO、小 LR、关掉 entropy bonus、actor 和 critic 不共享 backbone。少一个都崩。

效果：仿真里 +23.6%，真机 +30.7%，还能学新能力、跨 embodiment、6 小时微调改行为。

---

## 1. 为什么 BC 训出来的 robot policy 不行

想象一个学生在背题：你给他一万道带解答的题，他背下来，考试遇到原题写得飞快。但只要题目换一下条件、换个场景，他立刻懵。

BC 就是这种训练。你给模型大量专家轨迹（"在这个状态下，专家执行了这个 action"），模型学的是 conditional imitation：给定状态，输出专家会做的 action。

问题在哪：

**Compounding error。** BC 假设训练时见过的状态和部署时见过的状态分布一致。但只要某一步 action 预测有一丁点偏差，机器人就跑到了一个 BC 从没见过的状态，下一步预测更差，再下一步更差，指数级发散。长 horizon 任务里这个问题致命。

**Objective mismatch。** BC 优化的是 action prediction loss（"和专家一致"），但部署时我们关心的是 task completion（"任务完成没"）。这两个 objective 不等价——你可以每步 action 都和专家差不多，但累积起来没完成任务。相反，有些偏离专家的 action 可能反而更好（比如专家绕了个弯，policy 直走近路）。

所以这两年大家发现 RT-1、SPOC、OpenVLA 这些 BC 大模型在 demo 覆盖范围内不错，一出范围就 fail。

---

## 2. RL 为什么能救，但又为什么 from scratch 不行

RL 的好处是直接 optimize 你真正想要的：task completion。让机器人在仿真里试，完成任务给 reward=1，没完成给 0。policy 通过 trial-and-error 学会完成任务。

这个目标对齐了 deployment 的真实目标，所以 RL policy 天然会 recover、会 explore、会绕开 BC 不会的状态。

但 RL from scratch 在 robotics 上有两大死穴：

**Sample inefficient。** 长 horizon 任务（CHORES 里要 600 步），action space 20 维离散，机器人一开始随机探索几乎不可能凑巧完成任务。Poliformer 在 ObjectNav 上从 scratch 训 300M steps 才到 85%——这种规模的探索在很多 task 上根本起不来（Fetch 任务从 scratch 直接 0%）。

**Reward engineering。** 纯 sparse reward（任务完成才给 1）太稀疏，RL 信号几乎为零。常规做法是手 craft 一个 dense reward（靠近目标物体给 +0.1，朝向目标给 +0.05...）。但每个新任务都要专家手写 reward、调权重、防 collapse，根本 scale 不动。

---

## 3. FLaRe 的核心 insight：BC 当 prior，RL 当 aligner

把 BC 和 RL 各自的优势拼起来：

- BC 已经让模型在"大致做对"的状态空间里了，省掉了 from scratch 那种茫茫然探索
- RL 拿这个 prior 当起点，在仿真里 sparse reward 微调，把 policy 从"模仿专家"对齐到"完成任务"

这个 insight 跟 LLM 的 RLHF 一模一样：SFT 让语言模型先学会"大致像人话"，PPO 用人类偏好 reward 把它从"像人话"对齐到"听话有用"。同样的两阶段、同样的 prior + RL 微调思路。

直觉上：BC 给了你一个 90% 对的起点，RL 把它推到 99%。

---

## 4. 为什么 naive RL fine-tune 会崩

Empirical observation：拿 PPO 直接 fine-tune SPOC，training 立刻 collapse，Fetch 任务成功率掉到 0%。

为什么？因为 large transformer BC policy 已经在一个很精细的"good behavior manifold"上，RL 梯度来得猛、来得 noisy，一下子就把这个 manifold 推歪了。pre-trained 的 prior 被毁掉，policy 跑到 OOD action 区域，从此再也回不来。

这跟 LLM RLHF 早期的稳定性问题完全一样——OpenAI 在 [InstructGPT](https://arxiv.org/abs/2203.02155) 里也踩过这个坑。

[Wółczyk et al. 2024](https://arxiv.org/abs/2402.02868) 直接给了个 framing：**"Fine-tuning RL is secretly a forgetting mitigation problem."** 你不是在学新东西，你是在学新东西的同时别忘掉旧东西。

FLaRe 的核心 contribution 就是为 large BC model 量身定制的 anti-forgetting toolkit——4 个 trick，缺一不可。

---

## 5. 四个 stabilization tricks，逐个讲直觉

### 5.1 On-policy PPO，不用 off-policy SAC

off-policy（SAC、DDPG）能复用旧数据，sample efficient。但 off-policy 有个老问题叫 **deadly triad**：function approximation + bootstrapping + off-policy 三者同时出现时 value 估计容易发散。在 transformer + image observation 这种超 expressive approximator 上，deadly triad 几乎必然爆。

FLaRe 选择：**反正仿真里数据随便采，sample efficiency 不是瓶颈，优先稳定性，用 on-policy PPO。** PPO 的 clip 机制天然限制每 step policy 漂移幅度，是个隐式的 anti-forgetting 设计。

类比：off-policy 像用旧地图导航，地图旧了方向错可能越走越偏；on-policy 像实时重新打图，慢但不会因为旧数据累计偏差。

### 5.2 学习率小一个数量级

from-scratch PPO 在 ObjectNav SoTA 用 LR=2e-4。FLaRe fine-tune 必须降到 2e-5（1/10）。

直觉：pre-trained policy 已经在 good manifold 上了，gradient step 太大就把它推出 manifold。RL 的 gradient 又是 noisy 的（high variance 的 policy gradient estimate），大步 + 噪声 = 必然 drift 出去。

这跟 LLM RLHF 里 PPO LR 比 SFT LR 小一个数量级是同样的经验法则。

类比：你已经站在悬崖边一块小石头上，下一步要走，你肯定迈小碎步试探，不会大跨步。

### 5.3 关掉 entropy bonus

PPO 标准 objective 有个 entropy bonus：

$$
\mathcal{L} = \mathcal{L}^{\text{CLIP}} - c_v \mathcal{L}^{\text{VF}} + c_e \mathcal{H}[\pi_\theta]
$$

entropy bonus 鼓励 policy 输出的 action 分布更"分散"，促进 exploration。在 from-scratch RL 里这很重要，因为一开始 policy 是 near-uniform 的，不推一下它就不 explore。

但 fine-tune 时候这玩意儿是毒药：**BC policy 本来 entropy 就低**（专家轨迹学出来就比较 deterministic），entropy bonus 强行把这个分布打散，训练初期 entropy gradient 主导，policy 被快速推到 random 区域，pre-trained prior 丢失。

所以 FLaRe 设 $c_e = 0$。

类比：from-scratch 是教一个啥都不会的小孩，鼓励他多试新东西；fine-tune 是教一个已经会做菜的大厨做新菜，你跟他说"你随便炒什么都行"，他反而把基本功都丢了——你要让他从已经会的菜里小改。

### 5.4 Actor 和 critic 不共享 backbone

标准 RL practice 是 actor 和 critic 共享早期 feature extractor（图像 backbone + transformer encoder 前几层）。优点：参数共享，feature 学习互相帮助。

但 fine-tune 时候这玩意儿也毁 prior：critic 的 value 估计很 noisy（尤其 sparse reward 下），它的梯度反传回 shared backbone，把 actor 用来做 action prediction 的好 feature 给污染了。actor 的预测能力退化，policy 崩。

FLaRe 招数：actor 和 critic 各自独立 transformer，都从 SPOC 权重 clone 出来，critic 的 value head 重新 random init。两边梯度互不打架。

类比：actor 是决策者，critic 是评估者。如果两人共用一个大脑，评估时的纠结会污染决策时的果断；分两个大脑，决策归决策、评估归评估。

---

## 6. 这四个 trick 一个都不能少

Ablation（Fig. 6b）在 Fetch 任务上：

| 完整 FLaRe | 66.9% |
|---|---|
| 换 SAC | 0% |
| LR 增 10× | 0% |
| Shared actor-critic | 0% |
| Entropy bonus = 0.2 | 0% |

任何一项缺失都 catastrophic collapse。这说明 fine-tune large BC model 跟 from-scratch RL 是两种不同的 game，规则完全不一样。

---

## 7. 仿真规模 + KV-cache 让大规模 RL feasible

FLaRe 在 AI2THOR + ProcTHOR 上做大规模仿真 fine-tune：

- 150k procedural generated houses（场景多样）
- 800K+ Objaverse 3D objects（物体多样）
- domain randomization（color aug、random crop、posterize）
- DINOv2 frozen 做 sim-real feature bridge

KV-cache 是工程关键：causal transformer attention 是 $O(n^2)$ 复杂度，episode 几百步就跑不动。KV-cache 缓存历史 K/V，新 step 只算新 token 对 cached keys 的 attention → $O(n)$ 复杂度。这让大规模 RL 在算力上 feasible。

---

## 8. 实验讲了什么

### 8.1 已见任务（CHORES-S）

四个 task：ObjectNav、Fetch、PickUp、RoomVisit。FLaRe 平均 79.5%，比 prior SoTA 高 23.6%。

关键对照：Poliformer-Dense 用 dense reward + privileged info + 训 300M steps，FLaRe 只训 20M steps，结果 FLaRe 在三个任务上反超。说明 BC prior + sparse reward > dense reward from scratch，long-horizon mobile manipulation 上尤其明显。

### 8.2 新能力（分布外任务）

三个 base model 没见过的 task：

- **ObjNavRelAttr**："找最大的苹果" → 要搜索所有同类物体、比较属性、再 decide
- **RoomNav**："去厨房" → navigate 到 room type 而不是 object
- **ObjNavAfford****："找可以坐的东西" → affordance 推理

FLaRe 在这些 task 上都 SoTA。意义：pre-trained features 有 transferable structure，RL fine-tune 能 discover 如何用这些 features 完成新 capability。这是 BC-only 方法做不到的——BC 学不出来的行为，RL 能从 sparse reward 里 emergent 出来。

这跟 LLM 里 RLHF 让模型 emergent reasoning 是同一个现象：**RL + pre-trained model 能 discover 新行为模式**。

### 8.3 真机直接 sim-to-real

Stretch RE-1 机器人，无任何 real-world fine-tune，平均 80.7% 成功率，比 prior SoTA 高 30.7%。

证明 DINOv2 frozen + 大规模 domain randomization 的 sim-real bridge 有效。

### 8.4 跨 embodiment

把只在 Stretch-RE1 上 BC 训的 SPOC，用 FLaRe fine-tune 适配到 Locobot（不同 action space、不同 camera）。trick：mask 掉 invalid action、用闲置 action 槽位控制 camera。ObjectNav 上 FLaRe 72.0%，比 Poliformer zero-shot 57.5% 高一截。

意义：foundation policy 的 representation 和 behavior prior 能跨 embodiment 迁移，只需简单 action remap + RL fine-tune。

### 8.5 行为 shaping

训完之后还能 post-hoc 微调行为偏好，6 小时 fine-tune：

- 加 step penalty −0.01/step → episode 长度从 258 降到 222（更高效），SR 几乎不变
- 加 collision penalty −0.5/collision → 碰撞次数从 10 降到 3.1，SR 几乎不变

说明 RL fine-tune 后的 policy 是 **steerable** 的——可以按部署偏好微调，BC policy 做不到这个（BC 只能 mimic demo 里的行为）。

---

## 9. 跟其他路线的关系

### 9.1 跟 LLM RLHF 同构

| 阶段 | LLM RLHF | FLaRe |
|---|---|---|
| Pretrain | LM pretrain on web | DINOv2 vision pretrain |
| SFT | instruction tuning | BC on expert trajectories |
| RL fine-tune | PPO on preference reward | PPO on task completion reward |
| Stabilization | KL penalty to SFT model | small LR + no entropy + independent AC |
| Steerability | helpful/harmless/honest | step penalty / collision penalty |

两套 pipeline 几乎一一对应。

### 9.2 跟 RT-2 / OpenVLA 互补

RT-2、OpenVLA 走 VLA 路线，把大 VLM 直接 fine-tune 输出 action token，强 generalization 但仍 BC。FLaRe 是个互补方向：**不管 base model 怎么 pretrain，sparse-reward RL fine-tune 都能突破 BC plateau**。理论上把 FLaRe 套到 OpenVLA 上 fine-tune 应该能复现类似 gain。

### 9.3 跟 Decision Transformer / Poliformer 系列同源

Poliformer 用 causal transformer 做 navigation，从 scratch on-policy RL。FLaRe 复用了它的架构思路（Llama 2 decoder block、KV-cache），但出发点是 BC pretrain 而非 from scratch。

### 9.4 JSRL / PIRLNav 是最近的前序工作

[JSRL](https://arxiv.org/abs/2304.06107) 用 prior policy 渐进 roll in 数据。[PIRLNav](https://arxiv.org/abs/2301.07902) 用 LR scheduling warm-start value function 缓解 shared backbone 问题。FLaRe 在他们的基础上做得更彻底——actor/critic 直接分离、稳定性的四件套全配上、scale 到 large transformer + real robot。

---

## 10. 限制

Paper 自己承认：reliance on simulation。对 liquid、soft object、deformable 这种 sim 不准的任务，fine-tune 困难。可能需要 real-world fine-tune，但 on-policy real-world RL 慢得 prohibitive。

未来可能的方向（speculative）：
- Offline-to-online RL：用 [Cal-QL](https://arxiv.org/abs/2310.10543) 或 [IQL](https://arxiv.org/abs/2110.06169) 在 real robot demo data 上预训，再 on-policy fine-tune
- World model 路线：[DreamerV3](https://arxiv.org/abs/2301.04121)、[Genie](https://arxiv.org/abs/2402.15391) 在 latent imagination 里 fine-tune，部分摆脱 sim 依赖
- Diffusion policy + RL：[Diffusion Q-Learning](https://arxiv.org/abs/2307.04140) 类方法 fine-tune diffusion policy
- VLM 提供 dense reward shaping：替代 hand-crafted reward
- Hierarchical：high-level VLM planner + low-level FLaRe policy 处理 ultra-long-horizon

---

## 11. 最最直觉的一图流总结

把整篇 paper 浓缩成一句话：

> 把 BC 训出来的 robot foundation model 当作先验，用 sparse-reward PPO 在大规模仿真里 fine-tune，关键是用 4 个 anti-forgetting tricks 保护 prior 不被 RL 梯度毁掉——on-policy、小 LR、关 entropy bonus、actor 和 critic 不共享 backbone——这样就能让 model 从"模仿专家"对齐到"真正完成任务"，并在分布外任务、真机、新 embodiment 上 generalize。

这套 recipe 在 LLM 那边已经被 RLHF 验证过，FLaRe 把它完整搬到 robotics，证明在 large transformer + image observation 上同样 work，且能 sim-to-real。

更深层的 take-away：**pretrain + RL fine-tune 是跨模态通用 recipe**，关键矛盾永远是"学新东西 vs 别忘旧东西"，解决矛盾的 toolkit 也类似。这条线未来肯定会继续延伸——diffusion policy + RL、VLA + RL、world model + RL——都是这个 framing 下的变体。

参考链接：
- FLaRe: https://robot-flare.github.io
- SPOC: https://arxiv.org/abs/2310.15107
- Poliformer: https://arxiv.org/abs/2406.20083
- InstructGPT: https://arxiv.org/abs/2203.02155
- Wółczyk forgetting: https://arxiv.org/abs/2402.02868
- JSRL: https://arxiv.org/abs/2304.06107
- PIRLNav: https://arxiv.org/abs/2301.07902
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- DINOv2: https://arxiv.org/abs/2304.07193
- ProcTHOR: https://arxiv.org/abs/2206.06994
- Phone2Proc: https://arxiv.org/abs/2303.06604

---

# FLaRe: Large-Scale RL Fine-Tuning for Robotic Foundation Models — 技术深度讲解

---

## 1. Big Picture：这篇 paper 在解决什么问题

Robotics 社区最近两三年的主线之一是用 **large-scale multi-task behavior cloning (BC)** 训 large transformer policy（RT-1/RT-2/RT-X/SPOC/Octo/OpenVLA/RoboCat...）。这些 model 形态各异，但 recipe 一致：transformer + 大量 expert demonstrations + multi-task。

直接 deploy 的问题在 BC 的本质：**policy 被约束在 expert 轨迹覆盖的 state manifold 上**。一遇到 distribution shift（unseen states、长 horizon 累积误差），compounding error [Ross et al., DAgger](https://arxiv.org/abs/1011.0686) 就让 policy 跑飞。BC 的 objective（action matching）和真实 deployment objective（task completion）之间存在结构性 mismatch。

RL 反过来直接优化 task completion。但 RL from scratch 在 robotics 上有两大痛点：(a) sample inefficient，长 horizon + 大 action space 下几乎 exploration 不动；(b) 需要 hand-crafted dense reward，scalability 差。

FLaRe 的核心 insight 很简单：**把 BC pretrain 出来的 foundation policy 当作先验，用 sparse-reward RL 把它从"模仿专家"对齐到"完成任务"**。这跟 LLM 里的 RLHF pipeline（SFT→PPO）非常像——只是这里 SFT 换成了 BC on robot trajectories。理解这点后整个 paper 的设计决策就有了 organizing principle。

Project page: https://robot-flare.github.io

---

## 2. 问题形式化（POMDP + sparse reward）

每个 task $T \in \mathcal{T}$ 被建模为 language-conditioned POMDP：

$$
(S,\ \mathcal{A},\ \mathcal{P},\ R,\ O,\ \mathcal{L},\ P(s_0),\ \gamma)
$$

变量含义：
- $S$：**latent state space**（agent 不知道真实 state，只观测）
- $\mathcal{A}$：action space（CHORES 里是 20 个 discrete actions，含 base/arm/gripper 的离散位移 + `done` + `terminate`）
- $\mathcal{P}$：Markov transition $s_{t+1} \sim \mathcal{P}(\cdot | s_t, a_t)$
- $O$：observation space（两路 ego-centric RGB 384×224 + text instruction）
- $\mathcal{L}$：natural language instructions 的集合
- $P(s_0)$：initial state distribution（每个 episode 开始采样）
- $\gamma$：discount factor（paper 里 $\gamma = 0.99$，Table IV）
- $R: \mathcal{L} \times S \to \{0, 1\}$：**sparse binary reward**，只在任务完成时给 1，否则 0

Task $T$ 定义一组语言指令 $\mathcal{L}_T$，episode 开始采样 $l_T \in \mathcal{L}_T$ 和 $s_0 \sim P(s_0)$。目标训练 policy $\pi_\theta^T$ 最大化期望 return：

$$
\max_\theta\ \mathbb{E}_{\mathcal{L}_T,\ \pi_\theta^T}\left[\sum_t R(s_t, l_T)\right]
$$

直觉：所有 task 共享同一 observation/action space（同一 robot 的传感器和执行器），但每个 task 自己的语言指令和 success criterion 不同。Reward 是 sparse binary，**这是 FLaRe 可 scalability 的关键**——加新 task 只需定义 success criteria，无需 shaping reward 工程。

---

## 3. 模型架构（SPOC base + RL head）

FLaRe fine-tune 的 base model 是 [SPOC](https://arxiv.org/abs/2310.15107)（Imitating Shortest-Path Experts），一个 multi-task transformer 用于 mobile manipulation。架构（Fig. 7）：

### 3.1 Vision backbone（frozen DINOv2）

输入两路 RGB：navigation camera $i_a \in \mathbb{R}^{H \times W \times 3}$ 和 manipulation camera $i_b \in \mathbb{R}^{H \times W \times 3}$（CHORES 里 $H=224, W=384$）。

每个 image 独立过 frozen [DINOv2](https://arxiv.org/abs/2304.07193)，输出 patch-wise 表示：

$$
r \in \mathbb{R}^{\frac{H}{14} \times \frac{W}{14} \times h}
$$

变量含义：
- $14$ 是 patch size
- $h$ 是 DINOv2 hidden dim
- $\frac{H}{14} \times \frac{W}{14}$ 是 patch grid 大小

Reshape + linear projection 到 encoder dim：
$$
\nu_{\text{raw}} \in \mathbb{R}^{n_{\text{patch}} \times d_{\text{encoder}}}
$$

加 learnable **camera-type embedding** 区分 nav vs. manipulation camera → final visual features $\nu$。

DINOv2 frozen 的目的：sim-to-real transfer。DINOv2 自监督预训练在真实图像上，sim 和 real 都能 produce 相近 feature distribution。

### 3.2 Text encoder（frozen T5）

Natural language instruction $l$ 过 [Sentence-T5](https://arxiv.org/abs/2108.08877) 得 text feature $\tau$。这跟 LLaVA 那套 vision-language 接入是类似 trick。

### 3.3 Transformer state encoder（BERT-style [STATE] token）

输入：visual features $\nu$ + text features $\tau$ + learnable **STATE token** $\sigma$，concat 起来过非 causal transformer encoder。取对应 STATE token 的输出作为 state vector $s \in \mathbb{R}^d$。这是 BERT [CLS] token 的 trick——用专门一个 token 做 information pooling。

这个模块 produce 一个 **text-conditioned visual state representation**。

### 3.4 Causal transformer decoder（Llama 2 decoder）

FLaRe 把 SPOC 原 decoder 换成 [Llama 2](https://arxiv.org/abs/2307.09288) decoder block 来加速训练和推理。Causal decoder 处理 partial observability + long-horizon memory：消费每 timestep 的 state vector $s_t$，加 sinusoidal temporal positional encoding + previous action embedding，跨 episode 时间维度做 causal attention，输出 belief vector → actor head → action logits。

输入 token 序列是 $\{s_0, s_1, \ldots, s_t\}$，causal mask 让 $s_t$ 只能看到 $s_{\le t}$，等价于 recurrent policy。这个 design 跟 [Decision Transformer](https://arxiv.org/abs/2106.01345)、[Poliformer](https://arxiv.org/abs/2406.20083) 一脉相承。

### 3.5 RL head：value network

**独立**初始化一个和 policy 一样的 transformer（同架构、同 SPOC 权重），policy head 换成 random init 的 value head。这步很重要，后面会讲为什么。

---

## 4. Stabilization Techniques（论文核心 contribution）

这是 FLaRe 真正的贡献。Empirical observation：直接拿 PPO fine-tune SPOC，training 会 collapse。Fig. 6(b) ablation 显示移除下面任一项 → Fetch success rate 立刻掉到 0。四项：

### 4.1 On-policy PPO（不用 off-policy SAC）

Off-policy 方法（SAC、DDPG）能复用旧数据 → sample efficient。但 off-policy 受 **deadly triad** [Sutton & Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) 困扰：function approximation + bootstrapping + off-policy 三者同时出现时 value 估计容易发散。在 large transformer + 高维 image observation 下尤其敏感。

FLaRe 选择：完全在 simulation 里 fine-tune，sample efficiency 不是 bottleneck，**优先稳定性** → PPO [Schulman et al.](https://arxiv.org/abs/1707.06347)。PPO clipped surrogate objective：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\epsilon,\ 1+\epsilon)\hat{A}_t\right)\right]
$$

变量含义：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$：importance sampling ratio，新旧 policy 在同一 $(s_t, a_t)$ 上的概率比
- $\hat{A}_t$：GAE 估计的 advantage，$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$，其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error
- $\epsilon = 0.1$（Table IV，clipping ratio）
- $\gamma = 0.99$，GAE $\lambda = 0.95$

### 4.2 Small learning rate（10× 减小）

PPO from scratch 在 ObjectNav 上的 SoTA 用 $\text{lr} = 2 \times 10^{-4}$。FLaRe fine-tune 时降到 $\sim 2 \times 10^{-5}$（10× 减小）。注意 paper 里 Table IV 显示 0.0002，但 ablation 章节说 baseline 用 $2 \times 10^{-4}$、ablation 测试 $2 \times 10^{-4}$ 是 original 的 10×，所以原 LR 应该是 $2 \times 10^{-5}$。这里 Table IV 有 typo 嫌疑。

Intuition：pre-trained policy 已经在合理 action manifold 上，gradient step 太大就把它推到 OOD 区域，destroy 学到的 prior。这跟 LLM RLHF 里 PPO LR 比 SFT LR 小一个数量级的做法一致（[InstructGPT](https://arxiv.org/abs/2203.02155) 也是这样调）。

### 4.3 Disable entropy bonus

PPO 标准 objective 含 entropy bonus：

$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_v \mathcal{L}^{\text{VF}}(\theta) + c_e \mathcal{H}[\pi_\theta](s_t)
$$

变量含义：
- $c_v = 0.5$（value loss weight）
- $c_e$：entropy weight，标准 PPO 默认 0.01
- $\mathcal{H}[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$

FLaRe **设 $c_e = 0$**。原因：pre-trained policy 已经 deterministic enough（BC expert trajectory 学出来 entropy 本来就低），再加 entropy bonus 在 training 初期梯度被 entropy term 主导，会快速 distort pre-trained action distribution，触发 unlearning。

这跟 from-scratch PPO 不同：from scratch policy 一开始 near-uniform，需要 entropy bonus 鼓励 exploration。fine-tune 时 exploration prior 已经在，bonus 反而 destructive。

参考 [Wółczyk et al., "Fine-tuning RL models is secretly a forgetting mitigation problem"](https://arxiv.org/abs/2402.02868) —— 正是这个 insight 的理论支撑。

### 4.4 Disable feature sharing（actor/critic 独立）

Standard practice：actor 和 critic 共享 feature extractor（图象 backbone + 早期 transformer layer）。优点：参数共享、feature 学习互相帮助。

FLaRe 反过来：**actor 和 critic 完全独立 transformer**，只是从 SPOC 权重 clone 出来，critic head random init。

原因：pre-trained SPOC 的 visual/state features 已经很好。如果 critic gradient 反传回共享 backbone，会把 critic 的 noisy value-estimation gradient 混进 actor 的 action prediction 路径，破坏 pre-trained features，导致 action prediction deteriorate。

这个现象在 [PIRLNav](https://arxiv.org/abs/2301.07902) 里也观察到——他们用 LR scheduling 来 warm-start value function 缓解，FLaRe 直接 architectural separation 更彻底。

### 4.5 Ablation 全览（Fig. 6b）

| 移除项 | Fetch success rate |
|---|---|
| 完整 FLaRe | 66.9% |
| → SAC 替 PPO | collapse to 0 |
| → LR 增 10× | collapse to 0 |
| → Shared actor-critic | collapse to 0 |
| → Entropy bonus = 0.2 | collapse to 0 |

**任何一项缺失都 catastrophic failure**。这说明 fine-tune large BC model 跟 from-scratch RL 是两种不同的 game。

---

## 5. 大规模仿真 + KV-cache

### 5.1 AI2THOR + ProcTHOR + Objaverse

- [AI2THOR](https://arxiv.org/abs/1712.05474)：interactive 3D environment
- [ProcTHOR](https://arxiv.org/abs/2206.06994)：procedurally generated 150k houses
- [Objaverse](https://arxiv.org/abs/2212.01651)：800K+ annotated 3D objects

加上：
- **Domain randomization**：color aug + random crop + posterize
- **DINOv2 frozen features** 做 sim-real bridge

150k houses 是一个相当大的 scale，确保 RL fine-tuning 见到的 house distribution 足够 diverse，policy 不会 overfit 到 specific layout。

### 5.2 KV-cache 加速 transformer inference

Transformer causal attention 复杂度 $O(n^2)$ 在 episode 长度大时 prohibitive。用 [KV-cache](https://arxiv.org/abs/2211.05102)（[Pope et al., "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102)）缓存之前 step 的 K, V，新 step 只算新 token 对 cached keys 的 attention → $O(n)$ 复杂度。

这对 large-scale RL 至关重要：每个 PPO update 需要 32 rollouts × 数百 steps × full transformer forward，KV-cache 把 inference cost 砍一个数量级才 feasible。

---

## 6. 实验结果

### 6.1 Seen capabilities（Table I, CHORES-S benchmark）

四个 task：ObjectNav、Fetch、PickUp、RoomVisit。Reporting Success (SEL)，SEL = Episode-length weighted Success [Eftekhar et al.](https://arxiv.org/abs/2311.04193) 衡量 efficiency（短 episode 高权重）。

| Task | FLaRe | SPOC (IL only) | Poliformer-Sparse (RL) | Poliformer-Dense (RL, privileged) |
|---|---|---|---|---|
| ObjectNav | 85.0 (67.6) | 55.0 (42.2) | 14.5 (10.4) | 85.5 (61.2) |
| Fetch | **66.9 (54.7)** | 14.0 (10.5) | 0.0 | 0.0 |
| PickUp | **91.8 (90.4)** | 90.1 (86.9) | 0.0 | 90.1 (88.7) |
| RoomVisit | **70.4 (67.1)** | 40.5 (35.7) | 12.5 | 12.5 |

平均 success rate 79.5%，比之前 SoTA +23.6% absolute。

注意几个点：
1. Poliformer-Dense 用 **hand-crafted dense reward + privileged info + 训练 300M steps**（FLaRe 只训 20M）。FLaRe 在 ObjectNav 与之打平，在 Fetch/RoomVisit 远超——dense reward 在 long-horizon mobile manipulation 上反而没有 BC prior + sparse reward 好。
2. RL-only from-scratch（Poliformer-Sparse）在 Fetch 这种需要 navigation + manipulation 串联的长 horizon 任务上完全学不动（0%），凸显 BC prior 的关键性。

### 6.2 Novel capabilities（Table II）

三个 distribution 外的 task，需要 base model 没见过的 reasoning skill：

- **ObjNavRelAttr**：找相对属性最大的物体（"find the largest apple"）
- **RoomNav**：navigate 到 room type（"go to the kitchen"）
- **ObjNavAfford**：affordance 推理（"find something I can sit on"）

| Task | FLaRe | Poliformer (Sparse) | SPOC++ (BC + extra demos) | Poliformer (Dense) |
|---|---|---|---|---|
| ObjNavRelAttr | **71.0 (63.6)** | 6.7 | 54.5 (44.6) | 36.1 |
| RoomNav | **91.6 (85.6)** | 57.0 | 74.5 | 75.0 |
| ObjNavAfford | **79.7 (70.6)** | 35.5 | 62.4 | 53.8 |

意义：FLaRe 在 base model 完全没见过的 task 上还能学得动，且超过用更多 expert demo + dense reward 的 baseline。这说明 pre-trained features 具备 transferable structure，RL fine-tune 能 discover 如何用这些 features 完成新 capability。这是 paper 的 strong argument：**path towards continual adaptation**——加新 task 只需定义 success criteria + 语言指令。

### 6.3 Real world（Table III, Stretch RE-1）

直接 sim-to-real，no real-world fine-tuning，no adaptation。Navigation 用真实 policy，manipulation grasping 用 [SPOC 启发式 grasp model](https://arxiv.org/abs/2310.15107)。

| Task | FLaRe | SPOC | Poliformer-Dense |
|---|---|---|---|
| ObjectNav | **94.4** | 50.0 | 83.3 |
| Fetch | **66.7 (55.6 policy)** | 33.3 (11.1) | X |
| PickUp | **86.7 (66.7 policy)** | 66.7 (46.7) | X |
| RoomVisit | **75.0** | 50.0 | X |

Real-world avg 80.7%，比 best prior +30.7% absolute。

### 6.4 Cross-embodiment（Locobot）

Locobot vs Stretch-RE1：action space 不同（Locobot 没 arm）、camera 参数不同（mount 更低、视场更窄但可旋转）。

Trick：mask out invalid actions，把两个 invalid action 槽位 repurpose 来控制 camera 旋转。

| Method | SR ↑ | SEL ↑ |
|---|---|---|
| FLaRe (fine-tune from SPOC) | **72.0** | 47.2 |
| Poliformer zero-shot | 57.5 | 30.1 |
| Poliformer (sparse) | 44.0 | 29.7 |

意义：foundation policy 的 representation 和 behavior prior 可以迁移到 embodiment 完全不同的机器人，只需简单 action remapping + RL fine-tune。这跟 [RT-X](https://arxiv.org/abs/2310.08864) 路线方向一致——cross-embodiment generalization。

### 6.5 Behavior shaping（cross-objective）

测试能否 post-hoc 修改 policy behavior，**只加 reward term、6 小时 fine-tune**：

| Setting | SR | Ep. Len | # Collisions |
|---|---|---|---|
| FLaRe baseline | 66.9 | 258.2 | 10.0 |
| + step penalty $-0.01$/step | 65.7 | **222.8** | 10.0 |
| + collision penalty $-0.5$/coll | 66.7 | 251.2 | **3.1** |

SR 几乎不变，行为按 reward term 引导调整。这说明 RL fine-tune 后的 policy 是 **steerable** 的，可以按 deployment preference 微调——这是 BC policy 不具备的能力（BC 只能 mimic）。

---

## 7. Hyperparameters（Table IV + C 节）

关键值：

| Parameter | Value |
|---|---|
| Total rollouts per PPO update | 32 |
| Learning rate | 0.0002（疑似 typo，ablation 暗示实际 2e-5）|
| Mini batch per update | 1 |
| Update repeats | 4 (epochs per batch) |
| Max gradient norm | 0.5 |
| $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| PPO clip $\epsilon$ | 0.1 |
| Value loss weight | 0.5 |
| **Entropy loss weight** | **0.0** ← 关键 |
| PPO update steps | 128 |
| State encoder layers | 3 |
| State encoder hidden dim | 512 |
| State encoder heads | 8 |
| Causal decoder layers | 3 |
| Causal decoder hidden dim | 512 |
| Causal decoder heads | 8 |

Training steps：
- ObjectNav/RoomVisit: 20M (vs baseline 60M, 3× less)
- Fetch/PickUp: 50M (vs baseline 100M, 2× less)
- Novel tasks: 50M / 20M
- Cross-embodiment: 30M

15× training time reduction over prior SoTA（前 SoTA Poliformer-Dense 在 ObjectNav 上训了 300M steps）。

---

## 8. Intuition 总结 & 联想

### 8.1 FLaRe 跟 RLHF 的同构

LLM RLHF pipeline：pretrain LM → SFT on instructions → RL (PPO) on human preference reward。FLaRe pipeline：pretrain vision encoder (DINOv2) → BC (SFT equivalent) on shortest-path expert → RL (PPO) on task completion reward。两者都遇到 stabilization 问题，solution 类似（small LR、KL regularization 或 entropy 控制）。可以读 [InstructGPT](https://arxiv.org/abs/2203.02155)、[Ziegler et al. on RLHF instability](https://arxiv.org/abs/1909.08593) 做 cross-reference。

### 8.2 "Fine-tuning is forgetting mitigation"

[Wółczyk et al. 2024](https://arxiv.org/abs/2402.02868) 论证 RL fine-tuning 的 hidden 问题是 catastrophic forgetting of pre-trained prior。FLaRe 的四项 stabilization 全是 anti-forgetting tricks：
- Small LR：limit parameter drift
- Disable entropy bonus：don't push policy away from BC distribution
- Disable feature sharing：don't corrupt pre-trained features with critic gradient
- On-policy PPO：clipping 限制每 step policy 漂移幅度

这跟 [EWC](https://arxiv.org/abs/1612.00796)、[AdaLoRA](https://arxiv.org/abs/2303.10512)、[LoRA](https://arxiv.org/abs/2106.09685) 这些 continual learning / parameter-efficient fine-tuning 思路同源。

### 8.3 Deadly triad in transformers

为什么 off-policy (SAC) 在 transformer policy 上 fail？Sutton & Barto 的 deadly triad：
1. Function approximation（neural net）
2. Bootstrapping（TD target 用 V(s') 估计）
3. Off-policy update（用旧 data）

Transformer + image obs 让 function approximator 极度 expressive，bootstrapped target noise 放大，off-policy 数据 distribution shift 严重 → divergence 几乎必然。这跟 [Decision Transformer](https://arxiv.org/abs/2106.01345) 系列工作强调的 "RL as sequence modeling" 也在反 off-policy 在大模型上的脆弱性。

### 8.4 BC + RL 的历史脉络

- [DAgger](https://arxiv.org/abs/1011.0686)：interactive BC 解决 distribution shift，但仍需 expert
- [DDPGfD](https://arxiv.org/abs/1509.02971)：demo augmented RL，需要 dense reward
- [AWAC](https://arxiv.org/abs/2006.09359)：advantage-weighted regression，offline→online
- [JSRL](https://arxiv.org/abs/2304.06107)：jump-start with prior policy，gradually roll in
- [PIRLNav](https://arxiv.org/abs/2301.07902)：BC pretrain + PPO fine-tune for ObjectNav，用 LR schedule

FLaRe 的 advance：把这些 idea 放到 large multi-task transformer + real robot 上验证，证明 stabilization design 能 scale。

### 8.5 Sim-to-real bridge

两件事让 sim→real work：
1. DINOv2 frozen → robust 跨 domain feature
2. Massive domain randomization + 150k procedural houses

更激进的 sim-real 思路是 [Phone2Proc](https://arxiv.org/abs/2303.06604)（用真实手机扫描重建仿真房子），FLaRe real-world eval 就用了 Phone2Proc 扫的 6-room apartment，这间接验证了 Phone2Proc pipeline 的有效性。

### 8.6 限制 & 未来方向

Paper 自己承认：reliance on simulation。对涉及 liquid、soft object、deformable 的任务（sim 不准），fine-tune 困难。可能需要 real-world fine-tune，而 on-policy real-world RL 慢得 prohibitive。

未来可能的方向（hallucinate a bit）：
- Offline-to-online RL fine-tune：用 [Cal-QL](https://arxiv.org/abs/2310.10543) 或 [IQL](https://arxiv.org/abs/2110.06169) 在 real robot demo data 上 pretrain，再 on-policy fine-tune
- 用 world model（[DreamerV3](https://arxiv.org/abs/2301.04121)、[Genie](https://arxiv.org/abs/2402.15391)）做 latent imagination，部分摆脱 sim 依赖
- Diffusion policy + RL：用 [Diffusion Q-Learning](https://arxiv.org/abs/2307.04140) 类方法 fine-tune diffusion policy
- 用 [VLM](https://arxiv.org/abs/2308.12966) 提供 dense reward shaping 替代 hand-crafted reward
- Long-horizon task 用 hierarchical RL：high-level VLM planner + low-level FLaRe policy

### 8.7 跟 RT-2 / OpenVLA 路线对比

[RT-2](https://arxiv.org/abs/2307.15818) 和 [OpenVLA](https://arxiv.org/abs/2406.09246) 走 vision-language-action model 路线，用 large pre-trained VLM 直接输出 action token，强 generalization 但仍 BC。FLaRe 是个互补方向：**不管 base model 怎么 pretrain，sparse-reward RL fine-tune 都能突破 BC plateau**。如果未来有人把 FLaRe 应用到 OpenVLA 上 fine-tune，应该能复现类似 gain。

### 8.8 Long-horizon reasoning 的 emergent ability

ObjNavRelAttr 任务要求 agent 找所有同类型物体、比较属性、再 decide。这超出 BC 训练数据覆盖的 behavior。FLaRe 学出来 71.0%——这暗示 **RL optimization 在足够 expressive 的 BC-initialized policy 上能 discover 多步 reasoning 行为**，类似 reasoning emergent from RLHF in LLM（[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[OpenAI o1](https://arxiv.org/abs/2407.18464) 那条线）。这是个 deep connection：**RL + pre-trained model → emergent reasoning**，跨 modularity 都成立。

---

## 9. 一句话总结

FLaRe 把"BC pretrain → RL fine-tune"这条 RLHF 风格的 recipe 完整搬进 robotics，核心 contribution 是 4 个 anti-forgetting stabilization tricks（on-policy、small LR、no entropy bonus、independent actor-critic），让大规模 transformer policy 能从 sparse reward 学到 masterful 且 steerable 的行为，real-world 平均 80.7% 成功率把 prior SoTA 甩开 30%+ absolute。

参考链接汇总：
- FLaRe project: https://robot-flare.github.io
- SPOC: https://arxiv.org/abs/2310.15107
- Poliformer: https://arxiv.org/abs/2406.20083
- PPO: https://arxiv.org/abs/1707.06347
- DINOv2: https://arxiv.org/abs/2304.07193
- ProcTHOR: https://arxiv.org/abs/2206.06994
- AI2THOR: https://arxiv.org/abs/1712.05474
- JSRL: https://arxiv.org/abs/2304.06107
- PIRLNav: https://arxiv.org/abs/2301.07902
- Wółczyk forgetting: https://arxiv.org/abs/2402.02868
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- KV-cache: https://arxiv.org/abs/2211.05102
- Sutton & Barto RL: http://incompleteideas.net/book/RLbook2020.pdf
- DAgger: https://arxiv.org/abs/1011.0686
- Phone2Proc: https://arxiv.org/abs/2303.06604
