---
source_pdf: EXPO-FT Sample-Efficient Reinforcement Learning.pdf
paper_sha256: 2f0ed5838a778571e9b5aa1312ba293dfcd6a5d5d1ab29ed1010b1e0697bf8e2
processed_at: '2026-08-18T11:58:09-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 EXPO-FT

好，我换个讲法。假设我们坐在咖啡馆，你在白板上画给我看。

---

## 故事的开头：问题是什么

你有一个 π0.5 — Physical Intelligence 搞的那个 VLA，给它一张图、一句话"把花插进瓶子"，它就能输出机器人动作。听起来很美好。

**但问题是它不够可靠。** 你给它30次机会，它可能只能成18次。剩下的12次它把花捏坏了、对不准瓶口、或者手抖了一下。

那怎么办？两条路：

**第一条路：RL from scratch。** 拿个小小的 Gaussian policy，用 RL 从头训。优点是 sample efficient、能训到接近100%。缺点是它啥 prior 都没有，遇到新的初始状态就傻眼。你让它抓个鸡蛋，鸡蛋位置稍微挪一下它就不会了。

**第二条路：finetune VLA。** 直接在 π0.5 上面跑 RL。优点是有 prior，见多识广。缺点是 π0.5 是个 flow matching policy，你想想 backprop through 整个 denoising process — numerical 不稳定，inference 慢，gradient 容易爆。现有的 finetune 方法要么不收敛，要么需要海量样本。

EXPO-FT 说：**我两个都要。** 我既要 VLA 的 prior，又要 sample efficient，还要 100% 可靠。

project page: https://pd-perry.github.io/expo-ft/

---

## EXPO 的核心 intuition：别动 VLA，挂个小弟

这是整个故事最关键的一步。先讲原始 EXPO (https://openreview.net/forum?id=aFjSjkB6CV)。

你看 π0.5 输出一个 action $a$。我们想"修正"它。传统做法是直接改 π0.5 的参数 — 太贵，太不稳定。

EXPO 说：**别动它。** 在旁边挂个小 MLP，叫 **edit policy** $\pi_{\text{edit}}$。这个小弟的工作就是：看 VLA 给的 action $a$，然后吐出一个修正量 $\hat{a}$，bounded 在 $[-\beta, \beta]$ 之间。

最终执行的 action 是：

$$\tilde{a} = a + \hat{a}$$

VLA 给个大致对的方向，edit policy 微调一下。

**为什么这样 work？** 因为 VLA 已经知道"大概怎么抓鸡蛋"，它只是不知道"在这个具体鸡蛋、这个具体角度下要稍微多用0.02的力"。这个微小修正，小 MLP 完全能 learn。

edit policy 的训练 loss:

$$\mathcal{L}(\pi_{\text{edit}}) = -\mathbb{E}\left[ Q_\phi(s, a + \hat{a}) - \alpha \log \pi_{\text{edit}}(\hat{a} | s, a) \right]$$

翻译成人话：**让 edit policy 输出的修正，能最大化 Q 值，同时保持一定的 entropy（探索）**。$\alpha$ 是温度，控制探索强度。第一项是 "往高 Q 方向走"，第二项是 "别太确定，多采样几个"。

---

## OTF Policy：Q 当裁判

光有 edit policy 还不够。还有个问题：**到底该信 VLA，还是信 edit？**

EXPO 的答案是：**让 Q-function 当裁判。**

每次决策:
1. VLA 随机采 8 个 action chunks (它的 stochasticity)
2. edit policy 给每个 chunk 加修正
3. 总共 16 个候选 (8 base + 8 edited)
4. 全部丢给 Q-function 评估
5. 选 Q 最大的那个执行

$$\tilde{a}^* = \arg\max_{a \in \cup_{i=1}^{N}\{a_i, \tilde{a}_i\}} Q_\phi(s, a)$$

直觉：**VLA 给候选，Q 选最好的**。这比直接 mean/softmax 输出强多了，因为你不知道哪个 VLA sample 是"好的"。Q 知道。

---

## Q 怎么训？TD learning + RedQ ensemble

Q-function 的训练 loss (公式3):

$$\mathcal{L}(\phi) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \left[ \left( r_t + \gamma Q_{\phi'}(s_{t+1}, \tilde{a}_{t+1}^*) - Q_\phi(s_t, a_t) \right)^2 \right]$$

人话：**当前 state-action 的 Q 值，应该等于 "即时 reward + 下一 state 的最佳 Q 值"**。这是 TD learning 的标准套路。

但有个坑：**Q 容易 overestimation**。特别是我们 UTD=20 (每个 env step 做 20 次 gradient update)，Q 值容易越训越大爆炸。

解法：**RedQ ensemble** (https://openreview.net/forum?id=AY8zfZm0tDd)。训 10 个 Q-network，target 用随机抽 2 个的最小值。取 min 就是保守估计，抑制 overestimation。

直觉：**10 个评委，取最苛刻的两个的最低分，避免集体吹捧。**

---

## EXPO-FT 的两个新东西

到这里 EXPO 已经够了，但 EXPO-FT 还要加两个东西才能用到真实 VLA 上。

### 第一个：Action Chunking

π0.5 不是输出单个 action，它输出一整段 future actions，比如 16 步。然后执行前 4-8 步，再重新 plan。

这叫 **action chunking**。问题：原始 EXPO 是为 single-step action 设计的。直接套到 chunk 上不对。

EXPO-FT 的改法：**把 chunk 当作一个整体来 train**。

- edit policy 输出整个 chunk 的修正 $\hat{a}_{t:t+C}$
- Q-function 评估整个 chunk
- TD target 用 $s_{t+C}$ (chunk 结束的 state)，而不是 $s_{t+1}$

公式4 (edit loss):

$$\mathcal{L}(\pi_{\text{edit}}) = -\mathbb{E}_{(s_t, a_{t:t+C}) \sim \mathcal{D}, \hat{a}_{t:t+C} \sim \pi_{\text{edit}}} \left[ Q_\phi(s_t, a_{t:t+C} + \hat{a}_{t:t+C}) - \alpha \log \pi_{\text{edit}}(\hat{a}_{t:t+C} | s_t, a_{t:t+C}) \right]$$

公式5 (TD loss):

$$\mathcal{L}(\phi) = \mathbb{E}_{(s_t, a_{t:t+C}, s_{t+C}) \sim \mathcal{D}} \left[ \left( r_t + \gamma Q_{\phi'}(s_{t+C}, \tilde{a}_{t+C:t+2C}^*) - Q_\phi(s_t, a_{t:t+C}) \right)^2 \right]$$

人话：**把 MDP 重新定义为"每一步是一个 chunk"**。一个 meta-step = 一个 chunk = C 个真实 step。Q 学的是"这一整段 chunk 有多好"，而不是"这个 single action 有多好"。

直觉：**对齐了 VLA 的输出接口和 RL 的训练接口**。VLA 说 chunk，RL 也听 chunk。不然 VLA 输出 16 步，RL 只训 1 步，coordination 全乱了。

### 第二个：Human-in-the-Loop

这个是借 HIL-SERL (https://arxiv.org/abs/2410.21845) 和 HG-DAgger (https://doi.org/10.1109/ICRA.2019.8793698) 的思想，但用得很巧妙。

机制：**人拿着 SpaceMouse 站旁边**。机器人执行 action chunk 的时候，人看着觉得"这步要出事了"，就拨一下 SpaceMouse，覆盖当前几个 action。

人给的 action $\bar{a}_{t':t''}$ 被插进 chunk 里:

$$(a_t, \dots, \bar{a}_{t'}, \dots, \bar{a}_{t''}, \dots, a_{t+C-1})$$

整个 chunk (含人介入的部分) 都进 replay buffer。

**关键区别：** HG-DAgger 只用这些 intervention 做 BC。EXPO-FT 把它们当作 RL 的 off-policy data，同时训 Q 和 edit policy。

直觉：**人不是"教"机器人，是"演示"几个高 return 的轨迹**，让 Q-function 早期就能学到 "好的 trajectory 长什么样"。然后 RL 在这个基础上 self-improve，最终超越人的水平。

Figure 4 的 intervention rate 曲线：开始时高 (>50%)，后期降到 0%。这是 "人在早期 carry，RL 在后期接管" 的典型 shape。

---

## 系统架构：Learner 和 Actor 分家

这部分是工程上的精彩点。

你看 π0.5 是 ~3B 参数。inference 一次要几百 ms。Q-ensemble 是 10 个 network。edit policy 还在跑。再加 real-time 机器人控制 10Hz。

如果全在一个进程里跑，会卡死。所以 EXPO-FT 拆成两个：

```
┌────────────────────┐         ┌────────────────────┐
│  LEARNER (GPU)    │         │  ACTOR (Robot)    │
│                   │  <->    │                    │
│  - VLA inference  │         │  - Step env       │
│  - Q training     │         │  - Human interven │
│  - Edit training  │         │  - Reward compute │
│  - Replay buffer  │         │  - Send tuples    │
└────────────────────┘         └────────────────────┘
```

Learner 在 GPU server 上慢慢训。Actor 在机器人旁边实时跑，收集 (s, a, r, s') tuple 发回去。

**Synchronous vs Asynchronous：**
- GPU 少 (≤2): 同步。简单，但慢。
- GPU 多: 异步。环境交互和 gradient update 并行。快，但 data 会 stale。

paper 用同步，因为只有 2 块 H200。

直觉：**这是 real-robot RL 的工程必备**。SERL (https://arxiv.org/abs/2401.16013) 也是这么搞的。没有这种解耦，real-time training 不可能。

---

## Critic 不共享 VLA encoder — 反直觉但合理

这里有个有意思的细节。

VLA 有自己的 visual encoder (大 ViT 或 SigLIP)。Q-function 也需要看图。最自然的想法是**共享 encoder** — 用 VLA 的 representation。

但 EXPO-FT 不共享。Critic 用了一个独立的 **ResNet-50** (3,4,6,3 blocks, 64 filters, 512-dim embedding)。

理由 paper 说得很直白：**VLA encoder 太大，shared inference 在 tight actor-learner loop 里 prohibitive**。

直觉：**RL 训练要 query Q-function 几十次每步** (16 candidates × ensemble queries × target queries)。如果每次都要 forward 一个大 ViT，根本跑不动。小 ResNet-50 足够 representation，快 10 倍。

代价：representation 可能不如 VLA encoder 那么好。但 paper 实验证明 30/30，所以这个 trade-off 是值得的。

---

## 训练流程：从 SFT warm-start 到 RL

实际训练一个新 task 的完整流程：

1. **Zero-shot 试一下 VLA**。看 VLA 直接能不能做。如果成功率高 (>40%)，直接进 RL。如果不行，下一步。

2. **SFT warm-start**。用 LoRA (https://openreview.net/forum?id=nZeVKeeFYf9) finetune VLA。LoRA 只训 low-rank adapters，便宜且避免 catastrophic forgetting。训到 ~40% 成功率。

3. **RL finetuning**。开始 EXPO-FT loop:
   - 机器人 rollout
   - 人随时准备 intervene
   - (s, a, r, s') 进 replay buffer
   - 每 episode 或每 batch 做 updates
   - 训到 100%

直觉：**SFT 给一个"及格"的起点，RL 把"及格"推到"完美"**。SFT 自己只能到 40-70%，但 RL 能继续往上推。

---

## 实验结果到底有多强

8 个 task，平均 19.1 分钟 online data，**全部 30/30 = 100%**。

对比：

| 方法 | 平均成功率 | 备注 |
|------|-----------|------|
| EXPO-FT | **30/30** | 19分钟 |
| HG-DAgger | 22.1/30 | 只 BC |
| SFT | 20.5/30 | 仅 supervised |
| DSRL | 19/30 | latent steering |
| HIL-SERL | 5.5/30 | from scratch, 在复杂 setting 下崩 |

**HIL-SERL 为什么崩？** paper 解释：他们的实验 initial state randomization 比 HIL-SERL 原版大得多。HIL-SERL 没 VLA prior，扛不住大 state space。

**DSRL 为什么不如 EXPO-FT？** DSRL 是在 VLA 的 latent space 里 steer，只能在 prior distribution 的 modes 内优化。如果 task 需要 prior 没见过的 action，DSRL 表达不出来。EXPO-FT 的 additive edit 可以 express prior 之外的 (虽然 bounded)。

**HG-DAgger 为什么在 Pool Shot 上只有 14/30？** Pool Shot 是 dynamic task — 击球力度差一点，球轨迹差很多。BC 有 compounding errors 问题 (DAgger 原论文 https://proceedings.mlr.press/v15/ross11a.html 讲过)：小的 misalignment 累积，policy 自己 fix 不回来。RL 能 self-correct。

---

## 8 个 task 都是什么

Figure 3 列了 8 个 task，覆盖很广:

1. **Egg Flip**: 用铲子翻 3D 打印的煎蛋。Contact-rich + dynamic + 精确时机。
2. **String Light Routing I/II/Insert**: 三个 sub-task — 挂两个灯泡 + 插电源。Long-horizon + 精确对位。
3. **Candy Scoop**: 用勺子舀糖果倒到另一个容器。Deformable + 视觉 messy。
4. **Cube Pick**: 抓方块。Large initial state randomization。
5. **Flower Insert**: 把花插进细颈瓶。Tight tolerance。
6. **Pool Shot**: 击球入袋。Dynamic + 精确 force control。

每个 task 都有不同的 hyperparameter (Table 4):

- Precision task (Flower, Pool, Light Insert): edit scale = 0.05 (小修正)
- Dynamic task (Egg, Candy): edit scale = 0.2 (大修正)
- Replan C = 4 或 8 (high precision 用小 C，更多反馈)

直觉：**hyperparameter 反映 task 结构**。precision 不允许大扰动；dynamic 需要灵活；long-horizon 需要更多 compute per episode。

---

## Reward 设计：sparse binary

所有 task 用 **rule-based binary reward**：成功 = 1，否则 = 0。Success detector >95% accuracy。

例子:
- **Egg Flip**: 检测 yolk pixel 在 pan region 的 orientation 翻转
- **Pool Shot**: 三个条件 AND — cue 不接触 black, black 入袋, cue 不入袋
- **Flower Insert**: gripper 在 target region + 无 stem pixel 在瓶口以下

直觉：**sparse binary reward 是 friend，不是 enemy**。Dense reward shaping 容易引入 local optima 和 reward hacking。binary reward + VLA prior + human intervention 让 exploration 变得 tractable。

---

## 一些容易被忽略的细节

### Augmentation (OpenPI-style)

每张图都做:
- Random crop 95% + resize 224×224
- Rotation ±5°
- Color jitter (brightness/contrast/saturation ±0.1)
- Side + wrist 两个 camera view 独立 augment
- Current + next observation 独立 augment

直觉：**small replay buffer 下必须防 overfit**。每张图"看"很多次，augmentation 让 Q-function 不会 overfit 到 specific pixels。

### Update-to-data = 20

每个 env step，做 20 次 gradient update。这是 sample-efficient RL 的关键。

直觉：**榨干每个 sample 的信息**。real-robot data 贵，每个 sample 要用 20 次。但高 UTD 会让 Q 爆炸 — 所以需要 RedQ ensemble。

### 8 + 8 = 16 candidates

VLA 采 8 个 chunk，edit 给每个加修正 = 16 候选。选 Q 最大的执行。

直觉：**8 个是 diversity 和 compute 的平衡**。再多 inference 慢；再少 diversity 不够。

---

## Limitations — paper 自己承认的 + 我猜的

paper 自己说:
1. **Human reset**: 每个 episode 之间要人 reset 环境。累。
2. **Computational cost**: VLA 几B参数，inference 频率受限。

我猜的额外问题:
3. **Edit bound $\beta$ 是 hand-tuned**: 不同 task 0.05/0.1/0.2，能不能自适应?
4. **Binary reward 需要task-specific success detector**: 跨 task generalization 受限
5. **30 trials 评估**: 统计上 confidence interval 不算强
6. **Long-horizon**: String Light Routing 是 3 stage，真 long-horizon (50+) 没测
7. **Cross-embodiment**: Franka 训的能不能 transfer 到 UR5?

---

## 这个工作的 research 地位

我觉得这是 **"VLA finetuning for deployment" 的 milestone**。

之前的 work 要么 from scratch (HIL-SERL), 要么 latent steering (DSRL), 要么 on-policy (SimpleVLA-RL, πRL)。EXPO-FT 是第一个同时做到:
- **full VLA finetune** (not auxiliary policy)
- **action chunking** (native VLA interface)
- **human-in-the-loop** (sample efficiency)
- **off-policy** (sample efficiency)
- **100% success rate** (reliable)

跟 Physical Intelligence 的 π0.6 (https://arxiv.org/abs/2511.14759, offline advantage conditioning) 和 Google DeepMind 的 Gemini Robotics 1.5 (https://arxiv.org/abs/2510.03342) 形成竞争格局。

下一步预测:
1. **Automated reset** via learned reset policy
2. **Multi-task RL**: 同时 finetune 多个 task
3. **Cross-embodiment**: 一个 VLA finetune 到多个 robot
4. **World model** integration: 让 Q 用 world model imagination, 更 sample efficient
5. **Reward learning** from preferences (RLHF): 减少 rule-based engineering

---

## 一句话总结 (给你)

**EXPO-FT = "VLA 给候选，edit policy 修正，Q 当裁判，人早期 carry — 20 分钟训到 100%"**

更深的 intuition: **不要让一个大模型干所有事**。VLA 干 representation + base behavior，Q 干 value judgment，edit policy 干 local refinement，human 干 early exploration。每个组件干自己最擅长的，组合起来比任何单一方法都强。

这跟 AlphaGo 的 "policy network + value network + MCTS"、或者 ChatGPT 的 "pretrain + RLHF + tool use" 是同一种设计哲学：**modular, 各司其长**。

希望这个人话版本能 build 起来 intuition。需要展开哪一块继续问。

---

# EXPO-FT: Sample-Efficient RL Finetuning for VLA Models 深度解析

## 1. 整体动机与核心论点

这篇paper要解决的核心问题是：**pretrained VLA models (像 π0.5, OpenVLA, RT-2) 在real-world deployment上始终达不到可靠的成功率**，而现有的两条路径都有缺陷：

- **RL from scratch** (如 HIL-SERL, SERL): sample efficient但无法利用VLA的semantic/behavioral priors
- **VLA finetuning** (如 DSRL, SimpleVLA-RL, πRL): 利用prior但要么convergence不稳定，要么sample efficiency不够

EXPO-FT 的claim是：**通过将 EXPO 算法扩展到 action chunking + human-in-the-loop interventions，可以在 ~19分钟online data 内将 π0.5 finetune 到 30/30 成功率**，跨8个高难度任务（鸡蛋翻转、台球击球、插花、灯串布线等）。

项目主页: https://pd-perry.github.io/expo-ft/
EXPO原始论文: https://openreview.net/forum?id=aFjSjkB6CV
π0.5论文: https://arxiv.org/abs/2504.16054

---

## 2. EXPO 基础回顾 (build intuition的关键)

要理解 EXPO-FT，必须先理解 EXPO。EXPO的核心思想可以用一个intuition概括：**"不要直接finetune一个庞大的flow/diffusion policy，而是在它旁边挂一个小corrector，然后让Q-function决定听谁的"**。

### 2.1 为什么不能直接对VLA做policy gradient？

现代VLA (π0.5, OpenVLA) 多用 flow matching 或 diffusion policy，这类policy的likelihood计算需要解ODE/SDE，backprop通过整个denoising过程代价巨大且数值不稳定。SAC-style的policy gradient $\nabla_\theta Q(s, a)$ 在这里会失效。

### 2.2 EXPO的解法：Edit Policy

EXPO维持两个policy：

1. **Base policy $\pi_{\text{VLA}}$**: 大VLA模型，参数frozen或slow-updated，输出 base action $a$
2. **Edit policy $\pi_{\text{edit}}$**: 小型 MLP (3层256宽)，输出 edit $\hat{a} \in [-\beta, \beta]$ (bounded)，加到base action上得到 $\tilde{a} = a + \hat{a}$

Edit policy 的 loss (论文公式1):

$$\mathcal{L}(\pi_{\text{edit}}) = -\mathbb{E}_{(s_t, a_t) \sim \mathcal{D}, \hat{a}_t \sim \pi_{\text{edit}}} \left[ Q_\phi(s_t, a_t + \hat{a}_t) - \alpha \log \pi_{\text{edit}}(\hat{a}_t | s_t, a_t) \right]$$

变量解释:
- $s_t$: state (RGB images + proprio)
- $a_t$: base action sampled from $\pi_{\text{VLA}}$
- $\hat{a}_t$: edit sampled from $\pi_{\text{edit}}$
- $Q_\phi$: Q-network with parameters $\phi$
- $\alpha$: temperature (entropy regularization weight), 初始化为1.0
- $\beta$: edit bound (0.05/0.1/0.2 per task)

直觉：**edit policy 学的是"如何在base action的小邻域内做最大化Q的修正"**，类似Residual Policy Learning的思想但用stochastic policy + entropy regularization。

### 2.3 OTF (On-The-Fly) Policy

推理时和TD backup时，EXPO 用 OTF policy 在 base action 和 edited action 中选Q最大的:

$$\tilde{a}^* = \arg\max_{a \in \cup_{i=1}^{N}\{a_i, \tilde{a}_i\}} Q_\phi(s, a)$$

这里 $N$ 通常取8 (paper中 base draws 8 stochastic chunks, edit draws 8 chunks, total 16 candidates)。这就是 **value-guided sampling**，类似 Diffusion Q-Learning (Q-score matching) 或 FASTER (https://arxiv.org/abs/2604.19730) 的思想。

直觉：**OTF policy 不直接修改VLA参数，而是在inference时用Q作"裁判"在VLA的多个采样里选最好的**，这样保留了VLA的prior，同时用Q提供task-specific的value guidance。

### 2.4 Q-function 训练

TD loss (公式3):

$$\mathcal{L}(\phi) = \mathbb{E}_{(s_t, a_t, s_{t+1}) \sim \mathcal{D}} \left[ \left( r_t + \gamma Q_{\phi'}(s_{t+1}, \tilde{a}_{t+1}^*) - Q_\phi(s_t, a_t) \right)^2 \right]$$

变量解释:
- $\phi'$: target network parameters (Polyak averaged)
- $\gamma = 0.99$: discount factor
- $\tilde{a}_{t+1}^*$: OTF action at next state
- $r_t$: sparse binary reward

使用 **RedQ-style ensemble**: 10个Q-networks, target取2个random的最小值 (减少overestimation), Polyak coefficient $\tau_Q = 5 \times 10^{-3}$。这是从 RedQ (https://openreview.net/forum?id=AY8zfZm0tDd) 借鉴的trick，对于高UTD (update-to-data ratio = 20) 训练至关重要。

---

## 3. EXPO-FT 的两大核心创新

### 3.1 Temporally Extended Actions (Action Chunking)

**这是EXPO-FT相对于EXPO最重要的算法扩展**。

现代VLA几乎都使用 action chunking: 一次predict $H$ 个future actions $a_{t:t+H}$, 执行前 $C \leq H$ 个，这样减少了replanning频率，提供temporal abstraction。π0.5 的 $H$ 通常是16，$C$ 通常是4-8 (task-dependent)。

EXPO原始公式只对single-step action $a_t$ 工作。EXPO-FT 把所有公式扩展到 chunk level:

**Edit policy loss (公式4):**

$$\mathcal{L}(\pi_{\text{edit}}) = -\mathbb{E}_{(s_t, a_{t:t+C}) \sim \mathcal{D}, \hat{a}_{t:t+C} \sim \pi_{\text{edit}}} \left[ Q_\phi(s_t, a_{t:t+C} + \hat{a}_{t:t+C}) - \alpha \log \pi_{\text{edit}}(\hat{a}_{t:t+C} | s_t, a_{t:t+C}) \right]$$

**TD loss (公式5):**

$$\mathcal{L}(\phi) = \mathbb{E}_{(s_t, a_{t:t+C}, s_{t+C}) \sim \mathcal{D}} \left[ \left( r_t + \gamma Q_{\phi'}(s_{t+C}, \tilde{a}_{t+C:t+2C}^*) - Q_\phi(s_t, a_{t:t+C}) \right)^2 \right]$$

关键变化:
- Q-function 现在输入整个 chunk $a_{t:t+C}$ 而非 single action
- TD target 用 $s_{t+C}$ (chunk结束时的state) 而非 $s_{t+1}$
- $\gamma$ 折扣现在隐含了 $C$ 步而非1步 (这里paper的写法略简化，理论上应该是 $\gamma^C$，但实际中 $\gamma^{C}$ 与 $\gamma$ 在 $C$ 较小且 $\gamma$ 接近1时差别不大，paper直接用 $\gamma$)

直觉：**把MDP重新定义为 "meta-MDP"，每个meta-step是一个chunk**，这样Q-function学的就是"这个chunk有多好"，而非"这个action有多好"。这种temporal abstraction让RL训练与VLA的输出接口对齐。

### 3.2 Human-in-the-Loop Interventions

借鉴 HIL-SERL (https://arxiv.org/abs/2410.21845) 和 HG-DAgger (https://doi.org/10.1109/ICRA.2019.8793698) 的思想，但有关键不同：

**机制:** 在 action chunk $a_{t:t+C}$ 执行过程中，human operator 用 SpaceMouse 可以在任意 $t'$ 到 $t''$ 之间 ($t \leq t' \leq t'' \leq t+C-1$) 介入，提供 corrective actions $\bar{a}_{t':t''}$，最终执行的 sequence 变成:

$$(a_t, \dots, \bar{a}_{t'}, \dots, \bar{a}_{t''}, \dots, a_{t+C-1})$$

整个 chunk (包含 human correction) 都加入 replay buffer $\mathcal{D}$。

**与HG-DAgger的本质区别:** 
- HG-DAgger 只用 human intervention 做 supervised learning (BC loss)
- EXPO-FT 把 intervention 作为 RL 的 off-policy data，同时 train Q-function 和 edit policy

**与HIL-SERL的本质区别:**
- HIL-SERL 从 scratch 训 Gaussian policy，无法利用 VLA prior
- EXPO-FT finetune VLA + edit policy，能跨多个task generalize

直觉：**human intervention 的作用是"填充 replay buffer 中的 high-return trajectories"**，这样 Q-function 能在早期就学到正确的 value landscape，edit policy 能学到有意义的 corrections。否则在 high-dimensional action space + sparse reward 下，纯random exploration基本不可能成功。

Figure 4 的 intervention rate 曲线也证实了这点：早期intervention rate高 (有时>50%)，随训练降低到 ~0%。

---

## 4. 系统架构

### 4.1 Learner-Actor 解耦

paper Section 4.3 描述的架构很有工程价值：

```
┌─────────────────────────────┐         ┌──────────────────────────┐
│   LEARNER (GPU server)      │  comm.  │   ACTOR (Robot side)     │
│                             │ ←─────→ │                          │
│  - π_VLA inference          │         │  - Environment stepping  │
│  - Q-ensemble training      │         │  - Human interventions   │
│  - Edit policy training     │         │  - Action execution     │
│  - Replay buffer            │         │  - Reward computation   │
└─────────────────────────────┘         └──────────────────────────┘
```

- **Synchronous mode**: GPU有限时 (≤2 GPUs) 用同步，避免async带来的stale data问题
- **Asynchronous mode**: 多GPU时用异步，环境交互和gradient update并行

直觉：**这种解耦对real-robot RL至关重要**，因为VLA inference (π0.5 ~3B params) + Q-ensemble (10 nets) 训练很慢，必须把环境交互放到独立process才能提高throughput。这与SERL (https://arxiv.org/abs/2401.16013) 的设计哲学一致。

### 4.2 Critic 架构 (重要细节)

paper Section C.1 揭示了一个反直觉的选择: **critic 不与 VLA 共享 visual encoder**。

- VLA encoder (通常是大ViT或SigLIP): frozen during RL
- Critic encoder: 独立的 **ResNet-50** (stage depths (3,4,6,3), 64 filters, 512-dim image embedding)
- Proprio: 64-dim embedding
- 融合后输入 3-layer MLP (width 256) 输出 Q value

理由: "the VLA encoder is large, making shared inference computationally prohibitive during the tight actor-learner loop"

直觉：**这是个权衡 - 共享encoder能利用更好的representation，但代价是inference慢10倍+**。这里选了速度，因为RL需要大量Q-function queries (UTD=20 + 16 candidates per decision + target networks)。

### 4.3 Action Sampling 细节

每次决策:
1. Base π0.5 draw 8 stochastic action chunks (不同 noise samples)
2. Edit policy 对每个chunk输出一个 edit
3. Edit scale (0.05/0.1/0.2) 缩放 edit
4. 16 candidates (8 base + 8 edited) 全部输入 Q-ensemble
5. 选 Q 最大的 chunk 执行 (deterministic top-Q, 非 softmax)

这个设计有几个intuition:
- **8 samples**: 平衡diversity和compute
- **Deterministic top-Q**: 训练时是 exploitation，不是 exploration。exploration来自VLA本身的stochasticity + edit policy的entropy
- **Edit scale分task**: 高precision task (Flower Insert, Pool Shot) 用 0.05, 避免大perturbation破坏精细对位；dynamic task (Egg Flip, Candy Scoop) 用 0.2, 允许更大的motion修正

---

## 5. 实验结果深度分析

### 5.1 主结果 (Table 1 & 2)

**8个任务平均:**
- EXPO-FT: **30/30** (100%)
- HG-DAgger: 22.1/30 (74%)
- SFT: 20.5/30 (68%)
- DSRL: 19/30 (63%)
- HIL-SERL: 5.5/30 (18%) [在更复杂setting下]

**Online data平均: 19.1分钟** (跨task: 14-35分钟)

直觉：**100%成功率是震惊的点** - 不是"接近可靠"，是"完全可靠"。这对real deployment意义重大，意味着可以真正考虑production use。

### 5.2 为什么 HIL-SERL 在这个 setting 下 fail？

paper Table 1 显示 HIL-SERL 在 Egg Flip (13/30), Flower Insert (8/30), Pool Shot (1/30) 上表现很差。原因分析：

1. **Larger initial state randomization**: paper 提到 "our experiments involve a substantially larger initial state space"。HIL-SERL 原论文 task (主板装配、IKEA装配) 的 initial state 比较固定。
2. **无 VLA prior**: 从scratch学 Gaussian policy，需要大量exploration来覆盖state space。19分钟对它远远不够。
3. **Dynamic task (Pool Shot)**: 击球需要精确的 force control，从scratch学很难在有限样本内收敛。

直觉：**VLA prior 的价值在于"跨state generalization"** - 即使没见过的initial state, VLA也能给出 reasonable action，然后 edit policy 做局部修正。Gaussian policy from scratch 没有这种 generalization。

### 5.3 为什么 DSRL 不如 EXPO-FT？

DSRL (https://arxiv.org/abs/2506.15799) 也是 finetune diffusion/flow policy，但用 latent space RL (perturb noise)。它的局限：

- **只能在 prior distribution 的 modes 内优化**: 如果 task 需要 prior 没见过的 action (e.g. 更大的修正), DSRL 无法 express
- **无 human-in-the-loop**: 完全 autonomous exploration，sample efficiency 受限

EXPO-FT 的 edit policy 是 **additive action-space correction**, 可以 express prior 之外的 action (虽然bounded by $\beta$); human intervention 还能直接提供 high-return trajectories。

直觉：**两种finetune方式的本质区别: latent steering = "在VLA的imagination里找最好的"; additive edit = "在VLA的output上做小修正"**。后者表达力更强，但需要更careful的训练。

### 5.4 为什么 HG-DAgger 在 dynamic task (Pool Shot 14/30) 上失败？

HG-DAgger 只做 BC on interventions。Pool Shot 需要:
- 精确的 force control (太大力 → cue ball 入袋; 太小力 → 黑球不入)
- Compounding errors: 小的 misalignment 导致击球点偏移，球轨迹完全不同

BC 的 known issue: **compounding errors** (Ross et al. DAgger paper, https://proceedings.mlr.press/v15/ross11a.html)。Policy error 累积，无法 recover。RL 的 self-improvement 才能修正这些 errors。

### 5.5 Training Curves (Figure 4)

观察几个task的training curves (虽然我看不到具体图，但从paper描述):
- **EXPO-FT**: monotonic increase to 100%, intervention rate 从高降到0
- **HIL-SERL**: 在复杂task上 flat-line 或非常缓慢上升
- **HG-DAgger**: 早期上升快但 plateau 在 suboptimal

直觉：**EXPO-FT 的曲线形状反映了两阶段: 早期 human intervention 主导 (类似于 BC warm-start), 后期 RL self-improvement 主导**。这是个非常优雅的design。

### 5.6 Episode Time (Figure 5)

paper Section A.1 提到 episode time 随训练降低。这说明 policy 不仅成功率提升，执行效率也提升。直觉：**RL 让 policy 学到了"shortest path to success"**, 而 VLA prior 可能存在不必要的 exploratory motion。

---

## 6. 详细技术细节 (Appendix C 中的goldmine)

### 6.1 π0.5 LoRA 初始化

paper 用 π0.5 + **LoRA** (https://openreview.net/forum?id=nZeVKeeFYf9) 做 supervised finetuning 初始化。这是个关键 trick:

- 不直接 finetune 全部参数 (太贵且容易catastrophic forgetting)
- 用 LoRA 只 train low-rank adapters
- RL 阶段也保留 LoRA structure? (paper 没明说, 但"initialized from a task-specific LoRA supervised-finetuning checkpoint" 暗示 RL 阶段 freeze LoRA 或继续 update LoRA)

### 6.2 数据增强 (OpenPI-style)

augmentation config:
- **Random crop 95% + resize 224x224**
- **Rotation ±5°**
- **Color jitter (brightness/contrast/saturation ±0.1)**
- 每个camera view (side + wrist) 独立增强
- Current 和 next observation 独立增强

直觉：**这种 augmentation 防止 Q-function overfit 到 specific pixels**, 在 small replay buffer 下尤其重要。

### 6.3 Optimization Hyperparameters

Global hyperparameters (Table 3):
- Adam, lr = 3e-4
- $\gamma = 0.99$
- $\alpha_0 = 1.0$
- $\tau_Q = 5 \times 10^{-3}$ (critic Polyak)
- $\tau_\pi = 10^{-3}$ (policy Polyak, 更慢)
- Batch size = 64
- **UTD = 20** (高! 20 gradient updates per env step)

直觉：**UTD=20 是 sample-efficient RL 的关键**。意味着每个env step, learner做20次gradient update。这能榨干每个sample的信息，但需要 RedQ ensemble 防止 overestimation (否则高UTD会让Q爆炸)。

### 6.4 Task-Specific Hyperparameters (Table 4)

观察 task hyperparameter 的选择 build intuition:

| Task | Edit Scale | Replan C | Updates/Ep | Training Steps |
|------|------------|----------|------------|----------------|
| Egg Flip | 0.2 | 8 | 4 | 10k |
| Light Insert | 0.05 | 4 | 3 | 9k |
| Candy Scoop | 0.2 | 8 | 6 | 12k |
| Flower Insert | 0.05 | 4 | 1 | 8k |
| Pool Shot | 0.05 | 8 | 1 | 10k |

Pattern:
- **Precision tasks (Flower, Light Insert, Pool)**: edit scale 0.05 (小修正), C=4或8
- **Dynamic tasks (Egg Flip, Candy Scoop)**: edit scale 0.2 (大修正), C=8
- **High updates/episode (Candy Scoop 6)**: 视觉messy + 多步task需要更多updates
- **Low updates/episode (Pool Shot 1, Flower 1)**: 单步critical action, 不需要太多updates

直觉：**hyperparameter 反映了 task 的本质结构** - precision task 不容忍大扰动, dynamic task 需要灵活修正, long-horizon task 需要更多 compute per episode。

### 6.5 Reward Design

paper 用 **rule-based binary reward** (success detector >95% accuracy)。例子:
- **Egg Flip**: 检测 yolk pixels 在 pan region 的 orientation 变化
- **Candy Scoop**: 检测 candies 在 scoop 上升到 threshold 后是否在 target container
- **Pool Shot**: 三个条件 (cue不接触black, black入袋, cue不入袋)
- **Flower Insert**: gripper在target region + 无stem pixel在瓶口以下

直觉：**这种 sparse binary reward 是 sample-efficient RL 的 friend, 不是 enemy**。Dense reward shaping 容易引入 local optima 和 reward hacking。Binary reward + VLA prior + human intervention 的组合让 exploration 变得 tractable。

---

## 7. 与相关工作的对比

### 7.1 EXPO-FT vs 其他 VLA RL 方法

| 方法 | 训练方式 | Prior利用 | Human介入 | Action Chunking | Off-policy |
|------|----------|-----------|-----------|------------------|------------|
| EXPO-FT | Edit + Q | 直接 | 是 | 是 | 是 |
| DSRL | Latent steering | 在modes内 | 否 | ? | 是 |
| SimpleVLA-RL | On-policy PPO | 直接 | 否 | 是 | 否 |
| πRL | On-policy | 直接 | 否 | 是 | 否 |
| ConRFT | Consistency + separate policy | 部分 | 否 | ? | 是 |
| RL Token | Auxiliary policy | 部分 | 否 | ? | 是 |
| Policy Decorator | Lightweight decorator | 部分 | 否 | ? | 是 |

EXPO-FT 的独特组合: **full VLA finetune + action chunking + human-in-loop + off-policy + edit policy**。

### 7.2 与 π0.6 的对比

π0.6 (https://arxiv.org/abs/2511.14759) 用 advantage conditioning offline RL。EXPO-FT 是 online finetune。Online 的优势: data efficiency (能从自主收集的经验中学习, 不局限于 offline data); 劣势: 需要 real-time robot interaction。π0.6 和 EXPO-FT 可能是互补的 - π0.6 大规模 offline pretrain, EXPO-FT task-specific online finetune。

### 7.3 与 Residual RL 的关系

EXPO-FT 的 edit policy 本质是 **action-space residual policy**。相关工作:
- Residual Off-Policy RL (https://arxiv.org/abs/2509.19301): 在 BC policy 上加 residual
- Policy Decorator (https://openreview.net/forum?id=e5jGTEiJMT): model-agnostic refinement
- Residual RL (Silver et al. 2018)

EXPO-FT 与这些工作的关键不同: edit policy 是 stochastic (有 entropy regularization), 用 Q-value 直接 train (而非 reward shaping), 与 VLA 的 action chunking 完全兼容。

---

## 8. Limitations 与未来方向

paper Section 6 提到:
1. **Human resets**: 需要 human 在 episode 之间 reset 环境。这对大规模 deployment 是负担。自动化 reset 是 future work。
2. **Computational cost**: finetune 几B参数的VLA inference 频率受限。

我额外的intuition about limitations:
3. **Edit bound $\beta$ 是 hand-tuned**: 不同 task 需要不同 scale, 未来可以自适应
4. **Binary reward 需要task-specific success detector**: 跨task generalization 受限
5. **Critic encoder 不共享 VLA encoder**: 可能 suboptimal representation, 但工程必需
6. **Long-horizon task**: 当前最复杂的 String Light Routing 是3-stage, 真正长 horizon (50+ steps) task 还没验证

未来方向我猜测:
- **Automated reset** via learned reset policy
- **VLA encoder distillation** to small critic encoder
- **Hierarchical RL**: high-level VLA + low-level edit policy
- **Multi-task RL**: 同时 finetune 多个 task
- **Reward learning** from human preferences (RLHF) 来减少 rule-based reward engineering
- **World model** integration: 让 Q-function 用 world model imagination, 进一步提升 sample efficiency

---

## 9. 给 Karpathy 的intuition summary

如果用一句话概括: **EXPO-FT = "用 EXPO 的 edit policy 机制finetune π0.5, 加上 action chunking 和 human intervention, 在 real robot 上 20 分钟训到 100%"**。

更深的intuition:

1. **Decouple representation from value**: VLA 提供 representation 和 base behavior, Q-function 提供task-specific value guidance. 不要让 Q-function 学 representation (太贵), 也不要让 VLA 学 value (不稳定)。

2. **Additive > Latent steering**: 在 action space 加 edit 比在 latent space steer 更具表达力, 特别是 task 需要 prior 之外的行为时。

3. **Action chunking 让 RL 接近 VLA 的"语言"**: VLA 输出 chunks, RL 也按 chunks 训练, 接口对齐让 finetune 稳定。

4. **Human intervention = off-policy expert demos**: 不是替代 RL, 是给 RL 提供 high-return seeds。这比纯 BC 强, 因为后续 RL 能超越 human demo quality。

5. **Sparse reward + VLA prior 是 sample-efficient RL 的 key**: 不要 dense reward shaping, 让 VLA 提供 prior, RL 提供 refinement。

6. **UTD=20 + RedQ ensemble**: 高 sample efficiency 的核心。每个 env step 做20次 updates, 但用 ensemble 防止 Q 爆炸。

7. **100% 成功率的可达性**: 这是个重要 milestone - RL finetune VLA 可以达到 "production-ready", 不只是"good enough"。

---

## 10. 关键 references

- **EXPO**: https://openreview.net/forum?id=aFjSjkB6CV (基础算法)
- **π0.5**: https://arxiv.org/abs/2504.16054 (VLA backbone)
- **HIL-SERL**: https://arxiv.org/abs/2410.21845 (human-in-loop baseline)
- **HG-DAgger**: https://doi.org/10.1109/ICRA.2019.8793698 (intervention baseline)
- **DSRL**: https://arxiv.org/abs/2506.15799 (latent steering baseline)
- **RedQ**: https://openreview.net/forum?id=AY8zfZm0tDd (Q ensemble trick)
- **SERL**: https://arxiv.org/abs/2401.16013 (sample-efficient real RL system)
- **DAgger**: https://proceedings.mlr.press/v15/ross11a.html (intervention原始思想)
- **LoRA**: https://openreview.net/forum?id=nZeVKeeFYf9 (parameter-efficient finetuning)
- **π0.6**: https://arxiv.org/abs/2511.14759 (offline RL VLA 对比)
- **ResNet-50**: https://arxiv.org/abs/1512.03385 (critic backbone)
- **FASTER**: https://arxiv.org/abs/2604.19730 (value-guided sampling for diffusion)
- **Policy Decorator**: https://openreview.net/forum?id=e5jGTEiJMT (residual policy)
- **Residual Off-Policy RL**: https://arxiv.org/abs/2509.19301
- **RL Token**: https://arxiv.org/abs/2604.23073 (auxiliary policy approach)
- **Diffusion Policy Policy Optimization**: https://openreview.net/forum?id=mEpqHvbD2h (DPPO)
- **πRL**: https://arxiv.org/abs/2510.25889 (online RL for flow VLA)
- **SimpleVLA-RL**: https://openreview.net/forum?id=TQhSodCM4r (scaling VLA RL)
- **Interactive Post-Training for VLA**: https://arxiv.org/abs/2505.17016
- **VLA-RL**: https://arxiv.org/abs/2505.18719
- **ConRFT**: https://arxiv.org/abs/2502.05450
- **TQL**: https://arxiv.org/abs/2602.01439 (Transformer Q-Learning, 同作者)
- **Value Flows**: https://arxiv.org/abs/2510.07650 (同作者)
- **Gemini Robotics 1.5**: https://arxiv.org/abs/2510.03342 (另一个 generalist robot)
- **RL-100**: https://arxiv.org/abs/2510.14830 (real-world RL at scale)
- **GRRL**: https://arxiv.org/abs/2512.01801 (dexterous manipulation)
- **Align-then-Steer**: https://openreview.net/forum?id=T3i7Ifeatk (latent guidance)
- **Q-score matching**: https://openreview.net/forum?id=StkLULT1i1
- **RLPD**: (Ball et al. ICML 2023, Efficient Online RL with Offline Data)
- **Implicit Imitation Guidance**: https://openreview.net/forum?id=CgupPwA40q (同作者)
- **What matters for batch online RL**: https://openreview.net/forum?id=usw1NVkczu (同作者)
- **Q-learning with adjoint matching**: https://openreview.net/forum?id=vd4eNAdt06
- **Policy Agnostic RL**: https://arxiv.org/abs/2412.06685
- **Self-improving VLA with residual RL**: https://openreview.net/forum?id=eUGoqrZ6Ea
- **RLIF**: https://proceedings.iclr.cc/paper_files/paper/2024/file/9c537882044c8b5352c363e840872ddb-Paper-Conference.pdf

---

## 11. 我对这个工作的整体评价

**优点:**
- 100% 成功率 + 19分钟 sample efficiency 是非常 strong 的 result
- 系统设计完整 (learner-actor 解耦, augmentation, ensemble, etc.)
- Open-source codebase 对社区价值大
- Ablation 通过 task hyperparameter 间接体现
- 8个 task 跨 precision/dynamic/deformable, 覆盖面广

**值得追问的点:**
1. 30 trials 评估样本量偏小, 100% 在统计意义上 confidence interval 如何?
2. 没有明确的 ablation: 如果不用 human intervention, 收敛需要多久? 如果不用 action chunking, single-step EXPO 在这些 task 上表现如何?
3. Edit policy 的 capacity (3x256) 是否足够? 更大 edit policy 能否进一步提升?
4. Pool Shot 是 dynamic task, 但 paper 没测更多 dynamic task (e.g. 投掷, 抓取移动物体)
5. 跨 embodiment 泛化: 不同 robot (Franka vs UR5 vs xArm) 是否需要重新 SFT?

**这个 work 在 research roadmap 中的位置:**
- 它是 "VLA finetuning for real deployment" 的一个 milestone
- 与 OpenAI/Physical Intelligence 的 π0.6, Google DeepMind 的 Gemini Robotics 1.5 形成 "real-world generalist robot" 的竞争格局
- 下一步预测: 1) 更大规模 multi-task RL finetune; 2) Automated curriculum; 3) World model + RL; 4) Cross-embodiment RL finetuning

希望这个深度解析帮助 build intuition about why EXPO-FT works 和 where it fits in the broader research landscape!
