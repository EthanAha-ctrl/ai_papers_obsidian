---
source_pdf: ClosedLoop_NVIDIA.pdf
paper_sha256: b1aad1c6c7a346128389f00c06309dfabe806762c37b3b65cb71f8096a137dcd
processed_at: '2026-08-03T16:03:26-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：GTR²S到底干了什么

Andrej, 我把上一篇技术拆解翻译成"白板讲解"的版本。还是要带公式和数据,因为intuition不能靠口号build,得靠具体数字anchor。

---

## 一、故事起点：开车模型现在怎么玩

Autonomous driving end-to-end planning现在主流分两派:

**派别A (Continuous Regression)**: 神经网络直接输出一条轨迹 $a = (x_1, y_1, \theta_1, x_2, y_2, \theta_2, \dots, x_T, y_T, \theta_T)$,每个waypoint都回归。代表:UniAD, VAD。
- 优点:简单直接。
- 缺点:开车本质是**multi-modal**的——前面有辆慢车,你可以左道超、右道超、跟车、刹车,好几种合理选择。让一个network回归"唯一答案"会把multi-modality平均掉,得到一个骑在车道线上的"中庸解"。

**派别B (Trajectory Scoring)**: 预先定义一个大的trajectory vocabulary $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$,比如n=8192条候选轨迹。模型只学一个scorer $\pi_\theta(a_i | s_t)$,最后 $\arg\max_i \pi_\theta(a_i | s_t)$ 选一条。代表:Hydra-MDP, GTRS-Dense, DriveSuprem。
- 优点:天然处理multi-modality,softmax会自动把概率分到多个合理候选上。
- 缺点:action space大,credit assignment难。

GTR²S走的是派别B。

---

## 二、要解决的核心痛点:Open-loop训练的坑

现在这些trajectory scorer都在NAVSIM这种**open-loop dataset**上训练。什么叫open-loop?

$$\mathcal{L}_{\text{IL}} = -\sum_t \log \pi_\theta(a^*_t | s_t)$$

意思是:看一帧 $s_t$,预测人类会选哪条轨迹 $a^*_t$,跟ground truth比loss。每一帧都是独立的,模型只在自己跑的轨迹上看过世界。

但真实部署是**closed-loop**:模型自己选action,execute,世界变化,再看下一帧。问题来了:

- 第0帧,模型预测的轨迹跟人类差一点点,可能就5cm偏移。
- 这5cm偏移导致 $s_1$ 跟人类见过的 $s_1$ 不一样了。
- 模型在"没见过的 $s_1$"上预测更差,可能偏10cm。
- 越偏越远,这个叫**compounding error**,DAgger paper (https://arxiv.org/abs/1011.0686) 早就讨论过。

类比:你照着菜谱炒菜,每一步都照做,但盐多放了5g,后面加水就按原比例加,结果越来越咸。你没见过"咸了之后的状态"。

GTRS-Dense (pretrained baseline) 在HUGSIM closed-loop上overall HD-Score只有**9.8**(满分100),说明open-loop训出来的模型在closed-loop下基本**走不动**。

---

## 三、解法:把模型塞进simulator,自己开,自己学

这就是reinforcement fine-tuning。Recipe很straightforward:

1. 用HUGSIM (3DGS渲染的photo-realistic driving simulator) 起一个closed-loop环境。
2. 把pretrained GTRS-Dense模型塞进去当policy,自己跑rollout。
3. 跑出来的 $(s_t, a_t, r_t)$ 用PPO做更新。

这个recipe之前RAD (https://arxiv.org/abs/2502.13144) 做过,但有两个limitation:
- RAD的action space < 100候选,GTR²S scale到8192。
- RAD还需要人类demo算reward,GTR²S完全rule-based reward,不依赖human。

---

## 四、三个工程要点(每个都重要)

### 4.1 Parallelized Data Collection

3DGS渲染一帧比CARLA慢一个数量级,因为要做gaussian splatting (https://arxiv.org/abs/2308.14737)。单GPU串行rollout基本走不动。

解法:用IMPALA架构 (https://arxiv.org/abs/1802.01561),多个GPU instance并行跑environment,collect完一个scenario的trajectory后发给learner做PPO update。8×A100,batch size 192,3 epochs。

### 4.2 Reward设计 — 这里有个反直觉的细节

每一步的reward:

$$r_t = RC_t \cdot \Big(\prod_{p_t \in P} p_t\Big) - \prod_{m_t \in M}(1 - m_t)$$

逐项:
- $RC_t$ — Route Completion,这一步沿reference route前进了多少。范围 $[0,1]$,这是progress信号。
- $P$ — soft penalty集合,每个 $p_t \in [0,1]$ 表示某项soft metric的compliance (comfort, TTC)。越接近1越好。
- $M$ — hard penalty集合,每个 $m_t \in \{0,1\}$ 表示某项hard violation (collision, off-road)。

逻辑:
- 第一项 $RC_t \cdot \prod p_t$:progress只有在comfortable/safe的时候才有分,comfort差就被打折。
- 第二项 $\prod(1-m_t)$:没hard violation时 = 1,有任何hard violation时 = 0。

**反直觉的地方**:如果撞车了 ($m_t=1$),第二项变0,reward不扣分;如果完美一步 ($m_t=0$),第二项 = 1,reward反而扣1分。看起来"撞车reward更高"。

为什么这么设计?因为GTR²S相对CaRL (https://arxiv.org/abs/2504.17838) 改了策略:CaRL撞车就终止episode,GTR²S**让agent继续开**。如果撞车直接给大负reward然后终止,agent只学到"撞车状态→放弃"。让reward在violation后保持信号,agent能学"撞了之后怎么恢复"。

这个细节看似小,其实是让大action space + closed-loop fine-tuning能work的关键之一。

### 4.3 KL Regularization防止Catastrophic Forgetting

PPO的clipped surrogate objective:

$$\mathcal{L}(\theta) = \mathbb{E}\Big[\min\big(k_t(\theta) \cdot A_t,\ \mathrm{clip}(k_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)\Big]$$

其中 $k_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是importance weight,$A_t$ 是advantage,$\epsilon$ 控制trust region。

这个objective只限制per-sample的ratio,不限制整体policy分布。在LLM RLHF里早就发现纯PPO会导致catastrophic forgetting——模型helpful了但变笨了。

GTR²S加了一个KL term(具体form report没写,通常是 $\beta \cdot \mathrm{KL}(\pi_{\text{pretrained}} \| \pi_\theta)$)拉回pretrained分布。Ablation里KL的contribution是overall HD从12.3拉到14.3,大概+2分,贡献显著。

类比LLM RLHF的KL to SFT policy:https://arxiv.org/abs/2203.02155

---

## 五、Advantage Function选择 — 这是这篇paper最干净的ablation

GAE公式:

$$A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\delta_t$ — TD error,1-step的"惊喜"。
- $\gamma$ — discount factor,典型0.99。
- $\lambda \in [0,1]$ — bias-variance trade-off:
  - $\lambda=0$: $A_t = \delta_t$,pure 1-step TD,low variance high bias。
  - $\lambda=1$: $A_t = \sum \gamma^l r_{t+l} - V(s_t)$,Monte Carlo return,high variance low bias。

Ablation (Table 2)直接对比三种advantage:

| Advantage | Overall RC | Overall HD |
|---|---|---|
| $\sum_{t'} r_{t'}$ (return, $\lambda=1$) | 26.9 | 16.7 |
| $\delta_t$ (1-step TD, $\lambda=0$) | 26.5 | 17.6 |
| $A_t$ (GAE, $\lambda\in(0,1)$) | **29.5** | **19.6** |

Intuition:driving reward高variance(随机traffic导致同state下return差异大),纯Monte Carlo噪声大;但又需要多步信号才能看到"现在这一步的选择会怎么影响5秒后",纯1-step TD太短视。GAE用 $\lambda$ 在两者间插值,典型 $\lambda=0.95$。

这个ablation告诉我们的general lesson:**任何medium-horizon, stochastic环境下的RL fine-tuning,GAE几乎一定优于return或pure TD**。

---

## 六、实验数据读法

Table 1 (HUGSIM KITTI-360, out-of-domain eval):

| Method | Overall RC | Overall HD |
|---|---|---|
| UniAD | 12.4 | 2.2 |
| VAD | 12.4 | 1.6 |
| LTF | 15.3 | 2.0 |
| GTRS-Dense (pretrained baseline) | 17.2 | 9.8 |
| **GTR²S** | **29.5** | **19.6** |

读法:
- HD-Score从9.8到19.6,提升**+9.8绝对分**,相对提升100%。
- RC从17.2到29.5,提升**+12.3绝对分**,相对提升70%。
- 这是**out-of-domain eval**——训练用nuScenes+Waymo,测在KITTI-360,scenario完全没见过。说明学到的不是simulator-specific overfitting,是general closed-loop driving ability。

**但要诚实说**:Hard/Extreme场景几乎没改善(HD: 2.6→2.3,1.3→2.4)。Paper承认"some cases unsolvable"(比如narrow road对头车)。Closed-loop benchmark的天花板还在那里。

---

## 七、为什么这个Recipe能Scale到8192 Action Space

这是paper的真正technical contribution,虽然report写得很compact。我的intuition:

1. **Pretrain init质量高**:GTRS-Dense已经在NAVSIM上训得不错,softmax分布基本concentrate在合理候选上,RL不需要从0 explore。
2. **KL anchor防drift**:8192维softmax很容易collapse到极端分布(比如全probability给一个trajectory),KL拉回pretrained分布。
3. **Value function $V(s_t)$ 提供credit assignment**:8192候选里只选1个,纯PG信号极sparse。Value baseline把"这一步是不是good state"的信息propagate给所有候选。
4. **Scoring heads frozen**:Fig.1 caption说"fine-tunes all components except open-loop metric scoring heads"。保留这些head相当于保留一个open-loop sanity check的auxiliary signal,policy head不会完全drift到open-loop定义的"好"之外。

这四点合起来让8192 action space的PPO fine-tuning能work。对比RAD < 100候选——RAD可以更aggressive,因为action space小,exploration容易;GTR²S必须conservative,因为action space大,容易崩。

---

## 八、跟LLM Post-Training的对应

这是我觉得这篇paper最有意思的framing:

| LLM RLHF | GTR²S |
|---|---|
| SFT pretrained model | NAVSIM-pretrained GTRS-Dense |
| Prompt $x$ → response $y$ | State $s_t$ → action $a_i$ |
| Reward model $r(x,y)$ | Rule-based reward $r_t$ |
| PPO + KL to SFT | PPO + KL to pretrained |
| Vocab tokens (~32k) | Trajectory candidates (8192) |
| Helpfulness vs safety | Progress vs safety |
| Helpful but unsafe (toxic) | Progress but crash |
| Safe but useless (refuse all) | Safe but can't move (low RC) |

Recipe同构。这暗示一个general lesson:**SL pretrain + RL fine-tune with KL anchor**是个universal pipeline,AlphaGo (SL policy net → RL policy net, https://www.nature.com/articles/nature16961) 也好,LLM也好,driving也好,都work。

---

## 九、个人觉得没解决/没讲清楚的地方

1. **Hard/Extreme scenarios没动**:HD 2.6→2.3。需要curriculum learning或adversarial scenario mining。
2. **3DGS sim-to-real gap**:HUGSIM是photo-realistic,但dynamic object建模有限。real road test效果未知。
3. **No training curve**:report太短,看不到convergence动态,不知道几epoch开始稳定。
4. **In-domain数字缺**:没给nuScenes/Waymo in-domain closed-loop分数,只给out-of-domain KITTI-360。无法判断gap大小。
5. **只对比PPO**:没试SAC、GRPO (https://arxiv.org/abs/2401.04088)、DPO-style offline RL。GRPO去掉value function可能更简单,值得试。

---

## 十、整体Take-away (一句话版)

**把NAVSIM open-loop pretrain的trajectory scorer塞进HUGSIM closed-loop simulator,用PPO + KL + GAE + rule-based reward做fine-tune,不依赖人类demo,把8192 action space scale起来,在out-of-domain的KITTI-360上HD-Score从9.8翻到19.6。这个recipe同构于LLM RLHF和AlphaGo的SL→RL pipeline,验证了它是universal。**

---

## 主要References

- GTR²S baseline GTRS-Dense: https://arxiv.org/abs/2506.06664
- HUGSIM: https://arxiv.org/abs/2412.01718
- RAD: https://arxiv.org/abs/2502.13144
- CaRL: https://arxiv.org/abs/2504.17838
- NAVSIM: https://arxiv.org/abs/2406.13361
- Hydra-MDP: https://arxiv.org/abs/2406.06978
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- IMPALA: https://arxiv.org/abs/1802.01561
- 3DGS: https://arxiv.org/abs/2308.14737
- KITTI-360: http://www.cvlibs.net/datasets/kitti-360/
- nuScenes: https://www.nuscenes.org/
- Waymo Open: https://waymo.com/open/
- DAgger: https://arxiv.org/abs/1011.0686
- DeepSeek-R1 / GRPO: https://arxiv.org/abs/2401.04088
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- LLM post-training survey: https://arxiv.org/abs/2502.21321
- AlphaGo: https://www.nature.com/articles/nature16961
- UniAD: https://arxiv.org/abs/2212.10156
- VAD: https://arxiv.org/abs/2303.12077
- TransFuser: https://arxiv.org/abs/2205.15997
- VADv2: https://arxiv.org/abs/2402.13243
- Centaur: https://arxiv.org/abs/2503.11650

---

Andrej, 上面把这篇technical report的"白板讲解"给完了。核心intuition我总结成三条,可以直接记:

1. **Open-loop pretrain + closed-loop RL fine-tune是universal pipeline**(AlphaGo→RLHF→driving都验证)。
2. **大action space RL必须conservative**(KL anchor + 高质量pretrain init + value baseline + frozen auxiliary heads),激进exploration会崩。
3. **GAE在medium-horizon stochastic环境下几乎一定胜过return或pure TD**(ablation干净证明了)。

---

# GTR²S: Reinforcement Fine-tuning of End-to-end Trajectory Scorers 深度解析

Andrej, 这篇paper来自NVIDIA + Fudan团队（Zhenxin Li等），核心是**用PPO对预训练的trajectory scorer做reinforcement fine-tuning**，从open-loop NAVSIM训练的GTRS-Dense出发，在HUGSIM closed-loop benchmark上把HD-Score从9.8拉到19.6（+9.8%绝对提升），route completion从17.2到29.5（+12.3%）。技术report本身比较compact，但里面藏着很多值得深挖的design choice。

参考链接：
- Paper: https://arxiv.org/abs/2506.06664 (GTRS-Dense baseline)
- HUGSIM: https://arxiv.org/abs/2412.01718
- RAD: https://arxiv.org/abs/2502.13144
- CaRL: https://arxiv.org/abs/2504.17838
- NAVSIM: https://arxiv.org/abs/2406.13361
- Hydra-MDP: https://arxiv.org/abs/2406.06978
- PPO原paper: https://arxiv.org/abs/1707.06347
- GAE原paper: https://arxiv.org/abs/1506.02438
- IMPALA: https://arxiv.org/abs/1802.01561
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- KITTI-360: http://www.cvlibs.net/datasets/kitti-360/
- nuScenes: https://www.nuscenes.org/
- Waymo Open Dataset: https://waymo.com/open/

---

## 1. 为什么这篇paper值得读：核心Problem Framing

Autonomous driving end-to-end planning里有两大paradigm：

**(a) Continuous trajectory regression**：直接回归waypoints $(x_t, y_t)_{t=1}^{T_H}$，代表方法UniAD、VAD。问题：uni-modal输出，对multi-modal driving behaviors（左转or右转、超车or跟车）建模困难，且regression loss对"差不多对"的trajectory不够discriminative。

**(b) Trajectory scoring / candidate selection**：预先定义一个大的离散trajectory vocabulary $\mathcal{A} = \{a_i\}_{i=1}^n$，模型只学一个scoring function $s_\theta(a_i | s_t)$，最后 $\arg\max_i \pi_\theta(a_i | s_t)$ 选一条。代表方法Hydra-MDP、GTRS、DriveSuprem、TransFuser的scoring变体。

GTR²S走的是(b)路线，action space规模约 **8,192 trajectories**（dropout后从16,384降到8,192）。这比RAD的<100 candidates大了两个数量级，这是这篇paper真正撑住的scalability claim。

### Open-loop → Closed-loop的Gap

Open-loop training（NAVSIM那种）本质是：
$$\mathcal{L}_{\text{IL}} = -\sum_t \log \pi_\theta(a^*_t | s_t)$$
其中 $a^*_t$ 是专家演示。问题在distribution shift：模型在 $s_0$ 误差一点点，到 $s_1 = f(s_0, a_0)$ 就跑到专家没见过的state，后续 $s_2, s_3, ...$ 越偏越远 — **compounding error**。这是经典DAgger论文（https://arxiv.org/abs/1011.0686）讨论的问题。

GTR²S的解法：把pre-trained scorer塞进closed-loop simulator (HUGSIM基于3DGS渲染)，让它自己rollout，用rule-based reward做PPO fine-tune。**核心insight**：预训练的 $\pi_{\theta_{\text{old}}}$ 已经会开车了（NAVSIM训练），RL只负责"patch" closed-loop下的failure mode，不需要从scratch学。

---

## 2. Trajectory Scorer架构详解（Fig. 1）

```
[sensor input s_t: front + front-left + front-right crops]
        │
        ├──► Image Backbone (ViT/ResNet) ──► Image Tokens {x_j}_{j=1}^{N_img}
        │
        ├──► Trajectory Tokenizer ──► Trajectory Tokens {τ_i}_{i=1}^{n}
        │       (each a_i → polyline waypoints → PE + MLP embedding)
        │
        ▼
[Transformer Decoder]
   Cross-attention:  τ_i  ← attends to →  {x_j}
        │
        ▼
[Policy Head]  ──►  π(·|s_t) ∈ Δ^{n-1}    (softmax over 8192 candidates)
[Scoring Heads S_1..S_m]  ──►  {S_i(·|s_t)}_{i=1}^m   (open-loop metrics like EPDMS)
```

关键细节：

- **Input**: 3个crop拼成wide-front image（frontal + center-cropped front-left + center-cropped front-right）。这相当于扩大horizontal FOV，对intersection turning场景重要。
- **Trajectory vocabulary**: 16,384条候选（来自NAVSIM的anchor设计），训练用dropout mask到8,192。每条trajectory是 $(x_t, y_t, \theta_t)_{t=1}^{T_H}$ waypoints，horizon约4-5s。
- **Policy head**: 这是PPO要优化的head。
- **Scoring heads**: 这些是open-loop metrics head（EPDMS的子指标如no-collision, drivable-area-compliance, comfort, progress, TTC），**在RL fine-tuning时frozen**（见Fig.1 caption: "fine-tunes all the components except the open-loop metric scoring heads"）。

Intuition：scoring heads是open-loop的interpretable monitor，保留它们相当于保留一个"open-loop sanity check"的auxiliary signal，避免policy head在RL下完全drift。

---

## 3. PPO目标函数：变量逐项解剖

### 公式(1) — Clipped Surrogate Objective

$$\mathcal{L}(\theta) = \mathbb{E}\Big[\min\big(k_t(\theta) \cdot A_t,\ \mathrm{clip}(k_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\big)\Big]$$

逐项解释：
- $k_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ — **probability ratio / importance weight**。分子是当前policy在 $(s_t, a_t)$ 上的概率，分母是behavior policy（即data collection时的policy snapshot）的概率。两者都是同一个softmax over 8192 trajectories。
- $A_t$ — **advantage estimate**（见GAE公式2），表示 $a_t$ 比平均baseline好多少。
- $\epsilon$ — **trust region半宽**，PPO原paper取0.2。控制单次update的policy变化幅度。
- $\min(\cdot, \cdot)$ — **pessimistic bound**：当 $A_t > 0$（好动作），允许ratio最大到 $1+\epsilon$；当 $A_t < 0$（坏动作），允许ratio最小到 $1-\epsilon$。防止过度update。

Intuition：PPO本质是"trust region + clipping"版的policy gradient，比vanilla PG稳定，比TRPO计算便宜（不需要solve constrained optimization）。在LLM post-training（RLHF, GRPO）里也是de-facto标准（参考DeepSeek-R1, GRPO：https://arxiv.org/abs/2401.04088）。

### 公式(2) — Generalized Advantage Estimation (GAE)

$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \cdot \delta_{t+l}$$

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)$$

变量解释：
- $\delta_t$ — **temporal-difference (TD) error** at step $t$。$r_t$ 是即时reward，$\gamma V(s_{t+1})$ 是bootstrapped未来value estimate，$V(s_t)$ 是当前state value estimate。
- $\gamma \in [0, 1]$ — **discount factor**，控制future reward的衰减。driving场景里通常0.9-0.99。
- $\lambda \in [0, 1]$ — **GAE trade-off parameter**，控制bias-variance：
  - $\lambda = 0$：$A_t = \delta_t$，pure 1-step TD，low variance, high bias（依赖value function准确性）。
  - $\lambda = 1$：$A_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$，Monte Carlo return，high variance, low bias。
  - $\lambda = 0.95$ 是常见default。

GTR²S的ablation（Table 2）直接对比了三种advantage：
- $\sum_{t'=t}^{\infty} r_{t'}$ （return-based，相当于 $\lambda=1$）：overall HD-Score 14.3-16.7
- $\delta_t$ （1-step TD，相当于 $\lambda=0$）：overall HD-Score 17.6
- $A_t$ （GAE）：overall HD-Score 19.6 ✓ best

这个ablation很干净，说明driving场景下**value function bootstrap有用**（value head学到了reasonable的state value），完全Monte Carlo方差太大。

---

## 4. Reward Shaping：公式(3) 拆解

$$r_t = RC_t \cdot \Big(\prod_{p_t \in P} p_t\Big) - \prod_{m_t \in M}(1 - m_t)$$

变量解释：
- $RC_t$ — **Route Completion** at step $t$，定义为agent本step沿reference route前进的距离比例，范围 $[0, 1]$。这是"progress"信号，鼓励往前开。
- $P$ — **soft penalties set**，每个 $p_t \in [0, 1]$ 是某项soft metric的"compliance score"（越接近1越好）。包括：
  - Comfort（jerk, lateral acceleration）
  - Time-to-collision (TTC)
- $M$ — **hard penalties set**，每个 $m_t \in \{0, 1\}$ 是hard metric的binary violation flag（1=violation）。包括：
  - Drivable-area compliance（是否出road）
  - No-at-fault collisions（是否撞车且自己有责任）

设计intuition：
- 第一项 $RC_t \cdot \prod p_t$：**progress only when comfortable/safe**。如果comfort很差（$p_t \to 0$），整个第一项被打掉。
- 第二项 $\prod(1-m_t)$：**任一hard violation → 整个term归零**。这是"and"语义：任何一个hard violation都destroy reward。
- 减号连接：reward = good_progress - hard_violation_indicator。极端值：完美一步 $r_t = RC_t \cdot 1 - 1 \cdot 1 = RC_t - 1 \le 0$（即使完美也最多0），撞车 $r_t = RC_t \cdot 1 - 0 \cdot ... = RC_t$（？）。

等等，再读一遍：$\prod(1-m_t)$，如果 $m_t = 1$（violation），则 $(1-m_t) = 0$，所以 $\prod = 0$，则 $-0 = 0$。如果没violation，$m_t = 0$，$(1-m_t) = 1$，$\prod = 1$，则 $-1$。

所以正确解读：
- 无hard violation: $r_t = RC_t \cdot \prod p_t - 1$（progress被soft penalty打折，再减1）
- 有任一hard violation: $r_t = RC_t \cdot \prod p_t - 0$（不减1，看似reward更高？？）

这反直觉，但符合CaRL的intuition：**violation时让reward信号"幸存"下来，不让agent从violation state学到"以后就放弃"**。GTR²S相对CaRL的改动是：CaRL在hard violation时**终止episode**，而GTR²S**继续rollout**，所以需要reward function本身在violation后还有信号引导agent恢复。这个设计让agent能从"撞了之后"继续学恢复，避免sparse reward + early termination的样本低效。

参考CaRL paper: https://arxiv.org/abs/2504.17838

---

## 5. Data Collection: Parallelized Actor–Learner

这是工程细节，但很关键。3DGS渲染比CARLA那种geometric rendering慢，每帧需要gaussian splatting（参考https://arxiv.org/abs/2308.14737）。所以单GPU串行rollout throughput极低。

GTR²S借鉴IMPALA（https://arxiv.org/abs/1802.01561）的actor-learner架构：
- 多个GPU instance并行跑environment rollout（每个instance一个scenario）
- Actor用policy snapshot $\pi_{\theta_{\text{old}}}$ 跑完一个scenario或固定horizon
- 收集 $\{(s_t, a_t, r_t, s_{t+1}, \pi_{\theta_{\text{old}}}(a_t|s_t), V_{\text{old}}(s_t))\}$ 发给learner
- Learner做PPO update，sync新policy回actors

Trajectory执行细节：模型输出 $a_i$（waypoints），通过**kinematic bicycle model + LQR controller**（HUGSIM提供）转成steering rate + acceleration，再execute in simulator。这层controller相当于把"高维trajectory planning"映射到"低维control"，是decoupled design。

Training setup: 8×A100, batch size 192, 3 epochs, lr 2e-5, weight decay 0.0（注意是0，**不regularize weights**，全靠KL regularization约束drift）。

---

## 6. KL Regularization — 为什么必须

PPO的clipped surrogate objective只限制**ratio** $k_t(\theta)$ 在 $[1-\epsilon, 1+\epsilon]$ 内，但这是per-sample的local constraint，不限制**整体policy分布** $\pi_\theta(\cdot|s)$ 相对 $\pi_{\theta_{\text{pretrained}}}(\cdot|s)$ 的偏离。

LLM RLHF实践（参考https://arxiv.org/abs/2502.21321）发现：纯PPO容易catastrophic forgetting预训练能力。在driving里这表现为：RL fine-tune后model可能在closed-loop下更好，但open-loop metric（NAVSIM EPDMS）会崩，丧失generalization。

GTR²S加KL term（具体form未在report里写出，常见是 $\beta \cdot \mathbb{E}_s[\mathrm{KL}(\pi_{\theta_{\text{pretrained}}}(\cdot|s) \| \pi_\theta(\cdot|s))]$）。

Ablation（Table 2）：
- 无KL: overall HD-Score 12.3（nuScenes-only），14.3-16.7（N+W）
- 加KL: 提升到17.6-19.6

证明KL对"preserve planning ability"关键。

---

## 7. Action Space Scale的Challenge

8192 trajectories的action space比LLM vocab（32k tokens）小一个量级，但远超典型RL setup。难点：

**(a) Credit assignment**：从8192个candidate里选一个，reward signal sparse。GAE通过value function $V(s_t)$ 帮忙做credit assignment（estimating $V(s_t)$ 给所有candidate的"expected future return"提供baseline）。

**(b) Exploration**：预训练 $\pi_{\theta_{\text{pretrained}}}$ 在softmax输出里已经concentrate在少数几个high-score candidate上，RL exploration主要靠PPO的stochastic sampling + entropy bonus（report未明确提entropy，但PPO default包含）。

**(c) Softmax over 8192 logits的数值稳定性**：需要large logit range + stable softmax，log-sum-exp trick必备。

对比RAD（<100 candidates）：RAD的action space小，可以用更aggressive RL（更多exploration, less reliance on pretraining），但表达能力弱。GTR²S用大action space + conservative fine-tuning（PPO + KL + GAE）trade-off。

---

## 8. 实验结果深度解读

### Table 1: HUGSIM KITTI-360（out-of-domain eval）

| Method | Easy RC | Easy HD | Med RC | Med HD | Hard RC | Hard HD | Ext RC | Ext HD | Overall RC | Overall HD |
|---|---|---|---|---|---|---|---|---|---|---|
| UniAD | 16.6 | 4.7 | 11.7 | 1.9 | 12.2 | 1.7 | 9.4 | 0.6 | 12.4 | 2.2 |
| VAD | 13.8 | 2.9 | 12.5 | 1.4 | 12.2 | 1.4 | 11.0 | 0.8 | 12.4 | 1.6 |
| LTF* | 24.5 | 8.0 | 13.8 | 2.8 | 12.1 | 1.7 | 11.1 | 0.3 | 15.2 | 3.1 |
| LTF | 21.9 | 5.2 | 15.6 | 3.0 | 14.0 | 7.0 | 11.3 | 1.6 | 15.3 | 2.0 |
| GTRS-Dense | 29.0 | 19.4 | 24.4 | 16.9 | 9.1 | 2.6 | 8.0 | 1.3 | 17.2 | 9.8 |
| **GTR²S** | **55.0** | **41.7** | **47.5** | **34.9** | 9.7 | 2.3 | 8.8 | 2.4 | **29.5** | **19.6** |

关键观察：
1. **Easy/Medium的巨幅提升**：HD-Score从19.4→41.7（easy），16.9→34.9（medium）。说明RL fine-tuning在"简单可控场景"下收益最大，agent学会更激进的progress（RC翻倍）且不违反safety。
2. **Hard/Extreme几乎没动**：Hard HD 2.6→2.3（反而略降！），Extreme 1.3→2.4。Paper承认"some cases are unsolvable"（如oncoming vehicle on narrow road）。这是closed-loop benchmark的天花板：如果scenario本身需要agent做出训练分布外的行为，PPO也救不了。
3. **Pre-trained baseline GTRS-Dense已经很弱**：open-loop NAVSIM训练的GTRS-Dense在HUGSIM上overall HD只有9.8，说明open-loop→closed-loop gap非常大。GTR²S把gap缩小了一半左右。

LTF两行的差异（带*的是HUGSIM paper官方报告，不带*是re-evaluation）说明re-evaluation的protocol细节影响巨大，需要小心解读绝对数字。

### Table 2: Ablation

5行配置：
| Data | KL | Adv. | Overall RC | Overall HD |
|---|---|---|---|---|
| N | ✗ | return | 20.5 | 12.3 |
| N | ✓ | return | 26.5 | 14.3 |
| N+W | ✓ | return | 26.9 | 16.7 |
| N+W | ✓ | δ_t | 26.5 | 17.6 |
| N+W | ✓ | A_t (GAE) | **29.5** | **19.6** |

Insight：
1. **Data diversity（N→N+W）**：HD从14.3→16.7（+2.4）。Waymo数据多了diverse场景，特别是highway和dense urban。
2. **KL regularization**：12.3→14.3（+2.0）单独的贡献。
3. **Advantage choice**：return→δ_t→GAE，分别是16.7→17.6→19.6。GAE的 $\lambda$-interpolation明显胜出，验证了bias-variance trade-off在driving reward（高variance, multi-step horizon）下的重要性。

---

## 9. 与LLM Post-Training的类比

这篇paper在framing上明确说"suitable for post-training Large Language Models"（引用https://arxiv.org/abs/2502.21321）。我觉得这个类比很有启发：

| LLM RLHF/GRPO | GTR²S |
|---|---|
| Pretrained LLM (SFT) | Pretrained trajectory scorer (NAVSIM IL + EPDMS) |
| Prompt → response | State s_t → action a_i |
| Reward model | Rule-based reward (RC, comfort, collision) |
| PPO/GRPO + KL to SFT | PPO + KL to pretrained |
| Helpfulness vs safety trade-off | Progress vs safety trade-off |
| Action space: vocab tokens | Action space: trajectory candidates |

GTR²S的"scoring heads frozen during RL"对应LLM RLHF里"keep auxiliary LM head frozen, only tune policy head"的trick（罕见但有用）。

---

## 10. Limitations & Open Questions

1. **Hard/Extreme scenarios没解决**：HD-Score 2.3在hard场景说明agent基本走不动。可能需要：
   - Curriculum learning（先easy后hard）
   - Hard scenario mining + oversampling
   - Hierarchical planning（high-level decision + low-level trajectory）

2. **3DGS sim-to-real gap未验证**：HUGSIM用3DGS渲染，但real deployment（NVIDIA自己的DRIVE Sim或真实路测）效果未知。3DGS对dynamic object（行人、车辆）建模有限。

3. **Reward依赖hand-crafted rules**：$RC_t, P, M$ 都是engineered。未来可能用learned reward model（preference-based RL, 例如https://arxiv.org/abs/2204.05862 LEAP）。

4. **No comparison to other RL algorithms**：只跑PPO。SAC、GRPO、DPO-style offline RL可能更适合。GRPO（DeepSeek提出，https://arxiv.org/abs/2401.04088）去掉了value function，可能更简单。

5. **In-domain vs out-of-domain gap**：Paper只report了KITTI-360（out-of-domain）的数字，没给nuScenes/Waymo in-domain的closed-loop数字。in-domain应该更高，让reader看到gap大小。

---

## 11. 关键References汇总

- **GTR²S系列前作**：
  - Hydra-MDP: https://arxiv.org/abs/2406.06978 (CVPR 2024 Challenge winner)
  - Hydra-MDP++: https://arxiv.org/abs/2503.12820
  - GTRS-Dense: https://arxiv.org/abs/2506.06664 (NeurIPS 2024)
  - DriveSuprem: https://arxiv.org/abs/2506.06659

- **Closed-loop simulators**：
  - HUGSIM: https://arxiv.org/abs/2412.01718
  - NAVSIM: https://arxiv.org/abs/2406.13361
  - CARLA: https://carla.org/

- **RL fine-tuning in driving**：
  - RAD: https://arxiv.org/abs/2502.13144
  - CaRL: https://arxiv.org/abs/2504.17838
  - CarDreamer: https://arxiv.org/abs/2410.22760

- **RL fundamentals**：
  - PPO: https://arxiv.org/abs/1707.06347
  - GAE: https://arxiv.org/abs/1506.02438
  - IMPALA: https://arxiv.org/abs/1802.01561
  - TRPO: https://arxiv.org/abs/1502.05477

- **LLM post-training**：
  - InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
  - DeepSeek-R1 / GRPO: https://arxiv.org/abs/2401.04088
  - LLM post-training survey: https://arxiv.org/abs/2502.21321

- **Datasets**：
  - nuScenes: https://www.nuscenes.org/
  - Waymo Open: https://waymo.com/open/
  - KITTI-360: http://www.cvlibs.net/datasets/kitti-360/
  - PandaSet: https://scale.com/open-datasets/pandaset

- **End-to-end driving baselines**：
  - UniAD: https://arxiv.org/abs/2212.10156
  - VAD: https://arxiv.org/abs/2303.12077
  - TransFuser: https://arxiv.org/abs/2205.15997
  - VADv2: https://arxiv.org/abs/2402.13243
  - Centaur: https://arxiv.org/abs/2503.11650

---

## 12. 我的整体Take-away

这篇technical report写得很compact但工程意义大。核心贡献是把"trajectory scorer + RL fine-tuning"这条路线**scale到8192 action space且不依赖human demonstrations**，从工程上验证了"NAVSIM open-loop pretrain → HUGSIM closed-loop PPO fine-tune"这套pipeline可行，且三个trick（data diversity, KL reg, GAE）每个都给了干净的ablation。

对你（Karpathy）的intuition building而言，我建议重点关注：
1. **Open-loop pretrain + closed-loop RL fine-tune是general recipe**，从AlphaGo的SL→RL，到LLM的SFT→RLHF，到driving的NAVSIM→HUGSIM PPO，paradigm趋同。
2. **Large discrete action space + PPO是scalable的**，前提是有好的pretrain init + KL anchor。
3. **Reward shaping的"violation后继续rollout"是个值得记住的设计**，避免sparse reward + early termination的样本低效。
4. **GAE > TD > return**在driving这种medium-horizon, high-variance reward的任务上稳定胜出，可以generalize到其他robotics tasks。

唯一的遗憾是technical report太短，没有architecture figure的全部细节、没有training curve、没有failure case analysis。期待full version。
