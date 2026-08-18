---
source_pdf: Being-M0.7 A Latent World-Action Model.pdf
paper_sha256: 5f55c05abd058d59aceaf021f55221fc72d2d8818f2501ba8cda0dfd354212fb
processed_at: '2026-08-18T02:29:55-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Being-M0.7 人话版

好,我用最直觉的方式跟你聊聊这篇 paper 到底在干嘛。

## 一句话概括

**让 robot 先在脑子里"想象"未来会怎样(看到啥、身体怎么动),再根据这个想象去执行动作。** 跟人类做事一样——你抓杯子之前,脑子里其实已经预演了"手伸过去、杯子在这个位置、我会这样握住"。Being-M0.7 就是把这个"预演"过程显式建模出来,然后让一个轻量级 action head 去执行。

## 为什么 humanoid 这么难搞?

三个痛点,作者全怼了一遍:

**痛点 1:数据贵得离谱。** teleop 一台 29-DoF humanoid,你要同步 egocentric video + proprioception + whole-body command,操作员还得戴 VR 头套加 ankle tracker。千小时级就到顶了。对比 fixed-base arm 的 Open-X-Embodiment 数据轻松上万小时,humanoid 数据根本 scale 不起来。

**痛点 2:pixel-level world prediction 是坑。** 像 Cosmos、Genie、MotionWAM 这些 world model 路线,直接生成未来 video frames。问题是 humanoid egocentric 视角下,robot 自己在快速移动,camera 在抖,生成的 pixel 很多时候都是模糊的垃圾。更关键的是——**robot 控制根本不需要 pixel fidelity**。你抓个杯子,需要知道的是"杯子在桌上、我的手要伸过去",而不是"杯子的釉面反光是啥纹理"。pixel generation 把大量 capacity 浪费在 control-irrelevant 的 appearance 上。

**痛点 3:上下半身割裂。** 大部分 pipeline 把 locomotion 和 manipulation 分开学,导致 robot 走路的时候手不知道干嘛,抓东西的时候脚站不稳。whole-body coordination 始终差点意思。

## Being-M0.7 的核心套路

面对这三个痛点,作者的应对是:

### 应对痛点 1:用 human 数据预训练

human 的 egocentric video(Ego4D、Nymeria 那些)有万小时级,motion capture 数据(AMASS、HumanML3D)也海量。关键是——**human 怎么动、看到啥、怎么交互,本质上跟 humanoid 要做的事是同构的**。human 走过去抓东西,humanoid 也是走过去抓东西;human 弯腰开柜门,humanoid 也是弯腰开柜门。差的就是 morphology(腿长不一样、关节 limit 不同)。

作者的 solution 是搞了一个 **unified motion representation**:用 head 作为 root(因为 egocentric video 是头戴相机,head 的 motion 直接决定视觉流),只保留 head + 两个 hand + 两个 foot 共 5 个 keypoint,22D。这个表示既能从 human motion capture 构造,也能从 humanoid forward kinematics 构造,**直接把 morphology gap 给桥掉了**。

然后数据分三路:
- $\mathcal{D}_{VM}$: paired video-motion(最贵,supervise joint 分布)
- $\mathcal{D}_V$: video-only(便宜,supervise vision marginal)
- $\mathcal{D}_M$: motion-only(也便宜,supervise motion marginal)

合起来超过 10,000 小时,比单纯 paired 数据扩了一个数量级。

### 应对痛点 2:latent space 做 world modeling,别碰 pixel

所有 video frame 通过 **frozen DINO encoder** 压成 latent token。模型在 latent space 里预测未来,不碰 pixel。

$$Z = \mathcal{E}(V)$$

这里 $\mathcal{E}$ 是 DINO encoder,$V$ 是 raw video,$Z$ 是 latent。paper 标题里 "latent" 就是从这来的。

直觉上这相当于:**把"未来世界长什么样"压缩成"未来世界的语义状态向量"**。DINO 本来就是在 internet image 上 self-supervised 学出来的,latent space 自然就 encode 了 "object position"、"scene layout"、"interaction affordance" 这些 semantic 量,而不太 care texture。

### 应对痛点 3:prior 阶段就 jointly 学 vision + motion

作者用 **Mixture-of-Transformers (MoT)**([Liang et al., 2024](https://arxiv.org/abs/2411.04996))来同时处理 vision 和 motion 两个模态。每个 transformer block 内部:

- Vision tokens 走自己的 Q/K/V projection + FFN
- Motion tokens 走自己的 Q/K/V projection + FFN
- 但 **attention 是共享的**——所有 vision 和 motion token 拼在一起做一次 attention

这样 vision 和 motion 可以 cross-talk,但参数空间分开。missing modality 的时候,只对 present modality 算 loss,三种数据(paired / video-only / motion-only)都能喂进去。

**关键**:motion branch 学的就是 whole-body coordination——head 怎么动、手怎么配合、脚怎么踩。这个 coordination 在 prior 阶段就从海量 human 数据学到了,post-training 不用从零开始。

## 两层架构:Prior + Action Expert

paper 的公式 1 把整个问题分解成:

$$P(Z_\mathcal{Q}, M_\mathcal{Q}, \mathbf{a}_q | Z_\mathcal{C}, M_\mathcal{C}, I, O_q) \approx P(\mathbf{a}_q | Z_\mathcal{Q}, M_\mathcal{Q}, O_q, I) \cdot P(Z_\mathcal{Q}, M_\mathcal{Q} | Z_\mathcal{C}, M_\mathcal{C}, I)$$

人话翻译:
- **左边那个 $P(Z_\mathcal{Q}, M_\mathcal{Q} | ...)$ 是 prior world model**:给定历史 $Z_\mathcal{C}$（看到的 video history）、$M_\mathcal{C}$（motion history）、$I$（instruction），预测未来 $\mathcal{Q}$ 的 latent vision $Z_\mathcal{Q}$ 和 motion $M_\mathcal{Q}$
- **右边那个 $P(\mathbf{a}_q | Z_\mathcal{Q}, M_\mathcal{Q}, O_q, I)$ 是 action expert**:给定 prior 预测的未来 + 当前 robot observation $O_q$,生成 executable action $\mathbf{a}_q$

变量再展开说:
- $\mathcal{C} = \{1, ..., K\}$:context interval,前 K 帧(K=5)
- $\mathcal{Q} = \{K+1, ..., T\}$:future interval,要预测的 future(T=25)
- $Z_t$: 第 $t$ 帧的 visual latent
- $M_t$: 第 $t$ 帧的 motion vector（22D）
- $O_q$: query 时刻 $q$ 的 robot observation,包含 image + proprioception + normalized execution progress
- $\mathbf{a}_q$: 要生成的 action chunk

这个 decompose 的妙处是 **frequency separation**:
- Prior 跑在 5Hz,慢但可以深（30 layer,768 hidden）
- Action expert 跑在 50Hz,快必须浅（6 layer,384 hidden）

一次 prior rollout 的中间隐状态 cache 起来,被 action expert 反复 query **9 次**（3.5s / 0.4s ≈ 9）,等于把 expensive prior 的算力摊销到 9 个 control windows。这就是为什么 heavy prior 负担得起——它不是每帧都跑。

## Action Expert 怎么连到 Prior

这个连接是 paper 里我最喜欢的设计。公式:

$$A_q^{j+1} = \mathcal{B}_j(A_q^j | [H_{\ell_j}^\theta(\tau), A_q^j, O_q], \tau)$$

逐项解释:
- $A_q^j$: action expert 第 $j$ 层的 hidden state
- $\mathcal{B}_j$: 第 $j$ 个 expert block
- $H_{\ell_j}^\theta(\tau)$: prior model 第 $\ell_j$ 层在 flow timestep $\tau$ 的 hidden state
- $\ell_j$: expert 第 $j$ 层连到 prior 的哪一层。配置上是 [1, 7, 13, 18, 24, 30]——prior 30 层均匀取 6 层
- $O_q$: 当前 robot observation

直觉上,这个设计让 expert 同时看到 prior 的:
- **浅层（layer 1）**: fine-grained visual feature
- **中层（layer 7, 13）**: mid-level motion + scene dynamics
- **深层（layer 18, 24, 30）**: high-level future planning

而 $O_q$ 提供实时的 closed-loop correction。**信息单向流:prior → expert,action 信息不回传到 prior**。这避免了 expert 训练时干扰 prior 的 generative representation。

## 训练:Flow Matching + Geometry Loss

### Pre-training

用 flow matching([Lipman et al., 2023](https://arxiv.org/abs/2210.02747)),linear path:

$$x_\tau = (1-\tau)x_0 + \tau \epsilon, \quad u^\star = \epsilon - x_0$$

变量:
- $x_0$: clean target（visual latent 或 motion token）
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise
- $\tau \sim \mathcal{U}(0,1)$: flow timestep
- $u^\star$: target velocity field
- $u_\theta$: model 预测的 velocity

Loss 是 MSE:
$$\mathcal{L}_V = \frac{1}{|\Omega_V|} \sum_{t \in \Omega_V} \| u_\theta^V(z_{\tau,t}, \tau, I, Z_C) - u_t^{V\star} \|_2^2$$
$$\mathcal{L}_M = \frac{1}{|\Omega_M|} \sum_{t \in \Omega_M} \| u_\theta^M(m_{\tau,t}, \tau, I, M_C) - u_t^{M\star} \|_2^2$$

$\Omega_V, \Omega_M$ 是 valid token index 集合（根据样本里有哪些模态决定）。

motion branch 还额外加 geometry loss:
$$\mathcal{L}_{\text{geom}} = \lambda_{\text{traj}} \mathcal{L}_{\text{traj}} + \lambda_{\text{local-vel}} \mathcal{L}_{\text{local-vel}}$$

- $\mathcal{L}_{\text{traj}}$: decoded global head/root position（欧式空间轨迹）要正确
- $\mathcal{L}_{\text{local-vel}}$: frame-to-frame end-effector velocity 要正确

这相当于**额外监督解码后的几何物理量**,防止 flow matching 优化到 motion 表示的"流形"外面去。直觉上就是:你不能只让 latent space loss 小,还得让 decode 出来的轨迹真的物理合理。

### Post-training

robot trajectory 同时提供 vision + motion + action 三种 target。action expert 也用 flow matching:

$$\mathcal{L}_A(\theta, \phi) = \mathbb{E} \| u_\phi^a(\tilde{\mathbf{a}}_\tau, \tau, H^\theta(\tau), o_q) - (\epsilon_a - \mathbf{a}_q) \|_2^2$$

变量:
- $\mathbf{a}_q = a_{q:q+h-1}$: ground-truth action chunk（h=10 但执行时 chunk 是 20 因为 50Hz）
- $\epsilon_a \sim \mathcal{N}(0, I)$: action noise
- $\tilde{\mathbf{a}}_\tau = (1-\tau)\mathbf{a}_q + \tau \epsilon_a$: noisy action
- $H^\theta(\tau)$: prior 中间层 hidden state
- $o_q$: 当前 observation

最终 loss:
$$\min_{\theta, \phi} \mathcal{L}_V(\theta) + \mathcal{L}_M(\theta) + \lambda_a \mathcal{L}_A(\theta, \phi)$$

注意 $\mathcal{L}_V$ 和 $\mathcal{L}_M$ 还在更新 prior,等于 robot 数据反过来 fine-tune prior 的 visual-motion dynamics,让 prior 更适应 robot egocentric 分布。

## Inference 的频率分层

| 组件 | 频率 | 刷新间隔 |
|---|---|---|
| Prior world-action model | 5 Hz | 每 3.5s |
| Action expert | 50 Hz | 每 0.4s |
| Low-level WBC (SONIC) | 50 Hz | 持续 |

流程:
1. Prior 用 N=10 flow matching steps 去噪,得到 future $(Z_\mathcal{Q}, M_\mathcal{Q})$ 和中间层 cache $H_n^\theta$
2. $H_n^\theta$ 转 KV cache
3. Prior 每 3.5s 刷一次
4. Action expert 每 0.4s 调一次,用 cached prior context + 最新 $O_{\text{cur}}$
5. 单次 expert query < 0.01s,符合 50Hz 实时性

action 更新 rule:
$$\hat{u}^n = u_\phi^a(\mathbf{a}^{n-1}, \tau_n, H_n^\theta, O_{\text{cur}})$$
$$\mathbf{a}^n = \mathbf{a}^{n-1} - \hat{u}^n \Delta \tau_n$$

## 实验:7/15 vs 3/15 vs 2/15

在 Unitree G1 上跑了 4 个 task:Mirror Toy Grasping、Water-Tank Fish Scooping、Tabletop Organization、Obstacle-Avoidance Basket Carrying。

定量比较（vs GR00T-N1.6 和 Ψ₀）:

| Method | Mirror Near (0.5m) | Mirror Far (1m) | Fish | Overall |
|---|---|---|---|---|
| GR00T-N1.6 | 1/5 | 0/5 | 1/5 | 2/15 |
| Ψ₀ | 0/5 | 1/5 | 2/5 | 3/15 |
| **Being-M0.7** | 3/5 | 1/5 | 3/5 | **7/15** |

7/15 这个绝对值仍然不高,但相对优势明显。我的解读:
- **Mirror task 优势大**:因为 mirror reasoning 需要推理"反射的物体在哪",本质是 partial observability + future prediction。GR00T-N1.6 和 Ψ₀ 都没显式 world model,只能 react to current observation,在 mirror 这种需要"infer 不可见物体"的场景天然吃亏
- **Fish task 也强**:因为水会扭曲视觉,需要 model 预测物体真实位置。latent world model 学到的是 semantic state,不容易被水波 texture 干扰
- **Mirror Far 只有 1/5**:说明远距离 + reflection 仍然很难。可能 DINO latent 在远处小物体上 encoding 不够 sharp

## 我的几个直觉判断

### 1. Latent > Pixel 这个判断是对的

从 Cosmos、Genie 这些 pixel world model 的实际效果看,physical video generation 在 ego-motion 下质量都不好。DINO latent 既保留了 semantic 语义,又避免了 texture obsession。这是这条路线的核心 bet。

但问题是——**DINO latent 是 frozen 的**。它在 internet image 上训练,robot egocentric 分布可能 OOD。paper 里 prior 在 post-training 阶段继续更新,但 DINO encoder 本身没动。如果 robot 视角的 visual distribution 跟 internet image 差太多,frozen DINO 可能 bottleneck。未来可以考虑 unfreeze DINO 或用 robot-specific visual encoder。

### 2. 22D motion 表示可能信息量不够

只保留 head + 双手 + 双脚 5 个点,丢了:
- 躯干姿态（脊柱弯曲、侧倾）
- 关节 angle limit
- 接触力信息

对于需要精细 posture 的任务（弯腰钻桌底、贴墙走）,这个表示可能不够。未来扩展方向:
- 增加 torso / pelvis / knee keypoint
- 或者用 latent full-body representation + decoder
- 或者分层:5 个 keypoint 做 coarse plan,full body 用另一个 branch 学

### 3. Prior 3.5s refresh 是个隐患

如果 robot 突然遇到 obstacle,需要快速 re-plan。3.5s 内只能靠 action expert 用旧 prior context 做 closed-loop correction。motion-level feedback path 部分缓解,但 prior context 是 stale 的。

可能的改进:
- 缩短 refresh 到 1-2s
- 或者让 action expert 有"trigger replan"的能力,当 observation 偏离 predicted 太多时主动 invalidate cache
- 或者引入 reactive layer（类似 MPC 里的 emergency stop）

### 4. 7/15 说明 humanoid loco-manipulation 真的难

绝对值低不是 paper 的问题,是 humanoid 这个 problem 本身的问题。29-DoF whole-body + dexterous hand + egocentric partial observability + real-time control——叠加起来很难。但相对 baseline 的 3.5x 提升（7 vs 2）说明这个方向 promising。

### 5. 没给 scaling curve 和 ablation

paper 全是 real-world 实验,缺:
- 去掉 motion branch 会怎样?
- prior 规模 scaling 怎么影响?
- 用不同 visual encoder（CLIP vs DINO vs 自训练）会怎样?
- paired vs video-only vs motion-only 各自贡献多少?
- prior refresh rate 怎么影响 latency vs accuracy?

这些都是 follow-up paper 该回答的。当前 paper 更像 architecture proposal + proof of concept。

## 跟其他 humanoid foundation model 的关系

| Model | 路线 | 关键区别 |
|---|---|---|
| **GR00T N1** ([link](https://arxiv.org/abs/2503.14734)) | VLA + diffusion action | 无显式 world model |
| **Ψ₀** ([link](https://arxiv.org/abs/2603.12263)) | Video pretrain → flow action expert | Video 只做 representation,不 generative |
| **WholeBodyVLA** ([link](https://arxiv.org/abs/2602.10106)) | Latent action from video | 学 latent action,不预测 future state |
| **MotionWAM** ([link](https://arxiv.org/abs/2606.09215)) | Pixel world model → motion condition | Pixel-level,贵且 ego-motion 下 noisy |
| **Being-M0.7** | Latent world model + motion + action expert | Latent + 联合 video-motion + 解耦执行 |

Being-M0.7 的独特定位:**它是第一个把 latent world model + unified motion + future-conditioned action expert 三件事合在一起做的 humanoid 模型**。每件事单独看都不新——latent world model 有 Cosmos、motion generation 有 MDM、future-conditioned policy 有 Decision Transformer——但合起来针对 humanoid 这个 specific problem,是新的。

## 这条路线如果走通会怎样?

往远了想,如果 latent world-action model 这条路线真的 scale 起来,会有几个有意思的衍生方向:

1. **Latent space interpretability**:能不能 decode $Z_\mathcal{Q}$ 回 image 看预测?能不能可视化 robot 在"想象"什么?这对 trust 和 debug 重要。
2. **Multi-modal future sampling**:future 本身有多模态性（可以走左或右）。能不能 sample 多个 future,做 trajectory ensemble 或 MPC-style optimization?
3. **Active perception**:让 prior 主动预测"如果我转头看那边会看到什么",做 active sensing。这跟 next-best-view 经典问题挂钩,但在 generative framework 下可以做。
4. **Hierarchical world model**:5Hz 还是太粗。三层:1Hz scene graph + 5Hz visual-motion latent + 50Hz action?
5. **Cross-embodiment scaling**:unified motion representation 已经铺路,迁移到 H1、Apollo、甚至 quadruped 是开放问题。humanoid 之间的 morphology gap 其实不大,关键是 hand design 差异。
6. **Long-horizon planning**:目前 prior 只预测 ~4s future（T=25 frames @ 5Hz）。如果加 hierarchical structure,prior 自己能 rollout 多步,做 long-horizon task planning。这跟 Sora-style long video generation 的 challenge 类似。
7. **World model + RL**:latent world model 天然适合做 model-based RL——在想象里 rollout,evaluate action,选最优。这跟 Dreamer([Hafner et al., 2019](https://arxiv.org/abs/1912.01603))思路一致,但 latent space 要比 pixel efficient 得多。

## 相关参考

- **Being-M0.7 本身**:paper 没给 arXiv link,但 Being 系列前作在 https://arxiv.org/abs/2605.00078 (Being-H0.7)
- **Mixture-of-Transformers**: https://arxiv.org/abs/2411.04996
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **DiT (AdaLN)**: https://arxiv.org/abs/2212.09748
- **DINOv3**: https://arxiv.org/abs/2508.10104
- **π0**: https://arxiv.org/abs/2410.24164
- **GR00T N1**: https://arxiv.org/abs/2503.14734
- **Ψ₀**: https://arxiv.org/abs/2603.12263
- **WholeBodyVLA / EgoHumanoid**: https://arxiv.org/abs/2602.10106
- **MotionWAM**: https://arxiv.org/abs/2606.09215
- **SONIC**: https://arxiv.org/abs/2511.07820
- **Ego4D**: https://arxiv.org/abs/2110.07058
- **Nymeria**: https://arxiv.org/abs/2406.06043
- **AMASS**: https://amass.is.tue.mpg.de/
- **HumanML3D**: https://github.com/EricGuo5513/HumanML3D
- **ACT (action chunking)**: https://tonyzhaozh.github.io/aloha/
- **Ego-Exo4D**: https://arxiv.org/abs/2311.18258
- **Du et al. Video Policies**: https://arxiv.org/abs/2302.01111
- **Cosmos World Model**: https://arxiv.org/abs/2501.03575
- **Genie (interactive env)**: https://arxiv.org/abs/2402.15391
- **Dreamer (model-based RL)**: https://arxiv.org/abs/1912.01603
- **Decision Transformer (future-conditioned)**: https://arxiv.org/abs/2106.01345
- **Future-Conditioned Unsupervised Pretraining**: https://arxiv.org/abs/2303.01010

---

总结一句:**Being-M0.7 在 humanoid 这个具体 problem 上,把"先用海量 human 数据学一个 latent world prior,再用少量 robot 数据 fine-tune 一个轻量 action head 去执行"这个 recipe 做出来了**。它不是 end-to-end 一把梭,而是认真思考了 humanoid 的 time-scale separation、data scarcity、morphology gap 三个瓶颈,给了有原则的分解。

7/15 这个数字说明 promising 但远未成熟。但作为 proof of concept,latent world-action model 这条路线值得 follow。

---

# Being-M0.7: Latent World-Action Model for Humanoid Loco-Manipulation 深度解析

这篇 paper 在我看来是一个相当 elegant 的设计,它把 humanoid robot 控制重新表述为一个 **latent world model + future-conditioned action expert** 的两层架构。直觉上,作者试图把"在想象中规划未来"和"高频执行动作"这两件事解耦,跟 π0、GR00T N1 等纯 VLA 路线相比,多了 world modeling 的语义层;跟 Cosmos、MotionWAM 等纯 video generation 路线相比,又避免了 pixel-level rollout 的昂贵代价。

## 1. 核心动机:Intuition behind the design

humanoid loco-manipulation 的难点可以拆成三个相互纠缠的问题:

1. **Demonstration 稀缺**: teleoperating 一台 29-DoF+ 的 humanoid 极其困难,数据采集成本远超 fixed-base arm。
2. **Pixel-level future prediction 太贵且包含太多无关信息**: egocentric 视角下,机器人自身的快速运动 + camera jitter 会让 video generation 质量很差,而且大部分 capacity 浪费在重建不重要的 texture 上。
3. **Upper body / lower body 割裂**: 大多数 pipeline 把 manipulation 和 locomotion 分开学,导致 whole-body coordination 弱。

Being-M0.7 的核心 idea 是:**先把"未来世界的语义状态 + 身体 motion plan"作为一个联合的 latent representation 学出来(用海量 human 数据),再把这个 prior 当作 high-level context,用一个轻量级 action expert 把它"翻译"成可执行的 robot command**。这本质上是把 LLM 里"先思考再回答"的 chain-of-thought 思想搬到 robotics 上,但思考的内容是 latent visual state + body motion,而不是 token。

## 2. Problem Formulation:两层分解的数学形式

paper 把问题分解成两个 conditional distribution(公式 1):

$$P(Z_{\mathcal{Q}}, M_{\mathcal{Q}}, \mathbf{a}_q \mid Z_{\mathcal{C}}, M_{\mathcal{C}}, I, O_q) \approx P(\mathbf{a}_q \mid Z_{\mathcal{Q}}, M_{\mathcal{Q}}, O_q, I) \cdot P(Z_{\mathcal{Q}}, M_{\mathcal{Q}} \mid Z_{\mathcal{C}}, M_{\mathcal{C}}, I)$$

变量含义:
- $V$: egocentric video;$Z = \mathcal{E}(V)$ 表示经过 DINO encoder 后的 latent visual representation
- $M$: compact motion sequence(head-root + 双手双脚的 22D 表示)
- $I$: task instruction(text)
- $\mathcal{C} = \{1,...,K\}$: context interval,前 K 帧作为观察历史
- $\mathcal{Q} = \{K+1,...,T\}$: future interval,需要预测的 future
- $O_q$: 时刻 $q$ 的 robot observation(image + proprioception + normalized execution progress)
- $\mathbf{a}_q$: 要生成的 robot action

这个分解的妙处在于:**它把低频的 world modeling(5Hz)和高频的 action generation(50Hz)分离开**。一次 prior rollout 的中间隐状态可以 cache 起来被 action expert 反复 query,这就避免了在每个 control step 都跑一个 expensive 的 generative model。

## 3. Data Curation:10,000+ 小时混合模态数据

数据集由三部分组成:
$$\mathcal{D} = \mathcal{D}_{VM} \cup \mathcal{D}_V \cup \mathcal{D}_M$$

- $\mathcal{D}_{VM}$: paired video-motion(egocentric video + 同步人体 motion),supervises joint distribution
- $\mathcal{D}_V$: video-only,只约束 visual marginal
- $\mathcal{D}_M$: motion-only,只约束 motion marginal

来源涵盖 Ego4D、Xperience、Nymeria、Bones-SEED、SnapMoGen、HumanML3D、Lafan1,加上 Being-H0.5 和 Being-M0.5 的 internal 数据。这个 design 的 intuition 是:**paired data 量太少(千小时级),但 video-only 和 motion-only 都各自海量(万小时级)**,通过 Mixture-of-Transformers 的 modality-specific branch,可以让单模态样本也参与训练,等价于分别约束 marginal 分布,paired data 负责对齐两个 marginal。

### 3.1 Latent Visual Representation

所有 frame 通过 **frozen DINO encoder** 编码:$Z = \mathcal{E}(V)$。模型在 DINO latent space 中做生成,而**不在 pixel space 中生成**。这一步是 paper 标题里"latent"的关键来源:
- 大幅降维(224×224×3 → 几百维 latent token)
- 强迫模型关注 semantic state 和 object layout,忽略 texture、lighting 等控制无关细节
- 避免 pixel-level video prediction 在 fast ego-motion 下的崩坏

### 3.2 Unified Motion Representation:Head-root 22D

这是一个我认为非常聪明的 design choice。常规人体 motion representation 用 pelvis 作为 root,但 Being-M0.7 选择 **head 作为 root**。原因很直觉:egocentric video 是头戴相机拍的,head 的 motion 直接决定了视觉流,把 head 设为 root 就把 motion 表示和 visual 流"对齐"到同一个参考系。

compact representation 只保留 5 个关键关节:**head(root) + left hand + right hand + left foot + right foot**,共 22D。这个表示:
- 不追求全身体感细节,但保留了 interaction 和 contact 最关键的信息
- 可以同时从 human motion capture 和 humanoid robot forward kinematics 构造出来,**直接桥接 morphology gap**
- 在 inference 时提供 motion-level feedback path:当前 robot kinematics 可以和预测的 motion plan 比较,做偏差纠正

## 4. Model Architecture:Mixture-of-Transformers

### 4.1 Prior World-Action Model

prior model 用 **Mixture-of-Transformers (MoT)** 实现,这是 Meta 2024 年提出的 sparse multimodal 架构([Liang et al., 2024](https://arxiv.org/abs/2411.04996))。其核心 idea 是:**modality-specific 参数 + 共享 multimodal attention**。每个 block 内:
- Vision tokens 和 motion tokens 各自有独立的 LayerNorm、Q/K/V projection、output projection、FFN
- 然后把所有 projected Q/K/V **拼接在一起做一次共享 attention**
- 这样跨模态交互通过 attention 完成,但参数空间是分离的

这使得同一个架构可以同时吃 paired、video-only、motion-only 三种数据——missing modality 时只对 present modality 计算 loss 就行。

Flow timestep $\tau$ 通过 **modality-specific AdaLN** ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748))注入,即 vision 和 motion 各自有独立的 AdaLN 参数。

Attention mask 设计:
- Context frames(前 K 帧):clean tokens,只能 attend 到 history,不能看 future
- Future frames(后 T-K 帧):noisy,可以 attend 到 full history 以及彼此(non-causal target attention)
- 这允许模型**并行 denoise 整个 future chunk**,类似 masked prediction + diffusion 的混合

### 4.2 Action Expert

action expert 是一个独立的 6-layer flow transformer,挂在 prior model 的多个中间层上。核心公式:

$$A_q^{j+1} = \mathcal{B}_j(A_q^j \mid [H_{\ell_j}^\theta(\tau), A_q^j, O_q], \tau)$$

含义:
- $A_q^j$: expert 第 $j$ 层的 action token hidden state
- $\mathcal{B}_j$: 第 $j$ 个 expert block
- $H_{\ell_j}^\theta(\tau)$: prior model 第 $\ell_j$ 层在 flow timestep $\tau$ 下的 hidden state
- $\ell_j$: expert 第 $j$ 层连接到 prior model 的哪一层(实际配置是 [1, 7, 13, 18, 24, 30],即在 30 层 prior 里均匀取 6 层)
- $O_q$: 当前 robot observation
- **单向信息流**:prior → expert,prior 不接收 action 信息

这个设计让 expert 可以同时利用 prior 的浅层 fine-grained 特征和深层 high-level future dynamics,而 $O_q$ 提供了 closed-loop responsiveness。轻量 expert 的好处是 inference 速度快(<0.01s/query),能够跟 50Hz control rate。

### 4.3 架构 hyperparameters 总结

| Component | Layers | Hidden | Heads | FFN ratio |
|---|---|---|---|---|
| Video-Motion Prior | 30 | 768 | 24 | V:6, M:2 |
| Action Expert | 6 | 384 | 8 | 4 |
| Connected prior layers | 1, 7, 13, 18, 24, 30 (均匀采样) | | | |
| Action chunk length | 10 (但执行 chunk 是 20, 因为 50Hz) | | | |

prior 总规模大概在 ~300M-400M 参数量级,expert 大概 30M-50M,设计上非常不对称,符合"低频世界模型可以大,高频 policy 必须轻"的直觉。

## 5. Training Recipes

### 5.1 Flow Matching Objective

paper 用 flow matching([Lipman et al., 2023](https://arxiv.org/abs/2210.02747))而非 DDPM-style diffusion。线性 path:
$$x_\tau = (1-\tau)x_0 + \tau \epsilon, \quad u^\star = \epsilon - x_0$$

其中 $x_0$ 是 clean target(visual latent 或 motion token),$\epsilon \sim \mathcal{N}(0, I)$ 是噪声,$\tau \sim \mathcal{U}(0,1)$ 是 flow timestep。模型预测 velocity field $u_\theta(x_\tau, \tau, I, Z_C, M_C)$,loss 是 MSE:

$$\mathcal{L}_V = \frac{1}{|\Omega_V|} \sum_{t \in \Omega_V} \| u_\theta^V(z_{\tau,t}, \tau, I, Z_C) - u_t^{V\star} \|_2^2$$
$$\mathcal{L}_M = \frac{1}{|\Omega_M|} \sum_{t \in \Omega_M} \| u_\theta^M(m_{\tau,t}, \tau, I, M_C) - u_t^{M\star} \|_2^2$$

$\Omega_V$ 和 $\Omega_M$ 分别是该 sample 中 valid visual / motion token 的 index 集合。这个 mask 机制让 mixed-modality training 自然落地。

### 5.2 Geometry-aware Auxiliary Loss

针对 motion branch,paper 加了两个 geometry loss:
$$\mathcal{L}_{\text{geom}} = \lambda_{\text{traj}} \mathcal{L}_{\text{traj}} + \lambda_{\text{local-vel}} \mathcal{L}_{\text{local-vel}}$$

- $\mathcal{L}_{\text{traj}}$: 约束 decoded global head/root 位置轨迹正确(在欧式空间)
- $\mathcal{L}_{\text{local-vel}}$: 约束 frame-to-frame end-effector velocity(在 head frame 下)

这相当于让模型在 latent/velocity space 学习的同时,**额外监督解码后的几何物理量**,防止 flow matching 优化到 motion 表示的"流形"外面去。

### 5.3 Pre-training Objective

$$\min_\theta \mathbb{E}_{s \sim \mathcal{D}_{VM}} [\mathcal{L}_V + \mathcal{L}_M + \mathcal{L}_{\text{geom}}] + \mathbb{E}_{s \sim \mathcal{D}_V} [\mathcal{L}_V] + \mathbb{E}_{s \sim \mathcal{D}_M} [\mathcal{L}_M + \mathcal{L}_{\text{geom}}]$$

注意 motion-only 也有 $\mathcal{L}_{\text{geom}}$,但 video-only 没有,因为 video-only 没有 motion target 可以解码。

### 5.4 Post-training Objective

robot trajectory 同时提供 vision target、motion target(via forward kinematics)、action target 三种 supervision。loss 加权:

$$\min_{\theta, \phi} \mathcal{L}_V(\theta) + \mathcal{L}_M(\theta) + \lambda_a \mathcal{L}_A(\theta, \phi)$$

$$\mathcal{L}_A(\theta, \phi) = \mathbb{E} \| u_\phi^a(\tilde{\mathbf{a}}_\tau, \tau, H^\theta(\tau), o_q) - (\epsilon_a - \mathbf{a}_q) \|_2^2$$

这里一个细节:$\mathcal{L}_V$ 和 $\mathcal{L}_M$ 在 post-training 阶段仍然在更新 prior,等于让 robot 数据反过来 fine-tune prior 的 visual-motion dynamics,使 prior 更适应 robot egocentric 分布。

## 6. Inference:Hierarchical Frequency Design

inference 的时间结构是这个系统最 practical 的部分:

| Component | Frequency | Cache refresh |
|---|---|---|
| Prior world-action model | 5 Hz | every 3.5s |
| Action expert | 50 Hz | every 0.4s |
| Low-level WBC (SONIC) | 50 Hz | continuous |

具体流程:
1. Prior 用 N=10 flow matching steps 去噪,得到 future $(Z_\mathcal{Q}, M_\mathcal{Q})$ 以及中间层隐状态 $H_n^\theta$
2. $H_n^\theta$ 被转换为 KV cache,policy 侧缓存
3. Prior rollout **每 3.5s 刷新一次**
4. Action expert 每 0.4s 调用一次(预测 20 个 50Hz action = 0.4s horizon),reusing cached prior context
5. 每个 action query 用最新 $O_{\text{cur}}$ 做 closed-loop correction

Denotation update rule:
$$\hat{u}^n = u_\phi^a(\mathbf{a}^{n-1}, \tau_n, H_n^\theta, O_{\text{cur}}), \quad \mathbf{a}^n = \mathbf{a}^{n-1} - \hat{u}^n \Delta \tau_n$$

**一次 prior rollout 大约支撑 ~9 次 action query**,这就是为什么 heavy prior 可以负担得起——它的算力被"摊销"到 9 个 control windows 上。Action expert query 单次 < 0.01s,符合 50Hz 实时性。

## 7. Hardware & Teleop Pipeline

- Robot: Unitree G1
- Arms: dual 7-DoF + Linker Hand O6 dexterous hands
- Camera: Intel RealSense D435i(head-mounted)
- GPU: 单卡 RTX 4090
- Low-level controller: SONIC([Luo et al., 2025](https://arxiv.org/abs/2511.07820)),输出 29-DoF whole-body command
- Teleop: PICO VR headset + 2 ankle trackers + 2 handheld controllers,XRoboToolkit 做 SMPL pose estimation,SONIC 把 SMPL pose 转 robot command

注意 robot observation $O_q$ 包含三个部分:
- egocentric image(通过 frozen DINO 编码)
- proprioceptive state(joint positions + joint velocities + gravity direction projected into robot frame)
- normalized execution progress(在 [0,1] 范围内表示当前 chunk 执行到哪了)

execution progress 这一项的设计很直觉:让 expert 知道"我现在的 chunk 执行到第几帧了",从而可以做 action smoothing / 重新规划判断。

## 8. Experiments

### 8.1 四个 Real-world 任务

| Task | Capability tested |
|---|---|
| Mirror Toy Grasping | Partial observability + reflection reasoning + whole-body approach |
| Water-Tank Fish Scooping | Tool use + visual distortion(water) + object dynamics |
| Tabletop Organization | Long-horizon multi-stage manipulation |
| Obstacle-Avoidance Basket Carrying | Locomotion-manipulation coordination + obstacle avoidance |

### 8.2 定量结果

| Method | Mirror (Near 0.5m) | Mirror (Far 1m) | Fish | Overall |
|---|---|---|---|---|
| GR00T-N1.6 | 1/5 | 0/5 | 1/5 | 2/15 |
| Ψ₀ | 0/5 | 1/5 | 2/5 | 3/15 |
| **Being-M0.7** | 3/5 | 1/5 | 3/5 | **7/15** |

整体成功率 **7/15 vs 3/15 vs 2/15**,优势显著但绝对值仍然不高,说明 humanoid loco-manipulation 离 robust generalization 还有距离。

### 8.3 与 baselines 的对比直觉

- **GR00T-N1.6**([Bjorck et al., 2025](https://arxiv.org/abs/2503.14734)):VLA 架构 + diffusion transformer action head,但缺少显式 world modeling。在 mirror reasoning 这种 partial observability 任务上表现差,说明没有 future prediction 能力难以推理不可见物体。
- **Ψ₀**([Wei et al., 2026](https://arxiv.org/abs/2603.12263)):从 egocentric video 学 latent action,再 post-train flow-based action expert。思路接近,但 Ψ₀ 的 video pretraining 主要做 representation learning,而 Being-M0.7 是 **generative world model**,显式预测 future latent state。
- **Being-M0.7** 的优势主要来自两处:(1) latent world modeling 让模型显式 reason about future;(2) unified motion representation 让 upper/lower body coordination 在 prior 阶段就学会,而非 post-training 才开始。

## 9. 与相关工作的关系图

### 9.1 与 Being 系列前作的关系

- **Being-H0.5**([Luo et al., 2026](https://arxiv.org/abs/2601.12993)): cross-embodiment generalist,human-centric pretraining
- **Being-H0.7**([Luo et al., 2026](https://arxiv.org/abs/2605.00078)): latent world-action model from egocentric videos,**没有 motion**
- **Being-M0.5**([Cao et al., 2025](https://arxiv.org/abs/2508.07863)): real-time controllable VLM model
- **Being-M0.7**: 把 H0.7 的 latent world model 扩展到 video-motion 联合建模,加上 future-conditioned action expert

可以看出 Being 系列在持续向"显式 world modeling + 解耦执行"的方向迭代。

### 9.2 与 MotionWAM 的对比

MotionWAM([Zheng et al., 2026](https://arxiv.org/abs/2606.09215))也是 humanoid world action model,思路相近,但有关键区别:
- MotionWAM 直接用 egocentric video generation model 的 denoising feature 条件化 motion model,**仍然在 pixel space**
- Being-M0.7 完全在 latent space 做 world modeling,避免 pixel generation 的开销和 noise

### 9.3 与 π0 / GR00T N1 的对比

- π0([Black et al., 2024](https://arxiv.org/abs/2410.24164)): VLA + flow matching action expert,**没有显式 world model**。prior knowledge 全部压缩在 VLM 的 weights 里。
- GR00T N1: 类似 VLA + diffusion action
- Being-M0.7: **把 world model 显式化为一个 generative head**,等于在 VLA 上多加了一个"想象未来"的模块。这跟 Du et al. 的 [Learning Universal Policies via Text-Guided Video Generation](https://arxiv.org/abs/2302.01111) 思路一脉相承,但 latent 化了。

### 9.4 与 Cosmos 的关系

Cosmos([Agarwal et al., 2025](https://arxiv.org/abs/2501.03575))是 NVIDIA 的 world foundation model,pixel-level world generation。Being-M0.7 可以看作"Cosmos 的 latent 版本针对 humanoid 优化"——它放弃了 pixel fidelity,换来了 control-relevant 的语义压缩。

## 10. Intuition 总结:为什么这个设计可能 work

从 first principles 思考这个 architecture 为什么合理:

1. **Time-scale separation**:humanoid control 本质上有两个时间尺度——~200ms 的"想象和规划"和 ~20ms 的"反馈执行"。把它们分到两个网络里,等于显式建模了这种 hierarchy。
2. **Latent > Pixel for control**:robot 真正需要的是"前方有桌子、toy 在 mirror 里、手要伸过去抓",而不是"前方桌子的木纹纹理"。DINO latent 把这层抽象做好。
3. **Motion 作为 coarse action plan**:直接预测 50Hz 的 29-DoF command 是高维稀疏的,但预测 5 个关键点的 5Hz motion 是低维稠密的——这是更"sample efficient"的 supervision signal。
4. **Mixed-modality training via MoT**:让海量单模态数据也能用上,这是 scaling 的关键。如果只用 paired data,数据量卡在千小时级,永远 scale 不上去。
5. **Future-conditioned action expert**:这跟 [Future-Conditioned Unsupervised Pretraining for Decision Transformer](https://arxiv.org/abs/2303.01010) 的思想类似——让 policy 知道"未来应该长什么样",比单纯 react to current state 更强。

## 11. 我的几个观察 / 潜在 concerns

- **22D motion 表示信息量**:只保留 5 个 keypoint 会丢掉躯干姿态、脊柱弯曲等。对于需要精细 posture control 的任务(比如弯腰捡东西),这个表示可能不够。未来可能需要扩展到更多 keypoint 或 latent full-body representation。
- **Latent space drift**:DINO latent 是在 internet image 上训练的,robot egocentric 分布可能 OOD。Paper 里 prior 在 post-training 阶段继续更新,等于做了 light fine-tuning,但没说 DINO 本身有没有更新——从描述看是 frozen。
- **Prior cache refresh 3.5s 太长?**:如果 robot 突然遇到 obstacle,需要快速 re-plan,3.5s 的 refresh rate 可能太慢。motion-level feedback path 部分缓解,但本质问题没解决。
- **Quantitative 结果 7/15 仍然偏低**:说明这个方向 promising 但远未成熟。Mirror Far 只有 1/5,说明远距离 + reflection reasoning 仍然非常困难。
- **No simulation experiments**:paper 全是 real-world 实验,缺少 ablation 和 scaling curve。比如:去掉 motion branch 会怎样?prior 规模 scaling 怎么影响?这些都没给。
- **Comparison fairness**:Ψ0 和 GR00T-N1.6 是否在相同 robot 上 fine-tuned?数据量是否对齐?paper 没明确说明,但定量比较的可信度依赖于此。

## 12. Future directions 我觉得值得探索

1. **Latent space 的可解释性**:能不能可视化 latent world model 预测的 $Z_\mathcal{Q}$?能不能 decode 回 image 看看预测的是不是合理?这对 debugging 和 trust 很重要。
2. **Multi-modal future sampling**:目前 flow matching 给一个 sample,但 future 本身有多模态性(可以走左边或右边)。能不能用多个 sample 做 trajectory ensemble 或 MPC?
3. **Hierarchical world model**:5Hz 的 prior 还是太粗。能否引入三层:1Hz scene graph + 5Hz visual-motion latent + 50Hz action?
4. **Active perception**:让 prior 主动预测"如果我转头看那边会看到什么",而不仅仅是被动预测 future。
5. **Cross-embodiment scaling**:unified motion representation 已经为这个铺路,但 paper 只在 G1 上验证。迁移到 H1、Apollo、甚至非 humanoid 是开放问题。

## 13. 相关参考链接

- **Being-M0.7 本身**(假设 arXiv 链接):paper 中没有给,但 Being 系列前作在 https://arxiv.org/abs/2605.00078 (Being-H0.7)
- **Mixture-of-Transformers**: https://arxiv.org/abs/2411.04996
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **DiT (AdaLN)**: https://arxiv.org/abs/2212.09748
- **DINOv3**: https://arxiv.org/abs/2508.10104
- **π0**: https://arxiv.org/abs/2410.24164
- **GR00T N1**: https://arxiv.org/abs/2503.14734
- **Ψ0**: https://arxiv.org/abs/2603.12263
- **WholeBodyVLA**: https://arxiv.org/abs/2602.10106 (类似 paper, EgoHumanoid)
- **MotionWAM**: https://arxiv.org/abs/2606.09215
- **SONIC**: https://arxiv.org/abs/2511.07820
- **Ego4D**: https://arxiv.org/abs/2110.07058
- **Nymeria**: https://arxiv.org/abs/2406.06043
- **AMASS**: https://amass.is.tue.mpg.de/
- **HumanML3D**: https://github.com/EricGuo5513/HumanML3D
- **ACT (action chunking)**: https://tonyzhaozh.github.io/aloha/
- **Ego-Exo4D**: https://arxiv.org/abs/2311.18258
- **Du et al. Video Policies**: https://arxiv.org/abs/2302.01111
- **Cosmos World Model**: https://arxiv.org/abs/2501.03575
- **Genie (interactive env)**: https://arxiv.org/abs/2402.15391

---

整体看,Being-M0.7 是一个**设计相当克制、目标相当明确**的工作。它没有追求"end-to-end 一把梭",而是认真思考了 humanoid 这个 problem 本身的时间尺度、数据稀缺、形态鸿沟三大瓶颈,然后给了一个有原则的分解。Latent world model + future-conditioned action expert 这个架构可能成为 humanoid foundation model 的一个重要范式——就像 LLM 中的 "think before you speak" 一样,robot 也开始学会"imagine before you act"。
