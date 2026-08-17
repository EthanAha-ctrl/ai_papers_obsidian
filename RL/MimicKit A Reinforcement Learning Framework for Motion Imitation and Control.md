---
source_pdf: MimicKit A Reinforcement Learning Framework for Motion Imitation and Control.pdf
paper_sha256: 3e337492362e7380052c9fdb1f1343840e065bde87e15236af9e0fcb007c03d7
processed_at: '2026-08-05T18:22:15-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 MimicKit

Andrej, 我换个角度,用大白话把这篇 paper 捋一遍。

## 这玩意儿到底是个啥

Xue Bin Peng 这哥们儿搞 motion imitation 搞了快十年,从 2018 年的 DeepMimic 一直到 2025 年的 ADD,积累了大量经验。MimicKit 就是把他这些年踩过的坑、调过的参、写过的代码,打包成一个开源 framework 放到 GitHub 上 (https://github.com/xbpeng/MimicKit)。

你可以理解成: motion imitation 领域的 "nn.zero_grad()" 到 "optimizer.step()" 的标准模板。以前每个研究者要自己从零搭一套系统,现在直接用他的就行。

## 要解决啥问题

假设你想让一个虚拟小人学会后空翻。你有两个选择:

**选择 A: 写 reward function**
告诉 AI "脚离地越高分越高,旋转越快分越高,落地站稳分越高..."。问题是后空翻这种动作,你怎么定义 "好"?写出来的 reward function 基本都是 garbage in garbage out。

**选择 B: 给个示范**
找个真人录一段后空翻的 motion capture 数据,告诉 AI "照着这个学"。这就是 motion imitation。

MimicKit 走的是 B 路线,并且提供了四种具体的 "照着学" 的方法。

## 四种方法,用大白话

### DeepMimic (2018, https://arxiv.org/abs/1804.02717)

**比喻**: 像描红练字。

你给 AI 一段 reference motion,每一帧都告诉它 "这个时候你的左手应该在这儿,右手应该在那儿,躯干旋转到这个角度"。AI 每一步都跟 reference 对比,差得越远扣分越多。

reward function 是手工拼出来的:

$$r_t = w^{pose} r_t^{pose} + w^{vel} r_t^{vel} + w^{end} r_t^{end} + w^{COM} r_t^{COM}$$

这里的 $w^{pose}, w^{vel}, w^{end}, w^{COM}$ 就是各部分的权重,你得自己调。走路可能 end-effector 权重高一点,后空翻可能 COM 旋转权重要高一点。每个 motion 都得重新调,烦死人。

好处: 精确,稳定,简单。Walk 的 tracking error 才 9mm,基本完美复刻。

坏处: 死板。学会一个 backflip 就只会这个 backflip,你让它往前走两步再 backflip 它就懵了。

### AMP (2021, https://arxiv.org/abs/2104.02180)

**比喻**: 像学风格,不学具体。

AMP 借鉴 GAN 的思路。你给它一堆 motion clips 当 "real data",训练一个 discriminator 去判断 "这个动作像不像真人做的"。policy 生成的动作如果骗过了 discriminator,说明风格学到了。

公式上,style reward 是:

$$r_t^{style} = -\log(1 + \exp(-D(s_t, s_{t+1})))$$

discriminator $D$ 吃的是一个 state transition $(s_t, s_{t+1})$,不是单帧。这点很关键,因为单帧容易作弊(站个 weird pose 也能骗过),transition 能捕捉 dynamics。

好处: 灵活。可以 compose 多个 motion,比如 walk + dance 混着来。适合配 task reward 做 "走到目标点,但走路姿势要帅" 这种任务。

坏处: 不精确。Table 1 里 Run 的 position error,AMP 是 0.163m,DeepMimic 是 0.013m,差 12 倍。而且 GAN 的老毛病 — 容易 collapse 到 local optima,training 不稳定。

### ASE (2022, https://arxiv.org/abs/2205.07506)

**比喻**: 像学一套 "动作词汇表",以后用词汇表造句。

ASE 是 hierarchical 的。Low-level policy 先学一堆 skills,每个 skill 对应 latent space 里的一个点 $z$。给 $z$ 就生成对应 skill 的动作。High-level policy 再学怎么选 $z$ 来完成具体 task。

训练 low-level 时加了一个 mutual information objective:

$$r_t^{MI} = \log \frac{\pi(a|s, z)}{\pi(a|s)}$$

意思是 "给定 $z$ 的 action 分布" 要跟 "不给 $z$ 的 action 分布" 有差异,逼着每个 $z$ 对应不同的 skill。这个思路来自 DIAYN (https://arxiv.org/abs/1802.06070)。

好处: skill 可复用。训练完一个 ASE controller,以后新任务只要训 high-level,low-level 直接用。

坏处: 训练复杂,两个 level 要分别训。

### ADD (2025, SIGGRAPH Asia 2025)

**比喻**: 像让 AI 自己学会怎么打分。

DeepMimic 要你手工写 reward function,AMP 完全不 track 具体动作。ADD 想当中间路线 — 还是要 track reference motion,但 reward function 让 discriminator 自己学。

"differential discriminator" 的意思我理解是 discriminator 不看 state 本身,看 simulated state 和 reference state 的 **差异** $(s - \hat{s})$。这样它能学到 "对于 backflip 来说,哪些 tracking error 重要,哪些不重要"。

从 Table 1 数据看,ADD 在 challenging motion 上 variance 最小。Backflip 的 position error,DeepMimic 是 $0.111^{\pm 0.054}$,ADD 是 $0.062^{\pm 0.001}$。而且 DeepMimic 在 Roll 上 std/mean 高达 115%,说明有些 random seed 直接训崩了,ADD 几乎没这个问题。

直觉上,ADD 解决了 DeepMimic "每个 motion 都要手调 reward" 的痛点,同时保留了 tracking 的精度。

## 系统架构,用大白话

paper 把整个系统切成四块:

```
Agent (学啥) → Model (NN 长啥样) → Environment (任务长啥样) → Engine (物理引擎是啥)
```

**Agent**: 决定用 PPO 还是 AWR,怎么 collect data,怎么 update model。对应代码 `mimickit/learning/`。

**Model**: 具体的神经网络。actor-critic 就两个网络,AMP 还要加 discriminator。对应 `mimickit/learning/` 里跟 Agent 配对的文件。

**Environment**: 具体任务逻辑。给 character 构造 observation,处理 action,算 reward,判断 episode 是否结束。对应 `mimickit/envs/`。

**Engine**: 底层物理仿真器。目前支持 Isaac Gym (https://arxiv.org/abs/2108.10470), Isaac Lab (https://arxiv.org/abs/2511.04831), Newton (https://github.com/newton-physics/newton)。对应 `mimickit/engines/`。

这种分层的好处: 同一个 backflip 任务,你可以今天用 Isaac Gym 训,明天换 Newton,代码几乎不动。同一个 humanoid character,你可以今天用 DeepMimic 训,明天换 ADD,也几乎不动。

## 几个工程细节值得提

### Done flag 的四种状态

这个设计很细致。一般 RL framework 只有 done/not-done 两种。MimicKit 分了四种:

- **NULL**: 还在跑
- **FAIL**: 摔倒了,扣分
- **SUCC**: 任务完成,加分
- **TIME**: 时间到了,但本来应该继续跑,用 value function bootstrap 估算未来 return

TIME 这个 flag 是关键。RL 里 episode 截断时,如果直接把 future return 当 0,会 bias 估值。正确做法是用 $V(s_T)$ 估算。MimicKit 显式区分了这个 case。

参考 OpenAI baselines 的 GAE 实现 (https://github.com/openai/baselines),last step bootstrap 是标准操作,但很多 framework 没显式区分这个 case。

### 4096 个并行环境

paper 里训练命令默认 `--num_envs 4096`。这是 Isaac Gym 的玩法: 4096 个 humanoid 同时在 GPU 上跑物理仿真,policy 也 batched 更新。throughput 能到 $10^5$+ FPS。

这种规模下,PPO 虽然 on-policy sample inefficient,但 GPU simulator 吐数据太快,不是 bottleneck。所以 MimicKit 主推 PPO,AWR 只是 off-policy 的备选。

这个思路跟 NVIDIA 整个 Isaac 生态一脉相承。Isaac Lab paper (https://arxiv.org/abs/2511.04831) 里也是这个架构。

### Motion data 格式

每帧 motion 存成:
```
[root_pos (3D), root_rot (3D), joint_rots...]
```

rotation 用 **exponential map**,就是 axis-angle 表示 $\mathbf{r} = \theta \hat{\mathbf{n}}$。长度是旋转角 $\theta$,方向是旋转轴 $\hat{\mathbf{n}}$。

为啥不用 quaternion?因为 quaternion 4 维,exponential map 3 维,少一维。而且 exponential map 插值方便,连续性好。缺点是接近 0 或 $2\pi$ 时数值不稳。

这个表示法来自 Grassia 1998 (https://www.tandfonline.com/doi/abs/10.1080/10867651.1998.10487493),是 graphics 老传统了。DeepMimic 沿用,MimicKit 也沿用。

humanoid 的 joint 顺序是 depth-first traversal kinematic tree,跟 MuJoCo XML 文件一致。具体顺序 paper 里列了 14 项,从 root 到 abdomen 到 neck 到四肢。

## 评估 metric 的设计巧思

paper 定义了两个 tracking error,设计很讲究。

### Position Tracking Error

$$e_t^{pos} = \frac{1}{N^{joint}+1}\left(\sum_{j} \|(\hat{\mathbf{x}}_t^j - \hat{\mathbf{x}}_t^{root}) - (\mathbf{x}_t^j - \mathbf{x}_t^{root})\|_2 + \|\hat{\mathbf{x}}_t^{root} - \mathbf{x}_t^{root}\|_2\right)$$

这里 $\mathbf{x}_t^j$ 是 joint $j$ 的 global position,$\hat{\mathbf{x}}_t^j$ 是 reference 的。

第一项算的是 **相对 root 的 joint 位置** 差异。$(\mathbf{x}_t^j - \mathbf{x}_t^{root})$ 把 joint 位置减去 root 位置,等于剥离了全局平移,只看 character 自身形状对不对。

第二项单独算 root position 差异,衡量全局运动(往前走了多远)。

这样分开算有道理: character 可能形状对了但位置偏了,或者位置对了但姿势歪了,两个 error 独立衡量更清晰。

### DoF Velocity Tracking Error

$$e_t^{vel} = \frac{1}{N^{joint}+1} \sum_{j} \|\hat{\dot{\mathbf{q}}}_t^j - \dot{\mathbf{q}}_t^j\|_2$$

$\dot{\mathbf{q}}_t^j$ 是 joint $j$ 的 local angular velocity。

这个 metric 衡量 motion smoothness。pose 对得上但 velocity 对不上,动作会 "颤抖",看着不自然。

## 实验数据的 story

Table 1 我仔细看了,讲几个有意思的点:

**1. AMP 整体最差,但不是没用**

AMP 的 position error 普遍比 DeepMimic/ADD 大一个数量级。Walk 上 AMP 0.132 vs DeepMimic 0.009,差 15 倍。

但 paper 说 "qualitatively AMP can still be effective at reproducing the general behaviors"。意思是 AMP 虽然不能精确复刻,但能学到 "大概那个样子"。这在需要 flexibility 的场景(比如游戏 NPC)其实够用。

**2. DeepMimic 在简单 motion 上 SOTA**

Walk 0.009m,Crawl 0.027m,GetupFacedown 0.023m,都是最低 error。手工 reward 调好的话,简单 motion 上确实无敌。

**3. ADD 在难 motion 上最稳**

Backflip: DeepMimic $0.111^{\pm 0.054}$, ADD $0.062^{\pm 0.001}$。ADD 不光 mean 低,std 还极小。

Roll 上 DeepMimic 的 std/mean = 115%,说明 5 个 seed 里有些直接训崩。ADD std/mean = 3.3%,稳如老狗。

这个对比是 ADD paper 的核心 selling point: 手工 reward 在 challenging motion 上 unstable,adaptive reward 解决这个问题。

**4. 实验条件对 DeepMimic 不利**

paper 里说为了公平比较,关掉了 pose-error termination。这是 DeepMimic 原版的标准 trick — pose 偏离 reference 太远就 early terminate。关掉之后 DeepMimic 变弱了,AMP 才能跟它比。

如果打开 pose-error termination,DeepMimic 和 ADD 都会更稳。这说明 paper 的实验设计是 "故意给 AMP 创造公平环境",但也意味着 DeepMimic 的真实能力被低估了。

## 我的联想

### 跟 RLHF 的类比

AMP 的 style reward 跟 RLHF 的 reward model 思路几乎一样:

- RLHF: 用人类偏好数据训 reward model,reward model 给 policy 打分
- AMP: 用 motion data 训 discriminator,discriminator 给 policy 打分

两者都是 "用数据学 reward function,再用 RL 优化 policy"。AMP 2021 年发表,比 RLHF 火起来还早一点。参考 InstructGPT (https://arxiv.org/abs/2203.02155) 是 2022 年。

### 跟 GAN 的关系

AMP 本质是 GAN for motion。GAN 的问题 AMP 都有:

- mode collapse: policy 只学到一种典型 motion
- training instability: discriminator 和 policy 博弈容易失衡
- local optima: paper 里明确说 AMP "more prone to converging to local optima"

ADD 用 differential discriminator 缓解了这个问题,思路类似 WGAN-GP (https://arxiv.org/abs/1704.00028) 用 gradient penalty 稳定 GAN training。

### 跟 LLM agent 的 skill library

ASE 学一个 latent space,每个点对应一个 skill。这跟最近 LLM agent 里的 "skill library" 概念很像:

- Voyager (https://arxiv.org/abs/2305.16291): Minecraft agent 把学到的 skill 存成 code,以后调
- ASE: physics agent 把 skill 存成 latent vector,以后 high-level policy 选

区别是 Voyager 的 skill 是 symbolic (code),ASE 的 skill 是 continuous (latent vector)。continuous 的好处是 smooth interpolation,坏处是可解释性差。

### 跟 Diffusion Motion Generation

最近 motion generation 领域 diffusion model 火爆:

- MDM (https://arxiv.org/abs/2209.10915): Motion Diffusion Model
- MoMask (https://arxiv.org/abs/2403.11031): Generative masked transformer

这些都是 kinematic 的 — 生成 motion 数据,不保证物理可行。可能出现 "脚穿地"、"重心违反物理" 的问题。

MimicKit 这套 RL-based 方法保证 physical plausibility,因为每一步都过物理引擎。一个有意思的方向: diffusion 生成 reference motion,MimicKit 训 controller 去 track,得到 physical-plausible 的 motion。类似 PhysDiff (https://arxiv.org/abs/2401.03121) 的思路。

### 跟 EUREKA 的结合

EUREKA (https://arxiv.org/abs/2310.12931) 是 NVIDIA 的工作,用 LLM 自动生成 reward function 代码。DeepMimic 的痛点是手工调 reward,ADD 用 discriminator 学 reward,EUREKA 用 LLM 生成 reward — 三条路解决同一个问题。

未来可能: EUREKA 生成 reward function candidate,ADD 的 discriminator refine,结合两者优势。

### 跟 World Model

MimicKit 的 RL training 每步都要跑物理仿真,仿真虽然 GPU 加速了但还是 expensive。World model (e.g. Dreamer, https://arxiv.org/abs/1912.01603) 学一个 dynamics model,在 imagination 里 rollout,减少真实仿真步数。

motion imitation 任务 dynamics 相对 structured (character physics),world model 可能学得好。如果 MimicKit 未来加 world model backend,training 速度可能再提一个数量级。

## 局限性,直说

**1. 只支持 NVIDIA 系**

目前 Engine 只支持 Isaac Gym, Isaac Lab, Newton。不支持 MuJoCo (CPU), Brax (Google TPU), 或者其他 simulator。对学术界不算 NVIDIA 生态的研究者不够友好。

**2. 没有 sim-to-real**

paper 里说 framework "support deployment on real robots" 但目前只是 future work。motion imitation 的 sim-to-real 有 gap: simulator 里 PD controller 完美执行,real robot 有摩擦、delay、motor dynamics 差异。

**3. 只有 PPO 和 AWR**

SAC, TD3, DDPG 这些都缺。motion imitation 任务 PPO 够用,但研究新 RL algorithm 的人可能想试别的。

**4. Policy 只有 MLP**

没有 transformer policy,没有 attention mechanism。最近 trend 是 transformer for control (e.g. Decision Transformer, https://arxiv.org/abs/2106.01345),MimicKit 没跟上。

**5. Motion data 格式封闭**

用 .pkl + exponential map,跟主流的 SMPL (https://smpl.is.tue.mpg.de/), BVH, FBX 格式有 gap。需要自己写 converter。

## 最终直觉

MimicKit 本质上是 Xue Bin Peng 的 "motion imitation 研究者工具箱"。它不发明新方法,而是把已有方法的 best practice 标准化。

类比一下:
- PyTorch Lightning 之于 PyTorch
- Hugging Face Transformers 之于 NLP
- MimicKit 之于 motion imitation

价值在于降低入门门槛,让后来者不用重新造轮子。对 motion imitation 领域的 research velocity 有正向作用。

从 Karpathy 你的视角看,这种 "把隐性知识显性化" 的工作很 valuable。你做 micrograd (https://github.com/karpathy/micrograd) 让人理解 backprop,MimicKit 让人能快速尝试 motion imitation 的各种变体。都是把复杂系统的细节 transparent 出来,推动领域进步。

参考链接汇总:
- MimicKit: https://github.com/xbpeng/MimicKit
- DeepMimic: https://arxiv.org/abs/1804.02717
- AMP: https://arxiv.org/abs/2104.02180
- ASE: https://arxiv.org/abs/2205.07506
- PPO: https://arxiv.org/abs/1707.06347
- Isaac Gym: https://arxiv.org/abs/2108.10470
- Isaac Lab: https://arxiv.org/abs/2511.04831
- Newton: https://github.com/newton-physics/newton
- DIAYN: https://arxiv.org/abs/1802.06070
- EUREKA: https://arxiv.org/abs/2310.12931
- Voyager: https://arxiv.org/abs/2305.16291
- PhysDiff: https://arxiv.org/abs/2401.03121
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Dreamer: https://arxiv.org/abs/1912.01603
- WGAN-GP: https://arxiv.org/abs/1704.00028
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- GAE: https://arxiv.org/abs/1506.02438
- AWR: https://arxiv.org/abs/1910.00177

---

# MimicKit: 详细技术讲解

Andrej, 这篇 paper 是 Xue Bin Peng (DeepMimic 的作者) 把多年 motion imitation 研究沉淀下来的开源 framework,本质上是把 DeepMimic / AMP / ASE / ADD 这些方法整合到一个统一架构里。我把它当成 motion imitation 领域的 "mini-RL-Games" 来看,可以参考 https://github.com/xbpeng/MimicKit。

## 1. 整体设计哲学

这个 framework 的核心抽象是把 RL 系统切成四层:

```
Agent (learning algorithm)  →  Model (NN architecture)  →  Environment (task logic)  →  Engine (physics simulator)
```

Agent 调用 Model, Model 输出 action, action 进入 Environment, Environment 调用 Engine 做物理仿真, Engine 返回新 state, 循环。

这种分层让一个 task (e.g. backflip) 可以无缝切换 simulator backend (Isaac Gym → Isaac Lab → Newton),而 learning algorithm 完全不动。这跟 NVIDIA Isaac Lab (https://arxiv.org/abs/2511.04831) 的设计哲学一致,也跟 CleanRL (https://github.com/vwxyzjn/cleanrl) 的单文件实现形成对比 — MimicKit 选择走 modular + configurable 路线。

## 2. MDP 形式化

paper 给出的标准 MDP objective:

$$J(\pi) = \mathbb{E}_{p(\tau|\pi)}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$$

变量含义:
- $\pi$: policy,把 observation 映射到 action distribution
- $\tau = \{\mathbf{o}_0, \mathbf{a}_0, r_0, \mathbf{o}_1, ..., \mathbf{o}_T\}$: trajectory
- $p(\tau|\pi)$: trajectory 在 policy $\pi$ 下的 likelihood
- $\gamma \in [0,1]$: discount factor,控制 future reward 的权重
- $T$: time horizon
- $r_t = r(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}_{t+1})$: transition reward

interaction loop:
- observation $\mathbf{o}_t$ 来自 partial state $\mathbf{s}_t$
- action $\mathbf{a}_t \sim \pi(\mathbf{a}_t | \mathbf{o}_t)$
- next state $\mathbf{s}_{t+1} \sim p(\mathbf{s}_{t+1} | \mathbf{s}_t, \mathbf{a}_t)$ (environment dynamics)

注意这里区分了 observation $\mathbf{o}_t$ 和 state $\mathbf{s}_t$ — 在 motion imitation 任务里 character 的 full state (joint positions, velocities, contacts) 通常都可观测,所以基本是 MDP 而非 POMDP。但区分开这层抽象对未来扩展到 partial observable 任务 (e.g. sensor noise, sim-to-real) 有意义。

## 3. Done Flag 的精妙设计

paper 里 done flag 有 4 种取值,这是工程师视角的细致设计:

| Flag | 含义 | 学习算法处理 |
|------|------|--------------|
| NULL | episode 未结束 | 继续累积 reward |
| FAIL | 失败终止 (e.g. 摔倒) | 加 terminal penalty |
| SUCC | 成功完成 (e.g. 到达目标) | 加 terminal bonus |
| TIME | 时间限制截断 | bootstrap with value function |

TIME 这个 flag 是关键设计:当 episode 因 horizon 截断时,理论上未来还有 return,所以用 value function $V(s_T)$ 估算 truncated return:

$$\hat{R}_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k} + \gamma^{T-t} V(s_T)$$

这是 finite-horizon MDP emulate infinite-horizon MDP 的标准技巧,在 PPO 实现里通常体现为 `gae_lambda` 计算 last step 时用 bootstrap value 而非 0。参考 OpenAI baselines 的实现: https://github.com/openai/baselines。

## 4. Engine 控制模式

Isaac Gym Engine 支持的控制模式:

- **none**: 仅可视化/调试,不施加控制
- **pos**: target rotation for PD controller (支持 1D revolute + 3D spherical)
- **vel**: target velocity for each joint
- **torque**: 直接指定 joint torque
- **pd_1d**: 仅 1D revolute joints (适合纯 1D 关节的机器人)

PD controller 的标准公式 (Isaac Gym 内部实现):

$$\tau = k_p (\theta_{target} - \theta_{current}) - k_d \dot{\theta}_{current}$$

其中:
- $\tau$: 输出 torque
- $k_p$: position gain (stiffness)
- $k_d$: velocity gain (damping)
- $\theta_{target}$: policy 输出的目标角度
- $\theta_{current}$, $\dot{\theta}_{current}$: 当前角度和角速度

motion imitation 任务里 $k_p$ 和 $k_d$ 是关键超参,典型 humanoid 设 $k_p \approx 1500$, $k_d \approx 50$,per-joint 微调。这个细节在 DeepMimic (https://arxiv.org/abs/1804.02717) 原论文里有详细说明。

## 5. 四种方法对比

### 5.1 DeepMimic (Peng et al. 2018)

论文: https://arxiv.org/abs/1804.02717

核心思想: 显式跟踪 reference motion 的每一帧。Reward function 是手工设计的 multi-term 组合:

$$r_t = w^p r_t^{pose} + w^v r_t^{vel} + w^e r_t^{end} + w^c r_t^{COM}$$

变量:
- $r_t^{pose}$: 关节 pose 差异 (joint 位置 + rotation)
- $r_t^{vel}$: 关节速度差异
- $r_t^{end}$: end-effector 位置差异
- $r_t^{COM}$: center of mass 位置差异
- $w^p, w^v, w^e, w^c$: 各项权重,典型 0.65/0.1/0.15/0.1

配合 Reference State Initialization (RSI): 每个 episode 从 reference motion 随机选一帧作为初始状态。再加 Early Termination (pose error 超过阈值就 FAIL),极大提升 sample efficiency。

优点: 精确跟踪,稳定可靠。缺点: 灵活性差,只能复刻单一 motion,不能 compose 多个 motion 来解决新任务。

### 5.2 AMP (Adversarial Motion Priors, Peng et al. 2021)

论文: https://arxiv.org/abs/2104.02180

核心思想: 借鉴 GAN,用 discriminator 学一个 style reward,而非手工设计。

$$r_t = w^{task} r_t^{task} + w^{style} r_t^{style}$$

style reward 来自 discriminator:

$$r_t^{style} = 1 - \frac{1}{2}\log(1 + \exp(-D(s_{t}, s_{t+1})))$$

discriminator $D$ 训练目标 (类似 GAN):

$$\max_D \mathbb{E}_{(s,s') \sim \pi_{ref}}[\log D(s,s')] + \mathbb{E}_{(s,s') \sim \pi}[\log(1 - D(s,s'))]$$

这里 $(s,s')$ 是 state transition,而不是单帧 state — 这能捕捉 motion 的 temporal dynamics,避免 character 在静止 pose 上"作弊"获得高分。

我联想到:
- 这个思路跟 RLHF 的 reward model 异曲同工 (https://arxiv.org/abs/2203.02155),都是用 discriminator 学习人类偏好,然后作为 reward
- AMP 的 style reward 是 non-zero-sum 的 (policy 探索),容易陷入 local optima,这是 paper 里明确指出的 weakness
- 类比 GAN 的 mode collapse 问题,AMP 也可能出现 "policy 只学到一种典型 motion" 现象

### 5.3 ASE (Adversarial Skill Embeddings, Peng et al. 2022)

论文: https://arxiv.org/abs/2205.07506

ASE 是 hierarchical RL:
- Low-level: 给定 latent $z$,生成 motion skill (adversarial + mutual information objective)
- High-level: 给定 task,从 latent space 选 $z$

low-level 训练目标:

$$r_t = r_t^{GAIL} + r_t^{MI}$$

其中 mutual information reward 鼓励 $z$ 和生成的 motion 有依赖:

$$r_t^{MI} = \log \frac{\pi(a|s,z)}{\pi(a|s)}$$

这是 mutual information $I(z; \tau)$ 的 variational lower bound,实际用 discriminator 估计 $\log \pi(a|s,z)/\pi(a|s)$。

这种思路让我想到:
- DIAYN (https://arxiv.org/abs/1802.06070) — diversity is all you need,发现 skill 的经典工作
- HRL 的 option framework (Sutton, Precup, Singh 1999)
- 最近 LLM-based agent 里的 "skill library" 概念 (e.g. Voyager, https://arxiv.org/abs/2305.16291) — ASE 给了 physics-grounded 的 skill embedding

### 5.4 ADD (Adversarial Differential Discriminator, Zhang et al. 2025)

SIGGRAPH Asia 2025 新工作,这个 paper 的引用里 https://arxiv.org/abs/2511.04831 是 Isaac Lab,但 ADD 本身的论文应该是 https://research.nvidia.com/labs/toronto-ai/ADD/ 或类似链接 (paper 引用 [Zhang et al. 2025] 还在 SIGGRAPH Asia 2025 Conference Papers,arXiv 可能还没公开)。

核心思想: 用 differential discriminator 自动学习 adaptive tracking reward,免去手工设计 reward function 的痛苦。

直觉解释: DeepMimic 的 reward 是手工设固定权重的 pose+vel+end-effector+COM 误差组合。但不同 motion 关注的方面不同 — backflip 关注 global rotation,walk 关注 end-effector 位置。ADD 让 discriminator 学习"对当前 motion 而言什么 tracking 误差重要"。

"differential" 的含义我推测是 discriminator 作用在 $(s - \hat{s})$ 这种差异上,而非直接作用在 state 上 — 类似 WGAN-GP (https://arxiv.org/abs/1704.00028) 的 gradient penalty 思路,但作用在差异空间。

从 Table 1 数据看,ADD 在 challenging motions (Backflip, Spinkick, Cartwheel) 上表现最稳定,这印证了 "adaptive reward" 的价值。

## 6. RL 算法

### 6.1 PPO (Schulman et al. 2017)

论文: https://arxiv.org/abs/1707.06347

PPO 的 clipped objective:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)\right]$$

变量:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$: importance sampling ratio
- $\hat{A}_t$: advantage estimate,通常用 GAE (https://arxiv.org/abs/1506.02438)
- $\epsilon$: clip range,典型 0.2

GAE 计算:
$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

PPO 适合 motion imitation 因为:
1. on-policy,GPU simulator 高吞吐下 sample efficiency 不是 bottleneck (4096 envs)
2. stability 比 TRPO 简单,比 SAC 容易调
3. 配合 RSI + early termination 时 on-policy 的 fresh data 更新价值大

### 6.2 AWR (Advantage-Weighted Regression, Peng et al. 2019)

论文: https://arxiv.org/abs/1910.00177

AWR 是 off-policy 替代,objective:

$$L^{AWR}(\theta) = \mathbb{E}_{(s,a) \sim D}\left[\exp\left(\frac{\hat{A}(s,a)}{\beta}\right) \log \pi_\theta(a|s)\right]$$

变量:
- $D$: replay buffer
- $\hat{A}(s,a)$: advantage estimate
- $\beta$: temperature,控制 advantage 加权的强度

直觉: advantage 大的样本权重大,advantage 小的样本权重小,等价于 weighted maximum likelihood。

AWR 在没有 GPU simulator 的场景 (e.g. real robot) 更合适,因为能利用 off-policy data。

## 7. Motion 数据表示

paper 里详细描述了 motion clip 的存储格式:

```
[root position (3D), root rotation (3D), joint rotations]
```

rotation 用 **3D exponential map** (Grassia 1998, https://www.tandfonline.com/doi/abs/10.1080/10867651.1998.10487493):

exponential map 是 axis-angle 表示 $\mathbf{r} = \theta \hat{\mathbf{n}}$,其中:
- $\hat{\mathbf{n}}$: 旋转轴 (单位向量, 3D)
- $\theta$: 旋转角度
- $\mathbf{r} = (r_x, r_y, r_z)$ 是 3D 向量,长度 = $\theta$,方向 = 旋转轴

优点:
1. 3D 表示,无 gimbal lock
2. 比 quaternion (4D) 少一个维度,连续性好
3. 插值方便

缺点: 接近 0 或 $2\pi$ 时数值不稳,需要 careful 处理

humanoid.xml 的 joint 顺序 (depth-first traversal of kinematic tree):

1. root position (3D)
2. root rotation (3D)
3. abdomen (3D)
4. neck (3D)
5. right_shoulder (3D)
6. right_elbow (1D)
7. left_shoulder (3D)
8. left_elbow (1D)
9. right_hip (3D)
10. right_knee (1D)
11. right_ankle (3D)
12. left_hip (3D)
13. left_knee (1D)
14. left_ankle (3D)

每帧 dimension = 3 + 3 + (3×9 + 1×5) = 3 + 3 + 27 + 5 = 38 dim

这种 depth-first traversal 保证 joint 顺序与 .xml kinematic tree 一致,是 MuJoCo / DeepMimic 的传统。参考 MuJoCo 文档: https://mujoco.readthedocs.io/。

## 8. 实验数据深度分析

Table 1 给出 11 个 motion 的 Position Tracking Error 和 DoF Velocity Tracking Error,5 个 random seed 平均。

### Position Tracking Error 公式

$$e_t^{pos} = \frac{1}{N^{joint}+1}\left(\sum_{j \in joints} \|(\hat{\mathbf{x}}_t^j - \hat{\mathbf{x}}_t^{root}) - (\mathbf{x}_t^j - \mathbf{x}_t^{root})\|_2 + \|\hat{\mathbf{x}}_t^{root} - \mathbf{x}_t^{root}\|_2\right)$$

变量:
- $\mathbf{x}_t^j$: simulated character joint $j$ 在时刻 $t$ 的 3D Cartesian position
- $\hat{\mathbf{x}}_t^j$: reference motion 中 joint $j$ 在时刻 $t$ 的 3D position
- $\mathbf{x}_t^{root}$: simulated character root position
- $\hat{\mathbf{x}}_t^{root}$: reference root position
- $N^{joint}$: 关节数

设计巧思:
1. 第一项算 **相对 root 的 joint 位置** 差异 — 这剥离了 global translation,只关注 character 自身形状
2. 第二项单独算 root position 差异 — 这衡量全局运动 (前进、跳跃距离)
3. 除以 $N^{joint}+1$ (而非 $N^{joint}+2$) — 因为 root 那一项只算一次

### DoF Velocity Tracking Error 公式

$$e_t^{vel} = \frac{1}{N^{joint}+1} \sum_{j \in joints} \|\hat{\dot{\mathbf{q}}}_t^j - \dot{\mathbf{q}}_t^j\|_2$$

变量:
- $\dot{\mathbf{q}}_t^j$: joint $j$ 的 local angular velocity
- $\hat{\dot{\mathbf{q}}}_t^j$: reference 的 local angular velocity

这个 metric 反映 motion smoothness — 即使 pose 对得上,velocity 对不上意味着动作 "颤抖"。

### 数据洞察

按 motion 类型分组分析:

| 类型 | AMP Position | DeepMimic Position | ADD Position | 趋势 |
|------|--------------|-------------------|-------------|------|
| 简单 (Walk, Jog, Crawl) | 0.05-0.13 | **0.009-0.027** | 0.009-0.028 | DeepMimic 完胜 |
| 中等 (Run, Spinkick, Dance A) | 0.06-0.16 | 0.013-0.078 | **0.025-0.165** | 接近 |
| 难 (Backflip, Sideflip, Roll, Cartwheel, GetupFacedown) | 0.09-0.39 | 0.023-0.144 | **0.017-0.152** | ADD 最稳 |

关键观察:
1. **AMP 整体最差**,Run (0.163) 是 DeepMimic (0.013) 的 12 倍,印证 distribution-matching 难以精确复刻
2. **DeepMimic 在简单 motion 上仍然 SOTA**,Walk error 仅 0.009m (9mm)
3. **ADD 在 challenging motion 上稳定性最好**,Backflip error 仅 0.062,DeepMimic 是 0.111
4. **DoF velocity error 普遍 AMP > DeepMimic ≈ ADD**,说明 AMP 动作更 "颤抖"

paper 里提到一个重要细节: 这些实验 **关闭了 pose-error termination** (DeepMimic 标准配置会启用),为了 AMP 公平比较。如果启用,DeepMimic 和 ADD 都会更稳定。这相当于 "刻意制造了 DeepMimic 的劣势场景" 来展示 ADD 的 adaptive 价值。

### Variance 分析

paper 给出 ± std,看 variance:
- AMP Roll: $0.141^{\pm 0.031}$,std/mean = 22%
- DeepMimic Roll: $0.115^{\pm 0.132}$,std/mean = **115%** ! 这个 variance 巨大,说明 DeepMimic 在 Roll 上某些 seed 完全 fail
- ADD Roll: $0.152^{\pm 0.005}$,std/mean = 3.3%,非常稳定

这个数据强烈支持 ADD 的设计初衷: **手工 reward 在 challenging motion 上不稳定,需要 adaptive**。

## 9. Vectorized Environment 与 GPU Simulation

paper 强调 "vectorized environments" 是关键架构决策。Isaac Gym (https://arxiv.org/abs/2108.10470) 把多个 env 放在一个 GPU 仿真里:

- 4096 envs in parallel (paper 默认 `--num_envs 4096`)
- physics simulation 全在 GPU
- observation / action / reward 都是 batched tensor
- PPO update 也 batched

这种架构下 sample throughput 可达 $10^5$+ FPS,使 on-policy PPO 在 motion imitation 任务上完全可行。

Newton (https://github.com/newton-physics/newton) 是更新的开源替代,由 LF Projects (Linux Foundation) 主导,值得长期关注。

## 10. 与相关工作对比

我把 MimicKit 放在更广的生态里看:

### 10.1 RL framework 层
- **RL-Games** (https://github.com/Denys88/rl_games): NVIDIA 内部 PPO 实现,Isaac Gym 默认 RL backend
- **SKRL** (https://github.com/Toni-SM/skrl): Isaac Lab 默认 RL library
- **RSL-RL** (ETH): ANYmal 训练用
- **CleanRL** (https://github.com/vwxyzjn/cleanrl): 单文件实现,教学用
- **Stable-Baselines3**: 通用但慢

MimicKit 定位: **motion imitation 专用**,实现 4 个核心方法,与上述框架互补。

### 10.2 Locomotion 框架
- **Legged Gym** (https://github.com/leggedrobotics/legged_gym): ETH quadruped 训练
- **IsaacLab** (https://github.com/isaac-sim/IsaacLab): NVIDIA 通用 robot learning
- **Walk These Ways** (https://arxiv.org/abs/2212.03452): MIT Cheetah

这些专注于 locomotion (行走/奔跑),没有 motion imitation 的 reference-guided 训练范式。MimicKit 补足这块。

### 10.3 Motion generation 方法 (非 physics-based)
- **MDM** (https://arxiv.org/abs/2209.10915): Motion Diffusion Model
- **MotionBERT** (https://arxiv.org/abs/2210.06599): Transformer motion model
- **MoMask** (https://arxiv.org/abs/2403.11031): Generative masked transformer

这些都是 kinematic motion generation,缺乏物理 grounding,生成的 motion 在物理 simulator 里可能不可行。MimicKit 的 RL-based approach 保证 physical plausibility。

### 10.4 LLM + Robotics
最近 trend 是用 LLM 做 high-level planning:
- **EUREKA** (https://arxiv.org/abs/2310.12931): LLM 自动设计 reward function
- **Dr.GPT**: LLM-based agent
- **RoboGen** (https://arxiv.org/abs/2311.00811): LLM 生成 robot task

这些可以与 MimicKit 结合 — LLM 选 motion 或生成 task spec, MimicKit 训 low-level controller。

## 11. 工程细节与潜在改进

paper 提到的一些工程细节值得展开:

### 11.1 Observation Normalization
Agent 会做 observation normalization,通常是 running mean/std (e.g. VecNormalize)。motion imitation 任务里 joint angle 和 velocity 的 scale 差异巨大 (角度 radian vs velocity rad/s vs position meters),normalization 是必须的。

### 11.2 Action Space
两种主流:
- **continuous action (Gaussian)**: $\mathbf{a}_t \sim \mathcal{N}(\mu(\mathbf{o}_t), \sigma^2)$,policy 输出 mean + log_std
- **learned std vs fixed std**: motion imitation 通常用 learned std with state-independent init

### 11.3 Symmetry Regularization
humanoid 有左右对称性,可以用 symmetry loss 加速学习 (https://arxiv.org/abs/2202.03543),MimicKit 没明说是否支持。

### 11.4 Curriculum Learning
复杂 motion (backflip) 直接训容易 fail。常用:
- 先训简单 motion (jump) 再 fine-tune 到 backflip
- Reference State Initialization (RSI) 让 agent 从 motion 中间开始,降低难度

paper 没专门讨论 curriculum,但 RSI 是 DeepMimic 标配。

## 12. 我对这篇文章的整体评价

**优点**:
1. **modular design** 让 method 切换、simulator 切换、character 切换都很自然,这是 ML system design 的 best practice
2. **统一接口** 把 4 个 motion imitation 方法放在一个 framework 里,便于横向对比
3. **GPU-first** 适应 Isaac Gym/Lab/Newton 这套 NVIDIA 生态
4. **开源完整** (https://github.com/xbpeng/MimicKit),带 pretrained model 和 motion data

**潜在短板**:
1. 目前只支持 NVIDIA GPU simulator (Isaac Gym/Lab/Newton),不支持 MuJoCo CPU 或 Brax
2. 没有 sim-to-real 流程,虽然 framework 提到 "real robots" 但目前只有 simulated experiments
3. 4 个方法都用 PPO 作为 base,缺少 SAC/TD3 等替代
4. 没有 transformer-based policy (MLP-only),与最近 trend (e.g. Decision Transformer) 脱节
5. 对 motion data format 依赖 .pkl + exponential map,与 SMPL/BVH/FBX 生态有 gap

**未来联想**:
- motion generation diffusion model 与 MimicKit 结合: diffusion 生成 reference motion,RL 让 motion physics-plausible
- LLM-driven motion selection: GPT 类模型根据 task 选 motion dataset 中的 clips, MimicKit 训 controller
- World model (e.g. Dreamer) 替代 PPO: 用 world model 学习 dynamics 减少真实 simulation 步数
- Foundation model for control: 一个 large policy 在多个 motion + 多个 morphology 上 pretrain,zero-shot transfer 到新 task

## 13. Key References

- DeepMimic: https://arxiv.org/abs/1804.02717
- AMP: https://arxiv.org/abs/2104.02180
- ASE: https://arxiv.org/abs/2205.07506
- PPO: https://arxiv.org/abs/1707.06347
- GAE: https://arxiv.org/abs/1506.02438
- AWR: https://arxiv.org/abs/1910.00177
- Isaac Gym: https://arxiv.org/abs/2108.10470
- Isaac Lab: https://arxiv.org/abs/2511.04831
- Newton: https://github.com/newton-physics/newton
- MimicKit: https://github.com/xbpeng/MimicKit
- DIAYN: https://arxiv.org/abs/1802.06070
- EUREKA: https://arxiv.org/abs/2310.12931
- Exponential Map (Grassia): https://www.tandfonline.com/doi/abs/10.1080/10867651.1998.10487493
- SMPL: https://arxiv.org/abs/2008.08504 (or original 2015)

## 14. 直觉总结

把 MimicKit 看作 motion imitation 领域的 "PyTorch Lightning" — 不是新方法,而是把已有方法 best practice 沉淀成 framework。四个方法代表了 motion imitation 的两条路线:

**Tracking-based (DeepMimic, ADD)**: 像 supervised learning,有 ground truth reference,精确但僵硬。
**Distribution-matching (AMP, ASE)**: 像 GAN,学 distribution 不学单一样本,灵活但 unstable。

ADD 是这两条路线的 hybrid — 用 adversarial 学 adaptive tracking reward,既有 tracking 的精度又有 distribution-matching 的自适应性。

从 Karpathy 视角看,这个 framework 最大的价值是 **降低 motion imitation 的实验门槛**,让研究者专注 method 创新而非工程实现。类似你做 micrograd (https://github.com/karpathy/micrograd) 让大家理解 backprop 的本意 — MimicKit 让大家能快速尝试 motion imitation 的各种变体。
