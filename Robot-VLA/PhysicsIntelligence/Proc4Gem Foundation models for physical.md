---
source_pdf: Proc4Gem Foundation models for physical.pdf
paper_sha256: f300df22f23e85fe8cbbb4d446e68c052ed4d8c421544a6c4fcc5be519dcafe4
processed_at: '2026-08-06T06:31:02-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Proc4Gem

## 一句话版本

**Google DeepMind 让 Gemini 看了几十万条 MuJoCo 仿真里机器狗推小车的视频,然后这玩意儿就能在真实世界里听人话推车了,而且比专门设计的小模型 baseline 强得多。**

听起来简单,但里面有一堆有意思的 design choice,我一个一个拆给你听。

---

## 问题为啥难

robot learning 圈子一直分裂成两派,互相看不起:

**Physics 派** (MuJoCo/Isaac Gym 那帮人): 物理超准,机器狗能后空翻,能搓魔方。但场景全是空旷 arena,一个 sofa 都没有。你让它去客厅推车,它根本不知道 sofa 是啥,更不知道"推到那个红色的椅子旁边"是啥意思。

**Semantic 派** (Habitat/ProcThor/AI2-THOR 那帮人): 场景超丰富,几千种 furniture,procedurally generated living rooms。但 physics 是 fake 的——robot 跟 object 交互基本就是 kinematic pick-and-place,没有真正的 contact dynamics。你让它推个 trolley,它根本不懂"用 body 持续顶住 trolley 侧面往前走"是什么物理过程。

Proc4Gem 想说的是:**老子两个都要**。既要 MuJoCo 级别的 contact physics,又要 ProcThor 级别的 scene diversity,还要 photorealistic rendering 让 visual grounding 能 transfer 到 real world。

任务选得也很鸡贼:让 quadruped 用 body 推 trolley 到 language 指定的 target。这个 task 你没法拆成 navigation policy + manipulation policy 串起来,因为推车动作从头到尾 body 都贴着 trolley,semantic grounding (知道往哪推) 和 physical control (怎么推) 是耦合在一起的。所以逼着你 end-to-end 学。

参考 ProcThor: https://arxiv.org/abs/2206.06994
参考 Habitat: https://arxiv.org/abs/2310.13724

---

## 整个 pipeline 长啥样

四步流水线,每步都有 reason:

### Step 1: 程序化生成客厅

资产库几千个 furniture,用 Gemini 给每个 asset 写 5 个 level 的 description:
- Level 1: "sofa"
- Level 2: "red sofa"  
- Level 3: "red sofa with six cushions"
- Level 4: "red sofa with six cushions and wooden legs"
- Level 5: 更细...

这 5 个 level 后面用来测 linguistic generalization——你训练时只见过 level 3,部署时人说 level 1 的简单话能不能听懂?人说 level 5 的复杂话能不能听懂?

资产摆放用 hierarchical recipe:先 sample 整个房间,再 sample 子区域 (比如 dining area),再在子区域里 sample 属于这个区域的 furniture。这样生成的 scene 看起来像 living room,而不是一锅乱炖的 asset soup。

Physics 用 MuJoCo 跑 contact dynamics,rendering 用 Unity 跑 photorealistic RGB。这俩解耦是关键 insight——MuJoCo graphics 很丑但 physics 牛,Unity physics 很弱但 graphics 漂亮。各干各擅长的,两边都不妥协。

MuJoCo: https://arxiv.org/abs/2012.07164 (MJX)
Unity: https://unity.com/

### Step 2: 训 privileged expert

Expert policy 用 model-free off-policy RL 训,大概 SAC 之类。Expert 的 input 是 **privileged state**——God view 信息:trolley 精确位姿、target 精确位置、robot proprioception。不需要看图像。

为啥不直接训 vision-based expert? 因为 RL 要几百万 env steps,如果每步都 render RGB,计算爆炸。Privileged expert 网络小、训练快、不需要 rendering。Domain randomization across episodes 让 expert 鲁棒。

RL 的标准 objective:

$$J(\pi_e) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t \, r(s_t, a_t, s_{t+1})\right]$$

变量:
- $\pi_e$ = expert policy
- $s_t$ = privileged state (包含全局 object position 等作弊信息)
- $a_t$ = action,quantized 3D velocity command $(v_x, v_y, \omega_z)$
- $\gamma$ = discount factor,通常 0.99
- $r$ = reward,主要是 trolley-target 距离 + survival bonus

### Step 3: 收 rollout + 蒸馏到 Gemini

Expert 训好后,在 procedurally generated scenes 里 collect 几十万条 successful episode。每条 episode 包含:
- 两个 onboard camera 的 RGB 图像序列
- Target 的 text description (随机从 5 个 level 采样)
- Robot 的 action 序列

切成 length-8 chunks 作为 BC 数据。为啥 8? 后面讲,挺反直觉的。

然后 Gemini 用 next-token prediction loss fine-tune,本质就是 behavior cloning,只不过 policy 是个巨大 VLM:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[\sum_t \log p_\theta(y_t \mid y_{<t}, x)\right]$$

变量:
- $\theta$ = Gemini 参数
- $x$ = multimodal input (images + text)
- $y_t$ = 第 $t$ 个 action token
- $y_{<t}$ = 之前已经生成的 action tokens (autoregressive)
- $\mathcal{D}$ = expert trajectory dataset

Action 怎么塞进 LM 范式?**把连续 3D velocity $(v_x, v_y, \omega_z)$ quantize 成 discrete tokens**。这样 Gemini 直接用 LM head output action,不用改架构。RT-1 也是这思路:https://arxiv.org/abs/2212.06817

### Step 4: Real-world deployment

Gemini 太大不能上 robot,所以搞了个分布式异步 hierarchical control:
- **Low-level node**: robot onboard NUC11TNBv7 上跑,50Hz,执行 low-level locomotion controller(也是 RL 训的,把 velocity command 转 joint torques)
- **High-level node**: off-robot workstation 上跑,2Hz,通过 WiFi query 远程 Gemini
- **Caching**: Gemini 返回的 action 被 cache,每 0.5 秒更新一次给 low-level

Network latency 和 jitter 靠这个 caching 吸收。RT-2 也用类似策略:https://arxiv.org/abs/2307.15818

---

## Baseline 是谁

为了公平对比,他们 reimplement 了 SPOC:https://arxiv.org/abs/2312.02976

SPOC 架构:
- Frozen SigLIP text/image encoder(预训练在 Webli dataset,109 种语言)
- Goal-conditioned transformer encoder + decoder
- Input: 同样的 image + text,但每个 camera 一张图,resized 224×224
- Context length: 8 (match Gemini)
- Train 4 epochs(再多 overfit)

SPOC 是 "frozen encoder + small policy head" 这种典型小模型 embodied AI recipe。对比要回答的问题:**大规模 pretrain 的 foundation model 当 policy 本身,比 frozen-encoder + small-head 组合强多少?**

SigLIP: https://arxiv.org/abs/2303.15343

---

## 实验结果最有意思的几个点

### Simulation 里两个模型差不多

Procedural scene 上 10,000 trials/setting:
- Privileged expert: 68.9%
- SPOC baseline: ~62%
- Gemini: ~62%

Student 能 recover expert 的 90% 性能。两个 student 模型几乎打平,说明在 sim 里 foundation model 的优势没显出来。

Multilingual 几乎 free:意大利语 description 几乎不掉点,因为 SigLIP 和 Gemini 都在 109 种语言上预训练过。

Fixed scene (real world 房间的 3D 扫描重建) 上加了点料:

| 模型 | 不加 3D 扫描资产 | 加 3D 扫描资产 |
|---|---|---|
| SPOC | 59.6% | 62.1% |
| Gemini | 53.0% | **70.0%** |
| Expert | 85.4% | 85.4% |

**Gemini 加了点 visual diversity 直接从 53 涨到 70,baseline 几乎不涨**。这暗示大模型更能 leverage 额外的 visual diversity 来学 generalization,小模型可能已经 overfit 到自己的 capacity 极限。

### Real world 才是分水岭

10 trials/setting,三档难度:
- Easy: robot 一直能看到 trolley 和 target
- Medium: 接近 trolley 时 target 出视野
- Hard: robot 面对 trolley,target 在背后,要 explore

**Hard setting 里 baseline 掉 40%,Gemini 几乎不掉**。

最 striking 的 OOD 测试:
- Giraffe plushie 1.5m 高,完全没见过的 target 类别:baseline **0%**,Gemini **70%**
- 推向人类:Gemini 能 work
- 推向另一个机器狗:Gemini 能 work  
- Trolley 加 10kg 配重:Gemini 仍能 work

**Gemini 的 Internet-scale pretraining + sim2real physics domain randomization 居然能 handle 这么多 OOD**。Baseline 看到 giraffe plushie 完全傻眼,因为 SigLIP embedding 没在这个具体 geometry 上 train 过。

### Figure 8 是最有意思的图

Cumulative success rate vs episode time:
- Short episode:baseline 反而高(反应快)
- Long episode:Gemini 持续上升并超过 baseline
- **Gemini 更 "cautious",recover 能力更强**

这给 deploy 大模型到 robot 一个强论据:短 term 的"快"不是目标,长 term 的"鲁棒"才是。

---

## 几个反直觉的技术 insight

### Context length 8 就够

现代 VLA paper 都吹长 context。Mobility VLA 用 topological graph + 长 context 做 multi-room navigation:https://arxiv.org/abs/2407.07775

但 Proc4Gem 发现 doubling 到 16 不涨点(55% vs 62%)。原因:**这个 task 的 temporal structure 是 short-to-medium range,需要的 memory 就是"最近 8 步看到了啥"**。长 context 的收益要靠 task structure unlock,不是越长约好。如果你要做 multi-room navigation 那种长 horizon task,长 context 才有用。

### Quantized action token 是把 robotics 塞进 LM 范式最干净的方式

连续 velocity $(v_x, v_y, \omega_z)$ 直接 quantize 成 discrete tokens,Gemini 用 LM head output。优点:不改架构,直接复用 LM training pipeline。缺点:action resolution 受 quantization 精度限制。但 2Hz 控制频率下粗粒度 action 够用,因为 low-level controller 做 fine-grained tracking。

### Sim2Real 的 budget 应该花在 rendering fidelity 上

传统 sim2real 花很多 budget 在 system identification (friction, mass, motor delay 的 domain randomization)。Proc4Gem 证明:**photorealistic rendering 的 budget 同样重要,甚至更重要**——因为 VLM 靠 visual features 做 grounding,rendering fidelity 直接决定 grounding 能不能 transfer。

MuJoCo + Unity 解耦很关键。NeRF2Real 那种把真实 scene 扫成 NeRF:https://arxiv.org/abs/2304.07360,realism 高但 scale 不起来,因为只能扫固定场景。Proc4Gem 用 procedural generation + Unity 达成 scale × realism 的 trade-off sweet spot。

---

## 我觉得可疑的地方

1. **2Hz control 对 contact-rich task 够吗?** Trolley pushing 的 contact dynamics 有不少 sub-200ms 的 event,2Hz 可能丢信息。Low-level controller 50Hz,但 high-level plan 2Hz 的话 reactive recovery 受限。这可能是为啥 Gemini 需要 "cautious"——它没法快速 react,只能慢慢 deliberate 避免危险。

2. **Success 定义宽松**:30-60 秒推到即可,没要求 precise pose。更 precise 的 manipulation 可能让 Gemini 的 coarse action space 不够用。

3. **Single task**:只测了 pushing。Multi-task generalization (开抽屉、捡物体、坐 sofa) 没验证。

4. **Expert ceiling 只有 68.9%**:即使 God view privileged info,expert 本身也才 68.9%。Student 62% 已经是 expert 的 90%,distillation 效率很高,但绝对值低。Reward shaping 或更长 training 能不能 push 到 95%? 论文没探讨。

5. **Sim2Real gap 的 credit assignment 没做 ablation**:real world Gemini 在 easy setting 也低于 expert 在 sim fixed scene 85.4%,这个 gap 到底来自 physics mismatch (trolley mass, friction) 还是 visual mismatch? 不知道。

---

## 这篇 paper 的 bigger picture

Proc4Gem 在生态里的位置很有意思:

- **RT-1/RT-2**: real robot data + large model → 直接 deploy
  - https://arxiv.org/abs/2212.06817
  - https://arxiv.org/abs/2307.15818
- **Octo**: open-source generalist policy,跨 embodiment real data
  - https://arxiv.org/abs/2405.12213
- **RoboCat**: self-improving foundation agent
  - https://arxiv.org/abs/2306.11706
- **Proc4Gem**: pure sim data + foundation model → 直接 deploy

Proc4Gem 证明了一个可能性:**如果 sim 足够 diverse + realistic,real data 甚至不是必须的**。如果这个 scaling law 成立,sim data 可以"compute into data",foundation model 的 robot data bottleneck 可以用更多 compute 解决。

但前提是四个东西都 scale up:
- Procedural generation diversity 覆盖 real scene distribution
- Physics fidelity 让 contact dynamics 可信
- Rendering fidelity 让 VLM 的 visual grounding 能 transfer
- Foundation model 本身够 strong (Gemini 级别)

这四样**都**要做足,sim2real 才能跳过 real data collection。门槛非常高。

Future work:
1. Longer context 做 multi-room navigation (Proc4Gem 只测了一个 room 内)
2. Diffusion/video generation augment scene diversity,GenSim2 在做:https://arxiv.org/abs/2410.02598
3. 直接 RL fine-tune LLM 不通过 expert + BC。POLIFORMER 是 on-policy RL + transformer 尝试:https://arxiv.org/abs/2406.20083
4. Gemini Robotics 这个同期工作也值得看:https://arxiv.org/abs/2503.05487

---

## 我的 takeaway

Proc4Gem 给我的最大 intuition:**sim2real 的瓶颈从来不是简单的"sim 不够真",而是"sim 不够 diverse"和"sim data 没用对 model"**。

- 单纯追 physics fidelity,你得到 Unitree A1 在空 arena 里跳得超漂亮但看到 sofa 就懵
- 单纯追 semantic diversity,你得到 ProcThor 里 robot 能 navigate 但 push 不动 paper cup
- 单纯追 small policy 在 sim 里 overfit,你得到一个在 MuJoCo 里 solve task 但 deploy 到 real 就死

Proc4Gem 说:三件事都做,且都做 sufficient scale,再用一个真正 strong 的 foundation model glue 起来——就能跳过 real data。这是个非常 "AI native" 的思路,把 sim 当 data factory 而不是 verification tool,把 foundation model 当 universal student 而不是 task-specific network。

如果 sim 真的能成为 robot foundation model 的"互联网级数据源",robotics 会迎来一波类似 LLM 在 2020 年那种 paradigm shift。我现在不确定的是这个 scaling law 到底成不成立——paper 只在一个 task 上 demo,没 scale 到 full task suite。但作为 proof of concept,它已经够 striking 了。

---

# Proc4Gem: 用Procedural Generation给Foundation Model注入Physical Agency

## 1. 论文核心thesis

Proc4Gem想做的事情,一句话概括:**能不能只用simulation数据,把Gemini这种multimodal foundation model直接fine-tune成contact-rich的whole-body robot policy,且zero-shot transfer到real world。**

Robot learning领域long-standing一个dichotomy:
- 一派做whole-body control (locomotion, dexterous manipulation),用high-fidelity physics simulator (MuJoCo, PyBullet, Isaac Gym),但场景semantic diversity极低,通常就是empty arena或者简单terrain
- 一派做embodied AI navigation/manipulation,用Habitat/ProcThor/Ai2thor这种scene-rich simulator,但physics被simplified成kinematic pick-and-place,没有真正contact dynamics

Proc4Gem想force两者merge到一起:既要semantic diversity (procedurally-generated living rooms with thousands of assets),又要contact-rich physics (MuJoCo + photorealistic Unity rendering)。然后让一个VLM直接吃这些simulation trajectory,通过behavior cloning变成policy。

任务选得很巧妙:让quadruped (Barkour) 用body推trolley到language指定的target object。这个task**无法用modular solution拆成navigation+manipulation**,因为push动作本身需要body和trolley有持续contact,同时还要semantic grounding到target object,所以必须end-to-end。

Project page: https://sites.google.com/view/proc4gem

---

## 2. System Pipeline拆解

整个pipeline分四步,每一步都有non-trivial的设计选择:

### Step 1: Procedural Scene Generation

**Asset dataset**: 几千个furniture assets (sofas, chairs, dining tables, coffee tables)。用Gemini给每个asset生成5个levels of detail的natural language descriptions。这5个level从简单("sofa")到详细("red sofa with six cushions"),用来测试linguistic generalization。

**Hierarchical placement recipe**: 不是random scatter assets,而是按语义hierarchy:
1. 先sample room整体
2. 在room内sample (可能nested) areas,例如"dining area"
3. 在每个area内sample属于该category的asset instances

这种hierarchical采样让生成的scene有"living room"的semantic structure,而不是混乱的asset堆。

**Physics + Rendering split**: 关键design choice是**physics在MuJoCo跑,rendering在Unity跑**。MuJoCo擅长contact dynamics但不擅长photorealistic graphics;Unity擅长graphics但physics很弱。两者解耦,各做各的强项。Unity用GPU multi-view rendering,overhead极小。

参考的ProcThor: https://arxiv.org/abs/2206.06994
RoboCasa: https://arxiv.org/abs/2406.02523

### Step 2: Privileged Expert RL

Expert policy用**model-free off-policy RL**训练(论文没明说具体算法,大概率是SAC或类似actor-critic),它的input是privileged state——也就是God view信息:目标object的精确位置、robot精确pose、trolley精确pose等。

Expert的RL objective标准形式:

$$J(\pi_e) = \mathbb{E}_{s_0 \sim \rho_0, \, a_t \sim \pi_e(\cdot|s_t), \, s_{t+1} \sim P(\cdot|s_t, a_t)} \left[ \sum_{t=0}^{T} \gamma^t \, r(s_t, a_t, s_{t+1}) \right]$$

变量解释:
- $\pi_e$: expert policy
- $s_t$: privileged state at time $t$,包含robot proprioception + 全局object positions
- $a_t$: action (quantized 3D velocity command: $v_x, v_y, \omega_z$)
- $\rho_0$: initial state distribution (从procedural scene sampler来的)
- $P(\cdot|s_t, a_t)$: MuJoCo dynamics transition
- $\gamma$: discount factor (typically 0.99)
- $r$: reward (与trolley-target距离相关 + survival bonus + time penalty)

Expert网络small,因为input是structured state vector,不需要rendering。这非常关键,因为RL training要几百万env steps,如果每步都render RGB,计算成本爆炸。

Domain randomization across episodes用来robustify expert。

### Step 3: Rollout Collection + Distillation

Expert训好后,在procedurally-generated scenes里collect "hundreds of thousands of successful episodes"。每个episode包含:
- 文本target descriptions (从5个level随机采样)
- 两台onboard camera的RGB images
- Robot actions

把episode切成**length-8 trajectory chunks**作为BC训练数据。为什么是8?这是key insight:这个task是short-to-medium range navigation + contact manipulation,不需要超长memory。论文实验也证明doubling到16 context不涨点。

Gemini用**next-token prediction loss** fine-tune,这本质上就是behavior cloning,只不过policy是个大规模VLM:

$$\mathcal{L}_{BC}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x) \right]$$

变量解释:
- $\theta$: Gemini的fine-tune parameters
- $x$: multimodal context = image sequence from both cameras + text description
- $y = (y_1, ..., y_T)$: tokenized action sequence (3D velocity command quantized成tokens)
- $\mathcal{D}$: expert trajectory dataset (length-8 chunks)
- $y_{<t}$: previously generated action tokens (autoregressive)

Action space是quantized 3D planar velocity: forward-backward ($v_x$), right-left side-step ($v_y$), yaw turn ($\omega_z$)。Quantization让action变成discrete tokens,这样Gemini这种autoregressive LM可以直接output,不用改架构。

### Step 4: Deployment (分布式异步hierarchical control)

这是real-world system engineering的核心。Gemini太大无法on-robot跑,所以:
- **Low-level node**: 跑在robot onboard NUC11TNBv7,50Hz,执行low-level locomotion controller(也是RL训练的,把velocity command转成joint torques)
- **High-level node**: 跑在off-robot workstation,2Hz,通过WiFi query remote Gemini instance
- **Caching策略**: Gemini返回的action被cache住,每0.5秒 (2Hz)更新一次给low-level controller

Network latency和jitter就靠这个caching吸收。这是deploy大模型到robot的实战技巧,RT-2也用类似策略:https://arxiv.org/abs/2307.15818

---

## 3. Baseline (SPOC) 对照

为了公平比较,他们reimplement了SPOC架构作为baseline。SPOC架构:

- **Encoder**: frozen SigLIP text/image encoder (pre-trained on Webli dataset, 109 languages)
- **Policy head**: goal-conditioned transformer encoder + transformer decoder
- **Input**: 同样的images和text,但每个camera一个image,resized到224×224
- **Context length**: 8 (match Gemini)
- **Training**: 4 epochs (再多就overfit,sim performance开始decline)

SPOC原paper: https://arxiv.org/abs/2312.02976
SigLIP: https://arxiv.org/abs/2303.15343

注意SPOC的"frozen encoder + small policy head"是典型小模型做embodied AI的recipe。Proc4Gem的对比要回答:**大规模pretrain的foundation model,作为policy本身,比这种frozen-encoder + small-head的组合强多少?**

---

## 4. 实验结果详解

### Simulation results (procedural scenes, 10,000 trials/setting)

| 模型 | Train desc | OOD desc | Italian desc |
|---|---|---|---|
| Privileged RL expert | 68.9% | 68.9% | 68.9% |
| SPOC baseline | ~62% | 略降 | 几乎无降 |
| Gemini | ~62% | 略降 | 几乎无降 |

关键观察:
- **Student能recover大部分expert performance** (62% vs 68.9%)
- **Multilingual capability几乎free** (Italian描述几乎不掉点),因为SigLIP和Gemini都在109种语言上pretrain过
- **Description verbosity level对performance影响小**,说明模型能handle不同程度的language specification

### Simulation fixed scene (real-world room的3D扫描重建版)

| 模型 | No 3D-scanned assets | With 3D-scanned assets |
|---|---|---|
| SPOC baseline | 59.6% ± 0.49% | 62.1% ± 0.49% |
| Gemini | 53.0% ± 0.50% | 70.0% ± 0.46% |
| Privileged expert | 85.4% | 85.4% |

非常interesting:**加入3D-scanned real assets到training里,Gemini从53%涨到70%,baseline只从59.6涨到62.1**。这暗示Gemini这种大模型更能leverage extra visual diversity来学习visual generalization。Fixed scene比procedural scenes整体success rate高,因为人工设计的scene更clean、更少clutter。

### Real-world results (核心battlefield)

10 trials/setting。三档难度:
- **Easy**: 机器人能持续看到trolley和target
- **Medium**: 接近trolley时target会出视野
- **Hard**: 机器人面对trolley,target在背后,需要explore

在hard settings,**baseline平均drop 40%成功率,Gemini基本不掉**。Figure 8最illuminating:
- 早期(short episode),baseline成功率反而高(它反应快)
- 但后期,Gemini的cumulative success rate持续上升并超过baseline
- **Gemini更"cautious",recover能力更强**

最extreme的OOD测试:
- **Giraffe plushie (1.5m高,完全没见过的target类别)**: baseline **0%**, Gemini **70%**
- 推向人类: Gemini能work
- 推向another robot dog: Gemini能work
- Trolley加10kg weight: Gemini仍能work

这是整篇paper最striking的结果:**Gemini的Internet-scale pretraining + sim2real physics domain randomization,居然能handle这么多OOD**。Baseline看到giraffe plushie完全不知道往哪推,因为SigLIP embedding可能没在这个具体geometry上被train过。

---

## 5. 几个关键技术insights

### Insight 1: Context length 8够用

这个有点反直觉。现代VLA paper都吹长context (Mobility VLA用topological graph + 长context做multi-room navigation: https://arxiv.org/abs/2407.07775)。但Proc4Gem发现doubling到16不涨点(55% vs 62%)。原因:这个task的temporal structure相对short-horizon,需要的memory就是"最近8步看到了什么"。**长context的收益要靠task structure来unlock**,而不是越长约好。

### Insight 2: Gemini的"cautiousness" vs baseline的"reactiveness"

Figure 8显示Gemini在short episode上成功率低,但long-term success更高。这个pattern很像我之前讲过的"slow thinking vs fast thinking"。Gemini这种大model推理latency大,但每次action更deliberative;baseline小model反应快,但在real world的edge case上recover不了。这给deploy大模型到robot一个强论据:**短term的"快"不是目标,长term的"鲁棒"才是**。

### Insight 3: Sim2Real的"Realism budget"应该花在哪

传统sim2real花很多budget在system identification(domain randomization on friction, mass, motor delay等)。Proc4Gem证明:**photorealistic rendering的budget同样重要,甚至更重要**——因为vision-language model靠visual features做grounding,rendering fidelity直接决定grounding能否transfer。

MuJoCo做physics + Unity做rendering的decoupling很关键。NeRF2Real那种把真实scene扫成NeRF然后训policy的方法:https://arxiv.org/abs/2304.07360,虽然realism高但scale不起来,因为只能扫固定场景。Proc4Gem用procedural generation + Unity达成scale × realism的trade-off sweet spot。

### Insight 4: VLM pretraining在physical task上的迁移

Gemini本身从未见过robot action,从未推过trolley,但pretrain的visual+language grounding足够strong,以至于在sim data上fine-tune一下就能work。这印证了RT-2的发现:https://arxiv.org/abs/2307.15818, web-scale VLM pretraining的knowledge确实能transfer到physical action,只需要少量robot data(这里是sim data)做alignment。

### Insight 5: Quantized action作为token

把连续velocity command quantize成discrete tokens,这样Gemini直接用LM head output action。这是把robotics塞进LM范式最干净的方式之一。RT-1也这么做:https://arxiv.org/abs/2212.06817。缺点是action resolution受quantization精度限制,但2Hz控制频率下粗粒度action够用,因为low-level controller做fine-grained tracking。

---

## 6. 我对Limitations的几个疑问

虽然paper很clean,几个可能concern:

1. **2Hz control频率对contact-rich task够吗?** Trolley pushing的contact dynamics其实有不少sub-200ms的event,2Hz可能丢不少信息。Low-level controller虽然50Hz,但high-level plan在2Hz的话,reactive recovery能力受限。这可能是为什么Gemini需要"cautious"——它没法快速react,只能慢慢deliberate避免危险。

2. **Trolley推到target的成功定义比较宽松**: 30-60秒内推到即可,没有要求precise pose。更precise的manipulation (如准确把trolley对齐到sofa前面)可能让Gemini的coarse action space不够用。

3. **Single task**: 只测了pushing。能不能handle diverse task suite (open drawer, pick up object, sit on sofa)未验证。Multi-task generalization是VLA的holy grail,这里没碰。

4. **Expert performance ceiling**: 68.9% in procedural scene。这意味着即使用God view privileged info,expert本身也只在68.9%。Student的62%已经是expert的90%——distillation效率很高。但绝对值低。Reward shaping或更长的training能否push expert到95%?论文没探讨。

5. **Sim2Real gap的credit assignment**: 在real world Gemini在easy setting也低于expert在sim的固定场景85.4%,这个gap到底来自physics mismatch (trolley mass, friction) 还是visual mismatch?没做ablation。

---

## 7. 更大的picture和future direction

Proc4Gem在生态里的位置:

**Foundation model + sim data路线**:
- RT-1/RT-2: real robot data + large model → 直接deploy
- Octo: open-source generalist policy,跨embodiment real data
- RoboCat: self-improving foundation agent
- **Proc4Gem**: pure sim data + foundation model → 直接deploy

Proc4Gem证明了一个可能性:**如果sim足够diverse + realistic,real data甚至不是必须的**。如果这个scaling law成立,sim data可以"compute into data",foundation model的robot data bottleneck可以simply用更多compute解决。

但前提是:
- Procedural generation的diversity要够大 (覆盖real scene distribution)
- Physics fidelity要够高 (contact dynamics可信)
- Rendering fidelity要够高 (VLM的visual grounding能transfer)
- Foundation model本身要够strong (Gemini级别)

这四样东西**都**要scale up,sim2real才能跳过real data collection。这其实是非常高的门槛。

Future work (paper里也提了):
1. **Longer context**: leverage Gemini的长context window做multi-room navigation (Proc4Gem只测了一个room内)
2. **Diffusion / video generation augmentation**: 用generative model进一步augment sim scene多样性,GenSim2已经在做: https://arxiv.org/abs/2410.02598
3. **直接RL fine-tune LLM**: 不通过expert + BC,直接在sim里对Gemini做RL fine-tune。这是最ambitious但最有潜力的方向,因为BC有imitation gap,RL能在sim里直接optimize task reward。POLIFORMER是on-policy RL + transformer的尝试: https://arxiv.org/abs/2406.20083

---

## 8. 我的intuition总结

Proc4Gem给我的最大intuition是:**sim2real的瓶颈从来不是"sim不够真"那么简单,而是"sim不够diverse"和"sim data没用对model"**。

- 单纯追求physics fidelity,你得到Unitree A1在empty arena里跳得超漂亮但看到sofa就懵。
- 单纯追求semantic diversity,你得到ProcThor里robot能navigate但push不动一个paper cup。
- 单纯追求small policy在sim里overfit,你得到一个在MuJoCo里solve task但deploy到real就死。

Proc4Gem说:三件事都做,且都做sufficient scale,再用一个真正strong的foundation model把它们glue起来——就能跳过real data。这是个非常"AI native"的思路,把sim当成data factory而不是verification tool,把foundation model当成universal student而不是task-specific network。

我会很期待看到这个approach scale到full-house task suite和multi-embodiment。如果sim真的能成为robot foundation model的"互联网级数据源",robotics community会迎来一波类似LLM在2020年那种paradigm shift。

参考链接汇总:
- Project page: https://sites.google.com/view/proc4gem
- ProcThor (procedural generation): https://arxiv.org/abs/2206.06994
- SPOC (baseline): https://arxiv.org/abs/2312.02976
- RT-2 (VLA sim2real思路): https://arxiv.org/abs/2307.15818
- RT-1 (quantized action token): https://arxiv.org/abs/2212.06817
- RoboCasa (近期类似工作): https://arxiv.org/abs/2406.02523
- Barkour robot: https://arxiv.org/abs/2305.14654
- MuJoCo: https://arxiv.org/abs/2012.07164 (mujoco mjx paper)
- POLIFORMER (transformer + on-policy RL): https://arxiv.org/abs/2406.20083
- Mobility VLA (long context VLA): https://arxiv.org/abs/2407.07775
- Gemini Robotics (DeepMind同期工作): https://arxiv.org/abs/2503.05487
- Octo (open generalist): https://arxiv.org/abs/2405.12213
