---
source_pdf: Counterfactual rewards promote collective transport.pdf
paper_sha256: de7b1e506cfb9def614759a99665e2461551318cacbe50a0560c70f24980a9f8
processed_at: '2026-08-03T17:39:27-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

一群 6 微米大的小破球,被激光像遥控车一样一个个控制着,然后用强化学习教会它们像蚂蚁一样齐心协力推一根大棍子。

---

## 一、这些小东西到底长什么样

想象一个 **头发丝的十分之一** 那么大的 silica 小球,一半涂上 carbon 黑壳,叫 **Janus particle**(两面神粒子,因为一半黑一半白)。把它扔进一种特殊的液体(lutidine + water 的混合液),这种液体有个怪脾气:温度一到 34°C 就会 **自己分层**(demixing,像油水分离)。

然后你用一束 **532 nm 绿色激光** 打在小球的 carbon 那一半上,carbon 吸光发热,局部温度超过 34°C,小球周围液体就开始分层。因为只有一半被加热,分层是不对称的,这种不对称就 **推着小球往前走**。原理有点像你放个炮仗在水里,一边炸一边推着东西跑。

参考:https://doi.org/10.1038/s41598-017-14216-9

---

## 二、怎么"遥控"这么多小球

难点在这:你要同时遥控 200 个小球,但只有一束激光。解法很 brute force — **用 acousto-optical deflector 让激光束在 100 kHz 速度扫描**,轮流照每个小球。因为小球被照之后的"推力"能维持大概 100 ms,所以只要你 1 秒里照到它几次,它就觉得自己是被"持续推着"的。这就像你妈一秒叫你十次"快写作业",你主观体验就是"一直在被催"。

每个小球的动作就 4 种,非常有限:

| 动作 | 怎么实现 |
|---|---|
| **Forward** | 激光照 cap 后端,功率 2.7 µW,直走 |
| **Stationary** | 同位置,弱光 0.3 µW,just 原地待着 |
| **Left turn** | 两个激光 spot 不对称打 cap 两侧,差热产生 torque |
| **Right turn** | 同上,反方向 |

每 10 秒做一次决策,期间小球最多走 6 µm(一个身位)或转 36°。

参考 setup 细节:https://www.clemens-bechinger.com/

---

## 三、小球的"视野"有多惨

你想象自己就是这个 6 µm 的小球。你 **没有眼睛**,只能感知自己前方 180°(transport task 扩到 360°)的"拥挤程度"。视野被切成 5 个 cone(扇区),每个 cone 36°,对每个 cone 你只知道一个数字 — 这个扇区里其他东西(其他小球或 rod)离你有多近,越近数字越大。

公式很简单:

$$
o_i^l = \min\left(\sum_{j \neq i}^{M_l} \frac{\sigma}{|\mathbf{r}_{ij}|}, 1\right)
$$

变量解释:
- $o_i^l$:小球 $i$ 在第 $l$ 个 cone 里感知到的"有多挤"
- $\sigma$:小球直径 6 µm,当 length scale
- $\mathbf{r}_{ij}$:小球 $i$ 到小球 $j$ 的距离
- $M_l$:这个 cone 里有多少其他物体
- $\min(\cdot, 1)$:怕太近爆掉,clip 在 1

也就是说,**每个小球自己合成了一张 10 像素的"图像"**(rotation task 5 cones × 2 species = 10 个数;transport task 10 cones × 3 species = 30 个数),就这破输入,要它学会推一根 100 µm 的棒子。

---

## 四、为什么不能简单给所有人同一个 reward

最 naive 的做法:每一步看 rod 转了多少角度,把数字原样发给所有 agents 作为 reward。这叫 **team reward**。

问题来了:

**Lazy agent 问题**:某个小球躲一边睡觉不干活,但只要其他小球把 rod 转了,它一样拿 reward。那它学不到"我得去帮忙"这个信号。

**Credit assignment 问题**:某个小球看到自己 reward 涨了,但搞不清是因为自己刚才那个 turn 帮上忙了,还是远处的另一个小球刚 push 了一下。它只看得到自己 5 个 cone 里那点东西,根本不知道全系统发生了啥。

结果就是 reward signal 噪声极大,training 收敛慢,最终 policy 也烂。

---

## 五、Counterfactual reward — 这篇 paper 的真正招数

idea 简单粗暴:**给每个小球单独算一份 reward,算法是"如果你不在的话,大家能干多少活"。**

具体来说,每个 timestep 做这件事:

1. 实际跑实验,记录所有 N 个小球一起干活,rod 转了 $\omega_t$ 度 — 这是 $P_t$
2. 在电脑里跑一个粗糙的物理 simulation,把小球 $i$ 删掉,看其他 N-1 个小球会让 rod 转多少 — 这是 $P_{t \backslash i}^v$
3. 小球 $i$ 的 reward 就是两者的差:

$$
r_{t,i} = \beta\left(P_t - \underline{P_{t \backslash i}^v}\right)
$$

- $r_{t,i}$:小球 $i$ 这一步的专属 reward
- $P_t$:真实实验里 rod 转的角度
- $\underline{P_{t \backslash i}^v}$:虚拟环境里去掉 $i$ 之后 rod 转的角度(做了一点 rescaling,见下)
- $\beta$:就是 scale 一下,让 reward 落在 0-1 量级

**直觉**:如果删掉你这个小球,大家还是把活干得一样好,那你就是 lazy agent,reward ≈ 0。如果删掉你,rod 突然不转了,那你就是核心功臣,reward 大。

---

## 六、两个聪明的 trick 让这招真的能用

### Trick 1:Rescaling 抵消 model 不准

虚拟 simulation 用的是非常粗糙的 overdamped Langevin 模型,显然跟真实实验对不上。比如 simulation 说 rod 能转 5°,实验里只转了 3°,那 $P_{t \backslash i}^v$ 也被高估了。

解法是用一个 ratio 抵消 systematic bias:

$$
\underline{P_{t \backslash i}^v} = P_{t \backslash i}^v \cdot \frac{P_t}{P_t^v}
$$

- $P_t$:实验里所有 N 个小球让 rod 转的角度(真实值)
- $P_t^v$:simulation 里所有 N 个小球让 rod 转的角度(虚拟值)
- $P_{t \backslash i}^v$:simulation 里去掉小球 $i$ 后 rod 转的角度

这个 $\frac{P_t}{P_t^v}$ 就是个"校准系数",把 simulation 系统性高估或低估的部分抵消掉。跟 importance sampling 一个思路。

### Trick 2:Re-simulation 不加 noise

这点反直觉。你可能觉得"要仿真真实环境,得加 Brownian noise 啊"。但想想:

- 实验:$P_t = P_t^{\text{真实信号}} + \eta_{\text{实验噪声}}$
- Re-sim 加 noise:$P_{t \backslash i}^v = P_{t \backslash i}^{v,\text{信号}} + \eta_{\text{仿真噪声}}$
- 两者相减:$r_{t,i} = (\text{signal diff}) + (\eta_{\text{实验}} - \eta_{\text{仿真}})$

两份独立噪声相减,variance 翻倍,信号被淹没。所以 re-sim 用 **deterministic model**,只让实验那边贡献 noise。这是个非常实用的工程 trick — 用 model 的 *shape* 不用它的 *细节*。

参考 difference rewards 原始 paper:https://doi.org/10.1142/S0219525901000257

---

## 七、任务 1:让小球们转一根 rod

Rod 是个 6×6×100 µm 的椭圆柱,3D 打印出来的。一个小球根本推不动,要 ~30 个小球合作。

**Global performance**(每次实验直接测):

$$
P_t = |\theta_t - \theta_{t-1}|
$$

就是这一步 rod 转了多少度,取绝对值(swarm 自己选顺时针或逆时针)。

**训练过程特别有意思,分三个阶段**:

| Episode | 现象 | 本质 |
|---|---|---|
| 0-10 | 小球乱逛,偶尔撞上 rod 拿点 reward | Random policy 探索 |
| 10-20 | 小球学会"往 rod 靠",平均距离下降,但 $\omega \approx 0$ | 学到 local signal:"靠近 rod = 好" |
| 20-40 | 小球学会聚集在 rod **两端**,从 opposite sides push,$\omega$ 突然跳到饱和 | 学到 collective signal:"位置 + 协调" |

最关键是 episode 20 之后 — 小球们自发学会 **"两端聚集"**。如果你 naive 想,会以为小球均匀分布在 rod 周围推最有用,但其实不,只有 **推 rod 两端** 才能产生 torque。RL 自己发现了这个 lever arm physics。

paper 还定义了一个 **geometric torque** 量化这个:

$$
T_{\text{geom}} = \left|\sum_i^N \mathbf{r}_i \times \mathbf{u}_i\right|
$$

- $\mathbf{r}_i$:小球 $i$ 相对 rod 中心的位置
- $\mathbf{u}_i$:小球 $i$ 的朝向
- Cross product 衡量"切向力分量",只有小球在 rod 端点且朝切向推才贡献大

---

## 八、任务 2:把 rod 搬到指定位置

更难。要同时控制 rod 的三个自由度:x 平移、y 平移、旋转。而且每 episode 开始,target 位置随机放,rod 朝向也随机。

**Performance function** 换成 potential-based:

$$
V = \frac{1}{60}\sum_{k=1}^{60} d_k
$$

- 把 rod 切成 60 段,target 也切成 60 段
- $d_k$:rod 第 $k$ 段到 target 第 $k$ 段的距离
- $V$ 是平均距离,越小越好

每步 reward 是 $V$ 的 **负变化**:

$$
P_t = V_{t-1} - V_t
$$

rod 靠近 target 时 $P_t > 0$,远离时 $P_t < 0$。用 potential-based reward 是因为 Ng et al. 1999 证明这种 reward shaping **不改变 optimal policy**,不会让 agent 学出 spurious behavior。

成功把 rod 完全送进 target 的话,所有人额外拿 +500(稀疏 reward trick)。

### 三个自发涌现的子策略

训练完之后小球们自己悟出三种推法:

| 子策略 | 怎么推 | 效率 |
|---|---|---|
| **Transverse transport** | 全堆在 rod 一侧整体推 | 高,接触面大 |
| **Rotation** | 推 rod 两端 opposite sides | 高,lever arm 长 |
| **Longitudinal transport** | 沿 rod 表面 *滑动*,靠 friction 推 | 低,但 rod 太细没别的办法 |

最妙的发现:小球们自发选 **"先 transverse 推到 target 中心,再 rotate 对齐"** 的顺序,而不是反过来。如果反过来,得用低效的 longitudinal transport,慢得多。

更绝的:如果初始 rod 已经跟 target 共线了,小球们会 **故意先把 rod 推歪**,这样就可以走 transverse 路线,最终反而更快。这是 RL 自己发现的 physical shortcut,跟 AlphaGo 的 "move 37" 一个味道 — 看起来"错"的操作其实是最优的。

参考:https://doi.org/10.1021/nn003685k (potential-based reward shaping)

---

## 九、Training 算法细节

paper 用 **PPO**(Proximal Policy Optimization),Schulman 2017 那个。所有 agents **共享一个 policy network**,就是同一个神经网络。

网络架构极小:
- Input:10 或 30 个 scalar(rotation task 10,transport task 30)
- Hidden:3 层,32 / 16 / 16 节点
- Output:4 维 softmax over 4 actions

整个网络才几千参数,跟大模型完全不是一个量级,但够用 — 因为 task 复杂度有限,而且 parameter sharing 让每个 agent 的经验都贡献到同一个 network,等于 batch size 被放大了 N 倍。

PPO 的 loss 不细展开,标准做法:

$$
L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

- $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$:新旧 policy 在这个 action 上的概率比
- $A_t$:advantage,这里直接用 counterfactual reward 减 baseline
- $\epsilon$:clip 范围,防止 policy 更新太激进

paper 没用 centralized critic(MAPPO 那种),因为 counterfactual reward 已经把 variance 压下去了。

参考:
- PPO: https://arxiv.org/abs/1707.06347
- MAPPO: https://arxiv.org/abs/2103.01955
- IPPO: https://arxiv.org/abs/2011.09533

---

## 十、Robustness 实验数据

### 10.1 故障容忍(Fig. 6C)

每一步随机选一些 robot,把它们 policy 选的 action 替换成 random action(随机抽,这样 malfunctioning robot 不会自己漂走)。结果:

| Malfunction % | 归一化性能 |
|---|---|
| 0% | 1.0 |
| 20% | ~1.0 |
| 50% | ~0.30 |
| 70%+ | ~0 |

20% 失效完全没影响 — policy 学到的 strategy 自带 redundancy,swarm 不依赖每个 agent 都对。

### 10.2 Scalability(Fig. 6D)

在 N≈35 训练,然后改 N 测试:

| N | 性能 |
|---|---|
| 9 | ~0.5 |
| 20 | ~1.0 |
| 35 (训练 size) | 1.0 (peak) |
| 75 (2x) | <0.5 |

小 group 没劲推不动,大 group 互相打架 — 这是 active matter physics 里 MIPS(Motility-Induced Phase Separation)那个临界密度味道,太稀没力,太密互相阻碍。

但 paper 在 simulation 里对大 N *继续训练*,性能又能上去(Fig. S12)。说明 scalability 限制来自 *fixed policy*,不是算法本身。

参考 MIPS review:https://doi.org/10.1146/annurev-conmatphys-020911-125106

---

## 十一、Multi-object demo(Fig. 7)

最后炫技:把 swarm 分成 3 个 team,每个 team 推一根 rod,3 根 rod 可以 **独立选旋转方向**。

这个 demo 看似简单,其实是 **distributed control 相对 global field 的本质优势**。如果用 magnetic field 控制整个 swarm,所有 rod 必须同时同方向转,因为所有 agent 感受同一个 field。这里 laser 独立寻址,3 个 team 完全解耦。

对 **lab-on-a-chip** 应用来说,这意味着可以在一个芯片上同时跑多个独立 assembly 任务,是 parallelism 的根本来源。

---

## 十二、为什么 Karpathy 你应该 care

### 12.1 极低维 input 也能学出复杂策略

10-30 个 scalar 输入,几千参数的网络,学出"找 rod 末端 → 协调 push → 自然分阶段 → 故意 misalign 走捷径"。这印证你常说的大多数情况下 bottleneck 在 reward signal 与 exploration,不在网络 capacity。Counterfactual reward 干的就是把 reward signal 弄干净这件事。

### 12.2 是 "physical world backprop" 的干净 demo

整个 setup 等价于在 *真实物理环境* 里 backprop gradient。laser 控制 = forward pass,counterfactual re-sim = virtual gradient 计算,PPO update = backward pass。这跟你在 Tesla 看过的端到端 driving 有点像 — physical world 提供 ground truth signal,RL 算法负责 update policy。只不过这里 world 是真实的微米尺度流体物理。

### 12.3 是 "emergent behavior 超越 hand-designed rule" 的 example

Active matter physics 几十年用 Vicsek alignment rule 之类的 *fixed rules* 研究 collective motion。这里 RL 学出的 *end-clustering*、*sub-strategy switching*、*intentional misalignment* 这些行为,Vicsek model 根本不可能产生。换句话说,RL 在自动发现新的 collective behavior,而 active matter physics 可以拿来当 *研究对象* — 这两个领域应该多 cross-pollinate。

### 12.4 跟 emergent language 的潜在连接

paper 没让 agents 通信,但 rod 本身就是 *communication medium*。agent A 在 rod 一端 push,rod 旋转会让 agent B 在另一端的位置 / 接触状态变化,B 通过自己的 cone 感知到 — 这就是 **stigmergy**(蚂蚁信息素通信的物理版)。

如果加一个 explicit communication channel,swarm 能学到什么 emergent language?这个 setup 是测试 *emergent language in physical MARL* 的天然 testbed,带真实 noise,比纯 sim 强。

参考:
- Emergent language survey: https://arxiv.org/abs/2106.02457
- TarMAC: https://arxiv.org/abs/1806.07257
- CommNet: https://arxiv.org/abs/1605.07736

---

## 十三、一些可能的 follow-up

| 方向 | 想法 |
|---|---|
| **Latent-conditioned policy** | 让一个 network 根据 latent code 切换不同 role,可能学到 division of labor |
| **Differentiable physics for counterfactual** | 用可微物理做 re-sim,gradient 可以 backprop 到 model 本身,可能更准 |
| **Multi-step counterfactual** | 现在只外推 1 步,推 k 步可以 capture 更长 horizon 贡献,但 model 要更准 |
| **Hierarchical RL** | High-level 选 sub-strategy,low-level 控 motion,可能更 sample efficient |
| **Curriculum across swarm sizes** | 直接 train 在 N=10 到 N=200,得到 size-invariant policy |
| **3D holographic control** | Paper 末尾提了但没做,用 holographic optical tweezers 扩到 3D |
| **On-chip 部署小 ANN** | 配合 Miskin group 的 CMOS microrobots,真正 untethered autonomous microbot |
| **Inverse RL from ant data** | 用真实蚂蚁集体搬运数据做 IRL,看 reward function 长什么样 |

参考:
- Miskin microrobots: https://www.nature.com/articles/s41586-020-2626-9
- Differentiable physics survey: https://arxiv.org/abs/2402.12916
- Ant collective transport review: https://doi.org/10.1007/s00018-020-0359-8

---

## 十四、 criticisms

1. **Transport task 不是 end-to-end 实验训练** — 只在 simulation 训完 zero-shot 部署,真正复杂任务还没 prove out in situ training
2. **Counterfactual 是 naive 的** — 删掉 agent $i$ 后,其他 agent 应该 *重新对自己的新 obs 做反应*,paper 假设它们动作不变。COMA 的 gradient-based marginal 更 principled
3. **Laser 控制是 external 的** — 真正 autonomous microbot 还得 onboard 拿 sensor 自己决策,目前 setup 仍有 external infrastructure 依赖
4. **只有 2D** — 3D 是 future work,但 holographic optics 复杂度大很多
5. **Collision model 仍然 black box** — paper 把这个 *当 feature* 说"MARL 能 robust 对待",但没法量化 worst case

---

## 十五、最终直觉

这篇 paper 把 **active matter physics** 与 **multi-agent RL** 这两个之前不搭界的领域真正 bridge 起来了。技术 trick 不复杂(counterfactual reward + one-step re-sim + rescaling),但 *实验上跑通* 这件事本身就是 contribution — 微米尺度的 noise、collision、phoretic interaction 比 macro robot 复杂得多。

更深层意义:这是 *物理实验驱动的 learned collective behavior science* 的起点。Active matter 几十年用 hand-designed rules,这里换成 ANN,RL 在物理环境里 backprop,自动发现新规则。这条路走下去,可以做 *实验探索 collective behavior 空间*,而不只是 *理论建模*。

代码开源:https://github.com/vheuthe/microbot_rl

如果你想继续聊某一块 — 比如 counterfactual reward 与 COMA / QMIX 的具体数学对比、active matter 在 RL 里能产生什么 novel phase、或 sim-to-real gap 在 microbot 上的具体来源 — 我可以再展开。

---

# Paper 深度解析:Counterfactual Rewards Promote Collective Transport

## 一、Paper 大局观:这篇工作在解决什么问题

这篇 paper 来自 Clemens Bechinger group (Konstanz) 与 ICTP 的 Emanuele Panizon 合作,一作 Veit-Lorenz Heuthe。文章的核心 ambition 是把 swarm robotics 从 macroscopic 尺度推到 **micrometer 尺度**,同时保持 **每个 microrobot 独立可控** 这个 property。

之前 microscale 的 swarm 工作(e.g. magnetic colloids、bacteria swarms)几乎都用一个 **global field** 控制 — 一个 magnetic field、electric field 或 acoustic field 作用于整个 workspace,所有 agents 感受到的 field 是共享的,local interactions 只限于简单的 dipole-dipole 之类。这种做法 collective motion patterns 数量有限,无法让 swarm 同时在不同区域做不同任务。

Bechinger group 这次的方案:用 **laser + acousto-optical deflector** 在 100 kHz 速度扫描,可以独立寻址 ~200 个 Janus microparticles。每个 particle 像一个独立的"bumper car",自己感知局部信息、自己决策。但这就引出 paper 真正的 technical contribution — **怎么训练这么多 agents?**

核心 answer:**Multi-Agent Reinforcement Learning (MARL) + Counterfactual Rewards (CR)** 解决 credit assignment 问题,在 physical 实验里端到端训练 swarm 转动一个 rod。

参考链接:
- Bechinger group: https://www.clemens-bechinger.com/
- Active particles review (Rev. Mod. Phys. 2016, ref 28): https://doi.org/10.1103/RevModPhys.88.045006
- Difference rewards 原始 paper (Wolpert & Tumer, ref 39): https://doi.org/10.1142/S0219525901000257

---

## 二、Physical System 细节:Janus particles 怎么被 laser 推

### 2.1 Particle 设计与 propulsion 机制

- 主体:**6 µm silica sphere**(直径 σ = 6 µm)
- 一半表面镀 **80 nm carbon cap**(Janus 结构)
- 悬浮在 **lutidine-water binary mixture**(26.8 wt% lutidine),温度保持 28.2 °C,接近 lower critical solution temperature (LCSST) ≈ 34 °C
- Laser (532 nm) 照射 carbon cap → cap 局部升温超过 critical point → 流体局部 demixing → 形成对称性破缺的 solute gradient → **self-diffusiophoretic propulsion**

公式上,一个 Janus swimmer 的速度 $v \propto \nabla c$ (solute concentration gradient),方向沿 cap 法线指向 uncapped 一侧。这种机制属于 **active matter** 范畴,参考 Bechinger 的 Nat. Commun. 2020 (ref 54)。

### 2.2 Action space 与 laser 配置

每个 robot 有 **4 个离散 actions**:forward / left turn / right turn / stationary。每个 action 持续 **10 seconds**,通过不同的 laser spot 强度与位置实现:

| Action | Laser 配置 | Power |
|---|---|---|
| Forward | 单 spot 在 cap 后端(uncapped 一侧) | 2.7 µW |
| Stationary | 同位置,但弱 spot 抑制 rotational diffusion | 0.3 µW |
| Right turn | 两 spot 不对称分布在 cap 两侧,差热 → torque | 1.1 + 1.7 µW |
| Left turn | 两 spot 反过来 | 1.1 + 1.7 µW |

Forward action 中 robot 走过 6 µm (一个直径) 后自动切换为 stationary,补完 10 秒;rotation 走过 π/5 = 36° (恰好一个 cone 角度) 后也切回 stationary。这个 36° 不是巧合,是为了让 turn action 刚好跨越一个 detection cone。

### 2.3 噪声特性

- **Brownian diffusion coefficient** $D \approx 0.03\,\mu\text{m/s}$ → 一个 action 期间扩散位移 ~$0.1\sigma$ (10% 直径)
- 推进速度 ~$0.6\,\mu\text{m/s}$ → Péclet number $\text{Pe} = v\sigma/D \approx 100$ (以 robot 直径),或 $\text{Pe} \approx 2000$ (以 rod 长度 100 µm 计)
- Rotational diffusion FWHM ~30°/action
- **Collision-induced noise** 才是大头:robot 与 robot 或 robot 与 rod 撞上后会 "sticky"(由 demixing bubble 的 phoretic attraction 引起),orientation 大幅抖动(见 Fig. 3F-I)

直觉上,这就相当于让一辆尺寸 6 µm 的"碰碰车"在糖浆里推一根 100 µm 的"电线杆",而且你自己只能看到前面 5 个扇区里的车流密度。控制 challenge 主要不是单 robot 的 noise,而是 collision 后的 chaos + 不可建模的 phoretic interaction。

参考链接:
- Janus particle propulsion (Gomez-Solano et al. 2017, ref 34): https://doi.org/10.1038/s41598-017-14216-9
- Active colloids review (Liebchen & Mukhopadyay, ref 41): https://doi.org/10.1088/1361-648X/abee2c

---

## 三、MARL 框架:state / action / policy 三件套

### 3.1 State input — 5 cones + inverse distance weighting

每个 robot 把自己的视野(180° 或 360°)分成 5 个(或 10 个)cones,每个 cone 36°。对每个 cone $l$ 与每个"species"(其他 robots / rod / target)计算一个 scalar state input:

$$
o_i^l(s) = \min\left( \sum_{j \neq i}^{M_l} \frac{\sigma}{|\mathbf{r}_{ij}|},\, 1 \right) \quad \text{(Eq. 2)}
$$

变量解释:
- $o_i^l(s)$:robot $i$ 在 cone $l$ 上感知到的 occupancy 强度
- $\sigma$:robot 直径 = 6 µm(作为 length scale)
- $\mathbf{r}_{ij}$:robot $i$ 到 robot $j$ 的位移矢量,$|\mathbf{r}_{ij}|$ 是距离
- $M_l$:落在 cone $l$ 内的同类物体数量
- $\min(\cdot, 1)$:clip 到 [0, 1] 防止近距离爆掉

这是经典的 **inverse distance weighting**(Viscido et al. 2002, ref 55),粗略近似"被感知物体占我视野多少"。

直觉:这就相当于 robot 自己合成了一张 **10 像素的"图像"**(5 cones × 2 species,transport task 变 10 cones × 3 species = 30 scalars)。Karpathy 你应该 appreciate 这种 "perception as a tiny retina" 的极简设计 — **input 维度极低,信息密度极高**。

### 3.2 Policy network 架构

ANN 架构(paper 里写得比较吝啬,在 Supplementary 才有):
- **Input layer**:10 scalars(rotation task)或 30 scalars(transport task)
- **Hidden layers**:3 层,分别 32 / 16 / 16 节点
- **Output layer**:4 维 softmax over actions
- 激活函数未明示,推测 ReLU / tanh

所有 agents **共享同一个 policy**(parameter sharing,典型的 swarm MARL 设计),所以训练时所有 agents 的梯度一起更新这同一个网络。这不仅节省参数,更重要的是让 swarm 真的是 homogeneous。

### 3.3 训练算法

paper 用的是 **PPO**(Proximal Policy Optimization,Schulman 2017, ref 60),配合 parameter sharing。PPO 在 MARL 里被广泛使用,例如 MAPPO (Yu et al. 2022) 也证明 PPO + centralized critic 在 cooperative MARL 里很强。

但这里关键点:paper **没有用 centralized critic**(没给 actor 额外的 global state)。每个 agent 只用自己的 local observation 算 policy,这与 IPPO (independent PPO) 思路接近,但用了 counterfactual reward 替代 centralized critic 来 reduce variance。

参考链接:
- PPO (Schulman et al. 2017): https://arxiv.org/abs/1707.06347
- MAPPO (Yu et al. 2022): https://arxiv.org/abs/2103.01955
- IPPO (de Witt et al. 2020): https://arxiv.org/abs/2011.09533

---

## 四、Counterfactual Rewards — paper 的真正核心

### 4.1 Lazy agent problem

在 fully cooperative MARL(Dec-POMDP)里,最 naive 的 reward 方案是给所有 agents 同一个 **team reward** $r_{t,i} = P_t$ for all $i$。问题:
- Lazy agent:某 agent 不干活也拿一样 reward → 学习信号混乱
- High variance:某 agent 的 reward 大幅波动,但波动主要来自 *别人* 的 actions 而非自己 → gradient noise 极大
- Credit assignment:agent 只看到局部 obs,无法判断"这次 reward 涨了是因为我做了什么,还是别人做了什么"

### 4.2 Counterfactual reward 的核心 idea

经典 difference rewards(Wolpert & Tumer 2001, ref 39)定义:

$$
D_i(s) = R(s) - R(s_{\backslash i})
$$

其中 $R(s)$ 是实际 team reward,$R(s_{\backslash i})$ 是"如果 agent $i$ 不在,系统的 reward"。差值就是 $i$ 的"边际贡献"。

paper 把这个 idea 用在 sequential decision-making 里,**每个 timestep** 都计算一次:

$$
r_{t,i} = \beta\left(P_t - \underline{P_{t\backslash i}^v}\right) \quad \text{(Eq. 7)}
$$

变量解释:
- $r_{t,i}$:agent $i$ 在 timestep $t$ 的 individual reward
- $P_t$:timestep $t$ 的 *实际* global performance(从实验测得)
- $P_{t\backslash i}^v$:虚拟环境里去掉 agent $i$ 后外推一步的 performance
- $\underline{P_{t\backslash i}^v}$:rescaled 后的版本(见下面 Eq. 6)
- $\beta$:常数 scaling factor,把 reward 大致压到 [0, 1]

### 4.3 Rescaling — 解决 model 不准的问题

paper 用一个非常粗糙的 overdamped Langevin simulation 做 re-simulation,显然不会和实验完美一致。为了消除 systematic bias,他们用所有 agents 在虚拟环境的 performance $P_t^v$ 与实际 performance $P_t$ 的比值做 rescaling:

$$
\underline{P_{t\backslash i}^v} = P_{t\backslash i}^v \cdot \frac{P_t}{P_t^v} \quad \text{(Eq. 6)}
$$

直觉:如果 model 系统性高估 50% performance,那 $P_{t\backslash i}^v$ 也会被高估 50%,$\frac{P_t}{P_t^v}$ 这个 ratio 抵消掉这个 systematic error。这是一个非常聪明的 domain adaptation trick — 类比 RL 里的 importance sampling ratio。

### 4.4 Re-simulation 不加 noise — 反直觉但合理

paper 强调:re-simulation 时**不加 thermal noise**。直觉上你可能觉得"应该模拟真实环境啊"。但思考一下:

- 实验:$P_t = P_t^{\text{deterministic}} + \eta_{\text{exp}}$
- Re-sim with noise:$P_{t\backslash i}^v = P_{t\backslash i}^{v,\text{det}} + \eta_{\text{sim}}$
- Difference:$P_t - P_{t\backslash i}^v = (P_t^{\text{det}} - P_{t\backslash i}^{v,\text{det}}) + (\eta_{\text{exp}} - \eta_{\text{sim}})$

两份独立 noise 相减会让 variance **翻倍**(假设 noise 同量级)。所以 re-sim 用 deterministic model,只让实验那边的 noise 贡献 variance — 信号 noise ratio 更好。

这是一个 **用 model 的 *shape* 而不是 *细节* 的典型例子**,和 model-based RL 里 **short-horizon rollout** 的思路完全一致:短 horizon 用 approximate model 没事,只要 sign 和 magnitude 大致对。

### 4.5 Computational cost

每个 timestep 要做 $N$ 次 re-simulation($N$ 是 agent 数量,30-200)。但每次 re-sim 只外推一步,所以总开销是 $O(N \times \text{single step cost})$。在 200 agents 时单 step 也就 ms 级,完全可承受。这与 COMA (Foerster et al. 2018) 那种需要 centralized critic 计算 marginal 的方法相比,实现简单很多 — paper 显然故意避开了 critic 网络。

参考链接:
- COMA (Foerster et al. 2018): https://arxiv.org/abs/1705.08926
- Counterfactual baselines in MARL: https://arxiv.org/abs/1812.01858

---

## 五、两个 Task 的细节

### 5.1 Task 1:Rod rotation

**Global performance**:

$$
P_t = |\theta_t - \theta_{t-1}| = |\omega_t| \quad \text{(Eq. 3)}
$$

$\theta_t$ 是 rod 在 timestep $t$ 的 orientation(0° 到 180°,因为 ellipsoid 双对称)。取绝对值意味着 swarm 自由选 clockwise 或 counter-clockwise,训练时随机锁定一个方向。

**Geometric torque** 用来评估 collective coordination:

$$
T_{\text{geom}} = \left| \sum_i^N \mathbf{r}_i \times \mathbf{u}_i \right| \quad \text{(Eq. 1)}
$$

- $\mathbf{r}_i$:robot $i$ 相对 rod 中心的位置矢量
- $\mathbf{u}_i$:robot $i$ 的朝向单位矢量
- Cross product 衡量"切向力分量",只有当 robot 在 rod 两端、推力沿切向时才贡献大

**训练动力学**(Fig. 4):
- Episode 0-10:random policy,robot 乱逛,偶尔撞上 rod 拿到 reward
- Episode 10-20:robot 学会往 rod 靠(平均距离下降),但 angular velocity 还是 ~0(没协调)
- Episode 20-40:robot 学会聚集在 rod **两端**对称分布,从 opposite sides 推 → $T_{\text{geom}}$ 突然上升,$\omega$ 跳到饱和值
- 后续:torque 继续上升但 $\omega$ 饱和 — 因为 rod 已经达到 fluid drag 决定的 terminal angular velocity

直觉:agents 先学"靠近 rod"(local reward signal 强),再学"在 rod 上找好位置"(reward signal 弱但 informative),最后学"协调两端 push"(最微弱但决定性的信号)。这种 **curriculum 自然涌现** 是 RL 在物理实验里最迷人的现象之一。

### 5.2 Task 2:Targeted transport

更复杂:需要同时控制 rod 的 3 个 DOF(translation x、y、rotation θ)。Vision field 扩展到 360°(10 cones)加上 target 作为第三个 species。

**Performance function** 基于 potential:

$$
V = \frac{1}{60} \sum_{k=1}^{60} d_k \quad \text{(Eq. 4)}
$$

- 把 rod 分成 60 个 virtual segments
- 把 target region 也分成 60 个对应 segments
- $d_k$:rod segment $k$ 与 target segment $k$ 的 pairwise distance(考虑两种 rod 朝向,取 lower)

为什么用 potential-based reward?引用 Ng et al. 1999 (ref 56) 的经典结论:**potential-based reward shaping 不改变 optimal policy**,所以加 $V_{t-1} - V_t$ 不会引入 spurious behavior。

$$
P_t = V_{t-1} - V_t \quad \text{(Eq. 5)}
$$

这是 negative change in potential,rod 靠近 target 时为正。

**Success bonus**:如果整根 rod 完全进入 target region($V < 8\,\mu$m),所有 agents 拿 +500 final reward。这是 episodic sparse reward 的经典设计。

### 5.3 自然涌现的 sub-strategies

trained swarm 自发学到 3 个 sub-strategies(Fig. 5C):

| Mode | 怎么做 | 效率 |
|---|---|---|
| **Transverse transport** | Robots 聚在 rod 一侧整体推 | 高 (大接触面积) |
| **Rotation** | Robots 推 rod 两端,opposite sides | 高 (大 lever arm) |
| **Longitudinal transport** | Robots 沿 rod 表面 *滑动*,靠摩擦力推 | 低 (但 slender rod 没别的办法) |

**最有趣的发现**:swarm 自动选择 **"先 transverse transport 再 rotate"** 顺序,而不是"先 rotate 再 longitudinal push"。这是 emergent planning,因为 longitudinal transport 效率太低。在 Fig. 5F 对比实验里,如果初始 rod 与 target 共线,robots 反而会 *故意 misalign* rod,以便走 transverse 路线,效率更高。这是 RL 意外发现 physical shortcut 的典型案例 — 类似 AlphaGo 的"move 37"。

直觉:这种 **任务自动分阶段** 是大规模 RL 在 physical system 里最有用的 emergent property,因为它把连续控制问题在时间上 sparse 化了。

### 5.4 Sim-to-real zero-shot transfer

Transport task 在实验里端到端训练时间不可承受,所以 paper 在 virtual env 训练后 **zero-shot 部署** 到实验。成功率 >90% within 3000 actions per episode。

虚拟环境用了粗糙的 overdamped Langevin 模型,显然无法捕捉 sticky collision 之类复杂交互,但 zero-shot 居然能 work。这印证了 RL policy 的 **closure property** — policy 学到的是 "robust action selection under partial observation",对 model mismatch 也有一定 tolerance。

paper 还提到:在实验中继续 fine-tune 并不进一步提升性能(Fig. S10),说明 sim-to-real gap 不是 online fine-tune 能解决的 — 需要更好的 model,但 model 不存在。

参考链接:
- Sim-to-real in RL: https://arxiv.org/abs/2104.09407
- Domain randomization: https://arxiv.org/abs/1703.06907

---

## 六、Robustness & Scalability 实验数据

### 6.1 Malfunction tolerance(Fig. 6C)

每步随机选一部分 robot,把它们的 action 替换成 random action(随机而非固定 set,避免 malfunctioning robots 自己漂走)。归一化 angular velocity:

| Malfunction % | Normalized Performance |
|---|---|
| 0% | 1.0 |
| 20% | ~1.0 (无显著下降) |
| 50% | ~0.30 |
| 70%+ | ~0 (失效,robot 全部扩散走) |

直觉:20% 失效还能保持性能,因为 RL policy 自带 redundancy — swarm 不需要每个 agent 都正确执行。这与 *Dec-POMDP robust solution* 的理论预测一致。

### 6.2 Scalability(Fig. 6D)

Trained at N≈35,tested across N = 9 to 75:

| N | Normalized Performance |
|---|---|
| 9 | ~0.5 |
| 20 | ~1.0 |
| 35 (train size) | 1.0 (peak) |
| 75 (2x+) | <0.5 |

小 group 性能下降因为 total torque 不够;大 group 性能下降因为 robot 在 rod 周围形成 dense cluster,反而 *互相阻碍* — collision-induced chaos 在大 N 下 dominate。

但 paper 在 simulation 里 *with further training* on larger N,性能可以继续上升(Fig. S12)。说明 scalability issue 来自 *fixed policy*,不是 *algorithm* 本身。

直觉:swarm 有"最佳密度"。太稀没力,太密互相打架。这与 active matter physics 里 MIPS (Motility-Induced Phase Separation) 的临界密度现象异曲同工 — 参考 Cates & Tailleur review。

参考链接:
- MIPS review (Cates & Tailleur 2015): https://doi.org/10.1146/annurev-conmatphys-020911-125106

---

## 七、Multi-object manipulation demo(Fig. 7)

Paper 最后 demo:swarm 分成多个 teams,每个 team 控制一根 rod,各 rod 可以独立选 rotation 方向。**不需要改任何 framework**,只是 software switch — 这就是 distributed control 相对 global field 的真正优势。在 magnetic field 控制下,所有 rods 必须同时同方向转,这里完全解耦。

这种 parallelism 对 **lab-on-a-chip** 应用至关重要 — 你想在同一个 chip 上同时做多个 assembly 任务,必须独立控制。

---

## 八、一些 Karpathy 可能会感兴趣的 intuition

### 8.1 极低维 input 也能 work 的启示

10-30 scalars 输入,3 层 hidden(32/16/16),竟然能学出"找 rod 末端 → 协调 push → 控制方向"这种 non-trivial 策略。这印证了你经常提的观点:**大多数情况下,数据效率瓶颈不在网络 capacity,而在 reward signal 与 exploration**。Counterfactual reward 解决的就是 reward signal 这边 — 给每个 agent 一个 informative、低 variance 的 gradient。

### 8.2 与 emergent communication / language game 的关系

这篇 paper 没让 agents 之间通信 — 但 rod 本身就是"通信媒介":agent A 在 rod 一端 push,rod 旋转会让 agent B 在另一端的位置 / 接触状态变化,B 感知到变化就间接知道 A 在 push。这就是 **stigmergy**(蚂蚁信息素通信的物理 analog)。

如果加上 explicit communication channel, swarm 能学到什么 language?Karpathy 你一定会想到 *emergent language in MARL*(Lazaridou et al. 2018, Mordatch & Abbeel 2018)。这篇 paper 提供了一个 *physical testbed* 测试这些 idea — 比纯 simulation 实验多一重 physical noise 的 robustness 检验。

参考链接:
- Emergent language (Lazaridou et al. 2018): https://arxiv.org/abs/1705.11192
- Multi-agent emergent communication survey: https://arxiv.org/abs/2106.02457

### 8.3 Counterfactual reward ≈ 单步 model-based rollout

Counterfactual reward 本质上是 **one-step model-based rollout** 用作 reward shaping。这与 model-based RL 里 *Dreamer* 系列(Hafner et al.)的 multi-step rollout 区别只是 horizon = 1。理论上可以扩展到 multi-step counterfactual,但要 model 更准。Karpathy 在你 RL 课程里讲 model-based RL 时提到 *short-horizon rollout 才靠谱* — 这篇 paper 就是把这个 intuition 用在 reward signal 上。

参考链接:
- Dreamer V3 (Hafner et al. 2023): https://arxiv.org/abs/2304.10187
- MuZero: https://arxiv.org/abs/1911.08265

### 8.4 与 Janus particles / active matter physics 的桥

这篇 paper 实际上把 **active matter physics** 和 **MARL** 这两个领域 bridge 起来了。Active matter 长期关注 emergent collective motion(如 Vicsek model),但 *策略是 fixed*(简单 alignment rule)。这里让策略 *learned*,相当于把 Vicsek 的 alignment rule 换成 ANN,在物理环境里 backprop。这是 *physics-informed RL* 的一个范式。

理论上,你可以用这套 framework 探索:什么样的 collective behavior 是 Vicsek-type rule 学不到但 RL 能学到的?比如 paper 里的 *end-clustering rotation* — Vicsek 默认会均匀围绕 rod 推,但 RL 学到"聚集两端"更高效。这是 *learned collective behavior 超越 hand-designed rule* 的 concrete example。

### 8.5 一些 hallucinated 联想

- **Micro-RL + Miskin 的 mass-manufactured microrobots**(ref 22, Cornell 2020):Miskin group 用 CMOS 做出 ~100 µm 大小、可电子集成的 walking microrobots。理论上可以 *on-chip 部署* 一个 32-16-16 的小 ANN,让每个 microbot 独立运行 policy,完全 untethered。这是 paper 末尾 discussion 提到的方向,但还没人做出来。
- **Microbot 自组装 + DNA origami**:如果 cargo 不是 rod 而是 *DNA origami 结构*,swarm 可以作为可编程的 "active assembler" — 这种方向在 *Sitti group* (ref 29, Alapan et al. 2019) 有雏形,但 *控制策略是 hard-coded*。换成 MARL 可以学更复杂 assembly sequence。
- **Biological swarm + RL**:已经有 *C. elegans optogenetic control*(ref 23),如果能用 MARL 训练 worm 群体做 collective task,会很有意思 — 不过 worm 不是 homogeneous agent。
- **可微物理 + MARL**:paper 用粗糙 Langevin model 做 counterfactual。如果用 *differentiable physics*(Brakkee et al. 或 PhiNet 类)做 counterfactual,gradient 可以直接 backprop 到 model,可能进一步提升 credit assignment 精度。但 paper 故意只用 model 的 *shape*,detailed gradient 反而可能 overfit。
- **GPU 加速 sim-to-real**:200 agents × 100 kHz laser × 几小时训练,这个实验产生 *海量物理数据*。这些数据本身可以训练一个 *learned physics simulator*,然后再用 learned simulator 做 counterfactual — meta-level。这篇 paper 没做但完全可行。
- **与 Emergent Abilities of LLMs 的类比**:swarm 在 N=35 训练,N=20 仍 work,N=9 还有一半性能 — 这种 *"scaling 工作但不在极小 N"* 的现象与 LLM emergent abilities 论文(Michaud et al. 2023)的 *task density per parameter* 论点异曲同工。

参考链接:
- Miskin microrobots (Nature 2020): https://www.nature.com/articles/s41586-020-2626-9
- Sitti shape-encoded assembly (ref 29): https://doi.org/10.1038/s41563-019-0507-4
- Differentiable physics survey: https://arxiv.org/abs/2402.12916
- Emergent abilities of LLMs: https://arxiv.org/abs/2206.07682

---

## 九、批评性思考 & 可能的 follow-up

### 9.1 局限

1. **End-to-end training 只在 rotation task 做** — transport 靠 sim-to-real。这意味着对真正复杂任务,*in situ training* 还没 prove-out。
2. **共享 policy** 意味着 agents 不能 specialize。但 ant colony 里不同 ants 有不同 roles。如果用 mixture-of-policies 或 *latent-conditioned policy*,可能学到更复杂 division of labor。
3. **Counterfactual reward 假设 single agent 移除后其他 agents 行为不变**。这是 *naive counterfactual*,严格来说应该让其他 agents 重新对自己的新 obs 做反应。COMA 的 gradient-based marginal 计算更 principled,但需要 critic。
4. **Laser 控制是 *external* 的**,真正的 microbot 不能 onboard 拿 laser 指自己。这意味着 deployment 阶段仍然需要 external infrastructure。这与 paper discussion 里 "future autonomous microbot" 的愿景有 gap。
5. **只有 2D**:rod 在 2D 平面运动,3D 还没做。Paper 末尾提 holographic methods 可以扩展,但 no proof-of-concept。
6. **Collision model 仍然是 black box** — 反而 paper 把这个 *当 feature*,说"MARL 能 robust 对待"。但没法量化 *最坏情况* 性能。

### 9.2 可能的 follow-up direction

- **Latent-conditioned policy** 让 swarm 学到 heterogeneous roles
- **Communication channel** 让 agents 显式 message-passing(像 TarMAC、CommNet)
- **Curriculum learning across swarm sizes** — 直接 train 在 N=10 到 N=200 范围,得到 size-invariant policy
- **Differentiable physics simulator + counterfactual** — 让 re-sim 用 learned differentiable model,可以 *amortize* counterfactual computation
- **Hierarchical RL** — high-level policy 选 sub-strategy(rotation / transverse / longitudinal),low-level policy 控制 motion
- **Real-time adaptation / meta-RL** — train 一个 policy 能 *快速 adapt* 到新 task(reward function 在 inference 时切换),这其实是 paper 想做的 *generality* 的更强版本
- **Inverse RL from ant trajectories** — 用真实 ant collective transport 数据做 IRL,看看 reward function 长什么样,与 learned policy 对比

参考链接:
- TarMAC: https://arxiv.org/abs/1806.07257
- CommNet: https://arxiv.org/abs/1605.07736
- Ant collective transport review: https://doi.org/10.1007/s00018-020-0359-8

---

## 十、关键公式汇总表

| # | 公式 | 用途 |
|---|---|---|
| Eq. 1 | $T_{\text{geom}} = \|\sum_i \mathbf{r}_i \times \mathbf{u}_i\|$ | 几何 torque 衡量 collective coordination |
| Eq. 2 | $o_i^l = \min(\sum_j \sigma/\|\mathbf{r}_{ij}\|, 1)$ | Inverse distance weighted state input |
| Eq. 3 | $P_t = \|\theta_t - \theta_{t-1}\|$ | Rotation task performance |
| Eq. 4 | $V = \frac{1}{60}\sum_{k=1}^{60} d_k$ | Transport task potential |
| Eq. 5 | $P_t = V_{t-1} - V_t$ | Transport task performance (negative change in potential) |
| Eq. 6 | $\underline{P_{t\backslash i}^v} = P_{t\backslash i}^v \cdot \frac{P_t}{P_t^v}$ | Counterfactual rescaling (systematic bias 抵消) |
| Eq. 7 | $r_{t,i} = \beta(P_t - \underline{P_{t\backslash i}^v})$ | Individual counterfactual reward |

---

## 十一、结论

这篇 paper 是 microscale swarm robotics 与 MARL 真正成功 *双向打通* 的 first demonstration。技术贡献:**counterfactual reward 用 single-step re-simulation 实现,unbiased、低 variance、不需要 critic**。系统贡献:**laser-controlled Janus particles 能形成独立可控 microbot swarm,可以训练、可以 robust、可以 scale、可以并行多任务**。

更深层的意义:这篇 paper 在 *物理环境* 里 backprop,把 *RL objective* 与 *active matter physics* 耦合 — 这条 path 通往 *实验驱动的 learned collective behavior science*。Active matter 之前一直用 hand-designed rules(Vicsek、quorum sensing),这篇 paper 给出 *automated rule discovery via RL* 的范式。

Karpathy 你应该会 appreciate 这个 setup 的极简之美:6 µm particle × 200 agents × 10 scalars obs × 4 actions × tiny ANN,却 emergent 出 *rod-end clustering*、*sub-strategy decomposition*、*intentional misalignment for efficiency* 这些 sophisticated behaviors。这是 *physical world backprop* 的一个干净 demo,有 *embodied AI* 的味道,也有 *ALife (artificial life)* 的味道。

代码:https://github.com/vheuthe/microbot_rl
DOI:https://doi.org/10.5281/zenodo.13380516

---

如果你想深入聊任何一块 — 比如 counterfactual reward 与 COMA / QMIX 的详细对比、active matter physics 在 MARL 里能产生什么 novel collective phase、或 sim-to-real 在 microbot 上的具体 gap 与解法 — 我可以继续展开。
