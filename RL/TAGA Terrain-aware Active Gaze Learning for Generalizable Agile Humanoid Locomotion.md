---
source_pdf: TAGA Terrain-aware Active Gaze Learning for Generalizable Agile Humanoid
  Locomotion.pdf
paper_sha256: fa79adb16c19e206326217a6353c37e1c732613abd43672d1be29b8721e9a558
processed_at: '2026-08-12T12:42:27-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TAGA 人话版

Andrej，好嘞，我把刚才那坨技术细节重新用更直觉的方式聊一遍。

---

## 问题是什么

想象你蒙着眼睛走路，有人每秒给你念一段周围地形的描述。如果描述太宽泛——"前方 2 米内有台阶、有 gap、有石头"——信息量太大，你抓不住重点，不知道下一脚该踩哪。如果描述太聚焦——"你脚下正前方 5 厘米处高度 +3 厘米"——你又看不到远处的 gap，等走到 gap 边缘已经来不及准备了。

humanoid robot 面临的就是这个困境。height scan 给你一张 21x21 的局部地形图，精度够，但视野小；depth camera 给你前方远处的 depth image，视野大，但脚下看不到。两样东西都想要，但 onboard computer（Jetson Orin）算力有限，没法把 21x21=441 个 cell 全塞进 policy network 反复算。

那怎么办？

---

## TAGA 的核心 trick

**让 robot 自己学会"往哪看"**。

具体来说：policy 拿 depth image 和 proprioception（body 状态）当输入，输出一个 2D 坐标 $(r_x, r_y) \in [0,1]^2$，表示"在 height scan 上我应该 focus 哪个 11x11 的小块"。然后把这个小块 crop 出来，送进下游的 locomotion policy。

就这样。简单粗暴。

---

## 为什么这事能 work

关键在于 crop 用的是 **differentiable bilinear sampling**（从 Spatial Transformer Network 借来的）。意思是 crop 这一步梯度能 flow 回去，于是 gaze head 的参数能被 task reward 训练。

那 reward 怎么训练 gaze 呢？逻辑链是：

1. 如果 gaze 看错地方了，crop 出来的 patch 里没有关键 terrain 信息
2. policy 拿到 useless patch，做不出好的 action
3. action 不好，task reward 低
4. PPO gradient 推 gaze head 往"对的方向"调
5. 经过几万 iter，gaze 学会看 task-relevant 区域

**没有任何 gaze 的 ground truth label**。gaze 行为完全是 emergent 的。

这跟 self-attention 学 weight 的逻辑一模一样——attention weight 不是 explicit supervised，是从 end-task loss 里 backprop 出来的。TAGA 只是把 soft attention 换成了 hard crop，省了 compute。

---

## 涌现出来的 gaze 行为有多惊艳

Paper Figure 4 的可视化是我觉得整个 paper 最 compelling 的部分。几个例子：

**跨 1.2m gap 时**：
- 远处接近 gap → gaze 从脚下跳到 gap 对面边缘
- 起跳瞬间 → gaze 短暂回看当前边
- 落地后 → gaze 留在新支撑区

这跟人类跨 gap 之前的 anticipatory gaze 完全一致。人类研究里这叫 "gaze anchoring"——你起跳前眼睛先锁定 landing target，让 CNS 有时间 plan motor command。

**踩 stepping stones 时**：gaze 跳到下一个 sparse foothold。
**走 narrow beam 时**：gaze 沿 beam 中心线。
**走 stairs 时**：gaze 看下一级台阶边缘。
**平地行走时**：gaze 保持 local，覆盖附近 height changes。

没有任何一个这些 behavior 是 hand-crafted 的。全部从 RL reward 涌现出来。

---

## 两级 attention 的 hierarchy

TAGA 的设计是个两级 attention pipeline：

**Level 1: Active Gaze Module（宏观）**
- 输入：vision embedding + proprioception embedding
- 输出：归一化 2D 坐标 $r_t \in [0,1]^2$
- 作用：在 21x21 height scan 上选一个 11x11 子区域
- 实现：differentiable bilinear crop

这相当于决定"fovea 中心朝哪看"。

**Level 2: Visuomotor Fusion Encoder（微观）**
- 输入：cropped 11x11 patch + vision/proprioception query
- 机制：cross-attention，query 来自 vision+proprioception，key/value 来自 cropped patch
- 输出：128 维 fusion embedding $e_t^{pg}$

这相当于在 fovea 内部再做一次 soft attention，决定"这个小块里哪些 pixel 最 relevant"。

这跟人类视觉系统的 fovea + peripheral 分工特别像。Fovea 给你高分辨率但小视野，peripheral 给你大视野但低分辨率。Level 1 决定 fovea 朝哪，Level 2 决定 fovea 内部关注什么。

---

## MoE Action Decoder

最后一步是 Mixture-of-Experts。简单说就是 5 个 expert MLP，每个学一种 locomotion style（平地走、跨 gap、踩 stepping stone、走 beam、爬楼梯），gating network 根据当前 state 软路由，输出是 expert 的 weighted sum。

为什么要 MoE？单一 MLP 在这么多 terrain mode 下容易 mode collapse——学好了 gap 跨越可能忘了怎么走平地。MoE 让每个 expert 专注一个 mode，gating 学 mixture，表达力强很多。

这个设计借鉴自 [CMoE](https://arxiv.org/abs/2603.03067)，思路跟 LLM 里的 [Switch Transformer](https://arxiv.org/abs/2101.03961) 同源。

---

## 训练 pipeline 的两个 stage

**Stage 1: Clean training (30k iter)**
- 没有噪声、没有 domain randomization
- 让 robot 先学会 core skill
- 为什么不能一开始就加 noise？因为初期 policy 还没学会基本 locomotion，noise 会把 robot 直接 destroy，探索不到 useful behavior。先 clean 学好再 fine-tune，是 legged robot RL 的 standard warm-start trick。

**Stage 2: Fine-tune with domain randomization (10k iter)**
- 加 actuation noise、perception noise、external push、terrain perturbation
- 降低 entropy coefficient（已经掌握 skill，不需要太多 exploration）
- 让 policy 学会 robust to sim-to-real gap

Domain randomization 覆盖的范围很广，列一下：

- **Dynamics**: payload [-1, +3] kg、CoM offset、motor delay 0-3 steps、friction、restitution、random push
- **Proprioception noise**: angular velocity ±0.2 rad/s、joint position ±0.01 rad、joint velocity ±1.5 rad/s
- **Depth camera**: blur、stereo failure、reflections、sky artifacts、self-occlusion、contour corruption（这是相当 thorough 的 depth degradation simulation）
- **Height scan**: z-noise ±0.05m、ray-cast drift ±0.05m、random corruption
- **Terrain**: Perlin roughness、gap-edge transition width randomization、virtual edge obstacles

这是为什么 TAGA 在 real world 上 robust 的核心原因——sim 里已经见过各种 weird sensing condition。

---

## AMP：让 motion 不要太 weird

Paper Section 5.1 的 ablation 里有个很微妙的点。Table 1 显示去掉 AMP（TAGA-NoAMP）的 task success rate 跟 TAGA 几乎一样：

- Gaps: 96.40% vs 98.30%
- Stepping stones: 97.20% vs 97.90%
- Beam: 97.60% vs 98.50%

看起来 AMP 没啥用？但 Figure 5 一看就明白了。没有 AMP 时 robot 走路姿势很 weird——knee collapse、shuffling steps、turning 时不自然。

这些 weird motion 在 sim 里能 task success，但 real world 上 fragile。actuator stress 不均匀、sim-to-real gap 大、小扰动就崩。AMP 用 AMASS motion capture data 训一个 discriminator，给 policy 一个 "motion 自然度" reward，让 policy 不仅 task success 还要 human-like gait。

这其实是个相当 general 的 insight：**task reward 只能告诉你"做到了没"，告诉不了你"做得漂不漂亮"**。漂亮跟 robustness 高度相关，因为 human-like motion 通常 actuator stress 均匀、动态稳定 margin 大。

AMP 来自 [Peng et al. 2021](https://arxiv.org/abs/2104.02180)，是 GAN-style 思路：discriminator 区分 policy motion vs reference motion，policy 想骗过 discriminator。这跟 LLM RLHF 里 reward model 的角色有 parallel——都是给一个"软"的偏好信号。

---

## Asymmetric Actor-Critic

Critic 用 privileged information（ground-truth velocity + full uncropped height scan），actor 只用 partial observation。

这是 RL for robotics 的经典 trick。直觉是：value function 不需要部署，所以可以 cheat 用 oracle 信息，让 value estimate 更准、critic gradient 更稳。Actor 仍然学 partial-observation policy。

[Chen et al.](https://arxiv.org/abs/2010.14489) 系列工作 systematize 了这个 idea，现在已经是 legged robot RL 的 de facto standard。

---

## Reward function 的 key terms

完整 reward 在 Appendix C.2 Table 4，我挑几个有意思的讲：

**Velocity tracking 用 exp kernel**:
$$r_{vel} = \exp\left(-\|v_{torso}^{xy} - c_t^{xy}\|^2 / \sigma_v^2\right)$$

为什么不是 L2？因为 exp kernel smooth 且 saturate——velocity 已经接近 target 时 gradient 自动减小，避免 robot 拼命追 velocity 导致不稳定。这是 legged robot RL 的 common trick。

**Torso orientation penalty 有 gating**:
$$r_{torso\_ori} = -\frac{\|\omega_{torso}\|}{G_{flat}(h_t)} \|g_{torso}^{xy}\|^2$$

$G_{flat}(h_t)$ 是 gating function，只在 terrain 平坦时惩罚 torso tilt。不平坦时允许 tilt 是合理的——爬陡坡时 body 必然倾斜。

**Foot stumble penalty**:
$$-2.0 \cdot \mathbb{I}(\exists f \in \mathcal{F}: \|f_t^{xy}\| > 4|f_t^z|)$$

这是 penalize 脚横向滑动远大于垂直力的情况——典型的"绊倒"模式。fine-tune 阶段 weight 从 -2.0 升到 -5.0，因为已经掌握 skill，可以更严格约束。

**High hip-link acceleration termination** (Appendix C.4 第 5 条):
- foot in contact 时 hip-pitch link linear accel > 225 m/s² 就 terminate

这条很有意思。sim 里 robot 可能学到用 stiff landing 快速 recover balance，但这种 motion 在 real hardware 上会 destroy actuator。225 m/s² 是经验 threshold，过滤掉这种 stiff high-impact behavior。

---

## 实验结果的核心 takeaways

### Ablation 里的几个关键 insight

**Q1: Vision 和 height scan 是 complementary 的**

- CReF（vision-only baseline）在 stepping stones 上崩到 52.30%（看不到脚下精确几何）
- TAGA-HSOnly（height-scan-only）在 gaps 上崩到 93.10%（看不到远处 gap 来 prepare）
- TAGA 用两者达到 98.30% / 97.90%

**Q2: Active gaze 的价值**

- TAGA vs TAGA-FullScan：性能相当，但 FullScan 用 8 GPU / 49 days，TAGA 用 4 GPU / 17 days。**65.2% training cost reduction**。
- TAGA-InactiveGaze（固定 crop 位置）在 gaps 上崩到 57.10%。Active gaze 是 critical，不只是"use a crop"。

**Q3: AMP 的价值不在 task success，在 motion quality 和 robustness**

TAGA-NoAMP task success 跟 TAGA 接近，但 motion weird，sim-to-real fragile。

### Real-world SOTA

**120cm gap crossing** 是目前 perceptive humanoid 报告的最大 gap 距离，比之前 SOTA（90cm by Vel-Tracking AME-2）提升 33%，比多数方法（40-80cm）提升 50%+。

注意 Unitree G1 的 leg length 大约 60-70cm。跨 120cm gap 意味着 robot 必须 jump + 用 momentum，单纯 walking across 是不可能的。这种 dynamic maneuver 之前在 vision-based humanoid 上很少见到。

---

## 几个我个人的联想

### "Software 2.0" 在 robotics 上的 purest 范例

你之前讲 Software 2.0，核心是 specify objective + let gradient find solution。TAGA 完全是这个 paradigm：

- 没有 hand-crafted gaze rule
- 没有 hand-crafted foothold planner
- 没有 hand-crafted motion primitive
- 只有 reward + observation + architecture inductive bias

最终 gaze 行为、foothold selection、terrain adaptation 全都 emergent。这是 Software 2.0 在 robotics 上的 very clean example。

### 从 RAM 到 TAGA 的 lineage

DeepMind 2014 年的 [RAM (Recurrent Attention Model)](https://arxiv.org/abs/1406.6247) 用 RL 学 glimpse location policy 做 MNIST classification。TAGA 在 spirit 上跟 RAM 高度相似——都是 "RL-trained active attention"。区别在 RAM 是 classification，TAGA 是 real-world robot control。

"Active attention as RL policy" 这条 line 发展了 12 年，终于在 real robot 上 work 了。

### Hard attention 的复兴

Soft attention 在 transformer 时代 dominate，但 TAGA 用 hard attention（crop）。Hard attention 的好处是 computational sparsity——只 process 一小块而不是整个 feature map。对 onboard compute 受限的 robot，这 critical。

我觉得这是个 general trend：当 compute constrained 时，hard attention 会回归。LLM 里 long context 也面临类似问题，dense attention 不 scale，可能需要 RL-trained retrieval policy 决定 attend 哪些 chunk。这跟 [RETRO](https://arxiv.org/abs/2112.04426) 思路类似，但 retrieval policy 可以 RL 学而不是 uniform。

### 跟 induction heads 的 parallel

[Anthropic 的 induction heads work](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) 发现 LLM 里 attention 自然学到 "copy pattern"，emergent in-context learning。TAGA 发现 attention 自然学到 "look at next foothold"，emergent terrain-aware perception。

两个 case 都是 **attention learns the inductive bias needed for the task**。这种 pattern 在不同 modality（language vs robotics）里反复出现，说明 attention 是 universal mechanism for learning "where to look"。

### Future direction 我觉得值得 follow

1. **Actuator thermal modeling**: paper Section 7 提到 dynamic maneuver 的 thermal load 高，长时间运行 actuator 过热 degrade 控制精度。sim 里 actuator 不会过热，real 里会，这是个 hidden sim-to-real gap。类似 [Ankile et al.](https://arxiv.org/abs/2404.02897) 的工作方向。

2. **Height scan 用 neural implicit representation 替代**: paper 提到 height scan quality 在 tall grass、occlusion 多的场景容易脏。可能用 NeRF-style implicit terrain 解决，但 [Hoeller et al. NSR](https://arxiv.org/abs/2209.08853) 试过效果一般。这块还有空间。

3. **Active perception 推广到其他 modality**: TAGA 给了 visual gaze 的 template。可以推广到 auditory（turn head toward sound）、tactile（actively probe surface）、memory（actively retrieve relevant past episodes）。这些都是 RL-learnable active perception policies。

---

## 最核心的一句话

TAGA 让 humanoid robot 学会了"走路时该往哪看"，这件事完全从 task reward 涌现出来，没有显式监督。这件事 work 本身就是个强证据：**reward signal 蕴含了 implicit supervision for perception strategy**，只要 perception 跟 task success 相关，RL gradient 就会自动 find 它。

这跟 LLM 里 attention 学到 induction heads、robotics 里 emergence of gait patterns，都是同一个 story——**specify the objective, let the gradient do the talking**。

References:
- TAGA: https://marmotlab.github.io/taga-humanoid/
- RAM: https://arxiv.org/abs/1406.6247
- STN: https://arxiv.org/abs/1506.02025
- AMP: https://arxiv.org/abs/2104.02180
- CMoE: https://arxiv.org/abs/2603.03067
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Induction Heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c04
- RETRO: https://arxiv.org/abs/2112.04426
- AMASS: https://amass.is.tue.mpg.de/

---

# TAGA: Terrain-aware Active Gaze Learning for Agile Humanoid Locomotion 深度解析

Andrej，很高兴能跟你聊这篇 paper。我读完之后非常兴奋，因为这篇工作里有几个我觉得在 robotics RL + perception 范式里相当 elegant 的设计——尤其是 emergent gaze behavior 完全没有显式监督这件事，这跟你之前在 neural nets / RL 的演讲里反复强调的 "let the gradient do the talking" 的哲学高度一致。下面我尽量把直觉和细节都铺开。

Project page: https://marmotlab.github.io/taga-humanoid/
Isaac Lab (训练框架): https://arxiv.org/abs/2511.04831
AMP (adversarial motion priors): https://arxiv.org/abs/2104.02180
AME (Attention-based Map Encoding, 前序工作): https://www.science.org/doi/10.1126/scirobotics.adv3604

---

## 1. 一句话总结直觉

Humanoid locomotion 在 challenging terrain 上同时需要 **far-field preview**（远处要看到下一个 gap / 落脚点）和 **near-field precise geometry**（脚底下要精确知道高度，否则落脚就崩了）。但是 onboard compute 受限（Jetson Orin），height scan 又不能无限放大。TAGA 的核心 trick 是：**用 vision + proprioception 学一个 active gaze policy，决定 height scan 上哪个 KxK 子区域（K=11）对下一步最 task-relevant**，把这个子区域 crop 出来送进下游 policy。

这个设计的美妙之处在于：crop 操作通过 differentiable bilinear grid sampling 实现，于是整个 pipeline 是 end-to-end 可微的，**gaze 行为完全从 RL reward 涌现出来，没有任何 gaze 的 ground-truth 监督**。这跟你常说的 "emergent behavior from objective" 范式同源。

---

## 2. Problem Formulation 详解

TAGA 把 humanoid perceptive locomotion 建模为 POMDP $\mathcal{M} = \langle S, \mathcal{O}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$。注意几个关键的 observation 维度：

### Observation space

$$o_t = \{ p_t^H, d_t, h_t^{xyz} \}$$

- $p_t^H = \{p_{t-4}, \dots, p_t\}$：5 帧 proprioception 历史。每帧 $p_t = \{\omega_{b,t}, g_{b,t}, q_t, \dot{q}_t, a_{t-1}, c_t\}$
  - $\omega_{b,t} \in \mathbb{R}^3$：torso angular velocity
  - $g_{b,t} \in \mathbb{R}^3$：projected gravity vector（torso frame 下的重力方向，提供 body tilt 信息）
  - $q_t, \dot{q}_t \in \mathbb{R}^{29}$：joint positions / velocities（G1 是 29 DOF）
  - $a_{t-1} \in \mathbb{R}^{29}$：上一时刻动作
  - $c_t = (v_x^{cmd}, v_y^{cmd}, \dot{\psi}^{cmd}) \in \mathbb{R}^3$：velocity command

- $d_t \in \mathbb{R}^{1 \times 36 \times 64}$：forward-facing depth image，提供 long-range preview
- $h_t^{xyz} \in \mathbb{R}^{3 \times 21 \times 21}$：egocentric height scan，**注意是三通道 (x, y, z)**，每个 grid cell 存一个 3D terrain point

**这里有个非常关键的细节**：height scan 用的是 (x,y,z) 而不是单纯 z 值。这意味着模型不仅知道 "高度差"，还知道每个点的真实 3D 坐标，这对 sparse foothold / gap 的几何 reasoning 很重要——单纯 z 值在 gap 边缘容易产生歧义（不知道 gap 是悬崖还是缓降）。

### Action space

$a_t \in \mathbb{R}^{29}$：joint position targets at 50Hz，被 200Hz PD controller tracking。这个 PD tracking 的两层 hierarchy 是 legged robot RL 的标准 trick——你可以把它想成 "policy 输出的是 desired joint trajectory, low-level controller 负责追"。这种 hierarchy 让 policy 不需要学 motor dynamics 的细节，也让 sim-to-real 更稳。

### Objective

$$\max_\theta \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t \tilde{r}_t\right]$$

其中 $\tilde{r}_t = r_t^{env} + \eta r_t^{AMP}$，$\gamma = 0.99$。

---

## 3. 架构细节：四段式设计

TAGA 的网络结构图（Fig. 3）可以分成四个 stage。我把每一段都拆开讲。

### Stage 1: Independent Encoders

- **Depth encoder** $\phi_d$：CNN，把 $d_t \in \mathbb{R}^{1 \times 36 \times 64}$ 编码成 $e_t^d \in \mathbb{R}^{128}$。这个 embedding 捕捉远处的 terrain preview。
- **Proprioception encoder** $\phi_p$：MLP，把 $p_t^H$ 编码成 $e_t^p \in \mathbb{R}^{128}$。这个 embedding 捕捉 robot 的 dynamic state 和 command context。

两个 128 维 embedding 都会被后面的 module 复用，这种 reuse 是有意的：vision 不只是给 gaze 用，也要给 visuomotor fusion 当 query。

### Stage 2: Task-Relevant Active Gaze Module（这是 TAGA 的核心创新之一）

这是整个 paper 我觉得最 elegant 的地方。具体形式：

$$r_t = f_{roi}([e_t^d, e_t^p]), \quad r_t \in [0,1]^2$$

这里 $f_{roi}$ 是个轻量 prediction head（MLP），输入是 vision embedding 和 proprioception embedding 的 concatenation，输出是归一化的 2D 坐标 $(r_t^x, r_t^y) \in [0,1]^2$。这两个值表示在 height scan 上的 "应该看哪里" 的位置。

接下来是关键的 crop 操作：

$$\tilde{h}_t^{xyz} = \text{Crop}(h_t^{xyz}, r_t), \quad \tilde{h}_t^{xyz} \in \mathbb{R}^{3 \times K \times K}, K = 11$$

实现细节（Appendix B）：先把归一化坐标映射到 grid 坐标
$$ (u_t, v_t) = (\lfloor r_t^x \cdot M \rfloor, \lfloor r_t^y \cdot M \rfloor), \quad M = 21 $$

然后取以 $(u_t, v_t)$ 为中心、$K \times K$ 大小的子区域。

**The crucial trick**: crop 操作用 **differentiable bilinear grid sampling** 实现。这是从 STN (Spatial Transformer Network, [Jaderberg et al. 2015](https://arxiv.org/abs/1506.02025)) 借来的技术。如果用 hard crop（integer indexing），gradient 无法 backprop through crop 这一步，gaze head 就学不到东西。bilinear sampling 让 $r_t$ 的微小扰动能传播回 $f_{roi}$ 的参数，整个 pipeline 才能端到端训练。

**Intuition**: 你可以把 active gaze 想成 "learnable attention with hard structural bias"。普通的 self-attention 是 soft 的（所有 token 都有 weight），但 soft attention 在 21x21 = 441 个 cell 上做计算量大且容易稀释信息。TAGA 用 hard crop 做了一个 "winner-take-all" 的近似：直接选一个 11x11 的局部 patch。这本质上是一种 **structured sparsity prior**，强迫信息集中在一个小区域。

**为什么 gaze 会 emergently 学会 task-relevant**？因为没有显式监督告诉它 "应该看下一个落脚点"。但因为下游 locomotion policy 只能从 cropped patch 里获取 terrain 信息，**如果 gaze 看错了地方，policy 就会失败、reward 就低、PPO 就会推 gaze head 往正确方向移动**。这跟 self-attention 是同一个故事——attention 学到的 weight 来自 end-task gradient，不是来自 explicit label。

Fig. 4 的可视化非常 compelling：
- 在 1.2m gap 跨越时，gaze 从当前支撑区移动到对面 gap 边缘（preparing to cross）
- 在 stepping stones / beams 上，gaze 指向 sparse footholds 或 traversable strip
- 在连续 terrain 上，gaze 保持 local，覆盖附近的 height changes

这跟 humans / animals locomotion 时的 anticipatory gaze behavior 高度相似——人类跨 gap 之前会先看一眼对面边缘。这种 biological plausibility 是 emergent 的，不是 hand-crafted 的，这让 paper 更有说服力。

### Stage 3: Visuomotor Fusion Encoder

这一步把 cropped patch $\tilde{h}_t^{xyz}$ 进一步精细化处理。设计是 **cross-attention**：

1. CNN + MLP 把 cropped patch 编码成 pointwise terrain embeddings
$$E_t^m \in \mathbb{R}^{K \times K \times 128}$$
这里 "pointwise" 的意思是每个 (i,j) cell 都有一个 128 维 embedding，融合了 CNN 提取的 local 几何特征和 MLP 嵌入的 spatial coordinates (x,y,z)。

2. Query 来自 vision + proprioception 的 projection：
$$q_t = f_{proj}([e_t^p, e_t^d])$$

3. Multi-head cross-attention:
$$e_t^{pg} = \text{MHA}(q_t, E_t^m, E_t^m)$$
即 $q_t$ 作为 query，$E_t^m$ 同时作为 key 和 value。

**Intuition**: 这一步是在问 "在 cropped patch 里，哪些点对当前 locomotion decision 最重要"。Query 来自 proprioception + vision，所以 attention 会随着 robot 状态变化而变化——比如 robot 正在 swing leg 时，attention 可能集中在落脚区；正在准备 push off 时，attention 可能集中在 gap edge。

这是 TAGA 的两级 hierarchy：
- Level 1（gaze module）：宏观选区域（21x21 → 11x11）
- Level 2（fusion encoder）：微观选关注点（11x11 内部 attention weight）

**类比**: 这跟人类视觉系统的 fovea + peripheral 分工很像。Fovea 给你高分辨率但小视野，peripheral 给你大视野但低分辨率。Gaze module 决定 "fovea 看哪里"，fusion encoder 决定 "fovea 里哪些 pixel 最相关"。这种 biologically plausible 的两级 processing hierarchy，我觉得是这篇 paper 的 implicit contribution。

### Stage 4: MoE Action Decoder

最后用 Mixture-of-Experts 提高表达力：

$$e_t^\pi = [e_t^p, e_t^{pg}]$$
$$\alpha_t^i = \text{Softmax}(g(e_t^\pi))_i, \quad i = 1, \dots, N_e, \quad N_e = 5$$
$$a_t = \sum_{i=1}^{N_e} \alpha_t^i \mathcal{E}_i(e_t^\pi)$$

这里 $g(\cdot)$ 是 gating network，$\mathcal{E}_i$ 是第 i 个 expert（MLP）。所有 expert 都贡献 action，权重是 soft routing 出来的。

**Intuition**: 不同 terrain 需要不同的 locomotion style——平地是 walking，gap 是 jumping，beam 是 careful balancing。单一 MLP 难以 express 所有这些 mode（mode collapse 风险）。MoE 让每个 expert 学一个 mode，gating 学 mixture。这个 trick 在 [Switch Transformer](https://arxiv.org/abs/2101.03961)、[GShard](https://arxiv.org/abs/2006.16668) 等大模型里被验证过，CMoE（[Ma et al.](https://arxiv.org/abs/2603.03067)）把它带进 humanoid locomotion。TAGA 借用了 CMoE 的设计但用了更小的 $N_e=5$。

---

## 4. 训练 Pipeline 细节

### Asymmetric Actor-Critic

Actor 只能用 $o_t$（partial observation），但 critic 可以用 privileged information，包括：
- Ground-truth base linear velocity
- Full uncropped height scan

这是 RL for robotics 的经典 trick，源自 [Pinto et al.](https://arxiv.org/abs/1710.06542)、[Chen et al.](https://arxiv.org/abs/2010.14489) 系列 work。直觉是：value function 不需要部署，所以可以 cheat 用 oracle 信息，让 value estimate 更准、critic gradient 更稳。Actor 仍然学 partial-observation policy，部署时只用 actor。

### Curriculum Learning

Terrain 分 10 个 difficulty levels，每个 robot 根据当前 success/failure 动态调整 level——成功升级，失败降级。这是 automatous curriculum 经典设计，类似 [POET](https://arxiv.org/abs/1901.01753)、[PLR](https://arxiv.org/abs/2010.03910) 思想，但用 simpler heuristic。

Terrain 类型（Appendix C.1）：
- Gap crossing（最大跨距）
- Stair climbing（ascending + descending）
- Sparse footholds / stepping stones（precise foothold placement）
- Narrow beams
- Box obstacles
- Elevated platforms
- Sloped surfaces（up + down）

Sampling probability：stepping stones 0.3（最 demanding），stairs/slope 各 0.05，剩下均分。这个 prior 反映了作者判断 "sparse foothold 是最难、最需要 active gaze 的 task"。

### Two-Stage Training

参考 AME 的设计：
- **Stage 1 (clean)**：30k iter，no noise, no domain randomization。让 robot 先学会 core skill。
- **Stage 2 (fine-tune)**：10k iter，reduced entropy coefficient + domain randomization。注入 actuation variation、visual degradation、height-scan noise、external pushes、terrain perturbation。

**Intuition**: 在 noisy 环境下从零训练很难 explore（noise 会把 robot 一开始就 destroy）。先 clean 学好 skill 再 fine-tune 加 noise，本质上是一种 warm-start 的 sim-to-real strategy。

### Domain Randomization（Appendix C.5）

详细列一下 randomization 的 scope，这个对 sim-to-real 很重要：

**Robot dynamics**:
- Base payload: [−1.0, 3.0] kg
- Base CoM offset: [−0.05, 0.05] m in x, y, z
- Motor delay: 0–3 delay steps
- Friction: static [0.3, 1.0], dynamic [0.3, 0.8]
- Restitution: [0.0, 0.5]
- Random push: every 10–15s, planar velocity [−0.5, 0.5] m/s

**Proprioception noise**:
- Base angular velocity: ±0.2 rad/s
- Projected gravity: ±0.05
- Joint positions: ±0.01 rad
- Joint velocities: ±1.5 rad/s

**Depth camera** (30Hz during fine-tune):
- Contour corruption
- Random depth artifacts
- Reflections
- Sky artifacts
- Gaussian blur
- Stereo failure (too-close surfaces)
- Robot self-occlusion
- Clip to [0.4, 3.0] m

**Height scan**:
- z-axis noise [−0.05, 0.05] m
- Ray-cast drift [−0.05, 0.05] m per axis
- Random corruption of returns

**Terrain geometry perturbation**:
- Perlin roughness
- Randomized gap-edge transition widths
- Virtual edge obstacles

这是非常 comprehensive 的 randomization，覆盖了 sensing、dynamics、terrain 三大 axis。

### AMP（Adversarial Motion Priors）

AMP 来自 [Peng et al. 2021](https://arxiv.org/abs/2104.02180)，思想类似 GAN：用 discriminator 区分 policy motion 和 reference motion (AMASS retargeted)，policy reward 包含一个 "motion 自然度" 项。

公式：$\tilde{r}_t = r_t^{env} + \eta r_t^{AMP}$，$\eta$ 是 AMP reward coefficient。

**为什么需要 AMP**: 单纯 task reward 会让 robot 找 weird solutions——比如 inward knee collapse、shuffling steps（Fig. 5）。这些 weird solutions 在 sim 里能 task success，但在 real world 容易 fragile（actuator stress 不均、sim-to-real gap 大）。AMP 把 human motion capture 作为 distribution-level style prior，让 policy 不只能 task success，还能 "human-like"。

**关键 ablation**: Table 1 显示 TAGA-NoAMP task success rate 跟 TAGA 接近（gaps 96.40% vs 98.30%, stepping stones 97.20% vs 97.90%），但 Fig. 5 可视化显示 motion quality 差很多，而且 sim-to-real robustness 显著下降。这印证了 "AMP 不影响 nominal performance 但影响 robustness" 的直觉。

---

## 5. Loss Function 完整形式

$$\mathcal{L}_{policy} = \mathcal{L}_{PPO}(\tilde{r}) + c_v \mathcal{L}_{value} - c_e \mathcal{H}(\pi_\theta) + \lambda_c \mathcal{L}_{con} + \lambda_b \mathcal{L}_{roi}$$

变量含义：
- $\mathcal{L}_{PPO}(\tilde{r})$：PPO surrogate loss on augmented reward $\tilde{r} = r^{env} + \eta r^{AMP}$
- $\mathcal{L}_{value}$：critic MSE regression
- $\mathcal{H}(\pi_\theta)$：policy entropy bonus（鼓励 exploration）
- $\mathcal{L}_{con}$：MoE-terrain contrastive loss
- $\mathcal{L}_{roi}$：gaze boundary penalty
- $c_v, c_e, \lambda_c, \lambda_b$：loss weights

### MoE-Terrain Contrastive Loss（Appendix D.1）

这个 loss 借鉴自 CMoE。直觉是：让 MoE gating 学到 "terrain-aware" routing，而不是 uniform mixture。

具体形式：

对每个 embedding $e$（既包括 actor gate embedding $e_i^g$，也包括 terrain embedding $e_i^h$ from full height scan），定义：

$$P(e) = \text{softmax}(e C^\top / T_{con})$$
$$Q(e) = \text{Sinkhorn}(e C^\top)$$

其中 $C$ 是 shared prototype dictionary（$K$ 个 learnable prototypes），$T_{con}$ 是 temperature。Sinkhorn 用的是均衡分配约束。

Loss：

$$\mathcal{L}_{con} = -\frac{1}{2BK} \sum_{i=1}^{B} \left[ Q(e_i^g)^\top \log P(e_i^h) + Q(e_i^h)^\top \log P(e_i^g) \right]$$

**Intuition**: 这是一种 SwAV-style ([Caron et al. 2020](https://arxiv.org/abs/2006.09882)) 对比学习。$Q$ 用 Sinkhorn 做 balanced assignment（避免 collapse 到同一 prototype），$P$ 用 softmax 做 soft prediction。两个视角（gate vs terrain）互相 predict，强迫它们在 shared prototype space 里 align。结果就是 gating 学会根据 terrain 类型 route 到不同 expert。

### Gaze Boundary Loss（Appendix D.2）

$$\mathcal{L}_{roi} = \frac{1}{2B} \sum_{i=1}^{B} \left( [m - r_i^x]_+ + [r_i^x - (1-m)]_+ + [m - r_i^y]_+ + [r_i^y - (1-m)]_+ \right)$$

变量：
- $r_i = (r_i^x, r_i^y) \in [0,1]^2$：归一化 gaze location
- $m = 0.05$：boundary margin
- $[\xi]_+ = \max(\xi, 0)$：positive-part operator

**Intuition**: 防止 gaze degenerate 到边界。如果没有这个 penalty，policy 可能学到一个 "lazy" solution——总是 gaze 在边界，因为边界附近 crop 出来的 patch 可能包含部分边界外的 "empty space"（被默认为 z=0），这种 spurious "flat" signal 可能让 policy 误以为前方平坦。boundary loss 强迫 gaze 留在有效区域中央。

### Symmetry Augmentation

利用 humanoid 左右 morphological symmetry，做数据增强：
- 镜像 proprioception observations
- 镜像 height scans
- 镜像 AMP states
- 镜像 actions（左右 joint swap）
- 对 depth 用 horizontally flipped virtual camera view

both original + mirrored samples 都进 PPO update。这种 augmentation 让 sample efficiency 翻倍（来自 [Mittal et al. ICRA 2024](https://arxiv.org/abs/2310.08107)）。

---

## 6. Reward Function 详解（Appendix C.2 Table 4）

Reward 是典型的 legged robot RL reward design，分几大类：

### Task Objective

- **Alive**: 3.0（每步 alive 给 reward）
- **Linear velocity tracking**: $2.0 \cdot \exp\left(-\|v_{torso}^{xy} - c_t^{xy}\|^2 / \sigma_v^2\right)$
- **Yaw velocity tracking**: $3.0 \cdot \exp\left(-(\omega_{torso}^z - c_t^\omega)^2 / \sigma_\omega^2\right)$
- **Yaw command regularization**: $-1.0 \cdot |c_t^\omega|$（鼓励 small yaw command）
- **Forward progress**: $-0.5$ penalty if commanded forward but not moving forward

**Intuition**: 用 exp kernel 而不是 L2，是 legged robot RL 的常见 trick——exp 提供 smooth gradient 且 saturate，避免 robot 拼命追 velocity 导致不稳定。

### Posture Regularization

- Torso angular velocity: $-0.05 \cdot \|\omega_{torso}^{xy}\|^2$
- Torso orientation: $-2.0 \cdot \frac{\|\omega_{torso}\|}{G_{flat}(h_t)} \|g_{torso}^{xy}\|^2$，其中 $G_{flat}(h_t)$ 是 gating function，只在 terrain 平坦时惩罚 torso tilt——不平坦时允许 tilt 是合理的
- Pelvis orientation: $-0.5 \cdot \|g_{pelvis}^{xy}\|^2$

### Joint Regularization

- Joint torque: $-1.5 \times 10^{-7} \cdot \|\tau_t\|^2$
- Joint velocity: $-5.0 \times 10^{-4} \cdot \|\dot{q}_t\|^2$
- Joint acceleration: $-1.25 \times 10^{-7} \cdot \|\ddot{q}_t\|^2$
- Link acceleration: $-0.01 \cdot \frac{1}{|B|}\sum_{b \in B} \|\dot{v}_t^b\|$
- Hip/arm/waist deviation from default
- Joint position/velocity/torque limits

### Contact and Gait

- Undesired contact: $-1.0 \cdot \sum_{b \notin \mathcal{F}} \mathbb{I}(\|f_t^b\| > \epsilon_f)$
- Foot air time: $0.25 \cdot \mathbb{I}_{cmd} \mathbb{I}_{single} \min_{f \in \mathcal{F}} T_{mode}^f$
- Air/contact time variance: $-0.7 \cdot (\text{Var}_{f \in \mathcal{F}}(\text{clip}(T_{air}^f, 0.5)) + \text{Var}_{f \in \mathcal{F}}(\text{clip}(T_{contact}^f, 0.5)))$ — fine-tune 时 weight 升到 $-2.0$
- Foot stumble: $-2.0 \cdot \mathbb{I}(\exists f \in \mathcal{F}: \|f_t^{xy}\| > 4|f_t^z|)$ — fine-tune 升到 $-5.0$
- Foot slide: $-0.1 \cdot \sum_{f \in \mathcal{F}} \mathbb{I}_{contact}^f \|v_t^{xy,f}\|$
- Foot orientation: $-0.5$
- No-fly: $-2.0 \cdot \mathbb{I}(\sum_{f \in \mathcal{F}} \mathbb{I}_{contact}^f = 0)$
- Feet too near: $-1.0 \cdot [d_{min} - \|x_t^L - x_t^R\|]_+$ — fine-tune 升到 $-5.0$

### Fine-tune only

- Volume penetration: $-1.0 \cdot \sum_{x \in \mathcal{V}} \mathbb{I}(\delta_x > 0)(\|v_x\| + \epsilon)\delta_x$
- Stand still: $-0.3 \cdot \mathbb{I}(\|c_t^x\| < \epsilon_c) \mathbb{I}(\|c_t^\omega\| < \epsilon_c) (\|q_t - q^0\|_1 - b)$

注意 fine-tune 阶段很多 penalty weight 都加大了（foot stumble 从 -2 升到 -5、feet too near 从 -1 升到 -5、air/contact variance 从 -0.7 升到 -2.0），这是有意的——fine-tune 时已经掌握 skill，可以更严格约束来打磨 robustness。

---

## 7. Termination Conditions（Appendix C.4）

1. **Timeout / terrain boundary**: 20s episode 或 robot 出界
2. **Illegal non-foot contact**: torso/pelvis/waist/shoulder/elbow/hip/knee 接触地面（threshold 1N）
3. **Bad torso orientation**: torso projected gravity 与 upright 夹角 > 0.8 rad (~45°)
4. **Low base height**: root < 0.5m 或 root clearance < 0.2m
5. **High hip-link acceleration during contact**: foot in contact 时 hip-pitch link linear accel > 225 m/s²

**第 5 条很有意思**。这是一个很 specific 的 termination，针对 "stiff high-impact landing"。Intuition: sim 里 robot 可能学到用 stiff landing来快速 recover balance，但这种 motion 在 real hardware 上会 destroy actuator。225 m/s² 这个 threshold 是经验值。

---

## 8. 实验结果分析

### Simulation Ablation（Table 1）

| Method | GPUs | GPU-days | Gaps | Steps | Beam | HighPlat | C1 | C2 |
|---|---|---|---|---|---|---|---|---|
| CReF (vision baseline) | 2 | ~10 | 97.40% | 52.30% | 96.50% | 98.70% | 85.20% | 43.10% |
| TAGA-HSOnly | 4 | ~17 | 93.10% | 92.50% | 98.30% | 99.60% | 90.50% | 91.50% |
| TAGA-InactiveGaze | 4 | ~14 | 57.10% | 83.20% | 95.60% | 100% | 72.70% | 48.80% |
| TAGA-FullScan | 8 | ~49 | 99.50% | 98.00% | 97.50% | 100% | 93.40% | 92.50% |
| TAGA-NoAMP | 4 | ~16 | 96.40% | 97.20% | 97.60% | 99.50% | 89.80% | 91.60% |
| **TAGA (Ours)** | 4 | ~17 | **98.30%** | **97.90%** | **98.50%** | **100%** | **93.70%** | **93.90%** |

关键 takeaways：

1. **CReF 在 stepping stones 上崩了（52.30%）**: 纯 vision 没有精确 local geometry，sparse foothold 无法 precise placement。这强力佐证了 "vision 单独不够" 的论点。

2. **TAGA-HSOnly 在 gaps 上崩了（93.10% 比 TAGA 低 5%）**: 没有 vision preview，robot 无法 anticipate 远处 gap，到达边缘来不及 prepare。这印证了 "height scan 单独不够" 的论点。

3. **TAGA vs TAGA-FullScan**: 性能相当（多数 terrain 在 1% 内），但 FullScan 用了 8 GPUs / 49 GPU-days，TAGA 只用 4 GPUs / 17 GPU-days。**65.2% training cost reduction** with comparable performance。这是 gaze module 的核心 win。

4. **TAGA-InactiveGaze 在 gaps 上严重退化（57.10%）**: 固定 crop 位置导致远 foothold 不在窗口内。Active gaze 是 critical，不只是 "use a crop"。

5. **C1/C2 (OOD terrain)**: TAGA 93.70% / 93.90% 显著优于所有 baseline。这表明 active gaze 学到的策略是 generalizable 的，不只是 memorize specific terrain。

### Real-World（Table 2）

TAGA 在 Unitree G1 上 deploy，关键 metrics：

| Capability | TAGA | 之前 SOTA |
|---|---|---|
| Gap traversal | **120 cm** | 90 cm (Vel-Tracking AME-2) |
| Sparse foothold spacing | **70 cm (uneven)** | 60 cm (RPL, flat) |
| Platform | 40 cm | 50 cm (PIM) |
| Beam | ✓ | various |
| Stairs | ✓ | various |

**120 cm gap 是最大的 real-world perceptive humanoid gap crossing 报告**，比之前 SOTA 提升 50%。注意 Unitree G1 的 leg length 大约 60-70cm，跨 120cm gap 意味着 robot 必须 jump + 用 momentum，不是单纯 walking across。

### 资源消耗

- 8,000 parallel envs in Isaac Lab
- 4 × RTX 5090 GPUs
- ~17 GPU-days total (30k iter stage 1 + 10k iter stage 2)
- Inference: onboard Jetson Orin
- Action 50Hz, PD 200Hz, depth 30Hz

---

## 9. Emergent Behavior 分析

我觉得 paper 最有意思的部分是 Fig. 4 展示的 emergent gaze pattern。让我深入讲：

### (a) Gap Crossing 的 Gaze 轨迹

1. **Phase 1 (approach)**: robot 远离 gap，gaze 集中在当前支撑区
2. **Phase 2 (preparation)**: 接近 gap 边缘，gaze 跳到对面 gap edge
3. **Phase 3 (crossing)**: 起跳瞬间，gaze 短暂回到当前边
4. **Phase 4 (landing)**: 落地后，gaze 留在新的支撑区

这跟 humans / animals locomotion 的 anticipatory gaze 完全一致。人类研究里这叫 "gaze anchoring"——眼睛提前锁定 landing target，让 CNS 有时间 plan motor command。TAGA 自发涌现了这种行为，没有任何 gaze supervision。

### (b) 不同 terrain 的 Gaze pattern

- Stepping stones: gaze 跳到下一个 sparse foothold
- Beams: gaze 沿 beam 中心线
- Stairs: gaze 看下一级台阶边缘
- Continuous terrain: gaze 保持 local，覆盖附近 height changes

这表明 TAGA 学到的不是 "terrain-specific rule"，而是一个 **terrain-conditioned active perception policy**，可以根据当前 robot state 和 terrain 类型动态决定看哪里。

### (c) OOD 测试

测试了训练时没见过的 terrain 组合，gaze 仍然合理。这表明学到的策略不是 memorize，而是 generalize。

---

## 10. 跟相关工作的关系

### Mapping-based methods

- [Miki et al. Science Robotics 2022](https://www.science.org/doi/10.1126/scirobotics.abk2822): ANYmal 用 elevation map
- [Hoeller et al. ANYmal Parkour](https://www.science.org/doi/10.1126/scirobotics.adi7566)
- [He et al. AME](https://www.science.org/doi/10.1126/scirobotics.adv3604): attention-based map encoding for quadruped
- [Zhang et al. AME-2](https://arxiv.org/abs/2601.08485): 扩展到 humanoid

TAGA vs AME: AME 用 attention 处理整个 height scan，没有 active gaze。TAGA 把 attention 分两层——一层做 hard crop（active gaze），一层做 soft attention（fusion encoder）。**计算复杂度: O(M²) → O(K²)**, $K \ll M$。

### Vision-based methods

- [Cheng et al. Extreme Parkour](https://arxiv.org/abs/2304.02758): depth-only quadruped parkour
- [Agarwal et al. egocentric vision](https://arxiv.org/abs/2309.15227)
- [Zhuang et al. HPL](https://arxiv.org/abs/2407.04005): humanoid parkour
- [Long et al. PIM](https://arxiv.org/abs/2407.15218): perceptive internal model
- [Hao et al. CReF](https://arxiv.org/abs/2603.29452): cross-modal recurrent fusion（TAGA 的 baseline）

Vision-based 的核心问题：forward-facing depth 看不到脚下，sparse foothold 难做。TAGA 用 height scan 补这块。

### Active Perception

- [ADAPT](https://arxiv.org/abs/2603.16328): adaptive perception clipping for noise robustness
- [CART](https://arxiv.org/abs/2604.14344): context-aware temporal sequence selection

TAGA 跟这些不同：它们是 temporal selection，TAGA 是 spatial selection（gaze location）。TAGA 的 active perception 是 "where to look"，前两者是 "what past info to use"。

### MoE in Locomotion

- [Ma et al. CMoE](https://arxiv.org/abs/2603.03067): contrastive MoE for humanoid
- [Wang et al. MoRE](https://arxiv.org/abs/2506.08840): mixture of residual experts

TAGA 直接复用 CMoE 的设计，但 expert 数量从 CMoE 的更大值降到 5。

---

## 11. 我的 Intuition 和联想

### 11.1 Gaze = Learnable RoI Pooling with End-Task Gradient

这篇 paper 在我看来本质上是把 computer vision 里的 **RoI Align / STN** 思想搬到了 robot control 里，但用 RL end-task gradient 而不是 R-CNN 那种 region proposal network 来 supervise。这跟 [Vision Transformers with Saliency](https://arxiv.org/abs/2104.08853)、[RAM (Recurrent Attention Model, Mnih et al. 2014)](https://arxiv.org/abs/1406.6247) 是同一个 family。

RAM 是 2014 年 DeepMind 的工作，用 RL 学一个 glimpse location policy 做 image classification。TAGA 在 spirit 上跟 RAM 高度相似——都是 "RL-trained active attention"。区别在于：
- RAM 在 MNIST/ImageNet 上做 classification
- TAGA 在 locomotion control 上做，reward 来自 task performance 而不是 classification accuracy
- TAGA 的 gaze 直接影响 control，RAM 只影响 classification logits

我觉得 Karpathy 你一定会喜欢这个 lineage：从 RAM 到 TAGA，"active attention as RL policy" 这条 line 已经发展了 12 年，终于在 real-world robotics 上 work 了。

### 11.2 Hard Attention 的复兴

Soft attention 在 transformer 时代 dominate，但 TAGA 用的是 hard attention（hard crop）。Hard attention 的好处是 **computational sparsity**——你只 process 一小块区域而不是整个 feature map。这对 real robot 的 onboard compute 很关键。

我觉得这是一个 general trend：当 compute constrained 时，hard attention 会回归。可能 vision model 也会重新发现这条 path（[Vision Saliency Transformers](https://arxiv.org/abs/2204.03645) 已经有这个 trend）。

### 11.3 Emergent Behavior 的 Beauty

Paper Section 4 有一句：

> "we find that such gaze behaviors can naturally emerge through reinforcement learning alone, without requiring additional supervision or explicit guidance"

这是整个 paper 我觉得最 profound 的地方。Active gaze 没有 supervision，但 gaze 自然学到了 "approach gap 时 gaze 转向 gap 对面"。这说明：

**Reward signal 蕴含了 implicit supervision for perception strategy**

只要 perception 策略与 task success 相关，RL gradient 就会自动 find it。这跟 LLM RLHF 中 reward model 让 model 学到 "诚实、有帮助" 类似——这些 behavior 不是 explicit annotated，而是 emerge from objective。

### 11.4 与 LLM 中 Attention 的类比

这让我想到 LLM 里 attention 的 emergent pattern：
- [Anthropic 的 induction heads work](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html): attention 自然学到 "copy pattern"，emergent in-context learning
- TAGA: attention 自然学到 "look at next foothold", emergent terrain-aware perception

两者都是 **attention learns the inductive bias needed for the task**。这种 pattern 在不同的 modalities（language vs robotics）里反复出现，说明 attention 是一个 universal mechanism for learning "where to look"。

### 11.5 关于 Limitations

Paper Section 7 提到 actuator 过热问题。这其实是个很 realistic 的 limitation：dynamic maneuvers（jump、push off）的 thermal load 高，长时间运行会 degrade 控制精度。这意味着 sim-to-real 还有一个 hidden gap：**sim 里 actuator 不会过热，real 里会**。

Future work 提到 "uncertainty-aware control under degraded hardware"——这暗示需要把 actuator thermal dynamics 也 model 进 sim，类似 [Ankile et al.](https://arxiv.org/abs/2404.02897) 的 motor thermal modeling。这是个值得 follow 的方向。

### 11.6 关于 height scan quality 的脆弱性

Paper 说 "poor height scan quality on complex terrain can cause improper gaits or failures"。这是一个 fundamental limitation of mapping-based methods：依赖 elevation map 的质量。在 tall grass、indoor cluttered 等 occlusion 多的场景，height scan 容易脏。Paper 加了不少 noise randomization 来 mitigate，但根本问题没解。

可能的 future direction: 用 neural implicit representation（NeRF-style）替代显式 height scan，让 model 自己 reconstruct 一个 implicit terrain。但 [Hoeller et al. NSR](https://arxiv.org/abs/2209.08853) 试过，效果一般。

### 11.7 跟你 (Karpathy) 的 "Software 2.0" 思想的关系

你之前在 Software 2.0 演讲里讲，未来 software 是 specify objective + 让 gradient 找 solution。TAGA 完全是这个 paradigm：
- 没有 hand-crafted gaze rule
- 没有 hand-crafted foothold planner
- 没有 hand-crafted motion primitive
- 只有 reward + observation + architecture inductive bias

最终 gaze 行为、foothold selection、terrain adaptation 都是 emergent。这是 Software 2.0 在 robotics 上的 purest 范例之一。

### 11.8 跟 Optimal Control 的对比

传统 humanoid locomotion 用 Trajectory Optimization + MPC（如 [Mastalli et al.](https://arxiv.org/abs/2010.08014)、[Carpentier et al.](https://arxiv.org/abs/1902.07766)），explicitly plan footholds 用 contact-implicit optimization。这种方法的优点是 interpretability，缺点是：
1. 需要 accurate dynamics model
2. compute expensive，real-time 难
3. 不容易 generalize to unseen terrain

TAGA 用 RL + active gaze 替代了 explicit planning。Policy 直接 map observation 到 action，gaze 隐式实现了 "where to land" 的决策。这是 Software 2.0 替代 Software 1.0 的经典 case。

### 11.9 关于 Active Perception 的 future

我觉得 TAGA 揭示了一个更大的 trend：**未来 robot perception 会从 "passive observe everything" 走向 "active decide what to observe"**。

- Visual: gaze where to look（TAGA 做了）
- Auditory: turn head toward sound source
- Tactile: actively probe surface
- Memory: actively retrieve relevant past episodes

这些都是 RL-learnable active perception policies。TAGA 给出的是一个 template，可以推广到其他 modalities。

### 11.10 关于在 LLM/Vision model 里借鉴 active gaze

LLM 现在用 dense self-attention 处理整个 context。但如果 context 极长（百万 token+），dense attention compute 不 scale。一个可能的 direction：用 RL 训练一个 "gaze policy" 决定 retrieval 哪些 chunk。这跟 [RETRO](https://arxiv.org/abs/2112.04426)、[Memorizing Transformers](https://arxiv.org/abs/2203.08983) 思路类似，但 retrieval policy 用 RL 学而不是 uniform。

我觉得这个方向有 research potential，TAGA 给了一个很好的 robotics analog。

---

## 12. 总结

TAGA 的核心贡献可以浓缩成几条：

1. **Hierarchical active perception**: vision preview → gaze module → cropped patch → fusion encoder → action。两级 attention，一级 hard crop，一级 soft cross-attention。

2. **Emergent gaze without supervision**: gaze behavior 完全从 RL reward 涌现，跟 human anticipatory gaze 高度一致。

3. **Computational efficiency**: 比 full-scan 节省 65.2% training cost，达到 comparable performance。

4. **Real-world SOTA**: 120cm gap crossing (real Unitree G1)，比之前 SOTA 提升 50%。

5. **Multimodal fusion done right**: vision + proprioception + height scan 三模态，各司其职——vision 给 preview，proprioception 给 state，height scan 给 local geometry，gaze 把它们 align。

最终 deploy 在 Jetson Orin 上，real-time inference。这是一篇非常 elegant 的 systems-meets-learning paper，整体设计感觉是经过深思熟虑的——每个组件都有 ablation 支撑，没有 decorative complexity。

Reference links:
- TAGA Project Page: https://marmotlab.github.io/taga-humanoid/
- Isaac Lab: https://arxiv.org/abs/2511.04831
- AMP: https://arxiv.org/abs/2104.02180
- AMASS: https://amass.is.tue.mpg.de/
- AME: https://www.science.org/doi/10.1126/scirobotics.adv3604
- RAM (Recurrent Attention Model): https://arxiv.org/abs/1406.6247
- STN (Spatial Transformer Network): https://arxiv.org/abs/1506.02025
- SwAV: https://arxiv.org/abs/2006.09882
- Switch Transformer: https://arxiv.org/abs/2101.03961
- Software 2.0 (Karpathy): https://karpathy.medium.com/software-2-0-a64152b37c04
- Induction Heads (Anthropic): https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

如果你想 dive deeper，我特别推荐看 Fig. 4 的 gaze visualization——那是我觉得整篇 paper 最 compelling 的 qualitative evidence。还有 Appendix D.1 的 contrastive loss 设计，是 MoE 在 robotics 上用 contrastive learning 的一个干净实现。
