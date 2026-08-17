---
source_pdf: Understanding the Sense of Self through Robotics.pdf
paper_sha256: 2c8b3ecf297692066de9ce760029cdd3790b3c1a06b7b5c86d3f8310618de6a8
processed_at: '2026-08-12T19:16:20-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 paper

## 一句话版本

这篇 paper 说的是：**我们能不能用造机器人的方式，来搞清楚"我"这个东西到底是怎么从一堆零件里冒出来的？**

---

## 为什么这件事有意思

你想想看，你自己——"我"——是什么？

你有一具身体，你能感觉到"这是我的手"，你能感觉到"这个动作是我做的"，你记得昨天发生了什么，你觉得自己是同一个人。这些东西加起来，就是所谓的 "self"。

但是这个 self 是从哪来的？它是大脑里某个特定的地方产生的吗？还是整个身体+大脑+环境一起算出来的一个 pattern？

哲学吵了几百年也没吵明白。这篇 paper 的作者说：**别吵了，我们来造一个试试。**

造机器人的好处是：你必须把每个细节都搞清楚才能造出来。你不能含糊地说"哦 self 是意识涌现的结果"，你得真的写出代码、接好传感器、装好电机，让它能动、能感知、能学习。如果造出来的机器人表现出了类似 self 的行为，那你就知道你的理论可能是对的；如果造不出来，那你的理论就少点东西。

---

## Self 拆成零件

作者把 self 拆成两大块：

### Minimal self（最小版自我）
就是你不需要思考就有的那些感觉：
- **SoO**（Sense of Ownership）：这是我的手，这是我的身体
- **SoA**（Sense of Agency）：这个动作是我做的，不是我被动被推的
- **Self-other distinction**：区分"我"和"外面的世界"
- **Perspective**：我有一个看世界的视角

这些东西婴儿出生几个月就有了，不需要语言，不需要反思。一条鱼大概也有这个级别的 self。

### Extended self（扩展版自我）
更高阶的：
- **时间上的连续性**：我记得过去，我能想象未来
- **人际 self**：我理解别人也有 mind，有 ToM（Theory of Mind）
- **Narrative self**：我能讲一个关于"我是谁"的故事

这个大概是人类独有的，大概从几岁到青春期慢慢发展出来。

---

## 怎么用机器人造 minimal self

### Body schema：机器人怎么知道自己长什么样

一个机器人刚启动，它不知道自己有几个关节、手臂能伸多远、关节弯到什么角度会碰到自己。它得自己学。

**方法**：让它乱动（motor babbling），同时记录 proprioception（关节角度）和 tactile（皮肤触觉）的信号。然后用一个 autoencoder 把这些高维信号压缩成一个低维的 latent space $z$：

$$
z = E_\phi(x_{\text{proprio}}, x_{\text{tactile}}, x_{\text{visual}})
$$

这个 $z$ 就是机器人的 **self-manifold**——它编码了"我的身体能怎么动、能感知到什么"的不变量。你可以把它想象成机器人版的 body schema。

**为什么叫 manifold？** 因为它是一个低维的曲面/子空间，嵌在 high-dimensional sensorimotor 空间里。机器人能做的动作、能感知的东西，只占所有可能的 sensorimotor signal 的一个很小的子集，这个子集就是 manifold。

**直觉**：就像你闭着眼睛也能知道自己的手在哪，因为你大脑里有一个身体的 latent model。机器人学的就是这个。

### Self-other distinction：怎么区分"我"和"外面的东西"

经典的 **comparator model** 说是这样的：

$$
\hat{s}_{t+1} = M(s_t, a_t) \quad \text{（我预测动作后的感觉）}
$$
$$
e = s_{\text{actual}} - \hat{s} \quad \text{（预测误差）}
$$

如果 $e$ 很小，说明"这个感觉是我自己的动作产生的"；如果 $e$ 很大，说明"这个感觉是外界的"。

**但问题来了**：Lanillos 等人让 iCub 机器人照镜子，发现这个模型不够用。因为镜子里你的动作和你自己的动作是 perfectly correlated，comparator 会说"镜子里的是我"，但实际上镜子里的是你的镜像。

所以他们提出了 **double comparator**：除了 prediction error，还得检查 **spatio-temporal contingency**——这个视觉信号是不是真的 *由我的动作 causally 产生的*，还是只是 correlated。

**直觉**：comparator 像是在问"这个感觉和我预测的一致吗"，double comparator 还多问一句"这个感觉真的是我造成的吗"。前者是 correlation，后者是 causation。

### Sense of Agency：怎么感觉"这是我做的"

人类有个很奇怪的现象叫 **intentional binding**：如果你主动按键，然后听到一声 beep，你会觉得按键和 beep 之间的时间比你被动按键时要短。这个时间压缩 $\Delta t < 0$ 就是 SoA 的 implicit measure。

**Robotics 怎么用这个？** Navare 等人做了一个实验（paper 里的 Fig. 2）：人和 iCub 机器人一起按键，有时人按有时机器人按，然后 beep。他们发现：**如果人觉得机器人有 intention，人会形成 sense of joint agency (SoJA)**——无论谁按键，人都觉得时间被压缩了。

但更有意思的是 social modulation：
- 如果任务框架是"避免损失"（合作避免出错），和有 intention 的机器人合作反而 *降低* 个人 SoA——好像责任分散了
- 如果任务框架是"一起完成目标"（positive），和 humanoid 合作 *升高* SoA

**直觉**：SoA 不是一个固定的个人属性，它会根据 social context 动态变化。你在团队里感觉到的"我做了这件事"和一个人时不一样。这暗示 self 在 social 互动中是被 reconfigured 的。

---

## 怎么用机器人造 extended self

### 时间上的 self：episodic memory + mental time travel

让机器人记录经验 $(o_t, a_t, r_t, c_t)$（observation, action, reward, context），然后用 similarity retrieval 回忆。更高级的是用 forward model 在 latent space 里 rollout，想象未来：

$$
z_{t+k}^{\text{imagined}} = g_\psi^{(k)}(z_t, a_{t:t+k}^{\text{simulated}})
$$

这跟 Dreamer / MuZero 的 world model roll-out 是同一个思路。机器人在 latent space 里"想象"未来，就是 mental time travel 的 robotic 版本。

### Theory of Mind：机器人怎么猜人心里想什么

Vinanzi 等人在 iCub 上做了一个 Bayesian ToM：机器人维护一个关于人类 belief 的 belief：

$$
b_t^{\text{robot}}(\text{human believes } X) \propto P(\text{human did } A_t \mid \text{human believes } X) \cdot b_{t-1}^{\text{robot}}(X)
$$

比如经典的 false-belief task：人看到 object 在位置 A，然后 object 被移到 B，但人不在场。机器人要推断"人还会去 A 找"。iCub 能做到这个。

### Narrative self 的萌芽

Pointeau & Dominey 让 iCub 从 episodic memory 里蒸馏出 abstract facts，然后用语言描述自己。这就是 narrative self 的种子——把碎片化的经验编织成一个连贯的 self-story。

---

## 机器人当实验工具去研究人的 self

不只是造机器人，还可以用机器人当 probe 去戳人的 self。

### Rubber Hand Illusion 的 robotic 版

经典的 RHI：给你看一个 rubber hand，同时刷你的真手和 rubber hand，你会觉得 rubber hand 是你的。

争论：
- **Bottom-up**：只要视觉和触觉 spatiotemporally correlated，就能产生 ownership，连桌子都行
- **Top-down**：需要一个 pre-existing body map 提供 anatomical constraint

机器人可以 parametrically 调形态（humanoid vs 抽象机械臂）、调 motor repertoire，来 break 这个 deadlock。

### 第三根拇指实验

Kieliba et al. 2021 给人加一根机器人拇指，训练 5 天后：
- 行为上：augmented hand 运动控制改善
- 神经上（fMRI）：真手指的 neural representation 变得 less distinctive

**直觉**：body schema 是 plastic 的，可以被外物 co-opt。你的身体表征不是一个固定的 hardcoded map，而是一个能被经验 reshape 的 latent space。

### Teleoperation 中的 body transfer

让人 teleoperate 一个 humanoid 机器人，用 full-body motion capture。一段时间后，人会觉得 robot body 是自己的——叫 "beaming" effect。这说明 SoO 可以跨载体 transfer。

---

## 机器人模拟精神疾病

这是最 speculative 但最酷的部分。

### Schizophrenia 的 robotic model

精神分裂有不同的 subsyndrome，对应 self 的不同 disturbance：
- **Reality distortion**（幻觉、妄想）：SoO 出问题，觉得想法不是自己的
- **Passivity syndrome**：SoA 出问题，觉得动作不是自己做的
- **Disorganization**：narrative self 和 ToM 出问题

**Yamashita & Tani 2012** 做了一个 two-layer RNN：

```
Intentional layer (high-level goal/intention)
       ↕ error signal
Sensorimotor layer (low-level motor execution)
```

他们人为 *perturb* 两层之间的 error signal：
- **Mild perturbation**：机器人行为正常，但内部 latent activity 出现异常波动 → 对应精神分裂早期的 aberrant feelings/thoughts（患者行为还正常，但主观体验已经异常）
- **Strong perturbation**：机器人行为变得 disorganized、stereotyped → 对应 chronic schizophrenia 的 overt symptoms

**直觉**：同一个 architecture，不同 noise/precision regime 下产生不同症状。这支持 predictive coding 的观点——精神分裂是 prediction error 处理的 precision 出问题，不同 severity 对应不同 precision 损坏程度。

### Autism 的 robotic analog

Autism 主要是 interpersonal self 的差异——ToM 模块 under-developed 或不同配置。机器人可以把 ToM 模块做 under-trained，观察行为差异。

### Depression

Depression 里 self 在"轴上 collapse"——affective flattening + SoA 降低 + 时间变慢。robotic model：降低 value system gain $\eta$（RL 里的 reward sensitivity），SoA 的 temporal binding 减弱。

---

## 当前机器人 self 的最大硬伤

### Interoception gap

这是 paper 反复强调的问题。动物的 self 很大程度上 grounded 在 **interoception**——对心跳、呼吸、饥饿、疲劳的感知。这些信号驱动 homeostasis 和 emotion。

但机器人的 internal milieu 贫瘠得多：
- 不需要呼吸
- 不需要消化
- 不会饿
- 不会累
- 没有心跳

Seth & Tsakiris 的 **Beast Machine thesis** 说：interoception 是 self 的 grounding。没有 interoceptive prediction error，机器人的 SoO/SoA 永远是"残缺版"。

**补救方向**：
- Self-healing soft robots（材料级的 homeostasis-like 过程）
- Embodied energy harvesting（机器人"吃 fuel"，接近 autopoietic）
- Neuromodulated control（dopamine/serotonin 的 analog 作为 drive signal）

**直觉**：你之所以觉得"这是我"，很大程度上是因为你 *感觉* 到你的身体——心跳、呼吸、饥饿、疼痛。机器人现在只有外感受和本体感受，缺了内感受这一层，所以它的 self 永远差一口气。

### Enactivist critique

Sharkey & Ziemke 说得更狠：当前机器人是 *mechanistic embodiment*，不是 *phenomenal embodiment*。真正的 organism 是 **autopoietic machine**——它 *主动维持自己的存在*，抵抗 decay。机器人不 self-maintain，所以它的 self 顶多是模仿，不是真的。

**直觉**：你之所以有 self，是因为你 *在乎* 自己活着。你的身体不断在对抗 entropy，这种对抗本身是 self 的基础。机器人不会死，所以它不在乎自己，所以它没有真正的 self。

---

## 我觉得最酷的几个点

1. **Self 可以从 sensorimotor invariance 里 emerge**。你不需要 consciousness，不需要 language，只需要一个 body + sensors + actuators + learning，就能学出一个 self-manifold，它编码"我的身体能怎么动"。这是 minimal self 的种子。

2. **Comparator model 不够**。Robotics 实证证明纯 prediction error 不够区分 self 和 other，需要 contingency check。这是 robotics 给认知科学的反向贡献。

3. **SoA 是 social 的**。你和机器人合作时，你的 SoA 会根据 social frame 升高或降低。Self 不是个人属性，是 dyad 的 dynamic。

4. **Yamashita-Tani 的精神分裂模型太优雅了**。同一个 RNN，调 inter-layer error 的 magnitude，就能从"潜伏症状"过渡到"显性症状"。这暗示精神疾病可能不是 *不同的* architecture，而是 *同一个* architecture 在不同 noise regime 下的表现。

5. **Body schema 是 plastic 的**。第三根拇指实验证明你的身体表征可以被外物 reshape，甚至 neural representation 都变了。Self-manifold 不是 hardcoded 的。

---

## 如果我要 implement 这个

我会这样设计一个 minimal self robot：

```
1. Sensors:
   - 关节编码器（proprioception）
   - 人工皮肤（tactile）
   - 相机（visual, egocentric）
   - 电池电量 + 温度 + 损伤检测（interoception!）

2. Self-manifold learner:
   - Autoencoder: (proprio, tactile, visual) → z
   - Forward model: z_t, a_t → z_{t+1} predicted
   - 学习目标: reconstruction + forward prediction

3. Self-other distinction:
   - Comparator: prediction error e
   - Contingency learner: C(s, a, s') → self/other flag
   - 如果 e 小 AND contingency 高 → self-caused

4. SoA module:
   - Track prediction error across time
   - Low average error → high SoA
   - 可以用 intentional binding paradigm 验证

5. Interoceptive drives:
   - battery < threshold → "hungry"
   - damage detected → "pain"
   - temperature > threshold → "overheating"
   - 这些 signal 调节 action selection 和 attention

6. 上面再叠:
   - Episodic memory (record + replay)
   - ToM (Bayesian belief about human's belief)
   - Narrative (distill + language)
```

关键是我会 **把 interoception 塞进去**——电池电量、损伤信号、温度。让 robot 的 self-manifold 不仅编码"我的身体形状"，还编码"我的身体 *状态*"。这样 robot 才会有 Damasio 说的 proto-self 的雏形——一个 grounded in bodily need 的 self。

然后 perturb 这个 system：
- 调高 prediction error threshold → 看会不会出现 schizophrenia-like 行为
- 调低 interoceptive sensitivity → 看会不会出现 depersonalization-like 状态
- Under-train ToM → 看会不会出现 autism-like social behavior

---

## Reference links

- Paper 本体: https://doi.org/10.1126/scirobotics.adn2733
- Open access 版: https://eprints.whiterose.ac.uk/id/eprint/219433/
- Yamashita & Tani 精神分裂模型: https://doi.org/10.1371/journal.pone.0037843
- Lanillos mirror self-other: https://doi.org/10.3233/FAIA200368
- Hafner self-manifold: https://doi.org/10.3389/fnbot.2020.554578
- Navare SoJA 实验: https://doi.org/10.1126/scirobotics.adj3665
- Kieliba 第三拇指: https://doi.org/10.1126/scirobotics.abd8101
- Vinanzi 机器人 ToM: https://doi.org/10.1098/rstb.2018.0032
- Seth & Tsakiris Beast Machine: https://doi.org/10.1016/j.tics.2018.08.003
- Bongard self-modeling star robot: https://doi.org/10.1126/science.1126495
- Blanke & Metzinger minimal phenomenal selfhood: https://doi.org/10.1016/j.tics.2008.10.003
- Botvinick & Cohen Rubber Hand Illusion: https://doi.org/10.1038/35816
- Haggard intentional binding: https://doi.org/10.1038/nn825
- Metta iCub: https://doi.org/10.1016/j.neunet.2010.08.010
- Hoffmann robotic homunculus: https://doi.org/10.1109/TCDS.2017.2783894
- Prescott & Dominey temporal self: https://doi.org/10.1098/rstb.2023.0415
- Moulin-Frier DAC-h3: https://doi.org/10.1109/TCDS.2018.2853146

---

**总结一句话**：这篇 paper 说的就是，self 这玩意儿不是什么神秘的东西，它可以被拆成零件，每个零件都能用 robotics 去搭、去测、去 perturb。我们现在已经能搭出 minimal self 的雏形（body schema + comparator + contingency），extended self 也有萌芽（memory + ToM + narrative），但最大的 gap 是 interoception——机器人不饿不累不会死，所以它的 self 永远差了一口气。把这一口气补上，是下一步的关键。

---

# Paper 解析: *Understanding the Sense of Self through Robotics* (Prescott, Vogeley, Wykowska, *Science Robotics* 2024)

Paper 的核心 thesis: **robotics 可以从三个互补路径为 self 这一科学问题提供 leverage**——(i) 作为 embodied model 把 self 的子系统逐个 build 起来；(ii) 作为 controlled experimental probe 去刺激/扰动 human self；(iii) 通过 perturbed self-model 去复现 psychiatric disorders 的 diversity of selves。贯穿全文的 technical spine 是 **predictive processing + layered cognitive architecture**:self 被视为由一组 coupled generative models 在 brain-body-environment 三角耦合中 minimization prediction error 时涌现的 attractor dynamics。

DOI: https://doi.org/10.1126/scirobotics.adn2733
White Rose open access: https://eprints.whiterose.ac.uk/id/eprint/219433/

---

## 1. Theoretical scaffolding: Self 作为 system-level emergent property

作者采纳的是 **systems biology** 视角(Bertalanffy General System Theory):organism 是 complex dynamical system,系统整体的 emergent properties 不出现在任一 component 中。Self 因此被理解为 **分布在 brain + body + niche 中的 attractor dynamics pattern**(Gallagher 的 pattern theory of self, [Gallagher 2013](https://doi.org/10.3389/fnhum.2013.00135)),并由以下三个 core phenomena 绑定而成:

1. **Sense of Ownership (SoO)** — "mine-ness" 的体验质量,通过 pronominal syntax 和 flexible **self-manifold** 边界表达。
2. **Sense of Agency (SoA)** — "我是 action 的 author"的体验,基于 prediction-outcome congruence 或 causal efficacy inference。
3. **Transtemporal unity** — 跨时间的一致性 narrative self(Dennett 的 *center of narrative gravity*)。

**Harnad 的 robot functionalism** 立场:non-symbolic sensorimotor capacities 是 grounding cognition 和 self 的 primary substrate。这与 enactivism/embodied cognition (Varela, Thompson, Rosch; Seth; Bruineberg)一致。

### 1.1 Minimal vs Extended self 的 decomposition

| Aspect | Minimal self | Extended self |
|---|---|---|
| Core subsystems | SoO, SoA, self-other distinction, weak 1PP | transtemporal, interpersonal, narrative |
| Brain level | brainstem-level + early cortical (Damasio *proto-self*, Panksepp *primal self*) | hippocampal/DMN, mPFC, TPJ |
| Developmental onset | birth ~ 2-3 months(Rochat) | 4 months - adolescence(ToM, autobiographical memory) |
| Reflective? | pre-reflective | reflective, narrative |
| Robotic 例 | iCub body schema [Hoffmann 2018](https://doi.org/10.1109/TCDS.2017.2783894); star-like robot [Bongard 2006](https://doi.org/10.1126/science.1126495) | iCub ToM [Vinanzi 2019](https://doi.org/10.1098/rstb.2018.0032); Cog [Scassellati 2002](https://doi.org/10.1023/A:1015588527845) |

关键:**Minimal self 不需要 meta-cognition**。Infant 在 2-3 months 已有 SoO+SoA 但缺乏 conceptual self。这给了 robotics 一个低门槛切入点——只需要 pre-reflective 的 sensorimotor loop 就能 emulate 起步阶段。

---

## 2. Robotics 作为 self 的 constructive model

### 2.1 Body schema 与 **self-manifold**(Hafner 2020, [Front. Neurorobot.](https://doi.org/10.3389/fnbot.2020.554578))

这是 paper 里最 connectionist-friendly 的概念。Hafner 把 **self-manifold** 定义为 robot 在 sensorimotor 空间上学到的低维 latent manifold,z = f_φ(o, a),其中 invariances(body morphology, joint limits, reachable space)被显式编码。

形式化:给定一个 robot 的 proprioception $x_p \in \mathbb{R}^{D_p}$(joint angles)、tactile $x_t \in \mathbb{R}^{D_t}$(skin pressure)、visual $x_v \in \mathbb{R}^{D_v}$(egocentric image),encoder 学习

$$
z = E_\phi(x_p, x_t, x_v), \quad (\hat{x}_p, \hat{x}_t, \hat{x}_v) = D_\theta(z)
$$

Loss(简化版 autoencoder + sensorimotor prediction):

$$
\mathcal{L} = \underbrace{\|x_p - \hat{x}_p\|^2 + \|x_t - \hat{x}_t\|^2 + \|x_v - \hat{x}_v\|^2}_{\text{reconstruction}} 
+ \lambda \underbrace{\|z_{t+1} - g_\psi(z_t, a_t)\|^2}_{\text{forward latent dynamics}}
$$

其中:
- $z \in \mathbb{R}^d$, $d \ll D_p + D_t + D_v$,下标 $t$ 为 time step;
- $a_t$ 为 motor command(关节速度或扭矩);
- $g_\psi$ 是 latent forward model,对应 cerebellar forward model 的功能对应物;
- $\lambda$ 权衡 reconstruction vs prediction。

学到的 $z$ 就是 robot 的 **self-representation latent space**,可以看作 parietal cortex body schema 的 robotic analog。Lanillos & Cheng [IEEE TCDS 2017](https://doi.org/10.1109/TCDS.2017.2650620)在 iCub 上的 *sensorimotor contingencies* 实现就是这一思路的实例化,通过在 latent space 中 thresholding prediction error 来区分 self-caused vs externally-caused sensor events。

### 2.2 Self-other distinction:Comparator Model 与其批判

**经典 comparator model**([Frith, Blakemore, Wolpert](https://doi.org/10.1016/S1364-6613(00)01504-3);David, Newen, Vogeley 2008, [Conscious. Cogn.](https://doi.org/10.1016/j.concog.2008.06.004)):

$$
\hat{s}_{t+1} = M(s_t, a_t) \quad \text{(forward model)}
$$
$$
e_{\text{proprio}} = s_{t+1}^{\text{actual}} - \hat{s}_{t+1} \quad \text{(proprioceptive prediction error)}
$$
$$
\text{SoA} \propto -\|e_{\text{proprio}}\|^2
$$

直觉:如果我自己产生的 motor command 预测的 sensory consequence 与实际 sensory feedback 一致,那么这事件就是 *my* action,SoA 高。Schizophrenia 的 *passivity syndrome* 被解读为这个 comparator 的 deficit:即便 action 是 self-generated,error signal 也异常大,所以患者感觉"是别人让我做的"。

**Lanillos, Pages, Cheng 2020**([Front. AI Appl.](https://doi.org/10.3233/FAIA200368))在 iCub 上做 mirror self-recognition,发现纯 comparator model 不够用,提出 **double comparator model**:

$$
\text{SoA} = f\big(\underbrace{-\|e_{\text{proprio}}\|^2}_{\text{classical comparator}}, \underbrace{C(s_t, a_t, s_{t+1})}_{\text{spatio-temporal contingency learner}}\big)
$$

其中 $C(\cdot)$ 是一个学到的 contingency 模块,评估 sensor event 与 motor command 的时序+空间因果关系。原因:当 robot 在 mirror 中看到镜像,motor command 和 visual feedback 的简单 correspondence 也会成立,但镜像不是 "self-action"——需要 contingency check 来排除纯 correlative 假阳。

这是 robotics 给理论心理学的一个反向 contribution:实证展示 comparator model 不充分。

### 2.3 Sense of Ownership:Rubber Hand Illusion 的 robotic 形式

经典 RHI(Botvinick & Cohen, [Nature 1998](https://doi.org/10.1038/35816))。Multisensory correlation 框架:

$$
P(\text{ownership} | \tau_{vis}, \tau_{tac}) \propto \exp\big(-\alpha\|\tau_{vis} - \tau_{tac}\|^2 - \beta\, d_{anatomical}\big)
$$

其中 $\tau_{vis}, \tau_{tac}$ 是 visual 和 tactile stimulation 的时间序列,$d_{anatomical}$ 是 rubber hand 相对于 participant 真实手的 anatomical plausibility 距离。$\alpha, \beta$ 是个体差异参数。

争论:
- **Bottom-up view**(Armel & Ramachandran, [Proc. R. Soc. B 2003](https://doi.org/10.1098/rspb.2003.2364)):spatiotemporal correlation 足够,可以把 ownership 转移到桌子、纸箱这类非 body-shaped 物体。
- **Top-down view**(Tsakiris, [Neuropsychologia 2010](https://doi.org/10.1016/j.neuropsychologia.2009.09.028)):需要 pre-existing body map 提供 anatomical constraint。

Robotics 可以 **parametrically manipulate** robot morphology(humanoid vs zoomorphic vs abstract)和 motor repertoire 来 break 这个 deadlock,例如 iCub(child-like) vs Tiago(adult-like industrial) vs 完全 abstract robotic arm。

### 2.4 Sense of Agency:**Intentional Binding** 与 vicarious SoA

**Intentional binding**(Haggard, Clark, Kalogeras, [Nat. Neurosci. 2002](https://doi.org/10.1038/nn825)):

$$
\Delta t = t_{\text{perceived}}(outcome) - t_{\text{actual}}(outcome)
$$

对 voluntary action, $\Delta t < 0$(perceived compression);对 externally-caused, $\Delta t > 0$ 或 = 0。这一 implicit measure 在 HRI 中特别有用。

**Vicarious SoA**(Strother et al. 2010; Roselli et al. 2022, [Sci. Rep.](https://doi.org/10.1038/s41598-022-18107-7)):人类在观察 humanoid robot 执行 action 时,如果 robot 看起来 human-like 且被 attribute intentionality,会产生 *vicarious* binding,即把 robot 的 action 视为 quasi-self。

**Navare et al.**([Sci. Robot. 2024](https://doi.org/10.1126/scirobotics.adj3665),Fig. 2 of the paper)做了 **Sense of Joint Agency (SoJA)** 实验:
- Human + iCub 在 complementary keypress 任务中合作,产生 auditory beep。
- Measure: $\Delta t_{\text{keypress} \to \text{beep}}$ 估计的 perceived compression。
- Result: 只有当 participant attribute intentionality 给 robot 时,SoJA 形成,$\Delta t$ 显著 < 0,无论 beep 由谁 keypress 触发。
- EEG 同步显示 P300/FRN 类似 human-human joint action 的 marker。

**Social modulation**:
- Ciardo et al. 2020 ([Cognition](https://doi.org/10.1016/j.cognition.2019.104109)):在 loss-avoidance task 中,attribution of intentionality 给 robot **降低** individual SoA(diffusion of responsibility)。
- Sahaï et al. 2023:在 positive joint goal task 中,与 humanoid 合作 **升高** SoA。

两个结果看起来矛盾,实际反映 **social frame**(loss avoidance vs shared goal)是 SoA 的 modulator。这点很关键:**Self 不是 isolated property,而是 dyad/collective context 中的 dynamic variable**。

### 2.5 Interoception gap — 当前 robot 的硬伤

Paper 反复指出:**robot 的 internal milieu 比 animal 贫瘠得多**。没有 respiration, digestion, excretion,heartbeat,circadian rhythm。Seth & Tsakiris 的 *Beast Machine* thesis([TICS 2018](https://doi.org/10.1016/j.tics.2018.08.003))强调 interoception 是 self 的 grounding。没有 interoceptive prediction error,robot 的 SoO/SoA 永远是"残缺版"。

补救方向(作者提到但未深入):
- Self-healing sensorized soft robots([Roels 2022](https://doi.org/10.1016/j.materel.2022.100003))— material-level 的"homeostasis-like"过程;
- Embodied energy harvesting([Aubin et al., Nature 2022](https://doi.org/10.1038/s41586-021-04162-0))— robot 吃自己的 fuel,接近 autopoietic machine 概念;
- Neuromodulated control architectures(Krichmar):dopamine/serotonin/acetylcholine 的 analog 作为 drive-modulating signal,使 robot 进入 homeostatic-allostatic loop。

公式上,可以把 interoception 加入 free energy:

$$
F = \underbrace{F_{\text{extero}}(o_{\text{vis}}, o_{\text{aud}}, \ldots)}_{\text{exteroceptive}} + \underbrace{F_{\text{proprio}}(o_{\text{proprio}})}_{\text{proprioceptive}} + \underbrace{F_{\text{intero}}(o_{\text{intero}})}_{\text{interoceptive — currently weak in robots}}
$$

### 2.6 Perspective taking:weak 1PP 与 strong 1PP

Blanke & Metzinger([TICS 2009](https://doi.org/10.1016/j.tics.2008.10.003)):

- **Weak 1PP** = spatial frame of reference + global body representation + perspective originating within body。Self-manifold 拥有 robot 即可满足。
- **Strong 1PP** = reflective self + narrative identity。属于 extended self。

这是把 phenomenology 切成可工程化的台阶——一个 minimal self 不必 assume hard problem of consciousness。

---

## 3. Extended self 的 robotic 实现

### 3.1 Episodic/autobiographical memory + mental time travel

Prescott & Dominey([Phil. Trans. R. Soc. B 2024](https://doi.org/10.1098/rstb.2023.0415)):iCub 上的 episodic memory 系统。形式化(简化):

$$
\text{memory}_t = \{(o_t, a_t, r_t, c_t)\}_{t=1}^T
$$

其中 $c_t$ 是 context vector。Recall 时通过 similarity retrieval:

$$
\text{retrieve}(q) = \arg\max_{t} \cos(\text{embed}(q), \text{embed}(c_t))
$$

Mental time travel = replay + generative imagination:在 latent space上 用 forward model $g_\psi$ roll-out alternative futures:

$$
z_{t+k}^{\text{imagined}} = g_\psi^{(k)}(z_t, a_{t:t+k}^{\text{simulated}})
$$

这与 hippocampal replay 的功能对应物很像,且与 Karpathy 你自己提过的 world-model / MuZero / Dreamer 系列有强 family resemblance。

### 3.2 SLAM 与 spatial self

Spatial localization 与 temporal localization 共用 hippocampal/DMN substrate([Karapanagiotidis 2017](https://doi.org/10.1016/j.neuroimage.2016.12.018))。Robotics 的 **SLAM** 是成熟的工程化对应物:

$$
p(x_t, m | z_{1:t}, u_{1:t}) \quad \text{(posterior over pose + map)}
$$

NeuroSLAM([Yu et al. 2019](https://doi.org/10.1007/s00422-019-00802-2))直接把 rat brain 的 place/grid/head-direction cell 编码塞进 SLAM,这个 cross-fertilization 是 robotics ↔ neuro 的好例子。

### 3.3 Theory of Mind 与 intentional stance

Dennett 的 **intentional stance**([Dennett 1989](https://mitpress.mit.edu/9780262540528/the-intentional-stance/)):人类把 mental states 归给目标系统以预测其行为。Robot 因为 humanoid form + behavior,常被 human 默认采用 intentional stance([Marchesi 2019](https://doi.org/10.3389/fpsyg.2019.00450);[Bossi 2020](https://doi.org/10.1126/scirobotics.abb6688))。

**Vinanzi, Cingano, Wykowska 2019**([Phil. Trans. R. Soc. B](https://doi.org/10.1098/rstb.2018.0032))在 iCub 上实现了一个 *computational ToM* 模型:robot 维护一个 Bayesian belief about human's belief:

$$
b_t^{\text{robot}}(\text{human's belief}) \propto P(\text{human action}_t \mid \text{human's belief}_t) \cdot b_{t-1}^{\text{robot}}
$$

类似 Bayesian Theory of Mind(Baker, Saxe, Tenenbaum)。Robot 通过观察 human 在 false-belief task 中的选择来推断 human 是否还相信一个已被移走的 object 还在原位。

### 3.4 Narrative self 的萌芽

Pointeau & Dominey([Front. Neurorobot. 2017](https://doi.org/10.3389/fnbot.2017.00027)):从 episodic memory 中 distill 出 abstract facts,然后通过 language production 串成 self-report。这是 narrative self 的 robotic seed。

DAC-h3([Moulin-Frier et al., IEEE TCDS 2018](https://doi.org/10.1109/TCDS.2018.2853146))在 iCub 上构建 **Distributed Adaptive Control** 架构 layer 化版本,把 self-model 作为 layer 间的 binding variable。

---

## 4. Robots 作为 experimental probes 去 probe human self

这部分 paper 把 robot 从 subject 变 apparatus,关键 example:

### 4.1 Teleoperation 与 body transfer("beaming")

- Ventre-Dominey et al. 2019([Sci. Rep.](https://doi.org/10.1038/s41598-019-46528-5)):short beaming procedure 后 participant 把 robot face 当作自己的 face(enfacement)。
- Jung et al. 2022:full-body motion capture teleoperation 产生 strong body transfer。
- Jazbec et al. 2017:android robot teleoperation 中 SoO 部分转移到 robot body。

### 4.2 Prosthetics & augmentation 的 body plasticity

**Kieliba et al. 2021**([Sci. Robot.](https://doi.org/10.1126/scirobotics.abd8101)):"third thumb"实验,5 天训练后:
- Behavioral:augmented hand 的 motor control 改善。
- Neural(fMRI):biological fingers 的 neural representation 在 augmented hand 上变得 less distinctive——身体表征 *re-organized*。
- 直觉:body schema 是 plastic latent space,可以被外物 co-opt。

这验证了 *body schema* 既是 bottom-up(sensorimotor)又是 top-down(anatomical prior),cross-cutting 2.3 中 debate。

### 4.3 HRI 中的 SoA modulation(Fig. 2)

已在 2.4 详述。要点:SoA 不是 monolithic,而是 social-frame dependent 的 dynamic variable。

---

## 5. Diversity of selves:Robotics → Computational Psychiatry

### 5.1 Schizophrenia 的 subsyndromes(Liddle 1987)

| Subsyndrome | Self-relevant disturbance | Robotic test |
|---|---|---|
| Psychomotor poverty | SoA flattening, retardation | motor planning layer 中 reward/prior weakening |
| Disorganization | incoherent narrative self,ToM deficit | intentional layer 中 noise injection |
| Reality distortion(hallucinations, delusions) | SoO over thought insertion, SoA deficit for own action | comparator 中 $\|e_{\text{proprio}}\|$ 阈值提升 |

Vogeley et al. 1999([Conscious. Cogn.](https://doi.org/10.1006/ccog.1999.0393)):prefrontal cortex 实现 self model 的 essential functions。Smith, Badcock, Friston 2021([Psychiatry Clin. Neurosci.](https://doi.org/10.1111/pcn.13144)):predictive coding 在临床神经科学中。

### 5.2 Yamashita & Tani 2012([PLOS ONE](https://doi.org/10.1371/journal.pone.0037843))

最 striking 的具体 robot psychosis model。架构是 two-layer RNN:

$$
h_t^{(s)} = \tanh(W_{hs} x_t + W_{hh}^{(s)} h_{t-1}^{(s)}) \quad \text{(sensorimotor layer)}
$$
$$
h_t^{(i)} = \tanh(W_{hi} h_t^{(s)} + W_{hh}^{(i)} h_{t-1}^{(i)}) \quad \text{(intentional layer)}
$$

Intentional layer $h^{(i)}$ 通过 top-down prediction 约束 sensorimotor layer $h^{(s)}$。*Error signal* between layers:

$$
e_t^{\text{inter-layer}} = h_t^{(s)} - \hat{h}_t^{(s)}(h^{(i)})
$$

实验:人为 *uncompensate* inter-layer error(类似于削弱 top-down precision in predictive coding):
- Mild level → robot 行为还在 normal range,但 latent activity 出现 aberrant fluctuations → **对应 mild symptoms 如 aberrant feelings/thoughts**。
- High level → robot overt behavior 变 disorganized/stereotyped → **对应 chronic schizophrenia**。

这是 *symptom severity gradient* 的 robotic analog。强 prediction:不同 symptom dimensions 可以由同一 architecture 在不同 noise/precision regime 下产生。

### 5.3 Comparator model 不足 → schizophrenia 研究

[David et al. 2008](https://doi.org/10.1016/j.concog.2008.06.004);[Frith & Done 1989, Psychol. Med.](https://doi.org/10.1017/S0033291700012400)。Robotics 可以 systematic perturb comparator 的 forward model $M$,或 perturb error threshold,观察行为对应症状。这是 computational psychiatry 的 embodied 版本。

### 5.4 其他 disorders(简略)

- **Depersonalization**:SoO + SoA 同时 weakened,perspectivity drift。Robotic model:同时 perturb self-manifold latent regularizer + comparator threshold。
- **Autism**:interpersonal self 不同组织(Baron-Cohen)。Vinanzi 的 ToM robot 是 probe;反过来把 robot 的 ToM 模块做 *under-trained* 可以 mimic autism-like behavior。
- **Depression**([Davey & Harrison 2022, Transl. Psychiatry](https://doi.org/10.1038/s41398-022-01419-5)):self 在 axis 上 collapse——affective flattening + SoA 降低 + time-slowdown。Robot model:降低 value system gain $\eta$ in RL,SoA temporal binding 减弱。

---

## 6. Open questions 与批判性评估

### 6.1 Subjectivity vs consciousness

Paper 谨慎区分:
- **Subjectivity** = first-person perspective + phenomenal selfhood(Blanke & Metzinger 的 minimal phenomenal selfhood = global body identification + spatiotemporal self-location + weak 1PP)。
- **Consciousness** 是更强 requirement,需要 supporting transtemporal unity 等高级功能,但 itself 不 imply self-model(meditation 和 psychedelics 中 consciousness 在 self-model 解体情况下仍可存在;[Letheby & Gerrans 2017](https://doi.org/10.1093/nc/nix016))。

Gallagher 2000([TICS](https://doi.org/10.1016/S1364-6613(99)01417-5))甚至认为 Tani 1998([J. Conscious. Stud.](http://www.imprint.co.uk/art/tani.pdf))那个 predictive robot 已是 *minimal phenomenal selfhood* 的可能 instantiation。

### 6.2 Enactivist critique(Sharkey & Ziemke 2001;Di Paolo 2003)

当前 robot 仍是 *mechanistic embodiment*,而非 *phenomenal embodiment*:
- 不是 autopoietic machine — 不 self-maintain;
- 不是 open thermodynamic system 抵抗 decay;
- 缺 real homeostasis + affect。

要满足 *strong* enactivist self,robot 必须能"饿死、累坏、自愈"。这正是 2.5 中 interoception gap 的根源问题。

### 6.3 Ethics

[Metzinger 2009](https://www.basicbooks.com/books/the-ego-tunnel/):build subjectivity-endowed robots 等于冒着创造能 *suffer* 的 moral patient 的风险。Bryson 2018:patiency 不是设计 virtue。但若 robot 在 socially-assistive 场景有 self-aspect 反而 *更好用*——更 safe(自身 embodiment awareness)、更 contextual(trans-temporal memory)、更 empathic(self-other model)。

### 6.4 Layered architecture 的 spec(综合)

把 paper 多次提及的 layered predictive architecture 写成伪 spec:

```
Layer 0 (body/skin): tactile + proprio + intero sensors  → low-dim latent z_0
Layer 1 (brainstem analog): homeostatic drives d={energy, safety, ...}
                            modulatory signals μ(d)
Layer 2 (forward models): ŝ_{t+1} = M(s_t, a_t); error e_proprio
Layer 3 (self-other): C(s, a, s') contingency check → self/other flag
Layer 4 (peripersonal space + weak 1PP): PPS map + body-centered frame
Layer 5 (episodic memory + mental time travel): replay/rollout
Layer 6 (ToM + interpersonal): b_robot(human's belief)
Layer 7 (narrative self + abstract concept): language-anchored identity
Layer 8 (meta-cog + strong 1PP): self-reflection, self-report
```

每一层都是一个 generative model + precision $\pi^{(l)}$(predictive coding 中的 precision weighting)。Schizophrenia/Yamashita-Tani 的 perturbation = 调某层 $\pi^{(l)}$。

Free-energy 总目标(简化):

$$
\mathcal{F} = \sum_l \pi^{(l)} \|e^{(l)}\|^2 + \sum_l \mathrm{KL}\big[q^{(l)} \| p^{(l)}\big] + \text{regularizers (homeostatic, narrative coherence, ...)}
$$

变量含义:
- $l$ = layer index;
- $\pi^{(l)} \in \mathbb{R}_+$ = precision(posterior variance 的倒数),调节该层 prediction error 的 trust;
- $e^{(l)}$ = 该层 bottom-up prediction error;
- $q^{(l)}$ = approximate posterior over layer's latent causes;
- $p^{(l)}$ = prior(generative model 的 top-down prediction)。

---

## 7. 我自己 build intuition 的几个 takeaway

1. **Self ≈ multi-layer latent world-model 中一个被 sticky-tag 的 sub-manifold**。这个 tag 在 robot 中可以通过 sensorimotor invariance 自己 emerge(self-manifold),在 human 中还加上 interoceptive + social grounding。
2. **SoA 与 SoO 是 error budget 的两面**:SoA 关心 *proprio/extero prediction error 的大小*,SoO 关心 *body-boundary 的 multisensory correlation 是否 fall 在 self-manifold 内*。
3. **Comparator model 是 ablation**:robotic 实证证明它 *under-specify* SoA。double comparator / contingency learner 才 minimal sufficient。
4. **Interoception 是当前 robot self 的最大瓶颈**。所有 extero/proprio 的 SoO/SoA emulation 都建立在"不饿不死的 substrate"上,缺 intero 一层,phenomenal selfhood 永远 gap。
5. **Computational psychiatry 的 embodied turn**:psychiatric subsyndrome 可以 map 到 layered architecture 中 precision/gain/noise 的 perturbation,Yamashita-Tani 是 proof of concept。
6. **Social frame modulates self**:SoA 在 HRI 中升高或降低取决于 task 是 joint-positive 还是 loss-avoidance。这把 *self* 从个体 property reframe 成 *dyadic/collective dynamic*。
7. **Ethics 的工程化前 question**:在 build self-endowed robot 之前要先决定我们想要的 *kinds* of self,以免造出一个 moral patient 而 regret。

---

## Reference links(精选)

- Paper 本体:https://doi.org/10.1126/scirobotics.adn2733
- Bongard, Zykov, Lipson (resilient self-modeling):https://doi.org/10.1126/science.1126495
- Asano, Okada, Inaba (Kenshiro):https://doi.org/10.1126/scirobotics.aaq0899
- Metta et al. (iCub):https://doi.org/10.1016/j.neunet.2010.08.010
- Lanillos, Pages, Cheng (mirror self-other):https://doi.org/10.3233/FAIA200368
- Lanillos & Cheng (sensorimotor contingencies):https://doi.org/10.1109/TCDS.2017.2650620
- Hoffmann et al. (robotic homunculus):https://doi.org/10.1109/TCDS.2017.2783894
- Hafner et al. (self-manifold):https://doi.org/10.3389/fnbot.2020.554578
- Yamashita & Tani (schizophrenia RNN):https://doi.org/10.1371/journal.pone.0037843
- Yamada et al. (fetal brain/body model):https://doi.org/10.1038/srep27893
- Blanke & Metzinger (minimal phenomenal selfhood):https://doi.org/10.1016/j.tics.2008.10.003
- Tsakiris (body-ownership top-down):https://doi.org/10.1016/j.neuropsychologia.2009.09.028
- Botvinick & Cohen (RHI):https://doi.org/10.1038/35816
- Armel & Ramachandran (bottom-up RHI):https://doi.org/10.1098/rspb.2003.2364
- Haggard, Clark, Kalogeras (intentional binding):https://doi.org/10.1038/nn825
- Navare et al. (SoJA, Fig.2):https://doi.org/10.1126/scirobotics.adj3665
- Roselli et al. (vicarious SoA):https://doi.org/10.1038/s41598-022-18107-7
- Ciardo et al. (loss-avoidance reduces SoA):https://doi.org/10.1016/j.cognition.2019.104109
- Kieliba et al. (third thumb plasticity):https://doi.org/10.1126/scirobotics.abd8101
- Seth & Tsakiris (beast machine):https://doi.org/10.1016/j.tics.2018.08.003
- Vinanzi et al. (robot ToM):https://doi.org/10.1098/rstb.2018.0032
- Moulin-Frier et al. (DAC-h3):https://doi.org/10.1109/TCDS.2018.2853146
- Pointeau & Dominey (narrative self seed):https://doi.org/10.3389/fnbot.2017.00027
- Scassellati (Cog ToM):https://doi.org/10.1023/A:1015588527845
- Prescott & Dominey (robot temporal self):https://doi.org/10.1098/rstb.2023.0415
- Friston (embodied inference):https://doi.org/10.15502/9783958570102
- Smith, Badcock, Friston (predictive coding clinical):https://doi.org/10.1111/pcn.13144
- David, Newen, Vogeley (comparator/SOA):https://doi.org/10.1016/j.concog.2008.06.004
- Frith & Done (alien control):https://doi.org/10.1017/S0033291700012400
- Vogeley et al. (PFC self-model):https://doi.org/10.1006/ccog.1999.0393
- Sharkey & Ziemke (mechanistic vs phenomenal embodiment):https://doi.org/10.1016/S1389-0417(01)00039-3
- Aubin et al. (embodied energy):https://doi.org/10.1038/s41586-021-04162-0
- Roels et al. (self-healing soft robots):https://doi.org/10.1016/j.materel.2022.100003
- Roncone et al. (peripersonal space):https://doi.org/10.1371/journal.pone.0163713
- Karapanagiotidis et al. (DMN mental time travel):https://doi.org/10.1016/j.neuroimage.2016.12.018
- Yu et al. (NeuroSLAM):https://doi.org/10.1007/s00422-019-00802-2
- Bossi et al. (resting state predicts robot attitude):https://doi.org/10.1126/scirobotics.abb6688
- Marchesi et al. (intentional stance humanoid):https://doi.org/10.3389/fpsyg.2019.00450
- Letheby & Gerrans (psychedelic ego dissolution):https://doi.org/10.1093/nc/nix016
- Metzinger (Ego Tunnel):https://www.basicbooks.com/books/the-ego-tunnel/
- Bryson (patiency not a virtue):https://doi.org/10.1007/s10676-017-9443-1
- Davey & Harrison (depression self on axis):https://doi.org/10.1038/s41398-022-01419-5
- Botvinick & Cohen RHI(完整版):https://doi.org/10.1038/35816

如果要把 paper 的工程化 agenda 真正 implement 出来,最 critical 的下一步在我看来是:**把 interoceptive channel(energy state + damage state + neuromodulatory signal)塞进 self-manifold 的 latent space**,让 robot 学到的 $z$ 不仅编码"我的身体形状",还编码"我的身体状态"。这才会让 self-manifold 接近 Damasio 的 *proto-self* 概念,而不是停在 sensorimotor layer。再往上一层挂 episodic memory + ToM + narrative,就构成了 paper Fig. 1 的完整 developmental trajectory 的 robotic instantiation。
