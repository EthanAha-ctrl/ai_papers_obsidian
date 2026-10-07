---
source_pdf: DEXTERITYGEN Foundation Controller for Unprecedented Dexterity.pdf
paper_sha256: 8658ddb91b39fcf47b8f1ebf32798be79ca5185d31208f6d9ea9154a23beab02
processed_at: '2026-08-03T20:37:28-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DexGen 用人话讲

## 一句话版

你想让机械手转笔、打螺丝、推注射器，可是人直接拿手柄teleop机械手时，物体"啪"就掉了——因为你看不见指尖接触、感受不到力、机械手跟人手长得也不一样。DexGen 就是在机械手和你的teleop指令之间插了一个"安全滤镜"，它把你那些会掉东西的蠢动作，自动**掰弯**成"还能稳住"的动作，同时尽量保留你想干的那个事。

---

## 生活类比：训练轮 + 滑雪板

会滑雪的人大概都有这个经验：初学者站着不动都会摔，但教练在前面拉你、后面扶你，你想往左倒他都给你拽回来。你只要给一个大致方向（"我想往左"），教练负责不让你摔倒，同时尽量顺着你。

DexGen 就是这个教练。你给它 coarse command（"我想让指尖往左上移"），它看看现在机械手的状态，从它脑子里那个"31.7年RL经验"的大库里找一组**既安全又能往左上移**的动作，然后发给机械手。

如果你给的command很危险（比如你手一抖让食指张开，物体要掉了），DexGen 会**直接 override 你**，让指尖像粘在物体上一样继续维持接触。Paper里把这个现象叫做 "magnetic effect"——指尖有磁力一样吸在物体上。

---

## 为什么这事之前没人做出来

Dexterous manipulation 一直是robotics里的老大难。之前的路线主要有两条，每条都有硬伤：

### 路线A：人teleop + imitation learning

人带着VR手套、看着屏幕，操纵机械手做事，记录数据训imitation policy。

**硬伤**：物体老掉。原因一堆：

- **Partial observability**：在手里转的时候你看不见接触点，不知道力往哪使。现在那点binary vibration的haptic feedback根本传不出复杂的接触几何。
- **Embodiment gap**：人手指软、有纹路、有指纹，机械手指硬、有棱角。同样的指尖位置，接触动力学完全不同。Paper里作者说他们早期实验发现"指尖形状稍微改一点，物体运动就完全变"。
- **Motion complexity**：Allegro Hand 16个DOF，任何一个DOF给歪一点，grasp就破了。
- **Force inaccuracy**：现在的teleop系统全是position control，你想加力只能靠"位置指令超过实际位置产生的误差"，这在force-sensitive任务里基本没用。

所以你让人teleop做"用注射器推液体"这种事，他连注射器都拿不稳，谈什么推液体。

### 路线B：Sim-to-real RL

在仿真里训一个policy，直接迁移到真机。

**硬伤**：

- **Sim-to-real gap**：仿真里的物理、传感器跟真实世界有差距。视觉policy更惨，Dextreme 为了训一个物体旋转policy，渲染了5M张图做domain randomization，烧钱烧算力。
- **Reward specification**：你想让RL学会"用螺丝刀拧螺栓"，reward怎么写？写不出来。接触丰富、长horizon任务的reward engineering是个黑洞。

### DexGen 的洞察

Paper的核心洞察就一句话：**RL擅长学低级motion primitive，人擅长给高级semantic意图，那就各干各的**。

- 用RL在仿真里大规模学一堆"dexterous primitive"（转、平移、regrasp），攒成数据集
- 用这些数据训一个生成模型 `p_θ(action | state)`，学到"什么动作在什么状态下是safe且effective的"
- 真机inference时，人给coarse command，模型把command **project** 到safe distribution上

这跟[Cheng et al. RSS 2024 Expressive Whole-Body Control](https://expressive-humanoid.github.io/) 在humanoid上的思路一脉相承——teleop command作为high-level prompt，底层用学到的prior去"稳住"。只是DexGen第一次把这个思路推到了finger-level的dexterous manipulation上。

---

## 三个关键技术决策，逐个人话讲

### 决策1：为什么用 Diffusion + Guided Sampling，而不是直接训policy

假设你直接训一个 `π(a | state, human_command)`，让神经网络同时学"什么时候听人的、什么时候override人"。这个multimodal learning问题超级难，因为safe action分布是multi-modal的——同一个state下可能有十种不同safe动作，但听人的command后只剩一种。

DexGen 走的是另一条路：**先学一个prior `p_θ(a|o)`**（不带human command），再用guided sampling把human command**后验注入**进去。

数学上，我们想sample的分布是：
$$\Delta x \sim p_\theta(\Delta x | o) \cdot \exp(-\text{Dist}(\Delta x, \Delta x_{\text{input}}))$$

- $p_\theta(\Delta x | o)$：在当前observation $o$ 下，safe & effective动作的prior分布
- $\exp(-\text{Dist})$：Boltzmann-style的能量函数，惩罚偏离human command $\Delta x_{\text{input}}$
- 乘起来：**在safe manifold上找最接近human意图的点**

实现上，每个diffusion reverse step，对denoised sample $\mu$ 做修正：
$$\mu \leftarrow \mu - \alpha \Sigma \nabla_{\Delta x} \text{Dist}(\Delta x, \Delta x_{\text{input}})$$

- $\alpha$：guidance strength，paper实验里这是关键超参，太小→人意图丢失，太大→unsafe action泄漏
- $\Sigma$：该diffusion step的variance
- $\nabla_{\Delta x} \text{Dist}$：distance对生成样本的梯度

直觉：diffusion prior像一个大盆地（safe action manifold），你的human command像一只手在盆地上推一个球。推得轻→球停在盆底附近，safe但意图不太准；推得重→球被推出盆地，危险。

这个思路最早在[Janner et al. ICML 2022 Planning with Diffusion](https://diffusion-planning.github.io/)里提出，DexGen是第一次把它用到dexterous manipulation的shared autonomy上。

### 决策2：为什么用 Keypoint 作为中间表示

DexGen 的架构是：
```
[state o] + [mode c] → Diffusion Model → Δx (keypoint motion)
                                        ↓
                          Inverse Dynamics MLP → a_t (joint target)
```

中间的 $\Delta x$ 是**finger keypoint的3D offset**，不是joint angle。

为什么这么搞？Paper试了 $K=4$（只用4个fingertip）效果差，$K=8$（PIP joint + fingertip）才行。

直觉算一笔账：

- Allegro Hand 16 DOF
- $K=4$ × 3D = 12D空间，**信息不够**，12D无法recover 16 DOF的全部动作
- $K=8$ × 3D = 24D，有冗余但够用

更深的原因：**keypoint space是人和机器人的共同语言**。人teleop通过retargeting给的本来就是fingertip position command，在keypoint space里做guided sampling天然对齐。如果用joint space，human command还要先过一道inverse kinematics，误差越积越多。

### 决策3：Anygrasp-to-Anygrasp 这个pretraining task 的妙处

这是整篇paper我最喜欢的设计。DexGen 不去训"用screwdriver""用syringe"这些task-specific policy。它训的是**grasp之间的transition**。

具体做法：

1. 对每个物体，用Grasp Analysis + RRT生成~100K个grasp configs，每个grasp = (finger joint pos $q$, object pose $p$)
2. RL rollout时随机初始化一个grasp，用k-NN找一个nearby grasp作goal
3. 达到goal后，再k-NN找下一个nearby grasp，持续走

为什么必须是nearby？Paper在Goal Dynamics里说：如果goal太远，RL会早早plateau，学不会。这本质是curriculum learning。

这个task为什么"对"？因为**所有in-hand manipulation本质都是grasp graph上的random walk**。用screwdriver拧螺丝这件事，可以decompose成：

```
pick up → reorient → regrasp到functional grasp → use
```

每一步都是grasp-to-grasp的transition。Paper的Figure 5画得特别清楚：dataset覆盖了grasp graph上的edge，DexGen学到了每个state下的action distribution（紫色阴影），long-horizon task就是sequential prompting沿着这个graph走（紫色三角）。

这思路跟作者前作[Yin et al. RSS 2023 Rotating without seeing](https://rotating-without-seeing.github.io/)一脉相承，只是DexGen把离散的primitive call升级成了连续的keypoint motion prompting，granularity大幅提升。

---

## Grasp Generation 的简化技巧

标准force closure要同时满足force balance和torque balance，但paper用了个简化版：

$$\min_{\{f_i\}} \left\|\sum_i f_i n_i\right\|^2 \quad \text{s.t.} \quad \forall i, f_i \geq 0, \exists i, f_i = 1$$

- $f_i$：第 $i$ 个contact点的法向力大小
- $n_i$：第 $i$ 个contact点的法向量
- 目标：找一组非平凡力（至少一个 $f_i=1$ 保证非零解）使net force接近0
- 这只是force closure的**必要条件**（缺torque），但够用且高效

那个 $\exists i, f_i=1$ 约束不好直接optimize，所以paper把它decompose成 $n$ 个subproblem（分别固定 $f_1=1, f_2=1, \ldots$），取最小值。这是工程上的小聪明。

---

## RL Reward 里的trick

$$r = w_{\text{goal}} r_{\text{goal}} + w_{\text{style}} r_{\text{style}} + w_{\text{merge}} r_{\text{merge}}$$

最有意思的是 $r_{\text{style}}$：
$$r_{\text{style}} = \sum_i \alpha_i \|\dot{x}_{\text{tip}}^i\|$$

- $\dot{x}_{\text{tip}}^i$：第 $i$ 个fingertip的velocity
- 调 $\alpha_i$ 正负可以得到"快"或"慢"风格的policy

Paper在Appendix C里说，他们训了多个不同 $w_{\text{style}}, w_{\text{reg}}$ 的policy，有的快有的慢，一起收集data。这样DexGen的prior覆盖了不同tempo的manipulation，real world里不管人teleop快还是慢都能align。

这是个数据多样性的工程trick，朴素但有效。

---

## Dataset 规模

- **$10^{10}$ transitions** = 31.7 年 real-world experience
- 生成耗时：300 GPU hours（rollout RL policies）
- 训练DexGen：96 GPUs × 3 days

Paper诚实地承认：人类dexterity是millions of years evolution的产物，$10^{10}$ transitions远不够。但zero-shot已经能unlock之前做不到的skill。

---

## 实验结果人话讲

### Simulation：让蠢policy变聪明

构造两种"蠢policy"：

- $\pi_{\text{noisy}}(a|s) = \pi_{\text{exp}}(a|s) + \mathcal{U}(-\alpha, \alpha)$：专家policy加uniform noise
- $\pi_{\text{slow}}(a|s) = \mathcal{U}(0, \alpha) \pi_{\text{exp}}(a|s)$：slowdown版

结果：DexGen把holding duration提升 **10-100x**，甚至让"绝大多数action是noise"的policy也能succeed。

这说明DexGen的prior极强，能从mostly-noise里把意图recover出来。

### Real World：teleop从全失败到50-80%成功率

| Task | Teleop SR | Teleop+DexGen SR |
|------|-----------|------------------|
| Reorient Large (Up) | 0/20 | 12/20 |
| Reorient Small (Up) | 0/20 | 13/20 |
| Reorient Large (Down) | 0/20 | 10/20 |
| Reorient Small (Down) | 0/20 | 9/20 |
| Func Grasp | 0/10 | 7/10 |
| Func Grasp (Horizontal) | 1/10 | 6/10 |
| Regrasp (Ball) | 0/10 | 5/10 |
| Regrasp (Cylinder) | 0/10 | 5/10 |

Raw teleop几乎全失败（<5% TTF），DexGen提升到50-80% SR。

**关键insight**：人teleop单独**完全不能**做dexterous manipulation。DexGen不是"锦上添花"，是"从无到有"。

### Long-horizon Tool Use 首次实现

| Screwdriver Stage | SR |
|-------|----|
| Reorient | 16/20 |
| Regrasp | 11/20 |
| Align | 5/20 |
| Use | 3/20 |

| Syringe Stage | SR |
|-------|----|
| Reorient | 15/20 |
| Regrasp | 9/20 |
| Use | 4/20 |

每stage都有合理SR，但chaining起来衰减严重。Paper坦承这仍是open challenge，但首次实现这种long-horizon dexterous tool use teleop本身就是突破。

---

## Limitations 和我的直觉

### Limit 1: 没有Touch Sensing

DexGen只用joint angle proprioception隐式推断force（通过PD control error）。这对fine-grained任务不够。

我的直觉：joint error → force的mapping在contact stiffness高时信噪比低。tactile sensor能直接给contact geometry，这对inverse dynamics model应该有显著提升。参考[AnyRotate CoRL 2024](https://anyrotate.github.io/)显示sim-to-real touch transfer可行。

### Limit 2: 没有Vision

当前hand-eye coordination全靠人teleoperator的眼睛。Screwdriver对准bolt这种任务，机器人端没有visual servoing，align阶段SR只有5/20。

我的直觉：vision应该放high-level还是low-level？如果放low-level，DexGen就要从proprioception-only变成visuo-tactile-conditioned，diffusion prior的dimensionality爆炸。如果放high-level，需要VLA-style的vision-to-motion policy给 $\Delta x_{\text{input}}$，DexGen只做safety projection。后者更modular，我赌这个方向。可以参考[RT-2](https://robotics-transformer2.github.io/)或[OpenVLA](https://openvla.github.io/)这种VLA做high-level。

### Limit 3: Zero-shot Sim-to-Real

Paper没做real-world finetuning，纯zero-shot deployment。考虑到sim-to-real gap，这已经很惊人。但[DEFT CoRL 2023](https://deft-rl.github.io/)显示real-world finetuning能显著提升，DexGen + finetuning应该是自然next step。

---

## 更广的联想

### 跟Diffusion Policy的区别

[Diffusion Policy (Chi et al. RSS 2023)](https://diffusion-policy.cs.columbia.edu/)也用diffusion model做control，但DexGen的关键区别是 **guided sampling for shared autonomy**。Diffusion Policy是end-to-end imitation，DexGen是prior + online guidance。

### 跟Shared Autonomy传统工作的区别

传统shared autonomy [Reddy et al. RSS 2018](https://sites.google.com/view/shared-autonomy-rl)假设离散goal set，做intent inference。DexGen在 **连续high-DOF action space** 做 shared autonomy，用diffusion prior替代离散goal inference。这是shared autonomy的新范式。

[Yoneda et al. RSS 2023 To the noise and back](https://diffusion-shared-autonomy.github.io/)是最接近的工作，也用diffusion做shared autonomy，但DexGen处理的是更general的dexterous manipulation。

### 跟Foundation Model for Robotics的对比

[RT-1](https://robotics-transformer1.github.io/), [RT-2](https://robotics-transformer2.github.io/), [Octo](https://octo-models.github.io/), [OpenVLA](https://openvla.github.io/)都是end-to-end VLA foundation model，condition在language上。DexGen走了不同的路：**low-level foundation controller，condition在continuous motion command上**。这跟[ALOHA Unleashed CoRL 2024](https://aloha-unleashed.github.io/)的action chunking + diffusion思路有呼应，但DexGen的pretraining data来自RL而非human demo，scale上有优势。

### 跟Active Inference的哲学呼应

DexGen的guided sampling本质是 **minimize surprise**：prior期望维持contact（safe manifold），human command是observation，posterior是两者折衷。这跟Friston的active inference框架[Friston 2010](https://www.nature.com/articles/nn.2756)有结构相似性，虽然paper没提。

---

## 我猜的Follow-up方向

1. **VLM as high-level policy**：用GPT-4V/Gemini看scene给semantic motion command $\Delta x_{\text{input}}$，DexGen做low-level safety。完全替代human teleop。
2. **Tactile-conditioned DexGen**：加[DIGIT](https://digit.ml/)或[GelSight](https://gelsight.com/) tactile input，diffusion prior学visuo-tactile-action joint distribution。
3. **Hierarchical DexGen**：当前mode conditioning只有default/precision。可以学一个discrete latent（VQ-VAE风格）自动discover manipulation modes。
4. **Real-world finetuning with human preference**：DexGen给prior，human teleop时收集preference data，RLHF/DPO finetune diffusion prior。
5. **Bimanual DexGen**：当前单手，extend到[Bimanual Dexterity Shaw et al. CoRL 2024](https://bimanual-dexterity.github.io/)是obvious next step。
6. **Diffusion prior as world model**：inverse dynamics model已经学了state-motion-action mapping，加forward model可以做planning in motion space。

---

## Reference Links

- **DexGen project page**: https://zhaohengyin.github.io/dexteritygen
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **Planning with Diffusion**: https://diffusion-planning.github.io/
- **Rotating without seeing**: https://rotating-without-seeing.github.io/
- **Dextreme**: https://dextreme.org/
- **OpenAI Dactyl**: https://openai.com/research/learning-dexterity
- **AnyTeleOp**: https://anyteleop.github.io/
- **DEFT**: https://deft-rl.github.io/
- **To the noise and back**: https://diffusion-shared-autonomy.github.io/
- **ALOHA Unleashed**: https://aloha-unleashed.github.io/
- **OpenVLA**: https://openvla.github.io/
- **Octo**: https://octo-models.github.io/
- **Bimanual Dexterity**: https://bimanual-dexterity.github.io/
- **AnyRotate**: https://anyrotate.github.io/
- **FiLM conditioning**: https://arxiv.org/abs/1709.07871
- **DDPM**: https://arxiv.org/abs/2006.11239
- **DDIM**: https://arxiv.org/abs/2010.02502
- **IsaacGym**: https://developer.nvidia.com/isaac-gym
- **Expressive Whole-Body Control**: https://expressive-humanoid.github.io/

---

## 一句话收尾

DexGen 把dexterous manipulation从"end-to-end learn everything"转向"**learn safe action prior, prompt with coarse intention**"。这跟LLM的prompt engineering哲学一致：foundation model提供prior，human/policy提供prompt，posterior是两者融合。Diffusion guided sampling让"在safe manifold上project任意command"变得tractable，keypoint作为中间representation让human-robot对齐natural，Anygrasp-to-Anygrasp让pretraining task足够general又足够learnable。作为一个"initial attempt towards foundational low-level controller"，它demonstrate了unprecedented dexterity（syringe, screwdriver teleop首次实现），这本身就是strong proof of concept。

---

# DexterityGen (DexGen) 深度解析

## 一、Core Thesis：把 dexterous manipulation 拆成两层

DexGen 的核心 thesis 可以一句话概括：**dexterous manipulation 的难点不在"想做什么"，而在"怎么不把东西弄掉"**。人类 teleoperator 知道要把 screwdriver 转过来对准 bolt，但一旦涉及 contact-rich 的 finger-level control，object 就会从手里掉下来——因为 partial observability（看不到 contact）、embodiment gap（human finger vs robot finger 的几何/摩擦差异）、force inaccuracy（position control 间接产生 force）。

所以 DexGen 把问题拆成两层：
- **High-level**：human teleoperation / imitation policy 给 coarse motion command（"我想把指尖往这个方向移"）
- **Low-level**：DexGen 这个 foundation controller 把 unsafe 的 command **project** 到一个 learned safe action distribution 上，maximally preserve intention while guaranteeing safety

这个分层思路在 locomotion 里（teleoperation command 作为 high-level prompt）已有先例 [Cheng et al. RSS 2024](https://expressive-humanoid.github.io/)，但 extend 到 finger-level dexterous manipulation 是这篇 paper 首次。

---

## 二、为什么用 Diffusion + Guided Sampling（而不是 RL policy 或 VAE）

这是 build intuition 的关键。DexGen 没有直接训一个 `π(a|o)` 的 policy，而是训了一个 generative model `p_θ(a|o)`，然后 inference 时用 guided sampling。

### 2.1 Diffusion Model 基础（DDPM）

前向过程（加噪）：
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t$$
- $x_0$：原始 data sample（这里指 keypoint motion $\Delta x$）
- $x_t$：第 $t$ 步加噪后的 sample
- $\alpha_t$：noising schedule，控制每步加多少 noise
- $\epsilon_t \sim \mathcal{N}(0, I)$：Gaussian noise
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$：累积 product，随 $t \to \infty$ 趋近 0，即 $x_t \to \mathcal{N}(0, I)$

反向过程（去噪）：
$$p(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$
- $\mu_\theta$：神经网络预测的 mean（即预测 $x_0$）
- $\sigma_t^2$：variance schedule

### 2.2 Guided Sampling 的 magic

这是 DexGen 能 work 的核心机制。我们要 sample 的 target distribution 是：
$$\Delta x \sim p_\theta(\Delta x | o) \cdot \exp(-\text{Dist}(\Delta x, \Delta x_{\text{input}}))$$

直觉解读：
- $p_\theta(\Delta x | o)$：在当前 state $o$ 下，"safe & effective" action 的分布（从 31.7 年 RL 数据中学来）
- $\exp(-\text{Dist})$：一个能量函数，惩罚生成样本偏离 human command
- 两者乘积 = **在 safe manifold 上，找最接近 human command 的点**

实现上，在每个 diffusion reverse step，对 denoised sample $\mu$ 做修正：
$$\mu \leftarrow \mu - \alpha \Sigma \nabla_{\Delta x} \text{Dist}(\Delta x, \Delta x_{\text{input}})$$
- $\alpha$：guidance strength（关键超参，太小 human 意图丢失，太大 unsafe action 泄漏）
- $\Sigma$：该 diffusion step 的 variance
- $\nabla_{\Delta x} \text{Dist}$：distance function 对 $\Delta x$ 的 gradient

距离函数用简单的 L2：
$$\text{Dist}(\Delta x, \Delta x_{\text{input}}) = \sum_{i=1}^T \|\Delta x_i - \Delta x_{\text{input}}\|^2$$
- $T$：future horizon（paper 用 $T=2$，即 0.2s）
- $\Delta x_i$：第 $i$ 步未来 keypoint offset
- $\Delta x_{\text{input}}$：human teleop 给的 fingertip offset command

### 2.3 为什么这个比直接 RL policy 好？

如果直接训 `π(a|o)`，human command 只能作为 observation 拼进去，policy 需要学会"什么时候听 human，什么时候 override"。这是非常难的 multi-modal learning 问题。

DexGen 的 guided sampling 本质上是 **projection operator**：先把 unsafe command 拉到 safe distribution 附近，再让 diffusion 的 prior 把它"吸附"到 manifold 上。这就是 paper 里描述的 "magnetic effect"——指尖像粘在 object 上一样，因为 diffusion prior 学到的是"维持 contact"的 mode。

参考 [Janner et al. ICML 2022 "Planning with Diffusion"](https://diffusion-planning.github.io/) 首次把 guided sampling 引入 sequential decision making。

---

## 三、Anygrasp-to-Anygrasp：Pretraining Task 设计

这是整个 pipeline 最有 insight 的部分。DexGen 不直接训"用 screwdriver"这种 task-specific policy，而是训一个 **task-agnostic 的 grasp transition** 能力。

### 3.1 Task 定义

对每个 object：
1. 用 Grasp Analysis + RRT 生成 ~100K 个 grasp configurations，每个 grasp = (finger joint position $q$, object pose $p$)
2. RL rollout 时，随机初始化一个 grasp，用 k-NN 找一个 **nearby** grasp 作为 goal
3. Reach goal 后，再 k-NN 找下一个 nearby grasp，持续 rollout

为什么必须 nearby？paper 在 Goal Dynamics 里解释：如果 goal 太远，RL plateau 很早，学不会。这是 curriculum learning 的思想。

### 3.2 Grasp Generation 的 force closure 简化

标准 force closure 需要同时满足 force 和 torque balance，但 paper 用了一个简化版（Algorithm 5 的替代）：
$$\min_{\{f_i\}} \left\|\sum_i f_i n_i\right\|^2 \quad \text{s.t.} \quad \forall i, f_i \geq 0, \exists i, f_i = 1$$
- $f_i$：第 $i$ 个 contact point 的法向力大小
- $n_i$：第 $i$ 个 contact point 的法向量
- 目标：找一组非平凡力（至少一个 $f_i=1$ 保证非零解）使 net force 接近 0
- 这是 **force closure 的必要条件**（不含 torque），但够用且高效

实现技巧：$\exists i, f_i=1$ 这个约束不好直接 optimize，所以 decompose 成 $n$ 个 subproblem（分别固定 $f_1=1, f_2=1, \ldots$），取最小值。

### 3.3 为什么 Anygrasp-to-Anygrasp 是"对的" pretraining task

In-hand manipulation 的本质就是 **grasp 之间的 transition**。任何 tool use 任务（screwdriver, syringe）都可以 decompose 成一系列 grasp transition：
- Pick up → reorient → regrasp to functional grasp → use

Figure 5 把这个画得很清楚：dataset 覆盖了 grasp graph 上的 edge，DexGen 学到了每个 state 下的 action distribution（紫色阴影），long-horizon task 就是 sequential prompting 沿着这个 graph 走（紫色三角形）。

这跟 [Yin et al. RSS 2023 "Rotating without seeing"](https://rotating-without-seeing.github.io/)（同一作者前作）的思路一脉相承，但 DexGen 把离散 primitive call 升级成了连续 keypoint motion prompting，granularity 大幅提升。

---

## 四、Architecture 细节

### 4.1 双模块设计

```
[State o] + [Mode c] → [Diffusion Model] → Δx (keypoint motion)
                                              ↓
                              [Inverse Dynamics MLP] → a_t (joint target)
```

- **Diffusion Model**：UNet-based，3 blocks encoder + 3 blocks decoder，hidden dim 768，FiLM conditioning [Perez et al. AAAI 2018](https://arxiv.org/abs/1709.07871) 在 middle layers 注入 state/mode embedding
- **Inverse Dynamics**：residual MLP，输出 $\mathcal{N}(\tilde{q}_t | \text{state}, \Delta x)$

### 4.2 为什么用 keypoint 而非 joint action 作为中间表示

这是非常关键的设计 choice。Paper 试了 $K=4$（fingertips only）和 $K=8$（PIP joints + fingertips），$K=4$ 表现差，inverse dynamics loss 大。

直觉：
- Allegro Hand 16 DOF
- $K=4$ keypoints × 3D = 12D 空间，**信息不足**，无法 span 完整 action space
- $K=8$ × 3D = 24D，有冗余但能 recover 16 DOF

更深层原因：**keypoint space 是 human-robot 的共同语言**。Human teleop 通过 retargeting 给的是 fingertip position command，在 keypoint space 做 guided sampling 天然 align。如果用 joint space，human command 还要经过 inverse kinematics，误差累积。

### 4.3 Mode Conditioning

输入是一个 one-hot vector，主要 label 是 "default"（绝大多数 data），特殊场景如 screwdriver 用 "precision rotation" mode。

这个设计解决一个 subtle 问题：dataset 里 99% 的 action 都是"hold object"，如果不给 "release" mode，模型很难被 prompt 松开 object。但 paper 发现实际中 **直接 disable DexGen 就能 release**，不需要专门的 release mode——这暗示 diffusion prior 的"hold" bias 非常强。

### 4.4 Inference 速度

- DDIM sampler，8-12 steps
- 27ms per step → 37Hz 推理
- 控制频率 10Hz（每步生成 0.2s future motion，即 $T=2$ steps × 0.1s）

---

## 五、RL Training 细节

### 5.1 Reward Function

$$r = w_{\text{goal}} r_{\text{goal}} + w_{\text{style}} r_{\text{style}} + w_{\text{reg}} r_{\text{reg}}$$

**Goal reward**：
$$r_{\text{goal}} = \exp(-\alpha_{\text{pos}} \|p_{\text{obj}} - p_{\text{target}}\|^2 - \alpha_{\text{orn}} d(R_{\text{obj}}, R_{\text{target}})) - \alpha_{\text{hand}} \|q - q_{\text{target}}\|^2 + \alpha_{\text{bonus}} \mathbf{1}(\text{goal achieved})$$
- $p_{\text{obj}}, R_{\text{obj}}$：object 当前 position 和 rotation
- $p_{\text{target}}, R_{\text{target}}$：target grasp 的 object pose
- $d(R_1, R_2)$：rotation 之间的距离（可能是 geodesic）
- $q, q_{\text{target}}$：finger joint position 和 target grasp 的 finger configuration
- $\mathbf{1}(\cdot)$：goal achieved 的 indicator bonus

**Regularization**：
$$r_{\text{reg}} = -\alpha_{\text{work}} |\dot{q}^T \tau| - \alpha_{\text{action}} \|a\|^2 - \alpha_{\text{tau}} \|\tau\|^2$$
- $\dot{q}^T \tau$：mechanical work（joint velocity × torque），惩罚做功
- $\|a\|^2$：action magnitude penalty
- $\|\tau\|^2$：torque penalty

**Style reward**（关键的多样性 trick）：
$$r_{\text{style}} = \sum_i \alpha_i \|\dot{x}_{\text{tip}}^i\|$$
- $\dot{x}_{\text{tip}}^i$：第 $i$ 个 fingertip 的 velocity
- 调 $\alpha_i$ 的正负号可以得到"快"或"慢"风格的 policy

### 5.2 为什么需要 diverse rewards

Paper 在 Appendix C 强调：训多个不同 $w_{\text{style}}, w_{\text{reg}}$ 的 policy，有的快有的慢，一起收集 data。这样 DexGen 的 prior distribution 覆盖不同 tempo 的 manipulation，real world 里不管 human teleop 快还是慢都能 align。

### 5.3 PPO 设置

- IsaacGym，8192 parallel environments
- Asymmetric actor-critic：actor 只看 proprioception + relative goal transform，critic 看 full state
- MLP [1024, 512, 512, 256, 256]
- LR 5e-4, batch 8192, clip 0.2, $\gamma=0.99$, GAE $\tau=0.95$

### 5.4 Domain Randomization

| Component | Range |
|-----------|-------|
| Object mass | [0.03, 0.25] kg |
| Object friction | [0.5, 1.2] |
| Object shape scale | ×U(0.95, 1.05) |
| Hand initial joint noise | [-0.05, 0.05] |
| Hand friction | [0.5, 1.2] |
| PD P gain | ×U(0.8, 1.1) |
| PD D gain | ×U(0.7, 1.2) |
| Random force | scale 1.0/2.0, prob 0.2, decay 0.99 every 0.1s |
| Joint obs noise (white) | $\mathcal{N}(0, 0.025)$ |
| Joint obs noise (episode) | persistent per-episode bias |

注意 wrist pose 也随机化——这让 policy 学会对抗 gravity，exhibit prehensile manipulation（不只是 palm-down 的平移）。

---

## 六、Dataset 规模与 Compute

- **$10^{10}$ transitions** = 31.7 年 real-world experience
- 生成耗时：300 GPU hours（rollout RL policies）
- 训练 DexGen：96 GPUs，15 epochs，3 days
- 这规模在 dexterous manipulation 里是 unprecedented 的

Paper 诚实地承认：human dexterity 是 millions of years evolution 的产物，$10^{10}$ transitions 远不够。但 zero-shot 已经能 unlock 之前做不到的 skill。

---

## 七、实验结果解读

### 7.1 Simulation：Noisy Policy Recovery

构造两种 suboptimal policy：
- $\pi_{\text{noisy}}(a|s) = \pi_{\text{exp}}(a|s) + \mathcal{U}(-\alpha, \alpha)$（加 uniform noise）
- $\pi_{\text{slow}}(a|s) = \mathcal{U}(0, \alpha) \pi_{\text{exp}}(a|s)$（slowdown）

结果：DexGen 把 holding duration 提升 **10-100x**，甚至让"绝大多数 action 是 noise"的 policy 也能 succeed。

Guidance strength $\alpha$ 的 ablation：有 sweet spot。太小 → 安全但 human intent 丢失（不 reach goal）；太大 → unsafe action 泄漏（duration 下降）。

### 7.2 Real World：Teleoperation Tasks

| Task | Teleop SR | Teleop+DexGen SR |
|------|-----------|------------------|
| Reorient Large (Up) | 0/20 | 12/20 (75% TTF) |
| Reorient Small (Up) | 0/20 | 13/20 (79%) |
| Reorient Large (Down) | 0/20 | 10/20 (63%) |
| Reorient Small (Down) | 0/20 | 9/20 (58%) |
| Func Grasp | 0/10 | 7/10 (87%) |
| Func Grasp (Horizontal) | 1/10 | 6/10 (80%) |
| Regrasp (Ball) | 0/10 | 5/10 (78%) |
| Regrasp (Cylinder) | 0/10 | 5/10 (74%) |

Raw teleop 几乎全失败（<5% TTF），DexGen 提升到 50-80% SR。这证明 **human teleop 单独完全无法做 dexterous manipulation**，DexGen 不是"锦上添花"，是"从无到有"。

### 7.3 Long-horizon Tool Use 首次实现

Screwdriver 阶段分解：
| Stage | SR |
|-------|----|
| Reorient | 16/20 |
| Regrasp | 11/20 |
| Align | 5/20 |
| Use | 3/20 |

Syringe：
| Stage | SR |
|-------|----|
| Reorient | 15/20 |
| Regrasp | 9/20 |
| Use | 4/20 |

每个 stage 都有合理 SR，但 chaining 起来 SR 衰减严重（3/20, 4/20）。Paper 坦承这仍是 open challenge，但首次实现这种 long-horizon dexterous tool use teleoperation。

---

## 八、Limitations & 我的 Intuition

### 8.1 Touch Sensing 缺失
DexGen 只用 joint angle proprioception 隐式推断 force（通过 PD control error）。这对 fine-grained 任务（syringe 推液体的力道）不够。Paper 提到未来加 tactile sensing，参考 [AnyRotate CoRL 2024](https://anyrotate.github.io/) 显示 sim-to-real touch transfer 可行。

直觉：joint error → force 的 mapping 在 contact stiffness 高时信噪比低，tactile sensor 能直接给 contact geometry，这对 inverse dynamics model 应该有显著提升。

### 8.2 Vision 缺失
当前 hand-eye coordination 全靠 human teleoperator 的眼睛。Screwdriver 对准 bolt 这种任务，机器人端没有 visual servoing，align 阶段 SR 只有 5/20。

直觉问题：vision 应该放 high-level 还是 low-level？如果放 low-level，DexGen 就要从 proprioception-only 变成 visuo-tactile-conditioned，diffusion prior 的 dimensionality 爆炸。如果放 high-level，需要 VLA-style 的 vision-to-motion policy 给 $\Delta x_{\text{input}}$，DexGen 只做 safety projection。后者更 modular，我赌这个方向。

### 8.3 Zero-shot Sim-to-Real
Paper 没做 real-world finetuning，纯 zero-shot deployment。考虑到 sim-to-real gap，这已经很惊人。但 [DEFT CoRL 2023](https://deft-rl.github.io/) 显示 real-world finetuning 能显著提升，DexGen + finetuning 应该是自然 next step。

---

## 九、更广的 Connection

### 9.1 与 Diffusion Policy 的关系
[Diffusion Policy (Chi et al. RSS 2023)](https://diffusion-policy.cs.columbia.edu/) 也是用 diffusion model 做 control，但 DexGen 的关键区别是 **guided sampling for shared autonomy**。Diffusion Policy 是 end-to-end imitation，DexGen 是 prior + online guidance。

### 9.2 与 Shared Autonomy 的关系
传统 shared autonomy [Reddy et al. RSS 2018](https://sites.google.com/view/shared-autonomy-rl) 假设离散 goal set，做 intent inference。DexGen 在 **连续 high-DOF action space** 做 shared autonomy，用 diffusion prior 替代离散 goal inference。这是 shared autonomy 的新范式。

[Yoneda et al. RSS 2023 "To the noise and back"](https://diffusion-shared-autonomy.github.io/) 是最接近的工作，也用 diffusion 做 shared autonomy，但 DexGen 处理的是更 general 的 dexterous manipulation（不只是 goal-reaching）。

### 9.3 与 Foundation Model for Robotics 的关系
[RT-1](https://robotics-transformer1.github.io/), [RT-2](https://robotics-transformer2.github.io/), [Octo](https://octo-models.github.io/), [OpenVLA](https://openvla.github.io/) 都是 end-to-end VLA foundation model，condition 在 language 上。DexGen 走了不同的路：**low-level foundation controller，condition 在 continuous motion command 上**。这跟 [ALOHA Unleashed CoRL 2024](https://aloha-unleashed.github.io/) 的 action chunking + diffusion 思路有呼应，但 DexGen 的 pretraining data 来自 RL 而非 human demo，scale 上有优势。

### 9.4 与 Active Inference / Cybernetics 的哲学呼应
DexGen 的 guided sampling 本质是 **minimize surprise**：prior 期望维持 contact（safe manifold），human command 是 observation，posterior 是两者折衷。这跟 Friston 的 active inference 框架 [Friston 2010](https://www.nature.com/articles/nn.2756) 有结构相似性，虽然 paper 没提。

---

## 十、可能的 Follow-up 方向（我的猜测）

1. **VLM as high-level policy**：用 GPT-4V/Gemini 看 scene 给 semantic motion command $\Delta x_{\text{input}}$，DexGen 做 low-level safety。完全替代 human teleop。
2. **Tactile-conditioned DexGen**：加 [DIGIT](https://digit.ml/) 或 [GelSight](https://gelsight.com/) tactile input，diffusion prior 学 visuo-tactile-action joint distribution。
3. **Hierarchical DexGen**：当前 mode conditioning 只有 default/precision。可以学一个 discrete latent（VQ-VAE 风格）自动 discover manipulation modes。
4. **Real-world finetuning with human preference**：DexGen 给 prior，human teleop 时收集 preference data，RLHF/DPO finetune diffusion prior。
5. **Bimanual DexGen**：当前单手，extend 到 [bimanual dexterity Shaw et al. CoRL 2024](https://bimanual-dexterity.github.io/) 是 obvious next step。
6. **Diffusion prior as world model**：inverse dynamics model 已经学了 state-motion-action mapping，加 forward model 可以做 planning in motion space。

---

## 参考 Links

- **Project page**: https://zhaohengyin.github.io/dexteritygen
- **Diffusion Policy (相关方法)**: https://diffusion-policy.cs.columbia.edu/
- **Planning with Diffusion (guided sampling 灵感)**: https://diffusion-planning.github.io/
- **Rotating without seeing (同作者前作)**: https://rotating-without-seeing.github.io/
- **Dextreme (sim-to-real dexterous RL baseline)**: https://dextreme.org/
- **OpenAI Dactyl**: https://openai.com/research/learning-dexterity
- **AnyTeleOp**: https://anyteleop.github.io/
- **DEFT (real-world finetuning)**: https://deft-rl.github.io/
- **To the noise and back (diffusion shared autonomy)**: https://diffusion-shared-autonomy.github.io/
- **ALOHA Unleashed**: https://aloha-unleashed.github.io/
- **OpenVLA**: https://openvla.github.io/
- **Octo**: https://octo-models.github.io/
- **Bimanual Dexterity**: https://bimanual-dexterity.github.io/
- **AnyRotate (touch-based rotation)**: https://anyrotate.github.io/
- **FiLM conditioning**: https://arxiv.org/abs/1709.07871
- **DDPM**: https://arxiv.org/abs/2006.11239
- **DDIM**: https://arxiv.org/abs/2010.02502
- **IsaacGym**: https://developer.nvidia.com/isaac-gym
- **LEAP Hand (硬件参考)**: https://leap-hand.com/
- **Expressive Whole-Body Control (humanoid teleop prompt)**: https://expressive-humanoid.github.io/

---

## 总结：DexGen 的核心贡献是什么

DexGen 的真正贡献是把 dexterous manipulation 从"end-to-end learn everything"转向"**learn safe action prior, prompt with coarse intention**"。这跟 LLM 的 prompt engineering 哲学一致：foundation model 提供 prior，human/policy 提供 prompt，posterior 是两者融合。

技术上，**diffusion guided sampling** 是这个 framework 的 enabler——它让"在 safe manifold 上 project 任意 command"变得 tractable 且 differentiable。**keypoint 作为中间 representation** 让 human-robot 对齐 natural。**Anygrasp-to-Anygrasp** 让 pretraining task 足够 general 又足够 learnable。

Limitations 也很诚实：no touch, no vision, zero-shot only。但作为一个 "initial attempt towards foundational low-level controller"，它 demonstrate 了 unprecedented dexterity（syringe, screwdriver teleop 首次实现），这本身就是 strong proof of concept。
