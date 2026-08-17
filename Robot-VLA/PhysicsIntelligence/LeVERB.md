---
source_pdf: LeVERB.pdf
paper_sha256: 472a1c62b252076a432679a9556bc2fb46e0ae2447f3fed46c0c86116f1841a0
processed_at: '2026-08-05T14:34:46-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞，Karpathy，我用大白话再过一遍，顺便把一些之前没展开的 intuition 也聊透。

---

## 一句话版本

LeVERB 做的事情就一句话：**让人形机器人能听懂你说"去前面那把绿椅子坐下"，然后它自己看着摄像头画面走过去、转身、坐下，整个过程在真机上 zero-shot 跑通**。

听起来好像没啥了不起，但你想想——这事在 humanoid 上之前没人真的做出来过。要么只能听语言没眼睛（LangWBC），要么只能看图像但动作僵硬（NaVILA），要么干脆要人遥操作（OmniH2O）。LeVERB 是第一个把 vision、language、whole-body dynamics 三者打通的工作。

---

## 为什么这事难

你可能会想，manipulation 领域 RT-2、π0 不早就 vision-language-action 了吗？搬到 humanoid 不就行了？

问题在于 humanoid 的本质区别：它是**高维非线性动态系统**。桌面上机械臂是 quasi-static，你给它一个 end-effector pose，PD 控制器就能跟。humanoid 有 floating base、12+ 个 joint、还有双脚跟地面的接触 switching——你给它一个"target pose"，它可能直接摔倒。

更核心的问题是：**什么样的"指令格式"适合 humanoid？**

- 如果太底层（每帧 joint torque），VLA 根本学不动，维度太高
- 如果太高层（base velocity + end-effector pose），覆盖不了 sitting、reaching 这种 whole-body coordinated motion
- 中间层怎么设计？手工设计一定有盲区

LeVERB 的 insight：**让模型自己学一个 latent verb vocabulary**。不是人去定义"前进"、"坐下"是什么样子，而是用 CVAE 从数据里学一个 256 维的 latent space，这个 space 既编码了 vision-language 的语义，又编码了 whole-body motion 的 style。然后 RL 训出来的 low-level controller 负责把 latent verb 翻译成 joint command。

---

## 数据怎么来的（这是我觉得最聪明的部分）

训 VLA 要数据，humanoid 的视觉演示数据几乎没有。teleop 太慢，dynamic rollout 是 chicken-and-egg。LeVERB 的解法很 hacky 但很 effective：

**只回放 kinematics，不做 dynamics**。

具体说：你从 AMASS/LAFAN 拿到 retargeted humanoid motion（就是一堆 joint position 轨迹），在 IsaacSim 里**强制让机器人 kinematically 跟着走**（不经过物理控制器，直接 set pose），同时用 ray-tracing 渲染出 photorealistic 的画面。然后在周围随机摆物体、随机材质、随机光照、随机相机角度，一条 motion 重渲染 100 次，配上一句语言指令。

154 条 trajectory × 100 randomization = 17.1 小时 vision-language 数据。再加 460 条 language-only（白噪声图像）补足 motion 多样性，凑到 ~20 小时。

**为什么 kinematic replay 够用？** 因为 VLA 只需要理解 task-level semantics——"画面里前面有把椅子，指令说走过去坐"——它不需要知道脚底接触力是多少。dynamic execution 交给 LeVERB-A 来做。所以训练数据可以是"假"的，只要视觉和语言语义是真的就行。

这个思路跟 manipulation 领域用 scripted demonstrator 生成 grasping data 是一个 lineage，但更激进——humanoid 的 dynamic 比 arm 复杂 10 倍，他们居然敢 bypass。

---

## 架构：Kahneman 的 System 1 / System 2 借喻

作者引用 Kahneman 的双系统理论来类比：

- **System 2 (LeVERB-VL)**：慢思考，10Hz，吃 vision + language，输出 latent verb z。类似人脑皮层处理语言和视觉推理。
- **System 1 (LeVERB-A)**：快反应，50Hz，吃 proprioception + z，输出 joint command。类似脊髓反射回路。

频率解耦是工程关键：System 2 在外接 4090 上跑，推理 latency 可能 80-100ms，如果让 System 1 等 System 2，robot 会摔倒。所以 System 1 每 5 步（100ms）才换一次 z，中间自己 hold 住并 reactive control。这个 hold-and-resample 设计让两个 system 解耦。

---

## LeVERB-VL 的核心：Residual CVAE

这是 paper 最有技术含量的部分，我说得再直白一点。

**目标**：给一张图 + 一句话，输出一个 256 维的 latent z，这个 z 要包含足够信息让下游 controller 知道怎么动。

**朴素方案**：直接训一个 VAE，encoder 吃 image+text，decoder 吃 z 重建 future motion。问题：VLA 模型要同时扛语义理解 + motion 细节，容易 overfit，而且 motion 细节（步幅、坐姿角度）其实跟视觉语义是 orthogonal 的。

**LeVERB 的方案**：把 latent 拆成两部分加起来。

$$
\mu = \mu_\rho + \mu_E
$$

- $\mu_\rho$：VLA prior 看图和文字后预测的 latent mean，负责"语义粒度"信息（"我要去左边椅子坐下"）
- $\mu_E$：kinematics encoder 看 ground-truth future motion 后预测的 mean，负责"motion style 粒度"信息（具体怎么走、坐多深）
- $\sigma$：直接用 encoder 的，不用 VLA 的

KL loss 强迫 encoder 只编码 VLA 看不出来的那部分——相当于 information bottleneck，逼 VLA 专注语义、encoder 专注细节。两者相加得到完整 latent。

**部署时只有 VLA prior 在跑**（$\mu_\rho$），encoder 是训练时的特权信息，部署时没有 future motion 可看。但因为训练时 residual 结构让 VLA 学到了"语义部分"，部署时它单独输出的 $\mu_\rho$ 已经足够指导 controller。

这个设计借鉴了 **MotionVAE**（Ling et al., SIGGRAPH 2020, https://dl.acm.org/doi/10.1145/3386569.3392442），但加了 residual connection，是个很优雅的变体。

---

## 那个 Discriminator 是干啥的

数据集里 38% 是 language-only（白噪声图像），62% 是 vision-language。这两类数据如果直接混训，VAE 的 latent space 会分裂成两个 cluster：有图的 latent 和没图的 latent 统计上完全不一样。部署时只有有图的输入，但 language-only 学到的 motion skill 用不上——generalization 大打折扣。

**解法**：在 latent z 上接一个 discriminator，预测"这个 z 是不是来自真实图像"。用 **Gradient Reversal Layer (GRL)** 反传梯度，让 VLA encoder 被 adversarially 训练成"分不清有没有图"——也就是 modality-invariant。

这个 idea 来自 **Neural-Fly**（O'Connell et al., Science Robotics 2022, https://arxiv.org/abs/2203.10774）的 domain adaptation。LeVERB 把它移植到 VLA latent space，是个很小但很关键的 trick。Ablation 里去掉 discriminator (ND) 后 visual navigation 直接从 80% 掉到 75%、再到 distractor 场景的 55%，证明它真的在起作用。

---

## LeVERB-A：DAgger 蒸馏

System 2 训完冻住，开始训 System 1。

**Step 1**：训一堆 teacher。每个 task category 训一个 PPO policy，teacher 能看到 privileged 信息（reference motion 的 future pose），任务是让 robot 精确跟踪 kinematic trajectory。Reward 是 DeepMimic-style（torso position/orientation/body position/joint error），加 action rate penalty、joint limit penalty、termination penalty。Domain randomization 做满（friction、restitution、armature、push）准备 sim-to-real。

**Step 2**：用 DAgger 蒸馏 student。Student 是一个 2-layer Transformer，输入是 proprioception + latent z（作为两个 separate token）。训练时：
- 随机选 trajectory 和起点
- 从 LeVERB-VL 的 latent distribution $\mathcal{N}(\mu_\rho, \sigma_\rho)$ **采样** z（关键！不是用 mean）
- Hold z 固定 5 步，让 student 跟 teacher 的 action 做 Huber loss
- DAgger 的 on-policy 数据采集让 student 在自己的 state distribution 上学习，避免 covariate shift

**为什么训练时 sample 不用 mean？** LeVERB-VL 学的是 multimodal mapping——同一句话+同一个场景可能对应多种合理 motion。如果训练只用 mean，student 见到的 z 分布是 unimodal 的，部署时遇到真实的多模态 latent 就 OOD。Ablation NLS（No Low-level Sampling）证明了这点：visual navigation 几乎全崩。

**部署时反而用 mean**——因为真机部署要求 deterministic 保证安全。这个 train-sample / deploy-mean 的 asymmetry 是工程上的关键 trick。

---

## 实验数字讲了什么

Table 2 是核心结果。我挑几个关键对比：

| 变体 | Overall | Visual Nav (VNF Objective) | Sit | Locomotion |
|---|---|---|---|---|
| LeVERB full | 58.5% | 80% | 100% | 100% |
| ND (no discriminator) | 33% | 75% | 0% | 100% |
| NE (no kinematics encoder) | 53% | 75% | 100% | 100% |
| NVL (no VL module, direct embedding) | 25.5% | 0% | 40% | 100% |
| NLS (no low-level sampling) | 6.5% | 0% | 5% | 25% |
| NS (no sampling at all) | 7.5% | 0% | 10% | 50% |

几个观察：

1. **NVL 在 visual navigation 全崩（0%）但 locomotion 100%**：说明纯语言指令不需要 high-level reasoning，low-level controller 能直接消化。但一旦加视觉，low-level 自己根本不够用——必须有 VLA planning 层。

2. **NS（naive deterministic hierarchical VLA）只有 7.5%**，比 full LeVERB 低 **7.8×**。这是 paper 标题级的对比——证明 CVAE latent structure + 两层 sampling 的必要性。如果有人想复现 humanoid VLA，这是最容易踩的坑：直接把 VL embedding 接 controller，deterministic conditioning，会完全 fail。

3. **ND 在 Sit 任务上从 100% 掉到 0%**：Sit 是 walk + turnaround +坐下三段式复合动作，需要 language-only 数据学到的 sitting motion skill 迁移过来。去掉 discriminator 后，language-only 和 vision-only 的 latent 分裂，sitting skill 无法迁移，Sit 全崩。

4. **NE 只微降 5 个点**：说明 residual 结构里 kinematics encoder 的贡献相对小，VLA 本身就能 carry 大部分语义。但 NE 在 unseen scene 上鲁棒性下降，因为 latent 更细粒度、更易 overfit。

---

## Real-World 部署的工程细节

Unitree G1 上跑的架构其实很朴素但很实用：

- **System 1**：robot onboard CPU，C++ + ONNX runtime，50Hz。Sensor 500Hz（joint encoder + IMU + state estimator）。Action 是 joint position target，velocity 设 0，kp/kd 用模拟器里的 stiffness/damping。latent command 通过 ROS2 topic 接收。
- **System 2**：外接 RTX 4090 PC，10Hz。输入是 USB 第三方相机 + onboard RealSense，30 FPS 1080×720。输出 latent verb 通过 ROS2 topic 广播。

**Sim-to-real 验证方法很聪明**：因为真机上闭环视觉 feedback 链路还没完全 trust，作者先在 sim 里跑闭环任务，录下 LeVERB-VL 输出的 latent verb sequence，然后在真机上**开环重放** latent verbs 给 LeVERB-A。如果真机动作正确，说明：
1. LeVERB-VL 在真实图像上能 generalize（视觉理解 work）
2. LeVERB-A 的 dynamic control 能 sim-to-real（控制器 work）

Figure 4 展示两个能力：unseen verb-object 组合（"rest on the box"）能正确执行；chair 摆在不同位姿，robot 能视觉判断并 walk+turn 落座。这是 spatial reasoning from vision 的证据。

---

## 我觉得这篇 paper 真正的贡献

扒开所有技术细节，我觉得 LeVERB 的核心 contribution 在三个层面：

### 1. 概念层面：latent verb 是 humanoid VLA 的正确接口

之前 humanoid VLA 的工作要么用显式 action（base velocity、end-effector pose），要么用 full-body keyframe。前者 expressiveness 不够，后者需要 task-specific tuning。LeVERB 证明**学出来的 latent verb vocabulary** 可以同时覆盖 navigation、sitting、reaching 这种 heterogeneous whole-body skill，而且 vision-language 语义能直接映射进去。

这个 insight 我认为会延续到后续所有 humanoid VLA 工作——就像 manipulation 领域 latent action 已经成为标准（LAPA、π0 的 flow matching）一样。

### 2. 工程层面：几个关键 trick

- **Kinematic replay 数据生成**：bypass dynamic 的 chicken-and-egg，让数据 scaling 变得 tractable
- **Residual CVAE**：把 semantic 和 motion detail 解耦，VLA 不用背所有细节
- **GRL discriminator**：让 mixed-modality data 真的能共用 latent space
- **Sample-train / mean-deploy**：训练时让 student 见到真实 latent distribution，部署时 deterministic 保证安全
- **Frequency decoupling + ROS2**：10Hz/50Hz 解耦让 VLA 推理 latency 不卡 controller

每个 trick 单独看都不新，但组合在 humanoid VLA 上是第一次 work，而且 ablation 证明缺一不可。

### 3. Benchmark 层面：LeVERB-Bench 填空白

之前 humanoid benchmark 要么没 vision（HumanoidBench），要么没 physics（Mimicking-Bench），要么渲染不 photorealistic。LeVERB-Bench 是第一个三者全有的，而且 closed-loop evaluation 而不是 open-loop tracking。154 条 trajectory 听起来少，但 ×100 randomization + procedural generation 后覆盖了 10 个 task category，足够做 fair comparison。

---

## 我对后续的猜测

作者在 Limitations 里承认两件事：horizon 太短（只预测几秒）、System 1 没 fast vision feedback。我猜测后续工作会往几个方向走：

1. **RL fine-tuning latent verb**：现在 LeVERB-VL 是纯 behavior cloning，部署后没有 on-policy correction。如果在 latent verb space 上做 PPO finetune（类似 RLHF 之于 LLM），可能突破 58.5% 的 plateau。作者自己在 Conclusion 里也提到这个方向。

2. **In-the-loop vision for System 1**：现在 System 1 是 proprioception-only，做不到 reactive visual servoing。如果把 vision 加到 System 1 的 observation 里（类似 ExBody2 的做法），可能实现更 agile 的任务。但频率约束是个挑战——50Hz vision inference 在 onboard CPU 上目前还做不到。

3. **更大数据规模**：17 小时 VL 数据在 LLM 时代看起来很小。如果用 internet video（Humans in 4D, Goel et al., https://arxiv.org/abs/2305.20091）retarget 到 humanoid，数据量可以提升几个数量级。LeVERB 的 kinematic replay pipeline 天然支持这种扩展。

4. **Long-horizon planning**：当前 latent verb 只预测几秒 future，长任务需要 hierarchical planning 或者 latent space 上的 tree search。可以借鉴 LocoMujoco (https://arxiv.org/abs/2311.02496) 或者 LM-Nav (https://arxiv.org/abs/2303.01260 实际 https://arxiv.org/abs/2303.01260) 的思路。

---

## 一句话总结 intuition

LeVERB 的核心 insight 用一句话说就是：**humanoid VLA 的正确接口不是显式 action，也不是 end-to-end latent，而是"学一个既编码语义又编码 motion style 的 latent verb vocabulary，用 residual CVAE 把两者解耦，用 frequency decoupling 让 VLA 和 controller 各跑各的频率，用 sample-train/mean-deploy 让训练分布匹配部署安全"**。

这套组合在 humanoid 上第一次跑通 zero-shot sim-to-real，虽然 58.5% overall success rate 离实用还有距离，但证明这条路是 viable 的。后续如果加 RL finetune + 大规模 video data，我预期 humanoid VLA 会快速跟进 manipulation VLA 的成熟度曲线。

如果你想我进一步 dig 哪个部分——比如 CVAE 的 ELBO 推导、DAgger 的 covariate shift 数学、或者 reward shaping 的具体 trade-off——我可以再展开。

---

# LeVERB 深度技术讲解

Karpathy 你好，这篇 LeVERB 是 UC Berkeley + CMU + SFU + NTNU 合作的工作，做的是 **humanoid whole-body control (WBC) 上第一个 vision-language latent action model**，并且提供了第一个 photorealistic + physics-based 的 sim-to-real benchmark。我会从 motivation、benchmark、dual-process 架构、CVAE 训练目标、DAgger distillation、ablation 实验到 real-world deployment 宋完整拆解，并把变量、公式、实验数字都讲清楚。

---

## 1. Motivation 与 gap

当前 VLA 的生态被 tabletop manipulation 主导（RT-2, OpenVLA, π0, Octo）。这些方法有一个隐含假设：**底层 controller 已经是 hand-crafted 的低维 action vocabulary**——end-effector pose、base velocity、keypoints。这在 quasi-static manipulation 上没问题，但 humanoid 是 high-dimensional nonlinear dynamic system，pelvis 浮动 + 12+ joints + 接触-rich，hand-crafted verb 集合既覆盖不全也无法表达 agile whole-body motion。

更具体地：
- **NaVILA** (https://arxiv.org/abs/2412.04453) 把 action 抽象成 direction + distance，只能做 navigation
- **Humanoid-VLA** (https://arxiv.org/abs/2502.14795) 预测 full-body pose keyframes，expressiveness 高但需要 task-specific tuning
- **LangWBC** (https://arxiv.org/abs/2504.21738) 用 CVAE 学 latent，但只有 language 没有 vision
- **OmniH2O** (https://arxiv.org/abs/2406.08858), **HumanPlus** (https://arxiv.org/abs/2406.10454), **ExBody2** (https://arxiv.org/abs/2412.13196), **AMO** (https://arxiv.org/abs/2505.03738) 都需要 teleop 或显式 pose 指令
- 现有 humanoid benchmark 比如 **HumanoidBench** (https://arxiv.org/abs/2403.10506) 没 photorealistic rendering；**Mimicking-Bench** (https://arxiv.org/abs/2412.17730) 没 physics

LeVERB 的核心 insight：**学一个 latent verb vocabulary 同时承担 vision-language 语义和 whole-body motion 语义**，再让 RL-trained WBC 把 latent verb 翻译成 dynamics-level joint commands。这套借鉴 Kahneman 的 System 1 / System 2 框架（参考 [18]，Arthur Jensen 的 mental chronometry），high-level System 2 用 VLA 做 semantic reasoning（10Hz），low-level System 1 做 reactive whole-body control（50Hz），中间用 latent vector z 做单向接口。

---

## 2. LeVERB-Bench：合成数据生成

这是工作里很关键的 contribution，因为没有现成的 humanoid vision-language dataset 可用。

**核心 trick：用 kinematic replay 替代 dynamic control 来采集数据**。具体流程：

1. 从 AMASS / LAFAN / in-house RL policy 拿到 retargeted humanoid kinematic trajectories（154 条 vision-language + 460 条 language-only）
2. 在 IsaacSim 里 **kinematically replay** 这些 trajectories（不需要 PD 控制器跟得上），同时用 **ray-tracing rendering** 把场景打光成 photorealistic
3. Procedural generation：每条 trajectory 随机化 100 次——scene background、object color/material、task setup、camera view (1 egocentric + 2-3 third-person)、左右 mirror
4. Language 标注：要么手动写 egocentric instruction，要么用 VLM (VILA, https://arxiv.org/abs/2312.07533) 自动 annotate

最终得到 **17.1 小时 vision-language data + 2.7 小时 language-only data**，分布见 Table 1。Category 涵盖 Navigation / Towards / Around / Locomotion / Sitting / Reaching，平均 trajectory 长度约 4 秒。

**为什么 kinematic replay 够用？** 作者的 bet 是：vision-language 模型只需要从图像里提取 task-level semantics（"前面有个椅子"），具体 dynamic execution 由 LeVERB-A 承担，所以训练数据可以 "fake dynamic"——只要视觉上像、语言上对就行。这个 assumption 在 ablation 里被验证：尽管有 minor artifacts，配合高质量 low-level policy 闭环时仍然 work。这是相当聪明的 data scaling 思路，类似于 manipulation 领域用 script 生成 demonstration 而不强行做 dynamic grasping。

Appendix A 给了 5 步 procedural generation：
1. Scene-level randomization (background color/material)
2. Object-level randomization (chair/desk color/material)  
3. Task placement (strategic placement 让 instruction 有意义，比如把目标放 trajectory 末端 = "walk towards xx")
4. 100 demos per trajectory，含 ego + 第三方 cameras
5. Mirror 一半 demos 翻倍

---

## 3. Dual-Process 架构与 marginalization

整体策略写成 **marginalized policy**：

$$
\pi_\theta(a_t \mid o_t) = \int \tau_{\theta_A}(a_t \mid z_t, o_t^{prop}, a_{t-1}) \cdot \rho_{\theta_{VL}}(z_t \mid I_t, c) \, dz_t
$$

变量含义：
- $a_t$: 50Hz 的 dynamics-level action（joint position targets）
- $o_t = [o_t^{prop}, I_t, a_{t-1}, c]^T$: full observation
- $o_t^{prop}$: proprioception（base lin/ang vel、joint pos/vel、gravity projection）
- $I_t$: egocentric + 第三方 camera 图像，30 FPS, 1080×720
- $c$: textual instruction
- $z_t$: latent verb，是 LeVERB-VL → LeVERB-A 的单向接口
- $\rho_{\theta_{VL}}$: System 2，参数 $\theta_{VL}$
- $\tau_{\theta_A}$: System 1，参数 $\theta_A$

**频率解耦**：VL 跑 10Hz（4090 GPU 上），A 跑 50Hz（robot onboard CPU, ONNX runtime, C++）。每 5 步 A 采样一次新 z 保持固定（hold-and-resample），这样 System 2 推理 latency 不会卡住 System 1 的实时性——这是整个架构能不能 deploy 的关键。

公式 (1) 是一个 hierarchical 的边缘化表达，意味着高阶动作是一个**分布**而不是 deterministic action——这给后面的 CVAE + sampling 训练埋下伏笔。

---

## 4. LeVERB-VL（System 2）：Residual CVAE

这是 paper 最有技术含量的部分。目标：把 vision + language 映射到一个 smooth、regularized latent verb space。

### 4.1 模块拆解

**Vision encoder**：SigLIP (https://arxiv.org/abs/2303.15343) 的 ViT-B/16 visual component，在 WebLI 上预训练，**全程 frozen**。两个视角的图像分别过 ViT 得到 patch tokens，再 attention-pool 成单 token $i_t^{ego}$ 和 $i_t^{exo}$。每个 token 是 768 维。

**Text encoder**：同一个 SigLIP 的 text tower，也是 frozen，输出 language token $l_t$。

**Transformer backbone**：input sequence = $[l_t, i_t^{ego}, i_t^{exo}]$，**只取当前 frame 不带 temporal history**（防 overfitting，参考 SpawnNet https://arxiv.org/abs/2410.08785 和 Mandlekar et al. https://arxiv.org/abs/2108.03298）。Transformer + MLP head 输出 prior 分布 $\mathcal{N}(\mu_\rho, \sigma_\rho^2)$。

**Privileged kinematics encoder $E_\psi$**：一个 MLP，**输入是 future ground-truth states** $s_{t+1}, \ldots, s_{t+M}$（flatten 后），输出 posterior mean $\mu_E$ 和 variance $\sigma_E$。这里 $M$ 是 future horizon，$s$ 是 13-joint state（root joint 的 $(x,y,z)$ + yaw/roll/pitch 转 6D rotation [54]；其他 12 joints 只用 $(x,y,z)$ position w.r.t. root）。

### 4.2 Residual latent 构造（关键设计）

借鉴 **MotionVAE** (Ling et al., https://arxiv.org/abs/2004.07294 实际是 https://dl.acm.org/doi/10.1145/3386569.3392442)，但改成 residual：

$$
\mu = \mu_\rho + \mu_E, \qquad \sigma = \sigma_E
$$

也就是 **mean 加性 residual、variance 直接用 encoder 的**。

Intuition：VLA prior $\mu_\rho$ 负责"语义粒度"的 latent（"去左边椅子坐下"这个意图），kinematics encoder $\mu_E$ 补"motion style 粒度"的细节（具体步幅、坐姿角度）。KL loss 强迫 $\mu_E$ 只编码 VLA 看不出来的那部分信息，相当于 information bottleneck。这个 residual 结构让 VLA 不用承担 motion 全部细节，避免 VL 模型被 motion 信号 overfit。

后验 $q(z_t \mid s_{t+1:t+M}, I_t, c, o_t) = \mathcal{N}(\mu_\rho + \mu_E, \sigma_E^2)$，先验 $p(z_t \mid I_t, c) = \mathcal{N}(\mu_\rho, \sigma_\rho^2)$。

Reparameterization：$z_t = \mu + \sigma \cdot \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

### 4.3 Kinematics Decoder $D_\psi$

MLP，输入 $(s_t, z_t)$，输出 $\hat{s}_{t+1}, \ldots, \hat{s}_{t+M}$——重建 future states。注意这里 future state 是 **delta action**：root 给 delta position + delta rotation（6D），其他 joints 给 delta position。作者在 Appendix C 单独做了一个 sanity check：如果 decoder 只给 $s_t$ 不给 $z_t$，reconstruction loss 收敛到远高于有 $z_t$ 的版本——证明 latent 真的承载了 future prediction 信息，decoder 不能 trivially reconstruct。

### 4.4 Discriminator $f_\psi$ + Gradient Reversal Layer

问题：数据集里 38.4% 是 language-only（白噪声 image 喂给 VL），其余 61.6% 是 vision-language。这两种 source 在 latent space 容易分成两个 cluster，破坏 generalization。

解法：在 $z_t$ 上接一个 binary discriminator 预测 "image 是否是真实图像"，用 **Gradient Reversal Layer (GRL, Ganin & Lempitsky, https://arxiv.org/abs/1409.7495)** 反传梯度。这样 LeVERB-VL 被 adversarially 训练成 **modality-invariant**：blind 和 non-blind 输入产生统计上 indistinguishable 的 latent。

### 4.5 总训练目标

$$
\mathcal{L}(\theta_{VL}, \psi) = \underbrace{\mathbb{E}_{\rho_{\theta_{VL}}(z|I,c,s_t)} \left[ \| D_\psi(s_t, z) - s_{t+1:t+H}^R \|_2^2 \right]}_{\text{trajectory reconstruction}} + \underbrace{\beta_1 D_{KL}(q(z_t) \| p(z_t))}_{\text{distributional alignment}} + \underbrace{\beta_2 \mathcal{L}_{disc}(z_t)}_{\text{adversarial alignment}}
$$

变量：
- $s_{t+1:t+H}^R$: ground-truth reference future state (delta action representation)
- $\beta_1 = 10^{-1}$, $\beta_2 = 5 \times 10^{-4}$
- $\beta_1, \beta_2$ 都套了 scheduler，前 40% training epoch 从 0 线性升到 1，warmup 防 early training instability

**训练 setup**：2× NVIDIA Ada 6000 GPU，global batch size 512，总 trainable 参数 102.56M（ViT-B backbone）。Appendix B 里 ablate backbone size：ViT-Tiny → ViT-Small → ViT-Base，最后选 ViT-Base 做 trade-off。latent dim = 256。

---

## 5. LeVERB-A（System 1）：DAgger Distillation

LeVERB-VL 训完之后 latent space 冻住，下面训练能消化 z 的 controller。

### 5.1 训练 Teacher $T_\xi$（per-task PPO）

每类 motion 训一个 teacher，**接收 privileged observation $o_t^{priv}$ + reference motion commands**（包括 future reference joint pos/vel + reference torso pose relative to actual），输出 joint position action。

Reward 设计（Appendix Table 4）很 DeepMimic-style (https://arxiv.org/abs/1804.02717)：

| Reward term | Weight | Form | $\sigma$ |
|---|---|---|---|
| Global Torso Position | 0.5 | $\exp(-\|\mathbf{p}_{motion} - \mathbf{p}_{robot}\|^2 / \sigma^2)$ | $\sqrt{0.25}$ |
| Global Torso Orientation | 0.3 | $\exp(-\text{quat\_error}(\mathbf{q}_{motion}, \mathbf{q}_{robot})^2 / \sigma^2)$ | $\sqrt{0.5}$ |
| Global Body Position | 0.5 | $\exp(-\|\mathbf{x}_{motion} - \mathbf{x}_{robot}\|^2 / \sigma^2)$ | $\sqrt{0.25}$ |
| Joint Position Error | -1 | $-\|\boldsymbol{\theta}_{motion} - \boldsymbol{\theta}_{robot}\|$ | - |
| Joint Velocity Error | -0.1 | $-\|\dot{\boldsymbol{\theta}}_{motion} - \dot{\boldsymbol{\theta}}_{robot}\|$ | - |
| Action Rate | -0.001 | $-\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ | - |
| L2 Joint Limit | -100 | $-\mathbb{I}_{\text{violate\_limit}}$ | - |
| Termination | -200 | $-\mathbb{I}_{\text{done}}$ | - |

Early termination：
- $\|\mathbf{p}_{motion} - \mathbf{p}_{robot}\| > \tau_{pos} = 0.5$ m
- $|\text{proj}_z(\mathbf{g}_{motion}^B - \mathbf{g}_{robot}^B)| > \tau_{ori} = 0.8$（base 朝向偏离过大）

Domain randomization（sim-to-real 关键）：
- Ground friction $\in [0.3, 0.8]$
- Ground restitution $\in [0, 0.5]$
- Joint default pos offset $\in [-0.05, 0.05]$ rad
- Joint armature scale $\in [0.2, 2.0]$
- 周期性 push robot：每 [10,15]s 加 x-y velocity $\in [-0.5, 0.5]$ m/s

Teacher 架构：3-layer MLP，hidden 512→256→128，ELU activation。

### 5.2 训练 Student LeVERB-A $\tau_{\theta_A}$

Student 是 Transformer，2 layers / 4 heads / hidden 128，**把 $z_t$ 和 proprioception 作为 separate tokens 喂进 transformer**（不是 concat），attention dropout 0.3。

**DAgger 训练（Ross et al., https://arxiv.org/abs/1011.0686 实际 https://arxiv.org/abs/1105.1146）+ Huber loss**：

关键点：episode 起点随机选 trajectory 里某 timestep t，**每 H 步从 $\mathcal{N}(\mu_{\rho,t}, \sigma_{\rho,t})$ 采样一个新的 $z_t$**——hold H 步（对应 System 1-2 resampling interval），然后让 student 跟 teacher 的 $a_t$ 做 supervised loss。

**为什么 sample 而不是用 mean？** 论文强调这是关键设计：LeVERB-VL 学的是 multimodal mapping $p(\text{motion} \mid \text{VL})$，如果只 sample mean 就变成 unimodal 近似，下游 policy 看到的 z 分布和部署时 $\rho_{\theta_{VL}}$ 输出的分布 mismatch。这个观察在 ablation NLS 里被验证：NLS (No Low-level Sampling) 在 visual navigation 任务上几乎完全失败。

**部署时** LeVERB-A 用 $\mu_\rho$ 而非采样——因为部署阶段希望 deterministic 控制以保证安全。

Student observation 包含 gravity vector projection（用于估计 roll/pitch），**故意不加 temporal history**——因为如果 student 能从历史推 future action，latent command 就失去信息量，这是 ablation 里观察到的退化。

### 5.3 DAgger 闭环必要性

Teacher 是 per-task 训练的，但 student 必须消化 z 里的任意语义组合。DAgger 的 on-policy 数据采集让 student 在自己产生的 state distribution 上对齐 teacher，避免 covariate shift——这对闭环 visual control 尤其关键，因为开环 supervised 训出来的 student 一旦偏离 trajectory 就会越走越偏。

---

## 6. 实验：Closed-Loop Benchmark

### 6.1 Ablation variants

| 缩写 | 含义 |
|---|---|
| ND | No Discriminator，去掉 GRL adversarial loss |
| NE | No kinematics Encoder，所有 motion 信息全压到 VLA |
| NVL | No LeVERB-VL，直接把 VL embedding 当 latent 给 controller（参考 LangWBC 的做法） |
| NLS | No Low-level Sampling，训练 student 时用 mean |
| NS | No Sampling 两处都关，naive deterministic hierarchical VLA baseline |

### 6.2 Task subcategories

- VNF/VNR：visual navigation 目标在 front（easy）/rear（需要 turnaround）
- Objective / Distractor / Cluttered：场景纯净度递减
- VNS：完整 walk+turnaround+sit sequence（最难）

### 6.3 Table 2 数字解读

**LeVERB full**：58.5% overall，VNF/Objective 高达 80%，Sit 100%, Stand 90%, Locomotion 100%。

**NE (53%)** 微降：去 kinematics encoder 后 latent 更细粒度但缺少 semantic compression，对 unseen scene 鲁棒性下降。

**ND (33%)** 大降：visual navigation 几乎全崩。原因正是 discriminator 没了，VL-only 和 language-only 数据 latent 分裂，generalization 大幅下降——证明 GRL 对齐 modality 是 cross-modal 数据混合的必需品。

**NVL (25.5%)**：直接用 VL embedding 当 latent，在 visual-language 任务几乎全 fail（VNF Objective 0%，Cluttered 0%），但在 text-only locomotion 100%——印证 atomic language command 不需要 high-level reasoning，但 vision 一旦进入，low-level policy 单独无法消化。

**NLS (6.5%)**：student 训练用 mean，部署 mismatch 导致 visual nav 全崩。

**NS (7.5%)**：naive deterministic hierarchical baseline，比 LeVERB full 低 **7.8×**——这是 paper 标题级的 ablation，证明 CVAE latent 结构 + 两层 sampling 的必要性。

### 6.4 关键 take

- **Vision 进入必须配 hierarchical latent**：low-level controller 单独消化 vision 信息会 fail
- **CVAE + residual + discriminator 三者缺一不可**：ND、NE、NLS 各自崩一个角度
- **Sample 训练 + mean 部署** 的 asymmetry 是工程上重要的 trick：训练时让 student 见到真实 z 分布，部署时用 mean 保证 safety
- **Vocabulary generalization**：Figure 4 显示 LeVERB 能正确响应 unseen verb-object 组合（"rest on the box"），证明 latent space 真的学到了组合语义

---

## 7. Real-World Deployment on Unitree G1

部署架构很工程化，值得记录：

**System 1 (LeVERB-A)**：
- 硬件：Unitree G1 + joint encoder + IMU + custom state estimator
- 频率：sensor 500Hz，inference 50Hz，**ONNX runtime, C++ 实现**
- 接口：latent command 通过 ROS2 topic 接收
- Action：joint position target，velocity=0，kp/kd = 模拟器 stiffness/damping

**System 2 (LeVERB-VL)**：
- 外接 RTX 4090 PC，10Hz 推理
- 输入：USB 第三方 camera + onboard RealSense，30 FPS, 1080×720
- 输出：latent verb ROS2 topic 广播

**Sim-to-real 策略**：作者用一个很聪明的"开环重放"验证法——在 sim 里跑 closed-loop 任务（带 unseen verb-object 组合、unseen chair pose），把 LeVERB-VL 输出的 latent verb sequence 录下来，**在 real robot 上开环重放 latent verbs 给 LeVERB-A**。如果 dynamics controller 跟随正确，说明：
1. LeVERB-VL 的视觉理解 generalize 到真实 camera image
2. LeVERB-A 的 dynamics 控制能 sim-to-real

Figure 4 演示两个能力：
- 词汇泛化：unseen 语言指令 → 正确 motion
- 空间推理：chair 摆在不同 pose，robot 视觉判断并 walk+turn 落座

---

## 8. Limitations

作者坦率承认：
1. **Limited horizon**：只预测几秒 future，没有 long-term planning capability（和 π0、OpenVLA 一样）
2. **System 1 缺 fast vision feedback**：因为 LeVERB-A 是 proprioception-only，做不到 reactive visual servoing。这个限制让 agile/dexterous 任务（比如接抛物、踩 narrow beam）仍无法实现

更深层 limitation 我推测：
- 154 条 VL trajectory 即使 ×100 randomization 也只有 17 小时数据，规模相对小；如果模型 fail 在 unseen scene category（比如 outdoor），无 fallback
- kinematic replay 有 artifact（脚可能 slide、身体可能 float），VL 学到的 visual-motion 关联有 systematic bias
- teacher 是 per-task PPO 训的，task 多样性扩展时 teacher 数量线性增长，scalability 一般
- DAgger distillation 假设 teacher 在 training distribution 外还能给出 expert action——如果 student 把 state 带到 OOD 区域，teacher action 可能 garbage，distillation 会有 systemic error

---

## 9. 我对这篇工作的几条 intuition

1. **Kinematic replay 是 VLA data 生成的小 breakthrough**。以往要训 vision-conditioned WBC，要么 teleop 真机收集（极慢），要么训 dynamic controller 然后做 rollout（chicken-and-egg）。LeVERB 直接 bypass dynamic 阶段，只回放 kinematics + photorealistic rendering。这个 trick 类似 manipulation 领域的 "scripted demonstrator" 思路，但更激进——因为 humanoid 的 dynamic 比 arm 复杂得多。

2. **Residual CVAE 是分割 semantic / motion detail 的优雅方案**。VLA 不必背 motion style 的细节，kinematics encoder 只补 difference。这个 residual 思想可以推广到任何"high-level semantic + low-level detail"的 latent space 设计。

3. **Modality-invariant latent via GRL** 对 mixed-modality dataset 几乎是必需品。这个 idea 来自 Neural-Fly (O'Connell et al., https://arxiv.org/abs/2203.10774) 的 domain adaptation，移植到 VLA latent space 很自然但很有效。

4. **Sample-train / mean-deploy** 是 hierarchical latent VLA 的关键工程 trick。这篇文章在 ablation 里直接证明 deterministic conditioning 会 fail，是个很 actionable 的发现。**π0** (https://arxiv.org/abs/2410.24164) 也有类似 flow-matching 思想，但 LeVERB 是 explicit 把它做成 train-deploy asymmetry。

5. **Frequency decoupling** 10Hz/50Hz + ROS2 topic 接口，工程上简洁可复现。这比 GR00T N1 (https://arxiv.org/abs/2503.14734) 和 AGIbot World (https://arxiv.org/abs/2503.06669) 的端到端 VLA 更适合 bipedal 控制时延约束。

6. **未做的下一步**——作者提到 RL post-training / finetuning 来 align closed-loop latent distribution。这暗示 LeVERB-VL 是 behavior cloning-only，部署后没有 on-policy correction。如果加 **PPO 在 latent verb space 上 finetune**（类似 RLHF 之于 LLM），可能突破 58.5% 的 plateau。也可以考虑 LAPA (Ye et al., https://arxiv.org/abs/2410.11758) 的 video pretraining 做更强 visual prior。

---

## 10. Reference Links

| 论文 | Link |
|---|---|
| DeepMimic [37] | https://arxiv.org/abs/1804.02717 |
| AMP [38] | https://arxiv.org/abs/2104.02180 |
| PULSE [30] | https://arxiv.org/abs/2310.04582 |
| MaskedMimic [46] | https://arxiv.org/abs/2410.14203 |
| TokenHSI [36] | https://arxiv.org/abs/2503.19901 |
| LangWBC [43] | https://arxiv.org/abs/2504.21738 |
| NaVILA [9] | https://arxiv.org/abs/2412.04453 |
| Humanoid-VLA [11] | https://arxiv.org/abs/2502.14795 |
| OmniH2O [16] | https://arxiv.org/abs/2406.08858 |
| HumanPlus [12] | https://arxiv.org/abs/2406.10454 |
| ExBody2 [19] | https://arxiv.org/abs/2412.13196 |
| AMO [22] | https://arxiv.org/abs/2505.03738 |
| Expressive WBC [10] | https://arxiv.org/abs/2402.16796 |
| SigLIP [51] | https://arxiv.org/abs/2303.15343 |
| VILA [24] | https://arxiv.org/abs/2312.07533 |
| π0 [4] | https://arxiv.org/abs/2410.24164 |
| OpenVLA [20,21] | https://arxiv.org/abs/2406.09246 |
| DAgger [39] | https://arxiv.org/abs/1105.1146 |
| PPO [40] | https://arxiv.org/abs/1707.06347 |
| MotionVAE [26] | https://dl.acm.org/doi/10.1145/3386569.3392442 |
| GRL [13] | https://arxiv.org/abs/1409.7495 |
| RT-2 [6] | https://arxiv.org/abs/2307.15818 |
| HumanoidBench [41] | https://arxiv.org/abs/2403.10506 |
| GR00T N1 [3] | https://arxiv.org/abs/2503.14734 |
| AGIbot World [7] | https://arxiv.org/abs/2503.06669 |
| Octo [35,45] | https://arxiv.org/abs/2405.12213 |
| LAPA [49] | https://arxiv.org/abs/2410.11758 |
| Gemini Robotics [44] | https://arxiv.org/abs/2503.20020 |
| Mimicking-Bench [28] | https://arxiv.org/abs/2412.17730 |
| Neural-Fly [34] | https://arxiv.org/abs/2203.10774 |
| SpawnNet [25] | https://arxiv.org/abs/2406.07738 (推测) |
| What Matters [32] | https://arxiv.org/abs/2108.03298 |
| On Continuity of Rotations [54] | https://arxiv.org/abs/1812.07035 (CVPR 2019) |
| Humans in 4D [14] | https://arxiv.org/abs/2305.20091 |
| Twist [50] | https://arxiv.org/abs/2505.02833 |
| 3D-VLA [53] | https://arxiv.org/abs/2403.09631 |

---

## 总结

LeVERB 是 humanoid VLA 一个相当 tight 的 system paper：**问题定义清晰**（latent verb 接口）、**架构最小但有效**（residual CVAE + GRL + DAgger）、**benchmark 填空白**（photorealistic humanoid VL）、**real-world zero-shot** 跑通。它的最大 contribution 在我看来是把 VLA 从 manipulation 的高维 implicit action 接口推广到 humanoid 这种 high-DoF dynamic system 的 hierarchical latent action，并且用 sample-train / mean-deploy + frequency decoupling 解决了工程上 deployability。后续如果有人能用 RL fine-tune latent verb 或者扩展到 in-the-loop vision，会进一步打开 humanoid VLA 的天花板。

如果你想进一步 dig，我推荐先看 Appendix C 的 CVAE effectiveness sanity check（小实验但是关键假设验证），以及 Appendix D Table 4 的完整 reward + randomization spec——这是复现 sim-to-real 的核心 contract。
