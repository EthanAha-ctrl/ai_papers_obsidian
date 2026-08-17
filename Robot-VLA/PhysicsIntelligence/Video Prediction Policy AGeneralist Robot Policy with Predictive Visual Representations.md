---
source_pdf: Video Prediction Policy AGeneralist Robot Policy with Predictive Visual
  Representations.pdf
paper_sha256: 7a6d966b78a205efe0943e9188f16ec33e470526da1066df5406922fafb4f0d6
processed_at: '2026-08-13T00:34:09-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 VPP

---

## 一句话总结

与其让机器人 policy 自己去"想象"未来，不如直接借用视频生成模型已经"想好"的未来。

---

## 问题是什么

你给机器人看一张照片，问它下一步该干嘛。

传统做法：用一个 vision encoder（比如 ResNet）把这张照片变成一堆数字，然后让 policy network 从这堆数字里推断出动作。

问题在于：**一张静态照片信息量太少了**。机器人要抓桌子上的杯子，光看当前画面，它根本不知道杯子接下来会被推到哪、手应该从什么角度伸过去。这些"未来会发生什么"的信息，传统 encoder 完全给不了，policy network 只能自己在脑子里硬猜。

这就好比你给一个人看一张比赛截图，让他预测球员下一秒该往哪跑——他得自己脑补整场比赛的走势。

---

## VPP 的思路

视频生成模型（比如 SVD、Sora 那一类）已经在海量互联网视频上训练过了，它本身就具备"看一眼当前画面就能脑补出接下来几秒会发生什么"的能力。

VPP 的想法特别简单：**我为什么不直接把这个"脑补能力"拿过来用？**

但这里有个聪明的地方。正常用视频生成模型，你得跑几十步 denoising 才能生成出清晰的未来视频，太慢了，机器人控制要 10Hz 以上的频率，你跑个 diffusion 要几秒钟，黄花菜都凉了。

VPP 发现了一个特别有意思的现象：**你不用跑完所有 denoising 步骤，只跑一步 forward pass，模型内部的中间特征里就已经包含了未来的运动趋势**。

虽然这一步生成的画面是糊的（Figure 4 里可以看到，模糊但能看出物体和机械臂的运动方向），但这个"糊"的特征对 policy 来说已经够了。policy 不需要看到高清的未来画面，它只需要知道"东西大概往哪个方向动"。

这就像你瞄一眼路况，不需要看清每辆车的车牌号，只要知道大概车流方向就能决定自己怎么开。

---

## 具体怎么做

### 第一步：训练一个"专门懂操作的"视频生成模型

拿一个已经预训练好的视频生成模型（Stable Video Diffusion，15亿参数），然后在机器人操作数据 + 互联网人类操作视频上 fine-tune。

这样做的好处是：
- 互联网视频给了模型**通用的物理常识**（东西会掉、杯子能装水、抽屉能拉开）
- 机器人操作视频给了模型**对这个领域的熟悉度**

训练目标就是标准的 video diffusion loss，让它学会根据当前帧 + 语言指令预测未来 16 帧视频。

### 第二步：把视频模型当"带预测功能的 vision encoder"用

这一步是核心创新。

训练好视频生成模型后，**不再用它生成视频了**，而是把它当成一个 vision encoder：

1. 把当前观测画面 $s_0$ 和纯噪声拼在一起，送进视频生成模型
2. 只跑一次 forward pass（不做 multi-step denoising）
3. 从模型的 up-sampling layers 里把中间特征抠出来
4. 这些特征的维度是 $T \times C \times W \times H$，其中 $T$ 就是时间维度——也就是说特征本身就包含了未来 $T$ 步的预测信息

然后把这些特征喂给一个小的 Diffusion Policy head，让它根据这些"已经包含未来信息"的特征预测动作。

### 关键设计：Video Former

因为多视角（第三视角 + 手腕视角）和多层的特征维度太大，VPP 设计了一个 Video Former 模块来做压缩：

- 先用 learnable tokens 通过 spatial attention 把每帧的空间信息压缩
- 再用 temporal attention 把时间维度的动态信息整合
- 输出一组固定长度的 tokens，喂给 policy head

这个设计让推理时间从 ~450ms 降到 ~140ms，控制频率能达到 7-10Hz，满足 real-time 闭环控制需求。

---

## 为什么这样能 work

直觉上可以这样理解：

**视频生成模型已经学会了"物理世界怎么运作"**。它见过无数人抓杯子、推抽屉、倒水的视频，它知道物体该怎么动、手该怎么伸。

当你给它看一张机器人操作场景的照片，它的内部特征里已经在"规划"未来该怎么演变了——虽然你只是跑了一次 forward pass，但模型的权重里已经编码了对物理动力学的理解。

下游的 policy head 只需要学一件事：**在视频模型预测的未来轨迹里，机械臂应该怎么动才能对上那个轨迹**。这就是 inverse dynamics——从"未来状态"反推"当前动作"。

这比让 policy 从原始静态图像直接预测动作要容易得多，因为最难的部分（理解物理、预测未来）已经被视频生成模型做完了。

---

## 实验结果有多猛

### 模拟环境（Calvin ABC→D）

Calvin 是一个长视野任务基准，要求机器人连续完成 5 个任务，而且在没见过的 D 环境测试。

| 方法 | 平均完成任务数 |
|:---|:---|
| GR-1（之前 SOTA） | 3.06 |
| **VPP** | **4.33** |
| VPP（只用 10% 数据） | 3.25 |

VPP 用 10% 的数据就超过了之前用 100% 数据的 SOTA。这说明视频生成模型的物理先验太强了，下游只需要少量数据"对齐"动作空间就够了。

### 真实世界灵巧手

在 12 自由度灵巧手上测了 4 个工具使用任务，这些工具（勺子、锤子、电钻、移液管）在训练集里完全没出现过：

| 方法 | 勺子 | 锤子 | 电钻 | 移液管 |
|:---|:---|:---|:---|:---|
| Diffusion Policy | 0% | 20% | 0% | 0% |
| GR-1 | 30% | 10% | 20% | 0% |
| **VPP** | **90%** | **60%** | **80%** | **40%** |

这个结果说明视频生成模型的泛化能力传递给了 policy：即使没见过这些工具，视频模型能预测出合理的未来，policy 只需要跟踪机械臂的运动就行。

---

## 和其他方法的区别

| 方法 | 怎么用视频生成 | 问题 |
|:---|:---|:---|
| UniPi | 先生成完整未来视频，再在两帧之间学 inverse dynamics | 太慢，开环控制 |
| SuSIE | 用 Instruct-Pix2Pix 生成一张目标图，再学 policy | 只用一帧未来信息，信息量不够 |
| GR-1 | 自回归地逐帧生成视频 + 动作 | 生成质量不如 diffusion，没用预训练视频模型 |
| **VPP** | 用视频模型内部特征作为 predictive representation | 单步 forward pass，高频闭环，利用预训练先验 |

VPP 的独特之处在于它**不生成视频，只借特征**。它把视频生成模型当成一个"自带未来预测功能的 vision encoder"来用，既拿到了未来信息，又避免了 multi-step denoising 的延迟问题。

---

## 更深层的启示

### 1. Latent space 里的世界模型

Yann LeCun 一直说我们应该在 latent space 做预测，不要生成像素。VPP 虽然用的是 diffusion model，但实际效果上它就是这么做的——不要求生成高清未来画面，只要中间特征包含正确的物理演化方向就行。

### 2. Video Foundation Model = Physical Intelligence Foundation

OpenAI 说 Sora 是 world simulator，VPP 在机器人上验证了这个说法。视频生成模型在互联网数据上学到的东西，本质上就是物理世界的动力学。这个先验可以直接迁移给机器人。

### 3. Robotics 的 scaling law

VPP 的结果暗示机器人学习可能也在走 LLM 的路：用巨大的视频基础模型吸收互联网视频的物理知识，然后用少量真机数据做对齐。10% 数据超过 100% 数据的 SOTA，这就是 robotics 版的 "few-shot learning"。

### 4. 泛化机制

VPP 在未见过的物体和场景上能泛化，原因是双重的：
- 视频模型在互联网数据上见过足够多的类似场景，能预测出合理的未来
- Policy head 只需要跟踪机械臂在预测未来中的运动，不需要关心新物体是什么、背景长什么样

这把"理解新环境"和"生成动作"解耦了，泛化压力主要在视频模型上，而视频模型恰恰是最能泛化的部分。

---

## 一句话再总结

VPP 发现视频生成模型的中间特征自带未来预测能力，把这些特征当作 vision encoder 的输出喂给 policy head，等于免费给 policy 装了一个世界模型。不用真的生成视频，跑一次 forward pass 就够，速度快到能闭环控制。

---

Andrej，这篇 paper 的核心 idea 非常 elegant，直接触及了当前 robot learning 的一个核心痛点：vision encoder 缺乏对 future dynamics 的建模能力。相比于传统的基于 single-image reconstruction 或者 contrastive learning 的 vision encoder (如 R3M, VC-1, Voltron)，Video Prediction Policy (VPP) 直接将 Video Diffusion Models (VDMs) 的内部 representation 当作一种 "predictive visual representation" 来用，构建了一个 implicit inverse dynamics model。这种设计让 policy 能够利用未来轨迹的先验知识来指导当前的 action 生成。

下面我为你详细拆解这篇 paper 的技术细节、架构设计、实验数据，以及它背后更深层的物理直觉与关联。

---

### 1. Core Intuition: 为什么 VDM 的内部特征比普通 Vision Encoder 更好？

传统的 robot policy 通常是这样运作的：Vision Encoder (比如 ResNet, ViT, 或者 CLIP) 提取当前帧 $s_0$ 的 spatial feature，然后一个 policy head (MLP, Diffusion) 根据这个 feature 预测 action $a$。这里的问题是，当前的静态画面 $s_0$ 包含的信息有限，policy network 实际上需要隐式地在内部 "想象" 未来会发生什么，才能决定下一步怎么动。这把极强的动态推理压力全压在了 policy head 上。

VPP 的 insight 非常直接：既然 Video Diffusion Models (比如 SVD, Sora) 已经在海量互联网视频上训练出了强大的未来预测能力，为什么我们要把它当成一个普通的 image encoder 用？或者为什么要等它完整 denoise 出一整段视频再去做 inverse dynamics (像 UniPi 那样)？VPP 发现，仅仅把当前观测 $s_0$ 和纯噪声输入 VDM，执行一次 single forward pass，VDM 内部的 up-sampling layers 产生的 feature map $F_p \in \mathbb{R}^{T \times C \times W \times H}$ 已经隐式包含了未来 $T$ 步的轨迹信息。这里的 $T$ 维度明确对应了时间，这意味着 feature 本身就是 "predictive" 的。policy head 只需要在这个已经包含未来先验的 feature 上做 tracking，去拟合 inverse dynamics，极大降低了学习难度。

---

### 2. Architecture 深度解析

VPP 采用 two-stage training pipeline。

#### Stage 1: Text-guided Video Prediction (TVP) Model Fine-tuning

VPP 选择 Stable Video Diffusion (SVD, 1.5B parameters) 作为 base model。原版 SVD 只接受 initial frame 作为 condition，VPP 在此基础上通过 cross-attention 注入了 CLIP language feature $l_{emb}$，使其能响应 language instruction。同时，为了训练效率，输出分辨率调整为 $16 \times 256 \times 256$ (T=16 frames)。

训练 TVP model 的 loss function 如下：

$$ \mathcal{L}_{video} = \lambda_H \mathcal{L}_{D_H} + \lambda_R \mathcal{L}_{D_R} + \lambda_C \mathcal{L}_{D_C} $$

变量解析：
*   $D_H$: Internet human manipulation datasets (如 Something-Something-v2)
*   $D_R$: Internet robot manipulation datasets (如 RT-1, Bridge, BC-Z)
*   $D_C$: Self-collected downstream task datasets
*   $\lambda_H, \lambda_R, \lambda_C$: 数据集采样权重，用于平衡不同质量和规模的数据。

每个子 loss $\mathcal{L}_D$ 就是标准的 diffusion objective：

$$ \mathcal{L}_D = \mathbb{E}_{x_0 \sim D, \epsilon, t} \| V_\theta(x_t, l_{emb}, s_0) - x_0 \|^2 $$

*   $x_0$: Ground truth clean video sequence $s_{0:T}$
*   $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: 加噪后的 latent。$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ 是 noise schedule 的累积乘积。
*   $V_\theta$: 整个修改后的 SVD 网络。

这里的关键在于 VPP 将 internet-scale 的 physical dynamics knowledge 通过 diffusion loss 蒸馏进了网络参数中，使其成为一个强大的 manipulation-focused 物理先验引擎。

#### Stage 2: Action Learning with Predictive Visual Representation

这是 VPP 最核心的创新。VPP 把 Stage 1 训好的 TVP model 当作一个 frozen 的 vision encoder，而不去做耗时的 multi-step denoising。

**Feature Extraction & Aggregation**
把当前观测 $s_0$ 和 final noised latent $q(x_{t'} | x_0)$ (通常就是纯白噪声) 拼接送入 TVP，取 up-sampling layers 的 features。为了综合利用不同抽象层度的信息，VPP 提出了一个 auto-aggregation 方法：

$$ L_m = V_\theta(x_{t'}, l_{emb}, s_0)_{(m)}, L_m \in \mathbb{R}^{T \times C_m \times W_m \times H_m} $$

*   $m$: 第 $m$ 个 up-sampling layer。
*   $C_m, W_m, H_m$: 该层的 channel, width, height。

将所有层的 feature 线性插值到统一尺寸 $W_p \times H_p$：

$$ L_m' = \text{Interpolation}(L_m), L_m' \in \mathbb{R}^{T \times C_m \times W_p \times H_p} $$

然后在 channel 维度 concat：

$$ F_p = \text{concat}(L_0', L_1', \ldots, L_m', \text{dim}=1) \in \mathbb{R}^{T \times (\sum C_m) \times W_p \times H_p} $$

这里涵盖了 low-level 的纹理和 high-level 的语义动态。

**Video Former**
对于多视角 (static view, wrist view) 机器人，每个视角都会生成一个 $F_p$。由于 $F_p$ 维度极高，VPP 设计了 Video Former 将其压缩为 fixed-length tokens。

Video Former 初始化了 learnable queries $Q_{[0:T, 0:L]}$，执行 Spatio-Temporal Attention：

$$ Q' = \{\text{Spat-Attn}(Q[i], (F_p^{static}[i], F_p^{wrist}[i]))\}_{i=0}^T $$
$$ Q'' = \text{FFN}(\text{Temp-Attn}(Q')) $$

*   $\text{Spat-Attn}$: 在单帧内对多视角特征做 spatial attention。
*   $\text{Temp-Attn}$: 在时间维度上做 attention，捕捉时序动态。

输出 $Q''$ 就是一组浓缩了未来预测信息且固定长度的 condition tokens。

**Diffusion Policy Head**
VPP 使用了 Multimodal Diffusion Transformer (MDT) 作为 action head。通过 cross-attention 将 $Q''$ 注入 DiT blocks。Action 生成同样基于 diffusion 过程：

$$ \mathcal{L}_{diff}(\psi; A) = \mathbb{E}_{a_0, \epsilon, k} \| D_\psi(a_k, l_{emb}, Q'') - a_0 \|^2 $$

*   $a_0$: Ground truth action sequence。
*   $a_k = \sqrt{\bar{\beta}_k} a_0 + \sqrt{1 - \bar{\beta}_k} \epsilon$: 加噪的 action。
*   $D_\psi$: DiT denoiser network。

通过 Single-step forward pass 提取 feature，VPP 避免了 multi-step denoising 的巨大延迟，在 RTX 4090 上实现了 7-10 Hz 的闭环控制频率，这对于机器人 real-world deployment 至关重要。

---

### 3. Experiments 数据解析与直觉

VPP 的实验结果非常 striking。

**Calvin ABC->D Benchmark**
Calvin 是一个长视野任务基准，要求机器人连续完成 5 个任务。ABC->D 意味着在 ABC 环境训练，在完全未见过的 D 环境测试，极度考验 generalization。

| Method | Avg. Len ↑ |
| :--- | :--- |
| Robo-Flamingo | 2.47 |
| Uni-Pi | 0.92 |
| Susie | 2.69 |
| GR-1 (SOTA) | 3.06 |
| **VPP (Ours)** | **4.33** |
| VPP (10% Data) | 3.25 |

VPP 将 Avg. Len 从 3.06 拉升至 4.33 (相对提升 41.5%)。更夸张的是，VPP 仅用 10% 的数据就达到了 3.25，超越了用全量数据训练的 GR-1。这强烈证明了 internet-scale pre-trained VDM 提供的 physical prior 远比从头学习更高效。10% 数据足以 align action space 和 visual predictive space。

**Real-World Dexterous Manipulation**
在 12-DoF 灵巧手的实验中，VPP 在 Tool-use Tasks 上达到了 68% 的成功率。

| Method | Spoon | Hammer | Drill | Pipette |
| :--- | :--- | :--- | :--- | :--- |
| DP | 0.0 | 0.2 | 0.0 | 0.0 |
| Susie | 0.4 | 0.2 | 0.1 | 0.0 |
| GR-1 | 0.3 | 0.1 | 0.2 | 0.0 |
| **VPP** | **0.9** | **0.6** | **0.8** | **0.4** |

Spoon, Hammer, Drill, Pipette 这些物体在训练集中完全未出现。传统方法几乎全部失败 (0.0-0.2)。VPP 能成功，是因为 VDM 预测出了合理的未来轨迹，policy 只需追踪机械臂的运动学姿态，把具体的 object interaction 交给了 VDM 的 robust visual priors 去泛化。

---

### 4. 更深层的直觉与联想

VPP 的成功揭示了几个深刻的道理，我觉得对 future robotics 和 embodied AI 极具启发性：

**A. Latent Space 中的 Implicit World Model**
Yann LeCun 一直提倡 JEPA (Joint-Embedding Predictive Architecture)，即在 latent space 预测未来，抛弃生成像素的细节包袱。VPP 实际上做了一个类似的事情，尽管它基于 VDM。VPP 并不要求 VDM 生成出完美的 photorealistic future pixels，它只需要 VDM 内部 up-sampling layer 的 feature map 包含正确的 "physical evolution direction"。Figure 4 里的 visualization 显示，one-step forward pass 的图像虽然很糊，但机械臂和物体的 motion trend 已经很清晰了。这正是 latent dynamics model 的精髓：不需要高清渲染，只要动力学正确。

**B. World Models as Feature Extractors**
OpenAI 的 Sora 论文提到 "Video generation models as world simulators"。VPP 是这个理念在 robotics 上的完美落地。之前大家用 CLIP 作 text-image alignment，用 MAE 作 representation，这些都是静态的。VPP 证明了 VDM 实际上是一种 "Animated Vision Encoder"。只要模型足够大，数据足够多，它内部自然涌现了对 physics, gravity, kinematics 的理解。我们不需要显式训练一个 physics engine，一个巨大的 Transformer 通过 denoising objective 就能把物理规律隐式记住。

**C. 与 Latent Action Pretraining (LAPA) 的对比**
Ye et al. 提出的 LAPA (Latent Action Pretraining from Videos) 也是从视频中学习 action，但它是通过 inverse dynamics 在 latent space 提取 discrete latent action tokens，然后再在真机数据上 decode。VPP 和 LAPA 的方向是一致的，都是利用 video 先验。区别在于，VPP 把 inverse dynamics 的计算推迟到了 downstream policy head，VDM 本身只负责提供 "predictive visual context"。这可能更灵活，因为 inverse dynamics 严重依赖于具体机器人的运动学约束，VDM 保持 robot-agnostic，下游 small policy 学 robot-specific mapping。

**D. Data Scaling Laws in Robotics**
VPP 用了 Something-Something-v2 (Human) + RT-1/Bridge (Robot) + Self-collected。通过采样比例 $\lambda$ 平衡。这暗示了 Robotics 可能也会走向 LLM 的路线：用一个巨大的 video foundation model 吸收所有 internet video，然后用极少量的 task-specific data 做 alignment。10% 的数据量超越了 SOTA，这就是 Robotics 的 "in-context learning" 或者 "few-shot imitation" 的一种体现。

**E. 潜在瓶颈：Visual Generalization vs. Action Generalization**
尽管 VPP 在未见过的物体上泛化良好，但它的上限受限于 VDM 的 visual generalization。如果 VDM 预测的未来完全偏离物理规律 (hallucination)，policy head 可能会产生错误的 action。另外，VPP 的 action head 目前是 closed-loop，但如果 action horizon 很长，单步 VDM 的预测 horizon (T=16) 会不会不够？Autoregressive latent rollout 在极端长视野任务上可能仍然有 Compounding Error 风险。

### References & Web Links

1.  **Video Prediction Policy (VPP) Project Page**: https://video-prediction-policy.github.io
2.  **Stable Video Diffusion (SVD) Paper**: https://arxiv.org/abs/2311.15127
3.  **GR-1 (Unleashing Large-Scale Video Generative Pre-training)**: https://arxiv.org/abs/2312.13139
4.  **UniPi (Learning Universal Policies via Text-Guided Video Generation)**: https://arxiv.org/abs/2302.00111
5.  **SuSIE (Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models)**: https://arxiv.org/abs/2309.17359
6.  **Latent Action Pretraining from Videos (LAPA)**: https://arxiv.org/abs/2410.11758
7.  **Sora (Video Generation Models as World Simulators)**: https://openai.com/research/video-generation-models-as-world-simulators
8.  **Open X-Embodiment Dataset**: https://arxiv.org/abs/2310.08864
