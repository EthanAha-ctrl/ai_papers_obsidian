---
source_pdf: X-VLA Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action
  Model.pdf
paper_sha256: 483c47a651b44456f7f5a71b8d0e586df2e32c034f188dfb0653f3f6cc2d8aa7
processed_at: '2026-08-13T06:22:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# X-VLA：用“身份证”搞定多机器人混训

用人话来总结这篇 paper (https://thu-air-dream.github.io/X-VLA/)，核心就是一句话：**为了在一个大模型里同时训练各种各样不同的机器人数据，X-VLA 给每个机器人都发了一张可学习的“电子身份证”，让模型全程知道自己正在操作哪具躯体。**

下面我们拆解背后的技术直觉。

## 1. 为什么多机器人混训会崩盘？

现在搞 Vision-Language-Action (VLA) 模型，大家都想把各种机器人的数据混在一起训，这样模型能见多识广，泛化性强。但问题在于，这些数据源存在巨大的 **heterogeneity (异质性)**：
- **机械臂型号不同**：有 Franka, UR5, WidowX，还有单臂和双臂。
- **摄像头视角不同**：有 top view, wrist view, left/right view。
- **控制接口不同**：有的用绝对位置 absolute EEF pose，有的用相对位移 relative XYZ。
- **采样频率不同**：有 15Hz，有 30Hz。

如果直接把这一锅粥的数据扔给 Transformer 跑 behavior cloning，模型直接懵逼。因为对于同一个“抓取”动作，在 Franka 15Hz 的 top view 里，和在 WidowX 30Hz 的 wrist view 里，特征分布完全是两个世界。模型分不清是视角变了，还是动作变了，最后训练 loss 炸裂，学不出通用策略。

## 2. 四种解决思路的对比与 Soft Prompt 的胜出

为了解决异构性问题，paper 里对比了四种方法：

1. **Domain-specific action projection**：在模型最后输出端，给每个机器人接一个专属的 action head。
   - **直觉缺陷**：太晚了。模型在前面几十层做视觉推理和特征提取的时候，根本不知道现在是哪个机器人，等到了最后一层才区分，前面的特征早就被混在一起搅乱了。
2. **HPT-style projection** (参考 HPT: https://arxiv.org/abs/2409.20537)：在输入端，用 cross-attention resampler 把不同机器人的 observation 强行映射到一个共享的 representation space 里。
   - **直觉缺陷**：会在中间层改变 feature 分布，把预训练 VLM 原本好好的视觉表征给破坏掉，训练很不稳定。
3. **Language prompts**：直接写自然语言描述，比如给模型输入 "Embodiment: Single Franka, Camera Setup: Left View, Freq: 15Hz"。
   - **直觉缺陷**：太费人工，没法 scale。而且文字描述很难精确表达硬件底层的潜在特征。
4. **Soft Prompts (X-VLA 的方案)**：给每个数据源分配一组随机初始化的 learnable embeddings $p_i \in \mathbb{R}^k$，这组向量通过 end-to-end 训练学出来。训练时，只要输入的是 Franka 的数据，就把 Franka 专属的 Soft Prompt 塞进输入序列里；如果是 UR5，就塞 UR5 的。

**Soft Prompt 的直觉**：这就是 NLP 领域 Parameter-Efficient Prompt Tuning (Lester et al. 2021, https://arxiv.org/abs/2104.08691) 在 robotics 的延伸。不用你去费劲写怎么描述这个机器人，直接让模型自己学一组向量来代表这个机器人。这组 prompt 就像“身份证”，在模型的最早期阶段就介入，告诉 Transformer：“注意，现在操作的是 Franka 的腕部视角，频率 15Hz，你要按这个设定来提取特征”。这完美保留了预训练 VLM 的表征，同时实现了 embodiment-aware。

## 3. 架构极简主义：抛弃 DiT，回归标准 Transformer

X-VLA 的架构设计 (Figure 10) 非常干净，没有花里胡哨的 AdaLN 或 MM-DiT，就是纯粹的 self-attention。这里面有个很强的直觉：**越简单的架构越能 scale**。

**架构图解析：**
```text
[Main View Image + Language Instruction] ---> Florence-Large VLM Encoder ---> Token A
[Wrist View Image] -------------------------> Florence Vision Encoder only -> Token B (避开VLM的language对齐)
[Proprioception R_t + Noisy Action A^t + Time t] -> Linear Projection -------> Token C
[Soft Prompt p_i for Embodiment i] ----------------------------------------> Token D
                                                                              |
                                                                              v
                       [Standard Transformer Encoder Stack (24 layers, hidden=1024)]
                                                                              |
                                                                              v
                                          [Action Output Projection] -> Action Chunk
```

**关键设计直觉**：
- **分离视觉流**：主视角和语言指令送进完整的 VLM，因为固定视角的画面语义稳定，适合跟语言做 high-level reasoning。但 wrist view（手腕摄像头）画面变化快、噪声大、跟语言语义关系弱，只过 vision encoder 就行。如果强行让 wrist view 跟语言去 cross-attention，反而会把 VLM 搞坏。
- **低维信号早融合**：机器人本体感觉 $R_t$ 和 flow-matching 的噪声 action $A^t$ 物理意义相近，直接 concat 加上时间 $t$ 投影成高维向量，跟图像 token 一起进 Transformer。

## 4. 公式深扒：Flow-matching 怎么生成 Action？

X-VLA 采用 Flow-matching (Lipman et al. 2023, https://arxiv.org/abs/2210.02747) 来生成 action chunk，思路类似扩散模型，但路径更直。

核心是学一个 velocity field $v_\theta(A^t, o, t)$，把 Gaussian noise 传输到真实的 action chunk。

**训练 Loss 公式：**
$$\mathcal{L}_{\mathrm{BC}}^{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), (o,A) \sim \mathcal{D}}\Big[\left\| v_\theta(A^t, o, t) - (A - A^0) \right\|^2\Big]$$

**变量解释：**
- $\theta$：神经网络参数。
- $t \sim \mathcal{U}(0,1)$：从 0 到 1 的均匀分布里采样一个时间变量 $t$。
- $o$：多模态 observation (图像、语言、proprioception)。
- $A$：ground truth 的 expert action chunk。
- $A^0$：从标准正态分布 $\mathcal{N}(0, I)$ 采样的纯噪声 action。
- $A^t = (1-t)A^0 + t A$：在时间 $t$ 时的中间状态。通过线性插值，$A^t$ 沿着从 noise 到 target 的直线（Optimal Transport path）走。
- $v_\theta(A^t, o, t)$：模型预测的速度场，告诉当前状态应该往哪个方向移动。
- $(A - A^0)$：真实的速度方向，就是从 noise 指向 target 的直线向量。
- $\|\cdot\|^2$：L2 范数平方，用来算预测速度和真实速度的误差。

**直觉**：训练就是让神经网络学会在任何中间时刻 $t$，给当前加噪的 action $A^t$ 指出一条通往真实 action $A$ 的直线路径。推理时，从随机噪声出发，顺着模型指的路走几步，就能走到精准的 action。这比传统 diffusion 的弯曲路径好优化得多。

## 5. 实验数据表解析：为什么这么做就 work？

Paper 里的 Table 1 完美展示了架构演进的过程。我们看 Validation Error (PT) 和 Adaptation Accuracy (AD) 的关联：

| Type | Improvement | Val Error (PT) | Acc (AD) |
|---|---|---|---|
| Baseline | Florence-base + Standard DiT-base | - | 4.1 |
| Pretraining | +Custom LR + Heterogeneous PT | 0.11 | 25.0 (崩了) |
| Data Processing | +Action alignment + Balanced sampling | 0.077 | 50.0 |
| Architecture | +Replace DiT w/ Transformer + Encoding pipeline | 0.053 | 64.6 |
| Architecture | **+Soft-prompt** | **0.041** | **73.8 (+9.2)** |
| Finetuning | +Two-step adaptation | 0.032 | 95.8 |

**直觉拆解：**
1. 乱混训直接让性能掉到 25.0，连不预训练的 39.6 都不如。证明 heterogeneity 是真坑。
2. 把 DiT 换成标准 Transformer encoder，配合解耦的 encoding pipeline，效果稳步上升。标准 self-attention 让 Soft Prompt 能在每个 layer 都和全局信息做交互，这种 in-context conditioning 比 DiT 的 AdaLN 调制更稳。
3. **加上 Soft Prompt 是最大的跃升**（+9.2%）。Soft prompt 彻底吸收了不同数据源的 hardware variance，让 backbone 能安心学纯粹的通用操作逻辑。
4. Validation Error 极度契合下游 Success Rate。这意味着我们可以光看预训练的误差就能判断模型行不行，大大加速了 robotics 的研发迭代。

## 6. Two-step Adaptation 的工程直觉

当遇到一个预训练没见过的新机器人，X-VLA 采用两步微调：
1. **Prompt warm-up**：冻结整个大 backbone，只学这个新机器人的 Soft Prompt。
2. **Joint adaptation**：解冻 backbone，带着 warm-up 好的 prompt 一起微调。

**直觉**：这和 LLaVA (https://arxiv.org/abs/2304.08485) 当年训练 visual projector 是一个道理。新初始化的 Soft Prompt 是随机噪声，如果一上来就和 backbone 一起猛训，随机梯度会把预训练辛辛苦苦学到的通用特征冲烂。先冻结 backbone 只训 prompt，让 prompt 先学会怎么去“读取”预训练模型的通用知识，等 prompt 稳定了，再联合微调让整体适配新任务。

## 7. 野路子：Soft-FOLD 与 DAgger 数据采集

Paper 里介绍了一个折衣服的数据集 Soft-FOLD，只有 1200 条数据，却打平了用海量数据训练的闭源大模型 $\pi_0$。靠的是 DAgger-style (Dataset Aggregation, https://arxiv.org/abs/1011.0686) 数据采集。

**直觉**：人类折衣服的动作太随机，如果随便录点数据扔给模型，模型会被各种不一致的噪声模式带偏。X-VLA 的团队每录 100 条数据就训一个 ACT 模型，故意让模型去跑，看它在哪种状态下会挂掉，然后再针对性地去采集拯救这种失败状态的数据。这叫“定向补齐数据分布”，用极少的数据量把最关键的 corner case 覆盖掉。

## 8. 联想与发散

Soft Prompt 的成功让我联想到 LLM 里的 In-Context Learning。在 LLM 里，你给几个 example 作为 prefix，模型就能学会这个 task。X-VLA 的 Soft Prompt 本质上就是在 input 空间给模型注入了关于当前 embodiment 的 context。

未来极有可能出现一个 **Robot Prompt Library**。我们把世界上所有机器人的 hardware configuration 都预训练成对应的 Soft Prompt 存在库里。当一个全新的机器人来了，我们不需要从头采数据微调，只要提取它的 hardware descriptor（比如运动学参数），去库里检索最相似的机器人 prompt（比如发现它最像 UR5），直接拿 UR5 的 prompt 作为初始化。Paper 里 Fig. 9 的实验也验证了这个方向：用预训练好的 UR5 prompt 去初始化 WidowX 的微调，早期收敛极快。这为实现机器人的 Zero-shot 或 Few-shot 部署提供了一条可落地的技术路径。

---

# X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA Model 详细解析

## 1. 论文核心 motivation 与 problem statement

X-VLA 这篇工作由 Tsinghua AIR 与 Shanghai AI Lab 联合推出，project page: https://thu-air-dream.github.io/X-VLA/。它解决的是 generalist Vision-Language-Action (VLA) model 在 cross-embodiment 大规模 pretraining 时遇到的 **heterogeneity (异质性)** 问题。

VLA model 的 dream 是：一个 backbone 能在 diverse robotic platforms (single-arm、bi-manual、humanoid 等各种机械臂硬件) 上预训练后，再少量数据 fine-tune 到新 robot 上去部署。但现实挑战是，不同 dataset 来自不同的 hardware configuration $h_i \in \mathcal{H}$，包括：

- **arm kinetics**: 不同 robot kinematics (Franka、UR5、Agilex、WidowX 等臂型完全不同)
- **control interfaces**: 有的用 absolute EEF pose，有的用 relative XYZ + abs rotation，有的用 joint position
- **camera configurations**: top view、wrist view、head view、left/right view，频率从 15Hz 到 30Hz
- **task distributions**: 不同数据源任务分布差异巨大

这些 dimension 上 heterogeneity 导致 distribution shift + semantic misalignment，naive 把所有数据混在一起训练会让 model confuse，pretraining 不稳定、adaptation 效果差。

之前的解决方案如 $\pi_0$ (https://arxiv.org/abs/2410.24164)、GR00T-N1 (https://arxiv.org/abs/2503.14734) 主要靠 **domain-specific action decoder heads** (per-embodiment 输出头)，但只解决 action space 这一个 dimension，对 camera setup、visual domain 等其他 heterogeneity 维度无能为力。

X-VLA 的洞察是：把这些 diverse hardware configuration 当成 **meta-learning / multi-task learning 中的 task-specific features**，用 **soft prompt learning** (https://arxiv.org/abs/2104.08691, Lester et al. 2021, Power of Scale for Parameter-Efficient Prompt Tuning) 这种 parameter-efficient 方法来 absorb 这些 domain-specific 信息。

---

## 2. Preliminary 公式拆解

### 2.1 VLA behavior cloning 目标

VLA model 用 expert demonstrations $\mathcal{D} = \{\tau_j\}_{j=1}^{M}$ 训练，每个 trajectory $\tau_j = \{(o_n, a_n)\}_{n=1}^{N_j}$，其中：

- $o_n$：multimodal observation at step $n$（视觉、语言、proprioception）
- $a_n$：对应的 expert action
- $A_n := [a_n, a_{n+1}, \ldots, a_{n+T}]^T$：action chunk，$T$ 是 chunk size (借鉴 ACT, https://arxiv.org/abs/2304.13705)

Behavior cloning loss：

$$\mathcal{L}_{\mathrm{BC}}(\theta) = \mathbb{E}_{(o_n, A_n) \sim \mathcal{D}}\left[\ell\big(\pi_\theta(o_n), A_n\big)\right]$$

这里 $\theta$ 是 policy 参数，$\ell(\cdot)$ 是 supervised loss (通常是 MSE)。

### 2.2 Flow-matching policy (核心)

X-VLA 用 flow-matching (Lipman et al. 2023, https://arxiv.org/abs/2210.02747) 而不是 diffusion 来生成 action。Flow-matching 学习一个 **velocity field** $v_\theta(A^t, o, t)$ 把 Gaussian noise $A^0 \sim \mathcal{N}(0, I)$ transport 到 target action chunk $A$。

迭代过程用 Euler-Maruyama ODE solver：

$$A^{t+\Delta t} = A^t + v_\theta(A^t, o, t) \Delta t$$

- $t \in [0,1]$：continuous time variable，表示从 noise ($t=0$) 到 data ($t=1$) 的过程
- $A^t$：在 time $t$ 处的中间 action sample
- $\Delta t$：ODE 积分步长
- $v_\theta$：velocity field neural network
- $o$：observation conditioning

训练目标用 **OT (Optimal Transport) path**，linearly interpolate noise 和 target：

$$A^t = (1-t) A^0 + t A$$

Loss:

$$\mathcal{L}_{\mathrm{BC}}^{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1), (o,A) \sim \mathcal{D}}\Big[\left\| v_\theta(A^t, o, t) - (A - A^0) \right\|^2\Big]$$

- $t \sim \mathcal{U}(0,1)$：从 uniform distribution 采样 time
- $A^0$：从 $\mathcal{N}(0, I)$ 采样的 noise sample
- $A$：ground-truth expert action chunk
- 目标 velocity $A - A^0$ 是从 noise 指向 data 的方向（OT 直线路径）
- $\|\cdot\|^2$：squared L2 norm

**Intuition**：flow-matching 比 diffusion 简洁在于路径是直的（OT），没有 stochastic noise schedule 的复杂度，且 inference 可以少步采样。$\pi_0$ 也是这个思路，X-VLA 借鉴并简化了 backbone。

---

## 3. Heterogeneity 处理的 4 种 strategy 对比（核心 contribution 之一）

论文 Section 3 系统比较了 4 种 handling heterogeneity 的方案 (Fig. 2)：

### (a) Domain-specific action projection (baseline)

每个 embodiment $h_i$ 一个 output projection head，把 shared backbone 的 token 映射到该 embodiment 的 action space。

**问题**：只在最后 stage 区分 embodiment，前期 reasoning 不知道当前是哪个 robot，camera setup、visual domain 等其他维度完全忽略。

### (b) HPT-style projection (Wang et al. 2024c, https://arxiv.org/abs/2409.20537)

在 multimodal input 进 backbone 之前，对每个 domain 用 cross-attention resampler 把 observation 映射到 shared representation space。

**问题**：intermediate projection layer 改变 feature distribution，容易 corrupt 预训练 VLM 的 representation，训练 unstable。

### (c) Language prompts

给每个 hardware setup 写自然语言描述，比如 "Embodiment: Single Franka, Camera Setup: Left View / Wrist View, Freq: 15Hz"，concat 到 task instruction 后面送进 VLM。

**问题**：要 handcraft 描述模板，scalability 差，且文本描述无法精确 capture hardware-specific latent feature。

### (d) Soft Prompts (X-VLA 选择)

给每个 data source $i \in \{1, \ldots, H\}$ 分配一组 learnable embeddings $p_i \in \mathbb{R}^k$，作为该 embodiment 的 prompt token。

形式化：$P^H = \{p_i\}_{i=1}^H$，期望 $p_i \approx \Phi(h_i)$，其中 $\Phi: \mathcal{H} \to \mathbb{R}^k$ 是 implicit mapping 从 hardware configuration 到 prompt space。

**关键点**：$\Phi$ 不预定义，是随机初始化后 end-to-end 优化得到的。Soft prompt 在 **action generation 的早期阶段**注入 backbone，引导 backbone 做 embodiment-aware learning。

**为什么 soft prompt 最好？**
- 保留 backbone 预训练 representation（不像 HPT 改 input distribution）
- 可 scale（不像 language prompt 要 handcraft）
- 提前融入 embodiment 信息（不像 action projection 只在最后区分）

Fig. 4 的 training curves 明确显示：soft prompt 训练 loss 最稳定、收敛最好；HPT-style 和 language prompt 都有 unstable dynamics。

---

## 4. X-VLA 架构详细解析 (Section 4.1, Fig. 10)

X-VLA 整体架构是 **VLM encoder + standard Transformer encoder stack**，没有任何 DiT-style AdaLN 或特殊 conditioning，靠 soft prompt + 简单 self-attention 就能 work。具体两条 stream：

### 4.1 High-dimensional observation stream

输入是多视角图像 Img = $\{\text{img}_v\}$ + 语言指令 $L$。

**关键设计：disentangle streams**。大多数 prior work ($\pi_0$、Octo https://arxiv.org/abs/2405.12213、GR00T-N1) 把所有 views 和 instruction 都塞进 VLM。X-VLA 分开：

- **Main view (fixed-view) + instruction**：走完整的 Florence-Large VLM (Xiao et al. 2024, https://arxiv.org/abs/2311.06242) encoder，因为 fixed-camera view 语义稳定，适合 VLM 高层 reasoning；
- **Auxiliary views (e.g., wrist view)**：只走 vision encoder (不加 language decoder 部分)，因为 wrist view 噪声大、变化快、对 fine-grained manipulation 重要，但和 language 没强 alignment，分开编码避免语义 gap 拖累 VLM。

这个设计直觉上很对：wrist view 主要是 low-level geometric cue，跟 "fold the cloth" 这种指令语义关系弱，没必要让 VLM 的 cross-attention 把它和 text 强行 align。

### 4.2 Low-dimensional proprioceptive-action stream

Proprioception $R_t$ (joint position、end-effector pose 等) + action-related tokens $A_t$ (flow-matching 的 noisy action sample) + time embedding $t$：

- $R_t$ 和 $A_t$ 都是 compact vector，物理语义相关
- 直接 concatenate $[R_t; A_t; t\text{-embedding}]$ 
- 用 lightweight linear layer 投影到高维 feature space
- 早期 fusion，让 model 有 robust proprioceptive-temporal grounding

### 4.3 完整 forward pipeline (Fig. 10)

```
[Main view img + instruction L] ─── Florence-Large VLM ───┐
[Wrist view img_v] ─── vision encoder only ────────────────┤
[Proprioception R_t + noisy action A^t + time t] ── linear ─┤
[Soft prompt p_i for current embodiment] ──────────────────┤
                                                             ▼
                          [standard Transformer encoder × 24 layers, hidden=1024]
                                                             ▼
                          [output projection → action chunk A]
```

- **绝大部分参数 shared**（VLM encoder + 24-layer Transformer backbone）
- **Unshared 参数仅 0.04%**：soft prompt + action-related input/output linear projection
- Bi-directional attention 让所有 modality 互相 fuse

### 4.4 为什么不用 DiT / MM-DiT / $\pi_0$-style？

Table 4 对比 validation error：

| Architecture | Val Error |
|---|---|
| DiT (AdaLN conditioning) | 0.077 |
| MM-DiT (separate modality params + attention) | 0.140 |
| $\pi_0$-style (parallel MLP-Mixer + VLM) | 0.056 |
| **X-VLA (Transformer encoder + soft prompt)** | **0.041** |

- MM-DiT 最差，因为 heterogeneous data 上 separate modality params 让训练 unstable
- $\pi_0$-style 复杂、参数多
- X-VLA 最简洁，validation error 最低

**Intuition**：standard self-attention + soft prompt 这种 in-context conditioning 方式比 AdaLN（DiT）或 parallel stream（$\pi_0$）更 stable，因为 soft prompt 作为 token 在每个 layer 都直接参与 attention，让 backbone 全程 embodiment-aware，而不是靠 modulation signal 间接控制。

---

## 5. Training Recipe 细节 (Section 4.2)

### 5.1 Pretraining + Finetuning pipeline

**Phase I (Pretraining)**：backbone $\pi_\theta$ 和所有 soft prompts $P^H$ jointly 用 flow-matching loss $\mathcal{L}_{\mathrm{BC}}^{\mathrm{FM}}$ 优化。

**Phase II (Adaptation)**：两步 procedure：

**(1) Prompt warm-up**：
- 引入新 prompt $p_{\mathrm{new}} \in \mathbb{R}^k$ for 新 hardware $h_{\mathrm{new}}$
- Pretrained weights frozen，只 update $p_{\mathrm{new}}$
- 让 prompt 先学会利用 backbone 的 embodiment-agnostic feature
- 1000 iterations

**(2) Joint policy adaptation**：
- Unfreeze backbone，jointly optimize backbone + warmed-up prompt
- 学习率逐渐 warm-up 到默认值 (再 1000 iterations)
- 完成新 domain specialization

**为什么两步？** 借鉴 VLM 训练 (LLaVA, https://arxiv.org/abs/2304.08485) 中 project layer warm-up 的思路：先让 new adapter/prompt 适配预训练 representation，避免直接 joint train 时随机初始化的 prompt 破坏 backbone。

### 5.2 Custom learning rate (LR)

关键 stabilization：soft prompt 和 vision-language modules 用 **reduced learning rate**。

**Intuition**：VLM 的 pretrained representation 是宝贵的 general reasoning 能力，如果用大 LR 快速 update，会发生 **catastrophic drift**，导致 pretraining 知识丢失。Reuss et al. 2025 (FLOWER, https://openreview.net/forum?id=ifo8oWSLSq) 和 Driess et al. 2025 (https://arxiv.org/abs/2505.23705) 也观察到这点。X-VLA 的 reduced LR 让 VLM 渐进适应 robotic grounding 而非 overwrite。

### 5.3 Aligned action representation

所有 dataset 的 action 统一到 **end-effector (EEF) pose**：

- **Position**：Cartesian xyz
- **Rotation**：Rotate6D (Zhou et al. 2019, https://arxiv.org/abs/1812.07035)，避免 Euler angle 的 discontinuity 和 quaternion 的 antipodal symmetry
- **Gripper state**：discretized binary

Loss 分解：
- Position + Rotation: MSE
- Gripper: BCE

### 5.4 Intention abstraction via temporal downsampling

低层 trajectory 太 fine-grained，含人类 random noise。X-VLA 不预测每一步 EEF pose，而是生成 **30 anchor points** 总结未来 4 秒的 intended trajectory。

**Intuition**：pretraining 阶段不需要学 low-level control，只要学 high-level intention。anchor points 是 abstract representation，让 model 关注 "我要去哪" 而非 "我每一步动多少"。这招和 latent action pretraining (Ye et al. 2024, https://arxiv.org/abs/2410.11758, LAPA) 有点类似思路。

### 5.5 Balanced data sampling

不用 round-robin，而是 **cross-domain + cross-trajectory shuffle**：每次 iteration 都让 model 见到 diverse balanced mixture，防止 dominant domain overfitting。

Table 8 给出各数据源采样权重：

| Data source | Num traj | Sampling weight |
|---|---|---|
| AGIBOT | 141K | 0.4 |
| Droid-Left | 45K | 0.15 |
| Droid-Right | 45K | 0.15 |
| RoboMind-Franka | 19K | 0.1 |
| RoboMind-Dual-Franka | 2K | 0.03 |
| RoboMind-UR | 25K | 0.1 |
| RoboMind-Agilex | 11K | 0.07 |

注意 AGIBOT 权重最高 (0.4)，虽然它 traj 数不是最多，可能因为它质量高、覆盖广。Dual-Franka 数据少但 weight 0.03，防止 underfitting。

---

## 6. Ablation path (Table 1) - 一步步看 component 贡献

Table 1 给出完整 ablation 路径，绿色 = 正面、红色 = 负面、灰色 = 中性：

| Type | Improvement | Val Error (PT) | Acc (AD) |
|---|---|---|---|
| Baseline (w/o PT) | Florence-base + Standard DiT-base | - | 4.1 |
| Pretraining Technique | +Custom LR (w/o PT) | 0.11 | 39.6 (+35.5) |
| | +Heterogeneous PT | | 25.0 (-14.6) ⚠️ |
| Data Processing | +Action alignment | 0.077 | 50.0 (+25.0) |
| | +Intension abstraction | | |
| | +Balanced data sampling | | |
| Architecture Design | +Replace DiT with Transformer encoder | 0.071 | 47.9 (-2.1) |
| | +Encoding pipeline | 0.053 | 64.6 (+16.7) |
| | +Soft-prompt | 0.041 | 73.8 (+9.2) |
| | +Scaling up | 0.032 | 89.6 (+15.8) |
| Finetuning Technique | +Two-step adaptation | 0.032 | 95.8 (+6.2) |

**几个关键观察**：

1. **Heterogeneous PT 单独没用反而 hurt** (25.0 < 39.6)：naive 把 heterogeneous data 混训反而比不 PT 还差，验证了 heterogeneity 是真问题。

2. **Validation error 和 downstream acc 强相关**：val error 从 0.11 → 0.032，acc 从 25% → 95.8%。所以可以用 val error 作为 PT performance 的 proxy。

3. **Replace DiT with Transformer encoder 单独看 hurt** (47.9 < 50)：单纯换 backbone 不一定 work，要和 encoding pipeline 配合。但配合后大幅提升 (64.6)。

4. **Soft prompt 是关键一跳**：从 0.053 → 0.041 val error，acc +9.2%。这步才真正解决 heterogeneity。

5. **Two-step adaptation** final 加 +6.2%，证明 Phase II 的 prompt warm-up + joint train 策略有效。

---

## 7. Scaling behavior (Section 5.1, Fig. 5)

三个轴 scaling 实验：

- **Model capacity**：从 base 到 0.9B (hidden 1024, 24 layers)
- **Data diversity**：从单一数据源到 7 个 data source
- **Data volume**：从 50K 到 290K episodes

Fig. 5 显示三条 scaling curve 都没 saturation，0.9B + 290K + 7 sources 还在持续下降。论文说受 compute 限制没继续 scale，但 trend 暗示更大模型 + 更多数据能继续提升。

**对比 LLM scaling law**：这点很有意思。LLM scaling law 在 text domain 非常成熟 (Chinchilla, https://arxiv.org/abs/2203.15556)。但 robotic data 异质性大、scaling behavior 一直 unclear。Wang et al. 2024c (HPT) 和 Lin et al. 2025 (https://openreview.net/forum?id=pISLZG7ktL) 都观察到机器人 data 也有 scaling law，但 slope 受 data heterogeneity 影响。X-VLA 用 soft prompt 把 heterogeneity absorb 掉，让 scaling 比较干净。

---

## 8. Adaptation experiments (Section 5.2)

### 8.1 Simulation benchmarks (Table 2)

X-VLA-0.9B 在 6 个 simulation benchmark 上对比：

- **LIBERO** (4 splits: Spatial/Object/Goal/Long, http://libero-project.github.io/)
- **Simpler** (VM, VA, WidowX, https://simpler-env.github.io/)
- **Calvin ABC→D** (long-horizon, https://calvinrobot.github.io/)
- **RoboTwin-2.0** (bimanual, https://arxiv.org/abs/2506.18088)
- **VLABench** (long-horizon reasoning, https://github.com/VLABench/VLABench)
- **NAVSIM** (autonomous driving, https://github.com/autonomousvision/navsim)

X-VLA-0.9B 在 5/6 benchmark 上 SOTA：

| Benchmark | X-VLA | Prior SOTA |
|---|---|---|
| Simpler-VM | 80.4 | 78.0 (FPC-VLA) |
| Simpler-VA | 75.7 | 72.7 (MemoryVLA) |
| Simpler-WidowX | **95.8** | 71.9 (MemoryVLA) |
| LIBERO-Spatial | 98.2 | 98.4 (MemoryVLA) |
| LIBERO-Object | 98.6 | 98.8 ($\pi_0$) |
| LIBERO-Goal | 97.8 | 97.9 (OpenVLA-OFT) |
| LIBERO-Long | **97.6** | 94.5 (OpenVLA-OFT) |
| LIBERO-Avg | **98.1** | 97.1 |
| Calvin ABC→D | 4.43 | 4.53 (FLOWER) |
| RoboTwin Easy | **70.0** | 46.4 ($\pi_0$) |
| RoboTwin Hard | **39.0** | 16.4 ($\pi_0$) |
| VLABench | **51.1** | 39.7 (GR00T-N1) |
| NAVSIM PDMS | **87.3** | 81.7 (UniVLA) |

**亮点**：0.9B 参数打败一众 3B ($\pi_0$)、7B (OpenVLA, MemoryVLA)、9B (UniVLA) 大模型，证明 soft prompt + 干净架构胜过 brute-force scale。特别 RoboTwin bimanual 任务上提升巨大（70% vs 46.4%），说明 soft prompt 在 bi-manual 这类高 heterogeneity 场景特别 work。

### 8.2 Real-world experiments (Fig. 7, 14)

3 个真实 robot：
- **WidowX** (single-arm, BridgeData v2 evaluation, https://arxiv.org/abs/2308.12952)：5 个 task，X-VLA 全部超过 baseline
- **AgileX** (bi-manual, Soft-FOLD cloth folding)
- **AIRBOT** (unseen during PT, 200 demos PEFT)

### 8.3 Soft-FOLD: dexterous cloth-folding (Appendix F)

引入 **Soft-FOLD** dataset（1200 episodes），在 bi-manual Agilex 上做 cloth folding：

- 用 **DAgger-style** (Ross et al. 2011, https://arxiv.org/abs/1011.0686) 数据采集：每 100 episode 训 ACT，识别 failure mode，targeted 补数据
- 任务分两阶段：smoothing (cloth 从混乱状态展平) + folding
- Smoothing 难度大，用 swinging motion 直到 keypoints (corner) 出现再 transition

X-VLA-0.9B 在 Soft-FOLD 上 throughput ~33 folds/hour，success ~100%，**与 closed-source $\pi$-folding 模型 comparable**（$\pi$ 应该用了大得多的数据）。

对比 baseline：
- $\pi_0$-base finetune 在 Soft-Fold 上：不如 X-VLA
- ACT from scratch：完全打不过

这结果挺 impressive，因为 cloth folding 是高度 dynamic deformable object manipulation，传统 method 很难。

### 8.4 PEFT experiments (Table 3)

用 LoRA (https://arxiv.org/abs/2106.09685) 测 parameter-efficient adaptation：

- **9M tunable params** (约 1% 总参数)
- LIBERO-Spatial: 95.4 (full $\pi_0$ 3B: 96.8)
- LIBERO-Object: 96.6 ($\pi_0$: 98.8)
- LIBERO-Goal: 96.0 ($\pi_0$: 95.8)
- LIBERO-Long: 84.2 ($\pi_0$: 85.2)
- Simpler-WidowX: 54.2 ($\pi_0$: 55.7)

**用 300× 更少的参数 (9M vs 3B) 就达到 comparable performance**，证明 X-VLA backbone 真的学到 embodiment-agnostic feature，LoRA 只需调出 domain-specific 的部分。

Table 6 数据效率实验：只用 10 demos 还能达到 91.1% on Libero-Goal (50 demos 92.8%)，data efficiency 极高。

---

## 9. Soft prompt 可解释性分析 (Section 5.3)

### 9.1 T-SNE 可视化 (Fig. 8)

7 个 data source 的 soft prompt 用 T-SNE 投影：

- 形成 well-structured cluster，每个 hardware configuration 一个 cluster
- **关键观察**：Droid-Left 和 Droid-Right 的 prompt 互相 intermingled，而不是分开
  - 因为它们只是 main view 不同（left vs right），本质都是 Franka single-arm
  - 说明 soft prompt 不 brute-force partition，而是利用 cross-embodiment similarity
  - 这点很有意思：prompt 在 latent space 自然捕捉到 robot 本质相似性

### 9.2 Cross-embodiment prompt transfer (Fig. 9)

WidowX (unseen in PT) 上 PEFT，3 种 setting：
1. **Random init prompt frozen**：slow adaptation, low final acc
2. **Pretrained UR5 prompt frozen** (single-arm similar to WidowX)：早期 transfer benefit 明显，但 final 受 domain gap 限制
3. **Two-step adaptation** (X-VLA 推荐)：最快、最高

**关键 insight**：pretrained prompt 在相似 embodiment 之间能 transfer，暗示未来 zero-shot/few-shot generalization 可以通过 "prompt retrieval" 实现——新 robot 找最相似的 pretrained prompt 作 init。

---

## 10. Failure attempts (Appendix E)

诚实记录两个失败尝试：

### 10.1 Heterogeneous Low-rank Adapter

给每个 domain 一个 LoRA-style adapter 并行 shared backbone。期望 adapter absorb domain variation，backbone 学通用 feature。**结果**：adapter 和 backbone 优化 dynamics 冲突，训练 unstable，泛化差。

**反思**：LoRA 设计是为 single downstream task，不是为 heterogeneous multi-domain pretraining。Soft prompt 之所以 work，可能因为 prompt 是 input-side intervention，比 LoRA 这种 weight-side intervention 更"轻"，不会和 backbone optimization 直接冲突。

### 10.2 Heterogeneity-guided MoE

设计 router 根据 embodiment cue 激活不同 expert。**结果**：router collapse，大部分 input 都路由到少数 expert，其他 expert underutilized。加 load-balancing regularization (https://arxiv.org/abs/2408.15664) 又导致 expert 频繁切换、训练 unstable。

**反思**：MoE 在 NLP multi-task work 是因为 task 之间差异大但每个 task 数据多；robotics 数据少且 embodiment 之间相似度高，router 难学到有意义的 routing。

---

## 11. Limitations & Future work (Appendix N)

1. **Scale 还小**：0.9B vs LLM 的 100B+ 量级。受限于 robotics data 规模和质量。
2. **Supervision 信号弱**：低维 action label 信息含量有限，缺 high-level reasoning、intent、multi-step dependency。Temporal downsampling 只是 heuristic，未根本解决。Future：3D spatial cue、physical dynamics、subgoal annotation、self-supervised objective。
3. **非 plug-and-play**：仍需 embodiment-specific adaptation + 少量 demonstration。Future：universal kinematic descriptor、physics-informed prior 减少对 task-specific data 的依赖。

---

## 12. 与相关工作对比 intuition

- **vs $\pi_0$ (https://arxiv.org/abs/2410.24164)**：$\pi_0$ 用 parallel MLP-Mixer action expert + PaliGemma VLM，架构复杂；X-VLA 用纯 Transformer encoder + soft prompt，更简洁，0.9B 打 $\pi_0$ 3B。
- **vs GR00T-N1 (https://arxiv.org/abs/2503.14734)**：NVIDIA 的工作也用 domain-specific action head，但没解决 camera/visual heterogeneity；X-VLA soft prompt 全维度 absorb。
- **vs HPT (https://arxiv.org/abs/2409.20537)**：HPT 用 input projection，容易 corrupt VLM；X-VLA soft prompt 保留 VLM representation。
- **vs UniAct (https://arxiv.org/abs/2506.19850)**：UniAct 用 universal action token，思路类似但侧重 action space alignment，X-VLA 更全面。
- **vs MemoryVLA (https://arxiv.org/abs/2508.19236)**：MemoryVLA 加历史 reasoning，7B model；X-VLA 0.9B 不靠 memory 也 competitive。

---

## 13. 我的 critical takeaways

1. **Soft prompt 这个 idea 在 robotics 终于 work**：NLP 早用烂了，但 robotics 之前没人系统验证。X-VLA 用大量实验证明它处理 heterogeneity 比 action head、HPT projection、language prompt 都好。

2. **Architecture simplicity wins**：标准 Transformer encoder + soft prompt + Florence VLM，没有 AdaLN、没有 MM-DiT、没有 parallel stream，反而最 stable。这呼应了 LLM 领域 GPT-style decoder 一统天下——简单架构 + scaling 比 fancy architecture 更 scalable。

3. **Validation error 作为 PT proxy 很实用**：robotics 实验 expensive，能用 val error 替代 downstream success rate 大幅加速 research iteration。

4. **Soft-FOLD dataset + DAgger** 是个 hidden gem：1200 episode 达到 closed-source 模型水平，DAgger-style 迭代数据采集是低资源场景的 practical recipe。

5. **Pretrained prompt transfer** 实验很有启发性：相似 embodiment 的 prompt 能 transfer，未来 prompt retrieval 方向很有想象空间——build 一个 prompt library，新 robot 自动检索最相似 prompt 作 init。

6. **Limitation 诚实**：承认 0.9B 还小、supervision 信号弱、非 plug-and-play。这些是 VLA 领域共性 challenge，X-VLA 没声称解决但提供了 incremental progress。

7. **Soft prompt 可能是 in-context learning 在 robotics 的体现**：LLM in-context learning 通过 prompt 激活不同 task 能力；X-VLA soft prompt 类似地激活不同 embodiment 能力，这可能是 foundation model 适应 multi-domain 的通用 mechanism。

整体来说，X-VLA 是 VLA 领域一个 solid 的工程贡献，soft prompt + 简洁架构 + 完整 ablation + 6 simulation + 3 real-world 的大规模 evaluation，确实可以作为 future VLA 研究的 strong baseline。Project page: https://thu-air-dream.github.io/X-VLA/，code 应该会开源。
