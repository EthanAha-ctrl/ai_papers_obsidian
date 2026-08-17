---
source_pdf: ProPhy Progressive Physical Alignment for Dynamic World Simulation.pdf
paper_sha256: 7125b919cc739c8ca57275902f0bd977b48f3fb121415c8c7b7d5a482009be76
processed_at: '2026-08-06T06:57:00-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，用最直白的话来说，这篇 paper 在干这么一件事：

## 一句话说清楚

现在的 video generation 模型生成视频时"不懂物理"——球能穿墙、水能往上流、碰撞后动量凭空消失。之前的方法要么"整体"判断这个视频大概是什么物理场景，要么干脆让模型自己瞎猜。ProPhy 的做法是**分两步**先把物理语义搞清楚，再**逐像素**地把物理规律对齐到正确的位置上。

## 为什么要这么搞

想象一个画面：一个人在雪地里篝火旁倒咖啡。

这里同时存在好几个物理现象——篝火在烧、咖啡在流、雪花在飘、火光照亮了周围。

之前的方法（比如 WISA）只能告诉你"这个视频里有燃烧、流体、降雪"，然后给整个视频加一个 global 的物理 guidance。问题是：**火焰该在哪个位置烧？咖啡该往哪个方向流？** 这些 coarse 信息根本回答不了。模型最后只能"差不多"地生成，结果经常出现火焰蔓延到雪地上、咖啡悬浮在空中之类的荒诞画面。

而 VideoREPA 那种 implicit 方法更惨——它试图让模型通过观看大量视频"悟"出物理规律，但没有告诉模型什么是物理、什么不是，结果模型学了一堆 surface pattern，遇到复杂场景就露馅。

ProPhy 的核心 insight 是：**物理现象在空间上是有定位的**。火焰在篝火位置，咖啡在杯子和地面之间，雪花在整个天空。如果不把物理规律"钉"到正确的空间位置上，模型永远生成不了物理正确的视频。

## 怎么做的

ProPhy 搞了一个两阶段的 "Mixture-of-Physics-Experts" 系统：

**第一阶段：Semantic Expert Block (SEB)**
读 text prompt，判断"这个视频大概涉及哪些物理类别"。比如看到"雪地篝火倒咖啡"，它会激活"燃烧""流体""降雪"这几个 expert。每个 expert 是一个 learnable 的 basis map，加到 video latent 上。这一步是 video-level 的，coarse 的，相当于告诉模型"你要注意这些物理现象"。

**第二阶段：Refinement Expert Block (REB)**
这是关键的 fine-grained 部分。它看每一个 token（视频 latent 中的每一个空间位置），判断"这个位置上具体在发生什么物理现象"。火焰位置的 tokens 激活燃烧 expert，咖啡位置的 tokens 激活流体 expert，天空位置的 tokens 激活降雪 expert。这样物理 guidance 就是 spatially anisotropic 的——不同位置响应不同的物理规律。

## 最聪明的部分：监督信号从哪来

这是这篇 paper 最 clever 的地方。

你训练 REB 需要 ground truth——每个 token 应该激活哪个物理 expert。但人工标注成本太高，而且物理现象的边界很模糊。

作者的发现：**VLM（如 Qwen2.5-VL）在 spatial grounding 上比 video diffusion model 强太多了**。你问 VLM"这个视频里火在哪里"，它的 attention map 能准确指到火焰位置；而 VDM 的 cross-attention 加了噪声去噪后，根本指不准。

所以 ProPhy 的策略是：
1. 问 VLM"描述一下视频里的物理现象"，拿到 attention map（物理现象在哪里）
2. 再问 VLM 一个 generic 问题（"描述一下视频背景"），拿到 background attention map
3. 两者相减，得到一个干净的 physical localization map
4. 用这个 map 去监督 REB 的 Refinement Router

本质上就是**把 VLM 的 spatial grounding 能力蒸馏到 video generator 里**。VLM 负责"看懂物理在哪里"，REB 负责"在正确的位置施加物理约束"。

## 为什么这个思路 work

核心 insight 是 VLM 和 VDM 的能力是互补的：
- VLM 理解物理语义、能定位物理现象，但生成不了视频
- VDM 能生成高质量视频，但不知道物理规律该往哪儿施加

ProPhy 把两者的长处拼起来：用 VLM 的理解能力做 teacher，训练 VDM 里的物理 router，让 VDM 学会"在哪里施加什么物理"。

推理的时候 VLM 不参与了，router 已经学会了物理定位，整个 pipeline 是 end-to-end 的。

## 实验结果说明了什么

在 VideoPhy2 benchmark 上，CogVideoX-5B + ProPhy 在 Joint metric（物理正确且语义正确的比例）上达到 26.7%，而 baseline CogVideoX 只有 22.3%，VideoREPA 只有 22.0%，WISA 只有 25.8%。

更重要的是 VBench 的 Dynamic Degree 指标：CogVideoX baseline 是 46.8，加了 ProPhy 直接飙到 72.0。这说明 ProPhy 不只是让视频"物理正确"，还让视频"更动态"——因为物理规律本身是 dynamic 的，对齐了物理就自然产生了更真实的运动。

最有趣的实验是 Expert Inversion（Figure 8）：把 router 的 logits 反过来，刚性的车门就会像布料一样飘动。这说明不同 expert 确实学到了不同的物理先验，而且可以 controllable——你有了物理属性的"旋钮"。

## Limitation 也很诚实

作者承认，这个方法本质还是在 fit data pattern，没有 enforce 真正的物理方程。物理分类只是限制了 expert 的参数空间到一个子集，但生成时还是靠 pattern matching 而不是求解 Newton's laws。

未来如果把物理微分方程嵌进去——比如流体 expert 内部求解简化的 Navier-Stokes，刚体 expert 求解 Hamiltonian——那才是真正从 pattern fitting 走向 principled simulation。

## 我的看法

这篇 paper 的核心 contribution 不在于架构多复杂，而在于一个很 practical 的 insight：**与其让 video model 从头学物理，不如借 VLM 的眼睛告诉它物理在哪里**。这个 cross-modal distillation 的思路很 elegant，而且 scalable——VLM 越强，ProPhy 的物理对齐就越准。它本质上是在 video generation 的 latent space 里做了一个 "physics grounding"，类似于 NLP 里的 entity grounding——把抽象的概念锚定到具体的 spatial location 上。

---

Andrej，这篇 paper《ProPhy: Progressive Physical Alignment for Dynamic World Simulation》非常契合你近期对 Video Generation Models 作为 World Simulator 的关注。当前的 Video Diffusion Models (VDMs) 如 Sora、Wan2.1、CogVideoX 在视觉质量上已经非常惊艳，但在物理一致性上依然存在巨大的 gap。ProPhy 的核心贡献在于提出了一种 Progressive Physical Alignment 框架，通过两阶段的 Mixture-of-Physics-Experts (MoPE) 机制，将 Vision-Language Models (VLMs) 的物理推理能力以 fine-grained 的方式蒸馏到 video generator 中，从而实现 spatially anisotropic (空间各向异性) 的物理响应。

下面我为你详细拆解这篇 paper 的技术细节、架构设计、公式推导以及实验数据，希望能 build up 你的 intuition。

### 1. 核心动机与问题分析

现有的 physics-aware video generation 方法主要存在两个痛点：
1. **Implicit physical guidance**: 例如 VideoREPA 和 PhysMaster，它们试图让模型隐式地学习物理规律，然而缺乏 explicit physical priors 导致在复杂场景下频繁违反基本物理定律（如动量不守恒、物体穿模）。
2. **Video-level module routing**: 例如 WISA，虽然引入了 Mixture-of-Physics-Experts (MoPE)，但它仅在 video level 进行 routing。如果视频中同时存在多个局部物理现象（例如铁球碰撞的同时伴随火花飞溅），coarse 的 video-level guidance 会 dispersing physical awareness，无法 focus on critical local areas。

ProPhy 的解决思路是引入 **Progressive Physical Alignment**。它将物理先验的提取分为两个阶段：Semantic Expert Block (SEB) 负责从 text prompt 推断 video-level 的物理语义；Refinement Expert Block (REB) 负责在 token level 进行 fine-grained 的物理动态对齐。

### 2. 架构详解

ProPhy 构建于 latent video diffusion backbones (如 Wan2.1-1.3B 和 CogVideoX-5B) 之上。除了原始的 backbone，ProPhy 引入了一个额外的 Physical Branch，包含三个核心组件：
1. **Semantic Expert Block (SEB)**: 接收 text embedding，输出 video-level 物理先验。
2. **Physical Blocks (PB)**: 结构与 backbone 的 Transformer Block 相同，并使用 backbone 的权重进行初始化，用于逐步累积物理信息。
3. **Refinement Expert Block (REB)**: 附着在最后一个 PB 上，执行 token-level 的物理属性细化。

#### 2.1 Semantic Expert Block (SEB) 与 Progressive Routing

SEB 的作用类似于一个 video-level 的物理路由器。它包含 $E_s$ 个 learnable physical basis maps $\boldsymbol{B_e} \in \mathbb{R}^{N \times C}$，每个 $\boldsymbol{B_e}$ 代表一种特定的物理知识（如燃烧、反射、刚体碰撞）。
这里的变量含义如下：
*   $N = (F / r_f) \times (H / r_s) \times (W / r_s)$：latent tokens 的数量。$F, H, W$ 分别是 video 的 length, height, width；$r_f, r_s$ 是时间和空间的下采样率。
*   $C$：latent dimension。

对于输入的 text embedding $y$，Semantic Router 输出一个归一化的权重向量 $\boldsymbol{\rho}_p \in \mathbb{R}^{E_s}$，控制每个 basis map 的贡献。增强后的 latent 表示为：

$$
\tilde{\boldsymbol{X}} = \boldsymbol{X} + \sum_{e=1}^{E_s} \rho_p^e \boldsymbol{B}_e \quad \text{(Eq. 1)}
$$

**Intuition**: 这里没有使用传统的 top-k MoE，因为 small batch sizes 训练时，standard top-k MoE 极易发生 mode collapse（只有少数 expert 被反复激活）。ProPhy 采用了 continuous weighted formulation（连续加权），让所有 expert 都有不同程度的参与，从而避免了 mode collapse。

#### 2.2 Refinement Expert Block (REB) 与 Token-level Routing

REB 操作在 token level。对于 $\tilde{\boldsymbol{X}}$ 中的每一个 token $\tilde{\boldsymbol{x}} \in \mathbb{R}^C$，Refinement Router 输出一个概率分布 $\boldsymbol{\rho}_r \in \mathbb{R}^{E_r}$，表示该 token 属于不同物理定律的概率。由于 token 数量巨大，且后续引入了 fine-grained alignment，mode collapse 的风险较小，因此这里采用了标准的 top-k MoE 策略：

$$
\tilde{\boldsymbol{x}}' = \sum_{i \in \mathrm{argtop}_k \boldsymbol{\rho}_r} \rho_r^i \mathbf{e}_\theta^i(\tilde{\boldsymbol{x}}) \quad \text{(Eq. 2)}
$$

**Intuition**: $\mathbf{e}_\theta^i$ 是第 $i$ 个 expert 的 forward function。这里使用 top-k 可以让模型在空间上产生 anisotropic response。例如，画面左侧是海浪（流体力学），右侧是沙滩（刚体摩擦），REB 能够让左侧的 tokens 激活流体 expert，右侧的 tokens 激活摩擦 expert，从而实现局部物理规律的精准对齐。

### 3. Physical Alignment Objectives

ProPhy 的精髓在于如何训练这两个 Router。作者设计了两个独立的 alignment objectives。

#### 3.1 Semantic Alignment (Coarse Alignment)

这个 loss 用于训练 SEB 的 Semantic Router。它利用了 WISA-80K 数据集中的 per-video physical category vector $\boldsymbol{q}_s \in \mathbb{R}^{E_{\mathrm{wisa}}}$。通过一个 linear layer 将 $\boldsymbol{\rho}_s$ 映射到与 $\boldsymbol{q}_s$ 相同的维度。对于 batch size 为 $B$ 的数据，计算 cosine-similarity pairwise matrix：

$$
P_s^{i,j} = \frac{\boldsymbol{\rho}_s^{(i)} \cdot \boldsymbol{\rho}_s^{(j)}}{\lVert \boldsymbol{\rho}_s^{(i)} \rVert \lVert \boldsymbol{\rho}_s^{(j)} \rVert} \quad \text{(Eq. 3)}
$$

同理计算 label matrix $\boldsymbol{Q}_s \in \mathbb{R}^{B \times B}$。Semantic Alignment objective 为：

$$
\mathcal{L}_{\mathrm{coarse}} = \sum_{1 \leq i < j \leq B} \| P_s^{i,j} - Q_s^{i,j} \|_2 \quad \text{(Eq. 4)}
$$

**Intuition**: 这个 loss 并不直接强制 $\boldsymbol{\rho}_s$ 匹配 one-hot 标签，而是强制 batch 内的样本保持相对距离。属于同一物理类别的样本应有相似的 routing weights，不同类别的样本 routing 差异要大。这种基于距离的 loss 比 BCE 更适合处理物理现象之间的模糊边界（例如燃烧和爆炸往往高度相关）。

#### 3.2 Fine-grained Alignment (VLM Distillation)

这是本文最关键的创新点。作者观察到 VLMs（如 Qwen2.5-VL-32B）在 spatial understanding of physical dynamics 上远胜于 generative models。因此，ProPhy 将 VLM 的 fine-grained localization capability 蒸馏到 REB 中。

**获取 VLM 监督信号的过程**：
1. 向 VLM 输入 video 和关于目标物理现象的 question。
2. 提取 VLM 生成 answer 时的 attention scores（query tokens 为 text answer tokens，key tokens 为 video tokens），得到 physical phenomenon map。
3. 向 VLM 输入一个 generic prompt，获取 background attention map。
4. 两者相减得到最终的 token-level alignment targets $\boldsymbol{Q}_r \in \mathbb{R}^{N \times E_{\mathrm{attn}}}$。

**构建 Mask 与 Loss**：
定义 mask $\boldsymbol{M} \in \mathbb{R}^{N \times E_{\mathrm{attn}}}$。对于标注存在的物理现象设为 1，其余为 0。同时，如果 $\boldsymbol{Q}_r$ 中某些值为负数（表示该区域没有明显的物理现象），则通过 $M = M \land \mathrm{sign}(Q_r)$ 将这些区域剔除。Fine-grained alignment loss 为：

$$
\mathcal{L}_{\mathrm{fine-align}} = \sum_{M^{i,e} = 1} \| P'_r{}^{i,e} - Q_r^{i,e} \|_2 \quad \text{(Eq. 5)}
$$

其中 $\boldsymbol{P}'_r$ 是 Refinement Router 的输出 $\boldsymbol{\rho}_r$ 经过一个 MLP 扩展维度后的结果（从 $E_r$ 扩展到 $E_{\mathrm{attn}}$）。这个 MLP 不仅做维度匹配，还起到了缓解直接 alignment 带来的 training conflict 的作用。

**最终总 Loss**：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{diffusion}} + \lambda_1 \mathcal{L}_{\mathrm{coarse}} + \lambda_2 \mathcal{L}_{\mathrm{fine-align}} + \lambda_3 \mathcal{L}_{\mathrm{fine-balance}} \quad \text{(Eq. 6)}
$$

其中 $\mathcal{L}_{\mathrm{fine-balance}}$ 是标准的 MoE load-balancing loss，防止 token-level routing 崩塌。$\lambda_1 = 0.1, \lambda_2 = 0.02, \lambda_3 = 0.01$。

### 4. 实验数据与架构图解析

#### 4.1 定量结果分析

在 VideoPhy2 benchmark 上的结果如下表所示：

| Method | ALL (PC) | ALL (SA) | ALL (Joint) | HARD (PC) | HARD (SA) | HARD (Joint) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Hunyuan Video | 64.2 | 19.2 | 24.7 | 52.2 | 7.2 | 5.0 |
| Wan2.1-1.3B | 57.8 | 30.0 | 24.8 | 45.6 | 36.7 | 8.3 |
| **Wan2.1-1.3B + ProPhy** | **65.0** | **32.0** | **26.5** | **48.9** | **12.2** | **7.2** |
| CogVideoX-5B | 67.2 | 29.0 | 22.3 | 51.1 | 9.6 | 5.0 |
| CogVideoX-5B + WISA | 69.1 | 31.5 | 25.8 | 51.7 | 11.1 | 5.0 |
| CogVideoX-5B + VideoREPA | 72.5 | 24.2 | 22.0 | 52.2 | 7.8 | 5.6 |
| **CogVideoX-5B + ProPhy** | **72.5** | **32.2** | **26.7** | **52.8** | **11.7** | **6.1** |

*注：PC = Physical Commonsense, SA = Semantic Adherence, Joint = 两者同时满足的比例。*

**Intuition**: 
1. **Joint Metric 的飞跃**: ProPhy 在 Wan2.1 上带来了 **+19.7%** 的 Joint 提升率。这表明 ProPhy 在不破坏语义 adherence 的前提下，极大地提升了物理常识。
2. **Hard Subset 的表现**: 在更复杂的 HARD 子集中，ProPhy 的 SA (Semantic Adherence) 提升尤为显著（CogVideoX 从 9.6 提升到 11.7）。这说明 Progressive Alignment 在多物理现象共存的高难度场景下，能够更准确地捕捉局部物理过程，避免全局物理误导造成的 semantic drift。

#### 4.2 Ablation Study 解析

Paper 中的 Table 3 和 Table 4 提供了极有价值的 ablation：

1. **PB vs LoRA**: 仅使用 LoRA 微调 backbone，Joint score 只能达到 24.8；引入 Physical Branch (PB) 后，即使没有 SEB 和 REB，Joint 也能达到 25.7。这说明引入额外的物理处理通路比单纯增加参数量更有效。
2. **SEB 与 REB 的互补性**: 
   - 只有 SEB: Joint = 26.0
   - 只有 REB: Joint = 26.2
   - SEB + REB: Joint = 26.5
   这验证了 Progressive 设计的必要性。SEB 提供全局物理 context，REB 在此基础上进行局部细化，两者是互补关系。
3. **Loss 设计**: 如果在 SEB 中使用 BCE loss 替代 Relative Distance loss，PC 指标会下降。如果在 REB 中只使用 align loss 而没有 balance loss，Joint 指标会暴跌至 21.6，这说明 token-level 的 MoE 极易发生 collapse。

### 5. 深度联想与 Intuition Building

1. **VLM Attention 作为 Physical Ground Truth 的合理性**:
   Paper 中的一个关键发现是：VDM 的 cross-attention map 在加入噪声后去噪，无法准确 focus 到物理现象发生的位置；而 VLM 能够精准定位。这暗示了当前 Diffusion Model 的 latent space 缺乏显式的物理 binding。VLM 通过大规模 image-text pair 的 contrastive learning，获得了强大的 grounding 能力。ProPhy 实际上是在做一种 **Cross-modal Spatial Knowledge Distillation**，把 VLM 的 grounding 能力迁移到 VDM 的 denoising trajectory 中。

2. **Expert Inversion 与 Disentanglement**:
   Figure 8 中的 Expert Inversion 实验非常巧妙。如果在推理时把 Refinement Router 的 logits 反转，原本刚性的车门会变得像布料一样飘动。这不仅证明了不同的 expert 确实学到了 distinct physical priors，还暗示了 ProPhy 具备了 **Controllable Physical Attribute Manipulation** 的潜力。这类似于在 World Simulator 中找到了物理属性的 "Knobs"。

3. **与 Physics Differential Equations 的结合**:
   Paper 在 Limitations 中提到，目前的 physical categorization 只是限制了 expert 的参数空间，并没有 enforce 显式的物理方程。这引出了一个极具潜力的研究方向：能否将 Neural ODEs 或 PINNs (Physics-Informed Neural Networks) 嵌入到 REB 的 expert 中？例如，让 fluid expert 在内部求解简化的 Navier-Stokes 方程，让 rigid body expert 求解 Hamiltonian 方程。这将是从 "Pattern Fitting" 走向 "Principled Simulation" 的关键一步。

### 6. Web Links for Reference

为了方便你进一步查阅，这里提供相关核心资源的链接：

1. **ProPhy Project Page**: [https://zijunwa.github.io/prophy/](https://zijunwa.github.io/prophy/)
2. **Wan2.1 (Base Model)**: [https://github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
3. **CogVideoX (Base Model)**: [https://github.com/THUDM/CogVideo](https://github.com/THUDM/CogVideo)
4. **Qwen2.5-VL (Used for VLM Distillation)**: [https://github.com/QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
5. **VideoPhy-2 Benchmark**: [https://arxiv.org/abs/2503.06800](https://arxiv.org/abs/2503.06800)
6. **WISA-80K Dataset**: [https://arxiv.org/abs/2410.06159](https://arxiv.org/abs/2410.06159) (NeurIPS 2025 paper)
7. **VideoREPA**: [https://arxiv.org/abs/2505.23445](https://arxiv.org/abs/2505.23445) (NeurIPS 2025 paper)

总结来说，ProPhy 通过两阶段 MoPE 和 VLM Attention Distillation，在 Video Generation 的 latent space 中建立了一种显式的、分层的物理先验注入机制。它巧妙地利用了 VLM 的 spatial grounding 优势来弥补 VDM 的物理盲区，为构建更加符合物理规律的 World Simulator 提供了一个极具启发性的 framework。
