---
source_pdf: Reconstruction or Semantics What Makes a.pdf
paper_sha256: ae409282f58b6d5c6f01b9f317abde8e040ad318d5bbd30bb41951da372ffa3b
processed_at: '2026-08-11T21:53:15-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

好，Andrej，我换一种方式讲。把它想象成我们在Mila的咖啡厅聊这个paper。

---

## 这paper到底在问什么

你看现在robotics world model的圈子，大家都在做**action-conditioned video generation**——给机器人一段observation和action sequence，让它预测future frames。主流做法是latent diffusion model (LDM)，就是先把image encode到一个latent space，在这个latent space里跑diffusion生成future latents，再decode回pixels。

问题来了：**这个latent space用什么？**

大部分人想都没想就用VAE——因为Stable Diffusion就用VAE，VAE是train来reconstruct pixel的，decoder能完美还原image，看起来很自然。VAE family包括SD3 VAE, VA-VAE, Cosmos这些。

但这群人（Chandar Lab）就觉得：等一下，**robot world model的objective和image generation的objective根本不一样啊**。

Image generation：我要生成漂亮的图，pixel fidelity最重要，VAE完美。
Robot world model：我要预测action会怎么改变世界，用来做planning、policy evaluation。pixel好看没用，如果latent里action info是smear的，policy学到错误的因果。

所以他们就问：**到底latent space应该optimize什么？是pixel reconstruction还是semantic/action structure？**

这个question在circle里被很多人avoid，因为大家默认VAE就是好的，没人systematically地ablate过。

---

## 他们怎么做的

非常clean的experimental design。**Fix所有其他变量，只vary latent space。**

具体来说：
- 同一个dataset (BridgeV2, 60K WidowX 250 robot demos)
- 同一个transition model (DiT, flow matching)
- 同一个action conditioning scheme
- 同一个training schedule (AdamW, lr=1e-4, 100 epochs)
- 同一个token count (256 tokens/frame, 通过控制patch size)

只换encoder + decoder pair。他们比了两类：

**Reconstruction family** (train来reconstruct pixel的)：
- SD3 VAE (D=16)
- VA-VAE (D=32)
- Cosmos (D=16)

**Semantic family** (train来encode semantic/structure的)：
- V-JEPA 2.1 (D=1024)
- Web-DINO (adapted from DINOv2, D=1024)
- SigLIP 2 (D=1152)

Semantic encoder的feature dimension都很高（1024+），diffusion直接train很难，所以他们用了两个trick：

1. **S-VAE adapter**：把1024维压缩到96维，让diffusion friendly。Adapter是个小transformer encoder-decoder，用KL regularization让latent分布well-behaved。

2. **Shallow-wide DDT head**：如果不用adapter直接用high-dim latent，在DiT output加个浅但宽的readout layer（2048维），解决width bottleneck。

还有个key trick叫**dimension-dependent noise schedule shift**：

$$\gamma = \sqrt{\frac{(256/p^2) d}{4096}}, \qquad \tau' = \frac{\gamma \tau}{1 + (\gamma - 1)\tau}$$

变量解释：
- $d$: DiT input channel数
- $p$: patch size
- $\tau$: 原始timestep $\in [0,1]$
- $\tau'$: shifted timestep
- $\gamma$: shift factor

intuition是：高维latent里相同noise level的SNR表现不同，需要shift schedule让中间状态更"clean"，模型更容易学。这是从SD3和RAE (Zheng et al.)借鉴的。

---

## 三轴评估——这是paper最contribution的地方

他们提出**visual fidelity不足以选world model**，设计了三个axes：

### Axis 1: Planning & Policy Performance

**CEM action recovery**：给定真实context latents和target future latents，用Cross-Entropy Method反推action，看误差多大。

公式：

$$a^{\text{plan}} = \arg\min_a \frac{1}{k}\sum_{j=1}^k \|p_\theta^{(j)}(\tilde{z}_t, \tilde{z}_{t+1}, a) - \tilde{z}^*_{t+1+j}\|_2^2$$

变量：
- $a$: 要优化的k-step action sequence
- $p_\theta^{(j)}$: 第j次autoregressive latent prediction
- $\tilde{z}^*$: 真实target latents
- $k \in \{1, 4\}$: horizon

CEM设置：400 candidates, 5 iterations, 50 elites。

**VLA-in-the-loop**：在world model里rollout OpenVLA-7B policy，50 steps × 20 tasks × 8 trials，用InternVL 3.5 + Qwen 3.6两个VLM judge打分，consensus成功率高说明world model能当policy evaluation environment。

**OOD robustness**：两种perturbation：
- Distractor object：用ChatGPT Images 2.0在initial frame里加OOD object
- OOD instruction：把"close oven"换成"open the oven"这种semantic-related但behaviorally不同的指令

### Axis 2: Pixel Fidelity & Scene Geometry

PSNR, SSIM, LPIPS, FID, FVD, t-LPIPS, PCK coverage（用CoTracker跟踪point看geometric stability），还有WorldArena的perceptual metrics。

Key trick：算每个encoder的"reconstruction ceiling"——只encode→decode，不过DiT。gap = WM_score - ceiling，gap小说明transition model没拖累decoder。

### Axis 3: Latent Representation Quality

**IDM (Inverse Dynamics Model)**：在frozen encoder latents上train一个transformer，从 $(z_t, z_{t+k})$ 预测action chunk。然后把同一个frozen IDM apply到world model生成的 $(\hat{z}_t, \hat{z}_{t+k})$，看Pearson r下降多少。

- Real r: encoder ceiling
- WM r: world model生成后的r

**Deep intuition**：这是paper里我最喜欢的metric。它直接probes latent space里action info有没有被保留。Pixel metrics看不见这种failure——VAE可能生成漂亮frame但latent里action info已经丢了。

**Success Classifier Probe**：在SOAR数据集（30.5K success/failure labeled episodes）上train分类器，看world model生成rollout后分类准确率drop多少。

---

## 发现了什么

### Finding 1: Semantic latents在policy上crush reconstruction latents

Table 1 DiT-S的结果：

| Encoder | VLA Consensus SR | CEM k=1 | CEM k=4 |
|---|---|---|---|
| VAE | 0.169 | 0.111 | 0.612 |
| Cosmos | 0.244 | 0.112 | 0.661 |
| V-JEPA 2.1 | **0.344** | **0.084** | **0.424** |
| SigLIP 2 | 0.325 | 0.082 | 0.523 |

Semantic family在VLA SR上比reconstruction高约10个点，统计显著（p=0.013）。CEM error差距更大（k=4: 0.42 vs 0.61）。

### Finding 2: IDM上semantic encoders惊人地好

Table 2:

| Encoder | Real k=1 | Real k=4 | WM k=1 | WM k=4 |
|---|---|---|---|---|
| VAE | 0.507 | 0.478 | 0.476 | 0.464 |
| V-JEPA 2.1 | 0.829 | 0.865 | 0.781 | 0.840 |
| Web-DINO | 0.820 | 0.845 | 0.729 | 0.794 |
| SigLIP 2 | 0.772 | 0.793 | 0.697 | 0.757 |

V-JEPA 2.1在real latent上的IDM Pearson r达到0.83-0.87，VAE只有0.48-0.51。

这说明什么？**Semantic encoder从一开始就把action info encode得更explicit**。VAE latent里action info是second-order的（要从pixel difference推断），semantic latent里是first-order的。

World model generation之后，semantic的gap也很小（0.04-0.08），说明DiT没破坏action geometry。

### Finding 3: Visual fidelity上scaling后VAE会反超

DiT-S时semantic在SSIM/LPIPS/FVD都赢。

DiT-L时：
- VAE FID: 5.35 (最好)
- VAE image quality: 0.598 (最好)
- 但V-JEPA 2.1₉₆ SSIM: 0.743 (仍最好)
- V-JEPA 2.1₉₆ LPIPS: 0.165 (仍最好)

**Scaling能recover reconstruction的visual优势，但semantic的policy advantage不会消失**。这个发现很重要——它说visual fidelity和action modeling是decoupled的capacity需求。

### Finding 4: Scaling narrows policy gap但not action-centric gap

DiT-L时：
- VAE VLA SR: 0.169 → 0.350（gap缩小）
- VAE CEM k=1 error: 0.111 → 0.120（没改善，反而略差）
- Semantic CEM k=1: 0.084-0.093（仍领先）

**Why?** CEM depends directly on latent transition structure，policy performance partly depends on rendered visual quality（policy看pixels）。Scaling改善visual → policy改善；latent的action geometry没改善 → CEM不变。

### Finding 5: Failure modes不同

- **Reconstruction (VAE/Cosmos)**: hallucinate task semantics。生成plausible but task-incorrect frames。VAE凭空"生成"一个white basket，Cosmos生成green towel。OOD instruction下保持prior action pattern，不理解new goal。
- **Semantic (V-JEPA)**: preserve task-level intent但miss geometric precision。比如under-open the drawer（drawer开得不够大）。

直觉理解：Reconstruction latent把budget花在pixel fidelity上，semantic结构是emergent property，容易被DiT prior覆盖。Semantic latent从一开始就把"task progress"做explicit，DiT很难破坏，代价是fine geometry。

### Finding 6: Adapter是双刃剑

$d=96$的S-VAE adapter对Web-DINO是sweet spot（Table 16）：

| Metric | $d_{16}$ | $d_{96}$ | $D_{1024}$ |
|---|---|---|---|
| VLA SR | 0.256 | 0.269 | 0.181 |
| FID | 8.37 | 6.00 | 7.63 |
| FVD | 7.65 | 5.51 | 6.66 |

Non-monotonic sweet spot。太压缩丢semantic info，不压缩diffusion难train。

但adapter会distort control geometry：
- V-JEPA 2.1₉₆ VLA SR更好（0.362 vs native 0.344）
- V-JEPA 2.1₉₆ CEM k=4略差（0.548 vs 0.424）

**Intuition**：adapter把"diffusion-friendly"和"control-friendly"分离了。做trajectory optimization（CEM）用native high-dim，做VLA visual evaluation用adapter。

### Finding 7: Multi-view finetune时semantic更robust

20 epochs finetune on 3-camera-view episodes：
- 所有encoder CEM都改善（VAE: 0.111→0.047, V-JEPA: 0.084→0.056）
- 但FVD变差（VAE: 6.8→12.9）——multi-view episodes少
- Semantic encoders更robust到这种degradation

---

## 为什么semantic latents这么好？我的hypothesis

VAE training objective是pixel reconstruction。Latent只需要encode **pixel-sufficient statistics**。Action是从pixel-difference推断的——这是second-order information。在VAE latent中action info被"smeared"在spatial locations里。

Semantic encoders (JEPA/DINO/SigLIP) training时就是prediction / contrastive。它们直接encode **object-centric, transformation-aware features**。Action effect本质上就是object transformation，所以semantic latent和action是first-order correlated。

这里有个deep point：**Latent space是objective function的fossilized form**。你train encoder时optimize什么，latent就encode什么。VAE optimize pixel → latent是pixel的压缩。Semantic encoder optimize prediction/contrastive → latent是structure的压缩。Robot world model需要的是后者。

OOD instruction = "fold" → "unfold"。Pixel level上fold/unfold的cloth appearance很接近。Semantic encoder在latent里把"folded state"和"unfolded state"作为distinct semantic clusters，DiT能正确switch。VAE latent里两者pixel-similar → 容易collapse到prior pattern。

---

## 对未来的implications

1. **Cross-embodiment evaluation**：只在BridgeV2做，应该测Franka, ALOHA, human hand。不同action space, gripper morphology, control frequency会带来新挑战。
2. **Partially finetune encoder**：现在encoder完全frozen，如果允许轻度finetune让encoder更action-aware？
3. **Video foundation models**：试video MAE, video CLIP的latent。
4. **Hierarchical latent**：semantic coarse + reconstruction fine的两层结构。
5. **Scaling story**：如果再scale下去（DiT-XL或更大），semantic的policy advantage会不会被visual advantage swallow？DiT-L时已经narrow了。

---

## Recipe（paper Section 5）

他们给的实际recipe：

1. 不要optimize visual realism alone——先选对action/task structure explicit的latent space
2. Default to semantic encoders (V-JEPA 2.1, Web-DINO, SigLIP 2)
3. Pair with adapter when decoded rollout quality or VLA-in-loop matters
4. Transition model: spatial-temporal DiT with causal temporal blocks + shallow-wide DDT head + dimension-aware noise shift
5. Spatial blocks stay non-causal（per-frame joint denoising）
6. Diffusion forcing for autoregressive rollout
7. Multi-axis evaluation: visual + latent + downstream

---

## 我的整体评价

这篇paper很罕见。大部分paper propose新method然后show SOTA。这篇做**controlled evaluation study**——fix所有变量，isolate一个design choice，show这个design choice对downstream robotics的影响远大于人们以为的。

它的核心contribution是**reframing the question**：从"what reconstructs best"到"what preserves action-relevant structure"。

对你（Karpathy）来说这可能特别resonant——你一直强调objective function定义了模型学什么。这里VAE latent的objective是pixel reconstruction，所以它encode pixel-sufficient info；semantic encoder的objective是prediction/contrastive，所以它encode structure info。World model需要的是后者。

Latent space是objective function的fossilized form。这句话我觉得可以概括整篇paper的insight。

潜在limitation：
1. 只在BridgeV2上，没做cross-embodiment
2. VLM judge可能有bias（虽然2个VLM consensus + non-VLM metrics缓解）
3. 没比non-diffusion semantic world models（DINO-WM等）——但这是scope choice
4. DiT-L时VAE在visual反超，再scale下去会怎样是open question

值得follow的工作：
- Project page: https://hskalin.github.io/semantic-wm/
- HuggingFace: https://huggingface.co/Nilaksh404/semantic-wm
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2/
- DINO-WM: https://dinowm.github.io/
- RAE: https://arxiv.org/abs/2510.11690
- S-VAE: https://arxiv.org/abs/2512.17909
- Bridge V2: https://proceedings.mlr.press/v229/walke23a.html
- OpenVLA: https://openvla.github.io/
- Flow matching: https://openreview.net/forum?id=PqvMRDCJT9t

如果让我predict next steps，我会想看：
- 把这个study扩展到cross-embodiment
- Hierarchical latent (semantic + fine)
- Action-aware encoder finetuning
- 用video foundation models的latent
- Policy distillation into latent space

你觉得这个paper的intuition符合你在world model上的经验吗？我个人觉得"latent space是objective function的fossilized form"这个point其实可以推广到很多其他domain——比如language model的representation也是预训练objective的fossilized form。RLM (RL-finetuned language model)的latent和base model的latent应该差别很大，类似地semantic encoder和reconstruction encoder的latent差别很大。

---

# Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models — 深度技术讲解

Andrej，这篇paper非常有意思，它问了一个看似简单但被很多人回避的问题：**当我们在LDM world model里选择latent space时，到底应该optimize什么？** 答案并非"reconstruction fidelity"，也并非"semantic abstraction"——而是要看**latent space对action-conditioned dynamics的保真度**以及对policy evaluation的支撑能力。下面我从问题定义、方法、实验、intuition四个层面展开。

---

## 1. 问题定义：为什么这个question non-trivial

机器人世界模型的input-output接口：

$$p(o_{t+1:t+K} \mid o_{t-H:t},\, a_{t-H:t+K-1})$$

- $o_t \in \mathcal{O}$: RGB observation
- $a_t \in \mathbb{R}^{d_a}$: 7-DoF end-effector action (position, rotation, gripper state)
- $H$: history length (paper用 $H=2$)
- $K$: rollout horizon (paper预测 8 future frames, frame skip 2)

由于partial observability，单帧observation不能决定下一帧，所以需要condition on finite history。这里把不同tasks看作**shared dynamics, task-dependent goals**的related POMDPs——这点很关键，因为它要求latent space同时支持shared motor dynamics 和 task-specific goal discrimination。

### LDM的三段式架构

$$z_t = f_\phi(o_t) \in \mathbb{R}^{N \times D},\quad N = h \times w \text{ (patches)},\ D = \text{native channel dim}$$

$$\tilde{z}_t = \alpha_\psi(z_t) \in \mathbb{R}^{N \times d},\quad \text{(optional adapter, } d=96\text{)}$$

$$\tilde{z}_{t+1:t+K} \sim p_\theta(\cdot \mid \tilde{z}_{t-H:t}, a_{t-H:t+K-1})$$

$$\hat{o}_{t+1:t+K} = \text{Dec}(\tilde{z}_{t+1:t+K})$$

核心trick是：**encoder $f_\phi$、adapter $\alpha_\psi$、decoder 都冻结，只train transition model $p_\theta$ (DiT)**。这样就把latent space作为唯一的实验变量隔离出来。这种controlled evaluation的范式我觉得非常Karpathy-style——把混杂变量固定下来再claim归因。

### 关键洞察：为什么visual fidelity ≠ useful world model

设想两个extreme case：
1. 一个**完美重建像素**的VAE latent，但latent space中object位置和action effect几乎uncorrelated
2. 一个**highly semantic**的DINO/JEPA latent，丢弃某些pixel details但保留了object-action affordance geometry

case 1的world model可以生成漂亮的frames但policy学到错误的causal structure；case 2的pixel略糊但action propagation在latent里是well-posed的。这篇paper就是在empirically分离这两个维度。

---

## 2. 方法：让high-dim semantic space在LDM里tractable

### 2.1 Compute parity trick

非常聪明的设计：**所有encoders都patchify到相同的token count**

$$N = (h/p)(w/p) = 16 \cdot 16 = 256$$

- VAE latent $32 \times 32 \times 16$ with patch $p=2$ → 256 tokens
- Semantic latents $16 \times 16 \times D$ with patch $p=1$ → 256 tokens

这样DiT backbone的compute完全相同，channel dimension $D$ 只影响input/output projection。Table 4里DiT GFLOPs基本一致(~316-318)，差异主要来自frozen encoder/decoder。

### 2.2 Flow matching objective

使用optimal-transport flow matching（Lipman et al., 2023）：

$$\tilde{z}_{\tau_i, i} = (1 - \tau_i)\tilde{z}_i + \tau_i \epsilon_i$$

- $\tau_i \sim p(\tau)$: 连续时间变量 $\in [0, 1]$
- $\epsilon_i \sim \mathcal{N}(0, I)$: noise sample
- $\tilde{z}_i$: clean latent (data)

预测的velocity field $v_\theta$ 学到的是：

$$u_i = \epsilon_i - \tilde{z}_i \quad \text{(target velocity, 从data指向noise)}$$

训练loss：

$$\mathcal{L}_{FM} = \mathbb{E}_{\tilde{z}, \epsilon, \tau}\left[\sum_{i=H}^{T-1}\left\|v_\theta(\tilde{z}_{\tau,i}, \tau_i, a_{0:T-1}, \ell) - (\epsilon_i - \tilde{z}_i)\right\|_2^2\right]$$

只对future frames算loss，history frame作为clean conditioning context但加少量Gaussian augmentation防止过拟合到perfect context：

$$\tilde{z}_{\text{aug}}^{\text{ctx}} = \frac{\tilde{z}^{\text{ctx}} + \sigma_h \eta}{\sqrt{1 + \sigma_h^2}}$$

### 2.3 Dimension-dependent noise schedule shift

这是从SD3和RAE借鉴的关键trick。高维latent的noise level感知需要shifted：

$$\gamma = \sqrt{\frac{(256/p^2) d}{4096}}, \qquad \tau' = \frac{\gamma \tau}{1 + (\gamma - 1)\tau}$$

- $d$: DiT input channel count (after adapter)
- $256/p^2$: tokens per frame
- 4096: normalization constant (经验值)

**Intuition**：高维latent中，相同noise level下signal-to-noise ratio的表现不同。$\gamma > 1$时，schedule在中段被"拉宽"——让模型有更多"clean intermediate states"学习机会。这对1024维的V-JEPA/DINO/SigLIP latents至关重要。

### 2.4 S-VAE Adapter

为了使semantic features diffusion-friendly，用S-VAE (Zhang et al., 2025)压缩到 $d=96$：

$$h = g_\psi^{\text{enc}}(z)$$
$$(\mu, \log\sigma^2) = W_{\mu,\sigma^2} h$$
$$\tilde{z} = \mu + \sigma \odot \xi,\quad \xi \sim \mathcal{N}(0, I)$$
$$\hat{z} = g_\psi^{\text{dec}}(\tilde{z})$$

Adapter loss有4项：

$$\mathcal{L}_{\text{adapter}} = \underbrace{\mathcal{L}_{\text{MSE}}(z, \hat{z}) + \lambda_{\text{cos}}\mathcal{L}_{\text{cos}}(z, \hat{z}) + \lambda_{\text{spec}}\mathcal{L}_{\text{FFT}}(z, \hat{z})}_{\text{semantic reconstruction}} + \lambda_{\text{KL}} D_{\text{KL}}(q_\psi(\tilde{z}\mid z) \| \mathcal{N}(0, I)) + \lambda_{\text{pix}} \mathcal{L}_{\text{pix}}(o, \hat{o})$$

各loss的含义：
- $\mathcal{L}_{\text{MSE}}$: magnitude error
- $\mathcal{L}_{\text{cos}} = 1 - \cos(z, \hat{z})$: 方向/语义结构保持
- $\mathcal{L}_{\text{FFT}}$: 沿spatial-token axis做1D FFT，penalize high-frequency structure loss
- $D_{\text{KL}}$: posterior $\mathcal{N}(\mu, \sigma^2 I)$ toward prior $\mathcal{N}(0, I)$ — 让latent在generation时分布well-behaved
- $\mathcal{L}_{\text{pix}}$: MSE + LPIPS + (1-MS-SSIM) on decoded pixels

### 2.5 Shallow-wide DDT head

对native high-dim semantic latents（不用adapter）的情况，用DDT (Wang et al., 2025)的shallow-wide head。DiT backbone不动，只在output加一个2048-dim readout width的浅层spatial refinement layer。这是为了解决"width bottleneck"——high-dim latent的最后一个prediction layer太薄。

---

## 3. 三轴评估protocol

这是这篇paper最contribution的地方。它提出**visual fidelity不足以选择world model**，并设计了3 axes：

### Axis 1: Planning & Policy Performance

**A. CEM action controllability**：给定真实context latents和target future latents，能否recover ground-truth action？

$$a_{t+1:t+k}^{\text{plan}} = \arg\min_{a_{t+1:t+k}} \frac{1}{k} \sum_{j=1}^k \left\| p_\theta^{(j)}(\tilde{z}_t, \tilde{z}_{t+1}, a_{t+1:t+k}) - \tilde{z}_{t+1+j}^* \right\|_2^2$$

- $p_\theta^{(j)}$: j-th autoregressive latent prediction
- $k \in \{1, 4\}$: horizon
- CEM: 400 candidates, 5 iterations, 50 elites (fraction 0.125)
- 初始化mean = $a^*$, std = 1/4 action range
- 报告 $\frac{1}{k}\sum_j \|a^{\text{plan}} - a^*\|_2$

**Intuition**：CEM error低，意味着latent dynamics对action changes是敏感的，且这种敏感性能被black-box optimizer利用。VAE的latent往往action geometry不explicit，导致CEM反推action困难。

**B. VLA-in-the-loop**：在world model里rollout OpenVLA-7B，50-step episodes × 20 tasks × 8 trials。用InternVL 3.5 + Qwen 3.6两个VLM judge做consensus scoring：

$$\text{CSR} = \frac{1}{N}\sum_i \mathbb{1}[\text{score}_i^{\text{InternVL}} \geq 0.5 \wedge \text{score}_i^{\text{QwenVL}} \geq 0.5]$$

**Borda rank**：$r_{\text{InternVL}} + r_{\text{QwenVL}}$，对judge calibration drift更鲁棒。

**C. OOD robustness**：在10个tasks上做两种perturbation：
- Distractor object: ChatGPT Images 2.0生成OOD object插入
- OOD instruction: 用Claude Opus 4.7生成语义相关但behavior不同的instruction（如 "close oven" → "open the oven"）

### Axis 2: Pixel Fidelity & Scene Geometry

包含reference-based (PSNR/SSIM/LPIPS/FID/FVD/t-LPIPS/PCK)和reference-free (MUSIQ image quality, LAION aesthetic, JEPA similarity, dynamic degree, flow score, depth AbsRel)。**Key**：对每个encoder都算"reconstruction ceiling"（encode→decode without DiT），gap = WM_score - ceiling，gap小表示transition model没拖累。

### Axis 3: Latent Representation Quality

**A. IDM (Inverse Dynamics Model)**：

在frozen encoder latents上训练一个patch-token Transformer，从 $(z_t, z_{t+k})$ 预测 action chunk $\hat{a}_{t:t+k-1} \in \mathbb{R}^{k \times d_a}$。然后**把同一个frozen IDM head应用到world model生成的latent pair $(\hat{z}_t, \hat{z}_{t+k})$**，看Pearson r的下降：

- Real r: encoder ceiling
- WM r: world model generation后的r

**Deep intuition**：如果encoder latent一开始就不encode action info，IDM在Real上就低；如果encoder好但DiT生成时破坏了action-discriminative geometry，WM r会大幅下降。这两种failure mode对pixel metrics都不可见！

**B. Success Classifier Probe**：

在SOAR数据集（30.5K success/failure labeled episodes）上训练spatio-temporal Transformer，分类 trajectory success。架构很有意思：spatial self-attn → temporal self-attn → cross-attn to frozen SigLIP 2 text tokens，最后mean pool over patches → linear head。**Instructions-mismatch negatives**强制它真正grounding success on video content而非ignore instruction。

评估时把frozen probe apply到WM生成的rollout，看drop in balanced accuracy。

---

## 4. 核心实验结果

### 4.1 VLA Policy Performance（Table 1, DiT-S）

| Encoder | Consensus SR | Borda ↓ | OOD SR (distractor) | CEM k=1 | CEM k=4 |
|---|---|---|---|---|---|
| VAE | 0.169 | 25 | 0.287 | 0.111 | 0.612 |
| VA-VAE | 0.175 | 23 | 0.250 | 0.097 | 0.543 |
| Cosmos | 0.244 | 16 | 0.362 | 0.112 | 0.661 |
| **V-JEPA 2.1** | **0.344** | 6 | 0.575 | 0.084 | 0.424 |
| V-JEPA 2.1₉₆ | 0.362 | 8 | 0.537 | 0.089 | 0.548 |
| Web-DINO | 0.212 | 21 | 0.512 | 0.090 | 0.474 |
| Web-DINO₉₆ | 0.300 | 11 | 0.512 | 0.090 | 0.531 |
| SigLIP 2 | 0.325 | 9 | 0.500 | 0.082 | 0.523 |
| SigLIP 2₉₆ | 0.331 | 15 | 0.588 | 0.086 | 0.537 |

观察：
- Reconstruction family (VAE/VA-VAE/Cosmos) SR在0.17-0.24
- Semantic family (V-JEPA/Web-DINO/SigLIP) SR在0.21-0.36
- CEM k=4 error：semantic (0.42-0.55) << reconstruction (0.54-0.66)

Family-level paired bootstrap (Table 12)：
- VLA SR: +9.8 pp, 95% CI [2.5, 17.7], p=0.013
- OOD SR: +13.6 pp, [8.8, 18.4], p < 5e-5
- CEM error: -0.027, [-0.041, -0.012], p=0.00015

这是统计显著的。

### 4.2 IDM Action Recoverability（Table 2, DiT-S）

| Encoder | Real k=1 | Real k=4 | WM k=1 | WM k=4 | Classifier Enc | Classifier WM |
|---|---|---|---|---|---|---|
| VAE | 0.507 | 0.478 | 0.476 | 0.464 | 0.835 | 0.716 |
| VA-VAE | 0.549 | 0.744 | 0.545 | 0.719 | 0.868 | 0.744 |
| Cosmos | 0.626 | 0.673 | 0.581 | 0.651 | 0.851 | 0.723 |
| V-JEPA 2.1 | 0.829 | 0.865 | 0.781 | 0.840 | 0.905 | 0.789 |
| Web-DINO | 0.820 | 0.845 | 0.729 | 0.794 | 0.906 | 0.788 |
| SigLIP 2 | 0.772 | 0.793 | 0.697 | 0.757 | 0.903 | 0.823 |

**惊人发现**：
- V-JEPA 2.1在Real latent上的IDM Pearson r高达0.83-0.87（vs VAE的0.48-0.51）——semantic encoders把action information编码得更explicit！
- WM后gap很小（0.04-0.08），说明DiT生成时保留了action geometry
- SigLIP 2的WM Classifier Acc最高 (0.823)——它在task-level outcome上semantic drift最小

### 4.3 Visual Fidelity（Table 3, DiT-S和DiT-L）

DiT-S：
- Semantic在SSIM, LPIPS, FVD, t-LPIPS上dominant
- VAE在FID, image quality上competitive但不是最好

DiT-L（scaling后）：
- VAE反超！FID 5.35 (vs semantic 6-7), image quality 0.598
- 但Semantic在SSIM (V-JEPA 2.1₉₆: 0.743)和LPIPS (V-JEPA 2.1₉₆: 0.165)仍最好

**关键insight**：scaling能recover大部分reconstruction优势，但semantic的policy advantage不会消失。

### 4.4 Failure modes（Section 4.5）

这是paper最qualitative也最insightful的部分。两者fail mode不同：

- **Reconstruction (VAE/Cosmos)**: hallucinate task semantics — 生成plausible but task-incorrect frames（如VAE凭空"生成"一个white basket，Cosmos生成green towel）。在OOD instruction下保持prior action pattern，不理解new goal。
- **Semantic (V-JEPA)**: preserve task-level intent但miss geometric precision（如under-open the drawer）。

**Why?** Reconstruction latent把budget花在pixel fidelity上，semantic结构在latent中是emergent property，容易被DiT的prior覆盖。Semantic latent从一开始就把"task progress"做explicit，DiT很难破坏它，但代价是fine geometry。

---

## 5. Multi-view和Adapter的trade-off

### Multi-view finetuning（Fig 5左）

20 epochs finetune on 3-camera-view episodes：
- 所有encoder的CEM action prediction都改善（VAE: 0.111→0.047, V-JEPA: 0.084→0.056）
- 但FVD变差（VAE: 6.8→12.9）——fewer multi-view episodes的数据不足
- **Semantic encoders更robust**到这种degradation

### Adapter dim ablation（Table 16, Web-DINO DiT-S）

| Metric | $d_{16}$ | $d_{96}$ | $D_{1024}$ |
|---|---|---|---|
| VLA SR | 0.256 | 0.269 | 0.181 |
| SSIM | 0.711 | 0.728 | 0.722 |
| LPIPS | 0.196 | 0.181 | 0.199 |
| FID | 8.37 | 6.00 | 7.63 |
| FVD | 7.65 | 5.51 | 6.66 |

**Non-monotonic sweet spot at $d=96$**！太压缩丢semantic info，不压缩diffusion难train。

### Adapter双刃剑（Section 4.6）

Adapters improve diffusion ease但distort control geometry：
- VLA SR更好（V-JEPA 2.1₉₆: 0.362 vs native: 0.344）
- 但CEM action error略差（k=4: 0.548 vs 0.424）—因为压缩丢失了fine-grained action geometry
- OOD robustness也略差

**Intuition**：adapter把"diffusion-friendly"和"control-friendly"分离了。如果你要做trajectory optimization（CEM），用native high-dim；如果你要VLA-in-the-loop visual evaluation，用adapter。

---

## 6. 跨DiT scaling的behavior（Table 10）

DiT-L：
- VAE的VLA SR从0.169→0.350（scaling收益最大）
- 但CEM k=1 error仍高于所有semantic (0.120 vs semantic 0.087-0.093)
- **Policy gap narrows但action-centric gap不narrow**

这是为什么？CEM depends directly on latent transition structure，而policy performance partly depends on rendered visual quality（policy看pixels）。Scaling改善了visual → policy改善；但latent的action geometry没改善 → CEM不变。

---

## 7. 对Open Questions的思考

### 7.1 为什么semantic latents的IDM r比VAE高这么多？

我的hypothesis：VAE training objective是pixel reconstruction，它的latent只encode**pixel-sufficient statistics**。而action是从pixel-difference推断的——这是second-order information。在VAE latent中action info被"smeared"在spatial locations里。

Semantic encoders (JEPA/DINO/SigLIP) training时就是prediction / contrastive——它们直接encode **object-centric, transformation-aware features**。Action effect本质上就是object transformation，所以semantic latent和action是first-order correlated。

### 7.2 为什么semantic latents在OOD instruction下更好？

OOD instruction = "fold" → "unfold"。这个差异是semantic-level的，pixel level上fold/unfold的cloth appearance很接近。Semantic encoder在latent里把"folded state"和"unfolded state"作为distinct semantic clusters，DiT能正确switch；VAE latent里两者pixel-similar → 容易collapse到prior pattern。

### 7.3 Scaling implications

这篇paper有个**反直觉的scaling story**：
- Visual fidelity: scaling helps reconstruction latents more（VAE在DiT-L反超）
- Policy performance: scaling helps reconstruction latents more（gap narrows）
- Action geometry: scaling helps semantic latents more（CEM gap不缩小，甚至扩大）

这暗示一个东西：**visual generation和action modeling是decoupled的capacity需求**。Reconstruction latents需要大DiT来"补回"semantic info；semantic latents从小DiT就有action advantage，scaling继续深化这个advantage。

### 7.4 与DINO-WM和V-JEPA 2-AC的关系

DINO-WM是autoregressive feature prediction world model（非diffusion）。V-JEPA 2-AC是JEPA-style predictor。这篇paper和它们complementary——它证明了在**diffusion**架构里semantic latents的优势依然成立。Diffusion的好处是能jointly denoise整个horizon，对multimodal futures和compounding errors更robust。

参考：
- DINO-WM: https://arxiv.org/abs/2506.15242 (实际上https://dinowm.github.io/)
- V-JEPA 2: https://ai.meta.com/blog/v-jepa-2/
- RAE: https://arxiv.org/abs/2510.11690
- S-VAE: https://arxiv.org/abs/2512.17909
- Bridge V2: https://proceedings.mlr.press/v229/walke23a.html
- OpenVLA: https://openvla.github.io/
- Flow matching: https://openreview.net/forum?id=PqvMRDCJT9t
- Cosmos: https://arxiv.org/abs/2501.03575
- V-JEPA 2.1: https://arxiv.org/abs/2603.14482

---

## 8. Recipe总结（Section 5）

paper给出一个实用recipe：

1. **不要optimize visual realism alone**——先选对action/task structure explicit的latent space
2. **Default to semantic encoders** (V-JEPA 2.1, Web-DINO, SigLIP 2)
3. **Pair with adapter** when decoded rollout quality or VLA-in-loop matters
4. **Transition model**: spatial-temporal DiT with causal temporal blocks + shallow-wide DDT head + dimension-aware noise shift
5. **Spatial blocks stay non-causal**（per-frame joint denoising）
6. **Diffusion forcing** (Chen et al., 2024) for autoregressive rollout
7. **Multi-axis evaluation**: visual + latent + downstream

---

## 9. 我的整体评价

这篇paper做了一件**很罕见的事**：在机器学习里，大部分papers propose a new method然后show SOTA。这篇paper做controlled evaluation study——固定所有其他变量，isolate一个design choice（latent space），然后show这个design choice对downstream robotics的影响远大于人们以为的。它的核心contribution是**reframing the question**：从"what reconstructs best"到"what preserves action-relevant structure"。

对你（Karpathy）来说，这可能特别resonant——你在"Software 2.0"和micrograd里都强调过：**objective function定义了模型学什么**。这里VAE latent的objective是pixel reconstruction，所以它encode的是pixel-sufficient info；semantic encoder的objective是prediction/contrastive，所以它encode的是structure/transformation info。World model需要的恰好转后者。**Latent space是objective function的fossilized form**。

潜在limitation：
1. 只在BridgeV2上做，没做ALOHA/Franka cross-embodiment
2. VLM judge可能有bias（虽然用2个VLM consensus + non-VLM metrics缓解）
3. 没比non-diffusion semantic world models（DINO-WM等）——但这是scope choice不是缺陷
4. DiT-L时VAE在visual metrics反超，如果再scale下去semantic的policy advantage会不会被visual advantage swallow？这是open question

如果让我predict next steps：
- 把这个study扩展到cross-embodiment（Franka, ALOHA, human hand）
- 把semantic encoder end-to-end finetune（partially）而非完全frozen
- Try video foundation models (e.g., video MAE, video CLIP)的latent
- 把policy distillation进latent space，让encoder本身action-aware
- 研究hierarchical latent: semantic coarse + reconstruction fine

很值得follow的工作，链接：https://hskalin.github.io/semantic-wm/ 和 https://huggingface.co/Nilaksh404/semantic-wm
