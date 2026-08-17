---
source_pdf: Latent Spatio-Temporal Chain-of-Thought for Robotic.pdf
paper_sha256: 6d5a954e08cba350515805501811771076d0714dad702304a57a13d98902f520
processed_at: '2026-08-05T12:17:18-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LaST₀ 人话版

## 一句话版本

让robot先在脑子里"想一下"要做什么，但这个"想"不在language space也不在image space，而是在一个compact的continuous latent space里快速过一遍future dynamics，然后conditioning高频action输出。

---

## 为什么这件事值得做

VLA最近的CoT路线有两个显而易见的坑。

**Text CoT的问题**: 你让模型输出"先抓锅铲，再靠近鸡蛋，然后铲尖插入鸡蛋底部..."这些text tokens。问题在于物理世界的信息密度远高于语言。铲子跟锅的接触角度、gripper闭合到什么程度、egg在铲上的滑动趋势——这些东西你用语言描述要么丢information要么啰嗦。而且生成几十个text token的latency就把你的control frequency拖垮了。

**Image CoT的问题** (CoT-VLA这类): 让模型预测future image作为"想象"。听起来很美，但生成一张256×256 RGB image要decode几百个token，1.1 Hz的速度在real robot上基本没法用。而且pixel-level generation的compute很多花在了background texture这种对action无关的细节上。

LaST₀的insight很简单: 你真正需要的"future reasoning"就三样东西——**future scene会怎么变(future RGB latent), future 3D结构会怎么变(future point cloud latent), robot自己会处于什么状态(future proprioception latent)**。每样东西压缩成一个token，预测4个future step，总共12个latent token。比生成text或image cheap两个数量级，但physical information content反而更focused。

---

## Latent CoT到底在predict什么

这是核心intuition。传统CoT你predict的是discrete tokens（text或image VQ codebook indices），LaST₀你predict的是continuous vectors。

对每个future step $k$:

$$z_k^v = \text{SigLIP}(I_{t+k}) \rightarrow \text{pool} \rightarrow \mathbb{R}^{2048}$$
$$z_k^p = \text{Uni3D}(P_{t+k}) \rightarrow \text{pool} \rightarrow \mathbb{R}^{2048}$$
$$z_k^s = \text{ActionTokenizer}(s_{t+k}) \rightarrow \mathbb{R}^{2048}$$

然后interleave起来:
$$\mathcal{Z}_{GT} = [z_1^v, z_1^p, z_1^s, z_2^v, z_2^p, z_2^s, ..., z_4^v, z_4^p, z_4^s]$$

Supervision用cosine similarity，不是MSE不是cross-entropy:
$$\mathcal{L} = \sum_t \left(1 - \frac{\hat{z}_t \cdot z_t^{GT}}{\|\hat{z}_t\| \|z_t^{GT}\|}\right)$$

为什么cosine？因为contrastive pretrained encoder（SigLIP, Uni3D）输出的embedding norm不一定informative，direction才informative。MSE会强迫模型去match norm，浪费capacity。cosine只要求direction对齐，更clean。

**关键ablation**: per modality 1个token足够，加到2或4个token没有显著提升。这说明latent reasoning不需要很多"step"，关键在于有一个continuous intermediate state让model condition on。这跟Coconut在language reasoning上的发现完全一致——continuous CoT的capacity不在token数，而在representation richness。

---

## Dual-System怎么不互相拖累

如果你用一个模型，slow reasoning和fast action都在里面跑，那inference时slow部分会block fast部分。

LaST₀用Mixture-of-Transformers (MoT)。两个expert住在同一个$d=2048$ latent space，share self-attention context，但FFN/QKVO/LayerNorm参数独立。Slow expert管latent CoT生成，fast expert管action输出。

然后异步频率: slow expert每4步跑一次 ($\kappa=4$)，fast expert每步都跑。中间3步fast expert通过KV cache cross-attend到slow expert已经写好的latent CoT上。

数字: slow expert 12.7 Hz，fast expert 22.1 Hz，整体15.4 Hz on RTX 4090。对比CoT-VLA的1.1 Hz，14× speedup。

**Training trick**: 不是固定训1:4 ratio，而是mix 1:1, 1:2, 1:4 ratio一起训。这样deploy时可以adaptive切换，且model对latent staleness robust。Mixed training带来3-7% absolute gain over固定ratio。

---

## 最让我买账的几个数字

1. **vs CoT-VLA同backbone**: 两个都用Janus-Pro，LaST₀ 82% vs CoT-VLA 66%, 14× faster。这是controlled comparison，说明latent CoT > explicit image CoT这件事是真实的，不是backbone差异。

2. **vs $\pi_{0.5}$**: LaST₀用400K trajectory pretrain就比$\pi_{0.5}$(海量pretrain)高17%。这个gap太大了不能用data解释，latent CoT确实在帮model"think ahead"。

3. **Long-horizon Place egg 3段连续**: LaST₀ 0.66→0.47→0.33, $\pi_{0.5}$ 0.47→0.20→0.07。Step 3差26 points。$\pi_{0.5}$靠current observation conditioning，task progress信息会随horizon丢；LaST₀的latent CoT隐式encode了task progress。

4. **Per modality 1 token足够**。这个ablation结果反直觉但深刻。Latent reasoning的capacity不在token budget，在于continuous representation的richness。

5. **跨embodiment generalization**: Franka 7-DoF, AgileX mobile 20-DoF, TienKung dexterous 26-DoF。只有noised-action MLP和final projector需要retrain，其他共享。说明latent CoT representation学到的是embodiment-agnostic的physical dynamics reasoning。

---

## 我会worry的几件事

**1. Latent target quality全靠encoder**。SigLIP-Large对manipulation场景的visual feature是否optimal？没做encoder choice ablation。如果换DINOv2或自己在robot data上finetune SigLIP，cosine supervision还work吗？

**2. Point cloud encoder只在train用**。Inference时模型"自己脑补"几何。如果scene geometry novel，没prior可以fallback。缺generalization to unseen geometry的实验。

**3. Open-loop latent prediction**。LaST CoT预测H步future，中间不和环境交互。如果中间有perturbation，predicted trajectory会stale。Mixed ratio training部分缓解，但没测试dynamic obstacle场景。

**4. "Ineffable physical dynamics capture"是核心claim但没法直接prove**。Latent是black box。Attention heatmap(Figure 10)显示LaST₀ attention确实聚焦在robot-object interaction region，这是indirect evidence但不是proof。需要更多probing experiments。

**5. CoT-VLA reimplementation的公平性**。作者自己说reimplement CoT-VLA到Janus-Pro上。原版CoT-VLA可能用了更强backbone或更多data，这个comparison只能说明"同backbone下latent > explicit"，不能说明"latent CoT这个paradigm本身更强"。

---

## 给你的intuition

LaST₀本质上是**robot领域的Coconut时刻**。NLP领域Coconut证明了continuous latent reasoning可以在language task上替代discrete CoT，LaST₀把这件事搬到robot domain并加了三个domain-specific insight:

1. **Latent target用multimodal future state** (image+PC+proprio) 而不是纯language latent。Robot domain的physical dynamics天然multimodal。
2. **Dual-system异步频率**解决了latent reasoning和high-freq control的scheduling冲突。NLP没这个问题，robot有。
3. **VGGT合成点云pretraining**解决了Open X-Embodiment缺depth的data engineering痛点。

下一步interesting的方向:
- **Reasoning trigger mechanism**: 现在slow expert每$\kappa$步blindly跑一次，如果只在task transition或error detection时触发，能再省compute。
- **Latent CoT scaling law**: 1 token足够这个结论是否随model size变化？7B或70B下optimal token数会不会变？
- **Discrete-to-continuous CoT的curriculum**: 是否可以先训discrete CoT再distill到latent space？

参考链接:
- 项目页: https://vla-last0.github.io/
- Coconut: https://arxiv.org/abs/2412.06769
- CoT-VLA: https://arxiv.org/abs/2503.22020
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054

---

# LaST₀ 深度技术解读

## 1. Motivation: 为什么latent CoT for VLA

VLA领域里最近的CoT路线分为两派，一类生成textual plan (VLA-R1, ECoT), 一类生成future visual observation (CoT-VLA, DreamVLA)。这两派都踩到了两个坑：

**Latency坑**: autoregressive decoding的token数决定了inference cost。CoT-VLA要生成future image tokens（典型几百上千token），single step推理时间掉到1.1 Hz，远低于15 Hz左右的closed-loop manipulation需要的频率。RT-2风格的textual CoT同理。

**Representation坑**: 语言是离散的symbolic space，物理世界是连续multimodal的(2D appearance + 3D geometry + proprioception)。用language描述"抓锅铲把鸡蛋从锅里舀起来"丢掉了大量information（铲尖与pan的相对角度、gripper closure程度、手腕torque分布等）。Karpathy你自己也常强调 - continuous latent space的capacity远大于discrete text tokens。

LaST₀的核心claim是: 把reasoning从explicit linguistic/visual space搬到compact continuous latent space，在token-efficient (3×H个latent tokens)的前提下encode multimodal physical dynamics，并通过dual-system异步频率协调，让slow reasoning和fast acting共存。

参考链接:
- 论文项目页: https://vla-last0.github.io/
- CoT-VLA: https://arxiv.org/abs/2503.22020
- VLA-R1: https://arxiv.org/abs/2510.01623
- DreamVLA: https://arxiv.org/abs/2507.04447

---

## 2. LaST CoT: Latent Spatio-Temporal Representation

### 2.1 三模态latent construction

对每个future timestep $k \in \{1,...,H\}$，从三个互补modality抽latent:

- **Visual latent $z_k^v \in \mathbb{R}^{d_v}$**: SigLIP-Large对future RGB frame $I_{t+k}$编码，得到feature map后做average pooling压缩为单token。这里$pooled$的rationale是token efficiency - 不压缩的话每帧N_img个token (SigLIP-Large对384×384输入大概729个patch token)。
  
- **Point cloud latent $z_k^p \in \mathbb{R}^{d_p}$**: Uni3D对future point cloud $P_{t+k}$编码。Uni3D是OpenAI的3D foundation model（基于EVA-02 backbone + point tokenizer + contrastive pretraining）。**关键设计**: point cloud encoder仅在训练时用，inference时不参与。这是为了避开depth sensor的不准问题，让模型在inference时只看RGB也能推理几何。

- **Proprioceptive latent $z_k^s$**: future robot state $s_{t+k}$ (7-DoF pose for single-arm)经action tokenizer压缩。

interleaved causal ordering:
$$\mathcal{Z}_{GT} = [\mathbf{z}_1^v, \mathbf{z}_1^p, \mathbf{z}_1^s, \mathbf{z}_2^v, \mathbf{z}_2^p, \mathbf{z}_2^s, ..., \mathbf{z}_H^v, \mathbf{z}_H^p, \mathbf{z}_H^s]$$

总长度 $3 \times H$ tokens，H=4时仅12个latent tokens。对比CoT-VLA要解码future image (典型256×256 RGB = 几百tokens的image VQ)，token数差了一个数量级。

参考链接:
- SigLIP: https://arxiv.org/abs/2303.15343
- Uni3D: https://arxiv.org/abs/2310.06773

### 2.2 Supervision: cosine similarity regression

LaST不用discrete token prediction的cross-entropy loss，改用cosine similarity loss：

$$\mathcal{L}_{latent} = \sum_{t=1} \left(1 - \frac{\hat{\mathbf{z}}_t \cdot \mathbf{z}_t^{GT}}{\|\hat{\mathbf{z}}_t\| \|\mathbf{z}_t^{GT}\|}\right)$$

其中:
- $\hat{\mathbf{z}}_t$: 模型自回归预测的第$t$个latent token (维度$d=2048$)
- $\mathbf{z}_t^{GT}$: ground-truth latent target (来自frozen encoder)

这个设计很关键。如果用MSE，会强制要求predict exact magnitude，但latent space的norm往往被encoder的归一化方式所决定（SigLIP输出一般L2-normalized）。cosine similarity只关心directional alignment，对scale不变，符合contrastive pretraining出来的embedding的几何性质。

这跟Coconut (Hao et al. 2024)的思路类似 - continuous CoT不需要discrete quantization，也不需要token vocabulary，直接在embedding space做regression。Karpathy你大概也注意到了，连续latent reasoning这块近半年进展很快 (Coconut, MONeT, PaTH)。

参考链接:
- Coconut (Training LLMs to reason in continuous latent space): https://arxiv.org/abs/2412.06769
- MONeT: https://arxiv.org/abs/2511.21395

### 2.3 Sequence structure和special tokens

设计三个special tokens: `<latent_start>`, `<latent_end>`, `<latent_pad>`

- Training时: 用 $\mathcal{Z}_{GT}$ 替换中间的`<latent_pad>`，teacher forcing训练。
- Inference时: 模型看到`<latent_start>`后跟一串`<latent_pad>`，自回归逐位填latent embedding（不是填token id，是填continuous vector）。

这是和discrete CoT很不同的地方 - inference时仍然走autoregressive，但每步output的是一个$d$-dim continuous vector，而不是logits over vocabulary。

---

## 3. Mixture-of-Transformers: Dual-System Architecture

### 3.1 为什么不用两个独立模型

如果slow reasoning expert和fast acting expert是两个分开的模型，它们之间信息传递要么通过重新encode latent sequence，要么通过额外adapter，引入latency和bottleneck。MoT设计让两个expert共享global self-attention context但参数独立，类似DeepSeek-MoE / Mixture-of-Transformers (Apple 2024)的思路，让fast expert可以"看到"slow expert已经写入KV cache的latent CoT tokens。

参考链接:
- Mixture-of-Transformers (Apple): https://arxiv.org/abs/2412.13670

### 3.2 具体参数分配

backbone: Janus-Pro，底层是DeepSeek-LLM 1.5B (24层, $d=2048$, hidden dim看DeepSeek架构)。

MoT改造：
- **共享**: self-attention的global context (即QKV attention计算本身在共享的token sequence上做)
- **不共享（每expert独立）**: 
  - FFN
  - Q, K, V, O projections ($W_Q, W_K, W_V, W_O$)
  - LayerNorm

这意味着两个expert住在同一个$d=2048$ latent space，但用各自的QKVO/FFN参数对input projection，最后在shared self-attention里interaction。

这是MoT区别于MoE的核心: MoE是router选expert，MoT是所有expert对全部token并行计算(类似multi-head但expert化)。

### 3.3 Action expert: Flow Matching head

Fast acting expert的action head用flow matching（不是diffusion，也不是discrete tokenizer）。

flow matching loss $\mathcal{L}_{flow}$的一般形式（Black et al. pi0 formulation）:

给定action chunk $\mathbf{a}_{t:t+H_a}$和Gaussian noise $\epsilon$, 定义$u_t(\mathbf{a}_t | \mathbf{a}_1)$为概率路径的velocity field：

$$\mathcal{L}_{flow} = \mathbb{E}_{t, \mathbf{a}_0, \mathbf{a}_1}\left[\|v_\theta(\mathbf{a}_t, t) - (\mathbf{a}_1 - \mathbf{a}_0)\|^2\right]$$

其中:
- $t \in [0,1]$: flow时间步
- $\mathbf{a}_0 \sim \mathcal{N}(0, I)$: 起点noise
- $\mathbf{a}_1$: ground-truth action chunk
- $\mathbf{a}_t = (1-t)\mathbf{a}_0 + t\mathbf{a}_1$: 线性插值路径
- $v_\theta$: 神经网络预测的velocity field

LaST₀里这个$v_\theta$由acting expert的最后hidden state + timestep MLP + noised-action MLP组合project出来。 timestep MLP用sinusoidal embedding初始化（类似Diffusion Transformer）。

Flow matching相比diffusion的好处：1) deterministic trajectory (linear)，2) 训练时不需要iterative denoising chain（ODE/SDE采样），3) inference时少step就能生成高质量action chunk。pi0用flow matching已经验证这条路在VLA里work。

参考链接:
- pi0: https://arxiv.org/abs/2410.24164
- Flow Matching for Generative Modeling (Lipman et al.): https://arxiv.org/abs/2210.02747

---

## 4. Dual-System Coordination: 异步频率

### 4.1 Frequency ratio $\kappa$

设计: 
- slow reasoning expert: 每 $\kappa$步激活一次 ($t \mod \kappa = 0$)
- fast acting expert: 每步都激活

测试时主要用 $\kappa = 4$ (即reasoning每4步更新一次latent CoT, action每步生成)。

Training时mix不同ratio (1:1, 1:2, 1:4)训练，让模型对latent stale程度robust。这点很聪明 - 如果只训1:4，模型遇到部署时real-time需要切换ratio就废掉；mixed training相当于curriculum over staleness。

### 4.2 KV cache机制

这是性能数字（15.4 Hz on RTX 4090）的关键：

- slow expert在$t \mod \kappa = 0$时跑一次自回归latent generation，写KV cache（K, V tensors对应latent tokens位置）。
- 中间$\kappa - 1$步，fast expert只encode当前observation token，然后对frozen的latent CoT KV cache做cross-attention（实际上是full self-attention with frozen portion），检索CoT信息，cost $O(1)$ per retrieval。
- 实测频率：slow expert 12.7 Hz，fast expert 22.1 Hz，整体 15.4 Hz。

这相当于把latent CoT当成一个"working memory"，每次reasoning更新一次写入，每个control step读取。本质上是hierarchical inference scheduling。

---

## 5. 训练Recipe

### 5.1 大规模pretraining (400K trajectories, 28M frames)

数据混合:
- BridgeV2: 20.93%
- Kuka: 20.22% 
- Fractal (RT-1): 13.67%
- Robo-Net: 11.53%
- Language Table: 7.72%
- BC-Z: 7.54%
- ManiSkill: 5.26%
- DROID: 4.82%
- RoboMIND: 0.2% (新数据集)
- ...

**关键技巧**: 大部分Open X-Embodiment数据没有depth sensor，但LaST CoT需要point cloud。他们用**VGGT** (Visual Geometry Grounded Transformer, CVPR 2025)对RGB frames生成synthetic point cloud。pretraining时用synthetic PC作为$z_k^p$的GT，fine-tuning时再用真实depth camera的点云，避免domain gap爆炸。

这个细节让我想起pi0.5也有类似用synthetic auxiliary supervision的策略。VGGT用DUSt3R-style architecture从image pair直接predict pointmap和camera pose，monocular也work。

参考链接:
- VGGT: https://arxiv.org/abs/2503.05118
- RoboMIND: https://arxiv.org/abs/2512.24653

### 5.2 SFT joint training

两个expert同时训练：
$$\mathcal{L}_{total} = \mathcal{L}_{latent}^{slow} + \mathcal{L}_{flow}^{fast}$$

action expert训练时latent CoT input会被随机延迟（即latent state可能对应几个step前的reasoning output），模拟部署时staleness。

### 5.3 Backbone选择: Janus-Pro

Janus-Pro是DeepSeek的多模态模型，特点: unified understanding + generation架构。LLM part是DeepSeek-LLM 1.5B。

为什么选Janus-Pro而不是Qwen-VL或LLaVA？我猜是因为Janus-Pro的设计本身就有generation head（可以生成image tokens），把latent space换成continuous CoT tokens比较natural。CoT-VLA也基于Janus-Pro重训，做了controlled comparison。

参考链接:
- Janus-Pro: https://arxiv.org/abs/2501.17811
- DeepSeek-LLM: https://arxiv.org/abs/2401.02954

---

## 6. 实验数据深读

### 6.1 RLBench仿真 (Table 1)

| Method | Mean S.R. | Infer. speed |
|---|---|---|
| OpenVLA (7B) | 0.40 ± 0.02 | 6.3 Hz |
| SpatialVLA | 0.46 ± 0.03 | 7.9 Hz |
| CogACT (7B) | 0.61 ± 0.04 | 9.8 Hz |
| CoT-VLA | 0.66 ± 0.03 | 1.1 Hz |
| $\pi_{0.5}$ (3B) | 0.65 ± 0.04 | 13.8 Hz |
| HybridVLA (7B) | 0.74 ± 0.04 | 6.1 Hz |
| LaST₀ (3.3B) | **0.82 ± 0.03** | **15.4 Hz** |

观察:
- LaST₀在3.3B参数下达到82% S.R.，比7B HybridVLA高8 points，比3B $\pi_{0.5}$高17 points。**参数效率比**很可观。
- vs CoT-VLA: 14× speedup + 16% success rate gain。这是LaST核心论点的最强证据 - 同backbone (Janus-Pro)下，latent CoT比explicit image generation CoT更快且更准。
- 在7/10个任务上达到最高。最难几个任务如"Phone on base", "Umbrella out", "Wine at rack"LaST₀优势最明显（20-40% absolute gain），这些任务都需要长horizon精确spatial reasoning，符合latent CoT的强项。

### 6.2 关键ablations (Figure 5)

**(a) 三模态贡献**:
- 仅2D image latent: 74%
- 仅3D point cloud latent: 76%  
- 仅robot state latent: 75%
- 2D+3D: 79%
- 2D+state: 80%
- **All three**: 82%

意外发现: 单modality latent都能从baseline 68%拉到74-76%。说明latent CoT本身就是个strong baseline，不需要modality多。但multi-modal stacking仍然有3-8%的cumulative gain。

**(b) 每modality token数**:
- 0 tokens (无CoT): 68%
- 1 token: 82% (+14%!)
- 2 tokens: ~82% (无显著提升)
- 4 tokens: ~82%

这是非常strong的结论: **per modality 1 token就足够**。再加token数是diminishing return。这跟Coconut在reasoning task上的观察一致 - latent reasoning不需要很多step，关键是给一个continuous "thinking step"即可。

**(c) Temporal coverage $H$**:
- $H=0$: 68%
- $H=4$: 82% (+14%)
- $H=5,6$: ~82%

H=4是甜点。多预测几步future对未来action generation help不大，因为action chunk本身就是短的。

**(d) Fast-slow ratio**:
- 1:1, 1:2, 1:4: 75-79%
- 1:8: 74% (明显下降)
- Mixed training + 1:4 inference: 82%

Mixed训练带来3-7% absolute gain over固定ratio训练。staleness robustness很重要。

### 6.3 Real-world (Table 2)

Franka单臂+双臂6个任务: LaST₀ mean 72%, vs $\pi_{0.5}$ 59%, vs CoT-VLA 50%, vs SpatialVLA 41%。

Long-horizon (Place egg on bread, 3段连续): 
- LaST₀: 0.66 → 0.47 → 0.33
- $\pi_{0.5}$: 0.47 → 0.20 → 0.07

Step 3时LaST₀比$\pi_{0.5}$高26 points。说明latent CoT对long-horizon task progression tracking更强 - latent state可以"记住"task progress，而$\pi_{0.5}$依赖当前observation conditioning，progress信息容易丢。

Mobile (AgileX, 20-DoF action): Arrange dishes 0.67, Sort spoon 0.27。两个任务难度差很大，但LaST₀在Arrange dishes上达到0.67，而$\pi_{0.5}$只有0.47。

Dexterous hand (TienKung, 26-DoF action): Open drawer 0.87, Place button 0.60。这两个任务对fine-grained finger control要求高，LaST₀的open drawer成功率0.87 vs $\pi_{0.5}$ 0.67。

跨embodiment scalability好, 因为只有noised-action MLP和final projector MLP需要retrain（架构里其他部分跨embodiment共享）。

### 6.4 Attention heatmap (Figure 4, 10)

LaST₀的attention高度集中在robot-object interaction region。CoT-VLA反而关注background texture，no-CoT baseline更分散。说明latent CoT通过cosine similarity supervision确实让model学到了task-relevant spatial focus，比image generation CoT更"focused"。

---

## 7. 我的critique和intuition

**优点**:
1. **Cosine similarity loss over continuous latent**这一步是clean formulation，避免了quantization或diffusion decoding，token-efficient的设计哲学贯穿全文。
2. **Mixed ratio training** + **KV cache for cross-expert attention**这两个工程细节让dual-system真正deployable。
3. **VGGT合成点云pretraining**解决了Open X-Embodiment缺depth的工程痛点，思路聪明。
4. **Per modality 1 token足够**这个ablation结果很有启发性，给未来latent CoT工作节省compute budget指了路。

**我会worry的点**:
1. **Latent supervision targets依赖frozen encoder质量**。SigLIP-Large对manipulation场景的visual feature是否最优？如果换DINOv2或CLIP+finetune，cosine loss还有效吗？paper没有做encoder choice ablation。
2. **Point cloud encoder只在训练时用**，意味着inference时模型"自己脑补"几何。如果scene geometry novel，模型没有prior knowledge可以fallback。没有做generalization to unseen geometry的实验。
3. **LaST CoT是open-loop prediction**。模型预测H步future state，但中间不和环境交互。如果中间有perturbation，predicted latent trajectory就会stale。他们用mixed ratio training部分缓解，但没测试dynamic obstacle avoidance scenario。
4. **Asymmetric design**: slow expert拿slow image (低频相机) + text，fast expert拿fast image (高频相机) + latent CoT。但实际部署中两个相机的calibration和frame rate差异如何handle？paper说用不同camera（FR3 setup里: third-person slow + wrist fast）。这种sensor asymmetry在产业化部署里会引入额外复杂度。
5. **Comparison公平性**: CoT-VLA被作者reimplement到Janus-Pro上，但CoT-VLA原paper可能用了更强backbone或更多data。HybridVLA 7B参数量比LaST₀ 3.3B大2倍多，parameter efficiency上LaST₀占优但绝对值上不一定可比。值得看的是当LaST₀ scale到7B会怎样。
6. **No comparison with pi0.5 GT variant** ($\pi_{0.5}$也有generative pretraining)。$\pi_{0.5}$用Open-X pretraining scale很大，LaST₀只用了400K trajectory，但都比$\pi_{0.5}$高17%。要么是RLBench-specific fine-tune差异，要么是latent CoT带来的真正gain。需要看cross-embodiment generalization eval。

**最有趣的intuition点**:
- 1 token per modality足够这个结论跟"thinking in latent space"这个研究方向的根本假设吻合：reasoning不需要discrete step-by-step token generation，关键在于activation state的trajectory。这跟Coconut, MONeT在language reasoning上的结论一致 - 现在VLA domain也有corresponding evidence。
- "Latent CoT captures ineffable physical attributes"这个claim本质上没法直接prove（latents是black box），但attention heatmap visualization (Figure 10) + 17% over $\pi_{0.5}$（同样大量pretraining，无显式CoT） + 16% over CoT-VLA（同样Janus-Pro backbone, explicit image CoT）三者共同支持了这一hypothesis。
- Dual-system architecture跟Kahneman的System 1/System 2类比很自然，但他们没用fast-slow weights sharing之外的inductive bias。其实可以加一个"reasoning-trigger" mechanism - 只在task transition或error detection时触发slow expert，进一步省compute。这点MONeT和ThinkAct可能有相关工作。

参考链接:
- ThinkAct: https://arxiv.org/abs/2507.16815
- LCDrive (autonomous driving latent CoT): https://arxiv.org/abs/2512.10226
- Latent CoT survey: https://arxiv.org/abs/2505.16782

---

## 8. 跟相关工作的connection

| Method | Reasoning form | Inference cost | Key difference vs LaST₀ |
|---|---|---|---|
| ECoT (Li et al. 2024) | Text CoT | Slow (text decoding) | Discrete, linguistic, no spatial |
| CoT-VLA | Future image prediction | 1.1 Hz | Explicit visual, expensive |
| DreamVLA | Multimodal future generation | Slow | Generation-based, similar idea |
| $\pi_{0.5}$ | No explicit CoT | 13.8 Hz | Direct VLA flow matching |
| HybridVLA | Diffusion+autoregressive | 6.1 Hz | Hybrid action head |
| SpatialVLA | Spatial prior | 7.9 Hz | 3D-aware, no temporal CoT |
| **LaST₀** | **Latent multimodal CoT** | **15.4 Hz** | **Compact, multi-modal, dual-system** |
| Coconut (NLP) | Continuous latent reasoning | Medium | Foundation work, language only |
| LCDrive | Latent rollout (driving) | Medium | Same idea, different domain |
| ThinkAct | Compressed motion plan | Medium | Latent planning |

LaST₀在VLA里站在几个research thread的intersection:
1. Continuous latent CoT (Coconut, MONeT) → 拿到robotic domain
2. Dual-system fast-slow (Kahneman-inspired) → MoT implementation
3. Multi-modal future prediction (DreamVLA, WorldVLA) → 改成latent space
4. Flow matching action head (pi0) → 借鉴action expert

整合度很高，engineering polish也很到位。

---

## 9. 总结一句

LaST₀本质上是把VLA的"reason-before-act"从discrete symbolic reasoning搬到continuous latent reasoning，用token-efficient (1 token per modality per step) + dual-system MoT + KV cache让latent CoT能real-time deploy。数字过硬(82% RLBench, 15.4 Hz, +13-14% real-world across三种platform)，但latent CoT的"ineffable physical dynamics capture"这个core claim还是靠indirect evidence支撑。下一步interesting的方向是discrete-to-continuous reasoning的scaling law，以及reasoning trigger mechanism (什么时候启动slow expert)。
