---
source_pdf: Evo-1 Lightweight Vision-Language-Action Model.pdf
paper_sha256: 60225cdb6318ac2311301e0b1858a52b406fe7ef21744320299cb5c2402c937e
processed_at: '2026-08-04T05:38:41-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Evo-1 用人话讲

## 1. 这篇paper到底在干啥

一句话：**他们做了一个0.77B的小VLA模型，不用robot data预训练，打爆了3.5B的π0和2.25B的SmolVLA。**

为啥这事儿难？你想想现在VLA field的"常识"：

- 想要strong generalization → 得用大backbone（7B起步）
- 想要cross-task transfer → 得用OXE/DROID大规模robot data pretrain
- 想要好performance → 得end-to-end fine-tune整个model

Evo-1说：这三条都可以推翻。key insight是**别把VLM的semantic space搞坏了**。

GitHub: https://github.com/MINT-SJTU/Evo-1

---

## 2. 为啥别人的VLM会被搞坏

让我给你build intuition。

你有个pretrained VLM（比如Prismatic-7B），它能很好地理解"红色杯子在桌子左边"这种visual-linguistic semantics。现在你想fine-tune它做robot control——给定image + instruction，输出action。

**Naive做法**：直接end-to-end训练，让VLM的output经过一个action head，backpropagation一路传回VLM每一层。

**会发生啥？** Action generation的gradient非常noisy（action space是continuous的、high-dimensional的、multimodal的），这些noisy gradient一路backprop到VLM，把VLM原本学好的"什么是杯子""什么是左边"的representation给搞乱了。

Paper里Figure 2直接可视化了这个现象：
- OpenVLA训练完，Prismatic-7B的attention map变得scatter、loss semantic coherence
- Evo-1训练完，InternVL3的attention map还保持clear、聚焦在task-relevant object上

这就像让一个受过良好教育的文科生去学杂技，学习方式不对的话，他可能连字都写不好了。

---

## 3. Evo-1的三个key design choice

### Choice 1: 用native multimodal的VLM

啥叫"native multimodal"？就是InternVL3从一开始就joint train image和text，而不是先train一个text-only LLM再嫁接vision encoder。

**Post-hoc alignment的问题**：你先train了一个很强的text LLM，它的representation space是为text generation优化的。然后你硬塞个vision encoder进来，通过projection layer对齐。这个对齐是surface-level的——model能产生"image of a cat"的description，但内部representation里image和text还是两个sub-system。

**Native multimodal的好处**：InternVL3从头开始joint learn，visual和linguistic representation在同一个embedding space里tight纠缠。这种representation对downstream fine-tune更robust——你fine-tune的时候，gradient扰动的是整个joint space，而不会单独drift visual或textual part。

InternVL3-1B的具体配置：
- InternViT-300M（从InternViT-6B蒸馏，用layer-wise negative cosine similarity loss）
- Qwen2.5-0.5B作为language decoder
- 448×448 input + pixel-unshuffle 4× downsampling，减少visual token数量

**InternVL3 paper**: https://arxiv.org/abs/2504.10479

**关键trick**：只用language branch的前14层，不要后面几层。为啥？因为深层的language decoder over-specialize到text generation（next token prediction），intermediate layers的representation更"raw"、更balanced between visual和linguistic，更适合做control conditioning。这跟SmolVLA的经验发现一致。

**SmolVLA paper**: https://arxiv.org/abs/2506.01844

---

### Choice 2: Cross-modulated Diffusion Transformer

这个我得多讲点，是paper最core的architecture innovation。

**背景**：现在VLA的action generation主流有两条路：
1. **Tokenize action**（OpenVLA）：把continuous action离散化成token，用autoregressive方式生成。优点是unified with LLM training，缺点是quantization error、generation慢。
2. **Diffusion/Flow matching**（π0, SmolVLA）：在continuous action space里denoise。优点是smooth、精确，缺点是需要action expert network。

Evo-1选了flow matching。为啥不选tokenize？因为continuous action对robot control更natural，flow matching比DDPM更sample-efficient。

**Flow matching的intuition**：

想象action space是个高维空间。Ground-truth action $A_t$ 是这个空间里的一个点。你从random noise $\epsilon$ 出发，想走到 $A_t$。

Flow matching的做法：定义一条从 $\epsilon$ 到 $A_t$ 的路径，让network学习这条路径上每个点的"velocity"（方向+速度）。训练时，你sample路径上的中间点 $A_t^\tau$，让network预测这个点应该往哪走。

公式3：
$$A_t^\tau = \tau A_t + (1-\tau)\epsilon$$

- $A_t$：ground-truth action
- $\epsilon$：noise
- $\tau$：interpolation coefficient，从Beta distribution采样，clamp到[0.02, 0.98]
- $A_t^\tau$：noisy action，训练时的input

clamp的intuition：$\tau$太接近0就是pure noise，network学不到东西；太接近1就是trivial target，没难度。Beta distribution让大部分sample集中在中间区域。

公式4 - 训练loss：
$$\mathcal{L}^\tau(\theta) = \mathbb{E}\left[\|\mathbf{v}_\theta(A_t^\tau, z_t, s_t) - \mathbf{u}(A_t^\tau | A_t)\|^2\right]$$

- $\mathbf{v}_\theta$：network预测的velocity field
- $\mathbf{u}(A_t^\tau | A_t) = A_t - A_t^\tau$：target flow direction（从当前点指向ground-truth的向量）
- $z_t$：VLM的fused representation
- $s_t$：robot state

训练就是让 $\mathbf{v}_\theta$ 去match target direction $\mathbf{u}$。

Inference时（公式5）：
$$\hat{A}_t = f_{\text{AE}}(z_t, s_t, A_t^\tau)$$

从pure noise开始，按network预测的velocity一步步走，走H=50步得到future action sequence $[\hat{a}_t, \hat{a}_{t+1}, \dots, \hat{a}_{t+H-1}]$。这50步action一次性预测出来，减少inference频率的要求。

**Flow Matching original paper**: https://arxiv.org/abs/2210.02747

---

**Architecture的核心创新**：纯cross-attention堆叠。

π0和SmolVLA的action expert是self-attention和cross-attention交替：
```
[Self-Attn] → [Cross-Attn] → [Self-Attn] → [Cross-Attn] → ...
```

Evo-1是纯cross-attention：
```
[Cross-Attn] → [Cross-Attn] → [Cross-Attn] → [Cross-Attn] → ...
```

**为啥这样work better？**

给你build intuition。Self-attention在action sequence内部做interaction——让action time step t和t+5互相attend。但flow matching的velocity prediction本质上是**conditional generation**：给定conditioning（$z_t$, $s_t$），预测当前noisy action $A_t^\tau$的velocity。这个prediction主要依赖conditioning信息，而不是其他action time steps。

Self-attention引入了额外的intra-action dependency modeling，这增加了optimization difficulty，而且不一定necessary。纯cross-attention让information flow更direct：conditioning → velocity prediction。

Ablation结果（Figure 8a）验证：纯cross-attention（Module A）比interleaved（Module B）在LIBERO-Long上高约10%。

**DiT paper**: https://arxiv.org/abs/2212.09748

---

### Choice 3: Integration Module的设计

这个连接VLM和action expert的module，paper里ablation了4种variant。

**最终方案（Module A）**：
1. 从VLM第14层extract $z_t$
2. 直接concatenate $z_t$ 和 robot state $s_t$，不project
3. Concatenated vector作为所有DiT layer的key-value
4. Noisy action $A_t^\tau$作为query

**为啥concatenate而不project？**

Project的意思是：用个linear layer把 $z_t$ 和 $s_t$ 都map到同一个shared embedding space。这看起来"elegant"，但实际loss information——你在force两个不同modality的feature进入同一个geometry。

Concatenate保留全部information，让cross-attention的attention weight自己学习怎么weighting $z_t$ 和 $s_t$。这更像"let the model decide"的philosophy。

**为啥所有DiT layer用同一套conditioning？**

Ablation里Module C试了layer-wise conditioning——每个DiT layer对应不同VLM layer的feature。看起来hierarchical、elegant，实际performance更差。

Intuition：layer-wise conditioning让每个DiT layer学习不同的conditioning interpretation，这增加了optimization difficulty。统一conditioning让所有DiT layer共享同一context，网络只需要学习"怎么用这个context"，更easy to optimize。

---

## 4. Two-stage Training - 最关键insight

这个training paradigm是paper最重要的contribution。

### Stage 1: Freeze VLM，只train action expert

**设置**：
```
VLM (frozen) → z_t → Integration Module (trainable) → Action Expert (trainable)
```

**Intuition**：Action expert是randomly initialized的。如果一开始就full fine-tune，action expert产生的noisy gradient会backprop到VLM，破坏pretrained semantic space。

先freeze VLM，让action expert学习"怎么读"VLM的output $z_t$。这个阶段action expert在fixed embedding space里学习alignment，gradient只在action expert内部flow，不扰动VLM。

Stage 1 hyperparams：lr=1e-5, 10k steps, batch=16。

### Stage 2: Unfreeze VLM，full fine-tune

**设置**：
```
VLM (trainable) → z_t → Integration Module (trainable) → Action Expert (trainable)
```

**Intuition**：Stage 1之后，action expert已经学会如何read VLM features，它产生的gradient更meaningful、less noisy。此时unfreeze VLM，gradient backprop到VLM是"good gradient"——告诉VLM"哪些features对action generation有用，请emphasize它们"。

这比一开始就full fine-tune温和得多。那时候action expert还random，产生的gradient是"noise"，会把VLM的semantic space搅乱。

Stage 2 hyperparams：lr=1e-5, 65k steps, batch=16。

### 类比

这就像教小孩学自行车：
- **Naive做法**：直接把小孩放车上让他自己骑，摔无数次，可能学会但过程painful，还可能留下心理阴影（VLM被破坏）
- **Evo-1做法**：先扶着车（freeze VLM，让action expert学习basic alignment），等小孩平衡感建立后松手（unfreeze VLM，joint refine）

### 验证：Attention map可视化

Paper Figure 7直接比较了single-stage vs two-stage训练后的VLM attention map：
- **Single-stage**：attention scatter，loss semantic focus，model不知道该看哪
- **Two-stage**：attention still聚焦在task-relevant object，semantic preservation成功

这visualization是这篇paper最convincing的证据。

---

## 5. 实验结果，看数据说话

### Meta-World (50 tasks, 4 difficulty levels)

| Model | Params | Easy | Medium | Hard | Very Hard | Avg. |
|---|---|---|---|---|---|---|
| Diffusion Policy | - | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA-H | 1.3B | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π0 | 3.5B | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA | 2.25B | 87.1 | 51.8 | 70.0 | 64.0 | 68.2 |
| **Evo-1** | **0.77B** | **89.2** | **76.8** | **77.2** | **79.2** | **80.6** |

看Very Hard那列：Evo-1是79.2%，SmolVLA是64.0%，π0只有30.0%。Very hard task需要最多cross-task generalization，这正是semantic preservation带来的优势。

### LIBERO (40 tasks)

| Model | Params | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|---|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SmolVLA | 2.25B | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| GR00T N1 | 2B | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | 3.5B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| **Evo-1** | **0.77B** | 92.7 | 97.7 | 96.3 | **92.3** | **94.8** |

LIBERO-Long最考验long-horizon reasoning，Evo-1拿了92.3%，比π0的85.2%高7.1个百分点。Long horizon需要model maintain task understanding across many steps，semantic preservation在这里显示价值。

### Real-World (xArm6, 4 tasks)

| Model | Params | GPU Mem | Freq | Success |
|---|---|---|---|---|
| SmolVLA | 0.45B | 2.0 GB | 12.7 Hz | 50.0% |
| OpenVLA | 7B | 15.1 GB | 7.9 Hz | 55.0% |
| π0 | 3.5B | 17.9 GB | 11.5 Hz | 73.0% |
| **Evo-1** | **0.77B** | **2.3 GB** | **16.4 Hz** | **78.0%** |

16.4 Hz的control frequency在RTX 4090d上，这对real-time robot deployment意义重大。OpenVLA只有7.9 Hz，太慢了，很多dynamic task做不了。

### Generalization (Pick and Place Can + disturbance)

| Condition | SmolVLA | Evo-1 |
|---|---|---|
| Base | 75% | 95% |
| Unseen distractor | 65% | 80% |
| Background color change | 60% | 75% |
| Position +30mm | 60% | 80% |
| Height +30mm | 60% | 70% |

所有disturbance条件下Evo-1都outperform SmolVLA。Background color change这种OOD visual disturbance上优势明显，说明VLM的visual representation确实被preserve得更好。

---

## 6. 这篇paper的大picture

让我zoom out讲讲为啥这篇paper重要。

### 对VLA field的意义

现在VLA field有两条路：
1. **Scale up**：更大的backbone、更多robot data、更强compute（π0、OpenVLA、GR00T N1路线）
2. **Efficient design**：更好的architecture、更聪明的training、更小的model（TinyVLA、SmolVLA、Evo-1路线）

Evo-1证明第二条路可以work，而且可以work得很好。0.77B打爆3.5B，这对academic lab和小公司意义重大——你不需要500台A100、不需要collect million条robot trajectory也能做出competitive VLA。

### 核心insight的generalization

"Preserve pretrained semantic space"这个insight其实很general：

- **LLM fine-tuning**：LoRA、QLoRA为啥work？因为它们limit gradient对backbone的perturbation，preserve pretrained knowledge。
- **CLIP adaptation**：linear probe vs full fine-tune的trade-off，linear probe preserve CLIP的general representation。
- **Continual learning**：EWC (Elastic Weight Consolidation)通过regularization保护important weights，防止catastrophic forgetting。

Evo-1的two-stage training本质上是一种curriculum：先让"student" (action expert) 学习如何read "teacher" (VLM)，再让teacher微微adapt。这个idea在很多transfer learning场景都applicable。

**EWC paper**: https://arxiv.org/abs/1612.00796

**LoRA paper**: https://arxiv.org/abs/2106.09685

### 局限和future work

Paper没充分讨论的：

1. **Action horizon H=50的ablation缺失**：为啥50不是10或100？更长horizon vs 更频繁replan的trade-off没分析。
2. **Beta distribution的参数没给**：Beta(α, β)的α和β是多少？这影响τ的采样分布，影响训练dynamics。
3. **Stage 2之后VLM到底变了多少**：attention map可视化qualitative，但quantitative analysis（比如feature的CKA similarity）缺。
4. **Failure mode没分析**：哪些task失败？为啥失败？
5. **Cross-embodiment transfer**：只测了xArm6和SO-100，没测zero-shot transfer到新embodiment。

可能的extension：
- 加RL fine-tuning在imitation learning基础上
- Hierarchical planning（像Hi-Robot）
- 3D perception（depth、point cloud）
- Tactile sensing
- Active perception（model自己控制camera）

---

## 7. 给你的intuition总结

如果只能记几句话：

1. **VLM的semantic space是宝贵资产，fine-tune时别搞坏它**。Two-stage training（先freeze再unfreeze）是preserve semantic space的简单有效方法。

2. **Native multimodal pretraining比post-hoc alignment更robust**。InternVL3从头joint train image+text，对downstream fine-tune的disturbance更tolerant。

3. **Pure cross-attention > interleaved self/cross attention**（在flow matching action generation场景）。Conditioning信息流比intra-action dependency更重要。

4. **Concatenate > project**（在integration module设计上）。保留全部information让attention自己weighting。

5. **Intermediate layer features > final layer features**（从VLM extract conditioning）。Final layer over-specialize到text generation，intermediate layer更balanced。

6. **0.77B可以打爆3.5B**。好的architecture design + 好的training strategy > 暴力scaling。

**Evo-1 GitHub**: https://github.com/MINT-SJTU/Evo-1

**InternVL3**: https://arxiv.org/abs/2504.10479

**Flow Matching**: https://arxiv.org/abs/2210.02747

**SmolVLA**: https://arxiv.org/abs/2506.01844

**π0**: https://arxiv.org/abs/2410.24164

**OpenVLA**: https://arxiv.org/abs/2406.09246

**DiT**: https://arxiv.org/abs/2212.09748

**LeRobot (data format)**: https://github.com/huggingface/lerobot

**DROID dataset**: https://droid-dataset.github.io/

**Open-X-Embodiment**: https://robotics-transformer-x.github.io/

**Evo-0 (前作)**: https://arxiv.org/abs/2507.00416

**GR00T N1**: https://arxiv.org/abs/2503.14734

**Hi-Robot**: https://arxiv.org/abs/2502.19417

**TinyVLA**: https://arxiv.org/abs/2409.12514

**VLA-Adapter**: https://arxiv.org/abs/2509.09372

**HybridVLA**: https://arxiv.org/abs/2503.10631

**π0-FAST**: https://arxiv.org/abs/2501.09747

希望这个version帮你build了intuition。如果你想deep dive某个specific part（比如flow matching的math、DiT architecture details、或者two-stage training的optimization landscape），尽管问。

---

# Evo-1: Lightweight Vision-Language-Action Model 深度解析

## 1. 核心问题与动机

当前VLA模型面临四个核心痛点：参数规模巨大（billions级别）、推理频率低（7-11 Hz）、end-to-end训练破坏VLM semantic space、依赖大规模robot data pretraining。Evo-1以0.77B parameters应对这四个问题，并且在三个simulation benchmark + real-world evaluation上取得SOTA或competitive结果。

让我build你的intuition：这篇paper的核心insight是**semantic preservation**。多数VLA模型在full fine-tuning过程中，pretrained VLM的attention map会发生semantic drift（Figure 2中Prismatic-7B出现degraded coherence），导致模型overfit到具体task，失去generalization。Evo-1通过two-stage training + native multimodal pretraining解决这个问题。

**GitHub repo**: https://github.com/MINT-SJTU/Evo-1

**InternVL3 paper**: https://arxiv.org/abs/2504.10479

**Flow Matching paper (Lipman et al.)**: https://arxiv.org/abs/2210.02747

**DiT (Peebles & Xie)**: https://arxiv.org/abs/2212.09748

---

## 2. 架构详解

Evo-1整体架构由三个核心component组成：Vision-Language Backbone → Integration Module → Cross-modulated Diffusion Transformer。

### 2.1 Vision-Language Backbone: InternVL3-1B

这里的关键设计choice值得仔细分析：

**为什么选InternVL3-1B？** 关键在于**native multimodal pretraining**。传统VLA model（如OpenVLA用Prismatic-7B）采用post-hoc alignment pipeline：先训练text-only LLM，然后retrofit去处理images。这种paradigm导致visual-linguistic alignment不够tight，在fine-tune时更容易发生semantic drift。

InternVL3-1B的结构：
- **Visual Encoder**: InternViT-300M，通过layer-wise negative cosine similarity loss从InternViT-6B蒸馏而来。这给你一个小而expressive的visual encoder。
- **Language Branch**: Qwen2.5-0.5B，仅0.5B params的transformer decoder。
- **Fusion mechanism**: patch-level image embeddings替换`<img>` placeholder token，输入shared transformer decoder做joint reasoning。

**Pixel-unshuffle downsampling**：每张RGB image resize到448×448后通过pixel-unshuffle操作，将spatial维度reduce 4×，减少visual token数量。这对inference speed至关重要——更少的visual token意味着cross-attention计算量降低。

**关键设计trick**：**只保留language branch的前14层**。Paper引用[26] SmolVLA的经验发现，intermediate layers展现出更强的cross-modal alignment。这给了你一个intuition：在VLA场景下，深层的language representations可能over-specialize到text generation，反而不利于visuomotor control。

**公式2**：
$$z_t = f_{\text{VLM}}\left(\{I_t^i\}_{i=1}^{N}, L_t\right)$$

变量解释：
- $z_t \in \mathbb{R}^{d_z}$：fused multimodal representation，同时encode视觉和语言信息
- $\{I_t^i\}_{i=1}^{N}$：N个视角的RGB observation（real-world setup中N=2：wrist camera + environment camera）
- $L_t$：language instruction
- $f_{\text{VLM}}$：整个VLM forward pass
- $t$：time step

这里$z_t$从第14层extract，作为后续action expert的conditioning。

### 2.2 Cross-modulated Diffusion Transformer

这是paper最core的architecture innovation。让我详细讲：

**Flow Matching vs DDPM**：Evo-1采用flow matching paradigm（Lipman et al. 2022, Liu 2022），而非传统DDPM。Flow matching的核心优势：
1. 训练更stable
2. Inference时ODE solver更efficient
3. Optimal transport视角更principled

**公式3 - Linear Interpolation**：
$$A_t^\tau = \tau A_t + (1-\tau)\epsilon$$

变量：
- $A_t$：ground-truth action sequence
- $\epsilon$：randomly sampled noise vector
- $\tau$：interpolation weight，从Beta distribution采样，clamped to [0.02, 0.98]
- $A_t^\tau$：interpolated noisy action

clamping的目的：避免$\tau$过小导致pure noise（训练不稳定），过大导致trivial target。

**公式4 - Flow Matching Loss**：
$$\mathcal{L}^\tau(\theta) = \mathbb{E}_{p(A_t|z_t, s_t), q(A_t^\tau|A_t)}\left[\|\mathbf{v}_\theta(A_t^\tau, z_t, s_t) - \mathbf{u}(A_t^\tau | A_t)\|^2\right]$$

变量含义：
- $\mathbf{v}_\theta$：time-conditioned velocity field，由DiT参数化，是要学习的network
- $\mathbf{u}(A_t^\tau | A_t)$：target flow direction，从当前$A_t^\tau$指向ground-truth $A_t$的向量
- $p(A_t|z_t, s_t)$：conditioned action distribution
- $q(A_t^\tau|A_t)$：interpolation distribution

直觉上：flow matching让network学习一个vector field，沿着这个vector field走，noise能flow到ground-truth action。loss让predicted velocity field match target flow direction。

**Architecture关键设计**：**纯cross-attention堆叠**，与π0、SmolVLA采用的self-attention + cross-attention交替结构不同。

为什么这样设计？我的理解：
1. Action sequence本身是temporal结构，self-attention对inter-action dependency建模有用，但在flow matching的vector field预测中，每个time step的velocity prediction主要依赖conditioning（$z_t$, $s_t$）而非其他action time steps
2. 纯cross-attention减少计算量（cross-attention的complexity是$O(n_{query} \cdot n_{kv})$，self-attention是$O(n^2)$）
3. 更stable的信息propagation（ablation中Module A胜出也印证这点）

**公式5 - Inference**：
$$\hat{A}_t = f_{\text{AE}}(z_t, s_t, A_t^\tau)$$

- $\hat{A}_t = [\hat{a}_t, \hat{a}_{t+1}, \dots, \hat{a}_{t+H-1}]$：预测的future action trunk
- $H=50$：action horizon，从implementation details可以看到
- $f_{\text{AE}}$：conditioned action expert network

**DiT layers = 8**，dropout = 0.2，action dimension = 24（padded，对应不同embodiment的最大配置）。

### 2.3 Integration Module

这个module的设计在ablation study中验证。最终采用**Module A: Mid-Layer Cross-Attention**：

设计：
1. 从VLM第14层extract fused feature $z_t$
2. **Concatenate** $z_t$ 与 robot state $s_t$（而非project到shared space）
3. Concatenated feature作为所有DiT layers的key-value
4. Noisy action $A_t^\tau$作为cross-attention的query

**为什么concatenate而非project？** Paper强调"preserve complete information from both perceptual embedding and proprioceptive state"。Project会force两者进入same dimension，可能loss information。Concatenate保留全部信息，让cross-attention自己学习如何weighting。

Ablation中比较的四种variant：
- **Module A** (最终采用): 中层feature + concat state + 纯cross-attention
- **Module B**: 中层feature + interleaved cross-self attention
- **Module C**: layer-wise cross-attention，每个DiT layer对应不同VLM layer的feature
- **Module D**: joint key-value，把VLM feature + state + noisy action都作为kv

结果（Figure 8a, LIBERO-Long）：Module A胜出。Paper的解释是其他variants引入信息propagation的中断，破坏continuity和consistency。

我的intuition：Module C的layer-wise injection看似能提供hierarchical perception，但实际增加了optimization difficulty，每层conditioning不同，network需要学习复杂的layer-specific alignment。Module A的uniform conditioning让所有DiT layers共享同一context，更容易学习。

---

## 3. Two-Stage Training Paradigm

这是paper第二个core contribution。让我详细讲为什么这样设计work。

### 3.1 Stage 1: Action Expert Alignment

**设置**：Freeze整个VLM backbone，仅train integration module + action expert。

**目的**：让randomly initialized的action expert weights先align到VLM的embedding space，而不back-propagate noisy gradient到pretrained backbone。

**直觉**：如果一开始就full fine-tune，random action expert会产生大量noisy gradient backprop到VLM，破坏pretrained semantic space。先freeze VLM让action expert学习如何read VLM features，建立coherent alignment。

### 3.2 Stage 2: Full-scale Fine-Tuning

**设置**：Unfreeze VLM，full fine-tune整个architecture。

**目的**：Joint refinement，让VLM也能slightly adapt到control task。

**直觉**：Stage 1后action expert已经stable，此时unfreeze VLM产生的gradient更stable，不会catastrophic破坏semantic space，但能fine-tune VLM让其对task-relevant features更sensitive。

### 3.3 Hyperparameters (Meta-World setup)

| Hyperparameter | Stage 1 | Stage 2 |
|---|---|---|
| Learning rate | 1e-5 | 1e-5 |
| Batch size | 16 | 16 |
| Max steps | 10k | 65k |
| Warmup steps | 1k | 1k |
| Gradient clipping | 1.0 | 1.0 |
| Weight decay | 0.001 | 0.001 |
| Resume from Stage 1 | No | Yes |

总训练量约75k steps，相对轻量。

### 3.4 Semantic Preservation的验证

Figure 2和Figure 7通过attention map可视化证明：
- Evo-1 (InternVL3-1B + two-stage)：训练后attention仍保持spatial consistency和semantic alignment
- OpenVLA (Prismatic-7B + end-to-end)：attention出现degraded coherence
- Single-stage训练的Evo-1：attention disrupted

**SmolVLA paper**: https://arxiv.org/abs/2506.01844

---

## 4. 实验结果深度分析

### 4.1 Meta-World Benchmark

50个manipulation tasks，分四个difficulty levels。

| Model | Params | Robo-Pretrain | Easy | Medium | Hard | Very Hard | Avg. |
|---|---|---|---|---|---|---|---|
| Diffusion Policy | - | No | 23.1 | 10.7 | 1.9 | 6.1 | 10.5 |
| TinyVLA-H | 1.3B | No | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π0 | 3.5B | Yes | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| SmolVLA | 2.25B | No | 87.1 | 51.8 | 70.0 | 64.0 | 68.2 |
| **Evo-1** | **0.77B** | **No** | **89.2** | **76.8** | **77.2** | **79.2** | **80.6** |

Evo-1比SmolVLA提升12.4%，比π0提升32.7%。值得注意的是very hard tasks（79.2% vs 64.0%），这表明semantic preservation带来strong generalization——hard tasks需要更多cross-task transfer能力。

### 4.2 LIBERO Benchmark

| Model | Params | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|---|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SmolVLA | 2.25B | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| GR00T N1 | 2B | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | 3.5B | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| **Evo-1** | **0.77B** | 92.7 | 97.7 | 96.3 | **92.3** | **94.8** |

LIBERO-Long上Evo-1最强（92.3%），这点很有意思——long horizon tasks需要compositional reasoning，semantic preservation帮助model maintain task understanding over longer horizons。

### 4.3 RoboTwin Benchmark (Dual-arm)

| Model | Click Alarmclock | Dump Bin | Place Bread | Place Can | Avg. |
|---|---|---|---|---|---|
| ACT | 14.0 | - | - | - | 14.0 |
| Diffusion Policy | - | - | - | - | 18.4 |
| RDT (1.2B) | - | - | - | - | 25.8 |
| π0 (3.5B) | 63.0/11.0 | 82.0/24.0 | 17.0/4.0 | 41.0/5.0 | 30.9 |
| **Evo-1** | **77.0/58.0** | 74.0/37.0 | 15.0/3.0 | 37.0/1.0 | **37.8** |

Evo-1在Click Alarmclock和Dump Bin上明显强于π0，但Place Bread上略低。Dual-arm coordination的成功表明lightweight model也能处理high-dimensional action space。

### 4.4 Real-World Experiments

4个tasks：Pick and Place Can, Pour Foam from Cup, Hand Delivery, Can Stacking。使用6-DoF xArm6 + parallel gripper + 2 cameras (wrist + environment)。

| Model | Params | GPU Mem. | Infer. Freq. | Success |
|---|---|---|---|---|
| SmolVLA | 0.45B | 2.0 GB | 12.7 Hz | 50.0% |
| OpenVLA | 7B | 15.1 GB | 7.9 Hz | 55.0% |
| π0 | 3.5B | 17.9 GB | 11.5 Hz | 73.0% |
| **Evo-1** | **0.77B** | **2.3 GB** | **16.4 Hz** | **78.0%** |

关键insight：Evo-1在RTX 4090d上达到16.4 Hz，远超π0的11.5 Hz。对于real-time robot control，30 Hz是理想目标，16.4 Hz已经sufficient for大多数manipulation tasks。

### 4.5 Generalization Experiments

Base task: Pick and Place Can。

| Condition | SmolVLA | Evo-1 |
|---|---|---|
| Base | 75% | 95% |
| Unseen distractor | 65% | 80% |
| Background color | 60% | 75% |
| Position +10mm | 75% | 95% |
| Position +20mm | 60% | 85% |
| Position +30mm | 60% | 80% |
| Height +10mm | 75% | 100% |
| Height +20mm | 65% | 90% |
| Height +30mm | 60% | 70% |

Evo-1在所有disturbance条件下都outperform SmolVLA，特别在background color change（75% vs 60%）和position shift（80% vs 60% at +30mm）上优势明显。这印证了semantic preservation带来的visual robustness。

---

## 5. 与相关工作的关联

### 5.1 VLA Model谱系

| Model | Backbone | Action Head | Params | Pretrain |
|---|---|---|---|---|
| RT-2 [34] | PaLI-X | Tokenization | 55B | Yes |
| OpenVLA [12] | Prismatic-7B | Discrete tokens | 7B | Yes (OXE) |
| π0 [7] | PaliGemma | Flow matching | 3.5B | Yes |
| TinyVLA [29] | Lightweight VLM | Diffusion | 1.3B | No |
| SmolVLA [26] | SmolVLM-2 | Flow matching | 2.25B | No |
| Hi-Robot [25] | - | Hierarchical | - | - |
| **Evo-1** | **InternVL3-1B** | **Cross-mod DiT** | **0.77B** | **No** |

**OpenVLA**: https://arxiv.org/abs/2406.09246

**π0**: https://arxiv.org/abs/2410.24164

**TinyVLA**: https://arxiv.org/abs/2409.12514

**Hi-Robot**: https://arxiv.org/abs/2502.19417

### 5.2 Flow Matching在Robotics中的应用

Flow matching的physics intuition：在action space中定义probability path，从noise distribution ($\tau=0$, pure noise) flow到data distribution ($\tau=1$, ground-truth action)。Velocity field $\mathbf{v}_\theta$指导这个flow。

与DDPM相比：
- DDPM：离散Markov chain，需要many steps
- Flow matching：连续ODE，可以用fewer steps with higher-order solver

π0-FAST [24]用autoregressive tokenization替代flow matching，但accuracy上略低（LIBERO 85.5% vs Evo-1 94.8%）。

**π0-FAST**: https://arxiv.org/abs/2501.09747

### 5.3 Semantic Preservation的更广意义

这个insight让我联想到：
- **LLM fine-tuning**：full fine-tune vs LoRA的trade-off
- **CLIP adaptation**：linear probe vs full fine-tune
- **Continual learning**：EWC (Elastic Weight Consolidation)通过regularization保护important weights

Evo-1的two-stage strategy本质上是一种**warmup curriculum**：先让"student" (action expert) 学习如何read "teacher" (VLM) 的output，再让teacher微微adapt。这与知识蒸馏的反向过程有点相似。

---

## 6. Critique和Open Questions

### 6.1 论文未充分讨论的点

1. **Action trunk H=50的choice**：为什么50？更长horizon的trade-off未ablation。
2. **Beta distribution for τ**：具体Beta参数未给出，可能影响训练stability。
3. **Stage 1 vs Stage 2的learning dynamics**：Stage 1后action expert的representation空间如何？Stage 2 fine-tune VLM的gradient magnitude如何控制？
4. **Cross-embodiment transfer**：paper测试了xArm6和SO-100，但不同embodiment间的zero-shot transfer未深入。
5. **Failure mode分析**：在哪些task上Evo-1失败？为什么？

### 6.2 可能的扩展方向

1. **Reinforcement Learning fine-tuning**：在imitation learning基础上加RL
2. **Long-horizon planning**：结合hierarchical structure like Hi-Robot
3. **3D scene understanding**：加入depth或point cloud
4. **Multi-modal tactile sensing**：加入触觉input
5. **Active perception**：让model主动控制camera视角

### 6.3 与最新工作的联系

**Evo-0** (reference [14]): https://arxiv.org/abs/2507.00416 - 这是同一group的前作，关注implicit spatial understanding。

**GR00T N1** (NVIDIA): https://arxiv.org/abs/2503.14734 - 2B humanoid foundation model，用了大量robot data pretraining。

**HybridVLA** [17]: https://arxiv.org/abs/2503.10631 - 结合diffusion和autoregressive。

**VLA-Adapter** [28]: https://arxiv.org/abs/2509.09372 - tiny-scale VLA的另一approach。

---

## 7. 总结性Intuition

让我用几句key insights总结：

1. **Native multimodal pretraining > post-hoc alignment**：InternVL3的joint training让visual-linguistic alignment更robust，fine-tune后不drift。

2. **Pure cross-attention > interleaved self/cross**：在flow matching的velocity prediction中，conditioning信息流比intra-action dependency更重要。

3. **Concatenation > projection**：保留全部信息让cross-attention自己学习weighting，比强制project到shared space更robust。

4. **Two-stage training > single-stage**：先warm-up student（action expert），再joint refine，保护teacher（VLM）的semantic space。

5. **Lightweight + semantic preservation可以beat large + pretrain**：0.77B params无需robot pretrain，超越3.5B的π0，key在于representation quality。

6. **Intermediate layer features更robust**：第14层的VLM features比最后layer更适合作为control conditioning，因为最后layer over-specialize到text generation。

这篇paper对VLA community的贡献在于证明：**通过careful architecture design和training strategy，lightweight model can achieve SOTA without massive pretraining**。这对降低robot learning的data/compute cost有重要意义，特别是让academic lab和小公司也能train competitive VLA models。

**LeRobot (HuggingFace)**: https://github.com/huggingface/lerobot - Evo-1使用的数据format。

**DROID dataset**: https://droid-dataset.github.io/ - 大规模robot dataset对比参考。

**Open-X-Embodiment**: https://robotics-transformer-x.github.io/ - cross-embodiment dataset。

希望这个深度解析帮你build了对lightweight VLA + semantic preservation的intuition。如果你对specific component（比如flow matching的mathematical derivation、DiT architecture details、或two-stage training的optimization landscape）想深入讨论，可以继续。
