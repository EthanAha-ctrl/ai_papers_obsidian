---
source_pdf: DexVLA Vision-Language Model with Plug-In Diffusion Expert for General
  Robot Control.pdf
paper_sha256: 17c827d9dafb7f1e2af751dc1a3e2594ae14cd27646badde3044b28d0fe95fb3
processed_at: '2026-08-18T05:31:21-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DexVLA

## 一句话版本

现在大家做 robot VLA 都在拼命堆 VLM 参数 (7B、3B 那么大),但这帮人没搞清楚一件事:**让机器人动起来的瓶颈根本不在"看懂"和"听懂",而在"怎么动"**。DexVLA 干的事就是:给 VLM 配一个真正会动东西的"小脑",一个 1B 参数的 Diffusion Transformer,然后用三步走的训练法让这俩家伙配合好,最后用 100 小时数据单卡 60Hz 干翻了 π₀。

项目主页:https://dex-vla.github.io/

---

## 为什么现在 VLA 不 work (我的直觉)

你看现在这些 VLA model,都有一个共同毛病:

**OpenVLA**:拿一个 7B 的 Llama,把 action 切成 256 个 bin,当 token 来 predict。这就好比你让我用"0-255 的数字"来描述"怎么打篮球"——量化误差先不提,关键是篮球动作是连续的、multimodal 的 (同一情况下你可以传球也可以投篮),用 next-token cross-entropy 一训练,模型就塌缩到"平均动作",动作变得犹犹豫豫。

**π₀**:VLM 3B + 一个小的 flow matching expert (大概 300M),好一些,但 expert 还是太瘦小,而且它要靠外挂 SayCan 每两秒切一次指令才能做长程任务——这就好比一个人做事每两秒就要停下来问"接下来干啥",不流畅。

**Octo**:93M 的小模型,Diffusion head,设计上没问题,但容量太小,做做简单 pick-place 还行,碰到 shirt folding 这种就歇菜。

我 (Karpathy) 看下来一个直觉:**大家都把算力和参数堆在"大脑皮层" (VLM) 上,但"小脑" (action generation) 还是只有几十兆到几百兆,这完全失衡**。你让一个博士级别的 VLM 指挥一个婴儿级别的小脑,结果就是动作僵硬、抖动、学不会精细活。

DexVLA 反过来干:**给小脑 1B 参数,让它真正学会动**。

---

## DexVLA 怎么做的 (架构层面的人话版)

整体长这样:

```
图像 + 语言指令 ──► Qwen2-VL 2B ─┬─► reasoning tokens (它自己想"接下来该干啥")
                                  │
                                  └─► action tokens (给小脑的条件)
                                        │
                                        ▼
                              1B Diffusion Transformer
                                        │
                                        ▼
                              输出连续 action chunk (未来 H 步动作)
```

几个关键点用人话解释:

### 1. 为什么要用 Diffusion 做 action head?

因为 action distribution 是 **multimodal** 的。啥意思?比如桌上有个杯子,你要抓它,可以从左边抓也可以从右边抓,两条 trajectory 都对。你要是用 MSE 回归,模型会学一个"左右平均"的中间姿势,啥也抓不到。Diffusion 天生能 model 多峰分布,这是 Diffusion Policy (Chi et al., https://arxiv.org/abs/2303.04137) 早就证明的事。

DexVLA 把这个 Diffusion head 做到了 **1B 参数**,32 层 Transformer,hidden 1280,16 heads。这么大的 Diffusion Transformer 用来做 robot action 是第一次。

### 2. VLM 的 reasoning 怎么传给 Diffusion head?

用 **FiLM** (Feature-wise Linear Modulation),公式很简单:

$$
\text{FiLM}(\mathbf{x}) = \boldsymbol{\gamma} \odot \mathbf{x} + \boldsymbol{\beta}
$$

- $\mathbf{x}$:Diffusion Expert 内部某一层的 feature
- $\boldsymbol{\gamma}, \boldsymbol{\beta}$:从 VLM reasoning token 算出来的 scale 和 shift 向量
- $\odot$:逐元素相乘

翻译成人话:**VLM 想"先把袖子对齐",这个想法不是一个 token 塞进 Diffusion head,而是整体地把 Diffusion head 的整个动作流形"扭一下"**,让它偏向"对齐袖子"那个区域。这比 concat conditioning 优雅得多,因为 noise 不会把 condition signal diffuse 掉。

### 3. Multi-head 设计:每个 embodiment 一个 output head

不同 robot 形态 (单臂、双臂、灵巧手) 的 DoF 和运动学完全不一样。DexVLA 给每种 embodiment 配一个独立 MLP head,共享 trunk。这就像 Octo (https://octo-models.github.io/) 的设计,但 trunk 大了 10 倍。

训练 loss:

$$
\mathcal{L} = \mathcal{L}_{diff} + \alpha \mathcal{L}_{ntp}
$$

- $\mathcal{L}_{diff}$:标准 DDPM noise prediction loss
- $\mathcal{L}_{ntp}$:VLM 的 next-token loss (保证它别忘词)
- $\alpha = 1$,实际观察到 $\mathcal{L}_{ntp}$ 很快收敛,后期优化重心自然偏向 diffusion

---

## 三阶段训练:这才是 paper 的精华

这是我觉得最妙的部分。DexVLA 不 end-to-end 一把梭,而是分三步走,每一步都有清晰的语义。

### Stage 1:只练小脑 (Cross-embodiment pre-training)

**关键决策:完全不用 VLM,只训练 1B Diffusion Expert**。

- 图像用 random-init ResNet-50 编码
- 语言用 DistilBERT (现成的)
- 数据:91 个 task,4 种 embodiment,~100 小时
- 纯 diffusion loss

为什么这么干?我 (Karpathy) 的理解:

1. **1B DiT 从零训练本身就难,要是同时 backprop 回 VLM,容易把 VLM 的 pretrain representation 弄崩**。这就像你不能让一个刚学走路的小孩同时背唐诗。
2. **算账**:只训 Diffusion Expert 是 0.89 epoch/hr,全 VLA 一起训只有 0.32 epoch/hr——**2.78 倍加速** (Table 6,https://arxiv.org/abs/2504.16054)。Stage 1 的 cross-embodiment motor pretraining 是廉价的。
3. **模块化 inductive bias**:VLM 已经在 internet data 上学会了语义,别让它被 motor signal 干扰;Diffusion Expert 应该先建立自己的 motor prior,类似婴儿先有 grasp reflex 再学语言。

这个 stage 学到的是**跨形态的通用运动技能**——抓取、移动、对齐这些抽象动作模板。

### Stage 2:对齐身体 (Embodiment-specific alignment)

接上 Qwen2-VL 2B,但 vision encoder 冻住,只训练 LLM 部分 + projection + Diffusion Expert。

- 数据 filter 成单种 embodiment
- 学习率从 1e-4 降到 2e-5 (保护 VLM)
- 5 epochs

这一步类比 LLaVA (https://arxiv.org/abs/2304.08485) 的 projector alignment:把 VLM 的高层 representation 对齐到特定 embodiment 的 action manifold。

**惊人的发现**:Stage 2 训完,模型在 shirt folding 上已经能拿 0.92 分,bin-picking、table bussing 也有不错表现。这意味着 VLM 的 semantic prior + Diffusion Expert 的 motor prior 加起来,能 zero-shot emerge 出复杂 skill。这就是 emergent capability 的典型例子。

### Stage 3:精细训练 (Task-specific adaptation)

- 类比 LLM 的 post-training / domain SFT
- 用 sub-step annotated 数据
- 学习率 2e-5,Cosine scheduler

**这一阶段最关键的 trick:Sub-step reasoning**。

把"fold the shirt"拆成 ["smooth wrinkles", "align sleeves", "secure folds"],训练时让 VLM 先 generate 这些 sub-step,再通过 FiLM 指导 action。推理时只输入"fold the shirt",模型自己 generate substep。

这个 ablation 数据非常震撼 (Table 7):
- 两个 stage 都不用 substep:0 分
- Stage 1 不用、Stage 2 用:0.07
- 两个 stage 都用:0.92

我 (Karpathy) 的解读:**sub-step 等价于在 continuous action manifold 上钉了一堆 discrete anchor points**,把长程任务切成短段,每段对应一个 sub-manifold,避免不同 sub-task 的 gradient 在共享参数里打架。本质上是把 mixture-of-experts 的思想用"时间切分"实现。

还有个对比 (Table 8):DexVLA 的 implicit substep reasoning (0.70) > π₀ 外挂 SayCan (0.58)。因为 SayCan 固定 2 秒切一次,容易 redundant 或 missing transition;DexVLA 是 state-adaptive,模型自己判断何时切换。

---

## 实验数据,挑几个最能说明问题的

### 1. Shirt folding (Figure 6) — 不用 task-specific 训练

- DexVLA:0.92
- OpenVLA、Octo、Diffusion Policy:全 ≈ 0

Shirt folding 需要 bimanual coordination + deformable object + 精确折叠,任何一个环节弱都做不到 0.92。这是个非常 litmus 的 task。

### 2. New embodiment,只给 100 demos (Figure 8)

- Franka + 灵巧手 pour drink:0.90 (DexVLA) vs 几乎 0 (OpenVLA/Octo)
- Bimanual UR5e packing:类似

100 demos 就能让一个新形态学会灵巧手 pouring,这说明 Stage 1 的 cross-embodiment motor prior 已经学到了"pouring"这个抽象 motor template,Stage 3 只是 specialize。本质是 **motor representation 的 few-shot transfer**。

### 3. Laundry folding (Figure 11) — 最硬的 long-horizon

> 2 分钟,衣服随机团成一团,要从 basket 里拿出来、摊平、折叠、叠到现有 stack 上。

- DexVLA:0.4
- π₀ (用 SayCan):0.2
- OpenVLA、Octo:0

π₀ 用了 10000 小时数据 + SayCan 高层 policy,DexVLA 只用 100 小时 + implicit reasoning,反而更强。这就是 implicit substep reasoning 的威力。

### 4. Size ablation (Table 3)

- UNet 93M:0.17 (作者观察到机器人抖动)
- DiT 410M:0.63
- DiT 1B:0.92

**Motor control 也有 scaling law**,而且比 language 似乎更陡。UNet 的 inductive bias (局部卷积 + skip) 适合图像,不适合 robot action (时序 + cross-embodiment)。

### 5. Zero-shot cross-embodiment (Appendix A.5) — 最让我兴奋

Stage 2 在 Franka + Robotiq gripper (1 DoF) 训练,推理时 swap 成 Inspire 灵巧手,**只用 1 DoF** (其他锁住):

- 30 unseen objects bin-picking
- 原 gripper:67%
- 换灵巧手:60%

只掉 7 个点!这说明:
1. **Visual representation 能跨形态 transfer** (即使外观差异大)
2. **Camera view 能 re-ground** (wrist camera 位置变了)
3. **Action manifold 在共享的 1-DoF 子空间上对齐**

这给我一个很强的 intuition:**embodiment-specific 的不是 action 维度本身,而是 DoF 之间的 correlation structure**。如果你 constrain 到一个 shared 1-DoF 子空间,跨 embodiment transfer 是可能的。这有点像 universal grammar,只不过发生在 action space。

### 6. LIBERO (Table 5)

| Method | Avg |
|---|---|
| DP | 79.7 |
| OpenVLA | 84.1 |
| π₀-FAST | 93.9 |
| π₀ | 97.1 |
| **DexVLA** | **97.3** |

在 simulation benchmark 上甚至略胜 π₀,说明 motor representation 确实更 expressive。

---

## 我 (Karpathy) 的几点直觉联想

### 1. Two-system brain hypothesis

人脑的小脑 (motor control) 和大脑皮层 (semantic reasoning) 是分离但互联的:
- 大脑皮层:慢、abstract、symbolic,负责"做什么"
- 小脑:快、continuous、习惯化,负责"怎么做"

DexVLA 完美对应:
- Qwen2-VL 2B = 大脑皮层
- Diffusion Expert 1B = 小脑
- FiLM = 皮层到小脑的神经调制通路

这比 OpenVLA 把所有东西塞进一个 LLM trunk 要符合神经科学 intuition。

### 2. Curriculum = Human development

- Stage 1:婴儿期 motor babbling,学 general motor skill,无语言
- Stage 2:学步期,把语言指令和 motor action 对齐
- Stage 3:成年期,通过 explicit practice 学复杂技能

### 3. Internal reasoning > External planner

π₀ 用 SayCan 外挂,本质是 multi-agent system,有 latency 和 coordination cost。
DexVLA 让 VLM 内部 generate substep,这是 in-context reasoning 的延伸,更像 internal monologue。

这让我想起 ReAct vs CoT 的讨论——**internal reasoning 总比 external tool call 更接近"智能"的本质**。

### 4. 联想到我做 nanoGPT 的经验

我做 nanoGPT 时就发现:base pretraining 学"language prior",SFT 学"task format",分工比一个胖网络万能化更有效。DexVLA 的三阶段本质就是这个思路搬到 VLA 上:
- Stage 1 = base pretraining (motor prior)
- Stage 2 = instruction tuning (alignment)
- Stage 3 = domain SFT (specialization)

### 5. Motor Chinchilla 还没出现

LLM 有 Chinchilla law告诉你算力和数据的最佳配比,VLA 还完全没有这东西。DexVLA 用 100 小时 + 1B Diffusion Expert 达到这个效果,说明 motor control 的 scaling law 可能比 language 更陡——action space 虽然 low-dim 但 high-precision,需要更多参数 encode continuous manifold。这背后可能藏着 motor control 的 Chinchilla-like law,值得深挖。

---

## 局限和我想看到的下一步

### Paper 自己承认的
- Zero-shot cross-embodiment 只在 1-DoF 子空间 work,全 DoF 灵巧手仍然失败
- 100 小时数据相对 Open-X (4000 小时) 还是小,没探讨 scale up 会怎样
- Sub-step reasoning 用 Gemini 2.0 + Grounding-DINO 自动标注,deformable object 状态细分仍然粗糙

### 我 (Karpathy) 想看到的
1. **Diffusion Expert 换成 Flow Matching / Consistency Model**:π₀ 用 flow matching 推理更快,DexVLA 用 DDPM 在 A6000 跑 60Hz,换 consistency distillation 可能 200Hz+
2. **Hierarchical Diffusion**:目前是 flat DiT,引入 hierarchical latent (类似 VQ-VAE + Diffusion) 可能更适合 long-horizon
3. **Diffusion Expert 作为独立 motor foundation model release**:Stage 1 训练完的 1B Diffusion Expert 本身就是 cross-embodiment motor prior,可以像 SigLIP 之于 VLM 那样让社区接各种 VLM
4. **Sub-step reasoning 的 scaling**:如果 sub-step 标注越多,模型 long-horizon 能力是否线性提升?
5. **Vision encoder 是否也该 fine-tune**:目前 frozen,实验显示 white shirt on white table 能 fold,但 extreme visual domain (比如水下) 可能需要 fine-tune

---

## 最后一段话

这篇 paper 给我 (Karpathy) 最大的启发是:**架构层面的 reframe 比堆参数更重要**。

大家都在喊"scale VLM",但 action representation 这块欠的债太久了。DexVLA 把 Diffusion Expert 做到 1B,用三阶段 curriculum 让它和 VLM 协同,用 sub-step reasoning 把 high-level planning 内化——每一步都很合理,加起来就是显著超过 π₀ 的效果。

这让我想起 2017 年 Transformer 那篇 paper:不是把 LSTM 堆到更大,而是换个架构。DexVLA 可能不是最终答案,但它指向了一个正确的方向——**VLA 时代的 action Chinchilla 还没出现,我们在 motor cortex 上欠的债,该还了**。

我尤其看好它 release 出 Stage 1 的 1B Diffusion Expert 作为 motor foundation model——这东西有可能像 CLIP 一样成为一个生态起点,让所有人都能接自己的 VLM 来做 robot。

---

## 主要 References

- DexVLA project: https://dex-vla.github.io/
- π₀ paper: https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://octo-models.github.io/
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ScaleDP: https://arxiv.org/abs/2409.14411
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- SayCan: https://say-can.github.io/
- TinyVLA: https://arxiv.org/abs/2409.12514
- Diffusion-VLA: https://arxiv.org/abs/2412.03293
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- LLaVA: https://arxiv.org/abs/2304.08485
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- HybridVLA: https://arxiv.org/abs/2503.10631
- Sparse Diffusion Policy: https://arxiv.org/abs/2407.01531
- Discrete Policy: https://arxiv.org/abs/2409.18707

---

# DexVLA 深度解读:让 Diffusion Expert 成为 VLA 的"小脑"

## 1. Paper 的核心 Thesis (一句话)

这篇 paper 的核心论点非常清晰:**当前 VLA 范式过度投资在 VLM backbone 的 scaling 上 (OpenVLA 7B, π₀ 3B),但 action representation 才是真正的 bottleneck**。DexVLA 的主张是把一个 **1B 参数的 Diffusion Transformer** 作为 plug-in "motor cortex" 接到 VLM 旁,并设计一个三阶段 curriculum 让这两个模块协同收敛。在 laundry folding、bimanual packing、dexterous pouring 这些任务上,用 **100 小时数据 + 单卡 A6000 60Hz**,干翻了 OpenVLA、Octo、Diffusion Policy,在长程任务上甚至超过 π₀。

这让我 (Karpathy) 联想到 LLM 早期大家都在 scaling transformer trunk,直到 RLHF 阶段才发现 reward model 和 value head 的设计才是关键;VLA 现在正经历类似的"action head renaissance"。

项目主页:https://dex-vla.github.io/

---

## 2. VLA 范式的瓶颈在哪里 (我的 intuition)

让我先把整个 VLA family tree 拉出来:

| Model | VLM backbone | Action head | 训练范式 |
|---|---|---|---|
| RT-1 / RT-2 | PaLI / PaLM-E | Tokenize 成离散 action bin | End-to-end next-token |
| OpenVLA | Llama-2 7B + DINO/SigLIP | Action token (256 bins × 7 DoF) | SFT on OXE |
| π₀ | PaliGemma 3B | Flow matching expert (~300M) | SFT + flow matching |
| Octo | ViT 93M | Diffusion head | Multi-task diffusion |
| TinyVLA | Phi-3 | Token action | LoRA + 对比学习 |
| **DexVLA** | Qwen2-VL 2B | **Diffusion Transformer 1B** | 三阶段 curriculum |

直觉层面我看到的几个核心问题:

**问题 1: Autoregressive token prediction 不适合 continuous control**。
离散化 action token (OpenVLA 把每个 DoF 切 256 个 bin) 引入 quantization error,而且 token-level cross-entropy loss 在 multimodal action distribution (一个 state 下多个合理 action) 上会"塌缩"到均值——这是 diffusion policy 早就指出的问题 (Chi et al. 2023, https://arxiv.org/abs/2303.04137)。

**问题 2: VLM 的 capacity 被"浪费"在 motor control 上**。
VLM 通过 internet-scale 数据学到的是 semantic grounding,而不是 motor primitives。强迫 VLM 同时承担语义理解和低层控制,会让两边都学不好——类似强迫 GPT-3 同时学 language modeling 和像素生成。

**问题 3: Cross-embodiment 训练时 action head 容易"互相打架"**。
不同 embodiment 的 DoF、kinematics、control frequency 完全不同。直接拼成一个 flat action vector 训练,会让不同 morphology 在共享参数空间中冲突,这正是 Octo 用 multi-head readout、DexVLA 用 multi-head diffusion expert 的原因。

DexVLA 的回答是:**让 VLM 做"what"和"why",让 Diffusion Expert 做"how"**。这和我做 nanoGPT 时的直觉一致——分工 (modularity) 比一个胖网络万能化更有效。

---

## 3. Architecture 详解

### 3.1 整体数据流

```
                  ┌──────────────────────────────────┐
                  │   Qwen2-VL 2B (frozen vis-enc)    │
   Image(s) ────► │                                  │
   Instruction ─► │  → reasoning tokens (L_ntp)      │ ──► Auto-regressive head
                  │  → action tokens                 │
                  └────────────┬─────────────────────┘
                                 │ projection (2×Linear + LayerNorm)
                                 ▼
                  ┌──────────────────────────────────┐
                  │   Diffusion Expert (1B, DiT)      │
                  │   FiLM(γ, β) ← reasoning tokens   │
                  │   Input: noisy action a_t, obs    │
                  │   Output: noise ε̂                 │
                  │   Multi-head: per-embodiment MLP  │
                  └──────────────────────────────────┘
                                 │
                                 ▼
                       Denoised action chunk {a_t..a_{t+H}}
```

### 3.2 Diffusion Expert 的设计

基于 ScaleDP (Zhu et al., https://arxiv.org/abs/2409.14411) 的 Transformer variant:
- 32 layers
- hidden dim = 1280
- 16 attention heads
- 总参数 ≈ 1B
- 输入: noisy action chunk $a^t_K$ + obs embedding (FiLM-conditioned)
- 输出: 预测的 noise $\hat{\epsilon}$
- Multi-head output: 每个 embodiment 一个独立 MLP readout (类似 Octo 的设计)

**为什么是 DiT 而不是 UNet?** Paper Table 3 给出消融:
- UNet (93M): 0.17
- DiT (410M): 0.63
- DiT (1B): 0.92

作者观察到 UNet 版本机器人手臂"抖动"——我 (Karpathy) 的解读是:UNet 的 inductive bias (局部卷积 + skip connection) 适合图像生成这种 spatially-structured 任务,而 robot action 是 temporal-sequential + cross-embodiment 的,Transformer 的全局 attention 更适合。同时参数过少会让不同任务在 weight space 里互相覆盖 (类似 catastrophic interference)。

### 3.3 训练目标公式

$$
\mathcal{L} = \mathcal{L}_{diff} + \alpha \, \mathcal{L}_{ntp}
$$

- $\mathcal{L}_{diff}$: 标准 DDPM noise prediction loss
$$
\mathcal{L}_{diff} = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\epsilon}, k}\Big[\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{a}_k, k, \mathbf{o}_t, \mathbf{c})\|^2\Big]
$$
  - $\mathbf{a}_0$: ground-truth action chunk
  - $\mathbf{a}_k = \sqrt{\bar{\alpha}_k}\mathbf{a}_0 + \sqrt{1-\bar{\alpha}_k}\boldsymbol{\epsilon}$: 加噪 action
  - $k \sim \mathcal{U}(0, K)$: diffusion timestep
  - $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: 采样噪声
  - $\mathbf{o}_t$: observation (image + proprioception)
  - $\mathbf{c}$: 来自 VLM reasoning tokens 通过 FiLM 注入的条件
  
- $\mathcal{L}_{ntp}$: VLM backbone 的 next-token prediction loss (language + reasoning tokens)
- $\alpha = 1$, 但作者观察到 $\mathcal{L}_{ntp}$ 早期就收敛,后期优化重心实际偏向 $\mathcal{L}_{diff}$。这其实是一种隐式的 curriculum,不用显式 schedule。

### 3.4 FiLM 注入 reasoning 的机制

$$
\text{FiLM}(\mathbf{x}) = \boldsymbol{\gamma}(\mathbf{r}) \odot \mathbf{x} + \boldsymbol{\beta}(\mathbf{r})
$$

- $\mathbf{r}$: VLM 输出的 reasoning token embedding
- $\boldsymbol{\gamma}, \boldsymbol{\beta}$: 通过 linear projection 从 $\mathbf{r}$ 生成,分别对 Diffusion Expert 内部 projection layer 的输出做 scale 和 shift
- $\odot$: 逐元素乘法

这个设计很优雅:reasoning 不是被 concatenate 到 action tokens 里 (那样会被 noise diffuse 掉),而是通过 FiLM 修改整个 diffusion 网络的激活分布,等价于让 reasoning **调制** action manifold 的形状,而不是 conditioning on 它。这和我做 GPT 训练时观察到 residual stream 的"direction modulation"非常像——embedding 做 linear combination,attention 做 routing,MoE 做 routing,FiLM 做 affine modulation,本质都是流形上的几何变换。

---

## 4. 三阶段 Curriculum 的深度解读

### Stage 1: Cross-Embodiment Pre-training

- **只训练** Diffusion Expert,完全不用 VLM
- Image encoder: random-init ResNet-50
- Language encoder: DistilBERT (off-the-shelf)
- 数据: cross-embodiment, 91 tasks, ~100h
- Loss: 纯 diffusion loss
- 学习率 1e-4, 5 epochs, AdamW (β1=0.9, β2=0.95)

**为什么 Stage 1 不用 VLM?** 这个设计非常聪明,我 (Karpathy) 看到的几个理由:

1. **Optimization stability**: 1B Diffusion Transformer 从零开始训练本身就不容易,如果同时 backprop 到 VLM,容易破坏 VLM 的 pre-trained representation。
2. **Compute efficiency**: Table 6 显示单独训练 Diffusion Expert 是 0.89 epoch/hr,训练整个 VLA 只有 0.32 epoch/hr——**2.78× 加速**。这意味着 Stage 1 的"cross-embodiment motor pre-training"是廉价的。
3. **Modular inductive bias**: VLM 已经在 internet data 上学会了 semantic understanding,不应该被低层 motor signal 干扰;Diffusion Expert 应该先建立自己的"motor prior",类似婴儿先学会 grasp reflex 再学语言指令。

这一阶段有点像我做 nanoGPT 时先做 base pretraining 再做 SFT:base 阶段学 "language prior",SFT 阶段学"task format"。

### Stage 2: Embodiment-Specific Alignment

- 接入 Qwen2-VL 2B (vision encoder frozen,LLM 部分 + projection + diffusion expert 联合训练)
- 数据: filter 成 single-embodiment
- 学习率 2e-5 (比 Stage 1 小 5×,保护 VLM 的 representation)
- 5 epochs

**类比 LLaVA 的 connector alignment** (Liu et al., https://arxiv.org/abs/2304.08485):LLaVA 也是先训练 projector 把 ViT feature 对齐到 LLM token space,再 instruction tuning。这里 Stage 2 做的是把 VLM 的 high-level representation 对齐到特定 embodiment 的 action manifold。

一个非常关键的实验现象:Stage 2 之后,模型已经能在 shirt folding 上拿到 0.92 分,bin-picking、table bussing 也有不错表现 (Figure 6)——**这意味着 VLM 的 semantic prior + Diffusion Expert 的 motor prior 相加,可以 zero-shot emerge 出复杂 skill**。这是 emergent capability 的好例子。

### Stage 3: Task-Specific Adaptation

- 类比 LLM 的 post-training / domain SFT
- 高质量 sub-step annotated 数据
- 学习率 2e-5,Cosine scheduler
- 5 epochs

**Sub-step reasoning 的设计** (这是 paper 最有意思的 trick):

把 "fold the shirt" 拆成 ["smooth wrinkles", "align sleeves", "secure folds"] 这种 sub-step sequence,并且训练模型**先生成** sub-step language,再用 sub-step reasoning 通过 FiLM 指导 action。

公式上:
- 训练时: $\mathbf{r} = \text{VLM}(\mathbf{o}_t, \text{substep}_t)$ 其中 $\text{substep}_t$ 是 ground-truth sub-instruction
- 推理时: $\mathbf{r} = \text{VLM}(\mathbf{o}_t, \text{"fold the shirt"})$ 模型自己 generate substep

Table 7 的 ablation 太关键了:
- Stage 1 + Stage 2 都用 direct prompt: 0 分
- Stage 1 用 direct, Stage 2 用 substep: 0.07
- 两个 stage 都用 substep: 0.92

我 (Karpathy) 的解读:sub-step reasoning 等价于**在 continuous action manifold 上引入 discrete anchor points**,把长程轨迹切成短段,每段对应一个 sub-manifold。这避免了不同 sub-task 的 gradient 在 shared parameter space互相冲突,本质是把 mixture-of-experts 的思想用"temporal segmentation"实现——和 Sparse Diffusion Policy (Wang et al., https://arxiv.org/abs/2407.01531)、Discrete Policy (Wu et al., https://arxiv.org/abs/2409.18707) 的思路遥相呼应。

Table 8 还有一个细节:DexVLA 的 implicit substep reasoning (0.70) > 外挂 SayCan (0.58)。
- SayCan 固定 2 秒更新一次,容易产生 redundant state 或 missing critical transition
- Implicit reasoning 是 state-adaptive 的,由 VLM 自己判断何时切换

这让我想起 LLM 的 chain-of-thought vs explicit planning agent 之争——内化 reasoning 总比外挂 planner 更省 token、更鲁棒。

---

## 5. 实验数据深度解读

### 5.1 主表 (Figure 6) — Without task-specific adaptation

任务:bin-picking easy, shirt folding, table bussing easy
- DexVLA 在 shirt folding 上 0.92,所有 baseline (OpenVLA, Octo, Diffusion Policy) 都接近 0
- 这意味着 shirt folding 是个 "litmus test" task——它需要 bimanual coordination + deformable object manipulation + precise folding,任何一环不行都做不到

### 5.2 New Embodiment (Figure 8) — Franka + dexterous hand, bimanual UR5e

只用 100 demonstrations fine-tune Stage 2 model:
- DexVLA 平均 0.90
- OpenVLA, Octo 几乎失败
- Diffusion Policy 从零训练也大幅落后

**这个结果让我非常震惊**。100 demos 在 dexterous hand pouring 上能学到 0.9,说明 Stage 1 的 cross-embodiment motor prior 已经学到了 "pouring" 的 abstract motor template,Stage 2 学到了 "language → action manifold" 的 alignment,Stage 3 只需要少量 task-specific 数据来 specialize。这本质上是一种**motor representation transfer**——类似 LLM 的 few-shot ICL,但发生在 action space。

### 5.3 Long-horizon (Figure 11) — Laundry folding, table bussing hard

- Laundry folding (>2 min, randomized crumpled initial state)
  - DexVLA: 0.4
  - π₀: 0.2 (用 SayCan)
  - OpenVLA, Octo: 0
- Dryer unloading: DexVLA 0.8, baselines 0

这是 paper 最强的 claim。π₀ 用了 10000 小时数据 + SayCan 的高层 policy,只用 100 小时 + implicit reasoning 的 DexVLA 反而更好。我的解读:
- π₀ 的 action expert 太小 (~300M),容量不足以 encode 复杂 long-horizon sub-manifold
- SayCan 的离散 subgoal 切换会丢失 fine-grained state context
- DexVLA 的 implicit reasoning 通过 FiLM 持续 modulate action manifold,信息流更平滑

### 5.4 Size ablation (Table 3)

UNet 93M: 0.17 (抖动)
DiT 410M: 0.63
DiT 1B: 0.92

参数 scaling 在 motor control 上同样有效!这让我想起 Kaplan 的 scaling laws——但 motor control 的 scaling 比 language 更陡,因为 action space 是 low-dim 但 high-precision,需要更多参数 encode continuous manifold。

### 5.5 LIBERO (Table 5)

| Method | Spatial | Object | Goal | Avg |
|---|---|---|---|---|
| DP | 78.3 | 92.5 | 68.3 | 79.7 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 84.1 |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 93.9 |
| π₀ | 96.8 | 98.8 | 95.8 | 97.1 |
| **DexVLA** | **97.2** | **99.1** | 95.6 | **97.3** |

LIBERO 是 simulation benchmark,主要测短程 task generalization。DexVLA 在 Spatial 和 Object 上甚至略胜 π₀,说明它的 motor representation 确实更 expressive。

### 5.6 Zero-shot Cross-Embodiment Transfer (Appendix A.5)

最让我 (Karpathy) 兴奋的实验:

- Stage 2 模型在 Franka + Robotiq gripper (1 DoF) 上训练
- 推理时 swap 成 Inspire dexterous hand,只控制 1 DoF (其他 5 DoF 锁定)
- 30 unseen objects,bin-picking,60% success
- 原 gripper: 67%

60% vs 67%——只掉 7 个点。这意味着:
1. **Visual representation 可以 transfer**:即使 dexterous hand 和 gripper 外观差异大,Stage 1 学到的 cross-embodiment visual prior 足以 generalize
2. **Camera view 可以 re-ground**:wrist camera 位置变了,模型仍能找到合理的 action
3. **Action manifold 在 1-DoF 子空间上对齐**:这是 multi-head design 的副作用,1-DoF 是两个 embodiment 共享的"最大公约数"

这给了我一个很强的 intuition:**embodiment-specific 的不是 action 维度本身,而是 DoF 之间的 correlation structure**。如果你 constrain 到一个 shared 1-DoF 子空间,跨 embodiment transfer 是可能的——这有点像 language model 的 universal grammar,只不过发生在 action space。

---

## 6. 与我 (Karpathy) 视角的关联

我 (Karpathy) 长期以来在 Tesla 自动驾驶 + Optimus 项目中思考的几个问题,DexVLA 给出了一些有意思的答案:

### 6.1 "Two-system brain" hypothesis

人脑的小脑 (motor control) 和大脑皮层 (semantic reasoning) 是分离但互联的:
- 大脑皮层:慢、abstract、symbolic,负责 "what to do"
- 小脑:快、continuous、习惯化,负责 "how to do"

DexVLA 的设计完美对应这个结构:
- Qwen2-VL 2B = 大脑皮层 (semantic grounding, sub-step reasoning)
- Diffusion Expert 1B = 小脑 (motor primitives, action chunk generation)
- FiLM = 皮层 → 小脑的神经调制通路 (cerebellar mossy fibers + climbing fibers)

这比 OpenVLA 把所有事情塞进一个 LLM trunk 要符合神经科学 intuition。

### 6.2 Curriculum = Human development

Stage 1:婴儿期的 motor babbling,学 general motor skill,无语言
Stage 2:学步期,把语言指令和 motor action 对齐
Stage 3:成年期,通过 explicit practice 学复杂技能

### 6.3 In-context reasoning vs implicit reasoning

π₀ 用 SayCan 这种 explicit planner,本质上是个 multi-agent system,有 latency 和 coordination cost。
DexVLA 让 VLM 内部 generate substep reasoning,这是 in-context reasoning 的延伸,更像 internal monologue。

这让我想起 ReAct vs CoT 的讨论——internal reasoning 总比 external tool call 更接近"智能"的本质。

---

## 7. 局限与未来方向

### 7.1 Paper 自己承认的局限
- Zero-shot cross-embodiment transfer 只在 1-DoF 子空间上 work,全 DoF dexterous 仍然失败
- Stage 1 仍然需要 cross-embodiment data,没有真正 zero-embodiment 的 pretraining
- 100 小时数据相对 Open-X (4000 小时) 还是小,但 paper 没探讨如果数据 scale up 会怎样

### 7.2 我 (Karpathy) 看到的开放问题
1. **Diffusion Expert 是否可以替换为 Flow Matching / Consistency Model?** π₀ 用 flow matching,推理更快。DexVLA 用 DDPM 60Hz 在 A6000 上,如果换 consistency distillation 可能跑到 200Hz+
2. **Multi-head 设计的扩展性**:91 个 embodiment 时 91 个 head 是否会让 last-layer bottleneck?
3. **Sub-step reasoning 的自动化**:目前用 Gemini 2.0 + Grounding-DINO 自动标注,这在 long-horizon task 上还行,但对 deformable object 状态细分仍然粗糙
4. **VLM 的 frozen vision encoder 是否限制了 visual generalization?** 实验显示 white shirt on white table 也能 fold,但如果面对从未见过的 extreme visual domain (e.g. underwater),可能需要 vision encoder 也 fine-tune
5. **Action chunk horizon H 的选择**:paper 没明确说 chunk size,这是个关键 hyperparameter——H 太小失去 smoothing,H 太大失去 reactivity

### 7.3 一些大胆的联想
- **Diffusion Expert 作为"action foundation model"**:Stage 1 单独训练完的 1B Diffusion Expert 本身就是个 cross-embodiment motor prior,可以 release 出来让社区用任何 VLM 接——类似 SigLIP 之于 VLM。
- **Sub-step reasoning 的 scaling law**:如果 sub-step 标注越多,模型 long-horizon 能力是否线性提升?这背后可能藏着 motor control 的 Chinchilla-like law。
- **Hierarchical Diffusion Expert**:目前是 flat DiT,如果引入 hierarchical latent (类似 VQ-VAE + Diffusion) 可能更适合 long-horizon。

---

## 8. 总结:为什么这篇 paper 重要

DexVLA 的核心贡献是 **architectural rebalancing**:把 motor control 的重要性重新提到和 semantic understanding 同一量级。这个 thesis 我 (Karpathy) 非常认同——在 autonomous driving 和 humanoid robot 上,大家都过度迷信"big VLM = good robot",忽略了 action representation 才是最后 10cm 的关键。

具体创新:
1. 1B DiT-based Diffusion Expert,multi-head,plug-in 设计
2. 三阶段 embodied curriculum:decoupled pretraining → alignment → task-specific
3. Sub-step reasoning 通过 FiLM 注入,内化 high-level planning

实验上用 100 小时数据 + 单卡训练 + 60Hz 推理,在多个 benchmark 上击败 OpenVLA、Octo、π₀,尤其是 long-horizon 任务和 zero-shot cross-embodiment transfer 上表现出乎意料地强。

这让我想起 2017 年的 Transformer——架构层面的 reframe 往往比单纯堆参数更重要。DexVLA 可能不是最终答案,但它指向了一个正确的方向:**VLA 时代的"action Chinchilla"还没有出现,我们在 motor cortex 上欠的债,该还了。**

---

## 主要 References

- DexVLA project: https://dex-vla.github.io/
- π₀ paper (Physical Intelligence): https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- Octo: https://octo-models.github.io/
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ScaleDP: https://arxiv.org/abs/2409.14411
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- SayCan: https://say-can.github.io/
- TinyVLA: https://arxiv.org/abs/2409.12514
- Diffusion-VLA: https://arxiv.org/abs/2412.03293
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- LLaVA: https://arxiv.org/abs/2304.08485
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- HybridVLA: https://arxiv.org/abs/2503.10631
- Sparse Diffusion Policy: https://arxiv.org/abs/2407.01531
- Discrete Policy: https://arxiv.org/abs/2409.18707
