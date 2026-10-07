---
source_pdf: Learn from Multimodal Parallel Generation to Act.pdf
paper_sha256: 2241254cc1f655500bb70f8d577cb245853f4a91d0dffea2d5765788e2621a08
processed_at: '2026-08-05T12:33:54-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MM-ACT

## 一句话版本

让 robot 同时学会三件事——**说话**（task planning）、**做梦**（想象未来画面）、**动手**（生成 action），而且这三件事用同一个 brain、同一种学习方式，互相还帮衬。

---

## 这 paper 到底在解决什么烦恼

想象你要训练一个 robot policy。最 naive 的做法是：给它看图、听指令，然后让它直接输出 action。这就是 OpenVLA、RT-2 这类的套路。

问题来了：**robot 不知道自己为什么要这样动**。

它就像一个只会照着菜谱机械翻炒的厨师，你问它"接下来该干嘛"，它答不上来；你问它"炒完之后锅里会变成什么样"，它也想象不出。它只会 from 手到 action 的 mapping。

于是有人想：那我们让它也学会 plan（CoT-VLA）和 predict future image（DreamVLA）呗。但现有做法有个别扭的地方——

**plan 用一种学习方式（next-token prediction），predict image 用另一种（diffusion），生成 action 又用第三种（flow matching 或又是 diffusion）**。相当于一个人脑子里有三套思维系统，切换来切换去，训练时也容易打架。具体说，backbone 是用 AR pretrain 的（预测下一个 token），fine-tune 时却要它做 denoising，这俩 objective 本质不同，模型很懵。

MM-ACT 的回答很简单粗暴：**全部统一成一种——mask prediction**。

---

## Mask Prediction 到底是啥直觉

你小时候做过那种"完形填空"吧？给你一句话，挖掉几个空，让你填回去。

```
"The robot picks up the <MASK> and places it on the <MASK>"
```

Mask prediction 就是这个游戏的 neural network 版本。给一段 token 序列，随机挖掉一些（替换成 `<mask>`），让模型猜被挖掉的是啥。

关键魔法在于：**挖掉的空可以一次性全部猜**，不需要从左到右一个一个来。因为用的是 bidirectional attention——每个 mask 位置都能看到所有其他位置（包括其他 mask）的 context。

这就是 discrete diffusion 的本质：
- $t=1$: 全部 token 都 mask（一张白纸）
- $t=0.5$: 一半 mask 一半保留
- $t=0$: 全部保留（恢复完整序列）

训练时随机采一个 $t$，按对应比例 mask，然后让模型预测被 mask 的那些 token。Loss 就是 cross-entropy。

对比 AR（autoregressive）：
- AR: 必须先输出第 1 个 token，才能看它来输出第 2 个，再输出第 3 个……串行
- Mask prediction: 一次性预测所有 mask 位置，**并行**

对 action chunk 来说这太重要了——你有一整个 chunk 的 action token 要输出，AR 得一个个来（慢），mask prediction 一次 forward 全搞定（快）。

参考 LLaDA: https://arxiv.org/abs/2502.09992

---

## 三个 modality 怎么塞进一个模型

### Tokenizer 层面

所有东西都变成 discrete token，拼成一个大 vocabulary：

```
[text vocab] + [image codebook 8192 个] + [action codebook 2048 个]
```

- **Text**: 用 LLaDA 的 BPE tokenizer，跟普通 LLM 一样
- **Image**: 256×256 图片 → 256 个 token（用 Show-o 的 image quantizer，本质是 VQ-GAN 类的离散化）
- **Action**: 每个 continuous scalar 归一化到 [-1,1]，然后分成 2048 个 bin 量化。比如 7-DoF action × chunk size 8 = 56 个 action token

### 序列怎么组织

一个 input 序列长这样：

```
<modal_token> + shared_context + [mask block]

例子:
<|mm2a|> [image tokens from camera 1] [image tokens from camera 2] 
        [instruction tokens] [robot state tokens] [MASK×56]

→ 模型看到 context, 一次性预测 56 个 action token
```

`<modal_token>` 就是路由开关：
- `<|mm2a|>`: 后面要生成 action
- `<|mmu|>`: 后面要生成 text（task planning）
- `<|t2i|>`: 后面要生成 image（future prediction）

**同一个 shared_context** 复用三次，只是 modal token 换一换，后面 mask block 换一换。

---

## Context-Shared Multimodal Learning：最核心的 trick

这是整篇 paper 的灵魂。

### 直觉解释

假设你在教一个小孩学做菜。两种教法：

**教法 A（action only）**：只让他看你的手怎么动，模仿动作。
**教法 B（context-shared）**：每次做菜前先让他口述"我接下来要切番茄、然后炒蛋、最后装盘"（text），再让他画一下"炒完之后盘子长啥样"（image），最后才让他动手（action）。

哪种教出来更靠谱？显然 B。因为 text 和 image 给 action 提供了 **语义锚点** 和 **后果预演**。

MM-ACT 就是教法 B 的 neural network 实现。

### 具体怎么做

训练时，对同一个 context（robot 当前看到的画面 + instruction），同时做三件事：

```
Forward 1: C_<|mmu|> + [text mask block]  → 预测 task planning text
Forward 2: C_<|t2i|> + [image mask block] → 预测 future image  
Forward 3: C_<|mm2a|> + [action mask block] → 预测 action chunk
```

三个 forward 的 loss 加起来一起 backprop：

$$
\mathcal{L} = \sum_{\text{modal}} \frac{\lambda_{\text{modal}}}{t} \cdot \text{CE}_{\text{masked tokens}}
$$

- $\lambda_{\text{modal}}$: 各 modality 的权重，控制谁说话算数
- $1/t$: 时间步 weighting，$t \to 0$ 时权重越大（强迫模型在 mask 很少时也能精确预测）
- CE 只算被 mask 位置的 cross-entropy

### 两阶段训练

```
Stage 1: 只练 text + image（λ_mm2a=0）
  → 让模型先学会 plan 和 imagine
  → 大约 500-800 steps

Stage 2: 三样一起练，action 为主（λ_mm2a=1, λ_text≈λ_image≈0.05-0.1）
  → action 是主角, text/image 当 auxiliary supervisor
  → 大约 27k steps
```

为什么分两阶段？因为 action generation 是最终目标，但如果一上来就三样一起练，模型容易顾此失彼。先让 text/image 收敛到一个 reasonable 的水平，再用它们当"老师"监督 action 学习。

---

## 实验结果说了啥

### LIBERO（in-domain）

| Model | Avg Success Rate |
|-------|-----------------|
| OpenVLA | 76.5% |
| π0 | 94.2% |
| DreamVLA | 92.6% |
| UniVLA | 95.5% |
| **MM-ACT** | **96.3%** |

LIBERO-Long（长 horizon 任务）上，加 text planning 后从 88% → 93%（+5%）。这很直觉——任务越长，越需要先 plan 再 execute。

### RoboTwin2.0（out-of-domain，关键测试）

| 训练配置 | Avg SR | 相对 baseline gain |
|---------|--------|-------------------|
| Action only | 43.13% | — |
| + Text | 46.50% | +3.37% |
| + Image | 48.75% | +5.62% |
| + Text & Image | **52.38%** | **+9.25%** |

这是 paper 最核心的 evidence：**context-shared learning 确实让 action 变强了**。而且 image 的贡献比 text 大（+5.62% vs +3.37%），说明 forward dynamics modeling（想象未来画面）对 action generation 的帮助更大。

OOD setting 下 MM-ACT 超过 π0（52.38% vs 48.13%），虽然 π0 用了大规模 robotic pretraining。这暗示 discrete diffusion 的 bidirectional attention 在 generalization 上有优势——跟 NLP 里 BERT vs GPT 的观察类似。

### Franka Real-World

| Model | Press Button | Stack Block | Sort V&F | Avg |
|-------|-------------|-------------|----------|-----|
| π0 | 75% | 70% | 65% | 70.0% |
| OpenVLA-OFT | 70% | 50% | 56% | 58.6% |
| MM-ACT | 80% | 70% | 66% | 72.0% |

real-world 只比 π0 好 2%，但没差的，关键是 inference 快很多。

---

## 一个反直觉发现

Table 5 显示：

| | Text Accuracy | Image PSNR |
|---|--------------|------------|
| Stage 1 | 81.5% | 12.08 |
| Stage 2 | 68.7% ↓ | 14.23 ↑ |

Text 在 Stage 2 变差了，image 反而变好了。为啥？

看 training loss curve（Figure 6）就懂了：
- Text loss 在 Stage 1 大约 100 步就掉到接近 0 → **过拟合**了，死记硬背训练集的 planning 模板
- Image loss 一直缓慢下降 → 还有学习空间，joint training 持续受益

intuition：text annotation 太模板化（就那几种 subtask 描述），模型很容易背下来；image 是高维连续的，学不完。所以 text 在 joint training 后泛化性下降，image 反而蹭 action 的光一起进步。

这给未来工作一个 hint：**text annotation 需要更多样化**，否则 cross-modal learning 的 text 那条腿会先瘸。

---

## Action decoding 的效率魔法

这是 MM-ACT 能做到 real-time 的关键。

### 两种 decoding 策略

**One-step parallel decoding**（action 用这个）：
- $t=1$, 全部 mask
- 一次 forward, 全部预测出来
- 0.22 秒 / chunk

**Re-mask parallel decoding**（text/image 用这个）：
- 多步迭代，每步预测一部分高置信度的 token，剩下低置信度的重新 mask
- 类似 discrete diffusion 的 iterative denoising
- 1.06 秒（6 步）

### Ablation 发现一个有意思的规律

| Chunk size | One-step SR | Re-mask SR | 差异 |
|-----------|-------------|------------|------|
| 8 (56 tokens) | 43.13% | 42.38% | -0.75% |
| 16 (112 tokens) | 43.75% | **56.75%** | +13.00% |

短 sequence → 一步搞定，迭代反而引入噪声
长 sequence → 一步太难，必须迭代 refine

这跟 image generation 必须 multi-step 的逻辑一致——**block size 越大，单步预测越难，越需要 iterative denoising**。

MM-ACT 选 chunk=8 + one-step，牺牲一点精度换 5× 速度，对 real-time control 来说是合理 trade-off。

---

## 跟其他 VLA 范式的本质区别

```
(a) Pure AR VLA (OpenVLA, RT-2)
    一切 next-token, 串行, 慢
    没有 explicit dynamics modeling

(b) Hybrid VLA (π0, DexVLA, WorldVLA)  
    text 用 AR, action/image 用 diffusion
    两个系统, objective 不一致
    pretrain(AR) vs finetune(diffusion) 打架

(c) MM-ACT (unified discrete diffusion)
    一切 mask prediction, 并行, 快
    单一 objective, 单一 attention mechanism
    text/image/action 互相监督
```

(c) 的优雅之处在于：**没有任何 modality-specific 的 architecture hack**。就是一个 transformer + 三套 tokenizer + 一个 modal token 路由。简单到让人怀疑是不是漏了什么。

---

## 我的 take

这篇 paper 最让我 buy in 的点：

1. **Discrete diffusion 作为 unifying framework 的优雅性**——text/image/action 三种 modality 用完全相同的 objective 训练，没有 KL divergence、flow matching ODE 这些复杂东西，就是 cross-entropy on masked tokens。

2. **Context-shared learning 的简单性**——不需要复杂的 distillation、不需要 auxiliary loss 的 fancy design，就是同一个 context 做三次 forward，loss 加起来。但效果实打实 +9.25%。

3. **Bidirectional attention 对 OOD generalization 的帮助**——这是 BERT vs GPT 的经典观察在 robotics 上的复现，很 make sense。

我有点 skepticism 的点：

1. **Action bin tokenizer 的精度上限**——2048 bins 看似够，但对需要亚毫米级精度的任务（插头、螺纹）可能不够。future work 可能需要 continuous action head hybrid。

2. **Image 只有 256×256**——对精细 manipulation 不够。MAGVIT-v2 的高分辨率 codebook 可能是下一步。

3. **Text 过拟合问题没根本解决**——Stage 2 accuracy 掉 13%，说明 current text annotation 太模板化。如果 task planning 能更 free-form、更 diverse，cross-modal learning 的上限会更高。

4. **Scale 没验证**——基于 MMaDA-8B，没试过 30B+。discrete diffusion 在大 scale 上的训练稳定性还是 open question。

总体看，这是 VLA 范式从 AR 向 discrete diffusion 迁移的一个 solid proof-of-concept。如果 LLaDA 路线在 scale 上能 match AR LLM（目前 8B 上已经接近 Llama-3-8B），那 unified diffusion VLA 很可能是下一代 VLA 的主流。

---

## 参考链接

- MM-ACT repo: https://github.com/HHYHRHY/MM-ACT
- LLaDA (base model 基础): https://arxiv.org/abs/2502.09992
- MMaDA (multimodal diffusion base): https://arxiv.org/abs/2505.15809
- Show-o (image tokenizer 来源): https://arxiv.org/abs/2408.12528
- D3PM (discrete diffusion 理论基础): https://arxiv.org/abs/2107.03453
- SEDD (score entropy discrete diffusion): https://arxiv.org/abs/2310.16834
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- DreamVLA: https://arxiv.org/abs/2507.04447
- WorldVLA: https://arxiv.org/abs/2506.21539
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- MAGVIT-v2 (image tokenizer): https://arxiv.org/abs/2310.05737

---

# MM-ACT 深度解析：从 Discrete Diffusion 视角重构建 VLA 的统一范式

Andrej，这篇 paper 我反复读了几遍，它实际上在做一件很有意思的事情——把 robotics VLA 从 autoregressive (AR) 范式推向 **discrete diffusion / masked prediction** 范式，并通过 "Context-Shared Multimodal Learning" 让 text、image、action 三个 modality 在同一 context 下互相监督。下面我从 intuition 层面逐层展开。

---

## 1. 问题动机：为什么 AR VLA 走到瓶颈？

当前 VLA 主流路径有三条，我画一张 taxonomy 图帮你 build intuition：

```
VLA 范式分类
├── (a) Pure AR (OpenVLA, RT-2, UP-VLA)
│     text/image/action 全部 next-token prediction
│     问题: action chunk 必须串行 decode, latency 高
│
├── (b) Hybrid AR + Diffusion (DexVLA, π0, Show-o style)
│     text 用 AR, image/action 用 diffusion head
│     问题: pretraining (token pred) 与 finetuning (denoising) objective misalignment
│     架构复杂: 需要 dual attention / dual system
│
└── (c) Unified Discrete Diffusion (MM-ACT, MMaDA, LLaDA)
      全部 modality 用 mask token prediction + bidirectional attention
      → 训练目标统一, 架构单一, action 可 one-step 并行 decode
```

MM-ACT 选择 (c) 的根本 insight 是：**discrete diffusion 的 mask prediction 框架天然适合 block-level parallel decoding**，而 action chunk 本质是一个 block（d_action × N_chunk 个 token），可以一次性 forward 出来。这避免了 π0 那种 flow-matching action expert 与 VLM backbone 之间的 latent space 桥接问题。

参考链接：
- LLaDA: https://arxiv.org/abs/2502.09992
- MMaDA: https://arxiv.org/abs/2505.15809
- Show-o: https://arxiv.org/abs/2408.12528
- π0: https://arxiv.org/abs/2410.24164

---

## 2. 架构设计细节：单一 Transformer，三种 tokenizer

### 2.1 Tokenizer 设计

```
┌─────────────────────────────────────────────────┐
│  Unified Vocabulary (concatenated)              │
├─────────────────────────────────────────────────┤
│  Text tokenizer (LLaDA, ~65k vocab)             │
│  Image codebook (Show-o quantizer, 8192 tokens) │
│  Action codebook (bin tokenizer, 2048 tokens)   │
└─────────────────────────────────────────────────┘
```

- **Image**: 256×256 → 256 discrete tokens (类似 VQ-GAN/MAGVIT-v2 的 spatial tokenization)
- **Action**: 每个 continuous scalar 归一化到 [-1, 1]，再用 bin tokenizer 量化。一个 action chunk = `d_action × N_chunk_size` 个 token。对于 Franka 7-DoF delta pose + gripper (假设 d_action=7), chunk=8 → 56 个 action token
- **Text**: 直接复用 LLaDA 的 BPE tokenizer

这里有个 subtle 的设计选择——action codebook 拼接到 vocabulary 末尾，**不影响** text/image tokenizer 的原有 token ID。这点对 continue pretraining 很关键，因为 backbone (MMaDA-8B) 的 text/image embedding table 不需要重新初始化。

### 2.2 Modal Token 控制流

模型通过 modal token `<|mm2a|>`, `<|mmu|>`, `<|t2i|>` 来路由生成目标：

```
C_modal = <modal_token> + shared_input + [mask_block]

<|mm2a|> → action generation (one-step PD)
<|mmu|>  → text generation (task planning, re-mask PD)  
<|t2i|>  → image generation (future image prediction, re-mask PD)
```

shared_input 是 interleaved 的多视角 image + instruction + (optional) robot state。**同一个 context** 复用三次，只是 modal token 不同 + 后面接的 mask block 不同。这是 "Context-Shared" 的核心。

---

## 3. 数学公式逐项拆解

### 3.1 Masking 过程 (Eq. 1, 2)

$$
q_t(x_t^i \mid f_{\text{modal}}(t), x_0^i) = (1 - f_{\text{modal}}(t)) \cdot \mathbf{1}\{x_t^i = x_0^i\} + f_{\text{modal}}(t) \cdot \mathbf{1}\{x_t^i = \langle\text{mask}\rangle\}
$$

变量含义：
- $t \in (0, 1]$: continuous time step，类似 diffusion 的 timestep
- $f_{\text{modal}}(t)$: mask schedule function，控制 t 时刻被 mask 的概率
- $x_0^i$: ground truth 第 i 个 token
- $x_t^i$: t 时刻第 i 个 token 的状态
- $\mathbf{1}\{\cdot\}$: indicator function

直觉：以概率 $f_{\text{modal}}(t)$ 把 token 替换成 `<mask>`，以概率 $1-f_{\text{modal}}(t)$ 保留原 token。当 $t=1$ 时全部 mask，$t=0$ 时全部保留。

**Schedule 差异**（这是很关键的设计）：

| Modality | Schedule | Formula | 直觉 |
|----------|----------|---------|------|
| Text | Linear | $f(t) = t$ | 均匀 unmask，与 LLaDA 对齐 |
| Image | Cosine | $f(t) = \cos(\frac{\pi}{2}(1-t))$ | 早期慢后期快，匹配 continuous denoising |
| Action | 固定 $t=1$ | 全 mask → one-step decode | 强制 fully parallel，无迭代 |

为什么 action 用 $t=1$？因为 action chunk token 数少（56 个），单步 forward 就能 well-predict；而 image 有 256 token，必须迭代 refine。

### 3.2 Unified Loss (Eq. 3)

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t}\left[\sum_{\text{modal} \in \mathcal{M}} \frac{\lambda_{\text{modal}}}{t} \sum_{i \in \mathcal{T}_{\text{modal}}} \mathbf{1}\{x_t^i = M\} \cdot \log p_\theta(x_0^i \mid C_{\text{modal}}, x_t)\right]
$$

变量拆解：
- $\mathcal{M} = \{\langle\text{mm2a}\rangle, \langle\text{mmu}\rangle, \langle\text{t2i}\rangle\}$: 三种 modal token
- $\lambda_{\text{modal}}$: 各 modality loss 权重（Stage 1 时 $\lambda_{\text{mm2a}}=0$，Stage 2 时 $\lambda_{\text{mmu}}, \lambda_{\text{t2i}} \approx 0.05\sim0.1$）
- $\mathcal{T}_{\text{modal}}$: 该 modality 对应的 token position 集合
- $1/t$: 这个 weighting 很有意思，参考了 continuous diffusion ELBO 的形式。当 $t \to 0$ 时 loss 权重趋于无穷，强迫模型在 mask 极少时也能精确预测剩余 mask

**关键 insight**: 三个 modality 共享同一套 cross-entropy on masked tokens 的目标，没有任何额外的 diffusion KL divergence、flow matching objective 之类的东西。这是真正的 "unified objective"。

参考 discrete diffusion ELBO 推导：
- absorbing state diffusion: https://arxiv.org/abs/2107.03453 (D3PM)
- LLaDA 的 ELBO: https://arxiv.org/abs/2502.09992

---

## 4. Context-Shared Multimodal Learning：训练范式的核心创新

### 4.1 两阶段策略

```
Stage 1: λ_mm2a = 0
  └─ 只训练 text + image generation
  └─ 让 backbone 学会 task planning 和 future image prediction
  └─ 约 500-800 steps

Stage 2: λ_mm2a = 1, λ_mmu ≈ λ_t2i ≈ 0.05-0.1
  └─ 三 modality 联合训练, action 为主
  └─ text/image loss 权重低, 起到 auxiliary supervision 作用
  └─ 约 27k steps (RoboTwin)
```

### 4.2 为什么 cross-modal learning 能 boost action？

这是 paper 最有意思的发现。Table 2 显示：

| 训练配置 | RoboTwin Avg SR | Gain |
|---------|----------------|------|
| Action only (Vanilla) | 43.13% | baseline |
| + Text (task planning) | 46.50% | +3.37% |
| + Image (future pred) | 48.75% | +5.62% |
| + Text & Image | **52.38%** | **+9.25%** |

intuition 解释（paper 没明说，但我推测）：

1. **Image generation 强制学习 forward dynamics**：要预测 action 执行后的 subgoal image，模型必须隐式建模 "这个 action 会把环境变成什么样"。这等价于学了一个 world model prior，而这个 prior 反过来 regularize action generation——bad action 会导致 inconsistent future image。

2. **Text generation 强制学习 task decomposition**：长 horizon 任务里，先 plan 出 subtask sequence，再 execute action。这给 action generation 提供了 high-level guidance signal，类似 Chain-of-Thought。

3. **Shared context 的信息瓶颈效应**：三个 head 共享同一套 encoder representation，representation 必须同时编码 "what to do" (text)、"what will happen" (image)、"how to move" (action)。这种 multi-task regularization 类似 multi-task learning 的共享 backbone 收益。

### 4.3 反直觉发现：Text 在 Stage 2 退化，Image 持续提升

Table 5 显示 text accuracy 从 Stage 1 的 81.5% 降到 Stage 2 的 68.7%，但 image PSNR 从 12.08 升到 14.23。

原因（paper Figure 6 给出 loss curve）：
- Text loss 在 Stage 1 约 100 steps 内就趋于 0 → **过拟合**到训练集模板，泛化性下降
- Image loss 一直缓慢下降 → 还有学习空间，joint training 持续受益

这暗示一个未来方向：text annotation 的多样性不足时，joint training 反而会 hurt text generation。需要更丰富的 task planning 数据来缓解。

---

## 5. Parallel Decoding 策略的 effectiveness-efficiency trade-off

### 5.1 Action Decoding Ablation (Table 6)

| 策略 | Chunk Size | Steps | SR | Time |
|-----|-----------|-------|-----|------|
| one-step PD | 8 | 1 | 43.13% | 0.22s |
| re-mask PD | 8 | 6 | 42.38% | 1.06s |
| one-step PD | 16 | 1 | 43.75% | 0.23s |
| re-mask PD | 16 | 6 | **56.75%** | 1.06s |

**关键 insight**: chunk size=8 时 re-mask 没收益（甚至 -0.75%），但 chunk=16 时 re-mask 暴涨 +13%。这告诉我们：

- 短 sequence (56 token) → one-step 足够，迭代 refine 反而引入 noise accumulation
- 长 sequence (112 token) → 一次性预测太困难，需要 iterative denoising 来逐步约束

这跟 image generation 里 256 token 必须 multi-step 的逻辑一致。本质是 **block size 与 model capacity 的匹配问题**。

### 5.2 推理速度

one-step PD → 0.22s/chunk, chunk=8 → 实际控制频率 40Hz chunk-level / 5Hz per-action。这对 real-time control 来说很合格了。

对比：
- OpenVLA: 7Hz (AR token-by-token)
- π0: ~5Hz action expert + flow matching iterations
- MM-ACT: 5Hz per action, 40Hz per chunk

参考: FAST tokenizer https://arxiv.org/abs/2501.09747 也是为了加速 action tokenization，但走的是 compression 路线而非 parallel decoding。

---

## 6. 实验结果深度对比

### 6.1 LIBERO (Table 1)

```
Model              Spatial  Object  Goal   Long   Avg
─────────────────────────────────────────────────────────
OpenVLA             84.7    88.4   79.2   53.7   76.5
π0                  96.8    98.8   95.8   85.2   94.2
OpenVLA-OFT         96.2    98.3   96.2   90.7   95.4
UniVLA              95.4    98.8   93.6   94.0   95.5
DreamVLA            97.5    94.0   89.5   89.5   92.6
─────────────────────────────────────────────────────────
MM-ACT (Vanilla)    97.8    99.4   94.8   88.0   95.0
MM-ACT (+Text Long)                93.0   96.3
```

注意 Long 任务从 88.0 → 93.0 (+5%)，说明 **task planning 的收益在 long-horizon 上最显著**。这跟人类做长任务需要先 plan 再 execute 的直觉一致。

### 6.2 RoboTwin2.0 (Table 2) — out-of-domain 关键测试

RoboTwin2.0 用 domain randomization 训练，unseen scene/object 评估。这是真正测试 generalization 的 setting。

- π0: 48.13% (用大 scale robotic pretraining)
- OpenVLA-OFT: 23.13% (在 OOD 上崩得很惨)
- MM-ACT (Vanilla): 43.13%
- MM-ACT (+Text&Image): **52.38%**

虽然 MM-ACT 没用 Open-X 等大规模 robotic pretraining，但靠 unified diffusion backbone + cross-modal learning 还是超过了 π0。这暗示 **discrete diffusion 的 bidirectional attention 在 OOD generalization 上比 AR 更鲁棒**——这跟 BERT vs GPT 在 NLP 上的观察一致。

### 6.3 Franka Real-World (Table 3)

```
Press Button: MM-ACT 80% > π0 75% > OpenVLA-OFT 70%
Stack Block:  MM-ACT 70% = π0 70% > OpenVLA-OFT 50%
Sort V&F:     MM-ACT 66% > π0 65% > OpenVLA-OFT 56%
```

real-world 提升 +2% 相对 π0，主要来自 image generation 带来的 dynamics modeling。

---

## 7. 与相关工作的深层对比

### 7.1 vs UniVLA / WorldVLA (Hybrid AR-Diffusion VLA)

| 维度 | UniVLA/WorldVLA | MM-ACT |
|-----|-----------------|--------|
| Text generation | AR (next-token) | Mask prediction (parallel) |
| Image generation | Re-mask diffusion | Re-mask diffusion |
| Action generation | Re-mask diffusion | **One-step PD** |
| Attention | Hybrid (causal + bidirectional) | Pure bidirectional |
| 训练目标 | AR CE + diffusion ELBO | Unified CE on masked tokens |

MM-ACT 的优势：单一 attention 机制，单一 loss 形式，pipeline 简单。

### 7.2 vs DreamVLA (Visual Prediction VLA)

DreamVLA 也做 future image prediction + action，但用 AR backbone。MM-ACT 在 LIBERO 上超过 DreamVLA +3.7%，主要来自 discrete diffusion 的并行 decode 能力 + 更好的 OOD generalization。

### 7.3 vs π0 (Flow Matching VLA)

π0 用 flow matching 在 continuous action space 上做 diffusion，需要 separate action expert (dual system)。MM-ACT 直接 discretize action 到 token，融入 unified vocabulary，**single transformer** 搞定。代价是 action 精度受 bin tokenizer 量化误差限制（2048 bins，每个 scalar ~0.001 精度，对 7-DoF pose 来说够用）。

---

## 8. 关键 Limitations 与未来方向

1. **Text 过拟合问题**：Stage 2 后 text accuracy 下降 13%。需要更 diverse 的 task planning annotation，或者用 RLHF-style 的 text refinement。
2. **Image 256 token 限制**：分辨率只有 256×256，对精细 manipulation（如插头插入）不够。可以考虑 MAGVIT-v2 的更高 codebook 或 multi-scale tokenizer。
3. **Action bin tokenizer 精度**：2048 bins 对高频控制可能不够，未来可能需要 continuous diffusion head hybrid。
4. **Re-mask vs One-step 的自动选择**：当前 chunk=8 用 one-step，chunk=16 用 re-mask，是手动选择的。可以设计 adaptive schedule 根据 confidence 自动决定迭代次数。
5. **Scale-up**：目前基于 MMaDA-8B，未来 scaling 到 30B+ 时 discrete diffusion 的训练稳定性需要验证（LLaDA 已经在 8B 上证明可行）。

---

## 9. 个人 Take-aways（build your intuition）

如果让我用一句话总结 MM-ACT 的核心贡献：

> **把 VLA 从 "VLM + action head" 的 dual-system 范式，迁移到 "unified discrete diffusion" 的 single-system 范式，并通过 context-shared multi-modal supervision 让 world model prior 和 task planner 自然成为 action generation 的 auxiliary signal。**

这个范式的深层意义在于：
- **Discrete diffusion 是 NLP + CV + Robotics 的真正 unifying framework**——三个 modality 都用 mask prediction，没有 modality-specific architecture hack
- **Cross-modal learning 不需要复杂的 distillation**，只需要 shared context + joint loss weighting
- **Parallel decoding 的 efficiency 来自 architecture 而非 engineering**——bidirectional attention 天然支持 block-level prediction

从更宏的视角看，这是 LLaDA/MMaDA 路线在 embodied AI 上的延伸。如果 discrete diffusion 在 scale 上能 match AR（目前 LLaDA-8B 已经接近 Llama-3-8B），那 unified diffusion VLA 很可能成为下一代 VLA 的主流范式。

---

## 参考链接汇总

- MM-ACT repo: https://github.com/HHYHRHY/MM-ACT
- LLaDA paper: https://arxiv.org/abs/2502.09992
- MMaDA paper: https://arxiv.org/abs/2505.15809
- Show-o: https://arxiv.org/abs/2408.12528
- D3PM (discrete diffusion foundation): https://arxiv.org/abs/2107.03453
- SEDD (score entropy discrete diffusion): https://arxiv.org/abs/2310.16834
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- π0: https://arxiv.org/abs/2410.24164
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- DreamVLA: https://arxiv.org/abs/2507.04447
- UniVLA: https://arxiv.org/abs/2506.21539 (注：实际为 WorldVLA)
- WorldVLA: https://arxiv.org/abs/2506.21539
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- MAGVIT-v2: https://arxiv.org/abs/2310.05737
- Bin tokenizer (RT-1): https://robotics-transformer.github.io/

如果你想更深入聊某个部分（比如 discrete diffusion ELBO 的推导、MMaDA backbone 的具体架构、或者 action tokenizer 的 quantization error 分析），可以继续追问。
