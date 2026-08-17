---
source_pdf: ENERVERSE.pdf
paper_sha256: 5e4fc79197ea60b1401fe2cc4734dcdbad39cf7a6fbc207fbff10888f40604bf
processed_at: '2026-08-04T04:34:17-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ENERVERSE：用人话说说这篇 paper 在干嘛

---

## 一句话总结

**让机器人先在脑子里"放电影"想象未来会发生什么，然后再根据这个想象出来的画面去执行动作。**

就这么简单。其他所有技术细节都是为了让这个"放电影"的过程更靠谱、更 3D、更长久。

---

## 这事儿为什么难？

你想让机器人去抓个东西。传统做法是：给它一堆人类演示的数据，让它直接学习 "看到这个画面，就做这个动作"。

但问题来了：
- 机器人看到的画面只是 2D 的，它不知道这个东西在 3D 空间里到底在哪
- 任务可能很长，要抓完一个再抓下一个，中间的过程怎么记住？
- 真实世界的数据太贵了，simulator 的数据又跟真实世界有 gap

所以核心矛盾就是：**机器人需要一个"脑内小剧场"来理解 3D 世界和未来会发生什么，但我们手头只有一堆 2D 视频和有限的动作数据。**

---

## ENERVERSE 的核心思路

作者的 insight 特别漂亮：**video generation model 本质上就是一个 "脑内小剧场"**。

你看，DynamiCrafter 这种模型，给它一张图和一个 prompt，它能生成接下来几秒的视频。这说明它已经学会了"世界怎么运转"的 prior。那能不能把这个 prior 拿来给机器人用？

**能，但直接用不行。**

为什么不行？因为 general video model 学的是"人类觉得好看的视频"，它的 latent space 里 encode 的是像素层面的 continuity，没有 encode "这个物体在 3D 空间里是怎么动的"、"这个动作会导致什么物理后果"。

所以 ENERVERSE 做了三件事来把这个 video model 改造成能用的 robotics foundation model：

---

## 第一件事：Chunk-wise Autoregressive Generation —— 怎么"放电影"

### 问题

你想让机器人想象未来，但未来可能是 100 帧甚至无限长的。你不可能一次性生成所有帧。

传统 video model 的问题：
- 一次生成太长会 collapse（质量崩溃）
- autoregressive 一帧一帧生成又太慢，而且误差累积

### ENERVERSE 的解法

**把未来切成 chunk，每次生成一个 chunk，像 GPT 生成 token 一样。**

具体来说，chunk size = 8（实验发现 8 最 sweet）。模型每次生成 8 帧，然后这 8 帧变成下一轮的 context，再生成下 8 帧，以此类推。

### 公式层面发生了什么

训练时就是一个标准 diffusion 的 loss：

$$\min_\theta \mathbb{E}_{t, \mathbf{z} \sim z_{data}, \epsilon \sim \mathcal{N}(0, I)} ||\epsilon - \epsilon_\theta(\mathbf{z}_t^{1:M}, \mathbf{o}_t^{1:K}, t)||_2^2$$

翻译成人话：
- $\mathbf{z}_t^{1:M}$：要预测的 8 帧（noisy version）
- $\mathbf{o}_t^{1:K}$：之前看到的 context frames
- $\epsilon$：真实加的 noise
- $\epsilon_\theta$：模型预测的 noise
- $t$：diffusion 的 timestep

模型学着从 noisy 的 8 帧里把 noise 预测出来，减掉 noise 就得到 clean 的未来 8 帧。

**用 v-prediction 而不是 ε-prediction**：

$$v_t = \alpha_t \epsilon_t - \sigma_t x_0$$

- $\alpha_t = \sqrt{\bar{\alpha}_t}$：当前 signal 的强度
- $\sigma_t = \sqrt{1 - \alpha_t^2}$：当前 noise 的强度
- $x_0$：clean data
- $\epsilon_t$：注入的 noise

为啥用 v-prediction？因为当 noise 很大的时候，直接预测 $\epsilon$ 数值会爆炸，预测 $v$ 更稳定。这个 trick 来自 Salimans & Ho 的 progressive distillation paper。

### 怎么知道任务结束了？

这招很妙。模型训练时见过一个特殊的 EOS frame（End-of-Sequence）。推理时，每一轮生成的 frame 都跟这个 EOS frame 算 L1 distance，distance 低于 threshold 就停。

**模型自己学会了"这个任务做完了"的信号。** 不需要外部的 task completion signal。

---

## 第二件事：Sparse Memory —— 怎么记住过去

### 问题

chunk-wise autoregressive 听起来很好，但有个隐患：每次生成新 chunk 时，你用什么作为 context？

最直觉的做法：用前面连续的几帧。但作者发现这样做模型会在 OOD（out-of-distribution）场景下直接 collapse —— Figure 7 里展示得很清楚，生成的画面越来越糊，最后完全坏掉。

### 为什么会 collapse？

因为连续帧之间太相似了。模型学会了"复制粘贴"前一帧这个 shortcut，一旦遇到没见过的情况，这个 shortcut 失效，模型就崩了。

### ENERVERSE 的解法

**训练时随机丢掉 80% 的 frames，只用 20% 作为 context。**

这听起来很反直觉 —— 给更少的信息反而更好？是的，因为：
1. 模型没法再靠"复制粘贴"作弊了，它必须真的理解"任务进展到哪了"
2. 80% 的 dropout 强制模型学习 task structure 而非 frame correlation
3. 推理时即使有些 frame 质量不好，模型也很 robust

**消融数据很惊人：**
- LIBERO-Long 任务，没有 sparse memory：30.8 分
- 有 sparse memory：73.0 分
- **差了 42 分！** 这是 single design choice 带来的最大提升。

### 推理时怎么用？

用 sliding window。最近几帧 + sparsely sampled 的历史帧作为 context。同时给 memory context 加上 cosine decay 的 noise（越远的 frame noise 越大），这个 idea 来自 DeepMind 的 Genie paper。

---

## 第三件事：Multi-view Diffusion —— 怎么理解 3D

### 问题

单视角视频天生缺 3D 信息。一个杯子在画面里看起来是这样的，但它到底有多深？被遮挡的部分长什么样？单视角根本回答不了。

最直觉的解法：装多个 camera。但
- 硬件贵
- I/O 带宽要求高
- 系统复杂度高
- 真实世界多相机标定很麻烦

### ENERVERSE 的解法

**Pretraining 时用多视角数据学 3D prior，inference 时只用单 camera + depth warping 渲染出 auxiliary views。**

这是整个 paper 最聪明的 design 之一 —— asymmetric pretraining-inference。

#### 怎么 encode camera 信息？

用 **ray direction map**。每个 pixel 对应一条从 camera center 射出的 ray，这条 ray 的方向 encode 了 camera 的 intrinsics 和 extrinsics。

数学上：
- pixel $(u, v)$ 在 camera 坐标系的 ray direction：$d = K^{-1}[u, v, 1]^T$
  - $K$ 是 intrinsics matrix
  - $[u, v, 1]$ 是 pixel homogeneous coordinate
- 转到 world 坐标系：$d_{world} = R \cdot d$
  - $R$ 是 extrinsics 的 rotation part

这个 ray map（6 channels）跟 image latent（4 channels）channel-wise concat，变成 10 channels 输入 backbone。

#### 怎么做 cross-view attention？

输入 shape 是 $BCVTHW$（Batch × Channel × View × Time × Height × Width）。

三种 attention 通过 reshape 实现：
- **Spatial attention**: reshape 成 $(BT)(VHW)C$ —— 同一 frame 内部 pixel 交互
- **Temporal attention**: reshape 成 $(BVHW)TC$ —— 同一 view 跨时间交互
- **Cross-view**: 在 spatial attention 时，同一 spatial location 跨 view 自然交互

关键 insight：**reshape 不破坏 pixel-to-ray 的对应关系**，所以 geometric information 保留。

#### 最关键的实验结果

| Model | Multi-view Pretrain | Test Input | Success Rate |
|---|---|---|---|
| DynamiCrafter + DP | No | S-RGB | 79.0 |
| ENERVERSE-A | Yes | S-RGB | 92.1 |
| ENERVERSE-A | Yes | S-RGB + 1 render | 93.0 |
| ENERVERSE-A | Yes | S-RGB + 2 render | 88.5 avg, Object 97.7 |

**注意第二行**：test 时只用 single RGB，但因为 pretraining 时学了 multi-view，success rate 从 79 跳到 92。这意味着 3D understanding 被 implicit bake into model weight 里了，不需要 test-time 多 camera 也能 benefit。

---

## 第四件事：ENERVERSE-D —— 怎么造数据

### 问题

你想 pretrain multi-view model，需要大量多视角 + 标定好的数据。真实世界采集太贵。Simulator 数据有 sim2real gap。

### ENERVERSE 的解法

**做一个 data flywheel：generative model + 4DGS 互相 refinement。**

流程：
1. 真实世界 sparse observation（比如一个 camera 的完整视频）
2. Generative model 生成其他视角的视频
3. 用 observed + generated 的多视角视频做 4DGS 重建
4. 4DGS 重建出来的 4D scene 再 render 成多视角图片
5. 这些 render 图片质量更高、几何更一致
6. 再把这些 render 图片 re-noise 后 feed 回 generative model 做 refinement
7. 循环往复

### 为什么 4DGS 在这里有价值？

Generative model 容易 hallucinate 出几何不一致的东西。4DGS 用 explicit Gaussian representation 强制几何一致。两者互补：
- Generative model 提供 texture 和 detail
- 4DGS 提供 geometric constraint

**量化结果**：在 self-occlusion 严重的 "arrange workpieces" 任务上，4DGS 减少了 40% 的 hallucination。

---

## 第五件事：ENERVERSE-A —— 怎么从"放电影"变成"做动作"

### 核心 trick

Policy head 不需要从头跑 video diffusion。它 **reuse 第一个 denoising step 的 feature**。

具体来说：
1. 第一个 denoising step（最 noisy 的 $k=K$）时，image 经过 UNet 到 middle block
2. 从 middle block 提取 feature $E$，shape 是 $BCVTHW$
3. 在 spatial 维度做 mean pooling，得到 $T \times C'$ 的 feature vector
4. 这个 $E$ 被 cache 住，后续所有 action denoising step 都用它
5. Policy head（18 个 DiT block + linear layer）用 $E$ 作为 condition，做 DDPM denoising 输出 action chunk

### 公式

$$a_{t:t+\tau-1}^0 \gets f_\theta(c, o_t, a_{t:t+\tau-1}^k, k) = h_\theta(E, a_{t:t+\tau-1}^k, k)$$

变量解释：
- $a_{t:t+\tau-1}^0$：clean action chunk（8 步动作，每步 7 维：xyz delta position + roll pitch yaw + gripper）
- $a_{t:t+\tau-1}^k$：noisy action at diffusion step $k$
- $k$：diffusion step index
- $c$：language prompt
- $E$：cached visual latent（第一个 denoising step 提取）
- $h_\theta$：policy head = DiT blocks + linear projection

### 为什么用 first denoising step 的 feature？

因为 first step 输入最 noisy，模型必须 extract 最 robust、最 task-relevant 的 feature。这跟人类理解图像的方式一致：你先看出"这是个杯子"，然后才注意到"杯子上有花纹"。Action prediction 需要 "这是个杯子" 这个 level 的理解，不需要 "花纹" 这个 level 的 detail。

### 效率

单张 RTX 4090：
- 280 ms 生成 8 步 action chunk
- 10.6 GB GPU memory

这个效率在 robotics 领域是可以用的。

---

## 训练策略的消融实验特别 informative

| Strategy | LIBERO-Spatial |
|---|---|
| All-Scratch（从零训） | Failed（不收敛） |
| With DC Pretrain（用 general video 预训练初始化） | 79.0 |
| One-Stage Co-Train（video + action 同时训） | 86.3 |
| **Two-Stage Finetune（先 video 后 action）** | **92.1** |

**关键 insight**：
1. 从零训完全不收敛 —— 证明 generative prior 是必要的
2. 两阶段比 co-train 好 —— 因为 video loss 和 action loss 的 gradient 方向可能 conflict，分阶段让 representation 先固化再学 action mapping 更稳定

---

## Attention Map 告诉我们什么

Appendix D 的 Figure 11 特别有意思。他们画了 policy head 里 cross-attention 的 map：

- Y 轴：action prediction（8 步）
- X 轴：前 4 列是 sparse memory，后 8 列是生成的 future space

发现：
- **早期 layer**：attention 几乎全在 future space —— 模型在"看自己想象的电影"
- **后期 layer**：attention 集中在 memory —— 模型回到"看真实历史"
- **中间 layer**：两者混合
- **早期 action step**：偏 memory
- **后期 action step**：偏 future

这跟人类 planning 的直觉一致：我马上要做的动作依赖"我现在看到什么"，但我 5 步之后要做的动作依赖"我预测会发生什么"。

---

## LIBERO 上的结果

| Model | Input | Avg |
|---|---|---|
| Diffusion Policy | S-RGB | 72.4 |
| Octo | S-RGB | 75.1 |
| OpenVLA | S-RGB | 76.5 |
| MAIL | S-RGB + G-RGB | 81.2 |
| **ENERVERSE** | **S-RGB** | **84.1** |
| **ENERVERSE** | S-RGBD → RGB + 2 Render | **88.5** |

只用 single RGB 就比需要双 camera 的 MAIL 高 3 分。加上 2 个 rendered view 后达到 88.5。

---

## 真实世界实验的 insight

Block Placement 任务（把 block 放进很 tight 的 compartment）：

| Metric | ENERVERSE | OpenVLA |
|---|---|---|
| Grasp | 1.0 | 0.89 |
| Place | 0.89 | 0.61 |
| Instruction Following | 0.78 | 0.96 |
| Overall | 0.67 | 0.61 |

**Place 这个 metric 的差距最大**：0.89 vs 0.61。因为 compartment 只比 block 稍微大一点，需要精确的 3D spatial understanding。这正好是 ENERVERSE 的 4D prior 的 sweet spot。

**Instruction Following 反而弱**：因为 OpenVLA 有 LLM backbone，language understanding 更强。ENERVERSE 只用 T5 encoder，language 能力弱一些。

---

## 这篇 paper 的 bigger picture

ENERVERSE 其实验证了一个 hypothesis：**robotics foundation model 的关键是 learn a good world representation，而不是 learn a good action mapper**。

你给它一个好的"脑内小剧场"（video diffusion pretraining），action prediction 就变成了相对 easy 的 task（只需要一个 190M 的 DiT head）。

这跟 LLM 的 story 类似：pretrain 学 representation，fine-tune 学 downstream task。只是 robotics 的 "representation" 需要 encode 3D + temporal dynamics，比 text 的 representation 复杂得多。

---

## 我觉得可以改进的方向

1. **EOS detection**：现在用 L1 distance threshold，太 hacky 了。可以学一个 classifier 或者用 uncertainty estimation
2. **Chunk size**：固定 8 不够灵活。长任务可能需要更大的 chunk，精细操作可能需要更小
3. **Rendered views 的 camera pose**：现在固定 ±30° around Z-axis，可以学 Next-Best View
4. **Language encoder**：T5 太弱了。换 LLaMA 或更现代的 LLM 可能提升 instruction following
5. **4DGS 的替代**：4DGS 训练慢，可以探索更 efficient 的 4D representation
6. **Action head**：现在用 DiT，可以试试 Flow Matching 或 Consistency Model 加速

---

## Reference Links

- ENERVERSE Project Page: https://sites.google.com/view/enerverse
- DynamiCrafter: https://arxiv.org/abs/2410.05752  
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- 4D Gaussian Splatting: https://arxiv.org/abs/2402.07708
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14560
- OpenVLA: https://arxiv.org/abs/2406.09246
- LIBERO: https://arxiv.org/abs/2306.03310
- CALVIN: https://arxiv.org/abs/2112.03227
- v-prediction: https://arxiv.org/abs/2202.00512
- Genie (memory noise): https://arxiv.org/abs/2401.15491
- GR-2: https://arxiv.org/abs/2410.06158
- AVID: https://arxiv.org/abs/2410.12822
- VidMan: https://arxiv.org/abs/2411.09153

---

**Bottom line**：ENERVERSE 本质上是在说，robotics 的 future 是 "想象 → 行动"。先在 4D latent space 里 imagine 未来会怎样，再把 imagine 出来的 representation decode 成 action。Sparse memory 让想象能持续很久，multi-view pretraining 让想象有 3D 结构，4DGS flywheel 让想象能逼近真实。这跟人类 planning 的方式惊人地一致 —— 我们也是先在脑子里 visualize 要做什么，然后才动手。

---

# ENERVERSE 深度解析

Andrej，这篇 paper 的核心 insight 非常有意思。它把 robotics manipulation 重新定义为一个 *future space prediction* 问题，然后用 video diffusion 的 imagination capability 来做 policy planning。这和 GR-2、AVID 这条线一脉相承，但 ENERVERSE 在几个关键维度做了突破。让我逐层拆解。

---

## 1. 核心问题动机

作者的核心论点：general video generation model（如 DynamiCrafter）虽然在 pixel level 表现好，但其 latent space 与 3D + temporal interconnected 的 robotics environment 之间存在 *substantial gap*。直接 fine-tune 这种模型来预测 action，会导致 action policy 不准确。

他们强调一点很重要：**pixel-level video quality 和 control success 之间没有直接单调关系**。他们的目标是通过 align latent space 来 encode 3D、action-conditioned dynamics，让 action 可以可靠地跟随生成的 trajectory。

这其实呼应了世界模型（world model）的哲学：不是生成好看的视频本身有用，而是 video generation 过程中学习的 internal representation 有用。参考 Yann LeCun 的 JEPA 思路，以及 Sora 作为 world simulator 的讨论（https://arxiv.org/abs/2405.03520）。

---

## 2. 架构全景

ENERVERSE 有三个主要模块：

| 模块 | 功能 | 关键技术 |
|---|---|---|
| ENERVERSE-G | Multi-view video generation | Chunk-wise autoregressive diffusion + sparse memory |
| ENERVERSE-A | Action policy head | DiT blocks + DDPM denoising on action chunks |
| ENERVERSE-D | Data flywheel | 4D Gaussian Splatting 闭环 |

整个 pipeline 的逻辑是：
1. Single RGB(-D) observation → depth warping → multi-view rendered images
2. ENERVERSE-G 生成 multi-view future space videos
3. UNet middle block 的 feature cache 给 policy head
4. Policy head 通过 DDPM denoising 输出 action chunks

---

## 3. Next Chunk Diffusion（核心技术）

### 3.1 Chunk-wise Autoregressive Generation

这部分最关键的设计是 **chunk** 作为 future space 的 minimal unit。模型不是预测单帧，而是预测一个 chunk（实验中 chunk size = 8 最优）。

**符号定义：**
- $\mathbf{o}_t^{1:K} = [\mathbf{o}_t^1, \dots, \mathbf{o}_t^K] \in \mathbb{R}^{K \times H \times W \times C}$：观察 frame 的 latent sequence
  - $K$：观察帧数（context length）
  - $H \times W$：spatial resolution
  - $C$：latent channels（VAE 编码后通常是 4）
  - $t$：denoising step
- $\mathbf{z}_t^{1:M} = [\mathbf{z}_t^1, \dots, \mathbf{z}_t^M] \in \mathbb{R}^{M \times H \times W \times C}$：预测的 latent sequence
  - $M$：chunk size（论文中 M=8）

**训练目标（v-prediction 形式）：**

$$\min_\theta \mathbb{E}_{t, \mathbf{z} \sim z_{data}, \epsilon \sim \mathcal{N}(0, I)} ||\epsilon - \epsilon_\theta(\mathbf{z}_t^{1:M}, \mathbf{o}_t^{1:K}, t)||_2^2$$

其中：
- $\theta$：denoising network 参数
- $t$：diffusion timestep
- $\epsilon$：ground truth noise（从标准正态采样）
- $\epsilon_\theta$：网络预测的 noise
- $\mathbf{z}_t^{1:M}$：noisy version of prediction latents
- $\mathbf{o}_t^{1:K}$：context latents（可以是 noisy 或 clean）

**v-prediction 公式：**

$$v_t = \alpha_t \epsilon_t - \sigma_t x_0$$

- $\alpha_t = \sqrt{\bar{\alpha}_t}$：signal scale（$\bar{\alpha}_t = \prod_{i=1}^t (1-\beta_i)$，累积 product）
- $\sigma_t = \sqrt{1 - \alpha_t^2}$：noise scale
- $x_0$：clean data
- $\epsilon_t$：注入的 noise

forward process: $x_t = \alpha_t x_0 + \sigma_t \epsilon_t$

v-prediction 相比 ε-prediction 的优势：在高 noise level 时更稳定，避免 signal/noise ratio 失衡。参考 Salimans & Ho 的 progressive distillation（https://arxiv.org/abs/2202.00512）。

**推理时的 autoregressive loop：**
1. 初始：clean observation frames 作为 context
2. 生成 M 帧 denoised frames
3. 新生成的 frames 成为下一轮的 clean context
4. 用 L1 distance 检测 EOS frame（predefined latent），低于 threshold 则终止
5. 重复直到 EOS

这个 EOS detection 机制很巧妙：通过 latent space 的 L1 distance 而非显式信号，让模型学会"任务何时结束"。

### 3.2 Sparse Memory Mechanism（关键创新）

传统方法用 *consecutive* frames 作为 context，问题在于：
- 信息冗余（相邻帧差异小）
- 长序列训练时计算成本高
- 容易在 autoregressive 生成中 collapse（Figure 7 显示无 sparse memory 时模型在 OOD 场景下崩溃）

ENERVERSE 的做法：随机 sparse sample 80% 的 frames 丢弃，只用 20% 作为 context。

**为什么这有效？**
1. **信息论角度**：视频相邻帧的信息熵低，sparse sampling 强迫模型学习 chunk prediction 的本质而非记忆相邻帧的微小变化。
2. **OOD robustness**：covariant shift 在 robot learning 中常见，sparse memory 强制模型处理"gap"而非依赖连续性。
3. **理论无限长度**：因为不依赖 consecutive frames，理论上可以处理任意长 sequence。

**训练时**：random clean frames（sparsely sampled）+ noisy frames → predict denoised latents

**推理时**：sliding window + cosine-related noise injection on memory context（参考 Genie 的设计 https://arxiv.org/abs/2401.15491）

**消融数据（Table 4）：**
- LIBERO-Long-SV without sparse memory: 30.8
- LIBERO-Long-SV with sparse memory: 73.0
差距 42.2 points，这是巨大的提升，证明 sparse memory 对 long-horizon 任务至关重要。

---

## 4. Multi-view Diffusion Generator Block

这是 ENERVERSE 区别于 AVID、VidMan 的核心。单视角视频生成无法恢复准确的 3D structure 和处理 occlusion。

### 4.1 Ray Direction Map Conditioning

为了 encode camera information，用 **ray direction map** 与 image latents channel-wise concat。

Ray map 的含义：每个 pixel 对应一条从 camera center 出发的射线方向，encode 了 intrinsics + extrinsics。这是 Plücker ray 或类似表示的思想，参考 Ray Conditioning（https://arxiv.org/abs/2306.10878）。

具体来说，对于 pixel $(u, v)$：
- 相机坐标系下的 ray direction: $d = K^{-1}[u, v, 1]^T$
- 世界坐标系下: $d_{world} = R \cdot d$，其中 $R$ 是 extrinsics 的旋转部分

这样 model 知道"每个 pixel 从哪个视角看"，实现 view-aware representation。

### 4.2 4D Latent Space Attention

输入 latent shape: $BCVTHW$（Batch × Channel × View × Time × Height × Width）

Reshape 操作：
- **Spatial attention**: $(BT)(VHW)C$ — 每个 frame 内部 pixel 之间 attention
- **Temporal attention**: $(BVHW)TC$ — 同一 view 跨时间 attention，捕捉 dynamics
- **Cross-view attention**: 在 spatial attention 时，通过 reshape 让同一 spatial location 跨 view 交互

这种设计保证了：
- pixel-to-ray alignment 保留
- cross-view geometric coherence
- temporal dynamics capture

### 4.3 为什么 multi-view pretraining 对 single-view 部署有益？

这是论文最有意思的 claim 之一。在 pretraining 阶段，模型从多视角学到 3D geometric prior。在 inference 时，即使只有 single camera，也可以：
1. 通过 depth warping 渲染出 auxiliary views
2. 把 auxiliary views 当作 multi-view 输入

**Table 6 的证据：**
| Model | Multi-view Pre-Train | Input at Test | SR |
|---|---|---|---|
| DynamiCrafter + DP | No | S-RGB | 79.0 |
| EnerVerse-A | Yes | S-RGB | 92.1 |
| EnerVerse-A | Yes | S-RGB with 1 Render | 93.0 |

13.1 points 的提升仅来自 multi-view pretraining！这说明 3D prior 不是 explicit 的多相机输入带来的，而是 implicit 地 baked into model weights 里的 representation。

---

## 5. ENERVERSE-D: Data Flywheel

这是解决 sim-to-real gap 的关键设计。

### 5.1 动机

- 精确标定的多相机 + robot action 数据昂贵
- Simulator 数据丰富但有 sim2real gap
- 直接用 generative model 生成会有 hallucination

### 5.2 Iterative Loop

1. **Sparse observation**：$n \ll m$ 个 robot-mounted cameras 提供完整 observation sequence（clean latents，跳过 noise injection）
2. **Multi-view generation**：对 unobserved target views 做 standard noisy-to-denoised diffusion
3. **4DGS reconstruction**：用 observed + generated multi-view videos + poses 重建 4D scene
4. **Re-render**：4DGS 渲染所有 target views，获得 high-fidelity、geometry-consistent frames
5. **Feedback**：re-rendered frames re-noise → 再过 multi-view generator → 再 4DGS 优化
6. **Iterate**：逐步降噪、提升重建精度、tighten cross-view consistency

### 5.3 为什么 4DGS 在这里有价值？

4DGS 提供了 **geometric constraint**。Generative model 容易产生 geometrically inconsistent hallucination，但 4DGS 的 explicit Gaussian representation 强制几何一致。两者互补：
- Generative model：adaptability、texture generation
- 4DGS：spatial consistency、geometry

**量化证据**：在 "arrange workpieces" 任务（frequent self-occlusion）上，4DGS 减少 40% 的 hallucination。

参考 4DGS（https://arxiv.org/abs/2402.07708）和 3DGS 原始 paper（https://arxiv.org/abs/2308.14560）。

---

## 6. ENERVERSE-A: Policy Head

### 6.1 Action Chunk Prediction

Action 表示：7D vector = [delta position (x,y,z), rotation (roll,pitch,yaw), gripper openness]

预测 action chunk $a_{t:t+\tau-1} \in \mathbb{R}^{\tau \times d}$，$\tau$=8，$d$=7

### 6.2 Denoising Process

**公式：**
$$a_{t:t+\tau-1}^0 \gets f_\theta(c, o_t, a_{t:t+\tau-1}^k, k) = h_\theta(E, a_{t:t+\tau-1}^k, k)$$

变量解释：
- $a_{t:t+\tau-1}^0$：denoised clean action chunk（target output）
- $a_{t:t+\tau-1}^k$：noisy action at diffusion step $k$
- $k \in \{1, \dots, K\}$：diffusion step index
- $c$：language prompt（T5 编码）
- $o_t$：current observation
- $E$：**cached visual latent** — 从 UNet middle block 在 first denoising step（最 noisy step）提取
- $h_\theta$：policy head = DiT blocks stack + linear projection to action space
- $f_\theta$：完整 denoising function

**关键设计决策：**

1. **E 只计算一次**：在第一个 denoising step（最 noisy 的 $k=K$）从 video diffusion backbone 提取 visual latent，然后 cache 跨所有 action denoising steps 复用。这是效率的核心来源。

2. **为什么用 first denoising step？** 此时 input 最 noisy，模型必须 extract 最 robust、task-relevant 的 feature（low-frequency semantic info），而非 high-frequency pixel detail。这和直觉一致：action prediction 不需要 pixel-perfect feature，而是 task-understanding feature。

3. **View aggregation**：visual latent 通过 mean over spatial dimensions 在 UNet middle block 聚合，shape 变成 $T \times C'$（$T$=video length，$C'$=aggregated channels）。这样 multi-view 信息自然融合。

### 6.3 Training Objective

DDPM-style denoising MSE：
$$\mathcal{L} = \mathbb{E}_{k, a^0, \epsilon} ||a^0 - f_\theta(c, o_t, a^k, k)||_2^2$$

参考 Diffusion Policy（https://arxiv.org/abs/2303.04137）。

### 6.4 两阶段训练策略

**Table 5 的消融：**
| Strategy | LIBERO-Spatial |
|---|---|
| All-Scratch | Failed |
| With DC Pretrain | 79.0 |
| One-Stage Co-Train | 86.3 |
| **Two-Stage Finetune** | **92.1** |

关键发现：
1. From scratch 完全不收敛 — 预训练初始化是必要的
2. Co-training（视频生成 + action 同时优化）有效但不如两阶段
3. 最佳策略：先 video pretrain → 再 fine-tune action head

**为什么两阶段更好？** 我的理解是：video generation pretraining 建立了 4D world representation，这个 representation 一旦固化后，action head 只需学习从 representation 到 action 的映射，不需要重新调整 representation。Co-training 时两个 loss 可能 conflict。

---

## 7. 实验结果深度分析

### 7.1 LIBERO Benchmark（Table 2）

| Model | Visual Input | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| OpenVLA | S-RGB | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| MAIL | S-RGB, G-RGB | 74.3 | 90.1 | 81.8 | 78.6 | 81.2 |
| **ENERVERSE** | **S-RGB** | **92.1** | **93.2** | 78.1 | 73.0 | **84.1** |
| **ENERVERSE** | S-RGBD → RGB + 1 Render | 93.0 | 95.0 | 81.0 | 73.0 | 85.5 |
| **ENERVERSE** | S-RGBD → RGB + 2 Render | 91.2 | **97.7** | **85.0** | **80.0** | **88.5** |

**Insights：**
- 即使单 S-RGB，ENERVERSE 也比需要双 RGB 的 MAIL 高 2.6 points
- 2 Render views 时 Object suite 达到 97.7，接近 ceiling — 多视角对 occlusion heavy 任务帮助巨大
- Long suite（long-horizon）是 ENERVERSE 的相对弱项，因为 CALVIN 协议要求 sequential task switching 不 reset memory

### 7.2 CALVIN ABC→D（Table 3）

| Method | Input | Avg Len |
|---|---|---|
| 3D Diffuser | S-RGBD, G-RGBD, P | 3.27 |
| GR-1 | S-RGB, G-RGB, P | 3.06 |
| **ENERVERSE** | S-RGB | 3.00 |

ENERVERSE 略低于 3D Diffuser 和 GR-1，但只用 single RGB（其他用 RGBD 或 proprioception）。作者指出 memory-based 模型在 task transition 时没有 reset 信号，是 harder setup。

### 7.3 Real-World Block Placement（Table 7）

ENERVERSE-A vs OpenVLA：
- Grasp: 1.0 vs 0.89
- Place: 0.89 vs 0.61（巨大优势 — 4D prior 对 precise insertion 任务关键）
- Instruction Following: 0.78 vs 0.96（弱项 — 缺 LLM backbone）
- Overall Success: 0.67 vs 0.61

Place subtask 的巨大差距验证了 4D spatial prior 的价值：compartments 只比 blocks 略大，需要 precise spatial understanding。

---

## 8. Attention Map 分析（Appendix D）

这是理解 model behavior 的关键实验。

**Figure 11 的发现：**
- Query axis: action prediction（8 steps）
- Key-Value axis: Sparse Memory（前 4 列）+ Predicted Future Space（后 8 列）

观察：
- (a) Early layer: attention 几乎全部在 future space — 模型 leverage 生成 prediction
- (d) Later layer: attention 集中在 sparse memory — 模型 transition 到 memory-based reasoning
- (c, e) Middle layers: 整合两者

**关键 insight**：早期 action steps 倾向 sparse memory，后期 action steps 倾向 future space。这说明模型有 *hierarchical temporal reasoning* — 近期 action 用 memory，远期 action 用 prediction。

这和人类的 planning 行为一致：立即要做的动作依赖当前状态（memory），未来要做的动作依赖预测（imagination）。

---

## 9. 训练细节（Table 10）

| Hyperparameter | Configuration |
|---|---|
| Diffusion steps | 1000（训练），DDIM 500 steps（推理） |
| Noise schedule | Linear, $\beta_0 = 0.00085$, $\beta_T = 0.0120$ |
| Video resolution | 320 × 512 |
| Chunk size | 8 |
| Latent channels | 4（image）+ 6（ray map） |
| Learning rate | $5 \times 10^{-5}$ |
| Max steps | 100,000 |
| Base model | DynamiCrafter 1.4B |
| Policy head | DiT 190M |

**推理效率**：
- Single RTX 4090: ~280 ms per 8-step action chunk
- Single view: 10.6 GB GPU memory
- Three views: 12 GB GPU memory

---

## 10. 关键 Insights 总结

### 10.1 Generative Prior 作为 World Model

ENERVERSE 验证了一个重要 hypothesis：video generation model 的 internal representation 可以作为 robotics 的 world model prior。关键不是生成的视频好看，而是 representation encode 了 3D + temporal dynamics。

### 10.2 Multi-view Pretraining 的 Generalization

最 surprising 的发现：multi-view pretraining 的 3D prior 能 transfer 到 single-view deployment。这意味着 model 学到的不是 explicit multi-view correspondence，而是 implicit 3D geometric understanding baked into feature space。

### 10.3 Sparse Memory 的 Representation Learning 价值

丢弃 80% frames 不仅节省计算，更强迫 model 学习 task structure 而非 frame-to-frame correlation。这是从 video 领域 frame redundancy 借鉴到 robotics 的成功 transfer。

### 10.4 4DGS 作为 Geometric Constraint

Generative model + 4DGS 的组合体现了 *complementary priors*：generative model 提供 adaptability 和 texture，4DGS 提供 geometric consistency。这可能是未来 sim2real 的标准范式。

---

## 11. 与相关工作的对比

| Method | Pretraining | 3D Prior | Memory | Data Engine |
|---|---|---|---|---|
| AVID | DynamiCrafter | No | No | No |
| VidMan | OpenSora | No | No | No |
| GR-2 | Web videos | No | No | No |
| **ENERVERSE** | Multi-view video diffusion | **Yes（ray map）** | **Sparse** | **4DGS flywheel** |

---

## 12. Limitations 和未来方向

作者自己指出：
1. Video 仍有 artifacts（object penetration, snappy transitions）— 但对 action 影响有限
2. Attention map 分析还不够深入
3. Rendered views 的 camera pose 是 heuristic 设置的（±30° around Z-axis）— 未来可集成 Next-Best View（https://arxiv.org/abs/2309.09556）

---

## 13. 我的思考

从 architecture 设计哲学看，ENERVERSE 做了一个聪明的 trade-off：
- **Pretraining 时**：用 simulator 的 multi-view + 真实标定数据，学习 3D prior
- **Deployment 时**：用 single camera + depth warping 渲染 auxiliary views，规避硬件成本
- **Data engine**：用 4DGS 作为 geometric anchor，refine generative output

这其实是 *asymmetric pretraining-inference* 设计：pretraining 要信息丰富（多视角），inference 要成本低（少视角 + 渲染）。

**与 LLM 类比**：sparse memory 类似 LLM 中的 long-context 处理（如 sliding window attention, sparse attention）。chunk-wise autoregressive 类似 token-level autoregressive，但 chunk = "thought unit"。

**潜在延伸方向**：
1. 把 chunk size 做成 adaptive（task-dependent）
2. EOS detection 用 learned classifier 而非 L1 threshold
3. 4DGS 替换为更 efficient 的 4D representation（如 4D NeRF 或 mesh-based）
4. Ray map 可以用 Plücker coordinate 更好地 encode camera（参考 https://arxiv.org/abs/2407.05875）

---

## Reference Links

- ENERVERSE Project Page: https://sites.google.com/view/enerverse
- DynamiCrafter: https://arxiv.org/abs/2410.05752
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- 4D Gaussian Splatting: https://arxiv.org/abs/2402.07708
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14560
- OpenVLA: https://arxiv.org/abs/2406.09246
- LIBERO: https://arxiv.org/abs/2306.03310
- CALVIN: https://arxiv.org/abs/2112.03227
- v-prediction (Progressive Distillation): https://arxiv.org/abs/2202.00512
- Ray Conditioning: https://arxiv.org/abs/2306.10878
- Genie (memory noise): https://arxiv.org/abs/2401.15491
- GR-2: https://arxiv.org/abs/2410.06158
- AVID: https://arxiv.org/abs/2410.12822
- VidMan: https://arxiv.org/abs/2411.09153
- Is Sora a World Simulator: https://arxiv.org/abs/2405.03520
- Plücker ray for video generation: https://arxiv.org/abs/2407.05875
- Next-Best View for robotics: https://arxiv.org/abs/2309.09556

---

希望这个分析帮你 build intuition about ENERVERSE 的核心设计 philosophy。它本质上是把 video generation model 当作 *implicit world model*，用 4D representation 来 bridge generative imagination 和 physical action。Sparse memory + multi-view pretraining + 4DGS flywheel 三者形成了一个 self-reinforcing system，这在 robotics foundation model 这条线上是非常有想象空间的设计。
