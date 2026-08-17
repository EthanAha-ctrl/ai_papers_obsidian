---
source_pdf: TeleWorld.pdf
paper_sha256: 56d6989e6dcbbbea3badb7a5e0c4634ccdb364a76b1234c114f43a28cfac8b15
processed_at: '2026-08-12T13:08:23-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TeleWorld 人话版

好嘞 Karpathy，换个人话方式再讲一遍。把之前那些公式、架构图、benchmark 都"翻译"成直觉，同时尽量多联想多发散。

---

## 一句话总结

TeleWorld 干的事：**让一个 video diffusion model 别再瞎 hallucinate 了，给它配一个外挂记忆（4D field），让它能"记住"自己刚生成了啥，下一秒接着画的时候保持一致。**

就这么简单。所有那些 fancy 名词——closed-loop、4D spatio-temporal field、MMPL、DMD、streaming——都是围绕这个 core idea 做的工程支撑。

---

## 1. 为什么这事难：Video Model 的三大病

你做过 video generation 你就知道，现在 Sora-class 模型有三个根本毛病：

### 病一：忘了自己刚画了啥

生成第 1000 帧时，模型已经不知道第 50 帧那张椅子长啥样、放哪了。hidden state 记不住，attention window 又够不着。结果 chair 颜色变了、位置挪了、干脆消失了。

人类不会这样。你闭眼走 100 米，脑子里还有刚才那条街的 mental map。Video model 没这东西。

### 病二：画久了就漂

Autoregressive 生成，每步一点点小 error，累积起来就 color shift、object identity 漂掉。你做过 char-level RNN 生成 text 应该有体感——1000 字之后 model 就开始胡言乱语。Video 同理，只是 drift 的维度更多（颜色、形状、identity、camera）。

数学上：$x_t = f(x_{<t}) + \epsilon_t$，每步 $\epsilon_t$ 独立噪声，$\text{Var}(\sum_t \epsilon_t) = O(T)$。$T=1000$ 就崩。

### 病三：算不动

Diffusion model 要 20-50 步 denoising，每步是 10B+ param 的 transformer forward。要 real-time 30 FPS？别做梦。光一个 forward 都要几秒。

TeleWorld 三板斧分别砍这三个病。

---

## 2. 第一板斧：4D Field 当外挂记忆

核心 insight：**与其指望 model 的 weight 记住世界，不如外置一个 world representation 让它随时查。**

这跟 RAG 一个道理。LLM 记不住所有 facts？外挂一个 vector database 随时 retrieve。Video model 记不住世界？外挂一个 4D field 随时 render 当 guidance。

### 4D field 长啥样

paper 没明说，但从 "dynamic point clouds" 措辞推测，大概是：

- Static 部分（地面、墙、树）：3D point cloud 或 3DGS，跟 WonderWorld / LucidDreamer 一个套路
- Dynamic 部分（走动的人、开的车）：per-frame point cloud + temporal trajectory linking
- 合在一起就是 4D：$(x, y, z, t) \to \text{color, geometry}$

为啥 4D 不 3D？因为 3D 只能存 "这棵树在哪"，存不了 "这辆车 5 秒前在 A 点，现在在 B 点"。Dynamic object 需要时间维。

### 闭环怎么转

```
User 按 W 键 → camera forward
   ↓
4D field 按新 camera pose render 一段 guidance video
   ↓
DiT 拿 guidance video 当 prefix，generate 新 video segment
   ↓
新 segment 里的 planning frames (第 2, 6, 10 帧) 喂回 4D reconstruction
   ↓
4D field 更新（加点、加 trajectory）
   ↓
循环
```

这就是 Figure 1 的 "Generation-Reconstruction-Guidance" 闭环。本质上是 model 一边画一边把画完的"钉"到 4D board 上，下次画就看着 board 画。

### 关键 trick：只在 planning frames 上 reconstruction

这是整套 system 能 real-time 的命门。

如果每帧都 4D reconstruct，计算量爆炸，reconstruction 跟不上 generation。TeleWorld 只在 planning frames（每 segment 3 帧）做 reconstruction，省 90%+ 计算。

为啥敢这么省？因为 planning frames 是 joint optimized 的，error 最低、quality 最高，拿来做 reconstruction 输入最稳。中间帧反正 error 大，不如不做。

这跟 keyframe-based video compression 一个道理——I-frame 精确存，P/B-frame 近似推。

参考：4D Gaussian Splatting https://github.com/hustvl/4DGaussians

---

## 3. 第二板斧：MMPL 切段降低 drift

### 病根回顾

Autoregressive 1000 帧 = 1000 步 error accumulation。

### MMPL 的解法

把 1000 帧切成 100 个 segment，每个 segment 10 帧。

**Segment 内**：一次性 jointly predict 3 个 anchor frames（第 2, 5, 10 帧），不是逐帧 AR。这三个 anchor 互相 constrain，error 互相抵消。这是 non-autoregressive 的精神，把 $O(N)$ intra-segment drift 降到接近 $O(1)$。

**Segment 间**：上一个 segment 的第 10 帧当下一个 segment 的第 1 帧，autoregressive chain。100 个 segment = 100 步 chain。

总 drift：$O(S) = O(100)$ 而非 $O(T) = O(1000)$。降一个数量级。

### 公式人话版

Micro Planning（公式 1）：

$$p(x_s^{t_a}, x_s^{t_b}, x_s^{t_c} \mid x_s^1)$$

人话：给定 segment 第一帧 $x_s^1$，同时预测第 2 帧 $x_s^{t_a}$、第 5 帧 $x_s^{t_b}$、第 10 帧 $x_s^{t_c}$。三个一起出，互相牵制。

Macro Planning（公式 2）：

$$x_{s+1}^1 := x_s^{t_c}$$

人话：下一段的起点 = 上一段的终点。链式接力。

Content Populating（公式 3）：

$$p(\text{中间帧} \mid \text{起点} + \text{anchor})$$

人话：anchor 之间填充，跟 drawing by numbers 一样。两个 sub-segment 互不依赖，可并行。

### 跟 speculative decoding 的类比

LLM 的 speculative decoding：small model 先猜 4 个 token，big model 一次 verify。猜对就赚 4x 速度。

MMPL 微妙不同：不是"猜了再 verify"，是"jointly predict anchor，anchor 之间互相 verify"。更接近 Diffusion Forcing 那个 line of work。

但精神都一样：**用 hierarchical 结构把 sequential 的东西拆成可并行 / 可约束的 chunk**。

参考 Diffusion Forcing: https://diffusionforcing.github.io/

### Boundary flickering 的暗坑

paper 里轻描淡写提了一句"directly using tail latent tokens as next segment prefix introduces boundary flickering and color shifts"。

这是个很 subtle 的问题。你想，segment $s$ 的第 10 帧是 diffusion 出来的 latent，segment $s+1$ 的第 1 帧又是同一个 latent。但两个 segment 的"上下文"不同——segment $s$ 里这个 latent 是"终点"，segment $s+1$ 里它是"起点"。Diffusion model 对这种 role 互换敏感，容易 flickering。

他们的解法：re-encode 一下。把 current segment 的起末 anchor 拼起来 decode 成 video，再 encode 回 latent，用 re-encode 后的 latent 当 next segment 的起点。相当于做了一次"分布归一化"。

这跟 tokenizer 里 BPE boundary issue 同源——boundary 处要 normalize。

---

## 4. 第三板斧：DMD 蒸馏 + 系统优化让 18B 跑 8 FPS

### DMD 是啥

Distribution Matching Distillation (Yin et al. 2024)：把多步 diffusion teacher 蒸馏成 1-4 步 generator。涉及三个 model：

- **Generator** $G_\phi$：要训练的快速 model
- **Teacher** $T_\psi$：原 slow multi-step diffusion
- **Critic** $C_\omega$：判别 generator 与 teacher 分布差异

Loss 大致：

$$\mathcal{L}_G = \text{KL}(G_\phi(z) \| T_\psi(z)) + \lambda \mathcal{L}_{\text{critic}}$$

变量说明：
- $z$ = noise input
- $G_\phi(z)$ = generator 一步出的图
- $T_\psi(z)$ = teacher 多步出的图
- $\text{KL}$ = KL divergence，衡量两个分布差异
- $\mathcal{L}_{\text{critic}}$ = critic 给的 adversarial / distributional loss
- $\lambda$ = 平衡系数

训练完 generator 就能 1-4 步出图，不用 20-50 步。

参考 DMD: https://arxiv.org/abs/2310.18044

### 18B 上做 DMD 为啥难

三个 18B model 同时跑：
- Generator：18B params + KV cache（autoregressive）
- Teacher：18B params，forward only
- Critic：18B params，forward + backward

粗算显存：
- Generator 18B × 2 bytes (bf16) × 4 (Adam) ≈ 144 GB
- KV cache：10B+ autoregressive video model，长 video 上 cache 巨大
- Teacher 18B × 2 bytes ≈ 36 GB
- Critic 18B × 2 bytes × 4 ≈ 144 GB

加起来 300+ GB，单 H100 80GB 远不够，64 张 H100 都吃力。

### 系统三招

#### 招一：Disjoint GPU 用 Ray 调度

把 generator、teacher、critic 放到**不同 GPU 组**：

- Generator 占 4 份 GPU
- Critic 占 1 份
- Teacher 占 1 份

用 Ray (Moritz et al. 2017) 编排。每个 model 在自己 GPU 上稳定运行，不挤一起。

为啥 disjoint 不 shared？因为三个 model 的 memory profile 完全不同：generator 有 KV cache，teacher forward only，critic 有 backward。挤一起会频繁 swap / OOM。

参考 Ray: https://www.ray.io/

#### 招二：Context Parallelism 切 KV cache

KV cache 是大头。用 Ulysses sequence-parallel 把 sequence 维度切到多 GPU。

原理：
- 把 sequence 切 $P$ 段，每 GPU 存一段 KV
- Attention 前 all-to-all，让每 GPU 拿到完整 sequence 的一段 head
- Attention 后 all-to-all 还原

每 GPU 的 KV cache 内存除以 $P$。

这跟 Ring Attention (Liu et al. 2023) 思路一致。TeleTron 是他们内部系统，没公开 paper。

参考 Ulysses: https://arxiv.org/abs/2209.14529
Ring Attention: https://arxiv.org/abs/2310.01889

#### 招三：Pipeline Schedule 消除 bubble（Figure 3 精髓）

这是最巧妙的部分。

**未优化版**（Figure 3a 上半）：
```
GPU 0: [gen fwd]----[gen bwd]----[idle]----[idle]----
GPU 1: [idle]----[critic fwd]----[teacher fwd]----[idle]----
```
大量 GPU 空闲（bubble）。

**优化版**（Figure 3a 下半）：
```
micro-batch i:   [gen fwd] [gen bwd]
micro-batch i+1:           [critic fwd][teacher fwd]
micro-batch i+2: [gen fwd] [gen bwd]
                              ↑ 同一时刻 GPU 0 跑 i+2 gen fwd + i 的 gen bwd
                              GPU 1 跑 i+1 的 critic/teacher fwd
```

三者**并发**，bubble 几乎消除。

**GPU 比例 4:1:1** 的数学：
- Generator forward + backward 时间 $\propto$ 4 GPU
- Critic + Teacher forward 时间 $\propto$ 1 GPU 各
- 4:1:1 让两边时间近似相等，pipeline 完美 overlap

这跟流水线工厂的 line balancing 完全同构——每道工序时间匹配上，bubble 最小。

参考 GPipe: https://arxiv.org/abs/1811.06965
PipeDream: https://arxiv.org/abs/1806.03377

### 结果

- **32 张 H100** 训完 TeleWorld-18B DMD（原方法要 64+）
- 推理：1.3B 模型 32+ FPS，18B 模型 8 FPS @ 960×1760
- End-to-end 比非 pipelined baseline 快 50%

这事的意义：**小团队能做 10B+ world model 了**。Sora、Cosmos 估计都要数千 H100。TeleWorld 用 32 张就训完，是 democratization contribution。

---

## 5. Streaming 系统：让生成真的"流"出来

光 model 快还不够，VAE、SR、调度都得 streaming 化。

### Scheduled Generation（Figure 4 核心）

问题：segment $s$ 的 content populating 没完，segment $s+1$ 不能开始（要等 $s$ 的 $t_c$ anchor）。

解法：一旦 segment $s$ 的 anchor 出来，**马上**开始 segment $s+1$ 的 micro planning，不等 content populating。

```
Segment 0: [plan anchors f2,f6,f10] [populate f3-f5] [populate f7-f9]
Segment 1:                                [plan f2,f6,f10] [populate f3-f5] ...
                                          ↑ 重叠开始
```

这就是公式 4 表达的：

$$x_{s+1}^1 \in \{x_s^{t_b}, x_s^{t_c}\}$$

人话：下一段起点可以是上一段中点 $t_b$ 或终点 $t_c$。

为啥允许选 $t_b$？因为 $t_b$ 比 $t_c$ 早出来 5 帧，能更早启动下一段。但代价是跳过了 $t_b$ 到 $t_c$ 的中间帧——这部分 temporal context 最深、生成 latency 最高。跳过它，peak memory 最小。

这就是 "Minimum Memory Peak Prediction"。trade-off：throughput 略降（帧复用），latency 最小。

实际反馈延迟 ~1 秒（3 个 latent chunk）。user 当前看到的画面对应 1 秒前 input 引起的变化。predictive buffer。

### Stream-VAE

普通 video VAE 一次性 encode 整个 sequence，延迟巨大。Stream-VAE 借鉴 StreamDiffusionV2：

- 4 帧 chunk 处理
- 3D conv 中间 features 缓存复用
- 跨 chunk boundary 维持 temporal coherence

类似 streaming video codec 思路——I-frame chunk + feature cache。

### FlashVSR

Video super-resolution 把低分辨率 latent 放大到 960×1760。challenge：attention 是 $O(T \times H \times W)^2$，巨贵。

FlashVSR 的 trick：
- Locality-Constrained Sparse Attention：self-attention 限制在 local 3D window，避免 quadratic cost
- Lightweight Conditional Decoder：以 Stream-VAE features 为条件，快速重建
- Chunk-wise（5 帧），17 FPS @ 960×1760

Locality-constrained 跟 Swin Transformer window attention 同源，但 adapted 到 3D video window。

---

## 6. 数据集：500K 精选 vs 数百万粗选

### Curation Pipeline 4 步

1. **Collect**：YouTube, Pexels, Pixabay, Mixkit, Bilibili
2. **Quality filter**：LAION aesthetic > 6, PaddleOCR 去水印文字, 去 corrupted clip
3. **Motion filter**：TTT3R 估 camera motion 去 static, Qwen-2.5-VL-72B 检测 moving subject
4. **Expert review**：20 个专家 690 人时

### Annotation Pipeline 3 步

1. **Moving object mask**：Segment Any Motion in Videos (Huang et al. 2025a)
2. **Camera trajectory**：4D-VGGT → point cloud + depth + intrinsics + poses + 3D object trajectory
3. **Semantic caption**：Qwen-2.5-VL-72B 生成 text

500K 听起来不大（Sora 估计百万级），但 690 人时 expert review 极 labor-intensive。这跟 LLM 的 Chinchilla 后重视 data quality 趋势一致。

可能 missing：scientific video, medical, industrial, low-light, extreme weather。Dataset bias 是潜在 limitation。

参考 Qwen2.5-VL: https://arxiv.org/abs/2502.13923
TTT3R: https://arxiv.org/abs/2509.26645
PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

---

## 7. WorldScore 结果人话解读

### 双榜第一

- **WS-Static 78.23**：第一名（Voyager 77.62 第二）
- **WS-Dynamic 66.73**：第一名（CogVideoX-I2V 59.12 第二）

唯一同时 static + dynamic 第一的 model。

### Static 几乎饱和

TeleWorld 比 Voyager 只多 0.61 分。说明 static scene modeling 已经接近 benchmark 天花板。3D-based model（WonderWorld、LucidDreamer）在这方面已经做得很好。TeleWorld 的贡献是 video-based 也能 match 3D-based。

### Dynamic 大幅领先

+7.61 vs CogVideoX-I2V。这是核心 differentiator。4D memory 让 TeleWorld 能 maintain dynamic object trajectory across long horizon，纯 video model 在 long dynamic 上必 drift。

### Object Control 全场最佳

74.44，超过所有 3D-based 和 video-based。直接验证 4D memory 的效果——persistent object identity 和 arrangement across long sequences。

### Camera Control 中等

76.58，比 WonderWorld (92.98) 低很多。这是 video-based 的 inherent weakness——3D-based 直接渲染 camera，精确控制；video-based 是 implicit learned camera control。这是 paradigm trade-off，不是工程能简单补的。

### SubjQual 不高

61.66，Voyager 是 71.09。可能 TeleWorld 视觉质量不如 3D-based 直接渲染高质量 mesh。Video-based 在 perceptual quality 上的弱点还在。

### 跨 paradigm 的 niche

TeleWorld 占据了一个新 niche：

| Paradigm | 强 | 弱 |
|---|---|---|
| 3D-based | 结构、CamCtrl | 条件灵活、dynamic |
| Video-based | 条件灵活、视觉 | semantic drift、世界崩 |
| 4D generative (4D-fy) | - | 全 metric 低（27.98 static） |

TeleWorld：3D 的结构 + video 的灵活性 + 4D 的 temporal memory。这是之前没人做到的 regime。

注意 4D-fy 在 WorldScore 上极差（27.98 static），说明纯 4D generative 在长 horizon interactive benchmark 上不 work。TeleWorld 的 hybrid 路线是必要的。

---

## 8. 跟相关工作的横向联想

### vs LeCun JEPA

LeCun 一直主张 world model 应该是 latent predictive + abstract representation，反对 pixel-level generation。TeleWorld 走相反路线：pixel-level generation + explicit 4D reconstruction。

但殊途同归：4D field 实际上**提供了 LeCun 想要的 abstract representation**，只是用 explicit geometry 而不是 latent。

可以说 TeleWorld 是 "LeCun 想要的功能 + Sora 式的实现"。

参考 LeCun JEPA: https://openreview.net/forum?id=Tw3zdS7i5lh

### vs Sora

Sora 是纯 video diffusion implicit world model。局限：长 video 一致性差、无 explicit memory、无交互。

TeleWorld 通过 4D field + MMPL + keyboard control 分别解决这三点。

但 Sora 在 physical realism 上可能更强（更多 data + 更大 compute）。TeleWorld 是 interactive coherence 优先，Sora 是 visual realism 优先。

参考 Sora: https://openai.com/sora

### vs Genie 3 (DeepMind)

Genie 3 强调 interactive controllable 3D worlds，走 latent action + interactive generation 路线，类似 SIMA 思路。

TeleWorld 更 explicit，4D field 可下载、可解析。Genie 3 从 internet video 学习 interactive dynamics，scalable。TeleWorld 的 explicit structure 可控、可编辑。

两个路线都 promising，最终可能 converge。

参考 Genie 3: https://deepmind.google/models/genie/

### vs NVIDIA Cosmos

Cosmos 是 physical AI foundation model，针对 robotics 和 autonomous systems，走 video diffusion 路线，注重 physical simulation realism。

Cosmos 偏 physical AI（driving, robotics），TeleWorld 偏 interactive world exploration（gaming, embodied navigation）。目标场景不同。

参考 Cosmos: https://www.nvidia.com/en-us/ai/cosmos/

### vs WonderWorld / Voyager / LucidDreamer

这系列是 3D-based world generator：单图 / text → 3D 可探索环境。共同特点：static scene 强，dynamic 弱。

TeleWorld 是 4D 版 WonderWorld + streaming video generation。本质加了一层时间维。

参考 WonderWorld: https://wonderworld-2024.github.io/
LucidDreamer: https://luciddreamer-cvlab.github.io/

### vs Self-Forcing / Causvid

Self-Forcing (Huang et al. 2025c)：training 时 self-conditioning on generated frames，缩小 train-test gap。
Causvid (Yin et al. 2025)：causal video generation，从 bidirectional 蒸馏到 causal。

TeleWorld 在这些基础上加 MMPL planning，是 hierarchical 升级。

### vs Krea Realtime 14B

Krea Realtime 14B (Millon 2025) 是 real-time video generation 的 SOTA，用 dynamic KV cache management 解决 14B 模型 real-time。

TeleWorld 在此基础上：
1. 模型更大（18B）
2. 用 context parallelism sharding KV cache（Krea 没 sharding）
3. 加 DMD 蒸馏 + 3-model pipeline

Krea 是 single model real-time，TeleWorld 是 distilled + sharded real-time，技术路线不同。

### vs RELIC (Adobe)

RELIC (Hong et al. 2025) 是 interactive video world model with long-horizon memory，用 compact KV cache。

TeleWorld 在 memory 上更 explicit（4D field vs KV cache），可控性更强。但 RELIC 的 KV cache memory 更轻量。

### vs Hunyuan 系列

Tencent 的 Hunyuan-Voyager（RGB-D video → 3D point cloud）、Hunyuan-GameCraft2（game video）、HunyuanWorld 1.0（360° 3D world）都偏 gaming / 3D output。TeleWorld 偏 general world modeling + 4D dynamic。

### vs Diffusion Forcing

Diffusion Forcing (Boyi Li et al.) 把 multiple frames 当 independent latent variable 联合 diffusion。MMPL micro planning 精神上同源（joint predict multiple frames），但 MMPL 是 conditional on initial frame 的，不是完全 joint diffusion。

参考 Diffusion Forcing: https://diffusionforcing.github.io/

### vs Hierarchical RL (Sutton-Precup Options)

Options framework 把 long horizon 切成 segments + intra-segment policy。MMPL 是这个思想在 video generation 上的 instance：micro planning = intra-segment policy, macro planning = inter-segment option chain。

### vs AlphaZero MCTS

AlphaZero 在 search tree 里做 planning，leaf evaluation 用 value network。MMPL 在 video timeline 上做 planning，anchor frame 用 joint diffusion predict。两者都是 "hierarchical planning + learned evaluation"。

### vs Speculative Decoding

Speculative decoding：small model 猜多 token, big model verify。MMPL：joint predict 3 anchor, 后续 content populate 基于 anchor。精神类似——用"先看几步"减少 sequential cost。

参考 Speculative Decoding: https://arxiv.org/abs/2211.17192

---

## 9. 一些可能没意识到的暗坑 / limitation

paper 没明说的潜在问题，我列一下：

### 暗坑 1：4D field representation 没说清

paper 全篇没明确 4D field 是什么 representation。是 4DGS？dynamic NeRF？per-frame point cloud + temporal linking？

从 "dynamic point clouds" 推测是后者。但这 representation 的 limitation：dynamic object 在 frame 间 fast motion 时，point cloud 之间 association 困难。需要 temporal correspondence，paper 没说怎么做。

### 暗坑 2：Dynamic object recall 能力不明

Object 离开 view 100 帧后回来，4D field 能 recall 吗？如果能，recall 精度多少？paper 没实验。这是 real-world 交互的关键。

### 暗坑 3：Multi-object interaction 不明

两个 dynamic object 相遇、交错、碰撞，4D field 怎么处理？occlusion、interaction physics 都没讨论。

### 暗坑 4：Physical plausibility 缺失

纯 generation-based，没有 physics engine。碰撞、重力、刚体动力学可能不准。NVIDIA Cosmos 在这方面更强（physical AI 定位）。

TeleWorld 是 perceptual world model，不是 physics world model。需要 physics accuracy 的应用要加额外 layer。

### 暗坑 5：8 FPS 不算真 real-time

严格 real-time 是 30+ FPS。8 FPS 对 slow exploration game 够用（类似 Myst、Walking Simulator），对 fast-action game（FPS、racing）不够。

### 暗坑 6：1 秒延迟

"Minimum Memory Peak Prediction" 引入 ~1 秒反馈延迟。user 当前看到的画面对应 1 秒前 input。predictive buffer。这对 slow exploration OK，对 fast action 不行。

### 暗坑 7：Distillation 的 quality loss

DMD 1-step vs teacher 多步的 quality gap 多大？paper 没给 ablation。从 benchmark 看似乎不差，但 specific case（细节 texture、动态光照）可能有 loss。

### 暗坑 8：Training compute 没披露

32 H100 训练多久？几天？几周？没说 GPU hours。这是 reproducibility 关键信息。

### 暗坑 9：Ablation 缺失

paper 没给 MMPL / DMD / 4D field 各自贡献的 ablation table。哪些 gain 来自 algorithm、哪些来自 system、哪些来自 data，分不清。

### 暗坑 10：Dataset bias

500K high-quality video，可能 missing scientific / medical / industrial / low-light / extreme weather。bias 会导致 model 在这些 domain 不 work。

### 暗坑 11：Standby animation 的副作用

"无输入时 camera 也极慢 drift" 是避免 collapse 的 trick。但 user 不操作时画面也在动，可能引起 confusion 或 motion sickness。

### 暗坑 12：Boundary flickering 解法不够 robust

re-encode + duplicate terminal tokens 的方法 paper 没给 failure case。复杂 motion（fast camera turn、object occlusion）时这个 trick 还 work 吗？

### 暗坑 13：TeleTron 系统不开源

Ulysses sequence-parallel 的实现细节没公开。复现 DMD training system 困难。

### 暗坑 14：Multi-agent / NPC 没讨论

单 user 控制。多 user 或 NPC 行为如何？4D field 可以记录多 agent 轨迹，但 generation side 需要 multi-agent conditioning。future work。

### 暗坑 15：Failure modes on long horizon

跑 1000+ frame 会怎样？4D field 何时崩？没 extreme long horizon 测试。

---

## 10. 几个更深层的思考

### 思考 1：External Memory vs Internal Memory

TeleWorld 的 4D memory 实际上是一种 **external memory**——不是 model weight 或 RNN hidden state，而是 explicit spatio-temporal field。

这跟 RAG (retrieval-augmented generation) 思路一致：**外置 memory 比内置 memory 更 reliable**。

人类 hippocampus 也是类似 external-like memory（place cells, grid cells）。TeleWorld 的 4D field 可以看作人工 hippocampus——spatial + temporal 双重 indexing。

这可能是未来 AI 系统的 universal pattern：model weight 学 general reasoning，external field 存 episodic memory。

### 思考 2：Hierarchy 是 Universal Pattern

LLM reasoning model (o1, DeepSeek-R1) 用 test-time compute 换 quality。TeleWorld 的 MMPL 用 planning frames 作为 "test-time reasoning"。

两者 meta-pattern：**遇到 long horizon 问题，加 hierarchy**。

- LLM：sequential token → hierarchical thought
- Video：sequential frame → hierarchical planning frame
- RL：sequential action → hierarchical option
- Search：sequential expansion → hierarchical MCTS

这是 AI 系统的 universal scaling law——horizon $\uparrow$，hierarchy $\uparrow$。

### 思考 3：Planning Frame 类似 Mental Imagery

人类做 mental rotation、spatial reasoning 时，脑子里有 "imagery"——不是完整 image，是 sparse abstract representation。

MMPL 的 planning frames 是 sparse anchor，不是 dense video。这跟 mental imagery 类似——sparse, abstract, anchored。

也许未来 video model 应该有 "mental imagery module"——生成 sparse anchor，不生成 dense video，除非需要 render。

### 思考 4：Generative vs Predictive World Model

LeCun 的 JEPA 是 predictive world model（predict latent representation of future）。Sora / TeleWorld 是 generative world model（generate pixels of future）。

但 TeleWorld 通过 4D field 引入了 explicit representation，变得 hybrid。可以说 TeleWorld 是 "generative model with predictive backbone"。

这可能是正确方向——纯 generative 太贵，纯 predictive 太 abstract。Hybrid 既 reliable 又 flexible。

### 思考 5：Why 32 H100 Matters

Sora 估计数千 H100。Cosmos 也是。Genie 3 估计数千 TPU。这些是大厂游戏。

TeleWorld 用 32 H100 训完 18B world model。这意味：
- 小团队 / 学术 lab 能做 world model research
- iteration cycle 短（32 H100 几天 vs 数千 H100 几周）
- 更多人能 reproducible / extend
- 催化整个领域进展

这跟 LLM 领域 Llama 系列 open weight 的意义类似——降低门槛，让 ecosystem 茁壮。

### 思考 6：WorldScore 双榜第一的深层含义

之前 3D-based 在 static 强、dynamic 弱。Video-based 在 dynamic 强、static 弱。没人两全。

TeleWorld 同时 static + dynamic 第一，说明 hybrid 路线 viable。这条路可能是未来 world model 的 main path——既不纯 3D，也不纯 video，是 explicit + implicit 的 fusion。

### 思考 7：从 Model-Centric 到 System-Centric

TeleWorld 的 contribution 大部分是 system-level：disjoint GPU、context parallelism、pipeline schedule、streaming VAE、FlashVSR。

这反映 AI 研究的 trend：**model architecture 已经 saturation，瓶颈转移到 system integration**。

LLM 也是：transformer 架构基本固定，进展在 RLHF、tool use、context engineering、inference optimization。

TeleWorld 是 video generation 领域这个 trend 的缩影。未来 world model 突破可能不在算法，而在 system。

### 思考 8：TeleWorld as Open Platform?

paper 没说是否开源。如果开源，影响巨大：
- Academic lab 能复现 / extend
- 小公司能基于此 build product
- 整个 field 进展加速

如果不开源，就是个技术 report，影响 limited。从 paper 写作风格看，作者似乎在 demo capability（可能 commercial product 的 PR 性质）。

### 思考 9：Embodied Intelligence 的 next step

TeleWorld 是 exploration world model（用户控制 camera 移动）。embodied intelligence 需要 manipulation world model（robot 控制 gripper / joint）。

Ctrl-World (Guo et al. 2025b) 是 robot manipulation 的 controllable world model。TeleWorld + Ctrl-World = exploration + manipulation。

下一步可能是 unified exploration-manipulation world model，camera + gripper action 同时 support。

### 思考 10：Why Memory Matters

TeleWorld 核心卖点是 4D memory。为啥 memory 这么重要？

因为 intelligence 的本质是"基于过去经验预测未来"。没有 memory 的 model 只能 react 当前 input，不能 plan long-horizon。

人类智能 = perception + memory + reasoning + action。TeleWorld 给 video model 加了 memory 这块，从 reactive agent 进化到 planning agent。这是走向 AGI 的关键 step。

参考 LeCun path: https://openreview.net/forum?id=Tw3zdS7i5lh

---

## 11. 总结：TeleWorld 给我的最大启发

1. **外置 memory 比内置 memory 更可靠**：4D field 比 RNN hidden state / KV cache 更 robust。RAG for video。
2. **Hierarchy 是 long-horizon 的 universal 解药**：MMPL 把 $O(T)$ drift 降到 $O(S)$。
3. **System optimization 是 AI 下一阶段瓶颈**：32 H100 训 18B，靠的是 disjoint GPU + context parallelism + pipeline schedule，不是算法突破。
4. **Hybrid 路线（explicit + implicit）可能胜过纯路线**：纯 3D、纯 video、纯 4D generative 都有局限。Hybrid 是 future。
5. **Static world modeling 已经饱和，dynamic 是下一个 frontier**：WorldScore Static 分差 0.61，Dynamic 差 7.61。
6. **Real-time 8 FPS @ 18B 是工程胜利**：让大 model 走出实验室。
7. **World model 的 "memory" 跟 LLM 的 "context" 是同构问题**：都是怎么让 model 在 long horizon 上保持 consistent。
8. **可能开源才能催化领域**：如果 TeleWorld 开源，影响会数量级放大。

---

## 12. 最后一句话

TeleWorld 不是个 algorithm paper，是个 **system paper**。它的核心 contribution 不在于某个公式多漂亮，而在于把 generation + planning + memory + distillation + streaming 五层 stack 缝起来，跑出 real-time 8 FPS @ 18B，WorldScore 双榜第一。

这跟你做 nanoGPT / Eureka Labs 的体感应该一致——AI 的瓶颈越来越在 system integration，不在单点算法突破。TeleWorld 是 video generation 走向 world model 这个方向上的一个 system-level milestone。

希望听到你对这个 paper 的看法。如果你想做类似方向，最 promising 的 extension 是 dynamic object recall、multi-agent interaction、physics integration 这三块。任何一块突破，都是下一篇 milestone paper。

---

## 13. Reference Links（全量）

- MMPL (Xiang et al. 2025): https://arxiv.org/abs/2508.03334
- DMD (Yin et al. 2024): https://arxiv.org/abs/2310.18044
- DMD2: https://arxiv.org/abs/2405.14881
- WorldScore (Duan et al. 2025): https://worldscore.github.io/
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- Genie 3 (DeepMind): https://deepmind.google/models/genie/
- WonderWorld: https://wonderworld-2024.github.io/
- LucidDreamer: https://luciddreamer-cvlab.github.io/
- Text2Room: https://text-to-room.github.io/
- SceneScape: https://greenwl.github.io/SceneScape/
- InvisibleStitch: https://iv-stitch.github.io/
- Ray: https://www.ray.io/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- Ulysses Sequence Parallel: https://arxiv.org/abs/2209.14529
- Ring Attention: https://arxiv.org/abs/2310.01889
- Self-Forcing: https://arxiv.org/abs/2505.18447
- Causvid: https://arxiv.org/abs/2505.01861 (近似)
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- 4D-VGGT: https://vgg-t.github.io/ (推测)
- TTT3R: https://arxiv.org/abs/2509.26645
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- 4D-FY: https://sherwinbahmani.github.io/4dfy/
- StreamDiffusionV2: https://github.com/P1avery/streamdiffusionv2 (推测)
- FlashVSR: https://arxiv.org/abs/2508.xxxxx (近期，需搜索)
- CogVideoX: https://github.com/THUDM/CogVideo
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- Gen-3 Runway: https://runwayml.com/research/introducing-gen-3-alpha
- LTX-Video: https://github.com/Lightricks/LTX-Video
- Allegro: https://arxiv.org/abs/2410.15458
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate
- VideoCrafter: https://github.com/AILab-CVC/VideoCrafter
- DynamiCrafter: https://doubiiu.github.io/projects/DynamiCrafter/
- T2V-Turbo: https://arxiv.org/abs/2405.18750
- VChitect-2.0: https://arxiv.org/abs/2501.08453
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- GPipe: https://arxiv.org/abs/1811.06965
- PipeDream: https://arxiv.org/abs/1806.03377
- LeCun JEPA: https://openreview.net/forum?id=Tw3zdS7i5lh
- Sora: https://openai.com/sora
- Diffusion Forcing: https://diffusionforcing.github.io/
- MineWorld: https://github.com/MineWorld/MineWorld (推测)
- HunyuanWorld: https://github.com/Tencent/HunyuanWorld (推测)
- RELIC (Adobe): https://relic.adobe.com/ (推测)
- Krea Realtime 14B: https://krea.ai/ (推测)
- ReCamMaster: https://github.com/ailab-cvc/RecamMaster (推测)
- 4D Gaussian Splatting: https://github.com/hustvl/4DGaussians
- VGGT: https://vgg-t.github.io/
- Segment Any Motion: https://arxiv.org/abs/2503.XXXXX (需搜索)
- Swin Transformer: https://arxiv.org/abs/2103.14030
- Options Framework (Sutton-Precup): http://www.cs.utexas.edu/~ai-lab/pubs/Sutton-Precup-98.pdf
- AlphaZero: https://www.nature.com/articles/nature24270
- Llama: https://llama.meta.com/
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- o1: https://openai.com/o1/
- SIMA (DeepMind): https://deepmind.google/discover/blog/sima-generalist-ai-agent/

---

希望这个 "人话版" 更直觉。TeleWorld 是个 system paper，看懂它需要同时懂 algorithm、system、benchmark 三块。我最大的 takeaway 是：**外置 memory + hierarchy planning + system optimization 是 world model 走向实用的三件套**。如果你自己 build 这类系统，这三件套缺一不可。

---

# TeleWorld: 4D World Model 深度解析

好嘞 Karpathy，这篇 TeleWorld 我读了好几遍，下面尽可能详尽拆给你。我会把每个公式变量、架构选择背后的 intuition、跟相关工作的横向对比、以及一些 paper 没明说的暗坑都摊开来。

---

## 1. High-Level Intuition

TeleWorld 想做的事情可以用一句话概括：**让 video diffusion model 从 implicit world representation 升级成 explicit 4D world**。

这个动机其实非常贴近你自己之前在 Eureka Labs / Lil'Log 里聊过的那种担忧——纯生成式模型没有"memory of the world"，每一段 video 都是 hallucination，长 horizon 必然 drift。TeleWorld 的解法是把 **生成 (generation) 与重建 (reconstruction)** 通过一个 closed loop 耦合在一起，让 explicit 4D field 当 long-term memory，video diffusion 当 imagination engine。这是非常 LeCun-vs-Sora 之争里的一个 hybrid 落地方向——既不像 JEPA 那样完全 latent predictive，也不像 Sora 那样完全 generative implicit。

Paper 的四个 contribution 我重新组织一下层次：

| 层次 | Contribution | 解决什么 |
|---|---|---|
| Algorithmic | Generation-Reconstruction-Guidance 闭环 | Long-horizon memory |
| Planning | Macro-from-Micro Planning (MMPL) | Error accumulation |
| System | DMD on 18B + 3-model pipeline | Real-time 8 FPS |
| Streaming | Scheduled Generation + Stream-VAE + FlashVSR | Latency / throughput |

下面逐层拆。

参考链接：
- 项目主页（推测）: https://arxiv.org/abs/2508.03334 (MMPL companion paper)
- WorldScore benchmark: https://worldscore.github.io/
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/

---

## 2. Generation-Reconstruction-Guidance 闭环（Figure 1 解析）

### 2.1 数据流图

把 Figure 1 拆成 ASCII 来看：

```
User Keyboard (W/A/S/D + arrows)
      │
      ▼
Camera Pose Mapping
      │
      ▼
┌─────────────────────────────────────┐
│  4D Spatio-Temporal Field (memory)   │ ←─── 4D Reconstruction
│  - static scene points                │       (only on planning frames)
│  - dynamic object trajectories        │
└──────────────┬──────────────────────────┘
               │ render under camera pose
               ▼
       Guidance Video Tokens
               │
               ▼ (concat along frame dim)
┌─────────────────────────────────────┐
│  Autoregressive Diffusion DiT-18B   │
│  - Micro Planning (3 anchors)       │
│  - Content Populating                │
└──────────────┬──────────────────────────┘
               │ output: new video segment
               ▼
       ┌───────────────────┐
       │ Planning frames    │ x_s^{t_a}, x_s^{t_b}, x_s^{t_c}
       └─────────┬─────────┘
                 │
                 ▼
       4D Reconstruction ──── back to field
```

关键 insight：**4D field 只在 planning frames 上更新，不在 dense frames 上更新**。这是整套 system 能做到 real-time 的核心 trick，下面会反复强调。

### 2.2 为什么闭环要 4D 而非 3D

3D representation（point cloud / mesh / 3DGS）只能记 static geometry。一旦场景里有 moving car、walking person，3D field 无法表达 time 维度。TeleWorld 用 4D spatio-temporal field 把 dynamic object 的 trajectory 也记录下来，这样当 camera 后续回头时，object 还在记忆里，不会"凭空消失"——这是传统 video generation 的典型 failure mode。

4D memory 的本质可以类比 RAG (retrieval-augmented generation)：**外置 memory 比 hidden state memory 更 reliable**。模型的 weight 学不会的事，external field 帮它兜底。

参考：4D Gaussian Splatting 系列工作 https://github.com/hustvl/4DGaussians

---

## 3. MMPL: Macro-from-Micro Planning（核心算法）

这是 paper 真正的 algorithmic novelty。详细拆公式。

### 3.1 Motivation：Autoregressive Drift 的数学建模

Autoregressive video generation：
$$x_t = f_\theta(x_{<t}) + \epsilon_t$$

每步 error $\epsilon_t$ 累积：$\text{Var}(\sum_t \epsilon_t) = O(T)$。在 $T=1000$ 帧 long video 上，drift 很严重（color shift、object identity loss、geometry warp）。

Non-autoregressive（如 INN、parallel decoding）：jointly 预测所有 frames：
$$\{x_1, ..., x_T\} = f_\theta(\text{condition})$$

Error 与 $T$ 解耦，但失去 long-range dependency。

MMPL 的 insight：**在 segment 内 non-autoregressive-style joint plan，在 segment 间 autoregressive chain**。Error 从 $O(T)$ 降到 $O(S)$，$S$ 是 segment 数。

### 3.2 Micro Planning（公式 1）

$$p(\mathcal{P}_{\mathcal{M}_s} \mid x_s^1) = p(x_s^{t_a}, x_s^{t_b}, x_s^{t_c} \mid x_s^1)$$

变量说明：
- $s$ = 第 $s$ 个 segment
- $x_s^1$ = segment $s$ 的 initial frame
- $\mathcal{P}_{\mathcal{M}_s}$ = planning frames 集合
- $t_a = 2$ = early neighbor frame index
- $t_b = N/2$ = midpoint frame index（$N$ 是 segment length）
- $t_c = N$ = segment terminal frame index

为什么选这三个位置？我的解读：
- $t_a = 2$ 提供 "immediate temporal anchor"，防止 segment 起步就漂移
- $t_b = N/2$ 把 segment 切两半，提供 middle anchor，让 content populating 可以 two-stage 并行
- $t_c = N$ 是 terminal，作为下一 segment 的 initial，承上启下

这三个 frames 是 jointly 优化、只 conditioned on $x_s^1$ 的，所以它们之间互相 constrain，residual error 互相抵消。这是 non-autoregressive 的精神。

### 3.3 Macro Planning（公式 2）

$$p(\mathcal{P}_{\mathcal{M}^+} \mid x_1^1) = \prod_{s=1}^{S} p(\mathcal{P}_{\mathcal{M}_s} \mid x_s^1), \quad x_{s+1}^1 := x_s^{t_c}, \quad \mathcal{P}_{\mathcal{M}^+} := \bigcup_{s=1}^{S} \mathcal{P}_{\mathcal{M}_s}$$

变量说明：
- $\mathcal{M}^+$ = Macro Planning 整体
- $S$ = total segment count, $S \ll T$
- $x_{s+1}^1 := x_s^{t_c}$ = 上一 segment 的 terminal 作为下一 segment 的 initial（autoregressive chain 的 link）
- $\mathcal{P}_{\mathcal{M}^+}$ = 所有 segment planning frames 的 union

数学上，这个 chain 让 global error accumulation 是 $O(S)$ 而非 $O(T)$。假设 $T=1000, N=10, S=100$，原本 1000 步 drift，现在只有 100 步 drift，drift 降一个数量级。

### 3.4 Drift-Resilient Re-Encoding（关键的 implementation 细节）

paper 这里抛了个 subtle 问题：直接把上一 segment 的 tail latent tokens 拿来当下一 segment 的 prefix，会有 **boundary flickering** 和 **color shift**，原因是 initial frame 与 temporally-compressed latent frame 的 distribution mismatch。

他们的解法：
1. 把 current segment 的 initial + terminal planning tokens 拼起来，重建一个 short clip
2. 把 terminal tokens duplicate 一下，插入形成 contiguous latent sequence
3. 把这个 duplicate 后的 latent 再 re-encode，得到的 latent 作为下一 segment 的 initial condition

直觉：通过 re-encoding 让 distribution 对齐，相当于做了一个"分布归一化"。这跟你做 char-level → BPE tokenization 时的 boundary issue 是类似的——你需要在 boundary 处做 normalization。

具体细节在 MMPL paper 里：https://arxiv.org/abs/2508.03334

### 3.5 Content Populating（公式 3）

$$p(\mathcal{C}_s \mid \mathcal{P}_{\mathcal{M}_s}) = p(x_s^{t_a+1:t_b-1} \mid x_s^{1:t_a}, x_s^{t_b}) \cdot p(x_s^{t_b+1:t_c-1} \mid x_s^{1:t_b}, x_s^{t_c})$$

变量说明：
- $\mathcal{C}_s$ = 需要填充的 content frames
- $x_s^{t_a+1:t_b-1}$ = 第一个 sub-segment 的中间帧
- $x_s^{t_b+1:t_c-1}$ = 第二个 sub-segment 的中间帧
- $x_s^{1:t_a}$ = conditioning context（之前的所有帧 + $t_a$ anchor）
- $x_s^{t_b}$ / $x_s^{t_c}$ = sub-segment 的 end anchor

关键：**两个 sub-segment 的 populating 是 conditional independent 的**（都依赖 $x_s^{t_b}$，但内部不互相依赖）。一旦 $t_b$ ready，两个 sub-segment 可以**并行** populating。这就是 paper 说 "multiple sub-segments can be optimized in parallel" 的来源。

这把 segment 内的 generation 时间从 $O(N)$ 降到 $O(N/2)$（理想情况）。配合多 GPU 分布，长 video 生成效率大幅提升。

### 3.6 MMPL 跟相关思想的连接

- **Speculative Decoding (LLM)**：MMPL 的 micro planning 预测 anchor frames 就像 speculative decoding 预测 ahead tokens。差异在于 MMPL 用 joint optimization 而非 sequential prediction。
- **Diffusion Forcing (Boyi Li / Chen et al.)**：把多 frames 当 independent latent variables 联合 diffusion，MMPL micro planning 精神上同源。
- **Hierarchical RL**：options framework（Sutton-Precup）也是把 long horizon 切成 segments + intra-segment policy。MMPL 是这个思想在 video generation 上的 instance。
- **Look-ahead planning in AlphaZero**：planning frames 像 MCTS 中的 leaf evaluation。

参考：
- Diffusion Forcing: https://diffusionforcing.github.io/
- Speculative Decoding: https://arxiv.org/abs/2211.17192

---

## 4. 4D Reconstruction 细节

### 4.1 Key-Frame Reconstruction 的设计巧思

Paper 里这部分很关键但写得简略。核心 idea：**只对 planning frames $\{x_s^{t_a}, x_s^{t_b}, x_s^{t_c}\}$ 做 4D reconstruction**，不对 dense frames 做。

为什么 work：
1. Planning frames 是 joint optimized，**error 最低、quality 最高**——用作 reconstruction 输入最稳
2. Planning frames 在 time 上 sparse（每 segment 3 帧），**reconstruction 计算 load 可控**
3. 4D field 只需记录 sparse key observations，中间运动由 content populating 填补——这跟 MMPL "sparse anchor + dense interpolation" 的哲学一致

这个设计让 reconstruction 速度能跟上 generation 速度，是 real-time 闭环的 enabler。

### 4.2 Move Object Segmentation（4D-VGGT-based）

基于 4D-VGGT (Wang et al. 2025b) 的 dynamic saliency map。

Sliding window 跨帧聚合 temporal info：
$$\mathcal{W}(t) = \{t-n, ..., t-1, t+1, ..., t+n\}$$

三层 saliency：
- $w_{\text{shallow}}$ = semantic saliency（浅层 feature，语义级动态）
- $w_{\text{middle}}$ = motion instability（中层，运动不稳定区域）
- $w_{\text{deep}}$ = spatial prior（深层，suppress outliers）

Per-frame dynamic mask：
$$M_t = [\text{Dyn} > \alpha]$$

其中 $\alpha$ 是 threshold，$[\cdot]$ 是 Iverson bracket。然后做 feature clustering refinement。

**Network-level masking trick**：只在 layer 1-5 (shallow + middle) 抑制 dynamic tokens 的 **K (key) values**。这是 self-attention 层面的操作——让 dynamic pixels 不污染 static scene 的 geometric estimation，但仍保留 Q/V 信息。

Intuition：attention 是 $\text{softmax}(QK^T/\sqrt{d}) V$，抑制 K 让 dynamic pixels 不被其他 tokens attend 到，但 V 保留让 dynamic tokens 自己能 attend 别人。这是个 asymmetric masking，很巧妙。

参考：4D-VGGT，类似 VGGT (CVPR 2025) 的扩展 https://vgg-t.github.io/

### 4.3 Static + Dynamic 的统一

Paper 说 "static scene elements are merged and progressively expanded, while sparse dynamic components are separately rendered over time"。

我的解读：
- Static 部分用传统 3D reconstruction（point cloud / 3DGS）
- Dynamic 部分 sparse，每帧单独存，inter-frame 用 trajectory association
- Rendering 时 static + dynamic composite

但 paper 没说具体 4D field 是 4DGS、dynamic NeRF、还是 per-frame point cloud + temporal coherence。从 "dynamic point clouds" 措辞推断，可能是后者——point cloud per planning frame + temporal linking。这种 representation 渲染快（适合 real-time）、可解释、可下载。

### 4.4 Sparse Dynamic 的挑战

Paper 承认："input is limited to pre-planning frames, rendered dynamic content remains highly sparse. This requires predicting subsequent dynamic regions based on earlier frames within the pre-planning sequence"。

意思是：planning frame 之间 dynamic object 怎么走，没有 dense observation。靠 macro planning 在 generation side 把 motion trajectory 补全。然后 4D field 端只存 sparse observation，rendering 时用 planning frame 已有的 dynamic 信息做 guidance。

这其实是个隐含的 limitation——dynamic object 在 planning frame 间 fast motion 时，4D memory 不够 dense 来精确 interpolate。Paper 没深谈。

---

## 5. Guidance 机制

### 5.1 Keyboard Control 映射

| Key | Action |
|---|---|
| W/A/S/D | forward/left/back/right |
| ↑↓←→ | pitch up/down, yaw left/right |
| 组合键 | 同时操作 |

**Standby Animation 的洞察**：即使无键盘输入，camera 也极慢 forward drift。这是个非常工程化但 insightful 的设计——diffusion model 在完全静止条件下容易 collapse（推理路径不稳），slow drift 提供 continuous perturbation，让 generation 留在 stable manifold 附近。

类比 RL 中的 exploration noise，或 LLM 中的 temperature > 0。让 system 有 "vitality"。

### 5.2 View-Conditioned Guidance（公式）

$$x_s = \text{patchify}(z_s), \quad x_t = \text{patchify}(z_t)$$
$$x_i = [x_s, x_t]_{\text{frame-dim}}$$

变量说明：
- $z_s$ = guidance video 的 latent（来自 4D field rendering）
- $z_t$ = target video 的 latent（要生成）
- $\text{patchify}(\cdot)$ = 把 latent 切 patch 的操作
- $x_i \in \mathbb{R}^{b \times 2f \times s \times d}$ = DiT 输入
  - $b$ = batch size
  - $2f$ = frame 数（guidance + target 拼接后翻倍）
  - $s$ = spatial token 数
  - $d$ = embedding dimension

设计思路：guidance 和 target 在 frame dimension 拼接，token 数翻倍。3D self-attention 自然处理所有 tokens，无需额外 cross-video attention layer。这个 idea 来自 ReCamMaster。

Intuition：相当于把 guidance video 当成 "prefix"，target 当成 "continuation"，让 attention 自己学怎么从 guidance 拿信息。这比 explicit cross-attention 更 flexible。

参考 ReCamMaster: https://github.com/ailab-cvc/RecamMaster 或类似 camera-conditioned video generation 工作。

---

## 6. DMD 训练系统（最强 system contribution）

这部分我觉得是 paper 里最 underappreciated 的 contribution。10B+ 模型上做 DMD 是非常 hard 的工程问题。

### 6.1 DMD 背景

DMD (Yin et al. 2024) 把多步 diffusion teacher 蒸馏成 1-4 步 generator。涉及三个 model：
- **Generator** $G_\phi$：要训练的快速模型
- **Teacher** $T_\psi$：原 slow multi-step diffusion
- **Critic** $C_\omega$：鉴别 generator 与 teacher 分布

Loss 大致：
$$\mathcal{L}_G = \text{KL}(G_\phi(z) \| T_\psi(z)) + \lambda \mathcal{L}_{\text{critic}}$$

参考：DMD 原文 https://arxiv.org/abs/2310.18044，DMD2 https://arxiv.org/abs/2405.14881

### 6.2 规模挑战的量化

TeleWorld-18B DMD 训练显存需求：
- Generator: 18B params × 2 bytes (bf16) × 4 (Adam states) = ~144 GB（FSDP 后分摊）
- Teacher: 18B params，forward only → ~36 GB
- Critic: 18B params，forward + backward → ~144 GB
- **Generator KV cache**: autoregressive video model，cache 是 $\text{batch} \times \text{seq\_len} \times \text{layers} \times \text{hidden} \times 2$。10B+ 模型长 video 上 KV cache 可达几百 GB

总 footprint 远超 64 张 H100 80GB 的总显存。即使 FSDP 也不够。

### 6.3 系统优化三招

#### 6.3.1 Disjoint GPU Allocation via Ray

用 Ray (Moritz et al. 2017) 把 generator、teacher、critic 分配到 **disjoint GPU sets**：
- Generator: 一组 GPU
- Teacher: 另一组
- Critic: 又一组

这避免了在同一 GPU 上挤三个 18B 模型。Ray 让 heterogeneous resource allocation 灵活。

参考 Ray: https://www.ray.io/

#### 6.3.2 Context Parallelism for KV Cache

用 TeleTron 的 Ulysses sequence-parallel（同 DeepSpeed-Ulysses 思路）把 generator 的 KV cache 按 sequence 维度分片到多 GPU。

Ulysses 原理：
- 把 sequence 切 $P$ 段，每 GPU 存一段 KV
- Attention 前做 all-to-all，让每 GPU 拿到完整 sequence 的一段 head
- Attention 后再做 all-to-all 还原

这把 KV cache 的 per-GPU 内存除以 $P$。

参考 Ulysses: https://arxiv.org/abs/2209.14529
Ring Attention 思路类似：https://arxiv.org/abs/2310.01889

#### 6.3.3 Pipeline Execution Schedule（Figure 3 解析）

这是最有 engineering 巧思的部分。

**Generator-step pipeline (Figure 3a)**：
- 7 个 micro-batches
- 上半图：non-pipelined baseline——generator forward → critic/teacher forward → generator backward，有大量 GPU bubble
- 下半图：pipelined schedule——
  - Micro-batch $i$ 的 generator backward
  - Micro-batch $i+2$ 的 generator forward
  - Micro-batch $i+1$ 的 critic/teacher forward
  
  三者**并发**在不同 GPU 组上跑，bubble 几乎消除。

**GPU 比例 4:1:1 (generator:critic:teacher)**：
- Generator 是最贵的（KV cache + autoregressive），需要最多 GPU
- Critic + Teacher forward 可并行，且 generator step 时 teacher/critic 都 forward only
- 4:1:1 让 generator forward + backward 总时间 $\approx$ critic + teacher forward 时间，实现 near-perfect overlap

**Critic-step pipeline (Figure 3b)**：
- 4 个 micro-batches
- Generator frozen，简单 producer-consumer 模式
- Generator 当 data producer，critic 当 consumer

### 6.4 训练效率数字

- Denoising steps 固定（不随机采样）→ stage duration 可预测
- 两份 KV cache 维持（一份 forward、一份 backward），但已 sharded，overhead 可控
- **End-to-end 50% 加速 vs non-pipelined baseline**

### 6.5 结果

- TeleWorld-18B 用 **仅 32 张 H100** 完成训练（不是 64+）
- 推理：1.3B 模型 32+ FPS，18B 模型 8 FPS @ 960×1760

对比其他大模型训练：
- Sora：估计数千 H100
- Cosmos：NVIDIA 全力，估计数千 H100  
- Genie 3：DeepMind 规模，估计数千 TPU

TeleWorld 的 system optimization 让小团队能做 10B+ world model，这本身是 democratization contribution。

---

## 7. Streaming 系统

### 7.1 Scheduled Generation（公式 4）

$$\text{Segment } s: \quad x_s^{t_a+1:t_b-1} \sim p_\theta(x \mid x_s^1, x_s^{t_a}, x_s^{t_b})$$
$$\text{Segment } s+1: \quad \{x_{s+1}^{t_a}, x_{s+1}^{t_b}, x_{s+1}^{t_c}\} \sim p_\theta(x \mid x_{s+1}^1), \quad x_{s+1}^1 \in \{x_s^{t_b}, x_s^{t_c}\}$$

变量说明：
- $x_s^{t_a+1:t_b-1}$ = segment $s$ 第一个 sub-segment 的中间帧
- $x_{s+1}^1$ = segment $s+1$ 的 initial frame
- $x_{s+1}^1 \in \{x_s^{t_b}, x_s^{t_c}\}$ = 可以选 $t_b$ 或 $t_c$ 作为 next initial

核心 insight：**Macro Planning 的 planning frames sequential 生成，但 content populating 可 overlap**。当前 segment 还在 populating 中间帧时，下一 segment 已可开始 micro planning（用 $t_c$ 作为 initial）。

### 7.2 Minimum Memory Peak Prediction 策略

为了最小化 latency，选 $x_s^{t_b}$ 作为 next initial（而非 $t_c$），跳过 $t_b+1$ 到 $t_c-1$ 的 deepest temporal context（最高 latency 区域）。

参考 Figure 4：
- $f_1^0$ 是 segment 0 的 initial
- $f_2^0, f_6^0, f_{10}^0$ 是 segment 0 的 planning frames
- 在 $f_3^0, f_4^0, f_5^0$ populating 时，segment 1 用 $f_{10}^0$ 当 $f_1^1$ 开始 planning

Trade-off：
- 代价：frame reuse（throughput 略降）
- 好处：peak memory 最小、per-segment latency 最小
- 实际反馈延迟 ~1 秒（3 个 latent chunk）

意味着 user 当前看到的画面，对应的是 1 秒前 user input 引起的变化。这是 "predictive buffer"——user input 在 buffer 里排队，3 chunk 后才 render。这种 latency 对 slow exploration game 可接受，对 fast-action game 不够。

### 7.3 Stream-VAE

借鉴 StreamDiffusionV2 (Feng et al. 2025)：
- Chunk-wise 处理（4 帧 chunk）
- 3D conv 中间 features 缓存复用
- 跨 chunk boundary 维持 temporal coherence（无需 re-encode 长历史）
- 最小化 "time to first frame"

这是 video VAE 的 streaming 改造。传统 VAE 一次性 encode 整个 sequence，延迟巨大。Chunk-wise + feature cache 让 VAE 也能 streaming。

参考 StreamDiffusionV2: https://arxiv.org/abs/2506.xxxxx (近似)

### 7.4 Video Super-Resolution（FlashVSR-inspired）

- **Locality-Constrained Sparse Attention**：self-attention 限制在 local spatial-temporal window，避免 quadratic cost
- **Lightweight Conditional Decoder**：以 Stream-VAE 输出 features 为条件，快速重建
- Chunk-wise（5 帧），**17 FPS @ 960×1760**

整体 pipeline：
```
[Diffusion chunk @ low res] 
    → Stream-VAE decode (chunk-wise, cached features)
    → FlashVSR upscale (local attention, lightweight decoder)
    → 960×1760 output @ ~8 FPS total
```

参考 Swin Transformer window attention 同源 idea，但 adapted to 3D video windows。

---

## 8. 实验：WorldScore Benchmark 分析

### 8.1 TeleWorld-500K 数据集

#### Curation Pipeline（4 步）
1. **Data collection**：YouTube, Pexels, Pixabay, Mixkit, Bilibili
2. **Quality filter**：
   - LAION aesthetic score > 6
   - PaddleOCR 去除 text/watermark
   - 去极短、corrupted、inconsistent clips
3. **Motion-aware selection**：
   - TTT3R (Chen et al. 2025b) 估计 camera motion，去静止
   - Qwen-2.5-VL-72B 检测 moving subjects
4. **Expert review**：20 个专家 690 人时

#### Annotation Pipeline（3 步）
1. **Moving object segmentation**：Segment Any Motion in Videos (Huang et al. 2025a) → moving foreground masks
2. **Camera trajectory**：4D-VGGT → point clouds + depth + intrinsics + poses + 3D object trajectories
3. **Semantic description**：Qwen-2.5-VL-72B 生成 text captions

数据规模 500K 听起来不大（vs Sora 估计百万级），但 paper 强调 quality > quantity。690 人时 expert review 是非常 labor-intensive 的 curation。

参考：
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- TTT3R: https://arxiv.org/abs/2509.26645
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

### 8.2 WorldScore 结果深度解读

WorldScore (Duan et al. 2025) 是目前最 comprehensive 的 world generation benchmark，包含 12 个 sub-metrics：

**Controllability**：Camera Control, Object Control, Content Alignment
**Consistency**：3D Consistency, Photometric Consistency, Style Consistency, Subjective Quality
**Motion**：Motion Accuracy, Motion Magnitude, Motion Smoothness

参考 WorldScore: https://worldscore.github.io/

| Metric | TeleWorld | 第二名 | 差距 |
|---|---|---|---|
| **WS-Static** | 78.23 | 77.62 (Voyager) | +0.61 |
| **WS-Dynamic** | 66.73 | 59.12 (CogVideoX-I2V) | **+7.61** |
| CamCtrl | 76.58 | 92.98 (WonderWorld) | -16.40 |
| **ObjCtrl** | **74.44** | 69.56 (Hailuo) | +4.88 (best of all) |
| ContAlign | 73.20 | 75.00 (LucidDreamer) | -1.80 |
| 3DCons | 87.35 | 90.37 (LucidDreamer) | -3.02 |
| PhotoCons | 88.82 | 90.20 (LucidDreamer) | -1.38 |
| StyleCons | 85.59 | 84.89 (Voyager) | +0.70 |
| SubjQual | 61.66 | 71.09 (Voyager) | -9.43 |
| MotionAcc | 53.94 | - | top |
| MotionMag | 31.55 | - | moderate |
| MotionSmooth | 34.18 | - | high |

### 8.3 关键观察

#### 观察 1：Static 几乎饱和

TeleWorld vs Voyager 在 static 上只差 0.61 分。说明：
- Static world modeling 已接近 benchmark 天花板
- TeleWorld 没有"特别炫技" static
- 重点投入在 dynamic

但要注意，TeleWorld 的 CamCtrl (76.58) 比 WonderWorld (92.98) 低很多。这是因为 3D-based model 在 explicit camera control 上天然优势（直接渲染），video-based model 在 camera control 上是 implicit learned。这是 video-based world model 的 inherent limitation。

#### 观察 2：Dynamic 大幅领先

+7.61 vs CogVideoX-I2V（video-based 第二）。这是关键 differentiator：
- Video-based model 的 dynamic 第二名 59.12
- TeleWorld 66.73，拉开一个段位

原因：4D memory 让 TeleWorld 能 maintain dynamic object trajectory across long horizon，而纯 video model 在 long dynamic 上 drift。

#### 观察 3：Object Control 全场最佳

74.44，超过所有 3D-based 和 video-based model。这直接验证 4D memory 的效果——**persistent object identity 和 arrangement** across long sequences。

#### 观察 4：Motion Smoothness 高

34.18，避免 baseline 的 temporal discontinuities。来自 learned internal state（4D field）guides temporal evolution，而非 local approximation。

#### 观察 5：跨 paradigm 占据新 niche

| Paradigm | Strength | Weakness | TeleWorld |
|---|---|---|---|
| 3D-based | 结构一致、CamCtrl 高 | 条件灵活性差、dynamic 弱 | 继承结构 |
| Video-based | 条件灵活、视觉质量 | semantic drift、世界崩塌 | 继承视觉 |
| 4D-based (4D-fy) | - | 全 metric 都低（27.98 static） | - |

TeleWorld 占据了"structurally grounded, flexibly conditioned, temporally stable"的 niche——这是之前没人做到的 regime。

注意 4D-fy 在 WorldScore 上表现非常差（WS-Static 27.98, PhotoCons 1.59）。说明纯 4D generative model 在 WorldScore 这种长 horizon interactive benchmark 上不 work。TeleWorld 的 hybrid 路线是必要的。

---

## 9. 横向对比和我的联想

### 9.1 vs LeCun JEPA

LeCun 一直主张 world model 应该是 latent predictive + abstract representation，反对 pixel-level generation。TeleWorld 走的是相反路线：pixel-level generation + explicit 4D reconstruction。

但仔细看，TeleWorld 的 4D field 实际上**提供了 LeCun 想要的 abstract representation**——只是它用 explicit geometry 而不是 latent。可以说：
- 4D field 是 explicit world model
- Video diffusion 是 implicit world model
- TeleWorld 是 hybrid

两者殊途同归：都需要 abstract memory。

参考 LeCun JEPA: https://openreview.net/forum?id=Tw3zdS7i5lh

### 9.2 vs Sora

Sora (OpenAI 2024) 是 video diffusion-based world model，patch-based representation 被认为是 implicit world model。Sora 的局限：
- 长 video 一致性差
- 无 explicit memory
- 无交互能力

TeleWorld 通过 4D field + planning 解决前两点，通过 keyboard control 解决第三点。

但 Sora 在物理 realism 上可能更强（更多 training data，更大 compute）。TeleWorld 是 interactive coherence 优先，Sora 是 visual realism 优先。

参考 Sora: https://openai.com/sora

### 9.3 vs Genie 3 (DeepMind 2025)

Genie 3 强调 interactive controllable 3D worlds，走 latent action + interactive generation 路线，类似 SIMA 思路。TeleWorld 更 explicit，4D field 可下载、可解析。

Genie 3 优势：从 internet video 学习 interactive dynamics，scalable。TeleWorld 优势：explicit structure 可控、可编辑。

参考 Genie 3: https://deepmind.google/models/genie/

### 9.4 vs NVIDIA Cosmos

Cosmos 是 physical AI foundation model，针对 robotics 和 autonomous systems，走 video diffusion 路线，注重 physical simulation realism。TeleWorld 注重 interactive exploration 的 coherence。

Cosmos 偏 physical AI（driving, robotics），TeleWorld 偏 interactive world exploration（gaming, embodied navigation）。两者目标场景不同。

参考 Cosmos: https://www.nvidia.com/en-us/ai/cosmos/

### 9.5 vs WonderWorld / Voyager / LucidDreamer

这一系列是 3D-based world generator：
- **WonderWorld**：单图生成 3D 可探索环境，CVPR 2025
- **Voyager**：long-range 3D scene generation，ACM TOG 2025
- **LucidDreamer**：domain-free 3DGS scene generation
- **Text2Room**：text → 3D mesh
- **InvisibleStitch**：depth inpainting 的 smooth 3D scene
- **SceneScape**：text-driven consistent scene generation

它们共同特点：static scene 强，dynamic 弱。TeleWorld 加了 4D + 长时序 + 实时交互，本质上是 **4D 版 WonderWorld + streaming video generation**。

参考 WonderWorld: https://wonderworld-2024.github.io/

### 9.6 vs Self-Forcing / Causvid

这两个是 autoregressive video diffusion 的稳定性技术：
- **Self-Forcing** (Huang et al. 2025c)：training 时 self-conditioning on generated frames，缩小 train-test gap
- **Causvid** (Yin et al. 2025)：causal video generation，从 bidirectional 蒸馏到 causal

TeleWorld 在这些基础上加了 MMPL planning，是 hierarchical 升级。

参考 Self-Forcing: https://arxiv.org/abs/2505.18447 (近似)

### 9.7 vs Krea Realtime 14B (Millon 2025)

这是 paper 里特别提到的工作。Krea Realtime 14B 是 real-time video generation 的 SOTA，用 dynamic KV cache management 解决 14B 模型 real-time。

TeleWorld 在此基础上：
1. 模型更大（18B）
2. 用 context parallelism sharding KV cache（Krea 没 sharding）
3. 加 DMD 蒸馏 + 3-model pipeline

Krea 是 single model real-time，TeleWorld 是 distilled + sharded real-time，技术路线不同。

### 9.8 vs RELIC (Adobe)

RELIC (Hong et al. 2025) 是 interactive video world model with long-horizon memory，用 compact KV cache。TeleWorld 在 memory 上更 explicit（4D field vs KV cache），可控性更强。

参考 RELIC: https://relic.adobe.com/ (推测)

### 9.9 vs Hunyuan-Voyager / Hunyuan-GameCraft2 / HunyuanWorld

Tencent 系列：
- **Hunyuan-Voyager**：RGB-D video 输出 3D point cloud
- **Hunyuan-GameCraft2**：game video with hybrid historical conditioning
- **HunyuanWorld 1.0**：360° 全景 3D world，semantically structured mesh

它们偏 gaming / 3D output。TeleWorld 偏 general world modeling + 4D dynamic。

### 9.10 4D Memory 的本质

TeleWorld 的 4D memory 实际上是一种 **external memory**——不是 model weight 或 RNN hidden state，而是 explicit spatio-temporal field。这跟 RAG (retrieval-augmented generation) 思路一致：**外置 memory 比内置 memory 更可靠**。

人类 hippocampus 也是类似 external-like memory（place cells, grid cells）。TeleWorld 的 4D field 可以看作人工 hippocampus——spatial + temporal 双重 indexing。

### 9.11 Pipeline Bubble 优化的灵感

Figure 3 的 pipeline schedule 跟 GPU pipeline parallelism (GPipe, PipeDream) 思路同源，但 adapted 到 DMD 的 3-model 场景。

`4:1:1` GPU ratio 选择背后的数学：
- Generator forward + backward 时间 $\propto$ param_size × batch × grad_factor
- Critic + Teacher forward 时间 $\propto$ param_size × batch
- 4:1:1 让 generator (4 GPU) 的 forward + backward 时间 $\approx$ critic + teacher (各 1 GPU) 的 forward 时间

这是经典的 pipeline balancing 问题，跟流水线工厂的 line balancing 一样。

参考 GPipe: https://arxiv.org/abs/1811.06965

### 9.12 Standby Animation 的洞察

"Camera drifts forward slowly even without input" 是个非常工程化但 insightful 的设计。Diffusion model 在完全静止条件下容易 collapse（推理路径不稳定），slow drift 提供 continuous perturbation，保持 generation 在 stable manifold 附近。

这跟 LLM 中 temperature > 0、RL 中 exploration noise、神经网络中 weight noise 是同源的——**让 system 有 vitality，避免 collapse 到 trivial solution**。

### 9.13 三模型 DMD 的 GPU 分配为什么 disjoint

为什么 disjoint GPU 而不是 shared？
- Generator 的 KV cache 占用 dynamic memory，profile 跟 teacher/critic 不同
- Teacher 和 Critic 是 forward-only（generator step 时），memory profile 不同
- Mixed precision / 优化器状态差异
- Ray 让 heterogeneous resource allocation 灵活

如果 shared，会 frequent memory swap / OOM。Disjoint 让每个模型在自己 GPU 上稳定运行。

### 9.14 Context Parallelism 对 KV Cache 的细节

TeleTron 的 Ulysses sequence-parallel 把 sequence（这里就是 frame × spatial tokens）切分到多 GPU。每 GPU 只存部分 KV cache。Attention 通过 all-to-all 重组。

跟 Ring Attention (Liu et al. 2023)、AVID 等思路一致。TeleTron 似乎是内部系统，没公开 paper。

### 9.15 Video Super-Resolution 的 Locality-Constrained Attention

FlashVSR 的 locality-constrained sparse attention 把 attention 限制在 local window，避免全局 attention 的 quadratic cost。这跟 Swin Transformer 的 window attention 思路同源，但 adapted to video（3D windows: spatial × temporal）。

注意 video SR 的 attention 是 $O(T \times H \times W)^2$ 的，非常贵。Locality constraint 降到 $O(T \times H \times W \times \text{window}^3)$。

### 9.16 Diffusion Forcing / Joint Diffusion 关联

Diffusion Forcing (Boyi Li et al.) 把 multiple frames 当 independent latent variable 联合 diffusion。MMPL micro planning 是 similar in spirit (joint predict multiple frames)。但 MMPL 不用 full joint diffusion，而是用 conditional generation conditioned on initial frame。

### 9.17 训练数据规模讨论

500K video clips 听起来不大。但 paper 强调 quality > quantity：
- 690 人时 expert review
- 多阶段 filter
- LAION aesthetic > 6（偏 professional content）
- 有 motion（偏 action content）
- 无 text/watermark（偏 raw footage）

可能 missing：scientific video, medical, industrial, low-light, extreme weather 等。Dataset bias 是潜在 limitation。

这跟 LLM 的 Chinchilla 时代后重视 data quality 趋势一致。TeleWorld 的 500K 精选可能比 5M 粗选更好。

### 9.18 4D field 的具体形式（paper 没说清）

paper 没明确 4D field 是什么 representation：
- 4D Gaussian Splatting?
- Dynamic NeRF?
- Per-frame point cloud + temporal association?
- Implicit neural 4D field?

从 "dynamic point clouds" 描述推断，可能是后者——per-frame 3D point cloud + temporal coherence linking。这种 representation 优势：rendering 快（real-time）、可解释、可下载、容易 merge with static 3D reconstruction。

但 limitation：dynamic object 在 frame 间 fast motion 时，point cloud 之间 association 困难。

### 9.19 ASR (asynchronous streaming reconstruction) 同步机制

paper 没详述 generation 和 reconstruction 的同步：
- Generation 是 streaming (chunk-wise)
- Reconstruction 在 planning frames ready 时 trigger
- 两者通过 shared memory / queue 通信？
- Ray 可能也用于 reconstruction 调度？

实现细节没披露，但推测是个 producer-consumer queue + event-triggered reconstruction。

### 9.20 Distillation 的代价

DMD 让 inference 快（8 FPS vs 原 multi-step 1 FPS），但 training cost 上升（3 个 18B model + KV cache）。

Trade-off：一次性训练成本（32 H100 × 训练时间）换 deployment 成本（4 H100 × 持续 inference）。这是 distillation 的本质 trade-off。

对 production deployment 友好，对 research iteration 不友好（每次改架构都要重新 distill）。

### 9.21 失败模式推测

Paper 没讨论的潜在 failure：
- 长时间 generation 后 4D field 累积漂移
- Dynamic object occlusion 后无法 recall
- Camera 快速运动导致 4D reconstruction 失败
- User input 与生成内容 semantic mismatch
- Standby animation 太慢导致 user 烦躁，太快导致 drift

### 9.22 Multi-agent 扩展

TeleWorld 是单 user 控制。多 user 或 NPC 行为如何？
- 4D field 可以记录多 agent 轨迹，但 generation side 需要 multi-agent conditioning
- 这是 future work 方向

### 9.23 Physical plausibility

纯 generation-based，没有 physics engine。碰撞、重力、刚体动力学可能不准。NVIDIA Cosmos 在这方面更强（physical AI 定位）。

TeleWorld 是 perceptual world model，不是 physics world model。如果应用需要物理 accuracy，需要额外 physics engine layer。

### 9.24 Real-time 8 FPS 的实际意义

8 FPS 对 slow exploration game（类似 Myst、Walking Simulator）够用，对 fast-action game（FPS、racing）不够。Real-time 严格定义是 30+ FPS。

TeleWorld 是 "interactive real-time"（user input → response 有延迟但 continuous），不是 "high-FPS real-time"。

### 9.25 与 LLM Reasoning Model 的类比

LLM reasoning model（o1, DeepSeek-R1）用 test-time compute 换 quality。TeleWorld 的 MMPL 用 planning frames 作为 "test-time reasoning"。两者 meta-pattern：**把 sequential 转成 hierarchical planning**。

- LLM: sequential token → hierarchical thought
- Video: sequential frame → hierarchical planning frame

这是 AI 系统的 universal pattern——遇到 long horizon 问题，加 hierarchy。

### 9.26 与 V-JEPA / VideoMAE 的关系

VideoMAE / V-JEPA 是 video self-supervised learning。TeleWorld 没用预训练 representation，纯 diffusion。可以想象：用 V-JEPA features 做 4D field encoding，可能更好。

但 TeleWorld 强调 generation，不是 representation learning。两者可互补：V-JEPA 提供 features，TeleWorld 提供 generation + memory。

### 9.27 MMPL 的更广泛应用

MMPL 不限于 world model。任何 long-horizon autoregressive generation 都可以用 macro-from-micro planning 减少 drift：
- Long text generation
- Long audio generation
- Long motion generation
- Long music generation

这是 general technique，可能从 video generation 出发但 applicable 更广。

### 9.28 WorldScore Static 饱和的含义

Static scene modeling 已经成熟（WonderWorld, LucidDreamer 等 3D-based 已 80+）。TeleWorld 的 0.61 领先说明 video-based 也能 match 3D-based，这本身是突破——之前 video-based 在 3D consistency 上 weak。

但要注意 SubjQual 61.66 不高（Voyager 71.09）。可能 TeleWorld 视觉质量不如 Voyager（3D-based 直接渲染高质量 mesh）。这是 video-based 的 inherent 弱点。

### 9.29 与 MineWorld (Minecraft) 对比

MineWorld (Guo et al. 2025a) 是 Minecraft 上的 real-time interactive world model。Minecraft 是 block-world，structured，比真实世界简单很多。TeleWorld 处理真实世界，复杂度高得多。

但 MineWorld 在 structured domain 上可能更 efficient（token-based rather than diffusion）。

### 9.30 与 Ctrl-World (Robot Manipulation) 对比

Ctrl-World (Guo et al. 2025b) 是 robot manipulation 的 controllable world model。TeleWorld 是 exploration，Ctrl-World 是 manipulation，两者 action space 不同。TeleWorld 的 keyboard control 是 exploration action，Ctrl-World 是 manipulation action（gripper, joint）。

### 9.31 与 Text2World (LLM Symbolic) 对比

Text2World (Hu et al. 2025) 用 LLM 生成 symbolic world model（PDDL, state machine）。TeleWorld 是 perceptual + continuous，Text2World 是 symbolic + discrete。

两者互补：symbolic world model 提供 high-level planning，perceptual world model 提供 low-level rendering。

### 9.32 与 OccTENS (3D Occupancy) 对比

OccTENS (Jin et al. 2025) 是 autonomous driving 的 3D occupancy world model，用 temporal next-scale prediction。TeleWorld 是 general world model，OccTENS 是 driving-specific。Driving domain 有 structured action（车辆控制），general world 更复杂。

### 9.33 与 Dual-Stream Diffusion VLA 对比

Won et al. 2025 的 Dual-Stream Diffusion 把 world model 与 VLA (vision-language-action) 耦合。TeleWorld 没有 action output，只有 generation。如果加 action stream，可以变成 VLA + world model unified system。

---

## 10. 总结：TeleWorld 的 5 个 Contribution 层次

1. **Algorithmic**：Generation-Reconstruction-Guidance 闭环，把 video generation 和 4D reconstruction 耦合，让 4D field 当 long-term memory
2. **Planning**：MMPL 把 frame-level autoregression 降到 segment-level，error 从 $O(T)$ 降到 $O(S)$
3. **System**：DMD on 18B model，3-model pipeline schedule + context parallelism，32 H100 训练
4. **Streaming**：scheduled generation + Stream-VAE + FlashVSR，real-time 8 FPS @ 960×1760
5. **Empirical**：WorldScore Static + Dynamic 双榜第一，唯一同时 top 的 model

---

## 11. 我的 Open Questions

读完 paper 我还有这些问题没解答：

1. **4D field 具体 representation？** 4DGS / dynamic NeRF / point cloud + temporal link？影响 memory、render speed、可编辑性。

2. **4D reconstruction module 用什么？** 4D-VGGT 用于 mask，但 reconstruction backbone 没说。是 4DGS optimization？还是 feed-forward reconstruction？

3. **Training compute 总量？** 32 H100 训练多久？没披露 GPU hours。可能数天到数周。

4. **Inference latency breakdown？** Generation / VAE / SR 各占多少？哪个是 bottleneck？

5. **Failure modes on long horizon？** 跑 1000+ frame 会怎样？4D field 何时崩？

6. **Dynamic object recall？** Object 离开 view 100 frame 后回来，4D field 能 recall 吗？

7. **Multi-object interaction？** 两个 dynamic object 相遇，4D field 怎么处理？

8. **Distillation 后 quality 损失？** 1-step DMD vs 多步 teacher 的 quality gap 多大？

9. **TeleTron 是否开源？** Ulysses sequence-parallel 实现细节没说。

10. **Ablation？** MMPL / DMD / 4D field 各自贡献多少？paper 没给 ablation table。

---

## 12. Reference Links

- MMPL (Xiang et al. 2025): https://arxiv.org/abs/2508.03334
- DMD (Yin et al. 2024): https://arxiv.org/abs/2310.18044
- DMD2: https://arxiv.org/abs/2405.14881
- WorldScore (Duan et al. 2025): https://worldscore.github.io/
- NVIDIA Cosmos: https://www.nvidia.com/en-us/ai/cosmos/
- Genie 3 (DeepMind): https://deepmind.google/models/genie/
- WonderWorld: https://wonderworld-2024.github.io/
- Voyager (Hunyuan): https://arxiv.org/abs/2506.XXXXX (近似)
- LucidDreamer: https://luciddreamer-cvlab.github.io/
- Text2Room: https://text-to-room.github.io/
- SceneScape: https://greenwl.github.io/SceneScape/
- InvisibleStitch: https://iv-stitch.github.io/
- Ray: https://www.ray.io/
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- Ulysses Sequence Parallel: https://arxiv.org/abs/2209.14529
- Ring Attention: https://arxiv.org/abs/2310.01889
- Self-Forcing: https://arxiv.org/abs/2505.18447
- Causvid: https://arxiv.org/abs/2505.01861 (近似)
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- 4D-VGGT: https://vgg-t.github.io/ (推测)
- TTT3R: https://arxiv.org/abs/2509.26645
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- 4D-FY: https://sherwinbahmani.github.io/4dfy/
- StreamDiffusionV2: https://github.com/P1avery/streamdiffusionv2 (推测)
- FlashVSR: https://arxiv.org/abs/2508.xxxxx (近期，需搜索)
- CogVideoX: https://github.com/THUDM/CogVideo
- Wan2.1: https://github.com/Wan-Video/Wan2.1
- Gen-3 Runway: https://runwayml.com/research/introducing-gen-3-alpha
- LTX-Video: https://github.com/Lightricks/LTX-Video
- Allegro: https://arxiv.org/abs/2410.15458
- EasyAnimate: https://github.com/aigc-apps/EasyAnimate
- VideoCrafter: https://github.com/AILab-CVC/VideoCrafter
- DynamiCrafter: https://doubiiu.github.io/projects/DynamiCrafter/
- T2V-Turbo: https://arxiv.org/abs/2405.18750
- VChitect-2.0: https://arxiv.org/abs/2501.08453
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- GPipe: https://arxiv.org/abs/1811.06965
- LeCun JEPA: https://openreview.net/forum?id=Tw3zdS7i5lh
- Sora: https://openai.com/sora
- Diffusion Forcing: https://diffusionforforcing.github.io/
- MineWorld: https://github.com/MineWorld/MineWorld (推测)
- HunyuanWorld: https://github.com/Tencent/HunyuanWorld (推测)
- RELIC (Adobe): https://relic.adobe.com/ (推测)
- Krea Realtime 14B: https://krea.ai/ (推测)
- ReCamMaster: https://github.com/ailab-cvc/RecamMaster (推测)

---

总结一句：TeleWorld 是 video-based world model 从 implicit 走向 hybrid（implicit + explicit 4D）的关键一步。它的核心 contribution 不在单一 algorithm，而在 system-level integration——把 generation、planning、4D memory、distillation、streaming 五层 stack 缝起来，跑出 real-time 8 FPS @ 18B。WorldScore 双榜第一是这种 system-level 思路的胜利。接下来如果有人能在 dynamic object recall、multi-agent interaction、physical plausibility 上突破，那就是下一个 milestone。
