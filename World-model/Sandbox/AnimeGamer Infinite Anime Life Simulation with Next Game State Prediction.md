---
source_pdf: AnimeGamer Infinite Anime Life Simulation with Next Game State Prediction.pdf
paper_sha256: 95f1dfccd3984f72a262720f095302a561fee9048ca0782cfd987a61cb4c5a5c
processed_at: '2026-07-18T08:04:53-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AnimeGamer 深度解析

Andrej，这篇 paper 的核心 idea 其实非常优雅——**把"游戏"重新定义为 next game state prediction 任务**，并用 MLLM 来充当 game engine。这与你常讲的 LLM as operating system / autoregressive world model 的直觉高度共鸣。下面我尽可能把每个技术细节拆开讲，帮你 build intuition。

---

## 1. Problem Framing：为什么这是 "Infinite Game"

### 1.1 Finite vs. Infinite Games 的定义

作者借用 James P. Carse 的哲学框架 [Carse 1986]：
- **Finite game**：played for the purpose of winning，有 boundary、fixed rules、definitive endpoint。Oasis (Minecraft) [1]、GameGen-X [9]、Playable Game Generation [75]、Diffusion-based DOOM [69] 都属于这一类——它们 replicate 已经存在的 game，接受 keyboard/mouse 这种 fixed command set。
- **Infinite game**：played for the purpose of continuing the play。没有 predetermined boundary，玩家通过 **open-ended language** 持续 interact，环境会 ever-evolving。

前作 Unbounded [41] 第一次提出 generative infinite game 的概念，用 LLM 作为 router 把多轮 dialogue 翻译成 T2I 的 caption + reference image。问题在于：
1. LLM 只接 text context → visual coherence 下降
2. 输出是 static image → 没有 dynamics

AnimeGamer 要解决的就是这两点。**核心哲学**：把 multi-turn game generation 看成 autoregressive sequence modeling，sequence 的每个 token 是一个 "game state"。

---

## 2. Game State s 的分解

每个 round 的 game state s 被分解为两部分：

$$s = (s_a, s_c)$$

- $s_a$：**dynamic animation shot**（一段视频 clip，展示角色动作）
- $s_c$：**character state**，包括三个标量值 stamina / social / entertainment，类似 Tamagotchi 那种 mood/health 指示。

关键洞察：text-only representation 无法 capture 视频里 character + motion 的全部信息，所以必须设计一个**比 text 更结构化的中间表示**。

---

## 3. Action-Aware Multimodal Representation（核心 contribution）

这是 paper 最有意思的部分。作者把一个 animation shot $s_a$ 分解成三个 modality：

$$s_a = \mathcal{E}_a(f_{md}, f_v)$$

其中：

### 3.1 三个 component 的语义

| Symbol | 含义 | 实现 |
|---|---|---|
| $f_v$ | overall visual reference | CLIP embedding of **first frame** |
| $f_{md}$ | action description | T5 text embedding of short motion prompt (e.g. "Softly talk") |
| $f_{ms}$ | motion scope / intensity | optical flow magnitude，量化成 5 个 level |

### 3.2 Encoder 架构（公式 1）

$$\mathcal{E}_a = \text{Concat}(\text{LN}(\text{MLP}(x)), \text{LN}(\text{MLP}(y)))$$

- $x$ 和 $y$ 分别是 $f_v$ 和 $f_{md}$ 的原始 embedding
- MLP 做 dimension alignment
- LN (LayerNorm) 做 scale alignment
- Concat 沿 **token dimension** 拼接

**为什么用 concat 而不是 element-wise add / cross-attention？** 这是 ablation study (Table 3) 直接 answer 的问题：
- **w/ addition**：CLIP-I 从 0.8672 掉到 0.7684
- **w/ cross-attn**：CLIP-I 从 0.8672 掉到 0.7264

直觉：$f_v$ 是 spatial positional information 密集的 visual feature，element-wise add 会破坏这个 structure；cross-attention 会引入"哪个 query attends 哪个 visual patch"的歧义；concat 把两个 modality 当成 sequence 的不同 token，保留各自信息，让下游 diffusion model 自己学如何 cross-modal attend。

### 3.3 为什么用 first frame 而非 random frame

Ablation "w/ rand. frame"：DreamSim 从 0.7928 掉到 0.4500。直觉：first frame 是 video 的"anchor"，类似 Pixar 那种 storyboard 的 first frame；random frame 会引入"哪个 frame 是 anchor"的 noise，让模型不知道该 attend 哪个时间点。

### 3.4 Motion Scope $f_{ms}$：被独立出来

注意 $f_{ms}$ **不进入** $\mathcal{E}_a$，而是作为 **decoder 的独立 condition**。直觉：motion intensity 是一个"全局控制信号"，对所有 frame 同等作用，类似 classifier-free guidance 里的 condition；如果把它塞进 $s_a$，再让 MLLM 预测，再 decode，会经过太多层 transformation，损失控制精度。所以 $f_{ms}$ 被当作 discrete target 让 MLLM 直接 next-token predict（5 个 level），然后独立 inject 进 decoder。

---

## 4. Animation Shot Decoder $\mathcal{D}_a$

### 4.1 基于 CogvideoX [77] 的改造

CogvideoX-2B 是一个 text-to-video diffusion model，用 DiT (Diffusion Transformer) 架构。AnimeGamer 的改造点：

1. 把原 text feature 输入替换成 action-aware multimodal representation $s_a$
2. 额外引入 $f_{ms}$ 作为 condition

### 4.2 Motion Scope 注入方式

$f_{ms}$ 经过 sinusoidal embedding + FC layers + SiLU activation，得到 embedding，然后 **加到 timestep embedding $f_t$ 上**：

$$f_t' = f_t + \text{SiLU}(\text{FC}(\text{SinEmb}(f_{ms})))$$

直觉：timestep embedding 控制"现在去噪到哪一步"，motion scope 也想表达一个全局级别的"how much motion"，两者语义类似（都是 global condition），加法是自然选择。

### 4.3 训练目标（公式 2）

$$\mathcal{L} = \mathbb{E}_{z, c, s_a, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c, s_a) \|_2^2 \right]$$

变量含义：
- $z$：video 经过 3D-VAE encode 后的 latent code
- $c$：original text condition（保留作为辅助 condition）
- $s_a$：action-aware multimodal representation（替换原 text feature 的主体）
- $\epsilon$：从 $\mathcal{N}(0, 1)$ 采样的 Gaussian noise
- $t$：diffusion timestep
- $z_t$：加噪后的 latent
- $\epsilon_\theta$：DiT denoiser，参数为 $\theta$

这是标准的 **v-prediction / ε-prediction** 形式，没有特殊 trick。

### 4.4 Warm-up → Joint Training 两阶段

- **Warm-up**：只训 $\mathcal{E}_a$，10,000 steps。目的：让 encoder 输出落到 decoder 期望的 input distribution 附近，避免 joint training 时 encoder gradient 太大、把 decoder 搞崩。
- **Joint training**：$\mathcal{E}_a$ + $\mathcal{D}_a$ 一起训，80,000 steps。

Ablation "w/o warm-up"：DreamSim 从 0.7928 掉到 0.5107，证实 warm-up 的必要性。

---

## 5. MLLM 作为 Game Engine

### 5.1 输入 / 输出结构

**Input**：historical multimodal context + current instruction
- 历史 game state $s^{(1)}, ..., s^{(t-1)}$ 的 representation
- 历史指令 $u^{(1)}, ..., u^{(t)}$
- 这些都通过 linear resampler 投影到 MLLM input space

**Output**：next game state representation
- $N=226$ 个 learnable query tokens → 输出 $N$ 个 action-aware multimodal representation（对应 $s_a$）
- 然后 special tokens `<MS>`, `<ST>`, `<SC>`, `<ET>` 分别输出 motion scope, stamina, social, entertainment

为什么 N=226？因为要对齐 CogvideoX 的 pretrained text encoder 输出 token 数，复用 pretrained weight，节省 compute。

### 5.2 训练目标（公式 3）

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{MSE}}$$

- $\mathcal{L}_{\text{CE}}$：Cross-Entropy，用于 discrete targets（$s_c$ 和 $f_{ms}$ 的 next-token prediction）
- $\mathcal{L}_{\text{MSE}}$：Mean Squared Error，用于 continuous $s_a$ representation 的 regression
- $\alpha$：loss weight

Ablation "w/ $\mathcal{L}_{\text{cos}}$"（公式 4）：

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{MSE}} + \beta \mathcal{L}_{\text{cos}}$$

加入 cosine similarity loss [Xiao et al. 2025]，$\alpha = \beta = 0.5$。结果显示效果 marginal（Table 3: CLIP-I 0.7856 vs 0.7628 with cos loss，反而略好一点点，但 motion quality 上无明显 gain）。直觉：MSE 已经 capture 了 magnitude 信息，cos 只补 direction，对 latent space alignment 帮助有限。

### 5.3 Sliding Window Inference

**Train-short-test-long scheme** [Ren et al. 2024]：训练时 sample random-length subset，推理时用 sliding window 维持 fixed context length，从而支持理论上的 infinite generation。

直觉：和 LLM 的 context window 处理一样，但这里 context 是 multimodal——需要把历史的 visual representation resample 成固定长度。

---

## 6. Three-Stage Training Pipeline

| Stage | 训练对象 | 数据 | Steps | LR |
|---|---|---|---|---|
| 1. Tokenizer/Detokenizer | $\mathcal{E}_a$ warm-up → joint $\mathcal{E}_a + \mathcal{D}_a$ | 100k WebVid 预训练 + 20k anime clips | 10k + 80k | 2e-4 |
| 2. MLLM | Mistral-7B + LoRA (r=32) | 20k multi-turn anime data | 15k | 5e-5 |
| 3. Decoder Adaptation | 只训 $\mathcal{D}_a$ | MLLM 输出作为 condition | 10k | 5e-5 |

**Stage 3 的 motivation**：stage 1 训好的 $\mathcal{D}_a$ 期望 input 是 $\mathcal{E}_a$(GT video) 的输出；但 inference 时 input 是 MLLM 的输出，两个 latent space 会有 misalignment。Stage 3 用 MLLM 的实际输出作 condition，让 $\mathcal{D}_a$ 学习这个 distribution shift，pixel-level 对齐 GT。

Ablation "w/o adapt"：CLIP-I 从 0.8132 掉到 0.6831（Table 1 vs Table 3 比较）。这是 dramatic drop，说明 stage 3 是必须的。

---

## 7. Dataset Construction Pipeline

由于 anime video 数据稀缺，作者自己造数据：

1. **10 部 anime films** → scene detection [10] → 按 2 秒切分 → ~20,000 video clips，每 clip 16 frames @ 480×720
2. **Captioning**：InternVL-26B [11] 接收 4 帧采样 + reference image，输出结构化标签：
   - `<S>` subject
   - `<MD>` motion description（simple phrase）
   - `<EV>` environment（one word）
   - `<MA>` movement adverb
   - `<SC>` social interaction (0/1)
   - `<ET>` entertainment (0/1)
   - `<ST>` stamina (-1/+1)
   - `<ML>` motion level (1-5)
3. **Motion Scope**：用 Memflow [16] 算 optical flow，取绝对值，threshold $r=0.2$ 滤背景，平均后量化成 5 个 level

这个 pipeline 可复现性高，玩家可以 plug in 自己喜欢的角色 film。

---

## 8. Baselines 设计

没有现成的开源 baseline，作者构造了三个：

- **GC** (Gemini + CogvideoX)：Gemini-1.5 做 router 输出 text caption + character state → fine-tuned CogvideoX-2B 生成 video
- **GFC** (Gemini + Flux + CogvideoX-I2V)：Gemini → Flux T2I (LoRA, r=32, 200k steps) → CogvideoX-I2V 把 image 转 video
- **GSC** (Gemini + StoryDiffusion + CogvideoX-I2V)：Gemini → StoryDiffusion (tuning-free) → CogvideoX-I2V

GSC 是 tuning-free 的 lower-bound，验证 "tuning-free 对这个 task 不够"。

---

## 9. Evaluation Benchmark 构造

**GPT-4o 作 benchmark constructor**：选 20 个 character，每个生成 10-round game，共 2,000 rounds，包含 940 distinct movements 和 133 unique environments。

**GPT-4v 作 judge**：评分维度包括 Overall / Instruction Following / Contextual Consistency / Character Consistency / Style Consistency / State Update。

**Human evaluation**：20 个 bachelor+ 学历、有 image/video generation 经验的 participant，9-round game × 50 samples，ranking → 绝对分数 (1st=10, 2nd=7, 3rd=4, 4th=1)。

---

## 10. 关键实验结果

### 10.1 Automatic Metrics (Table 1)

| Model | CLIP-I↑ | DreamSim↑ | CLIP-T↑ | ACC-F↑ | MAE-F↓ | Inference (s/turn)↓ |
|---|---|---|---|---|---|---|
| GSC | 0.7862 | 0.5019 | 0.3331 | 0.3163 | 0.8263 | 50 |
| GFC | 0.7662 | 0.5797 | 0.3325 | 0.2923 | 1.0212 | 63 |
| GC | 0.7960 | 0.6416 | 0.3339 | 0.4249 | 0.7223 | 25 |
| **AnimeGamer** | **0.8132** | **0.7403** | **0.4161** | **0.6744** | **0.4238** | **24** |

关键观察：
- Character consistency（CLIP-I, DreamSim）AnimeGamer 最高 → 因为 MLLM 直接 consume visual context
- Semantic consistency（CLIP-T）AnimeGamer 0.4161 vs baseline 0.33 → 显著提升 25%，说明 multimodal representation 比 text caption 更 expressive
- Motion quality（ACC-F）0.6744 vs GC 0.4249 → motion scope 显式控制有效
- Inference time 24s/turn，比 GFC (63s) 快 2.6 倍，因为不需要单独的 I2V 步骤

### 10.2 GPT-4V + Human Evaluation (Table 2)

| Model | Overall (GPT-4V) | Overall (Human) | Instr Follow (Human) | Context Consist (Human) |
|---|---|---|---|---|
| GSC | 5.35 | 2.29 | 2.96 | 2.71 |
| GFC | 4.96 | 4.27 | 3.57 | 3.20 |
| GC | 6.42 | 7.38 | 7.37 | 6.89 |
| **AnimeGamer** | **8.36** | **10.00** | **9.95** | **9.95** |

Human evaluation AnimeGamer 拿了满分 10.00 overall，这是非常 striking 的结果。说明 multimodal context 的价值在 user perception 层面被强烈感知到。

---

## 11. Ablation 关键发现 (Table 3)

### 11.1 Tokenizer variants

| Variant | CLIP-I | DreamSim | 备注 |
|---|---|---|---|
| w/ rand. frame | 0.8446 | 0.4500 | first frame 是关键 anchor |
| w/ less para | 0.8406 | 0.4481 | MLP 比 Linear 更好 |
| w/ addition | 0.7684 | 0.6173 | element-wise add 破坏 spatial info |
| w/ cross-attn | 0.7264 | 0.7084 | 引入额外歧义 |
| w/o warm-up | 0.8306 | 0.5107 | warm-up 稳定训练 |
| w/o $f_{ms}$ | 0.8533 | 0.6894 | CLIP-I 略升但 MAE-F 从 0.4029 暴涨到 1.2189 |
| **Ours** | **0.8672** | **0.7928** | - |

"w/o $f_{ms}$" 这一行很有意思：CLIP-I 反而升高（因为不动了 → 更容易保持 character consistency），但 motion quality 完全崩掉。这说明 motion scope 是 **decoupling motion intensity from character appearance** 的关键 mechanism。

### 11.2 Decoder adaptation

| Variant | CLIP-I | DreamSim | Motion ACC-F |
|---|---|---|---|
| w/o adapt | 0.6831 | 0.4937 | 0.3649 |
| w/ $\mathcal{L}_{\text{cos}}$ | 0.7628 | 0.5966 | 0.6649 |
| **Ours** | **0.7856** | **0.6084** | **0.6722** |

不加 adaptation 时 CLIP-I 0.6831，是所有 ablation 里最差的——证明 MLLM output space 和 $\mathcal{D}_a$ expected input space 的 misalignment 是 real problem。

---

## 12. 你的 Intuition 该往哪个方向 build

我个人读完之后的几个 take-away：

### 12.1 "Game = Autoregressive Sequence Model"

这是 paper 最深的 contribution。Game state s 就是 sequence 的一个 token，next-token prediction 就是 game engine。这和 GameNGen [69]、Genie [5]、Oasis [1] 的哲学是同源的，但区别在于：
- GameNGen：latent frame prediction，每帧一个 token，pure RL environment
- AnimeGamer：每"轮游戏"一个 token，token 内部是 multimodal representation，user instruction 是外部 condition

### 12.2 Representation 选择是 bottleneck

为什么 AnimeGamer 比 Unbounded 强这么多？不是因为 MLLM 比 LLM "更聪明"，而是因为 **representation 是 information bottleneck**。Text caption 损失太多 visual+motion 信息，再强的 LLM 也补不回来；action-aware multimodal representation 把 visual reference、action description、motion scope 三个 channel 分开 encode，让 MLLM 在一个 information-rich space 里做 prediction。

### 12.3 三阶段训练的工程智慧

- Stage 1：把 video token space 学好（监督信号最强）
- Stage 2：让 MLLM 学会 predict 这个 token space（language-aligned supervision）
- Stage 3：让 decoder 适应 MLLM 实际输出的 distribution（bridge gap）

这种 "pretrain alignment → learn predictor → fine-tune decoder for predictor output" 的 pipeline 在 unified comprehension+generation model 里是 emerging pattern（参考 SEED-Story [76]、MovieDreamer [80]、VideoAuteur [73]）。

### 12.4 Motion Scope 作为 Independent Channel

把 motion intensity 单独剥离出来作为 discrete target + sinusoidal condition，是一个非常聪明的 design。直觉：motion magnitude 是一个 low-dimensional、global、与 visual appearance 正交的 signal；如果硬塞进 $s_a$，会被 high-dim visual feature 淹没。这种 **modality-specific decoupling** 思路可以推广到很多 generation task（例如把 camera motion、lighting、artistic style 都做成 independent channel）。

### 12.5 局限性

Paper 自己承认：只在 closed-domain (训练过的 character) 上 evaluate，没探索 open-domain generalization。如果玩家想用一个 completely unseen character，当前 pipeline 需要先跑 dataset construction → fine-tune，这是 friction point。下一步可能是 in-context character customization（参考 DreamBooth-style 或 IP-Adapter-style 的 zero-shot character injection）。

---

## 13. Reference Links

- **Project page**: https://howe125.github.io/AnimeGamer.github.io/
- **Code & checkpoints**: https://github.com/TencentARC/AnimeGamer
- **Unbounded (前作)**: https://arxiv.org/abs/2410.18975
- **CogvideoX (decoder base)**: https://arxiv.org/abs/2408.06072
- **Mistral-7B (MLLM base)**: https://arxiv.org/abs/2310.06825
- **InternVL (captioner)**: https://arxiv.org/abs/2312.14238 (Chen et al.)
- **Memflow (optical flow)**: https://arxiv.org/abs/2404.04808 (Dong & Fu, CVPR 2024)
- **StoryDiffusion (GSC baseline)**: https://arxiv.org/abs/2405.14775 (Zhou et al., NeurIPS 2024)
- **Flux (GFC baseline T2I)**: https://github.com/black-forest-labs/flux
- **GameNGen (DOOM)**: https://arxiv.org/abs/2408.14837
- **Oasis (Minecraft)**: https://oasis.decart.ai/
- **Genie**: https://arxiv.org/abs/2402.15391 (Bruce et al., ICML 2024)
- **GameGen-X**: https://arxiv.org/abs/2411.00769
- **SEED-Story**: https://arxiv.org/abs/2407.08683
- **MovieDreamer**: https://arxiv.org/abs/2407.16655
- **VideoAuteur (cosine loss inspiration)**: https://arxiv.org/abs/2501.06173
- **DreamSim**: https://arxiv.org/abs/2306.09344
- **TimeChat (sliding window scheme)**: https://arxiv.org/abs/2312.13496 (Ren et al., CVPR 2024)
- **Ponyo reference**: https://en.wikipedia.org/wiki/Ponyo
- **Carse, Finite and Infinite Games**: https://en.wikipedia.org/wiki/Finite_and_Infinite_Games

---

## 14. 一句话总结给你

AnimeGamer 把 generative infinite game 重构成 **MLLM predicting multimodal tokens in autoregressive fashion**，关键 trick 是 action-aware multimodal representation（visual + motion text + motion scope 三 channel 分离）+ three-stage training（tokenizer pretrain → MLLM predictor → decoder adaptation）。整体 philosophy 和你常说的 "next-token prediction is all you need" 完全契合——只是 token 这里是 game state。
