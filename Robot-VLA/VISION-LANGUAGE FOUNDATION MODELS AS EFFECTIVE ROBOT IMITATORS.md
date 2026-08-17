---
source_pdf: VISION-LANGUAGE FOUNDATION MODELS AS EFFECTIVE ROBOT IMITATORS.pdf
paper_sha256: ff81c416a352674cbb38c6e8f150c59e50efd1dffe9c0ae06c5fb8e7254e8f10
processed_at: '2026-08-13T01:33:06-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RoboFlamingo

## 一句话版本

VLM 已经在网上看了几亿张图配文字，它其实已经认识 "drawer"、"red block"、"slider" 这些东西长什么样了。RoboFlamingo 就是把这个已经学好的 VLM 拿过来，加个小 LSTM 头，在 robot data 上 fine-tune 一下，就能让 robot 听懂人话做任务。

---

## 为什么要搞这个？

过去几年 robot manipulation 这块，想用 language 来 control robot，大致有三条路：

**第一条路（SayCan）**：让 LLM 当大脑做规划，输出 "第一步开抽屉、第二步拿红块、第三步放到slider上" 这种 text plan，然后底下接一堆 pre-trained 的 skill policy 去执行。问题是 LLM 完全不接地气，它不知道 "红块" 在当前场景里在哪，skill policy 又听不懂人话，两边脱节。

**第二条路（RT-1）**：干脆 train 一个 35M 的 vision-language-action transformer，把 action 也 tokenize，vision+language+action 全塞进 token space 一起训。问题是这种 model 从 scratch 训，需要海量 robot data，而且没 benefit from web 上的 image-text pre-training。

**第三条路（RT-2、PaLM-E）**：拿一个 540B 的 PaLI-X 或者 PaLM-E，把 action 也变 token，跟 web 上的 image-text data 一起 co-fine-tune。效果好，但是——第一，model 是 Google 私有的；第二，co-fine-tune 540B model 要的资源一般人玩不起；第三，catastrophic forgetting 风险大。

RoboFlamingo 就是给普通研究者一个 **能复现、能玩得起、效果还不差** 的方案。

---

## 核心思路：解耦 perception 和 policy

作者 key insight 很简单——VLM 在 web pre-training 里学的是 **"看图配文字"**，它是 per-frame 的，它不懂视频序列，它不懂 action。所以与其逼 VLM 去干它不擅长的事（输出 action token、理解时序），不如让它干它擅长的事（理解当前帧 + 当前 instruction），然后把 decision 和 history 交给一个小 module 去 model。

具体就是：
- **VLM backbone**：每一步吃进 camera image + language instruction，吐出一个 vision-language fused representation
- **Policy head**：一个 LSTM + MLP，吃进 VLM 的 representation，维护 hidden state 记住历史，输出 7-DoF action

就这么简单。VLM 负责 "我看懂现在画面里有个红块在左边、抽屉关着、指令是开抽屉"，policy head 负责 "那我现在要往前伸、抓住 handle、往外拉"。

---

## 为什么这样设计 work？

### 1. VLM 的 grounding 是白捡的

OpenFlamingo 在 web 上看过几亿张 "red block" 的图配 "red block" 的字，它已经知道 "red block" 在 vision feature space 里长啥样。你从 scratch 训一个 robotics model，哪来这种 grounding？所以 RoboFlamingo 在 ABC→D 这种 unseen environment 上比 HULC 好 3.7 倍——因为 visual concept 是 environment-agnostic 的，换个 kitchen 布局，"drawer" 还是 "drawer"。

### 2. Gated cross-attention 保护 pre-trained knowledge

Flamingo 有个特别聪明的设计——cross-attention layer 前面加个 gate，gate 参数 $\alpha$ 初始化为 0。$\text{Tanh}(0) = 0$，意思是训练刚开始时 visual information 完全不进 LLM，model 行为 = 纯 LLM。随着训练 $\alpha$ 慢慢涨，visual info 才一点点融进来。

这避免了 random init 的 cross-attention layer 一上来就把 LLM 的 representation 搅乱。对 robotics 场景特别重要——你 fine-tune 时 robot data 才多少，24k steps，要是不 protect pre-trained weights，很容易 catastrophic forgetting。

### 3. Perceiver resampler 让 latency 可控

ViT 输出几百上千个 patch token，全塞进 LLM 代价太大。Perceiver resampler 用 64 个 learnable query 主动从 ViT 输出里 "pull" 信息，把 token 数压到固定 64。不管你 camera 分辨率多大，输出永远 64×d，推理时间可预测。这对 robot deployment 很关键，你不希望换个高清摄像头就 OOM。

### 4. Policy head 独立 → 可以 open-loop

因为 VLM 和 policy head 是分开的，你可以让 VLM 一次 inference，policy head 连续输出 K 步 action，不用每帧都跑 VLM。VLM inference 几十毫秒到几百毫秒，对 10-20Hz 的 robot control loop 太慢，open-loop 是 practical 的 workaround。Figure 3c 显示 retrain 一下能保住大部分性能。

---

## 实验结果讲了啥？

### Main result

在 CALVIN benchmark 上，连续做 5 个任务的平均成功长度：

| 方法 | Avg Len |
|---|---|
| HULC (SOTA) | 3.06 |
| RT-1 | 2.45 |
| **RoboFlamingo** | **4.09** |

2× improvement，而且 RoboFlamingo 只用了 language-annotated data（CALVIN 全量数据的 1%），HULC 用的是 full data。

### Generalization

**Vision generalization** (在 ABC 训练，D 测试，D 是没见过的 kitchen 布局)：RoboFlamingo 2.48 vs HULC 0.67。这个 gap 巨大，证明 VLM 的 grounding 能 transfer 到 unseen visual context。

**Language generalization** (用 GPT-4 生成 50 种 synonymous instruction，比如 "rotate red block right" 变成 "give a rightward spin to the red block")：RoboFlamingo 1.85，freeze embedding layer 后 2.12，HULC 1.82。VLM 对 paraphrase 鲁棒性还行，但 freeze embedding 后更好——因为 frozen word embedding 保留了 web pre-training 的语义结构，"spin" 和 "rotate" 在 embedding space 里近，fine-tune 后可能被 robot data 的 narrow vocabulary 破坏掉这个结构。

### Data efficiency

用 10% language data（0.1% 全量）训练：

| Model | Avg Len |
|---|---|
| 3B | 0.05 |
| 4B | 0.48 |
| 9B | 0.83 |

数据越少，model size 越重要。这就是 scaling law 的体现——大 model pre-training 压缩了更多 world knowledge，sample efficiency 更高。Full data 时 3B 和 9B 差不多，说明 CALVIN 的 bottleneck 在小数据 regime 是 representation quality，大数据 regime 是 policy capacity。

---

## 几个有意思的 ablation

### 1. Policy head 选啥？

| Policy Head | 效果 |
|---|---|
| MLP (只看当前帧) | 最差 |
| MLP (历史帧一起送 ViT) | 一般 |
| GPT (显式输入历史) | 好 |
| LSTM (隐式 hidden state) | 好 |

MLP w/ hist 把历史帧塞进 ViT 让 cross-attention 自己学时序——效果不行。作者解释：OpenFlamingo 只在 image-text pair 上 pre-training，没见过视频，它的 cross-attention 不会自动学会时序聚合。所以你需要一个 explicit temporal module。这跟 Tesla autopilot 当年的 insight 一样——per-frame understanding 不够，closed-loop task 需要 temporal reasoning。

GPT 和 LSTM 差不多，作者选 LSTM 因为简单。

### 2. Pre-training 重要吗？

- 不 load OpenFlamingo 权重从 scratch 训：大幅下降
- Freeze VLM 只训 policy head：也差

Pre-training 给你 grounding，但 robotics 需要的 representation 和 web image-text 不完全一样，所以也得 fine-tune。经典的 transfer learning trade-off。

### 3. Co-training 防 forgetting

纯 fine-tune 后，在 COCO image caption 上 CIDEr 从 75.7 掉到 0.005——VLM 的原始能力几乎全丢了。Co-train（一个 batch robot data，一个 batch VL data）后能保住 0.346，CALVIN 上只损失一点点（3.76 vs 4.09）。

这个实验对实际部署有意义：如果你希望 robot 一边干活一边能回答 "你看到啥了" 这种 VQA 问题，co-train 是必须的。

---

## 我的几点延伸思考

**1. Action representation 的选择**

RT-2 选择 action token 化，进 LLM token space。RoboFlamingo 选择 action 不进 LLM，用一个独立 head 输出。前者 unified 但 expensive，后者 modular 但 lose 了 VLM 的 implicit planning 能力。我觉得未来一个 promising 方向是用 VLM 的中间 representation 做 latent planning，类似 Diffusion Policy 的思路，既保留 grounding 又避开 action token 化的 complexity。

**2. Video-LLM 可能是下一步**

OpenFlamingo 是 image-text 的，不会时序，所以需要 LSTM 补。如果用 Video-LLaVA 这种 video pre-trained VLM，可能直接搞定时序，省掉 policy head 里的 temporal module。但 video-LLM 现在 inference 还慢，trade-off 不一定划算。

**3. Diffusion Policy 作为 policy head**

LSTM + MSE regression 假设 action distribution 是 unimodal 的，但 manipulation 里经常 multimodal——比如抓杯子可以从左边抓也可以从右边抓。Diffusion Policy 天然 handle multimodal，接在 VLM 后面应该更好。这是很自然的 follow-up。

**4. Real robot 是 obvious next step**

Paper 只在 CALVIN sim 上做，作者自己在 conclusion 里承认。Open X-Embodiment 数据集出来后，把 RoboFlamingo 在 real robot data 上 fine-tune 是 obvious 的 follow-up。Modular design 让这个 transfer 很 straightforward。

**5. Hierarchical extension**

现在 policy head 是 flat 的 LSTM。如果要做什么 "泡咖啡" 这种 long-horizon task，需要 hierarchical——上层 VLM 把 "泡咖啡" 拆成 "拿杯子、接水、放咖啡、加热" 这种 subgoal，下层 VLM + policy head 执行 subgoal。RoboFlamingo 的 modular design 已经为这个留了空间，把 language instruction 换成 subgoal 就行。

---

## Reference Links

- RoboFlamingo: https://github.com/RoboFlamingo/RoboFlamingo
- OpenFlamingo: https://github.com/mlfoundations/open_flamingo
- CALVIN: https://github.com/mees/calvin
- RT-2: https://arxiv.org/abs/2307.15818
- RT-1: https://arxiv.org/abs/2212.06817
- SayCan: https://arxiv.org/abs/2204.01691
- PaLM-E: https://arxiv.org/abs/2303.03378
- Flamingo (original): https://arxiv.org/abs/2204.14198
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

---

## 一句话总结

VLM 已经认识红块和抽屉了，别浪费这个 grounding 从 scratch 训 robot。把 VLM 当 perception module freeze 大部分参数，加个小 LSTM 头学时序和 action，1B 参数单 GPU 就能 train，效果比 540B 的 RT-2 方案还好。这就是 modular design 在 scaling law 时代的胜利——知道哪里 freeze、哪里 fine-tune 比 all end-to-end 更重要。

---

# RoboFlamingo: 用 Pre-trained VLM 做 Robot Imitation 的 Elegant Decoupling

## 1. High-Level Intuition

Andrej，这篇 paper 的核心 insight 我觉得非常符合你一贯喜欢的 "the bitter lesson" 思路：**与其从 scratch 训一个 robotics-specific architecture，不如 leverage 已经在 web-scale data 上学过 grounding 的 VLM，用最小化的 adaptation 把它接到 robot control 上**。但和 RT-2 那种把 action token 化、co-fine-tune 整个 540B PaLI-X 的暴力做法不同，RoboFlamingo 做了一件很 elegant 的事——**decouple vision-language understanding（perception/grounding）和 sequential decision-making（policy）**：

- **Perception module**：复用 OpenFlamingo 的 vision encoder + perceiver resampler + gated cross-attention decoder，处理单帧 vision-language alignment。这一部分受益于 web-scale pre-training，已经知道 "red block"、"drawer"、"slider" 这些 concept 长什么样。
- **Policy module**：一个轻量级 LSTM + MLP head，explicitly model temporal history 并 output 7-DoF action。

这样的 decoupling 有几个直接 consequence：
1. 只需要 train ~1B 参数（resampler + gated cross-attention + policy head），LLM backbone frozen，单张 A100 server 就能跑。
2. Perceiver resampler + gated cross-attention 是 OpenFlamingo 原本就为 few-shot vision-language 设计的 module，adapt 到 robotics 的成本极低。
3. Policy head 可以做 open-loop control（一次 inference 输出 multi-step action），不需要每帧都跑一遍 VLM。

这点和 PaLM-E/SayCan 的 hierarchy 不一样，和 RT-2 的 monolithic co-fine-tune 也不一样，是一个介于两者之间的 "modular but end-to-end trainable" 设计。

---

## 2. Architecture Deep Dive

### 2.1 问题形式化

Task 被建模为 **Goal-Conditioned POMDP (GC-POMDP)**：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{T}, \rho_0, \mathcal{L}, \phi, f \rangle$$

- $\mathcal{S}$：state space（hidden，agent 看不到）
- $\mathcal{O}$：observation space（两个 camera view $I_t, G_t$ + proprioception）
- $\mathcal{A}$：7-DoF action（end-effector $\Delta$ pose + gripper open/close）
- $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$：environment dynamics
- $\rho_0$：initial state distribution
- $\mathcal{L}$：长度 $M$ 的 free-form language instruction $l$
- $\phi(s) \in \{0,1\}$：task success indicator
- $f(o|s)$：observation function

Policy 是 $\pi_\theta(a|o, l)$，由两部分组成：
- $X_t = f_\theta(o_t, l)$：Flamingo backbone 输出的 fused vision-language representation
- $a_t = p_\theta(X_t, h_{t-1})$：policy head 输出 action，$h_{t-1}$ 是上一时刻的 hidden state

### 2.2 Vision Encoder：ViT + Perceiver Resampler

两视角图像 $I_t, G_t$ 通过 ViT 编码为 visual token sequence：

$$\hat{X}_t^v = \text{ViT}(I_t, G_t)$$

其中 $\hat{X}_t^v = (\hat{x}_{t1}^v, \dots, \hat{x}_{tN}^v)$，$N$ 是 ViT 输出的 patch token 数量（通常几百到上千）。

接下来 **Perceiver Resampler**（来自 DeepMind 的 Flamingo paper，灵感来自 Perceiver）把 $N$ 个 visual tokens 压缩到 $N_r$ 个（通常 64），用 learnable query 做 cross-attention：

$$K_R = \hat{X}_t^v W_K^R, \quad V_R = \hat{X}_t^v W_V^R, \quad X_t^v = \text{softmax}\left(\frac{Q_R K_R^T}{\sqrt{d}}\right) V_R$$

变量含义：
- $Q_R \in \mathbb{R}^{N_r \times d}$：**learnable latent queries**，是 resampler 自己的参数，不依赖于输入。这是 Perceiver 的核心 trick——固定数量 learnable tokens 主动从输入里 "pull" 信息。
- $W_K^R, W_V^R \in \mathbb{R}^{d_v \times d}$：linear projection，$d_v$ 是 ViT 输出维度，$d$ 是 hidden dim
- $K_R, V_R$：从 visual tokens 投影得到的 key/value
- $\sqrt{d}$：scaled dot-product attention 的标准 normalization

**Intuition**：这个 design 的好处是无论输入图像分辨率多大、patch 数 $N$ 多大，输出永远是 $N_r \times d$ 的 fixed-size representation。对 robotics 尤其重要——我们希望推理 latency 可预测，不希望 camera 分辨率变化就改 LLM 的 context length。

### 2.3 Feature Fusion Decoder：Frozen LLM + Gated Cross-Attention

这是 Flamingo 最关键的 design。Decoder 有 $L$ 层，每层包含：

1. **Gated Cross-Attention Layer**：language token 作 query，visual token 作 key/value
2. **Standard Self-Attention + MLP**：从 pre-trained LLM 复制，**frozen**

形式化（第 $l$ 层）：

$$\hat{X}_t^l = \text{Tanh}(\alpha) \cdot \text{MLP}\left(A(X_t^l W_Q^C, X_t^v W_K^C, X_t^v W_V^C)\right) + X_t^l$$

$$X_t^{l+1} = \text{MLP}\left(A(\hat{X}_t^l W_Q^S, \hat{X}_t^l W_K^S, \hat{X}_t^l W_V^S)\right) + \hat{X}_t^l$$

变量含义：
- $X_t^l$：第 $l$ 层输入，$X_t^1 = X$（language instruction 的 embedding）
- $X_t^v$：vision tokens（来自 resampler）
- $W_Q^C, W_K^C, W_V^C \in \mathbb{R}^{d \times d}$：cross-attention 的 Q/K/V projection，**trainable**
- $\alpha \in \mathbb{R}$：**gating parameter，scalar，trainable**，初始化为 0
- $\text{Tanh}(\alpha)$：gate 值，$\text{Tanh}(0) = 0$，所以训练开始时 cross-attention 输出被 zero out
- $W_Q^S, W_K^S, W_V^S$：self-attention 的 Q/K/V，**frozen**（继承自 LLM）

**关键 intuition**：$\alpha$ 初始化为 0 是 Flamingo 的 masterstroke。训练开始时模型 = 纯 LLM，cross-attention 不起作用，所以模型输出依然是 fluent language。随着训练进行，$\alpha$ 慢慢学到合适的值，visual information 才被逐步引入。这就避免了 random-init 的 cross-attention layer 把 pre-trained LLM 的 representation 破坏掉（catastrophic interference）。

这个 design 在 robotics 场景下尤其有意义：我们希望 VLM 已经学到的 grounding ability 不被 robotics fine-tune 破坏。Appendix B.1 的 co-training 实验直接验证了这一点——纯 fine-tune 后 COCO CIDEr 从 75.7 掉到 0.005，但 co-train 能保住 0.346。

### 2.4 Policy Head：Max Pooling + LSTM + MLP

Policy head 把 decoder 输出 $X_t^L \in \mathbb{R}^{M \times d}$（$M$ 是 instruction 长度）转成 action：

$$\tilde{X}_t = \text{MaxPooling}(X_t^L)$$

$$h_t = \text{LSTM}(\tilde{X}_t, h_{t-1})$$

$$a_t^{\text{pose}}, a_t^{\text{gripper}} = \text{MLP}(h_t)$$

变量含义：
- $\tilde{X}_t \in \mathbb{R}^d$：对 $M$ 个 language token 做 max pooling，得到 per-step representation
- $h_t$：LSTM hidden state，编码 history
- $a_t^{\text{pose}} \in \mathbb{R}^6$ 或 $\mathbb{R}^7$：end-effector relative pose
- $a_t^{\text{gripper}} \in \{0, 1\}$：gripper open/close

**Intuition on max pooling**：这里用 max pooling 而不是 mean pooling 或 attention pooling 是一个 interesting choice。Max pooling 选每个 dimension 上最 active 的 token，相当于让最 "salient" 的 word/feature dominate。对 language-conditioned task，比如 "rotate red block right"，可能 "red" 这个 token 在 vision-grounding 维度上最 active，max pooling 让它的 feature 流到 LSTM。

### 2.5 Training Objective

$$\ell = \sum_t \text{MSE}(a_t^{\text{pose}}, \hat{a}_t^{\text{pose}}) + \lambda_{\text{gripper}} \text{BCE}(a_t^\text{gripper}}, \hat{a}_t^{\text{gripper}})$$

- $\hat{a}_t^{\text{pose}}, \hat{a}_t^{\text{gripper}}$：demonstration 的 ground-truth action
- $\lambda_{\text{gripper}}$：gripper loss 的权重，用来 balance pose（continuous regression）和 gripper（binary classification）的 scale difference

Imitation learning 的最大 likelihood objective（来自 Section 3 公式 1）：
$$\ell = \mathbb{E}_{(\tau, l)_i \sim \mathcal{D}}\left[\sum_{t=0}^{|\tau|} \log \pi_\theta(a_t | o_t, l)\right]$$

其中 $\mathcal{D} = \{(\tau, l)_i\}_{i=0}^D$，$D$ 是 trajectory 数量，$\tau = \{(o_t, a_t)\}$ 是 state-action pair sequence。

---

## 3. CALVIN Benchmark 实验分析

CALVIN 是 long-horizon language-conditioned manipulation 的标准 benchmark：
- 4 个 environment splits (A, B, C, D)，每个 6 hours human teleoperation，~2M steps
- 只有 **1% data 有 language annotation**（~24k steps）
- 34 个 distinct tasks，1000 个 unique instruction chains
- Evaluation：连续完成 up to 5 个 sequential instructions，"Avg Len" 表示平均完成几个 task

### 3.1 Main Results (Table 1 解读)

| Method | Training | Test | 1 | 2 | 3 | 4 | 5 | Avg Len |
|---|---|---|---|---|---|---|---|---|
| MCIL | ABCD (Full) | D | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 |
| HULC | ABCD (Full) | D | 0.889 | 0.733 | 0.587 | 0.475 | 0.383 | 3.06 |
| RT-1 | ABCD (Lang) | D | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 |
| **RoboFlamingo** | ABCD (Lang) | D | **0.964** | **0.896** | **0.824** | **0.740** | **0.66** | **4.09** |

**Key observations**：
1. RoboFlamingo 用 **language-annotated data only**（1%），却打败了用 full data 的 HULC（4.09 vs 3.06 Avg Len），这是 ~2× improvement。
2. 越到后面的 task，drop 越小。Task 5 success rate：RoboFlamingo 0.66 vs HULC 0.383。后面 task 的 initial state 严重依赖前面 task 的 ending state，state distribution 漂移大，需要 strong vision-language grounding 才能 generalize。VLM 的 grounding ability 在这里 shine。
3. RT-1 这种 action-tokenized transformer 在 CALVIN 上反而比 HULC 差（2.45 vs 3.06），暗示 RT-1 的 design 可能更适合 web-scale robotics data，不适合 CALVIN 这种小数据场景。

### 3.2 Zero-shot Generalization

**Vision generalization (ABC → D)**：
| Method | Avg Len |
|---|---|
| MCIL | 0.31 |
| HULC | 0.67 |
| RT-1 | 0.90 |
| **RoboFlamingo** | **2.48** |

D 是 unseen environment（不同 object layout、不同 camera 背景）。RoboFlamingo 的 2.48 vs HULC 的 0.67 是 ~3.7× gap。这验证了 pre-trained VLM 对 unseen visual context 的 robustness——CLIP-style image-text pretraining 学到的 visual concept 是 environment-agnostic 的。

**Language generalization (Enriched instructions)**：用 GPT-4 给每个 task 生成 50 个 synonymous instructions（Table 4 有 example）。结果：
| Method | ABCD→D Enriched Avg Len |
|---|---|
| HULC | 1.82 |
| RT-1 | 0.86 |
| RoboFlamingo | 1.85 |
| RoboFlamingo (freeze-emb) | 2.12 |

**Interesting point**：freeze embedding layer 后 RoboFlamingo 从 1.85 → 2.12。作者解释，因为 RoboFlamingo 直接用 word token，synonymous 句子的 token distribution 变化大；而 HULC 用 frozen sentence transformer 得到 sentence embedding，对 paraphrase 鲁棒。Freeze embedding 后 word embedding 保留 pre-trained 的语义结构，对 synonym 更鲁棒。

---

## 4. Ablation Studies 深度解析

### 4.1 Policy Head 架构对比 (Figure 3a)

四种 design：
- **(a) MLP w/o hist**：只用 current observation，丢弃 history → 性能最差，证明 history 信息关键
- **(b) MLP w hist**：把历史 frames 也送进 ViT，加 position embedding，让 cross-attention 自己学 temporal aggregation → 比 w/o hist 好但远不如 LSTM/GPT
- **(c) GPT**：decoder-only transformer policy head，explicitly 输入 visual history tokens
- **(d) LSTM**：implicit memory via hidden state

**关键 insight**：MLP w/ hist 比 LSTM/GPT 差很多。作者的解释是 **OpenFlamingo 只在 image-text pair 上 pre-trained，没见过 sequential video data，所以它的 cross-attention 不会自动学会 temporal aggregation**。这暗示 VLM 的 vision grounding 是 "frame-level" 的，要把它变成 "video-level" 需要额外的 temporal module。

这让我想到你以前在 Tesla 讲过的——driving 这种 closed-loop task需要 strong temporal reasoning，单纯 per-frame understanding 不够。RoboFlamingo 用 LSTM 这个最朴素的方法来 fill 这个 gap。

### 4.2 Pre-training 重要性 (Figure 3b)

- **No VL Pre-train**：不 load OpenFlamingo 的 resampler 和 cross-attention 权重，从 scratch 训 → 性能大幅下降
- **No VL Finetune**：完全 freeze VLM，只 train policy head → 也很差，证明必须 fine-tune VLM 来 adapt 到 robotics domain

**Intuition**：VLM pre-training 提供 visual grounding（"red block 是什么样的"），但 robotics 需要的 representation 和 web image-text 不完全一样（比如需要关注 gripper pose、affordance），所以需要 fine-tune。这是 representation transfer 的经典 trade-off。

### 4.3 Model Size & Data Efficiency (Table 3)

用 10% language annotated data（仅 0.1% of full CALVIN data）训练：

| Backbone | Total Param | Avg Len |
|---|---|---|
| M-3B | 3B | 0.05 |
| M-3B-IFT | 3B | 0.13 |
| G-4B | 4B | 0.48 |
| G-4B-IFT | 4B | 0.55 |
| M-9B | 9B | 0.83 |

**Clear scaling law**：数据越少，model size 越重要。9B 比 3B 高 ~16×。这和 LLM 的 scaling law 一致——大 model 的 sample efficiency 更好，因为 pre-training 已经压缩了更多 world knowledge。但在 full data 下（Table 2），3B 和 9B 性能接近（4.02 vs 3.87），说明 CALVIN 的 bottleneck 在 small data regime 是 representation quality，在 full data regime 是 policy capacity。

### 4.4 Instruction Fine-tuning 的作用

M-3B-IFT vs M-3B（IFT = Instruction Fine-Tuning）：4.09 vs 3.94（Avg Len）
G-4B-IFT vs G-4B：3.79 vs 3.67

**Intuition**：LLM 经过 instruction tuning 后，对 "free-form instruction" 这种 input format 更熟悉，能 better parse language goal。这对 language-conditioned robotics 是直接的 benefit。

---

## 5. Open-Loop Control 的 Practical Value

这是 paper 的一个 underappreciated 贡献。RoboFlamingo 因为 decoupled policy head，可以做 **open-loop control**：一次 VLM inference 输出 $k$-step action sequence，直接执行。这避免了每帧都要跑 VLM inference 的 latency 问题。

Figure 3c 显示：直接 open-loop 不 retrain 性能掉得厉害；用 **jump-step demonstration**（每 $k$ 步采一次样）retrain 后能保住大部分性能。

**Practical intuition**：VLM inference 即使在 A100 上也要几十到几百 ms，对 7-DoF 闭环控制（通常 10-20 Hz）不够。Open-loop 把 VLM 当 high-rate planner，policy head 做 low-rate predictor，是一个很 practical 的 deployment strategy。

---

## 6. Co-Training 防 Catastrophic Forgetting (Appendix B.1)

这是个 important 的 follow-up 实验。纯 fine-tune 后：
- COCO CIDEr：75.7 → 0.005
- VQAv2 Acc：40.92 → 4.09

VLM 的原始 vision-language ability 几乎全部 lost。Co-train（每个 batch 一半 robot data 一半 VL data）后：
- COCO CIDEr：0.346
- VQAv2：36.37
- CALVIN Avg Len：3.76 vs 4.09（fine-tune only）

Co-train 保住了大部分 VL ability，只牺牲一点 robotics 性能。这是一个很实用的 insight——如果你希望 robot policy 还能同时做 VQA（比如 human 问机器人 "你看到什么了"），co-train 是必要的。

---

## 7. 与 RT-2 / PaLM-E / SayCan 的对比 (我的联想)

让我把这些方法放在一个 spectrum 上：

| Method | VLM Use | Temporal | Trainable Param | Data Need | Open Source |
|---|---|---|---|---|---|
| SayCan | LLM as high-level planner | None (discrete skill) | 0 (frozen LLM) | Pre-defined skills | ✓ |
| PaLM-E | VLM end-to-end, output plan text | Implicit (in LLM) | 540B | Web VL + robot | ✗ |
| RT-2 | VLM end-to-end, output action tokens | Implicit | 540B (PaLI-X) | Web VL + robot | ✗ |
| RoboFlamingo | VLM per-frame + LSTM head | Explicit (LSTM) | ~1B | Robot only | ✓ |

**RT-2 的哲学**：action 也 token 化，让 VLM 直接生成 action token sequence。优点是 unified interface，缺点是必须 co-fine-tune 在 web-scale VL data 上，否则 catastrophic forgetting。

**RoboFlamingo 的哲学**：VLM 做 perception，policy head 做 control。Action 不进 LLM token space，避开 token 化的 complexity。Trade-off 是失去 VLM 的 implicit planning ability，必须 explicit model history。

我猜测未来 work 会探索一个中间路线：用 VLM 的内部 representation 做 latent planning（类似 Diffusion Policy 的 latent space），既保留 VLM grounding 又避免 action token 化。

---

## 8. 局限和未来方向

Paper 自己也提到几个 limitations：

1. **No real-robot experiments**：只在 CALVIN 仿真上验证。不过 Open X-Embodiment 的出现让 real-robot fine-tune 变得可行，作者在 conclusion 里提到这是 obvious next step。
2. **Instruction-conditioned only**：每步都要 language instruction，不像 BC-Z 那种 visual-goal-conditioned。对纯 visual demo 的 generalization 没测。
3. **LSTM 的 long-horizon 局限**：CALVIN 最多 5 步 task sequence，LSTM 够用。真 long-horizon（几十步）LSTM 会 struggle，需要 transformer-based policy head。
4. **No closed-loop error correction**：因为是 imitation learning + open-loop capable，没有 explicit error recovery 机制。Failed task 后没有 re-plan 能力。

**我的一些进一步联想**：
- 把 policy head 换成 **Diffusion Policy** 可能更好——diffusion 对 multi-modal action distribution 自然 handle，而 LSTM + MSE regression 假设 unimodal。
- 用 **Video-LLM**（比如 Video-LLaVA, LLaVA-NeXT-Video）替代 OpenFlamingo 可能省掉 LSTM——video pre-trained VLM 直接 understand temporal。但 video-LLM 现在 inference 还太慢。
- **Hierarchical structure**：上层 VLM 做 task plan，下层 VLM 做 per-step action，是 natural extension。但 paper 的 modular design 已经为这个留了空间。
- **Action chunking**（ACT, Diffusion Policy 的 trick）：policy head 输出 multi-step action 而不是 single-step，能 reduce compounding error。Paper 的 open-loop control 实验已经在往这个方向走。

---

## 9. Web Links for Reference

- **RoboFlamingo GitHub**: https://github.com/RoboFlamingo/RoboFlamingo
- **OpenFlamingo**: https://github.com/mlfoundations/open_flamingo
- **Original Flamingo paper (Alayrac et al., 2022)**: https://arxiv.org/abs/2204.14198
- **CALVIN benchmark**: https://github.com/mees/calvin
- **CALVIN paper (Mees et al., 2022)**: https://arxiv.org/abs/2112.03227
- **HULC paper**: https://arxiv.org/abs/2112.10374
- **RT-1 paper**: https://arxiv.org/abs/2212.06817
- **RT-2 paper**: https://arxiv.org/abs/2307.15818
- **PaLM-E paper**: https://arxiv.org/abs/2303.03378
- **SayCan paper**: https://arxiv.org/abs/2204.01691
- **VIMA paper**: https://vimalabs.github.io/
- **R3M paper**: https://arxiv.org/abs/2203.12601
- **Voltron paper**: https://arxiv.org/abs/2302.12766
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **Diffusion Policy (重要的未来方向 reference)**: https://diffusion-policy.cs.columbia.edu/
- **Perceiver (resampler 灵感来源)**: https://arxiv.org/abs/2103.03206
- **MPT LLM**: https://www.mosaicml.com/blog/mpt-7b
- **GPT-NeoX**: https://arxiv.org/abs/2204.06745
- **LLaMA**: https://arxiv.org/abs/2302.13971
- **BC-Z (visual goal conditioning)**: https://arxiv.org/abs/2202.05129

---

## 10. 一句话总结 Intuition

RoboFlamingo 的核心 lesson 是：**Pre-trained VLM 给你 "perception grounding" for free，但 "temporal decision-making" 还需要 explicit module**。把两者 decouple 让你能用 1B trainable parameter + 单 GPU 训练就 beat RT-2 这种 540B co-fine-tune monster。这是 modular design 在 scaling law 时代的胜利——不是所有事都要 end-to-end，知道哪里 freeze、哪里 fine-tune 才是 engineering 的 art。
