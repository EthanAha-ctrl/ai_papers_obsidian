---
source_pdf: Dita.pdf
paper_sha256: 79240f1fcb25a0c4962c09bb6e16c1cd9abcb816c7905814102f0ae3d0bbebb9
processed_at: '2026-08-03T22:41:42-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用大白话再讲一遍 Dita。

---

# 一句话版本

**别让一个小 MLP head 去猜 action，直接把 noisy action 塞进 Transformer 里，让它和 image patch 在 self-attention 里自己聊。**

---

# 问题是什么

现在做 generalist robot policy，主流两条路：

**路线 A：把 action 离散化**（OpenVLA 这派）
就类似把 action 当 token 来 predict。7 维 action 拆成 7 个 dimension，每个 dimension 切 256 个 bin，然后像 language model 那样 next-token predict。

问题：rotation 这种东西是连续的，你切 bin 切得再细也是离散的。而且不同 robot 的 action range 不一样，bin 的设计很 ad-hoc。

**路线 B：用一个小的 diffusion head**（Octo 这派）
主 Transformer 负责理解 vision + language，输出一个 embedding，然后挂一个小 MLP 或者小 DiT 在外面，用这个小网络去 denoise action。

问题：小网络看到的是 "压缩过的 embedding"，看不到 raw image patch。但 robot action 的精细控制恰恰需要看到 image 里的细节——比如杯子把手在哪个 patch、抽屉缝隙在哪个 patch。这些细节经过 Transformer fuse 成一个 embedding 后就被稀释了。

Dita 的观察很简单：**action 预测的本质，是 action 在问 image "我下一步该怎么动"。那为什么要先 compress image 再让 action 问？直接让它们在一个 attention layer 里对话不就行了。**

---

# Dita 怎么做的

架构其实非常 clean：

输入有四样东西拼成一个 sequence：
1. CLIP 编码的 language token（frozen，不更新）
2. DINOv2 编码的 image patch token（finetune，和主网络一起训练）
3. timestep embedding（告诉网络现在 denoise 到第几步了）
4. noised action token（加噪声的 action chunk，零填充到 hidden dim）

这四样东西 concat 成一条序列，喂进一个 LLaMA2-style 的 causal Transformer（12 层，hidden 768，总共 334M 参数）。Transformer 输出预测 noise。就这么简单。

**关键点：noised action token 和 image patch token 在同一个 self-attention 里交互。** 每一步 denoise，action token 都能重新 attend 到 raw visual patch。没有什么 "先 fuse 再 condition" 的 bottleneck。

对比一下：

- Octo：image → Transformer → 1个embedding → MLP head 反复看这个 embedding 去 denoise action
- Dita：image 和 noisy action 一起进 Transformer，action 每次都能直接看 image

这就像：你想问同事一个问题。路线 A 是先让同事把所有想法总结成一句话写下来，你再反复读这句话；路线 B 是你每次都直接跑去问同事。显然 B 更高效。

---

# 为什么这个设计 work

我的直觉是：robot policy 的核心 mapping 是 $\pi(a | o, l)$，从高维 visual signal 到低维 action。如果你先 compress $o$ 到 embedding 再 condition $a$，相当于在 embedding 维度上有个 bottleneck。直接让 action token attend image patch，相当于保留了 visual information flow 的全部带宽，让 Transformer 自己学哪些 patch 和 action 相关。

具体场景：你想让机器人去抓杯子。杯子把手的位置可能在 image 的某个具体 patch 里。如果是 Octo 那种设计，这个 patch 的信息先被压进 embedding，然后 MLP head 从 embedding 里 "捞" 这个信息。中间有信息损失。Dita 里 action token 直接 attend 到那个 patch，信息无损。

而且 action 是连续的，用 diffusion 自然处理 continuous action。不同 robot 的 action dimension 不一样？零填充就行，反正 noise 只加在有效 dimension 上。这就是 Dita 能 scale 到 OXE 这种 heterogeneous 数据集的原因。

---

# 数据结果怎么说

几个关键数字：

**SimplerEnv**（zero-shot real-to-sim 评估）：
Dita 在 variant 设置（背景变、物体位置变）下，pick coke can 79.1%，move near 52.1%，drawer 74-82%。OpenVLA 7B 参数在 variant 设置下是 54.5%、47.7%、17.7%。Dita 用 334M 吊打 7B。

**LIBERO-LONG**（long-horizon 任务）：
Dita 63.8%，OpenVLA 53.7%，Octo 51.1%。Long-horizon 任务最考验 visual grounding 和 action alignment，in-context conditioning 的优势最明显。

**CALVIN**（ABC→D，5步连续 subtask）：
Dita 平均完成 3.61 步，只用 single static RGB。作者自己的 diffusion head baseline 是 3.16 步。这个对比直接证明 in-context conditioning 的收益（+0.45 步）。

**ManiSkill2**（cluttered scene，camera view 变化）：
PickClutterYCB 这个最难任务，Dita 36%，diffusion head 24%，discretization 1%。Cluttered scene 对 visual grounding 要求最高，Dita 优势最大。

---

# 一些有意思的细节

**Denoising step 不需要太多**：训练用 1000 步 DDPM，推理用 DDIM 压到 20 步甚至 10 步，性能基本不掉。2 步还能跑 70%。原因很简单：action 是 7 维，比 image 生成那种几万维的 latent 小多了，不需要那么多 denoise step。这对 3Hz control frequency 很关键。

**2 frame observation 是 sweet spot**：1 帧不够看 motion，3 帧反而掉点（token 太多难收敛）。2 帧刚好能区分 workspace 里物体和 robot state。

**长 prediction horizon 帮助大**：预测 16 步 action chunk 比预测 2 步好很多，特别是复杂任务。这让我想到 Diffusion Policy 原论文的观察——predict action chunk 让 model "看到" 未来，planning 能力更强。

**LoRA 在 extreme variance 下会挂**：10-shot finetune 用 LoRA 只训练 11M 参数，在 background 变化、lighting 变化等 extreme disturbance 下完全失败。Full finetune（221M trainable）能扛住。这点很 practical——LoRA 不是银弹，容量不够就鲁棒不了。

**DINOv2 必须 end-to-end finetune**：DINOv2 训练在 web data，和 robot data 有 domain gap。所以学习率设 1e-5（比主网络小 10 倍）joint train。如果完全 frozen DINOv2，性能会掉很多。

---

# 为什么之前没人这么做

我猜原因有两个：

一是大家受了 VLM (PaliGemma, LLaVA) 那条线影响，觉得 "frozen 大 VLM + action head" 是 modular 的自然设计。π0 就是这样，用 PaliGemma 做 vision-language understanding，外面接 DiT 做 action generation。

二是 Octo 的 diffusion head 设计 "看起来够用了"，没让人意识到这是个 bottleneck。只有在 cross-embodiment dataset 上 scale 起来，碰到 diverse camera view、diverse action space、long-horizon task，这个 bottleneck 才暴露。

Dita 的贡献是把这个问题点出来，并证明 "把 denoising 移进 Transformer 内部" 是更优解。334M 参数、开源、简洁，结果比 7B 的 OpenVLA 还好。

---

# 我的 take

这篇 paper 给我的启示：**Architecture design 在某些 regime 下比 scale 重要。** 当现有架构有 fundamental bottleneck（这里是 embedding compression 信息损失）时，scale up 只是在错误的方向上堆 compute。先 fix architecture，再 scale，效率高得多。

Dita 给了一个 clean baseline，后续可以 scale up 到 1B+、加入 wrist camera、加更复杂的 observation history。这条路还能走很远。

References：
- Dita 主页：https://robodita.github.io
- OpenVLA 对比：https://openvla.github.io/
- Octo 对比：https://octo-models.github.io/
- DiT 原论文（in-context conditioning 灵感来源）：https://peebles.io/DiT

---

# Dita: Scaling Diffusion Transformer for Generalist VLA Policy 深度讲解

## 1. 核心定位与动机

Dita 是一篇来自 Shanghai AI Lab 的工作，核心 motivation 是解决现有 VLA (Vision-Language-Action) model 在 cross-embodiment dataset 上 generalization 受限的问题。Paper 主页：https://robodita.github.io

Karpathy 你应该熟悉 OpenVLA、Octo、RT-2 这条线。Dita 的关键观察是：**现有 generalist policy 在 action representation 上存在 bottleneck**：

- **Discretization 路线** (RT-1 [8], OpenVLA [32])：把每个 action dimension 离散化成 256 bins，丢失连续几何结构，对 rotation 这类操作特别不友好。
- **Diffusion head 路线** (Octo [72], π0 [5])：用一个 shallow MLP/小 DiT 作为 diffusion head，conditioning 来自 causal Transformer 输出的 fused embedding。问题在于 observation 的 fine-grained 信息（pixel-level patch token）被 early-fuse 成单个 embedding，diffusion head 看不到 raw visual token，无法 align action delta 与 visual nuance。

Dita 的核心 thesis：**让 diffusion denoising 直接发生在 Transformer 内部，通过 in-context conditioning，让 noisy action token 和 image patch token 在同一个 self-attention 计算图中交互**。

---

## 2. Architecture 深度解析

### 2.1 Multi-modal Input Tokenization

```
Language instruction → CLIP text encoder (frozen) → text tokens
Image observation (224×224) → DINOv2 (finetuned) → patch features
                  ↓
            Q-Former (depth=4, length=32) ← FiLM conditioning from text
                  ↓
            image features (32 tokens)
```

关键设计点：

1. **CLIP frozen**：language 这条线不更新，节省 compute。
2. **DINOv2 finetuned end-to-end**：作者强调 DINOv2 训练在 web data 上，与 robot-specific data 存在 domain gap，所以必须 joint optimization。learning rate 设为 1e-5（比 Q-Former 和 Transformer 的 1e-4 小 10 倍）。
3. **Q-Former with FiLM**：来自 BLIP-2 [33] 的设计，用 4 层 cross-attention 把 DINOv2 patch features 压缩到 32 个 token。FiLM (Feature-wise Linear Modulation, [57]) 把 text embedding 作为 affine 变换参数注入 image feature，实现 instruction-guided image feature selection。这一点很关键：Q-Former 不是单纯压缩，而是根据 language instruction 动态 query 视觉信息。

FiLM 的形式化：
$$
\text{FiLM}(x) = \gamma(c_{\text{lang}}) \odot x + \beta(c_{\text{lang}})
$$
其中 $\gamma, \beta$ 是从 language embedding 通过 MLP 学到的 scale 和 shift，$\odot$ 是 element-wise multiply。

### 2.2 Action Preprocess

End-effector action 表示为 7D vector：
- 3D translation ($\Delta x, \Delta y, \Delta z$)
- 3D rotation (Euler angle 或 axis-angle)
- 1D gripper position (open/close continuous)

为了与 image/language token 维度对齐，**用 zero padding 扩展到 hidden size 768**。Noise 只注入到前 7 维，padding 部分保持 zero。这点类似 RDT-1B [42] 的做法。

### 2.3 In-context Conditioning DiT — 核心 Architecture

这是 Dita 最核心的设计。Figure 2 三个 head 的对比：

**Left head (RT/OpenVLA)**: Discretization
```
[image tokens, lang tokens] → Causal Transformer → logits over 256 bins × 7 dims
```

**Middle head (Octo/π0)**: Diffusion head
```
[image tokens, lang tokens] → Causal Transformer → embedding e_t
                                                          ↓
                                                    MLP diffusion head
                                                    condition on e_t, t
                                                    denoise x^t → x^(t-1)
```

**Right head (Dita)**: In-context conditioning
```
[lang tokens | image tokens | timestep embedding | noised action tokens] 
                          ↓
                  Causal Transformer (LLaMA2-style, 12 layers, hidden 768)
                          ↓
                  predict noise ε_θ(c_lang, c_obs, t, x^t)
```

**In-context conditioning 的关键 intuition**：

在 Octo 这类设计里，image patch token 经过 causal Transformer 后被 fused 成固定 embedding，然后 small MLP head 反复 sample noise 做去噪。MLP head 看到的是 "compressed embedding"。

在 Dita 里，**noised action token 直接和 image patch token 共享同一个 self-attention 计算图**。每一步去噪时，action token 都能 attend 到 raw visual patch token。这就让 action 的细微变化 (action delta) 可以直接被 visual patch 的细微变化 (object pose 微调、background 干扰) 所 condition。

**为什么这个设计能 work？** 我的理解是：robot action prediction 本质是一个 $\pi(a | o_{t-1:t}, l)$ 的 mapping，其中 $o$ 是高维 visual signal，$a$ 是低维 action。如果先 compress $o$ 到 single embedding 再 condition action，相当于 bottleneck 在 embedding 维度。直接让 action token attend image patch，相当于保留了 visual information flow 的完整 bandwidth，让 Transformer 自己学哪些 patch 与 action 相关。

### 2.4 Transformer 规模

- LLaMA2-style architecture (RMSNorm, SwiGLU, RoPE)
- 12 个 self-attention block
- Hidden size = 768
- **总参数 334M，其中 trainable 221M**（CLIP frozen 占 113M）
- 这比 OpenVLA 的 7B 小一个数量级，比 Octo-Base (93M) 略大，比 RDT-1B (1B) 小很多

这点很重要：**Dita 用 334M 跑出比 7B OpenVLA 更好的 long-horizon 结果**，说明 architecture design 比 scale 更关键。

---

## 3. Training Objective 数学详解

### 3.1 DDPM 训练目标

Forward process (加噪)：
$$
q(\mathbf{x}^t | \mathbf{a}) = \mathcal{N}(\mathbf{x}^t; \sqrt{\bar{\alpha}_t} \mathbf{a}, (1 - \bar{\alpha}_t) \mathbf{I})
$$

其中：
- $\mathbf{a} \in \mathbb{R}^{T_a \times 7}$: action chunk (e.g., 16 step × 7 dim)
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$: cumulative noise schedule
- $\mathbf{x}^t$: noised action at step $t$

Reverse process (去噪)：
$$
p_\theta(\mathbf{a} | c_{\text{lang}}, c_{\text{obs}}) = \int p_\theta(\mathbf{a} | \mathbf{x}^0) \prod_{t=1}^{T} p_\theta(\mathbf{x}^{t-1} | \mathbf{x}^t)
$$

训练时 sample 一个 $t \in \{1, ..., T_{\text{train}}\}$，让 network 预测 noise $\epsilon$：

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{a}, \epsilon} \left[ \| \epsilon - \epsilon_\theta(c_{\text{lang}}, c_{\text{obs}}, t, \mathbf{x}^t) \|^2 \right]
$$

变量解释：
- $\epsilon \sim \mathcal{N}(\mathbf{0}, I)$: ground-truth Gaussian noise
- $\epsilon_\theta$: causal Transformer (Dita 主干)
- $c_{\text{lang}}, c_{\text{obs}}$: language 和 image 条件
- $t$: timestep embedding (sinusoidal positional encoding)
- $\mathbf{x}^t = \sqrt{\bar{\alpha}_t} \mathbf{a} + \sqrt{1 - \bar{\alpha}_t} \epsilon$: noised action

### 3.2 DDIM Inference (Eq. 1)

Paper 给的 inference 公式：

$$
\mathbf{x}^{t-1} = \alpha \left( \mathbf{x}^t - \gamma \mathcal{E}_\theta(c_{\text{lang}}, c_{\text{obs}}, t, \mathbf{x}^t) + \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}) \right)
$$

变量和上下标：
- $\mathbf{x}^{t-1}, \mathbf{x}^t$: 第 $t-1$ 步和第 $t$ 步的 noised action (上标是 timestep index)
- $\alpha, \gamma, \sigma$: noise scheduler [25] 中的参数 (DDIM coefficients)
  - $\alpha$: scaling factor，控制 state update 幅度
  - $\gamma$: step size，类似学习率
  - $\sigma$: stochastic noise std，决定 reverse process 的随机性
- $\mathcal{E}_\theta$: 学到的 noise prediction network
- $c_{\text{lang}}$: CLIP text token sequence
- $c_{\text{obs}}$: Q-Former 输出的 image feature token sequence
- $\mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$: 加性 Gaussian noise，$\sigma=0$ 时 DDIM 退化为 deterministic
- $\mathbf{I}$: identity matrix，维度等于 action chunk flatten 后的 dim

### 3.3 DDIM vs DDPM 训练/推理不对称

这是 paper 里一个值得关注的设计：
- **Training**: DDPM, $T_{\text{train}} = 1000$ timestamps
- **Inference**: DDIM, $T_{\text{eval}} = 20$ timestamps (zero-shot) 或 100 timestamps (finetune)

DDIM [69] 的核心思想是 reverse diffusion process 不需要严格 follow Markov chain，可以跳过中间步骤。这把推理速度从 1000 次前向降到 20 次，对 real-time control (3Hz) 至关重要。

---

## 4. Pretraining 设置

### 4.1 数据

- 用 OXE (Open X-Embodiment) [54] dataset，遵循 OpenVLA [32] 和 Octo [72] 的 dataset selection 和 weight assignment 方案
- Action normalized and filtered 类似 RT-X [54]

### 4.2 超参

| 超参 | 值 | 说明 |
|-----|----|----|
| $T_{\text{train}}$ (DDPM) | 1000 | 训练 timestamps |
| $T_{\text{eval}}$ (DDIM) | 20 | zero-shot inference |
| Observation frames | 2 | 历史帧数 |
| Action chunk length | 16 | 预测未来 16 步 action |
| Optimizer | AdamW | |
| LR (Transformer, Q-Former) | 1e-4 | |
| LR (DINOv2) | 1e-5 | |
| Batch size | 8192 | total |
| GPUs | 32× A100 | 256 samples/GPU |
| Training steps | 100,000 | |

---

## 5. Simulation Experiments 详解

### 5.1 SimplerEnv (Zero-shot Real-to-Sim)

SimplerEnv [37] 是评估 real robot 数据训练的 policy 在 simulation 中表现的 benchmark。Dita zero-shot 评估。

**Table 1 关键数据** (vs RT-1-X, Octo-Base, OpenVLA-7B):

| Method | coke_can match | coke_can variant | move_near match | move_near variant | drawer match | drawer variant |
|--------|----------------|------------------|------------------|-------------------|--------------|----------------|
| RT-1-X | 56.7% | 49.0% | 31.7% | 32.3% | 59.7% | 29.4% |
| Octo-Base | 17.0% | 0.6% | 4.2% | 46.2% | 22.7% | 1.1% |
| OpenVLA-7B | 16.3% | 54.5% | 47.7% | 47.7% | 35.6% | 17.7% |
| **Dita** | **76.4%** | **79.1%** | **49.1%** | **52.1%** | **82.0%** | **74.0%** |

(从 Table 6 倒推 zero-shot performance，对应 100-step inference)

关键观察：Dita 在 variant 设置下表现尤其强，说明 in-context conditioning 对 visual variance (background, texture, object position) 的 robustness 更好。

### 5.2 LIBERO (Finetuning Adaptation)

LIBERO [40] 4 个子集：
- SPATIAL: 物体相同 layout 变
- OBJECT: layout 相同物体变
- GOAL: 物体 layout 相同 task 变
- LONG: 10 个 long-horizon 任务

**Table 2 关键数据**:

| Method | SPATIAL | OBJECT | GOAL | LONG | Average |
|--------|---------|--------|------|------|---------|
| DP* | 78.3% | 92.5% | 68.3% | 50.5% | 72.4% |
| Octo | 78.9% | 85.7% | 84.6% | 51.1% | 75.1% |
| OpenVLA | 84.9% | 88.4% | 79.2% | 53.7% | 76.5% |
| **Dita** | 84.2% | **96.3%** | 85.4% | **63.8%** | **82.4%** |

LONG 上 +10% 是关键 insight：long-horizon 任务需要 model 把 visual observation 和 action sequence 紧密 align，in-context conditioning 恰好为此设计。

### 5.3 CALVIN (Long-horizon)

CALVIN [47] ABC→D 评估，1000 个 task chain，每个 5 个 subtask。

**Table 3 关键数据**:

| Method | Avg Len |
|--------|---------|
| RoboFlamingo (S+G RGB) | 2.47 |
| GR-1 (S+G RGB+P) | 3.06 |
| 3D Diffuser (S+G RGBD+P+Cam) | 3.27 |
| GR-MG (S+G RGBD+P) | 4.04 |
| SuSIE (S-RGB) | 2.69 |
| GHIL-Glue (S-RGB, finetuned gen model) | 3.69 |
| $\mathcal{E}_{\theta \sim s}^{\text{Diff}}$ (Dita's diff head baseline) | 3.16 |
| **Dita (Ours, S-RGB only)** | **3.61** |

关键对比：
- Dita 只用 single static RGB，达到 3.61，超过 GHIL-Glue 的 3.69 (大致持平)，但 GHIL-Glue 用了 SuSIE + InstructPix2Pix finetune，model 大得多
- Dita vs 自家 diffusion head baseline ($\mathcal{E}^{\text{Diff}}$): 3.61 vs 3.16，**+0.45 avg len**，直接证明 in-context conditioning 的收益
- Dita w/o pretrain vs w/ pretrain: 2.38 vs 3.61，**+1.23 avg len**，证明 OXE pretraining 的 transferability

### 5.4 ManiSkill2 (Camera View Generalization)

Paper 设计了一个新颖的 camera view generalization benchmark：从 300K random camera 池里采样 20 cameras 渲染每个 trajectory。

**Table 4 关键数据**:

| Method | Avg | PickC | StackC | S-YCB | C-YCB | EGAD |
|--------|-----|-------|--------|-------|-------|------|
| $\mathcal{E}^{\text{Disc}}$ (discretization) | 30.2% | 41.0% | 33.0% | 22.0% | 1.0% | 54.0% |
| $\mathcal{E}^{\text{Diff}}$ (diffusion head) | 58.6% | 86.0% | 76.0% | 37.0% | 24.0% | 70.0% |
| **Dita** | **65.8%** | 79.0% | 80.0% | **62.0%** | **36.0%** | 72.0% |

注意 PickClutterYCB (C-YCB) 这种 cluttered scene：Dita 36.0% vs Diff head 24.0% vs Disc 1.0%。Cluttered 场景对 visual grounding 要求极高，in-context conditioning 的优势在这里最明显。

### 5.5 Ablation 关键发现

**Observation length (Table 5)**:
- 2 frames > 1 frame > 3 frames
- 3 frames 反而下降，作者归因于 image token 数量增加导致 convergence 困难
- 2 frames + trajectory 32 是 sweet spot

**Trajectory length (Table 5)**:
- Trajectory length 越长越好 (2→4→8→16→32)
- 复杂任务 (PickClutterYCB) 从 9.0% (traj=2) → 25.0% (traj=16) → 36.0% (traj=32)
- 简单任务 (PickCube) 在 traj=4 后 plateau
- Intuition: 长预测 horizon 让 model anticipate target object 位置

**Denoising steps (Table 6)**:
- DDIM 10 步 ≈ 100 步 (Pick Coke match 82.0% vs 79.7%)
- 2 步还能 70.4%
- 关键 insight: action dim 远小于 image dim，不需要 image generation 那么多的 denoising step
- 这对 control frequency (3Hz) 至关重要

---

## 6. Real-Robot Experiments

### 6.1 Setup

- Franka Emika Panda (7-DoF)
- Robotiq 2F-85 gripper
- RealSense D435i (3rd person, ~1.5m away)
- 3Hz control frequency
- 10-shot finetuning (LoRA, 20k steps, batch 512)

### 6.2 任务设计

涵盖多个 skill category：
- Pick & Place (banana, kiwifruit)
- Pour (coffee beans, water)
- Stack (3 bowls, 3 Russian dolls)
- Pick & Rotation (banana insert, flip-top box)
- Pull & Push (drawer)
- Long-horizon (>3 steps): "close top drawer, open bottom drawer, put bowl in, close bottom drawer"

### 6.3 10-shot 结果 (Figure 5)

Dita 在 2-step tasks 上 63.8% 成功率，第二阶段贡献近一半。比 Octo 和 OpenVLA 在所有复杂任务上一致更好。

特别值得注意的 failure mode 分析 (Section 5.2)：
- OpenVLA: 能完成第一步，但 long-horizon 失败，比如 "completely misunderstanding the insert operation"
- Octo: rotation 任务更好，但 step 2 接近时常失败
- Dita: 能完成所有复杂任务，包括 3D rotation

### 6.4 LoRA vs Full Finetuning (Appendix C.2)

一个 paper 里 candid 的发现：**LoRA finetune 在 extreme variance 下失败**，因为只 5% (~11M) 参数 trainable，不足以 absorb image augmentation。Full finetune 在 extreme variance 下 20% 成功率，LoRA 0%。

这点对 practical deployment 很重要：10-shot setting 下 LoRA 不是银弹，full finetune 在 robustness 上更优。

---

## 7. Intuition Building: 为什么 In-context Conditioning Work？

让我把 Karpathy 你可能关心的核心直觉再总结一下：

### 7.1 信息瓶颈视角

Octo 的信息流：
```
image patches → Transformer → 1 fused embedding → MLP head → action noise
                    (bottleneck)         (lossy)
```

Dita 的信息流：
```
image patches ──┐
                ├──→ Transformer self-attention → action noise
action tokens ──┘
       (no bottleneck, full attention bandwidth)
```

每次 denoising step，action token 都能 re-attend 到 raw image patch token。这相当于把 denoising 视为 query 过程：action 问 image "我下一步该怎么动"。

### 7.2 Action Delta 与 Visual Delta 的 Alignment

Robot action 的精细控制需要 model 对 visual 中 object pose 的微小变化敏感。如果 visual 信息被 compressed 成 single embedding，pose 的 delta 在 embedding space 里被 diluted。In-context conditioning 让 action token 直接 attend 到包含 pose 信息的 patch token，delta 信号保留完整。

### 7.3 Scalability 视角

DiT [56] 在 image generation 上证明 Transformer 是 scalable diffusion architecture。Dita 把这个 scalability 引入 robot action space。Table 11 (Appendix D.2) 的 convergence analysis 显示 Dita 比 diffusion head baseline 收敛更快——这是 scalability 的直接证据。

### 7.4 Cross-embodiment 的统一性

不同 robot 的 action space 维度不同 (7D vs 11D bimanual)。Dita 通过 zero-padding + diffusion 自然处理 continuous action of any dim，而 discretization 路线需要为每个 robot 重新设计 bin。这是 Dita 能 scale 到 OXE 这种 heterogeneous dataset 的根本原因。

---

## 8. Limitations 与 Open Questions

Paper 没明说但能从数据推出来的 limitations：

1. **334M 容量限制**：Appendix B.2 提到 "our model (334M) lacks the capacity to fully accommodate the entire OXE dataset"，在 LIBERO-LONG 上 OXE-pretrained 反而不如 Droid-pretrained (63.8% vs 77.9%)。Scale up 应该能进一步提升。
2. **2-frame observation 上限**：Table 5 显示 3-frame 反而更差，说明 current architecture 在 observation length scaling 上有问题，可能需要 hierarchical attention 或 memory mechanism。
3. **3Hz control frequency**：对 fast dynamic task (e.g., peg-in-hole) 可能不够。但 Table 6 显示 2-step DDIM 还能 70%，说明有提速空间。
4. **无 wrist camera**：只用 3rd person view。Paper 说 architecture 支持额外 modality，但没实验。
5. **Action chunk 用 zero-padding**：不是最优，可能用 learnable embedding 或 FiLM-style action conditioning 更好。

---

## 9. 与相关工作的 Positioning

| 方法 | Action Rep | Denoising Location | Conditioning Style | Param |
|------|-----------|--------------------|--------------------|-------|
| RT-1 [8] | Discrete 256 bins | Transformer head | Early fuse | - |
| OpenVLA [32] | Discrete 256 bins | Transformer head | Early fuse (PaliGemma) | 7B |
| Octo [72] | Continuous | MLP head (out of Transformer) | Condition on embedding | 93M |
| π0 [5] | Continuous | DiT head (out of Transformer) | Condition on PaliGemma embedding | 3B |
| RDT-1B [42] | Continuous | DiT (separate) | Cross-attention | 1B |
| **Dita** | **Continuous** | **In-Transformer** | **In-context** | **334M** |

Dita 的独特性：denoising 不在 head 里，而是在主 Transformer 内部。这点和 π0 的设计哲学不同——π0 用 PaliGemma 做 VLM understand + DiT 做 action generation，是 modular design；Dita 是 unified design。

---

## 10. 我的 Intuition 总结

Karpathy，如果让我用一句话总结 Dita 的核心 contribution，那是：**把 diffusion denoising 从一个外部 head 移到 Transformer 内部，让 action token 直接和 visual token 在 self-attention 里交互**。

这个设计简单到让人怀疑 "为什么之前没人这么做"，但仔细想 Octo/OpenVLA 这条线是受到了 VLM (PaliGemma, LLaVA) 那种 "frozen VLM + action head" 思路的影响，自然走了 modular 路线。Dita 把 robot policy 视为 "diffusion over multi-modal token sequence"，自然走 unified 路线。

实验数据强烈支持这个设计：334M 参数下，在 LIBERO-LONG、CALVIN、ManiSkill2 cluttered 三个最难 benchmark 上都 SOTA，证明 architecture > scale 在某些 regime 下成立。

10-shot real robot adaptation 包括 "close top drawer, open bottom drawer, put bowl in, close bottom drawer" 这种 4-step long-horizon 任务，说明 in-context conditioning 学到的 representation 真的 transferable。

---

## References (Web Links)

- Dita project page: https://robodita.github.io
- OpenVLA: https://openvla.github.io/
- Octo: https://octo-models.github.io/
- OXE Dataset: https://robotics-transformer-x.github.io/
- DiT (Peebles & Xie): https://peebles.io/DiT
- DDPM (Ho et al. 2020): https://arxiv.org/abs/2006.11239
- DDIM (Song et al. 2020): https://arxiv.org/abs/2010.02502
- DINOv2: https://dinov2.github.io/
- BLIP-2 Q-Former: https://arxiv.org/abs/2301.12597
- LIBERO benchmark: https://libero-project.github.io/
- CALVIN benchmark: https://calvinrobot.github.io/
- ManiSkill2: https://maniskill2.github.io/
- SimplerEnv: https://simpler-env.github.io/
- RDT-1B: https://thuai-robotics.github.io/rdt-page/
- π0: https://www.physicalintelligence.company/blog/pi0
- GR-MG: https://arxiv.org/abs/2510.21102

---

需要我深入展开哪一块？比如 Q-Former + FiLM 的具体实现细节、DDIM noise scheduler 的 coefficient 推导、或 real-robot 任务设计的具体 subtask 分解？
