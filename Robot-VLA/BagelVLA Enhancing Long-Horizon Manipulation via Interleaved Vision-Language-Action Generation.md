---
source_pdf: BagelVLA Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action
  Generation.pdf
paper_sha256: 691d30709ce69fbd23c4366b41d30dc06ce5b3b406e36afa3f813e7c09b43cd0
processed_at: '2026-07-18T13:45:46-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BagelVLA: Interleaved Vision-Language-Action 深度解读

Andrej 你好,这篇 paper 的核心 insight 我觉得非常对 robot learning 的胃口 — 它把"思考(textual planning)"、"想象(visual forecasting)"和"行动(action generation)"三件事在一个 transformer 里串成一条因果链,而不是各自独立。让我从直觉到技术细节完整 build up。

---

## 1. 一句话的 intuition

长期以来 VLA 模型都把 high-level reasoning 塞进一个 black-box policy $\pi(a | v, L)$,但 long-horizon 任务里最致命的是 **task decomposition 错位** — 比如"红→黄→蓝→绿 stack"这种任务,模型如果不在 latent space 里显式地"说"出"我现在该抓红的",low-level action 就会 collapse 到平均 action distribution。BagelVLA 把这个 chain 显式化:

$$\text{instruction} \rightarrow \text{subtask text} \rightarrow \text{keyframe image} \rightarrow \text{action chunk}$$

每一步都是 next-step 的条件,而不是平行 head。

---

## 2. Architecture: Mixture of Transformers (MoT)

整体结构借鉴自 [Bagel](https://arxiv.org/abs/2505.14683)(ByteDance 的 unified understanding+generation model)。三个 expert 通过 self-attention 互联:

| Expert | Size | Input | Output | Encoder | Loss |
|---|---|---|---|---|---|
| Understanding (LLM) | 7B | Image/Text | Text | SigLIP2 + MLP | Cross-Entropy |
| Generation | 7B | Image (VAE) | Image | FLUX VAE + MLP | MSE (Flow Matching) |
| Action | 2B | Proprio/Action | Action | MLP | MSE (Flow Matching) |

**关键技术细节**:
- Understanding & Generation 共享 Qwen2.5-LLM-7B ([Qwen2.5 tech report](https://arxiv.org/abs/2412.15115)) 的骨架,但用不同 visual encoder — 一个 SigLIP2 用于语义理解,一个 FLUX VAE 用于像素级生成。
- Action expert 是缩小版 — MLP intermediate size 砍到 1/5,只剩 2B 参数。这是为了 KV-cache 友好,让 action chunk 能以更高频率 rollout(72Hz async,40Hz sync)。
- 三个 expert 通过 MoT 在 self-attention 层级互联 — action expert 可以 attend 到 generation expert 当前的 noisy latent $v_{t+k}^\tau$。

这跟 [VILA-X](https://arxiv.org/abs/2507.23682)、[F1](https://arxiv.org/abs/2509.06951)、[UniCod](https://arxiv.org/abs/2510.10642) 这些 MoT-based VLA 不同之处在于:**前两个 expert 都有真实预训练**(Bagel),而不是随机初始化后从 VLA data 里学。这是后面 generalization 数据的关键。

---

## 3. Interleaved Planning: 因式分解的目标函数

### 3.1 形式化

给定 global instruction $L$ 和当前观测 $v_t$,模型输出三元组 $(l_t, v_{t+k}, a_t)$:
- $l_t$ — 当前 subtask 的文本描述(比如 "Pick up the blue block")
- $v_{t+k}$ — keyframe,即完成 subtask 后的视觉状态
- $a_t$ — action chunk

联合分布按因果序分解:

$$
\mathcal{I} = \max_\theta \mathbb{E}_\mathcal{D} \log p_\theta(l_t | v_t, L) \cdot p_\theta(v_{t+k} | v_t, L, l_t) \cdot p_\theta(a_t | v_t, L, l_t, v_{t+k})
$$

变量含义:
- $\theta$ — 所有 expert 的共享参数
- $L$ — global instruction(整个任务的语言描述)
- $v_t$ — current observation(image + proprioception)
- $l_t$ — subtask label,自动标注(用 [Seed-1.5-VL-thinking](https://arxiv.org/abs/2505.07062) 合成)
- $v_{t+k}$ — keyframe,k 表示 subtask 完成时刻
- $a_t$ — action chunk(7~48 维不等)

这个 factorization 的 intuition 是 **causal dependency**:语言是 world model 的 condition,视觉是 inverse dynamics 的 condition。如果用 joint entropy 看,$H(a | v_t, L, l_t, v_{t+k}) \ll H(a | v_t, L)$ — 给了"将要看到的画面"再生成 action,几乎就是个 inverse dynamics 问题,极简单。

### 3.2 为什么比 black-box VLA 好

可以对比一下 [OpenVLA](https://arxiv.org/abs/2406.09246) / [RT-2](https://arxiv.org/abs/2307.15818) 的 formulation:$\max_\theta \log p_\theta(a_t | v_t, L)$。这里 $L$ 是高熵的(一个长指令隐含多 stage),模型必须 implicitly 同时做 decomposition + mapping,梯度信号被稀释。

BagelVLA 把 $l_t$ 显式化后,language head 被纯 language CE loss 训练,保留了 VLM 的 reasoning capability;action head 只需解决 well-conditioned 的 IDM 问题。

---

## 4. Dual Flow-Matching: 三种 conditioning 方案

这是 paper 最精彩的部分。两个 FM process — 一个生成 keyframe image($\mathcal{L}_v$),一个生成 action chunk($\mathcal{L}_a$)— 怎么耦合?

### 4.1 Flow Matching 复习

[Flow Matching](https://arxiv.org/abs/2210.02747) 学习一个 velocity field $v_\theta(x^\tau, \tau)$,把 noise distribution $p_0$ 平滑传输到 data distribution $p_1$:

$$x^\tau = (1-\tau) x^0 + \tau x^1, \quad x^0 \sim p_0, \quad x^1 \sim p_{\text{data}}
$$

其中 $\tau \in [0, 1]$ 是 flow timestep(0 表示噪声,1 表示 data)。模型预测 velocity $\hat{v}(x^\tau, \tau) \approx x^1 - x^0$,用 MSE 训练:

$$\mathcal{L} = \mathbb{E}\left[\|v_\theta(x^\tau, \tau) - (x^1 - x^0)\|_2^2\right]
$$

### 4.2 三种 scheme

**Scheme 1: Complete Denoise** (经典 WM + IDM)

$$
\mathcal{L}_v = \mathbb{E}\left[\|v_{v,\theta}(L, v_t, l_t, \tau, v_{t+k}^\tau) - (v_{t+k}^1 - v_{t+k}^0)\|_2^2\right], \quad v_{t+k}^\tau = (1-\tau)v_{t+k}^0 + \tau v_{t+k}^1
$$

$$
\mathcal{L}_{a1} = \mathbb{E}\left[\|v_{a,\theta}(L, v_t, l_t, v_{t+k}^{\tau=1}, \tau, a_t^\tau) - (a_t^1 - a_t^0)\|_2^2\right]
$$

这里 $v_{t+k}^{\tau=1}$ 表示 action expert 看到的是 fully denoised image。训练时 append ground truth keyframe 模拟这个状态。推理时需要 $N_1 + N_2$ 步 denoise。**问题**:domain shift 让生成 image 偏离 training distribution,action policy 在 OOD image 上表现差;加上 latency 太高(6.04s/chunk,见 Table 4)。

**Scheme 2: Joint Denoise**

$$
\mathcal{L}_{a2} = \mathbb{E}\left[\|v_{a,\theta}(L, v_t, l_t, v_{t+k}^\tau, \tau, a_t^\tau) - (a_t^1 - a_t^0)\|_2^2\right]
$$

Action expert attend 到 noisy image $v_{t+k}^\tau$。$N$ 步同步 denoise。**问题**:intermediate noisy states 在 test 时是 OOD(因为 test scene 颜色、布局变了),policy 退化。Latency 2.90s/chunk。

**Scheme 3: Single-Step Denoise** (默认)

$$
\mathcal{L}_{a3} = \mathbb{E}\left[\|v_{a,\theta}(L, v_t, l_t, v_{t+k}^{\tau=0}, \tau, a_t^\tau) - (a_t^1 - a_t^0)\|_2^2\right]
$$

Action expert 只看 keyframe denoise 的**第一步** $\tau=0$ 的 KV cache。这意味着 action 实际上 condition 在 **纯噪声** 或 **v_t-initialized noise** 上!推理时只需 $1 + N_2$ 步。Latency 1.23s/chunk。

### 4.3 Ablation 数据(Table 4)

| Scheme | Latency↓ | Calvin ABC-D↑ |
|---|---|---|
| Complete Denoise | 6.04s | 2.480 |
| Joint Denoise | 2.90s | 2.038 |
| Single-step (Eq. 2) | 1.23s | 3.345 |
| **RFG (Eq. 3)** | 1.23s | **3.600** |

**反直觉的发现**:Single-step 居然比 Complete 还好!作者解释是 test-time OOD intermediate state 问题。这让我想到 [VPP](https://arxiv.org/abs/2412.14803) 和 [Cosmos Policy](https://arxiv.org/abs/2601.16163) 这类完全依赖生成质量的 VLA,在 distribution shift 下都会被 image generation 误差拖累。BagelVLA 这个发现其实把 visual foresight 的 role 重新定位了 — **不是当 oracle,而是当 structural prior**。

---

## 5. Residual Flow Guidance (RFG): 全文最 elegant 的 trick

### 5.1 公式

Naive Single-step:
$$
v_{t+k}^{\tau=0} \sim \mathcal{N}(0, I) \tag{2}
$$

RFG:
$$
v_{t+k}^{\tau=0} \sim \mathcal{N}(v_t, I) \tag{3}
$$

变量含义:
- $v_t$ — current observation 的 VAE encoding
- $v_{t+k}^{\tau=0}$ — keyframe denoise 的 initial noise
- $\mathcal{N}(\mu, I)$ — mean 为 $\mu$、covariance 为 $I$ 的高斯

**就这么一行改动**。

### 5.2 为什么 work — entropy reduction argument

经典 flow matching 从 $\mathcal{N}(0, I)$ 走到 $v_{t+k}$,模型要重建**整个 scene**(背景、光照、桌面纹理、远处物体...)。这些静态信息 entropy 高,但 task-irrelevant。

RFG 从 $\mathcal{N}(v_t, I)$ 走到 $v_{t+k}$,模型只需预测 **residual change**(机器人 end-effector 移动、物体位置变化)。dynamic region 的 spatial extent 通常只占 image 的 10~30%,entropy 大幅下降。

数学上,如果定义 $r = v_{t+k} - v_t$,那么:

$$
\mathcal{L}_v^{\text{RFG}} = \mathbb{E}\left[\|v_\theta(v_t + \tau r, \tau) - r\|_2^2\right] = \mathcal{L}_v^{\text{residual}}
$$

这跟 [Residual Policy Learning](https://arxiv.org/abs/1812.06298) 的精神一致,但放到 flow matching 的 noise initialization 上。

### 5.3 效果(Figure 5 + Figure 6)

RFG 在 **10 步** denoise 时已经能生成高质量 keyframe,naive single-step 即使 20 步也模糊。这意味着:

1. **Train loss 下降更快** — task entropy 下降让 gradient 更 informative
2. **Test-time generalization 更好** — 模型只学了"变化"的 manifold,对 background variation 更鲁棒
3. **Action 学得更快** — Table 4 中 RFG 比 naive single-step 收敛更早,因为 v_t prior 给 action expert 提供了强 anchor

### 5.4 类比

这个 idea 让我想起 [Consistency Models](https://arxiv.org/abs/2303.01469) 里讨论的 "warm start" — 当 source 和 target distribution 接近时,ODE trajectory 短,单步也能学好。RFG 本质就是把 keyframe prediction 的 source distribution 从 random noise shift 到 "current frame + small perturbation",trajectory 大幅缩短。

也可以参考 [VideoCrafter](https://arxiv.org/abs/2310.19500) 系列里 image-to-video 的做法 — 用 first frame 作为 strong condition,后续 frame 只需预测 motion。RFG 是把这个 idea 嵌入 flow matching 的 noise initialization。

---

## 6. Two-Stage Training Recipe

### Stage 1: Pretraining (64× A800, 20k steps, bs≈1600)

| Data | 用途 | 量级 |
|---|---|---|
| General VQA ([LLaVA-Pretrain](https://arxiv.org/abs/2304.08460), [FineVision](https://arxiv.org/abs/2510.17269)) | 保留 language 能力 | 2.56M |
| Human-hand Data ([EgoDex](https://arxiv.org/abs/2505.11709)) | visual dynamics | 310k |
| Open-source Robot ([AgiBot](https://arxiv.org/abs/2503.06669), [Bridge](https://arxiv.org/abs/2308.12952), [RoboMind](https://arxiv.org/abs/2412.13877), ...) | language + visual | 382k |
| Self-collected | language + visual | 4.5k |

只训练 Understanding + Generation expert。目标:**transfer foundation model 的 reasoning & generation 能力到 embodied domain**。

关键 trick:对于没有 subtask 标注的 dataset(比如 Bridge),用 Seed-1.5-VL-thinking 自动标注 $l_t$ 和 temporal boundary,然后 filter。这是把 general VLM 当成"subtask labeler",思想类似 [GR-2](https://arxiv.org/abs/2410.06158) 的 video-language pretraining。

### Stage 2: Finetuning (各种下游场景)

引入 action expert,联合训练三个 loss $\mathcal{L}_l + \mathcal{L}_v + \mathcal{L}_a$。具体下游 setting:

- **Calvin ABC-D**: 8× A800, 30k steps, chunk=10, single view prediction
- **Robotwin**: 50 tasks × 50 demos, 60k steps, chunk=16 (effective horizon 48)
- **ALOHA basic**: 3k episodes, 50k steps, chunk=24, 三视角输入
- **ALOHA long-horizon**: 1.5k episodes, subtask 显式标注

### 关键 insight

Stage 1 用大规模无 action 数据预训练 **linguistic planning 和 visual dynamics** 这两个能力,Stage 2 才接 action。这避免了 [OpenVLA](https://arxiv.org/abs/2406.09246) 那种 "VLM pretrain → action finetune 后 VLM 能力 catastrophically forget" 的问题 — 因为 action expert 是新增的 2B 模块,不动 understanding/generation expert 的核心权重(虽然 finetune 时全部 unfreeze,但 action loss 不直接 push language head 的参数远离预训练 manifold)。

---

## 7. Inference: Asynchronous Execution

训练时随机用 preceding frame 替换 current frame,推理时:
- Understanding + Generation expert 的 KV cache 可以 **隔几个 chunk 才更新一次**
- Action expert 只接收新的 proprioception input,生成新 action chunk
- 这样 chunk frequency 从同步 40Hz 提升到 72Hz

这个 trick 跟 [HiRT](https://arxiv.org/abs/2410.05273) 和 [OpenHelix](https://arxiv.org/abs/2505.03912) 的 hierarchical inference 思路一致 — 高层 reasoning 慢更新,底层 control 快更新。本质是利用了 visual state 的时间冗余 — 在一个 subtask 内,场景变化缓慢。

---

## 8. 实验数据 deep dive

### 8.1 Calvin ABC→D ([paper](https://arxiv.org/abs/2112.03227))

Table 8:
| Method | 1 | 2 | 3 | 4 | 5 | Avg Len↑ |
|---|---|---|---|---|---|---|
| π0 | 0.937 | 0.832 | 0.740 | 0.629 | 0.510 | 3.65 |
| UP-VLA | 0.928 | 0.865 | 0.815 | 0.769 | 0.699 | 4.08 |
| VPP | 0.965 | 0.909 | 0.866 | 0.820 | 0.769 | 4.33 |
| w/o Keyframe | 0.909 | 0.792 | 0.676 | 0.546 | 0.422 | 3.35 |
| **BagelVLA** | **0.993** | **0.954** | **0.893** | **0.824** | **0.741** | **4.41** |

关键观察:
- ablation "w/o Keyframe-forecasting" 掉到 3.35 — visual forecasting 贡献巨大(超过 1 unit)
- BagelVLA vs VPP 的差距在 long-horizon(length 4-5)放大,说明 **interleaved planning 的优势在 long-horizon 上更显著**
- Calvin 本身是 single-step tasks,所以 textual planning 在这里没启用

### 8.2 Robotwin 2.0 ([paper](https://arxiv.org/abs/2506.18088))

Table 1:
| Model | Clean | Randomized |
|---|---|---|
| π0 | 46.42 | 16.34 |
| RDT | 34.50 | 13.72 |
| UP-VLA | 52.92 | 15.16 |
| w/o Textual | 54.00 | 19.20 |
| w/o Keyframe | 56.72 | 15.92 |
| **BagelVLA** | **75.26** | **20.87** |

w/o Textual vs BagelVLA 差 21 个点 — textual planning 在 long-horizon + unseen instruction 上贡献 dominant。Randomized 上 BagelVLA 比 π0 提升 4.5 点,但绝对值不高,说明 **distractor + unseen object 还是 VLA 的难点**。

### 8.3 Real-world long-horizon(Table 3)

Stack Cubes in Requested Order:
| Setting | Easy | Mid | Hard | Success | Planning Acc |
|---|---|---|---|---|---|
| π0 | 75 | 35 | 10 | 40.0 | 55 |
| VPP | 90 | 45 | 25 | 53.3 | 80 |
| **BagelVLA** | **95** | **65** | **60** | **73.3** | **95** |

Calculate and Place Symbols:
| Setting | Easy | Mid | Hard | Success | Planning Acc |
|---|---|---|---|---|---|
| π0 | 70 | 25 | 0 | 31.7 | 40 |
| VPP | 70 | 50 | 30 | 50.0 | 75 |
| **BagelVLA** | **80** | **65** | **45** | **63.3** | **85** |

Calculate 任务需要 model 先做算术(21+3=24)再 place block — 这考验 VLM reasoning 在 VLA 中是否保留。BagelVLA planning acc 85% 说明 reasoning 大部分保留,但 success 63.3% 说明 **action execution precision 还是 bottleneck**(模型知道该放哪,但放不准)。

### 8.4 OOD generalization(Figure 6)

Pick & Place Unseen 任务:
- π0: 55%
- VPP: 45%
- BagelVLA: 85% (+30 点 vs π0)
- w/o pretrain: 50%(掉了 35 点)

预训练的 general visual-language reasoning 能力直接 transfer 到 OOD object,这是 unified foundation model 初始化的核心红利。

---

## 9. 与同期工作的定位

| 模型 | Backbone | Visual Foresight | Linguistic Planning | 关键缺点 |
|---|---|---|---|---|
| [RT-2](https://arxiv.org/abs/2307.15818) | PaLI-X VLM | ✗ | ✗ | Discrete action,lose precision |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | Llama VLM | ✗ | ✗ | Same as above |
| [π0](https://arxiv.org/abs/2410.24164) | PaliGemma + FM | ✗ | ✗ | No explicit reasoning |
| [VPP](https://arxiv.org/abs/2412.14803) | Video Diffusion | ✓ | ✗ | Weak instruction following |
| [UP-VLA](https://arxiv.org/abs/2501.18867) | VLM + Diffusion | ✓ | ✗ | Two separate modules |
| [Cosmos Policy](https://arxiv.org/abs/2601.16163) | Video FM | ✓ | ✗ | No VLM backbone |
| [GR-2](https://arxiv.org/abs/2410.06158) | VideoLA | ✓ | partial | Coarse planning |
| **BagelVLA** | Unified MoT | ✓ (RFG) | ✓ (interleaved) | 2-stage training 复杂 |

BagelVLA 真正的差异化:**显式把 interleaved multimodal chain-of-thought 嵌入到 policy 里**,并通过 RFG 让 visual foresight 的代价足够低以至于 real-time 可用。

---

## 10. 我自己的几点思考 / 联想

### 10.1 关于 RFG 的延展

RFG 这个 trick 我觉得可以推广。任何 conditional generation,只要 conditioning signal 跟 target 有强 structural overlap,都可以用 "shifted noise initialization" 代替 pure Gaussian。比如:
- **Video prediction**: $z_0 \sim \mathcal{N}(z_{\text{first frame}}, I)$,自然成为 short-horizon i2v
- **Image editing**: $z_0 \sim \mathcal{N}(z_{\text{source}}, I)$,模型只需学 edit direction
- **Audio continuation**: $z_0 \sim \mathcal{N}(z_{\text{past audio}}, I)$

本质上 RFG 是把 flow matching 从 "generation from noise" 转成 "prediction of delta",这跟 [ControlNet](https://arxiv.org/abs/2302.05543) 加 condition 思路接近,但更轻 — 不需要额外 network。

### 10.2 关于 Single-step Denoise 的反直觉

Complete Denoise 表现差这件事,让我想到一个 general principle:**在 hierarchical policy 里,高层模块的输出误差对低层是放大器**。如果 visual foresight 在 test time 有 5% 误差,action policy 在 OOD image 上可能 50% 失效。Single-step 实际上是让 action 只用 visual **context**(KV cache 的 attention pattern),不用 visual **content**(生成的像素)。这跟"读思维过程 vs 读成品"的差别类似 — KV cache 里的 attention pattern 是 reasoning trace,可能比 final pixel 更鲁棒。

### 10.3 关于 Interleaved Planning 的未来

这个 paradigm 让我想到 [Chain-of-Thought](https://arxiv.org/abs/2201.11903) 在 LLM 中的作用 — 显式化中间步骤。BagelVLA 是把 CoT 扩展到 multimodal: text CoT → visual CoT → action。下一步自然是:
- **更长的 horizon planning**(几小时任务)
- **Tree-of-Thought 在 robot 上的对应**: multiple keyframe branches,选 best rollout
- **Verbalized uncertainty**:模型在 subtask 上输出 confidence,低置信度时主动 re-plan

### 10.4 一些 concern

- **数据效率**:Stage 1 用了 380M+ 样本,Stage 2 finetune 也需要每 task 50 demos,跟 sample-efficient imitation learning 的方向(比如 [DP3](https://arxiv.org/abs/2310.10639))还差很远
- **Action expert 还是用 FM**,在高精度任务(比如插 USB)上可能不如 [ACT](https://arxiv.org/abs/2304.13705) 的 transformer-based action head
- **Real-world 72Hz 是 nominal 频率**,实际推理 + 通讯 + control loop 的延迟没报告

### 10.5 跟你之前的 work 的联系

你之前讲过 [Micrograd](https://github.com/karpathy/micrograd) 和 nanoGPT 的 "everything is a sequence modeling problem"。BagelVLA 其实是这个思想的极致体现 — text、image latent、action chunk 都拼成一条 interleaved sequence,共享 transformer backbone。但跟 nanoGPT 不同的是,这里有两个 FM process 的非自回归 denoise,所以 masking 设计(见 Appendix B 和 Figure 7)很复杂 — 要同时防止 modality leakage 又要匹配 inference 时的 asymmetry。这个工程难度可能是 paper 不好讲清楚的部分。

另一个联想:你 [recent tweet](https://x.com/karpathy) 提到的 "Vibe Coding" — 让 LLM 写代码自己跑自己。BagelVLA 是 "Vibe Acting" — LLM 想 → imagine outcome → act。如果未来 VLA 走这条路,pretrain 阶段可能完全不需要 robot data,只用 internet video + text 就够,robot data 只在 finetune 阶段用来 align action space。这跟 paper 的 stage 1 数据构成(60% 是 robot data)其实有点不一致 — 也许他们发现纯 human video 还不够。

---

## 11. 总结

BagelVLA 的核心贡献,我归纳为三条:

1. **Architecture**: MoT-based 三 expert 统一框架,两个 7B expert 从 Bagel pretrain 初始化,2B action expert 新增
2. **Training**: 两阶段 — Stage 1 用大量无 action 数据学 planning + visual dynamics,Stage 2 finetune action。这避免了 VLM 能力的 catastrophic forgetting
3. **Inference**: RFG + Single-step Denoise 让 visual foresight 的 latency 从 6 秒降到 1.2 秒,同时因为只学 residual,generalization 更好

最值得深挖的是 **RFG** — 这个 trick 简单到可以一行代码实现,但 effect 在 ablation 里 +0.255 Calvin score(3.345 → 3.600),而且只在第一步 denoise 上 work,这暗示了 flow matching 在 robot policy 里的一个新方向:**让 generation 的 source distribution 接近 target,从而把 generation 问题转化为 refinement 问题**。

如果你想要 build intuition 的核心 takeaway,我建议是:**robot policy 的 visual foresight 不需要 full generation,只需要 attention pattern from a single denoise step on residual noise**。这跟 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 里 "用 denoise 过程的中间 feature 作 policy representation" 思路一致,但更极端 — 只取第一步。

---

## Reference Links

- [BagelVLA Project Page](https://cladernyjorn.github.io/BagelVLA.github.io)
- [Bagel (pretrained backbone)](https://arxiv.org/abs/2505.14683)
- [π0](https://arxiv.org/abs/2410.24164) | [π0.5](https://arxiv.org/abs/2504.16054)
- [OpenVLA](https://arxiv.org/abs/2406.09246) | [RT-2](https://arxiv.org/abs/2307.15818)
- [VPP](https://arxiv.org/abs/2412.14803) | [UP-VLA](https://arxiv.org/abs/2501.18867) | [Cosmos Policy](https://arxiv.org/abs/2601.16163)
- [Flow Matching](https://arxiv.org/abs/2210.02747) | [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Calvin Benchmark](https://arxiv.org/abs/2112.03227) | [Robotwin 2.0](https://arxiv.org/abs/2506.18088)
- [AgiBot World](https://arxiv.org/abs/2503.06669) | [BridgeData V2](https://arxiv.org/abs/2308.12952)
- [Seed-1.5-VL (用于自动标注)](https://arxiv.org/abs/2505.07062)
- [SigLIP2](https://arxiv.org/abs/2502.14786) | [FLUX](https://github.com/black-forest-labs/flux) | [Qwen2.5](https://arxiv.org/abs/2412.15115)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) | [ACT](https://arxiv.org/abs/2304.13705) | [RDT-1B](https://arxiv.org/abs/2410.07864)
- [Consistency Models (类比 RFG 思路)](https://arxiv.org/abs/2303.01469)
- [HiRT (asynchronous inference)](https://arxiv.org/abs/2410.05273)

如果想深挖哪个 specific component(engineering detail、attention mask 设计、数据 pipeline、failure case),告诉我具体方向,我可以再展开。
