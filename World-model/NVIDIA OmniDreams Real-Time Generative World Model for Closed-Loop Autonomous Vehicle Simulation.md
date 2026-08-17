---
source_pdf: NVIDIA OmniDreams Real-Time Generative World Model for Closed-Loop Autonomous
  Vehicle Simulation.pdf
paper_sha256: 0b2ec7dcb7726b694b67048228e90fede4c20f9eaa065b1a5cc0c6261fea1a79
processed_at: '2026-08-05T22:50:18-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OmniDreams 人话版

好，Andrej，我把这篇paper揉碎了用大白话讲给你听，顺便帮你build intuition。

## 这paper到底在干嘛

一句话：**NVIDIA 把 video diffusion model 训成一个能实时跑的"自动驾驶宇宙模拟器"**，policy model 给它一个 action，它就生成下一帧摄像头画面，loop 起来就是 closed-loop simulation。

你想想传统的 AV simulation 是怎么搞的——要么用 CARLA 这种 artist 手画的 assets（假得不行），要么用 NeRF/3DGS 重建真实场景（好看但只能演已经录过的路线）。问题是什么？你想测 policy 在"暴雨天遇到一辆掉了个床垫的车"这种 long-tail scenario 下的表现，reconstruction-based simulator 直接GG，因为它根本没见过这种数据。

OmniDreams 的 bet 是：**如果我用 21k 小时真实驾驶视频训练一个 video diffusion model，让它学会"世界长什么样、怎么演化"，那我就可以让它生成任何它没见过的场景**。

这思路其实跟 Sora 那帮人想的差不多——video model as world simulator——只不过 NVIDIA 把它特化到 AV，加了 action conditioning，还把它加速到 105 FPS 真正能跑 closed-loop。

## 核心架构 intuition

### Conditioning：模型需要看什么

OmniDreams 是个 causal DiT（Diffusion Transformer），每次生成一小撮 frames（chunk）。它condition 在四个东西上：

1. **First-frame RGB**：simulation session 的第一帧，告诉模型"世界初始长这样"
2. **Text prompt**：描述天气、光线、时间——比如"nighttime urban road, dry conditions"
3. **Abstract world scenario**：这是最关键的，是个"上帝视角"的结构化地图，包含 lane lines、bounding boxes、ego trajectory
4. **Memory cache**：之前生成过的 frames 的 KV cache

那个 **world scenario map** 你可以理解为：拿 HD map + 3D bounding boxes，用 camera intrinsics/extrinsics project 到 2D，画成一张"线条+色块"的抽象图。lane lines 是黄线，crosswalks 是紫色，cars 是蓝色 cuboids（还 shading 表示 heading），signs 是灰色。

**为什么不用 ControlNet 那种独立网络？** 论文的 argument 是：ControlNet overhead 太大，real-time 扛不住。他们用一个小 MLP 把 control signal 编成 compact tokens，直接 concatenate 到 visual tokens 前面送进 transformer。Separation of concerns：control 信号和 visual content 物理上分开，autoregressive generation 更稳定。

### Multi-View 的 trick

自动驾驶一辆车 7 个 camera，naive 想法是让所有 view 的所有 token 互相 attend——复杂度 $\mathcal{O}(N^2 T^2)$，$N$ 是 view 数，$T$ 是 temporal length。4 个 view × 几十帧，直接爆炸。

OmniDreams 的 factorization：
- **Temporal attention**：每个 view 内部自己 attend 自己的历史 frames
- **Cross-view attention**：每个时间步，跨 view 互相 attend 一次

复杂度变成 $\mathcal{O}(NT^2) + \mathcal{O}(N^2)$。$T$ 那一项不跟 $N$ 二次耦合了，$N^2$ 那一项不跟 $T$ 耦合了。

**Intuition**：这其实是假设 view 间的 consistency 主要来自"共享的 3D scene structure"，而每个 view 的 dynamics 主要来自自己过去的 frames。这种 factorization 在物理上说得通——左侧 camera 看到一辆车，右侧 camera 也看到那辆车，cross-view attention 让它们对齐；但每个 camera 的 motion 主要由自己过去几帧决定。

实现细节：每个 view 有个 learnable embedding，通过 AdaLN（adaptive layer norm）注入，跟 timestep embedding 一样的方式加进去。cross-view attention layer 放在 text cross-attention 之后、MLP 之前。所有新加的 weights 都 zero-init 保证稳定收敛——这是为啥这个 trick 能 work 的工程细节。

## 训练 pipeline：从 bidirectional 到 causal 到 distilled

这个 pipeline 设计得很巧妙，是层层递进的。

### Stage 1: 中训在 RDS 上

起点是 Cosmos-Predict 2.5，一个 bidirectional video generation model（能看前后 frames）。先在 16,600 小时真实驾驶视频上 mid-train，让模型懂"开车视频长啥样"。

数据：7 个 synchronized cameras，30 FPS，1080p，15 个国家。这数据量很猛，但跟 internet-scale video model 比还差一个量级，所以是 "mid-training" 而不是 from scratch。

### Stage 2: Multi-view adaptation

只对 OmniDreams-MV 做。开 view embedding，在 4 种 camera view 均匀混合数据上训，然后加 cross-view attention layers。

### Stage 3: World-Scenario Control

加 control branch（zero-init），先训 93-frame clips，再扩展到 189-frame clips 学长期 consistency。

### Stage 4: Diffusion Forcing（关键！）

这一步把 bidirectional model 转成 causal model，让它能 autoregressive 生成。

**核心 idea**：给每个 token 独立采一个 noise level $t_i$，而不是所有 token 同一个 $t$。

公式：$L_{DF} = \mathbb{E}_{\mathbf{x}^{1:T}, \epsilon}[\|\mathbf{u}_\theta(\mathbf{x}_t^{1:T}, \mathbf{t}) - \mathbf{v}_t\|^2]$

- $\mathbf{x}^{1:T}$：latent video 的 $T$ 个 frames
- $\mathbf{t} = [t_1, t_2, ..., t_T]$：每个 frame 独立采的 noise level
- $\mathbf{x}_t^{1:T} = (I - \mathbf{t}) \cdot \mathbf{x}^{1:T} + \mathbf{t} \cdot \epsilon$：每个 frame 独立加噪
- $\mathbf{v}_t = \epsilon - \mathbf{x}^{1:T}$：flow matching target

**为啥这个 trick 重要？** 传统 teacher forcing 训练时每个 frame 的 context 是 clean 的 ground truth，但 inference 时 context 是模型自己生成的 noisy output——这就是 exposure bias。Diffusion Forcing 通过让每个 token 见到不同 noise level 的 context，让模型在训练时就逼近 inference 时的 distribution。

而且这个 formulation 让模型同时是 next-token predictor 和 full-sequence denoiser，一身两职。Causal masking 用 Flex-Attention 实现，$\mathbf{x}^i$ 的 tokens 只能 attend 到 frames $\leq i$。

### Stage 5: Self Forcing + DMD（精华）

Diffusion Forcing 之后，模型能 autoregressive 生成了，但有两个问题：
1. 需要 many denoising steps，慢
2. Long rollout 时 compounding errors 累积

**Self Forcing 的核心 idea**：训练时让模型自己做 self-rollout，用自己生成的（imperfect）frames 做 context，而不是 ground-truth。

具体做法：
- 模型做 autoregressive self-rollout，每个 frame 用 $\kappa=2$ 步 diffusion 生成
- 每次训练 iteration 只随机选一个 denoising step $s \sim \text{Uniform}(1, ..., T)$ 做 backprop
- 之前 frames 的 KV cache embedding 被 detach（stop-gradient），gradient 只流当前 frame

**为啥 gradient detach？** 否则你要 backprop 整个 rollout chain，memory 爆炸。Detach 之后每个 frame 独立更新，但模型仍然在"用自己输出做 context"的 distribution 下训练。

**Rolling KV cache**：维护最近 $W$ 个 frames 的 cache，加新 frame 时 evict 最老的。复杂度从 $\mathcal{O}(WT^2)$ 降到 $\mathcal{O}(WT)$。训练时 attention 也限制在同样 window，match inference behavior。

**DMD Objective**：不用 pixel-wise reconstruction loss，用 video-level distribution matching：

$$\mathcal{L}_{DMD}(\theta) = \mathbb{E}\left[\frac{1}{2}\|\hat{x} - \text{sg}[\hat{x} - (\mathbf{f}_\psi(\hat{x}_t, t) - \mathbf{f}_\phi(\hat{x}_t, t))]\|^2\right]$$

- $\hat{x}$：self-rolled-out video clip
- $\mathbf{f}_\phi$：frozen real score network（学真实数据 distribution）
- $\mathbf{f}_\psi$：learned fake score network（学模型生成 distribution）
- $\text{sg}[\cdot]$：stop-gradient
- $\mathbf{f}_\psi - \mathbf{f}_\phi$：score difference，告诉 generator "往哪个方向调才能更像真实数据"

**Intuition**：这其实是 minimize $p_\theta$ 和 $p_{\text{data}}$ 之间的 reverse KL divergence，但在 video level 而不是 frame level。好处是不需要 paired data，不需要 pixel-wise alignment，只要 distribution match。

### Stage 6: Progressive Long-Context Teacher

Self Forcing 之后，short context teacher 训出的模型在 long rollout 时会有 "shifting artifacts"——当 rolling KV cache 超过训练 context window 时，模型见到没见过的 context length，开始退化。

**解决方案**：
1. 先训一个 bidirectional model with 很长 temporal context（bidirectional 不受 rolling cache 影响）
2. 用这个 long-context bidirectional model 作为 teacher，继续 finetune 之前 distilled 的 student

Table 6 的数据说话：

| Teacher | 0-5s FVD | 15-20s FVD | Δ |
|---------|----------|------------|---|
| Short-context | 109.3 | 409.2 | 299.9 |
| Progressive long-context | 95.5 | 268.4 | 172.9 |

Δ 是 final-window 减 first-window 的 FVD。Progressive teacher 把退化从 300 降到 173，效果显著。

**Intuition**：Bidirectional model 因为能看未来 frames，不存在 rolling cache 问题，所以可以训很长 context。然后把这个"长 context 知识"distill 给 causal student，让 student 学到长 context 的 visual prior，即使它自己只能看过去。

## Inference 工程：怎么做到 105 FPS

这部分是 training-free 的纯系统优化，但很关键。

### Local Temporal Attention

不要 full temporal attention over unbounded rollout。用 local window：SV 用 6 latent frames（24 RGB frames）window，MV 用 8 latent frames（32 RGB frames）window。

### Streaming Static-Shape KV Cache

预分配 cache 到 fixed size，tensor shape 保持 static，cache update 放 separate thread 不阻塞 main thread。这让你可以 capture CUDA Graph。

### CUDA Graphs

因为 shape static，DiT forward 可以 capture 成 CUDA Graph，第一次 chunk lazy capture，之后每个 chunk 重用同一个 graph submission。这省了 kernel launch overhead。

### Lightweight Encoders/Decoders

用 LightX2V 的 LightVAE 和 LightTAE 替换 Cosmos 原 VAE。MV 用 pixel shuffle 替换 LightVAE，latency <1ms。代价是 FVD 退化（Table 5：24.8 → 45.4），但换来 decode latency 从几十 ms 降到 5-30ms。

### Hoisting Step-Invariant Operators

RoPE frequencies 每 chunk 算一次（不是每 denoising step），patchify/unpatchify 也只在 chunk 开始结束算。这种小优化累积起来很可观。

### Multi-GPU Parallelism

DiT 有三个并行轴：$V$（view），$T$（temporal），$HW$（spatial）。16-GPU 四视图配置是 $V=4, T=4, HW=1$。

**为啥这个 order $V \to T \to HW$？** 
- Self-attention 是 per-camera，沿 $V$ 切分不会 split 一个 self-attention call，4 ranks 近线性加速
- 沿 $T$ 切分不会 split per-timestep cross-view attention，但沿 $HW$ 切分会 split 所有 attention
- 所以优先 $T$ 后 $HW$

用 in-house ring attention 实现，overlap KV-shard transfer 和 local compute。

### 最终性能

OmniDreams-SV：1 GB300，118ms 生成 8 frames，68 FPS
OmniDreams-MV：16 GB300，151ms 生成 16 frames（4 views × 4 frames），105 FPS per camera

这速度让 video diffusion 真的能用于 closed-loop——之前 Sora 这种 model 生成 1 分钟要等好久，现在 16 frames 0.15 秒搞定。

## AlpaSim 集成：怎么把 video model 嵌进 simulator

这部分是工程难点，因为 video model 作为 renderer 有三个 weird property：

### Challenge 1: Distributed Inference

Video model 多 GPU 跑时，标准 rank-based parallelism 假设你整个 system 都是 rank-parallel。但 AlpaSim 还要跑 policy、traffic、physics 这些其他 service，它们各自有自己的 parallelism 逻辑。

**解决方案**：在 rank 0 跑 gRPC server，接收 rendering requests，通过 NCCL events forward 给其他 ranks，完成后 frames gather 到 rank 0 serialize 回去。这是"两层 networking"——gRPC 跨 service，NCCL 跨 GPU rank。

### Challenge 2: Autoregressive State

Video model 内部 stateful（KV cache），不能任意顺序 render。

**解决方案**：Session-based abstraction。每个 rollout 开新 session，server 生成 session ID，关联 inputs 和 pre-allocated KV cache。Client 后续 requests 带 session ID。

### Challenge 3: Chunk-Based Generation

Video model 在 chunks 中生成（VAE temporal compression），policy 在 chunk 中间不能改 trajectory。

两种策略：
- **Post-fetch**：在 chunk 内 rollout policy 不给新 visual inputs，要求 policy 支持 out-of-sync，video frames logical time 在 request 之前，inject 到 timeline 会 out of order
- **Pre-fetch**（AlpaSim 用）：policy 一次生成 multi-step trajectories，每 chunk 边界做 prediction，commit 到 follow 直到下个 chunk。Video model call 在 advance timer 之前，保持 event ordering

**Intuition**：Pre-fetch 保留了 event ordering，避免了 out-of-sync 的 subtle bugs。代价是 policy 必须 predict 多步 trajectory，但 production AV stack 本来就这样。

## WAM：最 surprising 的发现

这部分我觉得是整个 paper 最 exciting 的——**OmniDreams 的 backbone 直接 finetune 成 policy，5× 参数压缩下打败了 VLA-based Alpamayo 1.5**。

### 架构

保持 causal DiT backbone 不变，加两个东西：
1. **DINOv2 features** + 30° front-telescope camera，通过 linear projection 加到 noisy video latent tokens
2. **History token**：编码前 1.6s ego motion（small MLP），append 到每个 frame 的 token set，跟 video patches interleave

**Attention mask 设计很讲究**：
- Video tokens 保持原 causality：只 attend 到 current 和 past video tokens
- History tokens 是 one-way coupling：每个 history token attend 到所有 current 和 past video tokens，但不 attend 其他 history tokens

**为啥这么设计？** History token 要从 video context 提取信息，但 history tokens 之间不需要互相 attend（它们是 ego motion 历史，不是 visual content）。这种 one-way coupling 让信息从 visual stream 流到 action stream，但反过来不流——这符合直觉：你看到的画面影响你的动作，但你的动作历史不直接影响你看的画面（间接通过 closed-loop 影响）。

### Trajectory Prediction

DiT 在 history token position 输出 $\mathbf{h}$，喂给 12-layer U-Net MLP，参数化 flow matching velocity field $\mathbf{u}_\phi(\boldsymbol{\tau}_s, s; \mathbf{h})$，over trajectory latent $\boldsymbol{\tau} \in \mathbb{R}^{64 \times 3}$（64 waypoints × 3D coordinates）。

预测 6.4s future ego trajectory at 10 Hz = 64 waypoints。

### Joint Training

Training 时 joint objective：video flow matching loss + trajectory flow matching loss。每个 frame 的 video noise level $u$ 和 trajectory noise level $s$ 独立采样。

**Intuition**：这样 history token 必须在 whatever noise level 的 visual context 下提取信息，不能 co-adapt 到单一 noise level。这是 data augmentation 的一种形式——让 policy head robust 到 visual input 的各种 noise level。

### Inference

只用一次 causal forward pass：4 video latent frames with small noise + 1 history token + 4 flow-matching steps sample trajectory。因为只关心 trajectory，DiT backbone 只跑一次，4 denoising steps 限制在 lightweight trajectory MLP。

### 结果

Physical AI NuRec dataset，574 scenes，10 Hz replanning，20s rollouts：

| Model | Params | Collision | Front | Lateral | Rear |
|-------|--------|-----------|-------|---------|------|
| Alpamayo 1.5 (VLA) | ~10B | 6.9% | 1.0% | 0.6% | 5.3% |
| OmniDreams WAM | ~2B | **4.2%** | **0.9%** | **0.4%** | **3.0%** |

WAM 用 1/5 参数，collision rate 从 6.9% 降到 4.2%，所有 sub-category 都 better。

**这暗示什么？** World model 在学"生成未来 frames"时，已经学到了足够 rich 的 scene representation，这些 representation 本身就是好的 policy backbone。这跟 robotics 领域 Ye et al. 2026 的发现一致（World Action Models are Zero-Shot Policies, https://arxiv.org/abs/2602.15922）。

**更深层的 intuition**：VLA 模型通常用 language 作为 reasoning bridge，但 driving 这种连续控制任务，language 可能不是最优的 intermediate representation。World model 学的是"visual dynamics + physics + geometry"，这些可能比 language tokens 更直接对应 driving 需要的 reasoning。所以 WAM 用更少参数就能 outperform VLA——它把"算力"花在了更 task-relevant 的 representation 上。

未来方向：**joint world-model-policy training**，一个 model 同时学 render 和 drive，而不是 post-train 分离。这可能才是 AV 的 ultimate architecture。

## Diffusion Fixer：顺带的小应用

OmniDreams 还能 post-train 成"reconstruction artifact 修复器"。

**问题**：NeRF/3DGS 在 novel-view extrapolation 时有 blur、ghosting、missing content、floating geometry。

**解决方案**：构造 paired data（degraded rendering + clean target），训练时从 degraded rendering 开始 denoising（不从 random noise 开始），学一个 flow 把 artifact-corrupted outputs map 到 clean-image manifold，同时 preserve scene layout 和 viewpoint。

这跟 Difix3D+、Artifix、DiffusionHarmonizer 思路一样，都是用 diffusion prior 修复 3D reconstruction artifacts。

参考：
- Difix3D+: https://arxiv.org/abs/2506.08009
- Artifix: https://arxiv.org/abs/2603.00492
- DiffusionHarmonizer: https://arxiv.org/abs/2602.24096

## 实验结果的关键 insight

### Distilled > Bidirectional（Table 4）

| Stage | FVD↓ | LET-AP↑ | F1↑ |
|-------|------|---------|-----|
| Bidirectional | 26.8 | 0.378 | 0.823 |
| Causal (DF) | 31.7 | 0.221 | 0.775 |
| **Distilled (Self Forcing)** | **24.8** | **0.400** | **0.828** |

Distilled model 竟然比 bidirectional teacher 还好！**Intuition**：Self Forcing 的 self-rollout + DMD distribution matching 让 distilled model 学到更 robust 的 representation，加上 high-quality distillation dataset（58k videos 集中在 challenging urban scenarios），distilled student 超越 teacher。这跟 RLHF 里 student 超越 teacher 的现象类似。

### Closed-Loop Ranking Preservation（Figure 13）

在 501-scene NuRec subset 上比较 4 个 policy：WAM > Alpamayo 4-cam > 2-cam > 1-cam。

**关键 finding**：NuRec 和 OmniDreams 上的 ranking 一致！

这意味着 OmniDreams 是 real world 的 faithful proxy——开发者可以在 OmniDreams 中比较 policy，得到与 logged data 类似的结论。这是 generative simulator 能替代 reconstruction 的核心证据。

### Visual Realism vs Trajectory Deviation（Figure 14）

NuRec：trajectory 偏离 ground truth 越远，FVD 越差（reconstruction 无法 extrapolate）
OmniDreams：即使 trajectory 偏离，visual realism 保持稳定

**Intuition**：Reconstruction 是"记住具体 rays"，新视角就崩；generative model 是"学会世界规律"，新视角也能 synthesize plausible content。这是 generative vs reconstructive 的本质区别。

## 我的整体 takeaway

1. **Video diffusion 终于够快用于 closed-loop**：105 FPS@720p，这是工程 milestone。FlashDreams 开源让社区能复现。

2. **Generative > Reconstruction for long-tail**：reconstruction 只能演已录场景，generative 能合成没见过的。这是 paradigm shift。

3. **WAM > VLA 是大信号**：world model representation 本身就是好的 policy backbone，5× 参数压缩还能 outperform。这可能改变 AV architecture 方向——unified world-action model 而不是分离的 world model + policy。

4. **Training pipeline 设计很巧妙**：bidirectional → causal (Diffusion Forcing) → distilled (Self Forcing + DMD) → progressive long-context teacher。每一步都有明确的 problem-solution pair。

5. **Chunk-based generation 是当前 limitation**：未来目标是 frame-at-a-time，但这需要 VAE temporal compression 的根本改进。

6. **Microservice architecture 很务实**：gRPC + NCCL 两层 networking，让 video model 能作为 subcomponent 嵌入大系统，不强制整个系统 rank-parallel。

7. **Attention factorization 是 multi-view scaling 的关键**：$\mathcal{O}(N^2 T^2) \to \mathcal{O}(NT^2) + \mathcal{O}(N^2)$，这假设 view 间 consistency 来自 shared 3D structure，每个 view 的 dynamics 来自自己历史，物理上说得通。

8. **Self Forcing + DMD 是 autoregressive video 的 future**：解决 exposure bias，video-level distribution matching，rolling KV cache，progressive teacher，这一套 recipe 可能成为标准。

9. **AV simulator 进入 Tier 4**：从 abstract graphics → artist assets → neural reconstruction → generative world models。每个 tier 解决上一 tier 的瓶颈。

10. **NVIDIA 的 vertical integration**：Cosmos backbone + OmniDreams model + AlpaSim simulator + Alpamayo policy + NuRec baseline + FlashDreams inference + GB300 hardware + SIL-Wheel data curation。整套 stack 全自己搞，这种 vertical integration 只有 NVIDIA 能做。

## 进一步 reading

- Cosmos foundation: https://arxiv.org/abs/2501.03575
- Self Forcing: https://arxiv.org/abs/2506.08009
- Diffusion Forcing: https://arxiv.org/abs/2504.03043
- DMD: https://arxiv.org/abs/2311.18828
- AlpaSim (开源): https://github.com/NVlabs/alpasim
- FlashDreams (开源): https://github.com/NVIDIA/flashdreams
- LightX2V: https://github.com/ModelTC/lightx2v
- Ring Attention: https://arxiv.org/abs/2310.01889
- StreamingLLM: https://arxiv.org/abs/2309.17453
- Waymo World Model: https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation
- WAM are zero-shot policies: https://arxiv.org/abs/2602.15922
- DINOv2: https://arxiv.org/abs/2304.07193
- Flex Attention: https://arxiv.org/abs/2412.05496

这篇 paper 我觉得是 2026 年 AV 领域的 milestone work，它不只是把 video model 应用到 AV，而是把整套 training/inference/simulation stack 都打通了。WAM 那个发现尤其值得 follow，可能开启 unified world-action model 的新范式。

---

# NVIDIA OmniDreams 深度解析

非常好的论文，这是 NVIDIA 在 2026 年初发布的一份重要的技术报告，展示了 generative world model 在 autonomous driving closed-loop simulation 中的实际应用。让我从 motivation、architecture、training、inference、application 五个层面来构建你的 intuition。

## 1. Motivation 与核心问题

### 1.1 为什么需要 generative world model？

传统 autonomous vehicle (AV) 的 closed-loop simulation 主要依赖 **reconstruction-based neural simulators**，比如 NeRF、3D Gaussian Splatting (3DGS)、NVIDIA NuRec。这些方法虽然能 photorealistic 渲染，但本质上是 **过拟合到原始采集数据**：

- 只能在 captured corridor 内做 what-if 测试
- 无法生成 extreme weather（雨雪风暴）
- 无法生成 articulated motion（如行人突然变向、车辆上掉落床垫）
- 无法生成 novel scene content

OmniDreams 的核心 motivation：用一个 **action-conditioned generative world model** 来作为 reactive environment，让 policy 在 closed-loop 中获得 scalable、comprehensive 的训练和评估。

参考链接：
- NVIDIA NuRec: https://developer.nvidia.com/omniverse/nurec
- Cosmos 论文: https://arxiv.org/abs/2501.03575
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### 1.2 关键性能指标

| 配置 | 模型参数 | 相机数 | FPS | 分辨率 | GPU |
|------|---------|-------|-----|--------|-----|
| OmniDreams-SV | 2B | 1 | 68 | 704×1280 | 1× GB300 |
| OmniDreams-MV | 2B | 4 | 105 (per camera) | 704×1280 | 16× GB300 |

这个 FPS 是 effective FPS = K / Total_ms，其中 K 是 chunk size (8 或 16 frames)。这意味着 **real-time interactive rendering** 是可行的，远超一般 video diffusion model 的速度。

## 2. Model Architecture

### 2.1 整体条件输入

OmniDreams 是一个 causal Diffusion Transformer (DiT)，输入条件包括四类：

1. **First-frame RGB**: 模拟 session 的第一帧，通过 Cosmos VAE encoder 编码为 clean latent tokens，初始化整个生成
2. **Text prompt**: 描述环境属性，用 Cosmos text encoder 编码，通过 cross-attention 注入
3. **Abstract world scenario**: 关键创新点，是一个 pixel-aligned 的结构化表示
4. **Memory cache**: streaming KV cache，存储之前生成的 tokens

### 2.2 World Scenario Map 的设计 intuition

这个抽象 state 编码了：
- **静态 HD map**: lane lines, road boundaries, stop lines, poles, crosswalks, road markings, traffic lights, traffic signs
- **动态 actors**: 3D bounding boxes (cuboids)，颜色 shading 区分 heading
- **Ego trajectory**: 来自 policy model 或 human driver 的动作

渲染方式：用 camera intrinsics 和 extrinsics 将 3D 元素 project 成 pixel-aligned 的 lines, surfaces, 2D-projected cuboids。

**这个设计的 intuition 是**：相比 ControlNet 之类的独立网络，OmniDreams 用一个小 MLP 编码成 compact control tokens，与 visual tokens concatenate 后送入 transformer。这种 lightweight control branch 的好处是：
- 计算开销小
- 控制信号与视觉内容清晰分离，稳定 autoregressive 生成
- 满足 real-time throughput

### 2.3 Multi-View Factorization

这是 OmniDreams-MV 的关键技术。naive 的 multi-view formulation 会让所有 token 互相 attend，复杂度：

$$\mathcal{O}(N^2 T^2)$$

其中 $N$ 是相机视图数，$T$ 是 temporal length。

OmniDreams 将 attention 分解为两部分：

**Temporal attention**（每个 view 内部）：tokens 通过 causal KV cache attend 到之前的 frames，捕捉 motion dynamics 和 temporal consistency。

**Cross-view attention**（每个 time step 跨 view）：不同 view 的 tokens 互相 attend，确保 geometry、object positions、motion 的 cross-view consistency。

分解后复杂度变为：

$$\mathcal{O}(N T^2) + \mathcal{O}(N^2)$$

- 第一项：每个 view 内部的 temporal attention
- 第二项：跨 view 的 attention（不依赖 $T$）

**Intuition**: 这是一种 factorized attention，类似 factorized graph neural network。每个 view 的 internal dynamics 用 temporal attention 处理，view 间的 shared scene structure 用 cross-view attention 处理。这样 attention cost 不再随 $N$ 和 $T$ 同时二次增长。

实现细节：
- 每个 view 有 learnable embedding，通过 adaptive layer normalization (AdaLN) 注入（类似 timestep embedding）
- Cross-view attention layer 加在每个 block 的 cross-attention 之后，MLP 之前
- View embedding 零初始化保证稳定收敛
- Cross-view attention 的 output projection 也零初始化

### 2.4 Multi-View Cross Block 结构

参考 Figure 4，每个 Multi-View Cross Block 包含：
1. Self-Attention (per-view, with causal masking)
2. Text Cross-Attention
3. **Cross-View Attention**（跨 V 个 view 的 tokens）
4. MLP

每个 sublayer 都用 AdaLN 调制 shift/scale/gate，conditioning 信号是 view embedding + time embedding 的和。

## 3. Training Strategy

OmniDreams 采用 multi-stage training：

### Stage 1: Mid-training on RDS (Real Driving Scene)
- 数据：16,600 小时，3M 个 20s clips
- 7 个 synchronized cameras (front-wide, front-telescope, front-left, front-right, rear-left, rear-right, rear-tele)
- 30 FPS, 1080p
- 15 个国家

### Stage 2: Multi-view adaptation (only OmniDreams-MV)
- Enable learnable view embeddings
- 训练在 front-wide, cross-left, cross-right, front-telescope clips 的均匀混合
- 添加 cross-view attention layers

### Stage 3: World-Scenario Control
- 添加 world-scenario control branch (zero-initialized)
- 先训 93-frame clips，再扩展到 189-frame clips

### Stage 4: Causal Masking + Diffusion Forcing
这是把 bidirectional model 转成 causal/autoregressive model 的关键。

#### Rectified Flow 基础

Cosmos 用 rectified flow objective。给定 latent video $\mathbf{x}$，noise $\epsilon \sim \mathcal{N}(0, I)$，timestep $t \in [0,1]$：

$$\mathbf{x}_t = (1-t)\mathbf{x} + t\epsilon$$

$$\mathbf{v}_t = \epsilon - \mathbf{x}$$

训练目标：

$$L = \mathbb{E}_{\mathbf{x},t}[\|\mathbf{u}_\theta(\mathbf{x}_t, t) - \mathbf{v}_t\|^2]$$

其中 $\mathbf{u}_\theta$ 是 velocity prediction network。

#### Causal Masking

对于 latent sequence $\mathbf{x}^{1:T}$，autoregressive 分解：

$$p(\mathbf{x}^{1:T}) = \prod_{i=1}^{T} p(\mathbf{x}^i | \mathbf{x}^{<i})$$

每个 $p(\mathbf{x}^i | \mathbf{x}^{<i})$ 参数化为 flow matching model $\mathbf{u}_\theta(\mathbf{x}_t^i | \mathbf{x}^{<i})$。

Causal masking 在 self-attention 层中：$\mathbf{x}^i$ 的 tokens 只能 attend 到 frames $\leq i$。可以推广到 block-autoregressive：

$$p(\mathbf{x}^{1:T}) = \prod_{j \in \{ki\}, i \in 1:T} p(\mathbf{x}^{j:j+k} | \mathbf{x}^{<j})$$

每次生成 $k$ 个 frames 的 block。

#### Diffusion Forcing Objective

关键创新：每个 token 独立采样 noise level $\mathbf{t} = [t_i]_{i \in 1:T}$，从 log-normal 分布：

$$\mathbf{x}_t^{1:T} = (I - \mathbf{t}) \cdot \mathbf{x}^{1:T} + \mathbf{t} \cdot \epsilon$$

$$\mathbf{v}_t = \epsilon - \mathbf{x}^{1:T}$$

$$L_{DF} = \mathbb{E}_{\mathbf{x}^{1:T}, \epsilon}[\|\mathbf{u}_\theta(\mathbf{x}_t^{1:T}, \mathbf{t}) - \mathbf{v}_t\|^2]$$

**Intuition**: Diffusion Forcing 让模型同时充当 next-token predictor 和 full-sequence denoiser。传统 teacher forcing 在每个 frame 用 clean ground-truth context，但 inference 时模型用自己的 (imperfect) 输出，造成 exposure bias。Diffusion Forcing 通过给每个 token 不同的 noise level，让模型在训练时就见到各种 noise level 的 context，更接近 inference 时的分布。

参考：Diffusion Forcing 论文: https://arxiv.org/abs/2504.03043

### Stage 5: Self Forcing + DMD Distillation

这是为了让模型 few-step 生成，同时消除 long autoregressive rollout 的 compounding errors。

#### Self Forcing via Self-Rollout

核心思想：训练时用模型自己的输出做 context，而不是 ground-truth。

- 模型做 autoregressive self-rollout，每个 frame 通过 $\kappa$-step diffusion 生成（$\kappa=2$，timestep schedule $[1000, 450]$）
- 为了训练 tractable，每次 iteration 只随机 sample 一个 denoising step $s \sim \text{Uniform}(1, \ldots, T)$ 做 backprop
- 之前 frames 的 KV cache embedding 被 detach，gradient 只流过当前 frame

#### Rolling KV Cache

为了支持 long video，用 fixed-size rolling KV cache：
- 维护最近 $W$ 个 frames 的 cache
- 加新 frame 时 evict 最老的
- 推理复杂度从 $\mathcal{O}(WT^2)$ 降到 $\mathcal{O}(WT)$（这里 $W$ 是 window size）
- 训练时也限制 attention 到同样的 rolling window

#### DMD Objective

Self Forcing 用 holistic video-level distribution matching，不是 pixel-wise reconstruction：

$$\mathcal{L}_{DMD}(\theta) = \mathbb{E}\left[\frac{1}{2}\left\|\hat{x} - \text{sg}\left[\hat{x} - (\mathbf{f}_\psi(\hat{x}_t, t) - \mathbf{f}_\phi(\hat{x}_t, t))\right]\right\|^2\right]$$

变量解释：
- $\hat{x}$: self-rolled-out video clip
- $\mathbf{f}_\phi$: frozen real score network（在真实数据上训练）
- $\mathbf{f}_\psi$: learned fake score network（estimate generated distribution 的 score）
- $\text{sg}[\cdot]$: stop-gradient operation
- $\mathbf{f}_\psi - \mathbf{f}_\phi$: score difference 提供 gradient，将 generator 推向 real data manifold

**Intuition**: 这个 objective 最小化 model distribution $p_\theta$ 和 data distribution $p_{\text{data}}$ 之间的 reverse KL divergence，在 video level 而不是 frame level，不需要 paired data 或 pixel-wise supervision。

参考：
- Self Forcing: https://arxiv.org/abs/2506.08009
- DMD: https://arxiv.org/abs/2311.18828

#### Progressive Training with Longer Teacher

Self Forcing 短 context teacher 训练后，模型仍会有 shifting artifacts（当 rolling KV cache 超过训练 context window 时）。

解决方案：
1. 先训一个 bidirectional (non-causal) video model with 很长 temporal context window
2. 用这个 long-context bidirectional model 作为 teacher，继续 finetune 之前 distilled 的 short-context student

实验数据（Table 6）显示效果显著：

| Training teacher | 0-5s | 5-10s | 10-15s | 15-20s | Mean | Δ |
|-----------------|------|-------|--------|--------|------|---|
| Short-context | 109.3 | 183.0 | 258.3 | 409.2 | 240.0 | 299.9 |
| Progressive long-context | 95.5 | 151.0 | 202.5 | 268.4 | 179.4 | 172.9 |

Δ = final-window FVD - first-window FVD，越小越好。Progressive teacher 把 Δ 从 299.9 降到 172.9，显著减少 long rollout 退化。

## 4. Inference Optimization

这部分是 training-free 的，只针对 distilled checkpoint 做系统级优化。

### 4.1 Local Temporal Attention

Full temporal attention over unbounded rollout 不现实。用 local-window attention：
- OmniDreams-SV: window of 6 latent frames (24 RGB frames)
- OmniDreams-MV: window of 8 latent frames (32 RGB frames)

### 4.2 Streaming Static-Shape KV Cache

关键工程优化：
- 预分配 cache 到 fixed size
- 保持 tensor shape static across rollout
- Cache updates 在 separate thread，让 main thread 不被 cache update latency 阻塞

### 4.3 CUDA Graphs

因为 KV cache 是 static shape，DiT forward 可以 capture 成 CUDA Graph，第一次 chunk 时 lazy capture，之后每个 chunk 重用。

### 4.4 Lightweight Encoders/Decoders

- 用 LightVAE (from LightX2V) 替换 Cosmos 原 VAE，平衡 reconstruction quality 和 speed
- OmniDreams-MV 用 pixel shuffle 替换 LightVAE，latency 可忽略
- LightTAE decoder for both MV 和 SV，极低 decode latency

参考 LightX2V: https://github.com/ModelTC/lightx2v

### 4.5 Hoisting Step-Invariant Operators

把不随 denoising step 变化的计算 hoist 出去：
- RoPE frequencies: 每 chunk 算一次，不是每 step
- Patchify/unpatchify: 只在 chunk 开始/结束时算

### 4.6 Performance Numbers

**OmniDreams-SV (Table 2)**:

| Stage | 1 GPU | 2 GPU | 4 GPU | 8 GPU |
|-------|-------|-------|-------|-------|
| World scenario encoding | 28ms | 26ms | 26ms | 26ms |
| Diffusion DiT | 84ms | 71ms | 49ms | 47ms |
| RGB Decoder | 6ms | 5ms | 5ms | 5ms |
| KV-cache update (parallel) | 42ms | 34ms | 23ms | 22ms |
| **Total** | **118ms** | **102ms** | **80ms** | **78ms** |
| **Effective FPS** | **68** | **78** | **100** | **103** |

**OmniDreams-MV (Table 3)**:

| Stage | 1 GPU | 4 GPU | 8 GPU | 16 GPU |
|-------|-------|-------|-------|--------|
| Diffusion DiT | 1184ms | 300ms | 179ms | 121ms |
| RGB Decoder | 105ms | 30ms | 30ms | 30ms |
| KV-cache update (parallel) | 558ms | 149ms | 91ms | 67ms |
| **Total** | **1289ms** | **330ms** | **209ms** | **151ms** |
| **Effective FPS** | **12** | **48** | **74** | **105** |

### 4.7 Multi-GPU Parallelism

DiT 有三个并行轴：
- $V$: camera-view axis (最多相机数)
- $T$: temporal axis within chunk (最多 latent frames per chunk)
- $HW$: spatial axis within one frame's attention

16-GPU 四视图配置：$V=4, T=4, HW=1$

**添加 rank 的顺序** $V \to T \to HW$，依据：
- Self-attention 是 per-camera，所以沿 $V$ 切分不会 split 一个 self-attention call
- 4 ranks 沿 $V$ 给近线性加速，saturate 这个 axis
- 之后选 $T$ 而不是 $HW$，因为 temporal sharding 不 split per-timestep cross-view attention
- 用 in-house ring attention 实现，overlaps KV-shard transfer 和 local attention compute

### 4.8 FlashDreams

把 streaming static-shape KV cache, local-window attention, CUDA-graph capture 等 recipes 打包成开源 inference stack，已验证在 Wan2.1-based backbone 上有 1.95× speedup，Lingbot-World 上 2.49× speedup。

参考：https://github.com/NVIDIA/flashdreams

## 5. Closed-Loop Integration with AlpaSim

### 5.1 AlpaSim 架构

AlpaSim 是 NVIDIA 开源的 research-oriented AV simulator：
- Microservice architecture
- 轻量级 core runtime
- RPC 到 specialized services (camera renderer, AV policy, physics)
- gRPC protocol
- Docker container 部署

参考：https://github.com/NVlabs/alpasim

### 5.2 三个核心挑战

#### Challenge 1: Distributed Inference

Video model 多 GPU 跑时，标准 rank-based parallelism 不适合作为 subcomponent 嵌入大系统。

**解决方案**：加一层 gRPC 层
- Rank 0 跑 gRPC server 接收 rendering requests
- 通过 NCCL events forward 到其他 ranks
- 完成后 frames gather 到 rank 0，serialized，sent back to AlpaSim

未来工作：用 RDMA-based gRPC，或把 gRPC 当 coordination layer，bulk transfer 通过 NCCL。

#### Challenge 2: Autoregressive Nature

Video models 内部 stateful，不能任意顺序 render。

**解决方案**：Session-based state abstraction
- 每个 rollout 开始一个新的 session request，包含 seed first frame 和 map representation
- Server 生成 fresh ID，关联 inputs 和 pre-allocated KV cache
- Client 在后续 requests 中带上 ID

#### Challenge 3: Chunk-Based Generation

Video models 在 chunks 中生成 frames（对应 VAE temporal compression），policy 在 chunk 中间不能修改 trajectory。

**两种策略**：

**Post-fetch generation**:
- 在 chunk 内 rollout policy 和 traffic，不给 policy 新 visual inputs
- 要求 policy 支持 last available frame lagging behind last ego position
- 视频帧 logical time 在 request 之前，inject 到 timeline 会 out of order

**Pre-fetch generation**（AlpaSim 采用）:
- Policy 和 traffic models 一次生成 multi-step trajectories
- 每 chunk 边界做 prediction
- 新 trajectories append 到历史，commit 到 follow 直到下个 chunk
- Trajectories 在 video frame timestamps 插值，send 给 video model
- Response frames 转成 events，inject 到 rollout state at precise timestamp
- 视频模型 call 在 advance simulation timer 之前

**Intuition**: Pre-fetch 保留了 event ordering，避免了 out-of-sync 处理的 elusive bugs。

## 6. World-Action Model (WAM)

这是论文最 surprising 的发现之一：OmniDreams 的 backbone 可以直接 finetune 成 competitive policy。

### 6.1 架构

保持 OmniDreams causal DiT backbone 不变，加两个 additions：

1. **DINOv2 features**: 用 dinov2_vitb14 encoder 的 patch features 和 30° front-telescope camera，通过 learnable linear projections 加到 noisy video latent tokens 作为 conditioning

2. **History token**: 编码前 1.6s ego motion（通过 small MLP），append 到每个 frame 的 token set，与 video patches 沿 temporal axis interleave

#### Attention Mask

- Video-to-video: 保持原 causality（video token 只 attend 到 current 和 past video tokens）
- History tokens: one-way coupling（每个 history token attend 到所有 current 和 past video tokens，但不 attend 其他 history tokens）

#### Trajectory Prediction

DiT output $\mathbf{h}$ at history-token position 喂给 12-layer U-Net-shaped MLP，参数化 flow matching velocity field $\mathbf{u}_\phi(\boldsymbol{\tau}_s, s; \mathbf{h})$ over future-trajectory latent $\boldsymbol{\tau} \in \mathbb{R}^{64 \times 3}$。

预测 6.4s 未来 ego trajectory at 10 Hz = 64 waypoints。

#### Joint Denoising

训练时 joint objective summing flow matching losses on video 和 trajectory latents，independently sampled flow-matching times（per-frame $u$ for video, per-sequence $s$ for trajectory）。

**Intuition**: 这样 history token 必须在 whatever noise level 的 visual context 下提取信息，而不是 co-adapt 单一 noise level。

#### Inference

Drop interleaved-noise schedule，跑一次 causal forward pass：
- 4 video latent frames with small noise
- 1 history token
- 4 flow-matching steps sample trajectory
- 只关心 trajectory，所以 DiT backbone 只跑一次
- 4 denoising steps 限制在 lightweight trajectory MLP

### 6.2 性能对比

在 Physical AI Autonomous Vehicles NuRec dataset（574 scenes，10 Hz replanning，20s rollouts）：

| Model | Params | Collision | Front | Lateral | Rear |
|-------|--------|-----------|-------|---------|------|
| Alpamayo 1.5 | ~10B | 6.9% | 1.0% | 0.6% | 5.3% |
| OmniDreams WAM | ~2B | **4.2%** | **0.9%** | **0.4%** | **3.0%** |

**WAM 用 1/5 的参数超越了 VLA-based Alpamayo 1.5！**

**Intuition**: World model 在生成 video 时学到的 representations 已经足够 rich 到可以作为 policy 的 backbone。这与 robotics 领域的发现一致（Ye et al., 2026 - World Action Models are Zero-Shot Policies: https://arxiv.org/abs/2602.15922）。

这暗示未来 AV 的方向：**一个模型同时学 render 和 drive**，而不是分离的 world model + policy。

## 7. Diffusion Fixer

OmniDreams 还可以 post-train 来修复 reconstruction-based simulator 的 artifacts。

### 7.1 问题

NeRF、3DGS 在 novel-view extrapolation 时会有：
- Blurred details
- Missing content
- Ghosting
- Floating geometry
- View-dependent artifacts

### 7.2 解决方案

构造 paired training data：degraded rendering from reconstruction-based simulator + corresponding clean target images。

训练时，不从 random Gaussian noise 开始 denoising，而是从 degraded rendering 本身开始，学习一个 flow 把 artifact-corrupted outputs map 到 clean-image manifold，同时 preserve scene layout, camera viewpoint, driving-relevant structure。

### 7.3 Inference

作为 autoregressive correction module on top of pre-reconstructed scenes。给定 reconstruction 系统的 rendered frame，OmniDreams refine image，通过 causal history 和 KV-cache conditioning 维持 temporal consistency。

参考相关 work：
- Difix3D+: https://arxiv.org/abs/2506.08009
- Artifix: https://arxiv.org/abs/2603.00492
- DiffusionHarmonizer: https://arxiv.org/abs/2602.24096

## 8. Experiments and Results

### 8.1 Training Stage Comparison (Table 4)

| Training stage | FVD↓ | Temp.Sampson↓ | LET-AP↑ | F1↑ | Cat.Acc↑ |
|----------------|------|---------------|---------|-----|----------|
| Bidirectional (AV adapted) | 26.8 | 1.83 | 0.378 | 0.823 | 0.957 |
| Causal (Diffusion Forcing) | 31.7 | 1.87 | 0.221 | 0.775 | 0.941 |
| Distilled (Self Forcing) | **24.8** | 1.90 | **0.400** | **0.828** | **0.961** |

关键发现：
- Bidirectional 是 upper bound（non-causal，不能 autoregressive 用）
- Causal Diffusion Forcing 性能下降（causal masking 限制了 attention）
- **Distilled Self Forcing 超越了 Bidirectional！**

**Intuition**: Self Forcing 的 self-rollout 训练 + DMD 的 distribution matching 让 distilled model 学到了比 teacher 更 robust 的 representations。加上 high-quality distillation dataset（58k videos 集中在 challenging urban scenarios），distilled model 在 FVD 和 downstream perception metrics 上都更好。

### 8.2 Decoder Trade-off (Table 5)

| Decoder | FVD↓ | LET-AP↑ |
|---------|------|---------|
| Original VAE | 24.8 | 0.400 |
| LightTAE | 45.4 | 0.376 |

LightTAE 换来了极低 decode latency（5-30ms），代价是 FVD 退化 ~20 点。这是 speed/quality 的 explicit trade-off。

### 8.3 Long Rollout Stability (Table 6)

见前面 Stage 5 的讨论。Progressive long-context teacher 是关键。

### 8.4 Long-tail Coverage

两个互补的 handles：

1. **Controllable Scenario Editing**: 通过 text prompt（weather, lighting, time of day）+ abstract world-scenario map（lane geometry, bounding boxes）+ first-frame RGB seed 做系统化 variation

2. **Out-of-Distribution Object Modeling**: 
   - Naive approach: 直接编辑 first-frame RGB 插入 OOD object，但会有 artifacts（因为没在 world-scenario map 中表示）
   - **Improved approach**: post-train with randomized dynamic-cuboid dropout，让模型学会从 visual history + first-frame seed + scene context 推断 object persistence 和 motion，不依赖 explicit cuboid trajectory

### 8.5 Closed-Loop Comparison (Figure 13)

501-scene subset of Physical AI NuRec dataset，20s rollouts，replan every 533ms。

比较 4 个 policy classes：
1. OmniDreams WAM
2. Alpamayo 1.5 (4 cameras)
3. Alpamayo 1.5 (2 cameras: front-wide + front-telescope)
4. Alpamayo 1.5 (1 camera: front-wide only)

**Key finding**: Policy ranking 在 NuRec 和 OmniDreams 上一致：WAM > Alpamayo 1.5 (4 cam) > Alpamayo 1.5 (2 cam) > Alpamayo 1.5 (1 cam)。

这说明 **OmniDreams 是 real world 的 faithful proxy**：开发者可以在 OmniDreams 中比较 policy，得到与 logged data 类似的结论。

### 8.6 Visual Realism vs Trajectory Deviation (Figure 14)

- NuRec: trajectory 偏离 ground truth 越远，FVD 越差（reconstruction 无法 extrapolate）
- OmniDreams: 即使 trajectory 偏离，visual realism 保持稳定（learned video prior + world-scenario conditioning 能 synthesize plausible observations）

### 8.7 Dynamic Object Quality (Figure 15)

OmniDreams 在 pedestrian quality 和 motion 自然度上明显优于 NuRec，因为 reconstruction-based 方法对 dynamic objects 很难处理。

## 9. Related Work 中的定位

### 9.1 World Models for AV

| 类别 | 代表工作 | 特点 |
|------|---------|------|
| Reconstruction-based | NuRec, NeuRAD, EmerNeRF, 3DGS | 忠实 captured corridor，extrapolation 差 |
| Video models as world sim | Sora, Movie Gen, Wan, Veo 3, Genie | 通用，internet-scale |
| Generative AV | DriveGAN, GAIA-1/2, Vista, MagicDrive, Drive-WM, GenAD, Cosmos-Drive-Dreams, Waymo World Model | AV 特化，可控性 |
| **OmniDreams** | - | Closed-loop interactivity + streaming KV cache + few-step distillation + 真实 microservice simulator |

### 9.2 Real-time/Streaming Video Diffusion

- Self Forcing / CausVid: train-test gap bridge
- StreamingT2V / FIFO-Diffusion: long rollout stability
- Diffusion Forcing: per-token noise schedule
- StreamingLLM: attention-sink tokens
- Ring attention: context-parallel kernel

### 9.3 Simulator Tiers

论文提出 AV simulator 的 4 个 tiers：

1. **Tier 1**: Non-visual / abstract graphics（Waymax, PufferDrive）- 大规模 RL
2. **Tier 2**: Artist-generated assets（CARLA, MetaDrive）- 物理和视觉 realism
3. **Tier 3**: Neural reconstruction（DriveArena, HUGSIM, WorldEngine, AlpaSim）- 数据驱动 + 高质量
4. **Tier 4**: **Generative world models（OmniDreams）** - 这是论文主张的新 tier

## 10. Key Takeaways 和 Intuition Building

### 10.1 为什么这个 work 重要？

1. **Real-time 闭环**: 105 FPS@720p on 16 GB300，让 video diffusion 真正可用于 interactive simulation
2. **Faithful proxy**: Closed-loop ranking 与 reconstruction-based 一致，证明 generative model 可以替代 reconstruction
3. **WAM > VLA**: World model 的 representations 本身就是好的 policy backbone，5× 参数 reduction
4. **Long rollout stability**: Progressive long-context teacher + Self Forcing + rolling KV cache 让分钟级 rollout 不退化
5. **Engineering 系统**: FlashDreams 开源 inference stack，AlpaSim microservice 架构，让整套系统 reproducible

### 10.2 核心设计 intuitions

1. **Causal + KV cache > Bidirectional for closed-loop**: Bidirectional 模型需要未来 frames，无法做 interactive generation。Causal + KV cache 让模型可以 streaming 生成，同时保持 long context。

2. **Lightweight control branch > ControlNet**: ControlNet 是独立网络，computational overhead 大。OmniDreams 用 small MLP 编码 control tokens，concatenate 到 visual tokens，minimal overhead。

3. **Factorized multi-view attention**: 不要做 full joint attention（$\mathcal{O}(N^2 T^2)$），分解成 temporal + cross-view（$\mathcal{O}(NT^2) + \mathcal{O}(N^2)$），让 scaling 可行。

4. **Self Forcing 解决 exposure bias**: 训练时用 self-rollout 而不是 ground-truth context，让模型见到 inference 时的 distribution。

5. **DMD 做 video-level distribution matching**: 不用 pixel-wise reconstruction，用 score difference 推 model 到 real data manifold。

6. **Progressive teacher 解决 long rollout**: 先用短 context teacher distill，再用长 context bidirectional teacher 继续 finetune，消除 shifting artifacts。

7. **WAM: 同一 backbone 同时做 render 和 drive**: 这暗示 future AV 可能不需要分离的 world model + policy，而是 unified model。

### 10.3 开放问题和未来方向

1. **Chunk size reduction**: 当前 chunk-based generation 限制了 policy 的反应速度，未来目标是 frame-at-a-time generation

2. **Joint world-model-policy training**: 目前 WAM 是 post-train，未来可以 jointly train render 和 drive

3. **RDMA-based frame transport**: 现在用 gRPC + JPEG encoding，未来用 RDMA 或 NCCL bulk transfer

4. **Sparse temporal attention**: Native sparse attention 等技术可以进一步加速 long context

5. **Lightweight super-resolution**: FlashVSR 等技术可以提升 resolution 不增加 compute

## 参考资源

主要论文和开源项目：
- OmniDreams / Cosmos: https://arxiv.org/abs/2501.03575
- AlpaSim: https://github.com/NVlabs/alpasim
- FlashDreams: https://github.com/NVIDIA/flashdreams
- Alpamayo: https://arxiv.org/abs/2511.00088
- Self Forcing: https://arxiv.org/abs/2506.08009
- Diffusion Forcing: https://arxiv.org/abs/2504.03043
- DMD: https://arxiv.org/abs/2311.18828
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- LightX2V: https://github.com/ModelTC/lightx2v
- DINOv2: https://arxiv.org/abs/2304.07193
- Flex Attention: https://arxiv.org/abs/2412.05496
- Ring Attention: https://arxiv.org/abs/2310.01889
- StreamingLLM: https://arxiv.org/abs/2309.17453
- Waymo World Model: https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation

这篇 paper 是 generative world model 在 AV 领域的一个重要 milestone，它展示了从 reconstruction-based 到 generative 的范式转变，并且通过精心设计的 training stages 和 system optimization 让这个范式 practical。WAM 的结果特别 exciting，暗示了 AV 乃至 robotics 领域 unified world-action model 的可能性。
