---
source_pdf: BagelVLA.pdf
paper_sha256: 691d30709ce69fbd23c4366b41d30dc06ce5b3b406e36afa3f813e7c09b43cd0
processed_at: '2026-07-18T13:51:22-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BagelVLA 深度解析: Interleaved Vision-Language-Action Generation for Long-Horizon Manipulation

Andrej, 这篇 paper 试图解决一个非常本质的问题: 如何让 embodied agent 同时具备 "思考" (linguistic planning)、"想象" (visual forecasting) 和 "执行" (action generation) 三种能力, 并让它们在一个 unified transformer 内部 interleaved 地协同工作。下面我从 motivation、formulation、architecture、training recipe、RFG 机制、实验数据等多个角度展开, 试图 build your intuition。

---

## 1. Motivation: 为什么需要 Interleaved Planning

当前 VLA 领域存在一个明显的 fragmentation:
- **Linguistic planning 路线** (SayCan, Code as Policies, RT-2 高层 planning): 擅长 task decomposition, 但缺乏对 physical outcome 的预见;
- **Visual forecasting 路线** (VPP, Cosmos Policy, GR-2): 擅长想象未来 frame, 但 instruction-following 在 long-horizon reasoning 任务上容易崩;
- **Pure action VLA** (OpenVLA, π0, RDT): 直接 mapping $p_\theta(a_t | v_t, L)$, 在 long-horizon 任务上 black-box mapping 不足以捕捉 causal chain。

BagelVLA 的核心 insight: **这三件事本来就应该 interleaved**。比如 "stack red→yellow→blue→green" 这样的全局 instruction, 模型每一步应该先 "说" 出当前 subtask ("pick red block"), 再 "画" 出 red block 被抓起来后的 keyframe, 最后 "输出" action chunk。这种 chain-of-thought 跨 modality 的展开, 让 action generation 不再是孤立的黑盒, 而是 grounded 在 explicit reasoning 和 explicit prediction 之上。

---

## 2. 形式化: Interleaved Planning 的 Joint Distribution

给定 global instruction $L$ 和当前 observation $v_t$, BagelVLA 建模联合分布:

$$p_\theta(a_t, v_{t+k}, l_t \mid v_t, L)$$

按 manipulation 的 logical dependency 因式分解:

$$\mathcal{L} = -(\mathcal{L}_l + \mathcal{L}_v + \mathcal{L}_a) = \max_\theta \mathbb{E}_\mathcal{D} \log \big[ p_\theta(l_t|v_t,L) \cdot p_\theta(v_{t+k}|v_t,L,l_t) \cdot p_\theta(a_t|v_t,L,l_t,v_{t+k}) \big]$$

**变量解释**:
- $L$: global instruction (e.g., "stack cubes in order red→yellow→blue→green")
- $v_t$: current observation (image + proprioception)
- $l_t$: immediate subtask text (e.g., "pick up the red block with left arm")
- $v_{t+k}$: future keyframe (k steps ahead, 标注的关键帧)
- $a_t$: action chunk (一段连续动作序列)

**Intuition**: 这个 factorization 的顺序非常关键。它强制 model 先做 semantic decomposition (用 VLM 的语言能力), 再做 world modeling (用 generative model 的想象能力), 最后做 control (用 action expert)。这模仿了人类做长程任务时的 "think → imagine → act" 循环。注意, 这与 naive 的 "先生成完整 plan 再执行" 不同, 是 per-step interleaved, 每一步都重新 reason。

---

## 3. Architecture: Mixture of Transformers (MoT)

BagelVLA 采用 MoT 架构, 三个 expert 共享 self-attention 但参数独立:

| Module | Understanding Expert | Generation Expert | Action Expert |
|--------|---------------------|-------------------|--------------|
| Size | 7B | 7B | 2B |
| Input | Image/Text | Image | Proprio/Action |
| Output | Text | Image | Action |
| Encoder | ViT (SigLIP2) + MLP | VAE (FLUX) + MLP | MLP |
| Image Res | 256×256 | 256×256 (VAE) | - |
| Hidden | 3584 | 3584 | 3584 |
| Intermediate | 18944 | 18944 | 3584 (1/5 of LLM) |
| Layers | 28 | 28 | 28 |
| Loss | CE | MSE (FM) | MSE (FM) |
| FM Timestep | - | LogitNormal(0,1) | Beta(1.5,1) |

**关键设计点**:
1. **初始化**: Understanding + Generation expert 从 Bagel (ByteDance 的 unified understanding-generation model) 初始化, 这意味着模型一开始就具备 multimodal reasoning 和 image generation 能力。Action expert 是新增的小 transformer (2B, MLP intermediate 缩到 1/5)。
2. **双视觉编码器**: SigLIP2 用于 understanding (semantic features), FLUX VAE 用于 generation (pixel-level latent)。这种分离避免了 VAE 的 high-frequency pixel 信息污染 semantic understanding。
3. **Action expert 的小尺寸**: 2B 是为了 KV-cache 友好, 配合 asynchronous execution 实现 40-72Hz 推理。
4. **FM Timestep 分布不同**: Generation 用 LogitNormal(0,1) (中间 timestep 采样多), Action 用 Beta(1.5,1) (偏后段 timestep 采样多, 即接近 ground truth 的 denoising step 权重更高)。这个不对称设计很 subtle, 可能反映 action 对最终精度更敏感。

**Attention 拓扑** (从 Fig.2 和 Appendix B 推断):
- LLM expert autoregressive 生成 $l_t$ token, attend 到 ViT features of $v_t$ 和 text instruction $L$;
- Generation expert 在 denoise keyframe 时, attend 到所有 input views 的 VAE+ViT features 和已生成的 $l_t$;
- Action expert 在 denoise action chunk 时, attend 到 VAE/ViT features、$L$、$l_t$、proprioception, 以及 generation expert 当前正在 denoise 的 image latent (这是 dual flow-matching 的关键)。

---

## 4. Dual Flow-Matching: 三种 Conditioning Scheme

这是 paper 最 technical 的部分。问题是: generation expert (denoise keyframe) 和 action expert (denoise action) 都是 flow matching, 它们之间如何交互?

### Scheme 1: Complete Denoise (Fig.3a)

$$\mathcal{L}_v = \mathbb{E}\big[||\mathbf{v}_{v,\theta}(L, v_t, l_t, \tau, v_{t+k}^\tau) - (v_{t+k}^1 - v_{t+k}^0)||_2^2\big]$$
$$\mathcal{L}_{a1} = \mathbb{E}\big[||\mathbf{v}_{a,\theta}(L, v_t, l_t, v_{t+k}^{\tau=1}, \tau, a_t^\tau) - (a_t^1 - a_t^0)||_2^2\big]$$

其中 $v_{t+k}^\tau = (1-\tau)v_{t+k}^0 + \tau v_{t+k}^1$, $a_t^\tau = (1-\tau)a_t^0 + \tau a_t^1$。

**变量**:
- $\mathbf{v}_{v,\theta}, \mathbf{v}_{a,\theta}$: 模型预测的 velocity field (flow matching 的核心, 学的是从 noise 到 data 的 vector field)
- $v_{t+k}^0$: initial noise (Gaussian)
- $v_{t+k}^1 = v_{t+k}$: ground truth keyframe
- $a_t^0$: initial action noise
- $a_t^1 = a_t$: ground truth action chunk
- $\tau \in [0,1]$: flow matching timestep

**逻辑**: 先完全 denoise image ($N_1$ 步), 再把 clean image 作为 condition denoise action ($N_2$ 步)。这是 classic World Model + Inverse Dynamics Model 组合。

**问题**: 推理 latency = $N_1 + N_2$, 而且 image generation 误差会 accumulate 到 action。

### Scheme 2: Joint Denoise (Fig.3b)

$$\mathcal{L}_{a2} = \mathbb{E}\big[||\mathbf{v}_{a,\theta}(L, v_t, l_t, v_{t+k}^\tau, \tau, a_t^\tau) - (a_t^1 - a_t^0)||_2^2\big]$$

**区别**: action expert attend 到当前 noisy keyframe $v_{t+k}^\tau$ (而非 clean $v_{t+k}^{\tau=1}$), 两者同步 denoise N 步。

**问题**: 训练时 action 看到的是 noisy image, 推理时也是 noisy, 理论上对齐。但 OOD 场景下 image denoise 中间态会 drift, 导致 action 跟着崩。

### Scheme 3: Single-step Denoise (Fig.3c) — BagelVLA 默认

$$\mathcal{L}_{a3} = \mathbb{E}\big[||\mathbf{v}_{a,\theta}(L, v_t, l_t, v_{t+k}^{\tau=0}, \tau, a_t^\tau) - (a_t^1 - a_t^0)||_2^2\big]$$

**关键**: action 只 attend 到 image denoise 的 **第一步** 的 KV-cache。这意味着 image 几乎还是纯噪声时, action 就开始生成了。

**Naive vs RFG**:
- Naive (Eq.2): $v_{t+k}^{\tau=0} \sim \mathcal{N}(0, I)$ — 纯 Gaussian noise
- RFG (Eq.3): $v_{t+k}^{\tau=0} \sim \mathcal{N}(v_t, I)$ — 以当前帧 $v_t$ 为均值的 Gaussian

**Intuition for RFG**: 这等价于让 flow matching 学一个 **residual** — 从当前 frame 到 future keyframe 的变化量, 而非从纯噪声重建整个 scene。这有两个好处:
1. Static background 不需要重新生成, model capacity 集中在 dynamic region (e.g., 机械臂、被操作物体);
2. Single-step denoise 就能 extract 足够 predictive 的 feature, 因为 $v_t$ 本身已经携带了 90% 的 scene 信息, model 只需预测 "what changes"。

这在 Fig.5 中可视化得很清楚: naive 方法在 5 steps 时 background 还是糊的, RFG 在 10 steps 就能生成高质量 keyframe。

### Ablation 数据 (Table 4, Calvin ABC-D, single-view):

| Scheme | Latency ↓ | ABC-D ↑ |
|--------|-----------|---------|
| Complete Denoise | 6.04s | 2.480 |
| Joint Denoise | 2.90s | 2.038 |
| Single-step (Naive) | 1.23s | 3.345 |
| RFG | 1.23s | 3.600 |

**反直觉发现**: Single-step 反而比 Complete Denoise 好! 作者解释是 OOD 场景下 (Calvin D split 颜色变化), Complete/Joint Denoise 在中间 step 遇到 OOD noisy state, action 跟着崩。Single-step 因为只看第一步 (initial noise + 当前 frame context), 反而 robust。这其实揭示了一个 deep 的 issue: **World Model 的中间 denoise state 是脆弱的, 不应该被 policy 直接 consume**。

---

## 5. Two-stage Training Strategy

### Stage 1: Pretraining — Linguistic Planning + Visual Dynamics

只 finetune Understanding + Generation expert, 不训 action。数据:
- General VQA: 2.98M pairs (LLaVA-Pretrain 558k + FineVision 2M) — 保 language 能力
- Human-hand Data: 310k episodes (EgoDex) — visual dynamics
- Open-source Robot (with subtask labels): 146k episodes (AgiBot, GR, Galaxea, Bridge, Robotwin)
- Open-source Robot (visual only): 297k episodes
- Self-collected Real Robot: 75k episodes

**关键**: 用 Seed-1.5-VL-thinking 自动标注 subtask $l_t$ 和 temporal boundary, 对公开数据集做 annotation pipeline。这是把 general multimodal reasoning 能力 transfer 到 embodied domain 的关键 step。

### Stage 2: Finetuning — Action Planning

引入 action expert, 全模型 finetune。数据:
- Calvin ABC: visual + action
- Robotwin: 50 tasks × 50 episodes = 2.5k, 全 interleaved
- ALOHA basic: 3k episodes
- ALOHA long-horizon: 1.5k episodes

**Progressive 设计的 motivation**: 如果一开始就联合训 action, action expert 的随机初始化会破坏 pretrained 的 language+vision 能力 (catastrophic forgetting)。先 stage 1 让 model 学会 "想" 和 "画" 在 embodied domain, 再 stage 2 把 action 挂上去, action expert 通过 attention 从 frozen-ish 的 understanding/generation expert 那里 distill 语义和预测信号。

---

## 6. Inference: Asynchronous Execution

推理时每步 denoise 只激活一个 expert (7B text/image 或 2B action)。Single-step denoise + KV-cache 让 BagelVLA 在单卡 RTX 5090 上 1.2s/chunk, chunk size 48 → 40Hz real-world frequency。

**Asynchronous trick**: 训练时随机用 preceding frame 替换 current frame, 让 model 学会 "stale context + fresh proprioception → action"。推理时只更新 proprioception KV, 不更新 vision/language KV, 频率提到 72Hz。这本质上是把 high-level context (vision, language plan) 和 low-level state (proprioception) 解耦 at inference time, 类似 hierarchical control 的思路。

---

## 7. 实验数据深度解读

### 7.1 Calvin ABC-D (Table 1, 8)

| Model | 1 | 2 | 3 | 4 | 5 | Avg Len ↑ |
|-------|---|---|---|---|---|-----------|
| π0* | 0.937 | 0.832 | 0.740 | 0.629 | 0.510 | 3.65 |
| UP-VLA | 0.928 | 0.865 | 0.815 | 0.769 | 0.699 | 4.08 |
| VPP | 0.965 | 0.909 | 0.866 | 0.820 | 0.769 | 4.33 |
| w/o Keyframe | 0.909 | 0.792 | 0.676 | 0.546 | 0.422 | 3.35 |
| **BagelVLA** | 0.993 | 0.954 | 0.893 | 0.824 | 0.741 | **4.41** |

**Observation**: 随 task length 增加, BagelVLA 的优势越明显 (length 5: 0.741 vs VPP 0.769 — 等等, VPP 反而高? 这里要看清, VPP 在 length 5 是 0.769, BagelVLA 0.741, 但 avg len 4.41 > 4.33。说明 BagelVLA 在 length 1-4 都更高, length 5 略低但整体期望更高)。w/o Keyframe 掉到 3.35, 证明 visual forecasting 是核心。

### 7.2 Robotwin (Table 1, 7)

| Model | Clean ↑ | Randomized ↑ |
|-------|---------|--------------|
| π0 | 46.42 | 16.34 |
| RDT | 34.50 | 13.72 |
| UP-VLA | 52.92 | 15.16 |
| w/o Textual | 54.00 | 19.20 |
| w/o Keyframe | 56.72 | 15.92 |
| **BagelVLA** | **75.26** | **20.87** |

**Observation**: 
- Textual planning 贡献: 75.26 vs 54.00 (+21.26), 主要在 Clean (in-domain);
- Keyframe forecasting 贡献: 75.26 vs 56.72 (+18.54);
- Randomized (OOD) 提升相对小 (20.87 vs 19.20 vs 15.92), 说明 visual+language planning 在 OOD 下也有一定泛化, 但 action mapping 本身的 OOD gap 仍难跨越。

### 7.3 Real-World Basic Tasks (Table 2)

| Model | Pick&Place Seen | Pick&Place Unseen | Water Flower | Stack Cubes | Put Flowers | Stack Bowls | Pour Fries | Sweep Rubbish | Press Button | Drawer Close | Avg |
|-------|-----------------|------------------|--------------|------------|-------------|-------------|------------|---------------|--------------|--------------|-----|
| π0 | 95 | 55 | 50 | 65 | 40 | 70 | 35 | 55 | 90 | 95 | 65.0 |
| VPP | 85 | 45 | 60 | 50 | 50 | 55 | 30 | 45 | 80 | 100 | 59.5 |
| **BagelVLA** | 95 | **85** | 60 | 80 | 35 | 90 | 45 | 80 | 90 | 95 | **75.5** |

**亮点**: Pick&Place Unseen 从 π0 的 55 提升到 85, 这是 general multimodal pretraining 带来的 semantic generalization — 模型见过大量 internet image-text, 对 "pear", "peach", "purple block" 等 OOD object 有 robust semantic grounding。

### 7.4 Long-Horizon Planning (Table 3)

| Task | Difficulty | π0 | VPP | w/o Keyframe | w/o Text | **BagelVLA** |
|------|------------|-----|-----|--------------|---------|--------------|
| Stack Cubes | Easy | 75 | 90 | 60 | 75 | **95** |
| Stack Cubes | Middle | 35 | 45 | 15 | 65 | **65** |
| Stack Cubes | Hard | 10 | 25 | 0 | 60 | **60** |
| Stack Cubes | Success | 40.0 | 53.3 | 25.0 | 73.3 | **73.3** |
| Stack Cubes | Plan Acc | 55 | 80 | 45 | 95 | **95** |
| Calculate | Easy | 70 | 70 | 60 | 80 | **80** |
| Calculate | Middle | 25 | 50 | 10 | 65 | **65** |
| Calculate | Hard | 0 | 30 | 0 | 45 | **45** |
| Calculate | Success | 31.7 | 50.0 | 23.3 | 63.3 | **63.3** |
| Calculate | Plan Acc | 40 | 75 | 30 | 85 | **85** |

**关键 insight**: 
- Planning Accuracy (95%, 85%) >> Success Rate (73.3%, 63.3%), 说明 model 知道该做什么, 但 action execution precision 不足。这指向一个明确的 future direction: action expert 需要更精细的 control, 或者用更 high-frequency 的 data 训;
- Calculate 任务 (需要 CoT 算术) 证明 VLM 的 reasoning 能力在 interleaved planning 中保留了下来, 这是 Bagel 初始化 + General VQA co-training 的功劳;
- w/o Textual planning 在 Stack Cubes Hard 只有 60 (vs 95 plan acc), 说明没有 language plan 时, model 即使 "知道" 该做什么也无法 consistent 执行 — language plan 起到了 memory/anchor 的作用。

---

## 8. 与 Related Work 的 positioning

- **vs π0**: π0 是 flow matching VLA 但没有 explicit planning/forecasting, 在 long-horizon 上明显落后;
- **vs VPP**: VPP 用 video prediction policy, 有 visual forecasting 但无 linguistic planning, 且无 unified backbone;
- **vs UP-VLA**: UP-VLA 是 unified understanding+prediction, 但 prediction 是 value function style 而非 generative keyframe;
- **vs Cosmos Policy**: Cosmos 直接 finetune video model, 缺 VLM backbone, instruction-following 弱;
- **vs F1, VILA-X, UniCod**: 这些用 action expert + unified backbone 但缺 explicit interleaved multimodal chain-of-thought;
- **vs Bagel (base)**: Bagel 是 general unified model, BagelVLA 把它 embodied 化, 加 action expert + RFG + 两阶段训练。

---

## 9. 个人思考与可能的延伸

1. **RFG 的深层意义**: $v_{t+k}^{\tau=0} \sim \mathcal{N}(v_t, I)$ 本质上是把 flow matching 从 "生成" 任务变成了 "编辑" 任务。这与 SDEdit、InstructPix2Pix 的思路相通 — 当你有 strong prior (current frame), 生成 residual 比生成 from scratch 容易得多。这个 idea 可以推广到 video prediction、scene editing 等领域。

2. **Single-step Denoise 的反直觉成功**: 这暗示了一个 deep 的 issue — 当前 generative model 的中间 denoise state 是 OOD 敏感的。如果 action policy 直接 attend 这些中间态, 会被误导。RFG 通过把 prior 注入 initial noise, 让第一步 denoise 就 extract 出 "what changes" 的信号, 这个信号比中间态更 robust。这或许能启发新的 "denoising-as-feature-extractor" 范式。

3. **Planning-Execution Gap**: Plan Acc 95% vs Success 73% 揭示了 action expert 是 bottleneck。可能的改进: (a) 用 diffusion policy 的更精细 action head; (b) hierarchical action — high-level action expert 输出 sub-goal, low-level 输出 joint torque; (c) 更高频 data 训练精细 motor control。

4. **Asynchronous Execution 的 risk**: 用 stale frame 训练虽然提速, 但在 dynamic scene (e.g., 物体被推动) 可能 fail。可以引入一个 lightweight "change detector" 决定何时 refresh vision KV。

5. **MoT 的扩展性**: 当前 3 expert (text/image/action)。未来可以加 audio expert (听碰撞声)、tactile expert (触觉反馈), 形成 5-expert MoT。Bagel 的 unified pretraining 让这种扩展相对自然。

6. **Data Engine 的可复现性**: 用 Seed-1.5-VL-thinking 自动标注 subtask 是关键, 但标注质量直接影响 plan acc。可以探索 active learning — 让 model 自己 flag 不确定的 trajectory 交人工标注。

---

## 10. 参考链接

- **Project Page**: https://cladernyjorn.github.io/BagelVLA.github.io
- **Bagel (base model)**: https://arxiv.org/abs/2505.14683
- **π0**: https://arxiv.org/abs/2410.24164
- **VPP**: https://arxiv.org/abs/2412.14803
- **UP-VLA**: https://arxiv.org/abs/2501.18867
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **Cosmos Policy**: https://arxiv.org/abs/2601.16163
- **Flow Matching**: https://arxiv.org/abs/2210.02747 (Lipman et al.)
- **Calvin Benchmark**: https://arxiv.org/abs/2112.03227
- **Robotwin 2.0**: https://arxiv.org/abs/2506.18088
- **AgiBot World**: https://arxiv.org/abs/2503.06669
- **SigLIP2**: https://arxiv.org/abs/2502.14786
- **FLUX**: https://github.com/black-forest-labs/flux
- **Qwen2.5**: https://arxiv.org/abs/2412.15115
- **Seed-1.5-VL**: https://arxiv.org/abs/2505.07062
- **F1 (related MoT VLA)**: https://arxiv.org/abs/2509.06951
- **VILA-X**: https://arxiv.org/abs/2507.23682
- **UniCod**: https://arxiv.org/abs/2510.10642
- **Chameleon**: https://arxiv.org/abs/2405.09818
- **Show-o**: https://arxiv.org/abs/2408.12528
- **RDT-1B**: https://arxiv.org/abs/2410.07864
- **3D Diffuser Actor**: https://arxiv.org/abs/2402.10885
- **Octo**: https://arxiv.org/abs/2405.12213
- **DROID**: https://arxiv.org/abs/2403.12945
- **EgoDex**: https://arxiv.org/abs/2505.11709
- **Zebra-CoT**: https://arxiv.org/abs/2507.16746
- **FineVision**: https://arxiv.org/abs/2510.17269
- **RoboMIND**: https://arxiv.org/abs/2412.13877
- **DreamVLA**: https://arxiv.org/abs/2507.04447
- **CogVLA**: https://arxiv.org/abs/2508.21046
- **HiRT**: https://arxiv.org/abs/2410.05273
- **OpenHelix**: https://arxiv.org/abs/2505.03912
- **GR-2**: https://arxiv.org/abs/2410.06158
- **GR-3**: https://arxiv.org/abs/2507.15493
- **Galaxea**: https://arxiv.org/abs/2509.00576
- **RobiX**: https://arxiv.org/abs/2509.01106
- **Gemini Robotics 1.5**: https://arxiv.org/abs/2510.03342
- **π0.5**: https://arxiv.org/abs/2504.16054
- **mimic-video**: https://arxiv.org/abs/2512.15692
- **UniUGP**: https://arxiv.org/abs/2512.09864
- **VLingNav**: https://arxiv.org/abs/2601.08665

---

## 总结

BagelVLA 的核心贡献是把 "interleaved multimodal chain-of-thought" 引入 VLA — 每一步都 explicit 地 reason (text) + imagine (image) + act (action), 三者通过 MoT 共享 attention 但参数独立。RFG 是工程上的关键 trick, 让 single-step denoise 就能 extract predictive feature, 把 inference latency 从 6s 压到 1.2s。实验数据清晰地显示: long-horizon 和 OOD 任务上, interleaved planning 的优势巨大 (Stack Cubes Hard 60 vs π0 10)。Planning-Execution Gap (95% plan acc vs 73% success) 指明了下一步方向 — action expert 的精细控制能力是下一个 bottleneck。

整个工作最优雅的地方在于: 它没有发明全新的 architecture, 而是把已有的 unified multimodal model (Bagel) + flow matching + MoT expert 重新组合, 通过精心设计的 data pipeline 和 training stage, 让 emergent multimodal reasoning 能力自然地 transfer 到 embodied domain。这印证了一个趋势: embodied AI 的突破越来越依赖于 foundation model 的 prior, 而非 task-specific 的 trick。
