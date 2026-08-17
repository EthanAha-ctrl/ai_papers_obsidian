---
source_pdf: World Action Models are Zero-shot Policies.pdf
paper_sha256: cb91d24a04717a6c77fd39594af7f3f85a05317b7078b0d9c8d8b4ddd932f483
processed_at: '2026-08-13T04:56:38-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 DreamZero

## 一句话版本

别再让 robot 从 "看到什么" 直接学 "该做什么" 了，让它先学会 "想象未来会发生什么"，再从想象里提取动作。

---

## 问题在哪

现在的 VLA (比如 GR00T, π0.5) 本质上在学一个 mapping：

> 看到画面 → 输出动作

听起来合理，但有个致命问题。你 training data 里有 "把可乐罐移到 Taylor Swift 旁边" 这个任务，robot 学会了。但你让它 "解开鞋带"，training data 里没有，它就懵了。

为什么？因为 VLM backbone 是从 static images 训出来的，它懂 "Taylor Swift 在哪"，但不懂 "手该怎么穿过鞋带孔"。**semantic 对了，physics 没了**。

---

## DreamZero 的思路

换个角度想：婴儿学抓东西的时候，并不是靠有人反复教 "这个动作对应这个 muscle activation"。婴儿是先**看**世界怎么运转，脑子里建了个 "如果手这么动，东西会那么变" 的 model，然后才学着把自己的动作往这个 model 里套。

DreamZero 干的就是这件事：

1. 先用 web-scale video 训一个 "世界会怎么演变" 的 model (这就是 Wan2.1 video diffusion model)
2. 再加一小块 "给定想象出来的未来画面，反推该发什么 motor command" (inverse dynamics)
3. 两个一起 joint train

核心公式：

$$\text{想象未来画面} \times \text{从画面反推动作} = \text{直接输出动作}$$

好处是：video model 已经在互联网上亿视频上学会了 physical dynamics (东西怎么掉、怎么抓、怎么转)，robot 只需要学 "在我的身体上，要实现这个画面变化该发什么命令"。

后者比前者简单太多了。

---

## 为什么这招管用

### 1. 不需要重复数据

传统 robot learning 的 wisdom 是：一个任务要重复采几百次 demo。

DreamZero 发现：**不需要**。500 hours 的 diverse, non-repetitive data (22 个不同场景，每集 4.4 分钟包含 42 个 subtask) 比 repetitive data 效果好得多。

道理很简单：video model 已经懂 physics 了，你需要它学的是 "我这只手在这个 configuration 下该怎么动"。Repetitive data 只会让它 overfit 到特定 configuration。Diverse data 让它学 general 的 mapping。

Ablation 数据很直接：diverse data 50% vs repetitive data 33%。

### 2. 可以 zero-shot 到新任务

10 个 training 里完全没见过的任务 (解鞋带、熨衣服、画画、握手...)：
- DreamZero: 39.5% task progress
- Pretrained VLA: 16.3%
- From-scratch VLA: <1%

为什么？因为 DreamZero 是先 "想象" 这个任务该怎么做 (video prediction)，再执行。Video model 的 generalization 能力是 VLA 没有的。

### 3. 跨身体可以只用视频

最 striking 的结果：用 12 minutes human video 或 20 minutes 别的 robot video (没有 action label！) 就能让 DreamZero 在 unseen tasks 上提升 42%。

原因：video-only data 直接 supervise 的是 marginal distribution $p(\text{未来画面}|\text{语言指令})$，加强了 world model 对 task dynamics 的理解。Action 部分还是用原 robot 的数据学。

### 4. 30 分钟换一个 robot

在 AgiBot G1 上 train 好的 model，用 30 minutes play data 就能 adapt 到 YAM robot 上，还保留了 zero-shot generalization。

这基本就是 foundation model 该有的样子了。

---

## 但有个大问题：太慢了

14B model + 16 diffusion steps = 5.7 秒出一个 action chunk。Robot 早就撞了。

DreamZero 用了一堆工程优化：
- Async execution (推理和执行并行)
- CFG 双 GPU 并行
- DiT caching (连续 velocity 相似就跳过)
- torch.compile + CUDA Graphs
- NVFP4 quantization

但最 clever 的是 **DreamZero-Flash**。

### DreamZero-Flash 的核心 insight

标准 training 时，video 和 action 用同样的 noise schedule。但 few-step inference 时，action 已经 denoise 完了，video 还 noisy 着 —— training 和 inference 不 match。

解法：training 时把 video 的 noise schedule 往 noisy 方向偏 (Beta(7,1) sampling，让 video 平均在 $t=0.125$ 的高噪声区)，让 action 保持 uniform。

这样 model 训练时就见过 "action 已经干净但 video 还很 noisy" 的情况，inference 时就 match 上了。

结果：1 step inference 从 52% (普通 DreamZero) 提升到 74% (DreamZero-Flash)，只比 4-step 的 83% 低 9%。

最终 38× speedup，7Hz real-time control。

---

## 失败模式很有意思

DreamZero 失败的时候，failure mode 是：video 生成错了，但 robot 忠实地执行了错误的 video plan。

比如让它 "拿起笔在白板上画线"，生成的 video 里 robot 拿起笔然后递给了另一只手 (没画线)，execution 真的就照着做了。

这说明：**policy 的瓶颈在 video generation，不在 action prediction**。这也意味着未来 WAM 的进步直接和 video generation 进步挂钩 —— 两个领域 unification 了。

---

## 总结成 intuition

1. **VLA 学的是 "what to do"，WAM 学的是 "what will happen" + "how to make it happen"**。后者更 generalizable 因为 physics 是 universal 的。

2. **Diverse data > Repetitive data** for WAMs。因为 IDM 需要见到多样的 state-action correspondences，repetitive data 会 overfit。

3. **AR > BD** for closed-loop robotics。AR 保留 native FPS，支持 arbitrary context length，KV cache 让 inference 快 3-4×。

4. **Training distribution 要 match inference distribution**。DreamZero-Flash 就是这个 principle 的直接应用。

5. **Video 是 bottleneck**。改善 video backbone 直接改善 WAM performance。未来 robot foundation model 的进步可能就来自 video model 的进步。

6. **Cross-embodiment 只需要 video**。这打开了一扇大门：海量 human egocentric video 可以直接用来提升 robot policy，不需要 action annotation。

> 项目主页: https://dreamzero0.github.io  
> 代码: https://github.com/dreamzero0/dreamzero

---

# DreamZero: World Action Models as Zero-shot Policies

## 1. 核心直觉: 为什么 WAMs 不同于 VLAs

Andrej，让我从最根本的直觉说起。这篇 paper 解决了一个困扰 VLA (Vision-Language-Action) models 的核心矛盾: VLA 继承了 VLM 的 semantic priors (比如识别 "Taylor Swift" 在哪里)，但是**缺乏** spatiotemporal priors —— VLM 是从 static image-text pairs 训练出来的，根本不懂 "physical dynamics"。

DreamZero 的核心 insight 是: 把 robot policy learning 从 state-action imitation 转化为 **inverse dynamics learning**。具体来说，joint prediction $\pi_0(\mathbf{o}_{l:l+H}, \mathbf{a}_{l:l+H} | \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)$ 可以分解为:

$$\underbrace{\pi_0(\mathbf{o}_{l:l+H}, \mathbf{a}_{l:l+H} | \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{DREAMZERO}} = \underbrace{\pi_0(\mathbf{o}_{l:l+H} | \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{video prediction}} \underbrace{\pi_0(\mathbf{a}_{l:l+H} | \mathbf{o}_{0:l+H}, \mathbf{q}_l)}_{\text{IDM}}$$

**变量解释**:
- $\mathbf{o}_{0:l}$: 从 step 0 到 step $l$ 的视觉 observation history (包括 past frames)
- $\mathbf{o}_{l:l+H}$: 从 step $l$ 到 $l+H$ 的未来 video frames, $H$ 是 horizon
- $\mathbf{a}_{l:l+H}$: 对应的连续 action chunks
- $\mathbf{c}$: language instruction (text condition)
- $\mathbf{q}_l$: proprioceptive state (robot joint positions etc.)
- $\pi_0$: policy distribution

关键 insight: video prediction model **已经**在 web-scale video 上学会了 physical dynamics priors。WAM 只需要额外学 inverse dynamics mapping (从 predicted visual futures → motor commands)。这比 VLA 从头学习 observation→action mapping 容易得多。

> References:
> - Project page: https://dreamzero0.github.io
> - Code: https://github.com/dreamzero0/dreamzero
> - Flow Matching: https://arxiv.org/abs/2210.02747
> - Wan2.1: https://github.com/Wan-Video/Wan2.1

---

## 2. Architecture 深度解析

### 2.1 Backbone 选择

DreamZero 基于 **Wan2.1-I2V-14B-480P** (Team Wan, 2025)，这是一个 image-to-video diffusion model，参数量 14B。设计上有几个关键选择：

1. **Autoregressive > Bidirectional** (Q3 ablation)
2. **Chunk-wise video denoising** + **Teacher forcing**
3. **Shared denoising timestep** between video and action modalities (at beginning of training)
4. **Minimal additional parameters**: state encoder, action encoder, action decoder

为什么不直接改 backbone 结构？因为想保留 video model 的 spatiotemporal priors。multi-view 数据被 concat 到 single frame 而非 architecture change。

### 2.2 Autoregressive 的优势

Figure 13 解释了为什么 AR 比 BD (bidirectional) 更适合 closed-loop robotics。核心问题: bidirectional diffusion 需要固定 length 的 sequence，当 language annotation 标注的是一个 long-horizon task 而 model 只能 generate 一小段 video interval 时，存在 **language-video misalignment**。

Bidirectional 的 "natural solution" 是 subsample video 来匹配 caption interval —— 但在 closed-loop setting 中，当 sampling point 落在 task 中间 (e.g., T=20)，subsample 会 distort native FPS，破坏 video-action alignment。

AR 通过 conditioning on video context (而非 subsampling) 完美 sidesteps 这个 dilemma：
- (1) Faster inference via KV-cache
- (2) Policy 能利用 visual history 作为 guidance
- (3) 避免 modality alignment challenges

---

## 3. Training Objective 数学细节

### 3.1 Flow Matching Formulation

DreamZero 使用 flow matching (Lipman et al., 2022; Liu et al., 2022) 作为 training objective，类似 recent video diffusion models (Ali et al., 2025; Teng et al., 2025)。

给定 chunk index $k > 0$ 和 denoising timestep $t_k \in [0,1]$，noisy video latent 和 noisy normalized action 定义为:

$$\mathbf{z}_{t_k}^k = t_k \mathbf{z}_1^k + (1-t_k) \mathbf{z}_0^k, \quad \mathbf{a}_{t_k}^k = t_k \mathbf{a}_1^k + (1-t_k) \mathbf{a}_0^k$$

**变量解释**:
- $\mathbf{z}_1^k$: clean video latent vector (from VAE encoding)
- $\mathbf{z}_0^k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian noise
- $\mathbf{a}_1^k$: clean normalized action
- $\mathbf{a}_0^k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian noise on action
- $t_k \in [0,1]$: flow matching time parameter, $t=0$ 纯噪声, $t=1$ 干净信号

**关键设计**: 同一 chunk 内所有 frames **share** 同一个 timestep $t_k$；**不同 chunks** 被分配 independent timesteps。这是 trajectory-level update 的关键 —— 类似 LLM 在 variable length tokens 上训练。

### 3.2 Joint Velocity Prediction Loss

训练 loss function:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z}, \mathbf{a}, \{t_k\}} \left[ \frac{1}{K} \sum_{k=1}^{K} w(t_k) \big\| \mathbf{u}_\theta \big( [\mathbf{z}_{t_k}^k, \mathbf{a}_{t_k}^k]; \mathcal{C}_k, \mathbf{c}, \mathbf{q}_k, t_k \big) - \mathbf{v}^k \big\|^2 \right]$$

**变量解释**:
- $\theta$: model parameters
- $w(t_k) > 0$: predefined weighting function
- $\mathcal{C}_k = \{(\mathbf{z}_1^j, \mathbf{a}_1^j)\}_{j=1}^{k-1}$: clean context from previous chunks (teacher forcing!)
- $\mathbf{c}$: text condition
- $\mathbf{q}_k$: proprioceptive state of $k$-th chunk
- $\mathbf{v}^k := [\mathbf{z}_1^k, \mathbf{a}_1^k] - [\mathbf{z}_0^k, \mathbf{a}_0^k]$: target velocity (从 noise 到 clean signal)
- $\mathbf{u}_\theta$: neural network predicting joint velocity for both modalities

**Intuition**: 这是 conditional flow matching —— 学习 vector field 从 noise distribution 到 data distribution，conditioned on clean history context, language, and proprioception。

### 3.3 Teacher Forcing Mechanism

Teacher forcing (Gao et al., 2024; Jin et al., 2024) 是关键: model denoise noisy **current** chunk conditioned on **clean** previous chunks。这避免了训练时 error propagation，同时允许 inference 时用 ground-truth observation 替换 predicted frames (在 KV cache 中)。

---

## 4. Real-time Inference: DreamZero-Flash 的 Magic

### 4.1 The Reactivity Gap

Naive DreamZero 在 single GPU 上需要 ~5.7 seconds per action chunk —— 三大 bottlenecks:
1. 16 diffusion steps for smooth actions
2. 14B DiT backbone compute
3. Sequential execution blocking robot motion

### 4.2 Asynchronous Closed-Loop Execution

关键 insight: **decouple inference from action execution**。Motion controller 持续执行 most recent action chunk 同时 inference 在 latest observation 上 concurrent 运行。这把 latency constraint 从 "inference 必须完成才能 robot 移动" 转化为 "inference 必须在 current action chunk expires 之前完成"。

对于 48-step action horizon at 30Hz (1.6s per chunk)，需要 <200ms inference latency。

### 4.3 System-level Optimizations

1. **CFG Parallelism**: classifier-free guidance (Ho & Salimans, 2022) 需要 conditional + unconditional 两次 forward pass。跨 2 GPUs 并行，减 47% per-step latency。
2. **DiT Caching**: 利用 velocity predictions 的 directional consistency。当 successive velocities 的 cosine similarity 超过 threshold $\epsilon$，reuse cached velocities，把 effective DiT steps 从 16 减到 4。

> References:
> - CFG: https://arxiv.org/abs/2207.12598
> - TeaCache: https://arxiv.org/abs/2411.19108
> - TaylorSeer: https://arxiv.org/abs/2503.06923

### 4.4 Implementation-level Optimizations

- **torch.compile + CUDA Graphs**: eliminate CPU overhead, fuse operators
- **NVFP4 Quantization** (Blackwell): weights/activations to NVFP4, sensitive ops (QKV, Softmax) to FP8, non-linear to FP16
- **cuDNN backend** for attention via PyTorch SDPA
- **Scheduler migration to GPU**: eliminate CPU-GPU sync stalls

### 4.5 DreamZero-Flash: Decoupled Noise Schedules (核心创新)

**Problem**: 标准 DreamZero 采样 $t_k \sim \mathcal{U}(0,1)$ for both modalities。但 few-step inference (e.g., 4 steps) 时，video tokens 还 inaccurate，provide noisy conditioning signal for action prediction。这造成 train-test mismatch: training 时 model 学的是 "predict actions when video and action at same noise level"，但 inference 时需要 "predict clean actions while video still partially noisy"。

**Solution**: bias video timesteps toward high-noise states:

$$t_k^{\text{video}} = 1 - \eta, \quad \eta \sim \text{Beta}(\alpha, \beta), \quad t_k^{\text{action}} \sim \mathcal{U}(0,1)$$

where $\alpha > \beta$ (e.g., $\alpha=7, \beta=1$)。For Beta(7,1)，$\mathbb{E}[\eta] = 0.875$，所以 $\mathbb{E}[t_k^{\text{video}}] = 0.125$ (mostly noisy!)。而 action timesteps 保持 uniform $\mathbb{E}[t_k^{\text{action}}] = 0.5$。

**直觉**: 这把 model 暴露到 "action 必须从 noisy visual context 中预测出来" 的 configurations，正好匹配 few-step inference regime —— action 从 $t=1$ denoise 到 $t=0$ in one step，while video remains partially noisy。

实验结果 (Table 3): 4-step DreamZero: 83% task progress；1-step DreamZero: 52%；1-step DreamZero-Flash: **74%** —— 只 drop 9%，但 2.33x faster！

---

## 5. 实验 Data 深度解析

### 5.1 AgiBot G1 Pretraining Data

500 hours teleoperation data across **22 environments** (homes, restaurants, supermarkets, coffee shops, offices, warehouses, labs, hotels)。每个 episode ~4.4 minutes，包含 ~42 subtasks —— **significantly longer-horizon** than typical robotic datasets。

Skill distribution: navigation, torso adjustments, manipulation —— 反映 real-world deployment requirements。

**关键 data philosophy**: 优先 diversity over repetition。一旦 task 在 50 episodes 中出现，就被 deprecated，teleoperators 被 incentivized 提出 new tasks。这是 **active long-tail expansion**！

> 这与你 Karpathy 经常讲的 "data is the new code" philosophy 完全一致 —— 这里 data diversity > data volume for generalization.

### 5.2 Main Results (Figure 8)

**Q1: Do WAMs learn better from diverse, non-repetitive data?**

On AgiBot G1 (seen tasks, unseen environments + unseen objects):
- **DreamZero**: 62.2% average task progress
- **Pretrained VLA** (GR00T N1.6, π0.5): 27.4%
- **From-scratch VLA**: near-zero

2x improvement！而且 baseline 是 pretrained on **thousands of hours** cross-embodiment data，DreamZero 只用 500 hours same-embodiment data。

On DROID-Franka: similar pattern —— DreamZero outperforms pretrained baselines trained on multiple robot embodiment data。

### 5.3 Zero-shot Unseen Task Generalization (Figure 9)

10 tasks **entirely absent** from pretraining (untying shoelaces, ironing, painting, cube stacking, removing hat from mannequin, shake hands):
- **DreamZero**: 39.5% (strong on "Remove Hat from Mannequin" 85.7%, "Shake Hands" 59.2%)
- **Pretrained VLAs**: 16.3%
- **From-scratch VLAs**: <1%

On DROID: 49% task progress, 22.5% success rate vs GR00T N1.6 (31%, 12.5%) and π0.5 (33%, 7.5%)。

### 5.4 Cross-Embodiment Transfer (Table 2)

Using only **video prediction objective** (no actions!) for cross-embodiment data:
- **DreamZero baseline**: 38.3% ± 7.6%
- **+ Human2Robot** (12 min human video): 54.3% ± 10.4%
- **+ Robot2Robot** (20 min YAM video): 55.4% ± 9.5%

42%+ relative improvement with just 10-20 minutes of video-only data！

**Intuition**: 因为 WAM 学的是 $p(\mathbf{o}_{t:t+H}, \mathbf{a}_{t:t+H} | \mathbf{o}_{0:t}, \mathbf{c})$ —— joint distribution。Cross-embodiment video-only data **strengthens** the world model's understanding of task dynamics and expected behavior，while maintaining the AgiBot joint video-action learning。

### 5.5 Few-shot Embodiment Adaptation (Figure 12)

Post-trained DreamZero-AgiBot on YAM robot with only 55 trajectories / 11 tasks (~30 minutes)。
结果: 仍 retain strong language following，generalize 到 novel objects (pumpkins, teddy bears, pens, cup noodles, paper bags)。

**Hypothesis**: (1) visual similarity of AgiBot G1 and YAM (both bi-manual parallel grippers)；(2) 更 fundamentally，learning implicit IDM from predicted videos is inherently more sample-efficient than direct policy learning —— model 只需 learn mapping from visual futures → actions，while leveraging pretrained video model's existing understanding of physical dynamics。

---

## 6. Ablation Studies (Table 4)

| Architecture | Model Size | Data | Task Progress |
|---|---|---|---|
| DREAMZERO (AR) | 14B | Repetitive | 33% ± 4.2% |
| DREAMZERO (AR) | 14B | Diverse | **50% ± 6.3%** |
| DREAMZERO (AR) | 5B | Diverse | 21% ± 4.2% |
| VLA | 5B | Diverse | 0% ± 0.0% |
| VLA | 14B | Diverse | 0% ± 0.0% |
| DREAMZERO (BD) | 14B | Diverse | 50% ± 14.4% |
| DREAMZERO (AR) | 14B | Diverse | 50% ± 6.3% |

**Key insights**:
1. **Data diversity > repetition** (33% → 50%) —— 因为 robust IDM 需要 diverse state-action correspondences
2. **WAM scales with model size** (21% → 50%)，而 VLA **doesn't** —— scaling VLA capacity alone doesn't fix diverse data difficulty
3. **AR vs BD**: similar task progress，但 AR 产生 substantially smoother motions + 3-4x faster inference (KV caching)。BD 的 std (14.4%) 远高于 AR (6.3%) —— 训练不稳定！

---

## 7. Inference Speedup Breakdown (Table 1)

| Optimization | H100 | GB200 |
|---|---|---|
| Baseline | 1× | 1.1× |
| + System-level (CFG Parallelism + DiT Caching) | 1.9×, 5.5× | 1.8×, 5.4× |
| + Implementation-level (torch compile + CUDA Graphs + Kernel & Scheduler + NVFP4) | 8.9×, 9.6× | 10.9×, 14.8×, 16.6× |
| + Model-level (DREAMZERO-Flash) | — | — |

**最终**: 38× speedup on GB200，5.7s → 150ms latency，enabling **7Hz real-time closed-loop control**！

---

## 8. Failure Case Analysis (Figure 16)

非常 interesting 的 failure analysis:
- AgiBot: DreamZero 生成 video 显示 robot 用 left arm pick up marker 然后 pass 给 right arm。但实际 execution 时，robot pick up marker top 部分但没 draw line on whiteboard，反而 pass marker to right arm —— faithful to generated video plan，但 plan 本身错了
- DROID: Generated video 显示 robot pick up bread instead of opening oven first。Execution faithful to (wrong) plan，robot 拿着 bread stuck at oven

**关键 insight**: 失败主要来自 **video generation errors**，而**非** action prediction errors。Policy faithfully executes whatever trajectory video predicts。这暗示 improving video backbone would directly translate to better WAM performance。

---

## 9. 与 Alternative World Model Architectures 的对比 (Appendix A)

### 9.1 Latent-Space World Models (JEPAs, Dreamer)

- V-JEPA 2 (Assran et al., 2025): predict future states in abstract latent spaces, achieve zero-shot planning after 62 hours
- Dreamer series (Hafner et al., 2019, 2020, 2023, 2025): compact latent dynamics for model-based RL

**Key distinction**: 这些方法 model forward dynamics $p(s_{t+1}|s_t, a_t)$，**require** separate IDM or explicit planning/search at deployment。

### 9.2 3D Point Cloud World Models

PointWorld (Huang et al., 2025): unified state+action in 3D spatial domain，embodiment-agnostic，but **also requires** MPPI sampling at inference。

### 9.3 WAM 的独特优势

WAMs jointly model $p(\mathbf{o}_{t:t+H}, \mathbf{a}_{t:t+H} | \mathbf{o}_{0:t}, \mathbf{c})$ —— **directly producing** action trajectories aligned with predicted visual futures，**without test-time optimization**。这是 enable 7Hz real-time control 的根本原因。

> References:
> - V-JEPA 2: https://arxiv.org/abs/2506.09985
> - Dreamer V3: https://arxiv.org/abs/2301.04104
> - PointWorld: https://arxiv.org/abs/2601.03782

---

## 10. Future Directions & Open Questions

### 10.1 Scaling Laws for WAMs

Paper 明确指出 WAM scaling laws (Kaplan et al., 2020 style) 还未被 fully explored。Hypothesis: WAMs 可能有 more direct scaling law for actions than VLAs —— 因为 action quality directly tied to video generation quality。

### 10.2 Learning from In-the-wild Human Data

当前 cross-embodiment 实验只用 12 minutes human data。Hypothesis: leveraging large-scale egocentric human video datasets (Ego4D, Action100M, Egodex) could enable WAMs to acquire diverse skills **without action annotation**。

> References:
> - Ego4D: https://arxiv.org/abs/2110.07058
> - Action100M: https://arxiv.org/abs/2601.10592

### 10.3 Long-Horizon Reasoning

DreamZero 当前是 System 1 model，short-horizon (6 seconds)。需要 System 2 planner 或者 WAM with extended context windows。Candidate approaches: modular dual-system (Shi et al., 2025) 或 unified (Deng et al., 2025)；video-based world models maintaining coherent generation over extended horizons (Ball et al., 2025; HunyuanWorld, 2025)。

### 10.4 Embodiment Design for WAMs

两个 opposing factors:
1. **Degrees of freedom**: Higher-DOF → more play data needed for accurate implicit IDM
2. **Human similarity**: Humanoids with dexterous manipulation may transfer more efficiently despite higher DOF，because can leverage both video pretraining priors **and** massive scale of human egocentric videos

**Hypothesis**: Human-like embodiments may win out by trading mechanical simplicity for access to web-scale human data —— the fuel for next-generation robot foundation models。

---

## 11. 我的 Intuition 构建

让我尝试为这篇 paper 构建 intuition:

### 11.1 为什么 joint video+action prediction 比 separate 更好？

考虑两个 separate models:
- Video model: $p(\mathbf{o}_{t:t+H}|\mathbf{o}_{0:t}, \mathbf{c})$
- IDM: $p(\mathbf{a}_{t:t+H}|\mathbf{o}_{0:t+H}, \mathbf{q}_t)$

Training separate models 时，IDM 永远看不到 video model 的 prediction errors。但 joint training 时，model 必须学会 **handle** video prediction uncertainty 同时 **predict** actions —— 这正是 deployment 时遇到的 regime！

### 11.2 为什么 diverse data >> repetitive data for WAMs？

考虑 IDM learning curve:
- Repetitive data: model 见过相同 (state, action) pairs，overfit specific configurations
- Diverse data: model 必须学 **general** mapping from visual futures → motor commands

类比: training LLM on diverse internet text >> training on repetitive single-domain text。WAMs 的 "language" 是 spatiotemporal dynamics —— 需要多样 "sentences" 才能学到 general grammar。

### 11.3 为什么 DreamZero-Flash 的 decoupling 有效？

回到 flow matching 的 physics:
- $t=0$: 纯噪声
- $t=1$: 干净信号

Few-step inference 时，action 在 $t=1 \to 0$ 之间快速 denoise，但 video 因为 high-dimensionality 收敛慢。如果 training 时 model 只见过 $t_{\text{video}} = t_{\text{action}}$，它会 expect clean video when predicting clean action。

DreamZero-Flash 的 Beta(7,1) sampling 让 model 见过 $t_{\text{video}} \approx 0.125$ (noisy) + $t_{\text{action}} \approx 0.5$ (partial) 的 configurations —— 正好匹配 1-step inference 时 video 还没收敛但 action 已经 denoise 完的 regime！

这是 **distribution matching** between training and inference —— classic ML 的 fundamental principle。

### 11.4 为什么 cross-embodiment transfer 用 video-only 就够？

因为 WAM 学的是 **joint distribution** $p(\mathbf{o}, \mathbf{a} | \mathbf{c})$。Marginalize out action: $p(\mathbf{o}|\mathbf{c}) = \int p(\mathbf{o}, \mathbf{a}|\mathbf{c}) d\mathbf{a}$。

Video-only data **directly** supervises the marginal $p(\mathbf{o}|\mathbf{c})$ —— strengthening the world model's understanding of task dynamics。Then AgiBot data supervises the conditional $p(\mathbf{a}|\mathbf{o}, \mathbf{c})$ for the target embodiment。

这是 **semi-supervised learning** 的 elegant case —— unlabeled (video-only) modalities from other embodiments **regularize** the shared latent space。

---

## 12. 关键 Takeaways

1. **WAMs vs VLAs**: VLA 学 semantic → motor mapping，WAM 学 physical dynamics + inverse dynamics。前者 overfit semantics，后者 generalize via shared physics。

2. **Diversity > Repetition**: for generalist policies, the conventional wisdom of "repeated demos per task" is wrong for WAMs. Diverse, heterogeneous data enables better generalization.

3. **AR > BD**: for closed-loop robotics, autoregressive architecture with KV-caching is fundamentally better than bidirectional —— preserves native FPS, supports arbitrary context length, avoids modality misalignment.

4. **Decoupled noise schedules**: for few-step inference, training distribution must match inference distribution. DreamZero-Flash's Beta(7,1) for video + uniform for action bridges this gap.

5. **Video-only cross-embodiment**: joint distribution formulation means unlabeled video from other embodiments directly improves world model —— semi-supervised scaling pathway via human egocentric video.

6. **Failure mode = video errors**: policy faithfully executes predicted video plans. This means future WAM progress directly tracks video generation progress —— unifying two seemingly separate research directions.

> **Open source**: https://github.com/dreamzero0/dreamzero
> **Project page**: https://dreamzero0.github.io
> **Training data gallery**: https://dreamzero0.github.io/training_data_gallery/

---

这篇 paper 在我看来是 robotics foundation models 的 paradigm shift —— 从 "scale up VLA" 到 "leverage video foundation models"。最 striking 的数字是 30 minutes of play data for new embodiment adaptation with zero-shot generalization retained —— 这才是真正的 foundation model behavior！
