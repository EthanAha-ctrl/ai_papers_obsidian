---
source_pdf: Fast-dDrive Efficient Block-Diffusion VLM for.pdf
paper_sha256: c07b34d1f29fd3686a6abb3e9e21be84e2ee017761683444e9a3886caf3b9091
processed_at: '2026-08-04T07:06:35-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最直白的大白话来拆解这篇 paper，核心就是讲了一件事：**怎么让又大又慢的 Vision-Language-Action (VLA) 模型，在车上的芯片里跑得飞快，同时还不丢精度，甚至逻辑更严密。**

为了 build your intuition，我们把这篇 paper 想象成在改造一条“自动驾驶流水线”。

### 1. 为什么要改造？旧流水线有什么毛病？

现在的 E2E autonomous driving VLA 主要是 Autoregressive (AR) 模式，就像挤牙膏，一次吐一个 token。
- **毛病 1：慢得要命 (Memory-bandwidth bound)**。在 batch size 为 1 的车机上，你为了算 1 个 token，得把整个 3B 参数的 model weights 从内存搬一遍。算 400 个 token 就得搬 400 遍，算力都在等内存搬运，所以 TPS 只有可怜的 50 左右。
- **毛病 2：一步错，步步错 (Exposure bias)**。轨迹是按时间顺序生成的，如果第 1 秒的坐标算歪了一点点，第 2 到 5 秒的坐标基于这个错误继续算，最后车子可能就冲出路面了。

为了解决这两个毛病，有人提出了 Full-sequence Diffusion (比如 dVLM-AD)。这就像做填空题，一次性把所有答案都蒙上，然后反复修改打磨。
- **优点**：全局视野，轨迹首尾呼应，不会出现前面错后面崩的情况。
- **毛病 1：慢上加慢 (No KV-cache reuse)**。因为每个 token 都在看所有其他 token，你每修改一次，整个 sequence 的 attention 就得全算一遍，完全没法用 KV cache 偷懒。
- **毛病 2：逻辑穿越 (Logical leakage)**。驾驶逻辑必须是“先看到行人，再决定刹车”。但全序列 diffusion 里，未来的“刹车轨迹”居然能反过来影响“看到行人”这个感知结果。模型为了合理化自己的错误轨迹，甚至会篡改前面的感知输出，这在 safety 上绝对不能接受。

### 2. Fast-dDrive 的破局之道：按块干活 (Block Diffusion)

既然“完全串行 (AR)”和“完全并行”都不行，Fast-dDrive 采取了折中方案：**Block-Causal Diffusion**。

把一整段输出切成几个 Block。Block 内部可以互相看 (bidirectional attention)，Block 之间必须严格按顺序来 (causal attention)。
- **直觉**：这完美匹配了驾驶的因果律。感知、解释、决策、轨迹，四大块。感知块内部互相参考定下来之后，才传给解释块。未来的轨迹块绝对看不到过去的感知块，彻底杜绝了“逻辑穿越”。
- **好处**：前面 Block 算完的 KV cache 可以存起来给后面用，不用重算，速度起飞。

### 3. 三个神级优化技巧

光有 Block Diffusion 还不够，作者针对驾驶任务的特点，加了三个极其巧妙的 trick。

#### Trick 1: Scaffold (搭脚手架，直接填空)
驾驶 VLA 输出的是个 JSON。JSON 里的括号、引号、Key 名字 (比如 `"trajectory":`) 全是固定的废话，只有 Value (比如具体的坐标 `[1.2, 3.4]`) 才是模型需要算的。这些废话占了 30% 的 token。

**做法**：直接把这些废话 token 当成 Scaffold (脚手手架) 冻结住。训练时永远不 mask 它们，推理时直接 pre-fill 进去。
**直觉**：这就好比做填空题，题干已经印好了，模型只需要填那几个空。不仅省了 30% 的计算量，还保证了 JSON 格式 100% 不会写错 (structural validity)。

#### Trick 2: SASD (区别对待，好钢用在刀刃上)
输出的四个部分重要性天差地别。轨迹算错会撞车，解释写得啰嗦一点完全无所谓。

**做法**：引入 Section-Importance-Weighted Loss (IWL) 和 Section-Adaptive Noise Schedule (SNS)。
看公式 $\mathcal{L}_{\mathrm{train}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t} \left[ -\sum_{s} \frac{w_s}{|\mathcal{M}_t^s|} \sum_{i \in \mathcal{M}_t^s} \log p_\theta(x_0^i \mid \mathbf{x}_t, \mathbf{c}) \right]$
- $s$ 代表不同的 section (CO/Expl/FMB/Traj)。
- $w_s$ 是权重。轨迹 $w_{\mathrm{traj}} = 3.0$，解释 $w_{\mathrm{explanation}} = 1.0$。
- $\mathcal{M}_t^s$ 是 step $t$ 时 section $s$ 里被 mask 的位置。
- **直觉**：训练的时候，轨迹算错重罚三倍，解释算错轻轻放过。同时，给轨迹部分加更狠的噪声 (Beta(2,1) 偏向高噪声)，逼着模型在极端恶劣的情况下也能把轨迹算对。这套操作只在 training 时做，inference 时零开销。

#### Trick 3: Scaffold Speculative Decoding (草稿与审核)
这是提速最猛的一招。借鉴了 self-speculative decoding 的思路，一套 weights 两个头：MDM head (并行做草稿) + AR head (串行做审核)。

**流程**：
1. 遇到一个 Block，Scaffold token 直接免检通过。
2. MDM head 一次性把所有 Value token 猜出来 (Draft)。
3. AR head 挨个检查 (Verify)。如果猜对了，皆大欢喜；如果第 3 个猜错了，前 2 个收下，第 3 个用 AR 自己的答案，后面全作废，白送 1 个 bonus token。
- **直觉**：就像老板写报告，秘书先把一段话全打出来，老板扫一眼。大部分时候秘书打得全对，老板直接签字通过 (相当于 1 次 forward pass 吐出几十个 token)。偶尔打错，老板改个错别字继续看。由于 Scaffold 免检，且 Block 对齐了语义，秘书的命中率极高，速度直接起飞。

### 4. 廉价换取高精度：Test-Time Scaling

文章还提供了一个“花小钱办大事”的 inference 方案：Shared-Prefix Multi-Trajectory Rollouts。

前面的感知、解释、决策是确定的 (只算一次，存好 KV cache)。到了轨迹部分，把 AR verifier 的温度调高一点，随机采样 $N$ 次 (比如 4 次)，得到 4 条略微不同的轨迹，然后直接求平均 $\tau_{\mathrm{out}} = \frac{1}{N} \sum_{i=1}^{N} \tau^{(i)}$。

- **直觉**：前面想好的策略不变，最后落脚的时候稍微犹豫一下，多画几条线，取个平均值。根据 variance-of-the-mean argument，方差直接降为 $1/N$。代价极小 (只重算最后那一小截轨迹)，精度还能再提 3%。

### 5. 实验数据撑腰

看 Table 4 的效率对比，直观感受有多猛：
- **AR baseline**: Latency 7855 ms，TPS 51.6 (慢)。
- **dVLM-AD (全序列 diffusion)**: Latency 9575 ms，TPS 35.2 (比 AR 还慢！)。
- **Fast-dDrive (Scaffold Spec)**: Latency 1919 ms，TPS 210.4 (快了 4 倍，精度还更高)。
- **Fast-dDrive + SGLang**: Latency 665 ms，TPS 608.5 (配合系统层优化，直接快了 12 倍！)。

在 WOD-E2E 和 nuScenes 两个数据集上，精度都是 SOTA。这意味着，原本只能放在云端的 3B 大模型，现在真正有了塞进车机实时跑的潜力。

### 6. 架构图与实验表深度拆解

为了让你彻底 build intuition，我们再深挖一下技术细节。

**Figure 2 架构图解析**:
图里展示的是 Training pipeline。
- 最前面是图像和文本的 context $\mathbf{c}$。
- 输出 sequence 被切成了四段：CO, Expl, FMB, Traj。
- 灰色的格子是 Scaffold tokens (预先填好，永远不 mask)。
- 白色的格子是 Value tokens。
- 虚线框代表 Block。Block 内部白色格子之间是双向箭头 (bidirectional attention)。
- Block 和 Block 之间是单向箭头 (causal attention)，比如 Traj block 只能看 FMB block，不能倒着看。
- 图下方的公式 $\mathcal{L} = \alpha \mathcal{L}_{\mathrm{train}}(\theta) + \beta \mathcal{L}_{\mathrm{AR}}(\theta)$ 说明这是双流训练，$\alpha=\beta=0.5$，Diffusion 和 AR 各占一半。这保证了同一套 weights 既能跑纯 Diffusion，又能跑 Speculative Decoding。

**Table 2 (WOD-E2E 测试集) 解析**:
- Top AR 方法 Poutine-Base 的 RFS 是 7.909，ADE@5s 是 2.940。
- Fast-dDrive (Scaffold Spec) 的 RFS 是 7.823 (略低一点点)，但 ADE@5s 是 2.907 (更准)，最关键的是 TPS 达到 210.4，是 Poutine 的 4 倍多。
- 加上 Inference scaling (N=4) 后，ADE@5s 降到 2.821，RFS 几乎持平。说明加码算力主要提升的是轨迹末端的绝对精度。

**Table 5 (Ablation Study) 解析**:
- 只有 IWL (Loss 加权) 时，RFS 7.807。
- 只有 SNS (Noise 调度) 时，RFS 7.855。
- 两者都开 (IWL + SNS) 时，RFS 达到 7.916。
- 证明这两个 trick 是互补的。IWL 直接放大关键 token 的梯度，SNS 则让模型在难对付的噪声区间多练，组合起来效果最好。

### 7. 更细节的技术联想与延展思考

这篇 paper 的设计充满了一种工程上的“妥协与折中”美感，我们可以顺着这个思路往下联想。

**联想 1: Structured Output Generation 的新范式**
以前让 LLM 输出合法 JSON，通常用 grammar-constrained decoding (在 logits 层面加 mask，比如 XGrammar)。Fast-dDrive 反其道而行之，直接在 input 端 pre-fill Scaffold。这种思路完全可以推广到所有的 Agent 场景。比如 tool use，function name 是固定的 (Scaffold)，只有 arguments 需要算 (Value)。如果在 block diffusion 架构下做，效率会高得惊人。

**联想 2: Continuous Diffusion 处理 Trajectory 的遐想**
现在的轨迹是 tokenize 成离散 string 算的 (比如 `+003.30`)。这其实有 quantization error。既然最后有 JMT (Jerk-Minimizing Trajectory) 做平滑插值，不如在 trajectory section 直接用 continuous diffusion 生成连续坐标。感知和解释用 discrete MDM，轨迹用 continuous diffusion。这就成了一个 Hybrid Diffusion VLA。相关的 cross-pollination 可以参考 Diffusion Policy 在 robotics 上的应用 (https://arxiv.org/abs/2303.04137)。

**联想 3: Closed-loop 下的 KV cache 困境**
Paper 里提到的 KV cache reuse 在 open-loop (数据集测试) 下完美成立。但到了 closed-loop (仿真器或实车)，下一帧的输入图像变了，前缀变了，KV cache 还能复用吗？严格来说，前一帧的 KV cache 在下一帧完全无效。这就需要一种 cross-frame KV reuse 机制，比如把 ego motion 做个 transform 补偿，或者依赖视觉 encoder (如 Qwen2.5-VL 的 native vision encoder) 提取稳定的 token 特征。

**联想 4: Block Size 的数学最优解**
Paper 里 Block size 是被 Section 强制对齐的 (Traj 有 3 个 block，每个大约 23 个 token)。但 Block Diffusion 原始论文 (https://arxiv.org/abs/2503.09573) 指出，block size 和 acceptance rate 之间有个数学上的 sweet spot。Block 太大，MDM draft 一次猜几十个 token，AR verify 时容易在中间崩掉，导致 bonus token 只有 1 个，效率暴跌；Block 太小，退化成 AR。Section-aligned 破坏了这个数学最优。如果能设计一种 dynamic block sizing，在保证 Section causal 的前提下，细调 block 边界，acceptance rate 应该还能再提一截。

**联想 5: Shared-Prefix Rollout 与 Bayesian Optimal Estimator**
在 trajectory section 跑 N 次 rollout 然后求平均，本质上是 Monte Carlo 采样求后验期望。如果我们不取 mean，改取 median，或者用 DBSCAN 聚类后取最大簇的中心，会不会更 robust？Mean 对 outlier 太敏感了，如果某一次 rollout 直接 hallucinate 了一个急转弯，mean 轨迹就废了。这种 robust aggregation 在 diffusion planner (比如 DiffusionDrive, https://arxiv.org/abs/2411.15239) 里有很多先例。

**联想 6: Causal Confusion 的隐含解药**
Causal confusion 是 IL (Imitation Learning) 里的经典 bug，模型学了个捷径，比如看车速表猜刹车，而不是看前面有红灯。Block Diffusion 强制 trajectory block 只能通过 perception block 获取信息，这就阻断了 model 从 ego-state 直接拟合 trajectory 的捷径。这在无形中起到了 regularizer 的作用，虽然 paper 没把这点作为主卖点，但对实际 driving model 的 generalization 极其重要。

### 8. Reference Links

- Block Diffusion 原理: https://arxiv.org/abs/2503.09573
- Fast-dVLM 架构基础: https://arxiv.org/abs/2604.06832
- Fast-dLLM 加速机制: https://arxiv.org/abs/2505.22618
- LLaDA 基础 Diffusion LM: https://arxiv.org/abs/2502.09992
- dVLM-AD 对照基线: https://arxiv.org/abs/2512.04459
- Qwen2.5-VL 模型 backbone: https://arxiv.org/abs/2502.13923
- SGLang 推理引擎: https://arxiv.org/abs/2312.07104
- Waymo WOD-E2E 数据集: https://arxiv.org/abs/2510.26125
- DiffusionDrive 轨迹规划联想: https://arxiv.org/abs/2411.15239
- Diffusion Policy 机器人动作联想: https://arxiv.org/abs/2303.04137

总结一句，Fast-dDrive 通过 Scaffold 填空 + 切块干保因果 + 草稿审核提速度，把大模型 VLA 的推理延迟从秒级压到了百毫秒级，是一次工程与算法结合的极佳 demo。

---

# Fast-dDrive: Block-Diffusion VLA for Autonomous Driving — 深度技术解析

下面我会从 motivation 出发,把这篇 paper 的每一条技术脉络拆开,补足公式、变量、architecture intuition、与相关工作的联系,以及我自己的延伸思考。

---

## 1. Problem Framing: 为什么 driving VLA 需要 third paradigm

### 1.1 E2E autonomous driving 的两难

End-to-end driving 把 perception、reasoning、planning 合到一个 trainable system 里。VLAs (Vision-Language-Action models) 进一步用 natural-language CoT 来解释决策,处理 long-tail scenarios。代表工作:

- DriveVLM (Tian et al. 2024, https://arxiv.org/abs/2402.12289)
- DriveCoT (Wang et al. 2024, https://arxiv.org/abs/2403.16996)
- AutoVLA (Zhou et al., NeurIPS 2025, 见 https://arxiv.org/abs/2509.20710)
- Poutine (Rowe et al. 2025, https://arxiv.org/abs/2506.11234)
- dVLM-AD (Ma et al. 2025, https://arxiv.org/abs/2512.04459)

但 deployment 要同时满足:
- trajectory 全局一致 (reasoning 和 plan 不能矛盾)
- batch-size-1 latency 必须够低,能在车上跑

### 1.2 AR VLA 的两个根本性 pain points

**Pain point 1: Memory-bandwidth bound at batch-size 1**

对 AR decoding,每个 forward pass 只 commit 1 token,但要把整个 model weights 从 HBM load 一遍。H100 的 HBM bandwidth 大约 3.35 TB/s,3B 模型 (FP16 ≈ 6 GB) 的 weight load 时间 ≈ 6 GB / 3.35 TB/s ≈ 1.8 ms/token。这是 strict lower bound,对应论文里 AR baseline 的 51.6 TPS (1000 ms / 1.8 ms ≈ 555 tokens/s 是理论上限,但 attention + projection + activation 的开销拉低到 ~50 TPS)。

**Pain point 2: Exposure bias on waypoints**

Driving VLA 通常输出 5 s 的 trajectory,离散成 waypoints tokens。AR factorization 是:

$$p(\mathbf{x}_0) = \prod_{i=1}^{L} p(x_0^i \mid x_0^{<i}, \mathbf{c})$$

如果前面 waypoint token 有小误差,后面 token condition 在 noisy context 上,error compounding,5s plan 后段可能变得 physically implausible。这正是 Huang et al. 2025 hallucination survey (https://arxiv.org/abs/2311.05232) 在 LM 上观察到的 exposure bias 在 driving 上的具体表现。

### 1.3 Full-sequence diffusion VLA 的两个结构性 cost

dVLM-AD 把整个 JSON response (perception + explanation + meta-behavior + trajectory) 当成一个 bidirectional denoising target。这解决了 exposure bias,但引入两个新问题:

**Cost 1: No KV-cache reuse**

Bidirectional attention 意味着每个 token 可以看其他所有 token。denoising 每步要重新计算整个 sequence 的 attention。如果 sequence 长 400 tokens,denoise 8 步,等于做了 8 次全序列 attention。AR 加 KV-cache 后,每步只需 query 新 token。差距非常大。

**Cost 2: Logical leakage (感知因果性被破坏)**

Driving output 有 inherent causal structure:
1. perception (看清楚场景)
2. explanation (基于 perception 推理)
3. meta-behavior (基于推理决定动作)
4. trajectory (基于动作生成轨迹)

如果整个 sequence 是一个 bidirectional block,trajectory tokens 在 denoising 时可以 retroactively 影响 perception tokens 的 prediction。这是"用结果倒推动因",在物理上不合理,在 driving safety 上是有害的 — model 可能输出 perception 来"合理化"它自己产生的 trajectory,而不是反过来。

---

## 2. Block Diffusion preliminaries — Fast-dDrive 的 backbone

### 2.1 Masked Diffusion Language Models (MDM) 基本公式

给定 target sequence $\mathbf{x}_0 = (x_1, \ldots, x_L)$ 和 conditioning context $\mathbf{c} = (\mathbf{v}, \mathbf{p})$ (visual features + text prompt)。

**Forward process**: 按照噪声调度 $\{\lambda_t\}_{t=1}^{T}$,把 tokens 随机替换成 `[MASK]`,得到 corrupted sequence $\mathbf{x}_t$。

**Reverse process**: 用 denoising policy $p_\theta$ 预测每个 masked position 的原始 token。

**Training objective** (Eq.1):

$$\mathcal{L}_{\mathrm{MDM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t} \left[ -\frac{1}{|\mathcal{M}_t|} \sum_{i \in \mathcal{M}_t} \log p_\theta(x_0^i \mid \mathbf{x}_t, \mathbf{c}) \right]$$

变量解析:
- $t$: diffusion step index,从 1 到 $T$
- $\mathbf{x}_0$: clean target sequence (ground truth)
- $\mathbf{x}_t$: 在 step $t$ 被 corrupt 后的 sequence
- $\mathcal{M}_t = \{i : x_t^i = \texttt{[MASK]}\}$: step $t$ 时所有被 mask 的 position index 集合
- $|\mathcal{M}_t|$: 该 step 被遮蔽的 token 数
- $p_\theta(x_0^i \mid \mathbf{x}_t, \mathbf{c})$: 神经网络对 position $i$ 预测的 ground-truth token 的概率

这个 loss 是 per-position cross-entropy,只在 masked position 上算。和 BERT 的 MLM 不同点在于这里是 generative 的 (denoise 从全 mask 开始),且 schedule 是连续的 noise level。

相关工作: 
- Sahoo et al. 2024 (https://arxiv.org/abs/2406.18577) 给出了 simple masked diffusion formulation
- Lou et al. 2024 (https://arxiv.org/abs/2406.11473) 用 ratio estimation 推导
- Shi et al. 2024 (https://arxiv.org/abs/2404.14457) 做了 generalized masked diffusion
- D3PM (Austin et al. 2021, https://arxiv.org/abs/2107.03006) 是早期 discrete diffusion 工作
- Diffusion-LM (Li et al. 2022, https://arxiv.org/abs/2205.11417) 是连续 diffusion 的早期尝试

### 2.2 Block-Causal Diffusion 的 attention pattern

Block Diffusion (Arriola et al. 2025, https://arxiv.org/abs/2503.09573) 把 sequence 切成 $B$ 个 block,每个 block 大小 $d$:

$$\mathbf{x}_0 = [\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_B]$$

**Attention pattern**:
- block 内 (intra-block): bidirectional attention (任两个 token 互相看)
- block 间 (inter-block): causal attention (block $j$ 能看 block $1..j-1$,不能看 $j+1..B$)

这样:
- KV cache 可以跨 block 复用 (block $j$ 计算完后,它的 KV 存着给后面 block 用)
- 每个 block 内部依然可以并行 denoise,得到 diffusion 的 speedup

直觉: 这相当于把"完全 AR"和"完全 bidirectional"两个极端 interpolate 一下。block size $d=1$ 退化成 AR,$d=L$ 退化成 full-sequence diffusion。

### 2.3 Fast-dVLM 的 self-speculative decoding

Fast-dVLM (Wu et al. 2026, https://arxiv.org/abs/2604.06832) 把 block diffusion 应用到 VLM,关键 trick 是 self-speculative decoding (来自 Fast-dLLM, https://arxiv.org/abs/2505.22618):

对每个 block:
1. **Draft (MDM head)**: 一次 forward pass,bidirectional attention,所有 masked token 同时预测 → draft $\{\tilde{x}_i\}$
2. **Verify (AR head)**: 一次 causal forward pass,逐 token 检查 draft,accept 第一个 mismatch 之前的 tokens,加上 1 bonus token

这是把 speculative decoding (Leviathan et al. 2023, https://arxiv.org/abs/2211.17192; Chen et al. 2023, https://arxiv.org/abs/2302.01318) 适配到 diffusion LM,draft 和 verifier 共享同一套 weights。MEDUSA (https://arxiv.org/abs/2401.10774) 和 EAGLE (https://arxiv.org/abs/2401.15077) 用 separate draft head,这里用 shared。

Fast-dDrive 的 Scaffold Speculative Decoding 就是这个 framework 的 driving-specific 扩展。

---

## 3. Fast-dDrive 的核心: Structure-Aware Scaffold Diffusion

### 3.1 Scaffold 的关键观察

现代 driving VLM 的输出是一个 schema-defined JSON,大致结构:

```json
{
  "critical_objects": { ... 12 个 binary detection ... },
  "explanation": "free-form reasoning text",
  "future_meta_behavior": {
    "longitudinal": "decelerate",
    "lateral": "keep lane"
  },
  "trajectory": "[[x1,y1], [x2,y2], ... 5 个 waypoints]"
}
```

**Key insight**: JSON keys、brackets、commas、quotes 都是 schema 决定的,完全 deterministic,不依赖 model。这些 token 占总输出的 ~30% (Table 1 显示 124 scaffold tokens / 404 total tokens ≈ 30.7%)。

把这些 deterministic tokens 当成 frozen scaffold $\hat{\mathbf{x}}_T$:
- 训练时永远不被 mask
- 推理时永远 pre-filled,直接 cache

**形式化**: 令 $\mathcal{A}$ 为 scaffold (anchor) positions,$\mathcal{E} = \{1, \ldots, L\} \backslash \mathcal{A}$ 为 editable value positions。Diffusion 只在 $\mathcal{E}$ 上跑 (Eq.2):

$$\mathcal{L}_{\mathrm{scaffold}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t} \left[ -\frac{1}{|\mathcal{M}_t|} \sum_{i \in \mathcal{M}_t} \log p_\theta(x_0^i \mid \mathbf{x}_t, \mathbf{c}) \right], \quad \mathcal{M}_t \subseteq \mathcal{E}$$

注意 $\mathcal{M}_t \subseteq \mathcal{E}$,意味着 mask 永远不落在 scaffold 上。

**收益**:
- 100% structural correctness by construction — JSON 永远 valid,不需要 post-process
- denoising workload 减少 30%
- model capacity 集中在真正需要预测的 value token 上

这个思路其实和 structured output generation 的 grammar-constrained decoding (e.g., Outlines, https://github.com/outlines-dev/outlines; XGrammar, https://arxiv.org/abs/2411.15159) 有内在联系,但思路反过来: grammar-constrained decoding 是在 logits 上施加约束,scaffold 是在 input 上 pre-fill。Related: "Accelerating Structured Chain-of-Thought in Autonomous Vehicles" (Gu et al. 2026, https://arxiv.org/abs/2602.02864) 这条线。

### 3.2 Section-aligned blocks — 解决 logical leakage

Fast-dDrive 把 block 边界对齐到 section 边界,得到 section-ordered blocks:

```
[CO blocks] → [Explanation blocks] → [FMB blocks] → [Traj blocks]
```

每个 section 内部允许多个 block (Table 1):
- critical_objects: 1 block (92 tokens: 12 value + 80 scaffold)
- explanation: 6 blocks (198 tokens: 192 value + 6 scaffold)
- future_meta_behavior: 1 block (24 tokens: 6 value + 18 scaffold)
- trajectory: 3 blocks (90 tokens: 70 value + 20 scaffold)
- 总计 11 blocks, 404 tokens (280 value + 124 scaffold)

**关键**: section 间是 strict causal (CO 完全生成完才生成 Explanation,以此类推),section 内是 bidirectional。这就完美匹配 driving 的 perceive-then-plan 因果性:

- perception 不能被 trajectory 影响 (没有 backward attention across sections)
- perception 可以互相参考 (bidirectional within CO section)
- explanation 可以看完整 perception (causal across sections)
- trajectory 可以看 perception + explanation + meta-behavior (causal across sections)
- trajectory waypoints 之间互相参考 (bidirectional within Traj section) — 这恰恰是我们想要的!因为 waypoints 之间物理上有相关性 (smooth trajectory),如果用 AR 一前一后,前面的误差会污染后面,正好是 exposure bias 问题

这是这篇 paper 最 elegant 的设计 — 用 block boundary 切在 section boundary,把因果结构硬编码到 attention pattern 里。

### 3.3 Section-aware training (SASD)

SASD 有两个机制,都在 training time only,zero inference overhead。

**Mechanism 1: Section-Importance-Weighted Loss (IWL)**

每个 section $s$ 赋一个权重 $w_s$,loss 变成 (Eq.3):

$$\mathcal{L}_{\mathrm{train}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t} \left[ -\sum_{s} \frac{w_s}{|\mathcal{M}_t^s|} \sum_{i \in \mathcal{M}_t^s} \log p_\theta(x_0^i \mid \mathbf{x}_t, \mathbf{c}) \right]$$

变量:
- $s$: section index (CO / Expl / FMB / Traj)
- $w_s$: section $s$ 的 loss weight
- $\mathcal{M}_t^s$: step $t$ 时 section $s$ 内被 mask 的 token positions
- $|\mathcal{M}_t^s|$: 该 step 该 section 被遮的 token 数

具体 weight (§4.1):
- $w_{\mathrm{trajectory}} = 3.0$ (最高,因为 collision risk)
- $w_{\mathrm{future\_meta\_behavior}} = 2.0$
- $w_{\mathrm{critical\_objects}} = 1.5$
- $w_{\mathrm{explanation}} = 1.0$ (最低,因为只是 reasoning text,不影响 safety)

Intuition: 这是 per-section 的 importance weighting,让 gradient 被 safety-critical section 主导。和 RLHF 里的 token-level reward 类似,但这里是 supervised 阶段。

**Mechanism 2: Section-Adaptive Noise Schedule (SNS)**

每个 section 用不同的 Beta 分布采样 noise level:

$$t_s \sim \mathrm{Beta}(\alpha_s, \beta_s)$$

Beta 分布 $B(\alpha, \beta)$ 的 PDF 是 $\frac{t^{\alpha-1}(1-t)^{\beta-1}}{B(\alpha, \beta)}$,support 在 $[0,1]$,均值 $\frac{\alpha}{\alpha+\beta}$。

具体参数:
- Trajectory: $\mathrm{Beta}(2, 1)$ — 偏向 high noise (mean 0.667),让 trajectory 在 hard denoising 上多练
- FMB: $\mathrm{Beta}(1, 1.5)$ — 偏 low noise (mean 0.4)
- CO: $\mathrm{Beta}(1, 2)$ — 偏 low noise (mean 0.333),perception 信号相对容易
- Explanation: $\mathrm{Beta}(1, 1)$ — 均匀分布

直觉: Beta(2,1) 把 sample 集中在 $t \to 1$ (heavy noise) 附近,意味着 trajectory 经常要在几乎全 mask 的状态下 denoise,逼 model 学 hard recovery。这和 MDM 里"non-uniform time sampling"的 trick 一样 — 把训练资源分配到 hard regime。

相关: Pries et al. 2024 的 noise schedule 优化,Git Re-Bart 的 difficulty-aware sampling,diffusion model 的 importance sampling (https://arxiv.org/abs/2206.13677)。

**Joint AR + Diffusion training** (Eq.4):

$$\mathcal{L} = \alpha \mathcal{L}_{\mathrm{train}}(\theta) + \beta \mathcal{L}_{\mathrm{AR}}(\theta), \quad \alpha = \beta = 0.5$$

- $\mathcal{L}_{\mathrm{train}}$: section-weighted MDM loss (diffusion branch)
- $\mathcal{L}_{\mathrm{AR}}$: 标准 causal LM loss,在 clean response labels 上算 (AR branch)
- $\alpha = \beta = 0.5$: 各占一半

为什么需要 joint training? 因为同一个模型 weights 要同时支持:
- Section Diffusion (纯 diffusion 推理)
- Scaffold Speculative Decoding (MDM head drafts + AR head verifies)

如果只训练 diffusion,AR head 退化,SS mode 不能用;反之亦然。Joint training 保留两个 head 的能力。这继承了 Fast-dVLM 的 dual-stream 设计。

### 3.4 为什么 SASD 设计成立 — schema rigidity 的 trade-off

我自己的一个思考: scaffold 冻结的代价是 model 永远不能输出 schema 外的结构。例如,如果某个场景有 30 个 critical objects 而 schema 只 hardcode 了 12 个 binary fields,就漏了。论文 §C Limitations 也提到这点。

更深的问题: schema 本身是不是该 learnable? 一种 hybrid: scaffold tokens 来自一个 schema generator (small LM) + value tokens denoise by main model。这能处理 schema 变化。类似 tool-use LLM 的 schema 灵活性。

---

## 4. Inference: 两个 mode 详解

### 4.1 Section Diffusion (SD)

这是 diffusion-only baseline:

```
prefill scaffold (一次) → 
for section in [CO, Expl, FMB, Traj]:
    for block in section.blocks:
        denoise block (MDM head, bidirectional attention within block)
        save KV cache for next block
```

特点:
- 不调 AR head
- KV cache 从 scaffold 和前面的 section/block 复用
- 每个 block 的 denoising 是迭代式 (multiple diffusion steps)

实验 (Table 4): latency 3006 ms, TPS 134.4, Tok/Step 3.28, ADE@5s 2.058, RFS 7.928

### 4.2 Scaffold Speculative Decoding (SS) — 关键算法

SS 是 paper 的核心 inference algorithm (Figure 3):

```
prefill scaffold → for each block b_j:
    1. Auto-accept scaffold: 把 b_j 内的 scaffold positions 直接 accept,不 draft 不 verify
    2. Draft (MDM head): 一次 bidirectional forward pass,同时预测所有 masked value positions
       得到 draft tokens {x̃_i}_{i ∈ E_j}
    3. Verify (AR head): 一次 causal forward pass,逐 token 检查:
       如果 argmax p_θ^AR(· | x_{<i}) == x̃_i,accept
       否则用 AR 的 token 替换,丢弃后面所有 draft tokens
       总是 accept 1 bonus token 在 reject 点
```

**为什么需要 bonus token?** 这是 speculative decoding 的标准设计: 如果 draft 和 verifier 一致到 position $k$,然后 $k+1$ 处不一致,接受 $k$ 之前的所有 tokens + AR 在 $k+1$ 的预测,即 $k+1$ 个 tokens 总共。lossless 保证就来自这个 "verify 直到 mismatch + 1" 机制 (见 Leviathan et al. 2023 的 Lemma 1)。

**为什么 scaffold 自动 accept 是合法的?** Scaffold tokens 是 schema-determined,概率 1 (在所有 valid 输出里都一样)。把它们"自动 accept"等价于"argmax 永远猜对",自然 lossless。

**Speedup 来源**:
1. Scaffold token 不进入 draft-verify 循环 → 节省 ~30% 的计算
2. Section-aligned block 给 MDM draft 完整的 semantic context,提高 acceptance rate
3. 每个 block 固定 2 forward passes (draft + verify),和 block size 无关

实验 (Table 4): SS latency 1919 ms (4.1× speedup over AR baseline), TPS 210.4, Tok/Step 4.90, ADE@5s 1.982, RFS 7.934 — 同时 accuracy 也最好

### 4.3 SGLang 集成 — 再 3× 加速

加 SGLang (https://arxiv.org/abs/2312.07104, https://github.com/sgl-project/sglang): latency 665 ms (11.8× speedup over AR baseline), TPS 608.5。这部分是 system-level 优化:
- CUDA graph 捕获固定 shape 的 forward
- Optimized kernel for block attention
- Better memory layout

这说明 algorithmic speedup 和 system speedup 是 multiplicative 的,6× (algorithmic) × ~3× (system) ≈ 18× — 但实际是 12×,因为 system 部分也有 diminishing return。

相关 system 工作: vLLM (https://arxiv.org/abs/2309.06180) 的 PagedAttention,FlashAttention (https://arxiv.org/abs/2205.14135),TensorRT-LLM。

---

## 5. Test-Time Inference Scaling — Shared-Prefix Multi-Trajectory Rollouts

### 5.1 思路

SS 默认是 deterministic (greedy AR verifier)。能不能用额外 inference compute 换 accuracy?论文给的方案:

1. 前 3 个 section (CO, Expl, FMB) 保持 greedy deterministic
2. 进入 trajectory section 时,在 AR verifier 上启用 softmax sampling (带 temperature)
3. 从同一个 shared prefix KV cache fork $N$ 次,各跑 trajectory section
4. 把 $N$ 条 trajectory 平均

### 5.2 形式化

设 $N$ 条 rollout trajectories 为 $\{\tau^{(i)}\}_{i=1}^{N}$,每条用 JMT (Jerk-Minimizing Trajectory) 拟合插值到 20 waypoints。最终输出:

$$\tau_{\mathrm{out}} = \frac{1}{N} \sum_{i=1}^{N} \tau^{(i)}$$

**Variance-of-mean argument**:

假设 $\tau^{(i)} = \tau^* + \epsilon_i$,其中 $\tau^*$ 是真值,$\epsilon_i$ 是独立 noise with variance $\sigma^2$。则:

$$\mathrm{Var}\left(\frac{1}{N} \sum \tau^{(i)}\right) = \frac{\sigma^2}{N}$$

所以 averaging $N$ 条把 residual variance 降 $1/N$,bias 不变 (如果 estimator 是 unbiased 的)。

直觉: 每条 rollout 在 5s 末端 waypoint 处最 disagree (距离远、不确定性大),平均让末端 waypoint 收敛到 GT。Figure 4(a) 显示得很清楚: 4 条 light blue rollout 在 5s 末端分散,平均 (dark blue) 紧贴 GT (black)。

### 5.3 为什么 stochasticity 只在 trajectory section?

前 3 section 的 posterior 很 sharply peaked (binary detections 和 categorical actions),sampling 它们:
- 不产生 useful diversity (本来就 high-confidence)
- 反而可能采样到错的 detection,污染 trajectory section

所以只在 trajectory section 上 sample,前 3 section 共享 prefix。

### 5.4 Shared prefix 的 efficiency trick

因为前 3 section 的 KV cache 在所有 $N$ 条 rollout 上一模一样,只需要 decode 一次,然后 fork $N$ 次。每条 rollout 只在 trajectory section 上做 SS (≈90 tokens)。

成本估算 (粗略):
- Full SS pass: 1919 ms (404 tokens)
- Trajectory-only SS: ~90/404 × 1919 ≈ 427 ms
- $N=4$: prefix (~1500 ms) + 4 × 427 = ~3200 ms ... 但论文报告 114.7 TPS / 280 tokens ≈ 2440 ms,说明 fork + reuse 实现更高效

实验 Table 2: $N=4$ rollout 让 ADE@5s 从 2.907 降到 2.821 (再降 3%),RFS 几乎不变,latency 翻 1.7× 左右。

### 5.5 与 test-time compute scaling literature 的关系

- Best-of-N (Cobbe et al. 2021, https://arxiv.org/abs/2110.14168): 用 verifier 选 $N$ 个 sample 里最好的
- Process reward (Lightman et al. 2023, https://arxiv.org/abs/2305.20050): process supervision
- Snell et al. 2024 (https://arxiv.org/abs/2408.03314): test-time compute 的最优分配
- Diffusion planners (Diffusion-ES, https://arxiv.org/abs/2308.13959; DiffusionDrive, https://arxiv.org/abs/2411.15239): 在 driving 上用 diffusion 做 trajectory ensemble

Fast-dDrive 的特色: shared prefix 让 $N$ 条 rollout 的 marginal cost 很小 (只 trajectory section),而传统 Best-of-N 是 $N$ 次完整 forward。

我的一个延展思考: 这其实是 mixture-of-experts / mixture-of-trajectory 的 special case — 你可以做更复杂的 ensemble,比如 trajectory diffusion 的 multi-mode sampling,然后 pick the mode closest to median (more robust to outlier)。

---

## 6. Experiments 详解

### 6.1 Datasets

**WOD-E2E** (Xu et al. 2025, https://arxiv.org/abs/2510.26125): 4,021 long-tail driving segments,每个 20s,split 2037/479/1505。Test 时只给前 12s,需要预测。

**nuScenes** (Caesar et al. 2020, https://www.nuscenes.org/): 1000 scenes,700/150/150 split,2 Hz keyframes。L2 error at 1/2/3s horizons。

### 6.2 Input

- RGB camera frames (nuScenes: 3 frames @ $t \in \{-1.0, -0.5, 0\}$s from CAM_FRONT;WOD-E2E: 3 front views @ $t=0$)
- ego state vector (position, velocity, acceleration, yaw, yaw rate)
- navigation command (text)
- 不用 LiDAR、radar、HD map

图像 resize 到 longer side 512px,然后 Qwen2.5-VL vision encoder patchify。

### 6.3 Metrics

- **ADE@Ts** (Average Displacement Error at T seconds): 预测轨迹和 GT 在未来 $T$ 秒的平均 L2 距离,越小越好
- **RFS** (Rater Feedback Score, https://arxiv.org/abs/2510.26125): Waymo 提出的人类评分,trust-region score,越大越好。这比 L2 更好地匹配人类偏好。
- **TPS** (tokens per second): throughput
- **Tok/Step**: 每次 forward pass commit 的 token 数 (AR=1,block diffusion >1)

### 6.4 WOD-E2E main results (Table 2)

| Method | Paradigm | RFS↑ | ADE@5s↓ | ADE@3s↓ | TPS↑ | Tok/Step |
|---|---|---|---|---|---|---|
| OpenEMMA* | AR | 5.158 | 12.476 | 6.684 | — | 1 |
| LightEMMA* | AR | 6.517 | 3.740 | 1.705 | — | 1 |
| NaiveEMMA | AR | 7.528 | 3.018 | 1.320 | — | 1 |
| AutoVLA | AR | 7.557 | 2.958 | 1.351 | 51.2 | 1 |
| Poutine-Base | AR | 7.909 | 2.940 | 1.270 | 51.2 | 1 |
| dVLM-AD | Diffusion | 7.633 | 3.022 | 1.285 | 35.2 | 2.82 |
| **Fast-dDrive (SS)** | Block Diff | **7.823** | **2.907** | **1.254** | **210.4** | **4.90** |
| + Scaling $N=4$ | Block Diff | 7.827 | **2.821** | **1.240** | 114.7 | 2.76 |

观察:
- Fast-dDrive SS 比 AR SOTA (Poutine-Base) RFS 略低 (7.823 vs 7.909),但 ADE 都更小,throughput 6× 高
- 加 inference scaling ($N=4$) 后 ADE 进一步降低,但 RFS 没显著变化 — 说明 variance reduction 主要改善末端 waypoint accuracy,而 RFS 是 trust-region score 可能更看整体 shape
- dVLM-AD 是 diffusion baseline,throughput 只有 35.2 (差 AR 还多),Fast-dDrive 6× 高

### 6.5 nuScenes results (Table 3)

Fast-dDrive avg L2 = 0.32m,vs dVLM-AD 0.41m,vs DriveVLM 0.40m,vs classical VAD 0.37m。这是 VLA 里 SOTA,还超越了无 reasoning 的 classical policy。

### 6.6 Efficiency breakdown (Table 4)

AR baseline (Qwen2.5-VL-3B): latency 7855 ms, TPS 51.6, Tok/Step 1
dVLM-AD: 9575 ms (0.8×), TPS 35.2 — full-sequence diffusion 比 AR 还慢!
Fast-dDrive Self-Spec: 3714 ms (2.1×), TPS 109.0, Tok/Step 2.41
Fast-dDrive Section Diffusion: 3006 ms (2.6×), TPS 134.4, Tok/Step 3.28
Fast-dDrive Scaffold Spec: 1919 ms (4.1×), TPS 210.4, Tok/Step 4.90 — scaffold 自动 accept 贡献最大
Fast-dDrive + SGLang: 665 ms (11.8×), TPS 608.5 — system 加速

### 6.7 Ablation (Table 5)

IWL (Section-Importance-Weighted Loss):
- No IWL, No SNS: RFS 7.735
- No IWL, SNS: RFS 7.855 (+0.120)
- IWL, No SNS: RFS 7.807 (+0.072) — wait,这个比 SNS-only 还低,有点奇怪
- IWL + SNS: RFS 7.916 (+0.181)

IWL 是 primary contributor,SNS 提供互补 gain。两个一起最强。

注: 这里 IWL-only 比 SNS-only RFS 低的数据有点反直觉,可能解释是 IWL 让 model 在 trajectory 上 over-fit hard noise regime,在 val set (easy) 上反而表现略降;SNS 自己就调节 noise 分布,所以 SNS-only 表现更好。两者组合中和了。这是个值得深究的点。

---

## 7. 延伸思考 & 联想

### 7.1 Block diffusion 的更广应用

这个 section-aligned block 思路可以推广到任何有 hierarchical structure 的 LLM 输出:
- Tool-use API calls: function name (scaffold) → arguments (value),block 切到 argument boundary
- Code generation: function signature → body,block 切到 statement boundary
- Math reasoning: claim → derivation → answer,block 切到 derivation step

每个 case 都是: 把 inherent causal structure encode 进 attention pattern,既保留 diffusion 的并行性,又维护逻辑因果。

### 7.2 与 continuous diffusion 的对比

论文用的是 discrete masked diffusion,但 continuous diffusion (e.g., Diffusion-LM, https://arxiv.org/abs/2205.11417; VDM, https://arxiv.org/abs/2107.00630) 在 trajectory 上其实更自然 — 因为 trajectory coordinates 是连续值。

想象一个 hybrid: discrete MDM 处理 perception/explanation (语言部分),continuous diffusion 直接生成 continuous trajectory coordinates。这样 trajectory 不需要 tokenize 成 string,避免 quantization error。

相关: Wayve 的 GAIA-1, NVIDIA 的 GameGAN,diffusion policy (https://arxiv.org/abs/2303.04137) 在 robotics 上的应用。

### 7.3 Causal confusion 在 diffusion 里的解

Causal confusion (driving 上的经典问题,见 https://arxiv.org/abs/2105.02596) — model 用 spurious signal (e.g., ego velocity) 替代真实 perception。Block diffusion + section-aligned causal ordering 强制 trajectory 必须通过 perception 才能产生,某种程度上 mitigate causal confusion,因为 trajectory tokens 不能直接 attend to ego-state-only context 而绕过 perception section。这是 paper 没明说但暗含的 property。

### 7.4 Self-speculative 的更广 trick

Self-speculative decoding 用同一个 model 的两个 head (MDM bidirectional + AR causal) 做 draft-verify。这个 idea 可以推广到:
- Mixture-of-depths (MoD): 用 shallow layer draft,deep layer verify (类似 LayerSkip, https://arxiv.org/abs/2404.16710)
- Mixture-of-experts: 用 cheap expert draft,expensive expert verify
- Sliding-window attention draft, full attention verify (类似 https://arxiv.org/abs/2310.00186)

### 7.5 RL post-training 在 block diffusion 上的可能

Poutine 用了 GRPO 做 RL post-training 提升 RFS。Block diffusion + SASD + Scaffold Spec 能不能加 GRPO?可以,但有几个 challenges:
1. GRPO 需要 sample 多条 trajectory 算 reward,block diffusion 的 sample 方式和 AR 不一样 (denoise 而非 sample next token)
2. Reward 一般在 trajectory 上,scaffold 上没 reward,这正好 match scaffold 不需要学的设定
3. Shared-prefix rollout 可以直接复用 — 已经是个 $N$-sample 机制

这条线很值得做。

### 7.6 Closed-loop evaluation 的挑战

论文 §C 提到只在 open-loop 上 evaluate。Closed-loop (CARLA, https://carla.org/,or Waymo Sim) 的挑战:
- Model 输出的 trajectory 会影响 ego state,下一帧 input 会变 — 这是 diffusion LM 的 KV-cache reuse 设计在 closed-loop 上要重新考虑的 (前一帧的 KV cache 可能 invalid)
- Long-horizon plan 的 re-plan 频率: 如果每 0.1s re-plan,trajectory 5s horizon 的后段其实永远没用上
- Reactive agents: 其他车辆会对 ego 反应,model 要 capture 这个 distribution shift

相关 closed-loop VLA 工作: LMDrive (https://arxiv.org/abs/2312.07488), CarLLaMA, NAVAR (https://arxiv.org/abs/2310.01825)。

### 7.7 Schema evolution 的连续学习

如果 schema 改了 (e.g., 增加 "construction_zone" detection),scaffold 要变。怎么 efficient update?
- Scaffold 本身可以从 schema definition 自动生成 (parser + tokenizer)
- 如果只是 add field,value tokens 在新 field 上要 fine-tune
- 如果是结构变化,可能要重新 SASD train

这指向一个更通用的 framework: scaffold 和 model 解耦,schema 是 declarative spec,model 学 value prediction。

### 7.8 关于 RFS vs ADE 的不一致

Table 2 显示 Fast-dDrive SS RFS 7.823 < Poutine 7.909,但 ADE 都更小。这暗示 RFS 不是单纯看 L2,而是看 trust region (是否符合 human-rated reference trajectory 的"trust range")。可能 Poutine 在某些 cases 上 trajectory shape 更"natural" (e.g., smooth lane change),而 Fast-dDrive 偶尔产生"numerically closer but shape-wise less natural" 的 trajectory。

加 inference scaling ($N=4$) 后 ADE 降但 RFS 没显著变,这其实暗示: averaging 主要平滑末端 waypoint noise,对 shape naturalness 没直接帮助。要做 RFS 的提升,可能需要 trajectory smoothness regularization 或者 reward model 直接对 RFS 优化。

### 7.9 数据 efficiency 的隐含 trick

WOD-E2E training: 30k CoT-annotated samples + 60k trajectory-only samples = 90k。但 trajectory-only 没解释 chain-of-thought,怎么和 SASD (4-section 输出) 兼容?可能是对 trajectory-only samples,只在 trajectory section 算 loss,其他 section 的 weight 设 0。这其实是个隐含的 curriculum: 先学 perception reasoning (CoT data),再学 trajectory-only (大量 trajectory data)。这种混合训练 trick 值得单独 paper 讨论。

### 7.10 Block size 的选择

Table 1 显示 block size 是 heterogenous 的 (explanation 有 6 个 block,trajectory 有 3 个),按 section 对齐。但 block diffusion 原始 paper (https://arxiv.org/abs/2503.09573) 用 fixed block size。Section-aligned 块大小是 ~32 tokens (192/6=32 for explanation, 70/3≈23 for trajectory)。如果 block size 变化对 acceptance rate 影响多大?这是个 ablation 没做的点。

直觉: block 越大,draft 并行度越高 (单次 forward 预测更多 token),但 AR verify 失败概率也越高 (一长串都对难)。block size 8-16 在 general LM 上 sweet spot (见 Fast-dLLM)。Section-aligned 等于 forced block size = section length / block count,可能不是最优。可以做 dynamic block size based on section uncertainty。

### 7.11 JMT interpolation

论文用 Jerk-Minimizing Trajectory (JMT) 把 5 个 waypoints (1s 间隔) 拟合到 20 waypoints (0.25s 间隔)。JMT 是 5 阶多项式拟合 position, jerk 平方最小。原始 paper: https://arxiv.org/abs/1807.07388。这个 trick 让 evaluation 在更细时间 grid 上,但 model 只预测粗 grid,可能 limit expressivity (无法预测 mid-second maneuvers)。如果直接 predict 20 waypoints,model 容量需求更大但精度可能更高。

### 7.12 与 VLA robotics 的 cross-pollination

Robotics VLA (OpenVLA, https://arxiv.org/abs/2406.09246; π0, https://arxiv.org/abs/2410.24164) 也输出 structured actions。Scaffold + section block diffusion 完全可以套用:
- "critical_objects" 换成 "objects_in_workspace"
- "explanation" 换成 "task_plan"
- "trajectory" 换成 "end_effector_waypoints"

Robotics 的 long-horizon plan 频率更高 (10-50 Hz vs driving 的 10 Hz),效率收益会更显著。

### 7.13 Reasoning 的可解释性 — diffusion 和 AR 的差别

AR 的 CoT 是单向的,每一步看前一步。Diffusion 的"reasoning"是迭代的 (denoise 多步),每步看全局。但 section-aligned block diffusion 的 reasoning 在 section 内 bidirectional,意味着 model 可以"同时考虑所有 reasons"再决定 explanation 最终形式 — 这比 AR 的线性推理更接近人类的"holistic thinking"。

这点上,block diffusion 的 explanation section 可能比 AR 的更 coherent,虽然论文没直接测这点 (没 explanation quality metric)。可以做 human eval。

### 7.14 Token-level vs semantic-level causal ordering

Section-aligned 是 semantic-level causal (perception → reasoning → action)。但 section 内是 bidirectional,允许 perception 内部互相参考。这其实放松了"完全感知 → 完全推理 → 完全动作"的硬顺序,允许 model 在感知内部微调 (e.g., 看到卡车后回头确认它是不是 emergency vehicle)。这是合理的 — 人类也是 iterative perception。

### 7.15 关于 hallucination 的对照

AR VLA 的 exposure bias 让 5s 后段 trajectory 偏离物理可行。Diffusion 的 holistic context 让 trajectory 全局一致,但 full-seq diffusion 的 logical leakage 让 perception 可能 hallucinate 来合理化错的 trajectory。Block diffusion 的 section-aligned causal 同时避免两者 — 这是 paper 的核心 thesis,我认为这是真正的 contribution。

### 7.16 Memory hierarchy 的实际意义

对 batch-size 1 on-vehicle deployment,model weights 在 SRAM/HBM。如果 model 大,HBM bandwidth 是 bottleneck。Scaffold pre-fill 让很大一部分 (30%) tokens 一次性 prefill,可以预 load 到 SRAM。Block-level KV cache reuse 让 KV size 控制在单个 block 量级。整个 memory hierarchy 更 friendly。

参考: FlashAttention (https://arxiv.org/abs/2205.14135), MosaicML 的 inference optimization。

### 7.17 Section weight 的敏感性

$w_{\mathrm{traj}} = 3.0$ 看起来 arbitrary。如果调到 5.0 或 10.0 会怎样?直觉是过大会让 perception underfit,反过来 degrade trajectory (因为 perception 错了)。可能有 sweet spot。论文没做 sensitivity analysis。一个更原则的方法: 用 reinforcement learning 自动学 section weights,类似 learned curriculum。

### 7.18 Bayesian 视角下的 shared-prefix rollout

Shared-prefix rollout 实际上是 approximate Bayesian model averaging — 每条 rollout 是 trajectory posterior 的一个 sample,平均是 posterior mean estimator (under squared loss 是 optimal)。如果用 median 而非 mean,可以更 robust to outlier (e.g., 一条 rollout hallucinate 突然急转)。可以做 robust aggregation 实验。

### 7.19 Diffusion step 数 vs block 数

论文没明确说每 block 的 diffusion steps 数。Fast-dVLM 原文是 8 步。如果 SS mode 下每 block 只 1 步 draft (因为 AR verify 已经 fix 了大部分),那 diffusion step 数可以更低。这是 SS 比 SD 快的另一原因 — SD 是 multi-step iterative denoise,SS 是 single-step draft + verify。

### 7.20 对比我之前的文章 intuition

Karpathy 你在 "State of GPT" 和多个 talk 里强调过 inference scaling,以及 AR 的 memory-bound issue。Fast-dDrive 实际上是你预言的 "non-AR decoding will become important for efficiency" 的一个具体 case — 在 driving 这种 latency-critical domain 上,AR 的 50 TPS 是致命 bottleneck,而 block diffusion + scaffold + SS 把它推到 600 TPS,跨越了 realtime deployment 阈值 (driving 一般要 ≥10 Hz,600 TPS 可以支持 60 Hz re-plan,绰绰有余)。

你之前讲过的 "deep learning is soldering plumbing together" 也适用 — Fast-dDrive 把 5 个 plumbing (block diffusion + scaffold + section alignment + SASD + SS + SGLang) solder 在一起,每个都不大,合起来 12× speedup。

---

## 8. Summary: 这篇 paper 真正的 contribution

把 paper 的 contribution 重新组织一下,我觉得有三层:

**Layer 1 (algorithmic)**: Section-aligned block diffusion + scaffold — 把 driving 的 inherent causal structure encode 到 attention pattern 里,解决 logical leakage,同时保留 diffusion 的全局一致性和 KV-cache 兼容性。

**Layer 2 (training)**: SASD — Section-weighted loss + section-adaptive Beta noise schedule,把 safety-critical tokens 的 learning capacity 集中起来,zero inference overhead。

**Layer 3 (inference)**: 
- Scaffold Spec (algorithmic): auto-accept scaffold tokens + section-aware draft-verify,4-6× speedup
- Shared-prefix rollout (test-time scaling): 从 deterministic SS 派生 cheap N-sample,2× 成本换 3% ADE gain
- SGLang 集成: 再 3× speedup

合计: 12× throughput speedup over AR baseline,WOD-E2E SOTA ADE,nuScenes SOTA L2。这把 high-capacity VLA 从"研究 demo"推到"on-vehicle deployable"。

---

## 9. 我对这篇 paper 的总评

**优点**:
- 整体设计 elegant,scaffold + section alignment 是 driving domain 的 natural fit
- Speedup 数字实在 (12× in pipeline, 4-6× algorithmic only)
- 两个 dataset SOTA,且 transfer 不需要 per-dataset tuning
- Ablation 清楚,虽然 IWL-only vs SNS-only 的数据有点反直觉

**值得深究**:
- RFS 没超 Poutine,可能 trajectory shape naturalness 还不够
- Closed-loop eval 缺失,open-loop ADE 不能完全反映 driving 能力
- Schema rigidity 限制扩展性
- Block size 是 section-aligned 的,不是搜出来的最优
- Sensitivity analysis on section weights 缺失
- Trajectory 是 1s grid + JMT 拟合,可能 limit mid-second 反应

**值得 follow-up 的方向**:
- Hybrid discrete + continuous diffusion (discrete for perception/explanation, continuous for trajectory)
- RL post-training (GRPO) 在 block diffusion 上的适配
- Closed-loop benchmark (CARLA, Waymo Sim)
- Schema-learnable variant
- Application 到 robotics VLA

---

## Web links reference

主要 paper:
- Fast-dDrive (本 paper)
- Fast-dVLM (Wu et al. 2026): https://arxiv.org/abs/2604.06832
- Fast-dLLM (Wu et al. 2025): https://arxiv.org/abs/2505.22618
- Block Diffusion (Arriola et al. 2025): https://arxiv.org/abs/2503.09573
- LLaDA (Nie et al. 2025): https://arxiv.org/abs/2502.09992
- dVLM-AD (Ma et al. 2025): https://arxiv.org/abs/2512.04459
- Dream (Ye et al. 2025): https://arxiv.org/abs/2508.15487
- MMaDA (Yang et al. 2025): https://arxiv.org/abs/2505.15809
- LLaDA-V (You et al. 2025): https://arxiv.org/abs/2505.16933
- DIMPLE (Yu et al. 2025): https://arxiv.org/abs/2505.16990

Driving VLA / E2E:
- DriveVLM (Tian et al. 2024): https://arxiv.org/abs/2402.12289
- DriveCoT (Wang et al. 2024): https://arxiv.org/abs/2403.16996
- AutoVLA (Zhou et al.): https://arxiv.org/abs/2509.20710
- Poutine (Rowe et al. 2025): https://arxiv.org/abs/2506.11234
- OpenEMMA (Xing et al. 2025): https://arxiv.org/abs/2506.15577
- LightEMMA (Qiao et al. 2025): https://arxiv.org/abs/2505.00284
- UniAD (Hu et al. 2023): https://arxiv.org/abs/2212.10156
- VAD (Jiang et al. 2023): https://arxiv.org/abs/2303.12077
- WOD-E2E (Xu et al. 2025): https://arxiv.org/abs/2510.26125
- nuScenes (Caesar et al. 2020): https://www.nuscenes.org
- DiffusionDrive (Liao et al. 2025): https://arxiv.org/abs/2411.15239
- Diffusion-ES (Yang et al. 2024): https://arxiv.org/abs/2308.13959
- Gu et al. 2026: https://arxiv.org/abs/2602.02864

Foundational diffusion / decoding:
- D3PM (Austin et al. 2021): https://arxiv.org/abs/2107.03006
- Diffusion-LM (Li et al. 2022): https://arxiv.org/abs/2205.11417
- Sahoo et al. 2024: https://arxiv.org/abs/2406.18577
- Lou et al. 2024: https://arxiv.org/abs/2406.11473
- Shi et al. 2024: https://arxiv.org/abs/2404.14457
- Speculative decoding (Leviathan et al. 2023): https://arxiv.org/abs/2211.17192
- Speculative sampling (Chen et al. 2023): https://arxiv.org/abs/2302.01318
- Medusa (Cai et al. 2024): https://arxiv.org/abs/2401.10774
- EAGLE (Li et al. 2024): https://arxiv.org/abs/2401.15077
- Self-speculative (Zhang et al. 2024): https://arxiv.org/abs/2401.00835

Test-time scaling:
- Cobbe et al. 2021: https://arxiv.org/abs/2110.14168
- Lightman et al. 2023: https://arxiv.org/abs/2305.20050
- Snell et al. 2024: https://arxiv.org/abs/2408.03314

System:
- SGLang (Zheng et al. 2024): https://arxiv.org/abs/2312.07104
- vLLM (Kwon et al. 2023): https://arxiv.org/abs/2309.06180
- FlashAttention (Dao et al. 2022): https://arxiv.org/abs/2205.14135
- Qwen2.5-VL (Bai et al. 2025): https://arxiv.org/abs/2502.13923

Robotics VLA:
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- Diffusion Policy: https://arxiv.org/abs/2303.04137

JMT:
- Werling et al. 2010 (经典): https://arxiv.org/abs/1807.07388 (相关 implementation reference)

Causal confusion:
- Causal confusion in IL: https://arxiv.org/abs/2105.02596

Hallucination survey:
- Huang et al. 2025: https://arxiv.org/abs/2311.05232

Structured output:
- Outlines: https://github.com/outlines-dev/outlines
- XGrammar: https://arxiv.org/abs/2411.15159

希望这个深度解析帮你 build intuition 关于 block diffusion VLA 在 driving 上的具体 trade-off 和 design choices。如果有哪个 section 你想再深挖 (e.g., JMT 拟合细节、SGLang kernel 实现、Beta noise schedule 的最优选择推导),告诉我我可以继续展开。
