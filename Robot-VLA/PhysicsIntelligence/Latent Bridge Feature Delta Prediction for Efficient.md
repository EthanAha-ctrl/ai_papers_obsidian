---
source_pdf: Latent Bridge Feature Delta Prediction for Efficient.pdf
paper_sha256: d0d8dccaffd5cd5c9cb9ead445392451463a87b93b5e69b14eb56c6872912c87
processed_at: '2026-08-05T12:06:38-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Latent Bridge 用人话讲

## 一句话版本

VLM 每一步产生的 feature 跟上一步几乎一模一样，那干嘛每步都跑 VLM？训练一个小模型预测"变化量"，中间几步用小模型顶上，VLM 偶尔跑一次校正就行。

---

## 问题在哪儿

Dual-system VLA 就是把 policy 拆成两段：

1. **大 VLM backbone**：吃图片+指令，吐 feature，慢得要命（46-63ms）
2. **小 action head**：吃 feature，吐 action，快（27-30ms）

控制机器人要 10-50Hz，也就是 20-100ms 内必须给一个 action。你光跑 VLM 就 60ms 了，action head 还要 30ms，加起来 90ms，刚刚卡在边缘。真实机器人要稳，这个频率根本不够。

更扎心的是：你跑完 VLM 拿到的 feature，跟上一个 timestep 的 feature，cosine similarity 在 stable layer 上 **>0.999**。意思是你花 60ms 算出来的东西，99% 跟上一步一样。

这就是浪费。Paper 的核心 motivation 就这一句。

---

## Latent Bridge 的 idea

既然 VLM output 变化很小，那中间几步就不跑 VLM，用一个轻量小模型预测"这一步 feature 相比上一步变了多少"。

公式就一行：

$$\hat{z}_{t+1} = \hat{z}_t + \mathcal{B}(\hat{z}_t, s_t, q_t, a_{t-1})$$

变量意思：
- $\hat{z}_{t+1}$: 这一步预测的 feature
- $\hat{z}_t$: 上一步的 feature（VLM step 上是 fresh VLM 输出，bridge step 上是小模型自己上一步的预测）
- $\mathcal{B}$: 小 bridge model
- $s_t$: 上次 VLM call 缓存的稳定中间层 feature，当 visual context
- $q_t$: 机器人本体感觉 state
- $a_{t-1}$: 上一步 action

所谓 **VLM call period $f$**：每 $f$ 步里，第 1 步跑 VLM，剩下 $f-1$ 步跑 bridge。$f=4$ 就是 1 次 VLM + 3 次 bridge，省 75% VLM call。

Bridge step cost 2-6ms vs VLM 60ms，所以 net speedup 能到 1.65-1.73×。

---

## 为什么预测 delta 而不是预测 absolute

这是 paper 最 elegant 的设计点。

Bridge 的 output layer **zero-init**。意思是训练之前，bridge 输出恒为 0，那么：

$$\hat{z}_{t+1} = \hat{z}_t + 0 = \hat{z}_t$$

未训练的 bridge = feature caching（直接复用上一步 feature）。这就是 trivial baseline，性能虽然差，但至少不会爆炸。

训练过程中 bridge 从 zero delta 慢慢学起来，相当于从 caching baseline 平滑过渡到 learned predictor。这种 "start from identity" 的 trick 在 ResNet 里见过，在 control 里的 incremental form 也见过，稳定性极好。

如果直接预测 absolute feature $\hat{z}_{t+1} = \mathcal{B}(\ldots)$，未训练 bridge 输出随机 noise，action head 吃到 garbage 直接崩。

---

## 两种 VLA 怎么接 bridge

### GR00T: feature-space bridge

GR00T backbone 和 action head 之间就一个 feature vector 接口，简单。Bridge 直接预测这个 feature 的 delta。

一个细节：bridge 只处理 **image token**，不处理 text token。为什么？因为 instruction 在一个 episode 内固定不变，text token 的 hidden state 在 consecutive step 间 cosine > 0.9999。预测这种 near-zero delta 会稀释 image token 上的 gradient。所以 text token 直接从 last VLM cache copy，bridge 只看 image token。

### π0.5: KV-cache bridge

π0.5 是 interleaved 设计，Gemma-2B backbone 和 Gemma-300M action expert 共享 18 层 transformer，每层 action token cross-attend 到 backbone 的 per-layer KV cache。

Bridge 改成预测 **18 层的 pre-RoPE K 和 V delta**：

$$\{\Delta K_l, \Delta V_l\}_{l=1}^{L} = \mathcal{B}_{\text{KV}}(\Delta e_t, e_t, \widehat{\text{KV}}_{t-1}, q_t, a_{t-1})$$

这里 $\Delta e_t = e_t - e_{t-1}$ 是 SigLIP vision embedding 的 delta（SigLIP 5ms，比 Gemma prefix 46ms 便宜 10×），告诉 bridge "视觉上变了什么"。然后 18 个 per-layer head 各自输出 K delta + V delta。

RoPE 有个坑：RoPE 是位置相关的 rotation，bridge 必须在 pre-RoPE 空间预测 delta。所以要从模型 post-RoPE cache 反推 pre-RoPE（用 inverse rotation），预测完 delta 再 re-apply RoPE 插回去。Appendix A 有完整推导。

---

## 最关键的训练 trick: DAgger

### 问题

Bridge 在 deployment 时是 **autoregressive chaining**：$t'=1$ 吃 fresh VLM feature，$t'=2$ 吃自己 $t'=1$ 的预测，$t'=3$ 吃自己 $t'=2$ 的预测……

但你训练时如果只在 sync data 上训（每步都有 fresh VLM feature 作为 input），bridge 学的是"input 总是 clean VLM feature"这个分布。Deployment 时 input 是自己上一步的 noisy prediction，分布就 shift 了，error 会 compound。

这就是 imitation learning 里经典问题：learner 见过的 input 分布跟它自己 rollout 产生的分布不一样。

### 解法

DAgger 的核心 idea：让 bridge 在 simulator 里自己 rollout，同时让 VLM oracle 在旁边 parallel 跑提供 ground-truth feature 当 label。Robot 按 bridge 的 action 走，但 supervision 来自 oracle。

具体：
1. 先用 sync data 训一个 R0 bridge（只见过 clean VLM feature 输入）
2. 把 R0 部署到 simulator，$f=3$，让 bridge 自己 chain predict
3. 同时 VLM 每步跑一遍，给 ground-truth feature target
4. 把 (bridge 自己的预测 input, oracle 的 ground-truth output) 收集成 DAgger pair
5. 把 sync data + DAgger data 混一起重训，从 R0 weights resume，LR 降 10×，得到 R1

效果 Table 4：

| Stage | π0.5 Spatial | π0.5 LIBERO-10 | GR00T LIBERO-10 |
|---|---|---|---|
| Feature caching | 46.67 | 52.67 | 42.33 |
| R0 (sync only) | 95.17 | 89.17 | 54.33 |
| R1 (+DAgger) | 99.00 | 93.67 | 67.17 |

R0 已经能恢复 93-96% sync performance 在短 horizon task 上，DAgger 在长 horizon 上 +13pp。

### 为什么 DAgger 必须在 simulator 里

因为要 VLM oracle parallel 跑给 label，real robot 上做不到。这是 paper 的一个 limitation。

---

## 为什么 Feature Caching 必败

最 naive 的 baseline 是不做 delta prediction，直接 $\Delta_t = 0$，复用 stale feature。Table 1 显示这个 collapse 得很惨：GR00T Object 从 99.83% 掉到 3.00%。

Figure 5 的 case study 讲得很清楚：在一个 LIBERO-Spatial episode 里，caching 的 KV cosine to ground truth 呈 **sawtooth pattern**——VLM call 时回到 ~1.0，然后随着 robot 移动线性下降到 ~0.91，下次 VLM call 又回 ~1.0，循环。Bridge 一直维持 ~0.99。

Sawtooth 的成因：cache 只编码 stale scene，robot 每动一步，stale KV 和 ground truth 的 gap 累积。Bridge 通过 cheap signals（SigLIP delta, state, previous action）主动预测 update，gap 几乎不扩大。

Caching 在 LIBERO-Object 掉到 3% 是因为 object grasping 对 feature 精度极高，KV cosine 0.91 已经足够让 action head 输出 garbage action。

---

## Long-horizon 怎么救：Stage 2 + 3

LIBERO-10 是长 horizon，bridge autoregressive chaining 的 error 在长 episode 里 compound 得厉害。R1 + DAgger 只能推到 67.17%（sync 93.00%）。Paper 加两个 optional stage：

### Stage 2: LoRA action head adaptation

Bridge feature 比 clean VLM feature noisy，action head 没见过这种 noisy input。对 action head 的 DiT 加 LoRA，训练用 50/50 mix of bridge feature + sync feature + Gaussian noise augmentation，让 action head 学会容忍 bridge noise。

67.17% → 73.17%。

### Stage 3: Phase-aware VLM scheduling

不要用固定 $f$，根据 previous action 的 translation magnitude 动态调：
- 高 motion（导航阶段）→ $f=2$（bridge 在快速运动时不可靠）
- 中 motion → $f=3$
- 低 motion（精细 manipulation）→ $f=4$（bridge 可以多 chain 几步）

73.17% → 89.17%。

短 horizon task 这两个 stage 在 seed noise 内，Stage 1 就够了。但 LIBERO-10 必须 Stage 1+2+3 全上才能接近 sync。

---

## 为什么 token pruning 在 dual-system 上不行

FastV 和 VLA-Cache 在 single-system VLA（OpenVLA 那种）上报告 1.3-1.5× speedup。但 Table 1 显示在 dual-system 上只能 1.08× net。

原因是 **Amdahl's law**。

Single-system VLA 整个 inference 就是 LLM，pruning LLM 直接加速整个 pipeline。Dual-system VLA 里 backbone 只占总 latency 60-70%（GR00T 63/90=70%，π0.5 46/76=60%），剩下是 action head。Pruning backbone 1.5× local speedup，net speedup 上限：

$$\frac{1}{(1-p) + p/s_{\text{local}}} = \frac{1}{0.30 + 0.70/1.5} \approx 1.30\times$$

实际 VLA-Cache 在 backbone 上只拿到 ~1.18× local，net 就 1.08×。

Latent Bridge 直接 skip backbone call，bridge step cost 2-6ms vs VLM 46-63ms，local speedup 10-30×，完全跳出 Amdahl ceiling。

这是 paper 最 important 的 architectural insight：**skip a call is fundamentally cheaper than making an existing call faster**。

---

## 实验结果总结

四个 LIBERO suite + RoboCasa 24 kitchen task + ALOHA bimanual transfer-cube：

| 指标 | GR00T | π0.5 |
|---|---|---|
| Sync avg SR | 96.58% | 96.96% |
| Bridge avg SR | 94.54% | 96.92% |
| Retention | 98.1% | 99.97% |
| Net speedup | 1.73× | 1.65× |
| VLM savings | ~75% | ~75% |

Cross-benchmark：
- RoboCasa 24 task zero-shot: 95.38% retention
- ALOHA 14-DoF bimanual: 97.73% retention
- Feature caching 在 RoboCasa 只 75.17%，ALOHA 52.27%

同一套 pipeline（sync collection + R0 + DAgger R1）跨 LIBERO/RoboCasa/ALOHA 直接迁移，不改任何东西。这是 task-agnostic 的关键证据。

---

## Intuition 总结

1. **VLM feature temporal redundancy 是金矿**：99% 的 consecutive step feature 是 redundant 的，这是 paper 的 empirical foundation

2. **Delta prediction + zero init = 稳定起点**：未训练 bridge = caching baseline，训练过程平滑过渡

3. **Autoregressive chaining 的 distribution shift 用 DAgger 修**：这是 paper 最核心的 training insight，长 horizon 上 +13pp

4. **Architecture-agnostic**：feature-space 和 KV-cache 两种接口都 work，说明 delta prediction 思想可以泛化

5. **Amdahl escape**：token pruning 在 dual-system 上被 Amdahl bound 1.3×，bridge 通过 skip 突破到 1.7×

6. **小 bridge 够用**：19M 参数和 148M 参数效果一样，delta prediction 是 low complexity 任务

7. **Phase-aware scheduling 对 long horizon 必要**：static $f$ 在 LIBERO-10 不够，动态分配 VLM compute

---

## 我觉得这篇 paper 最 elegant 的地方

它没有改 VLM，没有改 action head，没有改训练算法，纯加了一个外挂小模型 + 一个 DAgger pipeline，就能在几乎不掉性能的前提下省 75% 的 VLM compute。

而且这个 idea 的适用范围远超 VLA：任何 "大 backbone + 小 head" 的架构，只要 backbone output 在 time/space 上 redundant，都能套这个 pattern。LLM agent 的 tool use、video generation 的 frame 间 latent、scientific computing 的 solver iteration，都是潜在 target。

核心哲学就一句：**预测变化量比重新计算更便宜**。

---

# Latent Bridge: Feature Delta Prediction for Efficient Dual-System VLA Inference

## 1. Paper 核心定位

这篇 paper 来自 Duke University + Qualcomm AI Research + University of Florida，攻击的是 dual-system VLA 的 inference bottleneck。核心 idea 很 elegant：既然 VLM backbone 每 step 产生的 feature 在时间上有强烈 redundancy，就训练一个 lightweight bridge 去预测 feature delta，让昂贵的 VLM 只在每 f 步调用一次，中间 f-1 步由 bridge 串联预测（autoregressive chaining），整体把 VLM call 减少 50-75%，wall-clock speedup 达到 1.65-1.73×，success rate 保留 95-100%。

Reference links:
- arXiv 搜索 "Latent Bridge Feature Delta Prediction" 
- GR00T N1: https://arxiv.org/abs/2503.14734
- π0.5: https://arxiv.org/abs/2504.16054
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- RoboCasa: https://arxiv.org/abs/2406.02523
- ALOHA / Mobile ALOHA: https://arxiv.org/abs/2304.13705

---

## 2. Background: Dual-System VLA 的结构性 bottleneck

### 2.1 Single-system vs Dual-system VLA

**Single-system VLA**（如 RT-2, OpenVLA）直接把 action tokenize 然后让 LLM autoregressive 生成 action token，整个 inference 路径就是一个 LLM forward pass，速度被 sequential decoding 拖累。

**Dual-system VLA** 把 perception 与 action decoding 分开：
- **Stage 1 (VLM backbone)** $\mathcal{V}$: vision encoder + language model 处理 observation $o_t$，产生 feature $z_t = \mathcal{V}(o_t) \in \mathbb{R}^{N \times D}$
  - $N$: sequence length（image tokens + text tokens）
  - $D$: feature dimension
- **Stage 2 (Action head)** $\mathcal{A}$: lightweight policy network 把 $z_t$ 解码成 action $a_t = \mathcal{A}(z_t, q_t)$
  - $q_t$: proprioceptive state（机器人本体感觉）

代表工作：
- GR00T (NVIDIA): Eagle backbone (SigLIP2 + Qwen3) + DiT action head
- π0.5 (Physical Intelligence): PaliGemma (SigLIP + Gemma-2B) + Gemma-300M action expert (interleaved)
- Octo: ViT + diffusion head
- CogACT: continuous action decoder
- HPT: heterogeneous pre-trained transformers

### 2.2 Latency breakdown 与 Amdahl ceiling

Paper Table 11 给出在 A100 80GB + bf16 + torch.compile 下的实测延迟：

| Component | GR00T-N1.6-3B | π0.5 |
|---|---|---|
| VLM backbone (full) | 63 ms | 46 ms (Gemma-2B prefix) |
| Vision encoder (bridge step) | - | 5 ms (SigLIP) |
| Action head | 27 ms (DiT 32L) | 30 ms (1-step denoise) |
| Bridge (bf16 + compile) | 2 ms | 1 ms |
| Sync total (backbone + head) | 90 ms | 76 ms |
| Bridge total (bridge + head) | 29 ms | 36 ms |
| Avg per-step @ f=4 | 44 ms | 46 ms |
| Net speedup | 1.73× | 1.65× |

Backbone 占总 latency 的 60-70%。Appendix H 给出关键 Amdahl 分析：
- Backbone fraction $p = 70\%$ (GR00T) 或 $60\%$ (π0.5)
- 假设 token pruning 给 backbone 1.5× local speedup，net speedup 上限为：

$$\text{speedup} = \frac{1}{(1-p) + p/s_{\text{local}}}$$

带入 GR00T：$\frac{1}{0.30 + 0.70/1.5} \approx 1.30\times$
带入 π0.5：$\frac{1}{0.40 + 0.60/1.5} \approx 1.23\times$

这就是为什么 FastV 和 VLA-Cache 在 single-system 上报告 1.3-1.5×，但移植到 dual-system 上只能拿到 1.08× net。**Latent Bridge 跳出 Amdahl ceiling 的方式是直接 skip backbone call**，bridge step cost 2-6 ms vs VLM 46-63 ms，local speedup 达 10-30×。

Reference:
- FastV: https://arxiv.org/abs/2403.06764
- VLA-Cache: https://arxiv.org/abs/2502.02175
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818

---

## 3. Core Method: Latent Bridge

### 3.1 Problem formulation

给定 dual-system VLA policy $\pi(a_t | o_t) = \mathcal{A}(\mathcal{V}(o_t))$，引入 VLM call period $f$：backbone $\mathcal{V}$ 每 $f$ 步执行一次（在 $t = 0, f, 2f, \ldots$），bridge $\mathcal{B}$ 填补中间 $f-1$ 步。VLM call rate 为 $1/f$。例如 $f=3$ 意味着 1 次 VLM + 2 次 bridge，节省 $\frac{f-1}{f} = 67\%$。

在 offset $t' \in \{1, \ldots, f-1\}$ 上：

$$\hat{z}_{t+t'} = \hat{z}_{t+t'-1} + \mathcal{B}(\hat{z}_{t+t'-1}, s_t, q_{t+t'}, a_{t+t'-1})$$
$$a_{t+t'} = \mathcal{A}(\hat{z}_{t+t'}, q_{t+t'})$$

变量解释：
- $\hat{z}_{t+t'}$: bridge 在第 $t'$ 个 bridge step 预测的 feature
- $\hat{z}_{t+t'-1}$: 上一步的 representation。$t'=1$ 时为 fresh VLM feature $z_t$（boundary convention $\hat{z}_t := z_t$），$t' \geq 2$ 时为 bridge 自己上一步的预测 → autoregressive chaining
- $s_t \in \mathbb{R}^{N \times D}$: stable intermediate-layer VLM feature，从上次 VLM call cache，提供 visual context 而不必重跑 backbone
- $q_{t+t'}$: 当前 step 的 proprioceptive state
- $a_{t+t'-1}$: 上一步 action

**Trivial baseline: Feature caching** $\mathcal{B} = \text{Id}$，等价于 $\Delta_t = 0$，复用 stale VLM feature。这个 baseline 在 Table 1 上 collapse 到 3-56% SR，说明 learned delta prediction 是必要的。

### 3.2 为什么 delta prediction 而非 absolute prediction

观察 VLM 内部 hidden representations 的层间 dynamics：
- **Stable layers**（early/middle，consecutive step cosine > 0.999）：基本不变，携带 scene layout、object identity
- **Dynamic layers**（final，cosine ~0.95）：剧烈变化，是 action head 真正需要的

预测 delta $\Delta_t$ 让 bridge 在 zero-init 时自然退化到 caching baseline，training stability 极好。这与 control theory 中的 incremental control / residual policy 思想同源。

Reference:
- ResNet residual learning: https://arxiv.org/abs/1512.03385
- Residual policy learning: https://arxiv.org/abs/1812.06298

### 3.3 两种 instantiation

#### (A) Feature-space bridge (GR00T)

GR00T 在 backbone 与 action head 之间只有一个 feature vector 接口。Bridge 直接预测这个 feature 的 delta：

$$\hat{z}_{t+1} = \hat{z}_t + \Delta_t, \quad \Delta_t = \mathcal{B}(\hat{z}_t, s_t, q_t, a_{t-1})$$

- $\hat{z}_t \in \mathbb{R}^{N_{\text{img}} \times D}$: image token feature
- $N_{\text{img}} \ll N$ 因为 text token 被剔除（见下）

Bridge 架构是 DiT block with AdaLN：
1. **Self-attention** over input features $\hat{z}_t$ with learned positional embeddings
2. **Cross-attention** to stable context $s_t$，提供视觉 grounding 而无需 vision encoder
3. **AdaLN conditioning** on $(q_t, a_{t-1})$，每个 block 注入 state 与 action 信息
4. **Zero-initialized output projection**，未训练时输出 $\Delta_t = 0$ → 自动从 caching 起步

**Image-only processing 的关键 trick**：text-token hidden state 在 consecutive step 间 cosine > 0.9999（因为 instruction 在一个 episode 内固定），预测 near-zero text delta 会稀释 image-token delta 上的 gradient。Bridge 只处理 image tokens（$N_{\text{img}} \ll N$），text tokens 从 last VLM cache 直接 copy。这同时让 bridge 的 sequence length 与 instruction length 解耦。

#### (B) KV-cache bridge (π0.5)

π0.5 是 interleaved design：Gemma-2B backbone 与 Gemma-300M action expert 共享 transformer layers，共 L=18 层。每层中 action (suffix) tokens cross-attend 到 backbone (prefix) 的 per-layer KV cache。

Bridge 改为预测所有 L 层的 pre-RoPE key 和 value delta：

$$\{\Delta K_l, \Delta V_l\}_{l=1}^{L} = \mathcal{B}_{\text{KV}}(\Delta e_t, e_t, \widehat{\text{KV}}_{t-1}, q_t, a_{t-1})$$

变量解释：
- $e_t = \text{SigLIP}(o_t)$: 当前 step 的 vision embedding（SigLIP ~5ms，比 full Gemma prefix ~46ms 便宜 10×）
- $\Delta e_t = e_t - e_{t-1}$: 视觉变化的 delta signal
- $\widehat{\text{KV}}_{t-1}$: 上一步的 per-layer KV cache（VLM step 上是 fresh，bridge step 上是自回归链式预测）
- $\mathcal{B}_{\text{KV}}$: shared DiT backbone + 18 个 lightweight per-layer output heads

更新规则：
$$\hat{K}_{l,t} = K_{l,t-1} + \Delta K_l, \quad \hat{V}_{l,t} = V_{l,t-1} + \Delta V_l$$

**RoPE handling**（Appendix A 的重要细节）：RoPE 是位置相关的确定 rotation，bridge 必须在 pre-RoPE 空间预测 delta。从模型 post-RoPE cache 反推 pre-RoPE：

$$k_{\text{pre}} = k_{\text{post}} \cos\theta - \text{rotate\_half}(k_{\text{post}}) \sin\theta$$

预测 delta 后再 re-apply RoPE 才能正确 insert 回 cache。这个细节极容易踩坑。

**Per-layer output head**: 每个 head 是 LayerNorm + Linear，把 shared DiT hidden state ($d=768$) 映射到 512-dim（256 K delta + 256 V delta 拼接）。所有 head zero-init。

Reference:
- DiT (Diffusion Transformer): https://arxiv.org/abs/2212.09748
- AdaLN: https://arxiv.org/abs/2212.09748
- RoPE: https://arxiv.org/abs/2104.09864
- SigLIP: https://arxiv.org/abs/2303.02588
- PaliGemma: https://arxiv.org/abs/2407.07726

---

## 4. Training Pipeline: 三阶段 task-adaptive

### 4.1 Stage 1: Sync data collection + R0 supervised training

**Sync collection**: 部署 pretrained VLA 在 sync mode（每步 VLM），在 simulator 中 rollout，记录 tuples $(z_t, z_{t+1}, s_t, q_t, a_{t-1})$ along closed-loop trajectory。GR00T 每 LIBERO suite 收 300 episodes（~4,500-6,600 samples after filtering），π0.5 同上，ALOHA 50 episodes。

**R0 training**: 在 sync pair 上 offline 训练 bridge，loss 为 MSE + cosine：

$$\mathcal{L} = \|\hat{z}_{t+1} - z_{t+1}\|^2 + \alpha \left(1 - \frac{\hat{z}_{t+1} \cdot z_{t+1}}{\|\hat{z}_{t+1}\| \|z_{t+1}\|}\right)$$

变量解释：
- $\hat{z}_{t+1}$: bridge 预测的 next-step feature
- $z_{t+1}$: ground-truth VLM feature
- $\alpha = 1.0$: cosine loss 权重
- 操作仅在 image tokens 上，text token delta 被 mask 掉

R0 hyperparams:
- LR $3 \times 10^{-4}$
- GR00T: 200 epochs, batch size 64
- π0.5: 50 epochs, batch size 4
- AdamW, weight decay $10^{-4}$, cosine LR schedule, gradient clip norm 1.0

### 4.2 Stage 2: DAgger refinement (R1)

**核心问题**: Sync-only 训练时 bridge 总看到 clean VLM feature 作为 input，但 deployment 时 bridge 必须吃自己的 prediction（autoregressive chaining），存在 distribution shift。这跟 imitation learning 中 DAgger 要解决的根本问题一致。

**DAgger rollout**: R0 bridge 部署在 simulator 中（period $f$），同时一个 VLM oracle 在 parallel 跑，提供每步 ground-truth feature target。Robot 仍按 bridge policy 的 action 执行，但 label 来自 oracle。

生成 DAgger pair $(\hat{z}_{t+t'-1}, z_{t+t'})$ for each bridge step at offset $t' \in \{1, \ldots, f-1\}$：
- $t'=1$: input 是 fresh VLM feature $z_t$（boundary convention）
- $t' \geq 2$: input 是 bridge 自己（可能 noisy）的预测

**R1 training**: 把 sync data 与 DAgger data 按自然比例拼接（Table 12 ablation: 不 reweighting 最好，sync only 91.0%, DAgger only 92.5%, mixed 95.0% on GR00T LIBERO-Goal）。从 R0 weights resume，LR 降 10×（$3 \times 10^{-5}$ for LIBERO-10，长 horizon 需要更温和 adaptation 防止 collapse）。

DAgger 带来的 SR 提升：Table 4 显示
- π0.5 KV bridge: R0 ~93-96% → R1 97-100%，长 horizon +4pp
- GR00T feature bridge: R0 91-93% → R1 95-98%，LIBERO-10 上 +13pp（54.33 → 67.17）

Reference:
- DAgger (Dataset Aggregation): https://arxiv.org/abs/1011.0686
- Ross & Bagnell 2010 original DAgger paper

### 4.3 Stage 3: Optional enhancements

**Phase-aware VLM scheduling**: 根据 previous action 的 translation magnitude 动态调 $f$：
- $\|a_t^{\text{trans}}\| > \tau_{\text{nav}}$ → $f=2$（高 motion，bridge 不可靠）
- $\tau_{\text{manip}} < \|a_t^{\text{trans}}\| \leq \tau_{\text{nav}}$ → $f=3$（中 motion）
- $\|a_t^{\text{trans}}\| \leq \tau_{\text{manip}}$ → $f=4$（低 motion，manipulation 阶段）

这个 idea 与 adaptive computation / dynamic depth 思想一致。在 GR00T LIBERO-10 上把 R1 67.17% 推到 89.17%。

**LoRA action head adaptation**: 当 bridge feature 比 clean VLM feature noisy 时，对 action head 的 DiT 加 LoRA，训练用 50/50 mix of bridge + sync features 加 Gaussian noise augmentation，提升 action head 对 bridge noise 的 tolerance。

Table 4 显示 GR00T LIBERO-10: R1 67.17% → Stage 2 LoRA 73.17% → Stage 3 phase-aware 89.17%，总共 +34.84pp。

Reference:
- LoRA: https://arxiv.org/abs/2106.09685

---

## 5. Experiments 详细分析

### 5.1 Main results (Table 1)

**GR00T-N1.6-3B** (backbone 63ms, action head 27ms, sync total 90ms):

| Method | Spatial | Object | Goal | Long | Avg SR | Latency (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| Sync | 96.17 | 99.83 | 97.33 | 93.00 | 96.58 | 90 | 1.00× |
| +FastV | 86.17 | 84.00 | 85.00 | 58.83 | 78.50 | 77 (-15%) | 0.95× |
| +VLA-Cache | 91.17 | 90.00 | 89.00 | 85.17 | 88.84 | 75 (-17%) | 1.08× |
| +Feature Caching | 35.67 | 3.00 | 56.00 | 42.33 | 34.25 | 48 (-47%) | † |
| +Latent Bridge | 95.83 | 97.83 | 95.33 | 89.17 | 94.54 | 49 (-45%) | 1.73× |

**π0.5** (backbone 46ms, action head 30ms, bridge 6ms, f=4):

| Method | Spatial | Object | Goal | Long | Avg SR | Latency (ms) | Speedup |
|---|---|---|---|---|---|---|---|
| Sync | 98.83 | 98.17 | 97.00 | 93.83 | 96.96 | 76 | 1.00× |
| +FastV | 88.17 | 86.00 | 86.83 | 62.17 | 80.79 | 67 (-12%) | 0.95× |
| +VLA-Cache | 93.17 | 91.00 | 92.17 | 86.83 | 90.79 | 65 (-14%) | 1.08× |
| +Feature Caching | 46.67 | 57.33 | 68.83 | 52.67 | 56.38 | 45 (-41%) | † |
| +Latent Bridge | 99.00 | 97.67 | 97.33 | 93.67 | 96.92 | 46 (-39%) | 1.65× |

**关键观察**:
1. Latent Bridge 在两个 VLA 上都几乎无 SR 损失（94.54% vs sync 96.58% on GR00T；96.92% vs 96.96% on π0.5）
2. Feature caching collapse：GR00T Object 从 99.83% 掉到 3.00%，证明 learned delta prediction 必要
3. FastV 在 GR00T 上甚至 negative speedup（0.95×），因为 token pruning 破坏 dual-system action head 的 dense attention pattern
4. Episode length 增加 +5-6%（bridge 偶尔 detour），已被计入 net speedup

### 5.2 Cross-benchmark (Table 2, f=3)

| Benchmark | Tasks | Sync | Cache | Bridge | Retention | Cache Ret. |
|---|---|---|---|---|---|---|
| RoboCasa Door/Drawer | 6 | 78.33% | 66.67% | 75.83% | 96.81% | 85.12% |
| RoboCasa Appliance | 7 | 80.67% | 65.00% | 77.17% | 95.66% | 80.58% |
| RoboCasa Coffee | 3 | 53.33% | 41.67% | 56.67% | 106.26% | 78.13% |
| RoboCasa Pick-and-Place | 8 | 49.33% | 26.83% | 43.83% | 88.85% | 54.39% |
| RoboCasa Avg (24) | 24 | 66.22% | 49.78% | 63.16% | 95.38% | 75.17% |
| ALOHA sim (π0.5) | 1 | 88.00% | 46.00% | 86.00% | 97.73% | 52.27% |

跨 embodiment（Panda, ALOHA bimanual）、跨 action space（7-DoF vs 14-DoF）、跨 task distribution 同一 pipeline 直接迁移，证明 task-agnostic 设计。

### 5.3 Ablations

**Vision input ablation** (Table 3, π0.5 KV bridge):

| Variant | Sp | Ob | Go | L-10 |
|---|---|---|---|---|
| Full (148M) | 99.00 | 97.67 | 97.33 | 93.67 |
| w/o vision | 98.17 | 95.50 | 86.17 | 65.33 |
| w/o stable | 98.00 | 97.83 | 92.17 | 82.67 |
| Small (19M) | 99.17 | 98.83 | 97.50 | 93.50 |

去掉 vision input 在 LIBERO-10 掉 -28.34pp；去掉 stable context 掉 -11.00pp。19M 小 bridge 与 148M 持平甚至略高，说明 capacity 不是 bottleneck。

**VLM call period sweep** (Table 10, π0.5):

| f | Spatial | Object | Goal | L-10 | Savings |
|---|---|---|---|---|---|
| 1 (sync) | 98.7 | 98.3 | 97.0 | 94.0 | 0% |
| 2 | 98.5 | 98.0 | 97.0 | 93.5 | 50% |
| 3 | 99.0 | 100 | 99.0 | 92.5 | 67% |
| 4 (default) | 99.0 | 97.5 | 97.5 | 93.5 | 75% |
| 5 | 97.5 | 99.5 | 96.0 | 91.0 | 80% |
| 6 | 98.0 | 96.5 | 98.5 | 87.5 | 83% |
| 8 | 95.5 | 96.5 | 94.5 | 84.5 | 88% |
| 12 | 93.0 | 97.5 | 89.5 | 77.0 | 92% |

短 horizon suite 在 f=8 仍 >94.5%，LIBERO-10 在 f=8 掉到 84.5%，long-horizon error compounding。

**Per-layer KV prediction quality** (Table 8, π0.5):

| Metric | L0 | L5 | L10 | L17 |
|---|---|---|---|---|
| Bridge | 0.999 | 0.999 | 0.997 | 0.994 |
| Copy baseline | 0.997 | 0.998 | 0.993 | 0.995 |

早期层（L0-L5）KV 主要依赖 fixed text prompt，cos 0.999+，最容易预测；后期层（L10-L17）编码更多 observation-dependent content，更难。但 bridge 在所有层都优于 copy baseline。

**Chained-prediction cosine decay** (Table 9):

| Offset k | 1 | 2 | 3 | 5 |
|---|---|---|---|---|
| Bridge cos | 0.998 | 0.996 | 0.994 | 0.989 |
| Copy-only cos | 0.993 | 0.988 | 0.983 | 0.974 |

Bridge 在 offset 5 仍维持 0.989 vs copy 0.974，差距随 offset 增大而扩大。

---

## 6. Case Study: 为什么 Caching 必败

Figure 5 的 LIBERO-Spatial episode at f=3:
- Caching 的 KV cosine 在 sawtooth pattern 中掉到 ~0.91（VLM call 时回到 ~1.0，然后线性下降）
- Bridge 一直维持 ~0.99

Sawtooth 的成因：cache 只编码 stale scene，robot 每移动一步 stale KV 与 ground truth 的差距累积。Bridge 通过 cheap signals（SigLIP delta ~5ms, state, previous action）主动预测 update，几乎闭合 gap。

---

## 7. Intuition Building: 与现有 paradigm 的联系

### 7.1 与 World Models / JEPA 的关系

Latent Bridge 与 LeCun 的 JEPA (Joint-Embedding Predictive Architecture) 哲学相近：都在 representation space 预测未来 latent state，而非 pixel space。区别：
- JEPA 学 abstract state for representation learning
- Latent Bridge 直接在 VLM 的 hidden representation 上预测，目的是 accelerate inference 而非学 representation

V-JEPA 是 JEPA 的 video 版本，同样预测 latent 而非 pixel。Latent Bridge 可以视作 "JEPA for inference acceleration"。

Reference:
- JEPA / LeCun position paper: https://arxiv.org/abs/2306.02706 (LeCun 2022 position paper)
- V-JEPA: https://arxiv.org/abs/2404.08471
- Dreamer V3: https://arxiv.org/abs/2301.12578
- IRIS world model: https://arxiv.org/abs/2301.12578

### 7.2 与 Speculative Decoding 的关系

Speculative decoding 在 LLM 中用小模型 draft 多个 token 再大模型 verify，类似 idea 在于 "小模型预测、大模型校正"。Latent Bridge 的差别：
- 不 verify，直接用 prediction 走 action head
- 通过 DAgger offline 校正 distribution shift，而非 online verify
- 周期性 VLM call 等价于 implicit "anchor point"

Reference:
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Medusa: https://arxiv.org/abs/2401.10774

### 7.3 与 KV cache compression 的关系

SnapKV / StreamingLLM 通过保留重要 KV position 压缩长 context。Latent Bridge 不压缩，而是预测 delta。可以与 KV compression 正交组合：在 VLM step 上压缩 KV，bridge step 上预测 delta。

Reference:
- SnapKV: https://arxiv.org/abs/2404.14469
- StreamingLLM: https://arxiv.org/abs/2309.17453

### 7.4 与 Control Theory 中 Incremental Control 的同构

$\hat{z}_{t+1} = \hat{z}_t + \Delta_t$ 等价于 control 中 incremental form：
$$x_{t+1} = x_t + f(x_t, u_t) \Delta t$$

Zero-init output projection 让未训练 bridge = identity，对应 zero control input，系统稳定。这与 ResNet 的 residual learning 是同一种稳定性 trick。

### 7.5 与 Consistency Models / Flow Matching 的潜在组合

Paper Conclusion 提到：π0.5 的 action expert 仍要 traverse 18 层，留下 ~30ms denoising floor。如果能用 consistency distillation 把 action head 单步化，speedup 可能突破 2×。

Reference:
- Consistency models: https://arxiv.org/abs/2303.01469
- Flow matching: https://arxiv.org/abs/2210.02747
- π0 flow matching: https://arxiv.org/abs/2410.24164

### 7.6 与 Async Inference / Action Chunking 的对比

Action chunking（如 Diffusion Policy, π0）一次预测多步 action，可以 amortize perception cost。Latent Bridge 与之正交：可以在 chunk 内每个 control step 都用 bridge 更新 feature，避免 stale feature 问题。

Reference:
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT (Action Chunking Transformer): https://arxiv.org/abs/2304.13705

---

## 8. Limitations 与未来方向

Paper Section 5 与 Appendix J 列出：
1. **DAgger 需要 simulator**：Stage 3 需要 simulator 跑 bridge rollout + VLM oracle。Real-robot deployment 需 R0-only 或 offline DAgger 数据生成。
2. **State-in-prompt 不兼容**：π0.5 若把 proprioceptive state tokenize 进 language prompt（discrete_state_input=True），cached prefix KV 在 bridge step 携带 stale state。ALOHA π0.5 必须 fine-tune with discrete_state_input=False。
3. **Per-checkpoint training**：换 VLA checkpoint 需重训 bridge（2-4 小时/single GPU，开销可接受）。
4. **Episode length +5-6%**：bridge 偶尔 detour，已被计入 net speedup。
5. **Interleaved 架构 action expert floor**：π0.5 仍有 ~30ms denoising floor，consistency distillation 可能突破。

未来方向（paper 暗示 + 个人联想）：
- **Bridge + consistency distillation** 组合突破 2× speedup
- **Bridge + token pruning composability**：VLM step 上跑 pruning，bridge step 上 skip
- **Cross-embodiment bridge transfer**：当前 per-suite 训练，能否训一个 universal bridge？
- **Real-robot DAgger** via real VLM oracle（成本高但可行）
- **Hierarchical bridge**：短程 delta bridge + 长程 latent dynamics model

---

## 9. 关键 Takeaways（build intuition）

1. **Feature temporal redundancy is the goldmine**: dual-system VLA 的 VLM output 在 consecutive step 间 cosine >0.999 on stable layers，意味着中间步骤 99% 的 VLM compute 是 redundant。这是 paper 立足的 empirical foundation。

2. **Predict delta, not absolute**: zero-init 让 bridge 从 caching baseline 平滑起步，训练 stability 与 deployment robustness 都受益。

3. **Autoregressive chaining creates distribution shift → DAgger fixes it**: 这是 paper 最 important 的 training insight。R0 sync-only 训练 93-96%，DAgger R1 推到 97-100%，长 horizon 任务上 +13pp。

4. **Architecture-agnostic instantiation**: feature-space (GR00T) 与 KV-cache (π0.5) 两种接口都 work，说明 delta prediction 思想可以泛化到任何 dual-system VLA。

5. **Amdahl ceiling escape**: token pruning 在 dual-system 上 net speedup 被 Amdahl bound 在 ~1.3×，bridge 通过 skip backbone 完全突破这个 ceiling。

6. **Phase-aware scheduling 为 long-horizon 量身定做**: LIBERO-10 上把 67% → 89%，说明 static f 在 long-horizon 不够，需要 dynamic allocation。

7. **小 bridge 够用**: 19M 参数 bridge 与 148M 持平，意味着 delta prediction 是低 complexity 任务，bridge 容易部署。

---

## 10. 我的延伸思考

### 10.1 Bridge 是否可以做成 universal across checkpoints?

Paper 说 bridge 是 per-checkpoint。但 stable layer 的 cosine >0.999 是 VLM 通用 property，如果 bridge 学的是 "feature dynamics" 而非 "specific VLM internal"，是否可以 train 一个 universal delta predictor with cross-attention to VLM-specific embedding? 这与 HPT (Heterogeneous Pre-trained Transformers) 的哲学一致。

### 10.2 Bridge 与 Action Chunking 的最优组合

如果 action head 已 chunk size k，bridge 是否只需在 chunk 内预测一次（而非每 control step）? 这可能进一步降低 bridge call frequency。

### 10.3 Bridge 在 real robot 上的部署

DAgger 依赖 simulator。Real robot 上可以做 offline DAgger via recorded teleop data + offline VLM feature extraction。或者用 RL fine-tuning 替代 DAgger（更贵但更通用）。

### 10.4 Bridge 在其他 dual-system AI 上的扩展

任何 large backbone + small head 的架构都有类似 redundancy：
- LLM agent + tool use：可以预测 tool call 之间的 reasoning delta
- Video generation + post-processing：可以预测 frame 间 latent delta
- Scientific computing + surrogate model：可以预测 solver iteration 间 state delta

Reference:
- HPT: https://arxiv.org/abs/2409.20537
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1804.01508

---

## 11. 总结

Latent Bridge 是 dual-system VLA 加速方向的 elegant 工作，核心 contribution 有三：
1. **Delta prediction bridge**：轻量模型预测 VLM feature delta，跳过 50-75% backbone call
2. **Architecture-agnostic instantiation**：feature-space (GR00T) 与 KV-cache (π0.5) 两种接口都 work
3. **Task-agnostic DAgger pipeline**：解决 autoregressive chaining 的 distribution shift，跨 LIBERO/RoboCasa/ALOHA 直接迁移

在 1.65-1.73× wall-clock speedup 下保留 95-100% SR，远超 token pruning 在 dual-system 上的 Amdahl-bounded 1.08-1.30× ceiling。这个工作为 VLA 实时部署 (10-50Hz) 打开了一扇门，同时提出了一个普适 idea：**预测 delta 比加速原 forward 更便宜**。

主要参考链接汇总：
- GR00T N1: https://arxiv.org/abs/2503.14734
- π0.5: https://arxiv.org/abs/2504.16054
- π0: https://arxiv.org/abs/2410.24164
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboCasa: https://arxiv.org/abs/2406.02523
- ALOHA: https://arxiv.org/abs/2304.13705
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- Octo: https://arxiv.org/abs/2405.12213
- DAgger: https://arxiv.org/abs/1011.0686
- LoRA: https://arxiv.org/abs/2106.09685
- DiT: https://arxiv.org/abs/2212.09748
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- FastV: https://arxiv.org/abs/2403.06764
- VLA-Cache: https://arxiv.org/abs/2502.02175
- Speculative decoding: https://arxiv.org/abs/2211.17192
- Consistency models: https://arxiv.org/abs/2303.01469
- SigLIP: https://arxiv.org/abs/2303.02588
- PaliGemma: https://arxiv.org/abs/2407.07726
- SnapKV: https://arxiv.org/abs/2404.14469
- StreamingLLM: https://arxiv.org/abs/2309.17453
- V-JEPA: https://arxiv.org/abs/2404.08471
- HPT: https://arxiv.org/abs/2409.20537
- RoPE: https://arxiv.org/abs/2104.09864
- ResNet: https://arxiv.org/abs/1512.03385

这篇 paper 给我最大的 intuition boost 在于：**它把 "feature temporal redundancy" 这个观察提升到 actionable design principle**，并通过 DAgger 优雅地解决了 autoregressive chaining 的 distribution shift。Bridge 的 zero-init design 与 control theory 中的 incremental control 同构，体现了 ML 与 control 的深度连接。
