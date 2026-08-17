---
source_pdf: Composer 2 Technical Report.pdf
paper_sha256: 933e9f2720966b6d744b8295af0fe2c2fb4c3d9efc3d23a4d27b4b0b08c0d068
processed_at: '2026-08-03T16:41:45-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Composer 2 Technical Report 人话版

Andrej，我之前那版太学究气了，这次用聊天的方式重新讲一遍，把核心直觉讲透就行。

---

## 这篇 paper 到底在干嘛

Cursor 团队训了一个专门写代码的模型，叫 Composer 2。他们想证明一件事：**拿一个开源的 base model，加上自己的一套 post-training pipeline，就能训出一个 coding 能力逼近 GPT-5.4、但推理成本低得多的模型**。

base model 选的是 Kimi K2.5（1.04T 总参数，32B 激活参数的 MoE），post-training 分两步走：

1. **Continued pretraining**：猛灌代码数据，让模型"懂代码"
2. **Reinforcement learning**：让模型在真实的 Cursor 环境里做 agent 任务，做对了奖励，做错了惩罚

听起来简单，但每一步都有无数工程坑。paper 的价值就在于把这些坑全写出来了。

---

## 为什么不直接用 public benchmark

Cursor 团队觉得 SWE-bench 这类公开 benchmark 已经不能反映真实开发场景了，原因有四个：

**第一个问题：任务类型不对**。SWE-bench 基本就是"读 GitHub issue → 改几行代码 → 跑测试"，但真实开发是"产品经理甩来一句模糊的需求 → 你要自己搞清楚 context → 改几百行代码"。Terminal-Bench 更离谱，很多任务是"算 chess moves"这种抽象谜题，跟写代码没啥关系。

**第二个问题：prompt 太明确了**。SWE-bench 的 prompt 经常写成"在 `foo.py` 第 42 行把 `bar()` 改成 `baz()`"，但真实用户说的是"这个 retry 逻辑好像有点问题，你看看"。前者几乎没有 ambiguity，后者需要模型自己探索。

**第三个问题：数据污染**。SWE-bench 从开源 repo scrape，这些 repo 很可能已经混进训练集了。OpenAI 已经因为这个暂停了 SWE-bench Verified 的报告（https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/），因为他们发现 frontier model 能直接从记忆里"背出"gold patch。

**第四个问题：只看对错**。public benchmark 只测 functional correctness，但真实开发还要看 code quality、latency、cost、agent 跟你交互的体验好不好。

所以他们搞了个 CursorBench，从自己工程师的真实 coding session 里抽 task。Figure 7 给的对比很直观：

| Benchmark | 中位改动行数 | 中位 prompt 长度 |
|-----------|-------------|-----------------|
| SWE-bench | 7-10 行 | 1185-3055 字符 |
| CursorBench | 181 行 | 390 字符 |

CursorBench 的 task 就是"prompt 很短很模糊，但要改的代码很多"，这跟真实开发一致。

Figure 8 给了个 example：一个 retry loop 的 bug report，agent 要从 production logs 里发现 esbuild 0.20.2 的 transpilation bug——`var`-scoped error state 在 retry 之间没 reset。这种 task 你给 SWE-bench 根本没法构造。

---

## Continued Pretraining：先让模型"懂代码"

这阶段分三步：

**第一步**：在 32k context length 上跑大量 compute，data mix 以代码为主。这是预算的大头。

**第二步**：短阶段把 context 扩展到 256k。

**第三步**：更短的 targeted SFT，收尾。

paper 做了个很漂亮的 controlled experiment（Figure 2 left）：拿 Qwen3-Coder-30B-A3B，跑三个 log-spaced compute level 的 continued pretraining，每个 checkpoint 走相同的 SFT + RL，看 final loss vs RL reward 的关系。

结果：**cross-entropy loss 和 RL reward 几乎 log-linear 相关**。loss 降一个量级，RL reward 稳定提升。

这个发现的直觉是——RL 是在 latent solution manifold 上做 sharpening，pretraining 决定 manifold 的覆盖度。如果 pretraining 没把某个 solution region 覆盖到，RL 再怎么 sharpen 也找不到那块解。

### Multi-Token Prediction (MTP) 头

为了让 inference 更快，他们加了 MTP 头做 speculative decoding。这个设计的几个细节：

- MTP 头从 continued pretraining **中段 checkpoint** 开始训，不从 base 开始。直觉是：主 head 在剧烈变化时，MTP head 追不上。等主 head 稳定了再训 MTP，收敛快。
- 训练目标是 **self-distillation**：MTP head 去逼近主 head 的 logit 分布（soft target），不是 ground-truth token 的 one-hot。这样 draft 分布和 target 分布同源，speculative decoding 的 accept rate 高。
- 后两个阶段（long-context extension + SFT）把 MTP head 一起带上，让它跟主 head 同步适配。

参考 DeepSeek-V3 的 MTP 原始设计：https://arxiv.org/abs/2412.19437

---

## RL 阶段：算法层面的核心决策

这是 paper 信息密度最高的部分。我逐个决策讲。

### 决策 1：用 GRPO 但做减法

GRPO（Group Relative Policy Optimization）的核心是：每个 prompt 采 G 条 rollout，advantage = 这条 rollout 的 reward 减去 group 内平均 reward。不需要 critic。

Composer 2 在此基础上做了三个减法：

**减法 A：去掉 length standardization**。原版 GRPO 把 reward 除以 sequence length，Dr. GRPO（https://arxiv.org/abs/2503.20783）指出这会引入 length bias。直觉：一个 5000 token 的正确解不应该因为"长"就被自动打折。

**减法 B：不除 group std**。如果 group 内 8 条 rollout 全对（reward 全=1），std≈0，除以 std 会把 advantage 炸到无穷。这在 coding RL 里特别常见，因为代码 correctness 是 binary 的。paper 原话：

> "small behavioral differences get massively upweighted within a group where every rollout achieves equal correctness."

**减法 C：不 mask overlong rollouts**。DAPO（https://openreview.net/forum?id=2a36EMSSTp）建议把超过 max length 的 rollout mask 掉。Composer 2 实验下来发现没用，加上 self-summarization 已经结构性减少了 overlong 的发生。

### 决策 2：KL estimator 用 $k_1$ 不用 $k_3$

这块很精彩。RL 训练要加 KL regularization 防止 policy 跑飞。KL 的标准定义：

$$\mathrm{KL}(q \| p) = \mathbb{E}_{x \sim q}\left[-\log \frac{p(x)}{q(x)}\right]$$

记 $r(x) = p(x)/q(x)$，三个常见 estimator：

| Estimator | 公式 | 特点 |
|-----------|------|------|
| $k_1$ | $-\log r$ | 无偏，方差随 $r$ 远离 1 平滑增长 |
| $k_2$ | $(r-1)/2$ | 有偏，方差小 |
| $k_3$ | $(r-1) - \log r$ | 无偏，$r \to 1$ 方差趋 0，$r$ 远离 1 方差爆炸 |

很多开源 RLHF 实现默认用 $k_3$，因为 $p, q$ 接近时它方差最小。但在异步 RL 里，trainer 和 inference worker 的 policy 版本不一致，$r$ 经常远离 1。这时候 $k_3$ 的方差会爆炸，个别 sample 的梯度贡献被无界放大，训练崩。

Composer 2 选 $k_1$。虽然 $k_1$ 整体方差比 $k_3$ 高，但 bounded，不会因为个别 sample 炸掉。

参考 Schulman 的 blog（https://openai.com/research/approximating-kl-divergence）和 Amini et al. 2025（https://openreview.net/forum?id=um9kHMof0c）。

### 决策 3：Router Replay for MoE

MoE 模型在 inference engine 和 trainer 上跑 forward，由于浮点累加顺序、quantization 路径差异，router 可能选不同 expert。如果 trainer 算 log-prob 时用 expert A，但 inference 采样时用 expert B，policy gradient 就在优化一个"错误分布"的对数概率。

解决方案：inference 时返回每个 token 在每个 MoE layer 选的 expert indices，trainer forward pass 时强制 router 用这些 indices。router 的 gating score 仍然正常计算，gradient 仍然能流。

Composer 2 的扩展：加 **plausibility threshold**——只 replay gating score 高于某个阈值的 expert，其他用 router 自己的 top-k 候选补上。直觉是"信任 inference 的选择，但只信任 inference 自己也觉得合理的选择"。

参考 Ma et al. 2025：https://arxiv.org/abs/2510.11370

### 决策 4：Self-Summarization

长 horizon 任务 context 会爆。解决方案是让模型自己写 summary 串联多段 generation。

- summary 由模型自己生成（不是外部 model 帮忙总结）
- final reward 复制到所有 token 上，包括 summary token
- KV cache 在 summary 处可以重置

为什么 reward 复制到 summary token 是合理的：summary 本身没有 ground-truth reward，但它的质量直接决定后续 generation 能否成功。把 terminal reward 复制回去等于做 credit assignment——好 summary 被强化，坏 summary（丢信息）被弱化。

相比 prompt-based compaction（让外部 model 总结），self-summarization 让 summary 也成为 policy 的一部分，被 RL 直接优化，token 效率高一个数量级。

参考 Composer 1.5 的 blog：https://cursor.com/blog/self-summarization

### 决策 5：非线性 length penalty

公式：

$$C_{\text{length}\{k, q\}}(x) = \frac{(1 + kx)^{1-q} - 1}{k(1-q)}$$

变量含义：
- $x$：综合度量，包括 thinking tokens、tool calling tokens、tool output tokens、final message tokens、tool call 数量、turn 数量。就是"effort"的加权和
- $k$：曲率参数，控制函数多快进入饱和区
- $q$：凹度参数，控制函数凹的程度

几个极限帮你建直觉：

- $q = 0$：$C = x$，纯线性，没有"简单任务该快"的偏好
- $q \to 1$：用 L'Hôpital 取极限得 $C = \frac{\log(1+kx)}{k}$，对数增长，强凹，长任务被严重打折
- $q \in (0, 1)$：温和凹，这是实际使用的 regime

**为什么 concave down + increasing 是对的**：easy task 上（$x$ 小），函数斜率大，每多花一点 effort 被罚很重，逼模型快速答完；hard task 上（$x$ 大），斜率变小，effort 的边际成本趋于 0，模型可以放心想很久。

实际效果：模型学会在简单任务上做 parallel tool calls 来摊薄 #turns 的 cost。

---

## Infrastructure：训练系统的工程架构

### 并行策略

Composer 2 从 TP 主导改成 **CP (Context Parallelism) 主导**。

**为什么不用 TP**：TP 把 hidden dimension 切碎成 skinny local matmul，compute efficiency 低。

**为什么用 CP**：CP 只在 attention 那层做 sequence-wise 切分，保留 full hidden dim，通信开销小。

**MLA + CP 的实现细节**（MLA = Multi-Head Latent Attention，DeepSeek 的设计）：

1. 每个 CP rank 先在本地算 KV latent vector（低秩 latent）
2. all-gather latent vectors 跨 CP ranks
3. 各 rank 独立做 KV projection（虽然冗余，但 projection 计算量小，省掉了 projection 后大 tensor 的 all-gather 通信）
4. CP 通信和 Q projection 的计算完全 overlap

**Load imbalance 解决**（Liu et al. 2023, https://openreview.net/forum?id=WsRHpHH4s0）：naive CP 让前 chunk 处理的 token 少（attend 的过去 token 少），后 chunk 处理多。解决方案：把 sequence 切成 $2 \times \text{CP}$ 个 chunks，第 $i$ 个 rank 处理 chunk $i$ 和 chunk $2 \times \text{CP} - 1 - i$。每个 rank 工作量大致相等（前面少 + 后面多）。

**EP 和 TP 解耦**：原版 MoE 设计 EP 复用 TP rank group，限制了 EP scaling 上限。Composer 2 让 EP 独立从 DP + CP capacity 出来，可以 EP=8 + CP=2（pretraining）或 EP=8 + CP=8（RL）。

### NVFP4 量化改造

这是 paper 最硬核的工程部分。

**原版 NVFP4**（NVIDIA 标准, https://arxiv.org/abs/2509.25149）：
- Value: FP4E2M1（4 bit：2 bit exp + 1 bit mantissa + 1 bit sign）
- Scale: FP32 per-tensor

**Composer 2 的变体**（MoE forward）：
- Value: FP4E2M1（不变）
- Block scale: FP8E4M3，block size = 16
- Token scale: FP32 per-token

**为什么必须改**：

**原因 1：per-tensor scale 是 batch-variant 的**。scale 是整个 tensor 的 max-abs 除以 FP4 表示范围。每个 batch 的 max-abs 不同，scale 不同。这让训练在不同 batch 间数值不可复现，RL 训练直接发散。

**原因 2：per-tensor scale 有信息泄漏**。tensor-level max-abs 包含了"未来 token"的信息（因为 max 是全 tensor 算的）。反向传播时 past token 的梯度被 future token 的统计量"污染"。

直觉：per-tensor quantization 在 inference 上 OK（一次性 forward），在 training 上是 poison。per-token + per-block 双层 scaling 把 quantization 的 granularity 降到 token 级，每个 token 的 scale 只依赖自己，没有 cross-token 信息流。

**Backward pass 用 MXFP8**：forward 必须用低精度（NVFP4）让 trainer 跟 inference engine 数值对齐。backward 只在 trainer 上跑，可以放宽到 MXFP8（FP8E4M3 + FP8E8M0 per-32 block scale）换稳定性。

**IEEE-compliant vs fast-approx 数学**：
- NVFP4 quantization 必须用 IEEE-compliant float arithmetic（如 `__fdiv_rn`），用 fast-approx（如 `__fdividef`）会在 ~100 RL steps 后发散
- MXFP8 quantization 用 fast-approx 没事，选 fast-approx 换性能

FP4 的 dynamic range 太窄，任何 IEEE 标准外的近似误差都会被量化过程放大成 systematic bias，积累几百步后训练崩。FP8 的 range 够宽，近似误差被天然吸收。

参考 Cursor 的 kernel blog：https://cursor.com/blog/kernels

### RL Infrastructure：四个解耦服务

**Training**：Ray + PyTorch，全异步。centralized reconciler 做 slot-based sample lifecycle 管理。所有服务用 future 模式，upstream dependency ready 就触发 eager execution。Ray object store 自动 spill 到 NVMe。

**Environments**：Anyrun 平台，跑 untrusted code。每个 pod 是一个 Firecracker VM，能跑完整 dev environment（包括 browser + GUI）。Anyrun 支持 fork + snapshot（filesystem 和 memory level），让 mid-trajectory rollout checkpoint 成为可能。

**Inference**：和 Fireworks AI 合作。每 step 同步权重用 **delta compression**——每 rank 缓存上次上传，只传 diff。1T 模型 diff 压缩到几 GB，全 sharded 上传下载打满 egress 带宽，跨 region 通过 S3 bucket 不需要直接 connectivity。

**Evaluations**：pin production backend + Cursor client 的版本，跨 region weight sync 到 eval cluster。

**关键架构决策**：

1. **fault tolerance 到 process level**：硬件故障节点标 unhealthy 但训练继续，warm standby 顶上。避免了"一个 GPU 挂了整个 job 重启"。
2. **rollout-level checkpointing**：long-running coding rollout replay 很贵，除了 step-level model checkpoint，还要做 rollout-level（codebase env 的 memory snapshot）和 group-level（advantage-tagged sequence 写到 NFS）。
3. **Anygress egress 控制**：所有 pod 出网经过 Anygress proxy，注入 trusted root CA + TCP 层 redirect，让 proxy 对 pod 透明。模拟真实 dev environment 的网络行为。
4. **Tool 一致性**：用 production Cursor backend 的 shadow deployment 同时给 dataset prep 和 rollout 用，保证 train/test harness 完全一致。

---

## 结果分析

### 主表

Table 1 的关键数字：

| Model | CursorBench | SWE-bench Multi. | Terminal-Bench |
|-------|-------------|------------------|----------------|
| Composer 2 | 61.3 | 73.7 | 61.7 |
| Composer 1.5 | 44.2 | 65.9 | 47.9 |
| Composer 1 | 38.0 | 56.9 | 40.0 |
| Opus 4.6 High | 58.2 | 75.8/77.8 | 58.0/65.4 |
| GPT-5.4 | 63.9 | 76.8 | 66.5/75.1 |
| Kimi K2.5 (base) | 36.0 | 65.1/73.0 | 47.3/50.8 |

几个观察：

1. **Composer 2 vs Kimi K2.5**：CursorBench 从 36.0 → 61.3（+70%），但 SWE-bench Multi 只从 65.1 → 73.7（+13%）。CursorBench 的灵敏度远高于 public benchmark——base model 已经在 public benchmark 上接近饱和，只能在 CursorBench 上拉开差距。

2. **vs GPT-5.4**：CursorBench 上 GPT-5.4 是 63.9，Composer 2 是 61.3，差 2.6 个点。但 GPT-5.4 成本远高于 Composer 2。Figure 11 显示 Composer 2 在 cost-accuracy Pareto frontier 上是最优的。

3. **vs Opus 4.6 High**：CursorBench 上 Composer 2 (61.3) > Opus 4.6 (58.2)，但 Terminal-Bench 上 Opus 4.6 self-reported 是 65.4，Composer 2 是 61.7。这种 cross-benchmark 不一致正好印证了 domain mismatch 论点。

### Best-of-K vs Average Reward

Figure 5 的关键发现：**RL 训练让 average 和 best-of-K 同步上升，没有 trade-off**。

这个结果 paper 自己说"notable"，因为近期一系列 paper（Yue et al. 2025 NeurIPS oral, https://openreview.net/forum?id=4OsgYD7em5）claim RL 主要在做"reweighting existing correct trajectories"，导致 best-of-K 不动甚至下降。

Composer 2 的反例说明：在 agentic coding domain，RL 仍然在扩展正确解的 coverage。可能原因：

- Coding task 的解空间结构跟 math reasoning 不同，coding 解之间是结构性不同的 approach，RL 在 approach 之间做有效探索
- Self-summarization 让模型在 long-horizon 任务上学到"如何高效利用 context"，这种 skill 不在 base model 的 trajectory pool 里
- Tool use 引入了额外的 action 维度，RL 在 tool use 策略上学到新东西

---

## Base Model 为什么选 Kimi K2.5

Appendix B 给了评估细节：

| Model | FreshBench ↑ | State Tracking ↓ | NLL ↓ |
|-------|--------------|------------------|-------|
| DeepSeek V3.2 | 68.9% | 66 | 11.75M |
| Kimi K2.5 | 83.2% | 86 | 13.81M |
| GLM-5 | 79.2% | 92 | 14.11M |

有意思的是 Kimi K2.5 的 NLL 不是最低的（DeepSeek V3.2 更低），但 Cursor 还是选了 K2.5。推测原因：

1. **K2.5 是 MoE**（1.04T / 32B active），inference 成本远低于 dense 模型。Composer 要做 product 部署，inference cost 决定商业可行性
2. **FreshBench 上 K2.5 (83.2) 显著高于 DeepSeek V3.2 (68.9)**。FreshBench 测的是"recent library 知识 / 需要读 source code 才能 answer 的问题"，这跟 agentic coding 的实际 workload 更接近
3. **State tracking 上 K2.5 (86) 比 DeepSeek V3.2 (66) 差**，但 RL 阶段可以补救

关键 insight：Cursor 不用 agent benchmark 选 base model，因为"agentic capability 在 RL 阶段会剧变"。用更"primitive"的 benchmark（knowledge、state tracking、perplexity）选 base，因为它们更 predictive of post-RL ceiling。

---

## 这篇 paper 的核心 takeaway

1. **Domain-specialized model 通过 RL on top of open base 是可行路径**。不需要从头 pretrain，但需要一套完整的 post-training pipeline + harness engineering + infra。

2. **Public benchmark 已经不足以反映真实 agent 能力**。CursorBench 的设计哲学（短 prompt + 大改动 + 持续迭代）值得所有做 agent eval 的人学习。

3. **RL 算法上的减法比加法重要**。去掉 length normalization、不除 std、不 mask overlong、用 $k_1$ 不用 $k_3$——每一个减法都有清晰的 degenerate case 论证。

4. **MoE RL 的 router replay + plausibility threshold 是 infra 必需品**。不做这个，policy gradient 在优化错误分布。

5. **NVFP4 在 training 上必须改成 per-token scale**。per-tensor scale 的 batch-variant + 信息泄漏问题会让 RL 发散。

6. **Self-summarization 是 long-horizon agent 的结构性解决方案**，比 prompt-based compaction 在 token 效率和 reward signal 上都更优。

7. **RL 在 agentic coding 上仍然在扩展 coverage**，不是只做 reweighting。Figure 5 的结果如果 reproducible，会改写 RL-on-LLM 的理论叙事。

---

## 相关 paper 链接

- Dr. GRPO: https://arxiv.org/abs/2503.20783
- DAPO: https://openreview.net/forum?id=2a36EMSSTp
- DeepEP: https://github.com/deepseek-ai/DeepEP
- ThunderKittens: https://openreview.net/forum?id=0fJfVOSUra
- PipelineRL: https://arxiv.org/abs/2509.19128
- Kimi K2.5: https://arxiv.org/abs/2602.02276
- Amini et al. on KL estimators: https://openreview.net/forum?id=um9kHMof0c
- Yue et al. NeurIPS oral: https://openreview.net/forum?id=4OsgYD7em5
- Terminal-Bench: https://openreview.net/forum?id=a7Qa4CcHak
- Cursor self-summarization blog: https://cursor.com/blog/self-summarization
- Cursor third era blog: https://cursor.com/blog/third-era
- Cursor kernel blog: https://cursor.com/blog/kernels
- NVFP4 pretraining: https://arxiv.org/abs/2509.25149

希望这版讲得更清楚。核心就是：拿开源 base → 灌代码知识 → 在真实环境里做 RL → 用私有 benchmark 迭代。每一步的工程细节决定了能不能跑通。

---

# Composer 2 Technical Report 深度解读

Andrej，这篇 paper 我读了三遍，第一次看主线，第二次抠 RL 算法和 MoE infra 的工程细节，第三次重点对照 CursorBench 的设计哲学和 R1-zero 系列工作的差异。下面按"为什么这么设计 → 具体怎么做 → 工程上踩了什么坑"的顺序展开，尽量把直觉讲透。

---

## 1. 战略定位：为什么 Cursor 要自己训练 Composer 2

这篇 report 的核心主张其实可以用一句话概括：**把 agentic coding 当作一个独立的 RL domain 来做，而不是把 coding 当作通用 LLM 的一个 downstream 任务**。这个 framing 决定了后面所有的工程取舍。

几个观察点：

- **Base model 是 Kimi K2.5**（1.04T params / 32B active MoE），不是自研 base。Cursor 的核心 IP 在 post-training pipeline 和 harness，不在 pretraining。这跟 OpenAI/Claude 的垂直一体化路径完全不同。
- **训练分两阶段**：continued pretraining（强化 code 知识）+ asynchronous RL（强化 end-to-end agent 能力）。两阶段之间的 bridge 是 paper 里 Figure 2 那条 cross-entropy loss vs RL reward 的相关性曲线——这条曲线是整个 pipeline 能跑通的 empirical justification，类似于"Chinchilla 的 loss-to-downstream correlation"在 agentic setting 下的版本。
- **CursorBench 是整套系统的 ground truth**。public benchmark（SWE-bench Multilingual、Terminal-Bench）只是 sanity check，真正迭代靠 CursorBench。

我个人觉得这篇 report 最大的 contribution 不在 model，而在 **"如何把 RL scaling 在 agentic domain 跑稳"** 这套 recipe + infrastructure，特别是 Section 6.2 那段 Anyrun + Router Replay + Delta Compression 的组合拳。

参考：Cursor 自己关于 self-summarization 的 blog（https://cursor.com/blog/self-summarization）和 third era of software（https://cursor.com/blog/third-era）能补充更多 product-side 的 motivation。

---

## 2. Continued Pretraining：知识注入 + MTP 头

### 2.1 三阶段设计

- **Phase 1**：32k context length 上做大量 compute，code-dominated data mix。
- **Phase 2**：long-context extension 到 256k。短阶段。
- **Phase 3**：targeted SFT on coding tasks。更短。

这种"主预算 → 长上下文 → 短 SFT 收尾"的 stage 设计，和 Llama 3、Qwen 2.5 系列 report 里的 long-context extension recipe 几乎是一致的，没什么惊喜。真正有意思的是 paper 在 Figure 2 left 上做的 controlled experiment：拿 Qwen3-Coder-30B-A3B 做三个 log-spaced compute level 的 continued pretraining，每个 checkpoint 走相同的 SFT + RL，看 final loss vs RL reward。

**这条曲线的物理含义**：cross-entropy loss 每降一个量级，RL reward 几乎是 log-linear 改善。这说明 RL 阶段的"天花板"在很大程度上由 pretraining 阶段的 latent knowledge 决定。直觉上可以理解为——RL 是在 latent solution manifold 上做 sharpening，pretraining 决定 manifold 的覆盖度，coverage 不够的解空间，RL 再怎么 sharpen 也无解。

### 2.2 Multi-Token Prediction (MTP)

这部分是 DeepSeek-V3 的 MTP 路径的 cursor-flavored 改写。值得展开讲：

- **架构动机**：speculative decoding 需要一个 draft model，而把 draft model 参数化为"主 LM head 之外的额外预测头"可以让 draft 和 verify 共享大部分 backbone，省存储省 KV cache。
- **训练目标**：self-distillation，即让 MTP head 的输出 logit 分布去逼近主 LM head 在每个位置的 logit 分布。
  
  这里有个细节很关键：**self-distillation 的 target 不是 ground-truth next token 的 one-hot，而是主 head 的 softened logit distribution**。这样做的好处是 MTP head 学的是主 head 的"行为"，而不是数据本身的"硬标签"，可以让 speculative decoding 的 accept rate 更高，因为 draft 分布和 target 分布同源。

- **初始化时机**：MTP head 从 continued pretraining **中段 checkpoint** 开始训，不是从 base 开始。这个设计的直觉是——如果从 base 开始，主 head 还在剧烈变化，MTP head 一直追着移动的 target 跑，收敛慢；从中段切出来训，主 head 已经相对稳定，MTP head 学起来更稳。
- **后两阶段联合训练**：long-context extension 和 SFT 阶段把 MTP head 一起带上，让它跟着主 head 一起适配 256k context 和 coding SFT data。

参考 DeepSeek-V3 的 MTP 原始设计：https://arxiv.org/abs/2412.19437

### 2.3 MXFP8 训练精度

Composer 2 在 NVIDIA B300 上用 MXFP8 训练。MXFP8（Microscaling FP8）是 OCP 标准（https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf），与普通 FP8 的区别是——**per-32-element block scaling**，而不是 per-tensor 或 per-row scaling。这种粒度让 dynamic range 在 tensor 内部有更多自由度，对 MoE 这种 expert 输出分布差异大的场景尤其重要。

---

## 3. RL 阶段：算法层面的细节

这是 paper 信息密度最高的部分，我把它拆成几个独立的设计决策来分析。

### 3.1 Policy Gradient 基本形式

设 prompt 为 $x$，rollout 为 $\tau = (a_1, y_1, a_2, y_2, \dots, a_T, y_T)$，每条 rollout 由 policy $\pi_\theta$ 采样得到，group size 为 $G$（每个 prompt 采 $G$ 条 rollout）。Reward 是 binary-ish（correctness）+ 各种 auxiliary reward。

policy gradient 的基本形式是：
$$\nabla_\theta J = \mathbb{E}_{x, \tau \sim \pi_\theta}\left[ A(\tau | x) \cdot \nabla_\theta \log \pi_\theta(\tau | x) \right]$$

GRPO 的核心 trick 是用 group-relative advantage：$A_i = R_i - \mathrm{mean}(R_{1..G})$，去掉了 critic。

Composer 2 在此基础上做了几个**减法**：

#### 减法 1：移除 length standardization

原版 GRPO 会把 reward 除以 sequence length 做归一化（"length normalization"）。Dr. GRPO（Liu et al. 2025, https://arxiv.org/abs/2503.20783）指出这会引入 length bias——长短 rollout 在梯度上被等价对待，但实际上长 rollout 携带的信息密度不同。Composer 2 跟随 Dr. GRPO 直接去掉这一项。

**直觉**：长 reasoning chain 不应该被"自动打折"。一个 5000 token 的正确解和一个 500 token 的正确解，前者在某些任务上反而是"想清楚了再答"的体现，length normalization 会惩罚它。

#### 减法 2：不除 group std

原版 GRPO 还会除 group reward 的标准差。Composer 2 不做。理由很 sharp：

> "it results in the degenerate case where small behavioral differences get massively upweighted within a group where every rollout achieves equal correctness."

**直觉**：如果 group 内 8 条 rollout 全部正确（reward = 1），std ≈ 0，除以 std 会把 advantage 炸到天上去，相当于把"微小的不重要的行为差异"放大成巨大梯度信号，让模型去 optimize 噪声。这种 degenerate case 在 coding RL 上特别常见，因为代码 correctness 是 binary 的，全对或全错经常发生。

#### 减法 3：不 mask overlong rollouts

DAPO（Yu et al. 2025, https://openreview.net/forum?id=2a36EMSSTp）建议把超过 max sequence length 的 rollout mask 掉，避免模型学到"答到一半被截断还以为对了"的错误信号。Composer 2 实验下来发现小 scale 上没用，加上 self-summarization 已经天然限制了 overlong 的发生频率，干脆不做。

这个决策的潜台词是——**self-summarization 是 overlong 问题的 structural solution，而不是用 mask 这种 reward hacking-prone 的 post-hoc patch**。这个 framing 我觉得很重要。

### 3.2 KL Estimator 的选择

这部分 paper 写得很漂亮，应该单独拎出来讲。

KL divergence 的标准定义：
$$\mathrm{KL}(q \| p) = \mathbb{E}_{x \sim q}\left[-\log \frac{p(x)}{q(x)}\right]$$

记 $r(x) = p(x)/q(x)$，三个常见 estimator：

| Estimator | 公式 | 偏差 | 方差行为 |
|-----------|------|------|----------|
| $k_1$ | $-\log r$ | 无偏 | 方差随 $r$ 偏离 1 平滑增长 |
| $k_2$ | $(r - 1)/2$（近似） | 有偏 | 方差小 |
| $k_3$ | $(r - 1) - \log r$ | 无偏 | $r \to 1$ 方差趋于 0，$r$ 远离 1 方差爆炸 |

Amini et al. 2025（https://openreview.net/forum?id=um9kHMof0c）的 Figure 1 展示了 $k_3$ 在 $p, q$ 距离大时方差爆炸的现象。Composer 2 选 $k_1$。

**直觉**：在 RL 训练中，特别是在高度 asynchronous regime 下（trainer 和 inference worker 的 policy 版本不一致），$r$ 经常远离 1。$k_3$ 会让某些 sample 的梯度贡献被无界放大，训练不稳定。$k_1$ 的方差虽然整体比 $k_3$ 高，但是 bounded，gradient 不会因为个别 sample 炸掉。

这也是为什么很多开源 RLHF 实现训着训着崩了——默认用 $k_3$，policy 跑偏之后方差爆炸导致梯度雪崩。

Schulman 的 blog post（https://openai.com/research/approximating-kl-divergence）是这块的 canonical reference。

### 3.3 Router Replay for MoE

这是 Composer 2 infra 最有意思的一环。

**问题**：MoE 模型在 inference engine 和 trainer 上跑 forward pass，由于浮点累加顺序、quantization 路径等差异，router 选的 expert 可能不一样。如果 trainer 算 log-prob 时用的是 expert A，但 inference 采样时是 expert B，policy gradient 就在优化一个"错误分布"的对数概率，引入系统噪声。

**解决方案**：inference 时返回每个 token 在每个 MoE layer 上选的 expert indices，trainer forward pass 时强制 router 用这些 indices。但 router 的 gating score 仍然正常计算，gradient 仍然能流过去。

**Composer 2 的扩展**：基础 router replay 仍然不够，因为如果 inference 选了一个 gating score 极低的 expert（数值噪声导致的"运气"选择），trainer 强制 replay 会让 trainer 跑出非常奇怪的 activation 分布。Composer 2 加了一个 **plausibility threshold**——只 replay gating score 高于某个阈值的 expert，其他用 router 自己的 top-k 候选补上。

**直觉**：这个 threshold 等于在说"信任 inference 的 router 选择，但只信任那些 inference 自己也觉得合理的选择"。这降低了 trainer 和 inference 之间的 p99 numerics mismatch。

参考 Ma et al. 2025 关于 MoE RL router stability 的工作：https://arxiv.org/abs/2510.11370

### 3.4 Self-Summarization

Composer 1.5 就引入了，Composer 2 继续用。机制：

- 一个 rollout 不是单条 prompt→response，而是多段 generation 由 summary 串联。
- summary 由模型自己生成，不是 prompt-based compaction（外部 model 总结）。
- final reward 复制到所有 token 上（包括 summary token）。
- KV cache 在 summary 处可以重置，长 horizon 任务不需要保留全部 KV。

**为什么 reward 复制到所有 token 是合理的**：summary 是 rollout 内部的"内部 representation"，它本身没有独立的 ground-truth reward，但它的质量直接决定了后续 generation 能否成功。把 terminal reward 复制回去等价于做 credit assignment——好 summary 被强化，坏 summary（丢信息）被弱化。

**与 prompt-based compaction 的对比**：prompt-based compaction 需要额外 token 预算做总结，且 summary 质量无法被 reward 信号 refine。Self-summarization 让 summary 也成为 policy 的一部分，被 RL 直接优化。这在 long-horizon 任务上 token 效率高一个数量级。

### 3.5 Nonlinear Length Penalty

这个公式值得逐字拆解：

$$C_{\text{length}\{k, q\}}(x) = \frac{(1 + kx)^{1-q} - 1}{k(1-q)}$$

变量含义：
- $x$：weighted combination of (thinking tokens, tool calling tokens, tool output tokens, final message tokens, # tool calls, # turns)。是一个综合的"effort"度量。
- $k$：曲率参数，控制函数增长速度。$k$ 越大，相同 $x$ 下 cost 越早进入饱和区。
- $q$：凹度参数，决定函数凹的程度。

几个极限情况帮助 build intuition：

- $q = 0$：$C = x$，纯线性。没有"轻松任务该快"的偏好。
- $q \to 1$：用 L'Hôpital 取极限得到 $C = \frac{\log(1+kx)}{k}$，对数增长。强凹，长任务被严重打折。
- $q \in (0, 1)$：温和凹，介于线性和对数之间。这是实际使用的 regime。
- $q > 1$：函数 eventually 趋于 0，cost 反而下降，不合理。
- $q < 0$：凸函数，长任务 cost 增长更快，与目标相反。

**为什么 concave down + increasing 是对的**：在 easy task 上（$x$ 小），函数斜率大，每多花一点 effort 被惩罚得很重，逼模型快速答完；在 hard task 上（$x$ 大），斜率变小，effort 的 marginal cost 趋于 0，模型可以放心想很久。这跟人解决问题的行为模式一致——简单问题不应该纠结，复杂问题应该深思。

**实际效果**（paper Section 4.2 提到）：模型学到在简单任务上做 parallel tool calls 来摊薄 #turns 的 cost。

参考 Composer 1.5 的 self-summarization blog：https://cursor.com/blog/self-summarization

---

## 4. CursorBench：为什么 public benchmark 不够

### 4.1 四大 misalignment

Paper Section 5 列了 public benchmark 跟真实 dev workflow 的四个 mismatch：

1. **Domain mismatch**：SWE-bench 偏 bug-fixing，Terminal-Bench 偏 abstract puzzle（比如 compute chess moves），都不是 typical SE operation。
2. **Prompt over-specification**：public benchmark 通常给出明确指令，但真实 dev request 是 underspecified，可能有多种 valid architecture。
3. **Data contamination**：public benchmark 从 OSS repo scrape，容易泄漏到训练集。OpenAI 已经因为这个暂停 SWE-bench Verified 报告（https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/）。
4. **Narrow evaluation scope**：public benchmark 只看 functional correctness，忽略 code quality、latency、cost、interactive behavior。

### 4.2 CursorBench 的量化对比

Figure 7 给了硬数据：

| Benchmark | Median lines changed | Median prompt length (chars) |
|-----------|----------------------|-----------------------------|
| SWE-bench Verified | 7-10 | 1185-3055 |
| SWE-bench Multilingual | 7-10 | 1185-3055 |
| CursorBench | 181 | 390 |

CursorBench 的 task 更**短 prompt + 更大 code change**，这恰好反映了真实 SE 的特征——developer 给你一句模糊的 bug report，你要 cross-reference production logs、源码、observability 才能定位问题，最后改 200 行。

### 4.3 CursorBench 的迭代

Figure 9 显示 CursorBench 已经迭代到第 3 版，每版 task 规模（files changed、lines changed）几乎翻倍。这种**主动让 benchmark 进化**的设计很关键——avoid saturation。

### 4.4 两个 example task

- **Figure 8**：esbuild 0.20.2 的 transpilation bug，retry loop 里 `var`-scoped error state 在 attempt 之间没被 reset。需要从 terse bug report + production logs（含 red herring warnings）推断 root cause。
- **Figure 12**：streaming prefix detection bug。模型需要写一个 heuristic detector 扫 954 个 chat response JSON，找出 prefix-chain growing 的失败模式，还要发现"interleave stutter"变体。

这两个 task 的特征：**模糊的 symptom → 模型自己构造检测算法 → tune 阈值 → 输出量化结果**。这不是 SWE-bench 那种"读 issue → 改 5 行代码 → 跑测试"的范式。

---

## 5. Infrastructure：训练 + RL 的工程架构

### 5.1 Parallelism 设计

Composer 2 的并行策略有几个关键变化：

**变化 1**：从 TP 主导改为 **CP (Context Parallelism) 主导**。

- TP 的劣势：把 hidden dimension 切碎成 skinny local matmul，compute efficiency 低。
- CP 的优势：保留 full hidden dim，只在 attention 那一层做 sequence-wise 切分，通信开销小。

**MLA + CP 的实现细节**（paper Section 6.1）：

- 每个 CP rank 先在本地算 KV latent vector（MLA 的低秩 latent，对应 DeepSeek-V2/V3 的设计）。
- all-gather latent vectors 跨 CP ranks。
- 各 rank 独立做 KV projection（虽然有冗余，但 projection 计算量小，省掉了 projection 后的大 tensor 的 all-gather 通信）。
- CP 通信和 Q projection 的计算完全 overlap。

**Load imbalance 解决**（Liu et al. 2023 的 trick, https://openreview.net/forum?id=WsRHpHH4s0）：

- Naive CP 会让前 chunk 处理的 token 少（因为它们 attend 的过去 token 少），后 chunk 处理多。
- 解决方案：把 sequence 切成 $2 \times \text{CP}$ 个 chunks，第 $i$ 个 rank 处理 chunk $i$ 和 chunk $2 \times \text{CP} - 1 - i$。这样每个 rank 的工作量大致相等（前面的 chunk 少 + 后面的 chunk 多）。

**变化 2**：EP 与 TP 解耦。原版 MoE 设计 EP 复用 TP rank group，限制了 EP 的 scaling 上限。Composer 2 让 EP 独立从 DP + CP capacity 出来，可以 EP=8 + CP=2（pretraining）或 EP=8 + CP=8（RL）。

**DeepEP**（DeepSeek 开源的高 throughput token dispatch/combine 库，https://github.com/deepseek-ai/DeepEP）做 token dispatch，默认用 20 个 SM，留余量给 compute kernel 并发。token 在 dispatch 前先 quantize 到 MXFP8，combine 时回 BF16 提精度。

### 5.2 自定义 Kernel：NVFP4 改造

这是 paper 里最硬核的工程细节。

**原版 NVFP4**（NVIDIA 标准, https://arxiv.org/abs/2509.25149）：
- Value: FP4E2M1（4 bit，2 exp + 1 mantissa + 1 sign）
- Scale: FP32 per-tensor

**Composer 2 的 NVFP4 变体**（MoE forward）：
- Value: FP4E2M1（不变）
- Block scale: FP8E4M3，block size = 16
- Token scale: FP32 per-token

**为什么改**：

1. **per-tensor scale 是 batch-variant 的**：scale 是整个 tensor 的 max-abs 除以 FP4 表示范围，每个 batch 的 max-abs 不同，scale 不同。这让训练在不同 batch 间数值不可复现，RL 训练直接发散。
2. **per-tensor scale 有信息泄漏**：tensor-level max-abs 包含了"未来 token"的信息（因为 max 是全 tensor 算的），反向传播时 past token 的梯度会被 future token 的统计量"污染"。

**直觉**：per-tensor quantization 在 inference 上 OK（一次性 forward），在 training 上是 poison。per-token + per-block 双层 scaling 把 quantization 的 granularity 降到 token 级，每个 token 的 scale 只依赖自己，没有 cross-token 信息流。

**Backward pass 用 MXFP8**：

- Forward 必须用低精度（NVFP4）是为了让 trainer 数值上跟 inference engine 对齐，inference 也要低精度跑得快。
- Backward 只在 trainer 上跑，不影响 inference，可以放宽精度到 MXFP8（FP8E4M3 + FP8E8M0 per-32 block scale）换稳定性。

**IEEE-compliant vs fast-approx 数学**：

- NVFP4 quantization 必须用 IEEE-compliant float arithmetic（如 `__fdiv_rn`），用 fast-approx（如 `__fdividef`）会在 ~100 RL steps 后发散。
- MXFP8 quantization 用 fast-approx 没事，所以选 fast-approx 换性能。

这是个非常 subtle 的发现——FP4 的 dynamic range 太窄，任何 IEEE 标准外的近似误差都会被量化过程放大成 systematic bias，积累几百步后训练崩。FP8 的 range 够宽，近似误差被天然吸收。

参考 Cursor 的 kernel blog：https://cursor.com/blog/kernels

### 5.3 RL Infrastructure：异步 + 解耦

四个解耦服务：

- **Training**：Ray + PyTorch，全异步。centralized reconciler 做 slot-based sample lifecycle 管理。所有服务用 future 模式，upstream dependency ready 就触发 eager execution。Ray object store 自动 spill 到 NVMe。
- **Environments**：Anyrun 平台，跑 untrusted code。每个 pod 是一个 Firecracker VM，能跑完整 dev environment（包括 browser + GUI for computer use）。Anyrun 支持 fork + snapshot（filesystem 和 memory level），让 mid-trajectory rollout checkpoint 成为可能。
- **Inference**：和 Fireworks AI 合作。每 step 同步权重：delta compression（每 rank 缓存上次上传，只传 diff），1T 模型 diff 压缩到几 GB，全 sharded 上传下载打满 egress 带宽，跨 region 通过 S3 bucket 不需要直接 connectivity。
- **Evaluations**：pin production backend + Cursor client 的版本，跨 region weight sync 到 eval cluster。

**关键架构决策**：

1. **fault tolerance 到 process level**：被动 + 主动健康检查，硬件故障节点标 unhealthy 但训练继续，warm standby 顶上。这避免了"一个 GPU 挂了整个 job 重启"的传统痛点。
2. **rollout-level checkpointing**：long-running coding rollout replay 很贵，所以除了 step-level model checkpoint，还要做 rollout-level（codebase env 的 memory snapshot）和 group-level（advantage-tagged sequence 写到 NFS）。Job restart 时 scheduler 能决定是 dispatch 新 work 还是直接 load ready groups。
3. **Anygress egress 控制**：所有 pod 出网经过 Anygress proxy，注入 trusted root CA + TCP 层 redirect，让 proxy 对 pod 透明（不依赖 env var）。这模拟了真实 dev environment 的网络行为。
4. **Tool 一致性**：用 production Cursor backend 的 shadow deployment 同时给 dataset prep 和 rollout 用，保证 train/test harness 完全一致。某些 tool 在训练时 stricter（强 argument 检查）或被移除（improve steerability）。

### 5.4 Weight Sync 的工程优化

展开讲一下 delta compression 这块，因为它对 world-scale distributed RL 是关键：

- 每个 trainer rank 维护"上次上传的 weight shard"本地缓存。
- 每 step 计算新 shard vs 旧 shard 的 diff（typically 稀疏，因为 RL update 小）。
- Diff 上传到 S3 bucket，全 sharded 跨 ranks，打满 trainer cluster 的 egress 带宽。
- 每个 inference cluster（跨地理 region）独立从 S3 拉 delta chain，reconstruct 完整 weight，不需要 direct connectivity 到 trainer。
- Upload/download/hotload signaling 全 pipeline 在 background workers 跑，不阻塞 training。

Composer 2 实际部署时 inference 跨 US + Europe，trainer 在另一个 region，完全靠 S3 做 decoupling。这种"commodity cloud storage 做 RL weight bus"的设计很 clean。

---

## 6. Results 分析

### 6.1 CursorBench 主结果

Table 1 的关键数字：

| Model | CursorBench | SWE-bench Multi. | Terminal-Bench |
|-------|-------------|------------------|----------------|
| Composer 2 | 61.3 | 73.7 | 61.7 |
| Composer 1.5 | 44.2 | 65.9 | 47.9 |
| Composer 1 | 38.0 | 56.9 | 40.0 |
| Opus 4.6 High | 58.2 | 75.8/77.8 | 58.0/65.4 |
| GPT-5.4 | 63.9 | 76.8 | 66.5/75.1 |
| Kimi K2.5 (base) | 36.0 | 65.1/73.0 | 47.3/50.8 |

几个观察：

1. **Composer 2 vs Kimi K2.5**：CursorBench 从 36.0 → 61.3（+70%），但 SWE-bench Multi 只从 65.1 → 73.7（+13%），Terminal-Bench 从 47.3 → 61.7（+30%）。这暗示 CursorBench 的"灵敏度"远高于 public benchmark——base model 已经在 public benchmark 上接近饱和，只能在 CursorBench 上拉开差距。
2. **vs GPT-5.4**：CursorBench 上 GPT-5.4 是 63.9，Composer 2 是 61.3，差距 2.6 个点。但 GPT-5.4 的成本远高于 Composer 2（Figure 11 显示 Composer 2 在 cost-accuracy Pareto frontier 上）。这就是 domain specialization 的价值——用更小更便宜的模型逼近 frontier。
3. **vs Anthropic Opus 4.6 High**：CursorBench 上 Composer 2 (61.3) > Opus 4.6 (58.2)，但在 Terminal-Bench 上 Opus 4.6 self-reported 是 65.4，Composer 2 是 61.7。这种 cross-benchmark 不一致正好印证了 paper 关于 domain mismatch 的论点。

### 6.2 Best-of-K vs Average Reward

Figure 5 的关键 claim：**RL 训练让 average 和 best-of-K 同步上升，没有 trade-off**。

这个结果 paper 自己说"notable"，因为近期一系列 paper（Yue et al. 2025 oral at NeurIPS, https://openreview.net/forum?id=4OsgYD7em5；Tajwar et al. 2026, https://arxiv.org/abs/2602.02710）claim RL 主要在做"reweighting existing correct trajectories"，导致 best-of-K 不动甚至下降。

Composer 2 的反例说明：**在 agentic coding domain，RL 仍然在扩展正确解的 coverage**。可能的原因：

- Coding task 的解空间结构跟 math reasoning 不同，coding 解之间不是"小扰动"关系，而是结构性不同的 approach。RL 在 approach 之间做有效探索，扩大了 reachable solution set。
- Self-summarization 让模型在 long-horizon 任务上能学到"如何高效利用 context"，这种 skill 不在 base model 的 trajectory pool 里。
- Tool use 引入了额外的 action 维度，RL 在 tool use 策略上学到新东西。

这个 claim 如果 reproducible，对 RL-on-LLM 的研究方向有重要影响。

### 6.3 Cost-Accuracy Pareto

Figure 11 是 paper 的"商业说服力"图：

- Composer 2 在 inference cost 上接近 small/low-effort variants（如 GPT-5.4 medium），但 accuracy 跟 frontier high-effort variants 持平。
- 关键是 active params 只有 32B（虽然 total 1.04T），推理 FLOPs 大约是 GPT-5.4 这种 dense 或更大 active 模型的几分之一。

这就是 specialized model 的核心 argument——**通用模型为多任务保留 capacity 是有 tax 的，specialized model 把 capacity 全部分配给单 domain**。

---

## 7. Base Model 选择：为什么是 Kimi K2.5

Appendix B 给了 base model 评估细节，值得细看：

| Model | FreshBench ↑ | State Tracking ↓ | NLL ↓ |
|-------|--------------|------------------|-------|
| DeepSeek V3.2 | 68.9% | 66 | 11.75M |
| Kimi K2.5 | 83.2% | 86 | 13.81M |
| GLM-5 | 79.2% | 92 | 14.11M |

有意思的是 **Kimi K2.5 的 NLL 不是最低的**（DeepSeek V3.2 的 11.75M 更低），但 Cursor 还是选了 Kimi K2.5。理由 paper 含糊带过"general strong performance + infrastructure efficiency"。

我的推测：

1. **K2.5 是 MoE**（1.04T / 32B active），inference 成本远低于 dense 的 DeepSeek V3.2。Composer 是要做 product 部署的，inference cost 直接决定商业可行性。
2. **FreshBench 上 K2.5 (83.2) 显著高于 DeepSeek V3.2 (68.9)**。FreshBench 测的是"recent library 知识 / 需要读 source code 才能 answer 的问题"，这跟 agentic coding 的实际 workload 更接近。NLL 测的是 monorepo perplexity，更像"代码 fluency"，但 fluency 高不一定代表 agentic 能力强。
3. **State tracking 上 K2.5 (86) 比 DeepSeek V3.2 (66) 差**，但可能 RL 阶段可以补救。

**关键 insight**：Cursor 不用 agent benchmark 选 base model，因为"agentic capability 在 RL 阶段会剧变"。用更"primitive"的 benchmark（knowledge、state tracking、perplexity）选 base，因为它们更 predictive of post-RL ceiling。这个 framing 我觉得对所有做 post-training 的人都有启发。

---

## 8. 几个我没看到 paper 明说但值得思考的问题

1. **RL data 的来源**：paper 说 task distribution 来自"most common use cases"（Figure 3），但没说具体怎么 sample。Cursor 有海量 user session data，但 session → RL task 的 transformation pipeline 是怎么做的？reward labeling 怎么做？这块是 product-driven RL 的核心 IP，paper 显然不会写透。
2. **RL compute 预算**：paper 没说 RL 阶段跑了多少 tokens、多少 steps、多少 GPU hours。这个数字对评估"这套 recipe 的 sample efficiency"非常关键。
3. **Self-summarization 的具体 trigger**：什么时候模型决定 summarize？是基于 context window watermark，还是基于内容判断？这个 trigger policy 是 hand-coded 还是 learned？
4. **RL 中 reward hacking 的检测**：auxiliary reward 加了一堆（coding style、communication、tool call penalty），但 paper 没说怎么 detect 和 mitigate reward hacking。比如模型可能学会"写看起来 good style 但 actually wrong"的代码。
5. **Distribution shift between CursorBench iterations**：CursorBench 已经到 v3，v1 训练的模型在 v3 上还能不能 hold？paper 说"continually update"，但跨版本 comparability 怎么保证？

---

## 9. 我的整体评价

这篇 report 是 2026 年 agentic coding 领域我读过最信息密度高的一篇，几个 standout：

- **Section 6.1 的 NVFP4 改造**是教科书级别的"为什么 per-tensor quantization 在 training 上不 work"的 case study。
- **Section 4.1 的 KL estimator 分析**应该成为所有 post-training 工程师的必读。
- **Section 5 的 CursorBench 设计哲学**对整个 SE agent 评估方向有方法论价值。
- **Figure 5 的 best-of-K vs average 同步上升**结果如果 reproducible，会改写 RL-on-LLM 的理论叙事。

短板：

- RL data pipeline 完全黑盒。
- Compute budget 不公开，让 reproducibility 判断困难。
- Self-summarization 的 trigger 没讲清楚，需要看 Composer 1.5 的 blog 补充。

总的来说，paper 展示了一条"domain-specialized model through RL on top of open base"的完整工程路径，所有想做 vertical LLM 的团队都应该精读。

---

## 10. 一些相关 paper 推荐

如果你想顺着 Composer 2 的 reference 深入：

- **Dr. GRPO** (Liu et al. 2025): https://arxiv.org/abs/2503.20783 — RL 算法 bias 的系统分析。
- **DeepEP** (Zhao et al. 2025): https://github.com/deepseek-ai/DeepEP — MoE expert parallel 通信库。
- **DAPO** (Yu et al. 2025): https://openreview.net/forum?id=2a36EMSSTp — 系统级 LLM RL 工程实践。
- **ThunderKittens** (Spector et al. 2025): https://openreview.net/forum?id=0fJfVOSUra — Cursor 在用的 GPU kernel abstraction。
- **PipelineRL** (Piché et al. 2025): https://arxiv.org/abs/2509.19128 — asynchronous on-policy RL 的设计，跟 Composer 2 的 in-flight weight update 思想一致。
- **Kimi K2.5** report: https://arxiv.org/abs/2602.02276 — base model 的细节。
- **Terminal-Bench** (Merrill et al. 2026): https://openreview.net/forum?id=a7Qa4CcHak — 一个比 SWE-bench 更广的 agent benchmark。
- **Amini et al. 2025** on KL estimators: https://openreview.net/forum?id=um9kHMof0c — KL variance 分析的 canonical reference。
- **Yue et al. 2025** NeurIPS oral: https://openreview.net/forum?id=4OsgYD7em5 — "RL 不真的增加 reasoning capacity" 的反方观点，跟 Composer 2 Figure 5 形成对话。

希望这个讲解对你的 intuition building 有帮助。如果有特定 section 想再深挖（比如 NVFP4 的具体 bit-level layout，或者 Anyrun 的 fork/snapshot 实现细节），可以继续聊。
