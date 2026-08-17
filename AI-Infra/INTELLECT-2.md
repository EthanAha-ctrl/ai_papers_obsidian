---
source_pdf: INTELLECT-2.pdf
paper_sha256: aa46389841863626c9c136adb3bac8915165b2d95bfeee09c2abdeec4d95ac03
processed_at: '2026-08-05T10:03:42-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# INTELLECT-2 用人话说

## 一句话总结

一群人想证明：**不用几万张H100堆在一个机房里，而是靠全世界志愿者凑来的乱七八糟的显卡，也能训练出一个会推理的32B大模型**。他们做到了，还把所有代码和模型都开源了。

---

## 为什么要做这件事

先说背景。现在训练reasoning model（就是像o1、R1那种会"想一想再回答"的模型），标准做法是啥？

你得有一个巨大的data center，里面几千张H100用InfiniBand连着，跑RL training。这个门槛极高，全世界能玩得起的公司不超过20家。

但Prime Intellect这帮人观察到一个有意思的现象：

**RL training和pretraining不一样**。Pretraining的时候，所有GPU要时刻同步gradient，communication是killer。但RL training呢？

- 生成rollout（让model做题产生reasoning trace）这个过程，每个GPU自己干自己的，**完全不需要互相通信**
- 只有更新weight这一步需要通信，但这一步的compute占比很小

他们实测发现，**training和inference的FLOPs比例大概是1:4.5**。也就是说，绝大部分算力花在"让model做题"上，而不是"更新model参数"上。

这就引出一个疯狂的想法：**既然rollout generation这么parallelizable，那能不能把这部分丢给全世界志愿者来做？**

就像 Folding@Home 那种模式——你有个显卡闲置着，跑一段推理，上传结果，就为训练GPT-level的模型贡献了一份力。

---

## 整个系统怎么运转的

想象一个工厂流水线，分三个角色：

### 角色1：Inference Worker（全球志愿者）

你可能家里有张3090，或者公司有台闲置的A100。你装了他们的软件，软件会：

1. 从relay server下载最新的model weights（32B bfloat16，约62GB）
2. 收到一批题目（数学题、编程题）
3. 让model做这些题，生成reasoning trace（比如"让我想想...这道题要先用积分..."这样的思维链）
4. 生成的时候，每隔32个token偷偷存一个hidden state的hash（这是为了后面验证你没作弊）
5. 把结果打包成Parquet文件上传

关键：**你用的硬件可以很烂，速度可以很慢，随时可以掉线**。系统不在乎，因为还有几百个其他worker在同时干活。

### 角色2：TOPLOC Validator（受信任的验证节点）

你上传了rollout，但问题来了——**我怎么知道你没作弊？** 你可能：

- 用了个更小的model生成答案，骗我说这是32B model做的
- 用了quantized model省电
- 故意挑简单题做
- 提前截断sequence省compute

Validator要抓这些作弊。怎么抓？

**Computation check**：Validator拿你提交的token sequence，用真正的32B model跑一遍prefill（就是forward pass但不用generate），重建hidden state。如果你当时用的是正确model，hidden state应该和你提交的hash匹配。如果你偷偷用了quantized model或者别的model，hidden state对不上，你就被抓了。

这个验证过程**比生成快100倍**，因为：
- Prefill不用autoregressive decode，一个forward pass就完事
- Validator不用验证所有batch，random sampling验证一部分就行
- 你不知道哪些会被验证，所以必须全部honest

**Sampling check**：
- 看sequence是不是正常结束（要么到了max length，要么出了EOS token）
- 如果EOS结束，检查EOS probability > 0.1，防止你用低概率EOS提前截断
- 检查logit distribution的形状，proper sampling应该像exponential decay。如果你用了小model生成+大model prefill骗TOPLOC，distribution会bimodal

**Sanity check**：
- seed = node_address · step + submission_count，确保你不能cherry-pick easy题
- 检查所有scalar值在合理范围内
- 检查Parquet格式正确

如果你作弊被抓，你的contribution被reject，你被踢出compute pool。

### 角色3：Training Node（受信任的中央训练集群）

这是少数几张H100组成的集群，负责：
1. 收集所有validated rollouts
2. 用GRPO算法计算gradient，更新policy
3. 把新weights通过SHARDCAST广播给所有inference workers

这个centralized的部分compute占比很小（约1/5.5），但需要fast interconnect，所以没法decentralize。

---

## 三个核心技术创新

### 创新一：Asynchronous RL——让通信延迟消失

传统RL training是这样的：
```
Step 1: 用当前policy生成rollout → 训练更新policy → 用新policy生成rollout → ...
```
必须等上一步完成才能开始下一步，因为要用最新policy生成rollout。

但在decentralized setup里，新policy weights广播到全球所有worker需要14分钟。如果等所有worker都拿到新weights才开始生成rollout，效率太低了。

**解决方案：不等了，用旧policy生成rollout也行**

具体来说，worker可能用的是2-4步之前的policy weights来生成rollout。这会引入一些off-policy问题，但ablation实验（Figure 7）证明：**即使off-policy 4步，reward trajectory和synchronous baseline几乎一模一样**。

为什么这个work？直觉上说，policy每步更新很小，$\frac{\pi_{\theta_{new}}}{\pi_{\theta_{old}}}$的ratio在clip范围内（$\epsilon=0.2$），importance sampling的variance不会爆炸。rollout虽然是用旧policy生成的，但仍然包含valid的learning signal——model做对了一道难题，不管是哪个版本的policy做的，这个"做对了"的信号是有用的。

这个insight很关键：**RL training对off-policy的容忍度比想象的高**。这和pretraining完全不同——pretraining的gradient必须用当前model的forward pass计算，不能off-policy。

### 创新二：SHARDCAST——把62GB weights广播给几百个节点

问题是：training node生成新weights后，要快速分发给全球几百个inference workers。62GB的checkpoint，如果training node自己上传，uplink带宽被打满，成为瓶颈。

**解决方案：类似CDN的tree-topology relay网络**

Training node把checkpoint sharded，上传到几个relay server。Relay server之间互相转发形成tree结构。Inference worker从最近的relay下载。

几个关键trick：

**Pipelined streaming**：不用等整个62GB checkpoint上传到relay才开始下载。Shard 1上传到relay的同时，worker就可以开始下载shard 1，同时shard 2正在上传。这大幅减少end-to-end latency。

**Probabilistic relay selection**：每个worker不是greedily选最快的relay（那会contention），而是按expected throughput的比例随机采样：

$$\text{expected throughput} \propto \text{success rate} \times \text{bandwidth}$$

用EMA持续更新估计值。这个比greedy策略更好，因为可以同时利用多个relay的multiple connections。

**SHA-256 checksum验证**：worker下载完checkpoint，计算checksum和training node广播的reference对比。不匹配就discard重新下载下一个版本。防止download corruption。

**为什么不P2P？** 他们考虑过P2P weight transfer（worker之间互传），但最终没做。原因是P2P会暴露worker IP给peers，增加DoS attack风险。Design trade-off：centralized relay更安全但less decentralized。

### 创新三：TOPLOC——信任不可信的inference

这是最technically interesting的部分。核心问题：**怎么相信一个untrusted worker真的用指定model weights生成了response？**

**Naive方案的问题**：让validator把worker生成的sequence重新generate一遍对比？太慢了，等于算两遍。

**TOPLOC的方案**：Locality-sensitive hashing

Inference的时候，worker在logits processor里装个hook，每32个token捕获一次final hidden state。然后对这些activations做hash，存成commitment。这个hash过程在CPU上async执行，和GPU forward pass并行，overhead只有~1%。

Validator收到commitment后，用真正的32B model对worker生成的token sequence做prefill（一次forward pass），重建hidden state，验证hash是否匹配。

**为什么这能work？** LSH的特性是相似input hash到相同bucket。如果worker用了quantized model或不同model，hidden state会不同，hash不匹配。

**对GPU non-determinism的robustness**：不同GPU上floating point运算不完全deterministic。TOPLOC的hash scheme通过适当quantization和tolerance来处理这个。这个细节在他们的TOPLOC paper里。

---

## Training Recipe的关键改进

### Reward设计

$$r_{\text{total}}(y, l_{\text{target}}) = r_{\text{task}}(y) - \alpha \cdot |l_{\text{target}} - l_y|$$

两部分：
- $r_{\text{task}}$：题做对了给1，错了给0。Math用symbolic verification，code用unit test
- Length penalty：让model学会按prompt里指定的thinking token数来控制推理长度。$l_{\text{target}}$从{2000, 4000, 6000, 8000, 10000}里random sample，prompt里写"Think for 4000 tokens before giving a response"

$\alpha = 0.0003$，很小，避免length penalty dominate。

### Two-Sided GRPO Clipping——解决training instability

这个改动很小但很重要。先看原版GRPO：

```python
ratio = pi_new(token) / pi_old(token)
loss = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

当advantage > 0（好的rollout）：ratio被clip在[1-eps, 1+eps]，限制了policy update的幅度。

当advantage < 0（坏的rollout）：`min`操作会选择更小的那个值。如果ratio > 1+eps，`clip`会给1+eps，然后`min(ratio*A, 1+eps*A)`当A<0时会选`ratio*A`（因为更负=更小）。这意味着**ratio没有upper bound**。

实际中ratio可能爆到100+，导致huge gradient spike，training崩溃。

**Two-sided clipping的fix**：

```python
ratio = min(pi_new(token) / pi_old(token), delta)  # delta = 4
loss = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

加了$\delta=4$作为upper bound。仍然允许large update away from bad rollouts，但避免ratio爆炸。

$\delta > 1 + \epsilon$保证比positive advantage的clip更宽松，但bounded。

这个fix看似简单，但他们发现"concurrent work [25]"（MiniMax等）也独立发现了同样问题。说明这是large-scale GRPO training的common pain point。

### Data Filtering的双重策略

**Offline filtering**：训练前用DeepSeek-R1-Distill-Qwen-7B算每道题的pass@8。保留pass@8 ∈ [12.5%, 50%]的题。

- 太easy（pass@8 > 50%）：model总能做对，组内reward全1，advantage全0，no signal
- 太hard（pass@8 < 12.5%）：model总做不对，组内reward全0，advantage全0，no signal

只有"model有时能做对有时做不对"的题才提供learning signal。这个filtering至关重要——Figure 8显示，unfiltered dataset训练reward停滞，filtered后reward显著上升。

**Online filtering**：即使offline filtered了，某个batch里所有sample可能还是全对或全错（advantage全0）。所以持续sampling直到batch里所有sample都有non-zero advantages才训练。

副作用：需要更多inference compute per training step。但在decentralized setup里这是feature不是bug——更多inference work意味着能利用更多volunteer workers。

---

## Training Instability的现象学

这部分信息量很大，是32B scale RL training的"field notes"。

### 现象1：Gradient Norm Escalation

随着training推进，gradient norm持续增长，即使没有immediate spike。7B模型相对stable，32B模型明显unstable。

这个现象在large-scale pretraining中也有报告（Wortsman et al. 2023），但在RL training中更severe。

**Mitigation**：aggressive gradient clipping，threshold低到0.05-0.1。这不消除问题，但delay collapse，extend viable training period。

### 现象2：Clip Ratio Escalation

Token probability clip ratio持续上升，与gradient norm growth相关。Clip ratio tracks consecutive optimizer steps之间logits的difference。Logits drift越大，clip ratio越高。

### 现象3：Entropy Resurgence

这是最intriguing的pattern：
1. Training初期entropy下降（model变confident）
2. 约150步后entropy开始**回升**
3. Entropy回升之后通常跟着catastrophic collapse

这个pattern是一个early warning signal。Entropy回升可能表示policy开始explore"wrong direction"。

增加KL penalty weight能delay collapse，但slow learning。Stability-efficiency trade-off。

### 现象4：QwQ比R1-Distill更难train

QwQ-32B和DeepSeek-R1-Distill-Qwen-32B都基于Qwen 2.5 pretrained model，但QwQ更unstable。

**Hypothesis**：QwQ已经经历过一轮RLVR training，这让它的loss landscape更"sharp"。已经处于某个reward plateau的model，进一步RL容易把它push off。

**Implication**：RL training的"stacking"有问题——每轮RL都让model更难进一步fine-tune stably。这对iterative RL training（先RL一遍，再RL一遍）的设计有重要影响。

### 现象5：torch.compile导致崩溃

torch.compile在training后期导致catastrophic collapse，可能是某个faulty generated kernel。他们全codebase disable torch.compile，代价是slightly increased memory。

**Practical lesson**：critical training run中，torch.compile的numerical instability可能后期才显现。如果training不稳定，试试关掉torch.compile。

---

## 实验结果

### Benchmark性能

| Model | AIME24 | AIME25 | LiveCodeBench v5 | GPQA | IFEval |
|-------|--------|--------|------------------|------|--------|
| INTELLECT-2 | **78.8** | 64.9 | **67.8** | 66.8 | 81.5 |
| QwQ-32B (base) | 76.6 | 64.8 | 66.1 | 66.3 | 83.4 |
| DeepSeek-R1 | 78.6 | **65.1** | 64.1 | **71.6** | 82.7 |

INTELLECT-2在math和coding上略超QwQ-32B，但IFEval略降（因为只train math/coding没train instruction following）。

提升幅度不大，paper自己承认：QwQ-32B已经extensively RL-trained，再在上面做RL只能marginal improvement。需要better base model（如Qwen3）或更高质量dataset。

### Compute utilization

- SHARDCAST广播：14分钟（62GB，~590 Mb/s）
- TARGET-SHORT: 22分钟完成一个rollout step
- TARGET-LONG: 29分钟完成一个rollout step
- Training-to-inference FLOPs ratio: 1:4.5

Almost perfect overlap——training node完成一步的时候，inference workers刚好生成完下一步的数据。

---

## 核心Insight和Discussion

### Test-Time Compute Scaling天然适合Decentralization

Pretraining scaling：communication-bound，需要DiLoCo等技术reduce communication。

Test-time compute scaling：inference-heavy，inference是embarrassingly parallel。

随着reasoning model的thinking length增长，inference compute进一步dominate。这opens door to训练hundreds-of-billions-parameter models on globally distributed heterogeneous compute。

### Inference-Heavy RL的scaling dynamics

Model能力提升靠training on harder samples。但harder tasks → sparser rewards → more exploration needed → more inference compute per useful learning signal。

这reshapes了decentralized RL的scaling dynamics：
- Memory constraints不再是bottleneck（inference比training memory要求低）
- 更多consumer-grade hardware可以参与
- Heterogeneity从bug变成feature

### Asynchronous RL隐藏大部分通信开销

DiLoCo通过压缩gradient减少communication，但随model size增长communication再次成为瓶颈。

Asynchronous RL的strategy不同：不是减少communication，而是**overlap communication with computation**。即使model稍微off-policy，仍然能生成有用rollout。

他们发现delay 4-5步仍然stable。这能hide：weight broadcasting + environment verification + TOPLOC validation + KL log-prob computation。

---

## 我的一些思考

### 这篇paper的真正价值

不在于algorithm novelty，而在于**infrastructure innovation enabling new research paradigm**。

1. 证明了decentralized RL at 32B scale可行
2. 提供完整open-source infrastructure stack
3. 发现并fix了large-scale RL的several instability patterns
4. 为resource-constrained research group开辟new path

### 与Web3/Crypto的潜在联系

Prime Intellect Protocol用了decentralized ledger、cryptographic signing、on-chain events。这实际上是**proof-of-useful-work**变体。未来可能有token incentives机制——你贡献compute，获得token reward。

### Limitations和future work

几个方向值得follow：
1. Qwen3 base + INTELLECT-2 recipe能否达到R1-level performance？
2. Tool use RL的decentralized training（web browsing、code interpreter进reasoning chain）
3. Model merging在RL中的effectiveness——多个model独立训练后merge
4. 更长thinking budget（100K+ tokens）的scaling behavior
5. VinePPO这种inference-heavy RL方法是否更适合decentralized

### 为什么RL比pretraining更适合decentralization

这是整个paper的philosophical核心。总结一下key reasons：

1. **RL的rollout generation是embarrassingly parallel**——每个worker独立生成rollout，不需要通信
2. **RL容忍off-policy data**——用旧policy生成的rollout仍然有learning signal
3. **RL的compute分布偏inference**——training-to-inference ratio约1:4.5
4. **RL的verification可以automated**——math correctness、unit test不需要human annotator
5. **RL的memory requirement lower**——inference比training省memory，consumer-grade GPU能跑

这五个properties合在一起，使得RL training比pretraining更适合decentralized execution。

---

## 最后

这篇paper是infrastructure-first research的典范。它没有propose新算法，而是证明了**infrastructure design本身可以成为algorithmic progress的enabler**。

当你把training拆成decoupled components，当你接受slightly off-policy data，当你build cryptographic verification for trustless compute——这些engineering decisions加在一起，产生了一个新的research paradigm。

对于想做RL training但买不起H100 cluster的研究者，这篇paper提供了一条new path。所有代码开源在GitHub，你可以fork PRIME-RL，改进recipe，贡献你自己的inference workers，甚至发起你自己的decentralized training run。

这可能是这篇paper最大的contribution——**democratizing RL training for reasoning models**。

---

参考链接汇总：
- 论文：https://huggingface.co/PrimeIntellect/INTELLECT-2
- PRIME-RL：https://github.com/PrimeIntellect-ai/prime-rl
- TOPLOC：https://arxiv.org/abs/2503.16412
- Protocol：https://github.com/PrimeIntellect-ai/protocol
- Dataset：https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1

---

# INTELLECT-2 深度技术解析

这篇paper的核心贡献是在一个**permissionless, globally distributed**的compute网络上，通过**asynchronous reinforcement learning**训练了一个32B参数的reasoning model。这个工作触及了distributed training、verifiable inference、RL stability三个方向的交叉点，非常值得深挖。

---

## 1. 整体架构与设计哲学

INTELLECT-2 的核心思想是把 RL training pipeline 拆成三个完全解耦的角色：

1. **Inference rollout workers** (trustless, heterogeneous, consumer-grade GPU)：生成 reasoning traces
2. **TOPLOC validators** (trusted)：验证 rollout 的完整性
3. **GRPO training workers** (trusted, centralized)：聚合 verified data，更新 policy，通过 SHARDCAST 广播新 weights

这个设计的核心 insight 在于 **RL 比 pretraining 更适合 decentralization**，原因有三：

- **Asynchronous RL hides communication**：rollout 使用的可以是 2+ steps 前的 policy weights，而不是最新的。这与 DiLoCo 通过压缩 gradient 来减少 communication 不同——这里是通过接受 slightly off-policy 的 data 来 hide communication latency
- **Inference dominates compute**：training-to-inference FLOPs ratio 约 1:4.5，且随着 test-time compute scaling 会进一步偏向 inference
- **Inference is infinitely parallelizable**：rollout 之间没有 communication，任何硬件只要能跑 inference 就能贡献

这个 philosophy 与传统 RLHF pipeline（如 verl、TRL）截然不同，后者把 training 和 inference 绑在同一个 process 里，sequential 执行。PRIME-RL 把它们彻底分离成独立 executable，只通过 Parquet files 和 checkpoints 通信。

参考链接：
- PRIME-RL: https://github.com/PrimeIntellect-ai/prime-rl
- TOPLOC paper: https://arxiv.org/abs/2503.16412 (locality-sensitive hashing for verifiable inference)
- DiLoCo: https://arxiv.org/abs/2310.12296
- verl (HybridFlow): https://arxiv.org/abs/2409.19256

---

## 2. PRIME-RL：异步 RL 框架的核心机制

### 2.1 三种 RL execution mode 对比

Figure 6 展示了三种 mode，理解这个图对 build intuition 非常关键：

**Synchronous RL**：
- Training 和 inference 共享同一组 GPU
- Sequential 切换：先 inference，再 training
- 完全 on-policy，但 GPU 利用率低

**Centralized One-Step Asynchronous RL**：
- Dedicated training 和 inference nodes
- Inference 使用上一个 step 刚训练完的 policy
- Off-policy by 1 step

**Decentralized Two-Step Asynchronous RL** (INTELLECT-2 采用)：
- Inference workers 在 weight broadcast 完成前不能立即拿到新 weights
- 实际使用 2+ steps 前的 policy 生成 rollout
- Off-policy by 2+ steps

关键 ablation 实验（Figure 7）：在 DeepSeek-R1-Distill-Qwen-1.5B 上复现 DeepScaler，asynchrony level 从 1 到 4，reward trajectory 与 synchronous baseline 几乎一致。这说明 **off-policy 几步对 RL 训练效果影响很小**，这是整个 decentralized RL 成立的基石。

为什么 off-policy 几步是 OK 的？直觉上，policy 更新一步后，log-prob ratio $\frac{\pi_\theta}{\pi_{\theta_{old}}}$ 在 clip 范围内（$\epsilon = 0.2$）仍然 bounded，rollout 仍然提供有效的 advantage signal。这与 PPO 的 importance sampling 框架一致——只要 ratio 不爆掉，off-policy data 就可用。

### 2.2 Trainer 端的关键设计

**Log-prob recomputation**：PRIME-RL 不使用 inference workers 上报的 log-probabilities，而是在 training cluster 上用当前 policy 重新计算。原因是 vLLM 的 log-prob numerically unstable。这一点很重要——它意味着 inference workers 只需要返回 token sequence，不需要返回 log-prob，大大简化了 trustless inference 的验证。

**FSDP2 sharding**：model weights、gradients、optimizer states 都 sharded across GPUs，类似 ZeRO-3。32K sequence length + activation recomputation。

**Sequence packing**：32K context 下 sequence length variance 很大，padding 会浪费大量 compute。GRPO 的 token-level loss formulation 允许 cross-sample packing——通过 adapted attention mask 把多个 sample 拼接到 sequence dimension。这比 pretraining 的 packing 更 tricky，因为 RL 需要 preserve complete samples（advantage 是 sample-level 的），但 token-level loss 让这个 trick 变得可行。

### 2.3 Step counter endpoint 的设计

Inference workers poll 一个 endpoint，返回 "smallest step with insufficient rollouts"。这个设计很巧妙：

- Workers 可以 dynamic join/leave
- 不会因为某个 worker 挂掉而 block training
- 自然地实现了 pull-based 调度，类似 BitTorrent 的 tracker 模型

---

## 3. SHARDCAST：Policy Weight 广播网络

### 3.1 为什么要单独做 SHARDCAST

32B model 在 bfloat16 下约 62GB。要把这个广播到全球数百个 inference workers，传统方法（直接 HTTP 下载 from training node）会有几个问题：

1. **Bandwidth bottleneck**：training node 的 uplink 被打满
2. **Latency**：inference worker 要等整个 checkpoint 下载完才能开始 inference
3. **Fault tolerance**：下载中断要重新开始

SHARDCAST 的解决方案类似 **CDN + pipelined streaming**：

- 用 relay servers 做 tree-topology 转发
- Checkpoint sharded，可以 pipelined download（download shard 1 的同时 shard 2 在上传到 relay）
- 只保留最近 5 个 checkpoint 版本

### 3.2 Load balancing 的 EMA 采样策略

这个 part 比较有意思。如果每个 client 都 greedily 选最快的 relay，会导致 contention。他们用 **probabilistic sampling based on expected throughput**：

$$\text{expected throughput} \propto \text{success rate} \times \text{bandwidth}$$

每个 client 初始化时从所有 relay servers 下载 dummy file 来 estimate bandwidth 和 success rate。然后用 **exponential moving average (EMA)** 持续更新：

$$\hat{T}_t = \beta \cdot T_t + (1-\beta) \cdot \hat{T}_{t-1}$$

其中 $\hat{T}_t$ 是 t 时刻的 estimated throughput，$T_t$ 是实际观测值，$\beta$ 是 smoothing factor。

还加入 **healing factor** 来 encourage exploration of underutilized servers。这个设计本质上是一个 **multi-armed bandit with EMA estimate**，比 greedy 策略更 robust。

直觉上：probabilistic sampling 可以利用 multiple connections to different relays，total bandwidth > 任何 single connection。这类似于 BitTorrent 的 piece selection 策略。

### 3.3 Checksum 验证

每个 inference worker 下载完 checkpoint 后计算 SHA-256 checksum，与 training node 广播的 reference checksum 比对。不匹配就 discard 并尝试下一个 checkpoint。

一个值得注意的 trade-off：他们选择**不做 P2P weight transfer**，原因是 P2P 会暴露 worker IP 给 peers，增加 DoS attack 风险。这个 design choice 反映了 "permissionless but not fully trustless" 的现实——在 training node 和 inference worker 之间是 trustless 的，但 inference worker 之间是 isolated 的。

---

## 4. TOPLOC：Trustless Verifiable Inference 的核心

这是整篇 paper 最技术创新的部分。问题是：如何相信一个 untrusted inference worker 真的用指定 model weights 和指定 sampling 方式生成了 response？

### 4.1 TOPLOC 的基本原理

TOPLOC (Top-k Locality-Sensitive Hashing for Verifiable Inference) 基于 **locality-sensitive hashing (LSH)** 的思想：

**Key idea**：在 inference 过程中，每隔 32 个 token 捕获 final hidden states（通过 logits processor 的 hook），对这些 activations 做 hash 作为 commitment。Validator 用 prefill 重建这些 activations，验证 hash 匹配。

为什么这能 work？LSH 的性质是：**相似的 inputs 会 hash 到相同的 bucket**。如果 inference worker 用了 quantized model 或不同 model，reconstructed hidden states 会与 commitment 不匹配。

**Robustness to GPU non-determinism**：不同 GPU (甚至同型号 GPU) 上的 floating point 运算不完全 deterministic。TOPLOC 的 hash scheme 对这个 non-determinism robust——通过适当的 quantization 和 hashing tolerance。

**Overhead**：proof construction 只带来 ~1% 的 throughput reduction，因为：
- 每 32 个 token 才 hash 一次（不是每 token）
- Hash 在 CPU 上 async 执行，与 GPU forward pass parallel

### 4.2 验证的三个层次

**Computation checks**：
- TOPLOC proof 验证 final hidden states
- Validator 用 prefill 重建 activations
- 能 detect quantization、malicious model substitution、tensor parallel configuration mismatch

**Sampling checks**：
- **Termination check**：sequence 要么达到 max context length，要么以 EOS token 结束。如果 EOS 结束，要验证 EOS probability > 0.1，防止 worker 用低概率 EOS 提前截断
- **Token sampling check**：proper sampling 的 logit distribution 应该类似 exponential with mode at 1。如果 worker 用小 model generate token + 大 model prefill（来 pass TOPLOC），distribution 会 bimodal（modes near 1 and 0）。这个检测很 clever——它利用了 sampling distribution 的统计特征

**Sanity checks**：
- **Fixed data sampling**：seed = node_address · step + number_of_submissions_for_this_step。确保 worker 不能 cherry-pick easy samples
- **Value bounds check**：rewards、advantages 必须在 predefined bounds 内
- **Parquet formatting check**：确保文件能被 trainer dataloader 加载

### 4.3 为什么 Validator 比 Generator 快

Figure 3 提到 validator 可以比 inference provider 快 100×。原因：

1. Validator 只做 **prefill**（forward pass on 已知 sequence），不做 autoregressive generation
2. Validator 可以 **randomly sample** 要验证的 batches，不需要验证所有
3. Inference provider 不知道哪些 batch 会被验证，所以必须全部 honest

这构成了一个 **game-theoretic incentive**：honest behavior 是 dominant strategy，因为 dishonest 被抓的概率 × penalty > dishonest 的收益。

参考 TOPLOC 详细 paper: https://arxiv.org/abs/2503.16412

---

## 5. Training Recipe：GRPO 的关键改进

### 5.1 Reward 设计

**Total reward**:

$$r_{\text{total}}(y, l_{\text{target}}) = r_{\text{task}}(y) - \alpha \cdot |l_{\text{target}} - l_y|$$

变量解释：
- $y$：model output（response）
- $l_{\text{target}}$：target thinking length，从 discrete set (e.g., {2000, 4000, 6000}) 采样
- $l_y$：actual response length in tokens
- $r_{\text{task}}(y)$：binary task reward（math 正确=1，code 全部 unit test pass=1）
- $\alpha = 0.0003$：length penalty weighting

**为什么用 discrete set 而不是 continuous**？L1 paper 用 continuous range，这里简化为 discrete set 让 model 更容易 learn length following。这是一个 practical trade-off——discretize the objective space。

**为什么 code 用 binary reward 而不是 partial credit**？防止 reward hacking——model 可能 memorize public test cases 而不真正 solve 问题。这个 design choice 与 DeepCoder 类似。

### 5.2 Two-Sided GRPO Clipping（核心创新之一）

原版 GRPO objective:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t} \right) \right]$$

变量解释：
- $q$：prompt/question
- $\{o_1, ..., o_G\}$：从 old policy $\pi_{\theta_{\text{old}}}$ 采样的 G 个 outputs
- $o_{i,t}$：第 i 个 output 的第 t 个 token
- $o_{i,<t}$：第 i 个 output 的前 t-1 个 tokens（context）
- $\hat{A}_{i,t}$：group-relative advantage
- $\varepsilon$：clip parameter（通常 0.2）

**问题**：当 $\hat{A}_{i,t} < 0$（bad rollout）时，$\min$ 操作不会 clip upper bound。Policy 想远离 bad rollout，但如果 ratio $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ 变得非常大（比如 100+），会产生 huge loss spike，导致 training instability。

**Two-sided clipping 的解决方案**：引入额外 hyperparameter $\delta$ 作为 upper bound：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}[\cdots] \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \min\left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, \delta\right) \hat{A}_{i,t}, \text{clip}\left(\frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t} \right) \right]$$

$\delta = 4$ in experiments，$\delta > 1 + \varepsilon$ 确保仍然能 large update away from bad rollouts，但避免 ratio 爆炸到 100+。

**直觉**：这是 PPO/GRPO 的一个 asymmetry——对 positive advantage 有 upper clip (1+ε)，对 negative advantage 没有 upper clip。这个 asymmetry 的 original intent 是 "鼓励远离 bad rollout"，但在 large scale training 中会成为 instability source。Two-sided clipping 是一个 practical fix，concurrent work (MiniMax-01 等) 也发现了类似问题。

参考：
- DAPO: https://arxiv.org/abs/2503.14476
- Dr. GRPO: https://arxiv.org/abs/2503.20783
- GRPO original (DeepSeekMath): https://arxiv.org/abs/2402.03300

### 5.3 Data Filtering 的双重策略

**Offline filtering**：
- 用 DeepSeek-R1-Distill-Qwen-7B 计算 pass@8
- 保留 pass@8 ∈ [12.5%, 50%] 的 problems
- 太 easy（pass@8 > 50%）的 problems 没有 learning signal
- 太 hard（pass@8 < 12.5%）的 problems advantage 总是 0

**Online filtering**：
- GRPO 需要组内 reward 有 variance（否则 advantage 全 0，no signal）
- 持续 sampling 直到 batch 内所有 samples 都有 non-zero advantages
- 副作用：增加 inference compute per training step——正好利于 decentralized setup（更多 inference workers 可 onboard）

**Insight**：Figure 8 的 ablation 非常 informative。Unfiltered dataset 导致 reward stagnation，filtered dataset 后 reward 显著上升。这说明 **curriculum / difficulty filtering 是 RL training 的关键**，与 DeepScaler 的发现一致。

---

## 6. Training Instability at Scale 的现象学

这部分对 build intuition 非常重要，因为它揭示了 32B scale RL training 的 several pathology。

### 6.1 Escalating Gradient Norms

Figure 9a 显示：随着 training 推进，gradient norm 持续增长，即使没有 immediate spikes。这个现象与 model size 相关——7B 相对 stable，32B 明显 unstable。

**Mitigation**：aggressive gradient clipping (threshold 0.05-0.1)。这不 eliminate 问题，但 delay instability，extend viable training period。

### 6.2 Token Probability Clip Ratio Escalation

Figure 9b：clip ratio 持续上升，与 gradient norm growth 相关。Clip ratio 本质上 tracks consecutive optimizer steps 之间 logits 的 difference。Logits drift 越大，clip ratio 越高。

### 6.3 Entropy Resurgence Pattern

Figure 10 是一个特别有意思的 finding：
- Training 初期 entropy loss 下降（model 变得更 confident）
- 约 150 steps 后 entropy 开始 **回升**
- Entropy 回升之后通常跟着 catastrophic collapse

这个 pattern 在 RL training 中是一个 early warning signal。Entropy 回升可能表示 policy 开始 explore "wrong direction"，或 model 的 confidence 在 degrade。

增加 KL penalty weight 能 delay collapse，但也会 slow learning。这是一个 stability-efficiency trade-off。

### 6.4 QwQ vs DeepSeek-R1-Distill-32B 的稳定性差异

这是一个 important finding：QwQ-32B 比 DeepSeek-R1-Distill-Qwen-32B 更 unstable，尽管两者都 based on Qwen 2.5 pretrained model。

**Hypothesis**：QwQ 已经经过一轮 RLVR (RL with verifiable rewards) training，这使得它对 subsequent RL 更 sensitive。可能的原因：
- RL-trained model 的 loss landscape 更 "sharp"
- 已经在某个 reward plateau 上的 model，进一步 RL 容易 push it off
- 类似于 "catastrophic forgetting" 但发生在 RL 阶段

这个 finding 对 RL training 的 "stacking" 有重要 implication——每一轮 RL 都会让 model 更难进一步 fine-tune stably。

### 6.5 torch.compile 的 catastrophic failure

Figure 11 显示 torch.compile 导致 early instability 和 reward collapse，而 no-compile baseline stable across 1200 steps。可能是某个 faulty generated kernel 导致的。他们决定全 codebase disable torch.compile，代价是 slightly increased memory usage。

这是一个值得记住的 **practical lesson**：在 critical training run 中，torch.compile 的 numerical instability 可能会在 later stages 才显现。

---

## 7. 实验结果分析

### 7.1 Benchmark 性能

Table 1 的关键对比：

| Model | AIME24 | AIME25 | LiveCodeBench v5 | GPQA-Diamond | IFEval |
|-------|--------|--------|------------------|--------------|-------|
| INTELLECT-2 | 78.8 | 64.9 | 67.8 | 66.8 | 81.5 |
| QwQ-32B (base) | 76.6 | 64.8 | 66.1 | 66.3 | 83.4 |
| Qwen-R1-Distill-32B | 69.9 | 58.4 | 55.1 | 65.2 | 72.0 |
| DeepSeek-R1 | 78.6 | 65.1 | 64.1 | 71.6 | 82.7 |

**分析**：
- INTELLECT-2 在 AIME24 (+2.2)、LiveCodeBench (+1.7) 上超越 QwQ-32B
- IFEval 略降 (-1.9)，因为只 train on math/coding，没 train on instruction following
- DeepSeek-R1 在 GPQA 上明显领先 (71.6 vs 66.8)，可能因为更 extensive 的 general reasoning training

**Insight**：QwQ-32B 已经被 extensively RL-trained，再在上面做 RL 只能获得 marginal improvement。Paper 自己承认需要 better base model (如 Qwen3) 或更高质量 dataset 才能有更大提升。

### 7.2 Compute Utilization

- SHARDCAST broadcast 平均 14 分钟（62GB weights，~590 Mb/s throughput）
- TARGET-SHORT: 22 分钟内完成 rollout step
- TARGET-LONG: 29 分钟内完成 rollout step
- Training-to-inference FLOPs ratio: 1:4.5

这个 1:4.5 ratio 验证了 paper 的核心 thesis——**inference dominate compute**，这使得 decentralized RL feasible。

### 7.3 Asynchronous overlap 的效果

TARGET-SHORT:
- Broadcast: 14 min
- First data submission: 10 min after broadcast
- Verification: ~1 min (TOPLOC subset sampling)
- Sufficient batch: 22 min after broadcast
- Training execution: 22 min

几乎 perfect overlap！这是 asynchronous RL 的 ideal scenario。

TARGET-LONG:
- 更长 target length → 更长 generation time
- Sufficient batch: 29 min after broadcast
- Training execution: 21 min

这里有 slight imbalance，但仍然 acceptable。

---

## 8. Discussion 部分的核心论点

### 8.1 Test-Time Compute Scaling 天然适合 Decentralization

Paper 的论点：
1. Pretraining scaling: communication-bound，需要 DiLoCo 等技术 reduce communication
2. Test-time compute scaling: inference-heavy，而 inference is embarrassingly parallel

随着 reasoning model 的 thinking length 增长，inference compute 会进一步 dominate。这 opens door to training hundreds-of-billions-parameter models on globally distributed heterogeneous compute。

### 8.2 Inference-Heavy RL 的 scaling dynamics

Key insight: as models tackle harder tasks with sparse rewards，**exploration becomes the dominant cost**。Only a small subset of rollouts contains strong learning signals。

这 reshapes 了 decentralized RL 的 scaling dynamics：
- Memory constraints 不再是 bottleneck（inference 比 training memory 要求低很多）
- 更多 consumer-grade hardware 可以参与
- Heterogeneity become a feature, not a bug

---

## 9. Future Work 的几个方向

1. **VinePPO style methods**：用 Monte Carlo value estimation 替代 value network，inference-heavy，适合 decentralized
2. **Tool calls for reasoning**：web browsing、code interpreter、API calls 进 reasoning chain
3. **Crowdsourced RL environments**：社区贡献 verifiable environments
4. **Model merging + DiLoCo for RL**：多个 model 独立训练后 merge，或 continuous merging via DiLoCo

参考：
- VinePPO: https://arxiv.org/abs/2410.01679
- SkyRL: https://arxiv.org/abs/2505.15034
- RAGEN: https://arxiv.org/abs/2504.20073

---

## 10. 一些联想与延伸思考

### 10.1 与 OpenAI o1 / DeepSeek-R1 的关系

INTELLECT-2 本质上是在 **复现 R1-style RLVR training**，但在 decentralized infrastructure 上。这证明了：
- RLVR 的 training recipe 已经成熟到可以被 replicate
- 真正的 bottleneck 是 compute + data quality，而不是算法 novelty

### 10.2 与 Web3 / Crypto 的关系

Prime Intellect Protocol 用了 decentralized ledger、cryptographic signing、on-chain events。这实际上是一个 **proof-of-useful-work** 的变体——compute contribution 被记录在链上，可以想象未来 token incentives 机制。

参考 Prime Intellect Protocol: https://github.com/PrimeIntellect-ai/protocol

### 10.3 与 RLHF/RLAIF 的关系

INTELLECT-2 用的是 **verifiable rewards**（math correctness、unit tests），不是 human preference。这与 RLAIF 不同——verifiable rewards 更适合 decentralized，因为：
- Verification 可以 automated
- 不需要 human annotators
- Binary signal 更 clear，更容易 verify

### 10.4 与 DeepSeek-V3 / R1 的训练 insight 对比

DeepSeek-R1 的训练 stability 报告也提到了类似的 gradient norm escalation。INTELLECT-2 的 two-sided clipping 是对 R1 训练 recipe 的一个 important fix。可以想象，如果 R1 的训练也用了 two-sided clipping，可能能避免一些 instability。

### 10.5 Sequence Packing in RL 的 trick

这个 detail 值得单独强调。Pretraining 的 sequence packing 很 straightforward（因为 next-token prediction 是 local 的）。RL 的 packing 更 tricky：
- 优点：避免 padding waste，32K context 下效率提升显著
- 难点：RL 需要 preserve complete samples for advantage computation
- Trick：GRPO 的 token-level loss formulation 让 cross-sample packing 可行，通过 adapted attention mask

这个 trick 可以被 broader RL community 采用，特别是 long-context RL training。

### 10.6 关于 Asynchronous RL 的 theory

Figure 7 的 ablation 显示 4-step asynchrony 仍然 stable。但 theory 上，importance sampling 的 variance 会随 asynchrony 增长。一个 interesting research question：**asynchrony level 的理论上限是多少**？这与 policy update 的 step size、reward sparsity、task difficulty 都相关。

---

## 总结

INTELLECT-2 是一个 **infrastructure-first 的 research contribution**。它的核心价值不在于算法 novelty，而在于：

1. **证明 decentralized RL at 32B scale 是可行的**
2. **提供了完整的 open-source infrastructure stack**（PRIME-RL + TOPLOC + SHARDCAST + Protocol）
3. **发现了 large-scale RL training 的 several instability patterns**（gradient escalation、entropy resurgence、clip ratio escalation）
4. **提供了 practical fixes**（two-sided clipping、aggressive gradient clipping、data filtering）

对于想要复现或扩展 RLVR training 的研究者，这篇 paper 提供了一个完整的 blueprint。特别是对 resource-constrained 的 research group，decentralized RL 开辟了一条 new path——不需要 massive GPU cluster，只要能 aggregate 足够多的 consumer-grade compute 就能 train frontier reasoning models。

值得关注的后续方向：
- Qwen3 base model + INTELLECT-2 recipe 能否达到 R1 级别 performance？
- Tool use RL 的 decentralized training
- Model merging 在 RL 中的 effectiveness
- 更长 thinking budget (100K+ tokens) 的 scaling behavior

关键 open source links：
- Model: https://huggingface.co/PrimeIntellect/INTELLECT-2
- PRIME-RL: https://github.com/PrimeIntellect-ai/prime-rl
- Dataset: https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1
- Protocol: https://github.com/PrimeIntellect-ai/protocol

这篇 paper 代表了 AI research 的一个有趣 trend——**infrastructure innovation 成为 algorithmic progress 的 enabler**。当 training scale 达到一定 level，infrastructure 的 design 直接决定了 algorithmic exploration 的空间。
