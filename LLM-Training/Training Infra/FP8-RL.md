---
source_pdf: FP8-RL.pdf
paper_sha256: 1bc3b62ccfdcb3943d1103de8d5cff12d6f0a63445f598ba01104a68ff676f23
processed_at: '2026-08-04T10:19:27-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲 FP8-RL

Andrej，我换个腔调，像在咖啡厅给你讲这篇 paper 那样。

---

## 一句话先给你

**RL 训练 LLM 的时候，rollout（让 model 自己生成 response）这步慢得要死，占整个迭代时间 80%。这篇 paper 干的事情就是：把 rollout 那一路全换成 FP8 低精度，weight 该量化量化，KV-cache 该压压，然后训练这一路用 importance sampling 兜底别让 policy 崩。Qwen3-8B 上 rollout 快 44%，30B MoE 上快 30-50%，accuracy 还跟 BF16 baseline 对齐。**

就这么个事。听起来简单，但里面坑很多。

---

## 为什么 rollout 是 bottleneck

你想想 RL 的 loop 是什么样子：policy 更新一次 → 让这个新 policy 生成一堆 responses → 算 reward → 拿 gradient 更新 policy → 再生成一堆 → ……循环。

生成那一阶段就是 rollout。 rollout 慢在哪？两个地方：

**第一，autoregressive decoding 天生就是 memory-bound**。你生成一个 token，得把整个 model 的 weights 从 HBM 搬到 tensor core 算一遍。30B model 的 weights 在 BF16 下是 60 GB，你每生成一个 token 就搬一次，H100 的 HBM 带宽 3.35 TB/s 很快就吃满了。算的功夫根本没多少，时间全花在搬数据上。

**第二，long context 下 KV-cache 膨胀**。你生成 20K tokens，每个 token 都要在 KV-cache 里存一对 K 和 V。一个大 batch × 20K seq，KV-cache 能吃掉几十 GB。vLLM 的 PagedAttention 一旦 KV-cache 装不下就 preempt 你的 request（swap 出去或重算），throughput 直接跳水。

Seer (https://arxiv.org/abs/2511.14617) 实测 synchronous LLM RL 里 rollout 占 80% 时间。所以你想加速 RL，砍 rollout 是最直接的杠杆。

---

## FP8 是个什么杠杆

FP8 是 8-bit 浮点（1 sign + 4 exp + 3 mantissa，E4M3 格式），跟 BF16 比：

- **存储减半**：BF16 2 bytes，FP8 1 byte。weights 搬运快 2 倍，KV-cache 容量翻倍。
- **算力翻倍**：H100 的 FP8 tensor core 算力是 BF16 的 2 倍。

听起来 dream。但 FP8 早就被 inference 引擎（vLLM、SGLang、TensorRT-LLM）支持了，为啥 RL 里大家没用？

---

## 两个真正的难点

**难点一：RL 每 step 都要换 weights。**

普通 inference 是 "load weights once, serve forever"。RL 是 "每 step 都换 weights，换了立刻 serve 一次，再换"。这意味着：

1. 训练 backend (FSDP/Megatron) 算完 gradient 更新出新的 BF16 weights
2. 这些 weights 要立刻被 quantize 成 FP8
3. 推到 inference engine
4. inference engine 重新装填自己的 weight buffer
5. 开始生成

整套动作每 step 一次，开销必须小于 FP8 rollout 节省的时间。30B 模型 weight sync 一次涉及 30 GB，通过 NVLink 900 GB/s 大约 33 ms，相比 rollout 节省的几秒，完全可以接受。

但工程上要做对——inference engine 原本不支持运行时动态换 FP8 weights，你得 patch 它。

**难点二：低精度 rollout 跟高精度 trainer 对不上。**

这个是 algorithm 层的雷。你 trainer 在优化 π_θ（BF16），rollout 实际是用 π_θ^FP8 生成的。两者分布不一样——quantization 把 logits 微微扭曲了，每生成一个 token 偏一点，20K tokens 累积下来 sampling 分布能差挺多。

你拿 π_θ^FP8 生成的 trajectories 去更新 π_θ，等于在优化一个混合分布。长跑下去 policy 会漂移，甚至 collapse。

参考 [5] (https://richardli.xyz/rl-collapse) 把这个现象写得挺清楚——"Speed Kills Stability"，名字就告诉你速度上去了 stability 拉胯。

---

## 他们的方案：三层 FP8 + TIS 兜底

### 第一层：W8A8 Linear Quantization

把 attention 的 q/k/v/o_proj 和 MLP 的 gate/up/down_proj 全 quantize 成 FP8。MoE 的 expert fc1/fc2 也 quantize。**唯独 lm_head 不动**——lm_head 直接产生 logits，量化 logits 再 softmax 采样分布偏差太大，得不偿失。

quantization 方案抄 DeepSeek-V3 (https://arxiv.org/abs/2412.19437)：

- Weights 用 128×128 blockwise 量化，static（每个 RL step 量化一次，rollout 期间不变）
- Activations 用 dynamic 量化（每个 forward pass 实时算 scale）

公式：

$$\widehat{W}_{ij} = \text{round}\left(\frac{W_{ij}}{\text{scale}_{ij}}\right) \in \text{FP8}$$

$W_{ij}$ 是 weight matrix 第 i 行第 j 列的元素。$\text{scale}_{ij}$ 是这个元素所在的 128×128 block 内的最大绝对值。$\widehat{W}_{ij}$ 是量化后 FP8 值。

为啥 128×128？太小了 outlier 影响范围小但 GEMM tile 不友好；太大了 outlier 把整个 block 的精度拉垮。128×128 是 H100 tensor core tile 友好且 outlier 局部化的甜区。

参考 FP8 format 老祖宗 paper：https://arxiv.org/abs/2209.05433 (Micikevicius et al., 2022)。

### 第二层：KV-Cache FP8

这个收益其实比 linear quantization 还大。8B 模型在 20K context 下，linear W8A8 给 20% speedup，KV-cache FP8 单独给 38%！

为啥？因为 long context rollout 真正的 bottleneck 是 KV-cache 容量，不是 GEMM 算力。BF16 baseline 配置下 vLLM 频繁 preempt 请求——KV-cache 装满了，请求挂起。开 KV-cache FP8 容量翻倍，preemption 率大幅下降，throughput 阶跃式提升。

**但是**，KV-cache 量化有个细节：K 和 V 在 attention 里是 `softmax(QK^T / sqrt(d)) V`，scale 不准会直接破坏 attention 分布。所以需要 per-step recalibration QKV scales——每步 weight 更新后 K, V 的数值分布变了，scale 必须重算。

他们做了两种实现：
- **Inference-Side**（veRL）：每 step rollout 前重置 vLLM 内部的 `calculate_kv_scales` flag，让 vLLM 第一次 forward 时自动重新算 scale。
- **Trainer-Side**（NeMo-RL）：训练 backend 每步结束用训练数据子集算 scale，然后 sync 到 inference engine。开销 2-3% step time，但控制更细。

### 第三层：TIS 修正 mismatch

这是 algorithm 的关键。用 importance sampling 修正 low-precision rollout 引入的 off-policy 成分。

公式：

$$w(a|s) = \frac{\pi_\theta(a|s)}{\pi_\theta^{FP8}(a|s)}$$

$s$ 是 state（已生成的 context），$a$ 是 action（下一个 token）。$\pi_\theta$ 是 BF16 training policy，$\pi_\theta^{FP8}$ 是 FP8 rollout policy。$w(a|s)$ 就是 importance weight，用来 reweight 这个 token 在 gradient 里的贡献。

实现上：rollout 时 FP8 engine 记录每个 token 的 log-prob（FP8 路径算的）。训练时 BF16 model 重新 forward 这些 token 算 log-prob（BF16 路径算的）。两者相减得 $\log w$。

但 naive IS 有个问题：20K tokens 的 long sequence 下，ratio 累乘方差爆炸。一个 outlier token 的 $w = 100$ 会让整条 sequence 的 gradient 炸掉。

所以加 clip：

$$w_{TIS}(a|s) = \text{clip}(w(a|s), C)$$

$C$ 是 clipping threshold，paper 用 C=2。把 outlier 限制住，引入一点 bias 换 variance 控制，bias 是有界的。

参考 [7] (https://fengyao.notion.site/off-policy-rl) 是同作者群的早期工作，分析了"高效 RL 框架暗藏 off-policy"的陷阱。

---

## 实验结果：三个看点

### 看点一：Qwen3-8B Dense 上的 ablation (Figure 2)

三个配置：
1. BF16 baseline（橙色）
2. FP8 W8A8 + TIS（蓝色）
3. FP8 W8A8 不加 TIS（绿色，ablation）

结果：
- 蓝色（FP8 + TIS）跟橙色 baseline 完全对齐，validation accuracy / reward / response length 都贴得很紧
- 绿色（FP8 不加 TIS）明显退化，accuracy 掉，response length 异常

**这个 ablation 是 paper 最有力的证据：FP8 quantization 引入的 mismatch 不是白噪声，是有偏的，必须用 TIS 修正。**

rollout 性能：FP8 比 BF16 快 10-20%。8B 模型 GEMM 算力利用不算高，所以收益有限。长序列收益更大（memory-bandwidth bound 更严重）。

### 看点二：Qwen3-30B MoE 上的收益放大 (Figure 4, 5)

30B MoE 模型上 FP8 rollout 快 **30-50%**，是 8B dense 的 2-3 倍。

为啥 MoE 收益大？三个因素叠加：

1. **算力利用更高**：30B 模型 GEMM 大，FP8 tensor core 能跑满，绝对时间节省大。
2. **Memory 减半更值钱**：30B BF16 是 60 GB，FP8 是 30 GB。每 step weight sync + 每 token weight loading 都受益。
3. **KV-cache 容量释放**：FP8 化省下 30 GB HBM，能装更多 KV-cache，vLLM 并发更高，preemption 更少。**这是复合效应**——vLLM 的 preemption 一旦降下来，throughput 不是线性涨是阶跃涨。

但 MoE 有个新麻烦：**routing mismatch**。inference engine 和 training backend 在 numerics 上有细微差异，top-k expert 选择可能不一致。token 落到不同 expert，输出分布差很多。Paper 观测到 MoE 的 mismatch KL 持续增长（dense 模型是稳定的），但 TIS 还能 hold 住 accuracy。

如果更激进的场景 TIS hold 不住，需要 R3 (Rollout Routing Replay, https://arxiv.org/abs/2510.11370)——rollout 时记录每个 token 实际走的 expert id，training 时强制 replay 同样 routing。或者 MIS (Masked Importance Sampling, https://ringtech.notion.site/icepop) 把异常 ratio 的 token 直接 mask 掉。

### 看点三：KV-cache FP8 单独 38% speedup (Figure 8)

四种配置对比：

| 配置 | Speedup | Mismatch KL |
|---|---|---|
| BF16 baseline | 0% | ~0 |
| Linear W8A8 only | ~20% | 低 |
| KV-cache FP8 only | ~38% | 略高于 Linear |
| Full FP8 (Linear + KV + Attention) | **44%** | 最高但仍稳定 |

**KV-cache FP8 单独给 38%，比 linear 的 20% 还大**。这反直觉但其实合理——8B 小模型在 20K long context 下是 KV-cache bound 不是 compute bound。BF16 baseline 和 Linear W8A8 配置都频繁触发 vLLM preemption，开 KV-cache FP8 容量翻倍，preemption 大幅下降，throughput 阶跃式提升。

Full FP8 三层 quantization error 叠加，mismatch KL 最高，但 TIS 仍然能 hold accuracy。**这是 paper 的乐观结论：aggressive full-stack FP8 + TIS 是 viable 的**。

---

## End-to-End FP8：更进一步 (Section 2.4)

前面实验都是 training 用 BF16，rollout 用 FP8。这里探索 training + rollout 都用 FP8。

结果（Figure 9）：
- Accuracy 跟 BF16 baseline 对齐
- Mismatch KL 比 FP8 rollout-only 还低（precision alignment 减小 distribution drift）
- Training 侧额外 20% speedup

**Intuition**：trainer 用 BF16 但 rollout 用 FP8，两者数值路径 systemic 偏移。gradient 应用回 BF16 π_θ 让它进一步偏离 FP8 π_θ^FP8，形成正反馈。End-to-end FP8 让两者数值路径一致，drift 自然小。

不过 end-to-end FP8 仍有 mismatch（kernel-level 差异、quant/dequant round-trip loss），TIS 仍然要开。

参考 [22] (https://arxiv.org/abs/2510.26788, Qi et al., "Defeating the Training-Inference Mismatch via FP16")：用 FP16 替代 BF16 也能减小 mismatch，因为 BF16 的 7-bit mantissa 比 FP16 的 10-bit mantissa 精度低，autoregressive decoding 累积误差大。这是另一个角度的 precision alignment。

FP8 training 的技术基础见 FP8-LM (https://arxiv.org/abs/2310.18361) 和 DeepSeek-V3。NeMo-RL 在 paper 中承担 end-to-end FP8 实验，因为 NeMo 原生支持 Megatron-LM 的 FP8 training。

---

## 我的几点 take-away

### 1. 这篇 paper 真正的贡献是 integration

它没发明新算法。Blockwise FP8 来自 DeepSeek-V3，KV-cache FP8 来自 vLLM，TIS 是经典 IS 修正。Paper 价值在于 **把这几个 piece 拼成 production-ready stack 并跑通端到端验证**，加上 ablation 完整度高（FP8 加不加 TIS、KV-cache 单独 vs 联合、end-to-end vs rollout-only 都做了对比）。

### 2. TIS 是 paper 的灵魂

整个 paper 的隐含 message 是：**低精度 rollout 可以做，前提是你算法上修正 mismatch**。TIS 看起来土，但 work。MoE 上可能不够，需要 R3 / MIS。

### 3. KV-cache FP8 的收益被低估了

Linear W8A8 大家都关注，但 long context 下 KV-cache FP8 才是性能 killer feature。原因不是 FP8 算得快，是 **容量翻倍救了 preemption**。这个 insight 对 multi-turn agentic RL 更重要——context 越长 KV-cache 压力越大。

### 4. MoE routing mismatch 是更深层的问题

Paper 观测到 MoE 的 mismatch KL 持续增长，TIS 还能 hold。我倾向于**侥幸**——TIS 是 token-level reweighting，routing 错配会让 token 落到完全不同 expert，输出分布差异巨大，TIS 的 clip 反而 mask 掉了问题。R3 是更彻底的解法，但 paper 没用。值得追问：在什么 model size / training length 下 TIS 会 break？

### 5. 跟 bitwise consistency 路线的对比

vLLM + TorchTitan (https://blog.vllm.ai/2025/11/10/bitwiseconsistent-train-inference.html) 走另一极端：完全消除 mismatch，trainer 和 rollout numerics bitwise 一致。不需要 TIS，但限制严格——不能用 FP8 rollout（因为 trainer 是 BF16）。

我猜测未来方向是混合路线：**critical path（采样、reward 计算）bitwise 一致，non-critical path（GEMM、attention）允许 FP8 + IS 修正**。两者互补，不是互斥。

---

## 怎么用

veRL 里启用 FP8 W8A8：

```bash
actor_rollout_ref.rollout.quantization=fp8
actor_rollout_ref.rollout.name=vllm
```

KV-cache FP8 (veRL)：

```yaml
actor_rollout_ref:
  rollout:
    quantization:
      kv_cache_dtype: fp8_e4m3
      calculate_kv_scales: True
```

性能必备：
- CUDA 12.9+
- DeepGEMM (https://github.com/deepseek-ai/DeepGEMM)：DeepSeek 开源的 FP8 GEMM 库，vLLM 0.11+ / SGLang 0.55+ 默认启用

**别忘了开 TIS**。不开 TIS 跑 FP8 rollout，accuracy 会崩。

---

## 一句话再总结一次

**FP8 rollout 在 RL 里 work，关键是三件套——per-step weight re-quantization + per-step QKV scale recalibration + token-level TIS。工程拼装漂亮，algorithm 借力 importance sampling，KV-cache FP8 是被低估的 killer feature。MoE 的 routing mismatch 是潜在地雷，long run 稳定性还需要更多验证。**

如果你要我推荐下一步该读什么：先看 DeepSeek-V3 报告的 FP8 部分（理解 blockwise scaling 的 rationale），再看 [5] RL Collapse（理解 mismatch 为什么致命），最后看 vLLM bitwise consistency blog（理解另一条极端路线）。这篇 FP8-RL 本质是 engineering integration paper，看它怎么把几个 piece 拼起来 work，是读它的正确姿势。

---

# FP8-RL: LLM Reinforcement Learning 的 FP8 Rollout Stack 深度解析

Andrej, 这篇 paper 来自 NVIDIA 北京团队（Zhaopeng Qiu 等），是 2025 年下半年 LLM RL 系统 engineering 的代表作品之一。核心问题是：**RL 中的 rollout (generation) 占据迭代时间约 80%（Seer 报告 [1]），而 FP8 在静态 inference 上已经成熟，但 RL 有两个独有障碍——每步 weight 变化 + train-inference mismatch**。Paper 给出了一套 practical stack，Qwen3-8B 上 44% rollout speedup，Qwen3-30B MoE 上 30-50%。

下面我按 paper 顺序展开，但加入大量背景、intuition 和相关联想。

---

## 1. 问题背景：为什么 RL 的 rollout 是 bottleneck

在 RLHF / RLAIF / RLVR 这类 post-training 流水线中，每一步 optimization 都需要 fresh rollouts：当前 policy π_θ 生成 responses → reward → gradient update → 新的 π_θ → 再生成。这与 SFT / pretraining 的根本区别是 **生成-训练交替的同步耦合**。

Seer (https://arxiv.org/abs/2511.14617) 的测量显示 synchronous LLM RL 中 rollout 占 ~80% 端到端时间。原因有两个：

1. **Autoregressive decoding 是 memory-bandwidth bound**：每生成一个 token 需要加载全部 weights 和 KV-cache，arithmetic intensity 极低。一个 30B 模型生成 20K tokens 的 rollout，光是 weight loading 就是 30B × 2 bytes (BF16) = 60 GB / token，乘以 batch_size × seq_len 的总 token 数，H100 的 3.35 TB/s HBM 带宽很快被打满。
2. **KV-cache 在 long context 下膨胀**：KV-cache 大小 = `2 × num_layers × seq_len × num_kv_heads × head_dim × batch_size × dtype_bytes`。Qwen3-8B 是 GQA，但 20K context × 大 batch 下仍然经常触发 vLLM 的 preemption（KV-cache 空间不够时挂起请求）。

这两个瓶颈都指向 FP8——它能同时减半 memory traffic 和翻倍 KV-cache 容量。但 RL 的工程难点在于**weight 每步变化**和**低精度 rollout 偏离 trainer 假设**。

---

## 2. W8A8 Blockwise FP8 Quantization

### 2.1 量化格式与粒度

**E4M3 FP8 format**：1 sign bit + 4 exponent bits + 3 mantissa bits，dynamic range ≈ [−448, 448]，约 7-bit 有效精度。对比 E5M2（5 exp + 2 mantissa，range [−57344, 57344]，精度更低但动态范围更大），E4M3 适合 forward pass 因为 weights 和 activations 的数值范围相对窄，精度优先。

参考文献 [12] 是 FP8 format 的奠基性 paper：https://arxiv.org/abs/2209.05433 (Micikevicius et al., "FP8 Formats for Deep Learning")。

**Block size B = 128×128** 来自 DeepSeek-V3 (https://arxiv.org/abs/2412.19437) 的方案。Paper 公式 (1)：

$$\widehat{W}_{ij} = \text{round}\left(\frac{W_{ij}}{\text{scale}_{ij}}\right) \in \text{FP8}$$

变量含义：
- $W_{ij}$：weight matrix 在第 $i$ 行第 $j$ 列的元素（BF16/FP32 原始值）
- $\text{scale}_{ij}$：该元素所属 128×128 block 的 scaling factor，等于 block 内最大绝对值 $\max(|W_{\text{block}}|)$，可能乘上一个小的 over-amplitude factor 防止边界溢出
- $\widehat{W}_{ij}$：量化后的 FP8 值

**Intuition**：per-tensor quantization 用一个 scale 覆盖整个矩阵， outliers 会让 scale 过大、其他值精度全损；per-block 128×128 是 sweet spot——足够小让 outlier 只影响局部 block，又足够大让 tensor core 的 GEMM tile 友好（H100 tensor core 的 tile 通常 16×16 或 128×128 量级）。

DeepSeek-V3 实际用的是 **1×128 tiles for activations + 128×128 blocks for weights** 的混合粒度。Paper 这里说 weights 用 128×128 blocks，activations 是 dynamic per-token/per-tile 量化（paper 没有完全说清楚，但结合 vLLM/SGLang 的实现，activations 通常是 per-token 1×128）。

### 2.2 量化范围

**Quantized**：attention 的 `q_proj, k_proj, v_proj, o_proj`；MLP 的 `gate_proj, up_proj, down_proj`；MoE expert 的 `fc1, fc2`。

**Excluded**：embedding layers（vocab size 太大，量化开销高且对 representation 敏感）、normalization layers（RMSNorm 的 scale 参数很小，量化会破坏 stability）、`lm_head`（直接产生 vocabulary logits，量化会显著影响 next-token distribution）。

**Intuition**：lm_head 的 logits 是采样分布的直接来源。FP8 量化 logits 后再 softmax，由于 softmax 对 logits 的微小差异高度敏感（尤其是 top-k 之间），采样分布会偏离。这就是为什么 paper 显式排除 lm_head。

### 2.3 Static vs Dynamic

- **Weights static**：每个 RL step 的 weight 在 sync 阶段一次性量化，rollout 期间不变。这样 GEMM kernel 不需要在线 quantize weights。
- **Activations dynamic**：每个 forward pass 实时量化。因为 activations 随 input 变化大，static 量化会累积误差。dynamic 量化在 H100 上有硬件原生支持，开销很小。

---

## 3. Dynamic Weight Synchronization Pipeline

这是 paper 的工程核心。Figure 1 描述三阶段：

### Phase 1: Initialization
- 配置 vLLM/SGLang 启用 FP8 模式
- Patch inference engine 以支持 **运行时动态加载 FP8 weights**（这是非标准操作——通常 inference engine 启动后 weights 固定）

### Phase 2: Weight Synchronization (每个 RL step)
1. 从 training backend (FSDP/Megatron-LM) 取出最新的 BF16/FP16 weights
2. 用 blockwise 方案量化成 FP8
3. 通过 NCCL 或 SHM 推送到 inference engine
4. inference engine 更新自己的 weight buffer

### Phase 3: Inference
- inference engine 用预加载的 FP8 weights 生成
- activations 在 GEMM 前 dynamic 量化

**Intuition / 难点**：传统 inference 是 "load once, serve forever"。RL 是 "load every step, serve once"。每步 weight 同步的开销必须小于 rollout 节省的时间，否则得不偿失。30B MoE 模型 weight 同步一次涉及 ~30 GB 数据传输，在 8 卡 H100 间通过 NVLink (900 GB/s) 大约 33 ms，相比 rollout 节省的几秒，可忽略。

veRL 框架本身是 hybrid flow 设计：https://github.com/volcengine/verl (Sheng et al., HybridFlow, https://arxiv.org/abs/2409.19256)。它把 Actor/Critic/Reference/Reward 模型抽象为 single-controller，跨 training backend (FSDP/Megatron) 和 inference backend (vLLM/SGLang) 解耦。FP8-RL 就在这个抽象上插入 quantization hook。

---

## 4. Importance Sampling for Rollout Correction（算法核心）

这是 paper 的算法关键贡献。低精度 rollout 引入了 off-policy 成分。

### 4.1 Mismatch 的形式化

Trainer 优化的目标分布是 π_θ（BF16/FP16 高精度），但 rollouts 实际来自 $\pi_\theta^{FP8}$。两者由于 quantization error 不同，导致：

- 标准 PPO 假设 ratio = π_θ_new / π_θ_old ≈ 1（同一 policy 的两个时刻）
- 这里多了一层：rollout 实际是 π_θ^FP8，所以真正的 ratio 应该是 π_θ / π_θ^FP8

如果不修正，gradient 实际上是在优化一个混合分布，长期累积会让 policy 漂移甚至 collapse。参考 [5] (https://richardli.xyz/rl-collapse, "When Speed Kills Stability") 详细分析了这个现象。

### 4.2 公式 (2): Importance Weight

$$w(a|s) = \frac{\pi_\theta(a|s)}{\pi_\theta^{FP8}(a|s)}$$

变量含义：
- $s$：state（context，已生成 tokens）
- $a$：action（下一个 token）
- $\pi_\theta(a|s)$：BF16 training policy 在 s 下取 a 的概率
- $\pi_\theta^{FP8}(a|s)$：FP8 rollout policy 在 s 下取 a 的概率（rollout 时实际用的）
- $w(a|s)$：importance weight，用来在 gradient 中 reweight 这个 token 的贡献

**实现层面**：rollout 阶段，FP8 engine 生成 tokens 时记录每个 token 的 log-prob $\log \pi_\theta^{FP8}(a_t|s_t)$。训练阶段，BF16 model 重新 forward 这些 tokens 算出 $\log \pi_\theta(a_t|s_t)$。两者相减得到 $\log w$。

### 4.3 公式 (3): Truncated Importance Sampling

$$w_{TIS}(a|s) = \text{clip}(w(a|s), C)$$

变量：
- $C$：clipping threshold，paper 用 C = 2
- $\text{clip}$：把 $w$ 限制在 $[-C, C]$ 或 $[1/C, C]$（这里 paper 写得有点模糊，实际 PPO 系列通常是 $[1-C, 1+C]$ 或 $[\exp(-C), \exp(C)]$；TIS 论文常用的是双向 clip）

**Intuition**：Naive IS 的方差随 sequence 长度指数增长（每个 token 的 ratio 相乘）。20K tokens 的 long context 下，一个 outlier token 的 $w = 100$ 会让整条 sequence 的 gradient 爆炸。Clip 到 C=2 把 outlier 影响限制住，代价是引入 bias，但 bias 是可控的、有界的。

参考 [7] (https://fengyao.notion.site/off-policy-rl) 是同一作者群的早期工作，分析了"高效 RL 框架暗藏 off-policy"问题。

### 4.4 TIS vs PPO clip 的关系

经典 PPO 的 objective：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)\right]$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$。

在 FP8-RL 中，实际 ratio 是：

$$r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_\theta^{FP8}(a_t|s_t)} \cdot \frac{\pi_{\theta_{old}}^{FP8}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

但通常 θ ≈ θ_old（一步内），所以近似简化为 $\pi_\theta / \pi_\theta^{FP8}$。TIS 先对 $\pi_\theta / \pi_\theta^{FP8}$ clip，再做 PPO 的 clip。两层防护。

### 4.5 MIS 和 R3 (MoE 专用)

Paper 在 MoE 实验里观察到 mismatch KL 持续增长。原因是 **expert routing 不一致**：inference engine 和 training backend 在 numerics 和 kernel 上有细微差异，导致 top-k expert 选择不同。一个 token 被 router 分到不同 expert，输出完全不同。

解决方案：
- **MIS (Masked Importance Sampling)**：当 IS ratio 异常时直接 mask 掉该 token（不参与 gradient）。参考 IcePop (https://ringtech.notion.site/icepop, [19])。
- **R3 (Rollout Routing Replay)** (https://arxiv.org/abs/2510.11370, [20])：rollout 时记录每个 token 实际走的 expert id，training 时强制 replay 同样的 routing，从根本上消除 routing mismatch。
- **slime** (https://github.com/THUNLP-MT/slime, [17]) 是专门为 MoE routing consistency 设计的 framework。

Paper 实验里 Qwen3-30B MoE 只用 TIS 就把 KL 控制在可接受范围，没有触发 R3。但 paper 暗示更不稳定场景需要 R3。

---

## 5. 实验：Qwen3-8B Dense

### 5.1 Setup

- Model: Qwen3-8B-Base (https://huggingface.co/Qwen/Qwen3-8B-Base)
- Algorithm: DAPO (https://arxiv.org/abs/2503.14476, [10]) —— Qwen 团队的 PPO 变种，去掉了 KL penalty 到 reference model，加了 dynamic sampling 和 overlong filtering
- Benchmark: AIME24 (American Invitational Mathematics Examination 2024)
- Hardware: 8×H100
- Training: FSDP backend
- Rollout: vLLM
- Prompt batch size 32, n=16 responses/prompt → rollout batch 32×3×16 = 1536（×3 来自某种 group/packing，paper 没说清，可能是 3 个 sampling 重复）
- Training batch size = PPO mini-batch size = 32（确保 policy 只被 rollout outputs 更新一次，隔离 quantization 影响）
- Max response length 20K tokens
- TIS C=2

### 5.2 三个配置

1. **BF16 baseline without TIS** (orange)
2. **FP8 W8A8 + token-level TIS** (blue)
3. **FP8 W8A8 without TIS** (green, ablation)

### 5.3 关键观察 (Figure 2)

| Metric | BF16 baseline | FP8 + TIS | FP8 no TIS |
|---|---|---|---|
| Validation accuracy (AIME24) | 上升并稳定 | 与 baseline 对齐 | **明显退化** |
| Reward | 上升 | 与 baseline 重合 | 退化 |
| Response length | 增长（学习长 reasoning） | 同步增长 | 异常 |
| Mismatch KL $D_{KL}(\pi_\theta^{FP8} \| \pi_\theta)$ | ~0 | 稳定但 > 0 | 高且增长 |

**Intuition**：FP8 without TIS 的退化证明 quantization 引入的 mismatch 不是"白噪声"，而是有偏的——它系统性地让某些 tokens 的概率被低估/高估，累积到 reward signal 上就是有偏 gradient。

### 5.4 Rollout 性能 (Figure 3)

- FP8 比 BF16 快 **10-20%**
- 长序列收益更大（因为长序列 memory bandwidth bound 更严重，FP8 的 2× memory 节省直接转化为 wall-clock）

注意 8B 模型 GEMM 本身的 arithmetic intensity 不算高，FP8 tensor core 加速有限。20K context 下 KV-cache 成为瓶颈，但本实验没有开 KV-cache FP8（下一节才开）。

---

## 6. 实验：Qwen3-30B-A3B MoE

### 6.1 Setup

- Model: Qwen3-30B-A3B-Base (30B total params, 3B activated params per token)
- Hardware: 2×8×H100 = 16 卡
- 配置：BF16 + TIS (orange) vs FP8 + TIS (blue)——注意 MoE 即使 BF16 也加 TIS，因为 MoE 本身就有 routing mismatch

### 6.2 关键观察 (Figure 4)

- Validation accuracy / reward / response length：FP8 + TIS 与 BF16 + TIS 高度对齐
- **Mismatch KL 持续增长**（dense 模型是稳定）：MoE 的 expert routing 累积不一致导致
- TIS 仍然能控制住，accuracy 没崩

### 6.3 Rollout 性能 (Figure 5)

- FP8 vs BF16：**30-50% speedup**（是 8B dense 的 2-3 倍）

### 6.4 为什么 MoE 收益大很多

Paper 给出三个因素：

1. **Arithmetic intensity 更高**：30B 模型的 GEMM 矩阵大，FP8 tensor core 利用率高，绝对时间节省大。
2. **Memory traffic 减半更值钱**：30B BF16 是 60 GB，FP8 是 30 GB。每步 weight sync + 每 token weight loading 都受益。
3. **KV-cache 容量释放**：30B weights FP8 化省下 ~30 GB HBM，可以装更多 KV-cache，提升 vLLM 并发，减少 preemption。

第三个因素是**复合效应**：vLLM 的 preemption 是性能杀手。一旦开 FP8 把 preemption 率降下来，throughput 不是线性增长而是阶跃式提升。

---

## 7. FP8 KV-Cache Quantization

### 7.1 动机

Long context rollout 是 **memory-bound on KV-cache**，不是 compute-bound。20K context 下：
- KV-cache size (Qwen3-8B, GQA) ≈ `2 × num_layers × seq_len × num_kv_heads × head_dim × batch_size × 2 bytes`
- 大 batch × 20K seq 下 KV-cache 可以达到几十 GB
- vLLM 的 PagedAttention (https://arxiv.org/abs/2309.06180, [3]) 在 KV-cache 满时会 preempt 请求（swap 到 CPU 或 recompute）

KV-cache FP8 翻倍容量，直接缓解 preemption。

### 7.2 两种 calibration 范式 (Figure 6)

**Inference-Side calibration**（veRL 实现）：
- 利用 vLLM 原生的 `calculate_kv_scales` 机制
- vLLM 通常在第一次 forward 时算 QKV scales
- RL 中每步 weight 变了，所以 **每步 rollout 前强制 reset 这些 flags**，让 vLLM 重新算 scales
- 无需外部干预，耦合度低

**Trainer-Side calibration**（NeMo-RL 实现）：
- 训练 backend 在每步结束后用训练数据子集（prompts + generated responses）算 QKV scales
- 然后把 scales sync 到 inference engine
- 控制更细，但有 ~2-3% step time 的 calibration overhead
- NeMo-RL: https://github.com/NVIDIA/NeMo

### 7.3 QKV scale 是什么

KV-cache 量化不是直接把 K 和 V 转 FP8，而是：

$$\hat{K} = \text{round}(K / s_K) \in \text{FP8}, \quad \hat{V} = \text{round}(V / s_V) \in \text{FP8}$$

其中 $s_K, s_V$ 是 per-head 或 per-tensor 的 scale。这些 scale 需要从一个 calibration set 估算，通常是 max(|K|), max(|V|)。

由于 K 和 V 在 attention 计算中是 `softmax(QK^T / sqrt(d)) V`，scale 不准会直接破坏 attention 分布。所以 per-step recalibration 是必要的——每步 weight 更新后 K, V 的数值分布会变。

### 7.4 实验 (Figure 7, 8)

Qwen3-8B 上的四种配置：

| Configuration | Speedup | Mismatch KL |
|---|---|---|
| BF16 baseline | 0% | ~0 |
| Linear W8A8 only | ~20% | 低 |
| KV-cache FP8 only | ~38% | 略高于 Linear |
| Full FP8 (Linear + KV + Attention) | **44%** | 最高但仍稳定 |

**关键发现**：KV-cache FP8 单独就给 38% speedup，比 Linear 的 20% 还大！

**为什么 KV-cache 收益大**：8B 小模型在 20K long context 下，vLLM 监控显示 **BF16 baseline 和 Linear W8A8 配置都频繁触发 preemption**。开 KV-cache FP8 后容量翻倍，preemption 频率大幅下降，throughput 阶跃式提升。这是 **memory-bound workload 释放并发**的典型效应，不是简单的 2× speedup。

### 7.5 Full FP8 的 mismatch KL 最高但仍稳定

Full FP8 = Linear + KV-cache + Attention 都 FP8。三层 quantization error 叠加，KL 最大，但 TIS 仍然能 hold 住 accuracy。这是 paper 的乐观结论：**aggressive full-stack FP8 + TIS 是 viable 的**。

---

## 8. End-to-End FP8 (Section 2.4)

之前实验都是 training 用 BF16，rollout 用 FP8。这里探索 **training + rollout 都用 FP8**。

### 8.1 三个配置

1. BF16 training + BF16 rollout (baseline)
2. BF16 training + FP8 rollout (前面 Section 2.1 的方案)
3. **FP8 training + FP8 rollout** (end-to-end FP8)

### 8.2 结果 (Figure 9)

- **Accuracy**：FP8 training + FP8 rollout 与 BF16 baseline 对齐
- **Mismatch KL**：FP8 training + FP8 rollout 比 FP8 rollout-only 更低（precision alignment 减少了 distribution drift）
- **Training 侧 speedup**：~20%

**Intuition**：当 trainer 用 BF16 但 rollout 用 FP8，两者数值上 systemic 偏移。Trainer 算出的 gradient 是针对 BF16 π_θ，但 rollout 来自 FP8 π_θ^FP8，gradient 应用回 BF16 π_θ 会让它进一步偏离 FP8 π_θ^FP8，形成正反馈。End-to-end FP8 让两者数值路径一致，distribution drift 自然小。

不过 paper 也指出 end-to-end FP8 仍有 mismatch（kernel-level 差异、quantization/dequantization round-trip loss），所以**不能完全消除**，TIS 仍然需要。

参考 [22] (https://arxiv.org/abs/2510.26788, Qi et al., "Defeating the Training-Inference Mismatch via FP16")：用 FP16 而不是 BF16 也能减小 mismatch，因为 BF16 的 7-bit mantissa 精度比 FP16 的 10-bit mantissa 低，autoregressive decoding 累积误差更大。这是另一个角度的 precision alignment。

### 8.3 FP8 training 的技术基础

参考 [13] (https://arxiv.org/abs/2310.18361, FP8-LM, Peng et al.)：FP8 训练在 pretraining/SFT 上已被验证 2× speedup 且不损失 convergence。DeepSeek-V3 是工业级首个 FP8 训练成功案例。

FP8 training 的关键是：
- **Weights**: E4M3, blockwise 128×128
- **Activations**: E4M3, per-tile 1×128
- **Gradients**: E5M2（更大动态范围，因为 gradient 有 outlier）
- **Master weights**: 保留 BF16/FP32 副本用于 update
- **Optimizer states**: FP32（Adam 的 m, v 不能低精度）

NeMo-RL 在 paper 中承担 end-to-end FP8 实验，因为 NeMo 原生支持 Megatron-LM 的 FP8 training。

---

## 9. Related Work 的脉络梳理

### 9.1 FP8 Quantization

- [12] Micikevicius et al. 2022: FP8 format 定义
- [13] FP8-LM 2023: 系统 FP8 训练方法
- [8] DeepSeek-V3 2024: 工业级 FP8 训练，fine-grained quantization

### 9.2 RL Systems

- [14] DeepSpeed-Chat: 早期同步框架，资源利用率低
- [15] OpenRLHF (https://github.com/OpenRLHF/OpenRLHF): Ray 分布式 + vLLM
- [9] veRL (https://github.com/volcengine/verl): hybrid flow，多 backend 解耦
- [11] NeMo-RL: NVIDIA 栈，Megatron 紧耦合，原生 FP8
- [16] ROLL: 硬件感知 workload mapping
- [17] slime: SGLang 原生，MoE routing 优化
- [18] AReaL (https://arxiv.org/abs/2505.24298): 异步 RL，generation 和 training 完全解耦

### 9.3 Train-Inference Mismatch

- [5] RL collapse (https://richardli.xyz/rl-collapse): 系统性分析 mismatch 导致的 collapse
- [6] FlashRL (https://fengyao.notion.site/flash-rl): 8-bit rollouts
- [7] Off-policy RL (https://fengyao.notion.site/off-policy-rl): 高效 RL 框架的 off-policy 本质
- [19] IcePop (https://ringtech.notion.site/icepop): MoE 上的 masked IS
- [20] R3 (https://arxiv.org/abs/2510.11370): routing replay
- [21] Bitwise consistency (https://blog.vllm.ai/2025/11/10/bitwiseconsistent-train-inference.html): vLLM + TorchTitan 的 bitwise 一致性 RL
- [22] FP16 defeating mismatch (https://arxiv.org/abs/2510.26788): precision 选择对 mismatch 的影响

[21] 的 bitwise consistency 路线和 FP8-RL 的 importance sampling 路线是 **互补的两种哲学**：前者追求消除 mismatch（更难，限制多），后者接受 mismatch 但算法上修正（更灵活，但引入 bias）。

---

## 10. 实操配置 (Appendix A, B)

### 10.1 veRL 启用 FP8 W8A8

```bash
actor_rollout_ref.rollout.quantization=fp8
actor_rollout_ref.rollout.name=vllm
```

### 10.2 性能优化必备

- CUDA 12.9+
- DeepGEMM (https://github.com/deepseek-ai/DeepGEMM)：DeepSeek 开源的 FP8 GEMM 库
  - vLLM 0.11+ / SGLang 0.55+ 默认启用
  - 旧版需 `VLLM_USE_DEEP_GEMM=True`
- DeepGEMM 是 cutlass-based，针对 Hopper 的 FP8 tensor core 优化，比 cuBLAS 的 FP8 GEMM 快 1.5-2x

### 10.3 KV-cache FP8 (veRL Inference-Side)

```yaml
actor_rollout_ref:
  rollout:
    quantization:
      kv_cache_dtype: fp8_e4m3
      calculate_kv_scales: True
```

### 10.4 KV-cache FP8 (NeMo-RL Trainer-Side)

```yaml
policy:
  generation:
    vllm_cfg:
      precision: fp8
      kv_cache_dtype: fp8
```

Trainer-side 有 2-3% step time overhead，但 calibration 数据可控。

---

## 11. 我的 Intuition 总结与几个值得深挖的点

### 11.1 这篇 paper 的真正贡献

工程上，paper 的核心是 **"FP8 rollout + TIS" 这个组合的实证可行性**。它没有发明新算法，但把几个已知 piece 拼起来跑通了端到端：
1. Blockwise FP8 (DeepSeek-V3)
2. Dynamic weight sync (veRL 的工程)
3. TIS (经典 IS 修正)
4. KV-cache FP8 (vLLM 原生)

算法上唯一新意是 **per-step QKV scale recalibration** 的两种 calibration 范式对比，但其实都是工程实现，不是新算法。

### 11.2 MoE 的 routing mismatch 是更深层的问题

Paper 在 MoE 实验里观测到 mismatch KL 持续增长但 TIS 还能 hold，这是侥幸还是 robust？我倾向于**侥幸**——TIS 是 token-level reweighting，但 routing 错配会让 token 落到完全不同的 expert，输出分布差异巨大，TIS 的 clip 反而 mask 掉了问题。

R3 (Rollout Routing Replay) 是更彻底的解法，但 paper 没有用。值得追问：在什么 model size / training length 下 TIS 会 break？

### 11.3 End-to-end FP8 的 trade-off

End-to-end FP8 减小 mismatch 但牺牲了 trainer 侧的 numerical precision。对 RL 来说，gradient 的精度对稳定性很关键（PPO 的 advantage 估计对 noise 敏感）。Paper 没有展示 long training run (>1000 steps) 的稳定性，这是缺失的实验。

### 11.4 下一步：NVFP4 和更长 context

Paper future work 提到 NVFP4（4-bit floating point，Blackwell 架构原生）。NVFP4 的 dynamic range 更小，需要更精细的 per-block scaling（Microscaling formats, MXFP4）。Reported instability from accumulated quantization error 是关键担忧。

Multi-turn / agentic RL 的 long context 进一步放大 KV-cache 压力，KV-cache 量化收益会更大。

### 11.5 与 bitwise consistency 路线的对比

vLLM + TorchTitan (https://blog.vllm.ai/2025/11/10/bitwiseconsistent-train-inference.html, [21]) 走的是另一极端：完全消除 mismatch，让 trainer 和 rollout 的 numerics bitwise 一致。这样不需要 TIS，但限制严格（不能用 FP8 rollout，因为 trainer 是 BF16）。

我猜测未来的方向是 **混合路线**：critical path（采样、reward 计算）bitwise 一致，non-critical path（GEMM、attention）允许 FP8 + IS 修正。

---

## 12. 关键 References 汇总

- **Paper itself**: NVIDIA FP8-RL (本篇)
- **FP8 Formats**: https://arxiv.org/abs/2209.05433
- **FP8-LM**: https://arxiv.org/abs/2310.18361
- **DeepSeek-V3**: https://arxiv.org/abs/2412.19437
- **veRL / HybridFlow**: https://github.com/volcengine/verl, https://arxiv.org/abs/2409.19256
- **vLLM / PagedAttention**: https://github.com/vllm-project/vllm, https://arxiv.org/abs/2309.06180
- **SGLang**: https://github.com/sgl-project/sglang
- **DeepGEMM**: https://github.com/deepseek-ai/DeepGEMM
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM
- **NeMo**: https://github.com/NVIDIA/NeMo
- **DAPO**: https://arxiv.org/abs/2503.14476
- **Qwen3-8B-Base**: https://huggingface.co/Qwen/Qwen3-8B-Base
- **Qwen3-30B-A3B-Base**: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
- **OpenRLHF**: https://github.com/OpenRLHF/OpenRLHF
- **AReaL**: https://arxiv.org/abs/2505.24298
- **RL Collapse (Liu et al.)**: https://richardli.xyz/rl-collapse
- **FlashRL**: https://fengyao.notion.site/flash-rl
- **Off-policy RL Frameworks**: https://fengyao.notion.site/off-policy-rl
- **IcePop (MoE IS)**: https://ringtech.notion.site/icepop
- **R3 (Rollout Routing Replay)**: https://arxiv.org/abs/2510.11370
- **FP16 defeating mismatch**: https://arxiv.org/abs/2510.26788
- **Bitwise Consistency (vLLM + TorchTitan)**: https://blog.vllm.ai/2025/11/10/bitwiseconsistent-train-inference.html
- **Seer (RL bottleneck analysis)**: https://arxiv.org/abs/2511.14617
- **slime**: https://github.com/THUNLP-MT/slime

---

## 13. 一句话总结

FP8-RL 把 DeepSeek-V3 的 blockwise FP8 quantization + vLLM 的 KV-cache FP8 + 经典 importance sampling correction 三个 piece 拼成一套 **production-ready 的 RL rollout stack**，关键 insight 是 **per-step weight re-quantization + per-step QKV scale recalibration + token-level TIS** 这三件套共同维持稳定性。Qwen3-8B 上 44% speedup，Qwen3-30B MoE 上 30-50%，且 accuracy 与 BF16 baseline 对齐——前提是你别忘了开 TIS。

如果你想 build deeper intuition，我推荐先读 DeepSeek-V3 报告的 FP8 部分（blockwise scaling 的 rationale）、然后读 [5] RL Collapse（理解 mismatch 为什么致命）、最后读 vLLM bitwise consistency blog（理解另一条路线的极端）。FP8-RL 这篇 paper 本质上是 engineering integration paper，algorithm novelty 有限，但 **integration depth** 和 **ablation 完整度** 是它的价值所在。
