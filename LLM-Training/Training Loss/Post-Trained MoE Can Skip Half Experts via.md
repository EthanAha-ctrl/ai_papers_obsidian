---
source_pdf: Post-Trained MoE Can Skip Half Experts via.pdf
paper_sha256: ae1c140a5a3d1caefb11e23107524064c3c55792288ab6727234296ca3ce970d
processed_at: '2026-08-06T05:23:42-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ZEDA

## 这篇 paper 到底在干嘛

工业界手里一堆已经训练好的 MoE 模型（Qwen3-30B-A3B、GLM-4.7-Flash 这种），花了几千张 GPU 跑过 pre-training + SFT + RL + on-policy distillation，routing 分布已经收敛到一个非常精细的 equilibrium。你现在想让它推理更快——能不能**花 30 个小时**，把 expert 计算砍掉一半，性能基本不掉？

以前做 dynamic MoE 的工作都要 from scratch 重训，或者大改 router 架构，对 post-trained 模型动这种手术基本等于毁掉它。ZEDA 就是给这个场景设计的一套**微创手术方案**。

---

## 方法：两个 step，简单粗暴

### Step 1: 塞 Zero Expert

MoE 模型每一层有一堆 expert（Qwen3 是 128 个，每次激活 8 个）。ZEDA 干的事就是**往 expert pool 里再塞 64 个"假 expert"**，这些 expert 的输出永远等于 0。

$$Z_j(h) = 0, \quad \forall h, \forall j$$

router 还是从 `128+64=192` 个候选里选 top-8，但如果它选到了 zero expert，那个 slot 就等于"跳过"，不消耗 FFN FLOPs。

**核心直觉**：你完全不动原来 128 个 normal expert 的参数，也不重训 router 已有的权重，只是在 router 输出端多加 64 个 logits。这是最小创伤的架构改动。

router 初始化也很有讲究——新加的 64 个 zero expert 的 router 参数从 Gaussian 采样，mean 和 variance 匹配本层原有 router logit 的 scale。目的是**不打乱 router 已经校准好的 logit 量级**。

### Step 2: 两阶段 Self-Distillation

用原始的 post-trained MoE 当**冻结的 teacher**，让塞了 zero expert 的 student 模型学着模仿它。

**Stage 1: SFT**——teacher 采样 response，student 学着 predict 这些 response 的 token。

**Stage 2: On-Policy Distillation (OPD)**——student 自己采样 response，teacher 在同样的 trajectory 上给 token-level KL target。

为什么非要两阶段？Table 5 的 ablation 很说明问题：
- 只用 SFT: 73.6 avg
- 只用 OPD: 72.9 avg（最差）
- SFT → OPD: 74.2 avg

OPD 单独用效果差，是因为 router 还没稳定下来，OPD 同时要学 routing 策略 + 生成 coherent response，难度太大。SFT 先把 router 从 128 候选拓展到 192 候选后的行为稳定住，OPD 再来收 distribution gap。

---

## Group Auxiliary Loss: 最聪明的设计

### 问题：原始 load balancing loss 会毁掉 post-trained 模型

标准 MoE 的 auxiliary loss 强制所有 expert 均匀负载：

$$\mathcal{L}_A = \alpha \cdot \frac{N + N_Z}{K} \cdot \sum_{i=1}^{N+N_Z} f_i \cdot P_i$$

但 post-trained MoE 的 routing 是 **非均匀、input-dependent** 的——某些 expert 专长数学，某些专长 code。你强行让它均匀，specialization 就毁了。

Table 10 实验直接验证：用 $\mathcal{L}_A$ 替换 $\mathcal{L}_{GA}$，Qwen3 在 math benchmark 上从 81.0 跌到 59.5，AIME24 从 78.1 跌到 39.6。**崩盘**。

### 解法：只在 group 边界做平衡

把 128 个 normal expert 看作 group $\mathcal{E}$，64 个 zero expert 看作 group $\mathcal{Z}$，**只在两个 group 之间做负载平衡，group 内部完全不管**：

$$\mathcal{L}_{GA} = \alpha \cdot \frac{N + N_Z \cdot w}{K} \cdot \left( \frac{f_\mathcal{E} \cdot P_\mathcal{E}}{N} + \frac{f_\mathcal{Z} \cdot P_\mathcal{Z}}{N_Z \cdot w} \right)$$

变量里 $w$ 是 zero expert group 的相对权重，是控制 quality-efficiency trade-off 的核心 knob。最小化这个 loss 的均衡点满足：

$$r_{ZE}^{\text{target}} = \frac{N_Z \cdot w}{N + N_Z \cdot w}$$

Qwen3 取 N=128, $N_Z$=64, w=2，目标 $r_{ZE}$ = 50%。

**直觉**：normal expert 内部那 128 个之间的 routing 分布，post-training 已经调好了，你完全别碰；你只管"这一批 token 有多少比例应该走 zero expert（即跳过计算）"。这避免了 expert-level uniform 把 specialization 平均掉的灾难。

---

## 实验结果：又快又好

### 主表 (Table 1)

Qwen3-30B-A3B 上 11 个 benchmark（5 math + 4 code + 2 IF）：

| Method | Avg Acc | $r_{ZE}$ | 故事 |
|---|---|---|---|
| Original | 74.9 | 0.0 | baseline |
| AdaMoE | 54.8 | 51.9 | 数学崩盘（AIME24=25.0） |
| Dynamic Skipping | 68.1 | 43.8 | code 崩盘（MBPP+=70.0） |
| NET$_{SFT→OPD}$（直接砍 K=4） | 73.0 | 50.0 | naive 截断 |
| **ZEDA** | **74.2** | **51.2** | 只掉 0.7 分 |

几个关键 takeaway：

1. **AdaMoE 和 Dynamic Skipping 都呈现严重 capability imbalance**——一个数学崩一个 code 崩。说明对 post-trained 模型强行施加 dynamic routing 会破坏特定 capability。ZEDA 是唯一在所有 domain 都保持竞争力的方法。

2. **ZEDA 比 NET$_{SFT→OPD}$ 高 1.2 分**。NET 是直接把 K 从 8 砍到 4（static truncation），ZEDA 是 dynamic——不同 token 激活不同数量的 expert。这证明 token-level dynamism 真的有用，不是简单的"少算就完了"。

3. **IFBench 上 ZEDA 反超 original**（42.3 vs 39.7）。Zero expert 像一个免费的 regularizer。

### 成本 (Table 2)

8×H200 GPU 上：
- Qwen3-30B-A3B: 30 小时（其中 OPD 占 20h）
- GLM-4.7-Flash: 62 小时

对比 pre-training + post-training 动辄几千上万小时，这成本几乎可以忽略。

### OOD 泛化 (Table 7)

MMLU-Redux 和 GPQA-Diamond 完全 OOD（不在 SFT 数据分布内）：
- Qwen3: 76.2 vs original 76.7，$r_{ZE}$ 还有 47.2%
- GLM: 72.9 vs original 76.1，$r_{ZE}$ 还有 50.0%

**router 学到的是通用的 token-level computation allocation，不是 in-domain 过拟合**。

---

## 最有意思的发现：Emergent Computation Allocation

Section 4.1 我觉得是整篇 paper 最有洞察的部分。他们拿 110 个 prompt 做 rollout，记录每个 token 的 $r_{ZE}$、student log probability、entropy，再算 teacher logp 得到 $\Delta_{logp}$。

### 三条发现

**1. $r_{ZE}$ 与 $\Delta_{logp}$ 负相关**（Figure 3 left）

teacher-student 分布差异越大，student 越倾向于激活更多 normal expert。也就是说 student "知道自己跟 teacher 差得远的时候，会多花点算力"。

**2. $r_{ZE}$ 与 student entropy 负相关**（Figure 3 right）

student 自己 entropy 高（对 next token 不确定）→ $r_{ZE}$ 低（激活更多 normal expert）。

**3. $r_{ZE}$ 与 task difficulty 无关**（Table 3）

MATH-500 五个 difficulty level 的 $r_{ZE}$ 几乎一样（51.1~52.5），跟 AIME24 的 52.1 也接近。**模型不根据"任务难不难"分配算力，而是根据"token-level 特征"分配**。

### Response pattern 揭示的有趣现象（Figure 4, 5）

逐 token 可视化 $r_{ZE}$：
- **code fragment / math expression**: $r_{ZE}$ 高（少算）
- **natural language / reasoning text**: $r_{ZE}$ 低（多算）

这乍看反直觉——code/math 不是"更难"吗？但仔细想想就通了：code/math token 通常是**低 entropy、高确定性**的（比如 `for` 后面大概率是 `(`，`x` 后面大概率是 `=`），student 自己就能很确定地预测，$\Delta_{logp}$ 也小，所以可以放心激活 zero expert。而自然语言推理 token 是高 entropy 的、需要更多 expert 计算去确定 next token。

### 三个观察的统一解释

paper 给的解释非常 elegant：

$$\Delta_{logp} \uparrow \implies \mathcal{L}_{SFT}, \mathcal{L}_{OPD} \uparrow \implies \mathcal{L}_{GA} \text{ 相对 weight } \downarrow \implies r_{ZE} \downarrow$$

当 token 的 teacher-student gap 大时，task loss 主导，$\mathcal{L}_{GA}$ 想把 $r_{ZE}$ 拉高的力被压制，model 自然选择激活更多 normal expert 去缩小 gap。

再加上 entropy 和 $\Delta_{logp}$ 的关联（Ko et al. 2026），以及 code/math token 倾向低 entropy（Wang et al. 2025 "High-entropy minority tokens"）——三个观察被统一解释。

**这是一个非常漂亮的 emergent behavior**：ZEDA 没有显式编码 "how much computation to allocate"，但训练动力学自然让 model 学到了与 model uncertainty + distributional gap 相关的 allocation policy。

---

## Zero Expert vs Copy Expert: 一个反直觉的故事

Appendix B 讨论了另一种 zero-computation expert 设计——copy expert，定义为 $Z_j(h) = h$（identity mapping）。看起来也很便宜，就一个加法。

$$\tilde{y}_{copy} = \sum_{i \in \tilde{S} \cap \mathcal{E}} \tilde{g}_i(h) \cdot E_i(h) + \sum_{j \in \tilde{S} \cap \mathcal{Z}} \tilde{g}_j(h) \cdot h$$

后一项 $\sum_j \tilde{g}_j(h) \cdot h$ 不是 no-op，而是 input 的加权求和。

Table 9 实验结果触目惊心：

| Method | Avg Acc | AIME24 | AIME25 | AIME26 |
|---|---|---|---|---|
| Original | 82.8 | 80.9 | 71.0 | 72.3 |
| Copy Expert + $\mathcal{L}_{GA}$ | **20.7** | 1.0 | 2.9 | 0.8 |
| Zero Expert + $\mathcal{L}_{GA}$ | 81.0 | 78.1 | 66.2 | 71.3 |

copy expert 在 AIME 上几乎完全崩盘。

Figure 7 给出诊断：
- **Scale mismatch**：copy expert 输出与 original output 的 L2 距离远大于 zero expert。copy component 自己就是主因。
- **Direction mismatch**：copy expert 输出与 original 的 cosine similarity 显著低于 zero expert，且随 depth 增长而恶化。

**直觉**：copy expert = $\tilde{g}_j(h) \cdot h$ 引入了一个与原 MoE residual 分支完全不同 scale 和 direction 的信号，相当于往 model 里塞了一个 uncalibrated 的 skip connection。post-trained model 的 attention 和后续 layer 期待的是 expert residual 的特定 scale，强行注入这个新信号会破坏整个 flow。

zero expert 就干净多了——它就是真正的 no-op，不引入任何新信号，只是让那个 slot "消失"。

---

## 理论 FLOPs 分析：为什么 speedup 会随 length decay

Appendix D 给了完整的 FLOPs 推导。核心结论：

### Prefill Stage

每层 FLOPs（Qwen3 配置）：

$$F_{orig}^{pre} = \underbrace{4l^2 H_{attn}}_{\text{attention } O(l^2)} + \underbrace{4(1+g_{kv}) l H H_{attn}}_{\text{QKV+O proj}} + \underbrace{6KlHH_e}_{\text{MoE FFN}} + \underbrace{2NlH}_{\text{router}}$$

$$F_{ZEDA}^{pre} = 4l^2 H_{attn} + 4(1+g_{kv}) l H H_{attn} + 6(1-r_{ZE})KlHH_e + 2(N+N_Z)lH$$

ZEDA 压缩的是 MoE FFN 项（被 $1-r_{ZE}$ 折半），router 项轻微增加（N → N+N_Z），attention 项完全不变。

**关键**：attention 的 $O(l^2)$ 项随 length 主导整个 FLOPs，MoE 占比下降，所以 speedup 随 length decay。

### Decode Stage

decode 时 attention 的 $O(l^2)$ 项系数是 prefill 的一半（$2l(l-1)$ vs $4l^2$，因为每步只处理 1 个新 token + $t-1$ 个 cached token），所以 MoE FLOPs 占比更高，**decode speedup > prefill speedup**。

### 理论 vs 实测 (Table 13)

| Length | Prefill 理论 | Prefill 实测 | Decode 理论 | Decode 实测 |
|---|---|---|---|---|
| 1024 | 1.403x | 1.141x | 1.443x | 1.233x |
| 4096 | 1.261x | 1.203x | 1.341x | 1.236x |
| 8192 | 1.178x | 1.175x | 1.261x | 1.185x |

实测比理论低 ~10%，gap 来自 router overhead、memory bandwidth、kernel launch 等。但趋势完全匹配。

---

## 总结：ZEDA 的 ROI

ZEDA 给我的最大直觉是：**post-trained MoE 的 routing 分布是一个已经收敛到精细 specialization 的 equilibrium**。任何 dynamic MoE 方法只要打破这个 equilibrium——expert-level uniform、强行改变 K、router 重训——都会引发严重退化。

ZEDA 的妙处在于四点 minimal：

1. **Minimal architectural change**：加 zero expert 不动 normal expert 参数、不重训 router 已有权重，只在 router 输出端加 $N_Z$ 个新 logits
2. **Minimal constraint**：Group Auxiliary Loss 只在 group 边界做平衡，normal expert 内部路由分布原封不动
3. **Minimal training signal**：Self-distillation 用 teacher 自己教 student，避免引入外部 capability drift
4. **Emergent allocation**：Router 自己学到与 entropy / distributional gap 相关的 token-level allocation policy，没有人为编码任何 "difficulty heuristic"

最终用 30 小时（Qwen3-30B-A3B）换来：
- **50%+ expert FLOPs 减少**
- **~20% inference speedup**（8k context）
- **0.7 avg acc 损失**（OOD 上几乎无损）
- **部分 benchmark 反超 original**（IFBench）

这是一个 ROI 极高的工程方案，对工业界部署 post-trained MoE 模型有直接价值。

---

## 一些可以联想的方向

1. **Speculative decoding 与 zero expert 的关联**：两者都是 "skip compute when confident"，能否统一成一个 framework？zero expert 本质上就是 model 内部的 "speculative skip"。

2. **MoE quantization + ZEDA 的 synergy**：zero expert 已经省了 50% FLOPs，再叠 INT8/INT4 quantization 能否进一步压缩？两者是否会有 interference？

3. **Router 的 interpretability**：分析 zero expert 的 routing pattern 能否揭示 post-trained MoE 学到了什么 capability specialization？比如哪些 layer 的 expert 更"可跳过"？

4. **Per-layer $r_{ZE}$ 设置**：现在所有 layer 用同一 $w$，但深层和浅层可能需要不同 allocation policy。Figure 5 显示 $r_{ZE}$ 跨 layer 比较均匀，但这是 uniform $w$ 训出来的结果——如果允许 per-layer $w$ 优化，能否进一步压缩？

5. **DeepEP 集成**：paper 提到未来可与 DeepEP（https://github.com/deepseek-ai/DeepEP）这种 expert-parallel 通信库配合，可能让 speedup 翻倍。zero expert 减少激活 expert 数量直接降低 all-to-all 通信量，这是另一个效率提升维度。

6. **Agentic workload 下的行为**：paper 自己承认没测长 horizon agentic 任务。在 tool calling、multi-turn reasoning 下，zero expert 的 allocation policy 会不会失效？这可能是个值得探索的方向。

---

参考链接汇总：
- ZEDA paper: Tsinghua C3I/ZEDA group
- slime framework: https://github.com/THUDM/slime
- SGLang: https://github.com/sgl-project/sglang
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Thinking Machines OPD blog: https://thinkingmachines.ai/blog/on-policy-distillation
- Qwen3 技术报告: https://arxiv.org/abs/2505.09388
- GLM-4.5: https://arxiv.org/abs/2508.06471
- MoE++: https://arxiv.org/abs/2410.07348
- LongCat-Flash: https://arxiv.org/abs/2509.01322
- AdaMoE: https://aclanthology.org/2024.findings-emnlp.367
- MiniLLM: https://arxiv.org/abs/2306.08543
- GKD: https://arxiv.org/abs/2306.13649
- DeepEP: https://github.com/deepseek-ai/DeepEP

---

# ZEDA: 把 Post-Trained MoE 变成 Dynamic MoE 的 Self-Distillation 框架

## 1. 这篇 paper 在解决什么问题？

一个核心痛点：当前的 dynamic MoE 工作（MoE++, LongCat-Flash, AdaMoE, Ada-K Routing 等）几乎都要 from-scratch pre-training 或大幅改造 router，**但工业界手里都是已经花了海量 GPU 跑过 pre-training + SFT + RL + on-policy distillation 的 post-trained MoE**。直接对它们动手术风险极大——会破坏已经精心校准过的 routing 分布和 capability 分布。

ZEDA 的切入点：**能否在 ~30 小时（Qwen3-30B-A3B）成本下，把一个 post-trained static MoE 转成 dynamic MoE，砍掉一半 expert FLOPs，同时不掉点甚至部分 benchmark 还涨？**

这正是 paper 标题 "Post-Trained MoE Can Skip Half Experts via Self-Distillation" 的来源。

参考链接：
- 原文 ZEDA: https://arxiv.org/abs/ (Tsinghua C3I/ZEDA group)
- MoE++ 原始工作: https://arxiv.org/abs/2410.07348
- LongCat-Flash 技术报告: https://arxiv.org/abs/2509.01322
- Qwen3: https://arxiv.org/abs/2505.09388
- GLM-4.5: https://arxiv.org/abs/2508.06471

---

## 2. 方法核心架构

ZEDA 由两个 step 组成（Figure 1）：

### Step 1: Zero-Expert Injection (架构层)

把每个 MoE module 的 expert pool 从 `|ℰ| = N` 扩展为 `|ℰ'| = N + N_Z`，新增的 `N_Z` 个 zero experts 满足：

$$Z_j(h) = 0, \quad \forall h, \forall j \in \{1, \dots, N_Z\}$$

**关键直觉**：zero expert 恒等于 0，意味着 routing 到它就等价于"跳过这个 expert slot"，**不需要修改 expert 的 FFN 参数**。top-K 预算 K 保持不变，但被 zero experts 占用的 slot 不消耗 FFN FLOPs。

原始 MoE 输出：

$$y(h) = \sum_{i \in S(h)} g_i(h) \cdot E_i(h), \quad S(h) \subseteq \mathcal{E}, |S(h)| = K$$

注入后：

$$\tilde{y}(h) = \sum_{i \in \tilde{S}(h) \cap \mathcal{E}} \tilde{g}_i(h) \cdot E_i(h) \tag{Eq.1}$$

变量含义：
- $S(h)$, $\tilde{S}(h)$: 分别是 original 和 augmented top-K 集合
- $g_i(h)$, $\tilde{g}_i(h)$: normalized routing weight（softmax over router logits）
- $E_i(h)$: 第 $i$ 个 normal expert 的 FFN(h) = down(act(up(h) * gate(h)))
- $\tilde{S}(h) \cap \mathcal{E}$: 从 top-K 中过滤掉 zero experts

**重要细节（paper 里隐式强调）**：不 renormalize routing weights。即若 top-8 选了 [E_3, E_7, Z_2, Z_5, ...]，剩下 6 个 normal expert 的 routing weight 不重新归一化到 sum=1。这是因为原模型 routing weight 的 sum 在 pre-training 期间已经被 calibrated 到一个特定 output magnitude，renormalize 会人为放大 residual 分支的有效 scale（Table 6 ablation 验证：renorm 会掉 1.7 个点 avg acc）。

**Router initialization**：原来的 N 个 normal expert router 参数保留；新加的 N_Z 个 zero expert 的 router 参数从 Gaussian 分布采样，mean 和 variance 与本 module 原 router logits 匹配——保持 router logit 的 scale。

### Step 2: Two-Stage Self-Distillation (训练层)

**Stage 1: SFT (off-policy)**

$$\mathcal{L} = \mathcal{L}_{SFT} + \mathcal{L}_{GA} = -\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_T(\cdot|x)} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t}) \right] + \mathcal{L}_{GA} \tag{Eq.2}$$

变量：
- $\pi_T$: frozen teacher = 原始 post-trained MoE
- $\pi_\theta$: student = 注入 zero expert 后的 augmented MoE
- $\mathcal{D}$: 60k prompt（17k math + 15k code 来自 NVIDIA AceReason-1.1-SFT，28k chat 来自 Llama-Nemotron-Post-Training-Dataset）
- $y \sim \pi_T(\cdot|x)$: response 由 teacher 采样（off-policy）
- $\mathcal{L}_{GA}$: Group Auxiliary Loss（见下）

**Stage 2: On-Policy Distillation (OPD)**

$$\mathcal{L} = \mathcal{L}_{OPD} + \mathcal{L}_{GA} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ \sum_{t=1}^{|y|} \mathrm{KL}(\pi_\theta(\cdot|x, y_{<t}) \| \pi_T(\cdot|x, y_{<t})) \right] + \mathcal{L}_{GA} \tag{Eq.3}$$

关键差异：response $y$ 从 **student 自己** 采样（on-policy），teacher 仅作为 token-level KL target。这个设计直接借鉴 Thinking Machines 的 On-Policy Distillation blog（https://thinkingmachines.ai/blog/on-policy-distillation）和 MiniLLM (Gu et al. 2023, https://arxiv.org/abs/2306.08543) / GKD (Agarwal et al. 2024, https://arxiv.org/abs/2306.13649)。

**两阶段为什么必要**（Table 5 ablation 给出证据）：
- SFT only: 73.6 avg
- OPD only: 72.9 avg（最差，因为 router 还没稳定，OPD 同时学 routing + coherent generation 太难）
- SFT→OPD: 74.2 avg

直觉：SFT 先把 router 从 "128 候选" 拓展到 "192 候选"（Qwen3 情况）后稳定下来——它学会了什么时候选 zero expert；OPD 再用 student 自己的 rollout 把 distribution gap 收掉。

---

## 3. Group Auxiliary Loss（最有意思的设计）

### 3.1 原始 load balancing loss 的问题

标准 MoE 的 auxiliary load balancing loss（GShard / Switch Transformer）：

$$\mathcal{L}_A = \alpha \cdot \frac{N + N_Z}{K} \cdot \sum_{i=1}^{N+N_Z} f_i \cdot P_i \tag{Eq.4}$$

$$f_i = \frac{1}{|\mathcal{B}|} \sum_{h \in \mathcal{B}} \mathbb{1}\{i \in \tilde{S}(h)\}, \quad P_i = \frac{1}{|\mathcal{B}|} \sum_{h \in \mathcal{B}} \tilde{g}_i(h)$$

变量：
- $f_i$: batch $\mathcal{B}$ 中 token 路由到 expert $i$ 的 fraction（"被选中频率"）
- $P_i$: batch $\mathcal{B}$ 中 expert $i$ 的平均 routing probability（"软投票强度"）
- $\alpha$: scalar loss coefficient

**这个 loss 强制所有 expert 均匀负载**。但对 post-trained MoE 来说这很糟糕——post-training 已经让 routing 变成 input-dependent、非均匀的（某些 expert 专长数学、某些专长 code），强行 expert-level uniform 会破坏 specialization。

Table 10 实验证据：用 $\mathcal{L}_A$ 替换 $\mathcal{L}_{GA}$，Qwen3 在 math benchmark avg 从 81.0 暴跌到 59.5，AIME24 从 78.1 跌到 39.6。

### 3.2 Group-level balancing

ZEDA 只在 **两个 group 之间** 做 balancing：normal expert group $\mathcal{E}$（共 N 个）vs zero expert group $\mathcal{Z}$（共 $N_Z$ 个）。

$$\mathcal{L}_{GA} = \alpha \cdot \frac{N + N_Z \cdot w}{K} \cdot \left( \frac{f_\mathcal{E} \cdot P_\mathcal{E}}{N} + \frac{f_\mathcal{Z} \cdot P_\mathcal{Z}}{N_Z \cdot w} \right) \tag{Eq.5}$$

$$f_\mathcal{E} = \sum_{i \in \mathcal{E}} f_i, \quad P_\mathcal{E} = \sum_{i \in \mathcal{E}} P_i, \quad f_\mathcal{Z} = \sum_{i \in \mathcal{Z}} f_i, \quad P_\mathcal{Z} = \sum_{i \in \mathcal{Z}} P_i \tag{Eq.6}$$

变量：
- $w > 0$: zero-expert group 的相对权重（**这是控制 quality-efficiency trade-off 的核心 knob**）
- $f_\mathcal{E}, f_\mathcal{Z}$: 两个 group 各自的总激活 frequency
- $P_\mathcal{E}, P_\mathcal{Z}$: 两个 group 各自的总 routing probability

**最小化 $\mathcal{L}_{GA}$ 的均衡点**满足 $K_\mathcal{E} : K_\mathcal{Z} = N : N_Z \cdot w$，对应目标：

$$r_{ZE}^{\text{target}} = \frac{N_Z \cdot w}{N + N_Z \cdot w}$$

例如 Qwen3-30B-A3B：N=128, $N_Z$=64, w=2 → $r_{ZE}^{\text{target}} = \frac{64 \times 2}{128 + 64 \times 2} = 50\%$。

**直觉**：在 normal expert group 内部，仍然保留 post-trained 形成的"非均匀、input-dependent 路由"结构；只在 group 边界上做约束。这避免了 $\mathcal{L}_A$ 把 specialization 平均掉的灾难。

### 3.3 两个超参的 ablation

**w 的 ablation（Figure 6 right, Section 4.3.1）**：
- w=1.0: $r_{ZE}$ 偏低（~35%），效率不够
- w=1.5: $r_{ZE}$ 上升但还不够
- **w=2.0: 最佳平衡点，$r_{ZE}$~51%，acc 损失最小**
- w=3.0/4.0: $r_{ZE}$ 进一步上升但 acc 明显下滑

**α 的 ablation（Table 4, Section 4.3.2）**：α∈{0.001, 0.01, 0.1, 1.0}
- α=0.001: $r_{ZE}$ 仅 31.8%（aux loss 太弱没起到作用）
- α=0.01: $r_{ZE}$ 44.6%
- **α=0.1: $r_{ZE}$ 51.5%，最接近 50% target**
- α=1.0: $r_{ZE}$ 50.9% 但 SFT 信号被淹没

---

## 4. 实验主表

### 4.1 Table 1 解读

11 个 benchmark：5 math（AIME24/25/26, GSM8k, MATH-500）+ 4 code（LCB v5/v6, HumanEval+, MBPP+）+ 2 IF（IFBench, IFEval）。

**Qwen3-30B-A3B** (N=128, K=8):

| Method | Avg Acc | Avg $r_{ZE}$ | 备注 |
|---|---|---|---|
| Original | 74.9 | 0.0 | 基线 |
| AdaMoE | 54.8 | 51.9 | 数学崩盘（AIME24=25.0） |
| Dynamic Skipping | 68.1 | 43.8 | code 崩盘（MBPP+=70.0） |
| NET$_{SFT}$ | 72.3 | 50.0 | Naive 截断成 K=4 |
| NET$_{SFT→OPD}$ | 73.0 | 50.0 | 同上加 OPD |
| ZEDA$_{SFT}$ | 73.3 | 51.5 | |
| **ZEDA** | **74.2** | **51.2** | 仅比 original 掉 0.7 分 |

**GLM-4.7-Flash** (N=64, K=4):

| Method | Avg Acc | Avg $r_{ZE}$ |
|---|---|---|
| Original | 72.5 | 0.0 |
| AdaMoE | 57.1 | 47.0 |
| Dynamic Skipping | 67.8 | 37.5 |
| **ZEDA** | **71.8** | **53.0** |

**关键观察**：
1. AdaMoE 和 Dynamic Skipping 都呈现严重 capability imbalance——一个数学崩一个 code 崩，说明对 post-trained 模型强行施加 dynamic routing 会破坏特定 capability。
2. ZEDA 是**唯一一个在所有 domain 上都保持竞争力的方法**，这说明 zero-expert injection + group-level balancing 真的保留了 post-trained 的 routing specialization。
3. ZEDA 比 NET$_{SFT→OPD}$（直接砍 K=8→K=4）平均高 1.2 分，说明 dynamic 比 static truncation 更优——不同 token 需要不同的 computation budget。
4. IFBench 上 ZEDA 反而超过 original（Qwen: 42.3 vs 39.7；GLM: 47.3 vs 47.3），推测 zero expert 给了 instruction following 一定 regularization。

### 4.2 Adaptation Cost (Table 2)

| Model | 总时间 | SFT rollout | SFT 训练 | OPD |
|---|---|---|---|---|
| Qwen3-30B-A3B | 30.12 h | 8.16 h | 1.97 h | 19.99 h |
| GLM-4.7-Flash | 61.37 h | 14.56 h | 4.51 h | 42.30 h |

8×H200 GPU，对比 pre-training + post-training 的上千小时成本，几乎可以忽略。

### 4.3 OOD 泛化 (Table 7)

MMLU-Redux 和 GPQA-Diamond 完全 OOD（不在 SFT 数据分布内）：
- Qwen3-30B-A3B: 76.2 avg（vs 76.7 original），$r_{ZE}$=47.2%
- GLM-4.7-Flash: 72.9 avg（vs 76.1 original），$r_{ZE}$=50.0%

**ZEDA 在 OOD 上仍然能保持 50% 左右的 zero-expert 激活**，说明 router 学到的是通用的 "token-level computation allocation" 而不是 in-domain 过拟合。

---

## 5. Zero-Expert Activation Dynamics（最有洞察的 Section 4.1）

这是 paper 里最能 build intuition 的部分。对 110 个 prompt（10/benchmark）做 rollout，记录每个 token 的 $r_{ZE}$、student logp、entropy，再算 teacher logp，得到 teacher-student logp-dif $\Delta_{logp}$。

### 5.1 三条核心发现

**(1) $r_{ZE}$ 与 $\Delta_{logp}$ 负相关（Figure 3 left）**

token 在 2D 平面（$\Delta_{logp}$ vs $r_{ZE}$）上的分布显示：左上角（大 $\Delta_{logp}$、低 $r_{ZE}$）密度高。即 **teacher-student 分布差异越大，student 越倾向于激活更多 normal expert**。

**(2) $r_{ZE}$ 与 entropy 负相关（Figure 3 right）**

student entropy 高 → $r_{ZE}$ 低（激活更多 normal expert）。

**(3) $r_{ZE}$ 与 task difficulty 无关（Table 3）**

MATH-500 五个 difficulty level 的 $r_{ZE}$ 几乎不变（51.1~52.5），且与 AIME24 的 52.1 接近。**模型不根据"任务难不难"分配算力，而是根据"token-level 特征"分配**。

### 5.2 Response pattern (Figure 4, 5)

逐 token 可视化发现：
- **code fragment / math expression**: $r_{ZE}$ 高（少算）
- **natural language / reasoning text**: $r_{ZE}$ 低（多算）

这看似反直觉——code/math 不是"更难"吗？但解释是：code/math token 通常是**低 entropy、高确定性**的（e.g., 一个 token 后面紧跟 `(` 或 `=`），student 自己就能很确定地预测，$\Delta_{logp}$ 也小，所以可以放心激活 zero expert。而自然语言推理 token 是高 entropy 的、需要更多 expert 计算去确定 next token。

Figure 5 进一步展示 layer-wise 分布：$r_{ZE}$ 在 48 个 MoE layer 上**比较均匀**，没有 systematic pattern。

### 5.3 Connecting the dots

paper 给出非常 elegant 的解释（Section 4.1 结尾）：

$$\Delta_{logp} \uparrow \implies \mathcal{L}_{SFT} \uparrow, \mathcal{L}_{OPD} \uparrow \implies \mathcal{L}_{GA} \text{ 相对 weight } \downarrow \implies r_{ZE} \downarrow$$

即当 token 的 teacher-student gap 大时，task loss 主导，$\mathcal{L}_{GA}$ 想把 $r_{ZE}$ 拉高的力被压制，所以 model 选择激活更多 normal expert 去缩小 gap。

加上 entropy 和 $\Delta_{logp}$ 的关联（Ko et al. 2026, https://arxiv.org/abs/2603.11137）：高 entropy token 倾向于高 $\Delta_{logp}$，且 code/math token 倾向于低 entropy（Wang et al. 2025, https://arxiv.org/abs/2506.01939，"High-entropy minority tokens"）——三个观察被统一解释。

**这是一个非常漂亮的 emergent behavior**：ZEDA 没有显式编码 "how much computation to allocate"，但训练动力学自然让 model 学到了与 model uncertainty + distributional gap 相关的 allocation policy。

---

## 6. Zero Expert vs Copy Expert（Appendix B）

copy expert 定义为 $Z_j(h) = h$（identity mapping），理论上也几乎零计算（一个加法）。

$$\tilde{y}_{copy} = \sum_{i \in \tilde{S} \cap \mathcal{E}} \tilde{g}_i(h) \cdot E_i(h) + \sum_{j \in \tilde{S} \cap \mathcal{Z}} \tilde{g}_j(h) \cdot h$$

后一项 $\sum_{j} \tilde{g}_j(h) \cdot h$ 不是 no-op，而是 input 的加权求和。

**Table 9 实验**（仅 SFT）：
| Method | Avg Acc | AIME24 | AIME25 | AIME26 |
|---|---|---|---|---|
| Original | 82.8 | 80.9 | 71.0 | 72.3 |
| Copy Expert + $\mathcal{L}_{GA}$ | **20.7** | 1.0 | 2.9 | 0.8 |
| Zero Expert + $\mathcal{L}_{GA}$ | 81.0 | 78.1 | 66.2 | 71.3 |

copy expert 在 AIME 上**几乎完全崩盘**。

**Figure 7 给出诊断**：
- **Scale mismatch**：copy expert 输出与 original output 的 L2 距离远大于 zero expert。copy component 自己就是主因。
- **Direction mismatch**：copy expert 输出与 original 的 cosine similarity 显著低于 zero expert，且随 depth 增长而恶化——copy component 像一个"漂移力"持续把 normal component 拉离 original direction。

直觉：copy expert = $\tilde{g}_j(h) \cdot h$ 引入了一个**与原 MoE residual 分支完全不同 scale 和 direction 的信号**，相当于往 model 里塞了一个 uncalibrated 的 skip connection。post-trained model 的 attention 和后续 layer 期待的是 expert residual 的特定 scale，强行注入这个新信号会破坏整个 flow。

---

## 7. 理论 FLOPs 分析（Appendix D）

### 7.1 符号表 (Table 11)

| Symbol | 含义 | Qwen3-30B-A3B 取值 |
|---|---|---|
| $l$ | 序列长度 | 1024~8192 |
| $H$ | hidden size | 2048 |
| $H_{attn}$ | attention intermediate | 4096 |
| $g_{kv}$ | GQA 中 KV head ratio | 1/8 |
| $H_e$ | expert intermediate | 768 |
| $N$ | normal experts | 128 |
| $N_Z$ | zero experts | 64 |
| $K$ | 激活 expert 数 | 8 |
| $r_{ZE}$ | zero expert 激活比例 | 0.5 |

### 7.2 Prefill Stage

每个 token 激活 K 个 normal expert，FFN 成本 $6 K n H H_e$（up/gate/down 三个矩阵乘，每个 $2nHH_e$），router 成本 $2 N n H$（router 是 $H \to N$ 的矩阵乘）。

$$F_{MoE, orig}(n) = 6KnHH_e + 2NnH \tag{Eq.9}$$

ZEDA 下，FFN 只算 $(1 - r_{ZE})K$ 个 expert，router 要算 $(N+N_Z)$ 个 expert：

$$F_{MoE, ZEDA}(n) = 6(1 - r_{ZE})K n H H_e + 2(N + N_Z) n H \tag{Eq.10}$$

GQA attention 在 prefill 阶段：

$$F_{attn}^{pre} = 4l^2 H_{attn} + 4(1 + g_{kv}) l H H_{attn} \tag{Eq.11}$$

第一项是 Q×K score + attention×V 的 $O(l^2)$ 项；第二项是 Q/K/V/O projection 的 $O(l)$ 项。

把 attention + MoE 加起来，prefill 总 FLOPs（每层）：

$$F_{orig}^{pre} = 4l^2 H_{attn} + 4(1+g_{kv}) l H H_{attn} + 6KlHH_e + 2NlH \tag{Eq.12}$$

$$F_{ZEDA}^{pre} = 4l^2 H_{attn} + 4(1+g_{kv}) l H H_{attn} + 6(1-r_{ZE})KlHH_e + 2(N+N_Z)lH \tag{Eq.13}$$

prefill FLOP ratio：

$$\frac{F_{ZEDA}^{pre}}{F_{orig}^{pre}} = \frac{2lH_{attn} + 2(1+g_{kv})HH_{attn} + 3(1-r_{ZE})KHH_e + (N+N_Z)H}{2lH_{attn} + 2(1+g_{kv})HH_{attn} + 3KHH_e + NH} \tag{Eq.14}$$

**关键观察**：
- attention 项（$2lH_{attn}$ 和 $2(1+g_{kv})HH_{attn}$）在分子分母都一样，不可压缩
- MoE FFN 项被 $(1-r_{ZE})$ 折半
- router 项从 $N$ 变 $N+N_Z$（轻微增加）

随着 $l$ 变大，attention 的 $O(l^2)$ 项主导整个 FLOPs，MoE 占比下降，所以 **speedup 随 sequence length decay**。

### 7.3 Decode Stage

decode 时每步处理 1 个新 token + $(t-1)$ 个 cached token。score 和 attention-value 聚合对每个 step 只涉及 $t-1$ 个 cached：

$$F_{attn}^{dec} = \sum_{t=1}^{l} [4(t-1)H_{attn} + 4(1+g_{kv})HH_{attn}] = 2l(l-1)H_{attn} + 4(1+g_{kv})lHH_{attn} \tag{Eq.15}$$

注意 $O(l^2)$ 项系数从 prefill 的 $4l^2$ 变成 $2l(l-1)$（每步 $t-1$ 个 cached token）。

$$F_{orig}^{dec} = 2l(l-1)H_{attn} + 4(1+g_{kv})lHH_{attn} + 6KlHH_e + 2NlH \tag{Eq.16}$$

$$F_{ZEDA}^{dec} = 2l(l-1)H_{attn} + 4(1+g_{kv})lHH_{attn} + 6(1-r_{ZE})KlHH_e + 2(N+N_Z)lH \tag{Eq.17}$$

decode FLOP ratio：

$$\frac{F_{ZEDA}^{dec}}{F_{orig}^{dec}} = \frac{(l-1)H_{attn} + 2(1+g_{kv})HH_{attn} + 3(1-r_{ZE})KHH_e + (N+N_Z)H}{(l-1)H_{attn} + 2(1+g_{kv})HH_{attn} + 3KHH_e + NH} \tag{Eq.18}$$

### 7.4 理论 vs 实测 (Table 13)

Qwen3-30B-A3B 配置 + $r_{ZE}=0.5$：

| Length | Prefill 理论 | Prefill 实测 | Decode 理论 | Decode 实测 |
|---|---|---|---|---|
| 1024 | 1.403x | 1.141x | 1.443x | 1.233x |
| 2048 | 1.341x | 1.214x | 1.403x | 1.252x |
| 4096 | 1.261x | 1.203x | 1.341x | 1.236x |
| 8192 | 1.178x | 1.175x | 1.261x | 1.185x |

**两个 trend**：
1. Speedup 随 sequence length decay（attention $O(l^2)$ 主导）
2. Decode speedup > Prefill speedup（同 length）——decode 阶段 attention 的 $O(l^2)$ 项系数是 prefill 的一半（$2l(l-1)$ vs $4l^2$），所以 MoE FLOPs 占总 FLOPs 的比例更高，被压缩的相对收益更大

实测比理论低 ~10%，gap 来自 router overhead、memory bandwidth、kernel launch overhead 等。

---

## 8. 工程实现细节

- 框架：slime（清华的 RL post-training framework, https://github.com/THUDM/slime）+ SGLang（推理）+ Megatron（训练）
- OPD 设置：Sampled-Token OPD，batch=16 prompts × 2 sampled responses，sampling temperature=1.0，max gen length=32k，320 training steps
- 学习率：SFT $2 \times 10^{-5}$；OPD Qwen $5 \times 10^{-6}$ / GLM $1 \times 10^{-6}$
- 评测：temperature=0.6, top-p=0.95, top-k=20, max gen=38k（Qwen3 设置）
- 评测多次采样降低方差：AIME avg@32, code avg@8, others avg@1
- 8×H200 GPU

---

## 9. 与相关工作脉络的关系

### 9.1 Dynamic MoE 谱系

- **MoE++** (Jin et al. 2024, https://arxiv.org/abs/2410.07348)：首次提出 zero-computation expert 概念，但是 from-scratch pre-training 设置
- **LongCat-Flash** (Meituan, https://arxiv.org/abs/2509.01322)：工业级 MoE 验证 zero-computation expert 思路
- **AdaMoE** (Zeng et al. 2024, https://aclanthology.org/2024.findings-emnlp.367)：null expert 让 activated expert 数变化
- **Ada-K Routing** (Yue et al. 2024, https://arxiv.org/abs/2401.06368)：token-dependent K
- **DynMoE** (Guo et al. 2024, https://arxiv.org/abs/2405.14297)：jointly auto-tune N 和 K
- **Expert Threshold Routing** (Sun et al. 2026, https://arxiv.org/abs/2603.11535)：threshold-based activation
- **Dynamic Skipping** (Lu et al. 2024, https://aclanthology.org/2024.acl-long.325)：inference-time skipping

ZEDA 的差异化：**post-training 阶段操作** + **不修改 normal expert 参数** + **用 self-distillation 让 router 学新 policy**。

### 9.2 Self-Distillation 谱系

- Hinton et al. 2015 经典 KD (https://arxiv.org/abs/1503.02531)
- Sequence-level KD (Kim & Rush 2016, https://arxiv.org/abs/1607.04336)
- MiniLLM (Gu et al. 2023, https://arxiv.org/abs/2306.08543)：reverse KL for LLM KD
- GKD (Agarwal et al. 2024, https://arxiv.org/abs/2306.13649)：on-policy KD
- Thinking Machines OPD blog (https://thinkingmachines.ai/blog/on-policy-distillation)
- Born-again networks (Furlanello et al. 2018) - self-distillation without external teacher
- RAD (Hoshino et al. 2025, https://arxiv.org/abs/2505.22135)：full attention → linear attention via self-distillation
- HALO (Chen et al. 2026, https://arxiv.org/abs/2601.22156)：类似架构转换
- LaDiMo (Kim et al. 2024, https://arxiv.org/abs/2408.04278)：dense → MoE via layer-wise distillation

**ZEDA 把 self-distillation 第一次用于 "static MoE → dynamic MoE" 的架构转换**，这是个比较 novel 的 angle。

---

## 10. 局限与未来方向

paper 自己提到：
1. **未在 >30B 模型上验证**——更大 MoE 是否还能保持 ~50% expert skip？
2. **未覆盖 agentic workload**——长 horizon 任务下 router behavior 未知
3. **Speedup decay at long context**——8k 时还有 20% speedup，更长序列下 MoE 占比下降，speedup 会进一步 decay
4. **DeepEP 集成**（https://github.com/deepseek-ai/DeepEP）：未来可与 expert-parallel 通信库配合，可能让 speedup 翻倍

我个人觉得还有几个可以联想的方向：
- **Speculative decoding 与 zero expert 的关联**：两者都是 "skip compute when confident"，能否统一？
- **MoE quantization 与 ZEDA 的 synergy**：如果 zero expert 已经省了 50% FLOPs，再叠 INT8/INT4 quantization 能否进一步压缩？
- **Router 的 interpretability**：分析 zero expert 的 routing pattern 能否揭示 post-trained MoE 学到了什么 capability specialization？
- **不同 $r_{ZE}$ 的 per-layer 设置**：现在所有 layer 用同一 $w$，但深层和浅层可能需要不同 allocation policy

---

## 11. 总结直觉

ZEDA 给我的最大直觉是：**post-trained MoE 的 routing 分布是一个已经收敛到精细 specialization 的 equilibrium**。任何 dynamic MoE 方法只要打破这个 equilibrium（expert-level uniform、强行改变 K、router 重训）都会引发严重退化。

ZEDA 的妙处在于：
1. **加 zero expert 是 minimal surgical change**——不动 normal expert 参数、不重训 router 已有权重，只在 router 输出端加 N_Z 个新 logits
2. **Group Auxiliary Loss 是 minimal constraint**——只在 group 边界做平衡，normal expert 内部路由分布原封不动
3. **Self-distillation 是 minimal training signal**——用 teacher 自己教 student，避免引入外部 capability drift
4. **Emergent computation allocation**——router 自己学到与 entropy / distributional gap 相关的 token-level allocation policy，没有人为编码任何 "difficulty heuristic"

最终用 30 小时（Qwen3-30B-A3B）换来 50% FLOPs 减少 + ~20% inference speedup + 0.7 avg acc 损失（甚至 OOD 上几乎无损）。这是一个 ROI 极高的工程方案。

参考资源：
- ZEDA paper: 待上传至 arXiv
- slime framework: https://github.com/THUDM/slime
- SGLang: https://github.com/sgl-project/sglang
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- Thinking Machines OPD blog: https://thinkingmachines.ai/blog/on-policy-distillation
- Qwen3 技术报告: https://arxiv.org/abs/2505.09388
- GLM-4.5: https://arxiv.org/abs/2508.06471
- MoE++: https://arxiv.org/abs/2410.07348
- LongCat-Flash: https://arxiv.org/abs/2509.01322
- DeepEP: https://github.com/deepseek-ai/DeepEP
