---
source_pdf: TIME-MOE BILLION-SCALE TIME SERIES FOUNDATION MODELS WITH MIXTURE OF EXPERTS.pdf
paper_sha256: 6777a5dcf0286aa9c042e3d4914d2faf8ededbf1f84b493e27a04989d2d2178d
processed_at: '2026-08-12T16:16:21-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好,用大白话再讲一遍。

---

## 这篇 paper 在干嘛

之前 time series 的 foundation model(Moirai、TimesFM、Chronos 那一拨)有个尴尬:模型都很小,最大也就 700M 参数,而且都是 dense 的 —— 意思是每来一个输入,所有参数都要算一遍。模型想变大变强,inference 就贵,部署不起。

TIME-MOE 的核心想法就一句话:**把 LLM 那边的 sparse MoE 套路搬到 time series 上,让"模型有多大"和"推理有多贵"两件事解耦。**

总参数堆到 2.4B,但每个 token 只激活其中 1.1B(一个 shared expert + 两个 isolated expert)。容量跟着 total params 走,FLOPs 跟着 activated params 走,各走各的。

---

## 架构上几个关键选择

**1. Point-wise tokenization —— 一个时间点一个 token**

这个跟 Moirai/TimesFM 不一样,它们是 patch-based(把 32 个点打包成一个 token)。

patch 的好处是 sequence 短,attention 便宜。坏处是输出长度必须是 patch_size 的倍数,预测 horizon=7 这种就很难受。

point-wise 的好处是输出灵活 —— 任意 horizon 都能预测。坏处是 sequence 长,attention 贵。但配合 flash-attention 扛到 4096 还行。

这是个 trade-off,他们选了 flexibility 那一头。

**2. Shared expert + 8 个 isolated expert,top-K=2**

每个 MoE 层有 1 个 shared expert(永远激活,学通用 pattern —— trend、seasonality 这种所有数据都有的东西)+ 8 个 isolated expert(动态选 2 个激活,学 specialized pattern —— 比如 finance 的 volatility clustering、weather 的 diurnal cycle)。

router 是个简单的 linear projection,对每个 token 算 8 个分数,softmax 之后取 top-2,其他置零。

shared expert 用 sigmoid gate(独立门,不跟别人抢),isolated experts 用 softmax top-K(互相竞争)。这个设计直接抄 DeepSeek-MoE。

**3. Multi-resolution forecasting heads —— 同时预测 4 个 horizon**

这一招挺巧。模型最后有 4 个 output head,分别预测未来 1 步、8 步、32 步、64 步。训练时 4 个 loss 一起算,multi-task learning。

推理时给个目标 horizon,用贪心算法拼:要预测 100 步,就用 64 + 32 + 4(4 个 1-step autoregressive 补)。

为什么这招好?两个原因:
- 训练时 multi-task 让模型同时学 short-term 和 long-term,泛化更好。
- 推理时大部分 horizon 可以一步预测完成,不用 autoregressive 跑很多步,**反而更快**。ablation 里 heads 从 4 个减到 1 个,inference 慢 30 倍。

---

## 训练的稳定技巧

time series 数据比 language 脏多了 —— outlier 多、missing value 多、sensor 故障产生大段常数。所以训练稳定性是硬问题。

**Huber loss**:小残差用 L2,大残差用 L1。outlier 不会把梯度拉飞。去掉这个 loss,MSE 涨 0.005。

**Auxiliary load balance loss**:防止 routing collapse —— 如果不加,最强的 expert 会越来越强,其他 expert 训不到,MoE 退化成小 FFN,优势全没。这个 loss 鼓励 router 把 token 均匀分给各 expert。去掉这个,MSE 涨 0.013,是所有 ablation 里掉最多的。

**Time-300B 数据集**:309B 观测点,9 个 domain。但 Nature 占 90.5%(气候数据 Weatherbench + CMIP6 + ERA5),domain imbalance 很严重。Finance 和 Healthcare 几乎没有。这是个隐患 —— 在 Weather benchmark 上强不奇怪,在 Finance 上可能没那么强。他们做了 data cleaning pipeline:在 NaN 处切序列(不填补)、滑窗检测常数段并切掉。挺实在的工程。

---

## 结果怎么样

Zero-shot:比 Chronos_large 好 23%,比 Moment 好 30%,比 Moirai_large 好 11%。在 time series 领域这个 margin 算很大了。

In-distribution(只 finetune 1 个 epoch):比 task-specific dense model(从 scratch 训 50 epoch)还好 24%。这是 foundation model 范式在 time series 上 work 的强力证据 —— 预训练 + 少量 finetune > 从头训。

Scaling law 验证了:model 从 50M → 200M → 1.1B activated,data 从 100B → 200B → 300B,MSE 单调下降,sparse 始终 dense。

BF16 跟 FP32 几乎没差 —— time series 数据噪声大,精度优势被淹没。BF16 + flash-attention 训练快 50%,省 20% 显存。

---

## 我觉得几个问题

1. **没有 probabilistic forecasting** —— 只输出 point forecast,不给 confidence interval。Moirai 给 distribution,Chronos 给 quantile。实际部署很多时候要 uncertainty,这是硬伤。

2. **Nature 占 90.5%** —— domain imbalance 没解决。Finance/Healthcare 数据太少,这俩 domain 的 zero-shot 能力存疑。

3. **Point-wise 在 long-context 上会撞墙** —— 4096 还行,要处理一年分钟级数据(525K points)就不可行。patch-based 在 long-context 上更有优势。

4. **{1,8,32,64} 这个 horizon 集合是 hardcoded** —— 假设自然 horizon 是这几个数的组合。实际任务 horizon=7(一周)、24(一天)会有 leftover,要 autoregressive 补。

5. **Routing 是 token-level 的** —— 但 time series 的 heterogeneity 主要是 sequence-level(整条 series 属于某个 domain)。可能 hierarchical MoE(先 sequence-level 路由,再 token-level)更合理。

---

## 一句话总结

**TIME-MOE = LLM 那套 sparse MoE 套路 + point-wise tokenization 的灵活性 + multi-resolution heads 的效率,第一次在 time series 上把 sparse MoE scale 到 2.4B,证明 scaling law 在 time series 上也成立。**

核心贡献是工程上的 —— 把很多已知 pieces(SwiGLU、RoPE、RMSNorm、MoE、Huber loss、aux loss、channel independence)组合起来,配合一个大数据集,把 time series foundation model 推到一个新规模。创新点不在单个组件,在于组合 + scale + 验证。

---

# TIME-MOE 深度讲解

Andrej, 这篇 paper 是 time series foundation model 领域一个挺有意思的 milestone — 第一次把 sparse MoE 认真 scale 到 2.4B 参数(其中 1.1B 激活),并配合一个 309B 观测点的 Time-300B 预训练数据集。它本质上是把 LLM 领域(尤其是 DeepSeek-MoE、Switch Transformer 那一脉)的稀疏化经验搬到 time series 上,然后通过 point-wise tokenization + multi-resolution forecasting heads 解决了之前 Moirai/TimesFM/Moment 在 flexibility 上的硬伤。下面我尽量把每个组件拆开讲,把 intuition 讲透。

---

## 1. 高层 Motivation:为什么 time series 也需要 sparse MoE

之前的 time series foundation model(Moirai、TimesFM、Moment、Chronos、Timer)都是 dense model,而且都很小 — 最大也就 710M(Chronos-large),最常用的是 200M 量级。问题是:

- **dense model 的 inference cost 与 total parameters 线性挂钩** — 想要更好的精度就必须堆参数,堆了参数 inference 就贵,这在 real-world 部署(尤其是 edge / CPU / 消费级 GPU 上)很致命。
- **time series 数据高度 heterogeneous** — 9 个 domain(energy / finance / nature / healthcare / sales / transport / web / synthetic / other),频率从秒级到年级,分布跨度极大。一个 dense FFN 要同时拟合所有这些模式,容易互相干扰。
- **scaling law 在 time series 上还没被严格验证过**(Yao et al. 2025 那篇 "Towards neural scaling laws for time series foundation models" 是同期工作)。

TIME-MOE 的核心 thesis:**用 sparse activation 把"模型容量"和"每 token 计算量"解耦** — total params 从 113M → 2.4B,但 activated params 只从 50M → 1.1B,inference 时每个 token 仍然只过 1 个 shared expert + 2 个 isolated experts(K=2)。这样 model capacity 跟着 total params 走,FLOPs 跟着 activated params 走,二者脱钩。

paper 里 Figure 1 右图把这个 trade-off 拍得很清楚:横轴是 effective FLOPs per token,纵轴是 average MSE,sparse TIME-MOE 在相同 FLOPs 下始终 dense baseline 一大截。

---

## 2. 整体架构解析

架构(Figure 2)是一个 decoder-only transformer + 每层一个 MoE layer,3 个核心组件:

### 2.1 Input Token Embedding — point-wise tokenization

**关键设计选择:每个 time point 一个 token,而不是 patch tokenization。**

这一点跟 Moirai(patch_size=16)、TimesFM(patch_size=32)、Moment(patch_size=8)都不一样。TIME-MOE 选 point-wise 是有理由的:
- patch tokenization 在输入端就把 information 压缩了 — 一个 patch token 已经是 32 个点的混合表征,这对下游 horizon=1 这种 short-horizon 任务很不友好。
- patch-based model 的 output length 必须是 patch_size 的倍数(Timer 那个 truncated output 问题就是这个),而 point-wise 可以输出任意 horizon。
- point-wise 的代价是 sequence length 更长 → attention 是 O(n²),但配合 rotary positional embedding + flash-attention 可以扛到 4096 length。

embedding 用 SwiGLU(Shazeer 2020):

$$\mathbf{h}_t^0 = \mathrm{SwiGLU}(x_t) = \mathrm{Swish}(W x_t) \otimes (V x_t)$$

变量含义:
- $x_t \in \mathbb{R}$ 是第 $t$ 个 time point 的标量值(注意 channel independence,所以每个 univariate series 单独处理)。
- $W, V \in \mathbb{R}^{D \times 1}$ 是两个独立的 linear projection。
- $D$ 是 hidden dimension(base=384, large=768, ultra=1024)。
- $\mathrm{Swish}(z) = z \cdot \sigma(\beta z)$(SiLU),$\otimes$ 是 element-wise product。
- 输出 $\mathbf{h}_t^0 \in \mathbb{R}^D$ 是第 $t$ 个 time point 的 embedding。

直觉:SwiGLU 比 simple linear 多了一个 gating 通路 — $V x_t$ 决定"信号怎么传",$\mathrm{Swish}(W x_t)$ 决定"门开多大"。这对 time series 这种数值范围跨度极大(traffic vs temperature vs stock price)很重要,因为同一个 $D$-dim 空间里要能容纳所有尺度。

### 2.2 MoE Transformer Block

每一层 $l$ 的前向过程(公式 2-4):

$$\mathbf{u}_t^l = \mathrm{SA}(\mathrm{RMSNorm}(\mathbf{h}_t^{l-1})) + \mathbf{h}_t^{l-1}$$
$$\bar{\mathbf{u}}_t^l = \mathrm{RMSNorm}(\mathbf{u}_t^l)$$
$$\mathbf{h}_t^l = \mathrm{Mixture}(\bar{\mathbf{u}}_t^l) + \mathbf{u}_t^l$$

变量含义:
- 上标 $l$ 是 layer index,下标 $t$ 是 time step。
- $\mathrm{SA}$ 是 causal multi-head self-attention(有 causal mask,所以是 autoregressive)。
- $\mathrm{RMSNorm}$ 是 root mean square normalization(比 LayerNorm 少一个 mean shift,只有 scale,training 更稳定)。
- $\mathrm{Mixture}$ 是 MoE layer,见下面。

注意 RMSNorm 是 Pre-Norm 架构(在 SA 和 Mixture 之前都做一次),这跟 LLaMA 一致 — Pre-Norm 在 deep model(ultra 36 层)里训练更稳。

位置编码用 RoPE(rotary positional embedding,Su et al. 2024),不用 absolute positional encoding,这样 context length 可以外推 — 训练时 max_len=4096,推理时可以更短或更长。

QKV 层保留 bias,其他 layer 去掉 bias — 这是 PaLM 那一脉的做法(Chowdhery et al. 2023),bias 在 attention 的 QKV 上有助于 length extrapolation。

### 2.3 Sparse Mixture Layer(核心)

公式 5-8:

$$\mathrm{Mixture}(\bar{\mathbf{u}}_t^l) = g_{N+1,t} \mathrm{FFN}_{N+1}(\bar{\mathbf{u}}_t^l) + \sum_{i=1}^{N} g_{i,t} \mathrm{FFN}_i(\bar{\mathbf{u}}_t^l)$$

$$g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \mathrm{TopK}(\{s_{j,t} | 1 \leq j \leq N\}, K) \\ 0, & \text{otherwise} \end{cases}$$

$$g_{N+1,t} = \mathrm{Sigmoid}(\mathbf{W}_{N+1}^l \bar{\mathbf{u}}_t^l)$$

$$s_{i,t} = \mathrm{Softmax}_i(\mathbf{W}_i^l \bar{\mathbf{u}}_t^l)$$

变量含义:
- $N$ 是 non-shared expert 数量(配置表里都是 8)。
- $K$ 是 top-K gating 的 $K$(配置表里都是 2)。
- $\mathrm{FFN}_1, \ldots, \mathrm{FFN}_N$ 是 $N$ 个 isolated experts,每个是独立的 FFN。
- $\mathrm{FFN}_{N+1}$ 是 **shared expert**,总是激活。
- $\mathbf{W}_i^l \in \mathbb{R}^{1 \times D}$ 是 router( gating network)第 $i$ 个 expert 的权重向量(只有一行,所以 router 是个 linear projection $D \to N$)。
- $s_{i,t}$ 是 router 对 token $t$ 在 expert $i$ 上的 softmax 归一化分数。
- $g_{i,t}$ 是 sparse gate:只保留 top-$K$ 个 expert 的 $s_{i,t}$,其他置零。
- $g_{N+1,t}$ 是 shared expert 的 gate,用 **sigmoid**(不是 softmax,因为是独立的 binary gate,不跟 isolated experts 竞争)。

这个设计跟 DeepSeek-MoE 一致:**1 个 shared expert + N 个 isolated experts**。

直觉:
- **shared expert** 学习"通用知识" — 在所有 context 下都有用的特征(比如 trend、seasonality 的基本统计量),所以总是激活,且 gate 用 sigmoid(连续 [0,1],可学)。
- **isolated experts** 学习"专门知识" — 不同 domain、不同 frequency、不同 distribution 的 specialized patterns,通过 top-K sparse routing 动态选择。
- 为什么 top-K 而不是 top-1?top-1(Switch Transformer)训练时 routing 决策太硬,容易 collapse;top-K(K=2)保留了少量 diversity,又不至于太 dense。
- paper 里 Table 7 的 sensitivity 显示:Top1 → 0.264 MSE,Top2 → 0.262,Top4 → 0.262,Top6 → 0.265,Top8 → 0.269。Top2 是 sweet spot — 性能最好,inference speed 也最快(0.095 s/iter vs Top8 的 0.129)。

每个 expert 的 FFN 结构跟标准 transformer 的 FFN 一样(就是两层 MLP + SwiGLU activation),但 $d_{expert}$ 比标准 $d_{ff}$ 小:
- base: $d_{model}=384$, $d_{ff}=1536$, $d_{expert}=192$
- 也就是说一个 expert 的 hidden 是标准 FFN 的 1/8,8 个 isolated experts 加起来 ≈ 标准 FFN 容量,加上 1 个 shared expert → 总容量大约是 dense 的 ~1.1 倍,但激活只有 shared + 2 isolated = 1/4 容量。

---

## 3. Loss Function 设计

### 3.1 Auto-regressive Huber Loss

公式 9:

$$\mathcal{L}_{ar}(x_t, \hat{x}_t) = \begin{cases} \frac{1}{2}(x_t - \hat{x}_t)^2, & \text{if } |x_t - \hat{x}_t| \leq \delta \\ \delta \cdot (|x_t - \hat{x}_t| - \frac{1}{2}\delta), & \text{otherwise} \end{cases}$$

变量含义:
- $x_t$ 是 ground truth,$\hat{x}_t$ 是 prediction。
- $\delta$ 是 L1/L2 切换的阈值超参。

直觉:
- 小残差(< δ)用 L2 — smooth gradient,易收敛。
- 大残差(> δ)用 L1 — gradient 是常数 $\delta$,不会被 outlier 拉飞。
- time series 数据 outlier 极多(sensor 故障、traffic 异常、stock crash),MSE 会被 outlier 主导,training instability。Huber 是 robust regression 的经典做法,Wen et al. 2019 的 RobustTrend 就用过。
- Ablation(Table 5):去掉 Huber 用 MSE,average MSE 从 0.262 涨到 0.267 — 0.005 看着不多,但每个 benchmark 都涨,说明 robustness 是稳态提升。

### 3.2 Auxiliary Load Balance Loss

公式 10:

$$\mathcal{L}_{aux} = N \sum_{i=1}^{N} f_i r_i$$

$$f_i = \frac{1}{KT} \sum_{t=1}^{T} \mathbb{I}(\text{Time point } t \text{ selects Expert } i)$$

$$r_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t}$$

变量含义:
- $N$ 是 isolated expert 数量(不含 shared)。
- $f_i$ 是 fraction of tokens dispatched to expert $i$ — 实际"路由过去"的比例。
- $r_i$ 是 average router probability for expert $i$ — router 想给它的"平均意向"。
- $\mathbb{I}$ 是 indicator function。
- $K$ 是 top-K,$T$ 是 sequence length。

直觉:
- 这是 Switch Transformer(Fedus et al. 2022)那一脉的 load balance loss。
- 如果 expert $i$ 既被 router 高分评估($r_i$ 大)又实际被选得多($f_i$ 大),那 $f_i r_i$ 就大,loss 惩罚它 → 鼓励 router 把 token 分散开。
- $f_i$ 是不可导的(indicator),但 $r_i$ 可导,二者乘积的梯度通过 $r_i$ 反传到 router weights $\mathbf{W}_i^l$。
- 关键 ablation:去掉这个 loss(w/o auxiliary loss)→ 0.262 → 0.275,性能掉得最多。原因 paper 里讲了:routing collapse — 最强 expert 越来越强,其他 expert 训不到,整个 MoE 退化成一个小 FFN。这就把 sparse MoE 的全部优势抹掉了。

### 3.3 Multi-Resolution Composite Loss

公式 11:

$$\mathcal{L} = \frac{1}{P} \sum_{j=1}^{P} \mathcal{L}_{ar}(\mathbf{X}_{t+1:t+p_j}, \hat{\mathbf{X}}_{t+1:t+p_j}) + \alpha \mathcal{L}_{aux}$$

变量含义:
- $P$ 是 multi-resolution output projection 数量(默认 4)。
- $p_j$ 是第 $j$ 个 projection 的 horizon(默认 $\{1, 8, 32, 64\}$)。
- $\alpha = 0.02$ 是 auxiliary loss 的权重(很小,说明主任务还是 forecasting)。

直觉:
- 4 个 horizon 同时预测 → multi-task learning。模型被强制学习"既会 short-horizon 又会 long-horizon",防止 over-fit 到单一 horizon。
- $\{1, 8, 32, 64\}$ 是精心选的 — 1(单步,autoregressive 严格意义)、8(分钟到小时)、32(几小时到天)、64(几天)。覆盖多种 scale。

---

## 4. Multi-Resolution Forecasting Heads

公式 12:

$$\hat{\mathbf{X}}_{t+1:t+p_j} = \mathbf{W}_{p_j} \mathbf{h}_t^L$$

变量含义:
- $\mathbf{W}_{p_j} \in \mathbb{R}^{p_j \times D}$ 是第 $j$ 个 output projection 的 weight(直接从 $D$-dim hidden state 投到 $p_j$-dim output)。
- $\mathbf{h}_t^L$ 是第 $L$ 层(最后一层)的 hidden state。

注意这不是 autoregressive 的"一步步 unfold" — 是一次性从最后一个 token 的 hidden state 投影出未来 $p_j$ 步。这跟传统 RNN-based forecasting(DeepAR)很不一样,更接近 GPT 用 LM head 一次吐多个 token 的思路,但这里 head 是 continuous regression 而不是 categorical logits。

### Inference 时的 Greedy Scheduling Algorithm

Algorithm 1 给了一个贪心调度:给定 target horizon $H$,从最大的 $p_j$ 开始,如果 $\hat{H} + p_j \leq H$ 就用 $p_j$ 来填,直到填满 $H$。

例子:目标 $H=100$,heads 是 $\{1, 8, 32, 64\}$:
- $64 \leq 100$ → 用 64,$\hat{H}=64$
- $32 + 64 = 96 \leq 100$ → 用 32,$\hat{H}=96$
- $8 + 96 = 104 > 100$,跳过
- $4 + 96 = 100 \leq 100$ → 用 4 个 1-step?不对,应该用 1 个 1-step $\hat{H}=97$,然后还要再生成 3 步...

实际 paper 实验里 horizons 都是 64 的整数倍(96=64+32, 192=3×64, 336=64×4+64+32+8+8, 720=11×64+32+8+8+8),所以基本一次 autoregressive step 就够,只有 96 这种混合 horizon 需要拼。

Ablation Table 5 右边很关键:
- $\{1, 8, 32, 64\}$ → MSE 0.262, 0.095 s/iter
- $\{1, 8, 32\}$ → MSE 0.273, 0.130 s/iter
- $\{1, 8\}$ → MSE 0.320, 0.411 s/iter
- $\{1\}$ → MSE 1.382, 2.834 s/iter

有意思的观察:**heads 越多,既更准 又更快**。这是反直觉的 — 多了 head 好像应该更慢才对。但解释是:head 多了之后,大部分 horizon 可以一步预测完成,不需要 autoregressive 多步,所以 total forward pass 数量反而少。$\{1\}$ 配置下要预测 horizon=96 必须跑 96 次 forward(纯 autoregressive),所以慢 30 倍。

---

## 5. Time-300B Dataset

数据规模(Table 1):
- 309.09B observations, 48.22M sequences, 9 domains。
- Nature 占 90.5%(主要是 Weatherbench 74.6B + CMIP6 104.6B + ERA5 93.8B)— 这个 domain imbalance 是 paper 没充分讨论的潜在问题,我下面会提。
- Energy 5.17%, Synthetic 2.98%, Transport 0.69%, Web 0.58%, 其他都很小。

频率分布:秒级(Solar Power 4S)、分钟级(Electricity 15T)、小时级(大量)、日级、周级、月级、季度、年级。频率跨度极大是 time series foundation model 的关键挑战。

### Data Cleaning Pipeline(Algorithm 2)

paper 在 Appendix C 给了 source code,挺实在。两步:

**1. Missing Value Processing:**
```python
def split_seq_by_nan_inf(seq, minimum_seq_length=1):
    # 在 NaN/Inf 处把序列切开,保留非空子序列
    # 而不是用 mean/forward-fill 填补(这会扭曲原 pattern)
```

直觉:mean imputation 会把一段缺失的中间用常数填,这在 statistical property 上完全改变了一阶差分、二阶差分分布。直接切掉保留原始 pattern 更干净。

**2. Invalid Observation Processing:**
```python
def split_seq_by_window_quality(seq, window_size=128, zero_threshold=0.2, ...):
    # 滑窗扫,对每个 window 计算:
    #   - zero_ratio(常数序列检测)
    #   - first_diff_zero_ratio(一阶差分多为 0 → 平直段)
    #   - second_diff_zero_ratio(二阶差分多为 0 → 线性段)
    # 任一 ratio > 0.2 就判定 window 无效,切掉
```

直觉:有些数据采集系统在 sensor 故障时用 0 或常数填充,产生大段平直线。这些段对模型训练是有害的 — 模型会学到"长期为 0 是正常的",破坏 forecasting。0.2 阈值是个比较宽松的过滤 — 一段 128 个点里允许 25 个 zero-diff(<20%)。

清洗后实际训练用了 117B points(从 309B 采样),按固定 domain 比例和观测值分布采样每个 batch,缓解 domain imbalance。

---

## 6. Model Configurations & Training

Table 2 给了三个 model size:

| Model | Layers | Heads | Experts | K | $d_{model}$ | $d_{ff}$ | $d_{expert}$ | Activated | Total |
|-------|--------|-------|---------|---|------------|---------|--------------|-----------|-------|
| base  | 12 | 12 | 8 | 2 | 384  | 1536 | 192 | 50M  | 113M  |
| large | 12 | 12 | 8 | 2 | 768  | 3072 | 384 | 200M | 453M  |
| ultra | 36 | 16 | 8 | 2 | 1024 | 4096 | 512 | 1.1B | 2.4B  |

观察:
- base / large 只在 $d_{model}$ 上 scale(384 → 768),depth 不变。这是 width scaling。
- ultra 是 depth + width 一起 scale(12 → 36 layers, 384 → 1024)。
- 三个 model 都用 N=8 experts, K=2 — 不在 expert 数量上做 scaling。
- Activated params ≈ Total / 2.2(sparse ratio ~45%)。这跟 LLM 里 sparse MoE 通常 10-20% activated ratio 不一样 — TIME-MOE 是相对 dense 的 sparse,可能因为 time series 数据模式比 language 更共享(都需要 trend/seasonality 处理)。

Training 设置:
- 100,000 steps, batch=1024, max_seq_len=4096
- 4M time points per iteration(1024 × 4096)
- AdamW(lr=1e-3, weight_decay=0.1, $\beta_1=0.9$, $\beta_2=0.95$)
- Linear warmup 10K steps + cosine annealing
- 128 × A100-80G, BF16
- Sequence packing(Raffel et al. 2020)— 把多个短 sequence 拼到一个 4096-length batch 里,减少 padding 浪费

---

## 7. 实验结果

### 7.1 Zero-Shot(Table 3)

6 个 benchmark:ETTh1, ETTh2, ETTm1, ETTm2, Weather, Global Temp,horizons {96, 192, 336, 720},context {512, 1024, 2048, 3072}。

对比 16 个 baseline,包括 Moirai(small/base/large)、TimesFM、Moment、Chronos(small/base/large)。

TIME-MOE_ultra 平均 MSE 比 Chronos_large 低 23%、比 Moment 低 30%、比 Moirai_large 低 11%。这个 margin 在 time series 领域算很大了 — 一般 paper 提升 5% 已经能发。

### 7.2 In-Distribution(Table 4)

在 6 个 benchmark 上各 finetune 1 epoch(注意 — 只 1 epoch!这跟传统 task-specific model 训 50 epoch 不一样)。对比 iTransformer、TimeMixer、TimesNet、PatchTST、Crossformer、TiDE、DLinear、FEDformer。

TIME-MOE_ultra 平均 MSE 比 SOTA(task-specific dense model)低 24%。也就是说:预训练 + 1-epoch finetune > 从头训 50 epoch。这是 foundation model 范式在 time series 上的强力证据。

### 7.3 Scalability Analysis(Figure 3)

左图:sparse vs dense 的 training/inference cost。sparse 训练 cost 降 78%,inference cost 降 39%。

右图:model size × data size 的 scaling law 图:
- 3 个 sparse model 在 100B、200B、300B data 上训练的 MSE 曲线
- data 越多、model 越大,MSE 越低,monotonic
- sparse 始终 dense(同 activated params 规模)

### 7.4 BF16 vs FP32(Table 6, Table 13)

- BF16 MSE = 0.262, FP32 MSE = 0.261 — 几乎无差。
- BF16 训练速度 0.84 s/iter vs FP32 1.24 s/iter → 快 32%。
- BF16 训练显存 1.77 GB vs FP32 2.21 GB → 省 20%。
- 加 flash-attention:训练 +23%,inference +19%。

这个结果说明 time series 不像 LLM 那样需要高精度 — point values 本身噪声很大,FP32 的精度优势被数据噪声淹没。

### 7.5 Sparsification Analysis(Figure 4)

gating score 在 6 个 benchmark 上的可视化。每个 layer、每个 expert 在不同数据集上激活强度差异很大 — 比如 expert 2 在 ETTm1 上很强、在 Weather 上弱;expert 5 反过来。说明 expert specialization 真的发生,不是所有 expert 学到一样的东西。这是 MoE 工作 的核心 sanity check。

---

## 8. 跟其他 Time Series Foundation Models 对比

Table 8 给了对比:

| Method | Arch | Max Size | Input Token | Dataset Scale | Max Length | FFN |
|--------|------|----------|-------------|----------------|------------|-----|
| Time-MoE | Decoder-Only | 2.4B | Point | 309B | 4096 | Sparse |
| Moirai | Encoder-Only | 311M | Patch | 27B/231B | 5000 | Dense |
| TimesFM | Decoder-Only | 200M | Patch | 100B | 512 | Dense |
| Moment | Encoder-Only | 385M | Patch | 1.13B | 512 | Dense |
| Chronos | Enc-Dec | 710M | Point | 84B | 512 | Dense |
| Timer | Decoder-Only | 67M | Patch | 28B | 1440 | Dense |
| Lag-Llama | Decoder-Only | 200M | Point | 0.36B | 1024 | Dense |
| TimeGPT | Enc-Dec | ? | Patch | 100B | ? | Dense |

TIME-MOE 是唯一一个 sparse 的、最大、且训练数据最多的。MoE 并发工作 Moirai-MoE(Liu et al. 2024a)有 935M params,但 routing 设计不同。

---

## 9. 个人的几点 Critical Observation

### 9.1 Nature domain 占 90.5% 是个隐患

Time-300B 里 Nature 占 90.5%(主要是 Weatherbench、CMIP6、ERA5 三个气候模型数据)。这些数据虽然 time point 多,但同质性高 — 都是 temperature / pressure / wind 这类地球物理场。Finance 0.0001%、Healthcare 0.0001%、Sales 0.008% 几乎可忽略。

这意味着:
- TIME-MOE 在 Weather benchmark 上强势(同分布),在 Finance / Healthcare 这种 low-resource domain 上可能没那么强。
- paper 没给 Monash Forecasting Archive(M4、M5、Tourism)的细粒度结果,只给了 long-term forecasting 的 6 个 benchmark。
- 未来的 time series foundation model 真要 universal,需要解决 domain imbalance — 可能要类似 LLM 那种 domain upsampling 或 curriculum learning。

### 9.2 Point-wise Tokenization 的 cost

point-wise 的最大问题是 sequence length 爆炸。horizon=4096 时,attention 是 4096² = 16.7M 操作 per head per layer。ultra 有 36 层、16 head → 单次 forward attention 操作量 ~9.6B,这在 LLM 里早就是 long-context 范畴。

paper 用 flash-attention 缓解,但本质上 point-wise 在 long-context 上不如 patch-based 高效。TimesFM 的 patch_size=32 让 effective sequence length 降 32 倍,可以处理 32k+ context。

trade-off:point-wise 换来 output flexibility(任意 horizon),但牺牲了 context length。如果未来要做超 long-context(比如一年分钟级数据 = 525K points),point-wise 不可行,得回到 hierarchical 或 patch。

### 9.3 Multi-Resolution Head 的 inductive bias

{1, 8, 32, 64} 这个 horizon 集合是个 hardcoded inductive bias。它假设 forecasting 任务的自然 horizon 是这几个数的组合。但实际任务可能 horizon=7(一周)、24(一天)、168(一周小时)、365(一年)— 这些不是 {1,8,32,64} 的好组合,会有 leftover steps 需要 autoregressive 补全。

更好的做法可能是 continuous horizon prediction(比如用 neural process 或 implicit neural representation),或者让 head 数量随 context 自适应。这是 future work 的一个方向。

### 9.4 Huber loss 的 $\delta$ 没说怎么选

公式 9 的 $\delta$ 是关键超参 — 它决定 L1/L2 切换点。paper 没说 $\delta$ 怎么定(可能是数据 std 的某个分位数,或者 fixed value)。这其实很重要,因为不同 domain 的 outlier 尺度差几个数量级,固定 $\delta$ 不一定 robust。一个可能更好的方案是 studentized residual(用 running std 归一化)或者 adaptive $\delta$。

### 9.5 Top-K=2 可能不是 final answer

Table 7 显示 Top1 跟 Top2 几乎一样(0.264 vs 0.262),Top2 比 Top4 略好(0.262 vs 0.262),Top8 反而变差(0.269)。这个 non-monotonic 关系很有意思 — 说明 K 太小(routing 决策硬)+ K 太大(失去 sparsity)都有问题。

LLM 领域 DeepSeek-V2 用 top-K=6 + shared expert + auxiliary-free loss,效果比 top-K=2 好。TIME-MOE 没探索 K>2 的更细配置。可能 K=3 或 K=4 在更大 model 上更好。

### 9.6 Routing 是 token-level 的,没有 sequence-level routing

paper 里 router 对每个 time point 独立决策 expert。但 time series 的 heterogeneity 主要是 sequence-level(整个 series 来自某个 domain),不是 token-level(单个 time point 有自己的 domain)。所以更合理的设计可能是 sequence-level router + token-level router 两层,类似 hierarchical MoE。

### 9.7 没有 Probabilistic Forecasting

TIME-MOE 输出 point forecast,不给 prediction interval 或 quantile。Moirai 是 distribution head(学生 t)、Chronos 是 categorical bin(quantile binning)、Lag-Llama 是 likelihood head。实际部署中,downstream 决策往往需要 uncertainty quantification(库存管理、energy bidding、risk assessment)。这是 TIME-MOE 相对 Moirai 的硬伤 — 虽然 MSE 低,但实际应用场景受限。

---

## 10. 关键 References

paper 链接(arXiv):
- TIME-MOE 原文:https://arxiv.org/abs/2409.16040
- GitHub 代码:https://github.com/Time-MoE/Time-MoE

相关工作:
- Switch Transformer (Fedus et al. 2022): https://arxiv.org/abs/2101.03961
- DeepSeek-MoE (Dai et al. 2024): https://arxiv.org/abs/2401.06066
- Moirai (Woo et al. 2024): https://arxiv.org/abs/2402.02592
- TimesFM (Das et al. 2024): https://arxiv.org/abs/2310.10688
- Chronos (Ansari et al. 2024): https://arxiv.org/abs/2403.07815
- Moment (Goswami et al. 2024): https://arxiv.org/abs/2402.03890
- Timer (Liu et al. 2024d): https://arxiv.org/abs/2402.02368
- Moirai-MoE (concurrent, Liu et al. 2024a): https://arxiv.org/abs/2410.10469
- PatchTST (Nie et al. 2023, channel independence): https://arxiv.org/abs/2211.14730
- SwiGLU (Shazeer 2020): https://arxiv.org/abs/2002.05202
- RoPE (Su et al. 2024): https://arxiv.org/abs/2104.09864
- RMSNorm (Zhang & Sennrich 2019): https://arxiv.org/abs/1910.07467
- FlashAttention-2 (Dao 2024): https://arxiv.org/abs/2307.08691
- Scaling laws for neural language models (Kaplan et al. 2020): https://arxiv.org/abs/2001.08361
- Monash Forecasting Archive (Godahewa et al. 2021): https://openreview.net/forum?id=wEc1mgAjU-
- Neural scaling laws for TSFMs (Yao et al. 2025, ICLR): https://arxiv.org/abs/2410.12238
- Foundation models for time series tutorial (Liang et al. 2024 KDD): https://arxiv.org/abs/2310.10196
- Huber loss original (Huber 1964 / 1992 reprint): classic stat reference
- RobustTrend (Wen et al. 2019, Huber loss in TS): https://www.ijcai.org/proceedings/2019/0738
- LLaMA (Touvron et al. 2023, arch inspiration): https://arxiv.org/abs/2302.13971
- PaLM (Chowdhery et al. 2023, bias design): https://arxiv.org/abs/2204.02311
- Sparsely-Gated MoE Layer (Shazeer et al. 2017): https://arxiv.org/abs/1701.06538
- Vision Mixture of Experts (Riquelme et al. 2021): https://arxiv.org/abs/2006.16668
- T5 / Sequence Packing (Raffel et al. 2020): https://arxiv.org/abs/1910.10683
- LLaMA 3 (Dubey et al. 2024): https://arxiv.org/abs/2407.21783
- DLinear (Zeng et al. 2023): https://arxiv.org/abs/2205.13504
- SparseTSF (Lin et al. 2024): https://arxiv.org/abs/2404.14026
- iTransformer (Liu et al. 2024b): https://arxiv.org/abs/2310.06625
- TimeMixer (Wang et al. 2024b): https://arxiv.org/abs/2405.14616
- TimesNet (Wu et al. 2023a): https://arxiv.org/abs/2210.02186
- FEDformer (Zhou et al. 2022b): https://arxiv.org/abs/2201.12740
- Autoformer (Wu et al. 2021): https://arxiv.org/abs/2106.13008
- Lag-Llama (Rasul et al. 2023): https://arxiv.org/abs/2310.08278
- Pathformer (Chen et al. 2024, multi-scale transformer): https://arxiv.org/abs/2402.05956
- IME (Ismail et al. 2023, interpretable MoE): https://arxiv.org/abs/2306.00894
- TimeGPT (Garza et al. 2023): https://arxiv.org/abs/2310.03589
- TimeMixer++ (Wang et al. 2025, ICLR): https://arxiv.org/abs/2410.10832
- TS2Vec (Yue et al. 2022, contrastive TS pretraining): https://arxiv.org/abs/2106.10483
- DeepAR (Salinas et al. 2020): https://arxiv.org/abs/1704.04110
- N-BEATS (Oreshkin et al. 2020): https://arxiv.org/abs/1905.10437
- TFT (Lim et al. 2021): https://arxiv.org/abs/2106.13008
- AutoTimes (Liu et al. 2024c): https://arxiv.org/abs/2402.02370
- GShard (Lepikhin et al. 2020): https://arxiv.org/abs/2006.16668
- Sparsity in DL survey (Hoefler et al. 2021): https://arxiv.org/abs/2102.00554

Benchmark datasets:
- ETT (Zhou et al. 2021 Informer): https://arxiv.org/abs/2012.07436
- Weather (Wu et al. 2021 Autoformer): https://arxiv.org/abs/2106.13008
- BuildingsBench (Emami et al. 2023): https://arxiv.org/abs/2307.07191
- Weatherbench (Rasp et al. 2020): https://arxiv.org/abs/2002.00489
- ClimateLearn (Nguyen et al. 2023): https://arxiv.org/abs/2306.08259
- LargeST (Liu et al. 2023): https://arxiv.org/abs/2306.08259
- Beijing Air Quality (Chen 2019 UCI): https://archive.ics.uci.edu/dataset/501
- SDWPF KDD Cup 2022 (Zhou et al. 2022a): https://arxiv.org/abs/2208.04360
- Project Tycho (van Panhuis et al. 2018): https://www.tycho.pitt.edu/
- CDC FluView: https://gis.cdc.gov/grasp/fluview/

---

## 11. 总结:Intuition Takeaway

如果只能记 5 件事:

1. **Sparse MoE 解耦 model capacity 和 inference cost** — total params 大,activated params 小,FLOPs 跟着 activated 走。TIME-MOE_ultra 2.4B total / 1.1B activated,inference 在 8GB GPU 上能跑。

2. **Point-wise tokenization + multi-resolution heads 解决 output flexibility** — 不像 patch-based model 受 patch_size 约束,任意 horizon 都能预测。{1,8,32,64} heads 同时训练多 horizon,multi-task 效应让模型更 robust,inference 时还能用更长的 head 一步预测多步,反而更快。

3. **Time-300B + Huber loss + Auxiliary load balance loss = stable training** — time series 数据噪声极大,Huber 抗 outlier,Aux loss 防 routing collapse。这两个去掉各掉 0.005 和 0.013 MSE,后者是 MoE 的核心 stability trick。

4. **Decoder-only + Autoregressive + Channel Independence** — 跟 GPT 范式对齐,channel independence(来自 PatchTST)让模型可以处理 arbitrary-variate input,把 multivariate 问题 univariate 化,简化训练。

5. **Scaling law 在 time series 上首次严格验证** — model 50M→200M→1.1B activated,data 100B→200B→300B,MSE 单调下降,sparse 始终 dense。这为下一代 TSFM 的发展方向奠定了基础。

paper 留下的开放问题主要在 probabilistic forecasting、超 long-context、和 domain imbalance 上,这些都是后续 work 可以突破的点。MoE 在 time series 上的探索才刚开始。
