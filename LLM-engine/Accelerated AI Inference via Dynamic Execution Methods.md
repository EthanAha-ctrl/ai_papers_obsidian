---
source_pdf: Accelerated AI Inference via Dynamic Execution Methods.pdf
paper_sha256: 85bda5e967aeae2f8436acf0dc8392a29fffa8ee5d084e1679ce077a3ec478a8
processed_at: '2026-07-17T23:04:21-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Accelerated AI Inference via Dynamic Execution Methods 深度解析

Andrej, 这篇 paper 来自 Intel 团队(Haim Barad 等人),核心 thesis 是把 inference optimization 划分成两个 orthogonal 的 axis: **Model Compression**(static, approximate math) 和 **Dynamic Execution**(data-dependent, short-cut)。我把它当成一篇"taxonomy + experimental reproduction"的综述来看,真正有意思的是它把好几条加速线串在一起,讲它们共享同一种 cognitive science intuition。下面我把每个 piece 拆开讲技术细节,顺便补一些 paper 没展开的 related work,帮你 build intuition。

---

## 1. 两套优化范式的根本区别

Paper 在 Section 2 开了一个非常干净的二分法:

**Model Compression** 求解的是:
$$
L_c = \min |y' - y|, \quad y' = F'(x), \quad y = F(x)
$$

变量含义:
- $y$ — 原始模型 $F$ 在输入 $x$ 上的输出
- $y'$ — 压缩后模型 $F'$ 在同一 $x$ 上的输出
- $L_c$ — compression loss,衡量 $F'$ 相对 $F$ 的偏离
- $F$ 与 $F'$ 都是 deterministic function,只是参数 precision/bit-width/structure 不同

这个 formulation 的 key insight 是: **优化的是 model 本身,与 input $x$ 无关**。无论 $x$ 是简单还是困难,你跑的还是同一个 $F'$。这就是 "approximate math" 的本质 — 一个 brute-force 的 globally applicable approximation。

**Dynamic Execution** 则完全不同,它不动 $F$,而是改 computation flow:
$$
\text{cost}(x) = \text{work}(F, x, \text{policy}(x))
$$

这里 $\text{policy}(x)$ 是一个 data-dependent 的 routing/early-stop 决策。不同 $x$ 走不同路径,简单 $x$ 早停,难 $x$ 走满。

这两者 **正交**(orthogonal),可以叠加,这是 paper 反复强调的一点 — quantization 是把每个 op 做得更便宜,dynamic execution 是少做 op。Intel 在 OpenVINO / HuggingFace Optimum 里就是把它们一起集成的 ([OpenVINO](https://github.com/openvinotoolkit/openvino), [Optimum Intel](https://github.com/huggingface/optimum-intel))。

---

## 2. Early Exit: 构建 intuition 最好的入口

### 2.1 Figure 1 的几何 intuition

Paper 的 Figure 1 画了一个 2D feature space。关键 observation:
- Deep network 学出来的是一个 curved decision boundary(图中绿色平行线之间的复杂曲线)
- 但样本空间里大量点是远离边界的(easy examples),用一条 linear classifier(图中黑线)就能正确分类
- 只有落在 margin(两条绿线)内的点才需要 deep network 的全部 capacity

这其实就是 **support vector 的思想** — 真正决定边界的只是少数样本。用 entropy threshold 当 confidence measure:
$$
H(p) = -\sum_i p_i \log p_i
$$

其中 $p_i$ 是第 $i$ 类的 softmax probability。$H$ 小 → 模型 confident → exit early。threshold 越大 → 越早 exit → 越多 speedup,但 accuracy 越掉。

### 2.2 公式化 Early Exit

设原始 network 有 $L$ 层,在每一层 $l \in \{l_1, l_2, ..., l_k\} \subset \{1,...,L\}$ 挂一个 auxiliary classifier $g_l$:
$$
\hat{y}_l = g_l(h_l(x))
$$

$h_l(x)$ 是第 $l$ 层 hidden state。Exit policy:
$$
l^*(x) = \min \{ l \in \{l_1, ..., l_k\} : H(\text{softmax}(g_l(h_l(x)))) < \tau \}
$$

$\tau$ 是 entropy threshold,是个 hyperparameter,trade-off latency vs accuracy。

### 2.3 Paper 的实验结果(Figure 2, SST-2)

在 BERT / RoBERTa / ALBERT 上跑 GLUE,速度-质量曲线显示 **2x–4x speedup 几乎不掉分**。这和 [Liao et al. NAACL 2021 "A Global Past-Future Early Exit Method"](https://aclanthology.org/2021.naacl-main.161/) 以及 [EENet (Ilhan et al. 2023)](https://arxiv.org/abs/2301.07099) 的趋势一致。

有意思的延伸: 训练时也需要为 early exit 做准备,否则 later-layer classifier 在训练分布上太少。这是 [LayerDrop (Fan et al.)](https://arxiv.org/abs/1909.11556) 和 [Depth-Adaptive Transformer (Xin et al.)](https://arxiv.org/abs/1910.10379) 解决的事情 — 训练时随机 drop 层,让每一层都 self-sufficient。

---

## 3. Speculative Sampling: 拿小模型当"预言家"

### 3.1 核心 mechanism

Speculative decoding(也叫 speculative sampling)由两篇并行工作提出: [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192) 和 [Chen et al. DeepMind 2023](https://arxiv.org/abs/2302.01318)。

Setup:
- **Draft model** $q$: 小、快,autoregressive 生成 $K$ 个 token $\tilde{x}_1, ..., \tilde{x}_K$
- **Target model** $p$: 大、慢,一次 forward 并行验证这 $K$ 个 token

为什么能并行验证?因为给定 $\tilde{x}_{<t}$,target model 一次 forward 就能算出 $p(\tilde{x}_t | \tilde{x}_{<t})$,而 draft model 在生成时已经知道 $\tilde{x}_{<t}$,所以这 $K$ 个 conditional distribution 可以在 target model 的 **单次 forward 里同时拿到**(因为 transformer 的 KV cache 共享 prefix)。

### 3.2 Rejection sampling 公式

对每个 speculated token $\tilde{x}_t$:
$$
r = \min\left(1, \frac{p(\tilde{x}_t | \tilde{x}_{<t})}{q(\tilde{x}_t | \tilde{x}_{<t})}\right)
$$

变量含义:
- $p(\cdot)$ — target model 的概率分布
- $q(\cdot)$ — draft model 的概率分布
- $r$ — acceptance ratio,范围 $[0, 1]$

采样一个 uniform $u \sim U[0,1]$:
- 若 $u < r$ → **accept** $\tilde{x}_t$
- 若 $u \geq r$ → **reject**,从 **修正分布** $p'(x) = \max(0, p(x) - q(x))$ 重新采样,并停止接受后续 token

**关键性质**:这个采样过程的 stationary distribution 严格等于 target model 的 distribution,即 $p(x)$。也就是说 speculative decoding 是 **distribution-preserving**,生成质量数学上等价于直接从 $p$ 采样。这是 paper 反复强调的"without compromising quality"的来源。

### 3.3 接受率与 speedup

期望接受 token 数:
$$
\mathbb{E}[\text{accepted}] = \sum_{t=1}^{K} \prod_{j=1}^{t-1} \Pr[\text{accept } \tilde{x}_j]
$$

如果 $q \approx p$(draft 与 target 接近),接受率接近 1,理论 speedup 接近 $K$。实际典型值 $K=4$–$8$,speedup 2x–3x。

### 3.4 Paper 提到的实验

- **Chinchilla 70B**: 2x–2.5x decoding speedup, distributed setup
- **Whisper**: 2x throughput speedup,见 [Speculative Decoding for Whisper (Gandhi, 2023)](https://github.com/apple/ml-fast-sd)
- **CPU inference**: 通常只有 ~1.5x,因为 CPU 上 draft 和 target 都不便宜,KV-cache 操作的相对成本更高

Intel 自己的工作 [Barad et al. 2023](https://arxiv.org/abs/2311.04951) 把 speculative sampling 和 KV-cache 优化叠在 OpenVINO 上,这个叠加是 paper 强调的"multi-pronged strategy"。

---

## 4. EAGLE: Feature-level autoregression 的妙处

### 4.1 为什么 token-level speculative 有 bottleneck

传统 speculative decoding 的 draft model 在 **token level** 自回归,这有个根本问题: token 是离散的、high-entropy 的,predict next token 本身就难。Draft model 不强 → 接受率低 → speedup 上不去。

### 4.2 EAGLE 的 key insight

[EAGLE (Li et al. 2024)](https://arxiv.org/abs/2401.15077) 的 observation: **LLM 的 second-to-top-layer feature 序列是高度 compressible 的**。也就是说,feature vector $f_t$ 在时间上变化平滑,从 $f_{t-1}$ 预测 $f_t$ 比从 $x_{t-1}$ 预测 $x_t$ 容易得多。

Draft 转移到 feature level:
$$
f_t = G(f_{t-1}, h_{t-1}^{\text{target}})
$$

$G$ 是一个轻量 draft network(可能就是一个单层 transformer + projection),输入是上一时刻的 feature $f_{t-1}$ 和 target model 给的 hidden state $h_{t-1}^{\text{target}}$。

### 4.3 处理 feature-level 的不确定性

Feature 是 continuous 的,直接 autoregressive 预测会 error accumulate。EAGLE 的 trick: **引入 target model 上一个时间步的 token embedding** 来 disambiguate:
$$
f_t = G(f_{t-1}, h_{t-1}^{\text{target}}, \text{emb}(x_{t-1}))
$$

这相当于"告诉 draft model 上一时刻到底选了哪个 token",消除条件分布的多模态。

### 4.4 性能数据(Figure 3)

paper 报告:
- **2x speedup over gpt-fast**(这是 PyTorch 官方目前最快的 LLM 推理实现,见 [gpt-fast](https://github.com/pytorch-labs/gpt-fast))
- 13B 模型上: 3x faster than vanilla autoregressive, 2x faster than Lookahead, 1.6x faster than Medusa
- **LLaMA2-Chat 70B**: latency speedup 2.7x–3.5x, throughput doubled

[EAGLE-2 (Li et al. 2024)](https://arxiv.org/abs/2406.16858) 进一步引入 **dynamic draft tree** — 根据每个节点的 confidence 动态调整树结构,而不是固定 fanout。这本质上是在 draft 阶段做 adaptive computation,和 early exit 的思想是同一个家族。

---

## 5. Medusa: 多头并行预测

[Medusa (Cai et al. 2024)](https://arxiv.org/abs/2401.10774) 走的是另一条路: 不引入独立 draft model,而是在 target model 的 hidden state 上加 **多个 extra decoding heads**,每个 head $i$ 预测位置 $t+i$ 的 token。

形式化:
$$
\{x_t^{(0)}, x_{t+1}^{(1)}, ..., x_{t+K}^{(K)}\} = \text{Heads}(h_t^{\text{target}})
$$

每个 head 给 top-$V$ candidates,组合成 **tree-structured candidates**,用 tree attention 一次 forward 验证。

与 EAGLE 的区别:
- Medusa 的 head 预测的是 **future position**,$i$-th head 直接预测 $t+i$,所以是 non-autoregressive in nature
- EAGLE 是 autoregressive 在 feature level,信息流是串行的
- Medusa 训练简单(只加几个 head + LoRA-style fine-tune),EAGLE 训练稍复杂但接受率更高

实际数据看 EAGLE 略胜,这也是 paper 里 Figure 3 的结论。

---

## 6. Lookahead Decoding: 不用 draft model

[Lookahead Decoding (Fu et al. 2024)](https://arxiv.org/abs/2402.02057) 是一个很巧的设计,完全不需要 draft model,而是利用 LLM 自己的 **n-gram cache** 做并行验证。

Idea:
- 维护一个 pool of candidate n-grams(from previous decoding steps or external corpus)
- 在每个 decoding step,把这些 n-gram 作为 candidate continuation 喂给 LLM,用 Jacobi-style iteration 同时 verify
- 接受的 n-gram 直接扩展序列

数学上:Jacobi iteration 在不动点附近收敛快,LLM 在 generate mode 下经常处在 quasi-stationary region,所以 n-gram 验证效率高。Paper 报告 Lookahead 比 vanilla 快,但被 EAGLE 反超。

---

## 7. Contextual Sparsity: Deja Vu

[Deja Vu (Liu et al. 2023)](https://arxiv.org/abs/2310.17157) 的核心发现:对于给定 input,LLM 的有效计算其实只用到了一小部分 attention head 和 MLP neuron。

形式化: 存在 input-dependent 集合 $S_{\text{attn}}(x) \subset \{1,...,H\}$ 和 $S_{\text{mlp}}(x) \subset \{1,...,N\}$,使得:
$$
\| F_{\text{dense}}(x) - F_{\text{sparse}(S)}(x) \| < \epsilon
$$

$H$ 是 attention head 数,$N$ 是 MLP hidden dim,$\epsilon$ 是可容忍误差。

关键: $S$ 是 **context-dependent**,不能用静态 pruning 拿到。Deja Vu 训练两个轻量 predictor $\pi_{\text{attn}}$ 和 $\pi_{\text{mlp}}$ 来预测 $S$:
$$
S_{\text{attn}}(x) = \text{TopK}(\pi_{\text{attn}}(h(x)), k_{\text{attn}})
$$

$\pi$ 是一个小 MLP/linear layer,$h(x)$ 是 input 的 low-level feature。

Wall-clock speedup 大约 2x,quality 几乎不丢。这是 paper 把"dynamic execution"概念推广到 **structure sparsity** 的范例 — 不是 layer-level skip,而是 operator-level skip。

---

## 8. StepSaver: Diffusion model 的 early stopping

### 8.1 Diffusion 背景公式

Diffusion model 的前向加噪:
$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_0, (1-\alpha_t) I)
$$

变量:
- $x_0$ — 原始图像
- $x_t$ — 第 $t$ 步噪声图
- $\alpha_t$ — noise schedule,控制保留 signal 的比例,$\alpha_t \in [0, 1]$,递减

反向去噪(reverse process,即 generation):
$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

$\mu_\theta$ 是神经网络预测的去噪方向,$\sigma_t$ 是 noise scale。每个 step 跑一次 $\mu_\theta$ forward,典型 $T=50$–$1000$。

### 8.2 StepSaver 的 idea

Paper Section 3.6,参考 [Yu & Barad 2024](https://arxiv.org/abs/2408.02054): 训练一个 small NLP model(本质是个 regressor/classifier)接收 text prompt,直接预测这个 prompt 需要的最小 denoising step 数 $T^*(prompt)$:
$$
T^*(p) = \arg\min_T \{ T : \text{FID}(\text{generate}(p, T)) < \text{FID}_{\text{target}} \}
$$

$p$ 是 prompt,$T^*$ 是预测的最小 steps。

### 8.3 实验数据(Figure 4 的 table)

| Denoise Steps | Performance (2757 imgs, hours) | FID (lower better) |
|---|---|---|
| fixed 30 | 1.72 | 50.67 |
| fixed 50 (baseline) | 2.85 | 50.10 |
| fixed 100 | 5.64 | 48.13 |
| **Flexi-Recommended** | **1.89** | **48.69** |

关键观察:
- Fixed 30 最快但 FID 最差(50.67)
- Fixed 100 FID 最好但耗时 5.64h
- **Flexi-Recommended**(StepSaver)耗时 1.89h(比 fixed 50 快 1.5x,比 fixed 100 快 3x),FID 48.69 — 比 fixed 50 还好,接近 fixed 100

这个数据非常 striking: dynamic recommendation 不仅快,**质量甚至超过 fixed-baseline**。Intuition 是: 简单 prompt 不需要 100 步,100 步反而可能 over-denoise 导致细节失真;StepSaver 给简单 prompt 少 step,给复杂 prompt 多 step,本质是 per-prompt 的 noise schedule 自适应。

---

## 9. LLM Routing: pipeline-level dynamic execution

[RouteLLM (Ong et al. 2024)](https://arxiv.org/abs/2406.18665) 是另一个层面的 dynamic execution: 在 pipeline 层做 routing,prompt 进入一个 router,router 决定送 strong model 还是 weak model。

形式化:
$$
\text{model}(x) = \begin{cases} M_{\text{strong}}(x) & \text{if } r(x) > \theta \\ M_{\text{weak}}(x) & \text{otherwise} \end{cases}
$$

$r(x)$ 是 router 给的"难度分"。RouteLLM 用 matrix factorization 在 preference data 上训练 router,reported 在保持 GPT-4 90% 质量的前提下节省 2x cost。

这和 early exit 在概念上完全同构: 两者都是"data-dependent 模型选择",只不过 early exit 是 layer-wise,routing 是 model-wise。

---

## 10. 整体 intuition 总结

### 10.1 统一的视角

所有 dynamic execution 方法都共享一个抽象:
$$
\text{Compute}(x) = \text{AdaptivePolicy}(x) \circ F
$$

$F$ 是 frozen model,$\circ$ 是某种 select / skip / route 操作,$\text{AdaptivePolicy}$ 决定用多少 $F$ 的 capacity。

不同方法只是在不同 **粒度** 上做这个 adaptation:

| Method | Granularity | Decision |
|---|---|---|
| Early Exit | Layer | skip later layers |
| Speculative | Token sequence | accept/reject draft tokens |
| EAGLE | Feature | draft at feature level |
| Medusa | Position | parallel multi-token heads |
| Lookahead | N-gram | verify cached n-grams |
| Contextual Sparsity | Head/Neuron | skip operators |
| StepSaver | Diffusion step | stop denoising early |
| Routing | Model | select between models |

### 10.2 为什么 dynamic execution 普遍 work?

根本原因是 **数据的难度分布是重尾的**。在 NLP、图像、speech 各种 modality 上,大部分 input 其实是"easy"的: 一个常见的 token 接一个常见的 token,一个常见的构图接一个常见的构图。这些 easy 样本不需要 full model capacity。

从信息论角度看:
$$
I(x; \text{trivial computation}) \gg 0 \quad \text{for most } x
$$

意思是 trivial computation(浅层 / 少 step / 小模型)对多数 $x$ 已经 carry 了大量信息。只有 hard $x$ 需要深层计算。Dynamic execution 就是把这个 asymmetry 显式化。

### 10.3 与生物智能的连接(paper Section 1)

Paper 引用了几个 cognitive science 参考:
- [von Neumann "The Computer and the Brain" (1958)](https://en.wikipedia.org/wiki/The_Computer_and_the_Brain) — 大脑用低精度信号
- [Bullmore & Sporns 2012 "economy of brain network organization"](https://www.nature.com/articles/nrn3214) — 大脑的 sparse wiring
- [Achterberg et al. 2023 Nature Machine Intelligence](https://www.nature.com/articles/s42256-023-00748-9) — spatially embedded RNN 解释 structural/functional neuroscience
- [Kool & Botvinick 2018 "Mental labour"](https://www.nature.com/articles/s41562-018-0494-y) — 人会按 mental effort 选策略

这里 paper 想说的是: dynamic execution 不是 ad-hoc trick,而是 brain-style efficiency principle 的工程实现。Brain 的 energy budget 有限,所以按 task difficulty 分配 compute。LLM 推理其实可以学这个。

### 10.4 未来的方向: training-aware dynamic execution

Paper 的 Discussion 部分提了一个有意思的 call to action: 现在大家都是 post-hoc 加 dynamic execution,未来应该在 **model building 阶段** 就把它当 first-class citizen。这与 [Mixture of Depths (Raposo et al. 2024)](https://arxiv.org/abs/2404.02258) 的思路一致 — 训练时让每个 token 学会"跳层",而不是推理时用 entropy threshold 强行截断。还有 [Switch Transformer (Fedus et al.)](https://arxiv.org/abs/2101.03961) 在 MoE 上做了类似的 routing。

如果训练时模型就知道 "我会被 early exit",它会自然学到 shallow feature 就足以解决 easy examples 的表示,而不会把所有判别力都堆到最后一层。这是 paper 里最 forward-looking 的一个观察。

---

## 11. 几个 paper 没提但很相关的方向

1. **ML-Speculative Decoding**([SpecBench](https://github.com/huggingface/spec-bench) 里有 benchmark) — 多 draft model 级联,先小后大
2. **PagedAttention / vLLM**([Kwon et al. 2023](https://arxiv.org/abs/2309.06180)) — 这是 memory-level 优化,与 dynamic execution 正交
3. **MoE 推理加速**([Mixtral](https://arxiv.org/abs/2401.04088) 等) — MoE 本身就是 dynamic execution 的极端版:每个 token 只 activate 2/8 experts
4. **Adaptive computation time (ACT)**([Graves 2016](https://arxiv.org/abs/1603.08983)) — RNN 时代就有的 early exit 思想,LSTM 加 halt unit 决定何时停
5. **Branch-Train-Merge**([Li et al. 2022](https://arxiv.org/abs/2208.03609)) — 把 LLM 训成多 branch,推理时 route 到对应 branch
6. **Recurrent depth / loops**([Universal Transformers (Dehghani et al.)](https://arxiv.org/abs/1807.03819), [Geiping & Berens 2024 "Recurrent Depth"](https://arxiv.org/abs/2402.02422)) — 模型按 input 难度循环不同次数,这是最 "brain-like" 的 dynamic execution

---

## 12. 工程上的 caveat

Paper 把这些方法包装得很乐观,但实际部署有几个 gotcha:

1. **Batching 与 dynamic execution 冲突**: Early exit 在 batch 推理下很难,因为同 batch 不同样本 exit point 不同,要么 padding 要么不齐,实际 speedup 在 batch size 大时被严重 dilute
2. **Speculative decoding 的 memory cost**: draft model + KV cache,显存翻倍
3. **EAGLE 训练成本**: 需要在 target model 的 feature 上训 draft,有 alignment overhead
4. **Distribution mismatch**: 如果 deployment data 和训练 router/predictor 的 data 分布不一致,acceptance rate 会塌方
5. **Hardware 适配**: GPU 高度 batch-parallel 的设计对 dynamic execution 不友好,实际上 NPU / CPU 类的 dataflow 架构更匹配,这也是 Intel 力推这些方法的部分原因(他们自己 hardware 是 CPU + PVC)

---

## References

- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding," ICML 2023](https://arxiv.org/abs/2211.17192)
- [Chen et al., "Accelerating LLM Decoding with Speculative Sampling," 2023](https://arxiv.org/abs/2302.01318)
- [Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty," 2024](https://arxiv.org/abs/2401.15077)
- [Li et al., "EAGLE-2: Dynamic Draft Trees," 2024](https://arxiv.org/abs/2406.16858)
- [Cai et al., "Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads," 2024](https://arxiv.org/abs/2401.10774)
- [Fu et al., "Lookahead Decoding," 2024](https://arxiv.org/abs/2402.02057)
- [Liu et al., "Deja Vu: Contextual Sparsity," 2023](https://arxiv.org/abs/2310.17157)
- [Yu & Barad, "StepSaver," 2024](https://arxiv.org/abs/2408.02054)
- [Ong et al., "RouteLLM," 2024](https://arxiv.org/abs/2406.18665)
- [Liao et al., "Global Past-Future Early Exit," NAACL 2021](https://aclanthology.org/2021.naacl-main.161/)
- [Ilhan et al., "EENet," 2023](https://arxiv.org/abs/2301.07099)
- [Barad et al., "Leveraging Speculative Sampling and KV-Cache for OpenVINO," 2023](https://arxiv.org/abs/2311.04951)
- [gpt-fast (PyTorch)](https://github.com/pytorch-labs/gpt-fast)
- [Optimum Intel](https://github.com/huggingface/optimum-intel)
- [OpenVINO](https://github.com/openvinotoolkit/openvino)
- [Raposo et al., "Mixture of Depths," 2024](https://arxiv.org/abs/2404.02258)
- [Graves, "Adaptive Computation Time," 2016](https://arxiv.org/abs/1603.08983)
- [Dehghani et al., "Universal Transformers," 2018](https://arxiv.org/abs/1807.03819)
- [Geiping & Berens, "Recurrent Depth," 2024](https://arxiv.org/abs/2402.02422)
- [Kool & Botvinick, "Mental labour," Nature Human Behaviour 2018](https://www.nature.com/articles/s41562-018-0494-y)
- [Achterberg et al., Nature Machine Intelligence 2023](https://www.nature.com/articles/s42256-023-00748-9)

---

如果你要我进一步深挖某一块 — 比如 EAGLE 的 draft network 架构细节,或 StepSaver 的训练数据构造,或 speculative sampling 在 batch mode 下的 batching 策略 — 都可以接着展开。这篇 paper 的价值在于把本来散落在 10+ 个 arXiv 上的方法聚到一个 taxonomy 下,但每个具体技术细节都值得单独再讲一节。
