---
source_pdf: Memory-Augmented Transformers.pdf
paper_sha256: 25e7049a11f482f59591adeb0baf17964dbe5a24feeb60e756b7745fd1f31f11
processed_at: '2026-08-05T17:31:08-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好,我换个讲法,像咱们在白板前聊天的样子,把这篇 review 的故事讲清楚。

---

## 这篇 review 在讲什么故事

一句话:Transformer 有三个老毛病,人类 brain 早就有解法,工程师们过去六年在各显神通模仿 brain,现在终于有点样子了。这篇 review 就是把这场模仿秀梳理成一张地图。

三个老毛病分别是:
- **Context window 太短**。Self-attention 是 quadratic 的,sequence 一长就炸显存。大家用 sparse attention、token pruning、KV cache 凑合,但都是伤 fidelity 的折中。
- **Parameters 训完就 frozen**。模型不 update,新知识进不去,finetune 还会 catastrophic forgetting。
- **Energy 差好几个数量级**。Brain 用 milliwatts,靠 sparse、content-addressable、localized synaptic dynamics;Transformer 用 full-context quadratic attention + autoregressive linear-per-token KV cache。

Brain 那边的解法是 **三层 memory hierarchy**:sensory memory(超短缓冲,250ms 到 3s)、working memory(4-7 chunks 的 workspace)、long-term memory(终生存储,靠 hippocampal-neocortical consolidation)。工程师们一看,这架构其实可以照搬。

---

## Brain 那边到底在干嘛(用人话)

**Sensory memory** 就是 iconic / echoic buffer,250ms 到 3s,高带宽超短保留。Brain 在这个窗口里 parallel 分析一堆 stimuli,attention 标记的进 working memory,其余 decay 掉。

Transformer 那边对应的是 token embeddings + positional encoding。但 brain 的 sensory memory 会 adaptive gain、oscillatory temporal binding,Transformer 这块还是 static。

**Working memory** 是 prefrontal cortex 维持的 workspace,4-7 chunks,靠 theta-gamma coupling 维持 persistent firing。Ventral tegmental area 的 dopamine 信号 gate update,抑制 distraction。Prefrontal cortex 像 central executive,分配 attention、切换 task、协调 specialized buffer(phonological loop、visuospatial sketchpad)。

Transformer 的 self-attention 有点像这个,selectively weight tokens within fixed context window。但缺 neuromodulatory gating、缺 oscillatory binding、缺 energy-efficient recall。

**Long-term memory** 是真的长期,可以是几年甚至一辈子。Hippocampus 做 fast encoding,然后 sleep 期间 sharp-wave ripple replay 把 index 慢慢迁移到 neocortical network,这叫 systems consolidation。Episodic memory(hippocampal pattern completion)和 semantic memory(distributed cortical)双系统。Dentate gyrus 的 adult neurogenesis 持续插入 new neuron,做 pattern separation——把相似 experience 区分开。

Transformer 那边对应的是 parameter-encoded knowledge 和 external KV bank。RAG 就是临时去外部 index fetch 文档。但缺 consolidation,finetune 时 catastrophic forgetting 严重。

---

## 六个 computational principle(这是 review 的 blueprint)

Brain 那边总结出六条 principle,后面所有工程 trick 都对应其中一条或多条。我逐条用人话翻译:

**(1) Hierarchical Resource Allocation**。不同时间尺度匹配不同存储特性。Brain 的 cortical timescale gradient 在 striatum、thalamus、cerebellum 中 topographically mirrored。工程对应就是 multi-tier memory:token → chunk → segment → archive,每一层不同的 granularity 和 latency。

**(2) Attention-Memory Bidirectional Coupling**。Attention 决定 encode 什么,memory 决定 attend 什么。Hippocampus 在 attention task 里也被招募。工程对应就是 feedback loop:retrieval 影响 encoding,encoding 又影响下次 retrieval。TransformerFAM 的 feedback attention 就是这个。

**(3) Neuromodulatory Gating and Significance Filtering**。Dopamine 调控 glutamatergic signaling 决定 plasticity induction,acetylcholine 在 synaptic 和 network 两层 orchestrate。重点是 **意识处理 <5%,无意识 >95%**——expensive conscious-like processing 应该只留给 novelty detection 和 conflict resolution,常规操作走 parallel automatic pathway。工程对应就是 surprise-gated writes:只对 novel 信号写入 memory,常规信息走 cheap path。

**(4) Replay-Based Consolidation and Interference Management**。Hippocampal replay 在 sleep 中 reactivation patterns,cortex 那边 synaptic potentiation。这解决 stability-plasticity dilemma。工程对应 EM-LLM 的 episodic segmentation + replay,A-MEM 的 memory notes graph 自组织。

**(5) Content-Addressable Associative Retrieval**。CA3 subfield 是 autoassociative network,靠 STDP 做 pattern completion——给 partial cue 能重建完整 pattern。工程对应 ARMT 的 Hopfield energy basin。

**(6) Cross-Modal Integration and Binding**。Theta-gamma coupling 实现 cortical-hippocampal binding。工程对应 multimodal memory 和 oscillation-inspired binding。

---

## 工程界都在干嘛(三波演化)

Review 里 Table 2 画了一张 2019-2025 的演化表,我把它压缩成三波:

### 第一波 2019-2021:Foundation

核心 idea 就是**把过去 hidden states 存下来,下一段再用**。

- **Transformer-XL** (2019):segment-level recurrence。把前一段的 hidden states $h_{\tau-1}$ cache 住,跟当前段 $h_\tau$ 拼接做 attention,加上 relative positional encoding。这就是把 context 从 fixed window 扩到"过去 N 段的 cache"。

- **Compressive Transformer** (2019):cache 太占地方,所以 recent states 全分辨率保留,older states 通过 learned autoencoder 压缩成 $c_{\text{old}} = f_{\text{compress}}(h_{\text{old}}; \phi)$,$\phi$ 是 compression function 参数。Temporal range 提升 38%。

- **Memformer** (2020):fixed-size external KV store + similarity-based retrieval + MRBP(Memory Replay Backpropagation)。训练 memory cost 降低 55%。

- **ABC** (2021):learned policy 决定 token retention,不是固定 heuristic。这是第一次把 read 当成 RL action 来学。

这一波的核心:memory 就是**intelligent caching**,存什么、丢什么、什么时候读,都是 hand-designed rule。

### 第二波 2022-2024:Expansion

这一波百花齐放,四个方向同时爆发:

**Retrieval scaling**:从 thousands 扩到 billions entries。
- **RETRO** (Borgeaud et al., 2022):frozen BERT retriever + differentiable cross-attention,access 2 trillion token database,GPT-3 performance with 25× fewer params。这篇是 game-changer。
- **EMAT** (Wu et al., 2022b):MIPS(maximum inner product search)把 millions QA pairs 压成 neural codes,sub-millisecond retrieval。
- **Memory Layers at Scale** (Berges et al., 2024):product-key lookup,sub-linear top-k search across billion entries,end-to-end differentiable。

Product-key decomposition 这事挺聪明的:把 $d$-维 query 分解成两个 sub-keys $q_1 \in \mathbb{R}^{d/2}$, $q_2 \in \mathbb{R}^{d/2}$,分别跟两个 codebook $C_1, C_2 \in \mathbb{R}^{N \times d/2}$ 做内积,top-k 在笛卡尔积上完成。Lookup complexity 从 $O(N)$ 降到 $O(N^{1/2})$。

**Associative / hierarchical / graph 多样化**:
- **ARMT** (Rodkin et al., 2024):Hopfield energy basin,$O(1)$ pattern completion over 50M tokens。Hopfield energy function $E(s) = -\frac{1}{2}\sum_{i,j} W_{ij} s_i s_j - \sum_i b_i s_i$,$s_i$ 是 neuron $i$ 的 state,$W_{ij}$ 是 symmetric weight,$b_i$ 是 bias。Pattern completion 通过 energy descent 找 closest stored pattern。ARMT 还加 explicit erase + orthogonal projection 防 spurious attractor——new vector 先投影到与现有 attractor 正交的 subspace:
$$v_{\text{new}}^{\perp} = v_{\text{new}} - \sum_{i} \frac{\langle v_{\text{new}}, v_i \rangle}{\|v_i\|^2} v_i$$
这其实就是 dentate gyrus pattern separation 的工程版。

- **MemGPT** (Packer et al., 2023):OS-inspired,main context + archival store,通过 function call 管理 paging,unbounded context。这就是把 OS 的 virtual memory 搬到 LLM。

- **HippoRAG** (Gutiérrez et al., 2024):concept graph inspired by hippocampus,multi-hop QA 比 RAG 提升 20%,10-30× 便宜、6-13× 快。

- **MemWalker** (Chen et al., 2023):tree of text summaries,hierarchical navigation。

**Surprise-gated writes 出现**:
- **Titans** (Behrouz et al., 2024):per-token KL divergence threshold。$\text{surprise}_t = D_{\text{KL}}(p_t(\cdot) \| p_{t-1}(\cdot))$,$p_t$ 是 time $t$ 对 next token 的预测分布,$\tau$ 是 learned threshold,surprise 超过 $\tau$ 才 write。这就是把 dopamine-gated consolidation 工程化了。MLP-based memory,test time 实时 update weight,gating 防 catastrophic interference。

- **EM-LLM** (Fountas et al., 2024):Bayesian surprise detection + graph-theoretic boundary refinement,training-free 把 sequence 分段成 episodes,10M tokens。Surprise spike 同时触发 write 和 prune。

**State-based 的精细化**:
- **TransformerFAM** (Hwang et al., 2024):feedback attention loop,每层 attend 自己上一 time step 的 latent representation。$O(L)$ 复杂度,无限 context。这就是把 static attention 变成 dynamic working memory。
- **HMT** (He et al., 2024a):三层 cache(token / chunk / segment),100K-token 单 GPU,~2% params 匹配大 long-context 模型。

### 第三波 2025:Maturation

这一波是 hybrid + test-time adaptation 主导:

- **Transformer-Squared** (Sun et al., 2025):SVD 分解 FFN weight $W = U \Sigma V^T$,expert vector 编码到 $\Sigma$ 对角 entries,inference 时 mixing network 算激活系数 $\alpha_i = f_{\text{mix}}(x; \theta_{\text{mix}})$,effective weight 是 $W_{\text{eff}} = U \cdot \text{diag}(\alpha_1, \ldots, \alpha_k) \cdot V^T$。Real-time 切换 skill,不需要 weight update。

- **ATLAS** (Behrouz et al., 2025):这是 Titans 的升级版。Omega rule 在 sliding window 级别 re-weight,polynomial feature mapping $\phi(x) = [x, x^2, \ldots, x^p]$ 扩展 MLP capacity。Test-time 用 closed-form update(类似 recursive least squares),**super-linear memory growth without gradient descent**。这点挺有意思:在 linear attention 框架下,$p$-th order polynomial feature 等价于 $p$-layer MLP 的 representation power,但用 closed-form 更新,没有 backprop 开销。

- **NAMMs** (Cetin et al., 2025):genetic algorithm 进化 layer-wise retention mask,trim KV cache 80% 仍提升 long-context benchmark,zero-shot cross-modal transfer。这是把"该记什么、该忘什么"外化成可进化的 policy。

- **LM2** (Kang et al., 2025b):每层 decoder 加 external memory module + LSTM-style gates。$m_t^{(l)} = f_t^{(l)} \odot m_{t-1}^{(l)} + i_t^{(l)} \odot \tilde{m}_t^{(l)}$,$h_t^{(l)} = o_t^{(l)} \odot \tanh(m_t^{(l)})$。Cell state 是 external store 而非 internal hidden。128k-token multi-hop reasoning。

- **MemoryOS** (Kang et al., 2025a):三级 short-term / mid-term / long-term + 四模块 Storage / Updating / Retrieval / Generation,OS abstraction。LoCoMo benchmark F1 提升 49%。

- **zip2zip** (Geng et al., 2025):inference-time adaptive vocabulary,compression-based token merging 处理 unfamiliar patterns。

- **Peripheral Memory** (Zhai et al., 2025):Kolmogorov-Arnold Networks 做 parameter-encoded memory bank,LLM 作为 processor 接口——CPU-RAM 架构类比。

- **R³mem** (Wang et al., 2025b):reversible compression,hierarchical chunking 从 paragraph 到 sentence 到 sub-sentence,bidirectional transformation 保证压缩-解压 fidelity。

- **HRM** (Wang et al., 2025a):coupled dual recurrent modules,high-level 做 abstract planning,low-level 做 detailed computation,27M params 解决复杂 reasoning。这是把"computational depth"而非"context length"作为 reasoning 的 bottleneck。

---

## Memory operations 的四个核心动作

Review 第 4 节把所有模型按 Read / Write / Forget / Capacity / Self-Management 五个动作拆开。这块最 technical,我用最直观的方式说。

### Read:从固定相似度到学习型 policy

四个层次:
1. **Content-based addressing**(NTM、DNC、Kanerva Machine):原始形式,$O(N)$ 扫一遍所有 entry。
2. **Specialised similarity search**(Memory Layers at Scale、EMAT、Memorizing Transformer):把 retrieval 优化到 sub-millisecond 或 sub-linear。
3. **Associative retrieval**(ARMT、AiT、MemReasoner、MemLong):Hopfield energy basin / low-rank prior / bidirectional GRU。给 partial cue 重建完整 pattern。
4. **Adaptive graph / policy-driven reads**(CDMem、ABC、NAMMs):不用固定 similarity metric,学一个 policy $\pi_\theta(a_t | s_t)$ 决定"什么时候读、读多深"。

Insight:read 从 fixed heuristic 演化成 learned policy,这正是 brain 那边 prefrontal cortex 作为 central executive 分配 attention 的工程对应。

### Write:从无条件覆写到 surprise-gated

最 dramatic 的演化:
- 最早是无条件 overwrite,来一个写一个。
- 后来是 LSTM-style gate,$i_t, f_t, o_t$ 三个 gate 控制 commit 量。
- 再后来是 **surprise-triggered write**:Titans 用 KL divergence threshold,ATLAS 用 Omega rule 在 sliding window 上判断。
- 最新的是 **optimization-trace write**:Memformers (Dutta & Sra, 2024) 把过去 optimization gradient 当成 first-class memory register,新 task 时 reuse 计算痕迹;ATLAS 用 closed-form polynomial feature update。

Insight:write 的本质从"存储"演化成"active learning decision"。Brain 用 dopamine 决定什么 encode,Transformer 现在用 KL surprise 决定什么 write。这是同一个 principle:**expensive processing 只给 novel / conflicting 信号,常规走 cheap path**。

### Forget:从无差别 decay 到 selective erase

- **MemLong**:prune retrieval count 低于 threshold 的 keys。
- **MeMOTR**:confidence-weighted exponential decay,$w_t = w_{t-1} \cdot \lambda^{c_t}$,$\lambda \in (0,1)$ 是 decay rate,$c_t$ 是 confidence。
- **MemoryBank** (Zhong et al., 2024a):Ebbinghaus spacing effect model,$R(t) = e^{-t/S}$,$R$ 是 retention,$t$ 是 time since last access,$S$ 是 strength(随 re-access 增加)。这直接来自 1885 年 Ebbinghaus 的 forgetting curve。
- **ARMT**:periodic normalization + hard-delete outdated vector 防 spurious attractor。
- **EM-LLM**:prediction-error spike 触发 write-and-prune 同步 cycle。

Insight:forgetting 不是 bug 是 feature。Brain 那边 sleep 期间的 synaptic pruning 主动遗忘干扰项,工程这边从 indiscriminate decay 走向 utility-aware erase。

### Capacity:compression + hierarchy + sparsity

三种 complementary tactics:
- **Compression**:Compressive Transformer 用 autoencoder 压 aged states;EMAT / MATTER 把 millions QA pairs 映射到 short neural codes;zip2zip 在 token 级做 compression-based merging。
- **Hierarchy**:MemLong chunk + prune,80K tokens 单 GPU;LM2 tree-indexed memory;HMT 三层 buffer,~2% params 匹配大 long-context 模型。
- **Sparsity**:Memory Layers at Scale 的 product-key tables 跨 GPU shard;Dynamic Memory Compression (Nawrot et al., 2024) 学 head- 和 layer-specific KV sharing,inference memory 减 4×;MLKV (Zuhri et al., 2024) 跨 layer 共享 KV head,trim cache 6×。

### Self-Management:memory 作为 autonomous resource

最前沿的一层。Memory 不再是"模型用一块 cache",而是"模型管理一个 learning subsystem":
- **Transformer-Squared**:on-the-fly expert vector routing,不需要 weight update 就扩大 functional capacity。
- **Titans**:neuromodulatory gate + learned threshold + adaptive decay。
- **ATLAS**:Omega rule 在 sliding window 上 re-weight entire spans。
- **NAMMs**:evolve layer-wise retention mask from attention statistics。
- **ARMT**:orthogonal projection 防 interference——new vector 投影到与现有 attractor 正交的 subspace 再插入。这就是 dentate gyrus pattern separation 的工程版。
- **MemReasoner**:bidirectional GRUs 迭代 reread 直到 representation 收敛,防早期 fact 被覆盖。

---

## 三个最 cool 的 takeaway

如果让我从这篇 review extract 三条 insight:

**第一,Memory 从"被动 cache"演化成"主动学习系统"**。2019 年 Transformer-XL 把 memory 当成"过去 hidden states 的拼接",2024 年 Titans / ATLAS 把 memory 当成"inference time 持续更新的 second model"。这条演化线的本质是 **decoupling computation from storage**,让 LLM 在 deployment 中持续学习——这是真正通向 lifelong learning 的路径。Memory 不再是模型的附属品,而是与 model 并行的 learning subsystem。

**第二,Surprise-gated writes 是 neuromodulatory gating 的工程对应物**。Brain 用 dopamine 决定什么 encode,Transformer 现在用 KL divergence 决定什么 write。背后是同一个 principle:expensive conscious-like processing 只用于 novel / conflicting 信息,大部分走 parallel automatic pathway。Titans 的 per-token KL surprise、ATLAS 的 per-window Omega rule、EM-LLM 的 Bayesian surprise spike,都是这个 principle 的不同 granularity 实现。

**第三,Hopfield-style associative memory 复兴**。ARMT 在 50M tokens 上做 $O(1)$ pattern completion,通过 orthogonal projection 防 interference。这直接对应 hippocampal CA3 的 autoassociative dynamics + dentate gyrus 的 pattern separation。更 deep 的 connection:**attention 本质上就是 Modern Hopfield Network 的 retrieval operation**。Modern Hopfield 的 retrieval 公式 $\xi^{\text{new}} = X \cdot \text{softmax}(\beta X^T \xi^{\text{query}})$,$X \in \mathbb{R}^{d \times N}$ 是 stored patterns,$\xi^{\text{query}}$ 是 query,$\beta$ 是 inverse temperature——这就是 attention 的形式。Ramsauer et al. 2020 已经证明 attention 和 Modern Hopfield retrieval 在数学上等价。所以 Transformer 一直都在用 associative memory,只是不自知。

这个 insight 暗示 long-context 的真正解法不是更大的 context window,而是更聪明的 associative retrieval。

---

## 现在卡在哪里

Review 第 5.2 节列了三大 challenge:

**Scalability & Retrieval Bottleneck**。Approximate similarity search 随 memory 增大 accuracy 下降;product-key 解决了 linear → sub-linear 但 billion-entry 时 parameter overhead 仍有问题;graph-based retrieval 在 graph density / traversal depth 增大时 complexity 指数增长;distributed 实现的 consistency + latency 可能 negate 理论优势;cross-modal 统一 similarity metric 在 heterogeneous 数据上系统退化。

**Memory Interference & Coordination**。这是真正瓶颈。Similar context 触发 conflicting retrieval——external memory 在 concurrent access 下 catastrophic collision,parameter-encoded 在 continual update 下 gradient interference。Stability-plasticity dilemma 在不同架构表现不同:surprise-gated 在 novelty detection 强,但 gradual knowledge drift 上证据仍有限;policy-learned 在 task adaptation 强,但 continual learning 中 overfitting 风险高。Forgetting policy 在 non-stationary 环境下失效:A-MEM 的 evolutionary approach 有风险无意删掉 rare but valuable info。

**Evaluation & Standardization Gaps**。这是最严重的 gap。不同 benchmark 在 context length、task complexity、evaluation metric 上差异巨大,无法 cross-architecture 比较。最关键的缺失:long-term adaptation 评估、memory utilization efficiency 评估、interference mitigation 评估、adversarial / memory corruption / distribution shift robustness 评估。

---

## 我自己最感兴趣的方向

Review 写到这里基本停了,但我自己最想 follow 的一条线是 **ATLAS 的 Omega rule 与 Modern Hopfield Network retrieval 的等价性**。

ATLAS 用 polynomial feature mapping $\phi(x) = [x, x^2, \ldots, x^p]$ 扩展 capacity,test-time 用 closed-form update 实现super-linear memory growth without gradient descent。这在 linear attention 框架下等价于 $p$-layer MLP 的 representation power。

而 Modern Hopfield Network 的 retrieval $\xi^{\text{new}} = X \cdot \text{softmax}(\beta X^T \xi^{\text{query}})$ 本质就是 attention。如果把 polynomial feature mapping 看作 kernel trick,那 ATLAS 的 Omega rule 其实是在 kernelized Hopfield space 里做 closed-form test-time learning。

这条线如果能和 SSM / Mamba 那边的 linear-time inference 结合,可能催生一类 new architecture:既保留 attention 的 expressiveness(因为 attention = Hopfield retrieval),又能 inference-time 学 new knowledge(因为 closed-form update),还能 linear-time decode(因为 SSM framework)。Memory-augmented transformers 现在的 bottleneck 是 quadratic attention;SSM 的 bottleneck 是 fixed state 容量小。两者通过 kernelized Hopfield 中间层结合起来,可能是 next-gen long-context lifelong learner 的雏形。

这条线还没有 paper,但 review 给的 hint 已经够多——ATLAS 那篇 2505.23735 的 Omega rule、Ramsauer 2020 的 Modern Hopfield、Mamba 的 selective state space,三者数学上应该有一个统一形式。值得 deep dive。

参考:
- ATLAS: https://arxiv.org/abs/2505.23735
- Titans: https://arxiv.org/abs/2501.00663
- Modern Hopfield Networks (Ramsauer et al., 2020): https://arxiv.org/abs/2008.02217
- EM-LLM: https://arxiv.org/abs/2407.09450
- ARMT: https://arxiv.org/abs/2407.04841
- Memory Layers at Scale: https://arxiv.org/abs/2412.09764
- NAMMs: https://arxiv.org/abs/2405.05627
- TransformerFAM: https://arxiv.org/abs/2404.09173
- HippoRAG: https://arxiv.org/abs/2405.14831
- MemGPT: https://arxiv.org/abs/2310.08560
- RETRO: https://arxiv.org/abs/2112.04426
- Memorizing Transformer: https://arxiv.org/abs/2203.08913
- Compressive Transformer: https://arxiv.org/abs/1911.05507
- Transformer-XL: https://arxiv.org/abs/1901.02860
- Memory³: https://arxiv.org/abs/2407.01178
- HRM: https://arxiv.org/abs/2506.21734
- LM2: https://arxiv.org/abs/2502.06049
- MemoryLLM: https://arxiv.org/abs/2402.04624
- R³mem: https://arxiv.org/abs/2502.15957
- Dijksterhuis et al., 2024 (pronoun reactivates concept cells): https://www.science.org/doi/10.1126/science.ado9353
- Pink et al., 2025 (episodic memory for long-term LLM agent): https://arxiv.org/abs/2502.06975
- Du et al., 2025 (rethinking memory in AI): https://arxiv.org/abs/2505.00675
- He et al., 2024 (human-inspired long-term memory survey): https://arxiv.org/abs/2411.00489

---

# Memory-Augmented Transformers: 从神经科学到工程实现的系统综述

这篇 review 的核心贡献是给 Memory-Augmented Transformers 这个相对零散的子领域建立了一套 **三维 taxonomy**——functional objectives × memory representations × integration mechanisms——同时把神经科学的 biological memory 原理作为底层的解释框架。下面我尽量按 Karpathy 你喜欢的 first-principles 视角,把架构演化、关键公式、和 biological analogy 都串起来,build some intuition for you。

---

## 1. 这篇 paper 的定位与核心问题

文章把 Transformer 的核心缺陷抽象为三件事:

1. **Quadratic attention** 限制了 context window,导致必须用 token pruning / sparse attention / KV caching 等折中,但都损害 long-range dependency 的保真度。
2. **Static parameters** 让训练后的模型无法持续学习新知识,fine-tune 还会 catastrophic forgetting。
3. **Energy inefficiency**:brain 用 milliwatts,靠 sparse / content-addressable / localized synaptic dynamics;Transformer 用 full-context quadratic attention + autoregressive linear-per-token KV cache 处理,差好几个数量级。

这些 gap 推动了 Memory-Augmented Transformers 的演化。论文的 motivation 来自神经科学的 **three-tier memory hierarchy** + **six computational principles**,这两条线下面分别展开。

参考:
- 原文 arXiv:https://arxiv.org/abs/2506.17522 (Huawei team)
- 相关综述 He et al., 2024 (human-inspired long-term memory):https://arxiv.org/abs/2411.00489
- Du et al., 2025 (rethinking memory in AI):https://arxiv.org/abs/2505.00675

---

## 2. 神经科学基础:three memory systems + six principles

### 2.1 Three-tier memory hierarchy(Figure 1 上半部分)

- **Sensory memory**:iconic ≈250ms,echoic 2-3s。高带宽、超短保留。对应 Transformer 中的 token embeddings + positional encoding。
- **Working memory**:容量 4-7 chunks(Baddeley),靠 prefrontal-parietal 的 theta-gamma coupling 维持 persistent firing,ventral tegmental area 的 dopamine gating 决定更新。对应 self-attention 的 selective weighting,但缺 neuromodulatory gating 和 oscillatory binding。
- **Long-term memory**:hippocampal → neocortical 的 systems consolidation(睡眠时 sharp-wave ripple replay),episodic(hippocampal pattern completion)+ semantic(distributed cortical)双系统,dentate gyrus 的 adult neurogenesis 提供 pattern separation。对应 parameter-encoded knowledge 和 external KV banks。

### 2.2 Six computational principles(这是整篇 review 的设计 blueprint)

这部分最值得慢看,因为后面所有工程 trick 都对应其中一条或多条:

**(1) Hierarchical Resource Allocation**:不同时间尺度匹配不同存储特性。cortical timescale gradient 在 striatum、thalamus、cerebellum 中 topographically mirrored(Raut et al., 2020)。工程对应 multi-tier memory(token → chunk → segment → archive)。

**(2) Attention-Memory Bidirectional Coupling**:hippocampus / medial temporal lobe 在 attention task 中被招募,attention 决定 encoding,memory 决定 attention(Chun & Turk-Browne, 2007)。工程对应 feedback loop between retrieval 和 encoding(TransformerFAM 的 feedback attention)。

**(3) Neuromodulatory Gating and Significance Filtering**:dopamine 调控 glutamatergic signaling 决定 plasticity induction;acetylcholine 在 synaptic 和 network 两层 orchestrate。意识处理 <5%,无意识 >95% —— 这意味着 expensive conscious-like processing 应该只保留给 novelty detection / conflict resolution,常规操作走 parallel automatic pathway(Raichle et al., 2001)。工程对应 surprise-gated writes(下面 Titans 公式)。

**(4) Replay-Based Consolidation and Interference Management**:hippocampal replay 在 sleep 中 reactivation patterns → cortex 的 synaptic potentiation。这解决了 stability-plasticity dilemma。工程对应 EM-LLM 的 episodic segmentation + replay,以及 A-MEM 的 memory notes graph 自组织。

**(5) Content-Addressable Associative Retrieval**:CA3 subfield 作为 autoassociative network,通过 spike-timing-dependent plasticity 实现 pattern completion(Rolls, 2013; Kang & Toyoizumi, 2024)。工程对应 ARMT 的 Hopfield energy basin。

**(6) Cross-Modal Integration and Binding**:theta-gamma coupling 实现 cortical-hippocampal encoding。工程对应 multimodal memory 和 oscillation-inspired binding。

**核心 insight**:effective memory 系统需要 fast/slow learning 并存,有 explicit consolidation phase,优先 associative indexing 而非 positional indexing。

---

## 3. 三个 Taxonomy 维度(Figure 2)

### 3.1 Functional Objectives(4 类)

**Temporal Context Extension** 的演化轨迹特别清晰:
- Sliding Window Attention (SWA, Beltagy et al., 2020):$O(n \cdot w)$,$w$ 是 window size。**问题**:static sensory buffer。
- ABC (Peng et al., 2021):learned control policy 决定 token retention,引入 adaptive selection。
- TransformerFAM (Hwang et al., 2024):feedback attention loops,每层 attend 自己上一 time step 的 latent representation —— 这就是"transforming static windows into dynamic working memory"。
- ATLAS (Behrouz et al., 2025):polynomial feature mapping + Omega rule,super-linear memory capacity。
- KV caching 路径:Transformer-XL → Compressive Transformer → MemoryLLM → M+ → R³mem,从静态 cache 走向 compress-on-evict + neural router + reversible compression。

**OOD Learning and Adaptation** 的核心机制是 **surprise-driven writes**:
- EM-LLM (Fountas et al., 2024):Bayesian surprise detection + graph-theoretic boundary refinement,training-free 把 sequence 分段成 episodes。
- Titans (Behrouz et al., 2024):per-token KL divergence threshold,token-by-token 决定是否写入。
- ATLAS:在 sliding window 级别用 Omega rule 决定 multi-token context 是否值得长期记忆。
- zip2zip (Geng et al., 2025):inference-time adaptive vocabulary,通过 compression-based token merging 处理 unfamiliar patterns。
- Transformer-Squared (Sun et al., 2025):SVD 分解 feedforward layer,动态 blend expert vectors,inference time 切换 skill。

**Reasoning Enhancement**:context length 与 reasoning 不线性相关,需要 associative memory 维持 coherence。Memorizing Transformer (Wu et al., 2022a) 用 kNN-retrievable memory scaling 到 262K tokens;ARMT 在 50M tokens 上做 Hopfield-style pattern completion;LM2 (Kang et al., 2025b) 在 128k-token context 上 multi-hop reasoning;HRM (Wang et al., 2025a) 用 coupled dual recurrent modules 增加 computational depth,27M params 解决复杂推理。

**Knowledge Integration**:RETRO (Borgeaud et al., 2022) 用 frozen BERT retriever + differentiable cross-attention,GPT-3 performance with 25× fewer params + 2 trillion token database;EMAT (Wu et al., 2022b) 用 MIPS 把 millions QA pairs 压成 neural codes;HippoRAG (Gutiérrez et al., 2024) 用 concept graph 在 multi-hop QA 上比 RAG 提升 20%,10-30× 便宜、6-13× 快;Memory³ (Yang et al., 2024) 把文本 KB 转成 bank of sparse retrievable parameters,1.1×10⁸ chunks 嵌入 modified FFN;MemoryOS (Kang et al., 2025a) 三级存储 short-term / mid-term / long-term + 四个核心模块(Storage / Updating / Retrieval / Generation)。

### 3.2 Memory Representations(4 类)

**(A) Parameter-Encoded Memory**

Training-time 的代表:
- **DSI** (Tay et al., 2022):把 retrieval 变成 generative task,query → document mapping 通过 attention + FFN 实现。
- **Schrödinger's Memory** (Wang & Li, 2024):LLM 能从 minimal contextual cue 重建完整 dataset —— 这是 latent memory 在"superposition"态,只有 contextual trigger 才激活。
- **Memory³**:knowledge bank 作为 sparse retrievable parameters,用 aggressive sparsification + 两阶段 pretraining。

Test-time parameter learning(更革命的范式):
- **Titans**:MLP-based memory,通过 KL surprise threshold 实时更新 weights,gating mechanism 防 catastrophic interference。
- **ATLAS**:polynomial feature mapping 扩展 MLP capacity,Omega rule 在 sliding window 上调权重 —— **super-linear memory growth without gradient descent**,这点技术上很有意思,因为通常 memory capacity 是 parameter 数量的 sub-linear function。
- **Transformer-Squared**:SVD 分解 FFN weights $W = U \Sigma V^T$,把 expert vectors 编码到 $\Sigma$ 的对角 entries,inference 时通过 specialized MLP mixing network 动态 blend:
$$W_{\text{effective}} = U \cdot \text{diag}(\alpha_1, \alpha_2, \ldots, \alpha_k) \cdot V^T, \quad \alpha_i = f_{\text{mix}}(x; \theta_{\text{mix}})$$
其中 $\alpha_i$ 是 expert $i$ 的激活系数,$f_{\text{mix}}$ 是 mixing network,$\theta_{\text{mix}}$ 是它的参数。
- **Peripheral Memory** (Zhai et al., 2025):用 Kolmogorov-Arnold Networks 实现 parameter-encoded memory banks,LLM 作为 processor 与之接口 —— 类比 CPU-RAM 架构。

**(B) State-Based Memory**

- **Transformer-XL** (Dai et al., 2019):segment-level recurrence,cache 前一段的 hidden states $h_{\tau-1}$ 与当前段 $h_\tau$ 拼接后做 attention,relative positional encoding 让模型学到相对距离。
- **Compressive Transformer** (Rae et al., 2019):recent states 全分辨率,older states 通过 learned autoencoder 压缩:
$$c_{\text{old}} = f_{\text{compress}}(h_{\text{old}}; \phi)$$
其中 $\phi$ 是 compression function 的参数,compression ratio 通过 $|c_{\text{old}}| / |h_{\text{old}}|$ 控制。Temporal range 提升 38%。
- **HMT** (He et al., 2024a):三层 cache(token / chunk / segment),100K-token 在单 GPU 上跑。
- **TransformerFAM**:每层 attend 自己前一时间步的 latent。复杂度 $O(L)$,$L$ 是序列长度。
- **RMoE** (Qiu et al., 2024):GRU-maintained hidden state 捕获 routing history,在不同 layer 间传递 routing pattern,改善 expert selection。

**(C) Explicit Storage Memory**

- **Memformer** (Wu et al., 2020):fixed-size external KV store + similarity-based retrieval + MRBP(Memory Replay Backpropagation),训练 memory cost 降低 55%。
- **MemGPT** (Packer et al., 2023):OS-inspired,main context + archival store,通过 function call 管理 paging,unbounded context。
- **Memory Layers at Scale** (Berges et al., 2024):product-key lookup,sub-linear top-k search across billion entries,end-to-end differentiable。Product-key decomposition 的关键是把 $d$-维 query 分解成两个 sub-keys $q_1 \in \mathbb{R}^{d/2}$, $q_2 \in \mathbb{R}^{d/2}$,分别与两个 codebook $C_1, C_2 \in \mathbb{R}^{N \times d/2}$ 做内积,top-k 在两个 codebook 的笛卡尔积上完成。Lookup complexity 从 $O(N)$ 降到 $O(N^{1/2})$,end-to-end 仍 differentiable。
- **Mem0** (Chhikara et al., 2025):vector embedding + graph-structured representation 混合,production-ready persistent user memory。
- **Think-in-Memory** (Liu et al., 2023) 和 **MemLLM** (Modarressi et al., 2024):triplet memory 存储 subject-object-relation,通过 external memory 而非 parametric association 回答关系查询。

**(D) Hybrid / Multi-Scale Memory**

这是 2024-2025 年的主导方向:
- **LM2**:每层 decoder 加 external memory module + learnable gates,parameter-state hybrid。
- **Titans**:state-based attention + parameter-encoded long-term memory module,test-time 调整。
- **MemGPT**:state-based working context + explicit archival storage,通过 learned paging policy 协调。
- **ATLAS** 和 **NAMMs**:根据 task demands 和 surprise signals 在不同 memory store 之间动态分配。

### 3.3 Integration Techniques(3 类)

**(A) Attention-Based Fusion**:cross-attention between layer activation 和 memory bank(Memformer),thalamocortical filtering analogy。EMAT 在 early layer 发 retrieval query,把 KV pair 传过 decoder stage。LongMem 的 SideNet 把 retrieval 与 backbone update 解耦,residual connection 混合 live input 和 cached representation。Memorizing Transformer 用 kNN attention over rolling buffer,logarithmic complexity,模拟 human recency bias。

**(B) Gated Control Mechanisms**(neuromodulatory-inspired):
- Titans 的 KL-surprise gate 模拟 norepinephrine 在 novelty detection 中的作用。
- RA-DT (Schmied et al., 2024) 用 statistical surprise-based adaptive forgetting gate,multi-task RL 中 catastrophic forgetting 降低 40%。
- MeMOTR (Gao & Wang, 2023) 用 exponential decay + confidence-driven pruning,模拟 striatal pathway 的 stability-adaptivity balance。
- NAMMs (Cetin et al., 2025) 用 genetic algorithm 进化 token retention policy,GABAergic-like inhibition 实现 stability-plasticity 平衡。这个 GABA 类比其实非常贴切:inhibition 在 biological 电路里就是选择性"关掉"不相关 representation,而 NAMMs 的 retention mask 在 attention matrix 上做的就是这个事情。

**(C) Associative Memory Integration**:
- **ARMT** (Rodkin et al., 2024):Hopfield-style energy basin,$O(1)$ pattern completion over 50M tokens。Hopfield energy function:
$$E(s) = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j - \sum_i b_i s_i$$
其中 $s_i$ 是 neuron $i$ 的状态,$W_{ij}$ 是 symmetric weight,$b_i$ 是 bias。Pattern completion 通过 energy descent 找到 closest stored pattern。ARMT 还加 explicit erase operations + orthogonal projection 防止 spurious attractors。
- **AiT** (Sun et al., 2023):low-rank memory priors 作为 attractors,global workspace architecture。
- **MemReasoner** (Ko et al., 2024):bidirectional GRUs 实现 iterative read-update cycle。

---

## 4. Memory Operations 详解(Section 4 + Table 1)

这一部分最 technical,也是最值得细看的地方。

### 4.1 Read Operations

四个层次:
1. **Content-based addressing**(NTM, DNC, Kanerva Machine)—— 原始形式。
2. **Specialised similarity search**(Memory Layers at Scale, EMAT, Memorizing Transformer)—— 把 retrieval 优化到 sub-millisecond 或 sub-linear。
3. **Associative retrieval**(ARMT, AiT, MemReasoner, MemLong)—— Hopfield / low-rank prior / bidirectional GRU。
4. **Adaptive graph / policy-driven reads**(CDMem, ABC, NAMMs)—— 不再用固定 similarity,而是学习何时 / 多深地 probe memory。

**ABC (Peng et al., 2021)** 的核心 insight:read 不应该是 fixed heuristic,而是 learned policy $\pi_\theta(a_t | s_t)$,$s_t$ 是 memory state,$a_t$ 是 read action(包括 "什么时候读"、"读多深"),通过 RL 优化。

### 4.2 Write Operations

写策略演化最 dramatic:从 unconditional overwrite → gated writes → surprise-triggered writes → optimization-trace writes。

**Titans 的 surprise-gated write 公式**(精炼版):
$$\text{surprise}_t = D_{\text{KL}}\left(p_t(\cdot) \,\|\, p_{t-1}(\cdot)\right)$$
$$\text{write}_t = \mathbb{1}[\text{surprise}_t > \tau]$$
其中 $p_t(\cdot)$ 是模型在 time $t$ 对 next token 的预测分布,$\tau$ 是 learned threshold。surprise 高时写入,低时跳过 + decay。这模拟 dopamine-gated consolidation:rare events 优先编码。

**LM2 的 per-layer gating**:每层有自己的 input gate $i_t^{(l)}$、forget gate $f_t^{(l)}$、output gate $o_t^{(l)}$,控制 external store 在该层的 commit 量:
$$m_t^{(l)} = f_t^{(l)} \odot m_{t-1}^{(l)} + i_t^{(l)} \odot \tilde{m}_t^{(l)}$$
$$h_t^{(l)} = o_t^{(l)} \odot \tanh(m_t^{(l)})$$
这本质是 LSTM 的 memory cell,但 cell state 是 external store 而非 internal hidden state。

**ATLAS 的 Omega rule**:在 sliding window 上对 multi-token context 调权重,而非 per-token。配合 polynomial feature mapping:
$$\phi(x) = [\,x, x^2, x^3, \ldots, x^p\,]$$
把 input 投影到 $p$ 次多项式特征空间,在 test time 通过 closed-form update(类似 recursive least squares)实现 super-linear capacity growth without gradient descent。这是把 kernel methods 和 test-time learning 结合的思路 —— 在 linear attention 的框架下,$p$-th order polynomial feature 等价于 $p$-layer MLP 的 representation power,但用 closed-form 更新。

### 4.3 Forgetting Dynamics

从 indiscriminate decay → selective learned erase:
- **MemLong**:prune retrieval count 低于 threshold 的 keys。
- **MeMOTR**:confidence-weighted exponential decay,$w_t = w_{t-1} \cdot \lambda^{c_t}$,$\lambda \in (0,1)$ 是 decay rate,$c_t$ 是 confidence。
- **Titans** & **LM2** & **ATLAS**:adaptive gate-controlled decay。
- **ARMT**:periodic normalization + hard-delete outdated vectors 防 spurious attractor。
- **MemoryBank** (Zhong et al., 2024a):Ebbinghaus spacing effect 模型:
$$R(t) = e^{-t/S}$$
$R$ 是 retention,$t$ 是 time since last access,$S$ 是 strength(随 re-access 增加)。这直接来自 1885 年 Ebbinghaus 的 forgetting curve。
- **EM-LLM**:prediction-error spike 触发 write-and-prune 同步 cycle。

### 4.4 Capacity Optimization

三种 complementary tactics:
- **Compression**:Compressive Transformer 用 autoencoder 把 aged states 压缩;EMAT / MATTER 把 millions QA pairs / documents 映射到 short neural codes;zip2zip 在 token 级别做 compression-based merging。
- **Hierarchy**:MemLong chunk + prune,80K tokens 单 GPU;LM2 tree-indexed memory;HMT 三层 buffer,~2% params 匹配大 long-context 模型。
- **Sparsity**:Memory Layers at Scale 的 product-key tables 跨 GPU shard,sub-linear compute;Dynamic Memory Compression (Nawrot et al., 2024) 学 head- 和 layer-specific KV sharing,inference memory 减到 4×;MLKV (Zuhri et al., 2024) 跨 layer 共享 KV head,trim cache 6×。

### 4.5 Self-Management and Adaptation

最前沿的一层 —— memory 作为 autonomous resource:
- **Transformer-Squared**:on-the-fly expert vector routing,不需要 weight update 就扩大 functional capacity。
- **Titans**:neuromodulatory gate + learned threshold + adaptive decay。
- **ATLAS**:Omega rule 在 sliding window 上 re-weight entire spans。
- **NAMMs**:evolve layer-wise retention masks 从 attention statistics,trim KV cache 80% 仍提升 long-context benchmark,zero-shot cross-modal transfer。
- **ARMT**:orthogonal projection 把 new vector 投影到与现有 attractor 正交的 subspace 后再插入 —— 这其实就对应生物 CA3 的 pattern separation 机制:用 orthogonal coding 防 interference。具体:
$$v_{\text{new}}^{\perp} = v_{\text{new}} - \sum_{i} \frac{\langle v_{\text{new}}, v_i \rangle}{\|v_i\|^2} v_i$$
然后写入 $v_{\text{new}}^{\perp}$,保证与现有 stored vectors 的 inner product 接近 0。
- **MemReasoner**:bidirectional GRUs 迭代 reread 直到 representation 收敛,防早期 fact 被覆盖。

---

## 5. 演化轨迹(Section 5.1 + Table 2)

Table 2 是这篇 review 的"演化地图",值得仔细读:

- **2019-2021 Foundation**:Transformer-XL(2019, S/Plg)、Compressive Transformer(2019, E/Plg)、Memformer(2020, E/Plg)、ABC(2021, E)、WorkMATe(2021, S)。
- **2022-2024 Expansion**:EMAT、RETRO、DSI、Memorizing Transformer、MemBART、LongMem、MemGPT、MemWalker、AiT、AdaTape、MeMOTR、MemoryBank、TransformerFAM、HMT、MemoryLLM、HippoRAG、MATTER、Memory³、ARMT、MemLong、MemReasoner、EM-LLM、RA-DT、Memory Layers at Scale、Titans、RMoE、Schrödinger's Memory。这一阶段是 retrieval-augmented scaling(从 thousands 到 billions entries)+ associative / hierarchical / graph 多样化 + surprise-gated writes 出现。
- **2025 Maturation**:Transformer-Squared、LM2、NAMMs、R³mem、Memory-R+、Mem0、CDMem、ATLAS、MemoryOS、zip2zip、Peripheral Memory、MALT Diffusion、A-MEM、HRM。Hybrid storage + test-time adaptation + specialized access 成主流。

**Legend 中的几个标签**值得明确:
- Storage Class:P / S / E / H(PS / PE / SE / PSE)
- Integration Method:Plg (Plug-in) / Wrp (Wrapper) / Bsp (Bespoke redesign) —— 这个区分对 backbone compatibility 很关键,Plg 最好部署,Bsp 性能最强但改架构。
- Write Trigger:Stc / Sur / Pol / G
- Plasticity:F (Fixed after training) / TT (Testtime adaptable) —— **TT 在 2024 后明显成为默认**。

**收敛趋势**:
1. Hybrid storage 主导(单一 memory type 已少见)。
2. Write 从 static schedule → surprise-gated / policy-learned。
3. Retrieval 从 attention + similarity → graph / associative / hierarchical / expert routing。
4. Forgetting 从 FIFO / decay → LRU / selective / cycle-based / evolutionary。

---

## 6. Challenges 与 Open Problems(Section 5.2)

### 6.1 Scalability & Retrieval Bottlenecks

- Approximate similarity search 随 memory 增大 accuracy 下降。
- Product-key 解决了 linear → sub-linear,但 parameter overhead 在 billion-entry 系统上仍有问题。
- Graph-based retrieval 在 multi-hop reasoning 上强,但 graph density / traversal depth 增大时 complexity 指数增长。
- Distributed 实现的 consistency + latency 问题可能 negate 理论优势。
- Cross-modal 统一 similarity metric 在 heterogeneous 数据上系统退化。

### 6.2 Memory Interference & Coordination

- **Stability-plasticity dilemma** 在不同架构表现不同:surprise-gated 在 novelty detection 强,但 gradual knowledge drift 上证据仍有限。
- Policy-learned 在 task adaptation 强,但 continual learning 中 overfitting 风险高。
- **Memory interference** 是真正瓶颈:similar context 触发 conflicting retrieval。External memory 在 concurrent access 下 catastrophic collision;parameter-encoded 在 continual update 下 gradient interference。
- Forgetting policy 在 non-stationary 环境下:LRU / selective 在 structured 环境强,但 relevance pattern 不可预测时失效。A-MEM 的 evolutionary approach 有风险无意删掉 rare but valuable info。

### 6.3 Evaluation & Standardization Gaps

这是目前最严重的 gap —— 不同 benchmark 在 context length / task complexity / evaluation metric 上差异巨大,无法 cross-architecture 比较。最关键的缺失:
- 缺乏 long-term adaptation 评估
- 缺乏 memory utilization efficiency 评估
- 缺乏 interference mitigation 评估
- 缺乏 adversarial / memory corruption / distribution shift 的 robustness 评估

---

## 7. Future Directions(Section 5.3)

三个主要方向:

1. **Cognitive Flexibility and Lifelong Learning**:decoupling computation from storage,让模型在 deployment 时 tap external / hybrid memory banks 而无需 retraining。Test-time training + memory-driven optimization + selective forgetting + zero-shot transfer。
   - 参考 Dijksterhuis et al., 2024:concept cells 在 pronoun 引用时 reactivate,类似 LLM 把 episodic memory 整合到 parametric memory。https://www.science.org/doi/10.1126/science.ado9353
   - Pink et al., 2025:episodic memory 作为 long-term LLM agent 缺失组件的 position paper。https://arxiv.org/abs/2502.06975

2. **Human-Like Cognition for Agents**:short-term / working / long-term memory三层 + vector database + episodic / procedural 知识 retrieval。Open problems:
   - Agent 没能分离 episodic vs semantic,导致 recall 冲突。
   - Repeatedly attempt failed subtasks 而不利用 episodic feedback。
   - Static / manually defined metadata 限制 retrieval quality —— 需 agent 学动态 metadata。
   - Long-term memory 与 ontology / knowledge graph 整合增强 contextual grounding。

3. **Future Architectures and Ethical Considerations**:test-time training + memory-driven optimization + zero-shot transfer + multimodal memory + collaborative agent networks。但在 healthcare / education / personalized service 部署时,需要 explainable memory operations / data auditing / bias mitigation / user control。

---

## 8. Build Your Intuition:核心 takeaway

如果让我从这篇 review 中 extract 三条 intuition:

**第一,Memory 在 Transformer 中已经从"被动 cache"演化成"主动学习系统"**。2019 年 Transformer-XL 把 memory 当成"过去 hidden states 的拼接",2024 年 Titans / ATLAS 把 memory 当成"inference time 持续更新的 second model"。这条演化线的本质是 **decoupling computation from storage**,让 LLM 在 deployment 中持续学习 —— 这是真正通向 lifelong learning 的路径。

**第二,Surprise-gated writes 是 neuromodulatory gating 在工程上的对应物**。Brain 用 dopamine 决定什么 encode,Transformer 现在用 KL divergence 决定什么 write。这背后是同一个 insight:**expensive processing 应该只用在 novel / conflicting 信息上**,大部分信息应该走 parallel automatic pathway。Titans 的 per-token KL surprise 和 ATLAS 的 per-window Omega rule,都是这个 principle 的不同 granularity 实现。

**第三,Hopfield-style associative memory 的复兴**。ARMT 在 50M tokens 上做 $O(1)$ pattern completion,通过 orthogonal projection 防 interference —— 这直接对应 hippocampal CA3 的 autoassociative dynamics + dentate gyrus 的 pattern separation。这暗示着 long-context 的真正解法可能不是更大的 context window,而是更聪明的 associative retrieval。Hopfield energy function 在 2024 年通过 Modern Hopfield Networks (Ramsauer et al., 2020) 重新回到主流,ARMT 是这条线在 long-context 上的延伸。Modern Hopfield 的 retrieval 公式:
$$\xi^{\text{new}} = X \cdot \text{softmax}(\beta X^T \xi^{\text{query}})$$
其中 $X \in \mathbb{R}^{d \times N}$ 是 stored patterns,$\xi^{\text{query}}$ 是 query pattern,$\beta$ 是 inverse temperature(控制 retrieval 的 sharpness)。这其实就是 attention 的形式 —— **attention 本质上就是 Modern Hopfield Network 的 retrieval operation**,这个 connection 在 Ramsauer et al. 2020 已经被证明。所以从某种意义上,Transformer 一直都在用 associative memory,只是不自知。

参考:
- Modern Hopfield Networks (Ramsauer et al., 2020):https://arxiv.org/abs/2008.02217
- ARMT:https://arxiv.org/abs/2407.04841
- Titans:https://arxiv.org/abs/2501.00663
- ATLAS:https://arxiv.org/abs/2505.23735
- EM-LLM:https://arxiv.org/abs/2407.09450
- TransformerFAM:https://arxiv.org/abs/2404.09173
- Memory Layers at Scale:https://arxiv.org/abs/2412.09764
- Memory³:https://arxiv.org/abs/2407.01178
- HippoRAG:https://arxiv.org/abs/2405.14831
- MemGPT:https://arxiv.org/abs/2310.08560
- Compressive Transformer:https://arxiv.org/abs/1911.05507
- Transformer-XL:https://arxiv.org/abs/1901.02860
- Memorizing Transformer:https://arxiv.org/abs/2203.08913
- MemoryLLM:https://arxiv.org/abs/2402.04624
- R³mem:https://arxiv.org/abs/2502.15957
- NAMMs:https://arxiv.org/abs/2405.05627 (ICLR 2025)
- Transformer-Squared:https://arxiv.org/abs/2505.05627
- HRM:https://arxiv.org/abs/2506.21734

如果你想 deep-dive 某条线,我个人最感兴趣的是 **ATLAS 的 Omega rule** 与 **Modern Hopfield Network 的 retrieval** 之间的等价性,以及这种 closed-form test-time learning 能否推广到更深的 representation hierarchy —— 这条线如果能与 SSM / Mamba 结合,可能催生一类 new architecture,既保留 attention 的 expressiveness 又能 inference-time 学 new knowledge。
