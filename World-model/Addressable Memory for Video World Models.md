---
source_pdf: Addressable Memory for Video World Models.pdf
paper_sha256: d9fb2a5a98fe0b3bac20df23d223d3f9494b9785d26a3edf800eb6d66161f37d
processed_at: '2026-08-17T23:59:19-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 WorldTrace

## 一、先说清楚这是个啥问题

想象你在玩一个 AI 生成的开放世界游戏——不是真正的 Unreal Engine，是一个 transformer 在实时生成画面。你往前走，画面就生成；你转头，画面跟着变。

这种模型叫 **video world model**（视频世界模型），比如 NVIDIA 的 Matrix-Game、Decart 的 Oasis、Google 的 GameNGen。它们的工作方式是 **chunk by chunk** 生成：每生成一小段（比如 3 帧-latent），就把这段塞进一个叫 **KV cache** 的内存里，下一段生成时通过 attention 去"读"这个 cache，知道之前发生了什么。

听起来挺好。但有个致命问题：**只能撑几秒钟**。一旦生成超过训练时见过的 context 长度，画面就开始崩——几何乱掉、场景不一致、细节漂移。

大家以前以为是"内存满了，老内容被删了，所以忘了"。但这篇 paper 说：**内容其实还在 cache 里，是模型"找不到"了**。

打个比方：图书馆的书还在架子上，但目录卡上的索书号变成乱码了，你按编号去找书，根本找不着。问题不在"书丢了"，问题在"索引坏了"。

## 二、为什么会"找不到"——RoPE 的坑

transformer 里有个东西叫 **RoPE**（Rotary Positional Embedding），它把"这个 token 在第几帧"的信息编码成向量旋转的角度。query 和 key 之间的 attention score，部分取决于它们位置相差多少——这个差值（offset）会变成一个旋转角度，影响 attention。

训练时模型只在 offset $0$ 到 $5$ 之间见过（MG2 的训练 context）。一旦你生成到第 100 帧，想 attention 到第 1 帧，offset 就是 99——**训练时从来没见过这么大的 offset**。

这里有个细节：RoPE 有很多个 "frequency"，每个 frequency 旋转速度不同。
- **高频（快转）**：offset 变 1 步，相位就变 1 弧度。offset 一大，相位 wrap 好几圈，attention score 变成随机数——这部分废了。
- **低频（慢转）**：offset 变 100 步，相位才转一点点，cos 值几乎不变——这部分虽然稳定，但区分不了不同位置的 key，等于没有 positional 信息。

所以模型面对远端的 key，既没有精确的 positional 信号（高频废了），也区分不开谁是谁（低频不转），attention 就乱套了。这就是"找不到"的物理原因。

## 三、为什么"压缩"也会出问题

光解决"放在哪"还不够，还得解决"放什么"。cache 内存有限，你不可能把所有历史帧都 verbatim 存着。常见做法是 averaging——把多帧 key 平均成一个 summary slot。

但这里有个坑：如果你**直接在 RoPE 旋转后的空间里做 averaging**，相当于把指向不同方向的向量加起来平均。如果两帧时间差大，RoPE 把它们旋转到差不多相反方向，加起来互相抵消，summary 就被压扁了。

这就是"phase cancellation"——不同时间戳的 key 旋转到不同角度，平均时信号互相抵消。

## 四、两个问题耦合在一起

最 tricky 的是：**位置和内容耦合**。

- 你只修位置（把 summary 放到 in-distribution 的虚拟位置），但内容是被 phase cancellation 压扁的 averaging，模型读到了也没用。
- 你只修内容（用 canonical averaging 避免 cancellation），但位置是 OOD 的，模型根本 attention 不到。

之前的方案大多只修一个：
- **YaRN**：rescale RoPE frequency 把 offset 拉回 in-distribution 范围。修了位置，但内存随 $N$ 线性增长，跑长就 OOM。
- **Block-relative RoPE（Infinity-RoPE）**：把所有超出训练范围的 offset 都 cap 到 $\Delta t_{\text{train}}$。但这导致所有老 slot 都塌缩到同一个位置——多个 slot 变成"位置上不可区分"，attention 又分不清谁是谁了。这叫 "Block-relative collapse"。
- **MemRoPE**：用两个 EMA token（长期+短期）做 summary。只支持两个 summary，扩展性有限。

## 五、WorldTrace 怎么做

WorldTrace 同时解决两个问题，而且 **training-free**——不重新训练模型，只在 inference 时改 cache 的读写方式。

### 第一步：给每个 summary slot 分配一个"虚拟位置"

公式很简单：

$$t_s^v = q - (L_{\text{attn}} - 1 - s)$$

意思是：第 $s$ 个 summary slot 的虚拟位置，只取决于 $s$（slot 排名）和 $q$（当前 query 位置），跟生成到第几帧 $N$ 完全无关。

效果：
- 永远在训练见过的范围内（in-distribution）
- 每个 slot 有不同的 offset，互不塌缩
- 不管视频生成到多长，公式都一样

### 第二步：在 canonical space 做 averaging

WorldTrace-Field 的做法是三步：

1. **Unrotate**：把每个源帧的 key 旋转回"位置 0"（canonical space），去掉原始时间戳的 RoPE 旋转
2. **Average**：在 canonical space 里做平均，不会有 phase cancellation
3. **Re-rotate**：把平均后的 key 旋转到这个 slot 的虚拟位置

这样 summary slot 在虚拟位置上产生的 attention score，正好等于把所有源帧都"假想地"放到这个位置上后 attention score 的平均值。这是 paper 里 Proposition 2 给的数学保证。

还有个 subtle 的设计：canonical averaging 会让 summary key 的 norm 缩小到约 $1/\sqrt{M}$。作者**故意不**用 $\sqrt{M}$ 重新 rescale，因为更小的 logit 让 softmax 自然给"被严重压缩的远古历史"更小的权重——相当于 implicit 地告诉模型"远古的内容可信度低一点"，反而合理。

### 第三步：Landmark 模式——保留关键场景的原始内容

WorldTrace-Field 是 averaging，适合"连贯性"——比如你沿着一条路往前走，每一段都跟前一段平滑衔接。但如果你**绕一圈回到原点**，averaging 就不行了，因为它把细节都 blur 掉了，你认不出原来的场景。

这时候需要 **WorldTrace-Landmark**：不 average，直接 verbatim 保留"关键场景入口"的原始 key。

怎么判断哪帧是"关键场景入口"？用相邻帧 canonical key 的 cosine distance，距离 spike 的就是 scene-entry event（比如从走廊走进大厅，画面突变）。

还有一个工程细节：每个 chunk 生成后，cache 整体往后 shift 一个 slot。如果每次 shift 都做 unrotate-rerotate，bfloat16 精度会累积 floating-point drift。WorldTrace-Landmark 的解法是 **frozen canonical key**——在 landmark 时刻把 canonical key 计算一次永久 frozen，之后每次 shift 只重新做一次旋转，避免反复 unrotate-rerotate 的精度损失。

## 六、LoopBench：怎么测试"绕回来还认得出来"

作者造了个 benchmark 叫 **LoopBench**，专门测试 episodic recall（情节回忆）。

思路很 elegant：让模型生成一个"绕圈"轨迹——从 A 出发，经过 B、C、D，再回到 A。第一次访问 A 时生成的画面作为 reference，回到 A 时生成的画面作为 prediction，算两者的 CLIP 相似度。**不需要外部 ground-truth video，rollout 自己产生 reference 和 prediction**。

四个难度轴：
1. **Topology**：ABA（直返）、ABCA（三角）、ABCDA（方形）—— waypoint 越多越难
2. **Rollout length** $N$：detour 越长，cached offset 越出分布
3. **Camera orientation**：原地 pan 90°/180°/360° 再回来
4. **Multi-revisit** $R$：ABABA（重访 A 两次）、ABCBA、ABCDBA

## 七、结果如何

几个关键数字：

- **WorldTrace-Field 在连贯性上**：$N=48$ 时 TempSSIM 比 sliding window 提升 +15.5%
- **WorldTrace-Landmark 在回忆上**：ABA 上 PAC 比 sliding window 提升 +19.5%
- **长 horizon scaling**：$N=256$ 时，compression 方案 PAC 跌到 0.6，canonical-K anchoring 跌到 0.61，**WorldTrace-Landmark 还保持在 0.99**——只有 verbatim landmark 能 scale 到 minute-scale

这个数字说明：**averaging/compression 总会损失，只有 verbatim 保留关键帧才能真正 scale**。

## 八、Ablation 里最关键的发现

Table 4 是最 informative 的 ablation。他们同时变 position assignment 和 content compression：

- 只改 content（naive averaging + Block-relative position）：跟 sliding window 差不多，**没用**
- 只改 position（Field averaging + Block-relative position）：跟 sliding window 一样，**也没用**
- 两个都改（WorldTrace）：才有效

这说明：**位置和内容必须联合设计，单独修一个等于白修**。这是 paper 最核心的 insight 之一。

## 九、跟其他工作的关系

**LLM 的 KV cache compression（SnapKV、H2O、StreamingLLM）**：主要解决"哪些 token 该保留"，不解决"summary slot 该放在哪"——因为 LLM token 是个体，保留就在原位置，不存在 merge 后放哪的问题。video 里每个 chunk 需要 merge 多帧进一个 slot，merge 后的位置归属是 video 独有的问题。

**Landmark Attention**：vocabulary 类似，但它是 trained representative，WorldTrace 是 frozen canonical key + slot-rank position，而且 training-free。

**Memorizing Transformers / Compressive Transformers**：也用"recent + compressed memory"两层结构，但用 trained memory tokens 或 external kNN。WorldTrace 强调的是"位置可寻址性"——不只是"有 memory"，而是"memory 能被 attention 找到"。

## 十、我脑子里冒出来的几个延伸

1. **Geometry-aware canonical keys**：现在 canonical averaging 假设 source frames 是"同一场景的 views"。如果用 Plücker camera rays 做 warp，average 同一 scene content 的 frames，summary 会更 informative。LingBot-World 上 WorldTrace-Field 没显著提升，可能正是 Plücker conditioning 已经提供了这个 signal。

2. **Learned scene-entry policy**：现在 scene-entry detection 是 hand-crafted cosine threshold。训练一个小 policy 来决定何时 commit landmark、何时 evict，应该能 push recall 更高。

3. **Hierarchical memory**：现在只有 2 层（recent + summary）。能否做 3 层？类似 RAID 的 tiered storage——hot tier（verbatim recent）、warm tier（field averages）、cold tier（sparse landmarks）。

4. **Revisit-supervised memory training**：把"recall 成功"作为 reward，RL 训练 cache policy。

5. **Cross-architecture**：在 SSM-based world model（Mamba、SANA-WM）上呢？SSM 不用 attention，addressability 概念不同，但 state compression 的本质问题类似。

## 十一、一句话总结

如果让我用最朴素的话概括 WorldTrace 的核心 insight：

> **在用 RoPE 的 attention 里，"记忆"必须同时满足两个条件才能被用上——内容得在"训练见过的相对距离"上（addressable），内容得没被压缩压扁（informative）。WorldTrace 把这两个问题分开解决：用 slot-rank virtual position 解决"放哪"，用 canonical-space averaging 或 frozen landmark key 解决"放什么"，两者联合设计，让 O(1) 的 cache 在任意长视频生成都能 retrieve 到过去的内容。**

这个 insight 不只对 video world model 有用。任何用 RoPE + KV cache + 需要超长 horizon 的 autoregressive model——未来的 LLM agents、robot simulators、game engines——都会撞到这堵墙。WorldTrace 给出了一个非常 clean 的 solution template。

---

参考链接：
- WorldTrace project page: https://research.nvidia.com/labs/sil/projects/WorldTrace/
- RoPE 原文 (RoFormer): https://arxiv.org/abs/2104.09864
- YaRN: https://arxiv.org/abs/2309.00071
- Infinity-RoPE (Block-relative): https://arxiv.org/abs/2511.20649
- MemRoPE: https://arxiv.org/abs/2603.12513
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Matrix-Game 2.0: https://arxiv.org/abs/2508.13009
- Landmark Attention: https://arxiv.org/abs/2305.16300
- Memorizing Transformers: https://arxiv.org/abs/2203.12145
- Compressive Transformers: https://arxiv.org/abs/1911.05507
- StreamingLLM (attention sink): https://arxiv.org/abs/2309.17453
- BFloat16 RoPE drift: https://arxiv.org/abs/2411.13476
- Oasis (Decart): https://oasis-model.github.io

---

# WorldTrace: 为视频世界模型建立可寻址的长期记忆

让我从这篇NVIDIA/Princeton的工作讲起，这是一篇很"Karpathy style"的paper——它不是堆参数堆数据，而是把一个看似复杂的现象拆解到可以凭直觉理解的核心机制上。

## 一、问题的真正本质：Addressability vs. Storage

我们先要建立最核心的intuition。Autoregressive video world model（比如 Matrix-Game-2、Oasis、GameNGen 这一类）用 KV cache 来承载过去的视觉上下文，每个新生成的 chunk 都通过 attention 去"读"这个 cache。当 rollout 超过训练时见过的 context 长度 $\Delta t_{\text{train}}$ 时，模型开始崩坏——视觉质量退化、场景不再 consistent。

传统的诊断是："cache满了，老的内容被驱逐了，所以忘了"。但作者给出的诊断更深刻：**问题不在于内容丢失，而在于内容变得不可寻址（unaddressable）**。

这个区分非常重要。打个比方：你图书馆里书还在，但卡片目录上的索引号是乱码，你找不着了。这就是 WorldTrace 要解决的核心问题。

## 二、两个耦合的失败模式

### 2.1 Position determines addressability

让我从 RoPE（Rotary Positional Embedding，https://arxiv.org/abs/2104.09864）的数学讲起。RoPE 的精髓是把位置信息编码成 query/key 向量的旋转。

对于一个 RoPE frequency pair $f$（对应 head dimension 中的一对维度），query 在位置 $q$ 与 key 在位置 $k$ 的 attention score 贡献为：

$$a_{q,k}^f = \text{Re}\left( A_{q,k}^f \cdot e^{i\theta_f \delta_{q,k}} \right)$$

这里变量含义：
- $q$：当前 query frame 的绝对时间戳
- $k$：cached key frame 的绝对时间戳  
- $\delta_{q,k} = q - k$：相对位置偏移（causal attention 下 $k \leq q$）
- $\theta_f = \theta^{-f/c_t}$：第 $f$ 个 frequency pair 的角频率，$c_t$ 是时间轴 RoPE 对的数量（MG2 里 $c_t=22$，对应 44 维时间位置编码）
- $A_{q,k}^f$：在 canonical coordinates 下 query-key 的内容相似度
- $\text{Re}(\cdot)$：复数取实部

训练时模型只在 $\delta_{q,k} \in [0, \Delta t_{\text{train}}]$（MG2 是 $\Delta t_{\text{train}}=5$ 个 latent frame）内见过这个相对偏移。当 rollout 到 $N=48$ chunks（每 chunk 3 latent frame，所以 144 latent frames）时，远端的 key 相对当前 query 的 offset 达到 $\sim 140$，远超训练范围。

这里有个非常关键的细节，作者用 Figure 7 把它讲得很清楚。RoPE 的不同 frequency 组件的"波长"不同：

- **Fast components（高频，小 $f$）**：$\theta_f \approx 1.0$（$f=0$），旋转快，offset 变 1 步相位变 1 rad。在长 horizon 下，$\theta_f \cdot |\delta_{q,k}|$ 超过 $\pi$、甚至达到 $3\pi$、$8.5\pi$，相位 wrapping 多次，cos 值变成"在每次 offset 下随机抖动"——这部分的 positional 信号变成了噪声。
- **Slow components（低频，大 $f$）**：$\theta_f \approx 1.5 \times 10^{-4}$（$f=21$），旋转慢，offset 即使到 100，相位也只转了 $0.015$ rad，几乎不变——这部分是稳定的"semantic carrier"。

所以失败模式是这样：当 query 想去 retrieve 一个距离 100 步远的 key，fast components 给出乱七八糟的 phase，slow components 又无法区分不同位置的 key（因为它们相位几乎不变，cos $\approx 1$）。结果是 attention 分布退化，模型 retrieve 不到正确的内容。

这个分析对应 YaRN（https://arxiv.org/abs/2309.00071）的 NTK-by-parts 视角。WorldTrace 的切入点是：与其重映射 frequencies（YaRN 路线），不如**把要被记住的 key 重新放到一个 in-distribution 的虚拟位置上**。

### 2.2 Content compression determines informativeness

即使我们解决了"放在哪"，还要解决"放什么"。常见的压缩方式是 averaging，但 naive averaging 在 RoPE-rotated space 会出问题：

$$\bar{K}_{\text{naive}}^f = \frac{1}{M}\sum_{m=1}^{M} R(\theta_f t_m) K_m^f$$

变量说明：
- $M$：被压缩到一个 slot 的源帧数
- $t_m$：第 $m$ 个源帧的原始时间戳
- $K_m^f$：第 $m$ 帧在 frequency $f$ 上的 canonical（未旋转）key
- $R(\theta_f t_m)$：旋转矩阵，把 canonical key 旋转到位置 $t_m$

问题在于，不同 $m$ 对应不同的 $t_m$，所以 $R(\theta_f t_m) K_m^f$ 指向不同方向。如果 $t_1, t_2$ 距离近，旋转角度差小，averaging 还能保留信号；如果距离远（$\theta_f(t_2-t_1) \sim \pi$），两个向量指向相反方向，averaging 时 cancel out，summary slot 被压扁。

Table 5 的数据非常直观地展示了这一点：naive averaging 下 LatentDif 从 $N_s=1$ 的 0.257 增长到 $N_s=4$ 的 0.312（slot 越多，每个 slot 跨度越大，phase cancellation 越严重），而 canonical averaging 反而从 0.224 微涨到 0.257。

### 2.3 两个失败耦合

这是 paper 最 elegant 的论点之一：**位置和内容是耦合的**。
- 把 summary 放在 in-distribution 位置上，但内容是 phase-cancelled 的 averaging，仍然没用；
- 用 canonical averaging 保留内容，但放在 OOD 位置上，模型读不到，也没用。

Block-relative RoPE（Infinity-RoPE，https://arxiv.org/abs/2511.20649）是个例子：它把所有超出训练范围的 offset 都 cap 到 $\Delta t_{\text{train}}$。这解决了 OOD 问题，但 **Remark 1** 指出一个微妙的失败：所有老的 summary slot 都塌缩到同一个最小位置 $t_{\min}^v = \max(0, q - \Delta t_{\text{train}})$，多个 slot 变成位置上不可区分，attention 无法辨别谁是谁。这就是"Block-relative collapse"。

## 三、WorldTrace 的设计

### 3.1 Slot-rank virtual position

核心想法很简单：给每个 summary slot 一个**只依赖 slot rank、与绝对 horizon $N$ 无关**的虚拟位置：

$$t_s^v = q - (L_{\text{attn}} - 1 - s), \quad s = 0, \ldots, N_s - 1$$

变量：
- $L_{\text{attn}} = N_s + N_r$：本地 attention window 大小
- $s$：summary slot 索引，0 最老
- $N_s$：summary slot 数量
- $N_r$：recent verbatim window 大小

为什么这个设计满足四个性质：
1. **线性**：$t_s^v$ 是 $q$ 和 $s$ 的线性函数
2. **In-distribution**：$t_s^v \in [t_{\min}^v, t_{\max}^v]$，其中 $t_{\min}^v = \max(0, q - \Delta t_{\text{train}})$, $t_{\max}^v = q - N_r$，永远在训练见过的 offset 范围内
3. **Horizon-stable**：与 $N$ 无关，rollout 跑到 $N=10^6$ 也是同样公式
4. **Distinct**：每个 slot 有不同的 offset $(L_{\text{attn}}-1-s)$，互不塌缩

对比 Centroid-linear baseline（按 slot 内帧的平均时间戳线性映射到 in-distribution 范围）：它满足 (i)(ii) 但违反 (iii)，slot 位置会随 $N$ 增长而漂移出分布。Table 1 显示 Centroid-linear 在 $N=8$ TempSSIM=0.377，$N=16$ 涨到 0.479，但比 WorldTrace 的 0.413/0.545 都低。

### 3.2 Canonical key averaging (WorldTrace-Field)

在 slot-rank 虚拟位置的基础上，WorldTrace-Field 这样压缩：

$$K_{\text{field}}^f(t^v) = R(\theta_f t^v) \cdot \frac{1}{M}\sum_{m=1}^{M} R(-\theta_f t_m) K_{t_m}^f$$

步骤拆解：
1. **Unrotate**：$R(-\theta_f t_m) K_{t_m}^f$ —— 把每个源帧的 key 旋转回 canonical space（位置 $0$），去掉原始时间戳的 RoPE 旋转
2. **Average in canonical space**：在 unrotated 域内做平均，避免 phase cancellation
3. **Re-rotate**：$R(\theta_f t^v)$ —— 把 averaged key 旋转到目标 slot 的 virtual position $t^v$

**Proposition 2** 给出这个设计的数学保证——mean attention preservation：

$$\langle Q_q^f, K_{\text{field}}^f(t^v) \rangle = \frac{1}{M}\sum_{m=1}^{M} \langle Q_q^f, R(\theta_f t^v) K_m^f \rangle$$

也就是说，summary slot 在虚拟位置 $t^v$ 上产生的 attention score，正好等于把 $M$ 个源 key 都"假想地"重新放在 $t^v$ 上后，它们 attention score 的平均值。这是 pre-softmax score 上的精确保证，意味着我们不会因为压缩而偏向或远离 distant history。

注意，作者特别提到一个副作用：canonical averaging 会让 summary key 的 norm 缩小到约 $1/\sqrt{M}$（如果 source keys 互不相关）。他们**故意不**用 $\sqrt{M}$ 重新 rescale，因为更小的 logit 等价于 softmax 上对"被严重压缩的远古历史"施加更小的权重——这是个 implicit prior，反而合理。这让我联想到 LLM 里 attention sink 和 StreamingLLM（https://arxiv.org/abs/2309.17453）的发现——保留 sink token 是为了 attention 分布有地方可去；这里保留 summary 的低 logit 也是个"让模型少关注被压扁的远端"的设计选择。

### 3.3 Frozen landmark keys (WorldTrace-Landmark)

第二个 variant 解决 episodic recall：当你回到一个之前访问过的场景，模型应该能精确重建它。averaging 不行，因为 averaging 会 blur 掉具体细节。WorldTrace-Landmark 选择**verbatim 保留**特定的高价值帧——scene-entry 帧。

**Scene-entry detection**：用相邻帧 canonical key 的 cosine distance：

$$\text{dist}(K_m^f, K_{m-1}^f) = 1 - \cos(R(-\theta_f t_m)K_{t_m}^f, R(-\theta_f t_{m-1})K_{t_{m-1}}^f)$$

如果距离超过阈值 $\tau$，标记为 scene-entry event。

**Frozen canonical keys**：这里有个非常 subtle 的工程细节。每个 chunk 生成后，KV cache 整体往后 shift 一个 slot（老 landmark 从 slot $s$ 移到 slot $s-1$）。如果每次 shift 都做 unrotate-rerotate，bfloat16 精度会累积 floating-point drift（参考 Wang et al. https://arxiv.org/abs/2411.13476）。WorldTrace-Landmark 的解法：

$$K_{\text{land}}^f(t_s^v) = R(\theta_f t_s^v) \cdot R(-\theta_f t_{\ell^*}) K_{t_{\ell^*}}^f$$

- $t_{\ell^*}$：被选为 landmark 的原始帧的时间戳
- 关键：$R(-\theta_f t_{\ell^*}) K_{t_{\ell^*}}^f$（canonical key）**只在 landmark 时刻计算一次，永久 frozen**
- 每次 shift 只重新做一次 $R(\theta_f t_s^v)$ 旋转，避免反复 unrotate-rerotate 的精度损失

这是个非常工程师友好的设计——把可能产生 numerical drift 的链路缩短成单次操作。

### 3.4 Structured sparse attention 视角

我觉得 paper Appendix B 是最 insightful 的部分之一。作者把整个 memory compression 问题 reframe 成 **structured sparse attention approximation**。

对于 query $q$，标准 attention 是：

$$A(q,K,V) = \text{softmax}(qK^\top)V = \alpha_q V$$

其中 $\alpha_q \in \Delta_T$（$T$ 维概率 simplex）。压缩版引入投影矩阵 $P \in \mathbb{R}^{L_{\text{attn}} \times T}$：

$$\hat{A}(q,K,V;P) = \text{softmax}(q(PK)^\top) PV$$

**不同 cache 结构对应不同的 $P$ 矩阵**（见 Figure 6）：
- Sliding window：$P = [0 \;\; I_{L_{\text{attn}}}]$，one-hot 行向量选最近 $L_{\text{attn}}$ 帧
- + attention sink：第 0 行固定到初始帧，其余选最近 $L_{\text{attn}}-1$ 帧
- WorldTrace-Field：recent 行是 identity，summary 行是 uniform average over group
- WorldTrace-Landmark：summary 行是 one-hot 在 detected scene-entry 处

**Proposition 1（approximation bound）**：

$$\|r_q(P)\|_2 \leq \underbrace{\|\hat{\alpha}_q P - \alpha_q\|_1}_{\text{distribution mismatch}} \cdot \|V\|_{\infty, \text{row}}$$

这是 Hölder 不等式（$\ell_1/\ell_\infty$ duality）的直接应用。它告诉我们：**压缩误差的关键在于"真实 attention 分布 $\alpha_q$"和"压缩后能达到的分布 $\hat{\alpha}_q P$"之间的 $\ell_1$ 距离**。

更妙的是，**Appendix B.4 把这个 reduce 到 Nonnegative Matrix Factorization（NMF）**：

$$\min_{B \geq 0, P \geq 0} \|BP - A\|_1 \quad \text{subject to row-sum constraints on } B \text{ and } P$$

- $A \in \mathbb{R}^{|\mathcal{Q}| \times T}$：每个 query 的真实 attention 分布（每行一个）
- $B \in \mathbb{R}^{|\mathcal{Q}| \times L_{\text{attn}}}$：每个 query 的 prototype weights
- $P \in \mathbb{R}^{L_{\text{attn}} \times T}$：cache 投影矩阵

这个 reframe 的 beauty 在于：
- WorldTrace-Field 的 uniform-partition rows 对应 **k-means 的 $\ell_1$ 类比**（cluster-mean prototypes），用时间序分桶替代 learned assignment
- WorldTrace-Landmark 的 0/1 rows 对应 **k-medoids on attention distributions**，cosine-distance rule 是一个 $O(1)$ 单遍 surrogate

**Appendix B.5 给出何时 WorldTrace 接近 optimal**：

对"coherent queries"——old-history attention 取决于 group 而非具体帧（formalized 为 $\|\bar{\alpha}_q^{\text{old}} - \sum_i a_{q,i} u_i\|_1 \leq \varepsilon_q$）——WorldTrace-Field 满足：

$$\mathcal{I}(P_{\text{field}}) \leq \sum_{q \in \mathcal{Q}} \|\alpha_q^{\text{old}}\|_1 \varepsilon_q$$

对"recall queries"——old-history attention 集中在单帧（$\|\bar{\alpha}_q^{\text{old}} - e_{m(q)}\|_1 \leq \delta_q$）——WorldTrace-Landmark 满足：

$$\mathcal{I}(P_{\text{landmark}}) \leq \sum_{q \in \mathcal{Q}} \|\alpha_q^{\text{old}}\|_1 \delta_q$$

这两个 bound 是 conditional characterizations——它们告诉你"如果你的 queries 是 smooth/coherent 的，WorldTrace-Field 接近最优；如果是 concentrated/recall 的，WorldTrace-Landmark 接近最优"。这正好对应两种典型的 long-horizon 任务：smooth continuation（继续走同一条路）vs episodic recall（回到一个地标）。

## 四、LoopBench：评估 episodic recall

LoopBench 是这篇 paper 的另一个 contribution。它设计了一个**自评估**的 benchmark——rollout 自己产生 reference 和 prediction，不需要外部 ground-truth video。

四个 difficulty axes：
1. **Topology**：ABA（直接反转）、ABCA（L 形三角，对角 $5\sqrt{2} \approx 7$）、ABCDA（方形）
2. **Rollout length** $N \in \{8, 16, 32\}$：越长的 detour，cached offset 越出分布
3. **Camera orientation**：原地 pan 90°/180°/360°
4. **Multi-revisit depth** $R$：ABABA（重访 A 两次）、ABCBA、ABCDBA

Metric 是 **PAC (Position-Aligned CLIP)**：用 CLIP-ViT-H/14（https://arxiv.org/abs/2103.00020）计算几何对齐的 return-leg 和 forward-leg 帧的 cosine similarity。

Table 3 给了详细结果。让我摘几个关键数据：

- Tier 1（topology）：ABA 上 WorldTrace-Landmark PAC=0.864 vs sliding window 0.723（+19.5%）
- Tier 2（rollout length）：ABA $N=32$ 上 Landmark 0.825 vs sliding 0.627，长 horizon 优势放大
- Tier 4（multi-revisit）：ABABA $N=32$ 上 Landmark 0.941 vs sliding 0.892

Table 10 是个特别 informative 的 sweep：在 $N \in \{16, 32, 48, 64, 128, 256\}$ 上对比三个 tier：
- Compression only（sliding window / Naive / WorldTrace-Field）：PAC 在 0.4-0.6 区间徘徊
- Canonical-K anchoring（Latent re-anchor）：$N=128$ 还 0.782，$N=256$ 跌到 0.610
- Verbatim recall（WorldTrace-Landmark）：$N=256$ 还 **0.989**！

这告诉我们：**verbatim landmark 是唯一能 scale 到 minute-scale 的方案**。compression 总会损失，anchor 在长 horizon 也会崩。

## 五、消融与设计验证

Table 4 是个关键 ablation：在 ABA 上同时变 position assignment 和 compression method。

读这个表的方式：
- Naive averaging + Block-relative：PAC=0.570（$N=16$），但本质上 sliding window 加点 averaging
- Field averaging + Block-relative：0.540，**和 sliding window 一样**！因为 Block-relative 把所有老 slot 都塌缩到同一位置，averaging 起不到作用
- WorldTrace (Field)：0.555，略升

这个对比说明：**如果你只改 content 不改 position，等于没改**。反之如果你只改 position 不改 content，也只是 marginal improvement。两个必须同时改。

Table 5 验证 canonical vs naive averaging：在 Block-relative 固定 position 下，naive → canonical 让 $N_s=4$ 的 LatentDif 从 0.312 降到 0.233（-25.3%）。这就是 phase cancellation 的代价。

Table 9 对比 concurrent baselines：
- YaRN（https://arxiv.org/abs/2309.00071）做 NTK rescaling，$N=32$ PAC=0.490（比 sliding 0.401 好），但需要 $O(N)$ 内存，$N=100$ 就 OOM
- MemRoPE（https://arxiv.org/abs/2603.12513）用 dual-rate EMA，$N=32$ PAC=0.651
- WorldTrace-Landmark：0.964，碾压

## 六、几个延伸的直觉和联想

### 6.1 与 LLM KV cache compression 的对比

LLM 那边有大量 KV cache compression 工作（SnapKV、H2O、PyramidKV、StreamingLLM、DuoAttention，参见 https://arxiv.org/abs/2404.14469、https://arxiv.org/abs/2306.14048）。它们主要解决"哪些 token 该保留"——但**不解决"summary slot 该放在哪个位置"**，因为 LLM token 是 token-level 个体，保留就保留原位置，不存在"把多个 token merge 后该放在哪"的问题。

但 video world model 里，每个 latent frame 是一个"chunk"，需要 merge 多个 frame 进一个 slot。这个 merge 后的位置归属问题，是 video 独有的。WorldTrace 把这个 extra degree of freedom（virtual position）和 content writer 联合设计，是它的核心贡献。

### 6.2 与 Landmark Attention 的关系

Landmark Attention（https://arxiv.org/abs/2305.16300）是 Mohtashami & Jaggi 的工作，它的 idea 是训练 transformer 学会"哪些 token 是 landmark"，然后通过 landmark 来 gate off-cache block 的访问。WorldTrace-Landmark 借了这个 vocabulary，但有几个关键区别：
1. **Training-free**：不微调模型，纯推理时操作 cache
2. **Verbatim canonical-frame keys**：不是 trained representative，而是直接 frozen 原始 key
3. **Slot-rank virtual position**：每个 landmark 占一个 in-distribution 虚拟位置
4. **3D RoPE**：video 里位置是 (t, h, w) 三轴，比 1D 复杂

### 6.3 与 Memorizing Transformers / Compressive Transformers 的关系

这两篇经典 memory-augmented transformer 工作（https://arxiv.org/abs/2203.12145、https://arxiv.org/abs/1911.05507）也用"recent window + compressed memory"两层结构。但它们用 trained memory tokens 或 external kNN memory。WorldTrace 的 "addressable compressed memory" 是个新维度——它强调**位置可寻址性**，不只是"有 memory"，而是"memory 能被 attention 找到"。

### 6.4 Mean attention preservation 的深层意义

Proposition 2 那个 mean attention preservation 我觉得是个挺深的 idea。它说的是：**summary slot 在 softmax 之前能精确重现 source keys 在共享位置上的平均 score**。

这让我想到一个更 general 的设计原则：**memory compression 应该在 softmax 之前 preserve 分布**，而不是在 softmax 之后。因为 softmax 是个 non-linear operation，pre-softmax score 的 preservation 比起 post-softmax weight 的 preservation 更"干净"。

这也呼应了 Transformer 里 attention score 的几何——pre-softmax 是 dot product，linear，可加；post-softmax 是 normalization，non-linear。

### 6.5 关于 NMF framing 的直觉

把 cache compression 看成 NMF（Appendix B.4）这个 reframe 我觉得对未来工作有启发。如果 WorldTrace-Field 是 k-means 的 $\ell_1$ 版本，WorldTrace-Landmark 是 k-medoids，那中间还有一整套 clustering 算法可以套：
- **Online k-means**：partition 边界随 inference 自适应
- **Streaming coresets**：维护 small representative set
- **Soft-sparse rows**：interpolate between averaging 和 one-hot selection
- **Directly optimized P**：balance coherence 和 recall across all queries

这些都自然成为 future work 方向（作者在 H.2 也提到了）。

### 6.6 关于"分钟级" horizon 的 scaling

Table 10 显示 WorldTrace-Landmark 在 $N=256$ 还能保持 PAC=0.989。这相当于 $\sim 5$ 分钟的视频（每 chunk 3 latent frame，每 latent frame $\sim 0.5$s 解码），从"秒级"扩展到"分钟级"。这是个很 meaningful 的 scaling——之前 video world model 基本只能在训练 context 内 work，WorldTrace 让它在 constant memory 下 scale 到任意长。

### 6.7 Limitation 的诚实表达

作者在 H.1 很诚实地讲了限制：
- WorldTrace-Field 的 summary slot 平均 $T/N_s$ 帧，ratio 随 horizon 增长，组内细节必然 blur
- WorldTrace-Landmark 只能 recall $N_s$ 个 distinct scene，超出就 evict oldest
- 依赖 scene-entry detector，detector 失败就 recall 失败

这让我想到个延伸：能不能让 evicted landmark merge 进一个 "coarse residual slot"，让被淘汰的场景留下可恢复的 trace？这是个挺自然的 next step。

## 七、几个我觉得还可以 push 的方向

读这篇 paper 时我脑子里冒出几个想法：

1. **Geometry-aware canonical keys**：现在 canonical averaging 假设 source frames 是"同一场景的 views"。如果用 Plücker camera rays（参考 MosaicMem https://arxiv.org/abs/2603.17117、UCM https://arxiv.org/abs/2602.22960）做 warp，average 同一 scene content 的 frames，summary 会更 informative。LingBot-World 上 WorldTrace-Field 没显著提升（Table 11），可能正是 Plücker conditioning 已经提供了这个 signal。

2. **Learned scene-entry policy**：现在 scene-entry detection 是 hand-crafted cosine threshold。训练一个小 policy（conditioned on action discontinuities、pose changes、segmentation logits）来决定何时 commit landmark、何时 evict，应该能 push recall 更高。

3. **Revisit-supervised memory training**：作者提到 revisit task 提供 free training signal。如果 backbone frozen 但训练 cache policy（类似 RL with revisit score as reward），可能学到比 hand-crafted 更好的策略。这让我联想到 RLHF 的范式——把"recall 成功"作为 reward，policy 学会主动管理 cache。

4. **Hierarchical memory**：现在只有 2 层（recent + summary）。能否做 3 层或更多？类似 RAID disk 的 tiered storage——hot tier（verbatim recent）、warm tier（field averages）、cold tier（sparse landmarks）。每层用不同 writer，层间用不同 compression ratio。

5. **Cross-architecture transfer**：paper 在 MG2-1.3B 和 LingBot-World 14B 上都试了。但在 SSM-based world model（Mamba、SANA-WM https://arxiv.org/abs/2605.15178）上呢？SSM 不用 attention，addressability 概念不同，但 state compression 的本质问题类似。

## 八、总结性直觉

如果让我一句话概括 WorldTrace 的核心 insight：

> **在 RoPE-based attention 里，memory 必须同时是"可寻址的"（位置 in-distribution）和"信息保真的"（content 没 phase cancel）。WorldTrace 把这两个问题 decouple 成 slot-rank virtual position + canonical-space averaging/verbatim storage，联合设计，让 O(1) cache 在任意长 horizon 都能 retrieve。**

这个 insight 不仅仅对 video world model 有用。任何用 RoPE + KV cache + 超长 horizon 的 autoregressive model——包括未来的 LLM agents、robot simulators、game engines——都会遇到这个 addressability 问题。WorldTrace 给出了一个非常 clean 的 solution template。

---

参考链接：
- WorldTrace project page: https://research.nvidia.com/labs/sil/projects/WorldTrace/
- RoFormer (RoPE 原文): https://arxiv.org/abs/2104.09864
- YaRN (NTK-by-parts RoPE extension): https://arxiv.org/abs/2309.00071
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Matrix-Game 2.0: https://arxiv.org/abs/2508.13009
- Landmark Attention: https://arxiv.org/abs/2305.16300
- Memorizing Transformers: https://arxiv.org/abs/2203.12145
- Compressive Transformers: https://arxiv.org/abs/1911.05507
- StreamingLLM (attention sink): https://arxiv.org/abs/2309.17453
- BFloat16 RoPE drift (Wang et al.): https://arxiv.org/abs/2411.13476
- NMF (Lee & Seung): https://www.nature.com/articles/44565
- Plücker camera conditioning (LingBot): https://arxiv.org/abs/2601.20540
- MosaicMem: https://arxiv.org/abs/2603.17117
- UCM: https://arxiv.org/abs/2602.22960
- CLIP (Radford et al.): https://arxiv.org/abs/2103.00020
- GameNGen: https://arxiv.org/abs/2408.14837
- Oasis (Decart): https://oasis-model.github.io
- Mamba: https://arxiv.org/abs/2312.00752
