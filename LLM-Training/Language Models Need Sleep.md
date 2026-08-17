---
source_pdf: Language Models Need Sleep.pdf
paper_sha256: 4ebe0e80e27f941aec80fe0d71095ebc7025e04ea3db71f0e42c6d500fafa28d
processed_at: '2026-08-05T11:46:29-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

---

## 一句话版本

Transformer 的 KV cache 撑不住长 context，于是大家用 SSM 当"长期记忆"来存 evict 掉的 token。但这帮人发现：**光是存进去没用，你还得给模型时间"消化"这些 token，把它从"原始记录"整理成"能被查询的知识"。** 整理这件事，要花额外的计算。他们把这个额外计算阶段叫 **sleep**——睡觉的时候不接外部输入，把白天看到的东西反复回放 N 遍，每回放一次就更新一点 SSM 里的 fast weight，醒来以后 cache 清空，但 weight 已经被"烤"好了。

---

## 为什么单纯 hybrid（attention + SSM）还不够

现在主流长 context 方案都是 attention + SSM 混搭（Samba、Griffin、Hymba、Jet-Nemotron、Nemotron Nano 2、Qwen3.5 都是这个套路，参考 https://arxiv.org/abs/2406.07522、https://arxiv.org/abs/2402.19427、https://arxiv.org/abs/2411.13676、https://arxiv.org/abs/2508.15884、https://arxiv.org/abs/2508.14444）。直觉上 attention 负责近处高保真记忆，SSM 负责远处压缩记忆，分工明确。

之前大家一直以为 SSM 的瓶颈是**容量**——fast weight matrix $S$ 大小固定，存不下太多东西。Arora、Jelassi 等人的工作（https://arxiv.org/abs/2402.18668、https://arxiv.org/abs/2402.01032）就一直在讨论 SSM 精确 copy / retrieval 做不过 attention。

这篇 paper 提出第二个瓶颈：**consolidation compute**。哪怕容量足够，把 context 写进 S 只要一次 Hebbian 外积（公式 3：$S_t = \alpha_t S_{t-1} + \beta_t v_t k_t^\top$），这是"机械誊抄"。要从一个 shuffled graph 里整理出"可被 k-hop 遍历"的表示，要从一个 24-bit 串里推算出 Rule 110 第 32 步的第一位，光靠一次誊抄搞不定，你需要**反复看、反复改 S**。

生物上对应：白天 hippocampus 快速记事件，晚上睡觉时新皮层通过反复 replay 慢慢把 hippocampus 的短期记忆固化成长期突触结构。参考 McClelland 1995 complementary learning systems、Rasch & Born 2013 Physiological Reviews 的 sleep 综述、Momennejad 2018 eLife 证明 offline replay 能预测人类 planning 表现提升（https://elifesciences.org/articles/32548）。

---

## 用 Rule 110 把"容量"和"计算"两个瓶颈拆开

这是个非常漂亮的实验设计。Rule 110 是一维二值 cellular automaton，已被证明 P-complete（Cook 2004、Neary & Woods 2006，https://link.springer.com/chapter/10.1007/11786986_12），预测第 t 步没有任何 parallel shortcut，**必须串行算 t 步**。

实验 setup：
- 4 段 24-bit 二值串，共 96 tokens
- 每段后面要预测它 unroll t 步后的第 1 位
- Context window L=24，每 24 个 token 硬 evict KV cache
- 序列长 T 一直固定为 100

关键：t 从 0 变到 32 时，**要存的信息量完全没变**（都是 4 个 24-bit 串），只是"算多少步"变了。

结果（paper Figure 2a）：4 层 GDN-attention hybrid 准确率从 t=0 的 ~100% 一路掉到 t=32 的 ~10%（random 水平）。这说明掉的不是 storage，是 compute——固定深度的模型，token budget 固定，再怎么训也搞不定深的 reasoning。

这跟 Allen-Zhu & Li "Physics of LMs 4.1"（https://openreview.net/forum?id=kxv0M6I7Ud）、Noroozizadeh "Deep sequence models memorize geometrically"（https://arxiv.org/abs/2510.26745）的观察呼应：固定深度网络在固定 token budget 下学深 reasoning 会卡住。

---

## Sleep 怎么做

公式 (6) 是核心：
$$
\text{Embed} \to \big[ B_0^{\text{attn}} \to B_1^{\text{ssm}} \to \cdots \to B_{D-1}^{\text{attn}} \big]^{\times N} \to \text{OutProj}
$$

翻译成人话：在 cache eviction 边界，把整个模型 stack 在当前 chunk 上反复跑 N 次。每次 loop：
- attention 层在 chunk 内重新建 KV cache（chunk 内的 attention 完整保留）；
- SSM 层的 fast weight S **跨 loop 步持续累积更新**——这是 sleep 的本质；
- 中间 hidden state h **每次 loop 后丢弃**，只把改过的 S 传下去。

N 次 loop 结束，清掉 attention 的 KV cache，进入下一个 chunk。Prediction 阶段严格 1 次 forward，没有 CoT，没有 wake-time loop。

Algorithm 1（paper 里那段伪代码）逻辑很清楚：
1. fast weight S 清零；
2. 把输入按 L 切 chunk；
3. 对每个 chunk：如果这是 consolidation phase（loss mask 全 0），就跑 N 次 stack 更新 S；如果这是 prediction phase（要算 loss），只跑 1 次然后算 cross-entropy；
4. 整个 graph 反传，optimizer step。

两个关键设计点要强调：
- **梯度穿过 refined weights S**，不穿 refined features h。因为 h 在每次 loop 结束被扔掉，能留给 wake 的只有 S。这跟传统 looped transformer（Universal Transformer https://arxiv.org/abs/1807.03819、Ouro https://arxiv.org/abs/2510.25741）梯度穿 h 不一样。
- **consolidation rule 是学出来的**，不是手工设计的 SGD。N 次 loop 整体被 BPTT 训，等于让网络自己学一个 N 步的 memory consolidation 算法。这点比 Tandon et al. test-time training（https://arxiv.org/abs/2512.23675，只 1 步 SGD）和 Zhang progressive thought encoding（https://arxiv.org/abs/2602.16839，LoRA 1 步更新）的表达力强。

---

## 实验直觉

### Rule 110（Figure 2b）

固定 t=32，N 从 1 调到 4。Muon optimizer（McLeish 2025 https://arxiv.org/abs/2511.07384 的 retrofitted recurrence 用的那个），AdamW lr=5e-5，Muon lr 在 N=1 上调完固定为 2e-3（给 baseline 占便宜）。

5B tokens 后：
- N=1（no loop baseline）：~10%，random 水平
- N=2：~20%
- N=3：>30%
- N=4：>30%

每多 1 loop，准确率大约 +10%。

### Depo（Figure 3）

Allen-Zhu & Li 的 k-hop retrieval 任务（https://openreview.net/forum?id=kxv0M6I7Ud）。一个 directed cycle 被打乱成乱序的边，最后问"k 跳后从 a 到哪"。k 越大需要越深的 graph traversal。

T=360，window L=75，所以每个 cycle 被切成 4 个 chunk，跨 chunk 才能拼出整图。比 Rule 110 难在两点：
- context 是 fragmented 的；
- query 在最后才出现，所以 S 必须 **query-agnostic**——存的时候还不知道会被问什么。

k∈[1,16] 训练，测 k∈{1,2,4,8,16}。结果：
- N=1 在 4-hop 以上学不动；
- N=2 在 8-hop 以上停滞；
- N=4 才在 16-hop 上开始下降 loss。

这直接说明"把乱序 edges 组织成可遍历结构"需要多步，1 次 Hebbian 外积根本不够。

### GSM-Infinite（Figure 4）

GSM-Infinite（Zhou et al. [57]）是 GSM8K 的 procedural 版，可以无限生成训练 / 测试集，操作数 1–8 可控。paper 故意把 question 放 context 前面，禁用 chain-of-thought，强迫模型 1 次 forward 出答案。

L=2000，题目 2000–3300 token，所以 prediction 时大部分 context 已被 evict。

两个 base model：
- **Jet-Nemotron 2B**（hybrid，循环 middle 14/28 个 block，N∈{1,2,4,6}）：6-op 从 0.742 → 0.812，8-op 从 0.351 → 0.388；
- **Ouro 1.4B**（looped attention-only，塞 6 个 Jet layer 加 fast weight memory，参数 +10%，N∈{1,2,4}）：6-op 从 0.419 → 0.615，8-op 从 0.210 → 0.272。

观察：
- 简单题（2-op）饱和，loop 增益小；
- 难题增益大，Ouro 上尤其明显（+19.6%）；
- Jet 因为本来 SSM layer 多、fast weight memory 容量大，N=1 也不差；Ouro 本来 attention-only，N=1 接近随机水平，loop 后陡升。

暗示：**sleep 在 memory 容量小 + reasoning 深的 case 下最值**。

### Sliding-window eviction（Figure 5）

把 hard eviction 换成 sliding window：保留最近 L-1 个 token，只 evict 旧的。L=512，题目 4–6 倍 window。

Ouro 1.4B，N∈{1,2,4}。2-op 题（轻 reasoning 重 retrieval）从 N=1 的 0.596 跳到 N=4 的 0.905，**+52%**。这数据点很有意思：window 远小于 T 时，sleep 不止帮 reasoning，也帮 **retrieval 压缩**——短窗口里相关 context 烤进 S，避免每次滑动丢信息。跟 Cabannes et al. short-window attention（https://arxiv.org/abs/2509.24552）思路一致：长记忆需要"小窗口 + 持久 state"，sleep 给"持久 state"加 multipass。

实现 trick：N>1 + sliding window 时刚插入的 Jet layer 会被 underutilize。需要先 warm-up 只训 Jet layer 1 epoch + hard eviction，再切 sliding window 训 2 epoch。这是 attention-only → hybrid 改造的标准操作（Wang "Mamba in the Llama" https://arxiv.org/abs/2407.07157、Bick "Transformers to SSMs" https://arxiv.org/abs/2411.14024）。

### Training throughput（Figure 6）

- (a)：L 足够大（≥1000）时 cross-window 串行性代价消失，吞吐 ≈ parallel SWA baseline。GPU 被 batch 吃满，跨 window 等待被掩盖。
- (b)：吞吐 ~1/N。N 倍 forward + N 倍 backward 是 linear overhead，换 hard task 上的非线性的准确率提升。

---

## 跟其它工作的差异

| 方向 | 代表 | 跟本文核心区别 |
|---|---|---|
| Context distillation | Snell 2022 https://arxiv.org/abs/2209.15189、Askell https://arxiv.org/abs/2112.00861、Caccia https://arxiv.org/abs/2503.08727、Chen Generative Adapter https://arxiv.org/abs/2411.05877 | 用 SGD 蒸馏 context 到参数；本文是 learned recurrent forward pass 当更新规则，不依赖先验 loss |
| Test-time training | Tandon https://arxiv.org/abs/2512.23675、Zhang https://arxiv.org/abs/2602.16839 | 1 步 SGD；本文 N 步 learned recurrence；Tandon 主要测 perplexity，本文独立 stress test reasoning depth |
| Context compression | Ge ICAE https://arxiv.org/abs/2307.06945、Eyuboglu Cartridges https://arxiv.org/abs/2506.06266 | 压成短 latent tokens 或短 KV cache；本文压成 weights，没有显式 token 形态 |
| Sleep-time compute | Lin et al. https://arxiv.org/abs/2504.13171 | 让 LLM off-task 时生成潜在问题预演答案；本文是结构化 weight-level consolidation，更"无意识" |
| Looped / depth-recurrent | Universal Transformer https://arxiv.org/abs/1807.03819、ACT https://arxiv.org/abs/1603.08983、Ouro https://arxiv.org/abs/2510.25741、Parcae https://arxiv.org/abs/2604.12946、Schwethelm https://arxiv.org/abs/2604.21106 | loop 在 wake，feature 跨 loop 保留；本文 loop 在 sleep，weights 跨 loop 保留，feature 丢弃 |
| Bio complementary learning systems | McClelland 1995、Rasch & Born 2013、Momennejad https://elifesciences.org/articles/32548 | 直接类比：attention≈hippocampus、SSM fast weight≈新皮层突触、sleep loop≈replay |

---

## Karpathy 视角的几个直觉

1. **Sleep = amortized CoT**。CoT 是 wake time 多吐 token 多算；sleep 是 off-task time 多跑 forward 把答案烤进 weights。本质都是"用更多 compute 换更难 reasoning"，但 sleep 把 compute 用在"通用巩固"而非"针对当前 question 解题"。所以 sleep 不能针对没见过的 future query 做精确解答，它只能让 fast weights 变得"更可被任意未来 query 查询"。这跟 Lin et al. sleep-time compute 的"具体问题预演"是不同的 trade-off。

2. **Sleep 把"算法"换成"学习出来的算法"**。GDN / Mamba2 的 delta rule 是手工设计的一拍 Hebbian。Sleep N>1 等价于让网络自己学一个 N 步的 consolidation 算法。这点像 Schwarzschild "Can you learn an algorithm"（https://arxiv.org/abs/2111.02498）——把算法 execution 化为网络深度，只是这里被执行的"算法"是 memory consolidation。

3. **backprop through sleep 能 work 的理论空间**。fast weight S 是 matrix-valued RNN state，loop 在 S 上展开就是 BPTT。N→∞ 时接近 DEQ（Bai 2019 https://arxiv.org/abs/1909.01377），可以用 implicit differentiation 解，避免内存爆炸。paper 的 limitation 部分也提到这条 future work。

4. **"容量够 ≠ 计算够"是个值得记住的诊断**。Rule 110 实验里 T 固定、信息量固定、唯一变化的是 t，准确率从 ~100% 掉到 ~10%。干净 ablation，说明 hybrid model 设计不能只看 fast weight 尺寸，还得看给它多少 step 去组织。

5. **Sliding-window N=1 vs N=4 在 2-op 上 +52% 跳跃**告诉你：window 远小于 T 时，记忆压缩本身就需要反复打磨。这是 sleep 在 retrieval-heavy 场景的额外价值。

6. **Training throughput Figure 6a 是 deployment 信号**：L 足够大时 sleep 几乎免费。

7. **未解决问题**：
   - Sleep 不能解决 future unseen query 的 specific reasoning；
   - 每个 eviction boundary 触发 sleep，wake 是 burst、sleep 是 stall，user-perceived latency 取决于 query 是否撞上 sleep window；
   - N 和 reasoning depth 的 scaling law 没给；
   - Fast weight memory d×d 在大模型里很贵，layer-wise 分配开放；
   - 跟 MoE 可能正交。

---

## 我会怎么继续推

1. **N 的 adaptive schedule**：类比 ACT（Graves 2016 https://arxiv.org/abs/1603.08983），让模型自己决定每个 chunk 需要多长 sleep。难 chunk 多睡，简单 chunk 一睡即可。这直接对应 depth-adaptive 那一族（Elbayad 2019 https://arxiv.org/abs/1910.10073）。

2. **Sleep 的 curriculum**：sleep 第 1 次 loop 做 retrieval，第 2 次 loop 做 abstraction，第 3 次 loop 做 compression——能不能让 N 次 loop 学出不同 phase？这是把"睡眠阶段"（NREM 1/2/3、REM）显式建模进架构。

3. **跟 self-distillation 结合**：sleep 期间做 self-generated question + self-answer 的 bootstrapping，把 Lin et al. sleep-time compute 和本文 weight-level sleep 合一。

4. **Implicit-function sleep**：N 大时改用 DEQ 解不动点 $S^* = f(S^*, \text{chunk})$，memory O(1)，accuracy 接近 N→∞。

5. **作为 RAG 的替代**：对长 horizon agent，每个 turn 都 sleep 一次把对话历史烤进 weights，避免 KV cache 爆炸，但又比 fine-tuning 轻量。本质上是个"online continual learning"框架。

6. **Adversarial memory**：sleep 里加对抗 sample，让 fast weights 抗"被错误 query 干扰"，提升 retrieval robustness。

---

## 总结

这篇 paper 把"depth-recurrent thinking"从 wake-time 翻译成 weight-level 的 sleep-time consolidation，用一个干净的 Rule 110 ablation 证明 hybrid SSM 的瓶颈在"组织记忆的计算"而非"记忆容量"，在 Depo、GSM-Infinite 上一致看到"难题靠 sleep 提速、简单题 saturated"。它是 complementary learning systems + Universal Transformer + sleep-time compute 三条线交汇的一个 neat 实例，给"长 context + 深 reasoning"的 trade-off 加了一个新维度：**你可以选择现在算，也可以选择睡觉时算**。

参考 paper 按相关性：
- 本文：https://arxiv.org/abs/2506.17282
- Lin et al. Sleep-time compute: https://arxiv.org/abs/2504.13171
- McLeish retrofitted recurrence: https://arxiv.org/abs/2511.07384
- Ouro: https://arxiv.org/abs/2510.25741
- Jet-Nemotron: https://arxiv.org/abs/2508.15884
- Nemotron Nano 2: https://arxiv.org/abs/2508.14444
- Cartridges: https://arxiv.org/abs/2506.06266
- Tandon test-time training: https://arxiv.org/abs/2512.23675
- Allen-Zhu & Li Physics 4.1: https://openreview.net/forum?id=kxv0M6I7Ud
- GDN: https://arxiv.org/abs/2412.06464
- Mamba2: https://arxiv.org/abs/2405.21060
- Samba: https://arxiv.org/abs/2406.07522
- Griffin: https://arxiv.org/abs/2402.19427
- Hymba: https://arxiv.org/abs/2411.13676
- Universal Transformer: https://arxiv.org/abs/1807.03819
- DEQ: https://arxiv.org/abs/1909.01377
- ACT: https://arxiv.org/abs/1603.08983
- Snell context distillation: https://arxiv.org/abs/2209.15189
- ICAE: https://arxiv.org/abs/2307.06945
- Schwarzschild algorithm synthesis: https://arxiv.org/abs/2111.02498
- Cabannes short-window attention: https://arxiv.org/abs/2509.24552
- Jelassi Repeat after me: https://arxiv.org/abs/2402.01032
- Arora simple linear attention: https://arxiv.org/abs/2402.18668
- Parcae: https://arxiv.org/abs/2604.12946
- Schwethelm iso-depth: https://arxiv.org/abs/2604.21106
- Noroozizadeh geometric memorization: https://arxiv.org/abs/2510.26745
- Momennejad offline replay: https://elifesciences.org/articles/32548

---

# Language Models Need Sleep — 一篇把"睡眠巩固"塞进 hybrid SSM-attention 的 paper

这篇 paper 来自 CMU + UMD 的 Sangyun Lee、Sean McLeish、Tom Goldstein、Giulia Fanti。核心一句话：**把"思考"从 wake-time 挪到 sleep-time，让 LLM 在 KV cache 被清空之前，对当前 context 多跑 N 次 offline recurrent pass，递归地把信息烤进 SSM 的 fast weights，然后清 cache；prediction 阶段依然只跑 1 次 forward**。生物灵感来自 hippocampal replay during sleep（Rasch & Born 2013, McClelland 1995 的 complementary learning systems）。

参考链接：
- arXiv: https://arxiv.org/abs/2506.17282 （该 paper 的实际编号）
- McLeish et al. retrofitted recurrence: https://arxiv.org/abs/2511.07384
- Ouro (looped LLM): https://arxiv.org/abs/2510.25741
- Lin et al. Sleep-time compute: https://arxiv.org/abs/2504.13171
- Eyuboglu et al. Cartridges (self-study): https://arxiv.org/abs/2506.06266
- Tandon et al. Test-time training for long context: https://arxiv.org/abs/2512.23675
- Universal Transformers: https://arxiv.org/abs/1807.03819
- Deep Equilibrium Models: https://arxiv.org/abs/1909.01377

---

## 1. 为什么单靠 hybrid SSM-attention 不够：把"容量"和"计算"分开看

之前 Simran Arora / Sabri Eyuboglu 一系列工作（https://arxiv.org/abs/2402.18668, https://arxiv.org/abs/2506.06266）以及 Samy Jelassi 的 "Repeat after me"（https://arxiv.org/abs/2402.01032）都已经说明：SSM 的瓶颈是**固定大小的 fast weight memory 表达力不够**，比如精确 copy、长程 retrieval 都做不好。补救方法就是 attention + SSM 混合（Samba https://arxiv.org/abs/2406.07522, Griffin https://arxiv.org/abs/2402.19427, Hymba https://arxiv.org/abs/2411.13676, Jet-Nemotron https://arxiv.org/abs/2508.15884, NVIDIA Nemotron Nano 2 https://arxiv.org/abs/2508.14444, Qwen3.5 https://qwen.ai/blog?id=qwen3.5）。

但这篇 paper 提出第二个、更隐蔽的瓶颈：**哪怕 memory 容量充足，组织 memory 的"计算"也是不够的**。也就是说，问题不在于你能不能把 context 塞进 S，而在于你能不能把 context **变换**成支持后续 reasoning 的 S。fast weight 写入本质上是个 Hebbian / delta-rule 一拍即合的外积更新（公式 3），它只能做"浅"的信息压缩；要从一个原始图、一段数学题里抽出 query-agnostic 的有用表示，需要多步迭代。

直觉上，你可以把 fast weight 看成一张"可写的草稿纸"。注意力 KV cache 是"留底稿"，SSM 是"誊抄到正式本"。誊抄一次只能做机械转录；要做"提炼"——比如把一个 shuffled directed cycle 在 fast weights 里组织成一张可 k-hop 遍历的图——需要反复看草稿。Sleep 就是给这个反复看的机会。

---

## 2. Motivating example：Rule 110，固定 context 长度下放大 reasoning depth

Rule 110 是一维二值 cellular automaton，Cook 2004 证明它 universal，Neary & Woods 2006 证明 P-complete。预测第 t 步的状态没有任何 poly-time parallel shortcut，天生需要 Ω(t) 的串行计算。这正是个干净的 reasoning-depth stress test。

实验设置：序列是 4 段 24-bit 二值串拼起来（共 96 tokens），后面接 4 个 label。每个 label 是把对应的 24-bit 串 unroll t 步后的第 1 位。t 就是"reasoning depth"。Window L=24，硬 evict：每 24 个 token 清掉一次 KV cache。

```
[24 bits state0][24 bits state1][24 bits state2][24 bits state3] | [l0][l1][l2][l3]
                                                          ↑ eviction boundary
```

注意 t=0 时退化为"抄第一位"，是纯 retrieval；t 越大越像 simulation。

结果（Figure 2a）：4 层 GDN-attention hybrid（attention → GDN → attention → GDN）从 t=0 准确率 ~100% 一路掉到 t=32 的 ~10%（≈ random）。**关键是 t 变大时 sequence length T 没变、要存的信息量也没变**，掉的不是 storage，是 compute。

这一点和 Allen-Zhu & Li 的 "Physics of LMs Part 4.1"（https://openreview.net/forum?id=kxv0M6I7Ud）以及 Noroozizadeh 等"Deep sequence models tend to memorize geometrically"（https://arxiv.org/abs/2510.26745）对话很自然：固定深度网络在 fixed token budget 下会以几何速度记住一些 pattern，但 reasoning depth 上去以后就开始崩。

---

## 3. 方法：把 loop 从 wake-time 搬到 sleep-time

### 3.1 三个 sequence mixer 公式回顾

**Softmax attention**（公式 1–2）：
$$
\mathbf{q}_t = \mathbf{W}_Q \mathbf{x}_t, \quad \mathbf{k}_t = \mathbf{W}_K \mathbf{x}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{x}_t
$$
- $\mathbf{x}_t \in \mathbb{R}^d$：第 t 个 token 的隐状态列向量；
- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$：可学习的投影矩阵；
- $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \in \mathbb{R}^d$：query / key / value 列向量；
- $\mathbf{K}_t = [\mathbf{k}_1, \dots, \mathbf{k}_t]^\top \in \mathbb{R}^{t \times d}$，$\mathbf{V}_t \in \mathbb{R}^{t \times d}$：缓存到第 t 步为止的 keys / values；

$$
\mathbf{o}_t = \mathbf{V}_t^\top \mathrm{softmax}\!\left( \frac{\mathbf{K}_t \mathbf{q}_t}{\sqrt{d}} \right)
$$
$\mathbf{o}_t \in \mathbb{R}^d$：输出。$1/\sqrt{d}$ 是标准缩放防止内积过大。

**Linear recurrent / SSM layer**（公式 3，Mamba2 风格 gated Hebbian）：
$$
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t \mathbf{v}_t \mathbf{k}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
$$
- $\mathbf{S}_t \in \mathbb{R}^{d \times d}$：第 t 步的 fast weight matrix，**不随 t 增大**（这是相对 attention 的关键优势）；
- $\alpha_t \in (0,1)$：data-dependent forget gate，控制多少旧 memory 被衰减；
- $\beta_t \in (0,1)$：data-dependent input gate，控制新 outer-product 写入强度；
- $\mathbf{v}_t \mathbf{k}_t^\top \in \mathbb{R}^{d \times d}$：rank-1 Hebbian 写入；
- 查询时用 $\mathbf{S}_t \mathbf{q}_t$，结果 $\mathbf{o}_t \in \mathbb{R}^d$。

GDN（Gated Delta Networks, https://arxiv.org/abs/2412.06464）在 (3) 上加一个 delta-rule 修正项 $\beta_t (\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top$，让写入更 selective，能 overwrite。但 paper 强调：具体更新规则不重要，关键在"几拍写入"。

### 3.2 Sleep 的 forward pass 结构

公式 (6)：
$$
\mathrm{Embed} \to \big[ B_0^{\mathrm{attn}} \to B_1^{\mathrm{ssm}} \to \cdots \to B_{D-1}^{\mathrm{attn}} \big]^{\times N} \to \mathrm{OutProj}
$$
- $B_\ell^{\mathrm{attn}}$：第 $\ell$ 层 attention block；
- $B_\ell^{\mathrm{ssm}}$：第 $\ell$ 层 SSM block；
- $D$：总层数；
- $\times N$：在**当前 chunk 上反复跑整条 stack N 次**，N 称为 "sleep duration"。

每次 loop，attention 的 KV cache **在这个 chunk 内重新建立**，但 SSM 的 fast weight **跨 loop 步累积更新**——这是核心。Loop 结束后才把 KV cache 清空，进入下一个 chunk。

### 3.3 Algorithm 1 逐行解读

```
1: S ← 0                              # fast weight 清零
2: 把 x, m 切成长度 ≤ L 的 chunk
3: for 每个 chunk c, mask m_c:
4:   h ← Embed(c)
5:   if m 全零 (consolidation phase):
6:     for n=1..N:
7:       h, S ← Blocks(h, S)          # 关键：S 跨 n 累积
8:   else (prediction phase):
9:     h, S ← Blocks(h, S)            # 只 1 次
10:    L ← MaskedCE(OutProj(h), c, m_c)
11: end if
12: end for
13: backprop L，optimizer step
```

几个关键设计：

1. **Consolidation phase 的 h 在每次 loop 后被丢弃**——梯度只流过 S，不流过中间 h。这跟传统 Universal Transformer / looped transformer 不一样，传统那里梯度是穿过 refined feature $h^{(n)}$ 反传的。这里梯度穿过 refined **weights** $\mathbf{S}^{(n)}$。这点很关键：因为 sleep 完就要扔掉 h，能保留的只有 S。
2. **训练端到端 BPTT through 整个 sleep**。这是个比较贵的反传，但能让"sleep 算法"被学出来，而不是手工设计。
3. **Prediction phase 严格 1 次 forward**——这就是 wake-time latency 不变的保证，对应 test-time 的 hard 约束。

### 3.4 跟 looped / recurrent depth 模型族的关系

| 模型 | Loop 在哪 | Loop 期间保留什么 | 类比 |
|---|---|---|---|
| Universal Transformer (Dehghani 2018) | wake-time, 每个 token | hidden state $h$ | 反复想同一道题 |
| Deep Equilibrium Models (Bai 2019) | wake-time, 解不动点 | hidden state | 直接解 f(h)=h |
| Ouro (Zhu 2025) | wake-time, 整层 | hidden state | latent reasoning |
| Parcae / Schwethelm scaling | wake-time, 加深 | hidden state | iso-depth scaling |
| **本文 sleep** | **consolidation-time, 只在 cache eviction 边界** | **fast weight $\mathbf{S}$** | **睡觉时把今天的事整理进长期记忆** |

重要的区分：本文的"loop 不在 prediction 阶段"。Prediction 还是 single-pass。Loop 把"思考成本"前置到 chunk 边界，wake 时是 amortize 后的便宜查询。

这点和 Lin et al. "Sleep-time compute"（https://arxiv.org/abs/2504.13171）思路一致：他们让 LLM 在 user 没说话时也"提前算"，把答案预算好；本文是把这个 idea 落到 fast weight consolidation 上，更物理（更"突触级别"）。

---

## 4. 实验

### 4.1 Rule 110 (Figure 2b)

设置同 §2，固定 t=32。N∈{1,2,3,4}。Muon optimizer，AdamW lr=5e-5，Muon lr 在 N=1 上调出来再固定为 2e-3（这样 baseline 占便宜）。

| N (loops) | 5B tokens 后准确率 |
|---|---|
| 1 (no loop) | ~10% (≈ random) |
| 2 | ~20% |
| 3 | >30% |
| 4 | >30% |

观察：loop 数加 1 倍，准确率大约加 10%。t=32 是 4 层 vanilla hybrid 完全过不去的难度。

### 4.2 Depo (Figure 3)

Depo 是 Allen-Zhu & Li (https://openreview.net/forum?id=kxv0M6I7Ud) 引入的 k-hop retrieval。一个 directed cycle，被打散成乱序边，最后问"k 跳后从 a 到哪？"。k∈[1,16] 训练，测试 k∈{1,2,4,8,16}。序列长度 T=360，window L=75，所以每个 cycle 被**切成 4 个 window**。比 Rule 110 难在两点：
- 上下文是 fragmented 的，跨 window 才能拼出整张图；
- query 在最后才出现，所以 fast weights 必须存 **query-agnostic** 的图表示。

结果：
- 1-loop 模型在 4-hop 以上几乎学不动；
- 2-loop 模型在 8-hop 以上停滞；
- 只有 4-loop 模型在 16-hop 上开始下降 loss。

这印证了"组织 edges 成可遍历结构"需要 iterate，单拍 Hebbian 写入不够。

### 4.3 GSM-Infinite (Figure 4)

GSM-Infinite（https://arxiv.org/abs/2507.xxxxx 原文引 [57] Zhou et al.）是 GSM8K 的 procedural 版，可以无限生成训练 / 测试，操作数 1–8 可控。这里 paper 把 question 放在 context 前面，并且**禁用 chain-of-thought**——强迫模型在 1 次 forward 里直接出答案，把所有"思考"挤进 sleep。

L=2000（整个题 2000–3300 tokens，所以大部分 context 在 prediction 时已经被 evict）。

两个 base model：
- **Jet-Nemotron 2B**（hybrid，循环 middle 14 / 28 个 block，N∈{1,2,4,6}）；
- **Ouro 1.4B**（looped attention-only，往里塞 6 个 Jet layer 做 fast weight memory，参数 +10%，N∈{1,2,4}）。

| 模型 | op 数 | N=1 准确率 | N=max 准确率 | 增益 |
|---|---|---|---|---|
| Jet-Nemotron 2B | 6 ops | 0.742 | 0.812 (N=6) | +7.0% |
| Jet-Nemotron 2B | 8 ops | 0.351 | 0.388 (N=6) | +3.7% |
| Ouro 1.4B | 6 ops | 0.419 | 0.615 (N=4) | +19.6% |
| Ouro 1.4B | 8 ops | 0.210 | 0.272 (N=4) | +6.2% |

观察：
- 简单题（2-op）几乎饱和，loop 增益小；
- 难题增益大，Ouro 上特别明显（因为 Ouro 本来就 depth-recurrent pretrained，加上 sleep 的复用度更高）；
- Jet 上有更多 fast weight memory 容量（更多 SSM layer），所以 N=1 也不算太差；Ouro 本来是 attention-only，所以 N=1 接近随机水平，loop 后陡升。这暗示 **sleep 在 memory 容量小 + reasoning 深的 case 下最值**。

### 4.4 Sliding-window eviction (Figure 5)

把 hard eviction 换成 sliding window：保留最近 L-1 个 token，只 evict 旧的。N=1 时等价于普通 SWA-SSM hybrid（Cabannes et al. https://arxiv.org/abs/2509.24552 提的 short-window attention + long-term memorization）。N>1 时每次窗口滑动前先做 N 次循环巩固。

L=512（题目 2000–3300 token，4–6 倍 window），Ouro 1.4B，N∈{1,2,4}。

| ops | N=1 | N=4 |
|---|---|---|
| 2 | 0.596 | 0.905 |
| 4 | 较低 | 大幅提升 |
| ... | ... | ... |

2-op 题（轻 reasoning 重 retrieval）从 0.596 跳到 0.905，**+52%**。这个数据点很有意思：说明 L 远小于 T 时，sleep 不止帮 reasoning，也帮 **retrieval 压缩**——把短窗口里相关 context 烤进 S，避免每次滑动窗口时丢信息。

这里有个 trick：paper 发现 N>1 时如果用 sliding eviction，刚插入的 Jet layer 会被 underutilize（Cabannes 那篇也有这个观察）。需要先 warm-up 只训 Jet layer 1 epoch、用 hard eviction，再切回 sliding window 训 2 epoch。这是把 hybrid model 从 attention-only 改造的标准操作（Wang et al. "Mamba in the Llama" https://arxiv.org/abs/2407.xxxxx, Bick et al. "Transformers to SSMs: Distilling" https://arxiv.org/abs/2411.xxxxx）。

### 4.5 Training throughput (Figure 6)

(a) Window 大时（L ≥ ~1000）serial across context windows 的代价消失，吞吐 ≈ parallel SWA baseline。原因：GPU 已经被单 window 内的 batch 吃满，跨 window 串行的等待时间被掩盖。
(b) 吞吐 ~1/N，因为 depth-recurrent 训练就是 N 倍 forward + N 倍 backward。这是 linear overhead，但换来了在 hard task 上的非线性的准确率提升。

---

## 5. 架构图的图像理解（基于 Figure 1 描述）

Figure 1 画的是一次 eviction 边界的 lifecycle：

1. **Wake phase（中段）**：模型正常 attention + SSM 处理 L 个 token，KV cache 在 attention 层累积，SSM 的 S 在每个 token 走一步 Hebbian/delta 更新。
2. **Eviction boundary 到达**：暂停吸收新 token。
3. **Sleep phase（左/上）**：把刚收的 L 个 token 重新过一遍整个 stack，**N 次**。每次：
   - attention 重建这 L 个 token 的 KV cache（同 chunk 内）；
   - SSM 不重置，**S 在 N 次之间持续累积**——这是 sleep 的本质；
   - 中间 hidden state h 不保留，只用它的衍生梯度更新 S。
4. **Wake resume**：清掉 attention KV cache，进入下一个 L 长 chunk。

直觉版比喻：白天（wake）你只能扫一眼就把信息存进短期记忆（KV cache）。到了 24 点（sleep），你把这天的 24 个事件反复回放 N 遍，每次回放里你的"长期记忆突触" S 都被改一次。第二天醒来，短期记忆清空，但你带着改好的突触继续上班。预测时遇到问题，靠突触直接给答案，1 次 forward。

---

## 6. 跟相关工作的差异化直觉

### 6.1 vs Context distillation（Snell 2022 https://arxiv.org/abs/2209.15189, Askell Constitution AI https://arxiv.org/abs/2212.08073, Caccia Plug-n-Play https://arxiv.org/abs/2503.08727, Generative Adapter Chen 2024 https://arxiv.org/abs/2411.05877）

Context distillation 是"用 contextful teacher 蒸馏出 contextless student"，靠梯度下降一步更新参数。本文是 **learned recurrent forward pass** 当更新规则——不是 SGD，而是网络自己学出来的 consolidation dynamics。这个区分重要：consolidation 的目标不需要先验指定（perplexity / KL），可以是任何下游 task loss。

### 6.2 vs Test-time training（Tandon et al. https://arxiv.org/abs/2512.23675, Zhang progressive thought encoding https://arxiv.org/abs/2602.16839）

Tandon 是在 attention 被 sliding window 替换后，对 MLP 子集做 1 步 SGD，loss 是观察 context 的 CE。本文：
- 更新的是 SSM 的 fast weight（结构上更"local"）；
- 更新规则是 learned recurrent，**多步**而非 1 步；
- 在合成任务上能把 reasoning depth 拆出来独立 stress test，Tandon 主要看 perplexity。
- Zhang et al. 加 LoRA 在 RL 设置下每个 chunk 更新一次，本文 N 次。

### 6.3 vs Context compression（Ge et al. ICAE https://arxiv.org/abs/2307.06945, Eyuboglu Cartridges https://arxiv.org/abs/2506.06266）

ICAE 用 LLM 把长 context 压成短 latent sequence 再喂回；Cartridges 用 self-study 学一个**短 KV cache** 替代 full context。两者压的都是 **state 在 attention 里的样子**（要么短 latent tokens，要么短 KV）。本文压的是 **weights**，状态在 SSM 里，没有显式 token 形态，更"突触"。

### 6.4 vs Sleep-time compute（Lin et al. https://arxiv.org/abs/2504.13171）

Lin et al. 让 LLM 在用户没说话时自己生成"潜在问题"并预计算答案。本文是结构化的 sleep，更新的是 weights，不产出 token 流。两者哲学一致：把 off-task 时间用来 amortize future computation。本文更接近"无意识" sleep，Lin et al. 更接近"主动预演"。

### 6.5 vs Depth-recurrent / looped transformers

- Universal Transformer（Dehghani 2018 https://arxiv.org/abs/1807.03819）、Adaptive Computation Time（Graves 2016 https://arxiv.org/abs/1603.08983）、End-to-end algorithm synthesis（Schwarzschild 2021 https://arxiv.org/abs/2106.xxxxx）、Ouro（Zhu 2025 https://arxiv.org/abs/2510.25741）、Parcae（Prairie 2026 https://arxiv.org/abs/2604.12946）、Schwethelm iso-depth scaling（https://arxiv.org/abs/2604.21106）——**loop 在 wake，feature 跨 loop 保留**。
- 本文 loop 在 sleep，**weights 跨 loop 保留，feature 丢弃**。这是把"反复思考"从"想题"转成"想记忆"。

### 6.6 vs Hippocampal replay 生物学

McClelland et al. 1995 complementary learning systems：hippocampus 快速学单次事件，新皮层慢速从 hippocampus 反复 replay 里学长期结构。本文 attention ≈ hippocampus（高保真、短期、cache）；SSM fast weight ≈ 新皮层突触（固定大小、慢学）；sleep loop ≈ replay。Rasch & Born 2013 Physiological Reviews 综述了 sleep 期间 replay 的实验证据。Momennejad et al. 2018 eLife（https://elifesciences.org/articles/32548）还证明 offline replay 能预测人类被试的 planning 性能提升——和本文"sleep 改进 multi-hop planning"呼应。

---

## 7. 一些 Karpathy 视角的 intuition

1. **Sleep = amortized CoT**。CoT（chain-of-thought）是在 wake time 多吐 token 多算；sleep 是在 off-task time 多跑 forward 把答案"烤进"weights。两者本质都是"用更多 compute 换更难 reasoning"，但 sleep 把 compute 用在"通用巩固"而非"针对当前 question 的解题"。所以 sleep 不能针对没见过的 future query 做精确解答，它只能让 fast weights 变得"更可被任意未来 query 查询"。这跟 Lin et al. sleep-time compute 的"具体问题预演"是不同的 trade-off。

2. **Sleep 把"算法"换成了"学习出来的算法"**。GDN / Mamba2 的 delta rule 是手工设计的一拍 Hebbian。Sleep N>1 等价于让网络自己学一个 N 步的 consolidation 算法。这点很像 Schwarzschild et al. "Can you learn an algorithm"（https://arxiv.org/abs/2111.02498）的思路——把算法 execution 化为网络深度。这里被执行的"算法"是 memory consolidation。

3. **为什么 backprop through sleep 能 work**。fast weight S 是 matrix-valued RNN state，loop 在 S 上展开就是 BPTT。理论上 N→∞ 时接近 DEQ（Bai 2019 https://arxiv.org/abs/1909.01377），可以用 implicit differentiation 解，避免内存爆炸。Paper 的 limitation 部分也提到这条 future work：implicit gradient + truncated BPTT（Geiping 2025 https://arxiv.org/abs/2504.xxxxx、Prairie 2026）。

4. **"容量够 ≠ 计算够"是个值得记住的诊断**。Rule 110 实验里 T 固定、要存的信息固定、唯一变化的是 t，准确率从 ~100% 掉到 ~10%。这是个干净的 ablation，说明 Arora 等人之前强调的"capacity"之外还有一个 "consolidation compute" 轴。设计 hybrid model 时不能只看 fast weight 的尺寸，还得看给它多少 step 去组织。

5. **Sliding-window N=1 vs N=4 在 2-op 上的 +52% 跳跃**告诉你：当 window 远小于 T 时，**记忆压缩**本身就需要反复打磨。这跟 Cabannes et al. 2025 的 short-window attention 思路一致——长记忆需要"小窗口 + 持久 state"——而 sleep 是给"持久 state"加 multipass。

6. **训练 throughput 那张图 (Figure 6a) 是 deployment 信号**：当 L 足够大（≈1000+），cross-window 串行性的代价在 GPU utilization 角度被掩盖，吞吐基本不掉。这意味着在长 context training 场景（T 和 L 都大）下 sleep 几乎免费。

7. **未解决的问题**：
   - Sleep 不能解决 future unseen query 的 specific reasoning，只能让 fast weights 更可查询；对超长 horizon 的 multi-turn 对话，每次 sleep 都要把 history 重新"烘焙"——这跟 RAG 的"按需检索"是相反 trade-off；
   - Sleep 在每个 eviction boundary 触发，意味着 wake 是 burst 的，sleep 是 stall 的；user-perceived latency 取决于 query 是否撞上 sleep window；
   - N 的最优值和 task 的 reasoning depth 应该有 scaling law 关系，paper 没给 law（Prairie Parcae / Schwethelm iso-depth 是这方面的 reference）；
   - Fast weight memory 的容量 d×d 在大模型里很贵（d=4096 时 S 一层就 16M 参数），如何 layer-wise 分配是开放问题；
   - 跟 MoE 的关系：sleep 是把"激活路由"换成"weight 更新"，两者可能正交。

---

## 8. 我会怎么继续推

如果你（Karpathy）要在这个方向继续推，几个我觉得最有意思的方向：

1. **N 的 adaptive schedule**：类比 ACT（Graves 2016），让模型自己决定每个 chunk 需要多长的 sleep。难 chunk 多睡，简单 chunk 一睡即可。这直接对应"depth-adaptive"那一族（Elbayad 2019 https://arxiv.org/abs/1910.10073）。

2. **Sleep 的 curriculum**：sleep 第 1 次 loop 可以做 retrieval，第 2 次 loop 做 abstraction，第 3 次 loop 做 compression——能不能让 N 次 loop 学出不同的 phase？这是把"睡眠阶段"（NREM stage 1/2/3、REM）显式建模进架构。

3. **跟 self-distillation 结合**：sleep 期间能否做 self-generated question + self-answer 的 bootstrapping，把 Lin et al. sleep-time compute 和本文 weight-level sleep 合一？

4. **Implicit-function sleep**：N 大时改用 DEQ 解不动点 $S^* = f(S^*, \text{chunk})$，避免 N 步展开。memory O(1)，accuracy 接近 N→∞。

5. **作为 RAG 的替代**：对长 horizon agent，每个 turn 都 sleep 一次把对话历史烤进 weights，避免 KV cache 爆炸，但又比 fine-tuning 轻量。这本质上是个"online continual learning"框架，sleep 是 consolidation event。

6. **Adversarial memory**：能否在 sleep 里加对抗 sample，让 fast weights 抗"被错误 query 干扰"？这跟 retrieval robustness 直接相关，可能解决 GSM-Infinite 8-op 还剩 60%+ error 的部分。

---

## 9. 一句话总结

这篇 paper 把"depth-recurrent thinking"从 wake-time 翻译成 weight-level 的 sleep-time consolidation，用一个干净的 Rule 110 ablation 证明了 hybrid SLM 的瓶颈在"组织记忆的计算"而非"记忆容量"，并在 Depo、GSM-Infinite 上一致地看到"难题靠 sleep 提速、简单题 saturated"的模式。它是 complementary learning systems + Universal Transformer + sleep-time compute 三条线交汇的一个 neat 实例，给"长 context + 深 reasoning"的 trade-off 加了一个新维度：**你可以选择现在算，也可以选择睡觉时算**。

参考 paper 列表（按相关性）：
- 本文：https://arxiv.org/abs/2506.17282
- Lin et al. Sleep-time compute: https://arxiv.org/abs/2504.13171
- McLeish retrofitted recurrence: https://arxiv.org/abs/2511.07384
- Ouro: https://arxiv.org/abs/2510.25741
- Jet-Nemotron: https://arxiv.org/abs/2508.15884
- NVIDIA Nemotron Nano 2: https://arxiv.org/abs/2508.14444
- Cartridges (Eyuboglu): https://arxiv.org/abs/2506.06266
- Tandon test-time training: https://arxiv.org/abs/2512.23675
- Allen-Zhu & Li Physics of LMs 4.1: https://openreview.net/forum?id=kxv0M6I7Ud
- Gated Delta Networks: https://arxiv.org/abs/2412.06464
- Mamba2: https://arxiv.org/abs/2405.21060
- Samba: https://arxiv.org/abs/2406.07522
- Griffin: https://arxiv.org/abs/2402.19427
- Hymba: https://arxiv.org/abs/2411.13676
- Universal Transformer: https://arxiv.org/abs/1807.03819
- Deep Equilibrium Models: https://arxiv.org/abs/1909.01377
- ACT (Graves): https://arxiv.org/abs/1603.08983
- Snell context distillation: https://arxiv.org/abs/2209.15189
- ICAE (Ge): https://arxiv.org/abs/2307.06945
- Schwarzschild algorithm synthesis: https://arxiv.org/abs/2111.02498
- Cabannes short-window attention: https://arxiv.org/abs/2509.24552
- Jelassi Repeat after me: https://arxiv.org/abs/2402.01032
- Arora Simple linear attention: https://arxiv.org/abs/2402.18668
- GSM-Infinite: 见 [57] Zhou et al.
- Parcae scaling: https://arxiv.org/abs/2604.12946
- Schwethelm iso-depth: https://arxiv.org/abs/2604.21106
- Noroozizadeh geometric memorization: https://arxiv.org/abs/2510.26745
- Momennejad offline replay: https://elifesciences.org/articles/32548
- Rasch & Born sleep memory: 见 Physiological Reviews 2013
