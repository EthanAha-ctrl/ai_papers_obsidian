---
source_pdf: STREAMINGVLM REAL-TIME UNDERSTANDING.pdf
paper_sha256: 9446724a703fcea50b93abc5efb1c698b16d6585aa417d7c5c55d8a60c5c97d7
processed_at: '2026-08-12T11:19:30-07:00'
target_folder: LLM-engine/vllm
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 StreamingVLM

## 先说核心痛点

想象你在看一场 2 小时的世界杯球赛，每秒都要给出解说。对 VLM 来说，这事听起来简单，实际上几乎做不到。

为什么做不到? 三个坑:

**坑一**: Full attention。每个新 frame 要看所有历史 tokens。视频 2 小时 = 7200 秒 = 大概百万级 tokens。Attention 是 $O(T^2)$, 算力炸了。更糟的是 Qwen2.5-VL 训练时只见过有限长度 context, 超过训练长度后 RoPE index 外推, 模型直接输出 "shot shot shot shot shot..." (Figure 2 的真实案例)。

**坑二**: Sliding window 不 overlap。切成 chunk 各算各的。问题在于 chunk 边界处 context 突然 reset。你想说 "Ronaldo 这场比赛已经进了 3 个球", 但 chunk 开头时模型不知道之前发生了啥。Commentary 断断续续, 没有连贯性。

**坑三**: Sliding window with overlap。保留 recent tokens, 每个新 window 重新算 attention 到所有历史。Coherence 保住了, 但每秒都在 redundant recompute, latency 极高, 根本做不到 real-time。

## StreamingVLM 的"偷懒"思路

核心 idea 来自 StreamingLLM (arXiv:2309.17453, 同一作者 Guangxuan Xiao): **大部分历史其实可以忘掉, 只要保留三个东西**。

$$\text{KV Cache} = \underbrace{T_{\text{sink}}}_{\text{开头的固定 tokens}} + \underbrace{T_{\text{window}}}_{\text{最近的 text}} + \underbrace{V_{\text{window}}}_{\text{最近的 vision}}$$

具体值:
- $T_{\text{sink}} = 512$ tokens — system prompt 加之前 text 的开头
- $T_{\text{window}} = 512$ tokens — 最近的文本输出
- $V_{\text{window}} = 16s$ — 最近的视觉 tokens (大约 16 秒画面)

## 为什么 vision 短 text 长?这是 build intuition 的关键

**Vision tokens 时效性很强**: 16 秒足够覆盖一次进攻、一个罚球、一次 fast break。更早的画面信息密度低, 而且关键信息已经被 text commentary "压缩"成语言了。比如 5 分钟前那次进攻, 你需要的只是 "Ronaldo scored a goal" 这句话, 不需要原始像素。

**Text tokens 承载长期记忆**: "Ronaldo 这场进了 3 个球" 这句话随时可能被引用。过早 evict 会丢长期 context。所以 text window 比 vision window 长得多。

**Sink tokens 稳定 attention**: 来自 StreamingLLM 的发现 — Softmax 在 evicted positions 上会塌陷, 保留几个早期 tokens 当 "attention sink" 能稳住 attention 分布。

这种 asymmetric retention 是 paper 最 elegant 的设计, 背后直觉类似人类 working memory: 视觉短时记忆 (sensory memory 几秒到十几秒), 语言能压缩成更长期的 episodic memory。

## Contiguous RoPE — 看似无聊其实最关键

RoPE (Rotary Position Embedding) 给每个 token 一个位置 index。问题来了: 视频 2 小时 = 数百万 tokens, index 会涨到几百万。训练时最多见过几千的 index, 推理时几百万纯属 OOD, 性能崩塌。

**Contiguous RoPE 的做法**: evict 早期 tokens 后, 把剩下 tokens 的 index 左移, 让 index 永远保持在 $[0, T_{\text{sink}} + T_{\text{window}} + V_{\text{window}}]$ 这个 bounded range 内。

数学上, RoPE attention score 只依赖 **relative position** $m - n$:

$$\text{score}(q_m, k_n) \propto \cos((m-n)\theta)$$

只要 relative distance 保持 in-distribution, absolute index 可以随便 shift。Contiguous RoPE 就是利用这个性质, 让 absolute index 永远在训练见过的范围内。

Table 4 的 ablation 数据很 striking:

| RoPE | vs GPT-4o† | vs LiveCC† | vs LiveCC∞ |
|------|-----------|-----------|-----------|
| Native, infinite | 25.09 | 59.42 | 60.32 |
| Native, chunk 100s | 63.23 | 74.00 | 98.07 |
| Contiguous, infinite | **66.18** | **87.81** | **99.12** |

Native + infinite 几乎崩盘 (25.09)。Chunk 100s 能恢复一些 (因为每 100s reset 一次 index), 但丢了 long-term memory。Contiguous + infinite 完美: 既无限, 又稳。

Qwen2.5-VL 还有个额外复杂度 — vision tokens 用 3D RoPE (time, height, width)。Contiguous 3D RoPE 在三个维度上都做 left-shift, 匹配 interleaved vision-text layout。

## 训练策略 — Overlapped Chunk Full Attention

这是 paper 最 novel 的部分。问题: 你不能在 2 小时视频上训练 (quadratic cost 太贵), 但又希望模型推理时能稳定处理无限流。怎么训?

**Naive 方案**: 直接用 short clips 训练, 然后推理时用 sliding window。问题: training 和 inference attention pattern 完全不一样, 模型懵逼。

**StreamingVLM 方案**: 把长视频切成 consecutive chunks $\{C_1, C_2, \ldots\}$, 每个 chunk $W = 24s$, 相邻 chunk overlap $O = 12s$。每个 chunk 内 **full attention**, chunk 之间独立。

为什么 overlap 这么重要? Figure 4 的关键洞察: overlapped full attention 的 **effective attention pattern** 几乎完美匹配推理时的 sink + recent windows pattern。因为 chunk $C_i$ 的后 12s 在 $C_{i+1}$ 中作为前 12s 出现, 训练时模型见过 "前 12s 作 history + 后 12s 生成" 的 pattern, 正好对应推理时 "recent window + incoming frames"。

## V/T Interleaving — 1秒粒度的对齐

传统 VLM 把所有 vision tokens 放前面, text 放后面。StreamingVLM 在 chunk 内 **每秒 interleave 一次 vision 和 text**:

```
[1s vision][1s text][1s vision][1s text]...
```

这一秒没有解说? 插 placeholder "..." token, 但 layout 保持。Loss 只算 text positions (有 narration 的秒)。

这教会模型一件关键的事: **什么时候该说话, 什么时候闭嘴**。实时 commentary 不是每秒都讲, 球在过渡时解说员会停顿几秒。模型必须学会这个节奏。

这个设计的副作用 — Table 3 显示, 完全没有 VQA-specific SFT, StreamingVLM 在 LongVideoBench 上 +4.30, OVOBench Realtime 上 +5.96。说明 streaming perception 是通用能力, 会迁移。

## 数据 Pipeline — 工程量大

### 收集
5 种运动: basketball (712), soccer (544), ice hockey (402), baseball (399), American football (392)。总 2449 场, 6000+ 小时。WhisperX 提 ASR (解说员实时语音)。

### 清洗 (GPT-5 当裁判)
切 120s segments, GPT-5 对每句话三选一:
- Keep (46.32%): 正确的 commentary
- Edit (37.89%): commentary 但要修球员名/队名等细节
- Delete (15.79%): 广告、主持独白等 noise

edited 句子的 timestamp 在每个 word 上均匀分配 (一句 3-5 秒, 误差容忍)。

### High-Quality Annealing Data
14,786 samples, 每个 16-64s, 要求 80%+ 是 "realtime commentary" (GPT-5 判定)。这批数据专门 boost 实时动作解说能力, 而非 "Ronaldo 上赛季表现" 这种 meta 信息。

为什么需要 annealing? SFT 数据里有很多 team history、season info 之类的 meta 内容, 但真实场景下用户想要 "好球! Ronaldo 突破到禁区!" 这种实时动作描述。Annealing data 把模型行为往这个方向推。

## 训练 recipe

- Base: Qwen2.5-VL-Instruct-7B (arXiv:2502.13923)
- Step 1: SFT on Inf-Streams-Train (525K) + Live-WhisperX-526K (526K)
- Step 2: Annealing on high-quality subset (14K)
- Total: 128 H100-days

## 实验结果

### Captioning (Table 1)

Inf-Streams-Eval (2+ 小时视频) 上的 win rate:

| Model | vs GPT-4o† | vs LiveCC† | vs LiveCC∞ |
|-------|-----------|-----------|-----------|
| Qwen-2.5-VL-7B-Instruct† | 0.01 | 20.44 | 95.97 |
| LiveCC-7B-Instruct† | 15.73 | — | — |
| LiveCC-7B-Instruct∞ | 1.82 | — | — |
| **StreamingVLM∞** | **66.18** | **87.81** | **99.12** |

LiveCC∞ vs GPT-4o 只有 1.82% — full attention 在 2-5 分钟后超训练长度直接崩。StreamingVLM infinite mode 66.18% 击败 GPT-4o mini。

### ReKV Paradox (Table 2)

ReKV 是 training-free streaming 方法。但 paper 发现一个 paradox:

| Model | vs GPT-4o† | vs LiveCC† | vs LiveCC∞ |
|-------|-----------|-----------|-----------|
| Qwen + ReKV∞ | 0.00 | 19.56 | 63.57 |
| StreamingVLM + ReKV∞ | 0.00 | 0.00 | 0.00 |
| StreamingVLM + Ours∞ | **66.18** | **87.81** | **99.12** |

Qwen + ReKV: 模型没 fine-tune, 能力不足, 表现差。
StreamingVLM + ReKV: 模型 fine-tuned, 但 ReKV 的 eviction policy 破坏了 model 依赖的固定 context format, 经常无输出 (0.00)。

这是 paper 最 insightful 的发现: **training-inference alignment 是必须的**, training-free eviction 在 fine-tuned model 上根本不工作。这点社区很多人没注意到。

### VQA (Table 3)

| Model | MVBench | VideoMME | LongVideoBench | OVOBench Realtime |
|-------|---------|----------|----------------|-------------------|
| Qwen-2.5-VL-7B-Instruct | 67.34 | 65.10 | 54.70 | 56.00 |
| StreamingVLM | 69.16 | 65.10 | 59.00 | 61.96 |

零 VQA-specific SFT, 但 LongVideoBench +4.30, OVOBench Realtime +5.96。说明 streaming perception 是通用能力。

### Efficiency (Figure 7)

单 H100, latency 一直稳定 <100ms/token, 支持 8 FPS 实时。Full attention 很快 OOM, sliding window (no overlap) 周期性 spike, sliding window (overlap) 一直高 latency。

## Ablation 几个关键 insight

### Window Size (Table 5 左)

| $T_{\text{sink}}$ | $T_{\text{window}}$ | vs GPT-4o† | vs LiveCC∞ |
|------|-------|-----------|-----------|
| 512 | 0 | 69.68 | 99.19 |
| 0 | 512 | 66.76 | 98.69 |
| 512 | 512 | 73.64 | 99.38 |
| ∞ | ∞ (no eviction) | 60.41 | 98.55 |

No eviction 反而最差! 因为训练时学的是 eviction pattern, 推理时没 eviction 反而 OOD。Eviction 本身是模型学出来的能力。

### Vision Window (Table 5 右)

| $V_{\text{window}}$ | vs GPT-4o† |
|-------|-----------|
| 0s | 52.90 |
| 1s | 63.46 |
| 16s | **66.18** |
| 32s | 65.49 |

0s → 16s 巨大提升 (+13.28), 16s 后饱和。16 秒足够一个 action play, 再长没用。

### Overlapped vs Non-overlapped (Table 7)

| Strategy | vs GPT-4o† | OVOBench Realtime |
|----------|-----------|-------------------|
| Non-overlapping | 62.51 | 59.20 |
| Overlapped | **66.18** | **61.96** |

+3.67 win rate, +2.76 OVOBench。Overlap 是 training-inference alignment 的核心 trick。

## 跟 StreamingLLM 的关系

StreamingLLM (arXiv:2309.17453) 是 text LLM 的奠基工作。它的 recipe 是 "attention sink + sliding window + contiguous RoPE", **training-free**, 因为 text LLM 已经训得很好, 推理时直接套这个 recipe 就行。

StreamingVLM 把这套搬到 cross-modal streaming video, 但发现一个关键差异: **video 任务必须 fine-tuning** (base VLM 不懂 streaming commentary), 而 fine-tuning 后 model 依赖固定 context format, training-free eviction (ReKV) 直接破坏 format 导致无输出。

所以 StreamingVLM 的核心 contribution 就是 **加上 overlapped SFT**, 让 training attention pattern 跟 inference pattern 对齐。这是 StreamingLLM 在 text 领域不需要做的事。

## 我的几个 critical thoughts

### 这个 recipe 的本质

Sink + sliding window + contiguous RoPE 本质上是 **hard-coded information routing**。Sink tokens = "always-on" memory, recent window = working memory, evicted = forgotten。Karpathy 你在 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) 里讲 attention 就是 routing information, 这里是把这个 routing 部分硬编码, 部分学习。

### Asymmetric retention 的直觉

Vision 短 text 长, 类似人类记忆机制。视觉 sensory memory 几秒到十几秒, 然后 iconic memory 转化成 verbal memory 进入 long-term。StreamingVLM 用 text commentary 当 "verbal encoding", 这跟人类认知科学一致。

### Open questions

1. **Sink/window size 自适应**: 现在 $T_{\text{sink}}=512$, $V_{\text{window}}=16s$ 都是 hyperparameter。不同运动节奏不同 (basketball fast break vs baseball 慢节奏), adaptive sizing 可能更好。
2. **Eviction policy**: 现在 vision FIFO, text LRU。Learned eviction (类似 H2O 的 heavy hitter) 可能更好, 但 training-inference alignment 更难。
3. **Generalization**: 只测了 sports commentary。Autonomous driving, surveillance, embodied agent 场景未验证。
4. **Eval bias**: GPT-5 当 judge, pairwise win rate, 没 absolute quality metric。
5. **算力门槛**: 128 H100-days 不便宜, 复现成本高。

### 跟 scaling laws 的关系

Karpathy 你常讲 scaling。StreamingVLM 的 insight 是: **streaming scenario 下, naive scaling training context length 解决不了问题**。因为 quadratic cost 让你训不动 long context, 必须 align training 和 inference 都用 bounded context。这跟 LongLoRA (arXiv:2309.12307), LongVILA (arXiv:2502.03417) 的思路一致 — 长 context 训练昂贵, 要找 efficient alternative。

## Key Takeaways

1. **核心问题**: streaming video 理解的瓶颈是 training-inference misalignment, 不是单纯 context length。
2. **核心方案**: 三段 KV cache (sink + text window + vision window) + Contiguous RoPE + Overlapped SFT。
3. **关键 insight**: Asymmetric retention (vision 短 text 长) 利用 modality 信息密度差异; Overlap 让 training attention pattern 匹配 inference pattern。
4. **关键数据**: 单 H100 上 8 FPS, 2+ 小时视频 66.18% win rate vs GPT-4o mini, 零 VQA SFT 但 VQA 通用提升。
5. **关键 lesson**: Training-free eviction 在 fine-tuned model 上不工作 (ReKV paradox), 必须 training-inference joint design。

## Reference Links

- **Paper repo**: https://github.com/mit-han-lab/streaming-vlm
- **StreamingLLM** (基础): https://arxiv.org/abs/2309.17453
- **Qwen2.5-VL** (base): https://arxiv.org/abs/2502.13923
- **LiveCC** (baseline): https://arxiv.org/abs/2504.16030
- **ReKV** (对比): https://arxiv.org/abs/2503.00540
- **VideoLLM-online**: https://arxiv.org/abs/2406.11816
- **YaRN**: https://arxiv.org/abs/2309.00071
- **LongRoPE**: https://arxiv.org/abs/2402.13753
- **H2O**: https://arxiv.org/abs/2306.14048
- **SnapKV**: https://arxiv.org/abs/2404.14469
- **VideoMME**: https://arxiv.org/abs/2405.21075
- **MVBench**: https://arxiv.org/abs/2311.17005
- **OVOBench**: https://arxiv.org/abs/2501.05510
- **LVBench**: https://arxiv.org/abs/2406.08035
- **LongVILA**: https://arxiv.org/abs/2502.03417
- **LongLoRA**: https://arxiv.org/abs/2309.12307
- **LLaVA-OneVision**: https://arxiv.org/abs/2408.03326
- **VideoLLaMA 2**: https://arxiv.org/abs/2406.07476
- **InternVideo2**: https://arxiv.org/abs/2403.15377
- **InternVideo2.5**: https://arxiv.org/abs/2501.12386
- **MIT HAN Lab**: https://hanlab.mit.edu/

人话总结: 这篇就是把 StreamingLLM 的 text streaming recipe 搬到 video, 但发现 video 任务必须 fine-tune, fine-tune 后 training-free eviction 不工作, 所以补上 overlapped SFT 让 training 和 inference 对齐。Vision 短 text 长的 asymmetric retention 是利用 modality 信息密度差异的核心设计。最终单卡 8 FPS 跑 2 小时视频流, 击败 GPT-4o mini。

Karpathy 你觉得 training-inference alignment 是不是 streaming model scaling 的真瓶颈?我直觉上觉得这比单纯堆 context length 重要得多。

---

# StreamingVLM 深度解析

Hey Andrej, 这篇 paper 来自 MIT-Han-Lab (Song Han group) 和 NVIDIA, 第一作者 Ruyi Xu 和 Guangxuan Xiao — 后者正是 StreamingLLM 的作者。所以本质上, 这篇工作是把 text LLM 中 "attention sink + sliding window" 的 recipe 迁移到 cross-modal streaming video 场景, 并且补上了训练-推理对齐这一缺失的环节。

## 1. 问题本质: 为什么 streaming video 这么难

先 build intuition。视频流理解的核心矛盾在于:

- **Full Attention**: cost 是 $O(T^2)$, 其中 $T$ 是视频 token 数。对于无限流, 内存无界, 并且超过 training length 后 RoPE index 外推, 性能崩塌 (Figure 2 里 LiveCC-7B 在 2-5 分钟后开始重复 "shot shot shot...")。
- **Sliding Window (no overlap)**: 内存有界, 但每次 chunk 边界处 context reset, coherence 丧失。为了 coherence 你想要大 chunk, 但大 chunk 又回到 latency 问题。
- **Sliding Window (with overlap)**: 保留 recent tokens, 但每个 window 都要 recompute attention 到所有历史 tokens, 效率极差 (Figure 7 里 latency 一直很高)。

那么核心 question 就是: **能不能在 short videos 上训练, 但推理时稳定地处理 infinite stream?**

## 2. StreamingVLM 的三段式 KV Cache

推理时的核心设计 (Figure 3):

$$\text{KV cache} = \underbrace{T_{\text{sink}}}_{\text{attention sink}} + \underbrace{T_{\text{window}}}_{\text{recent text}} + \underbrace{V_{\text{window}}}_{\text{recent vision}}$$

具体数值:
- $T_{\text{sink}} = 512$ tokens (system prompt + previous text 的开头部分)
- $T_{\text{window}} = 512$ tokens (最近的文本输出)
- $V_{\text{window}} = 16s$ (最近的视觉 tokens, 覆盖 16 秒)

### 为什么是这种 asymmetric retention?

这是 build intuition 的关键点。视觉 tokens 和 text tokens 的"信息密度"和"时效性"不同:

- **Vision tokens**: 描述瞬时动作, 16 秒足够覆盖一个 play 的发生 (比如一个罚球、一次进攻)。更早的 vision tokens 信息已经被 text commentary "压缩"进了语言。
- **Text tokens**: 承载长期记忆 (比如 "Ronaldo 已经进了 3 个球"), 需要更长的窗口保留。
- **Sink tokens**: 来自 StreamingLLM 的洞察 — attention sink 稳定 attention 分布, 避免 softmax 在 evicted positions 上崩塌。

### Eviction 策略

older vision tokens 优先被 evict; 只有当 budget 超出时才 evict early text。这避免了 Figure 1(c) 中的 recompute 成本。

## 3. Contiguous RoPE — 防止位置漂移

这是 paper 里最 subtle 但最关键的 trick。直觉上:

当你 evict 早期 tokens 后, 后续 tokens 的 RoPE index 如果继续增长, 很快会超过 training length (Qwen2.5-VL 训练长度有限)。RoPE 在外推时性能骤降。

**Contiguous RoPE 的做法**: 当早期 tokens 被移除时, 后续和 incoming tokens 的 RoPE index 被 left-shift, 使得它们的位置与"最后一个保留 token"在数值上 contiguous。

一旦视频长度超过 total window size, effective RoPE index 停止增长, 保持在 bounded range 内。

### 数学直觉

假设 RoPE index 序列原本是 $[0, 1, 2, \ldots, N, N+1, \ldots]$。evict 前 $k$ 个后, native RoPE 变成 $[k, k+1, \ldots, N, N+1, \ldots]$ — index 继续增长。Contiguous RoPE 则 shift 成 $[0, 1, 2, \ldots, N-k, N-k+1, \ldots]$ — index 永远在 $[0, T_{\text{sink}} + T_{\text{window}} + V_{\text{window}}]$ 范围内。

### Qwen-VL 的 3D RoPE

Qwen-VL family 用 3D positional embeddings (time, height, width) for vision tokens。所以 StreamingVLM 用 **contiguous 3D RoPE**: vision tokens 按 3D 规则组装 index, 整体仍 left-shifted 保持 contiguous, 匹配 interleaved vision-text layout。

Table 4 的 ablation 非常 striking: native RoPE 在 infinite mode ($\infty$) 上 win rate 只有 25.09% vs GPT-4o, 而 contiguous RoPE 是 66.18%。这说明位置外推是无限推理的最大杀手之一。

## 4. 训练策略: Overlapped Full Attention SFT

这是 paper 的另一核心创新。问题: 你不能在无限长视频上训练 (quadratic cost), 但又希望模型推理时能稳定处理无限流。

### Overlapped Chunking

将长视频流切成 consecutive chunks $\{C_1, C_2, \ldots\}$, 每个 chunk 长度 $W = 24s$, 相邻 chunk 间 overlap $O = 12s$ (即 $0 < O < W$)。

每个 chunk 作为独立 training instance, **chunk 内 full attention** — 也就是每个 token 可以 attend 到同一 chunk 内所有 tokens。

### 为什么 overlap 这么重要?

Figure 4 的右图揭示关键: overlapped full attention 的 effective attention pattern 几乎完美匹配 inference 时的:
1. Attention sink (保留的开头 tokens)
2. Longer recent text window
3. Shorter recent vision window

因为 chunk $C_i$ 和 $C_{i+1}$ 有 12s overlap, 训练时模型见过 "前 12s 作为 history + 后 12s 生成"的模式, 这正好对应推理时"recent window + 新 incoming frames"的状态。

### V/T Interleaving

传统 VLM 把所有 vision tokens 放在 text 前。StreamingVLM 则在 chunk 内 **1s 间隔 interleave vision 和 text tokens**。这个设计匹配实时 commentary 的本质: 每秒有 vision frame, 可能有/没有 narration。

当某一秒没有 narration 时, 插入 placeholder token "..."。Loss 只算 text positions (aligned to per-second narration)。这教会模型 **when to speak and when to remain silent** — 这是 streaming behavior 的核心。

## 5. 数据 Pipeline

### Inf-Streams-Train
- 5 种运动: basketball (712), soccer (544), ice hockey (402), baseball (399), American football (392)
- 共 2449 full games, 总时长 6000+ 小时
- 360P-720P, 24 FPS
- WhisperX 提取 ASR (实时解说语音)

### 数据清洗 (GPT-5 做裁判)
分 120s segments, GPT-5 对每句话做 3 选 1:
- **Keep** (46.32%): 正确的 commentary, timestamp 保留
- **Edit** (37.89%): commentary 但需修正细节 (球员名/队名错误), 均匀分配 timestamp
- **Delete** (15.79%): 广告、主持独白等非合规内容

### Inf-Streams-Eval
- 20 full games, 平均 2.12 小时
- 分 100s segments, 至少 200 words
- GPT-5 做 pairwise judge (win rate)
- 两种 setting: chunk (†) 和 infinite (∞)

### High-Quality Annealing Data
14,786 samples, 每个 16-64s, 要求 "realtime commentary" 比例 > 80% (GPT-5 判定)。这数据专门 boost 实时动作解说能力, 而非团队历史等 meta 信息。

## 6. 训练细节

- Base model: Qwen2.5-VL-Instruct-7B
- **Step 1**: SFT on Inf-Streams-Train (525K samples) + LiveCC's Live-WhisperX-526K (526K samples)
- **Step 2**: 高质量 annealing (14K samples, 16-64s)
- 总算力: 128 H100-days

## 7. 实验结果

### Captioning (Table 1)

Inf-Streams-Eval 上的 win rate (vs baselines):

| Model | vs GPT-4o† | vs LiveCC† | vs LiveCC∞ |
|-------|-----------|-----------|-----------|
| Qwen-2.5-VL-7B-Instruct† | 0.01 | 20.44 | 95.97 |
| LiveCC-7B-Instruct† | 15.73 | — | — |
| LiveCC-7B-Instruct∞ | 1.82 | — | — |
| **StreamingVLM∞** | **66.18** | **87.81** | **99.12** |

注意 LiveCC∞ vs GPT-4o 只有 1.82% — 因为 full attention 在 2-5 分钟后超过 training length, 性能崩溃。StreamingVLM 用 infinite mode 直接 66.18% 击败 GPT-4o mini。

### ReKV 对比 (Table 2)

ReKV 是 training-free 的 streaming inference 方法。但有个 paradox:
- Non-fine-tuned models 用 ReKV: 能力不足, 表现差
- Fine-tuned models (StreamingVLM) 用 ReKV: ReKV 的 eviction policy 破坏了 fine-tuned model 依赖的固定 context format, 经常无输出 (0.00%)

这说明 **training-inference alignment 是必须的** — 不能 training-free 地指望 eviction 工作。

### VQA (Table 3)

| Model | MVBench | VideoMME | LongVideoBench | OVOBench Realtime |
|-------|---------|----------|----------------|-------------------|
| Qwen-2.5-VL-7B-Instruct | 67.34 | 65.10 | 54.70 | 56.00 |
| StreamingVLM | 69.16 | 65.10 | 59.00 | 61.96 |

注意: **没有任何 VQA-specific fine-tuning**, 但 StreamingVLM 在 LongVideoBench +4.30, OVOBench Realtime +5.96。这表明 overlapped SFT 教会的"streaming perception"是通用能力, 迁移到 VQA 也有效。

### Efficiency (Figure 7)

- Full Attention: 很快 OOM
- Sliding Window (no overlap): 周期性 latency spike (chunk 开头重建 context, 后期 latency 急升)
- Sliding Window (with overlap): 一直低效 (redundant computation)
- **StreamingVLM**: latency 稳定且低, 单 H100 上 8 FPS 实时

## 8. Ablation Insights

### Sink/Window Size (Table 5, 左)

| $T_{\text{sink}}$ | $T_{\text{window}}$ | vs GPT-4o† | vs LiveCC† | vs LiveCC∞ |
|------|-------|-----------|-----------|-----------|
| 512 | 0 | 69.68 | 89.42 | 99.19 |
| 0 | 512 | 66.76 | 86.03 | 98.69 |
| 256 | 256 | 70.17 | 91.79 | 99.62 |
| 512 | 512 | 73.64 | 92.33 | 99.38 |
| ∞ | ∞ (no eviction) | 60.41 | 72.08 | 98.55 |

关键 insight: no eviction (T=∞) 反而最差 — 因为没有训练过 eviction pattern, 推理时 context 增长导致 OOD。Eviction 本身是训练教出来的能力。

### Vision Window (Table 5, 右)

| $V_{\text{window}}$ | vs GPT-4o† | vs LiveCC† | vs LiveCC∞ |
|-------|-----------|-----------|-----------|
| 0s | 52.90 | 77.49 | 97.56 |
| 1s | 63.46 | 83.24 | 98.18 |
| 16s | **66.18** | **87.81** | **99.38** |
| 32s | 65.49 | 85.58 | 99.06 |

0s 到 16s 提升巨大 (+13.28), 16s 后基本饱和。说明 16s 足够覆盖一个 action play, 再长收益边际递减。

### Overlapped vs Non-overlapped (Table 7)

| Strategy | Inf-Streams∞ vs GPT-4o† | OVOBench Realtime |
|----------|----------|------|
| Non-overlapping | 62.51 | 59.20 |
| Overlapped | **66.18** | **61.96** |

Overlap 提供 +3.67 win rate 和 +2.76 OVOBench, 证明 overlap 是 training-inference alignment 的关键。

### Stability over time (Figure 8)

将 2 小时视频分 5 段 (20% intervals), StreamingVLM 在每段 win rate 几乎恒定, 证明 infinite inference 不退化。

### Sink Size Sensitivity (Table 8)

$T_{\text{sink}}$ 从 64 到 1024, win rate 从 72.04 → 74.82, 大致单调上升但 plateau。1024 时略优于 512, 但计算代价增加。

## 9. 与相关工作的 positioning

### Text LLM Streaming
- **StreamingLLM** (Xiao et al., 2024): attention sink + sliding window 的奠基工作。StreamingVLM 把这个 idea 带到 cross-modal, 但补上了 training strategy (StreamingLLM 是 training-free 的, 但 video 任务需要 fine-tuning)。
- **YaRN / LongRoPE / LongLoRA**: RoPE 外推方法。Contiguous RoPE 思路类似但针对 cross-modal step-wise updates。
- **H2O / SnapKV / ReKV**: KV cache compression via heavy hitters。但这些都是 text-only, 且 training-inference alignment 未探索。

### Streaming Video LLMs
- **VideoLLM-online (LIVE)**: 把 offline data 转 streaming dialogue, 但 >5 分钟性能下降。
- **VideoStreaming**: 固定 video token budget 处理长视频。
- **LiveCC**: 大规模 ASR-video alignment, sports commentary。StreamingVLM 基于 LiveCC 的 Live-WhisperX-526K 训练, 但加了 Inf-Streams-Train + overlapped SFT, 显著超越 LiveCC。

### Benchmarks
现有 benchmark (VideoMME, LVBench) 多为 retrieval/summary, 不要求 frame-level 理解, 低 FPS 也能通过。Inf-Streams-Eval 专为 near-infinite commentary 设计, 要求 second-level alignment, 更接近真实场景 (assistants, robots, autonomous driving)。

## 10. 我的 critical thoughts

### Strengths
1. **Training-inference alignment** 是关键 insight。ReKV 的失败 (Table 2) 证明 training-free eviction 在 fine-tuned model 上不工作 — 这是社区少有人注意到的点。
2. **Asymmetric retention** (长 text 窗 + 短 vision 窗) 利用了 modality 信息密度差异, 是 elegant 的设计。
3. **Contiguous RoPE** 解决外推, ablation (Table 4) 显示这是 deal-breaker。
4. **VQA 通用性提升**: 不用 VQA SFT 也能提升 LongVideoBench +4.30, 说明 streaming perception 是 foundational 能力。

### Open questions / 可改进
1. $T_{\text{sink}} = 512$ 是固定值, 但不同任务的最优 sink size 不同 (Table 8 显示 1024 略好)。自适应 sink 选择是个方向。
2. Vision window 固定 16s — 但不同运动节奏不同 (basketball vs baseball)。Adaptive vision window 可能更好。
3. 评测以 GPT-5 做 judge, 可能有 bias。Pairwise win rate 也是 relative metric, 没有 absolute quality。
4. 只测了 sports commentary, 其他 streaming 场景 (surveillance, autonomous driving) 未验证。
5. 128 H100-days 不算小算力, 复现门槛较高。

### 与 Karpathy 视角的连接
你在 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) 和 [Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNBPjAc) 里反复强调 attention 的本质是 routing information。StreamingVLM 的 sink + sliding window 本质上是在做 **hard-coded routing**: sink tokens 作为 "always-on" memory, recent window 作为 working memory, evicted tokens 作为 "forgotten"。

人类 working memory 也是类似的 (短时记忆 + 长期压缩)。这里的 asymmetric retention (vision 短, text 长) 类似人类把瞬时视觉体验快速压缩成语言记忆的过程。

## 11. 公式补充

### RoPE 基础
对于 query $q$ 和 key $k$ 在位置 $m, n$, RoPE 应用旋转:

$$q_m' = q_m e^{im\theta}, \quad k_n' = k_n e^{in\theta}$$

Attention score:

$$\text{Re}\langle q_m', k_n' \rangle = \text{Re}\langle q_m, k_n \rangle \cos((m-n)\theta) + \text{Im}\langle q_m, k_n \rangle \sin((m-n)\theta)$$

只依赖 relative position $m - n$。这就是为什么 contiguous RoPE 有效 — 只要 relative distance 保持 in-distribution, absolute index 可以 shift。

### KV Cache Update
时间步 $t$, incoming tokens $\Delta_t$:

$$\text{Cache}_{t} = \text{Sink}_{T_{\text{sink}}} \cup \text{TextRecent}_{T_{\text{window}}}(t) \cup \text{VisionRecent}_{V_{\text{window}}}(t)$$

每步 evict 旧 vision tokens (FIFO on vision), 必要时 evict 旧 text。RoPE index 重新 contiguous 映射到 $[0, T_{\text{sink}} + T_{\text{window}} + V_{\text{window}}]$。

### Overlapped Chunk Attention
训练时 chunk $C_i = [s_i, s_i + W]$, overlap $O$。Chunk 内 attention matrix:

$$A_{C_i}[p, q] = 1 \quad \forall p, q \in C_i$$

即 full attention within chunk。这模拟推理时 effective attention pattern (sink + recent windows)。

## 12. 关键链接

- **Paper GitHub**: https://github.com/mit-han-lab/streaming-vlm
- **StreamingLLM** (基础工作): https://arxiv.org/abs/2309.17453
- **Qwen2.5-VL** (base model): https://arxiv.org/abs/2502.13923
- **LiveCC** (baseline): https://arxiv.org/abs/2504.16030
- **ReKV** (对比): https://arxiv.org/abs/2503.00540
- **VideoLLM-online**: https://arxiv.org/abs/2406.11816
- **YaRN** (RoPE extension): https://arxiv.org/abs/2309.00071
- **LongRoPE**: https://arxiv.org/abs/2402.13753
- **H2O** (KV compression): https://arxiv.org/abs/2306.14048
- **SnapKV**: https://arxiv.org/abs/2404.14469
- **VideoMME** (benchmark): https://arxiv.org/abs/2405.21075
- **MVBench**: https://arxiv.org/abs/2311.17005
- **OVOBench**: https://arxiv.org/abs/2501.05510
- **LVBench** (long video): https://arxiv.org/abs/2406.08035
- **LongVILA**: https://arxiv.org/abs/2502.03417
- **LongLoRA**: https://arxiv.org/abs/2309.12307
- **LLaVA-OneVision**: https://arxiv.org/abs/2408.03326
- **VideoLLaMA 2**: https://arxiv.org/abs/2406.07476
- **InternVideo2**: https://arxiv.org/abs/2403.15377
- **InternVideo2.5**: https://arxiv.org/abs/2501.12386
- **MIT HAN Lab**: https://hanlab.mit.edu/
- **GPT-4o** (baseline): https://arxiv.org/abs/2303.08774

## 13. 总结

StreamingVLM 的核心 contribution 是 **training-inference alignment for streaming video**。三个支柱:

1. **Inference**: sink (512) + text window (512) + vision window (16s) + Contiguous RoPE
2. **Training**: overlapped chunk full attention + V/T interleaving + loss on narrated positions only
3. **Data**: 4000+ 小时 sports commentary + GPT-5 cleaning pipeline + high-quality annealing subset

最终结果: 8 FPS on single H100, 66.18% win rate vs GPT-4o mini on 2+ 小时 infinite streams, VQA 通用能力也有提升。

这篇工作大概率会成为 streaming VLM 领域的 reference 点, 类似 StreamingLLM 在 text LLM 中的地位。后续工作很可能在这个框架上加 adaptive window sizing, better eviction policies (learned 而非 FIFO), 或扩展到非 sports domain。

Karpathy 你看这个工作的直觉对吗?我特别好奇你对 training-inference alignment 这个点的看法 — 是否认为这就是 scaling streaming model 的关键 bottleneck?
