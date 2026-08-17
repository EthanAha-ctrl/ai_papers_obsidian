---
source_pdf: VideoARM Agentic Reasoning over Hierarchical Memory.pdf
paper_sha256: 9b8fd28a64e6218b718ea192a563504fd2aa64b9cfb2fd40502fc046f118739e
processed_at: '2026-08-13T00:48:07-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VideoARM 用人话版

Andrej, 我重新讲一遍,这次抛开 paper 那种 academic 八股,直接讲 intuition。

---

## 1. 这篇 paper 到底在解决什么问题?

想象你看一个 30 分钟的电影,有人问你:"Jane 在办公室见到 Grayson 的时候哭了,她当时什么心情?"

你怎么回答这个问题?你不会把 30 分钟电影每一帧都看一遍。你的做法是:
1. 先扫一眼全局,大概知道哪儿有办公室场景
2. 重点关注 Jane 和 Grayson 同时出现的片段
3. 发现某个片段视觉看不清楚情绪,于是去听那段对话的 audio
4. 听完发现 Jane 说的是"You and your knees are perfect" — 这明显是被 touched
5. 回答: B. Touched

这就是 VideoARM 干的事。它让 MLLM 模拟人类看长视频的方式:**先粗看,定位,再细看,实在不行听 audio,边看边记笔记**。

---

## 2. 之前的方法为什么不行?

### 2.1 笨办法一:全部塞进去 (context extension)
代表: Gemini-1.5-Pro, Long context transfer

直觉: 把整个 30 分钟视频所有 frames 全部喂给模型,让它自己看着办。

问题: 30 分钟视频按 2 FPS 采样就是 3600 帧,每帧 ~1105 tokens,光视觉就是 4M tokens。GPT-4o context 是 128K,根本塞不下。Gemini 1.5 Pro 号称 2M context 但实测效果也不行,因为 attention 被冗余信息稀释。

### 2.2 笨办法二:压缩 token (token compression)
代表: LongVU, DyCoke, AdaReTaKe

直觉: 既然太多,我就把不重要的 frames 扔掉,只保留关键 frames。

问题: 关键 frame 的判断是 query-agnostic 的。比如问"那个穿红衣服的人后来去哪了",压缩算法可能觉得"这是背景镜头"就丢了,但其实那个镜头里有红衣服的人经过。

### 2.3 笨办法三: 先建数据库再检索 (DVD 这类)
代表: [DVD](https://arxiv.org/abs/2505.18079), VideoTree

DVD 的做法:
1. 把视频切成 10 秒一个 clip
2. 对每个 clip 用 MLLM 生成 caption + embedding,建一个 database
3. 用户问问题,embed 问题,从 database 检索 top-k clips
4. 把 retrieved clips 喂给 LLM 回答

问题 A: 这个 database 是 **query-agnostic** 的。无论你问什么问题,DVD 都要先花 3.98M tokens 把整个视频描述一遍。问 1 个 query 和 100 个 query,预处理成本一样。

问题 B: 检索完之后 agent 看到的只是 retrieved clips,看不到全局。如果检索错了,没有 self-correction 机制。这就像你 Google 一个问题,只看 top 3 结果,从来不翻第二页。

VideoARM 的核心 insight: **砍掉预处理,边推理边建 memory**。

---

## 3. VideoARM 的核心思路:一个会记笔记的视频助手

### 3.1 整体循环

整个 framework 就是一个 loop,每一步 controller (由 o3 充当) 做四件事:

**Observe**: 看一眼当前 memory 里有什么 (HM³ 当前状态)
**Think**: 推理"下一步我需要看哪儿、用什么工具"
**Act**: 调一个 tool (例如 Scene Snapper 看 4000-6000 帧区间)
**Memorize**: 把 tool 的输出写回 memory,清掉短期 buffer

循环到 controller 觉得"我有信心回答了"或者跑完 10 步预算为止。

### 3.2 为什么这个比 DVD 好?

DVD: **先花 4M tokens 建库,然后 cheap query**
VideoARM: **不建库,每个 query 从头开始推理,但是用 adaptive sampling 把 token 控制在 80K 以内**

为什么 VideoARM 总能用 1/50 的 token 做得更好?因为:
- VideoARM 不需要看所有 clips,只看 query-relevant 的部分
- 看完一个地方发现线索指向另一个地方,可以 dynamically 调整方向
- HM³ 保留了 reasoning history,避免重复探索

---

## 4. HM³: 分三层记忆,模拟人脑

这个设计是 paper 真正的创新点。人脑有 sensory memory (感觉记忆)、working memory (工作记忆)、long-term memory (长期记忆)。HM³ 借鉴这个分法:

### Tier 1: Sensory Memory (感知记忆)
分两个 pool:

**Long-term perception pool $P_l$**: 当前 controller 关注的 frame interval 的 snapshot。用 3×2 image grid 压缩存储,每张图上面标了 frame index。这是 controller 的"视野"。
- 例如当前关注 5000-5350 帧,$P_l$ 里就存这个区间的 6 帧拼图
- 一旦 controller 用 Interval Localizer 换关注区间,$P_l$ 被覆盖

**Short-term perception pool $P_s$**: 临时存 fine-grained clips + audio,用于 Clip Explorer 之类的工具快速验证假设。用完就清空。
- 例如 Clip Explorer 探查 5000-5350 帧的 30 帧细节 + 对应 audio,放进 $P_s$
- Clip Analyzer 处理完,把结果存到 Result Memory,然后 $P_s \leftarrow \emptyset$

直觉上,**$P_l$ 是"我大致看哪儿",$P_s$ 是"我现在 zoom in 看这个细节"**。

### Tier 2: Result Memory (结果记忆)
这是 agent 的 "episodic memory"。每条记录:
- 第 $t$ 次迭代
- 看了哪个 frame interval
- tool 输出什么 (caption / sub-question answer / transcript)

按时间顺序排列。Controller 每次推理前先读 Result Memory:"我刚才看过哪些区段?得到什么线索?" 避免重复探索。

### Tier 3: Working Memory (工作记忆)
存 controller 自己的 reasoning trace $R_t$ + intended objective。

**关键设计**: 把 trace 从 MLLM 的 context **搬出来**。为什么不留在 context?因为 MLLM context 是有限的。如果每一步的 thought + observation 都堆在 context 里,跑 10 步就爆掉了。

把 trace 卸载到 Working Memory 后,**每一步开始时 refresh controller 的 context**,只读 HM³ 的当前 state。这样 controller 每一步都在"干净 context"上思考,reasoning 更 focused。

这是一个很 subtle 但重要的设计。它意味着 HM³ 取代了 MLLM 自己的 context 作为长期记忆载体。

---

## 5. 工具集:Controller 手里的 5 张牌

Controller 有两组工具,**5 个 tool 一一对应**:

### 5.1 Temporal Scoping Tools (决定看哪儿)

#### (a) Interval Localizer
作用: 给定当前 HM³ 的线索,定位 query 最相关的 frame interval $T_{long} = [t_i, t_j]$,并自适应决定采样帧数 $N_1 \in \{30, 60, 90, 150\}$。然后 uniform 采样 $N_1$ 帧,拼成 3×2 grid,写入 $P_l$。

**为什么 adaptive $N_1$ 重要?** 简单场景 30 帧够了,复杂场景需要 150 帧。Ablation Table 6 显示 adaptive (avg 49.8) 比 fixed 60 还好,说明 video 信息密度不均匀,按需采样更优。

#### (b) Clip Explorer
作用: 不改变 $P_l$,在 $P_l$ 当前关注的区间附近,做 fine-grained probing。固定帧数 $N_2$,直接把 frames + audio 存 $P_s$。

直觉: Interval Localizer 是"我换个区域看",Clip Explorer 是"我在当前区域附近 zoom in 看细节"。

### 5.2 Multimodal Understanding Tools (看完了提取信息)

#### (c) Scene Snapper
公式 (1):
$$V_C = \text{SceneSnapper}(\mathcal{F}), \quad \mathcal{F} \in P_l$$

输入: $P_l$ 中的 frames $\mathcal{F}$
输出: concise caption $V_C$

作用: 给当前 long-term pool 里的 frames 生成一句话总结。这是 **coarse-level global summarization**。

例子 (paper Figure 6): Scene Snapper 给 0-3000 帧生成 "The frames depict a competitive gaming event showcasing the Champion Selection phase for a League of Legends match..."

#### (d) Audio Transcriber
公式 (2):
$$A_C = \text{AudioTrans}(\mathcal{A}), \quad \mathcal{A} \in P_s$$

输入: $P_s$ 里的 audio segment $\mathcal{A}$
输出: transcript $A_C$ (用 whisper-1)

作用: 当视觉信息不够 (例如对话决定情绪但表情看不清),用 audio transcript 补充语义。

例子 (paper Figure 6 Jane 哭了那段): audio transcript 给出 "444-483: Do my knees look fat? ... 700-750: You and your knees are perfect." — 这种对话内容直接决定答案是 touched 而不是 sad。

#### (e) Clip Analyzer
公式 (3):
$$A_{sub}, S_{sub} = \text{ClipAnalyzer}(\mathcal{F}, Q_{sub}), \quad \mathcal{F} \in P_s$$

输入: $P_s$ 里的 frames $\mathcal{F}$ + sub-question $Q_{sub}$
输出: answer $A_{sub}$ + confidence score $S_{sub} \in [0,1]$

作用: **fine-grained local 语义分析**。controller 可以问 sub-question 例如"What is the woman's emotional expression in these frames?" 然后得到 "afraid, anxious... Confidence: 0.9"

confidence $S_{sub}$ 是个被低估的信号。理论上 controller 可以用它做 stopping criterion (低 confidence 就继续探索),但 paper 没说有没有用。

---

## 6. Controller: 让 MLLM 自己当老板

### 6.1 Action Space 设计

形式化: action space $\mathcal{A} = \mathcal{T}_{mu} \cup \{\text{ANSWER}\}$

注意一个细节: **action space 只包含 Multimodal Understanding Tools**,不包含 Temporal Scoping Tools。因为每次选了 $\mathcal{T}_{mu}$ 中的某个 tool,自动 trigger 对应的 $\mathcal{T}_{ts}$ 工具先更新 sensory memory。

这个设计简化了 controller 的决策:它只需要决定"我现在用 Scene Snapper / Audio Transcriber / Clip Analyzer 哪个",不用操心"我还要不要先调 Interval Localizer"。两步合并成一步。

### 6.2 推理循环 (公式 6-7)

公式 (6):
$$(A_t, P_t) \sim \pi_\theta(\cdot \mid R_t, M^{(t)}), \quad A_t \in \mathcal{A}$$

变量解释:
- $\pi_\theta$: agentic policy,实际是 controller MLLM (本文用 o3)
- $R_t$: 第 $t$ 轮 controller 生成的 reasoning trace
- $M^{(t)} = (M_s^{(t)}, M_r^{(t)}, M_w^{(t)})$: 当前 HM³ 状态 (sensory, result, working)
- $A_t$: 选中的 tool
- $P_t$: tool 参数 (frame range, sub-question 等)

公式 (7): memory 更新
$$M_s^{(t)} = T_t(M^{(t)}, Info)$$
$$O_t = A_t(P_t; M_s^{(t)})$$
$$M^{(t+1)} = M^{(t)} \cup (R_t, O_t)$$

直觉:
- 先用 temporal tool $T_t$ 更新 sensory memory (定位 frame, sample frames)
- 再用 multimodal tool $A_t$ 处理 frames,得到 observation $O_t$
- 把 reasoning trace $R_t$ 写入 Working Memory,$O_t$ 写入 Result Memory

### 6.3 关键设计哲学

Section 3.3.2 末尾点出: **"we intentionally avoid rigid workflows and pre-defined tool-usage rules"**

意思是: paper 没有 hardcode"先调 Scene Snapper 三次,然后调 Clip Analyzer 两次"这种 schedule。完全让 MLLM 自己根据 HM³ 状态决定下一步。

这听起来 risk,但 ablation Table 2 显示: **controller 用 o3 / gpt-5 这种强 reasoning model 能拿到 75+ 分,用 gpt-4o / Qwen3-VL 直接崩到 40-55 分**。

这说明: **VideoARM 的上限被 controller 的 reasoning 能力决定**。这种 agentic framework 是 reasoning model 时代的产物 — 没有 o3 这种 model,这个 framework 根本跑不起来。

### 6.4 Exploration vs Exploitation

Controller 自己平衡:
- **Exploitation**: 在 local interval 用 Audio Transcriber / Clip Analyzer 做高精度验证
- **Exploration**: 用 Scene Snapper 在 long span gather coarse 证据,refresh HM³,找 promising region

这是 RL 里经典的 exploration-exploitation dilemma。VideoARM 没有用任何 RL,完全靠 MLLM 自己 reasoning 来 balance。

---

## 7. 实验数据看什么

### 7.1 主结果 (Table 1)

| 方法 | Video-MME Long | LongVideoBench Overall | EgoSchema |
|---|---|---|---|
| DVD | 67.3 | 71.6 | 76.6 |
| GPT-4o baseline | 65.3 | 66.7 | 72.2 |
| OpenAI o3 baseline | 63.2 | 67.5 | 63.2 |
| **VideoARM (o3+gpt-4o)** | **81.2** | **78.0** | 76.2 |

三个观察:
1. VideoARM 在 Video-MME Long 上比 DVD 涨 +13.9 points,这是巨大的提升
2. GPT-4o 当 base model 是 65.3,加上 VideoARM framework 涨到 81.2,**framework 给 base model 加了 +15.9 分**
3. EgoSchema 上 VideoARM 没有完全 dominate,因为 EgoSchema 视频只有 3 分钟,VideoARM 的 token-saving 优势体现不出来

### 7.2 Token Efficiency (Table 7)

| 方法 | 30min 视频 1 query 理论值 | 10 视频 30 queries 实测 |
|---|---|---|
| DVD | 3.98M tokens | 64.21M |
| VideoARM | 80K (1/50) | 1.89M (1/34) |

公式 (5):
$$C_{ARM} \leq N \times 8000 = 10 \times 8000 = 80000$$

- $N = 10$: step budget
- $8000$: 每次迭代消耗的 token (visual + text + memory)

实测 1/34 而不是理论 1/50,是因为不是每次都跑满 10 步。但 1/34 已经是 game-changing 的 cost reduction — 同样预算能跑 34 倍的 queries。

### 7.3 Ablation 1: Controller 模型决定一切 (Table 2)

| Controller | 工具用 | Video-MME Long |
|---|---|---|
| Qwen3-VL | gpt-4.1 | 54.9 |
| gpt-4o | gpt-4.1 | 40.5 |
| gpt-5 | gpt-4.1 | 75.5 |
| **o3** | **gpt-4o** | **80.0** |

**这个表告诉你: framework 不是 magic,它放大了 base model 的 reasoning 能力**。gpt-4o 当 controller 直接崩盘,因为它多步推理能力不够。o3 / gpt-5 这种 reasoning model 才能撑住。

### 7.4 Ablation 2: 哪个工具最重要 (Table 3)

| Scene Snapper | Audio | Clip Analyzer | Video-MME Long |
|---|---|---|---|
| ✗ | ✓ | ✓ | 69.0 |
| ✓ | ✗ | ✓ | 70.5 |
| ✓ | ✓ | ✗ | 75.5 |
| ✓ | ✓ | ✓ | **76.5** |

- **Scene Snapper 贡献 +6.0** (long-range summarization 是 backbone)
- **Audio Transcriber 贡献 +6.5** (audio 对消歧极重要)
- Clip Analyzer 只贡献 +1.0 (可能因为前两个工具已经把 fine-grained 信息提取差不多了)

Audio 的 +6.5 让我意外。这说明 **视频理解里 audio 不是 nice-to-have,是必须的**,很多视觉模糊的情境只能靠对话内容消歧。

### 7.5 Ablation 3: HM³ 哪层最重要 (Table 4)

| Long-term $P_l$ | Result Mem | Working Mem | Video-MME Long |
|---|---|---|---|
| ✗ | ✓ | ✓ | 67.0 (-9.5) |
| ✓ | ✗ | ✓ | (反复循环,没结果) |
| ✓ | ✓ | ✗ | 75.5 (-1.0) |
| ✓ | ✓ | ✓ | 76.5 |

- **Long-term pool $P_l$ 最关键** (-9.5 if removed),因为它是 controller 的 attention narrowing 机制
- **Result Memory 是 must-have**,没它 controller 不知道下一步干啥
- **Working Memory 贡献小** (-1.0),但作为 reasoning trace 的可审计性,本身有价值

最后一行"纯 context 不用 HM³" 拿 74.5,比完整 HM³ 只少 2.0。这暗示: **HM³ 真正的价值不是省 token,而是结构化 attention** — 把 controller 的注意力聚焦到正确的信息上。

### 7.6 Ablation 4: Step Budget N (Table 5)

| N | Video-MME Short | Video-MME Long | LongVideoBench |
|---|---|---|---|
| 3 | 87.5 | 72.0 | 67.5 |
| 10 | 84.0 | 76.5 | 70.5 |

长视频上 N 越大越好 (76.5 > 72.0),符合直觉。**短视频上 N 越大反而略降** (87.5 → 84.0),这是过度探索引入 noise 的迹象。这暗示: **N 应该 adaptive**,短视频用小 N,长视频用大 N。

### 7.7 Ablation 5: Adaptive Frame Sampling (Table 6)

| Sampling | Video-MME Long |
|---|---|
| Fixed $N_1 = 30$ | 73.5 |
| Fixed $N_1 = 60$ | 74.0 |
| **Adaptive (avg 49.8)** | **76.5** |

**adaptive 用 49.8 帧平均,比 fixed 60 帧还更好**。这是 video 信息密度不均匀的实证 — 简单场景 30 帧够,复杂场景需要 150 帧。Controller 自己根据 interval 复杂度决定采样数。

---

## 8. 我看到的问题和局限

### 8.1 初始采样瓶颈
作者自己承认 (Section C): 整个 pipeline 的 quality 上限被初始 sampling 决定。如果 query-relevant event 极短 (1 帧) 或 object 严重遮挡 (Figure 5 中的 carousel),controller 无法启动有效 reasoning。

**这是 agent loop 缺少"重新采样"机制的根本问题**。可能的修复:
- Controller 在 $S_{sub} < 0.5$ 时自动 trigger 重采样
- 用 audio cue 作为 fallback anchor
- 用 dense optical flow / scene change detection 做 cheap pre-signal

### 8.2 闭源依赖
Ablation Table 2 显示 Qwen3-VL 当 controller 只有 54.9 分,远不如 o3 的 76.5。作者说开源 model 在 "jigsaw-style video interface" (3×2 image grid) 下 visual grounding 不够。

这暗示一个 **interface design 问题**: 3×2 grid 是为 GPT-4o tuned 的视觉 token layout,开源 VLM 在 dense grid 上还没跟上。如果想让 VideoARM 开源化,可能要重新设计 visual interface。

### 8.3 Agent 不可微
整个 pipeline 是 API 调用串联的,梯度无法 backprop。这意味着:
- 不能 end-to-end 训练
- 不能 fine-tune 中间 tool
- 改进只能靠 prompt engineering 或换更强 base model

这是所有 agentic video understanding 方法的共同瓶颈。Paper 没讨论这点,但我觉得是长期最大问题。

### 8.4 Confidence 信号没充分利用
Clip Analyzer 输出 $S_{sub} \in [0,1]$ confidence,但 paper 没说 controller 怎么用它。理论上 $S_{sub}$ 可以作为:
- Stopping criterion (低 confidence 继续探索)
- Exploration guidance (低 confidence 区域多 sampling)
- Tool choice 信号 (低 confidence 试试 audio 而不是 visual)

这块感觉是个 missed opportunity。

---

## 9. 跟其他工作的关联

### 9.1 vs. ReAct
[ReAct](https://arxiv.org/abs/2210.03629) 是 thought-action-observation 循环,所有 history 堆 context 里。VideoARM 是 ReAct + 分层 memory 的变种: 把 episodic memory 从 context 卸载到外部 store,context 只留当前 working state。这个思路在 [LongAgent](https://arxiv.org/abs/2402.11550) 上有类似探索。

### 9.2 vs. Generative Agents
HM³ 让我想到 [Stanford Generative Agents](https://arxiv.org/abs/2304.03442) 的 memory stream + reflection + planning。Sensory = 观察,Result = memory stream,Working = plan + reflection。这是 cognitive-science-inspired 的 memory architecture 落地。

### 9.3 vs. Memory Transformer / RAG
HM³ 是 **explicitly constructed structured RAG**。RAG 是从外部 corpus 检索,HM³ 是 agent 自己构造+查询。这种 "agent-built KB" 思路在 [VCA (ICCV 2025)](https://arxiv.org/abs/2505.01112) 也有体现。

### 9.4 vs. AlphaGo-style search
观察 → 假设 → 验证 → 收窄 的循环本质上是 **MCTS 的简化版**。Clip Analyzer 的 confidence $S_{sub}$ 类似 value function,但 VideoARM 没有显式 backpropagate value。如果引入 value-based pruning,可能进一步压缩 step budget。

### 9.5 vs. MoE
Controller (o3) + worker (gpt-4o) 的分工让我想到 **agentic 版本的 mixture-of-experts**: 大模型做高阶 reasoning,小模型做廉价 vision task。这种 asymmetric scaling 在 production 上很有意义。

---

## 10. 我会给作者的建议

1. **加 uncertainty-aware stopping**: 用 $S_{sub}$ 做 stopping criterion,别让 controller 拍脑袋决定何时 ANSWER
2. **HM³ 形式化为 graph**: 节点是 frame interval,边是 temporal/semantic relation,支持 multi-hop reasoning
3. **Multimodal fusion in HM³**: 当前 Result Memory 是 text-only。能否存 visual feature vector 实现 visual retrieval?
4. **Cross-query HM³ reuse**: 多个 query 共享 sensory memory,减少重复 preprocessing
5. **Audio 不只 transcript**: 加 audio feature (音乐情绪、声纹、环境音),不要只靠 whisper transcript
6. **Adaptive step budget N**: 短视频自动用小 N,长视频用大 N
7. **Tool choice policy**: 用 RLHF/DPO fine-tune 一个 router 预测 tool choice,减少对 o3 的依赖

---

## 11. 给你的直觉总结

VideoARM 的 mental model:

> **一个拿着笔记本看视频的研究员**
> 
> - **HM³ 是他的笔记本**: $P_l$ 是当前在看的那一页截图,$P_s$ 是临时草稿,Result Memory 是观察记录,Working Memory 是思考痕迹
> - **Tools 是他的工具箱**: Scene Snapper 是望远镜 (看大概),Clip Analyzer 是放大镜 (看细节),Audio Transcriber 是录音笔 (听对话),Interval Localizer 是导航 (决定下一步看哪儿)
> - **Controller 是他的大脑**: 每一步决定用哪个工具,工具用完记笔记,然后翻看笔记本决定下一步
> - **整个流程: 不预先看完整部电影,而是按 query 引导边看边记,看完一部分觉得线索够了就回答**

跟 DVD 的区别一句话: **DVD 是先把整本书读一遍做笔记再回答问题,VideoARM 是带着问题边读边记,自然更省时间**。

这就是 VideoARM 用人话讲的全貌。核心创新是 HM³ 分层 memory + adaptive coarse-to-fine tool scheduling,token 砍到 1/34 同时 SOTA。最大的局限是依赖强 reasoning model (o3) 和初始采样瓶颈。

参考链接:
- [VideoARM paper 本身]
- [DVD (Deep Video Discovery)](https://arxiv.org/abs/2505.18079)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [VideoTree](https://arxiv.org/abs/2412.19407)
- [VideoAgent (ECCV 2024)](https://arxiv.org/abs/2403.16971)
- [Video-MME](https://arxiv.org/abs/2405.21075)
- [LongVideoBench](https://arxiv.org/abs/2407.15754)
- [EgoSchema](https://arxiv.org/abs/2308.09131)
- [Whisper-1](https://arxiv.org/abs/2212.04356)
- [Stanford Generative Agents](https://arxiv.org/abs/2304.03442)
- [LongAgent](https://arxiv.org/abs/2402.11550)
- [VCA (Video Curious Agent)](https://arxiv.org/abs/2505.01112)

Andrej,如果哪一块你想再深挖,告诉我方向 — 我特别想展开的是 confidence $S_{sub}$ 的 underutilization 和 HM³ graph 化的可能方向。

---

# VideoARM: 深度解析 Agentic Reasoning over Hierarchical Memory

Andrej, 这篇 paper 我读得相当仔细。核心 idea 非常 elegant:把 long-form video understanding 重新 cast 成一个 **memory-driven agentic reasoning** 问题,而不是 retrieval-then-reason 的两阶段 pipeline。下面我把整篇 paper 的技术细节按层次拆解给你看,并尽量 build 你的 intuition。

---

## 1. Problem Setup: 为什么这个 task 真的难

Long-form video understanding 的难点在于三个维度同时叠加:
- **Temporal extension**: 视频长度从几分钟到一小时以上,信息呈现"延展式"展开,关键事件可能只占整个时间轴的 <1%
- **Dense multimodal cues**: visual + audio + (sometimes) text 字幕在同一时间轴上互相印证或互相补充
- **Long-range reasoning**: 一个 query 的答案可能需要 cross-modal、cross-temporal 的因果推断

现有的方法大致分三类(参见 Section 2):

| 方法族 | 代表 | 关键问题 |
|---|---|---|
| **Context extension** | Long context transfer [34], Gemini-1.5-Pro [20] | 上下文冗余、计算成本高,且大多数 video MLLM 实际处理 <1M tokens |
| **Token compression** | LongVU [17], DyCoke [19], AdaReTaKe [22] | 压缩导致 fine-grained detail 丢失,影响需要精确视觉推理的任务 |
| **Agent-based** | VideoTree [24], DVD [35], VideoAgent [5] | hand-crafted pipeline 限制 autonomy,或者 token-intensive 预处理 |

VideoARM 的定位是 agent-based,但要把 token cost 砍掉一个数量级,同时让 reasoning 更 principled。

---

## 2. vs. DVD: 这是这篇文章最关键的对比

DVD (Deep Video Discovery) [arXiv:2505.18079](https://arxiv.org/abs/2505.18079) 是这篇 paper 的直接对比 baseline。DVD 的设计:
1. 把视频切成固定 10 秒的 clips
2. 对每个 clip 用 MLLM 做 caption + embedding,建数据库
3. ReAct-style agent 通过 retrieval + reasoning 回答 query

这个设计有两个核心问题:

**Problem A: exhaustive preprocessing 是 query-agnostic 的浪费**
DVD 公式 (4):
$$C_{DVD} = T_v \times r_s \times t_f$$
- $T_v$: total video duration (秒)
- $r_s$: frame sampling rate (= 2 FPS in DVD config)
- $t_f$: 平均每帧的 token cost (~1105 tokens/frame for GPT-4.1 vision at 6 × 512×512 patches)

对 30 分钟视频 (1800 秒):
$$C_{DVD} = 1800 \times 2 \times 1105 = 3.98M \text{ tokens}$$
这还是 **保守下界**,没算 caption 生成、embedding 检索、reasoning 阶段的 token。

**Problem B: retrieval-centric 没有利用 MLLM 的 native visual reasoning**
DVD 一旦 query 来了,embed → retrieve → 看 retrieved clips。整个流程僵化在 retrieval paradigm 里,中间观察到的 evidence 不能 feed back 重新调整 agent 的关注范围。

VideoARM 的 insight:**砍掉 pre-built database,把 reasoning loop 当成主流程,memory 是 loop 的副产物**。

---

## 3. VideoARM 的核心架构:观察 → 思考 → 行动 → 记忆

VideoARM 的核心是一个 observe-think-act-memorize 的循环(图 2)。这个循环让我想到 ReAct [arXiv:2210.03629](https://arxiv.org/abs/2210.03629),但有两个关键升级:

1. **Memory 是分层的** — 不是把所有 history 平铺塞 context,而是按 semantic hierarchy 组织
2. **Tool 是 query-guided coarse-to-fine 的** — 没有先全部预处理的步骤,而是按需 sampling

### 3.1 HM³: Hierarchical Multimodal Memory

这是这篇 paper 真正的 novelty。它分三层,对应认知科学的 sensory → working → episodic:

#### Tier 1: Sensory Memory(感知层)
分两个 pool:
- **Long-term perception pool $P_l$**: 当前 controller 关注的 "基础 frame interval",用 3×2 image grid tiling 方式 compact 表示。这是一个 **volatile buffer**,只保留最近一段时间的 frame snapshot。
- **Short-term perception pool $P_s$**: 在 local exploration 时临时存储 fine-grained clips + audio segment。一旦 analysis 结果写入 Result Memory 就清空。这个设计很像 working memory 的 "transient workspace"。

直觉上,$P_l$ 是"我现在大致关注哪儿",$P_s$ 是"我现在 zoom 进去 zoom 看 details"。

#### Tier 2: Result Memory(结果记忆)
记录每次 tool 调用的:
- iteration index $t$
- analyzed frame interval $[t_i, t_j]$
- tool output(例如 caption $V_C$ 或 sub-question answer $A_{sub}$)

按 **temporal order** 组织,这样 controller 能反思"我刚才看过哪些区段",避免 redundant action,并 adapt 后续策略。这本质上是一个 episodic memory 的 episode list。

#### Tier 3: Working Memory(工作记忆)
记录 controller 自己的 **reasoning trace $R_t$ + intended objective**。

这里有个非常 clever 的设计:把 trace 从 MLLM 自己的 context **externalize** 出来。为什么?因为 MLLM 的 context window 是有限的,如果把每次 tool 调用的中间结果都留在 context 里,几轮就爆掉了。把 reasoning trace 卸载到 Working Memory,然后 refresh controller 的 context,这样可以保证后续推理的 focus 和 efficiency。

这也意味着 **HM³ 取代了 MLLM 自己的 context 作为长期记忆载体**,MLLM 只负责"读 HM³ → 决策 → 写 HM³"这个循环。

### 3.2 Toolsets: coarse-to-fine 的两个工具集

#### Temporal Scoping Tools $\mathcal{T}_{ts}$(时间范围界定)
- **Interval Localizer**: 给定 HM³ 当前的 clues,定位 query-relevant frame interval $T_{long} = [t_i, t_j]$,并 **自适应** 决定采样帧数 $N_1 \in \{30, 60, 90, 150\}$。把 frame composite 成 3×2 image grid,overlay frame index,写入 $P_l$。这一步是 **coarse attention narrowing**。
- **Clip Explorer**: 不改变 $P_l$,只是在 $P_l$ 当前关注区间附近做 fine-grained probing。固定帧数 $N_2$。直接把 frames 存到 $P_s$(不 tile),同时存 audio segment。

直觉上,Interval Localizer 是"我换个大区域看",Clip Explorer 是"我在当前区域附近 zoom in 看细节"。

#### Multimodal Understanding Tools $\mathcal{T}_{mu}$(多模态理解工具)
公式 (1)-(3) 给出了三个 tool 的形式:

$$V_C = \text{SceneSnapper}(\mathcal{F}), \quad \mathcal{F} \in P_l$$

Scene Snapper 给 $P_l$ 中的 frames 生成 concise caption。是 **global summarization**。

$$A_C = \text{AudioTrans}(\mathcal{A}), \quad \mathcal{A} \in P_s$$

Audio Transcriber 用 whisper-1 [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) 从 $P_s$ 取 audio 生成 transcript。这是 **audio grounding**,补 visual 不够的场景(比如对话决定情感)。

$$A_{sub}, S_{sub} = \text{ClipAnalyzer}(\mathcal{F}, Q_{sub}), \mathcal{F} \in P_s$$

Clip Analyzer 是 fine-grained local 语义分析,输出 answer $A_{sub}$ 和 confidence score $S_{sub} \in [0,1]$。这给后续 reasoning 一个 **可信度信号**,可以用于 decide 是否需要进一步探索。

工具执行后,把结果和对应时间间隔写入 Result Memory,然后清空 $P_s$。

---

## 4. Controller: 自主调度,无固定 workflow

这是这篇文章让我觉得最 interesting 的设计选择:**作者刻意 avoid rigid workflows 和 pre-defined tool-usage rules**,完全交给 MLLM 自己决定。

### 4.1 Action space 与 state

形式化:
- Initial query $Q$
- Tool set $\mathcal{T} = \mathcal{T}_{mu}$(注意:**action space 只用 $\mathcal{T}_{mu}$ 的工具,因为 $\mathcal{T}_{ts}$ 和 $\mathcal{T}_{mu}$ 是一一对应的** — 每次选了某个 multimodal understanding tool,自动 trigger 对应的 temporal scoping tool)
- Terminal action ANSWER
- Interaction history $H_t$
- Hierarchical memory $M^{(t)} = (M_s^{(t)}, M_r^{(t)}, M_w^{(t)})$ — sensory, result, working

### 4.2 推理循环公式 (6)-(7)

公式 (6):
$$(A_t, P_t) \sim \pi_\theta(\cdot \mid R_t, M^{(t)}), \quad A_t \in \mathcal{A}$$

- $\pi_\theta$: agentic policy(由 MLLM controller 实现,本文用 o3)
- $R_t$: 第 $t$ 轮的 reasoning trace(由 controller 生成)
- $M^{(t)}$: 当前 hierarchical memory 状态
- $A_t$: 选中的 action(从 $\mathcal{A} = \mathcal{T}_{mu} \cup \{\text{ANSWER}\}$)
- $P_t$: action 参数(例如 sub-question $Q_{sub}$, frame range 等)

公式 (7) 给出 memory 更新规则:
$$M_s^{(t)} = T_t(M^{(t)}, Info)$$
$$O_t = A_t(P_t; M_s^{(t)})$$
$$M^{(t+1)} = M^{(t)} \cup (R_t, O_t)$$

- $T_t$: 对应 $A_t$ 的 temporal scoping tool,先更新 sensory memory $M_s$
- $A_t$: 调用选中的 multimodal understanding tool,得到 observation $O_t$
- $(R_t, O_t)$ 追加到 memory:$R_t$ 进 Working Memory,$O_t$ 进 Result Memory

### 4.3 Exploration vs Exploitation 平衡
论文 Section 3.3.2 末尾点出 controller 的设计原则:
- **Exploitation**: 用 Audio Transcriber 和 Clip Analyzer 在 local interval 做高精度 verification
- **Exploration**: 用 Scene Snapper 在 long temporal span 上 gather coarse 证据,refresh HM³,identify promising regions

这个平衡是靠 **controller 的 reasoning 自己决定**,不是工程化 schedule。这就是为什么 ablation 显示 controller 模型选择很关键(下一节)。

---

## 5. 实验:数字告诉你什么

### 5.1 主结果(Table 1)

| 方法 | Video-MME Long | LongVideoBench Long | LongVideoBench Overall | EgoSchema |
|---|---|---|---|---|
| DVD [35] | 67.3 | 68.6 | 71.6 | 76.6 |
| Gemini-1.5-Pro | 67.4 | 58.6 | 64.0 | 71.1 |
| GPT-4o (baseline) | 65.3 | 60.9 | 66.7 | 72.2 |
| OpenAI o3 (baseline) | 63.2 | 60.6 | 67.5 | 63.2 |
| **VideoARM (o3+gpt-4.1)** | **75.3** | 69.2 | 73.7 | **78.2** |
| **VideoARM (o3+gpt-4o)** | **81.2** | **76.4** | **78.0** | 76.2 |

观察:
- VideoARM 在 Video-MME Long 上比 DVD 涨 +13.9 points(81.2 vs 67.3),提升幅度显著
- EgoSchema 上 VideoARM 没有完全 dominate,因为 EgoSchema 是 3 分钟 egocentric 视频 — 不长,VideoARM 的 token-saving 优势不明显
- VideoARM 给 base model GPT-4o 的提升:Video-MME 65.3 → 81.2(+15.9),证明 agentic loop 真的在放大 base capability
- OpenAI o3 baseline 反而比 GPT-4o 低(63.2 vs 65.3),但 VideoARM (o3+gpt-4o) 最高 — 说明 controller 和 worker 用不同模型最优

### 5.2 Token Efficiency (Table 7)

| 方法 | 理论估计(30min, 1 query) | 实测(10 videos, 30 queries, avg 41.3min) |
|---|---|---|
| DVD | 3.98M tokens | 64.21M |
| VideoARM | **0.08M (1/50)** | **1.89M (1/34)** |

公式 (5):
$$C_{ARM} \leq N \times 8000 = 80000$$
- $N = 10$(step budget)
- $8000$: 每次迭代的 token(包括 visual、textual、memory context)

实测的 1/34 比理论 1/50 稍差,因为 $N$ 没有每次都跑满。但 1/34 仍然是非常显著的 cost reduction,这意味着同样的预算可以做 34 倍的 queries。

### 5.3 Ablation 1: Controller 和 Tools 的模型选择(Table 2)

| Controller | Temporal Scoping | MM Understanding | Audio | Video-MME Long |
|---|---|---|---|---|
| Qwen3-VL | Qwen3-VL | gpt-4.1 | whisper-1 | 54.9 |
| gpt-4o | gpt-4o | gpt-4.1 | whisper-1 | 40.5 |
| gpt-5 | gpt-5 | gpt-4.1 | whisper-1 | 75.5 |
| o3 | o3 | gpt-4.1 | whisper-1 | 76.5 |
| **o3** | **o3** | **gpt-4o** | whisper-1 | **80.0** |

关键洞察:
- **Controller 推理能力是决定性因素** — gpt-4o/Qwen3-VL 当 controller 直接崩盘,而 gpt-5/o3 都能稳在 75+
- **Multimodal Understanding Tools 用更强的 vision model 收益大** — 固定 o3 当 controller,把 gpt-4.1 换成 gpt-4o 给 +3.5 points,说明 worker 端的 visual capability 同样关键
- 这种 **controller/worker 分工** 是 prompt engineering 上一个值得借鉴的设计

### 5.4 Ablation 2: Multimodal Understanding Tools 各组件(Table 3)

| Scene Snapper | Audio Transcriber | Clip Analyzer | Video-MME Long |
|---|---|---|---|
| ✗ | ✓ | ✓ | 69.0 |
| ✓ | ✗ | ✓ | 70.5 |
| ✓ | ✓ | ✗ | 75.5 |
| ✓ | ✓ | ✓ | **76.5** |

观察:
- **Scene Snapper 贡献最大**(+6.0 from no SS) — long-range global summarization 对 downstream reasoning 关键
- **Audio Transcriber 在视觉不够用时极有用**(+6.5 from no AT) — 证明 audio 不是花瓶,确实补 visual ambiguity
- **Clip Analyzer 贡献小**(+1.0) — 这个稍微反直觉,可能因为 fine-grained local analysis 的边际效用已经在 Scene Snapper 那一步被消化了

### 5.5 Ablation 3: HM³ 各组件(Table 4)

| Short-term | Long-term | Result | Working | Controller Context only? | Video-MME Long |
|---|---|---|---|---|---|
| ✗ | ✓ | ✓ | ✓ | - | 72.5 |
| ✓ | ✗ | ✓ | ✓ | - | 67.0 |
| ✓ | ✓ | ✗ | ✓ | - | (无结果,反复循环) |
| ✓ | ✓ | ✓ | ✗ | - | 75.5 |
| ✗ | ✗ | ✗ | ✗ | ✓(纯 context) | 74.5 |
| ✓ | ✓ | ✓ | ✓ | - | **76.5** |

关键 takeaways:
- **Long-term pool 最关键**(-9.5 if removed) — 因为它定义了 agent 的 attention narrowing 能力
- **Result Memory 是必须** — 没有它 controller 无法决定下一步,导致循环不停
- **纯 context 也 only -2.0**,说明 HM³ 真正的好处不是"省 token",而是把 attention 集中到正确的信息上
- Working Memory 只贡献 +1.0,但这是 reasoning trace 的可审计性,本身有外推价值

### 5.6 Ablation 4: Step Budget N(Table 5)

| N | Short | Long | LongVideoBench |
|---|---|---|---|
| 3 | 87.5 | 72.0 | 67.5 |
| 5 | 87.0 | 74.5 | 67.5 |
| 7 | 85.5 | 75.5 | 69.5 |
| 10 | 84.0 | 76.5 | 70.5 |

观察:
- 长 video 上 step budget 多收益大,符合直觉
- **短 video 上 step 多反而略降**(87.5 → 84.0) — 这是过度探索引入 noise 的迹象,值得注意。论文作者没有强调这一点,但我觉得这是合理推论
- $N=10$ 是论文选的折中点,但短视频可以 adaptively 减小 N

### 5.7 Ablation 5: Adaptive Frame Sampling(Table 6)

| Sampling Strategy | Video-MME Long | LongVideoBench |
|---|---|---|
| Fixed $N_1 = 30$ | 73.5 | 68.0 |
| Fixed $N_1 = 60$ | 74.0 | 70.5 |
| **Adaptive $N_1$(avg 49.8)** | **76.5** | **70.5** |

这是另一个值得 build intuition 的点:**adaptive 采样用更少的 frames(avg 49.8 < 60)拿到更好的结果**,说明 video information density 在时间轴上是非均匀的,adaptive 能在 easy segment 节约 token、在 complex segment 多采样。这是非常标准的 RL exploration 的思路。

---

## 6. Limitations 我觉得值得展开

### 6.1 初始采样的瓶颈
作者在 Section C 承认:**整个 pipeline 的 quality 上限被初始 temporal/visual sampling 决定**。如果 query-relevant event 持续极短(例如 1 帧)、或 object 严重 occluded(如 Figure 5 中的 carousel),controller 无法启动有效的 reasoning trajectory。

这个问题本质上是 **agent loop 缺少 "重新采样" 的 explicit mechanism**。可能的解决方向:
- 让 controller 在 confidence $S_{sub}$ 低于阈值时 trigger 重采样
- 用 audio cue 作为 fallback signal 来 anchor temporal localization
- 用 dense optical flow 或 scene change detection作为 cheap pre-signal

### 6.2 闭源模型依赖
论文坦白 open-source 替代品(Qwen3-VL)在 jigsaw-style video interface 下表现差很多。这暗示了一个 **interface design 问题**:3×2 image grid 是为 GPT-4o 这类模型 tuned 的视觉 token layout,开源 VLM 在 dense grid 上 visual grounding 还没跟上。

这让我想到:[VideoAgent (ECCV 2024)](https://arxiv.org/abs/2403.16971) 也面临类似问题。如果想让 VideoARM 在开源模型上 work,可能需要重新设计 visual interface,例如改用 frame index embedding 而不是 overlay text,或者用 hierarchical token(类似 ViT 的 pyramid)。

### 6.3 Agent 不可微
整个 pipeline 是 API 调用串联起来的,梯度无法 backprop。这意味着:
- 不能 end-to-end 训练
- 不能 fine-tune 任何中间 tool
- 改进路径只能靠 prompt engineering 或替换更强 base model

这是当前所有 agentic 视频理解方法的共同局限,论文没讨论这点,但我觉得是真正的长期瓶颈。

---

## 7. 跟其他工作的关联联想

### 7.1 跟 ReAct 的关系
ReAct [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) 提出的是 thought-action-observation 循环,所有 history 都堆在 context 里。VideoARM 是 ReAct + 分层 memory 的变种 — 把 episodic memory 从 context 卸载到外部 store,context 只保留当前 working state。这个思路在 [LongAgent](https://arxiv.org/abs/2402.11550) 上有类似探索。

### 7.2 跟 Memory-based LLM 的关系
HM³ 让我想到 [Generative Agents (Stanford)](https://arxiv.org/abs/2304.03442) 的 memory stream + reflection + planning。Sensory = 观察,Result = memory stream,Working = plan + reflection。这是一个 cognitive-science-inspired 的 memory architecture。

### 7.3 跟 video pre-training 的关系
VideoARM 完全 training-free,所以理论上可以 plug 进任何 MLLM。但 ablation 表明 base model 的 reasoning 能力是关键。这暗示一种 **asymmetric scaling**: 让大模型当 controller 做高阶 reasoning,小模型当 worker 做廉价 vision task。这种分工让我想到 mixture-of-experts 的 agentic 版本。

### 7.4 跟 Memory Transformer / RAG 的关系
HM³ 是一种 **explicitly constructed structured RAG**。RAG 是从外部 corpus 检索,HM³ 是 agent 自己构造+查询。这种 "agent-built KB" 思路在 [VCA (ICCV 2025)](https://arxiv.org/abs/2505.01112) 也有体现。

### 7.5 跟 AlphaGo-style search 的关系
观察 → 假设 → 验证 → 收窄 的循环本质上是 **Monte Carlo Tree Search 的简化版**: 每次 Clip Analyzer 的 confidence $S_{sub}$ 类似 value function,但 VideoARM 没有显式 backpropagate value。如果引入 value-based pruning,可能进一步压缩 step budget。

---

## 8. 我的总评

### 强项
- **Token efficiency 1/34 是 game-changing** 的 — 这意味着同样预算能处理长 34 倍的视频,或者跑 34 倍的 queries。这对 production deployment 是关键
- **HM³ 的分层设计是 elegant 的** — 把 cognitive science 的 sensory/working/episodic memory 落地到工程上,而且 working memory externalization 解决了 context overflow
- **Controller 不 hardcode workflow** — 让 MLLM 自主调度,这契合目前 reasoning model(o3, Claude 3.5 sonnet thinking 等)的设计哲学
- **Adaptive frame sampling** 是个 small 但 clever 的设计,带来 +2.5% 提升

### 弱项 / 我会问作者的问题
1. **没有显式 uncertainty-aware stopping criterion** — 完全靠 controller 自己 decide when to ANSWER。引入 confidence threshold 会不会更稳?
2. **Clip Analyzer 的 confidence $S_{sub}$ 怎么用?** 论文没说 controller 是否消费这个信号做决策。这是一个被低估的信号。
3. **没有 cross-query 的 HM³ reuse** — 多个 query 来时,每次都重新 build HM³。能否共享 sensory memory?
4. **Audio 只用了 whisper-1 transcript** — 没用 audio feature、声纹、音乐情绪等非语音信号,这是个 modality 浪费
5. **3×2 grid 是 GPT-4o specific** — 跨模型 generalization 没做实验

### 我会建议的方向
- 引入 **probabilistic policy over tools**:用 RLHF 或 DPO fine-tune 一个小的 controller router 来预测 tool choice,可以减少对 o3 的依赖
- 把 **HM³ 形式化为 graph**:节点是 frame interval,边是 temporal/semantic relation。这样支持 multi-hop reasoning
- 引入 **multimodal fusion in HM³**:目前 result memory 是 text-only。能否存 visual feature vector 实现 visual retrieval?
- 测试 **chain-of-tools 的可并行性**:当前是 strictly sequential。Clip Explorer + Scene Snapper 是否能并行?

---

## 9. 参考链接

主要参考:
- VideoARM paper 本身(已 attach)
- [DVD (Deep Video Discovery)](https://arxiv.org/abs/2505.18079) — 直接 baseline
- [ReAct](https://arxiv.org/abs/2210.03629) — Reasoning + Acting 范式起源
- [VideoTree](https://arxiv.org/abs/2412.19407) — 另一个 hand-crafted hierarchical baseline
- [VideoAgent (ECCV 2024)](https://arxiv.org/abs/2403.16971) — Memory-augmented agent
- [Video-MME benchmark](https://arxiv.org/abs/2405.21075) — 主评测集之一
- [LongVideoBench](https://arxiv.org/abs/2407.15754) — 长视频评测集
- [EgoSchema](https://arxiv.org/abs/2308.09131) — egocentric 长视频评测
- [Whisper-1](https://arxiv.org/abs/2212.04356) — Audio Transcriber 用的 ASR
- [GPT-4.1 / GPT-4o](https://arxiv.org/abs/2303.08774) — Controller 和 worker model
- [Generative Agents (Stanford)](https://arxiv.org/abs/2304.03442) — Memory stream 思想渊源
- [LongAgent](https://arxiv.org/abs/2402.11550) — 多 agent 长 context 协作
- [VCA (Video Curious Agent)](https://arxiv.org/abs/2505.01112) — 另一个 video curious agent

---

**总结一句**:VideoARM 是把 long-form video 理解从"先建库再检索"转成"边推理边构造 memory"的范式转变,把 token cost 砍到 1/34,在主流长视频 benchmark 上同时达到 SOTA。HM³ 的分层 memory 设计是真正可迁移的 idea,可以 plug 进任何 agentic reasoning framework。最大的局限是依赖 o3 这种强 reasoning model,以及初始采样的瓶颈。

Andrej,如果你对其中任何一块(尤其是 controller 的 reasoning loop 设计、HM³ 的形式化、或者 token efficiency 的 theoretical bound)想深挖,告诉我具体方向。
