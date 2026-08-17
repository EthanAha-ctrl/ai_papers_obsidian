---
source_pdf: VimRAGNavigating Massive Visual Context in Retrieval-Augmented Generation
  via Multimodal Memory Graph.pdf
paper_sha256: c6c5ab8fea250ed659df201504d6264ac9b7cee2678d4485e118dcc2d88c3f9d
processed_at: '2026-08-13T01:15:33-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VimRAG 人话版

好，我换个讲法，用大白话把这篇 paper 重新捋一遍。

---

## 这篇 paper 到底在解决什么问题

你想象一个场景：你给一个 AI agent 一个问题，比如"那个在 YouTube 教微积分的老师，他在第 3 节课里用的那个积分技巧叫什么名字？"

这个问题 agent 没法直接答，得去搜。搜到视频，看视频，找到关键帧，再 reasoning。如果一次搜不到，还得换个 query 再搜。

现在主流做法（ReAct）是把每一步的 thought、action、observation 全部拼成一个长长的 sequence 喂给 LLM。这在 text-only 场景还行，但在 multimodal 场景会崩。为什么？因为一张图动不动就 1000+ vision tokens，一个视频 segment 更夸张。搜 5 轮下来，context 里塞了几万个 visual tokens，但真正有用的可能就几百个。LLM 的 attention 被一堆垃圾 token 稀释，critical information 淹没在噪声里。

另一拨人（Mem1）想了个办法：每搜一轮就把结果压缩成一个 memory summary，只保留 text。这样 token 省了，但问题是——你把图片压成文字描述后，到 verification 阶段你想回去看原图的细节，发现没了。而且更严重的是，agent 忘了自己搜过什么，会反复搜同一个 query，陷入死循环。

VimRAG 说：这两个极端都不对。我们搞一个**中间方案**——把 reasoning 过程记成一个 graph（有向无环图），每个 node 记住"我搜了什么 query、拿到了什么 observation、提炼了什么 memory"。这样 agent 既不会迷路（graph 记录了探索路径），也不会被 visual garbage 淹没（memory 是 selective 的）。

---

## 三个核心 insight，用大白话讲

### Insight 1：Agent 需要知道"自己走过哪些路"

ReAct 的问题是把所有 history 平铺，agent 不知道哪些 action 是有用的、哪些是 dead end。Mem1 的问题是把 history 压成一个 summary，agent 完全丢失了"我搜过什么 query"这个信息。

VimRAG 的做法是建一个 graph。每个 node 存四个东西：
- 我是从哪些 parent node 来的（$p_i$）
- 我这个 node 对应的 search query 是什么（$q_i$）
- 我从 observation 里提炼的 text summary（$s_i$）
- 我保留的 visual memory tokens（$m_i$）

这样 agent 看 graph 就知道："哦，我已经搜过 query A 和 query B 了，A 没找到有用的东西（dead end），B 找到了关键信息，我现在应该基于 B 往下搜。"

这跟人查资料的方式很像。你查 Wikipedia 的时候，脑子里不是把所有读过的文字串成一条线，而是有一个"知识地图"——哪些页面看过了、哪些有用、哪些没用、它们之间什么关系。

### Insight 2：Visual memory 不能全留也不能全压

paper 里 Table 1 的实验特别说明问题。他们试了四种策略：

1. **全压成 text**（Pre-Caption）：token 最省，但 accuracy 只有 14.5%（image）/17.2%（video），因为 visual detail 丢了
2. **全留 raw visual tokens**：accuracy 45.6%/30.4%，但 token 消耗 15.8k，signal-to-noise ratio 很差
3. **retrieve 后压成 text**（Context-Aware Caption）：52.8%/39.5%，比 pre-caption 好，但还是丢了 visual detail
4. **selectively 保留关键 visual region**：58.2%/43.7%，accuracy 最高，token 只用 2.7k

结论很明确：**该留的 visual detail 要留，该扔的 noise 要扔**。但问题是——怎么判断哪些该留？

VimRAG 的答案是：用 graph topology 来判断。一个 visual item 有多重要，取决于三件事：
- 它本身有多 query-relevant（semantic score $\hat{p}$）
- 它所在的 node 被多少后续 node 依赖（out-degree $\deg^+$）
- 它被后续高价值 node reinforce 了多少（recursive feedback）

这三个信号合成一个 "energy" 值，energy 高的 visual item 拿到更多 token budget（更高 resolution），energy 低的被压缩或丢弃。

用大白话说：**重要的证据给高清，不重要的证据给马赛克或者直接扔掉**。

### Insight 3：Reward 不能一刀切

这是 RL training 的问题。现在主流做法是 outcome-based reward——答对了 $r=1$，答错了 $r=0$，然后把这个 reward 广播到 trajectory 里每一步。

但这有个大问题。假设 agent 搜了 5 轮，前 3 轮搜的都是没用的东西，第 4 轮搜到了关键信息，第 5 轮答对了。outcome reward 是 $r=1$。如果你把这个 reward 广播到每一步，那前 3 轮的垃圾 search 也会拿到 positive gradient。model 就会学到："搜一些没用的东西也没关系，反正最后答对了就有 reward。"

反过来，假设 agent 第 1 轮就搜到了关键信息，但最后 reasoning 阶段搞错了，答错了，$r=0$。如果你把这个 penalty 广播到每一步，第 1 轮那个好的 search 也会被惩罚。model 就会学到："好的 search 也被惩罚了，那我干脆别搜了。"

paper 的 Figure 3 实验验证了这一点。他们做了 counterfactual ablation：
- 负样本里，把 noise step 去掉，performance 能 recover → 说明失败不是因为没搜到东西，而是被 noise 干扰了
- 正样本里，把 evidence step 去掉，performance 不掉到 0 → 说明 model 部分依赖自己的 parametric knowledge

VimRAG 的解法是用 graph 来做 fine-grained credit assignment：
- 正样本里，找到从 root 到 answer 的 critical path，不在 path 上的 node 是 dead end，mask 掉不给 gradient
- 负样本里，找到那些 retrieval result 确实包含 relevant info 的 step，这些 step 不惩罚

这样 gradient signal 就精准了：好的 action 被 reward，坏的 action 被 penalty，ambiguous 的 action 不参与 update。

---

## 整个系统怎么跑的

我用一个具体例子串一遍。

假设 query 是："那个在 YouTube 教微积分的老师，第 3 节课里用的积分技巧叫什么？"

**Step 1**：Agent 看到这个 query，生成 thought："我需要找到这个 YouTube 频道和它的第 3 节课。" Action 是 $a^{ret}$，query 是 "calculus tutorial YouTube channel lecture 3"。创建 node $v_1$，parent 是 root。

**Step 2**：Search engine 返回一堆视频。Agent 调用 $a^{mem}$，对每个返回的 video item 做评估：
- Video 1：saliency mask $u=1$（有用），priority score $p=4$（比较重要）
- Video 2：saliency mask $u=0$（没用），$p=1$
- Video 3：$u=1$，$p=5$（关键！这个就是第 3 节课）

把 Video 1 和 Video 3 的 keyframe extract 出来，存成 visual memory tokens。生成 text summary $s_1$："找到 calculus tutorial 频道，第 3 节课讲的是 integration by parts。"

**Step 3**：Agent 看 graph，发现 $v_1$ 的 summary 提到了 integration by parts，但还没确认具体技巧名。生成新 thought："需要确认这节课具体讲了哪个技巧。" Action $a^{ret}$，query "integration by parts technique name"，parent 是 $v_1$。创建 $v_2$。

**Step 4**：Search 返回一些 text + image。Agent 做 perception，发现其中一个 image 是一个公式截图，$p=5$。存成 visual memory。

**Step 5**：Agent 判断 evidence 够了，执行 $a^{ans}$。从 root 到 $v_1$ 到 $v_2$ 到 answer node 构成 critical path。返回答案。

**Memory Encoding 阶段**（每步都在做）：
- $v_1$ 的 visual memory（Video 1 和 Video 3 的 keyframe）算 energy：
  - Intrinsic energy：$\hat{p} \times (1 + \deg^+(v_1))$。$v_1$ 被 $v_2$ 依赖，$\deg^+(v_1) = 1$。Video 3 的 $\hat{p}$ 高（$p=5$ 归一化到 $[0,1]$），所以 intrinsic energy 高。
  - Recursive reinforcement：$v_2$ 的 energy 反馈回来加到 $v_1$ 上。$v_2$ 是高价值 node（包含关键公式图），所以 $v_1$ 的 final energy $\Omega$ 被进一步推高。
  - 结果：Video 3 的 keyframe 拿到高 token budget（高 resolution），Video 1 拿到中等 budget，Video 2 被丢弃。

**Training 阶段**：
- 假设这个 trajectory 最终答对了，$r=1$。
- 从 answer node 反向遍历，critical path 是 $\{root, v_1, v_2, answer\}$。
- 假设 agent 还搜过一个 $v_3$（query "calculus basics"），但 $v_3$ 不在 critical path 上（dead end）。
- Pruning mask：$v_3$ 被 mask 掉（$\mu=1$），不参与 gradient update。$v_1, v_2$ 正常 update。
- 这样 model 就不会学到"搜 calculus basics 这种没用的 query 也会被 reward"。

---

## 几个关键公式的直觉

### Energy 公式（Eq. 6-7）

Intrinsic energy：
$$\mathcal{E}_{int}(m_{i,k}) = \hat{p}_{i,k} \cdot (1 + \deg^+_{\mathcal{G}}(v_i)) \cdot \exp(-\lambda(T - t_i))$$

直觉解读：
- $\hat{p}_{i,k}$：这个 visual item 本身有多 query-relevant。范围 $[0,1]$，越高越重要。
- $(1 + \deg^+)$：这个 node 被多少 child 依赖。$\deg^+ = 0$ 说明没有后续 node 依赖它（dead end），$(1+0)=1$，energy 不 boost。$\deg^+ = 3$ 说明 3 个后续 node 依赖它，$(1+3)=4$，energy 被 boost 4 倍。
- $\exp(-\lambda(T-t_i))$：时间衰减。$\lambda=0.1$，如果 node 是 10 步前创建的，衰减因子是 $e^{-1} \approx 0.37$，energy 打 37 折。这模拟人类的遗忘曲线。

Recursive reinforcement：
$$\Omega(m_{i,k}) = \mathcal{E}_{int}(m_{i,k}) + \gamma \sum_{v_j \in Child(v_i)} \overline{\Omega}(v_j)$$

- $\gamma = 0.3$：feedback discount。类似 RL 里的 discount factor $\gamma$。
- 这个公式说的是：一个 node 的最终价值 = 它本身的价值 + 它 lead 到的所有 child node 的价值（打个折）。

这跟 RL 里的 value function $V(s) = r + \gamma V(s')$ 是同一个结构。early node 的价值通过它 lead 到的后续 node 来体现。

### Token Allocation（Eq. 8）

$$b_{i,k} = \left\lfloor S_{total} \cdot \frac{\Omega(m_{i,k})}{\sum_{m'} \Omega(m')} \right\rfloor$$

这就是 softmax-style 分配。总 budget $S_{total}$ 按 energy 比例分给各个 visual item。energy 高的拿大头，energy 低的拿小头或被丢弃。

$S_{total} = 5 \times 256 \times 32 \times 32$：5 个 high-resolution patch，每个 patch 是 $256 \times 32 \times 32$ tokens（256 个 token，空间分辨率 $32 \times 32$）。这是 Qwen3-VL 的 high-resolution vision token 规格。

### Pruning Mask（Eq. 11）

$$\mu_t = \mathbb{I}(r=1) \cdot \mathbb{I}(v_t \notin \mathcal{P}_{ans}) + \mathbb{I}(r=0) \cdot \mathbb{I}(v_t \in \mathcal{R}_{val})$$

两项分别处理两种情况：
- 第一项：正样本里，不在 critical path 上的 node 被 mask（dead end 不给 reward）
- 第二项：负样本里，有 valuable retrieval 的 node 被 mask（好 retrieval 不被惩罚）

$\mu_t = 1$ 的 step 被排除出 gradient update。$\mu_t = 0$ 的 step 正常 update。

### Optimization（Eq. 12）

$$\max_{\pi_\theta} \mathbb{E}\left[\frac{1}{\sum n_g} \sum (1-\mu_{g,i}) \cdot \min(r_{g,i}(\theta)\hat{A}_{g,i}, \text{clip}(r_{g,i}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_{g,i})\right]$$

就是标准 PPO clipped objective，加了一个 mask $(1-\mu_{g,i})$。被 mask 的 step 的 loss 是 0，不产生 gradient。没被 mask 的 step 正常按 PPO 更新。

---

## 这篇 paper 真正的价值在哪

我觉得有三层价值：

**第一层（表层）**：在 multimodal RAG benchmark 上刷了 SOTA。这个本身不算什么，benchmark 数字大家都能刷。

**第二层（机制层）**：提出了"用 graph 结构来组织 agent memory"这个 idea。这比 ReAct 的 flat sequence 和 Mem1 的 single memory state 都更 expressive。graph 天然支持 branching（一个 node 可以有多个 child）、backtracking（dead end 不在 critical path 上）、merging（一个 node 可以有多个 parent）。这些是 linear/summary memory 做不到的。

**第三层（概念层）**：这篇 paper 其实是在说一个更大的 story——**agent reasoning 应该有结构，memory 应该服务于 future behavior 而不是 past storage，supervision 应该是 process-aware 而不是 outcome-only**。这三个 principle 其实是通用的，不只适用于 multimodal RAG。

---

## 跟其他工作的关系，用大白话

- **ReAct**：把 reasoning 当成一条线。VimRAG 说应该是 graph。
- **Mem1 / Mem0**：把 memory 当成一个压缩的 summary。VimRAG 说应该保留结构。
- **GraphRAG**：graph 是预构建的 knowledge graph（实体-关系图）。VimRAG 的 graph 是 agent 自己 build 的 reasoning trace graph（action-dependency 图）。完全不同的 abstraction level。
- **Tree of Thoughts**：也是 graph/tree 结构的 reasoning，但 ToT 是 explicit search（BFS/DFS），VimRAG 是 implicit memory（agent 自己决定怎么 expand graph）。
- **ColPali**：在 retrieval 层面保留 visual token 的细粒度。VimRAG 在 memory 层面做类似的事。
- **PPO**：标准 RL training。VimRAG 加了 graph-guided pruning mask 来做 fine-grained credit assignment。
- **Process Reward Model**（PRM）：对 reasoning 每步打分。VimRAG 用 graph topology 代替 PRM，不需要额外的 reward model。
- **Attention Sink / H2O**：KV cache compression。跟 VimRAG 的 visual token allocation 是同类问题，但 VimRAG 基于 reasoning context 而不是 attention pattern。

---

## 我的几个直觉判断

1. **Graph memory 是对的方向**。Linear history 在长 horizon agent 任务里一定会崩。graph 提供了结构先验，让 agent 能区分"我在哪"、"我从哪来"、"我要去哪"。

2. **Energy-based token allocation 的 idea 很好，但 energy function 太 hand-crafted**。Eq. 6 的 temporal decay、out-degree weighting 这些都是人为设计的。未来如果能 learn 这个 energy function（比如用 contrastive learning 或 inverse RL），效果可能更好。

3. **Graph pruning 做 credit assignment 是个 clever trick**，但依赖 critical path traversal 的准确性。如果 agent 的 graph 有很多 branch，critical path 很难准确定义。这跟 PRM 的问题类似——step-level supervision 的质量决定了 training 的效果。

4. **这篇 paper 的 RL 部分相对 lightweight**。本质上就是在 PPO 上加了个 mask，没有 value network、没有 reward shaping、没有 curriculum。这可能限制了方法的上限。但作为 first attempt，用 graph structure 来做 credit assignment 这个 idea 本身是有价值的。

5. **真正的 bottleneck 可能不在 memory structure，而在 retriever**。paper 的 limitation 里也提到了。如果 retriever 不好，agent 搜不到好东西，再好的 memory structure 也白搭。这跟 RAG 领域的普遍观察一致——retrieval quality 是 RAG 性能的天花板。

---

**参考链接**：
- [VimRAG Paper](https://arxiv.org/abs/2508.05748)
- [VimRAG GitHub](https://github.com/Alibaba-NLP/VRAG)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Mem1](https://arxiv.org/abs/2506.15841)
- [GraphRAG](https://arxiv.org/abs/2404.16130)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [ColPali](https://arxiv.org/abs/2407.01449)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Process Reward Models](https://arxiv.org/abs/2406.06592)

---

# VimRAG 技术深度解析

Karpathy，这篇 paper 我读完之后直觉上觉得它做的事情其实是在 agent reasoning 这个层面引入了"结构化先验"，把 ReAct 那种扁平的 thought-action-observation 流变成了一个真正有 topology 的 reasoning DAG。下面我把核心机制拆开讲，尽量把每个公式、每个设计决策的 intuition 都讲透。

---

## 1. 问题动机：为什么 linear history 在 multimodal RAG 里会崩

paper 的 Pilot Study 部分做了三个非常关键的实验，这是整篇文章的立论基础。

### 1.1 History-Accumulating Paradigm 的崩溃

标准 ReAct 的 context 是这样累积的（Eq. 1）：

$$\mathcal{H}_t = [q, \tau_1, a_1, o_1, \dots, \tau_{t-1}, a_{t-1}, o_{t-1}]$$

变量含义：
- $\mathcal{H}_t$：step $t$ 时的完整 history
- $q$：原始 user query
- $\tau_i$：第 $i$ 步的 thought（reasoning text）
- $a_i$：第 $i$ 步的 action（这里 action space 是 $\{a^{ret}, a^{mem}, a^{ans}\}$，即 retrieve / memorize / answer）
- $o_i$：第 $i$ 步的 observation（retrieval 返回的 multimodal 数据）

关键问题在于信息密度：

$$\frac{|\mathcal{O}_{crit}|}{|\mathcal{H}_t|} \ll \epsilon$$

其中 $\mathcal{O}_{crit}$ 是真正对当前 query 关键的 observation，$\epsilon$ 是一个很小的阈值。随着 $t$ 增大，critical information 被稀释在大量 token-heavy 的 visual observation 里。这在 text-only RAG 里还好（文本本身信息密度高），但 visual data 是 "token-heavy yet semantically sparse"——一张图可能消耗 1000+ vision tokens，但只有 10 个 token 是 query-relevant 的。

**Intuition**：这其实是 attention dilution 问题。softmax attention 在长 context 下会被无关 token 拉走注意力，类似于 "lost in the middle" 现象（[Liu et al. 2023](https://arxiv.org/abs/2307.03172)），但这里更严重，因为 visual tokens 的信息密度天然比 text 低。

### 1.2 Memory-Based Paradigm 的 Markovian Blindness

Mem1 这类方法（[Zhou et al. 2025](https://arxiv.org/abs/2506.15841)）的做法是压缩 history 成 memory state $m_t$（Eq. 2）：

$$m_{t-1} \xrightarrow{\pi_\theta(\cdot)} (\tau_t, a_t) \xrightarrow{Env} o_t \xrightarrow{\pi_\theta(\cdot | \tau_t, a_t, m_{t-1})} m_t$$

这里信息密度保持稳定 $|\mathcal{O}_{crit}|/|m_t| \approx C$，但引入了 **Markovian blindness**：agent 只看到当前 memory state $m_t$，不知道自己历史上做过哪些 retrieval action。在 multi-hop 场景下，这导致 agent 反复 query 同样的东西（Figure 2b 显示 invalid retrieval count 显著上升）。

**Intuition**：这跟 RL 里的 partial observability 是同一个问题。$m_t$ 是一个 lossy sufficient statistic，但在 agentic reasoning 这种需要"知道自己探过哪条路"的任务里，memory state 不够 expressive。

### 1.3 Visual Memory 的模态困境

Table 1 的四组对比非常 informative：

| Strategy | Retrieval→Memory | Avg Tokens | Image Acc | Video Acc |
|----------|------------------|-----------|-----------|-----------|
| Pre-Caption | Text→Text | 0.9k | 14.5% | 17.2% |
| Raw Visual Tokens | Vision→Vision | 15.8k | 45.6% | 30.4% |
| Context-Aware Caption | Vision→Text | 1.5k | 52.8% | 39.5% |
| Semantically-Related | Vision→Selective Vision | 2.7k | **58.2%** | **43.7%** |

关键 insight：把 vision 压成 text（strategy 3）虽然 token 省了，但 verification 阶段会丢掉 fine-grained visual detail，这就是 "semantic gap"。而 strategy 4 的 selective vision 用 2.7k tokens 达到了最好的 accuracy，说明**不是要不要存 vision，而是要 selectively 存 critical vision region**。

**Intuition**：这跟 [ColPali](https://arxiv.org/abs/2407.01449) 的 late interaction 思路有点像——不要过早把 visual info 压成 text embedding，而是保留 visual token-level 的细粒度，让 LLM 在 reasoning 时直接 attend 到具体的 visual patch。VimRAG 的做法是在 memory 层面做这个 selection，而不是在 retrieval 层面。

### 1.4 Sparse Reward 的 Credit Assignment 问题

Figure 3 的 counterfactual ablation 实验是这篇文章最 clever 的实验设计之一。他们把 trajectory $\tau = \{s_1, \dots, s_T\}$ 分成两个 disjoint subset：
- $S_{evd}$：evidence retrieval steps
- $S_{noise}$：noise/redundant steps

然后做 counterfactual：
- 正样本 $r=1$：去掉 evidence（$\tau \setminus S_{evd}$）看是否还能答对
- 负样本 $r=0$：只保留 evidence（$\hat{\tau} = S_{evd}$）看是否能 recover

结果（Figure 3b）：
- 负样本去掉 noise 后 performance recover → 失败是因为 reasoning over noise，不是缺 evidence
- 正样本去掉 evidence 后 non-zero performance → model 部分依赖 parametric knowledge

**Intuition**：这直接说明 outcome-based reward $r \in \{0,1\}$ 是一个极度 coarsened signal。把 $r=1$ 广播到 trajectory 里每一步，会让 noise step 也拿到 positive gradient（false positive）；把 $r=0$ 广播到每一步，会惩罚真正有价值的 retrieval（false negative）。这是 RL 里经典的 credit assignment problem，但在 agentic reasoning 里因为 step 之间有强 logical dependency 而更严重。

---

## 2. Multimodal Memory Graph：把 reasoning 建模成 DAG

这是 VimRAG 的第一个核心创新。他们把 reasoning process 形式化为一个 dynamic DAG：

$$\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$$

其中 $\mathcal{V}_t$ 是 node 集合，$\mathcal{E}_t$ 是 directed edge 集合，$t$ 是 reasoning step。

### 2.1 Node 定义

每个 node $v_i$ 是一个 epistemic state（Eq. 3）：

$$v_i \triangleq (p_i, q_i, s_i, m_i)$$

变量：
- $p_i$：parent node indices 集合，编码 local dependency structure
- $q_i$：decomposed sub-query（这个 node 对应的 search query）
- $s_i$：concise textual summary（从 observation 提炼的 text）
- $m_i$：multimodal episodic memory bank（visual tokens from retrieved docs/frames）

Edge set $\mathcal{E}_t = \{(v_j, v_i) | j < i\}$ 自然地编码了 reasoning flow。

**Intuition**：这个设计的关键在于 $p_i$ 是一个 set，不是单个 parent。这意味着 node 可以有多个 parent，形成真正的 DAG 而不是 tree。比如一个 sub-query 可能依赖于两个之前的 retrieval 结果，这时候 $p_i = \{j_1, j_2\}$。这比 Mem1 的线性 memory 要 expressive 得多，也比 ReAct 的 flat history 更 structured。

### 2.2 Graph Evolution as POMDP

他们把 graph construction 形式化为 POMDP（Eq. 4）：

$$a_t \sim \pi_\theta(\cdot | \mathcal{G}_{t-1}), \quad \mathcal{G}_t \leftarrow \Psi(\mathcal{G}_{t-1}, a_t)$$

- $a_t$：从 policy $\pi_\theta$ 采样的 action
- $\Psi$：environment operator（执行 search、返回 observation）

Action space 是 $\{a^{ret}, a^{mem}, a^{ans}\}$，分别对应三个 phase：

**Exploratory Expansion ($a^{ret}$)**：当 evidence 不足时，agent 创建 skeletal node $v_t' = (p_t, q_t, \emptyset, \emptyset)$，执行 query $q_t$ 检索 raw multimodal observation $\mathcal{O}_t$。

**Multimodal Perception & Memory Populating ($a^{mem}$)**：拿到 $\mathcal{O}_t$ 后，policy 调用 perception action 把 high-entropy information 蒸馏成 structured memory：

$$\mathcal{O}_t \rightarrow (s_t, m_t)$$

这里用 coarse-to-fine filtering：对每个 retrieved item 生成 binary saliency mask $u \in \{0,1\}$ 和 fine-grained semantic score $p \in [1,5]$。最终 finalize node $v_t = (p_t, q_t, s_t, m_t)$。

**Terminal Projection ($a^{ans}$)**：reasoning 路径足够时，执行 answer action。从 $v_{root}$ 到 $v_{ans}$ 的路径构成 critical logical path。

### 2.3 Video 的 Temporal Grounding

对 video observation（Eq. 5）：

$$\mathcal{O}_t^{video} = [(ts_k, f_k)]_{k=1}^{n}$$

- $ts_k$：第 $k$ 帧的 timestamp（格式 `<%0.1f seconds>`）
- $f_k$：第 $k$ 帧 image
- $n$：帧数

通过 $a^{mem}$ action，raw stream 被蒸馏成 $(s_t, m_t)$。这里利用 Qwen3-VL 的 temporal grounding 能力来 extract keyframes。

**Intuition**：这个 video 处理方式跟 [VideoChat-Flash](https://arxiv.org/abs/2501.00574) 的 hierarchical compression 思路类似，但 VimRAG 的区别是压缩后的 keyframe 会被存进 memory graph node，而不是直接进 context。这意味着后续 step 可以通过 graph topology 回溯到这些 keyframe，而不是在 linear context 里搜索。

### 2.4 Algorithm 1 解析

整个 inference pipeline（Algorithm 1）的循环逻辑：

1. **Context shaping**：把 graph $\mathcal{G}_t$ linearize 成 $\mathcal{H}_t$（通过 `LinearizeGraph`），喂给 policy 生成 $a_t$
2. **Topological expansion**：根据 $a_t$ 类型执行不同操作
   - $a^{ret}$：创建 node，search，populate memory
   - $a^{ans}$：连接 terminal node，返回 answer
3. **Dynamic visual memory shaping**：对每个 visual node 计算 energy，分配 token budget，压缩 memory

**Intuition**：这里有个很微妙的设计——graph 被 linearize 成 text 喂给 LLM。这是因为 LLM 本质上是 sequence model，不能直接 consume graph structure。所以 graph 的 topology 信息被编码成 linearized text（类似 graph serialization，参考 [GraphRAG](https://arxiv.org/abs/2404.16130)）。但跟 GraphRAG 不同的是，VimRAG 的 graph 是 agent 自己 build 的 reasoning trace，不是预构建的 knowledge graph。

---

## 3. Graph-Modulated Visual Memory Encoding：Energy-Based Token Allocation

这是第二个核心创新，解决 "visual memory resolution dilemma"。

### 3.1 问题形式化

给定 graph $\mathcal{G}$ 中的 node $v_i$，其 memory bank $\mathcal{M}_i = \{m_{i,k}\}_{k=1}^{K}$ 包含 $K$ 个 retrieved visual items。问题是如何在 token budget 约束下，给每个 $m_{i,k}$ 分配合适的 resolution（即 vision token 数量）。

### 3.2 Intrinsic Energy（Eq. 6）

$$\mathcal{E}_{int}(m_{i,k}) = \underbrace{\hat{p}_{i,k} \cdot (1 + \deg_{\mathcal{G}}^+(v_i))}_{\text{Structural-Semantic Relevance}} \cdot \underbrace{\exp(-\lambda(T - t_i))}_{\text{Temporal Decay}}$$

变量：
- $\hat{p}_{i,k} \in [0,1]$：normalized semantic priority（来自 perception phase 的 score $p \in [1,5]$）
- $\deg_{\mathcal{G}}^+(v_i)$：node $v_i$ 的 out-degree（在 graph 中有多少 child node 依赖于它）
- $T$：当前 step
- $t_i$：node $v_i$ 创建的 step
- $\lambda$：temporal decay 系数（paper 里 $\lambda = 0.1$）

**Intuition**：
- $\hat{p}_{i,k}$ 是 semantic relevance，query-relevant 的 visual item 得到更高 energy
- $\deg_{\mathcal{G}}^+(v_i)$ 是 topological centrality 的 proxy。如果一个 node 被很多后续 node 依赖（out-degree 高），说明它是 reasoning 的关键 hub，应该保留高 resolution。这跟 PageRank 的 intuition 类似——重要性的 node 是被很多重要 node 指向的 node
- Temporal decay $\exp(-\lambda(T-t_i))$ 模拟人类遗忘：越早的 memory 衰减越多。这跟 [Mem0](https://arxiv.org/abs/2504.19413) 的 recency weighting 思路一致

### 3.3 Recursive Reinforcement（Eq. 7）

$$\Omega(m_{i,k}) = \mathcal{E}_{int}(m_{i,k}) + \gamma \sum_{v_j \in \text{Child}(v_i)} \overline{\Omega}(v_j)$$

变量：
- $\Omega(m_{i,k})$：final energy
- $\gamma$：feedback strength（paper 里 $\gamma = 0.3$）
- $\text{Child}(v_i)$：$v_i$ 的 child node 集合
- $\overline{\Omega}(v_j)$：child node $v_j$ 的 average energy

**Intuition**：这是 backward pass，类似于 RL 里的 value backup 或 GNN 里的 message passing。直觉是：一个 early node 本身可能 semantic score 不高（$\hat{p}$ 低），但如果它 lead 到了高价值的后续 node，那它作为 "bridge" 的价值应该被 reinforce。这解决了 credit assignment 的 temporal aspect——early evidence 的价值往往在后续 reasoning 中才显现。

这跟 RL 里的 Monte Carlo return 或 TD backup 的结构很像：

$$V(s_t) = r_t + \gamma V(s_{t+1})$$

只不过这里的 "reward" 是 intrinsic energy，"value" 是 final energy $\Omega$。

### 3.4 Token Budget Allocation（Eq. 8）

$$b_{i,k} = \left\lfloor S_{total} \cdot \frac{\Omega(m_{i,k})}{\sum_{m' \in \mathcal{M}_{top}} \Omega(m')} \right\rfloor$$

变量：
- $b_{i,k}$：分配给 visual item $m_{i,k}$ 的 token budget
- $S_{total}$：总 token budget（paper 里 $S_{total} = 5 \times 256 \times 32 \times 32$，这是 5 个 high-resolution patch，每个 $256 \times 32 \times 32$ tokens）
- $\mathcal{M}_{top}$：基于 energy ranking 保留的 top-K items 集合

**Intuition**：这是 softmax-style 的 resource allocation。高 energy 的 item 拿到更多 token（更高 resolution），低 energy 的 item 拿到更少甚至被丢弃。这跟 attention mechanism 的 intuition 一致——重要信息应该获得更多 "compute budget"。

具体实现上，$b_{i,k}$ 决定了 ViT encoder 对该 visual item 的处理 resolution。高 $b_{i,k}$ → 高分辨率 patch → 保留 fine-grained detail；低 $b_{i,k}$ → 低分辨率或直接 discard。

**相关联想**：这个 mechanism 让我想到 [Token Merging](https://arxiv.org/abs/2210.09461)（ToMe）和 [FastV](https://arxiv.org/abs/2403.06764)，它们都是在 vision-language model 里做 token pruning。但 VimRAG 的区别是 pruning 决策基于 graph topology，而不是单纯基于 attention score。这使得 pruning 决策考虑了 reasoning context，而不仅仅是 visual saliency。

---

## 4. Graph-Guided Policy Optimization：Credit Assignment via Pruning

这是第三个核心创新，也是 RL training 的关键。

### 4.1 Trajectory Segmentation

首先定义 step $t$ 的 prompt（Eq. 9）：

$$\mathcal{C}_t = \{inst, q, \mathcal{L}(\mathcal{G}_t)\}$$

- $inst$：system instruction
- $q$：user query
- $\mathcal{L}(\mathcal{G}_t)$：linearized memory graph

每个 node construction unit（Eq. 10）：

$$\mathcal{H}^{(t)} = (\mathcal{C}_t, \tau_t, a_t^{ret}, o_t, \tau_t', a_t^{mem}) \rightarrow v_t$$

- $\tau_t$：lead to retrieval action 的 reasoning
- $\tau_t'$：synthesize memory action 的 reflection
- $a_t^{ret}$：retrieval action
- $a_t^{mem}$：memory action
- $o_t$：observation

Terminal block：$\mathcal{H}^{(T)} = (\mathcal{C}_T, \tau_{ans}, a^{ans})$

**Intuition**：这里把一个完整的 trajectory 分割成 atomic reasoning cycles，每个 cycle 对应一个 graph node 的构建。这为后续 step-level credit assignment 提供了 granularity。

### 4.2 Graph Pruning for Credit Assignment

核心思想：用 graph topology 来识别哪些 step 应该被 mask 掉，不参与 gradient update。

**Pruning False Positives（Dead-End States）**：
给定正样本 $(\mathcal{T}, r=1)$，从 answer node 反向遍历找到 critical path $\mathcal{P}_{ans} \subseteq \mathcal{G}$。不在 $\mathcal{P}_{ans}$ 上的 node $v \notin \mathcal{P}_{ans}$ 是 dead end（redundant exploration 或逻辑无关的分支），应该被 mask。

**Pruning False Negatives（Valuable Retrieval）**：
给定负样本 $(\mathcal{T}, r=0)$，用 reference annotation 识别哪些 step 的 retrieval result 包含 relevant information。这些 valuable retrieval action 应该从 negative gradient 中排除。

### 4.3 Pruning Mask（Eq. 11）

$$\mu_t = \underbrace{\mathbb{I}(r=1) \cdot \mathbb{I}(v_t \notin \mathcal{P}_{ans})}_{\text{Dead-Ends in Positive}} + \underbrace{\mathbb{I}(r=0) \cdot \mathbb{I}(v_t \in \mathcal{R}_{val})}_{\text{Valuable Retrieval in Negative}}$$

变量：
- $\mu_t$：binary mask，$\mu_t = 1$ 表示该 step 被排除出 gradient update
- $\mathbb{I}(\cdot)$：indicator function
- $r$：trajectory-level reward（0 或 1）
- $\mathcal{P}_{ans}$：正样本中的 critical path nodes
- $\mathcal{R}_{val}$：负样本中的 valuable retrieval nodes

**Intuition**：
- 第一项：正样本里的 dead-end step 不应该被 reward（否则 model 会学到 "走弯路也没关系"）
- 第二项：负样本里的 valuable retrieval 不应该被惩罚（否则 model 会学到 "好的 retrieval 也被惩罚了，那我就不 retrieve 了"）

这本质上是把 coarse outcome reward $r$ refine 成 step-level reward signal，类似于 **reward shaping** 或 **hierarchical credit assignment**。

### 4.4 Optimization Objective（Eq. 12）

$$\max_{\pi_\theta} \mathbb{E}_{q \sim \mathcal{D}, \{\mathcal{H}_g^{(i)}\} \sim \pi_\theta} \left[ \frac{1}{\sum_g n_g} \sum_{g=1}^{G} \sum_{i=1}^{n_g} (1-\mu_{g,i}) \cdot \min\left(r_{g,i}(\theta)\hat{A}_{g,i}, \text{clip}(r_{g,i}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_{g,i}\right) \right]$$

变量：
- $q \sim \mathcal{D}$：从 training distribution 采样 query
- $\{\mathcal{H}_g^{(i)}\}_{i=1}^{n_g}$：第 $g$ 个 rollout 的 $n_g$ 个 segments
- $\mu_{g,i}$：segment $i$ 的 pruning mask
- $r_{g,i}(\theta) = \pi_\theta(\mathcal{H}_g^{(i)}) / \pi_{\theta_{old}}(\mathcal{H}_g^{(i)})$：probability ratio（PPO 的 importance sampling ratio）
- $\hat{A}_{g,i}$：advantage estimate
- $\varepsilon$：clip range（PPO 的标准 hyperparameter）
- $\min(\cdot, \text{clip}(\cdot))$：standard PPO clipped objective

**Intuition**：这就是 PPO objective 加了一个 mask $(1-\mu_{g,i})$。被 mask 的 step 不参与 gradient 计算，相当于在 trajectory 里 "skip" 掉那些 ambiguous 的 step。这跟 [GSPO](https://arxiv.org/abs/2507.11071)（Group Sequence Policy Optimization）的 segment-level optimization 思路一致，但加了 graph-guided masking。

**相关联想**：这个 masking 思路让我想到 [Offline RL](https://arxiv.org/abs/2006.04779) 里的 behavior cloning with filtering——只从 "good" trajectories 里学习，跳过 "bad" actions。VimRAG 的创新在于用 graph topology 来定义 "good" / "bad"，而不是用 reward signal。

---

## 5. 实验结果深度分析

### 5.1 Main Results（Table 2）

在 Qwen3-VL-8B-Instruct 上：

| Method | HotpotQA | SQuAD | WebQA | SlideVQA | MMLongBench | LVBench | WikiHowQA | SyntheticQA | XVBench | Overall |
|--------|----------|-------|-------|----------|-------------|---------|-----------|-------------|---------|---------|
| Vanilla RAG | 64.0 | 64.2 | 48.1 | 16.2 | 14.8 | 15.7 | 37.0 | 29.7 | 27.2 | 37.6 |
| ReAct | 70.8 | 65.5 | 40.0 | 15.4 | 15.9 | 23.0 | 35.0 | 24.0 | 21.3 | 37.7 |
| MemAgent | 71.1 | 74.8 | 47.1 | 35.5 | 45.3 | 22.2 | 23.1 | 37.5 | 26.9 | 40.3 |
| Mem1 | 73.0 | 68.4 | 44.5 | 55.7 | 22.4 | 24.5 | 19.9 | 43.4 | 32.2 | 43.6 |
| **VimRAG** | **79.1** | **76.4** | **53.9** | **62.4** | **33.4** | **29.7** | **54.5** | **37.1** | **34.2** | **50.1** |

关键观察：
1. VimRAG 在所有 benchmark 上都达到 SOTA，overall 从 43.6 → 50.1（+6.5 points）
2. 在 visual-heavy benchmark（SlideVQA, MMLongBench, LVBench）上提升最显著，说明 graph-modulated visual encoding 的效果
3. 在 text benchmark（HotpotQA, SQuAD）上也有提升，说明 graph topology 对 reasoning 本身有帮助，不仅限于 visual

### 5.2 Ablation Study（Table 3）

| Memory Structure | | Memory Shaping | | Acc |
|------------------|---|----------------|---|-----|
| Iter. | Graph | Multi. | Std. | Graph Energy | |
| ✓ | | ✓ | | | 43.6 |
| ✓ | | | ✓ | | 47.1 |
| ✓ | ✓ | | ✓ | | 48.9 |
| ✓ | ✓ | | ✓ | ✓ | **50.1** |

逐步加入 component 的效果：
- Baseline（iterative summary + std shaping）：43.6
- + Graph topology：43.6 → 47.1（+3.5）→ 48.9（+1.8 with multimodal）
- + Graph energy allocation：48.9 → 50.1（+1.2）

**Intuition**：Graph topology 的贡献最大（+3.5），说明 structural bias 对 reasoning 的帮助最关键。Graph energy allocation 的贡献相对小（+1.2），但在 visual-heavy task 上可能更重要（paper 没有分 benchmark 的 ablation）。

### 5.3 RL Training Dynamics（Figure 5, 6b）

Figure 6b 显示 VimRAG 的 training entropy curve 比 baseline GSPO 更快收敛。Figure 5 显示 ablation 后 GGPO（graph-guided pruning）比无 pruning 的 baseline 更 robust。

**Intuition**：这说明 pruning mask 帮助 model 聚焦在 "clean" positive gradient 和 "safe" negative gradient 上，避免了 ambiguous update 导致的 training instability。这跟 [DPO](https://arxiv.org/abs/2305.18290) 里 preference data quality 比 quantity 更重要的发现一致——高质量 supervision 比大量 noisy supervision 更有效。

### 5.4 Inference Efficiency（Figure 6c）

VimRAG 的 trajectory length 显著短于 ReAct 和 Mem1，原因是 structured memory 避免了 redundant loop。

**Intuition**：这验证了 Pilot Study 的第一个 insight——graph topology 帮助 agent "记住" 自己做过什么，避免重复 query。这跟 [Tree of Thoughts](https://arxiv.org/abs/2305.10601) 里 exploration efficiency 的思路类似，但 ToT 是 explicit tree search，VimRAG 是 implicit graph memory。

---

## 6. 相关工作和延伸联想

### 6.1 Context Management 谱系

VimRAG 在 context management 的演进中处于这个位置：

1. **ReAct**（[Yao et al. 2022](https://arxiv.org/abs/2210.03629)）：append-all-history，flat sequence
2. **MemAgent**（[Yu et al. 2025](https://arxiv.org/abs/2507.02259)）：multi-conv memory agent，hierarchical context
3. **Mem1**（[Zhou et al. 2025](https://arxiv.org/abs/2506.15841)）：iterative summarization，single memory state
4. **Mem0**（[Chhikara et al. 2025](https://arxiv.org/abs/2504.19413)）：production-ready memory with recency/relevance weighting
5. **A-Mem**（[Xu et al. 2025](https://arxiv.org/abs/2502.12110)）：agentic memory with self-organization
6. **VimRAG**：structured graph memory with topology-aware token allocation

### 6.2 跟 GraphRAG 的区别

[GraphRAG](https://arxiv.org/abs/2404.16130) 是 Microsoft 的工作，用 knowledge graph 来增强 RAG。但 GraphRAG 的 graph 是**预构建的 entity-relation graph**，从 corpus 里 extract 出来的。VimRAG 的 graph 是**agent reasoning 过程中动态构建的 reasoning trace graph**，node 是 agent state，edge 是 reasoning dependency。

这是两个完全不同的 abstraction level：
- GraphRAG：knowledge-level graph（什么是 entity，什么是 relation）
- VimRAG：process-level graph（agent 做了什么，依赖什么）

### 6.3 跟 RL 的 Connection

VimRAG 的很多设计跟 RL 概念有对应：

| VimRAG Concept | RL Analogue |
|----------------|-------------|
| Memory Graph | Trajectory / Episode |
| Node $v_i$ | State $s_i$ |
| Action $a_t$ | Action in MDP |
| Critical Path $\mathcal{P}_{ans}$ | Optimal Trajectory |
| Dead-End Nodes | Suboptimal Branches |
| Energy $\Omega$ | Value Function $V(s)$ |
| Recursive Reinforcement | Bellman Backup |
| Pruning Mask $\mu$ | Reward Shaping / Credit Assignment |
| GGPO | PPO with reward decomposition |

这个 connection 很自然，因为 POMDP formulation 本身就是 RL 的标准框架。VimRAG 的贡献在于把 RL 的 credit assignment 思路应用到 LLM agent training 上，用 graph structure 来提供 step-level supervision signal。

### 6.4 跟 Hierarchical RL 的 Connection

Recursive reinforcement（Eq. 7）的计算方式：

$$\Omega(m_{i,k}) = \mathcal{E}_{int}(m_{i,k}) + \gamma \sum_{v_j \in \text{Child}(v_i)} \overline{\Omega}(v_j)$$

这本质上是 backward induction / value iteration 在 graph 上的应用。跟 [Options Framework](https://arxiv.org/abs/1606.01847)（Sutton et al.）的 hierarchical RL 思路类似——early action 的价值通过它 lead 到的后续 state 来评估。

### 6.5 跟 GNN 的 Connection

Energy computation 的 recursive structure 也跟 GNN 的 message passing 类似：

$$h_v^{(k)} = \text{UPDATE}(h_v^{(k-1)}, \text{AGGREGATE}(\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}))$$

VimRAG 的 energy computation 可以看作 1-layer GNN 在 reasoning graph 上的 inference。如果扩展到 multi-hop message passing，可能能捕捉更远距离的 dependency。

### 6.6 跟 Attention Sink / KV Cache Compression 的 Connection

Visual token allocation 的问题跟 [Attention Sink](https://arxiv.org/abs/2309.17453)（StreamingLLM）和 [H2O](https://arxiv.org/abs/2306.14048)（Heavy Hitter Oracle）的 KV cache compression 是同一类问题——如何在有限 budget 下保留最重要的 token。

区别在于：
- KV cache compression 基于 attention score（runtime, model-internal）
- VimRAG 的 token allocation 基于 graph energy（structural, reasoning-aware）

VimRAG 的方法更 "semantic"，因为它考虑了 reasoning context，而不仅仅是 attention pattern。

### 6.7 跟 Process Reward Model 的 Connection

GGPO 的 pruning mask 本质上是在做 **process-level reward**，而不是 outcome-level reward。这跟 [PRM800K](https://arxiv.org/abs/2406.06592)（OpenAI's process reward model）的思路一致——对 reasoning 的每一步打分，而不是只看最终结果。

但 VimRAG 的区别是：
- PRM 用单独的 reward model 给每步打分
- VimRAG 用 graph topology 来 infer 哪些 step 是 good/bad，不需要额外的 reward model

这使得 VimRAG 的方法更 lightweight，但也更 heuristic（依赖 critical path traversal 的准确性）。

---

## 7. 局限性和未来方向

paper 自己提到的 limitation（Appendix H）：
1. 依赖 base model 能力（Qwen3-VL）
2. multi-turn interaction 不适合 high real-time 应用
3. retriever 准确性限制

我认为还有几个值得思考的方向：

### 7.1 Graph Linearization 的 Loss

当前方法把 graph $\mathcal{G}_t$ linearize 成 text $\mathcal{L}(\mathcal{G}_t)$ 喂给 LLM。这个 linearization 会损失 structural information。未来的方向可能是用 [Graph Transformer](https://arxiv.org/abs/2012.09699) 或 [Structure-Aware Attention](https://arxiv.org/abs/2203.09553) 让 LLM 直接 attend to graph structure。

### 7.2 Energy Function 的 Learning

当前 energy function（Eq. 6-7）是 hand-crafted 的。未来的方向是 learn 这个 energy function，比如用 contrastive learning 来学 node importance。这跟 [Learning to Rank](https://arxiv.org/abs/1904.05024) 的思路类似。

### 7.3 Multi-Agent Extension

如果多个 agent 协作，可以形成一个 shared memory graph，每个 agent 贡献不同的 sub-graph。这跟 [Multi-Agent RL](https://arxiv.org/abs/1810.11707) 的 CTDE（Centralized Training Decentralized Execution）框架有 connection。

### 7.4 跟 Test-Time Compute Scaling 的 Connection

VimRAG 的 graph expansion 本质上是一种 test-time compute scaling——通过多步 reasoning + retrieval 来提升 accuracy。这跟 [OpenAI o1](https://openai.com/o1/) 的 test-time reasoning 思路一致。未来的方向可能是把 VimRAG 的 graph search 跟 MCTS 结合，做更 systematic 的 test-time exploration。

---

## 8. 总结

VimRAG 的核心贡献是在 multimodal RAG 场景下引入了三个互相 coupling 的设计：

1. **Structured Reasoning Topology**：把 reasoning 建模成 DAG，解决 linear history 的 information dilution 和 Markovian blindness
2. **Graph-Modulated Visual Memory Encoding**：用 energy-based allocation 解决 visual memory 的 token budget vs. fidelity dilemma
3. **Graph-Guided Policy Optimization**：用 graph pruning解决 sparse reward 的 credit assignment problem

这三个设计形成一个 coherent 的 framework：graph topology 既服务于 inference（memory encoding），又服务于 training（credit assignment）。这种 inference-training 的 alignment 是这篇文章最 elegant 的地方。

**最终 Intuition**：VimRAG 本质上是在说——agent reasoning 不应该是一个 flat sequence，而应该是一个 structured graph；memory 不应该是 passive storage，而应该是 active context shaper；supervision 不应该是 outcome-only，而应该是 process-aware。这三个 principle 对未来的 agentic AI 研究有指导意义。

---

**参考链接**：
- [VimRAG Paper (本篇)](https://arxiv.org/abs/2508.05748)
- [VimRAG GitHub](https://github.com/Alibaba-NLP/VRAG)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Mem1](https://arxiv.org/abs/2506.15841)
- [Mem0](https://arxiv.org/abs/2504.19413)
- [GraphRAG](https://arxiv.org/abs/2404.16130)
- [ColPali](https://arxiv.org/abs/2407.01449)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [GSPO](https://arxiv.org/abs/2507.11071)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [Process Reward Models](https://arxiv.org/abs/2406.06592)
- [Attention Sink / StreamingLLM](https://arxiv.org/abs/2309.17453)
- [H2O KV Cache Compression](https://arxiv.org/abs/2306.14048)
- [FastV](https://arxiv.org/abs/2403.06764)
- [VideoChat-Flash](https://arxiv.org/abs/2501.00574)
- [A-Mem](https://arxiv.org/abs/2502.12110)
- [MemAgent](https://arxiv.org/abs/2507.02259)
- [HowTo100M](https://arxiv.org/abs/1906.03327)
- [Options Framework](https://arxiv.org/abs/1606.01847)
- [Graph Transformer](https://arxiv.org/abs/2012.09699)
