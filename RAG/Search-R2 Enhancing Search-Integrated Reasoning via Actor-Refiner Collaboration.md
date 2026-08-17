---
source_pdf: Search-R2 Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration.pdf
paper_sha256: 74e5e503426ccd0a967fe4c41bdbe1c6e9fc7b3964087a6eee8899e888306ba4
processed_at: '2026-08-12T04:25:48-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Search-R2 人话版讲解

## 一句话总结

这篇paper说：**让LLM自己当"审稿人"，发现自己推理链哪里跑偏了，从那个点砍掉重写，而不是把整条思路扔掉重来**。这么做比单纯多采样几次更省算力、更准。

Paper link: https://arxiv.org/abs/2503.09516 (Search-R1, 本文baseline)
项目代码可参考 verl framework: https://github.com/volcengine/verl

---

## 这玩意儿到底在解决什么问题

### 先讲场景
你有一个LLM agent，它要做多跳问答，比如"谁执导了那部改编自X小说、且在Y年份上映的电影？"。这种问题一次搜索答不出来，需要：

1. 先搜小说作者
2. 再搜改编电影
3. 再搜导演

每一步都是一次 `<search>query</search>` → 拿回 `<information>...</information>` → 继续推理的循环。

### Search-R1 (前作) 哪里不行
Search-R1 用 RL (具体是 GRPO) 训练这个 agent，reward 只有最终的 Exact Match (答对=1，答错=0)。问题在于：

- 一条 trajectory 答对了，**不代表中间每一步都对**。可能第3步搜索结果其实和问题无关，但模型蒙对了答案
- 一条 trajectory 答错了，**不一定整条都废**。可能只是第2步的 query 写得不好，拿回一堆垃圾信息，把后面的推理全带沟里了

这就是 paper 里说的 **multi-scale credit assignment problem**：你想优化的是每一步的 search/reason 决策，但你手上只有整条 trajectory 末尾的一个 0/1 信号。结果就是：sample efficiency 低，训出来的 agent 经常发冗余、跑偏的 query。

用 rejection sampling 的笨办法是：答错的整条扔掉重新 roll。问题是已经正确的前半段也被浪费了。

---

## Search-R2 的核心 idea

把一个 agent 拆成两个角色（但共享同一套权重 θ）：

### Actor (演员 π_l)
就是原来那个生成 reasoning + search call 的 policy。用 Search-R1 的模板：

```
Answer the question. Reason inside <think>...</think>.
If you need info, call <search>query</search>.
Results come back in <information>...</information>.
Final answer in <answer>...</answer>.
```

### Meta-Refiner (审稿人)
Actor 生成完一条 trajectory 后，Meta-Refiner 上场做两件事：

**1. Discriminator (判断该不该修)**
判断这条 trajectory 整体 coherent 吗？公式上：
- π_d(ŷ|x) ∈ [0,1]：轨迹全局连贯的概率
- 如果 π_d ≥ τ (阈值)，接受，这条 trajectory 通过
- 如果 π_d < τ，标记需要修复

**2. Trimmer (定位该砍哪)**
找出 trajectory 第一次跑偏的位置 k+1：
- π_h(k|ŷ, x)：在位置 k+1 砍掉的概率分布
- 保留 prefix ŷ_{1:k}（前面好的部分）
- 用 base policy π_l 从 k 位置重新生成 suffix

这就是 **"cut-and-regenerate"** 机制。核心 insight：与其整条重来，不如外科手术式地只修坏掉的那一段。

### 算法流程 (Algorithm 1)
```
生成 ŷ ~ π_l(·|x)
for n = 0 to N_max:
    if π_d(ŷ|x) ≥ τ: return ŷ  // 接受
    采样 cut-point k ~ π_h(·|ŷ, x)
    prefix = ŷ_{1:k}
    重新生成 suffix ~ π_l(·|x, prefix)
    ŷ = [prefix, suffix]
return ŷ
```

---

## Reward 设计：为什么单纯 EM 不够

这是另一个关键创新。paper 设计了一个 **hybrid reward**：

$$R(y) = r_{outcome}(y) \cdot (1 + r_{process}(y))$$

变量解释：
- r_outcome(y) = I(a_pred = a_gold)，就是 Exact Match，答对=1 否则=0
- r_process(y) = (1/M) Σᵢ uᵢ，信息密度
  - M = trajectory 里总共有几次 search 调用
  - uᵢ ∈ {0,1} = 第 i 次搜索拿回的文档集合是否"有用"
  - "有用"由一个外部 LLM judge (DeepSeek-R1-Distill-Qwen-7B) 判定，标准是：包含能推出正确答案的线索，且不是冗余重复

### 为什么这么设计
乘法 gating 是关键：只有答对了 (r_outcome=1)，process reward 才生效。这防止了 reward hacking——模型可能学会一直发 query 刷高分但根本不答对问题。

直观上讲：r_outcome 告诉模型"做对了"，r_process 告诉模型"做对的方式好不好"。两条都答对的 trajectory，一条只搜了1次有用信息，另一条搜了5次才搜到有用信息，后者 reward 更低。

---

## 为什么必须 Joint Optimization（这是理论核心）

很多人可能想：Meta-Refiner 用现成的 prompt 指挥就行了，为什么要和 Actor 一起训？paper 的 Section 3.5 + 4 给了严格证明。我把核心 intuition 讲清楚：

### Performance Gain 的分解

设：
- J_base = E_{y~π_l}[R(y)]，原始 actor 的平均 reward
- J_meta = E_{y~q}[R(y)]，加了 Meta-Refiner 后的平均 reward
- ΔJ = J_meta - J_base

paper 把 ΔJ 拆成三项的乘积/求和：

$$\Delta J = \underbrace{\mathcal{A}_{prec}}_{\text{Selection Precision}} + \underbrace{\mathcal{V}_{inter}}_{\text{Intervention Volume}} \times \underbrace{S_{trim}}_{\text{Trimming Skill}}$$

三项含义：

**(1) Selection Precision A_prec**
$$\mathcal{A}_{prec} = \text{Cov}_{\pi_l}(\alpha(y), R(y) - J_{trim}(y))$$
- α(y) = discriminator 接受这条 trajectory 的概率
- R(y) - J_trim(y) = 接受它的收益减去修复它的潜在收益
- 直觉：discriminator 要能精准识别"这条值得留" vs "这条该修"。如果 discriminator 瞎判，接受了一堆本该修的垃圾，A_prec 是负的或零

**(2) Trimming Skill S_trim**
$$S_{trim} = \sum_k \text{Cov}(\pi_h(k|\hat{y}), G_k(\hat{y}))$$
- G_k(ŷ) = V^{π_l}(ŷ_{1:k}) - R(ŷ)，从位置 k 重新生成的"收益"
- 直觉：trimmer 要能精准定位"真正的故障点"，而不是随便砍一刀。如果它每次都砍在无关位置，S_trim ≈ 0

**(3) Intervention Volume V_inter**
$$\mathcal{V}_{inter} = 1 - Z_{acc}, \quad Z_{acc} = \mathbb{E}_{\pi_l}[\alpha(y)]$$
- 直觉：有百分之多少的 trajectory 被送去修了。如果 discriminator 太宽容什么都接受，V_inter≈0，trimmer 没机会发挥；如果太严格什么都拒，V_inter≈1，算力全浪费在修复上

### 关键结论

paper 的 Theorem 4.1 和 4.2 证明：**只有 A_prec > 0、S_trim > 0、V_inter 合理校准**，ΔJ 才严格大于 0。这三个条件必须同时满足。

如果只用静态 prompt（不训练 Meta-Refiner），discriminator 和 trimmer 的能力是固定的，没法保证 A_prec 和 S_trim 朝正方向走。**Joint optimization 让 GRPO 自动把这三个量推向最优**——把 discriminator 误判的 trajectory 的 advantage 反传回去，下次 discriminator 就学会判准了。

这就解释了为什么 paper Section 5.3 的 ablation 里，"+Meta-Refiner (不joint训)" 比 "+Meta-Refiner +Joint Optimization (完整版)" 差了一截：

| Qwen2.5-7B | Avg EM |
|---|---|
| Search-R1 | 35.0 |
| +Meta-Refiner (不joint) | 38.9 |
| +Process Reward | 39.6 |
| **Full (joint optimization)** | **40.4** |

---

## 实验结果亮点

### 主实验 (Table 2)
7 个 benchmark：NQ, TriviaQA, PopQA (general QA), HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle (multi-hop QA)

最关键的对比：
- **Search-R2 (Qwen2.5-7B) = 40.4 avg EM**
- **Search-R1 (Qwen3-8B) = 40.0 avg EM**

7B 的 Search-R2 居然打败了 8B 的 Search-R1，证明 Actor-Refiner 框架能弥补模型规模的劣势。32B 版本平均达到 50.8，在 Bamboogle (多跳) 上从 45.0 → 56.4，相对提升 25.3%。

### 效率分析 (Table 5)
训练时间开销对比：

| Model | Search-R1 (s/step) | Search-R2 (s/step) | Δ Time | Δ EM | Δ EM / Δ Time |
|---|---|---|---|---|---|
| 7B | 177.8 | 193.2 | +8.66% | +15.4% | 1.78 |
| 8B | 141.5 | 147.3 | +4.10% | +11.5% | 2.80 |
| 32B | 458.4 | 469.5 | +2.43% | +11.4% | 4.69 |

Δ EM / Δ Time > 1 意味着每多花一份算力换回多于一份的精度提升。32B 上开销最小（2.43%），因为分布式训练时边际成本被摊薄。

### 和双倍 rollout 的对比 (Appendix G, Table 7)
有人可能质疑：你这提升不就是多算了一次 regenerate 吗？我把 Search-R1 的 rollout 数从 n=5 翻倍到 n=10 当对照组：

- Search-R1 (n=10): avg 47.8 @ step 300
- Search-R2 (n=5, max revision=1): avg 50.8 @ step 300

Search-R2 用大约 3300 条 trajectory/step（因为只 ~30% trajectory 触发修复），打败了用 5120 条 trajectory/step 的 Search-R1。**外科手术式修复 >> 暴力多采样**。

### Revision 敏感度 (Table 4)
Qwen2.5-32B 上调整 max revision 次数：

| Max Revision | NQ | Avg |
|---|---|---|
| 1 | 50.8 | 49.3 |
| 2 | 50.9 | 50.2 |
| 3 | 51.4 | 50.6 |
| 4 | 51.6 | 50.9 |

收益递减很快：1→2 提升 0.9，3→4 只提升 0.3。说明大部分错误一次修复就够了，难 case 多修几次也救不回来。所以默认 max revision=1，效率最高。

---

## 质量评估 (Table 9, 用 GPT-5.1 当 judge)

六维度评估 trajectory 质量（700 对样本）：

| 维度 | Search-R2 win / Search-R1 win |
|---|---|
| Evidence Groundedness | 19.3 / 2.4 |
| Information Density | 36.4 / 6.3 |
| Non-Redundancy Efficiency | 32.0 / 4.7 |
| Query Timing Quality | 14.1 / 0.9 |
| Trajectory Coherence | 31.3 / 4.4 |
| Uncertainty Handling | 8.9 / 1.7 |

Search-R2 在所有维度碾压，尤其 Query Timing Quality（14.1 vs 0.9），说明训出来的 agent 知道什么时候该搜、什么时候别搜。

---

## 给你的 Intuition 总结

1. **Actor 单角色 → Actor + Refiner 双角色**：人类做题也会"边做边回头检查"，LLM agent 也需要
2. **整条丢弃 → 局部砍掉重写**：保留好 prefix，省算力
3. **稀疏 EM reward → hybrid reward (EM × info density)**：dense supervision 让 RL 知道哪条 trajectory 不只是答对了，还是"高质量地答对"
4. **静态 prompt refiner → jointly optimized refiner**：refiner 也要学习，否则它不知道该砍哪、什么时候该介入
5. **理论保证**：ΔJ > 0 当且仅当 selection precision > 0、trimming skill > 0、intervention volume 校准——这三个量的优化正是 GRPO 自然完成的

这套思路其实和人类写论文、写代码的流程很像：先写初稿 (Actor) → 自己 review 找问题 (Discriminator) → 定位到具体段落重写 (Trimmer) → 多轮迭代。把这个过程 end-to-end 用 RL 学出来，就是 Search-R2 的核心 contribution。

---

## 相关延伸阅读建议

如果你想顺着这条线深挖：
- **Self-Refine** (https://arxiv.org/abs/2303.17651): 启发式 self-correction，没有 RL
- **ReST** (https://arxiv.org/abs/2308.08998): Google 的 self-improvement 框架，迭代式 fine-tune
- **Process Reward Models** (https://arxiv.org/abs/2306.09031, OpenAI 的 "Let's Verify Step by Step"): step-level reward 的鼻祖
- **GRPO** (DeepSeekMath, https://arxiv.org/abs/2402.03300): 本文用的 RL 算法
- **verl** framework (https://github.com/volcengine/verl): 开源 RL training 框架，本文实验基于它

希望这个版本讲得更人话一点了。如果还有哪个部分你觉得不够清楚，告诉我具体哪段，我再展开讲。

---

# Search-R2: Actor–Refiner Collaboration 深度解析

## 1. 核心问题: Multi-Scale Credit Assignment

Search-R1 这类 agentic RAG 训练范式使用 trajectory-level 的稀疏 reward (Exact Match)，会把整条包含 query formulation → retrieval → reasoning → answer 的多步决策序列当作一个黑盒评分。这会引发一个严重的 credit assignment 问题：

- 一个 trajectory 答对，可能只是最后 reasoning step 的 lucky guess，中间有多次无意义的搜索
- 一个 trajectory 答错，可能是因为第 2 步一个不相关的 search query 把 reasoning 带偏，后续整条链全部 collapse
- Rejection sampling 的粗暴做法是把整条 trajectory 丢掉重新生成，浪费了已经正确的 prefix

Search-R2 的关键洞察：**局部错误需要局部修复，全局判断需要 dense supervision**。这要求把 actor 单一角色拆分为 Actor (生成器) 与 Meta-Refiner (诊断+修剪器) 两个角色，并且两者必须 jointly optimized。

## 2. 框架架构解析

整个系统由三个 policy 构成，全部共享底层 LLM 参数 θ：

### 2.1 Actor π_l(·|x)
原始 reasoning policy，按照 Search-R1 的模板格式生成：
- `
