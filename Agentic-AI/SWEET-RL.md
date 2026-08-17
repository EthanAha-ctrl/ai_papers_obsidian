---
source_pdf: SWEET-RL.pdf
paper_sha256: eb4b5b01ef90b76361d9b92c78202a7e342fd247af24c4e694c802730c4ba428
processed_at: '2026-08-12T11:37:00-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SWEET-RL

## 这篇 paper 在解决什么"痛点"

你训一个 LLM agent 让它干多轮交互的活儿——比如帮人写代码、做网页——会发现一个尴尬的局面: 单轮 (single-turn) 模式下 GPT-4o 写个函数成功率才 16%，给它 10 轮跟"人"来回 clarification 后能到 40%，但这个"来回问问题"的策略本身并不天然存在于 LLM 里，你得 fine-tune 出来。

怎么 fine-tune? 传统套路有两条:

**第一条: 看结果给 reward，用 PPO / REINFORCE 之类**。问题: horizon 一长 (10 轮)，credit assignment 炸了。最后一轮成功，功劳算谁的? 第一轮问了个好问题，还是第五轮答得漂亮? 高方差，学不动。

**第二条: DPO 之类的 preference optimization，直接对整个 trajectory 做对比**。问题: 它把整条 trajectory 当一个整体打分，相当于把"长程信用分配"踢回给模型自己悟。悟不出来，效果有限。

paper 里的 Table 2 显示 Llama-3.1-8B 加 Multi-Turn DPO 在 backend 上 34.4%，离 GPT-4o 的 40.4% 还有不少差距。这就是 SWEET-RL 要补的 6%。

核心 reference: [DPO paper](https://arxiv.org/abs/2305.18290), [PPO paper](https://arxiv.org/abs/1707.06347), [ARCHER (hierarchical multi-turn RL)](https://arxiv.org/abs/2402.19446)

---

## SWEET-RL 的核心 trick: "训练时作弊"

SWEET-RL 的 idea 用一句话讲: **让 critic 在训练时多看一眼 reference solution，actor 测试时看不到**。

举个具体场景: backend programming 任务，reference code 里有一个关键逻辑——返回的 list 可能包含 duplicate objects。一个聪明的 agent 在 turn 2 会主动问: "返回的列表允许有重复元素吗?"。

如果 critic 跟 actor 一样只看交互历史，它怎么知道这个问题问得好不好? 它看不到 reference，无从判断这个 clarification 是否挖到了关键点。结果它只能猜，猜不准，credit 就分错。

SWEET-RL 的解法很粗暴: **把 reference solution 喂给 critic**。critic 看到 reference，秒懂"哦原来 duplicate 是关键点，那 turn 2 问这个问题问得好，advantage 给高一点"。actor 看不到 reference，所以部署时不会作弊，但训练时享受 critic 提供的精准 step-level 信号。

这个 trick 在 robotics 里叫 "asymmetric actor-critic" ([Pinto et al. 2017](https://arxiv.org/abs/1710.06542))，critic 看 latent state，actor 看 RGB。SWEET-RL 把它搬到了 reasoning-heavy 的 LLM agent 场景，training-time info 是 reference solution 而不是物理 state。

---

## 为什么不直接学 Value Function

传统 RL 会训一个 value head 预测 expected return $V(s)$。SWEET-RL 不这么干，理由很实在:

- 预测 expected return 这个任务对 LLM 来说是 foreign task，需要加新的 classification / regression head
- 这偏离了 LLM 的 pre-training objective (next-token prediction)，少量 fine-tune data 下泛化差
- Figure 3a 的实验直接打脸: 用 value head 的 Best-of-N scaling 曲线很差，远不如 SWEET-RL 的 advantage parameterization

SWEET-RL 反过来: **直接学 advantage $A(s, a)$，并且复用 LLM 的 language modeling head**。

具体怎么复用? advantage 被参数化为当前 policy 和 reference policy 在每个 token 上的 log ratio，再除以 response 长度 $L$ 取平均:

$$A_{\theta}(o_t, a_t, c) = \frac{1}{L} \sum_{l=1}^{L} \log \frac{\pi_{\theta}(a_t^l | o_t, a_t^{1:l-1}, c)}{\pi_{\text{ref}}(a_t^l | o_t, a_t^{1:l-1}, c)}$$

这里:
- $o_t$: 第 $t$ 轮的观测 (交互历史)
- $a_t^l$: 第 $t$ 轮的第 $l$ 个 token
- $a_t^{1:l-1}$: 前 $l-1$ 个 token
- $c$: training-time info (reference solution)
- $\pi_{\theta}$: 被训练的 LLM
- $\pi_{\text{ref}}$: frozen 的初始模型
- $L$: response token 数

这个形式跟 DPO 的 implicit reward $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$ 几乎一样，区别就是多了个 $\frac{1}{L}$ 做长度归一化。

**除以 $L$ 看起来是个小细节，但致命重要**。Table 3 里去掉这个 normalization，success rate 从 40.4% 暴跌到 3.6%——actor 学会输出越来越短的 response 来 hack advantage。这是 LLM RL 训练里非常典型的坑。

---

## 两个 stage 的训练流程

paper 把训练拆成两步，Figure 2 画得很清楚:

### Stage 1: 训 advantage function $A_{\theta}$

同一个 task 下采两条 trajectory，cumulative reward 高的叫 chosen ($\tau^+$)，低的叫 rejected ($\tau^-$)。用 Bradley-Terry objective:

$$\mathcal{L}_A(\theta) = -\log \sigma\left(\sum_t \beta A_{\theta}(o_t^+, a_t^+, c) - \sum_t \beta A_{\theta}(o_t^-, a_t^-, c)\right)$$

这里:
- $\sigma$: sigmoid
- $\beta$: 温度系数
- $o_t^+, a_t^+$: chosen trajectory 在 turn $t$ 的观测和 action
- $o_t^-, a_t^-$: rejected trajectory 在 turn $t$ 的观测和 action
- $c$: reference solution (只有 critic 看)

效果就是让 chosen trajectory 里每轮的 advantage 高一点，rejected 里每轮低一点。**trajectory-level 的偏好被"摊薄"到 turn level**，这是 credit assignment 的关键。

### Stage 2: 把 advantage 当 reward 训 actor

actor 不看 $c$。每轮采样 16 个 candidate action，用 $A_{\theta}$ 打分，top-50% 当 chosen，bottom-50% 当 rejected，跑标准 DPO:

$$\mathcal{L}_{\pi}(\phi) = -\log \sigma\left(\beta' \log \frac{\pi_{\phi}(a^+|o_t)}{\pi_{\text{ref}}(a^+|o_t)} - \beta' \log \frac{\pi_{\phi}(a^-|o_t)}{\pi_{\text{ref}}(a^-|o_t)}\right)$$

注意这里 $\pi_{\phi}$ 和 $\pi_{\text{ref}}$ 都只看 $o_t$，不看 $c$，因为部署时 actor 看不到 reference。Stage 2 不需要任何 human 交互，纯离线。

---

## 为什么这个 trick 在期望意义下是"合法的"

有人会问: critic 看的东西比 actor 多，训出来的 policy gradient 不会偏吗?

Appendix B.2 的 Lemma B.2 证明: **不会**。关键等式:

$$\mathbb{E}_{c \sim d_t^{\pi}(\cdot|o_t, a_t)} A^{\pi}(o_t, a_t, c) = A^{\pi}(o_t, a_t)$$

意思是: 对 $c$ 求条件期望，带 $c$ 的 advantage 就回到普通 advantage。所以 critic 多看 $c$ 只是降低单样本的方差，期望上不偏离真正的 policy gradient。这是个 clean 的 trick: 训练时借力 privileged info，部署时不依赖。

附录里还有个 Lemma B.1 也很有意思: 在 deterministic transition 下

$$\sum_t r(o_t, a_t, c) = \sum_t A^{\pi}(o_t, a_t, c)$$

证明用 telescoping:

$$\sum_t A^{\pi} = \sum_t (Q^{\pi} - V^{\pi}) = \sum_t [r + V^{\pi}(o_{t+1}) - V^{\pi}(o_t)] = \sum_t r$$

中间的 $V$ 项互相抵消。这说明 trajectory-level preference 和 advantage-level preference 是等价的——意味着你可以从容易获得的 trajectory-level 标注 (跑 rollout 看 final reward) 推出 turn-level 监督信号，不用人工标 step-level label。

---

## ColBench: 配套 benchmark

Paper 还顺手做了个 benchmark，因为现有 benchmark 都不太适合研究 multi-turn RL。Table 1 里列了一堆 benchmark 都缺东西——WebArena ([arxiv.org/abs/2307.13854](https://arxiv.org/abs/2307.13854)) 有 reasoning 但 overhead 高，LMRL Gym ([arxiv.org/abs/2311.18232](https://arxiv.org/abs/2311.18232)) 能跑 RL 但 reasoning 要求低。

ColBench 两条 task:

**Backend Programming**: agent 和 human simulator (LLM) 交互最多 10 轮，写 ≤50 行 Python 函数。evaluator 跑 10 个 hidden unit test，0/1 reward。train 10k tasks，test 1k。

**Frontend Design**: agent 写 ~100 行 HTML，human simulator 是 VLM (Qwen2-VL-72B)，看 agent 渲染的网页 vs reference 网页，描述差异。evaluator 用 CLIP ([arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)) cosine similarity。train 10k，test 500。

最关键的 trick: **human simulator 能看到 reference artifact**。这把"模拟人"的不确定性大大降低——人不是真的随机说话，而是基于 ground truth 回答。代价是这个人有点过于理想，但换来了 minimum engineering overhead 和可自动化的评估。

---

## 实验结果亮点

### Table 2 主结果

| Setting | Backend Success | Frontend Win |
|---------|-----------------|--------------|
| Single-turn GPT-4o | 16.2 | 23.8 |
| Multi-turn zero-shot GPT-4o | 40.4 | 50.0 |
| Multi-turn zero-shot Llama-3.1-8B | 22.4 | 33.8 |
| Rejection FT (Llama-3.1-8B) | 28.2 | 38.6 |
| Multi-Turn DPO (Llama-3.1-8B) | 34.4 | 42.8 |
| **SWEET-RL (Llama-3.1-8B)** | **40.4** | **48.2** |

观察:

1. **Single-turn → Multi-turn 增益巨大**: GPT-4o 从 16.2 跳到 40.4。协作本身解锁了大量性能。
2. **8B + SWEET-RL ≈ GPT-4o**: 40.4 vs 40.4 (backend)，48.2 vs 50.0 (frontend)。小模型经过 task-specific RL fine-tune 能逼近大模型。
3. **SWEET-RL 比 Multi-Turn DPO 高 6%**: 这是 explicit credit assignment + asymmetric critic 的直接收益。
4. **o1-mini 协作能力一般**: zero-shot 协作下 o1-mini (43.2 tests passed) 反不如 GPT-4o (54.6)。强 reasoning 不等于好协作策略，下游 fine-tune 仍必要。

### Table 3 Ablation 的戏剧性

| Variant | Success Rate |
|---------|--------------|
| SWEET-RL | 40.4 |
| w/ Regression Head | 36.2 |
| w/o Training-Time Info | 31.2 |
| w/o Normalization | **3.6** |

三个发现:

1. **Regression head 比 mean log prob 差**: 加 value head 会让泛化变差，因为它偏离了 pre-training objective
2. **Training-time info 是主要增益来源**: 去掉 $c$ 后比 Multi-Turn DPO (34.4) 还差，说明 asymmetric info 是核心
3. **Length normalization 是命门**: 去掉 $\frac{1}{L}$，actor 学会输出极短 response，success 暴跌到 3.6%

### Figure 3a Best-of-N scaling

固定 actor，用不同 step reward model 做 Best-of-N 采样:

- SWEET-RL: 曲线最陡，scaling 最好
- SWEET-RL w/o Training-Time Info: 几乎无增益
- LLM-as-a-Judge: 增益有限，被长度/格式 distraction
- SWEET-RL w/ Value Function: scaling 差，value head 泛化失败

### Figure 3b Data scaling

3k 数据时 SWEET-RL 反而比 Multi-Turn DPO 差——critic 还没训好。数据上去后快速追赶并显著超越。这是 "先付 critic 训练的固定成本，之后享受更好 credit assignment" 的 trade-off。

### Table 4 强 base model + 弱 data

用 Llama-3.1-70B 作 base，但 data 还是 8B 生成的:

| Method | Success Rate |
|--------|--------------|
| Zero-shot | 35.0 |
| RFT | 31.9 (下降!) |
| Multi-Turn DPO | 41.8 |
| SWEET-RL | 45.6 |

RFT 失败特别有意思: 它让 70B word-by-word 模仿 8B 的 sub-optimal trajectory，相当于强模型"装弱"。SWEET-RL 因为做 credit assignment，能识别 8B trajectory 里哪些 turn 好，避免全盘照搬。这对实际部署很有启发——可以用便宜的小模型生成数据训大模型。

---

## 定性例子里的 emergent behavior

Appendix E 给的完整 trajectory 里有几个有意思的现象:

1. **SWEET-RL 训练后的 Llama-3.1-8B 会"等"**: 在 backend 任务里它真的来回问 5-6 轮才给答案，而 zero-shot GPT-4o 往往问 1-2 轮就跳到结论。Figure 7-8 vs Figure 10 对比很明显。

2. **自发出现 chain-of-thought 和 self-correction**: Figure 8 里 SWEET-RL agent 在 turn 5-6 还在做 reasoning 和自我修正，这是 RL 训练自然涌现的，不是 prompt 工程出来的。

3. **Frontend 任务学到"先 scratch 后 refine"策略**: Figure 11-16 里，agent 先给个粗略 HTML 探探 feedback，根据 human simulator 描述的差异再做精细 edit。这种 reward-maximizing 行为是自然学出来的，up to 16k token context 下也能 hold 住。

---

## 跟 PRM 的区别

paper Section 2 专门讨论了跟 Process Reward Model ([Lightman et al. 2023](https://arxiv.org/abs/2305.20050), [Setlur et al. 2024](https://arxiv.org/abs/2410.08146)) 的区别:

- **PRM**: 评估每个 reasoning step 的"正确性"，通常用于 test-time search 或加速 on-policy RL exploration
- **SWEET-RL critic**: 用作 credit assignment 的 intermediate reward proxy，直接优化 policy，不再 collect 交互数据

这个区别对 LLM agent 任务特别重要: 跟环境交互代价极高 (要 human 在线)，所以"训练时用 critic，部署时丢掉 critic" 是干净的设计。

---

## 我对这篇 paper 的整体评价

**亮点**:

1. **Asymmetric info 这个 trick 真的"work"**: Table 3 里去掉 training-time info 掉 10%，是 ablation 里最 dramatic 的一个。说明 credit assignment 的瓶颈确实在 critic 的"视野"。

2. **Advantage parameterization 借力 LM head**: 直接学 advantage 而不是 value，复用 LM head 而不是加新 head，与 pre-training 对齐。这思路跟 DPO 哲学一致，但在 multi-turn agent 上是首次系统化。

3. **Llama-3.1-8B 打平 GPT-4o**: 对开源社区是个 strong result。

4. **Lemma B.1 / B.2 给出理论支撑**: trajectory reward = sum of advantages ( telescoping 证明) 和 asymmetric critic 的 unbiasedness 证明，让方法不只是 empirically work。

**可能的问题**:

1. **Lemma B.1 假设 deterministic transition**: 实际 human simulator (LLM) 有随机性，这个假设多 hard? 随机下等价性还成立吗? paper 没讨论。

2. **只在 artifact creation 上验证**: ColBench 两条 task 都是产出具体 artifact 的。web navigation、tool use 这类更开放的任务上 SWEET-RL 还 work 吗?

3. **只做 offline**: on-policy 设定下 asymmetric critic 还能保持 unbiased 吗? paper 没测。但 on-policy 对 agent 任务 collect data 代价高，offline 设定可能正是合适的选择。

4. **Human simulator 过于理想**: ColBench 的"人"能看到 reference，比真人聪明。这个 gap 在真实部署时有多大?

**直觉上的 takeaway**: 这篇 paper 提供的 recipe 其实很通用——任何 multi-turn agent 任务，只要有某种"训练时 privileged info" (reference solution / expert demonstration / lookahead 结果)，都可以套 SWEET-RL 的框架。Advantage parameterization 和 length normalization 这两个细节对所有 LLM RL 工作都有参考价值。

代码: [github.com/facebookresearch/sweet_rl](https://github.com/facebookresearch/sweet_rl)
数据: [huggingface.co/datasets/facebook/collaborative_agent_bench](https://huggingface.co/datasets/facebook/collaborative_agent_bench)

相关值得读的 follow-up:
- [Step-KTO](https://arxiv.org/abs/2501.10799) (同作者群，stepwise binary feedback)
- [IRPO](https://arxiv.org/abs/2404.19733) (iterative reasoning preference optimization)
- [Free process rewards](https://arxiv.org/abs/2412.01981) (无 process label 训练 PRM)
- [ARCHER](https://arxiv.org/abs/2402.19446) (hierarchical multi-turn RL)

---

# SWEET-RL: 带有 Training-Time Information 的 Multi-Turn LLM Agent 训练

## 1. Paper 整体定位

这篇 paper 来自 FAIR at Meta 和 UC Berkeley，作者 Yifei Zhou 等人，发表于 2025 年 3 月。核心贡献有两块:

- **ColBench**: 一个专门为 multi-turn RL 算法设计的 benchmark，聚焦于人类-LLM 协作完成 artifact creation (后端代码 + 前端网页)
- **SWEET-RL** (RL with Step-WisE Evaluation from Training-Time Information): 一个 asymmetric actor-critic 结构的 RL 算法，critic 在训练时能 access actor 看不到的 training-time information (比如 reference solution)，并通过 Bradley-Terry objective 直接学 advantage function

paper 想解决的根本问题是: 在 multi-turn LLM agent 任务上，如何做有效的 credit assignment，同时不破坏 LLM 预训练带来的 reasoning/generalization 能力。

相关背景工作可以参考:
- DPO: [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- PPO: [arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
- ARCHER (hierarchical multi-turn RL): [arxiv.org/abs/2402.19446](https://arxiv.org/abs/2402.19446)
- Asymmetric actor-critic in robotics: [arxiv.org/abs/1710.06542](https://arxiv.org/abs/1710.06542)

---

## 2. 为什么需要一个新的 Benchmark: ColBench

### 2.1 现有 benchmark 的不足

Table 1 列出了现有 LLM agent benchmark 与 ColBench 的对比，三个核心 criterion:

| Criterion | 含义 |
|-----------|------|
| **RL** | 是否支持 multi-turn RL 训练 (有 training set，能跑 RL 算法) |
| **Complex Reasoning** | 是否挑战 reasoning 和 generalization 能力 |
| **Min Overhead** | 是否 minimum engineering overhead 用于 fast prototyping |

WebArena ([arxiv.org/abs/2307.13854](https://arxiv.org/abs/2307.13854))、SWE-Bench ([arxiv.org/abs/2310.06770](https://arxiv.org/abs/2310.06770)) 这类 benchmark 有 complexity 但 overhead 高，需要模拟真实环境。LMRL Gym ([arxiv.org/abs/2311.18232](https://arxiv.org/abs/2311.18232)) 和 RL4VLM ([arxiv.org/abs/2405.10292](https://arxiv.org/abs/2405.10292)) 支持 RL 但 reasoning 要求低。

ColBench 的设计哲学是: **agent 与 human simulator 多轮交互，最终要产出一个具体的 artifact (code 或 HTML)**，把任务的 success 直接对应到一个可客观评估的产物上。

### 2.2 ColBench 的两条任务线

**(a) Backend Programming**:
- Agent 与 human simulator 交互最多 10 轮，写一个 ≤50 行的 Python 函数
- 初始只给函数 signature 和高层描述
- Human simulator (LLM) 能看到 reference code，自然语言回答 agent 的 clarification 问题，但不写代码
- 评估: 10 个 hidden unit tests，每个 0/1 reward，只在 episode 结束时给出

**(b) Frontend Design**:
- Agent 与 human simulator 交互最多 10 轮，写一个 ~100 行的 HTML snippet
- Human simulator 是 VLM (Qwen2-VL-72B)，能看到 agent 渲染出的网页与 reference 网页，描述差异
- 评估: CLIP ([arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)) embedding 的 cosine similarity (Clip-vit-base-patch32)，范围 [0, 1]

数据规模: Backend 10k train / 1k test; Frontend 10k train / 500 test。Offline trajectories: Backend 15k, Frontend 6k，由 Llama-3.1-8B 作为 agent、Llama-3.1-70B 或 Qwen2-VL-72B 作为 human simulator 生成。

### 2.3 关键 trick: 用 reference artifact 作为 human simulator 的"作弊"信息

这是 ColBench 设计上很巧的一点: human simulator 不是真正模拟人，而是 LLM **access 到 ground-truth artifact**，回答 agent 的问题。这把"模拟人"的不确定性大大降低，同时让评估可以自动化 (代码跑 unit test; 网页用 CLIP cosine similarity)。代价是这个"人"太理想化了，但 paper 认为 minimum engineering overhead 是值得的 trade-off。

---

## 3. SWEET-RL 的核心 idea

### 3.1 POMDP 形式化

paper 把协作任务建模成 finite-horizon POMDP:

$$\mathcal{M} = \{\mathcal{O}, \mathcal{C}, \mathcal{A}, \mathcal{T}, \mu_1, \mathcal{R}, N\}$$

| 符号 | 含义 |
|------|------|
| $\mathcal{O}$ | observable state space (agent 能看到的交互历史) |
| $\mathcal{C}$ | hidden 部分 (training-time information，比如 reference solution) |
| $\mathcal{A}$ | action space (token 序列) |
| $\mathcal{T}$ | transition function |
| $\mu_1$ | initial state distribution (抽 $o_1 \in \mathcal{O}$ 和 $c \in \mathcal{C}$) |
| $\mathcal{R}$ | reward space |
| $N$ | 最大 turn 数 |

关键点: $c$ 在整个 episode 中保持不变。actor 永远看不到 $c$，critic 在训练时能看到 $c$。

每 turn $t$，agent 观测 $o_t$ (含全部历史)，输出 token 序列 $a_t^{1:L}$，得到标量 reward $r(o_t, a_t, c) \in \mathcal{R}$。目标最大化 $\sum_{t=1}^{N} r(o_t, a_t, c)$。

### 3.2 三个核心 RL 量

$$Q^{\pi}(o_t, a_t, c) = \mathbb{E}_{\pi}\left[\sum_{t'=t}^{N} r(o_{t'}, a_{t'}, c)\right]$$

- $Q^{\pi}$: 在状态 $(o_t, c)$ 下采取 $a_t$，之后按 $\pi$ 走，期望累积 reward
- 上标 $\pi$ 表示 "在策略 $\pi$ 下"，下标 $t$ 是时间步

$$V^{\pi}(o_t, c) = \mathbb{E}_{a_t \sim \pi}[Q^{\pi}(o_t, a_t, c)]$$

- $V^{\pi}$: 状态 value，对 $a_t$ 在 $\pi$ 下取期望

$$A^{\pi}(o_t, a_t, c) = Q^{\pi}(o_t, a_t, c) - V^{\pi}(o_t, c)$$

- $A^{\pi}$: advantage function，衡量 action $a_t$ 相对策略平均水平的额外收益

paper 的核心 claim: **直接学 $A^{\pi}$ 比先学 $V^{\pi}$ 再减出 $A^{\pi}$ 更好**，因为预测 expected return 本身是个难任务，与 LLM 的 next-token prediction 预训练目标不一致，泛化差。这个观察在 Figure 3(a) 的 Best-of-N 曲线和 Table 3 的 ablation 中得到验证。

### 3.3 Asymmetric Actor-Critic

这是 SWEET-RL 最关键的 insight。传统 actor-critic 中，critic 和 actor 看同样的 observation。SWEET-RL 让 critic 在训练时多看到 $c$ (reference solution)。

直觉: 比如 backend programming 任务，agent 在 turn 2 问了一个 clarification 问题。要判断这个问题问得好不好，理想情况下需要知道 reference code 是什么样的——如果问的问题恰好 reference code 里有对应关键信息，说明 agent 在正确的方向上挖掘。Actor 看不到 reference，所以这种判断只能交给一个 privileged critic 在训练时完成。这相当于给 critic 一个"作弊视角"，让 credit assignment 变得 tractable。

类似 idea 在 robotics sim-to-real 里有过 ([arxiv.org/abs/1710.06542](https://arxiv.org/abs/1710.06542))，critic 看 latent state，actor 看 RGB。SWEET-RL 把这个思路搬到了 reasoning-heavy 的 LLM agent 上。

---

## 4. 两阶段训练流程 (Figure 2 详解)

### Stage 1: 训练 turn-wise advantage function $A_{\theta}$

给定同一个 task 下的两条 trajectory $\tau^+$ (chosen) 和 $\tau^-$ (rejected)，由 cumulative reward 决定。用 Bradley-Terry ([arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)) objective:

$$\mathcal{L}_{BT} = -\log\left[\sigma\left(\sum_t \beta r(o_t^+, a_t^+, c) - \sum_t \beta r(o_t^-, a_t^-, c)\right)\right] \quad (1)$$

变量解释:
- $\sigma(\cdot)$: sigmoid 函数
- $\beta$: 温度/hyperparameter 控制奖励尺度
- $o_t^+, a_t^+$: 来自 $\tau^+$ 在 turn $t$ 的观测和 action
- $o_t^-, a_t^-$: 来自 $\tau^-$ 在 turn $t$ 的观测和 action
- $c$: training-time information (reference solution)

把 reward 替换成 advantage:

$$\mathcal{L}_A(\theta) = -\log\left[\sigma\left(\sum_t \beta A_{\theta}(o_t^+, a_t^+, c) - \sum_t \beta A_{\theta}(o_t^-, a_t^-, c)\right)\right] \quad (2)$$

效果: 让 chosen trajectory 里每个 turn 的 advantage 升高，rejected trajectory 里每个 turn 的 advantage 降低。

### Stage 1 的 advantage parameterization (Equation 3)

这是 SWEET-RL 最有"美感"的设计之一:

$$A_{\theta}(o_t, a_t, c) = \frac{1}{L} \sum_{l=1}^{L} \left[\log \frac{\pi_{\theta}(a_t^l | o_t, a_t^{1:l-1}, c)}{\pi_{\text{ref}}(a_t^l | o_t, a_t^{1:l-1}, c)}\right] \quad (3)$$

逐项解释:
- $a_t^{1:L}$: 第 $t$ turn 的 action (长度 $L$ 的 token 序列)
- $a_t^l$: 第 $t$ turn 的第 $l$ 个 token
- $a_t^{1:l-1}$: 前 $l-1$ 个 token (上下文)
- $\pi_{\theta}$: 被训练的 LLM (作为 advantage function)
- $\pi_{\text{ref}}$: frozen 的 seed model (类似 DPO 里的 reference policy)
- $\frac{1}{L}$: **关键 normalization**，除以 response 长度

这个 parameterization 的妙处:
1. 完全复用 LLM 的 language modeling head，不引入新的 task-specific head，与 pre-training objective 对齐
2. 形式上类似 DPO 的 implicit reward $r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$，但 SWEET-RL 对 turn 内 token 做 mean (除 $L$)
3. Normalization by $L$ 防止 actor 学到"输出越短越好"的 degenerate 策略 (Table 3 的 "w/o Normalization" ablation: success rate 直接掉到 3.6%)

### Stage 2: 用 advantage 作为 turn-level reward 训练 actor $\pi_{\phi}$

actor 不看 $c$，只看 $o_t$。在每个 turn 用当前 policy 采样 16 个 candidate actions，用 $A_{\theta}$ 打分，取 top-50% quantile 作 chosen，bottom-50% quantile 作 rejected，跑 DPO:

$$\mathcal{L}_{\pi}(\phi) = -\log \sigma\left(\beta' \frac{\log \pi_{\phi}(a^+ | o_t)}{\log \pi_{\text{ref}}(a^+ | o_t)} - \beta' \frac{\log \pi_{\phi}(a^- | o_t)}{\log \pi_{\text{ref}}(a^- | o_t)}\right) \quad (4)$$

注意 Eq (4) 的 $\pi_{\phi}$ 和 $\pi_{\text{ref}}$ 都不看 $c$，因为 actor 部署时看不到 reference。Stage 2 不需要任何 human 交互，纯离线。

---

## 5. 理论分析 (Appendix B)

### 5.1 Lemma B.1: Trajectory reward = Sum of advantages

**陈述**: 假设 transition $\mathcal{T}(o, a, c)$ 是 deterministic 的，则:

$$\sum_t r(o_t, a_t, c) = \sum_t A^{\pi}(o_t, a_t, c)$$

**证明** (telescoping):

$$
\begin{aligned}
\sum_t A^{\pi}(o_t, a_t, c) &= \sum_t [Q^{\pi}(o_t, a_t, c) - V^{\pi}(o_t, c)] \\
&= \sum_{t=1}^{N-1} [r(o_t, a_t, c) + \mathbb{E}_{a_{t+1}' \sim \mathcal{T}(\cdot|o_t, a_t, c)} V^{\pi}(o_{t+1}', c) - V^{\pi}(o_t, c)] + r(o_N, a_N, c) \\
&= \sum_{t=1}^{N} r(o_t, a_t, c) + \sum_{t=1}^{N-1} [\mathbb{E}_{a_{t+1}'} V^{\pi}(o_{t+1}', c) - V^{\pi}(o_t, c)]
\end{aligned}
$$

deterministic transition 下，$\mathbb{E}_{a_{t+1}'} V^{\pi}(o_{t+1}', c) = V^{\pi}(o_{t+1}, c)$，所以中间项 telescoping 求和为 0:

$$V^{\pi}(o_2, c) - V^{\pi}(o_1, c) + V^{\pi}(o_3, c) - V^{\pi}(o_2, c) + \cdots = V^{\pi}(o_N, c) - V^{\pi}(o_1, c)$$

但这里有个细节: 在最右端 $t=N$ 时只有 $r(o_N, a_N, c)$ 而没有 $V^{\pi}(o_{N+1}, c)$，所以 $V^{\pi}(o_{N+1}, c)$ 应该等于 0 (terminal state)，最终 telescoping 得到 $\sum_t r$。

**直觉**: 这告诉我们 trajectory-level BT objective 和 advantage-level BT objective 等价。所以直接学 advantage 不损失信息。

### 5.2 Lemma B.2: Asymmetric critic 给出 unbiased policy gradient

**陈述**: 下面两个 estimator 都是 $\nabla \mathbb{E}_{\tau \sim \pi}[\sum_t r(o_t, a_t, c)]$ 的无偏估计:

$$\sum_{t=1}^{N} \mathbb{E}_{o_t, a_t} A^{\pi}(o_t, a_t) \nabla \log \pi(a_t | o_t) = \sum_{t=1}^{N} \mathbb{E}_{o_t, a_t, c} A^{\pi}(o_t, a_t, c) \nabla \log \pi(a_t | o_t)$$

**证明核心步骤**: 关键是这一步

$$\mathbb{E}_{o_t, a_t} V^{\pi}(o_t) \nabla \log \pi(a_t | o_t) = \mathbb{E}_{o_t} \sum_a \pi(a_t | o_t) V^{\pi}(o_t) \nabla \log \pi(a_t | o_t) = \mathbb{E}_{o_t} V^{\pi}(o_t) \nabla \sum_a \pi(a_t | o_t) = 0$$

因为 $\sum_a \pi(a_t | o_t) = 1$，梯度为 0。这意味着 baseline $V^{\pi}$ 不影响 gradient 期望，所以 $Q^{\pi}$ 可以替换成 $A^{\pi} = Q^{\pi} - V^{\pi}$。

再利用条件期望:

$$\mathbb{E}_{c \sim d_t^{\pi}(\cdot | o_t, a_t)} A^{\pi}(o_t, a_t, c) = A^{\pi}(o_t, a_t)$$

即对 $c$ 求条件期望就回到普通 advantage。所以多看到 $c$ 不引入 bias，只是降低 variance。这非常关键，因为说明 SWEET-RL 的 asymmetric critic 在期望意义下不偏离真正的 policy gradient，只是单样本上估计更准。

---

## 6. 实验详解

### 6.1 主实验 (Table 2)

| Setting | Model | Backend Success Rate | Frontend Win Rate |
|---------|-------|----------------------|-------------------|
| Single-turn | GPT-4o | 16.2 | 23.8 |
| Multi-turn (zero-shot) | Llama-3.1-8B | 22.4 | 33.8 |
| Multi-turn (zero-shot) | GPT-4o | 40.4 | 50.0 |
| Multi-turn (zero-shot) | Llama-3.1-70B | 35.0 | 39.8 |
| RFT | Llama-3.1-8B | 28.2 | 38.6 |
| Multi-Turn DPO | Llama-3.1-8B | 34.4 | 42.8 |
| **SWEET-RL** | **Llama-3.1-8B** | **40.4** | **48.2** |

几个关键 takeaways:

1. **Single-turn → Multi-turn 的增益巨大**: GPT-4o 在 backend 上从 16.2% 跳到 40.4%，证明协作本身能解锁大量性能。
2. **Llama-3.1-8B + SWEET-RL ≈ GPT-4o**: 40.4 vs 40.4 (backend), 48.2 vs 50.0 (frontend)。8B 模型经过 SWEET-RL fine-tune 能逼近 GPT-4o 这种 proprietary 模型。
3. **SWEET-RL > Multi-Turn DPO 约 6% absolute**: 在 backend 上 40.4 vs 34.4，frontend 上 48.2 vs 42.8。这 6% 就是 explicit credit assignment + asymmetric critic 带来的增益。
4. **o1-mini 在协作任务上没明显优势**: zero-shot 协作下 o1-mini (43.2 backend) 反而比 GPT-4o (54.6 tests passed) 差。说明纯 reasoning 能力不直接转化为 multi-turn 协作策略，需要 task-specific fine-tuning。

### 6.2 Ablation (Table 3) - critic 设计选择

| Variant | % Tests Passed | Success Rate |
|---------|----------------|--------------|
| SWEET-RL | 56.8 | 40.4 |
| w/ Regression Head | 45.3 | 36.2 |
| w/o Train-Time Info | 44.0 | 31.2 |
| w/o Normalization | 4.2 | 3.6 |

三个关键 finding:

1. **Regression head 比 mean log prob parameterization 差**: 在 LLM backbone 上加一个 classification head 预测 expected success rate (SWEET-RL w/ Value Function) 效果显著差。原因: 这个 objective 与 pre-training 不对齐，泛化差。
2. **Training-time info 至关重要**: 去掉 $c$ 后，SWEET-RL 退化到比 Multi-Turn DPO (34.4) 略差的 31.2，说明 $c$ 是主要增益来源。
3. **Length normalization 防止 collapse**: 去掉 $\frac{1}{L}$ 后，actor 学到输出越来越短的 degenerate 策略，success rate 从 40.4 暴跌到 3.6。这是 LLM RL 训练里非常经典的坑。

### 6.3 Best-of-N scaling (Figure 3a)

Figure 3a 展示在固定 actor (Llama-3.1-8B) 上用不同 step reward model 做 Best-of-N 采样的成功率随 N 的变化:

- **SWEET-RL**: 显著最好，曲线最陡
- **SWEET-RL w/o Training-Time Info**: 大幅下降，几乎没增益
- **LLM-as-a-Judge** (Llama-3.1-8B pairwise compare): 增益有限，因为容易被长度和格式 distraction
- **SWEET-RL w/ Value Function**: scaling 较差，证明 value head 在 unseen task 上泛化失败

这个图直接说明 SWEET-RL 的核心优势: 在 turn level 上判别 action 好坏的能力。

### 6.4 Data scaling (Figure 3b)

随着 fine-tuning data 量增加:

- 3k 数据时: SWEET-RL 反而比 Multi-Turn DPO 差，因为 critic 还没训练好
- 数据增加后: SWEET-RL 快速追赶并显著超越
- 收敛性能: SWEET-RL 最好

这是一个值得注意的 trade-off: SWEET-RL 有更高的 sample complexity threshold，但渐近性能更好。这有点像 "先付一笔 critic 训练的固定成本，之后享受更好的 credit assignment 收益"。

### 6.5 强 base model + off-policy data (Table 4)

用 Llama-3.1-70B 作为 base model，但 offline data 仍由 Llama-3.1-8B 生成:

| Method | Success Rate |
|--------|--------------|
| Zero-shot | 35.0 |
| RFT | 31.9 (下降!) |
| Multi-Turn DPO | 41.8 |
| SWEET-RL | 45.6 |

RFT 失败的原因很有意思: 它强制 70B 模型 word-by-word 模仿 8B 模型生成的 sub-optimal trajectory，相当于让强模型"装弱"，反而退化。SWEET-RL 因为做 explicit credit assignment，能识别 8B trajectory 中哪些 turn 是好的，避免全盘照搬。

---

## 7. Qualitative 例子分析 (Appendix D, E)

### 7.1 Figure 6 - 不同 credit assignment 方法的对比

在 backend 任务 turn 2，agent 问了两个不同的 clarification 问题:

- **LLM-as-a-Judge**: 被长格式 distraction，给了更长更"详细"的回答高分
- **Value Function (regression head)**: 给第一个回答 97% success probability，但这只是 turn 2/10，根本不可能预测这么准。泛化失败。
- **SWEET-RL**: 正确识别第二个回答 advantage 更高，因为它问到了"返回的 list 是否允许 duplicate objects"这个关键问题。

### 7.2 Figure 7-8 - SWEET-RL Llama-3.1-8B 的完整 trajectory

观察到一个 emergent behavior: agent 在 turn 5-6 还在做 chain-of-thought reasoning 和 self-correction。这说明 RL 训练不只是学到"问问题"这个 surface behavior，还激发了更深层的 reasoning pattern。

### 7.3 Figure 9-10 - Zero-shot baseline 的失败模式

Llama-3.1-8B zero-shot 和 GPT-4o zero-shot 都有一个共同问题: 问了 1-2 个问题就跳到最终答案。它们能识别出该问什么，但没有"等收集够信息再答"的策略。

### 7.4 Figure 11-16 - Frontend Design 的长 trajectory

这个任务难度极高: agent 要处理 up to 16k tokens 的 HTML context。SWEET-RL 训练后的 agent 学到一个聪明策略: 先给一个 scratch solution 探探 feedback，最后才做精细 edit。这是 reward shaping 自然涌现的行为。

---

## 8. 与 Process Reward Model (PRM) 的关系

paper Section 2 专门讨论了与 PRM 的区别:

- **PRM** ([arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050), [arxiv.org/abs/2410.08146](https://arxiv.org/abs/2410.08146)): 评估每个 reasoning step的"正确性"，通常用于 test-time search 或加速 on-policy RL exploration
- **SWEET-RL 的 critic**: 主要用作 credit assignment 的 intermediate reward proxy，直接优化 policy，不需要再 collect 交互数据

这个区别很实际: LLM agent 任务里 collect on-policy data 涉及与外部环境 (人) 交互，代价极高，所以 SWEET-RL 的"训练时用 critic，部署时丢掉 critic"的设计非常合适。

---

## 9. 关键 Insights 和我的解读

### 9.1 Asymmetric information 是 credit assignment 的"作弊"

在 standard RL 里，credit assignment 难是因为 critic 和 actor 看到一样的东西，critic 没有"上帝视角"。SWEET-RL 通过让 critic 看 reference solution，本质上把 credit assignment 变成了 supervised learning 问题: 给定 reference，判断当前 action 是否在向 reference 靠拢。这绕过了 long-horizon credit propagation 的难题。

代价: 部署时 actor 没有 $c$，但 Lemma B.2 证明在期望意义下 gradient 仍无偏。所以训练时借力，部署时不依赖，是一个干净的 trick。

### 9.2 直接学 advantage 而非 value

传统 RL 学 $V$，然后 $A = Q - V$。但预测 expected return 对 LLM 来说是个 foreign task，需要新 head，泛化差。SWEET-RL 直接学 $A$，并用 mean log probability parameterize，完全复用 LM head。这把"学一个新东西"转化成"用现有 LM 能力做相对判断"，更容易。

### 9.3 Bradley-Terry 在 trajectory 和 turn level 的等价性

Lemma B.1 的 telescoping 论证告诉我们: trajectory-level preference 隐含 turn-level advantage preference。所以可以从 trajectory-level 标注 (容易获得，只要跑 rollout 看 final reward) 推出 turn-level 监督信号，避免人工 step-level 标注。这是 data efficiency 的关键。

### 9.4 Length normalization 防止 degenerate solution

这个 ablation (Table 3) 的结果非常戏剧化 (40.4 → 3.6)，说明 LLM RL 训练里小小的设计选择可能致命。除以 $L$ 后，advantage 不再单调依赖于 response 长度，避免了 actor 走 shortcut。

### 9.5 RL 比 imitation (RFT) 更适合 off-policy 强 base model

Table 4 的 RFT 失败案例很有启发性: 当 offline data 来自弱于 base model 的 policy 时，SWEET-RL 和 Multi-Turn DPO 这种做 credit assignment / negative gradient 的方法胜过纯模仿。这对实际部署很重要——你可以用便宜的小模型生成数据训练大模型。

---

## 10. 一些可以深挖的方向

1. **Non-deterministic transition**: Lemma B.1 假设 deterministic transition，但实际 human simulator 有随机性。这个 assumption 多关键? stochastic 下等价性还成立吗?

2. **Online SWEET-RL**: paper 只做 offline。online 设定下，asymmetric critic 还能保持 unbiased 吗? on-policy data 是否会让 critic 更准? 结合 RAFT ([arxiv.org/abs/2304.06767](https://arxiv.org/abs/2304.06767)) 或 PPO 怎么样?

3. **Critic 和 actor 共享 backbone**: 当前 critic 是单独的 LM。能不能用同一个 LM 的不同 head? 共享 backbone 会节省显存但可能引入 instability。

4. **Beyond reference solution**: training-time info 不止 reference solution。可以是 human preference labels，可以是 expert demonstration，甚至可以是 lookahead search 的结果。SWEET-RL 框架可以推广。

5. **Multi-agent extension**: 如果有多个 agent 协作，每个 agent 有不同的 privileged information，SWEET-RL 的 asymmetric critic 思路可以自然推广到 multi-agent credit assignment。

---

## 11. 总结

SWEET-RL 是一个**结合了几个已知 idea 但组合得很巧妙的工作**:
- Asymmetric actor-critic (来自 robotics sim-to-real)
- Bradley-Terry / DPO (来自 RLHF)
- Direct advantage learning (来自 RL theory)
- Mean log probability parameterization (类似 DPO implicit reward)

把它们放在一起，加上 ColBench 这个合适的 benchmark，得到了一个在 multi-turn LLM agent 上 SOTA 的算法。Llama-3.1-8B 经过 SWEET-RL 训练后能 match GPT-4o，这个结果对开源社区很有价值。

paper 也明确指出了 limitations: 只在 artifact creation 任务上验证，更广泛的 agent 任务 (web navigation, tool use) 还有待验证。但作为一个 method paper，SWEET-RL 给出的 recipe (asymmetric info + advantage learning + LM-head parameterization) 应该是可迁移的。

相关后续工作可以关注:
- ARCHER: [arxiv.org/abs/2402.19446](https://arxiv.org/abs/2402.19446) (hierarchical multi-turn RL)
- Step-KTO: [arxiv.org/abs/2501.10799](https://arxiv.org/abs/2501.10799) (stepwise binary feedback, 同作者群)
- IRPO: [arxiv.org/abs/2404.19733](https://arxiv.org/abs/2404.19733) (iterative reasoning preference optimization)
- Free process rewards: [arxiv.org/abs/2412.01981](https://arxiv.org/abs/2412.01981) (无 process label 训练 PRM)

代码开源在 [github.com/facebookresearch/sweet_rl](https://github.com/facebookresearch/sweet_rl)，数据在 [huggingface.co/datasets/facebook/collaborative_agent_bench](https://huggingface.co/datasets/facebook/collaborative_agent_bench)。建议直接跑一下 Backend Programming 这个 task，相对容易复现，能直观感受 asymmetric critic 的威力。
