---
source_pdf: Do As I Can, Not As I Say.pdf
paper_sha256: 68ce624b1df1f3c7479cb13953e18dd228bb699cc271005b5e27f2c79f2af153
processed_at: '2026-08-18T06:18:21-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SayCan 用人话讲

## 核心故事

你有一个 robot 在厨房里。你跟它说 "我洒了饮料，能帮我吗？"

如果你直接问 GPT，它会告诉你 "用吸尘器清理" 或者 "叫清洁工来"。这些回答在文字世界完全合理，但你的 robot 根本没有吸尘器这个 skill，它只会 "pick up sponge"、"go to table"、"put down object" 这些 basic 动作。

问题出在哪？LLM 脑子里装了一堆 semantic knowledge（"洒了饮料应该用海绵擦"），但它从来没在 physical world 里待过，不知道 robot 当前长什么样、能干什么、眼前有什么东西。

SayCan 的解法特别简单粗暴：**把 "这个 skill 对当前 task 有用吗" 和 "这个 skill 在当前环境能成功吗" 两个概率乘起来，选乘积最大的那个 skill 执行**。

就这么一句话。剩下的都是 implementation detail。

## 两个概率各管各的

### Say 部分：LLM 管 "该做什么"

你给 LLM 一个 instruction "我洒了饮料，能帮我吗？"，然后问它：在所有 candidate skills 里，"pick up sponge" 这个 skill 作为下一步的 likelihood 是多少？

LLM 用 scoring 模式（output probability）而非 generative 模式（generate text）来回答。你把所有 skill description 列出来：

- find a sponge
- pick up the sponge
- go to the table
- pick up the apple
- pick up the coke can
- ...

对每个 skill，LLM 给一个 probability。"pick up sponge" 在 "洒了饮料" 的 context 下 probability 会高，"pick up apple" 会低。

这里有个细节：为什么用 scoring 而不是让 LLM 直接生成？因为 LLM 生成出来的东西可能不在你的 skill list 里（"vacuum the floor"），可能格式乱了 parse 不出来，而且 generative mode 没有给你每个 option 的 explicit probability，没法跟 affordance 相乘。Scoring 模式天然给你一个 categorical distribution over skills，直接可以逐元素乘。

### Can 部分：Value Function 管 "能不能做"

光知道 "该 pick up sponge" 还不够。如果当前 scene 里根本没有 sponge，或者 robot 离 sponge 太远抓不到，这个 skill 执行了也会失败。

你怎么知道 robot 能不能成功执行一个 skill？作者发现了一个特别漂亮的 connection：**在 sparse reward 下（成功=1, 失败=0），RL 的 value function 恰好等于 "从当前 state 出发执行这个 policy 能成功的概率"**。

数学上很直接。Value function 定义：

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_t \gamma^t R(s_t, a_t) \mid s_0 = s \right]$$

变量：
- $V^\pi(s)$: 从 state $s$ 出发，执行 policy $\pi$ 的 expected discounted return
- $\gamma$: discount factor，如果设为 1 就是 undiscounted
- $R(s_t, a_t)$: step $t$ 的 reward
- $s_0 = s$: 初始 state

如果 reward 只在 episode 结束时给，成功=1，失败=0，且 $\gamma=1$：

$$V^\pi(s) = P(\text{success} \mid s, \pi) = p(c_\pi \mid s, \ell_\pi)$$

这就是 Gibson 说的 **affordance**——环境提供的 action possibility。SayCan 用 modern RL 的 value function 把这个 1977 年的 ecological psychology concept 给 instantiate 了。

所以 robot 每观察到一个新 state，就把所有 skill 的 value function 都算一遍，得到一个 "每个 skill 在当前 state 能成功的概率" 的 vector。Figure 2 展示了这个 value function space——当 scene 里有 apple 和 redbull 时，"pick up apple" 和 "pick up redbull" 的 value 高，其他 pick skill 的 value 低。

### 乘起来：Combined Probability

最终选择 skill 的公式：

$$\pi^* = \arg\max_{\pi \in \Pi} \underbrace{p(c_\pi \mid s, \ell_\pi)}_{\text{Can}} \cdot \underbrace{p(\ell_\pi \mid i)}_{\text{Say}}$$

变量：
- $\pi^*$: 最终选择的 skill
- $\Pi$: 所有可用 skills 的集合
- $p(c_\pi \mid s, \ell_\pi)$: skill $\pi$ 在 state $s$ 下能成功的概率（来自 value function）
- $p(\ell_\pi \mid i)$: skill $\pi$ 的 language description $\ell_\pi$ 对 instruction $i$ 有用的概率（来自 LLM）

直觉：你想要一个 skill 既 "有用"（LLM 说该做）又 "能做"（value function 说能成功）。任何一个为 0，乘积就是 0。

### 为什么这个分解成立？

作者做了一个假设：skill 成功了才有贡献，失败了贡献为 0。所以：

$$p(c_i \mid i, s, \ell_\pi) = p(c_\pi \mid s, \ell_\pi) \cdot p(\ell_\pi \mid i) + (1 - p(c_\pi \mid s, \ell_\pi)) \cdot 0$$

- $p(c_i \mid i, s, \ell_\pi)$: skill $\pi$ 真正能推进 instruction $i$ 的概率
- 第一项：skill 成功的概率 × 成功时对 task 有用的概率
- 第二项：skill 失败的概率 × 失败时对 task 的贡献（=0）

化简就是 $p(c_\pi \mid s, \ell_\pi) \cdot p(\ell_\pi \mid i)$。

这个分解的 power 在于：**两个因子完全独立，可以各自 scale**。LLM 可以换更大的（8B→62B→540B），value function 可以收集更多 data 训练，两边互不干扰。

## Iterative Planning：一步步来

选了一个 skill 之后，把它执行了，然后把选过的 skill description append 到 LLM 的 context 里，再问一遍 "下一个该做什么"。

比如 instruction "bring me a coke can"：

```
Step 1: LLM context = "bring me a coke can"
        → 选 "find a coke can"（LLM 高分 + affordance 高因为 scene 里有 coke）

Step 2: LLM context = "bring me a coke can. 1. find a coke can"
        → 选 "pick up the coke can"（LLM 理解 "找到之后该 pick up"，affordance 高因为 robot 在 coke 旁边）

Step 3: LLM context = "bring me a coke can. 1. find a coke can, 2. pick up the coke can"
        → 选 "bring it to you"（LLM 理解 "pick up 之后该 bring"，affordance 高因为 robot 手里有 coke）

Step 4: LLM context = "...1. find, 2. pick up, 3. bring it to you"
        → 选 "done"（LLM 理解任务完成）
```

这是一个 **autoregressive planning** 过程，跟 LLM 生成 text 的 autoregressive decoding 一模一样，只是每个 token 变成了一个 skill。

LLM 为什么能理解 "find 之后该 pick up，pick up 之后该 bring"？因为它的 training data 里有大量人类描述任务序列的 text，这些 sequential pattern 已经被 encode 进 model weights 了。

## Prompt Engineering：教 LLM 乖乖输出

光给 LLM instruction 它可能乱来。作者用 prompt engineering 来 constrain LLM 的输出格式。Prompt 长这样（Listing 1 的简化版）：

```
Robot: Hi, I'm a robot in an office kitchen.
Human: How would you bring me an orange?
Robot: 1. find an orange, 2. pick up the orange, 3. bring it to you, 4. put down the orange, 5. done.
Human: How would you throw away a coffee cup?
Robot: 1. find a coffee cup, 2. pick up the coffee cup, 3. go to trash can, 4. put down the coffee cup, 5. done.
Human: [你的实际 instruction]
```

LLM 看到这些 example，就学会了 "用编号序列回答，每步是一个 skill，最后加 done"。

Table 5 做了 ablation：
- 0 个 example：plan rate 10%（要求 terminate）或 52%（不要求 terminate）
- 1 个 example：64%
- 4 个 example：82%
- 17 个 example（full prompt）：88%

说明 LLM 本身已经 encode 了一些 task decomposition knowledge（0 example 也有 52%），但 prompt engineering 能显著提升并 constrain 格式。

## 技术细节：Value Function 怎么训

### RL Policy（Figure 9, MT-Opt 风格）

架构：
```
Image 640×512 → 7 Conv Layers → FiLM(language + robot state) → 11 Conv Layers → Sigmoid → Q ∈ [0,1]
```

关键设计：
- **Sigmoid output**：因为 reward 是 binary，Q-value constrain 到 [0,1]，直接当 probability 用
- **Asynchronous control**：inference 时 robot 还在执行上一个 action，所以 model 接收 "上一个 action 剩多少没执行" 作为 input
- **Language conditioning**：用 Universal Sentence Encoder 的 frozen embedding
- **Log loss** 而非 MSE：因为 binary reward，log loss 更适合 probability calibration
- **Prioritized experience replay** [88]：episode priority = $1 + 10 \cdot |p - 0.5|$，$p$ 是 replay buffer 中各 skill 的平均成功率，tune 到接近 50% success rate 来 balance data

训练资源：16 TPUv3，100 小时，3000 CPU workers 收集 episodes，另外 3000 CPU workers 算 target Q-values（让 TPU 专门算 gradient）。

### BC Policy（Figure 10, BC-Z 风格）

架构：
```
Image 256×320 → ResNet-18 → FiLM(language) → FC → action components
```

- **Action components**：arm position (continuous), arm orientation (continuous), gripper (discrete), terminate (discrete)
- **Loss**：MSE for continuous, cross-entropy for discrete
- **Data**：68000 teleoperated demos（11 个月，10 个 robot）+ 120000 autonomous episodes（success-filtered）

一个有趣的细节：BC policy 用 256×320 image（for speed），但 affordance value function 用 640×512 full resolution（因为 low resolution 学不好 Q-function）。Policy 不需要那么高 resolution 是因为它只需要输出 action，而 value function 需要判断 "能不能成功"，需要更精细的 visual understanding。

### Affordance Calibration

Learned value function 不是 perfectly calibrated 的 probability。不同 skill family 用了不同 calibration：

**Pick**（learned VF）：
$$p_\text{pick}^\text{aff} = \text{clamp}\left(\frac{v^\text{pick} - 0.2}{0.5 - 0.2}, 0, 1\right)$$

$v^\text{pick}$ 是 value function 输出，0.2 和 0.5 是 empirically 确定的 min/max threshold。

**Go to**（distance-based heuristic）：
$$p_\text{goto}^\text{aff} = \text{clamp}\left(\frac{100 - d}{100 - 0}, 0, 1\right)$$

$d$ 是到目标 location 的距离（米）。纯 heuristic，但 work fine。

**Place**：$p = 1.0$（总是 possible，靠 LLM 理解 "place 只在 pick 之后"）

**Terminate**：$p = 0.1$（小常数，确保没 feasible skill 时能终止）

这说明 SayCan **不要求所有 affordance 都来自 learned value function**。任何能回答 "这个 skill 在当前 state 能成功吗" 的 function 都行。开源版本甚至用 ViLD object detector 当 affordance——检测到 scene 里有目标 object，就认为 pick affordance 高。

## 实验结果的核心 Insight

### Grounding 必要性（Table 2）

| 方法 | Plan Rate | 
|------|-----------|
| PaLM-SayCan（完整） | 84% |
| No VF（只有 LLM） | 67% |
| Generative（LLM 生成+投影） | 74% |
| BC NL（直接把 instruction 喂 policy） | 0% |

三个关键发现：

1. **没有 affordance grounding，LLM 只有 67%**。加上 affordance 到 84%。Grounding 贡献了 17 个百分点。

2. **直接把 high-level instruction 喂给 language-conditioned policy = 0%**。Policy 只懂 "pick up the apple" 这种 atomic command，不懂 "I spilled my drink" 这种 abstract instruction。这证明 LLM 的 semantic parsing 是 indispensable 的。

3. **Scoring > Generative**（84% > 74%）。Generative 方案丢失了 per-option probability，无法跟 affordance 相乘，也无法提供 interpretability。

### LLM Scaling 直接 translate 到 Robot（Table 3）

| LLM | Plan | Execute |
|-----|------|---------|
| PaLM 8B | 38% | - |
| PaLM 62B | 72% | - |
| PaLM 540B | 84% | 74% |
| FLAN 137B | 70% | 61% |

这是 paper 最 exciting 的 finding：**LLM 从 8B 到 540B，robot execute rate 从 ~38% 到 74%**。第一次看到 NLP scaling law 直接 benefit robotics。因为 SayCan 把 LLM 和 robot 解耦了，LLM 的进步可以 zero-shot 传导到 robot。

FLAN 137B 虽然 instruction-tuned，但反而比 PaLM 540B 差。大规模 pretraining > task-specific fine-tuning。这个 finding 在后来 LLM 研究中被反复验证。

### Long-Horizon 的挑战

| Family | Plan | Execute |
|--------|------|---------|
| Long-Horizon | 73% | 47% |

长 sequence 中任何一步失败都会导致整体失败。而且 LLM 倾向于 **early termination**——bring 了第一个 object 就说 done，忘了还有第二个。这是 open-loop planning 的固有问题，后来 Inner Monologue [25] 通过 closed-loop feedback 解决了一部分。

### Embodiment 测试

这组测试特别 clever：同一个 instruction "put the coke on the counter"，但从不同初始状态开始：
- robot 手里有 coke，在 counter 旁边
- robot 手里没 coke，在 table 旁边
- robot 手里没 coke，在 counter 旁边

Plan rate 只有 64%。失败主要来自 affordance 误判——value function 没能准确区分 "我已经拿着 coke 了" 和 "我手里没东西"。这说明 affordance model 还有很大提升空间。

## Chain-of-Thought：解决 Negation

原始 SayCan 不能处理 negation（"bring me something that isn't a fruit"）。因为 LLM 对 negation 的理解本身就有问题 [19]。

作者集成了 chain-of-thought [24]：让 LLM 先生成一段 explanation，再用 scoring mode 把 explanation 加进 context 来 score skills。

```
Human: Can you bring a fruit-flavored drink without caffeine?
Explanation: The user has asked for a drink that is fruit-flavored 
and does not have caffeine, I will bring the lime soda.
Robot: 1. find a lime soda, 2. pick up the lime soda, 3. bring it to you, 4. done
```

这是 generative + scoring 的 hybrid：generative 用来 reasoning，scoring 用来选 skill。后来这个 pattern 在 LLM agent 设计中非常常见。

## Multilingual：Free Lunch

PaLM 训练了 multilingual corpora，所以 SayCan 能 zero-shot 处理中文、法文、西班牙文：

| Instruction | Plan Rate |
|-------------|-----------|
| bring me a can of coke | 1.0 |
| 拿一罐可乐给我 | 1.0 |
| apporte moi une canette de coca | 1.0 |
| tráeme una lata de coca cola | 1.0 |

只有一句法文的长 instruction 失败了。说明 LLM 的 semantic knowledge 是 language-agnostic 的 representation，surface form 不影响 underlying reasoning。

## Adding Skills：Zero-shot 扩展

新增 drawer manipulation skills 只需要三步：
1. Candidate skill list 加 "open the drawer", "close the drawer"
2. 提供 affordance function（heuristic：robot 在 drawer 旁边就 affordance=1）
3. Prompt 加几个 drawer example

结果：plan rate 100%，execute rate 33%（drawer 物理操作难）。**对其他 instruction 零影响**。

这个 extensibility 是 SayCan modular design 的直接 benefit。后来 RT-1, RT-2 往 model 里塞更多 skills 也是类似思路，只是用 end-to-end learning 替代了 modular interface。

## 我的 Intuition 总结

SayCan 的 essence 可以浓缩成一个 mental model：

**你的 brain 有两个 system 协作。System 1（LLM）负责 semantic reasoning——"洒了饮料该用海绵"。System 2（value function）负责 physical assessment——"眼前有没有海绵、手能不能够到"。两个 system 各自给出一个 probability，乘起来就是 "这个 action 既合理又可行" 的 probability。**

这个 framework 的 beauty 在于：
- **Decoupling**：semantic 和 physical 独立 scale，LLM 换大的不用动 robot，robot 收 data 不用动 LLM
- **Interpretability**：每一步都能看到两个 system 各自的 reasoning（Figure 6）
- **Extensibility**：新 skill 只需加 description + affordance + prompt example
- **Zero-shot**：不需要 fine-tune LLM
- **Scaling law transfer**：LLM 变强 → robot 直接变强

更深的 lesson：**很多 AI 系统的瓶颈不在单一 component 多强，在于如何 decompose 问题让各 component 各司其职**。LLM 强在 semantic reasoning，弱在 physical grounding；RL policy 强在 physical control，弱在 high-level planning。用一个概率接口把它们 stitch 起来，1+1 > 2。

这个 lesson 在今天的 AI agent 设计中依然 relevant——不管是 AutoGPT 的 task decomposition，还是 ReAct 的 reasoning+acting 交替，还是 Toolformer 的 tool use，本质上都是在做类似的 decomposition + interface design。

## References

- SayCan 项目页: https://say-can.github.io
- SayCan arXiv: https://arxiv.org/abs/2204.01691
- PaLM: https://arxiv.org/abs/2204.02311
- BC-Z: https://proceedings.mlr.press/v164/jang22a.html
- MT-Opt: https://arxiv.org/abs/2104.08212
- Chain of Thought: https://arxiv.org/abs/2201.11903
- CLIPort: https://arxiv.org/abs/2109.12098
- ViLD: https://arxiv.org/abs/2104.13921
- Universal Sentence Encoder: https://arxiv.org/abs/1803.11175
- Inner Monologue: https://arxiv.org/abs/2207.05608
- Language Models as Zero-Shot Planners: https://arxiv.org/abs/2201.07207
- Value Function Spaces: https://arxiv.org/abs/2111.03189
- Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
- Code as Policies: https://arxiv.org/abs/2209.07794
- RT-2: https://arxiv.org/abs/2307.15818
- ReAct: https://arxiv.org/abs/2210.03629
- Toolformer: https://arxiv.org/abs/2302.04761
- Gibson affordances: Gibson, "The Ecological Approach to Visual Perception", 1979

---

# SayCan: Do As I Can, Not As I Say 深度解析

## 1. 核心 Motivation 和 Intuition

这篇 paper 解决的核心问题是：**如何将 LLM 中蕴含的 semantic knowledge 提取出来，用于 embodied agent 的 real-world 任务执行**。

LLM 的问题在于它 purely text-trained，缺乏 physical grounding。Bender & Koller 在 [1] 中论述过，language model 只是 form 的建模，并不真正理解 meaning。SayCan 的作者用了一个非常生动的例子来说明：如果你问一个 kitchen robot "我洒了饮料，能帮我吗？"，LLM 可能会回答 "可以用吸尘器清理" 或者 "我可以叫清洁工来"——这些回答在 text space 是 reasonable 的 completion，但在 physical world 完全 infeasible，因为 robot 根本没有吸尘器这个 skill。

这里的关键 insight 是：**LLM 知道"应该做什么"（task knowledge），但不知道"能做什么"（affordance）。如果能把这两者解耦并重新组合，就能让 LLM 的 knowledge 真正 land 到 physical world**。

这就像人类：你脑子里有 "我要喝水" 的高层意图（semantic knowledge），但你的手能不能 reach 到杯子、杯子是不是空的（affordance），这取决于你当前的身体状态和环境。意图和能力的乘积才决定你实际会执行什么动作。

## 2. 方法核心：Say × Can 的概率分解

### 2.1 概率分解的 intuition

SayCan 的核心 mathematical trick 是把 "skill $\pi$ 能完成 instruction $i$" 这个概率 $p(c_i | i, s, \ell_\pi)$ 分解成两个独立可计算的因子：

$$p(c_i | i, s, \ell_\pi) \propto \underbrace{p(c_\pi | s, \ell_\pi)}_{\text{world-grounding (Can)}} \cdot \underbrace{p(\ell_\pi | i)}_{\text{task-grounding (Say)}}$$

变量含义：
- $i$: user 提供的 high-level natural language instruction（如 "I spilled my drink, can you help?"）
- $s$: 当前世界的 state（robot 的 observation）
- $\pi \in \Pi$: 一个 skill（policy），来自 skill set $\Pi$
- $\ell_\pi$: skill $\pi$ 的 language description（如 "pick up the sponge"）
- $c_\pi$: Bernoulli random variable，表示 skill $\pi$ 是否成功完成
- $c_i$: 表示是否完成了 instruction $i$

**为什么可以这样分解？** 作者做了一个关键的假设：如果一个 skill 成功了（$c_\pi = 1$），那么它对 instruction 的贡献概率是 $p(\ell_\pi | i)$；如果 skill 失败了（$c_\pi = 0$），那么它对 instruction 的贡献是 0。所以：

$$p(c_i | i, s, \ell_\pi) = p(c_\pi | s, \ell_\pi) \cdot p(\ell_\pi | i) + (1 - p(c_\pi | s, \ell_\pi)) \cdot 0$$

这个分解的 elegant 之处在于：**两个因子可以独立训练和改进**。$p(\ell_\pi | i)$ 完全依赖 LLM（可以换更大的 model），$p(c_\pi | s, \ell_\pi)$ 完全依赖 robot learning（可以收集更多 data）。这两个 research community 可以并行 scale。

### 2.2 LLM 作为 Say：Scoring 而非 Generating

一个非常重要的设计选择是：**不用 LLM 的 generative decoding，而是用它的 scoring 接口**。

对于每个 candidate skill $\ell_\pi \in \ell_\Pi$，计算：

$$\ell_\pi^* = \arg\max_{\ell_\pi \in \ell_\Pi} p(\ell_\pi | i)$$

这里 $p(\ell_\pi | i)$ 通过 LLM 的 log-probability 来估计。具体实现是把 instruction $i$ 作为 context，然后 score 每个 skill description 作为 completion 的 likelihood。

为什么这个设计 crucial？因为 generative decoding 可能产生：
1. 不在 skill set 中的 action（如 "vacuum the floor"）
2. 格式不规范、难以 parse 的 output
3. 无法与 affordance probability 相乘（因为 generative 没有 explicit per-option probability）

Scoring 接口天然地输出一个 categorical distribution over skills，可以和 affordance distribution 逐元素相乘。这也带来了 **interpretability**——Figure 6 展示了每一步 LLM 和 affordance 各自的 scoring，用户可以清楚看到 robot 在考虑什么。

### 2.3 Affordance 作为 Value Function

这是另一个关键的 insight。作者观察到：**在 sparse reward setting 下（成功=1, 失败=0），value function $V^\pi(s)$ 恰好等于"从 state $s$ 出发执行 policy $\pi$ 能成功的概率"**。

数学上，sparse reward 下：

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_t \gamma^t R(s_t, a_t) \mid s_0 = s \right]$$

如果 reward 只在 episode 结束时给（成功=1, 失败=0），且 $\gamma = 1$（undiscounted），那么：

$$V^\pi(s) = P(\text{success} \mid s, \pi) = p(c_\pi | s, \ell_\pi)$$

这就是 Gibson [10] 意义上的 **affordance**——环境对 agent 提供的 action possibility。SayCan 把这个 Gibsonian concept 用 modern RL 的 value function 来 instantiate。

Q-function 的训练用标准 TD loss：

$$L_{TD}(\theta) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}} \left[ R(s,a) + \gamma \mathbb{E}_{a' \sim \pi} Q_\theta^\pi(s', a') - Q_\theta^\pi(s, a) \right]$$

变量：
- $\theta$: Q-network 参数
- $\mathcal{D}$: replay buffer 中的 transition dataset
- $\gamma$: discount factor
- $a' \sim \pi$: next action 从 policy 采样

由于 reward 是 sparse binary，作者用 log loss 而非 MSE 来训练 Q-function（见 Appendix C.2），这让 Q-value 更好地 calibrated 为 probability。

### 2.4 完整算法

Algorithm 1 的伪代码：

```
Algorithm 1: SayCan
Input: instruction i, state s_0, skills Π with descriptions ℓ_Π
1: n = 0, π = ∅
2: while ℓ_{π_{n-1}} ≠ "done" do
3:   C = ∅
4:   for π ∈ Π, ℓ_π ∈ ℓ_Π do
5:     p_π^LLM = p(ℓ_π | i, ℓ_{π_{n-1}}, ..., ℓ_{π_0})   # LLM scoring
6:     p_π^affordance = p(c_π | s_n, ℓ_π)                 # Value function
7:     p_π^combined = p_π^affordance × p_π^LLM            # 两个概率相乘
8:     C = C ∪ {p_π^combined}
9:   end for
10:  π_n = argmax_{π ∈ Π} C
11:  Execute π_n(s_n) in environment, update s_{n+1}
12:  n = n + 1
13: end while
```

注意第 5 行，LLM 的 context 是逐步增长的：包含了之前选过的所有 skills 的 description。这让 LLM 能理解 "我已经 pick up 了 apple，现在应该 bring it to you 而不是再 pick up 一次"。这是一个 **autoregressive planning** 过程。

## 3. System 架构详解

### 3.1 整体架构（Figure 3 解析）

Figure 3 展示了 SayCan 的完整 pipeline：

```
[User Instruction i]
        ↓
    ┌───────────────────────────────────────┐
    │  LLM (PaLM 540B)                      │
    │  Input: i + prompt + previous skills  │
    │  Output: p(ℓ_π | i) for each skill     │
    └───────────────┬───────────────────────┘
                    │
                    ×  (element-wise multiply)
                    │
    ┌───────────────┴───────────────────────┐
    │  Value Function Space (VFS)            │
    │  Input: current observation s          │
    │  Output: p(c_π | s, ℓ_π) per skill     │
    └───────────────┬───────────────────────┘
                    │
                    ↓
            [argmax → selected skill π_n]
                    ↓
            [Execute π_n on robot]
                    ↓
            [Append ℓ_{π_n} to context, repeat]
```

这个设计的 elegance 在于 LLM 和 VFS 是 **完全解耦** 的两个 module，只通过一个 scalar probability 接口通信。这意味着：
- 可以 mix-and-match 不同 LLM（GPT-3, PaLM, FLAN）
- 可以 mix-and-match 不同 policy/value（BC-Z, MT-Opt, scripted）
- 开源版本甚至用 ViLD object detector 代替 value function（因为没有 RL 训练的 affordance）

### 3.2 RL Policy 架构（Figure 9, MT-Opt 风格）

RL model 的网络结构：

```
[Camera Image 640×512]
        ↓
[7 Conv Layers]  ← spatial features
        ↓                          [Language embedding (USE)]
                                    ↓
[FiLM conditioning with language + robot state + prev action]
        ↓
[11 more Conv Layers]
        ↓
[Sigmoid gate → Q-value ∈ [0,1]]
```

关键设计点：
- **Asynchronous control**：inference 时 robot 还在执行上一个 action，所以 model 接收 "上一个 action 还剩多少没执行" 作为 input [87]
- **Sigmoid output**：因为 reward 是 binary，Q-value 被 constrain 到 [0,1]，这样直接作为 affordance probability
- **Language conditioning**：用 Universal Sentence Encoder [15] 的 frozen embedding，而非 fine-tune

### 3.3 BC Policy 架构（Figure 10, BC-Z 风格）

BC model：

```
[Camera Image 256×320 (downsampled for speed)]
        ↓
[ResNet-18 backbone]
        ↓
[FiLM conditioning with USE language embedding]
        ↓
[FC layers → action components]
        ├── arm position (continuous)
        ├── arm orientation (continuous)
        ├── gripper open/close (discrete)
        └── terminate action (discrete)
```

注意 BC 和 RL 用了不同的 image resolution：BC 用 256×320（for speed），但 affordance value function 用 full 640×512（因为 lower resolution 学不好 Q-function）。这是一个 practical but interesting 的 finding。

### 3.4 Affordance Calibration（Appendix D.2）

不同 skill family 用了不同的 affordance function，这体现了 SayCan 的 flexibility：

**Pick skills**（learned value function）：
$$p_\text{pick}^\text{aff} = \text{clamp}\left(\frac{v^\text{pick} - v_\text{min}^\text{pick}}{v_\text{max}^\text{pick} - v_\text{min}^\text{pick}}, 0, 1\right)$$

其中 $v_\text{max}^\text{pick} = 0.5$, $v_\text{min}^\text{pick} = 0.2$。这是因为 learned value function 不是 perfectly calibrated 的 probability，需要 empirically 确定 min/max 来 renormalize。

**Go to skills**（distance-based）：
$$p_\text{goto}^\text{aff} = \text{clamp}\left(\frac{d_\text{max}^\text{goto} - d^\text{goto}}{d_\text{max}^\text{goto} - d_\text{min}^\text{goto}}, 0, 1\right)$$

其中 $d_\text{max}^\text{goto} = 100$m, $d_\text{min}^\text{goto} = 0$m。这是纯 heuristic，不是 learned 的，但 works fine。

**Place skills**：$p_\text{place}^\text{aff} = 1.0$（总是 possible，靠 LLM 理解 "place 只在 pick 之后" 来约束顺序）

**Terminate**：$p_\text{terminate}^\text{aff} = 0.1$（小常数，确保没有 feasible skill 时能终止）

这个设计说明：**SayCan 不要求所有 affordance 都来自 learned value function**。任何能输出 "这个 skill 在当前 state 能否成功" 的 function 都可以。这为 future work 留了很多空间——比如用 scene descriptor、object detector、甚至 human feedback 作为 affordance source。后续的 Inner Monologue [25] 就做了这个 extension。

## 4. 实验数据详解

### 4.1 主结果（Table 2）

| Method | Mock Kitchen Plan | Mock Kitchen Execute | Real Kitchen Plan | Real Kitchen Execute |
|--------|-------------------|----------------------|-------------------|----------------------|
| **PaLM-SayCan** | **84%** | **74%** | **81%** | **60%** |
| No VF (LLM only) | 67% | - | - | - |
| Generative (LLM gen + USE project) | 74% | - | - | - |
| BC NL (policy with raw instruction) | - | 0% | - | - |
| BC USE (policy with projected instruction) | - | 9% | - | - |

关键 takeaways：
1. **Grounding 几乎翻倍 performance**：No VF 只有 67% plan rate，加上 affordance 后到 84%。这是 grounding 必要性的最强 evidence。
2. **BC NL = 0%**：直接把 high-level instruction 喂给 language-conditioned policy 完全失败。这说明 policy 只懂 "pick up the apple" 这种 atomic command，不懂 "I spilled my drink, can you help?" 这种 abstract instruction。LLM 的 semantic parsing 是必要的。
3. **Generative vs Scoring**：Generative（74%）< Scoring（84%）。Generative 方案先让 LLM 生成文本，再用 USE embedding 投影到最近的 skill。这会丢失 explicit probability，无法与 affordance 相乘。
4. **Sim-to-real gap**：Real kitchen 比 mock kitchen execute 下降 14%（74%→60%），但 plan 只降 3%（84%→81%）。这说明 planning 部分（LLM + affordance）generalize 得很好，gap 主要在 low-level policy 的 domain shift。

### 4.2 Instruction Family 细分（Table 2 by family）

| Family | Num | Plan | Execute |
|--------|-----|------|---------|
| NL Single Primitive | 15 | 100% | 100% |
| NL Nouns | 15 | 67% | 47% |
| NL Verbs | 15 | 100% | 93% |
| Structured | 15 | 93% | 87% |
| Embodiment | 11 | 64% | 55% |
| Crowd Sourced | 15 | 87% | 87% |
| Long-Horizon | 15 | 73% | 47% |

分析：
- **NL Nouns 差**（67% plan）：因为有 15 个 objects + 5 个 locations = 20 个 nouns 可能，而 verbs 只有 6 个。Search space 更大。
- **Embodiment 最难**（64% plan）：这测试 robot 是否理解 "我已经拿着 apple 了，不用再 pick"。失败主要来自 affordance misclassification。
- **Long-Horizon 执行差**（73%→47%）：因为长 sequence 中任何一步失败都会导致整体失败。而且 LLM 倾向于 early termination（bring 了第一个 object 就说 done）。
- **Structured vs NL Verbs**：Structured 93% < NL Verbs 100%。这说明 LLM 反而更擅长理解 natural language，而非 structured command——因为它的 training data 主要是 natural language。

### 4.3 LLM Size Ablation（Table 3, Table 6）

| LLM | Plan | Execute |
|-----|------|---------|
| PaLM 540B | 84% | 74% |
| PaLM 62B | 72% | - |
| PaLM 8B | 38% | - |
| FLAN 137B | 70% | 61% |

这是 paper 最 exciting 的 finding：**LLM 改进直接 translate 到 robot 改进**。从 8B→540B，robot execute rate 从 ~38% 到 74%。这是第一次在 robotics 上看到 NLP scaling law 的直接 benefit。

FLAN 虽然 fine-tune 过 instruction following，但反而比 PaLM 540B 差。作者推测是因为 PaLM 的 training data 更 broad 更好，弥补了没有 instruction tuning 的劣势。这个 finding 在今天看来很有预见性——后来大家发现大规模 pretraining > task-specific fine-tuning。

### 4.4 Error 分析

作者报告 65% errors 来自 LLM，35% 来自 affordance。LLM 的典型 failure：
- **Negation**：不理解 "bring me something that isn't a fruit"（后来用 chain-of-thought 解决）
- **Ambiguous reference**：不理解 "bring me a drink with caffeine"
- **Early termination**：long-horizon 任务中提前说 done

Affordance 的典型 failure：
- Value function 对某些 object 的 grasp success 误判
- 对 drawer 是否能 open 的判断不准

## 5. Case Studies：扩展能力

### 5.1 Chain-of-Thought Reasoning（Table 4）

作者把 Wei et al. [24] 的 chain-of-thought prompting 集成进 SayCan。修改 prompt，加入 "Explanation:" 字段：

```
Human: Can you bring a fruit-flavored drink without caffeine?
Explanation: The user has asked for a drink that is fruit-flavored 
and does not have caffeine, I will bring the lime soda.
Robot: 1. find a lime soda, 2. pick up the lime soda, 3. bring it to you, 
       4. put down the lime soda, 5. done
```

关键修改：先用 LLM 的 **generative** mode 生成 explanation，然后再用 **scoring** mode 把 explanation 加进 context 来 score skills。这是 generative 和 scoring 的 hybrid 用法。

效果：能处理 negation（"isn't a fruit" → bring energy bar）和 reasoning（"more filling" → multigrain chips）。

### 5.2 Multilingual（Table 8）

PaLM 训练了 multilingual corpora，所以 SayCan 能 zero-shot 处理中文、法文、西班牙文。Plan rate 几乎不掉（只有一句法文失败）。这说明 LLM 的 semantic knowledge 是 language-agnostic 的 representation，只是 surface form 不同。

### 5.3 Adding Skills: Drawer Manipulation

新增 drawer skills 只需要：
1. 在 LLM 的 candidate skill list 中加入 "open the drawer", "close the drawer" 等
2. 为这些 skills 提供 affordance function（这里用 heuristic：robot 在 drawer 旁边就 affordance=1）
3. 在 prompt 中加几个 drawer 的 example（Listing 2）

结果：plan rate 100%，execute rate 33%（drawer 物理操作难）。**对其他 instruction 的性能没有影响**——这是 zero-shot extensibility 的强 evidence。

## 6. 与相关工作的联系

### 6.1 Grounding Language Models 的谱系

SayCan 处于一个有趣的位置：

- **Language → Action 直接 mapping**：如 EmbBERT [35], E.T. [36] ——这些需要大量 interaction data，且只处理 short-horizon task
- **LLM as planner (prompt engineering only)**：如 Huang et al. [23]（Language Models as Zero-Shot Planners）——只用 generative output，没有 grounding，对应 SayCan 的 "Generative" baseline
- **LLM fine-tuned with interaction**：如 Ouyang et al. [11]（RLHF）, Li et al. [45]（Pre-trained LMs for interactive decision-making）——需要 fine-tune LLM，成本高
- **SayCan**：用 pre-trained value function 作为 grounding，zero-shot，不需要 fine-tune LLM

SayCan 的独特之处是 **affordance 作为 external grounding signal，通过概率接口注入 LLM**，而不修改 LLM 本身。这让 LLM 和 robot learning 可以独立进步。

### 6.2 与 Task and Motion Planning (TAMP) 的关系

经典 TAMP [59, 60, 61] 用 symbolic planner（如 STRIPS）+ motion planner。SayCan 可以看作是 **learned TAMP**：
- Symbolic planning → LLM（semantic knowledge 替代 hand-coded domain）
- Motion feasibility → learned value function（替代 geometric motion planner）
- Task feasibility → LLM probability + affordance probability

这个对应很有意思。经典 TAMP 的 bottleneck 是需要人类 hand-craft 所有 primitives 和 constraints，无法 scale。SayCan 用 learning 替代了 hand-crafting，代价是失去了 symbolic reasoning 的 guarantees，但获得了 scalability。

### 6.3 与 Inner Monologue [25] 的关系

SayCan 的主要 limitation 是 **open-loop**——只在每步决策时 query affordance，但如果 skill 执行失败或环境变化，没有 feedback。Huang et al. 的 Inner Monologue 扩展了 SayCan 加入 closed-loop：用 success detector、scene descriptor、human feedback 作为 environment feedback，通过 text 注入 LLM context。这本质上是把 SayCan 的 affordance 接口 general化——任何 environment signal 都可以变成 text 注入 LLM。

### 6.4 与 Code as Policies / ProgPrompt 的关系

同期和后续工作如 Code as Policies（Liang et al.）和 ProgPrompt 用 LLM 生成 code 而非 natural language plan。这更 expressive（可以有 loops, conditionals），但失去了 SayCan 的 **per-skill probability scoring**，因此也无法直接与 affordance 相乘。SayCan 的 scoring 方案在 interpretability 和 grounding 上有独特优势。

## 7. Open Source 实现

作者开源了一个 tabletop 版本（Figure 8）：
- Robot: UR5
- Policy: CLIPort [26]（pick-and-place）
- Affordance: ViLD object detector [27]（因为没有 RL value function）
- LLM: GPT-3 [5]

这个开源版本的 affordance 用 object detection 替代 value function——**印证了 SayCan 的 modular design 允许不同 affordance source**。如果 scene 里有目标 object，ViLD 检测到，则 affordance 高。

Colab 链接：https://say-can.github.io/#open-source

## 8. Limitations 和深层思考

### 8.1 Skills 的 bottleneck

作者明确指出 primary bottleneck 是 skills 的 range 和 robustness。LLM 已经很强了（plan rate 84%），但 execute rate 只有 74%，gap 在 low-level skills。这提示了一个研究方向：**如何让 skill learning 也像 LLM 一样 scale**。后来的 RT-1, RT-2, RT-X 都在往这个方向走。

### 8.2 Open-loop planning 的 limitation

SayCan 在每步决策时 query affordance，但 skill 执行过程中没有 re-planning。如果 "pick up the apple" 失败了（apple 滑落），robot 不会知道，会继续 "bring it to you"（空手 bring）。Inner Monologue 部分解决了这个问题。

### 8.3 Natural language 作为 robot programming interface 的 question

作者在 conclusion 中提出了一个深刻的问题：**natural language 是否是 programming robot 的正确 ontology**？NL 的优点是 contextual、semantic rich、抽象层级合适。但缺点是需要 supervision（label），且对某些 task 不是最 descriptive 的 medium（比如 hindsight goal image [84] 可能更直接）。这个问题至今没有定论，后来 Visual Prompting、Goal-conditioned RL 等方向都在探索 alternative。

### 8.4 LLM bias 的继承

SayCan 继承了 LLM 的所有 bias [82, 83]：training data 的 bias、hallucination、negation 理解差。这些在 NLP 社区已知的 limitation 会直接 translate 到 robot。比如如果 LLM 对某些文化背景的 object 不熟悉，robot 就无法正确处理相关 instruction。

## 9. 对后续工作的影响

SayCan 开启了 LLM × Robotics 的研究浪潮：

1. **RT-2 (Google, 2023)**：把 vision-language model 直接 fine-tune 输出 robot action token，统一了 semantic knowledge 和 low-level control
2. **Code as Policies (Liang et al., 2023)**：用 LLM 生成 executable code
3. **VoxPoser (Huang et al., 2023)**：用 LLM 生成 3D value map 来 affordance
4. **Inner Monologue [25]**：closed-loop SayCan
5. **AutoGPT / BabyAGI**：autonomous agent loop，思路类似 SayCan 的 iterative planning

SayCan 的核心 insight——**把 LLM 的 semantic prior 与 learned affordance 通过概率接口结合**——成为了 LLM agent 设计的一个 paradigm。

## 10. 我的 Intuition 总结

如果要我用一句话总结 SayCan 的 essence：**LLM 提供 "semantic affordance"（这个 action 对 task 有用吗），value function 提供 "physical affordance"（这个 action 在当前 world 能执行吗），两者相乘得到 "真正可行的 next action"**。

这个 framework 的 power 在于：
- **Decoupling**：semantic 和 physical 可以独立 scale
- **Interpretability**：每一步都能看到 LLM 和 affordance 各自的 reasoning
- **Extensibility**：新 skill 只需加 description + affordance + prompt example
- **Zero-shot**：不需要 fine-tune LLM，直接用 pre-trained knowledge
- **Scaling**：LLM 变强，robot 直接变强

这也揭示了一个更深的道理：**很多 AI 系统的瓶颈不在于单一 component 多强，而在于如何 decompose 问题让各 component 各司其职**。LLM 强在 semantic reasoning，弱在 physical grounding；RL policy 强在 physical control，弱在 high-level planning。SayCan 用一个优雅的概率接口把它们结合起来，让 1+1 > 2。

## References

- SayCan 项目页: https://say-can.github.io
- SayCan arXiv: https://arxiv.org/abs/2204.01691
- PaLM: https://arxiv.org/abs/2204.02311
- BC-Z (Jang et al.): https://proceedings.mlr.press/v164/jang22a.html
- MT-Opt (Kalashnikov et al.): https://arxiv.org/abs/2104.08212
- Chain of Thought (Wei et al.): https://arxiv.org/abs/2201.11903
- CLIPort (Shridhar et al.): https://arxiv.org/abs/2109.12098
- ViLD (Gu et al.): https://arxiv.org/abs/2104.13921
- Universal Sentence Encoder (Cer et al.): https://arxiv.org/abs/1803.11175
- Inner Monologue (Huang et al.): https://arxiv.org/abs/2207.05608
- Language Models as Zero-Shot Planners (Huang et al.): https://arxiv.org/abs/2201.07207
- Value Function Spaces (Shah et al.): https://arxiv.org/abs/2111.03189
- RetinaGAN (Ho et al.): https://arxiv.org/abs/2105.07961
- Code as Policies (Liang et al.): https://arxiv.org/abs/2209.07794
- RT-2: https://arxiv.org/abs/2307.15818
- Gibson affordances: Gibson, "The Ecological Approach to Visual Perception", 1979
- Bender & Koller, "Climbing towards NLU": https://aclanthology.org/2020.acl-main.463/

如果你想 deep dive 某个方面——比如 value function calibration 的细节、prompt engineering 的 ablation、或者与 RT-2 的对比——可以告诉我，我可以进一步展开。
