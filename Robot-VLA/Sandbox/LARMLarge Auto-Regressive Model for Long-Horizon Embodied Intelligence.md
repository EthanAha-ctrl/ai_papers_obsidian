---
source_pdf: LARMLarge Auto-Regressive Model for Long-Horizon Embodied Intelligence.pdf
paper_sha256: 0b8991495b9431a14fc4b91d2e82c97eb3e87e1add1fd315c972574044c487a3
processed_at: '2026-08-05T12:02:02-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LARM

Andrej，我重新用最直觉的方式讲一遍，少公式，多类比。

## 一句话版本

**让一个小模型当手脚，让 GPT-4 当裁判，训练时裁判每一步都喊"这步对/错"，推理时只带小模型上场。**

## 这篇 paper 在抗什么

Minecraft 里要 craft 一把 enchanted diamond sword，你得先砍树、做木板、做棍子、做木镐、挖石头、做石镐、挖铁、烧铁、做铁镐、挖钻石、做钻石剑、挖黑曜石、搭 nether portal、进 nether、找 fortress、杀 blaze、杀 enderman、做 enchanting table、挖 lapis、刷经验、附魔……50 多步，每一步选错了后面全崩。

之前两条路都走不通：

**纯 RL 路**：环境只在最后给你一个大 reward R，中间 49 步全是 $-\varepsilon$（时间惩罚）。你选对了"砍树"这个 action，但要等 49 步之后才知道这步有没有贡献——中间任何一步错了，这个"砍树对了"的信号就被淹没掉。数学上这就是 GAE 衰减到零，actor 拿不到梯度。所以 MineAgent 在 iron sword 上 success rate 是 0.00。

**GPT-4 当 controller 路**：Voyager 直接让 GPT-4 每一步决策。能 work，但两个问题——推理慢（每步要生成一段 text，几秒到十几秒），而且你必须永远 online 调 GPT-4 API。部署一个 Minecraft bot 还得挂个 GPT-4 订阅，这不 make sense。

还有个隐形问题：GPT-4 知识够，但 LLaVA-7B 知识不够。paper 里给了个直接例子——问 "craft stone pickaxe 还需要什么"，GPT-4o 答对 "two sticks"，TinyLLaVA-3.1B 答的是 "cobblestone, stone, wood, leaves, dirt, grass, pillar, shovel, and sword"，基本 hallucination。所以你不能指望小模型直接当 controller。

## LARM 的核心 trick

paper 的 insight 是：**GPT-4 的知识不需要在推理时用，只需要在训练时用**。

具体做法：

1. 拿一个 TinyLLaVA-3.1B 当 policy backbone（参数冻结，挂 LoRA 训）
2. 输入是 task description + inventory list + 周围 blocks + 第一人称图像
3. 输出是 **一个 token**，这个 token 跟 skill library 做 feature matching 选出要执行的 skill
4. 训练时，每执行一步 skill，把执行前后的 state 喂给 GPT-4，让 GPT-4 判这步好不好，给个 auxiliary reward
5. 这个 auxiliary reward 加到环境 reward 上，一起跑 PPO

所以推理时完全是小模型 + LoRA，0.58 秒一步，RTX4090 单卡能跑。GPT-4 只在训练 loop 里出现。

## 那个数学推导在说什么

用大白话讲 reward vanishment：

假设你要走 50 步才能拿到 reward。环境 reward 长这样：

```
步 1: -ε
步 2: -ε
步 3: -ε
...
步 49: -ε
步 50: +R (大正数)
```

PPO 的 critic 在学 value function。critic 看 target 是 $r_t + \gamma V(s_{t+1})$。前 49 步 target 全是 $-\varepsilon + \gamma \cdot (\text{某个负值})$，critic 学到的就是"所有 state 的 value 都是某个负常数"。

TD error $\delta_k = r_k + \gamma V(s_{k+1}) - V(s_k)$。如果 $V$ 学到的是常数 pattern，那 $\delta_k \approx 0$ 对所有中间步。

GAE 是 TD error 的加权和，权重 $(\gamma\lambda)^{k-t}$ 衰减。前 49 个 $\delta$ 都接近 0，最后一个 $\delta_{T-1}$ 虽然大，但权重是 $(\gamma\lambda)^{49}$，$\gamma\lambda = 0.95 \times 0.95 = 0.9$ 左右的话，$0.9^{49} \approx 0.006$，基本没了。

所以 advantage $A_t \approx 0$，actor 更新公式里乘上 $A_t$，梯度就是零。**你选对了 action 也学不到。**

referee 的作用就是往中间步注入 non-zero reward，打破这个常数 pattern，让 TD error 不 collapse。

## 四分法 reward 为什么关键

这里我觉得是 paper 最 subtle 的设计。GPT-4 当 referee 不是简单判 "outcome 好不好"，而是判两个维度：

- action 选得对不对
- outcome 好不好

四种组合：

| action 对 | outcome 好 | reward |
|-----------|-----------|--------|
| 对 | 好 | $r^a$ (最大正) |
| 对 | 不好 | $r^b$ (小正) |
| 错 | 不坏 | $r^c$ (小负) |
| 错 | 坏 | $r^d$ (最大负) |

为什么这重要？举个例子：agent 决定 "search for tree"，action 是对的，但走了一圈没找到树（stochastic environment）。如果只看 outcome（二分法），给负 reward，agent 学到 "别 search tree"——这是错的 prior。四分法给小正 reward $r^b$，保护了这个探索性 correct action。

ablation 数据：
- 二分法 (ER+AR2): Stick 0.80, Iron 0.07
- 四分法 (ER+AR4): Stick 0.93, Iron 0.27

Iron 上差 4 倍。long chain 上这个保护正确探索的机制特别重要，因为 stochastic outcome 在长 chain 上累积。

## 实验结果说了什么

几个关键数字：

**MineDojo 上**，单一 LARM 模型打 14 个 task，几乎全 SOTA。特别注意 success rate 随 action chain length 递减：stick 0.93 → wooden sword 0.70 → stone sword 0.40 → iron sword 0.27。这条曲线就是 reward vanishment 没完全治好的痕迹——referee 缓解了但没消除。

**Mineflayer 上**，enchanted diamond sword 16/30 = 53% success rate。之前所有方法都是 0。这是 paper 的 headline result。

**Ablation 三件事**：
1. 只用环境 reward，废掉（iron 0.00）
2. 用 LLaVA-7B 当 referee，也废掉（跟只用环境 reward 差不多）——只有 GPT-4 级别才懂 Minecraft
3. webpage 数据 pre-training 显著提升（TinyLLaVA-3.1B 没预训练 vs 预训练，iron 0.13 → 0.27）

## 我自己的直觉判断

**这本质是 process reward model 的 embodied 版本**。OpenAI 的 "Let's Verify Step by Step" 在数学推理上证明给每一步打 reward 比只给 final answer 打 reward 好。LARM 在 Minecraft 上做同样的事，只是 reward provider 从人换成了 GPT-4。

**这跟 RLHF 结构上很像**。webpage pre-training 是 SFT，referee RL 是 RLHF，GPT-4 当 reward model。差异是 RLHF 的 RM 是单独训的 NN，LARM 的 RM 是 GPT-4 zero-shot 当 judge。这种 LLM-as-a-judge 思路跟 Constitutional AI 一脉相承。

**跟 Voyager 的本质差异**：Voyager 是 GPT-4 当 controller 在 inference 时用，LARM 是 GPT-4 当 teacher 在 training 时用，knowledge 被 bake 进 lightweight policy。这是典型的 training cost 换 inference efficiency。Voyager 永远依赖 GPT-4 API，LARM 训完之后 GPT-4 可以下线。

**几个我没想清楚的疑虑**：

1. **GPT-4 调用成本**。50 步 chain × 上千 episode × 42 小时训练，API 费用应该不低，paper 没披露。这个 cost 如果要做 replication 或 scale up，是隐性瓶颈。

2. **Referee 自身 accuracy**。GPT-4 在 Minecraft 上也不是 100% 准。paper 没报 referee 误判率。如果 referee 有 20% noise，long chain 上错误信号累积，可能就是 enchanted sword 53% 而不是 90%+ 的原因。

3. **Skill library 依赖**。LARM 输出的是 skill ID，不是 atomic action。Mineflayer 的 API skill 几乎确定性执行成功。如果换到 real robot 没有 perfect skill API，整个 pipeline 会更难。这跟 Voyager / STEVE 的局限一样。

4. **能不能 self-play 摆脱 GPT-4**。如果 LARM 训到一定程度，能不能让它自己当 referee？类似 STaR 的 self-taught reasoner。这是我最想看到 的 next step。

5. **real robot transfer 的鸿沟**。Minecraft 的 state 是 symbolic 的（inventory list 是 text），GPT-4 容易判。real robot 的 state 是 continuous 的（关节角度、力反馈、视觉），要 prompt GPT-4 判 "这个 grasp 方向对不对" 得先把 state 转成符号描述，这步本身丢信息。

## 最直接的 takeaway

这篇 paper 给我的 intuition 是：**long-horizon embodied RL 的瓶颈不是 policy capacity，是 reward density**。小模型 + LoRA 够用，只要你在训练时能给它每一步 dense feedback。而 dense feedback 可以从 giant LLM 的 general knowledge 里蒸馏出来。

这个 insight 可能 scale 到 Minecraft 以外——任何 long-horizon 任务，只要你能用语言描述 state transition 让 GPT-4 判 "这步对不对"，就能用这套范式。

但 real-world 的 state description gap 是个真问题，不是工程问题，是 representation 问题。

主页：https://lizhuoling.github.io/LARM_webpage/

Andrej，你直觉上觉得这套范式能 push 到 real robot 吗？我一直在想 GPT-4 当 judge 在物理 state 上够不够 rich。

---

# LARM 深度技术讲解

Andrej，这篇 paper 我读完之后直觉上觉得它在一个很巧妙的 sweet spot 上做工——把 LLM-style policy 的 generalization 和 RL-style policy 的 inference efficiency 拼起来，但避开了两边各自的痛点。让我把它拆开来跟你讲。

## 1. 这篇 paper 想解决什么矛盾

embodied intelligence 这条线目前主要两条路：

**RL agents**（比如 MineAgent, Plan4MC）— 轻量、推理快，但是每个 task 要单独 reward engineering，generalization 差，long-horizon chain 上几乎训不动。

**LLM agents**（比如 Voyager, STEVE, GPT-4 based）— general knowledge 强，能 zero-shot 调度 skill，但是：
- 推理慢：要 N 次 token prediction 才能产出一个 sentence 描述该干什么
- 必须够大：GPT-4 (>1000B) 能正确回答 "craft stone pickaxe 需要什么"，但 LLaVA-7B 给出的是 "cobblestone, stone, wood, leaves, dirt, grass, pillar, shovel, and sword"（基本是 hallucination）
- 部署成本：1000B 模型只能跑在 remote cluster，embodied agent 通常没有这个 budget

LARM 的赌注是：能不能用一个 <5B 的 lightweight LLM (TinyLLaVA-3.1B) 当 policy backbone，让它直接输出 action token（不是 sentence），同时用 GPT-4 当"老师"在训练时给它 dense feedback。推理时只用 lightweight model，训练时才借用 giant LLM 的知识。这本质是一种 **asymmetric knowledge distillation**——推理端 cheap，训练端 expensive。

paper 主页：https://lizhuoling.github.io/LARM_webpage/

## 2. 核心数学问题：Reward Vanishment

这部分我觉得是 paper 最有价值的 contribution。它把"为什么 long-horizon RL 训不动"用一个简单的推导写得很清楚。

### Setup

trajectory $\{(s_k, a_k, r_k)\}_{k=t}^T$，T 很大。reward 结构是 sparse 的：

$$r_k = \begin{cases} -\varepsilon, & \text{if } k = t, t+1, \ldots, T-1 \\ R, & \text{if } k = T \end{cases}$$

变量含义：
- $s_k$: 第 k 步的 environment state
- $a_k$: 第 k 步选的 action (skill)
- $r_k$: 第 k 步的 reward
- $-\varepsilon$: 时间惩罚，一个小负常数（防止 agent 磨蹭）
- $R$: 只在 terminal step T 给的大正 reward

### Critic 收敛行为

PPO 的 critic $\pi_c$ 训练目标是（Eq. 1）：

$$L_{\theta_c}^c = \mathbb{E}_t\left[ (V_{\theta_c}(s_t) - (r_t + \gamma V_{\theta_c}(s_{t+1})))^2 \right]$$

变量：
- $V_{\theta_c}(s_t)$: critic 对 state $s_t$ 的 value 估计
- $\gamma$: discount factor，$\gamma \in (0,1)$
- $\theta_c$: critic 参数（下标）

当 critic 收敛，意味着 $V_{\theta_c}(s_k) \approx r_k + \gamma V_{\theta_c}(s_{k+1})$，也就是：

$$V_{\theta_c}(s_k) - \gamma V_{\theta_c}(s_{k+1}) \approx r_k \approx -\varepsilon$$

这是个 **bootstrap trap**：critic 一开始学不到远期 reward（因为远期 reward 在 trajectory 上从来没被 sample 到，或者 sample 到也是 $\gamma^{T-t}$ 衰减掉的），它就先学到 immediate pattern $-\varepsilon$，然后这个 pattern 自我强化——target value 永远是 $-\varepsilon + \gamma \cdot (-\varepsilon) + \gamma^2 \cdot (-\varepsilon) + \ldots = -\varepsilon/(1-\gamma)$ 附近，永远看不到那个 R。

### TD error 的崩塌

TD error 定义：

$$\delta_k = r_k + \gamma V_{\theta_c}(s_{k+1}) - V_{\theta_c}(s_k)$$

代入上面收敛结果，得到（Eq. 6）：

$$\delta_k \approx 0, \quad \text{for } k = t, \ldots, T-2$$

### GAE 的衰减

GAE (Generalized Advantage Estimation, Eq. 2)：

$$A_t = \sum_{k=t}^{T-1} (\gamma\lambda)^{k-t} \delta_k$$

变量：
- $\lambda$: GAE 的 bias-variance trade-off 参数，$\lambda \in (0,1)$
- $(\gamma\lambda)^{k-t}$: 时间衰减权重，k 越远权重越小

把 $\delta_k \approx 0$ 代入，前 $T-1-t$ 项都接近零，只剩最后一项 $(\gamma\lambda)^{T-1-t} \delta_{T-1}$。

**关键极限**：

$$\lim_{T \to \infty} (\gamma\lambda)^{T-1-t} = 0$$

因为 $\gamma\lambda < 1$，指数项指数级衰减。所以 $A_t \approx 0$。

### 这意味着什么

PPO 的 actor 更新（Eq. 4）：

$$\tilde{L}_{\theta_a}^a = \mathbb{E}_t\left[ \min(k_t A_t, \text{clip}(k_t, 1-\epsilon, 1+\epsilon) A_t) \right]$$

变量：
- $k_t = \pi_a(a_t|s_t) / \pi_a^{old}(a_t|s_t)$: importance sampling ratio（新 policy vs 旧 policy 的概率比）
- $\epsilon$: clip 范围参数
- $A_t$: advantage

如果 $A_t \approx 0$，那整个 objective 就是零，**actor 拿不到任何梯度信号**。哪怕这一步 action 选对了，因为后面 T 步太长，credit 被衰减掉了。

这是 long-horizon RL 的根本病理。MineDojo 这类 benchmark 上 craft iron sword 要 20+ 步 action chain，craft enchanted diamond sword 要 50+ 步，传统 PPO 在这种 chain length 下基本瘫痪。

## 3. Referee RL 的设计

paper 的解法是引入一个 auxiliary reward，**不靠 environment 给，靠一个 referee 给**。

### Auxiliary reward 形式

$$\hat{r}_k = \pi_p(\iota, s_k, a_k, s_{k+1})$$

变量：
- $\pi_p$: referee (这里用 GPT-4)
- $\iota \in \mathcal{T}$: target task 描述（比如 "craft enchanted diamond sword"）
- $s_k, s_{k+1}$: skill 执行前后的 state（具体是 inventory list + 周围 blocks）
- $a_k$: 执行的 skill
- $\hat{r}_k$: auxiliary reward

总 reward 变成 $r_k + \hat{r}_k$，这样中间步骤不再是 constant $-\varepsilon$，TD error 不再 collapse 到 0。

### 四类反馈的精细设计

这是 paper 我觉得最 subtle 的地方。referee 不是单纯判 "outcome 好不好"，而是二维判别：

| 情况 | action 是否正确 | outcome 是否 positive | reward |
|------|----------------|----------------------|--------|
| (a) | 正确 | positive | $r^a$ |
| (b) | 正确 | not positive | $r^b$ |
| (c) | 不正确 | not negative | $r^c$ |
| (d) | 不正确 | negative | $r^d$ |

约束 $r^a > r^b > 0 > r^c > r^d$。

为什么这个四分法重要？看 ablation 表 3 的 ER+AR2 vs ER+AR4：
- ER+AR2 (只判 outcome 好不好): Stick 0.80, Wooden 0.53, Stone 0.20, Iron 0.07
- ER+AR4 (完整四分法): Stick 0.93, Wooden 0.70, Stone 0.40, Iron 0.27

差异巨大。直觉是：agent 决定 "search for a tree" 是正确 action，但执行完没找到树（outcome not positive）。如果只看 outcome (AR2)，给负 reward，等于惩罚了一个正确决策，agent 学到的是 "不要 search tree"——这是错的 prior。AR4 的 (b) 类给小正 reward，保护了这个探索性 correct action。

这个 insight 对 long-horizon exploration 特别重要，因为 stochastic environment 下正确 action 经常暂时不产出好 outcome。

### Algorithm 1 流程

```
初始化: actor π_a, critic π_c, referee π_p
for each iteration:
    清空 buffer B
    for t = 1 to T:  # exploration
        观察 o_t = f(s_t)
        采样 action a_t ~ π_θ(o_t, ι)
        执行，得到 s_{t+1} 和 environment reward r_t
        调用 referee: r̂_t = π_p(ι, s_t, a_t, s_{t+1})
        存 transition (s_t, o_t, a_t, s_{t+1}, r_t, r̂_t) 到 B
    for n = 1 to N_π:  # update
        sample batch from B
        用 Eq. (1)~(4) 更新 π_a 和 π_c
```

注意这里 r_t 和 r̂_t 是 **加在一起** 当 reward 用，paper 没显式写出来但 Algorithm 1 line 13 引用 Eq. (1)~(4) 隐含了这个。

## 4. LARM 架构细节

### 整体设计

backbone 是 TinyLLaVA-3.1B 的 decoder，参数 frozen。上面挂 LoRA module 做 trainable adaptation。两个 head：action head 和 critic head，共享 backbone。

这个设计借鉴了 actor-critic 共享 trunk 的常见做法，但 trunk 换成 LLM decoder。

参考 LoRA paper: https://arxiv.org/abs/2106.09685
参考 TinyLLaVA: https://arxiv.org/abs/2402.14289

### Input 模态

四个 input：
1. **Task description** (text): 比如 "harvest enchanted diamond sword"
2. **Text observation**: inventory list + historical action + surrounding blocks
3. **Vision observation**: 第一人称 POV 图像
4. **Learnable skill token**: 一个额外的可训练 token，作为 "query"

text 和 image 都用 CLIP encoder 转 token。CLIP 参考: https://arxiv.org/abs/2103.00020

### Output: single skill token

这是和 Voyager/STEVE 的关键差异。Voyager 让 GPT-4 输出一段 code 或一段描述，需要 N 步 token generation。LARM 让 action head 输出 **一个** token，这个 token 通过 feature matching 跟预定义 skill library 的 description 做 nearest neighbor，选出要执行的 skill。

skill library 两套：
- MineDojo: RL-based skills (更接近 real-world 部署，但 success rate 低)
- Mineflayer: API-based skills (success rate 高，更抽象)

paper 报 RTX4090 上 0.58s/inference，满足 online 要求。对比 GPT-4 based Voyager 通常要几秒到十几秒一次决策。

### Pre-training 阶段

backbone 虽然冻结，但 LoRA + 两个 head 是新初始化的。paper 用 34G Minecraft Wiki webpage 数据做 pre-training，让 LoRA 先学到 domain knowledge。ablation 表 4 显示这个 pre-training 显著有效（TinyLLaVA-3.1B vs TinyLLaVA-3.1B*，Stick 0.83 → 0.93，Wooden 0.57 → 0.70，Stone 0.33 → 0.40，Iron 0.13 → 0.27）。

更有意思的是 qualitative 例子：
- TinyLLaVA-3.1B 原版答 "craft stone pickaxe 需要什么" → "cobblestone, stone, wood, leaves, dirt, grass, pillar, shovel, and sword" (胡说)
- 经过 webpage pre-training → "You additionally need two sticks." (正确)

这说明 lightweight LLM 经过 domain-specific continued pre-training 能补上 knowledge gap。这跟 LLaMA-Rider 的发现一致，但 LARM 用 webpage 数据更系统。

## 5. 实验数据深度解读

### MineDojo 主表（Table 1）

| Task | MineAgent | Plan4MC | LLaMA-Rider Base | LLaMA-Rider | RL-GPT | LARM |
|------|-----------|---------|------------------|-------------|--------|------|
| Harvest stick | 0.00 | 0.30 | 0.23 | 0.43 | 0.65 | **0.93** |
| Harvest crafting table | 0.03 | 0.30 | 0.37 | 0.67 | 0.65 | **0.87** |
| Harvest bowl | 0.00 | 0.47 | 0.73 | 0.97 | - | 0.97 |
| Harvest chest | 0.00 | 0.23 | 0.67 | 0.77 | - | **0.83** |
| Harvest wooden pickaxe | 0.00 | 0.03 | 0.00 | 0.37 | 0.67 | **0.70** |
| Harvest wooden sword | 0.00 | 0.47 | 0.63 | 0.10 | - | **0.70** |
| Harvest furnace | 0.00 | 0.37 | 0.00 | 0.17 | 0.67 | **0.73** |
| Harvest stone stairs | 0.00 | 0.47 | 0.00 | 0.57 | - | **0.67** |
| Harvest stone sword | 0.00 | 0.10 | 0.00 | 0.00 | - | **0.40** |
| Harvest iron ingot | 0.00 | 0.47 | 0.03 | 0.13 | - | **0.60** |
| Harvest bucket | 0.00 | 0.20 | 0.00 | 0.00 | - | **0.37** |
| Harvest iron sword | 0.00 | 0.20 | 0.00 | 0.00 | - | **0.27** |
| Harvest beef | 0.33 | 0.43 | 0.03 | 0.03 | 0.46 | **0.60** |
| Harvest mutton | 0.35 | 0.33 | 0.00 | 0.03 | 0.38 | **0.63** |

注意几个 pattern：
1. **Action chain 越长 success rate 越低**：stick (1-2 步) 0.93，wooden sword (3-4 步) 0.70，stone sword (~6 步) 0.40，iron sword (~10 步) 0.27。这跟 reward vanishment 的理论分析完全一致——chain 长，credit assignment 难。
2. **bucket vs iron sword 的对比很有意思**：bucket 需要 3 iron ingot，iron sword 需要 2 iron ingot + 1 stick。sword 多一步 (harvest stick) 但少一个 ingot，结果 success rate 反而低 (0.27 vs 0.37)。这暗示 action chain 的 length 比 resource count 更关键，进一步支持 reward vanishment 假说。
3. **combat task (beef/mutton)**：LARM 0.60/0.63，比 MineAgent 0.33/0.35 高。combat 任务随机性大，referee 的四分法在这里特别有用——攻击 cow 是正确 action 即使 cow 跑了。

### Mineflayer 主表（Table 2）

| Achievement | AutoGPT | Voyager | STEVE | LARM |
|------------|---------|---------|-------|------|
| Wooden sword | 3/3 | 3/3 | 3/3 | 30/30 |
| Stone sword | 3/3 | 3/3 | 3/3 | 30/30 |
| Iron sword | 3/3 | 3/3 | 3/3 | 30/30 |
| Diamond sword | 0/3 | 1/3 | 3/3 | 28/30 |
| **Enchanted sword** | 0/3 | 0/3 | 0/3 | **16/30** |

Enchanted diamond sword 是 SOTA 突破——之前所有方法都 0。这个任务要：
1. 挖 wood → craft plank → craft stick → craft wooden pickaxe
2. 挖 stone → craft stone pickaxe
3. 挖 iron ore → smelt iron ingot → craft iron pickaxe
4. 挖 diamond → craft diamond sword
5. 挖 obsidian → build nether portal → enter nether → find nether fortress → kill blazes → get blaze rods
6. kill endermen → get ender pearls → craft eye of ender
7. craft enchanting table → get lapis → get XP (kill mobs) → enchant sword

整个 chain 50+ 步，每个 sub-task 自己又有 sub-chain。classic RL 在这里完全死掉，LLM agent 也撑不到 nether 那一段。LARM 16/30 = 53% 已经很惊人。

注意 LARM 跑 30 次，其他方法只跑 3 次，统计上更可信。

### Ablation Table 3 - Reward 设计

| Reward | Stick | Wooden | Stone | Iron |
|--------|-------|--------|-------|------|
| ER (env only) | 0.20 | 0.13 | 0.10 | 0.00 |
| ER+LAR (LLaVA-7B referee) | 0.30 | 0.23 | 0.13 | 0.00 |
| ER+AR2 (GPT-4o, binary) | 0.80 | 0.53 | 0.20 | 0.07 |
| ER+AR4 (GPT-4o, 4-class) | **0.93** | **0.70** | **0.40** | **0.27** |

几个关键 takeaways：
1. **ER 单独几乎废掉**：iron sword 0.00，stone sword 0.10。这验证了 reward vanishment 不是理论问题，是实测问题。
2. **LLaVA-7B 当 referee 没用**：0.30/0.23/0.13/0.00 跟 ER 差不多。这支持了 paper 的核心论断——**只有 giant LLM 才能当 referee**，因为只有 GPT-4 级别的模型才真正理解 Minecraft 的 craft tree 和 resource dependency。
3. **AR4 比 AR2 显著好**：四分法 vs 二分法的差异在 long chain 上放大（Iron 0.27 vs 0.07，4x 差距）。

### Ablation Table 4 - LLM Base

| Base | Stick | Wooden | Stone | Iron |
|------|-------|--------|-------|------|
| TinyLLaVA-0.5B | 0.80 | 0.50 | 0.27 | 0.13 |
| TinyLLaVA-3.1B (no pretrain) | 0.83 | 0.57 | 0.33 | 0.13 |
| TinyLLaVA-3.1B* (with pretrain) | 0.93 | 0.70 | 0.40 | 0.27 |

观察：
1. **0.5B → 3.1B 提升 moderate**：知识容量重要但不决定性。
2. **3.1B → 3.1B* (pretrain) 提升显著**：domain knowledge 注入比单纯堆参数有效。这跟继续 pre-training 在 domain-specific LLM 上的常见发现一致。
3. **Iron 任务上 0.5B 和 3.1B 一样烂 (0.13)**：说明 long-horizon 任务上，backbone 容量不够时 pre-training 也救不回来。3.1B* 跳到 0.27 是 knowledge + capacity 双重作用。

## 6. 我的 Intuition 和联想

### 6.1 这其实是 process reward model 的 embodied 版本

OpenAI 的 "Let's Verify Step by Step" (https://arxiv.org/abs/2305.20050) 和后续 PRM 工作的核心 idea 是：在数学推理这类多步任务上，给每一步 process reward 比只给 outcome reward 训练效果好得多。LARM 的 referee RL 本质上是在 embodied 任务上做 PRM——GPT-4 当 PRM，给每个 skill step 打分。

差异是：
- Math PRM: 人类标 step correctness
- LARM Referee: GPT-4 自动判 step correctness

这个 framing 让我想到，是否可以用 OpenAI 的 o1 / o3 这类 reasoning model 当 referee，可能比 GPT-4o 更准。

### 6.2 和 RLHF 的结构相似性

LARM 的训练流程其实很像 RLHF：
- LLM pre-training (webpage data) ≈ SFT
- Referee RL ≈ RLHF, where GPT-4 plays the role of reward model

差异是 RLHF 的 reward model 是单独训的 NN，LARM 的 reward model 是 GPT-4 直接 zero-shot 当 judge。这种 "LLM-as-a-judge" 思路跟 Constitutional AI (https://arxiv.org/abs/2212.08073) 也类似。

### 6.3 跟 AlphaCode / AlphaGeometry 的 search 范式对比

AlphaCode 用 LLM 生成大量 candidate solution 然后用 execution feedback 筛。AlphaGeometry 用 LLM 提供 auxiliary construction，用 symbolic engine 验证。这两个都是 "LLM + verifier" 范式。

LARM 是 "LLM-as-policy + LLM-as-verifier"，但 policy 是 lightweight 的，verifier 是 giant 的。这种 asymmetric 设计在 compute budget 受限时很合理。

### 6.4 潜在的问题和我的疑问

读 paper 时我有几个疑虑：

**1. GPT-4 调用成本**。每步都要调 GPT-4 判一次， enchanted diamond sword 训练 50 步 chain × 多少 episode？42 小时训练里 GPT-4 API 调用估计是六位数到七位数 token 量。这个成本 paper 没明说。如果做大规模 replication，API 费用是个隐性瓶颈。

**2. Referee 的可靠性**。GPT-4 在 Minecraft 这种 domain 上其实也不完美。paper 没报告 referee 自身的 accuracy。如果 referee 有 20% 误判率，long chain 上错误信号累积，可能 cap 了 final success rate。16/30 的 enchanted sword 成功率，上限可能部分受 referee noise 限制。

**3. Skill library 的依赖**。LARM 的 action 是 skill，不是 atomic action。Mineflayer 的 API skill 几乎确定性执行成功，MineDojo 的 RL skill 不稳定。这说明 LARM 的"generalization"其实是 high-level decision making 的 generalization，不是 low-level motor control。如果换到 real robot 上没有 perfect skill API，整个 pipeline 会更难。这跟 Voyager / DEPS 的局限一样。

**4. Action space 的 combinatorics**。paper 没说具体有多少个 skill。如果 skill library 里有 K 个 skill，action head 输出 K-way classification。K 多大？这影响 head 容量和 feature matching 的 noise。

**5. Critic 和 actor 共享 backbone 的稳定性**。LLM decoder 当 shared trunk 训 critic 通常不稳，因为 value scale 和 action logit scale 差很多。paper 没详细讲 head 设计。

### 6.5 与 Voyager 的对比直觉

Voyager (https://arxiv.org/abs/2305.16291) 用 GPT-4 在 inference 时做 planning，zero-shot 调度 skill。它的 "training-free" 优势是无须训练，劣势是 inference 慢且依赖 GPT-4 永远可用。

LARM 把 GPT-4 的知识通过 referee RL **bake 进** lightweight policy，inference 时不再需要 GPT-4。这是典型的 "training cost 换 inference efficiency" trade-off。

更深一层：Voyager 是 LLM-as-controller，skill library 是 external。LARM 是 LLM-as-policy，skill library 是 action space。前者把 LLM 当符号系统用，后者把 LLM 当 function approximator 用——更接近 RL 传统范式。

### 6.6 关于 long-horizon credit assignment 的更广联系

reward vanishment 这个问题其实在好多领域都出现：
- **Language model long-context**: chain-of-thought 长，gradient 在早期 token 上 vanish（虽然 mechanism 不同，是 BPTT 时的 gradient flow）
- **Multi-agent RL**: joint reward 拆到 individual agent credit assignment
- **Hierarchical RL**: high-level policy 的 sparse reward 问题
- **Chess/Go 的棋局评估**: AlphaGo 用 value network 做 dense position evaluation，本质也是给中间 step 提供 dense signal

LARM 的 referee 方案是 dense reward shaping 的一种 instance。reward shaping 的经典理论 (Ng et al. 1999, https://arxiv.org/abs/cs/9903064) 说 potential-based shaping 不改变 optimal policy。LARM 的 auxiliary reward 不是 potential-based，所以理论上可能改变 optimal policy——但因为 environment reward R 仍然只在 terminal 给，最终 objective 还是导向 task completion，auxiliary reward 更像 learning scaffold。

### 6.7 跟最近 VLA (Vision-Language-Action) 工作的关系

LARM 在结构上很像 RT-2 (https://arxiv.org/abs/2307.15818) 和 OpenVLA (https://arxiv.org/abs/2406.09246)——都是 LLM backbone + vision input + action output。差异：

- RT-2/OpenVLA: action 是 continuous robot action (7-DoF end-effector)，token 化成 discrete bins
- LARM: action 是 discrete skill ID，更 high-level

LARM 的 skill-based action space 更适合 Minecraft 这种开放世界，但限制了在 real robot 上的直接 transfer。如果要做 real-world embodied agent，可能需要 LARM-style high-level planner + RT-2-style low-level controller 的 hierarchy。

### 6.8 关于 future work 的联想

paper 提了 multi-agent collaboration case (Fig. 3)，但没展开。这个方向很有意思——如果多个 LARM agent 协作，referee 可以扩展成 judge joint state transition。这跟 Multi-agent RLHF 的近期工作有交集。

另一个方向是 **self-play referee**：如果 LARM 训到一定程度，能否让它自己当 referee？这样能摆脱 GPT-4 依赖。类似 STaR (https://arxiv.org/abs/2203.14465) 的 self-taught reasoner 思路，或者 SPIN (https://arxiv.org/abs/2401.01335) 的 self-play fine-tuning。

还有 **hierarchical referee**：低层 referee 判 atomic action correctness，高层 referee 判 sub-goal progress。这在 50+ 步 chain 上可能比单一 referee 更准。

## 7. 总结直觉

LARM 这篇 paper 在我的直觉里是这么定位的：

它是第一个把 **LLM-as-policy + RL fine-tuning** 在 embodied long-horizon 任务上做 work 的工作。之前要么 LLM zero-shot 调度 (Voyager)，要么 RL 从头训 (MineAgent)，要么 SFT 蒸馏 (STEVE)。LARM 用 PPO 训一个 LLM-based policy，靠 GPT-4 当 dense reward provider 解决 credit assignment。

数学上 reward vanishment 的分析虽然简单但清晰，四分法 reward 设计虽然 heuristic 但 ablation 证明有效。Enchanted diamond sword 这个 milestone 跨越了之前所有方法的 SOTA boundary。

局限是 skill library 依赖、GPT-4 API 成本、referee reliability 上限。但这些是工程问题，不是 fundamental barrier。

如果让我预测下一步：把 LARM 的 referee 换成 o3 reasoning model，把 skill library 换成 RT-2 style low-level policy，可能能 push 到 real-world embodied long-horizon task。或者用 self-play 让 LARM 自己当 referee，摆脱对外部 giant LLM 的依赖。

主要参考链接：
- LARM 主页: https://lizhuoling.github.io/LARM_webpage/
- MineDojo: https://minedojo.org/
- Voyager: https://arxiv.org/abs/2305.16291
- STEVE: https://arxiv.org/abs/2311.15209
- TinyLLaVA: https://arxiv.org/abs/2402.14289
- PPO: https://arxiv.org/abs/1707.06347
- LoRA: https://arxiv.org/abs/2106.09685
- CLIP: https://arxiv.org/abs/2103.00020
- LLaVA: https://arxiv.org/abs/2304.08485
- PRM (Let's Verify Step by Step): https://arxiv.org/abs/2305.20050
- Constitutional AI: https://arxiv.org/abs/2212.08073
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- Reward shaping theory (Ng et al.): https://arxiv.org/abs/cs/9903064
- STaR: https://arxiv.org/abs/2203.14465
- Mineflayer: https://github.com/PrismarineJS/mineflayer

Andrej，你直觉上觉得这个 referee paradigm 能 scale 到 real robot 吗？我一直在想 GPT-4 当 judge 在物理 state 描述上够不够 rich——Minecraft 的 inventory list 是 symbolic 的，real robot 的 state 是 continuous 的，要 prompt GPT-4 判 "这个 grasp 动作是否在正确方向" 可能需要把 state 转成符号描述，这步本身会丢信息。
