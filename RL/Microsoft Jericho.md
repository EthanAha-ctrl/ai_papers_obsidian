




好的，让我为你详细解析 **Microsoft Jericho** 这个项目。

---

# 🏰 Microsoft Jericho — Interactive Fiction (IF) Games 的 Reinforcement Learning (RL) 环境

**GitHub**: https://github.com/microsoft/jericho  
**论文**: [Interactive Fiction Games: A Colossal Adventure (AAAI 2020)](https://arxiv.org/abs/1909.05398)  
**博客**: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/by-making-text-based-games-more-accessible-to-rl-agents-jericho-framework-opens-up-exciting-natural-language-challenges/)

---

## 一、核心目标：这个 Project 到底做什么？

Jericho 是一个**轻量级 Python 接口**，将 **Reinforcement Learning (RL) Agent** 与**人类编写的 Interactive Fiction (IF) 游戏**连接起来。

### 什么是 Interactive Fiction (IF)?
IF 游戏就是**纯文字冒险游戏**（如经典的 *Zork*、*Hitchhiker's Guide to the Galaxy*）。玩家看到一段文字描述场景，然后**输入自然语言指令**（如 `open mailbox`、`go north`、`take sword`），游戏引擎返回新的文字描述。

**关键挑战**：与 Atari 等游戏不同，IF 游戏的 **action space 是组合爆炸的自然语言空间**，而非有限的 discrete button set。

---

## 二、技术架构：从第一性原理理解

### 2.1 问题建模：POMDP (Partially Observable Markov Decision Process)

每个 IF 游戏被建模为一个 POMDP tuple：

$$\langle S, T, A, \Omega, O, R, \gamma \rangle$$

其中各变量含义：

| 符号 | 含义 |
|------|------|
| $S$ | **State space** — 游戏的所有内部状态（玩家看不到完整状态） |
| $T$ | **Transition function** $T: S \times A \rightarrow S$ — 执行 action 后 state 如何变化 |
| $A$ | **Action space** — 所有可能的 text command（组合爆炸！） |
| $\Omega$ | **Observation space** — agent 能看到的 text description |
| $O$ | **Observation function** $O: S \times A \rightarrow \Omega$ — 给定 state 和 action，返回什么文字 |
| $R$ | **Reward function** $R: S \times A \rightarrow \mathbb{R}$ — 游戏分数变化 |
| $\gamma$ | **Discount factor** — 未来 reward 的折扣率 |

**为什么是 Partially Observable?** 因为 agent 只能看到当前房间的文字描述（$\Omega$），而不能看到整个游戏世界的完整状态（$S$），比如其他房间里物品的位置、NPC 的状态等。

### 2.2 Action Space 的组合爆炸问题

这是 IF 游戏最核心的挑战。假设一个游戏的 vocabulary 有 $|V|$ 个词，最大 command 长度为 $L$，则：

$$|A_{naive}| = |V|^L$$

例如 $|V| = 700$，$L = 5$，则 $|A| \approx 700^5 \approx 1.68 \times 10^{14}$！这使得 naive enumeration 完全不可行。

### 2.3 Jericho 的三大核心功能

#### ✅ 功能一：Z-Machine Emulator 接口

Jericho 底层封装了 **Frotz**（一个经典的 Z-Machine interpreter）。Z-Machine 是 **Infocom** 在 1979 年发明的 virtual machine，专门运行 IF 游戏。游戏文件格式为 `.z3`, `.z5`, `.z8` 等。

```
[Python Agent] ↔ [Jericho Python API] ↔ [Frotz C Engine] ↔ [Z-Machine Game File (.z5)]
```

关键 API：
```python
env = jericho.FrotzEnv("zork1.z5")
obs, info = env.reset()
obs, reward, done, info = env.step("open mailbox")
```

#### ✅ 功能二：Valid Action Detection (VAD)

这是 Jericho 最重要的技术创新！它利用了 Z-Machine 的 **world-change detection** 来判断一个 action 是否"有效"。

**原理**（第一性原理推导）：

1. **Save** 当前 game state $s_t$
2. 执行 candidate action $a$
3. **检查**：游戏的 internal world state 是否发生了变化？（通过比较 Z-Machine 的 RAM 中的 **world object tree**）
4. **Restore** 回 $s_t$
5. 如果 world state 变化了 → $a$ 是 **valid action**；否则是 invalid

这个过程是 **"speculative execution"**：试探性地执行每个候选 action，看看是否真正改变了游戏世界。

$$\text{Valid}(a, s_t) = \begin{cases} 1 & \text{if } \text{WorldState}(T(s_t, a)) \neq \text{WorldState}(s_t) \\ 0 & \text{otherwise} \end{cases}$$

#### ✅ 功能三：Template-Based Action Generation

为了缓解 action space 的组合爆炸，Jericho 引入了 **template** 机制：

**Template** = 带有 slot 的 action pattern，例如：
- `take OBJ`
- `put OBJ in OBJ`
- `open OBJ with OBJ`
- `go DIRECTION`

每个游戏都有一组预定义的 template（从 game file 中提取）。Agent 只需要：

1. **选择一个 template** $t_i$（如 `put OBJ in OBJ`）
2. **填充 slot** 中的 object（从当前 observation 中提取的 **interactive object**）

这将 action space 从 $|V|^L$ 降低到：

$$|A_{template}| = \sum_{i=1}^{|T|} |O|^{k_i}$$

其中 $|T|$ 是 template 数量，$|O|$ 是当前可见的 interactive object 数量，$k_i$ 是 template $i$ 中的 slot 数量。

例如：30 个 template，10 个 visible object，平均 1.5 个 slot → $|A_{template}| \approx 30 \times 10^{1.5} \approx 949$，远小于 $10^{14}$！

---

## 三、Benchmark Agent 架构

论文中实现了几个 baseline agent：

### 3.1 NAIL (Navigate, Acquire, Interact, Learn)

一个基于规则的 **heuristic agent**：
- 系统地探索地图
- 捡起所有可拾取的物品
- 尝试所有交互操作

### 3.2 DRRN (Deep Reinforcement Relevance Network)

基于 deep RL 的 agent：

$$Q(s, a) = f(\phi_s(s))^\top g(\phi_a(a))$$

其中：
- $\phi_s(s)$：将 text observation 编码为 state embedding（用 GRU/LSTM）
- $\phi_a(a)$：将 candidate action 编码为 action embedding（用另一个 GRU/LSTM）
- $f, g$：全连接层
- $Q(s, a)$：estimated Q-value（state-action pair 的预期累积 reward）

训练使用标准 **DQN loss**：

$$\mathcal{L} = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right]$$

其中 $\theta^-$ 是 **target network** 的参数（定期从 $\theta$ 复制，用于稳定训练），$r$ 是 immediate reward，$s'$ 是 next state。

### 3.3 Template-DQN

结合 template 机制的 DQN：

$$Q(s, t, o_1, ..., o_k) = Q_{template}(s, t) + \sum_{j=1}^{k} Q_{object}(s, o_j)$$

先选 template $t$（用一个 Q-network），再选填入各 slot 的 object $o_j$（用另一个 Q-network）。这是一种 **factorized action-value decomposition**。

---

## 四、游戏集合

Jericho 包含 **57 个经典 IF 游戏**，涵盖不同难度级别：

| 难度分类 | 示例游戏 | 特征 |
|----------|---------|------|
| Possible | *Zork1*, *Detective* | 较小的 state space，较少的 puzzle dependency |
| Difficult | *Enchanter*, *Wishbringer* | 需要多步推理 |
| Extreme | *Hitchhiker's Guide*, *Spellbreaker* | 高度复杂的 puzzle chain，大量 red herring |

每个游戏附带：
- **Max score**（最高可获得分数）
- **Walkthrough**（最优解步骤）
- **Template set**（该游戏的 action template）
- **Vocabulary**（该游戏的有效词汇）

---

## 五、为什么这个项目重要？（从 AI Research 角度）

### 5.1 对 NLU (Natural Language Understanding) 的挑战

IF 游戏要求 agent 理解：
- **Spatial reasoning**（地图导航）
- **Common sense**（黑暗中需要灯）
- **Long-term planning**（拿到钥匙 → 去锁着的门 → 开门）
- **Language grounding**（文字描述 → 世界模型）

### 5.2 对比其他 Text Game 环境

| 环境 | 特征 |
|------|------|
| **TextWorld** (Microsoft) | 程序生成的游戏，可控难度 |
| **Jericho** (Microsoft) | 人类编写的真实游戏，更复杂自然 |
| **ScienceWorld** | 科学实验场景 |
| **ALFWorld** | 将 text game 与 embodied AI 结合 |

Jericho 的独特价值在于它使用的是**真正的、为人类设计的游戏**，而非简化的合成环境。

### 5.3 与 LLM 时代的关联

随着 GPT-4、Claude 等 LLM 的出现，IF 游戏成为测试 LLM **agent 能力**的天然 benchmark：
- **Planning ability**
- **Memory management**（长 context 利用）
- **Tool use**（与游戏引擎交互）

---

## 六、代码示例

```python
import jericho

# 加载游戏
env = jericho.FrotzEnv("roms/zork1.z5")

# Reset
obs, info = env.reset()
print(obs)  # "West of House. You are standing in an open field..."

# 获取 valid actions
valid_actions = env.get_valid_actions()
# e.g., ['open mailbox', 'go north', 'go south', ...]

# 执行 action
obs, reward, done, info = env.step("open mailbox")
# obs: "Opening the small mailbox reveals a leaflet."
# reward: 0 (no score change)

# 获取游戏分数
score = env.get_score()
max_score = env.get_max_score()
```

---

## 七、总结：第一性原理 Intuition

```
核心 Insight:
IF 游戏 = 自然语言 + 部分可观测 + 组合爆炸的 action space + 长期规划

Jericho 解决的核心问题:
1. 如何让 RL agent 与 legacy Z-Machine 游戏交互？ → Frotz 封装
2. 如何从 10^14 个可能的 action 中找到有意义的那些？ → Valid Action Detection (speculative execution)
3. 如何结构化 action space？ → Template-based decomposition
4. 如何提供标准化 benchmark？ → 57 个游戏 + walkthrough + evaluation metrics
```

这个项目从根本上是在问一个问题：**AI 能否在完全由自然语言构成的世界中进行探索、推理和规划？** 这比 Atari 游戏更接近真实世界的 language grounding 挑战。

---

**参考链接**：
- 📄 论文: https://arxiv.org/abs/1909.05398
- 💻 代码: https://github.com/microsoft/jericho
- 📝 Microsoft Blog: https://www.microsoft.com/en-us/research/blog/by-making-text-based-games-more-accessible-to-rl-agents-jericho-framework-opens-up-exciting-natural-language-challenges/
- 📚 Survey on Text Games for RL: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00495/112801