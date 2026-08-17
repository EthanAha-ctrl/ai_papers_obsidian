---
source_pdf: Can VLMs Play Action Role-Playing Games Take.pdf
paper_sha256: 355e31c2fa4ebc3658eb9d8142be8402b2dc7c8cb8741d5d73723cf816adbf11
processed_at: '2026-08-03T14:54:11-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话说清楚这帮人在干嘛

Alibaba 的一帮人想把 GPT-4o 这种能看图的 AI 塞去玩《黑神话:悟空》, 看它能不能像人一样, 看着屏幕, 手按键盘鼠标, 把怪打死。

就这么简单。但里面有很多坑, 一个一个讲。

---

## 为什么要做这件事 — 背景的 intuition

先说现在 game AI 的三大流派, 每一流派都有自己的死穴。

### 流派一: 给游戏开个后门 (API-based)

代表作 Voyager, 让 GPT-4 玩 Minecraft。做法是游戏方提供一套 API, AI 可以直接读到 "我现在在 (100, 64, 200) 坐标, 前面 3 格有一棵橡树, 背包有 5 块木头" 这种结构化数据。AI 想了一下, 输出 `mine(oak_tree)`, 游戏就帮你执行了。

听起来很美, 问题在于: **99% 的游戏不开 API**。你去找 FromSoftware 说 "哥们把 Elden Ring 的 API 给我开一下我训个 AI", 人家看你像看傻子。商业游戏公司对自己的 memory layout 守口如瓶, 怕作弊怕得要死。所以这条路只能走 Minecraft 这种 moddable 的 sandbox。

### 流派二: 死磕强化学习 (RL)

代表作 DQN-play-Sekiro (https://github.com/analoganddigital/DQN_play_sekiro)。做法是把游戏画面当 state, 把按键当 action, 用 Deep Q-Network 学一个 policy $\pi(a|s)$。reward 就是 "打掉 boss 血" +1, "自己死" -1。

公式的话, DQN 在最小化 Bellman error:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]$$

变量解释:
- $s$ 是当前帧, $a$ 是动作, $r$ 是奖励, $s'$ 是下一帧
- $\theta$ 是网络参数, $\theta^-$ 是 target network 参数 (慢更新那个)
- $\gamma \in [0,1)$ 是 discount factor, 决定多看重未来 reward
- $\mathcal{D}$ 是 replay buffer, 存历史 transition

这个方法的问题, 一句话: **训完只能打这一个 boss**。Sekiro 你训了 200 万步学会打剑圣苇名一心, 换成打屑太郎? 对不起, 重新训 200 万步。这叫 **poor generalization**。原因是 RL 学到的是 "这个具体像素分布下, 按哪个键 reward 最高", 它根本不理解 "什么是剑, 什么是人, 什么是攻击动作"。它学的是个 lookup table, 不是概念。

而且 BMW 这种游戏 boss 有几十个, 你 RL 训一遍要一年, 黄花菜都凉了。

### 流派三: 看屏幕玩 (VLM-based)

代表作 Cradle (https://github.com/BAAI-Agents/Cradle), 让 GPT-4o 玩 RDR2 (荒野大镖客 2)。做法是截图 → GPT-4o 看一眼 → 输出 Python code 控制键盘鼠标。听起来最接近人怎么玩游戏。

但 Cradle 有个致命依赖: **它需要画面里有大量文字提示**。RDR2 是个慢节奏 game, 画面里到处是任务提示 "Go to Valentine", "Talk to Dutch", 菜单上都是字。VLM 靠 OCR 这些字就能拿到不少信息。

但 BMW 这种 ARPG, 战斗中画面里**几乎没字**。你打 Bullguard 的时候, 画面就是一头牛头怪举着斧头冲过来, 没有任何字幕说 "Bullguard 将要使用 charge attack, 建议左闪"。VLM 没字可读, 当场抓瞎。

所以 ARPG 是 VLM 的盲区。paper 选取 BMW, 就是要正面刚这个盲区。

---

## VARP 怎么做 — 直觉版

### 核心 idea: 把"反应"变成"选技能"

BMW 战斗是 60fps, 你要让 VLM 60 次每秒做决策? 不可能, GPT-4o 一次推理要好几秒。

那怎么办? 人也不是每一帧都在思考啊! 人是这么玩的:
1. 平时待机, 看着 boss
2. Boss 举斧头 → 脑子里 pattern match "哦这是三连劈"
3. 调出记忆里 "三连劈怎么躲" 的那套动作
4. 执行: 闪 闪 闪 砍 砍

VARP 就是模拟这个过程。它维护一个 **Action Library**, 里面是一堆 Python 函数, 每个函数是一套按键组合, 比如:

```python
def fight_bullguard_three_chop():
    """Counter to Bullguard's three consecutive axe chops.
    Dodge three times then light attack five times."""
    for _ in range(3):
        dodge()
    for _ in range(5):
        light_attack()
```

VLM 每次推理不是输出一个原子动作, 而是从这个库里挑一个合适的函数执行。这样 VLM 一秒思考一次也够用了 — 因为一次决策管接下来好几秒的动作。

这其实就是 **hierarchical control**: 高层 VLM 慢思考选 strategy, 底层 Python 函数快执行 atomic ops。跟人脑的 "System 2 想策略 + System 1 执行肌肉记忆" 挺像。

### Action Library 怎么来 — 三个来源

#### 来源 1: Predefined (人工写)
刚开始手动写几个常用 action, 比如 `light_attack_combo()`, `dodge_n(n)`, `recover_health()`, `fight_immobilization_spell_skill()`。每个函数配详细 text annotation, 用 OpenAI text-embedding-ada-002 编码成 1536 维向量存着。

#### 来源 2: SOAG 自己悟出来的

这个最有趣。假设你跟 Bullguard 打, 打着打着 GPT-4o 发现: "咦这牛头怪每次举斧头举到最高点之后, 0.8 秒就会三连劈下来"。它就把这个观察变成一个新 action:

```python
def fight_new_action_bullguard_raise_weapon():
    """When Bullguard raises weapon to highest point, 
    dodge 4 times then attack 5 times."""
    for _ in range(4):
        dodge()
    for _ in range(5):
        light_attack()
```

存进 library, 下次再见到 "举斧头" 的画面, 直接调这个函数。

更妙的是, 打完几次后, VLM 反思: "其实不用闪 4 次那么多, 闪 2 次中间夹一个反击更效率"。于是 update 这个函数, 加入 attack 间隙:

```python
def fight_new_action_bullguard_raise_weapon_v2():
    """Optimized: dodge, dodge, counter-attack, dodge, dodge, attack x3"""
    dodge()
    dodge()
    light_attack()  # 插入反击
    dodge()
    dodge()
    for _ in range(3):
        light_attack()
```

这就是 paper 说的 "self-optimizable"。**VLM 用自然语言当 reward, 在 context 里做 policy iteration**。没有 backprop, 没有 gradient, 全靠 prompt 让模型自己 "觉得" 怎么样更好。

如果形式化, 可以理解为 VLM 在隐式优化:

$$\pi^* = \arg\max_\pi \; \alpha \cdot \mathbb{E}[N_{\text{dodge\_ok}}] + \beta \cdot \mathbb{E}[N_{\text{hit}}] - \gamma \cdot \mathbb{E}[\Delta \text{HP}_{\text{lost}}]$$

- $\pi$ 是 action function (一个 Python combo)
- $N_{\text{dodge\_ok}}$ 是成功闪避次数
- $N_{\text{hit}}$ 是命中敌人次数
- $\Delta \text{HP}_{\text{lost}}$ 是自己掉血
- $\alpha, \beta, \gamma$ 是 VLM 在 prompt 里隐式权衡的权重 (paper 没显式给数)

这个思路本质上是 **verbal reinforcement learning**, 跟 Reflexion (https://arxiv.org/abs/2303.11366) 的 philosophy 一样: 不用梯度, 用文字反思来改 policy。

#### 来源 3: Human-Guided Trajectory 学人的

有些 very hard 任务 (比如 task 12 自动导航), VLM 完全没概念怎么走。paper 收集了 1000 条人类玩家的录像 + 键鼠 log, 建 Human-Guided Library。

遇到不会的任务:
1. 截当前画面
2. 在人类数据里找一张最像的截图
3. 取那张截图之后 10 秒的录像和操作
4. 让 VLM "看这段人类怎么走", 总结成一个 Python action function

这就是 **retrieval-augmented imitation**。本质是把人当作 oracle, 当 VLM 不会的时候 "翻人类视频教材"。概念上跟 RT-2 (https://arxiv.org/abs/2307.15818) 这种 Vision-Language-Action model 很像, 但 RT-2 输出的是 robot motor torques, VARP 输出的是 discrete Python code。

---

## DTSA — 为什么要把决策模块拆开

这个故事特别有意思, 也是我觉得最有 engineering insight 的部分。

### 问题: Attention 稀释

Transformer 的 self-attention 公式:

$$\text{Att}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

变量:
- $\mathbf{Q} \in \mathbb{R}^{L \times d_k}$ 是 query matrix, $L$ 是 sequence length
- $\mathbf{K} \in \mathbb{R}^{L \times d_k}$ 是 key matrix
- $\mathbf{V} \in \mathbb{R}^{L \times d_v}$ 是 value matrix
- $d_k$ 是 key/query 的维度
- softmax 在最后一维做归一化

关键: softmax 归一化是 **相对** 的。$L$ 从 100 涨到 5000, 每个位置分到的 attention weight $\alpha_j = \frac{\exp(\mathbf{q}\cdot\mathbf{k}_j/\sqrt{d_k})}{\sum_{j'} \exp(\mathbf{q}\cdot\mathbf{k}_{j'}/\sqrt{d_k})}$ 整体会被摊薄。

BMW 的 prompt 有多长? 多帧 1080p 截图 + OCR 文本 + 历史 reflection + action library description + CoT 历史, 轻松 5000+ tokens。关键信息 (比如 "血条只剩 20%") 淹没在大量无关 image patches 里, GPT-4o 就会忘掉。

paper 里这叫 "forgetting and hallucination", 实际就是 "Lost in the Middle" 现象 (参考 https://arxiv.org/abs/2307.03172): 中间的 token 最容易被忽略。

### 解法: 拆成 5 个 sub-module, 各管一摊

原来一个大 prompt 问 5 个问题: "敌人啥状态? 用啥战斗方式? 要不要回血? 用不用 spell? 最终选哪个 action?" 现在拆成 5 个独立 VLM 调用, 每个只问一个问题, 输入只塞相关的截图 crop。

| Sub-module | 只看什么 | 只回答什么 |
|---|---|---|
| Enemy Sub-module | 敌人 HP bar + 敌人位置 bounding box | 敌人状态摘要 |
| Combat Sub-module | 右下角 heavy attack icon | 用 light 还是 heavy |
| Health Sub-module | 玩家 HP bar | 要不要 recover_health |
| Spell Sub-module | spell skill 冷却 icon | 要不要放 spell |
| Integration Sub-module | 上面 4 个输出 | 最终选哪个 action |

paper 把这个比作 MLP, 我觉得比得有点牵强, 更准确的类比是 **mixture-of-experts 的人工版**: 每个 expert 处理一个 slice of input, gating function 是 Integration module。

效果: ablation 里去掉 DTSA, easy 任务也下降 — 因为 GPT-4o 经常忘记看血条, 血没了也不回血, 直接被打死。

---

## 实验结果的人话解读

### VARP vs 人类新手 (Figure 3)

- **Easy 任务 (1-8)**: 双方都接近 100%, 没区分度
- **Middle 任务 (9, Crow Diviner)**: VARP 40%, 人类也低
- **Hard 任务 (10, Bullguard)**: VARP 20% vs human 15.63% — **VARP 居然赢了人类新手!**
- **Very Hard (11, Wandering Wight)**: 双方都很低, VLM 推理太慢抓不到高速 telegraph
- **Very Hard (12, Navigation)**: 无人类指导 VARP 0%, 有指导 40%

Bullguard 这个数据最有意思 — 为什么 VARP 能超过人类新手?

我推测几个原因:
1. **人类新手会慌**。Bullguard 举斧头, 新手要么乱滚, 要么愣住。VARP 不会慌, 见到 raise weapon 直接调 `fight_new_action_bullguard_raise_weapon_v2()` 冷静执行
2. **人类新手冗余操作多**。Table 3 显示 task 10 人类平均 36.6 次推理 vs VARP 13.5 次。人类按了很多无效按键, VARP 每个 action 都是 deliberate 的
3. **但样本数存疑**。15.63% 如果是 16 人中 2.5 人成功, 这个比较 statistically 有点弱

### Ablation (Figure 4)

去掉 SOAG: middle/hard 任务掉得明显。逻辑链: hard 任务敌人 HP 高 → 战斗时间长 → 需要 SOAG 持续学习敌人 pattern → 没 SOAG 就一直用初始烂动作 → 打不过。

去掉 DTSA: easy 任务也掉。逻辑链: 没 DTSA → 长 prompt → attention 稀释 → GPT-4o 忘了看血条 → 血低不回血 → 简单怪也能被打死。

这俩 ablation 都 self-consistent, 说明 paper 的两个 contribution 都不是凑数的。

---

## 我觉得最 cool 和最 problematic 的几点

### Cool

1. **SOAG 这个 in-context verbal RL 思路很有想象力**。完全不用梯度, 靠 VLM 自己反思就能 evolve policy。虽然现在只组合 `dodge` 和 `light_attack` 两个原子操作, 但思路可以扩展到更多原子操作, 可能是 LLM agent 自我进化的一个 practical path

2. **Photo mode 暂停游戏做推理**这个 hack 很机智。虽然不是真实部署, 但作为 research 把 real-time constraint 暂时绕过去, 让我们能 isolate "视觉理解 + 动作规划" 的能力本身, 不被 latency 问题淹没

3. **数据集开源**。1000 条 BMW 的人类玩 + 键鼠 log, 这东西对社区有用, 可以做 imitation learning、behavior cloning、甚至 inverse RL

### Problematic

1. **Photo mode 是作弊**。ARPG 的核心是 real-time pressure, 你暂停了游戏, 这就完全不是 ARPG 了。真实部署怎么办? paper 没回答。要让这玩意儿 work, VLM 推理速度至少要 10x, 或者用一个小模型蒸馏

2. **SOAG 的 "reward" 是 VLM 自己用文字描述的**, 没 quantitative metric。同样打 Bullguard, 不同 prompt engineering 出来的 combo 可能完全不同。reproducibility 差

3. **Human-Guided Library 检索只取 top-1**。这非常脆弱 — 如果那张最像的截图恰好是人类失误的画面, VARP 就学到错了。应该 top-k + rerank, 或者用 cosine similarity 阈值过滤

4. **Very Hard 任务 (task 11) 90% 人类数据都被剔除了**, 这意味着 SOAG 和 Human-Guided 在最需要的任务上反而没数据可用。这是个 data bottleneck

5. **只测了 BMW 一个游戏**。BMW 第一章 boss 攻击 pattern 相对套路化 (raise weapon → 0.8s 后三连劈), 换成 Elden Ring 的 Malenia 这种有 waterfowl dance 复杂多阶段连招的, SOAG 还能不能悟出来存疑

---

## 这篇 paper 在大图里的位置

我把它放在这个 lineage 里:

```
LLM agent + text world
    Reflexion → ReAct → AutoGPT
        │
        ▼
LLM agent + game API
    Voyager (Minecraft)
        │
        ▼
VLM agent + screen (text-heavy games)
    Cradle (RDR2)
        │
        ▼
VLM agent + screen (action games, text-sparse)  ← 这篇 VARP 在这
    VARP (BMW)
        │
        ▼  (未来)
Real-time VLM agent, 无暂停, 多游戏泛化
    ?
```

VARP 是第一次把 VLM agent 拉进 ARPG 这个 "地狱难度" 场景, 证明 "off-the-shelf GPT-4o + 一个像样的 agent framework" 可以做到接近人类新手水平。但离真正"打 BMW 像人一样"还远 — photo mode 这个 cheat 拿掉之后, 整个 framework 能不能活下来, 是下一步要解决的核心问题。

我个人觉得未来最 promising 的方向是:
1. **小 VLM 蒸馏** — 把 GPT-4o 的决策能力蒸馏到一个 1B-7B 的小 model, 推理从 1fps → 15fps, 能跟得上 ARPG 节奏
2. **Diffusion policy 当底层** — VLM 高层做 strategy, 底层用一个 diffusion policy (类似 Diffusion Policy https://diffusion-policy.cs.columbia.edu/) 做连续动作生成, 比纯 Python combo 更 smooth
3. **多游戏 benchmark** — BMW 一个游戏说明不了泛化, 需要 Elden Ring、Sekiro、God of War 都做一遍

参考链接汇总:
- VARP 项目主页: https://varp-agent.github.io/
- Cradle (前作): https://github.com/BAAI-Agents/Cradle
- Voyager: https://github.com/MineDojo/Voyager
- Reflexion: https://arxiv.org/abs/2303.11366
- RT-2: https://arxiv.org/abs/2307.15818
- Lost in the Middle: https://arxiv.org/abs/2307.03172
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- DQN-play-Sekiro: https://github.com/analoganddigital/DQN_play_sekiro
- VideoGameBunny (未来方向): https://video-game-bunny.github.io/

如果你想让我深挖某一块, 比如 SOAG 具体的 prompt 长什么样、或者 DTSA 各 sub-module 之间是怎么传信息的、或者 Human-Guided Library 的截图相似度怎么算的, 告诉我, 我接着展开。

---

# VARP: 用 VLM 玩 AAA 级 ARPG 的研究

这篇 paper 是 Alibaba Group 团队的工作, 选用了刚发布不久的国产 AAA 级 ARPG **Black Myth: Wukong (BMW, 黑神话:悟空)** 作为研究平台, 探讨当前 Vision Language Models 在 "纯视觉输入 + 复杂动作输出" 场景下的能力边界。项目主页在 https://varp-agent.github.io/。

---

## 1. Why this paper? 动机层面的 intuition

当前 game agent 的研究被三股势力瓜分:

| 方法范式 | 代表作 | 输入形式 | 致命缺陷 |
|---|---|---|---|
| LLM + Game API | Voyager (Minecraft) | API 读取游戏状态 | 大部分 AAA 游戏不开 API |
| RL agents | DQN-play-Sekiro | 像素级 frame | 一个 boss 要训练几百万 step, 换 boss 要重训, zero-shot 泛化几乎为 0 |
| VLM + Screen | Cradle (RDR2) | 游戏截图 | 严重依赖画面里的文字提示 (字幕、菜单、HUD 文本), 一旦进入 ARPG 的实时战斗 (没有字幕, 只有动态视觉), VLM 就"瞎"了 |

BMW 这个游戏的特殊性在于: **画面文字稀疏, 战斗信息高度隐式, 节奏是秒级甚至毫秒级**, 完美暴露了三类方法各自的弱点。这正是一个 ideal 的 stress test。

paper 给的核心 insight 是: ARPG 中, 想纯靠 VLM 的"看一眼+推理+输出"一次到位几乎不可能 (因为 inference latency 在秒级), 但可以靠 **action library + retrieval + self-optimization** 来把 "实时反应" 转化为 "库的选择 + 偶尔创新"。这把 RL 的"反应式"和 LLM agent 的"反思式"做了一个 fusion。

---

## 2. VARP Agent 整体架构图解析

根据 Figure 1 (paper 里的 pipeline), 我把架构用文本拆出来:

```
Game Screenshot (1920x1080)
        │
        ▼
┌─────────────────────────────────────────────┐
│           VARP Agent                          │
│                                               │
│  ┌──────────────────────────────────────┐   │
│  │    Action Planning System            │   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │ Basic VLMs Group             │    │   │
│  │  │  - Information Gathering      │    │   │
│  │  │  - Self Reflection            │    │   │
│  │  │  - Task Inference             │    │   │
│  │  │  - Skill Curation (retrieval) │    │   │
│  │  │  - Decision Making (CoT)      │    │   │
│  │  └─────────────────────────────┘    │   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │ SOAG (生成新动作)             │    │   │
│  │  └─────────────────────────────┘    │   │
│  │  ┌─────────────────────────────┐    │   │
│  │  │ DTSA (5 个并行 sub-module)   │    │   │
│  │  │  Enemy/Combat/Health/Spell/  │    │   │
│  │  │  Integration                 │    │   │
│  │  └─────────────────────────────┘    │   │
│  └──────────────────────────────────────┘   │
│                                               │
│  ┌──────────────────────────────────────┐   │
│  │ Human-Guided Trajectory System      │   │
│  │   screenshot query → top-k → VLM     │   │
│  │   analyze → new action               │   │
│  └──────────────────────────────────────┘   │
│                                               │
│  三大 Library:                                │
│   - Situation Library (历史画面+反思)        │
│   - Action Library  (Python 函数 + embedding)│
│   - Human-Guided Library (screenshot, ops)  │
└─────────────────────────────────────────────┘
        │
        ▼
   Python code (mouse/keyboard ops) → Game
```

关键 design choice: **每一个 action 都是 Python 函数**, 函数注释做 embedding 后存到 Action Library。这其实就是把"动作"当作"技能代码"管理, 借用了 Voyager 的 skill library 思想, 但把 Minecraft 的 API 调用换成了 keyboard/mouse 的原子指令组合。

---

## 3. Action Planning System 深度解析

### 3.1 Basic VLMs Group (沿用 Cradle 的五个模块)

五个模块形成一个完整的 perceive-reflect-plan-act 闭环。下面把每个模块要做的视觉-语言对应说明清楚, 这对 build intuition 很关键。

#### (1) Information Gathering
- **Text 信息**: OCR 工具识别画面字幕、任务提示、通知 (如 "Bullguard 出现" 这种 popup)
- **Visual 信息**: Grounding DINO (https://github.com/IDEA-Research/GroundingDINO) 做开放集 object detection, 定位 character、enemy、interface icon 的 bounding box
- 输出写入 Situation Library

#### (2) Self Reflection
- 输入: 上一段视频帧序列 (因为 VLM 推理慢, BMW 用 photo mode 暂停游戏后再推理)
- 判断: 上一个 action 是否产生了预期效果? 任务是否完成?
- 失败时要 verbal reason 出失败原因, 写回 library 给后续 module 用

#### (3) Task Inference
基于 reflection 输出当前要做的子任务文本描述。这步的输出 text embedding 会用于下一步的 skill retrieval。

#### (4) Skill Curation (检索核心)
设 action library 为 $\mathcal{A} = \{(f_i, \mathbf{e}_i)\}_{i=1}^{N}$, 其中:
- $f_i$ 是第 $i$ 个 Python action function
- $\mathbf{e}_i \in \mathbb{R}^{d}$ 是用 OpenAI text-embedding-ada-002 生成的注释向量, $d=1536$

任务描述的 embedding 记为 $\mathbf{e}_q$, 则检索的相似度用 cosine:

$$\text{sim}(\mathbf{e}_q, \mathbf{e}_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_i}{\|\mathbf{e}_q\|_2 \cdot \|\mathbf{e}_i\|_2}$$

返回 top-$k$ 形成候选集 $\mathcal{C} \subset \mathcal{A}$。这里 $k$ 取若干个, 给 Decision Making 留多样选择。

#### (5) Decision Making (CoT 推理)
用 Chain-of-Thought 串起来回答一连串问题:
1. 当前是否进入战斗模式?
2. 是否需要回血?
3. 哪个 spell skill 可用?
4. 从 $\mathcal{C}$ 中挑哪一个 action 执行?

### 3.2 SOAG: Self-Optimizable Action Generation Module

这是这篇 paper 最有意思的部分, 解决了一个硬伤: **预定义的 action library 不可能覆盖所有 boss 的攻击 pattern**。BMW 里 Bullguard 这种 boss 的攻击动作有 "charging forward with axe" / "chopping the axe downwards three times consecutively" 这种 ARPG 玩家都懂的 attack telegraph, VLM 必须从零观察并设计 counter combo。

SOAG 的输入:
- Information Gathering 输出 + Self Reflection 输出
- 当前帧 $I_t$ 与上一关键帧 $I_{t-1}$
- 历史上这个敌人的攻击 pattern 摘要

输出是一个新的 Python 函数, 主体由两个原子操作组合:
- `dodge()` (闪避)
- `light_attack()` (轻击)

优化目标 (paper 里用文字描述, 我把 intuition 形式化):

$$\max_{a \in \mathcal{A}_{\text{combo}}} \; \mathbb{E}_{\text{episode}} \Big[ \alpha \cdot N_{\text{dodge\_success}} + \beta \cdot N_{\text{hit\_enemy}} - \gamma \cdot \Delta \text{HP}_{\text{self}} \Big]$$

其中:
- $N_{\text{dodge\_success}}$ 是成功闪避次数
- $N_{\text{hit\_enemy}}$ 是击中敌人的次数
- $\Delta \text{HP}_{\text{self}}$ 是玩家血量损失
- $\alpha, \beta, \gamma > 0$ 是权重 (paper 没显式给出, 是隐式由 VLM 在 prompt 里权衡)

paper 在 Section A.6 给了一个具体例子: 对 Bullguard 的 "raise weapon" telegraph, 初始 SOAG 生成 `dodge x4 → light_attack x5`, 经过几轮战斗后优化为 "在 dodge 间隔中插入 counter-attack", 大幅提升了 kill efficiency。这本质上是一个 **in-context 的 verbal policy gradient** — VLM 通过文字描述的 reward signal 自己 update policy, 而不是通过 backprop。

参考链接: 这种 self-improve 思路和 Reflexion (https://arxiv.org/abs/2303.11366) 一脉相承, 也类似 Voyager 的 iterative prompting。

### 3.3 DTSA: Decomposable Task-Specific Auxiliary Modules

这部分的 motivation 是 paper 里我最有共鸣的一段: **当 VLM 输入的 token 数过多时, self-attention 会稀释**。

形式化: 对于 transformer 的某层 attention head, 给定 query $\mathbf{q}$ 和 keys $\{\mathbf{k}_j\}_{j=1}^{L}$, attention 权重为:

$$\alpha_j = \frac{\exp(\mathbf{q}\cdot \mathbf{k}_j / \sqrt{d_k})}{\sum_{j'=1}^{L} \exp(\mathbf{q}\cdot \mathbf{k}_{j'} / \sqrt{d_k})}$$

当 $L$ 变大 (BMW 的 prompt 通常 5k+ tokens 包括多帧图像 patch tokens), softmax 把 probability mass 摊薄, 关键信息 (如血条像素位置、敌人 telegraph 帧差) 抢不到 attention。这就是 paper 里说的 "forgetting and hallucination"。

DTSA 的解法: 把原 Decision Making 拆成 5 个并行 sub-module, 每个子模块只关注一个问题:

| Sub-module | 输入关注点 | 输出 |
|---|---|---|
| Enemy Sub-module | 敌人 HP、position、action description | enemy state summary |
| Combat Sub-module | 右下角 heavy-attack 状态 icon | combat mode 选择 |
| Health Sub-module | 玩家 HP bar 像素变化 | 是否触发 recover_health |
| Spell-skill Sub-module | spell skill 冷却 icon | 是否使用 spell |
| Integration Sub-module | 上述 4 个输出 + candidate action set | 最终 action 选择 |

paper 用 MLP 做类比: 5 个 sub-module 就像 MLP 的 hidden units, 各自 specialize 一个 feature, 最后由 Integration 做融合。这个类比其实简化了 — 真正机制是 **reduction in per-module context length**, 让 attention 局部化, 缓解 long-context dilution。

参考 LLM long-context attention 稀释的研究: https://arxiv.org/abs/2307.03172 (Lost in the Middle)。

---

## 4. Human-Guided Trajectory System

### 4.1 数据收集 (Section 3.1)

- 200 名志愿者, 70% 是 BMW 新手
- 2 周时间
- 1000 条 valid records (大量被剔除, 比如 task 11 "Defeat Wandering Wight" 90% 数据被淘汰, 因为新手根本打不过)
- 每条 record = 鼠标键盘 log + 游戏截图序列 + 时间戳
- 部分数据被标记为 "clean", 剔除了过度点击、scroll 等冗余操作

数据分布见 Figure 6, 每个任务的占比: task 1 占 4.0%, task 2 占 12.5%, 等等。

### 4.2 Retrieval + Imitation 的 hybrid

这部分流程:
1. 当前游戏截图 $I_t$
2. 用 image embedding (paper 没明确说, 但暗示是某种 VLM 编码, 比如 CLIP-style image encoder) 在 Human-Guided Library 中找 top-1 相似截图 $I^*$
3. 取 $I^*$ 之后 $n$ 帧 $\{I^*_t, I^*_{t+1}, \dots, I^*_{t+n-1}\}$ 和对应的人操作 $\{o^*_t, \dots, o^*_{t+n-1}\}$
4. 喂给 VLM 让它"看完人类怎么走这段路", 总结出一个 Python action function

这是一个很巧的 **video-conditioned action summarization**, 类似 RT-2 (https://arxiv.org/abs/2307.15818) 的 VLA 思想, 但输出是离散的 Python code 而不是连续 motor torques。

case study (Figure 5): task 12 (autonomous navigation, very hard) 用 GPT-4o + human guidance, 成功率从 0% → 40%。证明了这种 retrieval-augmented imitation 是 work 的。

---

## 5. 实验数据深度解读

### 5.1 12 个任务的难度分级 (Table 2)

| ID | Task | Difficulty |
|---|---|---|
| 1 | Guidance (Defeat Erlang) | Easy |
| 2-5 | WolfScout/WolfStalwart/WolfSwornsword | Easy |
| 3, 6 | Gather / Open | Easy (非战斗) |
| 7-8 | WolfSoldier / Croaky | Easy |
| 9 | Crow Diviner | Middle |
| 10 | Bullguard (第一 boss) | Hard |
| 11* | Wandering Wight | Very Hard |
| 12* | Autonomous Navigation | Very Hard |

### 5.2 主实验 (Figure 3) — 各 VLM 成功率对比

关键观察:
- **Task 1-8 (Easy)**: VARP (GPT-4o) ~100%, Claude 3.5 Sonnet / Gemini 1.5 Pro 也都很高; human novice 接近 100%
- **Task 9 (Middle, Crow Diviner)**: VARP 平均 40%, human 也低
- **Task 10 (Hard, Bullguard)**: VARP 20%, human 15.63% — **VARP 居然超过 human novice**! 这是 paper 的 highlight
- **Task 11 (Very Hard, Wandering Wight)**: VARP 很低, human 也很低, 因为 VLM inference 是秒级 frame-grab, 抓不到 Wight 这种高速攻击的 telegraph
- **Task 12 (Very Hard, Navigation)**: 无 human guidance 时 VARP = 0%, 有 human guidance 时 40%

### 5.3 时间与推理次数 (Table 3)

| Task | GPT-4o time | GPT-4o count | Human count (估算) |
|---|---|---|---|
| 1 | 16.09 min | 71.6 | 98.7 |
| 2 | 0.53 min | 3.8 | 2.3 |
| 9 | 1.24 min | 8.3 | 16.7 |
| 10 | 2.20 min | 13.5 | 36.6 |

human 的 count 估算 = human 的原子操作数 / 8.6 (因为每个 VARP action 平均含 8.6 个原子操作)。

关键 insight: **在难任务 (9, 10) 上, human 做了大量冗余操作 (慌乱中乱按)**, 而 VARP 的 actions 更"精炼"。这点我觉得很有意思 — VARP 不会慌, 但代价是它不会"灵光一现"做出意外但有效的操作。

### 5.4 Ablation Study (Figure 4)

去掉 SOAG: middle/hard 任务显著下降, easy 任务影响不大 — 因为 easy 任务敌人 HP 低, 不需要持续学习攻击 pattern; hard 任务需要 SOAG 来 incrementally 学会 boss 的 telegraph。

去掉 DTSA: easy 任务也下降 — 因为没有 DTSA, VLM 在长 prompt 下会忘掉某些关键信息 (比如血量低要回血), 导致 easy 任务也失败。这是 long-context dilution 的直接证据。

---

## 6. 局限性 (paper 自己承认的)

1. **任务定义还太简单**: 12 个任务都在第一章
2. **只测了 BMW**: 没扩展到其他 ARPG (Elden Ring, Sekiro 等都没测)
3. **数据集小**: 1000 条远不够
4. **VLM 推理太慢**: 这是最根本的瓶颈 — BMW 的战斗是 60fps, VLM 是 1 fps 推理, 注定无法应对 high-APM 场景

paper 在 A.2 暗示未来想训一个 ARPG-specific VLM, 提到了 VideoGameBunny (https://video-game-bunny.github.io/)。

---

## 7. 我对这个工作的整体评价

**优点**:
- 第一次系统把 VLM agent 放到 AAA ARPG 这个 "hard case" 上 benchmark
- SOAG 这个"VLM 当策略网络, 用自然语言做 reward signal 自我优化"的思路很有想象力, 实际上是在做 in-context RL
- DTSA 用 MLP 类比 attention 稀释问题, 虽然不完全准确但工程上有效
- 数据集开源对社区有价值

**潜在的问题 / 我会追问的**:
1. **photo mode 暂停游戏** = 作弊, 这破坏了 ARPG 的核心 — real-time pressure。真实部署时不能用, 那这个 framework 怎么落地?
2. **Bullguard 超过 human novice** 这个结论需要谨慎 — paper 没说 human novice 样本数, 15.63% 这个数字如果是 16 人中 2.5 人打过的比例, 这个 statistical significance 存疑
3. SOAG 的"优化目标"是 VLM 自己用文字描述的, 没有量化的 reward。这导致 reproducibility 差 — 同样的敌人, 不同 prompt 的 SOAG 可能生成完全不同的 combo
4. Human-Guided Library 的检索只取 top-1, 这个非常脆弱。换成 top-k + rerank 可能效果更好
5. 1000 条数据里 90% task 11 被丢弃, 这意味着 very hard 任务的 imitation signal 其实几乎没有 — task 11 的失败可能更多是数据问题, 不是方法问题

参考链接补充:
- Cradle: https://github.com/BAAI-Agents/Cradle
- Voyager: https://github.com/MineDojo/Voyager
- Reflexion: https://arxiv.org/abs/2303.11366
- ReAct: https://arxiv.org/abs/2210.03629
- RT-2 (VLA model): https://arxiv.org/abs/2307.15818
- Lost in the Middle: https://arxiv.org/abs/2307.03172
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- GPT-4o: https://openai.com/index/hello-gpt-4o/
- Claude 3.5 Sonnet: https://www.anthropic.com/news/claude-3-5-sonnet
- Gemini 1.5 Pro: https://deepmind.google/technologies/gemini/
- Black Myth: Wukong 官网: https://www.heishenhua.com/
- VideoGameBunny: https://video-game-bunny.github.io/

如果你想深入某个模块 (比如 SOAG 的 prompt 设计、DTSA 的 sub-module 之间的信息流、或者 Human-Guided Library 的具体检索算法), 我可以再展开讲。
