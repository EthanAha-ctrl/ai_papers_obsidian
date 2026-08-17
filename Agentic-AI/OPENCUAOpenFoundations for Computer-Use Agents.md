---
source_pdf: OPENCUAOpenFoundations for Computer-Use Agents.pdf
paper_sha256: fd1c1e7b4d467acf63fc743c3925c7ceb015e5b1625f62c22295c0b9e2d82a70
processed_at: '2026-08-06T00:26:22-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OpenCUA 用人话讲

Andrej，让我把刚才那堆学术腔翻译成我们平时在 whiteboard 前画图聊天的感觉。

## 一句话版本

他们想教一个模型用电脑，**不是**让它看屏幕猜点哪——而是让它"想一下"再点。点之前自己跟自己说一段话："我看到啥、我刚才干了啥、有没有搞错、下一步该干啥"。就这么个"想"的动作，把成功率从 4% 拉到 45%。

**Project page**: https://opencua.xlang.ai

---

## 1. 这个事情为什么难

你想象一下你要训练一个 agent 代替你用 Chrome 装一个本地 extension。人来做就是：开菜单 → 点 Extensions → 点 Manage Extensions → 开 Developer mode → Load unpacked → 选 Desktop → 选 helloExtension → Select。8 步。

你要是只把"截图 + 这一步点哪"喂给一个 VLM 做监督学习，会发生什么？模型学到的是 $P(a_i | s_i)$——看到这张图就点这个坐标。但问题：

- **它不知道自己之前点过什么**。点完菜单开了一个 dropdown，下一张截图里 dropdown 已经开着，模型会困惑"我是不是该点 Extensions？"——它没有 history。
- **它不知道任务目标是什么**。截图里全是可点的东西：URL bar、tabs、书签、菜单按钮。哪个跟"装 extension"有关？模型分不清。
- **最致命的：它不会纠错**。点错了，弹出一个 login page，模型看到 login page 就懵了——训练数据里都是"正确路径"，它没见过自己搞砸之后怎么爬起来。

paper Figure 1 Left 那条灰色 "Base Recipe" 线就是这件事：直接 SFT 22.6K 条人类轨迹，OSWorld 4.4%。惨。

你跟你的直觉对一下——这就像只让一个小孩看别人骑自行车的视频，没有"你刚才摔了，因为重心偏左，下次往右靠一点"这种反馈。**纯模仿学不会平衡**。

---

## 2. 他们的核心动作：给数据"加嘴"

OpenCUA 干的事情，本质上是**给每一条 state-action pair 加一段内心独白**。

原始数据长这样：

```
screenshot_5.jpg → click(x=0.157, y=0.123)
```

加工后变成：

```
screenshot_5.jpg → 
  "我看到这是 Google 搜索页，搜索框里有 'gpt'。
   但我任务是要搜 'Travelers' 上 Wikipedia 找词义加脚注。
   之前几步：开文档 → 选词 → Ctrl+C 复制 → 切到这个 tab。
   现在搜索框是错的，我得清掉重输。
   下一步：点搜索框选中里面的文字，准备覆盖。"
  → click(x=0.157, y=0.123)
```

注意这里几个 component：

1. **Observation (L3)**: 我看到啥——active app、layout、关键元素
2. **Thought (L2)**: 反思——我之前做了啥、对不对、下一步该干啥
3. **Action (L1)**: 简洁的执行指令

这玩意叫 **reflective long Chain-of-Thought**。三层结构 $L_3 \to L_2 \to L_1$，对应 perception → reasoning → action。

---

## 3. CoT 怎么生成的——不是人写的

22.6K 条轨迹 × 平均 18.6 步 = 大概 40 万个 step。手写 CoT 不现实。

他们用 `claude-3-7-sonnet-20250219` 当三个角色 (Figure 5)：

### Reflector（质检员）
看每一步：
- 前后两张截图对比——这步真的有变化吗？
- action code 对不对？CoT 跟截图对得上吗？
- 如果错了：生成 "这步错了，原因是...应该..."，**这一步不进训练集**
- 如果对了：解释 action 带来了什么 state change

### Generator（写手）
输入：之前所有 reflection + action history + task goal + 当前 screenshot + 这一步的 action code
输出：结构化 CoT

**关键 trick**: 在 screenshot 上画一个红色 marker 标在 mouse 要点的位置，再 crop 一个 zoomed-in patch（参考 V* [https://arxiv.org/abs/2312.14135]）——让 LLM 看清楚坐标对应的是哪个 UI 元素。这是 visual grounding 的辅助 supervision。

### Summarizer（任务提炼员）
人类 annotator 写的 task goal 经常含糊："帮我搞一下这个文档"。Summarizer 把它改写成精确版本，并给整条轨迹打三个分：alignment / efficiency / difficulty。

**成本**: $0.6/task，总项目 $32K。这个价格你应该有感觉——比人工写 CoT 便宜 100 倍。

---

## 4. 最妙的一个细节：State-Action Matching 防 leakage

这个我要单独讲，因为它体现了真正的工程功力。

naive 做法：每一步 action 配一张它发生**前**的 screenshot。直觉上对。

但 paper 指出问题：**mouse 在 click 之前已经移到目标位置了**。那张截图里 cursor 就停在 button 上——模型只要复读 cursor 坐标就能"答对"。这没有学到任何东西，泛化必死。

他们的 fix：

```
1. 对 mouse click，先 backtrack 到 mouse 开始移动的起点
2. 从那个点向前找，找到最后一个视觉上 distinct 的 frame
3. 用那张 frame 作为 s_i
```

人话翻译：**让 state $s_i$ 是"我还不知道要点哪"那一刻的截图**，模型必须从"鼠标还在乱七八糟的地方"的 state 推断出该点哪。这才能学到真正的 perception → action 映射。

公式上，他们要学的是：

$$P(a_i \mid I, s_0, a_0, s_1, a_1, \ldots, s_i)$$

其中 $I$ 是 task instruction，$\langle s_j, a_j \rangle$ 是历史 state-action pair。如果 $s_i$ 已经 leak 了 $a_i$ 的信息（cursor 在 target 上），那 $P(a_i | s_i)$ 就退化成 copy cursor 位置。

---

## 5. 训练时的 trick：三种 CoT 都喂，推理只用中间一种

这个 ablation 结果违反直觉但非常 work (Table 6)：

| 训练数据 | OSWorld SR |
|---|---|
| 只用 L2 训练 | 13.1% |
| L1+L2+L3 混合训练 | **18.5%** |

| 推理用哪种 | OSWorld SR |
|---|---|
| L1 | 16.9% |
| L2 | **18.5%** |
| L3 | 17.6% |

直觉上："推理用 L2 最好，那训练就用 L2 呗"——错。训练混合三种最好。

我的理解：这是 multi-task learning 的 regularization。
- L1 给 action 直接 supervision——模型学"输出格式"
- L3 给 perception supervision——模型学"看截图"
- L2 给 planning supervision——模型学"想"
- 三者共享 backbone，互相正则

推理时只用 L2，因为 L3 描述截图会引入**跟任务无关的元素**，反而 dilute attention。L1 又太短没思考。L2 是 sweet spot。

这个发现其实呼应你以前讲过的："context 是有限的资源"。L3 什么都描述，把 attention 摊薄了；L2 只描述 task-relevant 部分，attention 集中。

---

## 6. 数据 scaling 的意外发现：跨 OS 数据互相加成

Figure 7 的实验我看完愣了一下：

| 训练数据 | OSWorld SR (Ubuntu benchmark) |
|---|---|
| 7K Ubuntu only | 9.8% |
| 7K Ubuntu + 14K Win&Mac | **18.5%** |

加了 Windows 和 macOS 的数据，反而在 Ubuntu benchmark (OSWorld) 上几乎翻倍。

直觉上：跨 OS 数据应该 negative transfer 才对——Windows 的按钮位置、菜单结构跟 Ubuntu 不一样。但实际：**学到的不是 UI 细节，是 reasoning pattern**。"看到 dropdown → 选正确选项"这种抽象 action pattern 跨 OS 通用。

这跟你讲过的 "data diversity > data specificity" 完全一致。多样性带来 generalization，特异性带来 overfitting。

继续加数据：3K→10K Ubuntu +72%，3K→14K Win&Mac +125%。**没看到饱和**——更多数据还会继续涨。

---

## 7. Test-time compute scaling——这是最有意思的

Figure 8 和 9 我反复看了好几遍。OpenCUA-QWEN2-7B 上跑 Pass@n：

| Step budget | Pass@1 | Pass@16 |
|---|---|---|
| 15 | 16.9% | 34.6% (+104%) |
| 50 | 18.4% | 39.2% (+113%) |

**同一个模型，sample 16 次比 sample 1 次翻一倍**。

人话翻译：模型其实**会做**很多任务，但单次 sample 的 variance 太大，经常走错路径。多试几次就能撞上正确路径。

更吓人的是 Figure 9：temperature 设成 0（deterministic decoding），Pass@16 vs Pass@1 仍有 18% gap。原因只是**初始 system date 等微小差异**。这意味着 CUA policy 在 multi-step rollout 中是个混沌系统——初始条件敏感。

这就是 test-time compute scaling 的 headroom：
- **Verifier + reranking**: 让一个 verifier model 判断哪条 trajectory 对，选最好的
- **Self-consistency**: 多 sample 投票
- **RL post-training**: Pass@16 - Pass@1 = 18% 是 GRPO / RLHF 的直接 reward signal

你 2024 年讲过的 "Reward is enough" 在 CUA 上的具体形态应该是：**verifier (判断 trajectory 对错的 model) + generator (CUA policy) + multi-sample + reranking**。这块是下一代 CUA 的最大蓝海。

---

## 8. AGENTNETBENCH——离线 eval 加速迭代 100 倍

OSWorld online 评测又慢又贵：每个 task 要起 VM、装 app、跑 agent、执行 evaluation script。一个 task 5-15 分钟，369 个 task 跑一遍要好几天。

AGENTNETBENCH 是 100 个 held-out 任务，离线 eval：给定 screenshot 和 history，看模型预测的下一步 action 对不对。**每个 step 几秒钟**。

设计上的妙处：
- 每个 step 标注**多个 valid action** (不只是单一 gold action)——因为电脑操作经常有多解
- 三类 action 分别评：
  - Coord (click/drag/scroll): 用 bounding box 判断坐标是否落进去
  - Content (write/press/hotkey): 用 edit distance / exact match
  - Function (terminate): 判断是不是在正确 step 终止

Figure 13 验证：离线 SR 与 online SR 是 power-law 相关。所以离线 benchmark 是 online performance 的可靠 proxy。开发迭代 cycle 从 days/task 降到 seconds/task。

**Reference**: https://xlang.ai/blog/osworld-verified

---

## 9. Failure mode——这是 next paper 的 roadmap

Appendix E 把失败归类，我觉得每条都是一篇 future paper：

1. **缺领域知识**: 不知道 VLOOKUP、不知道 bulk-fill——需要 RAG 或 domain training
2. **高精度 grounding 失败**: "把 H2O 的 2 改成 subscript"，选错了字符——需要字符级 grounding data
3. **Action 死循环**: 错误 action 没有可见效果就无限重复——reflection 不够强
4. **Termination 误判**: 任务完成了不终止 / 没完成提前说完成——termination 信号需要专门训练
5. **Long-horizon 失败**: >30 步任务 context 维持不住——需要 memory architecture
6. **错误感知不足**: 视觉上差 1 个像素判断成对，不会 undo+retry——perception + recovery 联合训练

每一类都对应一个明确的技术方向。你写下一版 CUA 的 paper 时，这就是 agenda。

---

## 10. 几个我自己的 take

### 10.1 CoT 是 in-context learnable algorithm

OpenCUA 验证你长期的观点：reflective CoT 直接注入训练数据，模型 SFT 后展现 error-correction 能力。这不是 emergent，是 supervised。跟 DeepSeek-R1 / o1 路线一致，只是从 math reasoning 搬到了 agent action prediction。

### 10.2 General data 帮 specialized task

35% general text SFT 数据**提升** agent performance (Figure 11)。反直觉但本质是：foundation model 的归纳偏置——general reasoning / instruction following 是 specialized skill 的 substrate。LLaMA-3 在 general data 上 SFT 后做 tool-use 反而比专门 tool-use 数据训出来的好，是同一个现象。

### 10.3 Grounding 是 perception layer, Planning 是 reasoning layer

Qwen2.5-VL-32B grounding 接近 OpenCUA-7B，但完整 agent task 上远落后 (Table 3)。说明：
- Grounding（看截图找坐标）已经够用
- Planning + Reflection（想清楚该干啥）才是 bottleneck

对 next-gen CUA architecture 的 implication：encoder/grounding 该收一收了，policy network 的 reflection quality 才是限制因素。

### 10.4 真正的 next frontier 是 RL post-training

Pass@16 vs Pass@1 的 18% gap 就是 reward signal。用 GRPO / PPO 在 OSWorld 上跑 RL，把 Pass@1 拉到 Pass@16 水平——这是 50% → 70% 的跃迁路径。需要解决的是 verifier（怎么判断 trajectory 对错，OSWorld 有 evaluation script 但稀疏），可能需要 process reward model。

---

## 11. 总结成一句话

OpenCUA 把"怎么从零搭一个 SOTA computer-use agent"的整个 stack 开源了——从 annotation tool、22.6K trajectory dataset、reflective CoT synthesis pipeline、三种 training recipe、到 7B/32B/72B 模型 + 离线 benchmark。

核心 insight 三条：
1. **Reflective long CoT** 是把模仿学习升级成推理学习的关键 augmentation
2. **Training mixture (L1+L2+L3) + Inference L2** 是最佳配置
3. **Pass@16 vs Pass@1 的 gap** 是下一代 RL post-training 的 reward signal

**References**:
- https://opencua.xlang.ai
- https://os-world.github.io
- https://xlang.ai/blog/osworld-verified
- https://arxiv.org/abs/2412.04454 (Aguvis)
- https://arxiv.org/abs/2501.12326 (UI-TARS)
- https://arxiv.org/abs/2502.13923 (Qwen2.5-VL)
- https://arxiv.org/abs/2312.14135 (V* visual grounding)
- https://www.anthropic.com/news/3-5-models-and-computer-use
- https://openai.com/research/operator

要我说，这篇 paper 真正的"教学价值"不在 45% 这个数字——在于它把"我们看到 4% 时该怎么 debug、怎么 ablation、怎么 design CoT、怎么 mix data"的整个工程思考过程摊在桌面上。这对社区是金矿。

---

# OpenCUA: 深度技术解析

Andrej，这篇 paper 我觉得是过去一年 CUA (Computer-Use Agent) 领域最重要的开源工作之一。让我从你的角度——把 neural network 看作 data → function approximator → behavior 的pipeline——来剖析它为什么 work，以及它的 design choices 在哪里体现了真正扎实的 engineering insight。

**Project Page**: https://opencua.xlang.ai  
**OSWorld Benchmark**: https://os-world.github.io  
**OSWorld-Verified blog**: https://xlang.ai/blog/osworld-verified  

---

## 1. 宏观图景：为什么直接 SFT state-action pairs 不 work

paper 开篇就抛出一个非常重要的负结果：**在 22.6K 条人类演示轨迹上直接 SFT VLM，OSWorld 成功率只有 4.4%** (Figure 1 Left "Base Recipe")。这个数字对你的直觉应该是一个验证——光是模仿人类行为表面 (action sequence) 不足以学到 generalizable policy，因为 agent 要学的不只是 "在 state $s_i$ 下输出 action $a_i$"，而要学一个**条件分布**：

$$P(a_i \mid I, s_0, a_0, s_1, a_1, \ldots, s_i)$$

其中 $I$ 是 task instruction，$s_i$ 是 screenshot observation，$a_i$ 是 pyautogui-style action。state-action pairs 只提供了 $P(a_i|s_i)$ 的 marginal，丢掉了**轨迹上下文、错误恢复、planning** 这些真正的 agentic capabilities。

OpenCUA 的核心 thesis 就是：**reflective long Chain-of-Thought (CoT) 是把模仿学习升级成推理学习的关键 augmentation**。这跟你在 nanoGPT / education 系列里讲的 "CoT 是 in-context learnable algorithm" 的观点高度一致——他们本质上是把 CoT 显式作为监督信号注入到训练数据里。

---

## 2. AGENTNET TOOL：人类演示的低摩擦捕获

### 2.1 设计哲学

这个 tool 跑在 annotator 的**个人电脑**上，后台 record 三类信号：

1. **Screen video** (via OBS Studio [28])
2. **Mouse/keyboard signals** (via DuckTrack [38] + OpenAdapt [29])
3. **Accessibility tree (Axtree)** (via OSWorld framework [46])

关键 design choice: **不要求 all-correct trajectory**。以前的 GUI dataset (e.g., AndroidControl [23]) 强求 "gold trajectory"，但 OpenCUA 团队的 insight 是：annotation errors 本身是宝贵的 supervision signal——只要 reflector 能识别它们，就能教会模型 detect-and-recover。这与 RLHF / DPO 中 "negative examples 也是 signal" 的思想相通。

### 2.2 Action Reduction 算法

原始信号频率极高 (typical task 产生数千个 low-level events)，直接喂给 VLM 训练是低效的。他们用 rule-based 方法压缩：

| Atomic Event | Compressed Action |
|---|---|
| mouse_move events | 仅保留 click/drag 的 start & end position (move 作为 precondition) |
| consecutive scrolls | 合并为单向 action，累积 wheel counts |
| consecutive key presses | 合并为 `write(text)` string |
| modifier combinations | 抽象为 `hotkey(key1, key2)` |
| multi-step gestures | 合并为 `dragTo` / `doubleClick` / `tripleClick` |

输出对齐到 Table 1 的 12 个 pyautogui action space: `click(x,y,button)`, `middleClick(x,y)`, `doubleClick(x,y,button)`, `tripleClick(x,y,button)`, `moveTo(x,y)`, `dragTo(x,y)`, `scroll(dx,dy)`, `hscroll(dx,dy)`, `write(text)`, `press(key)`, `hotkey(key1,key2)`, `wait()`, `terminate('success'/'failure')`.

### 2.3 State-Action Matching：防止 future leakage

这里有个**特别精妙**的细节，值得专门讲。naive 做法：对每个 action $a_i$，取 action 发生**前**的 frame $s_i$。但 paper 指出这会 leak future info：

> "the mouse may already be positioned over a button, making the prediction trivial."

也就是说，如果你 click 之前 mouse 已经移到 button 上，那张 screenshot 里 cursor 已经在 button 上——模型只需要从 cursor 位置 copy 坐标，预测 trivial 化，泛化失败。

他们的 solution: **对 mouse click 类操作，backtrack 到 mouse pre-movement phase 的起点，然后向后搜索最后一个视觉上 distinct 的 frame** 作为 $s_i$。这本质上是在做 "未来信息屏蔽"——让 state $s_i$ 与 action $a_i$ 的因果关系保持纯粹。

直觉上，这是在做"反 cheating"：模型必须从「不知道我接下来要点哪」的 state 推断 action，而不是从「cursor 已经在目标上」的 state 复读 cursor 坐标。

### 2.4 数据集统计

| 维度 | 数值 |
|---|---|
| Total tasks | 22,625 |
| Windows / macOS / Ubuntu | 12K / 5K / 5K |
| Avg steps/task | 18.6 |
| Applications | 140+ |
| Websites | 190+ |
| Resolution range | 720p → 4K |
| Annotation cost | ~$20K (6 months) |
| CoT synthesis cost | ~$0.6/task |
| Total project cost | ~$32K |
| Annotators | 634 |

跟 prior GUI datasets 对比 (Table 2)：AGENTNET 是**第一个 desktop trajectory-level dataset**，覆盖 3 OS、video、Axtree、inner monologue 全模态。Android 系 (AitW, AitZ, AMEX, GUI Odyssey) 都是 mobile；Web 系 (Mind2Web, AgentTrek) 只有 DOM 没有 video。

---

## 3. Reflective Long CoT Synthesis：核心技术贡献

这是 paper 的真正核心。把 raw demonstration (screenshot sequence + reduced actions) 升级成**带反思推理**的训练数据。

### 3.1 三层 CoT 层次结构

借鉴 Aguvis [52]，但显著深化：

```
L3 (Observation + Thought + Action)
  └─ 上下文观察，捕获 salient 视觉/文本元素
L2 (Thought + Action)  
  └─ reflective reasoning: 分析 state transition, 回忆之前步骤, 纠正错误, 规划下一步
L1 (Action)
  └─ 简洁的可执行 action code
```

inference flow: **L3 → L2 → L1**，mirrors perceptual-to-agentic decision flow (感知 → 反思 → 行动)。

### 3.2 Synthesis Pipeline (Figure 5)

三个 LLM 角色，都基于 `claude-3-7-sonnet-20250219`：

**Reflector**: 检查每个 step 的 correctness 和 redundancy
- 比较前后 screenshot
- 检查 action code 本身的正确性
- 检查 generated CoT 是否 align screenshot 和 code
- 如果 step 错误/redundant: 生成 reflection reason + 训练时 ignore 此 step
- 如果 step 正确: 解释 action 带来的 before/after state 差异

**Generator**: 条件于完整 context 生成 structured CoT
- 输入: previous reflections + action history + task goal + screenshots + action code
- **视觉 cues**: 红色 marker 标在 mouse action coordinate 上 + zoomed-in image patch (inspired by V* [43], 这是值得记忆的细节——直接用 visual grounding 帮 LLM 定位坐标)

**Summarizer**: 精炼 user-written goals → 精确任务目标
- 对每条 trajectory 评分: alignment, efficiency, difficulty

这套 pipeline 的关键 insight 是把 **"错误恢复" 显式作为训练 signal**——错误 step 不会被丢弃，而是生成 reflection 让模型学会 "我刚才点错了，应该...". 在 Appendix F 的 case study 里有个非常生动的例子 (Step 8 的 Chrome extension 安装)：agent 走错路径 (sign in page)，意识到不需要 sign in，回退，重新走 "Load unpacked" 路径成功。

### 3.3 为什么这个 CoT 设计 work

paper Section 5 的 ablation 给出量化证据：

| CoT 设计 | OSWorld SR (%) |
|---|---|
| Short-CoT (Aguvis baseline) | 11.5 |
| Advanced-CoT (OpenCUA reflective) | **15.3** |
| L2 only training | 13.1 |
| Mixture (L1+L2+L3) training | **18.5** |
| L1 inference | 16.9 |
| L2 inference | **18.5** |
| L3 inference | 17.6 |

几个非平凡的发现：

1. **训练时 mixture > L2 only** (18.5 vs 13.1)。即使 L2 在 inference 最好，训练时混合所有 level 反而更好。这违反直觉——但可以理解为 multi-task learning 中的 regularization: L1 给 action 直接 supervision, L3 给 perception supervision, L2 给 planning supervision。三者互补，单独训 L2 会丢失 grounding 和 perception signal。

2. **Inference L2 > L3** (18.5 vs 17.6)。L3 包含完整 Observation 但反而比 L2 差。paper 解释: L3 描述 screenshot 时会引入**与任务无关的元素**，误导模型。这是个很有意思的发现——more information ≠ better reasoning，冗余描述会 dilute attention。这跟你在 GPT 训练讲座中提到的 "context is a limited resource" 一致。

3. **Reflective CoT (含 error correction) 的收益 (11.5→15.3) 主要来自 self-correction 能力**。这个增益在 long-horizon task 上尤其关键——Appendix E error study 显示，没有 reflection 的 agent 在错误 action 上会**无限循环** (Action repetition 类失败)。

---

## 4. Context Encoding：history 的 token 效率 trade-off

### 4.1 Textual History

OpenCUA 用 **L1 CoT (Action only)** 作为 textual history，不是 L2。理由 (Figure 10):

- L2 history: 60 steps 累积下来 token 太多，必须截断
- L2 history 引入 hallucinations 反而分散注意力
- L1 紧凑，留更多 context window 给 visual history

但 inner monologue 中**保留 memory components** 来补偿 L1 信息缺失——即模型显式在 Thought 里回忆 "之前我做了X"。

### 4.2 Visual History

Figure 10 的 ablation:

| # Screenshots | OSWorld SR |
|---|---|
| 1 | 最低 |
| **3** | 最佳 trade-off |
| 5 | 略升 +3K tokens, 收敛变慢 |

直觉：GUI agent 完全靠 vision 观察状态变化，单 image 信息不足，但 5 张以上边际收益递减，且训练效率下降。

---

## 5. Training Recipes：三种策略对应不同 compute budget

### 5.1 Stage-2 only (轻量级 adaptation)

- 配置: 70% CUA (planning:grounding = 4:1) + 30% general SFT
- Qwen2-VL-7B: 30B tokens, 96×A100, 45h
- Kimi-VL-A3B: 20B tokens, 10h
- 数据: 18k Win&macOS + 10k Ubuntu trajectories

### 5.2 Stage 1 + Stage 2 (中等预算，性能最优)

**Stage 1** (grounding + understanding):
- 35B tokens (32B) 或 250B tokens (72B)
- 数据: grounding trajectories + tutorial demos + state-transition caption + general VL + general text
- LR $3\times10^{-5}$, batch 3584, 224×A100 (32B)
- LR $2.5\times10^{-5}$ decay to $1.5\times10^{-5}$, batch 600, 480×A100 (72B)

**Stage 2** (planning):
- 60B tokens (32B) 或 16B tokens (72B)
- 数据: 45% planning + 20% grounding + 35% general
- 18k Win&macOS + 20k Ubuntu (+ 8k rollout trajectories for 72B)
- **72B 的关键 trick**: 用 o3 [30] + Jedi [48] 在 Ubuntu 环境 rollout 8k 轨迹，然后**只把 reflection 部分放进 Stage 2**，让模型先在 Stage 1 学到 rich behavior，再在 Stage 2 学到 efficient, information-dense CoT format

### 5.3 Joint Training (大规模 general-purpose VLM)

- 200B tokens, 128×A100, 8 days
- 20% planning + 20% grounding + 60% general
- multi-image trajectory 数据训练 3 epochs (vs 其他 1 epoch)
- 得到 OpenCUA-7B (27.3% OSWorld)

### 5.4 为什么混入 general text data

Figure 11 ablation 证明: **35% general text data 反而提升 agent performance**。直觉上这违反 "domain-specific 数据更有效" 的常识，但 paper 解释: general text 提升 instruction understanding 和 generalization。这跟 Chinchilla / Llama 系列中"general pretraining 是 foundation"的观点一致——CUA 模型本质上还是一个 VLM，agent 能力是在 general intelligence 之上 fine-tune 出来的 specialized layer。

---

## 6. 实验结果：开源 SOTA

### 6.1 OSWorld-Verified (Table 3)

| Model | 15 steps | 50 steps | 100 steps |
|---|---|---|---|
| Claude Sonnet 4.5 | - | - | **61.4** |
| Claude 4 Sonnet | 31.2 | 43.9 | 41.5 |
| OpenAI CUA (Operator) | 26.0 | 31.3 | 31.4 |
| Seed1.5-VL | 27.9 | - | 34.1 |
| Qwen3-VL | - | - | 38.1 |
| UI-TARS-72B-DPO | 24.0 | 25.8 | 27.1 |
| Qwen2.5-VL-72B (base) | 4.4 | - | 5.0 |
| **OpenCUA-7B** | 24.3±1.93 | 28.1±0.7 | 26.6±0.6 |
| **OpenCUA-32B** | 29.7±1.5 | 34.1±1.0 | 34.8±0.9 |
| **OpenCUA-72B** | 39.0 | 44.9 | **45.0±1.1** |

关键观察：

1. **OpenCUA-72B 在 100-step budget 上 45.0%，开源 SOTA**，逼近 Claude 4 Sonnet (43.9%)，但还远落后于 Claude Sonnet 4.5 (61.4%)。
2. **15→50 steps 增益明显**，50→100 边际递减。原因有二：(i) 大多数 task 需要的步数在 15-50 之间；(ii) **当前模型还很不擅长 detect 自己的错误并 recover**，hallucination 和 repetitive loop 会浪费 extra steps。这点对你应该特别有意义——这正好说明 reflection quality 是下一个 bottleneck。
3. **Pass@3 显著提升**: OpenCUA-72B 从 Pass@1 45.0% → Pass@3 53.2% (+8.2%)。OpenCUA-32B 从 34.2% → 45.6% (+11.4%)。这说明**模型本身有 capability，只是 sample 一次命不中**。这正是 RL / reranking / multi-agent 的 headroom——未来 post-training 在这块能挖出大量增益。

### 6.2 GUI Grounding (Table 5)

| Model | OSWorld-G | ScreenSpot-Pro | ScreenSpot-V2 | UI-Vision |
|---|---|---|---|---|
| UI-TARS-72B | 57.1 | 38.1 | 90.3 | 25.5 |
| OpenCUA-32B | **59.6** | 55.3 | **93.4** | 33.3 |
| OpenCUA-72B | 59.2 | **60.8** | 92.9 | **37.3** |

OpenCUA-72B 在 ScreenSpot-Pro (60.8%) 和 UI-Vision (37.3%) 都是 SOTA。

**重要 insight**: Qwen2.5-VL-32B 在 OSWorld-G 和 ScreenSpot-V2 上接近 OpenCUA-Qwen2-7B，但**完整 OSWorld 任务上 OpenCUA 远超** (19.9% vs 23.0% on 100 steps)。这说明：**grounding 是必要条件，不是充分条件**。high-level planning + reflective reasoning 才是 reliable task completion 的真正 driver。

### 6.3 AGENTNETBENCH (Table 4) - 离线 benchmark

100 个 held-out tasks，多 gold action per step。Coord/Content/Function 三类 action 分别评估：

| Model | Coord SR | Content SR | Func SR | Avg SR |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 50.7 | 40.8 | 3.1 | 48.0 |
| OpenAI CUA | 71.7 | 57.3 | 80.0 | 73.1 |
| OpenCUA-7B | 79.0 | 62.0 | 44.3 | 75.2 |
| OpenCUA-32B | **81.9** | **66.1** | 55.7 | **79.1** |

离线 benchmark 与 online leaderboard 排序一致 (Figure 13)，且评估速度快很多 (无 environment setup)。这对研究迭代价值巨大——offline proxy 把开发 cycle 从 hours/task 降到 seconds/task。

---

## 7. Data Scaling Laws (Figure 7)

paper 验证了 data scaling 在 CUA 上成立：

| Data Setting | OSWorld SR |
|---|---|
| 7K Ubuntu only | 9.8% |
| 7K Ubuntu + 14K Win&Mac | 18.5% |
| 10K Ubuntu + 17K Win&Mac | 更高 |

**Out-of-domain data 没有 negative transfer**——加入 Win&Mac (不同 OS) 反而大幅提升 Ubuntu-focused OSWorld 性能 (9.8% → 18.5%)。这跟你的 "data diversity > data specificity" 的直觉一致：cross-domain 学到的是 generalizable reasoning pattern，不是 OS-specific UI conventions。

进一步 scaling 实验 (3K→10K Ubuntu, 3K→14K Win&Mac): 性能持续增长，Ubuntu +72%, Win&Mac +125%。**没有出现 saturation**——说明更多数据还能继续提升。

---

## 8. Test-Time Compute Scaling (Figure 8, 9)

这块对你应该最有共鸣——强化 scaling laws 在 inference time 的体现。

**Pass@n evaluation** (OpenCUA-QWEN2-7B, temp=0.1, 16 runs):

| Step Budget | Pass@1 | Pass@4 | Pass@8 | Pass@16 |
|---|---|---|---|---|
| 15 | 16.9% | ~26% | ~31% | 34.6% (+104%) |
| 50 | 18.4% | ~27% | ~34% | 39.2% (+113%) |

**Pass@16 比 Pass@1 翻倍**。这说明：

1. **模型 capability 上限远高于 Pass@1 表现**。Variance 的来源不是模型不会做，而是 sample 不稳定。
2. **更大 step budget × 更大 n 的增益最大**。50-step + Pass@16 = 39.2% >> 15-step Pass@1 16.9%。
3. variance 来源 (Section 5.1):
   - 同一 task agent 会选不同 solution (e.g., Ctrl+Shift+T vs menu history)
   - minor omission (忘记点 Save) / stray extra click 转化 success → failure
   - environment dynamics (CAPTCHA, 网络延迟, 机器差异)

Figure 9 的 temp=0 实验**更加令人不安**：即使 deterministic decoding，Pass@16 vs Pass@1 仍有 18%+ gap。原因只是**初始 system date 等微小差异**。这说明 CUA policy 在 multi-step rollouts 中对初始条件极度敏感——本质上是个 chaotic system。

**这是 next-breakthrough 的方向**: 让 single sample 命中率接近 Pass@16 上限，要么靠 self-consistency voting, 要么靠 better reflection 减少 trajectory divergence。

---

## 9. Failure Mode 分析 (Appendix E)

paper 把失败归为 6 类，值得逐个理解：

1. **Insufficient task knowledge**: 缺领域知识 (VLOOKUP, bulk-fill)。→ 需要 RAG / tool use / domain training
2. **High-precision grounding errors**: "H2O 改 2 为 subscript" 选错字符。→ 需要字符级 grounding data
3. **Action repetition**: 错误 action 无可见效果时无限循环。→ reflection 不够强
4. **Termination misjudgment**: 任务已完成不终止 / 未完成提前 terminate。→ termination signal 需要专门训练
5. **Long-horizon failures**: >30 步任务难以维持 coherent context。→ memory mechanism 不足
6. **Insufficient error perception**: 视觉上 1 字符偏差判断为正确，不会 undo+retry。→ perception + recovery 联合训练

这 6 类失败对 future work 是很好的 roadmap——每个都对应一个明确的技术方向。

---

## 10. 跟你直觉的连接点

Andrej，几个我觉得你会特别关心的点：

### 10.1 CoT 作为 in-context algorithm

OpenCUA 验证了你一直强调的 "CoT 是 in-context learnable algorithm" 假设。**reflective CoT 直接注入到 training data**，模型在 SFT 后展现 error-correction 能力——这不是 emergent，是显式 supervised。这跟 DeepSeek-R1 / o1 路线一致，但用在 agent action prediction 而非 math reasoning。

### 10.2 Data quality > Data quantity (但 quantity 也重要)

reflective long CoT 把 11.5% 拉到 15.3% (单点 +33%)，比单纯增加 data 规模更高效。但 Figure 7 显示 raw scale 同样 work。两者乘积：**高质量 CoT × 大规模 × diverse domains** 是 scaling recipe 的真正公式。

### 10.3 Test-time compute 是 next frontier

Pass@16 >> Pass@1 的 gap 说明我们离 model capability 上限还远。RL / self-play / verification-based reranking 在这块的潜力巨大。你的 "Reward is enough" 思路在 CUA 上的具体形态应该是: ** verifier (一个能判断 trajectory 是否成功的 model) + generator (CUA policy) + 多 sample + reranking**。

### 10.4 General data 帮 Specialized task

35% general text SFT 提升 agent performance。这反直觉但本质上是 foundation model 的归纳偏置：general reasoning/instruction following 是 specialized skill 的 substrate。这跟 LLaMA-3 在 general data 上 SFT 出来做 tool-use 反而比专门 tool-use data 训出来的好的现象一致。

### 10.5 Visual grounding vs Agentic reasoning

Qwen2.5-VL-32B grounding 接近 OpenCUA-7B，但完整 agent task 上远落后。这证明：**grounding 是 perception layer, planning + reflection 是 reasoning layer**。两层都需要，但 reasoning layer 是真正的 bottleneck。这对你设计 next-gen agent architecture 有 implication：encoder/grounding 已经够用，policy network 的 reflection quality 才是限制因素。

---

## 11. Limitations 和未来方向

paper Section A 的 limitations:
- 数据规模受限于人类标注 (扩展需要更多人力)
- annotator 不一定用最有效方法 (shortcuts, scripts) — 缺 expert-level demos
- 知情同意引入 selection bias (排除 privacy-sensitive 用户)

我的补充思考：
- **Synthetic data 路线**: 用 stronger agent (o3 + Jedi 已在 72B 上用) rollout 大量合成 trajectory，再用 reflector 过滤——这是 self-play 的雏形
- **RL post-training**: Pass@16 与 Pass@1 的 18% gap 是 RLHF / GRPO 的直接 reward signal
- **Hierarchical planning**: Long-horizon failures 说明需要 hierarchical decomposition (sub-goal planning)
- **Memory architecture**: 跨 session / cross-task memory 是 long-horizon 的真正解
- **Environment robustness**: temp=0 下仍 18% variance，说明 policy 对环境扰动极敏感——这指向 environment-awareness training

---

## 12. 论文工程价值

OpenCUA 的真正贡献不是某一个 SOTA 数字，而是**把整个 CUA stack 开源**：

- Annotation tool (AgentNet Tool): 让任何团队都能 collect 自己的 data
- Dataset (22.6K trajectories): 让任何团队都能 train baseline
- Processing pipeline: 把 raw demos → reflective CoT data 的 recipe 公开
- Training recipes: 三种策略对应不同预算
- Models (7B / 32B / 72B): 让任何团队都能 build on top
- AGENTNETBENCH: 离线快速评估，加速迭代

**Reference Links**:
- Project: https://opencua.xlang.ai
- OSWorld: https://os-world.github.io
- OSWorld-Verified: https://xlang.ai/blog/osworld-verified
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Aguvis: https://arxiv.org/abs/2412.04454
- UI-TARS: https://arxiv.org/abs/2501.12326
- V* (visual grounding inspiration): https://arxiv.org/abs/2312.14135
- DuckTrack: https://github.com/TheDuckAI/DuckTrack
- OpenAdapt: https://github.com/OpenAdaptAI/OpenAdapt
- Claude Computer Use: https://www.anthropic.com/news/3-5-models-and-computer-use
- OpenAI Operator: https://openai.com/research/operator

---

## 一句话总结

OpenCUA 把 "如何 build 一个 SOTA-level computer-use agent" 的全部 know-how——从 data collection tool 到 reflective CoT synthesis pipeline 到 multi-stage training recipe 到 offline eval benchmark——做成了 open-source cookbook。它的核心 insight 是：**reflective long CoT 是把 state-action pairs 从模仿学习升级成推理学习的关键 augmentation**，而 training-time mixture (L1+L2+L3) + inference-time L2 是最佳配置。剩下的 headroom 在 test-time compute (Pass@16 vs Pass@1) 和 RL post-training，这正是下一代 CUA 的研究方向。
