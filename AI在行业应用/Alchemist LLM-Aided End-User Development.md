---
source_pdf: Alchemist LLM-Aided End-User Development.pdf
paper_sha256: 953c29d541ed516d059376a6133d2a2fdc51ac916ce6b1cabc7d6a006bb919c0
processed_at: '2026-08-18T00:34:51-07:00'
target_folder: AI在行业应用
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Alchemist — 用人话说

Andrej, 好，我换个频道，咱们就像在咖啡馆聊这篇 paper 一样。

## 一句话版

**让生物化学家用大白话指挥机械臂配试剂**，不用一行 ROS code，全靠 LLM 当翻译。

---

## 1. 这群人想解决什么痛点

想象一下，Yale 的一个 chemistry PhD 学生，每天要做的事情是配 LB media（培养细菌的那种培养基），就是把各种 reagent 按比例倒进 beaker 混合。重复、低级、累人。

理想情况是机器人帮她做。但问题是：

- robotics PhD 可以写 ROS node 控制 UR5，可是他不懂 chemistry，不知道试剂顺序、操作流程
- chemistry PhD 懂流程，可她不懂 Python、不懂 ROS、不懂 motion planning

**两边人 knowledge domain 完全错位**。Alchemist 的工作就是：让 chemistry 那个人开口说人话，LLM 把人话翻译成 ROS code，然后机器人执行。

这其实就是你 Software 3.0 talk 的一个具体落地 case——自然语言当 programming language，LLM 当"编译器"。

---

## 2. System 长啥样

![Alchemist UI](https://doi.org/10.1145/3610977.3634969)

界面就三个主要 panel：

```
+-------------+-------------+-------------+
|   RViz 3D   |   Chat      |  Terminal   |
| visualization| with LLM   |  Python     |
|             | (+ voice)   |  REPL       |
+-------------+-------------+-------------+
      [Editor + File Tree, 默认藏起来]
```

几个有意思的设计 choice：

**Editor 默认藏起来**。这点其实挺反直觉——一个编程 IDE 把 editor 藏起来？作者解释：novice 用户一看到代码就紧张，能藏就藏。我觉得这是个很 deep 的 design insight——**对 end-user programming system 来说，"不显示 code" 本身就是一个 feature**。

**Chat 支持 voice**。用 OpenAI Whisper 做 ASR。可以想象 chemistry 博士手上戴着 gloves 拿 pipette 的场景，voice input 几乎是 must-have。

**RViz 是 preview 不是 execution**。Robot 动作不可撤销，撞翻一烧瓶 reagent 就废了。所以必须先在 RViz 里预览一下 LLM 生成的 code 会怎么动，再决定要不要真跑。

---

## 3. 后端三个关键 trick

### Trick 1: 两层 function library

这个我觉得是 system 设计最妙的地方。Function library 给 LLM 当 API 用，分两层：

**High-level**: `pour("beaker_500ml")` — 一行搞定"把当前手里的东西倒进 500ml 烧杯"

**Low-level**: `move(0.3, 0.2, 0.4, 0, 1.57, 0)` — 控制 end-effector 到 SE(3) pose

注意 **library 本身只有一套**，让 LLM 输出哪一层 granularity 的 code 是靠 **initial prompt 控制**。如果 user 是新手、task 是直接的，prompt 让 LLM 用 high-level function；如果 user 是专家、要精细控制，prompt 偏 low-level。

可以用条件分布来直觉理解：

$$P_{\theta}(\text{code} \mid U, \mathcal{L}_{hi}, \mathcal{L}_{lo}, \text{Ex}_{hi}) \approx \text{使用 high-level functions}$$

$$P_{\theta}(\text{code} \mid U, \mathcal{L}_{hi}, \mathcal{L}_{lo}, \text{Ex}_{lo}) \approx \text{使用 low-level functions}$$

其中：
- $U$: user prompt
- $\mathcal{L}_{hi}, \mathcal{L}_{lo}$: 两层 library 描述
- $\text{Ex}_{hi}, \text{Ex}_{lo}$: few-shot example 偏向哪一层

这个 design pattern 其实就是 **prompt-driven abstraction selection**，比维护两套 system 简洁多了。

### Trick 2: Grounded Prompting

这是 paper 里最 engineering 也最 useful 的部分。

观察：LLM 写 robotics code 时会反复犯一些错，比如：
- 用户说"pour"时，LLM 喜欢自己编一些 "move above the beaker" 的多余动作
- 用户说"add X"时，LLM 喜欢给一些想象的坐标，不用 AR marker
- 用户说"写个 generic function"时，LLM 经常只定义不调用，留下死代码

作者写了一个 keyword-triggered 的 prompt 改写器：

$$g(u) = \text{prefix} \oplus u \oplus \text{suffix} \oplus \bigoplus_{i} \mathbb{1}[w_i \in u] \cdot r_i$$

变量解释：
- $u$: 原 user prompt 字符串
- $g(u)$: 改写后的 prompt
- $\oplus$: 字符串拼接
- $\text{prefix}$: `"By using the function library you are provided,"` （强制 library 使用）
- $\text{suffix}$: `"make sure to move back to home after the task is finished."` （强制复位）
- $w_i$: trigger keyword，比如 `"pour"`、`"add"`、`"function"`
- $\mathbb{1}[\cdot]$: indicator function，keyword 在 $u$ 里时为 1
- $r_i$: 对应追加的规则

举例：

| User 说 | 自动追加 |
|---|---|
| "pour the 100ml into 500ml" | "Don't move above the beaker before pouring; just call the pour function. Also, after pouring, make sure you place the object back to where it was..." |
| "add new beaker" | "Make sure to use marker location." |
| "write a generic function" | "If you wrote a function, remember to add an example function call at the end." |

直觉上，这就是一个 **prompt-level linting / fix-up pass**。非常像编译器把 source code rewrite 成更规范的形式，只不过这里 rewrite 的不是 code 而是 prompt。

我个人觉得这个 trick 可以 generalize 得很远——任何 LLM-based code gen 都可以加一层"domain-specific prompt fix-up"。你在 OpenAI 内部应该看过类似 internal tools。

### Trick 3: Code Verification

LLM 输出的 code 常常少 `import rospy`、忘记 init ROS node 之类 boilerplate。系统加了一层 deterministic post-processing 自动补全。

这跟 grounded prompting 是 **defense in depth**：
- Grounded prompting 在 input 端 bias LLM
- Code verification 在 output 端 sanitize LLM 产物

两层一起把大部分 trivial error 干掉。剩下的 hard errors（motion planning 失败、perception 失败）留给用户在 RViz / chat 里迭代。

---

## 4. 实验结果最有意思的点

10 个 participant：
- 5 个 chemistry/biology PhD（novice）
- 5 个 robotics PhD（expert）

做同一个 toy task：把不同 cylinder 里的 beads 倒进 beaker 配混合。

**总完成时间几乎一样**（novice 1:03:02 vs expert 1:01:59）！但分解后差异巨大：

| | Programming | Debugging | Idle | Editor Use | 用 general func? | SUS |
|---|---|---|---|---|---|---|
| Novice | 23:09 | 13:40 | 26:13 | 0/5 | 2/5 | 56.0 |
| Expert | 16:41 | 6:38 | 43:27 | 4/5 | 4/5 | 68.5 |

几个让人 aha 的观察：

**Novice 几乎不打开 editor**。他们宁愿在 chat 里反复 prompt LLM 改，也不愿直接改代码。原话 (N2)："It would be difficult for me to troubleshoot by myself, as I lack the confidence to examine the code." 这印证了 Liu et al. CHI 2023 ["What It Wants Me To Say"](https://doi.org/10.1145/3610591) 的发现——end-user 和 code-gen LLM 之间存在 abstraction gap。

**Expert 反而 idle 时间更多**（43 vs 26 分钟）。他们在思考怎么设计 general function 结构、怎么组合 task。Novice 几乎不写 general function，喜欢 step-by-step prompting："pour the water from the 250ml graduated cylinder to the 500ml beaker. Put the 250ml graduated cylinder back to its original place gently." Expert 一上来就："Can you write a generic function called 'pick_and_pour', which allows me to insert the input..." —— 直接 abstraction。

**两组 errors 数量一样**（1.8 平均），但 type 不同。Novice 多是 factual / syntax；expert 多是 name confusion（哪个 beaker 是哪个）。

**N4 是 outlier**（SUS 12.5，几乎满分倒着看），因为他遇到好几次 motion planning 失败撞倒东西。但即便如此他仍然说："I would love to have a liquid handling system in our lab where I could simply press a button and say 'go' without worrying about it failing, but I don't think it's at that stage yet." —— 用户对 system 的耐心超出我预期。

---

## 5. 几个 lessons learned 用人话讲

### Lesson 1: LLM 写代码不靠谱，需要多层 guard

Grounded prompting + code verification 两层叠起来把大部分 trivial bug 干掉了。但 paper 也坦白说 "errors have not been completely eliminated"。

我直觉未来方向是把 **formal verification** 直接塞进 LLM 生成循环——比如用 [KeYmaera X](https://github.com/LS-Lab/KeYmaera-X) 做 hybrid systems verification，保证 robot 不会进入 unsafe state。最近 [Lean + LLM](https://arxiv.org/abs/2306.09111) 那条线也很 promising。

### Lesson 2: User 写 prompt 也烂

引用 [Zamfirescu-Pereira et al. CHI 2023 "Why Johnny Can't Prompt"](https://doi.org/10.1145/3613904.3642739)。Non-AI expert 写的 prompt 常常 vague、implicit、含糊。Alchemist 的两个 mitigation：
- 给 user manual + tutorial video 做 guided training
- Grounded prompting 在背后自动 fix-up

但 paper 自己承认 grounding rules 是 domain-specific，换 domain 要重写。我想的 alternative：
- LLM 自己 critique 自己的 prompt（[Self-Refine](https://arxiv.org/abs/2303.17651) 思路）
- 把历史 failure mode 做 RAG retrieval 当 context
- 用 [Constitutional AI](https://arxiv.org/abs/2212.08073) 风格的 self-critique

### Lesson 3: End-user 怕看代码

Novice 用户宁可在 chat 里绕远路也不开 editor。这其实暗示了一个 design direction：**对话界面应该让 user 用自然语言"调用" general function，而不是去 terminal 打 `pick_and_pour('beaker1', 'beaker2')`**。

更好的做法是把 program structure 可视化成 behavior tree 或 flow chart，novice 看到的是"程序结构"而非"代码文本"。Code 是给 LLM 看的，不是给人看的。

---

## 6. 我觉得这 paper 的 deep insight

Alchemist 单看每个 component 都不新：
- Chat + LLM 做 code gen：GitHub Copilot 早就有了
- RViz preview：ROS 标配
- Function library 两层 abstraction：API design 常识
- Grounded prompting：prompt engineering 常见 trick
- Code verification：编译器 basic pass

**但组合在一起揭示了一个 design pattern**：LLM 落地到任何 safety-critical domain 都需要这五层 scaffolding：

1. **Domain-specific function library** (相当于 API doc)
2. **Domain-specific grounding rules** (相当于 linting rules)
3. **Domain-specific verification** (相当于 type checker)
4. **Domain-specific visualization** (相当于 debugger)
5. **Multi-level abstraction** (让 novice 和 expert 同一个 interface)

每个垂直领域都要重做一遍。这就是为什么我觉得这种 end-user programming + LLM 的方向有产品潜力——chemistry 一遍、biology 一遍、manufacturing 一遍、hospitality 一遍，每个都值得一个 startup。

这跟你 Software 3.0 的判断高度一致：**LLM 是新编译器，但编译器周围还需要 linker、debugger、profiler、runtime 这套生态**。Alchemist 是 robotics domain 的一个早期 scaffolding prototype。

---

## 7. 几个值得深挖的联想

### 7.1 跟 [Code as Policies](https://arxiv.org/abs/2209.07753) 的关系

Code as Policies (Liang et al. 2023) 是 Google 的工作，LLM 直接 generate Python code 作为 robot policy，可执行、可组合、可泛化。Alchemist 是 Code as Policies 的 **end-user-friendly 版本**——加了 GUI、RViz preview、verification、两层 abstraction、grounded prompting。Code as Policies 是 research demo，Alchemist 是 product prototype。

### 7.2 跟 [Voyager](https://voyager.minedojo/) 的关系

Voyager (Wang et al. 2023) 在 Minecraft 里让 LLM agent 自动学习新 skill 并加入 skill library。Alchemist 现在的 function library 是固定的，user 自己反复用的 general function 不会自动加入 library。**未来方向很明显**：让 user 反复使用的 code pattern 自动 promote 成 library function，下次 LLM 可以直接 reference。这就是 Voyager 思路在 robotics end-user programming 的应用。

### 7.3 跟 [Robotic Chemist (Nature 2020)](https://www.nature.com/articles/s41586-020-2442-2) 的对比

Nature 那篇 robotic chemist 是 hardware-heavy 方案——专门的硬件 + 专门写的 control software，做 solid-state photocatalyst 合成 search。Alchemist 是 software-heavy 方案——通用机械臂 + LLM 驱动的灵活 task definition。两条路径未来一定会 merge：robotic chemist 那种 closed-loop Bayesian optimization 实验，task specification 由 Alchemist 这样的 LLM 系统给，hardware 用通用 UR5/Franka。

### 7.4 跟你 [nanoGPT](https://github.com/karpathy/nanoGPT) / [micrograd](https://github.com/karpathy/micrograd) 的精神联系

你一直强调 "build intuition by building from scratch"。Alchemist 的 function library 本质上就是 robot 的 "micrograd"——一个 minimal 但完整的 API surface 让 LLM 可以 compose 出任意 robot program。High-level function 像 `Module` class 的抽象，low-level function 像 `Value` 的 micro-op。两层 abstraction 让 user 可以选 granularity，这跟你 micrograd 里展示的 "you can use .backward() or write the chain rule manually" 的 dual-mode 思路一致。

---

## 8. References

直接相关：
- [Alchemist paper (HRI 2024)](https://doi.org/10.1145/3610977.3634969)
- [Forgetful LLMs (前作 arXiv)](https://arxiv.org/abs/2310.06646)
- [Code as Policies](https://arxiv.org/abs/2209.07753)
- [ChatGPT for Robotics (Microsoft)](https://www.microsoft.com/en-us/research/publication/chatgpt-for-robotics-design-principles-and-model-abilities/)
- [Voyager](https://voyager.minedojo.org/)
- [Robotic Chemist (Nature)](https://www.nature.com/articles/s41586-020-2442-2)
- [SayCan](https://arxiv.org/abs/2204.01691)
- [PaLM-E](https://arxiv.org/abs/2303.03378)
- [ProgPrompt](https://arxiv.org/abs/2209.10263)
- [Inner Monologue](https://say-can.github.io/)
- [TidyBot](https://arxiv.org/abs/2305.05658)
- [RT-2](https://robotics-transformer2.github.io/)
- [Inagaki et al. 2023 (bio lab automation)](https://arxiv.org/abs/2304.10267)

User study & prompt engineering：
- [Why Johnny Can't Prompt (CHI 2023)](https://doi.org/10.1145/3613904.3642739)
- [What It Wants Me To Say (CHI 2023)](https://doi.org/10.1145/3610591)
- [Expectation vs. Experience (CHI EA 2022)](https://doi.org/10.1145/3491101.3519665)
- ["According to..." Prompting](https://arxiv.org/abs/2305.13252)
- [Self-Refine](https://arxiv.org/abs/2303.17651)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)

Code gen evaluation：
- [EvalPlus (Liu et al.)](https://arxiv.org/abs/2305.01210)
- [LLM is Like a Box of Chocolates](https://arxiv.org/abs/2308.02828)

Formal verification for robotics：
- [Luckcuck et al. 2019 survey](https://dl.acm.org/doi/10.1145/3335699)
- [KeYmaera X](https://github.com/LS-Lab/KeYmaera-X)

Infrastructure：
- [AR Track Alvar ROS](http://wiki.ros.org/ar_track_alvar)
- [RViz](http://wiki.ros.org/rviz)
- [OpenAI Whisper](https://arxiv.org/abs/2212.04356)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

Karpathy context：
- [Software 3.0 talk](https://www.youtube.com/watch?v=LCEmiRj41tY)
- [Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [micrograd](https://github.com/karpathy/micrograd)

---

**直觉 takeaway**：Alchemist 是把"人话变可执行 robot program"这件事从 demo 阶段推到了 prototype IDE 阶段，它的价值不在某个 component 多 clever，在于揭示了 **LLM 落地到 safety-critical physical domain 需要哪五层 scaffolding**。这个 design pattern 我觉得会反复出现在 robotics、biology、chemistry、manufacturing 等领域的 LLM-based end-user programming 工具里。

---

# Alchemist: LLM-Aided End-User Robot Programming — 深度解析

Andrej, 这篇是 Yale + JHU 投在 HRI 2024 的工作，作者包括 Ulas Berk Karli、Juo-Tung Chen（与作者前作 [Forgetful LLMs](https://arxiv.org/abs/2310.06646) 一脉相承）、Victor Nikhil Antony 和 Chien-Ming Huang。核心 motivation 我觉得非常清晰：**把 end-user robot programming 从 "user specifies programming logic" 转向 "user specifies desired program outcomes, LLM produces detailed specifications"**，让生物化学家这种 domain expert 但 programming novice 的人能用自然语言驱动机械臂。这本质上和你之前反复讨论的 "Software 2.0/3.0" 想法是高度一致的——意图替代显式算法。

Paper link: https://doi.org/10.1145/3610977.3634969  
arXiv preprint 风格版：https://arxiv.org/abs/2310.06646 (前作 Forgetful LLMs)

---

## 1. 整体架构与设计哲学

Alchemist 的设计目标（DO1–DO5）本质上是把 "IDE for LLM-robot programming" 重新拆成几个关键 primitives：

| Design Objective | 直觉意义 |
|---|---|
| DO1 Natural Language | 把 program logic 从 user side offload 给 LLM |
| DO2 End-to-end workflow | 写代码 → preview → 调试 → 执行在一个 GUI 里完成 |
| DO3 Varied proficiency | 同一个 system 要同时伺候 novice 和 expert，所以 function library 必须分两层 abstraction |
| DO4 Visualization | 物理动作不可撤销，必须 RViz preview |
| DO5 Modularity | LLM-agnostic + robot-platform-agnostic |

让我重点说 DO3，因为这是整个 system 最巧妙的设计——**abstraction level 不是把 function library 做成两套，而是通过 initial prompt 让 LLM 自己选择输出哪种 granularity 的代码**。这点很关键，因为它意味着 system 的"接口复杂度"是 prompt-driven 的，不是 library-driven 的。

### 1.1 前端三层结构

```
+-------------------+-------------------+-------------------+
| 3D RViz Panel     | Chat Panel        | Terminal Panel    |
| (visualization)   | (LLM 对话/voice)  | (Python REPL +    |
|                   | (Whisper ASR)     |  built-in helpers)|
+-------------------+-------------------+-------------------+
        +--- Editor + File Tree (toggle, 默认隐藏) ---+
```

Editor 和 File Tree **默认隐藏**这一点很巧妙，作者明确解释：避免让 novice 用户被 code 吓退。这其实暗示了一个非常重要的 design principle：**对于 end-user programming system, "not showing code" is itself a feature**, 这和你之前讲 Agent/Micrograd 时的"complexity budget"是一回事——人类 cognitive load 是稀缺资源。

### 1.2 后端三大组件

后端有三个关键模块，我一个个讲：

#### (A) Function Library (双层 abstraction)

High-level example:
```python
pour(target_name: str) -> None
# 机器人把当前 gripped 的容器倒入 target_name 容器
```

Low-level example:
```python
move(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None
# 把 end-effector 移到指定 6-DoF pose
```

数学上，可以把 function library 看作一个 API 接口空间 $\mathcal{A} = \mathcal{A}_{hi} \cup \mathcal{A}_{lo}$，其中：
- $\mathcal{A}_{hi}$: 任务级 primitives，input 维度低，input domain 离散（如 target_name ∈ workspace objects）
- $\mathcal{A}_{lo}$: 几何级 primitives，input 维度高（6+），input domain 连续（SE(3) poses）

LLM 在生成 code 时选择哪个 subset $\mathcal{A}^{(k)} \subset \mathcal{A}$，由 initial prompt + 当前 task context 决定。这其实和 [Code as Policies](https://arxiv.org/abs/2209.07753) 中 "value functions as APIs" 的思想很相似，但 Alchemist 是把 value function 拆成两层 granularities。

#### (B) Initial Prompting

Initial prompt 有四个 component：

1. **System role prompt**: "你是 robot programming helper"，加上机器人元信息（DoF, end-effector type, warnings like "always use floating numbers"）
2. **Function library prompt**: 每个 function 的 name, inputs, functionality, outputs + axis/unit conventions
3. **Environment prompt**: 列出 workspace 中所有物理对象及其 dimensions（beakers, graduated cylinders）
4. **Example user prompt → example code output**: few-shot in-context example，**故意设计成用上 function library 里大部分 function**，强化正确 usage pattern

这点呼应了 Vemprala et al. 的 [ChatGPT for Robotics](https://www.microsoft.com/en-us/research/publication/chatgpt-for-robotics-design-principles-and-model-abilities/) (MSR-TR-2023-8) 中的 design principles，但 Alchemist 多了 environment prompt 这一层——这其实是一种 weak form of "grounding"。

可以用一个简化公式来描述 LLM 生成的条件分布：

$$P_{\theta}(\text{code} \mid \underbrace{s_{\text{role}}, \mathcal{L}_{\text{func}}, \mathcal{E}_{\text{env}}, \text{Ex}_{\text{few-shot}}}_{\text{system prompt}}, \underbrace{u_t}_{\text{user turn}}, \underbrace{g(u_t)}_{\text{grounding}})$$

其中：
- $\theta$: LLM 参数（GPT-4）
- $s_{\text{role}} \in \Sigma^*$: 角色 prompt 字符串
- $\mathcal{L}_{\text{func}}$: function library description
- $\mathcal{E}_{\text{env}}$: 环境描述（object names + dimensions）
- $\text{Ex}_{\text{few-shot}} = \{(u^{(i)}, c^{(i)})\}_{i=1}^{k}$: few-shot examples
- $u_t$: 第 $t$ 轮 user input
- $g(\cdot)$: grounded prompting 函数（见下）

#### (C) Grounded Prompting + Code Verification

这是我觉得 paper 里最有意思的 engineering contribution，作者明确说 "grounded prompting was the most effective approach" 相比于 parsing code 或加 rules to initial prompt。

Grounding 函数 $g: \Sigma^* \to \Sigma^*$ 定义为：

$$g(u) = \underbrace{\text{``By using the function library you are provided,''}}_{\text{prefix grounding}} \oplus u \oplus \underbrace{\text{``make sure to move back to home after the task is finished.''}}_{\text{suffix grounding}} \oplus \bigoplus_{i} \mathbb{1}[w_i \in u] \cdot r_i$$

其中 $\oplus$ 表示 string concatenation，$\mathbb{1}[\cdot]$ 是 indicator function，$w_i$ 是 trigger keyword，$r_i$ 是对应追加的规则。Paper 里给出的三条 conditional groundings：

| Trigger $w_i$ | 追加规则 $r_i$ | 直觉意义 |
|---|---|---|
| "add" | "Make sure to use marker location." | 强制用 AR marker 而非想象坐标 |
| "pour" | "Don't move above the beaker before pouring; just call the pour function. Also, after pouring, make sure you place the object back to where it was on the table and then open the gripper to release it." | 防 LLM 自己添加多余 motion 步骤 + 强制复位 |
| "function"/"generic"/"code" | "If you wrote a function, remember to add an example function call at the end." | 防 LLM 写出"定义但没调用"的死代码 |

直觉上这是一个 keyword-triggered 的 retrieval-augmented prompting，但 retrieval source 是手工的"prompt fix-up rules"，而不是文档库。这呼应了 Weller et al. 2023 ["According to..." prompting](https://arxiv.org/abs/2305.13252) 中 "prompt 自己 restructure 自己" 的思想，但更轻量。

Code verification 是另一层 rule-based post-processing，处理三类：
1. 缺失 `import rospy` 之类的 import 错误
2. ROS node initialization 缺失
3. Python version check

这其实是把 LLM 当成 "code skeleton generator"，然后用确定性 code-transformation pass 去补全 boilerplate——非常像 compiler 的 lower-pass。

---

## 2. Token 截断策略

Paper 里有一句很关键的话："we selectively truncate the middle segment of the conversation history and resend the API call to aid in error recovery."

这是一个典型的 **long-context reranking** 问题。假设完整对话 $H = (h_1, h_2, \ldots, h_N)$，他们保留：
- Prefix: $h_1, \ldots, h_k$（system prompt + few-shot + 早期 context）
- Suffix: $h_{N-m}, \ldots, h_N$（最近的对话）
- 截断中间 $h_{k+1}, \ldots, h_{N-m-1}$

直觉：开头决定"任务是什么"，结尾决定"现在在调试什么具体 bug"。中间的"对话主轴"通常不影响 error recovery，但占用大量 tokens。这个思想和 Anthropic 的 [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) 以及 Longformer 的 sliding-window attention 是同一类思路。

可以形式化为优化问题：

$$\min_{\text{truncate } \tau} \mathbb{E}_{\text{errors}}\Big[ \text{loss}(\text{LLM}(\tau(H))) \Big] \quad \text{s.t.} \quad |\tau(H)| \leq L_{\max}$$

他们用了一个非常实用的 heuristic：保留首尾两端，丢中间。这种 heuristic 在 code-assistant 领域其实非常 common，比如 Cursor、Continue 之类都做过类似。

---

## 3. Vision System: AR Marker-based Grounding

Alchemist 用 [AR Track Alvar](http://wiki.ros.org/ar_track_alvar) 作为视觉模块，**不是用 VLM 或 object detector**。关键 modification：每个 marker 编码了 grasping orientation 信息，marker 上有一个"pointy end"指示 gripper 应该怎么 approach/align，灵感来自 [Seifdgar et al. 2017 Situated Tangible Robot Programming](https://dl.acm.org/doi/10.1145/2909824.3020215)。

这其实是一个非常好的工程选择——**视觉不确定性 offload 给物理标记**，paper 里也承认这是 "fast and low cost way for quick testing and development"，并明说可以替换为 vision-language model（引用了 [MiniGPT-4](https://arxiv.org/abs/2304.10592)）。

我的直觉：这其实是把 grounding 问题从 software (VLM) 横向转移到了 hardware (physical tags)——典型的 "use the world as its own model" 思想，你肯定能 get 到这跟你的 [World Models](https://worldmodels.github.io/) 工作的内在联系。

---

## 4. 探索性实验设计

### 4.1 Task 选择

LB Media preparation——生物化学实验室里每周要做很多次的"准备培养皿"任务。把这个任务 toy-ify 成：把不同颜色的 beads（替代 reagent）从 graduated cylinders 倒进 beaker 混合。

非常聪明的选择：
- 是真实 lab 任务的简化版，生态效度高
- 涉及 pick-and-place + pouring 两个 primitives
- novice domain expert 能理解任务、但不懂 robotics
- expert robot programmer 反过来不懂 task domain

### 4.2 Participants 与 Measures

10 人，5 novice (bio/chem/biophysics PhD) + 5 expert (robotics PhD)。Measure matrix 我重新整理：

| Measure | 含义 |
|---|---|
| Prog. Time | 在 chat box 里 prompt LLM 的时间 |
| Debug. Time | 处理 errors 的时间 |
| Idle Time | 思考、看 user manual、看 RViz 等不直接交互时间 |
| Task Comp. Time | 总时间 |
| # Errors | 错误数量 |
| Error Types | Name/Syntax/Import/Factual/Physical |
| Editor Use | 是否手动改 code |
| Debug Method | prompt vs edit code |
| Use of Gen. Func. | 是否用 general function 而非 step-by-step |
| SUS | System Usability Scale（0-100，68 above average） |

### 4.3 关键发现（Table 2 重新解读）

| Group | Prog Time | Debug Time | Idle Time | Total | Errors | SUS |
|---|---|---|---|---|---|---|
| Novice mean | 23:09 | 13:40 | 26:13 | 1:03:02 | 1.8 | 56.0 |
| Expert mean | 16:41 | 6:38 | 43:27 | 1:01:59 | 1.8 | 68.5 |

惊人的相似性：**总完成时间几乎相同**。但分解后差异显著：
- Expert 编程快 28%、调试快 51%
- Expert idle 多 66%（在思考、规划 general function 结构）
- **两组 errors 数量相同**（1.8），但 type 不同

定性观察：
- **Novice 倾向 prompt debugging**（多轮 dialog 修复），不愿打开 editor
- **Expert 倾向 direct code editing**
- **Novice 避免 general functions**，喜欢 step-by-step prompting
- N4 是 outlier（SUS 12.5），原因是 motion planning 和 vision 失败多次撞倒东西——但即便如此 N4 仍说 "I would love to have a liquid handling system in our lab"

这印证了我一直觉得 paper 里没明说但很重要的一个 insight：**LLM 把 debugging 从 "code level" 上抬到了 "dialog level"，让 novice 可以 bypass syntax debugging 这一层**。但同时 expert 反而觉得 dialog debugging 比 code editing 更慢——这其实呼应了 Vaithilingam et al. 2022 [Expectation vs. Experience](https://doi.org/10.1145/3491101.3519665) 的发现。

---

## 5. Lessons Learned 的深度分析

### Lesson 1: LLMs Can Output Unreliable Code

引用的工作：
- [Liu et al. 2023 "Is Your Code Generated by ChatGPT Really Correct?"](https://arxiv.org/abs/2305.01210) - Code generation 评估
- [Ouyang et al. 2023 "LLM is Like a Box of Chocolates"](https://arxiv.org/abs/2308.02828) - LLM 非确定性
- [Luckcuck et al. 2019](https://dl.acm.org/doi/10.1145/3335699) - Formal verification for autonomous systems

他们提出的两步 mitigation：
1. **Code verification**（deterministic post-processing）
2. **Grounded prompting**（prompt-level biasing）

这两步叠在一起本质上是 **defense-in-depth for code generation**，类比于编译器的 lexer+parser+type-checker 多层验证。但还有更激进的路径：把 [formal verification](https://arxiv.org/abs/1902.05654) 直接嵌入 LLM 生成循环（最近 Anthropic、Google DeepMind 都在尝试用 Lean + LLM），这在 robotics 这种 safety-critical 场景是 future work 的金矿。

### Lesson 2: Effective LLM Prompting is Difficult

引用 [Zamfirescu-Pereira et al. 2023 "Why Johnny can't prompt"](https://doi.org/10.1145/3613904.3642739)——非 AI 专家写 prompt 容易出 vague/implicit 问题。

Alchemist 的两个 mitigation：
1. **Guided training assets**: user manual + tutorial video + training task
2. **Dynamic context-dependent grounding**: 即上面讲的 trigger keyword → 追加 rule

但 paper 自己也承认 grounding rules 是 domain-dependent。要扩展到新 domain 需要重新写 rules。Future direction 我会想到几个：
- **LLM 自己生成 grounding rules**: meta-prompt 让 LLM 自己 propose 规则
- **RAG with documented failure modes**: 把过去所有 errors 检索出来作为 grounding context
- **Constitutional AI 风格的自我 critique**: LLM 生成 code 后自我 review 一遍

### Lesson 3: End-User Aversion to Direct Coding

这是非常深刻的观察。Novice 用户宁可在 dialog 里绕远路，也不愿意打开 editor。这跟 Liu et al. 2023 ["What It Wants Me To Say"](https://doi.org/10.1145/3610591) 的发现一致——end-user 和 code-gen LLM 之间存在 abstraction gap。

Alchemist 的 design choice：editor 默认隐藏。但这其实是 workaround，更彻底的解决路径是：
- **Conversational function invocation**: 让用户用自然语言"调用" general function，而不是去 terminal 打 `pick_and_pour('beaker1', 'beaker2')`
- **Live preview of program structure**: 把生成的 code 实时可视化成 behavior tree 或 flow chart，让 novice 看到"程序结构"而不看到"代码文本"
- **Progressive disclosure**: novice 模式默认隐藏 code，但能 toggle 看到，progressive 解锁

---

## 6. 与相关 LLM-Robotics 工作的位置关系

整个 LLM-for-robotics 生态，我画一个粗略 taxonomy：

### 6.1 LLM as Task Planner
- [SayCan (Ahn et al. 2022)](https://arxiv.org/abs/2204.01691): LLM 提供任务分解，affordance 函数过滤不可行 action
- [ProgPrompt (Singh et al. 2023)](https://arxiv.org/abs/2209.10263): LLM 生成 Python-like task plan
- [Inner Monologue (Huang et al. 2022)](https://say-can.github.io/): LLM 内部 monologue 做 embodied reasoning

### 6.2 LLM as Code Generator
- [Code as Policies (Liang et al. 2023)](https://code-as-policies.github.io/): LLM 生成 Python code 作为 robot policy
- [ChatGPT for Robotics (Vemprala et al. 2023)](https://www.microsoft.com/en-us/research/publication/chatgpt-for-robotics-design-principles-and-model-abilities/): Alchemist 直接借鉴的 design principles
- [Language to Rewards (Yu et al. 2023)](https://arxiv.org/abs/2306.08647): LLM 把 NL 翻译成 reward function

### 6.3 LLM as End-User Programming Interface
- [Inagaki et al. 2023](https://arxiv.org/abs/2304.10267): bio lab automation，跟 Alchemist 最接近的工作
- **Alchemist**: 在此基础上加了 RViz preview + code verification + grounded prompting

### 6.4 LLM + VLM 多模态
- [PaLM-E (Driess et al. 2023)](https://palm-e.github.io/): embodied multimodal LLM
- [TidyBot (Wu et al. 2023)](https://arxiv.org/abs/2305.05658): personalized robot assistance with LLM

Alchemist 的独特定位是 **end-user programming system 而非 policy 或 planner**——它不是替换 robot 控制栈，而是替换 robot 编程工作流。这点和 Microsoft ChatGPT for Robotics 一样，但 Alchemist 多了 IDE-level 的 GUI integration。

---

## 7. 几个值得深究的技术细节

### 7.1 Two-level Abstraction 与 LLM 输出分布

可以用信息论视角分析。设 $A \in \{hi, lo\}$ 为 abstraction level 选择，$C$ 为 generated code, $U$ 为 user prompt。则：

$$P(A = hi \mid U) \propto P(U \mid A = hi) \cdot P(A = hi)$$

这里 prior $P(A = hi)$ 由 initial prompt 中的 example bias 决定（如果 example 主要用 high-level function，LLM 会倾向 high-level）。Likelihood $P(U \mid A)$ 则由 user prompt 的 specificity 决定：
- "Pour the 100ml into the 500ml beaker" → $P(A = hi \mid U)$ 高
- "Move the gripper to (0.3, 0.2, 0.4) with orientation (0, π/2, 0)" → $P(A = lo \mid U)$ 高

Alchemist 没显式建模这个，但通过 few-shot example 间接 biasing。这其实是一个潜在改进点：可以用一个轻量 classifier 在 user prompt 上先判 abstraction level，然后 conditionally 选择不同的 system prompt。

### 7.2 Grounded Prompting 的 Information-Theoretic 视角

Grounding rule $r_i$ 在 trigger $w_i$ 出现时被激活，相当于增加了一个 conditional prior：

$$P(\text{code} \mid U, r_i) = \frac{P(r_i \mid \text{code}) P(\text{code} \mid U)}{P(r_i \mid U)}$$

直觉：当 user prompt 里有 "pour" 时，"after pouring return object" 这个 rule 相当于告诉 LLM "code 必须在分布尾部包含 return-to-origin action"。这就是 prompt-based posterior shaping。

### 7.3 与 Reinforcement Learning from Human Feedback 的关系

Grounded prompting 其实可以看作一种 **prompt-time feedback injection**——而不是把 feedback 烧进 model weights（像 RLHF）。这种 in-context feedback 的优势是：
- 不需要 fine-tune
- domain-switching 容易（换 rules 即可）
- 可解释（用户能直接看到追加了什么）

劣势是：
- 占用 context window
- 不能学到长期偏好
- 对 LLM 的 in-context learning 能力依赖强

最近的 [Constitutional AI](https://arxiv.org/abs/2212.08073) 和 [Self-Refine](https://arxiv.org/abs/2303.17651) 思路可以看作是把这个机制自动化的尝试。

---

## 8. 局限性与未来方向（论文 + 我个人的延伸）

### 8.1 论文承认的
- Sample size 小（10 人）
- 单一 task domain（LB Media）
- Lab setting 而非 real-world deployment
- 没和 state-of-the-art system 对比

### 8.2 我觉得还可以延伸的

1. **Multi-turn Program Refinement as RL**: 把 LLM 生成 → 用户反馈 → 修正循环看作一个 bandit，每个修正的 prompt 是一次 reward signal，可以 online fine-tune LLM（或仅 fine-tune 一个 prompt head）。

2. **Vision-Language Model 替换 AR Markers**: 引用 [MiniGPT-4](https://arxiv.org/abs/2304.10592) 的 vision-language grounding 完全可以替代 AR tag——这样 environment prompt 可以从感知自动生成，不需要手工 enumerate beaker dimensions。

3. **Safety via Formal Verification**: [Luckcuck et al. 2019 survey](https://dl.acm.org/doi/10.1145/3335699) 提到 formal methods。具体到这个工作，可以用 [Frama-C](https://frama-c.com/) 或 [KeYmaera X](https://github.com/LS-Lab/KeYmaera-X) 把生成的 code 在 motion planning 层做 hybrid systems verification，保证 robot 不会进入 unsafe states。

4. **Hierarchical Code Generation**: 类似 [Voyager (Guanzhi Wang et al.)](https://voyager.minedojo.org/) 在 Minecraft 里做 skill library 增长——Alchemist 可以让 user 反复使用的 general function 自动加入 function library，下次 user prompt 就可以直接 reference。

5. **Cross-embodiment Generalization**: 现在支持 UR5, Franka Panda, TIAGo。但 function library 是手工 mapping 的，可以借鉴 [RT-2](https://robotics-transformer2.github.io/) 思路，让 LLM 直接 generate 跨平台 code 通过 embodiment description prompting。

6. **Active Learning for Prompting**: Novice user 给 vague prompt 时，LLM 应该主动 ask clarifying question 而不是直接 generate code。这是 [Inner Monologue](https://say-can.github.io/) 的精神，但 Alchemist 还没集成。

---

## 9. 与你的工作的潜在关联

Andrej，从我对你 [Software 3.0 talk](https://www.youtube.com/watch?v=LCEmiRj41tY)、[Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) 等内容的理解，Alchemist 这个工作本质上是 **Software 3.0 范式在 robotics 领域的一个落地 case**：

- Software 1.0: 写 C++/Python ROS node 控制 UR5
- Software 2.0: 训练 neural network policy（如 RT-2）
- Software 3.0: 用 NL 作为编程语言，LLM 作为"编译器"生成可执行 code

Alchemist 是一个非常具体的 Software 3.0 实例，而且它的限制——unreliable code, vague prompts, end-user aversion to coding——都是 Software 3.0 范式在落地时的共性挑战。你在 micrograd/nanoGPT 系列里反复强调的 "build intuition by building from scratch"，我觉得 Alchemist 这种 system paper 的读法应该是：**它告诉我们 Software 3.0 在 robotics 落地需要哪些 "engineering scaffolding"**——visualization panel, terminal, code verification, grounded prompting, two-level abstraction——这些其实都是 LLM 这层"编译器"周围还需要人类工程化补的洞。

更深一层，这个 paper 提示了一个 universal 模式：**LLM 落地到任何 domain（不止 robotics）都需要类似的 scaffolding**：
- Domain-specific function library (相当于 API documentation)
- Domain-specific grounding rules (相当于 linting rules)
- Domain-specific verification (相当于 type checker)
- Domain-specific visualization (相当于 debugger)

每个 domain 都要做一遍。这就是为什么我直觉觉得 end-user programming + LLM 是一个非常有产品潜力的方向——每个垂直领域都值得重做一遍。

---

## 10. Web References

主要论文：
- [Alchemist HRI 2024 paper](https://doi.org/10.1145/3610977.3634969)
- [Forgetful LLMs (前作)](https://arxiv.org/abs/2310.06646)
- [ChatGPT for Robotics (Microsoft, Vemprala et al.)](https://www.microsoft.com/en-us/research/publication/chatgpt-for-robotics-design-principles-and-model-abilities/)
- [Code as Policies (Liang et al.)](https://arxiv.org/abs/2209.07753)
- [SayCan (Ahn et al.)](https://arxiv.org/abs/2204.01691)
- [PaLM-E (Driess et al.)](https://arxiv.org/abs/2303.03378)
- [ProgPrompt (Singh et al.)](https://arxiv.org/abs/2209.10263)
- [Inner Monologue (Huang et al.)](https://say-can.github.io/)
- [Robotic Chemist (Burger et al., Nature 2020)](https://www.nature.com/articles/s41586-020-2442-2)
- [Inagaki et al. 2023 Bio Lab Automation](https://arxiv.org/abs/2304.10267)
- [Voyager (Wang et al.)](https://voyager.minedojo.org/)
- [RT-2 (Brohan et al.)](https://robotics-transformer2.github.io/)
- [TidyBot (Wu et al.)](https://arxiv.org/abs/2305.05658)

Prompt engineering & user study：
- [Why Johnny Can't Prompt (Zamfirescu-Pereira et al. CHI 2023)](https://doi.org/10.1145/3613904.3642739)
- [What It Wants Me To Say (Liu et al. CHI 2023)](https://doi.org/10.1145/3610591)
- [Expectation vs. Experience (Vaithilingam et al.)](https://doi.org/10.1145/3491101.3519665)
- ["According to..." Prompting (Weller et al.)](https://arxiv.org/abs/2305.13252)

Code generation evaluation：
- [Is Your Code Generated by ChatGPT Really Correct? (EvalPlus)](https://arxiv.org/abs/2305.01210)
- [LLM is Like a Box of Chocolates (Ouyang et al.)](https://arxiv.org/abs/2308.02828)

Related infrastructure：
- [AR Track Alvar ROS package](http://wiki.ros.org/ar_track_alvar)
- [RViz documentation](http://wiki.ros.org/rviz)
- [OpenAI Whisper](https://arxiv.org/abs/2212.04356)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

Related Karpathy works for context：
- [Software 3.0 talk (Andrej Karpathy)](https://www.youtube.com/watch?v=LCEmiRj41tY)
- [Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [micrograd](https://github.com/karpathy/micrograd)

---

## 11. 一句话总结直觉

**Alchemist 是把 Software 3.0 范式（自然语言 → 可执行程序）落地到 robotics 这类 safety-critical、physical-action domain 时的第一个相对完整的 IDE 探索**，它的核心贡献是揭示了 LLM 周围还需要哪些"工程支撑结构"才能让 novice user 安全、可靠、有效地使用：function library abstraction, grounded prompting, code verification, RViz preview, optional editor。这些 components 单看都不新颖，但组合在一起揭示了一个非常重要的 design pattern，我觉得在未来的 robotics end-user programming、乃至其他 domain 的 LLM-based programming tools 中都会反复出现。
