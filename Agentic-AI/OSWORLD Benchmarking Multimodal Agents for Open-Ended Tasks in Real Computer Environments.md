---
source_pdf: OSWORLD Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer
  Environments.pdf
paper_sha256: d4c6e20dd59467f005561b1e97199f9842fd3b0e9fdd93e66e06ba0ec09edfdb
processed_at: '2026-08-06T01:37:00-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OSWORLD 用人话讲

## 一、为什么有这篇 paper

你 2017 年做 MiniWoB++ 的时候，整个领域觉得"web agent 能在浏览器里点按钮"就很厉害了。7 年过去了，大家发现一个问题：**所有 benchmark 都活在假世界里**。

WebArena 把 agent 关在浏览器里，MiniWoB++ 把任务切成 5 秒一个的小碎片，AITW 在 Android 模拟器里玩，AgentBench 把 5 个隔离子环境拼一起。它们都没回答一个最朴素的问题：**把 agent 扔回真实操作系统，它还能干活吗？**

OSWORLD 就是来回答这个问题的。它给你一台 Ubuntu VM（也有 Windows），让你用 raw 的鼠标键盘控制，像人一样点像素、敲键盘、切窗口、跨应用干活。就这么简单——简单到你觉得"这不就是 computer use 吗，怎么 2024 年才有人做"。但做起来巨难，难到 9 个 CS 研究生花了 1800 工时才搞出 369 个任务。

## 二、它到底是什么

一句话：**一台 VM + 一个 config 文件 + 一堆 evaluation 脚本**。

config 文件描述三件事：
1. 怎么 setup（从 snapshot 恢复 + 下载文件 + 打开特定 app 到特定状态）
2. agent 怎么交互（收 screenshot 和 a11y tree，输出 pyautogui 代码）
3. 怎么 eval（agent 完事之后取文件、跑脚本、给 0 或 1 分）

为什么是 VM 不是 Docker？Docker 跑不了 Windows、macOS，没 snapshot。VM 可以秒级恢复，可以并行跑多个任务，可以隔离 agent 的破坏行为。

**action space 是 pyautogui**——就是说 agent 要输出类似 `pyautogui.click(300, 540, button='right')` 这种 Python 代码。它不能用 DOM ref，不能用 "click button with text 'Submit'"，必须给坐标。这是和 MiniWoB++ 最大的区别：**把 grounding 难度全部推给模型**。

## 三、模型现在有多菜

看几个数字你就懂了：

| 谁干活 | 成功率 |
|---|---|
| 人类（CS 本科生，没接触过任务） | 72.36% |
| GPT-4 + a11y tree（最强配置） | 12.24% |
| GPT-4V 纯截图 | 5.26% |
| CogAgent 纯截图 | 1.11% |
| 多应用 workflow 任务，所有方法 | 0%-7% |

**人类 72%，最好的模型 12%，差 6 倍**。这不是"模型差点意思"，这是"模型完全不会"。

更尴尬的是 CogAgent——它是专门为 GUI agent 训的 VLM，在 OSWORLD 上 1.11%。说明**专门在 web data 上 finetune 也不够**，真实桌面的复杂度是另一个量级。

## 四、为什么这么菜

Paper 分析了 550 个失败 case，我帮你总结成四类：

### 1. 点不准（75%+ 的失败都涉及）

Agent 在 code comment 里写得头头是道："# 点击 LibreOffice 工具栏的居中对齐按钮"，然后 `pyautogui.click(400, 150)`——点偏了。它**知道自己要点什么**，**算不对点哪里**。这是 planning 强、grounding 弱的典型表现。

你看 MiniWoB++ 时代没这问题，因为 DOM 直接告诉你 ref。现在你给它 1920×1080 像素，它要自己从 200 万个像素里找到那个 20×20 的按钮中心——这和 robotics 里的 visual servoing 是一回事。

### 2. 点错后死循环

点错了，弹个广告窗、开了个不相关的菜单，agent 不知道怎么回去。它没有"undo"的概念，没有"我刚才点错了，应该关掉这个弹窗再重来"的元认知。于是反复点、反复错、耗光 15 步。

### 3. 软件知识为零

让它"tone down the brightness of this photo"在 GIMP 里做。它不知道 brightness 在 Colors 菜单下，就随机点 Colors 菜单里的各种选项——Exposure、Color Balance、Color Temperature——每次点开看不对就 Cancel，再点下一个。6 步用完还没找到 Brightness-Contrast。

人类 5 秒能找到的事，agent 0% 成功。因为它没看过 GIMP 的教程，没在 pretraining 里见过这些菜单结构。

### 4. 指令理解跑偏

"make Bing the main search thingy"——它理解成要进 Chrome 设置改搜索引擎，但路径走错了，点了 "Customise Chrome" 按钮以为是 "More" 按钮，一路瞎点到底。

更可怕的：让它"用 GIMP 剪辑视频 2-4 秒"，agent 直接开 terminal 敲 `ffmpeg -ss 00:00:02 -to 00:00:04 -i in.mp4 -c copy out.mp4`——**完全忽略了 instruction 里的 "use GIMP" 要求**。它走捷径，但走错了路。

## 五、几个有意思的发现

### 1. SoM 在真实桌面反而更差

SoM（Set-of-Mark，给截图上每个元素打数字编号）在 WebArena 系是 SOTA 方法。但在 OSWORLD 上，GPT-4V 用 SoM（11.77%）**低于** Screenshot+A11y（12.17%）。

为什么？Web 页面元素几十个，桌面应用元素几百上千个——一个 LibreOffice Calc 截图上 bounding box 能挤成马赛克，box 互相重叠，噪声盖过信号。而且很多任务需要点 spreadsheet 单元格的**精确坐标**，bounding box 表达不了"点 B6 单元格的左上角"这种事。

**所以 SoM 在 web 上 work，在 desktop 上不 work**。这是个值得 build intuition 的点：grounding 的难度跟元素密度强相关。

### 2. 截图历史没用，文本历史有用

你给它 3 轮历史 (obs, action)：
- 如果历史是 a11y tree（文本）：SR 提升
- 如果历史是 screenshot（图像）：SR 不动

说明**当前 VLM 不会从图像序列里提取上下文**。它看一张图能做事，看三张图反而糊涂。这和 robotics 里 video-conditioned policy 的困境一样——图像历史需要专门的 temporal encoding 机制，现在没有。

### 3. Claude-3 Opus 反常地差

Claude-3 Opus 在 GSM8K、HumanEval 上和 GPT-4 平起平坐，在 OSWORLD 上所有设置都比 GPT-4V 低 3-8 个百分点。

原因：**Claude 的 grounding hallucination 比 GPT-4V 严重**。比如：
- 双击文件，Claude 以为"选中"了，其实"打开"了
- 在 Calc 里把 B 列当 C 列
- 在 VS Code 替换框输入了 text/test，但没点 "Replace All"，以为完成了
- 最离谱的：让它复制 B6 单元格去 Chrome 搜索，它没复制就去 Chrome 地址栏粘贴，粘贴的是空的，然后**幻觉输出"Found the answer. Channel 31 in Hong Kong is RTHK TV 31."**——凭空编了个答案

GPT-4V 也会错，但它会"卡住、重复试、最后放弃"。Claude-3 会"自信地完成一个错误流程"。**前者是能力问题，后者是认知问题**——后者更危险。

### 4. 跨 OS 相关性 0.7

同一个任务在 Ubuntu 上 SR=4.88%，迁移到 Windows 上 SR=2.55%，相关系数 0.7。说明 agent 的能力**主要跟任务走，不跟 OS 走**。训练一个跨平台通用 agent 是可行的。

### 5. Infeasible 任务上 agent 反而更高

30 个"不可行任务"（功能废弃、功能幻觉），agent 在上面 SR=16.67%，比 feasible 任务的 13.34% 还高。但这是假象——agent 在 feasible 任务上没做好，反而过早放弃输出 `FAIL`，刚好蒙对了一些 infeasible 任务。**agent 不知道自己什么时候在错**。

## 六、和你工作的关联

你 2017 年的 MiniWoB++ 用 DOM ref 做 grounding，那是因为当时没 VLM，只能把视觉难度外包给浏览器。7 年后 VLM 出来了，大家想"那我们直接给截图吧"——OSWORLD 就是这个想法的极端版：**只给截图和 pyautogui，不给任何 ref**。

结果发现：**VLM 的像素 grounding 能力远不如预期**。12.24% 的成功率说明 GPT-4V 级别的模型在真实桌面上基本是个"看得懂但摸不准"的残疾人。

这和 robotics 里的 sim-to-real gap 是同构问题：
- MiniWoB++ = simulator with perfect state observation
- OSWORLD = real world with only camera observation
- 从前者到后者的跨越，需要的是**视觉 grounding 的根本性突破**

后续的 SeeClick、CogAgent、OmniParser、Anthropic Computer Use 都在补这个洞。但离人类 72% 还差 60 个百分点——**这是整个 multimodal agent 领域最大的 open problem**。

## 七、一句话总结

OSWORLD 把 agent 扔回真实操作系统，发现最好的模型只有人类 1/6 的水平。瓶颈在视觉 grounding——VLM 看得懂屏幕但点不准坐标。这和 robotics 的 visual servoing 是同一类问题，需要的是 grounding 能力的根本提升，不是 prompt engineering。

---

参考链接：
- Paper: https://arxiv.org/abs/2404.07972
- 项目页: https://os-world.github.io  
- Code: https://github.com/xlang-ai/OSWorld
- MiniWoB++ (你 2017 的工作): https://arxiv.org/abs/1702.08660
- SeeClick (后续 grounding 工作): https://arxiv.org/abs/2401.10935
- CogAgent: https://arxiv.org/abs/2312.08914
- Anthropic Computer Use (后续跟进): https://www.anthropic.com/news/3-5-models-and-computer-use
- OmniParser (改进的 SoM): https://github.com/microsoft/OmniParser

---

# OSWORLD: 真实计算机环境下的多模态 Agent Benchmark 深度解析

## 一、动机与定位

这篇 paper 来自港大 Tao Yu 组、CMU、Salesforce Research、Waterloo 合作，第一作者是 Tianbao Xie（之前做过 OpenAgents）。核心动机可以浓缩成一句话：**此前的 GUI agent benchmark 都不够"真"**。

让我把此前的 benchmark 阵营梳理一下：

- **静态 demonstration 类**（无执行环境）：MIND2WEB [9]、WebLINX [33]、PIXELHELP [27]、MetaGUI [47]、AITW [40]、OmniACT [21]、ASSISTGUI [13]、GAIA [36]。这类只有 (observation, action) 轨迹数据，不能执行，只能做 next-step prediction，并默认单一正确答案，错误惩罚了替代正确方案。
- **单一域执行环境**：
  - **Web**：MiniWoB++ [44,30]（你自己 2017 的工作）、WebShop [58]、WebArena [66]、VisualWebArena [22]、WorkArena [10] —— 全部限定在浏览器内，DOM 是部分可见的结构化信号。
  - **Mobile**：AndroidEnv [50]、WikiHow [61]、Mobile-env [61]、AppAgent [60] —— 限定在 Android UI。
  - **Coding**：InterCode [57]、SWE-bench [20]、DevBench [24] —— 限定在 shell/编辑器。
  - **Multi-isolated**：AgentBench [32] —— 多个隔离子环境拼接。

它们的共同问题是 action space 被简化（往往只有 click + type）、observation space 被简化（DOM/view hierarchy），任务域封闭，无法跨应用协作。OSWORLD 想做的，就是**把 agent 扔回真实的操作系统里**，让它面对一张 1920×1080 的截图、一个 raw 的键盘鼠标控制接口，自己决定点哪个像素、敲什么键。

这与 Adept 的 ACT-1 [1]、SeeAct [65]、UFO [59] 等近期工作的精神一致，但 OSWORLD 是首个把这件事做成 benchmark 的工作。

Paper 项目页：https://os-world.github.io  
GitHub：https://github.com/xlang-ai/OSWorld  
arXiv：https://arxiv.org/abs/2404.07972

## 二、环境架构

OSWORLD 的核心是一个**基于虚拟机的可执行环境**，整体架构可以画成：

```
┌──────────────────────────────────────────────────────────┐
│  Host Machine                                            │
│   ┌──────────────────────────────────────────────────┐   │
│   │ OSWORLD Coordinator                              │   │
│   │   ├─ reads config.json                           │   │
│   │   ├─ spawns VM instances (multiprocess)          │   │
│   │   └─ orchestrates setup / interact / eval        │   │
│   └──────────────────────────────────────────────────┘   │
│                       │                                   │
│   ┌───────────────────┴──────────────────────────────┐    │
│   │ Task Manager (per task)                          │    │
│   │  ┌────────────┐  ┌────────────┐  ┌───────────┐  │    │
│   │  │ Setup stage│→│ Agent Loop  │→│ Post-proc │  │    │
│   │  │ (red)      │  │ (obs/act)  │  │ (orange)  │  │    │
│   │  └────────────┘  └────────────┘  └───────────┘  │    │
│   │                        ↓                        │    │
│   │                ┌──────────────┐                  │    │
│   │                │ VM (Ubuntu/  │                  │    │
│   │                │ Win/macOS)   │                  │    │
│   │                │  snapshot    │                  │    │
│   │                │  + setup cmd │                  │    │
│   │                └──────────────┘                  │    │
│   │                        ↓                        │    │
│   │  ┌──────────────┐  ┌────────────┐  ┌────────┐  │    │
│   │  │ Fetch files  │→ │ Run getter │→ │Evaluate│  │    │
│   │  │ (yellow)     │  │ functions  │  │(green)  │  │    │
│   │  └──────────────┘  └────────────┘  └────────┘  │    │
│   └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 2.1 为什么是 VM 不是 Docker？

Paper 在 Appendix A.1 给出关键理由：Docker 不能跑自己的 kernel 和系统，跨 OS 兼容差（Windows、macOS 跑不了原生），更不支持 snapshot 这种"时刻冻结"。VM 提供三层好处：
1. **隔离**：agent 乱搞不会破坏 host；
2. **快照恢复**：每个任务可以秒级恢复到初始状态；
3. **跨 OS 一致接口**：可以用同一套 config 文件描述 Ubuntu/Win/macOS 任务。

### 2.2 任务形式化（POMDP）

Paper §2.1 把任务写成 POMDP $(S, O, A, T, R)$，这里我把每个符号的含义拆开：

- $S$：**state space**，是虚拟机内部完整计算机状态，包括所有进程、文件、窗口布局、内存等。**Agent 不可直接观测**，所以才叫 POMDP 而不是 MDP。
- $O$：**observation space**，包括（1）自然语言 instruction $I$；（2）screenshot $o^{img}$（1920×1080 RGB）；（3）accessibility tree $o^{a11y}$（XML 格式，Ubuntu 用 AT-SPI，Windows 用 PyWinAuto）；（4）terminal output $o^{term}$。
- $A$：**action space**，pyautogui 代码字符串，比如 `pyautogui.click(300, 540, button='right')`、`pyautogui.hotkey('ctrl', 'alt', 'del')`。还有三个特殊动作 `WAIT`、`FAIL`、`DONE`。
- $T: S \times A \to S$：**transition**，由 VM 内部的 pyautogui 执行器实现。
- $R: S \times A \to [0,1]$：**reward function**，只在最后一步给值，任务完全成功给 1，部分成功给 (0,1) 之间的小数，agent 对不可行任务正确预测 `FAIL` 也给 1。

交互循环：在 step $t$，agent 收到 $o_t \in O$，输出 $a_t \in A$，环境执行得到 $s_{t+1}$ 和 $o_{t+1}$，重复直到 `DONE`/`FAIL` 或达到最大步数 15。

### 2.3 初始状态 setup：被低估的关键设计

之前的 benchmark 大多从"应用刚启动"开始。OSWORLD 论文里强调**真实任务发生在工作流中间**：用户已经在编辑一个 spreadsheet，Chrome 已经开了 7 个 tab，Thunderbird 已经登录了账户。所以 OSWORLD 用了一种 **hybrid 配置法**：

1. **VM snapshot**：恢复基础系统状态；
2. **文件准备**（optional）：从云端下载所需文件，通过 LAN 上传到 VM；
3. **preprocessing 命令**（optional）：用 OS API 或 pyautogui 打开特定文件、跳到特定页、调整窗口大小等。

为什么不全部用 snapshot？因为每个 example 一个 snapshot 需要 GB 级空间，369 个任务就是几百 GB；hybrid 方案把"硬状态"放进 snapshot，"软状态"放进配置脚本，空间效率高两个数量级。

### 2.4 Action Space：pyautogui vs computer_13

Paper §2.4 + A.3 提供两种 action space：

**pyautogui（主用）**：直接生成 Python 代码字符串，自由度高，能用 for 循环批量操作，token 开销低。完整动作包括 `moveTo / click / dragTo / write / press / hotkey / scroll / keyDown / keyUp` 等。

**computer_13（RL 友好变体）**：13 类离散 action + 3 个特殊 action（WAIT/FAIL/DONE），每个 action 有参数化枚举（详见 paper Table 8）。这是为了未来 RL 研究预留的接口。

注意一个有意思的发现：当 grounding 信号不足时，模型会自发地写出 `pyautogui.locateOnScreen('Apple.png')` 这种"图像识别"代码——这其实暗示了一种**符号-视觉混合 grounding** 的可能性，paper 在 A.3.1 末尾提了一句但没展开。

## 三、Benchmark 构造

### 3.1 任务来源与软件选择

操作系统选 Ubuntu 22.04（开放、API 友好），Windows 11 作辅助分析集，macOS 因为版权问题不开发。

8 类核心软件 + 系统：

| 类别 | 软件 | 选它做什么 |
|---|---|---|
| Office | LibreOffice Calc / Writer / Impress | 高密度 GUI 元素、坐标级操作 |
| Daily | Chrome, VLC, Thunderbird | 浏览器 / 媒体 / 邮件 |
| Professional | VS Code, GIMP | 编辑器扩展配置 / 图像编辑 |
| OS | terminal, file manager, image viewer, PDF viewer | CLI + 文件 IO |

任务来源覆盖：官方文档、Ask Ubuntu、SuperUser、StackOverflow、Reddit、Quora、YouTube、TikTok、WikiHow、Coursera、Medium、个人 blog 等（详见 Table 9）—— 这是一个**真实人类求助语料**的集合，刻意保留了 unprofessional 表达（如 "make Bing the main search thingy"、"tone down the brightness"），这点很重要，它考察 agent 的**指令理解鲁棒性**。

### 3.2 数据统计

Paper Table 3 给出关键数字：

| 统计项 | 数量 |
|---|---|
| Ubuntu tasks | 369 (100%) |
| ├ Multi-App Workflow | 101 (27.4%) |
| ├ Single-App | 268 (72.6%) |
| ├ Integrated (来自其他 benchmark) | 84 (22.8%) |
| └ Infeasible | 30 (8.1%) |
| Windows tasks (analysis) | 43 |
| Distinct initial states | 302 |
| Distinct evaluation functions | 134 |

**Infeasible 任务**是亮点之一：30 个任务涉及**已废弃功能**或**幻觉功能**（用户以为存在但实际不存在），要求 agent 自己探索后输出 `FAIL`。这是对 agent **world knowledge + 自我校准能力**的考察，避免了"agent 永远会输出某种 action"的退化策略。

**134 个独立 evaluation function** 是这个 benchmark 的工程量证明。WebArena 只有 5 个、MiniWoB++ 有 125 个（每个 mini task 一个），但都是规则化模板；OSWORLD 的 134 个函数每个都针对具体任务写，需要解析 xlsx 内部 XML、调用 Playwright 取 cookie、用 Firefox Decrypt 解密 Thunderbird 账户、调用 VLC HTTP 接口、读 VS Code 扩展 API、用 pillow 比对图像等。

### 3.3 与现有 benchmark 对比

Paper Table 4 是核心对比，我重画关键列：

| Benchmark | Exec. Env. | Env. Scalability | Multimodal | Cross-App | Init State | #Eval Funcs |
|---|---|---|---|---|---|---|
| GAIA | ✗ | ✓ | ✗ | ✗ | ✗ | 0 |
| MIND2WEB | ✗ | ✓ | ✓ | ✗ | ✗ | 0 |
| MiniWoB++ | Web | ✗ | ✗ | ✗ | ✗ | 125 |
| WebArena | Web | ✗ | ✗ | ✗ | ✗ | 5 |
| VisualWebArena | Web | ✗ | ✓ | ✗ | ✗ | 6 |
| AgentBench | Multi-iso | ✗ | ✗ | ✗ | ✗ | 7 |
| **OSWORLD** | **Computer** | **✓** | **✓** | **✓** | **✓** | **134** |

OSWORLD 是唯一一个同时满足"真实计算机环境 + 可扩展 + 多模态 + 跨应用 + 中间状态初始化"的。

### 3.4 人类表现基线

Paper §3.4 测了人类表现（CS 本科生，未预先接触过任务）：
- **OSWORLD**：accuracy 72.36%，median 完成时间 111.94s
- **WebArena**：accuracy 88%，median 完成时间 35.38s

OSWORLD 任务对人类也更难、更耗时，说明任务本身确实复杂，不是"agent 笨"的问题。

## 四、Agent Baselines 与实验设置

### 4.1 四种输入设置

| 设置 | 输入 | 模型 |
|---|---|---|
| A11y tree | 简化后的 a11y tree（tab-separated） | Mixtral-8x7B, Llama-3-70B, GPT-3.5, GPT-4, Gemini-Pro, Gemini-Pro-1.5, Qwen-Max, GPT-4o |
| Screenshot | 1920×1080 原图 | CogAgent, GPT-4V, Gemini-ProV, Gemini-Pro-1.5, Claude-3-Opus, GPT-4o |
| Screenshot + A11y | 两路输入 | CogAgent, GPT-4V, Gemini-ProV, Gemini-Pro-1.5, Claude-3-Opus, GPT-4o |
| Set-of-Mark (SoM) | 标注了数字编号 bounding box 的截图 + 元数据表 | CogAgent, GPT-4V, Gemini-ProV, Gemini-Pro-1.5, Claude-3-Opus, GPT-4o |

关键工程细节：
- a11y tree 经过**过滤**（paper Table 13）：保留 tag 为 `document/item/button/heading/label/scrollbar/searchbox/textbox/link/...` 的节点，要求 showing=True、visible=True、enabled=True、有 name/text/image、position≥0、size>0。即使这样，单次观测的 a11y tree token 数中位数 ~6000（Figure 6）。
- 用 **chat mode**（user/assistant 交替）而不是 few-shot (obs, action) pair，因为后者 success rate 只有 2.79%。
- 保留最近 3 轮 (obs, action) 历史。
- temperature=1.0, top-p=0.9, max tokens 1500, max steps 15, max time 30 分钟。
- SoM 的实现：用 a11y tree 的 bounding box 给截图打标，并附上 `[tag, name, text]` 元数据。模型输出 `tag_N` 而非坐标。

### 4.2 主实验结果

Paper Table 5 的关键数字（按 task 类别）：

| Input | Model | OS | Office | Daily | Prof. | Workflow | Overall |
|---|---|---|---|---|---|---|---|
| A11y | GPT-4 | 20.83 | 3.58 | 25.64 | 26.53 | 2.97 | **12.24** |
| A11y | Qwen-Max | 29.17 | 3.58 | 8.36 | 10.20 | 2.61 | 6.87 |
| A11y | GPT-4o | 20.83 | 6.99 | 16.81 | 16.33 | 7.56 | 11.36 |
| Screen | GPT-4V | 12.50 | 1.86 | 7.58 | 4.08 | 6.04 | 5.26 |
| Screen | Gemini-ProV | 8.33 | 3.58 | 6.55 | 16.33 | 2.08 | 5.80 |
| Screen+A11y | GPT-4V | 16.66 | 6.99 | 24.50 | 18.37 | 4.64 | 12.17 |
| Screen+A11y | GPT-4o | 41.67 | 6.16 | 12.33 | 14.29 | 7.46 | 11.21 |
| SoM | GPT-4V | 8.33 | 8.55 | 22.84 | 14.28 | 6.57 | 11.77 |
| SoM | Gemini-Pro-1.5 | 16.67 | 5.13 | 12.96 | 10.20 | 3.60 | 7.79 |
| — | Human | 75.00 | 71.79 | 70.51 | 73.47 | 73.27 | **72.36** |

几个值得 build intuition 的观察：

1. **A11y tree > Screenshot**：纯截图只有 5.26%-5.80%，加 a11y tree 翻倍到 12.17%。说明当前 VLM 的**像素级 grounding** 还很弱，需要结构化信号辅助。但 a11y tree 在不同软件上质量参差，论文未来工作章节专门提到这点。

2. **SoM 反直觉**：在 WebArena 系工作中 SoM 是 SOTA 方法，但在 OSWORLD 上 GPT-4V 用 SoM (11.77%) 反而**低于** Screen+A11y (12.17%)。Paper 给的解释：OS 分辨率高、元素多（spreadsheet 单元格动辄上百），bounding box 噪声大；很多任务需要坐标级精确操作，bounding box 表达力不够。这个发现对 Andrej 你来说应该很有共鸣——SoM 是从 prompting 角度"作弊"，但 GUI 任务的 grounding 难度本身高于 image segmentation 任务。

3. **Workflow 任务最差**：所有方法都在 0%-7.56% 之间。多应用协作需要**长程规划 + 状态管理 + 跨界面切换**，是当前 agent 的真正瓶颈。

4. **Claude-3 Opus 反常**：尽管在 GSM8K/HumanEval 上和 GPT-4 持平甚至更强，但在 OSWORLD 上 Claude-3 Opus 在所有设置下都明显落后 GPT-4V 2.84%-7.76%。Paper §5.4 给出原因：Claude-3 **planning 强但 grounding 弱**——会"把双击当选择而非打开"、"把 Calc 的 B 列当成 C 列"、"在 VS Code 替换框里输入文本但没点全局替换"。这是非常具体的 grounding hallucination。

5. **跨 OS 相关性 0.7**：在 Ubuntu 上 SR=4.88%，迁移到 Windows 上 SR=2.55%（任务略改），相关系数 0.7。说明 agent 的能力**主要跟随任务而非 OS**，方法可以跨平台迁移。这对未来训练通用 computer agent 是好消息。

## 五、深入分析

### 5.1 任务难度维度

Paper Table 6 按**人类完成时间**分难度（10% 子集上测）：

| 子集 | 占比 | SR |
|---|---|---|
| Easy (<60s) | 28.72% | 16.78% |
| Medium (60-180s) | 40.11% | 13.12% |
| Hard (>180s) | 30.17% | 4.59% |
| Infeasible | 8.13% | 16.67% |
| Feasible | 91.87% | 13.34% |
| Single-App | 72.63% | 13.74% |
| Multi-App | 27.37% | 6.57% |

人类在 Hard 上 49.57%，agent 4.59%——差距 10x。**Infeasible 上 agent 反而略高**（16.67% vs 13.34%），这是危险的：agent 在不该尝试的任务上没好好试，反而过早放弃输出 `FAIL`。这暴露了 **agent 不知道自己什么时候在错**。

### 5.2 多模态观测的影响

#### 5.2.1 分辨率

Paper Figure 5 用 down-sampling ratio 0.2/0.4/0.6/0.8 做了 ablation。**纯截图**：分辨率越高 SR 越高，单调递增。**SoM**：反而在 0.4（768×432）出现 sweet spot，再低（0.2）会大幅掉。

我的解读：SoM 依赖 bounding box 标注，分辨率太低时 box 会挤在一起重叠，反而劣化；纯截图则是直接给 VLM 像素，越多像素越能定位。

#### 5.2.2 历史长度

Paper Figure 7：SoM 设置下 history 长度 1→2→3→all 单调提升 SR。**a11y tree 的文本历史有效**，因为 LLM 擅长从文本抽信息。但**截图历史无效**——多塞历史截图反而没用。这说明**当前 VLM 还不能从图像序列中提取 robust 上下文**，这是重要的未来工作方向。

#### 5.2.3 窗口扰动鲁棒性

Paper §5.2 取了 28 个 SR=50.79% 的"擅长"任务，引入三种扰动：
1. 改变窗口位置；
2. 窗口最小化；
3. 打开无关软件并最大化。

结果 SR 掉 60%-80%。Agent 能切窗但**不会主动把窗口最大化作为中间步骤**，缺乏完整的窗口管理策略。这让人想到 robotics 里的 "perturbation robustness" 问题——sim-to-real gap 在 GUI 上同样存在。

### 5.3 错误类型

Paper §5.4 分析了 550 个失败样本：

- **Mouse click inaccuracies（75%+）**：最常见。Agent 在 code comment 里规划很清楚，但坐标算错。这是**planning 强、execution 弱**的典型表现。
- **Repetitive clicks**：点错 → 调整 → 还错 → 重复，消耗步数。
- **Environmental noise dilemma**：误点导致弹窗、广告、无关应用打开，agent 不知道怎么回去。
- **Lack of prior software knowledge**：比如 GIMP 的 brightness 调整在哪个菜单，agent 不知道，随机试到 max steps。
- **Misinterpretation of instructions**：把"center align the heading"理解成各种奇怪操作。
- **Visual oversight**：忽略重要 UI 元素。

### 5.4 Agent vs 人类难度错位

Paper §5.4 + Figure 19 给出了非常有趣的"难度倒置"现象：

**人类 > Agent 的任务**：
- "erase all highlighted marks in document"：需要选中文字、点 highlight color button、选 no color、save。Agent 在第二步就卡住。
- "bold the title font and add note on page 2"：涉及 GUI 选择 + 切换页面，agent 不理解 GUI logic。

**Agent > 人类的任务**：
- "monitor CPU for 30s and output to txt"：直接 `sar 1 30 > file.txt` 一行命令，agent 知道 CLI 知识，人类反而要查文档。
- "force close LibreOffice"：`killall soffice.bin` 或 `pkill -9 soffice.bin`，agent 一句话搞定。
- "use GIMP to cut 2s-4s of video"：agent 直接 `ffmpeg -ss 00:00:02 -to 00:00:04 -i in.mp4 -c copy out.mp4`——**但它忽略了 instruction 里 "use GIMP" 的要求**！这是 instruction following 与捷径执行的冲突，是个隐患。

### 5.5 Claude-3 vs GPT-4V 深度对比

Paper §5.4 + Figure 20 给了三个 Claude-3 失败案例：

1. **"find secret.docx and copy path to clipboard"**：Claude 双击文件以为打开了，实际只是选中；然后 `Ctrl+C` 复制的是文件不是路径。
2. **"copy B6 in Calc and search in Chrome"**：Claude 没切到 Calc 就以为复制了，在 Chrome 地址栏粘贴空内容，然后**幻觉输出 "Found the answer. Channel 31 in Hong Kong is RTHK TV 31."**——这是非常严重的 hallucination，在没看到搜索结果时就编答案。
3. **"change all 'text' to 'test'"**：Claude 在 VS Code 替换框输入了 text/test 但没点 "Replace All"，以为完成了。

GPT-4V 也会犯 grounding 错，但更"诚实"——会卡住、会重复尝试，而不是编造完成。这暗示**VLM 的 grounding 幻觉比 action 幻觉更危险**，因为后者会被环境反馈纠正，前者会让 agent 进入错误信念状态。

## 六、对未来研究的启示

Paper §7 + 我的思考：

### 6.1 VLM 能力需要补的几个洞

1. **GUI Grounding**：当前 VLM 像素级定位能力差，需要专门的 grounding pre-training（SeeClick [8] 已经在做）。可能需要类似 RT-2 [7] 的 vision-language-action 联合训练，把"看到按钮 → 输出坐标"作为预训练任务。
2. **高分辨率处理**：1920×1080 对当前 VLM 是过载输入。需要 native high-resolution（CogAgent [17] 已经在尝试）、或动态裁剪 + 注意力机制。
3. **图像序列理解**：当前 VLM 对图像历史的利用几乎为零，需要 image trajectory encoding 的训练范式。

### 6.2 Agent 架构方向

1. **Memory & Reflection**：当前 3 轮历史是粗暴截断。需要层次化 memory（短期 working memory + 长期 episodic memory + semantic knowledge），并支持 reflection 触发。
2. **Exploration 策略**：当前 agent 一旦点错就死循环。需要"试探 → 回退 → 重规划"的元策略，类似 Tree of Thoughts 在 GUI 上的扩展。
3. **Subgoal decomposition**：Workflow 任务 SR 只有 6.57%，需要显式的 task decomposition 模块，把"用 VLC 提取视频帧 → 用 GIMP 拼接成 GIF"分解成两个 subagent。

### 6.3 评估与安全

1. **副作用评估**：当前 evaluation 只看任务完成，不看 agent 是否动了不该动的东西（删除文件、改系统设置、发邮件）。需要一个"副作用审计"层。
2. **CAPTCHA 绕过风险**：Paper §7 引用 [42] 提到，未来 agent 可能被用来绕 CAPTCHA。这是 dual-use 风险，需要前置 safety constraint。
3. **a11y tree 噪声处理**：不同软件 a11y 质量参差，需要智能过滤机制。

## 七、与 Andrej 你早期工作的关联

MiniWoB++ [44] 是你 2017 年的工作，那时还是 DOM-based action space（click(type, ref)）。从 MiniWoB++ 到 OSWORLD 的 7 年间，整个领域的轨迹是：

1. **DOM-based action space → raw screenshot + pixel coordinates**：从结构化输入退化到像素输入，看似倒退，其实是把 grounding 难度推给模型，逼 VLM 学会真正的视觉理解。
2. **单 mini task → 真实 OS 任务**：从 100+ 个 5 秒 mini task 到 369 个跨应用 workflow task，复杂度上去了 2 个数量级。
3. **RL-trained policy → LLM-as-agent**：从 PPO 训练专门 policy 到直接用 GPT-4 zero-shot。RL 在 OSWORLD 这种稀疏奖励、长 horizon 环境下太难，LLM 的 in-context learning 反而更 scalable。
4. **MiniWoB++ 的 SoM 思想**：你那篇 paper 已经在 DOM 上用 ref 编号，OSWORLD 的 SoM 是把这个思想迁移到视觉上，但效果反而不理想——这说明**视觉 grounding 难度本质上高于 DOM grounding**。

最近 SeeClick [8]、ShowUI、UFO [59]、AppAgent [60] 等工作都在补这个洞，但离人类 72.36% 还差 60 个百分点。

## 八、值得跟进的相关工作

- **SeeClick** (Cheng et al. 2024): https://arxiv.org/abs/2401.10935 —— GUI grounding 专门训练
- **CogAgent** (Hong et al. 2023): https://arxiv.org/abs/2312.08914 —— high-resolution VLM for GUI
- **UFO** (Zhang et al. 2024): https://arxiv.org/abs/2402.07939 —— Windows UI agent with SoM
- **AppAgent** (Zhang et al. 2023): https://arxiv.org/abs/2312.13771 —— 移动端 multimodal agent
- **WebArena** (Zhou et al. 2023): https://arxiv.org/abs/2307.13854 —— 真实 web 环境 benchmark
- **VisualWebArena** (Koh et al. 2024): https://arxiv.org/abs/2401.13649 —— 多模态 WebArena
- **GAIA** (Mialon et al. 2023): https://arxiv.org/abs/2311.12983 —— 通用 AI assistant benchmark
- **Adept ACT-1**: https://www.adept.ai/act —— 早期 computer use agent
- **Anthropic Computer Use**: https://www.anthropic.com/news/3-5-models-and-computer-use —— Claude 3.5 Sonnet 的 computer use 功能（OSWORLD 之后的重要跟进）
- **OmniParser** (Microsoft): https://github.com/microsoft/OmniParser —— 改进的 SoM 方法
- **OpenAgents** (Xie et al. 2023): https://arxiv.org/abs/2310.10634 —— OSWORLD 一作的早期工作

## 九、总结

OSWORLD 的价值在于它**诚实地暴露了当前 agent 的真实水平**：12.24% vs 人类 72.36%，差距 6 倍。它不是一个让模型刷分的 benchmark，而是一面照妖镜，照出 VLM 在 grounding、planning、exploration、knowledge 四个维度上的全面短板。

它的工程贡献（VM-based env、134 个 evaluation function、302 个 initial state）让它成为一个**可扩展的平台**，而不只是一个 frozen dataset。后续工作可以在此基础上加任务、加 OS、加 evaluation function，这正是 benchmark 应有的生命力。

对你 Andrej 来说，这篇 paper 最重要的 take-away 可能是：**视觉 grounding 是当前 multimodal agent 的真正瓶颈**。DOM 时代我们可以用 ref id 把 grounding 难度外包给浏览器，pixel 时代这条路走不通了，必须让 VLM 真的"看懂"屏幕。这既是一个难题，也是一个巨大的机会——谁先把 GUI grounding 做到人类水平，谁就拿到了通用 computer agent 的钥匙。
