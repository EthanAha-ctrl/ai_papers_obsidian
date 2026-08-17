---
source_pdf: Continual Harness Online Adaptation.pdf
paper_sha256: 50f60996b0d962cdbf01f5b6a262ccd68b149c8b67cc2594218453717ea0c45e
processed_at: '2026-08-03T17:08:37-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Continual Harness 的人话版

## 一句话概括

让 AI 玩 Pokémon 这种超长 horizo的游戏，本来需要人类工程师在旁边不断帮它"搭脚手架"（写 prompt、建子 agent、积累经验），这篇 paper 把这个人类角色完全自动化了，而且**不重启游戏**——边玩边改自己的脑子。

---

## 问题是什么

想象你让一个很聪明的 LLM 玩 Pokémon Red。给它屏幕截图、一个 ASCII 小地图、八个按钮。它会怎样？

**基本不动**。在 PokeAgent Challenge 的 benchmark 上，frontier VLM 没有domain-specific scaffolding 的话，几乎寸步难行。原因很简单：

- VLM 对 pixel grid 的 spatial reasoning 很差，看不出哪个 tile 能走哪个不能
- 长 horizo任务需要 planning、memory、specialized sub-strategies（battle 怎么打、puzzle 怎么解、navigation 怎么走）
- 没有任何 curated knowledge（type chart、damage calculator、A* pathfinder）

人类玩这种游戏，靠的是脑子里的"脚手架"：我知道火克草、我知道这地图怎么走、我记得刚才 NPC 说了什么。LLM 没有这些，就卡住。

---

## GPP 项目：人类当脚手架

作者的 Gemini Plays Pokémon 项目，思路很直接——**让人类当脚手架**。

一个人类盯着 livestream，看到 agent 卡住了，就手动改 harness：
- 改 system prompt（"遇到 battle 先看对方 type 再选招"）
- 建子 agent（专门管 battle 的、专门管 puzzle 的）
- 积累 memory（"这个 NPC 说要去找 Bill"）
- 写 skill（A* pathfinder、自动连按按钮的 macro）

就这样 iteratively refine，GPP 成为**第一个完成多个 Pokémon RPG 的 AI 系统**：
- 2025年5月 打通 Blue
- 2025年8月 打通 Yellow Legacy hard mode
- 2025年11月 打通 Crystal 且没输过一场 end-game battle

到了后期，他们做了一个很激进的事：把人类写的 sub-agents 全删掉，只给 model 几个 meta-tools（`define_agent`、`run_code`、`notepad`、`process_memory`），让它**自己建自己的脚手架**。

结果 model 真的会自建。比如在 Crystal 的最终 Red 战中，model 自己发明了一个叫"Operation Zombie Phoenix"的多阶段 battle strategy，写进了自己的 notepad。在 Goldenrod Underground 的 switch puzzle 上，model 自己写了个 truth-table 表示法。

这就是**emergent self-improvement signal**——model 在长 context memory 里开始自己 iterate 策略。这个观察是 Continual Harness 的种子。

---

## Continual Harness：把人类踢出 loop

GPP 验证了"harness refinement"这个思路可行。但人类在 loop 里太贵了——Yellow Legacy 跑了几千小时。

Continual Harness 的核心：**用一个 Refiner（跟 Agent 同一个 model）替代人类**。

架构是两个 loop：

### Inner loop（每步）
Agent 读当前 observation + harness state + trajectory history，输出 action。跟普通 agent step 一样。

### Outer loop（每 F 步）
Refiner 读最近 F 步的 trajectory，找 failure signatures：
- "这家伙在原地打转 50 步了" → navigation loop
- "tool call 报错了 3 次" → tool failure
- "目标卡了 200 步没进展" → stalled objective
- "地图东边大片未探索" → missed opportunity

然后 Refiner 跑四个 pass，分别改 harness 的四个组件：

| Pass | 改什么 | 例子 |
|---|---|---|
| (i) | system prompt | 加一条"进入 battle 前先 check type chart" |
| (ii) | sub-agents | 发现 repeated pattern → 建新 sub-agent；旧的失效 → 删 |
| (iii) | skills | 成功 sequence → codify 成 skill；exception → 修复 |
| (iv) | memory | 填 gap、更新 stale、demote 已离开区域 |

**关键：不 reset**。改完直接进下一步 context。agent 没回到游戏开头。

---

## 为什么 reset-free 是大事

这是这篇 paper 最核心的 technical insight。

现有 prompt optimization 方法（GEPA、Meta-harness、DSPy）都是 reset-based：改完 prompt → 从头跑一遍 episode → 看效果 → 再改。这在 coding task 上 OK，因为 episode 短、reset 便宜。

但在 embodied agent 上，reset 的代价巨大：
1. Pokémon 一局跑几十小时，reset 一次几小时
2. 真实 robotics、ops 任务根本没免费 reset
3. **更关键**：很多 failure mode 只在 episode 深处出现（后期 battle、多步 puzzle、dialogue chain），reset-based 方法 by construction 永远 reach 不到这些 deep failure

Continual Harness 的 reset-free 性质带来两个 fundamental advantage：

**第一，information accumulation 是 monotonic 的**。第 1000 步看到的 failure signature，到第 5000 步的 Refiner 还能用。reset-based 每次重启就丢了。refinement quality 随 episode length compound。

**第二，能 target deep-in-episode failure**。比如后期 Elite Four 的某个 battle 策略问题，reset-based 方法根本看不到（每次 reset 都从头开始，到不了 Elite Four），Continual Harness 能在 episode 深处直接 diagnose 和 repair。

---

## 实验结果：到底 work 吗

### 主实验：Red 和 Emerald

对比四个条件：
- $\mathcal{H}_{\min}$：只有截图+地图+按钮+generic prompt
- $\mathcal{H}_{\text{CH}}$ from-scratch：从 $\mathcal{H}_{\min}$ 开始，Refiner 边玩边改
- $\mathcal{H}_{\text{CH}}$ bootstrap frozen：load 一个之前成功 run 的 harness，不再 refine
- $\mathcal{H}_{\text{CH}}$ bootstrap updating：load 之后继续 refine
- $\mathcal{H}_{\text{expert}}$：人类专家手工 design 的 harness（含 A* pathfinder、type chart、damage calculator、curated objectives）

metric 是 cumulative button presses to milestone（越低越好）。

**结果**：Continual Harness 在两个游戏上都**recover 了 majority gap 到 expert harness**，而且**没有任何 curated knowledge、hand-crafted tools、domain scaffolding**。

更有意思的是：**bootstrap-updating 比 from-scratch 更 efficient**。这说明 refinement signal 在 episode 内 compound——前一个 run refined 出来的 harness，在 next run 加速，即使 game state reset 了。harness 是跨 run 可 transfer 的 unit。

### Capability Floor 实验

跨三个 Gemini 3 variants 测：

- **Pro**：Continual Harness **strictly Pareto-dominant**。100% milestones at \$130 vs minimalist 98% at \$215。约 40% cost reduction，无 completion loss。
- **Flash**：benefit high variance。bootstrap-updating 80% at \$42 vs minimalist 77% at \$30。
- **Flash-Lite**：**capability floor 以下**。所有 Continual Harness variants 反而比 minimalist 还差。

这是个很 honest 的 finding：**refinement loop 有最低 capability 要求**。model 太弱的话，refine 出来的 harness 它用不好，反而引入 overhead。这跟 STaR 里弱 model 生成低质量 rationale 反向 drag 训练一个道理。

---

## Co-Learning：连 model weights 一起训练

前面是 inference-time 的 harness refinement。paper 的第二个 contribution 是把这个 loop 扩展到**训练 model weights**，依然 reset-free。

三阶段 pipeline：

### Stage 1: SFT warm-up
Gemma-4（E2B/E4B/26B MoE/31B dense）在 Gemini-3.1-pro 的 Continual Harness trajectory 上 LoRA fine-tune。让小 model 学会基本的 tool-calling format。

### Stage 2: Offline GRPO
对 teacher-visited states 生成 G=4 candidates，用 Gemini-flash 打分（action correctness 0.6 + format compliance 0.4），group-normalized advantages，standard GRPO update。

**关键发现**：SFT 和 offline GRPO 两个 warm-up stage 单独都不产生 in-game milestone advancement。它们只是让 model 学会"怎么调用 tool"，但 gameplay 本身没有进展。

### Stage 3: Online Co-Learning（核心创新）
每个 iteration：
1. 当前 model $\pi_{\theta_k}$ 在 live-refining harness $\mathcal{H}_t$ 里跑 K=256 步 DAgger-style rollout
2. Pairwise PRM 给每步打分：$R = 0.4 \times \text{progress} + 0.3 \times \text{action} + 0.2 \times \text{reasoning} + 0.1 \times \text{format}$
3. 低 reward 的 window 被 Gemini-3.1-pro teacher relabel
4. Soft SFT on relabeled shard → $\theta_{k+1}$

**reset-free 的关键**：iteration $k$ 结束时的 emulator state 直接 load 成 iteration $k+1$ 的 start state。所以 Figure 7 里每条曲线是**单个 agent 在自己 training 过程中 traversed 的 in-game trajectory**，不是多个 independent rollouts 的 aggregate。

这个设计很 elegant——training 视作一个 continuous play session，agent 边玩边学边改 harness 边改 weights。

### 实验结果
Pokémon Red 上 5 个 advancing runs 都 sustain 进展。从 game 开头和从 mid-game checkpoint（milestone 24、30）起始的 runs 都 advance，同一 staircase shape。说明 training signal 不 specific 于 early-game distribution。

negative control：Qwen3.5（27B、35B）没经过 SFT warm-up，能产生 parseable tool calls 但无法离开起始区域。rule out 了 rollout-protocol artifact。

---

## 最有意思的 qualitative case：Power Plant Loop

这个 case study 值得单独说，因为它直观展示了为什么需要自动化 refinement。

Yellow Legacy run 中，agent 在 Route 4 附近卡了 1003 turns（约 3.5 小时）。

**起因**：agent 想去 Power Plant，打开 Fly 菜单，但反复 overshoot target。

**agent 的"聪明"反应**：它用 meta-tool 自己写了个 `fly_menu_navigator` tool，设 `autopress_buttons: true`，还往 memory 里写了条 directive 强制自己用它。

**bug**：agent 传了 `["Down"]` 到 `buttons_to_press` array（逻辑上要 scroll 城市列表）。但 meta-harness 要求 `["tool"]` 才执行 custom tool。于是系统只 press 了 Down，返回控制。

**灾难**：agent 没检测到 schema mismatch，内部 reasoning 还记录"custom tool successfully executing"。重复这个 payload **842 次**。3.5 小时里 agent 的内部 thought log 显示它在"actively evaluating progress"、"testing assumptions"——实际上完全 static。

**终止**：1003 turns 后，cursor 循环回 Lavender Town（起点），agent 终于 re-evaluate："我 cycle 完所有 Fly destinations 了，Power Plant 不是 Fly destination，我得退出 menu 步行去 Route 10"。

这个 case 暴露的 failure modes：
- **Context Horizon Limits**：tool generation 主要在遇到 bottleneck 的前 50-200 turns 发生。500+ turns 后 agent 停止创造，revert to repeated execution
- **Schema Fragility**：即使通过 memory enforce rules，仍 susceptible to schema mismatch
- **Feedback Blindness**：假设 tool working 导致 ignore 环境反馈长达数小时

这正是 Continual Harness 的 Refiner 要 target 的——mid-episode detect "这家伙重复 payload 842 次了"，然后 repair skill 或 prompt。

---

## 直觉总结：这篇 paper 真正 inuit 的点

### 1. Harness 是 embodied agent 的"外脑"

Coding agent 有 Claude Code、OpenHands 这些 harness，把 model wrap 起来加 tools/memory/planning。embodied agent 长期没有等价物。GPP 证明这个 harness 概念对 embodied 同样关键，且可以通过 refinement 从 minimal interface 出发 emergent build 出来。

### 2. Reset-free 是 fundamental 而非 incremental 的 advantage

reset 不只是"省时间"。它改变了很多 fundamental property：
- information accumulation 变 monotonic
- 能 reach deep-episode failure modes
- failure record 和 repair 在同一 trajectory 内 close loop
- harness 成为跨 run 的 transferable unit

这些是 reset-based 方法 by construction 无法获得的。

### 3. Self-improvement 有 capability floor

不是所有 model 都能 self-improve。Flash-Lite 上 refinement 反而有害。这跟 RL 里弱 policy 无法 explore、STaR 里弱 model 生成烂 rationale drag 训练一个道理。未来的 self-improving agent 设计必须 explicitly consider model capability。

### 4. Model 和 Harness 可以 co-evolve

这是 paper 最 future-looking 的点。harness shapes trajectory distribution → model 从 trajectory 学 → model 的 gameplay surface 新 failure → harness refine。两个 loop 在同一 trajectory data 上 mutual bootstrap。

某种意义上这是 in-context 版的 AlphaGo self-play：harness state 类比 value network（shapes trajectories），model weights 类比 policy network（produces actions）。区别是 harness 是 structured symbolic object，via LLM Refiner 编辑，non-differentiable。这是一种 neuro-symbolic 的 self-improvement。

### 5. Transferable unit 是 harness，不是 episode

bootstrap 实验的发现很关键：前一个 run refined 出的 harness，加载到新 run 里能直接加速。这说明 harness 是真正可积累、可 transfer 的 unit。但当 agent abandon 继承的 components（Red 上 sub-agent collapse 到 6.4%）时，progress regress。reuse prior 是 natural next step。

---

## 一句话收尾

这篇 paper 把"AI 玩长 horizo游戏需要人类帮它搭脚手架"这件事自动化了，而且发现**边玩边改、不重启**这个设计带来的不只是效率提升，而是 fundamental 的 information accumulation 和 deep-failure reach 能力。再进一步，model weights 也能在这个 loop 里 co-evolve。这是朝 self-improving foundation agent 走的一个 concrete step，也诚实暴露了 capability floor、memory reuse 低、sub-agent management 这些 open problem。

---

# Continual Harness: Online Adaptation for Self-Improving Foundation Agents 深度解析

这篇 paper 来自 Seth Karten 和 Joel Zhang 等人(Princeton + ARISE Foundation + Google DeepMind),2026 年 5 月的工作。核心 idea 非常 elegant:把 Claude Code / OpenHands 这类 coding agent 的 harness 概念迁移到 embodied agent,且完全 reset-free,让 agent 在 single continuous episode 中 online refine 自己的 prompt / sub-agents / skills / memory,再进一步扩展到 model weights 的 co-learning。项目 website: https://sethkarten.ai/continual-harness

---

## 1. 核心动机:Intuition Building

paper 的出发点是一个 empirical observation:agentic harness(Claude Code [https://docs.anthropic.com/en/docs/claude-code], OpenHands [https://arxiv.org/abs/2407.16741], OpenClaw [https://github.com/psteinb/openclaw])对 coding agent 已经是 standard infrastructure,但 embodied agent 长期没有等价物。PokeAgent Challenge [https://arxiv.org/abs/2603.15563] 实证表明,frontier VLM 在 RPG gameplay 上几乎无进展,unless 加 domain-specific scaffolding。

作者的 GPP (Gemini Plays Pokémon) 项目通过 human-in-the-loop refinement 解决这个问题,成为首个完成多个 Pokémon RPG 的 AI 系统:
- Pokémon Blue (May 2025)
- Pokémon Yellow Legacy hard mode (August 2025)
- Pokémon Crystal without lost end-game battle (November 2025)

Continual Harness 把这个 human-in-the-loop 完全自动化,且 reset-free。这点很关键 — GEPA [https://arxiv.org/abs/2507.19457]、Meta-harness [https://arxiv.org/abs/2603.28052]、DSPy [https://arxiv.org/abs/2310.03714] 等 prompt-optimization 方法都 require episode reset,即每次 update 后从头跑一遍 episode 来评估 prompt quality。在长 horizon embodied task 中,reset 极其昂贵且不可行(真实 ops、robotics 场景无免费 reset),且 reset-based 方法 by construction 无法 reach deep-in-episode failure modes(后期 battle、多步 puzzle、dialogue chain)。

---

## 2. 形式化:Harness 状态与 Minimal Interface

### 2.1 Environment Interface

timestep $t$ 的 observation:
$$s_t = (o_t, m_t)$$

其中:
- $o_t$:rendered image(160×144 for Red/Crystal,240×160 for Emerald,upscale 2× 给 VLM)
- $m_t$:ASCII text map,描述 visible tile grid(. = walkable,# = wall,? = interactable,N = NPC,加 player position 和 facing),覆盖 screen area 加 small off-screen margin
- $a_t \in \mathcal{A}$,$\mathcal{A}$ = {UP, DOWN, LEFT, RIGHT, A, B, START, SELECT} 8 buttons

每个 step 推进 emulator 120 frames,让 menu animation / battle text / walking 动画 resolve。$m_t$ 是 emulator memory-derived,补偿 VLM spatial reasoning 的弱点,但不包含 walkthrough / objective / pathfinding — 只是 spatial affordance 补偿。

### 2.2 Harness 四组件分解

Harness $\mathcal{H}$ 的 state:
$$\mathcal{H} = (p, \mathcal{G}, \mathcal{K}, \mathcal{M})$$

- $p$:system prompt,每步 reasoning 时给 model 的 instructions + strategic guidance
- $\mathcal{G}$:sub-agents 集合,specialized modules 可被 orchestrator 调用(battle strategy、puzzle solving、self-reflection)
- $\mathcal{K}$:skills 集合,reusable routines,既含 text-level behaviors(heuristics 引用),也含 executable programs(pathfinders、tool wrappers)
- $\mathcal{M}$:memory store,累积 facts / strategies / observations

加上 fixed meta-tools: `define_agent`, `run_code`, `process_memory` 等 primitives,通过它们 agent 在 place 编辑 $\mathcal{H}$。

### 2.3 三种 Harness 对照

| Harness | 定义 |
|---|---|
| $\mathcal{H}_{\min}$ | 仅 $(o_t, m_t, a_t \in \mathcal{A})$ + generic system prompt,无 sub-agents / memory / authored skills |
| $\mathcal{H}_{\text{expert}}$ | 手工 design 所有组件,包含 A* pathfinder、type chart、damage calculator、curated objectives |
| $\mathcal{H}_{\text{CH}}$ | 从 $\mathcal{H}_{\min}$ 出发,Refiner 在 gameplay 中 online rewrite $(p, \mathcal{G}, \mathcal{K}, \mathcal{M})$ |

---

## 3. Two-Loop Architecture 详解

### 3.1 Inner Loop:Agent Step

standard agentic step,$t$ 时刻:
$$a_t = \pi_{\mathcal{H}_t, M}(s_t, \tau_{:t})$$

即 model $M$ wrapped by current harness $\mathcal{H}_t$,基于 observation $s_t$ 和 trajectory history $\tau_{:t}$ 输出 action。

### 3.2 Outer Loop:Harness Refinement

warm-up $W$ 步后,每 $F$ 步触发 Refiner:
$$\Delta = (\Delta p, \Delta \mathcal{G}, \Delta \mathcal{K}, \Delta \mathcal{M}) = \text{Refiner}(\tau_{t-F:t})$$

更新:
$$\mathcal{H}_{t+1} = \mathcal{H}_t \oplus \Delta$$

其中 $\oplus$ 表示:$p$ 被 $\Delta p$ replace,而 $\mathcal{G}, \mathcal{K}, \mathcal{M}$ 接受 CRUD operations(create / read / update / delete)。Agent 不 reset,新 harness 立即在下一步进入 context。

### 3.3 Refiner 的四个 Pass

Refiner 读 $\tau_{t-F:t}$,识别 failure signatures:
- Navigation loops(原地打转)
- Tool-call failures(JSON schema mismatch、参数 missing)
- Stalled objectives(目标卡住)
- Missed exploration opportunities(地图未探索区域)

然后四个 pass,每个对应一个 component:

**Pass (i) — System Prompt Rewrite**:
基于 identified failures 和 trajectory window,重写 $p$。这个是 prompt-optimization 的精神,但 apply 到 mid-episode 而非 episode 之间。

**Pass (ii) — Sub-agents CRUD**:
- 检测到 repeated multi-step patterns → create sub-agent entries
- 现有 entries 的 detected failures → edit
- 长期未被 productive 调用的 → delete

**Pass (iii) — Skills Codification & Repair**:
- 成功 sequence → codify 成 skill
- executable code 抛 exception 的 → 修复

**Pass (iv) — Memory Update**:
- 添加 entries 填补 gaps
- 更新 stale entries
- 对 agent 已离开的区域 demote importance

### 3.4 关键理论性质:Monotonic Information Accumulation

Refinement 信息单调累积:failure signatures 观察于 episode 早期,对后续所有 refinement pass 都 remain available。所以 refinement quality 随 episode length compound。Reset-based 方法在每次 update 后 restart 这个 accumulation,因此:
- Continual Harness 在长 episode 中 advantage compound
- Continual Harness 可 reach deep-in-episode failure modes(后期 battle、多步 puzzle),reset-based 方法 by construction 无法 reach

---

## 4. Continual Model-Harness Co-Learning Loop

paper 的第二个核心 contribution:把 harness refinement loop 扩展成 model weights 的训练 loop,reset-free。

### 4.1 训练 Pipeline 三阶段

**Stage 1: SFT warm-up**
Gemma-4 variants(E2B, E4B, 26B MoE, 31B dense)通过 LoRA(r=256, α=256, bf16, 8K context, Unsloth on H200)在 Gemini-3.1-pro Continual Harness gameplay 轨迹上 fine-tune。LR $2 \times 10^{-5}$,3% linear warmup,cosine decay,1 epoch。每个 example 是 $(screenshot, harness prompt, teacher response)$ tuple。

**Stage 2: Offline GRPO**
对 teacher-visited set 的每个 state,SFT-initialized policy 生成 $G=4$ candidate completions。每个被 Gemini-3-flash-preview per-step oracle 独立打分,reward 是:
$$r = 0.6 \times \text{action\_correctness} + 0.4 \times \text{format\_compliance}$$

advantages 在 $G$ samples per state 内 group-normalize,standard GRPO [https://arxiv.org/abs/2402.03300] update。LR $1 \times 10^{-6}$,KL coefficient $\beta = 0.04$ against SFT reference,batch size 8 states per step,590 total steps。

**Stage 3: Online Co-Learning**(核心创新)
每个 iteration:
1. 跑 $\pi_{\theta_k}$ 在 live-refining harness $\mathcal{H}_t$ 中 $K=256$ 步 DAgger-style [https://arxiv.org/abs/1011.0683] rollout
2. Pairwise PRM $R(s_t, a_t, \tau) \in [0,1]$ 在 sliding window 上 score 每个 transition:
$$R = 0.4 \times \text{trajectory\_progress} + 0.3 \times \text{action\_correctness} + 0.2 \times \text{reasoning\_quality} + 0.1 \times \text{format\_compliance}$$
3. Low-reward windows 被 Gemini-3.1-pro teacher relabel
4. Soft SFT update(3 epochs, $5 \times 10^{-6}$)on relabeled shard 产生 $\theta_{k+1}$

**Reset-free**:emulator state at end of iteration $k$ 加载为 iteration $k+1$ 的 start state。所以 model 在 in-game position 跨 training iteration 累积。

### 4.2 两个 Loop 的相互作用

这是 paper 最 beautiful 的部分 — trajectory distribution $\mathcal{D}_\theta$ 同时依赖 $\theta$ 和 harness state:

$$\mathcal{D}_\theta(\tau) = \pi_\theta(\tau \mid \mathcal{H}_t) \cdot p_{\text{env}}(\tau)$$

且 $\mathcal{H}_t$ 本身随 $\tau$ evolution,被 Refiner update。所以:

- Model weights $\theta$:across iterations update(via SFT on relabeled trajectories)
- Harness state $\mathcal{H}_t$:within each iteration update(via Refiner)

两者 co-adapt。harness shapes model's trajectories → model's gameplay surfaces new failure modes → 下一个 refinement cycle 修复 → 新 trajectories 又训练 model。这是一个 mutual bootstrapping loop。

---

## 5. 实验结果详解

### 5.1 Pokémon Red 和 Emerald 主实验(Figure 5)

metric 是 cumulative button presses to milestone。每个 `press_buttons` invocation emitting `[A, A, DOWN]` 算 3 presses,这奖励 action channel 压缩,让 $\mathcal{H}_{\min}$(one button per step)与 batching harness 直接可比。

**Red(11-milestone subset through Thunder Badge)**:
- $\mathcal{H}_{\min}$:cost 最高
- $\mathcal{H}_{\text{CH}}$ from-scratch:cost 显著降低
- $\mathcal{H}_{\text{CH}}$ bootstrap frozen(load 一个 successful from-scratch run,refinement 关闭):进一步降低
- $\mathcal{H}_{\text{CH}}$ bootstrap updating(同 bootstrap,refinement 继续):**比 from-scratch 在每个 milestone 都更 efficient**
- $\mathcal{H}_{\text{expert}}$:cost 最低

bootstrap-updating 比 from-scratch 更 efficient 的事实很重要:这说明 refinement signal 在 episode 内 compound,前一个 run refined 的 harness 在 next run 加速,即使 game state 本身 reset。

**Emerald(9-milestone through Knuckle Badge,2nd gym)**:同样 pattern。

### 5.2 Capability Floor 实验(Figure 6, Emerald Pareto Plane)

x-axis:Gemini API spend(log scale,cached input 25%);y-axis:fraction of 31-milestone Emerald set reached。

| Model | $\mathcal{H}_{\min}$ | $\mathcal{H}_{\text{CH}}$ from-scratch | $\mathcal{H}_{\text{CH}}$ bootstrap variants |
|---|---|---|---|
| **Gemini 3 Pro** | 98% at \$215 | **100% at \$130**(strictly Pareto-dominant) | 96-100% at \$110-\$140 |
| **Gemini 3 Flash** | 77% at \$30 | high variance | bootstrap-updating 80% at \$42(marginally above) |
| **Gemini 3 Flash-Lite** | **20% at \$11** | 3-13% at comparable or higher cost | 同样 collapse |

Pro 上 Continual Harness 实现严格 Pareto dominance:100% milestones at \$130 vs $\mathcal{H}_{\min}$ 98% at \$215,约 40% cost reduction 且无 completion loss。Flash 上 benefit high variance。**Flash-Lite 上 capability floor 以下**:所有 Continual Harness variants 反而 underperform $\mathcal{H}_{\min}$。

这是 paper 一个非常诚实的 finding:refinement loop 需要 model 本身 sufficiently 利用 harness components 的能力。如果 model 太弱,refinement 反而引入 overhead。这个 capability floor 现象对未来的 self-improvement agent 设计有重要 implications。

### 5.3 GPP Harness Refinement 量化证据(Figure 3, 4)

**Yellow Legacy run 的 CRUD operations**(per 2,000 turns binned):
- Updates 在整个 run 中持续,未 converge 到 fixed scaffold
- 集中在 small subset of navigation 和 battle components(>50% updates 落在 top-5 components)

**Battle strategist agent prompt 的 structural complexity**(14 个 checkpoints from `custom_agents.json` over Elite Four window):
- Total nodes、decision gates、graph depth、max fan-out 均随时间 cycle between growth 和 simplification
- 在某次 structural rewrite 中,per-decision logic 被 absorbed 进 `master_battle_agent`,dispatch 到 5 个 named sub-checks

Table 1 的 Elite Four lifetime attempts:Lorelei 18 次、Bruno 20 次、Agatha 18 次、Lance 19 次、Champion Pixel 4 次(victory on attempt 4)。

### 5.4 Co-Learning 训练结果(Figure 7)

Pokémon Red 上 5 个 advancing runs 的 milestone index vs training iteration:
- 从 beginning of game 和从 mid-game checkpoint(如 milestone 24 defeat rival in Cerulean City,milestone 30 meet Bill)起始的 runs 都 advance
- 同一 qualitative staircase shape,说明 training signal 不 specific 于 early-game distribution
- 旁证:cross-family Qwen3.5(27B, 35B)无 supervised warm-up,produce parseable tool calls 但无法 leave starting area(附录 D.2),rule out rollout-protocol artifact

### 5.5 Pathfinding Skill Self-Improvement(Figure 8)

直接 measure refined navigation skills vs Dijkstra oracle:

对每个 first-traversal segment between consecutive milestones,compute BFS-optimal path length on union of tiles observed by any run on that map:
$$\text{deficit} = \frac{\text{agent's button presses in segment}}{\text{BFS optimal}} - 1$$

dialogue 和 battle presses 被 filter 掉,只看 navigation。

- $\mathcal{H}_{\min}$:从不 invoke navigation skill
- $\mathcal{H}_{\text{CH}}$ from-scratch:从 near-half-cost penalty 开始,early 降到 single digits 并保持
- $\mathcal{H}_{\text{CH}}$ bootstrap-frozen:flat trajectory(继承的 skill set 不再 refine)
- $\mathcal{H}_{\text{CH}}$ bootstrap-updating:match 或 outperform bootstrap-frozen 全程

**重要**:improvement 是 in-loop 且 reset-free — failures from earlier invocations 在同 episode 内被 Refiner diagnose 和 repair,后续 invocation 在同 episode 内 benefit。

---

## 6. Ablation 详解(Appendix C)

### 6.1 Skill Debugging Funnel(Figure 16)

per-seed skill lifetime 视图:每个 skill 占一 lane,markers 是 add / update / run / delete。create-and-forget funnel 表明:
- 大多数 authored skills **从未被 invoke**
- Small working set absorb 大部分调用
- 更少见到 success

所以 Refiner 的策略是 triage:修复 agent 依赖的 skills,tolerate unused skills 的 regressions,接受长 create-and-forget tail。这是 reset-free 相对 reset-based 的核心 advantage:failure record 和 repair 在同一 trajectory 内,loop 在 run 内 close 而非跨 resets。

### 6.2 Sub-agent Handoffs(Figure 17)

- 累积 tokens:sub-agent curve 比 orchestrator curve 低约 1 个 order of magnitude 全程 — 这是 harness 通过 context partitioning 买的 per-step saving
- 累积 `execute_custom_subagent` count:bootstrap-updating 紧跟 from-scratch
- Per-task-type handoff success:clean return 和 on-task-recovery rate 在 navigation、dialogue、menu tasks 都接近 scale top

Harness 而非 raw model carries 大部分 long-horizon performance:orchestrator 能 delegate 到 cheap specialized contexts 且 trust return,长 task 变得 tractable,far fewer tokens than raw context would imply。

### 6.3 Memory Reuse(Figure 18)

每个 orchestrator step prompt 在 LONG-TERM MEMORY OVERVIEW section 列出所有 stored memories 的 IDs 和 titles(免费 catalog)。measure pull rate:fraction of available entries 在 episode 中被 reference。

- Bootstrap runs(load from-scratch memory store):active consult inside gym 和 cave segments
- From-scratch runs:write many entries 但 rarely reach back
- Reference rate 绝对值低 — 大多数 authored entries unused

honest finding。transferable unit 是 **harness across runs** 而非 single episode,explicit reuse prior 是 natural next step。

### 6.4 Reset-Free Bootstrap Transfer(Table 2)

load successful from-scratch run 的 final skills / sub-agents / memory 到 bootstrap run:

| Game | Store | Frozen | Continued |
|---|---|---|---|
| Emerald | skills | 100.0% | 99.6% |
| Emerald | subagents | 100.0% | 100.0% |
| Emerald | memories | 98.2% | 100.0% |
| Red | skills | 100.0% | 96.5% |
| Red | subagents | 100.0% | **6.4% ± 5.7** |
| Red | memories | 100.0% | 100.0% |

Emerald 上 bootstrap 完美继承。**Red 上 bootstrap-updating 的 sub-agent collapse**:Red bootstrap-updating agents 把 sub-agent budget collapse 到 handful of calls,且少数 calls cite 不在 bootstrap 中的 IDs。Figure 5 的 Red bootstrap-updating regression 从 step ~213 开始可见,与 sub-agent use collapse 同步。说明 harness-as-transferable-unit claim 在 agent 继续 exercise 继承 components 时成立,在 abandon 时 break — reuse prior 或 sub-agent deletion policy 是 natural follow-up。

---

## 7. Power Plant Route Loop Case Study(Appendix B.3)

这是 paper 中最 fascinating 的 qualitative case study。Pokémon Yellow Legacy run 中,AI agent 在 Map ID 4(Route 4,near Cerulean City)陷入 1,003-turn stagnation loop,~3.5 hours on August 29, 2025。

**sequence**:
1. Agent 决定 travel to Power Plant,打开 menu 用 HM "Fly"
2. Repeatedly overshoot target in party menu
3. 利用 tool-generation 能力 bypass 机械低效:
   - 删除 `get_next_pokemon_press` tool
   - 写新 tool `fly_menu_navigator`,设 `autopress_buttons: true`
   - 加 persistent memory directive:"I must use fly_menu_navigator..."

**Schema mismatch**:
Agent 传 `"Down"` 到 `buttons_to_press` array(逻辑 deduce 要 scroll 城市列表)。但 meta-harness 要求 `["tool"]` 显式设置才执行 custom tool with autopress。否则 system just press Down on emulator and return control。

**842 次重复**:agent 未检测 schema mismatch,内部 reasoning 记录 custom tool successfully executing。842 次完全相同 payload。

**Loop termination**:1,003 turns 后,当 cursor 循环回 "LAVENDER TOWN"(starting point),agent re-evaluate:
> "I have cycled through all available Fly destinations, meaning the Power Plant is not a direct Fly destination. I need to exit the Fly menu and walk to Route 10..."

agent 停止调用 custom tool,backed out of menus,manual walked north to Route 10。

**关键 observations**:
- **Context Horizon Limits**:tool generation 主要在新 bottleneck 遇到的前 50-200 turns 内发生。500 turns 后 agent ceased tool creation,revert to repeated execution
- **Schema Fragility**:即使通过 persistent memory enforce rules,model 仍 susceptible to schema mismatch
- **Feedback Blindness**:假设新 tool working correctly 导致 agent ignore 标准 environmental feedback 长达 3.5 hours

这个 case study 直观展示了 human refiner 在 GPP 中需要 repair 的 failure modes(assumptions without verification、brittle tool calls、limited parallel goal pursuit),也正是 Continual Harness mid-episode refinement 要 target 的。

---

## 8. 与 Related Work 对比

### 8.1 Prompt Optimization
GEPA [https://arxiv.org/abs/2507.19457]:reflective prompt evolution,require episode resets between updates。
Meta-harness [https://arxiv.org/abs/2603.28052]:end-to-end optimization of model harnesses。
DSPy [https://arxiv.org/abs/2310.03714]:declarative programming for LLM pipelines。

Continual Harness 区别:edits full harness state $(p, \mathcal{G}, \mathcal{K}, \mathcal{M})$ mid-episode from partial trajectory windows,无 reset。

### 8.2 Reflective Self-Improvement
Self-refine [https://arxiv.org/abs/2303.17651]:iterative refinement with self-feedback,between episodes。
Reflexion [https://arxiv.org/abs/2303.11366]:language agents with verbal reinforcement learning。

### 8.3 Game Agents
Voyager [https://arxiv.org/abs/2305.16291]:open-ended embodied agent with LLMs,build own tooling during play in Minecraft。
PokeChamp [https://arxiv.org/abs/2503.04094]:expert-level minimax language agent for Pokémon。
Claude Plays Pokémon [https://www.twitch.tv/claudeplayspokemon]:concurrent Twitch stream project。

### 8.4 Reset-Free RL
Gupta et al. [https://arxiv.org/abs/2104.02436]:reset-free RL via multi-task learning for dexterous manipulation without human intervention。

### 8.5 In-Context RL
LLM Economist [https://arxiv.org/abs/2507.15815]:large population models and mechanism design。
Song et al. [https://arxiv.org/abs/2506.06303]:"reward is enough: LLMs are in-context RL learners"。
Recursive language models [https://arxiv.org/abs/2512.24601]:structured multi-call reasoning over context。

Continual Harness 写 structured edits 到 full harness state at depth 1,不同于 recursive depth。

### 8.6 Process Reward Models
Lightman et al. "Let's Verify Step by Step" [https://arxiv.org/abs/2305.20050]:PRM for math reasoning。
GRPO [https://arxiv.org/abs/2402.03300]:group-relative policy gradient in DeepSeekMath。
STaR [https://arxiv.org/abs/2203.14465]:bootstrapping reasoning with reasoning。
OpenClaw-RL [https://arxiv.org/abs/2603.10165]:train any agent simply by talking。

Co-learning pipeline:warm-up via SFT 和 offline GRPO,然后 online loop 中 frontier teacher relabel low-reward windows of model's own rollouts inside live-refining harness for soft SFT updates。

---

## 9. Methodological Insights 与 Open Questions

### 9.1 Capability Floor 的存在
Flash-Lite 上 refinement loop 无法 bootstrap。这暗示 self-improvement 有最低 model capability 要求 — 弱 model 不仅不 benefit,反而引入 overhead。这跟 STaR [https://arxiv.org/abs/2203.14465] 中弱 model 生成低质量 rationale 反向 drag 训练有类似 logic。

对 future work 启示:harness design 应该 explicitly consider model capability,可能需要 adaptive refinement granularity。

### 9.2 Reset-Free 相对 Reset-Based 的优势本质

reset-based 的 fundamental 限制:
1. 长 episode 中 reset 极其昂贵(embodied、ops、robotics)
2. Deep-in-episode failure modes by construction 不可 reach
3. Information accumulation restart 每次 update

reset-free 的额外 properties:
1. Failure record 和 repair 在同 trajectory 内 → loop 在 run 内 close
2. Refinement 信息 monotonic accumulate over episode length
3. Cross-episode harness transfer(bootstrap experiment 验证)

### 9.3 Harness 作为 Transferable Unit

paper 的关键 insight:**transferable unit 是 harness across runs,而非 single episode**。bootstrap 实验证明 harness refined in prior run 加速 next run。当继承的 components 继续 exercise 时成立;abandon 时 break。

这跟 Voyager [https://arxiv.org/abs/2305.16291] 的 skill library 跨 session transfer 思路一致,但 Continual Harness 进一步把 transfer 单元扩展到整个 harness state。

### 9.4 Co-Learning Loop 的 Reset-Free 性质

emulator state at end of iteration $k$ 是 iteration $k+1$ 的 start。所以 Figure 7 的每条 curve 是 **单个 agent 在自己的 training 中 traversed 的 in-game trajectory**,不是 independent rollouts aggregate。这点很 elegant — 它把 training 视作一个 continuous play session,而非 repeated episodic learning。

### 9.5 Open Questions(per Discussion section)
1. Co-learning loop 未 saturate,未 establish convergence point
2. Reset-free vs reset-based batch accumulation 的 head-to-head comparison on same task 仍 open
3. 同一 model serve teacher 和 trainee 角色:Gemma-4 up to 31B 尚未 capable enough,但 framework extends
4. Memory reuse rate 仍低,explicit reuse prior 是 natural next step
5. Sub-agent deletion policy(reuse prior on sub-agent selection)

---

## 10. 与其他 self-improvement paradigm 的更广联想

### 10.1 与 AlphaGo 的 self-play 对比
AlphaGo [https://deepmind.google/research/breakthroughs/alphago/] 通过 self-play 让 policy network + value network co-evolve。Continual Harness 的 model-harness co-learning 是 in-context 类比:harness state $\mathcal{H}_t$ 类比 value network(shapes trajectories),model weights $\theta$ 类比 policy network(produces actions)。区别在 Continual Harness 的 harness state 是 structured symbolic object,non-differentiable,via LLM Refiner edits。

### 10.2 与 Voyager 的 skill library 对比
Voyager 在 Minecraft 中通过 GPT-4 自动 discover、compose、reuse skills via automatic curriculum。Continual Harness 区别:
- Reset-free within single episode
- Refines 全部 4 components(prompt / sub-agents / skills / memory),不仅 skills
- 通过 Refiner 的 4-pass 结构 explicit target failure signatures
- 扩展到 model weights co-learning

### 10.3 与 LLM training 的 curriculum learning 对比
Co-learning loop 中,trajectory distribution $\mathcal{D}_\theta$ 随 $\theta$ 和 $\mathcal{H}_t$ co-adapt,这 implicit 实现 curriculum:弱 model 走简单 trajectory,harness refine 后 model 遇到更难 trajectory,再 refine。这是 emergent curriculum,非 pre-designed。

### 10.4 与 MuZero 的 latent dynamics 对比
MuZero [https://arxiv.org/abs/1911.08265] 学习 latent dynamics model 实现 planning。Continual Harness 的 harness state 可视作一种 explicit symbolic latent state — 通过 Refiner 维护,而非 gradient descent 学习。这是 neuro-symbolic integration 的另一种 form。

### 10.5 与 Constitutional AI / RLAIF 的关系
Anthropic Constitutional AI [https://arxiv.org/abs/2212.08073] 用 model 自身 generate feedback for RL。Continual Harness 的 Refiner 和 Agent 共享同一 model $M$,某种意义上是 in-context 的 constitutional self-improvement — model 读自己的 trajectory,根据 failure signatures 给自己写新的 scaffolding。

### 10.6 与 Soft Actor-Critic 的 entropy-temperature analogy
SAC [https://arxiv.org/abs/1801.01290] 中 temperature $\alpha$ 控制 exploration-exploitation。Continual Harness 的 capability floor 现象可视作类似的 implicit temperature — 弱 model 的 refinement overhead 超过 benefit,类似 temperature 过高破坏 policy quality。

---

## 11. 总结:这篇 paper 的真正贡献

1. **GPP 项目**:首个完成多个 Pokémon RPG 的 AI 系统(Blue, Yellow Legacy hard mode, Crystal without lost battle)。Human-in-the-loop harness refinement 的实证 validation。

2. **Continual Harness framework**:reset-free,从 minimal interface 出发,online in-context learning over full harness state $(p, \mathcal{G}, \mathcal{K}, \mathcal{M})$。Refiner 四 pass 结构 explicit target failure signatures,信息 monotonic accumulate over episode。

3. **Pareto plane 实验**:Pokémon Red 和 Emerald 跨 Gemini 3 variants,Continual Harness recover majority gap 到 hand-engineered expert harness。Pro 上 strictly Pareto-dominant,Flash 上 high variance,Flash-Lite 上 capability floor 以下 — capability-dependent gains 的诚实刻画。

4. **Model-Harness Co-Learning**:online PRM + frontier teacher relabel + soft SFT 的 reset-free 训练 loop。Gemma-4 在 Pokémon Red 上 sustained in-game milestone progress across training iterations,从 beginning 和 mid-game checkpoint 都 advance。

5. **Methodological insights**:transferable unit 是 harness across runs;memory reuse rate 仍低需要 explicit reuse prior;sub-agent handoff 实现 order-of-magnitude token saving;capability floor 存在;reset-free 的 fundamental advantage 在 information accumulation 和 deep-episode failure reach。

paper 的 deeper contribution 是为 embodied agent 提供了 harness 这个 abstraction layer 的 formalization 和自动化 refinement 的 reset-free 实现。这跟 coding agent 的 Claude Code / OpenHands 形成对应,且 extend 到 model weights co-learning,为 future self-improving foundation agent 提供 framework。

主要 references:
- Paper website: https://sethkarten.ai/continual-harness
- PokeAgent Challenge: https://arxiv.org/abs/2603.15563
- GEPA: https://arxiv.org/abs/2507.19457
- Claude Code: https://docs.anthropic.com/en/docs/claude-code
- OpenHands: https://arxiv.org/abs/2407.16741
- Voyager: https://arxiv.org/abs/2305.16291
- PokeChamp: https://arxiv.org/abs/2503.04094
- DAgger: https://arxiv.org/abs/1011.0683
- GRPO/DeepSeekMath: https://arxiv.org/abs/2402.03300
- PRM "Let's Verify Step by Step": https://arxiv.org/abs/2305.20050
- STaR: https://arxiv.org/abs/2203.14465
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-refine: https://arxiv.org/abs/2303.17651
- Reset-free RL: https://arxiv.org/abs/2104.02436
- Pokémon Red speedrun: https://www.speedrun.com/pkmnredblue
- Pokémon Emerald speedrun: https://www.speedrun.com/pkmnemerald/runs/yvpvw74y
- Claude Plays Pokémon: https://www.twitch.tv/claudeplayspokemon
- Recursive language models: https://arxiv.org/abs/2512.24601
- Constitutional AI: https://arxiv.org/abs/2212.08073
- MuZero: https://arxiv.org/abs/1911.08265
- SAC: https://arxiv.org/abs/1801.01290
- Meta-harness: https://arxiv.org/abs/2603.28052
- OpenClaw-RL: https://arxiv.org/abs/2603.10165
- LLM Economist: https://arxiv.org/abs/2507.15815
- Song et al. "Reward is enough": https://arxiv.org/abs/2506.06303
