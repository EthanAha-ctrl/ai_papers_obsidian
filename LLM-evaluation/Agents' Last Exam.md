---
source_pdf: Agents' Last Exam.pdf
paper_sha256: db0bc77073c8906ec1b3612d72714a6f4b08bb8294a0589660186e2b887864a5
processed_at: '2026-07-18T05:40:32-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agents' Last Exam (ALE): 长horizon经济价值workflow的benchmark

下面我尽量把这篇paper的骨架、关键技术决策、和实验信号都拆开讲，目标是帮你 build intuition——为什么这个东西设计成这样、它真正测的是哪类能力、当前 frontier agent 在哪一类 failure 上崩。

Reference: https://agenthle.org / https://github.com/agenthle  / Leaderboard: https://huggingface.co/spaces/agenthle/leaderboard
Paper: https://agenthle.org (Berkeley, Yiyou Sun, Dawn Song 等执行, 250+ industry expert 协作)

---

## 1. 一句话定位 + 核心论点

> AI system 已经把 MMLU / GPQA / HLE / IMO / ICPC 这些 celebrated benchmark 刷穿，但是 **GDP-relevant 的经济 output** 几乎没动。这是一个 **utility problem**，本质是 evaluation problem：缺少对 long-horizon、economically valuable、real-world workflow 的 **sustained performance measurement**。

ALE 想补的就是这个空缺——它故意把"benchmark saturated"和"industry deployment"两件事绑死：**如果 frontier agent 在 ALE 上 saturated，那就意味着它们已经可以承担这些 profession 的 long-horizon tool-intensive workflow**。所以名字叫 "Last Exam"，双重含义：
- **Last as competence threshold**：通过这场 industry exam 意味着 agent 真的能干这行，而不只是会回答关于这行的问题；
- **Last as difficulty frontier**：通过把 evaluation 锚在 authentic long-horizon workflow 上，ALE 卡在当前系统能 reliably 完成的边界。

这种 framing 其实是对过去几年 benchmark-driving-progress 模式的有意识重演——ImageNet 之于 CV，SWE-bench 之于 coding agent——只不过这次锚的是 SOC/O\*NET 职业空间。

---

## 2. 现有 benchmark 的三角 trade-off

Paper 的 motivation 是三个 properties 几乎没法同时满足：

| Property | 谁满足 | 谁牺牲 |
|---|---|---|
| Realism (authentic professional workflow) | GDPval, RLI | 短 task 的 benchmark |
| Breadth (industry coverage) | MMLU, HLE | 单一 domain agentic benchmark |
| Verifiability (no human judge) | SWE-bench, OSWorld, Terminal-Bench | GDPval, RLI |

具体数字 (Table 2，按照 55-subdomain ALE taxonomy 重新映射):

- **MMLU**: ~16K items, 26/55 industries, knowledge QA, seconds, exact match
- **HLE** (Humanity's Last Exam, Nature 2026): ~2.5K, 20/55, knowledge QA, minutes, exact match / LLM judge — ALE 借了这个 naming convention
- **GPQA**: ~500, 8/55, seconds, exact match
- **SWE-bench**: ~2K, 5/55 (OSS Python repos), minutes-hours, unit tests
- **OSWorld**: ~400, 5/55 (desktop apps), minutes, state checks
- **WebArena**: ~800, 4/55 (web apps 6 sites), minutes, state checks
- **Terminal-Bench 2**: ~100, 6/55 (terminal workflows), minutes-hours, state checks — ALE-CLI 子集直接对标这个
- **GAIA**: ~500, general assistant, minutes, exact match
- **GDPval**: ~200, 16/55 (top GDP occupations), hours-days, **human expert judge**
- **RLI** (Remote Labor Index, Mazeika et al. 2025): ~250, 14/55 (Upwork categories), hours-days, **human expert judge**
- **ALE (ours)**: ~1.5K instances / 960 workflows, **55/55**, SOC 2018 + O\*NET, hours-weeks, **deterministic scripts + rubric**

paper 给出的关键数字：**16 个 prior benchmark 的 union 仍 leave 13/55 subdomains entirely uncovered**。这是 ALE 想填的具体 coverage gap。

---

## 3. Benchmark design 的三个 admission criteria

每个 submitted task 必须同时满足：

### 3.1 Representativeness
workflow 要 match 真实 professional practice，必须用 **domain expert 实际会用的 software**。例如建筑专家把 2D blueprint 转 3D，会用 **SolidWorks / Rhino**，不会用 AutoCAD。这一条不是 pedantic——它直接决定 agent 能否被训练/eval 在真正产业里使用的 tool surface 上。

### 3.2 Complexity
区分 **workflow vs action**——这是 paper 反复强调的概念切割。
- Undesired: "Apply a color filter in DaVinci" → 单个 local edit，是 action
- Better: "Move a running cheetah into another race video" → 需要 tracking + rotoscoping + compositing + color matching，是一个 coupled workflow

### 3.3 Verifiability
output 必须 admit deterministic check 或 unambiguous rubric。
- Undesired: "Design an RPG game with monsters" → 没有 objectively checkable target
- Better: "Reproduce the game mota.exe using RPGMaker XP" → map geometry、character attributes、event states 都可以自动比较

---

## 4. Taxonomy 怎么构造出来的：SOC 2018 + O\*NET pipeline

这是 paper 最 "infrastructure-y" 的部分，但很关键，因为它决定了 "55 subdomains" 是有 grounding 而不是 ad hoc 选出来的。

### 4.1 数据 backbone
- **SOC 2018** (U.S. Bureau of Labor Statistics): 23 major groups → 98 minor groups → 459 broad occupations → 867 detailed occupations。这是 occupational classification。
- **O\*NET 30.2** (U.S. Dept of Labor): 给每个 SOC occupation 配上 task statements、work activities、tools & technology records。Reference: https://www.onetcenter.org/database.html

### 4.2 Screening procedure
固定 GPT-4o mini prompt, temperature=0，对 **1,016 entries in O\*NET 30.2 Occupation Data** 筛选。每个 prompt instance 喂入: SOC code + title, O\*NET description, task statements, work activities, technology examples。

筛完之后把 O\*NET variants consolidate 到 shared SOC base codes，剩 **117 unique SOC base codes**。

### 4.3 Subdomain 聚合
按 **field / methodology / work product** 三个维度把 SOC code 聚成 workflow-level subdomain。一个 SOC code 可以落到多个 subdomain（如果它包含 separable workflows）。这一步出 **51 个 SOC-anchored subdomains**。

### 4.4 Frontier extension
SOC 2018 不覆盖新兴 digital workflow。所以补 **4 个 frontier subdomains + 7 个 extensions**，依据是 recent NIH / NSF / field-specific technical roadmap（[NIH strategic plan 2021–2025](https://www.nih.gov/about-nih/nih-wide-strategic-plan), [NSB 2024 indicators](https://ncses.nsf.gov/pubs/nsb20243), [CHIPS and Science Act 2022](https://www.congress.gov/117/plaws/publ167/PLAW-117publ167.pdf), [NIST AI RMF 1.0](https://www.nist.gov/itl/ai-risk-management-framework)）。

### 4.5 最终结构
**55 subdomains / 13 top-level domains**。具体：
- Manufacturing & Industrial Operations（含 CAM、PLC、SPC 等 subdomain）
- Biomolecular Structure & Design（docking、inverse-folding、codon optimization）
- 3D Animation & Interactive Media
- Robotics & Autonomous Systems（URDF、controls、planner）
- Finance、Law、Electrical Engineering、Agriculture/Environment、Visual & Media Arts、Education、Radiology、Audio Production、Cybersecurity 等

### 4.6 Cross-benchmark mapping
为了让 ALE 和 prior benchmark 在同一坐标系下比较，用 **LLM-assisted classifier** 把每个 prior benchmark 的 published categories（subject / application / repo / occupation）映射到 55-subdomain taxonomy 上。Figure 3 的 positioning map 就是这么画出来的。

---

## 5. Task construction pipeline：5 个 gate

![Pipeline](https://agenthle.org) — Figure 4 的五个 gate：

### Gate 1 — Expert Sourcing
通过 advisory committee（含 Patrick Bryant、Carl Boettiger、Tarek Zohdi、George Karniadakis、Jack Gallant 等领域 PI）招募 250+ industry practitioner。这些 practitioner 在日常实践中执行 complex software workflow，而不是 lab curator。

### Gate 2 — Task Submission
portal: https://agents-last-exam.org/submit/new/form  
专家上传 **过去已经做过的、花了几天到几周完成的 past project**。AI-assisted tool 帮 refine proposal 直到 5 个核心 component 完整：
1. natural-language description
2. input files
3. target software
4. expected deliverable
5. evaluation specification

对 source-grounded task 还可以带 public paper、dataset、standard、workflow documentation 作为 provenance record。

### Gate 3 — First-pass Review
会议式 decision: **major / minor revision, borderline accept, accept, strong accept**。任何 revision decision 会 loop 回 expert。

### Gate 4 — Task Implementation
工程团队把 specification 转成 runnable assets + provisioned software container + codified evaluation logic。包含 **engineer dry-run**——如果发现 gap 或 missing dependency，自动 email 通知 expert unblock 开发。

### Gate 5 — Final QC
expert committee peer review，检查三件事：
1. **reference output 是否正确**
2. **evaluation bounds 是否 calibrated**——不能 impossibly narrow，也不能 spuriously permissive
3. **problem context 是否 sufficient**，能否实际到达 final state

### Provenance 数字 (Figure 5)
- **1,490 task instances** 总数
  - **960 外部 submission**（按 first-pass review verdict 分）
  - **530 commissioned**（弥补 coverage gap 主动 commissioned）
- 按 release state：
  - **150 public**（~10%）
  - **1,017 private**
  - **323 unverified pending QC**

### Public/Private release strategy
benchmark contamination 是 long-term 最大的威胁。ALE 把 **private pool** 设为 90%，**rolling evaluation**：private task instance 周期性 rotate 进 public set，retired public task 被 replace，这样在新一代 model 上保持 uncontaminated evaluation surface。Appendix D.1 用 Claude Code + Opus 4.7 跑了 full pool 验证 public subset representativeness：

> Pearson **r = 0.89, p < 0.001**（per-cluster pass rate, public subset vs full pool）

private pool 的 pass rate 普遍更高，因为 Last-Exam tier 全在 public set，private 池更偏向 Near-Term level。

---

## 6. Evaluation pipeline architecture：三 component decoupling

这是 paper 的 Section 3.1 的核心，重点是把 **task specification、agent、environment** 彻底 decouple，使得任何一个可以独立 interchange。

### 6.1 Task Specification (main.py)
一个 executable Python 文件，encapsulating 5 个 construction element，暴露 3 个 lifecycle function：

- **`load()`** — 纯 declarative，返回 structured task object，包含 description、metadata、compute requirements (OS type、hardware specs)。**不建 remote connection，不 modify environment state**。定义 task 是什么。
- **`start()`** — 把 VM 转换成 deterministic starting state。通过 session API 操作 remote desktop：file-system ops、application management、keyboard/mouse、screen capture。
- **`evaluate()`** — 从 remote environment 取 agent 的 output artifact，对照 reference 或 rubric 打分，返回 normalized **score ∈ [0, 1]**。

### 6.2 Agent
harness + foundation model。接收 task description + metadata，进入 action loop：observe (screenshot / shell output / file content) → select action (mouse/keyboard/shell/file edit/API) → execute → repeat until 自己决定 terminate。

### 6.3 Environment — 四目录 layout
```
input/        # read-only，agent 读
software/     # pre-installed apps + deps
output/       # agent 唯一 writable target
reference/    # ground truth，agent 不可见，仅 evaluate() 用
```
这个 layout 是一个 clean contract：agent 从 input/ 读、写到 output/、被 reference/ 评分。

### 6.4 Compute 环境
- GCP VM
- Default: **c4-standard-4** (4 vCPUs, 16 GB RAM)
- 重 GPU 任务（3D rendering、simulation）: **g2-standard-8** + NVIDIA L4 GPU
- 少量 heavy numerical simulation 用高内存或多核配置，由 `load()` 声明

### 6.5 Decoupled design 的两个 guarantee
1. 任何 agent，只要 conform to action interface（shell command、GUI interaction、file I/O），就能在 any task 上被 eval；
2. 同一 task specification 可以不改地部署到 cloud VM 或 local container。

---

## 7. Agent capability taxonomy: GCUA 的 5 层

这是 paper 里最概念性的一块，我觉得也最有用——它把"agent 能干什么"显式拆成 5 个 functional layer，并据此论证为什么 ALE 需要 **Generalist Computer-Use Agent (GCUA)** 这个新类。

### 7.1 五层分解
| Layer | 含义 |
|---|---|
| **Brain** | LLM reasoning + planning |
| **Eyes** | GUI perception via screenshots |
| **Body** | orchestration + control flow |
| **Hands** | structured tool invocation |
| **Feet** | runtime substrate，action 实际生效处 |

### 7.2 现有 agent family 在这五层上的分布
- **CLI-agents** (SWE-agent, ForgeCode): Brain ✓ + Body ✓ + Hands ✓ + Feet ✓，**Eyes ✗ by construction**。ForgeCode: https://github.com/tailcallhq/forgecode
- **Framework-style agents** (OpenClaw): 严格意义不是 CLI-only，但 **不 ship native GUI module**。
- **GUI-agents** (VLA = vision-language action models): Brain ✓ + Eyes ✓，但 Body shallow、Hands narrow (主要是 mouse/keyboard)、Feet restricted → 没法 write code、manage files、sustain long workflow。
- **GCUA** (Claude Code, Codex, ALE-Claw): 5 层全 ✓

> 关键 framing：industry 把 "CUA" 等同于 "GUI-agent"，这是 incomplete conflation。ALE 用 "Generalist" 限定词把这个区分钉死——GCUA 是 GUI+CLI 的 union surface。

### 7.3 GUI-as-Tool vs GUI-as-SubAgent
- **GUI-as-Tool**: GUI 操作作为 ordinary tool 进入主 loop，单一 foundation model 在一个 action loop 内同时 reason shell output 和 visual feedback。**主 benchmark 用这个 mode**——测的是 integrated visual reasoning and action over full task。
- **GUI-as-SubAgent**: 把 GUI interaction 委托给专门 vision-language sub-agent。**只在 model 没有 native vision input 时用**（e.g. DeepSeek V4 Pro）。Reference: https://api-docs.deepseek.com/quick_start/pricing

### 7.4 CUA MCP bridge 的 14 个 desktop-action tool (Table 5)

**Keyboard**:
- `key` — press and release (supports hotkey, e.g. `["ctrl","c"]`)
- `key_down` — press 不 release（用于 modifier hold）
- `type` — 在 focused input field 输入 text
- `hold_key` — 按 hold 一段时间再 release

**Mouse**:
- `mouse_move` — move cursor 到 coordinate
- `click` — click at coordinate（left/right/middle；single/double/triple）

**Utility**:
- `scroll` — 指定方向 + amount 滚
- `screenshot` — 抓屏，可选 save 到 VM path
- `cursor_position` — 返回当前 cursor coordinate
- `wait` — pause 指定时长

### 7.5 Harness 的 6-phase main loop (Figure 8)
```
Phase 0: Initialization → system prompt + tool bindings
Phase 1: Context Building → 组装 current conversation state
Phase 2: LLM Call → 查询 foundation model
Phase 3: Decide → route to final delivery 或 tool invocation
Phase 4: Collect Tool Result → 拿回 execution outcome
Phase 5: Overflow Check → 若超 compaction threshold，触发 context compaction
回到 Phase 1；loop 在 model 决定 deliver 而非 act 时终止
```

### 7.6 System prompt builder 的模块
- Identity (agent persona)
- Memory (persistent cross-session state)
- Tool Guidance (per-tool usage convention)
- Runtime (environment metadata)
- Behavioral Rules (safety + policy)
- Skills (domain-specific capability)

通常通过 `CLAUDE.md` / `AGENTS.md` 配置。

### 7.7 Context manager 三层 compaction
1. **Micro-compaction**: 清掉 stale tool results in place
2. **LLM-based summarization**: 把 older conversation segment 压成 structured checkpoint
3. **Truncation**: 硬 enforce context-window limit（e.g. 400K 或 1M tokens）

这个 graduated 策略保 recent detail 又保 long-range planning state。

### 7.8 Sub-agent 机制
可 spawn **General sub-agent**（全 tool 访问）和 **Explore sub-agent**（read-only）等。运行在 isolated context window，返回 summarized result 给 parent loop。这是 parallel exploration + 限制 context consumption 的关键。

---

## 8. Evaluation modes: 两条正交 axis

### 8.1 Comparison form — 7 种 artifact mode (Table 3)

| Mode | Reference form | Locale | Judge | 代表 workflow |
|---|---|---|---|---|
| Exact / hashed value | secret / digest | host | code | `cybersecurity/snake_crackme` |
| Structured tabular | manifest of (field, value, tolerance) | host 或 VM | code | `finance/sec_10k_financial_parsing` |
| Geometric / spatial | STL / mesh / point cloud | VM | code | `manufacturing/gcode` |
| Visual appearance | reference screenshot | host | vision LLM | `game/mota_reproduction` |
| Behavioral / world state | deterministic state dump | VM | code | `architecture/parametric_energy_simulation` |
| Free-text / semantic | rubric | host | LLM | `finance/equity_research_summary` rubric component |
| Executable artifact | test set / oracle program | host 或 VM | code | `data_computer_science/data_pipeline_etl_instance_1` |

### 8.2 Score composition — 4 种 pattern

**Gate-and-score**: hard precondition 强制 0 on failure，然后 continuous score。防止 reward hacking。
- Canonical 例子: `manufacturing/gcode` 中 PowerMill collision/gouge gate 必须先 pass，否则即便 STL surface 距离再近也 0 分。

**Weighted rubric** (公式记号，关键变量说明):
$$
\text{score} = 0.70 \cdot \text{frac\_within}(0.3\,\text{mm}) + 0.30 \cdot \text{frac\_within}(2.0\,\text{mm})
$$
- 变量说明：
  - $\text{frac\_within}(t)$ = 把 agent 产出的 STL 表面采样 10,000 个点，对每个点计算到 reference STL 的最近距离，统计落在距离 $t$ 内的比例
  - $0.3\,\text{mm}$ 是 tight tolerance（精确制造要求）
  - $2.0\,\text{mm}$ 是 loose tolerance（粗加工可接受）
  - 权重 $0.70$ vs $0.30$ 让 tight tolerance 主导，loose tolerance 作为 fallback——避免 agent 用 "整体接近但 critical 区域歪了" 的策略 hack 满分

**Binary checklist averaging**: $N$ 个独立 yes/no 问题，score 是 mean。`game/mota_reproduction` 用这个 pattern，每个 probe 是 vision-LLM 的 yes/no。

**Pairwise file aggregation**: `utils.evaluation.collect_matching_files` 把 agent directory 里的每个 file 配对 reference file，跑同一个 scoring function，返回 mean。

### 8.3 LLM-as-judge 的明确克制
Paper 反复强调 ALE **默认拒绝 LLM judge**。理由三条：
1. **judge model drift across releases** → 会 silently re-rank agents
2. **"does this look right?" prompt 太 soft** → 没法 discriminate near-correct vs correct，恰恰是 benchmark 最需要 resolve 的 regime
3. **deterministic judge 可以 offline re-run**，没 API cost

### 8.4 实测分布 (Table 4)
- **Code-based deterministic**: 93.2% task workflows
- **LLM-as-judge**: 6.8%
- **Host-side scoring**: 88.5%
- **VM-side verifier**: 11.5%（CAD/CAM kernel、headless 3D renderer、licensed financial workbook 这类没法挪下 VM 的）

### 8.5 LLM-judge 的 narrow probe 原则
6.8% 的 LLM-judge task workflow 里，prompt **绝不问模型"这个 deliverable 看起来对吗"**。每条都是 narrow、evidence-anchored yes/no probe。verbatim 例子：

- `game/mota_reproduction`:
  - "Does the first image show that the game is developed using RPGMakerXP? One can identify whether there is an 'orange sun-like circle' in the top-left corner of the game window." (gate)
  - "Does the first image show with the same map layout as in the original game?"
  
- `audio/timbre_synthesis`:
  - "Does this image show either (a) a software synthesizer / VST plugin interface with visible parameters such as oscillators, filters, envelopes, LFOs, or effects, OR (b) a DAW arrangement / piano-roll / mixer view..."

- `game/skeletal_animation_reproduction`:
  - "Does the submitted preview match the reference body motion well enough to pass?"
  - "Does the replay rendered from final.blend agree with the submitted preview well enough to pass?"

三个 pattern：
1. 每个 probe 锁定一个 identifiable artifact（一个角标、一个 DAW track、一条 UV seam）
2. 很多 probe 是 gate——gate 不过，剩下 rubric 不查
3. 剩下 probe 都是 "...enough to pass?" 同 vs 异 comparator，不是 free-form quality oracle

---

## 9. 三个 difficulty tier

Single run on 1 个 ALE task 平均 $3–10 成本，耗时几十分钟到几小时。完整跑 150-task public set 贵。所以 ALE 把 task 分三档：

| Tier | 任务数 | 目的 | 当前 frontier 表现 |
|---|---|---|---|
| **Near-Term** | 59 | frontier agent 可部分完成，top pass ~30% | cost-effective leaderboard 竞争 + rapid iteration |
| **Full-Spectrum** | 55 | 覆盖每个 55 subdomain 至少 1 instance | comprehensive evaluation |
| **Last-Exam** | 36 | 最难，多数 agent 0% pass | milestone evaluation，不用于 routine 测 |

---

## 10. Main Results (Table 1)

### 10.1 Mainstream harness + GUI-as-Tool (Overall Pass Rate)
- **Codex (GPT-5.5)**: 26.2% — Near-Term 42.4%, Full-Spectrum 20.0%, Last-Exam 8.6%
- **ALE-Claw (GPT-5.5)**: 24.2%
- **Cursor (GPT-5.5)**: 22.5%
- **Cursor (Opus 4.7)**: 21.5%
- **Droid (GPT-5.5)**: 20.1%
- **ALE-Claw (Opus 4.7)**: 19.5%
- **Claude Code (Sonnet 4.6)**: 17.1%
- **Claude Code (Opus 4.7)**: 14.1%
- **Droid (Opus 4.7)**: 13.8%
- **ALE-Claw (GPT-5.4)**: 12.8%
- **Codex (GPT-5.4)**: 7.4%
- **Grok CLI (Grok 4.3)**: 6.7%

关键 reference: Codex (OpenAI): https://openai.com/index/introducing-codex/ ; GPT-5.5: https://openai.com/index/introducing-gpt-5-5/ ; Claude Opus 4.7: https://www.anthropic.com/news/claude-opus-4-7 ; Claude Code: https://docs.anthropic.com/en/docs/claude-code ; Cursor CLI: https://cursor.com/en-US/cli ; Docker Droid: https://docs.docker.com/ai/sandboxes/agents/droid/

### 10.2 Fixed harness (OpenClaw + GUI-as-Tool), 换 backbone
- **GPT-5.5**: 22.8% (Near-Term 39.0%, Full-Spectrum 18.2%, Last-Exam 2.9%)
- **Claude Opus 4.7**: 22.3%
- **Gemini 3.1 Pro**: 16.1% — https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/
- **Claude Opus 4.6**: 15.8%
- **DeepSeek V4 Pro†** (用 visual sub-agent): 15.1%
- **GLM 5.1†**: 13.0% — https://docs.z.ai/guides/llm/glm-5.1
- **Claude Sonnet 4.6**: 12.4%
- **Kimi K2.6**: 10.1% — https://platform.kimi.ai/docs/models
- **Qwen3.6 Plus**: 8.7% — https://www.alibabacloud.com/en/press-room/alibaba-unveils-qwen3-6-plus-to-accelerate-agentic
- **MiMo v2.5**: 8.7% — https://huggingface.co/collections/XiaomiMiMo/mimo-v25
- **MiniMax M2.7†**: 6.0% — https://huggingface.co/MiniMaxAI/MiniMax-M2.7

### 10.3 ALE-CLI (Linux-only subset, 106 tasks)
这是和 Terminal-Bench 直接对标。Codex (GPT-5.5) 在 Terminal-Bench 上是 82%，但在 ALE-CLI 上：

- **Codex (GPT-5.5)**\*: 26.4% overall (Near-Term 42.9%, Full-Spectrum 21.4%, Last-Exam 4.5%)
- **Claude Code (Sonnet 4.6)**\*: 16.5%
- **ForgeCode (Sonnet 4.6)**: 13.2%
- **Hermes (Sonnet 4.6)**: 13.1% — https://hermes-agent.ai/tools/hermes-agent-cli
- **Terminus (Sonnet 4.6)**: 11.8%
- **OpenHands (Sonnet 4.6)**: 9.0% — https://github.com/All-Hands-AI/OpenHands

> Terminal-Bench saturate 的 agent 在 ALE-CLI 上只能拿 26% overall、4.5% Last-Exam。这个 drop-off 本身就是 ALE 比 Terminal-Bench 难多少的量化信号。

### 10.4 Timeout 分析 (Table 6, 7)
5 小时 wall clock cap。整体 timeout rate 4.3%。
- Near-Term: 2.4% timeout, timeout score 42.1 vs other 46.8
- Full-Spectrum: 2.5%, 20.3 vs 21.4
- Last-Exam: 4.4%, 2.0 vs 4.6

按 harness 分：Claude Code timeout 5.1% 最高，Droid 0.4% 最低。OpenClaw 3.7%, Cursor 3.9%, Codex 2.1%, ALE-Claw 1.1%。

---

## 11. 关键 Analysis

### 11.1 Domain-level performance (Figure 9a)
Opus 4.7 和 GPT-5.5 在 13 domain 上的 score profile 高度相似：
- **最高**：computational mathematics 和 agriculture/environment ~60%
- **最低**：visual media 和 education <30%

这暗示两件事一起作用：(i) intrinsic model capability 在不同 domain 上不均匀；(ii) training-time 对 tool-use task 的 exposure 不均，code-adjacent domain 远多于 specialized professional workflow。

### 11.2 Tool usage (Figure 9b, 9c)
- **34% public task instance 把 GUI software 设为主要 tool**
- **但 GUI call share 在多数 configuration 上 stay small**
- agent 倾向于用 Bash/CLI substitute 代替 GUI——这跟下面 failure taxonomy 的 "agent 缺 domain knowledge，默认写 ad-hoc script 而不用 intended domain software" 是同一个现象

### 11.3 Failure taxonomy (Figure 9d, Appendix D.3)

这是用 Claude Code + Opus 4.7 的失败 task 跑出来的 two-stage pipeline：

**Stage 1 — Trajectory analysis**：OpenAI Codex 拿到 full run artifact（interaction_log.json, run_result.json, debug/eval/result.json, events.jsonl），produce 5-section analysis card（Conclusion / Task description / What right / What wrong / Scoring），每个 claim 必须引用具体 artifact file + field 作 evidence。**禁止读 transcript.jsonl** 以控制 cost。

**Stage 2 — Taxonomy classification**：GPT-4o at temperature 0 把 analysis card 分到二级 taxonomy。

**最终分布**（排除 timeout/resource 案例）：
- **Understanding 31%**
  - Domain Knowledge Gap: 25% — "Would a domain expert have avoided this mistake? If yes, classify here."
  - Hallucination/Fabrication: 6%
- **Approach 47%**（最大头）
  - Wrong Strategy: 30%
  - Incomplete/Abandoned: 17%
- **Execution 22%**
  - Output Format Error: 10%
  - Implementation Bug: 8%
  - GUI/Browser Failure: 4%

> **关键观察**：~78% 失败（Understanding + Approach）的 root cause 是 **domain knowledge 和 planning**，不是 execution capability。这正好解释了为什么 GUI underutilization——agent 不懂 domain，所以默认 fallback 到 ad-hoc script 而不是 intended domain software。

### 11.4 Model vs Harness ablation (Figure 12, Appendix D.4)
- 固定 OpenClaw harness，换 12 个 backbone model → **18.0 pp overall pass-rate spread**（Grok 4.3 = 5.3% 到 GPT-5.5 = 23.3%）
- 固定 GPT-5.5 backbone，换 5 个 harness → **6.0 pp spread** (19.3%–25.3%)
- 固定 Opus 4.7 backbone，换 3 个 harness → **5.3 pp spread** (14.7%–20.0%)

> **model choice 的 spread ≈ 3× harness choice 的 spread**。在 well-engineered harness 之间，prompting strategy / tool routing / context management 对 overall performance 的贡献 modest，**foundation model 的 reasoning + domain knowledge 是 dominant factor**。

这跟 11.3 的 failure taxonomy 完全一致——失败主要在 Understanding + Approach，这俩都是 model capability 而非 harness engineering 问题。

### 11.5 Cost/Time/Token efficiency (Figure 13)
- **ALE-Claw (GPT-5.5)**: $307, 48h, 1.35B token, score 48.0%
- **ALE-Claw (Opus 4.7)**: $1,141（3.7×）, score 42.0%（**6pp 更低**）
- **Cursor (GPT-5.5)**: $177, 156M token, 41.7%（token 经济）
- **Codex (GPT-5.4)**: $138, 13.1%（花得不多但分更低）

> 三个观察：
> 1. Cost 和 performance 只是 loosely correlated；
> 2. Wall-clock time 和 score 基本解耦——Claude Code (Sonnet 4.6) 181h 拿 35.3%，Droid (Opus 4.6) 23h 拿 27.3%；
> 3. Token consumption 不预测 performance——ALE-Claw (Opus 4.7) 用 1.35B token 还略低于 Cursor (Opus 4.7) 用 446M token。

---

## 12. Task Card 例子（5 个，最能 build intuition）

这是 Appendix B.3 里最值得读的部分——它把抽象的"workflow task"具体化到 instance level。

### 12.1 Task Card 1 — Injection Mold-Flow Analysis
- **Task ID**: `manufacturing/mold-flow`
- **Software**: Moldex3D CAE
- **Score**: 0.476
- **Input**: 4-cavity meshed project, `process_spec.json` (fill/pack/cool/warp setting), `results_template.json`
- **Reference output**: saved Moldex3D project + `output/results.json` with injection pressure, V/P switch pressure, clamp force, cooling/cycle times, volume, weight
- **Rubric**: hard gate reject missing/malformed/all-null results.json；主 score 比较 numeric field 与 hidden solver reference，**1% relative tolerance**
- **Failure mode**: agent GUI setup workflow 对了，但没等 solver run 完成，**直接 estimate metrics** 而不是 measured CAE results
- **Intuition**: GUI 操作正确 ≠ closed-loop numerical verification。expert 会 wait for result file → inspect/export solver data → populate JSON；agent 缺这层 "等 solver run 完再去读数据" 的 domain habit。

### 12.2 Task Card 2 — Orchestral Music Transcription
- **Task ID**: `audio/music_transcription`
- **Software**: Dorico (notation)
- **Score**: 0.0
- **Input**: audio brief + Dorico env + 27-instrument spec, tempo 140, Dorico Prelude / Akinola
- **Reference**: `transcription.pdf` + `transcription.mid` (multitrack, GM Program Change) + `overview.png`
- **Rubric**: missing PDF/MIDI/invalid screenshot → **hard zero**；过 gate 后 pitch 30%, rhythm 30%, dynamics 20%, instrument 10%, layout 10%
- **Failure mode**: agent 找到了 matching Dorico project 并 export 了 MIDI，但 **PDF 和 overview.png 没确认**
- **Intuition**: partial tool use 不能 translate 成 multi-artifact delivery。creative production task 的 deliverable 是 bundle，缺一个就归零。

### 12.3 Task Card 3 — Chroma Key Compositing
- **Task ID**: `media/chroma_key_from_reference`
- **Software**: DaVinci Resolve
- **Score**: 0.0
- **Input**: bird foreground clip `input.mp4` + reference frame `input.png`
- **Reference**: `output/output.mp4` 含 composited bird shot
- **Rubric**: sample frame from `reference/breakpoint.json`；foreground preservation gated by `roi_input_cv`；quality = full-frame + ROI edge IoU；vision judge 必须确认是真实 keying edit 而非 raw-source copy
- **Failure mode**: agent 用了 Resolve 并 export 了 MP4，但 **treat "plausible bird-over-sky clip" as target** 而不是 preserve intended foreground/background relationship from reference
- **Intuition**: 错在 **visual task grounding**，不在 software operation。agent 看了 reference 但没真正理解 reference 的角色定义。

### 12.4 Task Card 4 — Skeletal Animation Reproduction
- **Task ID**: `game/skeletal_animation_reproduction`
- **Software**: Blender
- **Score**: 0.429
- **Input**: `Singing.obj` 静态 mesh + `reference.mp4` body-motion + 19 个 exact bone name
- **Reference**: `final.blend` (rigged + animation) + `preview.mp4` (body motion, timing, pose range 对齐 hidden clean reference)
- **Rubric**: validity gate 要求两个 artifact + non-trivial rig；score = video match + replay consistency + minimal skeleton coverage + binary vision check
- **Failure mode**: `preview.mp4` **只有 ~4 秒**，reference ~13.6 秒；motion 被 damped（减少 mesh distortion）
- **Intuition**: file-level validity 通过 ≠ full motion-fidelity。agent 知道要做 rig + animation，但 motion 时长 / 振幅这两个 timing-维度直接被 compress 成"够意思就行"。

### 12.5 Task Card 5 — MicroDicom Chest X-Ray Adjudication
- **Task ID**: `radiology/microdicom_nih_cxr_reader_adjudication`
- **Software**: MicroDicom (DICOM GUI)
- **Score**: 0.333
- **Input**: 9 个 NIH CXR DICOM case + 2 个 proposed atelectasis box/case + reader notes + fixed TSV schema
- **Reference**: `adjudicated_boxes.tsv` + `adjudication_log.tsv` + `final_impressions.tsv`
- **Rubric**: hard gate (missing file / malformed header / invalid reader / wrong fixed label)；log/impression file 必须 exact match by case_id；box output 必须 select hidden-supported reader 且 IoU ≥ 0.50 against gold box
- **Failure mode**: 9 个 case 都写了 TSV，但 **只对几个 case 做了 visual review**，后面靠 coordinate + heuristic
- **Intuition**: file-format compliance > case-by-case visual-domain judgment。agent 知道要交 9 个 case 的 TSV，但缺 stamina / domain-confidence 去对每个 case 都做 visual adjudication。

---

## 13. ALE-Claw 这个 "reference implementation" 在做什么

Paper 在 Section 4.1 单独讲了 ALE-Claw。目的是测：**Section 3.2 描述的 basic GCUA components——single action loop + modular tools + GUI-as-Tool + context compaction——是否 suffice 达到 frontier harness 的 performance**。

实现细节（Appendix C.4）：
- 从 OpenClaw（原本 TypeScript，https://docs.openclaw.ai/agent）port 到 Python
- 删掉 production feature：scheduled-prompt subsystem (cron + heartbeat)、multi-channel gateway、skills system、plugin lifecycle hook
- system prompt 减 ~65%
- 加 CUA-specific adaptation：composite computer-use tool、vision-driven GUI sub-agent `delegate_gui`
- 保留 OpenClaw 的 context-management primitive 接近 verbatim

结果：在 fixed backbone 下，ALE-Claw 和 default OpenClaw performance 可比，证明上面那套 minimal component set 已经够用。这本身是个有意思的研究 finding：**frontier harness 之间的差异不大，model 才是大头**——这与 11.4 的 ablation 完全自洽。

> **副产物**：Python port 让 component 可以 in-place swap/ablate。原文写"this workflow is impractical against the original TypeScript runtime"。这是 paper 留给社区的一个工具型 contribution。

---

## 14. 跟 GDPval / RLI 的关键差异

paper 反复对比的 GDPval (OpenAI, Patwardhan et al. 2025) 和 RLI (Mazeika et al. 2025)：

| 维度 | GDPval | RLI | ALE |
|---|---|---|---|
| Size | ~200 | ~250 | ~1.5K instances |
| Industries | 16/55 | 14/55 | **55/55** |
| Task source | industry expert | mined (Upwork) | industry expert past project |
| Horizon | hours-days | hours-days | **hours-weeks** |
| Verification | **human expert** | **human expert** | **deterministic scripts + rubric** |
| Subject | top GDP occupation | Upwork categories | SOC 2018 + O\*NET 30.2 |

ALE 真正的 Δ 是 **scale + automation**——既保留了 GDPval/RLI 的 authentic professional workflow grounding，又用 deterministic verification 把 evaluation cost 从 human judge 摊薄到 code，从而可以 scale 到 1.5K instance 覆盖整个 SOC/O\*NET 空间。

---

## 15. 我自己对这篇 paper 的几个观察

### 15.1 ALE 真正打的是 "model capability" 而非 "harness engineering"
11.3 + 11.4 两条 evidence 拼起来，结论挺硬：**~78% 失败是 Understanding + Approach，model choice spread 是 harness choice spread 的 3×**。这跟过去 SWE-bench 时代 "harness engineering 重要" 的直觉相反——在 long-horizon + multi-domain + heterogeneous deliverable 上，harness 边际收益在递减，model 是 dominant lever。

对未来的 implication：**benchmark saturated 的瓶颈大概率在 pretraining data 上 domain expertise 的覆盖密度，不在 harness trick**。

### 15.2 "GUI underutilization" 是 symptom 不是 cause
34% task 需要 GUI software，但 GUI call share 很小。Paper 没把它当成 "GUI 能力不够" 来 sell，而是放到 failure taxonomy 里说"agent 缺 domain knowledge，所以默认 fallback 到 ad-hoc script"。这跟主流 narrative 不一样。这是个挺锋利的观察——**模型选 Bash 不是因为 Bash 更快，而是因为它不知道该用 Moldex3D / Dorico / DaVinci / Blender**。

### 15.3 Reference isolation + rolling evaluation 是 long-term validity 的核心
90% private pool + 周期性 rotation + LLM judge 6.8% (code-based 93.2%) + judge prompt 都是 evidence-anchored yes/no probe——这套加起来让 ALE 比 SWE-bench / OSWorld 在 contamination 上明显更 robust。Hugging Face leaderboard: https://huggingface.co/spaces/agenthle/leaderboard

### 15.4 "Last-Exam" tier 36 task 是真正的 frontier
- Codex (GPT-5.5) 在 Last-Exam 上 8.6%
- Claude Code (Opus 4.7) 在 Last-Exam 上 **0.0%**
- GLM 5.1、Kimi K2.6、Qwen3.6 Plus、MiniMax M2.7 也都 0%–2.9%
- 整体平均 full pass rate 2.6%

这是 paper 标题里"hard"的具体含义。Frontier model 在 36 个 Last-Exam task 上的 pass rate 接近 0，这给下一代 model 留的 headroom 极大。

### 15.5 几个 paper 没充分 address 的 tension

**T1 — Authenticity vs Reusability**: 1,490 instance 里 960 是 expert 过去已做过的真实项目。这意味着 task 在 expert 完成那一刻已经"freeze"，但产业软件版本（Moldex3D、PowerMill、DaVinci Resolve、Blender）会升级。Paper 没细说 software upgrade 后 task instance 怎么 maintain。

**T2 — Public subset 10% 的 risk**: r=0.89 representativeness 是 single-model (Claude Code + Opus 4.7) 上的。不同 model 的 domain profile 差异大（11.1 已说），所以 public subset 对一个 model representative 不代表对所有 model 都 representative。Appendix D.1 只测了一个 model。

**T3 — LLM-judge 的 6.8%**: 都是 vision-anchored yes/no probe。但 probe 本身是 task author 手写的——probe 的 coverage 和 discriminative power 直接决定 judge quality。Paper 没给 probe-coverage 的 quantification（每个 LLM-judge workflow 平均多少 probe？probe 的 inter-rater reliability 怎样？）。

**T4 — Cost 的 sustainable 性**: 单 task $3–10，150 task public set 跑下来 $200–$1,141 per configuration per run。Paper 主表跑了 ~14 harness × 多个 backbone，总成本是六位数 USD 量级。社区 follow 这个 benchmark 的成本壁垒不低。

**T5 — "Last Exam" 的 naming 拿了 HLE 的命名 pattern**: HLE (Phan et al., Nature 2026) 是 knowledge QA；ALE 是 agentic workflow。两者 framing 不同但 naming 有 lineage 暗示。这种 lineage 对 community buy-in 有帮助，对 idea 原创性的 perception 有 trade-off。

---

## 16. 一段总结性直觉

如果让我用一段话概括 ALE 给我的 intuition：

> **过去三年 agent benchmark 一直在算 "agent 能不能在一个 action 上做对"**（SWE-bench 的 patch pass、OSWorld 的 GUI state check、Terminal-Bench 的 CLI state check）。**ALE 在算 "agent 能不能在一个 professional workflow 上走完"**——这个 workflow 跨 GUI 和 CLI、要求 domain knowledge、要产出 heterogeneous deliverable、要在 long horizon 上 sustain。这种 task 的失败 mode 主要不是 "execution 错"，而是 "understanding 错" 和 "approach 错"——而这两类失败直接归因到 foundation model 的 reasoning + domain knowledge，跟 harness engineering 关系小。所以 ALE 同时是 (i) 一个 benchmark，(ii) 一个把 "model capability 限制" 显式 surface 到 GDP-relevant 空间的 instrument。Saturation of ALE 等价于 model 在这些 profession 上具备 expert-level stamina。

---

## 17. 关键 reference link 汇总

- ALE 主站: https://agenthle.org
- ALE GitHub: https://github.com/agenthle (paper 没给具体 path, 我推断)
- HuggingFace Leaderboard: https://huggingface.co/spaces/agenthle/leaderboard
- Submission portal: https://agents-last-exam.org/submit/new/form
- Demo / Benchmark Splits: https://agenthle.org/demo
- SOC 2018: https://www.bls.gov/soc/2018/
- O\*NET 30.2: https://www.onetcenter.org/database.html
- HLE paper (Nature 2026): https://www.nature.com/articles/s41586-025-09962-4
- GDPval: https://openai.com/index/gdpval/ (Patwardhan et al. 2025)
- RLI (Mazeika et al. 2025): https://remoteaborindex.com
- SWE-bench: https://swebench.com
- OSWorld: https://os-world.github.io
- WebArena: https://webarena.dev
- Terminal-Bench: https://terminalbench.com
- GAIA: https://gaia-benchmark.github.io
- Claude Code: https://docs.anthropic.com/en/docs/claude-code
- Codex: https://openai.com/index/introducing-codex/
- OpenClaw: https://docs.openclaw.ai/agent
- Cursor CLI: https://cursor.com/en-US/cli
- Docker Droid: https://docs.docker.com/ai/sandboxes/agents/droid/
- ForgeCode: https://github.com/tailcallhq/forgecode
- OpenHands: https://github.com/All-Hands-AI/OpenHands
- Hermes: https://hermes-agent.ai/tools/hermes-agent-cli
- ReAct (Yao et al. ICLR 2023): https://arxiv.org/abs/2210.03629
- NIST AI RMF 1.0: https://www.nist.gov/itl/ai-risk-management-framework
- CHIPS and Science Act 2022: https://www.congress.gov/117/plaws/publ167/PLAW-117publ167.pdf
- NIH Strategic Plan 2021–2025: https://www.nih.gov/about-nih/nih-wide-strategic-plan
- NSB 2024 Indicators: https://ncses.nsf.gov/pubs/nsb20243

---

如果你要我 next drill-down，我建议三个方向里挑一个：

1. **挑一个 industry cluster**（比如 Manufacturing & Industrial Operations 或 Biomolecular Structure & Design）把它的 subdomain-level task instance 拉出来，逐个过 task card，看 expert 实际怎么 deliver、agent 在哪一步崩；
2. **挑一个 harness**（比如 Claude Code 或 Codex）的 trajectory，把 6-phase main loop 在某个 Last-Exam tier task 上的 step-by-step trace 还原出来，看 context compaction 触发几次、tool mix 怎么演化；
3. **挑一个 evaluation mode**（比如 geometric/spatial 的 `manufacturing/gcode`）把 `evaluate()` 的完整 code 走一遍，看 18 个 workpiece instance 共享 scoring pipeline 的具体实现，把 $0.70 \cdot \text{frac\_within}(0.3\,\text{mm}) + 0.30 \cdot \text{frac\_within}(2.0\,\text{mm})$ 这个 score 公式的代码 contract 拆开看。

你想先 drill 哪一个？
