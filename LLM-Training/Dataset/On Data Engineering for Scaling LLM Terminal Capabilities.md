---
source_pdf: On Data Engineering for Scaling LLM Terminal Capabilities.pdf
paper_sha256: 4d04ef02b8241b70aaac63135469caeaf1d56a1375d57944e9dfd1672ac2087e
processed_at: '2026-08-05T23:35:38-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 在干嘛

Andrej，我用人话再讲一遍核心思想。

---

## 一句话总结

NVIDIA 的人想训练一个能在 Linux terminal 里干活的 AI agent，最大的瓶颈是**没数据**。他们想了个便宜办法造了一大堆训练数据，结果一个 32B 的小模型把 480B 的大模型干翻了。

---

## 为什么 terminal agent 这么难训练

你想想 ChatGPT 写代码，通常是"给你一个函数签名，你写一段代码"——这是一锤子买卖。但 terminal agent 是**连续剧**：你要先 `cd` 进目录，`pip install` 装依赖，跑一下看看报错，再改代码，再跑 test，test 挂了再 debug……每一步都依赖上一步的状态。

这种 sequential 的活儿，base model（像 Qwen3-32B）在 Terminal-Bench 上只能做对 3.4% 的任务。它根本不会用 shell。

**难点在于数据**：
- 你想让人工录 terminal session？贵得离谱，录几百个小时也就那么点数据。
- 你想让 AI 自己 rollout 生成 trajectory？每个 task 要起一个 Docker 容器，AI 要多轮交互，token 烧得飞起。

所以核心瓶颈不是 model architecture，是**搞不到足够多的高质量训练数据**。

---

## 他们干了啥：两条腿走路

### 第一条腿：把现成 dataset "翻译"成 terminal 格式

市面上已经有一堆高质量 dataset——数学题（OpenMathReasoning）、算法题（OpenCodeReasoning）、修 bug 题（SWE-Bench）。

他们就写了个 template，把这些题包成"在 terminal 里解决"的格式。比如一道数学题，本来是"算这个积分"，包成"在 terminal 里写个 Python 脚本算这个积分，结果写到 output.txt"。

**这条腿的好处**：便宜，不用 LLM 参与，纯模板套用，一下子搞了 22 万条数据。
**坏处**：继承原 dataset 的格式限制，没有 terminal 特有的 sequential interaction 设计。

### 第二条腿：用 LLM 合成 terminal 任务

这条腿分两种玩法：

**玩法 A：从 seed problem 改写**。拿一道已有的科学计算题，让 DeepSeek-V3.2 改写成一个新的 terminal task——要求 agent 装包、读输入、写代码、跑 test。还让 LLM 顺便生成 pytest 测试用例来验证对错。

**玩法 B：从技能 taxonomy 组合**。这个更狠。他们先列了 9 个 domain（security、data science、debugging 等），每个 domain 下列了一堆 primitive skill（比如 BFS/DFS、文件 I/O、加密操作），然后让 LLM **随机挑 3-5 个 skill 组合成一个新任务**。

举个例子：挑了"graph traversal" + "dependency resolution" + "file parsing"三个 skill，让 LLM 编一个"解析某个项目的依赖树并检测循环依赖"的任务。

**这条腿产出的 26 万条数据是性能主力**，因为它们是 terminal-specific 设计的，天然要求 multi-step 交互。

---

## 一个关键 trick：用 9 个预制 Docker 镜像

之前别人做 synthetic terminal task，每个任务都要让 LLM 生成一个 Dockerfile，然后 build 一下看看能不能跑通，build 失败了让 LLM 再修——这个 repair loop 本身就烧钱烧时间。

NVIDIA 这帮人说：**别折腾了，我就预先 build 9 个 domain 镜像**——data science 镜像里装好 pandas、scikit-learn；security 镜像里装好 cryptography 库。生成任务的时候直接用对应 domain 的镜像，agent 需要啥再自己 `pip install`。

这一下把 environment validation 成本砍了 20 倍，让大规模生成变成可能。这是整篇 paper 最"工程化"的智慧——**为了 scale，牺牲 per-task 的精细定制**。

---

## 最反直觉的发现：失败的数据也要留

他们实验了三种数据过滤策略：
1. **不过滤**：所有 trajectory 都要
2. **只留完成的**：teacher agent 跑到 `task_complete` 的才要
3. **只留成功的**：test 全过的才要

直觉上你肯定觉得"只留成功的"数据质量最高，训出来的 model 最好对吧？

**结果完全相反**：
- 不过滤：12.4 分
- 只留完成的：6.74 分
- 只留成功的：5.06 分

差了 2.4 倍！

为什么？因为 terminal agent 的核心能力不是"走 happy path"，而是**遇到报错怎么办**——命令失败了怎么 debug，依赖冲突了怎么修，test 挂了怎么排查。你把失败的 trajectory 全删了，model 就学不到 error recovery 这个最关键的技能。

失败的 trajectory 里藏着 teacher 怎么从 error state 里爬出来的示范，这个 signal 比成功 trajectory 更值钱。

这个发现对整个 agentic AI 训练都有启发：**noise 不一定是 noise，有时候是 signal**。

---

## 其他几个反直觉的点

**Curriculum learning 没用**：本来想先训简单数据（dataset adapter）再训难数据（synthetic task），结果发现一起训反而更好。可能是因为两类数据互为 regularizer，分阶段反而让第二阶段 overfit。

**Long context training 有害**：本来想"terminal session 可能很长，把 context window 扩到 64K 吧"，结果 32K 训练 + 40K 评测反而最好。因为长 trajectory 多是噪声，高质量的 trajectory 都在 32K 以内。

**Synthetic data > Adapter data**：纯 adapter 训练 9.66 分，加 synthetic task 后到 12.4 分。Synthetic data 才是性能主力，adapter 只是打底。

---

## 结果有多炸裂

| 模型 | 参数量 | Terminal-Bench 2.0 |
|---|---|---|
| Qwen3-32B（base） | 32B | 3.4 |
| Qwen3-Coder | 480B | 23.9 |
| GPT-OSS 120B | 120B | 18.7 |
| Gemini 2.5 Flash | - | 16.9 |
| **Nemotron-Terminal-32B** | **32B** | **27.4** |

32B 小模型把 480B 的 Qwen3-Coder 干翻了。14B 模型（20.2 分）比 120B 的 GPT-OSS（18.7 分）还强。

更夸张的是分类别看：Qwen3-32B 在 Data Querying、Model Training、Debugging、Security 这些类别上**完全是 0 分**——它压根不会。SFT 之后从 0 跳到 30-60 分。这说明 terminal 技能根本不是 emergent 的，必须 explicit 教。

---

## 核心 takeaway

1. **Data 比 scale 重要**：32B + 好数据 > 480B + 差数据
2. **失败数据有价值**：error recovery 是 terminal agent 的灵魂
3. **工程 trick 决定 scale**：9 个预制 Docker 镜像让生成成本降 20 倍
4. **Skill taxonomy + LLM 组合**是合成复杂任务的利器
5. **别迷信 long context 和 curriculum**：default 设置往往最好

一句话：**这篇 paper 的核心贡献是证明了"在 terminal agent 这个领域，聪明的 data engineering 比 raw scale 划算得多"**。

---

# Paper 深度讲解：On Data Engineering for Scaling LLM Terminal Capabilities

Andrej，这篇 paper 来自 NVIDIA 的 Renjie Pi、Grace Lam、Wei Ping 等人，核心贡献是揭示了训练 terminal agent 的 **data engineering pipeline**，并开源了 Nemotron-Terminal 模型家族（8B/14B/32B）和 Terminal-Corpus 数据集。我会从 problem motivation、pipeline 架构、teacher model 选择、SFT training 细节、ablation 结论等多个维度，尽可能详细地构建你的 intuition。

参考链接：
- HuggingFace collection: https://huggingface.co/collections/nvidia/nemotron-terminal
- Terminal-Bench: https://github.com/laude-institute/terminal-bench
- Harbor Framework: https://github.com/laude-institute/harbor

---

## 1. Problem Motivation: 为什么 Terminal Agent 这么难做

Terminal agent 和传统 coding agent 的本质区别在于 **sequential environment interaction**。传统 coding benchmark（如 HumanEval、MBPP）评估 isolated function generation——输入一个 signature，输出一段代码；而 Terminal-Bench 要求 agent 完成一个 **end-to-end workflow**：编译代码、训练模型、配置系统、调试环境，需要 agent 在多轮 turn 中维护 shell 状态、处理 errors、做 recovery。

Paper 里点出两个关键 bottleneck：
1. **Foundational resource scarcity**——需要 diverse task prompts + dependency files + pre-configured Docker environments，这三者同时齐备很罕见。
2. **Logistical complexity of trajectory collection**——人工录制 terminal session 成本极高，而用 LLM agent 自己 rollout 来生成 trajectory 又需要为每个 task 实例化 fresh Docker 容器，还要 multi-turn interaction，token 消耗和计算成本爆炸。

目前 terminal agent 提升的两条路线：
- **Agentic scaffolding 改进**（Claude Code、Warp、Factory Droid、Letta 等），但 scaffolding 是 model-specific 的 heavy engineering，base model 变强后 marginal benefit 会下降。
- **Base model 改进**——通过 SFT/RL 提升 terminal capability，本 paper 走这条路。

**核心 insight**：之前的 dataset adapters（如 DCAgent、mlfoundations-dev 在 HuggingFace 上的 bash_textbook_tasks_traces 等）把现有 benchmark 包成 terminal format，但这种 wrapper 继承了 source dataset 的 structural assumption，缺乏 sequential interaction 设计。Paper 想要的是一个 **principled 的 data engineering framework**，兼顾 generation efficiency 和 terminal-specific 的需求。

---

## 2. Terminal-Bench 2.0 和 Terminus 2 Agent：评测基础

### 2.1 Terminal-Bench 结构
Terminal-Bench 2.0（TB2.0）包含 89 个 hand-crafted、human-verified 的 task，覆盖 scientific computing、software engineering、ML、security、system administration、data science 等领域。每个 task 由四部分构成（见 paper Figure 2）：
1. **Natural language instruction**：描述目标。
2. **Docker containerized environment**：提供 execution context。
3. **Verification test suite**：程序化检查 task 是否完成（pytest-based，支持 configurable weights 做 partial credit）。
4. **Oracle solution**：展示一种 valid approach。

注意 paper 强调一点：他们自己生成的 synthetic task **不生成 oracle solution**，因为无人工验证的情况下生成 ground-truth code 太难；他们生成的是 "easy to verify yet difficult to solve" 的 task，用 synthesized test cases 来评估 agent 解的正确性。

### 2.2 Terminus 2 Agent Framework
Terminus 2 是 TB2.0 团队发布的 model-agnostic reference agent。和传统 coding agent 提供多个 specialized tool（如 read_file、write_file、run_tests）不同，Terminus 2 **只提供一个 interactive tmux session** 运行在 sandboxed Docker container 内。agent 通过发送 model-determined keystrokes 到 tmux session，自由使用任何 command-line tool。

每一步 agent 接收 current terminal output，输出一个 **structured JSON format**（见 paper Figure 3）：
```json
{
  "analysis": "Analyze current state, what's been done, what remains...",
  "plan": "Describe plan for next steps, which commands and why...",
  "commands": [
    {"keystrokes": "ls -la\n", "duration": 0.1},
    {"keystrokes": "cd project\n", "duration": 0.1}
  ],
  "task_complete": false
}
```

这个 JSON 的设计有几个细节值得关注：
- `keystrokes` 内容是 **verbatim** 的，不做 JSON escaping，直接作为 tmux keystroke 发送——所以 `\n` 是真正的换行，`C-c` 是 Ctrl+C 的 tmux-style escape。
- `duration` 字段控制命令执行后等待时间，immediate command 用 0.1s，slow command（make、python3 long script）适当调大但不超过 60s，preferring polling。
- `analysis` 和 `plan` 字段强制 agent 做 **chain-of-thought style 的 explicit reasoning**——这种 explicit planning 是 terminal agent 不会在 long horizon 中迷路的关键。

---

## 3. Terminal-Task-Gen Pipeline: 双流数据生成架构

这是 paper 的核心贡献，见 Figure 1。整个 pipeline 分为两支 + 后续 trajectory generation + post-processing。

### 3.1 Stream 1: Dataset Adaptation（粗粒度，广覆盖）

这个 stream 把现有的 math、code、SWE prompt dataset 通过 wrapper 转成 Terminal-Bench format，**不需要 LLM in the loop**，纯 template-based。

**Math Prompts**：使用 Nemotron-Cascade 的 Stage-2 prompt set，163K unique prompts from OpenMathReasoning。筛选规则是：剔除 DeepSeek-R1 response length < 2K tokens 的简单题，确保 difficulty。

**Code Prompts**：Nemotron-Cascade 的 Stage-2 code reasoning SFT data，79K prompts from OpenCodeReasoning，再 filter + dedupe 到 35K。

**SWE Prompts**：Nemotron-Cascade 的 SWE code repair data，127K instances from SWE-Bench-Train、SWE-reBench、SWE-Smith、SWE-Fixer-Train，每个 prompt 包含 problem statement + buggy code file 内容，filter 后 32K unique prompts。

**Adapter Format**：使用 Terminus 2 system prompt template（见 paper Appendix Figure 7），把原始 prompt 填入 `{instruction}` placeholder，append domain-specific suffix：
- Math adapter suffix（Figure 8）：指示 agent 用 Python 写解题代码、读取 input file、输出到指定 path。
- Code adapter suffix（Figure 9）：要求 agent 实现 algorithm、handle edge cases、写 test。
- SWE adapter suffix（Figure 10）：要求 agent 定位 bug、修复、跑 test。

对 SWE prompt 中提到的每个 code file，在 Docker environment 内 instantiate 对应的 file。注意：dataset adapter 生成的 task **没有 test cases**——只有 instruction + environment。这意味着 trajectory 的"对错"无法直接通过 test 验证，只能靠 trajectory 本身的质量。

**关键 design choice**：Dataset adaptation 是低成本的 volume scaling 通道，但继承 source dataset 的 structural assumption，缺乏 sequential interaction 设计。所以 paper 又设计了 Stream 2 来补 capability gap。

### 3.2 Stream 2: Synthetic Task Generation（细粒度，目标导向）

这部分分两种方法。

#### 3.2.1 Seed-based Generation
不是把 seed prompt 直接 wrap 进 terminal scaffold，而是用 LLM 把 seed problem **synthesize 成新的 terminal task**。

Seed data 结构包含：
1. Problem description（计算挑战的描述）
2. Optional domain label（biology、physics、optimization 等）
3. Optional reference solution（ground truth，**绝不暴露给 agent**）

LLM 作为 **task adapter** 做几个操作：
1. **Augmentation**：abstract problem statement 加上 concrete SWE requirements——agent 必须 install packages、从指定 file path 读 input、实现 solution、把结果写到 designated output location。
2. **Test data generation**：生成 realistic input data files，包含 edge cases 和 boundary conditions。
3. **pytest-based test synthesis**：检查 output file existence、format compliance、numerical accuracy（with floating-point tolerance）、edge case handling。当 seed 提供 reference solution 时，用它作为 ground truth 来 design test expectations。

**Solution Isolation Principle**：所有 generation prompt 都强制 instruction visible to agent 不能 reveal algorithm、implementation approach 或任何解题 code。这确保 task 需要 problem-solving 而非 solution retrieval。

**Conversion Guidelines**：复杂问题分解成 verifiable units；加 practical constraints（input size、precision）；output format 设计为 unambiguous programmatic verification。

#### 3.2.2 Skill-based Generation（这条是性能核心）

这个方法 fundamentally 不同：**从结构化的 primitive skill taxonomy 合成 novel task**，而不是 adapt 现有 problem。

**9 个 Task Domains**：data processing、data querying、data science、debugging、dependency management、file operations、scientific computing、security、software engineering。每个 domain 有 dedicated generation prompt。

**Skill Taxonomy**（见 Table 10）：每个 domain 的 primitive skills 跨越多个维度：
1. **Algorithmic skills**：graph traversal、constraint satisfaction、backtracking search
2. **Systems skills**：file I/O、process management、network configuration
3. **Data processing skills**：parsing、serialization、transformation pipelines
4. **Mathematical skills**：numerical integration、statistical modeling
5. **Testing skills**：validation、verification、benchmarking
6. **Web/security skills**：HTTP handling、authentication、vulnerability analysis

**Compositional Task Synthesis**：LLM 被指示把多个 primitives（typically 3-5 skills per task）以 non-trivial 方式组合，产出需要 integrated problem-solving 的 task，而非 isolated skill application。Generation prompt 强调 **novelty**：引导 model 发明新场景，最大化 diversity 和 coverage。

举例（来自 Table 10）：
- Security domain：craft exploit payloads to bypass authentication and identify vulnerabilities
- Software Engineering：implement graph traversal (BFS/DFS) for dependency resolution
- File Operations：parse structured formats (JSON/XML/CSV) with encoding and validation
- Data Querying：writing queries using formal syntax of declarative query languages for structured data
- Data Science：load and transform tabular data with groupby, filtering, and aggregation
- Debugging：resolve package dependency conflicts through constraint analysis

#### 3.2.3 Task Format and Pre-Built Docker Images

两个 generation method 都产出 standardized format：
1. Natural language task prompt
2. pytest-based test cases with configurable weights
3. Supplementary input files
4. Domain-specific Docker environment

**关键 design decision: Pre-Built Docker Images**：和之前 work（Austin 2025、Peng et al. 2025 的 LiteCoder）每个 task 生成 unique Dockerfile 不同，paper 维护 **9 个 fixed domain-specific base images**，每个预装该 domain 常用 packages（如 data science image 预装 pandas + scikit-learn；security image 预装 cryptography libraries）。

这带来三个 scalability 优势：
1. **消除 Dockerfile validation overhead**：避免 per-task environment generation 的 multi-turn repair loop，enable single-pass task creation。
2. **降低 resource footprint**：只需 9 个 shared base image 而非 build/cache 数千个 unique container。
3. **Decouple environment 和 task generation**：在稳定 environment 内产 diverse scenario，仍保留 agent install runtime dependency 的 flexibility。

这个 trade-off 很重要——pre-built image 牺牲了 per-task environment 的精细定制，但换来大规模 generation 的 tractability。这是 paper 在 data engineering 上的核心 trick 之一。

---

## 4. Teacher Model: DeepSeek-V3.2

Paper 选 DeepSeek-V3.2 作为 teacher model 生成 synthetic task 和 trajectory，理由是它在 TB2.0 上 performance 强（38.2 ± 2.9，接近 GPT-5 的 35.2、Claude Opus 4.5 的 57.8 之下但相当 competitive）。

为了进一步验证 DeepSeek-V3.2 适合产 dataset adapter trajectory，paper 在 standard benchmark 上用 Terminus 2 agent 做了 evaluation（Table 2）：
- AIME 2024 + 2025：93.33 pass@1
- LiveCodeBench v6：67.20
- SWE-bench Verified：52.40

这说明 DeepSeek-V3.2 在 terminal-adapted format 下仍然保持 strong reasoning 和 coding 能力，trajectory 质量有保证。

---

## 5. Data Filtering: 反直觉的关键发现

这一节的 ablation 结果非常重要，build intuition 的关键。

### 5.1 Decontamination
先用 14-gram overlap 移除任何和 TB2.0 test sample 有重叠的 prompt。然后做 quality filtering：移除 identity leak、丢弃含 Chinese character 的 response。

### 5.2 Trajectory Filtering（核心 ablation）

Paper 实验了三种 filtering 策略：
- **No filter**：保留所有 trajectory
- **Complete-only**：只保留 teacher 跑到 task_complete 的 trajectory
- **Success-only**：只保留 test cases 通过的 trajectory（仅对 synthetic task 可用，因为 dataset adapter 没 test）

#### Dataset Adapter 上的结果（Table 6）
- Math：complete-only 7.19 vs no-filter 5.39（差异不显著，但 no-filter 反而略低）
- Code：complete-only 6.07 vs no-filter 6.29
- SWE：complete-only 5.39 vs no-filter 7.02
- All：complete-only 8.09 vs no-filter **9.66**（采用 no-filter）

#### Synthetic Task 上的结果（Table 7）
- Complete-only：104,603 samples, 6.74 ± 2.20
- Success-only：83,448 samples, 5.06 ± 2.11
- No filter：264,207 samples, **12.4 ± 2.29**

**这个 gap 极其显著**：no-filter 比 success-only 高出 2.4 倍！

Paper 的解释非常 insightful：
> "Strict filtering is detrimental as it discards over half the available training data. Moreover, retaining unsuccessful trajectories appears to provide valuable supervision, exposing the model to realistic error states and recovery patterns that enhance overall robustness."

**Intuition**：terminal agent 的核心 challenge 不是 happy path，而是 **error recovery**——当命令报错、依赖冲突、test 失败时如何 recover。Success-only filter 把这些 error recovery 段全删了，model 学不到关键 robustness 技能。失败的 trajectory 包含 teacher 在 error state 下的 recovery attempt、命令纠错、依赖问题处理等信号——这些都是 terminal agent 必备的"反脆弱"能力。

这和你（Karpathy）常强调的 "data quality 不等于 data cleanliness" 一致——**noise 有时是 signal**。

---

## 6. Training Details: hyperparameter 全解析

### 6.1 SFT Hyperparameters
- **Learning rate**：2e-5
- **Weight decay**：1e-4
- **Epochs**：2
- **Max sequence length**：32,768 tokens
- **Global batch size**：128
- **Micro batch size**：1 per GPU（用 gradient accumulation 到 global bs）
- **Optimizer**：AdamW with β=(0.9, 0.95)（注意 β2=0.95 而非默认 0.999，这暗示训练时间短，short-tail gradient 估计够用）
- **LR scheduler**：cosine with 10% warmup
- **Gradient clipping**：1.0

### 6.2 Infrastructure
- 8B 和 14B：4 nodes × 8 GPU = 32 GPU，sequence parallelism = 2
- 32B：16 nodes × 128 GPU
- 全部使用 CPU offloading
- veRL 框架（Sheng et al. 2024 的 HybridFlow）做 SFT
- Harbor 框架（TB2.0 团队的）做 trajectory generation orchestration，扩展支持 Singularity（HPC cluster），容忍 fakeroot overlay limitation 的 rare failure
- Daytona 做 evaluation 的 isolated cloud sandbox

### 6.3 Long Context Training（Table 8）

Paper 实验 SFT max length 和 YaRN2 scaling 的组合，结果反直觉：

| SFT Max Len | Eval Max Len | SFT YaRN2 | Eval YaRN2 | TB2.0 |
|---|---|---|---|---|
| 32,768 | 40,960 | - | - | **13.0 ± 2.2** |
| 32,768 | 65,536 | - | √ | 11.9 ± 2.0 |
| 65,536 | 65,536 | - | - | 10.3 ± 2.0 |
| 65,536 | 65,536 | √ | √ | 11.9 ± 2.1 |

**最佳配置是 default Qwen3 context settings**：32K SFT + 40K eval，不用 YaRN2。

Paper 解释：扩展 context length 反而伤害 performance。大多数 high-quality supervision 已经在 default window 内，long-tail trajectory 多为 noisy 和 less informative。这是 terminal trajectory 的 token 分布特性——见 Appendix Figure 5，绝大多数 trajectory 在 32K tokens 以内。

**Intuition**：不要被"long context 总是更好"的迷思骗。对于 SFT，max length 决定了你 sample 的 trajectory 上限，但 long trajectory 质量下降。短而精的 supervision 比长而 noisy 的 supervision 更值钱。

---

## 7. Main Results: 性能跃迁

### 7.1 Terminal-Bench 2.0 主榜（Table 3）

| Model | Size | TB2.0 |
|---|---|---|
| Qwen3-8B (base) | 8B | 2.47 ± 0.5 |
| Qwen3-14B (base) | 14B | 4.04 ± 1.3 |
| Qwen3-32B (base) | 32B | 3.37 ± 1.6 |
| Qwen3-Coder | 480B | 23.9 ± 2.8 |
| GPT-OSS (high) 120B | 120B | 18.7 ± 2.7 |
| Gemini 2.5 Flash | - | 16.9 ± 2.4 |
| Grok Code Fast 1 | - | 14.2 ± 2.5 |
| GPT-5-Nano | - | 7.90 ± 1.9 |
| **Nemotron-Terminal-8B** | 8B | **13.0 ± 2.2** |
| **Nemotron-Terminal-14B** | 14B | **20.2 ± 2.7** |
| **Nemotron-Terminal-32B** | 32B | **27.4 ± 2.4** |

亮点：
- **Nemotron-Terminal-8B 从 2.47 提升到 13.0，5x 跃升**
- **Nemotron-Terminal-14B (20.2)** 超过 120B GPT-OSS (18.7) 和 Gemini 2.5 Flash (16.9)
- **Nemotron-Terminal-32B (27.4)** 超过 480B Qwen3-Coder (23.9)

这证明 **高质量 trajectory data 比 sheer parameter scale 更关键**——一个 32B 模型通过 SFT 打败 15x 大小的 baseline，是 data-centric AI 的强证据。

### 7.2 By Category 分析（Table 4）

这是更细粒度的诊断，揭示 base model 在哪些 category 上完全缺失 capability：

**Software & System**:
- Software Engineering (24 tasks): Qwen3-32B 5.0 → Nemotron-T-32B 31.7（6x 提升）
- System Administration (9): 6.7 → 31.1（5x）
- Debugging (3): 0.0 → 33.3（从 0 到 ~33%）
- Security (8): 2.5 → 27.5（11x）
- File Operations (4): 0 → 5.0

**Data & Science**:
- Data Science (8): 0 → 27.5
- Data Processing (4): 5 → 50.0（10x）
- Data Querying (1): 0 → 60.0
- Scientific Computing (7): 2.9 → 0（这里 paper 没特别说明为何下降，可能是 synthetic data 没覆盖这类，或 evaluation noise）

**Machine Learning**:
- Machine Learning (3): 0 → 13.3
- Model Training (4): 0 → 50.0

**Other**:
- Personal Assistant (1): 0 → 100（这个 task 数量太少，单 task 影响）
- Unknown (7): 8.6 → 34.3

**关键 insight**：Qwen3 base model 在 Data Querying、Model Training、Debugging、Security、File Operations 等 category 上几乎 0 分——这些是 terminal-specific 的 skill，pretrain 阶段没有专门训练。SFT 后从 0 到 30-60 分，证明 synthetic data 解锁了 base model 完全缺失的 capability。这说明 terminal capability 不是 emergent 的，需要 explicit supervision。

---

## 8. Ablation Studies

### 8.1 Dataset Component Ablation（Table 5）

Qwen3-8B 上各 data source 单独训练的效果：

**Dataset Adapters**:
- Math only (162,692 samples): 5.39 ± 1.65
- Code only (31,960): 6.29 ± 1.65
- SWE only (31,661): 7.02 ± 2.13
- All (226,313): **9.66 ± 2.11**

**Synthetic Tasks**:
- Seed-based only (124,366): 6.18 ± 1.91
- Skill-based only (139,841): **12.4 ± 2.38**
- All (264,207): **12.4 ± 2.29**

**两个关键观察**：
1. **Dataset adapter 中，单一 domain 都不如三者合并**——Math 5.39、Code 6.29、SWE 7.02 单独效果都低，合并后 9.66，证明 cross-domain data 的 complementary value。
2. **Synthetic task 中，skill-based 是主力**（12.4），seed-based 加入后 mean score 不变但 **variance 降低**——seed-based data 的作用是 robustness 而非 raw performance，提供 broad coverage 平滑了 skill distribution。

### 8.2 Curriculum Learning Ablation（Table 9）

两种策略对比：
- **Mixed single-stage**：所有 data 一起训练 → **13.03 ± 2.16**
- **Two-stage curriculum**：先 dataset adapter，再 synthetic task → 10.39 ± 1.71

**反直觉结论**：curriculum 反而更差。Paper 采纳 mixed training。

**Intuition**：curriculum learning 在这个场景下不如 mixed，可能因为：
1. 两阶段训练总 step 数相同，但第二阶段（synthetic task）训练步数减半，overfit 风险高。
2. dataset adapter 和 synthetic task 互为 regularizer，mixed 时 model 同时见到 broad coverage 和 targeted skill，inductive bias 更平衡。
3. Qwen3 base 已经有 strong foundation，不需要 coarse-to-fine 的 progressive unlocking。

### 8.3 Scaling Experiments（Figure 4）

Qwen3-8B 和 14B 在 synthetic training data 0%、1%、2%、5%、10%、100% 上的 performance：
- 两个 model 都随 data scale 单调上升
- 14B 在所有 data scale 上 absolute performance 都高于 8B
- 14B 从 additional data 获得的 **gain 更大**（slope 更陡）

**Scaling law intuition**：这里有两层 scaling 在 play：
1. **Model capacity scaling**：14B > 8B，相同 data 下表现更好
2. **Data efficiency scaling**：14B 从同样 additional data 获得更大增益——大 model 更能利用 data

可以用一个 conceptual 公式表达：
```
Performance(model_size, data_size) = α × log(model_size) + β × log(data_size) + γ × log(model_size) × log(data_size) + C
```
第三项 cross term 在 Figure 4 上体现为 14B 曲线比 8B 更陡——大 model 不仅 capacity 更高，data efficiency 也更高。

---

## 9. 技术细节深挖：Solution Isolation 和 Pre-Built Docker

### 9.1 Solution Isolation
这是 paper 反复强调的设计原则：generation prompt 强制 task prompt visible to agent 不含 algorithm hint、implementation approach 或解题 code。reference solution 仅用于 design test expectation。

这避免了一个常见 failure mode：如果 generation 时把 reference solution 放进 task description，agent 就退化成 retrieval task 而非 problem-solving，训练数据失去学习价值。

### 9.2 Pre-Built Docker 的 ROI 分析
9 个 domain base image vs per-task unique Dockerfile 的 trade-off：

**Per-task Dockerfile 的成本**（Austin 2025、Peng et al. 2025 的方法）：
- 每个 task 需要 LLM 生成 Dockerfile
- Dockerfile 可能 syntax error、依赖冲突、build 失败
- 需要 multi-turn repair loop，每个 turn 是一次 LLM call + Docker build
- 假设 10K task，每个 task 平均 3 turn repair，每个 turn 5s LLM call + 30s Docker build = ~10K × 3 × 35s ≈ 290 GPU-hours 仅 environment validation

**Pre-built image 的成本**：
- 一次性 build 9 个 domain image
- 每个 task 只需 LLM 生成 task prompt + test case，单次 LLM call
- 10K task × 5s = ~14 GPU-hours

ROI 提升约 20x。这就是 paper 强调 scalability 的实际意义。

---

## 10. Pipeline 全流程 Visualization

把整个 pipeline 用 ASCII 表达，build 一个完整 intuition：

```
                     ┌──────────────────────────────────────────┐
                     │       Existing High-Quality Datasets     │
                     │  OpenMathReasoning | OpenCodeReasoning   │
                     │  SWE-Bench | SWE-reBench | SWE-Smith     │
                     └────────┬──────────────────┬──────────────┘
                              │                │
                              ▼                ▼
                ┌──────────────────┐  ┌──────────────────────┐
                │ Stream 1: Adapter│  │ Stream 2a: Seed-based │
                │ Template wrap    │  │ LLM synthesize novel  │
                │ (no LLM in loop) │  │ task from seed problem │
                │ 226K samples     │  │ 124K samples          │
                └────────┬─────────┘  └───────────┬──────────┘
                         │                        │
                         │                        ▼
                         │           ┌──────────────────────┐
                         │           │Stream 2b: Skill-based │
                         │           │ 9 domain × taxonomy   │
                         │           │ Compose 3-5 skills    │
                         │           │ 140K samples          │
                         │           └───────────┬──────────┘
                         │                       │
                         └───────────┬───────────┘
                                     ▼
                    ┌──────────────────────────────┐
                    │  Pre-Built Domain Docker     │
                    │  (9 shared base images)      │
                    └────────────┬─────────────────┘
                                 ▼
                    ┌──────────────────────────────┐
                    │ Teacher: DeepSeek-V3.2       │
                    │ + Terminus 2 agent scaffold  │
                    │ Multi-turn rollout in Docker │
                    └────────────┬─────────────────┘
                                 ▼
                    ┌──────────────────────────────┐
                    │ Post-Processing              │
                    │ - 14-gram decontamination    │
                    │ - Quality filter (identity,  │
                    │   Chinese chars)             │
                    │ - No trajectory filter!      │
                    │   (key finding)              │
                    └────────────┬─────────────────┘
                                 ▼
                    ┌──────────────────────────────┐
                    │ Terminal-Corpus (~490K SFT)  │
                    │ 226K adapter + 264K synth   │
                    └────────────┬─────────────────┘
                                 ▼
                    ┌──────────────────────────────┐
                    │ SFT on Qwen3 (8B/14B/32B)    │
                    │ lr=2e-5, 2 epoch, 32K ctx   │
                    │ Mixed (no curriculum)        │
                    │ AdamW β=(0.9, 0.95)          │
                    └────────────┬─────────────────┘
                                 ▼
                    ┌──────────────────────────────┐
                    │ Nemotron-Terminal Family     │
                    │ 8B: 13.0 | 14B: 20.2 | 32B: 27.4 │
                    └──────────────────────────────┘
```

---

## 11. 和现有工作的差异化

Paper 在 Related Work 里清晰划定了边界：

**vs Claude Code / Codex CLI**：他们走 scaffolding 路线，本 paper 走 base model SFT。

**vs DCAgent / mlfoundations-dev dataset adapters**：那些是现成 dataset 的 wrapper，本 paper 系统研究 adapter 的有效性 + 提出 synthetic generation pipeline 补 capability gap。

**vs Evol-Instruct / AgentInstruct / LAB / MAGPIE**：那些是 generic instruction data scaling，本 paper 专门针对 terminal sequential interaction 设计。

**vs LiteCoder-Terminal（Peng et al. 2025）**：LiteCoder 用 multi-agent 系统 brainstorm + generate + validate，本 paper 用 simplified single-LLM + pre-built Docker，eliminate unnecessary coordination stages，scale 更好。

**vs Austin (Terminal Bench agentic data pipeline)**：multi-agent approach 时间和成本高，本 paper 通过 pre-built image + simplified system 优化 generation 成本。

---

## 12. Limitations 和 Future Work

Paper 自己提到的 limitation 和 future direction：
1. **不生成 oracle solution**——无人工验证，靠 synthesized test case 做 correctness check，存在 test case 本身错误的 risk。
2. **Scientific Computing category 在 SFT 后下降**（2.9 → 0），说明 synthetic data 没覆盖好这类，可能是 domain-specific generation prompt 不够。
3. **Future: RL on verifiable execution feedback**——paper 明确说接下来想用 RL + execution feedback 做 self-correction 和 optimal planning for long-horizon task。这是从 SFT 进化到 RLHF/RLEF 的明确路线图。

我额外观察的潜在问题：
- 32B 模型在 Personal Assistant category 上 100 分但只有 1 个 task，statistical significance 弱。
- Single-agent generation 虽然 scalable，但可能缺乏 multi-agent 的 diversity（不同 agent brainstorm 出不同 task type）。
- DeepSeek-V3.2 作为 teacher，可能继承其 bias（如偏好某些 code style、某些 library），limit student model diversity。
- 14-gram decontamination 可能不够 robust——同义改写后的 prompt 仍可能 leak。

---

## 13. 论文与 Broader Research Context

把这篇 paper 放在你的 research 视野下，几个 connection：

1. **e2e ML system 的 data-centric thinking**：和你在 Stanford CS231n 强调的 "bitter lesson" 一致——general method + scale > hand-crafted feature。这里 pre-built Docker + simplified generation > multi-agent 复杂系统。

2. **Trajectory quality vs quantity trade-off**：和 OpenAI 的 OLMO、Anthropic 的 RLHF 经验一致——**unsuccessful trajectory 也有 value**。这和你之前讨论 "RL from failure" 思路吻合。

3. **Synthetic data 的 scaling law**：Figure 4 的 log-log scaling 在 1%-100% 之间近似线性，和 Chinchilla 的 compute-optimal scaling 互补——这里 data scale 和 model scale 是两个独立 axis，cross term 表明大 model 更能利用 data。

4. **Terminal agent 作为 AGI proxy**：Terminal 是 real-world workflow 的微缩宇宙——stateful、compositional、verifiable。本 paper 在这个 proxy 上证明 small model + targeted data > giant model + generic data，对你最近关注的 "practical AGI" 思路有参考价值。

5. **和 RLVR（Reinforcement Learning from Verifiable Rewards）的关系**：terminal task 天然 verifiable（test case pass/fail），是 RLVR 的 ideal testbed。Paper 提到未来想用 RL，这正是 DeepSeek-R1、Kimi K2 Thinking 等已经在做的方向。本 paper 的 SFT 数据可以作为 RL 的 warm start，提供 base policy。

---

## 14. 关键 Takeaway 总结

1. **No-filter > strict filter**：error recovery trajectory 是 terminal capability 的核心 signal，不要洗掉。
2. **Mixed training > curriculum**：data 互为 regularizer，coarse-to-fine 在 strong base model 上不必要。
3. **Default context window 足够**：long context training 反而 hurt performance，long trajectory 多为 noise。
4. **Pre-built Docker >> per-task Dockerfile**：scalability 关键 trade-off，20x ROI 提升。
5. **Skill-based > seed-based > adapter**：targeted synthetic 是主力，seed-based 提供 robustness，adapter 提供 breadth。
6. **32B SFT > 480B baseline**：data-centric AI 在 terminal domain 的强证据。

希望这个 walkthrough 帮你 build 起完整的 intuition，Andrej。如果想深入某个具体 module（如 skill taxonomy 的某个 domain 的 prompt 设计、Terminus 2 的 JSON schema 细节、YaRN2 在 long context 的作用机制等），可以告诉我继续展开。
