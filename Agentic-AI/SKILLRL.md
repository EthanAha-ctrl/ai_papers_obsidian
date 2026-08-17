---
source_pdf: SKILLRL.pdf
paper_sha256: 7624ecf49b96ce6bbe097c8054c33f77ca72865a07d277fcd7bc9c0edffb1658
processed_at: '2026-08-12T07:27:55-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SKILLRL 用人话说

## 一句话

LLM agent 干完活就忘，下次重来从零开始。SKILLRL 教 agent 把经验总结成"技巧卡片手册"，训练时手册还自动更新，7B 小模型干翻 GPT-4o。

---

## 痛点在哪

想象你雇了个实习生做家务。第一天他成功洗了生菜放台面。第二天你让他热鸡蛋放台面，他从头摸索：先去微波炉那儿转一圈发现没鸡蛋，再去找鸡蛋，找到后忘了微波炉要先开门，直接 heat 报错...

你跟他说"上次你不是做过类似任务吗？" 他答"啥？我不记得了，我只存了上次完整操作录像但没看。"

这就是当前 LLM agent 的现状。ReAct、Reflexion 这类方法本质是 prompt-based，把上次的 trajectory（操作录像）塞进 context window。问题：

1. **录像太长**：ALFWorld 一个任务 50-100 步 action，每步 5-10 tokens，一个 trajectory 500-1000 tokens。塞几个进去 context 就爆了。
2. **废话太多**：agent 探索时到处走、开错抽屉、回退、重复 action。真正关键的 decision point 可能就 3-5 个，淹没在 100 步噪声里。
3. **失败经验被丢弃**：传统只存 success trajectory。但失败其实信息更 dense——失败一定有 specific failure point，成功却经常靠运气+冗余 action。

Mem0、ExpeL 这些 memory-based 方法改进了存储，但本质还是存 trajectory 或粗略 insight。MemRL 用 RL 更新 memory bank 但 policy frozen。EvolveR 同步更新 policy 与 memory 但 memory 还是粗存。

核心问题：**raw experience 信息密度太低，无法 transfer**。

---

## SKILLRL 怎么解决：三个核心招

### 招一：把录像蒸馏成技巧卡片

不存 raw trajectory，让 teacher model（OpenAI o3）看 trajectory 然后总结成 skill。

对成功 trajectory $\tau^+$：

$$s^+ = \mathcal{M}_T(\tau^+, d)$$

- $s^+$：从成功里提炼的 skill
- $\mathcal{M}_$：teacher model
- $\tau^+$：成功 trajectory
- $d$：task description

Teacher 提取的关键：critical decision points、correct reasoning、generalizable patterns。比如 ALFWorld 里 "go to fridge first for food items" 这类 principle。

对失败 trajectory $\tau^-$：

$$s^- = \mathcal{M}_T(\tau^-, d)$$

Teacher 提取 4 件事：(1) 哪步错了 (2) 错误 reasoning 是啥 (3) 应该怎么做 (4) 防止再犯的通用原则。

这就把冗长失败转成精炼 counterfactual lesson。比如"没拿 egg 就去 microwave 等了 5 步 → 应该先拿 object 再去 appliance → principle: No Appliance Before Object"。

人话类比：你不会背下整个考试答题过程，你会总结"看到这种题先列已知条件再套公式"。这就是 distillation。

### 招二：技巧卡片分两层放进手册

SKILLBANK 分两层：

$$\mathrm{SKILLBANK} = \mathcal{S}_g \cup \bigcup_{k=1}^{K} \mathcal{S}_k$$

- $\mathcal{S}_g$：通用技巧，跨所有任务适用。比如"系统搜索——每个位置搜过一次再回头"，"拿东西后直奔目的地别绕路"
- $\mathcal{S}_k$：特定任务技巧。比如 Heat 任务的"先开门再放再 heat"序列，Cool 任务的"object → fridge → countertop"顺序
- $K$：任务类别数

每个 skill 结构：name + principle + when to apply。

推理时检索（公式 4）：

$$\mathcal{S}_{\mathrm{ret}} = \mathrm{TopK}(\{s \in \mathcal{S}_k : \mathrm{sim}(e_d, e_s) > \delta\}, K)$$

- $e_d$：task description 的 embedding
- $e_s$：skill 的 embedding
- $\delta$：相似度阈值，paper 设 0.4
- $K$：top-K，paper 设 6

通用技巧 $\mathcal{S}_g$ 永远全塞进 context，特定技巧走 semantic similarity 检索。

然后 policy 在 skill-augmented context 下决策（公式 5）：

$$a_t \sim \pi_\theta(a_t | o_{\le t}, d, \mathcal{S}_g, \mathcal{S}_{\mathrm{ret}})$$

Token 效率：raw trajectory 平均 1450 tokens，蒸馏成 skill 后 1300 tokens，但信息密度高得多。10-20x compression。

人话类比：通用技巧像"开车注意事项"（适用所有路况），特定技巧像"雪地驾驶技巧"（只在雪天查）。你不会把整本《汽车驾驶手册》背下来，你按场景查相关章节。

### 招三：训练时手册自己进化

这是 SKILLRL 区别于所有 static skill library 方法的关键。

#### Cold-start SFT：先教模型用手册

base model 拿到手册也不会用——它从没学过"读到 skill → 按 skill 行动"这种模式。直接给 skill 没用（这有 ablation 证据）。

所以先用 teacher 生成 skill-augmented trajectory 当 SFT 数据：

$$\theta_{\mathrm{sft}} = \arg\min_\theta \mathcal{L}_{\mathrm{CE}}(\mathcal{D}_{\mathrm{SFT}}; \theta)$$

- $\mathcal{D}_{\mathrm{SFT}} = \{(d_i, \mathcal{S}_i, \tau_i^*)\}_{i=1}^N$：teacher 生成的 skill-augmented reasoning traces
- $d_i$：task description
- $\mathcal{S}_i$：retrieved skills
- $\tau_i^*$：理想 trajectory 示范如何用 skill
- $\mathcal{L}_{\mathrm{CE}}$：cross-entropy loss

SFT 后的 $\pi_{\theta_{\mathrm{sft}}}$ 同时当 RL 起点和 reference policy（KL 锚点）。

人话：先让师傅带徒弟走几遍流程，教会徒弟"看到手册第 X 条就该做 Y 动作"这个习惯。

#### Recursive Evolution：训练中手册更新

每 5 个 training step 做一次 validation。看每类任务 success rate $\mathrm{Acc}(C)$。若 $\mathrm{Acc}(C) < 0.4$，触发进化：

1. 收集失败的 validation trajectories $\mathcal{T}_{\mathrm{val}}^-$，用 diversity-aware stratified sampling（按 task category 分组，按失败严重度排序，round-robin 采样保持类别多样性）
2. Teacher 分析（公式 7）：

$$\mathcal{S}_{\mathrm{new}} = \mathcal{M}_T(\mathcal{T}_{\mathrm{val}}^-, \mathrm{SKILLBANK})$$

Teacher 三件事：(a) 找出当前 skill 没覆盖的 failure pattern (b) 提出新 skill 补 gap (c) 建议改进现有 ineffective skill

3. 更新：$\mathrm{SKILLBANK} \leftarrow \mathrm{SKILLBANK} \cup \mathcal{S}_{\mathrm{new}}$

训练初期手册 55 个 skill（12 general + 43 task-specific），训练结束 100 个（20 general + 80 task-specific）。

人话类比：你给实习生一本操作手册，他按手册干。每过一阵你看他在哪类任务老出错，针对性更新手册加新条目。手册与实习生共同进步。

#### RL 训练本身：GRPO

GRPO loss（公式 9）：

$$\mathcal{J}(\theta) = \mathbb{E}_{d, \{\tau^{(i)}\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min(\rho_i A_i, \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \right]$$

变量拆解：
- $\rho_i = \frac{\pi_\theta(\tau^{(i)} | d, \mathcal{S}_g, \mathcal{S}_{\mathrm{ret}})}{\pi_{\mathrm{old}}(\tau^{(i)} | d, \mathcal{S}_g, \mathcal{S}_{\mathrm{ret}})}$：新旧 policy 对同一 trajectory 概率比，注意是在 skill-augmented context 下算的
- $A_i = \frac{R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)}{\mathrm{std}(\{R_j\}_{j=1}^G)}$：normalized advantage，group 内相对好坏
- $R_i \in \{0, 1\}$：binary reward，任务成功=1 失败=0
- $G$：group size，paper 里 8
- $\epsilon$：clip ratio，限制 update step
- $\beta$：KL penalty 系数
- $\pi_{\mathrm{ref}} = \pi_{\theta_{\mathrm{sft}}}$：reference policy，锚到 SFT 后而非 base model

关键 design：KL 锚到 $\pi_{\theta_{\mathrm{sft}}}$ 而非 base model。这样 RL explore 时不会遗忘 SFT 阶段学到的 skill utilization capability。这是防止"学着学着忘了怎么用手册"。

GRPO 人话：每组采 8 个 trajectory，成功比平均好就强化，差就弱化。用 group mean 当 baseline 省了 critic network。clip + KL 双重防止跑偏。

GRPO 参考：https://arxiv.org/abs/2402.03300

---

## 实验数据怎么说

### ALFWorld + WebShop 主表

| 方法 | ALFWorld 平均 | WebShop 成功率 |
|---|---|---|
| GPT-4o | 48.0 | 23.7 |
| Gemini-2.5-Pro | 60.3 | 35.9 |
| Qwen2.5-7B (base) | 14.8 | 7.80 |
| ReAct | 31.2 | 19.5 |
| Reflexion | 42.7 | 28.8 |
| ExpeL | 46.3 | 11.2 |
| GRPO (vanilla) | 77.6 | 66.1 |
| Mem0+GRPO | 54.7 | 37.5 |
| EvolveR | 43.8 | 17.6 |
| **SKILLRL** | **89.9** | **72.7** |

关键数据点：

1. **SKILLRL 比 GRPO 高 12.3%**（77.6→89.9）。SKILLRL 本身就用 GRPO 当 optimizer，所以这 12.3% 完全归因于 skill augmentation。这是非常干净的对照实验——同一 optimizer，不同 context design。

2. **最大 gain 在 Cool (+23.0) 和 Pick2 (+22.8)**。Cool 任务需要 object → fridge → countertop 多步 precondition chain，base model 易漏步骤。skill 编码了 sequence constraint。

3. **比 Mem0+GRPO 高 35.2%**。Mem0+GRPO 是 competitive hybrid baseline（state-of-the-art memory mechanism + GRPO optimizer）。这 gap 证明 raw trajectory memory 不如 hierarchical skill abstraction。

4. **7B Qwen 经 SKILLRL 训练后超 GPT-4o 41.9%**。说明 structured experiential knowledge 可补偿 model scale。

### Search-augmented QA

| 方法 | NQ | TriviaQA | HotpotQA | Bamboogle | 平均 |
|---|---|---|---|---|---|
| Search-R1 | 39.3 | 61.0 | 37.0 | 36.8 | 38.5 |
| EvolveR | 43.5 | 63.4 | 38.2 | 54.4 | 43.1 |
| **SKILLRL** | **45.9** | 63.3 | **43.2** | **73.8** | **47.1** |

Bamboogle 上 SKILLRL 73.8% vs EvolveR 54.4%，gap 19.4%。Bamboogle 是 multi-hop reasoning benchmark，证明 hierarchical skill 对多步信息合成有效。

SKILLRL 只在 NQ + HotpotQA 训练，TriviaQA、2Wiki 是 OOD 但表现 competitive。这验证 skill 是 task-agnostic 的 transferable abstraction。

### Ablation（这是最有信息量的表）

| 配置 | ALFWorld | 变化 |
|---|---|---|
| SKILLRL full | 89.9 | - |
| w/o Hierarchical Structure | 76.8 | -13.1 |
| w/o Skill Library (用 raw trajectory) | 61.7 | -28.2 |
| w/o Cold-Start SFT | 65.2 | -24.7 |
| w/o Dynamic Evolution | 84.4 | -5.5 |

按贡献排序：**Skill abstraction > Cold-start SFT > Hierarchical > Dynamic evolution**

最关键的数据：**w/o Skill Library 用 raw trajectory 掉 28.2%**。这是 abstraction > memorization 假设的最强 evidence。memory 不是越多越好，signal density 才是关键。

第二关键：**w/o Cold-Start SFT 掉 24.7%**。base model 不会自发学 skill usage，必须显式 teach。这跟 instruction tuning 教 instruction following 类似，这里教 skill following。

### Figure 5: Evolution 动力学

- SKILLRL w/ evolution: 60 steps 达 80% success rate
- SKILLRL w/o evolution: 90 steps 才达更低 peak

Evolution 两大作用：(1) 加速收敛 (2) 提升 asymptotic ceiling。dynamic skill 帮 agent 跳出 local optima，static skill library 在某点 saturate。

---

## 几个直觉

### 1. 为什么 binary reward 也能 RL

$R_i \in \{0,1\}$ 看起来太粗，但 GRPO 的 advantage 是 group-relative。组内 3 个 success 5 个 failure，那 success 的 $A_i > 0$，failure 的 $A_i < 0$。group 越大 signal 越好（paper G=8）。绕过 reward shaping 难题。

### 2. 为什么 SKILLBANK 分两层

通用 skill 跨 task 适用但不够具体。特定 skill 细到解决 precondition 但不通用。两层互补：
- General: "拿东西后直奔目的地" 适用 Pick/Heat/Clean/Cool/Pick2 所有任务
- Task-specific: "Heat 任务先开 microwave 门" 只对 Heat 有效

不分层的话，要么太粗（全 general）要么太细（全 task-specific），都丢信息。

### 3. 为什么 evolution threshold δ=0.4

设太高（0.9）频繁触发但低质 skill——大多数 failure 已解决。太低（0.1）漏很多 failure mode。0.4 是 sweet spot：已有 decent performance 但仍有 significant failure population 可分析。

### 4. 为什么 diversity-aware stratified sampling

失败 trajectory 有多种 failure mode。若简单按 severity 取 top-M，可能全同 mode。stratified 按 category 分组，round-robin 保 categorical entropy，让 teacher 见到 diverse failure 才能生成 diverse skill。

人话：如果你只看"最严重"的失败，可能都是"找不到 object"这一类，teacher 只能生成"搜索技巧"。stratified 让 teacher 见各种失败才能生成各种 skill。

### 5. token 10-20x compression 怎么算

ALFWorld 一个 trajectory 50-100 actions × 5-10 tokens/action = 500-1000 tokens。蒸馏成 skill ≈ 50 tokens（name + principle + when to apply）。压缩 10-20x。

但 Figure 4 显示总 prompt 才从 1450 降到 1300，因为除了 skill 还有 task description、history、observation 等。skill 本身压缩大，但 prompt 里其他部分占大头。

### 6. Cool task +23% 的 root cause

Cool 任务步骤：找 object → 拿 → 去 fridge → cool → 去 countertop → 放。多步 precondition chain。base model 易漏"去 fridge"或"先 hold object 再去 appliance"。

skill `[coo_001]` 编码这 sequence。base model 没这 prior，靠 RL 从零学 sparse reward 慢。skill 给 strong prior，RL 只需 refine 而非 from scratch。

### 7. OOD 表现机制

训练 NQ + HotpotQA，TriviaQA 是 OOD。但 skill 是 "Iterative Refinement"、"Verify Early Abort Fast" 等 universal search strategy，不绑定特定 dataset。TriviaQA 是 single-hop 但有 OOD entity，skill 同样适用。

这印证核心 insight：abstraction is what transfers，instance 不 transfer。

---

## Case Study 看看 agent 实际怎么用 skill

### ALFWorld Heat: Egg → Microwave → Countertop

Core skills:
- `[hea_001]` Secure exact target first
- `[hea_003]` Open-Place-Heat sequence（先开门再放再 heat）
- `[hea_004]` No appliance before object（先拿 egg 再去 microwave）

Trajectory:
1. Search Countertop 1（无 egg）→ Countertop 2（找到 egg）
2. `take egg 1`（gen_002 Immediate Acquisition）
3. 去 Microwave 1
4. `open microwave 1`（hea_003）
5. `heat egg 1`
6. 去 Countertop 1
7. `put egg 1`

8 步完成，无 wasted action。base model 常见错误：先去 microwave 等，然后才找 egg，浪费 5 步。skill 编码了 action ordering constraint。

### ALFWorld Examine: Pencil + Desklamp

Skills:
- `[loo_001]` Seek lamp surfaces（Desk/Sidetable）
- `[exa_001]` Pair objects early（Target & Tool co-location）

Trajectory:
1. `go to desk 1`（desklamp 通常在 desk 上）
2. Observation: desk 1 上有 desklamp 1 + pencil 1（lucky co-location）
3. `take pencil 1`（exa_002 target before tool）
4. `use desklamp 1`

仅 3 步！这是 spatial reasoning 与 zero-waste navigation 的极致。若无 `[loo_001]`，agent 可能去 random location 搜，浪费 10 步。

### WebShop Abort-and-Retry

Task: portable bluetooth speaker, waterproof, black, <$30

1. Search "portable bluetooth speaker waterproof black under 30"
2. Results: [1] $28.99, [2] $19.99, [3] $35.50
3. Click [1]（gen_003 scan before click）
4. Colors: Blue, Red, Green, Army Green. **Black unavailable!**
5. `[gen_004] Verify Early, Abort Fast` → click Back to Search
6. Click [2] $19.99（gen_002 Iterative refinement）
7. Colors: Black, Blue, Red. Specs: IPX5 Waterproof
8. Click Black → $19.99 → Buy Now

关键：`gen_004 Verify Early, Abort Fast`。agent 不在 missing-color 页面纠结，立即 backtrack。这是 counterfactual reasoning 实例化——发现条件不满足马上换路径。

---

## 跟其他方法对比（人话版）

### vs Reflexion

Reflexion: agent 失败后 verbal self-reflection 写进 memory，下次 task 用。不更新参数。
SKILLRL: skill abstraction + SFT + RL co-evolution，参数也更新。
Reflexion ALFWorld 42.7% vs SKILLRL 89.9%。
Reflexion paper: https://arxiv.org/abs/2303.11366

人话：Reflexion 是"做完题写反思日记，下次考试翻日记"，SKILLRL 是"写解题套路卡 + 反复练题 + 卡片自动更新"。

### vs Voyager

Voyager: Minecraft agent，自动生成 skill library（code 形式）。不用 RL，用 curriculum + skill composition。
SKILLRL: text-based skill + RL co-evolution，适合 reasoning 任务。
Voyager: https://arxiv.org/abs/2305.16291

人话：Voyager 是程序员写 utility 函数累积代码库，SKILLRL 是学生攒解题套路卡然后刷题强化。

### vs ExpeL

ExpeL: 从 trajectory 蒸馏 insight 存 memory，inference 时 retrieval。无 RL。
SKILLRL: 多 RL co-evolution，skill 不仅 retrieve 还随 policy 一起进化。
ExpeL ALFWorld 46.3% vs SKILLRL 89.9%。
ExpeL: https://arxiv.org/abs/2308.13244

### vs MemRL

MemRL: RL 只更新 memory bank，policy frozen。不适应复杂环境。
SKILLRL: RL 同时 update policy 与 skill library。
MemRL ALFWorld 21.4% vs SKILLRL 89.9%。

人话：MemRL 是"只更新手册不培训员工"，SKILLRL 是"手册和员工一起进步"。

### vs EvolveR

EvolveR: joint update policy 与 memory，但 memory 仍是 raw trajectory 粗存。
SKILLRL: memory 是 hierarchical skill abstraction。
EvolveR ALFWorld 43.8% vs SKILLRL 89.9%。
EvolveR: https://arxiv.org/abs/2510.16079

人话：EvolveR 是"员工和操作录像一起进化"，SKILLRL 是"员工和精炼手册一起进化"。

---

## 我对这 paper 的思考

### 核心 insight

**Abstraction is the only thing that transfers.** Raw trajectory 是 instance，skill 是 abstraction。类比编程：你不会 memorize 每行 code，你学 pattern（design pattern、algorithm）。LLM agent 同理，需要 abstract principle 才能 generalize。

### 三个 mechanism 闭环

1. **Distillation**：instance → abstraction
2. **Hierarchical library**：让 abstraction 可组织、可检索
3. **Recursive evolution**：让 abstraction 持续 update 应对新 failure mode

### Cold-start SFT 是 critical bridge

base model 不天生会用 skill，SFT 教会"读到 skill → 按 skill 行动"。这跟 instruction tuning 教 instruction following 类似。ablation 显示 w/o SFT 掉 24.7%，第二大 drop。

这暗示一个更深的点：**skill 不是 parametric knowledge，skill utilization 才是**。skill 本身存在 SKILLBANK 里是 external，但"如何用 skill"这个能力是 parametric，得通过 SFT 编进 weights。

### KL 锚到 SFT policy 的深意

不只防 catastrophic forgetting，还保 skill utilization capability 不漂移。RL explore 时可能 drift 到不用 skill 的行为模式，KL 拉回来。这是 RL fine-tuning 的艺术——既要 explore 又要 don't forget。

类比：你教徒弟用手册，他练一阵可能自创野路子忘了看手册。KL 像你偶尔拉他回来看手册，保他养成查手册习惯。

### w/o Skill Library 掉 28% 的启示

这是最强 evidence 支持 abstraction > memorization。memory 不是越多越好，signal density 才是关键。

类比：你给学生 1000 道完整解题过程让他背，vs 给他 50 条解题套路。前者信息量大但学不到 pattern，后者信息量小但 transferable。

### 7B 超 GPT-4o 的意义

structured knowledge 可补偿 model scale。这指向未来 agent design 方向：不只 scale model，还要 scale structured experience。

这跟 retrieval-augmented generation (RAG) 思路类似，但 RAG retrieve document，SKILLRL retrieve skill。skill 是 document 的 abstraction，信息密度更高。

### Future work 方向

1. **Skill library pruning**：100 skill 持续增长，长期 bloat。应加 relevance decay 或 usage-based pruning。
2. **Dense reward**：binary reward 无法区分"almost success"与"完全失败"，dense reward 可能更好。
3. **Multi-agent skill sharing**：skill library 跨 agent 共享，是 modular agent design 雏形。
4. **Skill composition**：当前 skill 独立 retrieve，未来可探索 skill 间 dependency 与 composition。
5. **Catastrophic forgetting 评估**：跨 task RL 是否遗忘先前 task skill？paper 未做。

### 跟 Anthropic Agent Skills 的关系

paper Related Work 引用 Anthropic Claude 3 的 Agent Skills 概念（https://www.anthropic.com/news/claude-3-family）。Anthropic 提出skill 是 compact、reusable strategy 捕获 subtask essence。SKILLRL 把这概念 operationalize 成可训练、可进化的 library。

### 跟 continual learning 的关系

传统 continual learning (Parisi et al., 2019) 关注 predefined task 上 knowledge preservation。SKILLRL 的 recursive evolution 更像 open-ended continual learning——skill library 持续增长应对新场景。这是 continual learning 在 LLM agent 上的实例化。

### 跟 self-evolving agent 趋势

最近 self-evolving agent 方向（Gao et al., 2025; Xia et al., 2025）强调 agent 在 open-ended environment 主动获取 skill。SKILLRL 是这方向的 specific instantiation，用 RL + teacher distillation 实现 co-evolution。

Agent0 paper: https://arxiv.org/abs/2511.16043
Self-evolving survey: https://arxiv.org/abs/2507.21046

---

## 总结

SKILLRL 用人话就三步：

1. **别存录像，存套路卡**——把 raw trajectory 蒸馏成 hierarchical skill（通用+特定）
2. **先教模型用卡片**——cold-start SFT 让 base model 学会 skill utilization
3. **训练时卡片自动更新**——validation 失败触发 teacher 分析，skill library 与 policy co-evolve

效果：7B 小模型 + 好手册 > 大模型 + 没手册。说明 agent 的 capability 不只来自 model scale，还来自 structured experiential knowledge 的积累与组织。

这 paper 的意义在于：把"agent 如何从经验中学习"这个问题，从 naive memory storage 推进到 structured abstraction + co-evolution。这是 LLM agent 走向 continual learning 的关键一步。

GitHub: https://github.com/aiming-lab/SkillRL

---

# SKILLRL: Recursive Skill-Augmented Reinforcement Learning for LLM Agents

## 1. Paper 一句话总结

SKILLRL 解决 LLM agent 无法从过往经验中学习的根本问题. 现有 memory-based 方法把 raw trajectory 直接塞进 context window, 这存在严重 redundancy 与 noise 问题. SKILLRL 提出把 raw trajectory 通过 teacher model 蒸馏成 hierarchical SKILLBANK (general skills + task-specific skills), 通过 cold-start SFT 让 base model 学会使用 skill, 再通过 GRPO-based RL 训练, 训练过程中 skill library 与 policy co-evolve. 在 ALFWorld 上达到 89.9% success rate, 在 WebShop 上 72.7%, 平均超过 baseline 15.3%, 7B Qwen 甚至超过 GPT-4o 与 Gemini-2.5-Pro.

GitHub repo: https://github.com/aiming-lab/SkillRL

---

## 2. 背景: GRPO (Group Relative Policy Optimization)

GRPO 是 DeepSeek 提出的 RLHF 替代方案, 用于 SKILLRL 的 policy optimization. 理解 GRPO 对理解 SKILLRL 至关重要.

GRPO 与 PPO 最大差异在于不用 critic network, 用 intra-group relative reward. 公式 (1):

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{x, \{y_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(r_i A_i, \mathrm{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i\right) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \right]
$$

变量含义:
- $x$: input query (task description)
- $\{y_i\}_{i=1}^{G}$: model samples $G$ responses (group)
- $r_i = \frac{\pi_\theta(y_i|x)}{\pi_{\mathrm{old}}(y_i|x)}$: importance ratio, 衡量新旧 policy 对同一 trajectory 概率密度比
- $A_i = \frac{R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)}{\mathrm{std}(\{R_j\}_{j=1}^G)}$: normalized advantage, 用 group 内 reward 减去 mean 除以 std
- $\epsilon$: clip ratio, 限制 policy update step size, 防止太大破坏 stability
- $\beta$: KL penalty coefficient
- $\pi_{\mathrm{old}}$: policy before current update
- $\pi_{\mathrm{ref}}$: reference policy (通常 SFT model), 用 KL divergence 锚定防止 collapse

Intuition: GRPO 用 group mean 当 baseline 代替 critic, 这样省一个 critic network, 但需要 sample G 个 responses (paper 里 G=8). Advantage $A_i$ 是 group-relative 的: 比同 group 内其他 response 好 $A_i > 0$, 差则 $A_i < 0$. Clip 与 KL 是双重 regularization.

参考 DeepSeekMath: https://arxiv.org/abs/2402.03300

---

## 3. SKILLRL 框架三大核心组件

### 3.1 Experience-based Skill Distillation

传统做法只保留 success trajectory, SKILLRL 同时保留 success $\mathcal{T}^+$ 和 failure $\mathcal{T}^-$:

$$
\mathcal{T}^+ = \{\tau_i : r(\tau_i) = 1\}, \quad \mathcal{T}^- = \{\tau_i : r(\tau_i) = 0\}
$$

其中 $\tau_i = (o_0, a_0, r_0, ..., o_T, a_T, r_T)$ 是完整 trajectory, $r(\tau)$ 是 binary success indicator.

对 success 与 failure 做 differential processing:

For success trajectory $\tau^+ \in \mathcal{T}^+$ (公式 2):

$$
s^+ = \mathcal{M}_T(\tau^+, d)
$$

- $s^+$: distilled skill from success
- $\mathcal{M}_T$: teacher model (OpenAI o3)
- $\tau^+$: successful trajectory
- $d$: task description

Teacher 提取 critical decision points, correct reasoning, generalizable patterns.

For failure trajectory $\tau^- \in \mathcal{T}^-$ (公式 3):

$$
s^- = \mathcal{M}_T(\tau^-, d)
$$

Teacher 提取 4 个要素: (1) point of failure, (2) flawed reasoning/action, (3) what should have been done, (4) general principles to prevent similar failures. 这就把 verbose failure 转成 concise counterfactual lesson.

Intuition: failure 信息其实比 success 信息更 dense, 因为 success 经常有冗余 action, 而 failure 一定有 specific failure point. 把 failure 转成 counterfactual 类似于 importance sampling 的反面, 但 abstraction 层做.

### 3.2 Hierarchical Skill Library (SKILLBANK)

SKILLBANK 是 paper 的 central design, 分两层:

$$
\mathrm{SKILLBANK} = \mathcal{S}_g \cup \bigcup_{k=1}^{K} \mathcal{S}_k
$$

- $\mathcal{S}_g$: General Skills, universal strategic principles (e.g., systematic exploration, pre-action sanity check)
- $\mathcal{S}_k$: Task-Specific Skills for task category $k$ (e.g., "No Appliance Before Object" for ALFWorld Heat task)
- $K$: number of task categories

每个 skill $s \in \mathrm{SKILLBANK}$ 结构: name + principle + when to apply.

Skill Retrieval (公式 4):

$$
\mathcal{S}_{\mathrm{ret}} = \mathrm{TopK}(\{s \in \mathcal{S}_k : \mathrm{sim}(e_d, e_s) > \delta\}, K)
$$

变量:
- $e_d$: embedding of task description
- $e_s$: embedding of skill
- $\delta$: similarity threshold (paper 设 0.4)
- $K$: top-K retrieval (paper 设 6)

注意 $\mathcal{S}_g$ (general skills) 总是全部 retrieve, 只有 $\mathcal{S}_k$ 走 semantic similarity.

Policy conditioning (公式 5):

$$
a_t \sim \pi_\theta(a_t | o_{\le t}, d, \mathcal{S}_g, \mathcal{S}_{\mathrm{ret}})
$$

即 policy 在 skill-augmented context 下采样 action $a_t$.

Token efficiency: skill distillation 实现 10-20x token compression vs raw trajectory. Figure 4 显示平均 prompt 长度 1300 tokens vs raw memory 1450 tokens, 减少 10.3%, 但性能反而高. 这关键因为 abstraction 比 memorization 更 information-dense.

### 3.3 Recursive Skill Evolution

这是 SKILLRL 区别于 static skill library 方法的关键.

#### 3.3.1 Cold-Start SFT

base model 不天生会用 skill, 需要先 SFT teach:

$$
\theta_{\mathrm{sft}} = \arg\min_\theta \mathcal{L}_{\mathrm{CE}}(\mathcal{D}_{\mathrm{SFT}}; \theta)
$$

- $\mathcal{D}_{\mathrm{SFT}} = \{(d_i, \mathcal{S}_i, \tau_i^*)\}_{i=1}^N$: skill-augmented reasoning traces
- $d_i$: task description
- $\mathcal{S}_i$: retrieved skills
- $\tau_i^*$: ideal trajectory showing skill use
- $\mathcal{L}_{\mathrm{CE}}$: cross-entropy loss

$\pi_{\theta_{\mathrm{sft}}}$ 同时充当 RL 起点与 reference policy $\pi_{\mathrm{ref}}$.

Ablation 显示 w/o Cold-Start SFT: ALFWorld 65.2% (drop 24.7%), WebShop 46.5% (drop 26.2%). 这是单点最大的 ablation gap, 证明 SFT 是 critical step.

#### 3.3.2 Recursive Skill Evolution Mechanism

每个 validation epoch 后 (paper 里每 5 steps), 检查每类 task success rate $\mathrm{Acc}(C)$. 若 $\mathrm{Acc}(C) < \delta$ (paper 设 0.4), 则触发 evolution:

1. 收集 failed trajectories $\mathcal{T}_{\mathrm{val}}^- = \{\tau_j : r(\tau_j) = 0\}_{j=1}^M$ via diversity-aware stratified sampling (按 category 分组, 按 failure severity 排序, round-robin sampling 保持 categorical entropy)
2. Teacher 分析 (公式 7):

$$
\mathcal{S}_{\mathrm{new}} = \mathcal{M}_T(\mathcal{T}_{\mathrm{val}}^-, \mathrm{SKILLBANK})
$$

3. Update: $\mathrm{SKILLBANK} \leftarrow \mathrm{SKILLBANK} \cup \mathcal{S}_{\mathrm{new}}$

Teacher 3 个任务: (a) identify failure patterns not addressed by current skills, (b) propose new skills cover gaps, (c) suggest refinements for ineffective existing skills.

#### 3.3.3 RL-based Policy Optimization

GRPO loss (公式 9):

$$
\mathcal{J}(\theta) = \mathbb{E}_{d, \{\tau^{(i)}\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min(\rho_i A_i, \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}}) \right]
$$

变量:
- $\rho_i = \frac{\pi_\theta(\tau^{(i)} | d, \mathcal{S}_g, \mathcal{S}_{\mathrm{ret}})}{\pi_{\mathrm{old}}(\tau^{(i)} | d, \mathcal{S}_g, \mathcal{S}_{\mathrm{ret}})}$: importance ratio over skill-augmented context
- $A_i = \frac{R_i - \mathrm{mean}(\{R_j\}_{j=1}^G)}{\mathrm{std}(\{R_j\}_{j=1}^G)}$: normalized advantage, 公式 (8), $R_i \in \{0, 1\}$ 是 binary reward
- $\pi_{\mathrm{ref}} = \pi_{\theta_{\mathrm{sft}}}$: 锚定 SFT 后的 policy

关键: KL 锚到 SFT 后的 policy, 而非 base model. 这保证 RL 不会遗忘 SFT 阶段学到的 skill utilization capability. 这是个重要 design choice, 类似 DPO 中 reference model 的作用, 但作用是 skill-aware 的 stabilization.

Algorithm 1 完整流程:
1. Rollout base model 收集 trajectories
2. Teacher 蒸馏 success/failure 成 skills
3. 构建 SKILLBANK (general + task-specific)
4. Cold-start SFT 用 teacher 生成的 skill-augmented trajectories
5. RL loop: retrieve skills → sample G trajectories → GRPO update → 每 5 step validation → 若 SR < 0.4 触发 skill evolution

---

## 4. 实验 Setup

### 4.1 Environments

- **ALFWorld**: text-based household task (6 task types: Pick, Look, Clean, Heat, Cool, Pick2)
- **WebShop**: web shopping simulation (product search + constraint satisfaction + purchase)
- **Search-augmented QA**: 7 datasets
  - Single-hop: NQ, TriviaQA, PopQA
  - Multi-hop: HotpotQA, 2Wiki, MuSiQue, Bamboogle

### 4.2 Baselines

4 categories:
1. **Closed-source LLMs**: GPT-4o, Gemini-2.5-Pro
2. **Prompt-based / Memory-based**: ReAct, Reflexion, Mem0, ExpeL, MemP, SimpleMem
3. **RL-based**: RLOO, GRPO
4. **Memory-augmented RL**: MemRL, EvolveR, Mem0+GRPO, SimpleMem+GRPO

### 4.3 Implementation

- Base model: Qwen2.5-7B-Instruct
- Teacher model: OpenAI o3
- RL: GRPO, lr $1 \times 10^{-6}$, batch size 16, group size 8, 4 gradient accumulation
- $K=6$ task-specific skill retrieval
- $\delta=0.4$ evolution threshold
- 8x NVIDIA H100 80GB
- Total ~30 hours per experiment

---

## 5. Main Results 分析

### 5.1 Table 1: ALFWorld + WebShop

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | All | WebShop Score | WebShop Succ |
|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 75.3 | 60.8 | 31.2 | 56.7 | 21.6 | 49.8 | 48.0 | 31.8 | 23.7 |
| Gemini-2.5-Pro | 92.8 | 63.3 | 62.1 | 69.0 | 26.6 | 58.7 | 60.3 | 42.5 | 35.9 |
| Qwen2.5-7B (base) | 33.4 | 21.6 | 19.3 | 6.90 | 2.80 | 3.20 | 14.8 | 26.4 | 7.80 |
| ReAct | 48.5 | 35.4 | 34.3 | 13.2 | 18.2 | 17.6 | 31.2 | 46.2 | 19.5 |
| Reflexion | 62.0 | 41.6 | 44.9 | 30.9 | 36.3 | 23.8 | 42.7 | 58.1 | 28.8 |
| Mem0 | 54.0 | 55.0 | 26.9 | 36.4 | 20.8 | 7.69 | 33.6 | 23.9 | 2.00 |
| ExpeL | 21.0 | 67.0 | 55.0 | 52.0 | 71.0 | 6.00 | 46.3 | 30.9 | 11.2 |
| MemP | 54.3 | 38.5 | 48.1 | 56.2 | 32.0 | 16.7 | 41.4 | 25.3 | 6.40 |
| SimpleMem | 64.5 | 33.3 | 20.0 | 12.5 | 33.3 | 3.84 | 29.7 | 33.2 | 8.59 |
| RLOO | 87.6 | 78.2 | 87.3 | 81.3 | 71.9 | 48.9 | 75.5 | 80.3 | 65.7 |
| GRPO | 90.8 | 66.1 | 89.3 | 74.7 | 72.5 | 64.7 | 77.6 | 79.3 | 66.1 |
| MemRL | 62.8 | 38.5 | 22.2 | 12.5 | 8.00 | 0.00 | 21.4 | 29.5 | 9.20 |
| EvolveR | 64.9 | 33.3 | 46.4 | 13.3 | 33.3 | 33.3 | 43.8 | 42.5 | 17.6 |
| Mem0+GRPO | 78.1 | 54.8 | 56.1 | 31.0 | 65.0 | 26.9 | 54.7 | 58.1 | 37.5 |
| SimpleMem+GRPO | 89.5 | 36.3 | 60.0 | 50.0 | 64.9 | 26.3 | 62.5 | 67.8 | 46.9 |
| **SKILLRL** | **97.9** | **71.4** | **90.0** | **90.0** | **95.5** | **87.5** | **89.9** | **85.2** | **72.7** |

关键观察:

1. SKILLRL 比 GRPO 高 12.3% (77.6 → 89.9), 但 SKILLRL 本身用 GRPO 作 optimizer, 所以这 12.3% 完全归因于 skill augmentation. 这是个非常 clean 的 ablation: 同一 optimizer, 不同 context design.

2. 最大 gain 在 Cool (+23.0) 与 Pick2 (+22.8). Cool 任务需要 multi-step (object → fridge → microwave → countertop), Pick2 需要 sequential two-object tracking. 这些是 sparse-reward long-horizon 任务, skill prior 最有价值.

3. 比 Mem0+GRPO (54.7) 高 35.2%. Mem0+GRPO 是 competitive hybrid baseline. 这 35.2% gap 是 memory mechanism design 差异: raw trajectory memory vs hierarchical skill abstraction.

4. 7B Qwen 经 SKILLRL 训练后超过 GPT-4o 41.9%, 超过 Gemini-2.5-Pro 29.6%. 这说明 structured experiential knowledge 可补偿 model scale.

### 5.2 Table 2: Search-augmented QA

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5 (base) | 11.6 | 35.6 | 1.20 | 16.4 | 22.2 | 4.80 | 14.4 | 15.2 |
| CoT | 12.8 | 35.6 | 3.80 | 16.2 | 22.6 | 6.60 | 24.0 | 17.4 |
| RAG | 27.4 | 58.2 | 17.8 | 25.8 | 23.2 | 9.40 | 16.8 | 25.5 |
| Search-o1 | 19.4 | 40.6 | 11.4 | 17.0 | 27.0 | 8.60 | 30.4 | 22.1 |
| R1-Instruct | 21.0 | 44.9 | 17.1 | 20.8 | 27.5 | 6.00 | 19.2 | 22.4 |
| Search-R1 | 39.3 | 61.0 | 39.7 | 37.0 | 40.1 | 14.6 | 36.8 | 38.5 |
| ZeroSearch | 43.6 | 61.8 | 51.5 | 34.6 | 35.2 | 18.4 | 27.8 | 39.1 |
| StepSearch | - | - | - | 38.6 | 36.6 | 22.6 | 40.0 | - |
| EvolveR | 43.5 | 63.4 | 44.6 | 38.2 | 42.0 | 15.6 | 54.4 | 43.1 |
| **SKILLRL** | **45.9** | 63.3 | **45.9** | **43.2** | 40.3 | 20.2 | **73.8** | **47.1** |

关键:
1. Bamboogle 上 SKILLRL 达 73.8%, 比 EvolveR 高 19.4%. Bamboogle 是 multi-hop reasoning benchmark, 这 gap 证明 hierarchical skill 对 multi-step information synthesis 有效.
2. SKILLRL 仅在 NQ + HotpotQA 训练, 但在 TriviaQA, 2Wiki 等 OOD dataset 表现 competitive. 这验证 generalizable skill abstraction 的 transfer 性.

### 5.3 Table 3: Ablation Study

| Method | ALFWorld | WebShop |
|---|---|---|
| SKILLRL (full) | 89.9 | 72.7 |
| w/o Hierarchical Structure | 76.8 | 61.4 |
| w/o Skill Library (Raw Trajectories) | 61.7 | 50.2 |
| w/o Cold-Start SFT | 65.2 | 46.5 |
| w/o Dynamic Evolution | 84.4 | 70.3 |

各组件贡献分析:

- **w/o Hierarchical Structure** (-13.1 ALFWorld, -11.3 WebShop): 仅 task-specific skill 而无 general skill. 说明 universal strategic principle 提供基础 guidance, 是 transferable 跨 task 的.
- **w/o Skill Library (Raw Trajectories)** (-28.2 ALFWorld, -22.5 WebShop): 最大 drop. 直接证明 abstraction > memorization. Raw trajectory 含 exploratory action, backtracking, redundancy, 噪声淹没有效 signal.
- **w/o Cold-Start SFT** (-24.7 ALFWorld, -26.2 WebShop): 第二大 drop. Base model 不会自发学会用 skill, 必须显式 teach skill retrieval + interpretation + application.
- **w/o Dynamic Evolution** (-5.5 ALFWorld, -2.4 WebShop): 最小但仍显著. 静态 SKILLBANK 不够, co-evolution 应对 emergent failure modes.

按 ablation 贡献排序: Skill abstraction > Cold-start SFT > Hierarchical structure > Dynamic evolution.

### 5.4 Figure 3: Skill Library Growth

Initial: 55 skills (12 general + 43 task-specific)
Step 150: 100 skills (20 general + 80 task-specific)

Growth pattern:
- Task-specific skill 增长 dominant (43 → 80, +86%)
- General skill steady 增长 (12 → 20, +67%)
- Balanced cross-category expansion

这表明 dynamic evolution 主要补 task-specific gap, general skill 一开始已覆盖核心 universal principle.

### 5.5 Figure 4: Context Efficiency

- Raw memory baseline: ~1,450 tokens, 高 variance
- SKILLRL: ~1,300 tokens 平均, 10.3% 减少

但更关键的是 SKILLRL 用更少 context 达更高性能. 这反驳 "context 越多越好" 的 naive 假设. 信号 density 比信号 volume 更重要.

### 5.6 Figure 5: Evolution Dynamics

- SKILLRL w/ evolution: 80% success rate 在 60 steps
- SKILLRL w/o evolution: 需 90 steps 达更低 peak

Evolution 的两大作用:
1. 加速收敛 (60 vs 90 steps)
2. 提升 asymptotic performance (更高 ceiling)

这说明 dynamic skill 引入 + existing skill refinement 帮助 agent 跳出 local optima, 而 static skill library 在某一时刻 saturate.

---

## 6. SKILLBANK 实例 (Appendix C)

### 6.1 ALFWorld General Skills

| ID | Title | Principle | When to Apply |
|---|---|---|---|
| gen_001 | Systematic Exploration | Search every plausible surface/container once before revisiting; prioritize unseen locations. | Goal count not met and unexplored areas remain. |
| gen_002 | Immediate Acquisition | Take object immediately when visible/reachable. | First visual confirmation of goal-relevant object. |
| gen_003 | Destination First Policy | After picking up goal object, navigate directly to target receptacle and place it. | Holding any goal object while target location identified. |
| gen_005 | Use State-Changing Tools Early | Acquire object, then immediately use nearest appliance (heat/cool/clean) before placement. | After picking up object needing temp/cleanliness change. |
| gen_006 | Establish Spatial Relations | First locate reference object, adjust state if needed, then search/place in specified region. | Tasks with prepositions "under", "inside", "on". |
| gen_014 | Loop Escape Trigger | If last 3-5 actions don't change state, switch to untried search branch. | After consecutive no-progress observations. |
| gen_015 | Pre-Action Sanity Check | Confirm prerequisites (hand free, capacity, power) before manipulative commands. | Before commands that could legally fail. |

### 6.2 ALFWorld Common Failures

| ID | Failure | Root Cause | Mitigation |
|---|---|---|---|
| err_001 | Redundant Revisit | Lacks explicit memory of explored areas; strategy degenerates into local loops. | Maintain exploration map; prioritize unvisited candidates. |
| err_006 | Skipping State Changes | Conflates object presence with goal satisfaction; omits cleanliness/temp checks. | Integrate state precondition checks into planner before placement. |

### 6.3 WebShop General Skills

| ID | Title | Principle | When to Apply |
|---|---|---|---|
| gen_001 | Prioritize Core Keywords | Include product type, 1-2 functional attributes, hard constraints; omit secondary descriptors. | Before first search or refining over-specific queries. |
| gen_002 | Iterative Refinement | Adjust keywords or apply site filters instead of repeating failed query. | Results irrelevant or repeat despite multiple searches. |
| gen_003 | Scan Before You Click | Read titles, thumbnails, prices in results to ensure plausibility before opening link. | Search results page choosing next product to inspect. |
| gen_004 | Verify Early, Abort Fast | Immediately check category, attributes, price on product page; leave if any constraint violated. | First observation on every product detail page. |
| gen_006 | Confirm Hidden Attributes | Open Description/Features sections to ensure non-visible specs (material) meet constraints. | Constraints not evident from title or variant list. |
| gen_007 | Check Variant Pricing | For price ranges, select exact variant combination to verify specific price within budget. | Whenever price changes with variant selection or shows as range. |
| gen_013 | Purchase Decisively | Execute 'Buy Now' immediately once all constraints and prices confirmed on variant. | After validating every constraint on current product variant. |

---

## 7. Case Study 解析 (Appendix D)

### 7.1 ALFWorld Heat Task: Egg → Microwave → Countertop

Core skills invoked:
- `[hea_001]` Secure exact target first
- `[hea_003]` Open-Place-Heat sequence (先开 microwave 门, 再放, 再 heat)
- `[hea_004]` No appliance before object (先拿 egg 再去 microwave, 避免空等)

Trajectory:
1. Search Countertop 1 (no egg) → Countertop 2 (found egg)
2. `take egg 1` (gen_002 Immediate Acquisition)
3. Navigate to Microwave 1
4. `open microwave 1` (hea_003)
5. `heat egg 1` (hea_005 Direct Post-Heat Placement)
6. Navigate to Countertop 1
7. `put egg 1`

8 steps 完成, 无 wasted action. 关键: skill 编码了 action ordering constraint, 避免 base model 常见错误 (先去 microwave 等, 然后才找 egg).

### 7.2 ALFWorld Examine Task: Pencil + Desklamp

Core skills:
- `[loo_001]` Seek lamp surfaces (Desk/Sidetable)
- `[loo_002]` Switch lamp on
- `[exa_001]` Pair objects early (Target & Tool co-location)
- `[exa_004]` Single toggle rule

Trajectory:
1. `go to desk 1` (reasoning: desklamps usually on desks)
2. Observation: desk 1 has desklamp 1 + pencil 1 (lucky co-location, gen_006 spatial reasoning)
3. `take pencil 1` (exa_002 target before tool)
4. `use desklamp 1` (loo_002 toggle state)

仅 3 steps 完成! 这是 spatial reasoning 与 zero-waste navigation 的极致体现. 若无 `[loo_001]`, agent 可能去 random location 搜.

### 7.3 WebShop Abort-and-Retry

Task: portable bluetooth speaker, waterproof, black, <$30

1. Search "portable bluetooth speaker waterproof black under 30"
2. Results: [1] $28.99 (Waterproof Speaker), [2] $19.99 (Mini LED Speaker), [3] $35.50
3. Click [1] (gen_003 scan before click, gen_004 verify early)
4. Colors: Blue, Red, Green, Army Green. Black unavailable!
5. `[gen_004] Verify Early, Abort Fast` → click Back to Search
6. Click [2] $19.99 (gen_002 Iterative refinement)
7. Colors: Black, Blue, Red. Specs: IPX5 Waterproof
8. Click Black
9. Final price $19.99. All constraints verified → Buy Now

关键 skill: `gen_004 Verify Early, Abort Fast`. Agent 不在 missing-color 页面继续纠结, 立即 backtrack. 这是 counterfactual reasoning 实例化.

---

## 8. Algorithm 1 完整 Pseudocode 解读

```
Input: π_base, teacher M_T, environment E
Output: π_θ*, evolved SKILLBANK*

// Phase 1: Experience-based Skill Distillation
1. T+, T- ← Rollout(π_base, E)  // 收集 diverse trajectory
2. for τ+ ∈ T+: s+ ← M_T(τ+)  // 成功提炼
3. for τ- ∈ T-: s- ← M_T(τ-)  // 失败提炼

// Phase 2: Hierarchical Skill Library Construction
4. S_g ← general skills
5. for k in task types: S_k ← task-specific skills
6. SKILLBANK ← S_g ∪ ⋃_k S_k

// Phase 3: Cold-start Initialization
7. D_SFT ← M_T(E, SKILLBANK)  // teacher 生成 skill-augmented demonstrations
8. θ ← SFT(π_base, D_SFT); π_ref ← π_θ

// Phase 4: RL with Recursive Evolution
9. for epoch = 1 to N:
10.   for d in tasks:
11.     S_ret ← Retrieve(d, SKILLBANK)  // semantic TopK
12.     Sample {τ^(i)}_{i=1}^G ~ π_θ(·|d, S_g, S_ret)
13.     Compute {R_i}; update θ via GRPO
14.   if validation epoch:
15.     T_val^- ← failed validation trajectories  // diversity-aware sampling
16.     S_new ← M_T(T_val^-, SKILLBANK)  // teacher 分析 gap
17.     SKILLBANK ← SKILLBANK ∪ S_new
18. return π_θ, SKILLBANK
```

注意 4 个阶段: Distillation → Construction → Cold-start SFT → RL with Evolution. 每阶段不可缺 (ablation 验证).

---

## 9. 9 个细节 Intuition Building

1. **为什么 binary reward 也能 RL?** SKILLRL 用 binary success $R_i \in \{0,1\}$. GRPO advantage 是 group-relative, 即使 binary reward, group 内 success vs failure 比 mean 大小也能产生 useful advantage signal. 这绕过 reward shaping 难题.

2. **为什么 SKILLBANK 分两层?** General skill (e.g., systematic exploration) 跨 task 适用, 但不细到解决具体 task 的 precondition. Task-specific skill (e.g., heat task 的 "open microwave before heating") 编码 domain knowledge. 两层补足, 不冗余.

3. **为什么 KL 锚到 SFT 后 policy?** Base model 不会用 skill, SFT 教会. RL 若自由探索可能遗忘 skill usage capability. 锚到 $\pi_{\theta_{\mathrm{sft}}}$ 在 update 时拉回, 保 skill utilization 不漂移.

4. **为什么 diversity-aware stratified sampling?** 失败 trajectory 有多种 failure mode. 若按 severity 简单取 top-M, 可能全同 mode. Stratified 按 category 分, round-robin 保 categorical entropy, 让 teacher 见到 diverse failure 才能生成 diverse skill.

5. **为什么 evolution threshold δ=0.4?** 太高 (e.g., 0.9) 频繁触发但低质 skill (大多数 failure 已解决). 太低 (e.g., 0.1) 漏很多 failure mode. 0.4 是 sweet spot: 已有 decent performance 但仍有 significant failure population 可分析.

6. **token compression 10-20x 如何算?** ALFWorld 一个 trajectory 平均 50-100 actions, 每 action 5-10 tokens, total 500-1000 tokens. 蒸馏成 skill ≈ 50 tokens (name + principle + when). 压缩比 10-20x.

7. **为什么 SKILLRL 比 Mem0+GRPO 高 35.2%?** Mem0 存 raw trajectory, retrieval 时塞整个 trajectory 进 context. Agent 难 extract 关键 action, 被 noise 干扰. SKILLRL 抽象成 principle + when to apply, signal density 高, retrieval 也精确.

8. **cool task +23% 的 root cause?** Cool task 需要: 找 object → 拿 → 去 fridge → cool → 去 countertop → 放. 多步 precondition chain. base model 易遗漏 "去 fridge" 或 "先 hold object 再去 appliance". skill `[coo_001]` 编码这 sequence, 大幅提升.

9. **OOD 表现 (TriviaQA, 2Wiki) 的 mechanism?** 训练在 NQ + HotpotQA. skill 是 "Iterative Refinement", "Verify Early Abort Fast" 等 universal search strategy, 不绑定特定 dataset. TriviaQA 是 single-hop 但有 OOD entity, skill 同样适用.

---

## 10. 与相关工作对比

### 10.1 vs Reflexion (Shinn et al., 2023)

Reflexion: verbal self-reflection 后存进 memory, 下次 task 用. 不更新参数. 
SKILLRL: skill abstraction + SFT + RL co-evolution, 参数也更新.
Reflexion ALFWorld 42.7% vs SKILLRL 89.9%.

Reflexion paper: https://arxiv.org/abs/2303.11366

### 10.2 vs Voyager (Wang et al., TMLR)

Voyager: Minecraft agent, 自动生成 skill library (code form). 不用 RL, 用 curriculum + skill composition.
SKILLRL: text-based skill + RL co-evolution, 适合 reasoning 任务.
Voyager: https://arxiv.org/abs/2305.16291

### 10.3 vs ExpeL (Zhao et al., AAAI 2024)

ExpeL: 从 trajectory 蒸馏 insight, 存 memory, inference 时 retrieval. 无 RL.
SKILLRL: 多 RL co-evolution, skill 不仅 retrieve 还 co-evolve.
ExpeL ALFWorld 46.3% vs SKILLRL 89.9%.
ExpeL: https://arxiv.org/abs/2308.13244

### 10.4 vs MemRL (Zhang et al., 2026)

MemRL: RL 只更新 memory bank, policy frozen. 不适应复杂环境.
SKILLRL: RL 同时 update policy 与 skill library (co-evolve).
MemRL ALFWorld 21.4% vs SKILLRL 89.9%.

### 10.5 vs EvolveR (Wu et al., 2025)

EvolveR: joint update policy 与 memory, 但 memory 仍是 raw trajectory 粗存储.
SKILLRL: memory 是 hierarchical skill abstraction.
EvolveR ALFWorld 43.8% vs SKILLRL 89.9%.
EvolveR: https://arxiv.org/abs/2510.16079

---

## 11. Limitations 与 Future Work 推测

1. **Teacher model 依赖**: 用 OpenAI o3 蒸馏 skill 与 SFT data, 不可 fully reproducible.
2. **Skill retrieval 用 cosine similarity**: 语义相似 ≠ task 适用性, 可能 retrie 错 skill.
3. **Skill library 无 forgetting 机制**: 100 skill 持续增长, 长期可能 bloat. 应加 pruning 或 relevance decay.
4. **Binary reward**: 无法区分 "almost success" 与 "完全失败", dense reward 可能更好.
5. **No catastrophic forgetting 评估**: 跨 task RL 是否遗忘先前 task skill? Paper 未做.

---

## 12. 关键 Web Links

- **SKILLRL GitHub**: https://github.com/aiming-lab/SkillRL
- **GRPO (DeepSeekMath)**: https://arxiv.org/abs/2402.03300
- **Qwen2.5 Technical Report**: https://arxiv.org/abs/2309.16609
- **ALFWorld**: https://arxiv.org/abs/2010.03768
- **WebShop**: https://arxiv.org/abs/2207.01294
- **ReAct**: https://arxiv.org/abs/2210.03629
- **Reflexion**: https://arxiv.org/abs/2303.11366
- **Voyager**: https://arxiv.org/abs/2305.16291
- **ExpeL**: https://arxiv.org/abs/2308.13244
- **Mem0**: https://arxiv.org/abs/2504.19413
- **EvolveR**: https://arxiv.org/abs/2510.16079
- **Search-R1**: https://arxiv.org/abs/2503.09516
- **ZeroSearch**: https://arxiv.org/abs/2505.04588
- **PPO**: https://arxiv.org/abs/1707.06347
- **Anthropic Agent Skills (Claude 3)**: https://www.anthropic.com/news/claude-3-family

---

## 13. 我的 Intuition Building 总结

SKILLRL 的 central insight: **abstraction is the only thing that transfers**. Raw trajectory 是 instance, skill 是 abstraction. 类比 programming: 你不会 memorize 每行 code, 你学 pattern (design pattern, algorithm). LLM agent 同理, 它需要 abstract principle 才能 generalize.

3 个 mechanism 形成闭环:
1. **Distillation** 把 instance → abstraction
2. **Hierarchical library** 让 abstraction 可组织, 可检索
3. **Recursive evolution** 让 abstraction 持续 update 应对新 failure mode

Cold-start SFT 是 critical bridge, 因为 base model 不会自发学 skill usage. 这与 instruction tuning 类似, 但 instruction tuning 教 instruction following, 这里教 skill following.

KL regularization 锚到 SFT 后 policy 是 deep insight: 不仅防 catastrophic forgetting, 还保 skill utilization capability 不漂移. 这是 RL fine-tuning 的艺术 - 既要 explore, 又要 don't forget.

最有意思的 ablation: w/o skill library (raw trajectory) drop 25%, 这是 abstraction 假设的最强 evidence. memory 不是越多越好, signal density 才是关键.

7B Qwen 超 GPT-4o 41.9% 表明: structured knowledge 可 compensate scale. 这指向未来 agent design 方向: 不是只 scale model, 而是 scale structured experience.

对未来研究启示: skill library 可视为 "external cortex", 与 model parameter 解耦. 这让 continual learning 成为可能 (新 skill add 无需 retrain). 也让 multi-agent knowledge sharing 成为可能 (skill library 跨 agent 共享). 这是 modular agent design 的雏形.
