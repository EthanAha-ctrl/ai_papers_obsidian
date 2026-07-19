---
source_pdf: AUTORT EMBODIED FOUNDATION MODELS FO.pdf
paper_sha256: bd001b5d9f6259eb3c4afaeb3cbaba881c961d9eeda59091b492fe85b0d748b5
processed_at: '2026-07-18T12:09:25-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents

Andrej，下面我把这篇 Google DeepMind 的工作拆开讲透，重点是帮你 build intuition——为什么用 LLM/VLM 当 orchestrator 去驱动一支 robot fleet 在 unseen building 里 self-propose 任务并混合 teleop + autonomous 采数据，整个 stack 是怎么串起来的，每个公式和 prompt 在干什么，以及它跟 SayCan / RT-2 / Voyager / Constitutional AI 这条线怎么连。

参考链接:
- AutoRT (arXiv): https://arxiv.org/abs/2401.02132
- SayCan (Ahn et al. 2022): https://arxiv.org/abs/2204.01616
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- PaLM-E: https://arxiv.org/abs/2303.03378
- Voyager: https://arxiv.org/abs/2305.16291
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Reflexion: https://arxiv.org/abs/2303.11366
- ReAct: https://arxiv.org/abs/2210.03629
- Universal Sentence Encoder: https://arxiv.org/abs/1803.11175
- Natural Language Maps (Chen et al. ICRA 2023): https://arxiv.org/abs/2210.03033 (approx; open-vocab queryable scene repr)
- FlexCap (Dwibedi et al. 2024): https://arxiv.org/abs/2404.02686 (approx)

---

## 1. 核心问题：Data 是 bottleneck

 embodied foundation model 的痛点是 web 上有海量 vision-language 数据，但 grounded in physical world 的 action data 极度稀缺。RT-1 / RT-2 这条路线已经证明 internet-scale VLM 知识能 transfer 到 robot control，但是 real-world robot episodes 仍然只能靠 lab 里 teleop 一条条攒，scaling 被 human availability 卡死。

 AutoRT 想做的事：把 robot 放到一个完全 unseen 的 building，给它一个 high-level guidance（比如 "collect gardening tasks"），让 foundation model 自己看 scene、自己 propose 任务、自己 decide 用哪个 collect policy（teleop / scripted pick / RT-2），并且在 1 human : 3–5 robots 的比例下持续产出 diverse episodes。7 个月、4 栋楼、20+ robots、77k episodes、6650+ unique instructions。

这跟 Voyager 在 Minecraft 里 self-explore skill library 是一个 family，区别在于 AutoRT 把这件事搬到 real world，必须处理 safety、reliability、human-in-the-loop、unseen layout 这些 Voyager 不用面对的东西。

---

## 2. 系统架构：Policy Graph

实现上他们把整个 pipeline 表达成一个 **policy graph**。每个 node $\nu \in V$ 是一个 subpolicy

$$\pi(a \mid s, \text{data})$$

其中 $s$ 是 robot state，$a$ 是 action，$\text{data}$ 是沿着 graph 累积的 context（比如 VLM caption、proposed tasks、affordance decision）。node 之间用 transition condition

$$\beta: S \times \text{Data} \rightarrow \{0,1\} \times V$$

决定什么时候 yield control 到下一个 node、yield 到哪个。不动机器的 node（pure LLM query）输出 no-op action。这本质就是一个 LLM-augmented finite-state machine，跟 SayCan 的 affordance gating + RT-2 execution 是同构的，只不过 AutoRT 在 FSM 顶层加了 "task proposal" 这个 self-driven 节点，而 SayCan 是 user 给 instruction、LLM 只做 planning。

Pipeline 顺序（对应 Fig.1 / Fig.5）:

1. **Exploration / Navigation**：用 natural language map 把环境 encode 成一组 visual-language embedding $\phi_i$ 带位姿 $(x_i, y_i, z_i)$，sample 一个 navigation target。
2. **Scene description**：到位后拍一张图，VLM（PaLI 或 FlexCap）输出 caption + 5 个 object 名字。
3. **Task generation**：LLM（这里是 PaLM/Gemini 类，prompted not fine-tuned）读 constitution + scene caption + object list + 当前 collect policy 的 suffix，输出 N 个 candidate task strings。
4. **Affordance filter**：另一个 LLM call，self-critique，把每个 task 分类到 $\{\pi^{\text{teleop}}, \pi^{\text{script}}, \pi^{\text{rt2}}, \text{reject}\}$，附 reject reason。
5. **Execute**：从 accepted task 里 random sample 一个，run 对应 policy。
6. **Diversity scoring**：episode 结束后用 CLIP-based embedder 算 visual diversity，回传给人 supervisor 当 online feedback。
7. Reset，loop。

---

## 3. Navigation Sampling：怎么挑下一个去哪

环境用 Chen et al. 2023 的 natural language map 表示：robot 在 SLAM 过程中每个 detection 存一个 VLM embedding $\phi_i \in \mathbb{R}^d$ 和 3D 位置 $(x_i, y_i, z_i)$。给一个 text query $q$，找 $\phi_i$ 离 text embedding 最近的去 navigate。

但 AutoRT 不想每次都去同一个 "sponge"，所以它定义了一个固定的 query embedding $\phi_q$ = 一组 hardcoded object name（apple, basket, bowl, fridge, ... 一百多个，见 Appendix B）的 normalized average text embedding。然后对 map 里每个 candidate target $\phi_i$ 算 score:

$$\text{score}_i = \frac{\phi_i \cdot \phi_q - \min_j(\phi_j \cdot \phi_q)}{\max_j(\phi_j \cdot \phi_q) - \min_j(\phi_j \cdot \phi_q)}$$

变量解释:
- $\phi_i \in \mathbb{R}^d$：第 $i$ 个 map landmark 的 visual-language embedding（unit norm）。
- $\phi_q \in \mathbb{R}^d$：固定 query 的 averaged text embedding（unit norm）。
- $\phi_i \cdot \phi_q$：cosine similarity（因为都 normalized）。
- 分母是 min-max normalization，把 similarity 拉到 $[0,1]$。

然后按 $\text{score}_i^{\beta}$ 的概率采样 destination，$\beta$ 是 temperature。$\beta=1$ 用于 broad collection，更大 $\beta$ 用于 targeted collection。这个 trick 让 robot 偏向 "看起来像常见可操作物体" 的区域，又不至于 greedy 到死磕一个点。这跟 SayCan 里用 affordance probability 做 top-k sampling 是一个 flavor，只不过这里是在 navigation latent space 里做。

---

## 4. Robot Constitution：把 Asimov prompt 化

这是 AutoRT 最有意思的部分，直接借 Constitutional AI 的思路，把 safety / embodiment / guidance 全部写成 prompt rule，让 LLM 在 generation 和 filtering 两个 stage 都看见。四类 rule:

**Foundational (改 Asimov 三定律)**:
- F1: A robot may not injure a human being.（去掉了 "through inaction"，因为不想让 robot 偏向 inaction）
- F2: A robot must protect its own existence.（和原版第二定律 swap 了顺序，因为现实里 robot 更需要保护自己不被 human 派危险任务）
- F3: A robot must obey orders except conflict with F1/F2.

**Safety**:
- S1: 不碰 human / animal / living things
- S2: 不碰 sharp objects (knife)
- S3: 不碰 electrical objects (computer / tablet)

**Embodiment**:
- E1: 不 lift 比 book 重的东西（能 push plastic chair 但不能 move couch）
- E2: 单 arm，不能做 bimanual task（不能 open bottle）

**Guidance**:
- G1: "The human command, which the robot should follow if given: {guidance}" — 这是 user inject high-level intent 的钩子，比如 "collect gardening tasks"。

Intuition：把 constraints 从 hard-coded state machine 搬到 natural language rule，让 LLM 的 commonsense 来做 soft enforcement。代价是没有 guarantee，所以还得叠 traditional guardrail（force threshold、E-stop、line-of-sight human、proactive 移除 dangerous object）。这跟 pure RL safety shield 互补。

---

## 5. Task Generation prompt 的结构

Teleop 版 prompt 大致（Appendix D）:

```
Robot: I'm a robot operating in an office...
Robot: I am an ethical and law abiding robot that respects:
  {foundational_rules}
  {safety_rules}
  {embodiment_rules}
  {guidance_rules}
Robot: I will not include a task in my list that violates these rules.
Human: Here are 5 descriptions of the room {scene}.
       Here are a subset of objects in the room {objects}.
       None of the tasks should be boring pick and place...
       they should involve complex, creative, precise and dexterous motions
       eg "insert key into lock".
       Propose {num_tasks} manipulation tasks that the robot can do in the room:
Robot: 1.
```

关键设计:
- **不 fine-tune LLM**，保持 foundation model 的 generality，全靠 prompt。
- **scene + object 双注入**：scene 是 VLM caption，object 是 VLM 列的 5 个名字。FlexCap 比 PaLI 给更 descriptive 的 caption（尤其 color），所以 language diversity 更高（Table 2: 1.137 vs 1.100）。
- **policy-specific suffix**：因为不同 collect policy 能力不同，teleop prompt 鼓励 "complex dexterous"，scripted pick prompt 限制成 "pick X"，RT-2 prompt 限制成 RT-2 已知 skill set（pick / move near / knock / place upright / open / close）。这是 conditional task generation，让 proposal distribution 匹配 executor distribution。
- **negation prompt**："None of the tasks should be boring pick and place" — 用 negative instruction 把 LLM 推向 action-diverse 区域，因为 teleop data 最稀缺的是 dexterous motion，不是又一个 pick-and-place。

---

## 6. Affordance Filter：self-critique as classifier

第二个 LLM call 把每个 proposed task 分类:

```
1) scripted pick - which can only pick objects
2) teleop - asks a human for help
3) rt2 - use a learned policy that can pick, move near, knock, place upright, open/close
4) reject - if a task cannot be performed due to limitations above
```

输出格式: `task -> mode: reason`，比如:

```
peel the banana -> reject: violates rule E2, is a bimanual task
pick person -> reject: violates rule F1, cannot harm a person
pick tea bag -> scripted pick: pick tea bag
close drawer -> rt2: close drawer
```

这本质是一个 $k+1$ 类 classifier（$k$ 个 collect policy + 1 个 reject），用 LLM in-context 实现的 zero-shot classifier。Table 4 的 adversarial 实验显示: constitutional prompting 在 generation 和 filter 都开时，unsafe task 的 reject recall 能到 94%（34/36），只开 filter 时 11%（4/36），只开 generation 时 67%。所以**两层都要写 constitution**，单靠 filter 拦不住，因为 LLM 没在 generation 时被 prime 过就不知道 reject criteria。

这个设计跟 Reflexion / ReAct / Constitutional AI 的 self-critique 一脉相承，跟 SayCan 的 affordance function 也对应：SayCan 用 learned affordance $P(\text{affordance} \mid \text{skill}, \text{scene})$ 做 gating，AutoRT 用 LLM 做 gating，把 affordance 从 learned scalar 变成自然语言 reasoning。

---

## 7. Data Collection：混合 policy

三个 collect policy:
- **Scripted pick**（Appendix H）：hardcoded 的 down → close gripper → lift，第一版是 LLM 写的，人改了 edge case。占 73293 episodes，success 21%。便宜量大。
- **Teleop**：3060 episodes，success 82%。贵但 action-diverse。
- **RT-2**：936 episodes，success 4.7%。RT-2 在这种 in-the-wild + navigation 场景下掉得很厉害，跟训练分布差太远，所以采样率被调低。

采样概率 $p_i$ 动态调: 如果 1 person supervise 3 robots，则 $p_{\text{teleop}} < 1/3$，保证 human 不被 over-subscribe。整体策略是**保 teleop throughput**（不低于 1:1 baseline），宁可少帮 autonomous robot，因为 teleop data 的 action diversity 最值钱，autonomous 的 scripted/RT-2 反正便宜可以多跑。

---

## 8. Diversity Scoring：怎么量化 "有用"

### Language diversity

用 Universal Sentence Encoder（Cer et al. 2018）把 task string encode 成 512-d normalized vector，然后算 pairwise L2 distance:

$$d_{\text{lang}}(t_i, t_j) = \| \text{USE}(t_i) - \text{USE}(t_j) \|_2$$

Table 2 报 average pairwise L2:
- Lang.Table: 0.988
- BC-Z: 1.070
- RT-1: 1.073
- AutoRT w/PaLI: 1.100
- AutoRT w/FlexCap: 1.137
- Optimal（均匀球面）: 1.414

AutoRT 比 hand-designed 任务集 language 更 spread，FlexCap 又比 PaLI 好。这把 "data diverse" 从模糊概念变成可计算 metric，并且这个 metric 可以在不跑 robot 的情况下 ablate VLM，70 个 scene 就能比较——这是工程上很实用的 fast iteration loop。

### Visual diversity

用 CLIP（finetune 到 contrast {first image, goal image} vs caption，来自 Xiao et al. 2023）embed episode，做 k-means（$k=1000$）clustering，新 episode 的 diversity score = 到最近 centroid 的距离:

$$\text{vis-div}(e) = \min_{c \in C} \| \text{CLIP}(e) - c \|_2$$

距离越大越 novel。Fig.5 的 CDF 显示 AutoRT 的 teleop > AutoRT scripted > RT-1 baseline，所有 AutoRT 数据都比 RT-1 dataset 更 diverse，即使 RT-1 全是 teleop。原因是 AutoRT 跑在 4 栋不同 building + 随机撒 100+ random object，环境分布更宽。

Appendix E 还做了个很 cute 的 pilot：robot 把 diversity score 念出来，human 听着 score 主动改 scene（推翻 recycling bin、把 object 放 chair 上），把 diversity score 当 online reward 用来 steer human supervisor。这是 human-in-the-loop diversity maximization，把评估 metric 反过来当采集信号。

---

## 9. 实验：核心 table 解读

**Table 1 (data split)**: Scripted 73k/21%, Teleop 3k/82%, RT-2 0.9k/4.7%。RT-2 success 这么低是关键 finding——VLA model 在 lab train 到 SOTA，real building + navigation 一上就崩，说明 generalization gap 还巨大，这正好 motivate AutoRT 这种 in-the-wild 采数据。

**Table 3 (task generation quality)**: 在 5 scene / 75 task 上,
- Templated language: Relevance 27%, Feasibility 52%
- AutoRT unguided: 28% / 83%
- AutoRT guided: 61% / 77%

两点: (a) LLM 比 verb-object template 在 feasibility 上从 52%→83%，因为 template 会生成 "open keyboard" 这种 nonsense，LLM commonsense 自然避免。(b) guidance rule 把 relevance 从 28%→61%，证明 constitution 里的 G1 钩子真能 steer task distribution。这是 end-user 控制 robot 采什么数据的入口。

**Table 4 (adversarial safety)**: 故意摆 toy animal / 剪刀 / 人。最关键 cell: unsafe prompt + constitutional filter → %Safe 87%, Recall 94%。说明即使 generation 被下毒，filter 还能救回来；但 minimal filter 配 unsafe generation 只剩 26%。结论: constitution 必须**两 stage 都加**。

**Table 5 (downstream RT-1)**: co-finetune RT-1 在 50-50 pretrain + AutoRT mixture,
- Picking from different height: 0% → 12.5%
- Wiping: 10% → 30%
- 只用 teleop subset: picking 仍 0%，wiping 20%

最后一行很重要：non-teleop（scripted + RT-2）数据**有贡献**，光靠 teleop 3k 条不够 generalize height。这验证 hybrid collection 的价值——scripted 虽然 success 低、task 单调，但它覆盖了 teleop 没去过的 scene configuration，对 downstream generalization 有正贡献。

---

## 10. 跟相关工作的一条线

- **SayCan (Ahn 2022)**: LLM 给 plan，learned affordance gating。AutoRT 是 SayCan 的 "self-instruct" 版：用户不给 instruction，LLM 自己 propose，affordance 从 learned scalar 换成 LLM self-critique。
- **RT-2 (Brohan 2023)**: VLA model，既是 AutoRT 的一个 collect policy，也是 AutoRT 数据将来要 finetune 的 downstream。AutoRT 在某种意义上是给 RT-2 这类 VLA 做 data flywheel 的采集端。
- **Voyager (Wang 2023)**: Minecraft self-explore + skill library。AutoRT 是 real-world Voyager，多了 safety / teleop / human supervision。
- **PaLM-E (Driess 2023)**: embodied multimodal LM，把 observation 直接 embed 进 LM token。AutoRT 没把 VLM/LLM merge，而是 loose coupling（VLM caption → LLM text），工程上更便宜但信息 bottleneck 更窄（见 limitation 2）。
- **Constitutional AI (Bai 2022)**: rule-based self-critique for alignment。AutoRT 直接搬到 robot safety。
- **Reflexion / ReAct**: LLM agent self-reflection。AutoRT 的 affordance filter 就是 reflexion step。
- **BC-Z / RT-1 / Language Table**: 都是 hand-designed task list 的 teleop dataset，Table 2 直接对比 language diversity。
- **Fleet-DAgger (Hoque 2022)**: hybrid teleop + autonomous fleet。AutoRT 是同思路但 task 由 LLM propose 而非固定 curriculum。
- **Natural Language Maps (Chen 2023)**: AutoRT 的 navigation module 直接用这个。
- **FlexCap (Dwibedi 2024)**: richer caption → richer task proposal，Table 2 证明 VLM 选择对下游 language diversity 有 measurable 影响。

---

## 11. Limitations 里的 insight

1. **Autonomous policy quality 卡 throughput**: scripted 21% success，RT-2 4.7%，所以 77k episode 里真正 success 的不多。要 scale高质量数据需要更强的 autonomous collect policy（比如 RoboCat 类 self-improving）。
2. **VLM→LLM 信息 bottleneck**: caption 丢细节、hallucinate object、motion blur。PaLM-E 那种直接 image-token 进 LM 的 tight coupling 理论上能缓解，但贵。
3. **Sparse data 难学**: 77k episode / 6650 unique task ≈ 12 sample/task，极度 sparse，对 BC 不友好。RL 能用但 scalability 差。AutoRT 假设 collect 和 control 解耦，但最优应该是 **collect 和 policy co-evolve**（active learning / DAgger 闭环）。
4. **Prompt 没 guarantee**: constitution 提升安全但拦不住所有，必须 human-in-the-loop。这跟 LLM alignment 的开放问题同构。

---

## 12. 我会怎么延伸 think

- **Co-evolve collect & policy**: 现在是 open-loop 采完再 finetune RT-1。更激进的是让 RT-2 的 uncertainty（ensemble disagreement 或 next-action entropy）回传给 task generation prompt，让 LLM 主动 propose "policy 不确定" 的 task，把 uncertainty-driven exploration 接上。这跟 DAgger 的 on-policy 失败采样是同一个 idea 但 lift 到 task space。
- **VLM tight coupling**: 用 PaLM-E / Gemini-V 直接吃 image，跳过 caption bottleneck，task generation 能看到 texture、spatial relation。代价是 latency 和 cost，但对 feasibility 判断会准很多。
- **Constitution as learnable reward**: 现在 constitution 是手写文本。可以让 human critic 对采集 episode 打 label，再 RLHF 一个 "constitution reward model"，把 rule 升级成 learned preference，跟 Constitutional AI 的 RLAIF 完全对应。
- **Skill library like Voyager**: AutoRT 每个 episode 是孤立的，没有 skill abstraction。如果像 Voyager 那样把成功 episode 抽成 reusable skill code，下个 episode 的 task generation 可以 reference 已有 skill，affordance 判断会更准，也能 amortize exploration。
- **Multi-robot knowledge sharing**: 4 栋楼 20 robot 同时跑，但 map 是 copy 的。如果 shared memory 里实时同步 "哪类 scene 已采够"，task generation 可以做 coverage-driven 的 anti-redundant sampling，把 fleet 当 distributed exploration 算法。

---

## 13. 一句话 intuition

AutoRT 把 SayCan 的 "LLM plans, robot executes" 翻转成 "LLM self-proposes + self-critiques + constitutionally-constrained, fleet executes with hybrid teleop/autonomous"，用 foundation model 的 commonsense 当 task-space exploration prior，用 natural-language constitution 当 safety soft-shield，用 language+visual diversity score 当 coverage reward，最终把 robot data collection 从 "human 一条条 teleop" 变成 "human supervise fleet, foundation model drive exploration"。它本质上是在 task space 和 scene space 上做 LLM-guided active learning，把 data collection 本身当成一个 embodied agent 问题来解。

主要 take-away for building intuition: **foundation model 在 robot 里的最高 leverage 用法，现阶段不在低层 control（RT-2 success 4.7% 已经说明），而在高层 orchestration——决定看哪、做什么、谁来做、安不安全**。低层 action 仍然要 dedicated robot policy，但高层 reasoning + safety + diversity 的 "meta-control" 已经被 LLM/VLM 接管了，这是 AutoRT 真正示范的 paradigm shift。
