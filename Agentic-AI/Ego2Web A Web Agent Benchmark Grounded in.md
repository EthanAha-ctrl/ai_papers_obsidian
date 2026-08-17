---
source_pdf: Ego2Web A Web Agent Benchmark Grounded in.pdf
paper_sha256: b488d56b6afd97984cb1fccc7d2b451d6472ca1e9fd39d915016ff672269edaa
processed_at: '2026-08-04T02:27:22-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Ego2Web

## 一句话说清楚

现在所有 web agent benchmark 都在测"看屏幕截图 → 操作网页"，没人测"看你眼睛看到的真实世界 → 去网上办事"。Ego2Web 就是干这个的——给你一段第一人称视频，让你在网上把事办了。

---

## 为什么这个 task 值得做

想想你自己戴 AR glasses 的场景：

- 你在超市看到一包零食，想让 agent 去 Amazon 上查价格对比
- 你看到朋友的耳机，想让 agent 帮你买一个同款的
- 你看到药盒上的药名，想让 agent 帮你在网上预约配送

这些 task 的共同点：**必须先"看见"真实世界的东西，才能在网上执行对应的 action**。

但现在所有的 web agent benchmark（WebArena、VisualWebArena、OSWorld、WebVoyager、Online-Mind2Web）全都是纯数字感知——要么看截图，要么读 DOM 树，要么读 text instruction。没有任何一个 benchmark 测试"从第一人称视角的物理世界感知 → 网上行动"这个链条。

Ego2Web 填的就是这个 gap。

---

## 数据怎么造的

500 个 video-task pair。流程是半自动的：

**第一步**：从 Ego4D（3600 小时第一人称视频数据集）里挑 3 分钟 clip，用 Qwen3-VL 每 5 秒生成一段 structured caption，拼成 video 的 textual profile。

**第二步**：把 video profile 喂给 GPT-5，让它生成"必须依赖 video 内容才能解"的 web task instruction。比如"找 video 里第 4 个被拿起来的零食，去 Amazon 查它的 calories"。

**第三步**：人审。三个标准——task 确实依赖 video（不能只看文字就解）、task 在指定网站上真能执行、instruction 语法清晰。

最终分布：E-Commerce 230、Media Retrieval 132、Knowledge Lookup 92、Local/Maps 31、Others 15。覆盖 18 个网站（Amazon、YouTube、Wikipedia、Google Maps 等）。

这个分布故意 reflect 真实用户行为——买东西和看视频占大头，long-tail sites 引入 UI 多样性。

---

## Agent 要做的事

给定 video $V$ 和 instruction $I$，agent 在 live browser 上执行 action 序列 $A$，达到 goal $G$。

两个核心能力：

1. **Visual grounding**：从 video 里抽出 task-relevant 信息。比如"第 4 个被拿起来的零食是什么牌子什么口味"——这需要 fine-grained object recognition + temporal tracking（分清第 4 个和第 3 个）。

2. **Web execution**：根据 grounding 结果规划 web action——导航到 Amazon、搜索、scroll、点击正确 product page、提取信息。

这两个能力必须串起来。任何一端弱，整个 task 就崩。

---

## Evaluator 怎么搞的

Online evaluation 最大问题：怎么自动判断 agent 成功了没有？

之前的方法（WebVoyager、WebJudge）只看 screenshot + action trajectory + final text response，完全 ignore video。这有个大坑——agent 可能 trajectory 看起来对、final response 说的也对，但它 grounding 错了（比如 video 里是 SunChips Harvest Cheddar，它去买成 SunChips Garden Veggie，包装看起来差不多，text 也都写 SunChips）。

Ego2WebJudge 的核心改进：**把 video evidence 显式塞进判断流程**。

三 stage：

1. **Key-Point Identification**：LLM 从 instruction 里抽出必须达成的 key points（"识别第 2 个被拿起来的 snack bar → 打开正确 flavor 的 product page → 报告 calories → 报告 sugars"）。

2. **Key Screenshot Selection**：trajectory 通常 5-20 步，很多是 loading page、backtracking、UI error。MLLM 对每个 screenshot 打 1-5 分，只保留 task-relevant 的，避免 context overflow 稀释判断。

3. **Final Judgment**：MLLM 整合 instruction + key screenshots + action history + keypoints + video keyframes，做 binary 判断（success / failure）。

**关键 principle**：strict visual grounding。"False positives are worse than false negatives. When in doubt, mark as failure." 不能因为 agent 说自己做对了、网页 title 看起来对、search query 看起来对就判成功，必须 visual consistency 验证——object identity、brand、color、quantity、state 都要对上。

结果：Ego2WebJudge 跟 human eval 的 agreement rate 达 84%，比 WebJudge（78.4%）和 WebVoyager（74.7%）高 6-10 个点。

---

## 实验结果说了什么

测了 6 个 agent：SeeAct、Browser-Use + GPT-4.1、Browser-Use + Gemini-3-Flash、Claude 3.7 Computer Use、Claude 4.5 Computer Use、GPT-5.4。

**主表结果**：

| Agent | Human Eval SR |
|---|---|
| Claude 3.7 | 26.4% |
| Claude 4.5 | 32.8% |
| GPT-5.4 | 30.6% |
| SeeAct | 34.2% |
| BU-GPT-4.1 | 44.4% |
| **BU-Gemini-3-Flash** | **58.6%** |

**最直觉的解读**：BU-Gemini-3-Flash 碾压所有人，因为它是唯一一个能 native 吃 raw video + 用 Browser-Use 高效执行 action 的组合。

Claude 系列和 GPT-5.4 拉胯的原因很直接——它们在 computer-use 模式下没法直接吃 video，得先把 video 转成 textual caption 再喂进去。这一步 information loss 巨大，尤其是 temporal dynamics 和 fine-grained visual attributes 全丢了。所以 Claude 4.5 虽然比 3.7 在 action execution 上强，但视觉 grounding 的 deficit 补不回来。

---

## 分 domain 看

用 Gemini-2.5 Pro judge：

| Domain | Avg SR | 难度直觉 |
|---|---|---|
| Knowledge Lookup | 50.0% | 最容易——Wikipedia/StackExchange 结构化强，目标清晰（"找 backpack 上的大学名，查它 Wikipedia 创立年份"），visual grounding 只要识别 object 上的 text |
| Media Retrieval | 30.1% | 中等——YouTube/IMDb 上找视频，需要识别 video 里的 event 并做 cross-modal retrieval |
| Local/Maps | 23.1% | 难——Google Maps/Yelp 动态界面，多步交互，且 visual cue 常是 spatial（"街道左边的餐厅"），spatial reasoning 难 |
| E-Commerce | 21.7% | 难——Amazon 好操作但 visual grounding 严格，颜色偏差、相似品牌都算失败 |
| Others | 14.4% | 最难——long-tail sites UI 差异大，agent 泛化弱 |

---

## 最关键的 Ablation：视觉输入形式的影响

用 BU-Gemini-3-Flash agent，测三种 input：

| 输入方式 | Total SR |
|---|---|
| 无视觉（只看 instruction） | 4.4% |
| Detailed Caption（video 转成 text） | 23.6% |
| Raw Video | 48.2% |

**这是整篇 paper 最有信息量的一组数字**。

- 无视觉 4.4%——证实 task 确实 video-dependent，没视频任务不可解。
- Caption 23.6%——textual summary 能捕获大部分 semantic info（object name、brand、action），agent 至少能尝试，比无视觉强 5x。
- Raw video 48.2%——比 caption 再翻一倍。caption 丢失的关键信息是 **temporal dynamics 和 fine-grained visual attributes**。

分 domain 看提升幅度更有意思：
- Knowledge Lookup: 39.1% → 75.0%（+35.9）：这里 temporal grounding 极重要（"the second picked up snack"），caption 容易写错"第几个"，raw video 让 agent 自己做 temporal tracking，准确率翻倍。
- Media Retrieval: 29.5% → 50.7%（+21.2）：video 里的 event 需要 temporal localization，caption 模糊化严重。
- Local/Maps: 38.7% → 48.3%（+9.6）：spatial cues（"left side of street"）caption 难精确描述。

**核心 intuition**：对 egocentric video-grounded task，textual abstraction 是 lossy proxy，raw video 的 dense spatio-temporal signal 不可替代。这跟 video-language understanding 领域普遍结论一致——long-form video reasoning 上 caption-based 方法天花板远低于 native video 方法。

---

## Error Analysis：Agent 都怎么挂的

50 个 failure case 人工分析：

- **36% Object Misidentification**：看错 video 里的 object。比如把 SunChips Harvest Cheddar 看成其他口味。MLLM 的 fine-grained visual recognition 在 low-light / motion blur / occlusion 场景下还不够强。

- **18% Temporal Misunderstanding**：分不清"第 2 个"和"第 3 个"被拿起来的东西。current MLLM 对 egocentric video 里 action sequence 的 temporal tracking 还很弱，3 分钟 30+ narrations 的 clip 容易乱。

- **16% Cross-Modal Retrieval Failure**：video 里 object 识别对了，但 web 上没找到匹配。这是 video→web 的 semantic gap——agent 知道是"蓝色 Bosch 电钻"但 Amazon 搜索 query 构造不好，或者 product page 图片角度不同导致 visual verification 失败。

- **12% Coarse-Grained Matching**：找到 semantically similar 但不 exact match 的结果。比如要"官方 campus tour video"，agent 找到相关但非官方的视频。Agent 对 instruction constraint 的 strictness 理解不够。

- **18% Others**：CAPTCHA、authentication barriers、planning inefficiency（40 步不够用）、instruction misinterpretation。

Paper 里有个典型 failure case：task 是"找 video 里第 2 个被拿起来的 sauce 并报告 package size in fl oz"。Agent 错把 mustard 当成第 2 个 sauce（temporal misunderstanding），然后即使到了 product page 也没提取 fl oz 信息（cross-modal retrieval failure）。这是 **compositional failure**——temporal grounding error 传播到下游，整个链条崩盘。

---

## 这篇 paper 的深层信号

### 1. Native video MLLM 是必需的，caption proxy 不行

当前 community 有两条路线：caption-first（把 video 转 text 再用 LLM reasoning，便宜但 lossy）vs native video（直接把 video token 喂 MLLM，贵但保留 dense signal）。Ego2Web 的 ablation 给了定量证据——在 grounding 严格的 task 上，native video 比 caption 高 2x。

对 AR glasses / wearable assistant 这种应用，必须用 native video MLLM，不能靠 caption proxy。这个结论跟 Ego-R1（chain-of-tool-thought）、VideoTree（adaptive tree representation）、Crema 这些 native video reasoning 方向的工作互补。

### 2. Egocentric video-grounded web agent 是 multimodal agent 的下一个 frontier

它把两个 sub-field 强制 fusion：
- Video understanding 端：temporal grounding、fine-grained recognition、long-form reasoning
- Web agent 端：planning、action execution、cross-modal retrieval

任何一端弱都做不好。这正好对应 Project Astra 这类 AR glasses assistant 的真实需求。

### 3. Evaluator 的 strict principle 值得记住

"False positives are worse than false negatives." 在所有 grounding-要求高的 LLM-as-a-Judge 设计里，宽松 evaluator 会让 agent 通过"看起来对但实际错"的 case，benchmark 失去区分度。Ego2WebJudge 用 strict visual consistency check + video evidence cross-reference 把 false positive rate 压低，这是它能 AR > 84% 的核心。

### 4. 当前 SOTA 还有 ~40% headroom

Human oracle 估算 90%+，agent 最高 58.6%，中间 40% 的 gap 是明确的 research direction。主要 bottleneck：
- Temporal grounding（54% errors 来自 object misidentification + temporal misunderstanding）
- Cross-modal retrieval（16% errors 来自 video→web semantic gap）
- Native video input 支持（Claude / GPT-5.4 因 caption proxy 被严重拖后腿）

### 5. Open problems

- **Long-form video**：Ego2Web 只用 3 分钟 clip，真实 AR glasses 是 hours 级别，需要 hierarchical memory + retrieval（参考 HourVideo）
- **Exocentric + egocentric fusion**：EgoExo4D 提供 first + third person，未来 agent 可能需要多视角融合
- **Real-time constraint**：AR glasses 要求 low latency，当前 40 步 trajectory 太慢
- **Privacy / on-device**：第一人称视频高度敏感，cloud-based MLLM 不一定可行，需要 on-device video understanding + selective cloud action

---

## 跟你的 intuition 怎么接

Karpathy 你之前一直强调 "native multimodal" 的重要性——model 直接吃 pixel、直接吃 audio、直接吃 video token，不要中间抽象。Ego2Web 的 ablation 正好给了 quantitative 证据：raw video 48.2% vs caption 23.6%，差 2x。在 grounding-要求严格的下游 task 上，textual abstraction 是 lossy proxy，这个结论支持 native multimodal 路线。

另外，Ego2Web 揭示的 **compositional failure** 现象跟你常说的 "long horizon task 的 error propagation" 一致——temporal grounding 错了，下游 retrieval 和 verification 全错。这说明当前 agent 的各模块（perception → reasoning → action）是松耦合的，error 不衰减地传播。未来可能需要更 tight 的 end-to-end training，或者显式的 error recovery 机制（类似 Ego-R1 的 RL with tool-use thought）。

最后，这个 benchmark 的 evaluation 设计——"false positive 比 false negative 更糟"——是个很好的 principle，可以推广到所有 grounding / alignment 评测。当前很多 LLM-as-a-Judge 太宽松，导致 benchmark 区分度低。Ego2WebJudge 的 strict visual consistency check 是个值得参考的设计 pattern。

---

## References

- Ego2Web: https://arxiv.org/abs/2507.15406
- Ego4D: https://ego4d-data.org/
- EgoSchema: https://egoschema.cloudcv.net/
- VisualWebArena: https://arxiv.org/abs/2401.13649
- VideoWebArena: https://arxiv.org/abs/2410.19100
- OSWorld: https://os-world.github.io/
- Online-Mind2Web / WebJudge: https://arxiv.org/abs/2504.01382
- WebVoyager: https://arxiv.org/abs/2401.13919
- Browser-Use: https://browser-use.com/
- Anthropic Computer Use: https://www.anthropic.com/news/developing-computer-use
- OpenAI Operator: https://openai.com/index/introducing-operator/
- Ego-R1: https://arxiv.org/abs/2506.13654
- VideoTree: https://arxiv.org/abs/2412.01593
- HourVideo: https://arxiv.org/abs/2411.04998
- EgoExo4D: https://egoexo4d-data.org/
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- EgoPlan-Bench: https://arxiv.org/abs/2312.09791
- EgoThink: https://arxiv.org/abs/2403.17329
- WebArena: https://webarena.dev/
- Mind2Web: https://osu-nlp-group.github.io/Mind2Web/
- LLM-as-a-Judge (MT-Bench): https://arxiv.org/abs/2306.05685
- AssistantBench: https://assistantbench.github.io/
- Crema: https://arxiv.org/abs/2402.05889

---

# Ego2Web: 把 Egocentric Vision 接到 Web Agent 上

这篇 paper 来自 Google DeepMind 和 UNC Chapel Hill（Shoubin Yu, Mohit Bansal 等），核心 idea 是构造第一个把 **first-person egocentric video perception** 和 **web agent action execution** 串起来的 benchmark。问题 motivation 很简单：未来的 AR glasses / wearable assistant（Project Astra 那类）一定会有"我看到一个东西 → 帮我在网上做点什么"的需求，但当前所有 web-agent benchmark（WebArena, VisualWebArena, OSWorld, Online-Mind2Web, WebVoyager）都只关心 digital perception（screenshot / DOM / text instruction），完全 ignore 用户物理环境里看到的 visual content。Ego2Web 把这个 gap 显式化，并给出可 scalable 的 evaluation framework。

---

## 1. Task Formulation

Section 3.1 给出的形式化定义：

给定 egocentric video $V = \{f_1, f_2, \dots, f_t\}$ 和 task instruction $I$，agent 在 browser 环境 $B$ 上执行 action sequence $A = \{a_1, a_2, \dots, a_n\}$，达到 goal state $G$。

变量含义：
- $V$: 第一人称视频，$f_i$ 是第 $i$ 帧，$t$ 是总帧数
- $I$: 自然语言任务指令（必须依赖 video 内容才能解）
- $A$: action 序列，$a_j$ 是第 $j$ 个 web action（click, scroll, type, navigate...），$n$ 是总步数
- $B$: live browser environment（在线真实网站，sandbox）
- $G$: goal state，由 human annotator 或 Ego2WebJudge 判断是否达到

这定义看起来平凡，但关键点在于 **$I$ 的可解性是 conditioned on $V$ 的**：如果 agent 看不到 video，只看 $I$，任务是 under-determined 的（比如"find the same color toaster seen in the video"，没 video 就不知道什么颜色）。这就是 Ego2Web 和 VisualWebArena / VideoWebArena 的本质区别——后者视觉信息全在 webpage 上，前者视觉信息在 physical world。

需要测试的两种能力：
1. **Visual Perceptual Understanding and Grounding**：从 $V$ 抽取 object category / brand / color / temporal cue（比如"the fourth snack picked up"）
2. **Web Execution Reasoning**：根据 video perception 规划 $A$，在 $B$ 上执行直到 $G$

---

## 2. Data Generation Pipeline（半自动 LLM+Human）

整个 pipeline 在 Fig. 3 左半边：

### Step 1: Video Pool & Visual Parsing
Video 来源是 **Ego4D**（Grauman et al., 2022, 3600 小时 first-person video with dense narrations），通过 **EgoSchema**（Mangalam et al., 2023）筛选标准化的 3 分钟 clip（每段至少 30 条 human narration，确保语义密度够）。

用 **Qwen3-VL-7B** 每 5 秒生成一段 structured caption，caption schema 是 JSON：
```
{
  "video_description": "...",
  "objects": {
    "object_name_1": "color, shape, brand, state, interaction...",
    ...
  }
}
```
拼接成 video profile：
$$V_{meta} = \{\nu_{meta}^1, \nu_{meta}^2, \dots, \nu_{meta}^k\}$$

- $V_{meta}$: video 的 textual metadata 表征
- $\nu_{meta}^i$: 第 $i$ 个 5 秒 clip 的 structured caption
- $k$: clip 总数（3 分钟 video → $k = 36$）

这里有个 design choice 值得注意：用 Qwen3-VL-7B 而不是更大的模型做 captioning，是为了控制成本（500 个 video × 36 clips × 大模型 caption 太贵）。但 caption 质量直接决定下游 GPT-5 生成 task instruction 的质量，所以这是个 trade-off。

### Step 2: LLM-based Task Instruction Generation
把 $V_{meta}$ 和一个预定义 website 池（Amazon, Wikipedia, YouTube, Google Maps, Yelp... 共 ~35 个 domain）喂给 **GPT-5**，prompt 要求生成必须依赖 ≥2 个 visual cues 的 task instruction，且输出 strict JSON：
```json
{
  "suitable": true,
  "tasks": [{
    "difficulty": 1,
    "instruction": "...",
    "must_match": ["cue1", "cue2"],
    "timestamps": ["00:10-00:20"],
    "allowed_domains": ["youtube.com"],
    "why_video_dependent": "..."
  }]
}
```

`must_match` 字段是关键——它显式列出 instruction 必须匹配的 visual cues，后面 Ego2WebJudge 会用这个做严格 grounding 检查。

### Step 3: Human Verification & Refinement
Annotator 按三个标准过滤：
1. **Visual Grounding**: task 必须依赖 video 里可见信息（不能只靠 caption 解）
2. **Web Feasibility**: task 必须能在指定 website 上执行（不能让 agent 找一个 Amazon 上根本不存在的产品）
3. **Instruction Quality**: 语法清晰、目标可验证

最终保留 500 个 video-task pairs。分布（Table 6）：
- E-Commerce: 230 tasks（Amazon, eBay, Walmart, Apple, Adidas, Etsy, Target, BestBuy, IKEA, Nike）
- Media Retrieval: 132（YouTube, IMDb, Bilibili）
- Knowledge Lookup: 92（Wikipedia, StackExchange）
- Local/Maps: 31（Google Maps, Yelp, TripAdvisor）
- Others: 15（LinkedIn, Calendar, Reddit, NYTimes, Quora, Booking）

这个分布刻意 reflect 真实用户行为——e-commerce 和 media 占主导，long-tail sites 引入 UI 多样性挑战。

---

## 3. Ego2WebJudge：Visual-Grounded LLM-as-a-Judge

这是 paper 的核心 methodological contribution。已有的 online web agent evaluators（WebVoyager, WebJudge/Online-Mind2Web）只看 screenshot + action trajectory + final response，完全 ignore video。Ego2WebJudge 把 egocentric video evidence 显式塞进判断流程。

公式 (1)：
$$O = Ego2WebJudge(I, \nu, A, S)$$

- $O \in \{\text{success}, \text{failure}\}$: 二元判断结果
- $I$: task instruction
- $\nu$: annotated egocentric visual evidence clip（包含 task 所需的 object brand / color / action 等）
- $A = \{a_1, \dots, a_n\}$: agent 的 action history
- $S = \{s_1, \dots, s_k\}$: trajectory 中的 webpage screenshots

三 stage pipeline（Fig. 3 右半边）：

### Stage 1: Key-Point Identification
LLM 从 $I$ 蒸馏出必须达成的 critical key points。Motivation：Ego2Web 的 instruction 多步推理，比如"find the second boxed snack bar picked up in the video, open its product page, report calories and sugars per bar"——这里 keypoints 是：
1. 正确识别 second snack bar（不是 first / third）
2. 打开正确 flavor 的 product page
3. 报告 calories
4. 报告 sugars

把 instruction 拆成 explicit keypoints 给 judge 一个 prior，避免 LLM 被最终 response 的表面流畅性欺骗。

### Stage 2: Key Screenshot Selection
MLLM 对每个 screenshot $s_i$ 评分 1-5，超过 threshold $\tau$ 的保留为 key screenshots。Web trajectory 通常 5-20 步，很多是 loading page、backtracking、UI error。直接全塞给 MLLM 会 context overflow 且稀释判断质量。这个 design 借鉴了 WebJudge 但更严格——只保留 task-relevant 的中间状态。

### Stage 3: Final Outcome Judgment
MLLM-judge 整合四类输入：
1. Task instruction $I$
2. Selected key screenshots（Stage 2 输出）
3. Agent action history $A$
4. LLM-generated keypoints（Stage 1 输出）+ annotated keyframes from $\nu$

输出 strict format：
```
Thoughts: <reasoning based on key points, webpage evidence, and ego video evidence>
Status: success or failure
```

**关键 evaluation principle**（system prompt 里强调）：
- "False positives are worse than false negatives. When in doubt, mark as failure."
- 不能基于 agent 文本声称、webpage title、search query 判断成功
- 必须 visual consistency 验证：object identity / brand / color / quantity / state / action 都要对上
- Near / partial match = failure

这个 strict principle 是 Ego2WebJudge 比 WebJudge AR 高的核心原因——WebJudge 只看文本轨迹，容易被"看起来对"的 trajectory 骗过；Ego2WebJudge 强制 visual grounding cross-check。

---

## 4. Experiments

### 4.1 Setup
6 个 baseline agent：
- **SeeAct**（Zheng et al., 2024）: GPT-4V based, set-of-mark prompting
- **Browser-Use (BU) + GPT-4.1**（Müller & Žunič, 2024）
- **BU + Gemini-3-Flash**
- **Claude Sonnet 3.7 Computer Use**
- **Claude Sonnet 4.5 Computer Use**
- **GPT-5.4**

**关键 modality 差异**：
- GPT-4.1 / GPT-4o 类 agent：input 是 sparse keyframes
- Gemini-2.5/3/3.1 类 agent：input 是 raw video
- Claude 系列 / GPT-5.4：**无法直接吃 video**，先用 Gemini-3.1-Pro 把 video 转成 detailed timestamped caption，再把 caption 喂给 agent

这个差异是 Table 2 性能差距的主要 driver。

Max step = 40，human eval 用 3 annotators majority voting。

### 4.2 Main Results (Table 2)

| Evaluation | Base MLLM | Claude 3.7 | Claude 4.5 | GPT-5.4 | SeeAct | BU-GPT-4.1 | BU-Gemini-3-Flash |
|---|---|---|---|---|---|---|---|
| Ego2WebJudge | Qwen3-VL-Flash | 20.8 | 32.2 | 38.8 | 29.6 | 34.6 | **57.2** |
| Ego2WebJudge | Gemini-2.5 Pro | 17.8 | 24.8 | 23.6 | 25.2 | 34.6 | **48.2** |
| Ego2WebJudge | GPT-4o | 19.4 | 27.2 | 26.8 | 26.8 | 47.6 | **51.4** |
| Human Eval | — | 26.4 | 32.8 | 30.6 | 34.2 | 44.4 | **58.6** |

几个直觉：
1. **BU-Gemini-3-Flash 全面最优**：58.6% human SR，比第二名 BU-GPT-4.1 (44.4%) 高 14.2%。原因是它既能吃 raw video（dense temporal modeling）又有 Browser-Use 的高效 action execution framework。
2. **Claude 系列 / GPT-5.4 拉胯**：human SR 只有 26-33%。它们没有 native video input，靠 caption proxy，information loss 严重。Claude 4.5 比 3.7 好 6.4%，说明 action execution 能力提升有部分补偿，但视觉 grounding 的 deficit 补不回来。
3. **Ego2WebJudge vs Human Eval gap 小**：BU-Gemini-3-Flash 上，Gemini-2.5 Pro judge 给 48.2% vs human 58.6%，差 10.4%；GPT-4o judge 给 51.4%，差 7.2%。最差也就 ~10%，说明 Ego2WebJudge 是 human eval 的可行 proxy。
4. **不同 judge 之间 ranking 一致**：无论用 Qwen3-VL-Flash / Gemini-2.5 Pro / GPT-4o，agent 排序基本一致，只是绝对分数有偏移。Qwen3-VL-Flash 偏 optimistic（系统性高估），Gemini-2.5 Pro 和 GPT-4o 更接近 human。

### 4.3 Per-Domain Analysis (Table 3)

用 Gemini-2.5 Pro judge：

| Domain | Claude 3.7 | Claude 4.5 | GPT-5.4 | SeeAct | BU-GPT-4.1 | BU-Gemini-3-Flash | Avg SR |
|---|---|---|---|---|---|---|---|
| E-Commerce | 13.0 | 18.2 | 14.3 | 19.5 | 26.9 | 38.2 | 21.7 |
| Media Retrieval | 19.6 | 26.5 | 29.5 | 24.2 | 30.3 | 50.7 | 30.1 |
| Knowledge Lookup | 33.6 | 45.6 | 39.1 | 43.4 | 63.0 | **75.0** | 50.0 |
| Local/Maps | 6.4 | 12.9 | 29.0 | 19.3 | 22.5 | 48.3 | 23.1 |
| Others | 0.0 | 6.6 | 6.6 | 20.0 | 40.0 | 13.3 | 14.4 |
| **Total** | 17.8 | 24.8 | 23.6 | 25.2 | 34.6 | **48.2** | 29.0 |

直觉解读：
- **Knowledge Lookup 最容易**（avg 50%）：Wikipedia / StackExchange 结构化强，目标清晰（比如"找这个 backpack 上印的大学名，查它 Wikipedia 创立年份"）。Visual grounding 只要识别 text on object，web action 是 single lookup。
- **Local/Maps 最难**（avg 23.1%）：Google Maps / Yelp 动态界面，多步交互（filter、sort、scroll），且 visual cue 通常是 spatial（"the restaurant on the left side of the street"），对 agent 的 spatial reasoning 要求高。
- **E-Commerce 中等偏难**（avg 21.7%）：Amazon 等 site 好操作，但 visual grounding 严格——video 里看到"mint green toaster with brand X logo"，web 上必须找到 exact match，颜色偏差 / 相似品牌都算失败。
- **Others 最难**（avg 14.4%）：long-tail sites UI 差异大，agent 泛化弱。

### 4.4 Ego2WebJudge vs Prior Evaluators (Table 4)

Agreement Rate (AR) with human eval：

| Method | Base MLLM | Avg AR |
|---|---|---|
| WebVoyager | Gemini-2.5-Pro | 70.7 |
| WebJudge | Gemini-2.5-Pro | 76.1 |
| **Ego2WebJudge** | Gemini-2.5-Pro | **80.8** |
| WebVoyager | GPT-4o | 74.7 |
| WebJudge | GPT-4o | 78.4 |
| **Ego2WebJudge** | GPT-4o | **84.0** |

Ego2WebJudge 平均比 WebJudge 高 ~4-6%，比 WebVoyager 高 ~10%。改进来源就是引入 $\nu$（video evidence）做 cross-modal consistency check。

一个有意思的现象：Claude-based agents 在所有 evaluator 上 AR 都偏高（比如 Claude 3.7 + Ego2WebJudge + Gemini-2.5-Pro = 85.4%）。Paper 解释：Claude agents 经常直接 fail（任务没完成），human 和 auto 都判 fail，agreement 自然高。这说明 AR 不是越越高越好——失败一致 ≠ 评估准确。真正考验 evaluator 的是 partial success / nuanced case，BU-GPT-4.1 和 BU-Gemini-3-Flash 这种行为多样的 agent 上 AR 仍能维持 80%+ 才说明 evaluator robust。

---

## 5. Ablation: Visual Modality Hierarchy (Table 5)

这是 build intuition 最关键的一张表。用 BU-Gemini-3-Flash agent + Ego2WebJudge(Gemini-2.5-Pro)：

| Raw Video | Detailed Caption | E-Commerce | Media Retrieval | Knowledge Lookup | Local/Maps | Others | Total |
|---|---|---|---|---|---|---|---|
| ✗ | ✗ | 2.6 | 7.5 | 5.4 | 3.2 | 0.0 | **4.4** |
| ✗ | ✓ | 13.0 | 29.5 | 39.1 | 38.7 | 6.6 | **23.6** |
| ✓ | ✗ | 38.2 | 50.7 | 75.0 | 48.3 | 13.3 | **48.2** |

清晰的 hierarchy: **no visual (4.4%) < caption (23.6%) < raw video (48.2%)**

几个关键 takeaways：
1. **Language-only 完全崩**（4.4%）：证实 task 确实 video-dependent，caption 缺失时 instruction 不可解。
2. **Caption 提升 ~5x**（4.4→23.6）：structured textual summary 能捕获大部分 semantic info（object name / brand / action），让 agent 至少能尝试。
3. **Raw video 再翻倍**（23.6→48.2）：caption 丢失的关键信息是 **temporal dynamics 和 fine-grained visual attributes**。
   - Knowledge Lookup 提升 39.1→75.0（+35.9）：这个 domain 里"the second picked up snack"这类 temporal grounding 极重要，caption 把"second"写错了或漏了，agent 就错；raw video 让 agent 自己做 temporal tracking。
   - Local/Maps 提升 38.7→48.3（+9.6）：spatial cues（"left side of street"）caption 难精确描述，video 直接给。
   - Media Retrieval 提升 29.5→50.7（+21.2）：video 里的事件（"the moment the person opens the red box"）需要 temporal localization，caption 模糊化严重。

这个 ablation 给的核心 intuition：**对 egocentric video-grounded task，textual abstraction 是 lossy proxy，raw video 的 dense spatio-temporal signal 不可替代**。这跟 video-language understanding 领域的普遍结论一致——long-form video reasoning 上 caption-based 方法天花板远低于 raw video 方法（参考 VideoTree, CreMA, Ego-R1 的工作）。

---

## 6. Error Analysis（50 failures, BU-Gemini-3.1）

| Error Type | % |
|---|---|
| Object Misidentification | 36% |
| Temporal and Action Misunderstanding | 18% |
| Failure in Cross-Modal Retrieval | 16% |
| Coarse-Grained Matching Errors | 12% |
| Others (instruction misinterpretation, planning, CAPTCHA) | 18% |

直觉解读：
1. **36% Object Misidentification**：agent 看错 video 里的 object（比如把 SunChips Harvest Cheddar 看成其他口味）。这是 MLLM 的 fine-grained visual recognition 能力不足，尤其在 low-light / motion blur / occlusion 场景。
2. **18% Temporal Misunderstanding**：分不清"second picked up"和"third picked up"。这是 temporal grounding 能力不足——current MLLM 对 egocentric video 里的 action sequence tracking 还很弱，特别是 30+ narrations 的 3 分钟 clip。
3. **16% Cross-Modal Retrieval Failure**：video 里 object 识别对了，但 web 上没找到匹配。这是 video→web 的 semantic gap——agent 知道是"蓝色 Bosch 电钻"但 Amazon 搜索 query 构造不好，或者 product page 上图片角度不同导致 visual verification 失败。
4. **12% Coarse-Grained Matching**：找到 semantically similar 但不 exact match 的结果。比如要"官方 campus tour video"，agent 找到一个相关但非官方的视频。这是 agent 对 instruction constraint 的 strictness 理解不够。
5. **18% Others**：包括 CAPTCHA / authentication barriers（真实 web 评估的固有困难）、planning inefficiency（40 步不够用）、instruction misinterpretation。

Fig. 6 的 failure case 很 illustrate：task 是"找 video 里 second picked-up sauce 并在 Walmart 报告 package size in fl oz"。Agent 错把 mustard 当成 second sauce（temporal misunderstanding），然后即使到了 product page 也没提取 fl oz 信息（cross-modal retrieval failure）。这是典型的 **compositional failure**——temporal grounding error 传播到下游 retrieval 和 verification，导致整体崩盘。

---

## 7. Build Your Intuition: 关键联想

### 7.1 为什么 raw video >> caption 是重要信号
当前 MLLM community 有两条路线：
- **Caption-first**：把 video 转成 text，再用 LLM reasoning（便宜、scalable、但 lossy）
- **Native video**：直接把 video token 喂给 MLLM（贵但保留 dense signal）

Ego2Web 的 ablation 给了 quantitative 证据：在 grounding-要求严格的 task 上，native video 比 caption 高 2x。这对未来 agent 架构设计有指引——AR glasses 类应用必须用 native video MLLM，不能靠 caption proxy。参考 Ego-R1 (Tian et al., 2025) 的 chain-of-tool-thought、VideoTree (Wang et al., 2025b) 的 adaptive tree representation，都是 native video reasoning 方向。

### 7.2 Ego2Web vs VisualWebArena / VideoWebArena 的本质区别
- **VisualWebArena** (Koh et al., 2024)：视觉信息全在 webpage 上（screenshot reasoning），没有 physical world grounding
- **VideoWebArena** (Jang et al., 2024)：input 是 web trajectory video（不是 first-person physical video），测的是 long-context web agent
- **Ego2Web**：input 是 first-person physical video，output 是 web action。这是 **embodied → digital** 的 bridge，对应 AR glasses / wearable robot 场景

### 7.3 Ego2WebJudge 的设计哲学
"False positives are worse than false negatives" 这个 principle 值得记住。在 visual grounding task 上，宽松 evaluator 会让 agent 通过"看起来对但实际错"的 case，benchmark 失去区分度。Ego2WebJudge 用 strict visual consistency check + video evidence cross-reference 把 evaluator 的 false positive rate 压低，这是它能 AR > 84% 的核心。这个 principle 在所有 grounding-要求高的 LLM-as-a-Judge 设计里都适用。

### 7.4 Open Problems
1. **Long-form egocentric video**：Ego2Web 用 3 分钟 clip，真实 AR glasses 是 hours 级别，需要 hierarchical memory + retrieval（参考 HourVideo, Chandrasegaran et al., 2024）
2. **Exocentric + Egocentric fusion**：EgoExo4D 提供 first+third person，未来 agent 可能需要多视角 fusion
3. **Cross-modal alignment 的 supervised training**：当前 agent 都是 zero-shot，video→web 的 semantic gap 没有 fine-tune 修正
4. **Real-time constraint**：AR glasses 场景要求 low latency，当前 agent 40 步 trajectory 太慢
5. **Privacy / on-device**：first-person video 是高度敏感数据，cloud-based MLLM 不一定可行，需要 on-device video understanding + selective cloud action

### 7.5 与 Project Astra / Anthropic Computer Use / OpenAI Operator 的关系
- **Project Astra**：Google 的 multimodal assistant，目标就是 AR glasses，Ego2Web 直接对应其应用场景，paper 明确提到
- **Anthropic Computer Use**：在 Ego2Web 上 SR 26-33%，主要瓶颈是没有 native video input
- **OpenAI Operator**：类似，digital-only perception

这说明 Ego2Web 不只是 academic benchmark，它是未来 real-world multimodal agent 的 critical capability test。当前 SOTA 在这个 benchmark 上还有 ~40% headroom（human oracle ~90%+ 估算 vs agent 58.6%），是个明确的 research direction。

---

## 8. References & Further Reading

- Paper: https://arxiv.org/abs/2507.15406 (Ego2Web project page)
- Ego4D dataset: https://ego4d-data.org/
- EgoSchema: https://egoschema.cloudcv.net/
- VisualWebArena: https://arxiv.org/abs/2401.13649
- VideoWebArena: https://arxiv.org/abs/2410.19100
- OSWorld: https://os-world.github.io/
- Online-Mind2Web / WebJudge: https://arxiv.org/abs/2504.01382
- WebVoyager: https://arxiv.org/abs/2401.13919
- Browser-Use: https://browser-use.com/
- Anthropic Computer Use: https://www.anthropic.com/news/developing-computer-use
- OpenAI Operator: https://openai.com/index/introducing-operator/
- Ego-R1: https://arxiv.org/abs/2506.13654
- VideoTree: https://arxiv.org/abs/2412.01593 (CVPR 2025)
- EgoExo4D: https://egoexo4d-data.org/
- HourVideo: https://arxiv.org/abs/2411.04998
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- EgoPlan-Bench: https://arxiv.org/abs/2312.09791
- EgoThink: https://arxiv.org/abs/2403.17329
- WebArena: https://webarena.dev/
- Mind2Web: https://osu-nlp-group.github.io/Mind2Web/
- LLM-as-a-Judge (MT-Bench): https://arxiv.org/abs/2306.05685
- AssistantBench: https://assistantbench.github.io/

---

## 9. Summary

Ego2Web 的 contribution 可以总结成三层：

1. **Task formulation**：把 egocentric video perception 和 web action execution 串成一个 task，明确指出 physical grounding 是 web agent 的 missing capability
2. **Benchmark**：500 video-task pairs，semi-automatic LLM+human pipeline，覆盖 5 大 domain / 18 个网站，online evaluation on live websites
3. **Evaluator**：Ego2WebJudge，三 stage multimodal LLM-as-a-Judge，引入 video evidence 做 strict visual grounding check，AR 84% with human

实验揭示的核心 finding：
- 当前 SOTA agent 在 Ego2Web 上最高 58.6% SR，远低于 human oracle，~40% headroom
- **Raw video input >> caption proxy**（48.2% vs 23.6% vs 4.4%），证明 native multimodal perception 不可替代
- **Temporal grounding 是最大 bottleneck**（36%+18%=54% errors 来自 object misidentification 和 temporal misunderstanding）
- **Cross-modal retrieval gap**（16% errors）：video→web 的 semantic alignment 仍是 open problem
- Ego2WebJudge 比 WebJudge / WebVoyager AR 高 4-10%，证明 visual grounding 是 evaluator 的关键

对你（Karpathy）的 intuition 来说，这篇 paper 最重要的信号是：**egocentric video-grounded web agent 是 multimodal agent 的下一个 frontier**，它把 video understanding（temporal grounding, fine-grained recognition）和 web agent（planning, action execution, cross-modal retrieval）两个 sub-field 强制 fusion，任何一端弱都做不好。这跟 Ego-R1 / VideoTree / native video MLLM 这条线在 video understanding 端的工作互补——Ego2Web 提供了测它们 downstream impact 的 benchmark。
