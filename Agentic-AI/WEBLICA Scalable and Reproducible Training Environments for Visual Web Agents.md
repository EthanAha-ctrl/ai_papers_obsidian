---
source_pdf: WEBLICA Scalable and Reproducible Training Environments for Visual Web
  Agents.pdf
paper_sha256: 83d6dc3998752d64a41e4fd69e5278cf6df6e2ecd64c346616ed91bd3efe727f
processed_at: '2026-08-13T03:54:24-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 WEBLICA

## 一句话总结

教 AI 上网这件事，最大麻烦是**没地方练**。真网站不让随便点、会封 IP、还会变；假网站又太假、种类太少。Apple 这篇论文搞了个新办法：**把真网站"录下来"反复放** + **让 AI 自己造几万个假网站练手**，结果练出来的 8B 小模型上网水平超过了所有同体量开源模型，甚至逼近 OpenAI 和 Google 的闭源产品。

---

## 问题出在哪

想象你要训一个会帮你点外卖、填表格、查信息的 AI 助手。它需要通过**实际操作**来学——就像人学开车必须上路，不能只看驾驶手册。

但"上路"这件事在 web 上特别难：

- **真网站不稳定**：你今天跑通的任务，明天网站改版了就挂了；训练中途 timeout、bot detection 把你 ban 掉；网络一抖动整个 rollout 报废
- **真网站慢**：每次 action 等 1.5 秒，RL 训练一轮要跑上千次 rollout，算下来训练成本爆炸
- **假网站太窄**：WebArena 这类手动搭的环境只有几个固定 domain（一个 reddit、一个电商、一个 gitlab），agent 学会了这几个不代表会上别的网
- **纯 SFT 数据不够**：之前 Fara、MolmoWeb 这些工作都是收集几十万条"老师演示的轨迹"做监督学习，但 agent 只是模仿，没学会 trial-and-error——遇到没见过的页面就傻了

核心矛盾：**要 scale 就得上 live web，要 reproducible 就得脱离 live web**。这两件事看起来打架。

---

## Weblica 的解法：两条腿走路

### 第一条腿：录播真网站（Weblica-Cache）

你想想，一个 web page 的交互本质是什么——你 click 一个 button，浏览器发一个 HTTP request，服务器返回一些 HTML / JSON，页面更新。**只要把所有 HTTP 响应录下来，下次 click 同样的 button 时把录好的响应放回去，页面表现就跟真的一模一样**。

这跟你看 YouTube 视频一个道理——直播不能暂停倒退，但录播可以反复看。

**难点在哪**：很多 URL 参数是"volatile"的——时间戳 `?ts=1723456789`、session token `?sid=abc123`、CSRF token。第一次访问和第二次访问这些参数就变了，如果你傻乎乎用完整 URL 做 cache key，每次都 cache miss。

**Weblica 怎么解决**：
1. 让一个 32B 的 agent 先跑一遍任务，录下所有 HTTP 请求
2. 再跑一遍，找出哪些请求 miss 了
3. 自动对比两次请求，识别出"哪些参数是会变的"
4. 生成一条规则："对这个网站，cache key 里去掉 `ts` 和 `sid`"
5. 在完全断网的环境下回放验证，agent 能把任务跑完才保留

这套自动化 pipeline 让他们从 InstaV3 的 146K 网站任务池里筛出 **15,600 个可复现环境**。

**好处**：真网站的真实视觉、真实 DOM、真实 layout 全都保住了，训练时完全离线、50ms 一次 action、bit-level reproducible。

**代价**：只能做无 side-effect 的任务（浏览、搜索、填表），不能真的下单、发邮件——因为服务器状态不会真变。

---

### 第二条腿：AI 造网站（Weblica-Synth）

单靠录播覆盖不了所有能力，比如"登录流程"、"购物车管理"这些有状态的交互，cache 不好搞。于是他们换了个思路：**让 Claude Code 自己写网站**。

**具体流程**：
1. 先分析 Online-Mind2Web benchmark 上 agent 的成功/失败轨迹，用 GPT-5.2 提取出 19,721 个细粒度能力（"打开下拉菜单"、"切换 tab"、"日期选择器"等），聚合成 144 个能力组
2. 每次生成一个网站时，从三个维度采样：
   - **目标能力**（144 选 1）
   - **网站类型**（1,160 选 1：银行、瑜伽馆、动物园、航空...）
   - **视觉风格**（961 选 1：极简、拟物、双色调...）
3. Claude Code (Opus 4.5) 写纯 HTML/CSS/JS，无框架依赖，用 localStorage 模拟服务端状态
4. Z-Image-Turbo 生成产品图、banner 等素材
5. Claude Code 自己用 Playwright 截图、检查功能、迭代修复，直到网站和任务都跑得通

**为什么三维采样很重要**：如果不采样网站类型和视觉风格，Claude Code 会塌缩到一种"默认审美"——所有网站长得都像 SaaS landing page。加了这个多样性之后，有航空订票、有医疗记录、有健身房会员管理，五花八门。

**规模**：2,810 个网站、44,227 个任务，训练用 2,560 个，验证用 250 个。

**速度**：本地托管，50-150ms 一次 action，比真网站快 10 倍，端到端 RL 训练快 30-40%。

---

## 训练怎么训

### 阶段一：SFT 暖场

用 32B agent 在 InstaV3 上跑 51,700 条轨迹，用 LLM judge 筛掉失败的，只留成功的做 supervised fine-tuning。这一步让模型先学会"基本的上网行为模式"。

### 阶段二：RL 精修

**算法**：Dr. GRPO（GRPO 的改进版）。

**为什么不用标准 GRPO**：标准 GRPO 把 advantage 做 std normalization（$\hat{A}_i = (r_i - \mu)/\sigma$），这在长 trajectory 上有个隐性 bias——长轨迹 reward 方差天然大，被 $\sigma$ 除完之后 advantage 被压扁，等于变相惩罚长 trajectory。Web 任务经常 20-30 步，这个 bias 会逼 agent 走捷径。

Dr. GRPO 去掉 std normalization，只用 $\hat{A}_i = r_i - \mu$，让长 trajectory 不被惩罚。

**Reward 怎么算**：web 任务太开放没法程序化判断"成功了没有"，所以他们用 **GPT-4o 当裁判**。裁判看完整 trajectory（截图 + 动作序列），判断三种结果：
- `correct`：任务完成
- `incorrect`：agent 自己搞砸了
- `website failure`：agent 做得对但被技术问题挡住了（timeout、空白页、CAPTCHA）

第三种标签是关键创新——这些 trajectory **从训练 batch 里扔掉**，不算 agent 的错。否则 agent 会把"网页挂了"也归咎于自己，学出一个过度悲观的 policy。跟人类经验对齐：网站卡了不是你的错，别自我怀疑。

**Rollout filtering**：oversample 之后只保留"组内有成功也有失败"的 prompt 组——全成功或全失败的组 advantage 全是 0，没梯度信号。这是 DAPO 的思路。

**训练规模**：8 张 B200 GPU，每步 1,024 条 rollout（256 prompt × 4 rollout/prompt），每条最多 25 步交互，**单步约 28 分钟**。如果用 live web 走 1.5s/action，单步要 5 小时，根本训不动。

---

## 结果有多猛

### 主表（Table 1）的核心数字

**30 步预算，pass@1**：
- Weblica-8B：39.2%（Online-Mind2Web）
- 最强开源 baseline MolmoWeb-8B：35.3%（用了 ≥100 步 + 最多 10 次 retry）
- OpenAI CUA：58.3%（100 步）
- Gemini CUA：57.3%（100 步）

**60 步预算**：Weblica-8B 平均 47.6%，已经超过 MolmoWeb 的 42.4%

**240 步预算（pass@8）**：Weblica-8B 平均 68.9%，逼近 Gemini CUA 的 60.8%

翻译成人话：**用更少的步数打赢了所有同体量开源模型，用更多步数逼近了闭源 API 模型**。

### Test-time scaling 的关键发现

Figure 4a 有个特别有意思的对比：

- Weblica-8B：15→30 步预算，pass@1 从 32.6% → 39.2%（用得上更多步数）
- Base Qwen3-VL-8B：15→30 步预算，曲线几乎重叠（给更多步数也没用）

**直觉解释**：base model 给它更多步数也不知道干嘛——它没有"走错了回头重试"的 meta-skill，卡住了就卡住了。RL 训练把这种"试错恢复"能力学进了 policy。这跟你之前反复强调的观点完全一致——test-time compute scaling 需要 training-time 配合，光给 inference budget 不够，model 得先学会怎么用这个 budget。

### Ablation 的两个关键发现

**发现一：Synth 比 Cache 略好但不绝对**

| | Online-Mind2Web | DeepShop | WebTailBench |
|---|---|---|---|
| Cache only | 35.3 | **35.8** | 30.2 |
| Synth only | **39.2** | 34.2 | **33.5** |

Synth 在大多数 benchmark 赢，Cache 在 DeepShop（电商）赢——因为真实电商网站的产品目录复杂度 synth 造不出来。两者互补，但 mixing ratio 怎么调都没找到比 50/50 更好的，留给未来 curriculum learning。

**发现二：SFT 对小模型关键，对大模型收益递减**

| Model | Base | SFT only | RL only | SFT+RL |
|---|---|---|---|---|
| 2B | 13.3 | ~17 | ~20 | 24.1 |
| 4B | 23.2 | ~28 | ~31 | 35.2 |
| 8B | 28.6 | ~33 | ~36 | 39.2 |

2B 模型 cold-start RL（不经过 SFT 直接 RL）效果差，需要 SFT 先注入 prior。8B 模型 cold-start RL 已经不错了，SFT 带来的增量变小。**如果未来 base model 再大 10 倍，SFT 阶段可能可以直接跳过**。

**发现三：Grounding 没退化**

Weblica 训练数据里**完全没有 grounding-specific 数据**（没有"点这个按钮"的标注），但 ScreenSpot-v2 / ScreenSpot-Pro 等 grounding benchmark 上性能没掉甚至略涨。这说明 web navigation benchmark 上的提升**来自更好的多步规划能力**，而**不是更好的视觉定位能力**。瓶颈不在"看不准"，在"不知道接下来该干嘛"。

---

## 这篇论文真正的贡献是什么

表面上看是"又一个 web agent"，但深层有三件事：

**第一，把 environment construction 变成 LLM-driven pipeline**。过去搭 web 训练环境需要工程师手动搭网站、写 task、写 verifier。Weblica 用 Claude Code 自动生成网站 + GPT-4o 自动判 reward + GPT-5.2 自动提取能力——整个数据 pipeline 是 agent 驱动的。这是 "用 agent 造 agent" 的递归。

**第二，证明了 RL 在 web domain 可以 scale**。之前 web agent 的工作基本停在 SFT，RL 因为环境问题做不大。Weblica 用 cache + synth 把环境 scale 到几千个，RL 训练跑通了，而且 test-time scaling 效果显著——pass@k 近乎线性增长。这暗示 web agent 还远没到 RL 的 scaling ceiling。

**第三，引入 ternary reward signal**。`website failure` 这个标签看似工程 trick，实际是把 RLHF 的 binary reward 升级成 ternary——区分"我错了"和"环境有问题"。这对所有 stochastic environment 下的 RL training 都有借鉴意义，不局限于 web。

---

## 局限和未来方向

Paper 自己列了几条，挑最重要的说：

1. **Cache 是静态快照**，不反映网站随时间演化。部署到 live web 时可能 stale
2. **Synth 有 sim-to-real gap**，真实网站的个性化推荐、动态定价、复杂 state management 造不出来
3. **单轮任务**，没有多轮对话、human-in-the-loop、跨 session memory。真实 web usage 是"帮我订机票，哦不对改成下周的"这种 evolving goal
4. **只覆盖 web**，没覆盖 mobile / desktop app。Paper 最后说 "extending to generalist computer-use agents is a promising direction"——这是明确的下一步

---

## 我的整体判断

这篇工作本质上是把 web agent 训练从 "scale human demonstration" 范式切到 "scale agentic environment generation" 范式。数据不再靠人标注或老师 agent 演示，而是靠 agent 自己造环境、自己 judge、自己 RL。如果这条路线继续走通——更多 visual diversity、更复杂 stateful interaction、multi-turn + personalization——web agent 离 "真的能帮你上网办事" 会越来越近。

---

# WEBLICA: Visual Web Agents 的可规模化 + 可复现训练环境

Andrej, 这篇 paper 直接 tackle 的核心痛点是 web agent 训练数据的 scaling 问题——RL 训练需要可交互的环境，但 live web 不可复现、不稳定、有 bot detection；simulated environment (WebArena / VisualWebArena) 覆盖太窄。Apple 的解法是把两条路 merge：HTTP-level caching 抓真实 web 的 "frozen snapshot + interactive replay"，再用 LLM coding agent 合成大批量训练环境，做到 local serving、50-150ms/action、可复现。让我把它拆开讲。

---

## 1. POMDP Formulation 的具体形式

Web navigation 被建模成 POMDP $(S, \mathcal{A}, \mathcal{O}, T, R)$：

- $S$: browser 完整 internal state (DOM, JS heap, navigation history, cookies, localStorage...). Agent 看不到全部。
- $\mathcal{A}$: coordinate-based action space，关键设计是 **不用 set-of-marks / accessibility tree / DOM**，直接吃 screenshot 预测 pixel 坐标 $(x, y)$。Table 6 给出全集：
  - `click(x, y)`, `hover(x, y)`, `type(text, [x, y], [enter])`, `press(key)`, `scroll(direction, [amount])`, `go_back()`, `go_forward()`, `wait()`, `stop(response)`
- $\mathcal{O}$: observation $o_t = (s_t, u_t)$，其中 $s_t$ 是 $1280 \times 720$ screenshot，$u_t$ 是 URL string。**纯视觉**。
- $T(s_{t+1} \mid s_t, a_t)$: 用 Playwright 实现的 transition function（关键在于 Playwright 既支持 record 也支持 replay，这层是他们整个 cache 系统的承重墙）。
- $R$: 非程序化、非 string-match，**LLM-as-Judge**（GPT-4o），后面单独讲。

Policy: $\pi_\theta(a_t \mid o_{\le t}, \tau)$，ReAct-style。每步先吐 reasoning trace $r_t$，再吐 action $a_t$，append 进 history。

Intuition: 把 web agent 训练搬回 POMDP 框架的关键 trick 是把 "环境的不可复现性" 从 transition function 里挤出去——通过 cache 让 $T$ 在多个 epoch 之间 deterministic，否则 RL 训练 reward signal 完全是噪声。

---

## 2. Weblica-Cache：HTTP-level Caching 的核心 trick

### 2.1 为什么 HTTP-level 而不是 DOM snapshot

DOM snapshot 只能复现一帧，没有交互；HTTP-level cache 能保留所有 XHR / fetch / asset 请求，让页面在 replay 时仍然能响应 click / scroll / form submit——只要被请求的资源在 cache 里。这层 trick 类似 World of Bits (Shi, Karpathy et al., 2017, https://arxiv.org/abs/1703.00113) 当年的 minitasks，但 Weblica 把它从 toy minitasks 推到 146K 真实网站。

### 2.2 Volatile Parameter Problem —— 整套 cache 的真正难点

Recording 时 capture 所有 HTTP traffic，按 normalized request signature 索引。问题：很多 query param / header 是 volatile 的，比如 `_ts=1723456789`, `session_id=abc123`, `csrf_token=...`。这些 param 在两次 visit 之间变化，直接做 key 会 cache miss。

Weblica 的自动化 rule 生成 pipeline：

1. **Record**: 让 Qwen3-VL-32B-Instruct 跑一次 task，capture 所有 request parameters，**不做任何过滤**。
2. **Playback (cache miss detection)**: 再跑一次，把 cache miss 的 request 拿出来。
3. **Fuzzy-match**: miss 的 request 和 recording 里的 request 做 fuzzy match，找出 which parameters changed across visits——这些就是 volatile params。
4. **Rule synthesis**: 生成 site-specific caching rules：从 cache key 里 strip 掉 volatile params；对 non-essential endpoints (analytics, telemetry) 生成 synthetic responses (e.g., 204 No Content)。
5. **Validate**: 在 complete network isolation 下 playback，只有 agent 能完整 solve task 的 session 才被 retain。

最终从 InstaV3 (Trabucco et al., 2025, https://arxiv.org/abs/2502.06776) 的 146K task pool 里 retain 15.6K cached environments + tasks，叫 **Weblica-Cache**。

Intuition: 这是 LLM-aided reverse engineering of HTTP cache rules。过去要写 VCR / mitmproxy 的工程师手工写 rewrite rules，现在用 "record → replay → diff → synthesize rule" 把这件事自动化了。这套路子很像 Microsoft 的 NEMO 或 BrowserGym 的 recorder，但 scale 上去了一个数量级。

---

## 3. Weblica-Synth：LLM Coding Agent 生成可交互 web 环境

### 3.1 Capability Extraction

- 用 Qwen3-VL-32B-Instruct 跑 Online-Mind2Web (Xue et al., 2025, https://arxiv.org/abs/2504.01382) 的 task，收集成功 + 失败 trajectory 的 screenshots。
- 用 **GPT-5.2** (OpenAI 2025-12, https://openai.com/index/introducing-gpt-5-2/) 分析 screenshot pair，extract fine-grained web interaction capabilities。
- 得到 **19,721 个 fine-grained capabilities**（如 "tab interface navigation", "open dropdown menu"）。
- 聚合成 **144 个 high-level capability groups**（navigation / form input / date selection / map interaction 等）。

### 3.2 三维采样保证 diversity

每次生成 sample 三个 axis：

1. **Target capability group** (从 144 个里采)
2. **Website category**: 从 1,160 个 domain 里采（aviation, banking, yoga studio, zoology, ...）
3. **Visual style**: 从 961 个 style 里采（Editorial, Minimalist, Skeuomorphic, Duotone, ...）

Paper 在 §3.3 里特别强调："without them, generated websites converge to a narrow visual style and content" —— 这是 LLM 合成数据典型 failure mode，diversity 会塌缩。

### 3.3 生成 pipeline

- **Generator**: Claude Code + Opus 4.5 (Anthropic, https://www.anthropic.com/news/claude-opus-4-5), 写 framework-free HTML/CSS/JS（无 React / Vue 依赖，纯 vanilla）。
- **Stateful via localStorage**: 因为是 static site 没 backend，用 `localStorage` 跨 step 保存 cart 之类状态。
- **Image assets**: Z-Image-Turbo (Z-Image team, https://arxiv.org/abs/2511.22699) 生成产品图 / banner。
- **Self-validation**: Claude Code 自己用 Playwright 截图、检查 functional & no CSS issue，iterate 直到 pass。

### 3.4 规模

- 310 sites 覆盖 high-level capability groups
- 2500 sites 覆盖 fine-grained capabilities (按 frequency 排序)
- **Train**: 2560 sites, 44,227 tasks (Weblica-train)
- **Val**: 250 sites, 500 tasks (Weblica-val)
- Action-to-screenshot: 50-150ms vs real web ~1.5s（10× 速度提升，通过 local serving + Playwright animation skipping 实现）
- End-to-end RL training **30-40% 速度提升**

Intuition: 这是把 "agentic coding → agentic environment generation" 的递归。Claude Code 自己就是 agent，被用来生成训练 agent 的 environment。"用 agent 造 agent" 这个 pattern 你在 Eureka Labs / 之前的 talk 里也提过——agent 生成的 synthetic data 是 scaling 的下一代路径。

---

## 4. SFT 数据与 LLM-as-Judge Reward

### 4.1 SFT Trajectories

- Qwen3-VL-32B-Instruct 在 InstaV3 queries 上 rollout，多次 sample with diverse sampling params 提 coverage。
- 用 LLM judge filter，只 retain successful completion。
- 总计 **51.7K SFT trajectories**。
- 长度分布见 Figure 18。

### 4.2 LLM-as-Judge Reward (GPT-4o)

Judge prompt 设计很精细：

```
System: Analyze trajectory steps, determine if agent successfully completed task: {task}
Respond with one of:
- 'correct'
- 'incorrect'
- 'website failure'   ← 这个是关键创新
```

**'website failure' 标签的作用**：当 agent 行为合理但被 technical issue (timeout / 5xx / CAPTCHA / 空白页 / element 死活不 interactive) 阻挡时，这条 trajectory **从 training batch 里 discard**，不算 agent 的 fault。这避免了 agent 学到 "把错归到自己头上" 的悲观 bias。

Judge agreement with humans: **88%** (Appendix C.4)。

Intuition: 这其实是 RLHF 里desirability of "third outcome" 的一个具体工程实现。传统 binary reward 在 stochastic environment 下会让 agent over-attributing failure to self，导致 policy collapse。引入 'website failure' 等价于 importance sampling 里的 rejection sampling——把 environment-induced variance 从 gradient 里剔除。

---

## 5. RL Training: Dr. GRPO

### 5.1 算法选择

用 **Dr. GRPO** (Liu et al., 2025, https://arxiv.org/abs/2503.20783)，是 GRPO (DeepSeekMath, Shao et al., 2024, https://arxiv.org/abs/2402.03300) 的变体。区别在 advantage normalization：

**Standard GRPO**:
$$\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon}, \quad \mu = \frac{1}{n}\sum_j r_j, \quad \sigma = \sqrt{\frac{1}{n}\sum_j (r_j - \mu)^2}$$

其中 $r_i$ 是第 $i$ 个 rollout 的 reward，$n$ 是 group size (rollouts per prompt)。这里 normalization by $\sigma$ 等价于 implicit length bias——长 trajectory 容易有高 reward variance，被 $\sigma$ 拉平之后 advantage 被压扁。

**Dr. GRPO** 移除 std normalization：
$$\hat{A}_i = r_i - \mu$$

这等价于 token-level 平均 reward 而非 trajectory-level 总 reward，解决了 long-horizon 任务里 "longer trajectory 被惩罚" 的 bias。对于 web navigation（trajectory 经常 20-30 steps）这个 bias 很关键。

### 5.2 关键超参 (Table 5)

- LR: 1e-5 / 5e-6 / 1e-6 (constant schedule)
- KL coefficient: 0 ~ 0.05 (探索了一圈)
- Batch size: 64 / 128 / 256
- Rollouts per prompt $n$: 4 / 8 / 16
- Rounds $R$: 15 / 25 / 35
- PPO mini-batch: 32
- Max grad norm: 1.0
- Precision: bf16
- Image resolution: 1280×720
- Max context: 80,000 tokens
- Max new tokens per action: 512

### 5.3 训练效率

- 8× NVIDIA B200 GPUs per experiment
- 1024 rollouts per RL step (256 prompts × 4 rollouts per prompt)
- 每条 rollout up to 25 interaction rounds
- **单步训练时间 ~28 minutes**（local environment serving 是 critical enabler——如果用 live web 走 1.5s/action，单步要 ~30 分钟 × 10 = 5 小时，不可行）

### 5.4 DAPO-style filtering

Rollout 时 oversample，filter 只保留 "mixed success signals" 的 group（i.e., group 里既有 success 又有 failure）。这是 DAPO (Yu et al., 2025, https://arxiv.org/abs/2503.14476) 的核心 trick——避免 group 全 success (advantage 全 0, 没梯度) 或全 failure (advantage 全 0, 没梯度) 的退化情况。

---

## 6. 主要结果

### 6.1 Headline Numbers (Table 1)

| Model | Total Steps | Online-Mind2Web | DeepShop | WebTailBench | Avg | Weblica-val |
|---|---|---|---|---|---|---|
| OpenAI CUA | 100 | 58.3 | 24.7 | 25.7 | 36.2 | - |
| Gemini CUA | 100 | 57.3 | 62.0 | 63.0 | 60.8 | - |
| Yutori Navigator | - | 64.7 | - | - | - | - |
| Qwen3-VL-Instruct-8B (base) | 30 | 28.6 | 24.1 | 21.8 | 24.8 | 56.9 |
| UI-TARS-1.5-7B | 100 | 31.3 | 11.6 | 19.5 | 20.8 | - |
| GLM-4.1V-9B-Thinking | 100 | 33.9 | 32.0 | 22.4 | 29.4 | - |
| Fara-7B | ≥100 | 34.1 | 26.2 | 38.4 | 32.9 | - |
| MolmoWeb-8B | ≥100 | 35.3 | 42.3 | 49.5 | 42.4 | - |
| **Weblica-8B (k=1)** | **30** | **39.2** | 34.2 | 33.5 | **35.6** | **70.6** |
| Weblica-8B (k=2) | 60 | 50.3 | 45.4 | 47.0 | 47.6 | 79.0 |
| Weblica-8B (k=4) | 120 | 60.5 | 55.9 | 60.3 | 58.9 | 84.7 |
| Weblica-8B (k=8) | 240 | 68.8 | 65.8 | 72.2 | 68.9 | 88.6 |

关键 observation：

- **30 steps 已经 beat 所有 open-weight baselines**（其中 MolmoWeb-8B 用 ≥100 steps + 最多 10 retries）
- 60 total steps avg 47.6% > MolmoWeb 42.4%
- 240 total steps avg 68.9% ≈ Gemini CUA 60.8%（已经接近 closed-source SOTA）
- Test-time scaling 极其 work，pass@k 接近线性外推

### 6.2 Test-Time Scaling (Figure 4a)

两个 axis：
1. **Pass@k**: k 次独立 attempt，total step budget 线性 scaling。$k \in \{1, 2, 4, 8\}$
2. **Per-episode action budget**: 每个 attempt 内允许的 step 数。

Weblica-8B 在两个 axis 上都 improve：
- 15 → 30 steps per attempt: pass@1 从 32.6% → 39.2%
- 30 steps per attempt, pass@1 → pass@8: 39.2% → 68.8%

**Base Qwen3-VL-8B 对比**: 15 vs 30 steps per attempt 曲线几乎重叠——base model 不会用更多 steps，RL 训练才解锁了 "long-horizon credit assignment" 能力。这点直觉上很重要：base model 在 15 步卡住后给 30 步还是同样卡住，因为它没有 "再多试几步 / 换个思路" 的 meta-skill，RL 训练把这种 skill 学进 policy 了。

Intuition: 这跟你之前强调的 "test-time compute scaling 需要训练-time 配合" 完全一致——base model 的 prior 不支持利用更多 compute，必须 RL 让它"学会 fail-then-retry / try-different-approach"。

### 6.3 Ablation: Cache vs Synth (Figure 4b)

| | Online-Mind2Web | DeepShop | WebTailBench |
|---|---|---|---|
| Base | 28.6 | 24.1 | 21.8 |
| Cache-only | 35.3 | **35.8** | 30.2 |
| Synth-only | **39.2** | 34.2 | **33.5** |

Synth 在 OM2W / WTB 更好，Cache 在 DeepShop 稍好（DeepShop 是电商，Cache 抓到了真实 product catalog 复杂度）。Paper 说他们试了 mixing ratio 但没找到更好的，留给未来 curriculum learning。

### 6.4 Training Stage Ablation (Figure 5)

Online-Mind2Web pass@1 across model sizes:

| Model | Base | SFT only | RL only (cold-start) | SFT + RL |
|---|---|---|---|---|
| 2B | 13.3 | ~17 | ~20 | **24.1** |
| 4B | 23.2 | ~28 | ~31 | **35.2** |
| 8B | 28.6 | ~33 | ~36 | **39.2** |

关键 finding: **SFT initialization 对小 model 关键，对大 model 收益递减**——这是 scaling law 的体现，大 model 自己有足够 prior 不需要 SFT warm-up 也能 cold-start RL。

### 6.5 Grounding 不退化 (Table 2)

| Model | MMBench-GUI | ScreenSpot-v2 | ScreenSpot-Pro |
|---|---|---|---|
| Qwen3-VL-Instruct-8B | 82.85 | 93.95 | 54.71 |
| Weblica-8B | **83.74** | **94.50** | **55.28** |

关键 insight: Weblica 训练**完全没有 grounding-specific data**，但 grounding 没退化甚至略涨。这说明 navigation benchmark 上的 gain 来自 "improved navigation behavior"（什么时候点哪里、怎么 chain actions），而**不是 visual grounding 变好**。这是一个比较反直觉的 finding——大家容易假设 web agent 的瓶颈在 grounding，但实际瓶颈在 multi-step planning。

---

## 7. Limitations 与未来方向

Paper 自己列了 5 个 limitation，我觉得其中 3 个特别值得展开：

### 7.1 Cache 环境的 sim-to-real gap

Cache 是 frozen snapshot，不反映 website 随时间的演化（layout 改版、新 feature）。这意味着 cache-trained agent 部署到 live web 时会 stale。Paper 提到 "进一步探索方法 close this gap" 是 open direction。

### 7.2 Synth 环境的 sim-to-real gap

合成环境 capture "core navigation pattern" 但不 capture "real website 全部 complexity"。比如真实 e-commerce site 有 personalized recommendation、动态 pricing、库存同步——这些 synth 全没有。更强的 generative model（能生成更 faithful web）是 next step。

### 7.3 Single-turn tasks

当前每个 episode 是 isolated task，goal 在 episode 开始就固定。真实 web usage 是 multi-turn、human-in-the-loop、有 personalization & memory across sessions。这块是 web agent 的下一个 frontier，也对应你之前对 "agentic RL" 的兴趣。

---

## 8. 跟相关 work 的位置关系

### 8.1 vs WebArena / VisualWebArena

WebArena (Zhou et al., 2023, https://arxiv.org/abs/2307.13854) 是 self-hosted 真实开源 CMS (GitLab, Reddit, OneStopShop) 组合，hard-coded 几个 site。VisualWebArena (Koh et al., 2024, https://arxiv.org/abs/2401.13603) 加了 visual tasks。两个都只有 handful of domains，diversity 限制 generalization。Weblica 用 LLM 合成把 diversity 推到 2810 sites。

### 8.2 vs WebGym

WebGym (Bai et al., 2026, https://arxiv.org/abs/2601.02439) 走 live web + RL，有 reproducibility issue + training instability（timeout, bot detection）。Weblica 用 cache + synth 两条腿绕开这个。

### 8.3 vs Fara / MolmoWeb / OpenCUA / AgentTrek

这些都是 SFT-only approach：
- Fara (Awadallah et al., 2025, https://arxiv.org/abs/2511.19663): 145K trajectory, 70K domain
- MolmoWeb (Gupta et al., 2026, https://arxiv.org/abs/2604.08516): 100K synthetic + 30K human + GUI perception
- OpenCUA (Wang et al., 2025, https://arxiv.org/abs/2508.09123): demonstration data
- AgentTrek (Xu et al., 2024, https://arxiv.org/abs/2412.09605): trajectory synthesis via tutorial replay

**Weblica 的 unique value 是 RL training**。SFT-only 训练的 agent 缺 exploration / trial-and-error，在 stochastic real web 上脆。Weblica 用 LLM-judge reward + Dr. GRPO 解锁了 RL 在 web domain 的 scaling。

### 8.4 vs UI-TARS / UI-TARS-2

UI-TARS (Qin et al., 2025, https://arxiv.org/abs/2501.12326) 是 7B GUI agent 的 early strong baseline。UI-TARS-2 (Wang et al., 2025, https://arxiv.org/abs/2509.02544) 加了 multi-turn RL。Weblica 走纯 screenshot + coordinate，无 set-of-marks，UI-TARS 系列也走这条路。差异：UI-TARS-2 强调 multi-turn RL，Weblica 强调 environment diversity scaling。

### 8.5 vs Yutori 的 "Bitter Lesson for Web Agents"

Yutori 在 https://yutori.com/blog/the-bitter-lesson-for-web-agents 这篇 blog 里 argue：DOM / accessibility tree / set-of-marks 这些 "shortcut" representation 最终会被纯 screenshot + scale 击败。Weblica 直接引用这个观点，paper §1 提到 "set-of-marks annotations or DOM access... can hurt generalization due to the web's inconsistent underlying structure" [47]。Yutori Navigator 在 Online-Mind2Web 是 64.7% (paper Table 1 italic，但没列 steps / pass@k 信息)，是最强 API 模型。Weblica 240-step pass@8 是 68.8%，已经超过 Yutori 公开的 64.7%（虽然 evaluation setup 不同需谨慎比较）。

---

## 9. 我的几个直觉性观察

### 9.1 "Cache + Synth" 的真正哲学

这其实是 RL agent training 的 "frozen replay" 范式的 next gen。Atari 用了 ALE (Arcade Learning Environment)，DM Control 用了 MuJoCo snapshot，但 web 之前没有 "可复现 + scale" 的等价物。Weblica 把这件事做出来了——HTTP cache 类似 MuJoCo 的 deterministic physics，LLM synth 类似 procedural terrain generation。

### 9.2 'website failure' 标签的深层意义

这个 label 不只是工程 trick，本质是把 reward model 从 binary 升级到 ternary，等价于给 RL agent 一个 "epistemic state" 概念——区分 "我不知道" vs "我错了"。这个思路在数学 RL (e.g., RLHF with abstain) 和 tool-use RL 里都 emerging。

### 9.3 "Weblica-val" 这个 in-distribution benchmark 的意义

Weblica-val 是 250 个 held-out synth 环境 + 500 task，pass@1 是 70.6%。这个数字的 sub-text 是：**in-distribution 上 agent 已经接近天花板**，gap 主要在 sim-to-real。这是对 "environment generation 是 bottleneck" 假说的直接证据。

### 9.4 Dr. GRPO 的 length-bias fix 为什么对 web 关键

Web trajectory 经常 20-30 步，trajectory-level reward variance 大。Standard GRPO 的 std normalization 会让 short successful trajectory (reward=1, variance 小) 比 long successful trajectory (reward=1, variance 大) 优势大——这就 incentivize agent 走捷径而不是 robust long-horizon plan。Dr. GRPO 去掉 std normalization 让 advantage 直接是 $r_i - \mu$，长 trajectory 不被惩罚。

### 9.5 30 steps 的优势的解读

Weblica-8B 在 30 steps 已经 beat baseline 100 steps，这暗示 baseline 在浪费 steps——它们可能在错误的 page 上转圈，而 Weblica 学到了 "确认走错就 go_back / 换 navigation path" 的 meta-skill。这跟 §6.2 test-time scaling 的发现一致：base model 不会用 extra steps，因为它没有 "试错 recovery" 的 prior。

### 9.6 SFT 的 diminishing return

§5.4 的 finding "SFT initialization 对小 model 关键，对大 model 收益递减" 是一个 scaling-law-level 的 observation。这暗示：**SFT 在 small model 上相当于 inject prior**，大 model 自己 prior 够强不需要。如果未来 base model 再大 10×，可能 cold-start RL 直接训就行，SFT 阶段可以省略。

---

## 10. 一些 Open Questions / Future Direction

Paper §6 列了一些，我补几个：

1. **Reward hacking on synth environment**: LLM-judge 在 synth 环境上可能被 syntactic cue 欺骗（比如 agent 学会 "用 stop action 加上看起来像正确答案的 text" 来骗 judge）。Paper 没分析这个。
2. **Cache 环境的 long-horizon capability**: HTTP cache 不支持需要 server-side state change 的 task（比如真的下单、真的发邮件），所以 Weblica-Cache 上的 task 都是无 side-effect 的 navigation / form-fill。这限制了 RL 训练的任务类型。
3. **Visual diversity 的实际 effectiveness**: 961 个 visual style 听起来很多，但 Claude Code 生成时实际可能 collapse 到几个高频 style。Paper Figure 19 给了分布但没量化"effective diversity"（比如用 CLIP feature 的 entropy）。
4. **Real-world long-tail**: Weblica-val 是 in-distribution synth 的 held-out，但真实 web 的 long-tail（多语言、accessibility、weird DOM pattern）可能没被 synth 覆盖。Paper Limitations 提了 sim-to-real gap 但没量化。

---

## 11. 推荐延伸阅读

- **World of Bits** (Shi, Karpathy, et al., 2017): https://arxiv.org/abs/1703.00113 —— 你自己的 old work，HTTP caching for web minitasks 的祖先
- **InstaV3** (Trabucco et al., 2025): https://arxiv.org/abs/2502.06776 —— Weblica-Cache 的 task source
- **Dr. GRPO** (Liu et al., 2025): https://arxiv.org/abs/2503.20783 —— RL 算法
- **DAPO** (Yu et al., 2025): https://arxiv.org/abs/2503.14476 —— rollout filtering 思路来源
- **Online-Mind2Web** (Xue et al., 2025): https://arxiv.org/abs/2504.01382 —— 主要 eval benchmark
- **MolmoWeb** (Gupta et al., 2026): https://arxiv.org/abs/2604.08516 —— 最强 open-weight baseline
- **Yutori's Bitter Lesson for Web Agents**: https://yutori.com/blog/the-bitter-lesson-for-web-agents —— 哲学 align
- **UI-TARS-2** (Wang et al., 2025): https://arxiv.org/abs/2509.02544 —— multi-turn RL GUI agent 的并行 work
- **WebArena**: https://arxiv.org/abs/2307.13854
- **VisualWebArena**: https://arxiv.org/abs/2401.13603
- **Claude Opus 4.5**: https://www.anthropic.com/news/claude-opus-4-5 —— synth generator backbone
- **GPT-5.2**: https://openai.com/index/introducing-gpt-5-2/ —— capability extractor

---

整体我的 take：这是 web agent 训练从 "scale human demonstration" 范式转向 "scale agentic environment generation" 的标志性工作。把 environment construction 本身变成一个 LLM-driven pipeline（capability extraction + diverse sampling + self-validation），是 RL scale 上去的真正路径。如果 Apple 接下来把这条 pipeline 推到 mobile / desktop GUI (paper §6 提了 "beyond web")，加上 long-horizon RL 和 personalization，会是一个真正的 generalist computer-use agent 训练 stack。
