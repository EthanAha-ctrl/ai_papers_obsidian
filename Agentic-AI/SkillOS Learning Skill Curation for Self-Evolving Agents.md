---
source_pdf: SkillOS Learning Skill Curation for Self-Evolving Agents.pdf
paper_sha256: e55b880b9564c63f5f791c02f423d41bd356eecc07002f85c6babfce72ee39c6
processed_at: '2026-08-12T07:21:33-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SkillOS 到底在干嘛

## 1. 一个故事开头

想象你雇了一个程序员叫 Alex，能力还不错，但有个问题：**他做完一个项目就把所有经验全忘了**。下一个项目又从零开始，之前踩过的坑重新踩一遍。

你给他配了个助理叫 Sam。Alex 不变（frozen），Sam 一直在学。每次 Alex 做完一个 task，Sam 就翻 Alex 的工作记录，想三个问题：
- 这次的经验是不是值得单独记一条新笔记？(`insert`)
- 有没有老笔记需要更新一下？(`update`)
- 有没有过时或者误导性的笔记可以删掉？(`delete`)

时间一长，Sam 手里攒了一个"经验手册"（SkillRepo）。下次 Alex 接新活，Sam 先翻手册，把相关的那几页抽出来塞给 Alex："这次用这个"。Alex 就不用瞎试了。

**这就是 SkillOS 的全部 idea**。整个 paper 就是回答一个问题：**怎么让 Sam 学会整理这本手册？**

---

## 2. 为什么这件事难？

直觉上让 LLM 自己整理笔记很容易啊——prompt 它"把你学到的经验总结成 markdown"不就行了？[ExpeL](https://arxiv.org/abs/2308.10144) 这么干，[MemGPT](https://arxiv.org/abs/2310.08560) 这么干，[Voyager](https://arxiv.org/abs/2305.16291) 也这么干。这些方法都靠**人工写的规则**告诉 agent 什么时候该记、什么时候该删。

问题在于：**这些规则是工程师拍脑袋写的**，跟下游 executor 用不用得上完全脱节。你可能让 agent 记了一大堆"听起来很有道理但 executor 根本不会用"的笔记，或者该删的没删导致仓库爆炸。

更糟的是 **credit assignment**：Sam 在 task 5 决定 update 一条 skill，这条 skill 帮助了 task 12，这个反馈要等到 task 12 才出现。中间隔了 7 个 task。Sam 怎么知道 task 5 的 update 是好是坏？这就是 RL 里经典的 delayed reward 问题。

SkillOS 的答案：**用 RL 训练 Sam，但把 task 序列人工分组，让有依赖的 task 挨在一起，这样 delayed feedback 来得快一点**。

---

## 3. 技术细节：三个关键设计

### 3.1 Skill 长什么样

每个 skill 就是一个 Markdown 文件：

```markdown
---
name: find_object_under_lamp
description: use this when task says "look at X under Y"
---

# How to find objects under light sources

1. Locate the light source (desklamp, flashlight, etc.)
2. Search the area immediately around / under it
3. If object not found, expand search radius
4. Avoid opening unrelated containers
```

这跟 [Anthropic 的 Skills](https://github.com/anthropics/skills) 设计一样，但简化了——Anthropic 真正的 skill 是文件夹带脚本，SkillOS 只用单个 Markdown。论文 Limitations 里说"以后要支持多文件"。

Sam 操作这个 repo 用三个 function call，签名在 Figure 8 里：

```
insert_skill(skill_name, content)
update_skill(skill_name, new_name?, new_content?)
delete_skill(skill_name)
```

这相当于给 Sam 一个"操作系统的文件 I/O"接口。LLM 输出 JSON function call，外部解析后真的改 SkillRepo。

### 3.2 Grouped Task Stream：解决 credit assignment 的核心 trick

这个设计是整篇 paper 最聪明的地方。讲讲为什么。

如果 task 之间完全独立（比如随机抽的 AIME 数学题），Sam 在 task 1 学到的 skill 几乎不可能在 task 2 用上。Sam 的 update 决策好不好，根本没机会被检验，RL 信号就废了。

SkillOS 的做法：**先把训练数据按"skill 依赖性"分组成 stream**。

用 [Gemini-2.5-Pro](https://arxiv.org/abs/2507.06261) 给每个 task $x_i$ 标注 attribute set：

$$Z_i = (T_i, S_i, C_i, R_i, P_i)$$

变量解释：
- $T_i$ = Topics（比如 "Fourier transform", "algebra"）
- $S_i$ = Skills/Capabilities（比如 "substitution", "integration by parts"）
- $C_i$ = Math Concepts（比如 "Parseval's theorem"）
- $R_i$ = Heuristic Strategies（比如 "try special cases first"）
- $P_i$ = Common Pitfalls（比如 "forgetting the constant of integration"）

然后用 phrase-level soft-Jaccard 相似度配对 task。配对的 gate 有 6 个条件（附录 B.2.2）：
1. **Shared foundation**: 至少共享 1 个 concept + 1 个 skill
2. **Shared reasoning**: heuristic 和 pitfall 加起来至少匹配 1 个
3. **Not near-duplicate**: topic 相似度不能太高（避免同一题）
4. **Not too unrelated**: 整体相似度不能太低
5. **Progression**: 后面的 task 至少引入 1 个新 concept 或 skill
6. **Curriculum direction**: 难度递增

**为什么这个重要？** 因为它构造了一个"高中→大学"的课程结构。早期 task 让 Sam 建立基础 skill，后期 task 用上这些 skill，RL 信号就闭环了。

Ablation 里去掉 grouping，SR 从 61.2 掉到 57.3（Table 3）。这是最大的 degradation，证明 grouping 是核心。

### 3.3 Composite Reward：dense signal 救 RL

reward 公式：

$$r = r^{\text{task}} + \lambda_f \cdot r^{\text{fc}} + \lambda_u \cdot r^{\text{cnt}} + \lambda_c \cdot r^{\text{comp}}$$

各项含义，用人话讲：

- **$r^{\text{task}} = \frac{1}{|G|-1}\sum_{i=2}^{|G|}\mathbb{1}(\xi_i)$**：组内除第一个 task 的平均成功率。为什么跳过第一个？因为第一个 task 时 SkillRepo 是空的，Sam 还没干活，自然不该背锅也不该邀功。

- **$r^{\text{fc}} = \frac{1}{|G|}\sum \text{Valid}(c_i)$**：function call 有效性。Sam 生成 JSON，JSON 解析失败的就不算。这阻止 Sam 学会输出乱码来骗 reward。

- **$r^{\text{cnt}} = \frac{1}{|G|}\sum \text{Judge}(c_i)$**：用 [Qwen3-32B](https://arxiv.org/abs/2505.09388) 当外部 judge 给 Sam 输出的 skill content 打分。这是 dense signal，让 RL 早期阶段（task 全 fail）也能学东西。

- **$r^{\text{comp}} = \frac{1}{|G|}\sum \left(1 - \frac{|S_i|}{|\chi_i|}\right)$**：压缩奖励。$|S_i|$ 是更新后 repo 的 token 数，$|\chi_i|$ 是 Sam 输入 context 的 token 数。如果 Sam 把整段 trajectory 原样塞进 repo，$|S_i| \approx |\chi_i|$，reward 接近 0。这阻止 Sam 偷懒复制粘贴。

权重：$\lambda_f = 1.0, \lambda_u = 0.1, \lambda_c = 0.05$。任务成功率最重要，content 质量其次，压缩只是 safety net。

**为什么这个设计聪明**：纯 task success reward 在训练早期全是 0（curator 还没学好，executor 用不上 skill），RL 学不动。三个辅助 reward 提供 dense gradient，让 Sam 在"还不会帮上忙"的阶段也能学到"至少生成合法的、有意义的、不啰嗦的 skill"。

---

## 4. 训练怎么跑：GRPO 的角色

用 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)。它跟 PPO 的区别是**没有 critic**，用 group-relative baseline：

$$A^n = r^n - \frac{1}{N}\sum_{n'=1}^N r^{n'}$$

变量：
- $r^n$ = 第 $n$ 个 rollout 的 composite reward
- $N$ = group size = 8（每个 task group 跑 8 次独立 rollout）
- $A^n$ = 第 $n$ 个 rollout 相对于组内平均的 advantage

直觉：与其学一个 critic 网络估"这个状态值多少分"，不如直接看 8 次尝试里"我这次比平均好多少"。这跟 [AlphaGo Zero](https://www.nature.com/articles/nature24270) 用 self-play 当 baseline 是同一个思路——用真实的对比代替估计的价值。

Loss 是标准 PPO clip（公式 2）：

$$\mathcal{L} = \mathbb{E}_n\left[\min\left(\rho^n A^n, \text{clip}(\rho^n, 1-\epsilon, 1+\epsilon)A^n\right)\right]$$

- $\rho^n = \pi_S(c^n|\chi) / \pi_{\theta_{old}}(c^n|\chi)$ = 新旧 policy 给同一 curation sequence 的概率比
- $\epsilon$ = clip 范围，防止 policy 跑偏太远

**一个细节**：他们**丢弃了 KL 项**。原版 GRPO 有 KL penalty $\beta \cdot \text{KL}(\pi_S || \pi_{ref})$ 防止 policy 偏离 reference 太远，但 SkillOS 完全去掉了，"to encourage policy exploration"。这是合理的——curation space 非常 under-explored，初期需要 strong push 去发现新的 skill 编辑模式。

训练用 [verl 框架](https://arxiv.org/abs/2409.19256)，16 张 H100，ALFWorld 3 天，reasoning 2.5 天，WebShop 5 天。学习率 $1e-6$，batch 32，group 8。

---

## 5. 实验结果：三个让人意外的发现

### 发现 1：8B curator 打 Gemini-2.5-Pro 直接做 curator

这是论文最 punchy 的 claim。看 Table 1：

| Executor | SkillOS-gemini (Gemini curator) | **SkillOS** (Qwen3-8B curator) |
|---|---|---|
| Qwen3-8B executor | 50.7 | **61.2** |
| Qwen3-32B executor | 63.6 | **68.6** |
| Gemini-2.5-Pro executor | 79.3 | **80.2** |

Qwen3-8B 是个 8B 的小模型，Gemini-2.5-Pro 是 Google 的旗舰。直接 prompt Gemini 去 curate skill，居然不如 RL 训练过的 8B。

**人话解释**：Gemini 写作能力再强，它不知道 Alex（executor）会怎么用它写的 skill。它生成的 skill 太抽象、太通用、跟 executor 的"使用习惯"对不上。就像大学教授的笔记不一定适合小学生用。

RL 训练过的 Qwen3-8B curator 学到的是"**针对这个 executor 的 curation 风格**"——什么表述 executor 看得懂，什么细节 executor 容易忽略，什么结构 executor 不会用错。这种 ground knowledge 只能从交互中学。

这个 finding 对 self-evolving agent 领域非常重要：**curation is a learned skill, not an emergent capability**。

### 发现 2：Executor 越强，SkillOS 增益越大

看 SkillOS vs SkillOS-base 的差：

- Qwen3-8B executor：+7.9
- Qwen3-32B executor：+8.8
- Gemini-2.5-Pro executor：+9.5

直觉上我们会觉得 executor 越强，给它配 skill 帮助越小。结果反过来。论文解释：**强 executor 更能"榨干" skill 的价值**。

人话讲：菜鸟员工你给他再好的笔记他也用不好，老手你给他几条精炼提示他能起飞。Skill curation 不是补短板，是放大长板。

### 发现 3：训练动力学里的"phase transition"

Figure 4 显示训练过程中三个操作的比例变化：

- **训练初期**：insert 占绝对主导（70%+）。Sam 像"啥也不懂的新助理"，拼命塞新笔记。
- **训练中期**：update 崛起，insert 下降。
- **训练后期**：update 主导（50%+），insert 降到 20%，delete 缓慢上升到 10%。

这跟人类学习曲线高度吻合：**初学者积累，老手整理，大师剪枝**。

Figure 5 更有意思——skill 内容的演化：
- 早期：Sam 加 "additional guidance"、"tips" 这种 generic section，skill 变啰嗦但不实用
- 后期：Sam 加 "failure-handling logic"、"conditional branches"——可执行的控制流

而且 SkillRepo 整体结构也演化：从一堆 task-specific skill（"如何找 apple"）变成包含 meta-strategy（verification、fallback planning、strategy adjustment）的多层次 repo。

这是 RL 下的**emergent abstraction**——通过 reward pressure，curator 自发学会把具体经验压缩成抽象模式。这跟深度网络从 specific feature 学到 general feature 是同一个机制，只不过发生在符号层面。

---

## 6. Ablation 怎么看

Table 3：

| 变体 | Avg. SR | Steps |
|---|---|---|
| SkillOS full | 61.2 | 18.9 |
| w/o $r^{\text{cnt}}$（去掉 content reward） | 58.6 | 20.1 |
| w/o $r^{\text{comp}}$（去掉压缩 reward） | 60.0 | 19.3 |
| w/o grouping（随机 task 序列） | 57.3 | 20.6 |

三个观察：

1. **Grouping 最 critical**（-3.9 个点）。证明核心 idea 是对的：curation 是 long-horizon decision，需要 grouped stream 提供 feedback 通路。没有 grouping，RL 学不到"现在 update 影响未来"。

2. **Content reward 第二重要**（-2.6 个点）。dense signal 救了早期 RL。

3. **Compression reward 影响小但 consistent**（-1.2 个点）。它是 safety net，防止 trajectory 复制，不是主要 driver。

---

## 7. 跨 Executor 泛化

训练时用 Qwen3-8B 当 executor，测试时换 Qwen3-32B 和 Gemini-2.5-Pro。SkillOS 在所有组合上都 work（Tables 1, 2, 6）。

更绝的：**Gemini-2.5-Pro 直接当 curator 不如 RL 训练的 Qwen3-8B 当 curator**，即使 executor 用的是 Gemini-2.5-Pro 自己。这再次证明"raw intelligence ≠ curation ability"。

跨 domain（Figure 3）也有些有意思的发现：
- Reasoning 训练的 curator 迁移到 agentic task 表现不错——因为 reasoning skill 包含 decomposition、verification 这种 general 策略
- Agentic 训练的 curator 迁回 reasoning 不太行——因为 agentic skill 太 environment-specific

这跟我做 CoT 时的直觉一致：**抽象策略 > 具体知识**，在 transfer 上。

---

## 8. 我会质疑的地方

### 8.1 BM25 retrieval 是个明显 bottleneck

论文自己也承认。SkillRepo 一旦超过几百条 skill，BM25 的语义召回会非常弱。Karpathy 你在 [NeurIPS 2024 keynote](https://www.youtube.com/watch?v=l8pRSuUH7ic) 谈过 retrieval 是当前 RAG 的痛点，这里完全印证。下一步应该是 learned retriever 或者 agentic search（多步 query refinement）。

### 8.2 SkillRepo 会不会爆炸

Figure 4 显示 delete 始终在 10% 左右。真实 streaming 部署里 SkillRepo 可能无限膨胀。Compression reward $r^{\text{comp}} = 1 - |S_i|/|\chi_i|$ 只是软约束，没 hard cap。如果 SkillRepo 长到 BM25 都召回不动，整个系统会退化。这是部署层面没解决的问题。

### 8.3 Frozen Executor 的局限

论文说 frozen 是为了"隔离 curation 贡献"。但这意味着如果 executor 看不懂某种 skill 表述，curator 没法 retrain executor。Joint optimization 是 obvious 下一步，但论文说"训练成本太高"。

我怀疑还有个更深层的原因：**如果 executor 也训练，curator 会学到"故意写 executor 容易学的 skill"而不是"真正有用的 skill"**，reward 会 hack。Frozen executor 强制 curator 适应现实，而不是反过来。

### 8.4 LLM-as-judge 的 reward hacking

$r^{\text{cnt}}$ 用 Qwen3-32B 打分。如果 Qwen3-32B 有系统性偏好（比如喜欢长 skill，喜欢某种 style），curator 会学这个偏好而不是真正有用。论文没做 reward hacking 分析。

### 8.5 Grouped stream 在真实部署里不存在

训练时我们人工分组相关 task。真实 streaming 是 user-driven 的，task 之间可能完全无关。SkillOS 在 test-time 怎么处理无关 task？论文没讨论这个 deployment gap。我猜测训练学到的 curation policy 有一定鲁棒性（对无关 task 也知道 update/delete），但实验没专门验证。

### 8.6 Anthropic SKILL.md 的 multi-file 能力丢了

真实 Anthropic skill 是文件夹带 scripts，SkillOS 只用单 Markdown。这是为了简化 action space，但失去了 executable skill 的能力——只能存 declarative 知识，不能存可执行代码。Limitations 里提到了，是 obvious next step。

---

## 9. 联想到的相关工作

### 9.1 直接前作
- **[ReasoningBank](https://openreview.net/forum?id=jL7fwchScm)**：同一作者的前作，是本文直接 baseline。它蒸馏 reasoning insight 但用启发式 curation。
- **[MemP](https://arxiv.org/abs/2508.06433)**：procedural memory，advanced management 但 rule-based
- **[ExpeL](https://arxiv.org/abs/2308.10144)**：从 trajectory 蒸馏 insight，启发式

### 9.2 Skill library 类
- **[Voyager](https://arxiv.org/abs/2305.16291)**：Minecraft skill library，GPT-4 自动 curate。SkillOS 是 Voyager 的 learned curation 版本——把 curation 从 prompt engineering 变成 RL
- **[SkillWeaver](https://arxiv.org/abs/2504.07079)**：Web agent 自动发现 skill
- **[SkillNet](https://arxiv.org/abs/2603.04448)**：create/evaluate/connect skills

### 9.3 Memory as OS 类
- **[MemGPT](https://arxiv.org/abs/2310.08560)**：OS-inspired memory，prompt-based 控制。SkillOS 是 learned 版本
- **[A-MEM](https://openreview.net/forum?id=FiM0M8gcct)**：agentic memory

### 9.4 RL for memory/skill
- **[Memory-R1](https://arxiv.org/abs/2508.19828)**：RL for memory management
- **[SkillRL](https://arxiv.org/abs/2602.08234)**：教 small model 用 skill
- **[D2Skill](https://arxiv.org/abs/2603.28716)**：dynamic dual-granularity skill
- **[ARISE](https://arxiv.org/abs/2603.16060)**：intrinsic skill evolution
- **[GigPO](https://arxiv.org/abs/2505.10978)**：group-in-group policy optimization，跟 SkillOS 用 grouped RL 类似
- **[MEM1](https://openreview.net/forum?id=XY8AaxDSLb)**：synergize memory and reasoning
- **[UMEM](https://arxiv.org/abs/2602.10652)**：unified memory framework

### 9.5 Library learning 经典
- **[DreamCoder](https://arxiv.org/abs/2106.11590)**：library learning via neurosymbolic search。SkillOS 是它的 RL 版本
- **[AlphaProof / AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-w)**：skill 作为可组合 tactic
- **[Buffer of Thoughts](https://arxiv.org/abs/2312.08908)**：thought-as-skill

### 9.6 Streaming / Lifelong learning
- **[StreamBench](https://papers.nips.cc/paper_files/paper/2024/hash/c189915371c4474fe9789be3728113fc-Abstract-Datasets_and_Benchmarks_Track.html)**：streaming benchmark
- **[EvoMemory](https://arxiv.org/abs/2511.20857)**：test-time learning with self-evolving memory
- **[AutoScaling Memory](https://arxiv.org/abs/2510.09038)**：GUI agent memory

### 9.7 RL 方法
- **[GRPO](https://arxiv.org/abs/2402.03300)**：DeepSeekMath 的核心 RL 方法
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)**：RL for reasoning
- **[verl](https://arxiv.org/abs/2409.19256)**：训练框架

### 9.8 Karpathy 自己的脉络
你之前讲过 [Software 2.0](https://karpathy.medium.com/software-2-0-a64552e279ac)（神经网络作为 software），后来又讲 [Software 3.0](https://www.youtube.com/watch?v=LCEmiRjPEqQ)（prompt/English 作为 programming language）。SkillOS 实际上是 **Software 3.0 的 garbage collection 和 refactoring**——agent 自己重构自己的程序库。

更深层：这其实是 **meta-learning through symbolic memory**。传统 meta-learning（[MAML](https://arxiv.org/abs/1703.03400)、[Reptile](https://arxiv.org/abs/1803.02999)）通过 weight 更新做 meta-learning，需要 bilevel optimization，训练成本高。SkillOS 通过外部 symbol manipulation 做 meta-learning，只需要单层 RL，更 practical。

这条路在 LLM 时代可能比 weight-level meta-learning 更实用——因为 LLM 的 weight 已经很强了，真正瓶颈在外部知识管理。

---

## 10. 未来方向（论文 + 我的猜想）

### 论文提的
1. **Agentic search over memory**：替换 BM25 为多步 query refinement
2. **Hierarchical/compositional skills**：skill 调 sub-skill
3. **Multi-agent shared memory**：多个 agent 共享 SkillRepo，credit assignment 复杂

### 我加的
4. **Executable skills**：把 Anthropic 完整 SKILL.md 的 multi-file + scripts 能力加进来，让 skill 不只是 declarative knowledge
5. **Joint curator-executor optimization**：论文说太贵，但可以用 alternating optimization，每 N 步 curator 更新后做一次 LoRA executor fine-tune
6. **Reward hacking 防御**：用多个 judge ensemble 或者把 task outcome reward 重新加权
7. **Skill versioning**：每个 skill 维护版本历史，让 curator 学会"什么时候该 revert 到旧版本"
8. **Active task selection**：训练时让 curator 不只 curate skill，也主动选择下一个想"练习"的 task type，加速 learning

---

## 11. 一句话总结

SkillOS 干的事情，就是把"agent 怎么管理自己经验"这个看似复杂的 meta-problem，变成一个 well-defined RL subproblem：**冻住 executor，让一个小 model（curator）在 skill 文件系统上做 insert/update/delete 操作，用 grouped task stream + composite reward 把 delayed credit assignment 转化为 dense signal**。8B curator 打 Gemini-2.5-Pro 直接 curation 的结果证明：**curation 是个 learned skill，不是 raw intelligence 的副产品**。这个 finding 对整个 self-evolving agent 方向是奠基性的。

---

## 参考 Web Links

- [SkillOS paper (假设 arxiv)](https://arxiv.org/abs/2505.xxxxx)
- [Anthropic Skills](https://github.com/anthropics/skills)
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [ALFWorld](https://openreview.net/forum?id=0IOX0YcCdTn)
- [WebShop](https://arxiv.org/abs/2207.01206)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [ExpeL](https://arxiv.org/abs/2308.10144)
- [MemGPT](https://arxiv.org/abs/2310.08560)
- [DreamCoder](https://arxiv.org/abs/2106.11590)
- [verl framework](https://arxiv.org/abs/2409.19256)
- [DeepMath-103K](https://openreview.net/forum?id=kHB5Te5IWm)
- [Qwen3](https://arxiv.org/abs/2505.09388)
- [Gemini 2.5](https://arxiv.org/abs/2507.06261)
- [GPQA](https://openreview.net/forum?id=Ti67584b98)
- [ReasoningBank](https://openreview.net/forum?id=jL7fwchScm)
- [MemP](https://arxiv.org/abs/2508.06433)
- [SkillWeaver](https://arxiv.org/abs/2504.07079)
- [SkillRL](https://arxiv.org/abs/2602.08234)
- [D2Skill](https://arxiv.org/abs/2603.28716)
- [ARISE](https://arxiv.org/abs/2603.16060)
- [Memory-R1](https://arxiv.org/abs/2508.19828)
- [GigPO](https://arxiv.org/abs/2505.10978)
- [MEM1](https://openreview.net/forum?id=XY8AaxDSLb)
- [UMEM](https://arxiv.org/abs/2602.10652)
- [A-MEM](https://openreview.net/forum?id=FiM0M8gcct)
- [Buffer of Thoughts](https://arxiv.org/abs/2312.08908)
- [AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-w)
- [StreamBench](https://papers.nips.cc/paper_files/paper/2024/hash/c189915371c4474fe9789be3728113fc-Abstract-Datasets_and_Benchmarks_Track.html)
- [EvoMemory](https://arxiv.org/abs/2511.20857)
- [MAML](https://arxiv.org/abs/1703.03400)
- [Reptile](https://arxiv.org/abs/1803.02999)
- [Karpathy Software 2.0](https://karpathy.medium.com/software-2-0-a64552e279ac)
- [Karpathy Software 3.0](https://www.youtube.com/watch?v=LCEmiRjPEqQ)
- [Karpathy NeurIPS 2024 keynote](https://www.youtube.com/watch?v=l8pRSuUH7ic)
- [AlphaGo Zero](https://www.nature.com/articles/nature24270)

---

# SkillOS 深度讲解：让 Agent 学会"自我整理抽屉"的能力

Andrej 你好！这篇 paper 解决的问题非常 Karpathy 风格——它不是又做一个更强的 executor，而是问了一个更深层的问题：**LLM agent 怎么从自己的 trajectory 里学到一个"管理经验"的 policy？** 这本质是在学一个 meta-controller，而不是在学一个 solver。

---

## 1. 核心洞察：解耦 Executor 与 Curator

这篇 paper 最 critical 的设计决策是把 agent 拆成两个角色：

```
┌────────────────────────────────────────────────────┐
│  Frozen Executor π_L    ←─ 只会用 skill，不学习     │
│         ↑                          │                │
│    retrieve (BM25)            trajectory ξ_t        │
│         ↑                          ↓                │
│    SkillRepo S_t  ←────  Trainable Curator π_S      │
│                         (insert/update/delete)      │
└────────────────────────────────────────────────────┘
```

这个解耦让我立刻联想到 [DreamCoder](https://arxiv.org/abs/2106.11590) 的 dream phase 和 wake phase 分离，也让我想到 [Daydreamer](https://arxiv.org/abs/2206.14176) 的 world model vs. actor 解耦。但是 SkillOS 这里更激进——它把"在 OS 上编辑文件"这一抽象动作变成了 RL action space，相当于让 curator 在一个"知识文件系统"上做 gradient。

**关键直觉**：如果 executor 是一个"员工"，curator 就是这个员工的"个人助理"——员工不变（frozen），助理学着怎么帮他整理笔记。这个 framing 之所以重要，是因为它把 self-evolution 从一个"模型自己改自己权重"的难题，简化为"模型管理一个外部仓库"的问题。

---

## 2. SkillRepo：把 Skill 当文件系统对象

SkillOS 跟着 Anthropic 的 [Skills](https://github.com/anthropics/skills) 范式走，把每个 skill 表示为一个 Markdown 文件：

```markdown
---
name: skill_name
description: when to use this skill
---

# Skill Body
executable knowledge, workflows, constraints, heuristics
```

Curator 通过三个 function call 操作 repo：

- `insert_skill(skill_name, content)` — 创建新文件
- `update_skill(skill_name, new_name?, new_content?)` — 编辑文件
- `delete_skill(skill_name)` — 删除文件

这让我立刻联想到 [MemGPT](https://arxiv.org/abs/2310.08560) 把 memory 当作 OS 的设计。但 MemGPT 是 prompt-based 启发式控制，SkillOS 把这件事上升为 **learned policy**。这是质的飞跃——从"工程师写规则让 agent 整理内存"变成"agent 通过 RL 学会整理内存"。

---

## 3. 训练配方：Grouped Task Streams

第一个 RL 设计是解决 **credit assignment 问题**：当 curator 在 task $t$ 做了一个 update_skill 操作，这个操作好不好？答案要等到 task $t+k$ 用到这个 skill 时才揭晓。这是 **delayed, indirect feedback** 的经典难题。

SkillOS 的解法是把训练数据组织成相关的 task group：

$$\mathcal{D} = \{G_1, G_2, \ldots, G_M\}, \quad G_m = \{x_{m,1}, x_{m,2}, \ldots, x_{m,|G_m|}\}$$

其中所有 $x_{m,i}$ 共享 skill 依赖。这里 $G_m$ 是第 $m$ 组，$x_{m,i}$ 是该组第 $i$ 个 task。

**怎么 group？** 用 Gemini-2.5-Pro 给每个 task $x_i$ 标注 attribute set：

$$Z_i = (T_i, S_i, C_i, R_i, P_i)$$

变量含义：
- $T_i$ = Topics（主题）
- $S_i$ = Skills/Capabilities（所需能力）
- $C_i$ = Concepts/Theorems（数学概念）
- $R_i$ = Heuristic Strategies（启发式策略）
- $P_i$ = Common Pitfalls（常见陷阱）

然后通过 phrase-level soft-Jaccard 相似度 + 难度 curriculum 来配对。Dependency gate 包含 6 个条件（共享 foundation、共享 reasoning、非 duplicate、非 unrelated、有 progression、forward curriculum）。

**直觉**：这相当于给 agent 设计一套"高中→大学"的课程大纲。早期 task 让 curator 先建好基础 skill，后期 task 检验这些 skill 是否能 transfer。没有这个 grouping（ablation 显示掉到 57.3%），curator 学不到"我现在的 curation 决策如何影响未来"。

---

## 4. 复合奖励：Composite Reward

公式 (1) 是论文的核心 reward：

$$r = r^{\text{task}} + \lambda_f r^{\text{fc}} + \lambda_u r^{\text{cnt}} + \lambda_c r^{\text{comp}}$$

各项变量：
- $r^{\text{task}} = \frac{1}{|G|-1}\sum_{i=2}^{|G|}\mathbb{1}(\xi_i)$ — 组内除第一个 task 之外的平均成功率。为什么跳过第一个？因为第一个 task 时 SkillRepo 是空的，curator 还没动过手，所以不该归功/归咎。
- $r^{\text{fc}} = \frac{1}{|G|}\sum_{i=1}^{|G|}\text{Valid}(c_i)$ — function call 有效性，避免生成乱码 JSON
- $r^{\text{cnt}} = \frac{1}{|G|}\sum_{i=1}^{|G|}\text{Judge}(c_i)$ — Qwen3-32B 作为外部 judge 给内容打分
- $r^{\text{comp}} = \frac{1}{|G|}\sum_{i=1}^{|G|}\left(1 - \frac{|S_i|}{|\chi_i|}\right)$ — 压缩奖励，$|S_i|$ 是更新后 repo 的 token 长度，$|\chi_i|$ 是 curator 输入 context 的 token 长度。这阻止 curator 把整段 trajectory 复制进 repo

权重：$\lambda_f = 1.0, \lambda_u = 0.1, \lambda_c = 0.05$。

**直觉**：task outcome 是稀疏 delayed signal，其他三项是 dense intermediate signal。这让我想到 [AlphaGo](https://www.nature.com/articles/nature16961) 的设计——除了 final win/loss，还用 policy/value network 提供更密集的引导。这里 $r^{\text{cnt}}$ 和 $r^{\text{comp}}$ 起类似作用，避免 curator 在早期 rollout 阶段（reward 全 0）学不到任何东西。

---

## 5. GRPO 策略优化

用 [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300) 而不是 PPO，因为 group-relative baseline 省去了 critic：

$$A^n = r^n - \frac{1}{N}\sum_{n'=1}^N r^{n'}$$

其中 $r^n$ 是第 $n$ 个 rollout 的 composite reward，$N$ 是 group size（论文设为 8）。

Loss 是 clipped surrogate（公式 2）：

$$\mathcal{L} = \mathbb{E}_n\left[\min\left(\rho^n A^n, \text{clip}(\rho^n, 1-\epsilon, 1+\epsilon)A^n\right)\right]$$

变量：
- $\rho^n = \pi_S(c^n|\chi) / \pi_{\theta_{old}}(c^n|\chi)$ — importance ratio，新旧 policy 给同一个 curation sequence $c^n$ 的概率比
- $A^n$ — group-relative advantage
- $\epsilon$ — clip 范围
- $\chi$ — curator 的输入 context

**重要细节**：他们 **丢掉了 KL 项**。GRPO 原版有 KL penalty 防止 policy 偏离 reference，但这里完全丢弃以鼓励 exploration。这是个有趣选择——curator 在 curation 空间非常 under-explored，需要 strong push 去发现新 skill 编辑模式。

advantage 在 $c^n$ 的所有 token 上均匀分配（token-level granularity 而非 action-level），简化实现。

---

## 6. 实验数据深度解读

### 6.1 主表（Table 1, ALFWorld）

| Executor | No Memory | ReasoningBank | MemP | SkillOS-base | SkillOS-gemini | **SkillOS** |
|---|---|---|---|---|---|---|
| Qwen3-8B | 47.9 | 55.7 | 49.7 | 53.1 | 50.7 | **61.2** |
| Qwen3-32B | 54.5 | 61.4 | 55.7 | 59.8 | 63.6 | **68.6** |
| Gemini-2.5-Pro | 66.4 | 71.4 | 74.3 | 70.7 | 79.3 | **80.2** |

几个 critical 观察：

**(1) RL-trained 8B curator > Gemini-2.5-Pro 直接做 curator**。这是论文最强的 claim。Gemini-2.5-Pro 比 Qwen3-8B 强几个数量级，但 raw reasoning 能力 ≠ curation 能力。这就像写作好的人不一定是好编辑——curation 需要的是"知道 executor 用什么、executor 不擅长什么"，这种 ground knowledge 只能从 RL 中获得。

**(2) Executor 越强，SkillOS 增益越大**：
- Qwen3-8B：+7.9（vs SkillOS-base）
- Qwen3-32B：+8.8
- Gemini-2.5-Pro：+9.5

这是反直觉的——通常我们会觉得大模型 baseline 已经很强，improvement 应该变小。但 SkillOS 反过来，强 executor 更能"榨取"skill 的价值。这暗示 skill curation 不是补 executor 的短板，而是放大 executor 的长板。

**(3) SkillOS-gemini 在小 executor 上反而更弱**（50.7 vs SkillOS 61.2 with Qwen3-8B executor）。这是 curator-executor mismatch：Gemini 生成的 skill 太抽象太复杂，Qwen3-8B executor 用不动。Karpathy 你应该会觉得这是个非常 neat 的现象——这就像给小学生大学教授的笔记，不是知识不够好，是不匹配。

### 6.2 推理任务（Table 2）

| Executor | No Memory | SkillOS-base | SkillOS-gemini | **SkillOS** |
|---|---|---|---|---|
| Qwen3-8B (AIME24/25/GPQA avg) | 69.6 | 68.9 | 67.4 | **73.8** |
| Qwen3-32B | 74.0 | 74.7 | 73.2 | **79.7** |
| Gemini-2.5-Pro | 81.8 | 84.6 | 85.4 | **88.6** |

增益没有 agentic task 那么大。论文解释是 reasoning task 的 reusable skill 是抽象的 decomposition heuristics，而 agentic task 的 skill 是 procedural regularities（动作顺序、exploration 策略）。这跟我们做 [CoT prompting](https://arxiv.org/abs/2201.11903) 的直觉一致——CoT 的核心可重用部分往往是高阶策略，比"找到 desklamp 下的 CD"这种 spatial-temporal procedure 更难蒸馏。

### 6.3 效率（Steps）

ALFWorld 上 SkillOS 减少 2.2–3.1 步。这非常重要——它说明 SkillOS 不是通过"试更多次"提升 SR，而是通过"更早知道捷径"提升 SR。论文 Figure 19 给了个 case study：memory-free agent 在 "look at the CD under the desklamp" 上乱翻容器耗尽步数，SkillOS 直接去 desklamp 周围找。这就是 procedural skill 的价值。

---

## 7. Ablation 分析（Table 3）

| 变体 | Avg. SR | Steps |
|---|---|---|
| SkillOS full | 61.2 | 18.9 |
| w/o $r^{\text{cnt}}$ | 58.6 | 20.1 |
| w/o $r^{\text{comp}}$ | 60.0 | 19.3 |
| w/o grouping | 57.3 | 20.6 |

**最 critical 的是 grouping**（掉 3.9 个点）。这证明 paper 的核心 insight 是对的：要让 curator 学到 long-horizon curation policy，必须有 grouped task stream 提供 delayed feedback 通路。这也回应了 §2 提到的 [Wang et al. 2025a](https://arxiv.org/abs/2512.17102) 和 [Ye et al. 2026](https://arxiv.org/abs/2602.10652) 的工作——它们做 short-horizon adaptation，signal 不够。

**Compression reward 影响小但 consistent**——它主要防止 trajectory 复制，是个 safety net。

---

## 8. 训练动力学：Curator 行为演化（Figure 4）

这是论文最美的图，类似 Anthropic 在研究 interpretability 时展示的 phase transition：

- **训练初期**：insert 占绝对主导（"我不会，先塞新东西"）
- **训练中期**：update 开始崛起，insert 下降（"我已经有 skill 了，去 refine"）
- **训练后期**：update 主导，delete 缓慢上升（"我有冗余 skill，清理一下"）

这跟人类学习曲线高度吻合——初学者拼命积累笔记，老手花时间整理笔记。这暗示 RL 让 curator 真的学到了一个**层次化的 curation policy**：先 populate，再 refine，最后 prune。

---

## 9. Skill 内部结构演化（Figure 5）

两个 emergent phenomena：

**(a) Markdown section 演化**：早期 curator 添加 "additional guidance" "tips" 这种 generic section，让 skill 变啰嗦但操作性不强。后期 curator 添加 "failure-handling logic" "conditional branches"——这是可执行的控制流，不是装饰性文字。

**(b) Meta-strategy skills 涌现**：早期 SkillRepo 全是 task-specific skill（"如何找 apple"），后期出现 meta-skill（verification、fallback planning、system search、strategy adjustment）。这非常像 [Voyager](https://arxiv.org/abs/2305.16291) 的 skill composition 涌现，但 Voyager 是手动 curriculum，SkillOS 是自发的 emergence。

**直觉**：RL 在压缩 reward 信号的过程中，把"具体经验"压缩成"抽象模式"，类似深度学习从 specific feature 学到 general feature。这是 representation learning 在 symbolic knowledge层面的对应。

---

## 10. Skill 使用归因（Figure 6）

四个 metric：
1. Skill usage rate：100% examples 都用了 skill
2. Successful skill usage rate：用 skill 的 examples 中成功率
3. Skill coverage：SkillRepo 中实际被用到的比例
4. Avg skills per example：每个 example 用了几个 skill

SkillOS 在所有指标上都更优——**coverage 高 + per-example skill 数量少** = "每个 skill 都用得上，每个 example 不需要堆 skill"。这正是 curation 的目标：一个紧凑、每个 skill 都有用的 repo。

---

## 11. 跨任务泛化（Figure 3）

在 Qwen3-8B 上训练的 curator，迁移到 Qwen3-32B 和 Gemini-2.5-Pro 仍然 effective。论文还做了 cross-domain 实验：reasoning 训练的 curator 在 agentic 任务上反而表现不错（reasoning skill 包含 decomposition、verification 这种 general 策略），但 agentic 训练的 curator 不太能迁回 reasoning（因为 agentic skill 太 environment-specific）。

这跟我之前思考 [chain-of-thought generalization](https://arxiv.org/abs/2201.11903) 时的直觉吻合：抽象 reasoning 策略 > 具体 environment knowledge。

---

## 12. 我的批评与联想

### 12.1 优点
1. **解耦设计**：把 curation 从 executor 里独立出来，让 RL 训练成本可控
2. **Grouped task stream**：这是对 delayed credit assignment 的工程化解决方案，比简单 PPO 在长 horizon 上更稳
3. **Composite reward**：dense intermediate signal 救了早期 RL 阶段
4. **跨 executor 泛化**：curator 学到的是"executor-agnostic" 的 curation pattern

### 12.2 我会质疑的地方

**(1) BM25 retrieval 是个明显的 bottleneck**。论文自己也承认了。SkillRepo 长大后，BM25 的语义召回会很弱。Karpathy 你之前在 [NeurIPS 2024 keynote](https://www.youtube.com/watch?v=l8pRSuUH7ic) 提过 retrieval 是当前 RAG 系统的痛点，这里完全印证——learned retriever 应该是下一步。

**(2) SkillRepo 不会爆炸吗？** 论文 ablation 显示 delete 操作始终很少。在真实 streaming 部署里，SkillRepo 可能无限膨胀，BM25 召回会越来越糟。Compression reward 只是 "1 - ratio"，没有 hard cap。

**(3) Frozen executor 的局限性**。论文说 frozen 是为了"隔离 curation 的贡献"。但这意味着如果 executor 本身不理解某种 skill 的表述方式，curator 没办法 retrain 它。Joint optimization 是 obvious 下一步，但论文说"训练成本太高"。

**(4) LLM-as-judge 的 reward hacking 风险**。$r^{\text{cnt}}$ 由 Qwen3-32B 评分，如果 Qwen3-32B 有系统性偏好（比如喜欢长 skill 或某种 style），curator 会学这个偏好而不是真正有用。论文没有做 reward hacking 分析。

**(5) Grouped task stream 在真实部署时不存在**。训练时我们手工分组相关 task，但真实 streaming 是 user-driven 的，task 之间可能完全无关。SkillOS 在 streaming test 时怎么避免把无关任务都"硬塞进同一个 group"？这是 deployment gap，论文没讨论。

**(6) Anthropic SKILL.md 的 multi-file 能力被简化了**。真实 Anthropic skill 是文件夹，包含 scripts 和 resources，SkillOS 只用单 Markdown 文件。这是为了简化 action space，但失去了 executable skill 的能力——只能存 declarative 知识，不能存可执行代码。

### 12.3 联想到的相关工作

1. **[Voyager](https://arxiv.org/abs/2305.16291)**：Minecraft skill library，GPT-4 自动 curate。SkillOS 是 Voyager 的 learned curation 版本
2. **[ExpeL](https://arxiv.org/abs/2308.10144)**：从 trajectory 蒸馏 insight，但用启发式 curation
3. **[MemGPT](https://arxiv.org/abs/2310.08560)**：OS-inspired memory，prompt-based 控制
4. **[DreamCoder](https://arxiv.org/abs/2106.11590)**：library learning 的经典，但用 neurosymbolic search 而非 RL
5. **[AlphaProof / AlphaGeometry](https://www.nature.com/articles/s41586-023-06747-w)**：skill 作为可组合的 tactic
6. **[SkillRL](https://arxiv.org/abs/2602.08234), [D2Skill](https://arxiv.org/abs/2603.28716), [ARISE](https://arxiv.org/abs/2603.16060)**：同期工作，教 agent 用 skill 但不学 curation
7. **[GigPO](https://arxiv.org/abs/2505.10978)**：Group-in-Group Policy Optimization，也是 grouped RL
8. **[verl](https://arxiv.org/abs/2409.19256)**：训练框架
9. **[ReasoningBank](https://openreview.net/forum?id=jL7fwchScm)**：作者自己的前作，是本文的直接 baseline
10. **[Buffer of Thoughts](https://arxiv.org/abs/2312.08908)**：另一种 thought-as-skill 抽象

### 12.4 Karpathy 视角的延伸思考

我会把这篇论文放在你之前讨论 [Software 2.0](https://karpathy.medium.com/software-2-0-a64552e279ac) 和 [Software 3.0](https://www.youtube.com/watch?v=LCEmiRjPEqQ) 的脉络里看。SkillOS 实际上是在做 **Software 3.0 的 garbage collection**——program（prompt+skill）作为一等公民，agent 学会自己重构自己的程序库。

这跟 OS 的设计哲学很像：内核 不动，但用户态的工具会随时间被 organize 进越来越合理的目录结构。SkillOS 把这个 organize 过程变成了 learned policy。

更深层：这其实是 **meta-learning through symbolic memory** 的一种 instance。传统 meta-learning 通过 weight 更新，SkillOS 通过外部 symbol manipulation。这条路在 LLM 时代可能比 MAML/Reptile 更实用——因为 weight-level meta-learning 需要 bilevel optimization，而 symbol-level meta-learning 只需要单层 RL。

未来工作（论文也提了）：hierarchical skill（top-level 调 sub-skill）、agentic search over memory（不用 BM25，用 multi-step query refinement）、multi-agent shared memory。这三个方向任何一个都够一篇 follow-up paper。

---

## 13. 一句话总结

SkillOS 把"agent 怎么管理自己的经验"这个看似无限复杂的 meta-problem，变成了一个 well-defined RL subproblem——把 executor 冻结，让 curator 在 skill 文件系统上做 curation 操作，用 grouped task stream + composite reward 把 delayed credit assignment 转化为可学习的 dense signal。8B curator 打 Gemini-2.5-Pro 直接 curation 的结果证明：**curation is a learned skill, not an emergent capability**。这个 finding 对整个 self-evolving agent 方向都非常重要。

---

## 参考 Web Links

- Paper: [SkillOS arxiv (假设为 2505.xxxxx)](https://arxiv.org/abs/2505.xxxxx)
- Anthropic Skills: https://github.com/anthropics/skills
- GRPO/DeepSeekMath: https://arxiv.org/abs/2402.03300
- ALFWorld: https://openreview.net/forum?id=0IOX0YcCdTn
- WebShop: https://arxiv.org/abs/2207.01206
- ReAct: https://arxiv.org/abs/2210.03629
- CoT: https://arxiv.org/abs/2201.11903
- Voyager: https://arxiv.org/abs/2305.16291
- ExpeL: https://arxiv.org/abs/2308.10144
- MemGPT: https://arxiv.org/abs/2310.08560
- DreamCoder: https://arxiv.org/abs/2106.11590
- verl framework: https://arxiv.org/abs/2409.19256
- DeepMath-103K: https://openreview.net/forum?id=kHB5Te5IWm
- Qwen3: https://arxiv.org/abs/2505.09388
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- GPQA: https://openreview.net/forum?id=Ti67584b98
- ReasoningBank: https://openreview.net/forum?id=jL7fwchScm
- MemP: https://arxiv.org/abs/2508.06433
- AIME: https://artofproblemsolving.com/wiki/index.php/American_Invitational_Mathematics_Examination
- Buffer of Thoughts: https://arxiv.org/abs/2312.08908
- AlphaGeometry: https://www.nature.com/articles/s41586-023-06747-w
- GigPO: https://arxiv.org/abs/2505.10978
- SkillRL: https://arxiv.org/abs/2602.08234
- D2Skill: https://arxiv.org/abs/2603.28716
- ARISE: https://arxiv.org/abs/2603.16060
- UMEM: https://arxiv.org/abs/2602.10652
- AgentKB: https://arxiv.org/abs/2507.06229
- MEM1: https://openreview.net/forum?id=XY8AaxDSLb
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a64552e279ac
- Karpathy Software 3.0 (YouTube): https://www.youtube.com/watch?v=LCEmiRjPEqQ
