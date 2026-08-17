---
source_pdf: Embodied Multi-Modal Agent trained by an LLM from a Parallel TextWorld.pdf
paper_sha256: c68097a722ae232fedd52fcb415df558dfe0d077d03ae1d741c79e7f14be0cf9
processed_at: '2026-08-04T03:31:31-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EMMA 人话版

## 一句话概括

**让一个看不见的"盲人教授"(LLM)在文字世界里教一个"聋哑学徒"(VLM)在图像世界里干活。**

---

## 1. 这篇paper想解决什么问题?

想象你要训练一个机器人完成家务,比如"把洗好的苹果放进冰箱"。

你有三条路:

**路线A: 直接上GPT-4V**
给它看一帧画面,问它下一步干啥。结果它会说"我看到了一个柜子和一个苹果"然后就懵了。因为它从没真正在 embodied 环境里 interact 过,只是看过一堆 image-text pair。这就像让一个只读过菜谱的人去掌勺。

**路线B: Behavior Cloning**
你请人花几个月演示几千次,记录每个动作,然后让 VLM 模仿。问题是演示数据覆盖不到所有 corner case, robot 一旦偏离 expert trajectory 就不会 recover。这就像新手司机只会跟着教练走过的路线开,换条路就慌。

**路线C: RL**
reward sparse,你跟 robot 说"任务完成给你1分",它可能随机探索一万次才碰对一次。这就像让猴子打莎士比亚。

EMMA 走的是第四条路。

---

## 2. EMMA的核心idea

这里有个很关键的 observation:

**同一个 ALFWorld 任务,同时存在两个版本:**
- Visual version: agent 看pixel, 输出text action
- Text version: agent 看 PDDL 描述, 输出text action

这两个版本是**完全对齐的** — 同一个房子, 同一个物品布局, 同一个任务目标, 只是 modality 不同。

而 LLM (比如 GPT-3.5) 在 text version 上能拿到 91% success rate, 因为它有 commonsense + reasoning + planning, 给它文字描述它就能一步步规划。

所以 EMMA 的 idea 就很自然了:

> **让 LLM 在 text world 里当"老师", 教 VLM 在 visual world 里当"学生"。**

老师看不见画面, 但学生可以把每帧画面翻译成文字描述告诉老师, 老师根据文字描述给出下一步动作, 学生就模仿这个动作。

这就像盲人教授通过翻译指导聋哑学徒画画 — 教授有审美有构图知识, 学徒有眼睛有手, 配合起来就能完成作品。

---

## 3. 为什么这个idea不是trivial?

你可能会想: 这不就是 image captioning + LLM planning 的 pipeline 吗?

EMMA 的深刻之处在于: 它是**interactive + online**的, 而不是 open-loop 的。

如果是 open-loop:
1. VLM 看画面 → 生成描述
2. LLM 根据描述 → 生成动作
3. VLM 执行动作

这会有两个问题:
- captioning 会累积误差 (描述错了 LLM 就规划错了)
- 完全没利用 VLM 自己的 visual perception 能力

EMMA 的做法是: **让 VLM 学生自己在 visual world 里探索, 走到哪算哪, 每一步把当前画面转成 text 喂给 LLM 老师, 老师给建议, 学生用这个建议来更新自己的 policy**。

这就是 DAgger (Dataset Aggregation) 的精髓: 在学生自己实际会遇到的 state distribution 上训练, 而不是在老师的 distribution 上训练。

类比: 教练不示范, 让新手自己开车, 新手开到每个路口就问教练"我现在该怎么转", 教练说"左转", 新手就记住"在这种路口场景下要左转"。下次遇到类似场景新手就会了, 不用再问。

---

## 4. 两个world怎么对齐?

ALFWorld 这个 benchmark 本身就是为 cross-modality 设计的:

- Ai2Thor simulator 渲染 3D 场景 → visual world
- 从 simulator metadata 提取 PDDL state → text world

PDDL state 包含:
- Observed Objects: 你看到 cabinet 2, apple 1, fridge 1...
- Observed Relations: apple 1 is in cabinet 2
- Inventory: 你手里拿着什么
- Locations: 你在哪

这两个 world 的 state 是同步的: visual world 里 cabinet 2 关着, text world 里也写"cabinet 2 is closed"。

这种对齐让 cross-modality imitation 变得可能 — 老师和学生看的是同一个 underlying world, 只是 representation 不同。

---

## 5. LLM老师怎么设计?

老师其实是个**双角色复合体**:

### Actor (演员)
- 角色: 根据当前 text state + 历史 action + task instruction, 输出下一步 action
- prompt 里带 few-shot examples (ReAct style 但去掉 think step)
- 重要的是, prompt 里还带 **long-term memory** — 之前 trial 失败的反思总结

### Critic (评论员)
- 角色: 一个 trial 结束后 (成功或失败), 看完整 trajectory, 分析哪里做错了, 写一段 reflective feedback
- 比如: "你卡在 stoveburner 1 上一直 examine 不前进, 下次遇到 loop 要换 action"
- 这段 feedback 存进 long-term memory, 下次 trial 时塞进 actor 的 prompt

这其实借鉴了 Reflexion 的 idea, 但 EMMA 用它来**持续改进老师**, 然后老师再持续教学生。Fig.4 显示随着 trial 数增加, LLM 老师自己的 success rate 也在涨 (从 ~60% 涨到 ~85%), 同时 EMMA 学生也跟着涨, 两者 gap 越来越小。

---

## 6. DAgger-DPO的核心算法

### DAgger 部分: 解决 distribution shift

经典 BC 的问题: student 在 expert 的 trajectory 上训练, 但 student rollout 时偏离 expert trajectory, 遇到没见过的 state 就崩。

DAgger 的做法: 让 student 自己 rollout, 收集它实际遇到的 state, 然后 expert 给这些 state relabel, 加进 training set。下一轮 student 在这个 aggregated dataset 上训练。

这样 student 训练的 distribution 就是它 deployment 时会遇到的 distribution, 解决 distribution shift。

### DPO 部分: 比Cross-Entropy更好的loss

传统 BC 用 cross-entropy loss: 让 student 的 action distribution 尽量接近 expert 的 action。

EMMA 用 DPO loss: 把 (expert action $x_a^*$, student action $x_a$) 当成 preference pair, 让 student 学会"在这个 state 下, expert action 比 student action 更好"。

公式:
$$\mathcal{L} = -\log \sigma\left(\beta \log \frac{\pi_\theta(x_a^*|s_v)}{\pi_{ref}(x_a^*|s_v)} - \beta \log \frac{\pi_\theta(x_a|s_v)}{\pi_{ref}(x_a|s_v)}\right)$$

人话翻译:
- $\pi_\theta$: 正在训练的 student
- $\pi_{ref}$: 初始 BC 模型 (锚点, 防止 student 漂太远)
- $x_a^*$: expert 给的好动作 (positive)
- $x_a$: student 自己原来给的动作
- $\beta=0.1$: 正则强度
- $\sigma$: sigmoid, 把整个东西压到 (0,1)

这个 loss 在做的事: 让 student 相对于 $\pi_{ref}$, 在 expert action 上提升概率, 在自己原来的烂 action 上降低概率。

为什么比 CE 好? Paper 给的实验证据 (Fig.6 right): CE 收敛快但 plateau 低, DPO 收敛慢但最终高。直觉上 CE 是 absolute supervision, 容易让 student 抓住 easy task 的 surface pattern 然后停止探索; DPO 是 relative supervision, 提供 contrastive gradient, 更细粒度。

这跟 RLHF 里 DPO 替代 PPO 的逻辑类似, 不过这里用在了 imitation learning 上。

---

## 7. EMMA的architecture

基于 InstructBLIP 模块化:

```
Pixel → ViT (frozen) → visual embeddings
                              ↓
                    Q-Former (frozen, 32 query tokens)
                              ↓
                    32 visual tokens
                              ↓
              Linear Projection (TRAINABLE, 唯一更新部分)
                              ↓
              aligned visual tokens (project到LLM embedding space)
                              ↓
        [task instruction tokens, visual tokens] 
                              ↓
                    LLM Decoder (Vicuna-7B, frozen)
                              ↓
                          action token
```

为什么只 finetune linear projection layer?
- 省计算 (几M params vs 几B params)
- 避免 catastrophic forgetting (ViT 和 LLM 的预训练知识保留)
- 这个 layer 本质是 modality alignment bridge, 正好是需要 task-specific adapt 的部分

一个有意思的 detail: Q-Former 的 text input 被关掉了 (Table 2 里 Q-Former text input = False)。InstructBLIP 原本 Q-Former 也接收 instruction, 但 EMMA 实验发现关掉更好。直觉: visual feature extraction 应该是 task-agnostic 的, task conditioning 全交给 LLM decoder 通过 cross-attention 处理。

---

## 8. 结果有多impressive?

### 主表 (Table 1)

134 个 OOD tasks, 6 种任务类型:

| Agent type | 例子 | Avg Success Rate |
|---|---|---|
| Vision models | ResNet-18 | 5-6% |
| LLM (text env) | Reflexion | 91% |
| VLM (visual env) | InstructBLIP | 22% |
| **EMMA** | ours | **82%** |

EMMA 比 SOTA VLM (InstructBLIP) 高 **60 percentage points**。而且接近 LLM agent 在 text env 上的水平 (91%)。

### Per-task 看几个亮点

- Pick2 (找两个相同物体): 所有其他 VLM 都是 0%, EMMA 67%
- Clean: InstructBLIP 26%, EMMA 94%
- Cool: InstructBLIP 6%, EMMA 83%

### Noise robustness (Fig.5)

给 visual obs 加 crop noise, 给 text obs 加 token replacement noise:
- EMMA 在 visual noise 下性能缓慢下降
- Reflexion 在 text noise 下性能急剧下降

这是 VLM 相对于 LLM agent 的核心 practical advantage: 真实世界视觉信号 inherently noisy, text world 的 clean semantic 在 deployment 时反而是 privilege。

### Free-form instruction generalization (Table 2)

用 Amazon Mechanical Turk 的人写自然语言 instruction (OOD verbs/objects):
- EMMA: 82% → 68% (轻微下降)
- InstructBLIP: 22% → 1% (基本归零)
- 所有其他 VLM: → 0-4%
- Reflexion (LLM): 91% → 78% (也降但不崩)

EMMA 是唯一能在 free-form instruction 上 maintain 的 VLM。这说明 cross-modality imitation 把 LLM 的 generalization 能力 transfer 过来了。

---

## 9. Ablation 三个核心发现 (Fig.6)

### Ablation 1: 去掉 retrospection
只用 plain LLM actor relabel, 不用 critic feedback。结果: 最终性能低 10-15%。证明 critic 的 reflective feedback 是 fundamental。

### Ablation 2: 去掉 BC initialization
不用 BC 模型当 $\pi_{ref}$, 用 pretrained VLM 直接初始化。结果: 性能略降但仍然 beat 所有其他 VLM。证明 BC init 主要是 stabilizing, 不是 critical。

### Ablation 3: DPO 换成 CE
用 token-level cross-entropy 替代 DPO。结果: 收敛快但 plateau 低 15-20%。证明 DPO 的 relative supervision 避免 premature convergence。

---

## 10. 训练流程细节

### Phase 1: BC initialization
用 rule-based planner (有全知视角的 oracle) 生成 15247 episodes / 178585 image-text pairs。在这个 dataset 上 finetune linear projection layer, 得到 $\pi_{ref}$。

注意这个 oracle planner 有 unfair advantage: 它能看到 metadata, 不需要 perception。它只用来生成 BC data, 不参与后续 imitation。

### Phase 2: DAgger-DPO imitation
12 个 trial, 每个 trial:
1. EMMA student 在 visual env rollout 一个 episode (≤30 steps)
2. 把 trajectory 转成 text world trajectory
3. LLM critic 分析, 生成 reflective feedback, 加进 long-term memory
4. 对 trajectory 每个 step, LLM actor 给 expert action
5. 把 (state, student action, expert action) 加进 dataset D
6. 在 D 上 finetune linear projection 5 epochs

LLM expert 用 text-davinci-003 (2023年的 API, 现在已经 deprecated, 这是个 limitation)。

---

## 11. 这篇paper的"哲学"意义

EMMA 在我看来提示了一个重要的 paradigm:

> **LLM 的 reasoning 能力可以跨 modality 蒸馏, 即使两个 modality 的 representation 完全不同。**

LLM 从没看过 pixel, 但通过 parallel world alignment, 它在 text world 里学到的 planning 能力能 leak 到 visual world 的 VLM 里。

这暗示一条 AGI 路径:
- Phase 1: 在 text modality 训练出强大的 reasoner (LLM)
- Phase 2: 通过 cross-modality alignment 把 reasoning 能力 transfer 到 perception/motor modality
- Phase 3: 各 modality 的 embodied agent 协同, 接入真实世界

而不需要每个 modality 从头训一个 giant model。这跟 RT-2, PaLM-E 等 end-to-end VLA 路线形成对比 — 后者需要海量 robot demonstration, 而 EMMA 只需要 LLM API + parallel env。

---

## 12. 几个值得玩味的 design choices

1. **Q-Former text input 关掉**: InstructBLIP 原本设计 task-conditioned visual extraction, EMMA 发现 task-agnostic 更好。可能因为 task conditioning 留给 LLM decoder 更高效。

2. **Long-term memory size = 3**: 不是越多越好, 而是 high-signal summary。避免 context length 溢出 + 噪声 feedback 干扰。

3. **Few-shot prompt 里去掉 think step**: ReAct 原本有 chain-of-thought, 但 imitation 阶段只保留 final action。因为 VLM 学不了 LLM 的中间 reasoning, 只学 action mapping 更直接。

4. **DPO β=0.1**: 比较保守的正则。如果太大会让 $\pi_\theta$ 紧贴 $\pi_{ref}$ 不探索, 太小会漂太远 diverge。

5. **Episode length cap = 30**: 防止 stuck agent 无限消耗 API。失败 trial 也用来 generate critic feedback, 不浪费。

6. **只训 linear projection layer**: 这个 layer 本质是 "modality bridge", 正好是需要 task-specific adapt 的部分。其他 frozen 部分提供 prior knowledge。

---

## 13. Limitations (paper没明说但隐含的)

1. **依赖 simulator metadata**: 真实世界没有 ground truth PDDL state, 需要 perception module 先 captioning, 引入新误差。
2. **依赖 LLM API**: text-davinci-003 已 deprecated, 用 GPT-4 / Claude 贵很多。开源 LLM 可能 quality 不够。
3. **Discrete action space**: ALFWorld 是 high-level text action ("go to cabinet 1"), 不是 low-level motor control。DPO 在 token level 上定义, 不直接 extend 到连续 action。
4. **Single env generalization**: 没测 cross-environment (kitchen train, bathroom test) 的 generalization。
5. **LLM expert ceiling**: Fig.4 显示 EMMA converge 到 LLM expert 水平, 不能超过。要超越需要 RL fine-tuning 或 iterative self-play。
6. **Long-horizon**: episode length 30, real household task 可能上百 steps, LLM critic 的 retrospective 在长 trajectory 上能力未知。

---

## 14. 给Karpathy的几个intuition

如果你要 build 类似系统, 我觉得几个 key takeaways:

1. **Parallel world alignment 是 cross-modality transfer 的基础**。要设计 env 时就考虑 modality alignment, 而不是事后 captioning。

2. **Interactive imitation >> Static BC**。DAgger 的 distribution match 是关键, 哪怕 expert 弱一点, 只要 student 在自己 distribution 上学就比 static BC 强。

3. **DPO 作为 imitation loss 值得尝试**。相比 CE, DPO 的 relative gradient 在 discrete action space 上能避免 premature convergence, 尤其在 multi-task setting。

4. **Retrospective expert 比静态 expert 强**。让 expert 自己也通过 critic feedback 改进, 整个 system 是 self-improving loop。

5. **Modularized VLM + 只训 bridge layer** 是 sample efficient 的关键。full finetune 会 catastrophic forget 且 expensive。

6. **Free-form instruction generalization** 是 real-world deployment 的关键 metric, EMMA 在这点上远超其他 VLM 是最 impressive 的结果。

---

参考链接:
- [EMMA GitHub](https://github.com/stevenyangyj/Emma-Alfworld)
- [ALFWorld](https://arxiv.org/abs/2010.03768)
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
- [DAgger original paper](https://arxiv.org/abs/1011.0686)
- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [TextWorld engine](https://arxiv.org/abs/1806.11532)
- [Ai2Thor simulator](https://arxiv.org/abs/1712.05474)
- [EUREKA (对比工作)](https://arxiv.org/abs/2310.12931)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [PaLM-E](https://arxiv.org/abs/2303.03378)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [AutoGen](https://arxiv.org/abs/2308.08155)
- [DEPS](https://arxiv.org/abs/2302.01560)
- [GAIL](https://arxiv.org/abs/1606.03476)
- [ALFRED benchmark](https://arxiv.org/abs/1912.01734)
- [Chain-of-Thought](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)

---

# EMMA: Embodied Multi-Modal Agent 详解

## 1. Paper 的核心 Intuition

这篇 paper 想解决一个非常本质的问题: **visual world 里没有 scalable expert**。在 ALFWorld 这种 embodied environment 里, 你想训练一个 VLM agent 完成 household tasks (比如 "put a clean apple in fridge"), 但 RL reward sparse, behavior cloning 需要大量人类 demonstration, 直接用 GPT-4V 这种 SOTA VLM 在 visual ALFWorld 上几乎完全失败 (Fig.1a)。

而 LLM agent (比如 Reflexion, ReAct) 在 **parallel text world** 里却能拿到 ~91% success rate, 因为 LLM 有强大的 prior knowledge 和 reasoning 能力, 只要给它 textual state description 就能规划。

EMMA 的核心 insight: **把 LLM 在 text world 里的能力"蒸馏"到 VLM 在 visual world 里的能力**, 通过 cross-modality interactive imitation learning。这本质上是在做 **asymmetric co-training**, 两个 agent 活在两个 modality 里但 task structure 完全对齐 (parallel worlds)。

参考: [ALFWorld paper](https://arxiv.org/abs/2010.03768), [Reflexion paper](https://arxiv.org/abs/2303.11366)

---

## 2. Parallel Worlds 的构造

ALFWorld 本身就是一个 cross-modality benchmark, 它同时提供:
- **Visual environment**: 基于 Ai2Thor simulator 渲染的 3D 场景, agent 通过 pixel observation $s_v^t$ 感知
- **Textual environment**: 基于 TextWorld engine, 通过 PDDL (Planning Domain Definition Language) 描述 state $s_l^t$

关键点: 两个 world 是 **state-aligned** 的。同一个 task, 同一个初始 configuration, visual state 和 textual state 是等价的不同 modality 表示。Fig.2 给了一个 "clean apple and put in fridge" 的例子, 两个 world 里的 objects, locations, relations 完全对应。

这个对齐性是整个 cross-modality imitation 的前提。LLM expert 在 text world 里看到的 "cabinet 2 is closed, in it you see a spraybottle 2" 和 VLM student 在 visual world 里看到 cabinet 2 的 pixel frame 是同一个 underlying world state。

技术上, textual state 是从 simulator metadata 提取的:
- Observed Objects
- Observed Relations  
- Inventory
- Locations

然后通过 PDDL formalize, 喂给 TextWorld engine。这种 "ground truth state" 对 LLM expert 是 fully observable 的, 但 VLM agent 在 visual world 里只能通过 vision encoder 从 pixel 推断, 会有 noise 和 ambiguity。

参考: [TextWorld](https://arxiv.org/abs/1806.11532), [Ai2Thor](https://arxiv.org/abs/1712.05474), [PDDL](https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language)

---

## 3. EMMA 的 Architecture

EMMA 基于 InstructBLIP 模块化构造, 四个组件:

1. **ViT (Vision Transformer)**: ViT-L/14 from CLIP, frozen, 把 pixel observation $s_v$ encode 成 visual embeddings (patch tokens)
2. **Q-Former (Querying Transformer)**: 基于 BERT-base, 32 个 learned query tokens, 通过 cross-attention 从 visual embeddings 里 extract 32 个 visual tokens
3. **Linear Projection Layer**: 把 32 个 visual tokens 从 Q-Former 的 hidden dim 投影到 LLM 的 text embedding dim (这是**唯一 finetune 的部分**)
4. **LLM Decoder**: Vicuna-7B-v1.1, frozen, 接收 [task instruction tokens, visual tokens] 的 concatenation, autoregressive 生成 action $x_a$

**为什么只 finetune linear projection layer?** 
- Computational efficiency: 只有几百万参数可训练
- Avoid catastrophic forgetting: ViT 和 LLM 的预训练知识被保留
- 这种 modularized 架构允许 plug-and-play 任何预训练 vision encoder 和 LLM

注意 Table 2 里有个细节: **Q-Former text input = False**, 即移除了 InstructBLIP 原本 Q-Former 的 instruction input, 这在所有实验里都 improve 了 performance。这意味着 visual feature extraction 应该是 task-agnostic 的, task conditioning 完全交给 LLM decoder 通过 cross-attention with visual tokens 完成。

公式化: 
$$x_a \sim \pi_\theta(\cdot | x_{task}, s_v^t)$$

其中 $\pi_\theta$ 是整个 EMMA policy, $\theta$ 是 linear projection layer 的参数 (其他 frozen)。

参考: [InstructBLIP](https://arxiv.org/abs/2305.06500), [BLIP-2](https://arxiv.org/abs/2301.12597), [ViT](https://arxiv.org/abs/2010.11929)

---

## 4. Retrospective LLM Expert 设计

EMMA 的 "老师" 是一个 LLM expert, 由两个 specialized models 组成:

### 4.1 LLM Actor $M_a$

基于 text-davinci-003 (OpenAI 早期的 instruct model), prompt 包含:
- Few-shot examples (ReAct style, 但 strip 掉 think step, 只保留 action for imitation)
- Long-term memory $\mathcal{P}$ (存储 critic 的 reflective feedback)
- Current target environment + task instruction
- 历史 action sequence

输出: expert action $x_a^*$ for current step

### 4.2 LLM Critic $M_c$

同一个 text-davinci-003, 但 prompt 不同:
- Few-shot examples of "previous trial → retrospection" pairs
- EMMA 的完整 trajectory $\tau_l^i = [x_{task}, s_l^0, x_a^0, ..., s_l^T, x_a^T]$ (转成 text world 格式)
- 输出: reflective feedback $\mathcal{P}_i$ 分析失败原因

例如 critic 会说: "You were stuck in a loop examining stoveburner 1. You should have heated mug 1 with stoveburner 1, then put it in coffeemachine 1. Try a different action if stuck in a loop again."

### 4.3 Long-term Memory $\mathcal{P}$

FIFO queue, size 通常 1-3, append critic 的 feedback, 用于未来 trial 里 prompt actor。这个机制让 LLM expert 随着 trial 数量增加 progressively improve (Fig.4 显示 LLM expert 自己的 success rate 也在涨)。

这种 actor-critic with retrospective memory 的设计明显借鉴了 Reflexion 的思想, 但创新点在于: LLM expert 不直接执行 task, 而是 **作为 teacher** 持续 teach VLM student。

参考: [Reflexion](https://arxiv.org/abs/2303.11366), [ReAct](https://arxiv.org/abs/2210.03629)

---

## 5. DAgger-DPO 算法 (核心贡献)

### 5.1 为什么不用 Behavior Cloning (BC)?

Naive BC 在 expert distribution 上训练 student, 但 student rollout 时遇到 OOD states 就崩。这是经典的 **distribution shift / cumulative error** 问题 (Ross et al., 2011)。Fig.1b 显示, InstructBLIP 在 170K expert demonstrations 上 BC finetune 后仍然 fail, 因为 expert demonstration 都是 optimal trajectory, student 一旦偏离就不知道怎么 recover。

### 5.2 为什么用 DAgger?

DAgger (Dataset Aggregation) 通过 **online rollout student**, 让 student 访问自己实际会遇到的 state distribution, 然后用 expert 给这些 state relabel, 解决 distribution shift。

Algorithm 1 的核心 loop:
```
For each trial i:
  1. EMMA π_θ 在 visual env E_v 里 rollout 得到 τ_v^i
  2. 把 τ_v^i 转成 τ_l^i 在 text env E_l 里
  3. LLM critic M_c 分析 τ_l^i, 产生 feedback P_i
  4. 更新 long-term memory P ← P ∪ P_i
  5. For each step t in τ_v^i:
     - LLM actor M_a 在 text env 里给 expert action x_a^*
     - 把 (x_task, s_v^t, x_a^t, x_a^*) 加入 dataset D
  6. 在 D 上 finetune π_θ 几个 epochs
```

### 5.3 为什么用 DPO Loss 而不是 Cross-Entropy?

BC/DAgger 传统用 cross-entropy loss (token level)。但这篇用 **Direct Preference Optimization (DPO)** loss, 把 (expert action $x_a^*$, student action $x_a$) 作为一个 preference pair:

$$\mathcal{L}_{imit}(\cdot) \triangleq \log \sigma\left(\beta \log \frac{\pi_\theta(x_a^* | s_v)}{\pi_{ref}(x_a^* | s_v)} - \beta \log \frac{\pi_\theta(x_a | s_v)}{\pi_{ref}(x_a | s_v)}\right)$$

公式变量解释:
- $\pi_\theta$: 当前正在训练的 EMMA policy
- $\pi_{ref}$: reference agent, 通过 BC 在 demonstration dataset 上初始化得到的 policy (frozen, 不更新)
- $x_a^*$: expert action (正样本, LLM actor 给的)
- $x_a$: student action (负样本, EMMA 自己在 rollout 时产生的)
- $s_v$: visual observation
- $\sigma(\cdot)$: logistic sigmoid function
- $\beta$: hyperparameter (Table 2 设为 0.1), 控制 $\pi_\theta$ 偏离 $\pi_{ref}$ 的程度。$\beta$ 大 → 强正则, $\pi_\theta$ 紧贴 $\pi_{ref}$; $\beta$ 小 → 弱正则, 允许大偏离

直觉上, 这个 loss 在最大化:
$$\log \frac{\pi_\theta(x_a^*|s_v)}{\pi_{ref}(x_a^*|s_v)} - \log \frac{\pi_\theta(x_a|s_v)}{\pi_{ref}(x_a|s_v)}$$

即: 让 $\pi_\theta$ 相对于 $\pi_{ref}$ 在 expert action 上**提升概率**, 在 student action 上**降低概率**。

### 5.4 为什么 DPO 优于 CE? (Fig.6 right)

实验显示 CE Loss 收敛更快但 plateau 在更低 success rate, DPO 收敛慢但上限高。Paper 给的解释: CE loss 容易让模型快速学到 easy task 的 expert action, 但 suppress exploration 导致 complex task 学不到。DPO 通过 preference pair 的 relative 信号, 提供更细粒度的 gradient, 避免 premature convergence。

这点很有意思, 因为 DPO 原本是用于 RLHF 的, 这里被 reinterpret 成 imitation loss。本质上 DPO 是 Bradley-Terry preference model 的 closed-form, 它隐式地学一个 reward function $r(s, a) = \beta \log \frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}$, 然后 maximize $r(s, x_a^*) - r(s, x_a)$。

参考: [DAgger](https://arxiv.org/abs/1011.0686), [DPO](https://arxiv.org/abs/2305.18290)

---

## 6. Objective Function 的理论分析

完整的优化目标:

$$\theta^* = \arg\min_{\theta \in \Theta} -\mathbb{E}_{\pi_\theta}[\mathcal{L}_{imit}(\pi_\theta, \pi_{ref}, s_v, x_a, x_a^*)]$$

注意这个 expectation 是在 $\pi_\theta$ induced state distribution 下, 即 $\mathbb{E}_{s_v \sim d^{\pi_\theta}}$, where $d^{\pi_\theta}$ 是 student policy rollout 产生的 state distribution。这就是 DAgger 的精髓: 用 student 自己的 distribution 而不是 expert 的 distribution。

**non-i.i.d. supervised learning problem**: 因为 state distribution 依赖于 $\pi_\theta$ 本身, 而 $\pi_\theta$ 在训练过程中变化, 这不是标准 i.i.d. supervised learning。DAgger 通过 online rollout + dataset aggregation 保证收敛到 optimal policy (在 no-regret online learning 假设下)。

---

## 7. 实验结果详细分析

### 7.1 Main Results (Table 1)

134 个 OOD tasks, 6 个 task type: Pick, Clean, Heat, Cool, Look, Pick2

**EMMA 平均 success rate: 0.82**, 平均 19.5 steps

对比:
- **Vision models** (ResNet-18, MCNN-FPN): 0.05-0.06, 完全失败, 因为没有 semantic prior
- **LLM agents in text env**: Reflexion 0.91, AutoGen 0.77, DEPS 0.76, ReAct 0.54
- **VLM agents in visual env**: InstructBLIP 0.22, MiniGPT-4 0.16, BLIP-2 0.04, LLaMA-Adapter 0.13
- **GPT-4V (zero-shot)**: Fig.1a 显示基本失败

EMMA 比 SOTA VLM agent (InstructBLIP) 高 **60%** success rate, 比 LLM agent Reflexion 在 visual env 里也略高 (Reflexion 在 text env 才 0.91, 不能 deploy 到 visual env)。

### 7.2 Per-task Performance Breakdown

- Pick & Place: EMMA 0.71 (vs InstructBLIP 0.50)
- Clean & Place: EMMA 0.94 (vs InstructBLIP 0.26)
- Heat & Place: EMMA 0.85 (vs InstructBLIP 0.23)
- Cool & Place: EMMA 0.83 (vs InstructBLIP 0.06)
- Look: EMMA 0.88 (vs InstructBLIP 0.17)
- Pick2: EMMA 0.67 (vs InstructBLIP 0.00)

注意 Pick2 (找两个相同物体) 是最难的任务, 几乎所有 VLM agent 都 0%, EMMA 0.67 很 impressive。

### 7.3 Robustness to Noise (Fig.5)

把 visual observation 随机 crop + resize, 或者 textual observation 随机替换 token, 加 noise rate 0-50%。

EMMA (VLM) 在 visual noise 下性能 degradation 远小于 Reflexion (LLM) 在 text noise 下的 degradation。Paper 解释: ViT vision encoder 天然 robust to spatial noise (pretrained on augmentations), 而 LLM 直接处理 token, noise 直接破坏 semantic。

这是 EMMA 相对于 LLM agent 的关键 practical advantage: 真实世界视觉信号 inherently noisy, LLM 在 text world 里的优势 (clean semantic) 在 deployment 时反而成为 weakness。

### 7.4 Generalization to Free-form Instructions (Table 2, Fig.7)

134 个 unseen tasks, 用 Amazon Mechanical Turk human annotators 写的 free-form instructions 替换 template instructions。OOD verbs/objects 大量出现。

- EMMA: 0.82 → 0.68 (轻微下降)
- InstructBLIP: 0.22 → 0.01 (基本归零)
- MiniGPT-4, BLIP-2, LLaMA-Adapter: 全部降到 0.00-0.04
- Reflexion (LLM): 0.91 → 0.78 (下降也不多)

EMMA 是**唯一**能在 free-form instruction 上 maintain reasonable performance 的 VLM agent。Paper 把这归功于 cross-modality imitation 传递了 LLM 的 generalization 能力。

参考: [ALFRED benchmark](https://arxiv.org/abs/1912.01734)

---

## 8. Ablation Studies (Fig.6)

### 8.1 Retrospection 的作用 (left plot)
- EMMA w/ retrospection vs w/o retrospection
- w/o retrospection: 只用 plain LLM actor relabel, 不用 critic feedback
- 差距随 trial 数增加而扩大, 最终 EMMA 高 10-15%
- 证明 critic 的 reflective feedback 是 fundamental component

### 8.2 BC Initialization 的作用 (middle plot)
- w/o BC init: 用 pretrained VLM 直接初始化 (而不是用 BC-finetuned $\pi_{ref}$)
- 性能略降但仍然 outperform 所有其他 VLM agent
- BC init 主要起 stabilizing 作用, 不是 critical

### 8.3 DPO vs CE Loss (right plot)
- w/ CE Loss: 用 token-level cross-entropy 替代 DPO
- CE 初期收敛快但 plateau 低
- DPO 初期慢但 final performance 高
- 证实 DPO 避免 premature convergence to easy tasks

---

## 9. Training Details (Table 2)

### BC 阶段:
- 6 epochs, lr 1e-5, batch 128, AdamW (β1=0.9, β2=0.999), weight decay 0.05
- 15247 episodes / 178585 image-text pairs
- Inference beam size 5

### Imitation Learning 阶段:
- 12 trials, episode length 30
- Long-term memory size 3
- lr 5e-6, warmup 300 steps, batch 16
- 5 epochs per trial
- DPO β = 0.1
- LLM expert: text-davinci-003 (now deprecated, 这个 paper 用的是 2023 年版本的 API)

**关键细节**: 每轮 trial 后 D 不 reset, dataset 持续 aggregation, 这就是 DAgger 的 "aggregation" 部分。

---

## 10. Dataset Construction Pipeline (Sec.9, Appendix)

原始 ALFWorld task instructions 太少 (大约 8K), 不足以 finetune 大 VLM。Paper 设计了 automated pipeline:

1. 从 ALFWorld 提取 environment descriptions (objects, attributes)
2. Prompt text-davinci-003 generate new task instructions (Table 3 给了 kitchen example prompt)
3. 用 rule-based planner (full observability + 全知 world dynamics) 生成 expert demonstrations
4. 总计 15247 episodes, 178585 image-text pairs

这个 rule-based planner 有 "unfair advantage": 它访问 metadata 而不是 perception, 是 oracle。这个 oracle 只用来生成 $\pi_{ref}$ 的 BC 数据, 不参与 imitation learning 阶段。

---

## 11. 与相关工作的对比

### 11.1 vs EUREKA (Ma et al., 2023)
EUREKA 也用 simulator source info 作为 LLM context, 但 LLM 是 coding LLM, 输出 reward function, 然后用 RL optimize policy。EMMA 直接用 LLM 输出 action, 用 imitation 而不是 RL。RL pipeline 更复杂昂贵, EMMA 更简单 sample efficient。

### 11.2 vs RT-2, PaLM-E 等 VLA models
这些是 end-to-end VLA (vision-language-action), 通常用大量 robot demonstration finetune。EMMA 用 LLM expert 代替 human demonstration, 在没有大规模 demonstration 时更 scalable。

### 11.3 vs Voyager, DEPS 等 LLM agent
这些 LLM agent 只在 text env 里工作, 不能直接处理 visual input。EMMA 通过 cross-modality imitation 把 LLM 能力 transfer 到 VLM。

### 11.4 vs GAIL (Generative Adversarial Imitation Learning)
GAIL 学 reward function 然后用 RL, EMMA 用 DPO closed-form preference optimization, 避免显式 reward function 学习。DPO 实际上是 GAIL 在 preference feedback 下的特殊 case, 用 closed form 替代 adversarial training, 更稳定。

参考: [EUREKA](https://arxiv.org/abs/2310.12931), [RT-2](https://arxiv.org/abs/2307.15818), [PaLM-E](https://arxiv.org/abs/2303.03378), [Voyager](https://arxiv.org/abs/2305.16291), [GAIL](https://arxiv.org/abs/1606.03476)

---

## 12. Limitations 和 Future Directions

Paper 没明说但隐含的 limitations:

1. **依赖 simulator metadata**: textual state 从 simulator 提取, 真实世界没有 ground truth PDDL state。Deploy 到 real robot 需要 perception module 先生成 text description (e.g., VLM captioning), 但这引入新误差。

2. **依赖 LLM API expert**: text-davinci-003 现在 deprecated, 用 GPT-4 / Claude 可能更贵。LLM expert 的 quality 直接决定 EMMA 的上限 (Fig.4 显示 EMMA converge 到 LLM expert 的水平)。

3. **Discrete action space**: ALFWorld 是 high-level textual action (e.g., "go to cabinet 1"), 不是 low-level motor control。Real robot 需要连续动作空间, DPO 在 token level 上定义, 不直接 extend。

4. **Single environment generalization**: 没测 cross-environment (e.g., kitchen train, bathroom test) 的 generalization。

5. **Long-horizon tasks**: episode length 30, 但 real household task 可能上百 steps。LLM expert critic 在长 trajectory 上 retrospective 能力未知。

Future 可能方向:
- 用开源 LLM (Llama-3-70B) 替代 API LLM, 降低成本
- 把 cross-modality imitation 扩展到 audio, tactile 等 modality
- 用 self-play / iterative distillation 让 EMMA 自己变成 expert teaching newer version
- 结合 RL fine-tuning 在 DAgger-DPO 后 push performance 超过 LLM expert ceiling

---

## 13. 对 AGI 路径的启示

这篇 paper 在我看来提出了一个**很关键的 paradigm**: 

> Foundation model 在 text modality 里学到的 reasoning / planning 能力, 可以通过 cross-modality imitation transfer 到其他 modality, **即使两个 modality 的 representation 完全不同**。

这意味着 LLM 可能不需要直接看 pixel 也能 teach vision agent。LLM 是一个 abstract reasoner, 通过 parallel world 的 alignment, 它的 "intelligence" 可以 leak 到 perception-heavy 的 embodied agent 上。

这是否暗示 AGI 的 path: 先在 text world 里 build 出强大的 reasoner (LLM), 然后通过 cross-modality alignment transfer 到 visual/motor/audio/... world, 而不是从头在 each modality 里 train 一个 giant model?

EMMA 只在 ALFWorld 这种 toy environment 里验证, 但这个 paradigm scaling up 到 real robot, real autonomous driving, real embodied AI, 是一个非常有想象空间的 direction。

---

## 14. 实现细节的几个小 trick

1. **移除 Q-Former 的 text input**: InstructBLIP 原本 Q-Former 接收 instruction 作为 text input, 但 EMMA 设为 False, performance 提升。直觉: visual feature extraction 应该 task-agnostic, 让 LLM decoder 通过 cross-attention 处理 task conditioning 更高效。

2. **Long-term memory bounded to 3**: 避免 context length 溢出, 同时 critic feedback 是 high-signal summary 而不是 raw trajectory。

3. **Few-shot examples stripped of "think" steps**: ReAct 原本的 chain-of-thought 在 imitation 阶段被移除, 只保留 final action。这避免 EMMA 学习 LLM 的中间 reasoning (它没有这个 capability), 只学 action mapping。

4. **DPO β=0.1**: 比较保守的正则, 允许 $\pi_\theta$ 偏离 $\pi_{ref}$ 但不太远。Table 2 显示这个值经过 tuning。

5. **Episode length 30 cap**: 防止 stuck agent 无限消耗 LLM API。Failed trial 也用来 generate retrospective feedback。

6. **5 epochs per trial**: 不是训到 convergence, 每个 trial 后只训几个 epoch, 然后 rollout 新 trial。这是 DAgger 的 online characteristic。

---

## 15. 总结

EMMA 是一个 elegant 的 case study, 展示了 **cross-modality knowledge distillation from LLM to VLM** 的可行性。核心创新:

1. **Parallel world alignment**: 通过 PDDL 把 visual state 和 text state 对齐
2. **Retrospective LLM expert**: actor + critic with long-term memory, 让 LLM expert 自我改进
3. **DAgger-DPO**: 把 DAgger 的 interactive IL 和 DPO 的 preference optimization 结合, 解决 distribution shift + 提供 relative gradient signal
4. **Modularized VLM**: 只 finetune linear projection, 高效且避免 catastrophic forgetting

实验结果 impressive: 在 visual ALFWorld 上比 SOTA VLM 高 60-70%, 接近 LLM expert 在 text world 上的水平, 且对 noise 更 robust, 对 free-form instruction 有 generalization。

这个 paradigm 如果能 scale 到 real robot, 可能是 embodied AGI 的一条可行路径。

参考链接:
- [EMMA GitHub repo](https://github.com/stevenyangyj/Emma-Alfworld)
- [ALFWorld](https://arxiv.org/abs/2010.03768)
- [InstructBLIP](https://arxiv.org/abs/2305.06500)
- [DAgger](https://arxiv.org/abs/1011.0686)
- [DPO](https://arxiv.org/abs/2305.18290)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [EUREKA](https://arxiv.org/abs/2310.12931)
- [TextWorld](https://arxiv.org/abs/1806.11532)
- [Ai2Thor](https://arxiv.org/abs/1712.05474)
- [GAIL](https://arxiv.org/abs/1606.03476)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [PaLM-E](https://arxiv.org/abs/2303.03378)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [AutoGen](https://arxiv.org/abs/2308.08155)
- [DEPS](https://arxiv.org/abs/2302.01560)
