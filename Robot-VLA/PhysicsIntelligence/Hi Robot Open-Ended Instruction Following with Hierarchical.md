---
source_pdf: Hi Robot Open-Ended Instruction Following with Hierarchical.pdf
paper_sha256: a6400a0353705ae3974dec493bb21867885159133586d53da98bacf055fa4333
processed_at: '2026-08-04T23:42:35-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Hi Robot 人话版

Andrej, 我用大白话再过一遍, 像咱们喝咖啡聊天那样讲。

---

## 一句话概括

这个 system 让 robot 能听懂人话里的 "弯弯绕", 比如 "给我做个素食三明治, 别放西红柿, 顺便再给我朋友用火腿做一个", 还能在你中途插嘴说 "那个不是垃圾" 的时候听懂并改动作。

---

## 痛点在哪

现在的 VLA model, 比如你熟悉的 RT-2, OpenVLA, π₀, 它们能干的事是 "pick up the red cup onto the plate" 这种一句话一个动作的指令。你给它一句, 它干一个。

但真实世界里人不会这么说话。人说的是:

> "Hey robot, 帮我收拾下桌子, 只收垃圾别收盘子"

这句话里藏着好几个 sub-goal: 先得看哪些是垃圾哪些是盘子, 然后只对垃圾做 pick-and-place, 同时要抑制住 "把所有东西都收走" 的默认冲动。

更糟的是执行到一半, 人还会插嘴:

> "哎那个不是垃圾, 别动"

这时候 robot 得: 理解 "那个" 指的是当前手里正要抓的东西, 停下手, 放回去, 继续别的。

flat VLA 根本搞不定这个, 因为它没有 "思考" 的环节, 就是 image + text → action 的直射。

---

## 核心 idea: 两个脑子

Hi Robot 的方案特别直觉: 给 robot 装两个 "脑子", 一快一慢, 跟 Kahneman 那本 *Thinking Fast and Slow* 里讲的一样。

**System 2 (慢脑, high-level)**: 一个 VLM, 拿着 camera 图像 + 人的复杂指令, 想几秒, 吐出一句简单的 atomic command, 比如 "pick up the lettuce"。

**System 1 (快脑, low-level)**: 一个 VLA (就是 π₀), 拿着图像 + 那句 atomic command, 直接输出 robot 的连续动作, 50Hz 跑。

两者用 natural language 通信。慢脑每隔 1 秒或者收到人说话的时候重新想一次, 快脑一直在跑。

这个设计特别像你带一个实习生: 你告诉他 "做个素食三明治", 他脑子转一下, 决定先去拿面包, 于是给自己下达 "grab bread" 的小指令, 手就动了。做到一半你说 "别放 pickle", 他重新想, 改下一个小指令。

---

## 为什么不能一个 model 全干

你可能会问: 直接拿一个大 VLA, 把复杂 prompt 也喂进去, 让它端到端输出 action, 不就行了?

论文做了这个 ablation, 叫 **Flat VLA**, 就是用同一个 π₀, 把复杂 prompt 直接喂进去。结果很惨:

- 它会默认 "把桌上所有东西都收走", 无视 "只收垃圾" 的约束
- 中途插话完全没反应, 因为它只在 episode 开头读一次 prompt
- 遇到 "bus only yellowish things" 这种 partial instruction, 它分不清 "yellowish" 指什么

原因很直觉: 一个 model 既要负责 high-level 的语义 reasoning, 又要负责 low-level 的毫米级动作控制, 两头都做不好。分开之后, 慢脑专心想语义, 快脑专心抓东西, 各司其职。

---

## 两个层的形式化

慢脑的分布:

$$p^{hi}(\hat{\ell}_t \mid \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \ell_t)$$

- $p^{hi}$: high-level policy
- $\hat{\ell}_t$: 它吐出的 atomic command (比如 "pick up the lettuce")
- $\mathbf{I}_t^1, \ldots, \mathbf{I}_t^n$: $n$ 个 camera 的图像, $t$ 是物理时间步
- $\ell_t$: 人说的复杂 prompt

快脑的分布:

$$p^{lo}(\mathbf{A}_t \mid \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \hat{\ell}_t, \mathbf{q}_t)$$

- $p^{lo}$: low-level policy (π₀)
- $\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1}]$: 一个 action chunk, $H$ 是 chunk 长度
- $\mathbf{a}_t$: 单个 action vector, 比如 UR5e 是 7 维 (6 关节 + 1 gripper)
- $\mathbf{q}_t$: robot 当前 configuration (joint + gripper positions)

注意 $\hat{\ell}_t$ 替代了原始 $\ell_t$, 这就是 "慢脑翻译给人快脑听" 的数学表达。

---

## Action Chunking 为什么重要

快脑输出的不是单个 action, 是一整 chunk $\mathbf{A}_t$。这个思路来自 ACT (Zhao et al. 2023, https://arxiv.org/abs/2304.13705)。

直觉: 如果你一步一步 greedy 预测, 每步的小误差会 compounding, 走着走着就偏了。一次性预测未来 $H$ 步, 等于让 policy "看到" 整个 trajectory 的 temporal structure, 误差更 correlated, 累积更慢。

而且 chunking 让快脑以低频预测、高频执行, 所以慢脑可以慢悠悠地想, 不耽误快脑 50Hz 跑。

---

## 最妙的部分: Synthetic Data Generation

这是我觉得整个 paper 最聪明的点。

问题: 慢脑要学 "复杂 prompt → atomic command" 的 mapping, 但你的数据只有 teleoperated demos, 每个 demo 只标了 atomic skill (比如 "pick up lettuce")。你没有对应的复杂 prompt。

人工标? 太贵, 而且人想不出那么多花样。

Hi Robot 的招: **反过来想**。拿一个 large VLM (比如 GPT-4o 级别的), 给它看 (图像 + 当前 skill label), 问它: "如果有个用户看到这个场景, 说了什么话, 才会让 robot 执行这个 skill?"

形式化:

$$p^{gen}(\ell_t, u_t \mid \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \hat{\ell}_0, \ldots, \hat{\ell}_{t-1}, \hat{\ell}_t, \mathcal{P})$$

- $\ell_t$: 生成的 user prompt
- $u_t$: 生成的 robot 回话
- $\hat{\ell}_0, \ldots, \hat{\ell}_{t-1}$: 之前已经做过的 skills (给 VLM 上下文, 保证 multi-step coherence)
- $\hat{\ell}_t$: 当前 skill
- $\mathcal{P}$: prompt template, 包含 task description + scenario 分类

举例: 看到 robot 在抓 lettuce, VLM 可能生成:
- User: "Can you add some lettuce for me?"
- Robot: "Sure, adding lettuce now."

或者更复杂的:
- User: "I want a vegetarian sandwich, I'm lactose intolerant"
- Robot: "Got it, I won't put cheese on it."
- (对应 skill: pick up lettuce, 跳过 cheese)

这就把每条 demo "反向翻译" 成了带复杂 prompt 的 interaction episode。规模一上来, 你就有了几十万条 (复杂 prompt, 图像, atomic command) 的训练数据。

这个思路本质上是 **self-play 的一种变体**: robot 跟一个 imagined user 在对话。跟 Anthropic 的 Constitutional AI 用 AI feedback 训练 AI (https://arxiv.org/abs/2212.08073) 有异曲同工之妙。也像 Inverse RL 的 philosophy: 从 expert demos 反推 intent。

---

## Scenario 分类保证多样性

为了不让生成的 prompt 千篇一律, 论文在 prompt template 里加了 scenario 分类:

- **Negative task**: 用户说 "不要做什么" (e.g., "don't add tomatoes")
- **Situated correction**: 基于当前状态纠正 (e.g., "that's not trash")
- **Specific constraint**: 特殊约束 (e.g., "I'm lactose intolerant")

还有 response 分类: simple confirmation, clarification, error handling。

这确保生成的 interaction 覆盖 compositional language space, 而不是只会说 "pick up X"。

---

## 两个脑子的 architecture

两个 policy 用同一个 base: **PaliGemma-3B** (https://arxiv.org/abs/2407.07726), 3B 参数的开源 VLM。

- **慢脑**: PaliGemma 直接 fine-tune, 输出 language tokens, 用 cross-entropy loss
- **快脑**: PaliGemma + flow matching action expert, 这就是 π₀ (https://arxiv.org/abs/2410.24164), 输出连续 actions

Flow matching 跟 diffusion 的区别: diffusion 学的是预测 noise $\epsilon_\theta(\mathbf{x}_t, t)$, flow matching 学的是一个 vector field $v_t(\mathbf{x}; t)$, 把 noise distribution "流" 到 action distribution。更 general, training 更稳, sampling 更快。这也是 π₀ 能 50Hz 的关键之一。

Flow matching 原始 paper: https://arxiv.org/abs/2210.02747

---

## 训练细节

- AdamW, $\beta_1=0.9$, $\beta_2=0.95$, no weight decay
- Gradient clip max norm 1
- EMA decay 0.999
- LR warmup 1000 steps, 然后 constant $1 \times 10^{-5}$
- Batch size 512
- 慢脑训练: ~2 小时 on 8×H100

2 小时训完一个 high-level policy, 这效率很高, 主要因为 PaliGemma 才 3B, 而且 synthetic data 已经准备好, 不需要复杂的 RL loop。

---

## 实验长啥样

三个 task domain, 三个 robot platform:

| Task | Robot | DoF | 物理挑战 |
|------|-------|-----|---------|
| Table Bussing | UR5e | 7 | 抓 plate 要从 edge, singulate objects, 用 plate 倒 trash |
| Sandwich Making | Bimanual ARX | 14 | deformable ingredients, 精准放置 |
| Grocery Shopping | Mobile ARX | 14+2 base | bimanual + mobile, 多物体 |

每个 task × method 跑 20 trials, 两个 metric:

- **Instruction Accuracy (IA)**: 慢脑输出的 command 是否 align user intent + 当前 observation
- **Task Progress (TP)**: 长程任务完成比例

---

## 主结果 (Figure 5)

| Method | Table Bussing | Sandwich | Grocery |
|--------|--------------|----------|---------|
| Flat VLA | 低 | 低 | 低 |
| GPT-4o high-level | 中 (但常胡说) | 中 | 中 |
| Hi Robot | 高 | 高 | 高 |
| Expert Human | 最高 | 最高 | 最高 |

Hi Robot 比 GPT-4o 高 40%+ IA, 接近 expert human。

**GPT-4o 为什么拉胯**:
- 会 misidentify objects (把东西都叫 "plate" 或 "spoon")
- 会输出 "pick up bermuda triangle" 这种 nonsense
- gripper 还抓着东西就 issue "pick up new object"
- 没有 fine-tune 到 robot 的 affordance 上

**Flat VLA 为什么拉胯**:
- 无法 react to mid-task feedback
- 默认行为是 "pick up everything"

---

## 两个关键 Ablation

**Ablation 1: 去掉 synthetic data** (Figure 7)

只用 human-labeled data 训慢脑, 结果:
- 忽略 "this is not trash" 类 corrections
- 会加入 forbidden items 如 pickles
- Synthetic data 提供 compositional language coverage, 没有 it 就覆盖不到这些花样

**Ablation 2: 去掉 hierarchy** (Figure 8)

用 flat VLA + synthetic data (把 synthetic data 也喂给 low-level), 结果:
- 还是会 revert 到 "clearing all items"
- 无法 handle partial instructions ("bus only the yellowish things")
- 因为没有 high-level step 重新 check prompt 的机制

这两个 ablation 说明: **synthetic data 和 hierarchy 都必要, 缺一不可**。

---

## Latency 细节

消费级 RTX 4090 上:

**Low-level (per step)**:
| 组件 | ms |
|------|----|
| Image encoding | 14 |
| Observation processing | 32 |
| Action prediction (×10 denoise) | 27 |
| Total on-board | 73 |

约 14 Hz 原始, action chunking 后可 50Hz 执行。

**High-level (single decode step)**:
- RTX 4090: 47ms prefill + 13.2ms decode = 60ms
- H100: 17.3ms prefill + 5.7ms decode = 23ms

慢脑每 1 秒触发一次, 60ms 完全有余量。用户插话也能在 100ms 内响应, 感觉是实时的。

ASR 用 Whisper large-v2 本地跑, TTS 用 Cartesia API。

Whisper: https://arxiv.org/abs/2212.04356

---

## 系统怎么跑的

整个 pipeline:

1. 用户说话 → Whisper ASR → text prompt $\ell_t$
2. 慢脑拿 $\ell_t$ + camera images → 输出 $\hat{\ell}_t$ (atomic command + 可选 verbal response $u_t$)
3. 如果有 $u_t$, TTS 播给用户听, 从 $\hat{\ell}_t$ 里删掉
4. 快脑拿 $\hat{\ell}_t$ + images + $\mathbf{q}_t$ → action chunk $\mathbf{A}_t$
5. 执行 $\mathbf{A}_t$, 同时如果 1 秒到了或有新 user input, 回到 step 2

---

## 我的几个直觉

### 1. Language 当 interface 是天才设计

传统 hierarchical RL 用 latent vector 当高层给低层的信号, 不可解释, 不可组合, 不可注入人类 input。用 natural language 当接口, 你可以 inspect 中间 command 来 debug, 可以 compose ("pick up" + "the cup" 重组), 人可以随时 inject correction。而且 VLM 的 web-scale pretraining 直接迁移过来, 不用从零学 semantics。

### 2. Synthetic data 这个 idea 可以推广

本质上是 "inverse policy inference": 给定 (state, action), 推断 plausible intent。这跟 Inverse RL 的 philosophy 一致, 但用 VLM 的 world knowledge 来 generate 而不是 learn reward function。

你可以想象把这个 idea 用到别的地方: 给一段 code, 让 VLM 想象什么 user request 会触发这段 code; 给一个 navigation trajectory, 想象什么 navigation command 导致它。任何 "有 expert demo 但缺 intent label" 的场景都能用。

### 3. Small fine-tuned VLM > Large frozen VLM

Hi Robot 用 3B PaliGemma fine-tune, 吊打 GPT-4o (推测 1T+ params)。这跟 LLM 领域的观察一致: task-specific fine-tuning 比 raw scale 重要, 尤其是需要 grounding 在特定 domain data 的时候。GPT-4o 没有 robot 数据, 它不知道 robot 能干什么, 所以会乱发指令。

### 4. System 1 / System 2 的统一是 future

论文 future work 提到一个有意思的方向: 现在慢脑和快脑是两个 fine-tuned copy, 未来可以合成一个 model, inference time 切换 System 1 / System 2 模式。这跟 OpenAI o1 的 "runtime reasoning" 思路呼应。想象一个 VLA, 简单指令直接出 action (System 1), 复杂指令先 "想" 几步再出 action (System 2), 全在一个 model 里。

### 5. Long-context memory 是下一个 bottleneck

当前慢脑每次 inference 独立, 不维护对话历史。用户说 "remember, I'm allergic to peanuts" 10 步之后可能被忘记。这是下一个要攻克的点。LLM 领域的 long-context 技术 (RoPE scaling, ring attention 等) 直接可以迁移过来。

### 6. Low-level 的 proximal object bias 很经典

论文提到一个 failure mode: low-level 离 cheese 近就抓, 无视 "lactose intolerant" 的约束。这是 imitation learning 的经典 bias: demo 里 robot 经常抓近的东西, policy 就学到 "近 = 抓"。要 fix 这个, 需要 high-level 和 low-level 更紧的 coupling, 比如 high-level 知道 low-level 的 success rate, 这也是 future work。

### 7. 跟 SayCan 的对比

SayCan (https://arxiv.org/abs/2204.01691) 用 LLM 选 predefined skill, Hi Robot 用 fine-tuned VLM 选 learned skill。区别在于: predefined skill dexterity 有限, learned VLA 可以做 dexterous manipulation; 而且 Hi Robot 的 VLM fine-tune 过, 比 SayCan 的 frozen LLM 更 grounded。

---

## Limitation

论文自己承认的:

- 慢脑没 memory, 长 context reasoning 弱
- 快脑有 proximal object bias, 会临时忽略指令
- Error accumulation 和 OOD recovery 差 (东西掉了不会捡)
- 高低层解耦训练, 互相不知道对方能力

Mitigation 方向:
- 更强的 instruction-following backbone
- Long-context model
- Adversarial data generation for edge cases
- Failure recovery 的 data collection

---

## 我的最终 take

这篇 paper 的核心贡献在于: 它把 "robot 听人话" 这件事从 "单句 atomic command" 推进到了 "open-ended conversation with situated feedback"。方法上, hierarchy + synthetic data generation 这两个组合拳, 既解决了 data 稀缺, 又解决了 reasoning 深度。

更深层, 它展示了 robotics 和 LLM 融合的一个可行 path: VLM 当脑子, VLA 当肌肉, language 当神经信号。这条路如果走下去, 未来 robot 真的可能像跟同事一样跟你协作。

对咱们做 AI 的人来说, 最大的启发是: **fine-tune 小 model 在 domain data 上, 常常比 scale 大 model 更管用, 前提是你有好的 data generation pipeline**。Hi Robot 的 synthetic data generation 就是这个 pipeline 的典范。

---

参考链接:

- Hi Robot: https://www.pi.website/research/hirobot
- π₀: https://arxiv.org/abs/2410.24164
- PaliGemma: https://arxiv.org/abs/2407.07726
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- SayCan: https://arxiv.org/abs/2204.01691
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- ACT: https://arxiv.org/abs/2304.13705
- Flow Matching: https://arxiv.org/abs/2210.02747
- YAY Robot: https://arxiv.org/abs/2403.12910
- Whisper: https://arxiv.org/abs/2212.04356
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Kahneman book: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

---

# Hi Robot 论文深度解析

Andrej, 这是一篇非常值得细读的 paper,来自 Physical Intelligence (π) 团队,核心作者包括 Lucy Xiaoyang Shi, Karl Pertsch, Sergey Levine, Chelsea Finn 等。我会从 intuition 出发,逐层拆解方法、公式、实验,并尽可能多地补充相关联想。

论文链接: https://www.pi.website/research/hirobot

---

## 1. 核心问题与动机

### 1.1 问题定义

当前 VLA (Vision-Language-Action) models 比如 RT-2, OpenVLA, π₀ 能处理 "pick up the cup" 这类 atomic instructions,但是面对真实世界的复杂 prompt,例如:

> "Could you make me a vegetarian sandwich? I'd prefer it without tomatoes. Also, if you have ham or roast beef, could you make a separate sandwich with one of those for my friend?"

或者执行过程中的实时 correction:

> "that's not how you do it, you have to get lower, otherwise you'll keep missing"

flat VLA 就会失效。Hi Robot 的核心贡献就是让 robot 既能执行 dexterous manipulation,又能理解 open-ended 的语言交互。

### 1.2 System 1 / System 2 的类比

论文借用 Kahneman 在 *Thinking, Fast and Slow* (2011) 中的双系统理论:
- **System 1**: fast, automatic, reactive — 对应 low-level VLA policy,执行 atomic actions
- **System 2**: slow, deliberative, reasoning — 对应 high-level VLM policy,解析复杂 prompt,生成 atomic commands

这个类比很重要,因为它解释了为什么要分两层。flat VLA 试图用一个 model 同时做 System 1 和 System 2,会顾此失彼。Hi Robot 用两个 VLM,通过 language 作为 interface 通信。

参考: Kahneman, *Thinking, Fast and Slow*, Farrar, Straus and Giroux, 2011. https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

---

## 2. 方法详解

### 2.1 Hierarchical Inference 架构

整体架构见图 2。系统分解为两个 policy:

**High-level policy** (System 2):
$$p^{hi}(\hat{\ell}_t | \mathbf{I}_t^1, ..., \mathbf{I}_t^n, \ell_t)$$

- 输入: 多个 camera images $\mathbf{I}_t^1, ..., \mathbf{I}_t^n$ (base camera + wrist cameras),以及 open-ended user prompt $\ell_t$
- 输出: intermediate language command $\hat{\ell}_t$ (e.g., "pick up the lettuce"),以及可选的 verbal utterance $u_t$ 给用户

**Low-level policy** (System 1):
$$p^{lo}(\mathbf{A}_t | \mathbf{I}_t^1, ..., \mathbf{I}_t^n, \hat{\ell}_t, \mathbf{q}_t)$$

- 输入: 同样的 images,加上 high-level 输出的 $\hat{\ell}_t$,以及 robot state $\mathbf{q}_t$ (joint positions + gripper positions)
- 输出: action chunk $\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, ..., \mathbf{a}_{t+H-1}]$,共 $H$ 个 actions

**关键设计**:
- High-level 频率低: 每 1 秒或收到新 user input 时重新 inference
- Low-level 频率高: 持续输出 action chunks
- 当 high-level 输出包含 verbal utterance $u_t$ 时,先 TTS 播放,再从 $\hat{\ell}_t$ 中移除后传给 low-level

### 2.2 公式与变量解析

**(1) Observation 定义**:
$$\mathbf{o}_t = [\mathbf{I}_t^1, ..., \mathbf{I}_t^n, \ell_t, \mathbf{q}_t]$$

- $\mathbf{I}_t^i \in \mathbb{R}^{H_i \times W_i \times 3}$: 第 $i$ 个 camera 在 time $t$ 的 RGB image
- $n$: camera 数量 (single-arm UR5e 用 2 个,bimanual ARX 用 3 个)
- $\ell_t$: language prompt (string,会被 tokenize)
- $\mathbf{q}_t \in \mathbb{R}^{d_q}$: robot configuration,维度取决于 robot (UR5e 是 7D,bimanual ARX 是 14D,mobile ARX 是 14D config + 16D action)

**(2) Action chunk**:
$$\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, ..., \mathbf{a}_{t+H-1}]$$

- $H$: chunk horizon (来自 ACT / Action Chunking with Transformers 思路,Zhao et al. 2023)
- $\mathbf{a}_t \in \mathbb{R}^{d_a}$: single action
- Action chunking 的好处: 减少 compounding error,提高 temporal coherence,允许 low-frequency high-level + high-frequency low-level

**(3) VLM 分布**:
$$p(\ell' | \mathbf{I}, \ell) = \prod_{k=1}^{|\ell'|} p(\mathbf{x}_{t_p+k} | \mathbf{x}_1, ..., \mathbf{x}_{t_p+k-1}, \mathbf{I})$$

- $\ell = [\mathbf{x}_1, ..., \mathbf{x}_{t_p}]$: prefix tokens,prefix length 为 $t_p$
- $\ell' = [\mathbf{x}_{t_p+1}, ..., \mathbf{x}_{t_p+t_s}]$: suffix tokens,suffix length 为 $t_s$
- $\mathbf{x}_k$: 第 $k$ 个 token (注意:这里是 token index,不是 physical time step)
- 这是标准 autoregressive Transformer 的 factorization,PaliGemma-3B 用这种结构

**(4) VLA 的 action 输出**:
Standard VLA 把 action 离散化为 tokens,而 π₀ (Black et al. 2024) 用 **flow matching** 输出 continuous actions。Flow matching 的核心:学习一个 vector field $v_t(\mathbf{x}; t)$,从 noise distribution 到 action distribution 的 ODE flow。这与 diffusion models 相关但更 general (Lipman et al. 2023)。

参考:
- π₀: https://arxiv.org/abs/2410.24164
- Flow Matching: https://arxiv.org/abs/2210.02747
- ACT: https://arxiv.org/abs/2304.13705
- PaliGemma: https://arxiv.org/abs/2407.07726

### 2.3 Synthetic Data Generation — 最关键的创新

这是论文最有意思的部分。问题:如何获得 (复杂 prompt → 正确 low-level command) 的训练数据?人工标注昂贵且覆盖面窄。

**核心思路**: 反向生成。给定 robot 已有的 demonstrations (observation + skill label),让 large VLM 想象什么样的 user prompt 可能导致了这个 skill。

**形式化**:
$$p^{gen}(\ell_t, u_t | \mathbf{I}_t^1, ..., \mathbf{I}_t^n, \hat{\ell}_0, ..., \hat{\ell}_{t-1}, \hat{\ell}_t, \mathcal{P})$$

- $\ell_t$: 生成的 user prompt
- $u_t$: 生成的 robot verbal response
- $\mathbf{I}_t^i$: 当前 images
- $\hat{\ell}_0, ..., \hat{\ell}_{t-1}$: 历史 skill labels (提供 episode context)
- $\hat{\ell}_t$: 当前要解释的 skill
- $\mathcal{P}$: prompt template (包含 task description, scenario categories, response categories)

**Scenario 分类** (确保多样性):
- **Negative task**: "don't add tomatoes" — 用户告诉 robot 不要做什么
- **Situated correction**: "that's not trash" — 基于当前 task state 调整
- **Specific constraint**: "I'm lactose intolerant" — 饮食偏好等约束

**Response 分类**:
- Simple confirmations
- Clarifications
- Error handling

**为什么这个思路 work**:
1. VLM 的 world knowledge 能生成 plausible,grounded prompts ("I want something sweet" → chocolate)
2. Conditioning on prior skills 保证 multi-step coherence
3. 大规模生成覆盖 compositional language space,远超人工标注

**与相关工作的联系**:
- 类似 self-instruct (Stanford Alpaca) 的思路,但 grounded in visual context
- 类似 DAgger 的 motivation: 需要 on-policy data,这里用 synthetic 模拟用户交互
- 逆过程 reminiscent of "inverse reinforcement learning" 的味道: 从 expert demos 推断 intent

### 2.4 数据流总结

1. 收集 teleoperated demos $\mathcal{D}_{demo}$ (粗粒度 task labels 如 "make a sandwich")
2. Segmentation 成 short skills (1-3 秒),得到 $\mathcal{D}_{labeled} = \{(\hat{\ell}_t, \mathbf{I}_t^1, ..., \mathbf{I}_t^n)\}$
3. Heuristic 提取 movement primitives ("move the right arm to the left") 补充
4. 用 $p^{gen}$ (large VLM) 生成 synthetic interactions $\mathcal{D}_{syn}$
5. 训练 high-level policy on $\mathcal{D}_{syn} \cup \mathcal{D}_{labeled}$,cross-entropy loss
6. 训练 low-level policy on $\mathcal{D}_{labeled} \cup \mathcal{D}_{demo}$,flow matching loss

### 2.5 Model Architecture

- **Base VLM**: PaliGemma-3B (Beyer et al. 2024) — 3B 参数,open-source,balance 性能与效率
- **Low-level**: π₀,在 PaliGemma 上加 flow matching "action expert"
- **High-level**: 同样基于 PaliGemma,fine-tune 预测 commands
- 两者 architecture 几乎一致,区别仅在输出层 (flow matching vs. token prediction)
- 全参数 fine-tune,不冻结

**Optimization**:
- AdamW,$\beta_1 = 0.9$, $\beta_2 = 0.95$,no weight decay
- Gradient clip: max norm 1
- EMA decay: 0.999
- LR warmup 1000 steps,然后 constant $1 \times 10^{-5}$
- Batch size: 512
- 训练 high-level: ~2 小时 on 8×H100

---

## 3. 实验设置

### 3.1 三个 Task Domains

**Table Bussing** (清理餐桌):
- 把 dishes/utensils 放入 bussing bin,trash 放入 trash bin
- 物理挑战: plate 要从 edge 抓,要 singulate objects,甚至用 plate 倾倒 trash
- 复杂 prompts: "clean up only the trash, not dishes", "bus all the yellowish things"
- 实时 feedback: "this is not trash", "leave it alone"

**Sandwich Making**:
- 最多 6 种 ingredients + bread
- 物理挑战: deformable,delicate ingredients (cheese slices, lettuce)
- 复杂 prompts: "make a vegetarian sandwich, I'm allergic to pickles"
- 实时 corrections: "that's all, no more"

**Grocery Shopping**:
- 从 grocery shelf 取物,放入 basket,再把 basket 放到 table
- 使用 bimanual mobile manipulator
- 复杂 prompts: "get me something sweet", "get me some Twix and Skittles"
- Interjections: "I also want some KitKat"

### 3.2 三个 Robot Platforms

| Robot | DoF (config) | DoF (action) | Cameras |
|-------|--------------|---------------|---------|
| UR5e | 7 | 7 | wrist + over-shoulder |
| Bimanual ARX | 14 | 14 | 2 wrist + 1 base |
| Mobile ARX (Mobile ALOHA) | 14 | 16 (含 2D base) | 2 wrist + 1 base |

参考 Mobile ALOHA: https://arxiv.org/abs/2401.02117

### 3.3 Baselines

1. **Expert human high-level**: oracle,human 手动输入 low-level commands → 衡量 low-level 的上限
2. **GPT-4o high-level**: 用 GPT-4o API 做 high-level,low-level 还是 π₀ — 类似 advanced SayCan
3. **Flat VLA**: π₀ 直接处理 complex prompts,无 hierarchy
4. **Flat VLA + synthetic data**: π₀ 加 synthetic data,无 hierarchy — 隔离 hierarchy 的影响
5. **Hi Robot without synthetic data**: 只用 human-labeled data — 隔离 synthetic data 的影响

### 3.4 Metrics

**Instruction Accuracy (IA)**: high-level 输出是否 align with user intent + 当前 observation。20 trials per task per method,human evaluator blind to method。

**Task Progress (TP)**: 长程任务,衡量完成比例 (正确放置的 objects 比例)。

---

## 4. 核心实验结果

### 4.1 主结果 (Figure 5)

Hi Robot 在三个任务上显著超越 GPT-4o 和 flat VLA:
- IA 比 GPT-4o 高 40%+
- 接近 Expert Human 的水平

**GPT-4o 的问题**:
- (a) misidentifies objects — 把东西标错成 "plate" 或 "spoon"
- (b) skips subtasks — 漏掉步骤
- (c) ignores user intent — 无视 "leave the rest" 之类
- 还会输出 nonsensical commands 如 "pick up bermuda triangle"
- gripper 还抓着东西就 issue "pick up new object"

**Flat VLA 的问题**:
- 无法 react to real-time feedback
- 默认行为是 "pick up everything in sight"

### 4.2 Ablation 1: Synthetic Data 的重要性 (Figure 7)

去掉 synthetic data:
- 忽略 "this is not trash" 类 corrections
- 会加入 forbidden items 如 pickles
- Synthetic data 提供 compositional language coverage

### 4.3 Ablation 2: Hierarchy 的重要性 (Figure 8)

Flat VLA + synthetic data vs. Hi Robot:
- Flat 会 revert 到 clearing all items
- 无法 handle partial instructions ("bus only the yellowish things")
- Hi Robot 每个 high-level step 重新 check prompt,coherence 更好

---

## 5. 系统与 Latency 细节

### 5.1 Inference 硬件

- 1-2× NVIDIA GeForce RTX 4090 (consumer-grade)
- 也可用 H100

### 5.2 Latency 表格

**Low-level policy (per-step)**:
| Component | Time (ms) |
|-----------|-----------|
| Image encoding | 14 |
| Observation processing | 32 |
| Action prediction (×10) | 27 |
| Total (on-board) | 73 |
| Total (off-board + WiFi) | 86 |

→ ~14 Hz 原始,action chunking 后可 50 Hz

**High-level policy (single decoding step)**:
- RTX 4090: 47 ms (prefill) + 13.2 ms (decode) = 60.2 ms
- H100: 17.3 ms (prefill) + 5.7 ms (decode) = 23 ms

High-level 每 1 秒触发一次,完全有余量。即使 1 秒内多次 user interjection 也能及时响应。

### 5.3 语音处理

- ASR: Whisper large-v2 (本地运行)
- TTS: Cartesia API
- 输入: lavalier microphone

参考 Whisper: https://arxiv.org/abs/2212.04356

---

## 6. Failure Modes (Appendix C.4)

**High-level**:
- 缺乏 long-context memory,长程 reasoning 困难
- 比如多轮交互后忘记早期约束

**Low-level**:
- 临时忽略 instructions: 离 cheese 近就抓,无视 lactose intolerance (training bias toward proximal objects)
- Error accumulation 和 OOD recovery 差: 物体掉了不会 recover

**Mitigation 方向**:
- 更强的 instruction-following model
- Long-context model
- Adversarial data generation for edge cases
- Failure recovery 的 data collection 和 annotation

---

## 7. 与 Related Work 的细致对比

### 7.1 Flat VLA 方法

RT-2 (Brohan et al. 2023a), OpenVLA (Kim et al. 2024), π₀ (Black et al. 2024) 都是 flat,end-to-end VLA。Hi Robot 在这些之上加 high-level reasoning layer,能处理比 "put the cup on the plate" 复杂得多的 prompts。

参考:
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246

### 7.2 LLM/VLM + Predefined Skills

SayCan (Brohan et al. 2023b), PaLM-E (Driess et al. 2023), VoxPoser (Huang et al. 2023), Code as Policies (Liang et al. 2023) 用 LLM/VLM 输出 skill parameters,但 predefined skills 限制了 dexterity。Hi Robot 的 low-level 是 learned VLA,可以执行 dexterous manipulation。

参考 SayCan: https://arxiv.org/abs/2204.01691
参考 PaLM-E: https://arxiv.org/abs/2303.03378

### 7.3 Hierarchical Language-Conditioned Methods

- **OLAF** (Liu et al. 2023): 用 LLM 修改 trajectories,但无法 incorporate 实时 situated corrections
- **YAY Robot** (Shi et al. 2024): 能 handle situated corrections,但限于一个 prompt + 人工数据覆盖的 corrections。Hi Robot 用 VLM + synthetic data 打开 prompt 空间
- **RACER** (Dai et al. 2024): 用 physics simulator 构造 recovery behaviors;Hi Robot 只用 real demos,且适用于 open-ended prompts

参考 YAY Robot: https://arxiv.org/abs/2403.12910
参考 RACER: https://arxiv.org/abs/2409.14674

### 7.4 Hierarchical RL 的历史脉络

Hi Robot 让我联想到 classical hierarchical RL:
- **Options framework** (Sutton, Precup, Singh 1999): options = sub-policies with termination conditions
- **FeUdal Networks** (Vezhnevets et al. 2017): manager worker 分层,latent goals
- **HIRO** (Nachum et al. 2018): off-policy hierarchical

这些方法用 latent goals 作为 interface,Hi Robot 用 natural language 作为 interface — language 更 interpretable,compositional,且能 incorporate human input。

参考 Options: https://arxiv.org/abs/1606.05716

---

## 8. 我的 Intuition Building 和联想

### 8.1 Language 作为层级接口的好处

用 natural language 而非 latent vector 作为 high-level → low-level 的接口,有几个独特优势:
1. **Interpretability**: 可以 inspect 中间 command,debug 友好
2. **Compositionality**: 语言天然 compositional,"pick up" + "the cup" 可以重组
3. **Human-in-the-loop**: 用户可以 inject 任意层级的 correction
4. **Pretraining signal**: VLM 的 web-scale pretraining 直接迁移过来

### 8.2 Synthetic Data Generation 的更深思考

这个思路可以推广。本质上是在做 "inverse policy inference":给定 (state, action),推断 plausible intent。这和 Inverse Reinforcement Learning (IRL) 的 philosophy 一致,但用 VLM 的 world knowledge 来 generate 而非 learn reward。

更广义地,这是一种 **self-play** 的变体:robot 与 imagined user 交互。这与 Constitutional AI (Anthropic) 用 AI feedback 训练 AI 的思路异曲同工。

参考 Constitutional AI: https://arxiv.org/abs/2212.08073

### 8.3 System 1 / System 2 在 AI 中的体现

近期多个工作都在探索这个方向:
- **OpenAI o1**: 显式 reasoning chain,某种意义上是 "runtime System 2"
- **Anthropic 的 "Building Effective Agents"**: orchestrator-worker 模式
- **Tree of Thoughts** (Yao et al. 2023): 显式 search
- **Hi Robot**: 物理 robot 的 System 1/2 分层

Hi Robot 的独特之处:System 1 和 System 2 都是同一个 base VLM (PaliGemma),只是 fine-tune 方向不同。这暗示未来可能 unified model,inference time 切换模式 — 这也是论文 future work 提到的方向。

### 8.4 与 Diffusion Policy / Flow Matching 的关系

π₀ 用 flow matching 而非 diffusion。区别:
- **Diffusion**: 学习 $\epsilon_\theta(\mathbf{x}_t, t)$ 预测 noise
- **Flow matching**: 学习 vector field $v_t(\mathbf{x}; t)$,更 general,可以是任何 probability path

Flow matching 的好处:training 更稳定,sampling 更快,适合 continuous action space。这也是 π₀ 能达到 50 Hz 控制的部分原因。

参考 Diffusion Policy: https://arxiv.org/abs/2303.04137

### 8.5 Limitation 的深层分析

**Long-context memory 缺失**: 当前 high-level 每个 inference 独立,不维护对话历史。这意味着:
- 用户说 "remember, I'm allergic to peanuts" 10 步之后可能被忘记
- 多轮澄清会丢失上下文

**Low-level bias toward proximal objects**: 这是 imitation learning 的经典问题。Robot 学到 "看到 cheese 就抓",与 high-level 的 "don't eat cheese" 矛盾。这本质上是 high-level 和 low-level 解耦训练的代价。

**Coupling 的可能方案** (论文 future work 提到): 让 high-level 知道 low-level 的 success rate。可以想象一个 closed-loop:high-level issue command → low-level 尝试 → feedback → high-level 调整。这有点像 actor-critic,但 with language。

### 8.6 推广到 Multi-Task Unified Model

论文 Appendix A.2 提到 architecture 可以扩展到 unified multi-task formulation。想象一下:一个 high-level policy 跨所有任务,sandwich making 的 knowledge 可以 transfer 到 grocery shopping (都是 "grab food items")。这需要更大的 synthetic data coverage,但潜力巨大。

### 8.7 与 Scaling Laws 的关系

Hi Robot high-level 用 3B PaliGemma,已经显著超越 GPT-4o (推测 ~1T+ params)。这说明:
1. **Task-specific fine-tuning** 比单纯 scale 更重要
2. **Grounding in robot data** 是关键 — GPT-4o 缺这个
3. 但不排除未来用更大的 VLM backbone + 同样的 recipe 会更好

### 8.8 Action Chunking 的数学直觉

为什么 chunking 有效?假设每步误差 $\epsilon$,独立预测 $H$ 步的期望误差 $\sim H\epsilon$。但 chunk 是一次性预测,errors correlated,total error $\sim \epsilon \sqrt{H}$ (如果 errors somewhat independent) 或者更低。同时 chunking 让 policy 学到 temporal structure,不是 greedy step-by-step。

### 8.9 为什么 High-level 也需要 Visual Input

如果 high-level 只处理 language,GPT-4o 纯 text 也能 work。但 situated correction ("that's not trash") 需要 visual context 才能理解 "that" 指什么。这是 Hi Robot 用 VLM 而非 LLM 做 high-level 的根本原因。Figure 6 的 (a) 显示 GPT-4o 会 misidentify objects,部分原因是它的 visual grounding 不够 fine-tuned。

### 8.10 Robotics 与 LLM 的深层联系

Hi Robot 让我想到一个更深的命题:robotics 和 LLM 正在融合。VLA 让 robot "说" actions,VLM 让 robot "想" commands。Hi Robot 展示了这条路可以走多远 — 从 "pick up the cup" 到 "make me a vegetarian sandwich, I'm allergic to pickles, and by the way leave the tomatoes"。

未来可能的发展:
- **Unified VLA**: 单一 model,System 1/2 切换 in inference
- **Long-context robot memory**: 几小时对话 + 任务历史
- **Multi-robot coordination**: high-level 协调多个 robot
- **Self-improvement**: robot 从自己的 failure 中 synthetic 生成 correction data

### 8.11 关于 "Talk to your Robot" 的范式

Hi Robot 某种程度上实现了 "Talk to your Robot like a human" 的愿景。用户可以随意打断、纠正、补充。这比传统的 GUI/programming 接口友好太多。结合 Whisper ASR + Cartesia TTS,整个 pipeline 是 end-to-end speech-to-action。

这让我想到 Andrej 你之前提过的 "Software 2.0" — 这里是 "Robotics 2.0":不是 hand-coded controllers,而是 learned policies + language interface。

---

## 9. 总结

Hi Robot 的核心贡献:
1. **Hierarchical VLA**: VLM (System 2) + VLA (System 1),language 作为 interface
2. **Synthetic data generation**: 用 large VLM 反向生成 plausible user interactions,覆盖 compositional language space
3. **Open-ended instruction following**: 能处理 complex prompts + 实时 feedback
4. **Cross-platform generalization**: single-arm, bimanual, mobile 都 work

核心 takeaway: **task-specific fine-tuned small VLM 可以大幅超越 general large VLM,关键在 grounding + hierarchy + synthetic data coverage**。

这给我们的启示:robotics 的 data efficiency 和 language grounding 比 raw model scale 更重要。未来 unified model + 更长 context + closed-loop high/low coupling 会是 exciting 的方向。

---

## 参考链接汇总

- Hi Robot 主页: https://www.pi.website/research/hirobot
- π₀ (Black et al. 2024): https://arxiv.org/abs/2410.24164
- PaliGemma: https://arxiv.org/abs/2407.07726
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- SayCan: https://arxiv.org/abs/2204.01691
- PaLM-E: https://arxiv.org/abs/2303.03378
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- ACT (Action Chunking): https://arxiv.org/abs/2304.13705
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Flow Matching: https://arxiv.org/abs/2210.02747
- YAY Robot: https://arxiv.org/abs/2403.12910
- RACER: https://arxiv.org/abs/2409.14674
- Whisper: https://arxiv.org/abs/2212.04356
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Kahneman book: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- Options framework: https://arxiv.org/abs/1606.05716

希望这个解析能帮你 build intuition,Andrej。如果你想深入某个具体方面 (比如 flow matching 的数学细节、synthetic data generation 的 prompt template、或者 hierarchy 与 classical options 的对比),我很乐意继续展开。
