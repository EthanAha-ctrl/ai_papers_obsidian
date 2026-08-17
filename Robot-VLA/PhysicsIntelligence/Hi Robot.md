---
source_pdf: Hi Robot.pdf
paper_sha256: a6400a0353705ae3974dec493bb21867885159133586d53da98bacf055fa4333
processed_at: '2026-08-04T23:43:52-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Hi Robot 用人话讲

## 一句话版

Robot 以前只会听 "pick up the cup" 这种短指令，这篇 paper 让它能听懂 "帮我做个三明治，别放番茄，我朋友要个火腿的" 这种长句子，还能在你中途插嘴 "那个不是垃圾" 的时候当场改主意。

## 问题在哪

你给 robot 一个 prompt："能不能帮我做个 vegetarian sandwich？我不要 tomatoes。另外如果有 ham 的话给我朋友也做一个。"

这句话看起来简单，但它要求 robot 做好几件事：

1. 知道 vegetarian sandwich 是什么（哪些 ingredient 不能放）
2. 知道当前桌上有什么（看图）
3. 把这个大任务拆成一步步的小动作（先拿面包，再拿生菜...）
4. 如果做到一半你说 "算了别放 cheese"，它得能改

之前的方法——RT-1、RT-2、OpenVLA、π0——训练的时候只见过 "pick up the coke can" 这种 atomic command。你给它一句话 "pick up the coke can"，它 OK。你给它 "make me a vegetarian sandwich without tomatoes"，它就懵了，因为训练数据里从来没有这种 long-horizon + 多 constraint 的语言。

## 怎么做的——两层

用 Kahneman 的 System 1 / System 2 打比方：

- **System 1 是 fast thinking**：你看到桌子上的杯子，手就伸出去了，不经过大脑。对应 robot 的 low-level policy——它是个 VLA 模型（具体用的是 π0），给一句话 "pick up the cup" + 一张图片，它直接输出一串连续动作。
- **System 2 是 slow thinking**：你在想 "vegetarian sandwich 不该放 ham，所以下一步应该拿 lettuce 而不是拿 roast beef"。对应 robot 的 high-level policy——它是个 VLM（PaliGemma-3B fine-tune 过），给它图片 + 你的复杂 prompt，它输出一句简单的 atomic command，比如 "pick up one piece of lettuce"。

然后 System 1 拿到这句 atomic command 去执行。System 2 每秒 re-run 一次，或者在你说新话的时候立即 re-run，决定下一步该干什么。

所以整个 flow 是：

```
你说: "做个素食三明治，别放番茄"
        ↓
High-level VLM 看图 + 听你的话
        ↓
输出: "pick up bread" (atomic command)
        ↓
Low-level VLA 执行抓面包
        ↓
1秒后 High-level 再看图 + 记住你的要求
        ↓
输出: "pick up lettuce" (因为它知道不能拿 tomato)
        ↓
...循环
```

如果做到一半你说 "那个不是垃圾"，high-level 立刻被 trigger，看一眼图，发现你指的是碗不是垃圾，就改主意说 "pick up the bowl and put it back"。

## 数据的 trick——这是最聪明的地方

问题来了：你想训练 high-level policy，让它会处理 "做个素食三明治别放番茄" 这种话。但 teleop 收集的数据里，operator 只标注了 atomic label（"pick up lettuce", "pick up bread"），没人说过 "做个素食三明治"。

你要给每个 teleop episode 配上复杂的 user prompt，成本爆炸。怎么办？

**反向生成**：拿一个很强的大 VLM，给它一张图 + 一个 atomic skill label，让它想象 "如果人在旁边看着 robot 做这个动作，人之前可能说了什么话"。

比如给它图片（桌上有三明治食材）+ skill label "pick up lettuce"，它可能生成：

- "Can you add some lettuce for me?"
- "I want a vegetarian sandwich, please add lettuce"
- "Put some greens on there"
- "Could you make it healthier?"

还会生成 robot 该怎么回应，比如 "Sure, adding lettuce now"。

这样你把所有 teleop 数据的 atomic label 都反过来 expand 成各种可能的 complex prompt，得到一个巨大的 synthetic dataset，拿它训 high-level policy。

这个 trick 之所以 work，是因为：

1. Teleop 数据告诉模型 **robot 物理上能做什么**（skill 的物理覆盖）
2. 大 VLM 的 pretraining 告诉模型 **人话怎么说**（language 的覆盖）
3. 反向生成把两者 glue 起来，得到 "robot 能做的 skill × 人可能怎么说" 的组合空间

不用为每种新的说法去重新 teleop。VLM 的 world knowledge 免费送给你了——比如它能生成 "I'm lactose intolerant" 这种 prompt，然后 robot response "Sure, I won't put cheese on it"。这个 lactose intolerant → no cheese 的映射不在 robot 数据里，是从 VLM 的 pretraining 来的。

## 为什么比 GPT-4o 直接指挥强

他们试了用 GPT-4o API 当 high-level，给 low-level π0 发指令。GPT-4o 比 PaliGemma-3B 大很多，但效果差很多。原因：

1. **GPT-4o 不懂 robot 能做什么**。你给它一张图，它可能说 "pick up the bermuda triangle"（幻觉），或者把所有东西都叫 "plate"。因为没 fine-tune 过 robot 数据，它对 robot 的 affordance 没概念。

2. **GPT-4o 没有物理状态的 memory**。它会指挥 robot 去抓新东西，但 gripper 还占着上次的东西。fine-tune 过的 small model 反而知道这些，因为它见过 teleop 数据里 gripper 状态和 action 的关系。

3. **GPT-4o 是 black box API**，你不能 cheaply 把 robot affordances 烤进去，只能 prompt engineering 给它一个 allowed commands list 让它选。这个 list 是有限的，覆盖不了 open-ended 情况。

这个 takeaway 对做 robot 很重要：**scale 替代不了 domain grounding**。一个 fine-tune 过的 3B 小模型在 robot 任务上能赢过 100B+ 的通用 API 大模型。

## 为什么 hierarchy 比 flat 强

他们也试了 flat VLA：直接让 π0 消费 complex prompt，不加 high-level。还加了 synthetic data 进去训。结果还是比 hierarchical 差。

为什么？因为 flat model 看一次 prompt 就 commit 了，做完一步它不会回头想 "等等，用户说不要番茄，我应该跳过 tomato"。它倾向 revert 到 default behavior（看到什么抓什么）。

Hierarchical 每秒 re-evaluate 一次 high-level，每次都重新看图 + 重新读 prompt，所以能动态保持 constraint awareness。这就像你开车的时候每秒都在重新决定要不要变道，而不是一开始定好路线就闭眼开。

## 结果有多好

三个 task：清理桌子（UR5e 单臂）、做三明治（ARX 双臂）、超市购物（Mobile ALOHA 双臂+底盘）。

- 比 GPT-4o high-level 高出 40%+ 的 instruction accuracy
- 比 flat VLA 也在所有 task 上显著领先
- Expert human high-level 几乎完美——说明 low-level policy 已经很强，瓶颈全在 high-level reasoning

Ablation 两个关键点：

1. 拿掉 synthetic data，high-level 就处理不了 "this is not trash" 这种 situated correction，也 ignore "I'm allergic to pickles" 这种 constraint
2. 拿掉 hierarchy（变成 flat），即使保留 synthetic data，模型还是 revert to default behavior，处理不了 mid-task 的 "leave the rest" 这种 interjection

## 我觉得最关键的 intuition

1. **System 1 / System 2 在 robot 上比在 LLM 上更自然**。Robot 本来就有物理 frequency 分层——joint control 要 100Hz，task planning 1Hz 就够。hierarchy 是物理强制的，你天然就得拆。LLM 里大家还在争论要不要 CoT、要不要 reasoning tokens，robot 这边直接就 architectural 必然了。

2. **Language 当 hierarchy interface 是聪明的选择**。之前 hierarchy 的 robotics 工作（options framework、feudal RL、HiP) 用 latent vector 当 subgoal，不 interpretable，debug 难。用自然语言当 interface，你直接能读 high-level 在说什么，能 inject 人的 feedback，能 compositional。代价是 high-level 要被训练成 VLM，但这在 2024 年已经很 cheap 了。

3. **Backward data generation 是个 generic trick**。给它 (state, action) 反推 language conditioning，这个 idea 不限于 robot。任何 "数据里有 (x, y) 但没有 language prompt" 的场景都能用。想象一下 self-driving：有 (scene, trajectory) 的数据，让 VLM 反推 "去机场，赶时间"这种 prompt。或者 cooking robot：有 (kitchen state, cooking action) 反推 "做个 medium rare 的 steak"。

4. **Fine-tune > API 的 lesson**。GPT-4o 比 PaliGemma 大几十倍，但在 robot 上输给 fine-tune 的小模型。这说明 robot 任务需要把 physical affordances 烤进 weights 里，光靠 prompt engineering 给一个 allowed list 不够。未来 robot 公司的核心 asset 就是这些 fine-tune 过的 domain-specific 小 VLM。

5. **1 Hz high-level 够用是个 surprising finding**。本来你会觉得需要 skill completion detection、event-driven trigger 这种复杂 scheduling。但 fixed 1 Hz 就 work，说明 atomic skill 粒度选得对（1-3 秒），re-evaluate 频率刚好覆盖。这跟 LLM 里 "reasoning step 要多细" 的讨论类似——粒度选对了，简单 schedule 就够。

## Limitations

- **No memory**：high-level 每次只看当前 frame + 当前 prompt，记不住 5 分钟前你说过什么。要做 "今天早上我说过要 X" 这种需要 long-context memory。
- **High-level 不知道 low-level 成没成功**：两层 decoupled 训练。如果 low-level 抓失败了，high-level 还是以为成功了，继续往下列一步。需要 closed-loop feedback。
- **Synthetic data 依赖 prompt engineering**：每个 task 要手写 generation template，还不能完全自动。
- **Low-level 还会被 proximal object bias 带偏**：你说 lactose intolerant，但 cheese 就在 robot 手边，low-level 还是可能去抓，因为它训练数据里 proximal object 容易被抓。

## 跟其他方向的联系

- 跟 **LLM 里的 implicit CoT** (o1, R1)：都是 System 2 reasoning，但 robot 这边用 explicit language token 当 reasoning medium，LLM 那边用 latent。两边都在探索 unified model（一个 model 两 mode）vs separate model 的 trade-off。
- 跟 **RLHF / DAgger**：synthetic data generation 是 offline 版的 DAgger iteration。下一步可以做成 online——deploy 当前 policy，让人 correct，再 generate 新 synthetic data，闭环。
- 跟 **SayCan**：SayCan 是 inference-time 让 LLM 对 predefined skill 做 affordance ranking。Hi Robot 是 training-time 把 VLM fine-tune 成 high-level policy，skill 是 learned VLA 而不是 hand-defined。可以看成 SayCan 的 end-to-end 学习版本。
- 跟 **VLA 谱系** (RT-1 → RT-2 → π0)：都是 flat System 1。Hi Robot 在上面加 System 2，是自然延伸。

## 你可能想深入的方向

- Flow matching 在 π0 里具体怎么实现的（action expert 的 architecture、denoising step 数的 trade-off）
- Synthetic data generation 的 prompt template 具体长什么样（Appendix A 有但没全展开）
- High-level 和 low-level 合并成 unified model 的话，训练 objective 怎么设计
- 怎么加 long-context memory 让 robot 记住整个对话历史

相关链接：

- Hi Robot 项目页：https://www.pi.website/research/hirobot
- π0 paper：https://arxiv.org/abs/2410.24164
- PaliGemma：https://arxiv.org/abs/2407.07726
- SayCan：https://arxiv.org/abs/2204.01691
- YAY Robot：https://arxiv.org/abs/2403.12910
- Mobile ALOHA：https://arxiv.org/abs/2401.02117

想深入哪块直接问，我可以把公式拆到更细的 level。

---

# Hi Robot 深度讲解 — Hierarchical VLA for Open-Ended Instruction Following

Andrej 你好，这篇 paper 是 Physical Intelligence (Pi) 出的工作，作者阵容包括 Sergey Levine、Chelsea Finn、Karl Pertsch、Danny Driess 等人，第一作者是 Lucy Xiaoyang Shi。整体上可以看作是 **π0 + hierarchy + synthetic interaction data** 的组合，目标是让 robot 处理 open-ended、可被中途打断、可被 situated correction 的语言指令。我下面会逐层拆解，把 intuition 建起来。

---

## 1. The Big Picture — System 1 / System 2 in Robotics

这篇 paper 的核心 motivation 直接对应 Kahneman 的 *Thinking, Fast and Slow* 里的 System 1 / System 2 划分：

- **System 1 (fast, automatic)**: 对应 low-level VLA policy（π0）。它知道怎么把 "pick up the coke can" 翻译成连续动作 chunk。
- **System 2 (slow, deliberative)**: 对应 high-level VLM policy。它负责把 "Could you make me a vegetarian sandwich? I'm allergic to pickles" 这种 complex prompt + 当前 visual observation + 实时 user feedback 综合起来，分解成下一拍该执行的 atomic command。

之前的工作比如 RT-1 / RT-2 / OpenVLA / π0 都是 flat VLA，本质上是 System 1-only——它们直接把 raw prompt 当成 atomic command 来 condition，碰到 "make a vegetarian sandwich without tomatoes" 这种 prompt 就崩了，因为训练数据里没有这种 long-horizon 的复杂指令。

Hi Robot 的 solution 是 explicit hierarchy：把 $p(\mathbf{A}_t | \mathbf{o}_t)$ 拆成两层。这让人想起 LLM 里 chain-of-thought / implicit reasoning tokens 的争论，但这里是 **物理层面的 hierarchy**——System 2 输出的是自然语言中间指令 $\hat{\ell}_t$，System 1 再消费它。这比 latent reasoning tokens 更 interpretable，也更 modular。

---

## 2. Formal Setup

### 2.1 标准 VLA formulation

Observation:
$$\mathbf{o}_t = [\mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \ell_t, \mathbf{q}_t]$$

- $\mathbf{I}_t^i$: 第 $i$ 个 camera 的 image（base / wrist / overhead）
- $\ell_t$: language prompt
- $\mathbf{q}_t$: robot proprioception (joint positions + gripper state)

Action chunk:
$$\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H-1}]$$

- $H$: chunk horizon (π0 里大约 50)
- $\mathbf{a}_t$: 单步 action (e.g., 7-DoF for UR5e, 14-DoF for bimanual ARX, 16-D for mobile ARX)

Policy distribution: $p(\mathbf{A}_t | \mathbf{o}_t)$

### 2.2 VLM 的 factorization

VLM 本质上是一个 distribution over text suffixes:
$$p(\ell' | \mathbf{I}, \ell)$$

其中 $\ell$ 是 prefix (image + prompt tokens)，$\ell'$ 是 suffix (response tokens)。

Autoregressive 分解:
$$p(\ell' | \mathbf{I}, \ell) = \prod_{k} p(\mathbf{x}_{t_p + k} | \mathbf{x}_1, \ldots, \mathbf{x}_{t_p + k - 1}, \mathbf{I})$$

- $\mathbf{x}_t$: 第 $t$ 个 token (注意这里 $t$ 是 token index, not time step)
- $t_p$: prefix length
- $t_s$: suffix length

VLA 本质上就是把 $\mathbf{A}_t$ 通过 discretization (RT-2) 或 flow matching (π0) 编码成 suffix tokens。

### 2.3 Hierarchical decomposition

Hi Robot 的核心分解：

**High-level policy**:
$$p^{\text{hi}}(\hat{\ell}_t | \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \ell_t)$$

- 输入：images + open-ended prompt $\ell_t$ (可能包含 user interjection)
- 输出：atomic command $\hat{\ell}_t$（e.g., "pick up one piece of lettuce"）+ optional verbal utterance $u_t$

**Low-level policy**:
$$p^{\text{lo}}(\mathbf{A}_t | \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \hat{\ell}_t, \mathbf{q}_t)$$

- 输入：images + atomic command $\hat{\ell}_t$ (来自 high-level) + robot state $\mathbf{q}_t$
- 输出：action chunk $\mathbf{A}_t$
- 实现：π0 VLA, 用 flow matching 生成 continuous actions

注意 high-level **不接收** $\mathbf{q}_t$——它只关心 visual scene 和 user intent，不关心具体关节角。这是一个干净的 interface 划分。

### 2.4 Inference 的 frequency

- Low-level: 高频运行 (~10 Hz 推理，但用 action chunking 可以做到 50 Hz 控制)
- High-level: 每秒一次，或在 user interjection 时立即触发

这是一个简单但 effective 的 schedule。理想情况下应该有 "skill 完成检测" 来 trigger 下一拍 high-level inference，但他们用 fixed 1 Hz 已经 work，这点很有意思——说明 high-level 的指令粒度足够粗，1 秒的窗口足够让 low-level 完成 atomic skill（atomic skill 通常 1-3 秒）。

---

## 3. Data Pipeline — Synthetic Interaction Generation 是关键

这一节我认为是 paper 里最巧妙的部分。问题：你想训练 high-level policy 处理 "make me a vegetarian sandwich without tomatoes, oh and also make a ham sandwich for my friend" 这种 prompt，但 teleoperation 数据里只有 atomic skill label（"pick up lettuce", "pick up bread"），没有对应的 complex prompt。

### 3.1 思路：Reverse generation

他们用一个大 VLM $p^{\text{gen}}$（不是训练用的 high-level policy）做 **backward 数据生成**：给定 observation 和 skill label，generate 可能导致这个 skill 的 user prompt。

形式化:
$$p^{\text{gen}}(\ell_t, u_t | \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \hat{\ell}_0, \ldots, \hat{\ell}_{t-1}, \hat{\ell}_t, \mathcal{P})$$

- $\ell_t$: 生成的 user prompt
- $u_t$: 生成的 robot verbal response
- $\hat{\ell}_0, \ldots, \hat{\ell}_{t-1}$: 之前已经完成的 skill labels (上下文)
- $\hat{\ell}_t$: 当前 skill label
- $\mathcal{P}$: task-specific prompt template

这本质上是把数据集做 **backward relabeling**——给定 (state, action) 重新 imagine 出 language conditioning signal。这跟 RL 里的 inverse RL 思想、跟 LLM post-training 里的 SFT data generation、跟 DAgger 的 iteration 都有 conceptual 联系，但这里是 **observation + action label → language prompt** 的方向，比较新颖。

### 3.2 为什么这个 trick work

直觉上：
1. Teleop 数据覆盖了 **physical capabilities**（robot 能做什么 skill）
2. VLM 的 web-scale pretraining 覆盖了 **language understanding + world knowledge**
3. 把两者 glue 起来，就得到了 "robot 能做的 skill" × "可能的语言表达" 的组合空间

这避免了为每个新 prompt 类型去重新 teleop 收数据的 expensive loop。这也跟 SayCan 的 affordance scoring 思路有点像，但 SayCan 是 inference-time 的 re-ranking，这里是 **training-time 的 data augmentation**——把组合性 bake 进 weights 里。

### 3.3 Scenario categorization

为了让 synthetic data diverse，他们把 interactions 分成几类（follows Stephan et al., 2024 RLVF）：

- **Negative task**: "Don't add tomatoes"
- **Situated correction**: "That's not trash" (需要看 image 才能理解)
- **Specific constraint**: "I'm lactose intolerant"

Robot response 也分 type: simple confirmation, clarification, error handling.

这种 categorization 让生成过程可控，避免 VLM 自由发挥产生 garbage data。这是个典型的 **structured prompting** 实践。

### 3.4 World knowledge 的 role

VLM 的 pretraining 在这里悄悄起作用——比如 sandwich task 里，模型能 generate "I'm lactose intolerant" 这种 prompt，然后 robot response "Sure, I won't put cheese on it"。这种 dietary knowledge 不在 robot 数据里，是从 VLM 的 pretraining 来的。这是个很好的 demonstration of **frozen VLM knowledge → robot data augmentation** 的 pipeline。

---

## 4. Architecture Details

### 4.1 Base model

PaliGemma-3B 作为 high-level 和 low-level 共享的 base VLM。这是 Google 出的小型 VLM，3B 参数。

- High-level: fine-tune PaliGemma 用 cross-entropy next-token prediction，输出 $\hat{\ell}_t$
- Low-level: π0 VLA，在 PaliGemma 上加一个 **flow matching action expert**，输出 continuous action chunk

Low-level 的 flow matching 是 π0 的核心 trick——避免 action discretization (像 RT-2 那样把 7-DoF action tokenize 成 bin indices)，直接用 flow matching 学 continuous distribution $p(\mathbf{A}_t | \mathbf{o}_t)$。

### 4.2 为什么不用同一个 model

Paper 最后提到一个 future direction: 把 high-level 和 low-level 合并成一个 model，在 inference time 区分 System 1 / System 2 模式。这跟 LLM 里 o1 / R1 的 implicit CoT 思路相通——你不需要 separate model 来做 reasoning vs fast response，可以一个 model 在不同模式下运行。

但目前 decoupled 的好处是：
- High-level 可以用更便宜的训练 (2 hours on 8×H100)
- Low-level 的 action expert 不需要被 language generation 任务干扰
- 数据 decouple: high-level 用 synthetic，low-level 用 real teleop

### 4.3 Latency budget (实测 on RTX 4090)

Low-level policy per-step:
| Component | Time (ms) |
|---|---|
| Image encoding | 14 |
| Observation processing | 32 |
| Action prediction (×10 denoise steps) | 27 |
| **Total on-board** | **73** |
| Total off-board + WiFi | 86 |

High-level policy single decoding step:
- RTX 4090: 47 ms (prefill) + 13.2 ms (decode)
- H100: 17.3 ms (prefill) + 5.7 ms (decode)

So 总 system 能跑 ~10 Hz high-level inference + ~13 Hz low-level inference (单步)，结合 action chunking 能做到 50 Hz 控制。这是相当 real-time friendly 的数字。

---

## 5. Training Objectives

### 5.1 High-level policy loss

Cross-entropy for next-token prediction:
$$\mathcal{L}_{\text{hi}} = -\sum_{k} \log p^{\text{hi}}(\mathbf{x}_{k} | \mathbf{x}_{<k}, \mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \ell_t)$$

训练数据: $\mathcal{D}_{\text{syn}} \cup \mathcal{D}_{\text{labeled}}$

### 5.2 Low-level policy loss (flow matching)

Flow matching (Lipman et al., 2023) 的核心 idea 是学一个 vector field $v_\theta(\mathbf{A}, t)$，把简单 prior (Gaussian) flow 到 data distribution $p(\mathbf{A}_t | \mathbf{o}_t)$。

具体训练时，从 data $\mathbf{A}_t^*$ 出发，sample noise $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ 和时间 $t \sim \mathcal{U}[0, 1]$，构造:
$$\mathbf{A}_t^{\text{interpolate}} = (1 - t) \boldsymbol{\epsilon} + t \mathbf{A}_t^*$$

Loss:
$$\mathcal{L}_{\text{lo}} = \mathbb{E}_{t, \boldsymbol{\epsilon}} \left[ \| v_\theta(\mathbf{A}_t^{\text{interpolate}}, t, \mathbf{o}_t) - (\mathbf{A}_t^* - \boldsymbol{\epsilon}) \|^2 \right]$$

- $v_\theta$: 神经网络预测的 vector field (在 π0 里由 action expert 实现)
- $t$: flow time, $t=0$ 时是 noise, $t=1$ 时是 data
- $\boldsymbol{\epsilon}$: sampled Gaussian noise
- $\mathbf{A}_t^*$: ground-truth action chunk from teleop

推理时用 10 步 Euler ODE 从 noise sample 到 action chunk。这就是 latency 表里 "Action prediction (×10)" 的来源。

训练数据: $\mathcal{D}_{\text{labeled}} \cup \mathcal{D}_{\text{demo}}$ (不需要 synthetic，因为 low-level 只处理 atomic command)

### 5.3 Optimizer

AdamW, $\beta_1 = 0.9$, $\beta_2 = 0.95$, no weight decay, grad clip = 1, EMA decay = 0.999, LR = $1 \times 10^{-5}$ (constant after 1000-step warmup), batch size = 512.

注意 batch size 512 对于 robot data 来说很大——意味着他们有大量 teleop 数据。这跟 Pi 一向的数据 strategy 一致。

---

## 6. Tasks and Evaluation

### 6.1 三个 task domains

| Task | Robot | Physical challenge | Reasoning challenge |
|---|---|---|---|
| **Table Bussing** | UR5e single-arm | Edge-grasp plates, tilt to dump trash | "Bus only yellowish items", "this is not trash" |
| **Sandwich Making** | Bimanual ARX | Deformable ingredients, precise placement | "Vegetarian sandwich, no tomatoes", "I'm allergic to pickles" |
| **Grocery Shopping** | Mobile ALOHA (bimanual + base) | Tall bottles, navigation + manipulation | "Get me something sweet", "also want some KitKat" |

这三个 task 覆盖了 single-arm / bimanual / mobile 三种 robot platform，physical difficulty 递增，reasoning 维度也不同。

### 6.2 Baselines

1. **Expert human high-level**: oracle，人手写 atomic command。这衡量 low-level policy 的 ceiling。
2. **GPT-4o high-level**: 用 GPT-4o API 做 high-level reasoning，low-level 还是 π0。Prompt 里 hardcode 一组 allowed commands 让 GPT-4o 选——类似 SayCan 的升级版。
3. **Flat VLA**: 直接用 π0 消费 raw complex prompt，无 hierarchy。
4. **Flat VLA + synthetic data**: 把 synthetic data 加进 low-level 训练，但保持 flat 结构。这隔离 hierarchy 的贡献。
5. **Hi Robot w/o synthetic data**: 仅用 human-labeled data 训 high-level。这隔离 synthetic data 的贡献。

### 6.3 Metrics

- **Instruction Accuracy (IA)**: high-level 输出的 $\hat{\ell}_t$ 是否 align with user intent + 当前 observation。20 trials/task/method, human evaluator blind to method。
- **Task Progress (TP)**: 完成的 sub-goals 比例。

### 6.4 Main results (Figure 5)

Hi Robot 在三个 task 上都显著超过 GPT-4o 和 flat VLA。关键 qualitative observations：

- GPT-4o 经常 **misidentify objects** (把所有东西叫 "plate" 或 "spoon")，发出 nonsense commands 如 "pick up bermuda triangle"
- GPT-4o 缺乏 **physical grounding**——会指挥 robot 去抓东西，但 gripper 还占着上次的物体
- Flat VLA 完全 **无法 react to mid-task feedback**
- Hi Robot 能处理 "I also want a KitKat" 这种 mid-task interjection

Expert human high-level 几乎完美，说明 **low-level policy 已经很强**，瓶颈在 high-level reasoning。

### 6.5 Ablation takeaways

1. **Synthetic data is critical**: 没有它，high-level 处理不了 "this is not trash" 这种 situated correction，也 ignore "I'm allergic to pickles" 这种 constraint。
2. **Hierarchy is critical**: flat VLA 即使加了 synthetic data，依然比 hierarchical 差。因为 flat model 看一次 prompt 就 commit 了，无法 re-check；hierarchical 每秒 re-evaluate，能动态 adapt。

---

## 7. 与 Related Work 的关系梳理

### 7.1 Flat VLA 谱系

- **RT-1** (Brohan et al., 2022): Transformer-based, discretized actions, atomic instructions
- **RT-2** (Brohan et al., 2023a): VLM-based, action tokenization, web knowledge transfer to robot
- **OpenVLA** (Kim et al., 2024): 开源版 RT-2 思路
- **π0** (Black et al., 2024): PaliGemma base + flow matching, Hi Robot 的 low-level 就是它

这些都是 System 1-only。Hi Robot 在它们之上加 System 2。

### 7.2 LLM/VLM as planner 谱系

- **SayCan** (Brohan et al., 2023b): LLM 输出 next skill，affordance model re-rank
- **Code as Policies** (Liang et al., 2023): LLM 输出 code
- **VoxPoser** (Huang et al., 2023): VLM 输出 3D value maps
- **PIVOT** (Nasiriany et al., 2024): Visual prompting 迭代

这些方法 skill 是预定义的，dexterity 受限。Hi Robot 用 learned VLA 当 skill library，dexterity 高很多。GPT-4o baseline 实际上就是这类方法的升级版。

### 7.3 Language feedback 谱系

- **OLAF** (Liu et al., 2023): LLM 修改 trajectory，但不是 situated
- **YAY Robot** (Shi et al., 2024): 能处理 situated correction，但 only 一个 prompt + 只见过的 correction
- **RACER** (Dai et al., 2024): 用 simulator 构造 recovery
- **RLVF** (Stephan et al., 2024): Verbal feedback relabeling, 启发了 Hi Robot 的 scenario categorization

Hi Robot 的 synthetic data generation 是这些方向的 generalization——用 VLM 的 world knowledge 自动 cover 各种 feedback type。

### 7.4 Hierarchy 在 LLM 和 robotics 中的 dual

- 在 LLM 里，hierarchy 表现为 implicit chain-of-thought (o1, R1)
- 在 robotics 里，hierarchy 历史上是 options framework (Sutton et al., 1999) / feudal RL (Vezhnevets et al., 2017)
- Hi Robot 是 VLM 时代的 hierarchy——用 language 作为 hierarchy interface，这比 latent subgoal representation 更 interpretable

---

## 8. Limitations and Future Directions

Paper 自己提到的：
1. **No long-context memory**: high-level 每次只看当前 frame + current prompt，没有跨 episode memory。这意味着 "remember what I asked 5 minutes ago" 做不到。
2. **High-level 不感知 low-level 的 success**: 完全 decoupled 训练，high-level 不知道 low-level 是否真的完成了上一个指令。
3. **Prompt engineering for synthetic gen**: 依赖 $p^{\text{gen}}$ 的 prompt template $\mathcal{P}$，每个 task 都要写。
4. **Low-level 仍会被 proximal object bias 干扰**: 抓 cheese 即使 user 说 lactose intolerant。

我自己的联想：
- **Skill boundary detection**: 现在用 fixed 1 Hz 是 hack，理想是 event-driven (skill completion detection)。这跟 LLM 里 "when to stop thinking" 的问题同构。
- **Unified model**: paper 提到可以一个 model 两 mode，这跟 GPT-4o 的 unified multimodal 思路一致。但 unified 之后怎么训练？synthetic data 怎么 mix with real teleop？开放问题。
- **Self-play 数据闭环**: 现在是 VLM 一次生成 synthetic data。可以做 iterative——deploy 当前 high-level policy，让人来 correct，再 generate 新 synthetic data。这是 RLHF 在 robotics 里的对应物。
- **Long-horizon reasoning**: 现在 hierarchy 是 2-level，对于超长 horizon (e.g., "make dinner for 4 people") 可能需要 multi-level hierarchy 或者 explicit symbolic planning layer。

---

## 9. 我的 Intuition 总结

把这篇 paper 的 intuition 浓缩成几条：

1. **VLA 是 System 1，它处理 "怎么做" 的 physical skill；VLM 是 System 2，它处理 "做什么 next" 的 situated reasoning**。两者用 natural language 当 interface，比 latent vector 更 interpretable、更 composable。

2. **Synthetic data generation 是 backward relabeling**: 给定 (observation, skill) 让 VLM 反推 user prompt。这利用了 VLM 的 world knowledge 来 cheaply 覆盖 language × skill 组合空间，避免了 expensive teleop loop。

3. **Hierarchy 比 flat 强是因为 flat commit 一次就僵了，hierarchical 每秒 re-evaluate 能 dynamic adapt**。这跟 model-free RL 里 receding horizon control 的哲学一致。

4. **GPT-4o 这种 API VLM 即使很大也输给 fine-tuned 3B**——因为 fine-tuning 把 robot affordances 烤进 weights 里了，API VLM 即使 prompt-engineered 也缺乏 physical grounding。这是 LLM era 的一个重要 lesson：**scale 不能替代 domain-specific grounding**。

5. **PaliGemma-3B 这个尺寸选得很务实**——能 onboard 跑 real-time，能 fine-tune 2 小时，能复用给 high/low level。这暗示未来 robotics 的 sweet spot 可能是 mid-size (3-7B) VLM + 任务特定 fine-tune，而不是 API-based giant VLM。

6. **System 1 / System 2 在 robotics 里的 instantiation 比 LLM 更自然**——因为 robot 本来就有 physical frequency 分层 (joint control 100Hz vs task planning 1Hz)，hierarchy 是物理强制的。LLM 里的 System 1/2 还在 debate，robotics 里它就是 architectural 必然。

---

## References / Useful Links

- Hi Robot project page: https://www.pi.website/research/hirobot
- π0 paper (Black et al., 2024): https://arxiv.org/abs/2410.24164
- PaliGemma (Beyer et al., 2024): https://arxiv.org/abs/2407.07726
- OpenVLA (Kim et al., 2024): https://arxiv.org/abs/2406.09246
- RT-2 (Brohan et al., 2023a): https://arxiv.org/abs/2307.15818
- SayCan (Brohan et al., 2023b): https://arxiv.org/abs/2204.01691
- YAY Robot (Shi et al., 2024): https://arxiv.org/abs/2403.12910
- RLVF (Stephan et al., 2024): https://arxiv.org/abs/2402.10893
- Flow Matching for Generative Modeling (Lipman et al., 2023): https://arxiv.org/abs/2210.02747
- Mobile ALOHA (Fu et al., 2024): https://arxiv.org/abs/2401.02117
- Kahneman, *Thinking, Fast and Slow*: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- Physical Intelligence blog: https://www.physicalintelligence.company/blog

如果你想深入聊某个细节，比如 flow matching 在 π0 里的具体实现，或者 synthetic data generation prompt template 的具体设计，我可以再展开。
