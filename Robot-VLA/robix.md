---
source_pdf: robix.pdf
paper_sha256: b968dd4ed8a0098464a15c41c8170e889273834b386f72cbedba6284208f1dd8
processed_at: '2026-08-12T00:11:33-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Robix 用人话再讲一遍

好，我换个讲法，假装我们在白板前聊。

---

## 这玩意到底干啥的

你想象一下你在做一个机器人，让它帮你收拾餐桌。问题是这种任务特别烦：

- 桌上有一堆东西，汉堡、鸡腿、可乐、橙汁、咖啡、纸巾、叉子、刀
- 用户说"把最高卡路里的食物放进塑料盒里，再给我拿个饮料"
- 机器人得**想**：汉堡比鸡腿卡路里高，可乐配汉堡 OK，先放汉堡再放可乐
- 刚拿起可乐，用户突然说"我对咖啡因过敏"
- 机器人得立刻**改主意**：可乐有咖啡因，放回去，换橙汁
- 干完一波，用户又来一句"顺便把整个桌子收拾了，垃圾扔垃圾桶，含咖啡因的饮料也扔，其他东西放塑料盒"
- 机器人得**记住**前面已经放过汉堡橙汁了，现在继续处理剩下的
- 收着收着发现鸡腿没说怎么处理，得**主动问**用户"鸡腿要不要也扔了？"

这一连串——理解模糊指令、被打断还能改、记着之前干过啥、遇到歧义主动问、出错重试——就是 Robix 想搞定的全部。它不是去控制机械臂的电机，它是机械臂上面的**那个会想事儿的大脑**。

---

## 系统长啥样

两层，简单粗暴：

**上层 Robix（VLM）**：看图、听人说话、输出一句"下一步干嘛"加上可能给用户回句话。比如输出 "put the hamburger into the white plastic box"。

**下层 GR-3（VLA）**：拿到这句 atomic command，去真正控制机械臂执行这个动作。

就这么简单。Robix 不碰电机，只输出语言形式的 action。你把它想成一个**坐在控制室看监控画面、给工人发对讲机指令的调度员**，工人（GR-3）负责真去搬东西。

---

## 为什么要这么分层

你问为啥不直接 end-to-end 一个模型从像素到电机 torque？因为：

1. **long-horizon 任务（比如收拾整个桌子）需要 planning 和 reasoning**，低层 VLA 是 reactive 的，看一步走一步，搞不了十几步的任务
2. **human-robot interaction 需要语言能力**，用户随时可能插话改主意，低层控制没这个 interface
3. **分工让两边各自专注**，高层负责"想"，低层负责"做"，跟人类公司一样——CEO 不用自己搬砖

代价就是：高层和低层得对齐 action vocabulary。GPT-4o 直接来当高层就翻车了——它说 "put the biscuit box into the basket"，下层 VLA 根本不认识"biscuit box"这个词，只认得"Oreo"。所以 Robix 跟 GR-3 一起训，保证说的"人话"对方听得懂。

---

## 三步训练，每步干啥

### 第一步：continued pretraining——给 VLM 补"物理世界课"

Qwen2.5-VL 这种通用 VLM 你问它"这个杯子在桌上还是桌下"它经常答得稀烂，因为没怎么见过 3D 空间任务。所以先喂 200B tokens 的"补课数据"：

- **3D 空间理解**：多视角对应、3D bbox、深度排序、绝对深度、相机运动。让它知道图像里东西在物理世界哪儿
- **Visual grounding**：给描述画框，给框生成描述，数数，看 visual prompt。让它能精确指"那个红色的杯子在哪"
- **Task-centric reasoning**：判断任务完了没、某动作可行不、下一步该干啥。让它有 robot 任务感
- **通用多模态**：VQA、OCR、caption、STEM 推理，保持基础能力不掉

这一步出来叫 Robix-Base，已经比 Qwen2.5-VL 在 grounding 上强一大截（LVIS 从 30 提到 70），但还不会跟人交互、不会 planning。

### 第二步：SFT——教它 think-act-respond 这个套路

这是 paper 真正的精华。没有现成的"机器人跟人边聊边干活"的数据集，所以他们自己**合成**。

**数据从哪来**：
- 真人遥操作的 robot demo（自己内部的 + AgiBot 开源的）
- simulator 生成的场景 + 文生图补 sim 支持不了的物品（汉堡西瓜之类）

**合成 7 种交互场景**，每种专门练一个能力：

| 类型 | 例子 | 练啥 |
|---|---|---|
| Multi-stage | "收拾桌子把食物打包" | 长 horizon 任务规划 |
| Constrained | "收拾但别动食物" | 遵守负约束 |
| Open-ended | "把含糖最少的饮料放盒里" | commonsense 推理 |
| Interruption | 中途喊"停下我还要这个" | 实时改主意 |
| Invalid | "把桌子扔垃圾桶里" | 学会拒绝傻指令 |
| Ambiguous | "放个水果"（面前苹果橘子梨） | 主动问清楚 |
| Chat | "桌上有什么水果" | 纯聊天不动作 |

然后让强 VLM 给每个步骤生成一段**思考过程**（chain-of-thought），覆盖：当前场景有啥、上一步成功了没、目标还差啥、下一步该干啥。限制在 200 token 以内——因为机器人得实时反应，不能跟做数学题似的想半天。

**输出格式长这样**：
```
<|think_start|>用户要最高卡路里食物...汉堡比鸡腿高...可乐配汉堡...下一步放汉堡<|think_end|>
<|plan_start|>put the hamburger into the white plastic box<|plan_end|>
<|response_start|>好的<|response_end|>
```

think 是英文，response 是中文，plan 是英文 atomic command。三个字段串成一条 sequence，模型就这么训。

### 第三步：RL——修 SFT 留下的两个毛病

SFT 出来的模型有两个让人头疼的问题：

**毛病 1：think 跟 plan 对不上**
比如 SFT 模型 think 里说"该去 sink 了"，结果 plan 输出"navigate to the cupboard"——想的是一回事，做的是另一回事。因为 SFT 是 next-token prediction，think 和 plan 在 token 层面没有 explicit 一致性约束。

**毛病 2：think 本身就不通**
比如 think 里说"所有饮料都放完了"，结果桌上 milk 还在那儿——纯粹胡说八道。

**修法**：用 GRPO（DeepSeek 那一套 RL 方法），加一个**thought-action consistency reward**——具体就是用 Qwen-2.5-32B 当 judge，看 think 和 plan 是不是逻辑一致，不一致就给负 reward。

训练数据还混入通用视觉推理题（不只 robot 数据），防止模型学窄了。还做了 variance filtering——8 个 sample 的 reward 都一样（要么全对要么全错）的 question 直接丢掉，因为没有 contrastive 信号，留着浪费 batch。

RL 完之后 case study 显示：irrational reasoning 少了，thought-action 一致了，format 错误也少了。

---

## 结果怎么样

**离线评测**（Table 3）：Robix-32B-RL 在所有测试集第一。几个关键数字：
- OOD 任务上比 Gemini-2.5-Pro 高 3.0 / 11.8 个百分点
- 没 CoT 的版本在 open-ended 任务上掉 26.7 个点——证明 reasoning trace 是 OOD 泛化的载体
- GPT-4o 在 invalid instruction 上只有 79 分（容易听不该听的），Robix 100 分

**真机评测**（Figure 5/6）：
- 5 个真实任务平均 task progress 92.6%，跟 Gemini-2.5-Pro 91% 几乎打平
- 比 Qwen2.5-VL-32B 高 64.6 个点（这个对比说明训练 pipeline 真有用）
- GPT-4o 跟 GR-3 配对只有 64.4%，因为 action vocabulary 对不上

---

## 这篇 paper 真正的 contribution

不是模型架构（就是 Qwen2.5-VL continue training），是三件事：

1. **数据合成 pipeline**。7 类交互 × 4 个 reasoning module 的精细 design，把"没有 HRI+planning 数据"这个 fundamental 问题解决了。这才是 moat。

2. **Thought-action consistency reward**。直接 attack SFT 的 failure mode（想一套做一套），是个针对性很强的 reward design。

3. **证明 VLM-VLA alignment 是 deployment bottleneck**。GPT-4o 的失败说明通用大模型直接当 robot brain 不行，得跟低层 controller 在 action vocabulary 上对齐——这给 future work 指了方向：要么 co-train，要么 unified action tokenizer。

---

## 我觉得有意思 / 可疑的地方

**有意思**：
- "think 用英文、response 用中文、plan 用英文"这种多语言 setup 居然 work，说明模型内部 representation 是语言无关的
- Anytime interruption 的处理用 timing-aware heuristic（grasp 前停下、grasp 后放回去）很巧妙，让模型学到 gripper state awareness
- Proactive dialogue 这种"主动问"能力在现有 HRI 工作里很罕见，是真问题

**可疑**：
- Thought-action consistency 用 LLM judge，judge 本身可能不可靠，paper 没讨论 reward hacking
- Memory 只有 short-term（最近 N 个 visual observation），长任务怎么办 paper 自己承认没解决
- Action space 太窄（"put X into Y"），扩展到 general-purpose robot 怎么搞没说
- 真机只测了 5 个 task，scalability 存疑
- 没跟 Gemini Robotics end-to-end 路线 head-to-head 比

整体感觉：这是一篇 **engineering-heavy 的 system paper**，核心价值在 recipe 不在 novelty。它把"如何造一个会边想边干边聊的 robot brain"这件事从 0 走到 1，每一步都有具体的 data synthesis 策略和 reward design，可复现性比那些只贴架构图的高。

参考：
- Robix project: https://robix-seed.github.io/robix/
- GR-3: https://arxiv.org/abs/2507.15493
- UI-TARS (ActRe 思路来源): https://arxiv.org/abs/2501.12326
- DeepSeek-R1 (GRPO): https://arxiv.org/abs/2501.12948
- Hi Robot (类似 hierarchical 思路): https://proceedings.mlr.press/v267/shi25b.html

---

# Robix: 详解一个 Robot 的 "Brain" VLM

Andrej，这篇是 ByteDance Seed 出的 Robix。一句话总结：**Robix 是一个统一的 VLM，作为 hierarchical robot system 的高层 cognitive layer，把 reasoning + planning + human-robot interaction 三件事统一建模成一个 sequential decision-making 过程，通过 chain-of-thought 把 thought-action-response 串成一条序列。** 它的核心 contribution 不在 architecture（就是继续 train Qwen2.5-VL-7B/32B），而在 **数据合成 pipeline + 三阶段训练 + thought-action consistency reward** 这套 engineering recipe。

Project page: https://robix-seed.github.io/robix/

让我一层层拆给你看。

---

## 1. 系统架构：Hierarchical Robot System

Robix 不是 end-to-end 的 VLA，它只负责"高层认知"。系统是两层的（Figure 2）：

```
┌─────────────────────────────────────────────────┐
│  High-level: Robix (VLM)                        │
│  Input: 3 images (head/left/right gripper cams) │
│         + optional user utterance               │
│  Output: thought t_n, action a_n, response r_n  │
└─────────────────────────────────────────────────┘
                    ↓ atomic command
┌─────────────────────────────────────────────────┐
│  Low-level: VLA controller (GR-3)               │
│  or human teleoperation via UMI device          │
│  Execute: put the X into the Y                  │
└─────────────────────────────────────────────────┘
```

这个分层是关键 design choice。低层 VLA (GR-3, https://arxiv.org/abs/2507.15493) 只执行 atomic 动作如 "put the cup into the basket"，Robix 输出的就是这种 "atomic command"，而不是连续 control。这跟 Gemini Robotics (https://arxiv.org/abs/2503.20020) 和 RT-2 那种 end-to-end VLA 路线不同，更接近 Hi Robot (https://proceedings.mlr.press/v267/shi25b.html) 这种 hierarchical 思路，但 Robix 把 interaction 也塞进了同一个 model。

---

## 2. 数学 Formulation：一个 POMDP-like 的 Sequence Model

Section 2 给了一个公式，挺有信息量：

$$
P\Big(t_n, a_n, r_n \mid (o_1, u_1, t_1, a_1, r_1), \dots, [(o_{n-i}, u_{n-i}, t_{n-i}, a_{n-i}, r_{n-i})]_{i=1}^{N}, o_n, u_n\Big) \tag{1}
$$

**变量解释**：
- $t_n$: 第 $n$ 步的 **thought**（chain-of-thought reasoning trace，CoT 内部思考）
- $a_n$: 第 $n$ 步的 **action**（atomic command for low-level controller，比如 "put the fork into the basket"）
- $r_n$: 第 $n$ 步的 **verbal response**（给用户的自然语言回复，可选）
- $o_n$: 第 $n$ 步的 **observation**（3 张 camera 图像：head cam + left gripper cam + right gripper cam）
- $u_n$: 第 $n$ 步的 **user instruction**（可选，用户可以中途插话）
- $N$: **滑动窗口大小**，只保留最近 $N$ 个 visual observations

**关键 design insight**：这里有一个 memory hierarchy。完整的 thought-action 历史 $(t_1, a_1, r_1), \dots, (t_{n-1}, a_{n-1}, r_{n-1})$ 全部保留（文本 token 便宜），但 visual observations 只保留最近 $N$ 个。原因是 token budget：context length 32k，3 张图 × 多步历史会爆炸。这就是 paper Section 6 提到的 limitation——只有 short-term memory，缺 long-term memory with retrieval。

这跟 LLM agent 的 context engineering 思路很像：哪些信息塞进 context window，哪些放外部 memory 用 RAG 取。Robix 现在的做法是"文本全留、视觉只留 recent N"，是个简单但有效的 baseline。

---

## 3. 三阶段训练 Recipe

Robix-7B 和 Robix-32B 都是 continue training Qwen2.5-VL-7B/32B（https://arxiv.org/abs/2502.13923），总共 ~200B tokens，三阶段：

### Stage 1: Continued Pretraining — 建立 embodied reasoning 的底子

目标：让 VLM 有 embodied reasoning 的基础能力（3D spatial + visual grounding + task-centric reasoning），同时保持 general multimodal 能力。

**数据构成**（共 ~200B tokens）：

| Category | Size | Tokens | 子任务 |
|---|---|---|---|
| 3D Spatial Understanding | 30M pairs | 40B | Multi-view correspondence, 3D bbox detection, relative depth sorting, absolute depth estimation, egomotion prediction |
| Visual Grounding | 50M pairs | 70B | 2D bbox, point annotation, counting, visual prompt |
| Task-centric Reasoning | 5M examples | 10B | Task status verification, action affordance, next action prediction |
| General Multimodal Reasoning | 6M pairs | 10B | STEM, multimodal agent, visual inference |
| General Multimodal Understanding | 50M pairs | 80B | VQA, captioning, OCR |
| Instruction Tuning | 1M examples | - | 高质量 instruction-following subset |

**3D Spatial** 的数据来源是 Seed-1.5-VL 的 spatial corpus（https://arxiv.org/abs/2505.07062）+ ScanNet, ScanNet++, 3RScan, CA-1M, SUN RGB-D, ARKitScenes。这一块是为了补 VLM 在 navigation 和 manipulation planning 上缺失的 spatial awareness。

**Visual Grounding** 用了 bbox + center point 两种 format，坐标归一化到 [0, 1000]，跨分辨率统一。这点很重要——Qwen2.5-VL 原生 grounding 能力在 LVIS-MG 上只有 30.6 (7B) / 54.2 (32B) F1，Robix-7B/32B 提升到 70.2 / 79.2，这是 +39.6 / +25.0 的巨大提升。

**Task-centric reasoning** 的数据来自 AgiBot, BridgeData V2, Droid, Egodex, RoboVQA, HoloAssist, Ego4D。三个 sub-task 是关键：
- **Task Status Verification**: 判断 task/subtask 是否完成
- **Action Affordance**: 当前 context 下某 action 是否 feasible
- **Next Action Prediction**: 下一步该干嘛

这三个是 robot reasoning 的"原子能力"。

**训练超参**：
- 5% text-only data 混入（防 catastrophic forgetting）
- LR cosine schedule: $1 \times 10^{-5}$ → $1 \times 10^{-6}$，前 10% steps 线性 warmup
- Sequence length: 32,768 tokens
- Effective batch size: 1536× (7B) / 3008× (32B) sequence length（这个 × 我理解是 grad accumulation 步数）
- AdamW: $\beta_1=0.9, \beta_2=0.99$, weight decay 0.01

### Stage 2: Supervised Finetuning — 把 reasoning + interaction + planning 串起来

这是整个 paper 的 **真正核心**。挑战：没有大规模 multi-turn egocentric HRI + task planning 数据。所以他们造了一套 synthesis pipeline（Figure 4）。

#### 数据源
1. **Teleoperated robot demos**: 内部 GR-3 teleop data + AgiBot open-source。每个 episode 被 human annotator 切成 atomic action clips。
2. **Simulation + AIGC**: 内部 simulator 生成场景 + Seedream 2.0 text-to-image（https://arxiv.org/abs/2501.07062-ish）合成 simulator 不支持的物品（汉堡、意面、西瓜等）。

#### 7 类 Interaction Instructions

这是最体现 engineering 的地方，每一类都有专门的 synthesis 策略：

1. **Multi-Stage Instruction**: 选 >=10 个 atomic action 的 trajectory，从 task name 合成 user instruction（e.g., "clean up the table and pack the food on the plate"）

2. **Constrained Instruction**: 把 trajectory 切成 non-overlapping segments，每段合成一个 constrained 指令（e.g., "Clean up the table while leaving the food on the table"）。这训练模型遵守 negative constraint。

3. **Open-Ended Instruction**: 随机 sim 场景 + LLM 生成 commonsense 指令（e.g., "Place the drink with the least sugar into the carton" 对一个有 Sprite/Coke/OJ/soda 的场景）。这里 text-to-image 的 failure rate 高，**10% 通过 filter 后留下**。

4. **Anytime Interruption**: 把 "Stop!", "Hold on. I still need it" 这类 utterance 随机注入 task flow。**Timing-aware heuristic** 决定 robot 响应：
   - 如果 interrupt 发生在 grasp 之前 → halt 或 replan
   - 如果发生在 grasp 之后 → 把物品放回桌面再 replan
   这个设计是为了让模型维持 "gripper state awareness"。

5. **Invalid Instruction**: 4 类：
   - 物品不存在
   - 物理不可能（"put the table into the rubbish bin"）
   - 超出 robot 能力（"open the coke for me"）
   - 不安全（"throw the knife onto the sofa"）
   训练 robot 学会拒绝。

6. **Ambiguous Instruction**: 多个相似物品 + underspecified instruction（"put a fruit into the basket" 对 apple/orange/pear 场景），训练 robot 主动 clarify。

7. **Chat Instruction**: 在 context-appropriate 时机插入闲聊（"I want some fruit. What kind of fruit is on the table?"），训练 robot 用 response 而不是 action 回应。

#### Reasoning Synthesis

用 strong VLM (Seed-1.5-VL) 生成 <200 token 的 CoT trace，覆盖 4 个 reasoning module：

1. **Scene understanding**: 识别 task-relevant、operable、在 field of view 内的物品
2. **Task status reflection**: 反思 prior action 是否成功、是否需要重试、是否到了 milestone、gripper 是否 holding item
3. **Long-term instruction following**: 跨 long-horizon 持续追踪 initial goal + 中途指令（e.g., "After cleaning the table, grab me a drink from the fridge"）
4. **Next-step analysis**: 评估 reachability + 该 action 是否推进 goal

技术借鉴 **ActRe** + **Thought Bootstrapping**（来自 UI-TARS, https://arxiv.org/abs/2501.12326），再 model-based filter 掉 hallucinated/inconsistent trace。

**关键 trade-off**：跟传统 LLM reasoning 不同，robot reasoning 必须 **concise**（<200 tokens）以支持 real-time interaction。这是 robot CoT 和 math-CoT 的本质区别——前者要快，后者可以长。

### Stage 3: Reinforcement Learning — 修 thought-action consistency

SFT 之后还有两个问题：
1. **Irrational reasoning**: 生成 conflicting thoughts、缺 common sense、忽略 user instruction
2. **Thought-action inconsistency**: think 说要丢 tissue，plan 说要处理 paper cup

举例（paper Appendix C）：
> Task: "Put all the drinks on the table into the carton"
> SFT model think: "all the beverages have been put into the carton, task goal achieved" → 但其实 milk 还在桌上！
> RL model think: "the only beverage left on the table is milk. Next step should be to put the milk into the carton" ✓

#### RL 方法：GRPO

用 Group Relative Policy Optimization（来自 DeepSeekMath/DeepSeek-R1, https://arxiv.org/abs/2402.03300, https://arxiv.org/abs/2501.12948），不用 PPO。

#### 两个核心策略

**策略 1: Co-training with general visual reasoning data**

不只是用 robot interaction data，还混入 general visual reasoning（task completion verification, action affordance, object localization）。目的是减轻 irrational reasoning + 增强 inherent reasoning。

这个 co-training 的 intuition：robot data 太稀疏，纯 robot RL 会让模型 narrow，混入 general reasoning data 起到 regularization 作用。这跟 DeepSeek-R1 训 math reasoning 时混 code data 的思路一致。

**策略 2: Thought-Action Consistency Reward**

Reward 由三部分组成：
- Format reward（输出格式正确）
- Action accuracy reward
- **Thought-action consistency reward**（核心创新）

Consistency reward 的实现：用 Qwen-2.5-32B 作为 external judge LLM，prompt 它判断 action 是否 logically consistent with preceding thought。不一致给 negative reward。Judge prompt 见 Appendix A.5。

这个 reward 的妙处在于：它直接 attack SFT 模型的 failure mode——think 和 plan 解耦。SFT 学到的是 next-token prediction，thought 和 plan 在 token level 是分开的，没有 explicit constraint 强制它们一致。RL 加 consistency reward 就是在 policy gradient 里显式注入这个约束。

#### Variance Filtering（公式 2）

这是 GRPO 的 sample efficiency trick：

$$
\mathcal{D}_{\text{new}} = \left\{(x_n, y_n^*) \in \mathcal{D} \;\Big|\; \text{Var}\left(\{R(y_n^{(i)}, y_n^*)\}_{i=1}^{M}\right) > \tau, \; y_n^{(i)} \sim \pi_{\text{SFT}}(\cdot \mid x_n)\right\} \tag{2}
$$

**变量解释**：
- $\mathcal{D}$: 原始 dataset
- $x_n$: 第 $n$ 个 question（包含 observation, instruction, trajectory history）
- $y_n^*$: ground-truth answer
- $y_n^{(i)}$: 从 SFT policy $\pi_{\text{SFT}}$ 中采样的第 $i$ 个 candidate answer
- $R(y_n^{(i)}, y_n^*)$: reward function，给第 $i$ 个候选打 scalar score
- $M$: 采样数 = 8
- $\tau$: variance threshold = 0
- $\mathcal{D}_{\text{new}}$: 过滤后的 dataset

**Intuition**：GRPO 是 group-relative 的——它用 group 内 advantage $A_i = (R_i - \bar{R}) / \text{std}(R)$ 来 update policy。如果一组 8 个 sample 的 reward variance 为 0，意味着要么全对（$\bar{R}$ 高，但 $A_i = 0$）要么全错（$\bar{R}$ 低，$A_i = 0$），都没有梯度信号。这些 sample 留着反而让 batch 充满 noise。所以预先 filter 掉。

这个 trick 在 RLHF 里其实蛮通用，但 paper explicit 写出来挺好。$\tau=0$ 意味着只保留"至少有一个对、至少有一个错"的 sample——这些才是真正能提供 contrastive signal 的。

#### RL 框架
用 verl (HybridFlow, https://arxiv.org/abs/2409.19256)，ByteDance 自己的 RLHF infra。

---

## 4. 实验：关键 Insights

### Table 1: Fundamental Embodied Reasoning (31 benchmarks)

**3D Spatial Understanding (8 benchmarks)**: Robix-7B/32B 在 7/8 上超过 backbone Qwen2.5-VL-7B/32B。平均 73.4/75.8 vs backbone 66.9/70.7。也超过 Cosmos-Reason1-7B (64.0) 和 RoboBrain-32B (72.2)。但 DA-2k (depth anything) 输给 Gemini-2.5-Pro (83.0 vs 77.1) 和 Seed-1.5-VL-Think (87.5)。

**Visual Grounding (8 benchmarks)**: 全部超过 backbone。LVIS-MG 提升 +39.6/+25.0 F1，这是最显著的 gain。

**Task-centric Reasoning (5 benchmarks)**: Agibot-ER (自己造的 benchmark) 上 +12.8/+7.2 over backbone。RoboVQA 上略输 Gemini/GPT-4o，可能因为这个 benchmark 的 format 跟 Robix 训练 format 不匹配。

**General Multimodal**: 大致保持 backbone 性能，但显著落后 Gemini-2.5-Pro / GPT-4o / Seed-1.5-VL-Think 这些大模型。这是 trade-off——specialize 到 embodied 就会损失一些 general。

### Table 3: Offline Evaluation (curated benchmark)

这是 paper 最关键的 comparison。Robix-32B-RL 在所有 evaluation set 上排第一。

**几个 critical observations**：

1. **CoT 对 OOD 和 open-ended 至关重要**
   - Robix-7B-SFT-wo-R（no reasoning）vs Robix-7B-SFT（with reasoning）
   - Internal OOD: 69.9 → 77.1 (+7.2)
   - ID-OpenEnded: 60.0 → 86.7 (+26.7!!!)
   - 说明 reasoning trace 是 OOD generalization 的载体，没有 CoT 模型就是 pattern matching

2. **RL 的提升**
   - 7B-RL vs 7B-SFT: Internal OOD 77.1 → 85.4 (+8.3)
   - 32B-RL vs 32B-SFT: Internal OOD 83.5 → 86.8 (+3.3)
   - 7B 提升更大，可能因为 7B SFT 留下了更多 inconsistency 给 RL 修
   - Case study 显示 RL 修了三类问题：irrational reasoning, thought-action inconsistency, format error

3. **Gemini-2.5-Pro 是最强 baseline**
   - 在很多 ID benchmark 上接近 Robix-32B-RL
   - 但 OOD 上差距明显：Robix-32B-RL 比 Gemini 在 Internal OOD 高 3.0，AGIBot OOD 高 11.8

4. **GPT-4o 在 Invalid 上只有 79.2**
   - 这是 GPT-4o 容易 comply 不安全指令的体现
   - Robix-32B-RL 在 Invalid 上 100，因为 explicit 训练过 4 类 invalid instruction

### Figure 5: Online VLM Evaluation (UMI human teleop)

5 个真实任务，每个 task-model 跑 4 次取平均，metric 是 task progress（完成 subtask 百分比）。

- Robix-32B avg 92.6%, Gemini-2.5-Pro 91%（几乎打平）
- Qwen2.5-VL-32B 只有 28%（差距 64.6 pts）
- Gemini 的 latency 问题：有时 >30s response time，对 real-time interaction 不可用

### Figure 6: Online VLM-VLA System (Robix + GR-3 + ByteMini robot)

3 个 task，Robix-32B avg 92.5%, Gemini +4.3, GPT-4o +28.1。

**关键发现：VLM-VLA misalignment**
> GPT-4o 生成 semantically correct 但 unrecognizable 的 action。比如 VLA 能执行 "put the Oreo into the shopping basket" 但 GPT-4o 输出 "put the biscuit box into the shopping basket"，VLA 不认识。

这是一个很深的 insight：高层 VLM 和低层 VLA 必须在 **action vocabulary** 上对齐。Robix 因为和 GR-3 一起训练，action space 一致；GPT-4o 没这个 alignment，即使 reasoning 对也 fail。这暗示 future work 应该 co-train VLM 和 VLA，或者用 unified action tokenizer。

---

## 5. Critical Thoughts / Open Questions

我读这篇 paper 时的几个想法：

1. **Memory 是最大 limitation**。Paper 自己承认只有 short-term memory，long-horizon 任务会爆 context。下一步应该是把 LLM 的 context engineering（chunking, retrieval, summarization）搬到 robot。想象一下：把每个 task episode 总结成 "task memory card"，未来类似 task 时 retrieve。这跟 LeCun 的 JEPA / memory-augmented agent 方向接近。

2. **Thought-action consistency reward 用 LLM judge**。这个 reward model 本身可能不可靠。Qwen-2.5-32B 在复杂场景下判断 thought-action 是否一致，本身可能有 bias。未来可能需要 process reward model (PRM) 专门训。OpenAI o1-style 的 PRM 思路。

3. **Reward hacking 风险**。模型可能学到 generate 看起来 consistent 但实际上 wrong 的 thought-action pair。Paper 没讨论这个，但 RL in reasoning 里这是已知问题（DeepSeek-R1 报告过）。

4. **Action space 太窄**。现在只有 "put X into Y", "pick up X", "navigate to X", "open X", "close X"。这对 tabletop manipulation 够用，但 general-purpose robot 需要更丰富的 action space。Future work 可能要 expand 到 continuous action 或 hierarchical action（Robix 输出 sub-goal，下面再 decompose）。

5. **Teleop bottleneck**。SFT data 还是依赖 teleop，scaling 受限。Sim + AIGC 部分只占 10% 通过 filter，质量不够。下一步应该把 sim2real gap 解决，或者用 video diffusion model 生成 robot trajectory（比如 GR00T-style）。

6. **没和 Gemini Robotics 直接比**。Gemini Robotics (https://arxiv.org/abs/2503.20020) 是 end-to-end VLA，路径不同，但 head-to-head 比较 OOD generalization 会很有意思。Robix 是 cognitive layer，Gemini Robotics 是 full-stack，比较不公平但能 reveal trade-off。

7. **Proactive dialogue 这件事被低估**。Figure 1 的 "Should I throw away the drumstick on the green plate as well?" 这种主动 clarify 的能力，对真实 deployment 很关键。现有 HRI 工作很少做这个。这块的 evaluation 也不够 quantitative。

---

## 6. 总结直觉

Robix 的核心 intuition 是：**robot 的"大脑"应该像一个 thinking-out-loud 的 agent，每一步都先 think（scene understanding + task status + goal tracking + next step analysis），再 act（atomic command），必要时 respond（对用户）。** 这个 think-act-respond loop 用 chain-of-thought 串成一个 token sequence，SFT 教模型这个 format，RL 修 thought-action 之间的一致性。

它最值得学的几件事：
- **数据合成 pipeline 比模型架构重要**。7 类 instruction + 4 类 reasoning module 的精细 design 是真正的 moat。
- **RL 的 reward 设计要 attack 具体 failure mode**。thought-action consistency 是针对 SFT 留下的 inconsistency 量身定做，不是 generic reward。
- **VLM-VLA alignment 是真实 deployment 的隐藏 bottleneck**。GPT-4o 的失败证明通用 VLM 直接当 robot brain 不够，必须 co-design。

Reference links:
- Robix: https://robix-seed.github.io/robix/
- GR-3 (low-level VLA): https://arxiv.org/abs/2507.15493
- Seed-1.5-VL: https://arxiv.org/abs/2505.07062
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- UI-TARS (ActRe + Thought Bootstrapping): https://arxiv.org/abs/2501.12326
- verl (RL framework): https://arxiv.org/abs/2409.19256
- Hi Robot (hierarchical VLA + interaction): https://proceedings.mlr.press/v267/shi25b.html
- UMI (Universal Manipulation Interface): https://arxiv.org/abs/2401.01789
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- RoboBrain 2.0: https://arxiv.org/abs/2507.02029
- Cosmos-Reason1: https://arxiv.org/abs/2503.15558
- AgiBot World: https://arxiv.org/abs/2503.06669
- Seedream 2.0 (text-to-image): https://arxiv.org/abs/2501.07062
