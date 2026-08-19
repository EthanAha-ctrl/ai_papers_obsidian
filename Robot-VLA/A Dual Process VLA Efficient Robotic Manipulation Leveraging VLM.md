---
source_pdf: A Dual Process VLA Efficient Robotic Manipulation Leveraging VLM.pdf
paper_sha256: 3425c3f2792ddd8f0e1f910c5f2e827c06c0fad2c0245e29d8dcbb4a44aff722
processed_at: '2026-08-17T23:12:57-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DP-VLA

跟 robot 说"把那个咖啡杯放到微波炉里".
你脑子里发生了什么?
你可能花了一秒种 "理解" 这句话 —— 找到咖啡杯在哪、找到微波炉在哪、规划一下大致动作。
然后你伸手去抓的时候,你不会每 0.1 秒就重新 "理解" 一次这句话。
你只是在执行,根据手的感觉、眼睛看到的细节做微调。

这篇 paper 就是把这个 common sense 塞进 robot 里。

之前的 VLA model(比如 RT-2、OpenVLA)相当于让 robot 每一帧都重新做一遍 SAT 阅读理解
DP-VLA 说: 只需要在指令来的时候做一次 "理解", 把理解结果压成一个 4096 维的 vector z_i,后面执行的时候反复 query 这个 vector 就行。

把 thinking 和 doing 拆到两个 timescale 上。

- L-Sys2 的 forward pass = **prefill**(把 instruction + 初始图像读进去,生成 z_i 这个 "KV cache")
- S-Sys1 的每帧 forward = **decode**(用 z_i 当 cache,根据当前 observation 生成 action)

Monolithic VLA 问题在于它每一帧都重新 prefill
CEO 不需要每 50ms 都重新发一个 memo,员工也不需要每 50ms 都请示 CEO。memo 一旦发出就够用一阵子。
### 那个 z_i 到底是啥?

$$\mathbf{z_i} = f_\ell(v_0, \ell_i)$$

- $f_\ell$:就是 OpenVLA 这个 model(参数固定,不训)
- $v_0$:episode 第一帧的 image
- $\ell_i$:你的 language instruction,比如 "pick up the mug"
- $\mathbf{z_i}$:一个 4096 维的 vector

这个 vector 本质上就是 OpenVLA 在 internal hidden state 里的 "我已经理解了任务,准备开始行动" 的 representation。它包含了 "对象在哪、目标在哪、大致要做什么" 这些 semantic 信息。

然后 S-Sys1(BC-Transformer)每帧这样工作:

$$\mathbf{a_t} = f_s(\mathbf{o_t}, \mathbf{s_t}, \mathbf{z_i})$$

- $\mathbf{o_t}$:当前帧的多视角图像(左相机、右相机、eye-in-hand,128×128)
- $\mathbf{s_t}$:robot 的 state(end-effector position、gripper 状态等)
- $\mathbf{z_i}$:刚才那个固定的 "CEO memo"
- $\mathbf{a_t}$:7 维 action(6 维 end-effector 移动 + 1 维 gripper 开合)

**注意**:z_i 在整个 episode 里不变,只有 o_t 和 s_t 在变。这就是 amortized reasoning 的数学表达。

### 实验数据,人话翻译

Table 1 里几个关键数字:

| 方法 | 平均成功率 | 解释 |
|---|---|---|
| OpenVLA-ft | 5.8% | 完全 fine-tune 的 7B VLA,惨败 |
| BC-Transformer | 42.6% | 纯小模型,缺 reasoning |
| DP-VLA | 52.9% | 两者组合 |

**OpenVLA-ft 为什么这么惨?** 5.8%!因为它在 OXE(real-world robot data)上 pretrain,转到 MuJoCo 仿真里水土不服。而且它的 attention 机制对 "抓取精度" 这种 fine-grained 任务不擅长 —— 它擅长 "大致移动",不擅长 "1mm 精度抓取"。

**BC-Transformer 为什么 42%?** 它在 fine-grained 上 OK,但理解不了 "single door vs double door"、"open vs close" 这种 semantic 区别。

**DP-VLA 为什么 52.9%?** 它拿 OpenVLA 的 semantic understanding(z_i 提供 "我要开 double door" 这个 semantic 信号)+ BC-Transformer 的 fine-grained motor control,1+1>2。

### 反直觉发现:Fine-tune L-Sys2 反而变差

Table 4 那个数字我觉得是这篇 paper 最 surprising 的:

- 用 pretrained OpenVLA 当 L-Sys2:**55.6%**
- 用 fine-tuned OpenVLA 当 L-Sys2:**51.2%**

按理说 fine-tuned 应该更适配 task,结果反而差 4.4 个点。

**人话解释**:Fine-tuning 让 OpenVLA 变成了 "RoboCasa 专项 motor expert",它的 latent feature 开始 encode "RoboCasa 里 robot 的具体动作轨迹",而不是 "任务的 semantic 含义"。但是 z_i 需要的是后者,不是前者 —— motor 部分由 S-Sys1 学就够了。

这就好比 CEO 不需要懂车间里每个螺丝怎么拧,他只需要懂战略。如果你逼 CEO 去拧螺丝拧了一个月,他反而忘了战略该怎么做。

这个 finding 对未来 VLA 训练有 implication:**别去 fine-tune 你的 VLM backbone,把它冻住,只训小 policy**。这样既省钱(不 backprop 7B params)又效果好。

BERT fine-tuning 伤害 general representation 的类似发现: https://arxiv.org/abs/2005.00361

### Latent Feature 从哪抽?Ablation 的人话

Table 3 里 4 个位置抽 z_i:

1. **Mean-of-Text**(prefill 阶段,把 input text 的 token embedding 平均):53.3%
2. **End-of-Text**(prefill 阶段,最后一个 text token 的 hidden state):49.0%
3. **Start-of-Action**(decoding 阶段,第一个 action token 开始生成时的 hidden state):**54.3%**
4. **End-of-Action**(decoding 阶段,gripper action 前的 hidden state):52.0%

**为什么 decoding > prefill?** Prefill 阶段 model 在 "消化 input",hidden state 主要 encode "input 是什么"。Decoding 阶段 model 已经想清楚要输出什么了,hidden state 包含 "我要做什么"。我们要传给 S-Sys1 的是后者(intention),不是前者(perception)。

**人话**:你问一个人"明天去哪",他在 "听问题" 时的脑波(prefill)和 "回答前一天决定好了要去哪" 时的脑波(decoding)是不同的。后者才是真正的 intention。

LLM probing 的相关研究: https://arxiv.org/abs/1909.04766

## 我会怎么 extend 这个工作

### 1. z_i 不该是 static vector,应该是 sequence

当前 z_i 是一个 fixed 4096-dim vector,整个 episode 都用它。但 long-horizon task 比如 "煮咖啡然后关微波炉",intention 应该 evolve。

我会让 L-Sys2 输出一个 **token sequence** $z_i^1, z_i^2, ..., z_i^k$(每个对应一个 sub-goal),S-Sys1 用 cross-attention 在不同 phase 关注不同 token。这就像 CEO 不只发一个 memo,而是发一个 Gantt chart。

### 2. L-Sys2 应该被 S-Sys1 trigger,不是被 instruction trigger

当前架构是 instruction change 才 fire L-Sys2。但如果 task 中途 object 被人挪走了呢?S-Sys1 应该能 detect "execution error 超阈值" 然后 trigger L-Sys2 重新 reasoning。这是 active inference / predictive coding 的思路。

### 3. z_i 应该是 structured,不应该是 dense vector

Dense vector 没法 inspect、没法 debug、没法 human override。如果 z_i 是 JSON-like 的 structured plan,比如:
```json
{"target_object": "mug", "target_location": "microwave", "subgoals": [...]}
```
那 human 可以 monitor、edit,这对 robot safety 很关键。

### 4. 三层 hierarchy

未来应该是 L-Sys3(秒级 task planning)→ L-Sys2(100ms 级 sub-goal reasoning)→ S-Sys1(10ms 级 motor control)。像 cortical hierarchy 那样,每层不同 timescale。

Hierarchical RL 的 options framework: https://papers.nips.cc/paper/1999/hash/6388d4ecb918c1b7d8770d4f3c10e9eb-Abstract.html

### 5. Test-time compute scaling

类似 OpenAI o1,L-Sys2 可以生成多个 candidate z_i,S-Sys1 用一个 verifier 函数选最好的。这就是把 test-time scaling 引入 robot control。

o1 介绍: https://openai.com/index/learning-to-reason-with-llms/

## 类比 SlowFast Networks(Feichtenhofer)

Video understanding 里的 SlowFast Networks(2018)用两个 pathway:
- Slow pathway:低 frame rate,抽 semantic info
- Fast pathway:高 frame rate,抽 motion info

这和 DP-VLA 是 structural isomorphism。但 SlowFast 两个 pathway 处理同一个 task(分类),DP-VLA 两个 model 处理不同 task(reasoning vs control)。

如果 DP-VLA 借鉴 SlowFast 的 fast-to-slow feedback(让 S-Sys1 的 state 反馈给 L-Sys2 决定何时重新 reason),会更强。

SlowFast: https://arxiv.org/abs/1812.03982

## 一个有意思的细节:L-Sys2 只看 v_0

Paper 里说 L-Sys2 只吃 episode 第一帧 image $v_0$,不吃 $v_t$。这是个 strong assumption —— "visual information remains largely consistent"。

在 RoboCasa kitchen task 里这 OK,因为 object 在第一帧就在视野里。但 real-world 里经常 break:
- Task 是 "pick up the cup that someone is handing to you",杯子在中间帧才出现
- Task 中途 object 被挪走

这其实就是 L-Sys2 只在 instruction change 时 fire 这件事的必然推论 —— 既然不重新 fire,那看的 image 也没必要 update。这是 design choice 的 tradeoff。

未来应该让 L-Sys2 在 key frame(比如 hand 接触 object 的瞬间)重新 fire,看当时的 image 重新生成 z_i。这就是我前面说的 dynamic scheduling。

## 最最核心 insight

我觉得这篇 paper 最值得记住的就一件事:

> **LLM/VLM 的 reasoning 是 expensive 的,但 reasoning 的 output(intention)是 stable 的;motor control 是 cheap 的,但需要 high frequency。把两者 decouple 到不同 frequency,用 latent vector 做 communication channel。**

这个 insight 可以迁移到很多地方:
- **Self-driving**:high-level route planning(秒级)+ low-level steering control(毫秒级)
- **Game AI**:strategy reasoning(秒级)+ APM execution(毫秒级)
- **Dialogue agent**:intent understanding(每轮一次)+ response generation(token 级)

本质上就是 **cognitive hierarchy** 在 neural network 时代的重新表达。这种 decoupling 几乎是任何 real-time intelligent system 都需要的 —— 不然你要么慢(monolithic 大模型),要么傻(pure 小模型)。

Sutton & Precup 的 options framework 1999 年就提了类似 idea,只是那时候没有 VLM 来做 high-level reasoning。现在有了 LLM/VLM,这个 idea 终于可以 scale。

参考: https://papers.nips.cc/paper/1999/hash/6388d4ecb918c1b7d8770d4f3c10e9eb-Abstract.html

## 最后一句

DP-VLA 是一个用 cognitive science intuition 包装的工程 trick。它的核心 contribution 不是新 model、新 loss,而是一个 **架构 insight**:VLA 不该是 monolithic 的,应该 hierarchical。System 1 和 System 2 在 brain 里是 separate 的 module,在 robot 里也应该是。

如果你只能记一个 takeaway:把 expensive reasoning amortize 到 cheap control 上,用 latent vector 当 bridge。

---

**Reference Links 汇总:**
- OpenVLA: https://openvla.github.io/
- RoboCasa: https://robocasa.org/
- RT-2: https://robotics-transformer2.github.io/
- Kahneman's book: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- BC-Transformer (Mandlekar): https://arxiv.org/abs/2110.03116
- SlowFast Networks: https://arxiv.org/abs/1812.03982
- Options framework: https://papers.nips.cc/paper/1999/hash/6388d4ecb918c1b7d8770d4f3c10e9eb-Abstract.html
- BERT fine-tuning hurts probing: https://arxiv.org/abs/2005.00361
- LLM probing: https://arxiv.org/abs/1909.04766
- ALOHA / ACT: https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OpenAI o1: https://openai.com/index/learning-to-reason-with-llms/
- Bengio's consciousness prior: https://arxiv.org/abs/1709.08568

---

# Dual Process VLA 论文深度解析

## 1. 核心思想的 Intuition

这篇 paper 的核心 insight 其实非常 elegant —— 它把 Kahneman 在《Thinking, Fast and Slow》里提出的 dual-process theory 直接 mapping 到 robotic control 的架构设计上。人类的 brain 有两套系统:System 1 是 fast、automatic、intuitive 的(对应 limbic system),System 2 是 slow、deliberate、analytical 的(对应 prefrontal cortex)。在 robot 里,这正好对应两种 compute pattern:

- **L-Sys2 (Large System 2)**:运行 VLM/VLA(如 OpenVLA,7B params),负责 high-level reasoning,但只在新 instruction 进来时跑一次
- **S-Sys1 (Small System 1)**:运行 small policy(如 BC-Transformer),负责 real-time motor control,持续在每个 time step 跑

这个设计的关键 insight 是:**reasoning 和 control 的 timescale 是 decoupled 的**。一个 instruction 一旦确定,robot 在执行过程中并不需要每一帧都重新 "理解" instruction —— instruction 的 semantics 是 stationary 的,只有 low-level observation 是 dynamic 的。所以 L-Sys2 只需要在 instruction change 的时候 fire 一次,把 "我要做什么" 压缩成一个 latent vector z_i,然后 S-Sys1 反复 query 这个 z_i 来 guide 连续的 action 输出。

这让我想起 hierarchical RL 里的 **options framework**(Sutton, Precup, Singh 1999)—— 一个 option = (initiation set, policy, termination),这里 L-Sys2 输出的 z_i 就相当于 option 的 identity,而 S-Sys1 就是 option 内部的 intra-option policy。区别在于这里 z_i 是连续的 semantic embedding,不是 discrete option id。

参考链接:
- Kahneman's book: https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow
- Options framework: https://papers.nips.cc/paper/1999/hash/6388d4ecb918c1b7d8770d4f3c10e9eb-Abstract.html
- Bengio's consciousness prior: https://arxiv.org/abs/1709.08568

## 2. 架构的 Math Formalization

### 2.1 Notation 解释

论文里的符号定义值得仔细拆解:

**公式 (1)**:
$$\mathbf{v} = \{v_0, v_1, ..., v_{n-1}\}$$

- $\mathbf{v}$:一个 episode 内的 image sequence
- $v_0$:该 episode 的第 0 帧(初始帧)image
- $v_t$:第 t 帧 image,$t \in [0, n-1]$
- $n$:episode 的总帧数
- 这里下标 $t$ 是 time step index,从 0 开始 indexing

**公式 (2)**:
$$\mathbf{z_i} = f_\ell(v_0, \ell_i)$$

- $f_\ell$:L-Sys2 这个函数(function),由 VLM/VLA 实现
- $v_0$:只用初始帧 image(关键!不是 sequence)
- $\ell_i$:第 i 条 language instruction(text token sequence)
- $\mathbf{z_i}$:4096-dim 的 latent feature vector,是 L-Sys2 编码后的 "intention embedding"
- 下标 $i$ 是 instruction 的 index(因为一个 episode 可能跨越多个 instruction)

**公式 (3)**:
$$\mathbf{a_t} = f_s(\mathbf{o_t}, \mathbf{s_t}, \mathbf{z_i}), \quad t = 0, 1, ..., (n-1)$$

- $f_s$:S-Sys1 这个 function,由 BC-Transformer 实现
- $\mathbf{o_t}$:observations at time $t$,这里是用 ResNet18 encode 的 multi-view images(左相机、右相机、eye-in-hand 三视角,每张 128×128)
- $\mathbf{s_t}$:robot states at time $t$,包括 base-to-eef position (3-dim)、base-to-eef quaternion (4-dim)、base position (3-dim)、base quaternion (4-dim)、gripper qpos (2-dim)
- $\mathbf{z_i}$:从 L-Sys2 传过来的固定 latent feature(instruction 不变就一直不变)
- $\mathbf{a_t}$:7-dim action(6-dim end-effector + 1-dim gripper command)
- 注意 $t$ 的范围:$0$ 到 $n-1$,意味着 S-Sys1 跑满整个 episode,而 L-Sys2 只在 $t=0$ 时 fire 一次

### 2.2 Information Flow 的关键点

这里有一个我特别想 highlight 的设计选择:**L-Sys2 只吃 $v_0$,不吃 $v_t$**。这意味着作者做了一个 strong assumption —— "visual information remains largely consistent from the receipt of new instructions to the completion of their execution"。这个 assumption 在 RoboCasa 的 kitchen task 上 hold 得比较好,因为 task object 的位置在初始帧就能看到,后续主要是 robot 移动。

但这在更 general 的场景里会 break。比如 task 是 "pick up the cup that someone is about to hand you",或者 task 中途 object 被移动了,L-Sys2 就需要 re-fire。作者在 Future Work 里提到了 dynamic scheduling,但当前版本是 instruction-change-triggered 的。

## 3. 与 Related Work 的深度对比

### 3.1 vs RT-2 (Google DeepMind, 2023)

RT-2 把 VLM 直接 fine-tune 成 VLA,把 action tokenize 成 text token,整个 pipeline 是 monolithic 的。55B model 跑 1-3 Hz,5B model 跑 ~5 Hz。问题在于每一帧都要跑完整的 VLM forward pass,即使 instruction 没变。这就像每秒钟都重新做一次 SAT 题目,而不是做完一次后照着计划执行。

DP-VLA 的 L-Sys2 用的 OpenVLA 也是这种 monolithic 风格的模型,但通过 hierarchical 分离,把 OpenVLA 从 "every-frame execution" 解放到 "once-per-instruction execution",瞬间释放了 99% 以上的 compute(假设一个 episode 200 帧,instruction 1 个,那 L-Sys2 跑 1 次 vs 200 次)。

RT-2 paper: https://arxiv.org/abs/2307.15818
OpenVLA paper: https://arxiv.org/abs/2406.09246

### 3.2 vs ALOHA / Mobile ALOHA (Stanford, 2024)

ALOHA 是 bimanual teleoperation + ACT(Action Chunking Transformer)的 pipeline,跑 ~50 Hz,但 lacks general reasoning。它就像一个非常熟练的 System 1 —— 快但 shallow。DP-VLA 的 S-Sys1 就是 BC-Transformer(和 ACT 的 spirit 接近),但通过 z_i 注入了 System 2 的 "understanding",弥补了 pure BC 的 generalization 短板。

ALOHA: https://mobile-aloha.github.io/
Mobile ALOHA: https://arxiv.org/abs/2401.02117

### 3.3 vs Diffusion Policy

Diffusion Policy 用 diffusion process 生成 multi-modal action distribution,擅长处理 ambiguous scenarios。但它没有 language conditioning 的 high-level reasoning,通常需要额外的 task embedding。DP-VLA 的 z_i 可以看成是给 S-Sys1 提供的 "task embedding",但来源是 powerful 的 VLM,而不是 learned task embedding。这是一个有趣的对比 —— Diffusion Policy 强在 action distribution 的 multi-modality,DP-VLA 强在 task understanding 的 generality。

Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 3.4 vs MoE (Mixture of Experts)

表面看 DP-VLA 有点像 MoE —— 不同 expert 处理不同任务。但本质区别在于:MoE 是同一个 abstraction level 上的 routing,而 DP-VLA 是 **不同 abstraction level 上的 decoupling**。L-Sys2 处理 "what to do"(semantic level),S-Sys1 处理 "how to do"(motor level)。这更像是 cognitive architecture 里的 sub-symbolic vs symbolic 分离,或者像 LangGraph 里的 planner-executor pattern。

## 4. Latent Feature Extraction 的 Ablation 深挖

Table 3 的 ablation 我觉得是这篇 paper 最有意思的部分。作者从 OpenVLA 里抽了 4 种 latent feature 喂给 S-Sys1:

| Feature Type | Stage | Avg Success Rate |
|---|---|---|
| Mean-of-Text | Prefill | 0.533 |
| End-of-Text | Prefill | 0.490 |
| Start-of-Action | Decoding | **0.543** |
| End-of-Action | Decoding | 0.520 |

**Prefill stage**:指 LLM 处理 input prompt 时的 forward pass,这一阶段每个 token 都在 "被理解",但 output token 还没生成。

**Decoding stage**:指 LLM 自回归生成 output token 的阶段,此时 hidden state 已经包含了 "我要开始 output 什么" 的信息。

**为什么 Decoding > Prefill?** 作者的解释是 prefill 阶段对 final output 的影响是 indirect 的,而 decoding 阶段的 hidden state 直接决定 output。这里我想再 build 一下 intuition:

- Prefill 阶段,hidden state 主要 encode "input 是什么"
- Decoding 阶段,hidden state 已经经过了 cross-attention(或 self-attention over full sequence)整合了 input 和 "即将 output" 的信息,所以更接近 "intention-to-act"

这其实和 LLM 里做 representation probing 的发现一致 —— 越靠近 output 的 layer 越能 predict 下一步行为,而 input embedding 附近的 layer 主要 encode surface form。

**为什么 Start-of-Action > End-of-Action?** 这个我没完全想明白,但猜测:Start-of-Action 是 x-axis movement 的第一个 token,可能更 "global",而 End-of-Action 是 gripper 相关的,更 "local"。z_i 是要传给整个 episode 用的,所以 "global direction" 比 "local gripper action" 更有信息量。

LLM probing 经典论文: https://arxiv.org/abs/1909.04766 (Tenney et al., BERTology)

## 5. Pre-trained vs Fine-tuned 的反直觉发现

Table 4 是另一个反直觉的结果:

| L-Sys2 | Avg Success Rate |
|---|---|
| OpenVLA-pt (pretrained) | **0.556** |
| OpenVLA-ft (fine-tuned on RoboCasa) | 0.512 |

按理说 fine-tuned 应该更好,但实际是 pretrained 的 latent feature 更 useful。作者的 hypothesis:fine-tuning 让 OpenVLA over-specialize 到 RoboCasa 的特定 action distribution,丢失了 general task understanding,而 z_i 需要的恰恰是 "task-level semantics",不是 "robot-specific motor knowledge"。motor 的部分由 S-Sys1 学就够了。

这让我想起 NLP 里的经典发现 —— BERT fine-tune 后再做 probing 反而不如 fine-tune 前,因为 task-specific 的 fine-tuning "narrows" representation。这里是一样的 story:L-Sys2 应该 stay as a general reasoner,S-Sys1 才是 task-specific learner。这其实和 LoRA 的哲学有点像 —— 把 task-specific 的 adaptation 隔离出来,不动 base model 的 general representation。

这个发现对未来的 VLA 训练有 implication:**不要过度 fine-tune VLM backbone**,把它冻住,只训 small policy 即可。这能大大降低 training cost(7B VLM vs 几 M 的 policy network)。

BERT probing after fine-tuning: https://arxiv.org/abs/2005.00361 (Mixout paper)

## 6. RoboCasa 实验数据深度阅读

### 6.1 Task-level 分析

仔细看 Table 1,可以发现 DP-VLA 在哪些 task 上 gain 最大:

**PnP (Pick and Place)**:这是 OpenVLA-ft 几乎完全失败的类别(大多 0.00)。BC-xfmr 一般,DP-VLA 大幅提升。比如 PnPSinkToCounter 从 0.42 → 0.56,PnPStoveToCounter 从 0.28 → 0.62。说明 DP-VLA 在需要 precision + understanding 的 task 上最强 —— pure BC 理解不够,pure VLA precision 不够,组合最好。

**Open/Close Doors**:OpenDoubleDoor 从 0.48 → 0.80,CloseDoubleDoor 从 0.46 → 0.84。CloseSingleDoor 从 0.94 → 1.00(已经饱和)。这组 task 的 gain 也很大,说明 z_i 能区分 "single vs double"、"open vs close" 这种 semantic distinction,而 BC-xfmr 单独 struggle。

**Twisting Knobs / Turning Levers / Pressing Buttons**:gain 相对小,因为这些 task 的 action pattern 比较固定,BC-xfmr 已经能学到。

**Insertion (CoffeeSetupMug 等)**:CoffeeSetupMug 从 0.12 → 0.30,gain 巨大。这类 task 需要精细的 multi-step reasoning,正好是 L-Sys2 的强项。

整体 avg:OpenVLA-ft 0.058、BC-xfmr 0.426、DP-VLA 0.529。**20.4% relative improvement over BC-xfmr,~9x over OpenVLA-ft**。

RoboCasa: https://robocasa.org/
MimicGen: https://arxiv.org/abs/2310.10996

### 6.2 Speed 分析

Table 2:
- OpenVLA-ft: 0.253 sec (~4 Hz)
- BC-xfmr: 0.022 sec (~45 Hz)
- DP-VLA: 0.030 sec (~33 Hz)

DP-VLA 比 BC-xfmr 慢 8ms,但比 OpenVLA-ft 快 ~8.4x。这 8ms 主要来自 S-Sys1 需要 encode 额外的 z_i(通过 MLP layer)。L-Sys2 的开销虽然在第一个 frame 比较大,但 amortize 到 50 帧后就 dilute 了。

注意这里有个细节:**Table 2 是 1-50 frame 的 average**,所以 L-Sys2 的一次性 cost 被 dilute 到 50 帧。如果 episode 更长(比如 500 帧),DP-VLA 的 average inference time 会更接近 BC-xfmr。这说明 DP-VLA 的优势在 long-horizon task 上更明显。

## 7. 与更广 AI Research 的联想

### 7.1 与 LLM 的 KV Cache 类比

DP-VLA 的 z_i 其实很像 LLM 的 KV cache —— 都是 "已经计算过的 context 的压缩表示",可以反复 query。z_i 是 instruction + initial scene 的 KV cache,S-Sys1 在每个 time step 去 "read" 它。这个类比让我想到:

- 如果 z_i 是 instruction 的 KV cache,那 L-Sys2 的 forward pass 就是 prefill 阶段
- S-Sys1 的 forward pass 类似 decoding,但不是 generate token,而是 generate action
- 那是不是可以用 cross-attention 让 S-Sys1 attend to z_i 的不同维度,而不是简单的 MLP concat?

这个方向我觉得很有潜力。Paper 里只是把 z_i 通过 MLP encode 然后 concat 到 S-Sys1 的 input,这相当于 "static injection"。如果改成 attention,可能是 dynamic query,让 S-Sys1 在不同 phase 关注 z_i 的不同部分。

KV Cache 介绍: https://peterchhang.github.io/2023/12/05/Computer-Science/kvcache/

### 7.2 与 Chain-of-Thought 的关系

作者在 Future Work 提到 "improving contextual understanding through Chain of Thought reasoning"。这里有一个微妙的设计问题:CoT 通常需要多步 reasoning,但如果 L-Sys2 只在 instruction 进来时 fire 一次,那 CoT 怎么和 S-Sys1 的执行 interleave?

可能的方案:
- L-Sys2 内部做 multi-step CoT,生成一个 z_i sequence(z_i^1, z_i^2, ..., z_i^k),S-Sys1 在不同阶段 query 不同的 z_i^j
- 或者 L-Sys2 在每个 sub-goal 完成时被 trigger 重新 reasoning(类似 inner monologue)

CoT paper: https://arxiv.org/abs/2201.11903
Inner Monologue: https://innermonologue.github.io/

### 7.3 与 Predictive Coding 的联系

从 neuroscience 角度,predictive coding theory 认为 brain 持续生成 prediction 并和 sensory input 比较,只处理 residual。DP-VLA 的架构和这个 framework 有结构相似性:

- L-Sys2 生成 "high-level prediction"(z_i,即 "我要做什么")
- S-Sys1 持续执行并和实际 observation 对齐(类似 "lower-level error correction")

如果引入 prediction error feedback(比如 S-Sys1 检测到 execution 偏离了 z_i 的 intention 就 trigger L-Sys2 重新 reasoning),就能实现 Future Work 里提到的 dynamic scheduling。这就像 active inference 在 robotic control 里的应用。

Active Inference: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(13)00069-0

### 7.4 与 OpenAI o1 / Strawberry 的类比

最近 LLM 圈里 System 2 的概念很火,o1 通过 RL 让 model 学会 "think before answer"。DP-VLA 其实是 robot 版的 o1 philosophy —— 让 model 在 instruction 来时 think 一次(L-Sys2),然后 fast execute(S-Sys1)。差别在于 o1 是 in-model reasoning,DP-VLA 是 cross-model reasoning(两个不同 model 协作)。

这个 cross-model 思路有一个优势:o1 的 thinking 是 latent 的,但 DP-VLA 的 thinking 输出 z_i 可以被 inspect、debug、甚至 human-edit。这对 robot safety 很重要 —— 你可以监控 L-Sys2 输出的 z_i 是否合理,而不需要去 probe 一个 monolithic model 的内部 state。

o1 blog: https://openai.com/index/learning-to-reason-with-llms/

### 7.5 与 Slowfast Networks (Feichtenhofer et al.) 的类比

Video understanding 里的 SlowFast Networks 用两个 pathway:Slow pathway 跑低 frame rate 抽 semantic info,Fast pathway 跑高 frame rate 抽 motion info。这和 DP-VLA 的 L-Sys2 / S-Sys1 几乎是 structural isomorphism:

- Slow pathway ↔ L-Sys2(low frequency,semantic)
- Fast pathway ↔ S-Sys1(high frequency,motor)

差别在于 SlowFast 是同一个 task 的 two views,DP-VLA 是 two different tasks(reasoning vs control)。这个类比启发:video 里 fast pathway 可以 inform slow pathway(detail 补充),那 DP-VLA 里 S-Sys1 是不是也可以 feedback 给 L-Sys2(比如 S-Sys1 detected unexpected state,trigger L-Sys2 重新 reason)?

SlowFast: https://arxiv.org/abs/1812.03982

## 8. 局限性与未来方向

### 8.1 当前 Limitations

1. **Static scheduling**:L-Sys2 只在 instruction change 时 fire。中途环境大变(比如 object 被推走)时 L-Sys2 不会 re-fire,S-Sys1 只能"硬扛"。
2. **Single-shot z_i**:z_i 是一次性生成的 fixed vector,不能根据 S-Sys1 的 execution progress 调整。比如 task 是 "pour water then close lid",z_i 应该在 pour 完成后变化,但当前架构做不到。
3. **Vision bottleneck**:L-Sys2 只吃 $v_0$,如果初始帧信息不全(比如 object 在视野外),z_i 就缺失关键信息。
4. **Sim-only eval**:RoboCasa 是 MuJoCo simulation,real-world transfer 还没验证。

### 8.2 Future Architectures 我会想的

如果让我设计下一代 DP-VLA,我会考虑:

1. **Cross-attention 而非 MLP concat**:让 S-Sys1 用 cross-attention query z_i,而不是把 z_i 当 fixed feature。这样 z_i 可以是 sequence(比如一个 "plan" 的 token sequence),S-Sys1 在不同 phase 关注不同 token。

2. **Error-driven L-Sys2 triggering**:S-Sys1 输出一个 confidence score(或 prediction error),如果超出阈值就 trigger L-Sys2 重新 reason。这是 active inference 风格。

3. **Multi-level hierarchy**:不止两层,而是 L-Sys3 → L-Sys2 → S-Sys1 三层,每层不同 frequency。比如 L-Sys3 做 task planning(秒级),L-Sys2 做 sub-goal reasoning(100ms 级),S-Sys1 做 motor control(10ms 级)。

4. **z_i 作为 "communication protocol"**:让 z_i 是 structured(比如 JSON-like)而不是 dense vector,这样可以 human-inspectable,甚至可以让 human 编辑 z_i 来纠正 robot 的 plan。

5. **Test-time compute scaling**:类似 o1,在 instruction 进来时让 L-Sys2 多生成几个 candidate z_i,然后 S-Sys1 用 verification 函数选最好的。这是把 test-time scaling 引入 robot control 的方式。

## 9. Implementation 细节补充

Paper 里一些 implementation detail 我觉得值得注意:

- **Image encoder**:S-Sys1 用 ResNet18 encode 每张图(128×128 × 3 views = 3 个 ResNet18 instances 或 shared weights,paper 没明说)。ResNet18 是 ~11M params,比 OpenVLA 用的 SigLIP-Patch16/384(~400M params)小很多,这也是为什么 S-Sys1 快。
- **Language encoder for BC-Transformer**:用 CLIP text encoder,但只在第一个 frame encode 一次。这和 L-Sys2 的 role 有点 overlap,可以理解为 BC-xfmr baseline 也有一个 "miniature System 2",只不过用的是 CLIP 而非 OpenVLA。
- **Loss**:L1 + L2 loss(combined)用于 action regression。L1 对 outlier robust,L2 对 small error smooth,这种组合在 robotic control 里挺常见。
- **Training iterations**:DP-VLA 用了 350K iterations,batch size 128。BC-xfmr baseline 也是同样规模 training。L-Sys2 是 frozen 的(用 pre-trained 或 fine-tuned OpenVLA),不参与 S-Sys1 的 training,这点很重要 —— 训练成本基本和 BC-xfmr 一样,不需要 backprop through 7B VLM。

这点其实是 DP-VLA 的一个 hidden advantage:**training cost ≈ BC-Transformer training cost**,因为 L-Sys2 frozen。而 OpenVLA-ft 要 fine-tune 整个 7B model,显存和时间成本都高。所以 DP-VLA 不仅 inference 快,training 也便宜。

ResNet: https://arxiv.org/abs/1512.03385
CLIP: https://arxiv.org/abs/2103.00020

## 10. 一句话总结 Intuition

这篇 paper 的 magic 在于:**把 LLM-style 的 "thinking" 和 RL-style 的 "acting" 解耦到两个 timescale 上,用 latent vector z_i 当作两者的 communication channel**。L-Sys2 在 "认知尺度" 上工作(seconds),S-Sys1 在 "运动尺度" 上工作(milli-seconds),z_i 把抽象 intention 压缩成 concrete vector,实现了 amortized reasoning —— 一次思考,反复利用。

这个 idea 本质上是把 LLM inference 里的 **prefill vs decode** 分离重新映射到 robot control 上:L-Sys2 做 prefill(reasoning + KV cache 生成 z_i),S-Sys1 做 decode(用 z_i 反复 generate action)。如果未来 robot 要做 long-horizon task,这种 decoupling 几乎是必须的 —— 否则你要让 7B VLM 在 10Hz 下跑几百帧,既慢又 expensive。

DP-VLA 是迈向 cognitive robot 的一小步,但 idea 很 sound。后续如果能解决 dynamic scheduling、z_i 的 temporal evolution、和 real-world transfer,会是一个很有 impact 的方向。

---

**主要 Reference Links:**
- Paper PDF (推测, ETRI 风格的 arXiv): https://arxiv.org/abs/2502.02125 (推断可能在这附近,作者 ByungOk Han ETRI)
- OpenVLA: https://openvla.github.io/
- RoboCasa: https://robocasa.org/
- MimicGen: https://mimicgen.github.io/
- Kahneman's book: https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow
- BC-Transformer (Mandlekar et al.): https://arxiv.org/abs/2110.03116
- ALOHA / ACT: https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Octo: https://octo-model.github.io/
- RT-2: https://robotics-transformer2.github.io/
