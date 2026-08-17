---
source_pdf: DFM-VLA Iterative Action Refinement for.pdf
paper_sha256: 92e3057c9a7c926212255b3ec01e14c68ab4d895436800bcca17b4888e12f461
processed_at: '2026-08-03T21:01:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DFM-VLA

好，我换个讲法，咱们像在咖啡店白板上聊一样，把这篇 paper 拆开看看。

## 1. 这群人到底想解决什么痛点？

想象你在教一个机械臂做任务。模型需要输出一串动作 token（比如 16 个或 32 个 token 组成一个 action chunk），这些 token 决定机械臂接下来几百毫秒怎么动。

现在有三种主流 decode 方式，每种都有自己的毛病：

**第一种：Autoregressive (AR)**
就像你写作文的时候，一个字一个字往下写，写完"今"就不能改成"明"。OpenVLA、π0-FAST 就是这种。问题在于：万一第 3 个 token 写错了，后面的 token 全部基于这个错误继续生成，错误就像滚雪球。

**第二种：Discrete Diffusion (DD)**
像做填空题。一次性把所有空都摆出来，然后并行填。已经填好的格子就锁死，剩下的格子继续填。看起来很 parallel、很高效，但本质问题没解决——一旦某个格子被填了，它就被冻住了。Dream-VLA、dVLA 是这种。

**第三种：DFM-VLA（本文）**
他们提了一个新范式：**先打一个粗略的草稿，然后反复修改整个 chunk**。今天改改第 5 个 token，明天又回头改第 2 个 token，改到满意为止。这就像我们写文章真正的工作流——先 draft，然后 revise，revise，再 revise。

这种 "可逆 refinement" 能力，是 AR 和 DD 都没有的。Paper 里管这个叫 **irreversible commitment** 问题。他们把它命名为一个 contribution，我觉得挺准确的。

## 2. 为什么 robotics 比 NLP 更需要这个能力？

你可能会问：discrete diffusion 在 NLP 里也有人做啊（MDLM, SEDD, D3PM, LLADA），为什么 NLP 没那么强调 "可逆"，robotics 却必须强调？

我的理解是：

NLP 的 token 之间相对 "local"。你写错一个 "the"，文章还能读下去。但 action chunk 不一样——机械臂的 16 个 token 是一串连续动作，第 3 个 token 偏 1 厘米，后面 13 个 token 全部要跟着调整。token 之间是强耦合的。

CALVIN 这个 benchmark 特别能体现这点——它要求模型连续完成 5 个 sub-task，前一个 sub-task 失败，后面全部失败。所以 "错误累积" 在 robotics 里是放大效应，比 NLP 严重得多。

这就是为什么 paper 里 DFM-VLA 在 CALVIN 长程任务上比 AR 高 9 个点（5-step completion: 0.780 vs 0.690）。这 9 个点不是随便来的，是 "修正能力" 直接换来的。

## 3. 数学怎么"接地气"地理解？

Discrete Flow Matching 的数学乍看吓人，其实核心 idea 很朴素。我给你三个比喻串起来：

### 比喻一：天气演化

CTMC（Continuous-Time Markov Chain）就像天气系统。每个 token 是一个 "状态"（比如晴天、阴天、雨天），velocity field $u_t$ 就是天气之间的转移速率——"从晴转阴的概率速率"。

关键区别：CTMC 的状态可以反复跳。今天晴转阴，明天阴又转回晴。这跟 AR 的 "一次性决定" 和 DD 的 "一旦揭示就锁死" 都不同。

### 比喻二：水流 downhill

Embedding-guided velocity 的公式：

$$u_t^i(x^i, z | x_1) = p_t(x^i | x_1^i) \dot{\beta}_t [d(z^i, x_1^i) - d(x^i, x_1^i)]_+ \tag{7}$$

这个公式看着吓人，但用 "水流 downhill" 理解就清楚了：
- $x_1^i$ 是山脚下的目标点
- $x^i, z^i$ 是山上的两个位置
- $d(\cdot, x_1^i)$ 是离山脚的距离
- $[d(z^i, x_1^i) - d(x^i, x_1^i)]_+$ 表示只有当 $z$ 比 $x$ 离山脚更远时，水才会从 $z$ 流向 $x$

换句话说，**velocity 永远朝离 target 更近的方向流**。水流不会倒着上坡。这就是 "kinetic-optimal" 的意思——能量最优的路径。

### 比喻三：调焦相机

$\beta_t = c \cdot (t/(1-t))^\alpha$ 这个 scheduler 控制 "聚焦速度"。

- $t = 0$ 时 $\beta_0 = 0$：镜头完全失焦，画面糊
- $t = 0.5$ 时 $\beta$ 适中：开始能看清主体
- $t \to 1$ 时 $\beta_t \to \infty$：完全对焦，画面清晰

参数 $c$ 控制整体聚焦速度（粗调），$\alpha$ 控制聚焦曲线的弯曲程度（细调）。实验里 $c=3, \alpha=1$ 最佳，太大会过快锁定，太小会一直模糊。

## 4. 两种 Velocity 设计的 intuition

他们试了两种设计，我觉得这个对比本身就很有教学意义：

### Design A: Auxiliary Velocity Head

模型先算 hidden states $h_t$，再接一个 head 把 hidden states 映射成 velocity：

$$h_t = f_\theta(x_t, l), \quad u_t^\theta(\cdot | x_t) = u_t^{head}(h_t)$$

像什么？像你让一个学生同时学两件事：**(1) 答题**（预测 target action）和 **(2) 解释自己的答题过程**（输出 velocity field）。学生既要给答案，又要描述 "从当前状态到答案的路径"。这两个任务有关联但又不完全一样，模型 capacity 被分散。

### Design B: Embedding-Guided

模型只需要预测 target $x_1$ 的 categorical distribution，velocity 由公式 (7) 解析算出来。

像什么？像你让学生只管答题，"答题到目标的路径" 由教材统一规定（基于 embedding distance）。学生专注一件事，自然学得快。

实验结果（Figure 5）很说话：embedding-guided 在 20k 步就达到 95.7% 成功率，head-based 在 20k 步只有 91% 左右。**Sample efficiency 差了一大截**。

我的直觉：embedding-guided 利用了 FAST tokenizer 已经学好的 action embedding space。这个 embedding space 本身就编码了 "action 之间的语义距离"。把它当作 metric 来定义 flow path，相当于 "免费" 获得了一个 sensible 的 velocity structure。Head-based 想从零学这个 structure，自然更慢。

## 5. 两阶段解码像什么？

总共 16 步 decoding，前 14 步是 stochastic refinement，后 2 步是 greedy validation。

像什么？像画家作画：
- **前 14 步**：用宽笔刷大刀阔斧地探索，有时候大胆尝试新颜色（stochastic jump）
- **后 2 步**：换细笔刷稳定下来，确定最终细节（greedy decoding）

为什么需要后 2 步 greedy？因为纯 stochastic 到最后会有抖动——token 在最后几步还在小幅跳来跳去，输出不稳定。Greedy 强制收敛。

Table 4 的 ablation 很说明问题：

| $T_{fine}$ | $T_{val}$ | LIBERO |
|------------|-----------|--------|
| 16 | 0 | 94.8% |
| 14 | 2 | **95.7%** |
| 8 | 8 | 95.1% |

$T_{val} = 0$ 全 stochastic：稍弱（探索没收敛）
$T_{val} = 8$ 太多 greedy：也弱（探索不足，过早 commit）
$T_{val} = 2$：甜点

这跟 RL 里 epsilon-greedy schedule 一个道理——探索和利用的分配很关键。

## 6. Adaptive KV Cache 为什么这么有效？

Table 5 里 DFM + Adaptive Cache 把推理速度从 60.2 提到 121.0，几乎翻倍，性能还几乎不掉（4.42 → 4.40）。

为什么 DFM 能享受这个红利，AR 不能？

AR 的解码过程：每生成一个新 token，KV cache 必须扩展。前面的 token 的 K, V 一旦固定就固定，但新 token 的 attention 要看所有前面的 token。所以 KV cache 是 "append-only"，不能复用。

DFM 的解码过程：所有 token 一开始就在，每步只是状态在变。early iteration 大部分 token 已经接近 final 状态，变化很小。所以可以监测 "value features 的 cosine similarity"，只更新变化大的位置的 KV。

打个比方：AR 像写长篇小说，每加一段都要重新审视前面所有段落（必须重算 attention）。DFM 像修改一篇文章，大部分段落没动，只改了几个字，其他段落的 "上下文影响" 不变。

这个工程 trick 我觉得未来在 dLLM 上会非常重要。FAST-dLLM 已经在 LLM 上证明了这点，DFM-VLA 把它移植到 VLA，效果同样显著。

## 7. 实验数据背后的 intuition

### CALVIN（长程任务）

DFM-VLA+Embed 拿到 4.44 avg length（满分 5），比 UniVLA* 高 0.18，比 ReconVLA 高 0.19。

5-step completion 上 DFM-VLA 是 0.780，UniVLA* 是 0.690。差 9 个点，这个 gap 在 CALVIN 上是非常显著的。CALVIN 5-step completion 是最难指标——它要求模型连续完成 5 个 sub-task 不掉链子，错误一累积就崩。

DFM-VLA 能在这上面胜出，说明 "可逆 refinement" 确实在抑制错误累积。

### LIBERO（跨任务泛化）

DFM-VLA+Embed 平均 95.7%，比 Dream-VLA 高 3.1 个点，比 FlowVLA 高 7.6 个点。

最亮眼的是 Long suite：92.6%，比 π0-FAST 高 32 个点（60.2%）。Long suite 是 multi-stage compositional reasoning，需要长程规划。

我的直觉：AR 在 Long 上崩盘是因为 chunk 之间的衔接错误累积。DFM-VLA 通过 chunk 内部的 iterative refinement，让每个 chunk 自身更鲁棒，间接帮助 chunk 之间衔接。

### Real-World

三个任务，DFM-VLA 平均 70.8%，比 RDT 高 10.8 个点。

Real-world 比 simulation 更难——视觉噪声大、动力学不精确、控制误差累积。这些场景下 "修正能力" 价值最高。RDT 是 continuous diffusion，理论上也支持 refinement，但它在 continuous action space 操作，失去了 discrete VLA 的 LLM 兼容性。DFM-VLA 既拿到 refinement 又保留 discrete token 的优势。

### 数据规模（Table 6）

| Data | AR | DD | DFM |
|------|-----|-----|-----|
| 10% | 1.71 | 2.84 | 3.21 |
| 100% | 4.18 | 4.32 | 4.44 |

10% 数据时 DFM 比 AR 几乎翻倍（1.71 → 3.21）。低数据 regime 下 refinement 价值最大——单次 prediction 错误率高，能修正就赚大了。

这个发现很有现实意义。真实机器人数据采集成本极高，能 "用更少数据训出更好模型" 价值很大。

## 8. 我自己的延伸思考

### 思考 1：DFM 在 NLP 上为什么没这么大优势？

DFM-VLA 在 robotics 上比 AR 高 9 个点。同样的 idea 放到 NLP 上（比如 LLADA、Dream 7B），相对 AR LLM 的优势就没这么明显。

为什么？我前面提过——NLP token 之间相对 local，错一个 token 不致命。Robotics 的 action chunk 强耦合，错一个 token 整个 chunk 报废。

这暗示一个更深的观点：**discrete flow 的价值跟 token 之间的耦合强度成正比**。NLP 弱耦合，DFM 价值有限；robotics 强耦合，DFM 价值很大；那么 code generation、mathematical proof 这种 "结构性强" 的文本呢？我猜测 DFM 也会有显著优势，但还没人系统验证。

### 思考 2：Embedding-Guided 的更广含义

这个 idea 其实可以推广。Embedding-guided velocity 的核心是 "利用预训练 embedding 的 metric structure 来定义 flow path"。

在 VLA 里，embedding 来自 FAST tokenizer。在其他场景：
- **NLP**：可以用 word2vec / BERT embedding 来定义 flow path
- **Protein design**：protein embedding space 有结构意义，可以做 discrete flow
- **Music generation**：music token 的 embedding 有调性结构

任何 "token embedding 有 sensible metric structure" 的场景，embedding-guided flow matching 都可能优于 head-based。这是一个可推广的 design pattern。

### 思考 3：与 Continuous Diffusion VLA 的关系

π0, RDT 这些 continuous diffusion VLA 也支持 iterative refinement，只是在 continuous action space。DFM-VLA 在 discrete space。

哪个更好？Paper 里 DFM-VLA 在 CALVIN 上 4.44，π0-FAST 是 4.18，π0 自己（continuous）大概也在 4.2-4.3。所以 discrete DFM 看起来至少打平 continuous diffusion。

理论上 continuous 更精细（无限分辨率），discrete 更好接入 LLM backbone（保留语义理解）。DFM-VLA 的 real-world 实验显示它比 RDT（continuous diffusion）高 10 个点，这说明 discrete + LLM backbone 的 "语义先验" 比 continuous 的 "精度优势" 更值钱。

我猜测 future 方向是 discrete-continuous hybrid：discrete token 提供 semantic structure，continuous refinement 提供 fine-grained 调整。

### 思考 4：Refinement Step 数应该 Adaptive

现在固定 16 步。但简单任务（"把杯子拿起来"）可能 4 步就够，复杂任务（"做三明治"）可能需要 32 步。

一个 learned controller 决定何时停止——基于 prediction 的 entropy 或 confidence——应该能进一步提速。这跟 early exit 在 LLM 上的应用思路一致。

CEED-VLA 已经用 consistency distillation 减少步数，DFM-VLA 可以借鉴。

### 思考 5：与 CoT-Reaction 的结合

UD-VLA, MM-ACT, dVLA 这些工作把 chain-of-thought 引入 discrete diffusion VLA——让模型先输出 reasoning trace，再输出 action。DFM-VLA 现在只 refine action，没 refine reasoning。

如果用 EditFlow 的 insert/delete/replace 三种 operation，可以 jointly refine reasoning 和 action。模型可以 "边想边改"——reasoning trace 可以根据 action 的 refinement 反向修改。这是个很有想象空间的方向。

### 思考 6：理论分析缺失

Paper 是纯 empirical。没有 theoretical guarantee 说 DFM 比 AR 严格好。

一个可能的分析框架：用 approximation error propagation。AR 的 error 在 token 序列上线性累积（每一步 error 加上上一步 error 的影响）。DFM 的 error 可以被后续 step 修正，所以 error propagation 是 "衰减" 而不是 "累积"。

如果有人能形式化这个直觉，给出 DFM 优于 AR 的充分条件，那 paper 的 theoretical contribution 会强很多。

### 思考 7：跟 Self-Correction LLM 的联系

最近 LLM 圈也有 self-correction / self-refine 的工作（Reflexion, Self-Refine, Constitutional AI）。这些是 LLM 通过 prompt 让自己 review 并修正输出。

DFM-VLA 是 "intrinsic self-correction"——correction 机制 baked into decoding process，不需要额外 prompt。这是个有意思的对比：
- **Extrinsic self-correction**（LLM self-refine）：通过 prompt 让模型显式 review
- **Intrinsic self-correction**（DFM-VLA）：通过 decoding 机制隐式 refine

两者可能互补。DFM-VLA 处理 token-level refinement，self-refine prompt 处理 high-level planning refinement。

### 思考 8： humanoid 机器人上的潜力

Paper 只在 bimanual AgileX 平台上验证，3 个任务，每个 100 条轨迹。数据规模偏小。

Humanoid 机器人（Figure, 1X, Tesla Optimus）的 action space 更复杂——全身控制，几十个 DOF。DFM-VLA 的 refinement 能力在 humanoid 上价值更大，因为 DOF 越多错误累积越严重。

我猜测 NVIDIA GR00T N1、Figure 的 Helix 这些 humanoid VLA 内部可能已经在用类似 idea。DFM-VLA 的开源代码应该会被 humanoid 社区快速吸收。

### 思考 9：训练数据质量的影响

DFM-VLA 的 refinement 假设 "target action 是合理的"。如果训练数据本身有问题（比如某些 demonstration 含错误动作），refinement 朝错误方向 refine，反而更糟。

这意味着 DFM-VLA 对数据质量更敏感——它会把数据里的 noise 也 "refine" 出来。Paper 没讨论这点。Future work 可能要加 data filtering 或 confidence-aware refinement。

### 思考 10：与 Diffusion Models 的 deeper 统一

Continuous diffusion (DDPM, score-based) 和 discrete flow matching 都是 "iterative refinement" 框架。它们的本质区别：
- **Continuous diffusion**：在 $\mathbb{R}^d$ 上用 SDE 做 refinement
- **Discrete flow matching**：在 discrete space 上用 CTMC 做 refinement

数学上，CTMC 是 SDE 的离散版本。所以 DFM-VLA 可以看作 "把 continuous diffusion VLA (π0, RDT) 离散化"。这种 "离散-连续" 对偶在数学上很优雅，在工程上保留了 VLA 的 tokenization 优势。

未来如果有人能 unify 这两个 framework（比如用 score-based discrete diffusion），那会是很漂亮的理论工作。

## 9. Reference 链接

**核心理论**：
- Discrete Flow Matching (Gat et al.): https://arxiv.org/abs/2407.15575
- Kinetic-Optimal Discrete Paths (Shaul et al.): https://arxiv.org/abs/2412.03487
- Edit Flows (Havasi et al.): https://arxiv.org/abs/2506.09018

**VLA Baseline**：
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://openvla.github.io/
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- RDT-1B: https://arxiv.org/abs/2410.07864

**Discrete Diffusion LLM/VLA**：
- Dream 7B: https://arxiv.org/abs/2508.15487
- LLADA-VLA: https://arxiv.org/abs/2509.06932
- dVLA: https://arxiv.org/abs/2509.25681
- Dream-VLA: https://arxiv.org/abs/2512.22615

**Discrete Flow LLM**：
- Fudoki: https://arxiv.org/abs/2505.20147
- OneFlow: https://arxiv.org/abs/2510.03506
- Next-Omni: https://arxiv.org/abs/2510.13721

**KV Cache for Discrete Diffusion**：
- FAST-dLLM: https://arxiv.org/abs/2505.22618

**Benchmark**：
- CALVIN: https://calvinrobot.github.io/
- LIBERO: https://libero-project.github.io/

**Project Page**：
- DFM-VLA: https://chris1220313648.github.io/DFM-VLA/

---

最后说一句我整体的 take：这篇 paper 的核心 contribution 不是 "又一个 VLA"，而是 **给 discrete VLA decoding 范式补上了一个 missing piece**——可逆 refinement。Embedding-guided velocity 是个优雅的工程实现，两阶段解码是实用的工程优化。从更大的视角看，这是 discrete flow matching 从 NLP 走向 action modality 的里程碑，未来在 humanoid、CoT-action joint refinement、discrete-continuous hybrid 等方向都还有很大空间。

如果你想 build intuition，我建议自己手算一遍公式 (7) 在 $t=0.1, 0.5, 0.9$ 三个时刻的 velocity 数值，看看 "水流 downhill" 的感觉。然后跑一下他们开源的代码（项目页 https://chris1220313648.github.io/DFM-VLA/），观察 token 在 16 步 decoding 中的变化轨迹。这两个练习比读 10 遍 paper 都管用。

---

# DFM-VLA：用 Discrete Flow Matching 实现 Action Token 的 Iterative Refinement

Andrej，很高兴跟你讨论这篇 paper。我会从底层 intuition 出发，把数学结构、架构选择和实验现象串起来。

## 1. 核心洞察：Irreversible Commitment 问题

这篇 paper 锁定了一个非常具体且关键的问题。在 VLA 模型把 action 离散化成 token 之后，decoding 范式存在两种主流，但都有同一个根本缺陷：**一旦一个 token 被生成，它就被锁死了，无法回头修正**。

具体地看三种 paradigm 的区别（Figure 2 里画得很清楚）：

- **Autoregressive VLA (AR-VLA)**：用 causal attention，从左到右生成 action tokens。已经生成的 token 不再变动。例：RT-1, OpenVLA, π0-FAST。
- **Discrete Diffusion VLA (DD-VLA)**：BERT-style masking，只有 masked 位置可以被预测，已经被 predict 的位置就固定了。例：Dream-VLA, dVLA, LLADA-VLA。
- **DFM-VLA (本文)**：基于 Continuous-Time Markov Chain (CTMC)，任意位置在任何 iteration 都可以被 revisit 和修改。

这里有一个 subtle 但极重要的点：DD 看起来像 iterative refinement，但它的 "refinement" 只发生在 masked 位置。已 unmask 的位置一旦确定就不再变动。所以 DD 的 "refinement" 实际上是 "fill-in-the-blank"，没有任何 "改主意" 的能力。

在 robot manipulation 里这个限制特别痛：action chunk 内部有时间依赖性，前面的 action token 错了会污染后面的预测。CALVIN 这种 long-horizon benchmark 把这个问题放大——一个 5-step chain 任何一步出错后续全部失败。所以 "can I revise my earlier guess?" 这个能力在 robotics 中比在 NLP 中更重要。

直觉上：DFM-VLA 让模型像人一样 "先打个草稿，再回头改"。Early iteration 是 exploration，late iteration 是 refinement。

## 2. 数学背景：Discrete Flow Matching

要理解 DFM-VLA，先理解 discrete flow matching 的数学。这是 Gat et al. (NeurIPS 2024) 提出的 framework。

### 2.1 Probability Paths

设 source distribution $p(x)$，target data distribution $q(x)$，离散空间 $\mathcal{S} = \mathcal{T}^D$：
- $D$ = 离散变量的数量（在 DFM-VLA 里是 action chunk 的 token 长度）
- $\mathcal{T} = [K] = \{1, 2, ..., K\}$ = 大小为 $K$ 的有限字母表（DFM-VLA 里 $K = 1024$，action vocab 大小）

DFM 构造一组时间索引的分布 $\{p_t(x)\}_{t \in [0,1]}$，从 $p$ 平滑过渡到 $q$：

$$p_t(x) := \sum_{x_1 \in \mathcal{S}} p_t(x | x_1) q(x_1)$$

各维度独立：
$$p_t(x | x_1) = \prod_{i=1}^{D} p_t(x^i | x_1^i)$$

每个维度的 conditional path 用 mixture 形式：
$$p_t(x^i | x_1^i) = (1 - \kappa_t(x_1^i)) p(x^i) + \kappa_t(x_1^i) \delta_{x_1^i}(x^i) \tag{1}$$

变量解释：
- $x^i$ = 第 $i$ 个 token 的当前状态
- $x_1^i$ = 第 $i$ 个 token 的 target (clean) 值
- $\kappa_t \in [0,1]$ = scheduler，$\kappa_0 = 0$（纯 source 噪声），$\kappa_1 = 1$（纯 target）
- $\delta_{x_1^i}(x^i)$ = Dirac delta，只在 $x^i = x_1^i$ 时为 1

直觉：$p_t$ 在 source distribution 和 point mass on target 之间线性插值。$t = 0$ 时全是噪声，$t = 1$ 时全是 clean。

### 2.2 Probability Velocities

光有 path 不够，需要 dynamics 把状态从 $t$ 推到 $t + h$。这里用 CTMC：

$$x_{t+h}^i \sim \delta_{x_t^i}(\cdot) + h \cdot u_t^i(\cdot | x_t^i, x_1^i) \tag{2}$$

变量解释：
- $u_t^i(\cdot | x_t^i, x_1^i)$ = velocity field（速度场），也叫 transition rate
- $h$ = step size
- $\delta_{x_t^i}(\cdot)$ = 当前状态的 point mass

直觉：当前 token $x_t^i$ 以速率 $u_t^i$ "流向" target $x_1^i$。CTMC 的好处是允许每个 token 独立 update，并且 update 后的 token 还可以在下一 iteration 再 update——这就是 "reversible" 的本质。

**这是与 DD 的根本区别**：DD 的 mask token 一旦被 predicted 就固定；DFM 的 token 是 CTMC 的 state，每一步都根据 velocity field 决定是否跳到新状态。状态可以反复跳。

## 3. DFM-VLA 的方法

### 3.1 架构

整体继承 UniVLA：
- **Text tokenizer**：Emu3 tokenizer
- **Image tokenizer**：VQ tokenizer（MovQ，compression ratio 4），第三人称 + wrist-view，都是 $25 \times 25 = 625$ tokens
- **Action tokenizer**：FAST tokenizer → BPE 压缩 → action vocab size 1024
- **Modality boundary**：`boi/eoi` 包围图像，`boa/eoa` 包围 action
- **Noising/prediction**：只施加在 action modality 上

注意：FAST 本身把连续 action 编码成离散 token，BPE 再进一步压缩 token 数量。这里有个小直觉——BPE 让 "常见 action 子序列" 成为单个 token，类似于 NLP 里常见词合并。这对 DFM 有好处：refinement 单位更语义化。

### 3.2 Velocity Field 构造方式 1：Auxiliary Velocity Head

灵感来自 EditFlow。EditFlow 原本支持三种 edit operation（insertion, replacement, deletion），但 DFM-VLA 只保留 replacement，因为 action chunk 长度预定义。

形式化：给定 noisy action tokens $x_t$ 和 context $l$，backbone 先产生 hidden states，velocity head 把 hidden states 映射成 replacement velocities：

$$h_t = f_\theta(x_t, l), \quad u_t^\theta(\cdot | x_t) = u_t^{head}(h_t) \tag{3}$$

变量解释：
- $f_\theta$ = backbone network
- $h_t$ = hidden states
- $u_t^{head}$ = velocity prediction head

训练损失（基于 EditFlow 的 velocity matching）：

$$\mathcal{L}_{head} = \mathbb{E}_{t \sim \mathcal{U}[0,1], x_1, x_t} \left[ \sum_{x \neq x_t} u_t^\theta(x | x_t) - \sum_{i=1}^{N} \mathbf{1}_{[x_t^i \neq x_1^i]} \log u_t^\theta(x^i | x_t^i) p_{1|t}(x_1^i | x_t^i, l) \right] \tag{4}$$

变量解释：
- 第一项 $\sum_{x \neq x_t} u_t^\theta(x | x_t)$ = total outgoing rate，相当于约束速度场的 "质量守恒"
- 第二项 = cross-entropy-like term，只在 $x_t^i \neq x_1^i$ 时（即 token 还需要 refine 时）施加
- $\mathbf{1}_{[x_t^i \neq x_1^i]}$ = indicator，仅在当前 token 与 target 不同时为 1

直觉：这个 loss 既要求 velocity 在所有 "非当前状态" 上有合理分布，又要求模型在 "还需要 refine 的位置" 高度集中概率到正确 target。

### 3.3 Velocity Field 构造方式 2：Action-Embedding-Guided（推荐）

这是 paper 的核心创新。利用 action embedding space 的 metric structure 来定义 probability path：

$$p_t(x^i | x_1^i) = \text{softmax}(-\beta_t \cdot d(x^i, x_1^i)) \tag{5}$$

变量解释：
- $d: \mathcal{T} \times \mathcal{T} \to \mathbb{R}_{\geq 0}$ = 在 action token embedding 空间中的距离，$d(x^i, x_1^i) = 0$ iff $x^i = x_1^i$
- $\beta_t: [0,1] \to \mathbb{R}_{\geq 0}$ = monotonic schedule，$\beta_0 = 0$, $\beta_1 = \infty$

具体实例化：
$$\beta_t = c \left(\frac{t}{1-t}\right)^\alpha, \quad t \in [0, 1) \tag{6}$$

变量解释：
- $c > 0$ = 控制 $\beta_t$ 整体尺度
- $\alpha > 0$ = 控制 temporal curvature，决定 $t \to 1$ 时概率质量多快集中

直觉：$t = 0$ 时 $\beta_0 = 0$，softmax 退化为均匀分布（纯噪声）。$t \to 1$ 时 $\beta_t \to \infty$，softmax 退化为 point mass on $x_1^i$。中间 $t$ 的分布以 $x_1^i$ 为中心，越近的 token 概率越大——保留了 embedding space 的语义邻域结构。

**这个设计的关键 intuition**：action token 的 embedding 不是 arbitrary 的，FAST 训练时学到的 embedding 让 "相近的 action token 在语义上也相近"。所以用 embedding distance 来定义 path 等价于在 "语义上连续" 的方向上 flow。

**Kinetic-Optimal Velocity**（Shaul et al. 2024）：

$$u_t^i(x^i, z | x_1) = p_t(x^i | x_1^i) \dot{\beta}_t [d(z^i, x_1^i) - d(x^i, x_1^i)]_+ \tag{7}$$

变量解释：
- $[\cdot]_+ = \max\{\cdot, 0\}$
- $z^i$ = vocabulary 中的任意候选 token
- $\dot{\beta}_t$ = $\beta_t$ 对 $t$ 的导数
- $d(z^i, x_1^i) - d(x^i, x_1^i)$ = $z$ 比 $x$ 离 target $x_1^i$ 更远的量

直觉：这个 velocity 只在 $x^i$ 比 $z^i$ 更接近 target $x_1^i$ 时把概率从 $z$ 移到 $x$。换句话说，flow 永远朝向离 target 更近的方向——这是一种 "单调 refinement"。

训练目标（标准 cross-entropy）：
$$\mathcal{L}_{ce} = \mathbb{E}_{t \sim \mathcal{U}[0,1], x_1, x_t} \left[ -\log p_{1|t}(x_1 | x_t, l) \right] \tag{8}$$

变量解释：
- $p_{1|t}^\theta(\cdot | x_t, l)$ = 模型在每个 action token 位置输出的 categorical distribution

注意：embedding-guided 只需要预测 $x_1$ 的 categorical 分布，velocity 由 Eq. 7 解析计算。Head-based 需要专门预测 velocity。这就是为什么 embedding-guided 更 sample efficient（Figure 5）：模型只需要学一个 standard cross-entropy objective，velocity structure 由 metric path 自动给出。

### 3.4 两阶段解码

这是工程上一个很聪明的设计。总共 $T = T_{fine} + T_{val}$ 步（实验里默认 $T = 16$）。

**Stage 1: Iterative Refinement（前 $T_{fine}$ 步，默认 14 步）**

用 Euler 离散化 CTMC。对每个 token 位置 $i$，每一步 $t \to t + h$：

1. 从模型 sample 预测的 clean token：$x_1^i \sim p_{1|t}^i(\cdot | x_t, l)$
2. 计算 total outgoing rate：$\lambda^i = \sum_{x^i \neq x_t^i} u_t^i(x^i, x_t^i | x_1^i)$
3. 抽 uniform 随机数 $Z_{change}^i \sim U[0,1]$
4. 如果 $Z_{change}^i \leq 1 - e^{-h\lambda^i}$（发生跳），从归一化 rate $\frac{u_t^i(\cdot, x_t^i | x_1^i)}{\lambda^i}(1 - \delta_{x_t^i}(\cdot))$ 中 sample 新 token；否则保持 $x_{t+h}^i = x_t^i$

变量解释：
- $\lambda^i$ = token $i$ 的 "跳出去" 的总速率
- $1 - e^{-h\lambda^i}$ = 在 $h$ 时间内的跳概率（Poisson 过程的累积概率）
- 跳的目的地按 velocity 归一化后选择

直觉：低 $\lambda^i$ 表示模型对当前位置很自信，几乎不跳；高 $\lambda^i$ 表示模型想 refine，按 velocity 分布跳。这里的关键是 **跳后的 token 在下一 iteration 仍然可以被再跳**，这是 CTMC 的本质特性。

**Stage 2: Deterministic Validation（后 $T_{val}$ 步，默认 2 步）**

关闭 stochastic jump，用 greedy decoding：

$$x_{t+h} = \arg\max \text{softmax}(p_{1|t}(\cdot | x_t)) \tag{9}$$

直觉：早期 stochastic 让模型 explore 更广的 token space，后期 greedy 让最终输出稳定。Ablation（Table 4）显示 $T_{val} = 0$（全部 stochastic）性能略低（94.8% on LIBERO），$T_{val} = 2$ 最佳（95.7%），$T_{val} = 8$ 太多 greedy 又损害性能（95.1%）。这是一个典型的 "explore-exploit" tradeoff。

### 3.5 Adaptive KV Caching

借鉴 FAST-dLLM 的思路。观察：DFM 的 token 在 iterative refinement 中变化很小（early 阶段大部分 token 已经接近 final），所以 KV cache 可以复用。

策略：
- Instruction 和 observation 的 KV cache 全程固定
- Action-side cache 根据 current value features 和 cached value features 的 cosine similarity 自适应更新——只更新变化大的位置

效果（Table 5）：DFM + Adaptive Cache 把 speed 从 60.2 提到 121.0，几乎翻倍。同时性能几乎不掉（Avg. Len. 从 4.42 到 4.40）。

这个工程优化让我想起：discrete flow 的一个 "隐藏红利" 是 token 状态变化平缓，让 KV cache 复用变得可行。AR 完全不行（每步必须更新 KV），DD 在 unmask 后也不能复用（位置变了语义就变了）。DFM 的 "原地 refine" 天然适合 cache 复用。

## 4. 实验分析

### 4.1 CALVIN（Table 1）

CALVIN ABCD→D，1000 rollouts，5 个连续 sub-task：

| Method | 1-step | 3-step | 5-step | Avg. Len. |
|--------|--------|--------|--------|-----------|
| UniVLA* | 0.948 | 0.862 | 0.690 | 4.26 |
| UP-VLA | 0.962 | 0.879 | 0.812 | 4.42 |
| DFM-VLA+Head | 0.968 | 0.880 | 0.776 | 4.42 |
| **DFM-VLA+Embed** | **0.976** | **0.892** | **0.780** | **4.44** |

关键观察：
- 3-step completion 比 UniVLA* 高 3 个点（0.892 vs 0.862），5-step 高 9 个点（0.780 vs 0.690）
- 长程优势明显——这正是 iterative refinement 的价值所在，错误可以在后续 step 被纠正

### 4.2 LIBERO（Table 2）

| Suite | DFM-VLA+Embed | Dream-VLA | FlowVLA | π0-FAST |
|-------|---------------|-----------|---------|---------|
| Spatial | 96.8% | 97.5% | 93.2% | 96.4% |
| Object | **98.8%** | 94.0% | 95.0% | 96.8% |
| Goal | **94.4%** | 89.5% | 91.6% | 88.6% |
| Long | **92.6%** | 89.5% | 72.6% | 60.2% |
| Avg | **95.7%** | 92.6% | 88.1% | 85.5% |

关键观察：Long suite 上 DFM-VLA 比 π0-FAST 高 32 个点。Long 测的是 multi-stage compositional reasoning，正是错误最容易累积的场景。

### 4.3 Real-World（Table 7）

三个任务（bimanual AgileX 平台）：
- Pot Lift: 77.5%（DFM-VLA）vs 65.0%（RDT）vs 57.5%（Dream-VLA）
- Place Veg. to Pot: 70.0% vs 62.5% vs 62.5%
- Place Block to Plate: 65.0% vs 52.5% vs 42.5%
- 平均：70.8% vs 60.0% vs 54.2%

直觉：real-world 比 simulation 更能体现 refinement 价值——sim 数据分布窄，错误影响相对可控；real-world 噪声大，错误一旦产生不修正就完蛋。

### 4.4 Velocity Field 构造对比（Figure 5）

Embedding-guided 在 20k 步就达到 95.7%，head-based 在 20k 步只有 ~91%。Embedding-guided 收敛快、最终性能好——因为利用了 action embedding 的 prior structure，velocity 不需要从头学。

### 4.5 数据规模（Table 6）

| Data | AR | DD | DFM |
|------|------|------|------|
| 10% | 1.71 | 2.84 | **3.21** |
| 50% | 3.01 | 3.88 | **4.03** |
| 100% | 4.18 | 4.32 | **4.44** |

10% 数据下 DFM 比 AR 高 1.5（接近一倍！），比 DD 高 0.37。低数据 regime 下 iterative refinement 价值最大——因为单次 prediction 错误率高，能修正就更值钱。

### 4.6 效率（Table 5）

| Method | Avg. Len. | Speed |
|--------|-----------|-------|
| AR | 4.18 | 50.2 |
| DD + Adap. Cache | 4.27 | 118.3 |
| DFM + Adap. Cache | 4.40 | **121.0** |

DFM + Adaptive Cache 同时拿到最好的性能和最快的 speed。这是 DFM 的工程友好性体现。

## 5. 我的 Intuition 和思考

### 5.1 为什么这个 framework 在 robotics 中特别有效

action chunk 有强时间依赖性。AR 类似 greedy decoding——错了就崩。DD 像 batch fill-in，能并行但同样无法回头。DFM 引入 "draft-and-revise" 的机制，正好契合 robotics 的需求：先粗略预测整个 chunk，然后基于 chunk-level context 反复修正。

这让我想起 diffusion model 在 continuous action space 中的成功（如 π0, RDT）。那些模型本质上也是 iterative refinement，只是在 continuous space 上。DFM-VLA 把这个 idea 搬到了 discrete space，并且保留了 VLA 的 "discrete token" 优势——可以无缝接入 LLM-style backbone。

### 5.2 Embedding-Guided 的深层意义

我认为 Embedding-Guided formulation 是这篇 paper 最深刻的贡献。它把 action embedding 的 metric structure 显式编码进 probability path。这意味着 refinement 不是 random walk，而是 "沿着语义流形朝 target 滑动"。

考虑 Figure S1 的可视化：在 embedding space 中，token 从 random 位置出发，每一步朝 $x_1^{pred}$ 滑动，feasible jump set 越来越小，update 越来越 local。这是一种 "粗到细" 的 refinement——early iteration 跨大步，late iteration 微调。

这个 design choice 也呼应了 kinetic-optimal flow matching 的理论：在 metric space 上，最小传输能量的 path 就是沿测地线滑动。Embedding-guided 把 action embedding 当作 metric space，自然获得了 kinetic-optimality。

### 5.3 Irreversible Commitment 在 NLP 和 Robotics 中的差异

我注意到一个有趣的对比：在 NLP 中，discrete diffusion（如 MDLM, SEDD, D3PM）虽然也有 irreversible commitment 问题，但 NLP token 之间的依赖相对 "local"，错一个 token 通常不至于毁掉整个序列。在 robotics 中，action chunk 内部强相关，错一个 token 整个 chunk 都可能无效。所以 robotics 比 NLP 更需要 "可修正性"。

这暗示一个可能的方向：在 long-form text generation 中，DFM-style refinement 也可能有价值，尤其是对 "结构性强" 的文本（如代码、数学证明）。

### 5.4 两阶段解码的启发

两阶段（stochastic refine + greedy validate）本质上是 "探索-利用" 的分时方案。这让我想起 RL 中的 epsilon-greedy schedule。但这里更精妙：stochastic 阶段是基于 velocity field 的概率跳转，而不是 random exploration。

一个可能的 extension：让 $T_{val}$ 自适应——根据当前 prediction 的 entropy 决定何时切到 greedy。如果 entropy 已经很低，提前切；如果还有不确定性，多 refine 几步。

### 5.5 Adaptive KV Cache 的更广意义

这个工程 trick 其实揭示了一个深层点：discrete flow 的 "迭代式 refine" 比 AR 的 "顺序生成" 更适合 cache 复用。AR 每生成一个 token 必须更新 KV，因为 attention 依赖前面所有 token。DFM 所有 token 一直都在，只是状态在变——大多数时候变化平缓。

这暗示：discrete flow 可能是 future "高效推理" 的一个有前景方向，特别是在长序列生成中。Future work 可能要把这个 idea 移植回 LLM 本身。

### 5.6 与最近的 dLLM 工作的连接

最近一年 discrete diffusion/flow LLM 进展很快（LLADA, Dream 7B, Fudoki, OneFlow, EditFlow, URSA）。DFM-VLA 把这些 idea 第一次系统搬到 VLA。但 robotics 有特殊性：
- action 比 text 短（chunk 几十 token）
- action 比 text 时间依赖强
- action space 的 metric structure 更有意义（FAST embedding）

所以 DFM-VLA 不是简单 "把 dLLM 套到 robot"，是针对 robotics 特性定制的 velocity field 设计。Embedding-guided 就是这种定制化的体现。

### 5.7 局限和可能的 next step

paper 没有讨论到的几个点：

1. **Velocity field 的 capacity**：当前 velocity 是 analytic（embedding-guided）或 single head（head-based）。如果 action space 高度 multimodal（同一个 context 多种合理 action），velocity 可能需要更复杂的表达。可能需要 mixture of velocities 或 hierarchical velocity。

2. **Refinement step 数的 adaptive selection**：当前固定 $T = 16$。简单任务可能 4 步就够，复杂任务可能需要 32 步。一个 learned controller 决定何时停止可能更高效。

3. **与 CoT 的结合**：MM-ACT, UD-VLA 等工作把 CoT 引入 discrete diffusion。DFM-VLA 是否可以 jointly refine reasoning trace 和 action？EditFlow 已经证明可以 insert/delete/replace，DFM-VLA 可以借鉴。

4. **Theory of refinement benefit**：现在只有 empirical 证据。一个 theoretical analysis——在什么条件下 DFM 比 AR 严格更好——会让 paper 更有深度。可能可以用 approximation error 的 propagation 来分析。

## 6. 参考链接

为了让你更深入挖：

**Discrete Flow Matching 理论基础**：
- Discrete Flow Matching (Gat et al., NeurIPS 2024): https://arxiv.org/abs/2407.15575
- Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective (Shaul et al., 2024): https://arxiv.org/abs/2412.03487
- Edit Flows: Flow Matching with Edit Operations (Havasi et al., 2025): https://arxiv.org/abs/2506.09018
- OneFlow (Nguyen et al., 2025): https://arxiv.org/abs/2510.03506
- URSA (Deng et al., 2025): https://arxiv.org/abs/2510.24717

**Discrete Flow LLM/VLM**：
- Fudoki (Wang et al., 2025): https://arxiv.org/abs/2505.20147
- Next-Omni (Luo et al., 2025): https://arxiv.org/abs/2510.13721
- Dream 7B (Ye et al., 2025): https://arxiv.org/abs/2508.15487

**VLA Baseline**：
- OpenVLA (Kim et al., 2025): https://openvla.github.io/
- π0 (Physical Intelligence): https://arxiv.org/abs/2410.24164
- FAST (Pertsch et al., 2025): https://arxiv.org/abs/2501.09747
- UniVLA (Wang et al., 2025): https://arxiv.org/abs/2506.19850
- Dream-VLA (Ye et al., 2025): https://arxiv.org/abs/2512.22615
- RDT-1B (Liu et al., 2024): https://arxiv.org/abs/2410.07864

**Discrete Diffusion VLA**：
- Discrete Diffusion VLA (Liang et al., 2025): https://arxiv.org/abs/2508.20072
- LLADA-VLA (Wen et al., 2025): https://arxiv.org/abs/2509.06932
- CEED-VLA (Song et al., 2025): https://arxiv.org/abs/2506.13725
- dVLA (Wen et al., 2025): https://arxiv.org/abs/2509.25681

**Benchmark**：
- CALVIN (Mees et al., 2022): https://calvinrobot.github.io/
- LIBERO (Liu et al., 2023): https://libero-project.github.io/

**Tokenizer**：
- Emu3 (Wang et al., 2024): https://arxiv.org/abs/2409.18869
- MovQ (Zheng et al., 2022): https://arxiv.org/abs/2202.11878

**Discrete Diffusion LLM 综述**：
- Discrete Diffusion in LLM & Multimodal (Yu et al., 2025): https://arxiv.org/abs/2506.13759

**项目页面**：
- DFM-VLA: https://chris1220313648.github.io/DFM-VLA/

---

简短总结一下我的整体 take：DFM-VLA 是 discrete VLA decoding 范式的一个有意义推进。它把 irreversible commitment 这个隐含问题显式化，并用 CTMC 的 "可逆 state transition" 来解决。Embedding-guided velocity 是个优雅的工程选择，把 action embedding 的 metric structure 显式编码进 flow dynamics。两阶段解码和 adaptive KV cache 让方法在精度和效率上都站住脚。从更广视角看，这是 discrete flow matching 从 NLP 扩展到 action modality 的一个标杆性工作，未来在 CoT-action joint refinement、adaptive step control 等方向都还有空间。

如果你想 build deeper intuition，我建议重点看 Figure S1 的 embedding-space 可视化和 Figure 4 的单步 decoding diagram——这两个图把 DFM 的 "可逆 refinement" 机制讲得最清楚。
