---
source_pdf: Observing and Controlling Features in.pdf
paper_sha256: e2a68c8bc1d99e7b25cfead6c9dc64a1e4eda0af5a49cae661e5acbbb28b7004
processed_at: '2026-08-05T22:53:00-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话概括

这些 Stanford 的人发现，**robot 的大脑（VLA 模型）内部藏着一个"旋钮面板"**，你可以不 retrain、不 fine-tune，inference 的时候直接拧这些旋钮，就能让 robot 开/合夹爪、抬高手臂、放慢速度，同时还不破坏它干正事的能力。

---

## 背景故事

### LLM 圈子早就发现的事儿

搞 LLM 的人这两年都知道一个 trick：你不需要 fine-tune 一个 model 来改变它的 behavior，直接在它 forward pass 中间塞一个 **shift vector** 进去就行。想让 ChatGPT 更 aggressive？找到 "aggressive" 这个 direction，把 activation 往那个方向推一下，输出就变了。这叫 **activation steering**。

更关键的是，LLM 圈子还发现一个神奇现象：**model 内部用 linear 的方式编码 high-level features**。意思是，"sentiment"、"topic"、"personality" 这些抽象概念，在 transformer 的中间层里就是 **某个方向上的投影**。一个最简单的 linear classifier（`W^T x + b`）就能读出来。这叫 **linear representation hypothesis**。

### VLA 圈子的问题

VLA（Vision-Language-Action model）就是给 robot 用的 LLM，输入是图像 + 语言 + robot 自身状态，输出是 robot action。代表作品有 OpenVLA、π₀、π₀.₅ 这些。

VLA 架构上跟 LLM 有共同点：都有 transformer backbone。所以理论上 activation steering 应该能 port 过来。但 VLA 有三个 LLM 没有的麻烦：

1. **Multimodal**：输入不只有 text，还有 image、proprioception
2. **Continuous action**：输出不是 discrete tokens，是连续的 robot 命令
3. **Closed-loop**：这是最致命的 —— LLM 生成完就结束，VLA 每个 action 会改变 environment，environment 又变成下一个 input。你 intervene 一下，action 变了，environment 变了，下一个 input 变了，intervention 的效果会被 feedback 放大或衰减

之前有人 [5] 在 VLA 上试过 activation steering，做法比较粗暴，直接 **overwrite** activations，效果有但破坏 naturalness，task success rate 掉得厉害。

### 这篇 paper 做了什么

这些作者说：我们别瞎搞，把 LLM 那套 activation steering 的成熟方法 **严格 port 过来**，用 control theory 的语言 formalize 一下，然后看看 VLA 到底能不能像 LLM 一样被 steer。

他们的核心贡献是两点：

**第一，VLA 的 transformer 内部确实 linearly encode 了 robot 的 state 和 action**。你用最简单的 linear probe 就能从中间层 activation 里读出来 robot 的位置、姿态、夹爪状态。这证明 linear representation hypothesis 在 VLA 上成立。

**第二，用 minimal linear intervention 可以在 closed-loop 场景下 steer VLA 的 action，同时 preserve task success rate**。关键技巧是别 over-push，只做 **最小的 intervention** 让 observation 落到 desired region，其他什么都不动。

---

## 核心思想：Observability 与 Controllability

作者借用了 Kalman 控制理论里的两个经典概念：

### Observability：能不能从内部状态读出 feature？

你有个 VLA 在跑，中间层 `x_ℓ` 是个 d-维 vector。你想知道"这个时刻 robot 想把夹爪打开还是关闭"，能不能从 `x_ℓ` 里读出来？

答案：能，而且 **linear 地读**就行。存在一个 `W_ℓ` 和 `b_ℓ`，使得：
$$\zeta = W_\ell^T x_\ell + b_\ell$$

这里：
- `x_ℓ ∈ ℝ^d`：layer ℓ 的 hidden state
- `W_ℓ ∈ ℝ^{d×n}`：要学的 weight matrix（d 是 representation 维度，n 是 feature 维度）
- `b_ℓ ∈ ℝ^n`：bias
- `ζ ∈ ℝ^n`：你关心的 feature（比如 gripper state，height，speed）

怎么学这个 `W_ℓ`？收集一堆数据 `{(s^(i), ζ^(i))}`，把 `s^(i)` forward 到 layer ℓ 拿到 `x_ℓ^(i)`，然后用 supervised learning 拟合 `ζ^(i)` 就行。这就是 LLM 圈子标准的 **linear probing**。

### Controllability：能不能干预内部状态把 feature 推到 target？

你想让 gripper 保持 closed，但现在 observation 说它想 open。怎么干预？

最 naive 的做法：往 `W_ℓ` 方向猛推一把。问题是推太多会破坏其他 features，model 整体 behavior 变怪。

作者的解法：**最小能量 control**。找一个 `u_ℓ`，让 `x_ℓ + u_ℓ` 经过 observer 之后落在 desired region `[ζ_min, ζ_max]`，同时 `||u_ℓ||` 尽量小。

数学上：
$$u_\ell = \arg\min_{u} \|u\|_2^2 \quad \text{s.t.} \quad W_\ell^T(x_\ell + u) + b_\ell \in [\zeta_{\min}, \zeta_{\max}]$$

- `u_ℓ ∈ ℝ^d`：intervention vector
- `||u||_2^2`：intervention 的"能量"，要最小化
- 约束：intervened representation 被 observer 读出来必须在 [ζ_min, ζ_max] 区间内

这个优化问题有 **闭式解**（不需要在线求解优化）：

$$u_\ell = (\zeta_{\max} - \zeta_\ell) \frac{W_\ell}{\|W_\ell\|_2^2}, \quad \text{if } \zeta_\ell > \zeta_{\max}$$

$$u_\ell = (\zeta_{\min} - \zeta_\ell) \frac{W_\ell}{\|W_\ell\|_2^2}, \quad \text{if } \zeta_\ell < \zeta_{\min}$$

$$u_\ell = 0, \quad \text{otherwise}$$

这里：
- `ζ_ℓ = W_ℓ^T x_ℓ + b_ℓ`：当前 observation
- `ζ_max`, `ζ_min`：你想约束的上下界
- `W_ℓ / ||W_ℓ||_2^2`：feature direction 的归一化形式

**直觉解释**：观察一下当前 observation 超出 bound 多少，然后往 feature direction 的反方向推一个刚好让它回到 bound 的量。不推过头，不推不够。如果 observation 已经在区间内，**什么都不做**。

这个解来自 [Cheng & Alonso 2024] 的 Theorem 4.1（https://arxiv.org/abs/2405.15454），他们本来是给 LLM 做的，这里直接 port 过来。

---

## 为什么 minimal intervention 这么重要

这是这篇 paper 最关键的 insight，值得单独讲。

Representation space 是 high-dimensional（d ≈ 2048-4096），里面有无数个 feature directions。你想 steer 的只是其中一个（比如 gripper state），但 representation 里还编码了 task context、object affordances、spatial reasoning 等等无数个其他 features。

如果你推一个大 vector 进去，虽然 gripper feature 变了，但其他 features 也被波及了。结果就是：gripper 确实闭合了，但 model 可能忘了在干什么任务，开始乱搞。

Minimal L2 intervention 的哲学是：**只做必要的改变，leave everything else intact**。这是 paper 能在 closed-loop 里保持 90%+ success rate 的核心原因。

对比 [5] 之前的 activation overwrite 方法，他们直接用大 vector 覆盖 activations，success rate 掉得很惨。这篇 paper 的 minimal intervention 几乎不掉 success rate，这是关键改进。

---

## 两种 VLA 架构怎么处理

Paper 在两种主流架构上做了实验：

### OpenVLA（Transformer-based）

- DINOv2 + SigLIP 做 vision encoding
- Llama 2 backbone 做 autoregressive 预测
- Action 是 tokenized 的，最后 decode 成 robot 命令
- Intervention 在 transformer layers 上做，通过 final layer 影响 action

### π₀.₅（Transformer + Flow-Matching hybrid）

- VLM 处理 vision + language
- 独立的 "action expert" transformer 用 **conditional flow matching** 生成连续 action trajectories
- Flow matching layers 通过 cross-attention 关注 transformer layers
- Intervention 在 transformer layers 上做，通过 flow matching head 传导到 action

**关键发现**：两种架构都 work。这说明只要 transformer backbone 是 VLA 的一部分，对它的 intervention 就能影响最终 action，不管 head 是 autoregressive 还是 flow matching。

---

## 实验讲了什么

### Observer 实验（Figure 3, 4）

在 π₀.₅ 上用 Libero dataset，在 OpenVLA 上用 BridgeData V2：

**Probe 性能**：
- Linear probe 的 MAE（mean absolute error）显著低于 mean baseline
- Accuracy 显著高于 majority class baseline
- 说明 state 和 action **确实 linearly encoded** 在 transformer 中间层

**Robustness 实验**（Figure 4）：
- 加扰动 `x_ℓ + α`，看 action 变化
- π₀.₅：α 越大，action 变化越大，**单调且 smooth**，很 robust
- OpenVLA：delta yaw 不太 robust，delta gripper 还行但不如 π₀.₅ clean

**Layer depth 效应**：
- Earlier layers 的 intervention 效果更强
- 为什么？因为 representation 的 L2 norm 随 depth 增大（Figure 4 bottom），固定 α 在大 norm 上相对效应变小
- Intuition：early layers 是 feature **construction** 阶段，还有 leverage；later layers 是 feature **readout** 阶段，已经定型

### Controller 实验（Figure 5-10）

在 Libero spatial suite 上做 closed-loop rollout，10 tasks × 10 rollouts，NVIDIA 5090 GPU。

**Gripper state 控制**（Figure 6, 8）：
- Constraint：gripper 保持 open 或 closed
- 三种方法对比：no intervention、prompting、control（本文）
- Control：near-perfect constraint satisfaction（接近 100%），success rate > 90%
- Prompting 也能 steer 但不如 control
- No intervention 完全无法满足 constraint

**End-effector height 控制**（Figure 7, 8）：
- Constraint：EE 保持在初始高度以上/以下
- Control：near-perfect constraint satisfaction
- Success rate 有 modest drop（因为 constrained task 严格难于 unconstrained）
- 作者假设：更 robust 的 base model 可以消除这个 drop

**End-effector speed 控制**（Figure 9, 10）：
- Speed 是 derived quantity：`v = ||(Δx, Δy, Δz)|| / dt`，不直接是 model output
- **Slow down** 可靠
- **Speed up** 不准确（因为 training data 缺乏 fast speed regime，OOD 问题）
- Success rate 几乎完全保持

**关键 takeaway**：即使 feature 不直接是 model output，只要 linearly observable，controller 就能 steer。这暗示 VLA internal representations 编码了 derived/relational quantities。

---

## Closed-loop 为什么能 work

这是 paper 最 surprising 的部分。VLA 是 closed-loop，intervention 会通过 environment feedback 被 amplify 或 attenuate，按理说会 destabilize。但实验显示 stable。

作者的解释和我的理解：

### 1. Intervention 是 state-dependent 的

Eq. 7 的 controller 是 **feedback controller**，不是 open-loop 的 fixed shift。它根据当前 observation 计算 intervention：
- 如果 observation 已经在 desired region → `u = 0`，不干预
- 如果 observation 超出 bound → 推回到 bound

这意味着 model 不会"失控地"往某个方向推，而是 **自适应地** 调整。

### 2. 当 observation 进入 region，intervention 自动消失

这是 hard constraint 形式带来的好处。如果用 soft penalty（比如 `+ λ * max(0, ζ - ζ_max)`），intervention 会一直存在，即使 observation 已经在 region 内。这会导致 over-correction。

Hard constraint 的 minimal intervention 只在 violation 时激活，进入 region 就停止。这相当于一个 **dead-zone controller**，closed-loop 里很 stable。

### 3. Observer 是 robust 的

Remark 1 强调 observer 要 robust：`||f(x+ε) - f(x)|| < δ`。如果 observer 对小扰动敏感，intervention 会 oscillate。实验验证 observer 确实 robust（至少对 π₀.₅）。

### 4. Environment dynamics 本身有 damping

Robot 在 physical world 里动，不会瞬间跳跃。所以即使 intervention 有点 overshoot，environment 的物理约束会自然 smooth 掉。

---

## 跟 LLM activation steering 的关键差异

| 维度 | LLM | VLA (这篇 paper) |
|------|-----|------------------|
| Loop type | Open-loop generation | Closed-loop with environment |
| Output space | Discrete tokens | Continuous actions |
| Feature type | Sentiment, topic, persona | Robot states, actions |
| Safety stakes | Text generation | Physical world |
| Architecture | Pure transformer | Transformer + (possibly) flow matching |
| Intervention effect | One-shot | Cumulative through feedback |
| Success metric | Naturalness, coherence | Constraint satisfaction + task success rate |

最后一个点很关键：LLM steering 的 success metric 主要是"文本还 natural 吗"，VLA 必须同时看 **constraint satisfaction** 和 **task success rate**。只满足 constraint 但 task fail 了没用。这篇 paper 是第一个 **同时 report 两者** 的 VLA steering 工作。

---

## 我看到这 paper 的第一反应

### 惊喜的点

1. **Linear representation hypothesis 在 VLA 上也成立**。这本来不是 obvious 的 —— VLA 有 vision encoder、有 action head，可能 state/action 编码方式跟 LLM 不同。Paper 实验证明 transformer backbone 里确实 linearly encoded，这是 mechanistic interpretability for VLA 的基础性发现。

2. **Closed-loop 不爆炸**。这是我读之前最大的怀疑。activation steering 在 LLM 上是 open-loop，port 到 closed-loop 我以为会 oscillate 或 diverge。但 paper 实验显示 stable，而且 success rate 保持 90%+。这说明 feedback controller 的设计哲学（state-dependent, minimal intervention, dead-zone）天然适合 closed-loop。

3. **Cross-architecture generalization**。在 transformer-based (OpenVLA) 和 hybrid (π₀.₅) 两种架构上都 work，说明 method 不依赖 specific architecture details，只依赖 "transformer backbone + linear representation" 这个 general property。

### 让我皱眉的点

1. **OpenVLA 的 delta yaw 不 robust**（Figure 4）。这说明 linear representation hypothesis 在 OpenVLA 上不是所有 features 都成立。Paper 没深究为什么。可能是 OpenVLA 的 tokenized action representation 跟 π₀.₅ 的 continuous action representation 本质不同，某些 features 编码方式更 nonlinear。

2. **Speed-up 失败**。这是 OOD 问题。Training data 缺乏 fast regime，所以 representation space 的 "fast direction" 附近 density 低，推过去就 fall off manifold。这说明 method 在 distribution 中心区域 work，在 boundary 区域脆弱。

3. **Layer selection 是 black art**。Paper 说 "best performant layer"，但没给 principled criterion。哪个 layer 最 observable、哪个 layer 最 controllable，需要 sweep 一遍。实用起来有点麻烦。

4. **Multi-feature conflicts 未讨论**。同时控制 gripper + height + speed 怎么办？Eq. 6 是 single-feature 的。Multi-feature 需要 multi-objective optimization 或者 stacked single-feature interventions，但 stacked interventions会不会互相破坏？Paper 没碰这个问题。

5. **Safety guarantees 缺失**。Remark 1 的 robustness check 是 empirical 的。如果 deployment 时遇到 OOD input，observer 可能 fail，controller 基于 wrong observation 做 intervention，可能更危险。对 safety-critical robotics 来说，这需要 theoretical bounds。

---

## 联想与未来方向

### 1. SAE-based feature discovery

Paper 说 observer 需要 labeled data。但 LLM 圈子已经在用 **Sparse Autoencoders (SAEs)** 做 unsupervised feature discovery（https://transformer-circuits.pub/2024/scaling-monosemanticity/）。SAE 能从 raw activations 里发现 monosemantic features，不需要 labels。把这个 port 到 VLA 上，可以找 robot 自己内部编码的 features，可能包括人类没标注的高层 semantic concepts（task understanding, object affordances）。

### 2. Multi-feature control via Pareto optimization

Eq. 6 的 single-feature constraint 可以 generalize 成 multi-feature：
$$u = \arg\min ||u||^2 \quad \text{s.t.} \quad f_1(x+u) \in D_1, f_2(x+u) \in D_2, \ldots$$

多 feature 的约束可能 conflict，需要 Pareto front 或者 weighted sum。这是个有趣的 optimization 问题。

### 3. Mechanistic finetuning vs inference-time steering

[13]（https://arxiv.org/abs/2511.22697）做了 mechanistic finetuning：识别 task-relevant attention heads 并 fine-tune 它们。这是 weight-level intervention。本文是 activation-level intervention。两者互补：finetuning 改变 model 的 long-term behavior，steering 做 runtime adaptation。组合起来可能很 powerful。

### 4. Hypersteer for scale

[17]（https://arxiv.org/abs/2506.03292）的 Hypersteer 用 hypernetwork 在 scale 上做 activation steering。如果 VLA 需要同时处理多个 constraints（不同 task 不同 requirements），hypernetwork 可以 condition on task 来 generate steering vectors。

### 5. Transporting activations

[16]（https://arxiv.org/abs/2502.05471）用 optimal transport 在 activation space 间 transport。比 additive intervention 更 sophisticated，可以处理 multimodal feature distributions。对 VLA 的 multi-modal action distributions 可能更合适。

### 6. Task-level semantic features

Paper 只控制 low-level features（gripper, height, speed）。但 VLA 的 transformer 里应该也编码了 task understanding、object affordances、spatial relationships 这些高层 concepts。[12]（https://arxiv.org/abs/2502.04558）已经在 probing 这些。如果能 observe 并 control 这些 high-level features，就能 steer robot 的 **strategy**，不只是 **parameter**。这是 VLA interpretability 的 next frontier。

### 7. Safety guarantees

对 safety-critical robotics deployment，需要 **formal bounds** on intervention effects。Paper 的 empirical robustness check 不够。可能的方向：
- Lipschitz bound on observer
- Reachability analysis on closed-loop with intervention
- Contraction theory for stability guarantees

---

## 我会怎么 follow up 这工作

如果我在做 VLA interpretability，我会这么推进：

**Step 1**：在 π₀.₅ 上跑 SAE，看能发现什么 monosemantic features。是否包括 paper 没覆盖的高层 concepts（task goals, object relationships）？

**Step 2**：把 Eq. 6 扩展到 multi-feature，用 Pareto 或 weighted sum，在 closed-loop 里同时控制 gripper + height + speed，看 success rate 怎么变。

**Step 3**：结合 mechanistic finetuning [13]：先 finetune task-relevant heads，再在 finetuned model 上做 inference-time steering。看两者是否 synergistic。

**Step 4**：在 real robot 上 deploy，看 sim-to-real gap。Paper 只在 Libero sim 上做，real world 的 noise、latency 可能打破 closed-loop stability。

**Step 5**：建立 observer 的 Lipschitz bound，给 controller 设计 safety margin。这是 deployment 必需的。

---

## 总结一下直觉

这篇 paper 的 core message 用三句话说：

1. **VLA 的 transformer 内部 linearly encode 了 robot 的 state 和 action**，跟 LLM 一样，linear probe 就能读出来。

2. **用 minimal L2 intervention 可以把这些 features steer 到 desired region**，闭式解，O(d) 计算，real-time 友好。

3. **关键 trick 是最小化干预能量**，只动你关心的 feature，其他什么都不碰。这让 model 在 closed-loop 里保持 90%+ task success rate。

更深层的 message 是：**LLM mechanistic interpretability 的工具箱可以 port 到 VLA**，但需要小心 closed-loop 这个新挑战。这篇 paper 是 VLA interpretability 的 first principled step，后续肯定有很多 follow-up。

参考核心 paper：
- 本文：https://arxiv.org/abs/2509.00328（同期工作，类似思路但更早）
- 理论基础：https://arxiv.org/abs/2405.15454
- π₀.₅：https://arxiv.org/abs/2504.16054
- OpenVLA：https://proceedings.mlr.press/v270/kim25c.html
- Linear representation hypothesis：https://arxiv.org/abs/2401.01315
- Activation addition：https://arxiv.org/abs/2308.10248
- SAEs for interpretability：https://transformer-circuits.pub/2024/scaling-monosemanticity/

---

# Observing and Controlling Features in Vision-Language-Action Models — 深度技术解析

## 一、Paper 核心定位与 Motivation

这篇 paper 来自 Stanford 的 Marco Pavone 组（合著者包括 NVIDIA Research），核心贡献是把 LLM mechanistic interpretability 中成熟的 **activation steering** 范式正式 port 到 VLA（Vision-Language-Action）模型上，并用控制理论的 **observability / controllability** 双重概念把这个过程 formalize。

直觉上，作者想回答两个问题：
1. VLA 内部 transformer 的 hidden states 里，是否 **linearly encode** 了 robot 的 state 和 action（即 LLM 里著名的 *linear representation hypothesis* 在 VLA 上是否成立）？
2. 如果是，能否用 **最小的线性干预** 把这些 features steer 到 user-specified 的 region，同时 **preserve 原模型的 closed-loop task success rate**？

VLA 比 LLM 难的地方在于三点：(i) multimodal inputs；(ii) continuous action outputs（而非离散 tokens）；(iii) **closed-loop** —— action 改变 environment，environment 又变成下一个 input，所以 intervention 会被 feedback 放大或衰减。这一点是 LLM activation steering 完全没有的挑战。

参考链接：
- π₀ paper: https://arxiv.org/abs/2410.24164
- π₀.₅ paper: https://arxiv.org/abs/2504.16054
- OpenVLA: https://proceedings.mlr.press/v270/kim25c.html
- Linearly controlled language generation (Cheng & Alonso, 理论基础): https://arxiv.org/abs/2405.15454
- Activation Addition (Turner et al.): https://arxiv.org/abs/2308.10248
- Linear representation hypothesis (Park, Choe, Veitch): https://arxiv.org/abs/2401.01315
- Mechanistic interpretability for steering VLAs (Haon et al., 同期工作): https://arxiv.org/abs/2509.00328

---

## 二、VLA 架构剖析

Paper 覆盖两种主流 VLA 架构（Figure 2）：

### (a) Transformer-based VLA（OpenVLA, RT-2）
- Vision encoder（DINOv2 + SigLIP）+ language tokens → shared token space
- Llama 2 backbone autoregressively 预测 **tokenized actions**
- Action 是离散 tokens，再 decode 成 robot commands
- 最终 action `a = φ(x_T)`，只依赖 final layer

### (b) Transformer-Flow-Matching hybrid VLA（π₀, π₀.₅）
- VLM 处理 vision + language
- 独立的 "action expert" transformer 用 **conditional flow matching** 生成连续、高频 action trajectories
- Flow matching layers 通过 cross-attention 关注 transformer 对应层
- `a = φ(x_1, ..., x_T)`，**可以 condition 在中间层 representations 上** —— 这点很关键，意味着对中间层 intervention 会通过 flow matching head 传导到 action

**关键观察**：两种架构都包含 transformer backbone，所以 paper 把分析聚焦在 transformer 的 internal representations 上，证明这种聚焦已经足够影响 final action。

---

## 三、形式化：Feature-Observability 与 Feature-Controllability

### Transformer 前向传播的形式化

**Eq. (1)**：初始 embedding
$$x_0 = E(s), \quad x_0 \in \mathbb{R}^d$$

- `s`: 输入序列（image patches + language tokens + proprioceptive signals）
- `E`: embedding map（包含 vision encoder + token embedding + positional encoding）
- `x_0`: 初始 hidden state
- `d`: representation 维度（OpenVLA 用 Llama 2，d ≈ 2048-4096；π₀.₅ 的 action expert 也有自己的 d）

**Eq. (2)**：层间递推
$$x_{\ell+1} = L_{\ell+1}(x_\ell), \quad \ell = 0, \ldots, T-1$$

- `x_ℓ`: 第 ℓ 层后的 hidden state
- `L_ℓ`: 第 ℓ 个 transformer block（包含 self-attention + FFN + residual + LayerNorm）
- `T`: 总层数

### 两个核心 Definition

**Definition 1 (Feature-Observability)**：feature `ζ ∈ ℝ^n` 在 layer ℓ observable，当且仅当存在 observer map `f_ℓ: ℝ^d → ℝ^n` 使得 `f_ℓ(x_ℓ) = ζ`。

**Definition 2 (Feature-Controllability)**：给定 desired set `D ⊂ ℝ^d`，feature `ζ` 在 layer ℓ controllable，当且仅当存在 controller map `g_ℓ: ℝ^d → ℝ^d` 使得修改后的 `x̃_ℓ = g_ℓ(x_ℓ)` 通过后续层传播后导致 `ζ ∈ D`。

这两个概念直接类比 Kalman 的经典控制理论 [7]：observable 意味着能从 internal state 读出 feature，controllable 意味着能从 internal state 驱动 feature 到 target set。但两者独立 —— 一个 feature 可以 observable 但 not controllable（比如只读但不可写），反之亦然。**最 effective 的 steering 要求同时具备两者**。

---

## 四、Feature Observer 设计

### Linear Observer 结构

**Eq. (3)**：
$$f_\ell(x) := W_\ell x + b_\ell$$

- `W_ℓ ∈ ℝ^{d×n}`: weight matrix（d 是 representation 维度，n 是 feature 维度）
- `b_ℓ ∈ ℝ^n`: bias
- `ζ ∈ ℝ^n`: feature of interest

作者限制 features 为 robot states 和 actions：
- State space: `s = (x, y, z, φ, θ, ψ, g)` —— Cartesian position + roll/pitch/yaw + gripper aperture ∈ [0,1]
- Action: `a = Δs`

这个限制很务实：states/actions 直接可测量、可标注、跨任务通用。更高层的 semantic features（affordances, relational predicates）留作 future work。

### 训练 Observer（Algorithm 1）

**Eq. (4)** 是 binary cross-entropy loss（注意 paper 同时说这是 regression task，这里有个不一致 —— 实际上对 gripper 用 binary probe，对 Cartesian/orientation 用 regression probe，loss 应该是不同的）：

$$W_\ell, b_\ell = \arg\min_{W_\ell, b_\ell} -\sum_{i=1}^N \left[ \zeta^{(i)} \log(W_\ell^\top x_\ell^{(i)} + b_\ell) + (1-\zeta^{(i)}) \log(1 - (W_\ell^\top x_\ell^{(i)} + b_\ell)) \right]$$

- `N`: dataset 大小
- `s^(i)`: 第 i 个 input（prompt + images）
- `ζ^(i)`: 对应的 feature label
- `x_ℓ^(i)`: 把 `s^(i)` 前向传播到 layer ℓ 得到的 activation

**Algorithm 1 流程**：
1. 对每个 `s^(i)`，前向传播到 layer ℓ，收集 `x_ℓ^(i)`
2. 用收集到的 `{(x_ℓ^(i), ζ^(i))}` 训练 linear probe
3. 每个 layer ℓ 都独立训练一个 probe

**Remark 1** 强调 robustness：observer 训练后要经验证 `||f_ℓ(x+ε) - f_ℓ(x)|| < δ`，即小扰动不会导致 feature estimate 爆炸。这点对后续 controller 的可行性是前提。

### Linear observer 的 motivation

线性选择来自 **linear separability hypothesis** [14, 15] —— LLM 里被广泛验证的现象：sentiment、topic、persona 等高层语义特征在中间层 activation space 中线性可分。Paper 假设 VLA 的 transformer component 继承了这个性质，并在实验中验证（Figure 3）。

### 实验结果：Observer 性能（Figure 3）

- **π₀.₅ on Libero**：MAE 显著低于 mean baseline，accuracy 显著高于 majority class baseline
- **OpenVLA on BridgeData V2**：同样表现优秀

这证明：**robot states 和 actions 在 VLA 的 transformer representations 中是线性 encoded 的**。这是 VLA mechanistic interpretability 的基础性发现。

参考 linear probing in LLMs:
- Persona vectors: https://arxiv.org/abs/2507.21509
- Probing for symbolic states in VLA: https://arxiv.org/abs/2502.04558

---

## 五、Feature Controller 设计

### Linear Intervention 结构

**Eq. (5)**：
$$g_\ell(x) := x + u_\ell$$

- `u_ℓ ∈ ℝ^d`: additive perturbation（控制输入）

这是 **additive activation steering** 的标准形式，在 LLM 里被广泛使用 [18, 9, 17]。

### 最小干预优化问题

**Eq. (6)**：
$$u_\ell = \arg\min_{u \in \mathbb{R}^d} \|u\|_2^2 \quad \text{s.t.} \quad f_\ell(x_\ell + u) \in \mathcal{D}$$

- **目标**：最小化 intervention 的 L2 norm（**最小扰动原则**，preserve naturalness）
- **约束**：intervention 后的 representation 通过 observer 映射后落入 desired set `D`

**为什么最小化 ||u||?** 这是 paper 的核心 insight 之一：大扰动会破坏模型的其他 desirable behaviors（naturalness, coherence, closed-loop recovery）。最小扰动只改变目标 feature，leave everything else intact。这与 LLM 里 [3] 的哲学一致。

### 闭式解推导

假设：
- Observer 是线性的：`f_ℓ(x) = W_ℓ^T x + b_ℓ`（注意：这里 paper 写成 `W_ℓ x + b_ℓ`，但维度上 W 应该转置，所以实际上是 `W_ℓ^T x`）
- Desired set 是 1D 区间：`D = [ζ_min, ζ_max]`

我们要找最小 `||u||` 使得 `ζ_ℓ + W_ℓ^T u ∈ [ζ_min, ζ_max]`，其中 `ζ_ℓ = W_ℓ^T x_ℓ + b_ℓ` 是当前 observation。

**几何直觉**：`W_ℓ^T u` 是 `u` 在 `W_ℓ` 方向上的投影。要让 `||u||` 最小同时 `W_ℓ^T u` 等于某个 target 值 `Δζ`，最优 `u` 必须沿 `W_ℓ` 方向（任何垂直分量都是浪费）。

设 `u = c · W_ℓ`，则 `W_ℓ^T u = c · ||W_ℓ||_2^2 = Δζ`，所以 `c = Δζ / ||W_ℓ||_2^2`，即 `u = Δζ · W_ℓ / ||W_ℓ||_2^2`。

**Eq. (7)** 闭式解：
$$u_\ell = (\zeta_{\max} - \zeta_\ell) \frac{W_\ell}{\|W_\ell\|_2^2}, \quad \text{if } \zeta_\ell > \zeta_{\max}$$

$$u_\ell = (\zeta_{\min} - \zeta_\ell) \frac{W_\ell}{\|W_\ell\|_2^2}, \quad \text{if } \zeta_\ell < \zeta_{\min}$$

$$u_\ell = 0, \quad \text{otherwise}$$

- `ζ_ℓ`: 当前 observation
- `ζ_max`, `ζ_min`: 上下界
- 当 observation 已经在区间内，**不做任何干预**（这点很重要 —— 不破坏已经在 desired region 的行为）
- 当 observation 超出上界，往负方向推 `W_ℓ`
- 当 observation 低于下界，往正方向推 `W_ℓ`
- 推的幅度恰好让 observation 落到边界上（不是 overshoot，是 minimal projection）

这个解来自 [3] 的 Theorem 4.1。它的优雅之处在于：
1. **闭环形式**：无需在线优化，O(d) 计算
2. **最小能量**：在所有可行 intervention 中 L2 norm 最小
3. **Hard constraint**：保证 observation 落入 desired set（不是 soft penalty）

参考理论：
- Linearly controlled language generation: https://arxiv.org/abs/2405.15454
- Inference-time intervention (ITI): https://arxiv.org/abs/2306.03341

---

## 六、Observer + Controller 集成（Algorithm 2）

**Algorithm 2** 把 observer 和 controller 嵌入 forward pass：

```
Input: s, L_O (observable layers), L_C (controllable layers),
       W_ℓ, b_ℓ for ℓ ∈ L_O, ζ_min, ζ_max
Output: x_1, ..., x_T

1: x_0 ← E(s)
2: for ℓ = 1, ..., T do
3:   x_ℓ ← L_ℓ(x_{ℓ-1})
4:   if ℓ ∈ L_O then  // observe
5:     ζ_ℓ ← W_ℓ^T x_ℓ + b_ℓ
6:     if ℓ ∈ L_C then  // control
7:       u_ℓ ← closed-form from Eq. (7)
8:       x_ℓ ← x_ℓ + u_ℓ
9:     end if
10:  end if
11: end for
```

**关键设计选择**：
- `L_C ⊆ L_O`：controller 必须依赖 observation，所以 controllable layers 必须是 observable layers 的子集
- 计算 overhead 极小：observer 是矩阵乘法，controller 是 vector scale + add，相比 transformer layer 的 attention + FFN 几乎可忽略
- Online：无需 fine-tuning，无需 retraining，inference time 直接生效

**Remark 2** 强调 closed-loop challenge：LLM 是 open-loop generation（一次生成完事），VLA 是 closed-loop（每个 action 改变 environment，environment 反馈成下一个 input）。只要 intervention 不让 input 跑出 probe training data 的分布，方法就 transfer。实验证明这点成立。

---

## 七、实验结果深度分析

### 7.1 Feature-Observability 实验（Section V-A）

#### Robustness 实验（Figure 4）

加扰动 `x_ℓ + α`，看 action 的 mean change：
- **π₀.₅**：α 增大 → action 平滑增大。**所有 features robust**。
- **OpenVLA**：delta yaw **不 robust**（α 增大但 action 不按预期单调变化）；delta gripper 有 ordering 但不如 π₀.₅ 干净。

**Layer depth 效应**：
- Earlier layers 的 perturbation 更有效
- 越深，representation 的 L2 norm 越大（Figure 4 bottom）
- 固定 α 在大 norm 的 representation 上相对效应变小
- 这解释了为什么 π₀.₅ 和 OpenVLA 需要不同量级的 α

**Intuition**：early layers 是 "feature construction" 阶段，small perturbations 还没被 norm 放大；later layers 是 "feature readout" 阶段，representation 已经定型且 norm 大。

#### Classifier Image Space 可视化（Figure 5）

- Un-intervened：representations 散布在 classifier image space
- Fixed perturbation `α W_ℓ`：representations 偏移但仍可能超出 bounds
- **Proposed controller (Eq. 7)**：**所有 representations 严格落入 [ζ_min, ζ_max]**

这是 hard constraint 的直接体现 —— 不是 soft regularization，是 mathematical guarantee。

### 7.2 Feature-Controllability 实验（Section V-B）

#### Gripper State 控制（Figure 6, 8）

实验设置：Libero spatial suite，10 tasks，10 rollouts/task，NVIDIA 5090 GPU
- Constraint：gripper 保持 open 或 closed
- 比较三种方法：
  1. **No intervention**：基线
  2. **Prompting**：用 favorable initial condition（要求 open 就给 open gripper 起始状态）
  3. **Control (ours)**：用 Eq. 7 的 minimal controller

**结果**：
- Control 方法 **near-perfect constraint satisfaction**（接近 100%）
- 同时 **success rate > 90%**
- Prompting 也能 steer，但成功率不如 control
- No intervention 完全无法满足 constraint

#### End-effector Height 控制（Figure 7, 8）

- Constraint：end-effector 保持在初始高度以上/以下
- Control 方法 **near-perfect constraint satisfaction**
- Success rate 有 **modest drop**（因为 constrained task 严格难于 unconstrained）
- 作者假设：更 robust 的 base model 配合 good recovery behaviors 可以消除这个 drop

#### End-effector Speed 控制（Figure 9, 10）

- Speed 是 **derived quantity**：`v = ||(Δx, Δy, Δz)|| / dt`，不直接是 model output
- 结果：
  - **Slow down** 可靠实现
  - **Speed up** 不准确
  - 原因：training data 缺乏 fast speed regime，与 [5] 报告一致
- Success rate 几乎完全保持

**Critical insight**：即使 feature 不直接是 model output，只要它 linearly observable，controller 就能 steer。这暗示 VLA 的 internal representations 编码了 derived/relational quantities。

### 7.3 综合数据表（推断）

| Feature | Architecture | Dataset | Constraint Satisfaction | Success Rate (Control) | Success Rate (Baseline) |
|---------|--------------|---------|------------------------|------------------------|------------------------|
| Gripper state | π₀.₅ | Libero | ~100% | >90% | varies |
| Gripper state | OpenVLA | BridgeData V2 | high | high | varies |
| EE height | π₀.₅ | Libero | ~100% | modest drop | high |
| EE speed (down) | π₀.₅ | Libero | reliable | maintained | high |
| EE speed (up) | π₀.₅ | Libero | less accurate | maintained | high |

---

## 八、与 LLM Activation Steering 的关系与差异

### 传承关系

| LLM 概念 | VLA 对应 | 传承 |
|---------|---------|------|
| Linear representation hypothesis [14] | Linear observability of states/actions | 直接 port |
| Activation addition [18] | `g_ℓ(x) = x + u_ℓ` | 直接 port |
| Minimal intervention [3] | Eq. (6) 优化问题 | 直接 port |
| Closed-form controller [3] Thm 4.1 | Eq. (7) | 直接 port |
| Linear probes [16] | Linear observer (Eq. 3) | 直接 port |

### 关键差异

1. **Closed-loop vs open-loop**：LLM 生成是一次性的，VLA 每个 action 改变 environment，environment 反馈成下一个 input。Method 必须在 closed-loop 中保持 stable。
2. **Continuous action space**：LLM 是 discrete tokens，VLA 是 continuous actions（即使 OpenVLA 用 tokenized actions，最终也是连续 robot commands）。
3. **Hybrid architectures**：π₀.₅ 有 flow-matching head，intervention 通过 flow matching 传导到 action，传导路径比 LLM 复杂。
4. **Physical safety**：VLA intervention 直接影响 physical world，safety stakes 高得多。
5. **Feature granularity**：LLM steering 通常针对 sentiment/topic/persona，VLA 这里针对 low-level robot states/actions，更高层 semantic features 留作 future work。

---

## 九、Limitations 与 Future Work

Paper 自述的 limitations：
1. **需要 labeled data** 训练 observer —— 未来可探索 SAEs（Sparse Autoencoders）做 unsupervised feature discovery
2. **只覆盖 transformer component** —— diffusion/flow-matching head 的 interpretability 未触及
3. **只控制 low-level features** —— task goals, affordances, spatial relationships 等高层 semantic features 未探索
4. **缺乏 safety guarantees** —— intervention 效果的 principled bounds 未建立

### 我的联想：潜在的扩展方向

1. **SAE + Activation Steering 结合**：Anthropic 的 SAE 工作（https://transformer-circuits.pub/2024/scaling-monosemanticity/）发现 LLM 里 monosemantic features，如果 VLA 也能用 SAE 找到 monosematic features，observer 就不需要 labeled data 了。
2. **Mechanistic finetuning** [13]（https://arxiv.org/abs/2511.22697）：识别 task-relevant attention heads 并 fine-tune 它们，与本文的 inference-time intervention 是互补的 —— 一个改 weights，一个改 activations。
3. **Task reconstruction via text latents** [10]（https://arxiv.org/abs/2505.03500）：从 π₀ 的 hidden states 提取 task latents，可以 reconstruct 或 blend skills。这本质上是更高层 feature 的 observability。
4. **Hypersteer** [17]（https://arxiv.org/abs/2506.03292）：用 hypernetwork 在 scale 上做 activation steering，可以移植到 VLA 处理 multi-task multi-constraint 场景。
5. **Transporting activations** [16]（https://arxiv.org/abs/2502.05471）：用 optimal transport 在 activation space 间 transport，可能比 additive intervention 更适合 multimodal distributions。
6. **ReFT** [21]（https://arxiv.org/abs/2404.03592）：representation finetuning，intervention 是 learnable 的，可以适配 VLA。

---

## 十、Intuition 总结

### 1. 为什么 linear 就够用？

Linear representation hypothesis 的几何直觉：high-dimensional transformer representations 中，semantic features 往往对应 **directions**（方向），feature 强度是 projection 长度。所以一个 linear probe 就能 extract，一个 linear shift along feature direction 就能 steer。这是 LLM 里反复验证的现象，paper 证明 VLA 继承了这个性质。

### 2. 为什么 minimal intervention 重要？

Representation space 是 high-dimensional（d ≈ 2048-4096），features 是多对多的。Push 一个 feature direction 会 **leak** 到其他 features。Minimal L2 intervention 最小化 leak，preserve 其他 desirable behaviors。这就是 paper 反复强调的 "naturalness preservation"。

### 3. 为什么 closed-loop 能 work？

直觉上 closed-loop 会 destabilize intervention（intervention → action → environment → input → intervention），但实验显示 stable。原因可能是：
- Intervention 是 **state-dependent** 的（Eq. 7 依赖当前 observation），相当于 feedback controller
- 当 observation 进入 desired region，intervention 自动归零
- 所以这是一个 **self-correcting** 的 closed-loop controller

### 4. 为什么 earlier layers 更 effective？

Early layers 的 representation norm 小，fixed α 的相对效应大。同时 early layers 是 feature **construction** 阶段，干预这里有 leverage；later layers 是 feature **readout** 阶段，representation 已经定型。这与 LLM 里 steering 的经验一致 —— middle layers 通常最 effective。

### 5. 为什么 speed-up 比 speed-down 难？

Training data 里 fast motions 稀少，所以 representation space 的 "fast speed direction" 附近 density 低。Observer 在这里 extrapolation 不可靠，controller 推过去后 model 可能 fall off the manifold。这是 OOD 问题，不是 method 本身的局限。

---

## 十一、我的整体评价

### Strengths
1. **Formalization 干净**：把模糊的 "activation steering" 升级为 control-theoretic 的 observability/controllability，提供了清晰的 mathematical language。
2. **Closed-form controller 优雅**：无需在线优化，O(d) 计算，real-time friendly。
3. **跨架构验证**：在 transformer-based (OpenVLA) 和 hybrid (π₀.₅) 两种主流架构上都验证，generalizable。
4. **Closed-loop 实验扎实**：在 Libero 上做真实 closed-loop rollout，不只 open-loop generation eval。
5. **Naturalness preservation**：minimal L2 intervention 是 explicit objective，不只 empirical observation。

### Weaknesses / Open Questions
1. **Linear observer 的 loss 不一致**：Eq. 4 是 binary cross-entropy，但 paper 说有 regression probe。Implementation 细节不清。
2. **Layer selection 未充分讨论**：L_O 和 L_C 怎么选？paper 只说 "best performant layer"，没有 principled criterion。
3. **Multi-feature conflicts**：同时控制多个 features 怎么办？Eq. 6 是 single-feature，multi-feature 需要更复杂的 optimization。
4. **Safety bounds 缺失**：Remark 1 的 robustness check 是 empirical，没有 theoretical guarantees。
5. **High-level features 未触及**：只控制 states/actions，task-level reasoning 没碰。
6. **OOD robustness 未充分研究**：speed-up 的失败暗示 OOD 区域的脆弱性。

### 对 VLA field 的影响

这篇 paper 是 VLA mechanistic interpretability 的 **奠基性 formalization**。它把 LLM 里成熟的概念（linear probing, activation steering, minimal intervention）严格 port 到 VLA，证明 VLA 继承了 LLM 的 linear representation property，同时识别出 closed-loop 是新挑战。后续工作很可能沿着 SAE-based feature discovery、multi-feature control、high-level semantic steering、safety guarantees 这几条线展开。

特别是与同期工作 [5]（https://arxiv.org/abs/2509.00328）和 [13]（https://arxiv.org/abs/2511.22697）形成互补：本文是 inference-time linear intervention，[5] 是更 ad-hoc 的 activation overwrite，[13] 是 weight-level fine-tuning。三者合起来构成 VLA steering 的完整 toolkit。

---

## 参考文献链接汇总

**核心方法论文**：
- Linearly controlled language generation (理论基础): https://arxiv.org/abs/2405.15454
- Activation Addition: https://arxiv.org/abs/2308.10248
- Inference-time intervention (ITI): https://arxiv.org/abs/2306.03341
- Linear representation hypothesis: https://arxiv.org/abs/2401.01315
- Geometry of categorical concepts in LLMs: https://arxiv.org/abs/2410.16740
- Persona vectors: https://arxiv.org/abs/2507.21509
- ReFT (Representation finetuning): https://arxiv.org/abs/2404.03592
- Transporting activations: https://arxiv.org/abs/2502.05471
- Hypersteer: https://arxiv.org/abs/2506.03292

**VLA 相关**：
- π₀: https://arxiv.org/abs/2410.24164
- π₀.₅: https://arxiv.org/abs/2504.16054
- OpenVLA: https://proceedings.mlr.press/v270/kim25c.html
- RT-2: https://arxiv.org/abs/2307.15818
- Mechanistic interpretability for steering VLAs: https://arxiv.org/abs/2509.00328
- Mechanistic finetuning of VLAs: https://arxiv.org/abs/2511.22697
- Probing VLA for symbolic states: https://arxiv.org/abs/2502.04558
- Task reconstruction via text latents: https://arxiv.org/abs/2505.03500
- Libero benchmark: https://arxiv.org/abs/2306.03310
- BridgeData V2: https://arxiv.org/abs/2308.12952

**基础架构**：
- Attention is all you need: https://arxiv.org/abs/1706.03762
- Kalman filtering: https://en.wikipedia.org/wiki/Kalman_filter

**Sparse Autoencoders (future work 方向)**：
- Scaling monosemanticity (Anthropic): https://transformer-circuits.pub/2024/scaling-monosemanticity/
- SAEs for interpretability: https://arxiv.org/abs/2309.08600
