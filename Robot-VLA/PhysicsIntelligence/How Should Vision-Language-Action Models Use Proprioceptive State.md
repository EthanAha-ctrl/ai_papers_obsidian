---
source_pdf: How Should Vision-Language-Action Models Use Proprioceptive State.pdf
paper_sha256: f3c5dc6372feb878a86f4ca546520b31f763e9a56c5fb62bba3d9a4bd829c840
processed_at: '2026-08-19T11:37:23-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用大白话再讲一遍，把那些公式和表格的 intuition 翻译成人话。

---

## 这个 paper 在干嘛

VLA 这个圈子现在有个怪现象：所有人都在用 robot 的 proprioceptive state（关节角、末端位置、夹爪状态），但**每个人接法都不一样**，而且谁也说不清哪种接法好，因为大家的 backbone、数据、action 表示全都不一样，根本没法比。

这篇 paper 做的事情特别简单粗暴：**我把别的全锁死，只让你动"state 怎么接"这一个变量**，然后系统地扫一遍，看看到底什么有用什么没用。

backbone 用的是 π0.5（Physical Intelligence 的 flow-matching VLA），数据用 RoboCasa365，action 表示固定，训练 recipe 固定，eval protocol 固定。在这个固定 scaffold 上，他们实现了 5 种不同的 state 接口，然后对比。

参考: https://arxiv.org/abs/2504.16054 (π0.5)

---

## 5 种 state 接口，到底有什么区别

你把 VLA 想成一个三段式流水线：

```
[images + language] → VLM backbone → [context] → Action Expert → action chunk
```

state 可以从三个地方塞进去，而且塞的形式可以不同。这篇 paper 实现了 5 种：

### 1. State Prompt (sp) —— 把 state 变成文字

π0.5 原版做法。把 16 维 state 的每个数字量化到 256 个 bin，然后用 text tokenizer 序列化成 ~66 个 token，拼到 language prompt 里。

**人话**：state 变成一段文字描述，跟 "pick up the cup" 这种 instruction 一起喂给 VLM。

特点：零额外参数，但 66 个 token 要过 18 层 VLM，training 算力爆炸（1114 GFLOPs/sample）。而且只能用当前帧，因为你没法在 text prompt 里自然地 encode "5 帧前我在哪"。

### 2. VLM Prefix (vp) —— state embedding 塞进 VLM

用一个两层 MLP 把 state 投影成 continuous token，塞到 VLM 的 prefix 里，排在 image 和 language token 后面。

**人话**：state 不再假装是文字了，直接以 embedding 形式参与 VLM 的 multimodal attention。

state 先和 image、language 一起做 context modeling，然后通过这个 context 间接 condition action generation。

### 3. Action Prefix (ap) —— state 直接塞进 action expert

同样用 MLP 投影，但不塞进 VLM，而是塞到 action expert 的 causal suffix 里，放在 noisy action tokens 前面。

**人话**：state 绕过 VLM，直接告诉 action expert "我现在在哪，你生成 action 时考虑这个"。

这是最直接的 state → action 路径。

### 4. State Expert (se) —— state 有自己的 transformer

给 state 配一个独立的 transformer stream，平行于 VLM 和 action expert，有自己的 sequence modeling 能力，和 action module 在生成时交换信息。

**人话**：state 既不被压进 VLM，也不被折叠进 action suffix，而是有自己的"大脑"处理。参数最多（199M），但 marginal compute 极低。

### 5. Feature Modulation (fm) —— state 通过 FiLM 调制 action feature

state 不作为 sequence token，而是作为外部 memory，action expert 每一层通过 cross-attention 读 state，预测一个 scale $\gamma$ 和 shift $\beta$，然后调制 action feature：

$$\text{Mod}(z; S) = (1 + \gamma(z, S)) \odot z + \beta(z, S) \tag{4}$$

变量解释：
- $z$：action expert 某一层的 feature
- $\gamma(z, S)$：学到的 per-feature scale
- $\beta(z, S)$：学到的 per-feature shift
- $\odot$：Hadamard product

**人话**：state 不参与 attention，而是像一个"音量旋钮"一样，调节 action feature 的强度和偏移。

参考 FiLM 原始 paper: https://arxiv.org/abs/1709.07871

---

## 三个核心问题，三个核心答案

### RQ1: state 到底有用没用？

**Table 1 的 atomic macro 数据**：

| Interface | Macro SR | Δ vs no-state |
|-----------|----------|---------------|
| No state  | 54.6%    | -             |
| sp        | 57.7%    | +3.1 †        |
| vp        | 56.8%    | +2.1          |
| ap        | 55.7%    | +1.1          |
| se        | 57.6%    | +3.0          |
| fm        | 57.6%    | +2.9          |

†表示 paired task-bootstrap 95% CI [0.2, 6.1] exclude zero，是唯一统计显著的。

**人话**：state 确实有用，但只有 sp 达到了统计显著的 macro gain。其他的都是"有正向趋势但不够 significant"。

但如果你只看 macro average，就被骗了。**真正炸裂的发现是 family-level ranking 会 reverse**：

| Family | 描述 | Best interface | Worst interface |
|--------|------|----------------|-----------------|
| A (rearrangement) | pick-and-place, 大范围 positioning | **sp** 68.7% (+7.0) | se -0.1 |
| B (articulated) | 持续 contact, motion-phase | **vp** 68.8% (+6.1) | sp +1.6 (中游) |
| C (knob/switch) | 高精度局部 control | **se** 42.8% (+3.3) | vp -1.2 (有害!) |

**intuition**：

- **Family A 是大范围移动**：你要把东西从一个地方搬到另一个地方，"我现在在哪"这个信息本质上是一个**离散的语义状态**——"我在柜子左边还是右边"。这种信息最适合变成文字（sp），因为它和 language instruction 是同一类东西。
  
- **Family B 是持续 contact**：你在开抽屉、转把手，需要**平滑的 continuous state** 来做 motion-phase modeling。sp 把 state 量化成 256 bins 反而丢失了精度。continuous interface（vp/se/fm）都好。

- **Family C 是高精度局部操作**：你在拧旋钮、按开关，workspace 很小，要求 state→action 的**精确对齐**。这种情况下 state 不能经过 VLM 的"smooth averaging"，必须直接进 action expert。se 最好，vp 反而有害（VLM 把 state 信息"平均化"了，丢失了精度）。

**这条结论的人话**：**没有一个 universal best interface**。benchmark-wide average 会 hide 掉这种 reversal pattern。你 design VLA 时要看你的 task 是哪一类。

---

### RQ2: state history 有用吗？要多少帧？

**Figure 3 的 sweep 结果**：

从 K=1 扫到 K=96，发现：
- K=1 → K=8：性能提升
- K=8 → K=96：性能下降，尤其 Family C 暴跌

**人话**：**短 history 有用，长 history 有害**。

这个结论本身不意外，但 paper 做了一个特别聪明的 control，排除了一个 confounder。

**Confounder**：你增加 history length K，自然也增加了 conditioning tokens 数量。那性能提升到底是"temporal information 有用"还是"更多 tokens = 更多 capacity"？

**Slot-matched control**：把真正的 8 帧 history $(s_{t-7}, ..., s_t)$ 替换成 8 份当前 state 的 copy $(s_t, s_t, ..., s_t)$。这样 slot count 不变，但 temporal variation 被消除了。

**Table 2 的关键数据**（composite task, EEF-pose state, AP interface）：

| 设置 | SR |
|------|-----|
| K=1 (单帧) | 28.2% |
| K=8, repeat-current (8 份当前 state) | 30.8% |
| K=8, genuine history (真正 8 帧) | 39.0% |

**人话**：
- 从 1 帧到 8 份重复当前 state：+2.6%（这点 gain 纯粹来自 capacity）
- 从 8 份重复到真正 8 帧 history：**+8.2%**（这点 gain 纯粹来自 temporal variation）

paired task-bootstrap 95% CI exclude zero，所以这 8.2 points 是真的来自"时间序列里有信息"，不是来自"多了几个 token 位置"。

**为什么长 history 有害？intuition**：

1. **冗余**：相邻帧的 state 几乎一样，长 history 里 90% 是 redundant 信息，attention 要浪费 capacity 去"理解"这些 redundancy。

2. **Stale 信息干扰**：96 帧前的 state 可能已经和当前要做的 action 无关，但 model 还得 attend 到它，容易学到 shortcut（直接 copy 自己 recent trajectory，ignoring visual scene）。这是 De Haan et al. 2019 和 Wen et al. 2020 早就警告过的 "causal confusion in behavioral cloning"。

3. **State dominate context**：K=96 时 state tokens 是 96×1024 维，远超原本的 784 个 prefix tokens，整个 context 被 state 淹没。

参考: https://papers.nips.cc/paper/2019/hash/929bfb1c1efa8de1b389909c0be3d63a-Abstract.html (Causal Confusion in Imitation Learning)

---

### RQ3: state 应该从哪里注入？

**这是 paper 最 striking 的发现，我称之为 "routing crossover"**。

| 条件 | 最好 interface | 数据 |
|------|----------------|------|
| K=1, composite | **vp** (VLM side) | 34.4% vs ap 28.2% |
| K=8, composite | **ap** (action side) | 39.0% vs vp 33.8% |

**人话**：

- **单帧 state 应该走 VLM side**
- **多帧 state history 应该走 action side**

这是一个 crossover，同一个 state 信号，根据 temporal budget 不同，最优 injection route 会**翻转**。

**为什么会这样？intuition**：

**K=1 时 VLM side 好的原因**：单帧 state 的价值不在于它告诉你"dynamic 信息"，而在于它**contextualize 了 visual-language representation**。比如 state 告诉 model "我现在手在柜子左边"，这个信息和 image 里看到的柜子位置结合起来，帮助 VLM 理解"我在任务的哪个阶段"。这种 contextual 作用最适合让 state 参与 VLM 的 multimodal attention。

Appendix A.2 的 probe 直接证实了这点：在 VLM 最后 6 层，language-to-image attention redistribution 在 vp1 下是 17.3%，说明 state 确实在改变 VLM 内部的 attention pattern。

**K=8 时 action side 好的原因**：多帧 state 提供的是 **temporal evolution**，本质上是 velocity 信息（隐式地，state 的 temporal derivative ≈ velocity）。而 flow-matching action expert 做的事情恰恰是预测 velocity field（从 noisy action 到 clean action 的 velocity）。所以 state history 的 temporal derivative 和 action expert 的 prediction target **天然对齐**。

把这个信号直接送给 action head，比先通过 VLM compress 成 context 再 condition action 要**更直接、loss 更少**。

Appendix A.3 的 flow probe 定量证实了这点：

| Interface | Final alignment $\cos(c_t, r_t)$ | Normalized magnitude $\|c_t\|/\|r_t\|$ |
|-----------|----------------------------------|----------------------------------------|
| vp1       | 0.245                            | 0.297                                  |
| ap1       | 0.079                            | 0.174                                  |
| ap8       | 0.270                            | 0.382                                  |

变量解释（公式 5）：
- $c_t = \hat{a}_t^{\text{true}} - \hat{a}_t^{\text{off}}$：state-conditioned correction（true state forward pass 与 state-off forward pass 的 action 差）
- $r_t = a^* - \hat{a}_t^{\text{off}}$：residual from state-off to expert action（理想 correction 方向）
- $\cos(c_t, r_t)$：correction 方向是否和理想方向对齐
- $\|c_t\|_2 / \|r_t\|_2$：correction 强度

**人话**：
- ap1 的 correction 又弱又偏（alignment 0.079, magnitude 0.174）
- ap8 的 correction 又强又准（alignment 0.270, magnitude 0.382）
- 从 ap1 到 ap8，alignment 提升 +0.191（95% CI [+0.143, +0.239]），magnitude 提升 +0.208

而 vp1→vp8 基本没变化。**history 让 action-side state 的 correction 变得既更 aligned 又更强，但这种提升在 VLM-side 完全没出现**。

---

## PrepareToast case study：这个 gain 到底在 task 的哪个部分出现

这个 case study 把 aggregate result localize 到具体 subtask transition，非常 intuitive。

**PrepareToast**：放两个 task-relevant items 到 cabinet，然后返回 cabinet 关闭它。四个 monotonic milestone：

- S1: 第一个 item 放好
- S2: 两个 items 都放好
- S3: cabinet 重新关闭
- S4: gripper released

**AP1 vs AP8 的 reach rate**：

| Milestone | AP1 | AP8 | 差距 |
|-----------|-----|-----|------|
| S1        | 90% | 96% | +6   |
| S2        | 64% | 68% | +4   |
| S3        | 30% | 56% | **+26** |
| S4        | -   | -   | +26  |

**人话**：两个 policy 在前期 placement 阶段几乎一样。但一到"放完东西必须回头去关柜子"这个 late transition，AP8 就把 AP1 甩开 26 个 points。

这非常符合 intuition：placement 阶段是"看到东西 → 去拿 → 放下"，主要是 visual-driven，state history 帮助不大。但"放完东西回头关柜子"是一个**subgoal 切换**，需要 model 知道"我已经放完了，现在要切换到关柜子模式"。这个"我已经放完了"的信息，恰恰来自 state 的 temporal evolution（state 从"在柜子前"变成"在柜子外"，再变成"往回走"），而不是当前单帧。

更精细的 probe（公式 6）：固定 AP8 checkpoint，把有序 history 换成 8 份当前 state copy，测 action 变化：

$$D_q = \|\hat{\mathbf{a}}_q^{\text{true}} - \hat{\mathbf{a}}_q^\text{repeat}\|_2 \tag{6}$$

| 时机 | Action 变化 $D_q$ |
|------|------------------|
| Within a stage | 0.198 |
| Before boundary | 0.361 |
| **At boundary** | **1.033** |
| After boundary | 0.748 |

**人话**：policy 在 progress boundary 处对 temporal variation 最敏感（5.2 倍于 within-stage）。这定量证明 history 的价值集中在"subgoal 切换"那些时刻。

---

## 所以 design VLA 时该怎么办

把所有发现 compose 起来，paper 给出的 design default 是：

> **Inject the current frame into the VLM, route short histories to the action head, and validate anything deeper.**

人话翻译：

1. **当前帧 state → 走 VLM side**（sp 或 vp）。它的作用是 contextualize，帮助 VLM 理解"我在哪、我在做什么阶段"。

2. **Short history (K=8) → 走 action side**（ap 或 se）。它的作用是提供 temporal evolution，直接 inform velocity field prediction。

3. **更长的 history → 不要 raw stack**。要么用 compressed latent（memory token、recurrent state），要么干脆别用。Raw 96 帧 history 会 hurt 性能，尤其 precision task。

4. **看你的 task 类型选 interface**：
   - 大范围 pick-and-place → 偏向 sp/vp（VLM side, discrete-ish）
   - 持续 contact articulated object → 偏向 vp/se/fm（continuous, VLM or action side）
   - 高精度 knob/switch → 偏向 se（action side, independent stream）

5. **Compute constrained 时**：sp 虽然参数为零但算力爆炸（66 个 text token 过 18 层 VLM）。se 虽然 199M 参数但 marginal compute 极低（2.6/0.7 GFLOPs）。如果你要 scale history length, se 更友好。

6. **永远不要只看 macro average**。一个 benchmark-wide 数字会完全 hide 掉 family-level reversal。你 design 时一定要按 control semantics 分 family eval。

---

## 这篇 paper 真正的价值

它没有 propose 新 SOTA，没有新架构，没有新数据集。它做的事情是**把一个 ad hoc 的 design decision 变成一个 measurable 的 science problem**。

在 deep learning 领域，太多 paper 是"我提出 X，X 比 baseline 好 Y points，所以 X 好"。但 X 和 baseline 的差异里，到底哪个 component 起作用？没人说得清。这篇 paper 的 discipline 是：**每个 claim 都要 isolate 一个变量**，配 slot-matched control，配 paired bootstrap CI，配 fixed-checkpoint probe。

这就是 Karpathy 你一直强调的"做 science"的精神。这篇 paper 给了 community 两个 reusable 的东西：
1. 一组 **testable design principles**（current frame → VLM, short history → action, 长 history 别用 raw）
2. 一套 **reusable evaluation protocol**（family partition + slot-matched control + paired bootstrap + flow probe）

后续的 state-aware VLA 工作可以直接在这个 protocol 上 audit 自己的设计。

---

## Limitations 和我自己的额外思考

Paper 自己承认：
- **No real-robot validation**：全在 RoboCasa simulation。Real robot 的 proprioception noise、calibration drift 可能改变结论。
- **Purely kinematic state**：没考虑 force、tactile。Contact-rich task 里 F/T history 可能比 joint angle history 更重要。

我自己额外想到的：

- **Single backbone**：只在 π0.5 (flow-matching + VLM) 上测。如果换成 OpenVLA 那种 tokenized autoregressive action，结论可能不同。Flow-matching 的 velocity field prediction 天然和 state temporal derivative 对齐，这可能放大了 action-side history 的优势。Autoregressive action 可能没有这种对齐。

- **State dimension 固定 16**：对 high-DoF humanoid（30+ DoF）是否成立未知。高维 state 下 sp 的 66 tokens 可能变成 200+ tokens，cost 更爆炸；而 se/fm 的 cost 几乎不变。

- **K=8 是 empirical sweet spot for RoboCasa**：不同 task timescale 可能不同。RoboCasa 的 control frequency 和 task duration 决定了 8 帧 ≈ 多少秒的 history。换一个 sim 或 real robot 可能 K=4 或 K=16 更好。

- **Most comparisons 是 single seed**：只有 sp 的 atomic macro gain 达到 paired bootstrap 95% CI exclude zero。其他都是 single-seed point estimates。严格说只是 directional pattern 而非 statistically established。但这个 paper 是 exploratory study，这种 discipline 已经比 99% 的 VLA paper 好了。

- **Routing crossover 的 mechanism 没完全建立**：paper 说这是 directional pattern 而非 causal mechanism。在 joint-angle state 下 K=8 时 ap 和 vp converge（36.2% vs 35.8%），说明这个 crossover 可能不是绝对的，依赖 state coordinate system。

---

## 最终的 mental model

如果让我给一个一句话总结的 mental model：

**State 在 VLA 里有两个根本不同的角色**：
- **"我在哪"**（contextual, K=1, VLM side）—— 帮助 multimodal understanding
- **"我在动"**（dynamic, K>1, action side）—— 直接 inform velocity prediction

这两个角色不该用同一种 interface 处理。你 design VLA 时应该考虑用一个 **hybrid interface**：current frame 走 VLM，short history 走 action expert。

这就像人一样：你"知道自己在厨房"（proprioception 作为 context）和"你正在伸手去拿杯子"（proprioception 作为 dynamic feedback）是两个不同的大脑过程，不应该用同一个 neural pathway 处理。

参考链接汇总：
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- GR00T N1: https://arxiv.org/abs/2503.14734
- RoboCasa365: https://arxiv.org/abs/2603.04356
- Flow Matching: https://arxiv.org/abs/2210.02747
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- FiLM: https://arxiv.org/abs/1709.07871
- Causal Confusion in IL: https://papers.nips.cc/paper/2019/hash/929bfb1c1efa8de1b389909c0be3d63a-Abstract.html
- Fighting Copycat Agents: https://papers.nips.cc/paper/2020/hash/2cbfcdc66f3f1d24b6ea1c8f8c04b3c8-Abstract.html
- MemoryVLA: https://arxiv.org/abs/2508.19236

---

# VLA Models如何使用Proprioceptive State的深度解析

这篇paper做了一件非常好的science工作: 在一个固定的π0.5 flow-matching VLA backbone上，把"state怎么用"这件看似ad hoc的设计决策decompose成三个measurable axes，然后用controlled experiments给出testable的答案。这正好是deep learning领域最缺的那类ablation study。

参考链接:
- π0.5 paper: https://arxiv.org/abs/2504.16054
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645  
- GR00T N1: https://arxiv.org/abs/2503.14734
- RoboCasa: https://robocasa.org/
- Flow Matching: https://arxiv.org/abs/2210.02747
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

---

## 1. 为什么这个paper重要

VLA领域目前有个很尴尬的现状: 几乎所有模型都把proprioceptive state作为输入, 但每个模型接的方式都不一样, 而且这种差异与backbone、pretraining、data、action representation纠缠在一起, 完全incomparable。

具体来看三大主流做法的分歧:

- **π0.5** (Physical Intelligence, 2025): 把16维state量化成256 bins, 序列化成~66个text tokens拼到language prompt里, 走VLM text entry。**discrete + VLM-side + single-frame**
- **OpenVLA-OFT** (Kim et al., 2025a): 把state连续投影成embedding拼到language-model sequence里。**continuous + VLM-side**
- **GR00T N1** (Bjorck et al., 2025): state embedding直接喂给action head。**continuous + action-side**

这些设计选择互相confounded, 比如你要说是"VLM-side好"还是"discrete好"还是"π0.5这个backbone好", 根本无法从report的numbers里推断出来。Paper Section 2 Related Work那段说得很清楚: no prior study moves the same state signal across interfaces, history depths, and injection routes with everything else fixed。

这种controlled comparison正是Karpathy你在nanoGPT、micrograd里强调的那种"做science"的精神: 把变量isolate, 在固定其他条件的情况下sweep一个变量。

---

## 2. Methodology详解

### 2.1 Policy class (公式1-2)

决策时刻t, 接收 visual observation $o_t \in O$, language instruction $\ell \in \mathcal{L}$, 和最近K步的state窗口:

$$S_t^{(K)} = (s_{t-K+1}, \ldots, s_t), \quad s_t \in \mathbb{R}^{d_s} \tag{1}$$

变量解释:
- $s_t$: 单帧state, $d_s = 16$ (3 EEF position + 4 EEF quaternion + 3 base position + 4 base quaternion + 2 gripper joints)
- $K$: history length, K=1表示只用当前帧
- $S_t^{(K)}$: 有序的state序列

输出连续action chunk:

$$\hat{\mathbf{a}}_{t:t+H-1} = \pi_\theta(o_t, \ell, S_t^{(K)}) \in \mathbb{R}^{H \times d_a} \tag{2}$$

变量解释:
- $H$: action chunk length (这里A=50 action tokens)
- $d_a$: 每步action维度
- $\pi_\theta$: 由VLM backbone + flow-matching action expert组成的policy

关键insight: 所有state interface共享相同的prediction target, 只在$S_t^{(K)}$如何encode和inject上不同。这是controlled experiment的根本保证。

### 2.2 五种State Interfaces的架构细节

这是paper最核心的技术部分。看Figure 1能清楚看到data flow差异。我用文字描述一下每种interface在π0.5 scaffold里的具体wiring。

#### (1) State Prompt (sp)

最简单也最"VLM-native"的方式。把当前state的16个dimension各自独立quantize到256 bins, 然后用现有的text tokenizer序列化, 产生~66个prompt tokens拼到language instruction里。

**关键特性**:
- 零额外可训练参数 (所有state token通过language embedding space)
- 只能用当前帧 (因为text prompt没法encode temporal order而不引入更多结构)
- 训练时这些state token需要走完整VLM的18层Transformer

#### (2) VLM Prefix (vp)

Per-frame state通过两层projector形成continuous token:

$$h_i = \phi_2(\text{Swish}(\phi_1(s_i))) \tag{3}$$

变量解释:
- $\phi_1$: 把state (zero-padded from 16 to 32) lift到width $d$
- $\phi_2$: 保持width $d$
- 对vp: $d = 2048$ (匹配VLM hidden width)
- 对ap/se/fm: $d = 1024$ (匹配action expert hidden width)
- Swish是activation function

这些state tokens插入到bilateral VLM prefix里, 在image和language tokens之后。State先参与multimodal context modeling (每个image和language token都能attend到state), 然后通过conditioning prefix间接影响action generation。

#### (3) Action Prefix (ap)

State tokens ($d=1024$) 放在causal action suffix里, 在noisy action tokens之前。这样它们直接参与action expert在每个denoising step的velocity field prediction, 不经过VLM representation的压缩。

这是最直接的state→action路径。

#### (4) State Expert (se)

添加一个dedicated state-processing stream, 平行于VLM和action expert。State有自己的sequence-modeling path, 通过transformer stack处理, 与action module在generation过程中exchange信息。

**容量最大**: state既不被压缩进VLM, 也不被折叠进action suffix, 而是被自己的transformer stack处理。Table 1显示se增加199.30M trainable parameters, 远高于其他interface。

#### (5) Feature Modulation (fm)

State作为separate conditioning memory, 不作为ordinary sequence tokens。Action expert每一层通过cross-attention读state, 预测per-feature scale和shift:

$$\text{Mod}(z; S) = (1 + \gamma(z, S)) \odot z + \beta(z, S) \tag{4}$$

变量解释:
- $z$: action expert某一层的feature
- $\gamma(z, S)$: 学到的per-feature scale, 与z和S都有关
- $\beta(z, S)$: 学到的per-feature shift
- $\odot$: Hadamard product

这就是FiLM-style conditioning在flow matching action expert里的实现。增加123.84M parameters。

### 2.3 参数量和计算cost的trade-off

单帧条件下:
- sp: 0 trainable params, 但1114 training GFLOPs/sample, 282 inference GFLOPs/10-step call (最贵!)
- vp: 4.26M params, 16.9 / 4.3 GFLOPs
- ap: 1.08M params, 3.5 / 7.6 GFLOPs
- se: 199.30M params, 2.6 / 0.7 GFLOPs (最便宜!)
- fm: 123.84M params, 45.4 / 11.4 GFLOPs

这个对比很有意思: sp没有参数, 但因为66个text token要过18层VLM, 算力开销比所有continuous接口都贵2个数量级。se虽然参数多, 但因为是独立小stream, marginal compute极低。

Appendix C给了完整的derivation。关键公式:

$$F_{\text{tok}}(d, m) = 2L[d(H + 2 \times 256 + H) + 3dm] \tag{8}$$

变量解释:
- $L = 18$: Transformer层数
- $d$: host module hidden width
- $m$: FFN width (VLM: 16384, action: 4096)
- $H = 8 \times 256 = 2048$: grouped attention inner width
- 第一项: attention cost
- 第二项: FFN cost

VLM prefix的attention expansion (公式10):

$$\Delta F_{\text{VP, attn}} = 4HL(2PK + K^2 + AK) \tag{10}$$

Action prefix的attention expansion (公式11):

$$\Delta F_{\text{AP, attn}} = 4HL\left(PK + \frac{K(K+1)}{2} + AK\right) \tag{11}$$

变量解释:
- $P = 784$: 原始prefix token数 (3×256 image + 16 instruction)
- $A = 50$: action tokens
- $K$: state token数
- VLM prefix是bidirectional, 所以有$2PK$ (prefix看state双向)和$K^2$ (state内部双向)
- Action prefix是causal, 所以有$K(K+1)/2$ (state内部因果)

这就是为什么VLM prefix的training cost随K线性甚至超线性增长, 而action prefix的inference cost随K增长且需要per Euler step重复。

---

## 3. 三个Research Questions的实验设计

### 3.1 Benchmark设计 (Section 5.1)

用了RoboCasa365, 但做了一个非常重要的pre-partition: 按**control semantics**而非post hoc performance把45个atomic task分成三个family:

- **Family A** (rearrangement, pick-and-place): 探测large-range positioning of EEF and mobile base. 15 tasks.
- **Family B** (articulated-object interaction): 探测sustained contact和motion-phase modeling. 15 tasks.
- **Family C** (knob, switch, appliance control): small workspace, high local precision. 15 tasks.

**关键设计**: 每个family单独训练一个category expert, 这样cross-family结果反映的是interface在不同control demand下的行为, 而不是unified multi-task policy的泛化问题。这是controlled study的精髓。

另外有20个composite tasks (RoboCasa lifelong_learning_phase2 setting), 每个task chain 2-3个atomic subgoals在一个episode里。

每个atomic task用50个closed-loop rollout评估, 每个composite task用25个, 全程fixed seed schedule (控制scene init、object instance、placement)。

### 3.2 Slot-matched control (RQ2的关键设计)

这是paper最聪明的设计之一。当你sweep history length K, 自然也增加了conditioning tokens数量, 这本身可能就是性能提升的原因 (额外的capacity)。

为了separate temporal content vs conditioning capacity, 他们引入repeat-current control:

- **True history**: $S_t^{(K)} = (s_{t-K+1}, \ldots, s_t)$ - 真实有序序列
- **Repeat-current**: $S_t^{(K)} = (s_t, s_t, \ldots, s_t)$ - 把当前state复制K份, 保持slot count但消除temporal variation

如果repeat-current也能达到同样性能, 说明gain来自capacity而非temporal content。

---

## 4. 关键实验结果

### 4.1 RQ1: Current State是否有用

看Table 1的atomic macro:
- No state baseline: 54.6%
- sp: 57.7% (+3.1) † (paired task-bootstrap 95% CI [0.2, 6.1], 唯一exclude zero的)
- vp: 56.8% (+2.1)
- ap: 55.7% (+1.1)
- se: 57.6% (+3.0)
- fm: 57.6% (+2.9)

**关键观察**: 所有五个interface的point estimate都超过baseline, 但只有sp达到了统计显著的macro gain。

更重要的是**family-level ranking reverse**:
- Family A (rearrangement): sp最强 68.7% (+7.0), continuous interface基本没用甚至有害 (-0.1 到 +2.6)
- Family B (articulation): vp最强 68.8% (+6.1), se和fm接近 (+5.8, +5.5), sp掉到中游 (+1.6)
- Family C (knob/switch, 最难): se最强 42.8% (+3.3), vp竟然below baseline (-1.2)

**这个发现的intuition**: 不同task family对state representation有不同需求。
- Family A是大范围positioning, 离散化state与language alignment最自然, sp最好
- Family B是持续contact, 需要continuous state进行平滑motion-phase modeling, vp/se/fm都好
- Family C是高精度局部control, 需要state直接condition action expert而非经过VLM压缩, se最好, vp反而因为VLM的"smooth averaging"失去精度

这就是为什么paper说"there is no task-agnostic best interface"。一个benchmark-wide average (54.6 → 57.7)会完全hide掉这种reversal pattern。

### 4.2 RQ2: State History是否有用

Figure 3的sweep非常informative:
- **Short history helps**: K=8附近达到最consistent gains
- **Long raw history hurts**: K=96显著下降, 尤其Family C
- **非单调**: 不是more is better

Table 2是slot-matched control的关键证据:

EEF-pose state:
- AP: K=1 → K=8: 28.2 → 39.0 (+10.8) - 大幅提升
- VP: K=1 → K=8: 34.4 → 33.8 (-0.6) - 无变化
- SE: 25.8 → 28.0 (+2.2)
- FM: 27.8 → 32.2 (+4.4)

Slot-matched control (EEF-pose, AP):
- Current-only (重复当前state 8次): 30.8
- Genuine history (8个有序state): 39.0
- ΔSR: +8.2

**这个+8.2是论文最重要的发现之一**: 在固定slot count、固定image、固定instruction、固定flow noise的情况下, 把重复state替换成真有序history, 性能从30.8提升到39.0。这证明gain来自temporal variation而非capacity。

Joint-angle state (action space仍是EEF deltas):
- AP: 31.4 → 36.2 (+4.8)
- VP: 33.6 → 35.8 (+2.2)
- 仍然preserve qualitative advantage of short history

**结论**: useful range of history是有界的, 短history帮助、长raw history有害, 且这个结论across state representation robust。

为什么长history有害? 我个人的intuition (paper没有直接说): 
1. Raw state history有大量redundancy (相邻帧EEF pose几乎相同), VLM/attention需要消耗capacity去"理解"这种redundancy
2. 长history引入stale information, 当task需要precise state-to-action alignment时 (Family C), stale state会push model toward average action rather than precise current-state-conditioned action
3. 16维state在K=96就是1536维, 经过projector后是96×1024 tokens, 远超过原本的784 prefix tokens, 整个context被state dominate

Karpathy你在micrograd的视角下能想: 长history增加了loss landscape的非凸性, 浅层attention容易学到copy自己recent trajectory的shortcut (De Haan et al. 2019; Wen et al. 2020), 这种shortcut在short horizon下没那么容易学到。

### 4.3 RQ3: State应该从哪里注入

这是paper最striking的发现, 我称之为"routing crossover":

**K=1 (single frame)**:
- Composite tasks: vp1=34.4 vs ap1=28.2 (EEF), vp1=33.6 vs ap1=31.4 (joint)
- VLM-side wins明显
- Atomic tasks: 几个interface相差0.1 points, VLM-side微弱领先

**K=8 (short history)**:
- Composite tasks: ap8=39.0 vs vp8=33.8 (EEF), ap8=36.2 vs vp8=35.8 (joint)
- Action-side wins明显 (特别EEF case)
- Atomic: AP=59.6, 持平或超过其他

**设计规则** (paper Section 5.2末尾): 
> Inject single-frame state into the VLM, but route multi-frame state history to the action head.

这个crossover的intuition:

**K=1时为什么VLM side好**: 单帧state本身信息量少, 它的价值更多在于contextualize visual-language representation。比如state能告诉模型"我现在手在哪里", 这个信息与image里看到的cabinet position结合起来, 帮助VLM理解"我现在处于哪个阶段"。Appendix A.2的probe实验直接证实了这点: 在VLM最后6层, language-to-image attention redistribution在vp1下是17.3%, vp8下是22.0%, 都明显高于ap (因为ap的state不进入VLM prefix)。

**K=8时为什么action side好**: 多帧state提供的是temporal evolution信息, 这种信息本质上与velocity field prediction (flow matching) 对齐。Action expert在做的是预测从noisy action到clean action的velocity, 而state history的temporal derivative (隐式地) 提供了这种velocity的prior。把这个信号直接送给action head, 比先通过VLM compress成context再condition action要更直接。

Appendix A.3的flow probe给出了定量证据:

公式(5):
$$c_t = \hat{a}_t^{\text{true}} - \hat{a}_t^{\text{off}}, \quad r_t = a^* - \hat{a}_t^{\text{off}} \tag{5}$$

变量解释:
- $c_t$: state-conditioned correction at Euler step t (true state forward pass 与 state-off forward pass 的action差)
- $r_t$: residual from state-off to expert action (理想correction方向)
- $\cos(c_t, r_t)$: correction direction alignment
- $\|c_t\|_2 / \|r_t\|_2$: normalized correction magnitude

结果:
- vp1: final alignment 0.245, magnitude 0.297
- ap1: final alignment 0.079, magnitude 0.174
- ap8: final alignment 0.270, magnitude 0.382

注意ap1→ap8: alignment从0.079提升到0.270 (+0.191, 95% CI [+0.143, +0.239]), magnitude从0.174提升到0.382 (+0.208)。这定量证明short history让action-side state的correction既更aligned又更强, 而这种提升在VLM-side没出现 (vp1→vp8基本不变)。

### 4.4 PrepareToast case study (Appendix B)

这个case study很巧妙, 把aggregate result localize到具体subtask transition。

PrepareToast: 放两个task-relevant items到cabinet, 然后返回cabinet关闭。四个monotonic predicates:
- S1: 第一个item放好
- S2: 两个items都放好
- S3: cabinet重新关闭
- S4: 完成后gripper released

Table 8显示AP1 vs AP8:
- S1: 90% vs 96% (差不多)
- S2: 64% vs 68% (差不多)
- S3: 30% vs 56% (差26 points!)
- S4: 同样26 points差距

**Key insight**: 两个policy在前期placement stage几乎一样, 但在"必须从placement transition回cabinet closing"的late transition, AP8显著胜出。

更精细的probe (公式6):
$$D_q = \|\hat{\mathbf{a}}_q^{\text{true}} - \hat{\mathbf{a}}_q^{\text{repeat}}\|_2 \tag{6}$$

在固定AP8 checkpoint下, 把有序history换成8个current state copies, 测action变化:
- Within a stage: 0.198
- Before boundary: 0.361
- At boundary: 1.033
- After boundary: 0.748

**Boundary sensitivity是within-stage的5.2倍**。这定量证明policy在progress boundary处最rely temporal variation, 而不是单纯靠更多tokens。

---

## 5. Paper的Limitations

paper自己列了两个:
1. **No real-robot validation**: 全在simulation (RoboCasa)。Real robot的proprioception noise、calibration drift可能改变结论。
2. **Purely kinematic state**: 没考虑force、tactile等其他sensing modality。在contact-rich task里, force-torque history可能比joint angle history更重要。

我额外想到几个:
3. **Single backbone**: 只在π0.5 (flow-matching + VLM)上测试。如果换成tokenized action (像OpenVLA原版)或autoregressive action, 结论可能不同。
4. **State dimension固定**: $d_s = 16$。对humanoid (高DoF)是否成立未知。
5. **Single seed for most comparisons**: 只有sp的atomic macro gain达到了paired bootstrap 95% CI exclude zero。其他都是single-seed point estimates, 严格说只是directional pattern而非statistically established。
6. **History长度选择K=8**: 是empirical sweet spot for this benchmark, 不一定universal。Task timescale不同可能不同。

---

## 6. 设计规则总结 (build my intuition)

如果让我从paper提炼一个mental model, 大致是这样:

**Mental model 1: State在VLA里有两种fundamentally different roles**
- **Contextual role** (K=1, VLM-side): state作为"我在哪"的global context, 与image+language jointly形成situation understanding。这种role下, state通过VLM prefix参与multimodal context modeling最自然。
- **Dynamic role** (K>1, action-side): state作为"我在动"的local dynamic, 直接inform action expert的velocity field prediction。这种role下, 走action prefix最直接。

**Mental model 2: 不要把state当text**
sp虽然在Family A和single-frame macro最好, 但代价是66个text tokens过18层VLM, training cost 1114 GFLOPs/sample (比se贵400x)。如果compute-constrained, continuous interface (se/fm)以1-2个数量级更低cost达到相近性能。

**Mental model 3: 历史不等于更多context**
长history引入stale information和shortcut risk (BC copycat)。Short history (K=8)是当前benchmark的sweet spot, 但更长的history需要compressed latent (像memory tokens或recurrent state), 而非raw stack。这呼应了Diffusion Policy里用history的obs horizon vs action horizon分离的设计。

**Mental model 4: Controlled study的discipline**
每个claim都要在: 固定backbone + 固定data + 固定action representation + 固定eval protocol + slot-matched control下做。任何"我的新设计更好"的claim如果没做这种controlled comparison, 都要打折扣。

---

## 7. 对未来VLA design的implications

如果我在design一个新VLA, 基于这篇paper会怎么做:

1. **Default design**: sp-style state prompt for current frame (lightweight alternative: vp) + ap-style short history (K=8) for temporal context。Hybrid interface。
2. **Cost-conscious design**: se (State Expert)在所有continuous接口里marginal compute最低 (2.6/0.7 GFLOPs), 参数虽多但inference友好, 适合长history scaling。
3. **Precision tasks**: 走action-side (ap或se), 避开VLM-side的smoothing averaging。
4. **Long-horizon tasks**: 不要raw stack long history, 要compressed memory (这就回到memory token / recurrent的设计, 见Bulatov et al. 2022, Shi et al. 2025 MemoryVLA)。
5. **Benchmark reporting**: 不要只报macro average, 必须按control semantics分family report, 否则会hide reversal pattern。

Karpathy你在Eureka Labs里讲的"做science"的精神, 这篇paper是VLA sub-field里少见的真正符合这种精神的work。它没有propose新SOTA, 但它给community一个reusable evaluation protocol和一组testable design principles, 让后续工作可以站在一个更solid的empirical basis上。

**进一步阅读建议**:
- RoboCasa365 paper: https://arxiv.org/abs/2603.04356
- Latent action supervision for VLA (Lin et al. 2026): https://arxiv.org/abs/2605.04678
- MemoryVLA (Shi et al. 2025): https://arxiv.org/abs/2508.19236
- Causal confusion in imitation learning (De Haan et al. 2019): https://papers.nips.cc/paper/2019/hash/hash
- Fighting copycat agents (Wen et al. 2020): https://papers.nips.cc/paper/2020/hash/hash

Paper link (推测): https://arxiv.org/abs/2603.04356 (RoboCasa) 同期work, 实际paper link需要作者发布, 但作者来自HKUST(GZ), MMLab CUHK, AI2 Robotics X-Lab, 应该是2026年arXiv release。
