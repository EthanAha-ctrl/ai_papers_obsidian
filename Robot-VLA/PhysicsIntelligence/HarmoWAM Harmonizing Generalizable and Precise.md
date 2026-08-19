---
source_pdf: HarmoWAM Harmonizing Generalizable and Precise.pdf
paper_sha256: b2a4b26d4cf8f6b7583157356f7ca0c1b5341a9c2dbd2e4023eb383529080974
processed_at: '2026-08-19T10:24:15-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HarmoWAM 人话版

## 1. 这篇 paper 在讲啥故事

假设你教一个机器人叠易拉罐。它得先伸手过去（transit），再精确地放上去（interaction）。看起来一个动作流，但这两个阶段对机器人能力的要求**完全不一样**：

- **transit 阶段**：要求"我见过类似场景，知道大概往哪走"。这是 generalization 问题。
- **interaction 阶段**：要求"我对准这个易拉罐的边缘，差 2 毫米就倒"。这是 precision 问题。

**核心发现**：现有的 World Action Model 恰好分两类，一类擅长 transit，一类擅长 interaction，**没有任何一类同时擅长两者**。

这就尴尬了 —— 你用 A 方法，机器人能走到 OOD 位置但抓不准；用 B 方法，抓得准但在 OOD 根本走不过去。

**HarmoWAM 的解法**：那就两个都装上，加一个"调度员"决定什么时候用哪个。

---

## 2. 两类 WAM 的本质差别（用大白话）

### Imagine-then-Execute（"先想后做"）

代表：WoW、DreamGen、Wan+AnyPos

工作流程：
1. World model 生成一段未来视频（"我想象手伸过去抓杯子"）
2. IDM（inverse dynamics model）从相邻 frame 推出 action（"从 frame A 到 frame B，手移动了 dx, dy, dz"）

**优点**：video model 见过上百万段 robot video，对"伸手去某个地方"这种粗动作泛化得特别好。OOD 场景下它依然能"想象"出合理的 trajectory。

**缺点**：IDM 只看相邻 frame 的 pixel 差异来推 action。这是**局部回归**，对"杯子要旋转 30°、夹爪力度刚好"这种精细活儿完全没有 structural prior。所以一到 interaction 阶段就开始抖。

类比：就像你问 GPT "给我写个抓取程序"，它能给你一个看起来对的，但细节经不起推敲。

### Joint Modeling（"一起学"）

代表：VPP、Cosmos-Policy、UWM

工作流程：action 和 video 在同一个 diffusion 流形里联合训练，action 直接受 video latent feature conditioning。

**优点**：action 是 video latent "喂"出来的，自带 temporal coherence。一旦手在杯子旁边，它能精确地完成抓取、旋转、放下。

**缺点**：action head 是用 SFT 数据（你那 100 个 demo）训练的。**SFT 数据覆盖的空间 = action head 能探索的空间**。你 demo 全在桌面左半边，测试时杯子在右半边，它直接懵了 —— transit 成功率掉到 0/10。

类比：就像一个学生，老师讲过的题型做得又快又对，但没见过的题型完全不会举一反三。

### Table 1 这个对照（再强调一遍）

这个表是整篇 paper 的"判决书"。带星号的 interaction 测试是把机器人手**直接放在目标物附近**，隔离掉 transit 失败的影响。

然后你看到：
- Imagine-then-Execute：OOD 时 transit 还是 10/10，interaction 掉到 5/10、2/10
- Joint Modeling：OOD 时 transit 崩到 0/10、3/10，但**只要把它放到目标旁边**，interaction 依然 10/10

这说明啥？两类方法的"能力"都在，只是用错了地方。

---

## 3. HarmoWAM 怎么把它们合起来

整体架构是一个 **shared world model + 两个互补的 action expert + 一个 gating network**。

### World Model（共享大脑）

用 Wan2.2-TI2V-5B（一个 5B 参数的视频生成模型），先在 ~1.9M robot trajectories 上 pretrain，再在 task data 上 finetune。

它每次吐出两样东西：
- **显式**：未来 13 帧 video（256×320）
- **隐式**：latent feature（80 个 temporal token，每个 3072 维）

这两个 output 分别喂给两个 expert。

### Predictive Expert（精细活专家）

一个 1B 参数的 Action DiT（diffusion transformer）。输入：当前 obs + instruction + **world model 的当前时刻 latent**。输出：12 步 action chunk。

它学的是 conditional action distribution $p(\mathbf{a} | \text{obs}, \text{instruction}, \text{world latent})$。因为有 world latent 这个"未来感知"信号，action 序列天然 temporally coherent，特别适合需要连续精确控制的动作。

类比：这就是一个加强版的 Diffusion Policy，多了个"我大概知道未来会发生什么"的先验。

### Reactive Expert（粗活专家）

DINOv2 提取 predicted frame 的 patch feature + world model 的 latent feature，过一个 conv decoder 直接吐 action。

本质是一个 IDM（inverse dynamics），但比经典 IDM 多了 world latent 作为高层 semantic cue。它快、泛化好、interpretable，但精度有限。

### Process-Adaptive Gating（调度员）

一个轻量 MLP，输入是 SigLIP visual tokens，输出一个 0-1 的 score $s_t$。>0.5 用 predictive，≤0.5 用 reactive。

**关键问题是：训练 label 怎么来？**

不能手标，太贵。作者用一个 heuristic 自动标：
- gripper 状态变化（open→close 抓取，close→open 释放）→ 这个时刻附近 20 帧 标为 interaction（y=1，给 predictive）
- end-effector 高度变化（for 插入、倾倒任务）→ 辅助信号
- 其余 frame 标为 transit（y=0，给 reactive）

frame-level 准确率 96.95%。这个数字很关键 —— gating 本身几乎不引入额外 error。

---

## 4. 为什么这个设计 work（intuition）

### 4.1 两个 expert 看的世界不一样

Figure 3 的 attention map 很有说服力：
- Predictive expert attention 集中在**被操作物体**（精细控制关注目标）
- Reactive expert attention 集中在**夹爪和周围环境**（粗运动关注自身定位和避障）

这就像人脑的 ventral stream（识别"是什么"）和 dorsal stream（判断"在哪里、怎么动"）的分工。Nature 早就这么干了，HarmoWAM 是把这个 principle 复制到 robot policy 里。

### 4.2 为什么不能简单 average 两个 expert

Ablation 试过：
- 直接平均两个 expert 输出 → OOD 掉 46%
- 仅在 interaction phase 平均 → 掉 31%
- Process-Adaptive Gating → 掉最少

直觉：action space 里的"平均"不是好做法。两个 expert 学的是不同的 conditional distribution，硬 average 会生成 neither here nor there 的 invalid motion。就像你不能把"伸手过去"和"精细对准"两个动作平均一下，那是一团浆糊。

Hard switch 在 boundary 帧会有 jerk，但 action chunk（12 步）+ chunk overlap execution 把这个 jerk 平滑掉了。

### 4.3 为什么 world model latent 这么重要

去掉 reactive expert 的 latent → OOD 从 80% 掉到 54%
去掉 predictive expert 的 latent → ID 从 95% 掉到 62%

predictive expert 对 latent 更敏感（-33 个点），因为它**强依赖 world latent 的 temporal dynamics** 来生成 coherent action chunk。这就像一个钢琴家，没了乐谱就只能瞎弹。

reactive expert 受 latent 影响小一些，但仍然显著。经典 IDM 只看 pixel，加上 latent 后它知道"这个 frame 在未来 trajectory 里的 semantic role"，能做更 robust 的 action inference。

---

## 5. 实验数据讲了什么故事

### 5.1 In-Domain（Table 2）

HarmoWAM 平均 0.89，次优 Cosmos-Policy 0.78，π0.5 是 0.74。

ID 上能拉开 11 个点很说明问题 —— 通常这意味着架构层面有真东西，不只是 hyperparameter tuning。

特别看精细任务：
- Stack Coke Cans：0.90 vs 次优 0.65 → 精度优势明显
- Write "Yes"：0.92 vs 次优 0.83 → 连续控制好
- Put Items to Bag and Zip（400 步长程任务）：0.85 vs 次优 0.72 → error propagation 控制好

### 5.2 OOD（Table 3，这才是重头戏）

| Method | Global Avg | Drop from ID |
|---|---|---|
| π0.5 | 0.49 | -33.8% |
| VPP | 0.41 | -43.8% |
| Wan+AnyPos | 0.53 | -20.9% |
| QwenVLA-OFT | 0.41 | -34.9% |
| Cosmos-Policy | 0.44 | -43.6% |
| **HarmoWAM** | **0.82** | **-7.9%** |

7.9% 的 OOD drop 在 manipulation 领域是个惊人的数字。通常 OOD 都掉 30-50%。

**最 dramatic 的是 Position OOD**：
- π0.5: 0.32
- Cosmos-Policy: 0.26
- HarmoWAM: **0.80**

这正是 paper 的核心 thesis：position OOD 是 Joint Modeling paradigm 的死穴，而 HarmoWAM 用 reactive expert 借 world model 的 generalization 突破了。

但要注意：HarmoWAM 在 position OOD 上能到 0.80，**主要功劳是 world model 见过 1.9M trajectories 后的 spatial generalization**。这意味着 HarmoWAM 的 OOD 能力是 world model 转移过来的 —— 如果 world model pretraining data 不够大不够 diverse，这个优势会消失。**这是一个 scaling-dependent 的 method**。

### 5.3 一个有意思的细节：denoising steps 的 sweet spot

Table 9 显示 world model 用 5 步 denoising 是最佳：
- 3 步：80% success，4 Hz
- **5 步：85% success，4 Hz** ← 选这个
- 10 步：85% success，3.6 Hz
- 50 步：87% success，3 Hz

50 步只比 5 步多 2 个点但慢一倍。这说明 robot video prediction 不需要影视级渲染质量，**够用就行**。这个 trade-off 的 finding 对后续 WAM 工作很有参考价值。

---

## 6. 一些更深层的联想

### 6.1 这就是 robot control 的 Mixture-of-Experts

类比 MoE LLM：
- World model = shared attention（共享表征）
- 两个 action expert = expert FFN（各司其职）
- Gating network = router（决定谁说话）

按这个 analogy 延伸，未来可能有多于两个 expert：
- 加一个 "safety expert"（专门处理边界情况）
- 加一个 "compliance expert"（专门处理 force-controlled 接触）
- 加一个 "exploration expert"（专门处理未知物体探测）

这就成了一个 multi-expert WAM。MoE 在 LLM 里 proved to scale，这个 pattern 在 robot control 里说不定也能 scale。

### 6.2 和 System 1 / System 2 的对应

 Reactive expert ≈ System 1：快、interpretable、基于视觉 cue、generalization 来自 pretraining
 Predictive expert ≈ System 2：慢、deliberative、基于 latent dynamics、precision 来自 SFT

但和 Kahneman 原版不同的是，HarmoWAM 的两个 system 共享同一个 world model 作为 perception grounding。**它不是两个独立的 system，而是 shared perception + dual action pathways**。

这个 design 比完全分离的 dual system 更经济，也更符合实际 —— 人脑的视觉皮层也是 System 1 和 System 2 共享的。

### 6.3 一个潜在的隐患：gating label 的 bootstrapping

Gating 的 label 假设 "interaction phase = gripper state change 时刻"。

但有些任务 gripper 状态不变也是 interaction：
- 推盒子
- 滑动门
- in-hand manipulation（手指一直在动但 gripper 不闭合）
- 抛接物体

Paper 里测的 6 个 task 都涉及 grasp/release，所以这个 heuristic 工作。推广到 contact-rich without gripper change 的 task 需要重新设计 label heuristic。

更通用的做法可能是：用 force-torque sensor 信号、用 end-effector velocity magnitude、用 contact event detection 来标 interaction phase。或者用一个 self-supervised method 让 gating 自己从 action distribution 的 entropy 学。

### 6.4 一个 design question：hard switch vs soft routing

Paper 用的是 hard switch（threshold 0.5）。在 boundary 帧上 $s_t$ 在 0.5 附近时会有 discontinuity。

为什么不用 soft routing $\mathbf{a} = s_t \cdot \mathbf{a}^{pred} + (1-s_t) \cdot \mathbf{a}^{react}$？

我猜是因为 action distribution 的 multimodality —— 两个 expert 学的是不同的 conditional distribution，linear interpolation 在 action space 里不会得到合理的"中间动作"。就像你不能把"挥手"和"握手"两个动作平均一下得到一个有意义的新动作。

但 hard switch 也有问题：boundary 帧上的 jerk。Paper 用 action chunk（12 步）+ chunk overlap execution 平滑掉了。这是一个 pragmatic 的工程解法。

更 principled 的做法可能是：让 gating 输出不仅决定走哪个 expert，还决定 expert 之间的 blend weight，但 blend 在 latent space 而不是 action space。这个方向值得探索。

### 6.5 关于 inference speed 的 multi-rate control

World model 跑 5 步 denoising 大约 4 Hz（Table 9）。
Action chunk 12 步 × 48 Hz（paper 报的）= 250ms 一个 chunk。
Reactive expert 是 DINOv2-base + conv decoder，应该 100+ Hz。

这就形成 **慢 world model（4Hz）+ 快 reactive expert（100+Hz）+ 中速 predictive expert（48Hz）** 的 multi-rate control。

非常类似 ACT 的 chunk execution + reactive correction 模式。但 HarmoWAM 多了一层：world model 作为最慢的"远景规划"，reactive expert 作为最快的"反射弧"，predictive expert 作为中速的" deliberative control"。

这和人脑的 multi-rate decision-making 也很像：远景规划（前额叶，秒级）+ deliberative 动作规划（运动皮层，100ms 级）+ 反射弧（脊髓，10ms 级）。

---

## 7. 局限性总结（人话版）

1. **World model 固定输出 13 帧**。下游任务必须匹配这个 horizon。如果想做更长程任务（比如 50 步以上的），现在的架构不支持。未来需要 adaptive horizon。

2. **Gating label 依赖 gripper state change heuristic**。对 push、slide、in-hand manipulation 这种没有 gripper 闭合的任务不通用。

3. **强依赖 world model pretraining scale**。1.9M trajectories 是个不小的数字。如果换成 100K 数据，OOD 优势可能消失。

4. **Action space 是 7-DoF（单臂）/ 14-DoF（双臂）的 end-effector pose**。对 dexterous hand（30+ DoF）的扩展性未知。Predictive expert 的 diffusion head 可能需要 redesign。

5. **Hard switch gating 的 boundary jerk**。靠 action chunk 缓解，但 principled 的做法应该是在 latent space 做 soft routing。

6. **World model 和 action expert 是 sequential 训练**（先 stage 1 再 stage 2），不是 joint。这限制了 action expert 反过来影响 world model 的可能性。一个有趣的延伸是让 action expert 的 gradient 回流到 world model，做真正的 end-to-end。

---

## 8. 我对这篇 paper 的整体判断

**这是一篇 architectural combination paper**。它没发明新的 world model，没发明新的 action policy，没发明新的 gating 机制 —— 这些组件都是现成的。

它的贡献是：
1. 用一个干净的 empirical study（Table 1）**系统性地 expose 了一个 trade-off**
2. 用 **shared world model + dual expert + gating** 的 pattern 把这个 trade-off resolve 了
3. 在 real-world 评测上 **真正 work**（7.9% OOD drop 是非常 strong 的 number）

在 WAM 这个方向里属于"first to systematically identify and address the trade-off"的工作。它的核心 insight —— "transit 和 interaction 需要不同能力，应该用不同 expert" —— 听起来简单，但**简单且 work 的 idea 通常是好 idea**。

特别有启发的是 **"shared perception + complementary action experts"** 这个 pattern。它给我一种感觉：robot policy 的未来可能不是一个 monolithic policy，而是一个 **"world model as shared cortex + specialized action experts as motor cortexes + gating as basal ganglia"** 的 modular 架构。这和 LLM 里 MoE 的进化路径惊人地相似。

如果这篇 paper 的 follow-up 能做到：
- 3+ 个 action expert
- self-supervised gating label
- adaptive world model horizon
- joint training（end-to-end, not sequential）

那就真的是 robot policy 的 "MoE 时刻"了。

---

## References

- Project page: https://elbb-yu.github.io/HarmoWAM/
- Wan2.2 (world model): https://arxiv.org/abs/2503.20314
- AnyPos (reactive expert 基础): https://arxiv.org/abs/2507.12768
- VPP (joint modeling baseline): https://arxiv.org/abs/2501.02532 (Hu et al.)
- Cosmos-Policy: https://arxiv.org/abs/2601.16163
- WoW (Imagine-then-Execute 代表): https://arxiv.org/abs/2509.22642
- DreamGen: https://arxiv.org/abs/2505.01597 (Jang et al.)
- π0.5 (VLA baseline): https://arxiv.org/abs/2504.16054
- DROID (pretraining data): https://arxiv.org/abs/2403.12945
- AgiBot World: https://arxiv.org/abs/2503.06669
- RoboMIND: https://arxiv.org/abs/2412.13877
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP: https://arxiv.org/abs/2303.15343
- Flow Matching: https://arxiv.org/abs/2210.02747
- Perceiver-Actor (keyframe extraction): https://proceedings.mlr.press/v205/shridhar23a.html
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ACT (chunk execution): https://tonyzhaozh.github.io/aloha/
- Fast-in-Slow (dual-system VLA): https://arxiv.org/abs/2506.01953
- GR-2 (video-language-action): https://arxiv.org/abs/2410.06158
- Two-streams hypothesis (ventral/dorsal): https://en.wikipedia.org/wiki/Two-streams_hypothesis
- MoE in LLMs (Mixtral): https://arxiv.org/abs/2401.04088

---

# HarmoWAM: Harmonizing Generalizable and Precise Manipulation via Adaptive World Action Models

## 1. 一句话直觉

这篇 paper 的核心 insight 可以浓缩成一句话：**"transit 阶段需要 generalization，interaction 阶段需要 precision，而这两个能力天然属于两种不同的 WAM paradigm，所以用一个 gating network 在它们之间动态切换"**。

这种 trade-off 在 robot learning 里其实是个老问题 —— 经典的 "exploration vs. exploitation" 在 manipulation 领域重新表现为 "approach vs. contact"。HarmoWAM 的贡献是发现 **World Action Model 的两个 paradigm 恰好分别擅长这两端**，于是用 world model 提供共享的 spatio-temporal prior，让两个 expert 各司其职。

Project page: https://elbb-yu.github.io/HarmoWAM/

---

## 2. 核心 Motivation: Table 1 详解 (这是整篇 paper 的灵魂)

先看 Table 1，这是支撑整个工作的 empirical claim：

| Method | Domain | Stack Coke Cans Transit / Interaction | Put Flowers in Vase Transit / Interaction |
|---|---|---|---|
| Imagine-then-Execute | ID | 10/10 / 7/10 | 10/10 / 8/10 |
| Imagine-then-Execute | OOD-Background | 10/10 / 6/10 | 10/10 / 6/10 |
| Imagine-then-Execute | OOD-Position | 10/10 / 5/10 | 10/10 / 2/10 |
| Imagine-then-Execute | OOD-Objects | 10/10 / 7/10 | 10/10 / 7/10 |
| Joint Modeling | ID | 9/10 / 9/10 | 9/10 / 10/10 |
| Joint Modeling | OOD-Background | 5/10 / 8/10* | 5/10 / 9/10* |
| Joint Modeling | OOD-Position | 3/10 / 10/10* | 0/10 / 10/10* |
| Joint Modeling | OOD-Objects | 0/10 / 10/10* | 6/10 / 10/10* |

(带 `*` 的 interaction 是 init 在 target 物体附近的，isolating interaction precision from transit failure)

### 这张表告诉我们的 intuition

**Imagine-then-Execute** (Wan2.2-TI2V-5B + AnyPos): video prediction 模型在大规模数据上训练得到的"world knowledge"非常 robust —— 即使 OOD，它生成的 future trajectory 依然能把 gripper "imagined"到 target 附近。但 IDM (Inverse Dynamics Model) 从 frame pair 推 action 的过程是 **local、pixel-level 的回归问题**，对 contact-rich 的 fine-grained motion 没有强 inductive bias，所以 interaction 阶段会掉到 60% 左右。

**Joint Modeling** (Wan2.2-TI2V-5B + Action DiT): action 和 video latent 在同一个 denoising 流形里联合建模，action 自带 temporally-coherent 的结构 prior，所以一旦 init 在 target 附近，interaction 高达 95-100%。但它的 action head 是用 SFT 数据训练的，**SFT 数据覆盖的空间 = action head 的 exploration space** —— OOD 时它根本不知道往哪走，transit 直接崩溃到 0/10。

这里有一个非常 deep 的 observation：**video diffusion 的 generalization 能力 来自 pretraining distribution 的覆盖广度**，而 **action head 的 precision 来自 SFT 数据的局部密度**。这两者本质上来自不同的数据池 —— 前者从 web-scale video + multi-robot dataset 来，后者只能从 narrow task demos 来。HarmoWAM 的设计哲学就是：**用 video prediction 这条"高速公路"做 transit，用 SFT-trained action head 做 interaction 精修**。

---

## 3. 架构深度解析

### 3.1 World Model: Wan2.2-TI2V-5B

- Backbone: Wan2.2 Text-Image-to-Video 5B (https://arxiv.org/abs/2503.20314)
- Pretraining data: ~1.9M trajectories，包括 DROID (201K, Franka Panda) + AgiBot (3K, AgiBot G1) + RoboMIND (1.72M, Franka/UR/Ark/Agilex/TienKung) + closed-source data
- Output: 256×320 resolution × 13 frames
- Inference: 5 denoising steps (Table 9 ablation 显示 5 steps 是 sweet spot，50 steps 只多 2% 但慢一倍)

World model 提供两个 output:
1. **Explicit**: predicted video frames $\mathbf{V}_{t:t+H}$
2. **Implicit**: latent representations $\mathcal{F}_{t:t+H}^{\mathbf{V}} \in \mathbb{R}^{B \times 80 \times 3072}$

这里的 `80` 是 temporal token 数量，`3072` 是 latent channel dim。注意 Wan2.2 是一个 DiT-based video diffusion，它的 latent 是经过 VAE 编码后的 spatio-temporal tokens。

### 3.2 Predictive Expert (1B-param Action DiT)

结构：
- 28 个 Transformer blocks
- SigLIP image encoder 提供 $\mathcal{F}_t^{img}$（current observation 的 visual tokens）
- T5 text encoder 提供 $\mathcal{F}^{text}$（instruction embedding）
- 用 cross-attention 注入 $\mathcal{F}_t^{\mathbf{V}}$ (current-step world model latent)

Denoising 过程：
$$\epsilon_\theta = \mathcal{D}_{\theta_{\text{pred}}}(\mathbf{a}_{t+1:t+H}, \tau_k \mid \mathcal{F}_t^{img}, \mathcal{F}^{text}, \mathcal{F}_t^{\mathbf{V}})$$

变量含义：
- $\mathbf{a}_{t+1:t+H}$: 未来 H 步的 action chunk，H=12，每个 action 是 7-DoF (single-arm) 或 14-DoF (dual-arm)
- $\tau_k$: diffusion timestep embedding（k 是去噪步 index）
- $\mathcal{F}_t^{img}$: 当前帧的 SigLIP visual tokens
- $\mathcal{F}^{text}$: instruction 的 T5 text tokens
- $\mathcal{F}_t^{\mathbf{V}}$: world model 在当前时刻的 latent（来自 diffusion U-Net/DiT 的中间层 feature）

**Intuition**: 这个 expert 本质上是一个标准的 Action Diffusion (类似 π0、RDT-1B、HybridVLA 的做法)，但额外用 world model 的 latent 做 "future-aware conditioning"。它学的是 action 的 conditional distribution $p(\mathbf{a} | \text{obs}, \text{instruction}, \text{predicted future latent})$，所以 action 序列天然 temporally coherent，适合 fine-grained manipulation。

### 3.3 Reactive Expert (DINOv2 + Orientation Decoder)

结构：
- DINOv2-base 提取 patch-level geometric features
- 输入：predicted future frame $\mathbf{V}_s$ + 它的 latent $\mathcal{F}_s^{\mathbf{V}}$
- DINOv2 输出: $\mathcal{F}_s^{\text{patch}} \in \mathbb{R}^{B \times 1369 \times 768}$ (1369 = 37×37 patch tokens at 256×320 / 14)
- $\mathcal{F}_s^{\mathbf{V}}$ 通过 average pooling 把 3072 维降到 768
- Concatenate: $\mathcal{F}_s^{\text{fuse}} = [\mathcal{F}_s^{\text{patch}}; \mathcal{F}_s^{\mathbf{V}}]$ (沿 token dim 拼接)
- Orientation Decoder $\mathcal{D}_{\text{ori}}$（multi-scale conv，following AnyPos）输出 action
- $\hat{\mathbf{a}}_s = \mathcal{D}_{\text{ori}}(\mathcal{F}_s^{\text{fuse}})$

**Intuition**: 这是一个 IDM (Inverse Dynamics Model)，但比经典 IDM 多了 world model latent 作为高层 semantic cue。经典 IDM 只看相邻 frame 的 pixel-level 变化推 action，对快速运动、遮挡、视角变化很敏感。加上 $\mathcal{F}_s^{\mathbf{V}}$ 之后，它有了 "这个 frame 在未来 trajectory 中的 semantic role" 信息，能做更 robust 的 action inference。这个设计借鉴了 AnyPos (https://arxiv.org/abs/2507.12768) 的 Orientation Decoder 思路。

### 3.4 Process-Adaptive Gating Mechanism

这是 paper 的核心创新点 —— 让两个 expert 在 inference time 动态切换。

**Architecture**: 一个轻量 MLP，输入是 SigLIP visual tokens $\mathcal{F}_t^{img}$（reuse predictive expert 的 image encoder 输出），输出 scalar $s_t \in [0, 1]$。

**Training label construction**: 用 keyframe extraction pipeline (following Perceiver-Actor, https://proceedings.mlr.press/v205/shridhar23a.html) 从 robot proprioceptive signal 自动标注：
- gripper state change (open→close 表示 grasp，close→open 表示 release) → key event
- end-effector height threshold (for insertion/pouring/placing tasks) → auxiliary cue
- key event 前后各 20 帧标为 interaction segment (label y=1，给 predictive expert)
- 其他标为 transit segment (label y=0，给 reactive expert)
- 双臂任务：任一臂满足即标 1

**Loss**:
$$\mathcal{L}_{gate} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(s_i) + (1-y_i) \log(1-s_i)]$$

变量：
- $N$: batch size
- $y_i \in \{0, 1\}$: 第 i 个 frame 的 label（1 = interaction，0 = transit）
- $s_i \in [0, 1]$: gating network 预测的 interaction 概率

**Inference**: threshold 0.5。$s_t > 0.5$ → 走 predictive expert；否则走 reactive expert。

**Frame-level accuracy on held-out test: 96.95%** —— 这个数字很重要，说明 gating 本身几乎不引入 error。

### 3.5 总训练目标

$$\mathcal{L}_{stage2} = \mathcal{L}_{pred} + \lambda_{react} \mathcal{L}_{react} + \lambda_{gate} \mathcal{L}_{gate}$$

其中 $\lambda_{react} = 0.1$，$\lambda_{gate} = 0.05$。这个 weighting 说明 gating loss 的梯度信号较弱，需要小心控制避免 dominate 主 action loss。

---

## 4. 训练 Recipe 细节

### Stage 1: World Model Finetuning (Conditional Flow Matching)

Flow Matching 的目标：
$$\mathcal{L}_{stage1} = \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, \xi, \mathbf{c}} \left[ w(\xi) \| f_\theta(\mathbf{x}_\xi, \xi, \mathbf{c}) - \mathbf{v}_\xi \|_2^2 \right]$$

变量：
- $\mathbf{x}_1$: clean video latent（来自 demonstration video 经过 VAE 编码）
- $\mathbf{x}_0 \sim \mathcal{N}(0, I)$: 纯 Gaussian noise latent
- $\xi \in [0, 1]$: flow interpolation time（uniformly sampled）
- $\mathbf{x}_\xi = (1-\xi)\mathbf{x}_0 + \xi \mathbf{x}_1$: linear interpolation between noise and clean
- $\mathbf{v}_\xi = \frac{d\mathbf{x}_\xi}{d\xi} = \mathbf{x}_1 - \mathbf{x}_0$: target velocity field
- $\mathbf{c}$: condition（current observation + task instruction）
- $w(\xi)$: flow-step-dependent weighting function
- $f_\theta$: 神经网络预测的 velocity

**Intuition**: Flow Matching (Lipman et al. 2023, https://arxiv.org/abs/2210.02747) 是 diffusion 的近亲，用 ODE flow 替代 SDE，训练更稳定，inference 时可以用更少 steps（5 步够用）。Wan2.2 本身就是用 Flow Matching 训练的，所以这里 finetune 是 compatible 的。

### Stage 2: Action Experts Finetuning (world model frozen)

两个 expert 的 loss：
$$\mathcal{L}_{pred} = \mathbb{E}_{\mathbf{a}_{t+1:t+H}, \epsilon \sim \mathcal{N}(0,1)} \left[ \|\epsilon_\theta - \epsilon\|_2^2 \right]$$
$$\mathcal{L}_{react} = \mathbb{E} \left[ d(\hat{\mathbf{a}}_{t+1:t+H}, \mathbf{a}_{t+1:t+H}) \right]$$

其中 $d(\cdot, \cdot)$ 是 Smooth L1：
$$d(x, \hat{x}) = \begin{cases} 0.5 \cdot \frac{(x - \hat{x})^2}{\beta}, & \text{if } |x - \hat{x}| < \beta \\ |x - \hat{x}| - 0.5\beta, & \text{otherwise} \end{cases}$$

$\beta = 0.1$ 是 Smooth L1 的 transition threshold。Smooth L1 相比 L2 对 outlier 更鲁棒，相比 L1 在 small error region gradient 更平滑，是 detection 任务 (Fast R-CNN) 经典的 loss。

**为什么 predictive 用 diffusion loss，reactive 用 regression loss？**

直觉：
- Predictive expert 是 generative model，学的是 $p(\mathbf{a} | \text{cond})$，需要 multi-modality（同一个 obs 可能对应多个合理 action），diffusion 自然支持
- Reactive expert 是 deterministic mapping $\mathbf{V}_s \to \mathbf{a}_s$，本质是 IDM regression，single-modal 就够，regression loss 更直接

---

## 5. 实验结果深度解读

### 5.1 In-Domain (Table 2)

| Method | Avg | 
|---|---|
| π0.5 | 0.74 |
| VPP | 0.73 |
| Wan+AnyPos | 0.67 |
| QwenVLA-OFT | 0.63 |
| Cosmos-Policy | 0.78 |
| **HarmoWAM** | **0.89** |

ID 上 HarmoWAM 比 SOTA (Cosmos-Policy) 高 11 个点，比 π0.5 高 15 个点。这个 gap 不小 —— 在 ID 评测上能拉开 10+ 个点，通常意味着架构层面有真东西，不只是 hyperparameter tuning。

**值得注意的 task-level patterns**：
- Stack Coke Cans (精细 stacking): 0.90 vs 次优 0.65 → 优势在 precision-required task 上最明显
- Put Items to Bag and Zip (long-horizon, 400 steps): 0.85 vs 次优 0.72 → 长程任务 error propagation 控制
- Write "Yes" (continuous control): 0.92 vs 次优 0.83 → 适合 fine motor control

### 5.2 Generalization (Table 3, 这是真正的故事)

Global Avg (OOD)：
| Method | Global Avg | Drop from ID |
|---|---|---|
| π0.5 | 0.49 | -33.8% |
| VPP | 0.41 | -43.8% |
| Wan+AnyPos | 0.53 | -20.9% |
| QwenVLA-OFT | 0.41 | -34.9% |
| Cosmos-Policy | 0.44 | -43.6% |
| **HarmoWAM** | **0.82** | **-7.9%** |

HarmoWAM 的 OOD drop 只有 7.9%，其他 SOTA 都掉 20-44%。这个数字如果 reproducible 是非常惊人的 —— 通常 OOD drop 在 manipulation 领域是 30-50% 的常客。

**Unseen Position** 这一栏最 dramatic：
- π0.5: 0.32
- Cosmos-Policy: 0.26
- **HarmoWAM: 0.80**

这正好印证了 motivation：position OOD 是 Joint Modeling paradigm 的死穴（受 SFT 数据 spatial coverage 限制），而 HarmoWAM 用 reactive expert + world model 的 generalization 突破了这个限制。

### 5.3 Ablation Studies (Figure 5)

**Architecture ablation**：
- 去掉 reactive expert → position OOD 掉到 14% (从 80% 跌 66 个点) → 验证 reactive expert 是 transit generalization 的来源
- 去掉 predictive expert → position OOD 掉到 56%，object OOD 掉到 60% → 验证 predictive expert 是 interaction precision 的来源

**Gating ablation**：
- Averaging（每个 timestep 把两个 expert 输出平均）→ position OOD 跌 46%
- Keyframe-Based Averaging（仅在 interaction phase 平均）→ 跌 31%
- Process-Adaptive Gating → 跌最少

这说明 **简单的 ensemble 不行，必须有 stage-aware 的 routing**。原因很直觉：transit 阶段 predictive expert 的输出是 "SFT space 内的 action"，会被 averaging 拉回 SFT distribution，破坏 reactive expert 的 OOD generalization。

**World model latent ablation**：
- 去掉 reactive expert 的 video latent → ID 65%, OOD 54%
- 去掉 predictive expert 的 video latent → ID 从 95% 掉到 62%

video latent 对 predictive expert 影响更大 (-33 个点)，说明 **predictive expert 强依赖 world model 的 temporal dynamics 来生成 coherent action chunk**。

---

## 6. 一些深层的 intuition 和联想

### 6.1 这和 Kahneman 的 System 1 / System 2 的关系

这篇 paper 的两-expert 设计让人立刻想到 Kahneman 的 dual-system theory，也让人想到 Fast-in-Slow (https://arxiv.org/abs/2506.01953) 这类工作：

- **Reactive expert ≈ System 1**: fast, interpretable, 基于视觉 cue 的 reactive inference，generalization 来自 pretraining
- **Predictive expert ≈ System 2**: slow, deliberative, 基于 latent dynamics 的 structured planning，precision 来自 SFT

但 HarmoWAM 的 twist 是：**两个 system 共享同一个 world model 作为 "perception grounding"**，所以不是完全分离的两个 system，而是 "shared perception + dual action pathways"。

### 6.2 和 VLA / WAM 谱系的对照

WAM 的 paradigm 谱系（按对 world model 的依赖程度排序）：
1. **Pure VLA** (RT-1, OpenVLA, π0.5): 不用 world model，直接 obs+lang → action
2. **Joint Modeling** (VPP, Cosmos-Policy, UWM): world model latent 作 condition，joint training
3. **Imagine-then-Execute** (WoW, DreamGen, Vidar): 显式生成 future video + IDM
4. **Latent-only WAM** (Fast-WAM, https://arxiv.org/abs/2601.xxxxx): inference 不生成 video，但 latent 仍 supervision

HarmoWAM 同时利用了 paradigm 2 和 paradigm 3 的优势 —— reactive expert 是 paradigm 3 的强化版（加了 latent），predictive expert 是 paradigm 2 的强化版（用 explicit predicted frame 作 supervision signal）。

### 6.3 一个潜在的 concern: Gating 是 hard switch 还是 soft routing?

Paper 用的是 hard switch（threshold 0.5）。这意味着在 transition 帧上（$s_t$ 在 0.5 附近）会有 discontinuity。一个更 smooth 的做法是 soft routing: $\mathbf{a} = s_t \cdot \mathbf{a}^{pred} + (1-s_t) \cdot \mathbf{a}^{react}$。

但作者用 hard switch 可能是 intentional：soft routing 会让两个 expert 互相干扰（参考 ablation 里 Averaging 的失败）。在 action space 里，平均两个不同 distribution 的 action 经常生成 invalid motion。Hard switch 在 boundary 帧上会有 jerk，但用 action chunk (H=12) + 重叠执行 可以平滑掉这个 jerk。

### 6.4 关于 fixed horizon limitation

Paper 在 Section I (Limitations) 自己提到：world model 固定生成 13 帧，下游必须匹配这个 horizon。这其实是所有基于 pretrained video diffusion 的 WAM 通病。一个可能的 fix 是 **adaptive length prediction**，类似 RT-2 的 "action chunk early termination"，或者用 latent-level prediction（避免 pixel-level overhead）。

### 6.5 和 Diffusion Policy 的比较

Predictive expert 本质上是 Diffusion Policy (https://diffusion-policy.cs.columbia.edu/) 的 enhanced 版本，多了 video latent conditioning。它的 1B param 在 diffusion policy 谱系里算大（RDT-1B 是 1B，π0 是 3B，π0.5 也是 3B 级别）。但 48Hz 推理速度说明 action diffusion 部分效率很高，瓶颈在 world model 那 5 步 denoising。

### 6.6 关于 OOD 评测的 validity

Position OOD 测试的设置值得注意：作者把 workspace 分成 training region 和 test region，spatially disjoint。这是一个很强的 OOD 测试，比"位置稍微偏一点"严格得多。在这个 setting 下 π0.5 掉到 0.32 是 reasonable 的 —— 因为 π0.5 的 action head 是 flow matching，对 OOD spatial coordinate 外推能力有限。

但要注意：HarmoWAM 在 position OOD 上能到 0.80，主要功劳是 **world model 见过 1.9M trajectories 后的 spatial generalization**。这意味着 HarmoWAM 的 OOD 能力其实是 world model 的 OOD 能力转移过来的 —— 如果 world model pretraining data 不够大不够 diverse，这个优势会消失。这是一个 scaling-dependent 的 method。

### 6.7 Inference speed 分析

48 Hz with action chunk 12 → 实际 control frequency = 48/12 = 4 Hz 还是 48 Hz 持续输出？从 Table 9 看 world model 单独是 3-4 Hz，所以应该是 action chunk 一次性预测 12 步，然后执行过程中 reactive expert 实时更新（reactive expert 用 DINOv2-base + conv decoder，应该是 100+ Hz）。这就形成了 **慢 world model (4Hz) + 快 reactive expert (100+Hz) + 中速 predictive expert (48Hz)** 的 multi-rate control，非常类似 ACT (https://tonyzhaozh.github.io/aloha/) 的 chunk execution + reactive correction 模式。

### 6.8 联想到的其他工作

- **GR-2** (https://arxiv.org/abs/2410.06158): 也是 video-language-action model，用 web-scale video pretraining，但没有 dual-expert 设计
- **WoW** (https://arxiv.org/abs/2509.22642): Imagine-then-Execute 范式代表，是 paper 比较的 baseline Wan+AnyPos 的源头
- **Motus** (https://arxiv.org/abs/2512.13030): unified latent action world model，joint modeling 范式
- **UWM** (Unified World Models, https://arxiv.org/abs/2501.xxxxx): coupling video and action diffusion for pretraining
- **Cosmos-Policy** (https://arxiv.org/abs/2601.16163): NVIDIA 的 video-action joint modeling，HarmoWAM 的直接 competitor
- **WorldVLA** (https://arxiv.org/abs/2506.21539): autoregressive action world model

### 6.9 关于 Gating label 的 bootstrapping 问题

Gating 的训练 label 是从 proprioceptive signal 自动生成的（gripper state change + end-effector height）。这是一个 weak supervision —— 它假设 "interaction phase = gripper state change 时刻"，但有些任务（比如推盒子、滑动门）gripper 状态不变，仍然是 interaction。这种情况下 gating 会失效。

Paper 里测的 6 个 task 都涉及 grasp/release，所以这个 heuristic 工作。但推广到 contact-rich without gripper change 的 task（比如 in-hand manipulation）需要重新设计 label heuristic。这是一个 generalization 上的隐患。

### 6.10 关于 attention map 的解读 (Figure 3)

Paper 显示：
- Predictive expert 的 attention 集中在 manipulated object（精细控制关注目标）
- Reactive expert 的 attention 集中在 gripper + 周围环境（transit 需要避障 + 定位自身）

这从 cognitive science 角度非常 intuitive：精细动作关注 "what I'm acting on"，粗运动关注 "where I am and what's around"。这和 ventral/dorsal stream 的 "what vs. where" pathway 分工有异曲同工之妙 (https://en.wikipedia.org/wiki/Two-streams_hypothesis)。

---

## 7. 总结: 这篇 paper 的核心贡献和局限

### 贡献
1. **Empirical finding**: 系统性地 expose 了 WAM 两个 paradigm 的 trade-off (Table 1)，这个 finding 本身对 community 很有价值
2. **Architectural innovation**: 用一个 shared world model + dual expert + gating 的方式 reconcile 两个 paradigm，而不是发明一个新 paradigm
3. **Real-world generalization**: 7.9% OOD drop 是一个非常 strong 的 number

### 局限
1. World model 固定 horizon (13 frames)
2. Gating 的 label heuristic 假设 interaction = gripper state change，对 in-hand manipulation 等任务不通用
3. 依赖 large-scale world model pretraining (~1.9M trajectories)，scaling-dependent
4. Hard switch gating 在 transition 帧上的 jerk 问题（虽然 chunk execution 缓解了）
5. Action space 是 7-DoF (xyz + euler + gripper)，对更高 DoF (比如 dexterous hand) 的扩展性未知

### 我的整体评价

这是一篇 architectural combination paper，本质贡献是"把两件事拼起来 + 用 gating 选谁说话"。但这个 combination 是 motivated by a clean empirical finding，而且 execution 上做得很 solid（real-world 6 个 task，3 种 OOD，extensive ablation）。在 WAM 这个方向里属于 "first to systematically identify and address the trade-off" 的工作，预计会被后续工作 follow。

特别有启发的是 **"shared world model + complementary action experts"** 这个 pattern —— 它让我想到 Mixture-of-Experts (MoE) 在 LLM 里的设计：shared attention + expert FFN，由 router 决定哪个 expert 接管。HarmoWAM 可以看作 **"robot control 的 MoE：world model 是 shared attention，两个 expert 是 FFN，gating 是 router"**。这个 analogy 如果延伸下去，未来可能有多于两个 expert 的设计（比如再加一个 "safety expert"、"compliance expert" 等），形成一个 multi-expert WAM。

References:
- Project page: https://elbb-yu.github.io/HarmoWAM/
- Wan2.2: https://arxiv.org/abs/2503.20314
- AnyPos: https://arxiv.org/abs/2507.12768
- DROID: https://arxiv.org/abs/2403.12945
- AgiBot World: https://arxiv.org/abs/2503.06669
- RoboMIND: https://arxiv.org/abs/2412.13877
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP: https://arxiv.org/abs/2303.15343
- π0.5: https://arxiv.org/abs/2504.16054 (Physical Intelligence)
- Flow Matching: https://arxiv.org/abs/2210.02747
- Perceiver-Actor (keyframe): https://proceedings.mlr.press/v205/shridhar23a.html
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ACT: https://tonyzhaozh.github.io/aloha/
- WoW: https://arxiv.org/abs/2509.22642
- Cosmos-Policy: https://arxiv.org/abs/2601.16163
- GR-2: https://arxiv.org/abs/2410.06158
- Fast-in-Slow: https://arxiv.org/abs/2506.01953
- Two-streams hypothesis (ventral/dorsal): https://en.wikipedia.org/wiki/Two-streams_hypothesis
