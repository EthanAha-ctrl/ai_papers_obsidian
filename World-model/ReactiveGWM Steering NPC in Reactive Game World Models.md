---
source_pdf: ReactiveGWM Steering NPC in Reactive Game World Models.pdf
paper_sha256: c184f63c7b6951d73bfa00bb7d5ec96da0864fa305c4999c1c824e4ef564e888
processed_at: '2026-08-11T21:14:45-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ReactiveGWM

好，我换个说法，用更直觉的方式重新捋一遍。核心故事其实很简单，但里面藏着几个 clever 的工程决策。

---

## 一句话版本

现在的 game world model 把 NPC 当背景画，ReactiveGWM 给 NPC 装了个"策略脑"，而且这个脑可以拆下来装到别的 game 上用。

---

## 问题出在哪

你想想 GameNGen 跑 DOOM 的场景。你按键盘，doom guy 动，demon 也动。但模型学到的是"demon 的动作是 doom guy 动作的固定后续"——它没有 demon 的独立 input channel。demon 就像个木偶，你按什么键，它就按训练数据里统计出来的模式动。

这有什么问题？**demon 没有自己的意图**。你没法跟模型说"让 demon 变得更激进"或"让 demon 保守一点"。demon 的行为完全被 player action 的统计分布锁死了。

真实 game 里的 NPC 是有自己的 strategy 的——有的喜欢 rush，有的喜欢 keep distance 放飞行道具，有的喜欢 turtle 等你露出破绽。这种 strategy-level 的自主性在现有 world model 里完全缺失。

ReactiveGWM 做的事情就是：**给 NPC 开一条独立的 strategy 通道**。

---

## 数据怎么搞的——这是最 undervalued 的部分

你要训一个模型让它 follow NPC strategy，你得有 (video, player action, NPC strategy label) 三元组。前两个好搞，emulator 一跑就有。但 NPC strategy label 怎么标？

naive 做法：让 Gemini 看视频，直接说"这个 NPC 在 Offense 还是 Defense"。问题——Gemini 会 hallucinate，而且 inconsistency 很高。同一段视频跑两次可能得到不同 label。

ReactiveGWM 的做法分两步：

**Step 1**: Gemini 只看视频回答 12 个 **factual** 问题，比如"Guile 发了几个 Sonic Boom"、"Guile 是不是在 close range"、"谁攻击更多"。关键——**prompt 明确禁止 Gemini 说 strategy 名字**。它只能报事实。

**Step 2**: 一组 deterministic 的 if-else rule 把 facts 映射到 Offense/Control/Defense。

这个设计的精妙在于——**VLM 的不可靠性被限制在 fact 层**。即使 Gemini hallucinate 了某个 fact，rule engine 的映射仍然是 reproducible 的。你把"语义判断"这个最不可靠的环节尽量压缩，把"事实采集"结构化成 closed-set question，整个 pipeline 的 reliability 就上来了。

这跟你做 autonomous driving 数据标注的直觉应该一样——让 human 标"这辆车要变道"很 subjective，但让 human 标"这辆车的 left turn signal 亮没亮"就 objective 多了。ReactiveGWM 用了同样的思路。

---

## 架构的不对称设计——为什么 action 和 strategy 走不同的路

这是我觉得最 elegant 的部分。

Player action 是 10-dim 的 binary vector，每帧都有，信息量极小但极 dense。NPC strategy 是 3-way category 加一些 active/passive 行为描述，整个 clip 只有一个，信息量在语义层但极 sparse。

ReactiveGWM 用完全不同的方式注入这两个 signal：

**Player action → additive bias**

$$x^{(\ell)} \gets x^{(\ell)} + E_\ell(\bar{a}) \otimes \mathbf{1}_{h \times w}$$

变量解释：
- $x^{(\ell)}$ 是第 $\ell$ 个 DiT block 的 video latent，shape $[B, L, C]$，$L = f \times h \times w$ 是 token 数
- $E_\ell: \mathbb{R}^K \to \mathbb{R}^C$ 是一个 bias-free linear projection，把 10-dim action 映射到 channel dimension
- $\bar{a}$ 是 max-pooled action，shape $[B, f, K]$，$f$ 是 latent 帧数
- $\mathbf{1}_{h \times w}$ 是把每帧的 action embedding 在空间维度 broadcast（复制 $h \times w$ 次）
- 最终加到 residual stream 上

为什么用 add 而不是 cross-attention？几个直觉：

1. action 信息量太小，cross-attention 的 KV projection 是浪费。10-dim vector 用一个 linear 就够了。
2. action 是 spatial-global 的——按 LEFT 影响整个角色的 motion，不是某个 patch 的事。broadcast 到所有 patch 是合理的 inductive bias。
3. cross-attention 会引入 spatial attention 的参数，让模型有可能学到"action 只影响某个区域"这种无意义的 spurious correlation。

**NPC strategy → cross-attention**

strategy 走标准的 cross-attention。strategy prompt 是文本，有 semantic structure，需要 attention 来做 alignment。这条 channel 带宽大，能学到"Offense 这个词 → 主动接近对方"这种语义映射。

这个不对称设计的核心 insight 是：**control signal 的抽象层级决定了注入方式**。低层 dense signal 用 add，高层 sparse semantic 用 attention。

---

## 最神奇的发现——cross-attention 可以跨游戏 transplant

这是 paper 最 surprising 的 claim。在 SF2 上训好的 cross-attention module，直接 plug 进 SF3 的 vanilla model（SF3 vanilla model 只有 player action conditioning，没有 NPC strategy conditioning），SF3 model 立刻就能 follow NPC strategy。

这听起来违反直觉——SF2 和 SF3 是不同 game，character 不同，animation 不同，physics 不同，凭什么 cross-attention 能直接用？

paper 4.4 节做了 mechanistic 分析。我换个角度讲：

一个 DiT block 的 residual stream 由 3 个 pathway 更新：

$$x_\ell \gets x_\ell + \mathbf{SA}_\ell + \mathbf{CA}_\ell + \mathbf{FFN}_\ell$$

- $\mathbf{SA}$：self-attention，学 visual-temporal dynamics（谁在哪、怎么动）
- $\mathbf{CA}$：cross-attention，学 NPC strategy grounding（strategy 文本 → visual hint）
- $\mathbf{FFN}$：feed-forward，学非线性变换

paper 测了 cross-attention 占总能量比例：

$$\rho_\ell^{\text{cross}} = \frac{\|\mathbf{CA}_\ell\|^2}{\|\mathbf{SA}_\ell\|^2 + \|\mathbf{CA}_\ell\|^2 + \|\mathbf{FFN}_\ell\|^2}$$

结果：cross-attention 只占 **0.71%** 的能量。剩下 99.3% 都在 SA 和 FFN。

这意味着什么？cross-attention 是个**极低带宽 channel**。主视觉信号（角色 sprite、animation、physics）全在 SA + FFN 里。所以你把 SF2 的 cross-attention 拿出来装到 SF3 model 上，SF3 的 SA + FFN 还是 SF3 自己的，主视觉完全不受影响。

但 0.71% 能量这么小，真的能 steer NPC 行为吗？

paper 又测了 cross-attention 输出的方向变化：transferred model 的 $\mathbf{CA}^T$ 和 vanilla model 的 $\mathbf{CA}^V$ 的 cosine similarity 只有 0.55。

Intuition：cross-attention 输出虽然在 magnitude 上很小（0.71% 能量），但它的**方向**跟 vanilla 不一样。这个新方向是 SF2 上学到的 "strategy → visual hint" 映射。经过 30 个 DiT blocks 累积 × 30 个 diffusion steps 累积，这个 directional signal 足够把 NPC 的 trajectory 推向符合 strategy 的区域，但不足以破坏主视觉 token。

这让我想到 LoRA 的 insight——一个低 magnitude 但有特定方向的 perturbation，在深度网络里通过累积可以产生显著 output 变化。ReactiveGWM 的 cross-attention 替换本质上是一种 "module-level LoRA"，只不过它替换整个 weight matrix 而非加一个低秩 delta。

**为什么 cross-attention 能学到 game-agnostic 的东西？**

因为 strategy 概念本身是 game-agnostic 的：
- Offense = "主动接近 + 近身攻击"
- Control = "保持距离 + 放飞行道具"
- Defense = "被动防御 + 反应式"

这些 concept 跟具体 button mapping 或 character 无关。cross-attention 只需要学 "strategy 文本 → visual hint" 的语义对齐，不需要学 game physics。而 game physics 由 SA + FFN 学，这些 module 留在 target game 的 vanilla model 里。

**Intuition 总结**：strategy 是抽象的，physics 是具体的，二者在 architecture 层面就解耦了。cross-attention 学抽象层，SA/FFN 学具体层。transfer 时只搬抽象层，具体层保持 target game 自己的。

---

## 实验结果讲了什么

主结果表的关键 takeaways：

| 维度 | Vanilla | ReactiveGWM_base | ReactiveGWM_transfer | 解读 |
|------|---------|------------------|-----------------------|------|
| NPC Strategy (SF2) | 43% | 76% | 65% | strategy 通道确实 work |
| NPC Strategy (SF3) | 42% | 80% | 64-74% | transfer 也 work |
| Player Action | 97-100% | 95-100% | 95-100% | action control 几乎无损 |
| Visual Quality | baseline | 几乎一样 | 几乎一样 | strategy 不破坏 visual |

**1. NPC Strategy Following 跳跃式提升**：从 ~43% 到 ~76-80%。这是 paper 的 main result，证明 strategy prompt 确实能给 NPC 一个 explicit 信号。

**2. Player Action Control 几乎无损**：additive bias 注入 action 不干扰 backbone 的 action-conditioned generation 能力。这验证了 architecture 设计的 correctness——action 和 strategy 两条 channel 真的解耦了。

**3. Visual Quality 完全持平**：SSIM 和 LPIPS 几乎没变化。strategy conditioning 没有引入 visual artifact。这跟 4.4 节的能量分析一致——cross-attention 能量太小，不足以影响主视觉。

**4. Transfer 模型 strategy 准确率有所下降但仍然显著**：比 vanilla 高 20+ 个百分点，但比 base model 低 10-15 个百分点。这符合直觉——transfer 一定有 loss，但 loss 在可接受范围内。

**5. 用户研究揭示了 transfer 的弱点**：SF3 transfer 的 Control 类准确率只有 16%，但 Offense 是 100%。作者解释 Control 依赖 game-specific ranged attacks（Sonic Boom vs airborne projectile），动画、timing、trajectory 都不同，所以 transfer 不如 general 的 Offense/Defense 直接。这是个 honest limitation。

---

## 跟你 Tesla 工作的潜在呼应

你之前在 Tesla 做 autonomous driving 时，world model 也是核心议题。ReactiveGWM 的 factorization——"agent intent (high-level) + physics dynamics (low-level) 解耦"——跟 autonomous driving 里的 "agent prediction (intent) + scene dynamics" 解耦有精神上的相似。

其他 vehicle 是 reactive agents，跟 NPC 类似，需要预测它们的行为。如果用 ReactiveGWM 的思路做 driving sim：
- Ego vehicle action 通过 additive bias 注入（低带宽、vehicle-specific）
- Other vehicles 的 intent（lane change, merge, brake）通过 cross-attention 注入（高带宽、generic semantic）
- Scene physics（road, traffic light, weather）由 SA + FFN 学

这种 factorization 让 intent prediction module 可以跨场景 transfer（highway → urban → parking lot），而 scene dynamics 需要每个场景单独训。

---

## 我的几个疑问

### 1. Action Module 的 spatial broadcast 假设

action embedding 在空间维度 broadcast 到所有 patch，意味着同一帧的所有 patch 都接受**相同**的 action bias。这对 fighting game 合理（角色是 single entity，action 是 global），但对 multi-entity scene（FPS 里的多 NPC）就不一定了——每个 NPC 应该有独立的 action。

可能的方向：把 action 表示成 per-entity 的，用 cross-attention 让每个 entity token attends to 自己的 action embedding。或者用 pointer-style injection，action embedding 通过 attention 路由到 spatial location。

### 2. Strategy 只有 3 个 category 是不是太粗

Offense/Control/Defense 是 mutually exclusive 的三分类，但真实 fighting game 的 strategy 谱系远比这复杂（zoning, rushdown, turtle, mix-up, frame trap, okizeme...）。三分类的好处是清晰可学，坏处是表达力有限。

paper 在 prompt 里加了 Active 和 Passive 行为描述来补充 strategy 信号，这相当于把 strategy 拆成 (high-level category, low-level action composition) 的 hierarchical 结构。

可能的方向：把 strategy 表达成 continuous embedding 而非 3-way category，让模型自己 learn strategy manifold。或者用 LLM 生成 free-form strategy description，cross-attention 学更丰富的 alignment。

### 3. Transfer 的边界条件

paper 的 transfer 是 SF2 → SF3（同 series、同 genre、同 developer Capcom）。这两个游戏的 visual style、character design、move set 都很相似。所以 strategy module 的 transfer 严格说是在**"近邻游戏"**之间的 transfer，不是"任意游戏之间"。

真正的 game-agnostic test 应该是：
- SF (2D fighter) → Tekken (3D fighter)
- SF (fighter) → Mario (platformer)
- SF (fighter) → StarCraft (RTS)

跨 genre 的 strategy 概念可能完全不同（RTS 的 "macro/micro" 跟 fighter 的 "offense/defense" 没法对应），可能需要更 abstract 的 strategy representation。

### 4. VLM referee 的潜在 bias

NPC Strategy Following 评估依赖 VLM referee（Gemini + Qwen3-VL）。VLM 本身可能有自己的 bias（比如更容易把"接近对方"判为 Offense）。两个 VLM 的 ensemble mitigate 了一些，但还是个潜在 confound。

更严谨的评估可能是用 RL agent 做 opponent，看 NPC strategy 是否能 win 对应的 strategy matchup。或者用 emulator 暴露的 RAM 做 ground-truth state-based label 而非 VLM 判断。stable-retro 应该能 expose RAM，paper 没用这个有点可惜。

### 5. Diffusion latency 是个 real problem

paper 在 limitations 里提到 diffusion-based backbone 引入 high inference latency，无法 real-time interactive。这个限制对 game 来说是 dealbreaker——真正的 game engine 需要至少 30 fps。

可能的方向：
- DiT 改成 AR，每帧 inference 只需一次 forward（参考 GameNGen 的 real-time 部分）
- 用 consistency model 或 LCM 把 30-step diffusion 蒸馏到 1-4 step
- Hybrid：diffusion 训练 + AR 推理

---

## 跟相关 work 的定位

ReactiveGWM 站在两个脉络的交叉点上：

**World models for RL 谱系**：Ha & Schmidhuber 2018 [https://arxiv.org/abs/1803.10122] → Dreamer [https://arxiv.org/abs/1912.01603] → DreamerV2 [https://arxiv.org/abs/2010.02193] → DreamerV3 [https://arxiv.org/abs/2301.04104] → Genie 2 [https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/] → Genie 3。这些工作主要用 world model 做 policy learning，agent 在 latent imagination 里 rollout。

**Game world models 谱系**：DIAMOND [https://arxiv.org/abs/2412.09923] → GameNGen [https://arxiv.org/abs/2408.14837] → Oasis [https://oasis-model.github.io/] → Matrix-Game → LingBot-World → GameFactory [https://arxiv.org/abs/2504.01343]。这些工作主要用 diffusion 做 real-time game rendering。

ReactiveGWM 把这俩脉络 merge 了一下：用 diffusion 做 game rendering（第二个谱系的技术），但 explicitly model NPC 作为 autonomous agent（第一个谱系的精神）。

跟 ControlNet [https://arxiv.org/abs/2302.05543] 的关系：ControlNet 把 condition 通过 zero-conv 加到 frozen backbone，类似 modular 思路，但 ControlNet 是 train-from-scratch，不是 transfer。ReactiveGWM 的 cross-attention 替换更接近 IP-Adapter [https://arxiv.org/abs/2308.06721] 的 decoupled cross-attention 思路，但跨模型 transfer。

跟 LoRA [https://arxiv.org/abs/2106.09685] 的关系：LoRA 是低秩 perturbation，ReactiveGWM 的 cross-attention 替换可以理解为 "module-level LoRA"，只不过它替换整个 weight matrix 而非加一个低秩 delta。

---

## 总结

ReactiveGWM 的核心 contribution 可以浓缩成三层：

**1. Conceptual**：把 NPC 从 background pixel 升级为 autonomous agent，通过显式的 strategy conditioning channel 实现 reactive game world model。这回到了 Ha & Schmidhuber 原始 world model 的精神——model 环境里的所有 agent，而非只 model 主角。

**2. Technical**：action（low-bandwidth additive bias）和 strategy（high-bandwidth cross-attention）的 asymmetric injection，让两者解耦且互不干扰。这个设计直接对应了 control signal 的抽象层级——低层 dense signal 用 add，高层 sparse semantic 用 attention。

**3. Practical**：cross-attention module 的 game-agnostic transferability，让 strategy module 可以 plug-and-play 进不同 game 的 vanilla model。这背后的 mechanism 是 cross-attention 只占 0.71% 能量但能引入新的 directional signal，既不破坏主视觉又能 steer NPC 行为。

最让我印象深刻的是 4.4 节的 transfer 分析——通过能量占比和 cosine similarity 的定量测量，把 "为什么 transfer work" 这个看似黑盒的现象 demystify 了。这种 mechanistic analysis 在 game world model 文献里很少见，给后续工作提供了很好的 intuition。

**Web References 汇总**：
- ReactiveGWM 主页：https://inv-wzq.github.io/ReactiveGWM/
- Wan2.2：https://arxiv.org/abs/2503.20314
- GameNGen：https://arxiv.org/abs/2408.14837
- DIAMOND：https://arxiv.org/abs/2412.09923
- World Models (Ha & Schmidhuber)：https://arxiv.org/abs/1803.10122
- DreamerV3：https://arxiv.org/abs/2301.04104
- Genie 2：https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Oasis：https://oasis-model.github.io/
- Matrix-Game：https://matrix-game.github.io/
- ControlNet：https://arxiv.org/abs/2302.05543
- IP-Adapter：https://arxiv.org/abs/2308.06721
- LoRA：https://arxiv.org/abs/2106.09685
- SAM2：https://arxiv.org/abs/2408.00714
- Grounding DINO：https://arxiv.org/abs/2303.05499
- LPIPS：https://arxiv.org/abs/1801.03924
- stable-retro：https://github.com/Farama-Foundation/stable-retro
- Consistency Models：https://arxiv.org/abs/2303.01469
- Latent Consistency Models：https://arxiv.org/abs/2310.15105
- GameFactory：https://arxiv.org/abs/2504.01343

如果你想 drill down 到某个具体点——比如那个 max pooling 的 temporal alignment 细节、或者 VLM prompt 的 decision order 设计、或者 transfer 能量分析背后的数学直觉——我们可以继续展开。

---

# ReactiveGWM 深度解读：让 NPC 从背景像素升级为可策略控制的 agent

Andrej，这篇 paper 我觉得有几个有意思的点值得展开。它的核心 insight 跟你早期做的 world models 工作（Ha & Schmidhuber 2018）有精神上的呼应——**world model 应该是模拟环境的 engine，不是被动渲染视频的 renderer**——只不过这次是用 diffusion transformer 在 game 领域重做了一遍。下面我按"问题→数据→架构→transfer 机制→实验"的顺序拆解。

---

## 1. 这篇 paper 在解决什么问题：player-centric 的盲点

现有 game world models 的输入条件 vocabulary 几乎一致地只包含 **primary player 的 action stream**。比如 GameNGen [https://arxiv.org/abs/2408.14837]、DIAMOND [https://arxiv.org/abs/2412.09923]、Matrix-Game [https://matrix-game.github.io/]、Oasis [https://oasis-model.github.io/]、LingBot-World、GameFactory——它们的 generation 公式可以抽象成：

$$\mathbf{x}_{1:T} = \mathcal{F}_{\text{vanilla}}(x_0, \mathbf{a}_T, \mathcal{P}_{\text{vanilla}}) \tag{1}$$

变量说明：
- $x_0 \in \mathbb{R}^{H \times W \times 3}$ 是初始观察帧；
- $\mathbf{a}_T = \{a_0, a_1, \ldots, a_{T-1}\}$ 是 player 在 $T$ 步内的离散动作序列；
- $\mathcal{P}_{\text{vanilla}}$ 是描述整个场景的文本 prompt，里面把 background、player、NPC 的事件都揉在一起；
- $\mathbf{x}_{1:T}$ 是生成的未来帧序列。

问题在哪？$\mathcal{P}_{\text{vanilla}}$ 把 player 动作和 NPC 行为 entangle 进了同一条 descriptive 文本通道。NPC 没有自己独立的 decision channel，它的行为只能被当作"player 动作导致的背景变化"被动地学习。从训练分布的统计意义上讲，NPC 就是一个固定 action-conditioned 像素图样的 nuisance variable。这本质上把 world model 退化成了 "video renderer"，没有 capturing 真正的 reactive agent dynamics。

ReactiveGWM 的 reformation 是把 conditioning 拆开：

$$\mathbf{x}_{1:T} = \mathcal{F}(x_0, \mathbf{a}_T, \mathcal{P}_{\text{NPC}}) \tag{2}$$

其中 $\mathcal{P}_{\text{NPC}}$ **只**指导 NPC 的高层 strategy，不描述具体场景事件。这就把"player 的低层动作" 和 "NPC 的高层战术意图" 解耦成两条独立的 conditioning channel。

这条 reformulation 看起来简单，但实际让 training 分布完全不一样。Vanilla 模型学的是 $p(\mathbf{x} | \text{player action, scene description})$，而 ReactiveGWM 学的是 $p(\mathbf{x} | \text{player action}, \text{NPC strategy intent})$——前者把 NPC 行为当 deterministic 映射，后者把 NPC 当作可以由 strategy guidance 控制的 autonomous agent。

---

## 2. 数据构造：精妙之处在"两阶段标注"

**这一节我认为是整篇 paper 最被低估的部分**，因为它的设计直接决定了 transfer 能不能 work。

### 2.1 数据采集

使用 stable-retro [https://github.com/Farama-Foundation/stable-retro] 框架跑 SF2 [https://www.capcom.co.jp/] 和 SF3。Random agent 从 13 个 discrete behaviors (ID 0-12) 均匀采样，覆盖 idle、4 个 directional、6 个 attack (LP/MP/HP/LK/MK/HK) 和 2 个 jump direction。

关键细节是 **EDGE/HOLD 编码**：
- 方向键 (LEFT/RIGHT/DOWN) 在 10-frame decision block 内 **hold**；
- Attack 键和 UP 键在 block 开始的 **单帧 edge press**；
- 一个 decision block = 10 个 video frames。

这种不对称编码对 fighting game 来说是必须的——按住方向键 1 帧和 10 帧的语义差异巨大（前者是 step，后者是 walk），但 punch 按住 10 帧和按 1 帧在 game logic 里是同一个 action。EDGE/HOLD 编码本质是把"持续控制量"和"瞬时触发量"区分开。

### 2.2 两阶段 NPC strategy 标注

这是真正聪明的地方。作者没有直接让 VLM "判断" strategy，因为 VLM 容易 hallucinate。他们用 **"factual observation → deterministic classification"** 的两阶段 pipeline：

**Stage 1**: Gemini 看 5s clip，回答 **12 个 closed-set factual question**（表 4），比如：
- `guile_does_punch: yes/no`
- `guile_sonic_boom_count: 0/1/2+`
- `guile_engagement_range: close/mid/far`
- `guile_crouches_guard: yes/no`
- `who_attacks_more: ryu/guile/both/neither`

**关键约束**：prompt **explicitly forbids VLM from naming a strategy**——VLM 只能报告 observable facts。

**Stage 2**: 一个 deterministic rule engine 把 facts 映射到 3 个互斥 categories：
- **Offense**: `range=close ∧ advances=yes ∧ has_melee ∧ sonic_boom=0 ∧ pressure=yes ∧ who=guile`
- **Control**: `range∈{mid,far} ∧ advances=no ∧ sonic_boom≥1 ∧ ¬has_melee ∧ takes_damage=no`
- **Defense**: `¬has_melee ∧ sonic_boom=0 ∧ crouches_guard=yes ∧ who∈{ryu,neither} ∧ active⊆{Crouch,WalkL,WalkR} ∧ passive∩D≠∅`

匹配不到任何 rule 的 clip 直接被丢弃。

**为什么这个 pipeline 重要？** 因为它隔离了 VLM 的不可靠性。如果 VLM 直接输出 strategy label，任何 hallucination 都会污染 label。但在这个 pipeline 里，VLM 的 hallucination **最多只能污染 facts**，而 deterministic rule engine 保证 facts→label 的映射是 reproducible 的。这是"semi-supervised 的弱信任"原则：把"语义判断"部分尽量小化，把"事实采集"部分尽量结构化。

最终的 prompt 模板是：

$$\mathcal{P}_{\text{NPC}} = \{\text{Active}(b_1:d_1; \ldots), \text{Passive}(b_1':d_1'; \ldots), \text{Strategy}(c:\delta_c)\} \tag{3/6}$$

其中：
- $b_i$ 是 active 行为 tags（punch, kick, projectile 等），$d_i$ 是它们的 description；
- $b_i'$ 是 passive 行为 tags（block, hit-stun 等）；
- $c \in \{\text{Offense, Control, Defense}\}$ 是 Stage 2 label；
- $\delta_c$ 是从 per-category paraphrase pool 中 **用 MD5(video_path) mod |pool_c|** 选出来的 natural-language 描述。

最后这个 MD5 hash 选择的设计很巧——它保证 **bit-wise reproducibility**（同一 clip 永远得到同一个 paraphrase），但跨 clips 仍然让模型看到同一 strategy 的不同 surface form。这避免了模型 overfit 到某一句固定描述，强迫 cross-attention 学到的是 strategy 的语义而非 lexical pattern。这种"reproducible randomization"在 self-supervised learning 里是个常见 trick（参考 SimCLR [https://arxiv.org/abs/2002.05709] 的 multi-crop，或者 BYOL [https://arxiv.org/abs/2006.07733] 的 augmentation 分布设计）。

---

## 3. 模型架构：为什么用 additive bias 而不是 cross-attention 注入 action

### 3.1 Action injection：lightweight additive bias

ReactiveGWM 用 Wan2.2-TI2V-5B [https://arxiv.org/abs/2503.20314] 作为 backbone（DiT-based video diffusion）。Player action 的注入方式很关键。

先做 temporal alignment：原始 button 序列 $\bar{a}_{1:T} \in \{0,1\}^{T \times K}$ 在 $T$ 个 video frames 上有定义，但 VAE 把时间压缩了 $T_v$ 倍，latent 长度 $f = T / T_v$。所以用 **adaptive max-pooling along time axis** 把 $T$ frames 分成 $f$ 个 bin：

$$\bar{a}_{i,k} = \max_{t \in \mathcal{B}_i} a_{t,k}, \quad i \in [0, f), \; k \in [0, K)$$

每个 bin $\mathcal{B}_i = [\lfloor iT/f \rfloor, \lceil (i+1)T/f \rceil)$ 内取 **max**。这里用 max 而不是 mean 是个有趣的 choice——在 fighting game 里，一个 1-frame 的 punch press 在该 latent bin 内也应该被 detected，max pooling 保留了这种 sparse 但 important 的 signal；mean 会被稀释。

然后每个 DiT block $\ell$ 接一个独立的、**bias-free 的 linear projection** $E_\ell: \mathbb{R}^K \to \mathbb{R}^C$，把 action 表示映射到 hidden channel dimension $C$。再做 **spatial broadcast** 到 $h \times w$ patch grid，得到 shape $[B, L, C]$（其中 $L = f \times h \times w$）。最后加到 video latent $\boldsymbol{x}^{(\ell)}$ 的 residual stream：

$$x^{(\ell)} \gets x^{(\ell)} + E_\ell(\bar{a}) \otimes \mathbf{1}_{h \times w} \tag{4}$$

变量说明：
- $x^{(\ell)} \in \mathbb{R}^{B \times L \times C}$ 是第 $\ell$ 个 DiT block 的 video latent；
- $E_\ell(\bar{a}) \in \mathbb{R}^{B \times f \times C}$ 是 action embedding；
- $\mathbf{1}_{h \times w}$ 是把每个时间帧的 action embedding 在空间维度复制 $h \times w$ 次（broadcasting）；
- $\otimes$ 这里是 outer-product 风格的 broadcast。

**为什么不直接 cross-attention 把 action 喂进去？** 我的猜测：

1. **信息 bottleneck 角度**：action 是 10-dim discrete vector，信息量极小，cross-attention 的 KV projection 是浪费。linear projection + add 是最小带宽的注入方式。
2. **Spatial equivariance**：player action 在 spatial 上是 global 的（按 LEFT 影响整个 player character 的 motion），broadcast 到每个 patch token 是合理的先验。cross-attention 反而会引入 spatial-attention 的无意义参数。
3. **与 control net 哲学相关**：这个设计跟 ControlNet [https://arxiv.org/abs/2302.05543] 或 T2I-Adapter [https://arxiv.org/abs/2302.03863] 类似——conditioning signal 用 zero-conv 或 lightweight projection 注入，主 backbone 保持冻结或仅小幅扰动。

这种设计选择直接对应了 paper 4.4 节的 transfer 分析：因为 action module 是 lightweight additive bias，它在 transfer 时保留 source 模型的视觉/物理 dynamics 没有问题。

### 3.2 NPC strategy：通过 cross-attention grounded

而 NPC strategy $\mathcal{P}_{\text{NPC}}$ 通过标准的 **cross-attention** 注入。这个 asymmetry 很关键：
- **Player action**（low-level, dense, game-specific button encoding）→ additive bias（带宽极低、game-specific）；
- **NPC strategy**（high-level, sparse, game-agnostic 语义）→ cross-attention（带宽高、可学习 semantic alignment）。

Cross-attention 在 DiT block 里负责把 textual strategy "grounding" 进 visual-temporal latent。paper 4.4 节的关键发现是：**cross-attention 的能量占比只有 0.71%**，但这个低带宽 channel 仍然能 steer NPC 行为。这个发现很有意思，下面单独展开。

---

## 4. Transfer 机制：为什么 cross-attention 替换能 work

这是 paper 最 surprising 的 claim——把 Game 1 (SF2) 上训好的 cross-attention 直接 plug 进 Game 2 (SF3) 的 vanilla model，**零样本**就能让 SF3 model 执行 NPC strategy。

### 4.1 三个 module 的角色分工

DiT block $\ell$ 内部 residual stream 的更新由 3 个 pathway 构成：

$$x_{\ell} \gets x_{\ell} + \mathbf{SA}_\ell(x_\ell) + \mathbf{CA}_\ell(x_\ell, \mathcal{P}_{\text{NPC}}) + \mathbf{FFN}_\ell(x_\ell + \ldots)$$

- $\mathbf{SA}_\ell$：self-attention，学习 visual-temporal dynamics（spatiotemporal interaction）；
- $\mathbf{CA}_\ell$：cross-attention，学习 NPC strategy grounding；
- $\mathbf{FFN}_\ell$：feed-forward，学习非线性变换。

paper 测量 cross-attention 的能量占比：

$$\rho_\ell^{\text{cross}} = \frac{\|\mathbf{CA}_\ell\|^2}{\|\mathbf{SA}_\ell\|^2 + \|\mathbf{CA}_\ell\|^2 + \|\mathbf{FFN}_\ell\|^2} \tag{5}$$

变量说明：
- $\|\cdot\|^2$ 是 Frobenius norm 的平方，衡量该模块输出的"信号强度"；
- $\rho_\ell^{\text{cross}} \in [0, 1]$ 是 cross-attention 在该 block 的能量份额。

测量结果：$\overline{\rho^{\text{cross}}} = 0.71\%$ for ReactiveGWM_transfer，几乎和 Vanilla 模型的 0.70% 一样。也就是说，**99.3% 的 residual stream 能量都在 SA + FFN**，主视觉组件完全由它们决定。

这解释了为什么 **visual preservation** 成立——cross-attention 是结构上的 low-bandwidth channel，替换它对主 visual 信号扰动很小。

### 4.2 Directional difference：为什么 0.55 的 cosine 相似度就够了

但 cross-attention 信号虽小，方向却很重要。定义 directional difference：

$$\Delta_\ell := \mathbf{CA}_\ell^T - \mathbf{CA}_\ell^V$$

其中 $\mathbf{CA}_\ell^T$ 是 transferred 模型的 cross-attention 输出，$\mathbf{CA}_\ell^V$ 是 vanilla 模型的 cross-attention 输出。

测量：$\cos(\mathbf{CA}_\ell^V, \mathbf{CA}_\ell^T) = 0.55$ ——这是一个**适中的相似度**，既不是 1（完全不变，说明没学到新东西）也不是 0（正交，可能破坏 latent 分布）。

**Intuition**：cross-attention 输出虽然在 norm 上很小（0.71% 能量），但它的方向相对 vanilla 发生了显著变化。这个新方向是 source game 上学到的 "strategy → visual hint" 映射。在 30 个 DiT blocks 累积 × 30 个 diffusion steps 累积后，这个 directional 信号足以把 NPC 的 trajectory 推向符合 strategy 的区域，而不足以破坏主视觉 token。

这让我想到 **LoRA [https://arxiv.org/abs/2106.09685]** 的低秩 insight——一个小的 directional perturbation 在深度网络里通过累积可以产生显著的输出变化。ReactiveGWM 的 cross-attention 替换本质上是一种 "module-level LoRA"，只不过它替换整个 weight matrix 而非加一个低秩 delta。

### 4.3 为什么这种 transfer 是 game-agnostic

paper 的 claim 是 cross-attention 学到的是 **game-agnostic 的交互逻辑**。这个 claim 的成立条件是：
- Offense = "approach + close-range melee"
- Control = "keep distance + projectile zoning"
- Defense = "passive guard + reactive"

这些 strategy 在 fighting game 里是 **跨游戏的 invariant concept**，跟具体 button mapping 或 character 无关。Cross-attention 只需要学到 "strategy 文本 → visual hint" 的语义对齐，不需要学 game physics。

而 game physics（角色 sprite、攻击动画、击退距离）由 SA + FFN 学，这些 module 留在 target game 的 vanilla model 里。所以 transfer 后，**新 strategy 注入 + 旧 physics = 在新游戏里执行老 strategy**。

这是个非常 elegant 的 factorization：strategy 是抽象的，physics 是具体的，二者解耦。

---

## 5. 实验设计：三维评估框架

### 5.1 评估维度

paper 设计了一个 3-axis evaluation：

| 维度 | Metric | 评估什么 | 评估集 |
|------|--------|---------|--------|
| Player Action Following | Move-Acc | 4 个方向键的位移 | 100 runs (10 init frames × 10 actions) |
| Player Action Following | Att-Acc | 6 个攻击键的 frame-wise 检测 | 同上 |
| NPC Strategy Following | Categorical Accuracy | 3-way top-1 (Offense/Control/Defense) | 99 curated clips (33/category) |
| Visual Quality | SSIM | 结构相似度 | 99 clips |
| Visual Quality | LPIPS | 感知相似度 (AlexNet backbone) | 99 clips |

**Move-Acc 的细节**：用 SAM2.1 [https://arxiv.org/abs/2408.00714] + Grounding DINO [https://arxiv.org/abs/2303.05499] 做 player character segmentation，提取 bbox。然后基于 normalized coordinate [0,1] 设定阈值：

- LEFT: $x_T - x_0 \leq -0.025$
- RIGHT: $x_T - x_0 \geq +0.025$
- UP: $\min_t(y_t) - y_0 \leq -0.030$ (peak-only，因为 sparse UP 输入可能让角色在 clip 结尾仍 airborne)
- DOWN: $h_{\text{mid}} \leq 0.85 h_0$ **或** $y_{\text{mid}} - y_0 \geq 0.010$（高度变低 OR 中心下移）

DOWN 的两个 alternative 条件挺细致——蹲下不一定让 bbox 中心下移（角色可能站立时中心已经在某个位置），但高度一定会变。两个 condition 用 OR 连接是 robust 的设计。

**Att-Acc 的细节**：训练了 **ClipAttackNet**（ResNet-18 + 4-layer dilated TCN [https://arxiv.org/abs/1803.01271]），6-way attack classifier，在 ~5k labeled clips 上训练。3-stage fine-tuning：
1. Head-only training
2. Unfreeze layer4 + head
3. Unfreeze layer3/layer4 + head

Loss 是 BCEWithLogits，masked by valid frames。Validation checkpoint 选择基于 mean clip IoU @ threshold 0.7。推理时，每帧输出 6-way probability，frame 的 max $p_k > 0.7$ 算 attack-active，clip prediction 是最 confident active frame 的 key。

### 5.2 NPC Strategy Following 用 VLM referee

这个评估用了两个 VLM 作为 referee：
- Gemini [https://arxiv.org/abs/2503.18430]
- Qwen3-VL-8B [https://arxiv.org/abs/2505.09388]

Prompt 设计得很 careful——decision-oriented，要求 JSON-only 输出，并有明确的 **decision order (first match wins)**：
1. Sonic Boom? → Control
2. Extended distance crouch/zoning posture? → Control
3. Sustained forward movement OR >=2 close-range attacks? → Offense
4. Otherwise → Defense

还处理了 edge case：post-match KO animation、broken rendering、NPC missing 都返回 `npc_visible=false`，避免被误判。

**为什么用 VLM 做 referee？** 因为 strategy 是 **contextual judgment**，不是 frame-level pixel measurement。这种 subjective evaluation 用 VLM 比用 fixed metric 更接近人类判断。两个 VLM 的 ensemble 也在一定程度上 mitigate 单一 VLM 的 bias。

### 5.3 主结果表

| Method | Move-Acc | Att-Acc | Gemini | Qwen | SSIM | LPIPS |
|--------|----------|---------|--------|------|------|-------|
| **SF2** | | | | | | |
| Matrix-Game-3.0 | - | - | 3.0 | 24.2 | 0.084 | 0.755 |
| LingBot-World-Base | - | - | 30.3 | 46.5 | 0.142 | 0.679 |
| Vanilla | 97.5 | 96.7 | 43.4 | 44.4 | 0.427 | 0.315 |
| ReactiveGWM_base | 95.0 | 93.3 | 75.8 | 76.8 | 0.428 | 0.319 |
| ReactiveGWM_transfer | 97.5 | 93.3 | 64.6 | 64.6 | 0.421 | 0.318 |
| **SF3** | | | | | | |
| Matrix-Game-3.0 | - | - | 32.5 | 32.3 | 0.117 | 0.685 |
| LingBot-World-Base | - | - | 33.5 | 40.9 | 0.202 | 0.572 |
| Vanilla | 100.0 | 100.0 | 41.8 | 49.5 | 0.392 | 0.397 |
| ReactiveGWM_base | 100.0 | 100.0 | 79.8 | 78.8 | 0.394 | 0.391 |
| ReactiveGWM_transfer | 95.0 | 100.0 | 63.6 | 73.7 | 0.367 | 0.414 |

**几个关键观察**：

1. **NPC Strategy Following 跳跃式提升**：SF2 从 43% → 76%，SF3 从 42% → 80%。这是 paper 的 main result，证明 strategy prompt 确实能给 NPC 一个 explicit 信号。

2. **Player Action Control 几乎无损**：SF3 上 ReactiveGWM_base 甚至保持了 100% Move-Acc 和 Att-Acc。这证明 additive bias 注入 action 不会干扰 backbone 的 action-conditioned generation 能力。SF2 上略微下降（97.5→95.0, 96.7→93.3），但仍在误差范围内。

3. **Visual Quality 完全持平**：SSIM 和 LPIPS 几乎没变化（0.427 vs 0.428, 0.315 vs 0.319 on SF2）。这说明 strategy conditioning 没有引入 visual artifact。

4. **Transfer 模型的 strategy 准确率有所下降但仍然显著**：SF2 transfer 是 64.6%，SF3 transfer 是 63.6-73.7%。比 vanilla 高 20+ 个百分点，证明 transfer 是 work 的。

5. **Matrix-Game-3.0 和 LingBot-World-Base 在 SF2/SF3 上效果很差**：因为这些 baseline 不是为 fighting game 设计的，仅作为 reference。

### 5.4 用户研究

19 个 familiar with 2D fighting games 的参与者。

**NPC Strategy Following 的人类评估**：
- SF2: Vanilla 43.9% → ReactiveGWM_base 86.0% → ReactiveGWM_transfer 84.2%
- SF3: Vanilla 17.5% → ReactiveGWM_base 77.2% → ReactiveGWM_transfer 61.4%

Per-class breakdown 揭示了 **Control 类是 transfer 最难的**：SF3 transfer 的 Control 准确率只有 16%，但 Offense 是 100%。作者的解释是 Control 依赖 game-specific ranged attacks（Sonic Boom 在 SF2 vs airborne projectile 在 SF3），动画、timing、trajectory、spatial effect 都不同，所以 transfer 不如 general 的 Offense/Defense 直接。

**Player Action Following 的人类评估**：所有模型在 SF2/SF3 上得分都在 4.32-4.60 之间（5 分制），SEM 内无显著差异。这进一步支持了 architecture 的 action module 不干扰 action controllability。

---

## 6. 与相关工作脉络的关系

### 6.1 World Models 谱系

ReactiveGWM 站在两个脉络的交叉点上：

**World models for RL 谱系**：Ha & Schmidhuber 2018 [https://arxiv.org/abs/1803.10122] → Dreamer [https://arxiv.org/abs/1912.01603] → DreamerV2 [https://arxiv.org/abs/2010.02193] → DreamerV3 [https://arxiv.org/abs/2301.04104] → Genie 2 [https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/] → Genie 3。这些工作主要用 world model 做 policy learning，agent 在 latent imagination 里 rollout。

**Game world models 谱系**：DIAMOND [https://arxiv.org/abs/2412.09923] → GameNGen [https://arxiv.org/abs/2408.14837] → Oasis [https://oasis-model.github.io/] → Matrix-Game → LingBot-World → GameFactory [https://arxiv.org/abs/2504.01343]。这些工作主要用 diffusion 做 real-time game rendering。

ReactiveGWM 把这俩脉络 merge 了一下：用 diffusion 做 game rendering（第二个谱系的技术），但 explicitly model NPC 作为 autonomous agent（第一个谱系的精神）。它实际上是回到了 Ha & Schmidhuber 原始 world model 的精神——model 环境里的所有 agent，而非只 model 主角。

### 6.2 Controllable video generation 谱系

ReactiveGWM 的技术基础在 controllable video generation：
- Camera control: CameraCtrl [https://arxiv.org/abs/2404.02101], MotionCtrl [https://arxiv.org/abs/2312.07609]
- Motion control: DragAnything [https://arxiv.org/abs/2402.06146], DragNUWA [https://arxiv.org/abs/2308.08089], TORA [https://arxiv.org/abs/2410.27619]
- Trajectory/structure control: VideoComposer [https://arxiv.org/abs/2306.02060], Control-A-Video [https://arxiv.org/abs/2305.13840]

但所有这些 control signal 都是**物理/几何层面**的（trajectory, camera, skeleton）。ReactiveGWM 引入的是 **semantic-level control**——strategy 是一个抽象概念，不是低层 motion parameter。这是 control signal abstraction 的一次跃迁。

### 6.3 Modular transfer 谱系

Cross-attention 替换的 transfer 思路让我想到几个相关 work：
- **ControlNet** [https://arxiv.org/abs/2302.05543]：把 condition 通过 zero-conv 加到 frozen backbone，类似 modular 思路，但 ControlNet 是 train-from-scratch，不是 transfer。
- **IP-Adapter** [https://arxiv.org/abs/2308.06721]：decoupled cross-attention 让 image prompt 可以 plug-in，类似 modular 设计。
- **LoRA** [https://arxiv.org/abs/2106.09685]：低秩 perturbation，类似低带宽 signal injection。
- **AnimateAnyone** [https://arxiv.org/abs/2312.03853] / **AnimateAnyone 2** [https://arxiv.org/abs/2503.14340]：把 pose 控制和 identity 解耦，也是 modular 思路。

ReactiveGWM 的独特之处是**跨游戏 transfer**，而非单模型内的 module composition。它的 transfer 是 "strategy module 跨域 plug-in"。

---

## 7. 一些个人联想和疑问

### 7.1 关于 Action Module 的设计选择

paper 用 bias-free linear projection 而非 cross-attention 注入 action，理由是"lightweight"。但实际上，action 是 time-varying 的（每帧不同），broadcast 到空间维度意味着同一帧的所有 patch 都接受**相同**的 action bias。这个假设对 fighting game 合理（角色是 single entity，action 是 global），但对 multi-entity scene（FPS 里的多 NPC）就不一定了——每个 NPC 应该有独立的 action。

**可能的方向**：把 action 表示成 per-entity 的，用 cross-attention 让每个 entity token attends to 自己的 action embedding。或者更激进——用 pointer-style injection，action embedding 通过 attention 路由到 spatial location。

### 7.2 关于 Strategy 的三个 category 是不是太粗

Offense/Control/Defense 是 mutually exclusive 的三分类，但真实 fighting game 的 strategy 谱系远比这复杂（zoning, rushdown, turtle, mix-up, frame trap, okizeme...）。三分类的好处是清晰可学，坏处是表达力有限。

paper 在 prompt 里加了 Active 和 Passive 行为描述来补充 strategy 信号，这相当于把 strategy 拆成 (high-level category, low-level action composition) 的 hierarchical 结构。这种 hierarchical decomposition 在 RL 里也有先例（option framework [https://arxiv.org/abs/1604.06057]）。

**可能的方向**：把 strategy 表达成 continuous embedding 而非 3-way category，让模型自己 learn strategy manifold。或者用 LLM 生成 free-form strategy description，cross-attention 学更丰富的 alignment。

### 7.3 跟你的 Tesla 工作的潜在呼应

你之前在 Tesla 做 autonomous driving 时，world model 也是核心议题。ReactiveGWM 的 factorization——"agent intent (high-level) + physics dynamics (low-level) 解耦"——跟 autonomous driving 里的 "agent prediction (intent) + scene dynamics" 解耦有精神上的相似。其他 vehicle 是 reactive agents，跟 NPC 类似，需要预测它们的行为。

如果用 ReactiveGWM 的思路做 driving sim：
- Ego vehicle action 通过 additive bias 注入（低带宽、vehicle-specific）；
- Other vehicles 的 intent（lane change, merge, brake）通过 cross-attention 注入（高带宽、generic semantic）；
- Scene physics（road, traffic light, weather）由 SA + FFN 学。

这种 factorization 让 intent prediction module 可以跨场景 transfer（highway → urban → parking lot），而 scene dynamics 需要每个场景单独训。这跟你的 world model for autonomous driving 直觉应该是一致的。

### 7.4 Diffusion backbone 的 latency 问题

paper 在 limitations 里提到 diffusion-based backbone 引入 high inference latency，无法 real-time interactive。这个限制对 game 来说是 dealbreaker——真正的 game engine 需要至少 30 fps。

可能的解决方向：
- **Autoregressive generation**：DiT 改成 AR，每帧 inference 只需一次 forward（参考 GameNGen 的 real-time 部分）。
- **Distillation**：用 consistency model [https://arxiv.org/abs/2303.01469] 或 LCM [https://arxiv.org/abs/2310.15105] 把 30-step diffusion 蒸馏到 1-4 step。
- **Hybrid**：diffusion 训练 + AR 推理（参考一些最近 work）。

### 7.5 评估的潜在漏洞

paper 的 NPC Strategy Following 评估依赖 VLM referee。VLM 本身可能有自己的 bias（比如更容易把"接近对方"判为 Offense）。两个 VLM 的 ensemble mitigate 了一些，但还是个潜在 confound。

更严谨的评估可能是：
1. 用 RL agent 做 opponent，看 NPC strategy 是否能 win 对应的 strategy matchup；
2. 用 game engine 的 ground-truth state（如果 emulator 暴露 RAM）做 strategy label 而非 VLM 判断。

但 stable-retro 应该能 expose RAM，paper 没用这个有点可惜。

### 7.6 跟 GameNGen 的对比

GameNGen [https://arxiv.org/abs/2408.14837] 用两阶段（RL agent 收集 data + diffusion 模型学 dynamics）做 DOOM。它的 conditioning 也是 player action only。ReactiveGWM 跟它的核心区别在于 **NPC 的 explicit modeling**——GameNGen 里的 demon 是 scripted AI，被 model 成 deterministic dynamics；ReactiveGWM 里的 NPC 是有 strategy 的 autonomous agent，被 model 成 controllable policy。

如果用 ReactiveGWM 的思路重做 GameNGen：demon 的 strategy（aggressive rush, ambush, retreat）通过 cross-attention 注入，demon 的具体动作由 SA + FFN 学。这样 demon 可以有不同"性格"或"难度等级"，比 GameNGen 的固定 demon AI 更 flexible。

### 7.7 关于 Zero-shot Transfer 的边界条件

paper 的 transfer 是 SF2 → SF3（同 series、同 genre、同 developer Capcom）。这两个游戏的 visual style、character design、move set 都很相似。所以 strategy module 的 transfer 严格说是在**"近邻游戏"**之间的 transfer，不是"任意游戏之间"。

真正的 game-agnostic test 应该是：
- SF (2D fighter) → Tekken (3D fighter)
- SF (fighter) → Mario (platformer)
- SF (fighter) → StarCraft (RTS)

不过 paper 的野心是合理的——先在同 genre 内证明 transfer work，再 push 到跨 genre。跨 genre 的 strategy 概念可能完全不同（RTS 的 "macro/micro" 跟 fighter 的 "offense/defense" 没法对应），可能需要更 abstract 的 strategy representation。

---

## 8. 总结

ReactiveGWM 的贡献可以总结为三层：

1. **Conceptual**：把 NPC 从 background pixel 升级为 autonomous agent，通过显式的 strategy conditioning channel 实现 reactive game world model。
2. **Technical**：action（low-bandwidth additive bias）和 strategy（high-bandwidth cross-attention）的 asymmetric injection，让两者解耦且互不干扰。
3. **Practical**：cross-attention module 的 game-agnostic transferability，让 strategy module 可以 plug-and-play 进不同 game 的 vanilla model，避免 dense annotation cost。

最让我印象深刻的是 **4.4 节的 transfer 分析**——通过 $\rho^{\text{cross}}$ 的能量测量和 $\cos(\mathbf{CA}^V, \mathbf{CA}^T)$ 的方向测量，定量证明了 cross-attention 是 low-bandwidth 但 high-directional-impact 的 channel。这种 mechanistic analysis 在 game world model 文献里很少见，给 transfer 现象提供了 intuition。

**Web References 汇总**：
- ReactiveGWM 主页：https://inv-wzq.github.io/ReactiveGWM/
- Wan2.2：https://arxiv.org/abs/2503.20314
- GameNGen：https://arxiv.org/abs/2408.14837
- DIAMOND：https://arxiv.org/abs/2412.09923
- World Models (Ha & Schmidhuber)：https://arxiv.org/abs/1803.10122
- DreamerV3：https://arxiv.org/abs/2301.04104
- Genie 2：https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Oasis：https://oasis-model.github.io/
- Matrix-Game：https://matrix-game.github.io/
- DiT：https://arxiv.org/abs/2212.09748
- SAM2：https://arxiv.org/abs/2408.00714
- Grounding DINO：https://arxiv.org/abs/2303.05499
- ControlNet：https://arxiv.org/abs/2302.05543
- LoRA：https://arxiv.org/abs/2106.09685
- AnimateAnyone：https://arxiv.org/abs/2312.03853
- DragAnything：https://arxiv.org/abs/2402.06146
- LPIPS：https://arxiv.org/abs/1801.03924
- stable-retro：https://github.com/Farama-Foundation/stable-retro
- TCN：https://arxiv.org/abs/1803.01271
- Consistency Models：https://arxiv.org/abs/2303.01469
- Latent Consistency Models：https://arxiv.org/abs/2310.15105
- Options Framework (RL)：https://arxiv.org/abs/1604.06057
- GameFactory：https://arxiv.org/abs/2504.01343

如果你对其中某个细节特别感兴趣——比如 additive bias vs cross-attention 的选择理由、transfer 的能量分析、或者扩展到 non-fighting game 的思路——我们可以再 drill down。
