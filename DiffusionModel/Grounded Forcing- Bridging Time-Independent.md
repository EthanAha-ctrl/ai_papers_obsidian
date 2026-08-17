---
source_pdf: Grounded Forcing- Bridging Time-Independent.pdf
paper_sha256: 50c14d2b24bd57ad69e2c80453d401eeadf3efe4e0d60179ffe76508afd1f14e
processed_at: '2026-08-04T22:25:13-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我们抛开那些复杂的学术包装，用大白话把这篇 paper 的核心直觉拆解一下。

Autoregressive video generation 就像一个只看后视镜开车的司机。模型每次生成新的一帧 $x_t$，只能依靠过去的历史帧 $\mathbf{X}_{<t}$ 来做决定。如果这个视频只有几秒钟，司机记忆力很好，一切都很完美。但如果你想生成一个 1 分钟的长视频，或者实时互动的视频流，这个司机就会遇到三个要命的问题。

这篇 paper 叫 Grounded Forcing，核心目的就是给这个司机装上三个巧妙的系统，让他能无限开下去，还能随时听乘客的指令换路线。

---

### 1. 司机的第一个毛病：金鱼记忆

**Human Speak (大白话):** 
在 streaming 模式下，显存有限，KV Cache 只能装得下最近的一小段路（比如最近 6 帧）。随着时间推移，视频开头那个人长啥样、穿什么衣服，早就被新的 KV 挤出去了。模型画着画着，主角的脸就变了，背景也飘了。以前的方法（比如 LongLive）怎么解决呢？把第一帧死死钉在 Cache 里，永远不丢。但这有个傻地方：如果你中途用 prompt 换了个场景（比如从普通人变成 Hulk），模型还死盯着第一帧的那个普通人，导致新出现的 Hulk 永远画不对，出现了语义污染。

**The Fix (Dual Memory KV Cache):**
人脑有两种记忆：短期的工作记忆（记住刚才几秒钟发生了什么）和长期的核心记忆（记住我是谁、我在哪）。
Paper 设计了双缓存：
- **LTM (Local Temporal Memory)**: 只管最近几帧的微小动作变化，过时就扔。
- **GCM (Global Consistency Memory)**: 钉死几个最关键的语义锚点。但它会变！如果你引入了新角色 Hulk，GCM 会通过相似度计算，发现 Hulk 跟现有的锚点都不像（Importance 高），就把 GCM 里最没用、最重复的一个老锚点踢掉，换成 Hulk。这样，不管视频演化出多少个新角色，GCM 里永远保留着当前最核心的几个人物/背景特征，模型永远不会忘了“现在的”主角是谁。

**Math Intuition (数学直觉):**
$$\mathcal{Z}(z_t) = 1 - \max_{z \in \mathcal{M}_G} \text{CosSim}(z_t, z)$$
这里 $z_t$ 是当前帧特征。算它跟 GCM 里所有 anchor 的相似度，取最大值 $\mathcal{S}_{\text{max}}$。如果 $z_t$ 跟谁都不像（$\mathcal{S}_{\text{max}}$ 接近 0），那么 $\mathcal{Z}$ 就接近 1，说明这帧包含了“新信息”，必须存进 GCM。
$$\mathcal{R}(z_i) = \max_{z_j \in \mathcal{M}_G, j \neq i} \text{CosSim}(z_i, z_j)$$
这里用 leave-one-out 算 GCM 里哪个老 anchor 最多余。如果 $\mathcal{Z}(z_t) > \mathcal{R}(z_{\text{target}})$，就把最多余的 target 踢掉，换成新来的 $z_t$。这完全是对 semantic space 的一种 self-maintained coverage。

---

### 2. 司机的第二个毛病：时间戳爆表导致眩晕

**Human Speak (大白话):**
Transformer 用 RoPE 来感知时间。RoPE 本质上是给每一帧贴一个时间标签，比如 $t=1, 2, ..., 100$。模型在训练时，只见过 $t \in [0, 81]$ 的时间标签。如果在 inference 的时候视频太长，时间标签变成了 $t=500$，模型彻底懵了，它没见过 500 这个数字的频率旋转，Attention 算出来全是噪声，画面就开始抖动、崩溃。

**The Fix (Dual-Reference RoPE Injection):**
这里有个极其优雅的 trick：**在存 KV Cache 的时候，不要把 RoPE 算好存进去，存原始的 raw key！** 等到真正要算 Attention 的时候，再临时贴标签。
- **给 GCM 贴标签:** 所有的全局语义锚点，永远给它们贴上 $t=0$ 的标签。
  $$\mathbf{K}_{\text{GCM}} = \text{RoPE}(\mathbf{K}_{\text{raw}}, t=0)$$
  这意味着全局身份记忆是 time-invariant 的，它们存在于时间之外的维度。模型看向 GCM 时，感觉就像在看“永恒的真理”，不管视频生成到了第几千帧，GCM 的位置编码永远在训练流形内（$t=0$ 是训练时见过的）。
- **给 LTM 贴标签:** 滑动窗口里的局部帧，永远给它们贴上 $t \in [0, 21]$ 的相对标签。旧的被挤出去，新来的补在末尾，但标签永远在这个安全区间内。
  $$\mathbf{K}_{\text{LTM}} = \text{RoPE}(\mathbf{K}_{\text{raw}}, t_{\text{local}} \in [0, 21])$$

这样一来，不管生成多长的视频，RoPE 的时间标签永远在 $[0, 21]$ 这个训练见过的安全区里打转，彻底消除了 OOD (Out-of-Distribution) 导致的 visual drift。

---

### 3. 司机的第三个毛病：听见新指令就猛打方向盘翻车

**Human Speak (大白话):**
互动视频里，用户随时会改 prompt。比如第一句是 "A man walks"，第二句是 "Then he runs"。
以前的 Cache 刷新方法太粗暴，直接把旧 prompt 算出来的 KV 全部用新 prompt 的 KV 替换掉。这就好比司机一听到“跑”，脑子里关于“这个人长什么样”的记忆瞬间清空了，导致画出来的人连脸都变了。代词 "he" 所依赖的上下文被切断了。

**The Fix (Asymmetric Proximity Recache - APR):**
APR 是一个平滑的过渡机制。听到新 prompt 时，越靠近当前时刻的帧，受新 prompt 影响越大；越靠前的历史帧，越保留旧 prompt 的特征。
$$\mathbf{K}'_t = (1 - \alpha_t) \mathbf{K}^{\text{old}}_t + \alpha_t \mathbf{K}^{\text{new}}_t$$
这里 $\alpha_t$ 是一个按距离衰减的权重。距离当前越近（$d_t$ 越小），$\alpha_t$ 越大（最多到 0.8），越倾向于用新 KV。距离越远，$\alpha_t$ 趋近于 0，保持原样。
$$\alpha_t = \min\left(\alpha_{\text{max}}, 1 - \frac{d_t}{D_{\text{window}}}\right)$$
这个设计就像是给 Cache 做了一个 temporal low-pass filter。近期帧快速响应“动作”指令，远期帧死死守住“身份”信息。指令切换时，车头平稳转过去，完全不会翻。

---

### 拓展联想

**1. 与 StreamingLLM 的对比与进化**
LLM 里搞 infinite context 的鼻祖是 [StreamingLLM](https://arxiv.org/abs/2309.17453)，它发现把最初的几个 token 作为 Attention Sink 留着，模型就能无限生成。视频生成领域的 LongLive 照搬了这个思路，把第一帧当 Sink。但这在 video 里有个 bug：文本的初始 token 往往是语法结构，它不会变；但视频的第一帧只是个初始状态，如果后续语义大幅演化，死盯第一帧就会产生极强的 contamination。Grounded Forcing 的 GCM 其实是一个 **Dynamic Attention Sink**，它通过 similarity 驱动的 update，让 Sink 跟着语义演化走。这是对 StreamingLLM 思想在多模态下的一个重要修正。

**2. 与 Model-Based RL 的 State Abstraction 类比**
如果用 RL 的视角看，Autoregressive video generation 就是在学习一个 transition model $P(x_t | x_{<t})$。
Semantic forgetting 类似 state representation collapse，Visual drift 类似 representation OOD，Controllability loss 类似 policy-conditioning failure。
GCM 相当于把 latent state 拆分成了 stable part（环境、身份）和 transient part（当前动作）。这跟 RL 里的 world model（比如 [Dreamer](https://arxiv.org/abs/1912.01603) 系列）分离 slow features 和 fast dynamics 的思路如出一辙。

**3. Neural Turing Machine 的轮回**
GCM 的更新机制 $\mathcal{Z}(z_t) > \mathcal{R}(z_{\text{target}})$ 极其神似 [Neural Turing Machine](https://arxiv.org/abs/1410.5401) 的 erase/add 操作。本质上是根据 content-based addressing 来决定 memory 的写入。只是这里没有 trainable 的 controller，完全用确定的 geometric proximity 和 cosine similarity 规则来代替。这种 explicit memory mechanism 在极度长程的视频生成中可能比纯隐式的 hidden state 要 robust 得多。

**4. 未来可能的发展：Learnable Decay**
APR 里的 linear decay $\alpha_t$ 是手工设定的。如果能搞一个 meta-controller，根据当前 prompt 的语义距离动态调整 decay 速率，效果可能更好。比如从 "man" 换到 "cartoon man" 可以激进一点，从 "man walks" 换到 "man runs" 应该更平滑。这种 learnable cache control 在 LLM 里已经有相关探索（比如 [MemGPT](https://arxiv.org/abs/2310.08560) 的 OS 级内存管理），视频生成完全可以借鉴。

总结一下，这篇 paper 的核心在于：把视频生成中的 time-invariant semantics 和 time-variant dynamics 彻底解耦。用动态的 GCM 锚定身份，用相对的 RoPE 稳定位置，用渐变的 APR 平滑指令。这套组合拳让 autoregressive video 模型从只能做短剧，向能做无限流的长篇互动电影迈进了一大步。

---

# Grounded Forcing 深度技术解析

这篇 paper 由 Peking University 和 Alibaba 团队合作，提出了一个针对 autoregressive video synthesis 的综合框架，系统性地解决了长期困扰该领域的三大瓶颈。接下来我会从 motivation、数学形式化、三个核心组件的细节，以及与相关工作的联系角度进行深入讲解。

---

## 1. 核心问题的本质理解

Autoregressive video generation 在 streaming 场景下，本质上是把一个生成过程拆解为：
$$P(\mathbf{X}) = \prod_{t=1}^{T} P(x_t | \mathbf{X}_{<t}, c)$$

其中 $x_t$ 是第 $t$ 帧的 latent，$c$ 是 text condition，$T$ 是序列长度。这个公式看起来简洁，但在 $T \to \infty$ 时会出现三个相互纠缠的退化：

**第一个问题：Semantic Forgetting**
当 KV cache 使用 sliding window（只保留最近 $W$ 帧），随着 $t \gg W$，初始帧 $x_{1..k}$（包含 subject identity）从 attention 中被完全剔除：
$$\lim_{t \to \infty} P(x_t | \mathbf{X}_{<t}) \approx P(x_t | \mathbf{X}_{t-W:t})$$

这是一个信息论上的硬约束——identity 信息（如人物外观、场景布局）在生成初期被编码，但随着时间窗口滑动被丢弃。这个问题在 LLM 的 StreamingLLM 工作（[Xiao et al., 2023](https://arxiv.org/abs/2309.17453)）里通过 attention sink 部分缓解，但视频生成的多模态语义远比 token distribution 复杂。

**第二个问题：Visual Drift**
Standard RoPE 把每个 token 绑定到绝对 temporal index。在 training 时模型只见过 $t \in [0, T_{\text{train}}]$（Wan 模型约为 81 latent frames），而 streaming 时 $t$ 可以增长到几千。当 $t > T_{\text{train}}$，rotational frequencies $e^{im\theta}$ 进入 OOD (out-of-distribution) regime，attention pattern 退化。这是 distribution shift 在 positional encoding 上的体现。

**第三个问题：Controllability Loss**
当 prompt 切换时（如 "A man walks" → "Then he runs"），现有方法（LongLive 的 KV-recache、Infinity-RoPE 的 KV Flush）采用 uniform update：
$$\mathbf{K}' = (1-\alpha)\mathbf{K}^{\text{old}} + \alpha \mathbf{K}^{\text{new}}, \quad \forall t$$

其中 $\alpha$ 是常数。这忽略了时间近邻性——recent frames 需要快速响应新指令，distant frames 需要保留旧 identity。这种 symmetric refresh 要么导致 "semantic shock"（identity 突变），要么导致 "instruction ignoring"（指令未被采纳）。

---

## 2. Dual Memory KV Cache：解耦短期动态与长期语义

### 2.1 架构哲学

这个设计让我联想到 neuroscience 中的 dual memory system： hippocampus 处理 episodic memory（短期、细节），neocortex 处理 semantic memory（长期、抽象）。paper 把 KV cache 分为两个互补模块：

- **Local Temporal Memory (LTM)**: 滑动窗口（size=6 frames），存储 recent frames 的高频 motion 信息，受 sliding eviction 管控。
- **Global Consistency Memory (GCM)**: 固定 size=3 的 keyframe set，存储 semantic anchor（subject identity、background style），不受 eviction 影响。

### 2.2 Diversity-Aware Global Update 详解

这是 paper 的一个关键创新。GCM 不固定为初始帧，会动态更新以反映合法的 semantic shift（如 prompt 切换后加入新角色）。算法分为三步：

**Step 1: 计算 new frame $z_t$ 的重要性**
$$\mathcal{S}_{\text{max}}(z_t) = \max_{z \in \mathcal{M}_G} \text{CosSim}(z_t, z), \quad \mathcal{Z}(z_t) = 1 - \mathcal{S}_{\text{max}}(z_t)$$

变量解释：
- $z_t$：第 $t$ 步生成的 frame 的 latent representation
- $\mathcal{M}_G = \{z_1, ..., z_K\}$：当前 GCM 中的 $K$ 个 anchor latents（这里 $K=3$）
- $\mathcal{S}_{\text{max}}(z_t)$：$z_t$ 与 GCM 中所有 anchor 的最大余弦相似度
- $\mathcal{Z}(z_t)$：importance score，等于 $1 - \mathcal{S}_{\text{max}}$

intuition：如果 $z_t$ 与所有 anchor 都不相似（$\mathcal{S}_{\text{max}}$ 低），那么 $\mathcal{Z}$ 高，代表 $z_t$ 携带了 novel semantic information，值得加入 GCM。

**Step 2: 计算 existing anchor 的冗余度**
$$\mathcal{R}(z_i) = \max_{z_j \in \mathcal{M}_G, j \neq i} \text{CosSim}(z_i, z_j)$$

变量解释：
- $\mathcal{R}(z_i)$：第 $i$ 个 anchor 与其余 anchor 的最大相似度（leave-one-out）
- 高 $\mathcal{R}$ 表示 $z_i$ 与其他 anchor 语义重复，可被替代

注意 leave-one-out 设计避免了 self-similarity bias（如果 $z_i$ 与自己比较，相似度恒为 1）。

**Step 3: 替换决策**
$$\text{if } \mathcal{Z}(z_t) > \mathcal{R}(z_{\text{target}}), \text{then } \mathcal{M}_G \leftarrow (\mathcal{M}_G \setminus \{z_{\text{target}}\})) \cup \{z_t\}$$

其中 $z_{\text{target}} = \arg\max_{z \in \mathcal{M}_G} \mathcal{R}(z)$，即最冗余的 anchor。

这个机制保证 GCM 始终覆盖 semantic space 中的 diverse anchors，在 Figure 3(c) 的例子里，当 Hulk 出现后，Silver Wyvern 加入，GCM 同时保留两者作为 anchor，避免单一身份的固着。

---

## 3. Dual-Reference RoPE Injection (DR-RoPE)

### 3.1 标准 RoPE 的回顾

RoPE (Rotary Position Embedding, [Su et al., 2024](https://arxiv.org/abs/2104.09864)) 对 query 和 key 做 rotation：
$$\tilde{K}_t = \text{RoPE}(K_t, t), \quad \tilde{Q}_t = \text{RoPE}(Q_t, t)$$

其中 $t$ 是 absolute temporal index。RoPE 的核心性质是 query 和 key 之间的 relative position 通过 rotation 角度差编码：
$$\tilde{Q}_t \cdot \tilde{K}_s = f(Q_t, K_s, t-s)$$

这个性质使得 attention score 依赖于相对位置 $t-s$，但 absolute index $t$ 仍会影响 rotation magnitude。在 streaming 中 $t \to \infty$ 时，绝对值超出训练分布。

### 3.2 关键 trick：存储 pre-RoPE 的 raw keys

paper 的核心洞察是：**不要在 cache 时就应用 RoPE**。存储 $\mathbf{K}_{\text{raw}}$（pre-RoPE），在 attention 计算时动态注入 RoPE，允许对 GCM 和 LTM 赋予不同的 temporal reference frame。

**GCM: Time-Invariant Anchoring**
$$\mathbf{K}_{\text{GCM}} = \text{RoPE}(\mathbf{K}_{\text{raw}}, t_{\text{global}} = 0)$$

强制所有 GCM keys 的 temporal index = 0。这意味着：
- GCM keys 处于"始终是当前时刻"的位置
- Attention 把 GCM 视为 timeless semantic reference
- 当 t → ∞，GCM keys 仍然在训练 manifold 内（index 0 是 training 时见过的）

这个 trick 与 [RIFLex](https://arxiv.org/abs/2502.15894) 的 frequency rescaling 思路不同——RIFLex 调整 RoPE 的 frequency basis 来外推，而 Grounded Forcing 直接固定 GCM 的位置，使其"飞出"时间维度。

**LTM: Relative Contextualization**
$$\mathbf{K}_{\text{LTM}} = \text{RoPE}(\mathbf{K}_{\text{raw}}, t_{\text{local}}), \quad t_{\text{local}} \in [0, 21]$$

变量解释：
- $t_{\text{local}}$：当前 sliding window 内的相对 index
- $[0, 21]$ 是 Wan 模型的训练分布范围

每当 sliding window 前移，新的 frame 被分配 $t_{\text{local}} = 21$，旧的 frame 被驱逐，剩下的 frame 的 $t_{\text{local}}$ 重新映射到 $[0, 21]$ 范围。这保证 LTM keys 始终在 high-confidence positional manifold 内。

### 3.3 Multi-Shot Generation via LTM Reset

当 scene 切换时（如电影剪辑），通过特定 trigger token 检测场景变化，flush LTM 同时保留 GCM。这保证了：
- 局部 scene 切换干净（无 temporal contamination）
- 全局 identity 保留（如同一角色出现在不同场景）

### 3.4 与 Infinity-RoPE 的对比

[Infinity-RoPE](https://arxiv.org/abs/2511.20649) 提出了 Block-Relativistic RoPE，也是把绝对位置改为相对位置，但是 inference-time only 的方法。Grounded Forcing 在此基础上加入了 dual-reference 思想——LTM 和 GCM 用不同的 reference frame，这是一个 architectural innovation 而非 pure inference adaptation。

---

## 4. Asymmetric Proximity Recache (APR)

### 4.1 问题动机

Prompt 切换场景如 "A man walks" → "Then he runs" 中，"he" 这个代词依赖前文。Uniform refresh 会破坏这种 linguistic dependency。

### 4.2 Proximity-Weighted Interpolation 公式详解

$$\mathbf{K}'_t = (1 - \alpha_t) \mathbf{K}^{\text{old}}_t + \alpha_t \mathbf{K}^{\text{new}}_t$$

变量解释：
- $\mathbf{K}^{\text{old}}_t$：旧 prompt 阶段 cache 的 keys
- $\mathbf{K}^{\text{new}}_t$：新 prompt 下重新计算的 keys
- $\alpha_t \in [0, \alpha_{\text{max}}]$：timestep-dependent weighting coefficient
- $\alpha_{\text{max}} = 0.8$：上限，保证 baseline 保留旧语义

**Linear decay schedule:**
$$\alpha_t = \min\left(\alpha_{\text{max}}, 1 - \frac{d_t}{D_{\text{window}}}\right)$$

变量解释：
- $d_t$：token $t$ 到当前生成 frontier 的 temporal distance
- $D_{\text{window}}$：recache window 大小
- 当 $d_t = 0$（最接近 frontier），$\alpha_t = \min(0.8, 1) = 0.8$
- 当 $d_t = D_{\text{window}}$（最远），$\alpha_t = \min(0.8, 0) = 0$，完全保留旧 cache

### 4.3 这个设计的 intuition

这个设计本质上是一个 temporal kernel——recent frames 接受新 prompt 的"冲击"，distant frames 维持历史 anchor。这与 RNN 中的 gated update（如 LSTM 的 forget gate）有异曲同工之妙，但是基于 explicit temporal distance 而非 learned gate。

如果把这个公式放在 attention 视角下，相当于在 KV cache 空间做了一个 temporal low-pass filter：高频变化（新 prompt 指令）通过 recent frames，低频 identity 通过 distant frames 保留。

---

## 5. 实验数据分析

### 5.1 240s Video Generation (Table 1)

| Model | Aesthetic Quality | Background Consistency | Dynamic Degree | Imaging Quality | Motion Smoothness | Subject Consistency | Temporal Flickering |
|-------|------|------|------|------|------|------|------|
| LongLive | 0.6032 | 0.9208 | 0.60 | 0.7245 | 0.9892 | 0.8973 | 0.9770 |
| Rolling Forcing | 0.6237 | 0.9144 | 0.53 | 0.7317 | 0.9878 | 0.9034 | 0.9845 |
| Infinity-RoPE | 0.6048 | 0.9195 | 0.58 | 0.7154 | 0.9873 | 0.9066 | 0.9808 |
| **Ours** | 0.6174 | **0.9265** | 0.60 | 0.7204 | 0.9878 | **0.9163** | 0.9812 |

关键观察：
- **Subject Consistency 0.9163**：相比 LongLive 提升 +1.9 个百分点，这正是 GCM 的功劳。在 240s 长视频里，GCM 持续 anchor identity，避免 drift。
- **Background Consistency 0.9265**：同样第一，证明 GCM 不仅保护 subject，也保护 scene context。
- **Dynamic Degree 0.60**：与 LongLive 并列最高，证明 motion dynamic 没有因为 anchor 增强而牺牲——LTM 仍然捕捉高频运动。

### 5.2 Ablation Study (Table 2)

| Model | Aesthetic Quality | Background Consistency | Subject Consistency | Temporal Flickering |
|-------|------|------|------|------|
| w/o Dual Mem | 0.6473 | 0.8467 | 0.7322 | 0.9734 |
| w/o DR-RoPE | 0.6278 | 0.8729 | 0.7606 | 0.9716 |
| w/o APR | 0.6386 | 0.8603 | 0.7719 | 0.9636 |
| **Ours (full)** | 0.6440 | 0.8795 | 0.7770 | 0.9723 |

观察：
- **w/o Dual Mem**: Subject Consistency 降到 0.7322（-4.48%），Background Consistency 降到 0.8467（-3.28%），证明 GCM 是 consistency 的核心。
- **w/o DR-RoPE**: Aesthetic Quality 降到 0.6278（-1.62%），证明 positional encoding 对视觉质量影响显著。Positional drift 会破坏 attention pattern 导致 artifacts。
- **w/o APR**: Temporal Flickering 0.9636（vs full 0.9723），说明 prompt 切换时的 visual jitter 主要由 APR 缓解。

### 5.3 User Study (Table 3)

| Metric | LongLive | Infinity-Rope | Ours |
|--------|----------|---------------|------|
| Subject Consistency | 3.02 | 2.98 | **3.66** |
| Background Consistency | 3.04 | 3.08 | **3.78** |
| Aesthetic Quality | 3.45 | 3.34 | 3.42 |
| Text Adherence | 3.12 | 3.15 | **3.72** |

Text Adherence 3.72 显著高于 baseline（提升 0.6+），这是 APR 的功劳——APR 让模型既响应新 prompt 又保留上下文。

---

## 6. 与相关工作的联系和联想

### 6.1 与 LLM Attention Sink 的关系

[StreamingLLM](https://arxiv.org/abs/2309.17453) 发现 LLM 中保留 initial tokens 作为 attention sink 可以让 context window 无限扩展。LongLive 和 Infinity-RoPE 把这个思想迁移到 video，用 first frame 作为 sink。但 Grounded Forcing 的 GCM 是 dynamic 的——通过 diversity-aware update 让 anchor 集合 evolve，这是对 static attention sink 的本质改进。

### 6.2 与 Self-Forcing 训练范式的关系

[Self-Forcing](https://arxiv.org/abs/2506.08009) 通过在训练时 conditioning on self-generated frames 来弥合 train-test gap。Grounded Forcing 在此基础上加入了 dual memory 的训练，stage 2 在 long-horizon sequence 上训练，让模型学会利用 GCM。

### 6.3 与 Rolling Forcing 的对比

[Rolling Forcing](https://arxiv.org/abs/2509.25161) 通过 rolling-window joint denoising 让连续 frame 互相 refinement。这是 denoising-level 的创新，而 Grounded Forcing 是 memory-level 的创新，两者是 complementary 的。

### 6.4 与 World Model 的视角

如果从 world model 的视角看，autoregressive video generation 本质上是在学习一个 dynamics model $P(x_t | x_{<t}, c)$。三个挑战对应：
- Semantic forgetting = state representation collapse
- Visual drift = representation OOD
- Controllability loss = policy-conditioning failure

Grounded Forcing 的 dual memory 类似于 model-based RL 中的 state abstraction——分离稳定的 latent state（GCM）和 transient observation（LTM）。

### 6.5 与 Compressive Memory 的联想

可以联想到 [Memory Networks](https://arxiv.org/abs/1410.3916) 和 [RETRO](https://arxiv.org/abs/2112.04426) 这类外部记忆增强方法。GCM 本质上是一个 sparse keyframe memory，通过 similarity-based retrieval 和 update。这种思想在 long-context LLM 中也有体现，如 [MemGPT](https://arxiv.org/abs/2310.08560) 的 hierarchical memory。

### 6.6 与 Neural Turing Machine 的联想

GCM 的 update 机制让我联想到 [Neural Turing Machine](https://arxiv.org/abs/1410.5401) 的 content-based addressing——根据 query 与 memory 的相似度决定读写位置。Diversity-aware update 类似于 NTM 的"least used" memory eviction，但是基于 semantic redundancy。

---

## 7. 局限性和可能的扩展

### 7.1 GCM size 固定为 3 的局限

paper 中 GCM size = 3 是 fixed hyperparameter。如果视频语义极其复杂（如 10 个角色），3 个 anchor 可能不足。一个可能的方向是 adaptive GCM size，根据 latent diversity 动态调整。

### 7.2 APR 的 linear decay 可能不是最优

linear decay $\alpha_t = \min(\alpha_{\max}, 1 - d_t/D_{\text{window}})$ 是手工设计的 schedule。可能更 sophisticated 的 decay function（如 exponential、learned schedule）效果更好。

### 7.3 与 bidirectional diffusion 的鸿沟

paper 强调 AR 框架的 streaming 优势，但 bidirectional diffusion（如 [Wan](https://arxiv.org/abs/2503.20314)、[HunyuanVideo](https://arxiv.org/abs/2412.03603)、Sora）在 fidelity 上仍然领先。一个 hybrid 方向是 bidirectional training + AR inference，类似 [CausVid](https://arxiv.org/abs/2504.20432) 的 distillation 思路。

### 7.4 与 test-time compute scaling 的结合

可以联想到 [Self-Forcing++](https://arxiv.org/abs/2510.02283) 在 test-time 通过 long rollout 来 scale compute。Grounded Forcing 的 GCM 可以与 test-time scaling 结合——通过额外的 semantic verification step 决定 GCM update。

---

## 8. 总结

Grounded Forcing 的核心贡献在于**系统性地解耦 time-independent semantics 和 proximal dynamics**：

| 挑战 | 解决方案 | 核心机制 |
|------|---------|---------|
| Semantic forgetting | Dual Memory KV Cache | GCM anchor + LTM sliding |
| Visual drift | DR-RoPE | GCM 用 $t=0$，LTM 用 $t \in [0, 21]$ |
| Controllability loss | APR | Proximity-weighted $\alpha_t$ interpolation |

三个组件形成一个 interlocking system——Dual Memory 提供 structure，DR-RoPE 保证 positional stability，APR 提供 interaction flexibility。这个框架的思想可以推广到其他 autoregressive 模态（如 long-horizon audio generation、interactive 3D scene generation）。

参考资源：
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Rolling Forcing: https://arxiv.org/abs/2509.25161
- LongLive: https://arxiv.org/abs/2509.22622
- Infinity-RoPE: https://arxiv.org/abs/2511.20649
- Wan: https://arxiv.org/abs/2503.20314
- CausVid: https://arxiv.org/abs/2504.20432
- StreamingLLM: https://arxiv.org/abs/2309.17453
- RoPE: https://arxiv.org/abs/2104.09864
- RIFLex: https://arxiv.org/abs/2502.15894
- VBench: https://vchitect.github.io/VBench-project/
- DMD: https://arxiv.org/abs/2311.18828
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- MovieGen: https://arxiv.org/abs/2410.13720
- MemGPT: https://arxiv.org/abs/2310.08560
- Neural Turing Machine: https://arxiv.org/abs/1410.5401
- Memory Networks: https://arxiv.org/abs/1410.3916
