---
source_pdf: Behavior Prompting Policy.pdf
paper_sha256: e457d48683ce022eee6ad8814367e580adeae9054752a89df6afd02a1f4a592a
processed_at: '2026-07-18T14:20:04-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Behavior Prompting Policy 深度解析

非常好的 paper，这个工作把 LLM 的 in-context learning paradigm 直接迁移到 robotics manipulation，思路非常自然但又锋利。让我系统性地讲透。

**Paper 链接**: behavior-prompting.github.io  
**Code**: github.com/austinpatel/behavior-prompting (推断)  
**相关**: arxiv.org/abs/2402.10329 (UMI), arxiv.org/abs/2408.15980 (ICRT), arxiv.org/abs/2410.24164 (π₀)

---

## 1. 核心思想与 motivation

LLM 时代的 in-context learning 已经成为 paradigm，prompt 一段 example 就能让 model 适应新 task。Robotics 这边一直是 fine-tuning 主导：BC-Z [arxiv.org/abs/2202.02071]、RT-2 [proceedings.mlr.press/v229/zitkovich23a.html]、OpenVLA [arxiv.org/abs/2406.09246]、π₀ [arxiv.org/abs/2410.24164]、π₀.₅ [arxiv.org/abs/2504.16054] 这些 VLA 都是 language/goal-image conditioning，但它们做的是 **semantic adaptation**（新 object、新 scene），对 **low-level action adaptation**（新的 motion pattern、新的 manipulation strategy）几乎无能为力。

BPP 的核心 insight：**一个完整的 demonstration 同时包含 "what to do" 和 "how to do it"**。Language 只告诉 "what"，goal image 只告诉 "终态"，二者在 temporal/spatial 维度严重 underspecified。Drawing 这种 task 上尤其明显——goal image 只能告诉你最终图长什么样，不能告诉笔触顺序、起笔位置、中间 trajectory。

---

## 2. Behavior Prompt 的定义

**Behavior prompt** = 一个完整的 demonstration，包含三模态时间序列：

$$\mathcal{P} = \{(o_t, q_t, a_t)\}_{t=0}^{T}$$

其中：
- $o_t$: observation（图像，通常 wrist + third-person camera）at timestep $t$
- $q_t$: proprioception（end-effector pose、gripper width 等本体感受）
- $a_t$: action（end-effector 的 6-DoF target 或 joint velocity）
- $T$: prompt 总长度（变量，不固定）

关键点：prompt 在 robot 的 **同一 sensorimotor space** 里，意味着 prompt 的 obs 直接和 deployment 的 obs 同分布，policy 不需要做 modality translation（区别于 Vid2Robot [arxiv.org/abs/2310.01936] 那种 human video → robot action 的 cross-embodiment 设定）。

---

## 3. Behavior Prompting Policy (BPP) 架构

参考 Figure 2。整个 policy 分两个模块：**Prompt Encoder** 和 **Action Decoder**。这种解耦设计是和 ICRT 最大的架构差异（详见 Section G 的对比）。

### 3.1 Prompt Encoder

#### Chunk 构造

为减少 sequence length，prompt 被切成 chunks。每隔 $\Delta t$ 个 step 取一个 observation/proprio，但 actions 全部保留：

$$\text{chunk}_i = \{o_{i\cdot\Delta t},\ q_{i\cdot\Delta t},\ \{a_t\}_{t=(i-1)\cdot\Delta t}^{i\cdot\Delta t}\}$$

$\Delta t$ 通常设为 60，对应 1Hz 的 observation 频率（如果原始频率是 60Hz）。注意 action 不下采样是为了保留完整的低层行为细节，这点在 paper 的 ablation (Fig. 6b) 里被验证：aggressive downsampling < 1Hz 会显著降低 unseen drawing 的成功率。

#### Attention pooling

每个 chunk 内三种 modality 通过 attention pooling 合并成单个 embedding：

$$p_i = \text{AttnPool}(o_{i\cdot\Delta t},\ q_{i\cdot\Delta t},\ \{a_t\}_{t})$$

直觉：让 model 学到如何把同一 timestep 的视觉信息、本体感受和 action 序列 temporally align 到一起。这比把三种 modality 分开 tokenize 的好处：
1. **Sequence length 减小** $n$ 倍（$n$ = chunk 数），后续 transformer 计算量降低
2. **时间对齐**：避免了 cross-attention 时 obs 和 action 之间错位

Ablation Fig. 6c 验证 attention pooling 比 no-pooling 表现好。

最终 prompt 表示为 chunk embedding 序列：

$$P = [p_0, p_1, \ldots, p_n]$$

其中 $n$ 随 prompt 长度变化（variable-length prompt，类似 LLM 的 variable context）。

#### Transformer decoder + cross-attention

给定 current observation 历史 $\{o_{\text{cur}, -H}, \ldots, o_{\text{cur}, 0}\}$（$H$ 是 history length，每个 timestep 一个 token），policy 通过 **cross-attention** 从 $P$ 中提取相关信息：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$ 来自 current observation tokens（query）
- $K, V$ 来自 prompt chunk embeddings $P$（key/value）
- $d_k$ 是 key 的维度（paper 用 768）

这部分架构是 6-layer transformer decoder，hidden size 768，8 heads，57M params（Table 4）。

**关键 intuition**: cross-attention 实际上在做 "prompt lookup"——根据当前 obs 在 prompt 里检索最相关的 chunk。Fig. 5a 的 attention 可视化非常漂亮：在 DrawAnything 上，attention 沿对角线分布，说明 model 持续跟踪 "prompt 中和当前状态最接近的那一段"。这其实就是一个 learned retrieval mechanism。

在 LIBERO-Gen 上（Fig. 5b, d），attention 更离散，跟踪 "milestones"（比如下一个 object to interact、下一个 place location），因为 LIBERO 的任务是离散的 pick-place 而不是连续 drawing。

### 3.2 Action Decoder

用 Diffusion Policy [arxiv.org/abs/2303.04137] 的 CNN U-Net 架构 + FiLM [arxiv.org/abs/1709.07871] conditioning。

**Diffusion 公式回顾**：定义 forward process 加噪：

$$q(x_t | x_0) = \mathcal{N}(x_t;\ \sqrt{\bar{\alpha}_t}x_0,\ (1-\bar{\alpha}_t)\mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$ 是 cumulative noise schedule。逆向去噪：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t))$$

在 action diffusion 里 $x_0$ 是 ground-truth action chunk $a_{0:H_a}$（长度 $H_a$ 的未来 action sequence），训练时预测 noise $\epsilon$：

$$\mathcal{L} = \mathbb{E}_{t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]$$

其中 $c$ 是 conditioning，包含：
- current observation $o_{\text{cur}}$
- prompt encoder 输出的 relevant info（cross-attention output）
- diffusion step $k$（通过 FiLM embedding 注入）

架构参数（Table 5）：U-Net，down dims [256, 512, 1024]，kernel 5，151M params。

### 3.3 Training

每个 training step：
1. 采样一个 task，从该 task 的 demonstrations 中随机选一个作为 prompt $\mathcal{P}$
2. 从 **同 task 其他 demonstrations** 中采样一个 batch of (obs history, future action chunk) pairs
3. Loss：action diffusion loss

**关键设计**: 训练时 prompt 和 rollout 之间 **没有显式 spatial/temporal correspondence**。比如 prompt 是"在桌子左边 fold 衣服"，而 current obs 是"在桌子右边 fold 同一件衣服"，model 要自己学会对齐。这个 design choice 让 BPP 可以直接 train on existing multi-task BC datasets without extra annotation。

### 3.4 Inference

每个 rollout 选 **一个 prompt**，可以预计算并 cache prompt chunk embeddings $P$。每次 inference step：
1. Prompt encoder 跑一次，cross-attention 提取 relevant info
2. Action decoder 做 $K$ 步 diffusion denoising（paper 用 $K$ 步，标准 Diffusion Policy 设定）

这种 decoupling 让 inference 高效，不需要像 ICRT 那样每次 forward 都 reference 整个 prompt + 整个 history。

---

## 4. iPhUMI: Handheld Data Collection Interface

UMI [arxiv.org/abs/2402.10329] 的改进版本。Original UMI 用 GoPro，需要 SLAM mapping 整个 scene 才能定位 gripper pose。iPhUMI 把 GoPro 换成 iPhone 15 Pro，**利用 on-device ARKit 做 real-time SLAM**，bypass mapping step。

**Hardware**:
- 3D-printed mount 持有 iPhone
- 双指 gripper（标准 manipulation）或 marker + spring（compliant drawing，不需要 force-torque sensor）
- Bimanual 版本支持双 iPhone 共享 ARKit session（最多 3 个 device：双臂 + 头戴）

**Data modalities**（Fig. 15）:
- Main camera: 1920×1440 @ 60Hz
- Ultrawide camera: 640×480 @ 10Hz（提供 wider FoV，对 policy 性能有帮助 [1]）
- LiDAR depth: 256×192 @ 60Hz（paper 中未使用）
- Gripper pose: 60Hz（ARKit SLAM 输出 6-DoF pose）
- Gripper width: 10Hz（通过 ultrawide camera 看 ArUco tag 检测手指位置）

**Test-time prompting**: iPhone app 支持无线把 demonstration 传到部署 desktop，立刻 condition BPP。这是一个非常实用的 design——用户在厨房拿着 iPhUMI 演示一下怎么叠毛巾，robot 就能照做。

**对比**: 这种 setup 让数据收集成本极低。比如 paper 的 DrawAnything-Real 实验里，作者混合用了 200 个 iPhUMI 人类 demo + 800 个 scripted policy demo 来达到 1000 task 的 diversity。

---

## 5. Benchmarks

这是 paper 的另一个重要 contribution——提供了 reproducible 的 behavior prompting benchmark。

### 5.1 DrawAnything-Sim

- 2000 procedural drawing tasks，每个 5 demos @ random board rotations
- Drawing 由 line / Bezier curve / oval / free-space movement 组合，1-6 parts
- Eval: 50 unseen human-collected drawings，5 demos each @ varying rotations
- Metric: Chamfer distance (pixels)
- Action: 2D cursor control

### 5.2 DrawAnything-Real

- ARX robot arm + iPhone wrist camera + marker
- 6-DoF action（vs 2D in sim）
- 1000 training tasks（200 iPhUMI @ 5 demos + 800 scripted @ 6 demos）
- Eval: 10 tasks（4 training + 6 unseen）via iPhUMI human demos

### 5.3 LIBERO-Gen Combination

- 基于 LIBERO Spatial [arxiv.org/abs/2306.03310] 扩展
- 原始 10 tasks + 164 新 tasks = 174 total
- 每个环境有两个 identical bowls，task = (pick one bowl, place at one of 9 locations)
- **Eval**: hold out 10 (pick, place) combinations——pick 和 place location 都在 training 里单独见过，但 (pick, place) 组合没见过
- 评估 **instruction-following capability on unseen combinations**

### 5.4 LIBERO-Gen Chain

- 基于 LIBERO Goal 扩展
- 原始 10 tasks + 311 新 tasks = 321 total
- 每个任务是 **两个 single-step primitives 的链式组合**：first step ∈ {open middle/top drawer, push plate, turn on stove, pick-place}，second step 是 pick-place
- **Eval**: hold out 10 个 two-step chains，每个 chain 的两个 primitive 单独见过但 chain 没见过
- 关键 ablation：从 training 移除"second step" tasks（即只训 first-step 和 chained），看 model 能否 generalize 到新的 chained task。这是 **long-horizon adaptation** 的 stress test

---

## 6. 实验结果与 key findings

### 6.1 Behavior prompting works（Fig. 4）

**DrawAnything-Sim** (unseen drawings，Chamfer distance 越低越好):
- Goal-Image: baseline
- BPP: **80.7% error reduction vs Goal-Image**, **33.3% reduction vs ICRT**
- ICRT 为什么差？因为 ICRT 保留 entire rollout history in context，容易受 spurious correlation 影响 OOD [arxiv.org/abs/2410.06564]

**LIBERO-Gen Combination**:
- BPP 显著超过 Goal-Image 和 Language baselines
- 与 π₀.₅（用 LoRA finetune 100K steps，有 foundation pretraining）rivals——这点非常 impressive，因为 BPP **完全没有 pretraining**

**LIBERO-Gen Chain**:
- BPP 超 Language baseline 10.7%
- 在 ablation（移除 2nd step tasks from training）下，BPP 超 Language **20.8%**——说明 behavior prompt 的 dense sub-goal 信息在 long-horizon / 未见组合上更有价值

**Trend**: temporal task complexity 越高（pick-place → chained → dense drawing），behavior prompt 相对 goal-image 和 language 的优势越大。

### 6.2 Prompt encoder attention 行为（Fig. 5）

- **DrawAnything**: attention 沿对角线，即"prompt 中当前时刻最相似的 chunk"。这就是 dense sub-goal conditioning
- **LIBERO-Gen**: attention 更离散，跟踪 milestones（任务切换、下一个目标 object）

### 6.3 Prompt representation ablations（Fig. 6a-c）

- **Modalities**: observations 必须有（anchor prompt lookup）；actions 提供时间 transition；proprio 在 drawing 中无用（因为 cursor 已经在 obs 中显示）。但对 manipulation 任务，proprio 可能有更大作用
- **Downsampling frequency**: <1Hz 会严重退化
- **Attention pooling**: 比不 pooling 显著好

### 6.4 Training data composition ablations（Fig. 6d-f）

这是 paper 最 actionable 的 finding：

- **Task diversity > demos per task**: 固定 budget 下，more tasks + fewer demos/task >> fewer tasks + more demos/task
- **# train tasks scales**: 5 demos/task，随着 task 数量增加，unseen task 性能持续提升
- **Complex training tasks**: 只训简单 drawings（1-3 parts）→ 对复杂 unseen drawing 性能差；训复杂 drawings（4-6 parts）性能最好

**Intuition**: 这跟 LLM 的 in-context learning 一样——预训练 task 多样性 → model 学会"how to learn from context"，而不是 memorize specific tasks。

### 6.5 Laundry Folding case study（Appendix A）

3 个 sweater folding task（fold left arm / right arm / bottom up），bimanual iPhUMI 收集 ~150 demos/task。

结果（Table 1）:
- Language: 96-100% success
- BPP: 60-100% success，**偶尔做错 task**（比如让它 fold left arm，它 fold right arm）

**Important finding**: 在 **低 task diversity** 场景下，behavior prompt 反而不如 language。原因：prompt 引入太多 spatial/temporal 变化（prompt duration、object 配置都变），policy 容易 overfit 到 spurious cues（比如 background variation）。

这个 finding 很重要——它告诉我们 behavior prompting 的 **适用条件**：需要足够的 training task diversity，否则复杂 prompt representation 反而是 noise source。

---

## 7. 与 ICRT 的对比（Appendix G）

ICRT [arxiv.org/abs/2408.15980] 是最相近的工作，也用 demonstration 作为 prompt。但架构差异显著：

| 维度 | ICRT | BPP |
|------|------|-----|
| Architecture | Causal transformer decoder | Separate prompt encoder + diffusion decoder |
| Attention | Self-attention over [prompt + history] | Cross-attention (query=obs, kv=prompt) |
| Action generation | Autoregressive next-token | Diffusion |
| History | Entire rollout history (causal) | Fixed-length history |
| Training | Sequence of prompts + rollouts concatenated | One prompt + random (obs, action) pairs |
| Inference | References entire history each step | Pre-compute prompt, separate prompt lookup |

ICRT 的 limitation：
1. Causal attention 在 long history 下受 spurious correlation 影响
2. Fixed context length 限制 rollout duration
3. 每个 forward 都要 reference 整个 prompt + history（即使有 KV cache 也是 O(n)）

BPP 的优势：
1. Fixed history 避免 OOD drift
2. Prompt embedding 可 cache
3. Diffusion 比 autoregressive 在 continuous action 上更稳

---

## 8. Limitations 与 future directions

1. **Data diversity requirement**: 需要 substantial training diversity 才能涌现 prompting ability。低 diversity 时不如 language conditioning
2. **New action primitives**: 还没证据 BPP 能 generalize 到训练时完全没见过的 action primitives（比如训了 pick-place，prompt 一个 pushing behavior）
3. **Same environment**: prompt 和 deployment 在同一 environment，跨 environment（比如 sim prompt + real deployment）未验证
4. **Foundation pretraining**: 还没和 foundation model 结合（π₀.₅ + BPP 会怎样？）
5. **Dexterous hands**: 当前只在 gripper 上验证

未来方向：foundation behavior prompting model——大规模 pretraining + 单个 demonstration 适配新家庭环境。

---

## 9. 我的一些联想与思考

### 9.1 类比 LLM scaling laws

BPP 的 finding "task diversity > demos per task" 完全对应 LLM 的 in-context learning scaling：模型规模 + task diversity 决定 in-context ability，per-task data 边际效用递减。这暗示 robotics 也应该有类似 Chinchilla [arxiv.org/abs/2203.15556] 的 scaling law，只不过维度是 (# tasks, # demos/task, model size, prompt length)。

### 9.2 Prompt 作为 program

Behavior prompt 实际上是一种 **executable program**——demonstration 就是 trajectory program，policy 是 interpreter。这个视角下，BPP 类似 Neural Program Synthesis，但用 demonstration 而非 code 作为 program representation。这跟 Voyager [arxiv.org/abs/2305.16291] 用 code 作为 skill representation 是 two sides of the same coin。

### 9.3 Retrieval-augmented policy

Fig. 5 的 attention pattern 让我想到 RAG——cross-attention 实质是 "learned retrieval"。可以想象一个 hybrid：explicit retrieval（用 visual similarity 找 prompt 中最相关 chunk）+ learned refinement，类似 RetrievalQ [arxiv.org/abs/2404.05970] 在 robotics 上的版本。

### 9.4 与 VLA 的关系

VLA (π₀, RT-2, OpenVLA) 用 language 作为 prompt，本质是 discrete semantic instruction。BPP 用 trajectory 作为 prompt，是 dense sensorimotor instruction。两者正交——未来一个 unified model 应该能接受 heterogeneous prompts（language / goal image / partial demo / full demo），类似 VIMA [arxiv.org/abs/2210.03094] 的 multimodal prompt 思路但更彻底。

### 9.5 Test-time compute

BPP 在 inference 时 prompt embedding 可以 cache，但每次 action inference 还是要跑 K 步 diffusion。能否像 test-time scaling in LLM [arxiv.org/abs/2408.03314] 一样让 policy 在 inference 时做更多 reasoning？比如在 prompt 中检索多个候选 sub-trajectory 然后 select？

### 9.6 Bimanual 和 whole-body

iPhUMI 已经支持 bimanual，未来扩展到 whole-body mobile manipulation（参考 HoMMI [arxiv.org/abs/2603.03243]）是自然方向。Behavior prompt 在高 DoF 系统上信息密度更高，应该更有用。

### 9.7 世界模型视角

Behavior prompt 提供 dense sub-goals，这和 world model 的 trajectory rollout 类似。能否把 BPP 看作 "learned world model that takes prompt as conditioning"？这条线和 Dreamer [arxiv.org/abs/1912.01603] 系列、UniPi [arxiv.org/abs/2302.02560] 有连接点。

### 9.8 Failure mode 分析

Laundry case 暴露的 failure mode（做错 task）很有意思——说明 prompt representation 太复杂时，policy 可能 misinterpret prompt 的 "task identity" 部分。解决方向：disentangle task identity from execution detail in prompt representation。这跟 LLM 中 task description vs. few-shot example 的分离类似。

---

## 10. 总结

BPP 是一篇非常 clean 的工作，三个 contribution 互相支撑：
- **Algorithm** (BPP) 提供 in-context visuomotor policy 的实用架构
- **Data** (iPhUMI) 让 diverse data collection 和 test-time prompting 都 practical
- **Evaluation** (DrawAnything + LIBERO-Gen) 给社区提供 reproducible benchmark

最深刻的 finding 是：**task diversity is the primary driver of in-context learning ability**，这和 LLM 的经验完全一致，进一步验证了 robotics in-context learning 的 scaling 路径。

如果让我赌一下未来 1-2 年的方向：
1. BPP + foundation pretraining（π₀-style + BPP head）
2. Heterogeneous prompt（language + demo + goal image 混合）
3. Cross-embodiment behavior prompting（human video → robot execution，但用 BPP 的 dense conditioning 思路）
4. Behavior prompting 在 long-horizon task 上的 chain-of-thought 类似物

这个 paper 真的把 "in-context learning for robotics" 这个方向往前推了一大步，非常期待后续。

**Reference links**:
- Behavior Prompting: behavior-prompting.github.io
- UMI: arxiv.org/abs/2402.10329
- ICRT: arxiv.org/abs/2408.15980
- Diffusion Policy: arxiv.org/abs/2303.04137
- LIBERO: arxiv.org/abs/2306.03310
- π₀: arxiv.org/abs/2410.24164
- π₀.₅: arxiv.org/abs/2504.16054
- OpenVLA: arxiv.org/abs/2406.09246
- BC-Z: arxiv.org/abs/2202.02071
- Vid2Robot: arxiv.org/abs/2310.01936
- VIMA: arxiv.org/abs/2210.03094
- FiLM: arxiv.org/abs/1709.07871
- LoRA: arxiv.org/abs/2106.09685
- Attention Sinks: arxiv.org/abs/2309.17453
- Past-token prediction: arxiv.org/abs/2410.06564
