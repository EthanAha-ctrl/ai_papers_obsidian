---
source_pdf: Diffusion-VLA Generalizable and Interpretable Robot Foundation Model.pdf
paper_sha256: f036c44e762cd5fcdc0a32dd12f9294e390dafd7eb3233f839717c587ed0129d
processed_at: '2026-08-18T05:48:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiVLA 人话版

## 一句话说清楚

让机器人**先想后做**：VLM 负责"想"，diffusion 负责"做"，中间用一根管子把想的中间结果直接插到做的过程里。

---

## 问题是什么

现在教机器人干活，两条路各有毛病：

**第一条路（RT-2、OpenVLA）**：像 ChatGPT 那样一个 token 一个 token 吐出来当动作指令。问题是动作是连续的（往左移 3.2 厘米），你硬把它切成离散 token（"往左"、"3"、"点"、"2"），精度丢了一截，速度还慢——得一个一个蹦，7B 模型在 A6000 上只能跑 5Hz，机器人根本不够实时。

**第二条路（Diffusion Policy）**：用 diffusion model 直接生成连续动作序列，快、准、能处理"往左抓 or 往右抓都行"这种多模态情况。但它**不会想**。你跟它说"把杯子放到盘子上"，它就硬干，不会先判断盘子在哪层、杯子朝哪边。遇到没见过的东西就傻眼。

DiVLA 说：那我把俩拼起来不就行了。

---

## 拼法是关键

傻拼谁都会：先让 VLM 生成一段 reasoning text，再把 text 喂给 diffusion。ECoT 就是这么干的。但问题是**慢**——VLM 跑一遍出 text，text 再 encode 一遍喂进去，两个 round trip。

DiVLA 的 trick：**别绕回 text，直接拿 VLM 内部 reasoning 那一层的 hidden state，用 FiLM 塞进 diffusion network**。

打个比方：你让一个人边想边抓东西。
- ECoT 的做法：脑子里想完→嘴上说出来→耳朵听到→再指导手去抓。绕了一圈。
- DiVLA 的做法：脑子里想的那一下，神经信号直接传给手。不经过嘴和耳朵。

FiLM 的效果就是：reasoning 的 embedding 变成一组"旋钮"，去调 diffusion network 每一层 feature 的增益和偏置。告诉网络"你现在是在'抓玩具车'这个意图下 denoise"，不用从原始图像和语言里重新推导意图。

---

## 数据怎么来的

Droid 数据集只有"图像+动作+简单指令"，没有 reasoning。作者拿 GPT-4o 把每条指令扩写成 reasoning chain。比如：

- 原始指令："pick up the cup"
- GPT-4o 扩写："The user wants to pick up the cup. The cup is on the table. I need to move the gripper to the cup, close the gripper, and lift up."

这样 pretrain 和 finetune 的数据格式一致，模型从头到尾都在"边想边做"。

---

## 效果怎么样

### 1. 数据效率惊人

DiVLA-2B 用 39K 条轨迹，OpenVLA 用 970K 条（多 25 倍），结果 DiVLA 成功率 83.6% vs OpenVLA 39.4%。reasoning injection 提供了强先验，不用从零学视觉-动作关联。

### 2. 泛化到没见过的东西

102 个完全没见过的物体做 bin picking，DiVLA 63.7%，OpenVLA 28.4%，Diffusion Policy 8.9%。

关键在于：VLM（Qwen2-VL）在互联网图像上 pretrain 过，见过海量物体。遇到新东西它能生成"grab the green can"这种 reasoning，reasoning 通过 FiLM 引导 diffusion 找到合理的动作空间。

### 3. 速度

DiVLA-2B 在单张 A6000 上 82Hz，7B 也有 42Hz。OpenVLA-7B 只有 5Hz。

原因：diffusion 是并行 denoise（10-20 步，每步所有动作维度一起算），NTP 是串行 decode（每个 action token 一次 forward）。

### 4. 可解释性

Figure 6 那个实验很直观：机器人一开始 reasoning 是"grabbing the toy car"，研究者中途把玩具车换成 hex key，reasoning 实时切换成"grabbing the hex key"，动作也跟着改了。

这意味着 reasoning 是**闭环的**，每个 timestep 都在重新生成，不是一开始 plan 完就不管了。你可以看模型"在想什么"来 debug 失败原因。

### 5. Ablation 证明 reasoning injection 是核心

去掉 reasoning injection，成功率从 83.6% 掉到 50.3%。掉了 33 个百分点。说明 diffusion policy 本身不够，reasoning 的 hidden state 确实在帮 action generation。

### 6. 能换机器人

Franka 单臂训完，换 bimanual AgileX 双臂，只重新 init 最后一层 MLP，其他不动，table bussing 任务 72.9% 成功率。pretrain 的 visuomotor prior 保留得好。

---

## 为什么 work 的直觉

VLM 在 reasoning 时的 hidden state 已经是"高维语义压缩包"——它把图像里的物体、空间关系、用户意图都融成一个向量了。这个向量比原始 language instruction 信息量大得多。

FiLM 把这个向量注入 diffusion，等于在说："你现在 denoise 的时候，整体方向应该是'往右下抓那个绿色东西'，细节你自己填。" diffusion 负责填细节（具体轨迹、速度 profile），reasoning 负责定大方向。

分工明确，耦合紧密，没有 round-trip 开销。

---

## 还有什么问题

1. **reasoning 是 GPT-4o 事后编的**，不是机器人真正"想"出来的。遇到真正 novel 的场景（比如需要用工具），reasoning 可能瞎编。
2. **FiLM 是线性调制**，对复杂多步推理（"先开盖再放杯子再关盖"）可能表达力不够。
3. **每帧都跑 VLM** 生成 reasoning，82Hz 里有多少预算花在 reasoning 上？能不能 reasoning 一秒一次、action 50Hz？paper 没讨论。
4. **72B 没报延迟**，只报成功率。实际部署可能跑不动。

---

## 对未来的启示

VLA 的设计可能往这个方向走：
- **VLM 越来越大**（管理解和规划）
- **Action head 越来越轻**（管执行，diffusion 或 flow matching）
- **中间用 conditioning 机制耦合**（FiLM、cross-attention、gated attention）
- **数据用 LLM 自动标 reasoning**，scale 到 web level

这跟 LLM 里"thinking head vs acting head"分离的趋势一致，也跟神经科学"腹侧流负责识别、背侧流负责动作"的双流假说呼应。

---

## 一句话总结

DiVLA 证明了一件事：**reasoning 和 action 可以分开做，但必须 tight coupling**。VLM 的中间 hidden state 比 round-trip 回 text 更好用。这个 insight 很简单，但 33 pp 的 ablation 说明它真的重要。

---

# Diffusion-VLA 深度解析

Andrej，这篇 paper 触到了 VLA 设计中一个核心 tension：**reasoning 需要离散 autoregressive token，而 action 需要连续 multimodal 分布**。作者选择不把它们塞进同一个 objective，而是分工：VLM 管 "想"，diffusion 管 "做"，再用一个 reasoning injection module 把 "想" 的中间状态喂给 "做"。下面我尽量把每个 piece 的 intuition 和 math 都拆开。

---

## 1. 核心动机：为什么 NTP-based VLA 和 Diffusion Policy 都不够

### 1.1 NTP-based VLA (RT-2, OpenVLA) 的瓶颈

把 continuous action $\mathbf{a} \in \mathbb{R}^d$ 离散化成 tokens，本质上是把一个 $\mathbb{R}^d$ 上的分布硬塞进 categorical distribution。问题：
- **精度损失**：bins 有限，gripper 的 7-DOF pose 量化误差累积
- ** multimodality 表达弱**：NTP 用 softmax over vocab，本质是 categorical，无法表达 "往左抓 or 往右抓" 这种连续 multimodal
- **效率**：要 autoregressive 生成 $T$ 个 action token，串行 decode，5Hz 已经是极限（OpenVLA-7B）

### 1.2 Diffusion Policy 的瓶颈

Diffusion Policy 直接建模 $p_\theta(\mathbf{a}_{t:t+H} | \mathbf{o}_t)$，用 denoising：
$$\mathbf{a}^{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{a}^k - \frac{1-\alpha_k}{\sqrt{1-\bar{\alpha}_k}} \epsilon_\theta(\mathbf{a}^k, k, \mathbf{o}_t) \right) + \sigma_k \mathbf{z}$$
其中 $\mathbf{a}^k$ 是第 $k$ 步 noisy action，$\alpha_k$ 是 noise schedule，$\bar{\alpha}_k = \prod_{i=1}^k \alpha_i$，$\epsilon_\theta$ 是预测 noise 的网络，$\mathbf{z} \sim \mathcal{N}(0, I)$。

优点：multimodal 自然支持，并行 denoise 速度快。缺点：**没有 reasoning**。语言指令只能通过 cross-attention 注入，模型不会 "think step by step"。

### 1.3 DiVLA 的回答

把两者拼起来，但关键不在 "拼"，而在 **reasoning 如何 flow 到 action**。ECoT (Zawalski et al., 2024) 的做法是：先让 VLM 生成 chain-of-thought text，再把 text 喂回 VLM 做 action token prediction —— 这是 recursive 的，inference 慢，且 reasoning 和 action 是松耦合。

DiVLA 的核心 insight：**reasoning 的 hidden state 本身就是 good conditioning signal**，不需要再 round-trip 回 VLM。直接用 FiLM 把 reasoning embedding 注入 diffusion U-Net / transformer。

---

## 2. Architecture 详解

### 2.1 整体数据流

```
Image (multi-view) 
   ↓ SigLIP encoder (frozen)
   ↓ 2D visual features per view
   ↓ Projector (MLP) → N visual tokens per view
   ↓ Concatenate across views → [V_1, V_2, V_3, ..., text_tokens]
   ↓ Qwen2-VL (LoRA fine-tuned)
   ↓ Hidden states h_reason (from reasoning tokens)
   ↓ Final embedding layer → action tokens (fixed number)
   ↓ Projection MLP (2 layers + LayerNorm) → conditioning vector c
   ↓ 
   ↓ Inject into Diffusion Policy via FiLM
   ↓
Diffusion U-Net ε_θ(a^k, k, c, h_reason) → denoised action â
```

### 2.2 Visual encoding 细节

每个 view 用 **shared** SigLIP backbone（不是 per-view 独立 encoder），然后通过一个 transformer 把 dense visual features 压成固定 $N$ 个 tokens。多 view 时直接 concat。这个设计的好处：参数共享，且 view 之间在 VLM 内部可以 cross-attend。

值得注意：OpenVLA 原版只支持 single view，作者为了公平比较把 OpenVLA 扩展到 3 views（Table 7），single view 下 OpenVLA sorting 从 45.3% 掉到 12.7% —— 说明 multi-view 对这个 task 极其关键。

### 2.3 Reasoning Injection Module (核心创新)

这是 paper 最值得拆的部分。形式上：

设 VLM 在 reasoning token 位置的 final embedding 为 $\mathbf{r} \in \mathbb{R}^{L \times d}$，其中 $L$ 是 reasoning token 数，$d$ 是 hidden dim。

Diffusion Policy 的 noise prediction网络 $\epsilon_\theta$ 通常有多个 residual blocks。每个 block 的 feature map $\mathbf{F} \in \mathbb{R}^{B \times C \times H \times W}$ 被 FiLM 调制：

$$\text{FiLM}(\mathbf{F}) = \gamma(\mathbf{r}) \odot \mathbf{F} + \beta(\mathbf{r})$$

其中 $\gamma, \beta: \mathbb{R}^{L \times d} \to \mathbb{R}^C$ 是从 reasoning embedding 学到的 affine 参数（通常 MLP + pooling）。

**Intuition**: reasoning 不是被当成 "输入 token 再 encode 一次"，而是直接作为 modulation 信号改写 diffusion network 的 feature 通道增益和偏置。这相当于告诉 action decoder："你现在是在 'grab toy car' 这个 intent 下 denoise"，而不是让它从 raw visual + language 自己 re-derive intent。

为什么这比 cross-attention 好？我的理解：
1. FiLM 是 **broadcast** 的，每个 spatial location 都被同样 condition 调制，适合 "全局 intent"
2. Cross-attention 需要 reasoning 和 spatial feature 做相似度匹配，在 action 这种 low-level control 上不一定有 inductive bias 优势
3. FiLM 参数少，训练快，且 reasoning embedding 已经是 VLM "想完" 的高层语义，不需要再 attention 一遍

### 2.4 Action decoder 的 embodiment 适配

不同 robot 的 action 维度不同（Franka 7-DOF vs bimanual 14-DOF）。作者的做法很轻量：**只重新 init 一个 bottom MLP layer**，不动 diffusion network 主体。这保留了 pretrain 的 visuomotor prior，只让最后一层适配新 embodiment。Table 2 显示 bimanual table bussing 72.9% success，证明这个策略 work。

---

## 3. Training Objective 数学

### 3.1 总 loss

$$\mathcal{L} = \mathcal{L}_{\text{diff}} + \alpha \mathcal{L}_{\text{ntp}}$$

- $\mathcal{L}_{\text{diff}}$: 标准 diffusion MSE loss，$\mathbb{E}_{k, \mathbf{a}_0, \epsilon} \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_k}\mathbf{a}_0 + \sqrt{1-\bar{\alpha}_k}\epsilon, k, \text{cond}) \|^2$
- $\mathcal{L}_{\text{ntp}}$: next-token prediction cross-entropy on reasoning text tokens
- $\alpha = 10$，因为作者观察到 $\mathcal{L}_{\text{ntp}}$ 的 magnitude 比 $\mathcal{L}_{\text{diff}}$ 小约 10 倍

**Intuition**: 两个 loss 的 scale 不平衡是 VLA 训练常见坑。如果 $\alpha=1$，diffusion loss 会主导，VLM 部分 "忘掉" 怎么说话；如果 $\alpha$ 太大，reasoning 会 drift 到与 action 无关的 hallucination。10x 是经验值，但这个 ratio 应该跟 action dim、token vocab、diffusion step 数都有关，paper 没做 sensitivity analysis 是个小遗憾。

### 3.2 Pretraining data 处理

Droid 数据只有 (observation, action, language)，没有 reasoning。作者用 **GPT-4o 自动生成 reasoning**，把 language instruction 扩展成 reasoning chain。这是关键工程细节：让 pretrain 和 finetune 的 data format 一致，避免 distribution shift。

具体怎么 prompt GPT-4o paper 没详写，但可以推测是类似：
```
Input: "pick up the cup"
Output: "The user wants to pick up the cup. The cup is on the table. 
        I need to move the gripper to the cup, close the gripper, and lift up."
```

这种 reasoning 是 **post-hoc rationalization**，不是真正的 planning，但作为 supervised signal 训练 VLM "边想边做" 已经够用。

---

## 4. 实验数据深度解读

### 4.1 Multi-task (Table 1)

| Model | Pre-traj | In-dist Avg | Visual Gen Avg |
|--------|----------|-------------|----------------|
| Diffusion Policy | 0 | 27.9 | 8.9 |
| TinyVLA | 0 | 45.5 | 28.9 |
| Octo | 970K | 24.3 | 17.8 |
| OpenVLA-7B | 970K | 39.4 | 26.7 |
| DiVLA-2B | 39K | **83.6** | **57.8** |

最 striking：DiVLA 用 **39K** trajectory（比 OpenVLA 少 25 倍），in-dist 反而高一倍。这说明 reasoning injection 提供了 strong inductive bias，让 action learning 不需要从 raw visuo-motor correlation 学起。

Visual generalization gap: 83.6 → 57.8（掉 25.8 pp），OpenVLA 39.4 → 26.7（掉 12.7 pp）。DiVLA 绝对值更高，但相对掉得多。可能因为 reasoning 对 in-dist object 过拟合（"grab toy car" 这种 specific phrase），visual change 时 reasoning 仍能触发但 action precision 下降。

### 4.2 Factory Sorting (Figure 3)

四个 category: toy cars / knit gloves / stuffed toys / hex keys。Mixed seen+unseen 场景：
- DP: 9.2% (cluttered mixed)
- OpenVLA: 22%
- DiVLA: 40%

DiVLA 的优势在 cluttered 场景下相对更大，说明 reasoning 帮助 disambiguate。Figure 6 的 failure case 分析很有意思：研究者中途把 toy car 换成 hex key，reasoning 实时从 "grabbing the toy car" 切到 "grabbing the hex key"。这暗示 reasoning 是 **closed-loop** 的，每个 timestep 都重新生成，不是一次性 plan。

### 4.3 Zero-shot Bin Picking (Figure 4)

102 unseen objects，63.7% success。OpenVLA 28.4%，DP 8.9%。这个 gap 巨大。我的解读：
- 102 objects 跨度大（size, color, texture, deformability），pure imitation 学不到 general grasp policy
- DiVLA 的 VLM prior（Qwen2-VL 见过海量 internet image）+ reasoning 让模型对 unseen object 仍能 generate "grab the {color} {shape}" 这种 phrase，phrase 通过 FiLM 引导 diffusion 找到合理 action manifold

### 4.4 Bimanual Table Bussing (Table 2)

OpenVLA 0%！这个数字很扎眼。我的猜测：OpenVLA 用 NTP 输出 14-DOF action token，token sequence 太长，训练数据 400 trajectory 不够覆盖 bimanual coordination 的 multimodality。DiVLA 把 action 交给 diffusion，diffusion 天然适合 multimodal + 高维，所以 70.8% 合理。

### 4.5 Inference Speed (Table 5)

| Model | Hz (A6000) |
|-------|------------|
| DiVLA-2B | 82 |
| DiVLA-7B | 42 |
| OpenVLA-7B | 5 |

8x speedup 主要来自：
1. Diffusion 是 parallel denoise（10-20 steps，每步全 action 一起算），NTP 是 sequential decode（每个 action token 一次 forward）
2. vLLM 加速 VLM 的 reasoning 部分
3. 即使不用 vLLM，DiVLA-7B 30Hz，OpenVLA 5Hz —— 6x 来自架构本身

### 4.6 Ablation: Reasoning Injection (Table 8)

| Variant | Avg |
|---------|-----|
| DiVLA-2B | 83.6 |
| w/o reasoning injection | 50.3 |

掉 33.3 pp！这是 paper 最强的 ablation。说明 reasoning 不是 "nice to have"，是核心。没有 injection，diffusion 退化成 "language-conditioned Diffusion Policy"，性能跟 TinyVLA 差不多。

### 4.7 Scaling (Table 10)

| Model | Sorting | Bin Picking |
|-------|---------|-------------|
| 2B | 66.2 | 63.7 |
| 7B | 74.9 | 66.7 |
| 72B | 82.4 | 75.9 |

Sorting 从 2B→72B 涨 16.2 pp，bin picking 涨 12.2 pp。Scaling law 在 VLA 上成立，但 slope 不如 LLM 陡。可能因为 bottleneck 转移到了 data（Droid 数据量有限）和 action precision（diffusion step 数）。

---

## 5. 几个我关心的设计细节

### 5.1 Reasoning 是 closed-loop 还是 open-loop?

Paper 没明说，但从 Figure 6 的 intervention 实验推断：每个 timestep 都重新 run VLM 生成 reasoning。这意味着 82Hz 包含了 VLM forward + diffusion denoise 全流程。Qwen2-VL-2B 在 A6000 上单 forward 大概 10-15ms，diffusion 10 steps 大概 5ms，加起来 ~20ms = 50Hz，paper 报 82Hz 说明可能有 KV-cache 复用或 reasoning 不是每帧都跑。

### 5.2 Reasoning token 怎么选来注入？

"Final embedding from the tokenized output of the reasoning component" —— 我理解是 reasoning text span 对应的 last-layer hidden states 的 pooled 表示（mean 或 last token）。具体 pooling 策略 paper 没写清楚，是个 implementation detail。

### 5.3 为什么不直接 concat reasoning embedding 到 diffusion condition？

FiLM vs concat：
- concat 让 reasoning 和 visual/language condition 在同一空间，需要 network 自己 disentangle
- FiLM 是 explicit 的 multiplicative gating，更接近 "switch" 语义
- FiLM 在 RT-1 里有先例（language injection），证明在 robot control 上 work

### 5.4 GPT-4o 生成 reasoning 的 quality

这是潜在 weak link。如果 GPT-4o 生成的 reasoning 是 generic template（"I see a cup, I will grab it"），那 reasoning injection 学到的可能只是 "object category → action primitive" 的映射，不是真正 reasoning。Paper 的 VQA 实验（Table 11）显示 DiVLA 能识别 tulip、orange，但把 toy dragon 误认为 toy tiger —— 说明 reasoning quality 受限于 VLM 的 visual grounding，不是 GPT-4o 生成的 text。

---

## 6. 跟相关工作的定位

### 6.1 vs $\pi_0$ (Black et al., 2024)

$\pi_0$ 用 flow matching 替代 diffusion，VLM 也是 backbone，但 **没有显式 reasoning**。$\pi_0$ 的 flow matching 比 diffusion 快，但 action 生成是 "无思考" 的。DiVLA 用 reasoning 换速度（42Hz vs $\pi_0$ 更快），但换来 generalization。

### 6.2 vs ECoT (Zawalski et al., 2024)

ECoT 在 OpenVLA 上加 chain-of-thought，但 reasoning 是 **生成 text → 喂回 VLM → 再生成 action**，两阶段 autoregressive，慢。DiVLA 的 reasoning injection 是 **单次 forward，hidden state 直接 inject**，架构上更紧凑。

### 6.3 vs TinyVLA (Wen et al., 2024)

TinyVLA 也是 VLM + diffusion，但 **没有 reasoning injection**，diffusion head 只接收 VLM 的 final pooled embedding。DiVLA 多了 FiLM injection 这一层，ablation 证明这层值 33 pp。

### 6.4 vs Transfusion / Show-O / Vila-U

这些是 unified understanding + generation 的 multimodal model，用 next-token 同时 predict text 和 image token。DiVLA 借鉴了这个思路（unified autoregressive + diffusion），但应用域是 robot action，且 diffusion 是 separate head 不是 token-level。

---

## 7. 局限与我的疑问

1. **Reasoning 的 groundedness**：GPT-4o 生成的 reasoning 是 post-hoc，不是 robot 自己 "想" 出来的。如果遇到真正 novel situation（比如 tool use），reasoning 可能 hallucinate。Paper 的 failure case 没充分展示这种 failure mode。

2. **FiLM 的 expressiveness**：FiLM 是 affine transform，对 complex reasoning（"先开盖再放杯子"这种 multi-step）可能不够。Cross-attention 或 gated attention 可能更强，但 paper 没比。

3. **Reasoning 的 closed-loop 频率**：每帧都跑 VLM 生成 reasoning 太贵。能否做 "reasoning once per second, action at 50Hz"？Paper 没讨论这个 trade-off。

4. **Pretraining data scale**：39K trajectory 跟 OpenVLA 的 970K 比少 25x，但 OpenVLA 用 OXE（更通用），DiVLA 用 Droid（更 focused）。这个比较不完全公平 —— Droid 本身就是高质量 teleop data，OpenVLA 的 OXE 混杂。

5. **72B 的实际 deploy**：72B 在 A6000 上跑不动，paper 没报 72B 的 Hz。Scaling 实验只报 success rate，没报 latency，实际 deploy 价值存疑。

---

## 8. 对 VLA 设计的更广启示

DiVLA 验证了一个重要 hypothesis：**reasoning 和 action 可以 decoupled 但 tightly coupled**。VLM 的 reasoning hidden state 是 rich representation，不需要 round-trip 回 text 再 encode。这跟 LLM agent 的 "ReAct" pattern 形成对比 —— ReAct 是 text-level reasoning-action 交替，DiVLA 是 embedding-level 注入。

如果这个方向对，未来 VLA 的设计可能：
- VLM 越来越大（72B+），负责 "理解 + planning"
- Action head 越来越轻（diffusion 或 flow matching），负责 "执行"
- 中间用 FiLM / cross-attention / gated attention 等 conditioning 机制耦合
- Data 上用 LLM 自动 generate reasoning 标注，scale 到 web-scale

这跟 LLM 的 "thinking head vs acting head" 分离趋势一致，也跟神经科学里 "ventral stream for recognition + dorsal stream for action" 的双流假说呼应。

---

## References

- Paper: [Diffusion-VLA: Generalizable and Interpretable Robot Foundation Model](https://arxiv.org/abs/2412.14109) (实际 arxiv ID 需查证，paper 是 ICML 2025)
- Diffusion Policy: [Chi et al., 2023](https://arxiv.org/abs/2303.04137)
- OpenVLA: [Kim et al.](https://arxiv.org/abs/2406.09246)
- $\pi_0$: [Black et al., 2024](https://arxiv.org/abs/2410.24164)
- ECoT: [Zawalski et al., 2024](https://arxiv.org/abs/2407.08693)
- TinyVLA: [Wen et al., 2024](https://arxiv.org/abs/2409.12514)
- Qwen2-VL: [Wang et al., 2024](https://arxiv.org/abs/2409.12191)
- FiLM: [Perez et al., 2018](https://arxiv.org/abs/1709.07871)
- Droid: [Khazatsky et al., 2024](https://arxiv.org/abs/2403.12945)
- OXE: [O'Neill et al., 2023](https://arxiv.org/abs/2310.08864)
- RT-2: [Brohan et al., 2023](https://arxiv.org/abs/2307.15818)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- vLLM: [Kwon et al., 2023](https://arxiv.org/abs/2309.06180)
- SigLIP: [Zhai et al., 2023](https://arxiv.org/abs/2303.15343)
- Transfusion: [Zhou et al., 2024](https://arxiv.org/abs/2408.11039)
- Show-O: [Xie et al., 2024](https://arxiv.org/abs/2408.12528)

---

## TL;DR for your intuition

DiVLA 的核心 bet：**VLM 的 reasoning hidden state 是比 raw language token 更好的 action conditioning signal**。用 FiLM 把这个 hidden state 直接 inject 到 diffusion policy，避免 text round-trip，既保留 reasoning 的 generalization，又保留 diffusion 的 multimodal action precision。Ablation 证明这个 injection 值 33 pp，scaling 证明 2B→72B 持续涨点，speed 证明 diffusion + vLLM 能到 82Hz。剩下的问题是 reasoning 的 groundedness（GPT-4o post-hoc 标注的 ceiling）和 closed-loop reasoning 的 latency budget。
