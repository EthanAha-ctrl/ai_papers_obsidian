---
source_pdf: RLDX-1 Technical Report.pdf
paper_sha256: 882fba553790218cd215276818c0a8efd41c74db7d9b525d464d762f0850cea9
processed_at: '2026-08-12T00:04:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RLDX-1 用人话说

Andrej，好，我换个频道，抛开术语，用大白话给你讲这篇paper到底在干嘛。

---

## 一句话概括

现有的 robot brain（VLA）很会"看图说话"——给它一张桌面照片和一句指令，它能抓东西放东西。但一旦你让它抓传送带上跑着的瓶子，或者让它记住三分钟前你把东西塞进了哪个盒子，或者让它插一个看不见的插头，它就歇菜了。RLDX-1 就是给 robot brain 加了三个"感官模块"，让它不只是会看，还会**感知运动、记住历史、感受触碰**。

---

## 为什么要做这件事

你想象一下你自己在厨房倒水。你眼睛看着壶和杯子，但你其实还在用好几样别的东西：

- **手腕的力道**：你知道杯子什么时候满，靠的是手腕感觉到的重量变化，光看是看不出来的
- **时间记忆**：你记得三分钟前把糖罐子放在左边灶台上了，现在要拿的时候你直接去左边，不用满厨房找
- **运动预判**：有人扔给你一个苹果，你不用等苹果飞到你脸上才反应，你提前伸手去接

现有 VLA 模型（π₀、π₀.5、GR00T N1.6 这些）只有"视觉+语言"这一条通道。它们就像一个只会看不会摸、没有短期记忆、不会预判运动的人。所以一遇到需要这三种能力的任务就崩。

RLDX-1 的 thesis 就是：**光有 versatile intelligence 不够，robot 还得有 functional capabilities**。

---

## 怎么做到的

### 三个新"感官"

**1. Motion Awareness（运动感知）**

方法叫 STSS（Space-Time Self-Similarity）。intuition 很简单：视频里物体在动，那么相邻帧的对应位置会有变化，这个变化本身就在编码运动信息。STSS 显式地去算每个时空位置和它邻居的相关性，然后把运动特征 inject 回 vision encoder。

打个比方，普通 vision encoder 看四张连续照片就像看四张独立照片。加了 STSS 之后，它在每一层都会去算"这个位置的像素和旁边位置的像素、前一帧的位置有什么关联"，于是它就"感觉到"了运动方向和速度。

然后在 LLM 那一侧，前几层让所有帧的 token 都进去做 attention 累积 temporal context，到第 4 层之后把过去几帧压缩成一个 token（average pooling），只保留当前帧的全部 token。这样既 capture 了运动，又不会让 KV cache 爆炸。

**2. Long-Term Memory（长期记忆）**

多帧视频能覆盖几秒钟，但 shell game（猜杯子下面有没有东西）这种任务需要 30 秒以上的记忆。所以 RLDX-1 在 VLM 后面挂了一个 memory module——维护一个队列，存最近 3 个"历史 cognition feature"，间隔是 action chunk 的长度（ALLEX 是 40 步）。

每次推理时，把当前 cognition feature 和队列里的 3 个历史 feature 拼起来过一个 small Transformer，输出一个 memory feature。用 causal attention 保证时间顺序。

这就像你给人加了一个"小本本"——每隔一会儿记一笔，需要的时候翻一翻。比让 VLM 自己在 attention 里"记住" 30 秒前的事靠谱得多。

**3. Physical Sensing（物理感知）**

ALLEX 用关节扭矩，FR3 用扭矩 + AnySkin 触觉传感器（15 维力向量）。这些信号单独走一个 "physics stream" 进入 action model。

关键 trick：训练时不光预测 action，还预测 future physical signals。也就是说让 model "想象"未来 40 步的扭矩/触觉会怎么变。这逼着 model 内化物理交互的 dynamics——它得真理解"我这么动的话，指尖会感觉到什么"。

如果物理信号 unavailable（比如某个 robot 没装触觉），就把 P stream 的 attention mask 掉，model 自动 fallback 到纯视觉模式。很 elegant。

---

### MSAT：多流 Action Transformer

这部分是架构核心。intuition 是这样的：

你有四种完全不同的输入——cognition feature（高维视觉-语言压缩）、proprioceptive state（低维但高保真关节角）、physical signal（稀少但关键）、noisy action chunk。如果直接全部 concatenate 扔进一个 Transformer，会发生什么？数据多的 modality（cognition）会 dominate，数据少的 modality（touch）的信号被淹没。

MSAT 的解法：每个 modality 走自己的 stream，各自做 normalization 和 QKV projection，然后在 self-attention 的时候 concat 起来一起 attend，完了再 split 回各自的 stream 做 residual update。

这就像一个会议室里四个人各自带着自己的资料进来，开会的时候大家一起讨论（joint attention），但各自记各自的笔记（stream-wise residual）。信息交换了，但 representation space 没被打乱。

早期 block 是 triple-stream（C/A/P 分开），后期 block 合并成 double-stream（C-A 合一 + P）。这是借鉴 MM-DiT（Stable Diffusion 3 的架构）的设计，先分开各自 processing，再合并 joint reasoning。

训练用 flow-matching（和 π₀ 一样）——给 clean action 加噪声，让 model 预测从噪声到 clean 的 velocity field。推理时用 Euler 方法迭代 4 步去噪。

---

### Synthetic Data：用视频生成模型造数据

 humanoid robot 的 demonstration 数据很贵（teleop 收集，一小时可能就几十条）。RLDX-1 的解法是用 video generation model 来"造"数据：

1. 拿一个真实 demonstration 的第一帧
2. 用 FLUX.2 编辑这张图（换桌子、换光照、换背景、换物体，但保留 Canny edge 结构）
3. 用 Cosmos-Predict2 生成新视频
4. 用一个 Inverse Dynamics Model（IDM）给视频标注 action
5. **关键步骤**：motion-consistency filtering——把 IDM 预测的 action 在 simulator 里 replay 出来，和生成的视频用 V-JEPA2 编码后做比对，只保留 motion 对得上的样本

第 5 步是整个 pipeline 的精髓。生成的视频看起来 plausible 不代表 IDM 标的 action 真能 reproduce 那个视频。用 simulator 当 ground truth checker，把"video quality"问题转化成"motion alignment"问题。

结果：GR-1 Tabletop 上，纯 real data 41%，加 100% synthetic data 涨到 50.1%。

---

### 三阶段训练

- **Pre-training**（100K steps，195 小时 64×H200）：1.5M episodes，涵盖 single-arm / dual-arm / humanoid，学 general manipulation prior
- **Mid-training**（25K steps，15 小时）：分 ALLEX 和 FR3 两个 embodiment，注入 memory / motion / physics 三个新模块。trick 是前 2K 步冻结所有 pre-trained 参数只训新模块（alignment warmup），新模块参数 near-zero 初始化，防止随机初始化破坏已有 representation
- **Post-training**：task-specific fine-tune + 可选的 RECAP RL

RECAP RL 的创新是 text-based VLM critic——不搞一个新的 value head，直接让 VLM 用原生 text generation 预测一个整数 value。这避免了 distributional mismatch。在 Light Bulb Twisting 任务上，RECAP 训练 3 轮后比人类 teleop 还好（353 frames vs 人类 ~700 frames）。

---

### Inference 优化

这是个很实的 engineering contribution。问题很简单：robot control 是 closed loop，observation → inference → action，每一步都有 delay，delay 越大 action 越过期。原始 PyTorch eager 在 RTX 5090 上 71.2ms 一步。

两步优化：
1. **Static graph conversion**：预计算所有 runtime-dependent 的东西（RoPE、attention mask），整个 forward pass 变成一个 CUDA Graph，一步 launch。降到 48.9ms
2. **Custom fused kernels**：手写 Triton kernel 融合 RMSNorm + RoPE + Attention 等操作，减少 HBM round-trip。降到 43.7ms

1.63× speedup，而且对 memory/physics 模块的存在 robust。

---

## 实验结果讲人话

### 模拟器基准

LIBERO 上大家都 97% 左右，saturated 了，看不出差距。有意思的是 RoboCasa365 的 composite tasks——RLDX-1 19.0%（seen）和 5.6%（unseen），GR00T N1.6 只有 12.6% 和 2.6%。long-horizon compositionality 是 RLDX-1 的强项。

### 真机 humanoid（ALLEX）

这是最能说明问题的数据：

- **传送带抓取**：π₀.5 和 GR00T N1.6 基本只会用训练时见过的固定速度动作，遇到 unseen 速度就崩。RLDX-1 在 unseen 速度上还有 75% 成功率——它真的学会了根据传送带速度调整动作节奏
- **猜盒子**（long-term memory）：baselines 30% 左右（=随机猜），RLDX-1 91.7%
- **滑卡片**（contact-rich）：baselines 各种失败模式（滑不准、抓不起来、交接掉落），RLDX-1 进度分 97.2/100
- **倒水**（weight sensing）：baselines 连一次都完成不了，倒完水卡在倒水姿势不知道该停。RLDX-1 进度分 70.8，能感知杯子重量变化然后继续完成任务

### 真机 FR3

- **Shell Game**（猜杯子下藏东西）：baselines 50%，RLDX-1 91.7%——memory module 直接决定成败
- **Plug Insertion**（插头插入，完全 occluded）：baselines 20% 左右，RLDX-1 33.3%——touch/torque 让它知道什么时候对准了
- **Egg PnP**（抓鸡蛋不碎）：RLDX-1 61.1% vs π₀.5 45.8%——触觉感知 grip force

---

## 我的几个直觉判断

**1. 这个 thesis 站得住。** 过去两年 VLA 领域一直在卷 versatility（更大 VLM、更多 data、更广 embodiment），但真实世界的 manipulation 任务大量需要 motion / memory / physics 这三种能力。RLDX-1 指出了这个 gap 并且给了一个 clean 的 architectural solution。

**2. MSAT 是对的方向。** heterogeneous modalities 不应该暴力 concatenate。multi-stream + joint attention 让每个 modality 保留自己的 representation space 同时做 cross-modal exchange，这个 design pattern 会成为 VLA 的标配。

**3. Motion-consistency filtering 是 synthetic data 的关键 insight。** 用 video gen model 造 robot data 的 idea 不新（DreamGen 做过），但"用 simulator replay + V-JEPA2 比对"来过滤是新的。这解决了一个根本问题——生成的 video plausible ≠ 标注的 action 正确。

**4. RL 超越人类 demonstration 是个 milestone。** RECAP₃ 在 Light Bulb Twisting 上比人类 teleop 更高效。这说明 RL 在 VLA 上不只是"补齐短板"，还能"突破上限"。但目前只在单任务验证，generalization 还需探索。

**5. Inference optimization 不 sexy 但 critical。** 71ms → 44ms 看起来只是工程优化，但对 closed-loop control 来说这是"能不能用"和"用得好不好"的区别。手写 Triton kernel 融合 RMSNorm + RoPE + Attention 这种事，未来会是 VLA deployment 的标配工作。

**6. 一个值得思考的 counterfactual**：如果 VLM 足够大、data 足够多，这些 functional capabilities 会不会"涌现"出来，不需要 architectural module？目前论文的 ablation 数据（尤其 Conveyor PnP unseen speed 75%）说明 inductive bias 仍然 matters。但这个 question 会随着 scaling 继续被 re-examine。

---

## 相关链接

- [RLDX-1 项目主页](https://rlwrld.ai/rldx-1)
- [GitHub 代码](https://github.com/RLWRLD/RLDX-1)
- [HuggingFace 模型](https://huggingface.co/collections/RLWRLD/rldx-1)
- [Qwen3-VL](https://arxiv.org/abs/2511.21631) —— VLM backbone
- [π₀](https://arxiv.org/abs/2410.24164) / [π₀.5](https://arxiv.org/abs/2504.16054) —— baseline + flow-matching 范式
- [GR00T N1](https://arxiv.org/abs/2503.14734) —— baseline
- [MM-DiT (SD3)](https://arxiv.org/abs/2403.03206) —— MSAT 架构灵感
- [Flow Matching](https://arxiv.org/abs/2210.02747) —— action 生成
- [V-JEPA2](https://arxiv.org/abs/2506.09985) —— motion-consistency filtering
- [RECAP](https://arxiv.org/abs/2511.14759) —— RL post-training
- [ContextVLA](https://arxiv.org/abs/2510.04246) —— temporal compression
- [HAMLET](https://arxiv.org/abs/2602.18742) —— memory module
- [VPT (IDM)](https://arxiv.org/abs/2210.00032) —— inverse dynamics
- [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) —— video generation
- [AnySkin](https://arxiv.org/abs/2503.04246) —— tactile sensor

---

总结一句：RLDX-1 告诉我们，robot brain 不只需要"眼睛和嘴"，还需要"内耳（运动感知）、海马体（记忆）、皮肤（触觉）"。它用一套 clean 的 multi-stream 架构把这三样东西 plug 进了 VLA，然后用 synthetic data + 三阶段训练 + inference 优化把整个系统跑通了。ALLEX humanoid 上 ~90% 成功率 vs baselines ~40%，这个 gap 说明了 functional capabilities 的价值。

Hope this builds your intuition, Andrej!

---

# RLDX-1 技术报告深度解析

Andrej，这篇来自 RLWRLD（与 KAIST 合作）的 RLDX-1 技术报告是 2026 年机器人 VLA 领域的一篇重磅工作。我会从 architecture、data、training、inference 四个维度给你 build intuition，同时关联相关工作。

参考链接：
- 项目主页: https://rlwrld.ai/rldx-1
- GitHub: https://github.com/RLWRLD/RLDX-1
- HuggingFace: https://huggingface.co/collections/RLWRLD/rldx-1
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- π₀: https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- MM-DiT (SD3): https://arxiv.org/abs/2403.03206
- Flow Matching: https://arxiv.org/abs/2210.02747
- V-JEPA2: https://arxiv.org/abs/2506.09985
- RECAP: https://arxiv.org/abs/2511.14759

---

## 1. 核心论点：从 Versatile Intelligence 到 Functional Capabilities

论文的核心 thesis 很清晰：现有 VLA（如 π₀、π₀.5、GR00T N1.6）只关注 **versatility**（scene understanding + language generalization），却忽略了真实 dexterous manipulation 必备的 **functional capabilities**。论文聚焦三大功能：

1. **Motion awareness** —— 处理 dynamic environments（conveyor belt、moving objects）
2. **Long-term memory** —— sequential tasks 需要历史推理（shell game、cup swapping）
3. **Physical sensing** —— contact-rich tasks 需要 tactile/torque 信号（plug insertion、egg pick、pouring）

这是一个非常合理的 framing。传统 VLA 基本都是 static-frame + language-conditioned，而真实世界的 manipulation 任务大量需要这三种"超越视觉"的能力。这让我联想到你在 Eureka Labs 讲过的 intuition：robotics 的瓶颈往往不在 perception，而在 closed-loop feedback 和 temporal reasoning。

---

## 2. Neural Architecture 深度解析

RLDX-1 架构分为两大模块：**VLM backbone** 和 **Action Model (MSAT)**。

### 2.1 VLM：Qwen3-VL 8B + Cognition Tokens + 功能扩展

**基础 VLM 选择**：Qwen3-VL 8B。这是一个 native-resolution vision encoder 的 VLM，可以处理 arbitrary aspect ratio 不裁剪。这点很关键，因为 robot demonstrations 的相机视角通常不是正方形。

**Cognition Tokens 机制**（这是个非常 clever 的设计）：

给定 video observation $\mathbf{o}_{t-K:t}$ 和 language instruction $\mathbf{l}_t$，输入 token 序列构造为：

$$\mathbf{x} = [\mathbf{v}_t, \mathbf{l}_t, \mathbf{q}]$$

其中：
- $\mathbf{v}_t = \mathcal{E}_\theta(\mathbf{o}_{t-K:K})$ —— vision encoder 输出的 video features
- $\mathbf{l}_t$ —— language instruction tokens
- $\mathbf{q}$ —— 64 个 learnable cognition query tokens

只有 cognition token 对应的输出 $\mathbf{h}_t$ 被保留作为 cognition features，其他输出全部 discard。

这个设计相当于一个 **learnable cross-attention bottleneck**，让 VLM 主动"压缩"出 action-relevant 信息。这让我想到 Perceiver / Q-Former 的设计思想，但这里直接嵌入了 VLM 内部，让 cognition token 可以 attend to 整个 visual-linguistic context。实践上用 64 个 tokens 是合理的——既够 expressive，又不会让下游 MSAT 太重。

**VQA adaptation**：直接用 Qwen3-VL 做 robotics 缺乏 embodied grounding，所以构建了 robot-specific VQA dataset fine-tune，覆盖三个维度：
- Spatial relationships（end-effector 与 target object 的空间关系）
- Intermediate subtasks（task decomposition）
- Low-level actions（当前 frame 对应的 action）

ablation 显示 VQA training 把 RoboCasa Kitchen 从 57.5% 提到 60.9%。

**VLM 特征层选择**：ablation 显示 Layer 18（中间层）最好（60.9%），Layer 8 太浅（51.1%），Layer 28 太深（56.3%）。这个现象和 Bjorck et al. 2025 的观察一致——浅层缺语义，深层过度抽象丢细节。这让我想到 LLM 的 "middle layers contain factual knowledge" 的 interpretability 发现。

### 2.2 Functionality 1: Motion Awareness (STSS Module)

**核心模块**：Space-Time Self-Similarity (STSS)。

设 $\mathbf{v}_t^{(i)}$ 为前 $i$ 层 vision encoder 处理后的 video features。STSS 模块计算每个 spatio-temporal feature 与其 local neighbors 的相关性，得到 self-similarity tensor $\mathbf{S}_t$，然后通过 STSS encoder $S_\theta$ 生成 motion features，残差更新：

$$\tilde{\mathbf{v}}_t^{(i)} = \mathbf{v}_t^{(i)} + S_\theta(\mathbf{S}_t)$$

**关键设计 choice**：插入到 vision encoder 第 9 层（共 27 层，约 30% 深度）。motivation 来自 Joseph et al. 2026 的发现：physical cues 在 ~30% 深度最丰富。这是个有趣的可解释性发现，类似于 CNN 的 early layers 提取 edges/textures 的传统观察。

**LLM 端 temporal compression**（来自 ContextVLA, Jang et al. 2025a）：
- Early layers（前 4 层）：multi-frame tokens 按 temporal order 输入，利用 causal attention 累积 temporal context
- Layer 4 之后：past observations 压缩成 single context token（average pooling），只保留 current frame 全部 tokens
- 这样大幅减少 KV cache 和 FLOPs

之所以选 Layer 4 而不是 ContextVLA 原始的 Layer 2，是为了保留 Qwen3-VL 的 DeepStack design——前 4 层 fuse multi-level vision encoder features。

**直觉 build**：motion awareness 的关键 insight 是——视频帧之间的"差异"本身就编码了运动信息。STSS 显式建模这种 spatio-temporal self-similarity，让网络不必从头学习 motion extraction。这比单纯堆叠 frames 让 LLM 自己 figure out 强得多。

### 2.3 Functionality 2: Long-Term Memory Module

**Memory Queue**：

$$\mathbf{Q}_t = [\mathbf{h}_{t - n_{\text{mem}} \cdot H}, \ldots, \mathbf{h}_{t - 2H}, \mathbf{h}_{t - H}]$$

- $n_{\text{mem}} = 3$ —— 记忆容量
- $H + 1$ —— action chunk horizon（ALLEX 用 40，FR3 用 16）
- 采样间隔 = $H + 1$，避免冗余

**Memory Transformer**：

$$\mathbf{m}_t = \mathcal{M}_\theta([\mathbf{Q}_t, \mathbf{h}_t])$$

使用 causal attention，让 later timesteps 只 attend to 自己和更早的。同时把 $\mathbf{m}_t$ 和原始 $\mathbf{h}_t$ 都送进 action model——这样 model 既有"compressed long-term context"也有"raw current cognition"。

**直觉 build**：multi-frame observations 只能覆盖几秒，但 sequential tasks（如 cup swapping、shell game）需要 30 秒以上的 memory。Memory module 用一个 explicit queue + small transformer 解决这个问题，比让 VLM 自己 "记住" history 更可靠。这和 HAMLET (Koo et al. 2026) 思路一致。

### 2.4 Action Model: Multi-Stream Action Transformer (MSAT)

这是论文最重要的架构创新。

**Flow-Matching Formulation**：

训练时采样 denoising timestep $\tau \in [0,1]$ 和 noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，构造 noisy action chunk：

$$\mathbf{a}_{t:t+H}^{\tau} = \tau \mathbf{a}_{t:t+H} + (1-\tau)\boldsymbol{\epsilon}$$

- $\tau = 0$：纯噪声
- $\tau = 1$：clean action
- 训练目标预测 velocity field $\mathbf{a}_{t:t+H} - \boldsymbol{\epsilon}$

Loss：

$$\mathcal{L}(\theta; t, \tau, \epsilon) = \|\mathbf{u}_\theta(\mathbf{a}_{t:t+H}^{\tau}, \tau, \mathbf{c}_t) - (\mathbf{a}_{t:t+H} - \boldsymbol{\epsilon})\|_2^2$$

其中 $\mathbf{c}_t = [\mathbf{h}_t, \mathbf{m}_t, \mathbf{s}_t, \mathbf{p}_t]$ 是 conditioning inputs。

推理时用 Euler method：

$$\mathbf{a}_{t:t+H}^{\tau_{i+1}} = \mathbf{a}_{t:t+H}^{\tau_i} + (\tau_{i+1} - \tau_i)\mathbf{u}_\theta(\mathbf{a}_{t:t+H}^{\tau_i}, \tau_i, \mathbf{c}_t)$$

从 $\tau_1 = 0$ 到 $\tau_T = 1$，共 $T$ 步。

**MSAT 架构**：扩展 MM-DiT (Stable Diffusion 3 架构) 到 action modeling。

- **Early blocks**：triple-stream（C/A/P 分别走自己的 stream）
  - C stream: cognition features $[\mathbf{h}_t, \mathbf{m}_t]$
  - A stream: proprioceptive state + noisy actions $[\mathbf{s}_t, \mathbf{a}_{t:t+H}^{\tau}]$
  - P stream: physical signals $\mathbf{p}_t$（可选）

- **Late blocks**：double-stream
  - Merged C-A stream
  - P stream

**Joint self-attention**：每个 stream 独立做 normalization 和 QKV projection，然后 concatenate 沿 token 维度做 joint attention，再 split 回各 stream，stream-wise residual update。

**直觉 build**：MSAT 的精髓在于—— heterogeneous modalities 各自有不同的 data scale 和 semantics（cognition 是 high-dim visual-linguistic，proprioception 是 low-dim 但 high-fidelity，physical signals 是 data-scarce 但 critical）。如果直接 concatenate 它们扔进一个 transformer，小数据 modalities 会被 dominate。Multi-stream 设计让每个 modality 保留自己的 representation space，但通过 joint attention 做 cross-modal exchange。

**Functionality 3: Physical Sensing (P Stream)**：

P stream 处理 tactile/torque signals，关键设计：
1. **可禁用**：physical signals 不可用时 mask 掉 P stream 的 attention（让它不参与计算）
2. **Auxiliary objective**：预测未来 $L$ 步 physical signals $\mathbf{p}_{t+1:t+L}$

具体地，对 future physical signals 也做 flow-matching：

$$\mathbf{p}_{t+1:t+L}^{\tau} = \tau \mathbf{p}_{t+1:t+L} + (1-\tau)\boldsymbol{\epsilon}_\mathbf{p}$$

P stream 预测 velocity $\mathbf{p}_{t+1:t+L} - \boldsymbol{\epsilon}_\mathbf{p}$，与 action denoising 联合训练。

**直觉 build**：让 model "想象" future physical signals，强迫它 internalize physical interaction dynamics。这个 auxiliary loss 类似 world model 的 predictive objective，但聚焦在 physical modality。当 tactile/torque 不见时，model 已经学会了"如果在 contact 会怎样"的 prior。

**进一步设计 choices**：
1. **RoPE on A stream**：捕获 action chunk 内的 relative temporal structure
2. **τ 作为 in-context token**：通过 sinusoidal embedding + MLP，prepend 到 A sequence 作为单个 token，参与 attention。这取代了 adaLN（Peebles & Xie 2023 DiT 用的），避免了 per-block affine modulation
3. **RMSNorm + SwiGLU**：现代 Transformer 标准

**Embodiment sharing**：MSAT 参数 cross-embodiment shared，只用 lightweight embodiment-specific projection layers 适配不同机器人。这和 π₀ 的 design 一致。

---

## 3. Training Data 详解

### 3.1 Public Real-World Data

| Dataset | Embodiment | End-Effector | Episodes |
|---------|------------|--------------|----------|
| Open-X-Embodiment | Single-arm | Gripper | 870K |
| DROID | Single-arm | Gripper | 92K |
| Galaxea Open-World | Dual-arm | Gripper | 114K |
| AgiBot World (G) | Humanoid | Gripper | 239K |
| AgiBot World (H) | Humanoid | Hand | 36K |
| Fourier ActionNet | Humanoid | Hand | 30K |
| Humanoid Everyday | Humanoid | Hand | 9K |
| Synthetic Data | Humanoid | Hand | 150K |
| **Total** | | | ~1.5M |

OXE 用的是 "Magic Soup" mixture（Octo/OpenVLA 那套），主要 Fractal/Kuka/BridgeV2 各 16%。

### 3.2 In-house Data

**ALLEX humanoid**：48-DoF upper-body
- 7-DoF × 2 arms
- 15-DoF × 2 five-finger hands
- 2-DoF waist + 2-DoF neck
- Stereo egocentric cameras
- Joint torques from motor currents

**Franka Research 3 (FR3)**：DROID setup + AnySkin tactile sensor + joint torque
- Tactile: 15-dim（5 sensing units × 3-axis force）
- Torque: 7-dim（per-joint）

Teleoperation 系统：Meta Quest VR (head/waist) + Vive Trackers (wrists) + Manus Pro gloves (fingers) + IK。这是个相当 sophisticated 的 multi-device teleop stack。

### 3.3 Synthetic Data Pipeline（最有意思的部分）

整体 pipeline：
1. **Source video** → I2I editing (FLUX.2-dev) → I2V generation (Cosmos-Predict2) → V2V transfer (Cosmos-Transfer2.5) → IDM action annotation → 过滤
2. Augmentation axes:
   - **Task augmentation**: factorized instruction composition（behavior × target × placement × hand）+ skill-primitive-conditioned variation
   - **Scene augmentation**: I2I 编辑 table/object/lighting/background，condition on Canny edge map 保持 structure
3. **Filtering**:
   - Video quality filtering: VLM 评估 instruction following + trajectory plausibility（1-5 score）
   - **Motion-consistency filtering**（关键创新）: replay IDM-predicted actions in simulator → render rollout video → 与 synthetic video 用 V-JEPA2 + attentive probe 比较

**Motion-consistency filtering 详解**：

Attentive probe 结构：
- Frozen V-JEPA2 video encoder
- 单 cross-attention layer with learnable query token
- Attend to concatenated embeddings of two video clips
- Linear head 预测 alignment logit

Training pairs：
- **Positive**: real clip + simulator rollout from ground-truth action
- **Negative**: 
  - Same episode 时间窗口 shifted
  - 不同 episode 但同 task instruction

**直觉 build**：这是整个 synthetic pipeline 的核心 insight——generated video 看起来 plausible 不代表 IDM-annotated actions 真的能 reproduce 那个 video。Motion-consistency filter 用 simulator 做 "ground truth checker"，把 action-video alignment 问题转化为 motion matching 问题。V-JEPA2 作为 frozen encoder 提供了 strong motion representation。

参考：
- V-JEPA2: https://arxiv.org/abs/2506.09985
- FLUX.2: https://bfl.ai/blog/flux-2
- Cosmos: https://github.com/nvidia-cosmos/cosmos-predict2
- IDM (VPT): https://arxiv.org/abs/2210.00032

---

## 4. Training Procedure: Three-Stage Pipeline

### Stage 1: Pre-Training (100K steps, 64×H200, 195 hours)

- Global batch size 8192
- 4-frame video observations, temporal offsets $\{-6, -4, -2, 0\}$
- Actions/states 归一化到 $[-1, 1]$（per-dataset 1st/99th percentile）
- VLM backbone frozen except top 4 layers
- AdamW, lr=1e-4, constant schedule + 5% linear warmup
- **Embodiment-agnostic projection layer**：每个 batch 一小部分样本走 shared projection，为 unseen embodiments 提供初始化

### Stage 2: Mid-Training (25K steps, 15 hours on 64×H200)

- Batch size 1024, lr=5e-5
- **ALLEX**: 5:5 in-house:synthetic ratio, horizon 40
- **FR3**: 8:2 DROID:in-house ratio, horizon 16
- **Modality dropout 0.3**：每个 expanded modality 独立 dropout
- **2K-step alignment warmup**：frozen 所有 pre-trained params，只训 new modality-specific params
- P stream 参数 near-zero 初始化

**直觉 build**：mid-training 是 "functionality expansion"——把 pre-trained generalist 变成 embodiment-specific expert，同时注入 memory/motion/physics 三大新能力。Alignment warmup + near-zero init 是关键 stabilization trick——避免新模块的随机初始化破坏 pre-trained representations。

### Stage 3: Post-Training

**Adaptive Data Collection**：
- Base stage: 分解 task 为 atomic motion primitives（reach, grasp, move, place, wait），定义 consistency factors（固定）和 variance factors（多样化）
- Refinement stage: 训练 → 部署 → 发现 failure modes → 扩展 variance factors → 补充 demonstrations → 迭代

**Reinforcement Learning (RECAP + VLM Critic)**：

RECAP (Amin et al. 2025) 的核心是 decouple critic training from policy optimization。RLDX-1 的创新是 **text-based VLM critic**——不引入新 prediction head，直接用 VLM 的 native text-prediction interface 预测 integer value 作为 text token。

这避免了 prior VLM critics（Tan et al. 2025; Liang et al. 2026）的 distributional mismatch 问题。VLM 用 gemma3-4b-it，LoRA rank 128，1 epoch on success demonstrations only。

Algorithm 1（RECAP Post-Training）：
1. Train $V$ on $\mathcal{D}_l$
2. Annotate advantages $A \leftarrow V(\mathcal{D}_l)$
3. Train $\pi$ on $\mathcal{D}_l$ with advantage labels
4. For $i = 1, \ldots, N$:
   - $\mathcal{D}_l \leftarrow \mathcal{D}_l \cup \pi.\text{rollout}()$
   - $\mathcal{D}_{\text{succ}} \leftarrow \{(\tau, y) \in \mathcal{D}_l \mid y = \text{success}\}$
   - Train $V$ on $\mathcal{D}_{\text{succ}}$
   - Re-annotate advantages
   - Train $\pi$ with new $A$

**Test-time Best-of-N sampling**（Appendix E）：

基于 DEAS (Kim et al. 2026a)，用 IQL-style critic scoring chunks：
- In-chunk discount $\gamma_1 = 0.9$
- Chunk-level discount $\gamma_2 = 0.99$
- Expectile $\tau = 0.7$
- Sampling temperature $T \in [1.5, 2.0]$（sweet spot 1.5）

**关键发现**：BoN 对未收敛 policy（RECAP₁）有效（8.5→4.9 attempts），但对已收敛 policy（RECAP₂, RECAP₃）反而有害（+2.3, +2.2）。这和 LLM reasoning 中的 test-time scaling 经验一致——test-time sampling 是 exploration mechanism，well-converged policies 被 stochasticity 拖累。

参考：
- RECAP: https://arxiv.org/abs/2511.14759
- DEAS: https://arxiv.org/abs/2603.21341 (Kim et al. 2026a)
- IQL: https://arxiv.org/abs/2110.06178

---

## 5. Inference Optimization (Graph + Kernel)

这是个非常实用的 engineering contribution。

### 5.1 Graph Capture Optimization

**问题**：PyTorch eager 每次单独 launch kernels，累积 launch overhead。Torch Compile 部分缓解但无法 fully eliminate，因为 graph fragmentation——某些 RoPE 和 attention mask 依赖 runtime configuration。

**Solution**：static graph conversion——预计算所有 configuration-dependent tensors（RoPE、attention mask），整个 forward pass 作为 single CUDA Graph capture，per-step 只 launch 一次。

### 5.2 Kernel Optimization

**Short-prefill workload 特性**：
- VLM backbone + MSAT 都是 short sequence（相对 LLM prefill 而言）
- Compute-bound matmuls 交替 memory-bound operators（RMSNorm, RoPE, residual updates）
- Torch Compile 的 graph-driven fusion 错失 cross-operator fusion patterns

**Custom fused kernels**（Table 7）：

| Kernel | Fused Operations |
|--------|------------------|
| `fused_vision_attention` | RoPE(q) + RoPE(k) + Attn(q',k',v) |
| `fused_llm_attention` | RoPE(RMSNorm(q)) + RoPE(RMSNorm(k)) + Attn |
| `fused_add2_layernorm` | h_out + h_in + LayerNorm |
| `fused_add2_rmsnorm` | h_out + h_in + RMSNorm |
| `fused_add3_rmsnorm` | h_out + h_in + h_ds + RMSNorm（DeepStack 三路 residual） |
| `fused_memory_attention` | RoPE + Attn for memory module |
| `grouped_swiglu` | 两个 SwiGLU 并行计算（C-A stream 共享） |
| `fused_mlp_swiglu` | 单 SwiGLU |

**直觉 build**：核心是减少 HBM round-trips。未融合时每个 operator 都要 write/read global memory，fused kernel 让 intermediate tensors 留 on-chip（SRAM），data movement 和 computation 在单 kernel 内协调。

### Latency Results (Table 4)

| Inference Stack | w/o physics & memory | All-modality |
|----------------|---------------------|--------------|
| PyTorch Eager | 67.0 ms | 71.2 ms |
| CUDA Graph + Torch.Compile | 56.9 ms (1.18×) | 59.6 ms (1.19×) |
| + Static Graph Conversion | 46.2 ms (1.45×) | 48.9 ms (1.46×) |
| + Kernel Optimization | **41.6 ms (1.61×)** | **43.7 ms (1.63×)** |

RTX 5090 上从 71.2ms 降到 43.7ms，1.63× speedup。值得注意的是 speedup ratio 跨两个 variant 几乎相同——说明 optimization 对 memory module 和 physics stream 是 robust 的。

参考：
- Torch Compile: https://arxiv.org/abs/2402.05505
- Trinity (Park et al. 2026): tensor optimization
- Real-Time Chunking (RTC): https://arxiv.org/abs/2504.05406 (Black et al. 2025c)

---

## 6. Evaluation 实验数据全面分析

### 6.1 Simulation Benchmarks (Table 1)

| Method | LIBERO Short | LIBERO Long | LIBERO Avg | LIBERO-Plus | SIMPLER Google-VM | SIMPLER Google-VA | WidowX |
|--------|--------------|-------------|------------|-------------|---------------------|---------------------|--------|
| π₀-FAST | 93.9 | 60.2 | 85.5 | 64.2 | 61.9 | 59.0 | 48.3 |
| π₀ | 97.1 | 85.2 | 94.1 | 54.6 | 58.8 | 54.8 | 27.1 |
| π₀.5 | 98.0 | 92.0 | 96.9 | 86.5 | 72.7 | 68.4 | 46.9 |
| GR00T N1.5 | 90.0 | 76.0 | 86.5 | 66.3 | 52.4 | 43.7 | 62.0 |
| GR00T N1.6 | 97.4 | 94.4 | 96.7 | 72.6 | 76.1 | 57.1 | 57.1 |
| **RLDX-1** | **98.6** | **95.3** | **97.8** | **86.7** | **81.5** | **77.4** | **71.9** |

**Challenging benchmarks**：

| Method | RoboCasa Kitchen | GR-1 Tabletop | RoboCasa365 Avg |
|--------|------------------|---------------|------------------|
| π₀.5 | 62.1 | 15.4 | 16.9 |
| GR00T N1.6 | 66.2 | 47.6 | 26.9 |
| **RLDX-1** | **70.6** | **58.7** | **32.1** |

关键观察：
- LIBERO 上差距小（97.8 vs 96.9），因为 saturated
- RoboCasa365 composite tasks 差距大（19.0% vs 12.6% seen，5.6% vs 2.6% unseen）—— long-horizon compositionality 是 RLDX-1 的优势区
- GR-1 Tabletop（humanoid）58.7% vs GR00T N1.6 的 47.6%——humanoid 优势明显

### 6.2 OpenArm Humanoid (Figure 14)

| Task | π₀.5 | GR00T N1.6 | RLDX-1 |
|------|-------|------------|--------|
| Basic PnP | 41.7 | 37.5 | **50.0** |
| Directional PnP (Shelf) | 37.5 | 41.7 | **54.2** |
| Directional PnP (Dish Rack) | 41.7 | 33.3 | **54.2** |
| Unseen Object | 37.5 | 41.7 | **54.2** |
| Unseen Task | 45.8 | 41.7 | **54.2** |
| Object Grounding | 45.8 | 33.3 | **87.5** |

Object Grounding 上 RLDX-1 87.5%，GR00T N1.6 只有 33.3%（= random）。这非常说明问题——GR00T N1.6 能识别 category 但 instance-level grounding 失败。

### 6.3 ALLEX Humanoid (Figure 16) —— Functional Capability Showcase

| Task | π₀.5 | GR00T N1.6 | RLDX-1 |
|------|-------|------------|--------|
| Conveyor PnP (S1 seen) | 25.0 | 50.0 | **100** |
| Conveyor PnP (S2 unseen) | 0 | 50.0 | **75** |
| Conveyor PnP (S3 unseen) | 50.0 | 0 | **75** |
| Conveyor PnP (S4 seen) | 41.7 | 50.0 | **100** |
| Object-in-Box Selection | 33.3 | 29.2 | **91.7** |
| Card Slide-and-Pick (progress) | ~40 | ~50 | **97.2** |
| Pot-to-Cup Pouring (progress) | ~30 | ~30 | **70.8** |

这组数据最 impactful。Conveyor PnP 上 π₀.5 和 GR00T N1.6 都"塌"到固定速度——baselines 缺乏 motion awareness，只能 memorize seen speed。RLDX-1 在 unseen speed 上还有 75%——说明 motion module 真的学会了 motion interpolation。

Object-in-Box Selection：baselines 30% 左右（接近 random），RLDX-1 91.7%。memory module 直接决定了 sequential reasoning。

### 6.4 Franka Research 3 (Figure 18)

| Task | π₀.5 | GR00T N1.6 | RLDX-1 |
|------|-------|------------|--------|
| Spin Tracking | 32.3 | 26.0 | **97.9** |
| Pong Game | 33.3 | 33.3 | **81.5** |
| Cup Swapping | 25.0 | 16.7 | **45.8** |
| Shell Game | 50.0 | 50.0 | **91.7** |
| Plug Insertion | 20.8 | 16.7 | **33.3** |
| Egg PnP | 45.8 | 37.5 | **61.1** |

Shell Game 91.7% vs baselines 50%——这非常说明问题，memory module 让 model 能"记住"哪个 cup 下面有 cube。

### 6.5 RL Ablation: Light Bulb Twisting (Figure 21)

| Method | Episode Length (frames) | Attempts |
|--------|-------------------------|----------|
| Teleop (human) | ~700 | ~6 |
| BC (RLDX-1 IL only) | 1056 ± 326 | 12.7 ± 3.0 |
| RECAP₁ | ~700 | ~8.5 |
| RECAP₂ | ~500 | ~5 |
| **RECAP₃** | **353 ± 22** | **4.1 ± 0.3** |

RECAP₃ 比 human teleop 还好（更少 frames + 更少 attempts + 更小 std）——RL refinement 超越了 demonstration 上限。

---

## 7. Additional Insights

### 7.1 PEFT Evaluation (Table 6, Appendix D)

- Full FT: 62.67%, 2.38B trainable, 87 GiB VRAM (batch 32)
- Backbone LoRA r=64 + Action LoRA r=64: 55.33%, 397M trainable (5.72%), 35.93 GiB (batch 32), **23.71 GiB (batch 1)**
- Frozen backbone + Action LoRA r=64: 36.42% —— 26.25 point gap

关键 takeaway：backbone updates 是 essential 的（下游分布与 pre-training 差异大时），但用 LoRA on top-4 + action 可以在 24GB 消费级 GPU 上 fine-tune。

### 7.2 Synthetic Data Scaling (Table 3)

| Pre-training Data | Success Rate |
|-------------------|--------------|
| Real only | 41.0% |
| Real + 25% synth | 45.6% |
| Real + 50% synth | 46.6% |
| Real + 100% synth | **50.1%** |

合成数据 scaling 趋势明显——加得越多效果越好。9.1% 的绝对提升（GR-1 Tabletop）证明了 synthetic pipeline 的有效性。

### 7.3 Batch Size Effect (Table 10)

| Batch Size | LIBERO | RoboCasa Kitchen | GR-1 Tabletop |
|------------|--------|------------------|----------------|
| 64 | 97.4 | 66.9 | 36.8 |
| 256 | 97.8 | 69.6 | 53.2 |
| 1024 | - | 70.6 | **58.7** |

GR-1 Tabletop 从 batch 64 到 1024 提升了 22%! 这表明 humanoid manipulation 对 batch size 极度敏感——可能因为 action space 高维（48-DoF），需要更大 batch 才能覆盖 distribution。

---

## 8. 相关工作联想

### 8.1 Architecture Lineage

- **MM-DiT → MSAT**: Stable Diffusion 3 的双 stream（context vs subject）扩展到 robotics——cognition vs action，再加 physics stream
- **π₀ → RLDX-1**: flow-matching action expert + shared self-attention 的设计，扩展到 multi-stream
- **CogACT (Li et al. 2024a)**: cognition tokens 的灵感来源
- **ContextVLA (Jang et al. 2025a)**: temporal compression 思路
- **HAMLET (Koo et al. 2026)**: memory module 设计
- **STSS (Kwon et al. 2021)**: space-time self-similarity for motion

### 8.2 Synthetic Data

- **VPT (Baker et al. 2022)**: IDM 范式起源（Minecraft）
- **DreamGen (Jang et al. 2025b)**: video world models → robot data
- **GigaWorld (2025)**: world models as data engine
- **RoboCurate (Kim et al. 2026e)**: motion-consistency filtering 思路

### 8.3 RL for VLA

- **RECAP (Amin et al. 2025)**: decoupled critic + policy
- **DEAS (Kim et al. 2026a)**: chunk-level critic + IQL
- **Robot-R1 (Kim et al. 2025)**: RL for embodied reasoning

### 8.4 Inference Optimization

- **PyTorch 2 / Torch Compile (Ansel et al. 2024)**: baseline
- **Trinity (Park et al. 2026)**: tensor program optimization via equality saturation
- **RTC (Black et al. 2025c)**: real-time chunking for action policies

### 8.5 Failure Modes of Baselines

论文有个很好的 qualitative analysis：
- **π₀.5**: 在 unseen settings 上 degrade 显著，frequent stuck——likely 由于 weaker VLM backbone (PaliGemma 3B) + full VLM fine-tuning → overfitting
- **GR00T N1.6**: 识别 object categories 但 instance-level grounding 弱（Object Grounding 33.3% = random）

---

## 9. 我的几个思考

### 9.1 关于 "versatility is not enough" 的论点

这其实是一个相当深的观察。过去两年 VLA 领域的 race 主要在 versatility——更多 data、更大 VLM、更广 embodiment coverage。但 RLDX-1 指出真实 manipulation 还需要 functional capabilities：temporal reasoning、memory、physical sensing。这暗示了下一阶段 VLA 研究的方向——不再是"看更多"，而是"感知更深"。

### 9.2 Multi-stream vs Cross-attention

GR00T N1.5/N1.6 用 cross-attention 把 VLM hidden states 注入 action model。RLDX-1 用 joint self-attention（MSAT）。后者信息交换更 symmetric，前者是单向。哪种更好？论文数据支持 MSAT，但 ablation 没直接比较。

### 9.3 RL 的边界

RECAP₃ 超越 human teleop 是个 milestone。但 RL 只在 Light Bulb Twisting 上验证，其他任务仍是 IL。RL 的适用边界还需要更多探索。BoN 对 converged policy 反而有害——这是个重要的 negative result。

### 9.4 Memory module 的位置

Memory module 放在 VLM 后（post-VLM）。另一种选择是放在 VLM 内（如 MemoryVLA Shi et al. 2026）。前者模块化好，后者可能更 tight integration。RLDX-1 的选择更工程友好。

### 9.5 Synthetic data 的 torque paradox

论文提到一个有趣观察：synthetic ALLEX data 不含 torque signals，但与 torque-annotated real data 联合训练仍 improve performance。这暗示 model 学到了"video motion → implicit torque prior"的映射。这是个 promising direction。

### 9.6 与你的 teaching 直觉呼应

你在 CS231n 和更近的 work 中强调 "understand the data, understand the loss, understand the architecture"。RLDX-1 是个很好的 case study：
- **Data**: 三源（public/in-house/synthetic）+ motion-consistency filtering
- **Loss**: flow-matching velocity prediction + auxiliary physical signal prediction
- **Architecture**: MSAT 让 heterogeneous modalities 各司其职

三个维度的设计都服务于"functional capability"这个 north star。

---

## 10. 总结

RLDX-1 是 2026 年 robotics VLA 领域的一个重要 milestone。它的核心贡献：

1. **Conceptual**: 把 VLA 的能力维度从 versatility 扩展到 versatility + functional capabilities (motion/memory/physics)
2. **Architectural**: MSAT 用 multi-stream joint attention 统一 heterogeneous modalities
3. **Data**: synthetic pipeline with motion-consistency filtering 让稀缺场景可 scale
4. **Training**: three-stage pipeline（pre → mid → post）+ RECAP RL refinement
5. **Systems**: graph + kernel optimization 把 latency 从 71ms 压到 44ms
6. **Empirical**: ALLEX humanoid 上 ~90% success rate vs baselines ~40%

如果让我给这个 paper 提一个 critical question：**这些 functional capabilities 是否一定要 architectural 解决**？一个 alternative hypothesis 是——足够大的 VLM + 足够多 data 可能"涌现"出 motion awareness 和 memory。但论文的 ablation 数据（尤其 Conveyor PnP 在 unseen speed 上的 75%）说明 architectural inductive bias 仍然 matter。这正是 RLDX-1 价值所在。

希望这个 deep dive 帮你 build 了 intuition, Andrej! 如果你需要我深入某个具体 module（比如 STSS 的实现细节、MSAT 的 attention pattern、或 RECAP 的 critic training），告诉我。
