---
source_pdf: Fast-WAM Do World Action Models Need - Copy.pdf
paper_sha256: 4eea24883dcc4d8a5c0f760870f501baabb07db4eb65e5fd6c3b4b500601be8d
processed_at: '2026-08-04T07:25:34-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Fast-WAM

## 一句话版本

之前搞 robot 的人都说"我先把未来想象成一段视频，再根据想象来决定动作"，听起来很 fancy，但跑得慢。这篇 paper 发现：**你训练的时候让模型学怎么预测未来，这个 training process 才是真正起作用的；inference 的时候显式把未来生成出来其实没必要**。所以他们就搞了个 Fast-WAM：训练时保留"学预测未来"这个辅助任务，inference 时不生成视频，直接出动作，速度快 4 倍多，效果还差不多。

---

## 这 paper 在质疑啥

### 先说背景

现在做 robot policy 大致分两派：

**VLA 派**（Vision-Language-Action，像 $\pi_0$、OpenVLA 这些）：直接 $(图像, 语言指令) \to 动作$。简单粗暴，但只用图像文本预训练，不懂物理。

**WAM 派**（World Action Model，像 Motus、LingBot-VA 这些）：先 $(图像, 指令) \to 想象出未来视频 \to (图像, 指令, 想象的视频) \to 动作$。听起来更合理，因为模型"知道接下来会发生什么"。

WAM 派的 selling point 是：模型先 imagine 一下"如果我现在这样抓，杯子会怎么动"，然后再根据想象决定怎么动手。直觉上这比 VLA 那些"瞎抓"的高明多了。

但是！想象未来要花时间。video diffusion 要迭代去噪几十步，推理慢得要命，800ms 起步。Real-time robot control 通常要 10Hz 以上，800ms 等于 1.25Hz，根本没法用。

### 灵魂拷问

作者就问了一个特别 simple 的问题：

**这"想象未来"到底是真的有用，还是个 placebo？**

你想啊，"想象未来"这件事其实有两个作用混在一起：

1. **训练时学预测未来** → 让模型 weights 里装了物理 prior，知道"杯子掉下去会碎"、"手往前推东西会动"
2. **inference 时显式生成未来** → 给 action prediction 一个具体的 foresight signal

之前所有 WAM 都把这俩捆在一起做，谁也说不清到底是 (1) 还是 (2) 在起作用。

这就像你学游泳时教练让你每次下水前先在岸上比划动作 10 遍。之后你游泳游得好——但到底是因为岸上比划训练了你的肌肉记忆，还是因为你每次游泳前真的都在岸上比划 10 遍？这是个 confounder。

作者的 hypothesis：**真正有用的是 (1)，(2) 可能就是个 placebo**。如果这个成立，那 (2) 这个慢推理就可以直接砍掉。

---

## Fast-WAM 怎么做的

### 核心思路

训练时学预测未来，inference 时不预测未来。

具体讲：

**训练时**：模型有两个 branch 共享 attention：
- Video branch（5B 参数）：拿当前帧 + 未来帧去 denoise，学怎么预测未来视频
- Action branch（1B 参数）：拿当前帧去 denoise 出 action chunk

两个 branch 通过 shared attention 互通有无。Action branch 在训练时一直看 video branch 怎么"想"未来，潜移默化就把物理 prior 吸收进来了。这就像武术里师父练招式给徒弟看，徒弟看着看着自己也就会了。

**inference 时**：
- 把 future video branch 整个砍掉，不要了
- 只让 video branch 对当前帧做一次 forward（不 denoise，不生成视频）
- 拿到的 latent representation 直接喂给 action branch
- Action branch 10 步去噪出 action chunk

总共 190ms。对比那些 imagine-then-execute 的 WAM 要 800ms，快 4 倍多。

### 数学讲讲

标准 WAM 的 factorization（公式 2）：

$$p(a_{1:H} \mid o, l) = \int p(v_{1:T} \mid o, l) \, p(a_{1:H} \mid o, l, v_{1:T}) \, dv_{1:T}$$

翻译成人话：动作的条件概率 = 对"未来视频"做 marginalization。先想象所有可能的未来视频 $v_{1:T}$（每个有概率 $p(v|o,l)$），再对每个想象出来的未来算动作概率 $p(a|o,l,v)$，最后加权平均。

- $o$：当前 observation（图像）
- $l$：language instruction
- $a_{1:H}$：要预测的 action chunk，长度 $H=32$
- $v_{1:T}$：未来 $T=9$ 帧视频（4× temporal downsample 后）
- $p(v_{1:T}|o,l)$：video prediction prior
- $p(a_{1:H}|o,l,v_{1:T})$：condition 在想象出来的未来上的 action distribution

实际中没人真的积分，就 sample 一个未来视频然后 condition 在它上面。

Fast-WAM 的 formulation（公式 4）：

$$p_\theta(a_{1:H} \mid o, l) = p_\theta(a_{1:H} \mid z(o, l))$$

其中 $z(o,l)$ 是 video backbone 单次 forward 出来的 latent representation。**没有显式的 $v_{1:T}$ 这个中间变量了**。模型直接从 $z$ 出动作。

Training loss（公式 9）：

$$\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{vid}}$$

- $\mathcal{L}_{\text{act}}$：action chunk 的 flow matching loss
- $\mathcal{L}_{\text{vid}}$：future video latents 的 flow matching loss（co-training 信号）
- $\lambda$：balance coefficient（paper 没说具体值）

Flow matching loss（公式 6）：

$$\mathcal{L}_{\text{FM}}(y) = \mathbb{E}_{y, \epsilon, t} \left[ \| f_\theta(y_t, t, o, l) - (\epsilon - y) \|_2^2 \right]$$

- $y$：target（action 或 video latent）
- $\epsilon \sim \mathcal{N}(0, I)$：Gaussian noise
- $t \in (0,1)$：flow time，$t=0$ 是 clean data，$t=1$ 是纯 noise
- $y_t = (1-t)y + t\epsilon$：interpolated state
- $f_\theta$：模型预测的 velocity field
- $\epsilon - y$：target velocity（从 data 走到 noise 的常速度）

训练时模型学预测这个 velocity，inference 时反过来从 noise 走到 data。

---

## 关键设计：Attention Mask

这是最 subtle 的地方。Token 分三组：

| Token 组 | 训练 | 推理 | 能看谁 |
|---|---|---|---|
| Clean first frame（当前帧 latent）| ✓ | ✓ | 谁都不看，但别人能看它 |
| Noisy future video tokens | ✓ | ✗ | 内部双向 + 看 first frame |
| Action tokens | ✓ | ✓（noisy） | 内部双向 + 看 first frame，**绝对不能看 future video** |

**Action tokens 不能看 future video tokens**——这条是关键。

为什么？如果 action 能看 future video，模型会偷懒：直接从 noisy future 里 decode 出"未来该做什么动作"，shortcut 掉 representation learning。训练时会 cheating，inference 时又没有 future video，distribution shift 直接崩。

所以 mask 设计成：video branch 自己学预测未来，action branch 通过 shared attention 间接吸收 video branch 的 hidden state 信息（同一层内 attention 互通），但没法直接看 future tokens。

直觉上：让 action branch "听到" video branch 在"思考"什么，但看不到 video branch 的"答案"。

---

## 实验结果讲人话

### RoboTwin 2.0（双臂 50+ 任务）

| 方法 | 用没用 embodied 预训练 | 成功率 |
|---|:---:|:---:|
| $\pi_0$ | ✓ | 62.2% |
| $\pi_{0.5}$ | ✓ | 79.8% |
| Motus | ✓ | 87.8% |
| LingBot-VA | ✓ | 92.2% |
| LingBot-VA（无预训练）| ✗ | 80.6% |
| **Fast-WAM（无预训练）** | ✗ | **91.8%** |

Fast-WAM 不用任何 robot data 预训练，直接打平用了大规模预训练的 LingBot-VA。说明 video co-training 这个 auxiliary 信号本身就足够 strong。

### 关键 ablation：到底哪个 factor 重要？

| 变体 | 训练时学预测未来 | 推理时想象未来 | RoboTwin | LIBERO |
|---|:---:|:---:|:---:|:---:|
| Fast-WAM | ✓ | ✗ | 91.8% | 97.6% |
| Fast-WAM-Joint | ✓ | ✓（联合去噪）| 90.6% | 98.5% |
| Fast-WAM-IDM | ✓ | ✓（先视频后动作）| 91.3% | 98.0% |
| Fast-WAM 无 co-train | ✗ | ✗ | **83.8%** | **93.5%** |

人话解读：

1. **保留 co-training 但砍掉 inference 想象**（Fast-WAM）vs **保留 co-training 也保留 inference 想象**（Joint/IDM）：差距 1% 左右，几乎一样。
2. **保留 co-training**（Fast-WAM 91.8%）vs **砍掉 co-training**（83.8%）：差 8 个百分点。

这个对比直接说明：**真正起作用的是 co-training 那个训练信号，inference 想象那个东西可有可无**。

### 真实世界毛巾折叠：差距更夸张

| 变体 | 成功率 | 完成时间 | 延迟 |
|---|:---:|:---:|:---:|
| $\pi_{0.5}$（有预训练）| 最高 | 最短 | - |
| **Fast-WAM** | 高 | 较短 | **190ms** |
| Fast-WAM-Joint | 高 | 较长 | ~400ms |
| Fast-WAM-IDM | 最高 | 较长 | **810ms** |
| Fast-WAM 无 co-train | **10%** | 最长 | 190ms |

人话：
- **去掉 co-training，真实世界直接崩到 10%**（sim 上还有 83.8%）。这是 80% 的 absolute drop，远比 sim 上的差距大。
- **延迟上 190ms vs 810ms**——4.3× 加速，对 closed-loop control 是质的差别（5Hz vs 1.2Hz）

这个 sim-to-real 的 gap 放大效应很有意思：在 sim 上"没用想象未来"只掉一点点，到了 real-world 上"没用 co-training"就崩盘了。说明 real-world 的 dynamics 比 sim 复杂得多，没有 strong physics prior 根本没法泛化。

---

## 为什么想象未来没用？我的几个猜测

### 1. 信息已经在 weights 里了

Video DiT（Wan2.2-5B）本来就在海量 web 视频上预训练过，已经知道"杯子掉下来会怎样"、"手推东西会动"。再通过 robot data 的 video co-training fine-tune，这些 physics priors 全都 encode 在 weights 里了。

Inference 时的"想象未来"只是把这个 in-weights knowledge decode 出来。但 action branch 完全可以通过 shared attention 直接从 video branch 的 hidden states 里读这些 priors，不需要 decode 到 pixel space 再 encode 回来。

类比：你脑子里已经知道"开水烫别碰"，不需要每次都 verbally 默念这句话再决定不碰——你的 motor reflex 直接就避开了。

### 2. 生成的未来是 stochastic 的，反而 noisy

Video diffusion 是 stochastic 的——同样的 context 可以 sample 出不同的未来轨迹。Action condition 在某个 specific sample 上，如果这个 sample 不 representative，action 就偏。

而 latent representation 是 deterministic 的（单次 forward），相当于浓缩了所有 plausible futures 的 mixture。**显式生成单一未来 vs 隐式编码未来分布**——后者更鲁棒。

### 3. Short horizon 上想象力用武之地有限

Paper 用 $H=32$ 的 action chunk，大概 1-2 秒动作。短 horizon 上 current observation + learned priors 已经够用，没必要显式 lookahead。

真正需要想象力的场景：long-horizon planning，比如"做顿早饭"这种需要 lookahead 10 步以上的任务。但 paper 没测这个 setting，所以这个 finding 的 generalizability 还有边界。

### 4. 跟 LLM reasoning 的有意思的类比

| | LLM | Robot Policy |
|---|---|---|
| 显式推理 | Chain-of-Thought（每 token 都生成出来）| Imagine-then-execute（每帧都生成出来）|
| 隐式推理 | Internal reasoning（hidden states 里演化）| Fast-WAM（latent forward pass）|

DeepSeek R1 这些 work 也发现：distilled direct-answer 模型在很多 task 上接近 CoT 模型，但 hard reasoning task 上 CoT 仍有优势。Fast-WAM 的 finding 完全 parallel：**简单 task 隐式够用，复杂 task 可能还需要显式想象**。

### 5. 跟 model-based RL 经典辩论的呼应

Model-based RL 一直有个争论：world model 是用来 imagination 做 planning（Dreamer 那派），还是只用来 shape representation（TD-MPC 那派）？

- **Dreamer**：latent imagination + actor-critic，显式 rollout imagined trajectories
- **TD-MPC2**：latent dynamics model 主要做 representation，policy 不显式 rollout
- **Fast-WAM**：在 foundation model 时代重新投票，站了 TD-MPC 那派

这其实是个 philosophical debate：**世界模型是 simulator（用来 imagine）还是 encoder（用来 represent）？** Fast-WAM 的实验证据说：在 robot foundation model 这个 regime，encoder 解读比 simulator 模拟更值钱。

---

## 一些我想吐槽的细节

### 1. 没给 $\lambda$ 的具体值

公式 9 里 $\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{vid}}$，$\lambda$ 是多少？没说。这是个 critical hyperparameter，太大 action 信号被稀释，太小 co-training 没效果。一般这种 auxiliary loss 的 $\lambda$ 都要 sweep 一遍，paper 不 report 让复现困难。

### 2. Single chunk 的 simplification 太大

Paper 明说："for simplicity, we focus on single action chunk generation and omit the outer auto-regressive loop"。

但 long-horizon 任务（叠毛巾就是 long-horizon）实际执行时需要 outer loop——一个 chunk 执行完再看新 observation 生成下一个 chunk。这个 outer loop 里 imagination 可能重新变得 critical，但 paper 没测。

这是个 significant limitation，作者诚实地承认了。

### 3. Action expert 太小？

Action expert 用 $d_a = 1024$，1B 参数。Video branch 5B。这个 1:5 的比例是怎么选的？没消融。如果 action expert 太小，可能 underutilize video representations；太大可能 overfit。

### 4. Per-task 上有 outlier

仔细看 RoboTwin per-task 结果（Appendix Table 3），有些 task 上 Joint 变体明显崩：
- Open Microwave：Joint 只有 3%/14%，但 IDM 有 54%/53%，w/o co-train 34%/77%

这种 outlier 暗示 joint denoising 在某些 task 上可能不稳定（attention interference？），但 paper 没深究。

---

## 我对这 paper 的整体看法

**Positives**：
1. Question 问得好——disentangle 一个之前被 entangled 的 confounder，这种 controlled comparison 是 science 该有的样子
2. Method 简洁优雅——就是 attention mask + 砍 inference video branch，没引入复杂新机制
3. Real-world 实验 + sim-to-real gap 分析很有说服力
4. Engineering 上 190ms 这个数字真的能 deploy

**Concerns**：
1. Single chunk 设定下 imagination 没用，不能推论到 long-horizon 也是如此
2. Wan2.2-5B 这个 backbone 太强了，可能掩盖了某些效应
3. 没研究 stochastic / partial observation setting 下的 imagination 价值

**Big picture intuition**：

这 paper 真正在说的是：**在 foundation model 时代，"世界模型"的角色可能从 simulator 变成了 encoder**。

经典 world model（Dreamer 那派）的逻辑是：学一个准确的 dynamics model，然后在它里面 rollout、plan、imagine。但 foundation model 时代，video DiT 已经把整个世界的 dynamics 都 encode 在 weights 里了，你不需要再 rollout——你只需要 forward 一次把"我对当前世界的理解"提取出来就够了。

这跟 LLM 时代的趋势一致：以前我们用 knowledge graph 做推理，现在 GPT 把 knowledge 都装 weights 里直接 forward 出答案。

也许 robot policy 也是这条路：**不要 simulator，要 encoder；不要 imagine，要 internalize**。

至少在 short-horizon, single-chunk 这个 setting 上，Fast-WAM 给了一个干净的 evidence 支持 this view。

---

## 相关链接

主 paper：
- Fast-WAM 项目主页：https://yuantianyuan01.github.io/FastWAM/

基础模型和方法：
- Wan2.2 视频生成：https://arxiv.org/abs/2503.20314
- Flow Matching：https://arxiv.org/abs/2210.02747
- Rectified Flow：https://arxiv.org/abs/2209.03003
- Diffusion Policy：https://arxiv.org/abs/2303.04137

WAM 相关 baseline：
- Motus：https://arxiv.org/abs/2512.13030
- LingBot-VA（Causal World Modeling）：https://arxiv.org/abs/2601.21998
- Unified World Models：https://arxiv.org/abs/2504.02792
- Vidar：https://arxiv.org/abs/2507.12898
- NVIDIA WAM：https://arxiv.org/abs/2602.15922
- VPP：https://arxiv.org/abs/2412.14803
- UVA：https://arxiv.org/abs/2503.00200

VLA baselines：
- $\pi_0$：https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$：https://arxiv.org/abs/2504.16054
- OpenVLA：https://arxiv.org/abs/2406.09246

Benchmarks：
- LIBERO：https://arxiv.org/abs/2306.03310
- RoboTwin 2.0：https://arxiv.org/abs/2506.18088

Model-based RL 经典脉络：
- World Models（Ha & Schmidhuber）：https://arxiv.org/abs/1803.10122
- Dreamer：https://arxiv.org/abs/1912.01603
- TD-MPC2：https://arxiv.org/abs/2310.16828

Test-time compute scaling：
- Scaling Test-Time Compute：https://arxiv.org/abs/2408.03314
- DeepSeek R1：https://arxiv.org/abs/2501.12948

---

# Fast-WAM: Test-time Future Imagination 是必需的吗？

这篇文章问了一个非常 fundamental 的 question, 直击 World Action Models (WAMs) 这条 research line 的核心假设: **WAM 的 gain 到底来自哪里?** 是 training 时的 video prediction objective 塑造了更好的 representation, 还是 test-time 显式 generate 未来视频提供了 foresight? 这两个 factor 在 prior WAM 工作 [1-5] 中是 entangled 的, 这篇 paper 通过 controlled variants 把它们 disentangle 开来。

项目主页: https://yuantianyuan01.github.io/FastWAM/

---

## 1. 核心问题与 Motivation

### 1.1 现有 WAM 范式的 entanglement

主流的 WAM 工作 (如 NVIDIA 的 WAM paper [4], Motus [5], Unified World Models [6], Vidar [7]) 都遵循 **imagine-then-execute** 的范式, 大致可以归为两类 (Figure 1):

- **(A) Joint-modeling**: video tokens 和 action tokens 在同一个 diffusion model 里 jointly denoise, 通过 shared attention 让 action generation 与 future video modeling 耦合 (Motus [5], Unified World Models [6], GR-2 [24])
- **(B) Causal / Video-then-Action**: 先 generate future video, 再 condition on 生成的未来做 action prediction (Vidar [7], LingBot-VA [3], SuSIE [8])

这两种范式中, 同一个 model 既要 learn 从 video prediction 中获取物理 prior (training), 又要在 inference 时显式 synthesize 未来观察。两个 mechanism 完全 tangled 在一起。

### 1.2 两个 factor 的 disentangle

作者把 WAM 的 effectiveness 拆成两个独立的 source:

1. **Training-time video co-training objective**: 通过 $\mathcal{L}_{\text{vid}}$ 让 video backbone 学习物理动力学、object interaction、scene evolution 的 latent representations。这个 signal 通过 shared attention 流入 action branch。
2. **Test-time explicit future imagination**: 在 inference 时显式 sample/denoise 未来 video frames, 给 action prediction 提供 foresight。

Fast-WAM 的核心 hypothesis: **factor (1) 是 dominant 的, factor (2) 可能没那么重要**。如果这个 hypothesis 成立, 就可以用一个 architecture 同时拿到 WAM 的 representation advantage 和 VLA 的 inference efficiency。

直觉上这跟近期 LLM reasoning 的某些 finding 呼应: training-time supervision 的 richness 可能远比 test-time computation 更重要。例如 R1-distilled models 即使在 inference 时不显式 CoT 也能在很多 task 上表现不错 [R1: https://arxiv.org/abs/2501.12948]; AlphaGo 的 policy network 单 forward pass 也很强, MCTS 只是 boost。

---

## 2. Fast-WAM Architecture 详解

### 2.1 整体架构: Mixture-of-Transformer (MoT)

Fast-WAM build on top of **Wan2.2-5B** [36] 的 video DiT (https://arxiv.org/abs/2503.20314), 这是阿里巴巴的大规模 text-to-video model。整体架构是 **Mixture-of-Transformer with shared attention**, 类似 modular transformer 的设计 (这种设计让人想到 MoE, 但 granularity 是 modality-level 而不是 token-level, 类似 Modular Diffusion [https://arxiv.org/abs/2405.15115] 的思路)。

```
                ┌─────────────────────────┐
   Language ───►│  T5 Text Encoder         │──► text embeddings
                └─────────────────────────┘         │
                                                    │ cross-attn
   First frame ─► VAE ──► clean latent tokens ◄─────┤
                            (anchor)                │
                                │                   │
                                ▼                   │
                ┌──────────────────────────┐        │
                │  Video DiT (5B params)   │◄───────┘
                │  (shared attention)      │
                └──────────────────────────┘
                            │
                            │ latent world features z(o,l)
                            ▼
                ┌──────────────────────────┐
                │ Action Expert DiT (1B)   │──► action chunk a_{1:H}
                │ d_a = 1024, h = 32       │
                └──────────────────────────┘
```

Model 总参数量约 6B (5B video DiT + 1B action expert)。

### 2.2 输入 tokens 的三组分类

模型 input tokens 分为三组, 这是这个架构最 critical 的设计:

| Token group | 训练时 | 推理时 | 作用 |
|---|---|---|---|
| Clean latent tokens of **first frame** | 存在 | 存在 | Shared visual anchor, 把 video branch 和 action branch 接到同一视觉 context |
| Noisy latent tokens of **future video frames** (T 帧, 4× temporal downsampling 后为 9 帧) | 存在 | **删除** | Video modeling 的 target |
| Action tokens (horizon H=32) | 存在 | 存在 (作为 noisy) | Action chunk 生成 |

### 2.3 Attention mask 设计 (Figure 2b)

这个 mask 是整个 architecture 最微妙的部分, 用于 disentangle training 时的信息流:

```
                  Clean    Future video    Action
                  first    (noisy)         tokens
                  frame    tokens
Clean first frame  ◯       ✗               ✗        ← 不 attend 任何 token (anchor, 被别人看)
Future video       ✓       ✓               ✗        ← 内部 bidirectional, 可看 anchor
Action             ✓       ✗               ✓        ← 内部 bidirectional, 可看 anchor, **不可看 future video**
```

关键点: **Action tokens 不能 attend 到 future video tokens**。这防止了 future 信息 leak 进 action branch, 避免 "shortcut" — 即模型学到直接从 noisy future tokens 偷取未来信息来预测 action。如果允许这种 leak, action prediction 会被 future video 的 ground truth 污染, 训练时会过拟合, 也会破坏 disentangle 实验。

但 future video tokens 可以看 first frame anchor, action tokens 也可以看 first frame anchor — 通过 anchor 实现 video branch 和 action branch 的"软耦合"。这种 design pattern 在 multi-task diffusion 中也常见, 例如 [Chameleon-style MoE architectures]。

### 2.4 Inference 时: 单 forward pass 编码

Test time 的关键操作:

1. **Encode first frame**: 单帧图像通过 VAE encoder 得到 clean latent tokens $z_0$
2. **Video DiT 单次 forward**: 把 $z_0$ + language embeddings 通过 video DiT 一次 (no denoising, no iter), 输出 latent world representation $z(o, l) = f_{\theta_{\text{video}}}(z_0, o, l)$
3. **Action denoising**: action expert 用 $z(o, l)$ 作为 condition, 10 步 flow matching denoise 得到 action chunk $a_{1:H}$

```
Inference cost 分解:
  - VAE encode:            ~10 ms
  - Video DiT forward × 1:  ~80 ms  (5B model, single pass)
  - Action expert × 10:     ~100 ms (1B model, 10 denoise steps)
  ─────────────────────────────────
  Total:                    ~190 ms
```

对比 imagine-then-execute variants:
- **Fast-WAM-Joint**: video + action joint denoise × 10 steps. Video tokens 数量是 action tokens 的 ~9 倍, 每步 cost 高。~400 ms
- **Fast-WAM-IDM**: 先 video denoise × 10 (~700 ms) 再 action denoise × 10 (~100 ms). ~810 ms

---

## 3. 数学 Formulation 详解

### 3.1 Imagine-then-execute WAM 的 factorization

标准 WAM (公式 2) 把 action distribution 分解为 marginalize over future observations:

$$
p(a_{1:H} \mid o, l) = \int p(v_{1:T} \mid o, l) \, p(a_{1:H} \mid o, l, v_{1:T}) \, dv_{1:T}
$$

变量解释:
- $o$: 当前 observation (image, 可能多 camera concatenate)
- $l$: language instruction
- $a_{1:H}$: action chunk, horizon $H = 32$
- $v_{1:T}$: future visual observations, $T = 9$ frames after 4× temporal downsample
- $p(v_{1:T} \mid o, l)$: video prediction prior (imagine)
- $p(a_{1:H} \mid o, l, v_{1:T})$: inverse dynamics / action conditioned on imagined future

这个 marginalization 在实践中通常用 single sample 近似, 即生成一段 future video, 然后 condition on 它。

### 3.2 Fast-WAM 的 direct policy interface (公式 3-4)

$$
p_\theta(a_{1:H} \mid o, l) = p_\theta(a_{1:H} \mid z(o, l))
$$

其中 $z(o, l)$ 是 video backbone 在 current context 下产生的 latent world representation, 通过 **single forward encoding pass** 获得 — 没有显式 sampling $v_{1:T}$, 没有 iterative denoising。

这个 formulation 跟 VLA 形式上一致 (都是 direct mapping $o, l \to a$), 但 representation 的来源不同:
- VLA: image-text pretrained backbone (如 CLIP, SigLIP) 的 semantic features
- Fast-WAM: video DiT 经过 video co-training 后的 "physics-aware" latent features

### 3.3 Flow matching objective (公式 5-9)

训练用 **flow matching** (Lipman et al. 2023, https://arxiv.org/abs/2210.02747; Liu et al. rectified flow, https://arxiv.org/abs/2209.03003), 跟 Wan2.2 一致。

Interpolated sample (公式 5):

$$
y_t = (1-t) \, y + t \, \epsilon
$$

变量含义:
- $y$: target variable, 可以是 action chunk $a_{1:H}$ 或 future video latents $z_{1:T}$
- $\epsilon \sim \mathcal{N}(0, I)$: 标准 Gaussian noise
- $t \in (0, 1)$: flow time, $t=0$ 是 data (clean), $t=1$ 是 pure noise
- $y_t$: 在 $t$ 时刻的 interpolated state (沿直线从 $y$ 走到 $\epsilon$)

Flow matching loss (公式 6):

$$
\mathcal{L}_{\text{FM}}(y) = \mathbb{E}_{y, \epsilon, t} \left[ \| f_\theta(y_t, t, o, l) - (\epsilon - y) \|_2^2 \right]
$$

变量含义:
- $f_\theta(y_t, t, o, l)$: 模型预测的 velocity field, 给定 noisy state $y_t$、time $t$、observation $o$、language $l$, 输出对 velocity 的估计
- $\epsilon - y$: target velocity, 即从 $y$ 走到 $\epsilon$ 的常速度
- $o, l$ 作为 conditioning 通过 cross-attention 注入

训练时模型学习预测从 noise 走到 data 的反向 velocity (inference 时反过来走)。

总体 loss (公式 9):

$$
\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda \, \mathcal{L}_{\text{vid}}
$$

其中:
- $\mathcal{L}_{\text{act}} = \mathcal{L}_{\text{FM}}(a_{1:H})$: action prediction loss
- $\mathcal{L}_{\text{vid}} = \mathcal{L}_{\text{FM}}(z_{1:T})$: video co-training loss, $z_{1:T}$ 是 VAE 编码 future frames 得到的 latents
- $\lambda$: balance coefficient, paper 没明说具体数值

Logit-normal 分布用于采样 $t$, 跟 Wan2.2 一致 (倾向于采样中间 $t$ values, 而不是均匀)。

---

## 4. Controlled Variants 设计

为了 disentangle 两个 factor, paper 实现了三个 variants 跟 Fast-WAM 对比:

| Variant | Training video co-train | Test-time future imagination | 对应已有 paradigm |
|---|:---:|:---:|---|
| **Fast-WAM** | ✓ | ✗ | 本文新提出 |
| **Fast-WAM-Joint** | ✓ | ✓ (joint denoise) | Motus [5], Unified World Models [6], GR-2 [24] |
| **Fast-WAM-IDM** | ✓ | ✓ (video → action) | Vidar [7], LingBot-VA [3], SuSIE [8] |
| **Fast-WAM w/o video co-train** | ✗ | ✗ | Pure VLA-like baseline |

实现细节:
- **Fast-WAM-Joint**: 训练和 inference 时都允许 attention between video and action tokens (跨模态双向 attention), 让 action denoise 能看到 video denoise 的中间状态
- **Fast-WAM-IDM**: 训练时按 [3] 做法, 以 $p=0.5$ 概率给 ground-truth video tokens 加 noise augmentation (模拟 inference 时生成的 video 不完美)。Inference 时先 denoise video, 再用生成的 video latents 作为 condition 给 action expert
- **Fast-WAM w/o co-train**: 只保留 $\mathcal{L}_{\text{act}}$, 去掉 $\mathcal{L}_{\text{vid}}$, 但 architecture 和 inference 完全保持

这四个 variants 共享 backbone (Wan2.2-5B)、tokenizer、训练 recipe — 唯一变化的就是 video co-training 的有/无 和 inference 时 imagination 的有/无。这就是 controlled comparison 的精髓。

---

## 5. 实验数据深度分析

### 5.1 RoboTwin 2.0 结果 (Table 1)

RoboTwin 2.0 (https://arxiv.org/abs/2506.18088) 是 bimanual manipulation benchmark, 50+ tasks, 有 clean 和 randomized scenes 两类评估。

| Method | Embodied PT | Clean | Rand. | Average |
|---|:---:|:---:|:---:|:---:|
| π0 [10] | ✓ | 65.92 | 58.40 | 62.2 |
| π0.5 [11] | ✓ | 82.74 | 76.76 | 79.8 |
| Motus [5] | ✓ | 88.66 | 87.02 | 87.8 |
| LingBot-VA [3] | ✓ | 92.90 | 91.50 | 92.2 |
| LingBot-VA (Wan2.2 backbone) | ✗ | 80.60 | 80.60 | 80.6 |
| **Fast-WAM (Ours)** | ✗ | **91.88** | **91.78** | **91.8** |
| Fast-WAM-Joint | ✗ | 90.84 | 90.32 | 90.6 |
| Fast-WAM-IDM | ✗ | 91.16 | 91.34 | 91.3 |
| Fast-WAM w/o co-train | ✗ | 82.76 | 84.80 | **83.8** |

关键观察:
1. **Fast-WAM 不用 embodied pretraining 即达到 91.8%**, 跟用 pretrained 的 LingBot-VA (92.2%) 几乎打平, 远超无 pretraining 的 LingBot-VA (80.6%)。这表明 video co-training 提供了 strong data efficiency, 即使没有大规模 robot pretraining 也能 competitive。
2. **三个有 video co-training 的 variants (91.8, 90.6, 91.3) 差距很小**, 都在 ~1% 范围内。这意味着 test-time imagination 带来的 marginal gain 微乎其微。
3. **去掉 video co-training 后掉到 83.8%**, 跟 Fast-WAM 差 ~8 个百分点。这个 gap 远大于 imagination variants 之间的 gap。

### 5.2 LIBERO 结果 (Table 2)

LIBERO (https://arxiv.org/abs/2306.03310) 四个 suites: Spatial, Object, Goal, Long。

| Method | PT | Spatial | Object | Goal | Long | Avg |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenVLA [9] | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 [10] | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 [11] | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA [3] | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus [5] | ✓ | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| **Fast-WAM (Ours)** | ✗ | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| Fast-WAM-Joint | ✗ | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| Fast-WAM-IDM | ✗ | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 |
| Fast-WAM w/o co-train | ✗ | **89.2** | 99.2 | 95.4 | **90.0** | **93.5** |

更细粒度的观察:

1. **Fast-WAM 在 Object 上拿到 100%**, 跟 LingBot-VA (99.6) 和 Motus (99.8) 几乎饱和。
2. **去掉 video co-training 后, Spatial (98.2 → 89.2, -9.0) 和 Long (95.2 → 90.0, -5.2) 掉得最多**。Spatial suite 主要 test spatial relation reasoning, Long test long-horizon multi-step tasks。Video co-training 在需要空间理解和长时程 planning 的 task 上 benefit 最大, 这跟我们直觉一致 — 物理 dynamics 主要体现在 spatial 位置变化和 long-horizon 累积效应上。
3. **Object 和 Goal 上 w/o co-train 只掉 ~0.8-1.6 个百分点**, 表明这些 task 不那么依赖 dynamics understanding, 纯 visual recognition 即可。

这个 pattern 强烈暗示: **video co-training 的价值在于 teaching physics, 而非 teaching visual recognition**。

### 5.3 Per-task RoboTwin 结果 (Appendix Table 3)

细看 task-by-task, 我注意到几个 informative 的 cases:

| Task | Fast-WAM (C/R) | Joint (C/R) | IDM (C/R) | w/o co-train (C/R) |
|---|---|---|---|---|
| **Hanging Mug** | 58/62 | 71/56 | 66/62 | 18/17 |
| **Open Microwave** | 62/45 | 3/14 | 54/53 | 34/77 |
| **Place Can Basket** | 71/69 | 50/23 | 37/28 | 72/72 |
| **Stamp Seal** | 90/94 | 96/99 | 99/94 | 60/78 |
| **Turn Switch** | 61/59 | 73/72 | 59/74 | 44/45 |
| **Move Pillbottle Pad** | 100/99 | 99/99 | 98/100 | 84/61 |

观察:
- **Hanging Mug** 是 universally hard (最高也只 71%), 但 w/o co-train 直接崩到 18% — 这是 deformable + 精确 placement 的 task, 物理动力学至关重要
- **Open Microwave** 异常: Joint variant 只有 3%/14%, 而 w/o co-train (77%) 反而高, π0.5 95% — 这暗示 joint denoising 在某些 task 上可能不稳定 (可能是 attention interference 问题)
- **Stamp Seal** 在 w/o co-train 时大幅退化 (90 → 60), 说明这个 task 强依赖 dynamics understanding

### 5.4 Real-world Towel Folding (Figure 4)

毛巾折叠是 long-horizon + deformable object 的 challenge task, 在 Galaxea R1 Lite 平台评估。

| Variant | Success rate ↑ | Completion time ↓ | Latency |
|---|:---:|:---:|:---:|
| π0.5 (with PT) | 最高 | 最短 | - |
| π0.5 (no PT) | 低 | 长 | - |
| **Fast-WAM** | 高 | 较短 | **190 ms** |
| Fast-WAM-Joint | 高 | 较长 | ~400 ms |
| Fast-WAM-IDM | 最高 | 较长 | **810 ms** |
| Fast-WAM w/o co-train | **10%** | 最长 | 190 ms |

最 striking 的数字: **w/o video co-training 在 real-world 上 collapse 到 10% success rate**。这个 gap 比 simulation 中 (83.8% vs 91.8%) 大得多。为什么?

我的 hypothesis:
- **Sim-to-real gap 主要体现在 dynamics 上**, 模拟环境 dynamics 是理想的, real-world 有摩擦、形变、不确定性
- Video co-training 学到的 physics prior 在 real-world 是 critical generalization signal
- 没有 video co-training, model 只学到 visual→action mapping, 在 distribution shift 下泛化差
- Deformable object (毛巾) 的 dynamics 极其复杂, 没有强 prior 几乎无法处理

更 general 的 takeaway: **real-world 的 challenge 比 simulation 大得多, 任何 architecture 决策的影响在 real-world 上都会被放大**。

Latency 上, Fast-WAM 的 190ms vs IDM 的 810ms 是 **4.3× 加速**, 这对 real-time control 是巨大的 — 4 Hz vs 1.2 Hz 的 control frequency 在 closed-loop manipulation 上可能是质的差别。

---

## 6. 为什么 Test-time Imagination 帮助不大? — 直觉 Build-up

这是整个 paper 最 thought-provoking 的发现。我从几个 angle 尝试 build intuition:

### 6.1 Information 已经在 weights 里了

Video DiT 在 Wan2.2 pretraining 阶段已经学过海量视频, 知道"杯子掉下来会怎样"、"手推开抽屉会怎样"。再通过 robot data 的 video co-training fine-tune, 这些 dynamics priors 已经 in weights。Inference 时的 explicit imagination 只是把 in-weights knowledge "decode" 出来 — 但 action branch 也可以直接通过 shared attention 从 video branch 的 hidden states 中读取这些 priors, 不需要 decode 到 pixel space。

这跟 LLM 中的现象类似: model 知道答案, 不需要显式 verbalize 中间步骤也能 output。

### 6.2 Stochasticity of generation hurts

Video generation 是 stochastic 的 — 同样的 context 可以 sample 出 different future trajectories。如果 action conditioned on 一个 specific sampled future, 而这个 sample 不 representative, action 就会偏。这是为什么 latent space (z(o,l) 的 hidden states) 比显式生成更鲁棒: latent representation 是 deterministic 的 (single forward pass), 浓缩了所有 plausible futures 的 mixture, 而非单一 sample。

这个 insight 让我想到 diffusion model 的 classifier-free guidance 和 latent space classification — 在 latent space 操作通常比 pixel space 更 robust。

### 6.3 Short horizon 限制了 imagination 的价值

Paper 用 $H = 32$ 的 action chunk, 这是相对短 horizon (可能 1-2 秒的动作)。短 horizon 上 current observation + learned priors 已经足够。

真正需要 imagination 的场景是 long-horizon planning, 比如 "把所有衣服折好放进衣柜" — 需要 look ahead 10+ 秒才能合理分解 sub-goals。但 paper 的 single chunk setting 没测试这个。

可能的 future work: 用 outer auto-regressive loop 把 Fast-WAM 扩展到 long horizon, 看看 imagination 是否在长 horizon 上重新变得 critical。这跟 LLM 中的现象一致: easy problem direct answer OK, hard problem 需要 CoT。

### 6.4 Representation vs Computation 的 trade-off

这跟近期 "test-time compute scaling" 的讨论非常相关 [Snell et al., https://arxiv.org/abs/2408.03314]。OpenAI o1, DeepSeek R1 这些 work 表明 test-time compute (CoT) 可以 boost hard task。但 Fast-WAM 表明, 至少在 robot control 这个 domain, training-time compute (video co-training) 的 ROI 比 test-time compute (imagination) 更高。

但有个 caveat: Fast-WAM 用的 Wan2.2 已经是大规模 pretrained model, 它本身已经吸收了海量 "test-time compute" 价值的信息。所以准确的 framing 应该是: **pretraining-time compute >> fine-tuning-time compute > test-time compute** (在 robot 这个 domain, 给定 Wan2.2 base)。

### 6.5 跟 Model-based RL 的联系

这跟 model-based RL 的 debate 呼应:
- **Dreamer** [https://arxiv.org/abs/1912.01603]: latent imagination + actor-critic, 显式 roll out
- **MBPO**: short-horizon rollout for policy update
- **TD-style model-free**: 直接学 policy, 不显式 model

Fast-WAM 处在中间: model-based representation learning, model-free inference。这跟 Dreamer 的区别在于: Dreamer 的 imagination 是为了 credit assignment (通过 imagined trajectories 算 return), Fast-WAM 的 video 是为了 representation shaping。

最近的 **DreamerPro, TD-MPC2** 等也都在探讨 similar questions: latent dynamics model 何时帮助, 何时纯 model-free 更好。Fast-WAM 在 robot learning + diffusion foundation model 这个组合下给出了一个 data point: **representation > imagination**。

### 6.6 跟 VPP 和 UVA 的对比

相关工作里提到了两个 closest baseline:
- **VPP** (Video Prediction Policy) [34, https://arxiv.org/abs/2412.14803]: 用 video diffusion model 的 predictive visual representations condition policy
- **UVA** (Unified Video Action) [35, https://arxiv.org/abs/2503.00200]: joint model video + action, test-time skip video decoding

Fast-WAM 跟它们的区别:
1. VPP 用 video diffusion 的中间 representations 但训练时可能没 joint training; Fast-WAM 显式 co-train, 让 representation 专门为 action service
2. UVA 在 test-time 跳 video decoding, 但架构可能仍 instantiate video tokens; Fast-WAM 完全 remove video branch, 更彻底
3. Fast-WAM 的核心 contribution 是 controlled comparison disentangle 两个 factor, 而非仅仅提出 efficient variant

---

## 7. 关于 Architecture 的一些 Deeper Thoughts

### 7.1 Shared attention 的角色

Shared attention 让 video branch 和 action branch 在每个 transformer layer 都能 exchange information。这跟 modular networks 的 design philosophy 一致: 不同 modality / task 有 specialized weights, 但通过 shared interface (attention) 协作。

训练时, video branch 的 hidden states 编码了 "what will happen" 的预测, action branch 通过 attention 读取这些信息, 学习用这些 information 来 generate action。这相当于 **distillation of world knowledge from video branch to action branch during training**。

Inference 时, video branch 只做 forward (no future generation), 但它的 hidden states 已经被训练成 "physics-aware representations", action branch 读取这些就够。

这个机制让我想到 **Deep Mutual Learning** 和 **Co-distillation** 的思路, 但用的是 shared attention 而不是 explicit KD loss。

### 7.2 First-frame anchor 的作用

为什么用 clean first frame tokens 作为 anchor 而不是用 noisy future tokens 同时做 anchor?

我的理解:
1. **Action 是 condition on current state 的**, first frame 是 policy 的 input, 应该是 clean signal
2. **防止 shortcut**: 如果 action 可以 attend to noisy future tokens, 它会试图直接 decode future action 信息, 而不学习 representation
3. **Training/inference consistency**: inference 时只有 first frame, training 时也以 first frame 为 anchor, 两边 distribution 一致

### 7.3 Action expert 的尺寸: 1B vs 5B

Action expert 用 reduced hidden dim $d_a = 1024$ (vs video DiT 的 5B), 总参数 1B。这是合理的:
- Video modeling 是 dense prediction (每帧每个 token 都有 supervision), 需要大 capacity
- Action prediction 是低 dimensional output (32 actions × 7-8 dim), 不需要那么大 model
- 但通过 shared attention, action expert 可以 access video branch 的 5B capacity

这种 design 在 multi-modal foundation models 中越来越常见, 例如 MoE for vision-language models 中不同 modality 有不同 size experts。

---

## 8. Limitations 和 Future Directions

Paper 自己提到一个 limitation: 没测试 larger-scale pretraining 的影响。但我想从几个 angle 进一步思考:

### 8.1 Single chunk 的 limitation

Paper 显式 "for simplicity, we focus on single action chunk generation and omit the outer auto-regressive loop"。这是 significant simplification — real robot task 通常需要 multi-chunk execution。

如果 outer loop 加入, imagine-then-execute 的价值可能重新显现: 长时程任务需要 planning, 而 planning 需要 imagine future。但 paper 没测, 这是 open question。

### 8.2 Long-horizon evaluation 缺失

Towel folding 是 long-horizon, 但报告的是 single task。Multi-task, longer-horizon evaluation (e.g., 10-step reasoning for making breakfast) 可能 benefit 更多 from imagination。

### 8.3 Stochastic environment evaluation

所有实验都是 deterministic evaluation (success rate over trials)。在 stochastic environment 中, 显式 imagination + planning 可能有更大价值 — 可以 sample multiple futures 做 contingency planning。这是 model-based RL 的传统优势。

### 8.4 Action expert capacity 没消融

Action expert 用 1B params, 但没消融 expert size 对 co-training 效果的影响。如果 action expert 太小, 可能 underutilize video representations; 太大又可能 overfit。

### 8.5 λ (balance coefficient) 的 ablation

Paper 没给 λ 的具体数值和 ablation。λ 太大, action loss 被稀释; 太小, video co-training 信号弱。这个 sweet spot 的研究是 future work。

---

## 9. 跟更广 Research Context 的联系

### 9.1 Test-time compute 的 hierarchy

| 类别 | 例子 | Fast-WAM 中的对应 |
|---|---|---|
| Pretraining-time compute | Wan2.2 的 web-scale video pretraining | 已 implicit 利用 |
| Fine-tuning compute | Robot data co-training | **dominant factor** |
| Test-time latent compute | Single forward pass encoding | 190ms 的 cost |
| Test-time explicit compute | Iterative video denoising | 4× slower, marginal gain |

Paper 的 finding 可以概括为: **在 robot control + video foundation model 这个 setting 下, fine-tuning compute (video co-training) >> test-time explicit compute (imagination)**。

### 9.2 跟 LLM reasoning 的 analogy

- **Imagine-then-execute**: 类似显式 CoT, 每个 token 都生成出来, 可解释
- **Fast-WAM**: 类似 "internal reasoning" — representations 在 hidden states 中演化, 不显式输出

R1 系列模型的实验也表明, distilled direct-answer 模型在很多 task 上接近 CoT model。但 hard reasoning task 上 CoT 仍有优势。这跟 Fast-WAM 的 finding 一致: 大多数 robot task 不需要 hard reasoning, 但某些 (long-horizon, planning-heavy) 可能仍 benefit。

### 9.3 跟 World Models 经典 paper 的脉络

- **Ha & Schmidhuber 2018** [World Models, https://arxiv.org/abs/1803.10122]: VAE + MDN-RNN + Controller, latent imagination
- **Dreamer** [https://arxiv.org/abs/1912.01603]: latent imagination for actor-critic, 显式 rollout
- **PlaNet**: planning in latent space
- **DreamerV3**: 通用 model-based RL
- **TD-MPC2**: model-based + policy, 强调 representation
- **Fast-WAM**: representation via co-training, no imagination at inference

Fast-WAM 在 robot foundation model 时代重新诠释了 "world model" 的角色 — 不是用来 imagine, 而是用来 shape representation。这跟 Dreamer 系列 (用 world model 做 imagination + planning) 形成鲜明对比。

### 9.4 跟 Diffusion Policy 的联系

Diffusion Policy [Chi et al., https://arxiv.org/abs/2303.04137] 直接用 diffusion 做 action generation, 没有 world model。Fast-WAM 可以看作 Diffusion Policy + video co-training — 保留 Diffusion Policy 的 inference efficiency, 加上 WAM 的 representation advantage。

$\pi_0$ 和 $\pi_{0.5}$ 也是 flow matching + action generation, 但它们靠大规模 embodied pretraining 学到 generalization, 而 Fast-WAM 靠 video co-training (一种 auxiliary loss) 来实现类似效果, 不依赖 embodied pretraining。这是两种不同的 generalization strategy:
- π0.5: data-scale generalization (web-scale robot data)
- Fast-WAM: representation generalization (video prior transfer)

### 9.5 跟 Foundation Model 的 Scalability

Wan2.2-5B 是当前 SOTA 的开源 video model 之一。Fast-WAM 选用它意味着 video representation 越强, action performance 越好。这暗示一个 scaling law: **video model FLOPs → action performance**, 通过 video co-training 传递。如果这个 hypothesis 成立, 跟着 video model scaling 走就能 boost robot policy, 不需要单独 scale robot data — 这对 robot learning 是个 exciting direction。

---

## 10. 我的 Intuition 总结

回到 Karpathy 风格的直觉 build-up, 我对这篇 paper 的 takeaways:

1. **WAM 的 video modeling 主要是 representation shaper, 不是 foresight generator** — 这个 distinction 在 prior work 中没被 disentangle, 这篇 paper 给出 clean experimental evidence
2. **Foundation model pretraining (Wan2.2) + auxiliary fine-tuning (video co-training) 是强大的 recipe**, 比 large-scale embodied pretraining 更 sample-efficient
3. **Test-time compute 在 robot control 上 ROI 不如 training-time compute** — 至少在 short-horizon single chunk setting 上
4. **Disentangled experimentation 是这篇 paper 的 methodological contribution**, 这种 "controlled variants under shared framework" 的做法应该被更多 foundation model paper 采用
5. **Real-world gap > sim gap**, 任何 architecture decision 在 real-world 上都被放大 — w/o co-training 在 sim 上掉 ~8%, 在 real 上掉 ~80% — 这意味着 sim-only evaluation 在某些 hypothesis 验证上可能 misleading

可能的后续 research direction:
- 在 long-horizon multi-chunk setting 下重新 evaluate imagination 的价值
- 探索更大 video backbone (e.g., Wan2.2-14B) 的 effect
- Stochastic / partially observable environment 下的 imagination value
- 把 Fast-WAM 推广到 navigation, locomotion 等其他 embodiment

---

## Reference Links

主 paper 和项目:
- Fast-WAM 项目主页: https://yuantianyuan01.github.io/FastWAM/

Backbone 和基础 model:
- Wan2.2 视频生成 model: https://arxiv.org/abs/2503.20314
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Rectified Flow (Liu et al.): https://arxiv.org/abs/2209.03003
- Diffusion Policy: https://arxiv.org/abs/2303.04137

WAM 和相关 baseline:
- WAM (NVIDIA, Ye et al.): https://arxiv.org/abs/2602.15922
- Motus: https://arxiv.org/abs/2512.13030
- LingBot-VA (Causal World Modeling): https://arxiv.org/abs/2601.21998
- Unified World Models: https://arxiv.org/abs/2504.02792
- Vidar: https://arxiv.org/abs/2507.12898
- VPP (Video Prediction Policy): https://arxiv.org/abs/2412.14803
- UVA (Unified Video Action): https://arxiv.org/abs/2503.00200
- SuSIE (Du et al.): https://arxiv.org/abs/2302.00111
- GR-2: https://arxiv.org/abs/2410.06158

VLA baselines:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- GR00T N1: https://arxiv.org/abs/2503.14734
- Gemini Robotics: https://arxiv.org/abs/2503.20020

Benchmarks:
- LIBERO: https://arxiv.org/abs/2306.03310
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088

Model-based RL 经典:
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Dreamer: https://arxiv.org/abs/1912.01603
- TD-MPC2: https://arxiv.org/abs/2310.16828

Test-time compute scaling:
- Scaling Test-Time Compute (Snell et al.): https://arxiv.org/abs/2408.03314
- DeepSeek R1: https://arxiv.org/abs/2501.12948
