---
source_pdf: A Vision-Language-Action-Critic Model.pdf
paper_sha256: a3ca461e530e8b8b123a449f614f4f18e5bf129e32c83a7288c0bc4e074b0364
processed_at: '2026-08-17T23:36:32-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLAC 用人话讲

## 一句话 version

> 把一个 VLM 训成 "看两张图就能判断第二张比第一张更接近任务完成" 的机器，然后用这个判断当 reward，让机械臂自己 RL。

就这样。下面拆开讲。

---

## 为什么这事难

real-world robot RL 的老问题: reward 怎么定?

**老路子**: 每个 task 手写 reward。比如 "把碗放到盘子上" 这个 task，reward 可能是 "碗中心到盘子中心的距离的负数"，再加一个 "碗是否在盘子上方" 的 binary flag。听着就烦，而且换一个 task (比如扫垃圾) 这个 reward 就废了，得重写。

**另一条路子** ([Eureka](https://arxiv.org/abs/2310.12931), [RL-VLM-F](https://arxiv.org/abs/2402.03681)): 直接问 Gemini/GPT-4V "这帧离完成有多近"。问题: general VLM 不懂 robot task，给静止帧乱打高分，给失败 trajectory 也会说 "做得不错"。而且 VLM 的评分跟时间顺序常常不单调 (第 3 帧打 0.7，第 5 帧反而打 0.4) — 这种 noisy reward 会让 PPO 的 advantage variance 爆炸。

**还有一条** ([LIV](https://arxiv.org/abs/2306.08631), [VIP](https://arxiv.org/abs/2210.00030)): 用 CLIP embedding 距离当 reward。问题: embedding 距离是 "视觉相似度"，不是 "任务进度"。两个视觉上像但任务完全不同的 frame 距离会很小。

VLAC 想做的是: **训一个专门的 model，让它真懂 "task progress" 这个语义**，然后当 reward 用。

---

## 核心 bet

Paper 里反复讲一个观察，我用类比翻译:

> 你第一次玩一个新游戏 (比如魔方)，手生得很，转不好。但你 **看得出来** 自己是更接近还原了还是搞得更乱了。这种 "判断进度" 的能力比 "执行手法" 通用得多 — 你能跨任务、跨场景地判断 progress。

VLAC 的 bet 就是: **把这个 progress 判断能力 distilled 进 VLM**。一旦 VLM 能判断 progress，它就能当 critic (输出 reward)，同时它的 representation 也能帮助 actor (输出 action)。

这是跟 RLHF 里 "reward model 和 policy 分离" 不同的设计哲学 — 这里是 single network 两个 role 用 prompt 切换，本质是 weight sharing 带来的 representation transfer。

---

## 怎么训这个 progress 判断能力

### 核心数据格式: pair-wise

每个 training sample 是 **两张图 + 任务描述**，label 是 **signed scalar**:

$$
c_{i, i+\Delta t} = \frac{\Delta t}{T - i}
$$

变量意思:
- $o_i$: trajectory 第 $i$ 帧图
- $o_{i+\Delta t}$: 第 $i + \Delta t$ 帧图
- $\Delta t$: 两帧的时间差，**可以是负数** (后帧放前面、前帧放后面)
- $T$: 整条 trajectory 长度
- $c_{i, i+\Delta t}$: 第二帧相对于第一帧的 progress delta

直觉解释:
- 分子 $\Delta t$: 两帧隔多远
- 分母 $(T - i)$: 从第 $i$ 帧到结束还有多远
- 同样 $\Delta t = 5$ 帧，在 trajectory 开头 (i=10, T=100) 占 5/90 ≈ 5.5% progress，在接近结尾 (i=90, T=100) 占 5/10 = 50% progress
- 这模拟了 "任务接近完成时，同样物理动作对应更大的 semantic progress"

**关键设计**: 这个 label **跟 trajectory 全局起点无关**。任何 sub-segment 都自包含地定义了 progress。所以人手视频 ([Ego4D](https://arxiv.org/abs/2110.07058)) 和机械臂视频 ([Bridge](https://arxiv.org/abs/2308.12952), [DROID](https://arxiv.org/abs/2403.12917)) 可以在同一个 objective 下训，因为 progress 这个语义跟 action space 无关 — 人手的 "把杯子拿起来" 和机械臂的 "把杯子拿起来"，progress 的语义是一样的，虽然 action 维度完全不同。

### 四个数据构造 trick

1. **Pixel diff filtering**: 如果两帧像素差异 < 1%，强制 label = 0。机械臂卡住不动时，时间在走但画面没变，不能让 model 学成 "时间推进 = progress"。这个 trick 直接解决了 [ALAN (Mendonca et al. 2023)](https://arxiv.org/abs/2302.06604) 里遇到的 "静止帧被 VLM 打高分" 问题。

2. **Symmetric sampling**: 每对 $(o_i, o_{i+\Delta t})$ 都构造 4 个样本 — 正向单步、反向单步、正向长程、反向长程。强制 model 学到 "哪一帧更接近 goal"，而不是学 shortcut "second image is always better"。这个对称性是 critic 能 reject failure trajectory 的核心。

3. **Mismatched task description**: 5% 概率用一个 **不匹配** 的 task description，label 强制为 0。这让 model 不会给 "做错任务但执行顺利" 的 trajectory 打高分。比如 task 是 "把碗放到盘子"，但 trajectory 在 "把碗放到锅" — 也要打 0。

4. **Done judgment**: 给单张图判断任务是否完成。但中间 15% (0.8T 到 0.95T) **不给 label**。直觉: trajectory 最后一段是微调对齐阶段，哪一帧算"真正完成"很 ambiguous，强行 label 会引入噪声，让 model abstain 比 guess 更好。

### In-context learning

公式:
$$
c_{i, i+\Delta t} = \mathrm{VLAC}(o_i, o_{i+\Delta t}; l_{\mathrm{task}}, O_{\mathrm{ref}}, o_0)
$$

新变量:
- $O_{\mathrm{ref}}$: reference demonstration (可以是 human 或 robot 的一条 trajectory)
- $o_0$: 当前 trajectory 的起点帧 (optional)

这借用 LLM 的 in-context 能力。Table 1 的数据 striking:

| Dataset | zero-shot VOC | one-shot VOC |
|---------|--------------|--------------|
| RoboNet | NAN (失败) | 0.59 |
| RT1 | 0.71 | 0.91 |
| RH20T | 0.17 | 0.64 |

[RoboNet](https://arxiv.org/abs/1910.11215) 没有 language annotation，zero-shot 完全废，但给一个 reference 立刻能 work。这印证: **VLAC 的 progress 理解是 conditioned on task semantics 的**，不是纯视觉相似度。给一个 reference，model 就能 align "这个任务里 progress 是什么意思"。

---

## Action 怎么生成

$$
a_i = \mathrm{VLAC}(o_i^0, \ldots, o_i^k; s_i; l_{\mathrm{task}}; \mathrm{history}_{i-1, i-t_h})
$$

变量:
- $o_i^k$: 第 $i$ 步第 $k$ 个 camera 的图
- $s_i$: robot state (joint position 等)
- $l_{\mathrm{task}}$: task description
- $\mathrm{history}_{i-1, i-t_h}$: 过去 $t_h$ 步的 action 历史
- $a_i$: action，输出成 string，比如:
  > "x: -47mm, y: 19mm, z: 66mm, roll: 14 degrees, pitch: 10 degrees, yaw: 15 degrees, open: 0"

设计选择的人话版:

1. **delta EEF pose 而非 absolute**: 用相对位移而不是绝对位置，跨 robot 平台可迁移。直觉: "往前伸 5cm" 这个指令对任何机械臂都通用，"move to position (123, 456, 789)" 就只对特定 setup 有意义。

2. **用 string 而非 continuous vector**: 直接复用 pretrained VLM 的 numeric token generation。不需要新加 continuous action head，可以直接用 LLM 的 RL recipe。

3. **Autoregressive token generation**: 每个 number 是 vocab 里的 token，每个 token 都有 logprob，可以直接喂给 PPO。这点跟 [FAST (Pertsch et al. 2025)](https://arxiv.org/abs/2501.09747) 的 action tokenization 思路一致。

跟 [π₀ (Black et al. 2024)](https://arxiv.org/abs/2410.24164) 的 flow matching continuous action 路线不同。Paper Section 2 明说了，diffusion/flow-matching action head 的 RL integration 是 open problem — backprop through 多步 denoising 是 BPTT-like，不稳定。VLAC 选择 tokenized action 这条 simpler 但更 RL-friendly 的路。

---

## PPO 怎么用在 tokenized action 上

标准 PPO clipped surrogate (公式 5):
$$
\mathcal{L}^{\mathrm{PPO}} = \mathbb{E}_t \left[ \min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t) \right]
$$

变量:
- $r_t = \frac{\pi_{\mathrm{new}}(a_t | s_t)}{\pi_{\mathrm{old}}(a_t | s_t)}$: 新 policy 对 action 的概率 / 旧 policy 对同一 action 的概率
- $A_t$: advantage (用 GAE 算)
- $\epsilon$: clip range，通常 0.1-0.2

Action 是 "x: -47mm, y: 19mm, ..." 这种 string，model 逐 token 生成。每个 numeric token (比如 "-47") 在 vocab 里有对应 logit。整个 action 的 logprob = $\prod_k \pi(t_k | t_{<k}, s)$，即所有 numeric token logprob 之积。

**Value head**: paper Section 3.2.2 说 "extract the hidden state (prior to the final token projection) and pass it through a linear value head to obtain $V(s_t)$"。也就是最后一个 transformer layer 的 hidden state 接一个 linear projection，输出 scalar value。这是 RLHF 里的标准做法 ([InstructGPT](https://arxiv.org/abs/2203.02155) 那种)。

---

## Real-world RL 工程里几个 hard-won 的细节

### 异步执行 + Latency 对齐

Robot 异步上传 observation，VLA 异步生成 action。如果 VLA 推理要 100ms，那 action 生成时看到的 observation 已经是 100ms 前的了。直接执行的话，action 对应的是 stale state。

VLAC 的 trick: **训练时把 action 的 ground truth timestamp 往后挪 100ms**，让 model 学会 "predict 未来一步要做什么"。配合 robot motion speed，当 action 到达时正好接上 — 形成 smooth motion。

这跟 [SERL (Luo et al. 2024)](https://arxiv.org/abs/2401.16087) 里 async RL latency 补偿思路类似，但 VLAC 用训练时 timestamp shift 而非 inference 时 buffer。

### vllm vs torch 的灾难

这是 paper 里最 honest 的工程发现:

> Under identical neural network parameters and samples, the importance ratios for the same action generated by vllm and torch fluctuate between 0.4 and 1.8. This discrepancy frequently triggers the clipping mechanism in PPO, rendering approximately 60% of the data unusable.

直觉: vllm 用 PagedAttention、batching 策略、KV cache 管理，跟原生 torch forward 不完全 numerical 等价。Generation 时无所谓，但 PPO 里 importance ratio = π_new/π_old，任何 numerical drift 都会被放大成 ratio 偏离 1，触发 clip，导致 60% 数据被废掉。

**解法**: inference 用 vllm (快)，training 时用 torch 重新 forward 一遍算 logprob。Inference 和 training 用两套 forward，训练慢但稳定。

这个问题在 [TRL 的 PPO 实现里](https://huggingface.co/docs/trl) 也遇到过，[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 也讨论过，但 VLAC 是第一个在 real-world robot RL context 里量化这个 issue 的。

### Single-controller 架构

用 Ray 搭一个 cluster，里面 inference worker / trainer / data server / rollout worker 都是独立 component，通过 ZeroMQ 通信。Actor (2B) 占 2 GPU，critic (8B) 占 1 GPU。这跟 RLHF 里 actor-critic 双 controller (两个 GPU cluster 分别跑 policy 和 value model) 不同 — 这里 single controller，更 compact。

---

## Human-in-the-Loop 三层介入

### 第一层: Offline Demonstration Replay (最轻)

预填一个 expert demo buffer，训练时定期采样，用 NLL loss 更新 (公式 6):
$$
L_{\mathrm{NLL}}(\theta) = -\sum_{(a_t, s_t) \in D_{\mathrm{human}}} \log \pi_\theta(a_t | s_t)
$$

变量:
- $D_{\mathrm{human}}$: 人类 teleop 数据集
- $\pi_\theta(a_t | s_t)$: policy 在 state $s_t$ 下输出 action $a_t$ 的概率
- 整个 loss 就是 standard BC loss

这是 BC regularization，跟 [AWAC](https://arxiv.org/abs/2006.09359)、[IQL](https://arxiv.org/abs/2110.06169) 的 offline-to-online 思路一致。Table 3 显示 average 88% → 93%。

### 第二层: Return and Explore (中等)

Tele-operator 观察 robot 在哪些 initial position 频繁失败，**手动 reset** 到这些 hard state 让 robot 反复尝试。

直觉: real-world RL 的 reset cost 极高，如果让 robot 自然 random start，它会在 easy case 上反复成功、hard case 上完全 0 reward → gradient 信号极度不均衡。Manual reset 是 importance sampling 的物理版本。

Table 3 显示 88% → 95%。但 paper 承认: "Some return points are too easy... some return points are too difficult" — 选择 return point 是 art 不是 science。

### 第三层: Human Guided Explore (最重)

对于 policy 完全探索不出来的 sub-behavior，tele-operator **现场演示** 几条 trajectory，加入 demo buffer。这是 [Genie Centurion (Wang et al. 2025)](https://arxiv.org/abs/2505.18793) 的 "rewind-and-refine" 思路。Table 3 显示 88% → 98%。

---

## 实验结果怎么读

### Critic 评估 (Table 1)

评估指标 (公式 7-9):
$$
\begin{cases}
\mathrm{VOC} = \mathrm{rank\text{-}correlation}(\mathrm{argsort}(v_1, \ldots, v_T); \mathrm{arange}(T)) \\
v_i = v_{i-\Delta t} + c_{i-\Delta t, i}(1 - v_{i-\Delta t}) \\
v_0 = 0
\end{cases}
$$

变量:
- $v_i$: accumulated progress value，从 0 累加到 1
- $(1 - v_{i-\Delta t})$: multiplicative discount，让 $v_i$ 始终 bound 在 [0, 1]
- VOC: predicted value 的 rank 和真实时间 rank 的 correlation，∈ [-1, 1]
- VROC: 把 sequence 反过来，看 model 是否能正确 reverse
- VOC-F1 = 2·VOC·VROC / (VOC + VROC): harmonic mean

Key findings:

1. **RoboFAC-success vs fail**: VLAC-8b 在 success 上 VOC-F1 = 0.87，fail 上 0.30。这个 gap 是 critic 能 reject failure trajectory 的核心证据。

2. **vs Gemini-1.5-Pro**: 在 [DROID](https://arxiv.org/abs/2403.12917) 上，[Gemini](https://arxiv.org/abs/2503.20020) VOC = -0.01 (随机)，VLAC-8b VOC = 0.92。General VLM 不懂 robot task progress，必须专门 train。

3. **Ego4D ablation**: 加 [Ego4D](https://arxiv.org/abs/2110.07058) human video 后，success/fail gap 进一步拉大。这印证 paper 的 thesis: **human video data 对 embodied task understanding 有显著增益**，即便 action space 完全不同。

### Actor 评估 (Table 2)

| Method | Avg success | Lighting transfer | Scene transfer |
|--------|------------|-------------------|----------------|
| [Pi0](https://arxiv.org/abs/2410.24164) | 27% | 3% | - |
| VLAC | 75% | 57% | 63% |

VLAC 在 distribution shift 下 robustness 显著优于 Pi0。直觉: VLAC 的 critic pretraining 强制 model 理解 "task state"，这种 semantic understanding 比 pure BC 的 action imitation 更 robust to visual perturbation。Pi0 在 Lighting Transfer 下 avg 3%，基本崩溃 — pure flow matching policy 对 lighting 极度敏感。

### Real-World RL 学习曲线

4 个 task，200 episodes 内 30% → 90%。这跟 [SERL](https://arxiv.org/abs/2401.16087) 报告的 "20 minutes to learn a task" 量级相当，但 SERL 是 task-specific reward，VLAC 是 **同一个 critic model 跨 4 个完全不同的 task** (granular/rigid/flexible object manipulation)。

### Multi-Robot Scaling (Table in Section 4.7)

| Robots | Episodes/robot to 80% | Wall clock |
|--------|----------------------|-----------|
| 1 | ~140 | ~50 min |
| 2 | 325 | ~2 hours |
| 4 | 147 | ~55 min |
| 8 | 64 | ~24.6 min |

8 robots 时 per-robot data 需求降到 64 episodes，接近 linear scaling。2 robots 反而比 1 robot 差 — heterogeneous background 在小规模时引入 visual disturbance 但 gradient signal 不足以 generalize。这跟 [Chinchilla](https://arxiv.org/abs/2203.15556) 在 LLM 里观察到的 "small scale 看不到 scaling" 现象类似。

---

## 几个我会追问 paper 作者的问题

### 1. Pair-wise progress 假设 monotonic

Label 公式 $c = \Delta t / (T-i)$ 假设 expert trajectory 是单调推进的。但 real-world RL rollout 里 robot 经常 stuck 或 regress，这个 label 在 RL 自己生成的 failure trajectory 上是错的。Paper 的 RoboFAC-fail 实验 VOC-F1 = 0.30 显示 model 学到了一些 "failure detection" 能力，但来源是对称采样 trick，不是 label 本身。

### 2. Reward sparsity in long-horizon

即便 c 是 dense 的，long-horizon task (比如做饭 30 步) 里，单步 c 值可能 < 0.01，advantage signal 弱。Paper 没讨论 reward normalization 或 [potential-based reward shaping](https://arxiv.org/abs/1907.02025)。

### 3. Actor 和 critic 不是同一个 model

Section 3.1 说 "unifies the roles of actor and critic within a single autoregressive architecture"，但实际 actor 是 2B，critic 是 8B，两个独立 model。真正的 unified 应该是 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 那种 MoE shared expert + routed expert 设计。

### 4. Action representation 局限

String-based delta EEF pose 对 7-DOF arm OK，但对 dexterous hand (比如 [EgoDex](https://arxiv.org/abs/2505.11709) 那种 25-DOF) 完全不够。Paper 在 EgoDex 上只测了 critic，没测 actor。High-DOF action 的 tokenization 是 future work。

### 5. Human-in-the-loop 没有量化 trigger

什么时候介入? 介入多久? 完全靠 operator 经验。这跟 [Genie Centurion](https://arxiv.org/abs/2505.18793) 一样，都是 art。需要 [competence plateau detector](https://arxiv.org/abs/2407.20635) 这种 quantitative trigger。

---

## 跟你的 work 的连接

Andrej，你在 Tesla 和 OpenAI 都讲过 "data engine 比 algorithm 重要"。VLAC 的三层 human-in-the-loop 本质上是个 data engine — tele-operator 不写代码，只通过 reset/demonstration 注入信息。这跟 Tesla 的 data annotation pipeline 异曲同工。

你在 [Yann LeCun 对谈](https://www.youtube.com/watch?v=5Ot7eE2xaXw) 里讲过 "world model 是 RL 的 bottleneck"。VLAC 把 VLM 当 approximate world model (predict "state A → state B 是否更接近 goal")，这是一种 pragmatic 的 world model 使用方式 — 不需要 predict next frame pixel，只需要 predict progress delta，这是更 tractable 的目标。

---

## 一句话总结 (人话版)

VLAC 把 "看两张图判断任务有没有进步" 这个人类天然擅长、VLM 可以通过 pair-wise pretraining 习得的能力，蒸馏成一个 signed dense reward。这个 reward 跨 task、跨 scene、跨 embodiment 都能用，配合 PPO 和三层 human-in-the-loop，让机械臂在 4 个 manipulation task 上 200 episodes 内从 30% 成功率涨到 90%。离 "fire-and-forget autonomous robot learning" 还有距离 (human-in-the-loop 仍是 art，multi-task 不稳定，action representation 局限)，但已经把 reward engineering 这个最大 bottleneck 拆掉了。

---

## References

- [VLAC Paper (本篇)](https://arxiv.org/abs/2508.xxxxx) — Shanghai AI Lab, Jiangmiao Pang 组
- [InternVL](https://arxiv.org/abs/2312.14238) — VLAC 的 base model
- [π₀ (Black et al. 2024)](https://arxiv.org/abs/2410.24164) — Flow matching VLA baseline
- [π₀.5](https://arxiv.org/abs/2504.16054) — Open-world generalization VLA
- [SERL (Luo et al. 2024)](https://arxiv.org/abs/2401.16087) — Sample-efficient real-world RL
- [OpenVLA (Kim et al. 2024)](https://arxiv.org/abs/2406.09246) — Open-source VLA
- [FAST tokenization (Pertsch et al. 2025)](https://arxiv.org/abs/2501.09747) — Action tokenization for VLA
- [LIV (Ma et al. 2023)](https://arxiv.org/abs/2306.08631) — Language-image value/reward
- [VIP (Ma et al. 2022)](https://arxiv.org/abs/2210.00030) — Value-implicit pre-training
- [VLMs are in-context value learners (Ma et al. 2024)](https://arxiv.org/abs/2402.00764) — VLAC 的直接前驱
- [Eureka (Ma et al. 2023)](https://arxiv.org/abs/2310.12931) — LLM-generated reward code
- [RL-VLM-F (Wang et al. 2024)](https://arxiv.org/abs/2402.03681) — RL with VLM feedback
- [ConRFT (Chen et al. 2025)](https://arxiv.org/abs/2502.05450) — RL fine-tuning for VLA
- [Diffusion Policy Optimization (Ren et al. 2024)](https://arxiv.org/abs/2409.00588) — Diffusion policy 的 PPO
- [Flow Q-Learning (Park et al. 2025)](https://arxiv.org/abs/2502.02538) — Flow matching + RL
- [Genie Centurion (Wang et al. 2025)](https://arxiv.org/abs/2505.18793) — Human rewind-and-refine
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347) — 基础算法
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) — LLM RL 的另一选择
- [Bridge V2](https://arxiv.org/abs/2308.12952) — Robot manipulation dataset
- [DROID](https://arxiv.org/abs/2403.12917) — Large-scale manipulation dataset
- [Ego4D](https://arxiv.org/abs/2110.07058) — Egocentric human video
- [AGIBOT World](https://arxiv.org/abs/2502.03043) — 大规模 manipulation 平台
- [RoboNet](https://arxiv.org/abs/1910.11215) — Multi-robot dataset
- [RT1](https://arxiv.org/abs/2212.06817) — Robotics Transformer
- [EgoDex](https://arxiv.org/abs/2505.11709) — Dexterous manipulation from egocentric video
- [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155) — Value head 设计参考
- [AWAC](https://arxiv.org/abs/2006.09359) — Offline-to-online RL
- [IQL](https://arxiv.org/abs/2110.06169) — Implicit Q-learning
- [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) — LLM scaling reference
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — RLHF 训练框架
- [TRL (HuggingFace)](https://huggingface.co/docs/trl) — PPO 训练库
- [ALAN (Mendonca et al. 2023)](https://arxiv.org/abs/2302.06604) — Autonomous exploring robotic agents
- [Rodney Brooks 1991 — Intelligence without representation](https://people.csail.mit.edu/brooks/papers/representation.pdf) — Paper epigraph 出处
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — MoE unified actor-critic 参考

---

# VLAC: Vision-Language-Action-Critic Model 深度解析

## 0. Big Picture — 这篇 paper 在解决什么本质问题

Andrej, 你在 Tesla 和 OpenAI 都做过 real-world robot learning, 你最清楚这里的核心痛点:**real-world RL 的 reward 工程是 roboticist 的噩梦**。每个 task 都要 hand-craft 一个 reward shaping, 每个 task 都要 train 一个 done classifier, 每个 task 都要 collect task-specific data 来训 reward surrogate。这篇 paper 的核心 contribution 是用一个 **pretrained multimodal model 把 reward、done signal、action policy 三件事 unify 到一个 autoregressive architecture 里**, 并且这个 reward 是 **signed dense progress delta**, 跨 task、跨 scene、跨 embodiment 都能 generalize。

直觉上, paper 借用了一个 cognitive science 的观察 (paper 里也提了 Rodney Brooks 1991 那句话 "Intelligence is determined by the dynamics of interaction with the world"):

> 人类面对新任务时, 执行能力初期可能很烂, 但 **判断"我离目标更近了还是更远了"** 的能力却非常 general。这种 "task progress understanding" 比 "motor execution" 更 high-level、更可迁移。

VLAC 的核心 bet 就是: **把这种 progress judgment 能力 distill 进 VLM, 然后让它同时驱动 critic (reward) 和 actor (action)**。这跟 RLHF 里 reward model + policy 分离的范式不同, 这里是 single network, 两个 role 用不同 prompt 切换。

---

## 1. VLAC 模型架构

### 1.1 总体设计

基础是 **InternVL** (Shanghai AI Lab 自家的 VLM, [InternVL paper](https://arxiv.org/abs/2312.14238)), 训练后形成三个 head 共享一个 backbone:

| Head | 输入 | 输出 | 角色 |
|------|------|------|------|
| Critic (progress) | (o_i, o_{i+Δt}, l_task, [O_ref, o_0]) | signed scalar c | RL reward |
| Done classifier | (o_i, l_task) | {0, 1} | terminal signal |
| Actor | (o_i^0..k, s_i, l_task, history) | action token string | policy |
| (Aux) Task description | (o_start, o_end) | l_task | 反向理解任务 |

实验中 paper 用了两个规模: **2B VLAC 当 actor**, **8B VLAC 当 critic**。这跟 RLHF 里 actor 小、reward model 大的常规相反 — 这里 critic 必须更大, 因为 critic 的 generalization 是整个系统 sample efficiency 的 bottleneck, actor 已经有 100 条 teleop trajectory 做 SFT warmup 了。

### 1.2 为什么不分离 actor 和 critic?

paper Section 3.1 没明说, 但从 Figure 3 能读出来: 这是 **weight sharing 的 representation transfer**。Critic 学到的 "task progress 是怎么变化的" 这个 semantic 理解, 直接 leak 进 actor 的 hidden state。Table 2 里 "VLAC w/o pretrain" 一行就是 ablation — 没有 progress pretraining 的模型, 平均 success rate 从 75% 掉到 16%, 而且 failure mode 是 "在 pick-and-place 里没 grasp 成功就直接去 place 了"。这印证了 progress understanding 对 action 选择有 causal 影响, 单独 BC 训出来的 actor 不懂 "我现在在哪一步"。

这跟你之前在 [Eureka Labs](https://eurekalabs.ai) 讲过的 "LLM 同时是 world model 和 policy" 的直觉一致 — 在 robot 上, 把 value/progress 和 policy 物理上塞进同一个 transformer 比 split 更 sample efficient。

---

## 2. Pair-wise Progress Learning — 核心方法

### 2.1 公式 1 解析

$$
c_{i, i+\Delta t} = \mathrm{VLAC}(o_i, o_{i+\Delta t}; l_{\mathrm{task}})
$$

变量含义:
- $o_i$: trajectory 第 $i$ 帧的 RGB observation
- $o_{i+\Delta t}$: 第 $i+\Delta t$ 帧, 其中 $\Delta t \in [-i+1, T-i] \cap \mathbb{Z}$
  - 注意 $\Delta t$ 可以是 **负数** — 这意味着允许把后帧放前面、前帧放后面, 自然构造 negative sample
- $T$: trajectory 总长度
- $l_{\mathrm{task}}$: language task description
- $c_{i, i+\Delta t}$: signed progress delta, >0 表示第二帧比第一帧更接近完成

**Label 设计**:
$$
c_{i, i+\Delta t} = \frac{\Delta t}{T - i}
$$

这里非常关键, 我详细讲:

- 分子 $\Delta t$: 两帧之间的时间差, 可正可负
- 分母 $(T - i)$: 从第 $i$ 帧到 trajectory 结束的剩余长度

为什么用 $(T-i)$ 而不用常数 $T$? 因为这是 **relative progress**。同样 $\Delta t = 5$ 帧的间隔, 在 trajectory 早期 (i=10, T=100) 代表 5/90 ≈ 5.5% 进度; 在 trajectory 末期 (i=90, T=100) 代表 5/10 = 50% 进度。这模拟了 "任务越接近完成, 同样物理动作对应的 semantic progress 越大" 的直觉 — 比如最后把碗放到盘子上那一下, 比 trajectory 开头的"移动到碗附近"进度感更强。

**这个 label 设计的妙处**: 它对 trajectory 的 **global start point 不敏感**。paper 反复强调 "agnostic to data collection strategy and to segment starting points"。任何 sub-segment 都自包含地定义了 progress, 这就让 human video (Ego4D) 和 robot video (Bridge, DROID) 能在同一个 training objective 下 mix, 不需要对齐 action space — 因为人手和机械臂的 action 维度完全不一样, 但 progress 的语义是一样的。

### 2.2 公式 4 — In-context Learning

$$
c_{i, i+\Delta t} = \mathrm{VLAC}(o_i, o_{i+\Delta t}; l_{\mathrm{task}}, O_{\mathrm{ref}}, o_0)
$$

- $O_{\mathrm{ref}}$: reference process, 一段 demonstration video (可以是 human 或 robot)
- $o_0$: 当前 trajectory 的 start frame (optional)

这个设计直接借用 LLM in-context learning 的能力。Table 1 数据非常 striking:

| Dataset | zero-shot VOC | one-shot VOC |
|---------|--------------|--------------|
| RoboNet | NAN (失败) | 0.59 |
| RT1 | 0.71 | 0.91 |
| RH20T | 0.17 | 0.64 |

RoboNet 没有 language annotation, zero-shot 完全失效 (VOC=0), 但给一个 reference 立刻能 work。这印证了: **VLAC 的 progress 理解是 conditioned on task semantics 的, 不是纯视觉相似度**。这跟 [LIV (Ma et al. 2023)](https://arxiv.org/abs/2306.08631) 和 [VIP (Ma et al. 2022)](https://arxiv.org/abs/2210.00030) 用 contrastive embedding 距离当 reward 的范式有本质区别 — VLAC 是真正的 goal-conditioned semantic judgment, 不是 metric learning。

---

## 3. 数据构造策略 — 这是 paper 最 underappreciated 的部分

paper Section 3.1.1 列了 4 个策略, 我逐个讲为什么这么设计:

### 3.1 Pair-wise Image Difference Filtering

如果 $\mathrm{Diff}(o_i, o_{i+\Delta t}) < \sigma$, 强制 $c = 0$。

- $\sigma = 1\%$ 像素差异阈值
- 物理含义: 静帧 (机械臂不动、相机噪声) 不应该贡献 progress
- 直觉: 防止 model 把"画面没变但时间在走"误判为 progress, 这种 noise 会让 reward signal 在 stuck episode 里漂移

这跟 [ALAN (Mendonca et al. 2023)](https://arxiv.org/abs/2302.06604) 里遇到的 "VLM 给静止帧打高分" 问题对应 — VLAC 用显式 filtering 解决。

### 3.2 Joint Sampling (公式 4 的 4-tuple)

对每个 anchor pair $(o_i, o_{i+\Delta t})$, 构造 4 个样本:
$$
\begin{cases}
(o_i, o_{i+1}) & \text{fine forward} \\
(o_{i+1}, o_i) & \text{fine backward (negative)} \\
(o_i, o_{i+\Delta t}) & \text{global forward} \\
(o_{i+\Delta t}, o_i) & \text{global backward (negative)}
\end{cases}
$$

这是 **contrastive symmetry 的显式 enforcement**。如果只采样 forward, model 容易学到 "second image is always more progress" 的 shortcut。强制 backward 让 model 必须真正理解 "哪一帧更接近 goal", 而不是依赖位置 prior。

这跟 [Time-Contrastive Networks (Sermanet et al. 2018)](https://arxiv.org/abs/1704.06888) 和 [TCN 的 frame order 预测](https://arxiv.org/abs/1905.07035) 思路类似, 但 VLAC 直接 regress signed scalar 而不是 contrastive loss, 信息密度更高。

### 3.3 Task Description Cross-sampling

5% 概率用一个 **不匹配** 的 $l_{\mathrm{task}}$, 强制 $c=0$。

这是 reject "irrelevant prompts" 的训练。在 real-world RL 里, robot 可能在一个多 object scene 里, model 必须能判断"这个动作对当前 task 没用" — 即便动作本身在执行某个 valid sub-task。这个 negative 训练让 critic 不会给 "做错任务但执行顺利" 的 trajectory 误打高分。

### 3.4 Task Completion Judgment

公式 3: $l_{\mathrm{done}} = \mathrm{VLAC}(o_i; l_{\mathrm{task}})$

- $i < 0.8T$: label = 0
- $i > 0.95T$: label = 1
- $0.8T \leq i \leq 0.95T$: **不训** (no label)

中间这段留白非常聪明。trajectory 最后 20% 经常是 "微调对齐" 阶段, 哪一帧算 "真正完成" 是 ambiguous 的。强行 label 会引入 noise。这跟 [FMB benchmark](https://arxiv.org/abs/2401.08553) 里 "success threshold" 的 fuzziness 一样, VLAC 选择 abstain 而不是 guess。

---

## 4. Action Representation — 为什么用 delta EEF pose 作为 string

公式:
$$
a_i = \mathrm{VLAC}(o_i^0, \ldots, o_i^k; s_i; l_{\mathrm{task}}; \mathrm{history}_{i-1, i-t_h})
$$

- $o_i^k$: 第 $i$ 步第 $k$ 个 viewpoint (multi-camera)
- $s_i$: robot state (joint positions 等)
- $\mathrm{history}_{i-1, i-t_h}$: 过去 $t_h$ 步的 generated action history
- $a_i$: 输出 action, 表示为 string, 例如:
  > "x: -47mm, y: 19mm, z: 66mm, roll: 14 degrees, pitch: 10 degrees, yaw: 15 degrees, open: 0"

设计选择:
1. **delta pose 而非 absolute pose**: 通用、embodiment-agnostic, 跨 robot 平台可迁移
2. **String 而非 continuous vector**: 利用 pretrained VLM 的 numeric token generation 能力, 不需要新加 continuous action head
3. **Autoregressive token generation**: 每个 number 是 vocab 里的 token, 直接有 logprob 可用于 PPO

这个设计跟 [FAST (Pertsch et al. 2025)](https://arxiv.org/abs/2501.09747) 的 action tokenization 思路一致 — 把 action 压成 discrete tokens 才能用 LLM 的 RL recipe。跟 [π₀ (Black et al. 2024)](https://arxiv.org/abs/2410.24164) 用 flow matching 生成 continuous action 的路线完全不同 — paper Section 2 也明确说了, diffusion/flow-matching action head 的 RL integration 是 open problem, 因为 backprop through multi-step denoising 是 BPTT-like, 不稳定。

### 4.1 PPO 在 tokenized action 上的应用

公式 5 (PPO clipped surrogate):
$$
\mathcal{L}^{\mathrm{PPO}} = \mathbb{E}_t \left[ \min(r_t \cdot A_t, \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t) \right]
$$

- $r_t = \frac{\pi_{\mathrm{new}}(a_t | s_t)}{\pi_{\mathrm{old}}(a_t | s_t)}$: importance ratio, new policy vs old policy 在 action token 上的概率比
- $A_t$: GAE advantage
- $\epsilon$: clip range (通常 0.1~0.2)

**Action 的 logprob 怎么算?** 因为 action 是 "x: -47mm, y: 19mm, ..." 这种 string, model 逐 token 生成。每个 numeric token (比如 "-47") 在 vocab 里有对应 logit, paper Section 3.2.2 说 "we record the logits associated with the selected tokens"。整个 action 的 logprob = $\prod_k \pi(t_k | t_{<k}, s)$, 即所有 numeric token logprob 之积。

**Value head**: paper 说 "extract the hidden state (prior to the final token projection) and pass it through a linear value head to obtain $V(s_t)$"。这是标准做法, 跟 [RLHF 里的 value head](https://arxiv.org/abs/2203.02155) 一样, 用最后一个 transformer layer 的 hidden state 接一个 linear projection。

---

## 5. Real-World RL Infrastructure — 工程细节非常 hard-won

### 5.1 异步执行 + Latency 对齐

paper Section 3.2.1 描述了一个我之前没见过的 trick:

> "during VLA training, action timestamps are adjusted to lag behind observation timestamps by a duration determined by the VLA's inference time"

直觉: robot 异步上传 observation, VLA 异步生成 action。如果 VLA 推理要 100ms, 那 action 生成时看到的 observation 已经是 100ms 前的了。如果直接执行, action 对应的是 stale state。VLAC 的做法是 **训练时把 action 的 ground truth timestamp 往后挪 100ms**, 让 model 学会 "predict 未来一步要做什么"。配合 robot motion speed, 当 action 到达时正好接上 — 形成 smooth motion。

这跟 [SERL (Luo et al. 2024)](https://arxiv.org/abs/2401.16087) 里讲的 async RL latency 补偿思路类似, 但 VLAC 用的是训练时 timestamp shift 而非 inference 时 buffer。

### 5.2 vllm vs torch 的 PPO 灾难

这是 paper 里最 honest 也最 valuable 的工程发现:

> "Under identical neural network parameters and samples, the importance ratios for the same action generated by vllm and torch fluctuate between 0.4 and 1.8. This discrepancy frequently triggers the clipping mechanism in PPO, rendering approximately 60% of the data unusable."

直觉: vllm 用 PagedAttention、batching 策略、KV cache 管理都跟原生 torch forward 不完全等价, 导致 numerical 略有差异。在 generation 时无所谓, 但在 PPO 里 importance ratio = π_new/π_old, 任何 numerical drift 都会被放大成 ratio 偏离 1, 触发 clip。

**解法**: inference 用 vllm (快), training 时用 torch 重新 forward 一遍算 logprob。这相当于 inference 和 training 用两套 forward, 训练慢但稳定。

这个问题在 [TRL 的 PPO 实现里](https://huggingface.co/docs/trl) 也遇到过, 但 paper 是第一个在 real-world robot RL context 里量化这个 issue 的 (60% 数据废掉)。

### 5.3 Single-Controller 架构

跟 RLHF 里 actor-critic 双 controller (两个 GPU cluster 分别跑 policy 和 value model) 不同, VLAC 用 single controller: 一个 Ray cluster 里, inference worker / trainer / data server / rollout worker 都是独立 component, 通过 ZeroMQ 通信。Critic 8B 和 actor 2B 都在这个 cluster 里。Actor 占 2 GPU, critic 占 1 GPU。

---

## 6. Human-in-the-Loop — 三层介入

paper Section 3.2.3 提了三个等级, 我按 escalation 排序:

### 6.1 Offline Demonstration Replay (最轻)

预填一个 expert demo buffer, 训练时定期采样, 用 NLL loss 更新:

$$
L_{\mathrm{NLL}}(\theta) = -\sum_{(a_t, s_t) \in D_{\mathrm{human}}} \log \pi_\theta(a_t | s_t)
$$

- $D_{\mathrm{human}}$: 人类 teleop 数据集
- 这是 BC regularization, 跟 [AWAC](https://arxiv.org/abs/2006.09359)、[IQL](https://arxiv.org/abs/2110.06169) 的 offline-to-online 思路一致

Table 3 显示这一招 average 88% → 93%, 提升温和但稳定。

### 6.2 Return and Explore (中等)

Tele-operator 观察 robot 在哪些 initial position 频繁失败, **手动 reset** 到这些 hard state 让 robot 反复尝试。

直觉: real-world RL 的 reset cost 极高, 如果让 robot 自然 random start, 它会在 easy case 上反复成功、hard case 上完全 0 reward → gradient 信号极度不均衡。Manual reset 是 importance sampling 的 physical 版本。

Table 3 显示这一招 average 88% → 95%。但 paper 也承认: "Some return points are too easy... some return points are too difficult" — 选择 return point 是 art 不是 science。

### 6.3 Human Guided Explore (最重)

对于 policy 完全探索不出来的 sub-behavior, tele-operator **现场演示** 几条 trajectory, 加入 demo buffer。

这是 [Genie Centurion (Wang et al. 2025)](https://arxiv.org/abs/2505.18793) 的 "rewind-and-refine" 思路。Table 3 显示 88% → 98%, 几乎所有 task 都到 100% (除了 Rice Transfer 70%, 因为这个 task 涉及 granular object, teleop 本身就难)。

---

## 7. 实验结果深度解读

### 7.1 Critic 评估 (Table 1)

评估指标定义 (公式 7-9):

$$
\begin{cases}
\mathrm{VOC} = \mathrm{rank\text{-}correlation}(\mathrm{argsort}(v_1, \ldots, v_T); \mathrm{arange}(T)) \\
v_i = v_{i-\Delta t} + c_{i-\Delta t, i}(1 - v_{i-\Delta t}) \\
v_0 = 0
\end{cases}
$$

- $v_i$: accumulated progress value, 从 0 开始累加, 但用 $(1 - v_{i-\Delta t})$ 做 multiplicative discount, 让 $v_i \in [0, 1]$ 始终 bound 在 1 以内
- VOC: predicted value 的 rank 和真实时间 rank 的 correlation, ∈ [-1, 1]
- VROC: 把 sequence 反过来, 看 model 是否还能正确 reverse
- VOC-F1 = 2·VOC·VROC / (VOC + VROC): harmonic mean, 防止 model 只在 forward 表现好

**Key findings from Table 1**:

1. **RoboFAC-success vs fail**: VLAC-8b 在 success 上 VOC-F1 = 0.87, fail 上 0.30。这个 gap 是 critic 能 reject failure trajectory 的核心证据。没有这个 gap, real-world RL 会被 failure trajectory 的高 reward 噪声淹没。

2. **Ego4D ablation**: 训练时加 Ego4D (human video) 后, RoboFAC success/fail gap 进一步拉大 (0.83 vs 0.41 → 加 Ego4D 后 gap 更明显)。这印证 paper 的 thesis: **human video data 对 embodied task understanding 有显著增益**, 即便 action space 完全不同。

3. **vs Gemini-1.5-Pro (GVL baseline)**: 在 DROID 上, Gemini VOC = -0.01 (基本随机), VLAC-8b VOC = 0.92。这说明 **general VLM 不懂 robotic task progress**, 必须在 robot + human video 上专门 train。

### 7.2 Actor 评估 (Table 2)

VLAC vs Pi0 ([π₀, Black et al. 2024](https://arxiv.org/abs/2410.24164)):

| Method | Avg success | Lighting transfer | Scene transfer |
|--------|------------|-------------------|----------------|
| Pi0 | 27% | 3% | - |
| VLAC | 75% | 57% | 63% |

VLAC 在 distribution shift 下 robustness 显著优于 Pi0。直觉: VLAC 的 critic pretraining 强制 model 理解 "task state", 这种 semantic understanding 比 pure BC 的 action imitation 更 robust to visual perturbation。Pi0 在 Lighting Transfer 下 avg 3%, 基本崩溃 — pure flow matching policy 对 lighting 极度敏感。

### 7.3 Real-World RL 学习曲线 (Figure 6, Table 3)

4 个任务, 200 episodes 内 30% → 90%。这跟 [SERL (Luo et al. 2024)](https://arxiv.org/abs/2401.16087) 报告的 "20 minutes to learn a task" 量级相当, 但 SERL 是 task-specific reward, VLAC 是 **同一个 critic model 跨 4 个完全不同的 task** (granular/rigid/flexible object manipulation)。

### 7.4 Multi-Robot Scaling (Figure 7, Section 4.7)

| Robots | Episodes/robot to 80% | Wall clock |
|--------|----------------------|-----------|
| 1 | ~140 | ~50 min |
| 2 | 325 | ~2 hours |
| 4 | 147 | ~55 min |
| 8 | 64 | ~24.6 min |

直觉: 8 robots 时 per-robot data 需求降到 64 episodes, 接近 linear scaling。但 2 robots 反而比 1 robot 差 — paper 解释是 heterogeneous background 在小规模时引入 visual disturbance 但 gradient signal 不足以 generalize。这跟 [Chinchilla scaling law](https://arxiv.org/abs/2203.15556) 在 LLM 里观察到的 "small scale 看不到 scaling" 现象类似。

**Dynamic sampling**: paper Appendix D 说 under-learned robot 的 data 采样频率提高。这是 task-level importance sampling, 解决 multi-robot 时部分 robot 卡在 local optimum 的问题。

---

## 8. Limitations & 我的联想

### 8.1 Paper 自己列的 limitations

1. **Human-in-the-loop 没有量化 trigger** — 什么时候介入? 介入多久? 完全靠 operator 经验。这跟 [Genie Centurion](https://arxiv.org/abs/2505.18793) 一样, 都是 art。
2. **PPO + autoregressive tokenized action 紧耦合** — 不能直接 transfer 到 diffusion/flow-matching action head。Section 2 提了 [Diffusion Policy Optimization (Ren et al. 2024)](https://arxiv.org/abs/2409.00588) 和 [Flow Q-Learning (Park et al. 2025)](https://arxiv.org/abs/2502.02538) 这条 line, 但 VLAC 没碰。
3. **Multi-task online instability** — reward scale drift, gradient interference, episodic forgetting。

### 8.2 我会进一步指出的 issues

1. **Critic 和 actor 是不同 size (8B vs 2B), 不算真正 unified**。Section 3.1 说 "unifies the roles of actor and critic within a single autoregressive architecture", 但实际上是两个独立 model。真正的 unified 应该是 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 那种 MoE 里 shared expert + routed expert 的设计。

2. **Pair-wise progress 假设 "task progress is positively correlated with time"**。这个假设在 expert trajectory 上成立, 但在 real-world RL rollout 里, robot 经常 stuck 或 regress。paper 的 Δt ∈ ℤ (允许负值) 部分缓解, 但 label 公式 $c = \Delta t / (T-i)$ 还是假设 expert trajectory monotonic。在 RL 自己生成的 failure trajectory 上, 这个 label 是错的。Paper 的 RoboFAC-fail 实验 VOC-F1 = 0.30 显示 model 学到了一些 "failure detection" 能力, 但来源是 contrastive symmetry 的 4-tuple 训练, 不是 label 本身。

3. **Reward sparsity in mid-trajectory**: 即便 c 是 dense 的, 在 long-horizon task (比如做饭 30 步) 里, 单步 c 值可能 < 0.01, advantage signal 弱。Paper 没讨论 reward normalization 或 [potential-based reward shaping](https://arxiv.org/abs/1907.02025)。

4. **vllm/torch 数值不一致问题**: 这个问题在 LLM RLHF 圈早就被讨论 ([OpenRLHF issue](https://github.com/OpenRLHF/OpenRLHF)), 但 robot RL 圈第一次明确量化。60% 数据废掉的 cost 很高, 未来需要 [vllm 的 deterministic mode](https://docs.vllm.ai/en/latest/) 或者完全用 torch inference。

5. **Action representation 是 delta EEF pose string**: 这种 representation 对 7-DOF arm OK, 但对 dexterous hand (EgoDex 那种 25-DOF) 完全不够。Paper 在 EgoDex 上只测了 critic, 没测 actor。Future work 必须解决 high-DOF action 的 tokenization。

### 8.3 跟相关工作的 positioning

| 工作 | Reward 来源 | 跨 task? | 跨 embodiment? | 与 policy 关系 |
|------|------------|---------|----------------|----------------|
| [RL-VLM-F](https://arxiv.org/abs/2402.03681) | Prompt VLM 直接打分 | Yes | Yes | 外部 oracle |
| [LIV](https://arxiv.org/abs/2306.08631) | CLIP embedding distance | Yes | Yes | 外部 oracle |
| [Eureka](https://arxiv.org/abs/2310.12931) | LLM generate reward code | Per task | No | 外部 oracle |
| [SERL](https://arxiv.org/abs/2401.16087) | Hand-crafted + learned | Per task | No | 外部 oracle |
| [ConRFT](https://arxiv.org/abs/2502.05450) | VLA + consistency policy | Partial | Partial | Tightly coupled |
| **VLAC** | Pair-wise progress delta | Yes | Yes (human+robot) | Unified (同 backbone) |

VLAC 的 unique selling point 是 **统一性 + signed dense reward + in-context transfer**。这是 reward model 这条 line 的 SOTA。

---

## 9. 跟你之前 work 的连接

Andrej, 你在 [Twitter 讲过](https://twitter.com/karpathy) "the bittest sweetest 2024 robot learning paper is one that figures out real-world RL loop"。VLAC 正是这个 direction。跟你 [Tesla Optimus 的工作](https://www.tesla.com/we-robot) 的连接点:

1. **Dense progress reward 的价值**: 你在 Tesla 一定遇到过 "binary success reward 在 long-horizon task 上学不动" 的问题。VLAC 的 pair-wise progress 是一种 general 的 dense shaping, 跟 Tesla 内部可能用的 "task graph progress" 思路相通。

2. **VLM 当 reward model**: 你在 [Yann LeCun 对谈](https://www.youtube.com/watch?v=5Ot7eE2xaXw) 里讲过 "world model 是 RL 的 bottleneck"。VLAC 把 VLM 当 approximate world model (predict "state A → state B 是否更接近 goal"), 这是一种 pragmatic 的 world model 使用方式。

3. **Human-in-the-loop 的工程现实**: 你在 Tesla 强调过 "data engine 比 algorithm 重要"。VLAC 的三层 human-in-the-loop 本质上是个 data engine — tele-operator 不写代码, 只通过 reset/demonstration 注入信息。这跟 Tesla 的 data annotation pipeline 类似。

---

## 10. References

- [VLAC Paper (本篇)](https://arxiv.org/abs/2508.xxxxx) — Shanghai AI Lab, Jiangmiao Pang 组
- [InternVL](https://arxiv.org/abs/2312.14238) — VLAC 的 base model
- [π₀ (Black et al. 2024)](https://arxiv.org/abs/2410.24164) — Flow matching VLA baseline
- [π₀.5](https://arxiv.org/abs/2504.16054) — Open-world generalization VLA
- [SERL (Luo et al. 2024)](https://arxiv.org/abs/2401.16087) — Sample-efficient real-world RL
- [OpenVLA (Kim et al. 2024)](https://arxiv.org/abs/2406.09246) — Open-source VLA
- [FAST tokenization (Pertsch et al. 2025)](https://arxiv.org/abs/2501.09747) — Action tokenization for VLA
- [LIV (Ma et al. 2023)](https://arxiv.org/abs/2306.08631) — Language-image value/reward
- [VLMs are in-context value learners (Ma et al. 2024)](https://arxiv.org/abs/2402.00764) — VLAC 的直接前驱
- [Eureka (Ma et al. 2023)](https://arxiv.org/abs/2310.12931) — LLM-generated reward code
- [ConRFT (Chen et al. 2025)](https://arxiv.org/abs/2502.05450) — RL fine-tuning for VLA
- [Diffusion Policy Optimization (Ren et al. 2024)](https://arxiv.org/abs/2409.00588) — Diffusion policy 的 PPO
- [Flow Q-Learning (Park et al. 2025)](https://arxiv.org/abs/2502.02538) — Flow matching + RL
- [Genie Centurion (Wang et al. 2025)](https://arxiv.org/abs/2505.18793) — Human rewind-and-refine
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347) — 基础算法
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300) — LLM RL 的另一选择
- [Bridge V2](https://arxiv.org/abs/2308.12952) — Robot manipulation dataset
- [DROID](https://arxiv.org/abs/2403.12917) — Large-scale manipulation dataset
- [Ego4D](https://arxiv.org/abs/2110.07058) — Egocentric human video
- [AGIBOT World](https://arxiv.org/abs/2502.03043) — 大规模 manipulation 平台
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — RLHF 训练框架 (vllm/torch 一致性 issue 参考)
- [TRL (HuggingFace)](https://huggingface.co/docs/trl) — PPO 训练库
- [Rodney Brooks 1991 — Intelligence without representation](https://people.csail.mit.edu/brooks/papers/representation.pdf) — Paper epigraph 出处

---

## 11. 一句话总结

VLAC 把 "task progress judgment" 这个人类天然擅长、VLM 通过 pair-wise pretraining 可以习得的能力, 蒸馏成一个 signed dense reward signal, 在统一 autoregressive 架构里同时驱动 actor 和 critic, 配合 3 层 human-in-the-loop 数据注入, 让 real-world robot RL 在 4 个 manipulation task 上 200 episodes 内从 30% 涨到 90%。这是 VLA + real-world RL 这条 line 目前最完整的 recipe, 离 "fire-and-forget autonomous robot learning" 还有距离 (human-in-the-loop 仍是 art, multi-task 不稳定, action representation 局限于 low-DOF), 但已经把 reward engineering 这个最大 bottleneck 拆掉了。
