---
source_pdf: Act2Goal.pdf
paper_sha256: 8b27463858672122c331db3b8be6ad8ecaea0b8a6d033ba5e4d28ae429d22308
processed_at: '2026-07-18T00:41:54-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Act2Goal 深度技术讲解

Andrej，这篇 paper 我会从 intuition、architecture、formula、training、experiments 几个维度拆开讲，尽量把每个设计选择背后的 reasoning 都挖出来。

---

## 1. Paper 的核心 motivation

传统的 goal-conditioned policy (GCP) 把 (current observation, goal image) 直接 map 到 action，这种 single-step action prediction 在 long-horizon 任务下会崩，原因是缺乏 explicit 的 task progress 表示。Policy 只能 overfit demonstration 中的 state-action pairs，没有 intermediate guidance，所以 OOD 场景下很容易迷路。

Act2Goal 的核心 thesis 是：**让 policy 先"想象"从 current observation 到 goal 之间的 visual trajectory，再把这个想象作为 action expert 的条件**。这相当于给 policy 配了一个"mental simulator"，让它不依赖 instantaneous state-action correlation，可以 long-horizon 推理。

这和你的 blog post "State of the Art of RL in LLMs" 里提到的 "world model as a learned simulator" 思路一致——把 dynamics 从 policy 里 decouple 出来，让 policy 学会"在脑中预演"。

Reference:
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Dreamer (Hafner et al.): https://arxiv.org/abs/1912.01603
- Hindsight Experience Replay: https://arxiv.org/abs/1707.01495

---

## 2. Architecture 拆解

整个系统分两个 transformer-like 模块（Fig. 3）：

### 2.1 Goal-Conditioned World Model (GCWM)

基于 Genie Envisioner architecture（作者团队前作，1.6B parameters），做了一个关键修改：**移除 language conditioning，加入 goal visual condition**，变成 pure vision-based world model。

输入：
- Multi-view current observation frames → VAE encoder → latents $z_t$
- Multi-view goal frames → VAE encoder → latents $z_g$
- Noisy latents $\epsilon$（与 $z_{pred}$ 同 shape）

输入沿 hidden states sequence 维度 concat，通过 Video DiT blocks refine 成 MSTH-structured latent frames。

### 2.2 Action Expert

Architecture 与 world model **isomorphic**（同结构、同 DiT block 数量，但 width 减少），这是关键设计——让两个 model 的 layer-wise features 可以直接 cross-attention 对齐。

Action expert 输入：
- Robot proprioceptive state $c_p$
- World model 的 layered transition features $c_w = \{h_{world}^1, \ldots, h_{world}^L\}$（每一层 DiT block 的 hidden state）
- Noise $\zeta$

通过 cross-attention 把 world model 的 visual representation 注入 action 生成。这种 layer-wise cross-attention 比 single-feature injection 更强，因为不同层捕获不同 abstraction level 的信息（浅层 edge/texture，深层 semantic）。

### 2.3 具体数字（Appendix A）

- World model predict 4 个 latent frames（2 proximal + 2 distal）
- 3D VAE decoder 解码成 9 个 proximal visual frames + 9 个 distal visual frames
- Action expert 输出 54 个 proximal actions（执行前 50）+ 9 个 distal actions（只做 guidance，不执行）
- Inference latency 200ms / 50 actions ≈ 4ms/action，闭环控制够用
- 1.6B Genie Envisioner pretrained，Stage 1 fine-tune 7×24 hours on 16×A800

Reference:
- Genie Envisioner: https://arxiv.org/abs/2508.05635
- DiT (Diffusion Transformers): https://arxiv.org/abs/2212.09748
- Flow Matching: https://arxiv.org/abs/2210.02747

---

## 3. Flow Matching 数学详解

Act2Goal 用的是 continuous flow matching（Lipman et al. 2023 风格），比 DDPM-style diffusion 更简洁，inference 时是 deterministic ODE。

### 3.1 World Model 公式

**Eq. 1**: $z_{pred} = f_\theta(z_t, z_g, \epsilon)$

变量解释：
- $z_t$: current observation 经过 VAE 压缩后的 latent（$t$ = "current time"，下标表 timestamp）
- $z_g$: goal image 的 VAE latent（$g$ = "goal"）
- $\epsilon$: 标准高斯 noise，shape 与 $z_{pred}$ 相同
- $f_\theta$: flow matching network（参数 $\theta$），把 noise 变换成 structured visual sequence latent
- $z_{pred}$: 预测的中间 visual states latent sequence

**Eq. 2**: $z^{(n+1)} = z^{(n)} + \frac{1}{N} v_\theta(z^{(n)}, z_t, z_g)$

变量解释：
- $z^{(n)}$: 第 $n$ 步的 latent 状态，$z^{(0)} = \epsilon$
- $N$: total refinement steps
- $v_\theta$: 学习的 vector field，给定当前 latent + current obs + goal，输出更新方向
- $\frac{1}{N}$: step size，相当于 Euler method 的 step

这就是 ODE $\dot{z} = v_\theta(z, z_t, z_g)$ 的 Euler 离散化。和 DDPM 的 stochastic reverse process 不同，flow matching 是 deterministic 的，inference 更快也更稳定。

### 3.2 Action Expert 公式

**Eq. 3**: $a_{pred} = g_\phi(c_w, c_p, \zeta)$

变量解释：
- $g_\phi$: action flow matching network（参数 $\phi$）
- $c_w = \{h_{world}^1, \ldots, h_{world}^L\}$: world model $L$ 层 DiT block 的 hidden states 集合
- $c_p$: robot proprio state（关节角度等）
- $\zeta$: action noise
- $a_{pred}$: 预测的 action sequence

**Eq. 4**: $a^{(n+1)} = a^{(n)} + \frac{1}{N} u_\phi(a^{n}, c_w, c_p)$

变量解释：
- $a^{(n)}$: 第 $n$ 步 action latent
- $u_\phi$: action vector field
- 其余与 Eq. 2 对称

直觉：action 生成也是 flow matching，这意味着 action trajectory 不是 regression 出来的一个 point，而是从一个 distribution 里 sample 出来的 trajectory，可以 capture multi-modality（同一 goal 可以有多种完成路径）。

Reference:
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Stochastic Interpolants: https://arxiv.org/abs/2303.08797

---

## 4. Multi-Scale Temporal Hashing (MSTH) — 这是 paper 最有 idea 的部分

### 4.1 直觉

Long-horizon manipulation 有两个矛盾需求：
- **Local reactivity**: 闭环控制要 dense，每 step 都要 respond to 当前 observation
- **Global consistency**: 长程 goal 不能丢，不能被 local noise 带偏

Single-scale chunk（比如固定 predict 16 个 actions）会有问题：
- Chunk 太短：长程 goal 漂移
- Chunk 太长：闭环反应慢，对 perturbation 脆弱

MSTH 的解法是 **logarithmic temporal sampling**——近端 dense，远端 sparse，且 sparse 的间隔 logarithmic 增长。

### 4.2 公式

**Eq. 5**: $d_m = P + \lfloor \frac{K - P}{\log(M+1)} \cdot \log(m+1) \rfloor$, $m = 1, \ldots, M$

变量解释：
- $K$: total imagined trajectory length（比如 100 帧）
- $P$: proximal horizon（dense 段长度，比如 20 帧）
- $r$: vision sampling stride（在 proximal 段内每 $r$ 帧采一次 visual state）
- $M$: distal frame 数量
- $m$: distal frame 索引（1 到 $M$）
- $d_m$: 第 $m$ 个 distal frame 的 timestep index
- $\lfloor \cdot \rfloor$: floor function

举例：$K=100, P=20, M=5$
- $d_1 = 20 + \lfloor 80/\log(6) \cdot \log(2) \rfloor = 20 + \lfloor 44.7 \cdot 0.693 \rfloor = 20 + 30 = 50$（实际数字略不同，但思路是间隔递增）
- $d_2 \approx 20 + \lfloor 44.7 \cdot 1.099 \rfloor = 20 + 49 = 69$
- $d_3 \approx 20 + \lfloor 44.7 \cdot 1.386 \rfloor = 20 + 61 = 81$
- $d_4 \approx 20 + \lfloor 44.7 \cdot 1.609 \rfloor = 20 + 71 = 91$
- $d_5 \approx 20 + \lfloor 44.7 \cdot 1.792 \rfloor = 20 + 80 = 100$

可以清楚看到 distal 间隔是 logarithmic 增长的：30 → 19 → 12 → 10 → 9。**这正好对应未来不确定性随时间指数增长**——越远的未来越不需要 precise 时间锚定，只需要 coarse 方向。

### 4.3 Vision vs Action 的不对称设计

这是个很巧妙的细节：
- **Proximal vision**: stride $r$ 采样（比如每 5 帧采 1 个 visual state）——视觉不需要每帧
- **Proximal action**: 每 timestep 都 predict（dense）——action 必须每帧执行
- **Distal vision**: logarithmic 采样——global guidance
- **Distal action**: 与 distal vision 对齐——latent guidance，不执行

这种不对称反映了"视觉是 sparse guidance，action 是 dense control"的本质。Diffusion Policy 里的 action chunking 是 single-scale 的，MSTH 是 multi-scale 的 abstraction。

### 4.4 Ablation 数据（Table III）

Whiteboard writing task：

| Setting | Short (≤3) | Medium (4-6) | Long (≥7) |
|---|---|---|---|
| ID w/o MSTH | 0.95 | 0.35 | 0.10 |
| ID w/ MSTH | 0.95 | 0.90 | 0.90 |
| OOD w/o MSTH | 0.60 | 0.20 | 0.00 |
| OOD w/ MSTH | 0.93 | 0.90 | 0.88 |

**关键观察**：
- Short words: MSTH 几乎没影响（因为短任务不需要 long-horizon reasoning）
- Long words OOD: 没有 MSTH 完全失败 (0.00)，有 MSTH 达 0.88

这说明 MSTH 真的是为 long-horizon 而生，short-horizon 任务它不增加 overhead 也不掉点。

Reference:
- Diffusion Policy action chunking: https://diffusion-policy.cs.columbia.edu/
- ACT (Action Chunking with Transformers): https://tonyzhaozh.github.io/aloha/

---

## 5. Two-Stage Offline Training

### 5.1 Stage 1: Joint training

Loss:
- **Eq. 6**: $\mathcal{L}_v = \mathbb{E}_{t \sim U(0,1), z_0, z_1, z_t, z_g}[\| v_\theta(t, \phi_t(z), z_t, z_g) - (z_1 - z_0) \|^2]$

变量解释：
- $t \sim U(0,1)$: flow matching time variable，从 0 到 1 均匀采样
- $z_0, z_1$: 起点和终点的 latent（$z_0$ 是 noise，$z_1$ 是 target visual sequence）
- $\phi_t(z)$: linear interpolant $\phi_t(z) = (1-t) z_0 + t z_1$
- $z_t$: current observation latent（注意不要和 flow time $t$ 混淆，paper 用了同一个符号有点 ambiguous）
- $z_g$: goal latent
- $v_\theta$: vector field network
- $(z_1 - z_0)$: target direction（linear path 的 tangent）

- **Eq. 7**: $\mathcal{L}_a = \mathbb{E}_{t \sim U(0,1), a_0, a_1, c_w, c_p}[\| u_\phi(t, \psi_t(a), c_w, c_p) - (a_1 - a_0) \|^2]$

变量解释对称：$a_0$ 是 action noise，$a_1$ 是 target action sequence，$\psi_t(a)$ 是 interpolant。

- **Eq. 8**: $\mathcal{L}_{stage1} = \mathcal{L}_v + \lambda \cdot \mathcal{L}_a$, $\lambda = 0.1$

$\lambda = 0.1$ 意味着 visual loss 权重更高，因为 world model 是 foundation，action expert 是下游。Stage 1 让 world model 学会生成 actionable trajectory，不只是 plausible video。

### 5.2 Stage 2: End-to-end BC

$\mathcal{L}_{stage2} = \mathcal{L}_a$ only

冻结视觉部分？没有，是 end-to-end，gradient 从 action loss 回传到 world model，让 visual representation 为 action 而 optimize。这是 representation learning for control 的经典 trick——让 perception 为下游 task 服务，而不是为 reconstruction 服务。

直觉：Stage 1 学"看得见的未来"，Stage 2 学"为 action 服务的未来"。两阶段 decouple 是为了 stability，否则 end-to-end 从头训 world model 容易 collapse。

---

## 6. Online Autonomous Improvement (HER + LoRA)

### 6.1 Algorithm 1 步骤解读

```
1) Initialize replay buffer B, LoRA params φ
2) While not converged:
   a) Execute policy one episode, store (o, c_p, a, o') at each step
   b) For each transition: B ← B ∪ {(o, c_p, a, o')}
   c) If |B| ≥ N:
      i) For k = 1 to K:
         A) Sample batch from B
         B) g' ← o'  // Hindsight relabel
         C) L = E[‖π_θ(o, c_p, g') - a‖²]
         D) φ ← φ - α∇_φ L
      ii) B ← ∅
```

关键点：
- **Hindsight relabeling**：无论原本的 goal 是什么，把实际到达的 observation $o'$ 当作新 goal $g'$。这把"失败 rollout"转成"成功 trajectory"——只是 goal 不一样了。
- **LoRA finetuning**：只更新 LoRA layer（rank 64），base model frozen。这让 on-robot 训练变得可行——RTX 4090 上 5 分钟一轮。
- **Buffer clear**：每轮训练后清空 buffer，避免 stale data 干扰。这是个 engineering 选择，可能 lose some data efficiency 但保证 fresh。

### 6.2 为什么这 work

GCP 的特殊性质：**任何 achievable state 都可以做 goal**。这和 task-conditioned policy 不同——task 必须有 reward，但 goal 是 visual state，自然就是 self-supervised。

HER 的精髓是 "failure is just success at a different goal"。一个机器人抓不到红色杯子但抓到了蓝色杯子，那这个 trajectory 就是"如何到达蓝色杯子"的 demo。world model 可以学习这种"如何到达任意 visual state"的 universal dynamics。

### 6.3 数据效率

实验显示 (Fig. 6 right)：
- All rollouts > Successful only > Failed only
- 即使只用 failed rollouts 也能显著提升

这说明 HER 的核心价值在 failed data——成功数据本身已经被 demo 覆盖了，failed rollout 才是 exploration 的真正 signal。

Reference:
- HER original: https://arxiv.org/abs/1707.01495
- LoRA: https://arxiv.org/abs/2106.09685
- DAgger: https://arxiv.org/abs/1011.0552

---

## 7. Experiments 深度分析

### 7.1 RoboTwin 2.0 Simulation (Table I)

| Mode | Model | Move Can | Pick Bottles | Place Cup | Place Shoe |
|---|---|---|---|---|---|
| Easy | DP-GC | 0.18 | 0.04 | 0.03 | 0.04 |
| Easy | π0.5-GC | 0.54 | 0.13 | 0.16 | 0.30 |
| Easy | HyperGoalNet | 0.11 | 0.08 | 0.08 | 0.01 |
| Easy | **Act2Goal** | **0.62** | **0.80** | **0.64** | **0.52** |
| Hard | DP-GC | 0.00 | 0.00 | 0.00 | 0.00 |
| Hard | π0.5-GC | 0.42 | 0.06 | 0.04 | 0.06 |
| Hard | HyperGoalNet | 0.00 | 0.00 | 0.00 | 0.00 |
| Hard | **Act2Goal** | **0.13** | **0.43** | **0.13** | **0.15** |

观察：
- Easy mode: Act2Goal 普遍 2-3× baselines
- Hard mode: 大部分 baselines 直接 0.00，Act2Goal 仍有 0.13-0.43
- Pick Bottles Hard mode 0.43 vs π0.5-GC 0.06 是 7× 差距

Hard mode 的巨大 gap 说明 world model 的 imagination 真的提供了 OOD generalization 的 capability——baselines 只能 mimic training distribution，Act2Goal 能 reason 出新场景的 visual path。

### 7.2 Real-World Tasks (Table II)

三个任务设计很 clever：
- **Whiteboard Word Writing**: 测 compositional generalization（OOD 是没见过的 word combination）
- **Dessert Plating**: 测 visual distraction robustness
- **Plug-In Operation**: 测 skill transfer（metal workpiece → bottle）

OOD 成功率：0.90 / 0.48 / 0.30——Whiteboard 写字 OOD 几乎和 ID 一样好（0.93 vs 0.90），这说明 compositional generalization 是 world model 的强项——它能 imagine "letters 组成新 word" 的 visual trajectory。

Plug-In OOD 0.30 看似低，但 baselines 全是 0.00，相对差距巨大。

### 7.3 Online Improvement (Fig. 6, 7)

- Simulation: 3 rounds 收敛，最高 8× improvement
- Real-world Whiteboard drawing: 15 minutes 从差到好
- Real-world Plug-In OOD: 0.30 → 0.90（3×）in minutes

这个 data efficiency 令人印象深刻。原因是 LoRA 参数量小（rank 64），replay buffer 小（size 20），每轮只 10 epochs，所以 on-robot training 真的可行。这是 embodied AI 走向 self-improving agent 的关键 step。

Reference:
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- AgiBot World: https://arxiv.org/abs/2503.06669
- π0.5: https://arxiv.org/abs/2504.16054

---

## 8. Related Work 的 Positioning

### 8.1 vs Language-conditioned policies (RT-2, OpenVLA, π0)

Language 是 high-level instruction，缺乏 fine-grained spatial precision。"把杯子放在那里" 不如直接给一张目标图。Act2Goal 用 visual goal 绕开了 language ambiguity。

### 8.2 vs Diffusion-based GCPs (GoalGAIL, score-based DP)

这些方法直接从 (obs, goal) → action，没有 intermediate visual reasoning。Act2Goal 多了一层 world model imagination，提供了 temporal structure。

### 8.3 vs CoA (Chain of Action)

CoA 从 goal keyframe 反向生成 action sequence 保持 long-horizon consistency。Act2Goal 是 forward generation + MSTH，更接近 receding horizon control 的思路。

### 8.4 vs WorldVLA, GE-Act, VidMan

这些是 language-conditioned world model + action expert。Act2Goal 是 goal-conditioned，更纯粹——不依赖 language parser，直接用 visual goal。

### 8.5 vs Mimicplay, Track2Act

这些用 point tracking 或 human play video 提供 guidance。Act2Goal 用自己生成的 video trajectory，更 self-contained。

Reference:
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- CoA: https://arxiv.org/abs/2506.09990
- WorldVLA: https://arxiv.org/abs/2506.21539
- MimicPlay: https://arxiv.org/abs/2302.12422
- Track2Act: https://arxiv.org/abs/2409.17543

---

## 9. 我的 Intuition 总结

### 9.1 为什么 World Model 是 GCP 的天然补集

GCP 的 failure mode 是"看到 goal 但不知道中间路径"。World model 正好补上这层——它专门学"从 A 到 B 的中间长什么样"。两者结合是 complementary 的：world model 提供"地图"，action expert 提供"驾驶"。

### 9.2 为什么 MSTH 的 logarithmic spacing 是对的

未来不确定性 ~ $\exp(\alpha t)$ 增长（chaotic dynamics、perturbation accumulation）。Logarithmic sampling 的信息密度 ~ $\log(t)$，正好匹配"近端要精，远端要粗"的认知需求。这和人类 motor control 的 multi-timescale planning 类似——打网球时近端肌肉控制 dense，远端战术 planning sparse。

### 9.3 为什么 HER + LoRA 在 GCP 上特别 work

GCP 的 goal 是 visual state，所以任何 achieved state 都是 valid goal——这是 HER 的天然 setting。LoRA 让 online update 在 robot 上的 RTX 4090 都能跑。两者结合是 self-improving robot 的 minimum viable architecture。

### 9.4 还有什么值得探索

- **Hierarchical world model**: 当前 MSTH 是 hand-crafted 的 logarithmic spacing，是否可以 learnable hierarchy？
- **Counterfactual world model**: 让 world model imagine "如果 action 是 X，会怎样"——这接近 model-based RL 的 MPC setting。
- **3D world model**: 当前是 2D video latents，3D Gaussian Splatting world model 可能给更精确的 spatial reasoning。
- **Multi-agent goal**: 当前是 single robot + single goal，扩展到 multi-robot shared goal。
- **Language + visual goal hybrid**: 让 language 提供 task category，visual goal 提供 precise target，可能是更实用的 interface。

### 9.5 局限性（paper 没明说但值得注意）

- World model 的 visual fidelity 限制——如果生成的 trajectory 不够 accurate，action expert 会被误导
- 200ms inference latency 对高频 contact-rich task（比如插入 1.5cm hole）可能不够 tight
- Real-world online improvement 需要 manual reset，并非完全 autonomous
- Distal action 不执行只做 guidance，但 paper 没分析 distal action 的 ablation

---

## 10. 对你（Karpathy）的关联思考

你之前在 "Software 2.0" 和 "State of SOTA RL in LLMs" 里讨论过 neural network 作为 learned program 的 idea。Act2Goal 是这个思路在 robotics 上的具象化——world model 是"learned physics engine"，action expert 是"learned controller"，HER+LoRA 是"learned online patch"。

和你最近推的 "vibe coding" 概念也有 resonance——这里的 visual goal 就是"vibe specification"，policy 负责把它 operationalize。MSTH 给了一种 hierarchical 的 vibe decomposition：近端 vibe 是 dense motor primitive，远端 vibe 是 coarse task direction。

如果你要 critiquing 这篇 paper，我猜你会问：
- 为什么不直接 train 一个 universal policy（image+goal → action）end-to-end？答案是 long-horizon 的 data efficiency——explicit world model 提供 auxiliary supervision signal
- 为什么 flow matching 不用 diffusion？答案是 inference speed 和 conditioning 的 simplicity
- 为什么 LoRA 不更新 full model？答案是 on-robot compute budget

希望这个讲解 build up 了你对 Act2Goal 的 intuition。如果你想 deep dive 任何一个 component（比如 flow matching 数学、Genie Envisioner architecture 细节、HER 的 sampling 策略），告诉我，我可以再展开。

Reference summary:
- Project page: https://act2goal.github.io/
- Flow Matching: https://arxiv.org/abs/2210.02747
- HER: https://arxiv.org/abs/1707.01495
- LoRA: https://arxiv.org/abs/2106.09685
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Genie Envisioner: https://arxiv.org/abs/2508.05635
- π0.5: https://arxiv.org/abs/2504.16054
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- AgiBot World: https://arxiv.org/abs/2503.06669
