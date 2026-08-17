---
source_pdf: Latent Reasoning VLA.pdf
paper_sha256: 6c63ec5bd22405710cf6cba14cd11948ef2d0e598e4f5e6cb8053c220854a514
processed_at: '2026-08-05T12:12:06-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LaRA-VLA 用人话讲

## 一句话总结

**让 robot 在脑子里 "想" 完之后再动手，但这个 "想" 发生在 neural network 的 hidden state 里，不变成人类可读的文字。**

---

## 1. 为什么需要这个东西

想象你给 robot 一个指令："把红色方块放到篮子里"。

一个 dumb robot policy 就是：**看到像素 → 直接输出关节角度**。这种 end-to-end 在简单任务上 work，但 long-horizon 任务（比如 "先找红块、再抓、再放篮子"）会失败，因为缺乏中间 reasoning。

所以大家开始给 VLA 加 chain-of-thought (CoT) — 让 model 先 generate 一段文字 reasoning（"我看到红块在左边，需要先移动到那里，然后闭合夹爪..."），再 output action。

**问题来了**：

- **慢**: generate 几百个 text token 要几百毫秒，robot 需要 10-50Hz control，根本跟不上。Physical 机器人 control latency 超过 100ms 就很难做 fine-grained manipulation。
- **不对齐**: text token 是 discrete 的，但 perception (RGB pixels, depth) 和 action (joint torques, end-effector poses) 都是 continuous 的。用 discrete token 表达 reasoning 就是 representation mismatch。

LaRA-VLA 的核心 insight：**CoT 有效的原因是它提供了 structured intermediate computation，而不是因为它用 English 表达。** 如果把 structure 压到 continuous latent 里，就能既保留 reasoning 的 benefit 又解决 latency 和 mismatch。

---

## 2. 核心方法: 三阶段训练

### Stage 1: 教 model 用 English reasoning

先在 Qwen3-VL (https://arxiv.org/abs/2511.21631) 上 fine-tune，给它 standard CoT supervision。输入是 image + instruction，output 是 explicit text CoT + predicted next-frame visual latent + action tokens。

Loss 是三个加起来：

$$\mathcal{L}_{\text{Stage I}} = \mathcal{L}_{\text{cot}} + 0.1 \cdot \mathcal{L}_{\text{vis}} + \mathcal{L}_{\text{act-dis}}$$

- $\mathcal{L}_{\text{cot}}$: text CoT 的 next-token prediction loss（cross-entropy）
- $\mathcal{L}_{\text{vis}}$: predicted visual latent 和真实 next frame latent 的 L1 距离
- $\mathcal{L}_{\text{act-dis}}$: action tokens 的 autoregressive loss

这里关键 design choice：**visual latent 用 EMA encoder 产生**。

$$\bar{\theta}_v^t = \tau_v \cdot \bar{\theta}_v^{t-1} + (1-\tau_v) \cdot \theta_v^t$$

- $\bar{\theta}_v^t$: 第 $t$ 步的 EMA target encoder 参数
- $\theta_v^t$: online encoder 在第 $t$ 步的参数
- $\tau_v$: decay rate（接近 1，比如 0.99）

为什么需要 EMA？如果 target 和 predictor 是同一个 encoder，会发生 representation collapse — encoder 输出常数，predictor 也输出常数，loss 为 0 但什么都没学到。EMA 让 target "slow-moving"，提供稳定 anchor，这是 BYOL (https://arxiv.org/abs/2006.07733) 和 MoCo (https://arxiv.org/abs/1911.05722) 的核心 trick。

**Intuition**: 这一步相当于让 model 学会 "用 English 表达 reasoning + 想象下一帧画面 + 决定动作"。

### Stage 2: 慢慢把 English "压缩" 到 latent

直接 hard switch 到 latent training 会 collapse — 没了 text supervision，latent 不知道该 encode 什么。

所以用 curriculum：开始时 100% text CoT tokens，然后逐渐 mask 掉一部分，用 learnable latent token 替代。比如 25% → 50% → 75% → 100% masked。

Loss 仍然保留 visual + action supervision：

$$\mathcal{L}_{\text{Stage II, final}} = 0.2 \cdot \mathcal{L}_{\text{vis}} + \mathcal{L}_{\text{act-dis}}$$

注意 visual loss 权重从 0.1 提到 0.2，因为现在 visual prediction 是 latent reasoning 的主要 implicit supervision。

**Intuition**: 这一步相当于 "去掉 training wheels"。一开始 English text 像 scaffolding 帮 latent 学会 structure，然后慢慢撤掉 scaffolding，让 latent 自己维持 structure。Visual 和 action signal 是 "保底"，防止 latent 漂走。

这和 CoDi (https://arxiv.org/abs/2502.21074) 的 self-distillation 思路类似，但 LaRA-VLA 蒸馏的是 multi-modal latent，不只是 answer token。

### Stage 3: 接上 action expert 生成连续动作

前两阶段 action 用 autoregressive tokens，Stage III 换成 flow matching (https://arxiv.org/abs/2210.02747)，类似 π0 (https://arxiv.org/abs/2410.24164) 的 design。

Flow matching 的核心 idea：定义一条从 noise 到 action 的 "flow"：

$$\mathbf{a}_\tau = (1-\tau) \cdot \epsilon + \tau \cdot \mathbf{a}_t$$

- $\mathbf{a}_\tau$: flow time $\tau$ 处的 interpolated action
- $\tau$: flow time，$\tau \sim \mathcal{U}(0, 1)$ uniform sampled
- $\epsilon$: Gaussian noise $\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\mathbf{a}_t$: ground-truth action at time step $t$

Action expert（一个 16-layer Diffusion Transformer）预测 velocity field：

$$\mathcal{L}_{\text{act-con}} = \mathbb{E}_{\mathbf{a}_t, \epsilon, \tau} \left[ \left\| v_{\theta_a}(\mathbf{a}_\tau, \tau \mid \mathbf{h}_t) - (\mathbf{a}_t - \epsilon) \right\|_2^2 \right]$$

- $v_{\theta_a}(\cdot)$: action expert 预测的 velocity field
- $\theta_a$: action expert 参数
- $\mathbf{h}_t$: multi-modal latent context（从 VLM 来的）
- $(\mathbf{a}_t - \epsilon)$: target velocity（noise 到 action 的 constant velocity）

**Intuition**: Flow matching 学习一个 vector field，把随机 noise "流" 到 action distribution。比 diffusion 的 reverse SDE 简单稳定，training 效率高，π0 已经证明了。

$\mathbf{h}_t$ 聚合四样东西：
- Current visual latent（"现在画面长啥样"）
- Language instruction（"要干啥"）
- Text reasoning latent（"怎么分解任务"，从 Stage II 继承）
- Predicted future visual latent（"下一帧画面长啥样"）

这四样对应 robot control 的四个认知功能：感知现状、记住目标、规划方法、想象未来。

**关键 design**: 不需要额外 action latent。因为前两阶段的 inverse dynamics supervision 让 latent context 已经 encode 了 action-relevant 信息，直接从 shared latent 出 action 就行。

---

## 3. Attention 设计 (容易被忽略但很重要)

参考 Figure 3 的 attention pattern：

**Token types**:
- Text tokens（Stage I/II 是 instruction + CoT，Stage II/III 变成 text latents）
- Current image tokens
- Future image tokens（`<img next>`）
- Action tokens（Stage I/II 才有，Stage III 移除）

**Attention 规则**:
- Future image tokens: causal attend to text + current image，但 future tokens 之间 bidirectional（spatial coherence）
- Action tokens: attend to 所有前面的 text、current image、future image，加上之前生成的 action tokens（causal within action sequence）
- Stage III: action tokens 从 attention 中移除，只保留 text + vision reasoning

**为什么这么设计**：

1. Future prediction 依赖 current observation 和 reasoning，符合物理因果性
2. Action 依赖 reasoning 和 predicted future（考虑 consequence），不是直接从 raw pixels 出
3. Future tokens 之间 bidirectional 让 prediction 在 spatial 维度 coherent，不会出现 "左半边是手，右半边是桌子" 的荒谬预测

这种 design philosophy 和 Coconut (https://arxiv.org/abs/2412.06769) 的 latent thought 类似，但 LaRA-VLA 扩展到 multi-modal。

---

## 4. Data Pipeline

CoT annotation 需要三个 components:
1. **Subtask decomposition**（"先抓、再移动、再放"）
2. **Target object localization**（"红块在哪里"）
3. **Motion reasoning**（"往左前方移动"）

现有 pipeline 要么太冗余（ECoT https://embodied-chain-of-thought.github.io/ 标注所有物体的 bbox），要么缺失（Emma-x https://arxiv.org/abs/2502.01243 缺 target localization）。

LaRA-VLA 的 pipeline: **anchor-first, generate-later**

**Semantic anchors**: Qwen3-VL 从 first frame + instruction 识别 manipulated object。

**Temporal anchors**: 用 gripper state changes 把 trajectory 切成 atomic stages (pre-grasp, grasp, move, release)，boundaries 作为 keyframes。

**Generation**:
- Subtask description: Qwen3-VL 基于 instruction + keyframes 生成
- Target bbox: GroundingDINO (https://arxiv.org/abs/2303.05499) + SAM3 (https://arxiv.org/abs/2511.16719) 多帧 ensemble + 线性插值
- Motion reasoning: 从 end-effector trajectory 计算 global motion (toward segment goal) 和 local motion (instantaneous)，离散化为 directional descriptors

构建了两个 dataset：
- **LIBERO-LaRA** (基于 LIBERO https://libero-project.github.io/)
- **Bridge-LaRA** (基于 SimplerEnv https://simpler-env.github.io/)
- 加 real-robot demos

---

## 5. 实验结果

### LIBERO (Table 2)

LIBERO 4 个 task suite，每个 10 个 task，50 rollouts/task:

| CoT Type | Method | Spatial | Goal | Object | Long | Avg |
|----------|--------|---------|------|--------|------|-----|
| No CoT | OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| No CoT | π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| No CoT | OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| Text CoT | ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| Text CoT | π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| Text CoT | DeepThinkVLA | 99.0 | 96.6 | 96.4 | 96.2 | 97.0 |
| Vis CoT | DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| Vis CoT | F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| Vis CoT | UD-VLA | 94.1 | 95.7 | 91.2 | 89.6 | 92.7 |
| Latent | Fast-ThinkAct | 92.0 | 97.2 | 90.2 | 79.4 | 89.7 |
| **Latent** | **LaRA-VLA** | **96.4** | **98.6** | **99.8** | **96.6** | **97.9** |

LaRA-VLA 拿下 SOTA，特别 Object suite 99.8% (近乎完美)，Long suite 96.6% (long-horizon 最难，优势明显)。比同类的 Fast-ThinkAct 高 8.2 个点。

### SimplerEnv-WidowX (Table 3)

Real-to-sim generalization 测试：

| Method | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Avg |
|--------|-----------|------------|-------------|--------------|-----|
| OpenVLA | 0.0 | 0.0 | 0.0 | 4.1 | 1.0 |
| Octo | 47.2 | 9.7 | 4.2 | 56.9 | 29.5 |
| OpenVLA-OFT | 12.5 | 4.2 | 8.3 | 37.5 | 39.6 |
| π0 | 29.1 | 0.0 | 16.7 | 62.5 | 40.1 |
| CogACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| ThinkAct | 58.3 | 37.5 | 8.7 | 70.8 | 43.8 |
| F1 | 50.0 | 70.8 | 50.0 | 66.7 | 59.4 |
| UD-VLA | 58.3 | 62.5 | 54.1 | 75.0 | 62.5 |
| **LaRA-VLA** | **95.8** | 62.5 | 25.0 | **91.7** | **68.8** |

LaRA-VLA average 68.8% (best)。Put Spoon 95.8% 碾压，Put Eggplant 91.7%。Stack Block 25% 较低 — 可能是 dual-arm 复杂任务的 limitation。

### Real-World (Figure 5)

四个 long-horizon 任务，对比 ACT (https://tonyzhaozh.github.io/aloha/) 和 GR00T N1.5 (https://arxiv.org/abs/2503.14734)。LaRA-VLA 在所有任务上领先。

Subtask breakdown (Table 6) 显示 "Find Block & Place It" 任务 subtask 强耦合时 LaRA-VLA 优势最明显 — latent reasoning 在 cross-subtask coherence 上特别有效。

### Ablation (Table 4)

| Text-CoT | Latent Text-CoT | Latent Vis-CoT | SR (%) |
|----------|-----------------|----------------|--------|
| × | × | × | 55.21 |
| √ | × | × | 58.33 |
| × | √ | × | 64.58 |
| × | √ | √ | 68.75 |

**关键发现**：
- Explicit text CoT 只 +3.12%
- Latent text CoT +9.37% (vs baseline)
- 加 latent visual CoT 再 +4.17%

这说明 **latent representation 是核心创新，不是 text tokens 本身**。CoT 的 benefit 主要来自 structured intermediate computation，无论用 English 还是 latent 都能拿到，但 latent 更 compact 更 aligned with continuous action space。

### Efficiency (Figure 7)

LaRA-VLA: 135ms/rollout
Explicit CoT methods: ~1350ms/rollout

**90% latency reduction**。135ms ≈ 7.4Hz，够 real-time robotic control 用。Explicit CoT 的 1Hz 在 real robot 上基本不可用。

### Latent Collapse (Figure 6)

潜在风险：latent representations 退化成 homogeneous / uninformative 状态。

Figure 6 显示 latent tokens 形成 well-separated, semantically coherent clusters，不同 reasoning components 有 functional specialization，language instruction tokens 占据 distinct subspace — 没有 collapse 迹象。

为什么没 collapse？四个 anchor 同时作用：
- Visual prediction supervision (anchor 到 perceptual space)
- Action grounding (anchor 到 action space)
- EMA encoder (stable target)
- 限制 1 token per step (用 expressiveness 换 stability)

参考 SIM-CoT (https://arxiv.org/abs/2509.20317) 对 implicit reasoning scaling instability 的分析。

---

## 6. 用人话讲 Intuition

### 6.1 为什么 latent > explicit?

CoT 通过两个 pathway 起作用：
1. **Computational pathway**: 多步 forward computation = implicit multi-step reasoning
2. **Structural pathway**: structured intermediate representations (subtask, spatial, motion)

Text tokens 同时提供两个 pathway 但有 cost: discrete + verbose。

Latent 也提供两个 pathway：
- Computational: 多个 latent steps (即使 1 token per step，但 latent 是 high-dimensional，比如 2048-dim)
- Structural: 通过 curriculum 从 explicit CoT 蒸馏来的 structure 隐式 encode 在 latent 里

Latent 优势：
- High-dimensional continuous space → expressiveness 远超 discrete tokens
- 直接和 continuous perception/action 对齐 → 无 mismatch
- 无需 autoregressive generation → 低 latency

### 6.2 为什么 inverse dynamics 而不是 forward dynamics?

**Forward dynamics**: $(\mathbf{v}_t, \mathbf{a}_t) \to \mathbf{v}_{t+1}$ — 给 action 预测 next state
**Inverse dynamics**: $(\mathbf{v}_t, \mathbf{v}_{t+1}) \to \mathbf{a}_t$ — 给两个 state 推断 action

LaRA-VLA 选 inverse 因为：
1. Action prediction 从 visual prediction 派生，两个 prediction tasks 共享 representation
2. Visual latents 自然 encode "导致 state transition 的 action 信息"
3. 避免 forward dynamics 的 compounding error（预测 $\mathbf{v}_{t+1}$ 本身就难）

这个 idea 来自 World Models (https://worldmodels.github.io/) 和 Dreamer (https://arxiv.org/abs/1912.01603) 的 inverse dynamics 设计。

### 6.3 为什么三阶段而不是两阶段?

两阶段 (explicit → latent) 的问题：
- Stage I 到 Stage II 是 hard transition，容易 instability
- 没有专门 adapt 到 action generation

三阶段优势：
- Stage I → II: smooth curriculum，latent 逐步接管
- Stage II → III: action generation 从 AR tokens 换 flow matching，分别 optimize
- Stage III 可以单独 tune action expert，不影响 reasoning latent

类似 progressive distillation 在 diffusion model 加速中的应用 (https://arxiv.org/abs/2202.00512)。

### 6.4 EMA encoder 防 collapse 的本质

考虑 $\mathcal{L}_{\text{vis}} = \|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_1$：

如果 $\mathbf{z}_{t+1}$ 用 online encoder 产生，online encoder 和 predictor 可以一起 collapse — encoder 输出常数，predictor 输出同一常数，loss=0 但什么都没学到。

EMA encoder 的 target 是 slow-moving（$\tau_v \approx 0.99$），online encoder 无法快速 "追上" target 的变化，被迫学习有意义的 representation。

这是 self-supervised learning 的经典 trick (BYOL https://arxiv.org/abs/2006.07733, MoCo https://arxiv.org/abs/1911.05722)，LaRA-VLA 巧妙借用到 visual latent prediction 上。

---

## 7. Limitations

Paper 自己承认两个：

1. **Latent collapse risk**: 虽然实验中没观察到，但随着 latent token 数量增加风险上升。当前限制 1 token per step → expressiveness 受限。

2. **Training efficiency**: Curriculum strategy 导致 CoT-related tokens 数量随 training 增加，Stage I 和 Stage II 中间阶段开销大。

我自己再加几个观察：

3. **Single token capacity**: 1 token × 2048-dim 够不够 encode "find red block, plan grasp, plan trajectory"？Multi-token latents 需要解决 collapse — open question。

4. **Generalization to new tasks**: Latent reasoning 比 explicit CoT 更容易 overfit training tasks 吗？Latent representation 的 composability 如何？

5. **Interpretability trade-off**: 牺牲了 explicit CoT 的 interpretability。能否 decode latent 回 text 用于 debugging？Fast-ThinkAct (https://arxiv.org/abs/2601.09708) 探索了 verbalizable latent planning。

6. **Ultra-long horizon**: LIBERO Long 是 10 个 task 的 long-horizon，几十个 primitive actions。对于 100+ steps 的 ultra-long horizon，latent reasoning 能否 maintain coherence？可能需要 memory mechanisms (episodic + semantic)。

---

## 8. 和其他方法的对比直觉

### vs ECoT (https://embodied-chain-of-thought.github.io/)
ECoT 用 explicit text CoT，包括 subtask decomposition + bounding boxes + motion reasoning。问题：所有 scene objects 都标 bbox 太冗余，inference 慢。LaRA-VLA: latent + 自动 pipeline + 只标 target object。

### vs ThinkAct (https://arxiv.org/abs/2505.02686)
ThinkAct 用 reinforced visual latent planning。也用 latent 但 design 不同，主要在 visual latent。LaRA-VLA: text + visual latents 都用，multi-modal alignment 更完整。

### vs Fast-ThinkAct (https://arxiv.org/abs/2601.09708)
同属 latent CoT 但 LaRA-VLA 高 8.2 个点。可能原因：
- LaRA-VLA 的 multi-modal alignment (text + visual latents)
- Three-stage curriculum vs 两阶段
- EMA encoder stabilization
- Inverse dynamics supervision

Fast-ThinkAct 强调 verbalizable latent — interpretability 好，但可能 expressiveness 受限。

### vs Coconut (https://arxiv.org/abs/2412.06769)
Coconut 是 LLM-only 的 latent CoT。LaRA-VLA 把 idea 扩展到 VLA + multi-modal setting，加上 visual prediction 和 action grounding 的 supervision。

### vs π0 (https://arxiv.org/abs/2410.24164)
π0 用 flow matching 出 action 但没有 reasoning。LaRA-VLA Stage III 直接借鉴 π0 的 flow matching design，但加了 latent reasoning conditioning。

---

## 9. 个人思考: 未来方向

1. **Adaptive reasoning length**: 类似 Ocean-VLA (https://arxiv.org/abs/2505.11917) 的 adaptive reasoning — 根据任务难度调整 latent steps。简单任务 1 step，难任务多 step。

2. **Hierarchical latent reasoning**: high-level planning latents (subtask level) + low-level execution latents (motion level)。

3. **Multi-step visual prediction**: 不只 next frame，预测 multi-step future — 更强 world modeling，类似 Dreamer (https://arxiv.org/abs/1912.01603)。

4. **Active perception**: latent 中包含 "where to look next" 的 attention guidance。

5. **Continual learning**: latent reasoning 的 incremental learning for new tasks，避免 catastrophic forgetting。

6. **Cross-embodiment transfer**: latent reasoning 在 one embodiment 上训练后能否 transfer？Action 不 transfer 但 reasoning latents 可能可以。

7. **Verbalizable latents**: 让 latent 可以 decode 回 text 用于 interpretability，类似 Fast-ThinkAct 思路。

8. **Reasoning distillation from larger models**: 用 GPT-4 level reasoning 蒸馏到 latent space。

---

## 10. 总结一句话

**LaRA-VLA = Curriculum-based latent reasoning + Multi-modal alignment (text + visual) + EMA stabilization + Flow matching action generation**

核心 insight: **CoT 的 benefit 来自 structured intermediate computation，不来自 English 表达**。把这个 structure 压到 continuous latent，既保留 reasoning benefit 又解决 latency 和 mismatch。

Empirical evidence 强: LIBERO 97.9% SOTA, 90% latency reduction, ablation 证明 latent > explicit (+9.37% vs +3.12%)。

这是朝着 efficient + general + real-time embodied intelligence 的重要一步，把 LLM 领域的 latent reasoning idea 成功迁移到 multi-modal VLA setting。

---

## Useful Links

### LaRA-VLA & 类似工作
- LaRA-VLA Project: https://latent-reasoning-vla.github.io/ (推测)
- Coconut: https://arxiv.org/abs/2412.06769
- SIM-CoT: https://arxiv.org/abs/2509.20317
- CoDi: https://arxiv.org/abs/2502.21074
- SoftCoT: https://arxiv.org/abs/2502.12134

### VLA 基础模型
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054

### VLA + CoT methods
- ECoT: https://embodied-chain-of-thought.github.io/
- ThinkAct: https://arxiv.org/abs/2505.02686
- Fast-ThinkAct: https://arxiv.org/abs/2601.09708
- DeepThinkVLA: https://arxiv.org/abs/2511.15669
- CoT-VLA: https://arxiv.org/abs/2412.06420
- DreamVLA: https://arxiv.org/abs/2412.01243
- UP-VLA: https://arxiv.org/abs/2501.05148
- F1: https://arxiv.org/abs/2509.06951
- UD-VLA: https://arxiv.org/abs/2511.01718
- OneTwoVLA: https://arxiv.org/abs/2505.11917

### Benchmarks
- LIBERO: https://libero-project.github.io/
- SimplerEnv: https://simpler-env.github.io/

### Perception & Grounding
- GroundingDINO: https://arxiv.org/abs/2303.05499
- SAM3: https://arxiv.org/abs/2511.16719

### Action generation
- Fast tokenizer: https://arxiv.org/abs/2501.09747
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT: https://arxiv.org/abs/2212.09748

### Robotics foundation models
- GR00T N1.5: https://arxiv.org/abs/2503.14734
- ACT/ALOHA: https://tonyzhaozh.github.io/aloha/
- CogACT: https://arxiv.org/abs/2411.19650
- Octo: https://octo-models.github.io/

### Representation learning
- BYOL: https://arxiv.org/abs/2006.07733
- MoCo: https://arxiv.org/abs/1911.05722
- VL-JEPA: https://arxiv.org/abs/2512.10942

### World models
- World Models: https://worldmodels.github.io/
- Dreamer: https://arxiv.org/abs/1912.01603

### 其他相关
- OpenHelix: https://arxiv.org/abs/2505.03912
- Long-VLA: https://arxiv.org/abs/2502.02111
- ActionSketcher: https://arxiv.org/abs/2601.01618
- GraspVLA: https://arxiv.org/abs/2503.13598
- Emma-x: https://arxiv.org/abs/2502.01243

希望这个 "人话版" 帮你 build intuition。如果想深入聊任何 specific design choice（curriculum schedule 细节、EMA decay rate 选择、flow matching vs diffusion、single-token latent capacity），随时说。

---

# LaRA-VLA: Latent Reasoning for Vision-Language-Action Models

非常exciting的一篇paper,核心idea是把chain-of-thought (CoT) reasoning从discrete text tokens **内化**到continuous latent representations中,既保留了reasoning的结构化好处,又避免了inference时大量token generation的开销。下面深入展开。

---

## 1. Motivation: 两个fundamental pain points

现有VLA + CoT方法存在两个核心问题:

### Problem A: Inference Latency Bottleneck
Text-based CoT在inference时需要autoregressively generate几百个text tokens,导致:
- **KV-cache膨胀**: sequence length暴涨
- **Memory consumption高**
- **Latency unacceptable**: control frequency低于5Hz甚至1Hz (Zawalski et al., 2025)

对于real-time robotic control来说,1Hz基本等于不可用。robot需要50Hz左右的高频control signal。

### Problem B: Representational Mismatch
现有CoT都是discrete的:
- Textual CoT → discrete language tokens
- Visual CoT → discrete VQ-VAE tokens

但robotics的perception和action都是 **continuous spaces**。把reasoning约束在discrete tokens中,引入了fundamental mismatch。

**Key insight**: CoT effective的原因是它 **exposes structured intermediate reasoning**, 而不是因为它用natural language表达。如果structured reasoning可以内化到continuous latent中,就能同时解决latency和mismatch两个问题。

---

## 2. Architecture Overview

### Backbone
- **VLM**: Qwen3-VL (Bai et al., 2025a) — https://arxiv.org/abs/2511.21631
- **Visual encoder**: 直接继承Qwen3-VL的image encoder,保证training全过程的visual representation consistency
- **Action expert**: 16-layer Diffusion Transformer (DiT),由self-attention和cross-attention交替组成,conditioning on learned latent representations生成continuous action trajectories

### Special Tokens
- `<img next>`: 表示predicted visual latents,在early-stage latent reasoning learning中提供explicit supervision和alignment
- Action tokens: Stage I/II用autoregressive design (Pertsch et al., 2025),Stage III切换到DiT flow matching

### Latent Components
1. **Textual CoT latents**: 取代discrete text CoT tokens
2. **Visual goal latents**: 对齐到perceptual features (由shared visual encoder产生)
3. **Multi-modal latent context** $\mathbf{h}_t$: 聚合current visual + instruction + text reasoning latent + predicted future visual latent

---

## 3. 三阶段Curriculum Training (核心方法)

整体设计哲学: **先explicit supervision建立structure, 再逐步internalize到latent, 最后couple到action generation**。

### Stage I: Explicit CoT Fine-Tuning

Goal: 让VLM适应embodied manipulation,建立structured reasoning + visual prediction + action grounding。

**Loss 1: CoT Supervision Loss**

$$\mathcal{L}_{\mathrm{cot}} = -\sum_{t=1}^{T_{\mathrm{CoT}}} \log p_\theta(c_t \mid c_{<t}, \mathbf{v}, \mathbf{x})$$

变量解释:
- $\mathcal{L}_{\mathrm{cot}}$: negative log-likelihood of ground-truth CoT sequence
- $T_{\mathrm{CoT}}$: CoT token序列总长度
- $t$: CoT token的index,从1到$T_{\mathrm{CoT}}$
- $c_t$: 第$t$个ground-truth CoT token
- $c_{<t}$: 前面$t-1$个CoT tokens (autoregressive context)
- $\mathbf{v}$: visual tokens序列 (来自image encoder)
- $\mathbf{x}$: instruction text tokens序列
- $p_\theta(\cdot)$: 由VLM参数$\theta$定义的条件token分布
- $\theta$: VLM的全部参数

这个loss就是标准的next-token prediction,用teacher forcing。

**Loss 2: Visual Latent Alignment Loss**

$$\mathcal{L}_{\mathrm{vis}} = \|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_1$$

变量解释:
- $\mathcal{L}_{\mathrm{vis}}$: L1 distance between predicted和target visual latents
- $\hat{\mathbf{z}}_{t+1}$: VLM从current context预测的next observation visual latent
- $\mathbf{z}_{t+1}$: 用EMA-averaged encoder对真实next observation编码得到的target latent
- $t$: current time step
- $\|\cdot\|_1$: L1 norm (对outliers更robust)

**Loss 3: EMA Encoder Update**

$$\bar{\theta}_v^t = \tau_v \bar{\theta}_v^{t-1} + (1-\tau_v)\theta_v^t$$

变量解释:
- $\bar{\theta}_v^t$: 第$t$次iteration的EMA-averaged visual encoder参数 (target network)
- $\bar{\theta}_v^{t-1}$: 上次iteration的EMA参数
- $\theta_v^t$: online visual encoder在第$t$次iteration的参数
- $\tau_v$: EMA decay rate (通常接近1,比如0.99)

这个设计直接借鉴自BYOL (Grill et al., 2020)和MoCo系列 — target network提供稳定的supervision signal,避免online encoder self-reinforcing导致的representation collapse。在LaRA-VLA中,防止predicted latent直接copy input的退化解。

**Inverse Dynamics Model for Action Supervision**

$$f(\mathbf{v}_t, \mathbf{v}_{t+1} \mid \mathbf{x}, c) = \mathbf{a}_t$$

变量解释:
- $f(\cdot)$: inverse dynamics function
- $\mathbf{v}_t$: current visual observation tokens
- $\mathbf{v}_{t+1}$: next visual observation tokens
- $\mathbf{x}$: language instruction
- $c$: intermediate reasoning step
- $\mathbf{a}_t$: action causing the state transition

Intuition: 不是直接预测action,而是从"两个consecutive states推断transition action"。这种formulation让visual prediction和action prediction耦合起来,visual latents自然encode了action-relevant信息。

采用Pertsch et al. (2025)的fast recursive generation框架: https://arxiv.org/abs/2501.09747,把coarse-grained action semantics propagate到所有latent representations。Action tokens用autoregressive objective训练,记为$\mathcal{L}_{\mathrm{act-dis}}$。

**Stage I Total Loss**:
$$\mathcal{L}_{\mathrm{Stage I}} = \mathcal{L}_{\mathrm{cot}} + 0.1 \mathcal{L}_{\mathrm{vis}} + \mathcal{L}_{\mathrm{act-dis}}$$

权重0.1表示visual alignment是auxiliary supervision。

---

### Stage II: Curriculum-based Replacement

Goal: 把explicit textual CoT **gradually** internalize到latent space,避免直接训练导致collapse。

核心策略: 
- 保持Stage I的losses
- 但gradually mask掉一部分discrete CoT tokens
- 用learnable latent representations替换masked tokens
- Mask比例按curriculum schedule增加,最终全部CoT被internalize

**为什么需要curriculum?**
- 直接训练latent reasoning没有supervision,容易collapse到trivial solution
- Curriculum让模型先学会explicit reasoning的structure
- 再逐步把这种structure"压缩"到latent
- Visual和action signals提供implicit supervision,确保latent不退化

Stage II后期: explicit CoT loss $\mathcal{L}_{\mathrm{cot}}$ anneal到0,最终只保留:
$$\mathcal{L}_{\mathrm{Stage II, final}} = 0.2 \mathcal{L}_{\mathrm{vis}} + \mathcal{L}_{\mathrm{act-dis}}$$

Visual loss权重从0.1提到0.2,因为visual prediction成为latent reasoning的主要implicit supervision。

---

### Stage III: Action Generation via Flow Matching

Goal: 把latent-conditioned VLM features接入action expert,实现efficient continuous action generation without explicit CoT。

**Flow Matching Formulation**

Linear interpolation between noise和action:
$$\mathbf{a}_\tau = (1-\tau)\epsilon + \tau\mathbf{a}_t$$

变量解释:
- $\mathbf{a}_\tau$: flow time $\tau$处的interpolated action
- $\tau$: flow time,从noise ($\tau=0$) 到action ($\tau=1$),$\tau \sim \mathcal{U}(0,1)$ uniform distribution
- $\epsilon$: standard Gaussian noise, $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\mathbf{a}_t$: ground-truth action at time step $t$
- $\mathbf{I}$: identity matrix (covariance of Gaussian noise)

**Flow Matching Loss**

$$\mathcal{L}_{\mathrm{act-con}} = \mathbb{E}_{\mathbf{a}_t, \epsilon, \tau}\left[\left\|v_{\theta_a}(\mathbf{a}_\tau, \tau \mid \mathbf{h}_t) - (\mathbf{a}_t - \epsilon)\right\|_2^2\right]$$

变量解释:
- $\mathcal{L}_{\mathrm{act-con}}$: continuous action flow matching loss (MSE)
- $\mathbb{E}_{\mathbf{a}_t, \epsilon, \tau}$: 对ground-truth action、Gaussian noise、flow time的期望
- $v_{\theta_a}(\cdot)$: action expert (DiT)预测的velocity field
- $\theta_a$: action expert的参数
- $\mathbf{a}_\tau$: interpolated action
- $\tau$: flow time
- $\mathbf{h}_t$: multi-modal latent context (从VLM获取)
- $(\mathbf{a}_t - \epsilon)$: target velocity (从noise到action的constant velocity)
- $\|\cdot\|_2^2$: squared L2 norm

**Intuition**: Flow matching学习一个vector field,把noise distribution"流"到action distribution。比起diffusion的reverse SDE,flow matching更简单稳定,training更高效。这个idea直接来自π0 (Black et al., 2024): https://arxiv.org/abs/2410.24164。

$\mathbf{h}_t$聚合了:
- Current visual observation latent
- Language instruction
- Text-based reasoning latent (from Stage II)
- Predicted future visual latent

由于Stage I/II的inverse dynamics supervision,这个context已经encode了coarse-grained action信息,所以**不需要额外的action latent**,直接从shared multi-modal latent生成action。

---

## 4. LaRA Attention Mechanism (关键设计)

这是这篇paper一个被低估但很重要的设计。Figure 3展示了四种token之间的attention pattern:

### Token Types
1. **Text tokens**: 
   - Stage I/II: language instruction + textual CoT
   - Stage II/III: text latents (replacing CoT)
2. **Current image tokens**: visual observation
3. **Future image tokens** (`<img next>`): predicted visual latents
4. **Action tokens**: autoregressively generated (Stage I/II)

### Attention Constraints

**Future image tokens**:
- Causally attend to text和current image tokens (不能看未来)
- Bidirectionally interact among themselves (互相coherence)

**Action tokens** (Stage I/II):
- 每个action token attend to所有preceding text, current image, future image tokens
- Plus previously generated action tokens (causal within action sequence)

**Stage III**:
- Action tokens从attention computation中移除
- 只保留text和vision tokens的reasoning
- Action由DiT单独生成,conditioned on latents

**为什么这样设计?**

这种约束让information flow更符合物理因果性:
- Future image prediction依赖current observation和reasoning
- Action依赖reasoning和predicted future state (考虑consequence)
- Bidirectional self-attention among future tokens确保prediction的spatial-temporal consistency

类比Coconut (Hao et al., 2024) https://arxiv.org/abs/2412.06769 中的latent thought design,但LaRA-VLA把它扩展到了multi-modal setting。

---

## 5. Data Collection Pipeline

非常elegant的设计: **anchor-first, generate-later paradigm**。

### 三大CoT components
1. **Subtask decomposition** (long-horizon reasoning)
2. **Spatial grounding** (target object localization)
3. **Motion reasoning** (directional execution)

### Anchors

**Semantic anchors**:
- 用Qwen3-VL从initial frame + task instruction识别manipulated object
- 提供textual reference给后续visual grounding

**Temporal anchors**:
- 通过gripper state changes分割trajectory为atomic stages
- Stages: pre-grasp, grasp, move, release
- Boundaries作为keyframes

### Generation (conditioned on anchors)

1. **Subtask description**: Qwen3-VL基于instruction + keyframes生成
2. **Target bounding boxes**: 
   - GroundingDINO (Liu et al., 2024) https://arxiv.org/abs/2303.05499
   - SAM3 (Carion et al., 2025) https://arxiv.org/abs/2511.16719
   - Multi-frame ensemble for Bridge dataset (5 uniformly sampled anchor frames)
   - Linear interpolation填补tracking discontinuities
3. **Motion reasoning**: 
   - 从end-effector trajectory计算global motion (toward segment goal)
   - Local instantaneous motion
   - Discretize为directional descriptors

### 为什么需要这种pipeline?

现有data collection的问题:
- ECoT (Zawalski et al., 2025) https://embodied-chain-of-thought.github.io/ : 对所有scene objects标注bounding boxes → highly redundant supervision
- Emma-x (Sun et al., 2025b) https://arxiv.org/abs/2502.01243 : 缺少target localization

LaRA-VLA的pipeline把三个components有机整合,提供compact + task-centric的CoT supervision。

### 构建的数据集
- **LIBERO-LaRA**: 基于LIBERO (Liu et al., 2023) https://libero-project.github.io/
- **Bridge-LaRA**: 基于SimplerEnv (Li et al., 2025) https://simpler-env.github.io/
- **Real-robot demonstrations**: 长horizon任务

---

## 6. Experiments

### 6.1 LIBERO Benchmark (Table 2)

LIBERO包含4个task suites,每个suite 10个single-arm tasks,50 rollouts/task。

| CoT Type | Method | Spatial | Goal | Object | Long | Avg |
|----------|--------|---------|------|--------|------|-----|
| No CoT | OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| No CoT | π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| No CoT | OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| Text CoT | ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| Text CoT | π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.8 |
| Text CoT | DeepThinkVLA | 99.0 | 96.6 | 96.4 | 96.2 | 97.0 |
| Visual CoT | DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| Visual CoT | F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| Visual CoT | UD-VLA | 94.1 | 95.7 | 91.2 | 89.6 | 92.7 |
| Latent CoT | Fast-ThinkAct | 92.0 | 97.2 | 90.2 | 79.4 | 89.7 |
| **Latent CoT** | **LaRA-VLA** | **96.4** | **98.6** | **99.8** | **96.6** | **97.9** |

**Observations**:
- LaRA-VLA在Object suite上达到99.8% (几乎完美)
- Long suite上96.6% (long-horizon manipulation最难)
- 超过所有Textual和Visual CoT baselines
- 超过同类的Fast-ThinkAct (89.7%)

### 6.2 SimplerEnv-WidowX (Table 3)

Real-to-sim generalization评估,24 rollouts/task。

| Method | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Avg |
|--------|-----------|------------|-------------|--------------|-----|
| OpenVLA | 0.0 | 0.0 | 0.0 | 4.1 | 1.0 |
| Octo | 47.2 | 9.7 | 4.2 | 56.9 | 29.5 |
| OpenVLA-OFT | 12.5 | 4.2 | 8.3 | 37.5 | 39.6 |
| π0 | 29.1 | 0.0 | 16.7 | 62.5 | 40.1 |
| CogACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 |
| ThinkAct | 58.3 | 37.5 | 8.7 | 70.8 | 43.8 |
| F1 | 50.0 | 70.8 | 50.0 | 66.7 | 59.4 |
| UD-VLA | 58.3 | 62.5 | 54.1 | 75.0 | 62.5 |
| **LaRA-VLA** | **95.8** | 62.5 | 25.0 | **91.7** | **68.8** |

**Observations**:
- Put Spoon上达到95.8% (碾压所有baseline)
- Put Eggplant 91.7% (visual generalization强)
- Stack Block相对较低 (25.0%) — 可能是dual-arm task的complexity

### 6.3 Real-world Experiments (Figure 5)

四个long-horizon manipulation tasks:
1. Put All Objects into the Basket
2. Sort All Fruits into the Basket  
3. Find the Block and Place It in the Basket
4. Stack Two Bowls

100 demonstrations/task at 30Hz, 12 rollouts/trial。
Baselines: ACT (Zhao et al., 2023) https://tonyzhaozh.github.io/aloha/ 和 GR00T N1.5 (Bjorck et al., 2025) https://arxiv.org/abs/2503.14734

LaRA-VLA在所有4个任务上超过baseline,特别是multi-stage reasoning和temporal coordination要求高的任务。

### 6.4 Subtask-level Breakdown (Table 6)

| Task | Subtask 1 | Subtask 2 | Overall |
|------|-----------|-----------|---------|
| Put All Objects | 50.0 | 41.7 | 41.6 |
| Sort All Fruits | 50.0 | 66.7 | 50.0 |
| Find Block & Place | 83.3 | 33.3 | 33.3 |
| Stack Two Bowls | 91.7 | 100.0 | 91.7 |

**Key insight**: Find Block & Place It任务中subtask耦合最强 (subtask 2依赖subtask 1的成功)。LaRA-VLA在这种tightly coupled任务上相对优势最明显,说明latent reasoning在维持cross-subtask coherence上特别有效。

---

## 7. Ablation Study (Table 4, SimplerEnv)

| Text-CoT | Latent Text-CoT | Latent Vis-CoT | SR (%) |
|----------|-----------------|----------------|--------|
| × | × | × | 55.21 |
| √ | × | × | 58.33 |
| × | √ | × | 64.58 |
| × | √ | √ | 68.75 |

**Critical findings**:
1. **Explicit text-CoT only marginal gain** (+3.12%): 直接generate text tokens作为reasoning,效果有限
2. **Latent text-CoT significant gain** (+9.37% over baseline): internalized reasoning远胜explicit reasoning
3. **Adding latent vis-CoT further +4.17%**: multi-modal latent alignment提供额外supervision

这说明: **latent representation是关键innovation,而非text tokens本身**。

---

## 8. Latent Collapse Analysis (Figure 6)

潜在风险: latent representations degenerate到homogeneous/uninformative状态。

**Findings**:
- Latent tokens形成well-separated, semantically coherent clusters
- 不同reasoning components有clear functional specialization
- Language instruction tokens (gray points)占据distinct subspace,与reasoning latents分离
- 没有collapse迹象

**为什么没有collapse?**
1. Visual prediction supervision (anchors latent到perceptual space)
2. Action grounding (anchors latent到action space)
3. EMA encoder稳定target representations
4. 当前限制single token per step — 这是个trade-off,限制expressiveness换取stability

参考SIM-CoT (Wei et al., 2025) https://arxiv.org/abs/2509.20317 识别的optimization instability问题。

---

## 9. Inference Efficiency (Figure 7)

| Method | Latency (ms/rollout) |
|--------|---------------------|
| Explicit CoT methods | ~1350ms |
| **LaRA-VLA** | **135ms** |

**90% reduction in inference time** vs explicit CoT-based方法。

这是Latent reasoning的核心selling point: 把verbose text generation换成了compact latent "thought steps"。

135ms意味着大约7.4Hz control frequency,远高于explicit CoT的1Hz,接近real-time robotic control的最低要求。

---

## 10. Implementation Details (Table 5)

### Hyperparameters

| | LIBERO | SimplerEnv | Real Robot |
|--|--------|------------|------------|
| Action Horizon | 8 | 16 | 25 |
| VLM LR | $1\times10^{-5}$ | $1\times10^{-5}$ | $1\times10^{-5}$ |
| DiT LR | $1\times10^{-4}$ | $1\times10^{-4}$ | $1\times10^{-4}$ |
| Optimizer | AdamW | AdamW | AdamW |
| LR Scheduler | Cosine | Cosine | Cosine |
| Warm-up Ratio | 0.1 | 0.1 | 0.1 |

### Training Steps (Stage I / Stage II / Stage III)
- LIBERO: 5k / 2k+2k+2k / 40k
- SimplerEnv: 10k / 5k+5k+10k / 60k
- Real Robot: 5k / 2k+2k+2k / 10k

Stage II分3个子stage,逐步增加CoT token的mask比例。

### Loss Weights (Stage I / Stage II / Stage III)

| Loss | Stage I | Stage II | Stage III |
|------|---------|----------|-----------|
| Action Token Loss | 1.0 | 1.0 | — |
| Image Next Loss | 0.1 | 0.2 | — |
| CoT Loss | 1.0 | 1.0 | — |
| DiT Loss | — | — | 1.0 |

Stage III只用DiT loss,因为latent reasoning已经internalized,只需要adapt到continuous action generation。

Hardware: 8 NVIDIA H100 GPUs。

---

## 11. Intuition Building (核心insight总结)

### 11.1 为什么latent reasoning比explicit CoT更有效?

CoT的作用机制 — 我个人的理解:

**Explicit CoT通过两个pathway发挥作用:**
1. **Computational pathway**: 多token生成 = 多次forward computation = implicit的multi-step reasoning
2. **Structural pathway**: structured intermediate representations (subtask decomposition, spatial grounding, motion reasoning)

Text tokens恰好同时提供了这两个pathway,但它们的cost是discrete + verbose。

**Latent CoT的两个pathway:**
1. **Computational pathway**: 多个latent steps (即使在curriculum后期只剩single token per step,但latent representation本身是high-dimensional)
2. **Structural pathway**: 通过curriculum从explicit CoT蒸馏来的structure,隐式encode在latent中

Latent的优势:
- High-dimensional continuous space → expressiveness远超discrete tokens
- 直接和continuous perception/action对齐 → 无mismatch
- 无需autoregressive token generation → 低latency

### 11.2 为什么EMA encoder至关重要?

考虑visual prediction loss $\|\hat{\mathbf{z}}_{t+1} - \mathbf{z}_{t+1}\|_1$:
- 如果$\mathbf{z}_{t+1}$用online encoder产生,online encoder可以和predictor共同collapse
- 比如encoder输出常数,predictor也输出同一常数,loss=0但无信息

EMA encoder的target是slow-moving,提供stable supervision signal,阻止这种self-reinforcing collapse。

这是BYOL (https://arxiv.org/abs/2006.07733) 的核心insight,在LaRA-VLA中巧妙应用。

### 11.3 为什么curriculum training?

直接训练latent reasoning的failure modes:
1. **Trivial latent**: latent tokens退化为noise或constant
2. **Incoherent latent**: 不同step的latent之间没有temporal coherence
3. **Disconnected from action**: latent reasoning和action generation脱节

Curriculum解决方法:
- Stage I: explicit CoT作为"teacher signal",建立structured reasoning
- Stage II: 渐进替换 — 让latent学习mimic explicit CoT的structure,但用更compact representation
- Stage III: latent直接couple到action — 强制latent representation是action-relevant的

类似CoDi (Shen et al., 2025) https://arxiv.org/abs/2502.21074 的self-distillation思路,但LaRA-VLA的distillation target是multi-modal latent,不是single answer token。

### 11.4 Inverse dynamics model的作用

考虑两种action supervision方式:
- **Forward dynamics**: $(\mathbf{v}_t, \mathbf{a}_t) \to \mathbf{v}_{t+1}$ — 给action,预测next state
- **Inverse dynamics**: $(\mathbf{v}_t, \mathbf{v}_{t+1}) \to \mathbf{a}_t$ — 给两个states,推断action

LaRA-VLA选择inverse dynamics因为:
1. Action prediction从visual prediction派生 — 两个prediction tasks共享representation
2. Visual latents自然encode了"导致state transition的action信息"
3. 避免forward dynamics的compounding error (预测$\mathbf{v}_{t+1}$本身很难)

这个idea来自World Models (Ha & Schmidhuber, 2018) https://worldmodels.github.io/ 和Dreamer (Hafner et al., 2019) https://arxiv.org/abs/1912.01603 的inverse dynamics设计。

### 11.5 Multi-modal latent context $\mathbf{h}_t$的composition

$$\mathbf{h}_t = \text{aggregate}(\text{current visual}, \text{instruction}, \text{text reasoning latent}, \text{predicted future visual})$$

四个components对应robot control的四个cognitive functions:
1. **Current visual**: "Where am I?"
2. **Instruction**: "What should I do?"
3. **Text reasoning latent**: "How to decompose the task?"
4. **Future visual**: "What will the next state look like?"

这种composition让action expert从reasoning的"output"开始工作,而不需要重新parsing sensory input。

### 11.6 为什么三阶段而不是两阶段?

两阶段方案 (e.g., explicit → latent):
- Stage I到Stage II (latent): hard transition,容易training instability
- 没有专门adapt到action generation

三阶段的优势:
- Stage I → Stage II: smooth curriculum,latent逐步接管
- Stage II → Stage III: action generation方式从AR tokens换成flow matching,分别optimization
- Stage III可以单独tune action expert,不影响reasoning latent

类似progressive distillation在diffusion model加速中的应用 — https://arxiv.org/abs/2202.00512

---

## 12. Related Work & Connections

### 12.1 Latent Reasoning in LLMs

**Coconut** (Hao et al., 2024) https://arxiv.org/abs/2412.06769
- 最早的latent CoT paper
- 在LLM hidden states中reasoning
- 支持richer internal search,better accuracy-efficiency trade-off
- LaRA-VLA把这个idea扩展到VLA + multi-modal

**SIM-CoT** (Wei et al., 2025) https://arxiv.org/abs/2509.20317
- 识别implicit reasoning scaling时的optimization instability
- 引入supervised stabilization
- LaRA-VLA的curriculum + EMA是类似的stabilization思路

**CoDi** (Shen et al., 2025) https://arxiv.org/abs/2502.21074
- Self-distillation把explicit CoT压缩到continuous latent
- LaRA-VLA用了类似的"先explicit再implicit"思路

**SoftCoT** (Xu et al., 2025) https://arxiv.org/abs/2502.12134
- Soft chain-of-thought for efficient LLM reasoning
- 用continuous tokens替代discrete CoT

### 12.2 VLA Models

**RT-2** (Zitkovich et al., 2023) https://robotics-transformer2.github.io/
- 第一个VLA,把VLM直接扩展到robot control
- Discrete action tokens

**OpenVLA** (Kim et al., 2025b) https://openvla.github.io/
- Open-source VLA
- 7B参数,基于Prismatic VLM

**π0** (Black et al., 2024) https://arxiv.org/abs/2410.24164
- Flow matching for action generation
- 3B参数,general robot control
- LaRA-VLA的Stage III直接借鉴π0的flow matching design

**π0.5** (Intelligence et al., 2025) https://arxiv.org/abs/2504.16054
- π0的升级版,open-world generalization
- 包含text CoT
- LaRA-VLA在Table 2上和π0.5 (96.8%)对比,达到97.9%

### 12.3 VLA + CoT Methods

**ECoT** (Zawalski et al., 2025) https://embodied-chain-of-thought.github.io/
- Embodied chain-of-thought
- Discrete text tokens,subtask decomposition + bounding boxes + motion reasoning
- 主要对比对象之一

**ThinkAct** (Huang et al., 2025) https://arxiv.org/abs/2505.02686
- Vision-language-action reasoning via reinforced visual latent planning
- 也用latent planning但design不同

**Fast-ThinkAct** (Huang et al., 2026) https://arxiv.org/abs/2601.09708
- ThinkAct的efficient版本
- Verbalizable latent planning
- 在Table 2中LaRA-VLA显著超过Fast-ThinkAct (97.9 vs 89.7)

**DeepThinkVLA** (Yin et al., 2025) https://arxiv.org/abs/2511.15669
- 增强VLA的reasoning capability
- 在LIBERO上达到97.0%,接近LaRA-VLA

**CoT-VLA** (Zhao et al., 2025a) https://arxiv.org/abs/2412.06420
- Visual chain-of-thought for VLA
- Discrete visual tokens via VQ
- LaRA-VLA用continuous latents替代discrete visual tokens

**DreamVLA** (Zhang et al., 2025b) https://arxiv.org/abs/2412.01243
- "Dreamed" with comprehensive world knowledge
- Visual CoT,discrete visual tokens

**UP-VLA** (Zhang et al., 2025a) https://arxiv.org/abs/2501.05148
- Unified understanding and prediction
- 同时用text和visual CoT
- 但两者都是discrete tokens
- LaRA-VLA的direct extension: 两者的continuous版本

**F1** (Lv et al., 2025) https://arxiv.org/abs/2509.06951
- Bridging understanding and generation to actions
- Visual CoT方法

**UD-VLA** (Chen et al., 2025b) https://arxiv.org/abs/2511.01718
- Unified diffusion VLA via joint discrete denoising
- Visual CoT

### 12.4 Robotics Foundation Models

**GR00T N1.5** (Bjorck et al., 2025) https://arxiv.org/abs/2503.14734
- NVIDIA的humanoid robot foundation model
- Real-world experiment的baseline之一

**ACT** (Zhao et al., 2023) https://tonyzhaozh.github.io/aloha/
- Action Chunking Transformer
- Bimanual manipulation
- Real-world experiment的baseline之一

**CogACT** (Li et al., 2024) https://arxiv.org/abs/2411.19650
- Foundational VLA model synergizing cognition and action

**Octo** (Ghosh et al., 2024) https://octo-models.github.io/
- Open-source generalist robot policy

### 12.5 Perception & Grounding

**Qwen3-VL** (Bai et al., 2025a) https://arxiv.org/abs/2511.21631
- LaRA-VLA的VLM backbone
- 提供strong built-in reasoning capability
- Visual encoder被直接继承

**GroundingDINO** (Liu et al., 2024) https://arxiv.org/abs/2303.05499
- Open-vocabulary object detection
- Marrying DINO with grounded pre-training
- Data pipeline中用于target object grounding

**SAM3** (Carion et al., 2025) https://arxiv.org/abs/2511.16719
- Segment Anything with concepts
- Data pipeline中用于precise bounding box

**VL-JEPA** (Chen et al., 2025a) https://arxiv.org/abs/2512.10942
- Joint Embedding Predictive Architecture for vision-language
- 类似EMA encoder的philosophy

### 12.6 World Models & Latent Prediction

**World Models** (Ha & Schmidhuber, 2018) https://worldmodels.github.io/
- 在latent space中predict未来
- LaRA-VLA的visual latent prediction延续这个思路

**Dreamer** (Hafner et al., 2019) https://arxiv.org/abs/1912.01603
- Latent imagination + actor-critic
- Inverse dynamics model的inspiration

**PlaNet** (Hafner et al., 2019) https://planet.deepmind.com/
- Recurrent state space model
- Latent dynamics learning

---

## 13. Limitations & Future Directions

### Limitation 1: Latent Collapse Risk
- 虽然实验中没有观察到collapse,但随着latent token数量增加,collapse风险上升
- 当前实现限制single token per step → expressiveness受限
- 未来方向: multi-token latent reasoning with better stabilization

### Limitation 2: Training Efficiency
- Curriculum strategy导致CoT-related tokens数量随training增加
- Stage I开销大,Stage II中间开销最大
- 未来方向: more efficient training strategies

### My additional thoughts (intuition building):

**Potential Future Directions:**

1. **Adaptive Reasoning Length**: 类似Ocean-VLA (https://arxiv.org/abs/2505.11917) 的adaptive reasoning — 根据任务难度调整latent steps
2. **Hierarchical Latent Reasoning**: 引入hierarchical latent structure — high-level planning latents + low-level execution latents
3. **Multi-step Visual Prediction**: 不只预测next frame,而是预测multi-step future — 更强world modeling
4. **Active Perception**: latent中包含"where to look next"的attention guidance
5. **Continual Learning**: latent reasoning的incremental learning for new tasks
6. **Cross-embodiment Generalization**: latent reasoning能否transfer到不同robot morphology
7. **Verbalizable Latents**: 类似Fast-ThinkAct的verbalizable latents — 让latent可以decode回text for interpretability
8. **Reasoning Distillation from Larger Models**: 用GPT-4 level reasoning蒸馏到latent space

---

## 14. Critical Analysis & Open Questions

### 14.1 Latent vs Explicit: When does which win?

Ablation显示latent text-CoT (+9.37%) 远超 explicit text-CoT (+3.12%)。但这是在 **embodied** setting下。在general reasoning tasks (math, code)中,explicit CoT可能更优,因为:
- Symbolic operations (arithmetic, logic)在discrete space更natural
- Compositionality through language structure

Latent reasoning在embodied setting的优势:
- Perception/action的continuous nature
- Implicit reasoning about spatial + temporal structure
- Efficiency要求高 (real-time control)

### 14.2 Single Token Latent — Sufficient?

当前设计1 token per step。如果reasoning step本身complex (e.g., "find red block among cluttered scene, plan trajectory avoiding obstacles"),单token的capacity是否够?

可能的mitigation:
- High-dimensional latent (e.g., 2048-dim Qwen3-VL hidden size)
- Cross-attention聚合multi-modal信息
- Flow matching在action space的expressiveness

但expressiveness上限存在。Multi-token latents需要解决collapse问题 — 这是open question。

### 14.3 Long-horizon vs Short-horizon Tasks

LaRA-VLA在Long suite (96.6%)上优势明显。但什么是"long"? LIBERO Long是10个tasks的long-horizon manipulation,大约几十个primitive actions。对于100+ steps的ultra-long horizon,latent reasoning是否能maintain coherence?

可能需要:
- Memory mechanisms (episodic + semantic)
- Hierarchical planning latents
- Temporal abstraction in latent space

### 14.4 Comparison with Fast-ThinkAct

Fast-ThinkAct (89.7%) vs LaRA-VLA (97.9%) — 同属latent CoT但差8.2%。可能原因:
- LaRA-VLA的multi-modal alignment (text + visual latents)
- Three-stage curriculum vs 两阶段
- EMA encoder stabilization
- Inverse dynamics supervision
- Attention mechanism design

Fast-ThinkAct是verbalizable latent planning — 强调interpretability,但可能expressiveness受限。

---

## 15. Comparison Table (Full Picture)

| Method | CoT Type | Text Form | Visual Form | Action Form | LIBERO | SimplerEnv |
|--------|----------|-----------|--------------|--------------|--------|------------|
| OpenVLA | No CoT | — | — | Discrete | 76.5 | 1.0 |
| π0 | No CoT | — | — | Continuous | 94.2 | 40.1 |
| OpenVLA-OFT | No CoT | — | — | Continuous | 97.1 | 39.6 |
| ThinkAct | Text CoT | Discrete | — | Continuous | 84.4 | 43.8 |
| π0.5 | Text CoT | Discrete | — | Continuous | 96.8 | — |
| DeepThinkVLA | Text CoT | Discrete | — | Continuous | 97.0 | — |
| CoT-VLA | Vis CoT | — | Discrete Vis | Discrete | 81.1 | — |
| DreamVLA | Vis CoT | — | Discrete Vis | Continuous | 92.6 | — |
| F1 | Vis CoT | — | Discrete Vis | Continuous | 95.7 | 59.4 |
| UD-VLA | Vis CoT | — | Discrete Vis | Discrete | 92.7 | 62.5 |
| UP-VLA | Both | Discrete | Discrete Vis | Discrete | — | — |
| Fast-ThinkAct | Latent | Latent | — | Continuous | 89.7 | — |
| **LaRA-VLA** | **Latent** | **Cont. Latent** | **Cont. Latent** | **Continuous** | **97.9** | **68.8** |

LaRA-VLA是唯一一个all-continuous的方法 (text latent + visual latent + continuous action),也是SOTA。

---

## 16. Final Intuition: What makes LaRA-VLA work?

综合所有分析,LaRA-VLA成功的key factors:

1. **Curriculum bridges explicit和latent**: 不直接训练latent (会collapse),不直接用explicit (会verbose)。Curriculum是两者之间的smooth transition。

2. **Multi-modal alignment**: Text latents和visual latents互相regularize — 避免任一modality degenerate。类似CLIP的multi-modal contrastive learning思想。

3. **EMA stabilization**: 类似BYOL/MoCo的target network设计,提供stable supervision。

4. **Inverse dynamics coupling**: Visual prediction和action prediction通过inverse dynamics耦合,latent同时服务两个prediction tasks。

5. **Three-stage decoupling**: Reasoning learning (Stage I/II)和action generation (Stage III)分别优化,避免joint training的instability。

6. **Attention design**: Causal + bidirectional混合pattern符合物理因果性和spatial coherence需求。

7. **Flow matching for actions**: 比起AR token generation,flow matching在continuous action space更高效更stable。

8. **Fast recursive generation**: 让action information在latent steps间高效propagate,避免每个step独立预测的redundancy。

整个framework的设计哲学: **保留CoT的structural inductive bias,抛弃discrete token的computational overhead和representational mismatch**。

---

## 17. Open Questions for Discussion

1. **Single token latent的capacity**: 1 token × ~2048-dim是否足够encode "find red block, plan grasp, plan trajectory"? 能否quantify expressiveness?

2. **Curriculum schedule sensitivity**: Stage II的mask比例如何schedule? 线性,exponential,还是adaptive? Paper中提到"predefined curriculum schedule"但没详述。

3. **Generalization to new tasks**: Latent reasoning是否比explicit CoT更容易overfit training tasks? Latent representation的composability如何?

4. **Cross-embodiment transfer**: Latent reasoning在one embodiment上训练后,能否transfer到another embodiment? Action generation可能不transfer,但reasoning latents或许可以?

5. **Real-time adaptation**: 能否在inference时根据observation error动态调整latent reasoning steps? 类似adaptive computation time。

6. **Interpretability trade-off**: Latent reasoning牺牲了explicit CoT的interpretability。能否decode latent back to text/visual for debugging?

7. **Multi-agent scenarios**: 多robot协作时,latent reasoning能否shared或communicate? Latent space的"language"是什么?

8. **Safety verification**: Latent reasoning的"decisions"如何audit和verify for safety-critical applications?

---

## 18. Useful Links

### LaRA-VLA
- Project Page: https://latent-reasoning-vla.github.io/ (推测,paper中提到但没给URL)

### Backbone Models
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://openvla.github.io/
- OpenVLA-OFT: https://arxiv.org/abs/2505.01600

### Latent Reasoning
- Coconut: https://arxiv.org/abs/2412.06769
- SIM-CoT: https://arxiv.org/abs/2509.20317
- CoDi: https://arxiv.org/abs/2502.21074
- SoftCoT: https://arxiv.org/abs/2502.12134
- Multimodal CoCoT: https://arxiv.org/abs/2508.12587
- Reasoning to Learn from Latent Thoughts: https://arxiv.org/abs/2503.18866

### VLA + CoT Methods
- ECoT: https://embodied-chain-of-thought.github.io/
- ThinkAct: https://arxiv.org/abs/2505.02686
- Fast-ThinkAct: https://arxiv.org/abs/2601.09708
- DeepThinkVLA: https://arxiv.org/abs/2511.15669
- CoT-VLA: https://arxiv.org/abs/2412.06420
- DreamVLA: https://arxiv.org/abs/2412.01243
- UP-VLA: https://arxiv.org/abs/2501.05148
- F1: https://arxiv.org/abs/2509.06951
- UD-VLA: https://arxiv.org/abs/2511.01718
- OneTwoVLA: https://arxiv.org/abs/2505.11917

### Robotics Benchmarks
- LIBERO: https://libero-project.github.io/
- SimplerEnv: https://simpler-env.github.io/

### Perception & Grounding
- GroundingDINO: https://arxiv.org/abs/2303.05499
- SAM3: https://arxiv.org/abs/2511.16719

### Action Generation
- Fast tokenizer: https://arxiv.org/abs/2501.09747
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT (Diffusion Transformer): https://arxiv.org/abs/2212.09748

### Robotics Foundation Models
- GR00T N1.5: https://arxiv.org/abs/2503.14734
- ACT / ALOHA: https://tonyzhaozh.github.io/aloha/
- CogACT: https://arxiv.org/abs/2411.19650
- Octo: https://octo-models.github.io/
- RT-2: https://robotics-transformer2.github.io/

### Representation Learning
- BYOL: https://arxiv.org/abs/2006.07733
- MoCo: https://arxiv.org/abs/1911.05722
- VL-JEPA: https://arxiv.org/abs/2512.10942

### World Models
- World Models: https://worldmodels.github.io/
- Dreamer: https://arxiv.org/abs/1912.01603

### Additional VLA papers
- OpenHelix: https://arxiv.org/abs/2505.03912
- Long-VLA: https://arxiv.org/abs/2502.02111
- ActionSketcher: https://arxiv.org/abs/2601.01618
- GraspVLA: https://arxiv.org/abs/2503.13598
- Emma-x: https://arxiv.org/abs/2502.01243

---

## 19. Final Summary

LaRA-VLA在embodied AI + reasoning的intersection上迈出了关键一步。核心贡献:

1. **Conceptual**: 把latent reasoning从LLM扩展到VLA + multi-modal setting
2. **Methodological**: 三阶段curriculum + EMA + inverse dynamics + flow matching的完整pipeline
3. **Empirical**: LIBERO 97.9%, SimplerEnv 68.8%, 90% inference latency reduction
4. **Data**: 两个structured CoT datasets (LIBERO-LaRA, Bridge-LaRA)
5. **Insight**: Latent CoT > Explicit CoT在embodied setting中 (+9.37% vs +3.12% in ablation)

整体哲学: **CoT的有效性来自structure,而非natural language表达**。当structure可以internalize到continuous latent时,既能保留reasoning的computational benefit,又能避免discrete token generation的overhead和representational mismatch。

这是朝着 **efficient + general + real-time embodied intelligence** 的重要一步,值得深入研究如何扩展到更complex scenarios (multi-robot, multi-modal sensing, hierarchical reasoning等)。

希望这个dive into对build your intuition有帮助!如果想深入讨论任何specific aspect (e.g., attention mechanism design choices, curriculum schedule, EMA vs alternatives, flow matching vs diffusion for actions), happy to继续展开。
