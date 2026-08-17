---
source_pdf: τ0-WM- A Unified Video-Action World Model for Robotic Manipulation.pdf
paper_sha256: e5abad225835b87d9a115a45c816f4ad7d9137d26ca8b3c9187bb581f88a18c8
processed_at: '2026-08-13T06:58:38-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，Andrej，我用大白话再给你捋一遍，重点放在 intuition 上。

---

## 这帮人到底想干嘛

Robot learning 有个老问题：你想让 robot 学会干活，就得给它看很多演示。但 robot 演示又贵又少，而人类日常活动的 video 多得是，可惜没有 action label。以前的做法要么只用 robot data（grounded 但 narrow），要么只用 video data（broad 但 ungrounded）。

$\tau_0$-WM 的核心想法很简单：**把这些乱七八糟的数据全塞进一个 model 里，让每种数据只教它该教的东西。** Robot data 教 action，video data 教 visual dynamics，failure data 教 "什么动作会搞砸"。用 mask 来控制每条数据贡献哪些 loss。

但光做到这一步还不够有意思。他们更进一步的想法是：**这个 model 不光要会输出 action，还要能在脑子里 "预演" 这个 action 执行后会发生什么。** 就像你伸手拿杯子之前，脑子里会先闪过杯子被拿起来的画面。如果你想象中发现这个动作会打翻杯子，你就会换一个动作。

这就是所谓的 World Model——model 不只是 reactive policy，它还能 imagine futures。

---

## Model 长什么样

 backbone 是 Wan2.2，一个 5B 参数的 video diffusion model，本来是做 text-to-video generation 的。他们在上面接了两个东西：

**VAM (Video Action Model)** — 这是 policy。它吃进当前画面、语言指令、robot state，同时吐出两样东西：一段 future video latent 和一段 action chunk。注意，action 和 video 是 joint denoising 的，它们在同一个 flow matching 过程里一起生成。Action branch 通过 cross-attention 去看 video branch 中间层的 features，这样 action 就能 aware 到 visual dynamics。

**ACVS (Action-Conditioned Video Simulator)** — 这是 evaluator。它跟 VAM 共享同一个 video backbone，但没有了 action branch。你给它一个 candidate action，它帮你 "rollout" 出这个 action 执行后的未来画面，还附带一个 dense reward trajectory（每一帧的 task progress 分数）。

关键 insight：**VAM 和 ACVS 是同一个 backbone 的两个 interface。** VAM 说 "应该做什么"，ACVS 说 "如果做这个会怎样"。它们共享同一套 learned visual dynamics representation。

---

## 训练 objective 的细节

两个 interface 都用 Flow Matching，loss 形式对称：

**VAM loss:**
$$\mathcal{L}_{\mathrm{VAM}} = \mathbb{E}\left[\lambda_z \| f_\theta^z(\tilde{\mathbf{z}}, u_z, \mathbf{c}_t, \mathbf{p}) - \mathbf{v}_{\mathbf{z}}\|_2^2 + \lambda_a \| f_\theta^a(\tilde{\mathbf{a}}, u_a, \mathbf{s}_t, \mathbf{h}) - \mathbf{v}_{\mathbf{a}}\|_2^2\right]$$

变量含义：
- $\tilde{\mathbf{z}}$: 加噪后的 future video latent（被 noise 扰动过的未来画面表示）
- $\tilde{\mathbf{a}}$: 加噪后的 action chunk
- $u_z, u_a$: noise level（flow timestep），控制加噪程度，从 0（纯噪声）到 1（clean data）
- $\mathbf{c}_t$: clean visual context，即当前 observation 经 VAE encode 后的 latent
- $\mathbf{p}$: language instruction 的 text embedding
- $\mathbf{s}_t$: robot proprioceptive state（关节角度等）
- $\mathbf{h}$: video branch 中间层输出的 features，action branch 通过 cross-attention 读取
- $f_\theta^z, f_\theta^a$: 分别是 video 和 action 的 vector field predictor（DiT 的输出 head）
- $\mathbf{v}_{\mathbf{z}}, \mathbf{v}_{\mathbf{a}}$: ground truth velocity，指向 clean data 方向的向量
- $\lambda_z, \lambda_a$: loss weights，论文里都设成 1

**ACVS loss:**
$$\mathcal{L}_{\mathrm{ACVS}} = \mathbb{E}\left[\lambda_z \| g_\phi^z(\tilde{\mathbf{z}}, u_z, \mathbf{c}_{t-M:t}, \mathbf{p}, \bar{\mathbf{a}}) - \mathbf{v}_{\mathbf{z}}\|_2^2 + \lambda_r \| g_\phi^r(\tilde{\mathbf{r}}, u_r, \mathbf{h}) - \mathbf{v}_{\mathbf{r}}\|_2^2\right]$$

区别在于：ACVS 把 candidate action $\bar{\mathbf{a}}$ 作为 condition 注入（通过 AdaLN 和 diffusion-time embedding），预测的是 reward trajectory $\mathbf{r}$ 而非 action。$\mathbf{c}_{t-M:t}$ 是 memory observations（过去 $M$ 帧的 clean latent），给 simulator 更多时序 context。

---

## 数据怎么混的

27,300 小时的数据，三种来源：

| 数据类型 | 时长 | 提供 action label? | 提供 video dynamics? | 提供 reward label? |
|---------|------|-------------------|---------------------|-------------------|
| Real-robot teleoperation | 17.8K hrs | ✅ 精确 | ✅ | 部分（含 failure trajectories） |
| UMI-style demonstrations | 6.5K hrs | ⚠️ weak signal | ✅ | ❌ |
| Egocentric human videos | 3.0K hrs | ❌ | ✅ | ❌ |

每条数据进 model 时带一个 supervision mask，告诉 model "这条数据你能学什么"。Egocentric video 的 action loss 直接被 mask 掉，model 只从中学 visual dynamics。这样 heterogeneous data 就能喂进同一个 training pipeline。

Ablation 结果很说明问题（Table I）：

| Pre-training Data | Zero-shot Avg Success | SFT Avg Success |
|-------------------|----------------------|-----------------|
| Robot only | 0.14 | 0.70 |
| Robot + UMI + Ego | 0.55 | 0.83 |

光用 robot data，zero-shot 基本废了（0.14）。加上 video data 后直接蹦到 0.55。这跟我们的直觉一致：video data 教会了 model "物体一般怎么动"、"手一般怎么抓"，这些 priors 迁移到 robot 任务上效果巨大。

---

## Test-Time Computation：最有意思的部分

这部分直接对标你一直在讲的 inference scaling。

问题背景：pre-trained on heterogeneous data 后，conditional action distribution 是 multi-modal 的。同一个场景下，model 可能采样出好几个不同的 action，有的精准有的拉胯。直接执行第一个 sample 出来的 action，运气好就成功，运气差就失败。

他们的解法是 coarse-to-fine 两阶段：

### Stage 1: Re-denoising Consistency Score (RCS)

从 VAM 采样 $N$ 个 candidate actions。对每个 candidate $\bar{\mathbf{a}}^{(i)}$，随机选 $K$ 个 flow timesteps，按 training 时的 flow matching process 重新加噪，然后用 model 的 action vector field 去 denoise，看 re-denoising error $\mathcal{E}_{\mathrm{RCS}}^{(i)}$ 有多大。

$$S_{\mathrm{RCS}}^{(i)} = -\mathcal{E}_{\mathrm{RCS}}^{(i)}$$

$$i^\star = \arg\max_i S_{\mathrm{RCS}}^{(i)}$$

Intuition：如果一个 action 位于 learned manifold 的高密度区（model 很有信心的区域），加噪后 model 应该能把它拉回来，error 小。如果是个 outlier，model 可能把它拉向别的 mode，error 大。这跟 diffusion model 里的 "noise-then-denoise" consistency check 是一个道理，极其 cheap，几乎不增加 latency。

### Stage 2: Low-quality Action Rectification (LAR)

如果最好的 candidate 的 RCS score 仍然低于 threshold $\gamma$：

$$S_{\mathrm{RCS}}^{(i^\star)} < \gamma$$

说明采样的 candidates 整体质量不行。这时调用 ACVS，对每个 candidate 都 rollout 一遍未来，预测 reward trajectory：

$$(\hat{\mathbf{z}}^{(i)}, \hat{\mathbf{r}}^{(i)}) = G_\phi(\mathbf{o}_{t-M:t}, \mathbf{p}, \bar{\mathbf{a}}^{(i)})$$

$$J^{(i)} = \max_{0 \leq q < H_a} \hat{r}_{t+q}^{(i)}$$

$J^{(i)}$ 是 imagined rollout 中达到的最大 task progress。选 $J$ 最大的那个 rollout $\hat{\mathbf{z}}^{(j^\star)}$，把它作为 future condition 重新喂给 VAM，让 VAM 在这个 "理想未来" 的引导下重新生成一个 refined action。

这步的逻辑很 elegant：**先想象哪个未来最好，再基于这个理想未来去指导 action 生成。** 有点像 latent space 里的 MPC——先 simulate，再 act。

Ablation 结果（Table II，single-attempt，不允许 retry）：

| Variant | Tissue→Box | Pen→Box | Avg |
|---------|-----------|---------|-----|
| w/o TTC | 0.55 | 0.30 | 0.43 |
| w. CFG | 0.25 | 0.15 | 0.20 |
| w. ACG | 0.40 | 0.35 | 0.38 |
| w. RCS | 0.65 | 0.35 | 0.50 |
| w. RCS + LAR | 0.70 | 0.50 | 0.60 |

CFG（Classifier-Free Guidance）反而降低了性能，这在 robotics 里不意外——CFG 在 image generation 里好用，但 continuous action space 里的 guidance 容易把 action 推到 OOD 区域。RCS + LAR 则稳定提升，说明 "evaluate before act" 的策略在 robotics 里是有效的。

---

## 部署速度

Single RTX 5090，end-to-end latency 约 220ms per query。Cache text representation 后降到 180ms。加上 KV cache（cross-attention 的 key/value 只算一次）、fused QKV projection、torch.compile，能压到 140ms。

Action chunk length 30，receding-horizon closed-loop execution。对于大部分 manipulation task 来说，140-220ms 的 latency 是可接受的范围。

---

## 我的几点 takeaways

1. **Video prediction 不只是 auxiliary objective。** 很多 VLA model 把 future prediction 当 auxiliary loss 来用，训完就扔。$\tau_0$-WM 把它保留到 inference time，作为 action evaluation 和 rectification 的工具。这跟 $\pi_{0.5}$ 和 Fast-WAM 的思路形成对比——Fast-WAM 甚至在 inference 时完全移除 future prediction 来降 latency。$\tau_0$-WM 的实验说明，在 difficult states 下，future prediction 的价值恰恰最大。

2. **Heterogeneous data 的正确用法是 mask-based supervision，不是统一格式。** 以前很多人想把 video data 和 robot data 统一成同一格式（比如用 VAE 把 video 也 encode 成 action-like representation）。$\tau_0$-WM 的做法更直接：让每种数据只贡献它有的 signal，用 mask 控制。简单但有效。

3. **Test-time computation 在 robotics 里的形态跟 LLM 不同。** LLM 的 test-time compute 是 chain-of-thought，生成更多 token。Robotics 的 test-time compute 是 sample 多个 actions，evaluate 它们的 futures，select 最好的。本质都是在 inference 时花更多 compute 来 improve output quality，但 mechanism 完全不同。

4. **Reward labeling 的做法值得注意。** 他们把 task 拆成 subtasks，用 Monte Carlo propagation 估计 frame-level reward，还故意加入了 failure data（failure segment 的 reward 为负）。这让 ACVS 学会区分 "看起来 plausible 但实际不 make progress" 的 action 和 "真正推进 task" 的 action。这比单纯用 terminal success signal 信息量大得多。

---

## 可能的联想与局限

- **Tactile modality 的缺失。** 论文自己在 conclusion 里提到了。Faucet 任务（接水管）对所有的 model 都很难，很可能就是因为视觉无法 capture 接触力反馈。单纯 visual world model 在 contact-rich task 上有天花板。
- **Reward head 的可靠性。** LAR 的效果完全依赖 ACVS 的 reward prediction 准不准。如果 ACVS 自己都判断错了，LAR 可能反而选到更差的 action。论文没有对 reward prediction accuracy 做详细 ablation，这块是个隐忧。
- **Latency vs. TTC 的 trade-off。** RCS 几乎不增加 latency，但 LAR 要调用 ACVS 做完整 rollout，latency 会显著增加。论文没有明确报告 LAR 触发时的总 latency。在实际 deployment 中，可能需要异步执行或 pipeline 化来 hide latency。
- **与 Dreamer 系列的关系。** Dreamer 也是在 latent space 里 rollout future，然后基于 imagined reward 做 planning。$\tau_0$-WM 的 ACVS + LAR 本质上是 Dreamer 的 diffusion-based 版本，但用了更 powerful 的 video generation backbone 和 learned reward head，而不是 RSSM + pixel reconstruction。
- **Multi-modal action distribution 的问题。** Flow matching 在 multi-modal distribution 上比 DDPM 好一些，但 sampling 时仍然可能 miss mode。RCS 在一定程度上缓解了这个问题，但更根本的解法可能是用 mixture-of-flow-experts 或 discrete action tokens。

References:
- Wan Video Generation Backbone: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- Flow Matching: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Dreamer (Latent World Model for Robotics): [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
- Visual Foresight (Early Action-Conditioned Prediction): [arXiv:1704.05543](https://arxiv.org/abs/1704.05543)
- Cosmos World Foundation Model: [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
- UMI (Universal Manipulation Interface): [arXiv:2402.10329](https://arxiv.org/abs/2402.10329)
- $\pi_{0.5}$ VLA Model: [OpenAI/Physical Intelligence, CoRL 2025](https://arxiv.org/abs/2604.14352)
- DiT (Diffusion Transformer): [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)

---

Andrej, 这篇关于 $\tau_0$-WM (tau-zero World Model) 的 paper 非常符合你对 Robot Foundation Model 和 Test-time Computation 的直觉。这项工作来自 Shanghai Innovation Institute 和 AGIBOT Finch 团队，核心在于将 Video Generation、Action Prediction 以及 Action Evaluation 统一到一个基于 Flow Matching 的 Video Diffusion Backbone 中，并利用 Test-time Computation 来实现 physical action 的 rectification。

下面我为你拆解其核心技术细节，希望能进一步 build up your intuition。

### 1. Core Intuition: Unified Predictive Representation

传统 Robotics 的 VLA (Vision-Language-Action) model 往往直接映射 observation 到 action，缺乏对未来状态的显式建模。$\tau_0$-WM 的核心哲学是：**A useful robot model must relate observations, actions, and future outcomes within a shared predictive framework.** 

该 framework 包含两个共享 backbone 的 interface：
*   **Video Action Model (VAM)**: Policy-facing interface。输入当前 observation、language instruction 和 robot state，联合预测 future visual latents 和 executable action chunks。
*   **Action-Conditioned Video Simulator (ACVS)**: Evaluation-facing interface。输入 candidate action chunks，预测 multi-view future rollouts 和 dense task-progress scores (rewards)。

这种 design 允许 model 在物理执行前，先在 latent space 中 "imagine" 动作的后果，并基于 imagination 的好坏来 rectify action。

### 2. Heterogeneous Data & Unified Supervision

$\tau_0$-WM 训练于 27,300 hours 的异构数据，这种异构性被视为 structured supervision source：
*   **Real-robot teleoperation (17.8K hours)**: 来自 AGIBOT-G01, ARX, Franka。提供 deployment-aligned continuous actions。
*   **UMI-style demonstrations (6.5K hours)**: Handheld gripper 设备收集，提供 broad manipulation behaviors，但 action signal weakly aligned with robot kinematics。
*   **Egocentric human videos (3.0K hours)**: 大规模人类日常操作视频，仅有 visual dynamics，无 robot-compatible actions。

为了在同一个 model 中训练这些数据，论文引入了 **modality-specific supervision masks**。对于 Egocentric videos，action loss 被 mask 掉，模型只学习 visual dynamics；对于 Real-robot data，video 和 action loss 同时激活。这种 mask 机制使得 model 可以从 vast video data 中学习 world priors，同时保持 robot action 的 grounding。

### 3. Architecture Deep Dive

整个 architecture 基于 Wan2.2-TI2V-5B (一个 Text-to-Video DiT model)。

**VAM Architecture (5.5B parameters)**:
*   **Video Branch (5B params)**: 采用 Wan VAE 将 multi-view observations encode 成 latent tensors。由于是 multi-view，view latents 在 spatial width 维度上 concat，形成 temporally aligned latent canvas。当前 observation 的 latent 保持 clean 作为 visual context，future latent slots 加噪并由 video DiT backbone denoise。
*   **Action Branch (0.5B params)**: 一个 DiT-style action decoder。Action tokens 先在 temporal horizon 内部 model dependencies，然后通过 **cross-attention** 汇聚 video branch 中间层输出的 features $\mathbf{h}$。Video features 包含了 clean visual context 和 language instruction 的 conditioning，从而为 action generation 提供 instruction-aware 和 dynamics-relevant 的 representation。

**ACVS Architecture**:
ACVS 移除了 Action DiT branch，完全复用 video generation backbone。Candidate action 的注入方式参考了 Cosmos，采用 action block injection：
$$ \mathbf{c}_{\ell}^a = \psi_D(\mathbf{b}_\ell), \quad \mathbf{m}_{\ell}^a = \psi_{6D}(\mathbf{b}_\ell) $$
其中 $\mathbf{b}_\ell$ 是对齐到 future latent slot $\ell$ 的 action block，$\psi_D$ 和 $\psi_{6D}$ 是 lightweight MLPs。$\mathbf{c}_{\ell}^a$ 注入到 diffusion-time embedding，$\mathbf{m}_{\ell}^a$ 注入到 AdaLN (Adaptive Layer Normalization) modulation embedding。这样，action signal 就 broadcast 到了 spatial tokens 和 camera views 上，conditioned 未来的 video generation。

### 4. Joint Flow-Matching Objective

$\tau_0$-WM 没有使用传统的 DDPM，而是采用了 Flow Matching (Lipman et al., ICLR 2023)。Flow Matching 通过匹配从噪声分布到数据分布的 vector field 来训练 generative model，在 continuous action space 和 video latent space 中表现更好。

VAM 的 loss function 如下：
$$ \mathcal{L}_{\mathrm{VAM}} = \mathbb{E} \Big [ \lambda_z || f_\theta^z (\tilde{\mathbf{z}}, u_z, \mathbf{c}_t, \mathbf{p}) - \mathbf{v}_{\mathbf{z}} ||_2^2 + \lambda_a || f_\theta^a (\tilde{\mathbf{a}}, u_a, \mathbf{s}_t, \mathbf{h}) - \mathbf{v}_{\mathbf{a}} ||_2^2 \Big ] $$

*   $\tilde{\mathbf{z}}, \tilde{\mathbf{a}}$: 加噪后的 future video latents 和 action chunks。
*   $u_z, u_a$: Noise levels (或 flow timesteps)。
*   $\mathbf{c}_t$: Clean encoded visual context。
*   $\mathbf{p}$: Language instruction。
*   $\mathbf{s}_t$: Robot state。
*   $\mathbf{h}$: Action branch cross-attend 的 intermediate video features。
*   $f_\theta^z, f_\theta^a$: Video 和 action 的 vector-field prediction heads。
*   $\mathbf{v}_{\mathbf{z}}, \mathbf{v}_{\mathbf{a}}$: Velocity targets (从 noise 指向 clean data 的 ground truth vector)。
*   $\lambda_z, \lambda_a$: Loss weights，实验中均设为 1。

**Intuition**: 这种 joint flow matching 让 video 和 action 在同一个 denoising trajectory 上协同演化。Action 的生成依赖于 video 的 dynamics ($\mathbf{h}$)，同时 action 的预测也反过来 regularize video branch 去关注 task-relevant 的 visual regions。

### 5. Test-Time Computation (TTC): The Coarse-to-Fine Strategy

这是本文最亮眼的部分，将 LLM 领域的 test-time compute scaling 引入到 robotics。由于 pre-trained on heterogeneous data，conditional action distribution 是 multi-modal 的。对于同一个 task，可能存在多个 feasible action sequences，但它们的 precision 和 robustness 不同。

**Stage 1: Re-denoising Consistency Score (RCS)**
VAM 先 sample $N$ 个 candidate action chunks $\{ \bar{\mathbf{a}}^{(i)} \}_{i=1}^N$。为了评估哪个 candidate 处于 conditional action manifold 的高密度区，论文提出了 RCS。
对于每个 candidate，随机采样 $K$ 个 flow timesteps，按照 flow-matching process 重新加噪，然后让 VAM 的 action vector field 去 denoise 它，计算 re-denoising error $\mathcal{E}_{\mathrm{RCS}}^{(i)}$。
$$ S_{\mathrm{RCS}}^{(i)} = -\mathcal{E}_{\mathrm{RCS}}^{(i)} $$
选取 $i^\star = \arg \max_i S_{\mathrm{RCS}}^{(i)}$。
**Intuition**: 如果一个 action 是模型非常有信心的（即位于 manifold 中心），那么给它加点噪声，模型应该能沿着原路把它拉回来。如果它是个 outlier，模型可能会把它拉向其他 mode，导致 re-denoising error 很大。这是一种极其轻量级的 distributional filter。

**Stage 2: Low-quality Action Rectification (LAR)**
如果 top candidate 的 RCS score 仍然低于 threshold $\gamma$ ($S_{\mathrm{RCS}}^{(i^\star)} < \gamma$)，说明 sampled candidates 整体质量不佳。此时触发 LAR，调用 ACVS。
ACVS 输入每个 candidate action，输出 imagined future rollout $\hat{\mathbf{z}}^{(i)}$ 和 dense reward trajectory $\hat{\mathbf{r}}^{(i)}$。
计算 rollout value:
$$ J^{(i)} = \max_{0 \le q < H_a} \hat{r}_{t+q}^{(i)} $$
选取使得 $J^{(i)}$ 最大的 candidate $j^\star$。注意，此时并不直接执行 $\bar{\mathbf{a}}^{(j^\star)}$，而是将 ACVS 生成的 best future rollout latent $\hat{\mathbf{z}}^{(j^\star)}$ 作为额外的 future condition，重新 query VAM，生成一个 refined action chunk $\tilde{\mathbf{a}}$。
**Intuition**: 这是一种 model predictive control (MPC) 的 latent space 版本。先想象哪个 action 能带来最好的未来，然后基于这个理想的未来去指导 action 生成。

### 6. Experimental Results & Ablations

论文在四个长视野、精细操作任务上进行了评估。实验结果表明：

*   **Heterogeneous Pre-training Matters**: Table I 显示，仅用 robot data 训练，zero-shot 成功率很低 (Avg. 0.14)。加入 UMI 和 Egocentric data 后，zero-shot 成功率飙升到 0.55。这说明 vast video data 极大增强了 model 的 visual understanding 和 general manipulation priors。
*   **TTC Boosts Performance**: Table II 对比了不同 test-time computation variants。在 single-attempt 严格测试下，baseline 只有 0.43 的成功率。CFG (Classifier-Free Guidance) 甚至降低了性能 (0.20)，因为 CFG 更适合 image generation，在 continuous action space 中可能引导到 OOD (out-of-distribution) 区域。RCS 提升到 0.50，而 RCS + LAR 达到了 0.60。这证明了 imagination-based rectification 的有效性。

### 7. Broader Connections & Intuition Building

*   **Visual Foresight 的现代复兴**: Chelsea Finn 和 Sergey Levine 在 2017 年提出了 Visual Foresight (arXiv:1704.05543)，用 action-conditioned video prediction 做 MPC。$\tau_0$-WM 本质上是这个 idea 的超大模型化、latent space 化和 diffusion 化版本。它不再预测 pixel-level 的未来，而是在 latent space 中 rollout，并使用 learned reward head 来评估，极大地提升了 fidelity 和 horizon。
*   **World Models vs. Reactive Policies**: 类似于 Dreamer (Hafner et al., arXiv:1912.01603)，$\tau_0$-WM 证明了 model 不仅要 react，还要 dream。通过 ACVS，model 可以在执行前预演后果，这对于 long-horizon tasks (如 School Bag 任务需要拉拉链、放物体、再拉拉链) 至关重要。
*   **Inference Scaling Laws in Robotics**: OpenAI 的 o1 模型展示了 LLM 在 test-time 的 scaling。$\tau_0$-WM 将这一理念移植到 robotics。通过 RCS 和 LAR，模型在 inference 时可以 allocate 更多 compute 来 sample、verify 和 rectify actions。这暗示了 Robot Foundation Model 的性能提升路径不仅在于扩大 pre-training scale，还在于深化 test-time reasoning。

**References for further reading:**
*   Wan2.2 Video Generation Backbone: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
*   Flow Matching for Generative Modeling: [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
*   Visual Foresight (Early Action-Conditioned Video Prediction): [arXiv:1704.05543](https://arxiv.org/abs/1704.05543)
*   Dream to Control (World Models for Robotics): [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
*   Cosmos World Foundation Model (Action Conditioning design): [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
*   Universal Manipulation Interface (UMI): [arXiv:2402.10329](https://arxiv.org/abs/2402.10329)
