---
source_pdf: HorizonDrive Self-Corrective Autoregressive World Model for Long-horizon
  Driving Simulation.pdf
paper_sha256: 01988fe0b0da9d5bfc6e783bdc5702fa728ea874f3d730f118ab6927998a2b23
processed_at: '2026-08-04T23:56:58-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HorizonDrive 人话版

Project page: https://zcliangyue.github.io/HorizonDrive

---

## 想象一个场景

你想造一个 driving simulator，让 self-driving car 在里面训练。不是那种 replay 录像的 simulator，是那种 agent 转一下方向盘，world 就得跟着 render 出新画面的真·interactive simulator。

你手头有个很强的 video diffusion model（Wan 2.1 1.3B，https://github.com/Wan-Video/Wan2.1），能根据 text + HD map + bbox + ego action 生成 10 帧 driving video。逻辑上很直接——生成 10 帧，append 进 history，滑 window，再生成下 10 帧，如此 AR rollout 到 minute 级别。

你跑一下，第 3 个 chunk 就开始糊了，第 10 个 chunk 车道线歪了，第 20 个 chunk 整个 scene semantic 崩了。

这就是 **exposure bias** 在 video generation 里的经典表现。训练时 condition on clean GT history，推理时 condition on 自己生成的 dirty history，small error 递归 compound。

---

## 之前的人怎么 fix 的

有两条 main line：

**Line 1: Student-side degradation training。** Self-Forcing (https://arxiv.org/abs/2506.08009) 之类，训练 student 的时候就把 condition 替换成 student 自己之前生成的 frame，让它适应 rollout error。听起来很对，但有个 ceiling——student 的 supervision 还是来自 teacher 的 single-pass output。Teacher 一次 forward 最多生成 40 帧（再长 DiT attention $O(L^2)$ 显存爆），所以 student 学到的 corrective horizon 被 teacher 单次窗口 cap 死。你想训 student 稳定 200 帧 rollout，teacher 最多给你 40 帧的 supervision，剩下 160 帧靠 student 自己泛化——泛化不出来。

**Line 2: Frame sink / anchor。** StreamingLLM (https://arxiv.org/abs/2309.17453) 在 LLM 里保留几个 anchor token 让 attention 有个 "sink" 锚定。视频里也有人试（Huang 2024a, Zhou 2024）。但 driving 场景 ego-motion 太快，anchor frame 几步之后就 stale 了，锚不住。

---

## HorizonDrive 的 aha moment

作者问了个很关键的问题：**teacher 自己能不能 AR rollout 到任意长，给 unbounded supervision？**

答案是能，前提是 teacher 自己 rollout-capable——不会在自己生成的 dirty context 上 drift。

这个 insight 反直觉的地方在于：大家都默认 teacher 是金标准，teacher 怎么会 drift？问题是，标准 video diffusion model 训练时只见过 clean context，你让它 condition 在自己生成的 dirty frame 上 rollout，它跟 student 一样会 drift。Drift 的 teacher 给 drift 的 supervision，student 学到的就是 drift。

所以瓶颈不在 student，在 teacher。**Fix supervisor first。**

---

## Stage 1: SRR — 把 teacher 改成 rollout-capable

叫 Scheduled Rollout Recovery，核心 idea 特别朴素：

**拿 base model $\mathcal{G}_0$ 先 rollout 一遍，把生成的 dirty history 当训练 input，让模型学着从 dirty history recover 出 clean GT future。**

具体公式（Eq. 6, 7）：
- 用 $\mathcal{G}_0$ rollout N 步，得到 corrupted trajectory $\hat{\mathbf{z}}$
- 取一段，condition 部分用 $\hat{\mathbf{z}}$（dirty），supervision target 用 $\mathbf{z}^*$（clean GT）
- 模型学：给定 dirty past，生成 clean future

这就是 **DAgger in latent space**（Ross et al. 2011, https://arxiv.org/abs/1011.0686）。DAgger 解 exposure bias 的方法就是把 student deployment 时会 visit 的 state 加进训练 set。SRR 把 teacher 自己 rollout visit 的 dirty state 加进训练 set，一个意思。

但直接拼 dirty → clean 有个问题：boundary 处 discontinuity 太硬，模型不知道怎么从 dirty 平滑过渡到 clean。于是有 **local pred-to-GT transition**（Eq. 8）——在 boundary 半径 $w$ 内做 linear blend：

$$\bar{\mathbf{z}}_i = \alpha_i \tilde{\mathbf{z}}_i + (1-\alpha_i)\mathbf{z}_i^*$$

$\alpha_i$ 从 1（全 dirty）线性降到 0（全 clean），相当于在 latent 空间架一座 temporal bridge。

然后 $w$ 还要 schedule：训练开始 $w=0$（硬切，逼模型硬 recover），慢慢加到 $w=8$（smooth blend，fine-tune fine-grained correction）。这个 curriculum 是 "先难后易"——先逼模型学 hard recovery，再 refine。

还有 **global boundary-decay sampling**——训练 start index $s$ 的 curriculum。Figure 3 的分析很 informative：
- rollout 越深，error heatmap 越强（累积 drift 越严重）
- 但早期 error 是 cross-case 共享的（generic），晚期 error 是 case-specific 的（semantic corruption）

所以 schedule 反着来：先从大 $s$（late rollout, 严重 semantic drift）开始训，让模型先学从 severe failure recover；再 decay 到小 $s$（early rollout, generic error）refine。Table 4 里 AR depth $N(k): 10 \to 4$。

这两个 schedule（local $w$ + global $s$）合起来是 SRR 的灵魂。单看公式平淡无奇，但 curriculum 设计背后是对 rollout error 结构的 empirical 分析——早期 error 共享、晚期 error 个体化，所以先打 hardest mode 突破天花板。

---

## Stage 2: TRD — 蒸馏成 real-time student

SRR 训出的 $\mathcal{G}_{\text{roll}}$ 稳了，但慢——multi-step diffusion，没法 real-time interactive。

于是 distill 成 4-step student。用的是 DMD（Distribution Matching Distillation, https://arxiv.org/abs/2311.18883）的变种，叫 Teacher Rollout DMD。

DMD 的核心：student generator $G_\theta$ 配两个 score function（real 和 fake），gradient 推 student 分布向 teacher 对齐：
$$\nabla_\theta \mathcal{L}_{DMD} = \mathbb{E}_\tau\left[-(s_{\text{real}}(z_{(\tau)}) - s_{\text{fake}}(z_{(\tau)}))\right]\frac{\partial G_\theta}{\partial\theta}$$

$s_{\text{real}} - s_{\text{fake}}$ 本质是 KL$(p_{\text{fake}} \| p_{\text{real}})$ 的 score difference，push student 分布向 teacher 靠。

TRD 在 DMD 上做了两个关键 modification：

**Modification 1: Teacher 自己也 AR rollout。**
这又是那个 insight——想给 student long-horizon supervision，teacher 不能只 single-pass。Teacher 用固定 $(T=11, K^\mathcal{T}=40)$ window 自己 rollout，memory 恒定 $O(T + K^\mathcal{T})$，跟 total horizon 无关。Student 用 $(T=11, K^\mathcal{S}=10)$ 也 rollout。每当 student 累积 rollout 覆盖一个 $K^\mathcal{T}=40$ interval，对这段算 DMD gradient backprop。Teacher 在同区间 rollout 给 supervision direction。

这个 asymmetric 设计很妙：teacher 长 chunk 多步（稳定但慢），student 短 chunk 少步（快但 chunky）。Student 用多个短 chunk match 同一个 teacher 长 chunk，等于把 teacher 的 long-horizon 稳定性 "摊薄" 到 student 的 fast interaction 上。

**Modification 2: Noise-truncated CFG。**
标准 DMD 用 CFG 强化 teacher gradient，但 video rollout 里 full CFG oversaturation 严重——反复 push 同一 condition 导致 color/texture mode collapse。HorizonDrive 把 CFG 限制在低 noise level（$\tau \le \tau_{\text{th}}$），再 schedule $\tau_{\text{th}}$ decay。

物理直觉：diffusion 中 high noise level CFG steer semantic structure，low noise level CFG refine detail。Full CFG 在 long rollout 里反复 re-steer semantic，越 steer 越 saturated。Truncate 到低 $\tau$ 后 CFG 只 refine 不 re-steer，避免 rollout 中条件反复放大。

$\tau_{\text{th}}$ schedule: 1000（step 0–100 full-range）→ decay to 0（step 400）。先让 full CFG 把 conditional controllability 立起来，再退到低 noise 只做 refinement。Ablation Table 3 证实：Full CFG 直接 FVD 184.06（vs None 110.81），Delayed CFG 92.99 最优。

---

## 为什么这套 work

最 informative 的是 Table 3 ablation：

| Init (Stu/Tea) | FVD |
|---|---|
| Base/Base | 141.88 |
| SRR/Base | 128.77 |
| Base/SRR | 107.54 |
| SRR/SRR | 92.99 |

单因子最大 jump 是 **Base-student / SRR-teacher（141.88 → 107.54）**，不是 SRR-student / Base-teacher（141.88 → 128.77）。这直接证实核心论点：**teacher 端 rollout-capability 是 dominant factor**。

直觉：distill 一个会 drift 的 teacher，student 再聪明也只能学到 drift 轨迹。SRR 把 teacher 改成 rollout-capable，supervision 质量天花板就抬高了，student 自然学得好。Student-side SRR 也有帮助（SRR/SRR 比 Base/SRR 好），但不是 dominant。

---

## 结果有多强

nuScenes val 上：

vs streaming baselines（相同 base + control，唯一变量是 long-horizon framework）：
- FID 13.82 vs 28.84–41.53（降 52%）
- FVD 92.99 vs 147.57–161.41（降 37%）
- ARE 2.60 vs 3.28–3.78（降 21%）
- DTW 3.27 vs 3.61–6.22（降 9%）

vs domain-specific driving generators：
- 短 clip (21 frame): FVD 84.53，group 内最好
- 长 clip (211 frame, N=20 rollout): FVD 92.99，跟 MagicDrive-V2 的 single-pass 241-frame（94.84）打平，FID 还更好（13.82 vs 20.91）

关键 point：**AR rollout + few-step denoiser 能 match single-pass + many-step denoiser 的质量**，同时 memory bounded 支持 minute-scale。这是个 fundamental compute-quality trade-off 的实证。

Figure 5 最直观：19 个 cumulative chunk 的 FID 曲线，HorizonDrive 全程平，Self-Forcing++ 单调退化。AR rollout 稳定性的直接可视化。

---

## Minute-scale + closed-loop demo

Sec G 和 Sec H 是 paper 真正想 demo 的 use case。

**Minute-level rollout**：sliding window 让 per-step compute = $O(T + K^\mathcal{S})$，跟 total horizon 无关。5090 上 $256\times512$ 1.8s/chunk → 5.6 FPS，$384\times768$ 5.8s/chunk → 1.7 FPS。能 indefinite rollout。

**Closed-loop**（Sec H, Figure 14）：planner 吃 generated frame 出 ego trajectory → re-encode 成 next action condition → 喂回 HorizonDrive。完全 self-generated signal，没有 GT trajectory 参与。Road geometry 和 agent behavior 全程 coherent。这才是 closed-loop driving simulator 该有的样子——policy 走 OOD trajectory 时 world model 仍 stable。

SRR 训的 "从 corrupted context recover" 能力正好对应这个需求——policy 走出的 trajectory 对 world model 来说就是 corrupted context，SRR 让 model 在这种 distribution shift 下仍能 generate plausible future。

---

## Meta intuition

整个 paper 给我的 biggest takeaway 是个很 general 的 lesson：

**在 hierarchical distillation 中，supervisor 的 reliability 决定 student 的天花板。**

这个 lesson 在很多地方都成立：
- **RLHF**: reward model 在 long horizon 自己漂，PPO policy 再精学也学不到 long credit assignment。Fix reward model first。
- **Agent distillation**: 如果 teacher agent 在 OOD state 上行为 unreliable，student 模仿到的就是 unreliable behavior。Fix teacher's OOD robustness first。
- **Code generation**: 如果 reference solution 在 edge case 上 buggy，distill 出的 student model 学到的就是 buggy pattern。

HorizonDrive 把这个 lesson 在 video generation 里 instantiate 得很干净：SRR fix teacher rollout reliability → TRD distill 到 student → student 继承 long-horizon stability。两个 schedule（local blend radius, global boundary decay）是 SRR 的 curriculum 灵魂，asymmetric chunk + noise-truncated CFG 是 TRD 的 efficiency 灵魂。

---

## 几个我会 follow up 的方向

1. **Online SRR**: 作者承认 SRR 是 offline（cache 每 2000 step refresh）。Online 版边 rollout 边训，让 teacher 持续 improve 在当前 deployment distribution 上的 robustness，类似 online DAgger。这个 gap 我觉得挺大——offline DAgger 的 distribution coverage 永远 lag。

2. **SRR + Diffusion Forcing 结合**: Diffusion Forcing (https://arxiv.org/abs/2404.01132) 让每个 token 独立 noise level，自然支持 rollout training。SRR 的 corrupted context idea 跟 Diffusion Forcing 的 per-token noise schedule 应该能 combine——一边 per-token noise，一边 condition on self-rollout。可能 eliminate 对 explicit blend window 的依赖。

3. **SRR 的一般化**: 这个 "拿 base model rollout 生成 dirty data，再训 model 从 dirty recover clean" 的 recipe 应该对所有 AR generation task 通用——LLM text generation、audio generation、3D generation。LLM 里类似的 exposure bias 问题一直没彻底解决，SRR 的 curriculum 设计（先难后易，local blend + global boundary decay）可能 transfer。

4. **Closed-loop evaluation benchmark**: Sec H 的 closed-loop demo 很 compelling 但只是 qualitative。需要一个 systematic benchmark——不同 planner（rule-based, learned）× 不同 scenario × 不同 horizon，quantify world model 在 closed-loop 下的 fidelity 和 stability。这比 open-loop FID/FVD 更能反映 driving simulator 的实用价值。

5. **Teacher rollout depth vs student quality 的 scaling law**: Table 3 显示 N=1 → N=4 → N=20 单调提升。N=100 会怎样？N=1000？有没有 scaling law 形式？这对实际部署很重要——train-time rollout depth 决定 deployment-time horizon ceiling。

---

## TL;DR

Driving world model long-horizon AR rollout drift 的根因是 teacher 不 rollout-capable。HorizonDrive 用 SRR 把 teacher 改造成能从自己 dirty prediction 中 recover GT 的 self-corrective supervisor，再用 TRD 把 long-horizon corrective behavior 蒸馏成 real-time student。整个 framework 在 bounded memory 下跑 minute-scale closed-loop driving simulation，metric 全面碾压 streaming baselines。核心 meta lesson：**fix supervisor first，student 的天花板由 supervisor 决定**。

---

# HorizonDrive: 把 teacher 改造 rollout-capable，再蒸馏出 long-horizon student

Paper 链接：https://zcliangyue.github.io/HorizonDrive  
代码 / project page: https://zcliangyue.github.io/HorizonDrive

---

## 1. 一句话核心

现有 driving world model 的 AR rollout drift，根因是 **teacher 本身不 rollout-capable**——它在自己的预测上 rollout 会漂移，于是 distill 出来的 student 也跟着漂。HorizonDrive 用两阶段把标准 video diffusion 改造成 self-corrective rollout teacher，再 distill 成 real-time short-chunk student，minute-scale rollout 不爆。

---

## 2. Problem setup 与 exposure bias

闭环驾驶仿真要的是 *agent → world → agent* 反复交互，world model 必须 AR rollout：
$$\hat{\mathbf{z}}_{T+1:T+K} \sim p_\theta(\mathbf{z}_{T+1:T+K} \mid \mathbf{z}_{1:T}, \mathbf{c}_{T+1:T+K}) \quad \text{(Eq. 5)}$$
其中 $\mathbf{z}_{1:T}$ 是 history buffer（latent，video-VAE 空间），$\mathbf{c}$ 是 driving control（text / HD-map / bbox / ego-action $\mathbf{a}=(\Delta x, \Delta y, \Delta\text{yaw})$）。生成 K-frame chunk append 进 history，window 滑 K 帧，再下一步。

训练用 clean GT history，推理用自己生成的 history——classic exposure bias。Error 在每个 AR step 上叠加，semantic drift 几步就发散。Self-Forcing (https://arxiv.org/abs/2506.08009) 之类的方法只在 student 端做 degradation training，但 supervision horizon 仍受限于 teacher 单次 forward 的窗口（DiT attention 是 $O(L^2)$，长窗口显存爆炸）。

---

## 3. 关键 insight：rollout-capable teacher 才是 bottleneck

> "Long-horizon supervision is dominated by the teacher's rollout reliability."

Ablation Table 3 row 2 (SRR-student / Base-teacher) 比 row 1 (Base/Base) 只小幅提升；row 3 (Base-student / SRR-teacher) 是最大单因子 jump（FVD 141.88 → 107.54）。说明 distill 一个会 drift 的 teacher，student 再聪明也只能学到 drift 轨迹。

类比：在 LLM RLHF 中，如果 reward model 在 long horizon 上自身就漂，再精的 PPO policy 也学不到长期 credit assignment。这里 teacher 在自己 prediction 上 rollout 稳定，相当于一个 self-consistent supervisor。

---

## 4. 三阶段 pipeline

### Stage 0 — Base model $\mathcal{G}_0$ (Sec 4.1)

Backbone: **Wan 2.1 1.3B T2V** (https://github.com/Wan-Video/Wan2.1)，full bidirectional attention。VAE 把 temporal compression ratio 从 4 → 1，保 full temporal resolution（driving 场景 fast ego-motion，丢帧就糊）。

Disentangled control（沿用 Ren et al. 2025 / Zhan et al. 2026 思路）：
- **Spatial structure** (HD map, bbox) → render 成 $\mathbf{z}_{bf}\in\mathbb{R}^{c\times f\times h\times w}$，conv adapter reshaping 成 layout tokens $\mathbf{h}_{bf}\in\mathbb{R}^{f\times s\times d}$，以 zero-init projector 加进 feature：$\mathbf{h}_{(t)} \mathrel{+}= f_{\text{zero}}(\mathbf{h}_{bf})$。Zero-init 保证初始不破坏 pretrained T2V prior。
- **Ego action** $\mathbf{a}=(\Delta x, \Delta y, \Delta\text{yaw})\in\mathbb{R}^{F\times 3}$：sinusoidal embed → AdaLN gating（Peebles & Xie 2023, DiT paper https://arxiv.org/abs/2212.09748），6 channels 拆成 pre-norm shift/scale + post-layer residual gate。

训练用 flow matching v-prediction：
$$\mathcal{L}_{CFM} = \mathbb{E}_{\epsilon\sim\mathcal{N}(0,I)} \| v_\Theta(z_{(t)}, t) - (z_{(0)} - \epsilon) \|_2^2 \quad \text{(Eq. 3)}$$
$z_{(t)}=\sigma_t z_{(0)} + (1-\sigma_t)\epsilon$（Eq. 2，$\sigma_t$ 是 noise schedule），$v_\Theta$ 是 velocity field。Condition window noise level $t=0$，generation chunk 加噪 supervised。

$\mathcal{G}_0$ 直接 AR rollout 就 drift——它是后续 SRR 的起点。

### Stage 1 — SRR: Scheduled Rollout Recovery (Sec 4.2)

目标：把 $\mathcal{G}_0$ 改造成 rollout-capable $\mathcal{G}_{\text{roll}}$。

**Step A: 用 $\mathcal{G}_0$ 生成 corrupted rollout history 作为训练 input。**
固定 history buffer size T，AR rollout N 步：
$$\hat{\mathbf{z}}_{s_n+1:s_n+K} = \mathcal{G}_0(\hat{\mathbf{z}}_{s_n-T+1:s_n}, \mathbf{c}_{s_n+1:s_n+K}), \quad s_n = T+(n-1)K \quad \text{(Eq. 6)}$$
得到 corrupted trajectory $\hat{\mathbf{z}}_{T+1:T+NK}$，error 跨步累积。

**Step B: 构造 supervised pair（公式 7）。**
取 generation start index $s$，把 conditioning history 替换成 rollout prediction，supervision target 保持 GT future：
$$\tilde{\mathbf{z}}_{s-T+1:s} = \hat{\mathbf{z}}_{s-T+1:s}, \quad \mathbf{z}^*_{s+1:s+K} = \mathbf{z}_{s+1:s+K}$$

直接拼 boundary 处 discontinuity 太硬，于是 **local pred-to-GT transition**（Eq. 8），在 boundary 半径 $w$ 内 linear blending：
$$\bar{\mathbf{z}}_i = \begin{cases} \tilde{\mathbf{z}}_i & s-T+1 \le i \le s-w \\ \alpha_i \tilde{\mathbf{z}}_i + (1-\alpha_i) \mathbf{z}^*_i & s-w+1 \le i \le s+w \\ \mathbf{z}^*_i & s+w+1 \le i \le s+K \end{cases}$$
$\alpha_i$ 从 1 线性降到 0，做一条 latent 空间的 temporal bridge。Table 4 schedule：$w: 0 \to 8$（step 0–8000）。先 sharp boundary（强制模型 hard recover）→ 后 smooth region（fine-grained correction）。这个 curriculum 思路让我想到 Bengio 的 scheduled sampling 和 DAgger——先难后易 vs 先易后难各有道理，这里走的是 "先暴露 hard mode，再 refine"。

**Step C: global boundary-decay sampling——$s$ 的 curriculum。**

Figure 3(b)(c) 的 error 分析很有信息量：
- heatmap 越靠 late rollout boundary error 越强（更多 AR step 累积 semantic drift）；
- cross-case cosine similarity 在 **early boundary 区间最高**——早期 error 是 generic/cross-case 共享的，晚期 error 是 case-specific 的 semantic corruption。

据此 schedule：训练先从大 $s$（late rollout, 严重 semantic drift）开始，让模型先学从 severe failure 中 recover；再 decay 到小 $s$（early rollout, generic degradation）。Table 4 schedule：AR depth $N(k): 10 \to 4$（step 0–8000）。

这跟一般 curriculum（先易后难）反着来，因为他们观察到 severe semantic drift 是 **failure mode 的天花板**，先突破它，generic error 自然能修。这点很 Karpathy-style：先 overfit 难的，再 broad refine。

Cache 每 R=2000 optimizer steps refresh 一次（用当前 $\theta$ 重新 rollout），保证 corrupted context 跟得上 student 改进。

### Stage 2 — TRD: Teacher Rollout DMD (Sec 4.3)

$\mathcal{G}_{\text{roll}}$ 稳但慢（multi-step diffusion），distill 成 real-time student。

**Setup:**
- Teacher $\mathcal{G}_{\text{roll}}^\mathcal{T}$ 和 student $\mathcal{G}_{\text{roll}}^\mathcal{S}$ 都从 $\mathcal{G}_{\text{roll}}$ init；teacher frozen。
- 共享 context window $T=11$；teacher chunk $K^\mathcal{T}=40$ (multi-step)，student chunk $K^\mathcal{S}=10$ (4-step)。
- Asymmetry：teacher 长 chunk 多步给稳定长期 supervision，student 短 chunk 少步给 fast interaction。

**Long-horizon supervision without quadratic memory:**
关键 trick——不增加 teacher 单次 generation length（避免 $O(L^2)$ attention）。Teacher 自己用固定 $(T, K^\mathcal{T})$ 做 AR rollout，每步显存恒定。Student 也用固定 $(T, K^\mathcal{S})$ rollout：
$$\hat{\mathbf{z}}^S_{s_n+1:s_n+K} = \mathcal{G}^S_{\text{roll},\phi}(\hat{\mathbf{z}}^S_{s_n-T+1:s_n}, \mathbf{c}_{s_n+1:s_n+K}), \quad n=1,\ldots,N \quad \text{(Eq. 9)}$$

每当 student 累积 rollout 覆盖一个 $K^\mathcal{T}$ interval，对最近 $K^\mathcal{T}$ frames 算 DMD gradient 立即 backprop；frozen teacher 在同一区间 rollout 给 distribution-matching direction。Update interval D=5 student chunks。

**DMD 背景回顾（Yin et al. 2024, https://arxiv.org/abs/2311.18883）：** student generator $G_\theta$ 配 real/fake score functions $s_{\text{real}}, s_{\text{fake}}$，gradient:
$$\nabla_\theta \mathcal{L}_{DMD} = \mathbb{E}_\tau \left[ -(s_{\text{real}}(z_{(\tau)}) - s_{\text{fake}}(z_{(\tau)})) \frac{\partial G_\theta}{\partial \theta} \right] \quad \text{(Eq. 4)}$$
$\tau$ 是 renoise level。$s_{\text{real}} - s_{\text{fake}}$ 实质上是 KL$(p_{\text{fake}} \| p_{\text{real}})$ 的 score difference，push student 分布向 teacher 对齐。

**Noise-truncated CFG:** 标准 DMD 用 CFG 强化 teacher gradient，但视频 rollout oversaturation 严重。Decoupled DMD（Liu et al. 2025a, https://arxiv.org/abs/2511.22677）限制 CFG 到低 noise level。HorizonDrive 进一步 schedule $\tau_{\text{th}}$ decay。完整 TRD gradient（Eq. 10）：
$$\nabla_\phi \mathcal{L}_{TRD} = \mathbb{E}_\tau \left[ -\underbrace{(s^{\text{real}}_{\text{cond}}(z_{(\tau)}) - s^{\text{fake}}_{\text{cond}}(z_{(\tau)}))}_{\text{Distribution Matching}} - \underbrace{\mathbf{1}_{\{\tau \le \tau_{\text{th}}\}}(\alpha-1)(s^{\text{real}}_{\text{cond}}(z_{(\tau)}) - s^{\text{real}}_{\text{uncond}}(z_{(\tau)}))}_{\text{Noise-truncated CFG}} \right] \frac{\partial \mathcal{G}^S_{\text{roll},\phi}}{\partial \phi}$$

- 第一项：cond（条件 score）real vs fake 差，是 distribution matching 主力。
- 第二项：indicator $\mathbf{1}_{\tau \le \tau_{\text{th}}}$ 把 CFG 只开在低 noise level；$(\alpha-1)$ 是 CFG 权重（$\alpha=6$）；$s_{\text{cond}} - s_{\text{uncond}}$ 是 classifier-free guidance 的 score 形式。
- $\tau_{\text{th}}$ schedule: 1000 (step 0–100) → decay to 0 (step 400)。先 full-range CFG 建立强 conditional controllability，再限制到低 noise 保 visual refinement——ablation 中 Full 直接 FVD 184.06（oversaturation），Delayed 92.99。

---

## 5. 实验

### Table 1 — vs long-horizon baselines (nuScenes val)

| 类别 | Method | FID↓ | FVD↓ | ARE↓ | DTW↓ |
|---|---|---|---|---|---|
| World model frameworks (no driving ctrl) | Matrix-Game3 | 35.69 | 338.22 | N/A | N/A |
| | Helios | 30.53 | 218.23 | N/A | N/A |
| | Causal-Forcing | 49.07 | 373.29 | N/A | N/A |
| | HY-WorldPlay | 33.51 | 580.72 | N/A | N/A |
| | LingBot-World | 37.67 | 325.55 | N/A | N/A |
| Streaming recipes (re-trained on our base+ctrl) | Self-Forcing | 41.53 | 161.00 | 3.47 | 6.22 |
| | Self-Forcing++ | 28.84 | 147.57 | 3.78 | 3.61 |
| | LongLive | 29.05 | 161.41 | 3.28 | 3.65 |
| **Ours** | **HorizonDrive** | **13.82** | **92.99** | **2.60** | **3.27** |

FID 降 52%，FVD 降 37%，ARE 降 21%，DTW 降 9%。

底部三组用相同 base + driving control modules，唯一变量是 long-horizon training framework——这是个非常干净的 ablation 对照，证实 SRR+TRD 这套框架本身的优势，而不是 base model 或 control 的功劳。

### Table 2 — vs domain-specific driving generators

| Method | N | Frames | FID | FVD |
|---|---|---|---|---|
| DriveDreamer | 1 | 8 | 14.90 | 340.80 |
| Panacea | 1 | 8 | 16.90 | 139.00 |
| DreamForge | 1 | 16 | 14.61 | 103.61 |
| Vista | 1 | 25 | 6.90 | 89.40 |
| HorizonDrive (short) | 1 | 21 | 12.54 | **84.53** |
| MagicDrive-V2 | 1 | 241 | 20.91 | 94.84 |
| HorizonDrive (long) | 20 | 211 | **13.82** | **92.99** |

短 clip 上 Vista FID 略好（6.90 vs 12.54）但只有 T+A 两个 control、25 frames；HorizonDrive 短 clip 也支持 full T+M+B+A。长 clip 上 20 步 AR rollout 竟打平 MagicDrive-V2 的 single-pass 241-frame，验证 "sequential rollout + few-step denoising" 能 compete "many-step single-pass"。

### Table 3 — TRD Ablation

| Init (S/T) | N | CFG | FID | FVD | ARE | DTW |
|---|---|---|---|---|---|---|
| Base/Base | 20 | Delayed | 19.24 | 141.88 | 2.76 | 3.30 |
| SRR/Base | 20 | Delayed | 20.34 | 128.77 | 3.15 | 3.39 |
| Base/SRR | 20 | Delayed | 14.44 | 107.54 | 2.75 | 3.80 |
| SRR/SRR | 4 | Delayed | 21.15 | 135.35 | 3.39 | 5.42 |
| SRR/SRR | 20 | None | 14.59 | 110.81 | 3.03 | 3.66 |
| SRR/SRR | 20 | Full | 20.84 | 184.06 | 2.77 | 3.28 |
| SRR/SRR | 20 | Early | 14.70 | 111.99 | 3.86 | 3.13 |
| **SRR/SRR** | **20** | **Delayed** | **13.82** | **92.99** | **2.60** | **3.27** |

读这个表的关键 takeaways：
1. **Teacher-side SRR 是 dominant**：Base/SRR (107.54) ≫ SRR/Base (128.77) ≫ Base/Base (141.88)。Rollout-capable teacher 才是 long-horizon supervision 的 ceiling。
2. **N=20 vs N=4 vs N=1**：DTW 5.42 → 3.66 → 3.27；FVD 139.28 → 92.99。Student 必须在自己的 long rollout chain 上被 supervised 才能 deployment-time 稳。
3. **CFG 必须截断 + schedule**：Full 直接 oversaturation FVD 184.06；Delayed 给 warmup 再限制到低 noise，最优。

### Figure 5 — Error accumulation 曲线
HorizonDrive 19 个 cumulative chunk 的 FID 全程稳定；Self-Forcing++ 单调退化。这是 AR rollout 稳定性的最直接可视化证据。

### Self-collected e2e dataset (Table 6)
更高 ego speed、更多 scenario 多样性。HorizonDrive 12.01 / 117.27，LongLive 28.39 / 374.94。证明 SRR+TRD framework 不只 fit nuScenes。

### Minute-level rollout (Sec G, Figure 13)
sliding window 让 per-step compute = $O(T + K^\mathcal{S})$，与 total horizon 无关。5090 上 $256\times512$ 1.8s/chunk → ~5.6 FPS，$384\times768$ 5.8s/chunk → ~1.7 FPS，能 minute-scale。

### Closed-loop simulation (Sec H, Figure 14)
Planner + world model loop，每步 planner 吃 generated frame 出 ego trajectory → re-encode 成 next action condition → 喂回 HorizonDrive。完全 self-generated signal，coherent road geometry 维持。这是 paper 真正想 demo 的 use case。

---

## 6. Intuition 与跨领域联想

### (a) Rollout-capable teacher ≈ Self-consistent supervisor
类比 RLHF：如果 reward model 在 long horizon 自己就漂，PPO policy 再精学也学不到 long credit assignment。这里 teacher 在自己 prediction 上 rollout 稳定，相当于一个 self-consistent long-horizon supervisor。Self-Forcing 之类的方法相当于 "student 自己 rollout 但 teacher 还是 single-pass"——supervisor horizon 短，student 学不到 long corrective behavior。

### (b) SRR ≈ DAgger in latent space
DAgger (Dataset Aggregation, Ross et al. 2011, https://arxiv.org/abs/1011.0686) 通过 aggregate student-visited states 进 training set解决 exposure bias。SRR 用 $\mathcal{G}_0$ 自己 rollout 生成 corrupted states 作为训练 input，本质就是 DAgger——把 student deployment 时会遇到的 state distribution 加入训练分布。Local pred-to-GT blending window 类似 DAgger 中的 mixed roll-in；global boundary-decay 类似 "先 visit hardest off-distribution state，再 refine"。

### (c) Scheduled sampling in LLM
Bengio 2015 (https://arxiv.org/abs/1506.03099) scheduled sampling 训练 RNN 用自己输出 vs GT 的混合比例 schedule。这里 local blending radius $w: 0 \to 8$ 实际上是个反向 scheduled sampling——先 sharp exposure（high mismatch），再 smooth blend（low mismatch）。这是因为 diffusion 在 latent space 已有强 prior，sharp boundary 反而是个 useful hard example。

### (d) Frame sink vs SRR
StreamingLLM (Xiao et al. 2023, https://arxiv.org/abs/2309.17453) 用 attention sink + rolling window 让 LLM 无限长 generate。视频生成里类似思路（Huang 2024a, Zhou 2024）保留 anchor frames。但 driving 场景 fast ego-motion 让 sink frame 信息快速 stale，不适用。SRR 的角度反过来——不靠 sink 锚定，而是训 model 自己从 stale/corrupted context recover。这跟 driving 的 inductive bias 更合。

### (e) Diffusion Forcing (Chen et al. 2024)
Diffusion Forcing (https://arxiv.org/abs/2404.01132) 把 next-token prediction 和 full-sequence diffusion 统一，每 token 独立 noise level，自然支持 rollout training。HorizonDrive 走的是另一条路——固定 condition/generation 二分（前者 $t=0$，后者加噪 supervised），rollout 部分通过 SRR 的 corrupted context 显式建模。两条路都能 reduce exposure bias，但 HorizonDrive 不改 DiT attention 结构，更适配现有 pretrained T2V。

### (f) Asymmetric distillation
Teacher 长 chunk 多步、student 短 chunk 少步的 asymmetry 让我想到 consistency model (Song 2023, https://arxiv.org/abs/2303.01469) 中的 teacher→student distillation，但这里多了 "teacher 自己也 rollout" 这一维。Memory budget 是 bounded 的，因为 sliding window 让 teacher per-step compute = $O(T+K^\mathcal{T})$；这个 bounded-memory under arbitrary horizon 是支持 minute-scale rollout 的关键。

### (g) CFG schedule 的物理直觉
Diffusion 中 CFG 在 high noise level ($\tau$ 大) 主要 steer semantic structure，low noise level 主要 refine detail。Full CFG 在 long rollout 中 oversaturation 的根因是 high-noise CFG 反复 push 同一 condition，导致 mode collapse 到 saturated color/texture。Truncate to low $\tau$ 后 CFG 只 refine，不再 re-steer——这就避免了 rollout 中条件反复放大。Delayed decay 给 warmup 阶段 full CFG 把 conditional controllability 立起来，再逐步退到低 noise 只做 refinement。

### (h) Limitation
作者自承 SRR 是 offline——rollout cache 每 R=2000 step refresh 一次。Online rollout-recovery（边 rollout 边训）是 future work。这个 limitation 跟 DAgger 在真实 RL 中遇到的 distribution shift 类似——offline DAgger 是 fixed dataset，online DAgger 是 interactive。Online 版本能让 teacher 持续 improve 在自己当前 deployment distribution 上的 robustness。

### (i) Closed-loop evaluation 的意义
Sec H 真正 closed-loop（planner ↔ world），全 self-generated signal——这是 paper 最有价值的 demo。现有 driving video generation 大部分是 open-loop（GT trajectory → video），而闭环测试需要 world model 在 policy 选择的 trajectory（OOD）下仍 stable。SRR 训的 "从 corrupted context recover" 能力正好对应这个需求——policy 走出的轨迹对 world model 来说就是 corrupted context。

### (j) 跟 Cosmos-Wonder-Dreams / MagicDrive-V2 的关系
Cosmos (Ren et al. 2025, https://arxiv.org/abs/2506.09042) 是 single-pass 241-frame；MagicDrive-V2 (Gao et al. 2025a) 也是 single-pass long clip。它们用 single-pass 大窗口 + 多步 denoiser，memory 是 $O(L^2)$。HorizonDrive 用 sequential rollout + few-step，memory 是 $O(T+K)$，长 horizon 上更 scalable，质量打平甚至更好。这是个 fundamental compute-quality trade-off 的实证——AR + fast denoiser 可以 match single-pass + slow denoiser。

---

## 7. 一句话总结给我的 intuition

**Long-horizon AR distillation 的瓶颈在 teacher 端的 rollout-capability，而非 student 端的 degradation training**。SRR 把 teacher 改造成能从自己 corrupted prediction 中 recover GT 的 self-corrective supervisor，再通过 TRD 把这个 long-horizon corrective behavior 蒸馏成 real-time student。两个 schedule（local blend radius、global boundary decay）是 SRR 的 curriculum 灵魂；asymmetric chunk + noise-truncated CFG 是 TRD 的 efficiency 灵魂。整个 framework 在 bounded memory 下跑 minute-scale closed-loop driving simulation，比 streaming baselines 降 52% FID / 37% FVD，ARE / DTW 也降 21% / 9%。

这工作的 meta lesson 我觉得是：**在 hierarchical distillation 中，fix supervisor first**。Student-side tricks 能 squeeze 几个百分点，但如果 supervisor 本身在 deployment regime 不可靠，student 的天花板就被 supervisor lock 住。这个 insight 在 RLHF、agent distillation、video rollout 里应该都成立。
