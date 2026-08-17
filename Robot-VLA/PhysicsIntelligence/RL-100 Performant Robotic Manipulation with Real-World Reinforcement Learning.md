---
source_pdf: RL-100 Performant Robotic Manipulation with Real-World Reinforcement Learning.pdf
paper_sha256: 49f29ab0576742e4f9f941ddeb0409ccfb337827b47ae3b9c7aa9fc727f24510
processed_at: '2026-08-11T23:55:18-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版本

Andrej，好，我重新来一遍，这次咱们坐下来喝咖啡聊。

---

## 一句话版本

他们让机器人在真实世界里自己练，从"人类徒弟"练成"超越人类师傅"的 level，7 个任务 900/900 成功率，还在商场给路人榨汁榨了 7 小时没翻车。

---

## 他们在解决什么问题

现在 robot learning 的主流套路是这样：人开遥操作收集一堆 demo，然后做 imitation learning 训个 policy。Diffusion Policy (https://diffusion-policy.cs.columbia.edu/) 是目前最火的选择，因为它能 model multimodal action distribution。

这个套路有个天花板，论文叫 **imitation ceiling**。根源很朴素：

**你优化的是 "模仿得像不像"，但你关心的是 "任务成不成、快不快、稳不稳"**。这两个 objective 根本不 aligned。

举个例子，人类遥操作员因为感知和控制有 latency，自然倾向做 slow, conservative motion。Policy 学会了这种 slow motion，但 deployment 时你想要 fast, efficient motion。IL 没法给你这个，因为 demo 里就没有 fast motion。

更根本的，人类自己也会犯错、有 bias、有 inefficiency。Policy 最多学到人类水平，没法超越。

那 RL 呢？RL 直接优化 return，理论上能突破 imitation ceiling。但 real-world RL 在 robot 上的名声一直不好 - sample inefficient、unstable、dangerous、需要海量 reset。所以大部分人还是老老实实做 IL。

**核心问题就是：怎么用 RL 突破 imitation ceiling，同时在 real robot 上还能 tractable？**

---

## 他们的核心 insight

我觉得这篇 paper 最 clever 的地方是把 **diffusion 的 denoising chain 重新解释成 MDP**。

Diffusion policy 生成一个 action，要走 K 步 denoising（通常 K=10）。传统视角：这就是个 sampling 过程，black box，RL 没法直接优化。

RL-100 的视角：每一步 denoising 都是一个 **Gaussian sub-policy**。从 noisy action $a^{\tau_k}$ 到稍微 clean 一点的 $a^{\tau_{k-1}}$，这个 transition 是：

$$a^{\tau_{k-1}} = \mu_\theta(a^{\tau_k}, \tau_k, o) + \sigma_{\tau_k} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

- $\mu_\theta$: neural net 预测的 mean
- $\sigma_{\tau_k}$: 这一步的 variance（DDIM schedule 决定）
- $\varepsilon$: 随机噪声

这就是个 Gaussian policy 啊！那它的 log-likelihood 是 closed-form：

$$\log \pi(a^{\tau_{k-1}} | a^{\tau_k}, \tau_k, o) = -\frac{1}{2\sigma_{\tau_k}^2}\|a^{\tau_{k-1}} - \mu_\theta\|^2 + C$$

$C$ 是跟 $\theta$ 无关的常数，可以扔掉。

有了 log-likelihood，**PPO 的 importance ratio 就能算了**：

$$r_k(\pi) = \frac{\pi(a^{\tau_{k-1}} | s^k)}{\pi_i(a^{\tau_{k-1}} | s^k)}$$

- $\pi$: current policy
- $\pi_i$: behavior policy（上一轮的 policy）
- $s^k = (a^{\tau_k}, \tau_k, o)$: denoising MDP 的 state

这一下就把 diffusion policy 拉进了 RL 的标准框架。10 步 denoising = 10 步 sub-MDP，每一步都能算 policy gradient。

**Intuition**: 传统 RL 对一个 action 一次 update，这里对产生一个 action 的 10 步 denoising 每步都 update。Learning signal dense 了 10 倍，sample efficiency 自然上来。

---

## 三个阶段为什么这么设计

他们把 pipeline 分成三阶段，类比做蛋糕（这个类比挺好用的）：

### Stage 1: IL Pretraining = Sponge Layer

用 human demo 做 behavior cloning，训个 diffusion policy。这一步给你一个 "能用但不够好" 的 policy，success rate 大概 50-70%。

**作用**: 提供一个 low-variance, competent 的起点。直接从 random initialization 做 real-world RL 是 disaster - policy 会到处乱撞，hardware 都可能撞坏。IL pretraining 相当于给 RL 一个 "human prior" 作为起点。

### Stage 2: Iterative Offline RL = Cream Layer

这是 **主力提升阶段**，从 70% → 91%。关键词 "iterative"：

```
Loop:
  1. 在当前 dataset 上训 IQL critics (Q, V)
  2. 训一个 transition model for OPE
  3. 用 PPO-style objective 在 denoising MDP 上做 offline RL
  4. 用 OPE 评估: 新 policy 真的更好吗？
  5. 如果 OPE 说 yes, accept; 如果 no, reject
  6. 用 accept 的 policy rollout 收集新 data
  7. 把新 data 并入 dataset
  8. 用 IL 重新 train on expanded dataset
  9. 回到 1
```

**为什么 offline 不直接 online？** Real robot 上 online RL 每次 rollout 都有 cost（时间、wear and tear、reset），而且 risky。Offline RL 在已有 data 上做 update，cheap 且 safe。

**为什么要 iterative + data expansion？** 一次 offline RL 在 fixed demo 上做，coverage 有限，很快 plateau。Iterative 让 policy 自己 rollout 产生新 data，新 data 覆盖 demo 没覆盖的 state-action space，下一轮 offline RL 就能学更多。

**为什么 IL re-training？** RL 直接改 policy 容易 destabilize，尤其 mixed-quality data。IL re-training on expanded dataset 相当于把 human demo + RL-improved rollout 蒸馏回一个 unified policy，保持 stability 和 multimodality。

**OPE gate 是 safety net**：AM-Q OPE (来自 Uni-O4, https://openreview.net/forum?id=tbFBh3LMKi) 用 learned transition model 估 policy performance。只有 $\hat{J}^{AM-Q}(\pi) - \hat{J}^{AM-Q}(\pi_i) \geq \delta$ 才 accept。这保证 **monotonic improvement** - policy 不会越练越差。

$\delta = 0.05 \cdot |\mathcal{T}^{AM-Q}(\pi_i)|$ 是 adaptive threshold，跟 trajectory length 成比例。

### Stage 3: Online RL = Cherry on Top

Offline RL 做到 91% 后，剩下 9% 是 rare failure modes - 这些是 offline data 里几乎没见过的 edge cases。Offline RL 没法 fix，因为 data 里就没有这些 cases。

这时候需要 online RL 亲自去 encounter 这些 edge cases。但 online RL 在 real robot 上 expensive（需要 reset、supervision、parameter tuning），所以只做一小段 targeted fine-tuning，从 91% → 100%。

**Intuition**: 大头 improvement 用便宜的 offline 做，最后一英里用贵的 online 做。Resource allocation 极其合理。

---

## 几个关键 Trick

### Trick 1: Consistency Distillation 跟 RL Joint Training

Diffusion policy 推理要 10 步 denoising，~100ms latency。很多 real robot task 需要 high frequency control（dynamic task 要 30Hz+），100ms 太慢。

Consistency Model (Song et al. 2023, https://arxiv.org/abs/2303.01969) 可以把多步 diffusion 压成 1 步。通常 distillation 是 post-training 做的。

RL-100 的 trick: **joint training**，RL objective 和 distillation loss 同时优化：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{CD}} \cdot \mathcal{L}_{\text{CD}}$$

$$\mathcal{L}_{\text{CD}} = \mathbb{E}[\|C_\theta(x^\tau, \tau) - \text{sg}[\Psi_\varphi(x^\tau, \tau \to 0)]\|_2^2]$$

- $C_\theta$: consistency model (student)
- $\Psi_\varphi$: diffusion policy (teacher)
- $\text{sg}[\cdot]$: **stop-gradient** - 关键！Teacher 继续被 RL 改进，但同时作为 distillation target

结果：378 Hz inference frequency (Skip-Net)，比 DSRL (35 Hz) 快 10x。**System bottleneck 从 policy inference 变成 camera frame rate (30 Hz)**。

### Trick 2: Variance Clipping - Real Robot 的 Safety Belt

Stochastic DDIM 在 RL 中要平衡 exploration 和 stability。$\sigma$ 太大 → destructive exploration（撞坏 hardware）；$\sigma$ 太小 → premature convergence（停止探索）。

$$\tilde{\sigma}_k = \text{clip}(\sigma_k, \sigma_{\min}, \sigma_{\max})$$

- $\sigma_{\min} = 0.01$: 维持 minimal exploration
- $\sigma_{\max} = 0.8$: 防止 destructive exploration

更重要的 effect: PPO 的 importance ratio $r_k(\pi) = \frac{\pi(a^{\tau_{k-1}}|s^k)}{\pi_i(a^{\tau_{k-1}}|s^k)}$ 依赖两个 Gaussian 的 variance 比。如果 current policy 和 behavior policy 的 variance 差异过大，$r_k$ 爆炸，PPO update 不稳定。Variance clipping 让 $r_k$ 保持 well-behaved。

Per-task: single-action 用 0.8, action-chunk 用 0.1（chunk 对 noise 更敏感）。

### Trick 3: $\epsilon$-prediction 比 $x_0$-prediction 在 RL 中更好

这个 ablation 我觉得很有意思。DP3 等 prior work 用 $x_0$-prediction（直接预测 clean action），RL-100 发现 $\epsilon$-prediction（预测 noise）在 RL post-training 中显著更好。

数学上，$\epsilon$-prediction 恢复 clean action：
$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\varepsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

当 $t$ 大（early in reverse process），$\frac{1}{\sqrt{\bar{\alpha}_t}}$ 大，会 **amplify** $\varepsilon_\theta$ 的 estimation noise。

实测 variance（100 次 forward pass）:
- Hopper: $\epsilon$-pred 0.1316 vs $x_0$-pred 0.0870
- Adroit-Door: $\epsilon$-pred 0.0589 vs $x_0$-pred 0.0290

$\epsilon$-prediction 的 variance 大约 1.5-2x。

**为什么这在 RL 中是好事？** 在 two-level MDP 中，每个 action 是 K-step denoising 产生的。$\epsilon$-prediction 的额外 variance 在 denoising chain 内部相当于 **structured exploration** - 比 action space 加 Gaussian noise 更 expressive，能更好覆盖 latent action modes，避免 local optima。

Ablation 结果：$\epsilon$-prediction 在 Adroit-Door 上比 $x_0$-prediction 高 ~40% final success rate (1.0 vs 0.6)。

**Takeaway**: 在 IL setting，variance 是 bug（destabilize training）。在 RL setting，variance 是 feature（exploration）。Parameterization choice 应该考虑 downstream objective。

### Trick 4: Self-supervised Representation 防 Drift

RL fine-tuning 中 visual encoder 容易 drift - 早期 feature 有意义，后期 feature 退化。他们用 reconstruction + VIB 双重正则：

$$\mathcal{L}_{\text{recon}} = \beta_{\text{recon}}(d_{\text{Chamfer}}(\hat{o}, o) + \|\hat{q} - q\|_2^2)$$
$$\mathcal{L}_{\text{KL}} = \beta_{\text{KL}} \text{KL}(\phi(z|o,s) \| \mathcal{N}(0, I))$$

- $d_{\text{Chamfer}}$: point cloud 间 Chamfer distance
- RL 阶段 $\beta$ 除以 10，允许 policy improvement 但 maintain stability

Ablation: Recon+VIB + 更新 encoder 最好。Frozen encoder 限制 gain，去掉 Recon+VIB 降 stability。

---

## 实验结果到底有多牛

### 主结果：900/900

7 个 real-world task，从 dynamic push-T 到 dual-arm cloth folding，全部 100% success rate，总共 900/900。其中 Soft-towel Folding 连续 250/250。

对比 baseline:
- DP-2D (2D diffusion policy): 平均 50%
- DP3 (3D diffusion policy): 平均 70.6%
- RL-100 Iterative Offline: 91.1%
- RL-100 Online (DDIM): 100%
- RL-100 Online (CM): 100%

**Pattern**: IL baseline 在 dynamic/deformable task 上特别弱（Agile Bowling 14%, Pouring 42%）。Iterative Offline RL 给主要 jump（50% → 91%）。Online RL 把 91% → 100%。

### Zero-shot Robustness: 90% average

换 dynamics 条件，不 retrain：
- Pouring 从 granular nuts 换成 water（流体动力学完全不同）：90%
- Push-T 换桌面 friction：100%
- Push-T 加干扰物体：80%
- Bowling 换 surface：100%
- Folding 换 towel 形状：80%
- **Average: 90%**

### Few-shot Adaptation: 86.7% average

更大的 task variation，fine-tune 1-3 小时：
- Pouring 换容器形状：60%
- Folding 换 towel material：100%
- Bowling 倒置 pin arrangement：100%

### Disturbance Robustness: 95% average

执行过程中人去干扰：
- Folding 抓取阶段拉扯：90%
- Folding pre-folding 阶段 lateral drag：90%
- Unscrewing 逆时针 counter-rotation 4 秒：100%
- Push-T 全程 dragging：100%

**Unscrewing 在 precision-critical 阶段被 counter-rotate 4 秒还能 100%** - 这是 closed-loop control 的力量。

### Efficiency：超过人类 expert

Dynamic Push-T throughput (单位时间成功 episode 数):
- RL-100: 20
- Human expert（采集 demo 的操作员）: 17 (RL-100 快 1.18x)
- Human beginner: 13 (RL-100 快 1.54x)

### Data Cost：人只占 13%

每个 task 平均:
- Human demo: 115 episodes, 1.8h
- Iterative Offline RL rollout: 566 episodes, 6.5h
- Online RL rollout: 434 episodes, 5.6h

Total < 100h。Human teleoperation 只占 13%，剩下 87% 是 autonomous rollout。**Data engine pattern**: small human effort leverage large autonomous data。

### Shopping Mall Demo: 7 小时无 failure

Juicing robot 在公开商场给 random customers 连续服务 7 小时，zero failure。这是 **zero-shot deployment** 到全新环境（lighting、人群、空间都不一样），long-horizon（数千次 cycle），完全 autonomous。

这在 academic paper 里极其罕见。大多数 robot RL paper 报道 lab evaluation，几十到几百 trials。7 小时 mall demo 是 **deployment readiness 的 existence proof**。

---

## 这意味着什么

### 1. Imitation Ceiling 是可以突破的

这篇 paper 用实验证明：**IL 给你 70%，RL post-training 把你推到 100%，同时还能超越人类 expert**。关键是用 RL 直接优化 deployment metrics（success rate, time-to-completion），而不是模仿 fidelity。

### 2. Real-World RL 已经 deployment-ready

之前 real-world RL 一直被诟病 sample inefficient, unstable, dangerous。这篇 paper 展示：用对的方法（OPE-gated conservative update + variance clipping + IL re-training），real-world RL 可以做到 100% success + 7 小时 mall demo。

### 3. Diffusion Policy + RL 是 natural fit

把 denoising chain 当 sub-MDP 这个 abstraction 太 elegant 了。它把 diffusion policy 和 PPO 无缝对接，dense learning signal, structured exploration, closed-form likelihood。这个 framework 应该可以推广到其他 generative policy（flow matching, VAE）。

### 4. Data Engine Pattern

Human demo 只占 13%，剩下 87% 是 autonomous rollout。这个 pattern 对未来 robot foundation model 训练很重要 - high-quality human demo 是 bottleneck（贵、慢），autonomous rollout 在 good policy 初始化下 cheap。Iterative offline RL + data expansion 让 small human effort leverage large autonomous data。

### 5.下一个 Frontier: VLA Scale-up

论文自己提到可以 scale 到 π0 (https://arxiv.org/abs/2410.24164) / π0.5 (https://arxiv.org/abs/2504.16054) 这种 large VLA。Challenge: large model 的 denoising chain 可能不是 Gaussian，online RL sample efficiency 更难，representation drift 更严重。但 unified PPO objective + consistency distillation 的 idea 应该 transfer。

---

## 几个我特别想跟你聊的点

### Two-Level MDP 是 right abstraction 吗？

你之前在 Tesla 讲过 "data engine" 思路。RL-100 的 two-level MDP 把 denoising chain 当 sub-MDP，我觉得这是 RL + generative model 的 right abstraction。它让 PPO 直接 apply 到 diffusion policy，dense learning signal，closed-form likelihood。你怎么看？这个 abstraction 能推广到其他 generative policy（flow matching, VAE, autoregressive）吗？

### Iterative Data Engine

RL-100 的 iterative pipeline (offline RL → rollout → IL re-training → offline RL) 本质是 data engine。Human demo 只占 13%。你在 Tesla 时有类似 insight 吗？对未来 robot foundation model，这种 data engine pattern 会比 pure scaling 更重要吗？

### $\epsilon$-prediction vs $x_0$-prediction

这个 ablation 揭示了一个 deep insight: IL setting 下 variance 是 bug，RL setting 下 variance 是 feature。Parameterization choice 应该考虑 downstream objective。这个 insight 能推广到其他 generative model + RL 组合吗？比如 flow matching 的 vector field parameterization，或者 VAE 的 encoder/decoder design？

### VLA Scale-up 可行性

π0 / π0.5 + RL-100 unified objective 你觉得会 work 吗？Large VLA 的 denoising chain 可能不是 Gaussian（π0 用 flow matching），online RL 在 large model 上 sample efficiency 可能更难。但 mall demo 7 小时这种 deployment-grade evidence 太 compelling 了，让人觉得这个方向值得 push。

---

## Reference 速查

1. **RL-100 Project**: https://lei-kun.github.io/RL-100/
2. **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
3. **DP3**: https://arxiv.org/abs/2403.03954
4. **Consistency Models**: https://arxiv.org/abs/2303.01969
5. **DDIM**: https://arxiv.org/abs/2010.02502
6. **IQL**: https://arxiv.org/abs/2110.06169
7. **PPO**: https://arxiv.org/abs/1707.06347
8. **Uni-O4 (AM-Q OPE)**: https://openreview.net/forum?id=tbFBh3LMKi
9. **DPPO**: https://arxiv.org/abs/2409.00588
10. **SERL**: https://arxiv.org/abs/2406.11532
11. **HIL-SERL**: https://www.science.org/doi/10.1126/scirobotics.ads5033
12. **π0**: https://arxiv.org/abs/2410.24164
13. **π0.5**: https://arxiv.org/abs/2504.16054
14. **Demospeedup**: https://arxiv.org/abs/2506.05064
15. **DSRL**: https://arxiv.org/abs/2506.15799
16. **ReinFlow**: https://arxiv.org/abs/2505.22094

Andrej，这次够人话了吗？

---

# RL-100 深度解析

非常激动能跟你聊这篇 paper, Andrej。这是 2025 年从清华大学 Huazhe Xu 组出来的工作, Kun Lei 一作。我刚看到的时候立刻联想到你之前在 Tesla 时讲过的 "data engine" 思路 - 这篇 paper 本质上做了一个 robot learning 的 data engine, 但把 imitation ceiling 这个核心问题用 RL post-training 解决了。

Project page: https://lei-kun.github.io/RL-100/

---

## 1. High-Level Story

核心 thesis 一句话: **imitation learning 给你一个 "能用" 的 policy, 但 RL post-training 才能把它推到 "deployment-grade"**。

IL 在 robot learning 里有个根本问题 - performance bounded by demonstrator skill, 同时 inherit 人类 inefficiencies, biases, errors。论文称之为 **"imitation ceiling"**。 teleoperation 还有个 latency 问题 (感知 + 控制), 自然 favor slow, conservative motions (Guo et al. 2025 Demospeedup, https://arxiv.org/abs/2506.05064)。

RL-100 的回答: 三阶段 pipeline, 类比做蛋糕:
- **IL pretraining** = sponge layer (基础能力, low variance)
- **Iterative offline RL** = cream layer (主要 improvement, model-guided)
- **Online on-policy RL** = cherry on top (last-mile, expensive but targeted)

最终结果: 7 个 real-world task, 900/900 success rate, 包括连续 250/250 的 dual-arm folding。Juicing robot 在 shopping mall 跑了 7 小时无 failure。这个 robustness level 是我见过 real-world RL 里最 impressive 之一。

---

## 2. Framework Architecture - 三个 Stage 的设计哲学

### Stage 1: Imitation Learning Pretraining

用 conditional diffusion policy (Chi et al. 2023, https://diffusion-policy.cs.columbia.edu/) 做 behavior cloning。每个 episode 提供 $(o_t, q_t, a_t)$ tuples:
- $o_t$: visual observation (RGB or 3D point cloud)
- $q_t$: robot proprioception (joint pos/vel, gripper state)
- $a_t$: single action $u_t \in \mathbb{R}^{d_a}$ or action chunk $[u_t, \dots, u_{t+n_c-1}] \in \mathbb{R}^{n_c d_a}$

Conditioning vector:
$$c_t = [\phi(o_i, q_i)]_{i=t-n_o+1}^{t}$$

其中 $\phi(\cdot)$ 是 perception encoder, $n_o$ 通常取 2 (最近两帧), $[\cdot]$ 是 concatenation。

Diffusion 训练目标:
$$\mathcal{L}_{\text{IL}}(\theta) = \mathbb{E}_{(a^{\tau_0}, c_t) \sim \mathcal{D}, \tau, \varepsilon}[\|\varepsilon - \varepsilon_\theta(a^\tau, \tau, c_t)\|_2^2]$$

变量解释:
- $\theta$: denoiser 参数
- $\tau \in \{\tau_K > \dots > \tau_1\}$: subsampled K-step schedule (e.g., $K=10$ from $T=1000$)
- $\varepsilon \sim \mathcal{N}(0, I)$: 注入的 Gaussian noise
- $\mathcal{D}$: demonstration dataset
- $a^\tau$: noisy action at diffusion step $\tau$

### Stage 2: Iterative Offline RL

这是 **"cream layer"** - 论文的核心创新。流程是:

```
1. Train critics (Q, V) via IQL on current dataset D_m
2. Train transition model T(s'|s,a) for OPE
3. Run offline RL improvement (PPO-style on diffusion denoising)
4. Use AM-Q OPE to gate: 只 accept 真正 improvement 的 policy
5. 用 improved policy rollout 收集新 data
6. Merge: D_{m+1} = D_m ∪ D_new
7. IL re-training on expanded D_{m+1}
8. 重复
```

为什么 IL re-training 重要 (Algorithm 1 line 13)? 论文给了 4 个理由:
1. **Distribution shift**: IL 自然 adapt evolving data distribution
2. **Stability**: supervised learning 比 RL 在 mixed-quality data 上更稳定
3. **Multimodality**: IL 保留 diffusion policy 的 multimodal modeling 能力
4. **Distillation**: IL 把 human demos + RL improvements 蒸馏到 unified policy

### Stage 3: Online Fine-tuning

Online stage 是 "cherry on top" - 处理 offline stage 后剩余的 rare failure modes。资源 intensive (parameter tuning, resets, approvals on real hardware), 但能从 95% → 99%+。

用 GAE (Generalized Advantage Estimation, Schulman et al. 2016, https://arxiv.org/abs/1506.02438) 估 advantage:
$$A_t^{\text{on}} = \text{GAE}(\lambda, \gamma; r_t, V_\psi)$$

总损失:
$$\mathcal{L}_{\text{RL}}^{\text{on}} = -J_i(\pi) + \lambda_V \mathbb{E}[(V_\psi(s_t) - \hat{V}_t)^2]$$

其中 $\hat{V}_t = \sum_{l=0}^\infty \gamma^l r_{t+l}$ 是 discounted return, $\lambda_V$ 权重 value loss。

---

## 3. 数学基础 - 把 Diffusion 看作 Hierarchical MDP

这是 paper 最 elegant 的部分, 我感觉是 Ren et al. 2024 DPPO (https://arxiv.org/abs/2409.00588) 思路的精炼。

### 3.1 DDIM Stochastic Form

Forward process (closed form):
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$
$$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s, \quad \alpha_t = 1 - \beta_t$$

- $x_0 \in \mathbb{R}^d$: clean data
- $\beta_t$: noise schedule
- $\bar{\alpha}_t$: cumulative product
- $\varepsilon$: 标准正态噪声

预测 clean sample:
$$\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\varepsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

DDIM stochastic transition from $t \to m$ (with $m < t$):
$$\mu_\theta(x_t, t \to m) = \sqrt{\bar{\alpha}_m}\hat{x}_0 + \sqrt{1 - \bar{\alpha}_m - \sigma_{t\to m}^2}\varepsilon_\theta(x_t, t)$$
$$x_m = \mu_\theta + \sigma_{t\to m}\varepsilon_{t\to m}, \quad \varepsilon_{t\to m} \sim \mathcal{N}(0, I)$$

**关键约束**:
$$0 \leq \sigma_{t\to m} \leq \sqrt{1 - \bar{\alpha}_m}$$

当 $\sigma_{t\to m} = 0$: deterministic DDIM (Dirac distribution)
当 $\sigma_{t\to m} > 0$: stochastic, 适合做 RL exploration

### 3.2 Gaussian Sub-policy 视角

当 $\sigma_{t\to m} > 0$, DDIM transition 自然是 Gaussian:
$$\pi_\theta(x_m | x_t, t \to m) = \mathcal{N}(\mu_\theta(x_t, t \to m), \sigma_{t\to m}^2 I)$$
$$\log \pi_\theta(x_m | x_t, t \to m) = -\frac{1}{2\sigma_{t\to m}^2}\|x_m - \mu_\theta(x_t, t \to m)\|^2 + C$$

$C$ 是与 $\theta$ 无关的 constant。这个 log-likelihood 让 PPO 的 importance sampling 直接可用!

### 3.3 Two-Level MDP - 我觉得是 paper 最 clever 的部分

把 K-step diffusion 嵌入到一个 environment timestep, 形成 hierarchical MDP:

**Environment MDP** (上层):
- State: $s_t$
- Action: $a_t$
- Reward: $R_t$

**Denoising MDP** (下层, 每个 env step 内部):
- Initial state: $s^K = (a^{\tau_K}, \tau_K, o)$, 其中 $a^{\tau_K} \sim \mathcal{N}(0, I)$
- State: $s^k = (a^{\tau_k}, \tau_k, o)$, for $k = K, \dots, 1$
- Action: $u^k = a^{\tau_{k-1}} \sim \mathcal{N}(\mu_\theta(a^{\tau_k}, \tau_k, o), \sigma_{\tau_k}^2 I)$
- Transition: $s^{k-1} = (u^k, \tau_{k-1}, o)$
- Reward: 只有 terminal reward $R(a^{\tau_0})$ (来自上层 env)

**Unified PPO Objective across denoising steps**:

$$J_i(\pi) = \mathbb{E}_{s_t \sim \rho_\pi, a_t \sim \pi_i}\left[\sum_{k=1}^K \min\left(r_k(\pi) A_t, \text{clip}(r_k(\pi), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中:
- $r_k(\pi) = \frac{\pi(a^{\tau_{k-1}} | s^k)}{\pi_i(a^{\tau_{k-1}} | s^k)}$: per-denoising-step importance ratio
- $A_t$: **environment-level advantage**, shared across all K denoising steps
- $\pi_i$: behavior policy at PPO iteration $i$
- $\rho_\pi$: discounted state distribution under current policy $\pi$

**Intuition**: share 同一个 $A_t$ across K denoising steps 给每个 denoising step 都提供 dense learning signal, 同时保持与 environment reward structure 的一致性。这解决了 diffusion policy 在 RL 中 reward sparsity 的核心问题。

---

## 4. Offline RL 细节 - AM-Q OPE Gate

### 4.1 IQL-style Critics

Offline RL 的 challenge 是 distribution shift - 评估 out-of-distribution actions 时 Q 容易 overestimate。IQL (Kostrikov et al. 2022, https://arxiv.org/abs/2110.06169) 用 implicit Q-learning 避免 query OOD actions。

Advantage 计算:
$$A_t^{\text{off}} = Q_\psi(s_t, a_t) - V_\psi(s_t)$$

### 4.2 AM-Q OPE (来自 Uni-O4, LEI et al. 2024)

Offline Policy Evaluation 不需要 environment interaction:
$$\hat{J}^{\text{AM-Q}}(\pi) = \mathbb{E}_{(s,a) \sim (\hat{T}, \pi)}\left[\sum_{t=0}^{H-1} Q_\psi(s_t, a_t)\right]$$

其中 $\hat{T}$ 是 learned transition model。

**Accept rule** (conservative, monotonic improvement):
$$\hat{J}^{\text{AM-Q}}(\pi) - \hat{J}^{\text{AM-Q}}(\pi_i) \geq \delta$$
$$\delta = 0.05 \cdot |\mathcal{T}^{\text{AM-Q}}(\pi_i)|$$

adaptive threshold: $\delta$ 跟 trajectory length成比例。如果 candidate policy 在 OPE 下不显著好于 behavior policy, reject。

Reference: Uni-O4 论文 https://openreview.net/forum?id=tbFBh3LMKi

---

## 5. Consistency Distillation - 把 K-step 压到 1-step

Real-world deployment 要求 high frequency control。Multi-step diffusion (K=10) 推理 ~100ms, 太慢。Consistency model (Song et al. 2023, https://arxiv.org/abs/2303.01969) 把它压到 1-step, ~10ms, **10x speedup**。

### 5.1 Consistency Distillation Loss

$$\mathcal{L}_{\text{CD}}(\theta) = \mathbb{E}_{x_0, \tau, \varepsilon}\left[\|C_\theta(x^\tau, \tau) - \text{sg}[\Psi_\varphi(x^\tau, \tau \to 0)]\|_2^2\right]$$

变量:
- $C_\theta$: consistency model (student)
- $\Psi_\varphi$: frozen diffusion teacher (我们的 $\pi_\theta$)
- $\text{sg}[\cdot]$: stop-gradient - 关键! 让 teacher 继续被 RL 改进, 同时作为 distillation target
- $x^\tau$: noisy input at level $\tau$

Inference 只需一次 forward pass:
$$x^0 \approx C_\theta(x^{\tau_K}, \tau_K), \quad x^{\tau_K} \sim \mathcal{N}(0, I)$$

### 5.2 Joint Training

Total loss combines RL 和 distillation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{CD}} \cdot \mathcal{L}_{\text{CD}}$$

$\mathcal{L}_{\text{RL}}$ 可以是 offline (Eq. 19 with IQL) 或 online (Eq. 21 with GAE)。

### 5.3 推理速度对比

| Model | Frequency | Params |
|-------|-----------|--------|
| RL-100 (CM) Skip-Net | **378 Hz** | 3.9M |
| RL-100 (CM) U-Net | 133 Hz | 39.2M |
| DSRL | 35 Hz | 52.3M |
| DPPO | 30 Hz | - |
| ReinFlow | ~50 Hz | smaller |

CM 的 Skip-Net 达到 378 Hz - 比 DSRL 快 10x 同时参数少 13x! 这个 frequency 足够 reactive control, 系统瓶颈变成 camera frame rate (30 Hz) 而不是 policy inference。

---

## 6. Variance Clipping - 让 Stochastic Diffusion 在 RL 中 Stable

这是 paper 一个 small but crucial trick。Stochastic DDIM 在 RL 中要平衡 exploration vs stability:

$$\tilde{\sigma}_k = \text{clip}(\sigma_k, \sigma_{\min}, \sigma_{\max})$$

- $\sigma_k$: 原始 DDIM variance (Eq. 5b)
- $\sigma_{\min} = 0.01$: 维持 minimal exploration, 防止 premature convergence
- $\sigma_{\max} = 0.8$: 防止 destructive exploration

为什么重要? PPO 的 importance ratio:
$$r_k(\pi) = \frac{\pi(a^{\tau_{k-1}} | s^k)}{\pi_i(a^{\tau_{k-1}} | s^k)}$$

如果 current policy 和 behavior policy 的 variance 差异过大, $r_k$ 会爆炸。Variance clipping 让 $r_k$ 保持 well-behaved。

**Per-task 调参**:
- Adroit, MuJoCo, single-action real tasks: 0.8
- Meta-World, chunk-action real tasks: 0.1

这个 per-task 选择说明: chunk action 对 noise 更敏感, 需要更紧的 bound。

---

## 7. Representation Learning - 防 Drift

RL fine-tuning 中 visual encoder 容易 drift, 导致 representation 不稳定。论文用 reconstruction + VIB 双重正则化:

$$\mathcal{L}_{\text{recon}} = \beta_{\text{recon}}\left(d_{\text{Chamfer}}(\hat{o}, o) + \|\hat{q} - q\|_2^2\right)$$
$$\mathcal{L}_{\text{KL}} = \beta_{\text{KL}} \text{KL}\left(\phi(z|o,s) \| \mathcal{N}(0, I)\right)$$

变量:
- $o, q$: 观测 point cloud 和 proprioception
- $\hat{o}, \hat{q}$: 从 encoded embedding $\phi(o, q)$ 重建
- $d_{\text{Chamfer}}$: Chamfer distance (point cloud 间距离)
- $\beta_{\text{recon}}, \beta_{\text{KL}}$: 权重, RL 阶段除以 10 (允许 policy improvement 同时 maintain stability)

Total IL objective:
$$\mathcal{L}_{\text{total}}^{\text{IL}} = \mathcal{L}_{\text{IL}} + \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}$$

Ablation (Fig. 16D) 显示:
- Recon+VIB + 更新 encoder: 最好最稳定
- 去掉 Recon+VIB: stability 和 final performance 下降
- Frozen encoder: gain 受限

---

## 8. Prediction Parameterization - $\epsilon$ vs $x_0$

这个 ablation 我觉得很有教育意义。Diffusion policy 文献里 DP3 等 adopt $x_0$-prediction, 但 RL-100 发现 **$\epsilon$-prediction 在 RL post-training 中显著优于 $x_0$-prediction**。

### 8.1 两种 Parameterization

$\epsilon$-prediction:
$$\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} f_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

$x_0$-prediction:
$$\hat{x}_0 = f_\theta(x_t, t)$$

### 8.2 为什么 $\epsilon$-prediction 在 RL 中更好?

关键 insight: variance amplification。Early in reverse process (large $t$), $\frac{1}{\sqrt{\bar{\alpha}_t}}$ 变大, magnifies estimation noise in $\varepsilon_\theta$ → higher variance $\hat{x}_0$。

Empirical variance measurements (100 forward passes per timestep):

| Task | $\epsilon$-pred | $x_0$-pred |
|------|-----------------|-------------|
| Hopper (locomotion) | 0.1316 | 0.0870 |
| Adroit-Door (manipulation) | 0.0589 | 0.0290 |

$\epsilon$-prediction 的 variance 大约 1.5-2x of $x_0$-prediction。

### 8.3 Implication

在 two-level MDP 中, 每个 action 是 K-step denoising trajectory 产生的。$\epsilon$-prediction 的额外 stochasticity 在 denoising chain 内部相当于 **structured exploration**, 改善 latent action modes 的 coverage, 帮助 policy-gradient fine-tuning 避免 local optima。

Ablation (Fig. 16E): online RL with $\epsilon$-prediction 在 Adroit-Door 上 faster, more reliable improvement; $x_0$-prediction 更慢且 premature convergence。

**Takeaway**: $x_0$-prediction 适合 low-variance imitation 和 open-loop execution; $\epsilon$-prediction 的 variance amplification 在 diffusion-based online RL 中是有益的 exploration 机制。

---

## 9. Action Chunk 处理

Chunk action 是 ACT (Zhao et al. 2023, https://tonyzhaozh.github.io/aloha/) 引入的 idea, 在 precision tasks 中减少 jitter 和 error compounding。

### 9.1 Single vs Chunk

**Single action** (single-step control):
- Standard MDP, per-step rewards $R_t$, discount $\gamma$
- 适合 reactive tasks (dynamic bowling, push-T)

**Action chunk** (chunked control):
- 每个 chunk of $n_c$ actions 作为 single decision
- Cumulative reward: $R_{\text{chunk}} = \sum_{j=0}^{n_c-1} \gamma^j R_{t+j}$
- Equivalent discount between chunks: $\gamma^{n_c}$
- 适合 coordination-heavy 或 high precision tasks (assembly, folding)

### 9.2 为什么这个区分重要?

论文在 Table 1 里给每个 task 选了 control mode:
- Dynamic Push-T: single-step (fast closed-loop reaction)
- Agile Bowling: single-step (release-timing at high velocity)
- Pouring: single-step (flow control under motion)
- Dynamic Unscrewing: action-chunk (time-varying alignment, torque regulation)
- Soft-towel Folding: action-chunk (dual-arm coordination, large deformation)
- Orange Juicing: action-chunk (confined-space insertion)

两种 mode 共享同一个 diffusion backbone, 只 output head 维度不同 ($\mathbb{R}^{d_a}$ vs $\mathbb{R}^{n_c d_a}$)。这让 framework 既 task-adaptive 又 architecture-consistent。

---

## 10. Experimental Results - 7 个 Real-World Task

### 10.1 Task Suite

| Task | Control | Embodiment | Modality | Key Challenge |
|------|---------|------------|----------|---------------|
| Dynamic Push-T | single | UR5 | rigid | 3mm clearance, friction varies |
| Agile Bowling | single | UR5 | rigid | high-velocity release timing |
| Pouring | single | Franka+LeapHand | granular/fluid | spillage, flow control |
| Dynamic Unscrewing | chunk | Franka+LeapHand | precision | time-varying torque, cross-thread |
| Soft-towel Folding | chunk | xArm+Franka+Robotiq (dual) | deformable | large deformation, coordination |
| Orange Juicing - Placing | chunk | xArm+Robotiq | deformable | fruit variability, confined |
| Orange Juicing - Removal | chunk | xArm+Robotiq | deformable | force-sensitive, slippery |

### 10.2 Main Success Rates (Table 3)

| Task | DP-2D | DP3 | Iterative Offline | Online DDIM | Online CM |
|------|-------|-----|------------------|-------------|-----------|
| Dynamic Push-T | 40 | 64 | 90 | 100 | 100 |
| Agile Bowling | 14 | 80 | 88 | 100 | 100 |
| Pouring | 42 | 48 | 92 | 100 | 100 |
| Soft-towel Folding | 46 | 68 | 94 | 100 | 100 (250/250) |
| Dynamic Unscrewing | 82 | 70 | 94 | 100 | 100 |
| Orange Juicing Placing | 78 | 88 | 94 | 100 | 100 |
| Orange Juicing Removal | 48 | 76 | 86 | 100 | - |
| **Mean** | **50.0** | **70.6** | **91.1** | **100** | **100** |

Total: **900/900 success rate**, 包括 Soft-towel Folding 连续 **250/250**。

注意几个 pattern:
1. IL baseline (DP-2D, DP3) 在 dynamic/deformable task 上特别弱 (Agile Bowling 14%, Pouring 42%)
2. Iterative Offline RL 给主要 jump (50% → 91%)
3. Online RL 把 91% → 100% (last mile)
4. CM 几乎完全 match DDIM performance, 但 inference 10x faster

Juicing-Removal 没用 CM - 因为 IK-induced pose discontinuities + slippery contact, one-step CM noise-sensitive。这个 detail 说明 consistency distillation 在某些 contact-rich 场景仍有局限。

### 10.3 Robustness - Zero-shot Adaptation

| Variation | Success |
|-----------|---------|
| Pouring (Water, granular→fluid) | 90% |
| Push-T (Changed surface friction) | 100% |
| Push-T (Interference objects) | 80% |
| Bowling (Changed surface) | 100% |
| Folding (unseen shape) | 80% |
| **Average** | **90.0%** |

Pouring from granular nuts 到 water 是巨大 dynamics shift - 流体动力学完全不同。90% success 说明 policy 学到的不是 specific physical parameter, 而是 **stable manipulation strategy**。

### 10.4 Few-shot Adaptation (1-3 hours fine-tuning)

| Variation | Success |
|-----------|---------|
| Pour (New container shape) | 60% |
| Folding (Changed towel material) | 100% |
| Bowling (Inverted pin arrangement) | 100% |
| **Average** | **86.7%** |

1-3 小时 fine-tuning 在 real robot 上是相当 sample efficient 的。

### 10.5 Disturbance Robustness

| Task & Stage | Success |
|-------------|---------|
| Folding (Stage 1: Grasping) | 90% |
| Folding (Stage 2: Pre-folding) | 90% |
| Unscrewing (counter-rotation 4s) | 100% |
| Push-T (dragging throughout) | 100% |
| **Average** | **95.0%** |

Unscrewing 在 counter-rotation 4 秒干扰下还能 100% - 这是 closed-loop control 的力量。Push-T 在整个 pushing 过程中持续 dragging 也能 100%。

### 10.6 Efficiency

**Episode length** (successful):
- Soft-towel Folding: 390 → 312 steps (1.25x fewer)
- Dynamic Unscrewing: 361 → 280 steps (1.29x fewer)

**Wall-clock time** (Orange Juicing-Placing):
- DP-2D: 10.6s
- RL-100 (DDIM): 10.2s
- RL-100 (CM): 9.2s (1.11x faster than DDIM, 1.15x faster than DP-2D)

**vs Human**:
- Dynamic Push-T throughput: RL-100 = 20 eps/unit time, Human expert = 17 (1.18x), Human beginner = 13 (1.54x)
- 不只是 match 人类, 而是 **surpass** 人类 expert

### 10.7 Data Collection Cost

| Stage | Episodes/task | Time/task |
|-------|---------------|-----------|
| Human demo | 115 | 1.8h |
| Iterative Offline RL | 566 | 6.5h |
| Online RL | 434 | 5.6h |

Total < 100h, human teleoperation 只占 13%。这个比例非常 attractive - 减少 expert labor 同时大部分 data 来自 autonomous rollout。

### 10.8 Shopping Mall Demo

Juicing robot 在公开 shopping mall 给 random customers 持续服务 **7 小时无 failure**。这是 zero-shot deployment 到全新环境 - 我觉得这是 paper 最强的 existence proof for deployment readiness。

---

## 11. Simulation Benchmark

对比 SOTA diffusion/flow offline-to-online RL methods:
- **DPPO** (Ren et al. 2024, https://arxiv.org/abs/2409.00588)
- **ReinFlow** (Zhang et al. 2025b, https://arxiv.org/abs/2505.22094)
- **DSRL** (Wagenmaker et al. 2025, https://arxiv.org/abs/2506.15799)

### 11.1 Performance Highlights

- halfcheetah-medium-v2: RL-100 = 10,000 vs DPPO = 4,500 (2.2x) vs DSRL = 3,000 (3.3x)
- Adroit Door, Hammer: RL-100 ~100%, DPPO plateaus at 0.9, ReinFlow <0.6 on hammer
- Meta-World peg-insert-side: RL-100 stable 1.0, ReinFlow <0.2

Sample efficiency: RL-100 在 1-2M steps 达 peak, baselines 需要 3-5M steps。

### 11.2 Ablations (Fig. 16)

**Visual modality** (2D vs 3D): 3D 在 clean, contact-rich scene (Adroit-Door) 更快更准。原因: 3D input 允许 precise geometric filtering (ROI crop), signal-to-noise ratio 高。

**Diffusion noise clipping**: 0.8 是 sweet spot for Adroit/MuJoCo/single-action; 0.1 for Meta-World/chunk-action。

**CM vs DDIM**: 几乎 identical learning curves, CM 不牺牲性能但 10x faster。

**Representation during fine-tuning**: Recon+VIB + 更新 encoder 最好; 去掉降 stability; frozen encoder 限制 gain。

**Diffusion parameterization**: $\epsilon$-prediction 比 $x_0$-prediction 高 ~40% final success (1.0 vs 0.6)。

---

## 12. 与 SERL/HIL-SERL 的对比

这是 paper 的 positioning argument, 我觉得重要:

**SERL** (Luo et al. 2024, https://arxiv.org/abs/2406.11532) 和 **HIL-SERL** (Luo et al. 2025, https://www.science.org/doi/10.1126/scirobotics.ads5033) 是 Stanford/UC Berkeley 的 real-world RL 工作, 报道 impressive on-robot learning。但有限制:

1. **Action-space shaping**: 限制 wrist rotation, encourage near-planar end-effector motion
2. **Short-horizon, low-dim control**: 适合 sample efficiency 但 cap performance on orientation-critical, contact-rich tasks
3. **依赖 demonstrations, 难以处理 precision 和 failure recovery**

RL-100 的差异:
- **Full 6-DoF control without hard rotation constraints**
- Diffusion/consistency visuomotor policy 捕捉 diverse human strategies
- OPE-gated PPO objective 实现近 monotonic improvement
- One-step consistency distillation 实现 high-frequency control
- 跨 dual-arm, deformable, dynamic tasks with larger cross-object generalization

具体来说, 日常 home/factory 任务需要 full SE(3) control 和 substantial reorientation, 比如:
- Towel folding (twist + regrasp)
- Orange juicing (insertion/ejection in confined cavities with large tilt)
- Controlled pouring (container tilt 是核心)
- Agile bowling (dynamic release + trajectory shaping)

---

## 13. 我 (Karpathy) 视角的 Intuition

### 13.1 Imitation Ceiling 的本质

IL 优化的是 "match the demonstrator" 这个 supervised objective。但 deployment 关心的是 success rate, time-to-completion, robustness - 这些 **不是 IL 的 direct objective**。这就是 imitation ceiling 的根源 - 你优化什么和你关心什么不 aligned。

RL-100 的 elegant 之处: 用同一个 PPO objective 跨 offline/online 阶段, 直接优化 deployment metrics。Reward 函数用 sparse +1 (success) + step penalty (efficiency) + jitter penalty (smoothness) - 直接对应 deployment criteria。

### 13.2 为什么 Offline-to-Online 的两个阶段都需要?

Pure offline RL 的局限: 数据是 fixed 的, 无法发现新 modes, value function 在 OOD actions 上 unreliable。
Pure online RL 的局限: sample inefficient, 在 real robot 上 dangerous, 需要大量 reset。

Iterative offline + targeted online 的优势:
- Iterative offline 通过 **data expansion** 持续扩大 coverage (better policy → better data → even better policy)
- OPE gate 保证 conservative, monotonic improvement (避免 dangerous degradation)
- Online stage 处理 offline 无法 reach 的 rare failure modes

这个让我想到 AlphaGo 的 pipeline: SL policy (imitate human) → RL policy (self-play) → value network。RL-100 是这个思路在 real robot 上的 instantiation。

### 13.3 Two-Level MDP 的深意

我觉得这是 paper 最 clever 的设计。传统 diffusion RL (e.g., DQL Wang et al. 2023, https://arxiv.org/abs/2209.09713) 把 diffusion 当 black-box sampler, 用 weighted regression 或 sampling-based 优化。RL-100 把 denoising chain 当 **sub-MDP**, 每个 denoising step 是一个 Gaussian sub-policy。

好处:
1. **Dense learning signal**: 每个 denoising step 都接收 reward gradient (通过 shared $A_t$), 避免 reward sparsity
2. **Importance sampling 可计算**: Gaussian sub-policy 的 log-likelihood closed-form (Eq. 7b), PPO 直接可用
3. **Exploration structure**: stochastic diffusion 提供 structured exploration within denoising chain, 比 Gaussian action noise 更 expressive

这跟 DPPO (Ren et al. 2024) 思路类似但更精炼 - DPPO 处理 diffusion policy 的 online RL, RL-100 把它 unify 到 offline+online setting 并加入 consistency distillation。

### 13.4 Consistency Distillation 的 Timing

通常 distillation 是 post-training 后做的 (e.g., One-Step Diffusion Policy Wang et al. 2025, https://arxiv.org/abs/2410.21257)。RL-100 **jointly** train RL objective 和 consistency distillation:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{CD}} \cdot \mathcal{L}_{\text{CD}}$$

Stop-gradient 让 teacher 继续被 RL 改进, 同时 distill 到 student。这个 joint training 比 post-hoc distillation 更 efficient - 不需要 separate distillation phase, 同时 teacher 和 student 始终 sync。

### 13.5 为什么 $\epsilon$-prediction 在 RL 中赢?

这个 ablation 我觉得揭示了 diffusion RL 的一个 deep insight。$x_0$-prediction 是 "mean teacher" - 直接预测 clean sample, variance 低, 适合 open-loop execution。$\epsilon$-prediction 是 "noise teacher" - 预测噪声, 然后通过 $\frac{1}{\sqrt{\bar{\alpha}_t}}$ 放大, variance 高。

在 IL setting, variance 是 bug (destabilize training)。但在 RL setting, variance 是 feature - 它提供 structured exploration。这跟 SAC 中 entropy maximization 思路类似 - exploration 不只是 action space noise, 而是 **representation-level stochasticity**。

这个 insight 可能可以推广: 在其他 generative model + RL 的组合中 (e.g., flow matching, VAE), parameterization choice 应该考虑 exploration needs, 不只是 reconstruction fidelity。

### 13.6 Variance Clipping 是 RL 在 Real Robot 上的 "Safety Belt"

Real robot RL 的最大风险是 destructive exploration - policy 产生 OOD action 导致 hardware damage 或 unsafe behavior。Variance clipping $\tilde{\sigma}_k = \text{clip}(\sigma_k, 0.01, 0.8)$ 是 simple but effective safety mechanism:
- Lower bound 0.01: 防止 policy collapse (premature convergence)
- Upper bound 0.8: 防止 destructive exploration

加上 conservative operating limits (论文提到 "controller follows conservative operating limits") 和 human supervision for sparse success signals, 这个 safety stack 让 real-world RL 变得 tractable。

---

## 14. Limitations & Future Work

论文自己提到:
1. **Reset and recovery** 仍是 practical bottleneck - 需要 autonomous reset mechanisms
2. **Scaling to larger VLA models** - 论文用的是 small diffusion policy, 但 RL-100 的 unified objective 理论上可以 apply 到 π0 (https://arxiv.org/abs/2410.24164) 或 π0.5 (https://arxiv.org/abs/2504.16054) 这种 large VLA
3. **Cross-embodiment 和 cross-task transfer** within single policy
4. **更复杂场景**: cluttered, partially observable, dynamic multi-object, occlusions, transparent materials

我补充几个观察:
- **CM 在某些 contact-rich task 上局限** (Juicing-Removal) - 说明 one-step inference 在 discontinuous dynamics 下 information loss
- **Reward engineering** 仍存在 - Dynamic Push-T 用 dense shaped reward, 其他用 sparse + penalty
- **Reset mechanism** 论文没详细讨论 - shopping mall 7 小时 demo 应该有某种 autonomous reset

---

## 15. Related Work Context

### 15.1 Diffusion + RL

- **Diffusion Q-Learning (DQL)** (Wang et al. 2023, https://arxiv.org/abs/2209.09713): 用 diffusion model 替代 Gaussian policy 在 offline RL
- **Weighted regression methods** (Kang et al. 2023; Lu et al. 2023; Ding et al. 2024): importance-weighted objectives 最大化 Q-function
- **Reparameterization gradient** (Psenka et al. 2024; He et al. 2023): gradient-based, 时间 backprop 挑战
- **DPPO** (Ren et al. 2024): policy gradient for diffusion policy
- **Consistency-based extensions** (Li et al. 2024): generalize diffusion + consistency policy 到 visual RL
- **FQL** (Park et al. 2025, https://arxiv.org/abs/2502.09389): flow-matching policy 避免 recursive backprop
- **DSRL** (Wagenmaker et al. 2025, https://arxiv.org/abs/2506.15799): RL entirely in latent noise space

### 15.2 Generative Diffusion in Robotics

- **Diffusion Policy** (Chi et al. 2023, https://diffusion-policy.cs.columbia.edu/): conditional diffusion over actions
- **DP3** (Ze et al. 2024, https://arxiv.org/abs/2403.03954): 3D point cloud extension
- **FlowPolicy** (Zhang et al. 2025a, https://arxiv.org/abs/2501.04190): consistency flow matching for 3D
- **H3DP** (Lu et al. 2025, https://arxiv.org/abs/2505.07819): triply-hierarchical diffusion policy
- **One-Step Diffusion Policy** (Wang et al. 2025, https://arxiv.org/abs/2410.21257): post-hoc distillation
- **π0** (Black et al. 2024a, https://arxiv.org/abs/2410.24164): VLA flow model
- **π0.5** (Intelligence et al. 2025, https://arxiv.org/abs/2504.16054): VLA with open-world generalization

### 15.3 Offline-to-Online RL

- **CQL** (Kumar et al. 2020, https://arxiv.org/abs/2006.04779): pessimistic value estimation
- **AWR** (Peng et al. 2019, https://arxiv.org/abs/1910.00108): advantage-weighted regression
- **Cal-QL** (Nakamoto et al. 2023, https://arxiv.org/abs/2307.04795): calibrated Q-values for online fine-tuning
- **Uni-O4** (LEI et al. 2024, https://openreview.net/forum?id=tbFBh3LMKi): PPO unify offline+online
- **RLPD** (Ball et al. 2023, https://arxiv.org/abs/2212.05312): mix offline+online data

### 15.4 Real-World RL

- **End-to-end visuomotor** (Levine et al. 2016, https://arxiv.org/abs/1504.00702)
- **QT-Opt grasping** (Kalashnikov et al. 2018, https://arxiv.org/abs/1806.10293)
- **SAC** (Haarnoja et al. 2018, https://arxiv.org/abs/1801.01290)
- **Reset-free RL** (Eysenbach et al. 2018; Gupta et al. 2021)
- **SERL** (Luo et al. 2024, https://arxiv.org/abs/2406.11532)
- **HIL-SERL** (Luo et al. 2025, https://www.science.org/doi/10.1126/scirobotics.ads5033)

---

## 16. 我的几个 Open Questions / Speculations

### 16.1 为什么 IQL 而不是 CQL?

IQL 的 advantage 是不 query OOD actions (用 expectile regression 估 V, 然后 Q 通过 V regression)。CQL 用 conservative penalty。在 diffusion policy + 离线数据上, IQL 可能更 stable 因为不依赖 Q function 在 OOD 上的 behavior。但这个选择 paper 没有详细 ablate。

### 16.2 VLA Scale-up 的挑战

论文提到可以 scale 到 π0 / π0.5 这种 large VLA。但 challenge:
1. Large VLA 的 denoising chain 可能不是 Gaussian sub-policy (e.g., flow matching)
2. Online RL 在 large model 上 sample efficiency 可能更难
3. Representation drift 可能更严重 (VLA 有 language + vision 多模态)

但 unified PPO objective 和 consistency distillation 的 idea 应该 transfer。

### 16.3 CM 在 Contact-Rich Task 的局限

Juicing-Removal 不能用 CM, 因为 IK-induced pose discontinuities + slippery contact 让 one-step CM noise-sensitive。这说明 consistency distillation 在 **discontinuous dynamics** 下 information loss。可能的解决:
- Adaptive K (在 contact event 时 multi-step, free space 时 one-step)
- Hybrid CM + DDIM (critical phase 用 DDIM)
- Better contact-aware representation

### 16.4 Data Engine 视角

RL-100 的 iterative pipeline 本质是 **data engine** - 这个跟你 Tesla 时讲的思路很像:
1. Initial data: human demos (high quality, expensive)
2. Policy rollout: autonomous, cheap, scalable
3. IL re-training on expanded data: 蒸馏 human + RL improvements
4. 下一轮 iteration

这个 data engine 让 small human effort (13%) leverage large autonomous rollout (87%)。对未来 robot foundation model 训练, 这种 data engine pattern 可能比 pure scaling 更重要 - 因为 high-quality human demos 是 bottleneck, 而 autonomous rollout 在 good policy 初始化下是 cheap。

### 16.5 为什么 Shopping Mall Demo 这么重要?

7 小时无 failure 的 shopping mall demo 是 deployment readiness 的 ultimate test:
1. **Zero-shot deployment**: 新环境, 新 lighting, 新人群
2. **Long-horizon**: 7 小时 = 数千次 cycle
3. **Real users**: random customers, 不是 controlled lab
4. **No safety driver**: 完全 autonomous

这种 deployment-grade evidence 在 academic paper 里非常罕见。大多数 robot RL paper 报道 lab evaluation, 几十到几百 trials。RL-100 的 mall demo 是 **existence proof** that real-world RL 已经 ready for production。

---

## 17. 总结

RL-100 是 2025 年我看到的最 comprehensive 的 real-world RL 工作。它把多个 idea cohesively 组合:
- Diffusion policy 作为 expressive policy class
- Two-level MDP 把 denoising chain 嵌入 RL framework
- Unified PPO objective across offline/online + denoising steps
- OPE-gated conservative improvement
- Joint consistency distillation for high-frequency inference
- Variance clipping for safe exploration
- Self-supervised representation for stability
- Iterative data expansion (data engine pattern)

实验上, 900/900 success rate, 7 小时 mall demo, 多 task/embodiment/representation generalization, 这些是 deployment-grade evidence。

我认为这篇 paper 标志着 real-world RL 从 "lab demo" 到 "deployment-ready" 的 transition。下一个 frontier 是 scale to VLA, 把这个 unified objective apply 到 π0/π0.5 这种 large foundation model, 保留 zero-shot semantic generalization 同时获得 deployment-grade reliability。

---

## Key References

1. **RL-100 Project Page**: https://lei-kun.github.io/RL-100/
2. **Diffusion Policy** (Chi et al. 2023): https://diffusion-policy.cs.columbia.edu/
3. **DP3** (Ze et al. 2024): https://arxiv.org/abs/2403.03954
4. **DDPM** (Ho et al. 2020): https://arxiv.org/abs/2006.11239
5. **DDIM** (Song et al. 2021): https://arxiv.org/abs/2010.02502
6. **Consistency Models** (Song et al. 2023): https://arxiv.org/abs/2303.01969
7. **IQL** (Kostrikov et al. 2022): https://arxiv.org/abs/2110.06169
8. **PPO** (Schulman et al. 2017): https://arxiv.org/abs/1707.06347
9. **GAE** (Schulman et al. 2016): https://arxiv.org/abs/1506.02438
10. **Uni-O4** (LEI et al. 2024): https://openreview.net/forum?id=tbFBh3LMKi
11. **DPPO** (Ren et al. 2024): https://arxiv.org/abs/2409.00588
12. **DSRL** (Wagenmaker et al. 2025): https://arxiv.org/abs/2506.15799
13. **ReinFlow** (Zhang et al. 2025b): https://arxiv.org/abs/2505.22094
14. **DQL** (Wang et al. 2023): https://arxiv.org/abs/2209.09713
15. **FQL** (Park et al. 2025): https://arxiv.org/abs/2502.09389
16. **One-Step Diffusion Policy** (Wang et al. 2025): https://arxiv.org/abs/2410.21257
17. **SERL** (Luo et al. 2024): https://arxiv.org/abs/2406.11532
18. **HIL-SERL** (Luo et al. 2025): https://www.science.org/doi/10.1126/scirobotics.ads5033
19. **CQL** (Kumar et al. 2020): https://arxiv.org/abs/2006.04779
20. **AWR** (Peng et al. 2019): https://arxiv.org/abs/1910.00108
21. **Cal-QL** (Nakamoto et al. 2023): https://arxiv.org/abs/2307.04795
22. **RLPD** (Ball et al. 2023): https://arxiv.org/abs/2212.05312
23. **π0** (Black et al. 2024): https://arxiv.org/abs/2410.24164
24. **π0.5** (Intelligence et al. 2025): https://arxiv.org/abs/2504.16054
25. **ACT / ALOHA** (Zhao et al. 2023): https://tonyzhaozh.github.io/aloha/
26. **DP3 / 3D Diffusion Policy** (Ze et al. 2024): https://arxiv.org/abs/2403.03954
27. **FlowPolicy** (Zhang et al. 2025a): https://arxiv.org/abs/2501.04190
28. **H3DP** (Lu et al. 2025): https://arxiv.org/abs/2505.07819
29. **Demospeedup** (Guo et al. 2025): https://arxiv.org/abs/2506.05064
30. **QT-Opt** (Kalashnikov et al. 2018): https://arxiv.org/abs/1806.10293
31. **SAC** (Haarnoja et al. 2018): https://arxiv.org/abs/1801.01290
32. **End-to-end visuomotor** (Levine et al. 2016): https://arxiv.org/abs/1504.00702

Andrej, 你怎么看这个 paper? 我特别好奇你对几个点的看法:
1. Two-level MDP 的设计 - 你觉得这是 RL + generative model 的 right abstraction 吗?
2. Iterative data engine pattern - 你在 Tesla 时有类似 insight 吗?
3. VLA scale-up 的可行性 - π0/π0.5 + RL-100 unified objective 你觉得会 work 吗?
