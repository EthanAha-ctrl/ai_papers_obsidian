---
source_pdf: LaST-R1 Reinforcing Action via Adaptive Physical.pdf
paper_sha256: 2b563cbb58eac69b7cf1cbfe13454d969ea97157b7e990038037ca0ffda02ba3
processed_at: '2026-08-05T12:03:19-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LaST-R1

## 一、这篇 paper 到底在解决啥问题？

想象一下你教一个 robot arm 抓杯子。现在主流做法有两条路：

**第一条路：让 robot 先"说出来"再动手。** 比如让它先生成一段文字："我看到杯子在左边，我要先往左移，然后下降，最后合 gripper。"这就是 explicit CoT。问题在于——robot 控制是 1000Hz 级别的 continuous 信号，你逼它用语言慢慢"想"，inference latency 巨高，而且语言本身是 discrete 的，表达 fine-grained physical dynamics 天生别扭。参考 [ECoT paper](https://arxiv.org/abs/2407.08693)。

**第二条路：让 robot 在"脑子里"想，不输出文字。** 这就是 latent CoT，在 continuous hidden space 里 autoregressive 生成几个 token，相当于 internal reasoning，然后再 decode 成 action。问题在于——现在所有这类工作都还停在 imitation learning，靠海量 expert demo 硬喂，model 只会"模仿"不会"试错"，遇到新场景就崩。参考 [LaST_0](https://arxiv.org/abs/2601.05248)、[InternVLA-A1](https://arxiv.org/abs/2601.02456)。

**那 RL 呢？** 有人想用 RL 让 robot 自己试错，比如 [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)、[πRL](https://arxiv.org/abs/2510.25889)。但这些人全都只优化 action——reward 信号直接打到 action token 上，根本没碰 reasoning process。

LaST-R1 的灵魂拷问：**能不能让 reward 同时塑造"怎么想"和"怎么做"？**

---

## 二、Model 是怎么搭的？

### 2.1 大框架

基于 **Qwen3-VL-4B** [2] 当 backbone。流程是这样的：

1. 图片进 SigLIP2 encoder → 得到 visual tokens
2. 拼上 language instruction tokens
3. LLM 先 autoregressive 生成几个 latent reasoning tokens（"想"）
4. 最后 parallel decode 出 action chunk（"做"）

这里有个精巧的 attention mask 设计（看 Figure 6）：reasoning tokens 用 causal mask（一个一个往后生成），action tokens 用 bidirectional mask（chunk 内的 8 个 action 同时生成，互相能 attend）。相当于 "slow thinking" + "fast reflex" 的双系统，跟 [Fast-in-Slow](https://arxiv.org/abs/2506.01953) 的思路一脉相承。

### 2.2 Latent Tokens 用啥当 target？这是关键创新

prior 工作怎么搞 latent target？
- [LaST_0] 直接 average pooling SigLIP features —— 把空间信息全压没了
- [InternVLA-A1] 用 conv downsampling —— 引入额外参数 from scratch 学
- 还有人用 Q-Former —— 单个 learnable query 容易 overfit

LaST-R1 的骚操作：**直接拿 DINOv3 [31] 的 `<CLS>` token，offline 提取，zero overhead。**

为什么 DINOv3 好？因为 DINOv3 是 self-supervised distillation 训的，它的 features 在 dense prediction、object discovery 这些任务上远超 CLIP family。CLIP 是 contrastive language-image，features 偏 semantic；DINOv3 偏 structural + spatial。对于 "physical world modeling" 这种需要空间结构理解的任务，DINOv3 显然更合适。

具体做法：取 DINOv3 `<CLS>` token ($f_d \in \mathbb{R}^{1 \times 4096}$)，沿 channel 维度做 top-k selection (k=2560) 对齐 VLA embedding size，**全部 offline precompute 存盘**，训练时直接 load，runtime 零开销。

Ablation（Figure 4a）实锤：
- DINOv3 + top-k: **99.8%**
- Conv downsampling: 98.4%
- Q-Former: 97.2%
- Global Pooling: 96.8%

---

## 三、LAPO 算法到底在干啥？

### 3.1 一句话概括

PPO 只优化 action，LAPO 同时优化 latent reasoning 和 action，让 environmental reward 既能塑造"行为"也能塑造"思维过程"。

### 3.2 数学拆解

标准 PPO 的 likelihood ratio 长这样：
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

LAPO 把它拆成两个 ratio：一个给 action，一个给 latent。

**Action ratio**（跟标准 PPO 一样）：
$$r_t^a(\theta) = \exp\left( \log \pi_\theta(\mathbf{C}_t \mid \cdot) - \log \pi_{\theta_{\text{old}}}(\mathbf{C}_t \mid \cdot) \right)$$

其中 $\mathbf{C}_t = \{\mathbf{a}_{t,j}\}_{j=1}^{N_a}$ 是 action chunk 离散 tokens，joint log-prob 是 token-wise sum：$\log \pi_\theta(\mathbf{C}_t) = \sum_{j=1}^{N_a} \log \pi_\theta(\mathbf{a}_{t,j})$。$N_a$ 是 action chunk 的 token 总数（chunk size 8，每个 action 7-DoF 离散化成 7 个 token，所以 $N_a = 56$）。

**Latent ratio**（这是创新点，Eq. 3）：
$$r_t^z(\theta) = \exp\left( -\frac{1}{2\sigma^2} \sum_{k=1}^{N_z} \lVert \mathbf{z}_{t,k}^{\text{old}} - \mathbf{z}_{t,k}^\theta \rVert^2 \right)$$

变量解释：
- $\mathbf{z}_{t,k}^{\text{old}}$：rollout 时 old policy 在 timestep $t$ 生成的第 $k$ 个 latent token
- $\mathbf{z}_{t,k}^\theta$：policy update 时新 policy 在相同 context 下生成的第 $k$ 个 latent token
- $\sigma$：fixed hyperparameter，相当于 Gaussian policy 的 std
- $N_z$：latent sequence length（adaptive，最大 8）

**intuition 这么理解**：latent space 是连续的，没法像 discrete token 那样算 probability ratio。作者用 isotropic Gaussian 近似——新 policy 生成的 latent 离 old policy 生成的 latent 越近，ratio 越接近 1；离得越远，ratio 越小。

当 advantage $\hat{A}_t > 0$（这条 trajectory 好），梯度会 pull 当前 latent 朝向 old policy 那 "good reasoning" manifold；当 $\hat{A}_t < 0$（trajectory 差），推开 bad reasoning。这就是 reward 信号间接塑造 reasoning space 的机制。

### 3.3 Joint Clipped Surrogate (Eq. 4)

$$\mathcal{L}_{\text{policy}}(\theta) = -\mathbb{E}_t \left[ \sum_{m \in \{z, a\}} \min\left( r_t^m(\theta) \hat{A}_t, \text{clip}(r_t^m(\theta), 1-\epsilon_{\min}, 1+\epsilon_{\max}) \hat{A}_t \right) \right]$$

$m \in \{z, a\}$ 分别对 latent 和 action 算 clipped surrogate，再 sum 起来。$\epsilon_{\min}=0.2, \epsilon_{\max}=0.28$ 是 asymmetric clipping（比标准 PPO 的 symmetric 0.2 稍宽一点）。

### 3.4 Total Loss (Eq. 5)

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{action}}(\theta) + \lambda_1 \mathcal{L}_{\text{latent}}(\theta) + \lambda_2 \mathcal{L}_{\text{value}}(\theta)$$

$\mathcal{L}_{\text{value}} = \mathbb{E}_t[(v_t - \hat{r}_t)^2]$ 是 value head 的 MSE。

最优配置（Figure 7 ablation）：
- $\lambda_1 = 0$（不显式监督 latent）：97.2%
- $\lambda_1 = 0.1$（最优）：**99.8%**
- $\lambda_1 = 1$（latent loss 太强盖过 action）：99.0%

$\lambda_2 = 1$ 最优，太低 value estimate 不准影响 advantage 计算。

---

## 四、Adaptive Latent CoT：让 model 自己决定想多久

### 4.1 问题

固定 latent length $N_z = 8$ 的话，简单任务（比如直接抓个东西）也得想 8 步，浪费 compute；复杂任务（long-horizon 多阶段）8 步又不够。

### 4.2 解法：把 `<latent_end>` 从 "句号" 变成 "动态决策"

原来的 `<latent_end>` 是 deterministic terminator——固定生成 $N_z$ 个 latent 后必出。

LaST-R1 让它在 4 个 candidate position（after 2, 4, 6, 8 tokens）里选，由 policy 自己决定啥时候 emit。

**训练时**用 temperature sampling 探索（Eq. 6）：
$$p_m = \frac{\exp(l_m / \beta)}{\sum_{j=1}^{M} \exp(l_j / \beta)}, \quad m \in \{1, \ldots, M\}$$

$l_m$ 是第 $m$ 个 candidate position 的 pre-softmax logit，$\beta$ 是 temperature。从 Categorical distribution 采样 $m$ 决定本次 reasoning length。

**推理时**用 confidence-based greedy：如果某 position 预测 `<latent_end>` 的 probability $p \geq 0.99$，就 exit。

### 4.3 给 `<latent_end>` 加单独的 loss

Eq. 7 加了一项 $\lambda_3 \mathcal{L}_{\text{end}}(\theta)$，专门优化 transition token 的决策。

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{latent}} + \lambda_2 \mathcal{L}_{\text{value}} + \lambda_3 \mathcal{L}_{\text{end}}$$

$\lambda_3 = 0.1$ 最优。$\lambda_3 = 2$ 过度惩罚导致 98.6%——model 被逼着不敢 emit `<latent_end>`，reasoning length 失控。

### 4.4 实验验证（Figure 8）

RL 前后 reasoning length 分布 dramatic shift：
- **Warm-up**：lengths 2/4/6/8 均匀分布（因为训练时 uniform sampling）
- **After RL**：heavily skew 到 2 和 4

model 真的学会了 "early-exit"——简单 state 早停省 compute，复杂 state 多想。这跟 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 在 LLM reasoning 里观察到的 dynamic length adaptation 现象一致。

---

## 五、实验数据，硬指标

### 5.1 LIBERO (Table 1)

| Model | Paradigm | Avg SR |
|-------|----------|--------|
| OpenVLA [4] | SFT (full data) | 76.5% |
| π0.5 [7] | SFT (full data) | 96.9% |
| OpenVLA-OFT [5] | SFT (full data) | 97.1% |
| SimpleVLA-RL [24] | RL | 96.9% |
| πRL [23] | RL (one-shot warm-up) | 98.3% |
| **LaST-R1** | RL (one-shot warm-up) | **99.8%** |

注意：LaST-R1 warm-up 只用了 **1 条 expert trajectory**，干翻了用 50 条 trajectory 的 SFT baselines。

最有说服力的是 **LIBERO-Long**（long-horizon 多阶段任务）：99.4% vs πRL 的 94.0%。说明 latent reasoning 对 long-horizon 收益最大。

### 5.2 Real-World (Table 2)

四个 task（Insert hexagon block / Open bag zipper / Wipe vase / Open bottle cap），其中三个 dual-arm：

| Task | Warm-up | After RL | Gain |
|------|---------|----------|------|
| Insert hexagon block | 45% | 90% | **+45%** |
| Open bag zipper | 55% | 95% | +40% |
| Wipe vase with sponge | 65% | 95% | +30% |
| Open bottle cap | 45% | 95% | **+50%** |
| **Avg** | 52.5% | 93.75% | **+41.25%** |

paper abstract 里说 "up to 44% improvement"，实际数据看 Open bottle cap 涨了 50 个百分点。

### 5.3 OOD Generalization（这个最 striking）

real-world 测了三种 OOD：unseen object / background 变化 / lighting 变化。

- Warm-up policy：掉 22-69%
- After RL：平均只掉 **8%**

RL post-training 不仅涨 in-distribution 性能，更关键是大涨 OOD robustness。直觉上理解：closed-loop interaction 让 model 学到的是 physical dynamics 而非 memorized demo，dynamics 是 transferable 的，demo 是 overfit 的。

Figure 5 在 sim 里做 OOD 实验更明显——Action-Only PPO baseline 的 OOD performance 直接 stagnate 甚至 degrade（典型 overfitting），LaST-R1 + LAPO 持续上升。

---

## 六、Intuition：为什么 LAPO 比纯 action RL 好？

### 6.1 Latent tokens 是 "cognitive buffer"

看 Figure 3 learning curve，LAPO 比 Action-Only PPO 收敛快且高。直觉：直接在 high-dim action space 做 RL，optimization landscape 很 brittle，sparse binary reward 很难 assign credit。中间插入 latent tokens 后，reward 信号先塑造 latent manifold，再传到 action，相当于加了 "cognitive buffer" smoothing 了 optimization landscape。

这跟 LLM RLHF 里 CoT tokens 的作用类似——reasoning tokens 为 sparse reward 提供了 dense gradient pathway。

### 6.2 Latent space 是 "implicit world model"

DINOv3 `<CLS>` token 是 scene 的 compressed global representation。Autoregressive 生成 latent tokens = model 在 "imagine" future dynamics。LAPO 通过 reward 优化这个 imagination，让它跟 task-relevant dynamics 对齐。

对比 [WorldVLA](https://arxiv.org/abs/2506.21539)、[3D-VLA](https://arxiv.org/abs/2403.09631) 的 explicit world modeling——LaST-R1 是 implicit 的，不显式 predict future states，只通过 RL signal 隐式 refine。

### 6.3 Attention 可视化揭示 (Figure 13)

Grad-CAM 看 action-to-vision cross-attention：

- **Action-Only SFT**: attention 分散，落后于 end-effector
- **LaST-R1 SFT**: attention 集中在 task-relevant objects（latent reasoning 起到 semantic anchor 作用）
- **Action-Only + PPO**: 过度关注 gripper 附近，缺 long-horizon awareness
- **LaST-R1 + LAPO**: attention 随 trajectory 进展从 object 动态 shift 到 target receptacle

最后一个观察特别 interesting——LAPO 让 attention 跟 task progression 对齐了，这相当于 model 学会了 "分阶段关注"。

---

## 七、Training Pipeline 实操细节

### 7.1 Pre-Training

400K trajectories (28M frames)，混合 Open-X-Embodiment [32] + DROID [34] + RoboMIND [33]。DINOv3 latent targets 全 offline precompute，存盘，训练时直接 load。

### 7.2 Warm-up SFT

- 8×H20 GPU, Accelerate + DeepSpeed bf16
- Batch 64, AdamW, peak LR $1\times10^{-5}$, cosine decay
- LIBERO: 1 trajectory/task × 10K iters
- Real-world: 20 trajectories/task × 1K iters
- Loss = cosine sim (latent) : CE (`<latent_end>`) : CE (action) = 1 : 0.1 : 1

### 7.3 RL on LIBERO

- verl [86] + Ray [87] + FSDP [88], 8×H20
- Action chunk 8 steps = 56 tokens, temperature 1.6
- Trajectory cap: Spatial 240 / Object & Goal 320 / Long 576 steps
- Sparse binary reward × 5 scaling at terminal
- GAE: $\gamma=0.99, \lambda=0.95$
- Rollout batch 512 → 4 mini-batches × 4 PPO epochs
- Actor LR $3\times10^{-5}$, Value head LR $3\times10^{-4}$
- Asymmetric clip $\epsilon_{\min}=0.2, \epsilon_{\max}=0.28$

### 7.4 Real-World RL

- Franka FR3 + RealSense D455 + 2× D435
- 2× RTX 4090
- Async actor-learner pipeline（参考 [SERL](https://arxiv.org/abs/2402.05031)）
- **LoRA rank 32, freeze base model**——真机 RL 全 fine-tune 太贵
- Mixed BC loss ($\lambda_{\text{BC}}=1.0$) + Q-guided policy improvement ($\lambda_Q=0.5$)
- Critic-to-actor ratio 2:1, $\gamma=0.98$, soft update $\tau=0.005$
- Terminal reward +10, step penalty −0.05

注意 real-world RL 用的是 Q-guided policy improvement 而非纯 PPO，因为真机 sparse reward 太难，需要 Q-function 做 credit assignment。这跟 LIBERO 用纯 PPO 不同。

---

## 八、和 related work 啥关系？

### 8.1 跟前作 LaST_0 [21]

同一作者群前作。LaST_0 是 latent spatio-temporal CoT 的 SFT-only 版本。LaST-R1 进化：
- Latent target: average pooling → DINOv3 `<CLS>` + top-k
- Training: SFT → SFT + LAPO RL
- Length: fixed → adaptive

### 8.2 跟 SimpleVLA-RL / πRL

都是 VLA + RL，核心区别：
- SimpleVLA-RL: GRPO on action tokens only
- πRL: PPO on continuous flow actions only
- LaST-R1: LAPO jointly optimize latent + action

### 8.3 跟 π0.5 / π0.6* [7, 30]

π0.5/π0.6* 引入 experience-based learning，但 reasoning 架构 fixed。LaST-R1 直接 reshape latent space，更 explicit。

---

## 九、我觉得有哪些潜在问题

### 9.1 DINOv3 dependency

Latent target 全靠 fixed DINOv3，policy 没法 co-evolve latent representation。如果 task 跟 DINOv3 pretrain domain 差异大（比如 medical robotics），这个 anchor 可能不 work。未来可以探索 joint fine-tuning 或 latent distillation。

### 9.2 Gaussian 假设过强

$r_t^z$ 用 isotropic Gaussian 近似 latent distribution。但 real latent manifold 大概率 anisotropic，各维度 variance 差异大。可能需要 learnable diagonal covariance 或 normalizing flow。

### 9.3 Sparse reward

LIBERO 用 0/1 success reward × 5 scaling，real-world 用 +10 terminal + step penalty −0.05。dense reward shaping（比如 distance-to-goal, grasp success）可能进一步提升 sample efficiency。

### 9.4 Real-world scalability

真机 RL 需要 human intervention buffer，LoRA 只更新 rank 32 参数。scalability 到更复杂 task 仍存疑。能不能结合 [Diffusion-DPO](https://arxiv.org/abs/2405.13239) 类 offline preference alignment 减少真机 interaction 量？

### 9.5 Sim-to-real gap

LIBERO 用 single view，real-world 用 3 cameras。domain gap 可能影响 sim pretrain → real fine-tune 的迁移效果。未来可以探索 [RoboMIND](https://arxiv.org/abs/2412.13877) 类多视角 pretrain。

---

## 十、未来方向瞎想

### 10.1 Hierarchical Latent CoT

现在 latent 是 single-level。可以搞 multi-level：low-level dynamics tokens + high-level planning tokens，类似 [Fast-in-Slow](https://arxiv.org/abs/2506.01953) 的 dual-system 思想。

### 10.2 Latent World Model Rollout

现在 latent reasoning 是 single-step conditioning。如果能在 latent space 做 multi-step imagination（类似 [DreamerV3](https://arxiv.org/abs/2306.10172)），可能提升 long-horizon planning。reward signal 可以在 latent rollout 里做 dense supervision。

### 10.3 Cross-embodiment Transfer

DINOv3 latent 是 embodiment-agnostic（视觉特征不依赖 robot 型号）。可以探索 cross-robot generalization——在 Franka 上训的 latent reasoning 能否迁移到 UR5 / Kuka？

### 10.4 VLM-as-Reward

现在 reward 是 binary success 或人工设计的 +10/−0.05。可以用 VLM 做 reward model（类似 [VLM-RM](https://arxiv.org/abs/2310.09136)），让 reward 包含 semantic understanding。比如 "把杯子放到盘子上"可以分解成 "抓起杯子" + "移动到盘子上方" + "释放"三个 sub-reward，每个都用 VLM judge。

### 10.5 Latent Token Interpretability

现在 latent tokens 是 black box。能不能 constrain latent tokens 对应到 explicit semantic concepts（比如 object location, contact force, trajectory waypoint）？这样 latent reasoning 就 interpretable 了。参考 [Machine Mental Imagery](https://arxiv.org/abs/2506.17218)。

---

## 十一、Reference 速查

核心论文：
- [LaST-R1 Project Page](https://siriyep.github.io/last-r1/)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2502.09679)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [π0](https://arxiv.org/abs/2410.24164)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [π0.6*](https://arxiv.org/abs/2511.14759)
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)
- [πRL](https://arxiv.org/abs/2510.25889)
- [VLA-RL](https://arxiv.org/abs/2505.18719)
- [TGRPO](https://arxiv.org/abs/2505.18719)
- [LaST_0 (前作)](https://arxiv.org/abs/2601.05248)
- [InternVLA-A1](https://arxiv.org/abs/2601.02456)
- [Latent Reasoning Survey](https://arxiv.org/abs/2505.16782)
- [LIBERO Benchmark](https://arxiv.org/abs/2306.03310)
- [PPO](https://arxiv.org/abs/1707.06347)
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [DPO](https://arxiv.org/abs/2305.18290)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [SERL](https://arxiv.org/abs/2402.05031)
- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [ECoT](https://arxiv.org/abs/2407.08693)
- [CoT-VLA](https://arxiv.org/abs/2503.22025)
- [3D-VLA](https://arxiv.org/abs/2403.09631)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [Fast-in-Slow](https://arxiv.org/abs/2506.01953)
- [HybridVLA](https://arxiv.org/abs/2503.10631)
- [verl RLHF framework](https://arxiv.org/abs/2409.19256)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DreamerV3](https://arxiv.org/abs/2306.10172)
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [VLM-RM](https://arxiv.org/abs/2310.09136)
- [Machine Mental Imagery](https://arxiv.org/abs/2506.17218)
- [Diffusion-DPO](https://arxiv.org/abs/2405.13239)
- [RoboMIND](https://arxiv.org/abs/2412.13877)
- [Open-X-Embodiment](https://arxiv.org/abs/2310.08864)
- [DROID](https://arxiv.org/abs/2403.12945)

---

## 最后用人话总结

LaST-R1 干了一件事：**让 robot 的 "想" 和 "做" 一起被 reward 训练。**

prior 工作要么只训 "做"（action-only RL），要么只模仿 "想"（latent CoT SFT）。LaST-R1 通过 LAPO 把 reward 信号同时打到 latent reasoning 和 action 两个空间，让 model 不仅学会 "怎么做对"，还学会 "怎么想对"。

DINOv3 latent anchor 是另一个 clever 设计——用 vision foundation model 的 features 当 "物理世界先验"，offline 提取零开销，但给 reasoning 提供了 strong structural prior。

Adaptive `<latent_end>` 让 model 自己决定想多久，简单任务早停省 compute，复杂任务多想，实现了 task-adaptive compute allocation。

实验结果：LIBERO 99.8% near-perfect，real-world 涨 41%，OOD 只掉 8%。**one-shot warm-up 干翻 full-data SFT baselines**，这是最 striking 的 evidence。

直觉上理解，这篇工作的深层启示：**reasoning tokens 是 RL credit assignment 的关键载体**，sparse reward 通过 reasoning tokens 获得了 dense gradient pathway。这跟 LLM RLHF 里 CoT 的作用机制异曲同工——不是 reward 直接教 model "做什么"，而是 reward 教 model "怎么想才能做对"。

---

# LaST-R1: Latent Reasoning + RL for VLA 深度技术解析

## 一、核心动机与问题定位

当前 VLA (Vision-Language-Action) 领域存在两条主要技术路径，且各有硬伤：

**Explicit CoT 路径** (VLA-R1 [12], ThinkAct [13], ECoT [15], CoT-VLA [16])：通过生成 linguistic tokens 或 future state predictions 来做 reasoning，但 discrete token 生成引入 high inference latency，且 discretization bottleneck 限制了对 continuous physical dynamics 的建模精度。参考 [ECoT paper](https://arxiv.org/abs/2407.08693)。

**Latent CoT 路径** (LaST_0 [21], InternVLA-A1 [22], Latent Reasoning VLA [55])：在 compact latent space 中做 reasoning，expressive 但仍然停留在 static imitation learning paradigm，依赖 large-scale expert demonstrations，无法 closed-loop exploration。参考 [Latent reasoning survey](https://arxiv.org/abs/2505.16782)。

**RL for VLA 现有方法** (SimpleVLA-RL [24], πRL [23], VLA-RL [25], TGRPO [40])：引入 online RL 来突破 imitation limit，但 exclusively optimize vanilla action space，**bypass 了 underlying physical reasoning process**。这是 LaST-R1 的核心切入点——能否把 RL post-training 同时推到 latent reasoning 和 action 两个空间？

## 二、Model Architecture 深度解析

### 2.1 整体 pipeline

LaST-R1 基于 **Qwen3-VL-4B** [2] backbone，包含：
- **Visual Encoder**: SigLIP2-Large，使用 2D-RoPE with interpolated absolute positional embeddings，输出 $f_v \in \mathbb{R}^{N_v \times 2560}$
- **LLM Backbone**: 处理 visual tokens + language tokens $(f_l \in \mathbb{R}^{N_l \times 2560})$
- **Action Tokenizer**: parameter-free discretization，扩展 vocabulary，parallel decoding with chunk size 8
- **Value Head**: 4-layer MLP，与 actor 共享 backbone，用于 RL 中的 state value estimation

关键的 hybrid attention mask 设计（Figure 6）：
- Vision + text + latent tokens 使用 **causal lower-triangular mask**（autoregressive reasoning）
- `<latent_end>` transition token 聚合 reasoning context
- Action tokens 使用 **bidirectional mask**（parallel decoding，chunk 内的 action tokens 互相 attend）

这个设计本质上是把 "slow reasoning" (autoregressive) 和 "fast execution" (parallel) 解耦在 attention 层面，类似于 System 1 / System 2 的 dual-system 思想，参考 [Fast-in-Slow](https://arxiv.org/abs/2506.01953)。

### 2.2 Latent Representation 设计（关键创新点）

Prior 方法的 latent representation 问题：
- **Global Pooling** [21]: average pooling SigLIP features，丢失 fine-grained spatial structure
- **Learnable parameters / Conv downsampling** [22]: 引入 inductive bias，from-scratch 训练
- **Q-Former** [44]: 单个 learnable query 通过 cross-attention 聚合，容易过拟合

LaST-R1 的方案：**DINOv3 [31] 的 `<CLS>` token + top-k channel selection**：
1. 从 DINOv3 提取 `<CLS>` token $f_d \in \mathbb{R}^{1 \times 4096}$
2. 沿 channel dimension 做 top-k selection（k=2560，匹配 VLA embedding size）
3. **完全 offline precompute**，训练和推理时 zero overhead

这里的 intuition：DINOv3 通过 self-supervised distillation 训练得到的 `<CLS>` token 包含 structurally rich, semantically dense 的 global representation，比 SigLIP 这种 contrastive language-image pretraining 的 features 更适合作为 "physical world modeling" 的 anchor。DINOv3 的 features 已被证明在 dense prediction, object discovery 等任务上远超 CLIP-family features。参考 [DINOv2/DINOv3](https://arxiv.org/abs/2508.10104)。

## 三、LAPO 算法数学详解

### 3.1 Formulation

VLA policy $\pi_\theta$ maps multimodal observation $s_t$ to action chunk $\mathbf{a}_{t:t+H} \in SE(3)$。Single-arm: 7-DoF (3 position + 3 Euler orientation + 1 gripper)，Dual-arm: 14-DoF。

**SFT objective** (Eq. 1)：
$$\mathcal{L}_{\text{SFT}}(\theta) = \mathbb{E}_{(s_t, \mathbf{a}_{t:t+H}) \sim \mathcal{D}} \left[ \log \pi_\theta(\mathbf{a}_{t:t+H} \mid s_t) \right]$$

**RL objective** (Eq. 2)：
$$\mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$
$$\nabla_\theta \mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_{t:t+H} \mid s_t) \hat{A}_t \right]$$

其中 $\gamma \in [0, 1)$ 是 discount factor，$\hat{A}_t$ 是 advantage estimate（用 GAE 计算）。

### 3.2 Step-level Likelihood Ratio（核心创新）

LAPO 的关键思想：把 latent tokens $\mathbf{Z}_t = \{\mathbf{z}_{t,k}\}_{k=1}^{N_z}$ 视为 implicit decision variables，让 reward signal 同时塑造 reasoning space 和 action space。

**Action likelihood ratio**:
$$r_t^a(\theta) = \exp\left( \log \pi_\theta(\mathbf{C}_t \mid \cdot) - \log \pi_{\theta_{\text{old}}}(\mathbf{C}_t \mid \cdot) \right)$$
其中 $\mathbf{C}_t = \{\mathbf{a}_{t,j}\}_{j=1}^{N_a}$ 是离散 action tokens，$\log \pi_\theta(\mathbf{C}_t \mid \cdot) = \sum_{j=1}^{N_a} \log \pi_\theta(\mathbf{a}_{t,j} \mid \cdot)$（joint log-prob 通过 token-wise sum）。

**Latent likelihood ratio** (Eq. 3)：
$$r_t^z(\theta) = \frac{\pi_\theta(\mathbf{Z}_t^{\text{old}} \mid \cdot)}{\pi_{\theta_{\text{old}}}(\mathbf{Z}_t^{\text{old}} \mid \cdot)} = \exp\left( -\frac{1}{2\sigma^2} \sum_{k=1}^{N_z} \lVert \mathbf{z}_{t,k}^{\text{old}} - \mathbf{z}_{t,k}^\theta \rVert^2 \right)$$

变量解释：
- $\mathbf{z}_{t,k}^{\text{old}}$：rollout 时 old policy 生成的第 $k$ 个 latent token
- $\mathbf{z}_{t,k}^\theta$：update 时新 policy 在相同 context 下生成的对应 latent token
- $\sigma$：fixed hyperparameter，控制 latent distribution 的 variance（类似 Gaussian policy 的 std）
- $N_z$：latent sequence length

这里 intuition：把 continuous latent 空间的 likelihood ratio 近似为 isotropic Gaussian，让 reward signal 通过 advantage $\hat{A}_t$ 间接塑造 latent representation 的 manifold。当 $\hat{A}_t > 0$（trajectory 好），优化会 pull 当前 latent 朝向 "good-reasoning" manifold；当 $\hat{A}_t < 0$，推开 bad reasoning。

### 3.3 Joint Clipped Surrogate Loss (Eq. 4)

$$\mathcal{L}_{\text{policy}}(\theta) = -\mathbb{E}_t \left[ \sum_{m \in \{z, a\}} \min\left( r_t^m(\theta) \hat{A}_t, \text{clip}(r_t^m(\theta), 1-\epsilon_{\min}, 1+\epsilon_{\max}) \hat{A}_t \right) \right]$$

变量：
- $m \in \{z, a\}$：分别对应 latent 和 action 两个 space
- $\epsilon_{\min}, \epsilon_{\max}$：asymmetric clipping thresholds（实验中 $\epsilon_{\min}=0.2, \epsilon_{\max}=0.28$）

实践上 decouple 为两个独立 loss：$\mathcal{L}_{\text{action}}(\theta)$ 和 $\mathcal{L}_{\text{latent}}(\theta)$。

### 3.4 Total Loss (Eq. 5)

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{action}}(\theta) + \lambda_1 \mathcal{L}_{\text{latent}}(\theta) + \lambda_2 \mathcal{L}_{\text{value}}(\theta)$$

其中 $\mathcal{L}_{\text{value}}(\theta) = \mathbb{E}_t [(v_t - \hat{r}_t)^2]$ 是 MSE value loss。

实验最优配置：$\lambda_1 = 0.1, \lambda_2 = 1$。从 Figure 7 ablation 看：
- $\lambda_1 = 0$（无 explicit latent supervision）：97.2% SR
- $\lambda_1 = 0.1$：99.8% SR（最优）
- $\lambda_1 = 1$：99.0% SR（latent loss 过强 overshadow action）

## 四、Adaptive Latent CoT Mechanism

### 4.1 Dynamic `<latent_end>` emission

把 `<latent_end>` 从 deterministic terminator 变为 dynamic transition signal：
- 若 policy 预测 `<latent_end>` 的 confidence $p \geq 0.99$，则终止 reasoning
- 限制 candidate positions 为 $M=4$ 个（e.g., after 2, 4, 6, 8 latent tokens），max length $N_{\max}=8$

### 4.2 Exploration via Length Sampling (Eq. 6)

提取 $M$ 个 candidate positions 的 pre-softmax logits $l_m$，加 temperature $\beta$：
$$p_m = \frac{\exp(l_m / \beta)}{\sum_{j=1}^{M} \exp(l_j / \beta)}, \quad \forall m \in \{1, \ldots, M\}$$

从 Categorical distribution 采样 $m \sim \text{Categorical}(p_1, \ldots, p_M)$ 决定 reasoning length。

训练时 sampling 探索，inference 时 confidence-based greedy exit。

### 4.3 Adaptive Length Optimization (Eq. 7)

对 `<latent_end>` token 加单独的 policy loss：
$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{action}}(\theta) + \lambda_1 \mathcal{L}_{\text{latent}}(\theta) + \lambda_2 \mathcal{L}_{\text{value}}(\theta) + \lambda_3 \mathcal{L}_{\text{end}}(\theta)$$

其中 $\lambda_3 = 0.1$ 最优。$\lambda_3 = 2$ 会导致 performance drop 到 98.6%（过度惩罚破坏探索平衡）。

### 4.4 Empirical Validation of Adaptive Length

Figure 8 显示 RL 前后 reasoning length 分布的 dramatic shift：
- **Warm-up** (uniform random sampling)：lengths 2/4/6/8 均匀分布
- **After RL**：heavily skew 到 2 或 4 tokens

这表明 policy 学会了 "early-exit" strategy——对简单 state 早停，节省 compute；对复杂 state 用更长 reasoning。这与 [DeepSeek-R1](https://arxiv.org/abs/2501.12948) 在 LLM reasoning 中观察到的 dynamic length adaptation 现象一致。

## 五、实验数据深度分析

### 5.1 LIBERO Benchmark (Table 1)

| Model | Paradigm | Spatial | Object | Goal | Long | Average |
|-------|----------|---------|--------|------|------|---------|
| OpenVLA [4] | SFT | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| GR00T-N1 [42] | SFT | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 [6] | SFT | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 [7] | SFT | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT [5] | SFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| SimpleVLA-RL [24] | RL | 98.2 | 98.7 | 98.8 | 91.7 | 96.9 |
| πRL [23] | RL | 99.6 | 100.0 | 99.6 | 94.0 | 98.3 |
| **LaST-R1** | RL | **99.8** | **100.0** | **100.0** | **99.4** | **99.8** |

关键观察：
1. **One-shot warm-up** 击败 full-dataset SFT baselines（π0.5 96.9%, OpenVLA-OFT 97.1%）
2. 相比同样 one-shot 的 πRL，在 **LIBERO-Long** 上的优势最大（99.4% vs 94.0%），说明 latent reasoning 对 long-horizon tasks 收益显著
3. Figure 3 显示 LAPO 比 Action-Only PPO 收敛更快且 asymptote 更高

### 5.2 Ablation Studies (Figure 4)

**(a) Latent Representation 方法对比**：
- DINOv3 + top-k: **99.8%** (proposed)
- Convolution downsampling: 98.4%
- Q-Former: 97.2%
- Global Pooling: 96.8%

Intuition: DINOv3 的 self-supervised features 比 contrastive CLIP-style features 更适合做 physical reasoning anchor。

**(b) Fixed Latent Length $N_z \in \{1, 2, 4, 8\}$**：
- Action-Only (no latent): 95.0%
- $N_z = 1$: 96.2%
- $N_z = 4$: ~98%
- $N_z = 8$: 98.4%

Performance 单调上升，但 4→8 增益 marginal，故 cap 在 8。

**(c) Adaptive M ∈ {1, 2, 4, 8}**：
- M=1 (fixed length 8): 98.4%
- M=4: **99.8%** (最优)
- M=8: 99.0% (略降，因 flexibility 过度 → 优化不稳定)

### 5.3 Real-World Results (Table 2)

四个 task（1 single-arm + 3 dual-arm）：

| Task | After Warm-up | After RL | Gain |
|------|---------------|----------|------|
| Insert hexagon block | 45% | 90% | +45% |
| Open bag zipper | 55% | 95% | +40% |
| Wipe Vase with Sponge | 65% | 95% | +30% |
| Open bottle cap | 45% | 95% | +50% |
| **Average** | **52.5%** | **93.75%** | **+41.25%** |

**OOD Generalization**（unseen object / background / lighting）：
- Warm-up policy：平均 drop 22-69%
- After RL：平均仅 drop 8%

这是 LaST-R1 最 striking 的结果——RL post-training 不仅提升 in-distribution 性能，更关键的是大幅提升 OOD robustness。Intuition: closed-loop interaction with physical world 让 model 学到 transferable dynamics 而非 memorized demonstrations。

## 六、Intuition Building: 为什么 LAPO 有效？

### 6.1 Latent Reasoning 作为 "Cognitive Buffer"

从 Figure 3 的 learning curve 看，LAPO 的 optimization landscape 显著 smoother。Intuition: latent tokens 充当 "cognitive buffer"，将 high-dimensional action space 的 RL 信号转化为 latent space 的 representation shaping，避免了直接在 action space 上的 brittle optimization。

这与 LLM RLHF 中观察到的现象类似——explicit CoT reasoning tokens 为 reward signal 提供了 gradient pathway，让 credit assignment 更易。

### 6.2 Latent Space 作为 "Implicit World Model"

DINOv3 提供的 global features 实际上是 physical scene 的 compressed representation。Autoregressively 生成 latent tokens 等价于 model 在 "imagining" future states / dynamics，这是 implicit world modeling。LAPO 通过 reward signal 优化这个 world model，使其与 task-relevant dynamics 对齐。

这与 [WorldVLA](https://arxiv.org/abs/2506.21539)、[3D-VLA](https://arxiv.org/abs/2403.09631) 的 explicit world modeling 路径不同——LaST-R1 是 implicit 的，且通过 RL 而非 SFT 来 refine。

### 6.3 Early-Exit 作为 Compute Adaptive Allocation

从 Figure 8 看，RL 优化后 policy 大量使用 short reasoning（2-4 tokens）。这是 task-adaptive compute allocation：
- Simple reactive motion: minimal reasoning
- Complex contact-rich / long-horizon: longer reasoning

这与 [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)、[Universal Transformers](https://arxiv.org/abs/1807.03819) 的思想相通，但在 robotics VLA 中首次通过 RL 实现端到端 learning。

### 6.4 Action-to-Vision Attention (Figure 13)

Grad-CAM 可视化揭示：
- Action-Only SFT: attention diffused, lagging behind end-effector
- LaST-R1 SFT: concentrated on task-relevant objects（latent reasoning 作为 semantic anchor）
- Action-Only + PPO: over-focus on gripper vicinity，缺 long-horizon awareness
- LaST-R1 + LAPO: dynamic attention shift from object → target receptacle as trajectory progresses

这说明 latent reasoning 在 visual grounding 上提供了 strong inductive bias，且 LAPO 进一步 refine 这个 attention pattern 使其与 task progression 对齐。

## 七、Training Pipeline 细节

### 7.1 Large-Scale Pre-Training

- 400K trajectories (28M frames) from Open-X-Embodiment [32], DROID [34], RoboMIND [33]
- DINOv3 latent targets 全部 offline precompute
- 主要 datasets: BridgeV2 (20.82%), Kuka (20.22%), Fractal (13.67%), Robo-Net (11.53%)

### 7.2 Warm-up SFT

- 8×H20 GPUs, Accelerate + DeepSpeed bf16
- Global batch size 64, AdamW, peak LR $1\times10^{-5}$, cosine decay
- LIBERO: 1 expert trajectory/task × 10K iterations
- Real-world: 20 trajectories/task × 1K iterations
- Loss = cosine sim (latent) : CE (`<latent_end>`) : CE (action) = 1 : 0.1 : 1

### 7.3 RL Post-Training on LIBERO

- verl [86] + Ray [87] + FSDP [88], 8×H20 GPU
- Action chunk size 8 (56 tokens), temperature 1.6
- Trajectory cap: Spatial 240, Object/Goal 320, Long 576 steps
- Sparse binary reward × 5 scaling at terminal
- GAE: $\gamma=0.99, \lambda=0.95$
- Rollout batch 512, 4 mini-batches × 4 PPO epochs
- Actor LR $3\times10^{-5}$, Value head LR $3\times10^{-4}$
- Asymmetric clipping: $\epsilon_{\min}=0.2, \epsilon_{\max}=0.28$

### 7.4 Real-World RL

- Franka Research 3 + Intel RealSense D455 + 2× D435
- 2× RTX 4090
- Continuous async actor-learner pipeline（参考 [SERL](https://arxiv.org/abs/2402.05031)）
- LoRA rank $r=32$，freeze base model
- Mixed BC loss ($\lambda_{\text{BC}}=1.0$) + Q-guided policy improvement ($\lambda_Q=0.5$)
- Critic-to-actor ratio 2:1, $\gamma=0.98$, soft update $\tau=0.005$
- Terminal reward +10, step penalty −0.05

## 八、与相关工作对比

### 8.1 vs. LaST_0 [21]

LaST_0 是同一作者群前作，提出 latent spatio-temporal CoT，但**仅 SFT**，依赖 average pooling 的 latent representation。LaST-R1 的进化：
- Latent representation: average pooling → DINOv3 `<CLS>` + top-k
- Training paradigm: SFT-only → SFT + LAPO RL
- Adaptive length: fixed → dynamic `<latent_end>`

### 8.2 vs. SimpleVLA-RL [24] / πRL [23]

两者都是 RL post-training for VLA，但：
- SimpleVLA-RL: GRPO on action tokens only
- πRL: PPO on continuous flow actions only
- **LaST-R1: LAPO jointly optimize latent + action**

### 8.3 vs. π0.5 / π0.6* [7, 30]

π0.5/π0.6* 引入 experience-based learning，但 latent reasoning 仍是 fixed architecture。LaST-R1 的 RL 直接 reshape latent space，更 explicit。

## 九、潜在局限与未来方向

### 9.1 潜在问题

1. **DINOv3 dependency**: latent targets 依赖 fixed DINOv3，无法 co-evolve with policy。未来可考虑 joint fine-tuning 或 latent distillation。
2. **Gaussian assumption for latent likelihood**: $r_t^z$ 的 isotropic Gaussian 近似可能过于简化，real latent distribution 可能 anisotropic。
3. **Sparse binary reward**: LIBERO 用 0/1 success reward，real-world 用 +10 terminal + step penalty。更 dense 的 reward shaping 可能进一步提升 sample efficiency。
4. **Real-world RL safety**: 真机 RL 需要 human intervention buffer，scalability 仍受限。
5. **Single view in LIBERO**: real-world 用 3 cameras，sim 用 single view，domain gap 可能影响 sim-to-real。

### 9.2 未来方向猜想

1. **Hierarchical Latent CoT**: 多层 latent abstraction（low-level dynamics + high-level planning），类似 [Fast-in-Slow](https://arxiv.org/abs/2506.01953)。
2. **Latent World Model Rollout**: 在 latent space 做 multi-step imagination（类似 [DreamerV3](https://arxiv.org/abs/2306.10172)），可能进一步提升 long-horizon planning。
3. **Cross-embodiment Latent Transfer**: DINOv3 latent 是 embodiment-agnostic，可探索 cross-robot generalization。
4. **Offline-to-Online LAPO**: 结合 [Diffusion-DPO](https://arxiv.org/abs/2405.13239) 类似的 offline preference alignment + online LAPO。
5. **Vision-Language Latent Alignment**: 类似 [VLM-RM](https://arxiv.org/abs/2310.09136) 用 VLM 作为 reward model 来 align latent reasoning。

## 十、Reference Links

核心论文：
- [LaST-R1 Project Page](https://siriyep.github.io/last-r1/)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2502.09679)
- [DINOv3](https://arxiv.org/abs/2508.10104)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [π0](https://arxiv.org/abs/2410.24164)
- [π0.5](https://arxiv.org/abs/2504.16054)
- [SimpleVLA-RL](https://arxiv.org/abs/2509.09674)
- [πRL](https://arxiv.org/abs/2510.25889)
- [VLA-RL](https://arxiv.org/abs/2505.18719)
- [LaST_0 (前作)](https://arxiv.org/abs/2601.05248)
- [InternVLA-A1](https://arxiv.org/abs/2601.02456)
- [Latent Reasoning Survey](https://arxiv.org/abs/2505.16782)
- [LIBERO Benchmark](https://arxiv.org/abs/2306.03310)
- [PPO](https://arxiv.org/abs/1707.06347)
- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [DPO](https://arxiv.org/abs/2305.18290)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [SERL](https://arxiv.org/abs/2402.05031)
- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [ECoT](https://arxiv.org/abs/2407.08693)
- [CoT-VLA](https://arxiv.org/abs/2503.22025)
- [3D-VLA](https://arxiv.org/abs/2403.09631)
- [WorldVLA](https://arxiv.org/abs/2506.21539)
- [Fast-in-Slow](https://arxiv.org/abs/2506.01953)
- [HybridVLA](https://arxiv.org/abs/2503.10631)
- [verl RLHF framework](https://arxiv.org/abs/2409.19256)

相关 reasoning RL 工作：
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- [Self-Taught Reasoner](https://arxiv.org/abs/2203.02155)

经典 RL + Robotics：
- [QT-Opt](https://arxiv.org/abs/1806.10293)
- [DreamerV3](https://arxiv.org/abs/2306.10172)
- [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)

---

## 总结

LaST-R1 的核心贡献是把 VLA 的 RL post-training 从 "action-only optimization" 推进到 "joint latent reasoning + action optimization"。LAPO 通过 step-level likelihood ratio 让 reward signal 同时塑造 internal reasoning space 和 external action space，而 adaptive `<latent_end>` mechanism 实现了 task-adaptive compute allocation。

从 intuition 角度看，这篇工作的深层启示在于：**reasoning tokens 不仅服务于 inference-time computation，更是 RL credit assignment 的关键载体**。这与 LLM RLHF 中 CoT tokens 的作用机制异曲同工——latent reasoning 为 sparse environmental reward 提供了 dense gradient pathway，让 policy 能够 "think about what to think about"。

个人而言，我认为这篇工作最 interesting 的方向是 DINOv3 latent anchor 的选择——把 self-supervised vision foundation model 作为 "physical world prior" 注入 VLA reasoning，可能是连接 perception 与 control 的一条 promising path。如果未来能把 latent space 与 explicit 3D scene representation（如 [3D-VLA](https://arxiv.org/abs/2403.09631)）对齐，可能进一步解锁 spatial reasoning 能力。
