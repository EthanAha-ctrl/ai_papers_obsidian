---
source_pdf: Navigation World Models.pdf
paper_sha256: 33b6532143148b60d991f7cfc0c6ff2c168a54c8dcd285e4dbee5eec84c51a5e
processed_at: '2026-08-05T21:57:10-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# NWM 用人话讲

好, 我把公式全扔了, 像在白板前给同事讲 idea 那样说一遍。

---

## 一句话版本

**训一个能"想象未来"的视频生成模型, 给它一张图 + 一个动作, 它生成你执行这个动作后会看到什么。然后用它来规划导航路径。**

就这么简单。剩下的都是工程细节。

---

## 为什么这事重要

现在 robot navigation 的主流做法是 behavior cloning: 给一堆 (observation, action) pair, 训一个 network 直接 mapping obs → action。NoMaD, GNM, ViNT 全是这套。

问题在哪?

**训完就冻住了**。你说"这个走廊别左转", 没辙, 得 retrain。你说"这个场景比较难, 多想一会儿", 也没辙, feed-forward network 计算量固定。

人类不是这么导航的。你走陌生商场, 会先在脑子里"想象"走左边会看到什么、走右边会看到什么, 然后选一个看起来能到目标的。这就是 planning, 不是 reaction。

NWM 就是想把这个"想象"能力学出来。

---

## 怎么学"想象"

数据是机器人第一视角视频 + 每一帧对应的动作 (往前走多少、转了多少度)。

训练任务: 给模型看过去 4 帧图 + 一个动作, 让它生成下一帧图。

就这么训。训 1B 参数的 diffusion transformer, 在 4 个 robot 数据集 + Ego4D 人类视频上一起训。

训完之后, 给它第一帧 + 一串动作, 它能 autoregressive 地生成后面 16 秒视频 — 你走这条路线会看到什么。

---

## 关键 trick 1: time shift 当 action

Ego4D 人类视频没有 action label (你不知道视频里人每秒往前走几米)。但所有视频都有一个天然 action: **时间**。

所以把 action 从 (平移, 转向) 扩成 (平移, 转向, 时间间隔)。

- 机器人数据: 三个都有
- Ego4D: 只有时间间隔, 平移转向设为零

一个架构, 两类数据, 无缝混训。这招很漂亮。

时间间隔范围 ±16 秒, 所以模型不只能预测下一帧, 能直接跳到 16 秒后。这让模型学到"走 16 秒大概到哪"这种粗粒度 prior。

---

## 关键 trick 2: CDiT 架构

标准 DiT 的问题: 把过去 4 帧的所有 token 拼一起做 self-attention, 计算量跟帧数平方成正比。1B 模型训不动。

CDiT 的 insight: **target frame 自己内部做 self-attention, past frames 只做 cross-attention 的 key/value**。

打个比方:
- DiT: 4 帧所有人坐一桌开会, 每人跟桌上所有人聊
- CDiT: 当前要画的那帧坐主桌自己人开会, 过去 3 帧坐旁边当"顾问", 主桌的人随时向顾问提问

计算量从 $O(m^2)$ 降到 $O(m)$, $m$ 是帧数。实测 4× FLOPs 节省, 而且效果更好 — 因为 target frame 内部的 dense interaction 没被稀释。

这跟 Flamingo (DeepMind) 处理多图的方式一模一样, 跟 Perceiver 也是同源思想。**凡是 long context + autoregressive, 都该考虑这种分离**。

---

## 怎么用来导航

两种模式:

### Mode A: standalone planning

给当前图 + 目标图, 找一串动作让模型生成的最后一帧跟目标最像。

具体: 假设轨迹是直线 + 末尾转弯, 只优化 3 个参数 (Δx, Δy, φ)。用 CEM (交叉熵方法, 一种无梯度优化):

1. 采 120 条候选轨迹
2. 每条用 NWM 生成视频, 比较最后一帧跟目标的 LPIPS
3. 选最好的, 更新采样分布
4. 跑 1 轮就够 (2 秒短规划)

结果: ATE 1.13, 比 NoMaD 的 1.93 好 41%。**一个视频生成模型在导航任务上打败了专门的导航 policy**。

为什么能赢? 因为 NoMaD 是 feed-forward, 看到 obs 直接吐 action, 没 search。NWM 是 plan + simulate + evaluate, 有 search。Search 本质上把更多 inference-time compute 换成了性能, 跟 AlphaGo MCTS vs supervised policy 一个道理。

### Mode B: ranking

拿现成 policy (NoMaD) 采 32 条轨迹, 用 NWM 每条都 simulate 一遍, 选 LPIPS 最低的执行。

这是 "world model 当 verifier", 跟 LLM 里 best-of-n sampling 一个套路。简单有效, ATE 从 1.93 降到 1.78。

---

## 为什么能加约束

Energy function 长这样 (人话):

```
cost = -相似度(最后帧, 目标) 
     + 1000 × (动作违反约束) 
     + 1000 × (状态不安全)
```

想加"别左转"? 把 action 集合里左转的设成 invalid, cost 暴增, CEM 自然不会选。

想加"别靠近悬崖边"? 判断生成帧里是不是悬崖, 是就 penalty。

**约束在 planning 时加, 不用 retrain**。这是 NWM 相对 supervised policy 最大的架构优势。论文里试了三种约束 (先直行后转、先转后直行等), 都能 meet, 性能只轻微下降。

---

## 泛化到没见过的环境

在 4 个 robot 数据集上训完, 拿去 Go Stanford (没见过的建筑) 测, 性能当然掉。

加 Ego4D (908 小时人类 egocentric 视频, 只有 time shift action) 一起训, Go Stanford 上 LPIPS 从 0.658 微微降到 0.652, 同时 in-domain RECON 从 0.295 升到 0.368 (变差)。

Trade-off, 但是值得的 — 拿一点 in-domain 性能换 OOD 泛化。这跟 LLM pretraining 经验完全一致: 多 domain 数据, 单一 domain 性能降, 整体泛化升。

定性效果 (Figure 8): 给一张没见过的图, NWM 能"想象"出走过去会看到什么, 虽然几秒后开始 mode collapse (生成结果慢慢 drift 向训练分布)。

---

## 工程: 能 real-time 吗

原始 NWM 生成一条 4 秒轨迹要 30 秒, 不能用。

三个 trick 叠加:

1. **Time skip**: 把相邻 action 合并, 16 步并成 8 步, 14.7 秒
2. **Distillation**: diffusion 250 步蒸馏到 6 步 (Phased Consistency Model), 0.4 秒
3. **4-bit quantization**: GPTQ, 估计 0.1 秒

前两个已经实测, 能跑到 2-10 Hz, 够 robot 用了。

---

## 跟其他 world model 的关系

| 工作 | 什么模型 | 什么环境 | 区别 |
|---|---|---|---|
| Dreamer (Hafner) | RSSM latent | Atari, mujoco | latent dynamics, 不生成 video |
| DIAMOND | diffusion UNet | Atari | 单环境, 小模型 |
| GameNGen | diffusion | Doom | 单环境, 1B+ 参数 |
| Genie | latent action | 2D 游戏 | 学 latent action |
| **NWM** | **diffusion transformer** | **多 robot + 人类** | **跨 embodiment, 真 navigation** |

NWM 的独特之处: 单一模型跨 4 种 robot + 人类视频, 做真导航规划。DIAMOND/GameNGen 是游戏, 不用处理 cross-embodiment 和真实世界视觉复杂度。

跟 LeCun 自己的 DINO-WM (https://arxiv.org/abs/2410.06984) 思路同源: 都是 pretrained visual feature + world model + CEM planning。NWM 是 generative (diffusion) 版本, DINO-WM 是 latent predictive 版本。

---

## 我的直觉总结

1. **Generation 就是 planning 的基础设施**。能 generate 未来 = 能 evaluate 未来 = 能 search。Sora 是 "world simulator" 的说法被群嘲, 但 NWM 在 navigation 上实锤了: 生成模型可以 drive 真实决策。

2. **Cross-attention 分离是通用 pattern**。CDiT 这个 target-self-attn + context-cross-attn 设计, 凡是 autoregressive + long context 都该考虑。LLM 处理多图、长 video, 迟早也得这样。

3. **Time shift = universal action**。任何无 label 视频都有"时间"这个 action, 都能塞进 action-conditioned world model。这给了 huge 无标签数据利用通道。

4. **Search beats reaction**。NWM standalone 打败 NoMaD 说明: 在 navigation 这个任务, inference-time search (MPC) > feed-forward policy。这跟 o1 / AlphaGo 的信号一致 — 推理时多想比参数做大更划算。

5. **Mode collapse 是当前 generation-based world model 的最大软肋**。OOD 环境几秒后就 drift, 论文 Figure 10 诚实地展示了。这是 diffusion world model 要真正落地 robot 必须解决的, 估计要靠更长 context + 更强 conditioning + 可能要 hybrid (latent + generative)。

---

## 相关链接

- NWM (推测 arXiv): https://arxiv.org/abs/2410.07222
- DIAMOND: https://arxiv.org/abs/2405.12399
- GameNGen: https://arxiv.org/abs/2408.14837
- NoMaD: https://arxiv.org/abs/2310.07393
- GNM: https://arxiv.org/abs/2210.03335
- ViNT: https://arxiv.org/abs/2306.14846
- DINO-WM: https://arxiv.org/abs/2410.06984
- DiT: https://arxiv.org/abs/2212.09748
- Flamingo (cross-attn 哲学): https://arxiv.org/abs/2204.14198
- Perceiver: https://arxiv.org/abs/2103.03206
- Genie: https://arxiv.org/abs/2401.15461
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Sora as world simulator: https://openai.com/research/video-generation-models-as-world-simulators
- LeCun path to autonomous intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Phased Consistency Models: https://arxiv.org/abs/2310.05444
- Ego4D: https://arxiv.org/abs/2110.07058

---

要我展开哪块都行 — 比如为什么 CEM 比 gradient descent 适合这里, 或者 mode collapse 具体怎么破, 或者怎么扩到机械臂 6-DoF。

---

# Navigation World Models 深度解析

非常精彩的一篇 paper, Amir Bar, Yann LeCun 等人 (FAIR + NYU + Berkeley) 的工作。这篇工作把 controllable video generation, world model, visual navigation 三条线拧到了一起, 核心创新是 CDiT (Conditional Diffusion Transformer) 架构 + 一个统一的 action-conditioned next-frame prediction formulation, 然后 model predictive control (MPC) 的方式来 plan trajectory。下面我从 intuition 出发, 把每个技术细节讲透。

---

## 1. 核心动机与 problem framing

### 1.1 为什么不用 supervised policy

当前 SOTA visual navigation policy 比如 NoMaD [55], GNM [53] 都是 "hard-coded" behavior cloning / diffusion policy。训练完之后, 你无法动态加入新约束 ("no left turns" "don't walk along cliff edge"), 也无法对 hard case 动态分配更多 compute。

NWM 的 insight 是: **学一个 environment simulator, planning 阶段再约束**。simulator 本身 model-agnostic, 可以把 constraint 写进 energy function, 用 derivative-free optimizer (CEM) 求解。这正好对应 LeCun 一直推的 "model-based planning + JEPA-style world model" 思路 (参考 LeCun 的 "A Path Towards Autonomous Machine Intelligence" https://openreview.net/pdf?id=BZ5a1r-kVsf)。

### 1.2 与 DIAMOND / GameNGen 的区别

DIAMOND (https://arxiv.org/abs/2405.12399) 和 GameNGen (https://arxiv.org/abs/2408.14837) 也是 diffusion world model, 但是 single-environment (Atari / Doom), 不需要 cross-embodiment generalization。NWM 关键挑战: **单一模型跨 robot (Spot, Jackal, Viking ATV, Roomba) + human (Ego4D)**, 而且不同 agent 的 step size 不同, action space 要 normalize。

### 1.3 与 NeRF / NVS 的区别

Novel View Synthesis 比如 NeRF (https://arxiv.org/abs/2003.08934), Zero-1-to-3 (https://arxiv.org/abs/2306.02868), GDC (https://arxiv.org/abs/2408.14841) 都依赖 explicit 3D priors。NWM 不要 3D, 直接从自然视频学 temporal dynamics。这一点很重要——它把 navigation 从 metric map / SLAM 那一套里解放出来, 让 model 自己 emerge allocentric representation (paper §6 讨论, 引用 [65] https://www.biorxiv.org/content/10.1101/2022.01.31.478370v1)。

---

## 2. Formulation 详解 (Equation 1, 2)

### 2.1 数据与动作

数据集:
$$D = \{(x_0, a_0, \dots, x_T, a_T)\}_{i=1}^n$$

- $x_i \in \mathbb{R}^{H \times W \times 3}$: egocentric RGB image
- $a_i = (u, \phi)$:
  - $u \in \mathbb{R}^2$: translation, 第一维 forward/backward, 第二维 right/left
  - $\phi \in \mathbb{R}$: yaw rotation angle (绕垂直轴)

注意 $a$ 是 3-DoF, 没有 pitch/roll, 这也是 paper §5 limitation 里提到的 future work: 扩到 6-DoF + 机械臂 joint。

### 2.2 World model 定义 (Eq. 1)

$$s_i = \text{enc}_\theta(x_i), \quad s_{\tau+1} \sim F_\theta(s_{\tau+1} \mid \mathbf{s}_\tau, a_\tau) \tag{1}$$

- $s_i$: pretrained VAE (Stable Diffusion VAE, 同 DiT) encoder 输出的 latent, 在压缩 latent 空间工作 → 显著节省 compute
- $\mathbf{s}_\tau = (s_\tau, s_{\tau-1}, \dots, s_{\tau-m})$: 过去 $m$ 个 observation 的 latent stack, paper 里 $m=4$
- $F_\theta$: stochastic mapping (diffusion 实现 stochasticity), 这点很关键——navigation 环境本质 stochastic (行人移动、物体被推动)

### 2.3 Time-shift 扩展 (Eq. 2) — 这是最 tricky 的部分

原 Eq.1 只能预测紧邻的下一帧。他们扩展 action: $\boldsymbol{a}_\tau = (u, \phi, k)$, 其中 $k \in [T_{\min}, T_{\max}]$ 是 time shift, 范围 ±16 秒。

给定当前 state $s_\tau$, 训练时随机采 $k$, 把 $s_{\tau+k}$ 当 target。对应的 cumulative action 从 $\tau$ 到 $m = \tau + k - 1$:

$$u_{\tau m} = \sum_{t=\tau}^{m} u_t, \quad \phi_{\tau m} = \sum_{t=\tau}^{m} \phi_t \mod 2\pi \tag{2}$$

- $u_{\tau m}$: 累积平移
- $\phi_{\tau m}$: 累积 yaw, $\bmod 2\pi$ 处理周期性
- $m = \tau + k - 1$: 累积的终点时间 index

**Intuition**: 这样模型既能学 "给定动作, 走到哪儿", 也能学 "给定时间间隔, 走到哪儿"。Ego4D 里只有视频没 action label, 只能用 $k$; 机器人数据有 action label, 可以同时用 $u, \phi, k$。

**Entanglement 风险**: 如果某个 landmark 永远在固定时刻 $t^*$ 出现, 模型会只依赖 $k$ 忽略 $u, \phi$, 或者反过来。Paper §3.1 末尾说用 natural counterfactuals (同一区域不同时间到达) + 多 goal sampling (4 个 goal/state) 来缓解。

---

## 3. CDiT 架构 (核心贡献)

### 3.1 标准 DiT 的问题

DiT (https://arxiv.org/abs/2212.09748) 把所有 context token 拼成一个长 sequence 做全 self-attention:

- 每帧 $n$ 个 token, $m$ 帧 → 总 token 数 $mn$
- Scaled Multi-head Attention 复杂度 $O(m^2 n^2 d)$, $d$ 是 token dim
- 对 context 长度二次, 训 1B 参数时吃不消

### 3.2 CDiT block 设计

CDiT block (Figure 2) 的关键设计:

1. **Self-attention 只在 target frame**: 当前正在 denoise 的 target frame 的 $n$ 个 token 之间做 self-attention, 复杂度 $O(n^2 d)$
2. **Cross-attention 引入 context**: target 的每个 query token 对 past frames 的 $mn$ 个 key/value token 做 attention, 复杂度 $O(mn^2 d)$ — **对 $m$ 线性**
3. **Skip connection** 融合 cross-attention 输出

总复杂度 $O(mn^2 d)$, 与 DiT 的 $O(m^2 n^2 d)$ 相比少了 $m$ 倍。Paper 报告 4× FLOPs 节省 + 性能更好 (Figure 5)。

**为什么 cross-attention 够用?** Intuition 是: target frame 的 token 之间需要做 dense interaction 来 refine 局部纹理; past frames 提供的是 "scene context", 类似 KV cache, 每个 target token 都需要 query 整个历史, 这正是 cross-attention 模式。这其实非常类似 Perceiver (https://arxiv.org/abs/2103.03206) 和 Flamingo (https://arxiv.org/abs/2204.14198) 的设计哲学。

### 3.3 Action & timestep conditioning (Eq. 3)

action $a \in \mathbb{R}^3$ (即 $u_1, u_2, \phi$), 每个 scalar 先 sine-cosine embed 到 $\mathbb{R}^{d/3}$, 再过 2-layer MLP, 拼成 $\psi_a \in \mathbb{R}^d$。timeshift $k$ 和 diffusion timestep $t$ 同样处理。

$$\xi = \psi_a + \psi_k + \psi_t \tag{3}$$

- $\psi_a$: action embedding
- $\psi_k$: time-shift embedding
- $\psi_t$: diffusion timestep embedding
- $\xi$: 聚合 conditioning vector

$\xi$ 喂给 AdaLN (Adaptive Layer Norm, [72] https://arxiv.org/abs/1911.07013), 生成 scale & shift 系数调制 LayerNorm 输出和 attention 输出。这是 DiT 标配。

**对 unlabeled data (Ego4D)**: 直接 omit $\psi_a$, 只留 $\psi_k + \psi_t$。这一招很 elegant — 同一架构无缝支持 labeled 和 unlabeled 数据混合训练。

### 3.4 Diffusion forward / reverse process

Forward:
$$s_{\tau+1}^{(t)} = \sqrt{\alpha_t} s_{\tau+1} + \sqrt{1 - \alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $\alpha_t$: noise schedule (DiT 默认, cosine schedule)
- $t \in \{1, \dots, T\}$: diffusion timestep, $T=250$ (后期 distillation 到 6 步)
- 当 $t \to T$, $s_{\tau+1}^{(t)} \to \epsilon$ (纯噪声)

Reverse: $F_\theta(s_{\tau+1}^{(t)} \mid \mathbf{s}_\tau, a_\tau, t)$ 是 denoiser。

### 3.5 Training objective

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{s_{\tau+1}, a_\tau, \mathbf{s}_\tau, \epsilon, t} \left[ \| s_{\tau+1} - F_\theta(s_{\tau+1}^{(t)} \mid \mathbf{s}_\tau, a_\tau, t) \|_2^2 \right]$$

- target: clean latent $s_{\tau+1}$
- prediction: denoised latent
- $\|\cdot\|_2^2$: MSE in latent space

附加 $\mathcal{L}_{\text{vlb}}$ (variational lower bound, [42] https://arxiv.org/abs/2102.09672) 监督预测 noise 的 covariance matrix, 跟 DiT 一致。

---

## 4. Navigation Planning (Eq. 4, 5)

### 4.1 Energy function

给定初始 latent $s_0$ 和 goal latent $s^*$, 找 action sequence $(a_0, \dots, a_{T-1})$ 使到达 $s^*$ 的 likelihood 最大。

$$\mathcal{E}(s_0, a_0, \dots, a_{T-1}, s_T) = -S(s_T, s^*) + \sum_{\tau=0}^{T-1} \mathbb{I}(a_\tau \notin \mathcal{A}_{\text{valid}}) + \sum_{\tau=0}^{T-1} \mathbb{I}(s_\tau \notin \mathcal{S}_{\text{safe}}) \tag{4}$$

- $S(s_T, s^*)$: unnormalized perceptual similarity score (LPIPS, decode 后用 AlexNet feature 算, [75] https://arxiv.org/abs/1801.03906)
- $\mathcal{A}_{\text{valid}}$: valid action set (约束 e.g. "no left then right")
- $\mathcal{S}_{\text{safe}}$: safe state set (约束 e.g. "don't approach cliff edge")
- $\mathbb{I}(\cdot)$: indicator, 违反时加巨大 penalty

**这是论文最 powerful 的地方**: constraint 是 planning 时 inject 的, 而不是 training 时 hardcode。这对 robotics 极其重要 — 现实 deployment 总有 unseen constraint (新障碍物、新规则)。

### 4.2 Optimization (CEM)

$$\arg\min_{a_0, \dots, a_{T-1}} \mathbb{E}_{\mathbf{s}}[\mathcal{E}(s_0, a_0, \dots, a_{T-1}, s_T)] \tag{5}$$

用 Cross-Entropy Method (CEM, [48] https://www.sciencedirect.com/science/article/pii/S0305048397000223), 一种 derivative-free, population-based 优化。Appendix §7 给了完整 hyperparameter:

- **Parameterize trajectory as straight line**: 只优化 endpoint $(\Delta x, \Delta y, \phi)$, 再均匀映射到 8 个 delta step (yaw 留到最后一步)
- **Gaussian $\mathcal{N}(\mu, \Sigma)$**: $\mu = (\mu_{\Delta x}, \mu_{\Delta y}, \mu_\phi)$, $\Sigma = \text{diag}(\sigma_{\Delta x}^2, \sigma_{\Delta y}^2, \sigma_\phi^2)$
- **$N=120$ candidate / iteration**
- **$M$ 次重复采样** (paper 中 $M=3$) 平均 LPIPS score, 因为 NWM 是 stochastic
- **只跑 1 iteration** (short-horizon 2s 够用)

CEM 选 top-performing candidate, 更新分布参数 (minimize 旧新分布间 cross-entropy, 名字由来)。这套跟 DINO-WM (https://arxiv.org/abs/2410.06984) 用 CEM planning 思路一致。

### 4.3 Ranking 模式 (与 NoMaD 结合)

除了 standalone plan, NWM 还能 rank 一个外部 policy 的 trajectory samples:

1. 从 NoMaD $\Pi(\mathbf{a} \mid s_0, s^*)$ 采 $n \in \{16, 32\}$ 个 trajectory
2. 每个 trajectory 用 NWM autoregressive rollout 得到 $s_T$
3. 算每个 $s_T$ 与 $s^*$ 的 LPIPS
4. 选 LPIPS 最低的执行

这是 "world model as verifier" 思路, 类似 LLM 里的 process reward model + best-of-n sampling。

---

## 5. 实验数据详解

### 5.1 Datasets

| Dataset | Agent | Hours | 用途 |
|---|---|---|---|
| SCAND [30] | Jackal + Spot | 8.7h, 25 miles | social nav |
| TartanDrive [60] | Yamaha Viking ATV | 5h, 630 traj | off-road |
| RECON [52] | Clearpath Jackal | 40h, 9 envs | open-world nav |
| HuRoN [27] | Roomba | 75h, 5 envs, 4k interactions | social indoor |
| GO Stanford [24] | Teleop robot | 25h, 27 buildings | **OOD eval only** |
| Ego4D [18] | Human (egocentric) | 908h subset | unlabeled, time-shift only |

Action 归一化: 把每帧位移除以 agent 的平均 step size (meters), 让不同 robot 的 action space 对齐。所有 backward motion 都被过滤 (跟 NoMaD 一致, backward 容易 jitter)。

### 5.2 主结果: Ablation (Table 1)

RECON 数据集, 4 秒未来预测:

| Ablation | LPIPS ↓ | DreamSim ↓ | PSNR ↑ |
|---|---|---|---|
| #goals=1 | 0.312 | 0.098 | 15.044 |
| #goals=4 | **0.296** | **0.091** | **15.331** |
| context=2 | 0.302 | 0.095 | 15.274 |
| context=4 | **0.296** | **0.091** | **15.331** |
| time only | 0.760 | 0.783 | 7.839 |
| action only | 0.318 | 0.100 | 14.858 |
| **action + time** | **0.295** | **0.091** | 15.343 |

**关键 takeaways**:
1. 多 goal sampling 帮 counterfactual learning
2. context=4 比 context=2 显著好 (short context 会 "lose track")
3. time-only 完全废 (LPIPS 0.76), 因为模型啥都不做只跳时间
4. action + time 协同最好 — 验证了 entanglement 担心没必要

### 5.3 CDiT vs DiT (Figure 5)

CDiT 在 1B 参数 scale 下 LPIPS 显著优于 DiT, 而且 4× FLOPs 更少。即使把 CDiT-L (参数比 DiT-XL 少) 比 DiT-XL, CDiT-L 仍快 4× 且更好。**Architecture > scale** 的一个 clean demonstration。

### 5.4 Video synthesis 质量 (Figure 6, Table 6)

RECON 上 16s @ 4FPS, FVD:

| Model | FVD ↓ |
|---|---|
| DIAMOND | 762.7 ± 3.4 |
| **NWM (1B)** | **200.97 ± 5.6** |

跨数据集:

| Dataset | DIAMOND | NWM |
|---|---|---|
| RECON | 762.7 | **200.9** |
| HuRoN | 881.9 | **276.9** |
| TartanDrive | 2289.7 | **494.2** |
| SCAND | 1945.1 | **401.7** |

NWM 在所有环境 FVD 都 ~3-4× 优于 DIAMOND, 主要是 architecture (CDiT vs UNet) + scale (1B vs DIAMOND's small) + data diversity 共同贡献。

### 5.5 Goal-conditioned navigation (Table 2, 7)

RECON, 2s trajectory:

| Model | ATE ↓ | RPE ↓ |
|---|---|---|
| GNM [53] | 1.87 | 0.73 |
| NoMaD [55] | 1.93 | 0.52 |
| NWM + NoMaD (×16 ranking) | 1.83 | 0.50 |
| NWM + NoMaD (×32 ranking) | 1.78 | 0.48 |
| **NWM standalone planning** | **1.13** | **0.35** |

**NWM standalone 比 NoMaD 好 41% ATE!** 这是很强的结果——一个 video generation model 居然在 navigation 上超越 dedicated navigation policy。Intuition: NWM 学到更 rich 的 environment representation, planning 时能 search, 而 NoMaD 是 feed-forward 没有 search。

跨数据集 (Table 7):
- HuRoN: NWM standalone ATE 4.12 (略差 NoMaD 3.73) — 因为 HuRoN 是 indoor social scene, NWM 学到的 visual prior 不够强
- TartanDrive: NWM 5.63 (远好 NoMaD 6.32, forward baseline 5.75)
- SCAND: NWM 1.28 (远好 NoMaD 2.24)

### 5.6 Constraint-aware planning (Table 3)

三种约束, 报告相对无约束的 final position / yaw 差异:

| Constraint | Rel. $\delta u$ ↓ | Rel. $\delta \phi$ ↓ |
|---|---|---|
| Forward first (forward 5 steps, turn 3) | +0.36 | +0.61 |
| Left-right first (turn 3, forward 5) | -0.03 | +0.20 |
| Straight then forward | +0.08 | +0.22 |

约束下性能小幅下降但合理, 证明 NWM 能在 planning 时 inject 约束而无需 retraining。

### 5.7 Generalization 到 unknown environment (Table 4)

加 Ego4D unlabeled 训练, 在 Go Stanford (OOD) 上:

| Data | Go Stanford LPIPS ↓ | RECON LPIPS ↓ |
|---|---|---|
| In-domain | 0.658 | 0.295 |
| **+ Ego4D** | **0.652** | 0.368 (↑) |

**Trade-off**: Ego4D 提升 OOD 泛化, 但稍微损害 in-domain (因为 Ego4D 分布拉远 in-domain prior)。这跟 SSL pretraining 经典 trade-off 一致。

### 5.8 Runtime (Table 8)

NWM 单 trajectory simulation, NVIDIA RTX 6000 Ada:

| Config | Time (s) |
|---|---|
| Baseline NWM | 30.3 |
| + Time Skip (合并 adjacent action, 16 → 8 states) | 14.7 |
| + Distillation (250 → 6 denoising steps, [70] Phased Consistency Model https://arxiv.org/abs/2310.05444) | 0.4 |
| + 4-bit Quantization (GPTQ [12] https://arxiv.org/abs/2210.17323) | 0.1 (estimated) |

Time Skip + Distillation 组合就能 real-time @ 2-10Hz, 这对部署至关重要。

### 5.9 Test-time adaptation (Table 9)

Go Stanford 上 fine-tune NWM 2k steps:
- ours: 0.652 LPIPS
- ours + TTA: **0.650**

TTA 略有提升, 且与 planning 是 orthogonal。这暗示 NWM 可以快速 adapt 到新环境, 与 LeCun 一贯主张的 "world model 应该是 self-supervised + adaptable" 一致。

---

## 6. Limitations (paper §5)

1. **Mode collapse on OOD**: 在完全没见过的环境, 模型预测慢慢 drift 向 training data 分布 (Figure 10)。GAN 时代经典问题 (https://arxiv.org/abs/1707.04930), diffusion 也没完全解决。
2. **Temporal dynamics 弱**: 行人移动这种 high-frequency dynamic 学不好, 大概是 4-frame context 太短。
3. **3-DoF 限制**: 没有 pitch/roll, 没法爬楼梯 / 机械臂。

未来方向显然: 更长 context (→ 更长 video transformer e.g. 30+ frames), 更多 data, 6-DoF action, 也许 latent action (Genie [6] https://arxiv.org/abs/2401.15461 路线)。

---

## 7. 我的 takeaways (build your intuition)

1. **World model 作为 navigation planner 的核心优势**: 不是学 policy 直接 mapping obs → action, 而是学 dynamics, 让 planner 在 test time search。这种 search-based approach 本质上把 "intelligence" 从 model weights 转移到 inference-time compute — 类似 AlphaGo 的 MCTS vs supervised policy network。

2. **CDiT 的 cross-attention 设计**: 这是一个 generic principle — 任何 autoregressive + long context 场景, 都应该考虑 "target-frame self-attn + context cross-attn" 分离。LLM 里类似的有 GPT-4V 处理多 image 时, image token 不进 full self-attn 而是做 cross-attn 节省 compute。CDiT 给了一个干净的 vision 版本实例。

3. **Time shift 作为 action**: 这是个很聪明的 trick, 把 unlabeled video 也纳入 action-conditioned world model 框架。Ego4D 没有 action label, 但所有 video 都有 "time" 这个天然 action。这让 NWM 能利用海量人类视频学 navigation prior, 类似 VPT (https://arxiv.org/abs/2206.07828) 用 unlabeled Minecraft video 训 prior。

4. **Energy-based formulation**: $\mathcal{E} = -\text{similarity} + \text{constraint penalties}$ 这正是 LeCun 推的 Energy-Based Model (EBM) 思路 (https://openreview.net/forum?id=BZ5a1r-kVsf)。Navigation 是 EBM 落地的一个 clean testbed, 因为约束天然可以写成 indicator function。

5. **Diffusion 作为 world model 的本质**: diffusion 不只是 image generator, 它的 stochastic reverse process 天然能 model 环境的 stochasticity — 给同一 $(s_\tau, a_\tau)$, 不同 noise sample 会得到不同 $s_{\tau+1}$, 这模拟了 "走这一步, 可能撞到人也可能没撞到"。这是 DIAMOND/GameNGen 路线的真正威力。

6. **Scaling law in world model**: CDiT 在 1B 参数仍 better than DiT, 说明 world model 还没到 scaling ceiling, 暗示 NWM-10B / NWM-100B 可能还有显著 gains。这跟 Sora (https://openai.com/research/video-generation-models-as-world-simulators) 是 "world simulator" 的论点呼应。

---

## 8. 进一步阅读 / 相关资源

- Paper PDF (推测 arXiv link): https://arxiv.org/abs/2410.07222 (NWM by Amir Bar et al.)
- 项目主页 (推测): 见 Meta FAIR publication page https://ai.meta.com/publications/
- DIAMOND: https://arxiv.org/abs/2405.12399
- GameNGen: https://arxiv.org/abs/2408.14837
- NoMaD: https://arxiv.org/abs/2310.07393
- GNM: https://arxiv.org/abs/2210.03335
- ViNT: https://arxiv.org/abs/2306.14846
- DINO-WM (similar idea, LeCun group): https://arxiv.org/abs/2410.06984
- Sora as world simulator: https://openai.com/research/video-generation-models-as-world-simulators
- LeCun JEPA / path to autonomous intelligence: https://openreview.net/pdf?id=BZ5a1r-kVsf
- DiT: https://arxiv.org/abs/2212.09748
- Stable Diffusion VAE: https://arxiv.org/abs/2110.02502
- Ego4D: https://arxiv.org/abs/2110.07058
- DreamSim: https://arxiv.org/abs/2306.09325
- Phased Consistency Models (distillation): https://arxiv.org/abs/2310.05444
- Cross-Entropy Method tutorial: https://www.sciencedirect.com/science/article/pii/S0305048397000223
- AdaLN (Understanding and Improving Layer Normalization): https://arxiv.org/abs/1911.07013

---

如果你 (Karpathy) 想进一步深挖某一方向, 比如 (a) CDiT 复杂度严格推导、(b) CEM 在 stochastic world model 下的 variance reduction、(c) NWM 与 JEPA 联系、(d) 如何把这套推广到 manipulation, 我可以再展开。
