---
source_pdf: ZTRS Zero-Imitation End-to-end Autonomous Driving with Trajectory Scoring.pdf
paper_sha256: 01d42209173a1fa3f24234b4953939a18d077d1d53cbbd2ad98f1a47f598fa63
processed_at: '2026-08-13T06:53:36-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ZTRS: 用人话讲讲这篇paper

## 一句话版本

ZTRS 完全抛弃 human driving demonstrations，只用 rule-based rewards，从 scratch 训练一个 end-to-end neural planner，直接吃 image 输出 trajectory，而且效果能 match 甚至超过 IL-based 方法。

---

## 背景：end-to-end driving 的两难

Autonomous driving 有两大 paradigm：

**Modular pipeline** (perception → prediction → planning → control)：可解释、可调试，但 cascading errors 严重，前面错了后面全错。

**End-to-end** (sensor → trajectory)：避免 cascading errors，能 leverage rich semantic cues，但训练难。

End-to-end 内部又分两派：

- **IL (Imitation Learning)**：学人类开的轨迹。问题：covariate shift（部署时 state 分布偏移），human demonstrations 可能 noisy / sub-optimal。
- **RL (Reinforcement Learning)**：用 reward 训练。问题：需要 simulator，高维 sensor simulation 极贵且难真实。之前的 RL 方法大多只用 symbolic inputs（3D boxes、maps），不碰 raw images。

所以现状是：IL 能用 sensor data 但有 inherent 问题，RL robust 但用不了 sensor data。ZTRS 想要 both worlds。

---

## 核心洞察：open-loop metrics 可以当 reward

作者的关键发现：**open-loop planning metrics（EPDMS）可以 offline 计算，不需要 environment interaction**。

给定一个 (state, trajectory) pair，不需要让车真开，就能用 rule-based 方式算出这个 trajectory 好不好——有没有 collision、有没有出车道、舒不舒服、progress 多少。

这意味着：
- 不需要 online interaction
- 不需要 expensive sensor simulator  
- 可以直接用 offline dataset 里的 states，枚举 trajectory candidates 算 reward

问题就变成了 **offline RL**，且 action space 是 **离散可枚举的**（trajectory set）。

---

## 架构（Figure 2 解析）

ZTRS 是一个 trajectory scorer，五个模块：

1. **Image backbone** (V2-99 或 ViT-L)：提取 L 个 image tokens $\{x_{img}^i\}_{i=1}^L$
2. **Trajectory tokenizer**：把 trajectory candidates 编码成 query tokens $\{x_{traj}^i\}_{i=1}^n$
3. **Transformer Decoder**：trajectory queries 对 image tokens 做 cross-attention，获取 driving context
4. **Policy head**：输出每个 trajectory 的概率 $\pi(a|s)$
5. **Scoring heads** (m 个)：输出每个 trajectory 在各个 metric 上的得分 $\{S_i(a|s)\}_{i=1}^m$

Action space $\mathcal{A} = \{a_i\}_{i=1}^n$，默认 $n=16384$，每个 trajectory 跨 4 秒、10Hz。这些 trajectory 用 K-means 在 nuPlan dataset 上 cluster 出来，覆盖大部分 driving possibility。

Inference 时，最终选的 trajectory 是 policy head 输出和 scoring heads 输出的加权平均。

---

## 核心算法：EPO (Exhaustive Policy Optimization)

这是 paper 最核心的技术贡献。

### Policy Gradient 回顾

Standard policy gradient：

$$g = \mathbb{E}[\Psi(s,a) \nabla_\theta \log \pi_\theta(a|s)]$$

变量解释：
- $\Psi(s,a)$：advantage function，衡量 action $a$ 在 state $s$ 下比平均好多少
- $\pi_\theta(a|s)$：policy，参数化 by $\theta$
- $s$：state（包含 sensor data + ego-vehicle status）
- $a$：action（这里是一个 trajectory）
- 期望对 $s \sim \mathcal{D}$（offline dataset），$a \sim \pi(\cdot|s)$ 采样

### 关键推导（Eq. 1→5）

作者做了一个漂亮的 algebraic manipulation。因为 action space 离散，把期望展开成对所有 action 的求和：

$$g = \sum_{a' \in \mathcal{A}} \Psi(s, a') \pi_\theta(a'|s) \nabla_\theta \log \pi_\theta(a'|s)$$

利用恒等式 $\nabla \log x = \frac{\nabla x}{x}$：

$$= \sum_{a'} \Psi(s, a') \pi_\theta(a'|s) \cdot \frac{\nabla_\theta \pi_\theta(a'|s)}{\pi_\theta(a'|s)}$$

$\pi_\theta$ 约掉：

$$= \sum_{a' \in \mathcal{A}} \Psi(s, a') \nabla_\theta \pi_\theta(a'|s)$$

**这就是关键结论**：离散 action space 下，policy gradient 可以写成对 **每个 action 的 likelihood 直接求梯度** 的形式，不需要 log-likelihood，不需要采样。

### EPO formulation

$$g := \sum_{\substack{a' \in \mathcal{A} \\ s \sim \mathcal{D}}} \Psi(s, a') \nabla_\theta \pi_\theta(a'|s)$$

Standard PG 只对 **采样到的一个 action** 做梯度。EPO 对 **整个 action space 的所有 action** 都做梯度。

Intuition：standard PG 像 sparse signal——"猜对一个 action 就奖励，猜错就惩罚"。EPO 像 dense multi-label classification——"对所有可能 action 打分，根据 advantage 调整每个 action 的 likelihood"。supervision density 差了几个量级。

### Advantage 怎么算

$$\Psi(s_t, a_t) = \mathcal{E}(s_t, a_t) - b(s_t, a_t, a_{t-1})$$

变量解释：
- $\mathcal{E}(s_t, a_t)$：EPDMS score，包含 safety、rule-compliance、progress 等多维度
- $b(s_t, a_t, a_{t-1}) = \lambda \cdot \mathbb{1}[\text{EC}(a_{t-1}, a_t)]$：correction term
  - $\lambda = 0.2$：常数权重
  - $a_{t-1} = \arg\max \pi(a|s_{t-1})$：上一时刻 policy 选的 action
  - $\text{EC}$：Extended Comfort metric 的 violation indicator（检测相邻两步 trajectory 是否 comfort-consistent）

$b$ 的作用：如果当前 action 和上一时刻 action 在 comfort 上不一致（比如突然急转弯），给 penalty。保证 temporal consistency，避免 oscillation。

最后 $\Psi$ 做 zero-mean unit-variance normalization（following Huang et al. 2022; Shao et al. 2024）。

---

## EPDMS 详解（Eq. 9）

$$\mathcal{E}(s, a) = \left(\prod_{m \in S_{\text{pen}}} m(s, a)\right) \cdot \left(\frac{\sum_{m \in S_{\text{avg}}} w_m m(s, a)}{\sum_{m \in S_{\text{avg}}} w_m}\right)$$

变量解释：
- $s$：current state
- $a$：4-second trajectory
- $S_{\text{pen}}$：penalty metric set，**乘法** 聚合，任一为 0 则整体为 0。包含：
  - NC (No-at-fault Collisions)
  - DAC (Drivable Area Compliance)
  - DDC (Driving Direction Compliance)
  - TLC (Traffic Light Compliance)
- $S_{\text{avg}}$：weighted metric set，**加权平均** 聚合。包含：
  - TTC (Time-to-Collision)
  - EP (Ego Progress)
  - LK (Lane Keeping)
  - HC (History Comfort)
  - EC (Extended Comfort, from Li et al. 2025b)
- $w_m$：metric $m$ 的 aggregation weight

设计逻辑：safety-critical 的 metric 用乘法（一个 fail 就全 fail），soft metric 用加权平均。这是 NAVSIM 的标准 metric。

---

## 实验

### 三个 benchmark

1. **Navtest** (NAVSIM eval set)：real-world open-loop planning，103k train / 12k eval
2. **Navhard**：在 NAVSIM challenging scenarios 上做 pseudo-simulation，3DGS 合成后续场景，244 initial + 4164 synthetic scenarios
3. **HUGSIM**：closed-loop driving，3DGS-rendered images，345 scenarios，分 easy/medium/hard/extreme 四档

### Navhard 结果（Table 1）

ZTRS (V2-99) 拿到 **45.5% EPDMS**，SOTA。对比：
- GTRS-Dense (ViT-L): 45.3%
- DriveSuprim (EVA-ViT-L): 44.7%
- DriveSuprim (ViT-L): 43.4%
- LTF (ResNet34): 23.1%

关键 metric 细节（ZTRS V2-99 Stage 2）：
- NC: 91.1, DAC: 90.4, DDC: 95.8, TLC: 99.0（safety 都很高）
- EP: 63.6（progress 偏低，说明 ZTRS 偏保守）
- EC: 66.1（comfort 最好，说明 temporal consistency 做得好）

### Navtest 结果（Table 2）

ZTRS 86.2% (ViT-L) / 85.3% (V2-99)，比 Hydra-MDP++ (85.6% ViT-L) 略好，但落后 DriveSuprim (87.1% ViT-L)。DriveSuprim 用了更 advanced 的 data augmentation 和 iterative refinement scoring architecture。

Human agent baseline 是 90.3%，所以还有 gap。

### HUGSIM 结果（Table 3）— zero-shot

ZTRS 没在 simulated data 上训练过，直接 zero-shot 测试：
- Overall RC: 42.6%（比 GTRS-Dense 38.0% 高 4.6%）
- Overall HD-Score: 28.9%（比 GTRS-Dense 28.6% 高 0.3%）
- Easy: RC 74.4, HD-Score 60.8（最强）
- Extreme: RC 21.9, HD-Score 11.0（仍然很难）

---

## Ablation（Table 4）— 最关键

这个 ablation 直接验证 EPO 的核心价值：

| IL | RL | Target | EPDMS | EC |
|---|---|---|---|---|
| ✓ | ✗ | Human trajectory | 86.2 | 80.5 |
| ✓ | ✗ | $\hat{\mathcal{E}}$ (max EPDMS traj) | 76.7 | 18.5 |
| ✗ | ll (likelihood, all actions) | $\mathcal{E}$ | 84.2 | 53.8 |
| ✗ | ll (likelihood, all actions) | $\mathcal{E} - b$ | 85.3 | 77.2 |
| ✗ | log-ll (sampled action) | $\mathcal{E} - b$ | 75.0 | 36.1 |

关键发现：

1. **用 max EPDMS trajectory 做 IL target → 很差（76.7%）**。因为很多 trajectory 都能拿高分，单个 target 学不到 underlying pattern，EC 只有 18.5 说明严重 oscillation。

2. **用 likelihood over all actions (EPO) → 84.2%**，提升 7.5%，但 EC 低（53.8%），有 oscillation。

3. **加 correction term $b$ → 85.3%, EC 77.2%**，解决 oscillation，EC 提升 23.4%。

4. **用 log-likelihood over sampled action (standard PG) → 75.0%**，明显差于 EPO 的 85.3%。这是 EPO 最直接的 validation：同样 reward，sample-based PG 远不如 exhaustive optimization。

### Action space size 实验（Table 5）

| Backbone | Train \|A\| | Infer \|A\| | Navtest | Navhard (real) | Navhard (sim) |
|---|---|---|---|---|---|
| V2-99 | 16384 | 16384 | 85.3 | 74.9 | 43.4 |
| V2-99 | 16384 | 8192 | 82.0 | 74.2 | **45.5** |
| ViT-L | 16384 | 16384 | 86.2 | 76.1 | 38.8 |
| ViT-L | 16384 | 8192 | 84.3 | 73.4 | **45.0** |

发现：real-world data 上大 action space 好，simulated data 上小 action space 好。和 GTRS 发现一致——减小模型复杂度有助于泛化到 unseen simulated data。

---

## 为什么 EPO work — 更深的 intuition

Standard PG 在 continuous / large action space 下必须采样，因为 enumerate 不了。但这里 action space 是 16384 个离散 trajectory，完全可以 batch 处理。

Enumerate 的好处：每个 action 都有梯度信号，不用担心采样不到好 action。这解决了 offline RL 的 **cold-start problem**——random policy 在 continuous space 里几乎采不到好 action，但 enumerate 可以 cover 整个 space。

本质上 EPO 把 RL 问题变成了一个 **structured classification**：给定 state，对所有可能 trajectory 打分，用 advantage 作为 soft label 监督。比 IL 的 hard label（只学一个 human trajectory）信息量大得多。

---

## 为什么不用 expert demonstrations 也能学出来

这是最 impressive 的部分。没有人类驾驶数据，policy 怎么知道"怎么开"？

答案：**reward function $\mathcal{E}$ 本身编码了大量 driving knowledge**。EPDMS 包含 collision avoidance、lane keeping、traffic light compliance、comfort 等，这些 metric 的设计 require domain expertise。

ZTRS 把"学人类轨迹"替换成"学 rule-based metric 的 optimal 解"。人类轨迹是 reward function 的一个 approximation（人类也大概遵守这些 rule），但 reward function 更 precise、更 dense。

某种程度上，这像 **AlphaZero 的思路**——不学人类棋谱，只用 self-play + reward，能超越人类。ZTRS 是 driving 版的"不学人类，只学 reward"。

---

## 局限性和 open questions

1. **Action space 设计很关键**：16384 个 trajectory 是 K-means cluster 出来的，如果 cluster 不好可能 cover 不到某些 maneuver。Discretization 的代价。

2. **Reward engineering**：EPDMS 是 hand-crafted 的，和 IL "learn from data" 相比更依赖 human expertise。

3. **Open-loop reward 的局限**：EPDMS 是 open-loop metric，不考虑其他 agents 的 reactive response。Closed-loop 性能可能受限（HUGSIM 结果不错但那也是 non-reactive 的 3DGS 场景）。

4. **Computational cost**：16384 个 trajectory 都要过 Transformer Decoder，训练 24×A100。Inference 时也要 score 所有 trajectory，real-time 性能存疑。

5. **Navtest 上还落后 DriveSuprim**：说明 IL + advanced architecture 在"简单"场景上更强。ZTRS 优势主要在 safety-critical / hard scenarios。

6. **EP 只有 63.6%**：ZTRS 偏 conservative，progress 不够。Reward 设计可能需要调 trade-off。

---

## 和相关工作的关系

- **Hydra-MDP / GTRS series** (Li et al.)：同一作者线，从 multi-target hydra-distillation → generalized trajectory scoring → ZTRS zero-imitation。清晰的演进路线。

- **DriveSuprim** (Yao et al., 2025)：iterative refinement trajectory scorer，IL-based，Navtest 最强。

- **GigaFlow / CaRL**：大规模 symbolic-input RL，证明 RL 能 scale，但不用 sensor data。ZTRS 补上 sensor-level gap。

- **RAD** (Gao et al., 2025)：3DGS-based RL，但仍需 IL pre-training。ZTRS 完全不需要。

- **DiffusionDrive** (Liao et al., 2025)：diffusion-based trajectory generation，IL-based。

- **Centaur** (Sima et al., 2025)：test-time training for robust driving。

---

## 相关链接

- ZTRS Paper: https://arxiv.org/abs/2507.02701
- ZTRS Code: https://github.com/woxihuanjiangguo/ZTRS
- NAVSIM: https://github.com/autonomousvision/navsim
- HUGSIM: https://arxiv.org/abs/2412.01718
- Hydra-MDP++: https://arxiv.org/abs/2503.12820
- DriveSuprim: https://arxiv.org/abs/2506.06659
- GTRS-Dense: https://arxiv.org/abs/2506.06664
- GigaFlow: https://arxiv.org/abs/2502.03349
- CaRL: https://arxiv.org/abs/2504.17838
- RAD: https://arxiv.org/abs/2502.13144
- Policy Gradient original (Sutton et al. 1999): https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80e0ec5fcbc88-Abstract.html
- Offline RL survey (Levine et al. 2020): https://arxiv.org/abs/2005.01643

---

## 总结

ZTRS 的贡献是 conceptually important 的：它证明了 end-to-end driving 可以完全不用 human demonstrations，只用 rule-based rewards 从 scratch 训练。核心技术 EPO 是一个 elegant 的 trick——在 discrete enumerable action space 下，把 sample-based policy gradient 变成 exhaustive gradient，获得 dense supervision。

结果上，ZTRS 在 safety-critical scenarios (Navhard) 上 SOTA，closed-loop (HUGSIM) 上超过 IL baselines，但 generic scenarios (Navtest) 上还落后最先进的 IL 方法。这符合直觉——reward-driven 方法在 hard cases 上 robust，但 IL 在模仿人类细腻行为上仍有优势。

未来方向：更好的 action space design（adaptive / learned trajectory sets）、closed-loop reactive reward、reducing inference cost。

---

让我仔细分析一下这篇paper，ZTRS: Zero-Imitation End-to-end Autonomous Driving with Trajectory Scoring。

先理解核心点：
1. 这是一个end-to-end autonomous driving的工作
2. 关键创新：完全不用imitation learning (IL)，只用reinforcement learning (RL)的rewards
3. 直接在高维sensor data上操作（不像之前RL方法只用symbolic inputs）
4. 核心算法：Exhaustive Policy Optimization (EPO)，一种针对可枚举action space的policy gradient变体
5. 用离线数据 + EPDMS (open-loop planning metrics) 作为rewards

让我想想作者的身份：是Andrej Karpathy在问。Karpathy是深度学习领域的专家，对neural networks、RL、autonomous driving都非常了解。所以回答要足够
