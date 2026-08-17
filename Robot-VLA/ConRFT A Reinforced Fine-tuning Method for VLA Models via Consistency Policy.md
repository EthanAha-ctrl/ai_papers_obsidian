---
source_pdf: ConRFT A Reinforced Fine-tuning Method for VLA Models via Consistency
  Policy.pdf
paper_sha256: 4356a2f2fed39e9286d22bd0238b918f4e2fa20fe183a0d94c6a37bab0b1667a
processed_at: '2026-08-03T16:57:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ConRFT 用人话讲

## 一句话总结

把一个已经预训练好的 VLA model（比如 Octo），用 20 条人类 demo 做个 offline 热身，然后放到 real robot 上，让 human 拿着 SpaceMouse 当 "safety net"，边探索边学，45-90 分钟就能把 success rate 从 39% 拉到 96%，而且 action 还更利索（episode 长度缩短 1.9 倍）。

核心 trick 就一句话：**用 consistency policy 当 action head，BC loss 和 Q loss 同时上，offline 先稳住，online 再放飞，human 兜底别撞坏东西**。

Project page: https://cccedric.github.io/conrft/

---

## 这篇 paper 到底在解决什么 pain point

先想象你有一个 Octo model（https://octo-models.github.io/），它在 Open X-Embodiment（https://robotics-transformer-x.github.io/）上预训练过，理论上是个 generalist。你现在要让它帮你 Insert Wheel——把椅子轮子插到底盘孔里，这是个 contact-rich task，亚毫米级 alignment。

你手里有 20 条 human teleoperation demo。你直接 SFT（supervised fine-tuning）训一下，结果发现：

**Pain point 1：human demo 本身就 trash**。你自己用 SpaceMouse 插轮子，手会抖，每条 trajectory 的 action sequence 都不一样，有时候你先对准再下压，有时候你边旋转边下压。NLL loss 会把这些 multi-modal action average 成一个 "mean action"——既不是第一种方式也不是第二种，是个四不像。

**Pain point 2：state coverage 不够**。20 条 demo 覆盖不了 wheel 和 slot 的所有相对位置。robot 真实执行时遇到 demo 没见过的 state，policy 直接懵掉。

**Pain point 3：SFT 学不到效率**。demo 里你插轮子花了 60 步，其中前 20 步在试探、中间 30 步在调整、最后 10 步才真正插进去。SFT 会让 policy 完美复刻这个低效过程，它学不会 "直接对准下压 15 步搞定"。

RL 理论上能解决 2 和 3（探索新 state + 优化 reward），但直接用 PPO from scratch 训 robot？早期 exploration 会把 robot 撞坏、把 wheel 甩飞。HIL-SERL（https://hil-serl.github.io/）试过从 scratch 训，48 分钟才到 31.9% success rate，还经常搞 destructive behavior。

ConRFT 的方案：**VLA 当 starting point（省去 exploration 的前期痛苦）+ offline Cal-QL 热身（建立 Q-function 和 policy 的稳定 init）+ online HIL RL（real-world 边探索边学，human 兜底 safety）+ consistency policy 当 action head（多模态 action + 1-4 步推理 real-time）**。

---

## Consistency Policy 到底是个啥

先从 Diffusion Policy（https://diffusion-policy.cs.columbia.edu/）讲起。Diffusion Policy 的想法是：action distribution 是 multi-modal 的（插轮子有多种成功路径），所以我学一个 diffusion model 来建模这个 distribution。生成 action 时，从纯噪声 $a^K \sim \mathcal{N}(0, K \cdot I)$ 出发，跑 K 步去噪（K 一般 10-100），每步去掉一点噪声，最后得到 clean action。

问题：10-100 步神经网络 forward，在 10Hz control loop 下根本跑不动。100 步神经网络 forward 大概 500ms，robot 早就漂了。

Consistency Policy（https://consistentpolicy.github.io/）的核心 insight：**扩散过程中的任意一步 noisy action $a^k$，都应该能被"一致性映射"直接拉回 clean action $a^0$**。无论你从 $a^{k_1}$ 还是从 $a^{k_2}$ 出发，都应该收敛到同一条 PF-ODE 轨迹上的同一个 clean action。

公式上看：

$$\pi_\psi(a|s) = f_\psi(a^k, k | E_\phi(s))$$

变量解释：
- $f_\psi$：consistency network，一个 2-layer MLP（hidden 256，Mish activation），参数 $\psi$
- $a^k$：当前 diffusion step $k$ 的 noisy action，$a^k \sim \mathcal{N}(0, kI)$，注意 variance 是 $k$ 不是 1，越大的 $k$ 噪声越多
- $k$：diffusion step，告诉 network 当前噪声水平
- $E_\phi(s)$：VLA model（frozen）输出的 state embedding，参数 $\phi$
- 输出：预测的 clean action

训练时，diffusion horizon $[\epsilon, K] = [0.002, 80.0]$ 被切成 $M = 40$ 个 sub-interval。边界点公式：

$$k_i = \left(\epsilon^{1/\rho} + \frac{i-1}{M-1}(T^{1/\rho} - \epsilon^{1/\rho})\right)^\rho, \quad \rho = 7$$

变量解释：
- $i$：sub-interval index，从 1 到 $M = 40$
- $\epsilon = 0.002$：最小 diffusion step，避免数值不稳定（$k = 0$ 时 noise variance 为 0，退化）
- $T = 80.0$：最大 diffusion step，对应纯噪声
- $\rho = 7$：控制 sub-interval 分布的非线性程度

为啥 $\rho = 7$ 这么大？因为 diffusion 在接近 clean data 端（小 $k$）信息密度高、变化剧烈，需要细 discretization；接近纯噪声端（大 $k$）信息稀疏，粗 discretization 就行。$\rho > 1$ 让边界点向 $\epsilon$ 端聚集。这跟 flow matching 里"低噪声端分布变化最剧烈"的观察一致。

推理时只需要 1-4 步去噪（从 $a^K$ 一路映射到 $a^0$），5-20ms latency，10Hz control 够用了。

---

## Offline 阶段：Cal-ConRFT 干啥

你有 20 条 demo，直接 Cal-QL 训 Q-function 会有问题——demo 太少，state coverage 窄，Q-value 估计不准，policy 跟着崩。Paper 里试过纯 Cal-QL，offline 后 success rate 0%。

所以 ConRFT 加了 BC loss 当 anchor。整个 offline loss：

$$\mathcal{L}_\pi^{offline}(\psi) = \beta \mathcal{L}_\pi^{BC} + \eta \mathcal{L}_\pi^{Q}$$

变量：
- $\beta = 1.0$：BC loss weight，offline 阶段主导
- $\eta = 0.1$：Q loss weight，offline 阶段辅助
- $\psi$：consistency network 参数

BC loss 具体形式：

$$\mathcal{L}_\pi^{BC} = \mathbb{E}_{(s,a) \sim \mathcal{D}, m \sim \mathcal{U}[1, M-1]}\left[d\left(f_\psi(a + k_m z, k_m | E(s)), a\right)\right]$$

变量解释：
- $(s, a)$：从 demo buffer $\mathcal{D}$ 采样
- $m$：从 $[1, M-1]$ 均匀采样的 sub-interval index，注意不包括 $M$（纯噪声端约束太弱）
- $z \sim \mathcal{N}(0, I)$：standard Gaussian
- $a + k_m z$：给 clean action $a$ 加上 step $k_m$ 对应的噪声，模拟 diffusion 过程的中间状态
- $d(x, y) = \|x - y\|_2$：Euclidean distance
- 整体含义：无论从哪个 diffusion step 开始，consistency network 都应该恢复到 demo 里的真实 action $a$

Q loss 具体形式：

$$\mathcal{L}_\pi^Q = -\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\psi}[Q(s, a)]$$

变量解释：
- 从当前 policy $\pi_\psi$ 采样 action $a$
- 最大化其 Q-value，等价于最小化负 Q
- 直觉：鼓励 policy 往 Q-value 高的 action 移动

Critic 用 Cal-QL（https://arxiv.org/abs/2303.05479）训：

$$\mathcal{L}_Q^{offline}(\theta) = \alpha\left(\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(\cdot|s)}[\max(Q_\theta(s,a), V^\mu(s))] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q_\theta(s,a)]\right) + \frac{1}{2}\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[(Q_\theta(s,a) - B^\pi \overline{Q}_{\bar\theta}(s,a))^2\right]$$

变量解释：
- $\theta$：Q-network 参数
- $\bar\theta$：target Q-network 参数（EMA 更新，稳定训练）
- $\alpha = 0.01$：conservative penalty 系数
- $V^\mu(s)$：reference policy（demo behavior policy）的 value，作为下限
- $B^\pi \overline{Q}(s,a) = r(s,a) + \gamma \mathbb{E}_{a' \sim \pi(\cdot|s')}[\overline{Q}(s', a')]$：Bellman backup
- $\gamma$：discount factor

第一项是 conservative regularization：对 policy 采样的 action（可能 OOD）Q-value 做惩罚（取 max 保证不低于 $V^\mu$），对 demo 里真实出现的 action Q-value 做 compensate，两者差值压低 OOD action 的 Q 估计，防止 Q-function 对没见过的 action 过度乐观。

第二项是 standard Bellman error，就是 TD learning。

为啥 offline 阶段 $\beta = 1.0, \eta = 0.1$？因为 20 条 demo 下 Q-function 还不可靠，主要靠 BC 学 policy，Q 只起 stabilization 作用。这跟小孩学走路先模仿大人、再自己探索优化是一个道理。

---

## Online 阶段：HIL-ConRFT 怎么搞

Offline 后 policy 大概 39% success rate（跟 SFT 持平），但关键在于 Q-function 已经有了一个 stable init。现在放到 real robot 上 online fine-tune。

Online loss 形式跟 offline 一样：

$$\mathcal{L}_\pi^{online}(\psi) = \beta \mathcal{L}_\pi^{BC} + \eta \mathcal{L}_\pi^{Q}$$

但 weight 变了：$\beta = 0.5, \eta = 1.0$，Q loss 现在主导，BC loss 退居二线当 anchor。

Critic loss 退化成 standard TD（去掉 conservative regularizer）：

$$\mathcal{L}_Q^{online}(\theta) = \mathbb{E}_{(s,a,s') \sim (\mathcal{D} \cup \mathcal{R})}\left[(Q_\theta(s,a) - B^\pi \overline{Q}(s,a))^2\right]$$

变量解释：
- $\mathcal{D}$：demo buffer，offline 阶段的 20 条 demo + online 阶段 human intervention 的新数据
- $\mathcal{R}$：replay buffer，policy 自己 rollout 的 transition
- $\cup$：两个 buffer 合并采样

为啥 online 阶段不需要 conservative regularizer？因为 online data 持续进入 $\mathcal{R}$，distribution shift 大大缓解，Q-function 不再对 OOD action 过度乐观。

Symmetric sampling：每个 batch 一半从 $\mathcal{D}$ 采，一半从 $\mathcal{R}$ 采。这保证 demo 数据不遗忘，policy 不会漂移到 unsafe region。

**Human-in-the-Loop 机制**：robot 执行 policy 时，human 拿 SpaceMouse 盯着。一旦发现 policy 要搞 destructive behavior（撞 obstacle、gripper 卡死、用力过猛），human 立即接管，teleop 完成 task。这些 intervention 数据 $(s, a_{intv}, r, s')$ 被加入 demo buffer $\mathcal{D}$（注意是 $\mathcal{D}$ 不是 $\mathcal{R}$），当作 high-quality guidance。

这个设计很关键：
1. Human intervention 直接提供 corrective action，policy 能学到 "从 bad state 怎么 recover"
2. 加入 $\mathcal{D}$ 意味着 BC loss 会 anchor 到这些 human action，防止 policy 重蹈覆辙
3. Intervention rate 随训练下降（Fig. 3），说明 policy 逐渐 internalize human guidance

---

## 实验数据背后的故事

### 主结果（Table I）

| Method | Avg Success Rate | Episode Length | Online Training Time |
|--------|------------------|----------------|----------------------|
| SFT (offline baseline) | 39.4% | 59.9 | - |
| Cal-ConRFT (offline) | 39.4% | 57.5 | - |
| HG-DAgger (online) | 65.0% (+65%) | 56.3 (1.1x) | 48.8 min |
| PA-RL (online) | 71.3% (+81%) | 51.1 (1.2x) | 48.8 min |
| **HIL-ConRFT (online)** | **96.3% (+144%)** | **30.7 (1.9x)** | 48.8 min |

几个有意思的点：

**Cal-ConRFT offline 跟 SFT success rate 一样（39.4%）**，那 offline 阶段加 Q loss 有啥用？看 Fig. 4 就懂了——从 SFT 起始的 online fine-tuning，早期 intervention rate 飙到 80%+（policy forgetting，啥都不会了）；从 Cal-ConRFT 起始的 online fine-tuning，intervention rate 平稳下降。Q loss 在 offline 阶段的价值是为 online stage 提供稳定 value init，让 policy update 有方向感。

**Episode length 1.9x 缩短**是 RL 的标志性优势。SFT 只能复刻 demo 效率，demo 里插轮子 60 步，policy 就学 60 步。RL 直接优化 discounted return，policy 发现 "直接对准下压 15 步搞定" reward 更高，自然学会更高效路径。Hang Chinese Knot task 最明显：SFT 52.6 步，HIL-ConRFT 26.8 步。

**Contact-rich task 的分化**：Insert Wheel 上 HG-DAgger 40%（比 Cal-ConRFT 35% 还好一点点），PA-RL 30%（倒退），HIL-ConRFT 80%。为啥 HG-DAgger 和 PA-RL 在 contact-rich 上拉胯？

- HG-DAgger 靠 human correction 做 SFT，但不同人插轮子的角度、力度都不一样，这些 correction 本身 noisy、inconsistent，introduce 冲突信号，policy 学不出 precise dexterous behavior
- PA-RL 用 policy-agnostic Q-function（Cal-QL 训），但 demo buffer + replay buffer state coverage 不够，Q-function 无法 generalize 到 wheel-slot 不同相对位置，导致 action optimization 没方向

HIL-ConRFT 为啥能赢？因为它直接用 task-specific reward 优化 policy，不依赖 human correction 的一致性；同时 BC loss + symmetric sampling 保证 policy 不漂移；HIL 提供 safety net 让 exploration 可以大胆尝试。

### Train from scratch vs Fine-tune VLA（Table II）

HIL-SERL（https://hil-serl.github.io/）从 scratch 训 RL，相同 48.8 min 只到 31.9%，且需要 >2 hours 才能收敛。这说明 VLA pre-training 的价值：它已经从 Open X-Embodiment 学到 general visual-language grounding，省去 RL 从 scratch 探索的痛苦前期。Fine-tune VLA 类似于 transfer learning，站在巨人肩膀上。

### Demo 数量对比（Table III）

| Method | Demo 数量 | Avg Success Rate |
|--------|-----------|------------------|
| Diffusion Policy | 150 human demo | 41.7% |
| SFT (Octo) | 150 human demo | 58.3% |
| RLDG | 150 RL-collected demo | 83.3% |
| HIL-ConRFT | 20 human demo + 80-120 online traj | 93.3% |

这个对比很 punchy：用 7.5 倍 human demo 训 supervised method，还不如 ConRFT 用 20 条 demo + online RL。RLDG 用 RL policy 收集的 "optimal" demo 训 SFT，83.3%，仍不如 ConRFT 直接 online RL fine-tune。

直觉解释：demo 数据质量 > 数据数量，但 online RL 直接优化 > 用 RL 改善 demo 再 SFT。因为 SFT 本质是 likelihood maximization，无法优化 reward；online RL 直接对 reward 做 policy gradient（这里是通过 Q-function），能发现 demo 里没有的更优 action。

### 跨 VLA backbone（Table IV）

在 RoboVLM（https://generalist-robot.github.io/）上用 Kosmos-2 (1.6B) 和 PaliGemma (3B) 两个 backbone 测，ConRFT 都从 ~50% 提到 100%。说明方法 model-agnostic，只要 VLA 有 frozen visual encoder + transformer backbone + 可 fine-tune 的 action head 就行。

---

## 一些细节 intuition

### 1. 为啥 BC loss 在 online 阶段不彻底去掉

Paper 在 online 阶段保留 $\beta = 0.5$ 的 BC loss，理由有二：

**Reason 1**：防止 policy 漂移太远。RL 探索高维 state-action space 容易 unstable，policy 可能突然崩溃。BC loss anchor 到 demo distribution，保证 policy 不会偏离到 unsafe region。

**Reason 2**：Contact-rich task 需要精确控制，RL reward 信号 sparse（只有 +10 完成 / -0.05 per step），policy 难以学到 precise action。BC loss 提供 dense supervisory signal，保持 action quality。

这跟 RLHF 里 KL penalty to reference policy 的逻辑一样：防止 PPO 把 LLM 推到 "reward 高但 language 不自然" 的 weird distribution。

### 2. Sub-interval $\rho = 7$ 的直觉

$\rho$ 控制 sub-interval 边界的非线性分布。$\rho = 1$ 是均匀分布，$\rho > 1$ 边界向 $\epsilon$ 端聚集。

为啥要向 $\epsilon$ 端（小 $k$，接近 clean data）聚集？因为 diffusion 过程在低噪声端信息密度高——大部分 distribution 的精细结构都在 clean data 附近，需要细 discretization 捕获；高噪声端（大 $k$）就是近似纯噪声，粗 discretization 足够。

这跟 score-based model 里 "score function 在高噪声端近似线性、低噪声端高度非线性" 的观察一致。$\rho = 7$ 是经验调出来的，paper 没做 ablation，但理论上应该跟 action distribution 的复杂度相关。

### 3. Frozen backbone 的 trade-off

Paper 只 fine-tune consistency policy（action head），visual encoder 和 transformer backbone 都 frozen。原因：

**Real-time constraint**：Online RL 需要高频 interaction（10Hz），fine-tune 整个 VLA 的 backward pass 太慢，GPU memory 也吃不消。

**Sample efficiency**：Fine-tune 大 backbone 需要更多数据，20 条 demo + 80-120 online trajectory 远远不够。

**Limitation**：Frozen backbone 的 visual representation 对新 task 可能不够 discriminative。比如 Insert Wheel 的 wheel-slot 细微纹理，frozen encoder 可能提取不出足够精细的 feature。Paper 在 limitation 里提到未来可以用 LoRA（https://arxiv.org/abs/2106.09685）做 partial fine-tuning。

### 4. Reward classifier 的脆弱性

Paper 用 binary classifier 给 reward（+10 完成，-0.05 per step）。这有两个 risk：

**Reward hacking**：Classifier 在 OOD state 上可能误判，policy 学到 "欺骗 classifier" 的行为。比如 robot 把 end-effector 摆到某个特定位置触发 false positive，policy 收敛到错误 behavior。

**Sparse reward**：只有 terminal +10 和 per-step -0.05，long-horizon task 学习效率低。Human intervention 部分缓解（提供 corrective demo），但根本问题没解决。

未来方向：用 VLM 当 dense reward model（https://arxiv.org/abs/2312.09386），或者 self-supervised reward shaping。

---

## 跟 RLHF 的类比

把 ConRFT 跟 LLM RLHF 对照看会非常清楚：

| LLM RLHF | ConRFT VLA |
|----------|------------|
| SFT on instruction data | SFT on demonstration |
| Reward model from preference | Binary success classifier |
| KL penalty to reference policy | BC loss to demonstration |
| PPO on-policy RL | Off-policy Q-learning + consistency |
| Human annotator ranking | Human teleoperation intervention |
| Synthetic environment (token sampling) | Real-world physical interaction |
| Sample efficiency 次要 (synthetic cheap) | Sample efficiency 关键 (real-world expensive) |
| Safety 次要 (weird text 无害) | Safety 关键 (robot 撞坏昂贵) |

ConRFT 本质是把 RLHF pipeline 移植到 VLA，但针对 physical interaction 特殊性做了三个关键调整：
1. Off-policy Q-learning 替代 on-policy PPO（提升 sample efficiency）
2. Consistency policy 替代 direct token sampling（处理 continuous multi-modal action + real-time inference）
3. HIL 替代 preference labeling（提供 corrective action 而非 ranking，同时充当 safety net）

---

## 可能的改进方向

### 1. Adaptive weight annealing

当前 $\beta, \eta$ 是手动 schedule（offline 1.0/0.1 → online 0.5/1.0）。更优雅的做法是根据 Q-function 的 TD error 或 uncertainty 自动调整。Q 可靠时 $\eta$ 大，Q 不可靠时 $\beta$ 大。类似 UCRL 或 Bayesian RL 的思路。

### 2. Dense reward via VLM

用 GPT-4V 或 VLA 本身当 reward model，提供 dense feedback。比如 "gripper 离 wheel slot 越近 reward 越高"，而非只有 terminal +10。这能大幅加速 contact-rich task 学习。

### 3. Multi-task ConRFT

当前每 task 训一个 policy。未来可探索 multi-task RL with shared VLA backbone，类似 MT-Opt（https://mt-opt.github.io/）。VLA 的 language conditioning 天然支持 multi-task，只需要 reward function 多任务化。

### 4. World model integration

结合 world model（如 Dreamer 系列，https://danijar.com/project/dreamer/）在 latent space 做 rollout，减少 real-world interaction。这对 safety 和 sample efficiency 都有帮助。

### 5. Hierarchical ConRFT

High-level VLA 做 task planning（"先抓 wheel，再对准 slot，再下压"），low-level CP 做 fine-grained control。类似 RT-2（https://robotics-transformer2.github.io/）+ Diffusion Policy 的组合，但加 RL fine-tuning。

---

## 核心直觉提炼

1. **VLA pre-training 省 exploration**：Open X-Embodiment 上预训练的 VLA 已经有 general visual-language grounding，fine-tune 只需适应新 task，省去 RL from scratch 的痛苦前期。
2. **Offline Cal-QL + BC 提供稳定 init**：20 条 demo 不够训 Cal-QL，加 BC loss anchor policy，同时 Q-function 有 stable init 为 online stage 铺路。
3. **Online HIL RL 直接优化 reward**：BC loss 只能复刻 demo 效率，RL 能发现更优 action，episode length 1.9x 缩短就来自这。
4. **Consistency policy 兼顾 multi-modal 和 real-time**：Diffusion Policy 多模态表达强但推理慢，CP 1-4 步去噪满足 10Hz control。
5. **HIL 当 safety net + corrective demo**：Human intervention 既防止 destructive behavior，又提供 recovery guidance，intervention rate 随训练下降说明 policy 逐渐内化。

---

参考链接汇总：
- ConRFT: https://cccedric.github.io/conrft/
- Cal-QL: https://arxiv.org/abs/2303.05479
- Consistency Policy: https://consistentpolicy.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Octo: https://octo-models.github.io/
- HIL-SERL: https://hil-serl.github.io/
- HG-DAgger: https://arxiv.org/abs/1902.02088
- PA-RL: https://arxiv.org/abs/2412.06685
- RLDG: https://arxiv.org/abs/2412.09858
- RoboVLM: https://generalist-robot.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- RT-2: https://robotics-transformer2.github.io/
- π₀: https://www.physicalintelligence.company/blog/pi0
- LoRA: https://arxiv.org/abs/2106.09685
- Dreamer: https://danijar.com/project/dreamer/
- MT-Opt: https://mt-opt.github.io/
- VLM-RM: https://arxiv.org/abs/2312.09386

---

# ConRFT: 深度技术解析

## 1. 核心动机与问题定位

这篇paper解决的核心问题是 **VLA model 在 real-world contact-rich manipulation task 上的 fine-tuning 困境**。Supervised Fine-Tuning (SFT) 依赖的 demonstration 数据存在三个根本性缺陷：

- **Sub-optimality**: 人类 teleoperation 收集的 trajectory 本身不是最优的，尤其 contact-rich task 如 Insert Wheel 需要亚毫米级 precision，人类操作会引入抖动和不一致 force
- **Inconsistency**: 不同 demonstration 之间 action distribution 存在 multi-modal 特性，NLL 或 MSE loss 会 average out 这些 mode，导致 policy 学到 "mean action" 而非正确 action
- **Limited state coverage**: 仅 20-30 条 demonstration 无法覆盖 MDP 的 state space，SFT policy 在 OOD state 下表现灾难性下降

作者观察到的 key insight 是：LLM 领域 RLHF 的成功启发了 RL-based fine-tuning 的潜力，但 VLA 面临 unique challenge——**real-world physical interaction 的 safety 和 sample efficiency 要求远高于 LLM 在 synthetic environment 中的探索**。直接把 PPO 等 on-policy RL 套到 VLA 上不可行，因为 sample efficiency 太低，且早期 exploration 阶段会产生 destructive behavior（撞坏 robot、损坏 object）。

参考链接：
- Octo: https://octo-models.github.io/
- RT-2: https://robotics-transformer2.github.io/
- π₀ (pi-zero): https://www.physicalintelligence.company/blog/pi0

## 2. 方法架构：两阶段 unified consistency-based objective

ConRFT 的核心设计哲学是 **"offline-to-online pipeline with shared training objective"**。这区别于传统 offline-to-online 方法（如 AWAC, Cal-QL）的地方在于：offline 和 online 阶段使用完全相同的 loss structure（Equation 3 和 Equation 5 形式一致），只是数据来源和 weight 配比不同。这种设计避免了 offline→online transition 时的 catastrophic forgetting，也减少了 hyper-parameter tuning burden。

### 2.1 为什么选择 Consistency Policy 作为 action head

Consistency Policy (CP) 来源于 consistency model 在 diffusion 上的加速蒸馏思想（参考 Consistency Policy: https://consistentpolicy.github.io/）。Diffusion Policy (DP, https://diffusion-policy.cs.columbia.edu/) 虽然在 multi-modal action distribution 上表现优秀，但需要 10-100 步去噪迭代，inference latency 太高不适合 10Hz real-time control。CP 通过一致性约束将 diffusion 过程压缩到 1-4 步。

具体地，给定 diffusion horizon $[\epsilon, K]$（这里 $\epsilon = 0.002$ 避免数值不稳定, $K = 80.0$），将其 discretize 为 $M = 40$ 个 sub-interval，边界点由公式决定：

$$k_i = \left(\epsilon^{\frac{1}{\rho}} + \frac{i-1}{M-1}(T^{\frac{1}{\rho}} - \epsilon^{\frac{1}{\rho}})\right)^{\rho}$$

其中 $\rho = 7$ 控制 sub-interval 间距的分布——靠近 $\epsilon$ 端（对应原 clean action）的 sub-interval 更密集，靠近 $K$ 端（对应纯噪声）的更稀疏。这种非均匀 discretization 反映了 diffusion 过程中信息密度的不对称性：接近 clean data 的步骤对最终 action quality 影响更大。

Consistency policy 的 forward pass 公式：

$$\pi_\psi(a|s) = f_\psi(a^k, k | E_\phi(s))$$

变量解释：
- $f_\psi$: consistency network，参数 $\psi$，paper 中用 2-layer MLP，hidden size 256，Mish activation
- $a^k \sim \mathcal{N}(0, kI)$: 第 $k$ 步的 noisy action，$k$ 是 diffusion timestep
- $E_\phi(s)$: VLA model 的 state encoding（visual encoder + transformer backbone 的 output），参数 $\phi$ 在 fine-tuning 中 frozen
- 输出: 被映射回 clean action space 的预测 action

直觉理解：$f_\psi$ 学习一个 "self-consistency" 函数，要求任意 diffusion step $k$ 上的 noisy action $a^k$ 都能被映射到同一条 PF-ODE (Probability Flow Ordinary Differential Equation) 轨迹上的 clean action。这意味着无论你从哪个噪声水平开始，都应该收敛到相同的 expert action。

### 2.2 Stage I: Cal-ConRFT (Offline)

Offline stage 的核心 loss 由两部分组成：

#### 2.2.1 Critic loss (Cal-QL)

$$\mathcal{L}_Q^{offline}(\theta) = \alpha\left(\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(\cdot|s)}[\max(Q_\theta(s,a), V^\mu(s))] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q_\theta(s,a)]\right) + \frac{1}{2}\mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[(Q_\theta(s,a) - B^\pi \overline{Q}_{\bar\theta}(s,a))^2\right]$$

这个公式有两项，逐项拆解：

**第一项（conservative regularization）**：
- $\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(\cdot|s)}[\max(Q_\theta(s,a), V^\mu(s))]$: 对 policy 采样的 action（可能是 OOD action）的 Q-value 进行 penalize，取 $\max$ 确保不低于 reference policy $V^\mu(s)$ 的 value
- $\mathbb{E}_{s,a \sim \mathcal{D}}[Q_\theta(s,a)]$: 对 demonstration 中真实出现的 action 的 Q-value 进行 compensate（拉高）
- 两者差值乘以 $\alpha$（conservative penalty 系数，offline 阶段 = 0.01）迫使 OOD action 的 Q-value 低于 in-distribution action 的 Q-value，防止 Q-function 对未见 action 给出过高估计

**第二项（standard Bellman error）**：
- $B^\pi \overline{Q}_{\bar\theta}(s,a) = r(s,a) + \gamma \mathbb{E}_{a' \sim \pi(\cdot|s')}[\overline{Q}(s',a')]$: Bellman backup operator，target 用 delayed target network $\overline{Q}_{\bar\theta}$（EMA 更新）保证训练稳定
- $\gamma$: discount factor
- 这一项就是标准 TD learning

Cal-QL 相对 CQL 的关键改进是 **"calibrated" initialization**——保证初始 Q-value 不低于 behavior policy 的 value $V^\mu(s)$，避免 Q-value 被 over-penalize 到 0，从而保留 demo 中有用信号。Paper "Cal-QL: Calibrated offline RL pre-training for efficient online fine-tuning" (https://arxiv.org/abs/2303.05479) 有详细论述。

#### 2.2.2 Policy loss (consistency BC + Q)

$$\mathcal{L}_\pi^{offline}(\psi) = \beta \mathcal{L}_\pi^{BC} + \eta \mathcal{L}_\pi^{Q}$$

其中：

**BC loss**:
$$\mathcal{L}_\pi^{BC} = \mathbb{E}_{(s,a) \sim \mathcal{D}, m \sim \mathcal{U}[1, M-1]}[d(f_\psi(a + k_m z, k_m | E(s)), a)]$$

- $(s,a)$ 从 demonstration dataset $\mathcal{D}$ 采样
- $m$ 从 $[1, M-1]$ 均匀采样（注意不包括 $M$，因为 $k_M = K$ 对应纯噪声端，约束太弱）
- $z \sim \mathcal{N}(0, I)$: standard Gaussian noise
- $a + k_m z$: 给 clean action $a$ 加上对应 diffusion step 的噪声
- $d(\cdot, \cdot) = \|\cdot - \cdot\|_2$: Euclidean distance
- 直觉：无论从哪个 diffusion step 开始 denoise，consistency network 都应该恢复到 demonstration 的真实 action $a$

**Q loss**:
$$\mathcal{L}_\pi^Q = -\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\psi}[Q(s,a)]$$

- 从当前 policy $\pi_\psi$ 采样 action，最大化其 Q-value
- 这是 advantage-weighted 形式的简化版，鼓励 policy 朝高 value action 移动

$\beta$ 和 $\eta$ 的 offline 配置是 $(1.0, 0.1)$，意味着 BC loss 占主导，Q loss 仅作为轻微的 value-guided correction。这个 ratio 反映了 offline 阶段数据稀缺时 Q-function 还不可靠的现实——Q-function 此时主要起 stabilization 作用，policy improvement 主要靠 BC。

### 2.3 Stage II: HIL-ConRFT (Online)

Online stage 的 critic loss 退化为标准 TD loss（去掉 conservative regularizer）：

$$\mathcal{L}_Q^{online}(\theta) = \mathbb{E}_{(s,a,s') \sim (\mathcal{D} \cup \mathcal{R})}[(Q_\theta(s,a) - B^\pi \overline{Q}(s,a))^2]$$

这是因为 online 数据持续加入 replay buffer $\mathcal{R}$，distribution shift 问题大大缓解，不再需要 conservative penalty。Data 采样采用 symmetric sampling：每个 batch 一半从 demo buffer $\mathcal{D}$ 采样，一半从 replay buffer $\mathcal{R}$ 采样，保证 demo 数据不遗忘。

Policy loss 形式不变，但 weight 配比变为 $\beta = 0.5, \eta = 1.0$，即 Q loss 占主导，BC loss 仅作为 anchor 防止 policy 漂移过远。这种 annealing 思路与 RLHF 中 KL penalty 的作用类似——防止 policy 过度偏离 reference（这里是 demo distribution）导致 unsafe behavior。

**Human-in-the-Loop (HIL) 机制**：当 robot 执行 policy 时，human operator 通过 SpaceMouse 监控，一旦发现 unsafe 或 unrecoverable 状态（如撞向 obstacle、gripper 卡死），立即接管并 teleop 完成 task。这些 human intervention 数据 $(s, a_{intv}, r, s')$ 被加入 demo buffer $\mathcal{D}$ 而非 replay buffer $\mathcal{R}$，因为它们被视为 "high-quality guidance"。

参考 HIL-SERL: https://hil-serl.github.io/

## 3. 实验数据深度分析

### 3.1 主结果（Table I）

| Method | Avg Success Rate | Avg Episode Length | Training Time |
|--------|------------------|---------------------|----------------|
| SFT (baseline) | 39.4% | 59.9 | - |
| Cal-ConRFT (offline only) | 39.4% | 57.5 | - |
| HG-DAgger | 65.0% (+65%) | 56.3 (1.1x shorter) | 48.8 min |
| PA-RL | 71.3% (+81%) | 51.1 (1.2x shorter) | 48.8 min |
| **HIL-ConRFT** | **96.3% (+144%)** | **30.7 (1.9x shorter)** | 48.8 min |

关键观察：

1. **Cal-ConRFT offline 性能与 SFT 持平（都 39.4%）**，但 online fine-tuning 后 HIL-ConRFT 远超 HG-DAgger 和 PA-RL。这说明 offline 阶段 Q loss 的价值不在于直接提升 offline performance，而在于为 online stage 提供稳定的 value initialization。Fig. 4 显示从 SFT 起始的 online fine-tuning 早期 intervention rate 飙高（policy forgetting），而从 Cal-ConRFT 起始的 intervention rate 平稳下降。

2. **Episode length 大幅缩短（1.9x）**，这是 RL-based method 相对 supervised method 的标志性优势。Supervised method 只能模仿 demo 的效率，而 RL 通过 reward 信号直接优化 discounted return，鼓励 policy 更快完成任务。例如 Hang Chinese Knot task，HIL-ConRFT episode length 仅 26.8 步，而 SFT baseline 52.6 步——policy 学会了更直接的挂 knot 路径。

3. **Contact-rich task 的突破**：Insert Wheel task 是最难的（亚毫米 alignment），HG-DAgger 仅 40% (-14% 比 Cal-ConRFT baseline 还差)，PA-RL 30% (-14%)，HIL-ConRFT 达到 80% (+129%)。HG-DAgger 失败的原因是 human correction 本身不一致（不同人插入角度不同），introduce noise。PA-RL 失败因为 policy-agnostic Q-function 在 contact-rich 场景下 state coverage 不足，无法 generalize 到 wheel-slot 不同相对位置。

### 3.2 与 train-from-scratch 对比（Table II）

HIL-SERL 从 scratch 训练，相同 48.8 min 平均只达到 31.9% success rate，且需要 >2 hours 才能收敛。这验证了 **pre-trained VLA 作为 initialization 的关键价值**——VLA 已经从 Open X-Embodiment 等大规模数据集学到 general visual-language grounding，只需要 fine-tune 即可适应新 task，省去了大量 exploration cost。

### 3.3 Demonstration 数量对比（Table III）

| Method | Demo 数量 | Avg Success Rate |
|--------|-----------|------------------|
| Diffusion Policy (DP) | 150 human demo | 41.7% |
| SFT (Octo) | 150 human demo | 58.3% |
| RLDG | 150 RL-collected demo | 83.3% |
| **HIL-ConRFT** | **20 human demo + 80-120 online trajectory** | **93.3%** |

这个对比非常有说服力：即使用 7.5 倍的 human demo 数据，supervised method 仍不如 ConRFT。RLDG 用 RL policy 收集的 "optimal" demo 表现更好（83.3%），但仍不如 ConRFT 直接 online RL fine-tuning（93.3%）。这说明 **demo 数据质量 > 数据数量，但 online RL 直接优化 > 用 RL 改善 demo 再 SFT**。

### 3.4 跨 VLA backbone 泛化（Table IV）

在 RoboVLM（https://generalist-robot.github.io/）上用 Kosmos-2 (1.6B) 和 PaliGemma (3B) 两个 backbone 测试，ConRFT 都能将 success rate 从 ~50% 提升到 100%，证明方法的 model-agnostic 特性。

## 4. 技术细节补充

### 4.1 Consistency Policy 的 sub-interval 设计

为什么 $\rho = 7$？这个参数控制 sub-interval 边界的非线性分布。当 $\rho > 1$ 时，边界点向 $\epsilon$ 端聚集。直觉上，diffusion 过程在接近 clean data 的步骤（小 $k$）信息密度更高，需要更细的 discretization 来捕获；而纯噪声端（大 $k$）信息稀疏，粗 discretization 足够。这与 flow matching / rectified flow 中"大多数分布变化集中在低噪声端"的观察一致。

### 4.2 Frozen backbone 的设计 trade-off

Paper 在 limitation 部分承认 frozen visual encoder 和 transformer backbone 限制了 policy 对 unseen scenario 的适应。但这是 real-time 性能的必要妥协——fine-tuning 整个 VLA 在 online RL 中需要大量 GPU memory 和 backward pass，难以满足 10Hz control loop。未来方向是 LoRA (https://arxiv.org/abs/2106.09685) 或 adapter-based partial fine-tuning。

### 4.3 Reward 设计的脆弱性

Paper 使用 binary classifier 作为 reward（+10 完成, -0.05 per step）。这种 sparse reward + classifier 的组合存在 reward hacking 风险——classifier 可能对某些 OOD state 误判为 success，导致 policy 学到 "欺骗 classifier" 的行为而非真正完成任务。Reference: Reward Hacking in RL (https://arxiv.org/abs/1609.02116).

## 5. 与相关工作的关联

### 5.1 与 RLHF 的对应关系

| LLM RLHF | ConRFT VLA |
|----------|------------|
| SFT on instruction data | SFT on demonstration |
| Reward model from preference | Binary success classifier |
| KL penalty to reference policy | BC loss to demonstration |
| PPO on-policy RL | Off-policy Q-learning + consistency |
| Human annotator feedback | Human teleoperation intervention |

ConRFT 本质上是把 RLHF 的 pipeline 移植到 VLA，但针对 physical interaction 的特殊性做了关键调整：off-policy 替代 on-policy（提升 sample efficiency），consistency policy 替代 direct token sampling（处理 continuous multi-modal action），HIL 替代 preference labeling（提供 corrective action 而非 ranking）。

### 5.2 与 Offline-to-Online RL 的关系

Cal-QL (https://arxiv.org/abs/2303.05479) 本身就是为 offline-to-online 设计的，核心思想是 calibrated conservative penalty 让 Q-value 在 offline→online transition 时不会 collapse。ConRFT 在此基础上加入 BC loss，解决 small dataset 下 Cal-QL 单独无法学到有效 policy 的问题。

### 5.3 与 Diffusion Policy 的对比

| Property | Diffusion Policy | Consistency Policy |
|----------|------------------|---------------------|
| Denoising steps | 10-100 | 1-4 |
| Inference latency | ~100-500ms | ~5-20ms |
| Multi-modal support | Yes | Yes (via consistency) |
| Training stability | High | Medium (needs careful tuning) |
| Sample efficiency | Lower | Higher (fewer denoising steps) |

ConRFT 选 CP 主要为了 real-time control 的 latency 考虑。

## 6. 个人思考与延伸

### 6.1 Q loss 和 BC loss 的 weight annealing

Offline: $(\beta, \eta) = (1.0, 0.1)$ → Online: $(\beta, \eta) = (0.5, 1.0)$

这种 annealing schedule 反映了一个重要直觉：**offline 阶段 Q-function 不可靠（数据少），主要靠 BC 学；online 阶段 Q-function 逐渐 reliable（数据增多），主要靠 Q 学**。这与 AlphaGo 中 supervised pre-training → RL fine-tuning 的思路一致，也类似 RLHF 中 SFT → RM → PPO 的 staged training。

更精细的设计可能是 **adaptive weight**：根据 Q-function 的 TD error 或 uncertainty 自动调整 $\beta/\eta$ ratio，而非手动 schedule。

### 6.2 Human intervention 的 data 效率

Human intervention 数据被加入 demo buffer $\mathcal{D}$ 而非 replay buffer $\mathcal{R}$，意味着它们被当作 high-quality demo 而非普通 transition。这有几个 implication：

1. **Symmetric sampling 保证 demo 不遗忘**：即使 demo 只有 20 条，每个 batch 都有一半来自 $\mathcal{D}$，确保 BC loss 持续 anchor policy。
2. **Intervention 数据的 reward 问题**：Human 接管时的 action 是 human 的，但 reward 仍然用 classifier 计算，可能产生 misleading signal（human 完成但 classifier 判 fail，或反之）。Paper 没有详细讨论这个问题。
3. **Intervention 频率的下降曲线**：Fig. 3 显示 intervention rate 随训练下降，说明 policy 逐渐 "internalize" human guidance。这种 learning curve 形态与 DAgger 类似。

### 6.3 与 VLA post-training 的未来方向

ConRFT 是 VLA post-training 的早期探索，未来可能的方向：

1. **Dense reward via VLM**: 用 VLM 本身（如 GPT-4V）作为 reward model，提供 dense feedback 而非 sparse binary classifier。Reference: VLM-RM (https://arxiv.org/abs/2312.09386)
2. **Multi-task ConRFT**: 当前每 task 训一个 policy，未来可探索 multi-task RL with shared VLA backbone，类似 MT-Opt (https://mt-opt.github.io/)
3. **World model integration**: 结合 world model（如 Dreamer 系列）减少 real-world interaction，进一步降低 safety risk
4. **Hierarchical ConRFT**: High-level VLA 做 task planning，low-level CP 做 fine-grained control，类似 RT-2 + DP 的组合

### 6.4 Limitation 的深层原因

Paper 提到 frozen backbone 的 limitation，但更深层的问题是 **visual representation 与 action 的 co-adaptation**。Frozen backbone 提供的 $E_\phi(s)$ 可能对新 task 的 visual feature 不够 discriminative（如 wheel slot 的细微纹理），导致 CP 难以学到 precision control。未来 LoRA fine-tuning backbone 可能显著提升 contact-rich task 性能。

Reward engineering 的 limitation 也值得深入。Binary classifier 的 sparse reward 在 long-horizon task 上学习效率低，而 dense reward 需要 hand-crafted shaping 或 learned reward model，两者都有 trade-off。Self-supervised reward（如 time-contrastive learning）可能是折中方案。

---

总结一下构建 intuition 的核心要点：

1. **ConRFT = Cal-QL (offline) + HIL online RL + consistency policy action head**，三者协同解决 VLA fine-tuning 的 sample efficiency、safety、multi-modal action 三个问题。
2. **BC + Q 的 dual loss 设计**让 offline 阶段在 small data 下仍能学到有效 policy 和稳定 Q-function，为 online stage 提供 good initialization。
3. **HIL 机制**让 human 充当 safety net，在 destructive behavior 发生前接管，同时提供 corrective demo，加速 policy convergence。
4. **Consistency policy**作为 action head，兼具 diffusion policy 的 multi-modal 表达能力和 1-4 步推理的 real-time 性能。

参考链接汇总：
- ConRFT 项目主页: https://cccedric.github.io/conrft/
- Cal-QL: https://arxiv.org/abs/2303.05479
- Consistency Policy: https://consistentpolicy.github.io/
- Octo: https://octo-models.github.io/
- HIL-SERL: https://hil-serl.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- RoboVLM: https://generalist-robot.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- π₀: https://www.physicalintelligence.company/blog/pi0
- PA-RL: https://arxiv.org/abs/2412.06685
- RLDG: https://arxiv.org/abs/2412.09858
- HG-DAgger: https://arxiv.org/abs/1902.02088
- LoRA: https://arxiv.org/abs/2106.09685
