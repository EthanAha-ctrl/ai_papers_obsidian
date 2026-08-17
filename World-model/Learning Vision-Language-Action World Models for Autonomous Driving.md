---
source_pdf: Learning Vision-Language-Action World Models for Autonomous Driving.pdf
paper_sha256: 4719e8f3bfc7c4314c06eeee71b0b1aa81341e3c3de8a61b4a07d762f163552d
processed_at: '2026-08-05T14:14:25-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 VLA-World

Andrej，我把那些公式都扔掉，咱就像在咖啡店聊这篇 paper。

---

## 它到底在干啥

开车的核心难题就一句话：**你得知道下一步会怎样，才能决定现在怎么开。**

现在的自动驾驶有两派人，各搞各的：

**VLA 派**：拿个多模态大模型，塞一堆相机画面进去，直接吐方向盘指令。有点像新手司机——眼睛盯着路面，但不怎么预判别人会怎么动，遇到复杂场景容易懵。

**World Model 派**：搞个生成模型，让它"做梦"想象未来几秒的画面。问题在于它只管做梦做得漂不漂亮，不管这个梦对开车有没有用。它可能生成一个很逼真的碰撞画面，然后自己觉得挺满意。

VLA-World 说：**这俩毛病正好互补，干嘛不结婚。**

---

## 它怎么结婚的

流程特别像人开车：

1. **先扫一眼**——看到周围有啥车、啥人、路边界在哪
2. **本能预判**——"我大概 0.5 秒后会到那儿，方向是左转"
3. **想象一下**——"如果我真往左转，0.5 秒后眼前会看到啥画面？"→ 生成一张未来帧
4. **反思**——盯着这张想象出来的画面看："哎，左前方有个行人快进车道了，刚才那个左转计划有点危险"
5. **改主意**——输出一个修正后的 3 秒轨迹

这个"想象→反思"的闭环就是核心。未来帧在这里充当了一个 **visual sketchpad**——把模糊的"未来可能怎样"变成一张具体的图，然后在图上做推理。跟你之前讲 LLM 时提到的 "let it think" 思路一模一样，只不过这里的 thinking 是 visual 的。

---

## 为什么这个思路 work

直觉上很好理解。你光看当前画面直接输出轨迹，信息量不够——你不知道别人会不会动、会不会有鬼探头。但你又没法精确预测所有人的轨迹，太复杂了。

VLA-World 的 trick 是：**用一张生成的未来帧把所有这些不确定性"塌缩"成一个具体的场景假设**。这张图编码了 ego motion、周围车的运动、场景变化——全在里面。然后 model 在这个具体假设上推理，比在抽象层面推理容易多了。

这就像你问"明天天气怎样"——直接预测一个抽象概率分布很难，但先想象一个具体场景（"明天下午 3 点乌云密布下雨"）然后判断这个场景靠不靠谱，就容易多了。

---

## 训练怎么搞

三步走，跟养小孩似的：

**第一步：教它画画**（Visual Pretraining）
拿 50 万张多视角画面训练它，让它学会"给定当前画面 + 一个动作，0.5 秒后这个摄像头会看到啥"。关键改进是让所有 6 个摄像头都能生成，这样左转右转直行都能想象对应的视角。

**第二步：教它开车概念**（SFT）
用 2 万条精心标注的数据，把 perception、prediction、generation、reasoning、action 这些环节串起来学。输出格式固定，用 XML 标签包好。这一步是打地基——没有这步，后面 RL 会瞎跑。

**第三步：教它反思**（GRPO RL）
这一步最有意思。对每个场景，让 model 生成 8 个不同的"思考路径"（有的保守让行，有的激进抢道），然后用一套 **rule-based reward** 打分：

- 格式对不对
- 短期预测准不准
- 生成的 visual token 合不合法
- 动作分类对不对（F1 score）
- 3 秒轨迹准不准 + 加速度平不平滑

然后算 group mean 和 std，比平均好的就 reinforce，比平均差的就抑制。**不需要 value function**，省一大堆内存。

关键 insight：用规则当裁判，不用学一个 reward model。因为 driving 是 safety-critical 的，规则可审计、不会被骗。学出来的 reward model 万一被 hack 了，后果不堪设想。

---

## 实验结果有多强

在 nuScenes 上：

**轨迹规划**：3 秒 horizon 误差 0.45m，碰撞率 0.08%。用同样的 Qwen2-VL-2B backbone，把前作 FSDrive 从 0.28m 压到 0.26m avg。换 7B backbone 直接干到 0.18m——清晰的 scaling law。

**未来帧生成**：FID 9.8，用 128×192 的 autoregressive 小模型打败了 GEM 那种 576×1024 的 diffusion 大模型。这说明 **action-conditioned generation 比纯 unconditional generation 高效得多**——条件信息大幅压缩了需要建模的分布复杂度。

**动作识别**：左转 F1 从 base model 的 22.75% 飙到 74.22%。这是最惊人的数字——说明反思推理让 model 真正理解了"左转意味着什么后果"，而非死记标签。

---

## Ablation 里的关键发现

去掉 SFT 的 RL 直接崩到 0.85m 误差。**SFT 是地基，RL 是装修**，没地基装修白搭。这跟 AlphaDrive 的观察一致。

去掉 reasoning module，误差从 0.30 飙到 0.85。反思是命门，没它整个 pipeline 就退化成 FSDrive。

Reward 里 trajectory 和 action reward 贡献最大。说明 RL 确实能 end-to-end 优化 planning 本身。

---

## 我认为的局限

1. **只 generate 0.5 秒**。真实驾驶需要 3-5 秒 foresight，VLA-World 只生成 0.5 秒然后靠 reasoning "推"出 3 秒。递归生成多帧理论上更强但计算贵。

2. **Open-loop evaluation**。nuScenes 是事后回放检查碰撞率，不是真正闭环。CARLA / nuPlan 闭环验证还缺。

3. **Rule-based reward 的天花板**。简单场景够用，复杂交互（比如博弈、礼让）的 nuance 规则可能 capture 不了。

4. **SFT/RL 阶段似乎只生成单视角**。虽然 pretrain 是 multi-view 的，但推理时想象的是哪个视角没说清楚。完整 360° 一致性还得验证。

---

## 最核心的一句

这篇 paper 最重要的贡献，我觉得是那个梯度公式（公式 18）——它严格表达了 "world model 应该被 decision reward 一起优化" 这个直觉。过去 world model 只管 reconstruction loss，跟 driving safety 脱节；VLA-World 让 generation 的 gradient 也来自 driving reward，这样生成的未来帧会天然 "highlight 风险"——不是随机做梦，是做对决策有用的梦。

Visual sketchpad 的思路特别 general，我预期会扩展到 robotics、embodied AI。想象一下机器人抓取：先想象抓下去会怎样，再反思这个想象合不合理，最后决定怎么抓。同一个 pattern。

---

# VLA-World 深度讲解

Andrej，这篇 paper 我读了之后觉得非常有意思，它的核心思想其实在某种意义上呼应了你之前在 Eureka Labs 和 YouTube 上讲过的 LLM 的 "System 1 vs System 2" 思考——把直觉的快速想象（world model 的未来帧生成）和缓慢的反思推理（VLA 的 chain-of-thought）结合起来。下面我尽量把每个技术细节都拆解开来讲。

---

## 1. Motivation：为什么 VLA 和 World Model 各自都不够

Paper 的起点是对当前 end-to-end autonomous driving 两大 paradigm 的诊断：

**VLA paradigm**（如 DriveVLM [53], OmniDrive [57], FSDrive [74]）:
- 学习一个直接映射 π_θ(τ_{t:t+H} | o_{1:t}, g)
- 优点：end-to-end、scalable、有 language reasoning
- 缺陷：**没有显式的 spatiotemporal dynamics 建模**，对其他 dynamic agents 的运动预测缺失，缺乏 world consistency

**World Model paradigm**（如 DriveDreamer [59], Drive-WM [61], OccWorld [78]）:
- 学习一个 transition p_ψ(w_{t+1} | w_t, a_t)，可以生成未来帧
- 优点：能"做梦"、能 anticipate
- 缺陷：只学 prior 分布做采样，**没有因果关系的理解**；pixel fidelity 高 ≠ driving safety 好

这种诊断其实在更广义的 LLM community 里也有共鸣——Janus [14, 63]、Show-o [65]、WorldVLA [11] 这类 unified understanding + generation 工作都在解决类似问题。VLA-World 把这个思路用到 driving 上，并加入了一个关键 insight：**短期预测的未来帧天然编码了丰富的 spatiotemporal 信息**，包括 ego motion 和周围 agents 的行为。

人类开车的类比（paper 里用的）非常 eloquent：
- 开放道路巡航 → System 1 直觉想象（world model）
- 行人突然闯入 → System 2 反思推理（VLA）

---

## 2. 核心数学 Formulation

### 2.1 联合分布的分解（公式 2，最关键）

$$p(\tau_{t:t+H}, x_{t+1} | o_{1:t}, g) = \underbrace{p(\tau_{t:t+H} | o_{1:t}, g)}_{\text{decision/policy}} \cdot \underbrace{p(x_{t+1} | o_{1:t}, \tau_{t+1})}_{\text{imagination/world model}}$$

**变量解释**：
- $\tau_{t:t+H} = \{p_{t+1}, ..., p_{t+H}\}$：未来轨迹，$p_{t+h} \in \mathbb{R}^2$ 是 BEV ego-centric 坐标下的 waypoint，$H$ 是 horizon（通常 6，对应 3 秒 @ 0.5s 间隔）
- $x_{t+1}$：next-frame image（某个 camera view 的）
- $o_{1:t}$：observation history，包含 multi-view images $I_t^{1:K}$ 和 ego status $S_t \in \mathbb{R}^{d_s}$（velocity, acceleration, yaw rate 等 CAN signals）
- $g$：mission goal（left/right/forward）

**关键 intuition**：纯 VLA 只管左边的 policy factor（把 $x_{t+1}$ marginalize 掉），纯 World Model 只管右边的 imagination factor。VLA-World 把两者乘起来，并加入反思闭环。

### 2.2 三步生成-推理 pipeline（公式 3, 4）

**Step 1 — 短期预测 + 想象**:
$$\hat{x}_{t+1} \sim p_\psi(x_{t+1} | o_{1:t}, \hat{\tau}_{t:t+1})$$

先用 short-term trajectory $\hat{\tau}_{t:t+1}$（0.5s）条件生成未来帧。

**Step 2 — 反思推理**:
$$\tilde{\tau}_{t:t+H} = f_{\text{ref}}(o_{1:t}, \hat{x}_{t+1}, \hat{\tau}_{t:t+1})$$

$f_{\text{ref}}$ 是 reflective reasoning module，在 $\hat{x}_{t+1}$ 上做因果解读，refine 出最终的 3s 轨迹 $\tilde{\tau}$。

**这个设计的妙处**：future image 充当了一个 "visual sketchpad"——把高维的未来采样到一个 trust-worthy 空间，然后在上面做 System-2 推理。这跟 OpenAI o1 / DeepSeek-R1 的 "thinking tokens" 思想类似，只不过这里的 thinking tokens 是 **visual** 的。

参考链接：
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- FSDrive (前身): https://arxiv.org/abs/2505.17685

---

## 3. 三阶段训练 Pipeline

这是 paper 的另一个核心贡献。整体流程：
```
Stage 1: Visual Pretraining  →  学 "怎么生成"
Stage 2: SFT (multi-task)    →  学 "driving 概念"
Stage 3: GRPO RL             →  学 "反思推理"
```

### 3.1 Stage 1: Visual Pretraining（公式 5）

跟 FSDrive [74] 不同，VLA-World 强调 **multi-view consistency**——FSDrive 只生成 front view，VLA-World 能为任意 camera view 生成未来帧。

**自回归 next-token prediction**：
$$P(Q_{t+1}^k) = \prod_{i=1}^{N} P_\theta(q_i^k | q_{<i}^k, h_t, L)$$

**变量**：
- $Q_{t+1}^k$：camera $k$ 的 next-frame 的 visual token 序列
- $q_i^k$：第 $i$ 个 discrete token，来自 VQGAN [18, 55] codebook
- $h_t = f_\phi(I_t, S_t)$：当前 multi-view + ego status 的 encoded context
- $L$：instruction（例如 "generate CAM_FRONT_LEFT 0.5s later"）
- $N$：token 序列长度

生成出来的 tokens 经过 VQGAN decoder 还原成 $\hat{I}_{t+1}^k$。

**为什么 multi-view 重要**：在 SFT/RL 阶段，planner 预测的 short-term trajectory 可能是左转、右转或直行，对应不同 camera view 的未来。如果只 pretrain front view，下游就不能合理想象 "如果左转会看到什么"。这个设计确保了 generation 的 goal-conditioned 能力。

参考：
- VQGAN (Esser et al.): https://arxiv.org/abs/2012.09841
- Qwen2-VL: https://arxiv.org/abs/2409.12191

### 3.2 Stage 2: SFT — Multi-task Mixed Dataset

SFT 把 6 个 module 串起来，形成完整的 perception→prediction→imagination→reflection→action loop：

| Module | 输入 | 输出 |
|---|---|---|
| **Perception** | 6 camera views + ego status | 3D detection, road shoulder distance, drivable area |
| **Short-term Prediction** | perception results + ego history | next 0.5s waypoint + direction |
| **Generation** | scene context + $\hat{\tau}_{t+1}$ | future frame $\hat{x}_{t+1}$ (visual tokens) |
| **Thinking** | $\hat{x}_{t+1}$ + context | risk assessment, safety margin |
| **Action** | think output | high-level maneuver (left/right/forward × keep/acc/dec/stop) |
| **Trajectory** | action + think | 3s waypoints $\tilde{\tau}_{t:t+H}$ |

数据集是作者自己 curate 的 **nuScenes-GR-20K**（GR = Generative Reasoning），20K samples 专门用于 generation + reasoning。预训练 dataset 大约 500K samples。

**Output 结构（用 XML-like tags）**:
```
<Perception> ... </Perception>
<Prediction> ... </Prediction>
<Visual> [visual tokens] </Visual>
<Think> ... </Think>
<Action> ... </Action>
<Answer> [waypoints] </Answer>
```

这种结构化输出对 RL 阶段的 reward 评估至关重要。

### 3.3 Stage 3: GRPO RL

这是 paper 里最有意思的部分之一，借鉴了 DeepSeek-R1 / DeepSeekMath [51] 的 GRPO 算法。

#### GRPO 公式（公式 7, 8）

对每个 prompt $o$，policy $\pi_\theta$ 采样 $G$ 个 candidate rollouts $\{o, o_1, ..., o_G\}$，每个得到 reward $r_i$。

**Group-normalized advantage（公式 7）**:
$$A_i = \frac{r_i - \mu}{\sigma}, \quad \mu = \frac{1}{G}\sum_j r_j, \quad \sigma = \text{std}(r_1, ..., r_G)$$

**变量解释**：
- $A_i$：第 $i$ 个 rollout 的 normalized advantage
- $\mu$：group mean reward
- $\sigma$：group std
- 这里 **不需要 value function（critic）**——和 PPO [49] 的关键区别，省内存

**Surrogate objective（公式 8）**:
$$J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(\tau_i|o)}{\pi_{\theta_{\text{old}}}(\tau_i|o)} A_i, \text{clip}\right)\right] - \beta D_{\text{KL}}(\pi_\theta, \pi_{\text{old}})$$

- 第一项：clipped policy ratio（和 PPO 一样）
- 第二项：KL penalty，防止偏离 SFT checkpoint 太远（防 reward hacking），$\beta$ 是系数，paper 里设 $1 \times 10^{-2}$
- $\pi_{\theta_{\text{old}}}$：old policy（前一次迭代）
- $\pi_\theta$：current policy

#### Reward 设计（公式 6）

$$R_{\text{all}} = \lambda_{\text{fmt}} R_{\text{fmt}} + \lambda_{\text{pred}} R_{\text{pred}} + \lambda_{\text{vis}} R_{\text{vis}} + \lambda_{\text{act}} R_{\text{act}} + \lambda_{\text{traj}} R_{\text{traj}}$$

| Reward | 作用 |
|---|---|
| $R_{\text{fmt}}$ | 输出格式合规（XML tags 完整） |
| $R_{\text{pred}}$ | 短期 trajectory + heading 预测准确；和长期轨迹一致 |
| $R_{\text{vis}}$ | visual token 数量匹配 + 每个都在 codebook 内 |
| $R_{\text{act}}$ | high-level action 的 F1 score |
| $R_{\text{traj}}$ | 3s trajectory 精度 + kinematic consistency（加速度变化小） |

**关键设计 choice**：用 **rule-based verifier** 而非 learned reward model。这避免了 reward model 被 hack 的风险，且对于 driving 这种 safety-critical 场景，规则可解释、可审计。这个思路和 Anthropic 的 Constitutional AI、OpenAI 的 process reward model 都有呼应，但更 lightweight。

参考：
- DeepSeekMath (GRPO 原始 paper): https://arxiv.org/abs/2402.03300
- PPO: https://arxiv.org/abs/1707.06347
- Easy-R1 framework (RL 训练): https://arxiv.org/abs/2409.19256

---

## 4. 短期 Trajectory 预测的 Physics-grounded 公式（公式 9-11）

这部分在 supplementary 里，但其实是理解 "为什么 generation 物理可信" 的关键。

**Kinematic state estimation（公式 9）**:
$$\mathbf{v}_t = \frac{\mathbf{P}_t - \mathbf{P}_{t-1}}{\Delta t}, \quad \mathbf{a}_{\text{hist}} = \frac{\mathbf{v}_t - \mathbf{v}_{t-1}}{\Delta t}$$

- $\mathbf{P}_t \in \mathbb{R}^2$：当前 BEV 位置
- $\mathbf{v}_t$：当前速度（finite difference 估计）
- $\mathbf{a}_{\text{hist}}$：历史惯性加速度
- $\Delta t$：采样间隔（0.5s）

**Goal-conditioned acceleration（公式 10）**:
$$\mathbf{a}_{\text{eff}} = (1-\lambda)\mathbf{a}_{\text{hist}} + \lambda \mathbf{a}_{\text{goal}}$$
$$\mathbf{a}_{\text{goal}} = \frac{2}{\tau^2}(\Delta \mathbf{P}_{\text{ideal}} - \mathbf{v}_t \tau)$$

- $\lambda \in [0, 1]$：自适应权重，平衡 inertia vs intention
- $\mathbf{a}_{\text{goal}}$：从当前状态到 goal command $c$ 所需的恒定加速度
- $\Delta \mathbf{P}_{\text{ideal}}$：command $c$ 隐含的目标位移

**最终预测（公式 11）**:
$$\hat{\mathbf{P}}_{t+\tau} = \mathbf{P}_t + \mathbf{v}_t \tau + \frac{1}{2}\mathbf{a}_{\text{eff}} \tau^2$$

标准的 kinematic equation，$\tau = 0.5s$。这个模块给 generation 提供了几何先验，确保生成的未来帧在物理上是 plausible 的。

**Intuition**：直行时 $\lambda \to 0$，纯惯性；急转弯时 $\lambda \to 1$，intention 主导。这种 adaptive blending 比纯 constant-acceleration 模型更鲁棒。

---

## 5. 理论分析：为什么 VLA-World 严格优于单独的 VLA 或 WM

这部分（supplementary A.3）我觉得是 paper 最 deep 的贡献，值得仔细讲。

### 5.1 联合优化目标（公式 13）

$$J(\omega) = \mathbb{E}_{p_\omega(\tau, x | o, g)}[R(\tau_{t:t+H}, x_{t+1})]$$

Driving 的本质：maximize expected return $R$（safety + comfort + rule compliance）over 联合分布 $p(\tau, x | o, g)$。

### 5.2 VLA 的 ELBO 视角（公式 14, 15）

纯 VLA 把 $x_{t+1}$ marginalize 掉：
$$\pi_{\text{VLA}}(\tau | o, g) \approx \int p^*(\tau, x | o, g) dx$$

由 ELBO:
$$\log p^*(\tau | o, g) \geq \mathbb{E}_{x \sim q}[\log p^*(\tau, x | o, g) - \log q(x | o, \tau)]$$

**Insight**：VLA 丢弃 $x_{t+1}$ 等价于优化一个 loose lower bound，丢失了 scene evolution 的预测信息。它直接 fit marginal，不理解 underlying causal variable $x$。

### 5.3 World Model 的问题（公式 16, 17）

World model 优化 reconstruction：
$$J_{\text{WM}}(\theta) = \mathbb{E}[-\log p_\theta(x_{t+1} | o, \tau)]$$

然后 planning 是 external search:
$$\tau^{\text{WM}} = \arg\max_\tau \mathbb{E}_{x \sim p_{\text{WM}}}[R(\tau, x)]$$

**关键问题**：生成 accuracy 和 planning utility 是 weakly coupled 的。一个高保真度的碰撞模拟对 reconstruction loss 是 valid 的，但对 agent 是灾难。**Decision reward $R$ 没有反传到 $\theta$**，imagination 和 consequence 脱节。

### 5.4 VLA-World 的梯度分解（公式 18）

$$\nabla_\omega J(\omega) = \mathbb{E}\left[\underbrace{\nabla_\omega \log \pi_\omega(\tau | o, g) \cdot R}_{\text{Policy Gradient}} + \underbrace{\nabla_\omega \log p_\omega(x | o, \tau) \cdot R}_{\text{World Model Gradient}}\right]$$

**这是 paper 最漂亮的式子**：decision term 和 imagination term 都被同一个 driving reward $R$ 优化。World model 不再只做 reconstruction，而是被 reinforce 去生成 "lead to high-reward outcomes" 的未来——即生成能 highlight risks 的 future，帮助 safety。

### 5.5 严格表达性

- **VLA 是 special case**：mask 掉 imagination branch（$p_\omega(x|\cdot)$ 退化成 delta function），就 recover 纯 VLA
- **WM 是 special case**：freeze $p_\omega(x|o,\tau)$ 参数 + 外部 optimizer，就 recover 纯 WM 的 trajectory search

VLA-World 的 hypothesis class 严格包含两者。

---

## 6. 实验结果详解

### 6.1 Planning 结果（Table 1）

| Method | 1s | 2s | 3s | Avg L2 | Avg Collision | LLM |
|---|---|---|---|---|---|---|
| UniAD* | 0.20 | 0.42 | 0.75 | 0.46 | 0.37 | - |
| BEV-Planner* | 0.16 | 0.32 | 0.57 | 0.35 | 0.34 | - |
| EMMA* (Gemini Nano) | 0.14 | 0.29 | 0.54 | 0.32 | - | Gemini |
| OmniDrive* | 0.14 | 0.29 | 0.55 | 0.33 | 0.30 | LLaVA-7B |
| FSDrive* | 0.14 | 0.25 | 0.46 | 0.28 | 0.10 | Qwen2-VL-2B |
| **VLA-World*** | **0.10** | **0.24** | **0.45** | **0.26** | **0.08** | Qwen2-VL-2B |

观察：
1. VLA-World 在所有 horizon 上都 SOTA，特别是 3s horizon（最难）误差 0.45m
2. 在 **同 backbone（Qwen2-VL-2B）** 下显著超越 FSDrive（0.26 vs 0.28 avg）
3. Collision rate 0.08 是所有方法里最低的，证明反思推理确实提升了 safety

### 6.2 Generation 结果（Table 2, FID）

| Method | Type | Resolution | FID↓ |
|---|---|---|---|
| DriveGAN | GAN | 256×256 | 73.4 |
| DriveDreamer | Diffusion | 128×192 | 52.6 |
| Drive-WM | Diffusion | 192×384 | 15.8 |
| GenAD | Diffusion | 256×448 | 15.4 |
| GEM | Diffusion | 576×1024 | 10.5 |
| Doe-1 | Autoregressive | 384×672 | 15.9 |
| FSDrive | Autoregressive | 128×192 | 10.1 |
| **VLA-World** | Autoregressive | 128×192 | **9.8** |

**重要 observation**：VLA-World 用 autoregressive + 128×192 这种相对受限的 setup，FID 9.8 打败了 GEM 这种 576×1024 的 diffusion 大模型。这暗示了 **action-conditioned generation** 比纯 unconditional generation 更高效——conditional 信息大幅压缩了需要建模的 distribution 复杂度。

### 6.3 Action Prediction（Table 3, F1 score）

| Method | Forward | Left | Right | Keep | Acc | Dec | Stop |
|---|---|---|---|---|---|---|---|
| Qwen2-VL-2B (base) | 62.43 | 22.75 | 28.65 | 40.70 | 50.23 | 49.21 | 41.04 |
| Qwen2-VL-2B† (nuScenes FT) | 92.60 | 61.78 | 66.52 | 56.42 | 74.32 | 76.10 | 74.85 |
| **VLA-World** | **95.88** | **74.22** | **75.06** | **60.98** | **81.42** | **80.04** | **81.24** |

Left/right turn 从 22.75% → 74.22% 的飞跃尤其惊人。这印证了 paper 的论点：通过 RL + 反思推理，模型学会了 "reason about consequences" 而非 "imitate labels"。

### 6.4 Ablation Studies（Table 4）

**(a) Training stages**:
| Variant | 1s | 2s | 3s | Avg |
|---|---|---|---|---|
| w/o PT | 0.35 | 0.56 | 0.81 | 0.57 |
| w/o SFT | 0.35 | 0.79 | 1.40 | 0.85 |
| w/o RL | 0.43 | 0.70 | 1.01 | 0.71 |
| Full | 0.11 | 0.27 | 0.52 | 0.30 |

**关键发现**：去掉 SFT 比去掉 RL 影响更大（0.85 vs 0.71）。这说明 **SFT 的 cold-start supervision 是地基**——RL 在没有 SFT 的情况下 navigate 不了 structured multi-step reasoning 的大搜索空间。这跟 AlphaDrive [30]、AutoDrive-R² [73] 的观察一致。

**(b) Pipeline components**:
| Variant | Avg L2 |
|---|---|
| w/o Perception | 0.75 |
| w/o Generation | 0.68 |
| w/o Reasoning | 0.85 |

Reasoning 模块去掉影响最大（0.30→0.85），证明反思是核心。Perception 也很重要（0.75），generation 相对影响小（0.68），作者解释是 "visual tokens 太多 dominate gradient"。

**(c) Reward components**:
| Variant | Avg L2 |
|---|---|
| w/o $R_{\text{pred}}$ | 0.41 |
| w/o $R_{\text{vis}}$ | 0.42 |
| w/o $R_{\text{act}}$ | 0.62 |
| w/o $R_{\text{traj}}$ | 0.72 |

$R_{\text{traj}}$ 和 $R_{\text{act}}$ 贡献最大，印证 "RL 能 end-to-end 直接优化 planning"。

### 6.5 Scaling Behavior（Table 6）

| Backbone | 1s | 2s | 3s | Avg |
|---|---|---|---|---|
| Qwen2-VL-2B | 0.11 | 0.27 | 0.52 | 0.30 |
| Qwen2.5-VL-3B | 0.05 | 0.08 | 0.76 | 0.29 |
| Qwen2-VL-7B | **0.03** | **0.03** | **0.47** | **0.18** |

7B 比 2B 提升约 40%，说明 VLA-World paradigm 享受清晰的 scaling law。这跟 LLM 的 scaling 一致，暗示这套方法在更大 backbone 下还有很大 headroom。

参考：
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- nuScenes: https://www.nuscenes.org

---

## 7. 与相关工作的对比和我的延伸思考

### 7.1 vs FSDrive [74]

FSDrive 是 VLA-World 的直接前身，也用 Qwen2-VL-2B + spatiotemporal CoT。关键区别：
1. FSDrive 只生成 front view；VLA-World multi-view
2. FSDrive 直接 regress waypoints 不评估物理 feasibility；VLA-World 有反思闭环
3. VLA-World 加了 GRPO RL 阶段

### 7.2 vs Doe-1 [80]

Doe-1 用 Lumina-mGPT-7B 做 closed-loop driving with world model，但没有 reflective reasoning 阶段。VLA-World 的反思是关键差异化。

### 7.3 vs AlphaDrive [30], AutoDrive-R² [73], DriveAgent-R1 [81]

这些是同期用 GRPO 训 VLA 的工作，但它们 **没有 world model / generation**——纯语言 CoT。VLA-World 把 visual generation 作为 reasoning 的 sketchpad，是 visual CoT 而非 text CoT。

### 7.4 vs Janus / Show-o / WorldVLA

这些是 unified understanding + generation 的通用工作，VLA-World 把这个范式 specialize 到 driving，并加了 action conditioning 和 reflective loop。

### 7.5 vs World Simulators (Sora, GAIA-1)

Sora [9] 和 GAIA-1 [22] 是通用 world simulators，但它们对 driving decision 的 reward 没有反传——VLA-World 的公式 18 是关键突破。

### 7.6 我对这套方法局限性的思考

1. **0.5s horizon 太短**：真实驾驶需要 3-5s foresight，VLA-World 只 generate 0.5s 未来帧。论文里 3s 轨迹是通过反思推理 "推" 出来的，不是 generate 出来的。递归 generate 多帧可能更强大但计算贵。

2. **Single camera generation**：虽然 pretrain 是 multi-view 的，但 SFT/RL 阶段似乎只生成一个 view 的未来帧。完整的 360° 一致性还需要验证。

3. **Rule-based reward 的天花板**：rule-based verifier 可解释但不一定能 capture complex scenario 的 nuance。未来可能需要 hybrid（rule + learned）。

4. **Open-loop evaluation**：nuScenes 是 open-loop benchmark，collision rate 是 retrospective checking。真正的闭环测试（如 CARLA,nuPlan）还需要验证。

参考：
- nuPlan: https://www.nuscenes.org/nuplan
- CARLA: https://carla.org

---

## 8. 关键 Takeaways（build intuition）

1. **Future frame 是 sketchpad，不是 final output**。Generation 在 VLA-World 里是手段，不是目的——它把高维未来压缩到 visual token 空间，供 System 2 反思。

2. **三阶段训练缺一不可**。PT 提供生成能力，SFT 提供 driving 概念地基，RL refine 推理策略。去掉 SFT 的 RL 没法 explore（ablation 已验证）。

3. **Joint optimization 是核心**。公式 18 表明 decision 和 imagination 被 same reward 优化，这从根本上解决了 world model 的 "高保真但不安全" 问题。

4. **GRPO 的 value-free 设计在 visual dynamics 高维空间里是优势**。比 PPO 省 memory，更适合 VLM。

5. **Scaling law 适用**。7B 显著优于 2B（40% 提升），暗示更大模型会更强。

6. **Rule-based reward + 结构化输出 是 RL for driving 的 practical 解**。避免了 learned reward model 的 hack 风险。

---

## 9. 相关参考链接汇总

**核心 paper**:
- VLA-World project page: https://vlaworld.github.io
- FSDrive (前身): https://arxiv.org/abs/2505.17685
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

**Backbone & 基础**:
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- VQGAN: https://arxiv.org/abs/2012.09841
- PPO: https://arxiv.org/abs/1707.06347

**Autonomous driving baselines**:
- UniAD: https://arxiv.org/abs/2212.10156
- VAD: https://arxiv.org/abs/2303.12077
- DriveDreamer: https://arxiv.org/abs/2309.09777
- Drive-WM: https://arxiv.org/abs/2310.01515
- OccWorld: https://arxiv.org/abs/2311.10838
- Doe-1: https://arxiv.org/abs/2412.09627
- OmniDrive: https://arxiv.org/abs/2406.08351
- EMMA: https://arxiv.org/abs/2410.23262
- AlphaDrive: https://arxiv.org/abs/2503.07608

**Unified understanding+generation**:
- Janus: https://arxiv.org/abs/2410.13848
- Janus-Pro: https://arxiv.org/abs/2501.17811
- Show-o: https://arxiv.org/abs/2408.12528
- WorldVLA: https://arxiv.org/abs/2506.21539

**Dataset & benchmark**:
- nuScenes: https://www.nuscenes.org
- nuPlan: https://www.nuscenes.org/nuplan

**Frameworks**:
- LLaMA Factory: https://arxiv.org/abs/2406.12710
- Easy-R1 (HybridFlow): https://arxiv.org/abs/2409.19256

---

总结一句，Andrej，这篇 paper 我觉得最重要的贡献是公式 18 那个梯度分解——它给 "world model 应该被 decision reward 优化" 这个直觉提供了严格的理论表达。三阶段训练 + GRPO + rule-based reward 是把这套理论落到 driving 场景的工程实现。Visual sketchpad 的思想很 general，我预期会扩展到 robotics、embodied AI 等更广领域。
