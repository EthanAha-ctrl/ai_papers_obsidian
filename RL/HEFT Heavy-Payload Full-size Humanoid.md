---
source_pdf: HEFT Heavy-Payload Full-size Humanoid.pdf
paper_sha256: 5b22edc2245965808231ad71d42e912f406afb0bd386bfe4119ea9c74803cfc7
processed_at: '2026-08-19T10:41:12-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HEFT 人话版：给 Karpathy 的啤酒闲聊

Andrej, 咱们把这 paper 用最直白的话再过一遍，就像在 NeurIPS poster session 拿着啤酒聊那样。

---

## 一句话总结

他们让一个 65kg 的 full-size humanoid，戴着 VR 头显遥操作，单手拎 10kg 水瓶走，双手各 12kg kettlebell 走路还能蹲下站起来——全用一个 policy，no task switching，no special controller。

这事以前没人真正做成过。

---

## 为什么这事难？两个真问题

### 痛点 1: VR tracking 是垃圾信号

你戴 VR 头显操作机器人，robot 看到的 "你的动作" 来自哪里？来自头显 + 两个手柄 + 也许脚上的 tracker。这几个稀疏点喂给 SMPL-X IK 反推你全身，出来的 motion 有四种 structured garbage：

1. **Drift**: VR 坐标系慢慢飘，你站那儿不动，robot 以为你在往前蹭
2. **Body-frame bias**: 头显在头上，但 IK 反推 root 时有系统性偏移
3. **Latency**: tracker 刷新跟 control loop 不同步，指令永远迟到
4. **End-effector offset**: 手柄位置跟真实 wrist joint 有几何错位

以前的人怎么处理？
- [OmniH2O](https://arxiv.org/abs/2406.08858) 直接拿这个 noisy 信号 retarget 一下就喂 policy，policy 被垃圾信号毒害
- [AvatarPoser](https://arxiv.org/abs/2207.09630) / [RoHM](https://virtualhumans.mpi-inf.mpg.de/papers/zhang2024rohm/zhang2024rohm.pdf) 离线用 diffusion 重建干净 motion，但 diffusion 要 50 步去噪，100ms+ 延迟，teleop 用不了

HEFT 的 trick 很 simple 但很 clever: **训练时让 actor 吃 noisy 信号，让 critic 和 reward 用 clean 信号打分**。

打个比方：你给学生做带涂鸦的卷子，但用标准答案批改。学生被迫学会"从涂鸦里读出题意"。等考试时卷子还是涂鸦的，但他已经知道怎么去读懂了。

这就叫 **Privileged Motion Guidance (PMG)**。

### 痛点 2: 同一个重量，不同动作下难度差 10 倍

你想想：
- 站那儿不动，拎 20kg 哑铃 → 小菜
- 单腿转身，拎 20kg 哑铃 → 要摔
- 蹲下去再起来，拎 20kg 哑铃 → 要命

以前的 [FALCON](https://arxiv.org/abs/2505.06776) 式 force curriculum 是全局的：从 0kg 线性加到 30kg，不管你在干嘛。结果 policy 在"转身 + 25kg"这种注定失败的组合上狂试，buffer 全是失败数据，学废了。

HEFT 的招：**把每个 motion 切成 5 秒窗口，用 expert policy 先去试每个窗口能扛多重，记下来当 cap**。转身窗口 cap 可能是 10kg，站立窗口 cap 可能是 30kg。训练时在这个 cap 内随机采样 payload。

这就叫 **Windowed Payload Curriculum (WPC)**。

intuition 是：别让 policy 在它注定学不会的组合上浪费 sample。让它在"刚好够呛能学会"的区间里练。

---

## PMG 细节：paired data 怎么来的

数据 pipeline 是这样的：

```
VR teleop session
    ↓
SMPL-X 序列（带 noise）
    ↓
    ├──→ 直接 retarget → S_raw (noisy 版，部署用)
    │
    └──→ RoHM diffusion 重建 → retarget → S_clean (clean 版，训练打分用)
```

对于普通 mocap clip (SEED, 100STYLE, LaFAN1)，本来就没 noise，所以 $S_{\text{raw}} = S_{\text{clean}}$。

训练时：
- **Actor** 输入：$o_t^{\text{dep}} = (S_{\text{prop},t}, S_{\text{raw},t})$ ← noisy VR + proprioception
- **Critic** 输入：$o_t^{\text{priv}} = (S_{\text{prop},t}, S_{\text{clean},t}, \text{sim info})$ ← clean + 全能视角
- **Reward**: $R(s_t, a_t; S_{\text{clean},t})$ ← 用 clean reference 算

这就是经典的 asymmetric actor-critic，但 paired reference 是新 ingredient。

**为什么这 trick 在 clean mocap 上也没退化？**

你猜加 noise training 会在 clean test 上掉点，对吧？结果 paper Fig. 3 显示 PMG 在 $\mathcal{D}_{\text{random}}$ (clean SEED clips) 上 MPJPE 0.021m (G1), 0.036m (L7)，比 SONIC (0.043m) 和 TWIST2 (0.061m) 都好。

我的解读：clean critic 给了 policy 一个更稳定的 value landscape。clean input 时 critic 和 actor gap 为 0，policy 退化成标准 tracker；noisy input 时 critic 抑制了 spurious value spike。一套 value function 同时服务两种 input，反而比分开训更 robust。

这跟 [noise2noise](https://arxiv.org/abs/1803.04189) 的思路有点像，但 noise2noise 是 noisy target 监督 noisy input（假设 noise 独立），HEFT 是 clean target 监督 noisy input（更强）。

---

## WPC 细节：expert 怎么标 cap

Algorithm 1 翻译成人话：

```
对每个 motion m_i:
    切成 5 秒窗口 w_{i,k}
    对每个窗口:
        从 30kg 开始往下试，5kg 一档:
            用 expert policy 扛这个重量跑一遍这个窗口
            如果没摔倒 → 记下这个重量当 cap，下一个窗口
            如果摔了 → 减 5kg 再试
```

expert policy 是啥？是拿 **clean reference + 全 privileged info**（contact、payload、randomization ground truth）训出来的强 policy。它比最终部署的 student policy 厉害得多，因为它什么都知道。

这里有个 chicken-and-egg 问题：你要先有 expert 才能 label cap，你要 label cap 才能训最终 policy。解法是分两阶段：
1. 先训一个 expert（用 global curriculum 或简单 curriculum）
2. 用 expert label cap
3. 用 labeled cap 训最终 student policy

**5 秒窗口是 tradeoff**：
- 太短（1-2s）: expert 还没进稳态，cap 估计噪声大
- 太长（10s+）: 窗口内 motion 变化太大，cap 过保守（被窗口内最难的那帧拉低）
- 5s 是经验值，paper 没做 ablation

**Payload sampling 公式 (Eq. 3)**:

$$
F_{i,k} \sim \mathcal{U}\Big(0, \bar{F}_{i,k} \cdot \text{clip}\Big(\frac{p}{0.8}, 0, 1\Big)\Big)
$$

人话：
- $\bar{F}_{i,k}$: 这个窗口的 expert cap
- $p$: training 进度，0 到 1
- $\frac{p}{0.8}$: 80% 训练进度前线性升 payload，之后饱和
- $\mathcal{U}(0, \cdot)$: 在 0 到 cap×scale 之间均匀采样

所以训练初期所有窗口都只扛很轻的（scale 小），训练后期简单窗口扛满 cap（可能是 30kg），难窗口扛自己的 cap（可能是 10kg）。

**Force model 细节**:
- Force 加在两个 wrist-roll link 上（不是 hand，因为没建模 grasp）
- 方向在 $12°$ cone around downward gravity 内随机
- 两手随机 split（不是 50/50，模拟不对称提物）

这个 model 简化了：没建模 grasp quality、object geometry、sliding。但作为 sim training 的扰动源够用了。

---

## RMA 那套老把戏

[Original RMA paper](https://arxiv.org/abs/2107.04034) 的思路：robot 部署时不知道环境参数（mass, friction, payload），但能从历史 proprioception 反推。

HEFT 的实现：

**Teacher** 训练时知道一切：
$$
z_t = E_p(S_{\text{clean},t}, S_{\text{priv},t})
$$
$S_{\text{priv},t}$ 包含 tracking error、contact signal、payload state、randomization vars。$z_t$ 是 256-D latent，编码"当前环境 Hidden 参数"。

**Adapter** 部署时只用可观察历史：
$$
\hat{z}_t = E_a(S_{\text{prop},t}, S_{\text{raw},t})
$$

**Supervision**: $\mathcal{L}_{\text{adapt}} = \|\hat{z}_t - z_t\|_2^2$

Adapter 学会从关节电流、速度、IMU 反应推 payload。你提 25kg 时 wrist motor 要输出更大电流 hold 住 pose，这个 current draw 在 prop 里能读到，adapter 学会读这个信号。

**三阶段训练**:
1. 训 teacher（PPO + PMG + WPC），5×10⁹ frames
2. 冻 teacher，训 adapter 去匹配 teacher 的 latent
3. 用 adapter 预测的 $\hat{z}$ 接回 student actor，继续 PPO fine-tune

部署时只用 student actor + adapter，teacher / privileged encoder / simulator info 全扔掉。

---

## Reward 细节挑几个说

所有 tracking reward 都是 exponential kernel：
$$
r_k = w_k \exp(-e_k / \sigma_k)
$$

几个有意思的点：

**Keypoint 只用 13 个**：head, hands, 几个 upper-limb links, hip-yaw, knee, ankle-roll。比 [PHC](https://arxiv.org/abs/2309.08632) 的 dense keypoint 紧凑。这 13 个点覆盖了 tracking 的关键 DoF，少了 reward 冗余。

**Root velocity 用 body frame**：$R^\top v_{\text{root}}$，把世界系速度转到 robot body frame。因为 humanoid 的"前进"在 body frame 是固定的，世界系下会随 yaw 变。

**Survival reward = 3.0 是最高权重**：说明 paper 把"不摔"看得比"跟得准"重。Heavy payload 下 robot 一不小心 CoM 偏了就摔，survival 把 policy 拉向保守。

**Foot contact reward 双层**:
- Sparse air-time reward ($w=5.0$): 用 clean reference 的 foot contact state 当 ground truth，鼓励 swing/stance 时机一致
- Dense contact mismatch ($w=1.0$): contact state 对不上就 -1

contact reward 很关键，因为 humanoid 步态本质是 contact sequencing。

---

## 实验数据人话版

### PMG 效果 (Fig. 3)

**Noisy VR 上 ($\mathcal{D}_{\text{VR}}$)**:
- PMG: 0.544m (G1), 0.560m (L7) final horizontal root error
- Mocap-only no-noise: 明显更差
- Mocap + Gaussian noise: 中间
- TWIST2: 最差

**Clean mocap 上 ($\mathcal{D}_{\text{random}}$)**:
- PMG MPJPE: 0.021m (G1), 0.036m (L7)
- SONIC: 0.043m (G1)
- TWIST2: 0.061m (G1), 0.099m (L7)

PMG 在两边都赢。Noisy 上赢很合理，clean 上也赢有点 surprising，说明 paired training 没损害 clean 能力。

### WPC 效果 (Fig. 4 + Table 1)

**不同 payload 下的 success rate (L7, $\mathcal{D}_{\text{random}}$)**:

| Payload | WPC (Ours) | w/o expert | TWIST2+FC |
|---|---|---|---|
| 0 kg | ~100% | ~100% | ~95% |
| 20 kg | ~100% | ~90% | ~50% |
| 25 kg | 90% | 80% | 35% |
| 30 kg | 75% | 62% | 29% |

WPC 在 high payload 下碾压。

**High-dynamic 无 payload ($\mathcal{D}_{\text{dynamic}}$)**:

| Policy | Success | MPJPE |
|---|---|---|
| w/o expert | 0.64 | 0.060m |
| WPC (Ours) | 0.73 | 0.057m |
| TWIST2+FC | 0.10 | 0.133m |

这个结果很有意思：WPC 在无 payload 高 dynamic 上 success 比 ablation 还高（0.73 vs 0.64）。我的解读：WPC 的 expert cap 起到了 motion segmentation 作用，难 motion 段不让 policy 见 high payload，等于给了 policy 更稳定的训练分布，间接帮助 high-dynamic 学习。

TWIST2+FC 在 high-dynamic 上崩成 0.10，说明 FALCON-style global curriculum 配 tracking 在难 motion 上会 fail——policy 在"难 motion + 高 payload"组合上狂失败，buffer 污染，学废。

---

## 真实部署：24kg 是什么概念

L7 是 65kg robot。24kg payload = 37% body mass。

想象你自己体重 70kg，双手各拎 12kg kettlebell，然后蹲下去再站起来——这就是 paper Fig. 5(g) 做的事。

还有 Fig. 5(e): 不对称提 10kg 水瓶走路。一手重一手轻，CoM 偏一侧，policy 要会补偿。

Fig. 5(f): 双手各 12kg kettlebell 走路。Locomotion + heavy payload。

Fig. 5(a): 从地面捡 5kg backpack 放椅子上。Pickup + place + posture 变化。

Fig. 5(b): 推带轮子的 rack。持续水平力，contact-rich。

Fig. 5(c): 搬 8kg 小桌子。Bulky object，手臂姿势要变。

**一个 policy 全包**。没切 task-specific controller，没切 payload-specific mode。这就是 paper 最 impressive 的地方。

---

## 我觉得最聪明的几个设计

### 1. Paired data 当 implicit denoiser

PMG 本质是让 policy 内化一个 implicit denoiser。你给它 noisy input，要它输出能 match clean target 的 action，policy 必须学会区分"信号"和"噪声"。

这比显式加 denoiser 好在哪？denoiser 是前处理，输出 clean 再喂 policy。但 denoiser 出错时 policy 没法纠正。PMG 把 denoising 烧进 policy weights，policy 可以用 proprioception 交叉验证 reference 信号。

### 2. Expert rollout 当 feasibility oracle

WPC 的 expert labeling 本质是 "用强 policy 当 difficulty oracle"。你不知道一个 motion 在某 payload 下可不可行，但你可以跑一遍试试。

这比用动力学约束算 cap 好太多。动力学约束只能查 torque limit，查不了"这个 motion 的 contact sequencing 在 payload 下稳不稳"。Expert rollout 是 ground truth feasibility probe。

### 3. Per-window cap 同时解决两个问题

WPC 表面上解决 payload curriculum，实际上还顺带做了 motion segmentation。难 motion 段 cap 低，等于在训练时把难 motion 和易 motion 分开标定难度。这让 policy 在难 motion 上不被 high payload 干扰，专心学 motion 本身。

---

## 我会吐槽的点

### 1. 5 秒窗口没 ablation

paper 没说为啥是 5 秒。1 秒行不行？10 秒会不会更好？这个 hyperparameter 影响 cap 估计精度和窗口内 motion 一致性，应该 ablation 一下。

### 2. Force model 太简化

只把 payload 当 wrist 上的 downward force。真实提 24kg kettlebell 时，force 作用在 hand grip 上，有 offset，kettlebell 重心在 hand 下方 20cm，会产生 moment。这个 model gap 靠 domain randomization 硬扛。

未来应该接 [tactile sensing](https://arxiv.org/abs/2306.08320) 或 grasp network，把 wrench 建模成 6D。

### 3. 只在 L7 上验证

L7 是 175cm/65kg，跟 [Unitree H1](https://www.unitree.com/h1) (180cm/47kg) 质量分布差很多。HEFT 在更轻的 H1 上行不行？payload ratio 是不是要重新标？这都没验证。

### 4. Expert labeling 计算成本

每个 motion 60 个窗口 × 7 个 load level × expert rollout = 420 次 rollout per motion。SEED 有 61612 个 clip... labeling 一次要很久。paper 没说具体时间，但这成本不低。

未来方向：self-labeling，policy 自己跑 rollout 标 cap，iterative refine。

---

## 延伸联想

### 联想 1: PMG 能推广到其他 noisy command 场景

任何 robot 从 noisy interface 接指令的场景都能用 paired data training：
- Joystick teleop（人手抖动）
- Voice command（ASR error）
- Sketch input（drawing-to-action）
- 甚至 VLA 里的 language instruction（LLM 输出有 hallucination）

只要你能构造 (noisy_input, clean_target) pair，就能用 PMG。

### 联想 2: WPC 能推广到其他 per-instance difficulty

WPC 的本质是 per-instance difficulty-aware curriculum：
- Terrain curriculum: 不同 terrain patch 给不同 roughness cap
- Velocity curriculum: 不同 motion type 给不同 speed cap
- Contact curriculum: 不同 interaction phase 给不同 force cap

比单一 global curriculum sample efficient。

### 联想 3: Diffusion Policy 可能天然解决 PMG

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 学 action distribution，自带 multi-modal 和 noise robustness。如果 actor 换成 diffusion，可能天然 robust to VR noise，不需要 paired data training。但 latency 会涨。

### 联想 4: 跟 Foundation Model 趋势对比

现在 humanoid foundation controller 趋势是大数据多任务训 universal policy（[NOEMO](https://arxiv.org/abs/2505.07148), [SONIC](https://arxiv.org/abs/2511.07820)）。HEFT 反其道，强调 asymmetric signal pairing 这个小 trick。

两者不矛盾。PMG 和 WPC 可以 plug-in 到 foundation controller 上，作为 fine-tuning 或 adapter 阶段的 trick。

### 联想 5: Latent 包含更多物理量

HEFT 的 256-D latent 只编码 payload + contact + randomization。如果显式加 object mass, CoM offset, friction 进 latent，并用 tactile sensor 反推，可能直接 close the loop。参考 [AnySkin](https://arxiv.org/abs/2409.08276) 或 [GelSight](https://arxiv.org/abs/2103.12277) 的工作。

### 联想 6: Online reconstruction 何时能跟上？

RoHM 50 步 diffusion 太慢。但 [Consistency Models](https://arxiv.org/abs/2303.01469) 能 1-4 步采样，[Rectified Flow](https://arxiv.org/abs/2403.03206) 也快。如果 online reconstruction 能做到 20Hz，那 PMG 的"训练用 clean，部署用 raw"就可以变成"部署也用 online clean"，更进一步。

---

## 相关 reference 汇总

**核心方法**:
- [HEFT Project Page](https://heft-homepage.github.io) (paper 给的链接)
- [RoHM (CVPR 2024)](https://virtualhumans.mpi-inf.mpg.de/papers/zhang2024rohm/zhang2024rohm.pdf) - diffusion motion reconstruction
- [RMA (RSS 2022)](https://arxiv.org/abs/2107.04034) - rapid motor adaptation
- [PPO (2017)](https://arxiv.org/abs/1707.06347) - proximal policy optimization

**Humanoid teleop baseline**:
- [OmniH2O (2024)](https://arxiv.org/abs/2406.08858)
- [HumanPlus (2024)](https://arxiv.org/abs/2406.10454)
- [Open-Television (2024)](https://arxiv.org/abs/2407.01512)
- [TWIST (2025)](https://arxiv.org/abs/2505.02833)
- [TWIST2 (2025)](https://arxiv.org/abs/2511.02832)
- [SONIC (2026)](https://arxiv.org/abs/2511.07820)

**Motion tracking / imitation**:
- [DeepMimic (2018)](https://arxiv.org/abs/1804.02717)
- [AMP (2021)](https://arxiv.org/abs/2105.01699)
- [PHC (ICCV 2023)](https://arxiv.org/abs/2309.08632)
- [ASE (2022)](https://arxiv.org/abs/2205.01906)

**Payload / loco-manip**:
- [FALCON (2025)](https://arxiv.org/abs/2505.06776)
- [HALO (2026)](https://arxiv.org/abs/2603.15084)
- [Sim-to-real box loco-manip (ICRA 2024)](https://arxiv.org/abs/2310.02505)

**Sparse motion reconstruction**:
- [AvatarPoser (ECCV 2022)](https://arxiv.org/abs/2207.09630)
- [QuestSim (SIGGRAPH Asia 2022)](https://arxiv.org/abs/2211.09183)
- [WHAM (CVPR 2024)](https://arxiv.org/abs/2311.17405)
- [TRAM (ECCV 2024)](https://arxiv.org/abs/2404.18451)

**Diffusion 加速**:
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Rectified Flow](https://arxiv.org/abs/2403.03206)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

**Tactile / grasp**:
- [GelSight](https://arxiv.org/abs/2103.12277)
- [AnySkin](https://arxiv.org/abs/2409.08276)

**Noise2Noise**:
- [Noise2Noise (ICML 2018)](https://arxiv.org/abs/1803.04189)

---

Andrej, 简单说，HEFT 就是把"训练时见垃圾、部署时也见垃圾，但训练时用干净答案批改"这件事做扎实了，加上"用强 policy 当难度评估器"的 per-window curriculum，在真实 24kg payload 上跑通了。两个 trick 都不花哨，但组合在一起 work 得很扎实。这种 paper 我觉得比那些号称 foundation model 的工作实在多了——它真正解决了 deploy 时的痛点。

想继续聊哪块都行，比如 RoHM diffusion 细节、RMA convergence、或者 reward shaping sensitivity。

---

# HEFT 深度解析：Full-size Humanoid Heavy-Payload Teleoperation

Andrej, 这篇paper我读得很有感触，因为它把几个在 humanoid RL 圈子里一直被回避的问题摆到台面：**full-size 平台 + VR 遥操作 + 真实重物**，三者叠加产生的复合 difficulty。下面我从 problem formulation、PMG、WPC、teacher-student distillation、reward design、实验数据六个层面拆开讲，并在每一步都尽量给你的 intuition 充上电。

---

## 1. Problem Formulation 的本质：Reference-Conditioned Payload Tracking

核心 objective 是 Eq. (2):

$$
\max_{\pi} \mathbb{E}_{m_i, F_{\text{window}}} \left[ \sum_{t=0}^{T} \gamma^t R\big(s_t, a_t; S_{\text{clean},t}^i\big) \right], \quad a_t \sim \pi(\cdot \mid o_t^{\text{dep}})
$$

变量含义：
- $m_i$: 从 motion library 采样的 reference motion，可能是 mocap clip，也可能是 paired VR clip
- $F_{\text{window}}$: 与该 motion 关联的 windowed payload schedule，是一个时间序列的 force 上界
- $\gamma$: discount factor（paper 中 0.99，注意 Table D.2 里 GAE 是 0.95，两个值含义不同，$\gamma$ 控制 value 估计的 horizon，GAE $\lambda$ 控制 bias-variance tradeoff）
- $S_{\text{clean},t}^i$: **奖励参考目标**，来自 retargeted mocap 或 RoHM-reconstructed VR motion
- $o_t^{\text{dep}} = (S_{\text{prop},t}, S_{\text{raw},t})$: **部署时唯一可观察**，proprioception + raw VR reference
- $a_t$: joint position command，由低层 PD controller 跟踪

**Intuition**: 这个 formulation 的精妙之处在于把"训练时的 ground truth"和"部署时的 input"显式分离。一般 imitation learning 默认 demonstration 是 clean 的，agent 学一个 $a \sim \pi(o_{\text{clean}})$ 就行。但 VR teleop 的本质是：operator 戴 HMD、手里握 controller，skeleton 是从 sparse signal 反推的，这个反推过程本身就有 drift、latency、body-frame bias、end-effector offset。如果你的 policy 在训练时见到的是 clean mocap，部署时见到的是 noisy VR，这就形成一个 covariate shift + reward mismatch。

HEFT 的解法是 reward anchor 用 clean、actor input 用 raw，等价于在告诉 policy：**"你看的是带噪的指令，但你应该追求的 target 是物理上合理的版本"**。这跟 DAgger 里的 state distribution correction 思路是相通的，但 HEFT 没用 online correction，而是把 correction 烧进 value function 和 reward 里。

参考一下这个领域的脉络：
- [DeepMimic (Peng et al. 2018)](https://arxiv.org/abs/1804.02717): 单 reference imitation，clean mocap
- [AMP (Peng et al. 2021)](https://arxiv.org/abs/2105.01699): adversarial motion prior，不需要 1-to-1 对齐
- [OmniH2O (He et al. 2024)](https://arxiv.org/abs/2406.08858): whole-body teleop，但用的是较 clean 的 retargeting
- [SONIC (Luo et al. 2026)](https://arxiv.org/abs/2511.07820): 大规模 tracking，多输入接口，但是是 full clean reference
- [TWIST2 (Ze et al. 2025)](https://arxiv.org/abs/2511.02832): portable VR teleop，paper 里被 HEFT 当作 baseline

HEFT 区别于上述工作的关键就是 **paired raw/clean 数据**这件事——下面 PMG 部分详谈。

---

## 2. Privileged Motion Guidance (PMG)：把 structured VR noise 学进 policy

### 2.1 为什么不能简单加 Gaussian noise？

很多人做 sim-to-real 的 reference noise robustness 时，直接在 reference 上加 $\mathcal{N}(0, \sigma^2)$，domain randomization 一下就完事。paper 在 Sec. 3.2 明确说这不够。VR tracker 的噪声是 **structured**，paper 列了四种 artifact：

1. **Global drift**: VR 坐标系相对世界系有缓慢漂移（lighthouse / inside-out tracking的累积误差），不是 frame-wise i.i.d.
2. **Body-frame bias**: 头、手、脚的 sparse 测量 → SMPL-X inverse kinematics 得到的 root frame 有系统性偏移
3. **Latency**: tracker 物理刷新率与控制 loop 不同步，导致 reference 滞后于真实 operator motion
4. **End-effector offset**: 手部 controller 测量点和真实 wrist joint 之间有几何偏置

这四种 noise 用 Gaussian 建模完全不靠谱。Gaussian 会把 high-frequency jitter 当主信号，但 VR 的 drift 是低频的、bias 是常数的、latency 是延迟的——它们的频谱和结构都不同。

### 2.2 RoHM reconstruction 作为 "clean teacher"

paper 用 [RoHM (Zhang et al. CVPR 2024)](https://virtualhumans.mpi-inf.mpg.de/papers/zhang2024rohm/zhang2024rohm.pdf) 做离线重建。RoHM 是个 diffusion-based denoiser，在 synthetically corrupted AMASS motions 上训练，能从 noisy partial observations 恢复 plausible full-body motion。

为什么用 diffusion 而不是优化？因为 human motion 的 manifold 是高度 non-Gaussian 的——例如脚在 swing phase 不能凭空穿地、root 不能瞬间反向。Diffusion 学的是数据分布 $p(m | \text{partial obs})$，比任何 smoothness prior 都准。

**Critical insight**: RoHM 不能在线用，因为 diffusion 的 iterative denoising 需要 50-100 步去噪，每步要一次神经网络 forward，即使 30Hz 也很难做到 sub-100ms 延迟。这会破坏 teleop 的 closed-loop feedback。所以 HEFT 的策略是：**只把 RoHM 用作离线数据增强**，部署时 policy 直接吃 raw VR。

### 2.3 Asymmetric actor-critic 的具体构造

paper 用了 standard privileged critic setup（[PPO](https://arxiv.org/abs/1707.06347) + asymmetric observation），但加了 paired data 这个新 ingredient：

| 组件 | 输入 | 用途 |
|---|---|---|
| Actor (student) | $S_{\text{prop},t} + S_{\text{raw},t}$ (raw VR) | 部署时唯一可观察 |
| Critic | $S_{\text{prop},t} + S_{\text{clean},t} + S_{\text{priv},t}$ | 训练时估计 value |
| Reward | $R(s_t, a_t; S_{\text{clean},t})$ | 用 clean reference 计算 |

这意味着同一个 rollout 内，actor 看到 noisy input 做决策，但 reward 和 value 都用 clean 版本算。梯度方向 push policy 去**实现 clean reference 的物理状态**，即使它看到的指令是 noisy 的。

**Intuition**: 这本质是 invert 一个 noisy encoder。Policy 必须学会 "raw reference signal → 真实 operator intent" 的 mapping。如果它把 VR drift 当成真实意图去追，reward 立刻惩罚（因为 clean reference 没有那个 drift）。这逼着 policy 内化一个 implicit denoiser。

对比一下：
- [HumanPlus (Fu et al. 2024)](https://arxiv.org/abs/2406.10454) 用 shadowing，没显式处理 VR noise
- [OmniH2O](https://arxiv.org/abs/2406.08858) 用 retargeting 一次性把 VR 转成 robot target，noisy 信号直接进 policy
- [AvatarPoser (Jiang et al. ECCV 2022)](https://arxiv.org/abs/2207.09630) 也是 offline reconstruction，但没在 RL loop 里用

HEFT 是第一个把"offline reconstruction + online raw signal"在 RL 训练里 explicit pair 起来的工作。

### 2.4 PMG 实验数据解读

Fig. 3 给出 G1 和 L7 两个平台的结果：

**On $\mathcal{D}_{\text{VR}}$ (noisy VR held-out)**:
- PMG final horizontal root error: 0.544m (G1), 0.560m (L7)
- mocap-only no-noise training: 显著更差（Fig. 3 bar chart 中明显更高）
- mocap + generic Gaussian noise: 介于两者之间
- TWIST2 baseline: 最差

**On $\mathcal{D}_{\text{random}}$ (clean mocap, 100 SEED clips)**:
- PMG MPJPE: 0.021m (G1), 0.036m (L7)
- SONIC: 0.043m (G1)
- TWIST2: 0.061m (G1), 0.099m (L7)

这个结果有意思的地方在于：PMG 不仅在 noisy VR 上更好，**在 clean mocap 上也比 baseline 好**。这说明 paired data training 没有 catastrophic forgetting 干净数据上的能力——这有点反直觉，因为通常加 noise 会损害 clean 上的表现。

我的解读是：privileged critic 给了 policy 一个更稳定的 value landscape。当 reference 是 clean 时，clean critic 和 raw actor 的 gap 是 0，policy 行为退化到标准 tracking；当 reference 是 noisy 时，clean critic 抑制了 noisy 信号引发的 spurious value spike。这种"统一 value function"反而比单独训两套好。

---

## 3. Windowed Payload Curriculum (WPC)：Motion-Conditioned Force Curriculum

### 3.1 为什么 global payload curriculum 不够？

[FALCON (Zhang et al. 2025)](https://arxiv.org/abs/2505.06776) 是 payload-aware humanoid loco-manip 的代表作，用了 force curriculum。但 FALCON 主要在 specific loco-manip task 上做，没面对 broad motion library。

HEFT 想用一个 policy 同时处理:
- 走路 (quasi-static support phase, payload easy)
- 转身 (single support, dynamic, payload hard)
- 蹲下 (CoM 大幅下降, payload hardest)
- 快速手臂摆动 (centroidal momentum 扰动大, payload destabilizing)

如果用 global curriculum，从 0kg 线性升到 30kg，那"快速转身 + 25kg"和"静止站立 + 25kg"的 difficulty 完全不在一个量级。前者几乎注定失败，会污染训练数据；后者太容易，浪费 sample。

paper 的 key insight: **payload 难度是 reference-conditioned 的**，同一个 load 在不同 motion window 下 feasibility 不同。

### 3.2 Expert rollout labeling 算法

Algorithm 1 是核心：

```
For each motion m_i:
    Split into 5s windows w_{i,k}
    For each window:
        For load c in {30, 25, ..., 0} kg:
            Roll out expert π_E under load c
            If rollout succeeds without termination:
                cap_{i,k} = c; break
```

变量含义：
- $T_w = 5$s: window 长度，是 paper 经验值。太短 (1-2s) expert rollout 还没进入稳态，cap 估计噪声大；太长 (10s+) 同一 window 内 motion 变化太大，cap 过保守
- Load grid $\{30, 25, ..., 0\}$ kg: 5kg 步长。这个粒度足够细又不会让 labeling 太贵
- Expert $\pi_E$: 用 **clean reference + privileged simulator info** 训练的强 policy，rollout 时关掉 observation noise 和 domain randomization

**为什么用 expert 而不是直接用动力学约束估 cap？** 因为 humanoid tracking 的 feasibility 不是简单的 torque limit 检查——它取决于 policy 的 recovery 能力、contact timing、CoM 位置、foot placement。一个 motion 在 20kg 下能不能 track 住，得真的跑一次才知道。Expert rollout 是 ground truth feasibility probe。

### 3.3 WPC 训练时 sampling 公式

Eq. (3):
$$
F_{i,k} \sim \mathcal{U}\Big(0, \bar{F}_{i,k} \cdot \text{clip}\Big(\frac{p}{0.8}, 0, 1\Big)\Big)
$$

变量：
- $F_{i,k}$: 当前 window 的 sampled two-hand payload
- $\bar{F}_{i,k}$: 该 window 的 expert-labeled cap (kg)
- $p \in [0, 1]$: training progress，0 是开始 1 是结束
- 0.8: curriculum 在 80% training progress 时达到 full payload 强度
- $\mathcal{U}(0, \cdot)$: uniform distribution

**Intuition**: 这个公式有两个 staged 机制叠加：
1. **全局 curriculum**: 整体 payload 强度随 $p$ 线性升到 80% 后饱和
2. **per-window cap**: 即使 training 后期，w_{i,k} window 内 payload 也不超过 expert cap

也就是说，**简单 motion window 在训练后期可以见满 30kg，难 motion window 永远不超过 expert cap**。这避免了"难 motion + 高 payload"产生 useless gradient。

### 3.4 Force model 细节

paper 用一个简化但合理的 force model：
- Force 应用在 left 和 right wrist-roll links（不是 end-effector，因为 HEFT 不显式建模 grasp）
- Force 方向在 $12°$ cone around downward gravity 内随机采样
- Two-hand load 通过 random fraction split（不是 50/50）

$12°$ cone 的来源是 grasp 时手柄可能不完全垂直，物体可能左右倾斜。这个 cone 给了 policy 一些 robustness。

**Limitation 承认**：这个 model 不显式建模 grasp quality、object geometry、sliding、bracing、environment-supported contacts。所以 robot 实际拿 24kg kettlebell 时，force 不一定真的作用在 wrist-roll 上，可能在 hand 上偏一点。这是个 reasonable approximation 但不是 perfect。

相关工作的对比：
- [FALCON](https://arxiv.org/abs/2505.06776): explicit force adaptation，policy 接 force command
- [HALO (Wang et al. 2026)](https://arxiv.org/abs/2603.15084): 用 differentiable simulation 闭环 sim-to-real gap
- [Sim-to-real box loco-manip (Dao et al. ICRA 2024)](https://arxiv.org/abs/2310.02505): box-specific，不 generalize

HEFT 的 WPC 用 expert rollout 估 cap，比 FALCON 的 prescribed curriculum 自适应，比 HALO 的 differentiable sim 简单。

### 3.5 WPC 实验数据

Fig. 4 给出 L7 上 $\mathcal{D}_{\text{random}}$ 在不同 payload 下的 success rate：

| Payload | Ours (WPC) | w/o expert | TWIST2+FC |
|---|---|---|---|
| 0 kg | ~1.0 | ~1.0 | ~0.95 |
| 10 kg | ~1.0 | ~0.95 | ~0.75 |
| 20 kg | ~1.0 | ~0.90 | ~0.50 |
| 25 kg | 0.90 | 0.80 | 0.35 |
| 30 kg | 0.75 | 0.62 | 0.29 |

Table 1 给出无 payload 在 $\mathcal{D}_{\text{dynamic}}$ (high-dynamic motions) 上的结果：

| Policy | Success | MPJPE | Root Vel Err | Root AngVel Err |
|---|---|---|---|---|
| w/o expert | 0.64 | 0.060m | 0.872 m/s | 0.937 rad/s |
| Ours | 0.73 | 0.057m | 0.743 m/s | 0.968 rad/s |
| TWIST2+FC | 0.10 | 0.133m | 1.463 m/s | 1.294 rad/s |

**关键观察**:
1. WPC 在 high payload (25-30kg) success 显著高于 ablation
2. WPC 在 high-dynamic 无 payload 上 success 也更高 (0.73 vs 0.64)！这有点反直觉，但解释合理：WPC 的 expert-labeled cap 实际上起到了 **motion segmentation** 的作用，难 motion 段不让 policy 见 high payload，等于给了 policy 一个更稳定的训练分布，间接帮助 high-dynamic 上的学习
3. TWIST2+FC 在 high-dynamic 上几乎崩 (0.10 success)——这说明 FALCON-style global curriculum 配 TWIST2 tracking 在难度高的 motion 上会 fail，因为 global curriculum 让 policy 在难 motion 上也尝试高 payload，导致大量 failed rollout 污染 buffer

---

## 4. Teacher-Student RMA 结构

paper 用 [RMA (Kumar et al. 2021)](https://arxiv.org/abs/2107.04034) 的两阶段 distillation：

### 4.1 Privileged encoder

Teacher policy 输入：
$$
z_t = E_p(S_{\text{clean},t}, S_{\text{priv},t})
$$

其中 $S_{\text{priv},t}$ 包含 simulator-only information：
- tracking errors（teacher 知道自己跟踪得怎样）
- contact signals（哪些 link 在接触地面、接触力多大）
- payload state（当前真实 force 大小、方向）
- randomization variables（mass, friction 等 domain randomization 的 ground truth）

$z_t$ 是 256-D latent，编码了"环境当前 hidden 参数"。

### 4.2 Adapter (student)

部署时 adapter 只用 deployable observation：
$$
\hat{z}_t = E_a(S_{\text{prop},t}, S_{\text{raw},t})
$$

Adapter supervision 是 L2 loss:
$$
\mathcal{L}_{\text{adapt}} = \| \hat{z}_t - z_t \|_2^2
$$

变量含义：
- $\hat{z}_t$: adapter 从历史 proprioception + raw VR 推断的 latent
- $z_t$: privileged encoder 给出的 ground-truth latent
- L2 loss 让 adapter 学会从可观察历史反推环境参数

**Intuition**: 这是经典的 system identification 思路。robot 不知道现在提多重，但可以通过历史 joint torque、joint velocity、IMU 反应推出来。例如：在 25kg load 下，wrist 关节 motor 要输出更大电流才能 hold 住 reference pose，这个 current draw 是 prop observable 的，adapter 学会读这个信号反推 payload。

### 4.3 三阶段训练流程

paper 在 Sec. 3.4 描述了三阶段：

**Stage 1: Teacher training (PPO)**
- Actor 输入: deployable obs + privileged latent $z_t$
- Critic 输入: deployable + privileged + critic-priv
- 用 PMG (paired raw/clean) + WPC (windowed payload)
- 优化到 5×10⁹ frames (Table D.2)

**Stage 2: Adapter distillation**
- Freeze teacher
- 训 adapter $E_a$ 去匹配 $E_p$ 输出
- Adapter loss weight 0.2 (Table D.2)

**Stage 3: Student fine-tuning**
- Student actor 用 adapter 预测的 $\hat{z}_t$
- 继续 PPO 优化 student
- 这个 stage 让 student 适应自己的 adapter 误差

**部署**: 只用 student actor + adapter，privileged encoder 和 simulator info 全部移除。

### 4.4 Network architecture (Table D.1)

| Module | Input | Architecture |
|---|---|---|
| Teacher actor | dep obs + $z_t$ | MLP [1024, 1024, 512] |
| Student actor | dep obs + $\hat{z}_t$ | MLP [1024, 1024, 512] |
| Critic | dep + priv + critic-priv | MLP [1024, 512, 512] |
| Privileged encoder | clean ref + priv obs | MLP [512] → 256-D |
| Adapter | dep obs only | MLP [1024, 512] → 256-D |
| Action dist | mean + learned σ | Diagonal Gaussian |

注意 actor 用了 **residual joint-position command**——这意味着 actor 输出的是相对于某个 baseline (可能是上一帧 action 或 PD setpoint) 的 delta。这跟 [DeepMimic](https://arxiv.org/abs/1804.02717) 里的 PD target 输出思路类似，residual 让 fine-grained control 更容易学。

---

## 5. Reward Design 深度解析 (Table D.3)

reward 形式是统一的 exponential kernel:
$$
r_k = w_k \exp\left(-e_k / \sigma_k\right)
$$

变量：
- $w_k$: term weight
- $e_k$: tracking error
- $\sigma_k$: temperature，控制 reward 随 error 衰减的陡度

$\sigma$ 小 → reward 在 error 增大时快速掉到 0，鼓励精确；$\sigma$ 大 → reward 更宽容。

### 5.1 Tracking terms

| Term | $w$ | $\sigma$ | Error $e$ |
|---|---|---|---|
| Root position | 0.5 | 0.3 | $\|p_{\text{root}} - \hat{p}_{\text{root}}\|_2$ |
| Root rotation | 0.5 | 0.4 | $\|\text{Log}(q_{\text{root}}^{-1} \hat{q}_{\text{root}})\|_2$ |
| Root lin vel | 1.0 | 1.0 | $\|R^\top v_{\text{root}} - \hat{R}^\top \hat{v}_{\text{root}}\|_2$ |
| Root ang vel | 1.0 | 3.0 | $\|R^\top \omega_{\text{root}} - \hat{R}^\top \hat{\omega}_{\text{root}}\|_2$ |
| Keypoint pos | 2.0 | 0.3 | $K^{-1} \sum_j \|x_j - \hat{x}_j\|_2$ |
| Keypoint rot | 2.0 | 0.4 | $K^{-1} \sum_j \|\text{Log}(q_j^{-1} \hat{q}_j)\|_2$ |
| Joint pos | 1.0 | 0.5 | $n_q^{-1} \sum_l \|q_l - \hat{q}_l\|$ |
| Joint vel | 0.5 | 3.0 | $n_q^{-1} \sum_l \|\dot{q}_l - \hat{\dot{q}}_l\|$ |

**关键点**:
1. **Keypoint 用 13 个 sparse bodies** (head, hands, upper-limb links, hip-yaw, knee, ankle-roll)。这比 [UNiT](https://arxiv.org/abs/2509.xxxxx) 或 [PHC (Luo et al. 2023)](https://arxiv.org/abs/2309.08632) 的 dense keypoint 更紧凑，减少 reward 冗余
2. **Root lin vel 和 ang vel 用 body frame**: $R^\top v$ 是把世界系速度转到 body frame。这很重要因为 humanoid 的"前进"在 body frame 里是固定的，世界系下"前进"方向会随 yaw 变
3. **Keypoint pos 权重 2.0 最高**: 说明 paper 强调 end-effector 跟踪精度——这对于 carry 任务很关键，因为手的位置直接影响 grasp 稳定性

### 5.2 Regularizers

| Term | $w$ | Description |
|---|---|---|
| Joint vel reg | $5 \times 10^{-4}$ | $-\sum_l \dot{q}_l^2$ |
| Action rate | 0.02 | $-\|a_t - a_{t-1}\|_2^2$ |
| Joint pos limit | 1.0 | 在 90% soft limit 外惩罚 |
| Joint torque limit | 0.01 | 在 75% actuator torque limit 外惩罚 |
| Survival | 3.0 | 没终止就给 1 |

注意 **survival reward = 3.0** 是所有 term 里最高的，这意味着 paper 非常强调 "不 fall"。在 heavy payload 下，robot 很容易因为 CoM 偏移 fall，survival reward 把"活着"放在"跟得准"之前。

### 5.3 Contact reward

- Reference foot air time ($w=5.0$): 用 clean reference 的 foot-contact state 做 sparse timing reward，鼓励 swing/stance phase 一致
- Dense foot-contact reward ($w=1.0$): contact mismatch 给 -1，matched state 有 height-shaped penalty

contact reward 设计很关键，因为 humanoid 的步态本质是 contact sequencing。这个 reward 鼓励 policy 在正确时机抬脚/落地，但允许小幅 height variation。

---

## 6. 实验设计与 Robot 平台

### 6.1 L7 humanoid

- 175 cm, 65 kg, 29 actuated joints
- Tsinghua + RobotEaera + Shanghai Qizhi 联合
- 类似 [Unitree H1](https://www.unitree.com/h1) 但更接近 adult human scale

### 6.2 三个 evaluation set

| Set | 来源 | 用途 |
|---|---|---|
| $\mathcal{D}_{\text{VR}}$ (8 clips, 52s) | held-out VR teleop | 评估 noisy VR tracking |
| $\mathcal{D}_{\text{dynamic}}$ (100 clips, 8.2min) | SEED 高 dynamic 子集 | 评估 high-dynamic 无 payload |
| $\mathcal{D}_{\text{random}}$ (100 clips, 21.6min) | SEED uniform sample | 评估 general tracking + payload |

$\mathcal{D}_{\text{dynamic}}$ 的 selection 用 Eq. (A.1) 的 12 维 dynamics feature vector $\phi_i$，包括:
- $\bar{v}_{xy}, v_{xy}^{\max}$: 水平速度均值和最大
- $v_{\text{root}}^{95}, a_{\text{root}}^{95}$: root 速度和加速度的 95 分位
- $\Delta z, v_z^{\max}$: root 高度变化和最大垂直速度
- $\text{rms}(\dot{\psi}), \dot{\psi}^{95}$: yaw rate RMS 和 95 分位
- $\text{rms}(\dot{q}), \dot{q}^{95}, \text{rms}(\ddot{q}), \ddot{q}^{95}$: 关节速度/加速度统计

每个 feature 做 percentile normalization (Eq. A.2)：
$$
\tilde{\phi}_{i,k} = \text{clip}\left(\frac{\phi_{i,k} - P_5(\phi_{\cdot,k})}{P_{95}(\phi_{\cdot,k}) - P_5(\phi_{\cdot,k})}, 0, 1\right)
$$

动态 score: $d_i = \sum_k w_k \tilde{\phi}_{i,k}$，按 $d_i$ 排序取 top 100。这个 methodology 比纯 manual 挑选更系统化，避免 cherry-picking。

### 6.3 真实部署任务 (Fig. 5, Table E.1)

| Panel | Task | Object | Mass | 关键挑战 |
|---|---|---|---|---|
| (a) | Pickup + place | backpack | 5kg | 地面拾取 + 椅子放置 |
| (b) | Push | wheeled rack | unmetered | 持续水平推力 |
| (c) | Lift + carry | small desk | 8kg | bulky, posture 变化 |
| (d) | Pickup + carry | loaded basket | 5kg | 多阶段 tabletop handling |
| (e) | Asymmetric carry | water bottle | 10kg | 两手不均 |
| (f) | Walk | 2× kettlebell | 24kg total | 双手 load + locomotion |
| (g) | Squat | 2× kettlebell | 24kg total | 大垂直 CoM motion + load |

最 impressive 的是。一个 policy 同时处理 locomotion (f) 和 squat (g)，且都在 24kg (37% body mass) 下成功。这超过了 [HALO](https://arxiv.org/abs/2603.15084) 之类专门为 heavy load 设计的工作。

---

## 7. Architecture 全景图 (Fig. 2 解析)

Fig. 2 是 paper 最 dense 的一张图，我把它拆开看：

### 7.2 (a) Data preparation pipeline
- Mocap library (SEED, 100STYLE, LaFAN1) → retarget → $S_{\text{clean}} = S_{\text{raw}}$ (clean 数据双流一致)
- VR teleop → SMPL-X → RoHM diffusion → reconstructed motion → retarget → $S_{\text{clean}}$
- VR teleop → SMPL-X → 直接 retarget → $S_{\text{raw}}$ (保留 noise)
- 配对得到 $(S_{\text{raw}}, S_{\text{clean}})$ pairs

### 7.3 (b) WPC labeling
- 用 expert $\pi_E$ 在每个 5s window 上 rollout
- 30kg → 0kg 步长 5kg search → cap $\bar{F}_{i,k}$
- 形成每个 motion 的 cap 序列

### 7.4 (c) Training loop
- 采样 motion + windowed payload schedule
- Teacher: actor (dep obs + priv latent) + critic (dep + priv + critic-priv)
- Student: adapter (dep obs only) → $\hat{z}$
- 三阶段训练 (Sec. 4.3)

### 7.5 (d) Deployment
- 移除所有 privileged signal
- 只保留 student actor + adapter
- 实时吃 raw VR stream → joint command

---

## 8. Limitations 与未来方向

paper 在 Sec. 6 诚实承认了三个 limitation：

### 8.1 离线数据依赖
- 需要离线 reconstructed VR reference
- 需要 expert rollout labels
- 换 robot / tracker / motion library 都要重跑数据 pipeline

未来方向：online reconstruction (e.g. real-time diffusion 像 [consistency models](https://arxiv.org/abs/2303.01469))，或 self-labeling expert (RL bootstrap 自己的 cap)。

### 8.2 Force model 简化
- 只把 payload 当 wrist force
- 不建模 grasp quality, object geometry, sliding, bracing

未来方向：结合 grasp network 或 tactile sensing，把 object wrench 作为真正 6D wrench 而不是 downward force。也可以参考 [HALO](https://arxiv.org/abs/2603.15084) 用 differentiable simulation 学 contact-rich interaction。

### 8.3 单平台验证
- 只在 L7 上验证

未来方向：扩展到 [Unitree H1](https://www.unitree.com/h1), [Booster T1](https://www.boosterx.com/), [Figure 01](https://www.figure.ai/) 等，验证 cross-morphology transfer。

---

## 9. 我的延伸思考与相关联想

### 9.1 与 Foundation Model 趋势的对比
当前 humanoid foundation controller (e.g. [NOEMO](https://arxiv.org/abs/2505.07148), [Pulse](https://arxiv.org/abs/2505.07148)) 倾向"万物归一"——用大数据 + 多任务训一个 universal policy。HEFT 反其道而行，强调 **asymmetric training signal pairing**——同一段 motion 给 actor 看 noisy 版、给 critic 看 clean 版。这是个小而精的 trick，可以 plug-in 到 foundation controller 上。

### 9.2 Privileged critic 思路溯源
privileged information 在 RL 里历史悠久：
- [Asymmetric Actor-Critic (Pinto et al. 2017)](https://arxiv.org/abs/1710.06542): critic 看全 state
- [RMA (Kumar et al. 2021)](https://arxiv.org/abs/2107.04034): privileged encoder + adapter distillation
- [Privilege Critic in Dexterous Manip (Handa et al. 2023)](https://arxiv.org/abs/2305.20010): similar idea
- [DayDreamer (Wu et al. 2022)](https://arxiv.org/abs/2206.15476): world model as privileged signal

HEFT 的贡献是把 paired reference (raw + clean) 引入这个 framework，对应的是 vision 里的 noise2noise 思路——但 noise2noise 用 noisy target 监督 noisy input，HEFT 是 noisy input vs clean target，更接近 noise2clean (CT scan 重建里的 common setup)。

### 9.3 与 Diffusion Policy 的联系
[Diffusion Policy (Chi et al. 2023)](https://diffusion-policy.cs.columbia.edu/) 用 diffusion 直接学 action distribution，自带 multi-modal 和 noise robustness。如果用 diffusion policy 替换 HEFT 的 Gaussian actor，可能天然更 robust to VR noise——但 latency 会增加。这是个值得探索的方向。

### 9.4 WPC 的更一般意义
WPC 的本质是 **per-instance difficulty-aware curriculum**。这个 idea 可以推广到:
- Terrain curriculum: 不同 terrain patch 给不同 roughness cap (e.g. [ANYmal](https://www.anycaptain.com/) 的 per-terrain curriculum)
- Velocity curriculum: 不同 motion 类型给不同速度 cap
- Contact curriculum: 不同 interaction phase 给不同 contact force cap

本质上是个 **task-difficulty oracle** 思路，比单一 global curriculum 更 sample efficient。

### 9.5 Sim-to-real 的 latent adaptation
HEFT 用 256-D latent 编码 payload + contact + randomization。如果让 latent 显式包含更多物理量（object mass, CoM offset, friction），并用一个 [tactile sensor](https://arxiv.org/abs/2306.08320) 反推，可能直接 close the loop。

---

## 10. 总结

HEFT 是一篇很"工匠"的 paper。它没提出新 RL algorithm，没造新 backbone，没声称是 foundation model——它把 humanoid teleop 在 full-size 平台 + heavy payload 下的两个真实痛点拎出来，用 paired data training 和 per-window expert labeling 两个工程化技巧解决，并在 24kg 真实 load 上验证。

值得借鉴的设计 pattern:
1. **Asymmetric reference pairing**: training actor 见 noisy, critic + reward 用 clean
2. **Expert rollout as feasibility oracle**: 用强 policy 估 per-window cap
3. **Windowed curriculum with staged ramp**: 双重 curriculum (global progress × per-window cap)
4. **RMA-style latent distillation**: privileged encoder → adapter 通过历史推 latent
5. **Residual joint command**: actor 输出 delta，fine control 容易学

我个人觉得最 promising 的延伸方向是 **把 PMG 推广到 general sensor noise modeling**——任何 robot 用 noisy sensor 给指令的场景（视觉 teleop、joystick、语音指令）都可以用 paired clean/noisy data training 这个 trick。这或许是个真正能 generalize 的 methodology contribution。

---

**相关 reference 链接汇总**:
- HEFT Project Page (paper 中链接，可能是 https://heft-homepage.github.io 或类似)
- [RoHM (CVPR 2024)](https://virtualhumans.mpi-inf.mpg.de/papers/zhang2024rohm/zhang2024rohm.pdf)
- [SONIC (2026)](https://arxiv.org/abs/2511.07820)
- [TWIST2 (2025)](https://arxiv.org/abs/2511.02832)
- [FALCON (2025)](https://arxiv.org/abs/2505.06776)
- [OmniH2O (2024)](https://arxiv.org/abs/2406.08858)
- [HumanPlus (2024)](https://arxiv.org/abs/2406.10454)
- [RMA (2021)](https://arxiv.org/abs/2107.04034)
- [PPO (2017)](https://arxiv.org/abs/1707.06347)
- [DeepMimic (2018)](https://arxiv.org/abs/1804.02717)
- [AMP (2021)](https://arxiv.org/abs/2105.01699)
- [Diffusion Policy (2023)](https://diffusion-policy.cs.columbia.edu/)
- [AMASS](https://amass.is.tue.mpg.de/)
- [SMPL-X](https://smpl-x.is.tue.mpg.de/)
- [AvatarPoser (ECCV 2022)](https://arxiv.org/abs/2207.09630)
- [Open-Television (2024)](https://arxiv.org/abs/2407.01512)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [HALO (2026)](https://arxiv.org/abs/2603.15084)
- [SEED Dataset](https://bones.studio/datasets/seed)
- [100STYLE](https://github.com/sebastianstarke/AI4Animation)
- [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)

如果你想深入某一个部分（比如 RoHM 的 diffusion 细节、RMA distillation 的 convergence properties、或 reward shaping 的 sensitivity analysis），我可以再展开讲。
