---
source_pdf: DexMachina.pdf
paper_sha256: 7b67a6ad37891526024d9c93e5561cc7fd0883b1f20a01e7bd4da197402d821b
processed_at: '2026-08-03T20:35:51-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DexMachina 用人话讲

## 1. 这篇 paper 想干嘛

想象你拿到一段视频: 一个哥们儿用双手打开 waffle iron, 然后翻过来再合上。你想让一个机器人手把这个动作"复刻"出来。

听起来简单, 实际巨难。三个原因:

**第一, 机器人手跟人手长得就不一样**。人有 21 个 DoF, Allegro hand 才 16 个, Inspire hand 才 6 个。人手食指能独立弯曲, 机器人可能三根手指联动。你硬把人的关节角度抄过去, 机器人手会穿模、捏不住东西。

**第二, 双手 + 长 horizon + articulated object 是地狱组合**。"长 horizon" 意思是任务有多个 phase: 比如"先把 waffle iron 抓稳, 然后左手松开, 右手单手开盖"。RL policy 一旦在 phase 1 没抓好, object 掉地, episode 立刻结束, 它永远看不到 phase 2 长啥样。99% 的 rollout 都白费。

**第三, 你不是要机器人"摆出人的姿势", 你要机器人"完成任务"**。这俩是两回事。Kinematic retargeting 是抄姿势, 看起来像人但 functional 上根本不动 object。DexMachina 要的是 functional retargeting: object 真的沿 demo 轨迹走。

所以这 paper 的核心一句话: **给一段人手操作 demo, 学一个机器人 policy, 真的能把 object 操出同样的轨迹**。

---

## 2. 方法是怎么解决的 — 三招组合拳

### 招数一: Action space 不要从零开始学

机器人有几十个 joint, RL 在这么高维空间里随机探索, 一辈子也学不出来。作者说: "我手上已经有人手 demo, 我先把人手 motion 通过一个叫 AnyTeleop 的 kinematic retargeting 算法映射到机器人 joint, 这给我一个很好的参考轨迹 $\mathbf{q}_t$。然后 policy 不要输出 absolute joint angle, 而是在 $\mathbf{q}_t$ 上输出 **residual**。"

具体分两类 joint:

**Wrist (6 DoF, 3 translation + 3 rotation)** 用 residual:
$$q_t^{\mathrm{wrist-T}} = \mathbf{q}_t[\mathcal{T}_w^T] + s_T \cdot a_t^{\mathrm{wrist-T}}$$

$\mathbf{q}_t[\mathcal{T}_w^T]$ 是从人手 retarget 来的 reference wrist 位置, $a_t^{\mathrm{wrist-T}} \in \mathbb{R}^3$ 是 policy 输出的小修正, $s_T$ 是 scale。Policy 只需要在 reference 附近微调, 不用从 random 起步。

**Finger** 反过来用 absolute:
$$q_t^{\mathrm{fingers}} = \ell_{\mathcal{T}_f} + \frac{u[\mathcal{T}_f] - \ell[\mathcal{T}_f]}{2} \cdot (a_t^{\mathrm{fingers}} + 1)$$

$\ell, u$ 是 joint 下限上限, $a_t^{\mathrm{fingers}} \in [-1, 1]^{|T_f|}$ 是 policy 输出。$a=0$ 对应中位, $a=1$ 对应上限, $a=-1$ 对应下限。意思是 finger 让 policy 自己决定 (因为人手 finger 和机器人 finger 对应关系很糊), wrist 沿 demo 走 (因为 wrist 大方向人是清楚的)。

这个 hybrid action 比 "all absolute" 或 "all residual" 都好, 见 Figure 8 ablation。直觉上: wrist 是任务大方向, 跟人走就行; finger 是细节执行, 给 policy 自由度自己摸。

类似思路在 residual policy learning 文献 (Silver, Johannink 2019, https://arxiv.org/abs/1812.06298) 里有, 这里用得很巧。

---

### 招数二: 给 policy "软指导", 不强制 mimic

光有 task reward 不够, policy 不知道"该怎么动"。作者从 demo 里提三种指导:

**(a) Motion imitation reward** (DeepMimic 思路, Peng et al. 2018, https://arxiv.org/abs/1804.02717):
$$r_{\mathrm{imi}} = \frac{1}{K}\sum_{i=1}^{K} \exp(-\beta_{\mathrm{imi}} \|\hat{x}_i - x_i\|_2)$$

$\hat{x}_i, x_i \in \mathbb{R}^3$ 是第 $i$ 个 hand link keypoint 的 achieved / reference 位置, $K$ 是 keypoint 总数, $\beta_{\mathrm{imi}}$ 控制对误差的敏感度 (越大越敏感)。这是 hand link 位置层面的 mimic, 比 joint angle mimic 更 embodiment-agnostic。

**(b) Behavior cloning reward**:
$$r_{\mathrm{bc}} = \frac{1}{J}\sum_{i=1}^{J} \exp(-\beta_{\mathrm{bc}} \|\hat{q}_i - q_i\|_2)$$

$\hat{q}_i, q_i$ 是 achieved / retargeted 的第 $i$ 个 joint 值, $J$ 是 joint 总数。Soft BC, 不需要 demo action label (只有 reference state), 通过拉 policy 接近 retargeted target 实现。

**(c) Contact reward** — 这个最细:

作者把 demo 里每一帧的"哪个手 link 碰 object 哪个 part"都近似出来, 得到 contact tensor $\mathcal{C} \in \mathbb{R}^{T \times N \times K \times 3}$ 和 validity mask $\mathcal{M} \in \{0,1\}^{T \times N \times K}$。$T$ 是 timestep, $N$ 是 object part 数, $K$ 是 hand link 数, 3 是 3D 位置。

然后定义 distance matrix $D^{(i,j)}$ 表示 policy 当前在 $(i,j)$ 对上的 contact 位置误差, 并做 mask-aware override:

$$D^{(i,j)} = \begin{cases} d_{\max}, & M_{\mathrm{demo}}^{(i,j)} \neq M_{\mathrm{policy}}^{(i,j)} \\ 0, & M_{\mathrm{demo}}^{(i,j)} = M_{\mathrm{policy}}^{(i,j)} = 0 \end{cases}$$

意思是: demo 说该碰但 policy 没碰 (或反之), 直接给 $d_{\max}$ 大惩罚; demo 说都不碰 policy 也没碰, distance 0 (对的)。

Final:
$$r_{\mathrm{con}} = \frac{1}{2NK}\left(\sum_{i,j} \exp(-\beta_{\mathrm{con}} D_{\mathrm{left}}^{(i,j)}) + \sum_{i,j} \exp(-\beta_{\mathrm{con}} D_{\mathrm{right}}^{(i,j)})\right)$$

左右手平均, $N$ 个 object part × $K$ 个 link 都过一遍。

直觉: 这是个"contact schedule"监督。Human demo 隐含告诉我们"什么时候哪个 finger 该碰 object 哪个 part", phase transition 时刻这个 schedule 是关键 hint。比如 grasp 阶段拇指要按住 box, open lid 阶段拇指松开食指抠 lid 边缘, schedule 不一样。

Total reward:
$$r_t = \lambda_{\mathrm{task}} r_{\mathrm{task}} + \lambda_{\mathrm{imi}} r_{\mathrm{imi}} + \lambda_{\mathrm{bc}} r_{\mathrm{bc}} + \lambda_{\mathrm{con}} r_{\mathrm{con}}$$

关键: $\lambda_{\mathrm{imi}}, \lambda_{\mathrm{bc}}, \lambda_{\mathrm{con}}$ 比 $\lambda_{\mathrm{task}}$ 小很多。意思是这些 auxiliary reward 是"软指导", 后期 policy 可以偏离 demo 找真正 functional 完成任务的动作。比如 Inspire hand (6 DoF) 跟人手差太远, 它学到的策略是"双手一起夹"而不是"人手单手抓", 这就靠 task reward 主导。

---

### 招数三: Virtual object controller curriculum — 真正的创新

**问题**: 前两招在 short task 上够用, 但 long-horizon articulated task 上还是炸。Policy 在 phase 1 一抓就掉, episode 立刻终止, 后面 phase 永远学不到。

**核心 idea**: 给 object 装一个"虚拟 PD controller", 一开始这个 controller 强力地把 object 沿 demo 轨迹拖, 让 policy 在旁边"看完整段任务怎么走"。然后 controller 力道慢慢 decay, policy 被迫接管。

具体: 给 object 加 7 个 virtual joint (6 DoF base pose + 1 DoF articulation), 用 PD controller actuate:
$$F = k_p (g_t - \hat{g}_t) - k_v \dot{\hat{g}}_t$$

- $g_t$: demo 里 timestep $t$ 的 object target state;
- $\hat{g}_t$: 当前 achieved object state;
- $\dot{\hat{g}}_t$: object 当前速度;
- $k_p$: spring gain (越大, virtual force 越强);
- $k_v$: damping gain (抑制震荡, critical damping 时 $k_v = 2\sqrt{k_p \cdot m}$)。

Gain 怎么 decay (Algorithm 1):

每个 PPO iteration 结束, 检查四个 reward 的 normalized 历史均值 $\mu_z$ (z ∈ {task, imi, bc, con}), $\bar{r}_z = R_z / L_{\max}$ (除以 max episode length 而不是实际 length, 这样短 episode 不会被误判):

```
if 所有 μ_z > σ_z (policy 学稳了):
    k_p ← k_p · φ_p  (指数衰减)
    if k_p ≤ 0.01: k_p = k_v = 0  (彻底关掉)
    k_v ← k_v · φ_v
```

**直觉拆解**:

阶段一 (k_p 大): object 被"拖着走", policy 怎么乱动都不会 drop, 它有机会看完整个 long-horizon task sequence。Auxiliary reward 引导 policy 学到"手在哪里、contact pattern 是什么"。

阶段二 (k_p 中): object 越来越多需要 hand 真实接触力支撑。Policy 必须 adjust motion 去"接住" object。这是个**连续过渡**, 不像开关那样 cliff, policy 不会突然崩。

阶段三 (k_p = 0): object 完全靠 hand 真实物理操作, policy 必须自己 form 出 task-completion 的策略。此时 task reward 主导 (因为 auxiliary reward 权重小), policy 可以偏离 demo motion 找真正 functional 的动作。

**为什么不在 hand 上加 teacher, 而在 object 上加**: 这是关键 twist。如果在 hand action 上加 teacher (像 residual RL 那样), policy action space 被锁死, 没法 explore alternative strategy。在 object 上加 teacher, hand action 完全 free, policy 可以学 hardware-specific 策略 (Inspire 学双手夹, Allegro 学单手抓)。同时 task 难度被 continuous 降低。

类似思路的祖宗: Mordatch contact-invariant optimization (https://arxiv.org/abs/1406.2869), DeepMimic reference state init, Mao privileged action curriculum (https://arxiv.org/abs/2502.15442), 还有 trajectory optimization 里的 "warm-start with relaxed constraints"。

---

## 3. 为什么 task reward 用乘积不用和

$$r_{\mathrm{task}} = \exp(-\beta_{\mathrm{pos}} d_{\mathrm{pos}}) \cdot \exp(-\beta_{\mathrm{rot}} d_{\mathrm{rot}}) \cdot \exp(-\beta_{\mathrm{ang}} d_{\mathrm{ang}})$$

- $d_{\mathrm{pos}} = \|\hat{g}_t^P - g_t^P\|_2$: object 位置误差, 3D L2 距离;
- $d_{\mathrm{rot}} = 2\cos^{-1}(|\langle \hat{g}_t^R, g_t^R \rangle|)$: object rotation 误差, 用 unit quaternion 内积算 geodesic distance, 单位 radian, 范围 $[0, \pi]$。绝对值 $|\cdot|$ 处理 quaternion double-cover ($q$ 和 $-q$ 同旋转);
- $d_{\mathrm{ang}} = \|\hat{g}_t^J - g_t^J\|_2$: articulation joint angle 误差 (比如 waffle iron 开合角度);
- $\beta_{\mathrm{pos}}, \beta_{\mathrm{rot}}, \beta_{\mathrm{ang}}$: 三个 scalar weight, 控制"desirable error scale", 越大对误差越敏感。

**为什么乘积**: 如果是加权和 $\lambda_1 r_{\mathrm{pos}} + \lambda_2 r_{\mathrm{rot}} + \lambda_3 r_{\mathrm{ang}}$, policy 会"摆烂"——position match 得好但 rotation 完全错, 总分还是不错。乘积里只要任何一项 error 大, 整个 reward 趋于 0, 强制 policy 必须三头都顾。这在 OpenAI Dactyl 的 Rubik's cube 工作 (https://arxiv.org/abs/1910.07143) 和 ObjDex (https://arxiv.org/abs/2411.04005) 里也是这个思路。

---

## 4. 实验结果一图概括 (Figure 3)

7 个 demo task, 4 个 representative hands, 加 5 seeds 平均。结果是:

- **Short task** (前 3 列 Ketchup-100, Box-170, Mixer-170): Task + Aux reward 已经能 work, curriculum 加成小;
- **Long task** (后 4 列 Box-300, Mixer-300, Notebook-300, Waffle-300): 没 curriculum 的方法几乎全 fail, DexMachina 大幅领先;
- **Kinematics Only**: 看起来像人手但 functional 几乎不动 object;
- **ObjDex baseline**: 作者用 Genesis 重新实现 (12k envs vs original 2k envs), 比原 paper 报告还高 (Ketchup-100 从 41.2% → >90%);
- **ManipTrans baseline**: decay gravity/friction 那种 curriculum 不稳定, 后期 reward 掉下去恢复不了。

---

## 5. Hand-specific strategies (Figure 4) — 这个图最直观

| Task | Hand | 学到的策略 |
|------|------|-----------|
| Notebook-300 | XHand | 左手 hold, 右手 close cover (跟人一样) |
| Notebook-300 | Inspire | 双手一起 hold, 一起 close cover (单手不够稳) |
| Mixer-300 | Allegro | 长手指直接 close lid, wrist 几乎不动 |
| Mixer-300 | Schunk | 手指短, wrist 大幅 motion 补偿 |

这说明 policy **适应硬件约束**。Inspire hand 6 DoF, 跟人手差太远, 它自己发明了"双手夹"策略, 比硬抄人手 motion 好。这正是 auxiliary reward 软指导 + curriculum 给的 explore 空间带来的。

---

## 6. Hand embodiment analysis (Figure 5) — 真正给做 hardware 的人的启示

加入 Ability hand 和 DexRobot Hand 一起测:

- **Allegro** 虽然长得不 anthropomorphic, 但手指长, in-air manipulation 稳, 学得最快最好;
- **Inspire / Ability / Schunk** size 差不多, 但 **Schunk 表现更好**, 因为它有 actuated fingertip + foldable palm;
- **Less-actuated hand** 学到的策略更不 human-like, 因为 human reference 对它不 feasible, 必须找 alternative;
- **Size 相似度不如 DoF / actuation 重要**。

启示: 设计 dexterous hand 时, **机构 capability > anthropomorphism**。这与 OpenAI Dactyl 选 Shadow Hand、RoboPianist (https://robopianist.github.io) 的发现一致。

---

## 7. 一句话总结方法

> **DexMachina = hybrid action (用 demo 当 wrist prior) + 三种 auxiliary reward (motion + BC + contact) + virtual object controller curriculum (给 object 装 PD teacher, gain 指数 decay 让 policy 平滑接管)**。

核心 insight: **把 teacher 放在 object 上而不是 hand action 上**, hand action space 完全 free, policy 能学 hardware-specific 策略; 同时 task 难度被 continuous 降低, long-horizon 探索问题解决。

---

## 8. Limitation & 我自己的延伸

**作者承认的 limitation**:
- State-based input, 实际部署需要 vision-based RL 或 teacher-student distillation;
- 依赖高质量 mocap 数据 (ARCTIC 那种), 收集成本高;
- Simulator hand asset 的 mass / inertia / collision 是估计的, 不能完全反映真实 hardware;
- 没做 real-world 实验。

**我延伸的 intuition**:

1. **"Decaying assistance on object"** 这个 inductive bias 本质是优化 landscape smoothing。Long-horizon task 的 reward landscape 有大量 local minima (早期 drop 之后所有 trajectory fail)。Virtual controller 在 demo trajectory 周围 form 一个 basin of attraction, landscape 被平滑化, policy gradient 能稳定收敛。Gain decay 让 basin 慢慢 shrink, policy 被"逼"自己 form basin。这跟 trajectory optimization 里的 "convex relax then tighten" 是一个套路, 但 port 到 RL 里需要 adaptive gain schedule based on reward stability, 这里做得干净。

2. **跟 LLM RLHF 的呼应**: Expert iteration (https://arxiv.org/abs/2305.06383) 先用 strong model 生成 demo, weak model 模仿, 然后提高 weak model 自主性。Iterative DPO 一开始用 strong reference, 慢慢让 weak policy dominate。都是同一 inductive bias 在不同 modality 的应用。

3. **Functional vs kinematic retargeting 的本质**: Kinematic 解 $\min_\theta \sum_t \|FK(\theta_t) - x_t^{\rm human}\|_2$ (pure geometric), Functional 解 $\min_\pi \sum_t F(g_t(\pi), g_t^{\rm demo})$ (dynamical + contact)。后者难但 task semantics 保留完整。DexMachina 选题价值在这里。

4. **作为 hand design benchmark**: Figure 5 实际是 "hardware capability via RL training performance" benchmark 雏形。设计新 hand, 丢进 DexMachina, 看 (a) final performance, (b) learning efficiency, 就能反推 hardware capability。可能成为 standard hand benchmark 之一。

---

## 9. Reference links 全套

**DexMachina**:
- 项目页: https://project-dexmachina.github.io

**Dataset / Simulator / RL lib**:
- ARCTIC dataset: https://arctic-project.github.io
- Genesis physics engine: https://github.com/Genesis-Embodied-AI/Genesis
- rl-games: https://github.com/Denys88/rl_games
- PPO paper: https://arxiv.org/abs/1707.06347

**Baseline & 相关 dexterous manipulation**:
- AnyTeleop (kinematic retargeting): https://arxiv.org/abs/2307.04577
- ObjDex: https://arxiv.org/abs/2411.04005
- ManipTrans: https://arxiv.org/abs/2503.21860
- DeepMimic: https://arxiv.org/abs/1804.02717
- Dactyl (OpenAI Rubik's cube): https://arxiv.org/abs/1910.07143
- Rajeswaran demo RL: https://arxiv.org/abs/1709.10087
- DexArt benchmark: https://dexart.github.io
- Dextreme (sim-to-real): https://arxiv.org/abs/2210.13767
- RoboPianist: https://robopianist.github.io
- xSkill: https://arxiv.org/abs/2307.09934
- VideoDex: https://arxiv.org/abs/2212.04498
- DexCap: https://arxiv.org/abs/2403.07788
- ArtiGrasp: https://arxiv.org/abs/2310.01203

**Robot hands**:
- Inspire Hand: https://inspire-robots.store
- Allegro Hand: https://www.allegrohand.com
- XHand (ROBOTERA): https://www.robotera.com
- Schunk SVH: https://schunk.com
- Ability Hand (PSYONIC): https://www.psyonic.io

**Curriculum / optimization 思路**:
- Contact-invariant optimization (Mordatch): https://arxiv.org/abs/1406.2869
- Privileged action curriculum: https://arxiv.org/abs/2502.15442
- Residual policy learning: https://arxiv.org/abs/1812.06298
- FoundationPose (ADD-AUC metric lineage): https://arxiv.org/abs/2312.00780
- PoseCNN: https://arxiv.org/abs/1711.00199

---

**一句话总结**: 这 paper 把"给一段人手 demo 学机器人操作"这个问题用"在 object 上挂个会 decay 的虚拟 PD 老师"的方式漂亮地解了, 让 policy 从观摩者平滑过渡到操盘者, 还能根据自己硬件情况学出 hardware-specific 策略。

---

# DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation

## 1. Paper 高层 intuition

这篇 paper 解决一个很具体的问题: 给定一段**人类双手操作 articulated object 的 demonstration** (例如打开 waffle iron, 翻开笔记本), 把它"翻译"成一段**robot dexterous hand 的 functional policy**, 并且要求这个 policy 真的能把 object 沿着 demo 的轨迹操作出来——不是仅仅 mimick 出 human-like 的姿态, 而是要 task 真的成功。

作者把这个叫做 **Functional Retargeting**:

> 给定 object $\eta$, 人类 demo $\mathcal{D}^{\eta} = \{G, H\}$ (T timesteps 的 object states $G$ + hand poses $H$), 以及一对 dexterous robot hands $\zeta$, 学习一个 policy $\pi_\theta^{\eta, \zeta}$ 使得:
> $$\pi_\theta^{\eta, \zeta} = \arg\min_\theta \sum_{t=1}^{T} F(\hat{g}_t, g_t)$$
> 其中 $F$ 是 object state 之间的 distance, $\hat{g}_t$ 是 achieved state, $g_t = \{g_t^P, g_t^R, g_t^J\}$ 分别是 position / rotation / articulation joint angle。

这里 articulate object 的 state 分解成 part pose + revolute joint angle, 比单纯的 rigid object tracking 更难, 因为需要同时匹配 articulated part 的几何位置和 internal joint angle。

核心 motivation 是: 长 horizon bimanual task 有三个 bug:
- **High-dim action space** (两只 16-DoF 的 hand = 32+ DoF);
- **Spatiotemporal discontinuity** (比如一只手抓稳后另一只手要中途换 grip 去 open lid, 这种 phase transition 极易 catastrophic failure);
- **Embodiment gap** (human hand 的 motion 不能直接 map 到 robot hand 的 feasible action)。

DexMachina 的核心 trick 是 **virtual object controllers with decaying strength** 的 curriculum: 一开始 virtual PD controller 直接"抱着" object 走 demo 轨迹, policy 在旁边看着学; 然后 controller gain 指数 decay, policy 慢慢 take over。这种想法在 motion planning warm-start、DeepMimic-style curriculum、privileged action learning 里都有类似的祖宗, 但用到 articulated object 的 functional retargeting 上是新的。

项目页: https://project-dexmachina.github.io

---

## 2. Method 细节拆解

### 2.1 Task Reward — 乘积形式强制 balance

每个 timestep 的 object state 分成 3 个 channel, 各自的 distance:

$$d_{\mathrm{pos}} = \|\hat{g}_t^P - g_t^P\|_2, \quad d_{\mathrm{rot}} = 2\cos^{-1}(|\langle \hat{g}_t^R, g_t^R \rangle|), \quad d_{\mathrm{ang}} = \|\hat{g}_t^J - g_t^J\|_2$$

变量含义:
- $\hat{g}_t^P, g_t^P \in \mathbb{R}^3$: achieved / target object position (3D);
- $\hat{g}_t^R, g_t^R \in \mathbb{R}^4$: achieved / target object rotation, 用 unit quaternion 表示;
- $\langle \cdot, \cdot \rangle$ 是 quaternion 内积, $|\cdot|$ 取绝对值是为了处理 quaternion double-cover ($q$ 和 $-q$ 表示同一旋转);
- $2\cos^{-1}(|\langle q_1, q_2 \rangle|)$ 就是 geodesic distance on $SO(3)$, 单位是 radian;
- $g_t^J, \hat{g}_t^J$: target / achieved articulation joint angle (对 ARCTIC 这类 articulated object 一般是 1 个 revolute joint, 比如 waffle iron 的开合角度);
- $\beta_{\mathrm{pos}}, \beta_{\mathrm{rot}}, \beta_{\mathrm{ang}}$ 是 3 个标量 weight, 控制 desirable error scale, 越大对误差越敏感。

Total task reward 是三个 channel 的**乘积**而不是和:

$$r_{\mathrm{task}} = r_{\mathrm{pos}} \cdot r_{\mathrm{rot}} \cdot r_{\mathrm{angle}} = \exp(-\beta_{\mathrm{pos}} d_{\mathrm{pos}}) \cdot \exp(-\beta_{\mathrm{rot}} d_{\mathrm{rot}}) \cdot \exp(-\beta_{\mathrm{ang}} d_{\mathrm{ang}})$$

**Intuition**: 乘积形式迫使 policy 不能在某一维上"摆烂"。如果是加权和, policy 可能选择 position match 得很好但 joint angle 完全错; 乘积里只要任何一个 channel error 大, 整个 reward 就趋近 0, 强制 policy 必须三头都顾。这种 multiplicative shaping 在 OpenAI Dactyl 的 Rubik's cube 工作、Rajeswaran 的 dexterous manipulation demo learning 里都见过类似思路。ObjDex (论文 [35]) 也是这种 design philosophy, DexMachina 沿用了它。

---

### 2.2 Data preprocessing — Kinematic retargeting + Contact approximation

先把人类 demo 的 MANO hand pose 通过 collision-aware kinematic retargeting (AnyTeleop 风格 [3], https://arxiv.org/abs/2307.04577) 映射到每个 dexterous hand 的 joint values $\mathcal{Q} \in \mathbb{R}^{T \times J}$ 和 keypoint positions $\mathcal{X} \in \mathbb{R}^{T \times K \times 3}$。

J = actuated joint 数 (例如 Allegro 16, Inspire 6+6 DoF 每只手), K = collision link 数。

关键 trick: pure kinematic retargeting 经常和 object penetrate, 所以作者把 retargeted joint values 作为 **soft control target** 在 simulation 里 replay, 同时 fix 住 object 在 demo target pose, 让 solver 自动 resolve penetration, 然后记录 achieved joint values 和 keypoint 作为 reference。这一步可以在 simulation 里高度并行 (T 个 timestep 互相独立), 见 Figure 6。

**Contact approximation** 更有意思。对每一对 object part mesh vertex $v_i^o$ 和 MANO hand mesh vertex $v_j^h$, 用 $L_2$ 距离阈值 $\gamma = 0.01$ (米) 找出 approximate contact vertex:
$$v_j^* = \arg\min_j \|v_i^o - v_j^h\|_2, \quad \text{if } \|v_i^o - v_j^h\|_2 < \gamma \text{, mark as contact}$$
然后用 farthest sub-sampling 把 contact 数量 cap 到 $N_c = 50$。最后把每个 contact point assign 给最近的 dexterous hand link center $\ell_m$, 求 mean 位置 $\bar{v}_m$。

输出:
- Contact tensor $\mathcal{C} \in \mathbb{R}^{T \times N \times K \times 3}$ (T = timesteps, N = object parts, K = hand collision links, 3 = 3D 位置);
- Validity mask $\mathcal{M} \in \{0,1\}^{T \times N \times K}$。

这个 contact 是 **per-link** granularity (而不是 per-vertex), 既节省 reward 计算 又对每个 hand link 提供"该不该碰、碰哪"的稀疏监督。

---

### 2.3 Hybrid Action Formulation — wrist residual + finger absolute

这是一个非常有意思的 design choice, 直接影响 sample efficiency。

记号:
- $a_t \in \mathbb{R}^J$: policy 输出, clip 到 $[-1, 1]$;
- $\mathbf{q}_t \in \mathbb{R}^J$: kinematic retargeting 得到的 reference joint values;
- $\mathcal{T}_f$: finger DOF 的 indices;
- $\mathcal{T}_w^T, \mathcal{T}_w^R$: wrist 6-DoF 的 translation / rotation indices, $|\mathcal{T}_w^T|=|\mathcal{T}_w^R|=3$;
- $s_T, s_R$: translation / rotation action 的 scale factor;
- $\ell, u \in \mathbb{R}^J$: lower / upper joint limits。

公式:
$$q_t^{\mathrm{wrist-T}} = \mathbf{q}_t[\mathcal{T}_w^T] + s_T \cdot a_t^{\mathrm{wrist-T}}$$
$$q_t^{\mathrm{wrist-R}} = \mathbf{q}_t[\mathcal{T}_w^R] + s_R \cdot a_t^{\mathrm{wrist-R}}$$
$$q_t^{\mathrm{fingers}} = \ell_{\mathcal{T}_f} + \frac{u[\mathcal{T}_f] - \ell[\mathcal{T}_f]}{2} \cdot (a_t^{\mathrm{fingers}} + 1)$$

**Intuition**: wrist (6-DoF base) 是动作的"骨架", demo 已经给了很好的 reference, policy 只需要 output **residual** 修正 (类似 residual policy / residual RL 文献的思路, e.g. Silver et al., Johannink et al. 2019, https://arxiv.org/abs/1812.06298); finger 是细节执行, demo 的 finger retargeting 不准 (embodiment gap 太大), 直接让 policy 在 joint limit 范围内输出 **absolute** target 更合理。

Ablation (Figure 8) 表明: 用 hybrid (wrist residual + finger absolute) 比 "all absolute" 或 "less-constrained wrist residual" 都好。这是因为它 **preconditioning 了 action space 的 prior**, 类似 transformer 里把 well-known embedding 当 init 而不是从 random 起步。

---

### 2.4 Auxiliary Rewards — Motion imitation + BC + Contact

#### Motion imitation reward
$$r_{\mathrm{imi}} = \frac{1}{K}\sum_{i=1}^{K} \exp(-\beta_{\mathrm{imi}} \|\hat{x}_i - x_i\|_2)$$

$\hat{x}_i, x_i \in \mathbb{R}^3$ 是第 $i$ 个 keypoint 的 achieved / reference 3D 位置, $K$ 是 keypoint 数。

这是 DeepMimic (Peng et al. 2018, https://arxiv.org/abs/1804.02717) 的 keypoint matching reward 思路, 让 hand 的 link 位置 (而不是 joint angle) 跟随 human reference。Keypoint 比 joint angle 更 embodiment-agnostic, 因为不同 hand 的 joint 配置不同但指尖位置在 task 上的功能意义是相似的。

#### Behavior cloning reward
$$r_{\mathrm{bc}} = \frac{1}{J}\sum_{i=1}^{J} \exp(-\beta_{\mathrm{bc}} \|\hat{q}_i - q_i\|_2)$$

$\hat{q}_i, q_i$: achieved / retargeted 的第 $i$ 个 joint value。

BC reward 是 soft 的 BC, 不需要 demonstration action label (因为只有 reference state, 没有 reference action), 通过拉 policy 靠近 retargeted joint target 实现。

#### Contact reward (这是这篇 paper 真正细致的地方)

定义距离矩阵 $D \in \mathbb{R}^{N \times K}$, 其中第 $(i, j)$ 项 $D^{(i,j)} = \|C^{(i,j)} - \hat{C}^{(i,j)}\|_2$ 表示 policy 的第 $j$ 个 link 与 object 第 $i$ 个 part 之间的 contact 位置 $L_2$ 距离。然后做 "mask-aware override":

$$D^{(i,j)} = \begin{cases} d_{\max}, & \text{if } M_{\mathrm{demo}}^{(i,j)} \neq M_{\mathrm{policy}}^{(i,j)} \\ 0, & \text{if } M_{\mathrm{demo}}^{(i,j)} = M_{\mathrm{policy}}^{(i,j)} = 0 \end{cases}$$

变量含义:
- $M_{\mathrm{demo}}^{(i,j)} \in \{0, 1\}$: demo 里第 $j$ 个 hand link 是否应该和第 $i$ 个 object part 接触;
- $M_{\mathrm{policy}}^{(i,j)} \in \{0, 1\}$: policy 当前 timestep 是否真的接触;
- $d_{\max}$: 一个大常数惩罚, 用在 "该接触却没接触" 或 "不该接触却接触了" 的情况;
- 当 demo 和 policy 都说"没接触", distance 设 0 (不惩罚, 因为都没碰就是对的)。

Final contact reward 对左右手平均:
$$r_{\mathrm{con}} = \frac{1}{2NK} \left( \sum_{i,j} \exp(-\beta_{\mathrm{con}} D_{\mathrm{left}}^{(i,j)}) + \sum_{i,j} \exp(-\beta_{\mathrm{con}} D_{\mathrm{right}}^{(i,j)}) \right)$$

**Intuition**: 这个 reward 本质上是一个 sparse "contact schedule" supervision。Human demo 隐含地告诉我们"什么时候哪个 finger 该碰 object 的哪个 part", policy 学的时候要把这个 contact pattern 复现出来。在 phase transition 时 (比如要从 grasp 切到 open lid), contact pattern 是关键 hint。

Total reward:
$$r_t = \lambda_{\mathrm{task}} r_{\mathrm{task}} + \lambda_{\mathrm{imi}} r_{\mathrm{imi}} + \lambda_{\mathrm{bc}} r_{\mathrm{bc}} + \lambda_{\mathrm{con}} r_{\mathrm{con}}$$

作者强调 $\lambda_{\mathrm{imi}}, \lambda_{\mathrm{bc}}, \lambda_{\mathrm{con}}$ 比 $\lambda_{\mathrm{task}}$ 小很多, 这样后期 policy 可以偏离 reference 去 optimize task (e.g. Inspire hand 因为只有 6 DoF, 学到的策略和 human demo 差很远, 见 Figure 4)。

---

### 2.5 Virtual Object Controllers — 核心 curriculum 创新这是整篇 paper 最有意思的部分, 需要详细 intuition。

**Motivation**: 长 horizon task 里, naive RL 经常在早期 phase 就 drop object, episode terminate, 永远看不到后续 phase。这让 policy 学到的都是 myopic 的策略, 无法 explore 到 "先抓稳 -> 中途换 grip -> open lid" 这种长 sequence。作者观察 (Section 4.3) 比如抓 box + open lid in-air, policy 一旦 drop 就 game over, 99% 的 rollout 都浪费在"试错 -> drop -> 死"的循环里。

**Solution**: 在 object 上 attach **virtual 1-DoF joints**:
- 6 个 virtual joints for base pose (3 translation + 3 rotation);
- 1 个 virtual joint for articulation (revolute)。

每个 virtual joint 由 **PD controller** actuate, 用 demo 的 $g_t$ 作为 control target。PD control law 是经典:
$$F = k_p (g_t - \hat{g}_t) - k_v \dot{\hat{g}}_t$$

其中 $k_p$ 是 spring gain, $k_v$ 是 damping gain。增益大时, object 被 virtual force 强力拉到 demo pose; 增益趋于 0 时, virtual force 消失, object 完全由真实物理 (gravity, hand contact force) 决定。

Curriculum scheduling (Algorithm 1):

**输入**: reward thresholds $\sigma_{\mathrm{task}}, \sigma_{\mathrm{imi}}, \sigma_{\mathrm{bc}}, \sigma_{\mathrm{con}}$; reward deques $D_z$ (history); 初始 gains $k_p, k_v$; decay ratios $\phi_p, \phi_v$; max episode length $L_{\max}$。

**Loop**:
1. 每个 PPO iteration 末, 对每个 done 的 environment:
   - 计算 achieved episode length $L$;
   - 计算 cumulative reward $R_z = \sum_t r_{z,t}$;
   - Normalize: $\bar{r}_z = R_z / L_{\max}$ (除以最大长度, 不是实际长度, 这样短 episode 不会因为累积量小而被误判为"差");
   - Append $\bar{r}_z$ 到 deque $D_z$;
2. 计算 deque 的 mean $\mu_z$;
3. 如果 $k_p = 0$ 就 continue (curriculum 结束);
4. **关键**: 如果所有四个 $\mu_z > \sigma_z$ (policy 在四个维度都"学稳了"), 则:
   - $k_p \leftarrow k_p \cdot \phi_p$ (指数 decay)
   - 当 $k_p \leq 0.01$ 时设 $k_p = k_v = 0$ (彻底关闭)
   - $k_v \leftarrow k_v \cdot \phi_v$

**为什么这个 work — intuition 拆解**:

(1) **Exploration bootstrap**: 一开始 $k_p$ 大, object 被"拖着走", policy 不论怎么乱动手都不会 drop。这让 policy 有机会 observe 整个 long-horizon task sequence, 而不是在第一秒就 game over。

(2) **Imitation phase**: virtual controller 在 actuate object 的同时, hand 也得跟着 motion reference (motion / BC reward 引导)。Policy 学到"如果我的手在哪、contact pattern 是啥, object 就能跟随 demo"。

(3) **Hand-off phase**: 当 $\mu_z$ 稳定高时 decay, object 越来越需要靠 hand 真实接触力来支撑, policy 必须 adjust motion 去"接住"object。这个过程是一个 **smooth transition**, 而不是 cliff, 因为 controller 的力是连续 decay 的。

(4) **Task priority**: 因为 auxiliary reward 权重小, curriculum 后期 task reward 主导, policy 可以偏离 reference motion 去找真正 functionally 完成任务的动作 (例如 Inspire hand 用双手夹住 object 而不是单手, 见 Figure 4)。

**为什么不用 ManipTrans 那种 "decay 物理参数" curriculum**: 作者的 ablation 表明 ManipTrans (decay gravity / friction / error threshold) 在 long-horizon articulated task 上 fail, 因为降低 gravity 之后 policy 依赖"无重力 cheat", 一旦 gravity 恢复就崩。Virtual controller 是 **structurally provide task-completion guidance**, 而 physics relaxation 只是让 task 变简单, guidance 性质完全不同。

这种 "decaying assistance" 思想在很多领域有原型:
- **Constrained optimization warm-start** (Mordatch et al. 2012 contact-invariant optimization, https://arxiv.org/abs/1406.2869): 先 relax contact constraint, 慢慢 tighten;
- **DeepMimic 的 reference state initialization**: 给 agent 一个 reference-init 状态, 让它先在 demo 附近 explore;
- **Privileged action learning** (Mao et al. 2025, https://arxiv.org/abs/2502.15442): 先用 privileged action 帮助 explore, 再 distill 出无 privileged 的 policy;
- **Adversarial motion prior** (AMP): 不强制 mimic, 用 reward shaping 让 motion distribution 接近 demo;
- **Self-paced learning** (Kumar et al.): 自动 adjust difficulty。

DexMachina 的 twist 是: 把"老师" (virtual controller) 放在 **object** 上, 而不是放在 hand 上。这让 hand action space 完全 free 给 policy 学, 同时 task 难度被 continuous 降。

---

## 3. Experiment 细节

### 3.1 Setup
- **Physics simulator**: Genesis (https://github.com/Genesis-Embodied-AI/Genesis), 比 IsaacGym 更稳定的 contact modeling + 更 memory efficient, 支持 12,000 parallel envs;
- **RL algorithm**: PPO (Schulman et al. 2017, https://arxiv.org/abs/1707.06347), via rl-games (https://github.com/Denys88/rl_games);
- **Data**: ARCTIC dataset (https://arctic-project.github.io/, Fan et al. CVPR 2023), 5 articulated objects (waffle iron / box / mixer / notebook / ketchup), 7 demo clips with diverse motion sequences;
- **Hands**: 6 dexterous hands — Inspire (https://inspire-robots.store), Allegro (https://www.allegrohand.com), XHand (https://www.robotera.com), Schunk SVH (https://schunk.com), Ability (https://www.psyonic.io), DexRobot Hand;
- **Hardware**: 单张 L40s 或 H100 GPU;
- **Repetitions**: 5 random seeds per (hand, task) combination;
- **Eval**: 每个保存的 best checkpoint 跑 20 episodes。

### 3.2 Evaluation metric — ADD-AUC

不用简单的 success rate (因为 3 个 threshold 都要选很 arbitrary), 也不用 raw tracking error (太多 dimension 看不出 high-level 趋势)。

借鉴 object pose estimation 的 ADD (Average Distance of Model points) metric, Wen et al. FoundationPose (https://arxiv.org/abs/2312.00780) 和 PoseCNN (https://arxiv.org/abs/1711.00199) 都用 ADD-AUC。

Twist: 因为是 articulated object, 对 **每个 part 单独 compute ADD**, 再 average, 最后 compute AUC over thresholds。

### 3.3 Baselines

1. **Kinematics Only**: 直接 replay kinematic retargeting 结果。Visual 像人手, 但 functionally 几乎不能 task 成功, 只能勉强 lift object 几厘米;
2. **ObjDex (re-implementation)** (Chen et al. 2024, https://arxiv.org/abs/2411.04005): original 是 IsaacGym, 这里在 Genesis 重新实现并改进 (12k envs vs 2048 envs), 在 short-horizon task 上比原 paper 报告更好 (Ketchup-100 从 41.2% → >90%, Mixer-170 从 57.6% → >70%);
3. **Task + Aux Rewards without curriculum**: 消融, 看 curriculum 的作用;
4. **ManipTrans (re-implementation)** (Li et al. 2025, https://arxiv.org/abs/2503.21860): 在我们的 setup 下复现 decay gravity/friction/error threshold 的 curriculum。

### 3.4 Main results (Figure 3)

- 7 个 demo task, 4 个 representative hands, 平均 success rate 对比;
- **DexMachina 在所有 hands + 所有 long-horizon tasks 上都超过 baselines**;
- Short-horizon tasks (前 3 列, Ketchup-100, Box-170, Mixer-170) 上, Task + Aux Rewards 已经可以 work, curriculum 增益相对小;
- **Long-horizon tasks** (后 4 列, Box-300, Mixer-300, Notebook-300, Waffle-300) 上, 没有 curriculum 的方法几乎全 fail, DexMachina 大幅领先;
- ManipTrans 的 decay-physics-parameter curriculum 不稳定, 训练后期 reward 掉下去恢复不了。

### 3.5 Hand-specific strategies (Figure 4) — 这个 figure 信息量很大

| Task | Hand | Strategy |
|------|------|----------|
| Notebook-300 | XHand | 左手 hold, 右手 close cover (跟随 human demo) |
| Notebook-300 | Inspire | 双手一起 hold, 一起 close cover (因为 Inspire 单手不够稳) |
| Mixer-300 | Allegro | 长手指直接 close lid, wrist 几乎不动 |
| Mixer-300 | Schunk | 手指短, wrist 大幅 motion 来补偿 |

这说明 DexMachina 的 policy **适应 hardware constraint**, 不是死板 mimic 人类。Auxiliary reward 是 soft guidance, 不是 hard constraint, 给 policy 灵活 explore 的空间。

### 3.6 Hand embodiment analysis (Figure 5) — 关键发现

加入 Ability 和 DexRobot Hand 一起评测:

| Finding | 解释 |
|---------|------|
| **Bigger + fully-actuated hands 学得更快更好** | Allegro 手指长, in-air manipulation 稳; |
| **Size 相似度不如 DoF 重要** | Inspire / Ability / Schunk size 差不多, 但 Schunk 有 actuated fingertip + foldable palm, 表现更好; |
| **Less-actuated hand 学的策略更不 human-like** | 因为相同 human reference 对它不 feasible, 必须找 alternative strategy; |
| **Allegro 虽然 anthropomorphism 弱但 capability 强** | 反直觉但合理: 长 finger 提供 contact stability; |

这与 OpenAI Dactyl (https://arxiv.org/abs/1910.07143) Shadow Hand 选择、RoboPianist (https://robopianist.github.io) 的发现一致: 在 RL 设定下, 机构 capability > anthropomorphism。

---

## 4. Ablation 细节

### 4.1 Action ablation (Figure 8)

对比三种 action formulation (no curriculum setting, 3 seeds):
1. **All absolute**: 所有 joint 用 absolute action (公式里的 finger 公式);
2. **Less-constrained residual**: wrist 也用 residual, 但 wrist limit 设成整个 demo 的最大 range (不严格);
3. **Hybrid (ours)**: wrist residual + finger absolute, wrist limit 严格基于 demo。

Result: **Hybrid 最好**。Auxiliary reward 都能提升三种方法, 但 hybrid 提升最多。

Intuition: Wrist 是动作 main frame, residual action 让 policy 修正 demo reference, 严格 bound 让 wrist action space 缩到"合理范围内", 大幅降低 sample complexity。

### 4.2 Curriculum ablation (Section 5.2)

- **ManipTrans-style physics parameter decay**: unstable, no clear improvement over no-curriculum;
- **DexMachina virtual controller decay**: stable improvement, 尤其在 long-horizon。

作者归因: physics parameter decay 只是让 task 简单, 不提供 "task-completion guidance"; virtual controller decay 直接把 object 拉到 demo target, 让 policy observe 整个 sequence, 这是 structural guidance。

---

## 5. 与相关工作 lineage 的联想

### 5.1 Curriculum learning 谱系

- **DeepMimic reference state init** (Peng et al. 2018): 起步在 demo state, 让 agent 在 demo trajectory 附近 explore, 不主动 decay;
- **Contact-invariant optimization** (Mordatch et al. 2012, https://arxiv.org/abs/1406.2869): trajectory optimization 里先 relax contact 决策变量, 慢慢 tighten, 用 warm-start 解决非凸;
- **Privileged action curriculum** (Mao et al. 2025, https://arxiv.org/abs/2502.15442): 给 agent 一个 privileged action 帮 explore, 训练过程中 decay privileged action;
- **Domain randomization curriculum** (OpenAI Dactyl, https://arxiv.org/abs/1910.07143): randomize physics params, 让 policy robust to sim-to-real gap, 但不主动 decay;
- **Decaying gravity/friction** (ManipTrans): reduce gravity 让 task 简单, 慢慢 restore, 容易 collapse;
- **DexMachina virtual controller**: 在 object 上加 virtual PD controller, decay gain 让 object 从"被动引导"到"主动被 hand 操作"。这种"放老师到 object 上"是 structural novelty, 区别于放老师到 agent 自己 action 上 (像 residual RL) 或放老师到 environment physics 上 (像 ManipTrans)。

### 5.2 Retargeting 谱系

- **Kinematic retargeting** (AnyTeleop, DexPilot, https://arxiv.org/abs/1911.01131): joint-level 或 keypoint-level optimization, 把 human motion 映射到 robot motion, 不管 task 是否成功;
- **Functional retargeting** (DexMachina, this paper): 给定 demo trajectory, 学一个能真正实现 task 的 policy;
- **Cross-embodiment skill transfer** (xSkill, https://arxiv.org/abs/2307.09934): 在一个 embodiment 学 skill embedding, 转到其他 embodiment;
- **VideoDex** (https://arxiv.org/abs/2212.04498): 从 internet video 学 affordance prior, 再用 RL fine-tune;
- **MyoDex** (https://arxiv.org/abs/2309.03130): generalizable prior for dexterous manipulation;
- **DexCap** (https://arxiv.org/abs/2403.07788): portable mocap 系统, 收集 dexterous task data。

### 5.3 Bimanual / articulated object 谱系

- **ARCTIC** (https://arctic-project.github.io): 第一个 large-scale bimanual articulated object manipulation dataset, 用 mocap + dense annotation;
- **ArtiGrasp** (https://arxiv.org/abs/2310.01203): physically plausible bimanual grasping + articulation synthesis;
- **RoboPianist** (https://robopianist.github.io): bimanual piano playing benchmark;
- **Bimanual Dexterity** (Pathak group, https://arxiv.org/abs/2411.18777): 复杂 bimanual task via RL;
- **DexArt** (https://dexart.github.io): articulated object RL benchmark, 但 single-hand 为主。

### 5.4 Dexterous RL benchmark 谱系

- **Dactyl / OpenAI Rubik's cube** (https://arxiv.org/abs/1910.07143): in-hand cube rotation, Hindsight Experience Replay + domain randomization;
- **DexPoint** (https://arxiv.org/abs/2109.12805): single-hand manipulation benchmark;
- **IsaacGym / IsaacLab** (https://arxiv.org/abs/2108.10470): GPU-accelerated RL;
- **Genesis** (https://github.com/Genesis-Embodied-AI/Genesis): generative physics engine, 比 IsaacGym 更通用更 stable;
- **RoboPianist** (https://robopianist.github.io): high-DoF bimanual piano benchmark。

---

## 6. Limitations (作者自己说)

1. **State-based input 依赖 privileged info**: 实际部署需要 vision-based RL 或 teacher-student distillation (类似 ObjDex / ManipTrans 的做法);
2. **依赖高质量 mocap 数据**: ARCTIC 用 mocap + dense annotation, 收集成本高。未来可能用 3D generative model (e.g. TeDy, https://arxiv.org/abs/2401.06181) 或 neural reconstruction 替代;
3. **Simulator hand asset 准确度**: open-source URDF 的 mass / inertia / collision shape 是估计的, 不能完全 capture 真实 hardware dynamics;
4. **没做 real-world experiment**: 暂时只在 sim 验证, 想做 sim-to-real 还要 distillation (像 Dextreme https://arxiv.org/abs/2210.13767 那样)。

---

## 7. 我自己的 intuition 拓展

### 7.1 为什么 "decaying assistance on object" 是个好 inductive bias

从 RL optimization landscape 角度看, long-horizon task 的 reward landscape 有大量 local minimum (早期 drop object 之后所有后续 trajectory 都 fail)。Virtual controller 在 object 上加一个 **basin of attraction** around demo trajectory, 把 landscape 平滑化。Policy gradient 在这个 smooth landscape 上能稳定收敛。然后随着 controller gain decay, basin 慢慢 shrink, policy 被"逼"到自己 form 一个 basin。这种"先 convex relax 再 tighten"在 trajectory optimization 里是 standard trick (e.g. LQR-RRT, CHOMP, TrajOpt), 把它 port 到 RL 里需要的是 reward shaping + adaptive difficulty, 这里通过 PD controller 的物理实现做的更自然。

类似思路: **guided policy gradient** (OpenAI learning to learn, ES + guidance), **asymmetric self-play** (OpenAI self-play, agent vs adversary 共同进化), **demonstration augmented PPO** (Rajeswaran 2017 DDPG + demos, https://arxiv.org/abs/1709.10087)。

### 7.2 跟 LLM / VLM RLHF 的呼应

LLM RLHF 里也有类似 "decaying assistance" 思路: e.g. **expert iteration** (https://arxiv.org/abs/2305.06383) 先用 strong model 生成 demos, weak model 模仿, 然后 gradually 提高 weak model 的自主性。或者 **rejection sampling fine-tuning** + **iterative DPO**, 一开始用 strong reference, 慢慢让 weak policy 自己 dominate。这是不同 modality 下的同一 inductive bias。

### 7.3 Functional retargeting vs Kinematic retargeting 的本质区别

**Kinematic retargeting** 解的是: $\min_\theta \sum_t \|{\rm FK}(\theta_t) - x_t^{\rm human}\|_2$, 即 joint angle $\theta$ 使 forward kinematics 接近 human keypoint $x$。这是 pure geometric。

**Functional retargeting** 解的是: $\min_\pi \sum_t F(g_t(\pi), g_t^{\rm demo})$, 即 policy $\pi$ 使得 rollout 出来的 object trajectory $g(\pi)$ 接近 demo $g^{\rm demo}$。这是 dynamical + contact。

后者难得多但 task semantics 保留完整, 这是 DexMachina 选题的根本价值。

### 7.4 对未来 hand design 的启示

Figure 5 的 hand embodiment analysis 实际上是一个 **"hardware capability benchmark via RL training performance"** 的雏形。如果你想设计新 dexterous hand, 把它丢进 DexMachina benchmark, 看 (a) final performance, (b) learning efficiency, 就能反推 hardware capability。

这与 NVIDIA Isaac Lab 的 design philosophy、与 Boston Dynamics 的 "evolutionary benchmark" 思路一致。未来可能成为 standard hand benchmark 之一。

---

## 8. Reference links 汇总

**DexMachina**:
- Project: https://project-dexmachina.github.io

**Dataset / Simulator / RL lib**:
- ARCTIC dataset: https://arctic-project.github.io
- Genesis physics engine: https://github.com/Genesis-Embodied-AI/Genesis
- rl-games: https://github.com/Denys88/rl_games
- PPO paper: https://arxiv.org/abs/1707.06347

**Baseline / Related dexterous manipulation**:
- AnyTeleop (kinematic retargeting): https://arxiv.org/abs/2307.04577
- ObjDex: https://arxiv.org/abs/2411.04005
- ManipTrans: https://arxiv.org/abs/2503.21860
- DeepMimic: https://arxiv.org/abs/1804.02717
- Dactyl (OpenAI Rubik's cube): https://arxiv.org/abs/1910.07143
- Rajeswaran demo RL: https://arxiv.org/abs/1709.10087
- DexArt benchmark: https://dexart.github.io
- Dextreme (sim-to-real): https://arxiv.org/abs/2210.13767
- RoboPianist: https://robopianist.github.io
- xSkill: https://arxiv.org/abs/2307.09934
- VideoDex: https://arxiv.org/abs/2212.04498
- DexCap: https://arxiv.org/abs/2403.07788
- ArtiGrasp: https://arxiv.org/abs/2310.01203

**Robot hands**:
- Inspire Hand: https://inspire-robots.store
- Allegro Hand: https://www.allegrohand.com
- XHand (ROBOTERA): https://www.robotera.com
- Schunk SVH: https://schunk.com
- Ability Hand (PSYONIC): https://www.psyonic.io

**Curriculum / optimization**:
- Contact-invariant optimization (Mordatch): https://arxiv.org/abs/1406.2869
- Privileged action curriculum: https://arxiv.org/abs/2502.15442
- Residual policy learning: https://arxiv.org/abs/1812.06298
- FoundationPose (ADD-AUC metric lineage): https://arxiv.org/abs/2312.00780
- PoseCNN: https://arxiv.org/abs/1711.00199

---

## 9. 总结 (TL;DR)

DexMachina 把"functional retargeting"形式化为: 给一段 human bimanual articulated object manipulation demo + 一对 robot dexterous hand, 学一个 RL policy 把 object 沿 demo 轨迹操作出来。

核心方法三件套:
1. **Hybrid action** (wrist residual + finger absolute), 缩 action space, bootstrap with demo prior;
2. **Auxiliary reward** (motion imitation + behavior cloning + contact matching), 提供 soft guidance;
3. **Virtual object controller curriculum** (核心创新), 在 object 上 attach virtual PD controller 用 demo target 作为 control goal, gain 随 reward stability 指数 decay, 让 policy 从"观摩者"平滑 transition 到"操盘者"。

Benchmark 6 个 hand + 5 个 articulated object, DexMachina 在 long-horizon task 上显著超过 ObjDex / ManipTrans / no-curriculum baselines。Hand embodiment analysis 揭示 bigger + fully-actuated hand (e.g. Allegro) 比 anthropomorphic 但 less-actuated hand (e.g. Inspire) 学得更好。

整体来看, 这是 **"decaying assistance curriculum"** 在 dexterous manipulation 上的一次干净漂亮的应用, 关键 insight 是把"老师"放在 object 上 (而不是 agent action 或 environment physics 上), 这个 twist 解决了 long-horizon 探索难题。后续可以接 vision-based policy distillation、sim-to-real transfer、更大规模 hand design benchmark。
