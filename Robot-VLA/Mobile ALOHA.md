---
source_pdf: Mobile ALOHA.pdf
paper_sha256: 698c3b093b388575ea32e214b074e21b2539f6dae04e1f3bc614c7e461f05286
processed_at: '2026-08-05T19:26:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Mobile ALOHA

## 一、这帮人到底干了啥

Stanford 的 Zipeng Fu、Tony Zhao 和 Chelsea Finn 把原来的 ALOHA 系统（就是那个两只机械臂 puppet 操作的便宜桌面机器人）搬到了一个带轮子的底盘上，花了 $32k 搞出来一台能满屋子跑、两只手干活的 mobile manipulator。然后他们发现一个 trick：**用原来桌面 ALOHA 的旧数据跟新的 mobile 数据一起 co-training，新任务只要 50 个 demo 就能学到 80-95% 的成功率**。

就这么一句话总结的事，但背后的 intuition 很有意思。

项目主页: https://mobile-aloha.github.io

---

## 二、为啥这事之前没人做成

### 2.1 硬件太贵

你要买个现成的 bimanual mobile manipulator，比如 PR2 或者 TIAGo，起步价 $200k 往上。普通 lab 根本买不起。而且这些 robot 要做 teleoperation 还得加各种外设：PR1 用两个 haptic device 加 foot pedal 控制 base；TIAGo 得用 motion capture 系统 retarget human motion，光标定就折腾死人。

参考: PR2 历史 https://www.willowgarage.com/pages/pr2/overview

### 2.2 软件层面也没人证明能 work

之前 mobile manipulation 的 paper 要么用 predefined primitives（"先导航过去，再抓"这种手工分解），要么用 RL 但 state space 分解。直接 end-to-end BC 把 base 和 arm 的 action concat 在一起扔给 transformer，没人确信能 work——因为 base 一个小偏移会让 arm end-effector 偏一大截，compounding error 会很夸张。

Paper 的 Section 1 把这两个痛点列得很清楚。

---

## 三、Hardware 设计的 "Aha Moment"

### 3.1 整体配置

- **底盘**: AgileX Tracer AGV，differential drive，$7000，max speed 1.6m/s，payload 100kg。这玩意本来是 warehouse 用的物流小车。
- **两只 follower arm**: Trossen ViperX 300 6-DOF，一只 $5-6k
- **两只 leader arm**: 同型号，给 operator 做 puppet 用
- **电池**: 1.26kWh lithium，14kg，既供电又当 ballast 降低重心
- **计算**: 一台 laptop，Nvidia 3070 Ti + Intel i7-12800H
- **摄像头**: 3 个 Logitech C922x，两个绑手腕，一个朝前
- **总成本**: ~$32k，跟一个 Franka Emika Panda 差不多

AgileX Tracer 链接: https://www.agilex.ai/products/108

### 3.2 灵魂设计：Tether + Backdrive

这地方真的聪明。原版 ALOHA 的 teleoperation 是 operator 两手各握一个 leader arm，通过 joint-to-joint 带动 follower arm。但双手都被占用了，base 怎么控制？

他们想出来一个极简方案：**用一条 tether 把 operator 的腰拴在 base 上**。Tracer 的轮子 torque off 之后 rolling resistance 只有 13N（vinyl floor 上测的），人腰一带就走，人一停就停。

为什么这招这么好：

1. **零延迟 haptic feedback**: robot 撞墙了，operator 通过 tether 瞬间感觉到，比任何 FPV + VR 方案都快
2. **自然协调**: 人要开柜门会本能后退一步，这个后退动作通过 tether 直接变成 base 的后退 action，collect 出来的数据就是 whole-body coordinated 的
3. **不用标定**: 不像 mocap 系统要先标 Vicon marker，tether 即插即用
4. **ergonomics 可调**: tether 点高度和 leader arm 位置都能上下调 30cm

这个设计让我想起 MIT Herkulex 那种 exoskeleton 方案的简化版——把人当作 base 的 active impedance controller。人是地球上最好的 controller，直接用就行。

---

## 四、Co-training 的核心 Finding

### 4.1 Setup

他们手头有两批数据：

- **Static ALOHA data**: 825 episodes，来自原版 ALOHA paper [104] 和 Waypoint-based imitation [81]，通过 RT-X [20] 发布。任务是 ziploc sealing、candy wrapping、battery slotting、coffee machine 等 12 种 tabletop task
- **Mobile ALOHA data**: 每个新 task 50 个 demo（High Five 和 Cook Shrimp 是 20 个）

RT-X 数据集: https://robotics-transformer-x.github.io

### 4.2 训练公式

训练 objective 长这样：

$$
\begin{align}
& \mathbb{E}_{(o^i, a^i_{\text{arms}}, a^i_{\text{base}}) \sim D^m_{\text{mobile}}} \left[ L(a^i_{\text{arms}}, a^i_{\text{base}}, \pi^m(o^i)) \right] \\
+ & \mathbb{E}_{(o^i, a^i_{\text{arms}}) \sim D_{\text{static}}} \left[ L(a^i_{\text{arms}}, [0, 0], \pi^m(o^i)) \right]
\end{align}
$$

变量解释：
- $o^i$: observation，包括 3 个 RGB 图像 + 14 维 arm joint position
- $a^i_{\text{arms}} \in \mathbb{R}^{14}$: 两个 arm 的 target joint position（含 2 个 gripper action）
- $a^i_{\text{base}} \in \mathbb{R}^{2}$: $(v, \omega)$ base 线速度和角速度
- $D^m_{\text{mobile}}$: 第 $m$ 个 mobile task 的数据集
- $D_{\text{static}}$: 825 episodes 的 static ALOHA 数据
- $\pi^m$: 任务 $m$ 的 policy
- $L$: imitation loss（ACT 是 smooth L1，Diffusion 是 denoising score matching）
- $[0, 0]$: **zero-padding**——把 static data 当成"base 不动"的 episode

mini-batch 以 50/50 概率从两个 dataset 采样，batch size 16。

### 4.3 为啥能 work 的直觉

这是 paper 最有意思的部分。表面上 static ALOHA 和 mobile ALOHA 差别巨大：
- task 完全不同（ziploc sealing vs 擦红酒）
- background 完全不同（黑色桌面 vs 厨房移动背景）
- arm 朝向不同（static 是两 arm 相对，mobile 是两 arm 并排朝前）
- 一个有 base 移动，一个没有

那为什么 co-training 还能 positive transfer？我想了几条 intuition：

**Intuition 1: Wrist camera 的 invariance**

Wrist camera 视角有个关键性质：gripper 永远在画面中央，背景永远在变。无论 base 移不移动，wrist camera 看到的"接近物体 → 抓住物体"这个 visual stream 本质相同。Hsu et al. [41] 专门论证过 wrist camera 对 manipulation 的重要性。所以 static data 学到的"如何从 wrist 视觉判断 gripper-物体距离"直接迁移。

参考 Hsu et al.: https://arxiv.org/abs/2203.12677

**Intuition 2: Motion prior 的频域分离**

Mobile manipulation 的 action 可以做频域分解：
- Base velocity: 低频（变化慢，时间常数秒级）
- Arm joint: 中频（接近物体时变化，时间常数 100ms 级）
- Gripper open/close: 高频（瞬间切换）

Static ALOHA data 提供了大量中频和高频的 prior——怎么 reach、怎么 grasp、怎么 close gripper。Co-training 等价于让 network 同时学到 mobile data 的低频 component 和 static data 的中高频 component。两者频域互补。

**Intuition 3: Regularization 效应**

50 个 demo 对 ACT 这种 5M+ 参数的 transformer 远不够。Table 1 中 Push Chairs 任务 4th/5th chair（OOD）从 0%/0% → 85%/89%——这是 generalization 提升，不是 memorization。Static data 充当 visual regularizer，让 network 见过更多 background、lighting、object variation，不至于把"墙上挂了某幅画"这种 irrelevant feature 学进去。

**Intuition 4: 没用 pre-training**

Table 4 显示 pre-train + finetune 反而比 no co-training 还差（40% vs 50%）。原因是 finetune 阶段小 dataset 会触发 **catastrophic forgetting**——network 在 finetune 10K 步里把 static data 学到的 features 冲掉了。Co-training 是 joint optimization，static gradient 和 mobile gradient 在每一步同时出现，无 forgetting 问题。

这跟 LLM instruction tuning 的发现一模一样：continue pretraining 必须跟 instruction tuning 交错做，不能先 pretrain 完再 finetune，否则忘光。

---

## 五、三个 Base Algorithm 都能 co-train

这点很关键——证明 co-training 不是某个 algorithm 的 lucky accident。

### 5.1 ACT [104]

原版 ALOHA 用的方法。Architecture：
- ResNet18 backbone 提 image feature
- 4 层 encoder transformer + 7 层 decoder transformer
- Hidden dim 512, feedforward 3200, 8 heads
- $\beta = 10$ 的 KL weight（CVAE latent $z$）
- Chunk size $k = 10$：一次性预测未来 10 步 action
- Learning rate 2e-5, batch size 16

ACT codebase: https://github.com/ToniAKMMP/act

**Action chunking 公式**：对 timestep $t$，policy 输出
$$\pi(o_t) = (\hat{a}_t, \hat{a}_{t+1}, \ldots, \hat{a}_{t+k-1})$$
执行时只取前 $k - d$ 步，$d$ 是 base 延迟步数。

**Intuition**: chunking 把 high-frequency policy inference 替换成低频 inference + 高频 playback。Latency 降下来，trajectory coherence 上去，jitter 少了。这跟 LLM 中 KV-cache 思路异曲同工——一次 forward 多步输出。

### 5.2 Diffusion Policy [18]

Architecture：
- ResNet18 backbone
- UNet noise predictor
- DDIM scheduler, training 50 步, inference 10 步
- EMA power 0.75
- Chunk size 64
- Image augmentation: RandomCrop + ColorJitter + RandomRotation ±5°
- Learning rate 1e-4, batch size 32

Diffusion Policy 主页: https://diffusion-policy.cs.columbia.edu/

训练 objective：
$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \epsilon, a_0} \left[ \| \epsilon - \epsilon_\theta(a_t, t, o) \|^2 \right]$$

变量：
- $a_0$: clean action chunk
- $\epsilon \sim \mathcal{N}(0, I)$: 加的噪声
- $a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$: noised action at step $t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$: cumulative noise schedule product
- $\epsilon_\theta$: UNet 预测的噪声
- $t$: uniform 采样于 $\{1, \ldots, T\}$，$T=50$

**Diffusion 在 Wipe Wine 上 65% 比 ACT 95% 差**：paper hypothesis 是 50 demos 不够。Diffusion Policy 之前的工作（Chi et al.）都用 250+ demos。Diffusion 的 multimodality 在低数据 regime 下反而 overfitting。但 Push Chairs 上 100%，因为 Push Chairs 是 quasi-static 任务，multimodality 不严重。

### 5.3 VINN + Chunking [63]

VINN 是 retrieval-based method，不走 end-to-end：
- 训 BYOL [37] encoder 学 visual representation
- Test time 对 observation $o$ 找 $k$-nearest neighbor in training set，用 neighbor 的 action
- Chunk size 100
- State weight 5, camera feature weight 1:1:1

VINN paper: https://jyp220.engin.umich.edu/vinn/

**关键**: VINN 的 co-training 只 co-train BYOL encoder，action retrieval 部分没法直接 leverage static data——因为 static data 的 action 跟 mobile task 的 action 在不同空间。这解释 Table 2 中 VINN 的 mixed results：Wipe Wine 反而退步 5%，Push Chairs 提升 20%。BYOL encoder 见过更多 visual variation 是好事，但 retrieval action 本身没用 static data，所以收益打折扣。

**Intuition**: VINN 类似 "non-linear kNN lookup"，对 representation 质量极敏感但对 action 本身不学习。这跟 ACT/Diffusion 的 end-to-end 思路形成对比。

---

## 六、七个任务的人话讲解

### 6.1 Wipe Wine (50 demos, 26s/demo)

机器人 1.5m × 1.5m 区域任意位置 + yaw 30° 初始化。流程：
1. 导航到 sink 拿 towel
2. 走回 kitchen island
3. 左手 lift wine glass
4. 右手 wipe 桌面 + glass 底部
5. 放回 glass

Co-training vs no co-training: 95% vs 50%——**绝对提升 45%**。

Sub-task "Lift Glass and Wipe" 是提升最大环节：95% vs 58%。这个 sub-task 需要 bimanual coordination：一手抬一手擦。Static ALOHA 大量 bimanual task 提供了 "如何同时控制两 arm 做不同动作" 的 prior。

### 6.2 Cook Shrimp (20 demos, 75s/demo)

最长的 task。流程：
1. 右手倒油到 hot pan
2. 右手倒 raw shrimp
3. 左手 tilt pan + 右手 spatula flip shrimp
4. 转身把 shrimp 倒进 bowl
5. 放 pan 回 table

成功率 40%——唯一低于 80% 的 task。为啥这么难：
- 75s 长 horizon，20 demos 数据量太少
- Flip shrimp 是 dynamic contact：shrimp 与 pan 有 stiction + slipping，需要 fast wrist motion
- 白碗 + 白虾 contrast 低

这个 task 暴露了 BC 的 limitation——dynamic contact 的高频 modes 在 50 demos 下根本覆盖不全。

### 6.3 Rinse Pan (50 demos, 22s/demo)

流程：
1. 左手 grasp pan
2. 转身到 faucet
3. 右手开 faucet（knob 4cm 长 0.7cm 直径，shiny 不锈钢）
4. 左手接水 swirl
5. 倒水
6. 放 pan 到 rack

Co-training vs no co-training: 80% vs 0%——**绝对提升 80%**！Sub-task "Turn On Faucet" 是 80% vs 0%。

为啥差异这么大：faucet knob 是 0.7cm 直径 shiny 金属，光照变化剧烈。Static ALOHA 12 类 task 里有 "open plastic portion cup with lid" 这种需要 fine approaching 小物体的 task，提供了"如何 visual servo 到小 shiny object"的 prior。50 mobile demos 不足以学到这个，但加 825 个 static demos 直接补上。

### 6.4 Use Cabinet (50 demos, 30s/demo)

流程：
1. 走到 cabinet
2. 双手 grasp 两个 handle
3. **base 后退同时拉开门** ← 关键 whole-body coordination
4. 双手 grasp pot handle
5. 前进把 pot 放进去
6. 后退关门

成功率 85%/85%，co-training 提升不大但稳定。Pot 重 1.4kg，超过单 arm 750g payload，必须 bimanual。

这 task 最能说明 whole-body control 的必要性：开柜门时 arm 必须保持 grasp，同时 base 后退拉开。如果 arm 和 base 分开控制，根本做不到 fluid motion。

### 6.5 Call Elevator (50 demos, 45s/demo)

流程：
1. 从 15m 外随机位置出发
2. 绕过 column
3. 精确定位到 button 旁（2cm × 2cm button）
4. 按 button
5. 转 90° 进电梯（30cm clearance）

成功率 95%/0%——**绝对提升 95%**！Sub-task "Press Button" 是 100%/5%。

这个 task 是 navigation + fine manipulation 的极端组合。15m 导航需要 base 精确 dead-reckoning（没有 SLAM），2cm button 需要 cm 级 precision。Open-loop replay 完全不可能 work（Appendix A.4 显示 20 次 replay 偏 10cm 散布 20cm）。Policy 必须 closed-loop visual servo 到 button。

### 6.6 Push Chairs (50 demos, 40s/demo)

5 个椅子排成一排，每个 5kg。Training 只 demo 前 3 个 chair，测试 5 个。

Co-training 在 4th/5th chair（OOD）提升最大：85%/89% vs 0%/0%。这是 generalization 而非 memorization 的最强证据。

### 6.7 High Five (20 demos, 40s/demo)

机器人绕 kitchen island 转，人从前面来就 high five，人走开继续转。Test 时换衣服换人。

成功率 85%/85%。这 task 没精度要求但测 HRI 能力。

---

## 七、Action Chunking 处理硬件延迟的 Trick

Section 6 提到一个 engineering detail。Mobile base 有约 $d \approx 2$ 步延迟（base velocity controller response 慢），arm 几乎无延迟（position control 响应快）。

如果 chunk size $k = 10$，怎么处理？

**Trick**: 执行 chunk 前 $k - d = 8$ 个 arm action（即时响应），同时执行 chunk 后 $k - d = 8$ 个 base action（base 在第 9, 10 步才需要这些 target，等它追上来正好对齐）。

这相当于在时间轴上把 base action chunk shift $d$ 步，让 hardware 物理延迟跟 action sequence 自然 align。

**Intuition**: 这是 control theory 中经典的 "transport delay compensation"（Smith predictor 那一类），但用 action chunking 优雅地实现，不需要显式 system identification。

---

## 八、Open-Loop Replay 实验（最 informative 的 sanity check）

Appendix A.4 的实验值得单独讲。

他们把一个 demo 的 base velocity 和 arm action 直接 open-loop replay 20 次（不复位 base，不修正 error），最后让 arm reach 出去 tap 一张纸，标记 tap 位置。

结果：
- 原始位置：红 ✕
- 20 次 replay 位置：20 个红点
- **所有点偏左 ~10cm**
- **散布沿一条线 ~20cm**

这意味着：
1. Base velocity control 有 systematic bias（差速轮 calibration + 地面摩擦不对称）
2. 接触地面 stochastic 摩擦 + delay 引入 random spread
3. Open-loop replay **0% 成功**

**Intuition**: 这就是为什么 imitation learning 在 mobile manipulation 上比 tabletop 难得多——每一步 observation 都已经 drifted，policy 必须从 RGB 实时纠正，不能依赖 proprioception。Mobile ALOHA 的成功证明 CNN backbone + wrist camera 提供足够 visual servoing signal，policy 内隐学到了 base-arm coordinated correction。

这也是为什么 SayCan [4] 那种用 LLM 分解 task + 用 primitive 执行的路线在 mobile manipulation 上有局限——primitive 执行也是 open-loop，遇到 drift 就崩。

---

## 九、User Study 的小细节

8 个 CS graduate student（5 女 3 男，21-26 岁），4 个无 teleop 经验。每人 3 分钟自由探索，然后做 Wipe Wine 和 Use Cabinet 两个 task。每个 task expert 先 demo，然后 participant 做 5 次。

结果：
- Wipe Wine: 46s → 28s（-39%）
- Use Cabinet: 75s → 36s（-52%）
- 5 次后接近 expert 速度

**Intuition**: tether + puppet 设计 ergonomics 太好，operator 不用学就能上手。这跟 VR headset + FPV 那种"晕动症半小时才能适应"形成鲜明对比。Teleoperation ergonomics 直接决定 data collection throughput——如果 operator 训练要一周，50 demos 都 collect 不出来。

---

## 十、Limitations & 我的猜想

Paper Section 9 自己列的：
1. Footprint 90×135cm 太宽，窄门难过
2. Arm 固定高度，到不了低 cabinet / oven
3. 单任务 BC，不能 self-improve
4. 只用 expert operator，没处理 suboptimal data

我额外想几条：

### 10.1 Wheeled base 的根本限制

Differential drive 无法侧移，narrow corridor 要 S-curve。不能爬楼梯。这些是 wheeled mobile manipulator 共有问题，要解决得换 bipedal humanoid。

未来路径：把这个 co-training recipe 推广到 bipedal humanoid（Figure 01、Unitree H1、Tesla Optimus）。Tether 方案改一下：operator 穿 force plate 鞋，bipedal 跟随 operator 行走。参考 Purushottam et al. [68] 做的 wheeled humanoid whole-body bilateral teleoperation 用 force plate。

### 10.2 Cook Shrimp 失败暗示什么

40% 成功率暗示 BC 在 dynamic contact task 上有 glass ceiling。Shrimp 与 pan 的 stiction 随温度湿度变化，每次接触 mode 不同。BC 只能学到 average trajectory，遇到 novel contact mode 就崩。

解决路径：BC initialize + RL fine-tuning。用 Mobile ALOHA BC policy 做 behavior prior，在 simulator 里用 domain randomization 跑 RL 优化 dynamic contact。参考 RoboCat [11] 的 self-improvement loop。

RoboCat: https://www.deepmind.com/blog/robocat

### 10.3 大规模 Foundation Model 路线

Mobile ALOHA + static ALOHA 共 ~1000 demos 训 7 个 task。如果社区共建 mobile ALOHA fleet（像 DROID 那样 crowdsourcing），10k-100k demos 可训练真正 bimanual mobile foundation model。

DROID: https://droid-dataset.github.io

Co-training recipe 可推广为：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mobile bimanual}} + \mathcal{L}_{\text{static bimanual}} + \mathcal{L}_{\text{single-arm cross-embodiment}} + \mathcal{L}_{\text{internet VLM}}$$

这是 RT-2 [13] 路线的 bimanual mobile 版本。

RT-2: https://robotics-transformer2.github.io

### 10.4 跟 Simulation 的关系

Paper 完全 real-world，没 sim。Reproducibility 受限。下一步显然在 Isaac Lab / MuJoCo MJX 建 Mobile ALOHA URDF，sim demos + real demos co-training，用 domain randomization 跨 sim-to-real gap。

Isaac Lab: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0.0-stable/Isaac-Lab-4.0.0/index.html
MuJoCo MJX: https://mujoco.readthedocs.io/en/stable/mjx.html

### 10.5 跟 VLA (Vision-Language-Action) 的连接

Mobile ALOHA 没用 language。加 language conditioning 是显然下一步：用 RT-2 风格的 VLM backbone + Mobile ALOHA co-training data。这样 robot 就能听 "把虾煎了盛碗里" 这种自然语言指令。

参考 Octo [61] 的 open-source generalist policy 思路：把 Mobile ALOHA 数据加进 Octo training mixture。

Octo: https://octo-models.github.io

---

## 十一、最后的 Intuition 总结

Mobile ALOHA 教会我的几件事：

1. **Whole-body control 不需要分解**：直接 concat 16 维 action vector 扔给 transformer，让它自己学 coordination。比 [38, 48, 58, 94] 那种 decomposed action space 简单一个数量级，效果反而好。这跟 LLM end-to-end 比 pipeline NLP 强是一个道理。

2. **Co-training > Pre-training**：domain overlap 时 joint training 把 source data 当 regularizer，比 pretrain+finetune 显著好。这跟 LLM continue pretraining + instruction tuning 交错做是一个道理。

3. **Wrist camera 是 transfer 的物理基础**：base 移动改变 world frame 视角，但 wrist camera 视角对 base 几乎不变。这是为什么 static → mobile transfer 能 work。设计 mobile manipulator 一定要给 arm 装 wrist camera，不能只靠 top camera。

4. **Hardware simplicity 决定 data quality**：tether + backdrive 比 VR/mocap/exoskeleton 简单一个数量级，且 collect 的数据更"自然"——operator 本能的 step-back 动作直接成为 demo 的 whole-body coordination。Hardware 设计直接影响 algorithm 能 work 与否。

5. **Closed-loop visual servoing 是 mobile manipulation 的本质**：open-loop replay 0% 成功说明 base drift 不可避免，policy 必须 closed-loop 修正。这也是为什么不用 SLAM——直接从 RGB 学视觉纠正，更 robust。

6. **50 demos 是 sweet spot**：对 transformer-based ACT，50 demos 让 co-training 起作用；<25 demos 时即使 co-training 也跌到 ~70%（Figure 4）。这个数字对未来大规模 data collection 有指导意义——每个 task 50-100 demos 是性价比最高的区间。

---

## 十二、参考资源汇总

**核心 paper**:
- Mobile ALOHA: https://mobile-aloha.github.io
- ALOHA 原版: https://tonyzhaozh.github.io/aloha/
- ACT: https://tonyzhaozh.github.io/act/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- VINN: https://jyp220.engin.umich.edu/vinn/

**Dataset**:
- RT-X: https://robotics-transformer-x.github.io
- DROID: https://droid-dataset.github.io
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

**Related foundation model**:
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://robotics-transformer2.github.io
- Octo: https://octo-models.github.io
- RoboCat: https://www.deepmind.com/blog/robocat

**Hardware**:
- AgileX Tracer: https://www.agilex.ai
- Trossen ViperX 300: https://www.trossenrobotics.com/viperx-300-robot-arm.aspx
- Logitech C922x: https://www.logitech.com/products/webcams/c922-pro-stream-webcam.html

**Simulation**:
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- MuJoCo: https://mujoco.readthedocs.io/
- RoboCasa: https://robocasa.ai

**Author pages**:
- Zipeng Fu: https://zipengfu.com
- Tony Zhao: https://tonyzhaozh.github.io
- Chelsea Finn: https://ai.stanford.edu/~cbfinn
- Stanford IRIS Lab: https://iris.stanford.edu
- Stanford REAL Lab: https://real.stanford.edu

---

简短一句话总结：Mobile ALOHA 用 $32k 硬件 + tether  teleoperation + co-training with old static data，把 bimanual mobile manipulation 从高端研究设施才能玩的 game 民主化到普通 lab 可及的水平，同时证明 co-training 比 pre-training 更适合 robot learning 的 cross-domain transfer。这是 robotics 进入 "ALOHA 时代" 的标志性工作。

你想深挖哪一块我都能继续展开——比如 ACT 的 CVAE latent 为什么 $\beta = 10$、Diffusion Policy 的 DDIM 10 步推理为啥不发散、或者 tether 长度对 backdrive 力的 leverage 影响。

---

# Mobile ALOHA 深度技术讲解

## 一、Paper 核心定位与历史脉络

Mobile ALOHA 出自 Stanford 的 Zipeng Fu、Tony Z. Zhao 和 Chelsea Finn 之手（Chelsea Finn 同时是 ALOHA 原版 [104] 与 RT-X [20] 的工作者），本质上是把原版 ALOHA bimanual puppeteering system 从桌面扩展到 whole-body mobile manipulation。它在 robotics community 中标志着 **low-cost bimanual mobile manipulation + imitation learning at scale** 进入"厨房级别"应用阶段。

参考链接：
- Project page: https://mobile-aloha.github.io
- ALOHA 原版: https://tonyzhaozh.github.io/aloha/
- RT-X dataset: https://robotics-transformer-x.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- VINN paper: https://jyp220.engin.umich.edu/vinn/

---

## 二、Hardware Architecture 深度剖析

### 2.1 系统拓扑

Mobile ALOHA 的 mechanical architecture 由三层构成（自下而上）：

1. **Mobile base layer**: AgileX Tracer AGV，differential drive，max speed 1.6 m/s，payload 100kg，cost $7,000。为什么选 Tracer 而不是 Clearpath Husky？价格便宜 5x 以上，且 low-profile (17mm height) 允许把 battery 沉到地面附近降低 CoM (Center of Mass)。
2. **Compute & power layer**: 14kg 的 1.26kWh lithium battery（既当 power source 又当 ballast weight，smart trick），加上一台 consumer-grade laptop with Nvidia 3070 Ti (8GB VRAM) + Intel i7-12800H。
3. **Manipulation layer**: 两个 Trossen Robotics ViperX 300 6-DOF arm（一主一从 leader-follower 配对，外加两个用于 teleoperation 的 leader arm），共 14 DoF arm + 2 DoF base = 16 DoF action space。

### 2.2 Whole-Body Teleoperation 的 "Tether" 设计直觉

原版 ALOHA 是 puppeteering setup：operator 双手各握一个 leader arm，通过 joint-to-joint 的 teleoperation 驱动 follower arm。但双手已经被占用，如何同时控制 base？

Paper 的解决方案极简：把 operator 的腰通过一条 tether 物理连接到 base frame。Tracer 的 wheels 在 torque off 时 rolling resistance 仅 13N，operator 可以"backdrive" wheels——本质上是把整个 base 当成一个被动的 2-DoF planar joint。这相当于：

$$\text{Operator} \xrightarrow{\text{waist}} \text{Base frame} \xrightarrow{\text{differential drive}} \text{linear vel } v, \text{angular vel } \omega$$

这种设计的妙处在于 **coarse haptic feedback**：robot 撞到墙，operator 通过 tether 直接感受到冲击；这比 VR headset 加 FPV 视频流的方案更"亲肤"，且不需要 SLAM 或 motion capture 标定。

**Intuition**: 把 human 当作 base 的 active impedance controller，human 的 proprioception + visual + haptic 三通道融合零延迟地完成 whole-body control。这是为什么 user study 显示 5 次试验后 operator 时间下降 39%-52%。

### 2.3 为什么不用其他方案

| 方案 | 问题 |
|------|------|
| VR headset + FPV | latency 高，operator 容易 disoriented |
| Foot pedals 控制 base（如 PR1 [93]）| 双脚已被站立姿态占用，ergonomics 差 |
| Motion capture retargeting [5] | 需要标定，single arm only |
| Exoskeleton [32, 45, 72] | 成本高，bingham 难 |

---

## 三、Co-training 形式化讲解

### 3.1 Action 与 Observation 空间

定义：
- $o^i \in \mathcal{O}$: observation，包含
  - 2 个 wrist camera RGB 图像（左、右）
  - 1 个 top egocentric RGB 图像（向前方拍摄）
  - 14 维 arm joint positions（proprioception）
  - 注意：base pose **不在** observation 中，这是关键设计——policy 不依赖 global localization
- $a^i_{\text{arms}} \in \mathbb{R}^{14}$: 两个 gripper 的 target joint position（含两个 continuous gripper action）
- $a^i_{\text{base}} \in \mathbb{R}^{2}$: $(v, \omega)$，linear 和 angular velocity of base

### 3.2 训练 Objective 数学公式

Paper Section 4 给出 co-training objective：

$$
\begin{align}
& \mathbb{E}_{(o^i, a^i_{\text{arms}}, a^i_{\text{base}}) \sim D^m_{\text{mobile}}} \left[ L(a^i_{\text{arms}}, a^i_{\text{base}}, \pi^m(o^i)) \right] \\
+ & \mathbb{E}_{(o^i, a^i_{\text{arms}}) \sim D_{\text{static}}} \left[ L(a^i_{\text{arms}}, [0, 0], \pi^m(o^i)) \right]
\end{align}
$$

**变量逐一解释**：
- $D^m_{\text{mobile}}$: 第 $m$ 个 Mobile ALOHA 任务的 dataset（每个任务约 50 demos）
- $D_{\text{static}}$: 来自 [81, 104] 通过 RT-X [20] release 的 825 episodes static ALOHA data，包括 ziploc sealing、candy wrapping、battery slotting 等 12 种 task，与 mobile tasks 完全 disjoint
- $\pi^m$: 任务 $m$ 的 policy，输入 $o^i$，输出 16 维 action
- $L$: imitation loss。对 ACT 是 smooth L1 loss on action chunk；对 Diffusion Policy 是 denoising score matching loss；对 VINN 无 explicit loss
- $[0, 0]$: **zero-padding** 的 base action——把 static data 当作"base 不动"的特殊 episode

**关键 sample 策略**: 每个 mini-batch 以 50/50 概率从 $D^m_{\text{mobile}}$ 和 $D_{\text{static}}$ 采样，batch size 16。Table 3 ablation 显示 30%/50%/70% 三种比例 success rate 95/95/90%，非常 robust。

### 3.3 为什么 Co-training 有效的直觉

Paper 的核心 finding 是**跨 morphology、跨 task、跨 background** 的 positive transfer。直觉来源：

1. **Wrist camera invariance** [41]: wrist camera 视角与 base 是否移动**几乎不变**（gripper 永远在画面中央），所以 static data 学到的"如何接近物体、如何抓取"的 visual features 直接迁移到 mobile。
2. **Motion prior**: static ALOHA 包含大量 "approach object → grasp → lift" 的 motion prior，这构成 mobile manipulation 的高频成分；base velocity 是低频成分。Co-training 等价于让 network 同时学到 high-frequency arm control 和 low-frequency base control。
3. **Regularization**: 50 个 demos 对一个 transformer-based policy（ACT 5M+ params）远不足，static data 充当 visual regularizer，防止 overfitting to wall background、lighting。Table 1 显示 Push Chairs 的 4th/5th chair（OOD）从 0%/0% 跃升到 85%/89%——这是 generalization 而非 memorization。

### 3.4 Pre-training 失败的 intuition

Table 4 显示先 pretrain 再 finetune 反而比 no co-training 还差（40% vs 50%）。直觉：finetune 阶段 small dataset 会触发 **catastrophic forgetting**，network 在 10K 步里把 static data 学到的 features 完全冲掉；co-training 是 joint optimization，static 和 mobile gradient 在每个 step 同时出现，无 forgetting 问题。

这与 LLM 中 instruction tuning 的发现一致——继续 pretraining 与 instruction tuning 必须交错，否则会"忘掉"pretraining 知识。

---

## 四、Imitation Learning Methods 兼容性

Paper 验证三种 base algorithm 均受益于 co-training，这本身就是 strong claim——说明 co-training 不是某一种 specific algorithm 的副作用。

### 4.1 ACT (Action Chunking Transformer) [104]

**Architecture**:
- Backbone: ResNet18 (pretrained) 提取 image features
- Encoder: 4-layer transformer，输入 proprioception + image embeddings
- Decoder: 7-layer transformer，输出 **chunk size k = 10** 的 action sequence
- Hidden dim 512, feedforward 3200, 8 heads
- $\beta = 10$ 的 KL divergence weight（VAE latent $z$）
- Dropout 0.1
- Learning rate 2e-5, batch size 16

**Action Chunking 公式**:
对每个 timestep $t$，policy 一次性预测未来 $k$ 步：
$$\pi(o_t) = (\hat{a}_t, \hat{a}_{t+1}, \ldots, \hat{a}_{t+k-1})$$
执行时只取前 $k - d$ 步以减小 base-arm 延迟不一致带来的 compounding error。这里 $d$ 是 mobile base 的延迟步数，paper 中 $d \approx 2$。

**Intuition**: chunking 是 ACT 的灵魂——它把 high-frequency policy inference 替换为低频 inference + 高频 playback，既减小 latency，又让 trajectory 在时间维度 coherent（避免 jittery motion）。

### 4.2 Diffusion Policy [18]

**Architecture**:
- 同样 ResNet18 backbone
- Noise predictor: UNet
- DDIM scheduler [85]，training 50 steps, inference 10 steps
- EMA power 0.75
- Chunk size 64
- Image augmentation: RandomCrop 0.95, ColorJitter, RandomRotation ±5°
- Learning rate 1e-4, batch size 32

**训练 objective**:
$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \epsilon, a_0} \left[ \| \epsilon - \epsilon_\theta(a_t, t, o) \|^2 \right]$$
其中 $a_t = \sqrt{\bar{\alpha}_t} a_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$，$t$ uniform in $\{1, \ldots, T\}$，$\bar{\alpha}_t$ 是 noise schedule cumulative product。

**为什么 Diffusion 在 Wipe Wine 上 65% < ACT 95%**: Paper hypothesis 是 50 demos 不够——Diffusion Policy 之前的 works（如 Chi et al.）使用 250+ demos。Diffusion 的 multimodality 在低数据 regime 下容易过拟合到 training distribution。但 Push Chairs 上 100%，因为 Push Chairs 是 quasi-static 任务，multimodality 问题小。

### 4.3 VINN + Chunking [63]

**Architecture**:
- BYOL [37] self-supervised encoder 训练在 combined mobile + static data
- Retrieval: 对 test observation $o$，在 training dataset 中找 $k$-nearest neighbor，用其 action 作为 prediction
- Chunk size 100
- State weight 5, camera feature weight 1:1:1

**重要细节**: VINN 的 co-training 只 co-train BYOL encoder，action retrieval 机制无法直接 leverage OOD static data（因为 static data 的 action 不直接用在 retrieval）。这解释 Table 2 中 VINN 的 mixed results——Wipe Wine 反而退步 5%，Push Chairs 提升 20%。

**Intuition**: VINN 类似于"非线性 nearest-neighbor lookup"，对 visual representation 质量极敏感。Co-training 让 BYOL encoder 见过更多 visual variation，但对 action retrieval 本身没帮助。这与 ACT/Diffusion 的 end-to-end training 不同。

---

## 五、Tasks 设计哲学与实验数据深度解读

### 5.1 七个任务的难度谱系

| Task | Demo 长度 | 难点 | 成功要求 |
|------|----------|------|---------|
| Wipe Wine | 1300 steps (26s) | bimanual coordination + navigation | 抓 towel → 擦 → 放 glass |
| Cook Shrimp | 3750 steps (75s) | dynamics of flipping, low contrast | 倒油 → 倒虾 → 翻 → 装盘 |
| Rinse Pan | 1100 steps (22s) | shiny knob perception, visual servoing | 抓 pan → 开水 → 洗 → 放 |
| Use Cabinet | 1500 steps (30s) | heavy load 1.4kg, bimanual pull doors | 开柜 → 拿锅 → 放 → 关 |
| Call Elevator | 2250 steps (45s) | 15m navigation, 2cm×2cm button precision | 导航 → 按 → 进入 |
| Push Chairs | 2000 steps (40s) | 5kg chair friction, OOD generalization | 推 3 椅 → 测试 5 |
| High Five | 2000 steps (40s) | human-robot interaction, dynamic | 导航 → 互动 |

### 5.2 Table 1 数字解读

Wipe Wine (50 demos):
- Grasp Towel: 100%/95% (co-train/no-co-train)
- Lift Glass and Wipe: 95%/58% ← **这是 co-training 起作用最明显的地方**
- Place Glass: 100%/90%
- Whole Task: 95%/50% (绝对提升 45%)

为什么 Lift Glass and Wipe 差异最大？这个 sub-task 需要 bimanual coordination：左手 lift glass、右手 wipe。Static ALOHA data 大量是 bimanual task，提供了"如何同时控制两个 arm 做不同事"的 prior。Mobile data 50 demos 不足以学到这种 coordination，co-training 直接补上。

Rinse Pan 的 Turn On Faucet 子任务：80%/0%——**绝对提升 80%**！为什么？faucet knob 是 4cm 长 0.7cm 直径的 shiny 金属，光照变化剧烈。Static ALOHA 的 12 类 task 中有 "open plastic portion cup with lid" 等需要 fine approaching 的 task，提供了"如何处理 small shiny object"的 visual prior。

Call Elevator 的 Press Button: 100%/5%——shiny button 2cm×2cm，且需要 base 精确定位。Co-training 提供了"如何抓小物体"的 prior。

### 5.3 Open-Loop Replay 实验（Appendix A.4）

这是 paper 中最 informative 的 sanity check 之一：把 demo 的 base velocity 和 arm action 直接 open-loop replay 20 次，记录 end-effector 位置散布。

结果：**所有 20 次 replay 偏左 ~10cm，散布 ~20cm**。这意味：
1. Mobile base velocity control 有 systematic bias（可能差速轮 calibration 问题）
2. 接触地面 stochastic 摩擦 + delay 引入 random spread
3. Open-loop replay 0% 成功——所以 policy 必须 closed-loop 修正

**Intuition**: 这正是为什么 imitation learning 在 mobile manipulation 上要"困难得多"——每一步的 observation 都已经 drifted，policy 必须从 RGB 实时纠正，而不是依赖 proprioception。Mobile ALOHA 的成功证明：CNN backbone + wrist camera 提供足够 visual servoing signal，policy 内隐地学会了 base-arm coordinated correction。

---

## 六、Action Chunking 与 Delay Compensation

Section 6 提到一个 engineering detail：mobile base 有约 $d$ 步 delay，arm 几乎无 delay。Paper 的 trick：

如果 chunk size $k = 10$，$d = 2$：
- 执行 chunk 前 8 个 arm action（即时响应）
- 同时执行 chunk 后 8 个 base action（base 在第 9,10 步才需要这些 target，已经追上）

这相当于把 action chunk 在时间轴上 **shift** base 部分 $d$ 步，让 hardware 物理延迟与 action sequence 自然对齐。

**Intuition**: 这是 robotics 中经典的 "transport delay compensation"，但在 imitation learning 框架下用 action chunking 优雅地解决，无需显式 system identification。

---

## 七、Failure Modes 与 Limitations

### 7.1 Cook Shrimp 失败分析

Cook Shrimp 是唯一低于 80% 的 task（40%）。Paper hypothesis 是 demo 数太少（20 个）+ 任务 75s 长 + 翻虾 shrimp 与白碗低 contrast。

更深一层分析：翻 shrimp 是 **non-quasi-static dynamic task**——shrimp 与 pan 之间有 stiction + slipping，stir-fry 动作需要 fast wrist motion 在 pan 表面刮擦。这种 dynamic contact 在 50 demos 下根本无法覆盖所有 contact modes。需要 RL fine-tuning 或更多 demos。

### 7.2 Hardware Limitations

1. **Footprint 90×135cm**：通过窄门困难，Paper 提到进入 elevator 只有 30cm clearance。
2. **Fixed arm height**：无法到达低矮 cabinet / oven。Paper 提到未来加 elevation DoF。
3. **Differential drive**：无法侧向平移（holonomic motion 缺失），导致 narrow corridor 任务需要 S-curve maneuver。
4. **Wheeled base**：无法爬楼梯，无法穿越 unstructured terrain（虽然能过 10mm 障碍 + 8° 坡）。

---

## 八、与相关工作的 positioning

### 8.1 与原版 ALOHA [104] 的关系

原版 ALOHA 是 bimanual tabletop，cost $20k。Mobile ALOHA 加 $7k Tracer + battery + compute → $32k。**继承的设计哲学**：low-cost、repairable、puppeteering teleoperation、wrist camera。

### 8.2 与 RT-X [20] 的关系

RT-X 是 2023 年 Google DeepMind 牵头的 cross-embodiment dataset，包括 22 种 robot。Mobile ALOHA 利用 RT-X 中的 static ALOHA subset（825 episodes）做 co-training。这是 RT-X 首次在 bimanual mobile 场景验证 transfer learning 价值。

### 8.3 与 RT-2 [13]、Octo [61]、RoboCat [11] 的对比

这些是 single-arm 大规模 pretrain+finetune 工作：
- RT-2: vision-language-action model，60 robot + 互联网 VLM 知识
- Octo: 800k episodes open-source generalist policy
- RoboCat: self-improving agent

Mobile ALOHA 是 **bimanual + mobile** 的 first demonstration。Co-training 用的是同 morphology（ALOHA）的 static data，不是 cross-embodiment，所以 transfer 更直接。

### 8.4 与 Diffusion Policy [18] 的对比

Diffusion Policy 在 Columbia 的 CoppeliaSim 和 real-world pick-and-place 250+ demos 下表现最好。Mobile ALOHA 只用 50 demos，所以 ACT 更适合。但 Paper 验证 Diffusion 在 Push Chairs 上 100%——quasi-static 任务下 Diffusion multimodality 优势体现，co-training +30%/+20% 也说明 Diffusion 在低数据下 overfitting 被 co-training 缓解。

---

## 九、Intuition 总结：Mobile ALOHA 真正教会我们什么

1. **Whole-body control 不需要分解**：直接 concat 16 维 action vector，无 hierarchy、无 primitives、无 base/arm decomposition，让 transformer 自己学 coordination。这与 [38, 48, 58, 94] 的 decomposed action space 路线形成对比。

2. **Co-training > Pre-training**：当 source 和 target domain 有 overlap（同样的 arm，wrist camera 视角相近）时，joint training 把 source data 当 regularization，比 pretrain+finetune 显著好。

3. **Wrist camera 是 transfer 的关键 invariance**：base 移动改变 world frame 视角，但 wrist camera 视角对 base 几乎不变——这是为什么 static → mobile transfer 能 work 的物理基础。Hsu et al. [41] 的发现在这里被验证。

4. **Hardware simplicity wins**：tether + backdrive wheel 比 VR/mocap/exoskeleton 简单一个数量级，且 collect 的数据更"自然"——operator 本能的 step-back 动作自然成为 demo 的一部分。

5. **Closed-loop 是必需**：open-loop replay 0% 成功，policy 必须从 visual feedback 实时纠错。这说明 imitation learning 在 mobile manipulation 上本质是 visual servoing + motion prior 的混合。

6. **50 demos 是 sweet spot**：对 transformer-based ACT，50 demos 让 co-training 起作用；<25 demos 时即使 co-training 也跌到 ~70%（Figure 4）。

---

## 十、可能的相关联想（hallucination-allowed）

### 10.1 与 LLM 时代的连接

Mobile ALOHA 的 co-training recipe 与 LLM 中 instruction tuning 的"continue pretraining + instruction tuning 交错"极为相似。如果未来要做 VLA (Vision-Language-Action) model 的 bimanual mobile 版本，co-training 公式可推广为：

$$\mathcal{L}_{\text{VLA}} = \mathcal{L}_{\text{internet VLM}} + \mathcal{L}_{\text{static ALOHA}} + \mathcal{L}_{\text{mobile ALOHA}} + \mathcal{L}_{\text{language grounding}}$$

RT-2 [13] 已经做了 internet VLM + robot data，但只在 single-arm 上；把 Mobile ALOHA + RT-2 风格合在一起是显然的下一步。

### 10.2 与 Humanoid Robot 的连接

Mobile ALOHA 是 wheeled bimanual；humanoid（如 Figure 01、Optimus、Unitree H1）是 bipedal bimanual。Paper [68]（Purushottam et al.）做了 wheeled humanoid 的 whole-body bilateral teleoperation 用 force plate。Mobile ALOHA 的 tether 方案可推广到 bipedal——把 operator  tether 到 humanoid torso，operator 用双脚 walking 来 backdrive humanoid（当然 safety 是巨大挑战）。

### 10.3 与 RL Pretraining 的可能结合

Mobile ALOHA 用 supervised BC，limitation 是不能 self-improve。下一步显然是 **offline RL + online RL fine-tuning**：
- 用 BC + co-training 初始化 policy
- 在 simulator 中用 RL fine-tune（domain randomization for base dynamics）
- 在 real-world 用 RL with safety constraint 进一步 refine

这与 RoboCat [11] 的 self-improvement 路线相似，但加入了 mobile base 这个 6 DoF 复杂度。

### 10.4 与 Foundation Model 路线对比

Octo [61] 是 800k episodes 训练的 generalist policy，但 single-arm。要做 bimanual mobile foundation model，需要：
- 大规模 bimanual data（Mobile ALOHA 这种 system 才能大规模 deploy）
- Cross-task transfer 验证（Mobile ALOHA 已验证 7 task）
- Cross-embodiment transfer（不同 base、不同 arm）

Mobile ALOHA + ALOHA 共 ~1000 demos 已经够 7 个 task。如果社区共建 mobile ALOHA fleet，10k demos 可训练一个真正 bimanual mobile foundation model。

### 10.5 与 Simulation 的关系

Paper 完全在 real-world 验证，无 simulation。这限制 reproducibility。可能的 extension：
- 在 Isaac Gym / MuJoCo MJX 中建 Mobile ALOHA URDF
- 用 sim demos + real demos co-training（domain randomization 处理 sim-to-real gap）
- 这与 ManiSkill [49]、RoboCasa 等 benchmark 路线呼应

参考链接：
- MuJoCo MJX: https://mujoco.readthedocs.io/en/stable/mjx.html
- ManiSkill: https://github.com/haosulab/ManiSkill
- RoboCasa: https://robocasa.ai

### 10.6 与 Teleoperation Learning 的连接

User study 显示 5 trial 后 operator 时间下降 52%。这与 [57] Lynch et al. "Learning Latent Plans from Play" 的 play data 路线呼应——如果允许 non-expert 大量 play data 收集，co-training 数据来源可大幅扩展。Mobile ALOHA 的 ergonomics 让 play data collection 成为可能，未来可做大规模 crowdsourcing teleoperation data。

### 10.7 与 DROID Dataset 的关系

DROID（Droid dataset: https://droid-dataset.github.io）是 2024 年的大规模 manipulation dataset，76k trajectories cross-embodiment。Mobile ALOHA 可贡献 mobile bimanual subset 到 DROID，反之 DROID single-arm subset 可作为 co-training source。

### 10.8 关于 Affordance Learning

Mobile ALOHA 学到的 policy 内隐地学到了 kitchen affordance：faucet 在哪、cabinet handle 怎么抓、elevator button 在哪。这与 [4] SayCan 用 LLM 推理 affordance、[80] Concept2Robot 从 instruction 学 affordance 形成 contrast。Mobile ALOHA 走的是 end-to-end visual affordance，没用 explicit language。下一步可以加 language conditioning（如 RT-2）。

---

## 十一、Reimplementation 细节补遗

如果想要复现 Mobile ALOHA：

1. **Hardware BOM**:
   - AgileX Tracer: $7,000
   - 2× ViperX 300 6DOF: ~$12,000
   - 2× ViperX 300 leader arm (用于 teleop): ~$5,000
   - 1.26kWh lithium battery + BMS: ~$1,500
   - Laptop with 3070 Ti: ~$2,000
   - 3× Logitech C922x: $300
   - 3D printed mounts + tether + cables: ~$1,000
   - Total: ~$29k-$32k

2. **Software stack**:
   - ROS 2 或自定义 Python 控制循环
   - Tracer CAN bus protocol
   - ViperX 300 通过 USB serial (leader) + 通过同 USB serial (follower)，position control mode
   - Camera streaming 用 OpenCV + multiprocessing
   - 数据存储用 HDF5 或 zarr (推荐 zarr + DKL format，与 DROID 兼容)

3. **Training infra**:
   - ACT codebase: https://github.com/MarkFzp/act-plus-plus
   - Diffusion Policy codebase: https://github.com/real-stanford/diffusion_policy
   - VINN codebase: https://github.com/jyp110/VINN
   - Co-training 时把 static ALOHA data 转为同 format，zero-pad base dim

4. **关键 hyperparameter**: chunk size $k = 10$ for ACT, $k = 64$ for Diffusion, $k = 100$ for VINN。这是 paper Section A.3 透露的——chunk size 与 algorithm 的 temporal modeling 能力正相关。

---

## 十二、结语

Mobile ALOHA 在我看来是 robotics "iPhone moment" 之一：它把 bimanual mobile manipulation 从 $200k+ PR2 / TIAGo 级别的研究设施，democratize 到任何标准 lab 用 $32k 就能复现。Co-training 的 finding 则预示了一个新范式：**future robot foundation model 不会只 train 在 robot data 上，而是 train 在所有可用 robot data + 可迁移 human data + internet vision-language data 上**，每种 data 通过 zero-padding / masking 自然融入统一 framework。

Paper 的 limitation 也清晰：单任务 BC、wheeled base、固定 arm height、无 self-improvement。下一步显然是：bipedal humanoid + multi-task + RL fine-tuning + language grounding。Mobile ALOHA 提供了这块拼图的 hardware 和 co-training baseline。

参考更多资源：
- Stanford IRIS Lab: https://iris.stanford.edu
- Stanford REAL Lab: https://real.stanford.edu
- Zipeng Fu personal page: https://zipengfu.com
- Tony Zhao personal page: https://tonyzhaozh.github.io
- Chelsea Finn lab: https://ai.stanford.edu/~cbfinn

如果你（Andrej）对其中某一处技术细节——比如 ACT 的 CVAE latent、Diffusion Policy 的 DDIM 10 步推理实现、或者 tether 的 ergonomics 设计——想进一步深挖，告诉我，我可以拆得更细。
