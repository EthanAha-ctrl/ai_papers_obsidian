---
source_pdf: Learning from Massive Human Videos for Universal Humanoid Pose Control.pdf
paper_sha256: 4393053e06c5437f658addb813559fbc1785c36761cee25eee1eaf91c9933b30
processed_at: '2026-08-05T12:57:20-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用人话讲，这篇 paper 的核心 idea 非常直觉。想象一下你想教一个 humanoid robot 跳霹雳舞或者打拳击。传统方法要么得在仿真里用 reinforcement learning 调几个月 reward function，要么得人穿上一套动捕服用手把手 teleop 教它，数据获取成本极高且难以 scale。这帮作者发现了一个绝妙的漏洞：互联网上已经有几千万个人类跳霹雳舞、打高尔夫、弹吉他的视频了，人类和 humanoid robot 的骨架又长得那么像，为什么不直接把 YouTube 视频当成 robot 的教学 demo 呢？

整个工作其实就是把人类视频转化为 robot 动作的自动化工厂流水线，然后在流水线产出的 2000 万帧数据上训练一个 GPT 模型来生成动作。

下面我分模块把里头的门道、公式细节和实验数据给你拆解一下，同时尽量多地延伸相关联想。

### 1. 数据工厂流水线: 从 YouTube 视频到可执行的 Robot Action

要把一段 wild YouTube 视频变成真机可执行的动作，中间隔着五道工序。

**第一道：Video Mining**
作者设计了 400 多个 search term（如 "karate front kick training"），从 YouTube 和学术数据集狂下视频。为了过滤掉无关画面，他们用 YOLOv8 做单帧单人检测，只保留画面里恰好有一个人的帧。同时计算灰度图的 pixel difference，只保留有显著运动的片段，最后拼成至少 64 帧的 clip。

**第二道：Video Captioning**
为了让之后能用自然语言控制 robot，需要给视频打标签。这里直接调用现成的 video-LLM（Video-LLaMA 2），并设计了 prompt 强制模型只描述动作（如 "a man doing a backflip"），不许描述人的长相穿着，确保标签 action-centric。

**第三道：3D Human Pose Estimation**
这里用 VIBE 模型从 2D 视频反推 3D 的 SMPL 人体参数。
公式： $\mathcal{P}_{human}(\beta, \theta, t_{root}) = F_{pose}(\mathcal{V})$
变量解释：$\mathcal{V}$ 是 video clip，$\mathcal{P}_{human}$ 是输出的人体姿态，$\beta$ 控制体型胖瘦，$\theta$ 是各个关节的旋转角度，$t_{root}$ 是人体根节点的全局 3D 平移。

**第四道：Motion Retargeting**
这一步是把人类动作“套”到 Unitree H1-2 robot 上。因为人高马大和 robot 比例不同，直接套关节角度会极其违和。作者的解法很巧：先优化 SMPL 的 shape parameter $\beta$，让人类在 T-pose 下的 12 个关节位置和 robot 的 12 个关节位置一模一样。
公式： $\min_{\beta} \| \mathcal{P}_{joints}^{T} - \mathcal{P}_{robot}^{T} \|_2$
上标 $T$ 表示标准的 T-pose，$\mathcal{P}_{joints}^{T}$ 是人类 T-pose 下的关节坐标，$\mathcal{P}_{robot}^{T}$ 是 robot 的出厂 T-pose 坐标。约束条件是 $|\beta_i| < 5$，防止 $\beta$ 暴走把人体模型扭曲成怪物。
调整好体型后，直接把人类关节的 3D 位置赋给 robot，再通过 Inverse Kinematics (IK) 算出 robot 的 DoF (Degree of Freedom) 目标角度。IK 求解时加了二阶差分平滑惩罚 $\mathcal{L}_s = \sum (2q[i] - q[i-1] - q[i+1])$，保证动作没有突变的顿挫感。

**第五道：Goal-conditioned RL Policy**
经过第四步，我们有了 kinematic 层面的 robot 目标动作，但这在物理上是不安全的，直接发给真机必定摔跤。所以需要训练一个 RL policy 充当“底层保镖”。
公式： $\pi : \mathcal{G} \times \mathcal{O} \mapsto \mathcal{A}_{robot}$
$\mathcal{G}$ 是 goal（上层给的目标 keypoint 和 DoF），$\mathcal{O}$ 是 observation（robot 本体的角速度、重力投影等），$\mathcal{A}_{robot}$ 是输出的 27 维 target DoF positions。
用 PPO 在 Isaac Gym 里开 6192 个并行环境狂训，reward 包含 imitation tracking reward 和各种 stability regularization。一个硬核工程细节：为了保护真机极易损坏的 ankle roll joint，在 sim 训练时每一步都强行把这个关节的 action 清零。这种 hardware-aware 的设计是 sim-to-real 成功的关键。

### 2. UH-1 模型架构：动作领域的 GPT

有了 2000 万帧的 $\langle \text{Text}, \text{Action} \rangle$ pair，就可以训大模型了。作者没有用 Diffusion，选择了 autoregressive Transformer，因为更符合 scaling law。

**Action Tokenizer (VQ-VAE)**
连续的 27 维 action vector 没法做 autoregressive 预测，得先变成离散的 token。这里用 1D 卷积把每 $K=4$ 帧压缩成一个 token。Codebook size 设为 $2048 \times 512$。
传统的 VQ-VAE 重建 loss 只管绝对位置，这里作者加了一个极为关键的 first-order similarity loss：
$\mathcal{L}_{recon} = \sum_i^T (|a_i' - a_i| + |(a_{i+1}' - a_i') - (a_{i+1} - a_i)|)$
$a_i$ 是第 $i$ 帧的 ground truth action，$a_i'$ 是重建出的 action。第一项是位置 L1 loss，第二项是速度（一阶差分）的 L1 loss。由于相邻帧动作变化极小，如果只优化位置 loss，模型容易学到偷懒输出常量，加上速度约束强制它关注 motion 的 differential structure，生成时才不会卡顿或者抖动。

**UH-1 Transformer**
用 CLIP text encoder 提取文本特征 $l$，然后丢进一个 18 层、1024 维的 GPT 架构里，auto-regressively 预测下一个 action token。
训练目标： $\mathcal{L}_{learn} = -\sum \log \prod p(z_i | z_{1:i-1}, l)$
$z_i$ 是要预测的第 $i$ 个 action token，$z_{1:i-1}$ 是已经生成的前序 tokens，$l$ 是 text embedding。这和 LLM 预测下一个 word token 在数学上完全同构。 humanoid 动作被彻底当作一种“外语”来学。

### 3. 两种控制模式与实验数据表

UH-1 支持两种 deploy 模式。
Text-to-Keypoint (闭环)：生成高层 keypoint 目标，喂给前面训好的 RL policy 兜底，适合全身联动和行走。
Text-to-Action (开环)：直接生成底层 DoF angle 发给 PD controller，由于没有 RL 纠错，只能控制上半身，下半身靠另一个预训练的 locomotion policy 站着。

**HumanoidML3D Benchmark 结果**
作者把 HumanML3D 数据集 retarget 成了 humanoid 版本进行对比测试。

| Methods | FID↓ | MM Dist ↓ | Diversity ↑ | R Precision ↑ |
| :--- | :--- | :--- | :--- | :--- |
| Oracle | 0.005 | 3.140 | 9.846 | 0.780 |
| MDM | 0.582 | 5.921 | 10.122 | 0.617 |
| T2M-GPT | 0.667 | 3.401 | 10.328 | 0.734 |
| UH-1 (ours) | **0.445** | **3.249** | 10.157 | **0.761** |

指标解释：FID 衡量生成动作分布与真实分布的相似度，越低越好；MM Dist 衡量文本和动作的匹配距离；R Precision 衡量检索准确率。UH-1 在 FID 上比 two-stage 的 MDM 降了 23%，这证明 end-to-end 直接生成 humanoid action 比先生成 human motion 再 retarget 更优，避免了误差级联。

**Real-World 真机实验结果**
在 Unitree H1-2 上测试了 12 个 text instruction，每个跑 10 次。比如 Boxing (90%), Clapping (100%), Play Guitar (100%), Wave to Friend (100%)。综合成功率接近 100%。这是非常惊艳的真机结果，证明了从海量 YouTube 视频学来的动作确实能直接迁移到物理硬件上。

### 4. Intuition 总结与疯狂联想

这篇 paper 极其漂亮的点在于把机器人学习彻底解耦成了 data scaling 问题和 LLM next-token prediction 问题。传统的 robot learning 苦哈哈地在实验室里调参，这篇工作直接搞了个 web-scale 数据自动化管线。

1. ** Scaling Laws 的平移**：论文做了 dataset scaling ablation，从 1% 增到 100%，FID 线性下降，Diversity 上升。这证明 humanoid control 领域同样存在类似 LLM 的 Chinchilla scaling law。只要 YouTube 上的视频不缺，这个框架可以无限 scale 下去。
2. ** Sim-to-Real 的范式转移**：以前的 sim-to-real 是在 sim 里学 policy 直接迁真机。现在是 internet video -> sim retargeting -> sim RL safety filter -> real robot。互联网视频成了最重要的数据源，sim 只是加工厂和试炼场。
3. ** 与最新 VLA 架构的联想**：现在的 $\pi_0$、GR00T 或者 Figure 的 Helix 大多采用 Diffusion policy 或 Flow matching。UH-1 选了 Autoregressive Transformer 证明了后者在 motion 生成上的 scaling 潜力。下一步如果 UH-1 把视觉信号（如 DINOv2 特征）也作为 condition 纳入，就可以做成一个完整的 Vision-Language-Action (VLA) 模型，实现真正的 loco-manipulation（边走边抓东西）。
4. ** 误差传播的隐忧**：整条管线极其长，VIBE 估计误差、Shape $\beta$ 优化误差、IK 求解误差、VQ-VAE tokenizer 重建误差。能跑出 100% 成功率说明 humanoid control 对 motion 噪声的容忍度其实挺高，也许 upper body 动作的本质就是寻找一个 dynamic stable 的 basin。
5. ** AGI 具身化的垫脚石**：人类之所以通用，是因为我们从小看别人怎么做就学会了怎么动。这个工作本质上是给 robot 装了一个“看 YouTube 学习”的大脑。未来如果我们把 web 上的第一人称教学视频也喂进去，配合 ego-centric vision，robot 就能真正看视频学会做菜、修车了。

总而言之，这是一个 data engineering、RL control 和 large model 三位一体的精彩工作，把 humanoid learning 推到了 internet-scale pre-training 的时代。

### Web links for reference:
1. UH-1 Project Page: https://usc-gvl.github.io/UH-1
2. VIBE (video 3D pose estimator): https://github.com/yfeng95/VIBE
3. HumanML3D (dataset & metrics): https://github.com/EricGuo5513/HumanML3D
4. T2M-GPT (baseline model): https://github.com/Mael-zos/T2M-GPT
5. MDM (Motion Diffusion Model): https://github.com/GuyTevet/motion-diffusion-model
6. Unitree H1 (robot platform): https://www.unitree.com/h1
7. Video-LLaMA 2 (captioning): https://github.com/DAMO-NLP-SG/VideoLLaMA2
8. VQ-VAE (tokenizer backbone): https://arxiv.org/abs/1711.00937
9. CLIP (text encoder): https://openai.com/research/clip
10. Isaac Gym (RL simulator): https://developer.nvidia.com/isaac-gym

---

# Learning from Massive Human Videos for Universal Humanoid Pose Control 深度讲解

Andrej，这篇 paper 由 USC 的 Yue Wang group 联合 Jitendra Malik (UC Berkeley) 和 Toyota Research Institute 完成，核心 idea 是**把互联网人类视频当作 humanoid robot 的 demonstration 来源**，绕开 robot teleoperation 的 data collection bottleneck。下面我尽可能细地展开。

Project page: https://usc-gvl.github.io/UH-1

---

## 1. Motivation 与 problem setting

机器人学习的 scaling 问题一直是核心痛点。vision / language 领域靠 web-scale 数据得到了 foundation models（DINOv2 [1]、CLIP [2]、SAM [3]），但 robot demonstrations 仍然昂贵且小规模。RT-1/RT-2 [4,5]、OpenVLA [6]、Open X-Embodiment [7] 这些 manipulation foundation model 虽然在 data 上 scale，但数据获取成本高、多样性受限于 teleop 和 scripted demos。

对于 humanoid robot，问题更尖锐：
- DOFs 远多于机械臂（UH-1 用 Unitree H1-2，27 个 actuated DOFs + 7-dim root state）
- traditional RL 方法 [8,9,10,11] 主要聚焦 locomotion / jumping 这种 narrow skill set
- teleop 方法 [12,13,14,15] 依赖 mocap 或 VR suit，scale 不起来
- humanoid skeleton 与 human 相似 → **internet human videos 是免费的、海量、diverse 的 demonstration**

关键 question：能否从 raw internet videos 学到 **language-conditioned universal humanoid pose control**？答案就是这个 paper 的 Humanoid-X dataset + UH-1 model。

Reference: 类似 idea 在 manipulation 领域有 affordance learning [16]、flow learning [17,18]、world model [19]；human motion generation 领域有 MDM [20]、T2M-GPT [21]、HumanML3D [22] 等。这篇把这些 idea 首次大规模迁移到 humanoid control。

---

## 2. Humanoid-X dataset 的 5 种 modalities

每个 motion sample 是一个 5-tuple：

$$
\langle \mathcal{V}, \mathcal{T}, \mathcal{P}_{human}, \mathcal{P}_{robot}, \mathcal{A}_{robot} \rangle
$$

- $\mathcal{V}$：原始 video clip（20 fps MP4）
- $\mathcal{T}$：text caption
- $\mathcal{P}_{human}(\beta, \theta, t_{root})$：SMPL-based 3D human pose，其中 $\beta$ 控制 shape，$\theta$ 控制 joint rotations，$t_{root}$ 是 global root translation
- $\mathcal{P}_{robot}$：humanoid robot keypoints（12 个 joints 的 3D 位置），用于 high-level control
- $\mathcal{A}_{robot}$：target DoF positions（27-dim），用于 low-level PD control

总规模：163,800 motion samples，240.3 小时，20.7M frames，11,897 word vocabulary。这个规模比 HumanML3D（约 14K clips）大一个数量级以上。

---

## 3. Data pipeline 五个 stage

### Stage 1: Video mining

来源混合：
- Academic digital human datasets：AIST、AMASS [23]、Charades、EgoBody、GRAB、HAA500、HuMMan、IDEA400
- Action understanding：Kinetics700
- Internet：YouTube，用 Google Cloud API 搜 400+ 个 query term

Query term 设计覆盖 8 大类：martial arts、fitness、sports、dance、music performance、daily activities、animal-inspired、rehabilitation。比如 "karate front kick training"、"yoga handstand practice"、"tennis serve technique tutorial"、"violin bowing technique while standing demonstration"。这种 query 设计很关键——避免直接搜 "single person"，因为视频 title 不会这么写。

Video processing pipeline：
1. 统一下采样到 20 fps
2. YOLOv8 [24] 做单人检测，保留恰好 1 个人的 frame
3. 计算 ROI 内 consecutive frame 的 grayscale 像素差，保留 motion 显著的 frame
4. 拼接 ≥64 consecutive frames 形成 clip
5. 这个 batch-based filtering（小 threshold per frame + 大 threshold per batch）保证 clip 内 motion 连续

最终 163,800 clips。

### Stage 2: Video captioning

用 Video-LLaMA 2 [25] 生成 caption：

$$
\mathcal{T} = F_{caption}(\mathcal{V})
$$

Prompt 设计很考究，强制：
- 不描述 appearance
- 必须含 "a man/woman doing something [adverb]"
- 描述 interaction item / body part / location
- 一句话，不以 "in the video" 开头

这样保证 caption 是 action-centric 而不是 scene-centric。文本 vocabulary 11,897 个 word，其中 noun 6048、verb 3206，足够 diverse。

### Stage 3: 3D human pose estimation

用 VIBE [26]（video-based SMPL estimator）从 video 得 SMPL 参数：

$$
\mathcal{P}_{human}(\beta, \theta, t_{root}) = F_{pose}(\mathcal{V})
$$

Root translation 用 weak-perspective camera 参数反推。给定 scale $s$ 和 2D translation $\mathbf{t}=(t_x, t_y)$、focal length $f$、image width $W_{img}$：

$$
t_z = \frac{f}{s \cdot 0.5 \cdot W_{img}}
$$

root translation vector $\mathbf{T}_{root} = (t_x, t_y, t_z)$。这里用 weak-perspective 反推 depth 是常见 trick，精度有限但够用，因为后续 retarget 只需要相对 motion。

### Stage 4: Motion retargeting

这是把 human motion 转到 humanoid 的核心。选了 12 个共同 joint（hip、knee、ankle、shoulder、elbow、wrist 各左右）。

**Step 4a：shape parameter 优化**
因为 human 和 humanoid 的 limb proportion 不同，先优化 $\beta$ 让 human 的 T-pose joint positions 匹配 humanoid 的 T-pose joint positions：

$$
\min_{\beta} \| \mathcal{P}_{joints}^{T} - \mathcal{P}_{robot}^{T} \|_2
$$

subject to:
$$
\mathcal{P}_{joints}^{T} = F_{fk}(\mathcal{P}_{human}(\beta, \theta^T, t_{root}))
$$

其中 $F_{fk}$ 是 human 的 forward kinematics，$\theta^T$ 是 T-pose 的 joint rotation。

为了避免 $\beta$ 把 human mesh 弄得变形过度，限制 $|\beta_i| < 5$。这是个重要 regularizer——否则优化器会 hack 出畸形的"human"来匹配 humanoid 比例。

**Step 4b：keypoint transfer**

用最优 $\beta^*$ 替换原 $\beta$，再做 FK 得到调整后的 joint positions $\mathcal{P}_{joints}'$，直接 set 为 humanoid keypoints：
$$
\mathcal{P}_{robot} := \mathcal{P}_{joints}'
$$

**Step 4c：inverse kinematics 得 DoF positions**

$$
q_{robot} = F_{ik}(\mathcal{P}_{robot})
$$

用 Adam optimizer [27] 解 IK，loss：
$$
\mathcal{L}_{ik} = \mathcal{L}_r + \lambda \mathcal{L}_s
$$

retarget loss：
$$
\mathcal{L}_r(q_{robot}, s_{root}) = \| F_{rk}(q_{robot}, s_{root}) - \mathcal{P}_{robot} \|_1
$$

其中 $F_{rk}$ 是 robot 的 forward kinematics，$s_{root}$ 是 root translation + orientation。

smoothing term（二阶差分）：
$$
\mathcal{L}_s(q_{robot}) = \sum_{i=1}^{n-2} (2q_{robot}[i] - q_{robot}[i-1] - q_{robot}[i+1])
$$

$\lambda = 0.05$。这个二阶差分惩罚加速度突变，对后续 RL policy 的稳定 tracking 很重要。

### Stage 5: Goal-conditioned RL policy

$\mathcal{P}_{robot}$ 和 $q_{robot}$ 是 kinematic 的，没有 physical validity。要 deploy 到真机还需要一个能处理 dynamics、balance、torque limit 的 policy $\pi$：

$$
\pi : \mathcal{G} \times \mathcal{O} \mapsto \mathcal{A}_{robot}
$$

Goal space $\mathcal{G} = \mathcal{G}^e \times \mathcal{G}^m$：
- $\mathcal{G}^e$：joint angles + keypoint translations（upper body expression）
- $\mathcal{G}^m = \langle \mathbf{v}, rpy, h \rangle$：root linear velocity、roll/pitch/yaw、body height

Observation $\mathcal{O} = [\omega_t, r_t, p_t, \Delta y, q_t, \dot{q}_t, \mathbf{a}_{t-1}]^T$：
- $\omega_t$：root angular velocity
- $r_t, p_t$：root roll, pitch
- $\Delta y = y_t - y$：current vs desired yaw 差
- $q_t, \dot{q}_t$：joint position & velocity
- $\mathbf{a}_{t-1}$：上一时刻 action

Output action $\mathcal{A}_{robot} \in \mathbb{R}^{27}$：每个 joint 的 target position，经 PD controller 转 torque：
$$
\tau = K_p(\mathbf{a} - q) - K_d \dot{q}
$$

Training 用 PPO [28]，6192 parallel envs in Isaac Gym，21 timesteps per rollout，5 epochs，4 minibatches，learning rate 1e-3，GAE $\lambda=0.95$，discount 0.99。

**Reward 设计很关键**，分三组：

**Imitation rewards** (Table 2 in appendix)：
- DoF position: $\exp(-0.7\|\mathbf{q}_{tar} - \mathbf{q}\|)$，weight 3.0
- Keypoint position: $\exp(-\|\mathbf{t}_{tar} - \mathbf{t}\|)$，weight 2.0
- Root linear velocity: $\exp(-4.0\|\mathbf{v}_{tar} - \mathbf{v}\|)$，weight 6.0
- Root roll & pitch: $\exp(-\|\Omega_{tar}^{\phi\theta} - \Omega^{\phi\theta}\|)$，weight 1.0
- Root yaw: $\exp(-|\Delta y|)$，weight 1.0

**Regularization rewards** (Table 3)：包括 feet height、time in air、drag、contact force、stumble、DoF acceleration、action rate、energy、collision、DoF limit violation、DoF deviation、vertical velocity、horizontal angular velocity、projected gravity。这些是 humanoid RL 的标准 stability regularization，参考了 [9,11,29]。

**细节亮点**：为保护真机的 ankle roll joint，每步强制把这两个 DOF 的 action 设为 0。这是 hardware-aware policy design 的典型 trick——sim 没问题不代表真机没问题。

最终这个 policy $\pi$ 在 humanoid action $\mathcal{A}_{robot}$ 这一 modality 产出**physically valid + safe** 的 action sequence。

---

## 4. UH-1 model architecture

UH-1 是 language-conditioned 的 autoregressive generation model，从 text $\mathcal{T}$ 生成 $\{\mathcal{P}_{robot}, \mathcal{A}_{robot}\}$：

$$
\pi_{UH-1} : \mathcal{T} \mapsto \{\mathcal{P}_{robot}, \mathcal{A}_{robot}\}
$$

### 4.1 UH-1 Action Tokenizer (VQ-VAE based)

参考 [30]，把连续 action 序列 $\mathcal{A}_{robot} = [a_1, \ldots, a_T] \in \mathbb{R}^{T \times d_1}$ 离散化。

**Encoder $E$**：1D conv + residual blocks + ReLU，stride 2 下采样。temporal downsampling rate $k=4$——也就是每 4 帧压成 1 个 token。

**Codebook** $C = \{c_1, \ldots, c_N\}$，$c_n \in \mathbb{R}^{d_2}$。paper 里 $N = 2048 \times 512$（这里是 codebook size × code dim）。

**Quantization**：
$$
\hat{z}_i = \arg\min_{c_n \in C} \|z_i - c_n\|_2
$$

**Decoder $D$**：nearest-neighbor upsampling + conv，重建 $\mathcal{A}_{robot}' = D(\hat{Z})$。

**Loss**：
$$
\mathcal{L}_{vqvae} = \mathcal{L}_{recon} + \mathcal{L}_{embed} + \alpha \mathcal{L}_{commit}
$$

其中：
$$
\mathcal{L}_{embed} = \|sg[Z] - \hat{Z}\|_2, \quad \mathcal{L}_{commit} = \|Z - sg[\hat{Z}]\|_2
$$

$sg[\cdot]$ 是 stop-gradient，这是 VQ-VAE 的 standard EMA 替代版本（这里用 commit loss 直接 update codebook）。

**关键改造**：reconstruction loss 不只是 L1，而是加了 first-order similarity 和 root regularization：

$$
\mathcal{L}_{recon} = \sum_i^T \big(|a_i' - a_i| + |(a_{i+1}' - a_i') - (a_{i+1} - a_i)|\big)
$$

第一项是 L1，第二项是**一阶差分（速度）的 L1**。这个改动非常重要——humanoid action 在相邻帧间变化不大，纯 L1 重建容易得到 jittery 的 trajectory，加上 first-order similarity 可以约束**速度层面的一致性**，对生成 motion smoothness 有直接帮助。

更细的 appendix 版本：
$$
\mathcal{L}_1(X, X_{re}) + \beta \mathcal{L}_1(\Delta[X], \Delta[X_{re}]) + \gamma \mathcal{L}_1(X_{re}^{root}, \mathbf{0})
$$

第三项 $\mathcal{L}_1(X_{re}^{root}, \mathbf{0})$ 是 root regularization——惩罚重建后 root 的非零项，大概是想让 codebook 更聚焦在 body motion 而不是 global translation（global translation 由 $\mathcal{G}^m$ 单独 track）。

**为什么 $k=4$ 而不是 $k=1$？**
作者明确指出："humanoid actions won't change much in adjacent frames"，所以每个 token 表示一段短 clip，既保持 temporal smoothness 又 ease learning。这也是 video generation 领域（如 MAGVIT [31]）的常见做法。

### 4.2 UH-1 Transformer

18-layer Transformer，16 attention heads，$d_{model}=1024$，类似 GPT 架构。

Input sequence：text embedding $l$（由 CLIP text encoder 编码 $\mathcal{T}$）+ action token sequence $\mathcal{Z}_{token} = [z_1, \ldots, z_{T/K}]$。

训练目标：autoregressively 预测下一个 codebook index：
$$
\mathcal{L}_{learn} = -\sum_{\mathcal{Z} \in \mathcal{D}} \log \prod_{i=1}^{|\mathcal{Z}|} p(z_i | z_{1:i-1}, l)
$$

加入 [End] token 表示生成终止。

**为什么 Transformer over Diffusion？**
作者做了 ablation（Table 4）：Diffusion FID 0.624、MM Dist 5.536；Transformer FID 0.379、MM Dist 3.232。Transformer 显著更好。作者的解释是 Transformer 在 large-scale data 上更 scalable——这与 LLM 领域的 empirical 经验一致。Diffusion 在 motion generation 上有优势（如 MDM [20]、MotionDiffuse [32]）是当 dataset 较小时，但 data scale 上去后 autoregressive transformer 更能利用 scaling law。这个观察与 DART [33]、LlamaGym 类工作一致。

### 4.3 两种 control mode

**Text-to-Keypoint mode (closed-loop)**：
UH-1 生成 $\mathcal{P}_{robot}$ 序列 → 喂给 goal-conditioned RL policy $\pi$ → $\pi$ 输出 $\mathcal{A}_{robot}$ → PD 控制。整个链路有 RL policy 兜底，robustness 更高。

**Text-to-Action mode (open-loop)**：
UH-1 直接生成 $\mathcal{A}_{robot}$ 序列 → PD 控制。但作者发现 open-loop 只对 upper body 可靠，所以 lower body 用 pre-trained locomotion policy。这是个工程上的折衷——upper body 不涉及 balance，open-loop 可以；lower body 必须有 closed-loop feedback。

Fig 8 实验显示两种 mode 平均 success rate 都 >89%，keypoint mode 略高（95%+），action mode 略低（90%）。

---

## 5. Experiments

### 5.1 HumanoidML3D benchmark

作者把 HumanML3D [22] 用同样的 retarget pipeline 转成 humanoid 版本，叫 HumanoidML3D，作为 benchmark。

对比 MDM [20] 和 T2M-GPT [21]（two-stage baseline：先生成 human motion 再 retarget）。

| Method | FID↓ | MM Dist↓ | Diversity↑ | R Precision↑ |
|---|---|---|---|---|
| Oracle | 0.005 | 3.140 | 9.846 | 0.780 |
| MDM | 0.582 | 5.921 | 10.122 | 0.617 |
| T2M-GPT | 0.667 | 3.401 | 10.328 | 0.734 |
| UH-1 | **0.445** | **3.249** | 10.157 | **0.761** |

FID 改善 23%，R Precision 接近 oracle。**end-to-end 直接生成 humanoid action 比先生成 human motion 再 retarget 更好**——因为 retarget 是 lossy 的，且两阶段 error 累积。

### 5.2 Scaling 效果

预训练在 Humanoid-X → finetune 在 HumanoidML3D：
- FID: 0.445 → 0.379
- MM Dist: 3.249 → 3.232
- Diversity: 10.157 → 10.221

数据 scaling：1% → 100% Humanoid-X，FID 从 0.689 降到 0.463，Diversity 从 5.900 涨到 6.149。**清晰的 scaling law**，类似 LLM 的 power law。

### 5.3 Codebook size ablation

512 → 1024 → 2048，FID 0.539 → 0.463，Diversity 6.050 → 6.149。更多 motion primitive 带来更 diverse 的 generation。作者说没试更大是因为 computational resource 限制。

### 5.4 真机实验

Unitree H1-2，12 个 instruction，每个跑 10 次：

| Instruction | Text-to-Keypoint | Text-to-Action |
|---|---|---|
| Boxing | 90% | 70% |
| Clapping | 100% | 100% |
| Cross Arms | 80% | 80% |
| Embrace | 100% | 100% |
| Golf Putt | 90% | 100% |
| Open Bottle & Drink | 100% | 100% |
| Play Guitar | 100% | 100% |
| Play Violin | 100% | 80% |
| Pray | 100% | 100% |
| Left Hand Punch | 100% | 100% |
| Right Hand Punch | 100% | 90% |
| Wave to Friend | 100% | 100% |

平均接近 100% success rate。这是非常 strong 的 real-world 结果。

---

## 6. 我的 intuition 与 critical thoughts

### 6.1 这是 robot learning 的 "ImageNet moment" for humanoid

类似 ImageNet [34] 让 vision 从 small dataset 进到 web-scale，Humanoid-X 让 humanoid control 从 narrow skill learning 进到 language-conditioned universal control。pipeline 完全 automated，意味着可以无限 scale——只要有更多 video query term 和更多 YouTube 爬取，就能扩展到更多 action categories。

### 6.2 与 LLM 类比

UH-1 的架构就是 GPT 跑在 motion token 上：text → motion token sequence。codebook 是 "motion vocabulary"，类似 BPE token。这种 framing 让 humanoid learning 直接受益于 LLM 的 scaling law 经验。

### 6.3 关键 design choice：first-order similarity loss

这是 paper 的小细节但很关键。motion 与 language 不同——adjacent frames 高度 correlated。如果只用 L1 reconstruction，codebook 会学到 "static" token，generation 容易 stuck。加上 first-order similarity 强制 codebook 也编码 motion 的 differential structure，让 autoregressive 生成能 escape stationary mode。这个 trick 在 audio codec (SoundStream [35], EnCodec [36]) 也有类似设计。

### 6.4 与 ExBody / OmniH2O / HumanPlus 的区别

这些 [9,12,14] 都是 teleop 工作，需要 human 实时控制。UH-1 是 autonomous agent，从 video 学习后可以独立按 text 执行。这是 **policy generalization vs. teleoperation fidelity** 的根本不同。

### 6.5 Limitations 作者承认的

- 只做 pose control，没做 manipulation（没 gripper）
- 没做 loco-manipulation
- codebook size 没能 scale 更大

### 6.6 我想到的潜在问题

1. **VIBE 在 in-the-wild video 上 SMPL 估计精度有限**，特别是 fast motion、occlusion、unusual pose。Error 会通过 retarget 传到 $\mathcal{A}_{robot}$。可以试着用更现代的 estimator，比如 4DHumans [37]、SLAHMR [38]。
2. **First-order similarity 在 VQ-VAE 里的 codebook collapse 问题**：作者没讨论 codebook utilization。可能大量 code 没被用，是 dead code。
3. **Closed-loop vs. open-loop 的 gap**：paper 显示 closed-loop 略好，但只略好。意味着 RL policy 已经足够 robust 到 handle keypoint noise。这反过来问：如果 RL policy 这么 robust，是不是直接学 text→$\mathcal{A}_{robot}$ 的 end-to-end 就够了？keypoint 这一层是否必要？paper 里 keypoint mode 主要提供 whole-body control（包括 locomotion），action mode 只 upper body——所以 keypoints 的核心价值是作为 locomotion goal interface，而不是作为 generation target。
4. **Long-horizon generation**：autoregressive transformer 在 long motion 上 error accumulation 是已知问题。这里 motion 都比较短（<10s 居多），如果要做 30s+ 的 choreographed motion，可能需要 hierarchical generation 或 retrieval-augmented 方案。
5. **Text 是 video caption 不是 instruction**：caption 描述 video 里的 action，但 user 给 robot 的 command 可能更 abstract / compositional（"先挥手再转身再坐下"）。这个 gap 需要进一步 instruction tuning。
6. **Diversity metric 在 Table 1 上 oracle 是 9.846，UH-1 是 10.157，反而更高**——这意味着生成分布比 ground truth 还 broad。可能 small FID 与 high diversity 的 trade-off 被推到了某个 sweet spot。
7. **与 RT-2 / OpenVLA 的对照**：那些 manipulation VLA 把 image 作为 input，UH-1 只接 text。这意味着 UH-1 不能 perceive scene。要 deploy 到真机做 loco-manipulation，必须加 vision encoder（如 SigLIP [39] 或 DINOv2 [1]），把 UH-1 升级成 VLA-style architecture。这应该是 future work 的 obvious next step。

### 6.7 联想到的相关方向

- **π0 [40]、Helix [41]、GR00T**：这些是最新 VLA，处理 whole-body manipulation，可以借鉴他们的 flow matching / diffusion policy head 替换 UH-1 的 autoregressive head。
- **World model for humanoid**：Genie 2 [42]、V-JEPA 2 [43] 的 generative world model 可以做 planning，UH-1 可以扩展为 world-model-based control。
- **AMASS [23] → humanoid retargeting at scale**：这篇的 retarget pipeline 完全 automated，可以应用到 AMASS 的全部 40+ hours mocap。
- **Video pre-training for control**：MVP [44]、R3M [45]、VIP [46] 的 contrastive video pre-training 思想可以用到 humanoid observation encoder。
- **Helix (Figure AI)** [41]：类似 idea，但用 flow matching 而非 autoregressive transformer，可以对照阅读。

---

## 7. Web references

1. DINOv2: https://arxiv.org/abs/2304.07193  
2. CLIP: https://arxiv.org/abs/2103.00020  
3. SAM: https://arxiv.org/abs/2304.02643  
4. RT-1: https://robotics-transformer1.github.io/  
5. RT-2: https://arxiv.org/abs/2307.15818  
6. OpenVLA: https://arxiv.org/abs/2406.09246  
7. Open X-Embodiment: https://robotics-transformer-x.github.io/  
8. Radosavovic et al. Humanoid locomotion: https://arxiv.org/abs/2402.19469  
9. ExBody: https://arxiv.org/abs/2402.16796  
10. HumanoidGym: https://arxiv.org/abs/2404.05695  
11. Real-world humanoid locomotion (Science Robotics): https://www.science.org/doi/10.1126/scirobotics.adi9579  
12. HumanPlus: https://arxiv.org/abs/2406.10454  
13. OmniH2O: https://arxiv.org/abs/2406.08858  
14. H2O / Learning teleoperation: https://arxiv.org/abs/2403.04436  
15. H1 / Hover: https://arxiv.org/abs/2410.21229  
16. Hum2Rob (Bahl et al.): https://arxiv.org/abs/2207.02447  
17. Flow as interface: https://arxiv.org/abs/2407.15208  
18. General flow: https://arxiv.org/abs/2401.11439  
19. Structured world models from human videos: https://arxiv.org/abs/2304.02651  
20. MDM: https://arxiv.org/abs/2209.14915  
21. T2M-GPT: https://arxiv.org/abs/2211.00540  
22. HumanML3D: https://arxiv.org/abs/2205.01509  
23. AMASS: https://amass.is.tue.mpg.de/  
24. YOLOv8: https://arxiv.org/abs/2305.12748  
25. Video-LLaMA 2: https://arxiv.org/abs/2406.07476  
26. VIBE: https://arxiv.org/abs/1912.05656  
27. Adam: https://arxiv.org/abs/1412.6980  
28. PPO: https://arxiv.org/abs/1707.06347  
29. AMP / Adversarial motion priors: https://arxiv.org/abs/2104.02180  
30. VQ-VAE: https://arxiv.org/abs/1711.00937  
31. MAGVIT: https://arxiv.org/abs/2212.05199  
32. MotionDiffuse: https://arxiv.org/abs/2208.15001  
33. DART: https://arxiv.org/abs/2402.16226  
34. ImageNet: https://www.image-net.org/  
35. SoundStream: https://arxiv.org/abs/2107.03312  
36. EnCodec: https://arxiv.org/abs/2210.13438  
37. 4D Humans: https://arxiv.org/abs/2305.20091  
38. SLAHMR: https://arxiv.org/abs/2304.10991  
39. SigLIP: https://arxiv.org/abs/2303.15343  
40. π0: https://www.physicalintelligence.company/blog/pi0  
41. Helix: https://www.figure.ai/news/helix  
42. Genie 2: https://deepmind.google/technologies/genie/genie-2/  
43. V-JEPA 2: https://ai.meta.com/blog/v-jepa-2-world-model-browser-observation-sora/  
44. MVP: https://arxiv.org/abs/2303.04137  
45. R3M: https://arxiv.org/abs/2203.12601  
46. VIP: https://arxiv.org/abs/2210.03001  

---

总结一句直觉：**这篇 paper 把 humanoid control 从 narrow skill learning 拉到了 web-scale pre-training 范式**，pipeline 是 video → caption + SMPL → retarget → RL-aligned physical action → text-action pair → VQ-VAE tokenize → GPT-style autoregressive。每个 stage 都有 engineering 细节（shape $\beta$ 优化、ankle roll 保护、first-order similarity loss、k=4 temporal downsampling），整体效果是真机近乎 100% success rate。下一步 obvious extension 是加 vision encoder 做 loco-manipulation VLA，与 π0 / Helix / GR00T 在同一个赛道上。
