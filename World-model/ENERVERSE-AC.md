---
source_pdf: ENERVERSE-AC.pdf
paper_sha256: ae2c7fd06d0beb6ffd66d0f38107dcaa1db1d581db3a2f15f58ec07c83252002
processed_at: '2026-08-04T04:29:16-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 ENERVERSE-AC

Andrej，我把刚才那堆公式拆解翻译成大白话，但保留技术骨架，让你能 build intuition。

---

## 一句话版本

训练一个 video generation model，喂给它 "机械臂现在长啥样" 和 "机械臂接下来要干啥"，它就能 roll out "机械臂接下来会变成啥样" 的视频。于是你既不用真实机械臂也不用 MuJoCo，就能 test 你的 policy 到底行不行。

---

## 为什么这事难

robot imitation learning 的 policy 训完之后，你得知道它行不行。传统两条路：

**路 A**：部署到 real robot 上跑。准，但贵——一台机械臂几十万，每次 test 都占机器几个小时，还要人在旁边盯着。换一个 task 就得重新 setup scene。

**路 B**：在 MuJoCo / PyBullet / Isaac Gym [arxiv.org/abs/2108.10470](https://arxiv.org/abs/2108.10470) 里跑。便宜，但是你得先建 robot URDF、object mesh、scene layout。建一个 task 的 asset 可能要好几天，而且 sim-to-real gap [arxiv.org/abs/2102.10798](https://arxiv.org/abs/2102.10798) 让 policy 在 sim 里 100% 成功率到 real 上掉 30%。

**路 C (新)**：video generation model 当 world simulator。Sora [openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators) 证明 video generation 能学到某种 physics prior，但 Sora 是 text-to-video，你给它 "机器人抓瓶子" 它生成一个好看的视频，跟你的 policy action 没关系——你不能用 Sora 测你的 policy。

EVAC 要解决的就是这一 gap：**让 video generation 响应 action**。这样你给 policy 一个初始画面，policy 输出 action，EVAC 输出下一帧画面，再喂回 policy，闭环形成 simulator。

---

## 他们怎么做的——三个关键 idea

### Idea 1: 把 action 画成图，而不是塞成向量

机械臂 action 通常是 7 维向量：$[x, y, z, roll, pitch, yaw, openness]$。最 naive 的做法是把这个 7 维 vector 通过 cross-attention 注入 UNet。问题：diffusion model 对 "抽象数字 → 像素位置" 这种 grounding 学得很差，它擅长的是 image-to-image。

EVAC 的招：把 action **画**到 image 上。
- EEF 的 $(x,y,z)$ 通过 camera 内参外参 project 成 image 上的 $(u,v)$ pixel 位置
- roll/pitch/yaw 三个朝向轴用三个不同颜色的小箭头画在 EEF 位置上
- gripper openness 用一个小圆圈，亮 = open，暗 = closed
- 左右臂用不同颜色区分

这个 "action map" 跟 RGB image 一样是 2D 的，用 CLIP vision encoder [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020) 编码，跟 RGB feature map 在 channel 维度上拼一起，喂给 UNet。

直觉：把抽象信号转成 model 已经擅长处理的格式。diffusion model 的 backbone 在 billion-level image 上 pretrain 过，它知道 "图像里这个位置有个朝这个方向的箭头" 意味着什么，不需要从零学。

参考类似思路：CrayonRobo [arxiv.org/abs/2505.02166](https://arxiv.org/abs/2505.02166) 的 visual prompting、AnimateAnyone [arxiv.org/abs/2311.17117](https://arxiv.org/abs/2311.17117) 的 pose signal injection、MotionCtrl [arxiv.org/abs/2312.04405](https://arxiv.org/abs/2312.04405)。

### Idea 2: 光有位置不够，还要有速度

光画 EEF 当前位置有个坑：同一个最终 pose 可能对应完全不同的运动过程。比如 "把平底锅往上抛" vs "把平底锅往上慢摇"，两者的 trajectory 终点 pose 可能一模一样，但中间 dynamics 天差地别。如果不告诉 model 速度/加速度，它会混淆，导致生成视频里物体 (e.g., 锅里的火腿) flickering 或者突然消失——这是 paper Figure 9 ablation 展示的真实问题。

EVAC 加一个 **Delta Action Attention Module**：
- 计算 $\Delta a_t = a_t - a_{t-1}$，本质是 EEF velocity
- 通过 linear projector 压成固定长度 token
- 通过 cross-attention 注入 UNet

直觉：一阶导数捕捉速度，让 model 能区分 "快动 vs 慢动"。paper 里提到还能隐式 benefit 到二阶 (加速度)，因为快 tossing 是高加速度脉冲，慢 shaking 是低加速度振荡，速度序列本身的 pattern 已经带了这个信息。

公式上就是 cross-attention：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $Q$ 来自 UNet 当前 layer 的 feature
- $K, V$ 来自 delta action token + reference image token
- $d_k$ 是 key 维度，除以 $\sqrt{d_k}$ 是 scaled dot-product attention 的标准操作

这里有两个 level 的 condition 互补：**spatial action map** 告诉 model "gripper 现在在哪、朝哪、开没开"，**delta action** 告诉 model "gripper 现在动多快、加速还是减速"。

### Idea 3: 多视角 + 动态相机用 ray map 编码

robot manipulation 通常要 head camera (固定视角看全局) + wrist camera (装在机械臂上跟手走) 两个视角。wrist camera 看手指细节，对抓取很关键。

问题：wrist camera 跟 arm 一起动，EEF 相对 wrist camera 的位置几乎不变——所以你把 EEF pose 投影到 wrist image 上，每帧投影位置都一样 (Figure 3 下排)。spatial action map 在 wrist view 上完全失效，model 不知道 arm 在动。

解决：**ray map**。对每个相机的每个 pixel，算它对应的 ray：
- $o_r \in \mathbb{R}^{h \times w \times 3}$：ray origin，也就是 camera center 在 world coordinate 的位置
- $d_r \in \mathbb{R}^{h \times w \times 3}$：ray direction，每个 pixel 看出去的方向

对固定 head camera，ray map 每帧一样；对动态 wrist camera，ray map 每帧都在变，**变的就是 camera 的 SE(3) pose 变化**，也就是 arm 的运动。

ray map 跟 trajectory map 一起 concatenate 进 UNet input。这样 wrist view 通过 ray map 隐式知道 "相机现在在动"，相当于把 camera motion 编进去了。

ray map 思路来自 NeRF [arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934) 那套相机参数化，CAT3D [arxiv.org/abs/2405.10314](https://arxiv.org/abs/2405.10314) 也用了类似 idea。

多视角之间通过 **spatial cross-attention** 交互，让 head view 知道 wrist view 在看啥，反过来也是。这让生成的多视角视频在空间上一致，不会 head view 显示抓到了 wrist view 显示没抓到。

---

## 怎么训练的

backbone 用 DynaCrafter [arxiv.org/abs/2310.12190](https://arxiv.org/abs/2310.12190) 这一类 UNet-based latent diffusion model。流程：

1. RGB video $O$ 过 frozen VAE encoder $\varepsilon$ 得到 latent $z$
2. UNet 在 latent space 上 denoising，condition signal 是 action map + delta action + ray map + reference frame
3. 用 v-prediction loss 训练

v-prediction 公式：

$$v = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1-\bar{\alpha}_t} z_0$$

- $z_0$: clean latent
- $\epsilon$: 加的 noise
- $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$: noise schedule cumulative product
- $t$: timestep

预测 target 是 $v$，不是 $\epsilon$。v-prediction 在 high-noise regime 下比 $\epsilon$-prediction 梯度更稳，对 long-horizon autoregressive generation 重要——因为 chunk-by-chunk 推下去，error 累积，需要每步都稳。

数据来源：AgiBot-World [arxiv.org/abs/2503.06669](https://arxiv.org/abs/2503.06669)，210 个 task 100 万条 trajectory。他们特意挖了 **failure case**——这点直觉上反常识但很关键：success-only 数据会让 model 学到 "任何 action → success" 的 spurious correlation，看到失败 action 也生成 success 视频。加入 failure data 让 model 学到真实 "action → outcome" mapping。

训练成本：单视角 32 张 A100 跑 2 天，多视角 32 张 A100 跑 8 天。这个 compute 在 robot learning 圈算中等。

---

## 两个应用

### Data Engine (增广训练数据)

人工采 20 条 demo 贵且不够 diverse。EVAC 可以这么扩：
1. 把 trajectory 按 gripper openness 切成 fetching / grasping / homing 三段
2. 取 fetching 段，固定终态 $a_{t_b}$，给初态 $a_{t_b-N}$ 加 spatial perturbation 得到 $a'_{t_b-N}$
3. 线性插值生成中间 action
4. **小聪明**：reverse action 序列，用真实终态画面 $O_{t_b}$ 当 condition，**逆向** generate fetching 段画面。因为终态画面信息密度高，从已知终态反推初态比正推稳。
5. 再 reverse 回来得到正向 trajectory

实验数据 (Table 1)：
- Baseline: 20 条 expert demo 训 GO-1，success rate = **0.28**
- +30% 合成数据: success rate = **0.36** (相对提升 28.6%)

绝对数字不算高，但相对提升明显，说明 generated data 确实 carry 了有用 information。

### Policy Evaluator (替代 real robot 测 policy)

闭环：
1. 给 initial frame $O_t$ 和 instruction
2. Policy (GO-1) 输出 action chunk
3. EVAC 用 $(O_t, \text{action chunk})$ 生成 K 帧 future video
4. 取最后帧当新 $O_t$，回到 step 2
5. Stop 条件：policy 输出的 action magnitude 低于阈值
6. Human 或 Video-MLLM [arxiv.org/abs/2306.05225](https://arxiv.org/abs/2306.05225) 看 video 判 success/fail

Figure 7 左：4 个 task 上 EVAC 评估和 real-world 评估的绝对 success rate 有小差异，但**相对 trend 完全一致**——哪个 task 好、哪个 task 差，EVAC 看得出。这意味着 EVAC 可以用来做 policy selection (在几个 checkpoint 里选最好的) 和 training monitoring (训到第几步开始退化)。

Figure 7 右：同一个 policy 在不同 training step 上的表现，EVAC 和 real-world 都显示随 step 增加 success rate 上升。这点很重要——说明 EVAC 不仅区分 task 难易，还能区分 policy 微小能力差异。

直觉：world model 当 simulator，关键不在于绝对复刻 real world，而在于 **preserve relative ordering**。如果 task A 比 task B 难，EVAC 也要说 A 比 B 难；如果 policy $\pi_1$ 比 $\pi_2$ 好，EVAC 也要说 $\pi_1$ 比 $\pi_2$ 好。绝对值可以偏，相对序不能错。

---

## 长程 generation 怎么不漂

chunk-wise autoregressive 的通病是 error 累积，跑到后面画面糊掉或者物体变形。EVAC 用 **sparse memory mechanism**：保留 4 个 historical frame 的 latent，每个来自上一个 chunk 的生成结果。新 chunk 生成时，这 4 个 memory latent 跟当前 observation latent 一起 concatenate 喂给 UNet。

效果：单视角能稳到 **30 chunks**，多视角稳到 **10 chunks**。多视角撑不住更久是因为 wrist camera background 里有人走动之类的 dynamic noise，generation 复杂度高。

memory size = 4 是实验 balance 出来的，太小信息不够，太大显存爆且容易过拟合到 history。

---

## 这事的根本意义

paper 表面上是 video generation model，实际推的是一件事：**robot learning 的 evaluation paradigm 从 physical 转向 generative**。

传统路径：real robot → simulator → real robot，sim 在中间当便宜但不准的 proxy。
EVAC 路径：real robot (少量采数据) → generative world model (大量 rollout) → real robot (最终验证)。

generative world model 的优势：
1. 不用建 asset——它从真实 video 学，scene/object 都隐式在 weights 里
2. 能 simulate 失败——只要训练数据里有 failure case
3. 跟 real world 的 gap 可能比传统 sim 小，因为它直接从 real video 学 dynamics 而不是 hand-craft physics

劣势：
1. 长程会漂，10/30 chunks 不一定够长 task 用
2. action space 限制——unit circle gripper encoding 不通用，dexterous hand 要重做
3. 多视角 dynamic background 干扰没解决
4. 没接 RL——现在只能 evaluate，不能 optimize policy

---

## 我觉得最 promising 的下一步

### A. 接 Dreamer-style model-based RL

EVAC 当 learned dynamics model，policy 在 EVAC 里 rollout 做 actor-critic update。这跟 Dreamer [arxiv.org/abs/1910.01341](https://arxiv.org/abs/1910.01341) 的 RSSM dream 思路一样，只是把 RSSM 换成 high-fidelity video diffusion。

挑战：diffusion model 一步生成太慢 (几百 step denoising)，RL rollout 要几万步，直接接不动。需要 consistency model [arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469) 或 adversarial diffusion distillation [arxiv.org/abs/2312.08411](https://arxiv.org/abs/2312.08411) 之类 few-step distillation 把推理成本降下来。

### B. 把 failure data 推到 extreme

paper 现在只加了 "假抓" 这种 failure。但 real world failure mode 多得多：object slip、collide、tip over、gripper jam、sensor noise。如果能把 failure type taxonomy 化，主动采各类型 failure，world model 能 simulate 的 boundary 会大幅扩。

这跟 Adversarial Data Collection [arxiv.org/abs/2503.11646](https://arxiv.org/abs/2503.11646) 思路一致——人类故意制造 perturbation 采 hard case。

### C. Wrist camera background 用 matting 处理

multi-view 只能撑 10 chunks 主因是 wrist view background 动。可以预处理：用 SAM-2 [arxiv.org/abs/2408.00714](https://arxiv.org/abs/2408.00714) 分出 robot arm + object mask，background 用 static frame inpaint 替换掉。这样 generation model 只需要 focus on foreground dynamics。

### D. Latent action 替代 raw action

GO-1 自己用 latent action representation [arxiv.org/abs/2410.08001](https://arxiv.org/abs/2410.08001) 提升 generalization。EVAC 也该试：condition 在 latent action space 而不是 raw 7D pose，可能让 condition 更 compact、更易学。

### E. Self-consistency loss

现在 model 各 chunk 独立生成，靠 memory 弱耦合。可加 explicit self-consistency loss：让 chunk N 结尾帧和 chunk N+1 开头帧 latent 距离小，强制 continuity。这是 video super-resolution 和 long video generation [arxiv.org/abs/2304.08849](https://arxiv.org/abs/2304.08849) 里的常见 trick。

### F. 跟 3D Gaussian Splatting 结合

ray map 已经在 encode camera geometry，可以更进一步让 model 输出 3DGS [arxiv.org/abs/2008.04031](https://arxiv.org/abs/2008.04031) 而不是 image。这样 multi-view consistency 是 explicit 的 (3DGS 本身就是 multi-view 一致)，且能实时 render 任意新视角。这跟 EVAC 当 evaluator 的目标完全 align。

---

## 最直白的类比

把 EVAC 想成 "机械臂版的 Flight Simulator"。

飞行模拟器不需要真飞机，因为空气动力学被很好地建模，你能任意试动作看结果。机械臂一直没有这样的模拟器——传统 sim 像 X-Plane，物理准但场景要你建，且 sim-to-real gap 大；EVAC 像 Microsoft Flight Simulator 新版，画面是从真实卫星图和 AI 生成学的，玩起来更接近真飞行体验，虽然底层 physics 可能不严格。

对 robot policy 来说，"玩起来像真的" 比 "physics 严格" 更重要——因为 policy 是从真实数据学的，它在真实分布上 work，所以 simulator 只要能 reproduce 真实分布就行，EVAC 直接从真实 video 学分布，闭环。

---

## 卡帕西你会怎么 hack 这事

如果是我，会做三个 quick experiment：

1. **Probe action grounding**：把 action map 换成 random noise，看生成质量掉多少。如果掉很多说明 action map 真的 carry 信号；如果没掉多少说明 delta attention 在干主要活，action map 是 window dressing。
2. **Probe failure generalization**：训两个 model，一个只加 "假抓" failure，一个加 "假抓 + slip + collide" 多种 failure，看 evaluator 能区分多少种 fail mode。这能测 world model 是真学到 failure dynamics 还是只学到 "sometimes 显示 robot 手空着"。
3. **Probe long-horizon**：测 50 chunk、100 chunk 看 quality 退化曲线。如果指数退化说明 memory 机制治标不治本，需要 fundamentally 更强的 temporal conditioning；如果线性退化，加 memory size 可能就够。

---

## 总结

EVAC 把 video generation model 从 "好看的 demo" 推到 "可用的 robot evaluation infra"。三个核心 idea——action visual prompting、delta motion attention、dynamic camera ray map——都是把抽象信号 convert 到 model 已擅长处理的格式。failure data 加入让 model 学到 "action → outcome" 而不是 "action → success" 的真实 mapping。

它现在更像 "便宜的 relative evaluator"，还不到 "精确 absolute simulator"。但作为 robot learning infra 的方向，这条路大概率会越走越宽——因为 asset-free + gap-small 这两个 property 是传统 sim 永远给不了的。

paper project page: [annaj2178.github.io/EnerverseAC.github.io](https://annaj2178.github.io/EnerverseAC.github.io/)

---

# ENERVERSE-AC 深度解析

Andrej，这篇paper 来自 AgiBot 团队，是对他们之前 EnerVerse [arxiv.org/abs/2501.01895](https://arxiv.org/abs/2501.01895) 工作的重要扩展。核心 idea 是构建一个 **action-conditional world model**，同时充当 robotic imitation learning 的 **data engine** 和 **policy evaluator**，绕开传统 physical simulator (MuJoCo/PyBullet/Isaac Gym) 的 sim-to-real gap 和 asset 构建成本问题。

---

## 1. Problem Formulation: 为什么需要 action-conditional

传统 robotic imitation learning 评估面临一个根本矛盾：policy 在 dynamic environment 中需要 real-time interaction，但部署到 physical robot 贵且不可规模化，构建 simulator 又需要 expensive digital assets (robot URDF, object meshes, scene setup)。

之前的 video-generation-as-world-simulator 工作 (Sora [openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators), Genie [arxiv.org/abs/2404.14522](https://arxiv.org/abs/2404.14522), RoboDreamer [arxiv.org/abs/2404.12377](https://arxiv.org/abs/2404.12377)) 主要做 **language/observation → video**，缺一个关键 piece：**world simulator 必须响应 agent 的 action**，才能真正用于 policy testing。EVAC 填补这一 gap——给定 initial observation 和 predicted action sequence，直接 roll out 未来视觉 observation。

---

## 2. Architecture 深度拆解

### 2.1 Latent Diffusion Backbone

形式化地，给定 RGB video set：

$$O \in \mathbb{R}^{V \times (H+K) \times 3 \times h \times w}$$

- $V$: views 数量 (head + wrist + ...)
- $H$: history frames 数量
- $K$: 要预测的 future frames 数量
- $h, w$: 单帧分辨率

通过 VAE encoder $\varepsilon$ 编码到 latent space：

$$z \in \mathbb{R}^{V \times H \times C \times h' \times w'}$$

其中 $C$ 是 latent channel (这里 $C=4$)，spatial 维度被压缩 (如 $320 \times 512 \to 40 \times 64$)。

Diffusion 过程：

$$z_t = p_\theta(z_{t-1}, c, t)$$

- $z_t$: denoising step $t$ 时的 latent
- $z_{t-1}$: 上一步 latent
- $c$: condition signal (这里来自 action trajectory)
- $t$: denoising timestep (从 $T=1000$ 走到 0)
- $\theta$: UNet 参数

注意这里用 **v-prediction** 而不是 $\epsilon$-prediction，对 high-noise regime 更稳定，这对 long-horizon chunk-wise autoregressive generation 很重要。

### 2.2 Multi-Level Action Condition Injection

这是 EVAC 的核心创新。单一 action encoding 不够，他们用 **two-level** 注入，分别捕捉 spatial 和 temporal 信息。

#### Level 1: Spatial-Aware Pose Injection

Robotic action $A \in \mathbb{R}^{(H+K) \times d}$，单臂 $d=7$：$[x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{openness}]$，双臂 $d=14$。

直接把 7D vector 喂给 cross-attention 太抽象，model 难以建立 "action → pixel" 的 grounding。他们的方案是把 action **visual化** 成 action map：

1. **位置投影**：用 calibrated camera intrinsics/extrinsics，把 EEF 在 world coordinate 的 $(x,y,z)$ 投影到 image pixel coordinate $(u,v)$
2. **朝向编码**：roll/pitch/yaw 三个 axis 用三个 unit vector $(\hat{u}_x, \hat{u}_y, \hat{u}_z)$ 在 image 平面上可视化，每个 axis 一种颜色 (类似 CrayonRobo [arxiv.org/abs/2505.02166](https://arxiv.org/abs/2505.02166) 的 visual prompting 思路)
3. **Gripper openness**：用 unit circle 编码，亮度 = 开合程度 (亮 = open, 暗 = closed)
4. **双臂区分**：左右臂用不同 color scheme

这个 action map 用 **CLIP vision encoder** 编码 (frozen)，得到 feature map，与 RGB image 的 feature map **沿 channel dimension concatenate**，作为 UNet input。

直觉：把 abstract action 变成 visual grounded signal，让 diffusion model 利用其 image-prior 知识 "看到" gripper 应该在哪、朝哪、开合状态如何。

#### Level 2: Delta Action Attention Module

只有 spatial action map 缺少 **motion dynamics**——同一 pose 可以对应静止、慢动、快动、急加速等完全不同视频。Delta module 显式编码 consecutive frames 间的 action 差：

$$\Delta a_t = a_t - a_{t-1}$$

这本质上捕捉了 velocity (一阶导数)。从 ablation (Figure 9) 看，他们还隐式 benefit 于 acceleration (二阶信息) 的可分性。

Delta sequence 通过 **linear projector** 压缩成 fixed-length latent tokens，与 reference image features 一起通过 **cross-attention** 注入 UNet：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q$ 来自 UNet feature，$K, V$ 来自 delta action tokens + reference image tokens。

Ablation 关键发现：**tossing vs shaking** 区分。两者 final pose 可能一样，但 tossing 是高加速度脉冲，shaking 是低加速度振荡。没有 Delta Module，model 会混淆，导致 object (e.g., ham) flickering 或突然消失。

### 2.3 Multi-View Extension 与 Ray Map

Robotic manipulation 强依赖 multi-view，特别是 **wrist camera** (跟随 arm 移动)。EVAC 在 EnerVerse 基础上加入 dynamic wrist view 支持。

挑战：wrist camera 随 arm 移动，EEF 相对 wrist camera 几乎静止，所以 "EEF 投影到 wrist image" 几乎是 constant (Figure 3 bottom row)。spatial action map 在 wrist view 上失效。

解决：**ray map encoding**。对每个 camera 每个 timestep，计算：

$$r = (o_r, d_r)$$

- $o_r \in \mathbb{R}^{h \times w \times 3}$: 每个 pixel 对应的 ray origin (camera position in world)
- $d_r \in \mathbb{R}^{h \times w \times 3}$: 每个 pixel 对应的 ray direction (camera orientation in world)

对 static head camera，ray map 基本恒定；对 dynamic wrist camera，ray map 随 arm 移动而变化，**隐式编码 EEF motion**。将 ray map (可视化成 RGB) 与 trajectory map concatenate 进 UNet input。

Multi-view 之间通过 **spatial cross-attention** 交互：每个 view 的 feature 同时 attend 到其他 view 的对应 spatial location，建立 cross-view consistency。

### 2.4 Memory Mechanism

为了支持 long-horizon generation，他们用 sparse memory：保留 4 个 historical frames 的 latent，每个来自 previous chunk 的生成结果。新 chunk 生成时，这 4 个 memory latent 与当前 observation latent concatenate 输入 UNet。这避免了 autoregressive drift，让 single-view 稳定到 30 chunks，multi-view 稳定到 10 chunks。

### 2.5 训练目标

Standard latent diffusion loss：

$$\mathcal{L} = \mathbb{E}_{t, z_0, \epsilon} \left[ \| v - v_\theta(z_t, c, t) \|^2 \right]$$

其中 $v = \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1-\bar{\alpha}_t} z_0$ 是 v-prediction target，$\bar{\alpha}_t$ 是 noise schedule cumulative product。

UNet input channels = 19 (从 Table 2): 4 (latent image) + 4 (condition latent image) + 4 (action map) + 6 (ray map) + 1 (dropout mask) = 19。

---

## 3. 训练数据策略

一个 underappreciated 的细节：**failure data matters**。他们专门从 AgiBot-World dataset [arxiv.org/abs/2503.06669](https://arxiv.org/abs/2503.06669) (210 tasks, 1M trajectories) 挖掘 failure trajectories，并开发自动化 pipeline 收集 teleoperation / real-robot inference 的 failure cases。

直觉：success-only data 让 model 学到 "action → success" 的 spurious correlation，遇到 action 不对应可执行任务的 case 会 hallucinate success (Figure 8)。加入 failure data 让 model 学到 "action → outcome" 的真实 mapping，包括 "action → fail"。

这是 world model 作为 simulator 的 critical property——必须能 simulate failure，否则 evaluator 永远说 success。

---

## 4. 双重应用

### 4.1 Data Engine

对 pick-and-place 类 primitive task：
1. 人工采 $M$ 条 trajectories
2. 通过 gripper openness 变化检测 $t_b$ (开始接触) 和 $t_e$ (结束接触)
3. 分割成 fetching / grasping / homing 三 phase
4. 对 fetching phase，固定 $a_{t_b}$，对 $a_{t_b-N}$ 做 spatial augmentation 得到 $a'_{t_b-N}$
5. Linear interpolation 生成中间 actions
6. **关键 trick**：reverse action sequence，用真实 $O_{t_b}$ 作为 condition，**逆向** generate fetching phase 的 frames (因为 fetching 是从随机位置到 $t_b$，倒过来是从已知终态推初态，更稳定)
7. Re-order 得到正向 trajectory

Table 1 结果：20 expert demos + 30% synthetic → SR 从 0.28 → 0.36 (+28.6% relative)。

### 4.2 Policy Evaluator

闭环 evaluation：
1. 给定 $O_t$ 和 instruction
2. Policy (GO-1) 生成 action chunk
3. EVAC 用 $(O_t, \text{action chunk})$ 生成 $O_{t+1}, ..., O_{t+K}$
4. 取最后 frame 作为新 $O_t$，loop
5. 当 generated action 低于阈值停止
6. Human 或 Video-MLLM 评估视频判断 success

Figure 7 结果：EVAC 评估与 real-world 评估的 **相对 trend 一致** (绝对值有小偏移)，且能正确反映 training step 增加时 success rate 上升的 pattern。这说明 EVAC 可替代 real robot 做 policy selection 和 training monitoring。

---

## 5. Limitations 与未来方向

1. **Gripper encoding 不通用**：unit circle + color 只适合 parallel-jaw gripper，dexterous hand 需要新 encoding
2. **Wrist camera 背景 noise**：人走动等 dynamic background 增加 generation 难度，导致 multi-view 只能撑 10 chunks (single-view 30 chunks)
3. **未探索 RL 集成**：action-conditional world model 天然适合做 model-based RL 的 dynamics model，可结合 Dreamer [arxiv.org/abs/1910.01341](https://arxiv.org/abs/1910.01341), TD-MPC 等 actor-critic framework，这是 future work 大方向

---

## 6. 关键 Intuition 总结

1. **Visual grounding > vector conditioning**：把 action visual 化成 image-aligned map 比直接 cross-attend vector 更 work，因为 leverage 了 diffusion model 的 image prior
2. **Multi-level condition 捕捉不同物理量**：spatial map 编码位置/朝向，delta module 编码速度/加速度，二者互补
3. **Ray map 是动态相机的 cheap encoding**：不需要显式 SE(3) transform，让 model 从 data 学到 camera motion 与 scene motion 的 correlation
4. **Failure data 是 simulator 的必要条件**：能 simulate failure 才能 evaluate failure
5. **Reverse generation 是 data augmentation 的 stable trick**：从已知终态反向推初态比正向推终态更稳定，因为终态信息密度高

---

## References

- EnerVerse: [arxiv.org/abs/2501.01895](https://arxiv.org/abs/2501.01895)
- AgiBot-World: [arxiv.org/abs/2503.06669](https://arxiv.org/abs/2503.06669)
- Sora: [openai.com/research/video-generation-models-as-world-simulators](https://openai.com/research/video-generation-models-as-world-simulators)
- Genie: [arxiv.org/abs/2404.14522](https://arxiv.org/abs/2404.14522)
- RoboDreamer: [arxiv.org/abs/2404.12377](https://arxiv.org/abs/2404.12377)
- DynaCrafter (backbone): [arxiv.org/abs/2310.12190](https://arxiv.org/abs/2310.12190)
- CrayonRobo (visual prompting): [arxiv.org/abs/2505.02166](https://arxiv.org/abs/2505.02166)
- CAT3D (ray maps inspiration): [arxiv.org/abs/2405.10314](https://arxiv.org/abs/2405.10314)
- Dreamer (world model RL): [arxiv.org/abs/1910.01341](https://arxiv.org/abs/1910.01341)
- Project page: [annaj2178.github.io/EnerverseAC.github.io](https://annaj2178.github.io/EnerverseAC.github.io/)

如果你想进一步 push 这个方向，最 promising 的 angle 似乎是把 EVAC 接入 actor-critic RL loop——用 EVAC 当 learned dynamics model，policy 在 EVAC 里 rollout 训练，类似 Dreamer 但用 high-fidelity video generation 替代 RSSM。另一个 angle 是解决 wrist camera background noise，可能用 background matting 或在 ray map 里 explicit encode dynamic background mask。
