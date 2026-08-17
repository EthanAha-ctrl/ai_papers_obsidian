---
source_pdf: Hunyuan-GameCraft High-dynamic Interactive Game Video Generation with
  Hybrid History Condition.pdf
paper_sha256: a14888c99c588cfd232913e4d74055eb868aa9eef0d45347808c3367d46a56dc
processed_at: '2026-08-05T08:20:45-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Hunyuan-GameCraft

Andrej，咱换个频道，不堆术语，就当在咖啡馆聊。

---

## 这 paper 在干啥

你给它一张图，比如《Cyberpunk 2077》里夜之城街头的一帧截图。然后你按键盘 W，它就给你生成角色往前走的视频；按 A，往左走；按住鼠标右键同时动鼠标，视角转。

就这么简单一个事儿——**把键盘鼠标变成视频**。

听起来像 game engine，对吧？区别在于 game engine 是代码渲染，这个是 diffusion model 生成。你给它 action，它"想象"出下一秒画面。

项目主页: https://hunyuan-gamecraft.github.io/

---

## 为啥这事儿难

你想想，这事儿有三个天然矛盾：

**矛盾一：离散按键 vs 连续画面**

W 键就两种状态，按 or 不按。但游戏里的运动是连续的——你按 W 三秒和按 W 半秒，走的距离不一样。如果模型只学"W = 前进"，它怎么知道走多远？

之前的 GameNGen (https://arxiv.org/abs/2408.14837) 在 DOOM 上就吃这亏，Oasis (https://www.decart.ai/articles/oasis-interactive-ai-video-game-model) 在 Minecraft 上也是。离散 key 喂进去，生成出来的动作要么死板要么乱套。

**矛盾二：跟随历史 vs 响应新动作**

这是最本质的矛盾。

你按 W 走了 5 秒，第 6 秒按 S 想后退。模型现在很纠结：
- 如果它"记性太好"，会想"刚才一直 W，那继续 W 吧"——S 信号被忽略
- 如果它"记性太差"，S 是响应了，但走 5 秒后场景早就漂了、墙穿了、NPC 消失了

短记忆响应灵敏但会崩，长记忆稳定但会卡。之前所有工作要么死板要么塌方，没人同时解决。

**矛盾三：质量 vs 速度**

Diffusion model 生成慢得要命。Matrix-Game (https://arxiv.org/abs/2501.08325) 跑 0.06 FPS，按一个键要等 16 秒。这能玩个屁。

要实时交互至少得 5-10 FPS。但加速通常掉质量——dynamic（动态感）和 fidelity（画面质量）一起掉。

---

## 他们怎么解的

三个 trick，对应三个矛盾。

### Trick 1: 把按键翻译成 camera 参数

公式 (1):

$$
\mathcal{A} := \left\{ \mathbf{a} = (\mathbf{d}_{\text{trans}}, \mathbf{d}_{\text{rot}}, \alpha, \beta) \right\}
$$

- $\mathbf{d}_{\text{trans}} \in \mathbb{S}^2$：平移方向，单位向量，在球面上（所以模长恒为 1，不用学）
- $\mathbf{d}_{\text{rot}} \in \mathbb{S}^2$：旋转方向，同上
- $\alpha \in [0, v_{\max}]$：平移速度，标量，0 到最大速度之间
- $\beta \in [0, \omega_{\max}]$：旋转速度，标量

人话翻译：**每个按键组合对应一个"方向 + 速度"的 6 维向量**。

- W = $(0, 0, 1, \alpha_{\text{walk}}, 0, 0)$（朝前走，速度 $\alpha_{\text{walk}}$）
- W + Shift = $(0, 0, 1, \alpha_{\text{run}}, 0, 0)$（同方向，更快）
- A = $(-1, 0, 0, \alpha_{\text{walk}}, 0, 0)$（朝左）

为啥聪明？因为现在 action 是**连续可插值**的了。W 和 W+Shift 之间可以 lerp 出"中等速度走"，模型不需要学"走路 vs 跑步"两件事，只需要学"方向"和"速度"两件事。

为啥是 $\mathbb{S}^2$（球面）？因为方向归一化。如果用普通 3D 向量 $(x, y, z)$，模型得自己学会 $x^2+y^2+z^2=1$，浪费容量。直接约束在球面上，模型只学球面坐标 $(\theta, \phi)$。

为啥去掉 roll？游戏里玩家几乎不 roll（除飞行模拟器），砍掉这个维度让训练数据更聚焦。

然后这个 6 维 camera 参数转成 Plücker embedding（每个像素一条 ray，6 维表示），喂给一个极轻量的 conv encoder，**直接加**到 video latent 的 patch token 上。完事。

参考 CameraCtrl（最早用 Plücker）: https://arxiv.org/abs/2404.02101

### Trick 2: Hybrid History Conditioning

这是 paper 最有意思的地方。

每生成一段 chunk（33 帧 @ 25fps = 1.32 秒），下一段生成时给模型看什么 history？三个选项：

- **(I) 单帧**：只看上一段最后一帧。响应灵敏，几步就崩
- **(II) 整段 clip latent**：看上一段整个 latent。稳，但模型惰性强，对新动作反应慢
- **(III) 多段 clip 拼接**：看前几段 latent。最稳，响应最差

paper 在 Fig 6 做了对比，结果完全符合直觉——单帧 RPE trans 0.07（响应准）但几步就漂，clip RPE trans 0.16（响应差）但画面稳。

他们的解法：**训练时三种随机混着用**。

$$
P(\text{single clip}) = 0.70,\ P(\text{multi clip}) = 0.05,\ P(\text{single frame}) = 0.25
$$

70% 时间给单段 clip（保稳定 prior），25% 给单帧（教响应新动作），5% 给多段（偶尔的长 context 训练）。

具体实现：head latent（mask=1，clean）和 chunk latent（mask=0，noisy）concat 在一起喂给 MM-DiT。head 部分在 flow matching 过程中保持 noise-free，像 anchor 一样引导 chunk 去噪。

$$
\frac{dz_t}{dt} = u_\theta(z_t, t, \text{condition})
$$

这里 $z_t$ 在 head 区域被钉死在 $t=0$，在 chunk 区域从 $t=1$ 走到 $t=0$。

人话翻译：**让模型同时学两种模式——"看历史延续"和"听命令动作"——通过训练时随机切换强迫它两个都会**。

这其实挺 elegant 的。之前的工作都是选一边站，要么 free generation 要么 long context，他们用 mixture 把俩能力塞进一个模型。

### Trick 3: PCM + CFG 蒸馏到 6.6 FPS

原始 diffusion 50 步，CFG 还要跑两次 forward（cond + uncond + 线性组合），慢到 0.25 FPS。

他们用 Phased Consistency Model (https://arxiv.org/abs/2410.07184) 把 diffusion 蒸馏到 8 步，再用 CFG distillation 把"两次 forward"压成"一次 forward"。

CFG distillation 公式 (2):

$$
L_{\text{cfg}} = \mathbb{E}_{w \sim p_w, t \sim U[0,1]} \left[ \| \hat{u}_\theta(z_t, t, w, T_s) - u_\theta^s(z_t, t, w, T_s) \|_2^2 \right]
$$

$$
\hat{u}_\theta(z_t, t, w, T_s) = (1+w)\, u_\theta(z_t, t, T_s) - w\, u_\theta(z_t, t, \emptyset)
$$

- $z_t$：noisy latent
- $t \in [0,1]$：noise level
- $w$：CFG guidance scale（训练时随机采，让 student 在各种 $w$ 下都能 match）
- $T_s$：prompt
- $\emptyset$：空 prompt
- $u_\theta$：teacher
- $u_\theta^s$：student

直觉：teacher 跑两次 forward 然后线性组合得到 guided 输出，student 一次 forward 直接吐出 guided 输出。Student 学的就是"teacher 在 guidance scale $w$ 下的最终结果"。

效果：

| | Ours | Ours+PCM |
|---|---|---|
| FPS | 0.25 | **6.6** |
| RPE trans | 0.08 | 0.08（不掉！） |
| Dynamic Average | 67.2 | 43.8（掉 35%） |
| FVD | 1554 | 1883（+21%） |

**控制精度完全保留**，因为 action signal 是显式 condition，蒸馏不掉。但 dynamic 掉 35%，因为 consistency model 倾向于保守生成（步数少不敢做大动作）。

人话翻译：**蒸馏之后响应还是准的（你按 W 真往前走），但画面没那么"活"了**。可玩性方面，6.6 FPS 接近可玩，虽然离真游戏 60 FPS 还差很远。

---

## 数据工程是隐藏 boss

这个 paper 技术创新看着不复杂，但数据工程是真功夫。

100+ AAA 游戏（GTA、RDR2、Cyberpunk 等），每个 2-3 小时录屏，1080p。然后：

1. **PySceneDetect** (https://github.com/Breakthrough/PySceneDetect) 切场景，得 6 秒连贯 clip，1M+
2. **RAFT** (https://arxiv.org/abs/2003.12039) 算光流找动作边界
3. **Kolors quality assessment** 过滤低质量
4. **OpenCV** 过滤暗场景（游戏过场动画太多暗的）
5. **Qwen2-VL** (https://arxiv.org/abs/2409.12191) 检测视觉异常
6. **Monst3R** (https://arxiv.org/abs/2410.03825) 从单目视频重建 6-DoF camera trajectory——这步是把"视频"变成"带 action label 训练数据"的关键
7. **Qwen2-VL** 生成结构化 caption（30 字符简版 + 100 字符详版，训练随机采样）

还有两个 trick 处理 distribution bias：

- 游戏视频 90% 时间在前进，模型会学成"啥都往前走"。他们做 **stratified sampling** 把位移方向分桶等概率采，再用 **temporal inversion**（视频倒放）让后退数据翻倍
- 真实视频的 camera trajectory 有重建误差，所以他们**自己渲染了 3000 个合成 motion sequence** 提供精确几何 prior。但合成数据缺动态物体，所以 Render:Live = 1:5 混合训练

Ablation 数据：
- 只用合成：DA 34.6（动态差），RPE trans 0.07（控制准）
- 只用真实：DA 77.2（动态好），RPE trans 0.16（控制差）
- 1:5 混合：DA 67.2, RPE trans 0.08（兼得）

---

## 整体效果

Table 2 主要数据：

| Model | FVD↓ | DA↑ | RPE trans↓ | FPS |
|---|---|---|---|---|
| Matrix-Game | 2260 | 31.7 | 0.18 | 0.06 |
| **Ours** | **1554** | **67.2** | **0.08** | 0.25 |
| Ours+PCM | 1883 | 43.8 | 0.08 | **6.6** |

比 Matrix-Game：
- FVD 好 31%
- Dynamic 大 2 倍
- 控制误差降 55%
- FPS 快 100 倍（PCM 版）

而且 GameCraft 能在真实世界图像上 work（Section 6, Fig 10）——这是 HunyuanVideo foundation model 的功劳，预训练 prior 没被 game 数据 erase。

---

## 我读完的几个直觉判断

**判断 1: Hybrid mixture 0.70/0.05/0.25 没理论依据**

paper 没扫这个 ratio，纯经验值。这是个低垂果实——grid search 一下肯定能找到更好的点。但这也是这类工作的通病，工程调参大于理论指导。

**判断 2: Memory 只是 short-term**

他们的 "memory" 就是上一段 clip 的 latent，大概 1.3 秒。玩家记得"5 分钟前我在那个房子里看到一把枪"——这种 long-term episodic memory 完全没有。

要真做出可玩世界，得有 external state tracker 或 retrieval-augmented generation。这接近 LeCun H-JEPA 的思路，maintain world model state 而不是 raw latent。

**判断 3: Action space 拓展是关键 future work**

camera-only 解决了 navigation。但游戏 90% 乐趣是"对世界做事"——开枪、跳跃、扔手雷。这些不是 camera 参数，是离散 event + 物理效应。

可能的扩展是在 action vector 上加 event token：$\mathbf{a} = (\mathbf{d}_{\text{trans}}, \mathbf{d}_{\text{rot}}, \alpha, \beta, \mathbf{e})$，$\mathbf{e}$ 是 multi-hot event。但训练数据得重新标注 event 类型，工作量巨大。

**判断 4: 和 GameNGen 对比的 hidden 信息**

GameNGen 在 DOOM 上做到 50 FPS，GameCraft 才 6.6。差距来源：
- DOOM 240p vs GameCraft 720p（9 倍像素）
- DOOM 单游戏可以过拟合 vs GameCraft 跨 100+ 游戏
- GameNGen 用 2-step diffusion + noise conditioning，GameCraft 用 8-step PCM

所以 GameCraft 的 6.6 FPS 在跨域 720p 设定下其实挺不错。要做到 60 FPS 估计得等下一代 model distillation + 硬件。

**判断 5: 真正的 contribution 是把矛盾解耦**

之前所有工作都在"稳定 vs 响应"的矛盾里选边站。GameCraft 用 hybrid training 把这个矛盾拆成两个独立能力，然后训练时随机切换强迫模型同时学。

这种"用 data mixture 解决 capability trade-off"的思路其实挺普适的。任何"两种能力看似矛盾"的场景都可以试试——比如 reasoning vs creativity，long context vs precise retrieval。

---

## 一句话总结

**把键盘鼠标翻译成连续 camera 参数（解耦方向和速度），用混合历史条件训练让模型同时学会"延续历史"和"响应动作"（解决长视频一致性 vs 控制精度矛盾），再用 PCM 蒸馏压到 6.6 FPS（实时交互门槛）**。

技术 trick 看着不复杂，但工程化程度是目前 interactive video generation 里最高的——跨 100+ AAA 游戏、continuous action、长视频 memory、接近实时 FPS，全开源。Genie 2 没开源，Oasis 限 Minecraft，GameNGen 限 DOOM，Matrix-Game 限 Minecraft。GameCraft 是第一个把所有维度都拉满的公开方案。

---

## 主要参考

- 项目主页: https://hunyuan-gamecraft.github.io/
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- PCM: https://arxiv.org/abs/2410.07184
- CameraCtrl: https://arxiv.org/abs/2404.02101
- GameNGen: https://arxiv.org/abs/2408.14837
- Oasis: https://www.decart.ai/articles/oasis-interactive-ai-video-game-model
- Matrix-Game: https://arxiv.org/abs/2501.08325
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Diffusion Forcing: https://arxiv.org/abs/2403.10695
- Flow Matching: https://arxiv.org/abs/2210.02747
- Monst3R: https://arxiv.org/abs/2410.03825
- RAFT: https://arxiv.org/abs/2003.12039
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- PySceneDetect: https://github.com/Breakthrough/PySceneDetect

---

# Hunyuan-GameCraft 深度讲解

Andrej，这篇 Tencent Hunyuan 团队的 paper 把 video diffusion 推进到一个近乎"可玩世界模型"的形态。核心是把离散的 keyboard/mouse 动作统一到一个连续 camera 参数空间，用 hybrid history conditioning 解决自回归长视频扩展的 consistency-accuracy trade-off，最后用 PCM 蒸馏把延迟压到 5 秒以内，做到 6.6 FPS 实时交互。我会把每个模块拆到公式变量级别，并补一些 paper 没明说但你应该关心的工程直觉。

项目主页: https://hunyuan-gamecraft.github.io/

---

## 1. 整体定位：在 game world model 谱系中的位置

Table 1 给了一个非常清晰的对比。把现有"可交互 world model / interactive video generation"放在四个 axis 上看：

| 工作 | Game Source | Resolution | Action Space | Generalizable | Dynamic | Memory |
|---|---|---|---|---|---|---|
| GameNGen (https://arxiv.org/abs/2408.14837) | DOOM only | 240p | Key | ✗ | ✓ | ✗ |
| GameGen-X (https://arxiv.org/abs/2410.09730) | AAA | 720p | Instruction | ✗ | ✓ | ✗ |
| Oasis (https://www.decart.ai/articles/oasis-interactive-ai-video-game-model) | Minecraft | 640×360 | Key+Mouse | ✗ | ✗ | ✗ |
| Matrix (https://arxiv.org/abs/2412.03568) | AAA | 720p | 4 Keys | ✓ | ✓ | ✗ |
| Genie 2 (https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) | Unknown | 720p | Key+Mouse | ✓ | ✗ | ✗ |
| GameFactory (https://arxiv.org/abs/2501.08325) | Minecraft | 640×360 | 7 Keys+Mouse | ✓ | ✓ | ✗ |
| Matrix-Game | Minecraft | 720p | 7 Keys+Mouse | ✓ | ✗ | ✓ |
| **Hunyuan-GameCraft** | **AAA** | **720p** | **Continuous** | **✓** | **✓** | **✓** |

直觉：唯一同时打勾 Generalizable + Dynamic + Memory，且 action 是 continuous 的。这点对 playability 是关键，因为 7 Keys+Mouse 的离散桶很容易在中间状态"卡死"，continuous 空间允许插值（比如"W+Shift 跑动"和"W 慢走"之间无级切换）。

---

## 2. Continuous Action Space：为什么用 𝕊² × [0, v_max]

### 公式 (1) 拆解

$$
\mathcal{A} := \left\{ \mathbf{a} = (\mathbf{d}_{\text{trans}}, \mathbf{d}_{\text{rot}}, \alpha, \beta) : \mathbf{d}_{\text{trans}} \in \mathbb{S}^2,\ \mathbf{d}_{\text{rot}} \in \mathbb{S}^2,\ \alpha \in [0, v_{\max}],\ \beta \in [0, \omega_{\max}] \right\}
$$

变量逐项：
- $\mathbf{d}_{\text{trans}} \in \mathbb{S}^2$：平移方向单位向量，定义在 2-sphere（三维空间中的单位球面）。意味着三个分量 $(d_x, d_y, d_z)$ 满足 $d_x^2+d_y^2+d_z^2=1$，去掉 1 个自由度，剩 2 个 DOF（用球面坐标 $\theta, \phi$ 参数化最自然）
- $\mathbf{d}_{\text{rot}} \in \mathbb{S}^2$：旋转方向单位向量，同样在 2-sphere 上。GameCraft 显式砍掉了 roll 自由度，只保留 yaw 和 pitch，所以旋转其实只剩 2 DOF
- $\alpha \in [0, v_{\max}]$：平移速度幅值（标量），上限 $v_{\max}$
- $\beta \in [0, \omega_{\max}]$：旋转角速度幅值（标量），上限 $\omega_{\max}$

paper 说 "they are the differential modulus of relative velocity and angle during frame-by-frame motion"——意思是 $\alpha, \beta$ 是相邻帧间相对位移/相对角度的模长，不是绝对位姿。这是 differential control，对应 controller 输入习惯（按住 W 一直走，松开停）。

### 为什么这个参数化聪明

1. **方向 × 速度 解耦**：传统 6-DoF pose $(t_x,t_y,t_z, r_x,r_y,r_z)$ 把方向和速度耦合在一个 6 维向量里，模型得自己学会"模长 = 速度"这件事；这里显式拆开，模型只学方向 + 学速度，各是一个低维问题
2. **球面 ≠ 欧氏**：把方向约束在 $\mathbb{S}^2$ 上避免了"学出来 $(0.7, 0.7, 0.1)$ 这种模长不为 1 的非归一化方向"。这个 manifold 约束本质是个 prior
3. **去掉 roll**：游戏玩家几乎不用 roll（除了飞行模拟器），砍掉后训练数据更聚焦，模型容量不被稀疏 roll 浪费
4. **可插值**：两个动作 $\mathbf{a}_1, \mathbf{a}_2$ 在 $\mathbb{S}^2$ 上用 SLERP 插值，在 $[0, v_{\max}]$ 上用 lerp，得到一个合法的动作。离散 keyboard space 做不到这点

### Plücker embedding 转换

paper 说"can be seamlessly converted into standard camera trajectory parameters and Plücker embeddings"。Plücker line coordinates 是 3D 几何中线的最小表示：

$$
\mathbf{L} = (\mathbf{d}, \mathbf{m}) = (\mathbf{d},\ \mathbf{p} \times \mathbf{d})
$$

- $\mathbf{d} \in \mathbb{S}^2$：线方向（与上面 $\mathbf{d}_{\text{rot}}$ 概念一致）
- $\mathbf{m} = \mathbf{p} \times \mathbf{d}$：moment，$\mathbf{p}$ 是线上任一点到原点的向量
- $\mathbf{m} \perp \mathbf{d}$ 恒成立（Plücker constraint）

CameraCtrl (https://arxiv.org/abs/2404.02101) 最早把 Plücker 引入 video generation。每条像素 ray 用一个 6D Plücker 表示，堆成 $H \times W \times 6$ 的 tensor。GameCraft 用同样的表示，但 encoder 大幅简化（见下）。

### Lightweight Action Encoder

paper 的原话："our encoding network consists solely of a limited number of convolutional layers for spatial downsampling and pooling layers for temporal downsampling. A learnable scaling coefficient is incorporated to automatically optimize the relative weighting during token-wise addition."

对比：
- **CameraCtrl**：cascaded residual blocks（多个 ResBlock 堆叠）做 Plücker encoder
- **MotionCtrl** (https://arxiv.org/abs/2312.03641)：transformer blocks
- **GameCraft**：conv + pooling 即可

为什么这么轻就够？因为 (1) Plücker embedding 本身就时空对齐到 video latent；(2) MM-DiT backbone 已经有多模态融合能力（来自 HunyuanVideo 的训练），不需要 encoder 重新学融合；(3) learnable scaling coefficient 让模型自适应 token 间的相对权重，是个"软 attention"的廉价替代。

Injection 用 **token addition**：把 action token 加到 patchified video token 上。ablation Table 4 显示 Token Addition (g) RPE trans 0.08，优于 Token Concat (c) 0.13 和 Channel-wise Concat (d) 0.11。直觉：加法保留了 video latent 的几何结构（不引入新维度），concat 在 channel/token 维度拼接会让 attention 路径复杂化。

参考 HunyuanVideo: https://arxiv.org/abs/2412.03603

---

## 3. Hybrid History-Conditioned Long Video Extension

这是 paper 的核心创新，我详细讲。

### 三种扩展范式对比（Fig 5）

**(i) Training-free inference from single image**
- 每次自回归只看上一段的最后一帧
- 缺点：历史 context 太少，迭代几步后 scene collapse（背景漂移、物体凭空消失）
- Fig 6(a) 显示质量塌方

**(ii) Streaming generation with non-uniform noise windows**
- StreamingT2V (https://arxiv.org/abs/2403.14773) 的思路：在一个 noise schedule 上滚动，老帧 noise level 高，新帧 noise level 低
- 缺点：和 causal VAE 不兼容。Causal VAE (https://arxiv.org/abs/2107.11553) 对首帧和后续帧的编码不均匀（首帧信息更"重要"），用 non-uniform noise 等于在 VAE 不擅长的部分施压
- 架构上与 image-to-video foundation model 不兼容

**(iii) Chunk-wise extension with head latent (本文)**
- 每个自回归 step：denoise 一个 chunk latent，由 head latent + action 信号引导
- Head latent 三种形式：
  - **(I) single frame latent**：只看上一 chunk 的最后一帧
  - **(II) final latent of previous clip**：上一 chunk 的整个 latent（信息更丰富）
  - **(III) longer latent clip segment**：上几段 chunk 的拼接（信息最丰富）

### Mask Indicator 机制

在 condition level 和 noise level 都做 concatenation：

```
[head_latent (mask=1) | chunk_latent (mask=0)]
```

- mask=1 的 region 保持 noise-free clean latent（不参与去噪）
- mask=0 的 region 是 noisy chunk latent，通过 flow matching 去噪
- 这个 mask 是 binary，但理论上可以做 soft mask（paper 没做，但是个明显的扩展点）

### Flow Matching 视角

HunyuanVideo 用 rectified flow（flow matching 的特例）。给定噪声 $z_t = (1-t) \cdot z_0 + t \cdot \epsilon$（前向加噪），速度场 $u_\theta$ 学习从 $z_t$ 到 $z_0$ 的 ODE 速度：

$$
\frac{dz_t}{dt} = u_\theta(z_t, t, \text{condition})
$$

paper 在 hybrid history conditioning 下，head 部分被钉死（$t=0$ 不变），chunk 部分从 $t=1$ 走到 $t=0$。这相当于 in-context conditioning，类似 ControlNet 但用 attention 内部 mask 实现。

Flow Matching 原始 paper: https://arxiv.org/abs/2210.02747

### Trade-off 的本质

paper 在 Fig 6 做了一个非常清晰的实验：

| 条件 | 历史信息 | 交互精度 | 一致性/质量 |
|---|---|---|---|
| (e) Image Condition | 一帧 | 高 (RPE trans 0.07) | 多步后塌方 |
| (f) Clip Condition | 一段 clip | 低 (RPE trans 0.16) | 高 |
| (g) Hybrid | 三种混合 | 高 (0.08) | 高 |

为什么 clip condition 让交互精度变差？训练数据来自切分的长视频，相邻 chunk 之间运动是连续的。模型学到"下一个 chunk 的运动是上一 chunk 的自然延续"这个 prior。prior 越强，对动作变化的响应越弱——比如上一 chunk 是 W 前进，下一 chunk 真值也是 W 前进，模型即使收到 S 信号也倾向于继续 W。

Image condition 没有这个问题（一帧信息少），但缺乏历史 context，几步迭代后 VAE 累积误差让背景漂移。

### Hybrid 训练策略

训练时按比例采样三种 condition：

$$
P(\text{single clip}) = 0.70,\quad P(\text{multi clip}) = 0.05,\quad P(\text{single frame}) = 0.25
$$

直觉：0.70 single clip 主导，提供强一致性 prior；0.05 multi clip 偶尔来一次长 context，教模型处理长依赖；0.25 single frame 提供足够样本让模型在历史信息少时也能做出动作响应。

**这个 mixture 是关键**。它在每个 training step 随机切换 head condition，强迫模型同时学习两种能力。否则单独训一种会偏向该种能力。

### 一个你应该问的问题

这个 mixture 0.7/0.05/0.25 是 paper 试出来的还是搜出来的？paper 没说，但 ablation Table 4 (e)(f)(g) 只对比了纯 image / 纯 clip / hybrid 三种，没扫 mixture ratio。这是个明显的 future work 方向——可以 grid search 找到最优 mixture。

---

## 4. 加速：PCM + CFG Distillation

### 公式 (2) 拆解

$$
L_{\text{cfg}} = \mathbb{E}_{w \sim p_w, t \sim U[0,1]} \left[ \| \hat{u}_\theta(z_t, t, w, T_s) - u_\theta^s(z_t, t, w, T_s) \|_2^2 \right]
$$

$$
\hat{u}_\theta(z_t, t, w, T_s) = (1+w)\, u_\theta(z_t, t, T_s) - w\, u_\theta(z_t, t, \emptyset)
$$

变量逐项：
- $z_t$：noisy latent at time $t$，从 $U[0,1]$ 采样的 noise level
- $t \in [0,1]$：flow matching 时间参数，$t=0$ 是 clean，$t=1$ 是纯噪声
- $w$：CFG guidance scale，从分布 $p_w$ 采样（不是固定值，让 student 在不同 $w$ 下都能 match teacher）
- $T_s$：prompt（text condition）
- $\emptyset$：空 prompt（unconditional）
- $u_\theta(z_t, t, T_s)$：teacher 的 conditional 速度场（带 prompt）
- $u_\theta(z_t, t, \emptyset)$：teacher 的 unconditional 速度场
- $\hat{u}_\theta$：teacher 的 CFG 输出，即 $(1+w) \cdot \text{cond} - w \cdot \text{uncond}$，标准 CFG 公式
- $u_\theta^s$：student 的速度场，**直接吃 $w$ 作为输入**，不需要外部 CFG

直觉：teacher 推理时要跑两次 forward（cond + uncond）然后线性组合，student 一次 forward 直接吐出 guided 结果。把 CFG 的"两次 forward + 线性组合"压进 student 单次 forward。

### Phased Consistency Model (PCM)

PCM (https://arxiv.org/abs/2410.07184) 是 Consistency Model (https://arxiv.org/abs/2303.01469) 的改进版本。原版 Consistency Model 在整个 trajectory 上施加 consistency constraint，PCM 把 trajectory 分成几个 phase，每个 phase 内 consistency，phase 之间用 diffusion 跳跃。这样能在 8 步内做到接近 diffusion 的质量。

游戏交互 8 步推理 = 单 action 5 秒以内 = 6.6 FPS。这个 FPS 接近可玩——虽然比真游戏 30/60 FPS 还差一个数量级，但已经足够 demo 和探索型 gameplay。

### 加速代价

Table 2 对比 Ours vs Ours+PCM：

| 指标 | Ours | Ours+PCM | 损失 |
|---|---|---|---|
| FVD | 1554.2 | 1883.3 | +21% |
| Image Quality | 0.69 | 0.67 | -3% |
| Dynamic Average | 67.2 | 43.8 | **-35%** |
| Aesthetic | 0.67 | 0.65 | -3% |
| Temporal Consistency | 0.95 | 0.93 | -2% |
| RPE trans | 0.08 | 0.08 | 0 |
| RPE rot | 0.20 | 0.20 | 0 |
| FPS | 0.25 | **6.6** | **26×** |

**关键 trade-off**：控制精度不变（RPE 0.08/0.20 完全保留），但动态性能掉 35%。直觉：consistency model 倾向于"保守"生成（少做大动作），因为 phase 间的 step 数少，模型不敢做激进运动以免 phase 内发散。这个 trade-off 在游戏里可以接受——玩家要的是"我按 W 角色真的往前走"，而不是"角色走得很炫"。

---

## 5. 数据集工程：这是被严重低估的部分

### 四阶段 Pipeline（Fig 3）

1. **Scene and Action-aware Partition**
   - PySceneDetect (https://github.com/Breakthrough/PySceneDetect)：检测场景切换，2-3 小时录屏切成 6 秒连贯 clip
   - RAFT (https://arxiv.org/abs/2003.12039) 计算光流梯度，找动作边界（比如快速 aim down sight 的瞬间）
   - 双层切分：scene-level 保证视觉连贯，action-level 保证动作边界对齐
   - 输出：1M+ clips at 1080p

2. **Data Filtering**
   - Quality assessment [17]（Kolors 团队的质量评估）：低分 clip 剔除
   - OpenCV 亮度过滤：暗场景剔除（游戏里大量过场动画很暗，对生成质量不利）
   - VLM 梯度检测：用 Qwen2-VL (https://arxiv.org/abs/2409.12191) 检测梯度异常（模糊、抖动）

3. **Interaction Annotation**
   - Monst3R (https://arxiv.org/abs/2410.03825)：从单目视频重建 6-DoF camera trajectory
   - 每帧标注 position + orientation
   - 这是把"视频"变成"带标签的可训练数据"的关键一步

4. **Structured Captioning**
   - 层级策略：30 字符简洁摘要 + 100+ 字符详细描述
   - 训练时随机采样（类似 CLIP text augmentation）

### Distribution Balancing

游戏视频有个明显的 **forward-motion bias**：90% 时间玩家在前进。如果直接训练，模型会学成"给任何动作都生成前进"。

两个策略：
1. **Stratified sampling of start-end vectors**：把 start-end 位移向量在 3D 空间离散成 buckets，每个 bucket 等概率采样
2. **Temporal inversion augmentation**：把视频时间反转，前进展变成后退。这样后退数据翻倍

这两个 trick 把 RPE trans 从 0.16（only live data）压到 0.08（最终模型）。

### Synthetic Data 的角色

paper 渲染了 3000 个高质量 motion sequence（从 curated 3D assets），采样多个起点位置，渲染 translation / rotation / composite 运动，不同速度。

为什么需要？**几何 prior**。真实游戏视频的 camera trajectory 是隐式的（要 Monst3R 重建才能得到），有重建误差。合成数据 ground truth 是精确的，能教模型"严格的几何规律"。但合成数据缺少动态物体（NPC、特效），所以单独训会动态性能差。

Ablation Table 4：
- (a) Only Synthetic: DA 34.6（动态差），RPE trans 0.07（控制准）
- (b) Only Live: DA 77.2（动态好），RPE trans 0.16（控制差）
- (g) Render:Live=1:5: DA 67.2, RPE trans 0.08（两者兼得）

**1:5 是甜蜜点**。这个比例的直觉：合成数据太"干净"，多了会让模型对真实视频的 noise 不 robust；少了又得不到几何 prior。

---

## 6. 实验：定量结果深入读

### Table 2 主对比

| Model | FVD↓ | IQ↑ | DA↑ | Aes↑ | TC↑ | RPE trans↓ | RPE rot↓ | FPS↑ |
|---|---|---|---|---|---|---|---|---|
| CameraCtrl | 1580.9 | 0.66 | 7.2 | 0.64 | 0.92 | 0.13 | 0.25 | 1.75 |
| MotionCtrl | 1902.0 | 0.68 | 7.8 | 0.48 | 0.94 | 0.17 | 0.32 | 0.67 |
| WanX-Cam | 1677.6 | 0.70 | 17.8 | 0.67 | 0.92 | 0.16 | 0.36 | 0.13 |
| Matrix-Game | 2260.7 | 0.72 | 31.7 | 0.65 | 0.94 | 0.18 | 0.35 | 0.06 |
| **Ours** | **1554.2** | 0.69 | **67.2** | 0.67 | **0.95** | **0.08** | **0.20** | 0.25 |
| Ours+PCM | 1883.3 | 0.67 | 43.8 | 0.65 | 0.93 | 0.08 | 0.20 | **6.6** |

几个值得注意的点：

1. **Dynamic Average 67.2 是 Matrix-Game 31.7 的 2.1 倍**。但 FVD 只比 Matrix-Game 好 31%。FVD 衡量分布距离，动态大的视频更难（分布更广），所以 GameCraft 在动态大 2 倍的情况下 FVD 还更小，含金量很高

2. **RPE trans 0.08 vs Matrix-Game 0.18**，误差降 55%。这正是 hybrid history conditioning + continuous action space 的功劳

3. **FPS：Matrix-Game 0.06** 几乎不可玩；**Ours+PCM 6.6** 接近实时

4. **Image Quality 0.69 < Matrix-Game 0.72**：GameCraft 在单帧质量上略输。这是因为 Dynamic Average 大意味着模型做更激进的运动，单帧质量自然要牺牲一点。这是 dynamic vs fidelity 的内在 trade-off

### Table 3 用户研究

GameCraft 在 5 个维度（Video Quality, Temporal Consistency, Motion Smooth, Action Accuracy, Dynamic）都拿到 4.42-4.61 分（满分 5），其他方法在 2.2-3.2 之间。这个差距比定量指标大得多——说明用户感知到的体验优势比 FVD 数字更显著。

### 跨域泛化

Section 6 展示 GameCraft 在真实世界图像上也能 work（Fig 10）。这是 HunyuanVideo 作为 foundation model 的功劳——预训练时学到的大量真实世界视频 prior 没被游戏数据 erase。

这点其实有点 surprising，因为 fine-tune 通常会 hurt 原域能力。可能的原因：
- Game 数据训练量相对预训练小
- Action space 是新维度，不直接冲突 video prior
- Hybrid condition 的 single frame mode 保留了原 image-to-video 能力

---

## 7. Limitations & 联想

paper 自己说 limitation 是 action space 只支持 open-world exploration，缺少 shooting/throwing/explosion 这类 game-specific action。

我的几个联想：

### 7.1 Action Space 扩展

连续 camera space 解决了 navigation，但游戏里大量 action 是"对世界做事情"——开枪、扔手雷、跳跃、闪避。这些 action 不是 camera 参数，是离散事件 + 物理效应。

可能的扩展：
- 在 $\mathcal{A}$ 上加 discrete event token：$\mathbf{a} = (\mathbf{d}_{\text{trans}}, \mathbf{d}_{\text{rot}}, \alpha, \beta, \mathbf{e})$，$\mathbf{e}$ 是 one-hot 或 multi-hot event vector
- 用 ControlNet-style 注入 event（与 camera action 解耦）
- 训练数据需要新标注 event 类型

### 7.2 Memory Mechanism 的深层问题

paper 的"memory"本质是 short-term context（前一段 clip 的 latent）。但玩家记得"5 分钟前我在那个房子里看到一把枪"，这种 long-term episodic memory 当前架构完全无法支持。

可能的路径：
- Retrieval-augmented generation：从历史 clip library 检索相关帧
- External state tracker：维护一个 symbolic state（"玩家在位置 X，枪在位置 Y"），喂给 LLM 做推理
- 这接近 LeCun 的 H-JEPA 思路：maintain world model state，不是 raw latent

### 7.3 与 GameNGen 对比

GameNGen (https://arxiv.org/abs/2408.14837) 在 DOOM 上做到 50 FPS，2-step diffusion + condition via noise augmentation。为什么 GameCraft 只有 6.6 FPS？

- DOOM 是 240p，GameCraft 是 720p（4-9 倍像素）
- DOOM 是单游戏，模型可以高度过拟合；GameCraft 要跨 100+ AAA games
- GameNGen 用 noise conditioning 加 historical frames，本质上也是个 history conditioning，但 DOOM 简单所以不需要 hybrid
- GameCraft 用 PCM 8 步，GameNGen 用 2 步——后者用了 fine-tuned small model 才能做到

### 7.4 与 Diffusion Forcing 的关系

Diffusion Forcing (https://arxiv.org/abs/2403.10695) 把 next-token prediction 和 full-sequence diffusion 结合。它的"full-sequence"是把整段视频当成一个序列，每帧有独立 noise level，模型同时处理不同 noise level 的帧。

GameCraft 的 hybrid history conditioning 是 Diffusion Forcing 的简化版：
- 历史帧 noise level = 0（clean）
- 新 chunk noise level 从 1 走到 0
- Diffusion Forcing 允许历史帧也有 noise level，做更灵活的 trade-off

GameCraft 这个简化更稳定（mask 是 binary），但牺牲了 Diffusion Forcing 的灵活性。一个 future direction 是把 mask 改成 soft noise level。

### 7.5 Causal VAE 的角色

paper 提到"causal VAE's uneven encoding of initial versus subsequent frames fundamentally limits efficiency and scalability"——这是反对 streaming 方案的关键论点。

Causal VAE (https://arxiv.org/abs/2107.11553) 用 causal attention 让首帧信息在 latent 里占主导。这设计是为了让"从首帧生成后续帧"任务里，首帧 latent 信息充分。但 streaming 方案需要后续帧也平等对待，causal VAE 反而成了 bottleneck。

GameCraft 绕开这个问题的方式：每个 chunk 独立 VAE 编码（chunk 内首帧信息主导），然后 chunk 之间用 latent-level conditioning。这等于把 causal VAE 的优点（chunk 内首帧强）保留，把缺点（chunk 间不平等）规避。

### 7.6 PCM 在 game 场景的特殊性

PCM 蒸馏后 Dynamic Average 从 67.2 掉到 43.8，掉 35%。这在 game 场景特别伤，因为 game 的卖点就是 high dynamic。

可能的改进：
- 在 distillation 时加 dynamic-preserving loss（比如 optical flow magnitude 的 L1 loss）
- 在 distillation dataset 上 oversample high-dynamic clips
- 用 adaptive step PCM：dynamic 大时多步，dynamic 小时少步

---

## 8. 复现关注点

如果你要复现或改进，几个关键超参：

- **Action mixture ratio** 0.70/0.05/0.25：paper 没扫这个，是个低垂果实
- **Render:Live = 1:5**：这是 ablation 试出来的，跨域可能要调
- **Hybrid mask binary**：可以尝试 soft mask（mask value in [0,1]）作为 continuous interpolation
- **6 秒 clip at 1080p**：与训练时的 chunk size（33 帧 @ 25 fps = 1.32 秒）不对应，说明训练时是从 6 秒 clip 里截 1.32 秒 window
- **192 H20 GPU, 30k+20k iterations, batch 48**：约 1.5M 样本 throughput，对应 1M clip 跑 1.5 epoch

---

## 9. 我的整体直觉

GameCraft 的真正贡献是**把游戏交互的离散 action 翻译成几何连续 camera 参数，然后用 hybrid training 让模型同时学"跟随历史"和"响应新动作"两种模式**。这俩能力在 prior 工作里是矛盾的，hybrid mixture 是个 elegant 解决。

技术风险点：
- Hybrid mixture 是 hand-tuned，没有理论指导
- Memory 只是 short-term，long-term memory 完全没有
- Action space 是 camera-only，game-specific action 需要重新设计
- PCM 加速后 dynamic 掉 35%，是 game 场景的硬伤

但作为 demo 级别的"interactive video world model"，这是目前我看到的最完整工程化方案。Genie 2 没开源，Oasis 限 Minecraft，GameNGen 限 DOOM，Matrix-Game 限 Minecraft。GameCraft 是第一个跨 100+ AAA game + continuous action + 实时 FPS 的公开方案。

---

## 主要参考链接

- 项目主页: https://hunyuan-gamecraft.github.io/
- HunyuanVideo (base model): https://arxiv.org/abs/2412.03603
- Phased Consistency Model: https://arxiv.org/abs/2410.07184
- CameraCtrl (Plücker embedding 先驱): https://arxiv.org/abs/2404.02101
- MotionCtrl: https://arxiv.org/abs/2312.03641
- GameNGen: https://arxiv.org/abs/2408.14837
- GameGen-X: https://arxiv.org/abs/2410.09730
- Oasis: https://www.decart.ai/articles/oasis-interactive-ai-video-game-model
- Matrix: https://arxiv.org/abs/2412.03568
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Diffusion Forcing: https://arxiv.org/abs/2403.10695
- Flow Matching: https://arxiv.org/abs/2210.02747
- Consistency Model: https://arxiv.org/abs/2303.01469
- Latent Consistency Model: https://arxiv.org/abs/2310.04378
- Monst3R: https://arxiv.org/abs/2410.03825
- RAFT: https://arxiv.org/abs/2003.12039
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Causal VAE: https://arxiv.org/abs/2107.11553
- PySceneDetect: https://github.com/Breakthrough/PySceneDetect
- StreamingT2V: https://arxiv.org/abs/2403.14773
- GameFactory: https://arxiv.org/abs/2501.08325
