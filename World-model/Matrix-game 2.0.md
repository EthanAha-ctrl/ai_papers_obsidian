---
source_pdf: Matrix-game 2.0.pdf
paper_sha256: 100144e119594acfc76dc21f9f38e4c7fb3b15ad7f0b65748347bf4865607c56
processed_at: '2026-08-05T16:51:53-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej, 咱们抛开那些学术黑话，用最直白的话来聊聊这篇 paper 到底在搞什么名堂。

简单来说，这帮人想用 AI 造一个“能在脑子里实时跑的游戏引擎”。你给它一张图，加上你此时此刻按下的键盘和鼠标，它立刻给你生成下一帧画面。就像你在打游戏，游戏引擎是根据物理代码算出下一帧，而这个 model 是根据它学到的“世界规律”直接“猜”出下一帧。

这事听起来玄乎，但难点就三个，Matrix-Game 2.0 就是围绕这三个痛点对症下药：

### 痛点 1：没数据怎么训？
你要教 AI “按下W键，画面会往前走”，你需要大量的视频，并且每一帧都得精确标注当时玩家按了什么键、鼠标怎么动的。现实里去录别人的游戏视频，按键和画面对不齐。
**他们的搞法：** 用 Unreal Engine 和 GTA5 自动造数据。
- 在 UE 里放一个 agent 让它自己乱跑，遇到墙就躲。为了防止算视角时 0.2% 的浮点误差在几百帧后累积成大错，他们用 double precision 去算 quaternion（四元数），硬是把误差抹平了。
- 把静止不动、卡在墙上的废数据全扔了，只要速度 $|\vec{v}| > \epsilon$ 的有效画面。
- 最后搞了 1200 小时极其精准的对齐数据。

### 痛点 2：算得太慢，没法实时
现在的 video diffusion models 基本都是 bidirectional attention（双向注意力），算一帧要把前后所有帧都看一遍。打游戏要求 25 FPS，你算一秒才出一帧，这没法玩。
**他们的搞法：** 把模型改成 Causal（因果）的，就像 LLM 一样，只看过去，推未来。然后疯狂加速：
- **Self-Forcing + DMD 蒸馏**：原来的 base model 要算很多步去噪，他们把它蒸馏成一个只需要 3 步的 student model。
- **砍 action module**：发现把键盘鼠标信号注入到 DiT block 的前半段就足够了，后半段直接砍掉，省一半算力。
- **VAE Cache**：解码历史画面时复用缓存。
三招下来，单张 H100 跑到 25 FPS，刚好够实时互动。

### 痛点 3：越生成越烂
Auto-regressive 模型有个死穴：训练时用真实画面当输入，推理时只能用自己生成的画面当输入。只要错一点点，雪球越滚越大，几十帧后画面就崩成马赛克或者静止画了。
**他们的搞法：** 
- **Self-Forcing**：训练的时候就让 model 吃自己生成的画面，让它学会在“自己的烂梦”里也能把轨迹拉回正轨。同时用 DMD loss 把它的分布锁在 teacher model 的高质量流形上，不让它乱飘。
- **限制 KV-cache 大小**：这是最反直觉的一点。大家都以为记忆越长越好，但他们发现 cache 存 9 帧时，模型一旦早期出了个小 artifact，就会把这个错记在 cache 里当真理，后面越错越离谱。把 cache 限制在 6 帧，强迫 model 忘掉过去，靠自己的 prior 和当前的 action 去纠错，长视频反而稳了。

### 一个有意思的细节：去掉 Text
现在主流的 SORA、Wan 都靠 text prompt 来控制生成。Matrix-Game 2.0 直接把 text branch 砍了个精光。他们的逻辑是：text 里面掺杂了人类语言的语义 prior，会干扰 model 对纯粹物理规律和 spatial intelligence 的学习。抛开语言描述，直接通过图像和动作去感知世界，反而能学到更本质的几何和物理直觉。

### 总结成一句话
它抛弃了语言标签，靠 1200 小时严丝合缝的按键录像训练，通过自我纠错的蒸馏方式，造了一个能在单卡上以 25 帧速度跑的实时互动世界模拟器，并且解决了长视频必然崩盘的顽疾。

For more intuition on the specific mechanisms:
- [Self-Forcing paper](https://arxiv.org/abs/2506.08009)
- [DMD (Distribution Matching Distillation)](https://arxiv.org/abs/2310.05910)
- [Wan 2.1 Architecture](https://arxiv.org/abs/2503.20314)

---

# Matrix-Game 2.0 深度讲解

这篇 paper 是 Skywork AI 在 interactive world model 方向的工作，核心目标是把 diffusion-based video model 从 "offline、batch、bidirectional" 的范式推进到 **real-time、streaming、causal** 的范式，同时还要支持 mouse + keyboard 的 frame-level 交互控制。下面我从 motivation、data、architecture、distillation、cache、experiments 几个层面把它讲透，build your intuition。

---

## 1. Motivation：为什么 existing interactive world model 不够用

Paper 一上来就把问题切成三个痛点，我重新组织一下：

**(P1) Data 缺失。** 现有的 interactive video dataset 几乎没有 frame-level 的 action annotation。普通 T2V / I2V dataset 只有 video 本身，没有对应的 keyboard / camera 输入信号。要训一个 "看到图像 → 给定动作 → 生成下一帧" 的模型，必须知道每一帧对应的 action 是什么。这件事在 real-world capture 上几乎不可能做精确（人按键盘的时刻和 frame timestamp 对不齐）。

**(P2) Bidirectional attention 不能 streaming。** Oasis、YUME、Hunyuan-GameCraft 这类模型，DiT block 用的是 full bidirectional attention，一帧的生成要看整个 sequence 的所有 frame。这意味着：
- 计算量随 frame 数二次方增长；
- 用户下一秒要按什么键，模型必须等历史全部 encode 完；
- 没法做 on-the-fly streaming。

**(P3) Auto-regressive 的 error accumulation。** 如果直接用 teacher-forcing 训 AR model（next frame based on previous GT frames），inference 时候用 self-generated frame 喂回去，train-test gap 会让 error 在长视频里越积越多，几十帧后画面就崩了。Oasis 就是典型例子。

Matrix-Game 2.0 的核心 claim 是同时解决这三个问题：用 UE + GTA5 pipeline 解决 data，用 **Self-Forcing + DMD distillation** 解决 P2/P3，最终在 H100 单卡跑 25 FPS、minute-level long video。

---

## 2. Data Pipeline：1200 小时 action-aligned 视频怎么造

这是 paper 里工程量最大的一块，分 UE 主线和 GTA5 副线。

### 2.1 Unreal Engine pipeline

输入：NavMesh + 3D scene。输出：MP4 + CSV (每帧的 keyboard 状态 + camera 参数)。

**NavMesh-based path planning**：基于 UE 原生 NavMesh 做自定义优化，单次 query latency < 2ms。关键是引入了 controlled stochasticity，让 agent 在合理 navigation constraint 内有行为多样性，防止 trajectory 退化成几种固定模式。

**RL-enhanced agent**：在 collision-based rule 之外，用 PPO 训了一个 agent，reward 是三项的加权和：

$$
R_t = \alpha \cdot R_{collision} + \beta \cdot R_{exploration} + \gamma \cdot R_{diversity} \quad (1)
$$

- $R_t$：时刻 $t$ 的 reward
- $R_{collision}$：碰撞惩罚项（越大越负）
- $R_{exploration}$：探索新区域奖励
- $R_{diversity}$：movement pattern 多样性奖励
- $\alpha, \beta, \gamma$：标量权重

rule-based collision 当 safety constraint，RL 提供 adaptive intelligence。这种 hybrid 设计比纯 rule 或纯 RL 都更稳定。

**Precise input capture**：用 UE 的 Enhanced Input system 同步 buffer 多个键盘事件：

$$
\mathrm{Input}_{\mathrm{frame}_i} = (\{k_1, k_2, \ldots, k_n\}, \mathrm{timestamp}_i) \quad (2)
$$

- $\mathrm{frame}_i$：第 $i$ 帧
- $k_j$：第 $j$ 个按键的 press/release 状态
- $\mathrm{timestamp}_i$：帧级时间戳

**Quaternion precision optimization**：原来 camera rotation 用 single-precision float 算 quaternion 会有 0.2% 的累积误差，换成 double precision 中间计算后误差降到 negligible。这点很小但很关键——0.2% 在长视频里就够把 camera 轨迹搞坏。

**Velocity-based data filtering**：

$$
\mathrm{validity} = \begin{cases} 1 & \text{if } \|\vec{v}\| > \epsilon \\ 0 & \text{otherwise} \end{cases} \quad (3)
$$

- $\vec{v}$：速度向量
- $\epsilon$：小正数阈值，处理浮点精度

把静止帧或物理上不合理的运动直接丢掉。同时用 OpenCV 做帧间冗余检测去重。

**多线程**：单张 RTX 3090 上跑 dual-stream，渲染线程 + 共享 memory pool，吞吐翻倍。

### 2.2 GTA5 pipeline

GTA5 用 Script Hook V 插件拦截 mouse + keyboard，每帧 RGB + 对应输入 JSON 同步落盘。三个模块：

- **Agent Behaviors**：autonomous navigation / NPC interaction / vehicle interaction，用 C# mod 注入
- **GTA V environment**：可调 vehicle density ∈ [0.1, 2.0]，NPC density ∈ [0.2, 1.5]，weather，time-of-day
- **Recording system**：OBS Studio 抓 MP4，Data Collector 写 CSV，synchronization mechanism 保证 frame-action 对齐

Camera 跟车的公式：

$$
\mathrm{Camera}_{position} = \mathrm{Vehicle}_{position} + \mathrm{offset} \times \mathrm{rotation} \quad (4)
$$

- $\mathrm{Vehicle}_{position}$：车辆世界坐标
- $\mathrm{offset}$：相机相对车辆的偏移向量
- $\mathrm{rotation}$：车辆当前朝向的旋转矩阵

per-tick 更新，保证 camera 始终和车辆姿态对齐。同时 runtime 查询 NavMesh 给 camera 做 spatial constraint，防止 camera 穿墙。

### 2.3 数据统计

最终规模 ~1200 小时：
- UE: 615 小时
- Minecraft: 153 小时
- Sekai (real-world open source): 85 小时（做了 frame resampling 对齐 UE 的 temporal dynamics）
- GTA driver: 574 小时
- Temple Run: 560 小时

整体 annotation accuracy >99%，camera rotation precision 提升 50×。

数据这块的 intuition 是：**interactive world model 的天花板不是 model size，而是 action-visual 对齐的精度**。0.2% 的 camera 误差在 100 帧后就足够让模型学到错误的 viewpoint-action mapping，所以 double precision + velocity filter 这些工程细节其实是 foundation。

---

## 3. Foundation Model Architecture

### 3.1 De-semanticized 设计

一个有意的选择：**完全去掉 text branch**。Wan 2.1、HunyuanVideo、SORA 都依赖 text 做 control，但 text 会把 linguistic prior 注入到 model 里，让 model 倾向做 "语义合理" 的生成，破坏纯物理 / 几何的 learning。Matrix-Game 2.0 的立场是 spatial intelligence——从 visual content + action 直接学物理 law，不要 text 当 crutch。

这一点和 LeCun 的 V-JEPA、World Labs 的 spatial intelligence 思路一致。

### 3.2 整体 pipeline

输入：单张 reference image + frame-level action sequence。
- Image → 3D Causal VAE encoder（spatial 8×8、temporal 4× 压缩）+ CLIP image encoder
- Action → action module 注入 DiT
- DiT 输出 latent tokens → 3D VAE decoder → video frames

VAE 压缩比 8×8×4 是从 Wan 2.1 直接继承的，相对于空间 8、时间 4 的压缩对训练效率贡献很大。

### 3.3 Action injection

两种 action 类型，两种注入方式：

**Mouse action（连续）**：mouse 的 $(dx, dy, dz)$ 直接 concat 到 input latent 上，过一个 MLP，再过 temporal self-attention 层。这里 concat 而不是 cross-attention，是因为 mouse 信号本身就是 dense、per-frame 的 continuous vector，concat + MLP 让它和 visual token 在同一空间里被 attention 处理更自然。

**Keyboard action（离散）**：keyboard 的多键状态做成 token sequence，通过 cross-attention 让 visual feature 去 query keyboard embedding。区别于 mouse 的原因是 keyboard 是稀疏离散事件集合，cross-attention 更适合表达 "visual frame 需要哪些键的信息"。

**RoPE 替换 sin-cos**：这是相对于 Matrix-Game 1.0 的关键改动。原始 Matrix-Game 用 sin-cos positional encoding 加到 keyboard input 上，sin-cos 在 long video 上 extrapolation 能力差；RoPE 在 attention 内部做 rotation，自然支持 long context 和 streaming 扩展。

公式上 RoPE 对 query/key 做：
$$
q'_i = R_i q_i, \quad k'_j = R_j k_j
$$
其中 $R_i, R_j$ 是基于相对位置 $i-j$ 的 2D rotation matrix，attention logit $\langle q_i, k_j \rangle$ 只依赖相对位置。这种设计让 KV-cache 在 AR generation 时天然契合——cache 里的 key 已经旋转过了，新来的 query 按相对位置算就行，不用重算历史。

### 3.4 Model 规模

从 SkyReels-V2-I2V-1.3B 初始化（Wan 2.1 架构），去掉 text branch，加 action module 到每个 DiT block，最终 1.8B 参数。训练 120k steps，lr=2e-5，batch=256。

---

## 4. Distillation：从 bidirectional teacher 到 causal few-step student

这是 paper 的方法核心。整个 distillation 分两阶段。

### 4.1 Stage 1: Causal student initialization via ODE trajectories

Teacher 是 bidirectional、多 step 的 Wan I2V foundation。直接拿 teacher 当 student 起点训 AR 会崩——因为 bidirectional attention 没有 causal mask，distribution 不对。所以先做一步 init：

1. 用 teacher 在不同 timestep $t \in \{3\text{-step subset of } [0, T]\}$ 上采样 ODE trajectories $\{x_t^i\}_{i=1}^N$
2. 把 N 帧 noisy input 切成 $L$ 个 chunk，每个 chunk 有独立的 timestep $\{x_T^i\}_{i=1}^L$
3. Student generator $G_\phi$ 拿 chunk + action + timestep，输出 denoised prediction
4. Loss 是简单 regression：

$$
\mathcal{L}_{\mathrm{student}} = \mathbb{E}_{x, t^i} \left\| G_\phi\left(\{x_{t^i}^i\}_{i=1}^L, \{c^i\}_{i=1}^L, \{t^i\}_{i=1}^L\right) - \{x_0^i\}_{i=1}^L \right\|^2 \quad (5)
$$

- $G_\phi$：student generator，参数 $\phi$
- $x_{t^i}^i$：第 $i$ 个 chunk 在 timestep $t^i$ 的 noisy latent
- $c^i$：第 $i$ 个 chunk 的 action condition
- $t^i$：第 $i$ 个 chunk 的 diffusion timestep（不同 chunk 可以不同 step）
- $x_0^i$：第 $i$ 个 chunk 的 clean latent（GT）

关键点：**每个 attention layer 的 key/value 加 block-wise causal mask**。这是把 bidirectional transformer 改造成 causal 的标准做法——只用历史 token 的 K/V。

这一步 fine-tune 6k steps，lr=6e-6。

### 4.2 Stage 2: DMD-based Self-Forcing

Self-Forcing 的核心 idea：训练时候不要用 GT 当历史，而是从 **student 自己的 distribution** 里 sample 历史帧，再 denoise 下一帧。这就直接消除了 teacher-forcing 的 train-test gap。

DMD (Distribution Matching Distillation) 在这里用来对齐 student distribution $p_{\theta, t}(x_t^{1:N})$ 和 teacher distribution $p_{real, t}(x_t^{1:N})$。具体来说：

- Teacher（多 step bidirectional）给一个 "理想" 的 multi-frame distribution
- Student（few-step causal）通过 self-conditioned generation 产生 sample
- DMD loss 强制 student sample 的分布逼近 teacher

直觉上：teacher 是 "上帝视角" 的完整序列分布，student 是 "盲人摸象" 一帧一帧往前推的分布，DMD 让两者在 marginal distribution 上一致，这就让 student 即使只看自己生成的历史，最终能落到的 manifold 和 teacher 一致，不会漂移到 OOD。

**Chunk size = 3 latent frames，attention local size = 6**。chunk size 是一次 self-force 多少帧，local size 是 KV-cache 看多远的历史。

### 4.3 为什么这套组合能解决 P3 (error accumulation)

Teacher-forcing：训用 GT，推用 self → mismatch → error accumulate。
Self-Forcing：训和推都 self → no mismatch → error 不累积。
DMD：保证 self-generated sample 还在 teacher 的 manifold 上 → 长视频不漂移。

加上 causal mask + KV-cache，三件事加一起才能做到 minute-level long video。

### 4.4 Action sequence design

Self-Forcing 是 data-free 的——不需要 GT frame，只需要 action sequence。这意味着可以 **手动设计** action 分布，让它更贴近真实用户的输入分布（而不是脚本产生的 random action）。这个细节容易被忽略，实际上对 controllability 影响很大。

---

## 5. KV-Cache 与 Long Video Generation

### 5.1 Rolling cache

每个 attention layer 维护一个 fixed-length KV cache，存最近的 latent + action embedding。超过 capacity 自动 evict 最老的 token。这是支持 "infinite length generation" 的关键。

### 5.2 Cache size 的反直觉发现

这是 ablation 里最有意思的一点。理论上 cache 越大历史信息越全，应该越好。但实验显示：

| Cache size | Long video quality |
|---|---|
| 9 latent frames | 早期 artifact 出现，越往后越崩 |
| 6 latent frames | 长程质量明显更好 |

Paper 给的解释是：**cache 越大，model 越依赖 cache 而不主动纠错**。早期某一帧的 artifact 被缓存下来，模型把它当 valid scene element 记住，下一帧又基于这个错误 cache 生成，artifact 被放大、被当成"事实"。

这其实和 LLM long-context 的 "lost in the middle" 现象有点像——context 太长模型反而不去用关键信息。这里则是 cache 太大模型不去主动修正。

### 5.3 I2V 训练时 cache 窗口的额外约束

I2V 场景下，第一帧是 reference image，长 video 推理时第一帧会被 cache 挤出去。为防止训练时 model 把第一帧当 crutch，paper 故意 constrains KV-cache window size，让训练时第一帧对后续 frame "不可见"，强制 model 学 action-conditioned prior。这是个很巧妙的 robustness trick。

---

## 6. Experiments

### 6.1 Minecraft 对比 Oasis

Table 1 的数据：

| Model | Image Quality ↑ | Aesthetic ↑ | Temporal Cons. ↑ | Motion smooth. ↑ | Keyboard Acc. ↑ | Mouse Acc. ↑ | Obj. Cons. ↑ | Scenario Cons. ↑ |
|---|---|---|---|---|---|---|---|---|
| Oasis | 0.27 | 0.27 | 0.82 | 0.99 | 0.73 | 0.56 | 0.18 | 0.84 |
| Ours | 0.61 | 0.50 | 0.94 | 0.98 | 0.91 | 0.95 | 0.64 | 0.80 |

Image quality 直接从 0.27 → 0.61，object consistency 从 0.18 → 0.64。Oasis 在几十帧后画面就崩成静态帧，所以 motion smoothness 和 scenario consistency 看着高，是 artifact 的副产品。Matrix-Game 2.0 真正能保持动态、连续的长视频。

### 6.2 Wild scenes 对比 YUME

Table 2：

| Model | Image Quality ↑ | Aesthetic ↑ | Temporal Cons. ↑ | Motion smooth. ↑ | Obj. Cons. ↑ | Scenario Cons. ↑ |
|---|---|---|---|---|---|---|
| YUME | 0.65 | 0.48 | 0.85 | 0.99 | 0.77 | 0.80 |
| Ours | 0.67 | 0.51 | 0.86 | 0.98 | 0.71 | 0.76 |

视觉指标基本打平，YUME 在 object/scenario consistency 上略高，但 paper 指出 YUME 在 OOD 场景几百帧后崩成静态画面，这两项指标虚高。Matrix-Game 2.0 的优势在 **速度**——YUME 太慢，做不了 interactive；Matrix-Game 2.0 在 25 FPS。

### 6.3 加速消融

Table 3 是从 15 FPS 到 25 FPS 的 stack：

| Techniques | Image ↑ | Aesthetic ↑ | Temporal ↑ | Motion ↑ | Keyboard ↑ | Mouse ↑ | Object ↑ | Scenario ↑ | FPS ↑ |
|---|---|---|---|---|---|---|---|---|---|
| (1) +VAE Cache | 0.61 | 0.51 | 0.93 | 0.97 | 0.91 | 0.95 | 0.68 | 0.81 | 15.49 |
| (2) (1)+Halving action modules | 0.61 | 0.51 | 0.94 | 0.97 | 0.92 | 0.95 | 0.63 | 0.81 | 21.03 |
| (3) (2)+Reducing denoising steps (4→3) | 0.61 | 0.50 | 0.94 | 0.98 | 0.91 | 0.95 | 0.64 | 0.80 | 25.15 |

三个 trick 叠加：
1. **Wan2.1-VAE caching**：VAE decoder 对历史 latent 做 cache，长视频不用每帧重新 decode 全部
2. **Action module 减半**：action module 只在 DiT block 的前半段注入（DiT 深 30+ block，前半就够 controllability，后半纯视觉 refinement）
3. **Denoising step 4 → 3**：distillation 后再砍一步

每个 trick 几乎不掉指标，FPS 从 15.49 一路涨到 25.15。这个 stack 的设计逻辑是：action module 和 denoising step 在长视频 inference 时占大头，VAE decode 占小头，所以先攻大头。

### 6.4 Generalization

GTA driving (Fig.14)、TempleRun (Fig.15) 都是 fine-tune 后的效果，证明 framework 对 dynamic scene 通用。

---

## 7. Limitations

Paper 自己承认三个：

1. **OOD 泛化**：camera 长时间向上看或长时间前进，在 OOD 场景会 over-saturated 或 degraded。原因是 action distribution 训练时没覆盖这种 long-horizon 极端动作。
2. **Resolution 352×640**：远低于 SOTA video gen model 的 720p/1080p。
3. **长程一致性**：没有 explicit memory mechanism，minute 级以上还是会有 drift。

Bad cases 见 Fig.17。

---

## 8. Build Your Intuition：几个值得记住的点

1. **Action-visual alignment 是 interactive world model 的真天花板**。0.2% camera 误差能毁掉长视频，所以 paper 花大力气在 UE pipeline 上做 double precision quaternion + velocity filter。这点比 model size 重要。

2. **De-semanticization 是有趣的赌注**。去掉 text branch 看似丢了一个强大的 condition signal，但换来的是 model 直接学物理 / 几何 prior，不被语言概念绑架。这和 V-JEPA、World Labs 的 spatial intelligence 路线一致。

3. **Self-Forcing + DMD 是解决 AR diffusion error accumulation 的正确组合**。Self-Forcing 消 train-test gap，DMD 保 distribution 不漂。单纯 self-forcing 容易 drift 到低质量 manifold，DMD 是 anchor。

4. **KV-cache size 不是越大越好**。这和 LLM long-context 的直觉相反。Cache 太大让 model 懒得纠错，cache 适中反而迫使 model 学会 self-correct。这个 finding 对所有 AR diffusion model 都有启发。

5. **Action module 减半不掉 controllability**。说明 action signal 在 DiT 早期就被吸收，后期 block 主要是 visual refinement。这暗示未来 action module 可以做成 shallow injection。

6. **Real-time interactive world model 的工程方程**：FPS = (denoising steps × DiT forward + VAE decode + cache overhead)⁻¹。要冲 25 FPS，每一步都得砍——distillation 砍 step，cache 砍 VAE，halving 砍 DiT forward，三个一起上才能到。

---

## 9. 和相关工作放在一起看

- **Oasis** (Decart, 2024)：第一个开源 real-time Minecraft interactive world model，但 quality 长程崩。Matrix-Game 2.0 在 quality 上碾压，工程上更鲁棒。
- **YUME** (2025)：wild scene 强但慢。
- **Genie 2** (DeepMind, 2024)：闭源，scale 大但没开源，没 action-level control。
- **Diffusion Forcing** (Chen et al., 2024)：AR + diffusion，但没解决 train-test gap，长视频会漂。
- **CausVid** (Yin et al., 2025)：bidirectional → causal distillation 的方法论，Matrix-Game 2.0 借鉴了思路。
- **Self-Forcing** (Huang et al., 2025)：核心方法基础。
- **DMD** (Yin et al., CVPR 2024)：one-step / few-step distillation 的 distribution matching 范式。
- **Wan 2.1**：底座 I2V foundation。

---

## 10. References & 进一步阅读

- Matrix-Game 2.0 项目主页（待开源）: https://github.com/Skywork/Matrix-Game
- Self-Forcing paper: https://arxiv.org/abs/2506.08009
- DMD (Distribution Matching Distillation): https://arxiv.org/abs/2310.05910  
- CausVid: https://arxiv.org/abs/2506.04498
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- Oasis: https://oasis-model.github.io/
- YUME: https://arxiv.org/abs/2507.17744
- Wan 2.1: https://arxiv.org/abs/2503.20314
- Matrix-Game 1.0: https://arxiv.org/abs/2506.18701
- Genie 2 (DeepMind): https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- V-JEPA (LeCun): https://arxiv.org/abs/2304.08471
- World Labs spatial intelligence: https://www.worldlabs.ai/blog
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864
- LTX-Video (real-time video gen baseline): https://arxiv.org/abs/2501.00103
- PPO: https://arxiv.org/abs/1707.06347

---

如果你接下来想深挖某一块，我建议三个方向：(a) Self-Forcing 的具体 loss 形式和 DMD 的 gradient 推导，这是 paper 没展开但最 math-heavy 的部分；(b) action module 的 cross-attention vs concat 设计的 ablation，paper 没做；(c) KV-cache size vs error correction 的更细致实验，这个 finding 太有意思了值得单独写一篇。
