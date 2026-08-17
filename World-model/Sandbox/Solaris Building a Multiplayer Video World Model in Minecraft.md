---
source_pdf: Solaris Building a Multiplayer Video World Model in Minecraft.pdf
paper_sha256: 602e78b6b2385c887ff0e57d835cf4824e59627ab9bfa5f70617c286cf606b7e
processed_at: '2026-08-12T08:42:58-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Solaris 用人话说

## 一句话版本

以前 AI 生成的"游戏世界视频"只能模拟一个人在玩,Solaris 第一次让 AI 能同时模拟两个玩家在同一个世界里互相看到对方、互相影响——这在 Minecraft 里实现了。

---

## 为什么要做这件事?

想象你在玩 Minecraft,你和朋友一起挖矿。你挖一个洞,你朋友的屏幕上必须同时看到"你在挖洞"。这件事人类觉得很自然,但 AI model 之前根本做不到。

以前的 video world model(Oasis、GameNGen 那些)只管自己一个视角,生成的画面像"只有我一个人在的世界"。但真实世界从来都是多人共享的——一个 player 放的 block、打的怪、走的路,必须立刻在另一个 player 的视野里同步出现。这种"两个视角看的是同一个 3D 世界"的约束,就是 Solaris 要解决的核心难题。

---

## 数据采集这件事为什么这么麻烦?

要训 multiplayer 模型,得有 multiplayer 数据。但 Minecraft 现有的 AI 框架都有硬伤:

- Malmo / MineRL / MineDojo:能出画面,但 action space 太底层,没训过的 agent 只会瞎跳,数据像猴子乱按键盘,没法用
- Voyager / Mineflayer:能编程控制 bot 干正经事,但不输出画面(纯文本模式)
- 没有一个框架同时满足 "能编程控制" + "多人" + "有画面"

所以团队从零造了 SolarisEngine。核心 hack 很妙:每个 "logical player" 其实是 **两个 bot 配对**——一个 Controller bot(Mineflayer,负责执行 action 和记录),一个 Camera bot(真正的 Minecraft 客户端,负责渲染画面),中间用一个 server 插件让 Camera 实时 mirror Controller 的所有动作。这样就用 Mineflayer 的控制力 + 真实 Minecraft 的渲染力,凑出一个能采集 multiplayer gameplay 的系统。

整个系统跑 Docker,卡死了自动 abort 跳过,bot 每次随机 teleport 到新地形。最终采了 1264 万帧,4 大类 14 种 episode。

---

## 模型架构到底改了什么?

基础是一个已有的 single-player video DiT(Matrix Game 2.0)。要改成 multiplayer,核心问题是怎么让两个 player 的信息互相交换。

两个 candidate 方案:

**方案 A:channel concat**(Multiverse 用的)
把 player 1 和 player 2 的画面沿 channel 维拼一起。问题:这默认了两个画面在空间上 pixel-by-pixel 对齐。但 3D 世界里两个 player 站在不同位置看同一个东西,视角根本对不齐。结果就是 Building 任务直接 0 分——模型根本学不会两个人看着同一个 structure 是什么意思。

**方案 B:sequence interleave**(Solaris 用的)
两个 player 的 token 沿 sequence 维交错排,丢进一个共享的 self-attention 层。attention 自己决定哪个 token 该看哪个。这样保留了每个 player 内部的 spatial 结构,又能跨 player 交换信息。每个 player 加一个 learned ID embedding 让模型知道"这是 player 1 还是 player 2 的 token"。这套设计 Building 任务 VLM 20.8 分,完爆 channel concat。

直觉上就是:别强行告诉模型 "这俩画面应该 pixel 对齐",而是让 attention 自己学 "这俩视角应该在哪里对应"。

---

## 训练为什么分 4 个 stage?

直接从头训 multiplayer causal model 根本收敛不了,得一步步来:

**Stage 1: Single-player pretrain**
拿 Matrix Game 2.0 当起点,在 VPT(2000+ 小时真实人类 Minecraft gameplay)上 finetune,把 action space 从只有 WASD 扩展到完整的 Minecraft action。这步给模型灌进 Minecraft 的视觉先验和 action grounding。Table 2 的 ablation 显示,skip 这步 Building 任务直接 0 分——没有海量单 player 数据打底,模型连"放 block 长什么样"都学不会。

**Stage 2: Bidirectional Multiplayer**
加 multiplayer self-attention,在多人数据上用全序列 diffusion 训。所有 frame 一起加噪一起去噪。这个 checkpoint 当后续 Self-Forcing 的 teacher。

**Stage 3: Causal Multiplayer**
从 Stage 2 中途分叉出来,加 causal mask(滑动窗口 attention,window 6 latent frame)。用 Diffusion Forcing(per-frame 独立 noise level)训,让模型能 autoregressive 生成。这个当 Self-Forcing 的 generator 初始化。

**简化点**:原 CausVid 流程要先 ODE regression + DMD distillation 再 Self-Forcing,论文发现直接 causal finetune 就够了,反而更好(Table 3 里 ODE Reg 的 Building FID 95.7 vs Causal FT 87.4)。

**Stage 4: Self-Forcing**
解决 train-test gap:训练时 context 是 ground truth,推理时 context 是自己生成的,质量越滚越差。Self-Forcing 让 student 在自己生成的 frame 上被监督。

---

## Checkpointed Self-Forcing 是什么?

这是论文最硬核的技术创新。问题起源:团队想让 teacher 的 context 比 student 长(long-context teacher 监督更强),但 student 生成用 sliding window。Naive 实现下,每个滑动窗口的中间计算图都得保留以备 backprop,内存 $O(L_t \cdot L_s)$ 爆炸。

解决思路就是 gradient checkpointing 的类比:

1. **先无梯度 rollout**:跑完整个滑动窗口生成,只 cache 每帧的 clean 估计和 noisy 状态,全程 stop_gradient
2. **再有梯度重算**:把这些 clean 和新采样的 noisy frame 拼一起(length 翻倍),用一个精心设计的 Teacher Forcing Mask 在一次 parallel forward 里重做所有 frame 的最后一步 denoising

Teacher Forcing Mask 的逻辑:noisy frame 只能看自己之前的 clean frame(sliding window 大小 $L_s$),clean frame 之间 causal。

这样内存从 $O(L_t \cdot L_s)$ 降到 $O(L_t)$。而且因为省了内存,原本必须 stop-gradient 的 KV cache 现在可以 backprop 了——这个额外改动让 Building FID 从 87.4 改善到 83.6,Consistency VLM 从 70.8 提到 71.4。

代价是 Movement action following 略降(78.6 → 68.2),因为 KV backprop 让 history representation 自由度变大,对 short-range precise action mapping 有干扰。但对需要 long-range consistency 的任务反而有利。

---

## Evaluation 怎么测的?

5 个维度,核心是用 VLM 当 judge 问 yes/no:

- **Movement**:一个 player 走,另一个看,VLM 判断观察者画面里对方往哪走
- **Grounding**:一个 player 转身看不见对方再转回,VLM 判断转回后是否看到对方
- **Memory**:两人都转身再转回,VLM 判断是否互相看到
- **Building**:一个 player 搭结构另一个看,VLM 判断是否看到 6 格外的结构
- **Consistency**:两人同时转 90 度到同侧或异侧,VLM 判断两视角是否看到相同景色

这比只看 FID 信息量大得多,FID 只能说"画面像不像",说不出"模型有没有理解 multiplayer 语义"。

---

## 几个反直觉的发现

1. **Pretrain 决定生死**:12.64M frames 的 multiplayer 数据其实不够模型学会 Minecraft 的 visual prior。VPT 那几千小时单 player 人类数据是命脉,没有它 Building 任务直接 0 分。直觉是:multiplayer 数据量太小,只够学 cross-view consistency,学不了基础 visual + action grounding。

2. **ODE Regression 初始化有害**:CausVid 那套 ODE regression + DMD distillation 的初始化,在这个 setting 下反而拖后腿。可能因为 multiplayer action 分布复杂,few-step distillation 把信息压掉了。直接 causal finetune 反而最好。

3. **Few-step 能力可以边训边学**:原 Self-Forcing 假设 generator 已经是 few-step model 才能开始训,论文发现 few-step 能力可以在 Self-Forcing 过程中同步学到,不用单独的 distillation stage。

4. **KV backprop 是 trade-off**:visual quality 普遍提升,但 pure action following 略降。这说明让 history representation 自由优化是一把双刃剑。

5. **Frame concat 在 Movement 上反而最高**:77.1 vs Solaris 68.2。但 qualitative 看 frame concat 会 action hallucination(no-op 时瞎动)。数字好看不代表真的好。

---

## 这篇论文真正的价值

Solaris 不只是一个 Minecraft model,它建立了一套完整的方法论:

1. **数据引擎设计范式**:Controller/Camera bot 分离 + Docker 编排,可以迁移到任何 multiplayer game
2. **Multiplayer DiT 改造范式**:sequence interleave + per-player ID embedding,对 multi-camera autonomous driving、多机器人协作都有启发
3. **Memory-efficient Self-Forcing**:Checkpointed 思路对任何 sliding-window autoregressive + backprop 的场景都有用,包括 model-based RL
4. **VLM-as-judge 评估范式**:把多能力评估降到 yes/no question,可自动化、可复现

Limitations 也很诚实:数据全是 bot 合成的有 distribution gap;player 离开对方视野后共享 context 丢失(没有 persistent memory,这是 video world model 的根本局限);只训了 2 player。

代码、数据、模型全开源,这对社区是实打实的贡献。未来方向也很清晰:更多 player、external memory、real human data adaptation、把 world model 当 environment 训 multi-agent policy。

---

参考链接:
- 项目主页:https://solaris-wm.github.io/
- Engine code:https://github.com/solaris-wm/solaris-engine
- Model code:https://github.com/solaris-wm/solaris
- Datasets & Models:https://huggingface.co/collections/nyu-visionx/solaris-data

---

# Solaris: 多人视频世界模型深度解析

## 一、论文核心定位与动机

Solaris 由 NYU Saining Xie 团队提出,定位是 **第一个真正意义上的 multiplayer video world model**。在 Minecraft 中,模型需要同时模拟两个玩家视角的观测视频,且这两个视角必须在时间和空间上保持一致。

传统 video world model(如 Oasis、GameGen-X、GameNGen)只建模 single agent 视角,核心假设是 "world state 从单一观察者的 POV 就足以重建"。但在真实多智能体环境中,A 在挖洞,B 站在旁边看,挖洞这件事必须同时反映在 A 的第一人称(看到自己挖)和 B 的第三人称(看到 A 在挖)中。这种 **cross-view consistency** 是 single-agent model 根本无法表达的结构性问题。

论文选 Minecraft 作为 testbed 的理由很扎实:
- **Unbounded 3D world**:procedural terrain 强制模型学 perspective consistency、occlusion handling、spatial memory
- **Dynamic malleable environment**:block place/break 产生持久状态变化,考验 model 对时间累积变化的记忆
- **Environmental stochasticity**:mob、weather 等随机性让模型必须 disentangle agent-caused 和 environment-caused changes
- **Open-ended complexity**:crafting、building、mining 提供近乎无限的复杂度

这是一个比 Multiverse(Gran Turismo 4 单赛道、U-Net 架构)复杂得多的 setting。

参考链接:
- 项目主页:https://solaris-wm.github.io/
- Model code:https://github.com/solaris-wm/solaris
- Engine code:https://github.com/solaris-wm/solaris-engine

---

## 二、SolarisEngine:数据采集系统

### 2.1 为什么需要从零造一个 engine

现有 Minecraft AI 框架的对照表很能说明问题:

| Framework | Controllability | Multiplayer | Graphics |
|-----------|-----------------|-------------|----------|
| Malmo | ✗ | ✓ | ✓ |
| MineRL | ✗ | ✗ | ✓ |
| MineDojo | ✗ | ✗ | ✓ |
| Voyager | ✓ | ✗ | ✗ |
| Mineflayer | ✓ | ✓ | ✗ |
| **SolarisEngine** | **✓** | **✓** | **✓** |

关键矛盾在于:
- Malmo/MineRL/MineDojo 都是 RL 框架,action space 是 low-level 的,没训练过的 agent 只会乱动,数据太 chaotic 不适合 world model 训练。如果用 RL 训练 agent,数据会被 reward shaping 偏置,失去 realistic gameplay 的多样性。
- Mineflayer 提供 high-level pathfinding、block placement 等 primitives,但有两个硬伤:(1) 没有图形渲染能力;(2) 不支持 multiplayer coordination。
- Voyager 用 Mineflayer 但是纯文本模式,没有 visual output。

### 2.2 架构设计:Controller Bot + Camera Bot 分离

这是整个 engine 最精巧的设计。Mineflayer 本身不能渲染,论文用一个 **"双 bot 对偶"** 的 trick 解决:

- **Controller bot**:运行 Mineflayer,执行 episode 代码,记录 low-level actions(WASD、camera、attack 等),但不渲染画面。
- **Camera bot**:运行官方 Minecraft Java client(headless 模式,GPU 加速渲染),通过一个 custom server-side plugin **mirror** Controller 的 state 和 actions,实时同步 animations。
- 两个 bot 通过 timestamp 后处理对齐 video 和 action。

一个 "logical player" = Controller + Camera 两个进程的组合。当前实现 2 player,理论上可扩展到任意数量。

### 2.3 Docker 编排与容错

整个系统跑在 Docker Compose 上,Controller / Camera / Server 各自一个 container。Python 脚本并行启动多个 Compose worker 做规模化采集。每个 episode 开始时 bot 随机 teleport 到新位置以多样化地形。

**容错机制** 是工程上的关键:由于 Minecraft 的随机性,episode 经常卡死或报错,系统通过失败检测、跨 bot 通知、当前 episode 全部 abort、刷新状态进入下一个 episode 的机制,实现无人值守连续采集。

### 2.4 数据集统计

最终采集到 **12.64 M frames(每 player 6.32 M)**,9240 episodes,4 大类(building/combat/movement/mining)14 种 episode type。录制 20 fps,大多数 episode 128-512 帧(6.4-25.6 秒)。

Action space 接近 VPT 的 gold standard,只缺 inventory 打开/关闭 和 raw mouse movement(只记录 relative pitch/yaw)。具体 action 列表(共 23 个):
- Sustained(持续型):forward, back, left, right, jump, sprint, sneak, camera(vec2f), mine
- Once(瞬时型):attack, use, mount, dismount, place_block, place_entity, hotbar.1-9

这里有一个细节值得注意:Pathfinder 插件以 178 度/秒的最大速度转视角,而手动 camera action 速度更慢,导致 mouse action distribution 严重偏向 fast camera moves(见 Fig 13)。这是合成数据的 inherent bias。

---

## 三、模型架构

### 3.1 数学设定

传统 single-agent world model 在 latent frame $\mathbf{x}_{SP} \in \mathbb{R}^{H \times W \times C}$ 上做扩散,论文把它推广到 multi-agent:

- 联合状态 $\mathbf{x}^t = \{x_1^t, \ldots, x_P^t\}$:t 时刻 P 个 agent 的所有视角
- 联合 action $\mathbf{a}^t = \{a_1^t, \ldots, a_P^t\}$
- 序列 $\mathbf{x} := \mathbf{x}^{1:T}$,shape 为 `(B, P, T, H, W, C)`
- action 序列 $\mathbf{a} := \mathbf{a}^{1:T}$,shape 为 `(B, P, T, D)`

建模概率:
$$p_\theta(\mathbf{x}) = \prod_{t=1}^{T} p_\theta(\mathbf{x}^t \mid \mathbf{x}^{<t}, \mathbf{a}^{<t})$$

含义是给定所有过去 frame 和所有 agent 的 action,预测下一时刻所有 agent 的联合观测。这是一个 **joint distribution** 而非 marginal,所以两个视角是 coupled 的,不是独立生成。

### 3.2 Flow Matching 训练目标

$$\mathcal{L}_{\theta} = \mathbb{E}_{\mathbf{x}, \mathbf{a}, \sigma, \epsilon}\left[\|\nu_\theta(\mathbf{x}_\sigma, \sigma, \mathbf{a}) - (\epsilon - \mathbf{x})\|_2^2\right]$$

变量解释:
- $\nu_\theta$: velocity field 预测网络(DiT)
- $\mathbf{x}_\sigma = (1-\sigma)\mathbf{x} + \sigma\epsilon$: forward 加噪过程,$\sigma$ 从 0(完全干净)到 1(完全噪声)
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:标准高斯噪声
- $\mathbf{a}$:action conditioning,通过 cross-attention 注入

**关键差异** 在 noise schedule:
- Bidirectional model:$\sigma \sim \mathcal{U}(0,1)$,$\boldsymbol{\sigma} = \sigma \cdot \mathbf{1}_{P \times T}$,所有 player、所有 frame 共享同一 noise level,做 **joint diffusion**
- Causal model:采用 Diffusion Forcing,$\boldsymbol{\sigma} \in [0,1]^{P \times T}$,每个 $\sigma_{p,t} \sim \mathcal{U}(0,1)$ 独立采样,这是 enabling autoregressive generation 的核心

### 3.3 DiT Block 改动

基础是 **Matrix Game 2.0**(一个 single-player controllable video DiT,在 Minecraft 等多游戏上预训练过)。改动有两处:

**(1) Action space 扩展**

Matrix Game 2.0 原始 action 只有 camera + WASD,论文扩到完整 MineRL action space,做法是增大 keyboard action module 的 input dim 并重新初始化权重。每个 player 独立跑 action module,通过 einops-style rearrange 把 player 维 fold 进 batch:`rearrange("B P T D -> (B P) T D")`。

**(2) Multiplayer Self-Attention**

这是整个架构的灵魂。如图 5 所示,所有 player 的 tokens 沿 sequence 维度 **interleave**(交错),然后通过一个共享的 self-attention 层,让不同 player 的 token 互相 attend。

具体细节:
- 对每个 player 的 token 独立应用 3D RoPE(Rotary Position Embedding),保持单 player 内部的时空结构
- 在每个 Multiplayer Self-Attention 层开始时,给每个 player 的 token 加上 **learned player ID embedding**,让模型能区分这是谁谁谁
- Cross-attention(第一帧 conditioning)保持不变,per-player 独立应用

这种 interleaving 设计 比 Multiverse 的 **channel concatenation**(把两个 player 的 frame 沿 channel 拼一起)更优,因为:
- channel concat 把每个 spatial location 强制耦合,丢掉了 spatial locality
- sequence interleaving 让 attention 自己学哪些 token 该 attend,保留了 per-player 的 spatial reasoning

---

## 四、4 阶段训练 Pipeline

这是论文最值得仔细研读的部分,整体逻辑是从易到难、从 bidirectional 到 causal、从短到长。

### Stage 1: Bidirectional Single-Player(120K steps)

初始化自 Matrix Game 2.0,在 **VPT dataset**(2000+ 小时人类 gameplay)上 finetune,把 action space 从 camera+WASD 扩到完整 Minecraft action。

为什么这步关键:VPT 是真实人类数据,涵盖丰富的 building、crafting、combat 行为,给模型一个 realistic gameplay 的 prior。Table 2 显示 without pretrain 的版本在 Building/Grounding/Memory 上严重退化(Building VLM 直接 0.0)。直觉是:multiplayer 数据有限(12M frames),但 action distribution 复杂,直接训根本学不到 Minecraft 的 visual prior;single-player pretraining 用海量 VPT 灌满 visual + action grounding,再迁移到 multiplayer 上只需要学 cross-view consistency。

context length 33 帧,3D VAE 全程 frozen。

### Stage 2: Bidirectional Multiplayer(120K steps)

在 multiplayer 数据集上用全序列 diffusion 训练(所有 frame 联合 noise/denoise),架构加上 multiplayer self-attention。这个 checkpoint **作为 Self-Forcing 的 teacher**。

### Stage 3: Causal Multiplayer(60K steps)

从 Stage 2 的 60K 中间 checkpoint 分叉出来(Stage 2 继续训到 120K 作为 teacher),用 **Diffusion Forcing + causal mask** 训练。

Causal mask 用 sliding window attention,window size 6 latent frame(24 real frame),这也是 inference 时 KV cache 的最大 size。这个 checkpoint 作为 Self-Forcing 的 **generator initialization**。

**重要简化**:原 CausVid pipeline 要先做 ODE regression 初始化,再 DMD few-step distillation,然后才 Self-Forcing。论文发现 **直接 causal finetune 就够了**(见 Table 3 ablation)。

### Stage 4: Self-Forcing + Checkpointed Self-Forcing

Self-Forcing 解决 train-test gap:训练时 teacher forcing(ground truth context),inference 时模型只能用自己生成的 frame 作 context,这种 mismatch 导致 quality 滚雪球式退化。Self-Forcing 让 student 在自己生成的样本上被监督,bridge 这个 gap。

---

## 五、Checkpointed Self-Forcing:核心技术创新

### 5.1 问题:naive Self-Forcing 的内存爆炸

原 Self-Forcing 要求 student context length $L_s$ == teacher context length $L_t$。论文想用 **long-context teacher**(更有信息量)监督 **short-context student**(更省内存、更长 horizon),这就需要 sliding window 生成。

但 naive 实现:每个 autoregressive step 生成一个新 window,backprop 时所有 window 都得保存在计算图里。总长度 $L_t$ 步,每步 $L_s$ frame,内存复杂度 $O(L_t \cdot L_s)$,很快 OOM。

### 5.2 核心思路:Gradient Checkpointing 的类比

借鉴梯度检查点的思想:把 rollout 和 backprop 解耦。

**Step 1 — Autoregressive rollout(无梯度)**
跑完整个 sliding window rollout,生成所有 clean estimates $\hat{\mathbf{x}}_0^{1:N}$ 和对应的 noisy transition states $\mathbf{x}_\sigma^{1:N}$,但全程 `stop_gradient`。只 cache 这些中间状态,不保留中间 attention/KV 计算图。

**Step 2 — Recompute(有梯度,并行)**
把 $\hat{\mathbf{x}}_0^{1:N}$(clean)和 $\mathbf{x}_{\sigma'}^{1:N}$(newly sampled noisy)沿 sequence 拼一起(length 翻倍),然后用一个 **Teacher Forcing Mask** 在一次 parallel forward pass 里重做所有 frame 的最终 denoising step,这次是带梯度的。

### 5.3 Teacher Forcing Mask 的设计

这个 mask 的逻辑(见 Algorithm 2 伪代码):
- 输入 sequence 长度 $2 \cdot L_t \cdot \text{tokens\_per\_frame}$,前半是 clean frames,后半是 noisy frames
- **noisy query** 可以 attend:
  - 同 frame 的 noisy key(self-attention within final denoise step)
  - 更早 frame 的 clean key(条件依赖,只能看过去)
- **clean query** 只能 attend:
  - 更早或同 frame 的 clean key(因果)
- **sliding window 约束**:$kv\_frame > q\_frame - L_s$,即每个 query 只看自己前 $L_s$ 个 frame

这样就把 sliding-window 的 sequential rolling cache 操作转成一个 parallel 矩阵操作,内存从 $O(L_t \cdot L_s)$ 降到 $O(L_t)$。

### 5.4 关键额外收益:KV backprop

原 Self-Forcing 在 KV cache 上加 stop-gradient(因为内存爆),论文因为有内存节省,可以 **允许梯度流过 KV representation**(Algorithm 1 line 29)。

Table 3 的 ablation 非常 clean:
- Causal FT + no Pre-DMD + **KV-BP=True**:Building VLM 20.8, FID 83.6(最好)
- Causal FT + no Pre-DMD + KV-BP=False:Building VLM 15.6, FID 87.4

KV backprop 在 Building 和 Consistency 上明显改善,但 Movement 的 action following 略降(从 78.6 降到 68.2)。直觉:KV backprop 让模型能调整 history 的 representation 以更好地预测未来,这对需要 long-range consistency 的 Building 任务有利,但对纯 action-to-motion 的 Movement 反而因为 representation 自由度变大而轻微下降。

---

## 六、Evaluation Benchmark

论文提出 5 个维度的评估,用 **VLM-as-judge**(看 frame 问 yes/no 问题)加 FID:

### 6.1 Movement
一个 bot 动(WASD + camera),另一个观察。VLM 判断观察者视角中 agent 的位置变化(closer/farther/left/right/no motion)。GT accuracy 100%。

### 6.2 Grounding
两个 agent 相对站立,一个转身看不见另一个,再转回来。VLM 判断转身后是否看到对方。测试模型对"自身位置相对世界的关系"的 grounding。GT 96.88%。

### 6.3 Memory
两个 agent 都转身、再转回。VLM 判断是否互相看到。测试 cross-time memory。GT 92.71%。

### 6.4 Building
一个 bot 建简单结构(square 或 strip),另一个观察。VLM 判断观察者是否看到 6 blocks 距离外的 structure。GT 98.96%。

### 6.5 Consistency
Normal 世界,两 agent 面对面,同时转 90° 到同侧或异侧。VLM 判断两个视角是否看到 same scenery(同侧应该相同,异侧应该不同)。GT 同侧 98.96%、异侧 93.75%。这是测 **cross-view geometric consistency** 的硬指标。

VLM 用同一 prompt 跑 3 次 estimate std。episode-level 准确(所有 query 点都对才算对)。

---

## 七、实验结果深度解读

### 7.1 Architecture 对比(Table 2)

| Method | Movement VLM | Grounding VLM | Memory VLM | Building VLM | Consistency VLM |
|--------|--------------|---------------|-----------|--------------|-----------------|
| Frame concat (Multiverse) | **77.1** | 53.1 | 37.5 | 0.0 | 49.5 |
| Solaris w/o pretrain | 69.3 | 29.2 | 18.8 | 0.0 | 49.5 |
| **Solaris** | 68.2 | **62.5** | **37.5** | **20.8** | **71.4** |

几个关键 insight:
1. **Frame concat 在 Building 上彻底失败(0.0)**:channel concatenation 假设两视角 pixel-wise 对齐,但 3D Minecraft 中两个玩家站在不同位置看同一个 building,视角根本无法 channel 对齐。FID 103.2 极差。
2. **Pretrain 是 Building 任务的命脉**:w/o pretrain 在 Building 也是 0.0,说明没有 VPT 的 prior,模型连 basic block structure 都学不会,更别说 multiplayer consistency。
3. **Frame concat 在 Movement 上反而最高**(77.1):直觉是 movement 任务两个视角 spatial locality 高,channel concat 反而保留了 spatial correspondence。但论文指出 qualitative 看 frame concat 会有 action hallucination(no-op 时模型瞎动)。
4. **Solaris 在 Consistency 上 71.4 vs 49.5**:cross-view consistency 是 multiplayer world model 的核心,Solaris 几乎碾压。

### 7.2 Self-Forcing Ablation(Table 3)

最有信息量的表:

| Init | Pre-DMD | KV-BP | Building VLM | Building FID | Consistency VLM |
|------|---------|-------|--------------|--------------|------------------|
| ODE Reg | ✗ | ✓ | 3.1 | 95.7 | 49.0 |
| Causal FT | ✓ | ✓ | 8.3 | 90.5 | 55.2 |
| Causal FT | ✗ | ✗ | 15.6 | 87.4 | 70.8 |
| Causal FT | ✗ | ✓ | **20.8** | **83.6** | 71.4 |

发现:
1. **ODE Regression 初始化极差**:Building VLM 3.1,FID 95.7。说明 CausVid 那套 ODE regression + DMD distillation 的初始化在这个 setting 下反而有害,可能是因为 Multiplayer action distribution 太复杂,few-step distillation 把 single-step 信息压缩丢掉了。
2. **Pre-DMD(在 Self-Forcing 之前先做 few-step distillation)无效**:Causal FT + Pre-DMD 在 Building FID 90.5 vs 没有 87.4。原 Self-Forcing 假设 generator 是 few-step model,论文发现 few-step 能力可以在 Self-Forcing 中同步学到。
3. **KV-BP 普遍提升 visual quality**:FID 在所有任务上都改善,但 action following 在 Movement 上有 trade-off。

### 7.3 训练超参(Table 6)

- 所有 stage LR 1e-4(Self-Forcing generator 3e-6,critic 3e-7)
- Adam β1=0.9, β2=0.95, weight decay 0
- Batch 32-64
- Self-Forcing generator 只训 240 步!critic 1200 步!
- 用 Google v5p TPU(128 / 64 cores)

Self-Forcing 步数极少(240 步)说明这是高 LR 微调,主要靠 pretrain 阶段打下的基础。

---

## 八、Qualitative Capabilities(Fig 10)

论文展示了几个 impressive 的 emergent capability:
1. **Inventory counter 同步**:一个 player 放置 block 后,inventory 数量在两个视角里同步更新,说明模型学到 GUI state 跟 player action 的因果
2. **全局天气同步**:rain 在两个视角同时开始,说明 global state 被正确建模
3. **Mining 动画 + torch placement**:per-player 的 item state 同步,active item 切换正确反映
4. **复杂地形 PvP**:两个玩家在 terrain 上战斗,动作连贯

---

## 九、Limitations 与 Future Work

论文坦诚列出几个核心 limitation:

1. **Synthetic data bias**:所有数据来自 pre-programmed bot,action 和 visual distribution 都和真实人类有 gap。未来的方向是更好地 leverage 单 player 人类数据(VPT)做 distribution alignment。

2. **No persistent memory**:player 离开对方视野后,共享 context 丢失,trajectory 开始 diverge。这是 video world model 的根本局限:没有 explicit persistent state,全靠 latent frame 隐式编码。这点 Genie 3、WorldMem 等工作尝试用 external memory module 解决。

3. **只有 2 player**:当前 framework 支持任意 N player,但实际只训了 2。扩展到更多 player 时,attention 复杂度 $O((P \cdot N_{tok})^2)$ 会爆炸,可能需要 sparse attention 或 hierarchical 设计。

---

## 十、核心 Intuition 总结

让我把整篇论文的"思维链"梳理一遍,这是 build intuition 最关键的部分:

**1. 为什么 multiplayer video world model 难?**
因为 joint state $\mathbf{x}^t$ 是 P 个视角的耦合,模型必须同时学:
- 单视角内的 temporal coherence(传统 video model 已解决)
- 跨视角的 spatial consistency(完全新的问题)
- 跨视角的 action grounding(A 的 action 必须在 B 的视角产生正确 visual effect)
- 3D 几何一致性(两个视角看到的 world 必须是同一个 3D world 的不同投影)

**2. 为什么需要 SolarisEngine?**
合成数据是 multiplayer world model 的 bottleneck。真实 multiplayer 数据采集成本极高,而 RL 训练的 agent 数据被 reward 偏置。Mineflayer + Camera bot 的分离设计是工程上的优雅解法:high-level controllability + real graphics。

**3. 为什么 sequence interleaving 比 channel concat 好?**
Channel concat 假设两视角 pixel-wise spatial 对齐,在 3D 不变视角下完全不成立。Sequence interleaving 让 attention 机制自己学跨视角 correspondence,保留了 per-player 的 spatial locality,这是为什么 Building 任务上 Solaris 20.8 vs Multiverse 0.0 的根本原因。

**4. 为什么需要 staged training?**
直接从 scratch 训 multiplayer causal model 几乎不可能收敛。Single-player pretrain 用海量 VPT 数据灌 visual + action prior;bidirectional multiplayer 加 cross-view attention 但保留全序列建模的简单性;causal finetune 转 autoregressive;Self-Forcing bridge train-test gap。每一步只解决一个问题。

**5. 为什么 Checkpointed Self-Forcing 重要?**
Long-context teacher 提供更强的监督信号,但 naive sliding window Self-Forcing 内存爆。Gradient checkpointing 的思想——前向重算换内存——被巧妙应用到 autoregressive rollout 上。一次性 parallel recompute 让内存从 $O(L_t \cdot L_s)$ 降到 $O(L_t)$,还能额外打开 KV backprop。

**6. 为什么 KV backprop 有 trade-off?**
原 Self-Forcing 在 KV 上 stop-grad 是因为内存,不是因为算法上不对。允许 KV backprop 让 history representation 自由优化以利于未来预测,这对需要 long-range consistency 的 Building 任务有利,但 action following 这种 short-range precise mapping 反而被 representation 的自由度干扰。

---

## 十一、相关工作的延伸阅读

为构建完整 intuition,推荐 follow-up 阅读:

- **Self-Forcing**(Huang et al. 2025):https://arxiv.org/abs/2506.08009 — 原始 Self-Forcing paper,Solaris 的直接基础
- **Diffusion Forcing**(Chen et al. NeurIPS 2025):https://arxiv.org/abs/2507.01392 — per-frame independent noise level 的核心思想
- **CausVid**(Yin et al. CVPR 2025):https://arxiv.org/abs/2505.16847 — bidirectional to causal 的转换,Solaris 简化了它的初始化
- **Matrix Game 2.0**:https://arxiv.org/abs/2508.13009 — Solaris 的 backbone
- **VPT**(Baker et al. NeurIPS 2022):https://arxiv.org/abs/2206.11795 — single-player pretrain 数据源
- **Multiverse**(Enigma 2025):唯一前作 multiplayer world model,但限制大
- **Voyager**(Wang et al. TMLR 2024):https://arxiv.org/abs/2305.16291 — Mineflayer 高级用法参考
- **RELIC**(Hong et al. 2025):https://arxiv.org/abs/2512.04040 — 并行工作,也研究 long-context teacher 的 memory-efficient 实现,但用 multiple rolling passes,Solaris 用 single parallel pass 更高效
- **WorldMem**(NeurIPS 2025):https://arxiv.org/abs/2510.18805 — external memory 解决 persistent memory 问题
- **Genie 3**(Google DeepMind 2025):https://deepmind.google/discover/blog/genie-3/ — DeepMind 的最新 world model 进展
- **Mineflayer**:https://github.com/PrismarineJS/mineflayer — SolarisEngine 的基础

---

## 十二、研究脉络与未来方向

Solaris 的意义远不止一个 Minecraft world model。它建立了一套 **multiplayer world model 的完整 recipe**:

1. **Data engine 设计范式**:Controller/Camera 分离 + Docker 编排 + primitive library,这套架构可以迁移到任何 multiplayer game(如 Roblox、Unreal Engine game)
2. **Multiplayer DiT 改造范式**:sequence interleaving + per-player ID embedding,这个思路对 multi-camera autonomous driving、多机器人协作都适用
3. **Memory-efficient Self-Forcing**:Checkpointed Self-Forcing 不只对 video generation 有用,对任何 sliding-window autoregressive + backprop 的场景(RL 的 world model training、long-horizon planning)都有价值
4. **VLM-as-judge 评估范式**:把 multi-capability evaluation 降成 yes/no question,可复现、可自动化,比纯 FID 信息量大得多

未来值得探索的方向:
- **N > 2 player**:需要 sparse / hierarchical attention 处理 $O(P^2)$ 复杂度
- **Persistent memory**:结合 external memory bank(如 WorldMem 的 memory tokens)
- **Real human data distillation**:用少量真实 multiplayer 数据做 domain adaptation
- **Downstream policy learning**:把 Solaris 当 environment 训 multi-agent policy,这是 world model 真正的价值所在
- **Cross-game generalization**:SolarisEngine 能否扩展到非 Minecraft 游戏

论文代码、数据集、模型都开源了,这对社区是巨大贡献:
- Models:https://huggingface.co/collections/nyu-visionx/solaris-models
- Datasets:https://huggingface.co/collections/nyu-visionx/solaris-data

---

## 总结

Solaris 的核心贡献是把 video world model 从 single-agent 推进到真正的 multi-agent setting,通过精心设计的数据引擎、架构改造、staged training 和 memory-efficient Self-Forcing,在 Minecraft 这个高复杂度 open-world 上验证了可行性。它不只是一个 model,更是一套可复用的方法论,为 multi-agent embodied AI、synthetic data generation、model-based RL 打开了一扇新的门。Limitations(persistent memory、synthetic bias)明确指出了下一代工作的方向。
