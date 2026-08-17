---
source_pdf: FAVLA AForce-Adaptive Fast–Slow VLA model for Contact-Rich Robotic Manipulation.pdf
paper_sha256: 898bd1e3698361c0566f58073d02a964ef50c518ae8d3a1b2ed3a7c0f7c32a85
processed_at: '2026-08-04T08:03:59-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话聊聊 FAVLA 到底在干嘛

Hi Andrej, 咱们抛开那些公式，用人话把这篇 paper 的逻辑捋一遍。

这事儿本质上就是教机器人怎么“察言观色”地干活，特别是在那种需要精细接触的活儿上，比如插 USB 或者装齿轮。

## 1. 痛点在哪：现在的机器人有点“迟钝”

现在主流的 VLA 模型（比如 π₀）干活是这样的：它的大脑（VLM）像个老教授，慢悠悠地看一眼图、读一下指令，然后给机械臂扔一沓动作指令（action chunk）说“你按这步骤做”。机械臂就闷头执行，中间基本不抬头看路。

如果只是去桌子上拿个苹果，这没问题。但如果是把 USB 插进接口，麻烦就大了。机械臂在执行那沓指令的半路上，如果稍微偏了一点，怼到接口边缘了，这时候会产生瞬间的力突变（force spike）。可是老教授还在慢悠悠思考下一步呢，机械臂也没有实时看力传感器的习惯，结果就是硬怼，要么把 USB 插头弄坏，要么触发了机器人的安全保护直接停机。

为什么会这样？因为各种传感器的频率对不上。摄像头一秒钟拍 30 张图，但力传感器一秒钟能读 200 次数据。现在的做法是把所有数据强行拉平到 30 Hz 一起喂给模型。这就好比你戴着只每秒响 30 次的耳机去听一秒钟响 200 次的警报，很多关键的高频警报直接被漏掉了。

## 2. FAVLA 的绝招：快慢双脑分离

FAVLA 的核心思路就是借鉴了人类大脑的快慢双系统（Kahneman 的 System 1 / System 2），把模型拆成两半，各干各的。

**慢脑（Slow VLM）**：还是那个老教授，一秒钟想 30 次。它负责看图、读指令，顺便看看过去一小段历史的力数据，搞清楚“我现在在哪、要干嘛、刚才撞了没”。它想完一次，就把思考的成果（KV Cache）放在黑板旁边，供快脑随时查阅。

**快脑（Fast Action Expert）**：这是一个小而精的模块，专门负责“纠偏”。它跑得飞快，可能一秒钟跑几百次。它一边盯着最新的高频力传感器数据，一边参考慢脑留在黑板上的大局观，随时微调机械臂的动作。如果突然感觉阻力大了，它立刻让机械臂往回缩一点或者换个角度。

这样一拆，慢脑不用被高频数据烦死，快脑也不用去理解复杂的图像语义，专门干好“力反馈闭环控制”这事儿。

## 3. 两个关键的小聪明

光拆开还不够，FAVLA 还搞了两个特别精巧的设计来让这俩脑配合得更好。

**第一个叫 Force Adapter（力信号注射器）**。
以前的模型是把力信号也变成 token 拼在一堆图像 token 后面。问题是图像 token 几百个，力 token 才几个，模型一做 attention，注意力全被图像吸走了，力信号等于白给。
FAVLA 的做法是，在快脑的每一层网络里，单独开一个“小窗口”（cross-attention），让动作 token 主动去问力信号：“嘿，现在力咋样了？”。这样力信号就不会被淹没，每一层都能直接影响动作的生成。这就像给快脑单独配了一副力觉眼镜，随时盯着看。

**第二个叫 Force Variance Head（力波动预测器）和自适应频率**。
快脑跑那么快，很费算力的。如果机械臂在空中移动（free-space），根本没接触，跑那么快纯属浪费。那怎么知道什么时候该跑快呢？
FAVLA 让慢脑在思考大局的时候，顺便预测一下：“接下来这小段动作，力传感器的数据会不会剧烈波动？”（预测一个 force variance 标量）。
如果预测说“没啥波动，挺平稳”，那快脑就偷懒，一秒钟只跑 1 次，跟普通 VLA 一样开环执行。
如果预测说“哎呦，接下来要撞上了，力要乱跳了”，快脑立刻拉满转速，一秒钟跑 N 次，严阵以待，随时准备根据力反馈微调动作。
这招叫“好钢用在刀刃上”，既省了算力，又保证了关键时刻的反应速度。

## 4. 实验结果说话：又快又稳还不爱坏东西

Paper 里跑了四个真实的接触密集型任务：插 USB、装齿轮、翻纸箱、擦板子。

结果很漂亮：
1. **成功率高**：平均 80.8%，比最强的 baseline (ForceVLA) 还高 13.8 个百分点。
2. **下手更轻**：这是最关键的。装齿轮的时候，以前的方法平均用 12 N 的力，FAVLA 只用 7.7 N。这意味着它干活更“温柔”，不容易把工件怼坏，这在工业里非常重要。
3. **消融实验很 clean**：单单加上 Force Adapter 就有大提升，再加上频率自适应又提一截。说明每个零件都在干活。

特别有意思的一个反例：如果把力信号简单地拼到 π₀ 的输入里（π₀ + Force），效果反而比纯视觉的 π₀ 还差（用力更大了）。这就完美印证了前面说的“token dilution”——简单粗暴地加力信号，反而干扰了原本的视觉策略，成了猪队友。FAVLA 的精细设计刚好避开了这个坑。

## 5. 一句话总结

FAVLA 就是给机器人装了一个“慢悠悠看大局的脑子”加上一个“死盯力传感器随时纠偏的快手”，并且让慢脑子告诉快脑子“什么时候该紧张起来跑快点”。这样机器人干精细活儿的时候，既能看懂图纸，又能在撞上的瞬间灵活闪避，不至于像个铁头娃一样硬怼。

这思路其实跟你讲神经网络时强调的 “inductive bias” 很像——不同的数据有不同的物理特性（频率、时序），硬把它们塞进同一个统一的框架里效率很低，顺着它们的物理性质做架构设计，往往事半功倍。

---

# FAVLA: Force-Adaptive Fast-Slow VLA 详解

Hi Andrej, 这篇 paper 处理的是 contact-rich manipulation 里面一个相当本质的问题——**sensing modality 之间的 frequency mismatch**。下面我把整篇 paper 拆开,从 motivation → architecture → inference strategy → experiments → 相关联想,一层层把 intuition 给你 build 起来。

---

## 1. Motivation: 为什么 unified-frequency fusion 在 contact-rich 任务里会崩

当前的 VLA 主流范式是 **VLM + Action Expert (AE)** 的 two-stream pipeline,以 π₀ (Black et al., 2024) 为代表:
- VLM 跑在低频率(约 15 Hz,受限于 camera frame rate 和大模型推理成本),输出一段 **action chunk** $A_t = \{a_t, a_{t+1}, \dots, a_{t+H-1}\}$
- AE 执行这个 chunk 期间基本上是 **open-loop** 的,要等下一次 VLM 更新才能感知外部变化

在 pick-and-place 这类 free-space 任务里,这种 open-loop chunk 执行问题不大,因为物体动力学是 slow-varying 的。但是 contact-rich 任务里存在 **三类瞬态事件**:
1. **Impacts** —— 接触瞬间力在毫秒级从 0 跳到几十 N
2. **Stick-slip** —— 摩擦力的 stick-slip oscillation,频率在 10–100 Hz
3. **Jamming** —— 插装/装配里的卡滞,需要立即减压或微调

而典型的 sensor sampling rate 是:
- RGB camera: 15–30 Hz
- 6-axis F/T sensor: 100–1000 Hz
- Joint torque / proprioception: 200–1000 Hz

在 unified-frequency pipeline 里,所有 modality 必须对齐到最慢的那个(visual),这就意味着 high-frequency force data 必须 downsample 到 15–30 Hz。**关键瞬态信息在 downsampling 里被 alias 掉了**。等到 VLM 跑下一轮 inference,机械臂可能已经撞坏工件了。

> 这其实就是经典的 Nyquist 问题,只不过现在发生在 end-to-end learned policy 的 input 层面。可参考 classic control 文献关于 *sample-data systems* 的讨论: https://www.sciencedirect.com/topics/engineering/sample-data-system

paper 里反复强调的一个工业 motivation:high-precision assembly 里,延迟的 force response 会触发 robot 的 safety stop,整条产线 reset,代价很高。

---

## 2. 核心 idea: Fast-Slow 解耦

FAVLA 借了一个非常直觉的类比——Kahneman 的 **System 1 / System 2** 框架,不过搬到 robotics 上:
- **Slow system = VLM**:scene understanding、language grounding、long-horizon planning,这些是 "slow" variable,30 Hz 已经够用
- **Fast system = Action Expert + Force Adapter**:contact regulation、reactive correction,这些需要 100 Hz+ 才能闭环

这跟神经科学里的 *perception-action dual-stream* 也有呼应,可参考 Goodale & Milner 的 ventral/dorsal stream: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3257269/

FAVLA 的关键设计选择:
- slow VLM 跑在 **fixed low frequency**
- fast AE 跑在 **variable high frequency**,频率由 VLM 预测的 future force variance 自适应调度
- slow 和 fast 之间通过 **KV cache reuse** 通信,避免每次 AE inference 都重新跑 VLM

---

## 3. Architecture 详解

### 3.1 输入形式化

公式 (1) 定义了 timestep $t$ 的 observation:

$$
O_t = \{\mathcal{T}_t^{(k)}\} \cup \{\mathbf{s}_t\} \cup \{\mathbf{f}_{t-\tau+1:t}, \mathbf{f}'_{t-\tau+1:t}\}
$$

变量含义:
- $\mathcal{T}_t^{(k)}$ —— 第 $k$ 张 RGB 图像(paper 里用 external camera + wrist camera 两张,$k \in \{1,2\}$)
- $\mathbf{s}_t \in \mathbb{R}^7$ —— proprioceptive state,7 维 = 6-DoF TCP pose (xyz + roll/pitch/yaw) + 1 维 gripper width
- $\mathbf{f}_{t-\tau+1:t} \in \mathbb{R}^{\tau \times 6}$ —— **history** force sequence,低频更新(随 VLM 一起进),6 维 = $[f_x, f_y, f_z, m_x, m_y, m_z]$,前 3 个是 force,后 3 个是 torque
- $\mathbf{f}'_{t-\tau+1:t} \in \mathbb{R}^{\tau \times 6}$ —— **latest** force sequence,高频更新(直接喂给 AE)
- $\tau$ —— time window 长度,paper 里是 10 steps

注意 paper 把 force 拆成了两路:**一路走慢路**当 context(让 VLM 知道过去 force 的趋势),**一路走快路**做 reactive correction。这个拆分非常关键,后面 force adapter 也是基于这个 split 设计的。

### 3.2 Slow VLM Backbone

架构基于 π₀,用 **PaliGemma (Beyer et al., 2024)** 作为 VLM,2.6B 参数,18 层 transformer,hidden dim 2048。

输入 token 组成:
1. **Vision tokens** —— SigLIP encoder 把 RGB 图 patchify 成 tokens
2. **Language tokens** —— Gemma tokenizer 处理 instruction
3. **History force tokens** $\mathbf{z}_f$ —— 通过一个 **TCN (Temporal Convolutional Network) tokenizer** 把 $\mathbf{f}_{t-\tau+1:t}$ 编码成 $N_f$ 个 token

TCN tokenizer 是这篇 paper 一个比较 subtle 的设计:
- 1×1 temporal conv 把每个 timestep 的 6D force 投影到 $d_f$ 维
- 然后 4 层 causal dilated temporal conv block,dilation rate = $[1, 2, 4, 8]$ (典型的 TCN receptive field 指数增长设计,参考 Bai et al. 2018: https://arxiv.org/abs/1803.01271)
- residual + layer norm
- 最后 downsample 到 $N_f = 4$ 个 token

**为什么用 TCN 而不是 transformer**?我猜是因为 force history 是低频的且长度短(10 steps),TCN 的归纳偏置(locality + temporal causality)更合适,而且参数少(25.2M vs 如果用一个小 transformer 会更大)。

VLM 输出:
$$
\mathbf{H}_{\mathrm{VLF}} \in \mathbb{R}^{S_{\mathrm{pre}} \times d}
$$
其中 $S_{\mathrm{pre}}$ 是 prefix token 总数(视觉 + 语言 + 力),$d = 2048$ 是 hidden dim。

同时 cache 每一层的 KV:
$$
\mathbf{K}^{(\ell)} \in \mathbb{R}^{S_{\mathrm{pre}} \times n_{kv} \times d_h}, \quad \mathbf{V}^{(\ell)} \in \mathbb{R}^{S_{\mathrm{pre}} \times n_{kv} \times d_h}
$$
- $\ell$ 是 layer index
- $n_{kv}$ = key/value head 数(因为用 Grouped Query Attention,这个数小于 query head 数)
- $d_h = 256$ 是 head dimension

这个 KV cache 是 fast-slow 解耦的核心——slow VLM 跑一次,AE 多次复用,类似 LLM inference 里的 prefill-decode split。

### 3.3 Fast Force-Injected Action Expert

AE 是一个 300M 的小 transformer,18 层,hidden dim 1024。比 VLM 小一个量级,才能跑得快。

AE 的输入:
1. VLM 提供的 KV cache $\{\mathbf{K}^{(\ell)}, \mathbf{V}^{(\ell)}\}$
2. 最新的 proprioceptive state $\mathbf{s}_t$
3. 最新的 force sequence $\mathbf{f}'_{t-\tau+1:t}$
4. Noisy action token $\mathbf{z}_a$ —— flow matching 的噪声化 action

输出是 action chunk $A_t \in \mathbb{R}^{32 \times 7}$(chunk size $H = 32$,action dim = 7,对应 $\Delta x, \Delta y, \Delta z, \Delta\phi, \Delta\theta, \Delta\psi, g$)。

#### Force Adapter 设计

这是 paper 最本质的架构创新。一般的做法是把 force tokens 也拼到 input sequence 前面,让 AE 通过 self-attention 处理。但 paper 指出这种方式有 **token dilution** 问题:vision tokens 几百个,force tokens 只有 4 个,attention 权重会被 vision 主导,force 信号被淹没。

Force adapter 的做法是:在每个 transformer layer 里,在 self-attention (with VLM KV cache) 之后,**额外加一个 cross-attention**,让 noisy action token $\mathbf{z}_a$ 主动去 query 最新 force tokens:

$$
\mathbf{z}_a' = \mathbf{z}_a + \mathrm{Attn}(\mathbf{Q}_a, \mathbf{K}_f, \mathbf{V}_f)
$$

变量含义:
- $\mathbf{z}_a$ —— 当前 layer 的 noisy action token representation
- $\mathbf{Q}_a$ —— 由 $\mathbf{z}_a$ 投影得到的 query
- $\mathbf{K}_f, \mathbf{V}_f$ —— 由 latest force tokens $\mathbf{z}'_f$ 投影得到的 key/value
- $\mathbf{z}_a'$ —— force-conditioned update,additively 残差注入

直觉上:每一层 AE 都让 action 显式地去"听"force 信号,force 信号不会被视觉 token 稀释。这个设计跟 Flamingo 的 cross-attention gated xattn layer (Alayrac et al., 2022: https://arxiv.org/abs/2204.14198) 思路很接近,只不过这里是 force 而不是 vision。

更深的联系:这其实跟 **conditional flow matching** 的 conditioning 机制类似。Flow matching 本质上是在学一个 conditional vector field $v_t(x_t | c)$,其中 $c$ 是 conditioning。FAVLA 相当于把 conditioning 从"只放在 input 层"扩展到"每一层都做 force-conditioned refinement"。

### 3.4 Force Variance Head

这是另一个关键设计——VLM 不光输出 context,还输出一个 scalar $\tilde{\nu}_t \in [0,1]$ 表示**对未来 force 波动性的预测**。

#### 监督信号构造

给定未来 $W$ 步的 force 序列 $\mathbf{f}_{t:t+W-1} \in \mathbb{R}^{W \times 6}$,$W$ = action chunk size = 32:

$$
\nu_t = \sum_{j=1}^{6} w_j \, \mathrm{Var}(\mathbf{f}_{t:t+W-1}^{(j)})
$$

- $j$ 是 6 个 force/torque 分量的 index
- $w_j$ 是每个分量的权重(paper 没明说,推测是 uniform 或可学习)
- $\nu_t$ 是 6 个分量方差的加权和,反映未来一段时间的 contact volatility

然后做 EMA smoothing 去噪:

$$
\bar{\nu}_t = \mathrm{EMA}(\nu_{\leq t}; \alpha)
$$

EMA 的递归形式是 $\bar{\nu}_t = \alpha \nu_t + (1-\alpha)\bar{\nu}_{t-1}$,$\alpha$ 是 smoothing factor。这步是为了避免单帧 force 噪声导致的 label 跳变。

再 normalization:
$$
\tilde{\nu}_t = \tanh\left(\frac{\sqrt{\bar{\nu}_t}}{\sigma}\right)
$$

- $\sigma$ 是数据集 force variance 的标准差(用于尺度归一化)
- $\tanh$ + $\sqrt{\cdot}$ 把信号压到 $[0,1)$,避免极端值

最后 head 是个 MLP,(256, 128) hidden,1 维输出,用 GELU 激活,直接 regress $\tilde{\nu}_t$。

#### 训练 loss

总 loss:
$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{action}} + \lambda \mathcal{L}_{\mathrm{var}}
$$

- $\mathcal{L}_{\mathrm{action}}$ —— flow matching 的 standard action generation loss(conditional flow matching objective,Lipman et al., 2023: https://arxiv.org/abs/2210.02747)
- $\mathcal{L}_{\mathrm{var}}$ —— force variance regression loss(MSE)
- $\lambda = 0.1$ —— auxiliary loss 权重,小一点防止 dominate 主任务

#### Intuition

这个 head 实际上是让 VLM 学会一个 "contact prediction" 的 meta-task。学这个任务迫使 VLM 在 representation 里编码 "接下来会不会有接触" 这个 latent 信息,而这个信息正好用来调度 AE 频率。这跟 *auxiliary learning* 的经典思路一致——auxiliary task 不是为了它自己,是为了 shape representation。

类比:ALOHA 的 ACT (Zhao et al., 2023) 里的 temporal ensemble 也类似这种"借东风"的设计。

---

## 4. Force-Adaptive Fast-Slow Inference Strategy

这是 paper 的另一半核心创新,解决两个问题:
1. Fixed frequency 不能适应 task phase(free-space 和 contact 阶段需求完全不同)
2. 同一 chunk 内多次 AE 独立 inference 会有 stochastic 不一致,产生 jitter

### 4.1 Adaptive Inference Frequency

公式 (6):

$$
n_t = \max\left(1, \lceil \tilde{\nu}_t \cdot N_{\mathrm{max}} \rceil\right)
$$

变量:
- $\tilde{\nu}_t \in [0,1]$ —— VLM 预测的 force variance
- $N_{\mathrm{max}}$ —— 最大 frequency ratio(paper 里实验从 1 试到 4,推测生产设置可能是 8 或 16)
- $\lceil \cdot \rceil$ —— ceiling,向上取整保证至少一次 inference
- $n_t$ —— 这一周期内 AE 要跑多少次

调度逻辑:
- $\tilde{\nu}_t \to 0$(free-space):$n_t = 1$,AE 跑一次,像传统 VLA 一样开环执行 chunk
- $\tilde{\nu}_t \to 1$(强 contact):$n_t = N_{\mathrm{max}}$,AE 高频闭环修正

这其实跟 MPC 里的 **adaptive horizon / adaptive sampling** 思路很类似——control 频率本身变成一个可调参数,根据当前 system dynamics 的 "interestingness" 调度。

参考 RTC (Reactive Temporal Control) 和 AWM (Adaptive World Models) 的思路: https://arxiv.org/abs/2304.13705 (ACT 原文)

### 4.2 Consistent Action Ensemble

同一个 chunk 内多次跑 AE 会产生 jitter,因为每次采样的 noise 不同,flow matching 输出的轨迹会跳。两个策略解决:

**1. 固定 noise**:同一 visual cycle 内,所有 AE calls 共享同一个 sampled noise $\epsilon$。这样多次 inference 的差异只来自 conditioning 变化(主要是 force),不来自 stochastic noise。

直觉:这相当于把 flow matching 的 stochasticity "freeze" 住,让多步 inference 之间变成 "conditional on force 的连续 refinement",而不是 "independent sampling"。

**2. Temporal ensemble**:跨 visual cycle,用 ACT 的做法:
$$
a_t = \sum_{i} w_i \hat{a}_t^{(i)}
$$
其中 $\hat{a}_t^{(i)}$ 是过去第 $i$ 个 chunk 对当前 timestep $t$ 的预测,$w_i \propto \exp(-i \cdot \beta)$ 是指数衰减权重,$\beta$ 控制衰减速度。

这等价于一个 **exponentially weighted moving average over overlapping chunk predictions**,起到 low-pass filter 的作用,平滑高频 AE 调整带来的 jitter。

---

## 5. Experiments

### 5.1 Setup

- **Robot**: Monte dual-arm,7-DoF X-ARM,每只手臂末端有 6-axis F/T sensor + wrist RGB-D camera
- **Tasks**: 4 个 contact-rich 任务
  1. **USB Insertion** —— 毫米级精度要求
  2. **Gear Assembly** —— 齿轮咬合,过度用力会触发 safety stop
  3. **Box Flipping** —— 翻转纸箱,需要适应变化的 contact force
  4. **Board Wiping** —— 板擦擦板,要保持稳定 contact force
- **Data**: 80 trajectories for 高精度任务,50 for contact-rich 任务,total 260 trajectories / 198,250 frames / 1.84 hours
- **Sensors**: F/T @ 200 Hz,camera @ 30 Hz,数据 align 到 30 Hz 后训练
- **Teleoperation**: 3D SpaceMouse,force → audio 转换给操作员听(6 维 force 各对应一个 220-880 Hz 的 frequency,音量对应幅值)—— 这个 teleop 设计很巧,人通过耳朵感知 force
- **Training**: 30k iters,batch 8,AdamW,cosine decay,LoRA (VLM r=16 α=16,AE r=32 α=32),A100 80GB,6 小时

### 5.2 Baselines

1. **π₀** —— 纯视觉 baseline
2. **π₀ + Force** —— 把 force tokenize 后拼到 input sequence 的简单融合
3. **TA-VLA** (Zhang et al., 2025b) —— 加 auxiliary force prediction head 提升 force 注意力,https://arxiv.org/abs/2509.07962
4. **ForceVLA** (Yu et al., 2025) —— MoE 架构学习 force + image,https://arxiv.org/abs/2505.22159

### 5.3 主结果:Success Rate

FAVLA 平均 80.8%,比 π₀ 高 38 个百分点,比最强 baseline ForceVLA 高 13.8 个百分点。

各任务成绩(从 Fig. 6 推测):
| Task | π₀ | π₀+Force | TA-VLA | ForceVLA | FAVLA |
|------|----|---------|-------|----------|-------|
| USB Insertion | ~30% | ~50% | ~55% | ~65% | 80% |
| Gear Assembly | ~50% | ~60% | ~65% | ~80% | 93.3% |
| Box Flipping | ~50% | ~50% | ~65% | ~70% | 80% |
| Board Wiping | ~10% | ~30% | ~40% | ~50% | 70% |

Gear Assembly 和 Board Wiping 提升最显著——这两个任务最依赖 force 调控,正好验证 fast-slow 的价值。

### 5.4 Peak Contact Force(关键 safety 指标)

Table 1:
| Method | Gear Assembly | Box Flipping |
|--------|---------------|--------------|
| π₀ | 12.0 N | 12.2 N |
| π₀+Force | 13.0 N | 13.8 N |
| TA-VLA | 11.3 N | 10.0 N |
| ForceVLA | 10.9 N | 12.4 N |
| FAVLA | **7.7 N** | **9.9 N** |

Gear Assembly 上从 12.0 N 降到 7.7 N,降幅 4.3 N。**注意 π₀+Force 居然比 π₀ 还差**——这验证了 paper 反复强调的 token dilution 假设:简单把 force 加到 input sequence 反而干扰了 vision 主导的 policy,导致 force spike 更大。这是个很有信息量的反例。

### 5.5 Ablation 1: 各组件贡献

Table 2(Box Flipping / Board Wiping):
| Configuration | Box Flip | Board Wipe |
|---------------|----------|------------|
| Vision-Only | 50% | 10% |
| + Force-Injected AE | 65% | 60% |
| + Force Variance Prediction | 70% | 60% |
| + Force-Adaptive Inference | **80%** | **70%** |

分析:
- **Force-Injected AE** 单独就给 Box Flipping +15%,Board Wiping +50%——证实 force 信号本身价值巨大
- **Force Variance Prediction** 进一步 +5%(Box Flipping)——auxiliary task 让 representation 更好
- **Force-Adaptive Inference** 再 +10% / +10%——adaptive 频率本身价值显著

注意 Force Variance Prediction 单独加在 Board Wiping 上没提升,但加 Inference 后才跳到 70%。说明 variance prediction 的价值是通过 adaptive frequency 释放的——单独预测没用,要拿来调度才有效。这是个非常 clean 的 ablation。

### 5.6 Ablation 2: Static vs Adaptive Frequency

Fig. 7 比较 static frequency ratio $n \in \{1, 2, 4\}$ 和 adaptive:
- USB Insertion: static n=4 达到 ~70%,adaptive 达到 80%
- Gear Assembly: static n=4 达到 ~88%,adaptive 达到 93%

Adaptive 在所有任务上都不输给最好的 static setting,而且没有手动调 $n$ 的麻烦。Free-space 阶段不需要高频,static 高频反而会让 chunk 之间切换过快产生不稳定。

### 5.7 Adaptive Frequency Visualization

Fig. 8 展示了:
- (a) 原始 force 测量
- (b) 计算 force variance
- (c) 模型预测的 frequency ratio

可以清楚看到:在接触发生前一点点(模型预测到接触要来了),frequency ratio 就开始上升;free-space 阶段 ratio 回到 1。说明模型学会了 **anticipatory scheduling**——这其实是一种 learned predictive control,跟 MPC 里预测 horizon 的思路打通。

---

## 6. Failure Cases 分析(很有信息量)

paper 附录 Fig. 14 列了 4 个 task 的典型 failure:

1. **USB Insertion**: misalignment 时 force 信号跟 "成功插入" 的 force 信号太像,模型误判为完成 → 提前张手。这是 force-only 的 ambiguity 问题,vision 应该能 disambiguate 但没学好。

2. **Gear Assembly**: 卡住时持续下压 → force 触发 safety stop。这是 force adaptive 的边界——如果 misalignment 太严重,高频 correction 也救不回来,需要在更早的 phase 就做 re-plan。

3. **Box Flipping**: 为了不损坏盒子,force 太小 → 盒子滑掉。这是 compliance vs stability 的经典 trade-off。

4. **Board Wiping**: wiping trajectory 不对,漏了上面的 marker。这是 task coverage 问题,跟 force 控制关系不大。

这些 failure 都很有诊断价值,展示了 force-adaptive 的局限——它解决 contact regulation,但解决不了 task-level planning 错误。

---

## 7. 相关工作与 deeper intuition 联想

### 7.1 跟 Fast-in-Slow (Chen et al., 2025a) 的关系

Fast-in-Slow (https://arxiv.org/abs/2506.01953) 也做 fast-slow 分层,但思路不同:它把 fast manipulation 嵌入到 slow reasoning 里,fast 是 slow 的一个 "subroutine"。FAVLA 是 spatial 解耦(slow VLM vs fast AE 是两个 model),Fast-in-Slow 是 temporal 嵌套。两者可以互补。

### 7.2 跟 Reactive Diffusion Policy (Xue et al., 2025) 的关系

RDP 也用 slow-fast diffusion + visual-tactile,但没区分频率,也没 VLM。FAVLA 可以看成 RDP 的 "VLM-augmented + frequency-adaptive" 升级版。https://arxiv.org/abs/2503.02881

### 7.3 跟 Tactile-VLA 系列 (Huang et al., 2025b; Bi et al., 2025) 的关系

Tactile 系工作用 visuotactile sensor(DIGIT, GelSight 之类),信号更高频更细,但 sensor 装在 fingertips,覆盖面有限。FAVLA 用 wrist F/T sensor,覆盖整个末端工具的空间力,适合工业装配场景(末端装夹具)。两者其实是 complementary 的——tactile 给局部 contact patch,F/T 给全局 wrench。

### 7.4 跟经典 control: Hybrid Force-Position Control 的关系

Raibert-Craig 的 hybrid force-position control (1981) 把 task space 分成 force-controlled 和 position-controlled 的正交方向。FAVLA 实质上把这个思想 learned 化了——不同 phase、不同方向上 force 的重要性由 model 自动学习,不用 task designer 手动指定 constraint frame。

经典文献: https://ieeexplore.ieee.org/document/1164194

### 7.5 跟 Impedance Control 的关系

Hogan 的 impedance control (1985) 把机器人建模成 mass-spring-damper,通过调节 virtual stiffness 来适应 contact。FAVLA 没有显式 impedance,但 AE 在 force 反馈下高频调整 action chunk,效果类似一个 learned variable impedance——contact 时自动变 compliant,free-space 时自动变 stiff。

Hogan 原文: https://asmedigitalcollection.asme.org/dsc/article-abstract/107/1/1/113707/Impedance-Control-An-Approach-to

### 7.6 跟 LLM 领域的 speculative decoding 的关系

这个是我觉得最有意思的联想——FAVLA 的 fast-slow 结构跟 **speculative decoding** (Leviathan et al., 2023: https://arxiv.org/abs/2302.01318) 在结构上是 dual 的:
- Speculative decoding: small draft model 跑快,大 model verify
- FAVLA: big VLM 跑慢给 context,small AE 跑快做 action

两边都是利用"慢的大模型稀疏调用 + 快的小模型高频调用"这个 pattern,只不过一个为了 LLM 加速,一个为了 robot 反应。

### 7.7 跟 Predictive Coding 的关系

Force variance head 预测 future force variance,这跟 predictive coding / active inference(Friston,2010)的思路非常像——大脑持续预测未来 sensory 信号,根据 prediction error 调整 action。FAVLA 把"预测 future force 变动"当作调度信号,本质上是 active inference 的工程化实现。

Friston 文章: https://www.nature.com/articles/nn.2796

### 7.8 跟 Hierarchical RL 的 Options Framework 的关系

Sutton-Precup 的 options framework(1999)把 policy 分成 high-level option 和 low-level action。FAVLA 可以看成:
- VLM 选 "option" = 当前 force variance regime (低/中/高)
- AE 根据 option 决定自身 frequency,执行对应 temporal abstraction 的 policy

参考: https://link.springer.com/article/10.1023/A:1007621212390

### 7.9 跟 Async Actor-Critic (A3C) 的关系

A3C 也是 multiple workers at different frequencies,但都在同一个 model 里。FAVLA 把 worker (AE) 和 learner (VLM) 真正分到不同 model,而且 worker 频率可调。这其实更像 **real-time control system** 的 design pattern——sensor processing / planning / control 跑在不同频率的进程上,通过 shared memory 通信。

---

## 8. 我对这篇 paper 的批评性 thoughts

### 优点
1. **Problem formulation 干净**:frequency mismatch 是个真问题,而且之前 force-aware VLA 文献没正面处理
2. **Force adapter 设计 minimal 但 effective**:cross-attention 注入,不破坏主架构
3. **Variance head + adaptive frequency 形成 self-consistent 闭环**:预测-调度-执行
4. **Real-world 实验扎实**:4 个 task,baseline 选得合理,ablation 完整
5. **Failure case 分析诚实**:展示了方法的 limitation

### 可质疑的点
1. **$N_{\mathrm{max}}$ 上界**:实验里只到 $n=4$,真实工业场景可能需要 $n=16$ 甚至更高才能处理 1ms 级 impact。但 $n$ 越大,AE inference 成本越高,可能从 real-time 跌到 non-real-time。
2. **Variance vs Mean**:用 variance 作为 contact 预测信号略粗糙——一个稳定的强 contact(比如 stable wiping)variance 低但 force 大,这时候 AE 应该高频还是低频?可能 force magnitude 也应该进 scheduling。
3. **LoRA fine-tune**:用 LoRA 调 VLM,意味着 force representation 可能没真正进入 VLM 的 deep layers,adapter 的力信号实际靠 AE 层面吸收。如果 full fine-tune 会不会更好?
4. **数据规模太小**:1.84 小时数据训出 80% success rate,跟 Open-X 这种大数据集差距大。但 contact-rich 任务本来就难采集。
5. **比较 base 不够强**:baselines 里没有最新的 π₀.5 (Black et al., 2025),也没有 OpenVLA-OFT,可能比较不够 SOTA。不过 paper 接受时间可能比较早。
6. **没有 sim2real**:工业部署应该需要 sim 大规模训练,这篇纯 real-world data 限制了 scalability。

### 想做的 follow-up
1. 把 force variance head 扩展成 force distribution head(预测 next-force 的分布而非方差)
2. 加 tactile sensor 做第三级 modality,三级 frequency 的 fast-slow
3. 在 sim 里大规模训练,验证 fast-slow 在更多 task 上的 generalization
4. 跟 impedance control 做 hybrid:AE 输出不是 action 而是 impedance parameter + reference action

---

## 9. Key References

- π₀: https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2510.0...
- PaliGemma: https://arxiv.org/abs/2407.07726
- TCN (Bai et al.): https://arxiv.org/abs/1803.01271
- Flow Matching (Lipman): https://arxiv.org/abs/2210.02747
- ACT (Zhao et al.): https://arxiv.org/abs/2304.13705
- Flamingo (cross-attention): https://arxiv.org/abs/2204.14198
- LoRA: https://arxiv.org/abs/2106.09685
- OpenVLA: https://arxiv.org/abs/2406.09246
- ForceVLA: https://arxiv.org/abs/2505.22159
- TA-VLA: https://arxiv.org/abs/2509.07962
- Reactive Diffusion Policy: https://arxiv.org/abs/2503.02881
- Fast-in-Slow: https://arxiv.org/abs/2506.01953
- VLA-Touch: https://arxiv.org/abs/2507.17294
- Speculative Decoding: https://arxiv.org/abs/2302.01318
- Active Inference (Friston): https://www.nature.com/articles/nn.2796
- Hybrid Force-Position Control (Raibert-Craig): https://ieeexplore.ieee.org/document/1164194
- Impedance Control (Hogan): https://asmedigitalcollection.asme.org/dsc/article-abstract/107/1/1/113707/

---

希望这个 breakdown 对你 build intuition 有帮助。整篇 paper 的核心 intuition 其实可以浓缩成一句话:**让 slow VLM 专注"理解场景和预测接触",让 fast AE 专注"听 force 信号并即时修正动作",两者通过 KV cache 和 force variance signal 在频率上自适应耦合**。这跟大脑的 ventral stream(感知)和 dorsal stream(动作)的功能分工,以及 cortex(cold-slow)和 cerebellum(fast reactive)的协同,在结构上是非常 deep 的类比。

你最近在 Eureka Labs 做 education-related 工作,这个 fast-slow decomposition 其实也是教学 robot learning 时一个很好的 pedagogical frame——把"什么时候用大 model 推理,什么时候用小 model 反应"显式化,对初学者理解 robotic system design 很有帮助。
