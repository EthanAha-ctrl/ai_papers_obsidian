---
source_pdf: POS-ISP Pipeline Optimization at the Sequence Level for Task-aware ISP.pdf
paper_sha256: becba236382a6e9edaf4179384fcc56afe286ddc63d5385615d9a0042343c668
processed_at: '2026-08-06T05:21:47-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 POS-ISP

## 一句话讲清楚

相机的 ISP 把 sensor raw data 变成人看的图，是为 eyeball 优化的。对机器视觉任务（detection、segmentation）反而有害——低光场景下相机 JPEG 喂 YOLOv3 的 mAP 是 37.6，直接喂 minimally-processed raw 反而 44.1。所以问题变成：**怎么给每个下游任务定制一个 ISP**。

ISP 是一串 module（white balance → denoise → tone map → ...），每个 module 有自己的参数。要同时决定用哪些 module、什么顺序、每个参数调多少，这是离散+连续混合优化，不可微，没法直接 backprop。

## 这件事为什么难

ISP 跟搭积木有点像，但比搭积木难两个量级：

1. **离散选择不可微**：你选 module A 还是 module B，这是个 argmax，梯度没法穿过去
2. **顺序敏感**：先 denoise 再 sharpen 还是反过来，结果天差地别。denoise 在前能保住 sharpen 不放大噪声；sharpen 在前会把噪声也锐化，后面 denoise 就模糊掉刚锐化的细节
3. **参数耦合**：white balance 的 gain 调歪了，后面 color correction 的矩阵也得跟着调，参数互相牵着走
4. **reward noisy**：低光 detection 的 reward 来自 detector loss，一张图里检测漏一个物体 loss 就跳一下，方差极大

## 之前两波人怎么搞，为什么都不行

### 第一波：ReconfigISP，differentiable NAS 路线

核心 trick：把硬选择软化。每一步对每个候选 module 学一个 weight $\alpha_i$，stage 输出是 candidate 的加权和 $y = \sum \alpha_i M_i(x)$。这样可微了，gradient descent 一把梭。

问题：**训练时 mixed，推理时 hard argmax**。训练时 module 输出被 blend，推理时突然要选一个，behavior 不连续。低光场景 SNR 低，soft 和 hard 行为差异被放大，Tab. 1 里 ReconfigISP 47.8 mAP 反而输给 raw input 44.1。DARTS 那一脉 NAS 都有这个 train-test gap 的毛病。

### 第二波：DRL-ISP / AdaptiveISP，stepwise RL 路线

把 ISP pipeline 构造当 MDP：每个 stage 是一个 time step，state = 当前 image + 已选 module history，action = 下一个 module，用 actor-critic 训练。

致命问题在 **critic 要估计 future reward**：

$$V(s_t) \leftarrow r_t + \gamma V(s_{t+1})$$

这就是经典 RL 的 **deadly triad**：function approximation + bootstrapping + (off-policy)，三个同时存在 value function 会 diverge。Paper Fig. S2 直接画出来 critic loss 剧烈震荡和 spike。

更深一层：ISP 的 reward 高度 noisy。Dense prediction 任务（segmentation、depth）一个小像素偏差就能让 reward 剧烈跳变。Critic 拟合不准，actor 跟着飘。Tab. 2 里 AdaptiveISP 在 LIS-Dark segmentation 上 25.2 mAP，甚至比 raw input 27.8 还低，stepwise RL 在这种 noisy reward 场景直接崩。

## POS-ISP 的核心 insight

**把 stepwise MDP 拍扁成 sequence-level 问题**：

- 一次 forward 预测整个 module 序列 $\mathcal{A} = (a_1, ..., a_k)$
- 一次 forward 预测所有 module 的 parameters $\Theta$
- 跑完整 pipeline 得 $I_{out}$
- 用 terminal task reward 反传

没有 critic，没有 bootstrapping，没有 intermediate reward，直接消灭 deadly triad 里的一个 component。自然稳定。

这跟 NMT 早期 sequence-level RL 的工作（Ranzato et al. 2015）思想一样：把 BLEU 这种 sequence-level metric 当 reward，避免 token-level cross-entropy 的 exposure bias。POS-ISP 把整个 pipeline 当 sequence，task performance 当 reward。

## 几个巧妙的 trick

### Trick 1: Reward 设计成 relative improvement

$$R = \mathcal{L}_\mathcal{T}(I_{in}) - \mathcal{L}_\mathcal{T}(I_{out}) - P$$

第一项是直接拿 raw 喂 task 的 loss，相当于一个 per-image 的 implicit baseline。把 absolute loss 转成 "ISP 带来了多少 improvement"。

标准 REINFORCE 是 $\nabla J = \mathbb{E}[R \cdot \nabla \log \pi]$，如果 $R$ 是 absolute loss 方差大；减一个 baseline $b(s)$ 变成 advantage $A = R - b$，方差小但不改梯度期望。这里 $b = \mathcal{L}(I_{in})$ 是 per-image constant，对每个 image 自动 normalize reward。Elegant 的设计，让 vanilla REINFORCE 都能稳定 work。

### Trick 2: Parameter Predictor 不 condition on Sequence

听起来反直觉：参数明明是给特定 module 用的，为什么不告诉 predictor 要选哪些 module？

Ablation（Tab. S10）：
- Sequence-Conditioned: 47.5 ± 0.26
- Image-Only: **47.8 ± 0.08**

Image-Only 不仅 mean 高，std 也小一半。

直觉：训练早期 sequence policy entropy 高（0.5 左右），sequence 几乎随机。如果 parameter predictor 依赖 sequence，每次 sequence 一变，参数空间目标就跟着飘，parameter predictor 在追 moving target。Image-only 把 parameter predictor 限制在 "对所有 plausible sequence 都还算 OK 的参数"，相当于 strong regularization。

后期 sequence policy entropy 降到 0，收敛到 few dominant pipelines，parameter predictor 已经学了 average behavior，自然 fine-tune 到 dominant sequences。这是 implicit curriculum：参数预测器先学 broad behavior，sequence policy 逐步收窄 search space，两者协同收敛。GAN 里 generator/driminator 不能同时太强也是类似道理。

### Trick 3: Single Sequence per Task + Image-Adaptive Parameters

每个 task 只学**一个** sequence $\hat{\mathcal{A}}$，所有 image 用同一个 sequence；但 parameters 是 per-image 自适应的。

两个 motivation：
1. **硬件 reality**：camera ISP 的 module 顺序 baked into silicon/firmware，不能 per-image reconfigure。Module 内部参数可以 per-image 调。这跟实际 camera 部署 model 对齐。
2. **Task-sequence 耦合**：Detection 依赖 structural info，contrast/sharpening 倾向靠前；perceptual quality 任务倾向把 exposure/tone 放前面。

Tab. S11 验证 sequence 确实 task-specific：用 enhancement 的 sequence 跑 detection 是 47.6，用 detection 自己的 sequence 是 48.0。光 retrain 参数补不回来 sequence 的差异。

Tab. S12 验证参数 image-adaptive：Dark 域训练的参数用在 Dark 测试 53.6 mAP，Normal 域训练的参数用在 Dark 测试 52.6，差 1 mAP。参数不是 domain-agnostic average，对 illumination 敏感。

### Trick 4: GRU + FiLM + Temperature + Mask

- **GRU**：捕捉 inter-module dependency，比独立 probability table 高 0.3-0.8 mAP（Tab. 5）
- **FiLM 调制**：step embedding 显式告诉 decoder "现在是第几步"，因为 ISP 第 1 步该做基础调整，第 5 步该做精修，不同位置动作分布应该不同
- **Mask**：每个 step 屏蔽已选 module，强制 $a_i \neq a_j$，避免 "denoise → denoise" 这种 weird pipeline
- **Temperature sampling**：早期 $\tau = 2.5$ 探索，后期 $\tau = 0.2$ exploit，指数衰减

## 为什么 work：实验数据说话

### Object Detection (LOD-Dark)

| Method | mAP@0.5:0.95 | 备注 |
|---|---|---|
| Camera ISP | 37.6 | 反而 hurt |
| Input RAW | 44.1 | baseline |
| ReconfigISP | 43.7 | train-test gap |
| DRL-ISP | 44.2 | stepwise RL 不稳 |
| AdaptiveISP | 47.2 | 现有 SOTA |
| **POS-ISP** | **47.8** | +0.6 |

### Instance Segmentation (LIS-Dark)

| Method | mAP@0.5:0.95 |
|---|---|
| Input RAW | 27.8 |
| AdaptiveISP | 25.2 |
| **POS-ISP** | **32.1** |

这里 POS-ISP 比 AdaptiveISP 高 **6.9 mAP**，差距远大于 detection 的 0.6。Segmentation 是 dense prediction，pixel-level supervision，reward variance 极高，stepwise RL 的 critic 在这种 noisy reward 下直接崩。POS-ISP 用 terminal reward + implicit baseline，绕开了 critic instability，所以收益在 noisy reward 任务上特别明显。

### Computational Efficiency

| Method | Params (M) | Runtime (ms) |
|---|---|---|
| AdaptiveISP | 7.18 | 12.72 |
| **POS-ISP** | **0.53** | **1.55** |

参数减到 1/13，runtime 减到 1/10。Sequence-level formulation 的直接红利：推理时 sequence 已经 fixed（训练完 greedy decode 一次），只跑 parameter predictor 一次 forward，没有 stepwise controller 的重复开销。

### On-device (Galaxy S10 CPU, FP32, 单线程)

| Method | FPS |
|---|---|
| AdaptiveISP | 8.70 |
| **POS-ISP** | **30.86** |

2019 年的 mobile CPU 上跑 real-time task-aware ISP。Deployment 价值大。

## 几个有意思的细节

### Pipeline Length 不是性能来源

Tab. S6 把 POS-ISP 强制 length=5（跟 AdaptiveISP 一样）：

| Method | Length | LOD-Dark mAP |
|---|---|---|
| AdaptiveISP | 5 | 47.1 |
| POS-ISP | 5 | 47.4 |
| POS-ISP | ≤10 | 47.8 |

固定 5 也已经赢 AdaptiveISP，说明性能优势来自 sequence-level optimization 本身，不是 pipeline 更长。Dynamic length 再多 0.4，是 nice-to-have 但非必要。

### 加 Runtime Penalty 几乎不掉点

不加 penalty 默认预测 length=10 pipeline，end-to-end 82.57ms。加 penalty length 降到 3，end-to-end 1.61ms，mAP 只掉 0.1（47.8 → 47.7）。这个 knob 给部署极大灵活性。

### Multi-seed 稳定性

3-seed std：POS-ISP 0.08-0.16，AdaptiveISP 0.15-0.57。Fig. 5(a) 训练曲线平滑单调上升 vs AdaptiveISP 早期剧烈震荡。

## ISP Module 池长啥样

10 个 module，挑几个有技术含量的：

**Tone Mapping**：8 个 piecewise-linear basis function，本质是 8 段折线曲线的 control point 权重，参数维度 8，最复杂的一个 module。

**White Balance**：3 个 channel gain，但用 luminance-preserving normalization（$0.27R + 0.67G + 0.06B$ 是 ITU-R BT.601 luma 系数），保证 white balance 不改整体亮度只调 channel balance。

**Sharpen/Blur**：一个温和的 blur kernel，$\theta_{sh}' > 1$ 高频增强（sharpening），$\theta_{sh}' < 1$ blur，参数范围 $[10^{-5}, 10]$ 跨 6 个数量级，log-scale 处理。

## 整体 take-away

**当你能一次 forward 出完整决策序列，且 reward 是 terminal 的，避免 stepwise critic 是值得优先考虑的设计**。

这是个 general RL insight。RLHF 里 DPO（Direct Preference Optimization）也是类似思路：绕开 critic 估计的复杂路径，直接用 final outcome 优化。POS-ISP 在 ISP 这个具体场景验证了 sequence-level formulation > stepwise MDP，未来类似结构化决策问题（API chaining、tool use pipeline、robotics primitive sequence）都可能受益。

参考延伸：
- AdaptiveISP: https://arxiv.org/abs/2411.09375
- ReconfigISP: https://arxiv.org/abs/2107.11539
- DRL-ISP: https://arxiv.org/abs/2207.10234
- DPO: https://arxiv.org/abs/2305.18290
- Sequence-level RL in NMT (Ranzato 2015): https://arxiv.org/abs/1511.06732
- Project page: https://w1jyun.github.io/POS-ISP/

---

# POS-ISP 深度解读

## 1. 这篇 paper 想解决什么问题

**Task-aware ISP optimization** —— 传统 camera ISP 把 RAW sensor data 转成 sRGB，是为 human visual perception 优化的 fixed pipeline（white balance → tone mapping → ...）。对下游 vision task（detection / segmentation / depth / enhancement）而言，fixed pipeline 远非最优。

低光 detection 场景里有个反直觉的现象：Camera ISP 输出的 JPEG 喂给 YOLOv3，mAP 只有 37.6；直接喂 minimally-processed RAW，mAP 反而是 44.1（Tab. 1 LOD-Dark）。说明为 human 优化的 ISP 其实 hurt machine vision。Task-specific 的 ISP 必须重新设计。

Modular ISP（把 pipeline 拆成 white balance / denoise / tone map / ... 这些经典算子）有 practical 优势：可解释、可部署在 hardware/firmware、计算量小。问题在于 **同时优化 sequence（哪些 module、什么顺序）+ parameters** 是 non-differentiable 的。这就是 POS-ISP 切入的地方。

参考 link：
- AdaptiveISP (NeurIPS 2024): https://arxiv.org/abs/2411.09375
- ReconfigISP (ICCV 2021): https://arxiv.org/abs/2107.11539
- DRL-ISP (IROS 2022): https://arxiv.org/abs/2207.10234

---

## 2. 现有方法的核心痛点

### 2.1 NAS-based (ReconfigISP) 的 train-inference mismatch

ReconfigISP 用 differentiable proxy 近似每个 ISP module，在搜索时对每个 stage 给所有候选 module 一个 learnable weight $\alpha$，stage 输出是 candidate modules 输出的 weighted sum：

$$y = \sum_i \alpha_i \cdot M_i(x)$$

可微了，但是 **训练时 mixed output，推理时 hard argmax 选一个**。DARTS 那一脉 NAS 都有这个毛病，weight 训练时和离散 selection 之间存在 gap。Tab. 1 ReconfigISP 47.8 mAP 反而不如 input RAW，可见这个 gap 在低光下很严重。

### 2.2 Stepwise RL (DRL-ISP, AdaptiveISP) 的不稳定

把 ISP pipeline 构造建模成 MDP：每个 stage 是一个 time step，state = 当前 image + 已选 module history，action = 下一个 module。用 actor-critic 训练。

致命问题在 **critic 的 bootstrapping**：

$$V(s_t) \leftarrow r_t + \gamma V(s_{t+1})$$

在 function approximation + bootstrapping + (off-policy) 三个条件同时存在时，是 Sutton 经典的 **deadly triad**，Q/value estimation 会 diverge。Paper Fig. S2 直接可视化出 critic loss 的剧烈震荡和 spike，作者把这个图作为 motivation。

更深一层：ISP 这种 reward 高度 noisy 的场景下，intermediate reward 很难定义。Dense prediction task（segmentation, depth）一个小像素偏差就能让 reward 剧烈变化，critic 拟合不准，actor 跟着飘。

外加 stepwise 每步都要 forward controller，DRL-ISP 训练 1 个 pipeline 要 15.71ms，AdaptiveISP 12.72ms（Tab. 4），inference 阶段 pipeline 重建就要这开销。

---

## 3. POS-ISP 的核心 reformulation

**把 stepwise MDP 拍扁成 sequence-level RL**：

- 一次 forward 预测整个 module sequence $\mathcal{A} = (a_1, ..., a_k)$
- 一次 forward 预测所有 module 的 parameters $\Theta$
- 跑完整 pipeline 得到 $I_{out}$
- 用 terminal task reward 反传

这样就 **不需要 critic、不需要 bootstrapping、不需要 stepwise intermediate supervision**。

这跟 NMT 早期 sequence-level RL 的工作（Ranzato et al. 2015 "Sequence Level Training Recurrent Networks"）思想类似：把 BLEU 这种 sequence-level metric 当 reward，避免 token-level cross-entropy 训练导致 exposure bias。

公式 (1) 定义 pipeline：

$$I_{out} = \big( \mathcal{M}_{a_k}(\cdot; \theta_{a_k}) \circ \cdots \circ \mathcal{M}_{a_1}(\cdot; \theta_{a_1}) \big)(I_{in}) = F(I_{in}; \mathcal{A}, \Theta)$$

变量解读：
- $\mathcal{M}_{a_i}(\cdot; \theta_{a_i})$: 第 $i$ 个 module，下标 $a_i \in \{1,...,n\}$ 是从候选池里选的 module index，$\theta_{a_i}$ 是这个 module 的参数
- $\circ$: 函数复合，从右到左执行：先 $\mathcal{M}_{a_1}$，最后 $\mathcal{M}_{a_k}$
- $\mathcal{A} = (a_1, ..., a_k)$: module 序列
- $\Theta = (\theta_{a_1}, ..., \theta_{a_k})$: 对应参数集合
- 假设 $a_i \neq a_j \text{ if } i \neq j$（每个 module 最多用一次），把搜索空间从 $n^k$ 缩到 $\binom{n}{k} k!$，对 tractability 很关键

---

## 4. Network Architecture 细节

### 4.1 Sequence Predictor：GRU + FiLM + Mask + Temperature

公式 (2) 自回归分解：

$$p(\mathcal{A}) = \prod_{i=1}^{T} p(a_i \mid a_{<i})$$

- $a_{<i} = (a_0, ..., a_{i-1})$，$a_0 = \langle sos \rangle$ 是 start token
- $\langle eos \rangle$ 允许变长 pipeline（最多到选完所有 module 或主动终止）

GRU 部分比较标准：$a_{i-1}$ → embedding → GRU → $h_i$。一个值得注意的细节是 **step embedding 用 FiLM 调制 hidden state**（公式 21）：

$$\tilde{h}_t = h_t \odot (1 + \gamma_t) + \beta_t$$

- $\gamma_t, \beta_t \in \mathbb{R}^H$ 来自一个 step embedding $s_t \in \mathbb{R}^{16}$（$t \in \{1,...,T\}$）通过小 MLP
- $\odot$ 是 element-wise 乘法
- 直觉：纯 GRU 在不同 step 的 hidden state 隐含位置信息，但 step embedding 显式告诉 decoder "现在是第几步"，因为 ISP 第 1 步应该做基础调整（exposure, white balance），第 5 步应该做精修（color correction, sharpening），不同位置的动作分布应该不同。FiLM 调制（Perez et al. 2018）是 conditional learning 的经典 trick

FiLM 原论文: https://arxiv.org/abs/1709.07871

**Mask**：每个 step 把已经选过的 module 的 logit 设 $-\infty$，softmax 后概率为 0。强制 $a_i \neq a_j$。这避免了重复选 denoise 这种 weird pipeline。

**Temperature-controlled sampling**（公式 22）：

$$\tau(t) = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \exp\left(-\ln 2 \cdot \frac{t}{h}\right)$$

- $\tau_{\max} = 2.5$, $\tau_{\min} = 0.2$, $h = 3000$ (half-life)
- softmax 变成 $\text{softmax}(z / \tau)$
- $\tau$ 大时分布平、探索多；$\tau$ 小时分布尖、exploit 主流 pipeline
- 半衰期 $h$ 的设计让温度从 2.5 降到 0.2 大概一半在 3000 步完成，整个训练 15000 步会经历多个半衰期，最终几乎 deterministic
- 这是 Boltzmann exploration（Cesa-Bianchi et al. 2017: https://arxiv.org/abs/1705.10257）的标准做法

**Decoder 初始化 trick**：weight matrix 设为 0，所有 bias 设同一常数，保证初始策略是 uniform distribution over actions。这个对称初始化避免早期训练偏向某个 module，加速收敛。

### 4.2 Parameter Predictor：超轻量 CNN

- $I \in \mathbb{R}^{3 \times H \times W}$ → 自适应 avg pool 到 $64 \times 64$
- 3 个 conv block + LeakyReLU → $F \in \mathbb{R}^{4C \times 8 \times 8}$
- **Global avg pool + Global max pool 拼接** → $8C$ 维 feature（双池化拼接是细粒度图像检索里证明有效的小 trick，结合了 smooth global context 和 peak activation）
- 2 层 MLP → latent $z \in \mathbb{R}^D$
- Decoder MLP → $P$ 维向量（所有 module 参数拼接）
- $\tanh$ + rescale 到 $[0, 1]$

总参数 0.53M（Tab. 4），相比 AdaptiveISP 7.18M 是 13× 缩减。

**关键设计决策**：parameter predictor 只 condition on image，不 condition on sequence。这个看似反直觉的决策，背后有深层原因。

Tab. S10 ablation：
- Sequence-Conditioned (SC) variant: LOD-Dark 47.5 ± 0.26, LIS-Dark 31.6 ± 0.33
- Image-Only (IO): 47.8 ± 0.08, 32.2 ± 0.05

Image-Only 不仅 mean 高，std 也小一半。

我的直觉解读：训练初期 sequence policy entropy 高（Tab. S6 显示 entropy 在 0.5 左右），sequence 几乎随机。如果 parameter predictor 依赖 sequence，每次 sequence 一变，参数空间的目标就跟着飘，parameter predictor 在追一个 moving target。Image-only 把 parameter predictor 限制在 "给定 image，对所有 plausible sequence 都还算 OK 的参数"，相当于 strong regularization。

后期 sequence policy entropy 降到 0（Fig. 5(b)），收敛到几个 dominant pipelines，parameter predictor 已经学了 average behavior，自然 fine-tune 到 dominant sequences。这就是 **implicit curriculum**：参数预测器先学 broad behavior，sequence policy 逐步收窄 search space，两者协同收敛。

类似的现象在 GAN 训练里也能看到：generator 和 discriminator 不能同时太强，否则互相干扰。

---

## 5. Reward 和 Optimization

### 5.1 Reward 定义

公式 (3)：

$$R(I_{in}, \mathcal{A}, \Theta) = \mathcal{L}_\mathcal{T}(I_{in}) - \mathcal{L}_\mathcal{T}(I_{out}) - P(I_{out})$$

变量解读：
- $\mathcal{L}_\mathcal{T}$: 下游任务 $\mathcal{T}$ 的 loss（比如 detection 时的 bbox regression + objectness + classification）
- $I_{in}$: 输入 RAW
- $I_{out} = F(I_{in}; \mathcal{A}, \Theta)$: pipeline 输出
- 第一项 $\mathcal{L}_\mathcal{T}(I_{in})$: 直接拿 RAW 喂下游任务的 loss，作为 baseline
- 第二项 $\mathcal{L}_\mathcal{T}(I_{out})$: 拿 ISP 输出喂下游任务的 loss
- $R$ 衡量 ISP 带来的 loss 改善，**这个 $\mathcal{L}_\mathcal{T}(I_{in})$ 起到了 implicit baseline 的作用**，把 absolute loss 转成 relative improvement

公式 (4) Penalty：

$$P = \alpha_1 [I_{low} - \bar{I}_{out}]_+ + \alpha_2 [\bar{I}_{out} - I_{high}]_+$$

- $\bar{I}_{out}$: 输出图像的平均强度
- $I_{low} = 0.01$, $I_{high} = 0.9$
- $[x]_+ = \max(0, x)$: hinge function
- 阻止 ISP 退化成 "全黑" 或 "全白" 这种 trivial solution
- 借鉴自 AdaptiveISP 的 truncation condition

**为什么 implicit baseline 有效**：标准 REINFORCE 是

$$\nabla J = \mathbb{E}[R \cdot \nabla \log \pi(a)]$$

如果 $R$ 是 absolute loss，方差大；减一个 baseline $b(s)$ 变成 advantage $A = R - b$，方差小但不改变梯度期望。这里把 reward 设计成 $R = L(I_{in}) - L(I_{out})$，相当于 $b = L(I_{in})$（per-image constant，不依赖 action），对每个 image 自动 normalize 了 reward，方差大幅降低。这是个非常 elegant 的设计选择。

### 5.2 优化目标

公式 (5) Sequence predictor 的 REINFORCE loss：

$$\mathcal{L}_{seq} = -\hat{\mathbb{E}}_{\mathcal{A} \sim \pi} \left[ R(I_{in}, \mathcal{A}, \Theta) \cdot \sum_{i=1}^{k} \log \pi(a_i) \right]$$

- $\hat{\mathbb{E}}$: mini-batch 上的经验期望
- $\pi(a_i)$: step $i$ 选 $a_i$ 的概率
- $\sum_{i=1}^{k} \log \pi(a_i)$ 是整条 sequence 的 log-probability（独立假设下求和等于联合 log-prob）
- 负号因为我们要 maximize expected reward，loss 是 minimize negative
- 标准 policy gradient，没有 critic，没有 advantage function

公式 (6) Parameter predictor 的 supervised loss：

$$\mathcal{L}_{param} = \mathcal{L}_\mathcal{T}(I_{out}) + P(I_{out})$$

- 通过 backprop 直接优化
- pipeline 里大部分 module 是可微的（exposure, gamma, tone map, contrast, white balance, color correction 都可微；denoise/sharpen 也都设计成可微形式）
- 这个 loss 等价于 maximize reward（少了 $L(I_{in})$ 这个常量项）

### 5.3 交替优化

Algorithm：
1. 采样 sequence $\mathcal{A} \sim \pi$
2. 用 parameter predictor 得 $\Theta = \Theta(I_{in})$
3. 跑 pipeline 得 $I_{out}$
4. 算 reward $R$
5. 用 REINFORCE 更 sequence predictor（一次 forward，梯度通过 $\log \pi$ 反传）
6. 用 backprop 更 parameter predictor（梯度通过 pipeline 反传到 parameter predictor 的 weights）
7. 交替重复

这里一个 subtle 的点：sequence 是离散的，从 $\pi$ 采样后是 hard selection，梯度无法通过 sample 反传到 sequence predictor。所以用 REINFORCE（score function estimator）。Parameter 是连续的，可以 reparameterize，梯度直接 backprop。两个 predictor 用不同的优化策略，挺合理的混合设计。

---

## 6. 关键设计决策的 Ablation 解读

### 6.1 Single Sequence per Task + Image-Adaptive Parameters

这是 paper 最有 "engineering insight" 的设计：每个 task 只学**一个** sequence $\hat{\mathcal{A}}$，所有 image 用同一个 sequence，但 parameters 是 image-adaptive 的。

推理流程：
1. 训练完，从 sequence predictor greedy decode（argmax）出 $\hat{\mathcal{A}}$
2. $\hat{\mathcal{A}}$ 固定到 firmware 里
3. 推理时只跑 parameter predictor 一次 → 得到 $(\hat{\theta}_1, ..., \hat{\theta}_n)$
4. 从中选 $\hat{\Theta} = (\hat{\theta}_{a_1}, ..., \hat{\theta}_{a_k})$
5. 跑 pipeline 出 $I_{out}$

**两个 motivation**：

1. **硬件现实**：camera ISP 的 module 顺序是 baked into silicon/firmware 的，不能 per-image reconfigure。但 module 内部参数可以 per-image 调整。这个 design 跟实际 camera 的部署 model 对齐。

2. **Task-sequence 耦合**：sequence order 主要由 task 决定。Detection 这种依赖 structural info 的，contrast/sharpening 倾向于靠前；perceptual quality 任务倾向把 exposure/tone 放前面。

Tab. S11 的 cross-task 验证很有说服力：

| Source Task (Sequence) | mAP@0.5:0.95 |
|---|---|
| Image Enhancement | 47.6 |
| Instance Segmentation | 47.1 |
| **Object Detection** | **48.0** |

用 detection 自己的 sequence 跑 detection mAP 48.0；用 enhancement 的 sequence 跑 detection 只剩 47.6。差距看似不大但 consistent，说明 sequence 确实 task-specific，光优化参数补不回来。

Tab. S12 cross-domain 验证参数 image-adaptive：
- LOD-Dark 训练的参数用 LOD-Dark 测试：53.6 mAP
- LOD-Normal 训练的参数用 LOD-Dark 测试：52.6 mAP

下降 1 mAP 说明 parameter predictor 学的不是 domain-agnostic average，确实对 illumination 敏感。

Fig. S3 的 histogram 也显示：Dark 图像预测的 exposure/tone 参数偏强 brightness compensation，Normal 图像偏弱。这种分离是 parameter predictor 自然学到的，没显式 supervise。

Tab. S7 看 robustness：用 $\Delta < 0$ 的比例（ISP 后性能下降的图像占比）和 worst 5% mean drop 衡量：

| Method | LOD-All $P(\Delta < 0)$ ↓ | Worst 5% Mean ↑ |
|---|---|---|
| DRL-ISP | 0.185 | -0.28 |
| ReconfigISP | 0.252 | -0.31 |
| AdaptiveISP | 0.135 | -0.18 |
| **Ours** | **0.123** | **-0.16** |

POS-ISP 最低 12.3% 图像 performance 下降，且最坏情况 drop 最小。说明单 sequence + per-image params 这个解耦给了足够的 flexibility，没有因为 "一个 sequence 用所有 image" 而 collapse。

### 6.2 Pipeline Length 的影响

Tab. S6 把 POS-ISP 强制 length=5（跟 AdaptiveISP 一样）：

| Method | Length | LOD-Dark mAP |
|---|---|---|
| AdaptiveISP | 5 | 47.1 |
| POS-ISP | 5 | 47.4 |
| POS-ISP | ≤10 | 47.8 |

固定 5 也已经赢 AdaptiveISP，说明性能优势不是来自 pipeline 更长。Dynamic length 再多 0.4，是 nice-to-have 但非必要。这把 "性能来自架构还是长度" 这个 confounder 干净消除了。

### 6.3 GRU vs Probability Table (Tab. 5)

Probability table 是一个 $T \times n$ 的 learnable matrix，每个 step 独立采样，不考虑前面选了什么。这是个 strong baseline，因为它能学 "step 1 倾向于选 module X" 这种 marginal 信息。

GRU 提升约 0.3-0.8 mAP：
- LOD-Dark: 47.5 → 47.8
- LIS-Dark: 31.3 → 32.1

LIS-Dark 提升更大（+0.8），segmentation 任务对 module 顺序可能更敏感。这证实了 inter-module dependency 的建模价值。

直觉解释：white balance 应该在 color correction 之前；denoise 应该在 sharpening 之前（不然 sharp 完再 denoise 会模糊细节）。这种顺序约束需要 sequence model 捕捉。Probability table 看不到上下文，会采出 "sharpen → denoise" 这种 suboptimal order。

---

## 7. 实验数据全面解读

### 7.1 Object Detection (LOD-Dark, Tab. 1)

| Method | mAP@0.5:0.95 |
|---|---|
| Input RAW | 44.1 |
| Camera ISP | 37.6 |
| DRL-ISP | 44.2 |
| ReconfigISP | 43.7 |
| AdaptiveISP | 47.2 |
| **POS-ISP** | **47.8** |

Camera ISP 反而 hurt performance（37.6 vs 44.1），因为它为 human eye 调，加了大量 noise reduction + sharpening + color enhancement，破坏了 detection 需要的 structural info。

ReconfigISP 输给 input RAW，train-inference mismatch 在低光下放大（low SNR 下 hard selection 的 module 行为跟 soft mixed 差异更大）。

POS-ISP 比 AdaptiveISP 提升 0.6 mAP，主要来自更稳定的优化（multi-seed std 0.08 vs AdaptiveISP 0.25，Tab. 3）。

### 7.2 Instance Segmentation (LIS-Dark, Tab. 2)

| Method | mAP@0.5:0.95 |
|---|---|
| Input RAW | 27.8 |
| Camera ISP | 20.1 |
| DRL-ISP | 27.1 |
| ReconfigISP | 24.2 |
| AdaptiveISP | 25.2 |
| **POS-ISP** | **32.1** |

这里 POS-ISP 比 AdaptiveISP 高 **6.9 mAP**，差距远大于 detection 任务的 0.6。这是 paper 最 striking 的结果。

为什么 segmentation 提升这么大？Segmentation 是 dense prediction，pixel-level supervision。一个小像素偏差就能让 mask loss 剧烈变化，reward variance 极高。Stepwise RL 在这种 high-variance reward 下 critic 学不准，actor 跟着飘。POS-ISP 用 terminal reward + implicit baseline，绕开了 critic instability，所以收益更明显。

AdaptiveISP 在 LIS-Dark 上甚至比 input RAW 还低（25.2 vs 27.8），说明 stepwise RL 在 segmentation 这个 noisy reward 任务下确实崩了。

### 7.3 Computational Efficiency (Tab. 4)

| Method | Params (M) | MACs (M) | Peak GPU Mem (MB) | Runtime (ms) |
|---|---|---|---|---|
| DRL-ISP | 6.57 | 155.3 | 1013.9 | 15.71 |
| AdaptiveISP | 7.18 | 70.2 | 39.6 | 12.72 |
| **POS-ISP** | **0.53** | **15.1** | **14.4** | **1.55** |

Params 减到 1/13，MACs 减到 1/10，Peak memory 减到 1/70。Runtime 减到 1/10。这是 sequence-level formulation 的直接红利：推理时 sequence 已经 fixed（greedy decode 一次），只跑 parameter predictor 一次 forward，没有 stepwise controller 的重复开销。

### 7.4 End-to-end Runtime (Tab. S4)

不加 runtime penalty，POS-ISP 默认会预测长 pipeline（length 10），end-to-end 82.57ms（pipeline 执行慢）。

加 runtime penalty（跟 AdaptiveISP 一样的 trick），length 降到 3，end-to-end 1.61ms，mAP 只掉 0.1（47.8 → 47.7）。这个 knob 实用价值很高，给部署灵活性。

### 7.5 On-device Performance (Tab. S5)

Galaxy S10 CPU, FP32, 单线程，无量化：

| Method | End-to-end (ms / FPS) |
|---|---|
| AdaptiveISP | 115.0 / 8.70 |
| POS-ISP (w/ penalty) | 32.4 / 30.86 |

**30 FPS 真实时**。在 2019 年的 mobile CPU 上跑 real-time task-aware ISP，deployment 价值极大。

### 7.6 Stability (Tab. 3, Fig. 5)

3-seed std：POS-ISP 0.08-0.16，AdaptiveISP 0.15-0.57，DRL-ISP 0.20-0.47。

Fig. 5(a) 训练曲线：POS-ISP 平滑单调上升，AdaptiveISP 早期有剧烈震荡。

Fig. 5(b) Policy entropy 从 0.5 平滑降到 0，最终 pipeline likelihood 比初始高 20-60×。说明 policy 收敛到 confident 的 pipeline，且收敛过程稳定。

### 7.7 Cross-Task / Cross-Domain Generalization

Tab. S8（YOLOv13 detector，跟主 paper 用的 YOLOv3 不同）：
- LOD-Dark: POS-ISP 38.1, AdaptiveISP 36.8
- LOD-All: POS-ISP 48.1, AdaptiveISP 47.3

Reward backbone 换一个 detector，POS-ISP 仍稳定提升。说明优化框架 backbone-agnostic。

Tab. S9（Depth estimation with different losses: RMSE / AbsDiff / AbsRel）：
- 三个 loss 下 POS-ISP 都全面优于 AdaptiveISP
- 说明 sequence-level formulation 对 reward function 形式 robust

---

## 8. ISP Module 池细节

Tab. S13 列了 10 个 module，每个有公式和参数范围。挑几个有技术含量的讲：

### 8.1 Tone Mapping (公式 5)

$$\mathbf{I}' = \frac{8}{Z} \sum_{i=1}^{8} \theta_{t,i}' b_i(\mathbf{I}), \quad Z = \sum_{i=1}^{8} \theta_{t,i}'$$

- $b_i(u) = \max(0, \min(u - \frac{i-1}{8}, \frac{1}{8}))$: 8 个 piecewise-linear basis function，把 $[0,1]$ 切成 8 段
- $\theta_t' \in [0.5, 2.0]^8$: 8 个段的权重
- $Z$: 归一化因子，保证 sum of weights = 1
- 这本质是 piecewise-linear curve 的 8 个 control point，用 weights 控制每段斜率
- 参数维度 8，最复杂的一个 module

### 8.2 White Balance (公式 15, 16)

$$\tilde{\theta}_{wb}' = \frac{\theta_{wb}'}{0.27 \theta_{wb,R}' + 0.67 \theta_{wb,G}' + 0.06 \theta_{wb,B}'}$$

$$\mathbf{I}' = \text{diag}(\tilde{\theta}_{wb,R}', \tilde{\theta}_{wb,G}', \tilde{\theta}_{wb,B}') \mathbf{I}$$

- $\theta_{wb}' \in [1/1.1, 1.1]^3$: 3 个 channel 的 raw gain
- 分母是 luminance-weighted normalization：$0.27 R + 0.67 G + 0.06 B$ 是 luma 系数（ITU-R BT.601）
- 这个 normalization 保证 white balance 不改变整体 brightness，只调 channel balance
- $[1/1.1, 1.1]$ 这个窄范围避免极端 white balance

### 8.3 Contrast (公式 6, 7, 8)

S-curve 是经典 contrast enhancement trick：

$$I_S = \frac{1 - \cos(\pi I_{lum})}{2}$$

- $I_{lum} \in [0,1]$, $\cos(\pi \cdot 0) = 1$ → $I_S = 0$；$\cos(\pi \cdot 0.5) = 0$ → $I_S = 0.5$；$\cos(\pi \cdot 1) = -1$ → $I_S = 1$
- 中间段斜率高，两端饱和，标准 S-curve
- 然后用 $\theta_c'$ blend 原图和 S-curve 版本

### 8.4 Sharpen/Blur (公式 18, 19)

$$K = \frac{1}{13} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 5 & 1 \\ 1 & 1 & 1 \end{bmatrix}, \quad \mathbf{I}' = \theta_{sh}' \mathbf{I} + (1 - \theta_{sh}') \mathbf{I}_{blur}$$

- $K$ 是个温和的 blur kernel
- $\theta_{sh}' > 1$: 高频被增强（sharpening）
- $\theta_{sh}' < 1$: blur
- $\theta_{sh}' \in [10^{-5}, 10]$: 范围 6 个数量级，log-scale 处理

---

## 9. 跟其他研究方向的连接

### 9.1 Sequence-level RL in NMT

Ranzato et al. 2015 "Sequence Level Training Recurrent Networks with Recurrent Neural Networks" (https://arxiv.org/abs/1511.06732) 把 BLEU 当 sequence-level reward 训练 NMT，避免 token-level cross-entropy 的 exposure bias。POS-ISP 借鉴这个思想：把 pipeline 当 sequence，task performance 当 reward，整个 pipeline 一次预测。

### 9.2 Differentiable NAS 的 train-test gap

DARTS (Liu et al. 2019, https://arxiv.org/abs/1806.09055) 也是 soft mixing 训练 + hard selection 推理，gap 是 NAS 长期痛点。PC-DARTS, SPOS 等后续工作尝试弥补。ReconfigISP 沿用 DARTS 思路在 ISP 上做 NAS，自然继承这个问题。POS-ISP 用 RL 直接做 hard selection，绕开 mismatch。

### 9.3 Deadly Triad in RL

Sutton & Barto 的经典 RL 教科书里讲过：function approximation + bootstrapping + off-policy 三者同时存在时 value function 容易 diverge。Paper 引用 Fujimoto et al. 2018 (TD3, https://arxiv.org/abs/1806.09474) 和 van Hasselt et al. 2018 (https://arxiv.org/abs/1810.10275) 讨论 deadly triad。POS-ISP 通过移除 critic 直接消灭 triad 中的一个 component，自然稳定。

### 9.4 Modular Neural Architecture Search

类似思想在 NAS-for-pipeline 也出现过，比如 Tran et al. "AutoML for Preprocessing Pipeline"（https://arxiv.org/abs/2104.05061）用 RL 找 sklearn pipeline。ISP 这种结构化、约束清晰的 pipeline 比通用 ML pipeline 更适合 sequence-level RL。

### 9.5 Light-weight ISP

DeepISP (https://arxiv.org/abs/1804.08596), ParamISP (CVPR 2024, https://arxiv.org/abs/2403.17394) 等 end-to-end neural ISP 参数量很大，移动端难部署。POS-ISP 0.53M params + 30 FPS on Galaxy S10 是 deployment-friendly 的关键卖点。

### 9.6 Task-aware Image Processing

Tseng et al. 2019 (https://arxiv.org/abs/1906.06504) 用 differentiable proxy 优化 ISP hyperparameters，Qin et al. 2022 (https://arxiv.org/abs/2110.14091) 加 attention。这些方法用 proxy network 模拟 ISP，然后 gradient descent。POS-ISP 直接跑 ISP，用 task loss + REINFORCE，没有 proxy approximation error。

---

## 10. 局限和 Future Work（作者自己提的）

1. **Search space 扩大困难**：candidate modules 数量增加，搜索空间指数膨胀，收敛时间增长。可以考虑 hierarchical search（先选大类，再选具体 module）。

2. **Multi-task 不友好**：每个 task 要单独训练 policy。Multi-task 场景（一个 camera 同时跑 detection + segmentation + enhancement）需要多次独立训练，system 复杂度高。Future work 提到 unified multi-task model，可能用 multi-head policy + shared backbone。

3. **REINFORCE 仍然 high variance**：虽然 implicit baseline 降低了方差，但本质还是 score function estimator。如果 search space 进一步扩大，可能需要 PPO 那种 importance sampling + clipping，或者 variance reduction techniques（baseline network，但要小心不要重新引入 critic instability）。

4. **Pipeline length penalty 调参敏感**：Tab. S4 显示加 penalty 后 length 从 10 降到 3，性能几乎不变。但 penalty 系数需要手调，对不同 task 可能不同。Auto-tuning penalty 是个可能方向。

---

## 11. 直觉总结

POS-ISP 的核心 insight：**把 stepwise decision 改成 sequence-level prediction**，绕开 RL 中最不稳定的 critic 估计，同时一次 forward 出整个 pipeline 大幅降低推理开销。

具体 intuition：
1. ISP pipeline 的 sequence 决策其实是个 sequence-to-sequence 问题（用前面 module 决定后面），天然适合 autoregressive model
2. Pipeline 整体性能由 final task metric 决定，intermediate reward 是 artificial 的，反而引入 noise
3. Reward 设计成 relative improvement $L(I_{in}) - L(I_{out})$ 起到 implicit baseline 作用，让 vanilla REINFORCE 也能稳定 work
4. Single sequence per task + image-adaptive params 的解耦既匹配硬件 reality，又给足够 flexibility
5. Parameter predictor 不 condition on sequence 是 implicit curriculum 的自然结果，避免 moving target

这个工作其实在 general RL 上有个 take-away：**当你能完整一次 forward 得到 sequence 决策，并且 reward 是 terminal 的，避免 stepwise critic 是值得优先考虑的设计**。类似思想在 RLHF 的 sequence-level training 里也在兴起（参考 https://arxiv.org/abs/2305.18290, Direct Preference Optimization），都是绕开 critic 估计的复杂路径，直接用 final outcome 优化。

类似 paper 推荐延伸阅读：
- "Meta-learning for Semi-Automated Pipeline Design": https://arxiv.org/abs/2104.05061
- "Boltzmann Exploration Done Right": https://arxiv.org/abs/1705.10257
- "Addressing Function Approximation Error in Actor-Critic Methods (TD3)": https://arxiv.org/abs/1806.09474
- "Direct Preference Optimization": https://arxiv.org/abs/2305.18290
- "AdaptiveISP": https://arxiv.org/abs/2411.09375
- Project page: https://w1jyun.github.io/POS-ISP/
