---
source_pdf: RDT-1B.pdf
paper_sha256: fabfa885b63f2d68b8e7868549563c346d4323a596d24ee3e43d0a62b4ef9667
processed_at: '2026-08-11T21:05:47-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RDT-1B 用人话讲一遍

## 一句话概括

清华团队把 Diffusion Transformer 这个 image generation 的 architecture 搬到 robot 上，scale 到 1.2B 参数，在 46 个数据集上 pretrain，最后在 ALOHA 双臂上 finetune，效果吊打 ACT、OpenVLA、Octo 三个 baseline。

## 核心问题：为什么双臂这么难

单臂 robot 的问题大家已经折腾好几年了，ACT、Diffusion Policy 都能跑。双臂听起来就是"再来一只手"，但实际难度是指数级跳的，原因有三个：

**第一个，action distribution 变成 multi-modal。** 这是最致命的。单臂抓杯子，正确解法可能就那么几种，distribution 主峰明显。双臂呢——左手抓还是右手抓？左手扶右手倒？两只手一起抓？每种组合再叠加各种 joint trajectory，distribution 就裂成好几个峰。你用 MSE regression 去拟合，model 就会输出所有 peak 的算术平均，物理上完全不可行（比如左手该往左、右手该往右，平均下来两只手都不动）。

这就是为什么 ACT（用 VAE）在双臂上吃力——VAE 的 single Gaussian latent 把 multi-modal 压成单峰。OpenVLA 更惨，它把 14 维 action 离散化成 token，精度损失 + 长程 token 依赖直接让它收敛不了，论文 Fig. 8 显示 action token accuracy 在 60% 徘徊，部署就是 random behavior。

**第二个，数据稀缺。** ALOHA 一套硬件几万美金，采集一条 trajectory 要人 teleoperation，搞到 6K 条已经是极限。foundation model 要的数据量级是百万级，差三个数量级。

**第三个，跨机器人数据异构。** 你想从 Open X-Embodiment 那 46 个数据集借数据，但每个 robot 的 action space 都不一样——有的 6 DoF joint，有的 7 DoF，有的 EEF control，有的 base + arm。直接混训会 negative transfer。

## 他们的解法：四个 trick 串成一个 system

### Trick 1: 用 Diffusion modeling 解决 multi-modality

这个 idea 不新，Diffusion Policy (Chi et al. 2023) 早做过了。核心 insight 是 diffusion 天然能 represent multi-modal distribution——reverse process 可以 sample 出不同 mode，不需要把分布压成单峰。

公式 1 是 reverse process：

$$
\mathbf{a}_t^{k-1} = \frac{\sqrt{\bar{\alpha}^{k-1}} \beta^k}{1 - \bar{\alpha}^k} \mathbf{a}_t^0 + \frac{\sqrt{\alpha^k}(1 - \bar{\alpha}^{k-1})}{1 - \bar{\alpha}^k} \mathbf{a}_t^k + \sigma^k \mathbf{z}
$$

人话翻译：从纯噪声 a_t^K 出发，每一步把噪声去掉一点，K 步后得到 clean action a_t^0。每一项的意思——

- 第一项：往 x_0（干净 action）方向靠的权重，ᾱ^k 是累积的信号保留率，越往后越大
- 第二项：保留当前 noisy state 的权重，跟第一项互补
- 第三项：随机噪声扰动，避免 mode collapse 到单一解

公式 2 是 training loss：

$$
\mathcal{L}(\theta) = \text{MSE}(\mathbf{a}_t, f_\theta(\ell, \mathbf{o}_t, \sqrt{\bar{\alpha}^k}\mathbf{a}_t + \sqrt{1-\bar{\alpha}^k}\boldsymbol{\epsilon}, k))
$$

人话：随机加噪声到 action，让 network 预测原始 clean action，MSE loss。

这里有个细节——RDT 用的是 **x_0-prediction**（直接预测 clean action），不是 DDPM 原版的 ε-prediction。理由是 robot action 维度低（128D unified space），不像 image 几万维，x_0-prediction 数值稳定性没问题，而且 inference 时省一次 ε→x_0 转换。

Diffusion 在 image 上慢是因为维度高采样慢，但 action 才 128 维，DPM-Solver++ 压到 5 步，onboard RTX 4090 跑 6Hz chunk frequency，完全够用。这是 robot domain 用 diffusion 的 unique advantage。

### Trick 2: 改 DiT 的三个地方让它能训

vanilla DiT (Peebles & Xie 2023) 是为 image 设计的，直接搬到 robot 数据上会 explode（Fig. 4a）。原因：robot 物理量数值范围不稳定、有 high-frequency 跳变（碰撞瞬间速度突变、gripper 突然 close），跟 image 的 spatio-temporal continuity 完全不一样。

三个修改：

**QKNorm + RMSNorm**。QKNorm 在 attention 的 Q、K 上加 L2 normalization，防止 attention logit 爆炸。RMSNorm 替代 LayerNorm——关键差异是 RMSNorm 不做 centering（不减均值）。为什么不减？把 robot task 看成 time series forecasting，centering 会让 chunk 内的 action token 产生相对位移，破坏时序对称性。论文引用 UnitNorm (Huang et al. 2024) 的分析。实测不加这俩，1B 规模 pretrain 后期 loss 直接 explode。

**MLP Decoder**。原版 DiT 用线性层把 latent 投回 action space。RDT 换成 MLP。理由：robot action 是 joint angle + cartesian pose + gripper state 混在一起，物理量纲不同，回映需要非线性 warp。线性 decoder 表达不了这种非线性。

**Alternating Condition Injection (ACI)**。这是最巧的设计。DiT 原版用 adaptive layer norm（adaLN）注入 condition——把 class label 压成一个 token 调制每个 block。但 RDT 的 condition 是 image 和 language，**变长**且**信息密度不对称**。SigLIP 输出几百个 image patch token，T5-XXL 输出几十个 text token。如果同层 cross-attention 同时注入，image token 数量优势会让 attention sink 到 image，text 信号被淹没。

ACI 的 solution：奇数层 cross-attend image，偶数层 cross-attend text，交替进行。text 在 dedicated layer 拿到完整的 attention capacity，不被 image 抢资源。

消融实验（Fig. 4b）特别有说服力——去掉 ACI，Pour Water-L-1/3 任务的 "correct amount" 子指标从 100% 掉到 12.5%。这个任务需要精确理解 "one-third" 这个量词，纯靠 image 不够，必须有 language grounding，ACI 一去掉 language 信号就废了。

### Trick 3: Physically Interpretable Unified Action Space

解决跨机器人数据异构。设计一个 128 维的 unified action space，每个维度有明确物理语义：

```
[0, 10):   右臂 joint positions
[10, 15):  右夹爪 joint positions
[15, 25):  右臂 joint velocities
...
[50, 100): 左臂对称
[100, 103): base velocity
```

不同 robot 按 physical meaning 对齐填充，6 DoF 机器人就填前 6 位，其余 padding。单臂 robot 映射到右臂位，左臂全 padding。

这里有个 trick——padding 不能直接用 0，因为 "0 velocity" 物理上意味着静止，model 会困惑 "0 是静止还是不存在 sensor"。Solution 是 concat 一个 0-1 mask vector，告诉 model 每个维度是 valid 还是 padding。最终输入是 256 维（128 value + 128 mask）。

另一个反直觉的设计：**不做严格 normalization**。多数 prior work 把 action 归一化到 [-1, 1] 或 N(0,1)，但 RDT 只统一物理单位（m, rad, m/s）。理由：跨 robot 时 "1 m" 物理意义一致，归一化会破坏这个 shared prior，损害迁移。数值稳定性丢给 RMSNorm + QKNorm 处理。

### Trick 4: 大规模 pretrain + 小规模 finetune

pretrain 数据：46 个数据集，1M+ trajectories，21TB。主要来自 Open X-Embodiment，采样权重用 √N（N 是数据集大小）平衡大小数据集，避免大数据集 dominate。

finetune 数据：Mobile ALOHA 上自己采的 6K+ trajectories，300+ tasks，100+ objects，15+ 房间。language instruction 用 GPT-4-Turbo 扩展（每条原始 instruction 扩成 100 条 paraphrase + 1 条简化版），模仿 LLM instruction tuning 提升 language robustness。

训练成本：48 张 H100 pretrain 1 个月（1M steps），finetune 3 天（130K steps）。

## 实验告诉我们什么

7 个 task 覆盖 5 个维度——unseen object、unseen scene、instruction following、few-shot learning、dexterity。

主结果（Table 3）：RDT 平均 success rate 比 baseline 高 56%。OpenVLA 几乎全 0%，ACT 在 seen 数据上能跑但 unseen 直接崩，Octo 也不行。

几个有 insight 的数据点：

**Pour Water-L-1/3 vs Pour Water-R-2/3**：训练数据里只有 "little / half / full" 三个水位，测试 "1/3" 和 "2/3" 两个没见过的量词。RDT 的 "correct amount" 子指标 100% 和 75%。这说明 model 学到了 "fraction" 的 compositionality，不是死记 "half = 某个角度"。这是 language grounding 真正 work 的证据。

**Handover（5-shot）和 Fold Shorts（1-shot）**：预训练 + 1-5 个 demo finetune 就能学会全新 skill。这跟 LLM 的 in-context learning 不一样，是 lightweight finetune 的 meta-learning setting。pretraining 提供了 visual + physical prior，few-shot data 只是 task-specific 适配。

**Robot Dog**：推 joystick 让机器狗走直线。ACT 失败因为 joystick 跟 remote control 都是黑色，小 model 学不出 visual concept。RDT 靠 pretraining 见过海量 object，joystick 的 vision-language representation 学得好。这个 task 是 dexterity 的极限测试——joystick 角度差几度，机器狗就走偏。

**消融**（Table 2）：
- 去掉 diffusion（用 regression）：unseen object 12.5%，instruction following 12.5%。multi-modality 用 regression 就是死路。
- 小模型（166M）：instruction following 从 100% 掉到 25%。scale 对 language grounding 重要。
- 无 pretrain：unseen object 0%，unseen scene 25%。pretrain 主要贡献 visual generalization。

有意思的是无 pretrain 的变体在 instruction following 上还有 62.5%——暗示 pretrain 负责 visual + physical prior，finetune 负责 task-specific language grounding。两者分工明确。

## 我的直觉判断

### ACI 是个 generalizable pattern

ACI 的本质是 cross-attention 里对 condition modality 做 round-robin scheduling。这个 idea 可以推广到很多地方——多视角 image 之间 ACI、long context vs short context ACI、甚至 multi-document reasoning 时不同 document 轮流注入 attention。任何多模态 imbalance 的场景都能用。这是个 scalable design pattern，我觉得会被后续很多 paper 借鉴。

### Unified Action Space 的局限

128 维是 hand-crafted 的，针对 "robot with gripper arms"。对 humanoid（腿 + 身 + 双臂 + 头）、continuum manipulator（柔性机器人）、force-controlled robot 都不适用。未来方向可能是 learnable unified space——让 model 自己学跨 embodiment 的 latent action representation。但这会牺牲 physical interpretability，而论文 claim interpretability 是 transferability 的关键。这个 trade-off 值得研究。

### 5 步 diffusion 够不够

DPM-Solver++ 把 1000 步压到 5 步，但 mode coverage 理论上会差。5 步可能只能 sample 出主导 mode，丢长尾。Robot task 里如果需要"创意性"操作（多种抓取姿态中选少见但更优的），5 步可能不够。这个 paper 没分析。

### Robot foundation model 的 bottleneck 在 data 不在 model

RDT 是 robot domain 的 "GPT-2 时刻"——1B 参数已经很大，但相对 LLM 的 100B+ 还差两个数量级。Architecture 层面 RDT 基本 solved，剩下的 gap 在 data。LLM 可以 web crawl，robot data 必须实机采集。cross-embodiment pretrain 暂时绕过瓶颈，但 long-term 看 sim-to-real 和大规模 data collection 是 fundamental constraint。

Karpathy 你之前关注 robot data 是对的——下一篇大 paper 大概率在 sim-to-real 或者新硬件范式（mass teleoperation、UR training deploy 到 ALOHA）上突破，而不是 architecture 创新。

### 一个被低估的细节：control frequency 作为显式输入

不同数据集 control frequency 不一样（RT-1 是 3Hz，DROID 是 15Hz，RH20T 是 10Hz）。多数 prior work resample 到统一频率，损失信息。RDT 直接把 frequency c 喂给 model，让 model 自己适配。这个设计简单但关键——model 学到 "在 15Hz 下 0.1 秒移动 5cm" 和 "在 3Hz 下 0.1 秒移动 5cm" 是不同物理含义的动作。这是 physical reasoning 的一部分。

## 关键 reference

- RDT-1B paper: https://arxiv.org/abs/2410.07864
- Project page: https://rdt-1b.github.io/
- Code: https://github.com/thu-ml/RoboticsDiffusionTransformer
- DiT: https://arxiv.org/abs/2212.09748
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ACT / ALOHA: https://tonyzhaozh.github.io/aloha/
- OpenVLA: https://openvla.github.io/
- Octo: https://octo-models.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Mobile ALOHA: https://mobile-aloha.github.io/
- DPM-Solver++: https://arxiv.org/abs/2211.01095
- SigLIP: https://arxiv.org/abs/2303.15343
- T5: https://arxiv.org/abs/1910.10683
- 6D rotation representation: https://arxiv.org/abs/1812.07035
- UnitNorm (time series normalization): https://arxiv.org/abs/2405.15903

---

# RDT-1B 深度解析：双臂操作的 Diffusion Foundation Model

## 一、Paper 的核心 thesis

这篇 paper 来自清华朱军团队，做的事情是在 bimanual manipulation（双臂操作）这个场景下，用 Diffusion Transformer 路线把 model scale 推到 1.2B 参数，并解决三个相互耦合的工程难题：

1. **Action distribution 的 multi-modality**：双臂比起单臂，action space 翻倍，但更重要的是同一个 task 的解法数量组合式增长（左右手可以互换、可以协调抓握、可以一只手稳定一只手操作），导致 conditional action distribution p(a_t | ℓ, o_t) 出现严重多模态。如果用 MSE 回归去拟合，就会回归到所有 mode 的"平均"，物理上不可行（比如两个 mode 一个往左一个往右，平均就是不动）。

2. **Data scarcity**：双臂硬件成本远高于单臂，一个特定双臂机器人能采集到的 trajectory 通常 < 10K 条，远低于 foundation model 的数据需求。Solution：cross-embodiment pretraining，把 Open X-Embodiment 里 46 个数据集（1M+ trajectories, 21TB）拉进来 pretrain，数据量放大 3 个数量级。

3. **Heterogeneity across robots**：不同机器人的 action space 维度、语义、物理量完全不一致。Solution：设计一个 physically interpretable unified action space（128 维），按物理语义对齐填充。

核心 claim 是：Diffusion modeling + 大模型 + 多机器人预训练 + 统一 action space 这四个要素缺一不可，缺了任何一个 generalizability 就塌了。

Reference links:
- arXiv: https://arxiv.org/abs/2410.07864
- Project page: https://rdt-1b.github.io/
- Code: https://github.com/thu-ml/RoboticsDiffusionTransformer

---

## 二、Problem Formulation 的精确拆解

### 2.1 输入输出定义

给定语言指令 ℓ、时间步 t 的 observation o_t := (X_{t-T_img+1:t+1}, z_t, c)，policy 输出 action a_t 控制双臂。

变量含义：
- **X_{t-T_img+1:t+1}** := (X_{t-T_img+1}, ..., X_t)，RGB observation history，长度 T_img=2（论文里实测 2 帧就够，多了 overhead 不划算）
- **X_t := {X_t^1, X_t^2, X_t^3}**：三视角图像，分别对应 exterior（外部俯视/前视）、right-wrist（右手腕相机）、left-wrist（左手腕相机）
- **z_t**：低维本体感觉（proprioception），包含 joint position、joint velocity、gripper state、end-effector pose 等
- **c**：control frequency，不同数据集频率不一致（3Hz / 10Hz / 15Hz），论文把 c 显式喂给 model，让 model 自己做频率适配。这是一个被低估但很关键的设计
- **a_t**：通常是 desired proprioception z_{t+1} 的子集

### 2.2 Action chunking

不预测单步 a_t，而是预测 chunk a_{t:t+T_a} := (a_t, ..., a_{t+T_a-1})，T_a = 64。

这个 64 不是随便选的，参考 ACT (Zhao et al. 2023) 的 ablation。chunking 解决两件事：
- **Temporal consistency**：相邻动作有显式耦合，避免单步预测出现帧间不连续的"抽搐"动作
- **Covariate shift / error accumulation**：减少整条 trajectory 中的决策次数，避免 policy 在 compounding error 下漂移到 OOD 区域（参考 DAgger 论文 Ross et al. 2011）

### 2.3 Pre-train + Fine-tune 范式

数据集记为 D = {(ℓ^(i), o_t^(i), a_t^(i)) | 0 ≤ t < T^(i), 1 ≤ i ≤ N}。先在 D_pre（46 个多机器人数据集，多为单臂）预训练，再在 D_ft（Mobile ALOHA 双臂数据集，6K+ trajectories）微调。

这里有个隐含的设计选择：pretraining 用 cross-embodiment 但不是 cross-task generalist model（不像 RT-X/OpenVLA 想做 universal robot），目标只是把可迁移的 physical prior 蒸馏到双臂 policy 上。这个 framing 很重要，因为它决定 unified action space 的设计是 "preserve physical meaning" 而不是 "maximize cross-robot transfer"。

---

## 三、Diffusion Modeling 公式逐项解析

### 3.1 Reverse process（公式 1）

$$
\mathbf{a}_t^{k-1} = \frac{\sqrt{\bar{\alpha}^{k-1}} \beta^k}{1 - \bar{\alpha}^k} \mathbf{a}_t^0 + \frac{\sqrt{\alpha^k}(1 - \bar{\alpha}^{k-1})}{1 - \bar{\alpha}^k} \mathbf{a}_t^k + \sigma^k \mathbf{z}, \quad k = K, \ldots, 1
$$

每一项的物理意义：

- **a_t^k**：在 diffusion step k 时的 noisy action（输入），k 越大越接近纯噪声
- **a_t^0**：clean action（要预测的目标），由 f_θ 估计
- **a_t^{k-1}**：去噪一步后的 action
- **α^k**：第 k 步的 noise schedule 系数，DDPM 里典型取 β^k ∈ [0.0001, 0.02]，α^k := 1 - β^k
- **ᾱ^{k-1} := ∏_{i=1}^{k-1} α^i**：累积乘积，表示从第 0 步到第 k-1 步保留的信号比例
- **β^k := 1 - α^k**：第 k 步加的噪声比例
- **σ^k**：随机噪声系数（DDPM 里取 σ^k = √β^k 或 0）
- **z ~ N(0, I)**：标准高斯，仅在 k > 1 时引入随机性，k=1 时 ᾱ^0 = 1, z = 0（最后一步不采样噪声）
- **K**：总 diffusion step（训练时 K=1000，推理时用 DPM-Solver++ 压到 5 步）

第一项系数 (√ᾱ^{k-1} β^k)/(1-ᾱ^k) 是 "x_0 prediction" 的权重，第二项是 "x_k direction" 的权重，第三项是 stochastic noise。

### 3.2 Training objective（公式 2）

$$
\mathcal{L}(\theta) := \mathrm{MSE}\left(\mathbf{a}_t, f_\theta(\ell, \mathbf{o}_t, \sqrt{\bar{\alpha}^k}\mathbf{a}_t + \sqrt{1 - \bar{\alpha}^k}\boldsymbol{\epsilon}, k)\right)
$$

变量：
- **θ**：denoising network f_θ 的参数
- **k ~ Uniform({1, ..., K})**：训练时随机采样的 diffusion step
- **ε ~ N(0, I)**：训练时加的随机噪声
- **√(ᾱ^k) a_t + √(1 - ᾱ^k) ε := ã_t**：noisy action（论文里后面简写成 ã_t，省略 k 上标）

注意 RDT 选的是 **x_0-prediction**（直接预测 clean action），不是 ε-prediction。这个选择对 robot 任务来说是合理的：action 维度低（128 维 unified space），不像 image 那样几万维，x_0-prediction 数值稳定性没问题，且 inference 时可以省掉从 ε 到 x_0 的转换。

### 3.3 为什么 diffusion 而不是 VAE / Discretization

这是这篇 paper 的关键 framing：

| 方法 | 表达能力 | 精度 | 问题 |
|---|---|---|---|
| VAE (ACT) | 弱（单一 latent Gaussian） | 连续 | 单峰近似，多模态压成单峰 |
| Discretization (RT-2, OpenVLA) | 强 | 量化误差 | 双臂 14D 输出 token 数量爆炸，量化误差让精细控制不可行 |
| Diffusion (RDT) | 强 | 连续 | 采样慢，但 action 维度低所以可接受 |

OpenVLA 在他们的实验里基本全军覆没（success rate ~0%），就是吃了 discretization 的亏。双臂 action 维度太高，token 化以后 token accuracy 收敛不上去（论文 Fig. 8），训练时 action token accuracy 在 60% 徘徊，部署时直接是 random/static behavior。

---

## 四、Architecture 拆解：从 DiT 到 RDT 的三个关键改动

RDT 的 backbone 是 Diffusion Transformer (DiT, Peebles & Xie 2023)，但在 robot 数据上发现 vanilla DiT 训练会爆炸（Fig. 4a）。原因：robot 物理量数值范围不稳定、有 high-frequency 跳变（碰撞、阻尼、约束瞬间变化），跟 image/video 的 spatio-temporal continuity 完全不一样。

### 4.1 QKNorm & RMSNorm

**QKNorm**：在 attention 的 Q 和 K 上加 L2 normalization（Henry et al. 2020）。Attention 计算时 softmax(QK^T/√d) 在 Q、K 范数很大时会饱和（logit 过大），导致梯度消失或数值溢出。在 1B 规模训练时这个问题会被放大。

**RMSNorm 替代 LayerNorm**：LayerNorm 公式是 y = (x - μ)/σ ⊙ γ + β，包含 centering (减均值 μ)；RMSNorm 公式是 y = x/√(mean(x²) + ε) ⊙ γ，**不做 centering**。

为什么 centering 对 robot task 有害？论文引用了 UnitNorm (Huang et al. 2024) 的观点：把 problem 视为 time series forecasting，centering 操作会让 token 之间产生相对位移，破坏时间序列的对称性。具体来说，对一个 chunk 内的所有 action token 同步做 centering 会改变它们之间的相对位置关系，而 action 时序的对称性（比如周期动作）依赖于这种相对关系。

Fig. 4a 显示：不加 QKNorm & RMSNorm，loss 在 pretraining 后期会 explode。

### 4.2 MLP Decoder

原始 DiT 用线性层把 latent token 投回物理空间，但 robot action 是高度非线性的（关节角度、笛卡尔位姿、gripper 状态混合在一起，物理量纲不同）。RDT 用一个 MLP decoder（论文没写具体层数，从 Appendix B 推测是 2-3 层 with GeLU）替代。

直觉：线性 decoder 只能做仿射变换，对于 "action token 在 latent space 已分离，但回映到 action space 需要非线性 warp" 的场景力不从心。比如 gripper open/close 的 binary 状态、joint limit 附近的非线性约束，这些都需要非线性 head 来正确捕捉。

Fig. 4b 在 Robot Dog 任务上做了消融：去掉 MLP decoder，walk straight 这个 sub-task 的 success rate 显著下降。Robot Dog 这个任务对精度极其敏感（推 joystick 角度差几度，机器狗就走偏），所以是检验 MLP decoder 价值的最佳 benchmark。

### 4.3 Alternating Condition Injection (ACI)

这是我觉得这篇 paper 最巧妙的设计。

**问题**：传统 DiT 用 adaptive layer norm (adaLN) 把 condition 注入。adaLN 把 class label 压成一个 token，再调制每个 transformer block 的 scale 和 shift。但 RDT 的 condition 是图像和语言，**变长**且**信息密度不对称**：图像 token 数量远多于语言 token（SigLIP 输出几百个 image patch token，T5-XXL 输出几十个 text token）。adaLN 把它们都压成单 token 会丢失大量信息。

**Solution 1**：用 cross-attention 替代 adaLN，让 condition 以变长 token 序列形式注入。

**Solution 2 (ACI)**：相邻 transformer block 交替注入——奇数层只 cross-attend 图像 token，偶数层只 cross-attend 语言 token（或反之）。

为什么这么做？如果同一层同时注入图像和语言，image token 数量优势会让 attention sink 到图像上，text 信息被稀释。这就是为什么 OpenVLA 类模型在 instruction following 上能力有限——visual 信息 dominate 了 textual 信息。

ACI 让 text 在 dedicated layer 里获得完整的 attention capacity，不被 image token "抢资源"。

Fig. 4b 的消融：去掉 ACI，Pour Water-L-1/3 这个 task 的 "correct amount" sub-task 从 100% 掉到 12.5%。原因正是 Pour Water-L-1/3 需要精确理解 "one-third" 这个量词，这是 language-conditioned 的核心，image token 一抢就崩。

### 4.4 完整数据流

```
输入端：
├─ Language ℓ → T5-XXL (frozen) → MLP → text tokens [N_text, 2048]
├─ Image X ∈ R^{T_img × 3 × 384 × 384} → SigLIP (frozen) → MLP + 4D positional embedding (T_img, N_cam, N_patch, D) → image tokens
├─ Proprioception z_t → unify action space (128D) + pad mask (128D) → MLP with Fourier features → 1 token
├─ Noisy action chunk ã_{t:t+T_a} → unify action space (128×64) → MLP with Fourier features → 64 tokens
├─ Control frequency c → MLP → 1 token
└─ Diffusion step k → MLP → 1 token

低维 tokens concat → [1+64+1+1=67, 2048] + position embedding

主干：
67 个 token 进 DiT block × 28 层
  - Self-attention (with QKNorm & RMSNorm)
  - Cross-attention: 奇数层 cross-attend image tokens，偶数层 cross-attend text tokens
  - Feedforward

输出：
67 tokens → RMSNorm → MLP decoder → 回映到 64 个 action 的 unified action space (128D each) → 取出对应 robot 的实际 action 维度
```

模型规模：28 layers, hidden size 2048, 32 heads, 1.2B params。其中 SigLIP 和 T5-XXL 是 frozen 的，所以 1.2B 是 DiT 主干 + 适配 MLP 的参数，frozen encoder 单独算（T5-XXL 本身约 11B，SigLIP 几百 M）。

---

## 五、Unified Action Space 的设计

这是解决 Challenge 2 的核心。设计原则：**保留物理语义**，每个维度有明确含义，不同机器人按物理量对齐填充。

### 5.1 128 维向量布局

```
[0, 10):   右臂 joint positions (10 DoF 上限，6 DoF 机器人填前 6 位)
[10, 15):  右夹爪 joint positions
[15, 25):  右臂 joint velocities
[25, 30):  右夹爪 joint velocities
[30, 33):  右末端 position (xyz)
[33, 39):  右末端 6D pose (位置+6D 旋转表示，用 Zhou et al. 2019 的 6D 表示避免 gimbal lock)
[39, 42):  右末端 linear velocity
[42, 45):  右末端 angular velocity
[45, 50):  reserved

[50, 100): 左臂对称 (跟右臂结构完全一致)

[100, 102): base linear velocity
[102, 103): base angular velocity
[103, 128): reserved
```

单臂机器人映射到右臂位（左臂位全 padding），这样保证物理一致。

### 5.2 Padding 处理的关键 trick

如果直接用 0 padding，model 无法区分 "0 = 物理上的静止" 和 "0 = 不存在这个 sensor"。Solution：在 unified action space 旁边 concat 一个 0-1 mask vector（同样 128 维），表示每个维度是否 valid，形成 256 维输入。

这其实是一个 generalizable pattern：任何涉及 cross-embodiment 的 multi-modal fusion 都应该考虑这种 "value + availability mask" 的双输入设计。

### 5.3 不做严格 normalization

大多数 prior work 把 action 归一化到 [-1, 1] 或 N(0, 1)。RDT 不做这个，只统一物理单位（m, rad, m/s, rad/s）。理由：跨机器人时 "1 m" 在物理上是一样的，归一化会破坏这个 shared physical prior，反而损害跨机器人迁移。

直觉：归一化是把 action 分布压到模型容易学的尺度，代价是丢失了"绝对尺度"这个物理信息。RDT 选择保留物理量纲，把 normalization 的责任丢给 RMSNorm + QKNorm 来处理数值稳定性。

---

## 六、数据工程细节

### 6.1 预训练数据

46 个数据集，1M+ trajectories，21TB。Table 5 列了完整的采样权重。采样权重初始化为 √N_j（N_j 是第 j 个数据集大小），然后根据 loss 收敛情况手动调（慢收敛的数据集加权）。

为什么用 √N 而不是线性？√N 是个折中——完全平等会过度采样小数据集（重复 sample），完全按 size 加权会让大数据集 dominate，mini-batch 多样性下降。√N 是个常用的平衡采样启发式。

主要数据集：
- **RT-1 Dataset**（130K，13 个 embodiment，3Hz，6D EEF + gripper + base displacement）
- **DROID**（76K，Franka Panda 7-DoF，15Hz，7D joint position + gripper width）
- **RH20T**（110K，4 种 embodiment，10Hz，混合 6/7 DoF）
- **Mobile ALOHA Dataset**（1K+，双臂数据）
- **Open X-Embodiment** 的子集

### 6.2 微调数据

Mobile ALOHA robot，6K+ trajectories，300+ tasks，3M+ frames。100+ objects（rigid + non-rigid），15+ 房间，光照变化。Language instruction 用 GPT-4-Turbo 做 augmentation——每条原始 instruction 扩展 100 条 + 1 条简化版本。这模仿 LLM 里 instruction tuning 的做法，提升 language robustness。

### 6.3 Data preprocessing 的几个细节

- **图像输入固定为 3 视角**：external + right-wrist + left-wrist。单臂机器人的 wrist camera 映射到 right-wrist，缺失视角用背景色 padding
- **图像尺寸**：384×384，保持长宽比 padding 成 square
- **6D 旋转表示**：用 Zhou et al. 2019 的 6D 表示（不是 quaternion 也不是 euler），避免 gimbal lock，神经网络对 6D 表示学得更好
- **历史 proprioception** z_i (i<t) **不输入**——防止 model 用低维信号 shortcut 学到 fixed motion pattern，强迫它从 high-dim image 学可泛化的决策结构。这是个反直觉但很重要的设计

---

## 七、实验设计与关键发现

### 7.1 七个评估任务覆盖五个维度

| 任务 | 测试维度 | 关键挑战 |
|---|---|---|
| Wash Cup | Unseen Object (Q1) | 多 sub-task：取杯 → 开龙头 → 接水 → 倒水 → 放回；测试 2 个未见过的杯子 |
| Pour Water | Unseen Scene (Q1) | 3 个未见过的房间 |
| Pour Water-L-1/3 | Instruction Following (Q2) | "用左手倒 1/3"，从未见过 "one-third" 这个量词 |
| Pour Water-R-2/3 | Instruction Following (Q2) | "用右手倒 2/3" |
| Handover | 5-Shot Learning (Q3) | 仅 5 个 demo 训练 handover skill |
| Fold Shorts | 1-Shot Learning (Q3) | 仅 1 个 demo 训练折叠技能 |
| Robot Dog | Dexterity (Q4) | 推 joystick 让机器狗走直线，角度敏感 |

注意 Handover 和 Fold Shorts 的 few-shot 设置——**预训练后再在 1-5 个 demo 上微调**，跟 LLM 的 in-context learning 不是一个概念，而是用 few-shot data 做 lightweight fine-tuning。这更接近 meta-learning 的 setting。

### 7.2 主结果（Table 3 摘要）

RDT 在所有任务上 success rate 平均 56% 提升，主要 baseline（ACT, OpenVLA, Octo）多数任务 success rate 接近 0%。

特别值得分析的：

**Wash Cup**：ACT 在 seen cup 上能跑（取杯子成功率 50%），但 unseen cup 直接崩（12.5%）。OpenVLA 完全失败。RDT 在 unseen cup 2 上 50%。这说明 cross-robot pretraining 提供的 visual prior 对 OOD 物体识别很关键。

**Pour Water-L-1/3 vs Pour Water-R-2/3**：两个任务共享训练数据（18 + 19 + 19 = 56 个 demo 对应 little/half/full 三个水位），测试 "1/3" 和 "2/3" 这两个未见过的量词。RDT 在 "correct amount" 子指标上 100%（L-1/3）和 75%（R-2/3）。这说明 model 学到了 "fraction" 的 compositionality，而不是单纯记忆 "half = 这个角度"。

**Robot Dog**：ACT 失败原因是 joystick 跟 remote control 都是黑色，视觉对比度低，small model 学不出 "joystick" 的 visual concept。RDT 因为 pretraining 见过海量 objects，joystick 的 vision-language representation 学得更好。

### 7.3 消融研究（Table 2）

| 变体 | Unseen Object | Unseen Scene | Instruction Following |
|---|---|---|---|
| RDT (regress) | 12.5 | 50 | 12.5 |
| RDT (small) | 37.5 | 62.5 | 25 |
| RDT (scratch) | 0 | 25 | 62.5 |
| RDT (ours) | 50 | 62.5 | 100 |

四个关键 takeaway：

1. **去 diffusion（regress）**：unseen object 直接崩到 12.5%，instruction following 崩到 12.5%。Multi-modality 用 regression 处理就是死路一条。
2. **小模型（166M）**：generalization 还行但 instruction following 弱，说明 1B 规模对 language grounding 是有 scaling benefit 的。
3. **无预训练**：unseen object/scene 直接 0%/25%，但 instruction following 仍有 62.5%。这是个有意思的现象——预训练主要贡献 visual generalization，instruction following 更多靠 fine-tuning data（因为预训练数据 language quality 参差不齐）。这暗示 pretrain + finetune 分工：pretrain 负责 visual + physical prior，finetune 负责 task-specific language grounding。
4. **完整 RDT**：所有维度都最强，证明三个要素（diffusion + scale + pretrain）是 complementary 的。

---

## 八、训练与推理工程

### 8.1 训练

- **硬件**：48 张 H100 80GB，pretrain 1M steps（约 1 个月），finetune 130K steps（3 天）
- **Optimizer**：AdamW，lr = 1e-4，β_1=0.9, β_2=0.999, weight decay 1e-2, ε=1e-8, bf16 精度
- **Batch size**：32 × 48 = 1536
- **DeepSpeed** 做 ZeRO 分布式训练
- **数据加载**：producer-consumer 架构，把 Open X-Embodiment 的 TFRecord 解压到 hard disk buffer，consumer 乱序读取，避免内存 shuffle buffer 太小的问题
- **Noise schedule**：DDPM with squaredcos cap v2（glide cosine），1000 步
- **Monitoring trick**：训练中定期 sample action chunk，跟 ground truth 算 MSE，发现这个 MSE 跟 real robot performance 强相关。MSE 收敛就可以停训；过低可能 overfit

### 8.2 推理

- **DPM-Solver++** (Lu et al. 2022)：把 1000 步 diffusion 压到 5 步
- **Onboard GPU**：RTX 4090 24GB
- **Action chunk 频率**：6 Hz（每秒生成 6 个 chunk，每个 chunk 64 个 action）
- **Action 频率**：381 Hz（6 × 64 ≈ 384），实际执行时 chunk 内 action 顺序执行
- **No CFG**：Classifier-Free Guidance 试过但没用，反而让 robot 行为不稳定。这说明 robot task 跟 image generation 不一样，conditional distribution 本身就够 sharp，不需要 guidance sharpen

### 8.3 Data augmentation

- **Image augmentation**：color jitter + image corruption
- **Proprioception**：加 Gaussian noise，SNR = 40dB
- **Language**：GPT-4-Turbo 扩展（100 条 expanded + 1 条 simplified per task）
- Fine-tune 时去掉了每个 episode 开头的 static 段（operator 反应延迟）
- Episode 长度过滤：< 32 frame 丢掉，> 2048 frame 下采样到 2048

---

## 九、跟相关工作的对比与定位

### 9.1 vs Diffusion Policy (Chi et al. 2023)

Diffusion Policy 是 RDT 的直接前作。差异：
- Diffusion Policy 用 CNN backbone (1D temporal conv)，RDT 用 Transformer，scalability 强
- Diffusion Policy 10K-100K trajectory scale，RDT 1M+
- Diffusion Policy 是 single-robot，RDT 是 cross-robot
- Diffusion Policy 不处理 multi-modal condition（vision + language），RDT 用 ACI 处理

### 9.2 vs ACT (Zhao et al. 2023)

ACT 是 ALOHA 论文，bimanual manipulation 的 SOTA baseline。ACT 用 CVAE（条件 VAE）建模 action distribution，latent code 从 N(0, I) 采样，decoder 生成 action。

问题：CVAE 的 latent 是单模态 Gaussian，对 multi-modal action distribution 表达能力不足——它会把多模态压成单模态的"平均"，虽然 VAE 的 stochasticity 让它能 sample 出不同 mode，但 mode 之间没有 sharp 分离，所以 ACT 在 multi-modal task 上动作会出现"中间态"。

RDT 在 Pour Water 系列（左右手任选）上完胜 ACT 就是因为这个：ACT 不确定用哪只手，输出介于左和右之间，物理上不可行；RDT 的 diffusion 能从不同 mode 里 clean sample。

### 9.3 vs OpenVLA (Kim et al. 2024)

OpenVLA 是 7B 参数的 VLA model，基于 Prismatic VLM backbone，把 action 离散化成 token。在单臂上效果不错，但双臂场景彻底崩盘（success rate ~0%）。

为什么崩？双臂 14D action 离散化以后每一步要预测 14 个 token，长程依赖 + 量化误差双重打击。论文里 Fig. 8 显示 OpenVLA 在 fine-tuning 时 action token accuracy 在 60% 上下震荡，根本没收敛。这意味着即使全参数 fine-tune 也救不回来——是 modeling 方法的根本问题。

### 9.4 vs Octo (Ghosh et al. 2023)

Octo 是另一个 diffusion-based foundation model，最大版本 93M 参数。Octo 在 multi-robot 数据上预训练，但只在 Open X-Embodiment 的 25 个数据集上预训练，规模小一个量级。

Octo 在双臂上的问题：架构没有针对 bimanual 的特殊设计，cross-attention 简单堆叠 image 和 text condition，image token 淹没了 text 信息，导致 instruction following 弱。Table 3 显示 Octo 几乎所有任务都失败。

### 9.5 vs RT-2 / RT-X

Google 的 RT-2 也是 VLA discretization 路线。RT-2 没开源，但 OpenVLA 的复现基本代表了这条路。RDT 论文选 OpenVLA 作为 RT-2 路线的代表 baseline。

### 9.6 vs Mobile ALOHA (Fu et al. 2024)

Mobile ALOHA 也是 bimanual 论文，但用的是 co-training（把 ALOHA 数据跟 existing single-arm data 混合训练 ACT）。RDT 用 Mobile ALOHA 作为硬件平台和 fine-tuning 数据来源，但 modeling 路线完全不同——Mobile ALOHA 是 ACT，RDT 是 Diffusion Transformer。

---

## 十、Beyond 这篇 paper 的延伸思考

### 10.1 ACI 这个 idea 的 generalization

ACI 的本质是 "在 cross-attention 中对 condition modality 做 round-robin scheduling"。这个 idea 完全可以推广：

- 多视角图像之间也可以 ACI（exterior / right-wrist / left-wrist 在不同层交替注入）
- 长文本 / 短文本可以 ACI（避免 long context 淹没 key instruction）
- 历史 observation vs 当前 observation 可以 ACI

甚至可以推广到 LLM 的多 source attention——比如 multi-document reasoning 时不同文档轮流注入 attention，避免某个文档 dominate。这是个 scalable 的 design pattern。

### 10.2 Unified Action Space 的局限

128 维 unified space 是 hand-crafted 的，针对 "robot with gripper arms"。对于：
- 柔性机器人（continuum manipulator）：joint position 概念不适用
- 全身机器人（humanoid with legs）：action 维度爆炸，128 维不够
- 触觉/力控机器人：需要 force/torque 维度

未来方向可能是 learnable unified space——让 model 自己学一个跨 embodiment 的 latent action representation。但这会牺牲 physical interpretability，而论文 claim physical interpretability 是 transferability 的关键。这个 trade-off 值得深入研究。

### 10.3 x_0-prediction vs ε-prediction

RDT 选 x_0-prediction。对比：
- ε-prediction：训练更稳定（DDPM 原始论文），但 inference 需要从 ε 反推 x_0
- v-prediction (Salimans et al. 2022)：中间路线，数值稳定性更好
- x_0-prediction：低维 action 直接，目标明确

对于 robot action 这种低维（128D）、物理意义明确的目标，x_0-prediction 的 "目标对齐" 优势大于 ε-prediction 的 "训练稳定性" 优势。但这个选择在 1B 规模 pretraining 时是否还成立，paper 没做对比实验。

### 10.4 5 步 diffusion 够用吗

DPM-Solver++ 把 1000 步压到 5 步，action chunk 频率 6Hz。但 5 步 diffusion 的 mode coverage 能力比 1000 步差多少？Paper 里没详细分析。理论上 DDPM 的 mode coverage 跟采样步数强相关，5 步可能只能 cover 主导 mode，丢掉长尾 mode。这可能让 RDT 在"创意性"操作上受限——比如多种抓取姿态中只能 sample 出最常见的那个。

### 10.5 Pretraining data 的 quality bias

46 个数据集质量参差不齐，论文用了 √N 加权 + 手动调权。但更系统的方法可能是：
- 用 ICL-style evaluation 自动估每个数据集的"质量"
- 用 contrastive learning 学跨数据集的 task similarity，做 similarity-weighted sampling
- Active learning：在 pretraining 中动态发现 hard examples 加权

### 10.6 Failure modes 的分析缺口

Paper 没分析 RDT 在哪些情况下失败。比如：
- 长程任务（>100 步）的 error accumulation 如何
- 物理交互复杂度高的任务（柔性物体、可变形物体）成功率
- 跨 embodiment gap 大的 fine-tune 数据（比如从 RT-1 的 EEF control 到 ALOHA 的 joint control）的迁移效率

这些 failure analysis 下一篇 paper 应该补上。

---

## 十一、关键参考文献

- **DiT** (Peebles & Xie 2023): https://arxiv.org/abs/2212.09748 - RDT 的 backbone 来源
- **Diffusion Policy** (Chi et al. 2023): https://diffusion-policy.cs.columbia.edu/ - diffusion for robot policy 的开山之作
- **ACT / ALOHA** (Zhao et al. 2023): https://tonyzhaozh.github.io/aloha/ - bimanual manipulation baseline
- **OpenVLA** (Kim et al. 2024): https://openvla.github.io/ - VLA discretization baseline
- **Octo** (Ghosh et al. 2023): https://octo-models.github.io/ - diffusion foundation model baseline
- **Open X-Embodiment** (Collaboration et al. 2023): https://robotics-transformer-x.github.io/ - 预训练数据来源
- **DPM-Solver++** (Lu et al. 2022): https://arxiv.org/abs/2211.01095 - 推理加速
- **SigLIP** (Zhai et al. 2023): https://arxiv.org/abs/2303.15343 - 视觉编码器
- **T5-XXL** (Raffel et al. 2020): https://arxiv.org/abs/1910.10683 - 语言编码器
- **Mobile ALOHA** (Fu et al. 2024): https://mobile-aloha.github.io/ - 微调数据来源
- **6D rotation representation** (Zhou et al. 2019): https://arxiv.org/abs/1812.07035 - EEF pose 表示
- **RMSNorm** (Zhang & Sennrich 2019): https://arxiv.org/abs/1910.07467 - 替代 LayerNorm
- **QKNorm** (Henry et al. 2020): https://arxiv.org/abs/2010.04245 - attention 稳定
- **DDPM** (Ho et al. 2020): https://arxiv.org/abs/2006.11239 - diffusion model 基础
- **DAgger** (Ross et al. 2011): https://arxiv.org/abs/1011.0686 - error accumulation 理论基础
- **UnitNorm** (Huang et al. 2024): https://arxiv.org/abs/2405.15903 - time series 中 normalization 的分析

---

## 十二、一个 build intuition 的总结

RDT 这篇 paper 的核心 contribution 在我看来不是某个单一 trick，而是把几个 known good idea 在 robot 这条垂直 domain 上做了 system-level 的整合：

1. **Diffusion 表达 multi-modality**：来自 image generation 领域的成熟技术，应用到 robot policy 的关键 insight 是 action 维度低所以采样快，避开了 diffusion 在 image 上的 main drawback
2. **Transformer scalability**：来自 LLM，DiT 把这个 scalability 带到 diffusion model
3. **Cross-embodiment pretraining**：来自 RT-X / Open X-Embodiment，但通过 unified action space 让 transfer 更高效
4. **ACI 多模态 condition 注入**：原创性最强，解决了 cross-attention 在多模态 imbalance 下的 information loss

从 scaling law 视角看，RDT 是 robot foundation model 的 "GPT-2 时刻"——1B 参数在 robot domain 已经是当时最大，但相对 LLM 的 100B+ 仍有数量级差距。Paper 的 ablation 暗示 instruction following 对 scale 敏感（166M → 1.2B 提升 4 倍），未来 10B+ 的 robot foundation model 应该会带来质变。

但 robot domain 跟 LLM 的关键差异在于 data efficiency：LLM 可以靠 web crawl 拿海量数据，robot data 必须实机采集。RDT 用 cross-embodiment 暂时绕过了这个瓶颈，但 long-term 看 sim-to-real 和 real-world data collection 的成本仍然是 robot foundation model scaling 的 fundamental bottleneck。下一篇大 paper 大概率会在这两个方向之一突破——要么是大规模 sim data + sim-to-real，要么是新硬件范式（比如 UR 训练 + deploy 到 ALOHA，或者 VLA teleoperation 让 mass user 采集数据）。

这也是为什么 Karpathy 你之前在 robot 数据这块的关注点是对的：data 是 bottleneck，model architecture 在 RDT 这个层级已经基本 "solved"，剩下的 gap 在 data quantity 和 quality 上。
