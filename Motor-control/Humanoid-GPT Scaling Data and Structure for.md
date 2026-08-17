---
source_pdf: Humanoid-GPT Scaling Data and Structure for.pdf
paper_sha256: e96ea94a53c1c8f72eb99aa2b1ea8f6559e48a1bb22acdf2a72898cb4e214ea1
processed_at: '2026-08-05T07:57:53-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 Humanoid-GPT

Karpathy，我把这篇paper掰开揉碎讲，尽量像你给学生白板讲那种感觉，少装逼多讲直觉。

---

## 一句话讲它在干嘛

之前 humanoid 机器人追着人的动作模仿，都是用一个小 MLP，喂几百万帧的 motion capture 数据训练，结果就是——会跑会跳的动作能跟，但换个没见过的风格就趴窝了。这篇 paper 的主张就一句：**数据搞大 200 倍，模型换成 GPT-style Transformer，能力就自然涌现出来**。

听起来像是 LLM 范式直接搬过来。确实，你读了会发现整个 framing 就是 GPT recipe 的 robotics 版本。

---

## 之前为什么做不好？核心矛盾在 scale

Paper 开篇就讲了一个 tradeoff：**agility 和 generalization 的矛盾**。

- Beyond-Mimic [19]、ASAP [11] 这些，能在训练过的敏捷动作（跑跳后空翻）上做得很好，但换个没见过的动作就废了
- TWIST [42]、UniTracker [41] 泛化稍好，但一遇到高动态动作就掉链子

作者说这个 tradeoff 不是 fundamental 的，是数据不够大、模型不够大的 symptom。这个 argument 我觉得基本成立——LLM 早期也有 "大模型 overfit 小数据 vs 小模型 underfit" 的矛盾，后来 scaling 上去就消失了。

Table 1 一目了然，prior work 全在 6–9M frames 上挣扎，这篇直接干到 2B frames。

---

## 三个关键 move

我把这篇 paper 的 contribution 拆成三块，对应它的三个 pillar：

1. **Science of Scale**：2B frames 的 motion corpus，比之前大 200 倍
2. **Modern Structure**：GPT-style causal Transformer
3. **Balanced Diversity**：HME 聚类 + 平衡采样，防止 long-tail 被淹没

这三个加起来，得到 zero-shot tracking 能力。下面我一块块讲。

---

## 数据这块——为什么 2B frames 这么难搞

### 数据来源

把所有能找到的 motion 数据全拼了：

- **AMASS** [24]：经典 mocap 数据集，但局限于 studio
- **LAFAN1** [9]：robust in-betweening 那篇的 dataset
- **MotionMillion** [7]：million-scale 生成动作，注意是 generated 不是 captured
- **PHUMA** [16]：physically grounded，带 contact 和 joint constraint
- **Motion-X++** [43]：video 重建的
- **In-house capture**：自己补的

总规模 2B frames。

Pipeline：
1. **Retarget**：把 human motion 用 [2] 的 framework 映射到 Unitree-G1 的 29 DoF joint space。这一步会丢信息，G1 的 kinematics 跟人不一样，retarget quality 决定 ceiling
2. **Filter**：把 sitting on chair、swimming、stair climbing 这些需要 object interaction 的 clip 扔掉，因为 G1 在空场景里做不到
3. **Time-warping augmentation**：每个 clip uniform 加速减速，5× expansion，让 policy 见不同 speed

这里有个直觉你应该抓住：**time-warping 相当于 LLM 里 data augmentation 的 paraphrase**。同一个 motion 不同 speed 出现 5 次，policy 学到的不是"这个时间点的动作"，而是"这个 phase 的动作"，对 temporal robustness 帮助很大。

---

## HME——我觉得这是 paper 里最 clever 的一块

### 问题

你有 2B frames，但 motion 分布是 long-tail——walking/running 占 90%，kungfu/backflip 这种 rare behavior 只占很小一部分。如果 uniform sample 训练，policy 会 collapse 到 dominant mode，rare 的学不会。

### HME 的 intuition

先问自己一个问题：**怎么衡量两条 motion "像不像"？**

如果你直接在 raw joint angle space 算距离，没用——两条 walking，一条快一条慢，raw distance 可能很大，但语义上一样。你需要一个 **频谱分析式的 representation**。

HME 的做法：
1. 在不同 data partition 上训练 **Periodic Autoencoder** [33]（这是 Starke 等人 DeepPhase 的工作）。Periodic AE 把每个 joint 的 motion 压缩成 **amplitude + frequency** 的 latent——因为 locomotion 本质上是周期性的，gait cycle、arm swing 都有 period
2. 对每条 sequence，把所有 joint 的 amplitude/frequency 的 **mean 和 std** 拼起来，得到一个 compact embedding vector
3. 在所有 embedding 上跑 **K-Means**，得到 ~300 clusters，每 cluster 1k–2k sequences

这个 embedding 为什么比 raw motion 好？我类比一下：**这跟音乐里做 spectrum analysis 然后聚类是一回事**。你直接对 waveform 算距离没用，因为两条歌相位差一点 raw distance 就爆了。但你对 spectrum（哪些频率成分多）聚类，"都是快节奏鼓点"的歌会聚到一起，不管 phase 差多少。

HME 把 motion 投影到一个 "harmonic latent space"，那里 distance 有语义。

### 为什么这个 space 适合做 diversity 评估

Paper 用 HME 算两个 diversity 指标，公式 (3)：

$$\mathrm{gstd} = \exp\left(\frac{1}{D}\sum_{j=1}^{D}\log\sigma_j\right)$$

$$\mathrm{log\text{-}volume} = \frac{1}{2}\log\det(\Sigma + \epsilon I)$$

变量含义：
- $X = [x_1, \ldots, x_N]^\top \in \mathbb{R}^{N \times D}$：从 dataset uniformly sample 的 10,000 个 HME embedding
- $\Sigma$：它们的 covariance matrix，$D \times D$ 维
- $\sigma_j$：$\Sigma$ 第 $j$ 个 dimension 的 standard deviation
- $D$：embedding dimension
- $\epsilon$：小常数防止 $\det(\Sigma)$ 奇异

**gstd 是 std 的 geometric mean**。为什么用 geometric 而不是 arithmetic？因为 geometric mean 对"某个维度塌掉"特别敏感——只要有一个 $\sigma_j$ 接近 0，gstd 就爆掉。arithmetic mean 会被其他大 $\sigma_j$ 拉高，掩盖 collapse。所以 gstd 衡量的是"每个方向都有 spread"。

**log-volume 是 covariance ellipsoid 体积的 log**。如果 data 在每个方向都 spread，volume 就大。它比 gstd 更全局，因为它捕捉 cross-dimension correlation——如果两个维度高度相关（信息冗余），volume 也不会大。

Figure 3 的结果：curated dataset 比 AMASS 大 4–5 个量级 in log-volume。这个 gap 是真的很大——AMASS 在 latent space 是一个紧凑 blob，curated dataset 是 spread-out manifold。

**Karpathy 你应该立刻想到 LLM 的对照**：CommonCrawl 10T tokens 但全是 SEO 垃圾，不如 1T tokens 高质量 web text。这里一样——2B frames 如果全是 walking，不如 200M frames 但覆盖 dance/jump/kungfu。

---

## 模型——为什么非得 GPT-style

### Online tracking 的物理约束

这是 paper 最关键的论点之一。Motion tracking 在 deployment 时是 **causal** 的——policy 在 timestep $t$ 不能 access $t+1, t+2$ 的 observation。机器人不知道未来。

Prior work 里很多用 Transformer 但 **非 causal**——training 时 bidirectional，deploy 时 sliding window。这是 train-test distribution mismatch，跟 LLM 里 BERT 不能直接 generate 一样的道理。

Humanoid-GPT 直接用 **GPT-style causal attention**，training 和 deployment 完全一致。这消除了 distribution shift。

### 架构细节

从 Table 7 抠出来：
- **Layers**: 12
- **Channel dims**: 256 / 384 / 768 对应 S/B/L 三个 size
- **Params**: 5.7M / 22.1M / 80.4M
- **History length $H$**: 默认 32 帧，ablation 到 64 帧能再涨但 quadratic cost
- **Optimizer**: AdamW, lr=1e-4
- **Training iteration**: 200k

Token 怎么构造？公式 (2) 附近：

$$e_t = \mathrm{concat}(s_t, q_t^{\mathrm{ref}})$$

- $s_t$：current proprioceptive state，包括 per-joint position、velocity、root angular velocity、projected gravity、previous action $a_{t-1}$
- $q_t^{\mathrm{ref}}$：target reference pose at timestep $t$（从 motion clip 来）

Output 是 per-joint PD targets $a_t$。

Paper 没明确说 position encoding 用什么——既然是 GPT-style 大概率是 RoPE 或 learned PE，要看 code 才知道。我猜是 RoPE，因为时间序列上 RoPE 的 extrapolation 比 learned PE 好。

### 为什么 Transformer 比 MLP scaling 好

Table 2 是 key evidence。我挑几行重看：

| Backbone | Tokens | Params | SR ↑ | MPJPE ↓ | MPKPE ↓ |
|---|---|---|---|---|---|
| MLP (3-layer) | 2M | 0.25M | 76.89 | 0.1191 | 100.49 |
| Humanoid-GPT-S | 2M | 5.7M | 83.26 | 0.0853 | 62.65 |
| Humanoid-GPT-B | 200M | 22.1M | 88.27 | 0.0793 | 44.78 |
| Humanoid-GPT-L | 2B | 80.4M | 92.58 | 0.0735 | 40.99 |

两个关键观察：

**1. 小数据 + 大模型 = overfitting（U-curve）**

MLP-L 在 2M tokens 上 SR=75.25%，比 MLP-S 的 76.89% 还差。这是经典 over-parameterized under-data regime。LLM 里你肯定见过——大模型在小数据上 overfit。

**2. Transformer 在 200M→2B 仍在改进，MLP 在 200M 之后 marginal**

这个 gap 在 **MPKPE（keypoint position error, mm）** 上最明显：TCN-L 56.15mm vs Humanoid-GPT-S 43.25mm，30% gap。

为什么？我的 intuition 是：

MLP 只看 1 帧 history，本质是 **stateless reflex controller**——看到 state 就映射到 action，没有"过去 0.5 秒我在做 arm swing 的上举 phase"这种 context。所以它对 phase 推理很弱。

Transformer 看 32 帧 history，能 modeling **phase、momentum、intent**——它知道"过去 0.5 秒手臂在 swing 上举 phase"，下一个 action 应该 continue 上举而不是 reverse。这就是 LLM 里 "context window → in-context reasoning" 的同构现象。

你也可以这么想：**MLP 是 GPT-2 没有位置编码、context window=1 的版本**。它当然 scaling 上不去——你把 context window 锁在 1，模型再大也只能记住当前 token。

---

## 训练——两段式 pipeline

这是 system 最复杂的部分。

### Stage 1: RL Motion Experts

对每个 HME cluster（~300 个），train 一个 PPO expert。Policy 形式：

$$\pi: \mathcal{G} \times \mathcal{S} \mapsto \mathcal{A}$$

- $\mathcal{G}$：target reference pose 空间
- $\mathcal{S}$：proprioceptive observation 空间
- $\mathcal{A}$：per-joint action 空间（PD targets）

Reward（公式 1）是 keypoint-level，这是细节：

$$R_{\mathrm{kpt}}(t) = R_{\mathrm{pos}}(t) + R_{\mathrm{rot}}(t) + R_{\mathrm{vel}}(t) + R_{\mathrm{penal}}(t)$$

$$R_{\mathrm{pos}}(t) = \sum_{k \in \mathcal{K}} w_k \exp\left(-\alpha_{\mathrm{pos}} \|e_{k,t}^{\mathrm{pos}}\|_1\right)$$

$$R_{\mathrm{rot}}(t) = \sum_{k \in \mathcal{K}} w_k \exp\left(-\alpha_{\mathrm{rot}} \theta_{k,t}\right)$$

$$R_{\mathrm{vel}}(t) = \sum_{k \in \mathcal{K}} w_k \exp\left(-\alpha_{\mathrm{vel}} \|e_{k,t}^{\mathrm{vel}}\|_1\right)$$

变量一个个讲：
- $\mathcal{K}$：tracked keypoints 集合，包括 arms、hips、feet、pelvis
- $k$：keypoint index
- $w_k$：per-keypoint weight，**lower body=1.5, upper body=0.75**。lower body 权重高，因为摔了游戏就结束了，upper body 错一点问题不大
- $e_{k,t}^{\mathrm{pos}} \in \mathbb{R}^3$：keypoint $k$ 在 timestep $t$ 的 position residual（humanoid 实际位置 - reference 位置）
- $e_{k,t}^{\mathrm{vel}} \in \mathbb{R}^3$：velocity residual
- $\theta_{k,t}$：rotation error，通过 SO(3) log map 把 rotation error 从 SO(3) 流形投影到 $\mathbb{R}^3$ 的 axis-angle 表示
- $\alpha_{\mathrm{pos}}=1.0, \alpha_{\mathrm{rot}}=2.0, \alpha_{\mathrm{vel}}=0.03$：scaling factors，控制 exponential decay 速度

**为什么用 $\exp(-\alpha \|e\|_1)$ 这个 form？**

这是 DeepMimic [Peng 2018] 之后的经典 trick，我拆开讲：

1. 当 error 小（接近 0），$\exp(-\alpha \|e\|) \approx 1 - \alpha\|e\|$，gradient 是线性的，optimization 平稳
2. 当 error 大，$\exp(-\alpha \|e\|) \to 0$，reward saturates。这意味着 policy "放弃了" outlier——不会为了追一个 unreachable 的 reference 而爆掉 gradient
3. 用 L1 norm $\|e\|_1 = |e_x| + |e_y| + |e_z|$ 比 L2 robust to outlier dimension

如果用 L2，一个 dimension 突然炸了（e.g. foot 撞到地面），$e^2$ 会主导 gradient，policy 会 panic。L1 不会。

$R_{\mathrm{penal}}(t)$ 包括 self-contact penalty（防止 limb 穿插）和 smoothness penalty（防止 policy 用抖动来 "fit" reference 的小细节）。

### Domain Randomization

Table 4 列了一堆，关键几个：
- Floor friction: $\mathcal{U}(0.3, 2.0)$
- External force: interval $\mathcal{U}(5.0, 10.0)$s，velocity $\mathcal{U}(0.1, 1.0)$ m/s
- DoF friction scaling: $\mathcal{U}(0.5, 2.0)$
- Torso CoM shift: $\mathcal{U}(-0.15, 0.15)$ m

这些 randomization 是 sim-to-real 的核心。expert 在 disturbed dynamics 下也 stable，distill 给 student 后 student 继承这个 robustness。

### Stage 2: DAgger Distillation

这是 student Transformer 的训练。DAgger [31] 的核心 intuition：

**Naive Behavior Cloning 的问题是 covariate shift**——你只在 expert trajectory 上训练，student 一旦偏离 expert 轨道就 OOD，越偏越远，最后爆炸。

DAgger 的解法：让 student rollout，遇到 OOD state 就 query expert，把 expert 的 action 当 label。这样 student 见到的 state distribution 是自己实际 deploy 时的 distribution。

公式 (2)：

$$\hat{a}_{t-H+1:t} = \bigcup_{t_i \in \mathcal{T}} \mathrm{concat}\ t_i(s_{t-k}^{\mathrm{priv.}}, g_{t-k})$$

$$l = \mathcal{L}(G_\theta(e_{t-H+1:t}), \hat{a}_{t-H+1:t})$$

变量：
- $\mathcal{T}$：所有 expert teacher 集合（~384 个）
- $t_i$：第 $i$ 个 teacher
- $H=32$：history length
- $e_{t-H+1:t}$：长度 $H$ 的 token sequence
- $G_\theta$：student Transformer
- $\hat{a}_{t-H+1:t}$：aggregated teacher actions
- $\mathcal{L}$：SmoothL1Loss

**两个 key design 点**：

**1. 一次 forward pass 所有位置都 supervise**

这是 GPT 训练的标准 trick——你喂一个 sequence，每个 position 都 predict next token，所有 position 的 loss 都累加。这让 training 极其 sample efficient，一个 sequence 提供 $H$ 个 supervision 信号。

LLM 里这就是为什么 transformer 比 RNN training 快几个数量级——RNN 要 sequentially unroll，每步只有一个 loss signal；Transformer 一次 forward 拿到所有位置的 prediction。

**2. Batch size ≥ #experts**

Paper 说 batch size 必须 ≥ expert 数量，防止 mode collapse。Intuition：如果 batch 太小，一个 batch 可能只看到部分 expert 的 supervision，student 容易 forget 其他 expert 的 behavior。Batch 足够大才能在一个 update 里 cover 所有 expert distribution。

### 为什么不直接 PPO 训练 Transformer？

Paper 没直接 discuss 这个问题，但我的 intuition 是：

**1. PPO 在 billion-scale data 上不稳定**

RL 的 reward landscape 是 non-stationary 的——policy 变了，environment 的 response也变。scale 到 2B frames 时 reward hacking 会很严重。PPO 训练 100M frames 已经够 fragile 了，2B frames 你 reward shaping 要重新 tune。

**2. Expert distillation 解耦了"探索"和"scaling"**

每个 expert 只在自己 cluster 上 RL，问题规模小，PPO 收敛稳定。distillation 是 supervised learning，scale 干净——你只要喂 data，loss 下降就行，不需要关心 reward shaping。

**3. Compute 效率**

PPO 需要 environment interaction，慢。distillation 是 offline supervised，fully parallelize。Table 8 显示：
- PPO experts: 12,000 GPU hours (75%)
- Distillation: 3,000 GPU hours (25%)

但 distillation 处理的 data 远多于 PPO experts——因为 PPO 一个 expert 只看自己 cluster 的 1k-2k sequences，而 distillation 看全部 2B frames。

这个 pipeline 跟 LLM 里 **GPT-4 蒸馏到 small model** 几乎一模一样——expert 像 GPT-4 慢但强，student 像 GPT-4o-mini 快但靠 expert 蒸馏出来。

---

## Scaling Law——这是 paper 的灵魂

### Data scaling（Figure 7）

固定 Humanoid-GPT-B 架构，变化 $T \in \{2M, 20M, 200M, 2B\}$，**non-overlapping subsets**——这点很重要，确保是 data 真正的 scaling，不是 epoch 数增加。

观察：2B 的 marginal gain 比 200M→2B 之间开始递减，但仍改进。说明 **当前 model capacity 在 2B 开始进入 data-limited regime**——继续加 data 还会有收益，但 model capacity 也需要同步 scale。

这跟 LLM 的 Chinchilla scaling law [Hoffmann 2022] 类似——data 和 model 必须同步 scale，否则一方 saturate。Chinchilla 的结论是 compute-optimal training 下 model 和 data 应该 1:1 scale，本文似乎也观察到类似现象。

### Model scaling（Figure 8）

固定 2B tokens，Transformer-B vs MLP（comparable params）。

观察：Transformer 仍在改进，MLP 早期 saturate。这是 paper 的 key claim：**MLP 在 100M+ tokens 时 capacity bottleneck，Transformer 没有**。

**MLP 为什么 saturate？**

我的猜测：
1. MLP 只看 1 帧，没 temporal context，即使加 history 输入也要 flatten，参数膨胀快泛化差
2. MLP 的 inductive bias 不适合 structured temporal signal。attention 的 "selective temporal aggregation" 比 MLP 的 "global mapping" 更 efficient
3. MLP 容易 overfit 小 motion pattern，缺 in-context generalization

---

## Real-world 实验

### Sim-to-real transfer

Unitree-G1 上，4 个 unseen dance（Can Do Can Go!、Gokuraku Joudo、HuoYuanJia/Fearless、PokerFace）。Table 3 显示 real-world MPJPE 跟 simulation 接近——**zero-shot transfer 工作**。

怎么做到的？几个 factor：
1. Aggressive domain randomization
2. DAgger 让 student 见到 expert 的 recovery trajectory（不只是 ideal motion，还有 disturbed motion）
3. Causal mask 消除 train-test mismatch
4. Retargeting 在 inference 时实时进行——live MoCap → G1 joint space → Humanoid-GPT

### Latency optimization

最终部署 latency **< 1.5ms** on RTX 4090，50Hz control loop。优化：
1. ONNX export (FP32)
2. TensorRT engine，fused MLP + causal attention kernels
3. C++ streaming pipeline 减少 IPC latency

比 TWIST 快 ~5×。

这块你应该有共鸣——LLM deployment 里 kernel fusion + quantization + KV cache 这些优化是工程上决定能不能 deploy 的关键。这里也一样，端到端 1.5ms 在 RTX 4090 上不是 trivial 的，需要 TensorRT specific kernel。

---

## 我会 push back 的几个点

### 1. "Zero-shot" 定义偏弱

Paper 里 zero-shot 指 "test motions 不在 training set"。但：
- Retargeting pipeline train 和 test 都用，如果 test motion 经过同一 retargeter，retargeter 的 prior 会 leak
- Test motions (e.g. Can Do Can Go!) 是 famous dance，AMASS/MotionMillion 里可能有类似风格的，只是 exact clip 不同。这更像 style transfer / in-distribution generalization，不是真正 OOD

要真正测 zero-shot，应该用一个 motion type 完全没在 training 里出现的，比如 ballet 或 breakdancing 如果 training 里没有这类。

### 2. "Generative" 这个词用得有点松

Title 叫 "Generative Pre-Training"，但实际是 **behavior cloning from RL experts**，不是 generative modeling——没有 next-token distribution，没有 sampling，没有 temperature。

更准确的名字应该是 "DAgger-distilled Transformer policy"。"GPT" 是 marketing-friendly 命名。

**真正的 risk**：student 的上限是 expert 的 union。如果某个 motion type 没有 expert 覆盖，student 学不会——这跟 LLM 的 open-ended generation 本质不同。LLM 可以 generate unseen token sequence，Humanoid-GPT 不能 generate unseen motion type。

要真正 generative，需要：
- 训一个 VAE 或 diffusion 在 action space 上
- 或者用 next-token prediction with discrete action tokens
- 这两条路都还在早期

### 3. MPKPE 40mm 仍然大

Humanoid-GPT-L 的 MPKPE 是 40.99mm，~4cm 的 keypoint position error。对 1.3m 高的 G1，这相当于 ~3% body length。

对 locomotion 这个精度够了，但 manipulation 需要 < 5mm。Paper 没讨论这个 limit 何时 break——比如要做精细抓取，40mm 误差直接抓空。

要 push 到 < 10mm 可能需要：
- 加 vision modality（当前是 pure proprioception）
- 加 contact sensing
- 更细的 keypoint（finger level，不只是 body keypoint）

### 4. Proprioception only 离 VLA 还远

Conclusion 里提到 future work 加 vision/language。但当前 model 是 pure proprioception + reference pose。要 extend 到 VLA-style instruction，需要重设计 token structure 和 supervision——不是 trivial extension。

VLA 那边 RT-2、OpenVLA 这些已经把 vision-language 跟 action 接起来了，Humanoid-GPT 这个 pipeline 要 plug in vision 至少要解决：
- vision token 怎么跟 proprioception token fusion
- 训练数据需要 vision-action pair，不只是 motion-action pair
- inference latency 会爆——vision encoder 加进去 1.5ms 肯定打不住

---

## 我的整体 take

这篇 paper 的 contribution 在我看来：

1. **First systematic evidence that scaling law holds for humanoid motion tracking**——这是 conceptual contribution，比 SOTA 重要。证明了 LLM 那套 recipe 在 robotics locomotion 上 work
2. **Causal Transformer + DAgger distillation 的 pipeline** 是可复制的 recipe，类似 nanoGPT 之于 LLM
3. **HME diversity metric** 是一个 useful tool，可推广到其他 robotics dataset curation

局限：
1. Distillation 不是 generation，capability 上限是 experts 的 union
2. Proprioception only，离 VLA 还远
3. Zero-shot 定义偏 weak
4. MPKPE 40mm 对 manipulation 不够

我会把它看作 **robotics foundation model 路线上的重要 milestone**，类似 GPT-2 之于 LLM——证明了 scaling 在这个 domain work，但还没到 GPT-3 的 "emergent capability" 时刻。

下一步如果有人能把这个 pipeline 改成真正的 generative（next-action prediction with sampling），加上 vision modality，并且 push MPKPE 到 < 10mm，那就是 robotics 的 GPT-3 时刻。

---

## 我会继续 dig in 的方向

如果你想 follow up，几个值得挖的点：

1. **HME 的 latent space 结构** vs VQ-VAE codebook 的优劣。HME 是 continuous embedding，VQ-VAE 是 discrete codebook。后者更接近 LLM 的 token，可能更适合做真正的 next-token prediction
2. **DAgger 在 32K parallel env 下的 implementation detail**——Paper 说用 32K envs，这远超一般 RL pipeline。框架大概率是 IsaacLab 或 rsl_rl，需要看 code
3. **Causal Transformer 的 RoPE vs ALiBi 选择**——Paper 没明说。如果用 RoPE，长 history 的 extrapolation 会更好
4. **如何把这个 pipeline 改成 latent diffusion over actions**——把 expert actions 用 VQ-VAE tokenize，然后用 latent diffusion generate action sequence，这可能是通往真正 generative 的路径
5. **HME 能否 extend 到 manipulation**——manipulation 不完全是周期性的，Periodic AE 的 assumption 可能不成立

---

## References

- Project page: https://qizekun.github.io/Humanoid-GPT/
- Code: https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/
- DAgger paper (Ross, Gordon, Bagnell, AISTATS 2011): https://proceedings.mlr.press/v15/ross11a.html
- DeepPhase / Periodic Autoencoder (Starke et al., SIGGRAPH 2022): https://dl.acm.org/doi/10.1145/3528223.3530178
- DeepMimic (Peng et al., SIGGRAPH 2018) - exponential reward form 来源: https://xbpeng.github.io/projects/DeepMimic/2018_TOG_deepmimic.pdf
- ASAP (sim-to-real agile humanoid): https://arxiv.org/abs/2502.01143
- SONIC (100M frames MLP baseline): https://arxiv.org/abs/2511.07820
- TWIST: https://arxiv.org/abs/2505.02833
- UniTracker: https://arxiv.org/abs/2507.07356
- Beyond-Mimic: https://arxiv.org/abs/2508.08241
- PHUMA dataset: https://arxiv.org/abs/2510.26236
- MotionMillion (ICCV 2025): https://arxiv.org/abs/2503.13336
- AMASS: https://amass.is.tue.mpg.de/
- Unitree G1: https://www.unitree.com/g1
- MuJoCo: https://mujoco.org/
- TensorRT: https://developer.nvidia.com/tensorrt
- ONNX Runtime: https://onnxruntime.ai/
- Scaling laws for neural language models (Kaplan et al.): https://arxiv.org/abs/2001.08361
- Chinchilla scaling law (Hoffmann et al.): https://arxiv.org/abs/2203.15556
- Emergent abilities of LLMs (Wei et al.): https://arxiv.org/abs/2206.07682
- Retargeting matters (Araujo et al.): https://arxiv.org/abs/2510.02252
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/

要不要我具体 dig in 某一块？比如 HME latent space 怎么 visualize 看 cluster 质量，或者 DAgger 在 32K env 上的具体实现细节，或者把这个 pipeline 改成 latent diffusion 的具体设计？

---

# Humanoid-GPT: 深度技术解析

Andrej, 这篇 paper 的核心 thesis 我一句话概括：**把 humanoid motion tracking 从 "MLP + 小数据" 的范式，迁移到 "GPT-style causal Transformer + 2B-frame corpus" 的 scaling regime，从而打破 agility-generalization tradeoff**。这本质上是一个 "scaling law 是否在 robotics locomotion control 中成立" 的实证 paper，而答案是肯定的——而且第一性原理上很自洽。

下面我尽量把每个模块的 intuition、公式、变量含义、以及与你熟悉的 LLM scaling 范式的对应关系都拆开讲。

---

## 1. 为什么这个工作值得认真读——与 LLM scaling 的同构性

你写过 nanoGPT，所以你应该会立即注意到一件事：这篇 paper 的整个 framing 几乎是把 GPT 的训练 recipe 1:1 搬到 humanoid control 上：

| LLM | Humanoid-GPT |
|---|---|
| Token = text subword | Token = (proprioception, target pose) at timestep $t$ |
| Causal mask (left-to-right) | Causal mask (past→present，online control 强制) |
| Pre-training corpus = web text | 2B-frame retargeted motion corpus |
| Teacher distillation (e.g. GPT-4 distill to GPT-4o-mini) | DAgger distill from 384 RL experts → 1 Transformer |
| Scaling law: loss vs. params/data/compute | Scaling law: SR / MPJPE vs. tokens / params |
| ONNX/TensorRT deployment | ONNX + TensorRT，1.5ms latency on RTX 4090 |

但有一个关键区别：**这里的 "token" 不是离散的，而是连续的 state-action pair**，并且 supervision 不是一个 ground-truth next-token，而是一个 RL expert 的 action。所以严格说这是一个 **behavior cloning + DAgger** 的 distillation，而不是真正的 next-token prediction。这一点我后面会展开讲它的风险。

---

## 2. 数据侧：Science of Scale 的第一支柱

### 2.1 数据来源与 curation

Corpus 聚合了：
- **AMASS** [24]: 经典 mocap，studio-constrained，~7.2M frames，是 prior work 的事实标准
- **LAFAN1** [9]: Robust motion in-betweening
- **MotionMillion** [7]: million-scale generated motions（注意是生成的，不是 captured）
- **PHUMA** [16]: physically grounded，带 contact modeling + joint constraints + reduced foot-sliding
- **Motion-X++** [43]: video-reconstructed multimodal
- **In-house capture**: 真实世界覆盖

总规模 **2B frames**，比 prior work（6–9M frames）大 **~200–300×**。

Curation pipeline：
1. **Retargeting**: 用 [2] 的 framework 把 human motion → Unitree-G1 的 29-DoF joint space。这一步会丢失信息（human 和 G1 的 kinematics 不一样），所以 retargeting quality 直接决定了 tracking ceiling。
2. **Filtering**: 移除 sitting on chairs / swimming / stair climbing 等需要 explicit object interaction 的 clip，因为 G1 在 plain scene 里做不到。
3. **Time-warping augmentation**: 对每条 clip uniform accelerate/decelerate，**5× expansion**。这步的 intuition 是让 policy 见到不同 speed 的同一 motion，提升 temporal robustness。

### 2.2 Harmonic Motion Embedding (HME) —— 这是这篇 paper 最有 idea 的部分之一

问题：你有 2B frames，但 **long-tail distribution** 严重——常见 walking/running 覆盖了 90% 的 mass，rare behaviors (kungfu, backflip) 被淹没。直接 uniform sample 训练，policy 会 collapse 到 dominant modes。

HME 的做法：
1. 在不同 data partition 上训练多个 **Periodic Autoencoder** [33]（DeepPhase 的工作）。Periodic AE 的核心是把 motion 压缩成 per-joint 的 **amplitude + frequency** 的周期性 latent——因为 locomotion 本质上是周期性的（gait cycle, arm swing, etc.）。
2. 对每条 sequence，aggregate 每个 joint 的 amplitude/frequency 的 **mean + std**，得到一个 compact embedding vector。
3. 在所有 HME embedding 上跑 **K-Means**，pairwise distance 当 similarity，得到 ~300 clusters，每个 cluster 1k–2k sequences。

为什么这个比直接在 raw motion space 聚类好？因为 raw motion 是高维 noisy signal，距离度量不可靠；HME 把它投影到一个 **harmonic latent space**，distance 在那里才有语义——两条 motion 如果 harmonic 结构相似（e.g. 都是 double-support gait + periodic arm swing），它们的 HME 就近，即使 amplitude 不同。

### 2.3 Diversity 的定量度量——公式 (3)

作者定义了两个指标来衡量 dataset 在 latent space 的 coverage：

$$\mathrm{gstd} = \exp\left(\frac{1}{D}\sum_{j=1}^{D}\log\sigma_j\right)$$

$$\mathrm{log\text{-}volume} = \frac{1}{2}\log\det(\Sigma + \epsilon I)$$

变量含义：
- $X = [x_1, x_2, \ldots, x_N]^\top \in \mathbb{R}^{N \times D}$: 从 dataset 里 uniformly sample 的 10,000 个 HME embedding
- $\Sigma$: 它们的 covariance matrix
- $\sigma_j$: $\Sigma$ 第 $j$ 个 dimension 的 standard deviation
- $D$: embedding dimension
- $\epsilon$: small regularization term，防止 $\det(\Sigma)$ singular

**gstd** 的 intuition：这是所有维度 std 的 **geometric mean**（注意是 $\exp(\frac{1}{D}\sum \log \sigma_j)$，不是 arithmetic mean）。几何均值对低方差维度更敏感——只要有一个维度 collapse，gstd 就会大幅下降。所以它衡量的是 "data 在每个 latent direction 上是否有 spread"。

**log-volume** 的 intuition：这是 covariance ellipsoid 体积的 log。如果 data 在 latent space 各方向都 spread，volume 就大。它比 gstd 更全局，因为它捕捉 cross-dimension correlation。

Figure 3 的结果：curated dataset 的 log-volume 比 AMASS 大 **4–5 个量级**。这是一个非常大的 gap——AMASS 本质上是在 latent space 一个相对紧凑的 blob，而 curated dataset 是一个 spread out 的 manifold。

**Karpathy 你应该立刻联想到**：这其实就是 LLM 里 "data diversity matters more than data size" 的同构现象。CommonCrawl 10T tokens 但全是 SEO spam，不如 1T tokens 高质量 web text。这里也是——2B frames 如果全是 walking，不如 200M frames 但覆盖 dance/jump/kungfu。

---

## 3. 模型侧：Causal Transformer 的设计动机

### 3.1 为什么 causal，不是 bidirectional？

Online tracking 的物理约束：at test time，policy at timestep $t$ **不能 access $t+1, t+2, \ldots$** 的 observation。这是硬件上的 hard constraint（机器人不能预知未来）。

Prior work (e.g. HumanPlus [8], BumbleBee [37]) 用了 Transformer 但不一定是 causal——它们可能在 training 时用 bidirectional attention，deploy 时用 sliding window causal，导致 train-test mismatch。

Humanoid-GPT 直接用 **GPT-style causal attention**，training 和 deployment 完全一致。这一点其实非常关键——它消除了 distribution shift，类似于 LLM 里 train 和 inference 用同一个 causal mask。

### 3.2 Architecture 细节

从 Table 7 可以读出来：
- **Num Layers**: 12
- **Channel dims**: 256 / 384 / 768 (对应 Humanoid-GPT-S / B / L)
- **Params**: 5.7M / 22.1M / 80.4M
- **History length $H$**: 32 frames (default)，可以到 64 但 quadratic cost
- **Optimizer**: AdamW, lr=1e-4
- **Training iteration**: 200k

Token 构造（公式 2 附近）：

$$e_t = \mathrm{concat}(s_t, q_t^{\mathrm{ref}})$$

- $s_t$: current proprioceptive state（per-joint position, velocity, root angular velocity, projected gravity, previous action）
- $q_t^{\mathrm{ref}}$: target reference pose at timestep $t$

输出：per-joint PD targets $a_t$。

这里有个微妙点：**没有 explicit position encoding 提到**，但既然是 GPT-style，应该是 RoPE 或 learned positional embedding。Paper 里没明说，可能要查 code。

### 3.3 为什么 Transformer 比 MLP scaling 更好？Table 2 的实证

这是 paper 最 important 的实证表。让我重读一下：

| Backbone | Tokens | Params | SR ↑ | MPJPE ↓ |
|---|---|---|---|---|
| MLP (3-layer) | 2M | 0.25M | 76.89 | 0.1191 |
| TCN (8-layer) | 2M | 0.65M | 81.48 | 0.0885 |
| Humanoid-GPT-S | 2M | 5.7M | 83.26 | 0.0853 |
| Humanoid-GPT-S | 20M | 5.7M | 86.02 | 0.0802 |
| Humanoid-GPT-B | 200M | 22.1M | 88.27 | 0.0793 |
| Humanoid-GPT-B | 2B | 22.1M | 90.43 | 0.0768 |
| Humanoid-GPT-L | 2B | 80.4M | 92.58 | 0.0735 |

关键观察：
1. **小数据 + 大模型 = overfitting**：MLP-L 在 2M tokens 上 SR=75.25%，比 MLP-S 的 76.89% 还差。这是经典 U-curve。
2. **Transformer 在 200M→2B 仍在提升**，而 MLP/TCN 在 200M 之后 marginal gains 很小——classic scaling saturation。
3. **MPKPE（keypoint position error, mm）的 gap 最明显**：TCN-L 56.15mm vs Humanoid-GPT-S 43.25mm，30% gap。这说明 Transformer 在 **精细 motion fidelity** 上的优势远大于在 coarse success rate 上的优势。

**Intuition**：MLP 只看 1 帧 history，本质是 stateless reflex controller。Transformer 看 32 帧 history，能 modeling **phase, momentum, intent**——例如它知道 "过去 0.5s 在做 arm swing 的上举 phase"，所以下一个 action 应该 continue 上举而不是 reverse。这是 LLM 里 "context window → in-context reasoning" 的同构现象。

---

## 4. 训练 Pipeline：专家 → 通才 Distillation

这是整个 system 最复杂的部分，分两 stage。

### 4.1 Stage 1: RL Motion Experts（公式 1）

对每个 HME cluster，train 一个 PPO expert。Policy 形式：

$$\pi: \mathcal{G} \times \mathcal{S} \mapsto \mathcal{A}$$

- $\mathcal{G}$: target reference pose space
- $\mathcal{S}$: proprioceptive observation space  
- $\mathcal{A}$: per-joint action space (PD targets)

State $s_t^{\mathrm{priv.}}$ 包含：
- per-joint positions
- per-joint velocities
- root angular velocity
- projected gravity（用来感知 body orientation 相对于 gravity）
- previous control action $a_{t-1}$（关键，给 policy 关于自身历史的信息）

Reward（公式 1）是 keypoint-level：

$$R_{\mathrm{kpt}}(t) = R_{\mathrm{pos}}(t) + R_{\mathrm{rot}}(t) + R_{\mathrm{vel}}(t) + R_{\mathrm{penal}}(t)$$

具体：

$$R_{\mathrm{pos}}(t) = \sum_{k \in \mathcal{K}} w_k \exp\left(-\alpha_{\mathrm{pos}} \|e_{k,t}^{\mathrm{pos}}\|_1\right)$$

$$R_{\mathrm{rot}}(t) = \sum_{k \in \mathcal{K}} w_k \exp\left(-\alpha_{\mathrm{rot}} \theta_{k,t}\right)$$

$$R_{\mathrm{vel}}(t) = \sum_{k \in \mathcal{K}} w_k \exp\left(-\alpha_{\mathrm{vel}} \|e_{k,t}^{\mathrm{vel}}\|_1\right)$$

变量：
- $\mathcal{K}$: tracked body keypoints 集合（arms, hips, feet, pelvis）
- $k$: keypoint index
- $w_k$: per-keypoint weight，lower body=1.5, upper body=0.75（lower body 更重要，因为 fall 风险更高）
- $e_{k,t}^{\mathrm{pos}} \in \mathbb{R}^3$: keypoint $k$ 在 timestep $t$ 的 position residual（humanoid vs reference）
- $e_{k,t}^{\mathrm{vel}} \in \mathbb{R}^3$: velocity residual
- $\theta_{k,t}$: rotation error，via SO(3) log map（把 rotation error 从 SO(3) 投影到 $\mathbb{R}^3$ 的 axis-angle representation）
- $\alpha_{\mathrm{pos}}=1.0, \alpha_{\mathrm{rot}}=2.0, \alpha_{\mathrm{vel}}=0.03$: scaling factors，控制 exponential decay 速度

**为什么用 exponential form $\exp(-\alpha \|e\|_1)$？**

这是 locomotion RL 里的经典 trick（最早可以追到 DeepMimic [Peng 2018]）：
- 当 error 小时，gradient 接近线性 → 平稳优化
- 当 error 大时，reward saturates 到 0 → 不让 outlier 主导 gradient
- L1 norm $\|e\|_1$ 比 L2 更 robust to outlier dimension

$R_{\mathrm{penal}}(t)$ 包括 self-contact penalty 和 smoothness penalty，防止 policy 找到 weird solutions（e.g. 抖动来 "fit" reference）。

Domain randomization（Table 4）很 aggressive：
- Floor friction: U(0.3, 2.0)
- External force interval: U(5.0, 10.0)s, velocity magnitude U(0.1, 1.0) m/s
- DoF friction scaling: U(0.5, 2.0)
- Torso CoM position change: U(-0.15, 0.15) m
- 等等

这些 randomization 是 sim-to-real 的关键——它们让 expert 在 disturbed dynamics 下也 stable，distillation 后 student 继承这个 robustness。

### 4.2 Stage 2: DAgger Distillation（公式 2）

这是 student Transformer 的训练。DAgger [31] 的核心：student rollout，expert 提供 label，迭代训练。这避免了 naive BC 的 covariate shift 问题——naive BC 只在 expert trajectory 上训练，student 一旦偏离就 OOD。

公式 (2)：

$$\hat{a}_{t-H+1:t} = \bigcup_{t_i \in \mathcal{T}} \mathrm{concat}\ t_i(s_{t-k}^{\mathrm{priv.}}, g_{t-k})$$

$$l = \mathcal{L}(G_\theta(e_{t-H+1:t}), \hat{a}_{t-H+1:t})$$

变量：
- $\mathcal{T}$: 所有 expert teachers 的集合（~384 个）
- $t_i$: 第 $i$ 个 teacher
- $H=32$: history length
- $e_{t-H+1:t}$: 长度 $H$ 的 token sequence
- $G_\theta$: student Transformer
- $\hat{a}_{t-H+1:t}$: aggregated teacher actions
- $\mathcal{L}$: SmoothL1Loss

**关键设计点**：
1. **一次 forward pass, 所有位置都 supervise**——这是 GPT 训练的 standard trick（不是只 predict last token，而是每个 position 都 predict next）。这让 training 极其 sample efficient。
2. **Teacher 选择**：对每个 state，用哪个 expert 的 action？paper 没明说，但从 "We use SmoothL1Loss" 和 "concat $t_i(s, g)$" 推测，应该是用 cluster membership 决定——每个 state 对应一个 reference motion clip，clip 属于某个 cluster，用那个 cluster 的 expert 当 teacher。
3. **Batch size ≥ #experts**：防止 mode collapse，让每个 batch 至少看到所有 expert 的 supervision。

### 4.3 为什么不直接 PPO 训练 Transformer？

这是你可能会问的 question。Paper 没直接 discuss，但 intuition 是：
1. **PPO 在 billion-scale data 上不稳定**——RL 优化 landscape 是 non-stationary 的，scale 到 2B frames 时 reward hacking 严重。
2. **Expert distillation 解耦了 "RL 探索" 和 "scaling"**：每个 expert 只在自己 cluster 上 RL，问题规模可控；distillation 是 supervised learning，scale 干净。
3. **Compute 效率**：RL 需要 environment interaction，slow；distillation 是 offline supervised，可以 fully parallelize。Table 8 显示 PPO experts 12,000 GPU hours（75%），distillation 只 3,000 GPU hours（25%）——distillation 单位 compute 的 "data processed" 远高于 PPO。

这个 pipeline 跟 LLM 里的 **GPT-4 distill to small model** 几乎一模一样。

---

## 5. Scaling Law Analysis

### 5.1 Data scaling（Figure 7）

固定 Humanoid-GPT-B 架构，变化 $T \in \{2M, 20M, 200M, 2B\}$，**non-overlapping subsets**——这点很重要，确保是 data 本身的 scaling，不是 epoch 数增加。

观察：2B tokens 的 marginal gain 比 200M→2B 之间开始递减，但仍然在改进。说明 **当前 model capacity 在 2B 开始进入 data-limited regime**——继续加 data 还会有收益，但 model capacity 也需要同步 scale。

这跟 LLM 的 Chinchilla scaling law 类似——data 和 model 必须同步 scale，否则一方 saturate。

### 5.2 Model scaling（Figure 8）

固定 2B tokens，compare Transformer-B vs MLP（comparable params）。

观察：Transformer 仍在稳步改进，MLP 早期 saturate。这是 paper 的 key claim：**MLP 在 100M+ tokens 时 capacity bottleneck，Transformer 没有**。

**为什么 MLP saturate？**

我猜测有几个原因：
1. **MLP 只看 1 帧**，没有 temporal context。即使加 history 输入，也需 flatten，参数膨胀快，泛化差。
2. **MLP 的 inductive bias 不适合 motion**——motion 是 structured temporal signal，attention 的 "selective temporal aggregation" 比 MLP 的 "global mapping" 更 efficient。
3. **MLP 容易 overfitting 小 motion pattern**，缺乏 in-context generalization 能力。

---

## 6. Real-world Deployment

### 6.1 Sim-to-real transfer

Real-world evaluation 在 Unitree-G1 上，4 个 unseen dance motions（Can Do Can Go!, Gokuraku Joudo, HuoYuanJia/Fearless, PokerFace）。Table 3 显示 real-world MPJPE 跟 simulation 相近——**zero-shot transfer 工作**。

这是怎么做到的？几个 factor：
1. **Aggressive domain randomization** in PPO experts
2. **DAgger 让 student 见到 expert 的 "recovery" trajectory**——不仅 ideal motion，还有 disturbed motion
3. **Causal mask 消除 train-test mismatch**
4. **Retargeting 在 inference 时实时进行**——live MoCap → G1 joint space → Humanoid-GPT

### 6.2 Latency optimization（Figure 5）

最终部署 latency **< 1.5ms** on RTX 4090，50Hz control loop。优化：
1. ONNX export (FP32)
2. TensorRT engine，fused MLP + causal attention kernels
3. C++ streaming pipeline 减少 IPC latency

比 TWIST 快 **~5×**。这点对你应该有共鸣——LLM deployment 里 kernel fusion + quantization 也是同样的工程努力。

---

## 7. 我会 push back 的几个点

### 7.1 "Zero-shot" 的定义问题

Paper 里 zero-shot 指 "test motions 不在 training set 里"。但：
1. Retargeting pipeline 在 train 和 test 都用——如果 test motion 经过同一 retargeting，retargeter 本身的 prior 会 leak 进来。
2. Test motions (e.g. Can Do Can Go!) 是 famous dance，AMASS/MotionMillion 里可能有类似风格的 motion，只是 exact clip 不同。这更像是 **style transfer / in-distribution generalization**，不是真正的 OOD generalization（e.g. 从没见过的 motion type）。

### 7.2 Distillation vs. True Generative Modeling

Title 叫 "Generative Pre-Training"，但实际是 **behavior cloning from RL experts**，不是 generative modeling（没有 next-token distribution，没有 sampling，没有 temperature）。

更准确的名字应该是 "DAgger-distilled Transformer policy"。但 "GPT" 是 marketing-friendly 的命名。

**真实的 risk**：student 上限是 expert 的 union。如果某个 motion type 没有 expert 覆盖，student 学不会——这跟 LLM 的 open-ended generation 本质不同。LLM 可以 generate unseen token sequence，Humanoid-GPT 不能 generate unseen motion type。

### 7.3 MPKPE 的 absolute number

Humanoid-GPT-L 的 MPKPE 是 40.99mm。这是 ~4cm 的 keypoint position error。对于 1.3m 高的 G1，这相当于 ~3% body length。在很多精细 task 里（e.g. 抓取），这仍太大。

Paper 没讨论这个 limit 何时 break——例如 manipulation 需要的精度可能 < 5mm。

### 7.4 Lack of vision/language modality

Conclusion 里提到 future work 加 vision/language。但目前 model 是 **pure proprioception + reference pose**。要 extend 到 VLA-style instruction，需要重设计 token structure 和 supervision——不是 trivial extension。

---

## 8. 我的整体 take

这篇 paper 的 contribution 在我看来是：
1. **First systematic evidence that scaling law holds for humanoid motion tracking**——这是 conceptual contribution，比 SOTA 重要。
2. **Causal Transformer + DAgger distillation 的 pipeline** 是可复制的 recipe，类似 nanoGPT 之于 LLM。
3. **HME diversity metric** 是一个 useful tool，可推广到其他 robotics dataset curation。

但它的局限：
1. **Distillation 不是 generation**——capability 上限是 experts 的 union
2. **Proprioception only**——离 VLA 还很远
3. **Zero-shot 定义偏 weak**

我会把它看作 **"robotics foundation model" 路线上的一个重要 milestone**，类似于 GPT-2 之于 LLM——证明了 scaling 在这个 domain 工作，但还没到 GPT-3 的 "emergent capability" 时刻。

---

## Reference

- Project page: https://qizekun.github.io/Humanoid-GPT/
- Code: https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/
- DAgger paper: Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS 2011, https://proceedings.mlr.press/v15/ross11a.html
- DeepPhase (Periodic Autoencoder): Starke, Mason, Komura, *DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds*, SIGGRAPH 2022, https://dl.acm.org/doi/10.1145/3528223.3530178
- DeepMimic (exponential reward form): Peng et al., *DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills*, SIGGRAPH 2018, https://xbpeng.github.io/projects/DeepMimic/2018_TOG_deepmimic.pdf
- ASAP (sim-to-real agile humanoid): https://arxiv.org/abs/2502.01143
- SONIC (100M frames MLP baseline): https://arxiv.org/abs/2511.07820
- TWIST: https://arxiv.org/abs/2505.02833
- UniTracker: https://arxiv.org/abs/2507.07356
- Beyond-Mimic: https://arxiv.org/abs/2508.08241
- PHUMA dataset: https://arxiv.org/abs/2510.26236
- MotionMillion: https://arxiv.org/abs/2503.13336 (ICCV 2025)
- AMASS: https://amass.is.tue.mpg.de/
- Unitree G1: https://www.unitree.com/g1
- MuJoCo: https://mujoco.org/
- TensorRT: https://developer.nvidia.com/tensorrt
- ONNX Runtime: https://onnxruntime.ai/
- Scaling laws for neural language models (Kaplan et al.): https://arxiv.org/abs/2001.08361
- Chinchilla scaling law (Hoffmann et al.): https://arxiv.org/abs/2203.15556
- Behavior cloning & covariate shift (经典 reference): Pomerleau, *ALVINN: An Autonomous Land Vehicle in a Neural Network*, NeurIPS 1989, https://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network
- Emergent abilities of LLMs (Wei et al.): https://arxiv.org/abs/2206.07682
- GPT-4o system card: https://arxiv.org/abs/2410.21276
- Retargeting matters (Araujo et al.): https://arxiv.org/abs/2510.02252

---

如果你愿意，下一步可以深入聊：
- **HME 的 latent space 结构** vs VQ-VAE codebook 的优劣
- **DAgger 在 32K parallel env 下的 implementation detail**（IsaacLab? rsl_rl?）
- **Causal Transformer 的 RoPE / ALiBi 选择**（paper 没明说，需要看 code）
- **MPKPE 40mm 这个数字能否 push 到 < 10mm**——需要什么 architectural change
- **distillation 到 generation 的跨越**——能否把这个 pipeline 改成 latent diffusion over actions

哪个方向你最想 dig in？
