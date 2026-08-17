---
source_pdf: RDT2 Exploring the Scaling Limit of UMI Data Towards.pdf
paper_sha256: 6eb3a1d0ff2c17aeac9241f6aef0cb4caed2304560e98431dbf1c49968c2cbf1
processed_at: '2026-08-11T21:07:46-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RDT2 人话版

## 一句话讲清楚

他们拿一个手持夹子去 100 个家庭采了 1 万小时数据，喂给一个 7B 的 VLM，训完之后这个模型可以直接 zero-shot 扔到一个从没见过的机械臂上干活 —— 抓东西、擦桌子、抖瓶子都能做。换机器人、换场景、换物体、换指令，四个维度同时 unseen，模型还能 work。这件事以前没人做到过。

---

## 为什么这件事难

Robot learning 圈子一直有个尴尬：你训了一个模型在 Franka 上抓杯子特别 6，换到 UR5e 上它就傻了。原因很朴素 —— 不同机器人关节长度、工作空间、控制频率都不一样，model 学到的 action distribution 是绑死 hardware 的。以前大家要么 fine-tune（要重新采几百小时数据），要么搞一个统一的 action embedding（Open X-Embodiment、Octo 走这条路，但 zero-shot 还是不行）。

另一头，VLA 模型还有个 architecture 困境。Action 是连续的低维向量（14 维左右），但 VLM 输出的是 discrete token probability。你要怎么把这两头对接？

- 方案 A：把 action 离散化成 token，让 VLM autoregressive 预测。RT-2、OpenVLA 这么干。问题：quantization error + autoregressive 采样慢，打乒乓球这种高频任务直接废掉。
- 方案 B：直接用 diffusion model 输出连续 action。Diffusion Policy、π0 这么干。问题：diffusion loss 收敛慢，而且会损坏 VLM 里原本以 discrete probability 存储的 knowledge。

RDT2 的核心 insight：**两个都要，分阶段来**。先用离散方式预训练保护 VLM knowledge，再用连续方式 fine-tune 拿精度，最后 distill 拿速度。

---

## 数据是怎么来的

这是整个 paper 最"重"的部分。

原版 UMI（Chi et al. 2024, [arXiv:2402.10329](https://arxiv.org/abs/2402.10329)）是一个手持设备：一个 gripper + 一个 camera + 一个 SLAM tracker。人握着它在真实环境里做任务，它记录 6-DoF end-effector pose + gripper width。为什么这玩意儿能 cross-embodiment？因为不管你最后部署到什么 robot，只要装上同样的 gripper + 同样的 camera 视角，数据的 physical 语义就是一致的。

但原版 UMI 有几个毛病，扛不住 10,000 小时的大规模采集：

1. **3D 打印件不够硬**：PLA/PETG 累计几千小时后会形变，pose drift 累积
2. **SLAM 在白墙、玻璃、镜面场景会漂**：tracking 误差大
3. **Parallel jaw 够不到窄缝**：抽屉缝隙、cluttered space 够不着

RDT2 的硬件改造（Tab. 1）：

| Spec | Naive UMI | RDT2 UMI |
|---|---|---|
| Fabrication | 3D printing | CNC nylon 66 + glass fiber |
| Tracking | SLAM | Infrared (HTC VIVE Tracker 3.0) |
| Gripper | Parallel jaws | Linkage gripper (ZhiXing CTAG2F120) |

打比方：原版 UMI 像塑料玩具，RDT2 UMI 像工业级手持工具。Infrared tracker 给的是绝对位姿，不依赖环境 texture，白墙玻璃都不怕。Linkage gripper 像剪刀结构，能伸进窄缝。

他们造了 100 个这样的设备，发给 100+ 个家庭，采了 10,000 小时数据。涵盖 50+ 种任务（pick、pour、wipe、shake、stir、fold...），1000+ 种物体。这部分是 paper 最贵的工程投入，也是 zero-shot 能 work 的物理基础。

---

## 三阶段训练 pipeline

paper 的 Figure 2 是整个 story 的核心，三个 stage 一个一个讲。

### Stage 1：RVQ 离散化 + Cross-Entropy 预训练 VLM

**Goal**：让 7B Qwen2.5-VL 学会 "语义化的 action reasoning"，同时保护它预训练得到的 discrete probability knowledge。

**怎么做**：先把 action chunk 压成离散 token。具体地，一个 action chunk $\mathbf{A}_t \in \mathbb{R}^{T_a \times d}$（$T_a=32$ 是 chunk size，$d=14$ 是双臂 action 维度），先用 1D temporal CNN encoder $\phi_{\text{enc}}$ 压成 $n$ 个 $C$-维 latents $\{\mathbf{z}_i\}_{i=1}^n$。然后对每个 $\mathbf{z}_i$ 做 $m$ 层 residual quantization：

$$k_j^i = \arg\min_{1 \leq k \leq K} \|\mathbf{r}_{j-1}^i - \mathbf{e}_j(k)\|_2^2$$
$$\mathbf{r}_j^i = \mathbf{r}_{j-1}^i - \mathbf{e}_j(k_j^i)$$

讲人话：每一层有一个 codebook（大小 $K$），找离当前 residual 最近的 entry，记下它的 index $k_j^i$，然后把 residual 减掉这个 entry，进入下一层继续量化。最终一个 action chunk 被表示成 $n \times m$ 个 token index。

为什么用 RVQ 不用 FAST tokenizer 或者 uniform binning？Fig. 8 给了答案：相同 quantization error 下，RVQ 比 FAST 节省约 2/3 的 token，比 uniform binning 节省更多。token 少 = autoregressive 步数少 = 收敛快 + 推理快。RVQ 来自 audio compression 领域（SoundStream [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)、EnCodec），action 是低维连续信号，跟 audio 很像，用 audio 的成熟工具非常合理。

Tokenizer 训练 loss（Eq. 2）：

$$\mathcal{L}_{\text{vq}} = \mathbb{E}\Big[\underbrace{\|\mathbf{A}_t - \hat{\mathbf{A}}_t\|_2^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}(\mathbf{z}_i) - \hat{\mathbf{z}}_i\|_2^2}_{\text{codebook}} + \underbrace{\beta\|\mathbf{z}_i - \text{sg}(\hat{\mathbf{z}}_i)\|_2^2}_{\text{commitment}}\Big]$$

三项分别是 reconstruction loss、codebook loss（把 codebook entry 拉向 encoder output）、commitment loss（把 encoder output 拉向 codebook entry）。$\text{sg}(\cdot)$ 是 stop-gradient，$\beta$ 是 commitment weight。这是 VQ-VAE（[arXiv:1711.00937](https://arxiv.org/abs/1711.00937)）的标准 loss 扩展到 RVQ。

**Codebook collapse 防御**（这是 VQ 训练的经典坑）：低维 codebook、cosine similarity 替代 Euclidean、EMA 更新、inactive entry 周期重启。

然后把这个 RVQ 训出的 tokenizer 挂到 Qwen2.5-VL 上 —— 在 vocabulary 里保留 1024 个最低频位置给 action token，用标准 cross-entropy next-token prediction 训 128K steps。同时混入 vision-language data（Ego4D、HD-EPIC、RoboVQA、RoboBrain、PixMo-Cap-QA、Cambrian-10M，共 12M+ VQA pairs），防止 VLM 能力退化。

**为什么先离散预训练这么重要**？Fig. 6 的 ablation 给了答案：纯 diffusion 训练 loss 下降慢，AR pre-training + diffusion fine-tuning 收敛快 3-5 倍，最终 loss 更低。原因有两个：第一，VLM 的 knowledge 是以 discrete token probability 存储的，cross-entropy 是它的 native objective，不会破坏这种 distribution；第二，AR pre-training 给了 diffusion 阶段一个好初始化。

### Stage 2：Flow Matching 训练 action expert

**Goal**：拿到 high-precision continuous action，同时利用 Stage 1 训好的 VLM embedding。

**架构**：freeze 住 7B VLM backbone，新加一个 400M 的 action expert（RDT-1B 变体，14 层、hidden 1024、8 heads、4 KV heads，用 GQA [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) 加速）。Action expert 通过 cross-attention 把 VLM 每一层的 latent 都注入进来。

**Flow matching loss**（Eq. 3，[Lipman et al. 2022, arXiv:2210.02747](https://arxiv.org/abs/2210.02747)）：

$$\mathcal{L}_{\text{expert}}(\theta) = \mathbb{E}_{\{\ell, \mathbf{o}_t, \mathbf{A}_t\} \sim \mathcal{D}, \tau \sim \mathcal{U}(0,1)}\Big[\|\mathbf{v}_\theta(\tau, \mathbf{A}_t^\tau, \text{VLA}(\ell, \mathbf{o}_t)) - \mathbf{u}(\mathbf{A}_t^\tau | \mathbf{A}_t)\|_2^2\Big]$$

变量含义：
- $\tau \in [0, 1]$ 是 flow 时间，$\tau=0$ 纯噪声，$\tau=1$ 是 clean action
- $\mathbf{A}_t^\tau := (1-\tau)\boldsymbol{\epsilon} + \tau \mathbf{A}_t$ 是线性插值（OT-CFM 形式），$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\mathbf{v}_\theta(\cdot)$ 是 denoising network，$\theta$ 可训练
- $\text{VLA}(\ell, \mathbf{o}_t)$ 是 frozen VLM 的输出，整个 flow 过程只算一次（很关键的 efficiency trick）
- $\mathbf{u}(\mathbf{A}_t^\tau | \mathbf{A}_t) := \mathbf{A}_t - \boldsymbol{\epsilon}$ 是 ground-truth velocity field

推理就是 5 步 Euler integration（Eq. 4，$\delta\tau = 0.2$）：

$$\mathbf{A}_t^{\tau+\delta\tau} = \mathbf{A}_t^\tau + \delta\tau \cdot \mathbf{v}_\theta(\tau, \mathbf{A}_t^\tau, \text{VLA}(\ell, \mathbf{o}_t))$$

讲人话：flow matching 就是学一个 vector field，把噪声沿着一条直线"流"到 clean action。比 DDPM 的 1000 步快得多，5 步就行。

**Timestep sampling 的小 trick**：实际用 Logistic Normal（$\mu=0, \sigma=1$）采样 $\tau$，把训练 budget 集中在 $\tau \approx 0.5$ 这种 hard region，而不是 uniform 采样。empirically 更有效。

训了 66K steps，VLM frozen，只更新 action expert。

### Stage 3：One-step distillation

**Goal**：5 步积分对 table tennis 这种 high-dynamic 任务还是太慢。球飞过来 100ms 级别到位，5 步积分 + VLM forward 可能 50-100ms，反应不过来。需要 single-step inference。

**Distillation loss**（Eq. 5）：

$$\mathcal{L}_{\text{distill}}(\theta') = \mathbb{E}\Big[\|\mathcal{F}(\mathbf{A}_t^0, \ell, \mathbf{o}_t; \theta) - G(\mathbf{A}_t^0, \ell, \mathbf{o}_t; \theta')\|_2^2\Big]$$

- $\mathcal{F}(\cdot)$ 是 Stage 2 的 multi-step teacher，参数 $\theta$ frozen
- $G(\mathbf{A}_t^0, \ell, \mathbf{o}_t; \theta') := \mathbf{A}_t^0 + \mathbf{v}_{\theta'}(0, \mathbf{A}_t^0, \text{VLA}(\ell, \mathbf{o}_t))$ 是 student single-step generator，$\theta'$ 从 $\theta$ 初始化

**关键 trick**：teacher $\mathcal{F}(\cdot)$ 是 on-the-fly 算的，不预生成 dataset。这避免了 regression-based distillation 常见的 overfit to pre-generated data 问题。因为 action 是低维（14 维），5 步积分很快，on-the-fly 几乎不增加训练成本。这和图像/视频 diffusion distillation（hundreds of steps）完全不同，那种必须预生成。

Fig. 7 结果：RDT2-UltraFast 推理频率全场最高，比 π0.5（3B）还快，比 π0-FAST 也快，尽管参数量（7.4B）是它们两倍多。这就是 distillation 的威力。

---

## Zero-shot 4U 实验

这是 paper 最 impressive 的部分。他们设计了一个 "4U" protocol：
- **U**nseen embodiment（Franka FR3 + UR5e，都没在训练集里）
- **U**nseen scene（3 个新场景）
- **U**nseen object（100+ 新物体）
- **U**nseen instruction（语言增强 + 去重）

5 个 task：Pick、Pick & Place、Wiping、Shaking、Button Pressing。

**Statistical rigor**：Pick Task 上做了 1000 trials 验证 success rate 收敛（Fig. 4），后续实验统一 256 trials。Robotics paper 里做 1000 trials 来验证 statistical reliability 非常罕见，一般 10-20 trials 就上报了。这个 rigor 值得点赞。

结果（Fig. 3）：success rate 不算高（Pick & Place、Wiping 在 20-40% 量级），但 statistically significant。这本身是 milestone：**纯 human UMI 数据训出的模型能 zero-shot transfer 到 robot，同时 4U 全部 unseen**。

---

## Scaling law

他们跑了 4 个 model size，用 Hoffmann/Chinchilla 形式拟合（Eq. 6）：

$$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

- $N$ = model parameters
- $D$ = tokens consumed（每个 token 只用一次，1 epoch）
- $E$ = irreducible loss

拟合结果：$E \approx 2.11$, $A \approx 4.38 \times 10^3$, $\alpha \approx 0.44$, $B \approx 1.79 \times 10^2$, $\beta \approx 0.23$

**Intuition**：$\alpha \approx 0.44$ 比 LLM 的 $\alpha \approx 0.076$（Kaplan）/ $0.34$（Chinchilla）大很多，说明 robotics data 稀缺，model scale 更值钱。$\beta \approx 0.23$ 也比 LLM 大，data scale 回报显著。两个都要 scale 才能持续降 loss。这给 robotics foundation model 的未来投入提供了定量依据：**大规模 wearable data collection 是 critical path**。

---

## Fine-tuning 实验

baseline 是 π0.5（[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)）和 π0-FAST（[arXiv:2501.09747](https://arxiv.org/abs/2501.09747)），都用 OpenPI 官方代码训到收敛。

**Tab. 2 关键数据**：

| Task | Metric | RDT2 | π0.5 | π0-FAST |
|---|---|---|---|---|
| Cloth Folding | Success Rate | **77%** | 36% | 29% |
| Cloth Folding | Unseen Object | **51%** | 15% | 10% |
| Table Bussing | Progress Score | **0.58** | 0.39 | 0.30 |
| Unzipping | Success Rate | **45%** | 13% | 8% |
| Button Pressing | Reaction Time vs human (2661ms) | **+97ms** | +323ms | +981ms |
| Table Tennis | Hit Rate (1x/1.2x/1.5x/1.7x/2x) | **88/85/76/69/68** | 78/74/58/57/56 | N/A（太慢打不到球）|

讲人话：
- **Deformable object**（cloth folding、unzipping）RDT2 大幅领先，4 倍于 baseline。UMI 数据 + flow matching 都对 multimodal action distribution 友好。
- **Table tennis** 上 π0-FAST 直接 fail，说明 autoregressive 离散化对 high-dynamic 任务有根本性 bottleneck。RDT2-UltraFast 的 single-step distillation 是关键。
- **Reaction time** 差距（97ms vs 323ms vs 981ms）直接体现推理速度差异。

---

## Ablation

- **AR + Diffusion vs pure Diffusion**（Fig. 6）：纯 diffusion 训练 loss 下降慢 3-5 倍，AR pre-training + diffusion fine-tuning 收敛快、最终 loss 更低。验证 Stage 1 的核心 motivation。
- **Discretization 对比**（Fig. 8）：相同 quantization error 下，RVQ 比 FAST 节省约 2/3 tokens。RVQ 的 compact latent space 是关键。
- **Inference frequency**（Fig. 7）：RDT2-VQ（autoregressive）因 RVQ 紧凑所以 token 少、比 π0-FAST 快；RDT2-UltraFast（single-step distillation）全场冠军。

---

## 我的看法

**这篇 paper 真正的 contribution**：把 UMI 数据 scale、hybrid AR/Diffusion 三阶段训练、single-step distillation 这三件事拼在一起，实现了 robotics 第一次真正意义上的 zero-shot cross-embodiment + open-vocabulary generalization。工程上很重，思路很 clean。

**几个 critical 思考**：

1. **Zero-shot success rate 不高**（Pick & Place、Wiping 在 20-40%）。离 production-ready 还有距离，但作为 research milestone 很强。

2. **数据 collection 成本巨大**：100 个 UMI + 10,000 hours + 100 households。这是巨大的资本投入，复现这个数据集很难。Open-source 出来后会非常有价值。

3. **VLM 在 Stage 2/3 完全 frozen**：保护 knowledge 不被损坏，但也限制了 adaptation。如果 cross-embodiment gap 大（比如人形机器人），frozen VLM 可能不够。

4. **Cross-embodiment 的真正 holy grail**：Open X-Embodiment、Octo、RDT-1B 走统一 embedding space 路线，仍需 fine-tune。UMI 的 physical interface unification（同样 gripper + 同样 camera）是另一种思路，更工程化，但 zero-shot 效果可能更稳。两条路未来可能 merge。

5. **RVQ 来自 audio compression**：robotics 里 action 是连续低维信号，跟 audio 很像，用 audio 领域成熟工具非常合理。FAST tokenizer 用 DCT，在 sharp action 变化上不如 RVQ 表达力强。

6. **没讲 safety**：zero-shot cross-embodiment deployment 在 unseen 物理环境是 dangerous 的。paper 只在 Impact Statement 顺带提了 "需要 safety guardrails"，没具体讲。这是 future work。

7. **Table tennis 这个 benchmark 很厉害**：它直接把 inference latency 暴露出来，是少数能 benchmark 控制频率的 real-world task。比 static pick-place 严格得多。以后 robotics paper 应该多用这种 dynamic task。

8. **Privacy concern**：100+ households 数据采集，paper 的 Impact Statement 提到了 anonymization，但没说具体怎么做。这是个隐患。

---

## 一句话总结

RDT2 = UMI 数据 scale + RVQ 离散预训练保护 VLM knowledge + Flow Matching 连续 fine-tune 拿精度 + Distillation 单步推理拿速度。三阶段 pipeline 每一步都有明确的 motivation，拼在一起实现了 robotics 第一次真正的 zero-shot cross-embodiment + open-vocabulary generalization。Scaling law 给出了 quantitative 指引：大规模 wearable data collection 是 critical path。

如果你想 build on 这篇，我会关注三件事：
1. 能不能 scale 到 100K hours？数据 scale 的回报是否持续？
2. Frozen VLM 在更 extreme cross-embodiment（比如人形机器人、四足）下是否还够用？
3. Safety guardrail 怎么加？zero-shot deployment 在 unseen 物理环境是 dangerous 的。

---

## Reference links

- UMI: https://arxiv.org/abs/2402.10329
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RDT-1B: https://arxiv.org/abs/2410.07864
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π0-FAST: https://arxiv.org/abs/2501.09747
- OpenVLA: https://arxiv.org/abs/2406.09246
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Flow Matching: https://arxiv.org/abs/2210.02747
- VQ-VAE: https://arxiv.org/abs/1711.00937
- SoundStream (RVQ): https://arxiv.org/abs/2107.03312
- ViT-VQGAN: https://arxiv.org/abs/2110.04627
- VQ-VAE-2: https://arxiv.org/abs/1906.00446
- Chinchilla scaling: https://arxiv.org/abs/2203.15556
- Kaplan scaling: https://arxiv.org/abs/2001.08361
- Consistency Models: https://arxiv.org/abs/2303.01469
- Progressive Distillation: https://arxiv.org/abs/2202.00512
- Consistency Policy: https://arxiv.org/abs/2405.07503
- One-Step Diffusion Policy: https://arxiv.org/abs/2410.21257
- GQA: https://arxiv.org/abs/2305.13245
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Octo: https://arxiv.org/abs/2405.12213
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- ALOHA 2: https://arxiv.org/abs/2405.02292
- DexUMI: https://arxiv.org/abs/2505.21864
- DROID: https://arxiv.org/abs/2403.12945
- FastUMI: https://arxiv.org/abs/2409.19499
- FastUMI-100K: https://arxiv.org/abs/2510.08022
- ACT: https://arxiv.org/abs/2304.13705
- RT-2: https://arxiv.org/abs/2307.15818
- Ego4D: https://arxiv.org/abs/2110.07058
- RoboVQA: https://arxiv.org/abs/2311.00899
- RoboBrain: https://arxiv.org/abs/2502.21257

---

# RDT2: 用 UMI 数据探索 cross-embodiment zero-shot 的 scaling limit

Hey Andrej, 这篇 paper 我读得很兴奋, 因为它把几条最近 robotics 界非常关键的 thread 缠绕到了一起: 大规模 in-the-wild data collection (UMI), VLM backbone 复用, RVQ tokenization, flow matching, 和 distillation。它甚至直接对标了 π0.5 和 π0-FAST, 在 cloth folding 和 table tennis 这种任务上取得了 SOTA。让我一层层拆开讲。

---

## 1. Intuition-level overview

这篇 paper 想回答一个根本问题: **能不能用一个只在人类 UMI 数据上预训练的模型, zero-shot 部署到一个全新的 robot embodiment 上, 同时处理 unseen object / scene / instruction?**

这听起来 crazy, 因为传统 VLA 比如 RT-2、OpenVLA、π0 都是绑定特定 robot 的, 换个 robot 就要几百小时重采集重 fine-tune。RDT2 的 trick 在于:
- **Data 这侧**: UMI 是 embodiment-agnostic 的手持设备, 采集的 6-DoF pose + gripper width 本身就和具体 robot 解耦, 只要 deployment robot 装同样的 gripper + 同样的 camera 视角, 就有 minimal embodiment gap。
- **Model 这侧**: 把 7B 的 Qwen2.5-VL 当作 "world knowledge + reasoning" 的容器, 用 RVQ 把 action 离散化进 VLM 的 vocabulary (保留 VLM 离散概率知识), 然后用一个 400M 的 action expert 做 flow matching 输出连续 action, 最后 distill 成单步推理。

所以整个 pipeline 本质上是 **"先让 VLM 学会语义化的 action reasoning, 再让一个小 expert 学会精细 motor control, 再用 distillation 把多步压成一步"**。这是 hybrid AR + diffusion 的核心思路, 解决了 pure diffusion 训练慢 + 损坏 VLM 离散知识的问题。

paper 的 figure 2 (three-stage pipeline) 是整个 story 的核心, 建议重点看。

---

## 2. Hardware: 重新设计 UMI

原版 UMI (Chi et al. 2024, [arXiv:2402.10329](https://arxiv.org/abs/2402.10329)) 用 3D-printed PLA/PETG + SLAM tracking + parallel jaws, 在大规模 in-the-wild 部署时 rig 不够稳定, tracking 在 texture-less / transparent 场景会漂移, gripper 在 cluttered 空间够不到。

RDT2 的改造 (Tab. 1):
| Spec | Naive UMI | RDT2 UMI |
|---|---|---|
| Fabrication | 3D printing PLA/PETG | CNC nylon 66 + glass fiber |
| Tracking | SLAM | Infrared (HTC VIVE Tracker 3.0 × 4) |
| End-effector | Parallel jaws | Linkage gripper (ZhiXing CTAG2F120 replica) |

Intuition: 3D 打印件在反复夹持、几千小时使用后形变累积, 导致 calibration drift; CNC nylon+GF 解决刚度问题。SLAM 在 wall 是白的、玻璃、镜面场景会有严重 drift, infrared tracker 是 external reference, pose 直接是绝对值。Linkage gripper 比 parallel jaw 更紧凑, 能伸进抽屉缝隙。

这 100 个 UMI 设备散到 100+ households 采 10,000 小时数据, 这是 paper 最 expensive 的部分, 也是它真正 scale 起来的物理基础。

---

## 3. 三阶段训练 pipeline (核心)

### Stage 1: RVQ + Cross-Entropy 预训练 VLM

**问题动机**: VLM 在大规模 text+image 上预训练后, 它的 knowledge 是以 *discrete token probability* 的形式存储的。如果你直接拿 flow matching / diffusion loss 训练, 就会破坏这种 discrete distribution (Deng et al. 2025, [arXiv:2505.14683](https://arxiv.org/abs/2505.14683) 有讨论)。同时, diffusion loss 收敛很慢 (Pertsch et al. 2025 的 π0-FAST paper 也提到这一点, [arXiv:2501.09747](https://arxiv.org/abs/2501.09747))。

**RVQ tokenizer** (Eq. 1, Eq. 2):

先把 action chunk $\mathbf{A}_t \in \mathbb{R}^{T_a \times d}$ 用 1D temporal CNN $\phi_{\text{enc}}$ 压成 $n$ 个 $C$-维 latents $\{\mathbf{z}_i \in \mathbb{R}^C\}_{i=1}^n = \phi_{\text{enc}}(\mathbf{A}_t)$。这里 $T_a$ 是 chunk size (paper 里设 32), $d$ 是 action 维度 (paper 里 14, 双臂 6-DoF pose + 2 gripper width), $n$ 是 latent 个数, $C$ 是 latent channel。

然后对每个 $\mathbf{z}_i$ 做 $m$ 层递归量化:

$$k_j^i = \arg\min_{1 \leq k \leq K} \|\mathbf{r}_{j-1}^i - \mathbf{e}_j(k)\|_2^2$$
$$\mathbf{r}_j^i = \mathbf{r}_{j-1}^i - \mathbf{e}_j(k_j^i)$$

变量含义:
- $j$ 是 RVQ 的层级 index (从 1 到 $m$), 每层用一个独立的 codebook
- $\mathbf{r}_j^i$ 是第 $j$ 层的 residual, $\mathbf{r}_0^i = \mathbf{z}_i$ 是初始 latent
- $\mathbf{e}_j \in \mathbb{R}^{K \times C}$ 是第 $j$ 层的 codebook, $K$ 是 codebook 大小
- $k_j^i \in \{1, ..., K\}$ 是第 $j$ 层选中的 codebook entry index
- 最终 token 序列是 $\{k_1^i, ..., k_m^i\}_{i=1}^n$

解码: $\hat{\mathbf{A}}_t = \phi_{\text{dec}}(\{\sum_{j=1}^m \mathbf{e}_j(k_j^i)\}_{i=1}^n)$

RVQ 训练 loss (Eq. 2):
$$\mathcal{L}_{\text{vq}} = \mathbb{E}\Big[\|\mathbf{A}_t - \hat{\mathbf{A}}_t\|_2^2 + \|\text{sg}(\mathbf{z}_i) - \hat{\mathbf{z}}_i\|_2^2 + \beta\|\mathbf{z}_i - \text{sg}(\hat{\mathbf{z}}_i)\|_2^2\Big]$$

三项分别是: reconstruction loss, codebook loss (把 codebook entry 拉向 encoder output), commitment loss (把 encoder output 拉向 codebook entry), $\beta$ 是 commitment weight, $\text{sg}(\cdot)$ 是 stop-gradient。这是 VQ-VAE 标准 loss ([Van Den Oord et al. 2017](https://arxiv.org/abs/1711.00937)) + RVQ 的扩展 ([Zeghidour et al. SoundStream](https://arxiv.org/abs/2107.03312))。

**Codebook collapse 防御**:
- 低维 codebook (Yu et al. 2021 ViT-VQGAN, [arXiv:2110.04627](https://arxiv.org/abs/2110.04627))
- 用 cosine similarity 替代 Euclidean (Eq. 1)
- EMA 更新 codebook (Razavi et al. VQ-VAE-2, [arXiv:1906.00446](https://arxiv.org/abs/1906.00446))
- 周期性重启 inactive entries (Zeghidour et al. SoundStream)

**关键 insight (Fig. 8)**: 在相同 quantization error 下, RVQ 比 FAST tokenizer 节省约 2/3 的 tokens, 比 uniform binning 节省更多。token 少 = autoregressive 步数少 = VLA 收敛快 + 推理快。

**Model detail**: Qwen2.5-VL 7B ([Bai et al. 2025, arXiv:2502.13923](https://arxiv.org/abs/2502.13923)) 作为 backbone, 在 vocabulary 中保留 1024 个最低频位置给 action token, 训练 128K iterations, next-token prediction, batch per-GPU 96, 7 节点 × 8 GPU = 56 GPU, 全局 batch 5376。同时还混入 vision-language 数据 (Ego4D, HD-EPIC, RoboVQA, RoboBrain, PixMo-Cap-QA, Cambrian-10M, 共 12M+ VQA pairs) 保持 VLM 能力不退化。

---

### Stage 2: Flow Matching 训练 action expert

**动机**: AR token 量化有 error, autoregressive 采样慢。要拿到 high-precision continuous action, 同时利用 Stage 1 已经训练好的 VLM embedding, 就 freeze VLM, 训一个 400M 的 action expert。

**Action expert 架构**: RDT-1B 变体 ([Liu et al. 2024, arXiv:2410.07864](https://arxiv.org/abs/2410.07864)), 14 层, hidden 1024, 8 heads, 4 KV heads, 用 GQA ([Ainslie et al. 2023, arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) 替代 MHA 加速。通过 cross-attention 把 VLM 每一层的 latent 都注入 action expert。

**Flow matching loss** (Eq. 3, Lipman et al. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)):

$$\mathcal{L}_{\text{expert}}(\theta) = \mathbb{E}_{\{\ell, \mathbf{o}_t, \mathbf{A}_t\} \sim \mathcal{D}, \tau \sim \mathcal{U}(0,1)}\Big[\|\mathbf{v}_\theta(\tau, \mathbf{A}_t^\tau, \text{VLA}(\ell, \mathbf{o}_t)) - \mathbf{u}(\mathbf{A}_t^\tau | \mathbf{A}_t)\|_2^2\Big]$$

变量:
- $\tau \in [0, 1]$ 是 flow 时间, $\tau=0$ 是纯噪声, $\tau=1$ 是 clean action
- $\mathbf{A}_t^\tau := (1-\tau)\boldsymbol{\epsilon} + \tau \mathbf{A}_t$ 是 linear interpolation (flow matching 的一种 OT-CFM 形式), $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- $\mathbf{v}_\theta(\cdot)$ 是 denoising network, $\theta$ 是可训练参数
- $\text{VLA}(\ell, \mathbf{o}_t)$ 是 frozen VLM backbone 的输出 (整个 flow 过程只算一次)
- $\mathbf{u}(\mathbf{A}_t^\tau | \mathbf{A}_t) := \mathbf{A}_t - \boldsymbol{\epsilon}$ 是 ground-truth velocity field

推理 (Eq. 4):
$$\mathbf{A}_t^{\tau+\delta\tau} = \mathbf{A}_t^\tau + \delta\tau \cdot \mathbf{v}_\theta(\tau, \mathbf{A}_t^\tau, \text{VLA}(\ell, \mathbf{o}_t))$$

从 $\tau=0$ 积分到 $\tau=1$, $\delta\tau = 0.2$, 即 5 步 Euler integration。这比 standard DDPM 1000 步快得多, 是 flow matching 的优势。

**Timestep sampling**: 实际用 Logistic Normal ($\mu=0, \sigma=1$) 采样 $\tau$ 而不是 uniform, 让训练 budget 集中在 $\tau \approx 0.5$ 这种 hard region。

训练 66K steps, frozen VLM, 只更新 action expert。

---

### Stage 3: One-step distillation

**动机**: 5 步积分对 table tennis 这种 high-dynamic 任务还是太慢 (球飞过来 100ms 级别就到位)。需要 single-step inference。

**Distillation loss** (Eq. 5):

$$\mathcal{L}_{\text{distill}}(\theta') = \mathbb{E}\Big[\|\mathcal{F}(\mathbf{A}_t^0, \ell, \mathbf{o}_t; \theta) - G(\mathbf{A}_t^0, \ell, \mathbf{o}_t; \theta')\|_2^2\Big]$$

变量:
- $\mathcal{F}(\cdot)$ 是 Stage 2 的 multi-step generation process (Eq. 4), 参数 $\theta$ frozen
- $G(\mathbf{A}_t^0, \ell, \mathbf{o}_t; \theta') := \mathbf{A}_t^0 + \mathbf{v}_{\theta'}(0, \mathbf{A}_t^0, \text{VLA}(\ell, \mathbf{o}_t))$ 是 student single-step generator, $\theta'$ 从 $\theta$ 初始化, 可训练

**关键 trick**: $\mathcal{F}(\cdot)$ 是 on-the-fly 计算的, 不预生成 dataset。这避免了 regression-based distillation 常见的 overfit to pre-generated data 问题。因为 action 是低维 (14 维), 5 步积分很快, on-the-fly 几乎不增加训练成本; 这和图像/视频 diffusion distillation (hundreds of steps) 完全不同。

这个思路和 consistency models ([Song et al. 2023](https://arxiv.org/abs/2303.01469))、progressive distillation ([Salimans & Ho 2022](https://arxiv.org/abs/2202.00512))、consistency policy ([Prasad et al. 2024, arXiv:2405.07503](https://arxiv.org/abs/2405.07503)) 同源, 但简化了。

Fig. 7 的对比: RDT2-UltraFast 推理频率最高, 比 π0.5 (3B) 还快, 比 π0-FAST 也快, 尽管参数量 (7.4B) 是它们的两倍多。

---

## 4. Zero-shot 4U 实验 (Fig. 3)

设计了 "4U" protocol: **U**nseen embodiment, **U**nseen scene, **U**nseen object, **U**nseen instruction。

部署到两个 robot: Franka Research 3 (FR3) 和 UR5e, 两个都是从没在训练集中出现过的 embodiment。

5 个 task: Pick, Pick & Place, Wiping, Shaking, Button Pressing。

**Statistical reliability**: 在 Pick Task 上做 1000 trials, 看 success rate 收敛曲线 (Fig. 4)。然后所有后续实验用 256 trials 来平衡 reliability 和 labor cost。这种严谨在 robotics paper 里非常罕见, 一般 10-20 trials 就上报了。

**结果**: RDT2-VQ 和 RDT2-FM 在 4U setting 下能完成基础 open-vocabulary 任务, success rate 不高但 statistically significant。这本身就是一个 milestone: 纯 human UMI 数据训练的模型能 zero-shot transfer 到 robot。

---

## 5. Scaling law (Fig. 5)

用 Hoffmann et al. 2022 (Chinchilla, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) 和 Kaplan et al. 2020 ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361)) 的形式:

$$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

变量:
- $N$ = model parameters (含 vision encoder)
- $D$ = tokens consumed (each token only consumed once, 即 1 epoch)
- $E$ = irreducible loss (entropy of true distribution)
- $A, B$ = constants, $\alpha, \beta$ = scaling exponents

Fitting: $E \approx 2.1108$, $A \approx 4.3754 \times 10^3$, $\alpha \approx 0.4402$, $B \approx 1.7906 \times 10^2$, $\beta \approx 0.2251$

**Intuition**: $\alpha > \beta$ 意味着 model size scaling 比 data scaling 更 effective (per unit), 但两者都要 scale 才能持续降 loss。这和 Chinchilla 的结论一致。

paper 在 4 个不同 model size 上跑了, 得到 Fig. 5 左右两个 plot, 证实 model size 和 data scale 同时 scale 才有 predictable gain。这给 robotics foundation model 的未来投入提供了 quantitative guidance: **data collection from wearable devices (UMI) 是高度 scalable 的, 应该 all-in**。

---

## 6. Fine-tuning 实验 (Tab. 2)

baseline 是 π0.5 ([Intelligence et al. 2025, arXiv:2504.16054](https://arxiv.org/abs/2504.16054)) 和 π0-FAST ([Pertsch et al. 2025](https://arxiv.org/abs/2501.09747))。两个都用 OpenPI 官方代码训练到收敛。

**Cloth Folding** (deformable, multi-step):
- RDT2: 77% vs π0.5: 36% vs π0-FAST: 29%
- Unseen object: 51% vs 15% vs 10% (RDT2 是 baseline 的 4 倍)

**Table Bussing** (long-horizon):
- Progress score: 0.58 vs 0.39 vs 0.30
- Unseen scene: 0.33 vs 0.17 vs 0.11

**Unzipping** (fine bimanual):
- 45% vs 13% vs 8%

**Button Pressing** (dynamic, reaction time):
- 反应时间差 (vs human 2661ms): +97ms vs +323ms vs +981ms

**Table Tennis** (highly dynamic, hit rate at 1x/1.2x/1.5x/1.7x/2x speed):
- RDT2: 88/85/76/69/68
- π0.5: 78/74/58/57/56
- π0-FAST: N/A (太慢, 根本打不到球)

这组数据非常 informative:
- deformable object 上 RDT2 大幅领先, 因为 UMI 数据 + flow matching 都对 multimodal action distribution 友好
- table tennis 上 π0-FAST 直接 fail, 说明 autoregressive 离散化对 high-dynamic 任务有根本性 bottleneck; RDT2-UltraFast 的 single-step distillation 是关键
- reaction time 差距 (97ms vs 323ms vs 981ms) 直接体现推理速度差异

---

## 7. Ablation (Fig. 6, 7, 8)

**AR + Diffusion vs pure Diffusion** (Fig. 6): pure diffusion 训练 loss 下降慢, AR pre-training + diffusion fine-tuning 收敛快 3-5 倍, 最终 loss 更低。这验证了 Stage 1 的核心 motivation: 保护 VLM 的离散知识 + 提供好初始化。

**Discretization 对比** (Fig. 8): 在相同 quantization error 下, RVQ 比 FAST 节省约 2/3 tokens, uniform binning error 最低但 token 数爆炸。RVQ 的 compact latent space 是关键。

**Inference frequency** (Fig. 7): RDT2-VQ (autoregressive) 因为 RVQ 紧凑所以 token 少, 比 π0-FAST 快; RDT2-UltraFast (single-step distillation) 是全场冠军, 比 π0.5 还快尽管 model 大一倍。

---

## 8. 我的相关联想

1. **UMI 思路和 Mobile ALOHA ([Fu et al. 2024](https://arxiv.org/abs/2401.02117))、ALOHA 2 ([Aldaco et al. 2024](https://arxiv.org/abs/2405.02292))、DexUMI ([Xu et al. 2025](https://arxiv.org/abs/2505.21864)) 同源**: 都在解决 teleoperation 数据采集 cost 问题。但 UMI 是手持、embodiment-agnostic 的, 是最 scalable 的。

2. **三阶段 pipeline 让我想到 LLM 的 "pre-train → SFT → RLHF/DPO"**: Stage 1 是 "学语义", Stage 2 是 "学精细 motor", Stage 3 是 "speed up for deployment"。

3. **RVQ 来自 audio compression (SoundStream, EnCodec)**: robotics 里 action 是连续低维信号, 类似 audio, 用 audio 领域成熟的 RVQ tokenizer 非常合理。FAST tokenizer 用 DCT, 在 sharp action 变化上不如 RVQ 表达力强。

4. **Flow matching vs DDPM**: flow matching 是 linear interpolation, DDPM 是 stochastic Markov chain, flow matching 训练更稳、推理步数更少。Lipman et al. 2022 是理论奠基, RDT2 和 π0 都用 flow matching 是 robotics VLA 的趋势。

5. **Consistency distillation 在 robotics 里刚起步**: Consistency Policy (Prasad et al. 2024)、One-Step Diffusion Policy (Wang et al. 2024b, [arXiv:2410.21257](https://arxiv.org/abs/2410.21257)), RDT2 的 on-the-fly teacher 是个聪明的简化。

6. **Cross-embodiment 的真正 holy grail**: Open X-Embodiment ([O'Neill et al. 2024](https://arxiv.org/abs/2310.08864))、Octo ([Team et al. 2024](https://arxiv.org/abs/2405.12213))、RDT-1B 都试图用统一 embedding space, 但 still 需要 fine-tune。UMI 的 physical interface unification (同样 gripper + 同样 camera) 是另一种思路, 更工程化, 但 zero-shot 效果可能更稳。

7. **Privacy concern**: 100+ households 数据采集, 必然涉及隐私问题。paper 的 Impact Statement 提到了 anonymization, 但没说具体怎么做。这是个隐患。

8. **Scaling exponent $\alpha \approx 0.44$**: 这比 LLM 的 $\alpha \approx 0.076$ (Kaplan) / $\approx 0.34$ (Chinchilla) 大很多, 说明 robotics data 稀缺, model scale 更值钱。但 $\beta \approx 0.225$ 也比 LLM 大, 说明 data scale 也回报显著。这给 robotics 投入的定量依据。

9. **Table tennis 这个 benchmark 很厉害**: 它直接把 inference latency 暴露出来, 是少数能 benchmark 控制频率的 real-world task。比 static pick-place 严格得多。

10. **没提到 safety**: zero-shot cross-embodiment deployment 在 unseen 物理环境中是 dangerous 的, paper 只在 Impact Statement 顺带说 "需要 safety guardrails", 没具体讲。这是 future work。

---

## 9. 一些 critical 思考

- **Zero-shot success rate 不高**: Fig. 3 显示在 Pick & Place 和 Wiping 上 success rate 应该在 20-40% 量级 (具体数字图里看)。这意味着 RDT2 离 production-ready 还有距离, 但作为 research milestone 已经很强。
- **数据 collection cost**: 100 个 UMI + 10,000 hours + 100 households, 这是巨大的资本投入。复现这个数据集很难, open-source 出来后会非常有价值。
- **VLM 完全 frozen 在 Stage 2/3**: 这是 trade-off。frozen VLM 保证 knowledge 不被损坏, 但也限制了 adaptation。如果 cross-embodiment gap 大, frozen VLM 可能不够。
- **10K hours 是 effective unique data 还是包含 augmentation?** paper 没明确讲, 但 figure 5 的 scaling law plot 用的是 "tokens consumed", 每个 token 只用一次, 所以应该是 unique tokens。

---

## Reference links

- RDT2 paper (这篇文章): https://arxiv.org/abs/2507.03078 (推测)
- UMI: https://arxiv.org/abs/2402.10329
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RDT-1B: https://arxiv.org/abs/2410.07864
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- π0-FAST: https://arxiv.org/abs/2501.09747
- OpenVLA: https://arxiv.org/abs/2406.09246
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Flow Matching: https://arxiv.org/abs/2210.02747
- VQ-VAE: https://arxiv.org/abs/1711.00937
- SoundStream (RVQ): https://arxiv.org/abs/2107.03312
- ViT-VQGAN: https://arxiv.org/abs/2110.04627
- VQ-VAE-2: https://arxiv.org/abs/1906.00446
- Chinchilla scaling: https://arxiv.org/abs/2203.15556
- Kaplan scaling: https://arxiv.org/abs/2001.08361
- Consistency Models: https://arxiv.org/abs/2303.01469
- Progressive Distillation: https://arxiv.org/abs/2202.00512
- Consistency Policy: https://arxiv.org/abs/2405.07503
- One-Step Diffusion Policy: https://arxiv.org/abs/2410.21257
- GQA: https://arxiv.org/abs/2305.13245
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Octo: https://arxiv.org/abs/2405.12213
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- ALOHA 2: https://arxiv.org/abs/2405.02292
- DexUMI: https://arxiv.org/abs/2505.21864
- DROID: https://arxiv.org/abs/2403.12945
- FastUMI: https://arxiv.org/abs/2409.19499
- FastUMI-100K: https://arxiv.org/abs/2510.08022
- ACT: https://arxiv.org/abs/2304.13705
- RT-2: https://arxiv.org/abs/2307.15818
- Ego4D: https://arxiv.org/abs/2110.07058
- RoboVQA: https://arxiv.org/abs/2311.00899
- RoboBrain: https://arxiv.org/abs/2502.21257

---

## 总结一句话

RDT2 的核心 contribution 是把 **UMI 数据 scale + hybrid AR/Diffusion 三阶段训练 + single-step distillation** 这三件事拼在一起, 实现了 robotics 第一次真正意义上的 zero-shot cross-embodiment + open-vocabulary generalization。同时它给出的 scaling law 指引了未来 robotics foundation model 应该往哪个方向投入: 大规模 wearable data collection 是 critical path。

如果你 (Andrej) 想做相关方向, UMI 的 hardware 改造 + scaling law 曲线 + distillation trick 是最值得 build on 的三块。我会比较好奇下一步能否做到 100K hours, 以及 frozen VLM 在更 extreme cross-embodiment (比如人形机器人) 下是否还够用。
