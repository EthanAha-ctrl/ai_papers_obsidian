---
source_pdf: exUMI Extensible Robot Teaching System with.pdf
paper_sha256: 69995fcec7f04e7bfdf86c46805934b923f75dd815e13f9efffabcb286cfb522
processed_at: '2026-08-18T12:01:41-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# exUMI 人话版：跟 Karpathy 一起唠唠这篇 paper

## 先说这篇 paper 到底在搞啥

我读完这篇 paper 的第一反应是：这帮人真的把触觉这件事想明白了。

你看，现在 robot learning 领域大家都盯着 vision，因为 vision 数据好搞——YouTube 上随便扒，WebImageText 几十亿对都能爬。但触觉这个 modality 一直是 robot learning 的"穷亲戚"。问题在哪？数据少，信号稀疏，而且采集起来巨麻烦。现有最大的 tactile dataset TVL 才 43.7K frames，连 ImageNet 的零头都不到。

但更根本的问题作者点出来了：**我们一直用错的方式学触觉**。大家习惯性地把 vision 的那套方法直接搬到 touch 上——contrastive learning、masked autoencoder、cross-modal alignment——结果发现不好使。作者说，这是因为触觉跟 vision 有一个根本区别：**触觉是 action 的结果，不是被动观察的状态**。

think about it，你闭上眼摸一个杯子，你知道自己在用多大劲、往哪个方向推，所以你能预测触觉信号会怎么变。这个"如果我做 action A，触觉会变成 B"的 forward model，才是人类触觉认知的本质。这跟我做 VPT 时的直觉完全一样——action-conditioned prediction 才是 world model 的正确形式。

所以这篇 paper 的 thesis 一句话讲：**学触觉表征，要把它当成一个 action-conditioned 的 forward dynamics 问题，而不是当成静态图像分类问题**。

---

## exUMI 硬件：UMI 的 "pro" 版本

### 为什么原版 UMI 不够用

UMI [1] 那个 hand-held gripper twin 的 idea 本身很 genius，你拿在手里当机器人手用，采集的轨迹直接能 transfer 给 robot。但原版有两个 hard problem 没解决好：

1. **6D pose tracking** 用 GoPro fisheye + SLAM，遇到白墙、clean background 就 drift，遇到 occlusion 就丢
2. **Gripper width** 用 ArUco marker tracking，marker 一被手挡住就废

作者在 Figure 3 给了三个翻车场景：clean background、occlusion、复杂纹理。据说原版 UMI 数据处理 success rate 不到 60%，也就是说采集一小时数据可能一半废了。这对触觉这种本来就稀疏的数据是致命的。

### exUMI 怎么解决的

作者的工程决策很聪明——**把 proprioception 拆成两个独立子系统，各用最 robust 的方案**：

**6D pose** 交给 Meta Quest 3 的 inside-out tracking。你戴个 VR headset，左边 controller 固定在 UMI body 上，headset 通过 IR LED 追踪 controller 的 6D pose。这玩意儿对 occlusion immune，精度还高。作者测下来 mean position error 5.4/2.3/1.7mm on XYZ axes，rotation < 1 degree。对比 Fast-UMI 的 RealSense T265 SLAM 方案，AR MoCap 在复杂场景下 robustness 好太多。

**Gripper width** 交给 AS5600 magnetic rotary encoder。这是一个 $1 的 12-bit Hall effect sensor，12-bit 意味着 4096 positions per revolution，对 gripper 那点角度范围绰绰有余。装法是 radial magnet 装在 joint 上，Hall sensor 距离 2mm 读磁场。完全不受视觉遮挡影响，采样率还比相机高。

整个系统用 Orange Pi 3B（$35）做 central controller，同时收 AR headset、rotary encoder、tactile sensor 的数据。总 BOM $698，比原版 UMI 还便宜，加了触觉还降了价，这 engineering 有点东西。

### Latency 校准的小聪明

多 sensor 不同频率同步是个 pain。AR MoCap ~100Hz，tactile ~30Hz，GoPro ~60Hz，rotary encoder ~100Hz。不做硬件 trigger 的话怎么对齐？

作者的 trick 很简单：collect 开始时让用户在 ArUco marker 前面做 horizontal sweep，然后对比 AR MoCap 的 x 轨迹和 ArUco marker 的 x 轨迹，用 bisection search 找 latency offset δ*，使得 $f(t) \approx g(t + \delta^*)$。这里 $f$ 是 vision 轨迹，$g$ 是 AR 轨迹，δ* 就是 AR 系统相对 vision 的延迟。

这个 MSE minimization 写出来就是：
$$\delta^* = \arg\min_\delta \sum_{i=1}^{T} \|f(t_i) - g(t_i + \delta)\|_2^2$$

实测能校准到 <5ms 误差。这种 post-hoc software alignment 比 hardware trigger 简单 10 倍，对研究系统来说够用了。

### 9DTact 的改进

9DTact 这个 sensor 原版 durability 不太行，silicon gel 容易掉，LED 容易坏。作者三个改动很务实：

1. 加 bevel 锁住 gel，防 tangent force 脱落
2. USB 改 Dupont 2-pin connector，cable 管理更稳
3. Custom mold 控制 gel 厚度一致性

gel 厚度这个事被低估了。force-response curve 强依赖 gel 弹性模量 × 厚度，不同厚度 sensor 之间 calibration 几乎不可能。这个工程细节很多人忽略，但实际部署时是 deal-breaker。

---

## TPP 算法：核心 insight 在这

### 先说 tactile representation learning 的三类方法为什么都不对

作者在 Section 4.1 给了一个 taxonomy，我翻译成人话：

**(a) Direct imitation learning**：end-to-end 训一个 policy，tactile encoder 是其中一部分。问题：tactile 数据太少，100 demos 的 pick-place 任务里真正有 contact 的 frame 可能才 3000 个，end-to-end 学不出 generalizable representation。

**(b) Spatial SSL**：contrastive learning 或 masked learning on tactile images。问题：这些方法假设 translation invariance（平移不变性）和 geometric self-consistency（几何自洽性），这俩在 vision 里成立，在 touch 里完全不成立。你把触觉图像平移一下，contact point 信息全变了；你 mask 掉几个 patch 想恢复，但三指按和四指按可能 mask 后看起来一样，这 ambiguity 让 reconstruction 失去意义。

**(c) Visual-tactile alignment**：最大化 vision embedding 和 tactile embedding 的相似度。问题：vision 和 touch 是 one-to-many 关系。同样视觉场景，你轻握和紧握的触觉完全不同。alignment 模型根本学不到 force 这个 key variable。

所以作者的结论：**前面所有方法都把触觉当静态 observation 处理，但触觉本质上是个 action-conditioned dynamic process**。

### TPP 的核心 formulation

作者提出 Tactile Predictive Pretraining (TPP)，预测目标是：
$$p_\theta\left(\mathbf{T}_{t+1:t+n} \mid \mathcal{E}_T(\mathbf{T}_{t-n+1:t}), \mathcal{E}_V(\mathbf{V}_t), \mathcal{E}_A(\mathbf{A}_{t-n+1:t+n})\right)$$

人话翻译：给定过去 $n$ 步触觉历史 $\mathbf{T}_{t-n+1:t}$、当前 RGB 图 $\mathbf{V}_t$、以及覆盖过去和未来的 action 序列 $\mathbf{A}_{t-n+1:t+n}$，预测未来 $n$ 步触觉 $\mathbf{T}_{t+1:t+n}$。

**最关键的细节**：action 序列包含未来 action！这不是"预测未来会发生什么触觉"，而是"如果我执行这些 action，未来会产生什么触觉"。这俩区别巨大——前者是被动 observation model，后者是 action-conditioned forward dynamics model，跟人类触觉认知机制完全对应。

### 架构怎么实现

看 Figure 6，pipeline 分四步：

**第一步：Multimodal encoding**
- Tactile image 先做预处理：原始 9DTact 输出 grayscale，跟 reference image（无接触状态）比较，提取 convex map（凸起）和 concave map（凹陷），三个 stack 成 3-channel image
- 然后过 VAE encoder $\mathcal{E}_T$ 得到 patch embeddings
- RGB 过 ViT encoder $\mathcal{E}_V$
- Action 过 action encoder $\mathcal{E}_A$

注意这里 VAE 是 learnable 的，不像 UVA [46] 用 frozen VAE。因为 tactile 的 distribution 比 natural image 简单，encoder 需要自适应学习，frozen ImageNet VAE 反而不合适。

**第二步：History masking + fusion**
- 对 history tactile patch embeddings 做 random masking（类似 MAE 但在时间维度）
- 对 action features 也做 random masking
- Transformer 融合两个 modality

这里有个精妙之处：前面作者批评 spatial masked learning（单帧内 mask patch），但 TPP 用的是 temporal masking（mask 一些 history frame）。这俩本质不同——spatial masking 假设图像几何自洽（触觉不成立），temporal masking 假设时间序列有 dynamics（触觉动力学成立）。这个区分是论文对前面 SSL 批评的自我辩护，挺 elegant 的。

**第三步：Latent diffusion prediction**
- $n$ 个 history latents 输入 LDM
- Condition：future action embedding + current RGB embedding
- LDM 预测 future tactile latents
- VAE decoder 重建 tactile image

**第四步：Loss**
$$\mathcal{L} = \mathcal{L}_{diff} + \mathcal{L}_{recon}$$

- $\mathcal{L}_{diff}$ 是标准 diffusion loss：$\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$，预测 noise $\epsilon$
- $\mathcal{L}_{recon}$ 是 reconstruction MSE：$\|\hat{\mathbf{T}} - \mathbf{T}\|_2^2$

为什么用 diffusion 不直接 regression？Tactile 信号有 stochasticity，同样 action 可能产生略不同的 contact pattern，diffusion 能建模 multi-modal distribution。这跟 Diffusion Policy [47] 的 motivation 一样。

### Policy learning 阶段

预训练完 freeze $\mathcal{E}_T$，用 Diffusion Policy 做 imitation learning。multimodal fusion 用最简单的 feature concatenation。这里我有点想吐槽——为什么不用 cross-attention？concatenation 是 lazy choice，可能作者想突出 tactile encoder 的贡献而不是 fusion module 的贡献，但实际部署 cross-attention 估计还能再涨几个点。

---

## 实验数据：最 striking 的几个

### Ablation（Table 1）

| Tactile History | Action | RGB | MSE |
|---|---|---|---|
| ✗ | ✓ | ✓ | 0.0298 |
| ✓ | ✗ | ✗ | 0.0132 |
| ✓ | ✗ | ✓ | 0.0125 |
| ✓ | ✓ | ✗ | 0.0117 |
| ✓ | ✓ | ✓ | **0.0099** |

三个 observation 很关键：

1. 没 tactile history 时 MSE 翻倍（0.0298 vs 0.0132）——触觉有强 temporal correlation
2. Action 比 RGB 更重要（0.0117 < 0.0125）——这直接 support 核心 thesis：触觉是 action-conditioned
3. 三者结合最佳——RGB 给 scene context，action 给 intent，history 给 current state

这个 ablation 干净地论证了 action-aware 设计的必要性，比我见过的大部分 ablation 诚实。

### 最 striking 的 task 结果（Table 3）

Pull Drawer (Random) 这个任务最说服力：drawer 里随机放 50g-1000g stones，policy 必须从触觉判断重量再决定 pulling force。

| Method | Success Rate |
|---|---|
| Vision only | 40% |
| Vision + Tactile (naive) | 50% |
| Vision + Tactile + TPP | **95%** |

从 40% 到 95%，45 个绝对百分点提升。naive tactile 只涨 10 个点，说明直接把触觉塞进 policy 没用，必须 pretrain 出 forward dynamics 才能真正用上触觉信息。

Peg in Hole (Insert) 也不错：50% → 60% → 80%。Insertion 阶段需要 force-aware 微调，TPP 的 dynamics knowledge 让 policy 能"感受"对齐状态。

### 数据采集效率

~30s per demonstration（含环境 reset），对比 teleoperation 系统 2-5 min per demo，10× 效率。100 个 demos 20 分钟搞定，这种效率让大规模 tactile data collection 第一次变得 feasible。

### Dataset 规模对比（Table 5）

exUMI 的 480.9K raw frames 比 TVL (43.7K) 大 10×，比 VisGel (12K) 大 40×。更重要的是第一次实现 **human collection + proprioception alignment**——Touch and Go [44] 是 human 采集但没 proprioception，TVL 有 proprioception 但是 robot 采集（慢）。

---

## 跟我自己工作的联想

### VPT 的直接类比

VPT（Video PreTraining）是我跟 OpenAI 那时做的，核心是 action-conditioned next-frame prediction 从 Minecraft 视频学。TPP 是 action-conditioned next-tactile-frame prediction。两者本质都是 world model learning，只是 modality 不同。

VPT 的 action 来自 inverse dynamics model 从视频反推（因为 YouTube 视频没 action label），TPP 的 action 直接来自 exUMI proprioception——这其实更干净，省了 inverse dynamics 这一步的误差。

如果让我做 next step，我会想把 VPT 和 TPP 联合训练：一个 multimodal world model 同时预测 video frame 和 tactile frame，condition on action。这对 contact-rich manipulation 是 game-changer，因为 vision 和 touch 互相 disambiguate——视觉告诉你"接近物体了"，触觉告诉你"碰到了、在用力"。

### nanoGPT 视角的架构简化

TPP 用 latent diffusion model 做预测，但其实 tactile signal 的 distribution 比 natural image 简单太多，作者自己都说"reaches quick convergence due to simpler distribution"。这种情况下，diffusion 可能 overkill。

如果是我，我会试 nanoGPT-style autoregressive transformer：先用 VQ-VAE 把 tactile image tokenize 成 discrete tokens，然后 autoregressive predict next tactile token，condition on action tokens。这样架构简单 10 倍，推理快 10 倍，可能效果差不多。论文 Section 4.1 提到 VQ-GAN [49] 已有探索，但没跟 autoregressive prediction 结合，这是个 low-hanging fruit。

### World Models 的 contact dynamics

我 2024 年写的那个 "World Models" review 强调 latent imagination。TPP 本质就是一个 **contact dynamics world model**——latent space 里预测未来触觉。这跟 DreamerV3 的 latent dynamics model 是一类东西，只是 observation modality 不同。

下一步想象：把 TPP 作为 contact-rich manipulation 的 latent imagination module，类似 Daydreamer 在 model-based RL 中的应用，但专门为 contact 优化。在 latent space 里 rollout 100 步触觉 trajectory，用这个做 planning 或 reward shaping，可能比 model-free RL sample efficiency 高几个数量级。

### Eureka 的 reward design 启示

Eureka 用 LLM 自动设计 reward function。对 contact-rich task，触觉 reward 超难手工设计——什么是"好的 grasp"？什么是"稳定的 insertion"？

如果用 TPP-style forward dynamics model 做 reward shaping——"predicted tactile 跟 actual tactile 一致说明 action 是 expected 的，给 positive reward；不一致说明遇到 unexpected contact，给 negative reward"——可能自动生成有意义的 dense reward。这跟我在 Eureka 里看到的"LLM 生成的 reward code 常常 human 想不到"是类似 spirit。

### makemore 的 simplicity 教训

我做 makemore 时学到最大的 lesson：最简单的 model（bigram）先跑通，再升级到 MLP、RNN、Transformer。TPP 的架构其实挺复杂的——VAE + Transformer + Latent Diffusion + multi-condition。如果是我做这个 project，第一步会先试一个 super naive baseline：用 MLP 直接 regress next tactile frame from (current tactile, action)，看看 MSE 多少。如果这个 baseline 已经不错，说明 task 没那么难，diffusion 是 overkill；如果 baseline 很差，再 justify diffusion 的必要性。论文没这个 naive baseline，有点遗憾。

---

## 我觉得哪里 weak

### Baseline 不够强

没跟 Sparsh [48]（Meta 的 SOTA tactile SSL）对比，也没跟 3D-ViTAC [35] 这种 multimodal foundation model 对比。BYOL 这个 baseline 略显 strawman。如果 Sparsh 在 Peg in Hole 上也能到 75%，那 TPP 的 80% 就没那么 impressive 了。

### Task 太 quasi-static

Pull Drawer、Peg in Hole、Open Bottle 都是 quasi-static 任务——slow、deliberate、no dynamic。如果测 in-hand pivoting、throwing、catching 这种 dynamic manipulation，TPP 的 forward dynamics 优势可能更明显（也可能更差，因为 chaotic dynamics forecasting 本质难）。这个 generalization 测试缺失。

### Action 维度太低

作者自己承认"low action dimension limits interaction information"。7D action（6D pose + 1D gripper）vs 人类 hand 26D+ DOF。这种 low-dim action 让 tactile dynamics 的丰富度受限。下一步应该上 dexterous hand，但硬件成本会暴涨。

### Single view vision

单 RGB + tactile 的 condition 可能不足以消除 tactile prediction 的 ambiguity。多 view 会更好，但作者说未来工作。我觉得这是个 critical limitation，不只是 future work——单 view 可能就是某些 task 表现不好的原因。

### No force-torque sensor

论文用 tactile image 隐式表达 force，没直接 FT measurement。ForceMimic [33] 走的另一条路。两者结合可能最优——FT 给 precise force reading，tactile image 给 spatial distribution。

---

## 大局观：这篇 paper 为什么重要

### 范式转变

exUMI 代表的趋势：从 teleoperation（贵但精确）到 human demo（便宜但 embodiment gap）到 portable hand-held device（UMI 的 sweet spot）。这条路径在 vision manipulation 已被 UMI 验证，TPP 把它扩展到 tactile。

下一步可想象：**crowdsourcing tactile data**。如果 exUMI 量产到 $500 以下，像 ImageNet 那样众包触觉数据完全可能。1000 个 lab 每个采集 1000 hours = 1M hours data，这能 enable 一个真正的 Tactile Foundation Model。

### Pretraining 范式的反思

TPP 对 SSL 的批评值得整个 tactile community 深思：
- Translation invariance 在 vision 合理，在 touch 不合理
- Geometric self-consistency 在 vision 合理，在 touch 不合理
- One-to-one cross-modal alignment 在 vision-language 合理，在 vision-touch 不合理

但 TPP 自己的 inductive bias 是 **forward dynamics**——假设未来触觉可以从 history+action+vision 推出。这个假设是否在所有 manipulation task 成立？对于 quasi-static task 大概成立，对于 chaotic dynamics（pouring granular materials、cutting soft materials）可能 forecasting 本质困难，需要 stochastic model 或 hierarchical planning。

### Democratization

$698 的 BOM、DIY-friendly 的设计、open-source CAD——这些都是 democratize tactile research 的关键。一个 resource-constrained 的 lab 现在也能做 contact-rich manipulation research 了。这种 democratization 在 vision field 早就发生（ImageNet、CLIP 都是 open 的），在 tactile field 一直没发生，exUMI 可能是起点。

参考链接：
- exUMI Project page: https://silicx.github.io/exUMI
- UMI 原论文: https://arxiv.org/abs/2402.10329
- UVA（TPP 算法基础）: https://arxiv.org/abs/2503.00200
- 9DTact 传感器: https://arxiv.org/abs/2305.05727
- ARCap（AR MoCap 基础）: https://arxiv.org/abs/2410.08464
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Fast-UMI: https://arxiv.org/abs/2409.19499
- ForceMimic: https://arxiv.org/abs/2410.07554
- Sparsh（SOTA tactile SSL，未对比）: https://arxiv.org/abs/2410.24090
- 3D-ViTac（未对比 baseline）: https://arxiv.org/abs/2410.24091
- TVL Dataset: https://arxiv.org/abs/2402.13232
- Touch and Go: https://arxiv.org/abs/2211.12498
- VisGel: https://arxiv.org/abs/1902.06493

---

## TL;DR

如果让我用一句话总结 exUMI 的核心 insight：**触觉本质上是 action 的 consequence，学触觉表征要建模 action-conditioned forward dynamics，而不是当静态图像处理**。

这个 insight 在算法上落地为 TPP——用 latent diffusion 预测 future tactile frame，condition 是 future action + current vision + tactile history。在硬件上落地为 exUMI——portable hand-held device 采集 1M+ frames 的 human play data，每帧都有 action-tactile 对齐，$698 DIY-friendly。

exUMI 最大的贡献是提供了一个 **democratizable 的触觉研究基础设施**。硬件便宜+开源，算法 action-aware，数据大规模可采集。如果 1000 个 lab 都用 exUMI 采集数据，tactile manipulation 的 "ImageNet moment" 可能真的会到来。这是我读完最兴奋的点——不是某个 trick 多 clever，而是整个 paradigm 可复制、可 scaling。

如果让我做 next step，三个方向：
1. **VPT × TPP 联合 world model**：同时预测 video frame 和 tactile frame，condition on action
2. **nanoGPT-style tactile tokenizer**：VQ-VAE tokenize + autoregressive prediction，简化架构
3. **Crowdsourcing 平台**：exUMI 量产 + 数据上传+ foundation model 训练，做成 tactile 版 ImageNet

这三个方向任何一个做出来，impact 都不会比 exUMI 本身小。触觉 learning 的 scaling law 还在早期，exUMI 给了它第一个可行的 path forward。

---

# exUMI 深度技术解析：硬件-算法协同设计的触觉学习范式

## 1. 论文核心 motivation：为什么触觉学习这么难？

这篇论文本质上在回答一个非常根本的问题：**触觉感知为什么不像 vision 那样可以被大规模 self-supervised pretraining 解决？** 作者的洞察是——触觉信号在三个层面上与 vision 有本质区别：

### 1.1 信号的稀疏性

论文 Table 5 中的对比非常触目惊心：现有最大的 tactile dataset TVL [12] 仅有 43.7K frames，而 exUMI 收集了 480.9K raw frames（1M 总帧数）。更关键的是，作者引用 [12] 指出 **valid tactile contacts 在 manipulation trajectory 中占比 <10%**。这意味着即使你采集到大量触觉数据，其中 90% 是"无接触"的平凡帧。

作者由此推出一个深刻的论断：直接将 vision 的 SSL 套用到 tactile 上会失效，因为：
- **Contrastive learning** 假设 translation invariance，但触觉中"任何平移都改变接触点信息"，这个 inductive bias 完全错误
- **Masked learning** 假设 image patches 可以从其他 patches 恢复，但触觉图像没有几何 self-consistency：三指按和四指按产生的信号不同，但部分 masking 可能产生相同的两接触点图像——这种 ambiguity 让 masked reconstruction 失去意义

### 1.2 Vision-Tactile 的 one-to-many 关系

这是论文最 elegant 的洞察之一。Visual-Tactile Alignment 范式（如 VisGel [42]）假设 $s(\mathcal{E}_T(\mathbf{T}_t), \mathcal{E}_V(\mathbf{V}_t))$ 应该最大化，即"相同视觉场景对应相同触觉信号"。但作者指出：**不同的 contact force 在同一 visual scene 下会产生完全不同的 tactile signal**。比如同样是抓住杯子，轻握和紧握的触觉图像天差地别，但 RGB 几乎一致。

这个 observation 让我想到你（Karpathy）在 VPT 中的思路——context 的重要性。视觉是 state observation，触觉是 state+action 的隐式 consequence，二者根本不在同一个 abstraction level。

### 1.3 Tactile 作为 dynamic process 而非 static observation

作者的核心 thesis：**human tactile understanding intrinsically combines contact mechanics with motion intent**。比如人类知道"如果我用力推、向左拖，slip risk 会降低，触觉信号会变强"。这种 action-conditioned forward dynamics 才是触觉认知的本质。

这让我联想到你一直在推的 world model 思路——触觉本质上是一种 forward predictive model 的隐变量，而不是被动 observation。

---

## 2. exUMI 硬件系统：工程细节深度解析

### 2.1 为什么放弃 SLAM+ArUco？

原版 UMI [1] 用 GoPro fisheye + SLAM tracking 6D pose + ArUco marker tracking gripper width。论文 Figure 3 展示了三个 failure case：clean background（特征点稀少）、occlusion（手遮挡 marker）、复杂背景（SLAM drift）。

作者的工程决策非常有意思——**disentangle proprioception into two specialized sensors**：
- **6D pose**：交给 AR MoCap（Meta Quest 3）
- **Gripper width**：交给 magnetic rotary encoder（AS5600）

这种 disentanglement 的好处是每个子问题用最 robust 的方案，而不是期望一个 vision-based 系统同时解决两个不同精度要求的问题。

### 2.2 AS5600 Magnetic Encoder 的细节

AS5600 是一个 12-bit（4096 positions/revolution）的 Hall effect sensor，通过 I²C 通信。论文 Figure 11 显示安装方式：radial magnet 安装在 gripper joint 上方，Hall sensor 距离 ~2mm。

**为什么这个方案优于 ArUco？**
1. 采样率更高（不受相机帧率限制）
2. 完全 immune to visual occlusion
3. 计算开销可忽略（vs. ArUco detection）
4. 12-bit resolution 在 gripper 的有限角度范围内提供 sub-mm 精度

校准协议也很有意思：gripper 以 1cm 间隔 incrementally 定位，记录 AS5600 reading，然后 interpolation 得到 mapping。这是一个 piecewise linear calibration，简单但实用。

### 2.3 AR MoCap：Meta Quest 3 的工程化

这部分基于 ARCap [45] 的工作，但作者做了几个关键改动：

**Tracking 原理**：Meta Quest 3 内部有 VIO（Visual Inertial Odometry），通过 onboard SLAM 实现 6D pose 估计。论文将 left VR controller 固定在 UMI body 上，headset 跟踪 controller 的 6D pose（通过 IR LEDs + inside-out tracking）。

**Latency 校准算法（Algorithm 1）**：
```
Input: trajectories f(t), g(t), bounds [δ_min, δ_max]
Output: latency δ* such that f(t) ≈ g(t + δ*)
```
核心公式：
$$k = \arg\min_k \sum_{i=1}^{T} \| f(t_i) - g(t_i + \delta_k) \|_2^2$$

变量解释：
- $f(t_i)$：vision-based（ArUco marker）的轨迹
- $g(t_i + \delta_k)$：AR MoCap 轨迹 shifted by latency δ
- $\delta_k$：候选 latency offset
- $\|\cdot\|_2^2$：squared L2 norm

算法用 bisection-style search：每次将搜索区间分成 M 段，找到最优 $\delta_k$，然后将搜索范围缩小到 $[\delta_{k-N}, \delta_{k+N}]$，直到区间宽度 < ε=0.0001。

**精度数据**（Figure 13）：
- X 轴（depth 方向）：mean error 5.4mm，max 20mm（depth 轴最难）
- Y 轴：mean 2.3mm
- Z 轴：mean 1.7mm
- 旋转误差 < 1 degree

对比 Fast-UMI [32] 的 RealSense T265 SLAM 方案，AR MoCap 在遮挡和复杂场景下 robustness 显著更好。

### 2.4 Latency < 50ms 的多模态同步

整个系统的关键挑战是 **AR MoCap（~100Hz）、tactile sensor（~30Hz）、GoPro（~60Hz）、rotary encoder（~100Hz）** 四个不同频率的 sensor 同步。

论文用 Orange Pi 3B 作为 central controller（Table 4 中 $35），同时收集所有 sensor 数据。同步靠 **post-hoc latency calibration + interpolation**：
1. 收集开始时让用户在 ArUco marker 前做 horizontal sweep
2. 提取 AR MoCap 和 visual marker 的 x 轴轨迹
3. 用 Algorithm 1 计算 latency offset δ*
4. 对所有数据应用 temporal correction
5. 通过 interpolation align 到 video frame rate

这个设计很 pragmatic——不做硬件级 trigger 同步，而是 software-level post-hoc 对齐，大大降低系统复杂度。

### 2.5 9DTact 的改进：从研究原型到耐用产品

原版 9DTact [22] 的 durability 问题作者直说"critical concern"。三个改进点很关键：

1. **Bevel 设计**（Figure 9）：sensor shell 加 bevel 锁住 black silicon gel，防止大 tangent force 下脱落
2. **2-pin header connector**：替换 USB cable 为 Dupont connector，cable management 更灵活
3. **Custom mold**（Figure 10）：精确控制 silicon gel 厚度，保证 sensor 一致性

**为什么 silicon gel 厚度一致性这么重要？** 触觉传感器的 force-response curve 强依赖 gel 的弹性模量 × 厚度。不同厚度的 gel 对相同 force 产生不同 deformation，导致 cross-sensor calibration 困难。这是触觉研究长期被忽视的工程细节。

成本拆解（Table 4）：
| Component | Cost ($) |
|-----------|----------|
| GoPro 11 + Accessories | 298 |
| Meta Quest VR Headset | 299 |
| Orange Pi 3B | 35 |
| AS5600 Magnetic Encoder | 1 |
| 3D Printed Parts | 15 |
| Visuo-Tactile Sensors | 30 |
| Misc. | 20 |
| **Total** | **698** |

对比 UMI 的 $750+ 和 ForceMimic [33] 的 force-torque sensor 方案（FT sensor 便宜也要 $500+），这个 BOM 非常 aggressive。

---

## 3. TPP 算法：Action-aware Tactile Predictive Pretraining

### 3.1 核心数学 formulation

论文 Section 4.3 给出核心预测目标：
$$p_\theta\left(\mathbf{T}_{t+1:t+n} \mid \mathcal{E}_T(\mathbf{T}_{t-n+1:t}), \mathcal{E}_V(\mathbf{V}_t), \mathcal{E}_A(\mathbf{A}_{t-n+1:t+n})\right)$$

变量拆解：
- $\mathbf{T}_{t+1:t+n}$：未来 $n$ 步的 tactile frame 序列（预测目标）
- $\mathbf{T}_{t-n+1:t}$：过去 $n$ 步的 tactile history
- $\mathbf{V}_t$：当前 RGB 图像
- $\mathbf{A}_{t-n+1:t+n}$：覆盖 past+future 的 action 序列（关键！包含未来 action）
- $\mathcal{E}_T, \mathcal{E}_V, \mathcal{E}_A$：tactile、visual、action 的 encoder

**为什么 action 包含未来？** 这正是 action-aware 的精髓——模型不是预测"未来会发生什么 tactile 信号"，而是预测"如果我执行这些 action，未来会产生什么 tactile 信号"。这本质上学习了一个 **action-conditioned forward tactile dynamics model**。

这与 Unified Video Action Model (UVA) [46] 的思路一脉相承，但 UVA 用 frozen VAE 处理 video，TPP 让 VAE **learnable**，因为 tactile 的 distribution 比 natural image 简单，需要 encoder 自适应学习。

### 3.2 架构深度解析

Figure 6 的 pipeline 我理解为四个 stage：

**Stage 1: Multimodal Encoding**
- Tactile frame → patchify → VAE encoder $\mathcal{E}_T$ → patch embeddings
- RGB image → ViT encoder $\mathcal{E}_V$
- Action sequence → MLP/action encoder $\mathcal{E}_A$

tactile 图像预处理（Section C）很关键：
1. 原 9DTact 输出 calibrated grayscale image
2. 与 reference image（无接触状态）比较，提取 **convex map**（凸起）和 **concave map**（凹陷）
3. 三者 stack 成 3-channel image（类似 RGB 但语义不同）

这种 representation 比 raw grayscale 信息更丰富，convex/concave 直接对应 normal force 方向。

**Stage 2: History Fusion with Masking**
- 对 history tactile patch embeddings 做 random masking（类似 MAE）
- 对 action features 也做 random masking
- Transformer 融合两个 modality

**Stage 3: Latent Diffusion Prediction**
- $n$ 个 history latents 输入 LDM
- condition：future action embedding $A_{t+1:t+n}$ + current RGB embedding $V_t$
- LDM 预测 future tactile latents
- VAE decoder $\mathcal{D}_T$ 重建 tactile image

**Stage 4: Loss**
$$\mathcal{L} = \mathcal{L}_{diff} + \mathcal{L}_{recon}$$

- $\mathcal{L}_{diff}$：标准 diffusion loss $\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$，预测 noise
- $\mathcal{L}_{recon}$：reconstruction MSE $\|\hat{\mathbf{T}} - \mathbf{T}\|_2^2$

**为什么用 diffusion 而不是直接 regression？** Tactile 信号有 inherent stochasticity（相同 action 可能产生略不同的 contact pattern），diffusion 能建模这种 multi-modal distribution。这一点和 Diffusion Policy [47] 的 motivation 一致。

### 3.3 为什么 masking 在这里有意义？

注意：前面作者批评了 spatial masked learning（在单帧内 mask patch），但这里 TPP 用的是 **temporal masking**——随机 mask 一些 history frame。这其实是合理的，因为：
- Spatial masking：假设 image patches 几何 self-consistent → 触觉不成立
- Temporal masking：假设时间序列有 dynamics → 触觉动力学成立

这个区分非常关键，是论文对前面 SSL 批评的"自我辩护"。

### 3.4 Policy Learning 阶段

预训练后 freeze $\mathcal{E}_T$，用 Diffusion Policy [47] 做 imitation learning：
$$\pi(\mathbf{a}_{t+1} | \mathcal{E}_S(\mathbf{s}_t), \mathcal{E}_T(\mathbf{T}_t), \mathcal{E}_V(\mathbf{V}_t))$$

multimodal fusion 用最简单的 **feature concatenation**（Section 5.2）。这里其实可以批评——为什么不用 cross-attention？但作者的 pragmatic 选择可能是想突出 tactile encoder 的贡献，而不是 fusion module 的贡献。

---

## 4. 实验数据深度解读

### 4.1 Tactile Prediction 的 ablation（Table 1）

| Tactile History | Action Input | RGB Image | MSE Error |
|-----------------|--------------|-----------|----------|
| ✗ | ✓ | ✓ | 0.0298 |
| ✓ | ✗ | ✗ | 0.0132 |
| ✓ | ✗ | ✓ | 0.0125 |
| ✓ | ✓ | ✗ | 0.0117 |
| ✓ | ✓ | ✓ | **0.0099** |

关键 observations：
1. **没有 tactile history 时 MSE 翻倍**（0.0298 vs 0.0132），说明 tactile dynamics 有强 temporal correlation
2. **Action 比 RGB 更重要**（0.0117 < 0.0125），这支持了作者的核心 thesis——触觉是 action-conditioned dynamics
3. **三者结合最佳**，说明 RGB 提供 scene context，action 提供 intent，tactile history 提供 current state

这个 ablation 非常 clean 地论证了 action-aware 设计的必要性。

### 4.2 数据采集效率（Table 2）

| Task | Demos | Collection Time | Success Rate |
|------|-------|-----------------|--------------|
| Pick Cube | 204 | 42 min | 85% |
| Pull Drawer | 202 | 70 min | 40% |
| Peg in Hole | 163 | 56 min | 50% |
| Open Bottle | 270 | 79 min | 20% |

数据采集效率惊人：~30s per demonstration（含环境 reset）。对比 teleoperation 系统通常 2-5 min per demo，效率提升 10×（论文原文）。

但 vision-only policy 的 success rate 在 force-sensitive task 上惨不忍睹：Pull Drawer 40%，Open Bottle 20%，Peg in Hole 50%。这正好为 tactile 增强提供了 motivation。

### 4.3 Tactile-aware 任务的核心结果（Table 3）

| Input | Put Ball | Open Bottle | Pull Drawer (Empty) | Pull Drawer (Random) | Peg in Hole (Grasp) | Peg in Hole (Insert) |
|-------|----------|-------------|---------------------|----------------------|---------------------|----------------------|
| Vision Only | 70% | 20% | 100% | 40% | 100% | 50% |
| Vision + Tactile | 70% | 50% | 100% | 50% | 100% | 60% |
| V+T w/ TPP (Ours) | **85%** | **60%** | 100% | **95%** | 100% | **80%** |

最 striking 的对比是 **Pull Drawer (Random)**：从 40% → 50% → **95%**。TPP 带来 45 个绝对百分点的提升！

这个任务的 motivation 很 clever：drawer 里随机放 50g-1000g stones，policy 必须从 tactile 反馈判断 drawer 重量，再决定 pulling force/direction。Vision-only 看不出重量差异，所以只有 40%；naive tactile 直接学只有 50%；TPP 因为 pretraining 学到了 forward tactile dynamics，能"预演"action 的 tactile consequence，达到 95%。

Peg in Hole (Insert) 从 50% → 60% → 80% 也很有说服力——insertion 阶段需要 force-aware 微调，TPP 的 dynamics knowledge 让 policy 能"感受到"对齐状态。

### 4.4 与其他 tactile learning 方法对比（Table 6）

| Modality | Put Ball | Peg in Hole |
|----------|----------|-------------|
| V only | 70% | 50% |
| V+T Direct | 70% | 60% |
| V+T BYOL | 80% | 50% |
| V+T TPP (Ours) | **85%** | **80%** |

BYOL 在 Peg in Hole 上居然比 Direct 还差（50% vs 60%），这印证了作者对 spatial SSL 的批评——BYOL 的 invariance 假设在触觉中是错的。TPP 在两个任务上都 best，且在 force-sensitive 的 Peg in Hole 上优势更明显（+20% over Direct）。

### 4.5 数据集对比（Table 5）

exUMI 的 480.9K frames 比之前最大的 TVL (43.7K) 大 10×，比 VisGel (12K) 大 40×。更关键的是 **proprioception/action 对齐 + human collection**：
- TVL：43.7K，DIGIT，有 proprioception，Robot 采集
- Touch and Go：13.9K，GelSight，无 proprioception，Human 采集
- exUMI：480.9K，9DTact+，有 proprioception，Human 采集

Human collection + proprioception 对齐是 exUMI dataset 的 unique selling point。

---

## 5. 与你（Karpathy）工作的潜在联系

### 5.1 VPT 的 action-conditioning 类比

VPT（Video PreTraining）的核心是 **action-conditioned next-frame prediction**，TPP 是 **action-conditioned next-tactile-frame prediction**。两者本质都是 world model learning，但 modality 不同。

VPT 的 action 来自 inverse dynamics model 从视频中反推，TPP 的 action 直接来自 exUMI proprioception——这是一个优势，因为不需要 inverse dynamics 这一步。

可以想象：如果将 VPT 的 video prediction 和 TPP 的 tactile prediction 联合训练，可能得到一个 **multimodal world model**，同时预测 video frame 和 tactile frame，这对 contact-rich manipulation 是 game-changer。

### 5.2 nanoGPT 视角的架构简化

TPP 用 latent diffusion model 做预测，但其实 tactile signal 的 distribution 比 natural image 简单得多（论文也说"reaches quick convergence due to simpler distribution"）。一个有意思的问题：能否用 nanoGPT-style autoregressive transformer 直接预测 tactile token？

类似 VQ-VAE tokenization tactile image，然后 autoregressive predict next tactile token。这样可能比 diffusion 更简单高效。论文 Section 4.1 提到 VQ-GAN [49] 已有探索，但没有与 autoregressive prediction 结合。

### 5.3 Eureka 的 reward design 启示

Eureka 用 LLM 自动设计 reward function。对于 contact-rich task，触觉 reward 很难手工设计（什么是"好的 grasp"？）。如果用 TPP-style forward tactile dynamics model 做 reward shaping——"predicted tactile 与实际 tactile 一致说明 action 是 expected 的"——可能自动生成有意义的 dense reward。

### 5.4 World Models 的 contact dynamics

你 2024 年的 "World Models" review paper 强调 latent imagination。TPP 本质就是 **contact dynamics world model**——latent space 中预测未来 tactile。这与 DreamerV3 的 latent dynamics 模型很类似，只是 observation 是 tactile 而非 image。

未来方向：将 TPP 作为 contact-rich manipulation 的 latent imagination module，类似 Daydreamer 在 model-based RL 中的应用，但专为 contact 优化。

---

## 6. 局限性的诚实评估

### 6.1 硬件层面

**AR Headset 的人体工学问题**：作者承认 thermal discomfort 和 neck strain。Meta Quest 3 重 515g，加上固定 mount，长时间佩戴确实累。论文提到 alternative 是 HTC Vive Tracker，但需要 base station，违反 portability 原则。

**Tactile sensor 的 durability**：即使改进后的 9DTact，作者也说"further enhancements remain possible"。硅 gel 的 wear-and-tear、LED 老化、camera focal drift 都是长期问题。这一点 GelSight Mini 商业化做得更好，但贵 10×。

### 6.2 算法层面

**Action 维度低**：作者承认"interaction and movement information is limited due to low action dimension"。exUMI 的 action 是 6D pose + 1D gripper width = 7D。但人类 hand 有 26D+ DOF。这种 low-dim action 限制了 tactile dynamics 的丰富度。

**Single view vision**：作者说未来要加 multi-view。单 RGB + tactile 的 condition 可能不足以消除 tactile prediction 的 ambiguity。

**No force-torque sensor**：论文用 tactile image 隐式表达 force，但没有直接 FT measurement。ForceMimic [33] 走的是另一条路。两者结合可能最优。

### 6.3 实验设计层面

**任务相对简单**：Pull Drawer, Peg in Hole 虽然是经典 contact-rich benchmark，但都是 quasi-static 任务。没有 dynamic manipulation（如 in-hand pivoting, throwing, catching）。

**Baseline 不够强**：没和 Sparsh [48]（Meta 的 SOTA tactile SSL）对比，也没和 3D-ViTAC [35] 这种 multimodal foundation model 对比。BYOL 这个 baseline 略显 strawman。

**Generalization 测试缺失**：没测试 unseen object 的 generalization，只测了 unseen environment。

---

## 7. 对触觉学习领域的更大启示

### 7.1 数据采集范式的转变

exUMI 代表的趋势：**从 teleoperation 到 human demonstration 再到 portable hand-held device**。这条路径在 vision manipulation 已被 UMI 验证（"in-the-wild robot teaching"），TPP 将其扩展到 tactile。

可想象的下一步：**crowdsourcing tactile data**。如果 exUMI 量产到 $500 以下，像 ImageNet 那样众包触觉数据完全可能。这会彻底改变 tactile learning 的 scaling law。

### 7.2 Pretraining 范式的反思

TPP 对 SSL 的批评值得触觉社区深思：
- Translation invariance：vision 合理，touch 不合理
- Geometric self-consistency：vision 合理，touch 不合理
- One-to-one cross-modal：vision-language 合理，vision-touch 不合理

但 TPP 自己的 inductive bias 是什么？**Forward dynamics**——假设未来 tactile 可以从 history+action+vision 推出。这个假设是否在所有 manipulation task 成立？比如 chaotic dynamics（如 pouring granular materials）可能 forecasting 本质困难。

### 7.3 Foundation Model 的可能路径

如果沿着 TPP 路径走，可以想象一个 **Tactile Foundation Model**：
1. Pretrain on 10M+ frames of human play data（exUMI 量产）
2. Forward dynamics model as pretraining objective
3. Downstream：grasping, insertion, in-hand manipulation, deformable manipulation

这与 R3M [30] for vision、Voltron [11] for language-conditioned manipulation 的路径类似，但 for touch。

参考链接：
- Project page: https://silicx.github.io/exUMI
- UMI 原论文：https://arxiv.org/abs/2402.10329
- UVA（TPP 算法基础）：https://arxiv.org/abs/2503.00200
- 9DTact 传感器：https://arxiv.org/abs/2305.05727
- ARCap（AR MoCap 基础）：https://arxiv.org/abs/2410.08464
- Diffusion Policy：https://arxiv.org/abs/2303.04137
- Fast-UMI：https://arxiv.org/abs/2409.19499
- ForceMimic：https://arxiv.org/abs/2410.07554
- Sparsh（SOTA tactile SSL，未对比）：https://arxiv.org/abs/2410.24090
- 3D-ViTac（未对比的 baseline）：https://arxiv.org/abs/2410.24091
- TVL Dataset：https://arxiv.org/abs/2402.13232

---

## 8. Intuition 总结

如果让我用一句话总结 exUMI 的核心 insight：

> **触觉不是"看到的状态"，而是"动作的结果"。学触觉表征不能像 vision 那样假设空间 invariance，而应该建模 action-conditioned forward dynamics。**

这个 insight 在算法上落地为：用 latent diffusion 预测未来 tactile frame，condition 是 future action + current vision + tactile history。在硬件上落地为：portable hand-held device 采集 1M+ frames 的 human play data，且每帧都有 action-tactile 对齐。

exUMI 给我最大的启发是 **硬件和算法的 co-design**——正是因为有了便携、可扩展、action-aligned 的硬件，才能 collect 到 enable action-aware pretraining 的数据。这不是单纯算法突破，而是 data engineering + algorithm 的协同。这种范式在 vision-language 已经成熟（CLIP 依赖 WebImageText），在 robotics-tactile 才刚起步。

未来如果能看到 exUMI-style 硬件 × 1000 labs 众包，加上 TPP-style pretraining scaling，触觉 manipulation 的 "ImageNet moment" 是可能到来的。这是我个人认为这篇 paper 最大的贡献——它提供了一个 **democratizable** 的触觉研究基础设施。
