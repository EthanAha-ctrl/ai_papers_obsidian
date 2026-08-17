---
source_pdf: Habilis-β A Fast-Motion and Long-Lasting On-Device VisionLanguage-Action
  Model.pdf
paper_sha256: 6e2f1f2e64bc587eea71631620bdf77617c08ca71ab30b7d4af81fbc14f584c4
processed_at: '2026-08-04T23:20:09-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Habilis-β 人话版：这篇 paper 到底在干嘛

## 一句话总结

这篇 paper 说：**现在 VLA 圈子都在比"单次成功率"，但工厂老板根本不 care 你一次能不能做成，他们只 care 两件事——一小时能干多少活（TPH），以及平均多久需要人来救一次（MTBI）**。Habilis-β 就是围绕这两个指标重新设计了一套 VLA 系统。

## 为什么这件事值得 care

### 当前 VLA 圈子的 evaluation 有多离谱

你想象一下 GPT 评估只看"在精心准备的 prompt 下能不能答对一次"——但 ChatGPT 真正的成功是因为用户每天用 8 小时都不会想卸载。VLA 现在的 evaluation 就差不多是这个状态：

- 给一个 carefully curated initial state（机器人位置、物体位置都调好了）
- 跑一次 episode
- 看成功不成功
- 报 success rate

**问题在于**：真实部署里机器人是连续干活的，第一个 cycle 成功后物体位置就变了，第二个 cycle 的 initial state 已经 drift 了，第三个 cycle 可能上一个 cycle 的小误差让 gripper 角度偏了 2 度，第十个 cycle 整个状态分布跟你训练时见过的完全不一样。所以很多 paper 报 90% success rate，实际部署 10 分钟就 stuck 一次，根本没法用。

这个现象在 ML 里太常见了——benchmark overfitting。ImageNet top-1 从 70% 涨到 90% 那几年，大家觉得 vision 解决了，后来做 detection/segmentation/real-world video 才发现还差得远。VLA 现在就处在那个"success rate 涨得很爽但离部署还差十万八千里"的阶段。

### PRP 这个 evaluation plane 为什么是个 abstraction 突破

Habilis-β 提的 Productivity-Reliability Plane 本质上就是把 deployment readiness 投影到 2D：

- x 轴：TPH（Tasks per Hour）—— 生产力，能不能快
- y 轴：MTBI（Mean Time Between Intervention）—— 可靠性，能不能稳

然后每个 VLA system 是这个 plane 上的一个点。这个 abstraction 的好处是：

1. **把 tradeoff 显式化**：你可以"很快但经常 fail"，也可以"很慢但很稳"，这两个点在 PRP 上位置完全不同，但用 success rate 看不出来
2. **让 design choice 有 target**：你想往右上角推，每个 component 该贡献什么——play data 提升 unconditional prior 增强 recovery，cyclic data 对齐 deployment distribution，ESPADA 加速 transit，rectified flow distillation 省 latency 做 high-freq control，CFG 调指令服从强度——每个 trick 都对应一个移动方向

这非常像 Karpathy 自己当年写 "Software 2.0" 时讲的——把模糊的工程问题收敛到清晰的优化目标上。PRP 就是 VLA 的 "Software 2.0 objective"。

## 系统在干嘛：三个 stage 的 intuition

### Stage 1: Play Data Pre-training

**直觉**：你让一个人随便玩机器人，不告诉他任务是什么，就是瞎玩——抓东西、放下、regrasp、转手腕。这些 data 看起来 useless，但其实包含了大量 task demonstration 里没有的 recovery 行为。

为什么这个重要？假设你 train 一个 policy 只看"成功 pick 物体放进 bin"的 demonstration，那么部署时如果 gripper 没抓稳物体滑了，policy 从来没见过这种"物体在 gripper 里滑动"的状态，就 panic 了。play data 里这种状态海量存在，所以 pre-train 一个 unconditional motion prior 能让 policy 在 task-specific fine-tune 之前就有一套 robust 的低层物理交互直觉。

这跟 LLM pre-train 用 internet text、fine-tune 用 instruction data 的分层思路一模一样——play data 是"通用物理直觉 corpus"，task data 是"下游 instruction corpus"。

参考 Lynch et al. 的 Play data 工作：https://arxiv.org/abs/1912.07241

### Stage 2: Cyclic Task Post-training

**直觉**：normal VLA 数据集是"从干净初始状态开始 → 任务成功 → 停"。但真实部署是"任务成功 → 立刻进入下一个 cycle → 状态已经 drift → 再成功 → 再 drift"。

cyclic data 就是采集时**不 reset**，让一个 success 直接接下一个 success 的开始，把 cycle 之间的 transition 也录下来。这样模型见过"上一个 cycle 结束时物体歪了 → 下一个 cycle 该怎么处理"，部署时就不会在 cycle 2 就崩。

这个 idea 听起来 trivial，但实际上是解决 distribution shift 的一个很巧妙的方法。本质上是用 data design 把 deployment distribution 显式拉进 training distribution 里，对应到 distribution shift 文献里的 "coverage" 问题：https://arxiv.org/abs/2203.03669

### Stage 3: Rectified Flow Distillation

**直觉**：你的 action expert 是个 flow matching 模型，需要 ≥10 步 ODE 求解才能生成一个 action chunk。在 Jetson Orin 上跑 10 步太慢，闭环控制频率上不去。

rectified flow distillation 做的事是训一个 student，让它从 teacher ODE trajectory 上的任意中间点都能**两步跳到终点**。具体 loss 就是让 student 学会直接预测 teacher 的终点 $x_1$，而不是去模仿 teacher 的逐步 trajectory。

distill 完之后 inference 只需 2 步 function evaluation，latency 大幅下降。

但这里**真正的 insight 在下一步**——省下的 latency 不是用来 idle 的，而是用来 **re-infer 更频繁**。原来 10 步花 50ms，生成一个 16 步的 action chunk 执行 320ms；现在 2 步花 10ms，生成一个 4 步的 action chunk 执行 80ms，然后立刻 re-infer。chunk 越短，闭环反馈越紧，drift 累积被持续抑制，MTBI 自然就上去了。

这个"省 latency → re-invest 到 high-freq control"的思路是整篇 paper 最 Karpathy-style 的 insight。LLM 里 speculative decoding 省 latency 也是类似——省下来的 budget 不是给用户省 GPU 钱的，是用来让交互更流畅的。

参考 Rectified Flow：https://arxiv.org/abs/2209.03003
参考 InstaFlow（distillation 版本）：https://arxiv.org/abs/2309.06370

## ESPADA：让机器人"该快的时候快"

**问题**：teleoperation 数据天然是 conservative 的——人类遥操作时即使自由空间移动也会慢慢挪，因为人脑怕撞东西。policy 学到这种 behavior 后部署也是慢慢挪，TPH 上不去。

**naive 解决方案**：uniform downsample——把整个 trajectory 的 action step 都砍掉一半，让机器人跑 2 倍速。问题：接触阶段的精细控制被 alias 掉，pick 抓不稳，place 放不准。

**ESPADA 的做法**：用 VLM-LLM pipeline 对 trajectory 做语义分割，区分"自由空间移动"和"接触精细操作"。然后对自由空间段做 aggressive downsample（replicate-before-downsample 避免 aliasing），对接触段保留高频。

公式 9 那个 geometric consistency constraint 本质是说：你加速可以，但**机器人物理上走过的路径必须跟原来一致**。这样 policy 学到的 spatial action distribution 是一致的，只是 temporal density 自适应——自由空间少几步快速走完，接触阶段多步精细控制。

intuition 上就像人开车：高速上踩油门，进停车场就慢下来。policy 之前的问题是"全停车场模式"，ESPADA 让它学会"高速公路模式"。

参考 ESPADA 原文：https://arxiv.org/abs/2503.15877

## CFG：部署时调"听话程度"的 knob

classifier-free guidance 在 diffusion 里是个标准 trick，Habilis-β 把它搬到 flow matching。核心 idea：

训练时随机 mask 掉一部分 instruction（换成 null token），让模型同时学 conditional 和 unconditional 两种 prediction。inference 时把两个 prediction 拉出来线性组合：

$$v_{\mathrm{cfg}} = v_{\mathrm{uncond}} + w \cdot (v_{\mathrm{cond}} - v_{\mathrm{uncond}})$$

- $v_{\mathrm{cond}}$：指令条件下的 velocity field
- $v_{\mathrm{uncond}}$：指令被 mask 时的 velocity field（pure motion prior）
- $w$：guidance scale

$w=1$ 就是 normal conditional，$w>1$ 强化指令服从，$w<1$ 偏向 prior。

**直觉**：$v_{\mathrm{uncond}}$ 是"通用物理直觉"，$v_{\mathrm{cond}}$ 是"任务特定方向"。$v_{\mathrm{cond}} - v_{\mathrm{uncond}}$ 就是"指令带来的额外推力"。$w$ 控制这个推力放大几倍。

部署时 $w$ 调大 → policy 更主动 follow 指令但可能过于 aggressive 导致失控；$w$ 调小 → policy 更保守依赖 prior 但可能不严格执行任务。所以 $w$ 是个 deployment-time knob，根据具体任务调。

但 paper 5.3 里 honesty 地承认：**当前 robot dataset 规模撑不起强 unconditional prior**，所以 CFG 在某些 task 上 disable 反而更好。这跟 LLM 早期小模型上 CFG/temperature 调参常常反直觉是一个道理——prior 不强时 guidance 会放大噪声。

参考 CFG 原文：https://arxiv.org/abs/2207.09602

## 实验数据直觉解读

### 仿真结果直觉

Table 1 几个数字值得思考：

- **π0.5 + Normal**: TPH 120.5, MTBI 30.5, Success 47.8% —— baseline 很弱
- **π0.5 + Play+Cyclic**: TPH 207.1, MTBI 49.9, Success 69.3% —— 同模型同架构，光换数据策略就大幅提升，证明数据策略的普适性
- **Habilis-β (w/o ESPADA)**: TPH 250.5, MTBI 72.4, Success 76.8% —— reliability 最优点，加 ESPADA 之前的 Habilis-β 是"又快又稳"的 sweet spot
- **Habilis-β (full)**: TPH 572.6, MTBI 39.2, Success 78.5% —— productivity 最优点，ESPADA 加速一倍但 MTBI 下降，验证 PRP tradeoff

**关键 insight**：ESPADA 把 TPH 从 250.5 拉到 572.6（2.3x），但 MTBI 从 72.4 跌到 39.2（1.85x）。这个 tradeoff 在 success rate 几乎不变（76.8% → 78.5%）的情况下发生——说明跑得快确实增加了**绝对失败次数**，即便**相对失败率**没变。这就是 PRP plane 上"右上角移动一点、左上角移动一点"的差异——部署时根据 scenario 选 operating point。

### 真实世界结果直觉

Table 2 更有意思：

- **π0.5 Normal**: 19 TPH, 46.1s MTBI —— 真实世界 baseline 比 simulation 差很多，约 6 倍 degradation
- **Habilis-β full**: 124 TPH, 137.4s MTBI —— 6.53x TPH + 2.98x MTBI

真实世界 sim-to-real gap 在所有方法上都存在，但 Habilis-β 的 relative improvement 在真实世界甚至比仿真还大。这暗示 Habilis-β 的 design choices（play data、cyclic data、ESPADA、high-freq control）对真实世界的 distribution shift、drift、noise 更 robust，因为这些 design 本身就是为 deployment 设计的。

### GR00T N1.5 表现差的原因猜测

GR00T N1.5 在仿真上 TPH 1.2-5.0，在真实世界上 TPH 33-63，差距很大。可能原因：

1. GR00T N1.5 是 NVIDIA 的 generalist humanoid foundation，对 RoboTwin 这种 aloha-style 仿真任务的 fine-tune 可能没对齐好
2. Diffusion-style action generator 的 multi-step inference 在 high-freq control 上天然不利
3. GR00T 训练数据侧重 humanoid locomotion + whole-body manipulation，对 bimanual table-top 任务可能 distribution 覆盖不足

但 GR00T N1.5 + Play+Cyclic 也能从 1.2 提到 5.0（仿真）、33 提到 63（真实），说明数据策略的普适性跨架构成立。这是个 nice finding——意味着 play data + cyclic data 这个 recipe 可以作为通用 VLA data strategy 推广。

参考 GR00T N1.5：https://developer.nvidia.com/groot
参考 RoboTwin 2.0：https://arxiv.org/abs/2503.09168

## 跟其他 VLA 工作的 positioning 人话版

### 跟 π0 / π0.5 的关系

Habilis-β 架构上跟 π0 是一个 lineage——prefix-suffix + flow matching action expert。但 π0/π0.5 主要靠大规模 task demonstration + multi-step flow inference，Habilis-β 做了几个激进的 deployment-oriented 改造：

- 数据加 play + cyclic
- action expert 用 rectified flow distill 到 2 步
- inference 加 CFG knob
- ESPADA 做 speed shaping
- evaluation 改成 continuous-run

可以理解为 Habilis-β = π0 architecture + deployment recipe 全家桶。

参考 π0：https://arxiv.org/abs/2410.24164
参考 π0.5：https://www.physicalintelligence.company/blog/pi05

### 跟 OpenVLA / RT-2 的关系

OpenVLA、RT-2 走的是"VLM 直接输出 action token"路线，token-based decoding latency 高、high-freq control 困难。Habilis-β 走 flow matching 路线，expressivity 和 efficiency 取的 tradeoff 点更激进——2 步 flow inference 比 token autoregressive decoding 快一个数量级。

这有点像 LLM 里 autoregressive vs diffusion 的 tradeoff——autoregressive 灵活但慢，diffusion 快但表达受限。Habilis-β 选了后者并配 distillation 把效率推到极限。

参考 OpenVLA：https://openvla.github.io/
参考 RT-2：https://arxiv.org/abs/2307.15818

### 跟 Diffusion Policy 的关系

Diffusion Policy 是 flow matching 的前辈，用 iterative denoising 表达 multimodal action distribution。问题在于 step 数多（典型 10-20 步），latency 高。Habilis-β 用 flow matching + rectified distillation 把 step 压到 2，本质上是在 diffusion policy 的 expressivity 上做了一个 efficiency 改造。

参考 Diffusion Policy：https://arxiv.org/abs/2303.04137

## 我的整体 intuition

这篇 paper 真正的 contribution 我觉得不是某个 single trick，而是**把 VLA 从"capability-driven"切换到"deployment-driven"的 evaluation 范式**。每个 design choice 都对应 PRP 上一个具体的移动方向：

- Play data → 移动 MTBI 上（recovery 能力强）
- Cyclic data → 移动 MTBI 上（deployment distribution 对齐）
- ESPADA → 移动 TPH 右（速度提升），代价是 MTBI 略下
- Rectified flow distillation → 移动 TPH 右（高频控制），间接提升 MTBI（drift 抑制）
- CFG → deployment-time knob，根据 task 在 PRP 上动态调整 operating point
- On-device Jetson Orin → predictable latency，消除网络抖动

这种"指标驱动架构"的思路跟 ImageNet 时代 AlexNet → ResNet → EfficientNet 的 benchmark 驱动 architecture design 很像。VLA 圈子以前都在比 capability（single-trial success rate），现在 Habilis-β 提出比 deployment readiness（PRP），可能是个范式转换的早期 signal。

如果这个 evaluation 范式被广泛接受，未来 VLA 论文可能都要报 TPH/MTBI，就像 NLP 论文都要报 latency/throughput 一样。这对整个领域的成熟度是个好事——意味着大家开始认真对待 deployment，而不是只刷 benchmark。

参考 Habilis Console（论文的 data collection 平台）：https://tommoro-ai.github.io/habilis-console/
参考 Tommoro Robotics 官网：https://tommoro.ai

## 一句话 final takeaway

**Habilis-β 告诉我们：VLA 的下一个 frontier 不是"能不能做更难的任务"，而是"能不能连续干 1 小时不需要人救"——这跟 LLM 从"能不能答对 GSM8K"到"能不能当 copilot 用一整天"的演进方向完全一致**。

---

# Habilis-β: 一篇面向部署的 VLA 论文深度解读

## 一、论文的核心 positioning

Habilis-β 来自 Tommoro Robotics（https://tommoro.ai），核心论点非常 Karpathy-friendly：**当前 VLA 研究普遍把 "single-trial success rate under curated resets" 当作主要指标，但这个指标在工业部署场景下几乎是 useless 的**。部署真正关心的是两件事：

1. **Productivity（生产力）**：单位时间内完成多少任务 → 用 Tasks per Hour (TPH) 度量
2. **Reliability（可靠性）**：连续运行多久才需要人介入一次 → 用 Mean Time Between Intervention (MTBI) 度量

这两个指标构成本论文提出的 **Productivity-Reliability Plane (PRP)**。这让我想起 GPT-4o 早期在 Chatbot Arena 上做的那些评估——单轮 benchmark 高分不等于用户长期 retention 高。机器人领域显然更需要这种"长期 retention"式的评估，因为部署一旦卡住就要 E-stop，代价巨大。

论文中三个关键词：**Fast-Motion**（自由空间快、接触阶段精）、**Long-Lasting**（状态漂移下持续运行）、**On-Device**（边缘 Jetson Orin 本地跑）。这套三连击思路和 NVIDIA GR00T N1.5（https://developer.nvidia.com/groot）、Physical Intelligence π0.5（https://www.physicalintelligence.company/blog/pi05）形成鲜明对比。

## 二、PRP 评估框架

### 2.1 Continuous-Run Protocol

评估协议：机器人反复执行同一任务 cycle，持续固定时间 $T = 3600$ 秒（1 小时），环境仅在必要时重置。每次 intervention 的定义有两种：

- **Timeout**：在任务特定时间窗内未完成
- **Abort**（仅真实世界）：操作员主动 E-stop

这个设计本质上把 evaluation 从 episodic 变成 **streaming/continual**，更接近 OEE (Overall Equipment Effectiveness, ISO 22400-2:2014 https://www.iso.org/standard/54540.html) 和 MTBF (IEV 192-05-13 https://www.electropedia.org/) 的工业标准。

### 2.2 公式解析

**TPH（公式 1）**：
$$\mathrm{TPH} = \frac{N_{\mathrm{succ}}}{T / 3600}$$

变量含义：
- $N_{\mathrm{succ}}$：在 $T$ 秒连续运行内成功完成的任务次数
- $T$：wall-clock 持续时间（秒）
- 分母 $T/3600$ 把秒换成小时，所以 TPH 单位就是"每小时成功次数"

**MTBI（公式 2）**：
$$\mathrm{MTBI} = \frac{T}{K}$$

变量含义：
- $T$：总运行时间（秒）
- $K$：intervention 次数（包括 timeout 和 abort）

当 $T = 3600$ 时退化为 $3600 / K$，是个非常工程化的、容易测的指标。

### 2.3 为什么 PRP 比单次 success rate 更 informative

论文 Section 5 的 discussion 部分点破了一个反直觉现象：**更高 TPH 和更高 success rate 并不保证更高 MTBI**。原因是固定时间窗内跑得越快、尝试次数越多，即便 success rate 不变，**绝对失败次数**也会上升，从而 MTBI 下降。这相当于把 productivity-reliability tradeoff 显式化——一个 system 可以"很快但经常 fail"，也可以"很慢但稳定"，这两个点在 PRP 上不在同一个位置。这非常像 ML 中 accuracy-latency tradeoff，但更接地气。

## 三、系统架构（Figure 2 解析）

### 3.1 Prefix-Suffix 设计

整体架构遵循 $\pi_0$（https://arxiv.org/abs/2410.24164）提出的 prefix-suffix 范式：

- **Prefix**：预训练的 VLM（视觉语言模型 backbone），输入多视角图像 + 高级语言指令，输出融合后的多模态 token embeddings 作为 condition
- **Suffix**：Flow Matching Action Expert，吃 VLM prefix 的 token + 本体感知 $s_t$ + 噪声 action chunk $x_\tau$，输出 action chunk $(a_t, \dots, a_{t+H-1})$

输入输出形式化：
- 输入 observation：$o_t = (I_t, s_t, \ell)$
  - $I_t$：图像集（多视角，如 ZED 2i + 双腕 RealSense D405）
  - $s_t$：proprioception 本体感知状态
  - $\ell$：高级语言指令
- 输出：$H$ 步连续 action chunk，维度 $\mathbb{R}^{H \times d}$，$H$ 为 action horizon，$d$ 为 action dimension

每个 camera view 通过 ViT encoder 编码（Dosovitskiy et al., https://arxiv.org/abs/2010.11929），得到 visual tokens，再送入 VLM backbone fusion。

### 3.2 三阶段 Training Pipeline（Figure 1 Left）

**Stage 1: Play Data Pre-training**
- 数据：unstructured、unsegmented 的人类遥操作数据，**无 task label、无 goal specification**
- 目的：学习 task-agnostic interaction prior，覆盖 recovery、regrasping、repositioning 等成功 demonstration 几乎不会出现的状态
- 训练：language-unconditioned，只用 observation-action pair

**Stage 2: Cyclic Task Post-training**
- 数据：cyclic demonstrations——即"success 接着下一轮 success"的连续 stream，而非 success-then-stop 的孤立 episode
- 目的：让 policy 看到周期之间的 state drift 和 recovery transition
- 与 ESPADA 配合：对 demonstration 做 spatially aware downsampling

**Stage 3: Rectified Flow Distillation**
- Teacher：multi-step (≥10) flow matching action expert
- Student：2-step rectified flow model，蒸馏成可 on-device 高频跑的紧凑模型
- 目的：减少 inference cost，腾出 latency budget 做 high-frequency re-inference

### 3.3 Inference Pipeline（Figure 1 Right）

关键在于"用蒸馏省下的 latency 做什么"——答案不是 idle processor，而是**做更频繁的 re-inference**：更短的 action chunk + 更频繁的 closed-loop feedback，直接对抗 drift 累积。最后 CFG 作为 deployment-time knob，动态 balance 指令服从 vs 学习到的 interaction prior。

## 四、关键技术细节

### 4.1 Flow Matching Action Expert（公式 3-5）

Flow Matching 来自 Lipman et al.（https://arxiv.org/abs/2210.02747），核心思想是学习一个 velocity field $v_\theta$ 把噪声分布 $\epsilon \sim \mathcal{N}(0, I)$ 传输到数据分布 $a$。

**公式 3**：
$$\epsilon \sim \mathcal{N}(0, I), \quad \tau \sim \mathrm{Beta}(1.5, 1), \quad \tau \leftarrow 0.999\tau + 0.001$$

变量含义：
- $\epsilon$：从标准 Gaussian 采样的噪声 action chunk，形状 $\mathbb{R}^{H \times d}$
- $\tau$：denoising time，从 Beta(1.5, 1) 采样，让训练时偏向高噪声区域（$\tau$ 接近 1）
- 第三行的 affine 重映射 $\tau \leftarrow 0.999\tau + 0.001$ 把 $\tau$ 从 $[0, 1]$ 压缩到 $[0.001, 1.0]$，避免 $\tau = 0$ 时的数值奇点

**公式 4**：
$$x_\tau = \tau \epsilon + (1 - \tau) a, \quad u_\tau = \frac{dx_\tau}{d\tau} = \epsilon - a$$

变量含义：
- $x_\tau$：插值 noisy action，在 noise $\epsilon$ 和 data $a$ 之间线性插值
- $u_\tau$：target velocity field，即 $x_\tau$ 对 $\tau$ 的导数，是一个**常数向量** $\epsilon - a$（这是 flow matching 相对 diffusion 的好处之一：target 简单稳定）
- $\tau = 0$ 时 $x_0 = a$（纯 data），$\tau = 1$ 时 $x_1 = \epsilon$（纯 noise）

**公式 5**（训练 loss）：
$$\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{\epsilon, \tau, (o, a) \sim \mathcal{D}} \left[ \frac{1}{Hd} \| v_\theta(\tilde{o}, x_\tau, \tau) - u_\tau \|_2^2 \right]$$

变量含义：
- $v_\theta(\tilde{o}, x_\tau, \tau)$：神经网络预测的 velocity field，参数为 $\theta$
- $\tilde{o}$：observation with stochastically masked instructions（CFG 训练时的 random masking）
- $1/Hd$：归一化系数，把 loss 平均到每个 action dimension
- $\mathcal{D}$：trajectory-language pair 数据集

这个 loss 形式跟 diffusion policy（https://arxiv.org/abs/2303.04137）的 noise prediction loss 类似，但 target $u_\tau = \epsilon - a$ 是确定性的、不需要 noise schedule 的微妙加权。

### 4.2 Rectified Flow Distillation（公式 6）

Rectified Flow 来自 Liu et al.（https://arxiv.org/abs/2209.03003；https://arxiv.org/abs/2309.06370），其核心 insight 是：flow matching 学到的 ODE trajectory 一般是弯曲的，但可以**逐步 rectify 成直线**，从而用更少 step 求解。Habilis-β 用的是 distillation 版本（Hinton et al., https://arxiv.org/abs/1503.02531）：

**公式 6**：
$$\mathcal{L}_{\mathrm{distil}}(\phi) = \mathbb{E}_{o_t \sim \mathcal{D}, \tau \sim \mathcal{U}(0, 1), x_\tau \sim p_\theta(\cdot | o_t, \tau)} \left[ \left\| v_\phi(o_t, x_\tau, \tau) - \frac{x_1 - x_\tau}{1 - \tau} \right\|^2 \right]$$

变量含义：
- $\phi$：student 模型参数
- $\theta$：teacher 模型参数（fixed）
- $o_t$：environment time $t$ 时的 observation，由 VLM prefix 编码
- $x_\tau$：teacher 的 ODE trajectory 上 $\tau$ 时刻的中间 state，从 teacher 推出的分布 $p_\theta(\cdot | o_t, \tau)$ 采样
- $x_1$：teacher 完整积分到达的终点 action chunk
- $\frac{x_1 - x_\tau}{1 - \tau}$：target velocity，即从 $x_\tau$ 直接到 $x_1$ 的"直线方向"——这就是 rectification 的精髓：**让 student 学会从任意中间点一步跳到终点**

distill 之后，student 只需 **2 步 function evaluation** 就能生成 action chunk，相比 teacher 的 ≥10 步，inference latency 大幅下降，为 high-frequency control 腾出预算。

### 4.3 Classifier-Free Guidance（公式 7-8）

CFG 来自 Ho & Salimans（https://arxiv.org/abs/2207.09602），原本用于 diffusion，Habilis-β 把它迁移到 flow matching 上。

**公式 7**：
$$v_{\mathrm{cfg}} = v_{\mathrm{uncond}} + w \cdot (v_{\mathrm{cond}} - v_{\mathrm{uncond}})$$

变量含义：
- $v_{\mathrm{cond}}$：指令条件下的 velocity field 预测
- $v_{\mathrm{uncond}}$：指令被 mask 成 null token 时的 velocity field 预测
- $w$：guidance scale，>1 时强化指令服从，<1 时偏向 prior

Intuition：$v_{\mathrm{uncond}}$ 是"任务无关"的 motion prior，$v_{\mathrm{cond}} - v_{\mathrm{uncond}}$ 是"指令带来的额外方向"。$w$ 控制这个额外方向的放大倍数。$w \gg 1$ 时 policy 变得过于 proactive/aggressive，可能造成失控；$w \to 0$ 时退化为完全 unconditional 的 prior。

**公式 8**（guidance rescaling）：
$$v_{\mathrm{rescaled}} = v_{\mathrm{cfg}} \cdot \frac{\|v_{\mathrm{cond}}\|}{\|v_{\mathrm{cfg}}\| + \varepsilon}$$

变量含义：
- $v_{\mathrm{cfg}}$：公式 7 得到的 guided velocity
- $\|v_{\mathrm{cond}}\|$：conditional 预测的 L2 范数
- $\|v_{\mathrm{cfg}}\|$：guided 预测的 L2 范数
- $\varepsilon = 10^{-6}$：numerical stability 常数

这是 Lin et al.（https://arxiv.org/abs/2301.13657）在 WACV 2024 提出的技巧——单纯放大 $w$ 会把 velocity 的 magnitude 也放大，导致 action chunk 速度过快，所以这里把 rescale 到 conditional 的 magnitude，保留方向修正但不放大尺度。

### 4.4 ESPADA: Spatially Aware Downsampling（公式 9）

ESPADA 来自 Kim et al.（同作者前作，https://arxiv.org/abs/2503.15877），是 Habilis-β 实现 Fast-Motion 的关键 data-level shaping。

**思路**：
1. 用 VLM-LLM pipeline 对 trajectory 做语义分割，划分为 "casual"（transit/free-space）和 "precision"（contact-rich）phase
2. 对 casual phase 做 aggressive downsample，对 precision phase 保留高频
3. 采用 **replicate-before-downsample** 策略：先复制再下采样，避免短时接触被 alias 掉

**公式 9**（geometric consistency）：
$$\sum_{h=0}^{H'-1} \|\Delta x'_{t+h}\| \approx \sum_{h=0}^{H-1} \|\Delta x_{t+h}\|$$

变量含义：
- $H' = \lceil H / N \rceil$：downsample 后的 action horizon，$N$ 为 downsampling factor
- $\Delta x_{t+h}$：原始 end-effector displacement at step $t+h$
- $\Delta x'_{t+h}$：加速后对应的 displacement
- 两边求和分别表示原轨迹和加速轨迹的**总空间位移**

intuition：downsample 改变了时间密度，但**空间轨迹必须保持一致**——即机器人最终走过的物理路径不变，只是用更少的时间步走完。这样 policy 学到的是一致的空间分布，而 temporal density 可以根据 phase 自适应。这就是为什么 ESPADA 能大幅提升 TPH 同时基本保持 success rate。

### 4.5 High-Frequency Control

这是整个系统最 Karpathy 风格的 insight：**latency reduction 的回报不是用来 idle 的，是用来 re-infer 更频繁的**。直觉上：

- 传统 VLA：10 步 action chunk × 长执行 horizon → 在 chunk 末端可能已经偏离真实状态很远
- Habilis-β：2 步 action chunk × 高频 re-inference → 闭环反馈更紧，drift 累积被持续抑制

这种设计在 ESPADA 加持下更有效：ESPADA 把 free-space 压缩后，action chunk horizon 变短，而精度阶段仍保持高频控制，所以"短 chunk + 高频"恰好对应"快 transit + 精 contact"。

## 五、数据策略

### 5.1 三种 Data Collection Interface

论文 Appendix B 详细描述了三种数据采集接口，对应不同 tradeoff：

**1. Universal Data Device (UDD)**
- 灵感来自 UMI（https://arxiv.org/abs/2315.02702），但解决了 UMI 的 embodiment mismatch 问题
- 核心 design goal：用**实际部署用的 gripper**作为 front-end，避免 grasp geometry、actuation limit、camera extrinsic 的微小差异在 long-horizon 重复执行中累积
- 模块化快拆接口，支持不同 manufacturer gripper
- 6D pose tracking + gripper actuation logging，可以在没有机器人本体的环境下采集

**2. Meta Quest Hand Tracking**
- 用 VR 设备的 hand tracking，retarget 到 hand embodiment 或 "easy-gripper" abstraction
- 优势：无需 external motion-capture hardware、markers
- pinch/open gesture 直接映射到 gripper command

**3. Leader-Arm Teleoperation**
- manufacturer 提供的 leader-arm（master-follower）接口
- 高精度 robot-native demonstration，与 UDD/hand-tracking 互补

### 5.2 Play Dataset

Play data 来自 Lynch et al.（https://arxiv.org/abs/1912.07241；https://arxiv.org/abs/2104.04690）的传统，Habilis-β 把它用作 **language-unconditioned pre-training**。play data 的价值在于：

- 自然的 regrasping、repositioning、local correction 行为
- 覆盖 success demonstration 几乎不包含的 recovery-adjacent state
- 低成本采集，规模可大

Habilis-β 强调 play data **不是替代 task data**，而是扩展 task manifold 周围的 coverage，让 unconditional prior 更 robust。

### 5.3 Cyclic Task Dataset

这是论文的关键创新之一。Standard imitation learning 数据是 isolated "success-then-stop" episode，作者称为 **non-cyclic data**。问题在于：训练时模型从未见过"上一个 success 结束 → 下一个 cycle 开始"的 transition，部署时这种 transition 累积的 drift 会让 policy 快速 degrade。

Cyclic data 的做法：采集**连续流** demonstration，把一个 success 直接连到下一个 success 的开始，让模型看到 cycle 之间的状态过渡和 drift 累积。论文提到他们 curate 了约 10 小时 focused cyclic dataset 用于 target workflow，既作为 baseline 训练 corpus，也作为 play-pretrain 后的 fine-tuning target。

## 六、实验结果

### 6.1 仿真实验（Table 1 & Figure 4-6）

仿真环境：RoboTwin 2.0（https://arxiv.org/abs/2503.09168），Aloha bimanual platform，三个任务：Dump Bin Bigbin (DBB)、Place Dual Shoes (PDS)、Stack Bowls Three (SBT)。

**Table 1 核心数据**：

| Method | Data Regime | TPH | MTBI(s) | Success Rate |
|--------|------------|------|---------|-------------|
| π0.5 | Normal | 120.5 | 30.5 | 47.8% |
| π0.5 | Play + Cyclic | 207.1 | 49.9 | 69.3% |
| GR00T N1.5 | Normal | 1.2 | 20.4 | 0.5% |
| GR00T N1.5 | Play + Cyclic | 5.0 | 20.6 | 2.1% |
| Habilis-β (w/o ESPADA) | Play + Cyclic | 250.5 | 72.4 | 76.8% |
| Habilis-β | Play + Cyclic + ESPADA | 572.6 | 39.2 | 78.5% |

几个关键观察：

1. **π0.5 + Play+Cyclic** 单独就能把 TPH 从 120.5 提到 207.1，MTBI 从 30.5 提到 49.9——说明数据策略本身有**广泛适用性**，不是 Habilis-β 独享的
2. **GR00T N1.5 表现极差**（TPH 1.2-5.0），可能是因为 GR00T 对 cyclic 长时程控制不擅长，或者在 RoboTwin 2.0 的 aloha-style 任务上 fine-tune 困难
3. **ESPADA 的双刃剑效应**：加 ESPADA 后 TPH 从 250.5 → 572.6（2.29× 加速），但 MTBI 从 72.4 → 39.2（1.85× 退化）。这完美印证 PRP 的 tradeoff 直觉——跑得越快、尝试越多，绝对失败次数越多，即便 success rate 略升
4. **Habilis-β (w/o ESPADA) 是 reliability 最优点**，加 ESPADA 是 productivity 最优点，用户可根据部署场景选 operating point

**Figure 6: Standard RoboTwin 2.0 benchmark** 上 Habilis-β 在 100 episodes、curated reset 的标准协议下也达到 leaderboard 第一，证明 Fast-Motion 和 Long-Lasting 不是以 fundamental manipulation capability 为代价换来的。

### 6.2 真实世界实验（Table 2 & Figure 9-10）

平台：RB-Y1 humanoid（Rainbow Robotics），双臂 + ZED 2i 头部相机 + 双腕 RealSense D405，NVIDIA Jetson Orin 本地推理。任务：Dual-Bin Conveyor Packing (DBCP)——双 source bin 取物 + 动态 conveyor 同步投放。

**Table 2 核心数据**：

| Method | Data Regime | TPH | MTBI(s) | Success Rate |
|--------|------------|------|---------|-------------|
| π0.5 | Normal | 19 | 46.1 | 19.6% |
| π0.5 | Play + Cyclic | 79 | 94.6 | 67.5% |
| GR00T N1.5 | Normal | 33 | 53.7 | 33.0% |
| GR00T N1.5 | Play + Cyclic | 63 | 78.2 | 57.7% |
| Habilis-β | Play + Cyclic + ESPADA | 124 | 137.4 | 82.7% |

对比 π0.5 Normal baseline，Habilis-β 实现了 **6.53× TPH 提升**和 **2.98× MTBI 提升**。这个 magnitude 在真实机器人 VLA 文献中相当惊人。

**Intervention 定义**（真实世界）：
- **Abort**：operator-initiated E-stop，需要 external recovery
- **Timeout**：在 allotted window 内未完成

**Reset 规则**：
- 5 次连续 intervention → 手动 shuffle supply bin 解决 adversarial clutter
- Bin fill level < 10% 且 3 次连续失败 → replenish supply bin

这种规则使 long-duration evaluation 不会因为"环境耗尽"这种非 policy 因素而 fail，让 TPH/MTBI 真正反映 policy 性能。

### 6.3 Figure 10: Continuous Execution Timelines

这张图值得仔细看——1 小时连续执行的 timeline，绿色段是 success，深蓝段是 fail/timeout/abort，橙色菱形是 timeout，红色三角是 abort。Habilis-β 的 timeline 上 success 段密度显著高，intervention marker 显著少，直观体现 long-lasting capability。

## 七、与现有 VLA 工作的 positioning

### 7.1 与 $\pi_0$ / $\pi_{0.5}$ 的对比

$\pi_0$（https://www.physicalintelligence.company/blog/pi0）和 $\pi_{0.5}$（https://www.physicalintelligence.company/blog/pi05）是 Physical Intelligence 的 flow matching VLA 系列。Habilis-β 在架构上借鉴了 $\pi_0$ 的 prefix-suffix 设计和 flow matching action expert，但有几个关键差异：

1. **数据策略**：Habilis-β 引入 play data + cyclic data 双层组合，$\pi_{0.5}$ 主要靠大规模 task demonstration
2. **Speed shaping**：Habilis-β 用 ESPADA 做 phase-adaptive temporal compression，$\pi_{0.5}$ 没有显式 speed shaping
3. **Distillation**：Habilis-β 用 rectified flow distillation 压到 2 步，$\pi_{0.5}$ 仍保持 multi-step
4. **CFG**：Habilis-β 引入 CFG 作为 deployment-time knob，$\pi_{0.5}$ 没有
5. **Evaluation**：Habilis-β 主张 PRP/continuous-run，$\pi_{0.5}$ 仍用 single-trial success rate

### 7.2 与 GR00T N1.5 的对比

NVIDIA GR00T N1.5（https://developer.nvidia.com/groot）用 diffusion-style action generator，强调 humanoid general foundation。Habilis-β 实验中 GR00T N1.5 表现较差（real-world TPH 33→63），可能因为：
- GR00T N1.5 通用性强但 cyclic long-run 优化不足
- Diffusion policy 的 multi-step inference 天然 latency 高，难以 high-frequency control
- 在 RoboTwin 2.0 这种精细 bimanual 任务上需要更 aggressive 的 fine-tuning

### 7.3 与 OpenVLA、RT-2、Diffusion Policy 的关系

OpenVLA（https://openvla.github.io/）和 RT-2（https://arxiv.org/abs/2307.15818）把 VLM 直接 fine-tune 输出 action token，token-based 解码 latency 高。Diffusion Policy（https://arxiv.org/abs/2303.04137）用 iterative denoising 表达 multimodal action，但 step 数多。Habilis-β 走 flow matching + rectified distillation 路线，在 multimodal 表达力和 inference efficiency 之间取了一个非常激进的 tradeoff 点。

### 7.4 与 UMI / SAIL / DemoSpeedup 的对比

UMI（https://arxiv.org/abs/2315.02702）解决 data collection 便携性，但 embodiment mismatch 在 long-run 累积——这是 Habilis-β 设计 UDD 的直接动机。SAIL（https://arxiv.org/abs/2503.13760）和 DemoSpeedup（https://arxiv.org/abs/2410.22037）做 demonstration 加速但是 heuristic 的 uniform downsampling，会 alias 接触阶段。ESPADA 用语义做 phase-aware compression，避免了 uniform 的 aliasing 问题。

## 八、CFG 的 subtlety

Section 5 discussion 第三点很重要：**CFG 需要一个 strong unconditional prior**，这要求大规模 robot dataset，目前机器人数据规模还撑不起像 image generation 那样强的 prior。Empirically：
- 不同 task 上 optimal guidance scale $w$ 不同
- 某些 setting 下 disable CFG 反而最好
- CFG 应被视作 deployment-time knob，不是 universal default

这让我想到 LLM 中 temperature/top-p 的调参——理论上有最优值，但实践中 task-dependent，需要人工 sweep。

## 九、Limitations 和 Future Work

Section 6 提到三个方向：

1. **Dexterous embodiment**：UDD 和 hand-tracking 对五指灵巧手监督有限，需要扩展 action/observation alignment
2. **Tactile feedback**：当前缺触觉，contact-rich phase 鲁棒性受限
3. **Online adaptation**：当前是 offline training，长期部署中重复出现的 failure mode 无法自动修正，需要 safe online continual learning

这三个方向都是当前 VLA 领域公认的 hard problem，尤其 tactile + online adaptation 是下一波突破点。

## 十、对 Karpathy 的 Intuition 提示

如果你要把这篇论文放到自己脑子里的 VLA 大图中，关键 intuition 是：

1. **Evaluation 重塑数据需求**：single-trial success rate 隐含假设 episode 独立，但部署是 streaming 的。Continuous-run 暴露的 drift/recovery 状态分布与非-cyclic 数据不匹配——这是为什么 cyclic data 不仅"加多样性"而是"对齐部署 distribution"。
2. **Latency 是 design lever 而不是 cost**：distillation 省下的 latency 不浪费，直接 re-invest 到 high-frequency re-inference，闭环抑制 drift——这是把"快"和"稳"统一在同一架构内的关键 trick。
3. **Speed shaping 必须语义感知**：uniform 加速会破坏接触 phase 的精细控制，ESPADA 用 VLM-LLM pipeline 做语义分割后再加速，本质上把"什么时候该快"这个 reasoning 显式编码进数据。
4. **CFG 是 conditional generation 范式在 robotics 的延伸**：把 diffusion 时代的 CFG trick 迁移到 flow matching，让 deployment 有一个 explicit knob 控制指令服从 vs prior 强度。这个思路理论上简单，工程上需要 unconditional prior 足够强才有效。
5. **PRP 是 evaluation 层面的 Karpathy-style abstraction**：把 deployment readiness 收敛到一个 2D plane，让 tradeoff 显式化——这非常像 ImageNet top-1 vs latency 在视觉领域的角色。

相关 references web links:
- 论文 project page（如有）: https://tommoro.ai
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
- InstaFlow: https://arxiv.org/abs/2309.06370
- CFG: https://arxiv.org/abs/2207.09602
- Guidance rescaling: https://arxiv.org/abs/2301.13657
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://www.physicalintelligence.company/blog/pi05
- GR00T N1.5: https://developer.nvidia.com/groot
- RoboTwin 2.0: https://arxiv.org/abs/2503.09168
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- UMI: https://arxiv.org/abs/2315.02702
- Play Data (Lynch): https://arxiv.org/abs/1912.07241
- ESPADA: https://arxiv.org/abs/2503.15877
- OpenVLA: https://openvla.github.io/
- RT-2: https://arxiv.org/abs/2307.15818
- SAIL: https://arxiv.org/abs/2503.13760
- DemoSpeedup: https://arxiv.org/abs/2410.22037
- Habilis Console: https://tommoro-ai.github.io/habilis-console/

整篇论文给我的感觉是它把"deployment"这个词从 marketing 术语变成了可量化的工程约束集合（TPH/MTBI/PRP），每个 design choice（play data、cyclic data、ESPADA、rectified flow distillation、CFG、high-freq control）都直接对应一个 deployment 子目标。这种"指标驱动架构"的思路，和 ImageNet 时代"benchmark 驱动 architecture"一脉相承，是 robotics VLA 走向成熟的标志。
