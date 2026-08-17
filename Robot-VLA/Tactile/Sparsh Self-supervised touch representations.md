---
source_pdf: Sparsh Self-supervised touch representations.pdf
paper_sha256: 7cdb5564b17adaa4a08f65a885c1906693983e1185e506bc60f8dae60bd6d838
processed_at: '2026-08-12T09:07:17-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Sparsh 这篇 paper

---

## 这篇 paper 在干嘛

一句话：**给机器人手指头装个"通用大脑"，让它学会"摸"这件事本身。**

现在的触觉感知研究特别碎。每个 lab、每个 task、每个 sensor 都自己训一个模型。做 force estimation 训一个，做 slip detection 训一个，用 GelSight 训一个，换 DIGIT 又得重训。问题在哪？

**标数据太贵了。** 你想让机器人知道"我捏这个杯子用了多大力"，你得上 ATI F/T sensor，得精心夹持，得采几千个样本。想让机器人知道"东西要滑了"，你得用 friction cone 模型一个一个 sample 标。想做 in-hand manipulation，要标物体 6D pose... 基本上每个 task 的 ground truth 都得专门搭一套采集硬件。

Meta FAIR 这帮人就想：**vision 那边早就靠 self-supervised learning 解决类似问题了**。MAE、DINO、I-JEPA 这些方法在 ImageNet 上 pre-train 一个 backbone，下游啥 task 都能 frozen + light head 直接用。那 touch 能不能也这么搞？

答案是可以。他们 curate 了 46 万张触觉图，用四种 SSL 方法各训一个 ViT-B/14 backbone，然后在六个 task 上 frozen 测试。结果：**平均比 task-specific E2E 训练好 95%。**

---

## 触觉传感器长啥样

先讲清楚 input。Vision-based tactile sensor（GelSight、DIGIT、GelSight Mini 这些）本质结构是：

- 一块透明 elastomer gel（类似橡胶）
- 里面或旁边装 RGB LED
- 一个小相机对着 gel 拍

物体压上去，gel 表面变形，光照在变形的 gel 上产生明暗变化，相机拍下来就是一张 HxWx3 的 RGB 图。

**所以从模型角度，touch image 和 natural image 在 pixel 层面就是同构的。** 都是 RGB 图。这给 SSL 迁移提供了物理基础。

但是 touch image 和 natural image 有几个关键不同：

1. **信息高度局部。** 触觉只看到接触面那一小块，看不到全局 context。一张图可能只有中间一小撮有 contact，外面全是静态背景。
2. **高频细节基本无用。** LED 光斑渐变、gel 上的微小气泡、camera 对焦差异——这些对下游 task 完全没意义，但占 pixel 信号很大比例。
3. **sensor-specific bias 严重。** 同一个型号的两台 DIGIT，因为 LED 焊接位置差 1mm，整张图的静态光照分布都不一样。
4. **时序信号。** 单张图看不出来啥，必须看 short window 才能区分"静止按压"和"开始滑动"。

这些差异决定了你不能直接把 ImageNet 上的 SSL recipe 原封不动搬过来，得做一些适配。

---

## 他们做了哪些适配

### 适配一：时序拼接

触觉是时序信号。但 MAE、DINO、I-JEPA 都处理单张图。怎么把时序信息塞进去？

最简单粗暴的方法：**把当前帧 $I_t$ 和 5 步之前的帧 $I_{t-5}$ 在 channel 维拼接，变成 6 通道输入。**

60 FPS sensor 下，5 步 = 83 ms。这个数字不是随便选的——人类检测到 partial slip 后调整 grip force 的反应时间正好是 80ms 左右。所以这个窗口覆盖了"物理上有意义的时序尺度"。

V-JEPA 因为本来处理 video，就直接用 4 帧 clip，跨度 100ms。

**intuition：** touch sensor 帧率太高，相邻帧几乎一模一样，stride 太小 SSL 会学到 trivial identity mapping。stride 太大又丢失 temporal correlation。stride=5 是 slip detection 的物理时间尺度，刚好。

### 适配二：Background subtraction

每个 sensor 实例先采一张 "no-contact reference image" $I_{bg}$。训练时用 $I_t - I_{bg}$。

这一步看起来 trivial，但对 SSL 非常关键。为什么？

SSL 最大的 failure mode 是 model 学 shortcut。比如 DIGIT 三个 LED 在 gel 上的静态光斑分布是 sensor 实例独有的，如果 SSL 不做 background subtraction，model 很可能直接把"LED 光斑位置"当作 identity feature 记住，这在跨 sensor transfer 时立刻崩。

减掉 background 后，模型看到的是 "contact 引起的额外形变"，这是真正有信息的 signal。

### 适配三：哪个 SSL objective 适合 touch

这是 paper 的核心实验。他们用同一个 ViT-B/14 backbone，同一个 460k 数据集，只换 SSL objective，横向对比四种：

**MAE（pixel space reconstruction）**
把图遮掉 75%，让 encoder 看 25%，decoder 重建被遮部分的 pixel。loss 是 L2 pixel distance。

问题：touch image 的高频细节（LED 光斑、gel 纹理）对下游 task 没用，但 MAE 会把 model capacity 浪费在重建这些细节上。等于逼模型 memorize sensor-specific pattern。

**DINO / DINOv2（latent space self-distillation + clustering）**
student 和 teacher 是同一个架构，student 通过 EMA 跟踪 teacher。两者看同一张图的不同 crop，输出要在 prototype space 上一致。loss 是 cross-entropy between student/teacher probability。

直觉：DINO 学的是 "online clustering"——每张图被映射到一堆 prototype 里的某一个，student/teacher 必须一致。这种 objective 对 classification task（slip/no-slip, leather/cotton, success/fail）天然 align。

**I-JEPA / V-JEPA（latent space predictive）**
context encoder 看 masked image，target encoder（EMA）看 target block，predictor 预测 target 的 latent embedding。loss 是 latent embedding 的 L2 distance。

和 MAE 的关键区别：**不在 pixel 层面重建，在 latent 层面预测。** 模型不被惩罚去重建 LED 光斑这种无用高频，只需要预测 target block 的 abstract representation。

作者的 hypothesis：**latent space SSL 在 touch 上应该比 pixel space 强**，因为 touch 的高频细节无信息，pixel reconstruction 浪费 capacity。实验结果验证了这个 hypothesis。

---

## TacBench：六个 task 的 benchmark

光有 pre-trained model 不够，得证明它真的 general。作者搞了一个 TacBench，六个 task 覆盖三个层次：

**Tactile properties（物理属性）**
- [T1] Force estimation：3 轴力估计，RMSE metric
- [T1A] Force field visualization：法向 + 切向力场可视化，unsupervised
- [T2] Slip detection：滑检测，F1 metric

**Perception（感知）**
- [T3] Pose estimation：物体相对 sensor 的 SE(2) pose change
- [T4] Grasp stability：抓取成功/失败分类
- [T5] Textile recognition：20 类织物分类

**Manipulation planning（操作规划）**
- [T6] Bead maze：机器人沿 wire 滑珠子的 imitation learning policy

每个 task 都做了 data budget ablation：100% / 33% / 10% / 1% labeled data，看 frozen SSL encoder 在低数据下相对 E2E 的优势。

---

## 关键结果

### 整体数字

**Sparsh frozen + attentive decoder 平均比 E2E 好 95.1%**，在 33-50% labeled data 下。

DINO 和 I-JEPA 是表现最好的两个，MAE 明显弱。这验证了"latent space > pixel space for touch"。

### [T1] Force estimation

DIGIT 上用 1/100 数据（500 样本）：
- Sparsh (DINO) RMSE 98 mN
- E2E RMSE 188 mN

也就是说，**500 个样本下 frozen DINO backbone 几乎逼近 E2E 用全量 50k 样本的性能。** 这对实际 lab 意义巨大——你想标 force 通常采几百个样本就到极限了。

GelSight Mini 上差距更大：E2E 在 HD 分辨率下几乎训不动，Sparsh (DINO) 用 33% data 就把 RMSE 压到 23 mN。

### [T2] Slip detection

V-JEPA 是这个 task 的赢家。

1/100 data 下：
- Sparsh (V-JEPA) F1 = 0.760
- E2E F1 = 0.214（基本瞎猜）

V-JEPA 强是因为它输入 4 帧 clip，slip 本质是时序现象（sticking → incipient slip → sliding），4 帧 temporal context 让 model 直接学到 slip transition。

有意思的发现：作者展示了一个 failure case（Figure 13），ground truth 因为 friction coefficient $\mu$ 估计不准而标错，但 V-JEPA 的预测实际上比 ground truth label 更对。**SSL 学到的 representation 比人工 label 更准确**，这开启一个有意思的方向：用 SSL 反过来 refine noisy labels。

### [T3] Pose estimation

33% data 下：
- Sparsh (DINO) accuracy 0.834
- E2E accuracy 0.245

E2E 在低数据下直接崩塌——confusion matrix 显示它把所有样本 default 到 0 或 max class。Sparsh 能正确区分相邻的 bin（比如 0.5°-1° vs 1°-2°）。

### [T5] Textile recognition

这个 task MAE 意外地最强（0.599 vs DINO 0.527）。

为什么？织物分类本质是 texture recognition，texture 是高频细节，pixel-level feature 恰好有 inductive bias 优势。这是 paper 里唯一一个 pixel space SSL 赢的 task。

**intuition：** 不是所有 touch task 都需要 abstract representation。材料识别这种 task 需要高频，pixel space 反而对。但 force、pose、slip 这种 physics-based task，latent space 完胜。

### [T6] Bead maze

Franka arm + 装了 DIGIT 的手，夹珠子沿 wire 滑。纯触觉任务，vision 完全被手遮住。

用 Diffusion Policy 做 imitation learning，把原来 Diffusion Policy 里的 vision CNN encoder 换成 frozen Sparsh encoder。

Real robot rollout 结果：
- Sparsh (DINO) frozen：10.8 cm 平均距离
- E2E：6.7 cm

Sparsh 比 E2E 好 53%。**但所有 model 都没能完成整个 maze**，compounding error 让珠子最终掉下来。

作者很诚实地讨论了 limitation：这个 task 精度要求极高，一旦 grip 丢失无法 recovery，local decision-making 导致 drift。需要 temporal ensemble 或 force control 才能解决。

### Cross-sensor few-shot（Table 14）

这是最强的 generalization 证据。

把 GelSight-trained 的 textile classifier 拿到 DIGIT 上测：
- Zero-shot：Sparsh (DINO) 9.1%，E2E 3.6%（chance=5%）
- 10-shot：Sparsh (DINO) 61.8%，E2E 10.9%

DIGIT 和 GelSight 光学结构、分辨率、marker 完全不同，但 SSL pre-training 让 representation 跨 sensor invariant。10 个样本就能 transfer。E2E 学的全是 GelSight-specific pattern，10-shot 下基本等于瞎猜。

---

## 为什么 latent space SSL 在 touch 上更强

作者的 hypothesis 和实验验证：

1. **Touch image 高频细节无信息。** LED 光斑、gel 纹理、camera 噪声——这些对 force、slip、pose 完全没用，但占 pixel signal 大头。MAE 逼 model 重建这些，浪费 capacity。

2. **Touch 是物理 ambiguous。** 同一张 contact image 可能由不同 force + shape + material 组合产生。Pixel reconstruction 强迫 model 选一个 specific 解，latent prediction 允许 model 学 distributional representation。

3. **JEPA 的 predictor 学的是 abstract mapping。** "context → target" 的 abstract 关系，这种 abstraction 恰好是 force、slip、pose 这些物理量需要的。

DINO 和 I-JEPA 在不同 task 上各有千秋：
- DINO 强在 force、pose（physics-based，需要 spatial precision）
- I-JEPA 强在 slip、grasp stability（semantic understanding）
- 平均下来 DINO 比 I-JEPA 高 5.6%

V-JEPA 在 slip detection 上最强，但在其他 task 平均反而低，因为 4 帧 clip 的 temporal modeling 和 spatial abstraction 是 trade-off。

---

## 训练细节里几个有意思的点

**Online probe 监控。** JEPA 的 training loss 不能反映 convergence——latent distance 会持续下降但 representation 可能 collapse。作者用 DPT decoder 把 embedding 解码回 tactile image，看 reconstruction quality 作为 proxy metric。这是从 LeCun 那一脉学来的 trick。

**DINOv2 用 ViT registers 替代 cls token。** DINOv2 后期发现 ViT 内部会出现高 norm 的 artifact token（registers），把它们 repurpose 成 prototype prediction head 反而能清理 main tokens 的表示。Paper 借用了这个 trick。

**I-JEPA/V-JEPA 用 6.25x 更大的 LR。** 因为 JEPA 的 target 是 EMA-smoothed，gradient 方向更稳定，能承受激进 LR。MAE 不用 EMA，必须小 LR 防 collapse。

**Fine-tuning 的差异。** Latent space SSL（DINO/IJEPA/VJEPA）的 weight 处于 wider minima（EMA 更新让 loss landscape 平滑），full fine-tuning 能 fine-tune 到 task-specific optimum。MAE 直接 SGD 更新，weight 在 sharper minimum，fine-tuning 容易跳出 pre-trained basin。

---

## 对你（Andrej）可能的相关联想

这篇 paper 本质上是把 vision SSL 的 playbook 应用到 touch，验证了你和 LeCun 一直主张的几个观点：

**Latent space SSL > pixel space SSL for abstract reasoning.** 这是 I-JEPA / V-JEPA / DINO vs MAE 的核心对照。MAE 只在 texture recognition 这种需要高频细节的 task 上能赢。

**Pre-trained frozen + light decoder 在 low-data regime 碾压 E2E.** 500 个 labeled sample 就能让 frozen backbone 接近 E2E 全量数据性能。这正是 LLM recipe 在 robotics modality 上的复现。

**Cross-sensor generalization via shared SSL objective.** 类似 GPT 在多语言上 transfer，Sparsh 在 DIGIT / GelSight 之间 transfer。

**Temporal tokenization 是 key design choice.** stride=5 对应 80ms 物理时间尺度，类似 LLM tokenization 需要匹配 phoneme/word 物理尺度。

如果你想 follow up，几个方向：

- **Sparsh + VLA：** 接到 OpenVLA 或 π0 上，做 visuo-tactile-language policy
- **Tactile world model：** V-JEPA + action conditioning，预测 next tactile state given action
- **Cross-modal distillation：** 用 CLIP / SigLIP distill 到 Sparsh，让 touch 有 language grounding
- **SSL 反过来 refine noisy labels：** slip detection 那个 failure case 暗示这可行

---

## 我觉得这篇 paper 的真正贡献

不是某个 SOTA number，而是**把 touch sensing 从"每个 lab 训自己的 encoder"推向"用 pre-trained backbone + light decoder"的范式**。

这正是 vision 和 NLP 早就走过的路。Touch 因为 modality 特殊（数据稀缺、sensor 多样、ground truth 难采）一直没能走上这条路。Sparsh 通过 careful engineering（时序拼接、background subtraction、4 种 SSL 横向对比、6 个 task benchmark）证明这条路在 touch 上也 work。

后续如果有人 curate 更大的 touch dataset（比如 10M 级别），scale up backbone 到 ViT-L 或 ViT-H，很可能在 manipulation 上打开 dexterous manipulation 的新局面。

---

# Sparsh: Self-supervised Touch Representations for Vision-based Tactile Sensing 深度解析

Andrej, 这篇 paper 从 FAIR Meta 出来的,我觉得它本质上是把 computer vision 里 SSL 的那套 "recipe" (MAE, DINO, I-JEPA, V-JEPA) 搬到 vision-based tactile sensing 这个 "modality 受限、数据稀缺、ground truth 难采" 的领域,并且认真回答了 "latent space vs pixel space 哪个更适合 touch" 这个问题。下面我从 motivation → method → benchmark → results → intuition 的顺序展开,尽量把数字、公式、架构细节都摆出来。

---

## 1. Motivation: 为什么 touch 需要一个 general backbone

Vision-based tactile sensors (GelSight [1], DIGIT [3], GelSight Mini, GelSlim [2], TacTip, Digitac [4] 等) 本质上是一个 "elastomer + camera + 照明" 的光学结构:物体压在 gel 上,gel 表面形变被相机拍下来。所以从 raw signal 角度看,**touch image 和 natural image 在 pixel 层面是同构的**,都是 HxWx3 的 RGB。这给 SSL 迁移提供了物理基础。

但 touch 领域现状是:
- 每个 task(force estimation [17], slip detection [18], pose estimation [10], grasp stability [8]) 都训练 task-specific encoder;
- 每个 sensor(GelSight 2017 with markers vs DIGIT markerless vs GelSight Mini markerless)光学特性、分辨率、光照、gel 标记都不同;
- Ground truth 极难采: force 要 ATI F/T sensor + 精密夹具;slip 要标 friction cone;extrinsic contact 几乎 infeasible;
- 已有的 "touch representation learning" 工作(T3 [45], UniT [46])要么 sensor-specific,要么只支持 GelSight Mini markers 一种。

作者的核心 claim:**touch 也需要一个像 ViT/CLIP 那样的 general pre-trained backbone,可以在多个 sensor、多个 task 上 frozen + light decoder 直接用,在低 labeled data regime 下碾压 E2E**。

参考链接:
- Project page: https://sparsh-ssl.github.io/
- MAE paper: https://arxiv.org/abs/2111.06377
- DINO: https://arxiv.org/abs/2104.14294
- DINOv2: https://arxiv.org/abs/2304.07193
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2312.06692

---

## 2. 数据: 460k+ 触觉图像的 curation

数据规模在 touch 领域是关键瓶颈。作者合并了三个公开 dataset + 一个新 dataset:

| Dataset | 来源 | Sensor | Frames | 用途 |
|---|---|---|---|---|
| YCB-Slide [9] | 公开 | DIGIT (不同 optical) | ~180k | sliding 交互 |
| Touch-and-Go [20] | 公开 | GelSight | ~220k (全部) | in-the-wild 离散 contact |
| ObjectFolder 2.0 [21,37] | 公开 | 多模态真机 | ~81k | controlled 离散 contact |
| **Touch-Slide (新)** | 本文 | DIGIT | ~180k | 9 个 toy-kitchen 物体 sliding |

合计 ~661k images,其中 70% (约 462.7k) 用于 SSL pre-training,剩余 30% 留作 online probe 监控训练。

**关键设计:** SSL 训练里 Touch-and-Go 和 ObjectFolder 的所有 frame 都用(包括原始 paper 里被认为是 "non-contact" 的 frame),因为 SSL 不需要 label。这是一个重要 trick —— 把 "negative" 样本也纳入,等价于让模型见过 "no contact" 状态,这对下游 force estimation 的 zero baseline 很关键。

---

## 3. 方法: 把 SSL recipe 移植到 touch

### 3.1 输入 tokenization:时序拼接

触觉天然是**时序信号**,slip、force change 这些任务需要短窗口历史。作者用一个简单但有效的处理:

对于 image-based SSL 方法 (MAE, DINO, I-JEPA):
$$I_t \oplus I_{t-5} \to x \in \mathbb{R}^{h \times w \times 6}$$

把当前帧 $I_t$ 和 5 步之前的帧 $I_{t-5}$ 在 channel 维拼接成 6 通道。在 60 FPS sensor 下,5 步 = 83 ms,这恰好接近人类检测 partial slip 后调整 grip force 的反应时间 ~80 ms [53]。

对于 video-based SSL (V-JEPA):
$$[t, t-2, t-4, t-6] \in \mathbb{R}^{4 \times h \times w \times 3}$$

4 帧的 clip,跨度 ~100 ms。

**intuition:** 这个 stride 选择不是任意的 —— touch sensor 的帧率很高(60 FPS),相邻帧之间 pixel 差异极小(几乎静止),stride 太小会让 SSL 学到的是 trivial identity mapping,stride 太大会丢失 temporal correlation。stride=5 是 slip detection 的物理时间尺度,这是 paper 里少见的把 "neuroscience 约束" 编码进 input representation 的做法。

### 3.2 Background subtraction 处理 distractor

DIGIT 和 GelSight Mini 是 markerless 的,但 manufacturing discrepancy 会导致:
- LED 位置略有不同 → 静态光照分布不同
- gel 内部气泡/瑕疵 → 静态纹理
- camera 对焦微差 → 静态模糊分布

作者对每个 sensor 实例采集一个 "no-contact reference image" $I_{bg}$,训练时用 $I_t - I_{bg}$ 作为输入。这相当于显式告诉模型 "static shear 信息是垂直力施加时 gel 的整体形变模式",把 sensor-specific 的静态 distractor 从表示空间里抹掉。

**为什么这对 SSL 重要:** SSL 最大的 failure mode 是 model 学到 "shortcut" —— 比如直接 memorize LED 光斑位置作为 identity feature。Background subtraction 把这种 shortcut 显式去除,让 SSL 必须从 residual signal 里学习真正的 contact dynamics。

### 3.3 四个 SSL 范式:pixel space vs latent space 的对照实验

这是 paper 的核心 ablation:同一个 ViT-B/14 backbone,同一个数据集,只换 SSL objective。

#### (a) Sparsh (MAE) —— pixel space reconstruction

$$\mathcal{L}_{\text{MAE}} = \|\mathbf{I}_{\text{target}} - \mathbf{I}_{\text{recon}}\|_2^2$$

变量:
- $\mathbf{I}_{\text{target}} \in \mathbb{R}^{H \times W \times 3}$:被 mask 掉的 patch 的原始 pixel
- $\mathbf{I}_{\text{recon}}$:decoder 重建出来的 pixel
- $\|\cdot\|_2^2$:L2 squared 范数,在 masked patch 上求和

MAE 训练 75% mask ratio,encoder 只看 25% visible patches,decoder 是一个轻量 ViT 重建 masked patches。

**在 touch 上的问题:** tactile image 的高频细节(光照梯度、marker 点)对下游 task 没用,但 MAE 会把 model capacity 浪费在重建这些细节上。例如 DIGIT 的红绿蓝 LED 在 gel 上产生的渐变是高度 sensor-specific 的,pixel-level 重建会逼迫 encoder 记住这种 sensor pattern。

#### (b) Sparsh (DINO) / Sparsh (DINOv2) —— self-distillation + clustering

$$\mathcal{L}_{\text{DINO}} = -\sum \mathbf{p}_t \log \mathbf{p}_s$$

变量:
- $\mathbf{p}_s$: student network 输出经过 softmax + centering + sharpening 后的概率分布(在任意选定的 prototype 数量上,比如 65536 类)
- $\mathbf{p}_t$: EMA teacher network 输出的概率分布
- 求和是在 prototype 维度上做

DINO 的核心机制:
- Student 和 teacher 是 identical architecture(ViT-B/14);
- Teacher 通过 EMA 更新: $\theta_t \leftarrow \lambda \theta_t + (1-\lambda) \theta_s$,paper 里 $\lambda = 0.998$;
- Student 看 local + global crops,teacher 只看 global crops(防止 trivial collapse);
- Stop-gradient on teacher path;
- Centering + sharpening 防止 collapse to uniform。

DINOv2 在 DINO 基础上加了 iBOT [75] 的 patch-level MIM,所以同时有 image-level 和 patch-level supervision。Paper 里 DINOv2 用 ViT registers [104] 替代 [cls] token 来做 prototype prediction(registers 是 DINOv2 后期发现的高 norm artifact token,把它们 repurpose 成预测 head 反而能清理 main tokens 的表示)。

**intuition:** DINO 学到的是 "swav-style online clustering" —— 每个 image 被映射到一个 prototype space,student/teacher 必须一致。这种 objective 对 touch 极合适,因为很多 touch task 本质是 classification(slip vs no-slip, leather vs cotton, grasp success vs failure),prototype space 自然 align。

#### (c) Sparsh (I-JEPA) / Sparsh (V-JEPA) —— latent space predictive

$$\mathcal{L}_{\text{jepa}} = \sum_{i \in M} \sum_{j \in B_i} \|\hat{\mathbf{s}}_{y_j} - \mathbf{s}_{y_j}\|_2^2$$

变量:
- $M$: global context masks 的集合(对 image 做 random block masking 的 context 区域)
- $B_i$: 第 $i$ 个 context mask 对应的 target blocks 集合(target 是 image 的另一部分 local crops)
- $\hat{\mathbf{s}}_{y_j} \in \mathbb{R}^d$: predictor 网络基于 context encoder 的 output 预测出的 target embedding
- $\mathbf{s}_{y_j} \in \mathbb{R}^d$: EMA target encoder 在 target block 上输出的真实 embedding
- $d = 768$ (ViT-B/14 的 hidden dim)

I-JEPA 的核心架构:
- Context encoder: ViT-B/14,看 masked image (visible context)
- Target encoder: EMA of context encoder,看 target block(完整 unmasked)
- Predictor: small ViT,输入 context embedding + target mask position,输出对 target embedding 的预测

V-JEPA 把这个扩展到 video:用 tube masking [88] 沿时间轴做 mask,aspect ratio 多样化。Paper 里 V-JEPA 输入 4 帧 clip。

**为什么 latent space > pixel space 对 touch:** 
1. Tactile image 高频细节无信息,latent prediction 自动 ignore 这些(模型不被惩罚去重建 LED gradient);
2. Touch 是物理 ambiguous —— 同样的 contact image 可能由不同 force+shape 组合产生,pixel reconstruction 会强迫模型 memorize 一个 specific 解,latent prediction 允许 model 学到 distributional representation;
3. JEPA 的 predictor 学的是 "context → target" 的 abstract mapping,这种 abstraction 恰好是 force、slip、pose 这些物理量需要的。

---

## 4. TacBench: 6 个 task 的标准化 benchmark

这是 paper 另一个核心贡献。Task 设计覆盖三个层次:

| Task | 类型 | Sensor | 数据量 | Label | Metric |
|---|---|---|---|---|---|
| [T1] Force estimation | tactile property | DIGIT + GelSight Mini | 75k each | 3-axis force | RMSE (mN) |
| [T1A] Force field viz | tactile property | DIGIT + GelSight Mini | unsupervised | N/A | qualitative |
| [T2] Slip detection | tactile property | DIGIT | 125k (13% slip) | binary | F1 |
| [T3] Pose estimation | perception | DIGIT (Allegro hand) | 49k | SE(2) | accuracy |
| [T4] Grasp stability | perception | GelSight 2017 | 9.3k (Feeling of Success) | binary | accuracy |
| [T5] Textile recognition | perception | GelSight 2017 | 120k | 20 classes | accuracy |
| [T6] Bead maze | manipulation | DIGIT (Franka) | 34k | joint angles | distance traversed |

### 4.1 [T1] Force estimation 细节

数据采集:用 robot arm 把 DIGIT/GelSight Mini 压在固定 indenter 上(hemisphere/sharp/flat 三种),normal force 0-5N 随机,然后 slide 2mm 产生 shear。F/T sensor 给 3-axis force ground truth,60fps (DIGIT) 或 25fps (GelSight)。

训练:force 归一化到 [-1, 1],L1 loss,Adam optimizer。**关键 design:** sharp + sphere 用于 train,flat 用于 test,这强制 test 验证 generalization 到 unseen contact geometry。

### 4.2 [T2] Slip detection 细节

Slip labeling 用 friction cone:
$$\sqrt{f_x^2 + f_y^2} > \mu_s \cdot f_z \Rightarrow \text{slip}$$

其中 $\mu_s$ 是经验估计的 static friction coefficient,$f_z$ 是 normal force,$f_x, f_y$ 是 shear force。

**问题:** $\mu_s$ 是 indenter-sensor 对的属性,实验估计有误差,导致 ground truth boundary 不准 —— paper 在 Figure 13 里展示了一个 failure case,Sparsh (V-JEPA) 的预测实际上比 ground truth label 更合理(ground truth 因为 $\mu$ 估计错而标错)。

Joint training trick:同时训练 slip detection (cross-entropy) 和 force change Δ (MAE),因为 slip 和 Δforce 高度相关,joint supervision 互相 regularize。

### 4.3 [T3] Pose estimation 细节

输入: DIGIT observation $\mathbf{z}_t \in \mathbb{R}^{h \times w \times 3}$ + object pose $\mathbf{T}_t \in SE(3)$
预处理:转成相对 pose change $\mathbf{S}_t^{t-1} \triangleq (\Delta x, \Delta y, \Delta \theta) \in SE(2)$
Binning: translation ±5mm, rotation ±2°,log-uniform binning(regression-by-classification [10,66])
每个 DOF 一个 head,11 classes,cross-entropy loss。

**Log-uniform binning 的 intuition:** 大多数 pose change 集中在 0 附近,如果用 uniform binning,小 change 区域分辨率不够。Log-uniform 让 0 附近的 bin 更密,远处更稀,匹配真实数据分布。

### 4.4 [T6] Bead maze: manipulation policy

这是 paper 里唯一一个 policy learning task。
- Hardware: Franka arm + Allegro-style hand + DIGIT on fingers
- Task: 用 thumb+index 夹住 bead,沿着 wire maze 滑动
- Pure tactile:vision 完全 occlude by hand,bead 的微小位移无法从 vision 看到
- 数据:50 demonstrations,~34k (tactile image, joint angles) pairs
- Policy: Diffusion Policy [71],observation horizon=2, action horizon=8
- Action: $\mathbf{a} \triangleq (\Delta q_t, \Delta q_{t+1}, \ldots); \Delta q \in \mathbb{R}^7$ (Franka 7-DOF delta joint angles)
- 用 Sparsh encoder 替换 Diffusion Policy 原本的 CNN vision encoder

**为什么用 Diffusion Policy:** 这是 robot imitation learning 的 SOTA 之一,它的 denoising diffusion over action space 天然处理 multimodal action distribution,对 tactile 这种 noisy + multimodal signal 很合适。

---

## 5. 评估 protocol: frozen encoder + attentive probe

作者坚持 frozen evaluation,目的是 "what does SSL alone learn?"。

Probe 架构(Figure 7):
- **Attentive probe**(用于 T1, T2, T3, T4, T5):cross-attention layer + 2-layer MLP
  - Embedding dim 768,12 heads,depth 1,MLP ratio 4.0,layer norm yes
- **DPT decoder**(用于 T1A force field viz):reassemble + fusion modules at encoder layer 2,5,8,11,upsample 到 full resolution

**对比 baseline:** End-to-End (E2E) —— 同样 encoder + decoder 容量,但 encoder 随机初始化,全部参数一起训。

**Data budget ablation:** 每个 task 训练 100% / 33% / 10% / 1% labeled data,看 frozen SSL pre-training 在 low-data regime 的优势。

---

## 6. 核心实验结果与分析

### 6.1 总体数字

| Metric | 值 |
|---|---|
| SSL pre-training 平均提升 over E2E | **95.1%** (33-50% labeled data) |
| Sparsh (DINO) vs Sparsh (IJEPA) | DINO 平均高 5.6% |
| Sparsh (MAE) vs best latent SSL | MAE 平均低 5.57% |
| Sparsh (VJEPA) vs best | VJEPA 平均低 24.47% |

Table 13 的分解:
- Force estimation (DIGIT): DINO 比 E2E 好 28.31%
- Force estimation (GelSight): DINO 比 E2E 好 59.74%(GelSight HD resolution 对 E2E 更难)
- Slip detection: IJEPA 比 E2E 好 242.70%(E2E 在低 data 下崩塌到 F1=0.238)
- Pose estimation: DINO 比 E2E 好 235.89%
- Grasp stability: IJEPA 比 E2E 好 5.14%(task 本身上限)
- Bead maze: DINO 比 E2E 好 19.72%

### 6.2 [T1] Force estimation 的细节表(Table 5, DIGIT)

| Model | Full (50k) | 1/3 | 1/10 | 1/100 |
|---|---|---|---|---|
| E2E | 39.34 | 61.42 | 98.22 | 187.51 |
| Sparsh (MAE) | 36.61 | 45.96 | 58.55 | 115.39 |
| **Sparsh (DINO)** | 36.09 | **44.03** | **51.89** | 97.95 |
| Sparsh (DINOv2) | **29.31** | **26.85** | **37.66** | 185.86 |
| Sparsh (IJEPA) | 40.27 | 60.04 | 86.57 | 130.37 |
| Sparsh (VJEPA) | 39.38 | 56.34 | 76.11 | 130.83 |

**关键观察:** 
1. Full data 下,Sparsh (DINOv2) RMSE 29.31 mN,比 E2E 39.34 mN 低 25%;
2. 1/100 data (500 samples!) 下,Sparsh (DINO) 97.95 mN 仍然可用,E2E 已经崩到 187.51 mN —— 接近 random;
3. DINOv2 在 1/100 反而崩到 185.86,作者没明确解释,但合理猜测是 DINOv2 的 iBOT patch-level MIM 让 model 更依赖 large data 的 prototype calibration;
4. 1/100 data 下 Sparsh (DINO) 仍然 sub-100 mN force error,这是非常实用的结果 —— 实际 robotic lab 想标 force 通常只能采几百个样本。

### 6.3 [T2] Slip detection (Table 7)

V-JEPA 全面碾压:
- Full data: V-JEPA F1=0.820, E2E F1=0.767
- 1/100 data: V-JEPA F1=0.760, E2E F1=0.214

**V-JEPA 为什么强于 I-JEPA:** V-JEPA 输入 4 帧 clip,I-JEPA 输入 2 帧拼接。Slip 本质是时序现象(sticking → incipient slip → sliding),4 帧的 temporal context 让 model 直接学到 slip transition pattern,而 I-JEPA 必须从 2 帧 static representation 里推断。

### 6.4 [T3] Pose estimation (Table 8)

DINO 在 33% data 下 accuracy 0.834 vs E2E 0.245 —— E2E 在 low data 下 default 到 zero 或 max class(在 Figure 14 confusion matrix 里可见对角线塌成两个端点)。

### 6.5 [T5] Textile recognition (Table 10)

Sparsh (MAE) 在这个 task 上意外强(Full data 0.599,接近 best)。这反映 texture recognition 本质需要 pixel-level feature(材料纹理是高频细节),pixel space SSL 在这里反而有 inductive bias 优势。

### 6.6 [T6] Bead maze (Table 11)

Real robot rollout 平均距离:
- Sparsh (DINO) frozen: 10.80 cm
- Sparsh (IJEPA) frozen: 9.4 cm
- Sparsh (MAE) frozen: 10.2 cm
- E2E: 6.70 cm

Sparsh 比 E2E 高 20-53%,但**没有一个 model 能完成整个 maze** —— compounding error 让 bead 最终掉出来。作者把这个归因于:
- High precision 要求(bead 必须严格沿 wire,任何偏离都 fatal)
- 一旦 grip 丢失无法 error recovery
- Local decision-making 导致 trajectory drift

### 6.7 Cross-sensor few-shot (Table 14)

把 GelSight-trained textile classifier 用到 DIGIT:
- Zero-shot: Sparsh (DINO) 9.1% vs E2E 3.6% (chance=5%)
- 10-shot: Sparsh (DINO) 61.8% vs E2E 10.9%

**这是 paper 最强的 "general representation" 证据:** SSL pre-training 学到了 cross-sensor invariant feature,即使 DIGIT 和 GelSight 光学结构、分辨率、marker 都不同,10 个样本就能 transfer。E2E 在 10-shot 下只有 10.9% 接近 chance,因为它学到的全是 GelSight-specific pattern。

---

## 7. 架构与训练细节

### 7.1 Hyperparameters (Table 1)

| Model | EMA decay | LR | Batch size |
|---|---|---|---|
| Sparsh (MAE) | N/A | 1e-4 | 100 |
| Sparsh (DINO) | 0.998 | 1e-4 | 150 |
| Sparsh (IJEPA) | 0.996 | 6.25e-4 | 150 |
| Sparsh (VJEPA) | 0.996 | 6.25e-4 | 150 |

所有模型:8x A-100 80GB GPU,150 epochs,AdamW,weight decay cosine schedule 0.04→0.4,30 epochs LR warmup。

**Note:** I-JEPA/V-JEPA 用了 6.25x 更大的 LR,因为 JEPA 的 latent target 是 EMA-smoothed,gradient 方向更稳定,可以承受更激进 LR。MAE 不用 EMA,所以必须小 LR 防 collapse。

### 7.2 模型容量 (Table 2)

| Model | Parameters | FPS (RTX 3080) |
|---|---|---|
| Sparsh (MAE) | 86.25M | 104 |
| Sparsh (DINO) | 86.26M | 112 |
| Sparsh (IJEPA) | 86.39M | 112 |
| Sparsh (VJEPA) | 86.54M | 60 |

ViT-B/14 backbone 86M 参数,推理 60-112 FPS 完全满足 real-time control (60 FPS sensor)。

### 7.3 Online probe 监控

JEPA 类方法的 training loss 不能反映 convergence(latent distance 会持续下降但 representation 可能 collapse)。作者用 DPT decoder 把 Sparsh embedding 解码回 tactile image(Figure 5),看 reconstruction quality 作为 proxy。这是从 LeCun 那一脉 JEPA paper 里学的 trick。

---

## 8. Ablation: Fine-tuning vs Frozen (Appendix E.1)

| 策略 | 适用场景 | 结果 |
|---|---|---|
| Frozen + attentive probe | 默认 | SSL pre-training 学到什么 |
| Partial FT (last block) | 微调 | 性能 ≈ frozen,提升很小 |
| Full FT (all params) | regression task | latent-space SSL (DINO/IJEPA/VJEPA) 显著提升 |
| Full FT on MAE | pixel-space SSL | 提升有限,MAE weights 更 brittle |

**解释:** Latent-space SSL 的 weight 处于 wider minima(EMA 更新让 loss landscape 平滑),full FT 能 fine-tune 到 task-specific optimum。MAE 直接 SGD 更新,weight 在 sharper minimum,FT 容易跳出 pre-trained basin。

---

## 9. Limitations & 我的思考

作者承认的:
1. **数据偏 discrete contact**,shear interaction 稀少 —— 限制 slip/shear task 上限;
2. **没 ablate temporal history length** —— stride=5 是 hardcode,可能 sub-optimal;
3. **Bead maze 在真机上 fail** —— compounding error 问题没解决,需要 temporal ensemble [73] 或 force control;
4. **Tactile simulator 没用** [95-100] —— 因为 simulator 难以建模 shadows 和 per-sensor discrepancy,作者放弃 sim2real 路线,纯 real data SSL。

我额外想到的几点:
1. **Sensor diversity 仍然有限** —— 只支持 GelSight 家族 + DIGIT,没覆盖 TacTip(不同 optical principle)、BioTac(流体)、XELA uskin(分布式电阻)。Paper 里 [35] 已经探索 uskin,但 Sparsh 没扩展到这类。
2. **Action space 没进 SSL** —— bead maze 用 proprioception $q_t$ 作为额外 input,但 SSL pre-training 只用 tactile image。一个有意思的方向是把 (tactile, proprioception, action) 一起做 JEPA,类似 world model 风格 [52]。
3. **没有 multimodal alignment** —— Touch-and-Go [20] 本来就有 vision+touch pair,但 Sparsh 只用 tactile modality 做 SSL。可以想象一个 Sparsh-V2 用 contrastive (tactile, vision) 学 cross-modal representation,这对 in-hand manipulation [56,57] 这种 visuo-tactile fusion task 可能更强。
4. **Slip detection 的 friction cone label** 有系统 bias(Figure 13)—— V-JEPA 的预测比 label 更对,这暗示 SSL 学到的 representation 比 human-designed label 更准。这开启一个有意思方向:**用 SSL representation 反过来 refine noisy ground truth labels**,类似 Noisy Student [82] 或 self-training。
5. **V-JEPA 性能 average 比 I-JEPA 低 24%** 但 slip detection 上 V-JEPA 最强 —— 暗示 temporal modeling 和 spatial abstraction 是 trade-off。一个可能的改进:V-JEPA + longer temporal context + smaller spatial mask。
6. **Force field viz (T1A)** 是 unsupervised 的 depth+flow analogy —— 这个 trick 很聪明,把 optical flow 的 photometric consistency loss 直接搬到 tactile shear estimation。但 paper 里只给 qualitative 结果(Figure 10),没 quantitative。理论上可以用 marker-based sensor(GelSight 2017)的 marker tracking 做 ground truth 验证,作者没做。
7. **Bead maze 的 diffusion policy + frozen encoder** 没 fine-tune 反而比 fine-tune 好(Table 11, DINO: frozen 10.80 > fine-tuned 8.45)—— 这违反 diffusion policy paper [71] 的 "frozen pretrained vision encoder hurts imitation learning" 结论。可能解释:touch 和 vision 的 domain gap 不同,vision pre-training 在 ImageNet 上和 robot hand 视觉差很远,touch pre-training 在真实 touch data 上和 downstream task 更近,所以 frozen 不损害。
8. **DINOv2 在某些 task 上反而差**(Force estimation DIGIT 1/100崩,Pose estimation 全面弱)—— DINOv2 的 iBOT patch-level MIM 可能 over-specialize 到 patch-level texture,对 regression task 反而 hurt。这呼应 LeCun 一直强调的 "JEPA > MIM for abstract reasoning"。

---

## 10. 对你的工作可能的相关联想

Andrej,你之前讲过 nanoGPT、背 LLM 的 self-supervised next-token prediction,以及多次提到 "SSL 的下一个前沿是 JEPA / world model"。这篇 paper 在 touch domain 验证了你和 LeCun 长期主张的几个观点:

1. **Latent space SSL > pixel space SSL** for tasks requiring abstraction(force, pose, slip) —— 这是 I-JEPA / V-JEPA / DINO vs MAE 的对照;
2. **Pre-trained frozen encoder + light decoder** 在 low-data regime 碾压 E2E —— 这正是 LLM recipe 在 robotics modality 上的复现;
3. **Cross-sensor generalization via shared SSL objective** —— 类似 GPT 在多语言上 transfer,Sparsh 在 DIGIT/GelSight 之间 transfer;
4. **Temporal tokenization 是 key design choice** —— stride=5 对应 80ms 物理时间尺度,这类似 LLM tokenization 需要匹配 phoneme/word 物理尺度。

如果你想做 follow-up:
- **Sparsh + VLA**: 把 Sparsh 接到 OpenVLA 或 π0 这种 VLA 模型上,做 visuo-tactile-language policy;
- **Tactile world model**: 用 V-JEPA + action conditioning 做 touch-based world model,预测 next tactile state given action —— 这可能是 dexterous manipulation 的 "model-based RL" 路线;
- **Sparsh on digit skin**: 把 Sparsh 思路扩展到 BioTac 流体传感器或 electronic skin,需要重新设计 tokenization(不再是 image,而是 pressure array);
- **Cross-modal distillation**: 用 vision LLM (e.g. CLIP, SigLIP) distill 到 Sparsh,让 touch representation 有 language grounding —— 类似 [39] 的 Touch-Vision-Language dataset 思路。

---

## 11. 参考链接汇总

**Sparsh 本身:**
- Project page: https://sparsh-ssl.github.io/
- Paper (arXiv): https://arxiv.org/abs/2410.24090 (推测,你给的 markdown 没显式 arXiv ID,但作者+标题可定位)
- FAIR Meta tactile sensing: https://ai.meta.com/blog/tactile-sensing/

**SSL 基础:**
- MAE: https://arxiv.org/abs/2111.06377
- DINO: https://arxiv.org/abs/2104.14294
- DINOv2: https://arxiv.org/abs/2304.07193
- iBOT: https://arxiv.org/abs/2111.07832
- I-JEPA: https://arxiv.org/abs/2301.08243
- V-JEPA: https://arxiv.org/abs/2312.06692
- LeCun JEPA position paper: https://openreview.net/pdf?id=BZ5a1r-kVsf
- SSL Cookbook (Balestriero et al.): https://arxiv.org/abs/2304.12210

**Tactile sensors:**
- GelSight original: https://ieeexplore.ieee.org/document/8286649
- DIGIT: https://arxiv.org/abs/2005.14401
- GelSlim: https://ieeexplore.ieee.org/document/8593661

**Benchmark datasets:**
- YCB-Slide / MIDAS Touch: https://arxiv.org/abs/2212.00527
- Touch-and-Go: https://arxiv.org/abs/2211.12498
- ObjectFolder 2.0: https://arxiv.org/abs/2204.06849
- Feeling of Success: https://arxiv.org/abs/1710.05512
- Clothing Dataset: https://ieeexplore.ieee.org/document/8460653

**Concurrent / related:**
- T3: https://arxiv.org/abs/2406.13640
- UniT: https://arxiv.org/abs/2408.06481
- Binding Touch to Everything: https://arxiv.org/abs/2401.18084
- Tacchi: https://ieeexplore.ieee.org/document/10008932
- Diffusion Policy: https://arxiv.org/abs/2303.04137

**Robot manipulation with touch:**
- In-hand rotation (Qi et al.): https://arxiv.org/abs/2310.07691
- AnyRotate: https://arxiv.org/abs/2405.07391
- NeuralFeels: https://arxiv.org/abs/2312.13472

---

希望这个 review 给你 build 了足够的 intuition。Sparsh 本质上是把 "vision SSL recipe book" 应用到 touch 这个 underexplored modality 的 careful engineering paper,它的核心贡献是验证了 **latent-space SSL 在 touch 上比 pixel-space SSL 显著更优**,并通过 TacBench 把这个领域从 "每个 lab 训自己的 encoder" 推向 "用 pre-trained backbone + light decoder" 的范式 —— 这正是 LLM / vision foundation model 已经走过的路。
