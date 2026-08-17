---
source_pdf: Distill Any Depth.pdf
paper_sha256: 0edf0aacb4b3e88d1ed7d9f576109c81c4cca34fb7985aaa42422c7f88180f04
processed_at: '2026-08-03T22:33:05-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Distill Any Depth

咱们把这篇paper里那些花里胡哨的公式和图表先放一边，用最直白的大白话捋一捋这帮人到底在干嘛，以及他们为啥要这么干。

---

## 1. 这篇paper到底在解决啥痛点？

核心任务叫 **Monocular Depth Estimation (MDE)**，也就是给一张普通的2D RGB照片，让AI猜出里面每个东西离镜头有多远，输出一张深度图。

现在搞这个任务最猛的玩法叫 **Pseudo-label Distillation**（伪标签蒸馏）。
你手头有海量没标注的普通照片，怎么办呢？你先找个已经训练好的“老师模型”（比如DepthAnything v2），让它去看这些照片，凭它的经验猜出每张照片的深度，这就叫“伪标签”（pseudo-label）。然后你拿这些伪标签当成“标准答案”，去教一个“学生模型”。学生模型学着学着，就可能“青出于蓝而胜于蓝”，比老师还厉害。

但是，这里有个一直被大家忽略的坑。

以前大家教学生的时候，算误差有个习惯：要把老师和学生的预测结果做一个“全局归一化”（Global Normalization，也叫 SSI Loss）。
人话解释就是：算误差前，先把整张深度图的数值“拉伸”或者“平移”一下，让它们的尺度和位置差不多对齐。

这篇paper发现，这个“全局归一化”在蒸馏里是个巨大的坑。
为啥？因为“老师”也是会犯错的，它给的伪标签是有噪音的。
如果你算误差的时候，把全图的深度值混在一起算个平均值、方差啥的来做归一化，那如果照片边角或者某个区域老师猜得特别离谱，这种“错误”就会像墨水滴进清水里一样，把整张图的误差计算全给污染了。本来学生在某个局部学得挺好的，结果被其他区域的烂标签连累，搞得哪哪都不准。

---

## 2. 他们想出的第一招：换个算账的方式

既然“全局归一化”会传染错误，那到底该怎么算误差？他们试了四种方法：

1. **Global Norm**: 上面说的，整张图一起算，传染错误，不好。
2. **No Norm**: 啥归一化都不做，老师给啥原始数值，学生就死记硬背啥数值。这招在某些情况下出奇地好用，因为老师给的标签本来就是同一种风格，没必要再拉扯对齐。
3. **Local Norm**: 把图切成一小块一小块，每块自己内部算归一化。这能保住细节，但整张图的大结构容易散架。
4. **Hybrid Norm**: 把大结构和小细节结合起来，既看全图的宏观统计，也看切小块的微观统计。

他们得出的结论是：看情况。如果老师和学生看的是同一块局部区域，用 Hybrid Norm 甚至 No Norm 最好，因为这时候俩人处于同一个频道，不需要强行对齐。如果老师看局部、学生看全局，那就必须用 Hybrid Norm，因为这时候两边的尺度不在一个频道，不对齐没法比。

---

## 3. 他们想出的第二招：Cross-Context Distillation（跨上下文蒸馏）

这是这篇paper最核心的创新点。

他们在想：怎么让学生既学到“抠细节”的能力，又学到“看大局”的能力？

他们发现了一个有意思的现象。如果把照片的一小块裁下来单独喂给老师模型，老师给出的这一小块深度图细节特别丰富、特别锐利；但如果把整张照片喂给老师，它给的整体结构很合理，但细节就糊了。

所以，**Cross-Context Distillation** 就设计了两种套路来教学生：

**套路一：Shared-Context Distillation（共享上下文蒸馏）**
随机裁剪一张照片的某个小块，老师和学生**都看这个小块**。因为输入一样，学生可以直接对照老师的精细输出，学习怎么抠细节。

**套路二：Local-Global Distillation（局部-全局蒸馏）**
老师看的是几个裁剪下来的小块（细节多），而学生看的是**整张大图**。
然后，把学生预测的大图里对应那几个小块的区域抠出来，去跟老师的精细预测做对比。这等于逼着学生在“胸怀全局”的同时，还得在局部细节上跟老师的“特写镜头”看齐。

通过把这两招结合起来（$\mathcal{L}_{sc} + \lambda_1 \cdot \mathcal{L}_{lg}$），学生模型就做到了宏观和微观两手抓，两手都要硬。

---

## 4. 他们想出的第三招：Assistant-Guided Distillation（助教引导蒸馏）

光有一个老师教，学生容易学“偏”了。比如你的主老师是 DepthAnything v2 (DAv2)，它是个典型的 encoder-decoder 模型，算得快、大局观好，但细节就是糊。

于是他们请了个“助教”——GenPercept。这助教是个基于 Diffusion model 的模型。Diffusion model 就像画画的一样，想象力丰富，抠细节特别猛，但它看大局容易跑偏，而且算得特别慢。

怎么让主老师和助教配合教学生呢？
他们试了两种法子：
1. **Avg (平均法)**: 把俩老师的预测加起来取个平均当答案。结果直接崩盘了。因为俩老师风格不一样，经常在一个地方一个说东一个说西，一平均就得出个完全错误的中间值。
2. **Select (随机抽取法)**: 训练的时候，按 7:3 的比例，这次迭代有 70% 的概率用主老师的答案，30% 的概率用助教的答案。这招效果奇好！

这就好比你大部分时间跟着严谨的教授学打基础，偶尔去上大师兄的课学点野路子和精细手艺。学生模型在这种多样化、不冲突的训练下，既学到了主老师的高效和大局观，又学到了助教的抠细节能力，还不容易被某一个人的偏见带偏。

---

## 5. 这套组合拳打下来效果如何？

他们拿这套方法去几个权威的测试集（像 NYUv2, KITTI, ETH3D 等）上跑分，结果非常漂亮。

举个最猛的例子：在极难的数据集 DIODE 上，以前 SOTA 的 DAv2 错误率是 0.262。用了他们的 Cross-Context 加上 MiDaS v3.1 老师蒸馏，错误率直接干到了 0.142，几乎提升了一倍！

看他们放的对比图也能明显看出来，他们方法出来的深度图，边缘更锐利，物体的细节更清楚，而且全局结构也没乱掉。甚至把一些 Diffusion model 老师那种精细的生成能力，成功“压缩”给了一个算得飞快的学生模型，做到了既有 Diffusion 的质量，又有 DAv2 的速度。

---

## 6. 总结一下这篇paper的intuition

这篇paper其实就讲了一个道理：在用AI教AI的时候，别迷信那些看似高大上的“标准化”流程。

1. **别全局归一化**: 伪标签本身就是有瑕疵的，你把它们全局混在一起算统计量，只会让瑕疵扩散。局部对局部，全局对全局，该不归一化就不归一化。
2. **Cross-Context 教学**: 让学生跟老师看一样的局部学抠细节，再让学生看全局跟老师的局部对齐学大局观。
3. **找俩风格互补的老师**: 一个教基础，一个教细节，而且别把俩人的答案混在一起，今天听这个的，明天听那个的，学生反而学得更全面。

这种思想其实不仅适用于估深度，搞其他的稠密预测任务，比如估法向量、做分割，都可以照葫芦画瓢试试。

---

# Distill Any Depth 深度解析

这篇paper来自Westlake University AGI Lab和Zhejiang University of Technology，核心是在pseudo-label distillation范式下重新审视depth normalization策略，并提出Cross-Context Distillation + Assistant-Guided Distillation。下面我尽量把细节讲透，帮你build intuition。

paper链接: https://distill-any-depth-official.github.io/

---

## 1. 核心动机：为什么Global Normalization在distillation里是"有毒"的

### 1.1 SSI (Scale-Shift Invariant) Normalization的回顾

MiDaS [Ranftl et al., TPAMI 2020] 引入的SSI representation是为了解决跨dataset训练时depth的scale/shift歧义。其公式（paper的Eq.1）：

$$\tilde{d}_i^s = \mathcal{N}_{glo}(\mathbf{d}^s) = \frac{d_i^s - \mathrm{med}(\mathbf{d}^s)}{\frac{1}{M}\sum_{j=1}^{M}\left|d_j^s - \mathrm{med}(\mathbf{d}^s)\right|}$$

变量含义：
- $d_i^s$: student model在pixel $i$处的raw depth prediction
- $\mathbf{d}^s = \{d_1^s, d_2^s, \ldots, d_M^s\}$: 整张depth map
- $\mathrm{med}(\mathbf{d}^s)$: 整张depth map的median (用median而非mean是为了robustness against outliers)
- 分母 $\frac{1}{M}\sum_{j=1}^{M}|d_j^s - \mathrm{med}(\mathbf{d}^s)|$: Mean Absolute Deviation (MAD)，衡量depth分布的spread
- $M$: valid pixels总数
- 上标 $s$ 表示student, $t$ 表示teacher，下标 $i, j$ 表示pixel index

为什么用median+MAD而不用mean+std？因为depth分布往往右偏（远处物体少但depth值大），median和MAD对outliers更鲁棒。这也是MiDaS原paper的设计哲学。

### 1.2 SSI在distillation中的"耦合灾难"

关键观察：当用SSI loss做distillation时

$$\mathcal{L}_{Dis} = \frac{1}{M}\sum_{i=1}^{M}|\tilde{d}_i^s - \tilde{d}_i^t|$$

每个pixel $i$的loss不仅取决于teacher在该pixel的预测 $d_i^t$，还取决于整张图所有其他pixel的 $d_j^t$（通过 $\mathrm{med}(\mathbf{d}^t)$ 和MAD耦合进来）。

**这是问题的根源**：pseudo-label本身就带noise（teacher不是完美的），如果某个区域（比如天空、远处背景）teacher预测得离谱，会污染整张图的median和MAD，导致原本预测准确区域的归一化值也被扭曲。

paper的Fig.2用一个简洁实验验证这点：
- 取一张图的中心 $w/2 \times h/2$ region
- 两种对齐策略：(1) Global Least-Square——全图先对齐再crop中心；(2) Local Least-Square——先crop中心区域再对齐
- 结果：Local策略更好，说明global normalization"拖累"了local accuracy

**Intuition**: SSI归一化是为了让不同sensor捕捉的depth（绝对scale不同）能放在同一loss下训练。但在pseudo-label distillation里，teacher和student输出的depth本来就在同一个domain（都是同一个teacher的输出空间），强行global normalize反而引入了不必要的inter-pixel coupling，把noise放大。

### 1.3 这跟有ground truth的训练有何不同？

在有GT的训练里，不同dataset的depth来自不同sensor（LiDAR、structured light、stereo等），数值范围、scale、shift都不同，必须归一化才能混训。但distillation时，所有pseudo-label都来自同一个teacher，已经是homogeneous domain，归一化的"必要性"消失，反而引入artifacts。

---

## 2. 四种Normalization策略的对比

paper系统地比较了4种策略（Fig.4）：

### 2.1 Global Normalization (SSI)
就是上面的Eq.1-2。

### 2.2 Hybrid Normalization (来自HDN [Zhang et al., NeurIPS 2022])
将depth range分成 $S \in \{1, 2, 4\}$ 段，每个pixel在其所属的segment内做归一化：

$$\mathcal{L}_{Dis}^i = \frac{1}{|U_i|}\sum_{u \in U_i}\left|\mathcal{N}_u(d_i^s) - \mathcal{N}_u(d_i^t)\right|$$

变量含义：
- $U_i$: pixel $i$ 所属的所有context集合（hierarchical，pixel可能同时属于多个segment）
- $|U_i|$: pixel $i$ 所属context的数量
- $\mathcal{N}_u(\cdot)$: 在context $u$ 内做归一化（用该segment的median+MAD）

最终loss对所有pixel取平均（Eq.4）。HDN的hierarchical设计让pixel在多个granularity上被约束，既保留global structure又保留local detail。

### 2.3 Local Normalization
只用最细粒度的segment（$S=4$），每个pixel只在其最小的local context内归一化（Eq.5）。强调local detail但丢失global coherence。

### 2.4 No Normalization
直接用raw depth做L1 loss（Eq.6）：
$$\mathcal{L}_{Dis} = \frac{1}{M}\sum_{i=1}^{M}|d_i^s - d_i^t|$$

### 2.5 Table 1的关键发现

在Shared-Context Distillation下：
| Normalization | ETH3D AbsRel↓ | DIODE AbsRel↓ |
|---|---|---|
| Global Norm | 0.064 | 0.259 |
| **No Norm** | **0.057** | **0.239** |
| Local Norm | 0.070 | 0.245 |
| **Hybrid Norm** | **0.057** | **0.238** |

在Local-Global Distillation下：
| Normalization | ETH3D AbsRel↓ | DIODE AbsRel↓ |
|---|---|---|
| Global Norm | 0.065 | 0.239 |
| No Norm | 0.273 | 0.300 (崩了) |
| Local Norm | 0.076 | 0.244 |
| **Hybrid Norm** | **0.064** | **0.238** |

**关键insight**：
- Shared-Context下，Hybrid和No Norm并列最好——因为student和teacher输入一致，pseudo-label的domain高度homogeneous，不需要归一化
- Local-Global下，No Norm崩盘（0.273）——因为local crop和global image来自不同depth domain（不同尺度），必须归一化才能对齐
- Hybrid Norm在两种setting下都表现稳定——它的hierarchical设计是"最安全"的选择

---

## 3. Cross-Context Distillation：架构解析

### 3.1 动机：local detail vs global structure的trade-off

paper Fig.5展示了一个关键观察：
- Teacher model如果输入**整张图**：global structure好，但local detail糊
- Teacher model如果输入**local crop**：local detail锐利，但缺失全局context

这跟diffusion model和patch-based方法的差异有关。Diffusion model如Marigold在local patch上能生成精细depth，但拼成全图可能不一致；encoder-decoder model如DAv2 global structure好但细节平滑。

### 3.2 两个子策略

**Shared-Context Distillation (Eq.7)**：
$$\mathcal{L}_{sc} = \mathcal{L}_{Dis}(\mathbf{d}_{local}^s, \mathbf{d}_{local}^t)$$

teacher和student都接收**同一个随机crop的local patch**作为输入。这强制student在local scale上学习teacher的细节生成能力。crop size从644 pixels到image shortest side随机采样，保持1:1 aspect ratio，resize到560×560。

**Local-Global Distillation (Eq.8)**：
$$\mathcal{L}_{lg} = \frac{1}{N}\sum_{n=1}^{N}\mathcal{L}_{Dis}(\mathrm{Crop}(\mathbf{d}_{global}^s), \mathbf{d}_{local_n}^t)$$

变量含义：
- $\mathbf{d}_{local_n}^t$: teacher对第 $n$ 个local crop的预测
- $\mathbf{d}_{global}^s$: student对整张图的预测
- $\mathrm{Crop}(\cdot)$: 从student的global prediction中裁出对应第 $n$ 个crop的区域
- $N$: 采样patch总数

这里teacher接收local crop（细节丰富），student接收整张图（global），通过Crop操作让student的global prediction在local区域被teacher的fine-grained prediction监督。这是一种"非对称context"的distillation。

### 3.3 总loss (Eq.9)
$$\mathcal{L}_{total} = \mathcal{L}_{sc} + \lambda_1 \cdot \mathcal{L}_{lg} + \lambda_2 \cdot \mathcal{L}_{feat} + \lambda_3 \cdot \mathcal{L}_{grad}$$

- $\lambda_1 = 0.5$: Local-Global权重
- $\lambda_2 = 1.0$: feature alignment（来自DAv2的feature distillation）
- $\lambda_3 = 2.0$: gradient preservation（保留depth edges）

### 3.4 Table 2的ablation
| Shared-Ctx | Local-Global | ETH3D AbsRel↓ | DIODE AbsRel↓ |
|---|---|---|---|
| ✗ | ✗ | 0.075 | 0.270 |
| ✗ | ✓ | 0.064 (-14.6%) | 0.238 (-13.3%) |
| ✓ | ✗ | 0.058 (-22.6%) | 0.237 (-12.2%) |
| **✓** | **✓** | **0.056 (-25.3%)** | **0.232 (-14.1%)** |

两者结合最好，Shared-Context单独贡献更大（-22.6%），Local-Global叠加再提升一些。

---

## 4. Assistant-Guided Distillation

### 4.1 设计动机

单一teacher有系统性bias。DAv2作为primary teacher（encoder-decoder，高效、global consistent），但细节偏平滑；GenPercept作为assistant（diffusion-based，generative prior，detail丰富但stochastic）。

paper提出probabilistic sampling：以7:3的比例在每个iteration随机选择DAv2或GenPercept提供pseudo-label。

### 4.2 Avg vs Select（Table 4）

| Method | Strategy | ETH3D AbsRel↓ | DIODE AbsRel↓ |
|---|---|---|---|
| DAv2 only | - | 0.131 | 0.262 |
| GenPercept only | - | 0.096 | 0.226 |
| D+G | Avg. | 0.228 (崩了) | 0.371 (崩了) |
| **D+G** | **Select.** | **0.054** | 0.258 |

**Avg策略崩盘的intuition**：两个teacher在disagreement区域，averaging会把各自的error叠加起来。比如DAv2预测某区域是前景=1，GenPercept预测背景=0，平均得到0.5，比任何一个都错得更离谱。

**Select策略成功**：每个iteration只用一个teacher的"完整"prediction，保持pseudo-label的内部一致性，让student吸收两种teacher各自的strength，避免error reinforcement。

### 4.3 跟Noisy Student [Xie et al., CVPR 2020]的类比

这跟Noisy Student的思想类似：让student见到多样化的supervision signal，相当于data augmentation in supervision space，提升generalization。

---

## 5. 实验设置细节

### 5.1 数据
- 训练：SA-1B的200K子集（跟DAv2 protocol一致）
- 评估（5个zero-shot benchmark）：
  - NYUv2 (654 samples, indoor)
  - KITTI (697 samples, outdoor driving)
  - ETH3D (454 samples, indoor+outdoor high-res)
  - ScanNet (1000 samples, indoor RGB-D)
  - DIODE (dense indoor+outdoor)

### 5.2 Metrics
- **AbsRel** (Eq.10): $\frac{1}{M}\sum_i \frac{|d_i - d_i^*|}{d_i^*}$ — 相对误差，对scale invariant
- **$\delta_1$ accuracy** (Eq.11): $\max(\frac{d_i}{d_i^*}, \frac{d_i^*}{d_i}) < 1.25$ — 在25% tolerance内的pixel比例

评估时先做least-square的scale+shift alignment（跟MiDaS一致）。

### 5.3 训练配置
- Student: DAv2-Large, 从pre-trained weights初始化
- Iterations: 20,000
- Batch size: 8
- GPU: 单卡V100
- Decoder LR: $5 \times 10^{-5}$

### 5.4 Cross-Architecture Distillation (Table 3)

| Teacher | Student | Method | DIODE | ETH3D |
|---|---|---|---|---|
| DA-L | DA-S | Base | 0.290 | 0.110 |
| DA-L | DA-S | **Ours** | **0.262 (-9.6%)** | **0.098 (-10.9%)** |
| DA-L | MiDaS-L | Base | 0.313 | 0.147 |
| DA-L | MiDaS-L | **Ours** | **0.295 (-5.7%)** | **0.126 (-14.3%)** |
| MiDaS-L | MiDaS-S | Base | 0.303 | 0.150 |
| MiDaS-L | MiDaS-S | **Ours** | **0.272 (-10.2%)** | **0.120 (-20.0%)** |

注意：cross-architecture也能work（DA-L蒸馏到MiDaS-L），说明method不依赖于teacher/student同构。

---

## 6. 跟SOTA对比 (Table 5)

paper最终结果（Ours*是DAv2-Large student + 全套方法）：

| Method | NYUv2 AbsRel | KITTI AbsRel | DIODE AbsRel | ScanNet AbsRel | ETH3D AbsRel |
|---|---|---|---|---|---|
| DepthAnything v2 | 0.045 | 0.074 | 0.262 | 0.042 | 0.131 |
| GenPercept | 0.058 | 0.080 | 0.226 | 0.063 | 0.096 |
| Marigold | 0.055 | 0.099 | 0.308 | 0.064 | 0.065 |
| MiDaS v3.1 | - | - | - | - | 0.061 |
| **Ours† (MiDaS)** | **0.046** | **0.063** | **0.142** | **0.049** | 0.057 |
| **Ours* (DAv2)** | 0.043 | 0.070 | 0.233 | 0.043 | **0.054** |

特别值得注意：
- **Ours†**（MiDaS v3.1 + Cross-Context）在DIODE上达到0.142，远超其他方法（次优DAv2是0.262）
- 在ETH3D上Ours*达到0.054，超越Marigold（0.065）这种diffusion-based方法
- 在NYUv2上达到0.043，超越DAv2（0.045）和GenPercept（0.058）

注意paper附录C.1提到一个trick：Table 5的Ours*只用DAv2的backbone，DPT head随机初始化——这是为了真正展示distillation方法的效果，而不是pre-trained head的贡献。如果全用DAv2 pre-trained weights，很难区分improvement来自distillation还是来自已有的strong prior。

---

## 7. 为什么这个方法work？Intuition总结

### 7.1 Decoupling noise propagation
Global normalization把每个pixel的loss耦合到整张图的statistics上。Pseudo-label的noise通过median和MAD"扩散"到所有pixel。去掉或局部化normalization，noise就被限制在local region内。

### 7.2 Multi-scale supervision如同multi-scale training
Shared-Context用random crop size (644到full size) 实际上做了scale augmentation。Local-Global让student同时见到local detail和global structure的supervision，类似multi-scale feature pyramid的效果，但作用在supervision层面。

### 7.3 Teacher diversity as regularization
Diffusion model和encoder-decoder model有inductive bias的差异：前者强generative prior、后者强discriminative structure。Probabilistic sampling让student在两种supervision distribution间切换，类似dropout在supervision space上的效果，提升generalization。

### 7.4 蒸馏diffusion model到efficient model
Appendix C.3的实验显示，用GenPercept（diffusion）作为teacher蒸馏到DPT-based student，能继承diffusion的detail richness，但摆脱diffusion的高推理成本。这是把generative prior"压缩"到discriminative model的有效路径。

---

## 8. 一些值得深究的细节

### 8.1 为什么Select (7:3)而不是其他比例？
paper没有做比例的ablation，这是个潜在的研究点。7:3暗示primary teacher仍应是主导，assistant只作"调味"。如果反过来（3:7），可能student过度吸收diffusion的stochasticity，破坏global consistency。

### 8.2 Hybrid Norm的 $S=4$ 是否最优？
paper没有详细ablate $S$ 的选择。HDN原paper建议 $S \in \{1, 2, 4\}$。在cross-context里，crop size已经变化，$S$ 的最优值可能依赖crop size distribution，这值得进一步研究。

### 8.3 Local-Global的计算成本
Teacher要对 $N$ 个local crops做forward，这比单次global forward贵 $N$ 倍。paper没明确 $N$ 的取值，但训练只用200K iterations × 8 batch on单卡V100，说明 $N$ 不大。这是个efficiency-accuracy trade-off。

### 8.4 跟Depth Anything V2的关系
DAv2作为teacher和student base。DAv2已经用62M+ unlabeled images训练，本身是strong baseline。这个工作相当于在DAv2基础上用更好的distillation protocol再提升一波——增量但不trivial。

---

## 9. 相关工作的Web Links

- **Distill Any Depth (本paper)**: https://distill-any-depth-official.github.io/
- **Depth Anything V2**: https://arxiv.org/abs/2406.09414
- **Depth Anything V1**: https://arxiv.org/abs/2401.10891
- **GenPercept**: https://arxiv.org/abs/2403.06090
- **Marigold**: https://arxiv.org/abs/2312.02161
- **MiDaS (原始)**: https://arxiv.org/abs/1907.01341
- **MiDaS v3.1**: https://arxiv.org/abs/2307.15060
- **HDN (Hierarchical Normalization)**: https://proceedings.neurips.cc/paper_files/paper/2022/hash/df0ad8804cae7c45e7f6c8d905f9d3a7-Abstract-Conference.html
- **DPT (Dense Prediction Transformer)**: https://arxiv.org/abs/2103.13413
- **SA-1B (Segment Anything)**: https://ai.facebook.com/datasets/segment-anything-downloads/
- **Metric3D**: https://arxiv.org/abs/2307.10984
- **MoGe**: https://arxiv.org/abs/2410.19115
- **Noisy Student**: https://arxiv.org/abs/1911.04252
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **LeReS**: https://arxiv.org/abs/2010.06987
- **Omnidata**: https://omnidata.vision/
- **PatchFusion**: https://arxiv.org/abs/2312.02184

---

## 10. 我的整体评价

这个工作的核心贡献是**重新审视一个被take-for-granted的设计选择（SSI normalization）**。在supervised learning里SSI是必须的（heterogeneous sensors），但在pseudo-label distillation里它变成了noise amplifier——这是个insightful的observation。

Cross-Context Distillation本质是"asymmetric context supervision"，让student通过teacher在不同context下的行为，学到scale-invariant的depth representation。Assistant-Guided Distillation用probabilistic sampling把multi-teacher ensemble的思想蒸馏到单student，避免了ensemble的推理成本。

实验结果impressive，特别是在DIODE上从0.262到0.142的跃升（MiDaS v3.1 + Cross-Context）说明这套方法对challenging outdoor scene的generalization提升显著。

潜在局限：
1. 没有跟Depth Pro [Apple 2024] 这种metric depth SOTA对比
2. Assistant的比例选择没有ablation
3. 训练成本（teacher forward次数）的quantitative analysis缺失
4. 在metric depth（绝对深度）任务上的扩展性未验证

不过作为relative depth的distillation framework，这篇paper做得很扎实，cross-context的设计思路值得在其他dense prediction task（如normal estimation、surface reconstruction）上借鉴。
