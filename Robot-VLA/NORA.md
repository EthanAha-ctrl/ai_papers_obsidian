---
source_pdf: NORA.pdf
paper_sha256: 0e0b5466b1cdea7ef69866522192b8c0757cd3f8eda5d604b01a7b960aadd0a8
processed_at: '2026-08-05T22:37:08-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 NORA

## 一、这帮人在干啥

robotic manipulation 这个圈子最近卷 VLA model 卷得很厉害。大家都在搞 7B、13B 甚至 55B 的大模型来控制机器人。听起来很酷，但实际deploy到真机器人的时候，7B模型跑起来卡得要命，控制频率上不去，机器人就抖。

NORA这帮人说：**咱别盲目堆参数了，3B就够，关键是怎么用好这3B。**

他们就搞了个3B的VLA，在WidowX真机实验上平均成功率56.7%，把7B的OpenVLA（40%）和加了3D模块的SpatialVLA（11.1%）都打趴下了。

项目主页：https://declare-lab.github.io/nora

---

## 二、为什么之前的VLA不够好

### 2.1 OpenVLA的问题

OpenVLA是Stanford + TRI + Berkeley那帮人搞的7B VLA model（paper: https://openreview.net/forum?id=ZMnD6QZAE6）。架构是：

```
LLaMA-2 7B (language backbone)
    + DINOv2 (vision encoder 1)
    + SigLIP (vision encoder 2)
    + 256-bin action tokenization (粗暴量化)
```

问题有几个：

**第一个问题：vision encoder不够强。** DINOv2 + SigLIP这个组合是2023年初的SOTA，但到现在Qwen-2.5-VL、InternVL-2.5这些新一代VLM已经甩开它们了。DINOv2是self-supervised的，对object scale和spatial relationship感知一般；SigLIP是contrastive learning的，更偏image-text alignment。

**第二个问题：action tokenizer太粗糙。** OpenVLA把每个action dimension独立quantize到256个bin里。比如7-DoF机械臂的action是 $[x, y, z, r, p, y, g]$（xyz位移+rpy旋转+gripper），它把每个dimension独立binning：

$$a_{\text{token}}[i] = \text{QuantileBin}(a[i], 256)$$

这相当于把action当成7个独立的"字符"来处理。但实际机械臂的joint之间是高度correlated的——你伸手的时候shoulder和elbow是协同转的。这种correlation信息全丢了，模型只能从头学。

**第三个问题：7B太大了。** Inference的时候单卡24GB的GPU勉强能跑，但real-time control需要几十Hz的推理频率，7B model就算用vLLM也很难上20Hz。

### 2.2 SpatialVLA的问题

SpatialVLA（paper: https://arxiv.org/abs/2501.15830）是腾讯PCG和上海交大搞的，加了Ego3D position encoding和adaptive action grids，想增强3D spatial understanding。

听起来很合理——机器人操作当然需要3D理解嘛。但实际WidowX真机上avg只有11.1%，惨不忍睹。原因paper §4.3说得很直白：

> "despite its ability to correctly determine spatial orientation, its performance in object grasping is worse"

翻译成人话：**它知道该往哪儿放，但它抓不住东西。**

这其实是robotics的经典教训：spatial understanding和affordance grounding是两回事。你知道"杯子在右边"和你能"抓住杯子"之间隔着十万八千里。SpatialVLA加了3D模块但没解决grasp problem，反而因为额外module引入了noise，affordance point估计更差了。

---

## 三、NORA怎么做的

NORA的核心思路其实很简单：**用更好的backbone + 更好的tokenizer，把3B model榨干。**

### 3.1 换backbone：Qwen-2.5-VL-3B

为啥选Qwen-2.5-VL？paper给的理由是"balance between performance and efficiency"，但我觉得真正的killer feature是 **native resolution training**。

Qwen2.5-VL的技术报告（https://arxiv.org/abs/2502.13923）里说，它在训练时就用了原始图像分辨率，没强制resize到224×224。这意味着：

- 模型见过物体的真实大小关系
- 对"远小近大"这种spatial cue理解更准
- Object localization能力更强

这对robotics太关键了。你想啊，机械臂要抓一个carrot，你得知道carrot有多大、在哪个位置、什么朝向。DINOv2+SigLIP这种把图像resize到224×224的encoder，物体大小信息全丢了，只能靠semantic feature猜。

NORA在spatial task上的表现（"move banana close to pan" 80% vs OpenVLA 50%）就是这个native resolution training的直接收益。

### 3.2 换tokenizer：FAST+

FAST+是Karl Pertsch搞的（paper: https://arxiv.org/abs/2501.09747），核心idea是：**action序列在时间维度和action维度都有强correlation，用信号处理方法去相关，再用BPE压缩。**

具体三步：

**Step 1: DCT跨action dimensions**

对每个timestep的action $a_t \in \mathbb{R}^d$（$d$ = DoF数量），做1D-DCT：

$$\hat{a}_t[k] = \sum_{i=0}^{d-1} a_t[i] \cos\left(\frac{\pi}{d}\left(i + \frac{1}{2}\right)k\right)$$

变量解释：
- $\hat{a}_t[k]$: 第$k$个frequency bin的DCT系数
- $a_t[i]$: 原始action第$i$个dimension的值
- $d$: action dimension总数（比如7-DoF就是7）
- $k$: frequency index, $k \in \{0, 1, \ldots, d-1\}$

Intuition: DCT把correlated的joint空间变换到frequency domain。机械臂的joint motion主要是协同的（低频），独立噪声是高频。DCT后大部分能量集中在低频，高频可以aggressive quantize。

**Step 2: Quantization**

把DCT系数量化到256 bins（quantile-based binning，和OpenVLA类似但在frequency domain）。

**Step 3: BPE跨timesteps**

把quantized DCT coefficients沿时间轴做BPE。相邻timesteps的action高度相关（机械臂motion是smooth的），BPE能把高频pattern合并成单个token。

**效果对比：**

| | OpenVLA | NORA (FAST+) |
|---|---|---|
| Tokenizer粒度 | character-level（每dim独立） | subword-level（DCT去相关+BPE压缩） |
| Vocab size | 256 | 2048 |
| Token sequence length | 长（每timestep每dim一个token） | 短（BPE压缩） |
| 收敛速度 | 慢 | 快 |
| Action correlation | 丢失 | 保留 |

用一句话总结：**FAST+之于action，就像SentencePiece之于text。**OpenVLA在用character-level tokenization，NORA在用subword tokenization。

### 3.3 NORA-LONG：action chunking variant

NORA-LONG是NORA的变体，action chunk size从1变成5。也就是预测未来5步action而非只预测下一步。

公式上，从：

$$r_t \sim \mathcal{M}_\theta(r \mid c, o_t)$$

变成：

$$r_{t:t+5} = [r_t, r_{t+1}, r_{t+2}, r_{t+3}, r_{t+4}] \sim \mathcal{M}_\theta(r \mid c, o_t)$$

为啥要chunking？因为ACT（Zhao et al., 2023, https://arxiv.org/abs/2304.13705）和Diffusion Policy（Chi et al., 2024, https://arxiv.org/abs/2303.04137）都证明multi-step prediction能减少compounding error，让policy更smooth。

但有意思的是，NORA-LONG在真机上翻车了（见下文）。

---

## 四、实验结果的"人话"解读

### 4.1 Real-world WidowX结果

WidowX是一个比较便宜的单臂机器人（BridgeData用的那种），9个task，每个10次试验。结果如下：

| Task类型 | RT-1 (35B) | OpenVLA (7B) | SpatialVLA (7B) | NORA (3B) |
|----------|-----------|-------------|-----------------|-----------|
| Multiple objects (3 tasks avg) | 0% | 16.7% | 0% | 33.3% |
| OOD object (3 tasks avg) | 0.3% | 56.7% | 6.7% | 83.3% |
| Spatial (3 tasks avg) | 10% | 46.7% | 26.7% | 53.3% |
| **Overall avg** | **4.4%** | **40%** | **11.1%** | **56.7%** |

几个有意思的观察：

**1. RT-1几乎全军覆没（4.4%）**

RT-1是Google的35B transformer，训练数据主要是Google robot。WidowX的embodiment差太多，cross-embodiment transfer基本失效。这告诉我们：**scale不能弥补embodiment gap**，数据分布的match比参数量重要。

**2. SpatialVLA出乎意料地差（11.1%）**

它有3D encoding和adaptive action grids，听起来很fancy，但真机表现惨。原因是affordance estimation出了问题——它知道空间关系但抓不住物体。这印证了：**robotics的进步不靠加module，靠解决正确的问题。**

**3. NORA在OOD object上碾压（83.3% vs OpenVLA 56.7%）**

这归功于Qwen-2.5-VL更强的visual-semantic understanding。Qwen-2.5-VL在internet-scale image-text pretraining上见过大量object concept，OOD generalization能力远强于DINOv2+SigLIP。

**4. Multi-object task所有方法都差（avg 33.3%）**

"pick A, place A, pick B, place B"这种sequential task，所有VLA都挣扎。这是long-horizon planning的open problem，不是参数量能解决的。

### 4.2 LIBERO Simulation结果

LIBERO是Liu et al. 2023搞的benchmark（paper: https://arxiv.org/abs/2306.03310），分4个suite：Spatial、Object、Goal、Long。结果：

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| NORA (no AC) | 85.6 | 87.8 | 77.0 | 45.0 | 73.9 |
| NORA + AC | 85.6 | 89.4 | 80.0 | 63.0 | 79.5 |
| **NORA-Long + AC** | **92.2** | **95.4** | **89.4** | **74.6** | **87.9** |

关键insight：

**NORA-Long在LIBERO-Long上74.6%，比所有baseline高20%。** Action chunking对long-horizon task帮助巨大。

但奇怪的是：**NORA（无AC）在LIBERO-Long上只有45%，比OpenVLA的53.7%还低。** 这说明3B model在single-step prediction模式下，long-horizon planning能力不如7B。但加上AC后反超。

### 4.3 Action Chunking的矛盾

这是paper里最有意思的发现，必须详细讲。

**Simulation（LIBERO 20Hz）：NORA-Long最好，avg 87.9%**

**Real-world（WidowX，低频控制）：NORA-Long翻车**

具体翻车情况：
1. **执行全部5个action** → 机器人撞墙（accumulated action过大）
2. **只执行第一个action** → multi-object task只完成第一个object就停（0%成功率）
3. **Affordance estimation变差** → 倾向从2 o'clock方向抓，小物体抓不住

paper给出的假设：

> "Action chunking is more effective when operating at higher control frequencies."

证据：
- Diffusion Policy: 10Hz prediction → interpolate to 125Hz execution ✓
- OpenVLA-OFT+ ALOHA: 25Hz, action chunking有效 ✓
- LIBERO: 20Hz simulation, action chunking有效 ✓
- WidowX: 低频（推测<10Hz），action chunking失效 ✗

**Intuition:** Action chunking假设"当前观测足够预测未来N步"。这个假设在high-frequency control下成立（100ms内环境变化不大），但在low-frequency control下失效（500ms后环境可能因为robot自身运动而完全改变）。

数学化：

$$[a_t, a_{t+1}, \ldots, a_{t+N}] \approx \pi(o_t)$$

这个近似的误差取决于：
- $\Delta t$（control period）：越大误差越大
- $v_{\text{robot}}$（robot velocity）：越快误差越大
- $N$（chunk size）：越大累积误差越大
- Contact dynamics：越non-Markovian误差越大

WidowX的$\Delta t$大 + 接触抓取涉及contact dynamics + 控制频率低 = action chunking假设失效。LIBERO的$\Delta t$小 + 接触动力学被简化 = action chunking成立。

这其实暗示了一个deeper的insight：**action chunking本质上是个temporal smoothness prior**，它在quasi-static或者high-frequency regime下有效，但在dynamic或者low-frequency regime下会引入error。

### 4.4 NORA-LONG的奇怪affordance

paper §4.3有个很有趣的观察：

> "NORA-LONG estimates affordance points differently, consistently attempting to grip objects from the side — specifically around the 2 o'clock direction - whereas NORA tends to grip objects directly from above."

paper没深入解释，我推测是behavioral mode collapse。当chunk size=5时，模型要预测未来5步的联合分布：

$$\arg\max_\theta \sum_t \log p_\theta(a_{t:t+5} | o_t)$$

为了优化multi-step likelihood，模型倾向于选择low-variance action mode。"Side grasp"的future trajectory更可预测（不需要大幅re-orient），"top grasp"的future trajectory更复杂（需要lift+move）。所以模型为了likelihood牺牲了grasp quality。

这是multi-step prediction的固有trade-off：**越长horizon的prediction，模型越倾向于"safe but suboptimal"的behavior mode。**

---

## 五、Paper的honest reporting

NORA这篇paper有几个我觉得很赞的honest reporting点：

1. **承认NORA-LONG在真机上失败**——很多paper会cherry-pick对chunking有利的实验，但NORA如实报告了真机翻车
2. **承认multi-object task仍然困难**——avg 33.3%真的不高，paper没overclaim
3. **没回避distraction问题**——Fig. 6显示加distractor后性能大幅下降，paper没hide这个

这种honest reporting在当前VLA paper的"军备竞赛"氛围下挺难得的。

---

## 六、Paper没讲清楚的几个点

作为Karpathy你肯定也会注意到这些：

### 6.1 缺关键ablation

**没有NORA+naive binning vs NORA+FAST+的ablation**。我们不知道NORA的成功多少来自Qwen-2.5-VL，多少来自FAST+。如果这个ablation做出来FAST+贡献很小，那paper的核心claim就弱了。

### 6.2 Real-world实验规模小

9个task × 10 trials = 90次试验。成功率从40%到56.7%看起来差距大，但用binomial test算，n=90时这个差距的p-value其实不一定<0.05。paper没做statistical significance test。

### 6.3 没给inference speed

paper强调"real-time"和"reduced computational overhead"，但没给latency数字。8.3GB memory是给了，但tokens/sec、control frequency这些关键指标缺失。对一个主打efficiency的paper，这个缺失挺致命。

### 6.4 只在WidowX上验证

WidowX是比较简单的单臂robot。NORA能不能cross到Franka、UR5、ALOHA双臂、mobile manipulator？paper没验证。Open X-Embodiment训练数据包含多种robot，但generalization到没见过的embodiment是open question。

---

## 七、我自己的几个speculation

### 7.1 VLA的"scaling law"可能不一样

LLM的scaling law（Chinchilla等）告诉我们compute-optimal是参数量和数据量同步scale。但VLA的"数据"是trajectory，每条trajectory的information density远高于text。

NORA用3B model + 970k trajectories打爆7B model + 970k trajectories，可能暗示：**VLA的数据efficiency比LLM高，小模型+好backbone+好tokenizer就够。**

这可能改变整个VLA的scaling策略——不是堆参数，而是堆数据diversity和tokenizer efficiency。

### 7.2 Action representation比model size重要

OpenVLA用naive binning，NORA用FAST+（DCT+BPE）。这个差距可能比7B vs 3B的差距更大。

future direction: 用更聪明的action representation，比如：
- Manifold-aware tokenization（考虑joint limit和velocity limit）
- Learned tokenizer（用VQ-VAE学action codebook）
- Hierarchical tokenizer（coarse-to-fine action prediction）

### 7.3 Hierarchical policy可能是出路

NORA-LONG真机翻车的根因是flat policy做long-horizon prediction困难。natural solution是hierarchical policy：

```
High-level policy: 生成subgoal（如"pick carrot"）
    ↓
Low-level policy: 生成primitive action sequence
    ↓
Robot execution
```

这样high-level planner可以在秒级timescale reasoning，low-level controller可以在毫秒级timescale react。NORA-LONG的chunking其实是个poor man's hierarchical policy，但没显式分layer。

### 7.4 Distraction问题的根源

Fig. 6显示加distractor后性能暴跌。这是visual attention的问题——模型把attention分给了无关物体。

解决方向：
- 在pretraining时加grounding supervision（如"click on the grasp point"）
- 用attention sink或者contrastive attention让模型对instruction更sensitive
- Test-time用instruction-conditioned attention mask

---

## 八、一句话总结

NORA这篇paper告诉我们：**VLA不需要堆参数到7B+，3B + 好backbone（Qwen-2.5-VL的native resolution training）+ 好tokenizer（FAST+的DCT+BPE）就够了。** Action chunking在simulation里有效但在low-frequency真机上会翻车。Multi-object task和distraction robustness仍然是open problem。

Paper链接：
- NORA: https://declare-lab.github.io/nora
- arXiv版本: 可以在项目主页找到
- FAST+: https://arxiv.org/abs/2501.09747
- OpenVLA: https://openreview.net/forum?id=ZMnD6QZAE6
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- LIBERO: https://arxiv.org/abs/2306.03310
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT/ALOHA: https://arxiv.org/abs/2304.13705
- SpatialVLA: https://arxiv.org/abs/2501.15830
- TraceVLA: https://arxiv.org/abs/2412.10345
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA-OFT+: https://arxiv.org/abs/2502.19645

---

## 九、想听你的看法

Karpathy，作为vLLM和Eureka Labs的founder，你肯定对VLA的deployment efficiency有很多想法。几个问题想问你：

1. **3B是不是VLA的sweet spot？** 还是随着tokenizer和data改进，1B甚至500M也能work？我倾向于觉得action space的complexity决定了下限——7-DoF arm + gripper可能300M就够，bimanual + dexterous可能需要3B+。

2. **FAST+的DCT在manifold上valid吗？** 机械臂的joint space是torus（joint angle有limit），DCT在torus上不是最优transform。如果用manifold-aware的tokenization（比如学习chart上的DCT），效果会更好吗？

3. **Action chunking真机翻车这件事，是不是暗示VLA需要类似MPC的receding horizon control？** 每次执行chunk的第一个action，然后用新observation重新predict。这其实就是model predictive control的思想，但VLA社区好像没systematically探索。

4. **你最近讲的"system 1 vs system 2"thinking能不能应用到VLA？** System 1是fast reactive control（single-step prediction），system 2是slow planning（multi-step chunking）。NORA和NORA-Long的对比其实就是system 1 vs system 2，结果发现system 2在simulation work但真机fail。是不是需要一个meta-controller来动态切换？

5. **你以前讲过"software 2.0"和"software 3.0"的概念，VLA是不是software 3.0的embodied version？** VLA本质是用natural language instruction programming robot behavior，这就是software 3.0的core idea——用prompt engineering来program。NORA这种small efficient VLA可能让software 3.0真的在robotics落地。

期待你的thoughts！

---

# NORA:3B 参数的开源通用 VLA 模型深度解析

## 一、Paper 整体定位

NORA (Neural Orchestrator for Robotic Autonomy) 是一个 3B 参数的 Vision-Language-Action model，核心目标是在保持 task performance 的前提下，把 VLA 模型的计算开销从 7B+ 降到 3B 级别，使其能在 consumer-grade GPU 上进行 fine-tune 和 real-time deployment。这个工作由 Singapore University of Technology and Design 和 Lambda Labs 联合完成。

Paper 的核心 thesis 可以概括为：**"smaller VLM backbone (Qwen-2.5-VL-3B) + efficient action tokenizer (FAST+) > larger VLM backbone (LLaMA-2 7B) + naive action binning"**，这个结论在 real-world WidowX 任务上 56.7% vs 40% 的成功率差距中得到了验证。

参考链接：
- 项目主页: https://declare-lab.github.io/nora
- FAST tokenizer paper: https://arxiv.org/abs/2501.09747
- OpenVLA paper: https://openreview.net/forum?id=ZMnD6QZAE6
- Qwen2.5-VL technical report: https://arxiv.org/abs/2502.13923

---

## 二、Architecture 深度解析

### 2.1 整体架构

NORA 的架构本质上是一个标准的 autoregressive next-token prediction 框架，关键在于三个组件的协同：

```
Input: X_t = [o_t, c] = [[I_t^1, ..., I_t^n], c]
            ↓
   Qwen-2.5-VL-3B (M_θ)  ← backbone VLM
            ↓
   r_{t:t+N} ~ M_θ(r | c, o_t)  ← autoregressive token generation
            ↓
   FAST+_decode(r_{t:t+N}) → a_{t:t+N}  ← action chunk
```

公式 (1) 和 (2) 看起来简单，但背后有几个关键设计决策：

**变量说明：**
- $o_t = [I_t^1, \ldots, I_t^n]$: time $t$ 时刻的 visual observation，由 $n$ 帧图像组成。NORA 中 $n=1$（单帧）
- $c$: natural language task instruction（如 "put the carrot in pot"）
- $X_t = [o_t, c]$: 拼接后的 multimodal input
- $a_{t:t+N} = [a_t, \ldots, a_{t+N}]$: 从 time $t$ 到 $t+N$ 的 action chunk
- $r_{t:t+N} = [r_t, \ldots, r_{t+N}]$: FAST+ tokenize 后的 discrete action token sequence
- $\mathcal{M}_\theta$: 参数为 $\theta$ 的 VLM (Qwen-2.5-VL-3B)

### 2.2 为什么选 Qwen-2.5-VL-3B？

这里 paper 给出的理由是 "performance and efficiency balance"，但我觉得更深层的原因是 Qwen-2.5-VL 的 **native image resolution training**。Paper §2.1 明确提到：

> Qwen2.5-VL uses native image resolution during training, which aims to enhance the model's perception of real-world scale and spatial relationships.

这个特性对 robotic manipulation 至关重要。OpenVLA 用的是 DINOv2 + SigLIP 双 encoder，把所有图像 resize 到 224×224，这种 fixed resolution 会丢失物体尺度的绝对信息。而 Qwen-2.5-VL 在训练时就见过各种 resolution，对物体大小、距离的感知更准确——这直接解释了 NORA 在 spatial reasoning task（如 "move the banana close to the pan"）上的优势（80% vs OpenVLA 50%）。

**架构对比：**

| Component | OpenVLA (7B) | NORA (3B) |
|-----------|-------------|-----------|
| Language backbone | LLaMA-2 7B | Qwen-2.5-VL 3B |
| Vision encoder | DINOv2 + SigLIP (dual) | Qwen-ViT (native resolution) |
| Action tokenizer | 256-bin quantile binning | FAST+ (DCT + BPE) |
| Action vocab size | 256 (overwriting least-used tokens) | 2048 (new tokens added) |
| Action chunk size | 1 | 1 (NORA) / 5 (NORA-LONG) |

### 2.3 FAST+ Tokenizer 的数学原理

这是 NORA 相比 OpenVLA 最关键的改进之一。FAST+ (Pertsch et al., 2025) 的核心思想是：**连续 action 序列在时间维度和 action dimension 上都存在强相关性，可以用信号处理方法去相关后再用 BPE 压缩**。

**Step 1: Discrete Cosine Transform (DCT) 跨 action dimensions**

假设一个 action $a_t \in \mathbb{R}^d$（$d$ 是 DoF，比如 7-DoF 机械臂），对每个 timestep $t$，把 $d$ 维 action 做 1D-DCT：

$$\hat{a}_t[k] = \sum_{i=0}^{d-1} a_t[i] \cos\left(\frac{\pi}{d}(i + \frac{1}{2})k\right)$$

其中：
- $\hat{a}_t[k]$: DCT coefficient 第 $k$ 个 frequency bin
- $a_t[i]$: 原始 action 第 $i$ 个 dimension
- $d$: action dimension 数量

DCT 的物理意义：把 correlated 的 joint 空间（比如 robotic arm 的 7 个 joint angle 之间通常耦合）变换到 frequency domain。低频系数捕获 joint 的协同运动模式，高频系数捕获独立噪声。由于机器人 action 主要是协同的，大部分能量集中在低频，高频系数可以做 aggressive quantization。

**Step 2: Quantization**

把 DCT coefficients 量化到 256 bins（和 OpenVLA 类似的 quantile-based binning，但在 frequency domain）。

**Step 3: Byte-Pair Encoding (BPE) 跨 timesteps**

把 quantized DCT coefficients 序列（沿时间轴）做 BPE。由于相邻 timesteps 的 action 高度相关，BPE 可以把高频出现的 pattern 合并成单个 token。

**效果：**
- Vocabulary: 新增 2048 个 action token（而非 OpenVLA 的 256）
- Sequence length: 大幅缩短（因为 BPE 压缩）
- 训练收敛更快
- 推理速度更快

**Intuition building:** 可以把 FAST+ 理解成 "action 世界的 SentencePiece"。正如 BPE 把 "the" "the" "the" 压缩成重复 pattern，FAST+ 把机械臂的 "抬手-伸手-抓取" 这种 correlated joint motion 压缩成 compact token。OpenVLA 的 naive binning 相当于 character-level tokenization，NORA 的 FAST+ 相当于 subword tokenization。

---

## 三、Pre-training 细节

### 3.1 数据

Open X-Embodiment (OXE) dataset，970k real-world robot demonstrations，包含：
- BridgeV2 (Walke et al., 2023): WidowX robot
- DROID (Khazatsky et al., 2024): 多种 robot platform
- 其他 OXE subsets

所有图像 resize 到 224×224（虽然 Qwen-2.5-VL 支持 native resolution，但训练时为了 batch efficiency 仍然 resize）。

### 3.2 训练超参数

| Hyperparameter | Value |
|----------------|-------|
| GPU | 8×H100 (1 node) |
| Total GPU hours | ~4000 H100-hours |
| Training duration | ~3 weeks |
| Batch size | 256 |
| Optimizer | AdamW (Loshchilov & Hutter, 2017) |
| Total steps | 1.1M |
| Peak learning rate | $5 \times 10^{-5}$ |
| Warmup | Linear, first 50k steps |
| LR schedule | Linear warmup → cosine decay to 0 |
| Precision | bf16 |
| Attention | FlashAttention |

**AdamW 的 weight decay 公式：**

$$\theta_{t+1} = \theta_t - \eta \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t \right)$$

其中：
- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$: first moment estimate
- $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$: second moment estimate
- $\lambda$: weight decay coefficient
- $\eta$: learning rate

**Cosine decay schedule：**

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \pi\right)\right)$$

变量：$T_{\text{warmup}}=50\text{k}$, $T_{\text{total}}=1.1\text{M}$, $\eta_{\max}=5\times10^{-5}$, $\eta_{\min}=0$。

### 3.3 训练动态

Paper Fig. 2 显示 training loss 平稳下降，gradient norm 偶尔有 spike 但不影响整体收敛。这个现象在 LLM training 中很常见，通常 spike 对应数据中某些 outlier batch（比如特别长或特别难的 trajectory）。bf16 训练 + FlashAttention 的组合在 3B 规模上是合理的，不需要更激进的 optimization（如 8-bit optimizer 或 ZeRO-3）。

**Inference memory: ~8.3GB GPU memory** —— 这个数字意味着 NORA 可以在 RTX 4090 (24GB) 甚至 RTX 3090 (24GB) 上实时部署，而 OpenVLA 7B 通常需要 16GB+ 甚至量化才能跑。

---

## 四、实验结果深度分析

### 4.1 Real-world WidowX 实验结果（Table 1）

| Category | Task | RT-1 | OpenVLA | SpatialVLA | NORA |
|----------|------|------|---------|------------|------|
| Multiple objects | Put red bottle and hamburger in pan | 0 | 20 | 0 | **40** |
| Multiple objects | Put carrot and hotdog in pot | 0 | 0 | 0 | **30** |
| Multiple objects | Put corn and carrot in pan | 0 | 30 | 0 | **30** |
| OOD object | Put carrot in pot | 0 | 80 | 20 | **90** |
| OOD object | Put banana in pot | 1 | 40 | 0 | **90** |
| OOD object | Put blue cube on plate | 0 | 50 | 0 | **70** |
| Spatial | Put pink toy at right corner | 0 | 60 | 30 | **60** |
| Spatial | Put blue cube on right plate | 0 | 30 | 0 | **20** |
| Spatial | Move banana close to pan | 30 | 50 | 50 | **80** |
| **Average** | | **4.4** | **40** | **11.1** | **56.7** |

**关键观察：**

1. **RT-1 几乎全军覆没**（avg 4.4%）：RT-1 用的是专有大规模数据 + 35B 参数的 efficient transformer，但在 WidowX 上完全失效。这说明 WidowX 的 embodiment 和 RT-1 训练数据（主要是 Google robot）差距太大，cross-embodiment transfer 仍然困难。

2. **SpatialVLA 表现意外差**（avg 11.1%）：虽然 SpatialVLA 有 Ego3D position encoding 和 adaptive action grids，但在 WidowX real-world 上完全打不过 OpenVLA。Paper §4.3 给出解释："despite its ability to correctly determine spatial orientation, its performance in object grasping is worse"。这印证了 robotics 的经典教训：**spatial understanding ≠ manipulation success**。你理解了 "把东西放右边"，但你抓不住东西，照样失败。

3. **NORA 在 OOD object 上最强**（avg 83.3%）：这归功于 Qwen-2.5-VL 在 internet-scale image-text pretraining 上见过的物体概念远多于 DINOv2+SigLIP 的组合。Qwen-2.5-VL 的 visual encoder 本身就是 LLM-friendly 的，semantic grounding 更强。

4. **Multi-object task 仍然困难**（avg 33.3%）：所有 method 都在这类 task 上挣扎。这反映了当前 VLA 的根本限制：**long-horizon, multi-step planning** 仍然是 unsolved problem。即使 NORA 能识别两个物体，但 "pick A, place A, pick B, place B" 的 sequential execution 仍容易在中间步骤失败。

### 4.2 LIBERO Simulation 结果（Table 2）

| Model | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|-------|----------------|---------------|-------------|-------------|---------|
| OpenVLA fine-tuned | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| TraceVLA fine-tuned | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| NORA fine-tuned | 85.6 | 87.8 | 77.0 | 45.0 | 73.9 |
| SpatialVLA fine-tuned-AC | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| NORA fine-tuned-AC | 85.6 | 89.4 | 80.0 | 63.0 | 79.5 |
| **NORA-Long fine-tuned** | **92.2** | **95.4** | **89.4** | **74.6** | **87.9** |

**关键观察：**

1. **NORA-Long 在 LIBERO-Long 上达到 74.6%**，比所有 baseline 高出 ~20%。这直接证明了 action chunking 对 long-horizon task 的重要性。LIBERO-Long 是 entangled task（10 个 long-horizon task），需要模型对 future trajectory 有 explicit planning。

2. **NORA (无 AC) 在 LIBERO-Long 上只有 45%**，比 OpenVLA 的 53.7% 还低。这说明：**3B 模型在 single-step prediction 模式下，long-horizon planning 能力不如 7B**。但加上 action chunking 后反超。

3. **Action chunking 是 "free lunch" 吗？不是**——看 §4.3 的 real-world 实验就知道。

### 4.3 Action Chunking 的 Real-world vs Simulation 矛盾

这是 paper 中最有 insight 的部分之一。

**Real-world WidowX 上 NORA-LONG 的问题：**
1. 执行全部 5 个 action → 机器人撞墙（accumulated action 过大）
2. 只执行第一个 action → multi-object task 只完成第一个 object 就停止（success rate 0%）
3. Affordance point estimation 变差 → 倾向从 2 o'clock 方向抓取，小物体抓不住

**Simulation LIBERO 上 NORA-Long 最好：** avg 87.9%

**Paper 给出的假设：**

> Action chunking is more effective when operating at higher control frequencies.

具体证据：
- Diffusion Policy: 10 Hz prediction, interpolate to 125 Hz execution
- OpenVLA-OFT+: 25 Hz ALOHA 任务，action chunking 有效
- LIBERO: 20 Hz simulation，action chunking 有效
- WidowX: 控制频率较低（推测 <10 Hz），action chunking 失效

**Intuition：** Action chunking 假设 "当前观测足够预测未来 N 步"。这个假设在 high-frequency control 下成立（因为 100ms 内环境变化不大），但在 low-frequency control 下失效（500ms 后环境可能因为 robot 自身运动而完全改变）。

**数学化解释：**

假设 policy 是 Markovian 的，$a_t = \pi(o_t)$。Action chunking 假设：
$$[a_t, a_{t+1}, \ldots, a_{t+N}] \approx \pi(o_t)$$

这个近似的误差取决于：
1. $\Delta t$（control period）：越大，环境变化越大，chunk 误差越大
2. Robot velocity：越快，与环境交互越剧烈，chunk 误差越大
3. $N$（chunk size）：越大，累积误差越大

WidowX 的 $\Delta t$ 较大 + 实际抓取涉及 contact dynamics（高度 non-Markovian），导致 action chunking 假设失效。LIBERO simulation 的 $\Delta t$ 较小 + 接触动力学被简化，action chunking 假设成立。

### 4.4 Distraction 实验

Paper §4.3 Fig. 6 显示：加入 distractor object 后，NORA 和 OpenVLA 性能都显著下降。这说明**当前 VLA 模型的 visual attention 仍然脆弱**，容易被 distractor 干扰。

这其实是一个 visual grounding 问题。Qwen-2.5-VL 虽然在 VQA benchmark 上表现好，但 VQA 的 grounding 通常是 "指认" 而非 "操作"。Real-world manipulation 需要的是 **pixel-level affordance grounding**，这比 VQA 的 bounding-box grounding 精度要求高得多。

---

## 五、NORA-LONG 与 Affordance Estimation 的有趣发现

Paper §4.3 报告了一个非常 interesting 的 observation：

> NORA-LONG estimates affordance points differently, consistently attempting to grip objects from the side — specifically around the 2 o'clock direction - whereas NORA tends to grip objects directly from above.

这个现象的原因 paper 没有深入分析，我推测是：

**Action chunking 改变了 training distribution 的 statistics。** 当 chunk size = 5 时，模型需要预测 future 5 步的联合分布。为了让 future action 更 "smooth"，模型可能倾向于 "side grasp"（这种 grasp 的 future trajectory 更可预测，因为不需要大幅 re-orient），而非 "top grasp"（这种 grasp 之后需要 lift + move，future action 分布更复杂）。

这其实是 **behavioral mode collapse** 的一种表现：模型为了优化 multi-step likelihood，收敛到一个 "更容易预测" 但 "执行效果差" 的 grasp strategy。

数学上，可以理解为：

$$\arg\max_\theta \sum_t \log p_\theta(a_{t:t+N} | o_t)$$

当 $N$ 增大，$p_\theta(a_{t:t+N} | o_t)$ 的 entropy 增大，模型倾向于选择 low-variance action mode（side grasp），牺牲 mode-fidelity 换取 likelihood。

---

## 六、Case Study 启示

Paper §4.4 的三个 case study 提供了 qualitative insight：

1. **OOD object (carrot in pot)**: NORA ✓, OpenVLA ✓, SpatialVLA ✗
   - SpatialVLA 失败原因：affordance point estimation 错误
   - Insight: SpatialVLA 的 3D encoding 并没有帮助 affordance，反而可能因为额外 module 引入了 noise

2. **Spatial reasoning (banana close to pan)**: NORA ✓, OpenVLA ✗, SpatialVLA unstable
   - OpenVLA 失败原因：能 grasp 但方向理解错误
   - Insight: OpenVLA 的 7B LLaMA 在 spatial reasoning 上不如 Qwen-2.5-VL 的 native resolution training

3. **Multi-object (red bottle + hamburger in pan)**: NORA ✓, others ✗
   - 其他方法失败原因：grasp suboptimal locations
   - Insight: Multi-object task 需要 sequential planning + 精确 affordance，NORA 的 combination 表现最好

---

## 七、Critical Analysis 与我的思考

### 7.1 Paper 的 strengths

1. **Reproducibility**: 完整 open-source (code + checkpoint)，3B 规模在 academic 环境可复现
2. **Honest reporting**: 诚实地报告了 NORA-LONG 在 real-world 上的失败，没有 cherry-pick
3. **Action chunking 的 cross-frequency analysis**: 提供了 action chunking 何时失效的实证证据
4. **Efficient design**: 证明 3B + FAST+ 可以 beat 7B + naive binning

### 7.2 Paper 的 weaknesses

1. **Real-world evaluation 规模有限**: 9 个 task，每个 10 trials，statistical significance 不强
2. **没有 ablation on FAST+ vs naive binning**: 没有直接对比 NORA 用 FAST+ vs 用 OpenVLA 的 256-bin 在相同 backbone 上的效果，无法 disentangle FAST+ 和 Qwen-2.5-VL 各自的贡献
3. **Distraction 实验不充分**: 只测试了 3 个 task + distractor，没有量化 distractor 的类型/数量对性能的影响
4. **WidowX only**: 没有在 Franka, UR5, ALOHA 等其他 platform 上验证 cross-embodiment 能力
5. **没有 inference speed 数据**: paper 强调 "real-time" 但没给出 latency 数字（tokens/sec, Hz）

### 7.3 与相关工作的关系

**NORA vs OpenVLA**: NORA 可以看作 OpenVLA 的 "efficient successor"，用更好的 backbone (Qwen vs LLaMA) + better tokenizer (FAST+ vs 256-bin) 来弥补参数减少的损失。

**NORA vs π0** (Black et al., 2024, https://arxiv.org/abs/2410.24164): π0 用 flow-matching 而非 autoregressive token prediction，可以生成 continuous action trajectory。π0 在 dexterous task (laundry folding) 上更强，但 NORA 在 simple manipulation + generalization 上更简单有效。

**NORA vs COT-VLA** (Zhao et al., 2025, https://arxiv.org/abs/2503.22020): COT-VLA 用 visual chain-of-thought 生成 future visual goals 再生成 action。NORA 没有这个 explicit reasoning，但 Qwen-2.5-VL 的 implicit reasoning 可能已经足够。

**NORA vs SpatialVLA** (Qu et al., 2025, https://arxiv.org/abs/2501.15830): SpatialVLA 加了 Ego3D encoding，但 real-world 表现反而差。这提示 **naive 3D inductive bias 不一定优于 2D + scale**。

### 7.4 Future directions（我自己的联想）

1. **Hybrid action representation**: 在 high-frequency phase 用 action chunking，在 contact-rich phase 用 single-step。可以用一个 meta-controller 来切换。

2. **Closed-loop replanning with chunking**: 执行 chunk 的第一个 action 后立即用新 observation 重新预测，而非执行全部 chunk。这是 receding horizon control (MPC) 的思想，可以解决 NORA-LONG real-world 失败问题。

3. **Visual grounding supervision**: 在 pretraining 时加入 affordance grounding loss（如 "click on the grasp point" VQA task），可能解决 distraction 问题。

4. **Test-time compute**: 用 best-of-N sampling + verifier（如 grasp success classifier）来 improve single-attempt success rate。

5. **Distillation from 7B to 3B**: 用 OpenVLA 7B 作为 teacher，NORA 3B 作为 student，做 behavior cloning distillation。可能比直接训练 3B 更好。

---

## 八、核心 Takeaways

1. **3B VLA 可以 beat 7B VLA**: NORA 证明了小模型 + 好 backbone + 好 tokenizer 的组合优于大模型 + 弱 backbone + 弱 tokenizer。
2. **Qwen-2.5-VL 的 native resolution training 对 robotics 有实质帮助**: 这是第一次有 paper 系统验证这个特性对 manipulation task 的影响。
3. **Action chunking 不是 universal improvement**: 它依赖于 control frequency 和 task dynamics，low-frequency + contact-rich 场景下会失效。
4. **Spatial understanding ≠ manipulation success**: SpatialVLA 的 3D encoding 没有转化为 real-world 优势，affordance grounding 才是关键。
5. **Multi-object task 仍是 open problem**: 所有 VLA 都在这类 task 上挣扎，需要更 explicit 的 sequential planning mechanism。

---

## 九、相关参考链接汇总

**Core papers:**
- NORA project: https://declare-lab.github.io/nora
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- OpenVLA: https://openreview.net/forum?id=ZMnD6QZAE6
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Open X-Embodiment: https://arxiv.org/abs/2310.08864

**Related VLA works:**
- RT-2: https://arxiv.org/abs/2307.15818
- RT-1: https://arxiv.org/abs/2212.06817
- SpatialVLA: https://arxiv.org/abs/2501.15830
- TraceVLA: https://arxiv.org/abs/2412.10345
- COT-VLA: https://arxiv.org/abs/2503.22020
- π0: https://arxiv.org/abs/2410.24164
- EMMA-X: https://arxiv.org/abs/2412.11974
- Octo: https://octo-models.github.io/

**Datasets:**
- BridgeV2: https://openreview.net/forum?id=f55MlAT1Lu
- DROID: https://arxiv.org/abs/2403.12945
- LIBERO: https://arxiv.org/abs/2306.03310

**Training infrastructure:**
- AdamW: https://arxiv.org/abs/1711.05101
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ALOHA / ACT: https://arxiv.org/abs/2304.13705

**Vision backbones:**
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP: (Zhai et al., 2023, ICCV)
- CLIP: https://arxiv.org/abs/2103.00020

---

## 十、对你 (Karpathy) 的个人提问

作为一个对 VLA 有 deep interest 的人，我很好奇你怎么看这几个点：

1. **FAST+ 的 DCT 假设了 action 是 LLM-friendly 的 "subword"**，但 mechanical constraint（joint limit, velocity limit）让 action space 实际上是 non-Euclidean manifold。DCT 在这种 manifold 上的 validity 如何？是否应该用 manifold-aware 的 tokenization（如 Riemannian DCT）？

2. **Qwen-2.5-VL 的 native resolution training 是否暗示了 future VLA 应该用 dynamic resolution + patch-based processing**（类似 ViT 的 variable patch size）而非 fixed 224×224？这能否解决 distraction 问题？

3. **NORA 在 real-world 上 action chunking 失败的现象**，是否暗示了我们应该用 **hierarchical policy**（high-level: plan, low-level: control）而非 single flat policy？这让我想到你之前讲过的 "system 1 vs system 2" 思考。

4. **3B 是否是 VLA 的 sweet spot**？还是说随着 tokenizer 和 training data 改进，1B 甚至 500M 的 VLA 也能 work？OpenVLA-OFT+ (https://arxiv.org/abs/2502.19645) 已经在探索 fine-tuning 的 efficiency，但 pretraining 的 minimum scale 还不清楚。

期待你的 thoughts！
