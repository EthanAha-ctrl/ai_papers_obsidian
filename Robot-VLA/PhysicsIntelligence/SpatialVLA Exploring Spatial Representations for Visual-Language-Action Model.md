---
source_pdf: SpatialVLA Exploring Spatial Representations for Visual-Language-Action
  Model.pdf
paper_sha256: d63e7346ca0f74282a9a575c05ea025318d40c72883874a2de1841a90d0e791e
processed_at: '2026-08-12T09:45:45-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SpatialVLA 大白话版

Andrej, 我用最接地气的方式再讲一遍。

---

## 一句话说清楚这paper干了啥

现在的VLA模型就是个**瞎子** — 它看着2D图片猜3D世界的动作，相当于闭着眼睛开盲盒。SpatialVLA给它配了副**3D眼镜**，还把动作输出从"7个旋钮"简化成"3个方向键"。

---

## 核心痛点在哪

想象你教一个机器人"把杯子放到左边的盘子上"。

**OpenVLA看到的**：一张RGB图片，杯子在某个pixel位置，盘子在另一个pixel位置。它得从2D pixel pattern反推"杯子离我多远""盘子比杯子高还是低"——这种推理很脆弱，换个背景灯光就懵了。

**SpatialVLA看到的**：同样的图片，但每个pixel还附带一个3D坐标(x,y,z)，直接告诉你"杯子在(0.3, -0.2, 0.5)米处""盘子在(0.4, -0.3, 0.3)米处"。模型不用猜空间关系，直接看几何就行。

这就好比**你开车有仪表盘 vs 没仪表盘**的区别。没仪表盘你也能开，但每次得凭感觉估速度和油量，累且容易出错。

---

## 两个核心创新，用类比讲

### 创新1: Ego3D Position Encoding — 给图片加"深度标签"

**传统做法**：图片进来 → SigLIP提取2D特征 → 喂给LLM → 输出action。整个过程没有3D信息。

**SpatialVLA做法**：
1. 图片同时喂给SigLIP（拿语义特征）和ZoeDepth（拿深度图）
2. 深度图通过camera内参反算出每个pixel的3D坐标
3. 3D坐标用sinusoidal encoding + MLP变成embedding
4. 直接**加到**2D语义特征上

**类比**：就像给照片每个pixel贴个标签，上面写着"我在相机前方0.5米、偏左0.2米、高度0.3米"。LLM看到这个标签就知道物体在3D空间哪里了。

**关键点**：用相机自己的坐标系（egocentric），不需要知道相机装在robot哪个位置。每个robot用自己的camera frame，大家各算各的，避免标定噩梦。

**为啥用ZoeDepth而不是真depth sensor**：因为OXE里28个dataset大部分压根没depth sensor，你只能用预测的。而且实验发现ZoeDepth的平滑深度反而比某些noisy的sensor depth效果更好（72.7% vs 70.5%），因为它捕捉的是**相对空间关系**，不需要绝对精度。

---

### 创新2: Adaptive Action Grids — 动作离散化的艺术

这是paper最精彩的部分。

**老办法的蠢**：RT-2和OpenVLA把每个action维度在[-1,1]区间切256个等距bins。但robot实际动作分布是这样的——中间密集，边缘稀疏，像根胖茄子。你用等距切割，90%的bins浪费在robot永远不会去的空间。

**SpatialVLA的聪明**：
1. 先统计整个dataset的动作分布，发现是Gaussian形状的
2. 用**等概率切割**代替等距切割——每个bin包含相同概率密度的数据
3. 中心区域（高频动作）分到很多细bins，边缘区域（低频动作）用少量粗bins

**类比**：就像人口普查。北京上海人口密集，你划很多小区；西藏人口稀疏，你划一个大区就行。没必要全国划等大小网格，那样西藏的网格里就住几户人家，浪费行政资源。

**更骚的操作——polar coordinates**：
把平移(x,y,z)转成(φ,θ,r)，即"方向+距离"。

为啥？因为"向左"和"向左10厘米"是两件事——方向是skill，距离是参数。disentangle后模型学起来更轻松。就像你教人开车，"往左打方向盘"和"往左打多少度"是分开学的概念。

**最终的token数**：
- 平移：16×32×8 = 4096个grid（方向分辨率高，距离粗一点）
- 旋转：16×16×16 = 4096个grid
- 夹爪：2个grid（开/关）
- 总共8194个token

**但每次只生成3个token**（平移index + 旋转index + 夹爪index），不是7个。OpenVLA要生成7个token，所以慢。SpatialVLA 3个token搞定，inference速度直接快一倍多。

---

## Post-training的小妙招

新robot来了，action分布不一样怎么办？

**传统**：直接fine-tune，从头学新action space。

**SpatialVLA**：
1. 在新dataset上重新fit一个Gaussian分布
2. 生成新的grid分割
3. 新grid的embedding用**三线性插值**从老grid初始化

**类比**：你有一张北京地图（pre-trained grid），现在要画上海地图（新robot action space）。不用重新测绘，而是根据上海的经纬度在北京地图上找最近的几个点，加权平均一下，作为上海地图的初始版本。然后再fine-tune微调。

这个trick在LIBERO上带来+4.6%到+5.4%的提升，在小数据集上效果尤其明显——因为小数据集本身分布跟pre-training差得远，需要这种"知识迁移"。

---

## 实验结果，挑重点说

### Zero-shot效果

| 场景 | 对比 | 结果 |
|------|------|------|
| SimplerEnv Google Robot | vs RT-2-X (55B) | 3.5B的SpatialVLA反超 |
| SimplerEnv WidowX | vs RoboVLM | 42.7% vs 31.3% |
| Real WidowX 7个task | vs OpenVLA | 平均碾压，"放茄子进篮子"达100% |

**最有说服力的数字**：3.5B参数超越55B的RT-2-X。这说明**架构设计比堆参数重要**，3D空间表征的inductive bias比暴力scaling更值钱。

### Ablation最关键的两个发现

1. **Adaptive grid vs uniform linear**: +40%绝对提升。等概率切割完爆等距切割，这是information-theoretically更优的分法。

2. **Ego3D PE的贡献**: 在variant aggregation（不同visual condition）上+12.7%。3D信息让模型对光照、纹理、相机角度变化更鲁棒，因为它知道"物体在3D空间的位置没变，只是看起来不一样了"。

---

## 这paper的局限，直说

1. **Gaussian假设太天真**：如果robot只在单轴运动，Gaussian fit出来的grid会在其他轴collapse。Paper自己也承认这点，说未来可能用VAE或者mixture model。

2. **Long-horizon拉胯**：LIBERO-Long只有55.5%，是四个suite里最低的。单帧观测+4-action chunking搞不定需要planning的长序列任务。这部分需要memory、world model这些更重的架构。

3. **ZoeDepth是过渡方案**：真正的3D foundation model（DUSt3R、VGGT、Metric3D）出来后，ZoeDepth会被取代。现在用它是因为OXE没depth sensor，属于"有什么用什么"。

4. **没跟π0公平比较**：π0用flow matching + action expert，是完全不同的action生成paradigm。Paper只在部分benchmark上对比了π0，不够apples-to-apples。

5. **3 token的capacity有天花板**：8194个grid对单臂7D够用，但bimanual（14D）或者humanoid（30+ DoF）就捉襟见肘了。Paper说future work会做grid sharing，但这是non-trivial的设计挑战。

---

## 我的判断

这篇paper的价值在于**两点 insight**：

第一，**observation端的3D注入**是正确方向。不管具体用什么depth estimator，"给VLM 3D坐标"这个idea会活下来。未来的VLA model大概率都会加某种形式的3D position encoding，就像现在的VLM都会加position embedding一样理所当然。

第二，**action端的adaptive discretization**比uniform bins聪明太多。等概率切割这个idea本身不新（信息论教科书里就有），但把它用在这个场景、配合polar coordinates disentanglement、再加trilinear interpolation做transfer——这套组合拳是新的且work的。

**但我对autoregressive + discrete token这条路线的长期竞争力存疑**。π0的flow matching路线在continuous action space上更natural，而且inference可以batch sampling。SpatialVLA证明了discrete tokens能competitive，但"competitive"和"dominant"是两回事。最终哪种paradigm赢，还得看更多实验和scaling的验证。

一句话总结：**SpatialVLA是个工程上很solid、idea上很intuitive的工作，它把"3D空间感知"这个gap填上了，但离最终的generalist robot policy还有距离——long-horizon、更复杂的embodiment、更高质量的data，这些都是下一段要解决的硬骨头。**

---

# SpatialVLA 深度解析 — 用3D空间表征武装VLA Foundation Model

Andrej, 这篇paper切中了一个非常根本的问题: 当前主流的VLA models (RT-2, OpenVLA, π0) 本质上仍然是"2D creatures" — 它们通过2D pixel patches看到世界, 然后直接regress出7D action vector, 中间完全没有显式的3D空间表征。SpatialVLA的核心主张是: **空间感知能力是机器人操作的keypoint**, 必须同时从observation和action两端注入3D结构。下面我从架构、公式、训练机制和实验四个层面来build你的intuition。

---

## 一、核心问题与设计哲学

现有的VLA models存在两个misalignment:

1. **Observation heterogeneity**: 不同robot的camera安装位置各异(wrist/third-person), 导致3D observation space是非对齐的, 同一个像素位置在不同robot上对应不同的3D point
2. **Action heterogeneity**: 不同robot的DoF、controller、workspace都不同, 直接回归连续action难以跨robot迁移

SpatialVLA提出两条robot-agnostic的设计主线:
- **Ego3D Position Encoding**: 在camera坐标系下构建3D表征, 避免外参标定
- **Adaptive Action Grids**: 用基于数据分布的自适应离散化grid统一不同robot的action space

项目主页: [https://spatialvla.github.io](https://spatialvla.github.io)

---

## 二、Ego3D Position Encoding — 给VLM装上"深度眼睛"

### 2.1 核心pipeline

```
RGB image (224×224)
   ├── SigLIP encoder ──> 2D semantic features X ∈ R^(d×h×w)
   └── ZoeDepth ──> depth map D ──> back-projection π^-1 ──> 3D point P ∈ R^(3×h×w)
                                                       ↓
                                              sinusoidal γ(·) + MLP
                                                       ↓
                                              3D position embedding P'
                                                       ↓
                              O_3d = X + P'   (element-wise addition)
```

### 2.2 关键公式解析

**Back-projection** (paper中用π^-1表示):
对于每个pixel (u, v) with depth d, 在egocentric camera frame下的3D position:

$$
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = d \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

其中 K 是camera intrinsic matrix (3×3), (u, v) 是pixel coordinates, d 是ZoeDepth预测的深度值。这里 (x, y, z) 就是在相机自身坐标系下的3D坐标 — 关键点是: **不需要robot-camera extrinsic calibration**, 因为每个robot用自己的camera frame, 这就是"egocentric"的含义。

**Position encoding** (公式2):
$$
\mathbf{O_{3d}} = \mathbf{X} + \mathbf{P'} = \mathbf{X} + \text{MLP}(\gamma(\mathbf{P}))
$$

变量解释:
- **X ∈ R^(d×h×w)**: SigLIP提取的2D semantic features, d是embedding维度(2304 for PaliGemma2-3B), h×w是spatial resolution (16×16 for 224 input)
- **P ∈ R^(3×h×w)**: 每个pixel对应的3D position (x, y, z)
- **γ(·)**: sinusoidal positional encoding (类似NeRF的position encoding), 把3D坐标lift到高维频率空间
- **MLP**: learnable MLP, 把sinusoidal encoding映射到与X同维的embedding space
- **P'**: 最终的3D position embedding
- **O_3d**: 融合后的3D-aware visual token, 通过element-wise加法注入到semantic feature中

### 2.3 为什么用ZoeDepth而不是sensor depth?

这是个反直觉的设计选择。Paper在Appendix C Q3中专门解释:
- OXE的28个dataset中, 大部分**没有depth sensor**
- ZoeDepth提供smoother的相对深度, sensor depth通常noisy
- 实验对比 (Table IX): ZoeDepth 72.7% vs sensor depth 70.5% vs no depth 45.4%
- Ego3D PE捕获的是**relative spatial layout**, 不需要精确scale
- ZoeDepth只占8.6%参数, 每action耗时0.06s, 几乎零开销

ZoeDepth reference: [https://arxiv.org/abs/2302.12288](https://arxiv.org/abs/2302.12288)

### 2.4 Intuition

这个设计妙在: 2D semantic feature知道"这是什么", 3D position embedding知道"这在哪里"。当OpenVLA看到一张图片, 它无法分辨"杯子在桌上"和"杯子在地上"的区别 — 它只能通过pixel pattern间接推断。SpatialVLA直接把3D坐标喂给模型, 让transformer的attention机制能在3D空间中做reasoning, 这就是为什么它在LIBERO-Spatial上能达到88.2% success rate (vs OpenVLA 84.7%)。

---

## 三、Adaptive Action Grids — 重新思考action tokenization

### 3.1 传统做法的问题

RT-1, RT-2, OpenVLA都用**uniform linear discretization**: 把每个action dimension在[-1, 1]区间切成256个等距bins, 7D action对应7个token。问题:
- 实际action分布是高度non-uniform的 (Fig. 3a), 中心区域密集, 边缘稀疏
- 大量bins浪费在robot永远不会去的空间区域
- 7 tokens/action = 慢inference

### 3.2 SpatialVLA的action拆分

**Step 1: 7D action分解为3个语义组件** (公式3):
$$
\mathbf{a} = \{\mathbf{a}_{\text{trans}}, \mathbf{a}_{\text{rot}}, \mathbf{a}_{\text{grip}}\}
$$

- **a_trans = (x, y, z)**: translation ΔT, 进一步转成polar coordinates (φ, θ, r) — disentangle direction (φ, θ) from distance r
- **a_rot = (roll, pitch, yaw)**: rotation ΔR
- **a_grip**: 2个离散token (open/close)

**为什么用polar coordinates?** 在直角坐标系下, "向前移动10cm"和"向前移动1cm"对应完全不同的(x,y,z)组合, 但在polar下, direction (φ, θ)是共享的, 只有r不同。这让model能学到"方向是某种skill, 距离是另一种skill"的结构化知识。

### 3.3 Adaptive grid分割

对每个action variable, 先normalize到[-1, 1], 然后用Gaussian分布拟合整个pretraining dataset的action分布:

$$
\mathcal{N}(\mu^a, \Sigma^a)
$$

其中:
- **μ^a**: 该action variable在dataset上的均值向量
- **Σ^a**: 协方差矩阵 (paper实际用的是对角近似, 即每个variable独立fit Gaussian)

**等概率分割** (公式4):
$$
a_2, ..., a_M = \arg\min_{a_2, ..., a_M} \sum_{i=1}^{M-1} \left| \int_{a_i}^{a_{i+1}} f(x) dx - \frac{1}{M} \right|
$$

变量解释:
- **f(x)**: 拟合的Gaussian PDF
- **M**: 该variable上的grid数量
- **a_1 = -1, a_M = 1**: 边界固定
- **a_2, ..., a_{M-1}**: 优化变量, 使得每个bin [a_i, a_{i+1}] 包含的概率密度 ≈ 1/M

直观理解: 不是等距切分, 而是**等概率切分**。中心密集区域有更多细粒度bins, 边缘稀疏区域用少量粗bins覆盖。这是quantile-based discretization, 比uniform discretization在信息论上更高效。

**具体配置** (Appendix B):
- φ: 16 bins (范围[0, π])
- θ: 32 bins (范围[-π, π]) — 方向需要高分辨率
- r: 8 bins (范围[0, 3])
- roll, pitch, yaw: 各16 bins
- grip: 2 bins

总action token数:
$$
V = M_{\text{trans}} + M_{\text{rot}} + 2 = (16 \times 32 \times 8) + (16 \times 16 \times 16) + 2 = 4096 + 4096 + 2 = 8194
$$

### 3.4 单次action只需3个token

这是关键的inference加速:
- **RT-2/OpenVLA**: 每action生成7个token (每个dimension一个)
- **SpatialVLA**: 每action生成3个token (1 trans index + 1 rot index + 1 grip)

为什么是3而不是7? 因为 (φ, θ, r) 三个变量被linearized成一个combined index, 同理 (roll, pitch, yaw) 也是一个combined index。模型只需要预测3个离散token id, 然后通过gridification lookup恢复出6个连续值。

**Inference speed**: 20Hz on RTX 4090, 8.5GB memory — 这个数字对real robot deployment非常友好。

### 3.5 Action token embedding共享

Paper做了一个巧妙设计: 8194个action token的embedding**与LLM的text vocabulary embedding共享参数**。PaliGemma2有256k text vocabulary, action tokens被线性插入到这个embedding table中。这让action prediction可以无缝使用标准next-token prediction loss, 不需要额外的action head。

---

## 四、Pre-training与Post-training的两阶段设计

### 4.1 Pre-training

**Backbone**: PaliGemma2-3B ([arxiv.org/abs/2412.03555](https://arxiv.org/abs/2412.03555))
- SigLIP vision tower (27层)
- Gemma2 model (26层, 2304 hidden dim)
- Ego3D Position Embedding MLP
- Spatial Embedding (action tokens)

**Data**: 1.1M real robot episodes, 来自OXE + RH20T
- OXE: [https://robotics-transformer-x.github.io](https://robotics-transformer-x.github.io)
- RH20T: [https://rh20t.github.io](https://rh20t.github.io)

**Dataset mixture优化** (Appendix A, Table in paper):
- Bridge 15.34%, Fractal 14.71%, Droid 11.66%, BC-Z 8.64%
- RH20T新增5.67%
- **Down-weighting**: Kuka (缺乏clear prompts), FMB (导致robot右偏), Toto, Berkeley Fanuc

**关键trick**: 
- **Freeze text token embedding** E_text — 保留VLM的world knowledge
- **Train** action token embedding E_a, MLP, vision encoder, LLM backbone
- **Two-stage**: 160k steps全数据训练 → 移除DROID再训40k steps (Fig. 12显示这一步带来显著accuracy boost)

**Loss** (公式1):
$$
\mathcal{L}(\theta) = \mathbb{E}_{p(\mathbf{A}_t | \mathbf{o}_t)} \mathcal{L}(\mathfrak{a}_t, \tilde{\mathfrak{a}}_t)
$$

- **a_t**: ground-truth action token id
- **ã_t**: predicted token id via τ(O_3d, L)
- **L**: cross-entropy loss, 标准autoregressive next-token prediction

**Training cost**: 64×A100 GPU, 10 days, batch size 2048, AdamW lr=2e-5, linear scheduler, 0.005 warmup ratio, DeepSpeed ZeRO stage 1.

**Action chunking**: 预测T=4 future actions (12 tokens), 用ensemble执行 — 这借鉴了Diffusion Policy的思想, 减少distribution shift。

### 4.2 Post-training — Spatial Embedding Adaption

这是paper最有创新性的post-training设计。传统做法是full-parameter或LoRA fine-tuning, 但SpatialVLA提出: **重新离散化action grids**。

**核心想法**: 新robot setup的action分布不同于pre-training distribution, 直接用pre-trained的grid可能misalignment。在新dataset上重新fit Gaussian:

$$
\mathcal{N}(\mu_{\text{new}}, \Sigma_{\text{new}})
$$

然后生成新的grid G_new和token embeddings E_{a_new}。

**Trilinear interpolation初始化** (公式6):
$$
\mathbf{e}_{\mathfrak{a}_{\text{new}}}^i = \sum_{j=1}^{K} w_j \mathbf{e}^j
$$

变量解释:
- **e_{a_new}^i**: 新grid第i个token的embedding
- **{e^j}_{j=1}^K**: pre-trained grid中与第i个新grid**相邻**的K个token embeddings (在3D空间中最近的邻居)
- **w_j**: 基于距离的归一化权重 (类似trilinear interpolation的weights)
- **centroid (φ_new^i, θ_new^i, r_new^i)**: 新grid第i个cell的3D中心坐标

直观理解: 新grid的中心点落在pre-trained grid的某个3D cell内或附近, 用周围8个corners的embedding做加权平均作为初始化。这相当于把pre-trained的spatial action knowledge"插值"到新robot的action space, 而不是从头学。

**实验验证** (Table V):
- 大dataset (Fractal/BridgeV2): +2.9% (大dataset本身分布相近, gain有限)
- 小dataset (LIBERO): +4.6% to +5.4% across all 4 task suites
- LoRA + Spatial Embedding Adaption是最佳组合

---

## 五、实验结果全景

### 5.1 Zero-shot SimplerEnv (Google Robot)

Table I关键数字:

| Model | Visual Matching | Variant Aggregation |
|-------|----------------|---------------------|
| RT-2-X (55B) | 60.7% | 64.3% |
| RoboVLM | 56.3% | 46.3% |
| π0 (BF16) | 70.1% | - |
| **SpatialVLA zero-shot** | **71.9%** | **68.8%** |
| **SpatialVLA fine-tuning** | **75.1%** | **70.7%** |

关键insight: 3.5B的SpatialVLA超越了55B的RT-2-X, 这说明**架构设计比scale更重要** — 3D空间表征带来的inductive bias极其valuable。

### 5.2 LIBERO benchmark

Table III:

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| OpenVLA | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |
| TraceVLA | 84.6% | 85.2% | 75.1% | 54.1% | 74.8% |
| **SpatialVLA** | **88.2%** | **89.9%** | 78.6% | **55.5%** | **78.1%** |

LIBERO-Spatial +3.5%提升最明显 — 直接验证了Ego3D PE的价值, 因为这个suite专门测试spatial layout变化。

### 5.3 Real WidowX Zero-shot

Table II / XI关键结果:
- "Put Eggplant in Yellow Basket": fine-tuning达到**100% success** (vs RoboVLM 58.3%)
- "Put Green Cup on Pink Cloth": 81.81% grasp + 81.81% success (vs OpenVLA 36.36%)
- 7个task平均42.7% (zero-shot fine-tuning)

### 5.4 Franka multi-task fine-tuning

Fig. 6 + Table XIII:
- Single task: SpatialVLA 82% vs Diffusion Policy 81% (持平)
- Instruction following: SpatialVLA +12% over OpenVLA, Diffusion Policy只有26% (崩溃)
- Multi-task: SpatialVLA 57%, 显著超越其他generalist policies

### 5.5 Ablation核心发现 (Table IV)

| Setting | Pick Coke (VA) | Move Near (VM) |
|---------|----------------|----------------|
| Full SpatialVLA | 81.6% | 85.4% |
| Linear 256 bins | 40.7% (-40.9%) | 52.9% (-32.5%) |
| Uniform distribution | 77.9% | 55.0% |
| w/o Ego3D | 68.9% (-12.7%) | 62.0% (-23.4%) |
| Freeze LLM embedding | 70.2% | 50.7% |

**两个核心结论**:
1. **Adaptive grid vs uniform linear**: +40%绝对提升 — adaptive discretization是巨大winner
2. **Ego3D PE**: 在variant aggregation上带来+12.7% — 3D信息对generalization至关重要

### 5.6 Resolution sweep (Table VIII)

| Resolution | Pick Coke (VM) | Move Near (VM) |
|------------|----------------|----------------|
| U8196 (uniform) | 28.0% | 55.0% |
| 1026 | 67.3% | 54.2% |
| 4610 | 68.0% | 79.2% |
| 6166 | 74.0% | 79.2% |
| 8194 (adaptive) | 70.7% | 85.4% |

关键发现: **adaptive 4610 resolution > uniform 8196** — 用一半的bins, 性能反而更好, 因为adaptive grid集中资源在高频区域。

---

## 六、Limitations与Future Directions

Paper在Section V坦诚讨论了几个open problems:

### 6.1 Gaussian分布的局限性
- 极端场景 (如单轴运动) 会导致grid在某个axis上collapse
- 数据噪声会distort分布
- Future: VAE-based high-dimensional feature mapping + explicit grid partitioning

### 6.2 Autoregressive inference的speed bottleneck
- 当前21Hz, 但diffusion decoding (π0)可能更快
- Future: diffusion decoding + spatial grid + dynamic token number

### 6.3 Long-horizon tasks
- LIBERO-Long只有55.5% (最低的suite)
- Model只看current frame + history tokens, 缺乏显式memory mechanism
- Future: efficient historical information perception

### 6.4 Data quality
- OXE数据质量参差不齐
- Future: optimal data composition, high-quality subset distillation

---

## 七、与相关工作的定位对比

| Model | Observation | Action representation | Inference |
|-------|------------|----------------------|-----------|
| RT-2 ([arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)) | 2D only | 7 tokens × 256 bins (uniform) | Slow |
| OpenVLA ([openvla.github.io](https://openvla.github.io)) | 2D only | 7 tokens × 256 bins (uniform) | Slow |
| Octo ([octo-models.github.io](https://octo-models.github.io)) | 2D only | Diffusion head | Medium |
| π0 ([arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)) | 2D only | Flow matching | Fast |
| TraceVLA ([tracevla.github.io](https://tracevla.github.io)) | 2D + visual trace | Fine-tuned OpenVLA | Slow |
| **SpatialVLA** | **2D + Ego3D PE** | **3 tokens × 8194 adaptive grids** | **20Hz** |

SpatialVLA是第一个在observation端和action端**同时**注入3D空间结构的VLA model。3D-LLM ([arxiv.org/abs/2307.12981](https://arxiv.org/abs/2307.12981)) 和 3D-VLA ([arxiv.org/abs/2403.09631](https://arxiv.org/abs/2403.09631)) 之前也在3D-LLM上做了工作, 但它们的focus是3D understanding和prediction, 忽略了action space的3D特性。

---

## 八、Intuition总结

让我用最简练的方式总结核心intuition:

1. **2D features lack depth-awareness**: 一个杯子在桌上和地上的2D appearance可能相同, 但3D position完全不同。Ego3D PE让model显式知道object在3D space的位置。

2. **Uniform discretization wastes capacity**: Robot的action分布是高度concentrated的, 用uniform bins相当于把80%的capacity分配给永远不会用的action。Adaptive grids把capacity集中到高频区域。

3. **Polar coordinates disentangle direction from magnitude**: "向左"是一种concept, "向左多远"是另一种concept。Polar让model分别学习这两件事。

4. **Grid re-discretization是transfer的key**: 传统fine-tuning需要重新learn action semantics。SpatialVLA通过trilinear interpolation把pre-trained的spatial action knowledge"搬运"到新robot的action space, 这是parameter-efficient transfer learning在action space的analog。

5. **3 tokens vs 7 tokens**: Autoregressive generation的latency ∝ token数。3 tokens让inference速度提升2.3x, 这是real robot deployment的硬性需求。

---

## 九、Reference Links汇总

- **SpatialVLA**: [https://spatialvla.github.io](https://spatialvla.github.io)
- **OpenVLA**: [https://openvla.github.io](https://openvla.github.io) / [arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- **PaliGemma2**: [arxiv.org/abs/2412.03555](https://arxiv.org/abs/2412.03555)
- **ZoeDepth**: [arxiv.org/abs/2302.12288](https://arxiv.org/abs/2302.12288)
- **OXE dataset**: [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io)
- **RH20T**: [rh20t.github.io](https://rh20t.github.io)
- **SimplerEnv**: [simpler-env.github.io](https://simpler-env.github.io)
- **LIBERO**: [lifelong-robot-learning.github.io/LIBERO](https://lifelong-robot-learning.github.io/LIBERO/)
- **SigLIP**: [arxiv.org/abs/2303.15343](https://arxiv.org/abs/2303.15343)
- **π0**: [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- **Octo**: [octo-models.github.io](https://octo-models.github.io)
- **TraceVLA**: [tracevla.github.io](https://tracevla.github.io)
- **RT-2**: [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)
- **RT-1**: [arxiv.org/abs/2212.06817](https://arxiv.org/abs/2212.06817)
- **Diffusion Policy**: [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)
- **3D-LLM**: [arxiv.org/abs/2307.12981](https://arxiv.org/abs/2307.12981)
- **3D-VLA**: [arxiv.org/abs/2403.09631](https://arxiv.org/abs/2403.09631)
- **HPT**: [arxiv.org/abs/2409.20537](https://arxiv.org/abs/2409.20537)
- **CogACT**: [arxiv.org/abs/2411.19650](https://arxiv.org/abs/2411.19650)
- **RDT-1B**: [arxiv.org/abs/2410.07864](https://arxiv.org/abs/2410.07864)
- **FAST**: [arxiv.org/abs/2501.09747](https://arxiv.org/abs/2501.09747)
- **RoboVLM**: [arxiv.org/abs/2412.14058](https://arxiv.org/abs/2412.14058)

---

## 十、我的critical thoughts

Andrej, 站在你的视角看, 这篇paper有几个值得深挖的点:

1. **ZoeDepth是stop-gap solution**: 真正的3D foundation model (如DUSt3R, VGGT, Metric3D) 会取代ZoeDepth, 提供更准确的几何信息。Paper用ZoeDepth是因为OXE没有sensor depth, 但这是band-aid, 不是终局方案。

2. **Adaptive grids的Gaussian假设太强**: Paper自己承认Gaussian suboptimal。我直觉是mixture of Gaussians或者learned VQ-VAE会更powerful, 但会失去trilinear interpolation的优雅transfer mechanism。

3. **3 tokens的representational capacity有上限**: 8194 grids听起来很多, 但对bimanual manipulation (14D action) 或者humanoid (30+ DoF) 显然不够。Paper在limitation里提到future work会做grid sharing, 但这是non-trivial的设计挑战。

4. **Ego3D PE的frequency bias**: Sinusoidal encoding对近场物体分辨率高, 远场分辨率低。但robot manipulation的scale通常是50cm-2m, 这个range下的position encoding需要tuning, paper没有讨论这个细节。

5. **缺少与π0的apples-to-apples comparison**: π0用flow matching + action expert, 是不同的action generation paradigm。SpatialVLA用autoregressive + discrete tokens。两种paradigm的真正比较需要controlled experiment, 当前paper的比较有些unfair (π0没有在所有benchmark上跑)。

6. **Long-horizon是真正的痛点**: LIBERO-Long只有55.5%, 这说明single-frame observation + 4-action chunking不足以处理需要planning的long-horizon tasks。这部分需要architectural innovation (memory, hierarchical planning, world model), 不是单纯的spatial encoding能解决的。

7. **Action chunking的distribution shift**: 预测4个actions然后用ensemble执行, 这个trick来自Diffusion Policy。但autoregressive generation的4个tokens之间是correlated的, 不像diffusion samples那样可以独立ensemble。Paper没有详细讨论这个potential issue。

8. **与π0的philosophical分歧**: π0认为action generation应该是diffusion/flow matching (连续, multimodal), SpatialVLA认为应该是discrete tokens (autoregressive, LLM-friendly)。这场辩论还没定论, 但SpatialVLA证明了discrete + adaptive grids可以competitive, 这对LLM-centric路线是个好消息。

总的来说, SpatialVLA是一个工程上非常solid的工作, 它的核心contribution是把3D空间结构同时注入observation和action两端, 这在VLA领域是第一次。Trilinear interpolation的post-training机制尤其巧妙, 提供了一个principled的cross-embodiment transfer方法。未来的VLA foundation model大概率会吸收这两个设计, 即便具体的实现细节会演化。
