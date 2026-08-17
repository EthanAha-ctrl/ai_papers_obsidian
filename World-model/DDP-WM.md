---
source_pdf: DDP-WM.pdf
paper_sha256: f4294f2d9f6bb55a9fec71de167790fac908c279f76b7be8871003c11d2c5afd
processed_at: '2026-08-03T18:29:14-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DDP-WM 用人话说：一个关于"别给背景算命"的故事

## 0. 一句话总结

这paper说的是：**你拿ViT给一整张图算future state，其中90%的compute都在给静态background算命，纯浪费。但如果只算前景、background直接copy，你的planner会死。这篇paper告诉你怎么做到既省compute又能让planner活下来。**

---

## 1. 故事开始：一个让人头秃的延迟

DINO-WM是当前SOTA的world model，基于frozen DINOv2 features做dense prediction，在Push-T上accuracy很好。但作者发现一个问题：**一次MPC决策要120秒**。

120秒是什么概念？你的机器人推一个T-block，想一下下一步怎么推，想了2分钟。这期间T-block都没动，物理老师都要哭了。

为什么这么慢？因为DINO-WM是个dense Transformer，每一步prediction要对所有patches做self-attention。Push-T这个task里，T-block很小，background很大，但模型给每个patch都平等地算一遍attention——这就好比你查天气预报，把全国3000个县全查了一遍，但你只想知道今天出门要不要带伞。

**核心矛盾**：real-time MPC需要每秒几百次simulation，但dense Transformer $O(N^2)$ 的复杂度让你连1Hz都跑不到。

---

## 2. 作者的第一观察：物理是sparse的，模型是dense的

作者做了个很karpathy的事：**别急着设计model，先看数据**。

### Figure 1a：看model内部在干啥

把DINO-WM每层predictor的feature拿来做PCA visualization，发现一个尴尬的事：**background patches过了好几层expensive self-attention，feature几乎没变**。等于你在那里算半天矩阵乘法，结果输出约等于输入。

这就像你让一个学生做100道题，其中90道是"1+1=?"，他每道都认认真真列方程、画图、验算，最后告诉你"2"。

### Figure 1b：看真实dynamics有多sparse

把两帧consecutive frames的feature做差，再PCA。结果显示**绝大多数region的feature change接近0**，只有前景object那一点点区域有显著变化。

**直觉**：物理世界本身就是sparse的——你推一个block，动的就那个block和它接触的地方，其他地方该咋样咋样。但dense model不懂得这个道理，对所有patch一视同仁地算。

---

## 3. 第一个naive尝试：只算前景，background直接copy

既然background几乎不变，那我直接：
- 用一个小network找出哪些patch会变（localization）
- 只对这些patch做expensive prediction
- 其他patch直接copy上一帧的feature

听起来很合理对吧？FLOPs直接降一个数量级。

**结果**：open-loop pixel error确实很好，甚至比dense DINO-WM还低（Table 6里Push-T：427 vs 524）。作者心想，成了。

然后跑closed-loop MPC，**success rate只有62%**，比DINO-WM的90%差了快30个百分点。

WTF。

---

## 4. 坑在哪：一个反直觉的feature space性质

作者开始debug，发现了问题的根源。这里需要讲一个DINOv2 feature space的非平凡性质。

### DINOv2 feature不是独立的pixel

DINOv2是用self-attention训出来的，每个patch的feature**隐式编码了global context**。什么意思？

假设你的图里有个block在左边，background在右边。右边的background patch的feature里，其实混进了"左边有个block"这个信息。因为self-attention让所有patch互相看过。

现在block从左边移到中间。**pixel层面，右边的background没变**（颜色还是那个颜色）。**但feature层面，右边的background必须变**——因为它context里的block位置变了，它需要reflect这个变化。

### Naive Sparse违反了这个性质

Naive Sparse的copy-paste rule说：background pixel没变 → background feature不变 → 直接copy。

但这违反了DINOv2 feature space的intrinsic property。background feature本来应该有个small context-aware adjustment，你硬生生把它归零了。

后果是什么？**你给planner的cost landscape出现了cliffs**。

---

## 5. 用Cost Landscape可视化这个坑（这篇paper最漂亮的figure）

Figure 5是这个paper的soul。作者做了个很聪明的visualization：

拿一个成功的action sequence，固定前4步，第5步的action有两个dimension（比如x方向力和y方向力）。在这个2D action space上grid sample，每个点跑一遍DDP-WM算cost，画成3D surface。

### Naive Sparse的landscape

左边的surface**崎岖不平，全是尖刺和局部坑**。你想象CEM这种sampling-based optimizer在上面找最低点——就像一个盲人在满地陷阱的山地上找宝藏，大概率掉坑里出不来。

**这就是为什么closed-loop fail**：open-loop误差低没用，planner根本找不到optimal action，因为landscape没有smooth gradient指引方向。

### DDP-WM的landscape

右边的surface**像漏斗**，一个clear的global minimum在中间，周围smooth地slope下去。CEM随便sample都能收敛过去。

**这是整个paper最深刻的insight**：world model的好坏不在于open-loop error多低，在于它给planner造的optimization landscape有多smooth。这是一个"model value"的重新定义——model不是用来predict准的，是用来让planner好用的。

---

## 6. 解法：Low-Rank Correction Module (LRM)

既然问题是background feature需要small adjustment，但这个adjustment又不能dense地算（那就回到DINO-WM了），怎么办？

### 关键假设：background update是low-rank的

作者假设：所有background patches的update vectors $\{\Delta \mathbf{z}_i\}$ 落在一个low-dimensional subspace里。

**为什么有这个假设？** 直觉是这样的：background本身没动，它的update完全是由foreground动引起的。foreground就那么几个patch，它的motion可以用几个低维参数描述（位置变化、速度等）。background的context adjustment本质上是"因为foreground动了这么一下，我要reflect这个变化"，这个变化的信息量是low-dim的。

**Empirical验证**（Appendix A, Figure 6）：对ground truth background updates做PCA，cumulative explained variance曲线急剧上升后迅速饱和——确实是low-rank的。更妙的是，LRM学出来的updates的PCA曲线跟ground truth几乎一模一样，说明LRM真的学到了这个low-dim manifold。

### LRM的architecture：一个asymmetric cross-attention

$$\mathbf{z}_{t+1,\mathrm{bg}}' = \mathbf{z}_{t,\mathrm{bg}}' + \mathrm{CA}(\mathrm{Q}=\mathbf{z}_{t,\mathrm{bg}}', \mathrm{K}=\mathbf{z}_{t+1,\mathrm{fg}}', \mathrm{V}=\mathbf{z}_{t+1,\mathrm{fg}}')$$

变量解释：
- $\mathbf{z}_{t,\mathrm{bg}}'$ : 当前帧的background patches的feature，shape是 $\mathbb{R}^{N_{\mathrm{bg}} \times D}$，$N_{\mathrm{bg}}$是background patch数量，$D=404$是embed dim
- $\mathbf{z}_{t+1,\mathrm{fg}}'$ : 已经预测好的下一帧foreground features，shape是 $\mathbb{R}^{N_{\mathrm{fg}} \times D}$
- $\mathrm{CA}$ : cross-attention，Q来自background，K和V都来自foreground
- $\mathbf{z}_{t+1,\mathrm{bg}}'$ : 更新后的background features

**为什么是asymmetric的？** Q只能从background来，K/V只能从foreground来。这意味着：
- Background可以"看"foreground发生了什么，然后adjust自己
- Foreground不能反过来"看"background来改自己

这architecturally encode了物理上的**causal flow**：primary dynamics先发生（Stage 3已经预测好了foreground），background update是其consequence。信息流是不可逆的。

**对比DINO-WM的full self-attention**：那里background和foreground互相attend，导致模型可能学到"background影响foreground"这种non-causal spurious correlation。LRM的asymmetric design是个很强的inductive bias，告诉模型"因果方向是这样的，别瞎学"。

**计算成本**：$O(N_{\mathrm{bg}} \cdot N_{\mathrm{fg}} \cdot D)$，而 $N_{\mathrm{fg}} \ll N$，所以远小于full self-attention的 $O(N^2 \cdot D)$。这就是为什么能实现9× speedup。

---

## 7. 完整Pipeline串一遍

假设你在Push-T里推一个T-block，当前是frame $t$：

### Stage 1: History Fusion
$$\mathbf{z}_t' = \mathbf{z}_t + \mathbb{CA}(\mathrm{Q}=\mathbf{z}_t, \mathrm{K}=\mathbf{Z}_{\mathrm{hist}}, \mathrm{V}=\mathbf{Z}_{\mathrm{hist}})$$

变量：
- $\mathbf{z}_t$ : 当前帧DINOv2 features, $\mathbb{R}^{N \times D}$，$N$是patch数（Push-T大概196个），$D=404$
- $\mathbf{Z}_{\mathrm{hist}} = \{\mathbf{z}_{t-h+1}, ..., \mathbf{z}_{t-1}\}$ : 过去$h$帧的features
- $\mathbf{z}_t'$ : augmented features，隐式encode了velocity/acceleration

**人话**：每个current patch去历史信息池里query一下，看看自己过去几个时刻怎么动的，把时序信息inject进来。比DINO-WM把所有历史帧stack起来扔进full Transformer省太多了——那边是 $O((h \cdot N)^2)$，这里是 $O(h \cdot N^2)$。

### Stage 2: Dynamic Localization
$$m_i = \begin{cases} 1 & \text{if } \max(P_{\mathrm{sub},i}) > \tau \\ 0 & \text{otherwise} \end{cases} \quad \text{for } i=1,...,N$$

变量：
- $P_{\mathrm{sub},i} \in \mathbb{R}^4$ : 第$i$个patch的 $2 \times 2$ sub-region的change probability
- $\tau$ : threshold
- $m_i \in \{0,1\}$ : mask的第$i$个元素

**人话**：一个小ViT（6层, 192 dim）接收 $\mathbf{z}_t'$ 和action $\mathbf{a}_t$，预测每个patch的4个sub-region会不会变。如果4个里有任何一个超过threshold，整个patch就标记为"会变"。

**为什么搞sub-region？** Table 8的数据说明一切：
- Patch-level: IoU=0.34
- High-Precision (sub-region): IoU=0.89

精度差了2.6倍。因为Push-T里T-block可能只占patch的一半，patch-level预测会漏掉很多真正会变的区域，或者把整个patch都标成"会变"导致sparse不sparse了。

### Stage 3: Sparse Primary Predictor
拿mask $M$ 从 $\mathbf{z}_t'$ 提取前景 $\mathbf{z}_{t,\mathrm{fg}}'$，喂给一个powerful ViT（6层, 404 dim），预测 $\mathbf{z}_{t+1,\mathrm{fg}}'$。

**Adaptive Sparse Size**（Appendix E）：
$$k_{\mathrm{batch}} = \max(k_{\min}, \max_{i \in \mathrm{batch}}(k_i'))$$

变量：
- $k_{\min}=32$ : 硬件友好的最小sparse size
- $k_i'$ : 第$i$个sample实际检测到的changing regions数
- $k_{\mathrm{batch}}$ : 这batch实际feed给predictor的sequence length

**人话**：GPU讨厌variable length input。如果batch里有的sample只有5个changing patch，有的有50个，你得pad或者用ragged tensor，效率很低。这里设个下界32，少于32的random patch from background pad过去凑数，多于32的照单全收。这样tensor shape固定，GPU跑得飞快，又不丢信息。

### Stage 4: LRM (上面讲过了)

Background query foreground，做context-aware update。

### 最终输出
$\hat{\mathbf{z}}_{t+1} = \mathrm{concat}(\mathbf{z}_{t+1,\mathrm{fg}}', \mathbf{z}_{t+1,\mathrm{bg}}')$，按原来的spatial位置填回去。

---

## 8. MPC里的一个小trick：Sparse Cost Mask

$$\mathcal{L}_{\mathrm{MPC}} = \mathrm{MSE}(\hat{\mathbf{z}}_T \odot \mathbf{M}_{\mathrm{task}}, \mathbf{z}_{\mathrm{goal}} \odot \mathbf{M}_{\mathrm{task}})$$

变量：
- $\hat{\mathbf{z}}_T$ : 预测的final state features
- $\mathbf{z}_{\mathrm{goal}}$ : goal state features
- $\mathbf{M}_{\mathrm{task}}$ : 当前observation和goal image做pixel-wise diff生成的binary mask，只在differ region为1
- $\odot$ : Hadamard product

**人话**：算cost的时候，只算task-relevant region的feature error，background的error不算。因为background本来就不需要变，你把它算进cost只会引入noise，让planner分心。

Table 7的ablation显示这个trick单独贡献8%SR提升（90%→98%）。看似简单，但很多人想不到——大家习惯了dense MSE，觉得"多算总比少算好"，其实对planner来说是反直觉的干扰。

---

## 9. 训练策略：放弃end-to-end

Appendix H里作者承认了一个pragmatic的选择：**stepwise training，而非end-to-end**。

三步走：
1. 先训Localization Network（+History Fusion if used）
2. 冻住localization，训Primary Predictor，loss只算foreground
3. 冻住前两个，训LRM，loss只算background

**为什么不end-to-end？** 作者说multi-loss weight tuning is brittle and task-specific。你让foreground loss和background loss和localization loss一起backward，三个loss的gradient可能打架，你得调三个weight，每个task还得重新调，简直噩梦。

**但这暗示了一个潜在limitation**：stepwise training让每个module的representation不能co-adapt。如果localization network的error pattern和primary predictor的preference不匹配，系统suboptimal但没法fix。这是个engineering和principles的trade-off。

---

## 10. 实验数据里的几个"故事"

### 10.1 Push-T的9× speedup从哪来

Table 2: FLOPs从23G降到2.5G，9.2× reduction。

**为什么Push-T比Wall的speedup大？** Table里Wall只有3.1×。因为Push-T的T-block很小，前景patch很少，sparse后计算量暴降。Wall是个navigation task，moving agent覆盖更多patch，sparse gain小一些。

**直觉**：sparse化对"dynamics spatially concentrated"的task收益最大。这暗示了DDP-WM的适用边界——如果你的task里物体遍布整个画面都在动（比如群体行为），sparse gain会打折。

### 10.2 Throughput vs FLOPs的gap

Table 3: Push-T FLOPs降9.2×，throughput只提升9.2×（完美对齐）。但Wall FLOPs降3.1×，throughput只提升2.7×。

**原因**：memory bandwidth和kernel launch overhead在less sparsity场景下占比更大。FLOPs是理论计算量，实际throughput还受memory transfer和GPU utilization影响。当sparse程度不够高，fixed overhead占比大，speedup就打折。

### 10.3 MPC Latency的deployment意义

Table 4: Push-T 120s→16s。

**120s意味着什么？** 0.008Hz control frequency。你的机器人推一下T-block，等2分钟才能推下一步。这在任何real-time场景都是deal-breaker。

**16s意味着什么？** 0.06Hz。还是很慢，但已经进入"offline planning + online execution"的勉强可用区间。如果再优化一下（比如CEM iterations减少，或者用GPU更猛的卡），有希望进入seconds级，那就真的deployment-ready了。

### 10.4 vs Sparse Imagination的对比

Table 1: Sparse Imagination在Push-T只有78.3%，DDP-WM是98%。

**Sparse Imagination是什么？** Chun et al. 2025的工作，在imagination rollout阶段randomly drop tokens来加速。

**为什么差这么多？** Random dropping既没有localization（可能把important patch drop了），也没有LRM（landscape cliff问题没解决）。这印证了DDP-WM的核心thesis：**structured sparsity >> random sparsity**。你必须understand dynamics的structure才能sparse化，瞎sparse只会搞死planner。

---

## 11. 这个工作在更大的picture里处于什么位置

### 11.1 Model-based RL的evolution

```
Pixel prediction (World Models, Ha2018)
    → Latent dynamics (PlaNet, Dreamer series)
    → Pre-trained features (DINO-WM)
    → Structured latent dynamics (DDP-WM)  ← 你在这里
```

每一步都在把"在哪里predict"往更structured的方向推。DDP-WM把latent space进一步分解成primary和context两块，给不同computational budget。

### 11.2 跟JEPA哲学的关系

V-JEPA 2 (https://arxiv.org/abs/2506.09985) 是LeCun的predict-in-latent-space路线。DDP-WM继承了这个哲学（不reconstruct pixel，直接在DINOv2 feature space predict），但加了个structure prior：**latent dynamics本身是可分解的**。

这就像从"ensemble of weak learners"到"specialized committee"——不要所有neuron都干一样的事，让不同module专注不同性质的dynamics。

### 11.3 跟Object-Centric的对比

C-SWMs (https://openreview.net/forum?id=H1gax6VtDB), FOCUS, OC-STORM (https://arxiv.org/abs/2501.16443) 这些object-centric方法需要explicit object segmentation。对Rope（deformable）和Granular（100个粒子）是ill-defined——一个rope怎么分成object？100个粒子每个都是object吗？

DDP-WM的decoupling在feature level，不需要object概念。Rope的弯曲部分自然被localization标成foreground，granular的particle cluster也是。**这是它比object-centric方法更general的根本原因**。

### 11.4 一个没被讨论的follow-up方向

既然landscape现在smooth了，为什么还用CEM？CEM是sampling-based，要几百次rollout。如果landscape smooth，完全可以用gradient-based optimization（比如直接通过world model backprop gradient到action）。

作者在Section 5提到"smooth, tractable optimization landscape"但没follow up。这是一个明显的next step——如果gradient可用，MPC iterations可以从30次降到个位数，latency还能再降一个数量级。作者埋了这个hook没挖。

---

## 12. 我的几个吐槽

### 12.1 Localization的OOD问题

Localization network是separately trained的，如果scene distribution shift（比如光照变了，或者出现了训练集没见过的object），localization会fail。一旦localization fail，mask错，primary predictor漏掉真正moving的region，整个prediction崩。这个cascade failure paper完全没讨论。

### 12.2 Threshold $\tau$ 是个hyperparameter

公式 (2) 里的 $\tau$ 是preset的，paper没说具体值，也没说怎么选。对不同task（sparse的Push-T vs dense的Granular），最优 $\tau$ 肯定不一样。这是个hidden hyperparameter，reproducibility有风险。

### 12.3 LRM的capacity ceiling

LRM是single-layer cross-attention。对Push-T这种简单场景够用，但如果是多物体复杂interaction（比如几个block互相碰撞，context update涉及多物体信息融合），single-layer可能不够。Low-rank假设在更复杂场景下是否成立也没验证——PCA只在Push-T做了，Rope和Granular没做。

### 12.4 Constant LR schedule

Table 9显示用constant LR，不用cosine decay。这在ViT training里不常见。我猜是因为stepwise training让每个stage只有100 epochs，decay没时间发挥效果。但这也暗示了training scheme的suboptimality——如果是end-to-end with proper schedule，可能能更好。但作者选了pragmatic路线，放弃了一点optimality换reproducibility。

---

## 13. 最核心的Intuition

这篇paper教给我的最深的一课：

**World model的价值不在于predict准不准，在于给planner造的landscape好不好用。**

这跟RL里的"model value"概念呼应——一个model在open-loop看完美，但closed-loop让planner fail，那它的value是0。DDP-WM的核心贡献是把evaluation metric从open-loop error redirect到closed-loop plannability。

Figure 5那两个3D surface是整个paper的soul。左边（Naive Sparse）是rugged cliff，右边（DDP-WM）是smooth funnel。这一个figure说清了所有道理：为什么sparse难，为什么LRM work，为什么closed-loop和open-loop results diverge。

**真正的intellectual contribution是那个redirect——把"prediction accuracy"重新定义成"planning landscape smoothness"**。一旦你接受这个新definition，LRM的design就是水到渠成的——你需要一个low-cost module来smooth landscape，low-rank assumption告诉你这个module可以很简单，asymmetric cross-attention告诉你causal flow怎么encode。

---

## 14. 参考链接汇总

- DDP-WM GitHub: https://github.com/HCPLab-SYSU/DDP-WM
- DINO-WM (ICML 2025): https://proceedings.mlr.press/v267/zhou25t.html
- DINOv2: https://openreview.net/forum?id=a68SUt6zFt
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Sparse Imagination: https://arxiv.org/abs/2506.01392
- IRIS (ICLR 2023): https://openreview.net/forum?id=vhFu1Acb0xb
- DreamerV3: https://openreview.net/forum?id=0oabwyZbOu
- Push-T (Diffusion Policy): https://arxiv.org/abs/2403.03954
- C-SWMs: https://openreview.net/forum?id=H1gax6VtDB
- OC-STORM: https://arxiv.org/abs/2501.16443
- DynamicViT: https://arxiv.org/abs/2104.08756
- TokenLearner: https://proceedings.neurips.cc/paper_files/paper/2021/file/6a30e32e56fce5cf381895dfe6ca7b6f-Paper.pdf
- ViT original: https://arxiv.org/abs/2010.11929
- World Models (Ha & Schmidhuber 2018): https://proceedings.neurips.cc/paper_files/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf
- PlaNet: https://arxiv.org/abs/1811.04551
- Dreamer (2020): https://openreview.net/forum?id=S1lOTC4tDS
- Genie: https://proceedings.mlr.press/v235/bruce24a.html
- GAIA-2: https://arxiv.org/abs/2503.20523
- FlowDreamer: https://arxiv.org/abs/2507.18785
- Slot Attention: https://proceedings.neurips.cc/paper/2020/hash/8511df98c005c5be5511b00f5c5301d5-Abstract.html
- IFactor: https://arxiv.org/abs/2310.05285
- MAE: https://arxiv.org/abs/2111.06377
- VideoMAE: https://arxiv.org/abs/2203.12602
- MaskViT: https://openreview.net/forum?id=QAV2CcLEDh
- MWM: https://arxiv.org/abs/2306.00946
- Glance-and-Focus: https://arxiv.org/abs/2010.05360
- D4RL (PointMaze): https://arxiv.org/abs/2004.06112
- AdaptiGraph (Rope/Granular): https://arxiv.org/abs/2406.18014
- MPC survey (Garcia 1989): https://www.sciencedirect.com/science/article/pii/0005109889900024

---

# DDP-WM: Disentangled Dynamics Prediction World Model 深度解析

## 1. 论文的Motivation: 一个第一性原理的观察

这篇论文的起点是一个empirical observation，非常karpathy-style的"先看数据再想模型"的思路。

**Figure 1的两个观察**:
- (a) 对dense model (DINO-WM) 各predictor层的feature做PCA visualization，发现background patches的feature evolution几乎stagnant，大量self-attention计算是wasted的
- (b) 对consecutive frame的feature difference做PCA，发现绝大多数region (绿色) 的feature change接近0

这直接告诉我们：**physical dynamics is inherently sparse**，而dense self-attention的 $O(N^2)$ 复杂度是overkill的。

这个观察让我想起你之前讲过的"most of intelligence is just predicting the next token"的哲学，但这里反向了：**most of pixels aren't worth predicting**。

## 2. 关键Insight: 为什么Naive Sparse会失败

论文这里有个很关键的反直觉发现，Table 6的pixel error：

| Model | PointMaze | Push-T | Wall |
|---|---|---|---|
| DINO-WM (Dense) | 81 | 524 | 111 |
| DDP-WM (Ours) | 36 | 361 | 9 |
| Naive Sparse (w/o LRM) | 41 | 427 | 15 |

**Naive Sparse的open-loop error和DDP-WM几乎相同**，甚至远好于DINO-WM。但closed-loop MPC的success rate只有62% vs DDP-WM的98%。这是一个很关键的paradox。

**原因**: DINOv2这种基于self-attention的pre-trained representation，每个local feature都隐式编码了global context。当primary object移动时，即使是pixel层面static的background patch，其feature也需要做微小的context-aware adjustment。

Naive Sparse的copy-paste rule违反了这个feature space的intrinsic property，导致planner看到的cost landscape出现了**discontinuous cliffs**。Figure 5的两个3D cost surface plot非常直观：
- Naive Sparse: rugged, noisy, 无stable global minimum
- DDP-WM: smooth, funnel-shaped, 单一deep global minimum

这本质上对应着optimization landscape的Lipschitz continuity被破坏了。CEM这种sampling-based optimizer在rugged landscape上会陷入local optima。

## 3. Decoupling的核心假设

论文提出了一个核心假设：context-driven background updates具有**low-rank结构**。

形式化地说，所有background update vectors $\{\Delta \mathbf{z}_i\}$ 落在一个low-dimensional subspace里，等价于它们的Gram matrix是low-rank的。

Appendix A的PCA验证很decisive（Figure 6）：
- Ground Truth Updates的cumulative explained variance曲线**急剧上升后迅速饱和**
- LRM生成的updates的PCA曲线与ground truth**惊人相似**

这告诉我们LRM不仅是个computational trick，而是真正学到了physical dynamics的intrinsic low-dimensional manifold。这让我想到JePA的哲学：predict in latent space而不是pixel space，但DDP-WM更进一步——predict in a structured latent subspace。

## 4. 四阶段Pipeline的数学细节

### Stage 1: Historical Information Fusion

$$\mathbf{z}_t' = \mathbf{z}_t + \mathbb{CA}(\mathrm{Q}=\mathbf{z}_t, \mathrm{K}=\mathbf{Z}_{\mathrm{hist}}, \mathrm{V}=\mathbf{Z}_{\mathrm{hist}})$$

变量含义：
- $\mathbf{z}_t \in \mathbb{R}^{N \times D}$: 当前帧的patch tokens（$N$是patch数，$D$是embed dim=404）
- $\mathbf{Z}_{\mathrm{hist}} = \{\mathbf{z}_{t-h+1}, ..., \mathbf{z}_{t-1}\}$: 历史帧features集合
- $\mathbb{CA}$: single-layer cross-attention
- $\mathbf{z}_t'$: 时序augmented feature

直觉：DINO-WM是把所有历史帧features stack起来喂给full Transformer，$O((h \cdot N)^2)$复杂度。这里用single cross-attention把复杂度降到$O(h \cdot N^2)$。每个current patch去query历史信息pool来隐式encode velocity/acceleration。

### Stage 2: Dynamic Localization Network

输出 $P_{\mathrm{sub}} \in \mathbb{R}^{N \times 4}$，对每个patch的 $2 \times 2$ sub-region预测change probability。

$$m_i = \begin{cases} 1 & \text{if } \max(P_{\mathrm{sub},i}) > \tau \\ 0 & \text{otherwise} \end{cases} \quad \text{for } i=1,...,N$$

变量：
- $m_i$: 第$i$个patch的binary mask值
- $P_{\mathrm{sub},i}$: 第$i$个patch的4个sub-region的change probabilities
- $\tau$: preset threshold

**High-precision localization的decisive作用**（Table 8）：
- Patch-level: IoU=0.34, Precision=0.70, Recall=0.52
- High-Precision: IoU=0.89, Precision=0.91, Recall=0.98

这个细节很重要——sub-region resolution让localization quality几乎翻倍，直接传导到Table 5的pixel error从788→427（无LRM情况下）。

### Stage 3: Sparse Primary Dynamics Predictor

用mask $M$ 从 $\mathbf{z}_t'$ 提取前景subset $\mathbf{z}_{t,\mathrm{fg}}'$，喂给powerful ViT predictor。

**Adaptive Sparse Size机制**（Appendix E）：
$$k_{\mathrm{batch}} = \max(k_{\min}, \max_{i \in \mathrm{batch}}(k_i'))$$

变量：
- $k_{\min}=32$: 硬件友好的最小sparse size
- $k_i'$: 第$i$个sample实际检测到的changing regions数
- $k_{\mathrm{batch}}$: 实际feed给predictor的sequence length

这个设计是为了平衡GPU parallelism efficiency和dynamic scene complexity。当 $k_i' < k_{\mathrm{batch}}$ 时，从static background随机采样pad tokens填到 $k_{\mathrm{batch}}$。这避免了dynamic batching的overhead，又不会clip掉important dynamic info。

### Stage 4: Low-Rank Correction Module (LRM) - 核心创新

$$\mathbf{z}_{t+1,\mathrm{bg}}' = \mathbf{z}_{t,\mathrm{bg}}' + \mathrm{CA}(\mathrm{Q}=\mathbf{z}_{t,\mathrm{bg}}', \mathrm{K}=\mathbf{z}_{t+1,\mathrm{fg}}', \mathrm{V}=\mathbf{z}_{t+1,\mathrm{fg}}')$$

变量：
- $\mathbf{z}_{t,\mathrm{bg}}'$: 当前帧的background patch features
- $\mathbf{z}_{t+1,\mathrm{fg}}'$: 已预测好的next frame foreground features
- $\mathbf{z}_{t+1,\mathrm{bg}}'$: 更新后的background features

**Information flow is asymmetric and unidirectional**: background去query foreground，反过来不行。这architecturally encode了物理上的causal flow——primary dynamics先发生，background update是其consequence。

这与Full self-attention的对称性形成对比。DINO-WM让background和foreground互相attend，导致non-causal spurious correlations。LRM的asymmetric design是一个很强的inductive bias。

## 5. MPC细节: Sparse Cost Mask

$$\mathcal{L}_{\mathrm{MPC}} = \mathrm{MSE}(\hat{\mathbf{z}}_T \odot \mathbf{M}_{\mathrm{task}}, \mathbf{z}_{\mathrm{goal}} \odot \mathbf{M}_{\mathrm{task}})$$

变量：
- $\hat{\mathbf{z}}_T$: 预测的final state features
- $\mathbf{z}_{\mathrm{goal}}$: goal state features
- $\mathbf{M}_{\mathrm{task}}$: 由current observation和goal image的pixel-wise difference生成的binary mask
- $\odot$: element-wise multiplication

Ablation（Table 7）显示这个MPC Mask单独贡献了8%的提升（90%→98%），是个简单但重要的trick。

## 6. 训练策略: Stepwise Decoupled Training

Appendix H提到一个值得注意的engineering choice：

1. **Stage 1**: 训练Dynamic Localization Network（如用history，history fusion也joint trained）
2. **Stage 2**: 用stage 1的localization network，训练Primary Predictor，loss只算foreground MSE
3. **Stage 3**: 用前两个module，训练LRM，loss只算background MSE

作者明确说放弃了end-to-end joint training，因为multi-loss weight tuning is brittle and task-specific。这是个很pragmatic的选择。

## 7. 实验数据深度解读

### 7.1 Efficiency对比

| Task | FLOPs (DINO-WM) | FLOPs (Ours) | Reduction |
|---|---|---|---|
| Push-T | 23G | 2.5G | 9.2× |
| Wall | 7.8G | 2.5G | 3.1× |

Push-T的9.2× reduction比Wall的3.1×大很多——因为Push-T的dynamics更sparse（small T-block vs larger moving agent in Wall）。这印证了dynamics sparsity是稀疏化的杠杆点。

### 7.2 Throughput & Latency

| Task | Throughput DINO-WM | Throughput Ours | Speedup |
|---|---|---|---|
| Push-T | 170 | 1563 | 9.2× |
| Wall | 802 | 2170 | 2.7× |

注意Wall的throughput speedup (2.7×) 略低于FLOPs reduction (3.1×)——memory bandwidth和kernel launch overhead在less sparsity场景下相对占比更大。

| Task | MPC Latency DINO-WM | MPC Latency Ours | Speedup |
|---|---|---|---|
| PointMaze/10 | 39s | 5.5s | 7.1× |
| Push-T/30 | 120s | 16s | 7.5× |
| Wall/10 | 12s | 4.2s | 2.9× |

Push-T的120s→16s是deployment-enabling的改进。120s一个decision意味着只能做~0.5Hz control，对real-time manipulation基本不可用；16s虽然还是慢，但已经进入"勉强可用"的范畴。

### 7.3 Planning Performance

| Model | PointMaze | Push-T | Wall | Rope (CD↓) | Granular (CD↓) |
|---|---|---|---|---|---|
| IRIS | 74% | 32% | 4% | 1.11 | 0.37 |
| DreamerV3 | 100% | 30% | 100% | 2.49 | 1.05 |
| Sparse Imagination | 100% | 78.3% | 95% | — | — |
| DINO-WM | 98% | 90% | 96% | 0.41 | 0.26 |
| DDP-WM | 100% | 98% | 98% | 0.31 | 0.24 |

注意Sparse Imagination（random token dropping）在Push-T上只有78.3%，DDP-WM的98%说明**structured sparsity >> random sparsity**。这印证了Section 2.2的related work分析——general token sparsification techniques没有leverage physical dynamics的intrinsic structure。

## 8. 与相关工作的Intuition连接

### 8.1 vs. Sparse Imagination

Sparse Imagination (Chun et al., 2025, https://arxiv.org/abs/2506.01392) 在imagination rollout阶段randomly drop tokens。这是naive sparsity——既没有localization，也没有LRM。Table 1显示它在Push-T只有78.3%，证实了**random sparsity破坏了planning landscape**。

### 8.2 vs. Object-Centric Methods

C-SWMs (https://openreview.net/forum?id=H1gax6VtDB), FOCUS, OC-STORM (https://arxiv.org/abs/2501.16443) 这类object-centric方法需要explicit object segmentation，对Rope和Granular这种deformable/multi-body system是ill-defined。DDP-WM的decoupling发生在feature level而非object level，更general。

### 8.3 vs. JEPA哲学

V-JEPA 2 (https://arxiv.org/abs/2506.09985) 是LeCun的predict-in-latent-space哲学。DDP-WM是DINO-WM的predict-in-frozen-DINOv2-features路线，可看作"structured JEPA"——把latent dynamics further分解成primary和context两块，分别用不同computational budget建模。

### 8.4 vs. Token Sparsification in ViT

DynamicViT (Rao et al., 2021), TokenLearner (Ryoo et al., 2021), Glance-and-Focus (Wang et al., 2020) 都是classification场景的token pruning。它们的目标是reduce compute while preserving accuracy，但没有"prediction landscape smoothness"的概念。DDP-WM引入了一个新维度：sparsification要preserve **plannability**，这是个novel的objective。

## 9. 我的Critique和Open Questions

### 9.1 优点
- Empirical motivation扎实，Figure 1的两个PCA visualization是教科书级别的motivation figure
- Open-loop vs closed-loop paradox的分析非常深入
- LRM的asymmetric cross-attention design elegant
- Stepwise training pragmatic但放弃了representation的joint optimization

### 9.2 潜在Limitations

1. **Localization network是separately trained的**，如果scene distribution shift，localization会失败cascade到整个pipeline。Paper没讨论OOD robustness。

2. **Sub-region threshold $\tau$ 是preset hyperparameter**，paper没说怎么选。对different task可能需要重新tune。

3. **LRM的single-layer cross-attention capacity有限**。对更复杂场景（多物体interaction、长horizon），low-rank假设可能break。PCA只验证了Push-T。

4. **MPC的CEM仍是sampling-based**，30 iterations × N samples × H horizon的rollout在Push-T仍是16s。能不能combine with gradient-based planning（since landscape is now smooth）？这是论文埋下的明显follow-up hook。

5. **Historical fusion只用1层cross-attention**，对长horizon时序建模可能不足。DreamerV3用RSSM，IRIS用full transformer——这个trade-off paper没深究。

6. **End-to-end training被放弃**，这暗示joint training可能unstable。如果是这样，scaling到更大model/dataset会有engineering bottleneck。

### 9.3 与你的nanoGPT直觉的连接

你的"Attention is just a tensor operation"哲学在这里得到强化——LRM只是cross-attention，但通过asymmetric design encode了physical causality。这印证了"architecture encodes inductive bias"，而不仅仅是scale wins。

## 10. Architecture的Implementation细节

Table 10的架构参数：
- Dynamic Localization Network: 6 layers, 192 dim, MLP ratio 4.0
- Sparse Main Predictor: 6 layers, 404 dim, MLP ratio 4.0
- History Fusion: 1 layer cross-attention, 404 dim
- LRM: 1 layer cross-attention, 404 dim

注意Localization和Main Predictor的embed dim不同（192 vs 404）——localization是cheaper model因为只做binary prediction。Main predictor和DINOv2 features的404 dim对齐。

Table 9的训练超参：
- AdamW, lr=7e-4, weight decay=0.01, batch size=64, 100 epochs, constant LR schedule

Constant LR schedule有意思——通常ViT training用cosine decay，这里用constant可能是因为stepwise training让每个stage比较短，decay没必要。

## 11. 总结性Intuition

DDP-WM的核心thesis可以总结为一句：**Physical dynamics is sparse in space, low-rank in context propagation, and the planner's optimization landscape must be the primary objective, not open-loop prediction accuracy.**

这是一个深刻的insight。它把world model的evaluation从"predict accurately"重定向到"enable efficient planning"——这本质上对应RL的model-based setting下"model value"的定义。论文最contribution的地方不是speedup数字，而是Figure 5那个cost landscape visualization——它把"为什么sparse model在closed-loop下失败"这个问题可视化成了一个empirical现象，并给出了architectural fix。

这是一个典型的"先理解problem structure，再设计architecture"的工作，和你一直强调的"think deeply about the problem before throwing compute at it"的哲学完全对齐。

## 参考链接

- DDP-WM GitHub: https://github.com/HCPLab-SYSU/DDP-WM
- DINO-WM: https://proceedings.mlr.press/v267/zhou25t.html
- DINOv2: https://openreview.net/forum?id=a68SUt6zFt
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Sparse Imagination: https://arxiv.org/abs/2506.01392
- IRIS: https://openreview.net/forum?id=vhFu1Acb0xb
- DreamerV3: https://openreview.net/forum?id=0oabwyZbOu
- Push-T (Diffusion Policy): https://arxiv.org/abs/2403.03954
- C-SWMs: https://openreview.net/forum?id=H1gax6VtDB
- OC-STORM: https://arxiv.org/abs/2501.16443
- DynamicViT: https://arxiv.org/abs/2104.08756
- TokenLearner: https://proceedings.neurips.cc/paper_files/paper/2021/file/6a30e32e56fce5cf381895dfe6ca7b6f-Paper.pdf
- ViT (original): https://arxiv.org/abs/2010.11929
