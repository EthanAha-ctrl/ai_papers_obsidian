---
source_pdf: VLS Steering Pretrained Robot Policies via Vision–Language Models.pdf
paper_sha256: 8cb3c52f2ca3504b3876bdcef960ba51c69763055aa63fe52eff38b3d3c6ebd3
processed_at: '2026-08-13T03:12:09-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说VLS

## 一句话讲清楚

你有一个已经训好的robot policy（比如π-0.5或者diffusion policy），它能完成"把杯子放桌子中间"。现在你让它"把杯子放桌边"，它fail了。VLS做的事情就是：**不动policy权重，在它生成action的denoising过程里，偷偷把action往"桌边"那个方向拽**。

## 痛点到底是什么

这个fail看起来像是policy"变傻了"，但其实是imitation learning的固有毛病。training时它学到的是"看到桌子中央的视觉特征 → 输出某种action分布"。测试时如果杯子位置变了、桌子换了、instruction换了，输入分布偏移，policy的输出就乱了。

关键insight：**motor skill本身在的**。policy知道怎么抓、怎么放、怎么move，这些dynamic在training data里都见过。缺的是一个test-time的"控制器"告诉它"这次要满足什么新spatial constraint"。

类比一下：你是个钢琴家，练了十年，手指技术都在。现在给你一份新乐谱，你没见过，但你的手指技术完全够用。你不需要重新练十年钢琴，你只需要有人告诉你这次弹什么音、什么节奏、什么力度。

## 为什么不fine-tune

三个理由，都很实在：
1. 贵——大model fine-tune烧钱烧卡
2. 概念错配——skill已经在了，relearning是浪费
3. 暴力——想覆盖所有spatial variation就得无限扩training set，这是用training解决一个本来是inference的问题

LLM社区早就在做inference-time steering了（PPLM、classifier guidance、CFG），robotics这边还在主要靠retraining。VLS就是把LLM那套"不动权重只动sampling"的思路搬过来。

## VLS的三个核心trick

### Trick 1：用VLM写一段PyTorch代码当reward function

这是最有意思的一步。传统VLM-in-the-loop是让VLM说"这个action好不好"（discrete yes/no），信号稀疏。VLS让VLM输出一段**可执行的、可微的PyTorch代码**，这段代码定义了一个scalar function：输入action trajectory，输出一个score，表示这个action多好地满足当前OOD的spatial constraint。

具体流程：
- SAM把画面里的object抠出来
- DINOv2提semantic feature
- depth反投影成3D point cloud
- clustering得到一组keypoint $\mathcal{P}$（比如杯子中心、桌边的目标位置、障碍物位置）
- VLM看一眼observation + instruction + keypoint，输出："approach阶段reward = -dist(gripper, cup_center)；place阶段reward = -dist(cup_bottom, target_point) + obstacle avoidance term"

这段代码一旦生成，就实例化进计算图。gradient直接backprop through这段代码，得到dense gradient告诉你"action应该往哪挪"。VLM本身在graph外面，不参与gradient计算——它就是个编译器，一次性把semantic reasoning编译成数学约束。

这招借鉴了EUREKA（LLM生成RL reward）和ReKep（keypoint constraint），但用在了inference-time steering这个新场景。

### Trick 2：三个guidance机制组合干活

denoising过程里，VLS同时跑三件事：

**RBF repulsion**（早期主导）：batch里采了B个particle，早期让它们互相排斥，避免全部挤到同一个mode。直觉就是"先散开找找哪里有高reward，别一上来就挤死"。

**Gradient guidance**（中后期主导）：把VLM生成的reward function的gradient注入denoising update，把每个particle往高reward方向推。这是classifier guidance的标准玩法，diffusion减去gradient，flow matching加上gradient。

**Feynman-Kac resampling**（周期性）：隔几步按reward重采一次particle。reward高的复制，reward低的剪枝。这是SMC / particle filter的标准操作，把discrete的"选择"和continuous的"gradient推"组合起来。

Ablation证实三个都必要：去掉gradient guidance直接崩盘；去掉resampling或RBF性能小幅掉但稳定性差很多。结论很干净：**需要global exploration + local refinement + selection三者协同**，单靠任何一种都不够robust。

### Trick 3：闭环execution + Schmitt trigger切换stage

robot execution不是一次sample完事，是chunk-by-chunk的closed-loop。VLS做了两件事让closed-loop变smart：

**Adaptive guidance strength**：guidance力度 $\lambda$ 根据当前chunk的reward相对于stage起始的baseline reward来调。reward变差就加大力度使劲拽，reward变好就放松让base policy自己跑。直觉就是"发现跑偏了猛拽回来，平稳了就放手让它精细操作"。

**Schmitt trigger stage切换**：用双阈值防抖。reward超过 $R_{high}$ 才进下一个stage，掉到 $R_{low}$ 以下才回退。中间是dead zone。这避免了reward在边界附近抖动导致的stage反复横跳。Schmitt trigger是1938年analog电路的老东西，借过来用得恰到好处——机器人execution本来就有物理噪声，hysteresis是天然解药。

## 为什么这套组合真的work

把VLS拆开，每个零件都不新：classifier guidance 2021，FK for diffusion 2024，RBF repulsion 2023，Schmitt trigger 1938，VLM-generated reward 2023，diffusion policy 2023。

VLS的贡献是**把它们正确组装起来**解决robot policy OOD这个具体痛点。这种systems-style工作在robotics往往比单点novelty更有impact，因为deployment的真实瓶颈是"组合对了吗"而不是"有新trick吗"。

三个设计选择我觉得特别关键：
1. **VLM作为编译器，不作为verifier**——把稀疏的semantic判断结晶成dense的数学约束，这是从discrete signal升级到continuous gradient的关键
2. **Particle-based + gradient-based双轨**——gradient负责local refine，particle负责global search + selection，互补
3. **Closed-loop with hysteresis**——物理世界有噪声，single-threshold switching会震荡，hysteresis是工程上不得不加的东西

## 我的几个直觉

**直觉1：foundation model + test-time control是下一波**

LLM社区已经在做test-time scaling（CoT、best-of-N、inference-time steering）。Robotics这边因为physical execution的constraint，test-time control的形态会不一样，但方向一致：frozen大model + 轻量inference-time adaptation。VLS是这个方向的一个早期但有说服力的proof-of-concept。

**直觉2：VLM写代码比VLM打分更值钱**

让VLM输出"这个好/不好"是稀疏信号，让VLM输出一段可微的PyTorch代码是dense信号。这个"编译"思路我觉得会扩散到很多领域——任何需要把semantic intent转成continuous optimization signal的场景都适用。Code-as-reward比Code-as-policy更promising，因为reward是objective，policy是solution，让LLM写objective比让LLM写solution更符合LLM的reasoning强项。

**直觉3：Particle方法是处理multimodal robot action的天然解**

robot action distribution往往是multimodal的（从左边抓还是右边抓，先放A还是先放B）。单trajectory的gradient guidance容易陷入局部mode，particle-based方法天然handle multimodality——每个particle探索一个mode，最后FK resampling挑最好的。这个insight其实在image diffusion社区已经在用，robotics这边刚起步。

**直觉4：Computation cost是真实瓶颈**

paper诚实地承认了latency问题。batch sampling + MCMC inner loop + FK resampling，inference成本是base policy的好几倍。对real-time control严格的场景这是硬伤。我觉得future work会往两个方向走：(a) 自适应batch size + early stopping，reward一旦converge就停MCMC；(b) 把VLM reward生成compile成高效的symbolic constraint，避免每次都backprop through PyTorch code。

**直觉5：和world model的结合是明显下一步**

VLS的reward是geometric的——keypoint之间的距离、点是否在某个region。没用物理预测（action会不会碰撞、物体会不会滑落）。如果加入forward model预测action的物理后果，guidance可以更predictive而不只是reactive。这和DynaGuide的思路互补——DynaGuide用learned dynamics model，VLS用VLM-generated geometric reward，两者结合可能更strong。

## 一个贯穿的类比

把VLS想成**"一个老练的工头带一个熟练工人"**：
- 工人= base policy，手艺都在，但只会按习惯干
- 工头= VLS，不教工人手艺，只在每个关键节点指点"这次按这个规格来，往左一点，轻一点，好了稳住"
- 工头的指点= VLM生成的reward function，把"这次要满足什么新要求"翻译成具体可执行的指令
- 工人按指点调整动作= gradient guidance + FK resampling在denoising里steer
- 工头看进度决定进入下一工序= Schmitt trigger切换stage
- 工人干得顺了工头就少说话= adaptive guidance strength

这个类比的精髓是**decoupling skill from skill execution**。工人的手艺（skill）和工头的调度（execution control）是两层，各司其职。imitation learning把两层搅在一起，VLS把它们分开。

---

参考资料都是上一篇已经列过的，这里不重复，关键的几个：
- [VLS项目页](https://vision-language-steering.github.io)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [π0](https://arxiv.org/abs/2410.24164)
- [ReKep](https://arxiv.org/abs/2409.01652)
- [EUREKA](https://arxiv.org/abs/2310.12931)
- [Feynman-Kac Steering](https://arxiv.org/abs/2501.06848)

人话版就讲到这。希望这次core idea和设计选择更清楚了。这条路线我个人挺看好，因为它正面怼了"robot policy部署时OOD必崩"这个痛点，而且给出了一个training-free的可行解，工程上可落地，理论上也自洽。

---

# VLS: Vision-Language Steering 深度讲解

这篇paper的核心论点是：pretrained generative robot policies（diffusion / flow-matching）在OOD场景下失败，并不是因为缺motor skill，而是因为imitation learning把action generation和训练时的spatial configuration紧耦合。重新training代价大且conceptually misaligned。VLS提供了一个training-free的inference-time control方案：冻结base policy，用VLM生成differentiable reward function，通过gradient guidance + particle-level resampling在denoising过程中steer action distribution。

---

## I. Motivation与核心insight

### 1.1 失败模式的本质

imitation learning training objective：

$$\max_{\theta} \mathbb{E}_{(\mathbf{a}, o, l) \sim \mathcal{D}_{expert}} \left[ \sum_{t=1}^{T} \log \pi_{\theta}(\mathbf{a}_{t:t+T} | o_t, l) \right] \tag{1}$$

变量含义：
- $\theta$：policy网络参数
- $\mathbf{a}_{t:t+T}$：从environment time step $t$ 起的action chunk，horizon长度为$T$（chunk-based prediction是diffusion policy的标配，避免step-wise抖动）
- $o_t$：observation，通常是RGB(-D)图像 + robot proprioception
- $l$：language instruction
- $\mathcal{D}_{expert}$：专家示教数据集

关键观察：这个objective是**static且distribution-dependent**的。policy会过拟合training manifold里的spatial-semantic correlations。一个"把杯子放在桌子中央"训出来的policy，在"把杯子放在桌边"时不会失败因为没有这个motor skill，而是因为action generation没被decouple到能适应新spatial constraint。

类比人类：小孩学会了"中央放置"这个motor primitive，自然能迁移到"放在边缘、放在书堆上、放在拥挤的柜子里"。机器人缺的不是skill本身，是skill的test-time adaptation mechanism。

### 1.2 为什么不fine-tune

作者给出三个理由：
1. **Cost**：大规模VLA fine-tune很贵
2. **Conceptually misaligned**：required behavior已经在training data里，只是无法在test time selectively adapt
3. **Brute-force**：扩展训练分布覆盖所有spatial variation是一个inference-time control问题，用training解决是错配的

这让我想到LLM的inference-time steering系列工作：PPLM [Dathathri et al. 2019, arXiv:1912.02164](https://arxiv.org/abs/1912.02164)、classifier guidance for diffusion [Dhariwal & Nichol 2021, arXiv:2105.05233](https://arxiv.org/abs/2105.05233)、CFG [Ho & Salimans 2022, arXiv:2207.12598](https://arxiv.org/abs/2207.12598)。这些都是"不动权重，只动sampling"的思路。VLS把这个paradigm搬到robotics。

---

## II. 背景：Diffusion Policy与Flow Matching Policy

### 2.1 Diffusion Policy的denoising更新

DDPM-style update rule：

$$\mathbf{a}_{t:t+T}^{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{a}_{t:t+T}^{k} - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \epsilon(\mathbf{a}_{t:t+T}^{k}, o, l, k) \right) + \sigma_k \mathbf{z} \tag{2}$$

变量与上下标：
- $\mathbf{a}_{t:t+T}^{k}$：denoising step $k$ 时的noisy action chunk（$k \in \{K, K-1, \ldots, 0\}$，$K$是总步数，$k=K$是纯噪声，$k=0$是clean action）
- $\alpha_k$：第$k$步的noise schedule系数
- $\bar{\alpha}_k = \prod_{i=1}^{k} \alpha_i$：累积noise schedule
- $\epsilon(\cdot)$：noise prediction网络，预测加的noise
- $\sigma_k$：step $k$ 的stochasticity variance
- $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：standard Gaussian

### 2.2 Flow Matching Policy

flow matching用连续时间$k \in [0, 1]$，$k=1$对应noise，$k=0$对应clean action，建模为ODE：

$$\frac{d\mathbf{a}_{t:t+T}^{k}}{dk} = v(\mathbf{a}_{t:t+T}^{k}, o, l, k) \tag{3}$$

- $v(\cdot)$：velocity field，被一个网络参数化
- $k$：continuous time，本质和diffusion里的discrete $k$ 意义一致

paper统一用$k$记号避免符号冗余，这个处理很干净。$\pi_0$和$\pi_{0.5}$就是flow matching policy的代表 [Black et al. 2024, arXiv:2410.24164](https://arxiv.org/abs/2410.24164)。

### 2.3 Classifier Guidance的核心idea

经典classifier guidance的核心：用 $\nabla_{\mathbf{a}^k} \log p(y | \mathbf{a}^k)$ 修正score function，其中 $y$ 是条件。在robotics里 $y = (o, l)_{OOD}$ 是OOD的observation-language pair。但这个likelihood不可直接得到，需要一个**surrogate**。这就是VLS的核心任务。

对diffusion：

$$\hat{\epsilon} = \epsilon(\mathbf{a}_{t:t+T}^{k}, (o,l)_{OOD}, k) - \lambda \cdot \sqrt{1 - \bar{\alpha}_k} \cdot g(\mathbf{a}_{t:t+T}^{k}, (o,l)_{OOD}) \tag{4}$$

对flow matching：

$$\hat{v} = v(\mathbf{a}_{t:t+T}^{k}, (o,l)_{OOD}, k) + \lambda \cdot g(\mathbf{a}_{t:t+T}^{k}, (o,l)_{OOD}) \tag{5}$$

变量：
- $\lambda$：guidance scale hyperparameter，控制guidance强度
- $g$：guidance gradient，逼近 $\nabla_{\mathbf{a}^k} \log p((o,l)_{OOD} | \mathbf{a}^k)$

注意diffusion和flow matching的符号方向不同：diffusion里减去（因为 $\epsilon$ 是预测的noise），flow matching里加上（因为 $v$ 是velocity）。

---

## III. VLS方法详解

VLS由三个component组成，paper的Fig. 2展示了完整pipeline。我把它拆解为三个模块详细讲。

### III.A OOD Input Grounding与Programmatic Reward Generation

#### III.A.1 Geometric Scaffold构建

这是把高维的 $(o, l)_{OOD}$ 压缩成一组可计算的3D keypoints $\mathcal{P} = \{p_i\}_{i=1}^{n}$，$p_i \in \mathbb{R}^3$。

具体pipeline：
1. **VLM识别**task-relevant objects和regions（用observation + language instruction query VLM）
2. **SAM分割** [Kirillov et al. 2023](https://arxiv.org/abs/2304.02643)：对每个识别的object得到mask $\mathcal{M}$
3. **DINOv2 dense features** [Caron et al. 2021](https://arxiv.org/abs/2104.14294)：得到patch-wise feature map $\Phi \in \mathbb{R}^{H \times W \times d}$
4. **Mask过滤**：用 $\mathcal{M}$ 过滤 $\Phi$
5. **Depth reprojection**：masked pixels反投影成3D point cloud
6. **Point representation**：每个point = concat(DINO feature $d$维, 3D坐标 3维) = $(d+3)$维
7. **Clustering**：对object-centric point cloud聚类，得到keypoints $\mathcal{P}$

这个pipeline和**ReKep** [Huang et al. 2024, arXiv:2409.01652](https://arxiv.org/abs/2409.01652)非常接近，VLS也明确引用了。ReKep的关键insight是用relational keypoint constraints做spatio-temporal reasoning。VLS继承了这个spatial grounding机制，但接的是differentiable reward generation而不是constraint optimization。

#### III.A.2 VLM生成differentiable reward function

这是VLS最有意思的地方。VLM不只是输出文字描述，而是输出**可执行的PyTorch代码**，这段代码定义了一个differentiable scalar function over action trajectory。

形式化定义：
$$\mathcal{R}_s(\mathbf{a}_{t:t+T}^{k}, (o,l)_{OOD}) = f_{VLM}(\mathbf{a}_{t:t+T}^{k}, \mathcal{P}, s) \tag{6}$$

变量：
- $\mathcal{R}_s$：stage $s$ 的reward function（标量输出，对 $\mathbf{a}^k$ differentiable）
- $s \in \{1, \ldots, S\}$：task的第 $s$ 个stage（VLM把task分解成 $S$ 个sequential stages）
- $f_{VLM}$：VLM生成的programmatic reward function
- $\mathcal{P}$：上一步得到的keypoint set

VLM的两步query：
1. **Task decomposition**：把task分解成 $S$ 个stages（例如pick-and-place：approach → grasp → lift → move → place）
2. **Per-stage reward generation**：每个stage生成一个reward function

为了保证differentiability，VLM被constrained输出PyTorch [Paszke et al. 2019](https://arxiv.org/abs/1912.01746)函数，只使用differentiable tensor operations（distances, dot products, soft constraints等）。

关键设计选择：VLM本身是**off-graph、non-differentiable**的——它在inference开始时生成一次reward function，然后这个函数被实例化进计算图。gradient通过reward function backprop，但不通过VLM。这点很合理，否则要fine-tune VLM，破坏training-free的承诺。

这让我联想到几个相关工作：

- **EUREKA** [Ma et al. 2023, arXiv:2310.12931](https://arxiv.org/abs/2310.12931)：LLM生成reward function for RL，但EUREKA是给RL training用，VLS是给inference-time steering用
- **Code as Policies** [Liang et al. 2022, arXiv:2209.07753](https://arxiv.org/abs/2209.07753)：LLM生成robot control代码，但不是differentiable reward
- **VoxPoser** [Huang et al. 2023, arXiv:2307.05973](https://arxiv.org/abs/2307.05973)：LLM生成composable 3D value maps，是spatial affordance map，VLS把它升级成differentiable scalar function

#### III.A.3 Guidance gradient

得到reward function后，guidance gradient直接是：

$$g_s \triangleq \nabla_{\mathbf{a}_{t:t+T}^{k}} \mathcal{R}_s(\mathbf{a}_{t:t+T}^{k}, (o,l)_{OOD}) \tag{7}$$

这个gradient是dense、trajectory-level的，backprop through instantiated reward function即可得到。

### III.B Action Denoising Process Guidance

这是VLS的algorithmic核心。三个子模块：diversity initialization、gradient-based refinement、gradient-free resampling。

#### III.B.1 RBF Repulsive Force for Diversity

在每个environment time step $t$，独立采样 $B$ 个proposals $\{\mathbf{a}_{t:t+T}^{K}[i] \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\}_{i=1}^{B}$。早期denoising step容易collapse到一个narrow mode，所以加repulsive force：

$$g_{RBF}^{k}[i] = \nabla_{\mathbf{a}_{t:t+T}^{k}[i]} \sum_{j \neq i} \frac{1}{\|\mathbf{a}_{t:t+T}^{k}[i] - \mathbf{a}_{t:t+T}^{k}[j]\|_2 + \epsilon} \tag{8}$$

变量：
- $g_{RBF}^{k}[i]$：第 $i$ 个particle在step $k$ 的repulsive gradient
- $\mathbf{a}^k[i], \mathbf{a}^k[j]$：第 $i$ 个和第 $j$ 个particle的action proposal
- $\epsilon$：numerical stability small constant，防止除零

直觉：每对particle之间有inverse-distance repulsion，越近越推开。这促使batch保持宽覆盖，避免premature collapse到suboptimal mode。

灵感来源：[Corso et al. 2023, arXiv:2310.13102](https://arxiv.org/abs/2310.13102)的Particle Guidance，以及[Jeon et al. 2025](https://api.semanticscholar.org/CorpusID:280985003)的Tree-guided Diffusion Planner。RBF kernel做repulsion在molecule generation里也很常见。

#### III.B.2 Gradient-Based Refinement

把stage-specific reward gradient $g_s = \nabla_{\mathbf{a}^k} \mathcal{R}_s$ 套进Eq. (4)或(5)。为稳定noisy gradient，用MCMC-style multiple inner updates per denoising step。Algorithm 1显示diffusion policy用 $M=4$ 个inner updates，flow matching用 $M=1$。

这个MCMC trick来自 [Du et al. 2023, arXiv:2302.11552](https://arxiv.org/abs/2302.11552)的"Reduce, Reuse, Recycle"，他们用energy-based diffusion + MCMC做compositional generation。VLS把它从compositional场景搬到OOD steering场景。

#### III.B.3 Feynman-Kac Resampling

这是VLS最有理论味道的component。把 $B$ 个proposals看作interacting particle system，按reward-based potential周期性resample。

第 $i$ 个particle在step $k$ 的potential：

$$G_i^k = \exp\left(\mathcal{R}_s(\mathbf{a}_{t:t+T}^{k}[i], (o,l)_{OOD})\right) \tag{9}$$

归一化权重：

$$w_i^k = G_i^k / \sum_{j=1}^{B} G_j^k$$

然后multinomial resample：高reward的particle被复制，低reward的particle被剪枝。

理论背景：Feynman-Kac formulae [Del Moral 2004](https://link.springer.com/book/10.1007/978-1-4757-4337-9)是Sequential Monte Carlo (SMC) [Doucet et al. 2001](https://link.springer.com/book/10.1007/978-1-4757-3443-8)的数学基础。最近的[Feynman-Kac steering for diffusion models, Singhal et al. 2025, arXiv:2501.06848](https://arxiv.org/abs/2501.06848)把这个理论应用到image diffusion。VLS把它接到robot action diffusion。

**直觉**：gradient guidance是continuous、local的refinement，FK resampling是discrete、global的selection。前者解决"局部细调"，后者解决"全局剪枝"。两者组合才能在multimodal landscape里robust导航——这是ablation验证的核心claim。

### III.C Closed-Loop Execution Control与Stage Switching

这是VLS和很多offline planning方法的关键区别：execution是closed-loop的，根据feedback自适应。

#### III.C.1 Adaptive Guidance Strength

每个action chunk $t$ 内的guidance strength $\lambda_t$ 根据当前reward相对于stage起始时的baseline reward来调：

$$\lambda_t = \lambda_{\max} \cdot \text{sigmoid}\left(1 - \frac{\mathcal{R}_s^t}{\mathcal{R}_s^{base}}\right) \tag{10}$$

变量：
- $\lambda_{\max}$：guidance strength上限
- $\mathcal{R}_s^t$：当前chunk $t$ 在stage $s$ 下的final denoising reward
- $\mathcal{R}_s^{base}$：stage $s$ 第一个chunk的reward（baseline）

直觉：
- 当 $\mathcal{R}_s^t \ll \mathcal{R}_s^{base}$（reward变差），$\text{sigmoid}(1 - \text{小}) \to \text{sigmoid}(大) \to 1$，$\lambda_t$ 接近 $\lambda_{\max}$，强guidance
- 当 $\mathcal{R}_s^t \approx \mathcal{R}_s^{base}$（reward持平），$\text{sigmoid}(0) = 0.5$，中等guidance
- 当 $\mathcal{R}_s^t \gg \mathcal{R}_s^{base}$（reward提升），$\text{sigmoid}(负) \to 0$，$\lambda_t \to 0$，base policy接管

这个schedule的好处：coarse correction时强steering，fine manipulation时让frozen base policy的prior主导。前者依赖VLM生成的explicit constraint，后者依赖base policy学到的implicit motor skill。

#### III.C.2 Schmitt Trigger Stage Switching

经典Schmitt trigger [Schmitt 1938](https://iopscience.iop.org/article/10.1088/0950-7671/15/1/305)是analog electronics里的hysteresis comparator。VLS借这个idea防stage switching的oscillation：

$$Q_t = \begin{cases} \text{Advance stage}, & \mathcal{R}_s^t > R_{high} \\ \text{Maintain stage}, & R_{low} \leq \mathcal{R}_s^t \leq R_{high} \\ \text{Reinforce stage}, & \mathcal{R}_s^t < R_{low} \end{cases} \tag{11}$$

变量：
- $R_{high}, R_{low}$：两个reward阈值，$R_{high} > R_{low}$
- $Q_t$：switching signal
- "Advance stage"：进入stage $s+1$，query VLM生成新的reward function $\mathcal{R}_{s+1}$
- "Maintain stage"：继续用 $\mathcal{R}_s$，按Eq. (10)调 $\lambda_t$
- "Reinforce stage"：用更强guidance重试stage $s$

为什么需要hysteresis？如果只用单阈值 $R_{high}$，reward在边界附近抖动会导致stage反复切换。Schmitt trigger的双阈值设计保证：从stage $s$ 到 $s+1$ 需要 $\mathcal{R}_s^t > R_{high}$，但要从 $s+1$ 回退需要 $\mathcal{R}_{s+1}^t < R_{low}$，中间有dead zone防止抖动。这是非常实用的engineering trick。

#### III.C.3 Algorithm 1完整流程

```
Input: base policy π*, initial observation o_0, language instruction l, 
       chunk horizon T, sample batch size B
Output: action chunk a_{t:t+T}

1. Condition grounding and reward generation
   P ← SAM + DINOv2 + depth(o_0)
   {R_s(a_{t:t+T}, P)}_{s=1}^S ← f_VLM(o_0, l, P)

2. Initialize
   s ← 1
   M ← 4 if π* is diffusion else 1  // MCMC inner updates

3. For each action chunk index t:
   Sample initial proposals {a^k[i] ~ N(0, I)}_{i=1}^B
   for k = K to 0:
     // Diversity initialization
     g_RBF[i] ← repulsive gradient (Eq. 8)
     apply g_RBF to Eq. (4) or (5)
     
     // Gradient-based refinement
     g_reward ← ∇_a R_s(a, P)
     for m = 1 to M:
       apply g_reward to Eq. (4) or (5)
     
     // Gradient-free resampling
     G_i ← exp(R_s(a[i]))
     w_i ← G_i / Σ_j G_j
     resample {a[i]} according to {w_i}
   
   // Closed-loop control
   adapt λ_t via Eq. (10)
   update stage s via Eq. (11)
   
   return a_{t:t+T}[0]  // 选第一个particle
```

注意Algorithm 1 return $\mathbf{a}_{t:t+T}[0]$——选第0个particle作为final action。这暗示particle index 0是resampling后权重最高的，但paper没明确说，可能是按weight排序后取最优。

---

## IV. 实验结果深度解析

### IV.1 LIBERO-PRO Results (Table I)

LIBERO-PRO [Zhou et al. 2025, arXiv:2510.03827](https://arxiv.org/abs/2510.03827) 是LIBERO [Liu et al. 2023, arXiv:2306.03310](https://arxiv.org/abs/2306.03310)的OOD test suite，覆盖五种perturbation：object, position, semantic, task, environment。VLS主要测position（物体位置改变）和task（任务逻辑改变）。

| Method | Task Avg. | Position Avg. | Overall |
|---|---|---|---|
| OpenVLA | 0.00 | 0.00 | 0.00 |
| π-0 | 0.00 | 0.00 | 0.00 |
| π-0.5 | 0.75 | 20.75 | 10.75 |
| π-0.5 (LeRobot) | 23.13 | 24.25 | 23.69 |
| π-0.5 (LeRobot) + VLS | **38.50** | **35.13** | **36.81** |

解读：
- OpenVLA [Kim et al. 2024, arXiv:2406.09246](https://arxiv.org/abs/2406.09246) 和 π-0 在LIBERO-PRO上完全fail，这强烈说明即使是SOTA VLA，post-training entangles spatial reasoning with training context，OOD下严重退化
- π-0.5 [Black et al. 2025](https://arxiv.org/abs/2410.24164) 的open-world generalization能力有部分体现，但base success rate仍只有10.75%
- LeRobot [HuggingFace LeRobot](https://huggingface.co/lerobot) fine-tuned π-0.5 提升到23.69%
- VLS叠加后提升到36.81%，**绝对提升13.12%**，paper abstract里的"13% gain on LIBERO-PRO"由此而来

更细节的子项：VLS在Task Perturbation的Object子项从10.5%提升到41.0%，这是最大的提升点。Object Perturbation指替换目标object，VLM识别新object并生成新reward，能handle这种semantic shift。

### IV.2 CALVIN Results (Fig. 3)

CALVIN [Mees et al. 2021, arXiv:2112.03227](https://arxiv.org/abs/2112.03227) 是long-horizon manipulation benchmark，含articulated objects (door, drawer, button, switch) + 三个可动cube。

| Method | Movable Objects | Articulated Parts |
|---|---|---|
| Base Policy | ~12.7% | ~9.1% |
| ITPS [Wang et al. 2024, arXiv:2411.16627](https://arxiv.org/abs/2411.16627) | 中等 | 较好 |
| DynaGuide [Du & Song 2025, arXiv:2506.13922](https://arxiv.org/abs/2506.13922) | 中等 | 中等 |
| VLS | **94% (7.4×)** | **87% (9.6×)** |

关键观察：
- ITPS在articulated tasks上OK，因为这些task的target state是fixed（door关、drawer合），discrete selection-based steering能work
- ITPS在movable objects上fail，因为cube位置随episode变化，离散selection捕捉不到continuous spatial constraint
- DynaGuide用DINO feature distance做heuristic guidance，但heuristic不够expressive，捕捉不到task-specific spatial requirement
- VLS用VLM-generated reward，针对当前observation生成specific reward function，能精确steering

VLS比prior methods提升15-25 percentage points，这个margin在robotics manipulation里是非常大的。

### IV.3 Ablation Study (Fig. 4)

三个ablation variant：
- **w/o gradient guidance**：去掉 $g_s$，只保留RBF + FK resampling。性能**崩溃**到near-failure。这是VLS最核心的component，dense trajectory-differentiable guidance是primary driver
- **w/o FK resampling**：去掉Eq. (9)的resampling。success rate小幅下降，但efficiency和stability明显退化。FK防止premature collapse到suboptimal mode
- **w/o RBF diversity**：去掉Eq. (8)的repulsion。success rate小幅下降。RBF保证早期denoising的global coverage

结论：**gradient-free global exploration + gradient-based local refinement 都必要**。这符合combinatorial optimization的"explore-exploit"直觉：RBF做explore（broad sampling），gradient做exploit（local refinement），FK做selection（pruning）。

Batch size $K$ scaling：Fig. 4 (right)显示增大batch size提升success rate但增加inference latency，呈现compute-performance tradeoff。这是deployable系统的关键tuning knob。

### IV.4 Real-World Deployment (Fig. 5)

Franka Emika robot + frozen π-0.5。

**In-Distribution (Level 1 + Level 2)**:
- Level 1: 按instruction把orange放到指定plate (red or green)
- Level 2: 加一个banana，需要sequential选择target object和target plate

VLS平均69% vs baseline 50%，**提升19%**。

**Out-of-Distribution三个变体**:
1. **Appearance shift**: 替换red/green plate为unseen yellow plate
2. **Position shift**: swap两个plate位置，instruction不变
3. **Object shift**: 替换banana为unseen mug，instruction改成"place mug on green plate"

最challenging的Object shift：baseline 0%（完全fail，因为mug从没见过），VLS 40%。这显示VLM的open-vocabulary识别 + programmatic reward generation能handle unseen object。

评分标准：grasping correct object 50%，full completion 100%。这个design反映出robotics里"识别对+抓对"已经是巨大挑战，full completion是更高bar。

---

## V. 与Related Work的对比

### V.1 Imitation-Trained Policies under Small Environment Shifts

这是大背景：[Diffusion Policy, Chi et al. 2023, arXiv:2303.04137](https://arxiv.org/abs/2303.04137)、[Open X-Embodiment, O'Neill et al. 2024](https://arxiv.org/abs/2310.08864)、[DROID, Khazatsky et al. 2024, arXiv:2403.12945](https://arxiv.org/abs/2403.12945)、[BridgeData V2, Walke et al. 2023](https://arxiv.org/abs/2308.12952)。这些大规模imitation learning能产出expressive policies，但brittleness under small environment shift是公认limitation。VLS直接面对这个limitation，避开了retraining路线。

[The Colosseum, Pumacay et al. 2024, arXiv:2402.08191](https://arxiv.org/abs/2402.08191)系统量化了这个brittleness。VLS的实验设计参考了这类generalization benchmark的思路。

### V.2 VLM-based Scene Understanding with Re-optimization

VoxPoser [Huang et al. 2023, arXiv:2307.05973](https://arxiv.org/abs/2307.05973) 和 ReKep [Huang et al. 2024, arXiv:2409.01652](https://arxiv.org/abs/2409.01652) 用VLM生成scene representation，然后online re-optimize。这些方法需要rollout / repeated evaluation / online optimization，computational heavy，real-time control不友好。

[Open-world TAMP via VLM-inferred constraints, Kumar et al. 2024, arXiv:2411.08253](https://arxiv.org/abs/2411.08253)走的是TAMP路线。

VLS的区别：保留pretrained policy作为skill prior，用lightweight inference-time control代替heavy online optimization。

### V.3 Inference-time Steering

#### Value/Critic-Guided
- **V-GPS** [Nakamoto et al. 2024, arXiv:2410.13816](https://arxiv.org/abs/2410.13816)：用offline-learned value function re-rank actions
- **VGD** [Ye 2025](https://openreview.net/forum?id=dtMBW9W5jo)：把value/Q model的gradient inject进denoising

VLS的critique：这些方法用auxiliary learned objective重塑policy，但base policy不再是invariant的，被critic的preference改写了。VLS坚持base policy是invariant的，只有test-time constraint modulates execution。

#### Dynamics/World-Model Guided
- **DynaGuide** [Du & Song 2025, arXiv:2506.13922](https://arxiv.org/abs/2506.13922)：external dynamics model guide denoising，preserve diffusion prior
- **Latent Policy Barrier** [Sun & Song 2025, arXiv:2508.05941](https://arxiv.org/abs/2508.05941)：learned dynamics model predict latent states，trajectories stay in expert manifold under covariate shift

VLS的critique：依赖predictive modeling，对model error敏感，inference cost高。

#### Human/VLM-in-the-loop
- **ITPS** [Wang et al. 2024, arXiv:2411.16627](https://arxiv.org/abs/2411.16627)：human interaction signals steer sampling
- **FOREWARN** [Wu et al. 2025, arXiv:2502.01828](https://arxiv.org/abs/2502.01828)：VLM as verifier select from candidate plans
- **Do What You Say** [Wu et al. 2025, arXiv:2510.16281](https://arxiv.org/abs/2510.16281)：VLM check reasoning-action faithfulness，filter candidate sequences

VLS的critique：discrete + sparse supervision，sample-inefficient，需要fine-grained constraint satisfaction时不够。

- **VLA-Pilot** [Li et al. 2025, arXiv:2511.14178](https://arxiv.org/abs/2511.14178)：和VLS最接近，但VLA-Pilot focus在guiding pretrained policy handle OOD via gradient-guided denoising + dynamic stage transitions。VLS的differentiator：extensive sim + real-world testing，Feynman-Kac resampling的引入。

#### Online Improvement without Finetuning
- **Policy Decorator** [Yuan et al. 2024, arXiv:2412.13630](https://arxiv.org/abs/2412.13630)：residual refinement policy做online correction
- **USR** [Zhu et al. 2025](https://openreview.net/forum?id=DbBD2aT1OG)：unified latent steering + residual refinement
- **DSRL** [Wagenmaker et al. 2025, arXiv:2506.15799](https://arxiv.org/abs/2506.15799)：在diffusion latent/noise space优化，black-box access

VLS的区别：pure inference-time adaptation，无online learning。

### V.4 我自己的联想：Diffusion + SMC路线

VLS的Feynman-Kac resampling让我想到最近image diffusion社区的几个工作：
- [FK Steering, Singhal et al. 2025, arXiv:2501.06848](https://arxiv.org/abs/2501.06848)
- [Particle Guidance, Corso et al. 2023, arXiv:2310.13102](https://arxiv.org/abs/2310.13102)
- [Reduce-Reuse-Recycle, Du et al. 2023, arXiv:2302.11552](https://arxiv.org/abs/2302.11552)
- [Compositional Diffusion, Liu et al. 2022, arXiv:2206.01739](https://arxiv.org/abs/2206.01739)
- [Diffusion Forcing, Chen et al. 2024, arXiv:2402.18211](https://arxiv.org/abs/2402.18211)

VLS本质上把image diffusion community的particle-based inference-time scaling + classifier guidance迁移到robot action diffusion。这个迁移的非trivial之处在于：
1. Image的guidance function往往是预定义的（如text-image CLIP score），robot的guidance function需要根据task动态生成
2. Image sampling一次性输出，robot execution是sequential closed-loop，需要stage switching + adaptive strength
3. Action space比pixel space小但更structured，gradient的quality对最终动作影响更大

---

## VI. Intuition Building总结

### 6.1 核心intuition：Decoupling skill from skill execution

人类motor skill能跨spatial variation迁移，因为skill本身和skill execution decouple。Robot policies缺这个decoupling。VLS显式做这个decoupling：
- **Base policy** = motor primitives generator (frozen, invariance)
- **VLS** = execution controller (test-time, adaptive)

类比：base policy是trained musician的手指技术，VLS是sheet music上的dynamic marks告诉手指什么时候强、什么时候弱、什么时候停。技术没变，但execution被reshape。

### 6.2 三种guidance的分工

VLS内部三种guidance mechanism有清晰分工：

| Mechanism | 作用 | 何时主导 |
|---|---|---|
| RBF Repulsion | 多样性、防collapse | 早期denoising step |
| Gradient Guidance | 局部refinement toward constraint | 中后期denoising |
| FK Resampling | 全局selection / pruning | 周期性，cross-step |
| Adaptive λ | 在chunk间调guidance强度 | execution closed-loop |
| Schmitt Trigger | stage间切换 | stage boundary |

这种layered design是VLS鲁棒性的来源。单纯靠gradient guidance会premature collapse（ablation验证），单纯靠selection会sample-inefficient，三者组合才robust。

### 6.3 VLM作为reward function generator的深层意义

这是VLS在concept层面的关键insight：**VLM不只是semantic parser，而是differentiable function generator**。

传统的VLM-in-the-loop方法是VLM输出verification信号（accept/reject）。这种discrete supervision稀疏且sample-inefficient。

VLS让VLM输出**可微代码**，把semantic reasoning结晶成explicit mathematical constraint。这有几个深层好处：
1. **Differentiable** → 可以backprop through它，得到dense gradient
2. **Explicit** → 可解释、可调试、可compose
3. **Programmatic** → 不需要每次都query VLM（生成一次用整个episode），inference cost低
4. **Composable** → 多stage可以串联，stage间切换由Schmitt trigger管理

这条路线让我想到LLM-as-optimizer系列工作：[EUREKA](https://arxiv.org/abs/2310.12931)、[Text2Reward](https://arxiv.org/abs/2309.11437)、[Code as Policies](https://arxiv.org/abs/2209.07753)。VLS把这个idea用到了robot policy inference-time steering上，是一个值得探索的方向。

### 6.4 Limitations的诚实承认

paper最后承认VLS的limitation：batch sampling + MCMC + FK resampling引入高inference overhead。这是particle-based方法的固有cost。

Fig. 4 (right)的compute-performance tradeoff曲线显示，batch size增大提升success rate但增latency。对real-time control严格的场景（如高freq闭环）这是瓶颈。Future work方向：progress-aware reward signal generation + inference computation优化。

一个我能想到的extension：把FK resampling改成progressive的——只在reward variance高时resample，reward converged时省掉resampling。或者用learned scheduler替代fixed Schmitt trigger阈值。

### 6.5 工程直觉：为什么这套组合有效

把VLS拆开看，每个component单独都不是新东西：
- Classifier guidance: 2021
- Feynman-Kac for diffusion: 2024-2025
- RBF repulsion: 2023
- Schmitt trigger: 1938
- VLM-generated reward: 2023 (EUREKA)
- Diffusion policy: 2023

VLS的贡献在于**正确组合**这些component解决robot policy OOD adaptation问题。这是systems-style贡献而非纯novel mechanism贡献。在robotics领域，这种"组合已知technique解决实际痛点"的工作往往比pure novelty更有impact。

---

## VII. 可能的扩展与开放问题

最后我抛几个值得深挖的方向：

1. **Reward function的verification**：VLM生成的PyTorch代码可能有bug或non-differentiable operation。paper没详细讨论验证机制。是否需要formal verification / runtime type check / 单元测试？

2. **Multi-modal reward**：当前 $\mathcal{R}_s$ 是scalar，对multimodal task（如"放在A或B任一位置"）需要max over多个potential field。是否可以用mixture of Gaussians或者von Mises分布？

3. **Reward shaping的自动生成**：现在stage decomposition和reward shape都依赖VLM prompt engineering。能否learn一个meta-reward-generator？

4. **Particle效率**：B个particle是fixed。能否用adaptive batch size，early stopping判断哪个particle已经converged？

5. **和RLHF-style feedback的结合**：VLS的reward是VLM-generated的programmatic function，能否加入human feedback作为correction signal？类似RLHF + classifier guidance。

6. **跨policy迁移**：VLS生成的reward function能否跨policy迁移？比如给diffusion policy生成的reward能否用到flow matching policy？理论上reward function只依赖 $\mathcal{P}$ 和action，是policy-agnostic的，这点VLS已经implicitly利用（既适用diffusion也适用flow matching）。

7. **Long-horizon的stage decomposition**：现在VLM一次性分解所有stage，对超long-horizon task（如50步）可能不可行。能否incremental decomposition，每个stage结束时再生成下个stage？

8. **和World Model的结合**：VLS的guidance是geometry-based，没用dynamic model。如果加入forward model预测action的物理后果（如碰撞、滑落），guidance可以更predictive。这和DynaGuide [Du & Song 2025](https://arxiv.org/abs/2506.13922)的思路互补。

---

## VIII. 总结

VLS是一个**training-free, inference-time, particle-based, VLM-guided**的robot policy steering框架。它把image diffusion community的classifier guidance + Feynman-Kac resampling + particle diversity技巧，结合robotics特有的closed-loop stage switching和VLM-generated differentiable reward，组成一个完整pipeline。在CALVIN上提升31%，LIBERO-PRO上提升13%，real-world Franka实验验证可部署性。

核心贡献不在单点novelty，而在**系统级的正确组合**：frozen base policy + VLM-generated differentiable reward + RBF/gradient/FK三重steering + Schmitt trigger closed-loop control。这套组合针对"skill exists但execution needs adaptation"这个imitation learning的痛点，给出了第一个training-free的可行方案。

参考链接整理：
- VLS项目页: [vision-language-steering.github.io](https://vision-language-steering.github.io)
- Diffusion Policy: [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137)
- π0: [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- OpenVLA: [arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
- CALVIN: [arxiv.org/abs/2112.03227](https://arxiv.org/abs/2112.03227)
- LIBERO: [arxiv.org/abs/2306.03310](https://arxiv.org/abs/2306.03310)
- LIBERO-PRO: [arxiv.org/abs/2510.03827](https://arxiv.org/abs/2510.03827)
- ReKEp: [arxiv.org/abs/2409.01652](https://arxiv.org/abs/2409.01652)
- VoxPoser: [arxiv.org/abs/2307.05973](https://arxiv.org/abs/2307.05973)
- EUREKA: [arxiv.org/abs/2310.12931](https://arxiv.org/abs/2310.12931)
- SAM: [arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
- DINOv2: [arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)
- Classifier Guidance (DHARIWAL): [arxiv.org/abs/2105.05233](https://arxiv.org/abs/2105.05233)
- Feynman-Kac Steering: [arxiv.org/abs/2501.06848](https://arxiv.org/abs/2501.06848)
- Particle Guidance: [arxiv.org/abs/2310.13102](https://arxiv.org/abs/2310.13102)
- Reduce-Reuse-Recycle: [arxiv.org/abs/2302.11552](https://arxiv.org/abs/2302.11552)
- DynaGuide: [arxiv.org/abs/2506.13922](https://arxiv.org/abs/2506.13922)
- ITPS: [arxiv.org/abs/2411.16627](https://arxiv.org/abs/2411.16627)
- VLA-Pilot: [arxiv.org/abs/2511.14178](https://arxiv.org/abs/2511.14178)
- Latent Policy Barrier: [arxiv.org/abs/2508.05941](https://arxiv.org/abs/2508.05941)
- FOREWARN: [arxiv.org/abs/2502.01828](https://arxiv.org/abs/2502.01828)
- Do What You Say: [arxiv.org/abs/2510.16281](https://arxiv.org/abs/2510.16281)
- V-GPS: [arxiv.org/abs/2410.13816](https://arxiv.org/abs/2410.13816)
- DSRL: [arxiv.org/abs/2506.15799](https://arxiv.org/abs/2506.15799)
- Policy Decorator: [arxiv.org/abs/2412.13630](https://arxiv.org/abs/2412.13630)
- The Colosseum: [arxiv.org/abs/2402.08191](https://arxiv.org/abs/2402.08191)
- Schmitt Trigger 原始paper: [DOI 10.1088/0950-7671/15/1/305](https://iopscience.iop.org/article/10.1088/0950-7671/15/1/305)
- LeRobot π0.5 checkpoint: [huggingface.co/lerobot/pi05_libero_finetuned](https://huggingface.co/lerobot/pi05_libero_finetuned)
- Feynman-Kac Formulae (Del Moral 2004): [link.springer.com/book/10.1007/978-1-4757-4337-9](https://link.springer.com/book/10.1007/978-1-4757-4337-9)
- SMC (Doucet et al. 2001): [link.springer.com/book/10.1007/978-1-4757-3443-8](https://link.springer.com/book/10.1007/978-1-4757-3443-8)

希望这个深度梳理帮到你build up对VLS的intuition，以及对"inference-time steering for generative robot policy"这个新方向的sense。这条路线我个人感觉是robot policy deployment的下一波重要方向——frozen foundation model + lightweight test-time control，类比LLM社区的test-time scaling，但加了物理世界的constraint satisfaction这个独特维度。
