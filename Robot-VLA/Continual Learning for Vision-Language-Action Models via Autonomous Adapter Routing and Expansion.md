---
source_pdf: Continual Learning for Vision-Language-Action Models via Autonomous Adapter
  Routing and Expansion.pdf
paper_sha256: 58b0a03a1056eeefa79da58b888dc5e0adb387ce245f739b4ca4c4e4e3131d32
processed_at: '2026-08-03T17:11:49-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲CLARE这篇paper

## 先说个故事

想象你雇了个新厨师, 他从culinary school毕业, 知道基本的cooking techniques, 但你家厨房的灶台、锅具、调料, 他都不熟悉. 你教他做第一道菜 (炒鸡蛋), 他学会了. 然后你教第二道 (炖牛肉), 他学会了. 但问题来了 - 教他炖牛肉的时候, 如果用的是"标准教学方法" (不停调整他对所有菜的认知), 他会逐渐忘记怎么炒鸡蛋.

这就是catastrophic forgetting - robot学新task的时候, 把旧task的knowledge给overwrite了. 现有的解法都有problems:

- **Experience Replay**: 让厨师在学新菜的时候, 时不时复习下旧菜的notes. 但旧notes可能丢了, 或者太占地方
- **EWC**: 告诉厨师"这些sauce mixing技巧是炒鸡蛋的关键, 别动". 但到了第100道菜, 所有technique都被标记成"重要", 他就学不动新东西了
- **PackNet**: 把厨师brain分成几份, 一份学一道菜. 但brain容量有限, 几道菜之后就full了
- **Modular方法 (SDP)**: 给每道菜配一个separate小厨师. 但要知道"现在要做哪道菜"才能选对小厨师, 现实中robot不知道

CLARE的idea: 给厨师准备一堆"小本子" (adapters), 每学一道菜就可能写一个小本子. 厨师做菜时, 根据眼前看到的食材和order, 自己判断"这盘菜最像哪本小本子记录的菜", 然后翻开对应的小本子辅助做菜. 而且, 如果新菜和某本旧小本子很像, 就不写新本子了, 直接复用旧的. 这样小本子的数量增长是sub-linear的.

参考: 这篇paper的project page: https://tum-lsy.github.io/clare

## 背景context - 为什么这事儿难

VLA models (Vision-Language-Action) 是现在的robot policy state-of-the-art. 你给它一张图片 + 语言指令, 它output机器人要执行的action sequence. 这些models通常build在transformer上, 用diffusion或flow matching来生成multimodal的action distribution.

关键问题: 这些models pre-train完之后, 不能zero-shot解决你家的具体task. 你必须fine-tune. 但fine-tune就destroy旧knowledge. 这是VLA deployment的核心矛盾.

而且robot不像LLM, 你可以随便store 100TB的training data. Robot在real world里, data storage和privacy都是constraints. 这就是为什么exemplar-free continual learning重要.

参考Flow Matching paper: https://arxiv.org/abs/2210.02747
参考LIBERO benchmark: https://arxiv.org/abs/2306.03310

## CLARE的核心mechanism

### 1. Adapter是个"侧记事本"

想象VLA的主network是一个大厨师brain. CLARE不动主brain, 而是在某些FFN layer旁边挂个小notebook (adapter). 这个notebook是个简单的encoder-decoder:

$$A_\ell^i(\mathbf{x}_\ell) = W_{\ell,i}^{up} \cdot \text{ReLU}(W_{\ell,i}^{down} \cdot \mathbf{x}_\ell)$$

变量意思:
- $\mathbf{x}_\ell$: layer $\ell$ 的input feature, 维度是$d_\ell$ (比如768)
- $W_{\ell,i}^{down}$: 把768维压到$r$维 (比如16维), 就是个压缩matrix
- $W_{\ell,i}^{up}$: 把16维还原回768维
- ReLU: nonlinear activation, 让adapter有expressivity

这个adapter的output加到原FFN的output上, 不replace:

$$\text{FFN}_\ell(\mathbf{x}_\ell) = \text{FFN}_\ell^{pre}(\mathbf{x}_\ell) + A_\ell^*(\mathbf{x}_\ell)$$

老FFN的output不动, adapter只补一个"task-specific的delta". 这就是stability - 旧knowledge不会被touch.

### 2. Discriminator是"小本子的封皮画像"

每个adapter配一个autoencoder作为"封皮画像". 这个autoencoder用task data训练, 学着reconstruct该task的feature distribution.

为什么autoencoder能当discriminator? 因为autoencoder本质是学一个low-dimensional manifold. 如果input属于它的training distribution, 它能reconstruct得好 (error小); 如果input是OOD, reconstruct得烂 (error大).

$$e_\ell^j(\mathbf{x}_\ell) = ||\mathbf{x}_\ell - D_\ell^j(\mathbf{x}_\ell)||_2$$

变量意思:
- $D_\ell^j$: 第$j$个autoencoder, 对应stage $j$训练的task
- $\mathbf{x}_\ell$: 当前input feature
- $e_\ell^j$: L2 reconstruction error

推理时, router做的事儿就是:
$$j^* = \arg\min_{j} e_\ell^j(\mathbf{x}_\ell)$$

哪个autoencoder reconstruction error最小, 就用哪个对应的adapter. 整个过程不需要task label - robot看着眼前场景, 自己route到合适的adapter.

### 3. Dynamic Expansion - 何时写新本子

这是个key insight. 如果对每个新task都写一个新本子, 那本子数量linear增长, 浪费. 但如果都不写新本子, 又学不动新东西.

CLARE的做法: 用z-score判断"新task和所有旧task有多不像":

$$z_\ell^j = \frac{1}{|\mathcal{D}_n|} \sum_{\mathbf{x}_\ell \in \mathcal{D}_n} \frac{e_\ell^j(\mathbf{x}_\ell) - \mu_\ell^j}{\sigma_\ell^j}$$

变量意思:
- $\mu_\ell^j$, $\sigma_\ell^j$: autoencoder $j$ 在它training distribution上的reconstruction error的mean和std
- $z_\ell^j$: 新task的reconstruction error比training distribution高了几sigma

如果所有old autoencoders的z-score都 > 2.5 (默认threshold), 说明新task对所有人都是OOD, 确实要写新本子. 否则, 不写新本子, 新task复用最相似旧task的adapter.

这就是sub-linear growth的来源 - 相似的task复用, 只有真正不同的task才expand.

### 4. Auxiliary Discriminator - 防止routing漂移

这里有个巧妙的设计. 即使某个layer不expand, 也要加一个新的auxiliary discriminator, 链接到最相似的旧adapter. Why?

想象一个危险scenario: layer 1没expand (复用旧adapter A), layer 2 expand了 (新建adapter B). 训练时layer 2的adapter B看到的feature是经过旧adapter A的. 

下次如果layer 1 expand了新adapter C, 那么回到老task时, layer 1可能route到新adapter C (因为C更新), 那layer 2看到的feature就变了, adapter B可能失效.

为了避免这个drift, 即使layer 1没expand, 也加个auxiliary discriminator, 链接到旧adapter A. 这样回到老task时, 这个auxiliary discriminator会"记住"老task, route到正确的adapter A, 保持feature consistency.

这是个subtle但重要的engineering detail.

## 实验里到底发生了什么

### LIBERO benchmark

10个long-horizon tasks, 每个task 50个human demos, 90个short-horizon tasks做pre-training. Robot是Franka arm, kitchen环境, 要做pick-and-place, 开抽屉, 转knob等.

### 主结果 (Table III)

CLARE在DiT-EncDec上AUC=66.71, ER (有exemplar的baseline) 只有55.87. 在DiT-Dec上CLARE=75.11, ER=60.54. **CLARE不用旧data都比用旧data的ER强11-15个点**.

NBT (forgetting指标):
- SeqFFT: 70.33 (catastrophic forgetting严重)
- ER: 15.79 (还有些forgetting)
- CLARE: -0.80 (基本没forgetting, 甚至有点backward transfer)

Parameter cost: 每个task增加约2% of base model.

### Layer Ablation (Table II)

最重要的发现: 只在encoder加adapter远胜过在decoder加adapter.

DiT-Dec上:
- Linear projection (encoder): AUC=75.11
- Decoder: AUC=41.75

差了30多个点. 这说明VLA里, task-specific knowledge应该活在observation encoding阶段, action generation阶段更task-agnostic.

直觉: 不同task的图片差异巨大 (不同object, 不同scene), 所以encoder需要task-specific features. 但action space是共享的 (都是gripper move), 所以decoder不需要太多task-specific adaptation.

### Expansion Threshold Ablation (Figure 5)

$\gamma$从0调到20:
- Adapter数量从60降到16 (4x reduction)
- AUC从65降到57 (下降8个点)
- NBT始终接近0

即使aggressive compression (只16个adapter), 还是比ER强. 这说明autoencoder routing确实work - 能识别哪些task可以复用adapter.

## 几个关键insight

### Insight 1: Modular > Regularization for long sequences

EWC这种regularization方法在10个tasks就撑不住了 (PackNet在DiT-Dec只有4.84 AUC). Architectural方法虽然费点参数, 但capacity无限. 这印证了Dohare et al. 2024 Nature paper的发现 - regularization方法在long sequence下会suffer from plasticity loss.

参考: https://www.nature.com/articles/s41586-024-07611-7

### Insight 2: Autoencoder是天然OOD detector

相比MoE那种learned router network, autoencoder的优势是:
- 独立训练, 不影响其他module
- Reconstruction error是distribution distance的proxy
- 天然支持"add new expert" - 不需要retrain整个router

这设计很elegant. 每个task的"指纹"是该task的autoencoder, 加新task就加新指纹, 互不干扰.

### Insight 3: Routing consistency至关重要

Auxiliary discriminator那个设计点出了modular方法的一个核心挑战 - shallow layer的routing变化会propagate到deep layer, 导致feature distribution shift. 这个问题在modular networks里普遍存在, CLARE的解决方案是"always add discriminator, sometimes add adapter".

### Insight 4: Two-stage training稳定收敛

先train adapter (flow matching loss), 再freeze adapter train discriminator (reconstruction loss). 这避免了adapter和discriminator互相干扰. 简单但必要的engineering trick.

## 局限和open questions

### 1. 10个tasks够吗?

Lifelong deployment可能100+ tasks. 那时候:
- 200% parameter growth (still OK on modern hardware)
- Routing要在100+ discriminators中argmin (latency增加)
- 可能需要hierarchical routing (先coarse cluster再fine route)

### 2. Real robot验证缺位

LIBERO是simulation. Real robot有noise, calibration drift, sensor degradation. Autoencoder的OOD detection在real data上是否还accurate, 需要验证.

参考OpenVLA真实robot实验: https://openvla.github.io/

### 3. Routing failure的后果

如果两个task在某个layer的feature distribution真的很像, autoencoder可能misroute. 错route到wrong adapter, 可能导致policy failure. 

可能的mitigation: ensemble多个layer的routing votes, 或者用confidence-based fallback to base model.

### 4. Task boundary detection

CLARE假设有人告诉robot "现在学新task了, 这是新task的data". 但real open-world里, robot得自己detect "我现在的experience和过去不一样, 该学习了". 这是更难的问题.

参考Open-world learning: https://arxiv.org/abs/2103.04176

### 5. Knowledge sharing的缺失

每个adapter独立训练, 旧adapter完全frozen. 如果新task其实和旧task有shared structure, 新adapter无法leverage旧adapter的knowledge. 可能的改进: 让新adapter的initialization从最相似的旧adapter copy过来, 然后fine-tune. 但这会break严格的stability保证.

## 对bigger picture的看法

CLARE其实提出了一个general design pattern, 不限于robotics:

**Lightweight modular adapters + unsupervised routing + dynamic expansion = exemplar-free continual learning**

这个pattern理论上可以apply到:
- LLM的continual learning (每个domain一个adapter, autoencoder判断domain)
- Multi-modal models的continual capability expansion (visual, audio, video逐步加)
- Personalized AI (每个user一个adapter, 路由到最像的user adapter)

核心思想"don't overwrite, just add and route"是强大的. 这与Lecun的JEPA思路, 与Karpathy常说的"modular compositionality", 都有concept resonance.

CLARE的engineering细节 (z-score normalization, auxiliary discriminator, two-stage training)也很有借鉴价值, 解决了naive modular方法的具体pain points.

## 一句话总结

CLARE告诉robot: "你看到新东西了? 别慌, 别动旧脑细胞, 在旁边写个小notebook. 下次看到类似的东西, 自己挑对notebook来辅助决策. 不用记住所有旧data, 也不用别人告诉你现在在做哪道菜." 这就是exemplar-free, task-identifier-free的continual learning, 在robot manipulation上第一次work得这么好.

Project page: https://tum-lsy.github.io/clare
Paper arxiv (推测): https://arxiv.org/abs/2506.05225 (CLARE paper)

未来的robot要是真能deploy到千家万户, lifelong learning这种能力是must-have. CLARE提供了一个concrete, tested的recipe, 剩下的是scale up到real robots和100+ task sequences的engineering work.

---

# CLARE: Continual Learning for Vision-Language-Action Models - 深度技术解析

## 1. 论文核心问题与Motivation

机器人deploy到real world后, 必须持续学习new tasks (新家电、新环境、新物体配置). 传统的VLA fine-tuning recipe会直接更新shared parameters, 导致catastrophic forgetting - 旧任务的semantic grounding和policy性能同时退化. 现有continual learning methods的limitation:

- **Experience Replay (ER)**: 需要存储historical data, 但privacy和storage constraints往往不允许
- **Regularization (EWC, PackNet)**: 在fixed parameter budget内工作, long task sequence下capacity bottleneck
- **Architectural methods**: 通常需要oracle task identifier, 但open-world deployment中robots自己无法知道当前是哪个task

CLARE要解决的核心问题: **exemplar-free, task-identifier-free, sub-linear parameter growth的continual learning for VLAs**

参考链接:
- Continual learning review: https://arxiv.org/abs/1904.07499
- Catastrophic forgetting in neural networks: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(99)01294-2
- LIBERO benchmark: https://arxiv.org/abs/2306.03310

## 2. Problem Setup形式化

定义task sequence $\{\mathcal{T}_n\}_{n=1}^N$, $N$未知. 每个task $\mathcal{T}_n = (\rho_0^n, l_n)$ 包含:
- $\rho_0^n$: 初始state distribution
- $l_n$: 自然语言instruction

Base policy $\pi_0 = \pi_{\theta_0}$ 预训练于internet-scale data. Observation $\mathbf{o}_t = (I_t^1, \ldots, I_t^{N_c}, \mathbf{q}_t, l)$ 包含:
- $I_t^{n_c}$: 第$n_c$个camera的RGB image
- $\mathbf{q}_t$: proprioceptive state (end-effector pose, gripper state)
- $l$: language instruction

Policy输出action chunk $\mathbf{A}_t = (\mathbf{a}_t, \ldots, \mathbf{a}_{t+H-1}) \sim \pi_0(\cdot | \mathbf{o}_t)$, 其中:
- $H$: action chunk长度 (论文中$H=16$)
- 前$h \leq H$ actions被执行 (论文中$h=8$)
- 在$t+h$时刻replan, 这就是receding horizon control, 与ACT和Diffusion Policy一致

参考链接:
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT (Bimanual Manipulation): https://arxiv.org/abs/2304.13705

## 3. Base Policy: Flow Matching

CLARE使用flow matching而非传统DDPM作为generative modeling objective. 这个选择有重要implication:

### 公式(1) - Conditional Flow Matching Loss

$$\mathcal{L}(\theta_n) = \mathbb{E}_{s, (A^1, o), A^0} \left[||v_{\theta_n}(A^s, o, s) - (A^1 - A^0)||_2\right]$$

**变量详解**:
- $\theta_n$: stage $n$的模型参数 (含adapter params)
- $s \sim \mathcal{U}([0,1])$: 时间步, 从uniform distribution采样, 表示在flow路径上的位置
- $A^1$: target action chunk, 从expert dataset $\mathcal{D}_n$采样
- $A^0 \sim \mathcal{N}(\mathbf{0}, I)$: source action chunk, 从标准Gaussian采样
- $A^s = (1-s)A^0 + sA^1$: linear interpolation, 表示在直线路径上$s$位置的点
- $o$: observation条件
- $v_{\theta_n}(A^s, o, s)$: network预测的velocity field

**Intuition**: 这个loss训练网络预测一个vector field, 该vector field指示如何从简单分布(Gaussian)的任一点flow到target distribution. 关键insight是flow matching学习**直线**路径, 而DDPM学习curved stochastic path, 所以sampling时需要更少steps.

**Inference过程**: 通过Euler integration生成action chunk:
$$A^{s+\delta s} = A^s + \delta s \cdot v_{\theta_n}(A^s, \mathbf{o}_t, s)$$

从$A^0 \sim \mathcal{N}(\mathbf{0}, I)$开始, 用$K = \lceil 1/\delta s \rceil$步积分到$A^1$. 当$\delta s$较大时, 一步即可生成 - 这就是rectified flow的优势.

参考链接:
- Flow Matching for Generative Modeling (Lipman et al., ICLR 2023): https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2203.01560
- Flower (相关robotics应用): https://arxiv.org/abs/2503.07164

## 4. Modularized Adapters - 公式深度解析

### 公式(2) - Adapter结构

$$A_\ell^i(\mathbf{x}_\ell) = W_{\ell,i}^{up} \text{ReLU}(W_{\ell,i}^{down} \mathbf{x}_\ell)$$

**变量详解**:
- $\mathbf{x}_\ell \in \mathbb{R}^{d_\ell}$: layer $\ell$的input feature (FFN input)
- $W_{\ell,i}^{down} \in \mathbb{R}^{r \times d_\ell}$: down-projection matrix, 将$d_\ell$维压缩到$r$维 ($r \ll d_\ell$)
- $W_{\ell,i}^{up} \in \mathbb{R}^{d_\ell \times r}$: up-projection matrix, 恢复到$d_\ell$维
- $i$: adapter index (在第$\ell$层有$k_\ell$个adapters)

**与LoRA的关键差异**:
- LoRA: 通常linear bottleneck, 无activation function, 用于merge回主模型
- CLARE adapter: 加ReLU非线性, 保持独立不merge
- 非线性引入使adapter expressivity更强, 但也意味着不能简单merge到FFN

**为什么用ReLU不用GELU/SiLU**: ReLU是传统选择, 简单高效, 也避免了与base model内部的activation function耦合.

### 公式(3) - FFN扩展

$$\text{FFN}_\ell(\mathbf{x}_\ell) = \text{FFN}_\ell^{pre}(\mathbf{x}_\ell) + A_\ell^*(\mathbf{x}_\ell)$$

**关键设计决策**:
1. **加法而非替换**: 保留original FFN的输出, adapter只贡献"residual" task-specific knowledge
2. **加法而非concatenation**: 不改变FFN的output维度, 保持网络结构不变
3. **加法而非MoE的weighted sum**: 不需要soft routing weight, 只activate一个adapter

这与Progressive Neural Networks的思想类似, 但更parameter-efficient. PNN在每个新task都expand entire network, CLARE只在selected FFN layers的side branch添加少量参数.

参考链接:
- LoRA: https://arxiv.org/abs/2106.09685
- Progressive Neural Networks: https://arxiv.org/abs/1606.04671
- Adapter Tuning (Houlsby): https://arxiv.org/abs/1902.00751

## 5. Autonomous Routing - Autoencoder Discriminator

### 公式(4) - Reconstruction Error

$$e_\ell^j(\mathbf{x}_\ell) = ||\mathbf{x}_\ell - D_\ell^j(\mathbf{x}_\ell)||_2$$

**变量详解**:
- $D_\ell^j$: 第$j$个autoencoder discriminator, 对应于stage $j$训练的task
- $\mathbf{x}_\ell$: 当前input feature
- $e_\ell^j(\mathbf{x}_\ell)$: $L_2$ reconstruction error

**核心intuition**: Autoencoder学习的是data的low-dimensional manifold. 当input feature属于该autoencoder训练task的distribution时, reconstruction error小; 否则OOD, reconstruction error大. 这本质上是**density estimation的proxy**, 无需显式建模probability distribution.

### 公式(5) - Discriminator训练Loss

$$\mathcal{L}_{recon}(D_\ell^n) = \mathbb{E}_{\mathbf{x}_\ell \sim \mathcal{D}_n} [e_\ell^j(\mathbf{x}_\ell)]$$

只在当前stage的data $\mathcal{D}_n$上训练, 不mix旧data. 这保证exemplar-free的同时, discriminator能记住"什么样的feature属于task $n$".

### 公式(6) - Routing决策

$$A_\ell^*(\mathbf{x}_\ell) = B_\ell(D_\ell^{j^*})$$
$$j^* = \arg\min_{j \in \{1, \ldots, n\}} e_\ell^j(\mathbf{x}_\ell)$$

**变量详解**:
- $B_\ell: \mathcal{D}_\ell \to \mathcal{A}_\ell$: surjective mapping, 多个discriminators可映射到同一个adapter
- $j^*$: 选择reconstruction error最小的discriminator的index

**与MoE router的区别**:
- MoE: trainable router network, 但需要end-to-end training, 难以extend to new experts
- CLARE: 每个expert (adapter)都有对应的autoencoder作为"特征指纹", 无需joint training
- MoE: 通常soft routing, weighted combination
- CLARE: hard routing, 只activate一个adapter, 更efficient

### Two-stage Training Strategy

这是关键engineering detail:
1. **Stage 1**: 训练new adapters via flow matching loss (公式1), 此时discriminators还未训练
2. **Stage 2**: Freeze所有参数, 只训练new discriminators via reconstruction loss (公式5)

**Why two-stage**: Discriminator的input feature $\mathbf{x}_\ell$取决于shallow layers的adapters. 如果同时训练adapters和discriminators, adapters的变化会让discriminator看到的feature distribution不稳定, 训练难以收敛. Two-stage确保discriminator训练时feature distribution已经固定.

## 6. Dynamic Expansion - 公式深度解析

### 公式(7) - Z-score

$$z_\ell^j(\mathbf{x}_\ell) = \frac{1}{|\mathcal{D}_n|} \sum_{\mathbf{x}_\ell \in \mathcal{D}_n} \frac{e_\ell^j(\mathbf{x}_\ell) - \mu_\ell^j}{\sigma_\ell^j}$$

**变量详解**:
- $\mu_\ell^j$: discriminator $D_\ell^j$的reconstruction error running mean (在task $j$ training data上)
- $\sigma_\ell^j$: 对应的running standard deviation
- $|\mathcal{D}_n|$: 当前stage $n$的dataset大小
- $z_\ell^j$: 标准化的reconstruction error

**Intuition**: Z-score衡量"当前task的reconstruction error比training distribution高了多少个标准差". 如果所有discriminators的z-score都很大, 说明当前task的特征对所有旧task都OOD, 确实需要新capacity.

### Expansion Decision Rule

**条件**: 如果 $\forall j \in \{1, \ldots, n-1\}: z_\ell^j > \gamma$, 则expand layer $\ell$:
1. 添加新adapter $A_\ell^{k_\ell}$到layer $\ell$
2. 新discriminator $D_\ell^n$链接到新adapter: $B_\ell(D_\ell^n) = A_\ell^{k_\ell}$

**Threshold $\gamma$的物理意义**: $\gamma$控制"何时认为新task真的不同". $\gamma$大则conservative expansion, 更多新task会复用old adapters; $\gamma$小则aggressive expansion, 每个细微差异都创建新adapter.

论文中$\gamma = 2.5$, 即新task的reconstruction error需要超过training distribution的2.5个标准差才算OOD.

### 公式(8) - 不expand时的链接

$$B_\ell(D_\ell^n) = A_\ell^i = B_\ell(D_\ell^{j^*})$$
$$j^* = \arg\min_{j \in \{1, \ldots, n-1\}} \mathbb{E}_{\mathbf{x}_\ell \sim \mathcal{D}_n} [e_\ell^j(\mathbf{x}_\ell)]$$

**关键scenario分析**: 

考虑一种危险情况:
- Stage $n$: layer $\ell_2$ expand了, layer $\ell_1$ (shallower) 没expand
- Stage $n+1$: layer $\ell_1$ expand了, 添加新adapter $A_{\ell_1}^j$
- 当回到task $n$: router可能选中新adapter $A_{\ell_1}^j$ 而非正确的旧adapter
- 这导致layer $\ell_2$的input feature分布shift, 训练时$A_{\ell_2}^i$看到的feature和现在不同
- 结果: 任务失败, 因为$A_{\ell_2}^i$没见过这种feature

**解决方案**: 即使不expand也添加auxiliary discriminator, 让其链接到与当前task最相似的existing adapter. 这样:
- 当下次看到task $n$的data时, 该auxiliary discriminator reconstruction error最小
- Router会activate它链接的adapter
- 这个adapter与stage $n$训练时activate的是同一个
- Feature distribution保持consistent

### 边界条件: 强制expand shallowest layer

如果没有任何layer满足expand condition, 但新task又必须学新东西, 论文强制在shallowest layer $\ell_1$添加新adapter. 

**Why shallowest**: 论文观察到shallower layers的task间feature distribution shift更明显. 这与CNN的发现一致 - shallow layers捕获low-level features (texture, shape), 不同task的物体外观差异在shallow layer更显著.

## 7. 架构细节: Diffusion Transformer (DiT)

### 两种variants

**DiT-EncDec**: 
- Self-attention transformer encoder + denoising diffusion decoder
- 采用DDPM objective
- Adapters可插入encoder和decoder的所有transformer layers

**DiT-Dec**:
- Linear projection encoder + decoder-only transformer
- 采用flow matching objective
- Adapters可插入linear projection layer和decoder的transformer layers

### 视觉和语言编码

- **Vision encoder**: DINOv2 (frozen) - self-supervised预训练, 强semantic feature
- **Language encoder**: CLIP (frozen) - vision-language aligned
- **Proprioception**: linear projection

DINOv2 + CLIP的组合提供了strong visual-language prior, 无需在continual learning过程中重新train这些encoders. 这是parameter-efficient的关键.

参考链接:
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- DINOv2: https://arxiv.org/abs/2304.07193
- CLIP: https://arxiv.org/abs/2103.00020
- AdaLN conditioning: https://arxiv.org/abs/2212.09748

## 8. 实验设置详解

### LIBERO Benchmark

- 90 short-horizon tasks for pre-training (LIBERO-90)
- 10 long-horizon tasks for continual learning (LIBERO-10)
- Franka manipulator with parallel yaw gripper
- Kitchen environment
- 50 human expert demonstrations per task
- Long-horizon: pick-and-place, opening drawer, turning knob等

### Hyperparameters (Table I)

| Module | Adapters | Discriminators |
|--------|----------|----------------|
| FFN params | 0.26M | 0.33M |
| Projection params | 3.2M | 1.4M |
| Total | 3.46M | 1.73M |

- Base model: ~200M params
- Adapter: 3.46M ≈ 1.7% of base per task
- Discriminator: 1.73M ≈ 0.87% of base per task
- 总计 per task ≈ 2.6% of base

**Training details**:
- Adapter LR: 1e-4, cosine schedule, 20,000 steps
- Discriminator LR: 5e-4, constant schedule, 2,000 steps (短得多)
- Batch size: 32 for both
- Expansion threshold $\gamma = 2.5$

### Metrics定义

$$\text{AUC} = \frac{1}{N} \sum_{n=1}^N \left(\frac{1}{N-n+1} \sum_{m=n}^N r_{n|m}\right)$$

- $r_{n|m}$: 学完task $n, n+1, \ldots, m$后, 在task $n$上的success rate
- AUC对每个task的success rate取平均, 考虑所有后续stage的影响

$$\text{FWT} = \frac{1}{N} \sum_{n=1}^N r_{n|n}$$

- Forward Transfer: 学完task $n$后立刻的success rate, 衡量学习新task的能力

$$\text{NBT} = \frac{1}{N-1} \sum_{n=1}^{N-1} \left(\frac{1}{N-n} \sum_{m=n+1}^N (r_{n|n} - r_{n|m})\right)$$

- Negative Backward Transfer: 衡量forgetting程度
- NBT > 0: forgetting
- NBT = 0: no forgetting
- NBT < 0: backward transfer (后续学习反而提升旧task, 可能是shared representation的positive transfer)

参考链接:
- LIBERO: https://arxiv.org/abs/2306.03310
- New metrics for continual learning: https://arxiv.org/abs/1810.04656

## 9. Results分析

### Table II - Layer Ablation

| Backbone | Expandable Layers | AUC | FWT | NBT |
|----------|------------------|-----|-----|-----|
| DiT-EncDec | Encoder | 65.38 | 66.53 | 1.70 |
| DiT-EncDec | Decoder | 28.99 | 30.87 | 2.95 |
| DiT-EncDec | Enc. & Dec. | 66.60 | 65.77 | 1.50 |
| DiT-Dec | Lin. projection | **75.11** | **75.03** | 1.85 |
| DiT-Dec | Decoder | 41.75 | 45.47 | 7.02 |

**Key Insight**: 
- Encoder expansion远胜Decoder expansion
- Encoder + Decoder expansion与只expand encoder相当, 不增益
- 这说明task-specific knowledge主要应该存储在observation encoding阶段, 不在action generation阶段

**Why encoder dominates**:
1. **Knowledge定位**: Geva et al. (2021)发现Transformer FFN layers是key-value memories, 但更specific地, 不同layer存储不同abstract level的knowledge
2. **Feature granularity**: Encoder处理visual/language features, 不同task的visual差异显著; Decoder已经abstract到action层面, 不同task可能share action patterns
3. **Avoid interference**: Decoder的action distribution如果被adapter干扰, 会直接影响policy output

### Table III - Main Baseline Comparison

| Backbone | Method | AUC | FWT | NBT |
|----------|--------|-----|-----|-----|
| DiT-EncDec | SeqFFT | 21.00 | 71.13 | 70.33 |
| DiT-EncDec | SeqLoRA | 16.26 | 55.00 | 53.08 |
| DiT-EncDec | PackNet | 20.91 | 73.77 | 73.74 |
| DiT-EncDec | ER | 55.87 | 67.67 | 15.79 |
| DiT-EncDec | **CLARE** | **66.71** | 66.07 | **-0.80** |
| DiT-Dec | SeqFFT | 22.37 | 76.13 | 74.70 |
| DiT-Dec | SeqLoRA | 21.37 | 73.10 | 71.64 |
| DiT-Dec | PackNet | 4.84 | 37.20 | 41.34 |
| DiT-Dec | ER | 60.54 | 76.60 | 22.74 |
| DiT-Dec | **CLARE** | **75.11** | 75.03 | 1.85 |
| - | LOTUS | 52.93 | 58.12 | -7.16 |

**Critical Observations**:

1. **CLARE vs ER**: CLARE在两种backbone上都显著超过ER (即使ER使用了previous data), 差异达11-15% AUC. 这证明exemplar-free的architectural方法可以beat exemplar-based方法.

2. **FWT对比**: 
   - SeqFFT和ER的FWT高 (76左右), 因为它们fine-tune整个模型, 快速适配新task
   - CLARE的FWT略低 (75), 因为只训练small adapter
   - 但CLARE的NBT接近0, 而SeqFFT/ER的NBT很高 (70+), 表示severe forgetting
   - 这种trade-off符合continual learning的本质: stability vs plasticity

3. **LOTUS的negative NBT**: LOTUS达到-7.16的NBT, 表示backward transfer. 这归功于其使用ER的iterative training, 旧task的skill在训练新task时也被rehearsed. 但其绝对性能不如CLARE.

4. **PackNet的失败**: 在DiT-Dec上PackNet只有4.84 AUC, 因为progressive pruning很快耗尽capacity. 这验证了fixed parameter budget方法的局限.

5. **SeqLoRA的失败**: 即使LoRA是parameter-efficient, 但因为merge回主模型, 仍然overwrites旧知识.

### Figure 5 - Expansion Threshold Ablation

$\gamma$从0增到20:
- Adapter数量: 60 → 16 (减少4x)
- AUC: 65 → 57 (下降8)
- FWT: 67 → 57 (下降10)
- NBT: 仍接近0

**Key insight**: NBT保持0, 表示forgetting不会因为expand少而增加 - 因为old adapters仍然frozen. 但新task学习能力下降, 因为更多新task被压缩到fewer adapters.

**Comparison**: 即使$\gamma=20$只有16 adapters, AUC 57仍高于ER的55.87, 说明CLARE即使aggressive compression也优于baselines.

## 10. 与Related Work的深度对比

### 与MoE的对比

MoE (Mixture of Experts):
- 固定experts数量, learned router
- Soft routing: weighted combination
- 用于LLM scaling (Mixtral, DeepSeekMoE)

CLARE:
- 动态增加experts
- Autoencoder作为"特征指纹", 无需joint training
- Hard routing: 只activate一个
- 用于continual learning

CLARE的autoencoder routing本质上是一个**memory-based**的distance metric, 与MoE的parametric router有本质不同. Autoencoder的优势是decoupled training - 每个discriminator独立训练, 不影响其他, 适合continual setting.

参考链接:
- MoE原论文: https://arxiv.org/abs/1701.06538
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
- Switch Transformer: https://arxiv.org/abs/2101.03961

### 与LoRA的对比

LoRA:
- Linear bottleneck: $W x \approx W_0 x + B A x$, $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$
- 无activation function
- 训练后merge回主模型: $W_{new} = W_0 + BA$
- 不适合continual learning, 因为merge后knowledge混合

CLARE adapter:
- 非linear bottleneck (ReLU activation)
- 不merge, 保持独立
- 配对autoencoder用于routing
- 适合continual learning

**Why ReLU matters**: ReLU让adapter成为nonlinear transformation, expressivity更强. 在continual learning setting中, 每个task的knowledge可能highly nonlinear, linear adapter可能不够expressive.

参考链接:
- LoRA: https://arxiv.org/abs/2106.09685
- AdaLoRA: https://arxiv.org/abs/2303.10512
- QLoRA: https://arxiv.org/abs/2305.14314

### 与EWC的对比

EWC (Elastic Weight Consolidation):
- Fisher Information Matrix估计parameter重要性
- Quadratic penalty保护重要参数
- 在fixed parameter budget内工作
- Long task sequence下: 所有参数都变得"important", 无法继续学习

CLARE:
- 不penalty, 而是freeze整个base model
- 只train new adapters
- Capacity通过expansion增长
- Long task sequence下: 仍能学习, 只是参数增加

**Architectural方法 vs Regularization方法**: 这是continual learning的两条主要路线. Regularization方法 (EWC, SI, LwF)的优点是不增加参数, 但capacity有限; Architectural方法 (PNN, CLARE)增加参数但capacity无限. CLARE证明在long sequence (10 tasks)和complex task (long-horizon manipulation)下, architectural方法明显更优.

参考链接:
- EWC: https://arxiv.org/abs/1612.00796
- Synaptic Intelligence: https://arxiv.org/abs/1704.05052
- LwF (Learning without Forgetting): https://arxiv.org/abs/1606.09282

### 与LOTUS, SDP的对比

LOTUS:
- Hierarchical: skill library + meta-policy
- Meta-policy需要ER训练
- Skill library expansion, 但memory-intensive

SDP (Sparse Diffusion Policy):
- Task-specific expert modules
- 需要oracle task identifier
- 无法fully autonomous

CLARE:
- Flat architecture, 无hierarchy
- Exemplar-free
- Autonomous routing, 无需task identifier

CLARE在三个维度上都更优, 但LOTUS的hierarchical design可能在更long-horizon的task上有优势 - CLARE目前只测试了LIBERO-10的10个long-horizon tasks, 如果是100+ tasks的long sequence, hierarchical可能更scalable.

参考链接:
- LOTUS: https://arxiv.org/abs/2311.17659
- SDP: https://openreview.net/forum?id=Tc9z3mGyCU

## 11. Critical Analysis - 论文的局限与未来方向

### 局限1: Routing failure的risk

Autoencoder routing不是100% accurate. 如果两个task的feature distribution在某个layer很相似, autoencoder可能选错adapter. 错误的adapter可能activate一个无关task的adapter, 导致suboptimal或failure.

**Mitigation**: 论文的设计中, 每个layer独立routing, 即使某一层错, 其他层可能仍正确, 加上base FFN仍工作, total impact可能limited.

### 局限2: Feature drift累积

随着layers变深, feature distribution取决于shallow layers的adapters选择. 如果shallow layer routing正确但deep layer routing错误, deep layer看到的feature可能drift. 这就是为什么论文要auxiliary discriminator - 即使不expand也要保持routing consistency.

但这仍然不能完全解决: 如果新adapter在shallow layer添加, 即使是相同task, deep layer看到的feature也变了. 这是architectural方法的根本挑战.

### 局限3: Real-world transfer未验证

论文只在LIBERO simulation上验证. Real robot有:
- Sensor noise
- Actuator dynamics
- Distribution shift更大
- Safety constraints更strict

Real-world deployment需要更多验证.

### 局限4: Long sequence scalability

10个tasks是不错的结果, 但lifelong deployment可能100+ tasks. 
- 2% per task → 100 tasks = 200% base model size
- Routing decision需要在100+ discriminators中arg min, 计算开销增加

**Possible solution**: hierarchical routing - 先coarse routing到task cluster, 再fine routing到具体task. 类似LOTUS的hierarchical思想.

### 局限5: Open-world novelty detection

CLARE假设新task有expert demonstrations. 但real open-world deployment中, robot可能遇到完全unknown task, 没有demonstration. 如何判断"何时学习"和"学什么"是更根本的问题.

参考链接:
- Open-world learning: https://arxiv.org/abs/2103.04176
- Lifelong learning roadmap: https://arxiv.org/abs/2208.04109

## 12. 联想与延伸思考

### 与Karpathy的neural network intuitions的关联

Karpathy在多个lecture中强调neural network的feature hierarchy和representation learning. CLARE的发现 - encoder比decoder更适合存储task knowledge - 与Karpathy的micrograd/makemore中的observation一致: shallow layers学习low-level features, deep layers学习high-level abstractions.

但CLARE的发现似乎与Geva et al. (2021)的"FFN layers are key-value memories"略有矛盾 - 后者发现mid-layer FFN存储factual knowledge. 在VLA setting中, 可能visual encoder的FFN存储更多task-relevant procedural knowledge, 而不是semantic factual knowledge.

参考链接:
- Karpathy's CS231n: https://cs231n.github.io/
- Karpathy's micrograd: https://github.com/karpathy/micrograd
- Geva et al. 2021: https://arxiv.org/abs/2012.14913

### 与O1-style reasoning models的关联

现代reasoning models (OpenAI o1, DeepSeek R1)使用test-time compute来改善reasoning. 这与CLARE的adapter routing有概念相似性:
- O1: 动态分配test-time compute给不同reasoning steps
- CLARE: 动态激活不同adapters based on input

未来可能combine两者: 让VLA在test-time既做internal reasoning (chain of thought), 又做dynamic routing.

参考链接:
- OpenAI o1: https://openai.com/o1
- DeepSeek R1: https://arxiv.org/abs/2501.06651

### 与Instruction Tuning的对比

Instruction tuning (FLAN, InstructGPT)是task-agnostic的fine-tuning, 通过diverse instructions让model generalize. CLARE是task-specific的continual learning.

潜在future: combine两者 - 在每个task adapter内部不仅学task-specific knowledge, 也学instruction-following capability, 使adapter更flexible.

参考链接:
- FLAN: https://arxiv.org/abs/2109.01652
- InstructGPT: https://arxiv.org/abs/2203.02155

### 与VLA frontier的关联

最近的VLA进展:
- OpenVLA: open-source, 7B params, fine-tunable
- π0 (Physical Intelligence): flow matching based
- π0.5: open-world generalization
- SmolVLA: affordable VLA

CLARE的方法理论上可以apply到这些large-scale VLAs. 论文也提到 "Our ideas can be straightforwardly extended to large-scale VLAs in the future". 关键challenge:
- π0等用flow matching, 与CLARE compatible
- OpenVLA用next-token prediction (autoregressive), 需要适配
- 大模型的compute开销更显著, adapter的参数efficiency更重要

参考链接:
- OpenVLA: https://openvla.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- SmolVLA: https://arxiv.org/abs/2506.01844

### 与Diffusion Policy社区的关联

Diffusion Policy (Chi et al., RSS 2023)开启了robot policy的diffusion建模时代. CLARE实际上combine了两个前沿方向:
1. Flow matching (Lipman et al., ICLR 2023)
2. Architectural continual learning (PNN, Ad-MoE)

未来可能extend到:
- 3D Diffusion Policy (3D-aware)
- Video Diffusion Policy (利用temporal info)
- Cross-embodiment Diffusion Policy (multi-robot)

参考链接:
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- 3D Diffusion Policy: https://3d-diffusion-policy.cs.columbia.edu/

### 与Meta-learning的关联

Continual learning本质上与meta-learning有connection:
- MAML: 学习good initialization for fast adaptation
- CLARE: 学习modular structure for stable adaptation

未来可能combine: 用meta-learning学习如何initialize adapters, 使新task适配更快.

参考链接:
- MAML: https://arxiv.org/abs/1703.03400
- Reptile: https://arxiv.org/abs/1803.02999

### Engineering Considerations - 实际部署角度

1. **Memory管理**: 每个adapter约3.5M params, 用FP16存储约7MB. 100个tasks = 700MB, 在edge device可接受.

2. **Inference latency**: 每个expandable layer需要前向所有discriminators (虽然只activate一个adapter). 可以cache routing decisions, 因为同一task的rollout中routing通常一致.

3. **Online learning**: 论文是offline continual learning (每个task都有完整dataset). Online setting需要更incremental的adapter训练, 可能用meta-learning初始化加速.

4. **Failure detection**: Romer et al. 2025 (论文引用[40]) 提到了runtime failure prediction for generative policies. CLARE可以combine failure detection: 当policy uncertainty高或autoencoder routing confidence低时, 触发new task learning.

### 与Plasticity vs Stability的根本tension

CLARE的设计哲学:
- **Stability**: Freeze所有old parameters, 旧knowledge完全保留
- **Plasticity**: 添加new adapter, 新task有dedicated capacity

但这是硬切分 - 旧task和新task之间no knowledge sharing. 如果新task与旧task高度相似, 强制new adapter会浪费capacity, 也无法leverage prior learning.

可能的改进: soft expansion - 当task相似时, 复用旧adapter但fine-tune其少量参数 (与原frozen weights的small delta). 类似 progressive learning with regularization.

参考链接:
- Plasticity-loss in deep learning: https://www.nature.com/articles/s41586-024-07611-7

## 13. 算法完整Walkthrough

### Algorithm 1 (Training) 详细解读

```
Require: Pretrained VLA θ_0, expandable layers E, threshold γ

Initialize:
  For each layer ℓ in E:
    A_ℓ = ∅ (no adapters)
    D_ℓ = ∅ (no discriminators)
    k_ℓ = 0 (counter for adapters in layer ℓ)

For each new task T_n:
  θ_n ← θ_{n-1} (copy previous params)
  Collect data D_n
  
  For each layer ℓ in E:
    # Compute z-scores against existing discriminators
    For each existing discriminator D_ℓ^j (j=1 to n-1):
      Compute z_ℓ^j via equation (7)
    
    # Add new discriminator
    D_ℓ ← D_ℓ ∪ {D̃_ℓ^n}
    θ_n ← (θ_n, D_ℓ^n)
    
    # Expansion decision
    If n=1 OR all z_ℓ^j > γ:
      # Expand: add new adapter
      k_ℓ ← k_ℓ + 1
      A_ℓ ← A_ℓ ∪ {A_ℓ^{k_ℓ}}
      θ_n ← (θ_n, A_ℓ^{k_ℓ})
      B_ℓ(D_ℓ^n) = A_ℓ^{k_ℓ}  # link new discriminator to new adapter
    Else:
      # No expand: link to most similar existing adapter
      B_ℓ(D_ℓ^n) = argmin over j of E[e_ℓ^j]
  
  # Force expansion if no layer was expanded
  If n > 1 AND no layer was expanded:
    Expand shallowest layer ℓ_1 in E
  
  # Two-stage training
  Train adapters A_ℓ^{k_ℓ} for all ℓ in E using flow matching loss (eq.1) on D_n
  Train discriminators D_ℓ^n for all ℓ in E using reconstruction loss (eq.5) on D_n
```

### Algorithm 2 (Inference) 详细解读

```
Require: Adapters A_ℓ, discriminators D_ℓ, linking B_ℓ, input x_ℓ

For each layer ℓ in E:
  # Compute reconstruction errors
  For each discriminator D_ℓ^j:
    Compute e_ℓ^j(x_ℓ) = ||x_ℓ - D_ℓ^j(x_ℓ)||_2
  
  # Select most relevant adapter
  j* = argmin_j e_ℓ^j(x_ℓ)
  A_ℓ^* = B_ℓ(D_ℓ^{j*})  # get linked adapter
  
  # Compute output
  FFN_ℓ(x_ℓ) = FFN_ℓ^pre(x_ℓ) + A_ℓ^*(x_ℓ)
```

**Inference complexity**: 
- 每个expandable layer: $O(n \cdot d_\ell^2)$ for forward through all discriminators + $O(r \cdot d_\ell)$ for selected adapter
- 比base FFN: $O(d_\ell^2)$
- 额外overhead: $O(n \cdot d_\ell^2)$, 当$n$大时可能成为bottleneck

**Optimization**: 可以cache routing decisions within an episode, 因为同一task的observation变化不会让routing flip-flop frequently.

## 14. 总结与我的intuition

CLARE的核心insight可以总结为:

1. **Architectural decoupling beats regularization**: 在long-horizon复杂tasks下, 添加新capacity比限制old parameter updates更effective. 这与Dohare et al. 2024 Nature论文关于plasticity loss的发现一致.

2. **Autoencoder routing > Classifier routing**: Autoencoder的reconstruction error是天然的OOD detector, 无需supervised task labels, 且每个discriminator可独立训练, 适合continual setting.

3. **Encoder matters more than Decoder**: task-specific knowledge存储在observation encoding阶段, 而action generation阶段更task-agnostic. 这是VLA特有的insight.

4. **Dynamic expansion > Fixed expansion**: 不是每个task都需要fully new capacity, 通过feature similarity判断何时reuse, 何时create new, 可达到sub-linear parameter growth.

5. **Two-stage training stabilizes**: decouple adapter training和discriminator training避免feature distribution drift, 是engineering的关键.

**对整个continual learning field的implication**:

CLARE在VLA上证明了modular + autonomous routing的可行性. 这paradigm可能extend到:
- Multi-modal LLMs的continual learning (visual capabilities逐步增强)
- RL agents的continual skill acquisition
- Multi-task recommendation systems
- Personalization (每个user一个adapter)

这种modular, exemplar-free, task-agnostic的设计哲学, 可能是AGI-scale lifelong learning的可行路径之一.

**Open questions for future work**:
1. 如何处理adapter之间的knowledge transfer (avoid完全isolation)
2. 如何detect task boundary (从streaming data中识别新task开始)
3. 如何在real-time control中handle routing overhead
4. 如何combine with reinforcement learning (除了imitation learning)
5. 如何处理embodiment shift (新robot hardware)
6. 如何reason about "why" new adapter needed (causal understanding)

参考链接:
- Continual learning survey: https://arxiv.org/abs/1904.07499
- Plasticity loss paper (Nature 2024): https://www.nature.com/articles/s41586-024-07611-7
- Project website: tum-lsy.github.io/clare

---

这篇paper代表了robotics continual learning的一个重要方向 - 从regularization-based方法转向architectural + autonomous routing方法. CLARE的成功说明, 对于复杂VLA, modular capacity allocation和decoupled routing机制是更有前景的路径, 即使牺牲一些parameter efficiency. 论文的limitation主要在real-world验证和long-sequence scalability, 但framework本身是extensible的, 可以apply到下一代large-scale VLAs如π0.5, OpenVLA等.

最exciting的是, 这种design pattern可能不限于robotics - 任何需要lifelong learning的foundation models都可能受益, 包括multi-modal LLMs, personalized assistants等. CLARE提供了一个清晰的recipe: lightweight modular adapters + feature-similarity-based routing + autonomous expansion. 这可能是走向真正lifelong learning AI systems的key technical insight之一.
