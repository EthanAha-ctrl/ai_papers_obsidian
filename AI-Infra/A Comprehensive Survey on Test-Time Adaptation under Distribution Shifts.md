---
source_pdf: A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts.pdf
paper_sha256: 7fabe56c39dbeec844a39b5ca9d0def0222f2d5edcc25bdcdad5cb37e359afbf
processed_at: '2026-07-17T09:56:18-07:00'
target_folder: AI-Infra
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Test-Time Adaptation Survey 深度讲解

这篇 survey 是 Jian Liang、Ran He、Tieniu Tan 写的，发表于 IJCV，是第一篇把 TTDA、TTBA、OTTA 三个 topic 统一在 TTA 这个大伞下的综述。repo 在 https://github.com/tim-learn/awesome-test-time-adaptation 。

## 1. 核心问题与 motivation

传统 ML 假设 $p_S(x,y) = p_T(x,y)$（i.i.d.），但真实部署中 camera 变了、城市变了、hospital 设备变了，导致 distribution shift。Domain Generalization (DG) 只在 training 阶段做文章，Domain Adaptation (DA) 要同时拿到 source data 和 target data，TTA 的位置在中间——**只有 pretrained model + unlabeled test data，在 inference 之前快速 adapt**。

关键 trade-off：privacy（医学数据不能传 source）、latency（在线 streaming 要快）、memory（端侧设备存不下 source）。

## 2. 三种 setting 的精确定义

设 test 时有 $m$ 个 unlabeled mini-batches $b_1, \dots, b_m$：

| Setting | 数据形式 | 适配方式 | 代表方法 |
|---------|---------|---------|---------|
| **TTDA** | 整个 target domain，所有 $m$ 个 batch | multi-epoch 训练后再 inference | SHOT, SHOT++, 3C-GAN |
| **TTBA** | 单个 mini-batch ($B \geq 1$) | 每个 batch 独立 adapt，互不影响 | TTT, MEMO, PredBN |
| **OTTA** | streaming，每个 batch 只看一次 | 知识累积，前 batch 帮后 batch | Tent, CoTTA, EATA, SAR |

这三个 setting 之间有包含关系：OTTA 多 epoch 跑就退化成 TTDA；TTBA 假设知识可重用就变成 OTTA。这种统一视角是这篇 survey 最重要的 contribution。

详细 reference：
- SHOT: https://proceedings.mlr.press/v119/liang20a.html
- TTT: https://arxiv.org/abs/1909.13231
- Tent: https://arxiv.org/abs/2006.10726
- CoTTA: https://arxiv.org/abs/2203.05105
- EATA: https://arxiv.org/abs/2110.03395
- SAR: https://arxiv.org/abs/2302.12400

---

## 3. TTDA: Test-Time Domain Adaptation

### 3.1 Problem definition

给定 source 上训练好的 $f_S: \mathcal{X}_S \to \mathcal{Y}_S$ 和 unlabeled target $\mathcal{D}_T = \{x_1, \dots, x_{n_t}\}$，目标是在 transductive learning 方式下推所有 target 样本的 label。关键约束：adaptation 期间 **看不到 source data**，只能从 $f_S$ 里面拿"隐式知识"（BN statistics、classifier weights、logits 等）。

### 3.2 五大方法族

#### (1) Pseudo-labeling

通式：

$$\min_\theta \mathbb{E}_{(x,\hat{y}) \in \mathcal{D}_t} w_{pl}(x) \cdot d_{pl}(\hat{y}, p(y|x;\theta))$$

- $w_{pl}(x)$：每个伪标签样本的权重（filtering noisy labels）
- $d_{pl}(\cdot,\cdot)$：divergence，常用 cross-entropy $-\sum_c \hat{y}_c \log [p(y|x;\theta)]_c$
- $\hat{y} \in \mathbb{R}^C$：可以是 hard one-hot 或 soft distribution

**Centroid-based（SHOT 系）**：核心思想用 weighted k-means 思路 denoise。SHOT 的更新公式：

$$m_c = \frac{\sum_x p_\theta(y_c|x) \cdot g(x)}{\sum_x p_\theta(y_c|x)}, \quad c \in [1, C]$$
$$\hat{y} = \arg\min_c d(g(x), m_c)$$

- $m_c$：第 $c$ 类的 centroid，由 prediction probability 加权平均
- $g(x)$：feature encoder 输出
- $d(\cdot,\cdot)$：cosine distance
- 直觉：单个 sample 的 argmax 容易错，但 centroid 把同类 sample 的 feature 平均掉了 noise

SHOT 同时固定 classifier head $W$，只更新 feature encoder $\theta$，防止 classifier collapse。这是 source-free 设定下的关键 trick——保留 source classifier 作为 anchor。

**Neighbor-based（NRC、SSNLL、AdaContrast）**：假设 feature space 局部 smooth，用 kNN 的预测聚合：

$$\hat{p}_i = \frac{1}{m} \sum_{j \in \mathcal{N}_i} q_j$$

- $\mathcal{N}_i$：memory bank 中 $g(x_i)$ 的 $m$ 个最近邻 index
- 直觉：测试集里临近样本大概率同类，用邻居投票比 self-prediction 稳

**Complementary labels（negative learning）**：

$$\min_\theta -\sum_{i=1}^{n_t} \sum_{c=1}^{C} \mathbb{1}(\bar{y}_i = c) \log(1 - p_\theta(y_c|x_i))$$

- $\bar{y}_i$：从 $\{1,\dots,C\} \setminus \{\hat{y}_i\}$ 中随机选的 "negative label"
- 直觉：即使 $\hat{y}_i$ 错了，告诉模型"不是 $\bar{y}_i$"的概率仍有 $\frac{C-1}{C}$ 的正确率。Negative learning 提供了更平滑的 supervision signal。

**Optimization-based（ASL、KUDA）**：直接把"类别均衡"硬约束写进优化：

$$\min_{\hat{p}_i} -\sum_i \sum_c \hat{p}_{ic} \log p_\theta(y_c|x_i) + \lambda \sum_i \sum_c \hat{p}_{ic} \log \hat{p}_{ic}$$
$$\text{s.t.} \quad \sum_c \hat{p}_{ic}=1, \quad \sum_i \hat{p}_{ic} = \frac{n_t}{C}$$

- 第二项是 entropy regularizer 防止 trivial 解
- 约束 $\sum_i \hat{p}_{ic} = n_t/C$：强制每个类有同样多样本，避免 winner-takes-all
- 这是 Sinkhorn-Knopp 类最优传输问题的变体

#### (2) Consistency Training

**Data variations（FixMatch 思路）**：

$$\mathcal{L}_{fm}^{con} = \frac{1}{n_t} \sum_{i=1}^{n_t} \text{CE}\big(p_{\tilde{\theta}}(y|x_i), p_\theta(y|\hat{x}_i)\big)$$

- $\tilde{\theta}$：当前参数的 frozen copy（teacher）
- $\hat{x}_i$：strong augmentation 后的输入
- 直觉：weak augmentation 下的 prediction 当 pseudo label，监督 strong augmentation 下的 prediction

**VAT（Virtual Adversarial Training）**：

$$\mathcal{L}_{vat}^{con} = \frac{1}{n_t} \sum_i \max_{\|\Delta_i\| \leq \epsilon} \text{KL}\big(p_{\tilde{\theta}}(y|x_i) \| p_\theta(y|x_i + \Delta_i)\big)$$

- $\Delta_i$：在 $\epsilon$-ball 内最 adversarial 的 perturbation
- 直觉：找最难扰动的方向，强制 prediction 在该方向上不变——提升 local Lipschitz smoothness

**Model variations（Mean Teacher）**：

$$\mathcal{L}_{mt}^{con} = \mathbb{E}_{x \in \mathcal{D}_t} d_{mt}\big(p(y|x,\theta), p(y|\tau(x),\theta_{tea})\big)$$
$$\theta_{tea} = (1-\eta)\theta_{tea} + \eta\theta$$

- $\tau(\cdot)$：strong augmentation
- $\eta$：momentum 系数（通常 0.999）
- $\theta_{tea}$：teacher 的 EMA，时间维度的 ensemble，比 student 更稳
- 直觉：teacher 提供稳定 target，student 学 augmentation robustness

#### (3) Clustering-based Training

**Entropy minimization**：直接降低 prediction 的不确定度

$$\mathcal{L}_{tsa} = \frac{1}{n_t} \sum_i \frac{1}{\alpha-1} \bigg[1 - \sum_c p_\theta(y_c|x_i)^\alpha\bigg]$$

- $\alpha$：entropic index。$\alpha \to 1$ 退化为 Shannon entropy
- $\alpha = 2$ 时是 maximum squares loss $\sum_c p^2$，gradient 在高概率区线性增长，防 easy sample 主导
- 直觉：cluster assumption——decision boundary 应该在 low-density 区域，最小化 entropy 推动 boundary 远离 sample

**Mutual information maximization（SHOT）**：

$$\max_\theta \mathcal{I}(\mathcal{X}_t, \hat{\mathcal{Y}}_t) = \mathcal{H}(\hat{\mathcal{Y}}_t) - \mathcal{H}(\hat{\mathcal{Y}}_t | \mathcal{X}_t)$$
$$= -\sum_c \bar{p}_\theta(y_c) \log \bar{p}_\theta(y_c) + \frac{1}{n_t}\sum_i \sum_c p_\theta(y_c|x_i) \log p_\theta(y_c|x_i)$$

- $\bar{p}_\theta(y_c) = \frac{1}{n_t}\sum_i p_\theta(y_c|x_i)$：target 上 marginal class distribution
- 第一项 $\mathcal{H}(\hat{\mathcal{Y}}_t)$：diversity term，鼓励类别均衡，防止 collapse 到一类
- 第二项 $-\mathcal{H}(\hat{\mathcal{Y}}_t|\mathcal{X}_t)$：entropy minimization
- 直觉：单做 entropy min 会让所有样本塌缩到一类；加 diversity 项把 marginal 推向 uniform
- 等价于 $\text{KL}(\bar{p}_\theta(y) \| \mathcal{U})$，$\mathcal{U}$ 是 uniform distribution

#### (4) Source Distribution Estimation

把 SFDA 转成普通 DA——用 generator 或 selection 重构"伪 source domain"。

**3C-GAN**：

$$\min_{\theta_G} \max_{\theta_D} \mathbb{E}_{x_t}[\log D(x_t)] + \mathbb{E}_{y_t,z}[\log(1-D(G(y_t,z)))] - \lambda_s \mathbb{E}_{y_t,z} \sum_c \mathbb{1}(y_t=c)\log p(y_c|G(y_t,z),\theta)$$

- $z$：随机噪声
- $y_t$：随机采样的 conditional label
- $\lambda_s$：balance
- 直觉：用 conditional GAN 生成 target-style 的伪 source 样本，再用这些样本和真实 target 做 DA

**BN-matching constraint（SFDA-KTMA）**：

$$\mathcal{L}_{bn} = \sum_l \sum_i \|\mu_{g,l}^{(i)} - \mu_{s,l}^{(i)}\|_2 + \|\delta_{g,l}^{(i)^2} - \delta_{s,l}^{(i)^2}\|_2$$

- $\mu_{s,l}^{(i)}, \delta_{s,l}^{(i)^2}$：source model BN 层存的 running mean / variance
- $\mu_{g,l}^{(i)} = \frac{1}{B}\sum_z f_l^{(i)}(x_g)$：generator 生成数据在 BN 层的统计量
- 直觉：BN 统计量编码了 domain style（光照、对比度等），让生成数据匹配 source 的 BN 就能逼真还原 source domain

**Data selection（SHOT++、DaC）**：把 target 分两份，confident 的当 pseudo-source，剩下的当 target，做 intra-domain DA。SHOT++ 用 entropy 排序选 low-entropy sample。

#### (5) Self-supervised Learning

SHOT++ 加 rotation prediction auxiliary head；TTT++ 用 MoCo 在 source 学 self-supervised branch，target 用同样的 objective 继续 adapt。StickerDA 设计 sticker location 等 pretext task。

---

## 4. TTBA: Test-Time Batch Adaptation

### 4.1 BN Calibration

BN 层 forward：

$$\hat{x}_s = \gamma \cdot \frac{x_s - \mathbb{E}[\mathcal{X}_S]}{\sqrt{\mathbb{V}[\mathcal{X}_S] + \epsilon}} + \beta$$

- $\gamma, \beta$：learnable affine
- $\epsilon$：numerical stability
- 训练时 BN statistics 用 EMA 估计：
$$\mu_s \leftarrow (1-\rho)\mu_s + \rho\hat{\mu}_k, \quad \sigma_s^2 \leftarrow (1-\rho)\sigma_s^2 + \rho\hat{\sigma}_k^2$$
- $\rho$：momentum，训练时通常 0.1

**AdaBN** 的关键 insight：BN 的 statistics 是 domain-specific 的。直接把 source 的 $\mu_s, \sigma_s^2$ 替换成 target 上的估计 $\hat{\mu}_t, \hat{\sigma}_t^2$ 就能很大程度消除 domain gap。

**PredBN+** 的混合策略：

$$\bar{\mu}_t = (1-\rho_t)\mu_s + \rho_t\hat{\mu}_t, \quad \bar{\sigma}_t^2 = (1-\rho_t)\sigma_s^2 + \rho_t\hat{\sigma}_t^2$$

- $\rho_t$：test-time interpolation weight
- 直觉：完全替换风险大（batch 小、estimate 不稳），插值更安全

**TTN** 进一步修正方差估计：

$$\bar{\sigma}_t^2 = (1-\rho_t)\sigma_s^2 + \rho_t\hat{\sigma}_t^2 + \rho_t(1-\rho_t)(\hat{\mu}_t - \mu_s)^2$$

- 第三项：均值 shift 引起的额外方差
- 直觉：两个分布的混合方差 = 各自方差的加权 + 均值差的平方贡献，这是概率论基本公式

### 4.2 Model Optimization

**TTT（Test-Time Training）**：Y-shaped architecture
- Shared encoder $f_e(\cdot; \theta_e)$
- Classification head $h_c(\cdot; \theta_c)$
- SSL head $h_s(\cdot; \theta_s)$（如 rotation prediction）

训练目标：

$$\theta_e^*, \theta_c^*, \theta_s^* = \arg\min_{\theta_e, \theta_c, \theta_s} \sum_{i=1}^{n_s} \mathcal{L}_{pri}(x_i, y_i; \theta_e, \theta_c) + \mathcal{L}_{ssl}(x_i; \theta_e, \theta_s)$$

测试时对单样本 $x_t$：

$$\theta_e(x_t) = \arg\min_{\theta_e} \mathcal{L}_{ssl}(x_t; \theta_s^*, \theta_e)$$

- 只用 SSL loss 更新 encoder，再 inference
- 直觉：SSL loss 不需要 label，但学到的是 domain-relevant feature；通过最小化 SSL loss 把 encoder 拉向 test domain 的 feature space
- 关键假设：SSL task 和 primary task 共享底层 representation

**MEMO**：

$$\min_\theta \mathcal{H}\bigg(\frac{1}{K}\sum_{k=1}^K p_\theta(y|\tau_k(x_t))\bigg)$$

- $\tau_k$：第 $k$ 种 augmentation
- $K$：augmentation 数量
- 直觉：单 augmentation 容易过拟合到 augmentation artifact；averaged prediction 的 entropy 下降意味着模型对所有 augmentation 都给出一致 confident 预测
- MEMO 不需要修改 source training，纯 on-the-fly

### 4.3 Meta-Learning

**MLSR**（meta-learning for super-resolution）：

$$\min_\theta \sum_i \mathcal{L}\big(\text{LR}_i, \text{HR}_i; \theta - \alpha\nabla_\theta \mathcal{L}(\text{LR}_i\downarrow, \text{LR}_i; \theta)\big)$$

- 内层 $\nabla_\theta \mathcal{L}(\text{LR}_i\downarrow, \text{LR}_i; \theta)$：用低分辨率图和它再下采样的版本做 self-supervised
- 外层 $\mathcal{L}(\text{LR}_i, \text{HR}_i; \cdot)$：真实 supervised loss
- $\alpha$：inner learning rate
- 直觉：训练时模拟 test 时的 adaptation 流程，让初始 $\theta$ 在 inner step 之后就能在 supervised task 上 work

测试时：

$$\theta_t \leftarrow \theta^* - \alpha\nabla_\theta \mathcal{L}(\text{LR}\downarrow, \text{LR}; \theta^*)$$

### 4.4 Input Adaptation

不修改 model，改 input。**TPT**（Test-time Prompt Tuning）：frozen CLIP + learnable text prompt，最小化 marginal entropy。**OST** 用 Fourier style transfer 把 target input 推到 source manifold。

### 4.5 Dynamic Inference

**LAME**：不更新参数，在 feature space 用 Laplacian regularization 强制邻居预测一致：

$$\min_Q \text{tr}(Q^T L Q) + \sum_i \text{KL}(p_\theta(y|x_i) \| q_i)$$

- $Q$：调整后的预测矩阵
- $L$：feature graph 的 Laplacian
- 直觉：feature space 上邻居应该同类，用 graph smoothness 调整 logits

---

## 5. OTTA: Online Test-Time Adaptation

### 5.1 BN Calibration（online 版）

**ONDA** 的 running statistics update：

$$\mu_t = \rho\hat{\mu}_t + (1-\rho)\mu_{t-1}$$
$$\sigma_t^2 = \rho\hat{\sigma}_t^2 + (1-\rho)\frac{n_t}{n_t-1}\sigma_{t-1}^2$$

- $\frac{n_t}{n_t-1}$：Bessel correction，无偏估计
- 直觉：online 累积 BN 统计量，每个新 batch 都更新

**DUA** 用 decay 策略：$\rho$ 随 step 衰减，开始大幅更新 BN，后期逐渐稳定。**NOTE** 用 class-balanced memory bank 估计 BN，并只对 OOD sample 做 calibration——避免 in-distribution sample 的统计量被污染。

### 5.2 Entropy Minimization（Tent 系）

**Tent** 的核心：

$$\min_\theta \frac{1}{B}\sum_{i=1}^B \mathcal{H}\big(p_\theta(y|x_i)\big) = -\sum_c p_\theta(y_c|x_i)\log p_\theta(y_c|x_i)$$

- 只更新 BN 层的 affine 参数 $\{\gamma, \beta\}$
- 直觉：BN 的 affine 是最 cheap 但最 domain-sensitive 的参数。γ, β 决定 activation 的 scale 和 shift，对 domain style 敏感。
- Tent 之所以只动 BN affine：参数量小（几百 K），不容易过拟合；不动 encoder 主干，保留 source knowledge

**SAR（Sharpness-Aware Reliable entropy minimization）**：

$$\min_\theta \max_{\|\Delta_\theta\|_2 \leq \epsilon} \mathcal{H}(x; \theta + \Delta_\theta)$$

- $\Delta_\theta$：参数空间的 perturbation，在 $\epsilon$-ball 内
- 直觉：标准 entropy min 容易陷入 sharp minima，对小扰动敏感；SAM-style 的 minimax 让解在 flat 区域，提升对 future distribution shift 的 robustness
- "Reliable"：丢弃 gradient 大的样本（noise sample）

**EATA**：sample-efficient entropy minimization
- 只用 entropy 低于阈值的 sample 更新模型（reliable samples）
- Fisher regularizer：$\sum_i F_i (\theta_i - \theta_i^0)^2$，$F_i$ 是 Fisher information，对 important 参数施加更大惩罚
- 直觉：reliable sample 提供稳定 supervision；Fisher regularizer 防止 important 参数 drift 太远导致 source knowledge forgetting

### 5.3 Pseudo-labeling

**T3A**：完全 training-free，只调 classifier
- 维护 class prototype $m_c$（用 online unlabeled data 的 confident prediction 加权累积）
- 预测时用 sample 到 prototype 的距离

**Conjugate-PL**：用 convex conjugate 推导出"共轭伪标签"——形式上像 self-training，但 label 是从 loss 的 conjugate function 推出来的，理论上有更好的 convergence 性质。

### 5.4 Consistency Regularization

**CoTTA**（Continual TTA）：
- Teacher-student 框架 + weight stochastic restoration
- 每步以概率 $p_r$ 把部分参数 restore 回 source 初始值 $\theta^0$：
  $$\theta_i \leftarrow \begin{cases} \theta_i^0 & \text{with prob } p_r \\ \theta_i & \text{otherwise}\end{cases}$$
- 直觉：在 continual 设定下，每个 batch 可能来自不同 domain，模型容易 drift；stochastic restoration 像 regularizer，强制模型不偏离 source 太远

**AdaODM**：maximum classifier discrepancy，两个 classifier 输出差异最小化 feature encoder 更新。

### 5.5 Anti-forgetting Regularization

OTTA 的核心挑战：accumulated error + catastrophic forgetting。三大策略：

1. **保留小份 source data** 做 replay regularization（PAD、RMT）
2. **少参数更新**：Tent 只动 BN affine，AUTO 只更新 last feature block
3. **Stochastic restoration / Fisher regularizer / Self-distillation**（CoTTA, EATA, EcoTTA）

**EcoTTA** 的 self-distillation：
$$\mathcal{L} = \mathcal{L}_{ent} + \lambda \text{KL}(p_\theta(y|x) \| p_{\theta^0}(y|x))$$
- $\theta^0$：source frozen 模型
- 直觉：让 adapted 模型的预测不能离 source 太远，防止 drift

---

## 6. 实验数据表的关键观察

survey 里提到几个关键 benchmark：
- **CIFAR-10-C / CIFAR-100-C / ImageNet-C**：corruption robustness，15 种 corruption × 5 severity
- **ImageNet-R**：natural rendition shift（art, cartoon, deviantart）
- **ImageNet-A**：adversarial hard samples
- **Office-Home / VisDA-C / DomainNet**：DA 经典 benchmark
- **CIFAR-10.1**：unknown natural distribution shift

经验性结论：
1. Tent 在 CIFAR-100-C 上能把 ResNet 的 error 从 ~70% 降到 ~30%（仅更新 BN affine）
2. SHOT 在 Office-Home 上 average accuracy ~71%→~82%，逼近 supervised DA 上限
3. CoTTA 在 continual CIFAR-100-C 上比 Tent 稳定得多，避免后期 collapse
4. TTT 在 corrupted MNIST/CIFAR 上比 no-adapt 提升 5-10%

---

## 7. 关键 open problems

1. **Theoretical analysis 不足**：deep model 上 TTA 为什么 work？gradient flow 和 information theory 的分析还很初步（参考 https://arxiv.org/abs/2310.20199, https://arxiv.org/abs/2307.03133）
2. **Validation 困境**：test 时没有 labeled validation set，hyperparameter 怎么选？现有 paper 普遍 grid search on test data——unfair（参考 https://arxiv.org/abs/2307.03133 的 benchmarking）
3. **Continual / non-stationary**：CIFAR-100-C 上 sequential corruption，模型对后期 corruption 性能严重下降
4. **Foundation models**：CLIP、Stable Diffusion 时代的 TTA——TPT、PromptAlign、SwapPrompt 等用 prompt tuning 做 TTA（参考 https://arxiv.org/abs/2209.07511, https://arxiv.org/abs/2210.08813）
5. **Trustworthiness**：TTA 模型容易 overconfidence（参考 https://arxiv.org/abs/2303.12267）；backdoor 攻击风险（source 提供方可能植入后门，TTA 后激活）；class-wise fairness 缺失
6. **Tableular / Time series 数据**：deep learning 在 tabular 上不及 XGBoost，TTA 怎么迁移到 tree-based model？
7. **Memory-efficient continual adaptation**：端侧部署时 memory 受限，MECTA、EcoTTA 是早期尝试

---

## 8. Intuition 总结

把 TTA 看成 **"用 test data 的 unlabeled structure 信息修正 source model 的 domain-specific 假设"**：

- **BN statistics** 是 domain style 的 fingerprint → 直接 estimate / interpolate
- **Prediction entropy** 是 model confidence 的 proxy → minimize 推向 confident region
- **Local neighborhood in feature space** 揭示 cluster structure → consistency regularization
- **Marginal class distribution** 提供 prior → mutual information maximization 防 collapse
- **SSL task** 是 domain-agnostic 的 supervision → test-time SSL objective 拉近 feature

三个 setting 的核心差异：
- TTDA 能 multi-epoch 看到 whole domain → 可以做复杂 clustering、prototype 学习
- TTBA 只看一个 batch → 必须 lightweight（BN affine、prompt、input transform）
- OTTA 是 streaming + cumulative → 必须处理 forgetting + drift

详细的 awesome list 见 https://github.com/tim-learn/awesome-test-time-adaptation ，作者持续维护。如果对 SHOT 的具体实现感兴趣，可以参考 https://github.com/tim-learn/SHOT ，代码很 clean，是理解 SFDA 的最佳起点。
