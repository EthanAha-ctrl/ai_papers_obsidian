---
source_pdf: Self-Improving World Modelling with Latent Actions.pdf
paper_sha256: 33c07c3662131c629bfd61ddb7efd95124e511f70281a4cccb0ce1a6829cc866
processed_at: '2026-08-12T04:49:57-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SWIRL

## 1. 这篇 paper 在解决什么问题？

想象你在训练一个 AI，让它能预测"做了某件事之后世界会变成什么样"。比如给它一张照片和一句"把瓶子倒过来"，它应该能生成瓶子倒过来的新照片。这就是 **Forward World Modelling (FWM)**。

要训这样的 model，传统做法是准备一堆三元组：`[起始状态, 动作, 结果状态]`。但现实中动作标注太贵了——你想训一个能预测"切菜后下一帧"的视频 model，得有人一帧帧标"这里在切洋葱""那里在翻炒"。

SWIRL 的 insight：**我们手头有海量无标注的视频帧对（前帧、后帧），动作是 latent 的**。能不能让 model 自己把这个 missing 的 action 推出来，然后用它来训 forward model？

参考背景：Qiu et al. 2025 的 bootstrapping 工作已经尝试了类似思路 https://arxiv.org/abs/2506.06006 ，SWIRL 把它升级成一个完整的 self-improving cycle。

---

## 2. 核心 insight：两个 model 互相当老师

打一个比方：

- **FWM (Forward World Model)** 像个 **导演**：拿到剧本和动作描述，拍出下一帧画面
- **IDM (Inverse Dynamics Model)** 像个 **影评人**：看前后两帧画面，猜中间发生了什么动作

正常监督学习里，你需要一个外部"剧本作家"提供 ground truth 动作。SWIRL 说：**让导演和影评人互相教书**。

具体怎么互相教？

**Phase I（教导演）**：影评人看真实的 (前帧, 后帧) 猜一个动作 $z$，把这个 $z$ 喂给导演。导演根据 $z$ 拍出 G 个版本的"下一帧" $\hat{y}_1, \hat{y}_2, ..., \hat{y}_G$。然后影评人看每个 $\hat{y}_k$，反推它觉得这个画面中间发生了什么。如果反推出来的动作和当初给的 $z$ 一致，说明导演拍得"信息量足够"，能让人看懂发生了什么。这个一致性得分就是给导演的 reward。

公式化：
$$R_k = \log Q_\phi(z \mid x, \hat{y}_k)$$

- $x$: 真实前帧
- $z$: 影评人从真实 (x, y) 猜出的动作
- $\hat{y}_k$: 导演的第 k 个 rollout
- $Q_\phi$: 影评人 model，给一对 (x, $\hat{y}_k$) 输出动作分布
- $R_k$ 越大说明导演拍得越"可识别"

**Phase II（教影评人）**：现在轮到影评人学习了。给定真实 (前帧, 后帧)，影评人采样 G 个 candidate 动作 $z_1, ..., z_G$。每个 $z_k$ 喂给导演，看导演按这个 $z_k$ 拍出来的画面和真实后帧 $y$ 接近不接近。接近的话说明这个 $z_k$ 是个好的"动作解释"。

公式化：
$$R_k = \log P_\theta(y \mid x, z_k)$$

- $y$: 真实后帧
- $z_k$: 影评人猜的第 k 个候选动作
- $P_\theta$: 导演 model，给 (x, $z_k$) 输出 next state 分布
- $R_k$ 越大说明这个 $z_k$ 越"能解释观测到的转换"

两个 phase 交替跑，影评人越来越准、导演越来越会拍，互相 bootstrap。这就叫 **reciprocal cycle**，类似 AlphaZero 里 policy network 和 value network 互相 self-play（https://www.nature.com/articles/nature24270）。

---

## 3. 为什么这么做在数学上是对的？

Paper 给了两个定理证明两个 phase 各自对应一个 well-defined objective 的 lower bound 最大化。

### Phase I: 最大化 Conditional Mutual Information

**直觉**：导演要拍得"信息量足"，意思是不同的动作 $z$ 应该导致可区分的画面 $\hat{y}$。如果两个不同 $z$ 拍出来的画面一模一样，那看画面就反推不出动作，这叫"信息丢失"。

数学上度量这个就是 **Conditional Mutual Information**：
$$I(Z; \hat{Y} \mid X) = H(Z \mid X) - H(Z \mid \hat{Y}, X)$$

- $I(Z; \hat{Y} \mid X)$: 给定 $x$ 的条件下，$z$ 和 $\hat{y}$ 之间的 mutual information
- $H(Z \mid X)$: 已知 $x$ 时 $z$ 的不确定性（常数，不依赖 $\theta$）
- $H(Z \mid \hat{Y}, X)$: 已知 $x$ 和生成的 $\hat{y}$ 时 $z$ 的不确定性——**这个要尽量小**

最小化 $H(Z \mid \hat{Y}, X)$ 就是说"看了 $\hat{y}$ 之后 $z$ 越确定越好"。

但真实 posterior $P_\theta(z \mid \hat{y}, x)$ 算不出来（要 marginalize 整个 action space）。用 frozen 的 IDM 当 variational 近似（这是 Barber & Agakov 2003 的经典 IM algorithm trick https://papers.nips.cc/paper/2003/hash/ddbb9e2a1a63ce5f3e3f24e7304842dc-Abstract.html）：

$$-H(Z \mid \hat{Y}, X) \geq \mathbb{E}_{x, z, \hat{y}} [\log Q_\phi(z \mid x, \hat{y})]$$

这正好就是 Phase I 的 reward 期望值。所以 Phase I 在做 **variational information maximization**。

### Phase II: 最大化 ELBO

**直觉**：影评人猜的动作 $z$ 应该能解释观测到的真实转换 $y$，但又不能乱猜——要靠近"上一轮的信念"。

数学上就是经典 ELBO：
$$\log P_\theta(y \mid x) \geq \mathbb{E}_{z \sim Q_\phi}[\log P_\theta(y \mid x, z)] - D_{KL}(Q_\phi(z \mid x, y) \| \pi_{\text{ref}}(z \mid x))$$

- $\log P_\theta(y \mid x)$: 真实转换的 log-likelihood（要 maximize）
- 第一项: 影评人采的 $z$ 让导演能复现 $y$ 的期望 log-likelihood
- $D_{KL}$: 影评人当前分布不能离上一轮 $\pi_{\text{ref}}$ 太远，防止 collapse
- $\pi_{\text{ref}}(z \mid x)$: 上一轮的 IDM 当 prior

这和 KL-regularized policy gradient 完全等价（设 $\beta=1$, $R=\log P_\theta$）。所以 Phase II 在做 **coordinate ascent on ELBO**。

---

## 4. 训练流程的具体细节

### 4.1 Warm-up 必不可少

直接 RL 不行。Liquid-7B 这个 base VLM zero-shot 在 GEDIT-BENCH 上得分 0.03（基本废了），rollout 探索空间里几乎全是 invalid 输出，GRPO 算不出有意义的 advantage。

所以先用 PICO-BANANA-400K（https://arxiv.org/abs/2510.19808）+ AURORA（https://arxiv.org/abs/2407.03471）做 SFT warm-up，让 model 至少能输出"格式正确的图像编辑指令"和"格式正确的编辑后图像"。warm-up 后 GEDIT-BENCH 涨到 3.06，具备基本 capability 后才能进 RL 阶段。

这和 reasoning model RL 一样的限制——纯 from-scratch RL 极难启动，需要 SFT 提供最小 capability。

### 4.2 两个 phase 交替

**Algorithm 1 简化版**：
```
repeat:
    # Phase I: 训练 FWM, 冻结 IDM
    for batch (x, y) in D:
        z = IDM.sample(x, y)           # 用 IDM 从真实 transition 猜 action
        rollouts = FWM.sample(x, z, G)  # FWM 生成 G 个候选 next state
        rewards = [log IDM(z | x, r) for r in rollouts]  # IDM 当裁判打分
        FWM.update(GRPO, rollouts, rewards)
    
    # Phase II: 训练 IDM, 冻结 FWM
    for batch (x, y) in D:
        z_candidates = IDM.sample(x, y, G)  # IDM 生成 G 个候选 action
        rewards = [log FWM(y | x, z) for z in z_candidates]  # FWM 当裁判打分
        IDM.update(GRPO, z_candidates, rewards)
```

GRPO (Group Relative Policy Optimization) 来自 DeepSeek-R1（https://arxiv.org/abs/2501.12948）。它的核心是 group-relative advantage：同一组 G 个 rollout 内部做 normalization，不需要训练单独的 value function。这对生成任务特别友好，因为 value 估计在高维输出空间里方差巨大。

公式：
$$A_k = R_k - \frac{1}{G}\sum_{j=1}^{G} R_j$$

- $A_k$: 第 k 个 rollout 的 advantage
- $R_k$: 第 k 个 rollout 的 reward
- 减去组内均值，消除 reward scale 的 absolute bias

### 4.3 关键超参（Paper Appendix B）

- **GRPO rollout size G**: 16 是 sweet spot。G=8 太小方差大；G=64 性能略好但 compute 翻 4 倍
- **KL coefficient β**: 0.1。Phase II 中理论上要 β=1 才严格等价 ELBO，但实际 G=0.1 更稳
- **Learning rate**: Phase I 用 1e-6, Phase II 用 2e-7。Phase II 更敏感因为 IDM 输出离散 text token，lr 太大会破坏 prior
- **Temperature**: 0.75 + top-p 0.96。要保留足够 exploration，否则 GRPO 没有信号
- **Logit processor**: 强制 FWM 只生成 image token, IDM 只生成 text token。这避免 model 跨 modality 乱输出

---

## 5. 实验效果一览

### 5.1 AURORA-BENCH (Table 1)

这是核心 single-turn visual dynamics benchmark。

| Method | MagicBrush | Action Genome | Something | Whatsup | Kubric | Average |
|--------|-----------|---------------|----------|---------|--------|---------|
| Liquid-SFT | 5.46 | 2.76 | 3.00 | 3.60 | 7.00 | 4.36 |
| Liquid-Bootstrap | 5.27 | 3.02 | 2.86 | 3.88 | 5.57 | 4.11 |
| Liquid-SFT + Test-time Verification (N=8) | 6.18 | 3.28 | 3.74 | 4.48 | 6.28 | 4.77 |
| **SWIRL (IDM→FWM)** | 6.48 | 3.58 | 3.44 | 4.08 | 6.59 | 4.83 |
| **SWIRL (Iterative)** | 6.62 | 3.52 | 3.96 | 4.32 | 6.96 | **5.06** |
| BAGEL-14B | 8.14 | 6.72 | 6.90 | 5.40 | 5.82 | 6.44 |
| OmniGen2 | 7.04 | 4.96 | 6.00 | 6.60 | 5.54 | 6.05 |
| UniWorld-V1 | 7.42 | 7.06 | 7.37 | 8.20 | 6.76 | 7.36 |

**要点**：
- SWIRL (Iterative) 比 Liquid-SFT 提升 **16.1%** (4.36→5.06)
- 比单纯 IDM→FWM 单轮 (4.83) 还高，证明 self-improving 真的有用
- 比 test-time verification (N=8, 4.77) 高，意味着 SWIRL 的收益是 training-time 内化的，不靠 inference-time compute
- 比更大或更重监督的 BAGEL-14B / OmniGen2 / UniWorld-V1 在部分子集上 competitive

### 5.2 BYTEMORPH (Table 2)

测 fine-grained motion dynamics。

| Method | Camera Zoom | Camera Motion | Object Motion | Human Motion | Interaction | Average |
|--------|-------------|---------------|----------------|--------------|-------------|---------|
| Liquid-SFT | 57.23 | 56.51 | 43.13 | 38.07 | 40.70 | 43.38 |
| SWIRL (IDM→FWM) | 57.37 | 50.13 | 62.50 | 48.69 | 56.53 | 54.57 |
| SWIRL (Iterative) | 54.08 | 58.22 | 58.69 | 48.08 | 54.50 | 53.77 |
| SWIRL (Iter.+Share) | 53.16 | 55.86 | 62.10 | 53.78 | 53.81 | 55.72 |

**要点**：
- Object/Human Motion 上 SWIRL 提升巨大（+45% on Object Motion）
- Camera 控制几乎无提升——因为 VIDGEN-1M 视频大多是 static camera，没有 supervisory signal
- 共享参数 (Iter.+Share) 反而比 separate 略好（55.72 vs 53.77），但 AURORA 上 separate 更好——共享对 fine-grained motion 有帮助，对复杂 reasoning 有害

### 5.3 WORLDPREDICTIONBENCH (Table 3) 长程预测

测 T=1..6 autoregressive rollout。

| Model | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 |
|-------|-----|-----|-----|-----|-----|-----|
| Liquid-SFT | 3.09 | 2.09 | 1.40 | 1.17 | 1.07 | 0.97 |
| SWIRL (IDM→FWM) | 3.24 | 2.42 | 1.85 | 1.59 | 1.25 | 1.08 |
| SWIRL (Iterative) | 3.23 | 2.08 | 1.53 | 1.32 | 1.15 | 1.11 |
| BAGEL | 4.29 | 4.10 | 3.47 | 3.22 | 2.47 | 2.23 |

**要点**：
- Liquid-SFT 在 T=6 掉到 0.97，几乎是 random
- SWIRL (IDM→FWM) 单轮效果最好，T=6 达到 1.08
- **Iterative 反而不如单轮**——这是个有趣的发现，paper 没深入解释。我的猜测：iterative 让 FWM 越来越 narrow on single-step likelihood，multi-step error compounding 加快。需要 trajectory-level reward 才能解

### 5.4 STABLETOOLBENCH (Table 4) 文本世界

测 LLM 在 tool calling 上的 dynamics prediction。

| Model | ID-High | ID-Low | ID-Med | OOD | OOD-Fail | Average |
|-------|---------|--------|--------|-----|----------|---------|
| Qwen-2.5-3B-Instruct | 7.74 | 7.74 | 9.48 | 6.59 | 3.18 | 6.95 |
| Qwen-2.5-32B-Instruct | 7.13 | 7.13 | 10.22 | 7.75 | 4.34 | 7.31 |
| DeepSeek-7B-Chat | 5.38 | 15.17 | 4.44 | 8.52 | 4.17 | 7.54 |
| Qwen-2.5-3B-SFT | 16.03 | 12.87 | 17.51 | 14.86 | 2.99 | 12.85 |
| **Qwen-2.5-3B-SWIRL** | 16.47 | 16.90 | 21.20 | 15.57 | 2.92 | **14.61** |

**要点**：
- 3B model + SWIRL (14.61) **超过 32B model** (7.31) 一倍
- ID-Low +31%，ID-Med +21%
- 这是最强 evidence：对 structured dynamics learning，self-improving RL 比 scaling up parameters 高效得多

### 5.5 RL vs SFT (Figure 3, Table 7) 最强 ablation

同一批 unlabelled video，用 IDM pseudo-label 后比较 SFT vs RL。

| Sample # | SFT-Continue | SFT-Merge | SWIRL (RL) |
|----------|--------------|-----------|------------|
| 3.2K | 3.98 | 4.15 | 4.27 |
| 6.4K | 3.94 | 4.07 | 4.63 |
| 9.6K | 3.84 | 3.95 | 4.67 |
| 12.8K | 3.88 | 4.01 | 4.73 |

**关键 insight**：SFT 随数据增多**反而下降**，SWIRL 单调上升。

为什么？因为 visual dynamics 的 inverse dynamics 是 **ambiguous**——同一个 transition 可以用多种 valid action verbalisation 描述。比如"把杯子放左边"和"将杯子向左移动"都对。SFT 强制 token-level 模仿 IDM 的某个 specific verbalisation，相当于 overfit noise、压制 valid alternative。GRPO 的 reward 是"IDM 能否反推 z"，鼓励 consistency 不强制 replication，让 FWM 探索整个 rollout 空间，更鲁棒。

这个 insight 对所有 latent-variable self-training 都适用：**当 teacher 输出多模态时，distillation 用 RL 比 SFT 好**。

### 5.6 Latent Action 不塌缩 (Figure 6)

担心 reward hacking——IDM 学到把所有 transition 都映射到同一个 short cipher（比如 "do"），FWM 就 easy 优化。

实测：
- **Uniqueness > 94%** 跨所有 iterations
- **PPL ratio ≈ 1**——GPT-2 perplexity 和 ground truth 接近
- **Average action length 16.8 tokens**，超过 ground truth 的 7.2 tokens——IDM 反而生成更详细的描述来确保可识别性

这个 emergent behavior 很有意思：reciprocal cycle 中 IDM 自然倾向"过度描述"以保证 FWM 能从描述提取足够信息。KL term + CMI maximization 共同起作用。

---

## 6. 和其他框架的关系

| 框架 | 与 SWIRL 关系 |
|------|--------------|
| **EM algorithm** | E 步 ≈ Phase II（infer latent z），M 步 ≈ Phase I（update generative model）。但 SWIRL 两步都是 RL |
| **Wake-Sleep** (Hinton 1995, https://www.cs.toronto.edu/~hinton/absps/wakesleep.pdf) | Wake ≈ Phase II（update recognition），Sleep ≈ Phase I（update generative）。SWIRL 把 Sleep 阶段的 reconstruct 换成 InfoMax |
| **AlphaZero self-play** (https://www.nature.com/articles/nature24270) | Policy network 和 value network 互相 bootstrap。SWIRL 是 generative modelling 版的 self-play，FWM 和 IDM 互相 bootstrap |
| **CPC** (https://arxiv.org/abs/1807.03748) | 都在最大化 representation 和 future 的 mutual information，但 SWIRL 用 latent action 而非 context vector |
| **rStar-Math** (https://arxiv.org/abs/2501.04519) | Generator + verifier self-improve on reasoning chain。SWIRL 把 idea 推到 world modelling |
| **Dreamer** (https://danijar.com/dreamer/) | 学 latent world model 用于 planning。SWIRL 只学 transition model，但 latent action 是 explicit natural language，可以直接用作 action space |

---

## 7. 限制和开放问题

### 7.1 Long-horizon vs Iteration 的 trade-off

WORLDPREDICTIONBENCH 上 Iterative < 单轮。Paper 没解释。我的猜测：iteration 让 FWM 越来越 narrow on single-step likelihood，多步累积误差更快。可能解法：把 multi-step rollout 也放进 reward，或者 trajectory-level GRPO。这是最有意思的 open direction。

### 7.2 Shared weights 不稳定

Figure 2 右图，共享参数 Iter 3 反而退化。FWM 生成 image token、IDM 生成 text token，两个 modality 的 gradient 在共享 backbone 上互相 interfere。这和 unified VLM 中 understanding > generation 的 imbalances 现象同源（https://arxiv.org/abs/2509.24897）——multi-modal multi-task 在统一参数下优化冲突没解决。

### 7.3 Warm-up 仍需 SFT

完全 zero-shot 启动 SWIRL 不可能。random policy rollout 无法 cover valid state，GRPO 失效。这和 reasoning model 的 RL-from-scratch 限制一样。如何用更轻量的 warm-up（few-shot prompting + self-distillation）替代大规模 SFT 是工程方向。

### 7.4 Theoretical gap

Theorem 3.1 / 3.2 给了每一步的 learnability，但**没有证明整体 coordinate ascent 收敛到 global optimum**。Phase I 和 Phase II 互相提供 reward signal，可能存在 self-reinforcing bias——两个 model 共同固化错误 belief。这是 EM 也会有的 local optimum 问题，但 SWIRL 因为 RL 的 stochasticity 可能更严重。

### 7.5 Camera control 弱

BYTEMORPH camera 子项几乎无提升，因为 VIDGEN-1M 多是 static camera。需要混合 dataset (ego-centric / drone footage) 来补足。

---

## 8. 我的整体 takeaway

### 8.1 这篇 paper 的真正贡献

不是单个 trick，而是提供了一个 **latent variable model 的 RL 训练范式**：当你的数据缺某个变量（action），但有两个 complementary 任务（forward 和 inverse）时，让它们互为 reward 信号，用 RL 而非 SFT 交替优化。

理论上对应两个 lower bound：
- Phase I: Variational Mutual Information Lower Bound (Barber-Agakov 2003)
- Phase II: Evidence Lower Bound (经典 VAE)

这种 "InfoMax + ELBO" 双 lower bound 框架在 latent variable model 文献里并不新，但 SWIRL 第一次把它和 GRPO 结合、用于现代 LLM/VLM 的 world modelling。

### 8.2 最强的 evidence

3B Qwen + SWIRL 在 STABLETOOLBENCH 上超过 32B Qwen 一倍。这说明对 structured dynamics learning，self-improving RL 比 scaling up parameters 高效得多。对资源有限的研究组是关键 insight：不需要从头训 14B VLM，7B + SWIRL 即可。

### 8.3 最深的 insight

**当 teacher 输出多模态时，distillation 用 RL 比 SFT 好**。SFT 强制 token-level replication 会 overfit noise、压制 valid alternative；RL 只要求 consistency，让 student 探索整个 rollout 空间。这个 insight 对所有 self-distillation / synthetic data training 都适用——未来 RLHF 替代 SFT 可能是主流方向。

### 8.4 推广空间

理论上可以套到任何 "observation sequence with hidden cause" 的场景：
- 视频帧 + latent physical force → force prediction + next frame prediction
- 用户行为 + latent intent → intent inference + next action prediction  
- Code change + latent refactor intent → intent prediction + next code state prediction
- Medical records + latent disease progression → progression inference + next record prediction
- Financial time series + latent market regime → regime inference + next return prediction

只要你能定义 forward model 和 inverse model，并且 forward model 输出维度高、inverse model 输出相对低维，SWIRL 的 reciprocal cycle 就可能适用。

### 8.5 更深层的哲学

SWIRL 本质上把 **AlphaZero self-play** 思想搬到 generative world modelling。AlphaZero 里 policy network 和 value network 互相 bootstrap，这里换成 FWM 和 IDM。两者的共同点：**没有一个外部的 ground truth teacher，靠两个 complementary model 的 mutual specialization 进化**。

这暗示了一种新的 AI 训练范式：**未来越来越少的 ground truth supervision，越来越多 self-play / reciprocal cycle**。当 ground truth 标注成本爆炸时，让 model 互相当老师是唯一 scalable 的路径。

参考：
- SWIRL 原文：基于 https://arxiv.org/abs/2506.06006 (Qiu et al. 2025 bootstrap) 升级
- GRPO: https://arxiv.org/abs/2501.12948
- Liquid VLM: https://arxiv.org/abs/2412.04332
- VIDGEN-1M: https://arxiv.org/abs/2408.02629
- AURORA-BENCH: https://arxiv.org/abs/2407.03471
- STABLETOOLBENCH: https://arxiv.org/abs/2403.07714
- rStar-Math: https://arxiv.org/abs/2501.04519
- Dreamer: https://danijar.com/dreamer/
- Wake-Sleep: https://www.cs.toronto.edu/~hinton/absps/wakesleep.pdf
- AlphaZero: https://www.nature.com/articles/nature24270
- CPC: https://arxiv.org/abs/1807.03748
- Barber-Agakov IM: https://papers.nips.cc/paper/2003/hash/ddbb9e2a1a63ce5f3e3f24e7304842dc-Abstract.html

如果你想再深入某个点（比如 GRPO 在 multi-modal 输出上的具体实现细节、long-horizon 退化的 root cause、或者把 SWIRL 推广到 continuous action space 的可能路径），我可以再展开。

---

# SWIRL: Self-Improving World Modelling with Latent Actions —— 深度解析

## 1. 背景与 Motivation：World Model 的数据瓶颈

LLMs / VLMs 已经在 pretraining 阶段 implicit 地内化了 world model，比如代码执行预测、工具调用结果预测、空间推理等都展现出某种程度的 dynamics 内在表示。要让这种 capability 显式化、强化它，主流路径是收集 `(state_t, action_t, state_{t+1})` 三元组的 trajectories 来 SFT，比如 Coding World Model (Copet et al., 2025)、Tool-Use Outcomes (Guo et al., 2025b) 都走这条线。

**问题**：在 open-world 场景下（visual dynamics、web navigation、复杂工具调用），annotating 每一个 transition 的 action 代价过高。另一方面，inverse dynamics 本身 inherent ambiguous——从 `state_t` 到 `state_{t+1}` 之间可能存在多个合法的 action，纯监督学习在数据稀疏时会 brittle。

**关键 insight**：我们手头有海量 state-only sequences（视频帧对、HTML 转换日志、对话历史），action 是 latent variable。这就构成了一个 **latent variable inference + world modelling 的耦合问题**，需要一种 self-improving 的循环优化机制。

Paper 的核心贡献：把 forward world modelling (FWM) 和 inverse dynamics modelling (IDM) 当作两个互相 teacher / student 切换的角色，用 GRPO 交替优化，理论上是 Conditional Mutual Information lower bound 和 ELBO 的交替 ascent。

参考：
- Qiu et al. 2025 bootstrapping world models: https://arxiv.org/abs/2506.06006
- Barber & Agakov 2003 IM algorithm: https://papers.nips.cc/paper/2003/hash/ddbb9e2a1a63ce5f3e3f24e7304842dc-Abstract.html
- GRPO (DeepSeek-R1): https://arxiv.org/abs/2501.12948

---

## 2. 问题形式化

### 2.1 两个组件

| 符号 | 含义 |
|------|------|
| $x \in \mathcal{S}$ | source state (current state $s_t$) |
| $y \in \mathcal{S}$ | target state (next state $s_{t+1}$)，服从数据分布 $\mathcal{D}(y \mid x)$ |
| $\hat{y} \in \mathcal{S}$ | FWM 生成的预测 next state |
| $z \in \mathcal{A}$ | latent action（驱动 state transition 的 action） |
| $P_\theta(\hat{y} \mid x, z)$ | FWM，参数 $\theta$ |
| $Q_\phi(z \mid x, y)$ | IDM，参数 $\phi$ |

数据集 $\mathcal{D} = \{(x_i, y_i)\}$ 只有 state pairs，**没有 action labels**。

### 2.2 四类 environment

1. **Real-world visual**: observations 是 pixel-level images，actions 是 natural language → 用 unified VLM (Liquid-7B) 处理 interleaved image-text sequence
2. **Synthetic textual**: observations 和 actions 都是 language，由 simulator 控制 → 用 LLM (Qwen-2.5-3B)
3. **Web HTML**: states 是 HTML DOM，actions 是 interaction logs (clicks, etc.) → LLM
4. **Tool use**: states 是对话上下文+工具执行结果，actions 是 tool calls → LLM

---

## 3. SWIRL 方法：互为 Policy / Reward 的 Reciprocal Cycle

### 3.1 Intuition

整个框架的核心直觉可以总结为两条互相强化的约束：

- **Identifiability (Phase I)**：FWM 生成的 $\hat{y}$ 必须能被 IDM "回溯识别"——给定 $(x, \hat{y})$，IDM 推断出的 $\hat{z}$ 应该接近当初喂给 FWM 的 $z$。这本质上要求 FWM 把不同的 $z$ 映射到可区分的 $\hat{y}$，即 $z$ 和 $\hat{y}$ 之间的 conditional mutual information 大。
- **Data fidelity (Phase II)**：IDM 推断的 $z$ 必须能 "向前验证"——把 $(x, z)$ 喂给 FWM，应该能生成接近真实 $y$ 的样本。这是 ELBO 最大化，逼 IDM 找到的 latent action 真的能解释观测到的 transition。

这是一个典型的 **coordinate ascent on a coupled objective**，类似 EM 的 E-M 切换，但 E 和 M 步都换成了 GRPO。

### 3.2 Algorithm 1 逐步解析

```
Input:  D = {(x_i, y_i)} unlabelled
Init:   P_θ (FWM), Q_φ (IDM)   — 都从 SFT-warmed 模型开始
Hyper:  group size G, learning rates η_θ, η_φ
repeat
  === Phase I: optimize FWM, freeze φ ===
  for each batch (x, y) ~ D:
    z ~ Q_φ(·|x, y)              # 用 IDM 从真实 transition 推断 latent action
    {ŷ_1, ..., ŷ_G} ~ P_θ(·|x, z)  # FWM 采样 G 个 rollout
    for k = 1..G:
      R_k = log Q_φ(z | x, ŷ_k)  # frozen IDM 给 FWM 的 rollout 打分
      A_k^F = group-relative advantage
    θ ← θ + η_θ ∇_θ [ (1/G) Σ_k A_k^F log P_θ(ŷ_k | x, z) ]
  
  === Phase II: optimize IDM, freeze θ ===
  for each batch (x, y) ~ D:
    {z_1, ..., z_G} ~ Q_φ(·|x, y)  # IDM 采样 G 个 candidate actions
    for k = 1..G:
      R_k = log P_θ(y | x, z_k)   # frozen FWM 给 IDM 的候选 action 打分
      A_k^I = group-relative advantage
    φ ← φ + η_φ ∇_φ [ (1/G) Σ_k A_k^I log Q_φ(z_k | x, y) ]   (+ KL term)
until convergence
```

这里有几个关键 design choice 值得细细品味：

**(a) 为什么 reward 是 log-probability 而不是 probability?**

Phase I 中 $R_k = \log Q_\phi(z \mid x, \hat{y}_k)$。这是把 IDM 当作一个 energy-based discriminator，log-prob 形式天然适配 GRPO 的 advantage 计算，且数值上更稳定。从信息论角度，$\log Q_\phi(z \mid x, \hat{y})$ 就是 cross-entropy 项的负值，最大化它等于最小化 IDM 从 $\hat{y}$ 反推 $z$ 的负对数似然。

**(b) Phase I 中先采 $z \sim Q_\phi(\cdot \mid x, y)$ 用的是真实 $y$，而 reward 用的 $Q_\phi(z \mid x, \hat{y}_k)$ 用的是 rollout $\hat{y}_k$**

这是 paper 里最微妙的地方。$z$ 不是从某个 fixed prior 采的，而是从 **empirical belief distribution** 采的：

$$\tilde{P}(z \mid x) \triangleq \mathbb{E}_{y \sim \mathcal{D}(y \mid x)} [Q_\phi(z \mid x, y)]$$

意思是：对每个 $x$，把数据集中所有 $(x, y)$ 的 IDM posterior 平均一下，得到 $z$ 的"经验信念分布"。这个分布反映了 IDM 当前认为哪些 action 在这个 $x$ 上是常见的。FWM 要在这些"合理的" $z$ 上做出可区分的 $\hat{y}$。

**(c) Phase II 中 reference policy $\pi_{\text{ref}}$ 是 informative prior**

ELBO 中的 prior $P(z \mid x) = \pi_{\text{ref}}(z \mid x)$ 设置为"本轮开始时的 IDM"。这避免了 IDM 漂移到 degenerate 分布（比如只输出一个固定 token），KL 项把 IDM 锚在前一轮的位置上。这就是为什么论文里观察到的 latent action 不会塌缩成 short cipher（参考 Figure 6 的 uniqueness > 94%, PPL ratio 稳定）。

---

## 4. 理论分析：变分信息论视角

### 4.1 Phase I: Variational Information Maximisation

**Theorem 3.1**: 优化 FWM 最大化 frozen IDM 对生成样本赋予的 log-probability，等价于最大化 $I_{\tilde{P}}(Z; \hat{Y} \mid X)$ 的 variational lower bound。

证明链路：

**Step 1: CMI 分解**

$$I_{\tilde{P}}(Z; \hat{Y} \mid X) = H_{\tilde{P}}(Z \mid X) - H(Z \mid \hat{Y}, X) \quad (1)$$

- $H_{\tilde{P}}(Z \mid X)$: latent action 的条件熵，仅依赖 frozen IDM 和数据 $\mathcal{D}$，对 $\theta$ 是常数
- $H(Z \mid \hat{Y}, X)$: 给定生成的 $\hat{Y}$ 后 $Z$ 的不确定性——这个越小越好，意味着 $\hat{Y}$ 充分携带 $Z$ 的信息

所以最大化 CMI ↔ 最小化 $H(Z \mid \hat{Y}, X)$。

**Step 2: 变分下界**

真实 posterior $P_\theta(z \mid \hat{y}, x)$ 需要 marginalize 整个 action space，intractable。用 frozen IDM $Q_\phi(z \mid x, \hat{y})$ 做变分近似（Barber & Agakov 2003 的 IM algorithm 经典 trick）：

$$-H(Z \mid \hat{Y}, X) = \mathbb{E}_{x, z \sim \tilde{P}, \hat{y} \sim P_\theta} [\log P_\theta(z \mid \hat{y}, x)] \geq \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{z \sim \tilde{P}(z \mid x)} \mathbb{E}_{\hat{y} \sim P_\theta(\cdot \mid x, z)} [\log Q_\phi(z \mid x, \hat{y})] \quad (2)$$

不等式来自 $D_{KL}(P_\theta \| Q_\phi) \geq 0$。

**Step 3: 展开 $\tilde{P}$**

代入 $\tilde{P}(z \mid x) = \mathbb{E}_{y \sim \mathcal{D}(y \mid x)} [Q_\phi(z \mid x, y)]$，得到最终目标：

$$\mathcal{I}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \mathbb{E}_{z \sim Q_\phi(\cdot \mid x, y)} \mathbb{E}_{\hat{y} \sim P_\theta(\cdot \mid x, z)} [\log Q_\phi(z \mid x, \hat{y})] \quad (3)$$

这正是 Algorithm 1 Phase I 做的事：采 $(x, y)$ → IDM 推 $z$ → FWM rollout $\hat{y}$ → IDM 打分 $\log Q_\phi(z \mid x, \hat{y})$。

**Step 4: GRPO estimator**

目标对 $\theta$ 求梯度：

$$\nabla_\theta \mathcal{I}(\theta) = \mathbb{E}[\log Q_\phi(z \mid x, \hat{y}) \nabla_\theta \log P_\theta(\hat{y} \mid x, z)]$$

GRPO 用 group-relative advantage 替换 raw reward：

$$\widehat{\nabla_\theta \mathcal{I}}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{k=1}^{G} A_k^{\text{F}} \nabla_\theta \log P_\theta(\hat{y}_k \mid x, z) \quad (4)$$

其中 $A_k^{\text{F}} = R_k - \bar{R}$，$R_k = \log Q_\phi(z \mid x, \hat{y}_k)$。Group-relative 关键在于**消除 baseline bias**，因为同一组内 reward 的绝对 scale 不可靠，但相对 ordering 是稳定的。

**Intuition**: 这个阶段在告诉 FWM——"你生成的 $\hat{y}$ 越能让 IDM 准确反推出当初的 $z$，就越好"。也就是逼 FWM 把 latent action 的信息"刻"到生成结果里，使得 action 和 outcome 之间形成可识别的映射。

### 4.2 Phase II: ELBO Maximisation

**Theorem 3.2**: 用 KL-regularized policy gradient 优化 IDM，reward $R = \log P_\theta(y \mid x, z)$，$\beta = 1$，reference policy $\pi_{\text{ref}}$，等价于最大化 ELBO。

**Step 1: 边际似然**

$$\log P_\theta(y \mid x) = \log \sum_{z \in \mathcal{A}} P_\theta(y \mid x, z) \pi_{\text{ref}}(z \mid x)$$

引入 variational posterior $Q_\phi(z \mid x, y)$：

$$\log P_\theta(y \mid x) = \log \mathbb{E}_{z \sim Q_\phi(\cdot \mid x, y)} \left[ \frac{P_\theta(y \mid x, z) \pi_{\text{ref}}(z \mid x)}{Q_\phi(z \mid x, y)} \right]$$

**Step 2: Jensen 不等式**

$$\log P_\theta(y \mid x) \geq \mathbb{E}_{z \sim Q_\phi} [\log P_\theta(y \mid x, z)] - D_{KL}(Q_\phi(z \mid x, y) \| \pi_{\text{ref}}(z \mid x)) \triangleq \mathcal{L}_{\text{ELBO}}$$

**Step 3: 对照 GRPO 目标**

KL-regularized policy gradient 目标：

$$\mathcal{I}(\phi) = \mathbb{E}_{z \sim Q_\phi} [R(x, z, y)] - \beta D_{KL}(Q_\phi(\cdot \mid x, y) \| \pi_{\text{ref}}(\cdot \mid x)) \quad (5)$$

设 $R(x, z, y) = \log P_\theta(y \mid x, z)$，$\beta = 1$，$P(z \mid x) = \pi_{\text{ref}}(z \mid x)$，则 $\mathcal{I}(\phi) \equiv \mathcal{L}_{\text{ELBO}}$。

**Intuition**: 这个阶段告诉 IDM——"你推断的 $z$ 越能让 FWM 复现真实 $y$，越好，但别离上一轮的自己太远"。前者保证 data fidelity，后者防止 collapse。

### 4.3 与经典框架的关系

| 框架 | 与 SWIRL 的关系 |
|------|----------------|
| **EM algorithm** | E 步对应 Phase II（infer latent z），M 步对应 Phase I（update generative model）。但 SWIRL 两步都是 RL 而非 closed-form |
| **Wake-Sleep algorithm** (Hinton 1995) | Wake 阶段更新 generative model (recognition model frozen)，Sleep 阶段更新 recognition model (generative frozen)——结构上几乎完全对应 SWIRL Phase I / Phase II |
| **Iterative Amortized Inference** (Mishkin et al. 2020) | 把 variational posterior 用 amortized network 表示，迭代优化；SWIRL 把这种思想带到 world modelling |
| **AlphaZero self-play** | 两个角色轮流当 policy 和 reward/critic，互相提升。SWIRL 是 generative modelling 版的 self-play |
| **CPC (Contrastive Predictive Coding)** | 都在最大化 representation 和 future 之间的 mutual information，但 SWIRL 用 latent action 而非 context vector |
| **ALICE / Adversarial IRL** | 也有 forward / inverse 的 dual structure，但 SWIRL 用 RL reward 而非 adversarial discriminator |

参考：
- Wake-Sleep: https://www.cs.toronto.edu/~hinton/absps/wakesleep.pdf
- Iterative Amortized Inference: https://arxiv.org/abs/2007.02508
- AlphaZero: https://www.nature.com/articles/nature24270
- CPC: https://arxiv.org/abs/1807.03748

---

## 5. 实验设置细节

### 5.1 Base models

- **VLM**: Liquid-7B (Wu et al., 2024) — autoregressive unified VLM。选它因为 7B 体量适中且 autoregressive，可以直接套 GRPO，不需要为 diffusion-based generation 改写算法。Paper 在 Appendix C 强调 Liquid 本身不支持 image editing (zero-shot GEDIT-BENCH score 仅 0.03)，必须先 SFT warmup。
- **LLM**: Qwen-2.5-3B-Instruct (Qwen Team 2024) — 中等规模的有竞争力 instruct model

### 5.2 SFT Warm-up

这是必须的步骤。如果直接 RL，random policy 生成的输出全是 invalid，rollout space 探索失败。Warm-up 用：
- VLM: PICO-BANANA-400K (https://arxiv.org/abs/2510.19808) + AURORA (https://arxiv.org/abs/2407.03471)
- LLM: 每个 environment-specific episode 的一半（保留另一半去掉 action label 用作 SWIRL 训练）

### 5.3 RL 训练阶段

**Controlled stage**: 用 UCF-101、Moments in Time、Kinetics700 的 unlabelled 视频 mixture，只跑 Phase I (IDM→FWM)，做严格对比 Qiu et al. 2025 的 bootstrapping baseline。

**Scaled stage**: 从 VIDGEN-1M (https://arxiv.org/abs/2408.02629) 每轮均匀采样 30K 视频，提取 frame pairs (Chen et al. 2025d 的方法)，每轮训一个 epoch，交替 FWM/IDM。

**超参**:
- SWIRL (IDM→FWM): batch=64, lr=1e-6, cosine + 100 warmup, GRPO rollout=8, β=0.1
- SWIRL (Iterative): batch=128, lr=2e-7, 50 warmup
- SWIRL (Iter.+Share): lr=5e-7
- Decoding: temperature=0.75, top-p=0.96
- Logit processor 限制 FWM 只生成 image token，IDM 只生成 text token

### 5.4 Benchmarks

| Benchmark | Type | Metric |
|-----------|------|--------|
| AURORA-BENCH | single-turn visual dynamics | GPT-4o-as-judge (10pt), DiscEdit, CLIP |
| BYTEMORPH | single-turn visual, fine-grained motion | GPT-4o-as-judge |
| WORLDPREDICTIONBENCH | multi-turn (T=1..6) | GPT-4o-as-judge per turn |
| SCIENCEWORLD | textual physics | BERTScore, ROUGE-L |
| MIND2WEB | web HTML | BERTScore, ROUGE-L |
| STABLETOOLBENCH | tool calling | BLEU (ID-High/Med/Low, OOD, OOD-Fail) |

---

## 6. 实验结果深度解读

### 6.1 AURORA-BENCH (Table 1)

- **Liquid-SFT**: 4.36 avg
- **SWIRL (IDM→FWM)**: 4.83 (+10.8%)
- **SWIRL (Iterative)**: 5.06 (+16.1%)
- **SWIRL (Iter.+Share)**: 5.00

对比 baselines:
- Liquid-Bootstrap (Qiu et al. 2025): 4.11
- Liquid-SFT w/ Test-time Verification (N=8): 4.77
- BAGEL-14B: 6.44 (但参数 2x)
- OmniGen2: 6.05
- UniWorld-V1: 7.36

**关键观察**：
1. SWIRL 超过 test-time verification (N=8) — 不靠 inference-time compute，靠 training-time 的 reciprocal cycle
2. 迭代版本超过单轮 (IDM→FWM)，证明 self-improving 确实有效
3. Shared weights (Iter.+Share=5.00) < Separate weights (5.06)，因为 FWM 生成 image token 和 IDM 生成 text token 的 gradient 在共享 backbone 上互相干扰
4. 在 MagicBrush 子集上 SWIRL Iterative 6.62 vs Liquid-SFT 5.46，提升 21%

### 6.2 BYTEMORPH (Table 2)

- Liquid-SFT: 43.38
- SWIRL (Iter.+Share): 55.72 (+28.3%)
- SWIRL (IDM→FWM): 54.57 (+25.7%)

分项分析：
- **Camera Zoom/Motion**: 提升有限 (54.08 vs 57.23 / 58.22 vs 56.51) — 因为 VIDGEN-1M 视频大多 static，对 camera 控制 supervisory signal 弱
- **Object/Human Motion + Interaction**: 大幅提升（62.50 vs 43.13，48.69 vs 38.07）— fine-grained dynamics 学得好

### 6.3 WORLDPREDICTIONBENCH 长程 (Table 3 & Table 8)

这是测试 temporal consistency 的核心 benchmark，T=1..6 autoregressive rollout。

| Model | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 |
|-------|-----|-----|-----|-----|-----|-----|
| Liquid-SFT | 3.09 | 2.09 | 1.40 | 1.17 | 1.07 | 0.97 |
| SWIRL (IDM→FWM) | 3.24 | 2.42 | 1.85 | 1.59 | 1.25 | 1.08 |
| SWIRL (Iterative) | 3.23 | 2.08 | 1.53 | 1.32 | 1.15 | 1.11 |
| BAGEL | 4.29 | 4.10 | 3.47 | 3.22 | 2.47 | 2.23 |

**洞察**：
- Liquid-SFT 在 T=1 还行 (3.09)，但 T=6 直接掉到 0.97 — compounding error 严重
- SWIRL (IDM→FWM) 在 T=6 达到 1.08，+11.4% 相对提升
- **奇怪现象**：Iterative 在长程上反而不如 IDM→FWM 单轮。Paper 解释：多轮迭代在 single-step 上更强，但可能 overfit 到 single-step transition，长程累积误差更敏感。这暗示 iterative self-improvement 和 long-horizon consistency 之间存在 trade-off，是值得深挖的方向。

### 6.4 Textual Environments (Table 4)

- SCIENCEWORLD: BERTScore 96.06 (SFT) → 96.06 (SWIRL)，几乎饱和
- MIND2WEB: BERTScore 92.37 → 92.44，也接近饱和
- **STABLETOOLBENCH**:
  - ID-Low: 12.87 → 16.90 (+31.3%)
  - ID-Med: 17.51 → 21.20 (+21.1%)
  - Average: 12.85 → 14.61 (+13.7%)

对比大模型：
- Qwen-2.5-32B-Instruct: 7.31
- OLMO-3-7B: 9.18
- DeepSeek-7B-Chat: 7.54

SWIRL 在 3B 模型上跑出 14.61，**超过 32B 模型**。这说明 world modelling 这种 structured dynamics 学习任务，self-improving RL 比单纯 scale up parameters 更高效。

### 6.5 Ablation: RL vs SFT (Figure 3, Table 7)

这是最有说服力的 ablation。同一批 unlabelled video，先用 IDM 标注成 pseudo-label，然后比较：
- SFT-Continue: 在原模型上 continue training
- SFT-Merge: 拼接所有样本一起 SFT
- SWIRL (RL): GRPO 优化

| Sample # | SFT-Continue | SFT-Merge | SWIRL (RL) |
|----------|--------------|-----------|------------|
| 3.2K | 3.98 | 4.15 | 4.27 |
| 6.4K | 3.94 | 4.07 | 4.63 |
| 9.6K | 3.84 | 3.95 | 4.67 |
| 12.8K | 3.88 | 4.01 | 4.73 |

**关键 insight**：SFT 随数据增加 performance 反而**下降或停滞**，SWIRL 单调上升。原因：visual dynamics 的 inverse dynamics 是 ambiguous——一个 transition 可以对应多种 valid action verbalisation。SFT 强制 token-level 模仿 IDM 的某个 specific verbalisation，会 overfit noise、压制 valid alternative。GRPO reward 是 "IDM 能否反推 z"，鼓励 consistency 不强制 replication，让 FWM 探索整个 rollout 空间，更鲁棒。

这个 insight 对所有 latent-variable self-training 都适用——**当 teacher 输出多模态时，distillation 用 RL 比 SFT 好**。

### 6.6 Ablation: GRPO Rollout Size (Table 10)

| G | MB | AG | Something | Whatsup | Kubric | Avg |
|---|-----|-----|-----|-----|-----|-----|
| 8 | 6.04 | 3.38 | 3.45 | 4.26 | 6.98 | 4.80 |
| 16 | 5.78 | 3.24 | 3.60 | 4.50 | 6.98 | 4.80 |
| 32 | 5.72 | 3.30 | 3.37 | 4.18 | 6.80 | 4.68 |
| 64 | 5.98 | 3.50 | 3.80 | 4.16 | 7.06 | 4.90 |

G=16 性价比最高（4.80 vs G=64 的 4.90），compute 消耗减半多。

### 6.7 Iterative Dynamics (Table 9, Figure 2)

Separate weights:
- Iter 0: FWM avg=4.96, IDM=6.37
- Iter 1: FWM avg=5.06, IDM=6.52
- Iter 2: FWM avg=4.98, IDM=6.56

FWM 在 Iter 1 达到 peak，IDM 一直单调提升。这告诉我们：**FWM 改进 IDM 比 IDM 改进 FWM 更容易**——因为 IDM 是判别任务（输入两个 image + action 输出 action probability），FWM 是生成任务（高维生成），前者本身就有更强 inductive bias。这也解释了为什么 Phase II 的 reward signal 质量更高、收敛更稳。

### 6.8 Latent Action 不塌缩 (Figure 6, Appendix D)

担心 reward hacking——IDM 学到把所有 transition 都映射到同一个 short cipher，FWM 就 easy 优化。实测：
- Uniqueness > 94% 跨所有 iterations
- PPL ratio ≈ 1，GPT-2 perplexity 和 ground truth 接近
- Average action length 16.8 tokens，**超过** ground truth 的 7.2 tokens — IDM 反而生成更详细的描述来确保可识别性

这个发现很有意思：在 reciprocal cycle 中，IDM 自然倾向"过度描述"以确保 FWM 能从描述中提取足够信息。这是 emergent behavior，KL term 之外，CMI maximization 本身就有保持信息量的作用。

---

## 7. 关键 Limitations & 开放问题

### 7.1 Long-horizon vs Iteration 的 trade-off

WORLDPREDICTIONBENCH 上 Iterative < IDM→FWM。Paper 没深入解释，但这是最关键的 open problem。猜测：iteration 让 FWM 越来越 narrow，single-step likelihood 高但 multi-step rollout 时 error compounding 更快。可能需要把 multi-step rollout 也放进 reward，或者引入 trajectory-level GRPO。

### 7.2 Shared weights 不稳定

Figure 2 (右) 共享参数 Iter 3 反而退化。FWM 生成 image token、IDM 生成 text token，两个 modality 的 gradient 在共享 backbone 上互相 interfere。这其实和 unified VLM 中 understanding > generation 的 imbalances 现象 (Shi et al. 2025; Zhang et al. 2025) 同源——multi-modal multi-task 在统一参数下优化冲突问题没解决。

### 7.3 Warm-up 仍需 SFT

完全 zero-shot 启动 SWIRL 不可能——random policy 的 rollout 无法 cover valid state，GRPO 失效。这与 reasoning model 的 RL-from-scratch 限制一样。如何用更轻量的 warm-up（比如 few-shot prompting + self-distillation）替代大规模 SFT 是工程优化方向。

### 7.4 Camera control 弱

BYTEMORPH camera 子项几乎无提升，因为 VIDGEN-1M 多是 static camera。需要混合 dataset (e.g., 大量 ego-centric / drone footage) 来补足。

### 7.5 Theoretical gap

Theorem 3.1 / 3.2 给了每一步的 learnability，但**没有证明整体 coordinate ascent 收敛到 global optimum**。Phase I 和 Phase II 互相提供 reward signal，可能存在 self-reinforcing bias（两个 model 共同固化错误 belief）。这是 EM 也会有的 local optimum 问题，但 SWIRL 因为 RL 的 stochasticity 可能更严重。

---

## 8. 关联工作 & 思想延伸

### 8.1 与 Mode Collapse / RLHF Reward Hacking

Appendix D 专门讨论 latent action 不塌缩，这和 GAN mode collapse、RLHF reward hacking 是一类问题。SWIRL 的解决方案本质是 **reciprocal 的对称性**—— IDM 不能"作弊"让 FWM 简单，因为 IDM 自己也要被 FWM 验证；FWM 也不能简单复制 $y$，因为 IDM 推断的 $z$ 是从真实 $y$ 来的。这种 mutual check 机制类似 GAN 但用 RL 而非 adversarial training。

### 8.2 与 Self-Play Reasoning (rStar-Math, etc.)

rStar-Math / Self-Taught Reasoner 系列 (https://arxiv.org/abs/2501.04519) 也是 generator + verifier 互相 self-improve，但作用在 reasoning chain 上。SWIRL 把这个 idea 推到 world modelling，"verifier" 是 IDM。两者都依赖 mutual specialization：generator 学会 producing verifiable outputs，verifier 学会 discriminating valid outputs。

### 8.3 与 VAE / Wake-Sleep

经典 VAE 是 single-step ELBO，Wake-Sleep 是交替更新 generative 和 recognition model 但用 likelihood gradient。SWIRL 的 Phase II = Wake (update recognition model to fit data under generative model)，Phase I = Sleep 变体 (update generative model to be "identifiable" by recognition model)。SWIRL 把 Sleep 阶段的"reconstruct recognition samples"替换成"maximize recognition log-likelihood"，这是 InfoMax 而非 pure reconstruction，因此不需要 $z$ 的 prior sample，而是从 belief distribution $\tilde{P}(z \mid x)$ 采。

### 8.4 与 World Model for RL (Dreamer, etc.)

Dreamer (https://danijar.com/dreamer/) 学 latent world model 然后 actor-critic 在 latent space 规划。SWIRL 不做 planning，只学 transition model。但 SWIRL 学的 latent action 是 explicit natural language token，可以直接被 agent 用作 planning 的 action space。这是 world model 和 policy 耦合的一个新方向。

### 8.5 与 Tool Learning Self-Improvement

STABLETOOLBENCH 上 SWIRL 超过 32B 模型很有启发。当前 tool learning 的训练数据严重依赖 human-annotated tool call trajectories。如果 SWIRL 能从纯 tool output 序列学 dynamics，理论上可以 scale 到所有 API logs（GitHub Actions logs、CI/CD logs、各种 automation logs），实现大规模 unsupervised tool dynamics learning。

### 8.6 与 Recent Unified VLM 对比

Paper 把 SWIRL 和 BAGEL-14B、OmniGen2、UniWorld-V1 等更大或更重 supervised 的 unified VLM 对比，SWIRL 7B 在多个子集上 competitive。这说明 **post-training 阶段 reciprocal RL 比 pretraining scale 更高效**——只要 base model 有 minimal capability，SWIRL 能把它放大。这对资源有限的研究组是关键 insight：不需要从头训 14B VLM，7B + SWIRL 即可。

### 8.7 与 Iterated RLHF / RLAIF

RLAIF (Constitutional AI) 用 AI 给 AI 提供 reward。SWIRL 更进一步——AI 既当 policy 又当 reward，且 reward 是 multi-modal log-probability 而非 scalar preference。这种 **self-provided dense reward** 比 preference-based reward signal 信息量大得多，是未来 RL 训练范式的一个可能演化方向。

---

## 9. 关键 Design Choice 总结

| Design | Choice | 原因 |
|--------|--------|------|
| Action 表示 | natural language latent variable | 保持 interpretability，避免 reward hacking |
| 框架结构 | two-phase reciprocal | 对应 CMI + ELBO 双 lower bound |
| 优化算法 | GRPO 而非 PPO | 不需要 value model，group-relative baseline 对生成任务稳定 |
| Reward 形式 | log-probability | 数值稳定，对应 information theoretic objective |
| $\beta$ in Phase II | 1 | 让 KL-regularized objective 严格等于 ELBO |
| Reference policy | 上一轮 IDM | 提供 informative prior 防漂移 |
| Phase I 采 $z$ 来源 | $\tilde{P}(z \mid x)$ | 用数据驱动的 belief 而非 fixed prior |
| 参数 sharing | separate 更稳 | multi-modal gradient interference |
| SFT warm-up | 必须 | 否则 rollout 探索失效 |
| Rollout size G | 16 性价比最高 | 减少 variance 同时控制 compute |

---

## 10. 整体直觉总结

SWIRL 的核心 insight 用一句话说：**当 action 是 latent 时，forward model 和 inverse model 互为对方的最强 supervisory signal，因为它们各自解决的是 complementary 难度的任务**。FWM 难（高维生成），IDM 相对容易（判别 + 语言输出）；让容易的 IDM 当 reward 给 FWM 提供 dense supervision，让更准确的 FWM 当 reward 给 IDM 提供 likelihood verification，两个任务交替 climb 各自的 lower bound。

这本质是把 **AlphaZero self-play** 思想搬到 generative world modelling——value network 和 policy network 互相 bootstrap，这里换成 FWM 和 IDM。

更深层的：这套 framework 提供了一种 **latent variable model 的 RL 训练范式**，理论上可以套到任何 "observation sequence with hidden cause" 的场景。比如：
- 视频帧 + latent physical force → force prediction + next frame prediction
- 用户行为 + latent intent → intent inference + next action prediction  
- Code change + latent refactor intent → intent prediction + next code state prediction

参考资源：
- Paper: https://arxiv.org/abs/2506.06006 (Qiu et al. 2025 bootstrap)
- DeepSeek-R1 GRPO: https://arxiv.org/abs/2501.12948
- Liquid VLM: https://arxiv.org/abs/2412.04332
- VIDGEN-1M: https://arxiv.org/abs/2408.02629
- AURORA-BENCH: https://arxiv.org/abs/2407.03471
- STABLETOOLBENCH: https://arxiv.org/abs/2403.07714
- MIND2WEB: https://arxiv.org/abs/2307.13830
- SCIENCEWORLD: https://arxiv.org/abs/2203.07540
- PICO-BANANA: https://arxiv.org/abs/2510.19808
- rStar-Math self-play: https://arxiv.org/abs/2501.04519
- Dreamer world model: https://danijar.com/dreamer/
- Barber & Agakov IM algorithm: https://papers.nips.cc/paper/2003/hash/ddbb9e2a1a63ce5f3e3f24e7304842dc-Abstract.html
- Wake-Sleep (Hinton): https://www.cs.toronto.edu/~hinton/absps/wakesleep.pdf

---

如果你想深入某个 sub-topic，比如 GRPO 在 multi-modal 生成中的具体实现、long-horizon 退化的 root cause 分析、或者怎么把 SWIRL 推广到 continuous action space，我可以再展开。
