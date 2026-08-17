---
source_pdf: RLVR-World Training World Models with Reinforcement Learning.pdf
paper_sha256: 8fbbaf999af996d52e26954c281b43c8ec8bdc44e0a2f28b69a4c29e8afbc1a4
processed_at: '2026-08-12T00:08:52-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 的核心逻辑极其直白：**停止用代理指标训练 World Model，直接拿你最终用来评测的那个指标，作为 RL 的 reward 去微调模型。**

为了 build 你的 intuition，我们用最直白的话把这个故事拆解开。

### 1. 核心痛点：MLE 与 Task Metric 的脱节

设想你在训练一个预测机器人下一步动作的 video world model。传统做法是 MLE，也就是 next-token prediction。你把视频用 VQGAN 切成 token，让 Transformer 猜下一个 token。这导致几个极其严重的直觉问题：

第一，模型偏向生成“概率最大、最安全”的像素，导致画面模糊。因为 MSE 这种 loss 在多模态分布下，最优解就是所有可能画面的平均值。
第二，视频相邻帧之间大约有 20% 的 token 是完全不变的。模型发现“直接把上一帧的 token 复制过来当下一帧”能拿到极低的 cross-entropy loss，于是它学会了偷懒（即 paper 里说的 repetition，高达 48.6%）。
第三，Teacher-forcing 训练时，每一步只盯着眼前这一帧的 token 对错，完全没意识到多步 rollout 后 error 会 compound 爆炸。

总结起来，你优化的是 token-level likelihood，但你关心的是 frame-level 的 LPIPS 或者 state-level 的 accuracy，这两者之间存在巨大的 gap。

### 2. 破局思路：从 LLM 搬砖到 World Model

DeepSeek-R1 已经在数学和代码上证明了 RLVR 的威力：既然你能用规则验证答案对不对，那就别费劲去训一个容易被 hack 的 reward model，直接拿规则算分做 RL。

这篇 paper 把这个思路搬到了 World Model 上。不管你是预测下一帧视频，还是预测 web page 的 accessibility tree 变化，只要你能拿出一个 ground-truth 进行比对，你就能算出一个明确的分数。把这个分数当成 reward，跑 GRPO 去微调预训练好的 World Model。

### 3. 公式直觉：GRPO 怎么把分数变成梯度

GRPO 砍掉了 critic network。给定当前状态和动作，模型采样 G 个预测输出，算这组输出的 reward 均值和方差，然后把每个输出标准化：

$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

- $i$ 是 group 内 response 的 index。
- $t$ 是 token 位置。
- $R_i$ 是第 $i$ 个预测输出的分数（比如 LPIPS）。

如果某个预测比组内平均分高，它的 advantage $\hat{A}_{i,t}$ 就是正的，模型就提高生成这些 token 的概率；反之亦然。通过组内相对比较，过滤掉了任务本身固有的 baseline 难度差异。

然后套用 PPO 的 importance ratio 和 clip 机制：

$$\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left(r_t \hat{A}_{i,t}, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t}\right) - \beta D_{\text{KL}}[p_\theta \| p_{\text{ref}}]\right)\right]$$

- $r_t = p_\theta^{i,t} / p_{\theta_{\text{old}}}^{i,t}$ 是新老公式的概率比。
- $\text{clip}$ 防止 policy 更新步子迈太大。
- $D_{\text{KL}}$ 是 KL 散度惩罚，拽住模型不让它偏离 pre-trained 的 $p_{\text{ref}}$ 太远。
- $1/|o_i|$ 做长度归一化，避免长输出主导梯度。

### 4. 为什么几百步 RLVR 顶几十万步 MLE？

看 Figure 3 的曲线，pre-training 跑了 150k 步，LPIPS 还在 14.5 磨蹭。RLVR 只跑了 200 步，LPIPS 直接降到 13.4。

直觉上，MLE 的梯度信号被“预测大量不变的背景 token”这个任务严重稀释了，算力全浪费在无关紧要的像素上。RLVR 的 reward 直接在解码后的整帧图像上算 LPIPS，这个 gradient 信号极度密集且纯粹，100% 指向“把画面预测准”这个终极目标。哪怕 base model 只有 138M 参数，只要 reward 信号对齐，瞬间就能把潜力榨干。

### 5. RLVR 是如何干掉 Repetition 的？

Table 3 里，base model 的 repetition rate 高达 48.6%。用 rejection sampling 强行拒绝重复，LPIPS 只从 14.8 降到 14.4，几乎没救。这说明问题压根就不在模型偷懒上，核心在于它压根没学到 frame dynamics，它就只会复制。

引入 RLVR 后，reward 直接惩罚那些和上一帧像素太相似的预测。模型瞬间就不敢复制了，repetition 降到 9.9%，同时 LPIPS 大降到 13.4。如果再加一个显式的 repetition penalty reward term，repetition 直接干到 0.0%。

这就是 verifiable reward 的巨大威力：你想消灭什么 artifact，就把它写进 reward 里，模型自己会找出最优解。

### 6. Metric-oriented Optimization 的副作用

Figure 4c 做了个非常有意思的 ablation。分别用 MAE, MSE, PSNR, SSIM, LPIPS 做 reward 训了 5 个模型。结果发现，拿什么 metric 训，就在什么 metric 上测得最好。

这直接揭示了 RLVR 的本质：它在向特定的 reward function 过拟合。这也是 Figure 4a 里 test-time scaling 失效的根源。Base model 保留了极大的 diversity，当 sampling 数量 N 极大时（比如 N=100），base model 总能蒙到一个极好的样本。RLVR 训练把概率质量全集中到了 high-reward mode 上，导致 best-of-1 极强，但 best-of-100 反而干不过 base model。这跟 [Yue et al.](https://arxiv.org/abs/2504.13837) 在 LLM 里发现的“RL 压榨 reasoning diversity”现象完全一致。

### 7. Real2Sim 的落地价值

Figure 5 和 Table 13 展示了最实际的应用价值。拿 trained world model 去模拟机器人开抽屉，然后人工评判成不成功。Base model 在 converged RT-1 上评估出 48.9% 的成功率，跟真实的 81.5% 差了十万八千里。RLVR 微调后，评估成功率到 62.2%，跟手工搭建的物理仿真器 SIMPLER (60.1%) 持平甚至超越。

这揭示了一个 scalable 的未来：未来人肉去搭物理仿真环境可能被淘汰。直接训一个 general-purpose video world model，然后 RLVR 微调一下就能拿来评估 policy。

### 8. 我的延伸联想与 Future Directions

作者提到 RLVR 训练几百步就收敛了，表面看是效率高，实际是个巨大的 limitation。为什么上不去？因为 base model 的 capacity 见底了。如果在 1.5B 这种小模型上，RL 探索到的也就是 pre-trained 分布里的最优 mode，没法涌现出新的 capability。

顺着这个思路联想：

**Foundation World Model 亟待出现**：现在他们 per-dataset 从头训 tokenizer 和 transformer，这是 toy setting。真正解锁 RLVR 潜力的前提，是像 [Cosmos](https://arxiv.org/abs/2501.03575) 这种 14B 规模的 general-purpose video world model 跑通。如果 base model 足够大，包含了各种物理规律，RLVR 就能像 LLM 里的 post-training 一样，把模型潜力彻底激发出来。pre-training -> SFT -> RLVR 这套三段式在 video 上肯定会跑通。

**Reward Design 是下一个瓶颈**：LPIPS 这种 metric 依然是个 surrogate。如果模型生成了一段物理上完全违背常理（比如抽屉穿透桌子）但像素差异小的视频，RLVR 也会认为它是好的。要解决这个问题，可能需要引入 [Diffusion model 里的物理约束](https://arxiv.org/abs/2505.05470)，或者用 VLM-as-a-judge 结合 physics engine in the loop 来提供更 ground-truth 的 reward。甚至可以考虑把 [DINOv2](https://arxiv.org/abs/2304.07193) 这种语义特征空间结合进 reward 里，逼迫模型在语义和像素两个层面都对齐。

**OOD Generalization 的圣杯**：World model 的核心价值在于能 rollout 训练时没见过的 action。如果 RLVR 只是 overfit 到了训练集的 state transition 上，那它就只是个更准的 predictive model。[Satori](https://arxiv.org/abs/2502.02208) 在 LLM 上发现 RL 能激发 OOD reasoning，在 video world model 上能否激发 OOD counterfactual prediction？这是整个领域最 open 且最 critical 的问题。

### References
- [DeepSeek-R1 (Nature)](https://www.nature.com/articles/s41586-025-08830-7)
- [Does RL incentivize reasoning beyond base model](https://arxiv.org/abs/2504.13837)
- [Cosmos World Foundation Model](https://arxiv.org/abs/2501.03575)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [Satori](https://arxiv.org/abs/2502.02208)
- [FlowGRPO](https://arxiv.org/abs/2505.05470)
- [SIMPLER (Real2Sim)](https://arxiv.org/abs/2410.05033)
- [iVideoGPT](https://arxiv.org/abs/2410.13825)
- [DINO-WM](https://arxiv.org/abs/2411.04985)

---

# RLVR-World: 用 RLVR 训练 World Models 的深度解析

## 1. 核心洞察与动机

这篇 paper 来自清华 Mingsheng Long 组 (龙明胜), 核心论点非常清晰：world models 现在普遍用 MLE (maximum likelihood estimation) 做 pre-training 和 SFT, 但 MLE 与下游真正关心的 metrics (prediction accuracy, LPIPS, SSIM) 之间存在系统性 misalignment。他们把 DeepSeek-R1 里跑通的 **RLVR (Reinforcement Learning with Verifiable Rewards)** 范式迁移到 world model 训练上, 用 task-specific 的 verifiable reward 直接 fine-tune pre-trained world model。

关键 motivation 有几条, 我们逐个 build intuition:

**(1) MLE 与 task metric 的 misalignment**:
- Language model 用 next-token cross-entropy 训练, 但 evaluation 看的是 answer correctness; 这导致 hallucination 和 repetition ([Holtzman et al. 2019](https://arxiv.org/abs/1904.09751))。
- Video model 用 MSE 训练, 产生 blurry predictions ([Mathieu et al. 2016](https://arxiv.org/abs/1511.05440)), 因为 MSE 平均 over hypotheses。
- Teacher-forcing 训练时, 模型看不到 multi-step error accumulation, rollout 时 compound error。

**(2) Non-end-to-end 架构的根本限制**:
- 即使 reward 可微, 像 VQGAN + autoregressive Transformer 这种 pipeline (tokenizer 与 predictor 分开训练) 也无法直接 backprop reward signal。RLVR 借助 sampling + policy gradient 绕过了这个问题。
- 这条洞察其实呼应了 [Karras et al. on diffusion](https://arxiv.org/abs/2206.00364) 类似的讨论: surrogate loss 是训练用的 proxy, 不是终极目标。

**(3) World modeling 是 RLVR 的"自然契合"任务**:
- Math/code 任务里, 答案对错有规则可以 verify (e.g., 单元测试)。
- World model 里, next state 对不对也可以 verify (e.g., 与 ground-truth next state 算 LPIPS / F1)。
- 这与 RLHF 不同, RLVR 不需要 reward model, 避免 reward hacking via reward model over-optimization ([Gao et al. 2023](https://arxiv.org/abs/2210.10760))。

## 2. 统一的 Sequence Modeling 框架

这是 paper 最 unifying 的一步。他们把 language / video / proprioceptive 全部统一成 autoregressive next-token prediction。

### 2.1 模态特定的 tokenization

| Modality | Tokenization |
|----------|--------------|
| Text | BPE ([Gage 1994](https://dl.acm.org/doi/10.5555/185235.185256)) |
| Image/Video | VQGAN ([Esser et al. 2021](https://arxiv.org/abs/2012.09841)) 或 compressive tokenizer ([iVideoGPT](https://arxiv.org/abs/2410.13825)) |
| Continuous actions / proprioception | 256 uniform bins 离散化 |

**Visual tokenization 数学细节** (Section 3):

给定图像 $\boldsymbol{x} \in \mathbb{R}^{H \times W \times 3}$, encoder 把它映射到 latent $h \in \mathbb{R}^{h \times w \times d}$, 然后 nearest-neighbor lookup 到 codebook $C = \{e_i\}_{i=1}^K$ 得到离散 token map $z \in [K]^{h \times w}$。$K$ 是 codebook size, $h \times w$ 是 spatial 分辨率, $d$ 是 latent 维度。Decoder 从 $z$ 重建 $x$。

paper 实际用的是 **FSQ (Finite Scalar Quantization, [Mentzer et al. 2024](https://arxiv.org/abs/2309.15505))** 替代 VQ, 因为 FSQ codebook 利用率更高。FSQ levels $[l_1, l_2, ..., l_d]$ 决定 codebook size $K = \prod_d l_d$。Table 7 里 RT-1 用 $[7, 5, 5, 5, 5] = 4375$。

**Compressive tokenizer** (iVideoGPT 的关键 trick):
- Context encoder $E_c$ 把一个 context frame 编码成 $N$ 个 context tokens $z_c \in [K_2]^N$。
- Per-frame encoder $E_p$ 通过 cross-attention 利用 $z_c$ 的 feature, 把每帧压成 $\bar{n}$ 个 tokens, $\bar{n} \ll N$。
- Table 7: RT-1 上 $N = 1280, \bar{n} = 80$, 压缩率 16×。
- 直觉: 视频里大部分像素帧间不变, 共享 context 后, 每帧只需编码 residual。

### 2.2 Sequence 构造

对于 single-step prediction $p(s_{t+1} | s_{t-3:t}, a_{t-3:t})$:

$$x = \text{concat}(z_{t-3}, b_{t-3}, z_{t-2}, b_{t-2}, \cdots, z_t, b_t, [\text{bos}], \underline{z_{t+1}}, [\text{eos}])$$

其中 $z_t$ 是 frame tokens, $b_t$ 是 action tokens (13 维 × 256 bins), [bos]/[eos] 是特殊 token。codebook offset 防止 visual 和 action code 冲突, total codebook size = 4633。序列长度 = 4×(320+13) + 1 + 320 + 1 = 1654, 划线部分 321 是 response tokens。

对于 multi-step prediction $p(s_{t+1:t+7} | s_t, a_{t:t+6}, s_c)$:

$$x = \text{concat}(z_c, z_t, b_t, \underline{z_{t+1}}, b_{t+1}, \underline{z_{t+2}}, b_{t+2}, \cdots, \underline{z_{t+7}}, b_{t+7})$$

context tokens $z_c$ 让模型能用整个 trajectory 的 shared 信息。Total codebook size = 9006, sequence length = 1280 + 8×(80+13) = 2024。

## 3. RLVR 训练: GRPO 详解

### 3.1 GRPO 公式拆解

GRPO ([Shao et al., DeepSeekMath](https://arxiv.org/abs/2402.03300)) 是 PPO 的简化版, 砍掉了 value network, 用 group-relative baseline 估计 advantage。

给定 question $q$, 从 behavior policy $p_{\theta_{\text{old}}}$ 采样一组 responses $\{o_i\}_{i=1}^G$, 每个 response 的 reward 是 $R_i$, advantage 是组内归一化:

$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

**变量含义**:
- $i$: group 内 response 的 index, $i \in [1, G]$, $G$ 是 group size。
- $t$: response token 的位置, $t \in [1, |o_i|]$。
- $R_i$: 第 $i$ 个 response 的 scalar reward (这个 paper 里是 video-level 或 state-level metric, 与 token 无关, 所以同一 response 内所有 token 共享一个 $\hat{A}_{i,t}$)。
- mean/std: 在 group 内计算的样本均值和标准差。

优化目标 (Eq. 1):

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim p_{\theta_{\text{old}}}(\cdot|q)}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left(\min\left(\frac{p_\theta^{i,t}}{p_{\theta_{\text{old}}}^{i,t}}\hat{A}_{i,t}, \text{clip}\left(\frac{p_\theta^{i,t}}{p_{\theta_{\text{old}}}^{i,t}}, 1-\varepsilon, 1+\varepsilon\right)\hat{A}_{i,t}\right) - \beta D_{\text{KL}}[p_\theta \| p_{\text{ref}}]\right)\right]$$

**逐项解析**:
- $p_\theta^{i,t} = p_\theta(o_{i,t} | q, o_{i,<t})$: 当前 policy 给第 $i$ 个 response 的第 $t$ 个 token 的概率。
- $p_{\theta_{\text{old}}}^{i,t}$: 采样时的 policy 概率 (类似 importance ratio 分母)。
- $p_{\theta_{\text{old}}}^{i,t} / p_{\theta_{\text{old}}}^{i,t}$: PPO 的 importance ratio $r_t$。
- $\text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)$: 把 ratio 限制在 $[1-\varepsilon, 1+\varepsilon]$, $\varepsilon$ 是 clip 范围 (通常 0.1~0.2)。
- $\min(\cdot, \cdot)$: PPO pessimistic bound, 取 clipped 和 unclipped 的较小值, 这样 ratio 漂得太远时不会 over-optimize。
- $\beta D_{\text{KL}}[p_\theta \| p_{\text{ref}}]$: KL penalty 防止 policy 偏离 reference policy $p_{\text{ref}}$ (一般是 pre-trained model) 太远。$\beta$ 是 KL 系数, paper 里 $1 \times 10^{-3}$。
- $1/|o_i|$: 长度归一化, 避免 long response 主导。

### 3.2 Verifiable Reward 设计

**Reward 函数 (Eq. 3)**:

$$R_i = \text{sign}(D) \cdot D(\hat{s}_i', s')$$

- $D(\cdot, \cdot)$: 预测和真值的 metric。
- $\text{sign}(D) = -1$ 当 $D$ 越小越好 (e.g., MSE, LPIPS)。
- $\text{sign}(D) = +1$ 当 $D$ 越大越好 (e.g., accuracy, F1)。

**Text game (Eq. 4, task-specific reward)**:

$$R = \alpha_1 \cdot \text{acc}_{\text{all}} + \alpha_2 \cdot \text{acc}_{\text{changed}} + \alpha_3 \cdot \mathbb{I}(\text{correct})$$

- $\text{acc}_{\text{all}}$: 所有 properties 的 accuracy。
- $\text{acc}_{\text{changed}}$: 只看被 action 影响的 properties 的 accuracy (这是真正反映 model 理解 action 效果的部分)。
- $\mathbb{I}(\text{correct})$: 完全 match 的 binary 指示。
- paper 用 $\alpha_1=0.1, \alpha_2=1, \alpha_3=0.2$, 直觉是把 weight 集中在 changed properties 上, 但保留 small bonus 给完全正确, 给 all-accuracy 提供一点平滑梯度。

**Web page**: F1 score, 见 Eq. 5-8。

- $\Delta\hat{s} = \{\hat{c}_1, ..., \hat{c}_m\}$: 预测的 changed items set。
- $\Delta s = \{c_1, ..., c_n\}$: ground-truth changed items set。
- TP = $|\Delta\hat{s} \cap \Delta s|$, 严格 exact match (包括 ID, type, content, attributes)。
- 边界处理: 若 $\Delta\hat{s} = \Delta s = \emptyset$, precision 和 recall 都 = 1 (model 正确预测了"没有变化")。
- 这个设计很关键, 它把 "正确地说 nothing changed" 也奖励, 避免 model 乱编 changes。

**Video (Eq. 9-10)**:

$$R(\hat{s}_{t+1:t+7}, s_{t+1:t+7}) = -\sum_{\tau=t+1}^{t+7}\left[L_1(\hat{s}_\tau, s_\tau) + \text{LPIPS}(\hat{s}_\tau, s_\tau)\right]$$

- $L_1$: pixel-wise absolute error, 保留高频细节。
- LPIPS ([Zhang et al. 2018](https://arxiv.org/abs/1801.03924)): 用 AlexNet/VGG features 算 perceptual distance, 与人类视觉感知更对齐。
- 这个组合和 VQGAN 训练时的 reconstruction loss 一致, 直觉是: 让 reward 和 tokenizer 训练目标 aligned, 解码出来的图更接近 tokenizer "理解" 的 image manifold。
- Tokenizer 不更新, 只 update Transformer。

## 4. 实验结果详细解读

### 4.1 Text game state prediction (Table 1)

| Model | Unchanged | Changed | Overall |
|---|---|---|---|
| Base R1-Distill-Qwen-1.5B | 11.98% | 0.08% | 7.11% |
| SFT | 38.88% | 24.21% | 32.87% |
| RLVR (binary) | 73.57% | 33.14% | 57.01% |
| RLVR (task-specific) | **83.66%** | 33.80% | 63.24% |
| RLVR 7B (binary) | 83.08% | 40.33% | 65.53% |
| GPT-4 [Wang et al.] | 73.90% | 51.60% | 64.76% |

**直觉 build**:
- 1.5B base 在 changed cases 上几乎随机 (0.08%), 因为没见过这种 task format。
- SFT 教会了输出格式, 但 unchanged cases 的 38.88% 说明 model 还是倾向"copy 上一个状态"。
- RLVR binary reward 直接把 unchanged accuracy 提到 73.57%, changed 到 33.14% — 这是 reward 直接对齐 accuracy 的效果。
- Task-specific reward 在 unchanged 上又涨 10%, 因为额外 reward 引导 model 主动去 check changed properties。但 changed cases 提升很有限 — paper 说 1.5B base 的 capacity 是 bottleneck。
- 7B binary reward overall 超过 GPT-4, 但 changed 仍不如 GPT-4 (40% vs 51.6%), 说明 small RLVR model 在"简单 unchanged" 上能作弊超越大 model, 但真正难 case 还是 capacity 主导。

### 4.2 Web page state prediction (Table 2)

| Model | Precision | Recall | F1 | Web Agent SR |
|---|---|---|---|---|
| Base | 15.59% | 15.70% | 11.83% | n/a |
| SFT | 48.99% | 56.05% | 49.94% | 12.06% |
| RLVR | 72.77% | 64.55% | 65.11% | 14.29% |
| Δ | +48.5% | +15.1% | +30.3% | +18.4% |

**直觉**: Precision 涨得多 (+48.5%), recall 涨得少 (+15.1%), 说明 RLVR 主要让 model "不乱说 changes", 而不是发现更多 changes。这和 F1 reward 直接对应: F1 偏向 P=R 平衡, 但 model 之前 precision 太低 (乱说), 所以梯度主要拉 precision。

### 4.3 Video: RT-1 multi-step prediction (Table 3)

| Setting | Repetition↓ | MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|---|
| Base | 48.6% | 0.659 | 23.1 | 80.9 | 14.8 |
| Base w/ rep rejection | 0.0% | 0.593 | 23.3 | 81.0 | 14.4 |
| RLVR-World | 9.9% | **0.486** | **24.1** | **82.4** | **13.4** |
| RLVR + rep penalty | 0.0% | 0.506 | 24.0 | 82.2 | 13.7 |

**关键观察**:
1. Base model repetition rate 高达 48.6%, 即一半的预测帧是上一帧的复制。原因是 ~20% 的 tokens 跨帧不变, MLE objective 鼓励 model 走这个 shortcut。
2. Base w/ rep rejection: 用 rejection sampling 强制不重复, LPIPS 只从 14.8 → 14.4, 几乎没用 — 说明问题不是简单 "model 偷懒", 而是 model 没学到 frame dynamics。
3. RLVR 直接降到 9.9% 并把 LPIPS 降到 13.4, 即 model 真的学到了 dynamics。
4. +rep penalty reward 把 repetition 完全消除 (0%), metric 略微下降但仍远好于 base — 说明可以设计 non-differentiable reward 解决特定 artifact, 这是 RLVR 比 differentiable loss 强的地方。

### 4.4 训练效率对比 (Figure 3)

这是最 striking 的结果: **RLVR 只用几百个 gradient steps** 就达到 base model 用 150k 步 MLE 继续训练达不到的水平。LPIPS 14.5 vs 13.4。

直觉: MLE 优化的是 token-level likelihood, 大量梯度花在"predict unchanged tokens"上, 与 LPIPS 弱相关。RLVR 直接优化 LPIPS, gradient 信息密度高得多。这也呼应了 LLM 里 R1 的发现: RL 阶段比 SFT 阶段 token-efficient 多了。

### 4.5 与 SOTA 比较 (Table 4, PushT/Rope/Granular)

对比 DINO-WM ([Zhou et al. 2025](https://arxiv.org/abs/2411.04985)) 用 DINOv2 latent space, AVDC ([Ko et al. 2024](https://arxiv.org/abs/2310.08888)) diffusion:

| Model | PushT LPIPS | Rope LPIPS | Granular LPIPS |
|---|---|---|---|
| DINO-WM | 0.7 | 0.9 | 3.5 |
| RLVR-World | 0.70 | 2.08 | 2.42 |

- PushT 与 DINO-WM 持平。
- Rope 略差 (2.08 vs 0.9), paper 在 Table 10 解释是 small dataset 上的 overfitting, joint training + individual tune 后降到 1.65。
- **Granular 大幅超越 DINO-WM (2.42 vs 3.5)**, 这是 paper 最强的 SOTA 结果。Granular 是 particle-based, DINOv2 features 对这种 texture 不友好, 而像素级 reward 反而更直接。

### 4.6 Metric-oriented optimization (Figure 4c)

用 MAE/MSE/PSNR/SSIM/LPIPS 分别做 reward 训 5 个 model, 然后在 5 个 metric 上分别 eval。结果是对角线上最好 — 用什么 reward 训, 在什么 metric 上 eval 就最好。

直觉: 这直接验证了 "reward = metric" 的核心论点。MLE 训出的 model 在所有 metric 上都次优, 因为它没针对任何一个优化。这其实是个 slight warning: 选 reward 时要想清楚下游用什么 metric eval。

### 4.7 Test-time scaling (Figure 4a)

- Best-of-N 下, RLVR-World 的 single-sample 性能 > Base model best-of-5。
- 但 N=100 时 Base 反超 RLVR — 这呼应 [Yue et al. 2025](https://arxiv.org/abs/2504.13837) 的发现: RLVR 可能 overfit 到 reward, 减少了 sample diversity。

直觉: RLVR 把 mode mass 集中到 high-reward mode 上, best-of-1 更强, 但极端 best-of-N 时反而失去了 base model 那种 "wide-spectrum 探索" 能力。这对 future work 是个重要 hint: 怎么 retain diversity 同时 sharpen mode?

### 4.8 RL training scaling (Figure 4b)

GRPO group size 越大, 收敛越快, 最终性能越好。直觉: group size 大 = baseline 估计更准 + exploration 更广。但 cost 线性增加, trade-off 在那。

## 5. 应用: Real2Sim Policy Evaluation (Figure 5)

### Setup
- Policy: RT-1, RT-1-X (4 个 checkpoints)。
- Tasks: open/close top/middle/bottom drawers (6 个 task)。
- 用 trained world model rollout, 然后人工 annotate 是否成功。
- 对比 SIMPLER ([Li et al. 2024](https://arxiv.org/abs/2410.05033)) — 一个手工搭建的 sim。

### Table 13 结果 (节选 Open Drawer)

| | RT-1 (Begin) | RT-1-X | RT-1 (15%) | RT-1 (Conv) |
|---|---|---|---|---|
| Real | 0.0% | 51.9% | 70.4% | 81.5% |
| SIMPLER-VM | 0.0% | 29.6% | 46.3% | 60.1% |
| Base model | 4.4% | 18.8% | 50.0% | 48.9% |
| RLVR-World | 3.3% | **33.3%** | **62.2%** | **62.2%** |

**直觉**:
- Base model 在 RT-1 (Conv) 上仅 48.9%, 比 Real 81.5% 低 32.6%。RLVR 提到 62.2%, gap 缩小到 19.3%。
- SIMPLER-VM 60.1%, gap 21.4% — RLVR 已经超过手工 sim。
- 这是个 scalable 路线: 未来 general-purpose video world model 起来后, 直接 RLVR fine-tune 就能 sim-to-eval, 而不需要 per-task 手搭 sim。

paper 提到 VLM-based 自动 evaluation 不可靠 (GPT-4o, Gemini 2.0 Flash 判断 unstable), 只能 human annotation — 这是个有趣的 limitation, 反映了 visual understanding 任务的 reward 设计还不成熟。

## 6. Model Predictive Control (Table 2 & 12)

Web agent MPC: policy propose 20 actions → 选 3 个最频繁 → world model 预测 next state → summarizer 提 10 个 key changes → value model (DeepSeek-V3) 打分 (1-5) → 选最高分 action 执行。

结果 (Table 2): SFT 12.06% → RLVR 14.29%, +18.4%。Table 6 显示 domain-specific gain 不均匀, Gitlab +61.5%, CMS 反而 -4.8% — 说明 world model 提升对某些 domain 转化率高, 但 domain shift 还是大问题。

PushT MPC (Table 12): Base CEM 80% → RLVR 86%, 与 DINO-WM 持平。

## 7. Architecture & Training Detail 解析

### 7.1 Visual tokenizer (Table 7)

**Per-frame VQGAN**:
- Input 256×320, codebook K=4375 (FSQ 7×5×5×5×5), N=320 tokens/frame。
- Losses: $L_1$ (reconstruction) + perceptual (VGG features) + adversarial (discriminator)。
- 5×10^5 steps, 16 batch, segment 8。
- 这就是标准 VQGAN 配方 ([Esser 2021](https://arxiv.org/abs/2012.09841))。

**Compressive tokenizer** ([iVideoGPT](https://arxiv.org/abs/2410.13825)):
- Context $K_2 = 4375$, N=1280; per-frame $K_1=4375$, $\bar{n}=80$。
- Dual encoder-decoder $\{(E_c, D_c), (E_p, D_p)\}$, cross-attention 在 32×40 分辨率。
- 6×10^5 steps, segment 32, 7 frames sampled。

### 7.2 Transformer (Table 8)

- 12 layers, 768 hidden, 3072 FFN, 12 heads, RoPE $\theta=10000$。
- 这就是 GPT-2 small (138M params) 配置。
- Single-step: 9.9×10^5 pre-training steps; multi-step: 4.5×10^5。
- 注意这是个非常小的 model, RLVR 仍有大效果, 说明 gain 不来自 capacity 而来自 objective alignment。

### 7.3 RLVR training (Table 5, 8)

- Group size G=5 (text) 或 16 (video main result) / 32 (better but cost 高)。
- KL coefficient $\beta = 10^{-3}$ (small, 允许 policy 移动)。
- Learning rate $10^{-6}$ (text) / $5 \times 10^{-5}$ (video) — video LR 高 50×, 可能因为 video model 小。
- Total video RLVR steps ~200, 时间 3.5h (single-step) / 10h (multi-step), 4×40G A100。

## 8. Limitations & Future Directions (Section 7)

paper 自己列的几个 open problem, 我加点解读:

1. **Converges within hundreds of steps** — 看起来是优势, 实际是 limitation: 这说明 RLVR 在 sample 上"看到了 reward 的 ceiling"。如何 break through 是大问题。可能与 base model capacity 有关, 也可能与 group size 有关, 也可能与 reward sparsity 有关。

2. **OOD generalization** — [Shen et al. Satori](https://arxiv.org/abs/2502.02208) 发现 RLVR 让 LLM 在 OOD reasoning 上 generalize。paper 怀疑这能否迁移到 world model 的 counterfactual actions (e.g., 训练时没见过的 action)。这是个 fundamental question: world model 的价值恰恰在 OOD rollout, 不能 generalize 就没意义。

3. **General-purpose video world model** — 现在他们 per-dataset 从头训 tokenizer + transformer, 这是 toy setting。真正解锁 RLVR 需要像 [Cosmos](https://arxiv.org/abs/2501.03575), [Genie 2](https://arxiv.org/abs/2402.15391), 或 Sora-style 的 general-purpose video WM 先做出来。预训练-监督微调-RL 微调这个三段式在 LLM 上 work, 在 video 上还没被验证。

4. **Diffusion model 上的 RLVR** — paper 没做, 但 [DanceGRPO](https://arxiv.org/abs/2505.07818), [FlowGRPO](https://arxiv.org/abs/2505.05470) 已经并发在做。这是个有意思的方向, 因为 diffusion 的 tokenization 不像 autoregressive 那么自然, 但 reward signal 可以直接连到生成的 image。

5. **Task-aligned reward design** — LPIPS / SSIM 是 surrogate, 真正想测的可能是 "physical plausibility", "action consistency"。比如 robot 推抽屉, 抽屉应该是被推出的, 而不是凭空位移。需要更 sophisticated reward (physics engine in the loop? VLM-as-judge?)

## 9. 跟 LLM RLVR 工作的对比

| 维度 | LLM RLVR (R1) | World Model RLVR (本文) |
|---|---|---|
| Token | text tokens | visual + action tokens |
| Reward sparsity | sparse (binary correctness) | dense (per-frame LPIPS) |
| Group size | 通常 8-64 | 5-32 |
| Convergence speed | 几千步 | 几百步 |
| Verification | rule-based math/code | metric against ground truth |
| Failure mode | format hacking | pixel-level artifacts |

LLM RLVR 上看到的 "aha moment" 和 reasoning emergence, 这篇 paper 没观察到对应现象 — 因为 world model 是 prediction, 不是 reasoning。这其实是 fundamental difference: reasoning 是 model 主动生成中间步骤, world prediction 的"中间步骤"是 token, 但 token 之间没有 reasoning structure。

## 10. 与 concurrent 工作的关系

- **Genie 2 / Cosmos**: 这些是 general-purpose video world model 的努力, 是 RLVR-World 的 future base。
- **Sora as world simulator** ([OpenAI blog](https://openai.com/research/sora)): OpenAI 提了 world simulator 概念但没公开训练细节, RLVR-World 提供了 post-training 视角。
- **DINO-WM**: 用 pretrained DINOv2 features 做 latent space, 避开 end-to-end training。RLVR-World 的对比显示: 在 PushT 上持平, 在 Granular 上 RLVR 大胜 — 说明 pixel-level RLVR 在 texture-rich 任务上比 latent-space model 强。
- **SIMPLER**: 手工 sim 的 alternative, RLVR-World 显示 learned world model 可以 sim-to-eval 更接近 real。

## 11. 我对这篇 paper 的整体评价

**强点**:
1. Unification of language + video world modeling under autoregressive + RLVR 框架是干净的设计。
2. 几百 RLVR steps 达到 150k MLE 步的效果是 striking 的实证。
3. Mitigating repetition 那个 case study 把 MLE 的 failure mode 讲得很清楚。
4. Metric-oriented optimization 那个 ablation 直接证明了核心论点。
5. Real2Sim policy evaluation 跑通 end-to-end 应用, 落地价值清楚。

**Weaknesses / Open questions**:
1. **Base model 太小** (138M video, 1.5B text), 没法证明 scaling 上 RLVR 仍 effective。需要至少 1B video model + 7B+ text model 的 scaling law 验证。
2. **Group size 5 在 web page 上是不是太少了**: 5 个样本估 std 不稳, advantage noise 大。Table 5 显示 web page 用 G=5, text game 也 G=5。可能限制了 final performance。
3. **No comparison with differentiable reward**: 既然 reward 是 LPIPS, 能不能 chain rule 回到 token logits 通过 Gumbel softmax? Paper 没讨论。
4. **No KL ablation**: $\beta = 10^{-3}$ 是怎么定的? 太大太小会怎样?
5. **Granular 上 7B RLVR vs DINO-WM 显示 pre-trained features 优势**: 但他们没分析 RLVR 与 pre-trained features 结合能否更好。
6. **Reward hacking**: 是否有 mode 在 high LPIPS 但 physically implausible? Paper 没讨论。

## 12. 给 Karpathy 的 connection

这篇 paper 的核心论点其实和你经常强调的 "next-token prediction is not the objective we care about" 是一回事:

- 你在 [Recipe for training neural networks](http://karpathy.github.io/2019/04/25/recipe/) 里讲过: 仔细想你要 optimize 什么, 别盲信 loss。
- 你的 [RLHF blog](https://openai.com/blog/learning-to-summarize/) 和后来 RLVR 在 R1 上的爆发证明: post-training with task-aligned reward 是 powerful paradigm。
- 这篇 paper 把这个 paradigm 推广到 generative models 而不只是 reasoning models, 论点就是说: 任何 generative model 只要能 verifiably evaluate output, 都可以用 RLVR post-training。
- 你可能感兴趣的 future direction: 如果 base model 是 [Cosmos](https://arxiv.org/abs/2501.03575) 这种 14B 规模的 general-purpose video WM, RLVR 还能否几百步搞定? 还是 reward signal 在大 model 上会变得稀疏到无法 gradient?

## References

- [Paper project page](https://thuml.github.io/RLVR-World/)
- [DeepSeek-R1 (Nature)](https://www.nature.com/articles/s41586-025-08830-7)
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
- [iVideoGPT](https://arxiv.org/abs/2410.13825)
- [ByteSized32 State Prediction](https://arxiv.org/abs/2402.07095)
- [WebArena](https://arxiv.org/abs/2307.13854)
- [WMA (Web Agents with World Models)](https://arxiv.org/abs/2411.04182)
- [DINO-WM](https://arxiv.org/abs/2411.04985)
- [VQGAN](https://arxiv.org/abs/2012.09841)
- [FSQ](https://arxiv.org/abs/2309.15505)
- [LPIPS](https://arxiv.org/abs/1801.03924)
- [Cosmos World Foundation Model](https://arxiv.org/abs/2501.03575)
- [Genie (DeepMind)](https://arxiv.org/abs/2402.15391)
- [Sora as world simulators (OpenAI)](https://openai.com/research/sora)
- [SIMPLER (Real2Sim)](https://arxiv.org/abs/2410.05033)
- [RT-1](https://arxiv.org/abs/2212.06865)
- [AVDC (action-conditioned video diffusion)](https://arxiv.org/abs/2310.08888)
- [DanceGRPO](https://arxiv.org/abs/2505.07818)
- [FlowGRPO](https://arxiv.org/abs/2505.05470)
- [Satori](https://arxiv.org/abs/2502.02208)
- [PPO](https://arxiv.org/abs/1707.06347)
- [Reward model over-optimization scaling laws](https://arxiv.org/abs/2210.10760)
- [Karpathy: Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
- [Does RL incentivize reasoning beyond base model](https://arxiv.org/abs/2504.13837)

如果你想继续深入, 我可以再展开 (a) GRPO 与 PPO 的 advantage 估计 bias-variance 分析; (b) compressive tokenizer 的 cross-attention 设计为什么能避免 context information leakage; (c) Real2Sim 里 SIMPLER-VM 与 base model 在不同 task 上的 gap 来源分析; (d) 这个 framework 怎么扩展到 multi-agent world models (e.g., game environments)。
