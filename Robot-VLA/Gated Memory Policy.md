---
source_pdf: Gated Memory Policy.pdf
paper_sha256: 9569cb72d52dbe0249158a2aaa7ec37ad1c594c35e4f4b74740500b243922182
processed_at: '2026-08-04T12:22:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们用最直白的话来盘一盘这篇 paper。核心直觉其实非常简单：**robot 有时候需要回忆过去，但大部分时候不需要，如果强行让它一直盯着过去看，它反而会变傻**。

这篇 paper 就是教 robot 怎么“该记的时候记，不该记的时候就别管”。

### 1. 为什么 Naive 的长 History 会搞砸？

想象你教 robot 抓个方块。简单任务里，robot 只看当前一眼就知道怎么抓，这叫 Markovian task。但如果任务变成“看一眼方块颜色，过几秒后再把它放回对应颜色的盒子里”，robot 就得有 memory，这叫 Non-Markovian task。

最直觉的做法是把过去 120 步的图像和动作全塞给 Transformer。但 paper 发现这会带来灾难：
- **Overfitting**: Input space 暴涨，data 就那么多，模型直接死记硬背 training data 的 trajectory，一到 test 就废。
- **Distribution shift**: Test 的时候只要有一帧画面有点噪声，整个 120 帧的 sequence 就偏离了 training distribution，robot 直接懵了。
- **计算太贵**: 120 步的 self-attention 复杂度是 $\mathcal{O}(120^2)$，推理慢得要死。

### 2. GMP 的三个绝招

为了解决这些问题，作者提出了 Gated Memory Policy (GMP)，用了三个非常优雅的 trick：

#### 绝招一：Cross-attention + KV Cache (省钱)
不要把 history 跟当前帧拼在一起做 self-attention。把 history 单独拎出来，用 Cross-attention 让当前的 action query 去看 history 的 key 和 value。
而且借鉴了 LLM 里的 KV cache，过去的图像特征算一次就存起来，下次直接拿来用。复杂度直接从 $\mathcal{O}(H^2)$ 降到 $\mathcal{O}(H)$。

#### 绝招二：给 History 加点 Diffusion Noise (抗噪)
既然 clean history 会导致 overfitting，那就在 training 时给 history action 加点噪声。但怎么加很有讲究。
GMP 利用 diffusion 本身的特性：在预测 noise level 为 $k$ 的未来 action 时，conditioned on noise level 为 $k-1$ 的历史 action（比未来 action 干净一点）。
这样在 denoising 早期（$k$ 大），历史噪声大，防止 overfitting；后期（$k$ 小），历史噪声小，保留细节。而且 test 的时候也这么干，保证 train-test consistency。

#### 绝招三：Binary Memory Gate (选择性记忆)
这是最核心的创新。大部分时候 robot 根本不需要 history，所以搞个开关 $\mu_t$。
公式长这样：
$$ \mu_t = \mathbf{1}\{\sigma(\phi(I_t, P_t)) > 0.5\} \in \{0, 1\} $$
- $\mu_t$: timestep $t$ 的 memory gate 值。1 表示开，用 history；0 表示关，不用。
- $\phi$: 一个小 MLP network。
- $I_t, P_t$: 当前的图像和机械臂本体感觉。
- $\sigma$: sigmoid 函数。
- $\mathbf{1}\{\cdot\}$: 如果里面大于 0.5 就输出 1，否则 0。

如果 $\mu_t = 0$，cross-attention 的结果直接被抹掉，robot 只看当前帧。这不仅防干扰，还直接省掉了计算，推理速度飞起。

### 3. 这个 Gate 怎么训练？(Calibration 的妙处)

最头疼的问题是：怎么知道某个 timestep 需不需要 memory？如果 end-to-end 硬练，模型为了降低 training loss 会疯狂把 gate 拉到 1，结果就是 overfitting。

作者的解法非常聪明，搞了个 self-supervised calibration：
1. 把数据集分成 train 和 val。
2. 在 train 上训练两个 policy：一个永远不开 memory ($\pi$)，一个永远开 memory ($\pi_{\text{mem}}$)。
3. 在 val 上跑这两个 policy，看每个 timestep 的 action prediction error。
   - $\delta_t$: 不带 memory 的 error。
   - $\delta_t^{\text{mem}}$: 带 memory 的 error。
4. 如果 $\delta_t > \theta \cdot \delta_t^{\text{mem}}$ （paper 里 $\theta = 10.0$），说明不带 memory 错得太离谱，这个 timestep 标记为需要 memory ($\mu_t = 1$)，否则标 0。
5. 用这些标签训一个 binary classifier (就是那个 MLP $\phi$)。
6. 最后冻住这个 gate MLP，重新训练完整的 GMP policy。

直觉上就是：**如果在 validation set 上，加 history 能让 error 降 10 倍以上，那这个地方才真的需要 memory**。

### 4. 实验效果如何？

作者搞了个叫 MemMimic 的 benchmark，包含各种需要 memory 的任务。

| Task | Baseline (Long-hist DP) | GMP (Ours) | 直觉解释 |
| :--- | :--- | :--- | :--- |
| T4: Iterative Pushing (Sim) | ~40% | ~85% | 需要跨 trial 记住摩擦力，baseline 坚持不住同一个速度 |
| T6: Iterative Casting (Real) | 20% | ~60% | 需要记住上次甩过头了还是没够着，baseline 预测速度不一致 |
| RoboMimic (Markovian) | 显著 Drop | Competitive | 不需要 memory 的任务，gate 自动关掉，不受长 history 影响 |

数据很硬：在 non-Markovian tasks 上平均比 long-history baselines 好 30.1%，同时在 Markovian tasks 上不掉点。

### 5. 给 Karpathy 的直觉联想

看完这篇 paper，我有几个强烈的直觉：

1. **Attention Sparsity 的重要性**: 这和 LLM 里的稀疏 attention 异曲同工。大部分 token 是无用的，robot manipulation 里大部分 history 也是无用的。Binary gate 本质上就是学了一个 hard sparse mask。
2. **Validation Set 作为 Oracle**: 用 validation error 的比值来做 labeling，这其实是一种反事实推理的粗糙版。避免了 end-to-end 训练里的 shortcut learning。这和 [Consistency Models](https://arxiv.org/abs/2303.01469) 里用 teacher 指导 student 的思路有点像，都是用另一个模型的状态来指导当前模型。
3. **Diffusion 训练的 Train-Test Gap**: 之前 [Diffusion Forcing](https://arxiv.org/abs/2407.01392) 训练加噪声测试不加，这违反了 machine learning 的基本直觉。GMP 把 diffusion 的 noise schedule 同时用在 history 上，训练测试一致，这个设计太漂亮了。
4. **Overlapped Trajectory Training**: Section VIII-A 提到为了省显存，让 batch 里的 trajectory 互相 overlap，这样一张图只 encode 一次。这和 [Vision Transformers](https://arxiv.org/abs/2010.11929) 里 patch 复用的思想类似，工程上非常实用。

总而言之，这篇 paper 的核心贡献在于它击中了 robot learning 里一个被忽视的痛点：**无脑堆 context length 是不可取的**。它用一套组合拳（cross-attention 降复杂度 + noise injection 抗过拟合 + binary gate 选时机）把 long-history policy 做到了实用级别。这种“按需召回”的设计哲学，非常值得借鉴到其他 sequential decision making 的领域。

References:
- [Gated Memory Policy Project Page](https://gated-memory-policy.github.io)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392)
- [MemMimic Benchmark Details (in paper)](https://gated-memory-policy.github.io)
- [Straight-Through Estimator](https://arxiv.org/abs/1308.3432)

---

# Gated Memory Policy 详细技术解析

## 1. Problem Framing: Memory Spectrum in Robotics

这篇paper处理的核心问题是 robot manipulation 中 memory requirements 的 **spectrum**。让我先 build intuition 关于为什么这是一个 hard problem。

Robot manipulation tasks 可以沿着 memory dependency 分成三类：

- **Markovian tasks**: 当前 observation $I_t$ 足以决策，比如简单的 pick-and-place
- **In-trial memory**: 单次 episode 内需要 recall 过去的 observation/action，比如记住物体初始颜色
- **Cross-trial memory**: 跨 episode 聚合信息推断 unobservable physical properties，比如通过多次 casting 推断 friction coefficient

Naive approach 是直接 extend history window $H$，但这篇 paper 指出这会导致两个严重问题：

**Overfitting**: input space 随 $H$ 线性增长，training data 固定，模型容易 memorize training distribution 而非 generalize。这一点在 high-dimensional visuomotor policy 中尤其严重，因为 image observations 本身维度就很高。

**Distribution shift**: deployment 时任何一个 noisy frame 都会把整个 input sequence 推出 training distribution，导致 compounding errors。

这让我联想到 [Deep RL with RNNs](https://arxiv.org/abs/1906.08852) 中 similar 问题，以及 [Decision Transformer](https://arxiv.org/abs/2106.01345) 里 conditioning 长历史 context 的挑战。

---

## 2. Method: Gated Memory Policy (GMP) 架构解析

GMP 基于 **Transformer-based Diffusion Policy** (具体是 [DiT](https://arxiv.org/abs/2212.09748) backbone)，添加了三个 key components：

### 2.1 Background: Diffusion Policy Foundation

先理解 baseline。给定当前 image $I_t$ 和 robot proprioception $P_t$，policy 预测 future action trajectory：

$$A_{t:t+h} = \{A_t, A_{t+1}, \cdots, A_{t+h-1}\}$$

其中 $h$ 是 **action prediction horizon**。

Training loss 是 standard diffusion denoising objective：

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{A^0_{t:t+h}, \epsilon, k}\left[\|A^0_{t:t+h} - \varphi_\theta(A^k_{t:t+h}, I_t, P_t, k)\|_2^2\right]$$

变量解释：
- $A^0_{t:t+h}$: ground-truth action trajectory (上标 0 表示 noise level 0，即 clean)
- $A^k_{t:t+h}$: action trajectory 在 diffusion step $k$ 被 added noise 后的 noisy version
- $\epsilon \sim \mathcal{N}(0,1)$: Gaussian noise
- $k$: diffusion step index, $k \in \{0, 1, \cdots, K\}$
- $\varphi_\theta$: denoising network with parameters $\theta$
- $I_t, P_t$: visual and proprioceptive observation at timestep $t$

这里 $K$ 是 total diffusion steps，用 [DDIM scheduler](https://arxiv.org/abs/2010.02502) 在 inference 时 denoise。

### 2.2 Cross-Attention with Cached Tokens

这是第一个 key design。直接 concatenate 所有 history tokens 到 self-attention sequence 会导致 $\mathcal{O}(H^2)$ 的 computational cost。GMP 的解决方案是 separate cross-attention module。

**Architecture 细节**：

History 包含两部分：
- **Action history**: $n$ 个 chunks，每个 chunk 是 $h$ 个 actions：$\{A_{t-nh:t-(n-1)h}, \cdots, A_{t-h:t}\}$
- **Image history**: 每个 trajectory 只 sample 一帧 image (因为 high-frequency frames redundant)：$I_{t-nh;h:t} = \{I_{t-nh}, I_{t-(n-1)h}, \cdots, I_{t-h}\}$

Image encoding 用 [SigLIP2-B/16](https://arxiv.org/abs/2502.14786) pretrained ViT，然后接 multi-head attention pooling (MAP) 把所有 patches 聚合成 single token，大幅减少 visual token 数量。

**KV Cache insight**: 灵感来自 causal Transformers 的 [KV cache](https://arxiv.org/abs/1904.01038)，GMP 维护一个 sliding window of $n$ history trajectories，每个 trajectory 包含 1 个 aggregated image feature + $h$ 个 action tokens。这样 history tokens 可以被 retrieve without re-computation。

这使得 attention 复杂度从 $\mathcal{O}(H^2)$ (self-attention over concatenated sequence) 降到 $\mathcal{O}(H)$ (cross-attention: query 是 future action tokens, key/value 是 cached history tokens)。

### 2.3 Diffusion Noise Augmentation

这是第二个 key design，解决 overfitting 问题。Core idea: 不要用 clean history actions training，而是 inject diffusion-scheduled noise。

具体做法：在 diffusion step $k$，model 预测 future actions $A^k_{t:t+h}$ (noise level $k$) 时，conditioned on history actions $A^{k-1}_{t-nh:t}$ (noise level $k-1$，比 future actions **one step cleaner**)。

这创造了一个 noise schedule：随着 denoising 进展，noise level 递减：
- Early steps ($k = K, K-1, \cdots$): strong noise on history → strong augmentation，prevent overfitting
- Late steps ($k = \cdots, 2, 1$): minimal noise → capture fine-grained information

**Key distinction from [Diffusion Forcing](https://arxiv.org/abs/2407.01392)**: Diffusion Forcing 在 training 时 inject random noise levels 但 inference 时用 clean history，造成 train-test inconsistency。GMP 在 training 和 testing 都 apply noise，保持 consistency。

这个设计非常 elegant：noise schedule 与 diffusion denoising process 自然耦合，不需要额外的 noise injection mechanism。

### 2.4 Memory Gate: Binary Gating

这是第三个也是 most novel 的 component。

Given current $I_t$ 和 $P_t$，用 MLP $\phi$ + sigmoid $\sigma$ 产生 binary gate value：

$$\mu_t = \mathbf{1}\{\sigma(\phi(I_t, P_t)) > 0.5\} \in \{0, 1\}$$

变量解释：
- $\phi$: memory gate MLP network
- $\sigma$: sigmoid activation function
- $\mu_t$: binary gate value at timestep $t$, $\mu_t = 1$ 表示 recall memory, $\mu_t = 0$ 表示 skip memory
- $\mathbf{1}\{\cdot\}$: indicator function

Gate 应用在 cross-attention output 上：

$$\bar{\mathbf{z}}_{t:t+h} = \mu_t \mathbf{h}_{t:t+h} + \mathbf{z}_{t:t+h}$$

其中：
- $\mathbf{h}_{t:t+h} \in \mathbb{R}^{h \times d}$: history cross-attention output
- $\mathbf{z}_{t:t+h} \in \mathbb{R}^{h \times d}$: residual connection (current observation processing)
- $d$: DiT hidden dimension
- $\bar{\mathbf{z}}_{t:t+h}$: final output passed to feed-forward layer

**为什么 binary 而非 continuous**: paper 在 Section VIII-E 做了 ablation (Fig. 19)。Continuous gate training 时容易 collapse to 1 (overfit to history)，加 regularization 又会 hurt non-Markovian tasks。Binary gate 配合 calibration 可以 avoid 这个 dilemma。而且 binary gate 允许 completely skip cross-attention computation when $\mu_t = 0$，获得 actual speedup。

---

## 3. Memory Gate Calibration Procedure

这是 paper 中最 interesting 的部分，解决了一个 chicken-and-egg problem: 如何训练 gate 决定何时需要 memory？

### 3.1 Self-Supervised Calibration Pipeline

Fig. 3 展示的 4-stage process：

**Stage 1**: Split dataset $\mathcal{D} = \mathcal{D}_{\text{train}} + \mathcal{D}_{\text{val}}$，在 $\mathcal{D}_{\text{train}}$ 上训练两个 policies：
- $\pi$: memory gate always OFF (no history)
- $\pi_{\text{mem}}$: memory gate always ON (full history)

**Stage 2**: 在 $\mathcal{D}_{\text{val}}$ 上 evaluate 两个 policies $N$ rounds，计算每个 timestep $t$ 的 action prediction error:
- $\delta_t$: error of $\pi$ (no memory)
- $\delta_t^{\text{mem}}$: error of $\pi_{\text{mem}}$ (with memory)

**Stage 3**: Generate binary labels using ratio threshold $\theta$:

$$\mu_t = \begin{cases} 1 & \text{if } \delta_t > \theta \cdot \delta_t^{\text{mem}} \\ 0 & \text{otherwise} \end{cases}$$

Paper 用 $\theta = 10.0$ across all tasks。用 BCE loss 训练 gate MLP $\phi$。

**Stage 4**: Freeze gate weights $\phi$，在 full dataset ($\mathcal{D}_{\text{train}} + \mathcal{D}_{\text{val}}$) 上 retrain policy → final $\pi_{\text{gated}}$。

### 3.2 Intuition Behind Calibration

这个设计的 intuition 非常 clean：**如果 no-memory policy 的 error 显著大于 with-memory policy 的 error (10x)，说明这个 timestep 确实需要 memory**。反之则 memory 不必要。

这比 end-to-end training gate 好在哪里？End-to-end training 时，model 倾向于 use as much history as possible 因为这 reduces training error (overfitting)。Calibration 在 held-out validation set 上比较，避免了 overfitting bias。

Fig. 18 可视化了 calibration 结果：
- **Match Color**: $\delta_t > \theta \delta_t^{\text{mem}}$ 在 episode 中段 (robot placing cube back)
- **Iterative Pushing**: 每次推之前 $\delta_t > \theta \delta_t^{\text{mem}}$
- **RoboMimic Square**: 两个 policy error 接近，gate 几乎全程 OFF

### 3.3 Training Cost

Section VIII-F 给的 GPU-hours (H100)：

| Stage | Time |
|-------|------|
| Train $\pi$ (no memory) | 1.2h |
| Train $\pi_{\text{mem}}$ | 6.7h |
| Rollout for errors | 6.4h |
| Train Gate MLP | 1.3h |
| Train GMP (final) | 31.2h |

Total 约 47 GPU-hours，main cost 在 final GMP training。Calibration 本身 overhead 不大。

---

## 4. MemMimic Benchmark

Paper 提出了一个新 benchmark，专门测试 memory capabilities。Task 设计很 thoughtful：

### 4.1 In-Trial Memory Tasks (T1-T3)

- **T1 Match Color**: 4 bins 随机颜色，robot 抓 cube 后 colors shuffle，需要放回原色 bin。测试 **visual memory**
- **T1' Match Color with Random Delay**: 加 5-600 秒 random delay，测试 **memory length**。Result: 99.0%±1.0% success rate with 6000 frames buffer
- **T2 Discrete Place Back**: 4 bins 中一个有 cube，抓起 hold 2s 放回原 bin。测试 **spatial memory**
- **T3 Continuous Place Back**: cup + saucer 随机放置，先放 cup 到 saucer 再放回原位 (5cm 内)。测试 **spatial memory + real-world noise**
- **T3' In-the-wild Flip and Place Back**: 人 flip cup 90°，robot flip back + 放回原位。测试 **generalizability across unseen environments**

### 4.2 Cross-Trial Memory Tasks (T4-T6)

这些是 most challenging，要求 in-context adaptation：

- **T4 Iterative Pushing**: cube friction $\in [0.005, 0.015]$ 未知，6 trials 推到 target，最后 3 trials 成功算 success
- **T5 Iterative Flinging**: cloth mass $\in [0.1, 2.0]$ kg 未知，5 次 fling，最后 3 次成功
- **T6 Iterative Casting**: real-world，friction 未知，3 次 casting，最后 2 次成功

Cross-trial memory 的核心 challenge：policy 需要从过去 trial 的 outcome (overshoot/undershoot) 推断 unobservable physical property，然后 adjust action。这本质是 **in-context learning** 在 robot manipulation 中的应用。

---

## 5. Experimental Results 深度解析

### 5.1 Main Results: MemMimic

Paper 报告 **30.1% average success rate improvement** over long-history baselines on non-Markovian tasks。

具体数字 (从各 figure 读取)：

**T1 Match Color**:
- No-hist DP: stuck (no memory of pause duration)
- Mid-hist DP: random bin (insufficient memory)
- Long-hist DP: 100%
- GMP: 接近 100%

**T4 Iterative Pushing** (Fig. 8b):
- No-hist DP: ~20% (random velocity)
- Long-hist DP: ~40%
- Long-hist PTP: ~50%
- GMP: ~85%

**T6 Iterative Casting** (real):
- No-hist DP: 5% (always medium velocity)
- Long-hist DP: 20% (inconsistent velocity)
- GMP: ~60%

### 5.2 Markovian Tasks: RoboMimic (Fig. 11)

关键 finding: GMP 在 Markovian tasks 上保持 competitive performance，而 long-history baselines 显著 degrade。

| Method | Tool Hang | Square | Transport |
|--------|-----------|--------|-----------|
| No-hist DP | High | High | High |
| Long-hist DP | **显著 drop** | drop | drop |
| GMP | **competitive** | competitive | competitive |

这是因为 memory gate 在 Markovian tasks 上几乎全程 OFF ($\mu_t = 0$)，policy 自动 ignore history。

### 5.3 MIKASA-Robo Comparison (Fig. 12)

GMP 在 5 个 memory-intensive tasks 上 average **26.6% improvement** over [MemoryVLA](https://arxiv.org/abs/2508.19236)。Tasks 包括 ShellGameTouch, InterceptMedium, RememberColor3/5/9。

---

## 6. Ablation Studies 详细分析

### 6.1 Attention Visualization (Finding 2)

**Match Color** (Fig. 4c): $t=80$ (placing cube) 时 attention 集中在 $t=48$ (first observed colors)。模型自动识别关键 frame without supervision。

**Iterative Pushing** (Fig. 8c): Trial 4 pushing 时 attention 集中在 Trial 3 (undershoot) 和 Trial 2 (overshoot)。Policy 用这些 past outcomes 来 calibrate 当前 action。

这验证了 cross-attention 确实学到了 **task-relevant** 的 memory retrieval，不是 attention sink。

### 6.2 Memory Gate Statistics (Finding 3)

即使 non-Markovian tasks，gate 也不是一直 ON：
- Match Color: gate OFF 73% of time
- Iterative Pushing: gate OFF 58% of time

这意味着 memory 只在 critical moments 需要，大部分时间 current observation 足够。

### 6.3 Inference Speed (Finding 5, Fig. 14)

- Self-attention baseline: inference time 随 history length **quadratic** 增长
- GMP (Gate ON): **linear** 增长 (cross-attention)
- GMP (Gate OFF): **constant** minimal time

这解释了为什么 binary gate 比 continuous gate 好：continuous gate 永远要做 cross-attention，binary gate 可以完全 skip。

### 6.4 Noise Injection Ablation (Finding 6, Fig. 15)

在 Iterative Pushing 上对比 4 种 noise strategies：
- **No Noise**: over-rely on clean history，不 robust to distribution shift
- **Random Level**: 随机 noise level，可能 lose critical info
- **Diffusion Forcing**: train random noise, test no noise → train-test inconsistency
- **Diffusion Noising (GMP)**: train test consistent，best robustness

### 6.5 Calibration vs Joint Training (Finding 4, Fig. 13)

Joint training binary gate with STE:
- No regularization: gate → 1, poor on Markovian
- High regularization: gate → 0, poor on non-Markovian

Calibration approach: strong on **both** task types。

---

## 7. Technical Details & Implementation Insights

### 7.1 Overlapped Trajectory Training (Section VIII-A)

解决 GPU memory 问题。Standard approach: 独立 sample trajectories，每条 trajectory 重新 encode images。Overlapped approach: 构造 overlapped subsequences：

$$\tau_s = \{(I_s, A_s), \ldots, (I_{s+H-1}, A_{s+H-1})\}$$

其中 $s = 1, \ldots, T-H+1$。Consecutive subsequences share $H-1$ steps，所以每个 image $I_t$ 只 encode 一次，feature 被所有包含 $I_t$ 的 subsequences 复用。

Memory scales with **unique images per batch** 而非 total image occurrences，大幅节省。

### 7.2 Real-World Deployment Tricks

**T3 Continuous Place Back**:
- 20Hz training, 15Hz deployment (stability)
- Random trajectory interval (4-8 steps) for temporal robustness
- Asynchronous policy inference (robot moves during inference)
- Dynamic latency matching for smooth transitions

**T3' In-the-wild**:
- Dual camera (ultra-wide + main) both 256×256
- Gate always ON (special case)
- 10Hz on ARX X5

**T6 Iterative Casting**:
- 15Hz, predict 25 actions execute 15
- Random trajectory interval 12-18 steps
- Random image latency ±0.3s
- Waypoint-based position control (Fig. 17)

---

## 8. Related Work Context & Intuition Building

### 8.1 Structured Memory vs Learned Memory

Prior work 用 structured memory: [keyframe heatmaps](https://arxiv.org/abs/2501.18564), [object trajectory tracking](https://arxiv.org/abs/2508.15021), [visual trace overlays](https://arxiv.org/abs/2412.10345), [LLM textual plans](https://arxiv.org/abs/2306.15724), [vector database retrieval](https://arxiv.org/abs/2409.13682)。

这些方法 task-dependent 且 time-consuming to deploy。GMP 在 **raw image + action space** 学习 memory，task-agnostic。

### 8.2 Gated Networks 历史脉络

- [LSTM](https://arxiv.org/abs/1909.09586): input/output/forget gates 控制 recurrent state
- [GRU](https://arxiv.org/abs/1412.3555): simplified gating
- [GTrXL](https://arxiv.org/abs/1910.06764): gated self-attention for RL
- [Mamba](https://arxiv.org/abs/2312.00752): selective state spaces with learned gates
- [Gated Attention](https://arxiv.org/abs/2505.06708): modulate attention weights dynamically

GMP 的 gate 概念上类似，但应用场景不同：**trigger memory retrieval on demand**，而非 modulate internal state flow。

### 8.3 Long-Context Visuomotor Policies

- [RNN policies](https://arxiv.org/abs/2310.07732): unlimited history 但 training unstable
- [PTP](https://arxiv.org/abs/2505.09561): past-token prediction auxiliary task
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392): full-sequence diffusion
- [MemoryVLA](https://arxiv.org/abs/2508.19236): VLM-level memory
- [MeMERR](https://arxiv.org/abs/2510.20328): experience retrieval

GMP 区别在于：low-level action-observation alignment (vs VLM-level)，linear complexity (vs quadratic)，train-test consistent noise (vs Diffusion Forcing)。

---

## 9. Limitations & Future Directions

### 9.1 Finite Attention Window

Current implementation limited to finite window。Future: selective caching + token replacement based on importance。类似 [StreamingLLM](https://arxiv.org/abs/2309.17453) 的 attention sink + sliding window。

### 9.2 Action Prediction Error as Memory Proxy

$\delta_t$ 可能 unreliable for tasks with intrinsic ambiguity。Future: incorporate task semantics, multi-modal memory retrieval criteria。

### 9.3 Gate Calibration Generalization

Current calibration 是 per-task。Cross-task generalization of gate 没有探索。可能的方向：meta-learning gate，或用 VLM 提供 semantic memory cues。

---

## 10. Core Intuition Summary

让我总结这篇 paper 的 core insights：

**Insight 1: Memory is sparse in time**。大部分 timestep 不需要 memory，naive always-on memory 是 wasteful 甚至 harmful。Binary gate captures this sparsity。

**Insight 2: Memory needs differ across timesteps within same task**。Match Color 中只有 placing 阶段需要 recall 颜色，其他阶段不需要。Calibration 自动识别这些 critical moments。

**Insight 3: Train-test consistency matters for noise**。Diffusion noise schedule 在 train 和 test 都 apply，避免了 Diffusion Forcing 的 inconsistency。

**Insight 4: Cross-attention > self-attention for history**。把 history 放 separate cross-attention module 比 concatenate 到 main sequence 更 efficient，因为 history tokens 之间不需要 mutual attention。

**Insight 5: Binary > Continuous for gating**。Binary gate 允许 computational skip (actual speedup)，continuous gate 永远要做 attention。而且 binary gate 配合 calibration 避免 regularization dilemma。

这篇 paper 的 overall philosophy 很 elegant：**不要 force model 用 memory，让 model 自己 learn 何时需要 memory**。这与人 cognitive science 中的 working memory 概念吻合——我们也不时刻回忆所有过去，只在 task-relevant moments recall。

References:
- [Project Page](https://gated-memory-policy.github.io)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [DiT](https://arxiv.org/abs/2212.09748)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [SigLIP2](https://arxiv.org/abs/2502.14786)
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392)
- [PTP](https://arxiv.org/abs/2505.09561)
- [MemoryVLA](https://arxiv.org/abs/2508.19236)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [GTrXL](https://arxiv.org/abs/1910.06764)
- [RoboMimic](https://arxiv.org/abs/2108.03298)
- [MIKASA-Robo](https://arxiv.org/abs/2502.10550)
- [UMI](https://arxiv.org/abs/2402.10329)
- [StreamingLLM](https://arxiv.org/abs/2309.17453)
