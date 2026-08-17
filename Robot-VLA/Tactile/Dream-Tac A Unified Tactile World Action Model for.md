---
source_pdf: Dream-Tac A Unified Tactile World Action Model for.pdf
paper_sha256: 3c8413736c059c30e43d9a257fd18eba57defa8fea85f90c7b3aef03302f827a
processed_at: '2026-08-03T23:17:41-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Dream-Tac 在干啥

## 一、痛点：眼睛骗人，手不骗人

你让机器人去插 USB、切香蕉、削黄瓜皮，这种活有个共同点——**关键瞬间你看不见**。

摄像头看着 gripper 靠近 USB，但"对齐没对齐"、"碰没碰上"、"插进去多深"，RGB 里完全看不出来。USB 黑乎乎的，端口也黑乎乎的，几毫米的偏差在 224×224 的图像里就是一个像素的事，你怎么 policy 都看不准。

但 fingertip 上的 tactile sensor 不一样——它直接贴在物体表面，**有没有接触、压多深、滑没滑**，全是直接物理信号，骗不了人。Paper 里 Figure 1(a) 那个对比很直观：contact 前后 RGB 帧肉眼分不出差别，但 tactile image 像炸了一样变化。

所以问题不是"要不要用 tactile"，问题早就解决了——肯定要用。真正的问题是 **怎么用**。

---

## 二、核心洞察：tactile 不是普通 modality

大部分人加 tactile 的方式很直觉：把 tactile image 也喂进 transformer，跟 RGB 一样当 token 处理。但这篇 paper 抓住了一个关键观察：

**tactile 信号是"稀疏事件流"。**

什么意思？你抓一个东西，整个过程可能 20 秒：
- 前 5 秒 gripper 在空中移动，touch sensor 一动不动（基本就是 sensor noise 在抖）
- 第 5 秒碰上物体，touch 信号"啪"一下跳起来
- 第 6-15 秒持续接触，信号又有一定 pattern
- 第 16 秒松手，又是"啪"一下变化
- 最后几秒信号回到 baseline

也就是说，**80% 的时间 tactile 信号几乎没信息**，只有那几个"事件"瞬间信息密度爆炸。

如果你把它当普通 modality 喂进 transformer，让 self-attention 一视同仁地处理，会怎样？所有 token 都会持续 attend 到这些"几乎不变"的 tactile token，把它们的信息均匀累积进 representation 里。结果就是：**真正重要的 contact 瞬间，被淹没在一堆没意义的 baseline 信号里**。attention 被稀释了。

这是 paper 要解决的核心矛盾。

---

## 三、解法：装一个"事件触发的开关"

paper 的思路特别 Karpathy 风格：**不要学一个 gating network，直接用先验硬编码**。

具体怎么做？看前后两帧 tactile image 差多少：

- 差不多没动 → 现在 tactile 没啥信息，attention 别太关注它
- 差很多 → 现在 contact 状态在变，attention 该多看看它

这个"差多少"用一个标量 $\rho_t$ 衡量，就是两帧 RGB 像素级 mean absolute difference 除以 255 归一化。左右两个 fingertip sensor 各算一个，取 max（单边有 contact 也得触发）。

然后过一个 sigmoid 把这个 $\rho_t$ 压到 $[0.15, 1.0]$ 区间，得到一个 timestep 级别的 gate $g_t$。

这个 gate 怎么用？塞进 attention logit 里当一个 additive bias：

$$\text{logit}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}} + \alpha \cdot g_t \cdot (1 - M_i) \cdot M_j$$

看起来吓人，其实就一句话：**当 query 是非 tactile token、key 是 tactile token 时，给 attention 加一个 boost，boost 大小由 $g_t$ 控制。**

$(1-M_i) M_j$ 这个 asymmetric mask 是精髓：

| query 是 tactile 吗 | key 是 tactile 吗 | bias 起作用吗 |
|---|---|---|
| 是 | 是 | 不起作用 |
| 是 | 不是 | 不起作用 |
| 不是 | 不是 | 不起作用 |
| **不是** | **是** | **起作用** ✓ |

也就是说，**只有 action/visual/proprioceptive token "看" tactile token 的时候，才会被 gate 调制**。tactile 自己看自己、tactile 看 visual，都不受影响。这是个 directional inductive bias：让 action 去 "查 tactile 状态"，不让 tactile 反过来被 action 污染。

整张图最妙的地方在于：**gate 没有任何 learned 参数**。$(m, s, k, \epsilon) = (0.002, 0.001, 4, 10^{-6})$ 全是固定的超参。意思是 paper 把 "contact 时 tactile 剧烈变化" 这个先验知识直接 hard-code 进了 architecture 里，省得让模型去学。

这种 "能用先验就别学" 的设计哲学，背后是对 inductive bias 的笃信——学出来的 gate 可能 overfit 训练分布，硬编码的 gate 至少在 sensor noise profile 不变的时候永远 robust。

---

## 四、工程坑：怎么把这个 bias 塞进 FlashAttention

到这里理论上就结束了，但工程上有大坑。

FlashAttention 是一个 **fused kernel**——它把整个 attention 计算塞进一个 GPU kernel 里，中间不把 $S \times S$ 的 attention matrix 写回 HBM。这是它快的核心。代价是：**它假设 logit 就是 $\frac{q^\top k}{\sqrt{d}}$，你想加任何 dense additive bias 都会破坏这个 fused path**。

Cosmos-Policy 的 naive 实现就是栽在这了——加了 attention bias 之后训练时间从 24 秒/iter 飙到 80 秒/iter，慢了 3 倍多。

paper 用了一个很漂亮的 trick 化解。观察到 CASA 的 bias 其实是 **rank-one** 的：

$$\Delta_{ij} = \gamma_b \cdot a_i \cdot b_j$$

其中 $a_i, b_j \in \{0, 1\}$ 是 query/key 的 tactile mask。这个形式可以分解成 $u_i \cdot v_j$，也就是说 **可以假装它是 query 和 key 的一个额外 channel 的内积**：

- 把 query 从 $[d]$ 维扩到 $[d+1]$ 维，最后那个 channel 存 $\sqrt{\gamma_b} \cdot a_i$
- 把 key 同样扩一维，最后那个 channel 存 $\sqrt{\gamma_b} \cdot b_j$
- 它们的内积就自动等价于原始 attention logit + bias

FlashAttention 根本不知道你加了 bias，它只看到一堆 query 和 key 在做标准 dot product，照常 fuse 进 kernel 里。代价只是每 token 多了一个 channel 的存储，padding 用 0 填上对齐就行。

结果：训练时间从 80 秒/iter 降到 27 秒/iter，**2.9× 加速**。这是 paper 最实用的工程贡献——任何想给 attention 加 structured bias 又怕 FlashAttention 翻车的人都可以直接抄这个 trick。

paper 引用的是 FlashBias (https://openreview.net/forum?id=7L4NvUtZY3) 这个工作，他们是把这个 trick 用在 contact-aware bias 上。

---

## 五、推理加速：第 1 步和第 3 步是真的，其他步骤偷懒

diffusion 模型推理慢，因为要跑很多 denoising step。Dream-Tac 默认 10 步，在 A800 上 5Hz——对机器人控制太慢。

paper 做了一个诊断实验：观察相邻步骤的 action latent 变化。发现 **cosine similarity ≈ 0.997**——也就是说 10 个 step 里，9 个 step 的输出几乎一样。巨大的计算冗余。

但有意思的是，paper 也试了用 "timestep embedding 的变化" 来预测什么时候该重算（TeaCache 的思路 https://arxiv.org/abs/2411.19108），结果发现 **不靠谱**——timestep embedding 的变化和 action latent 的变化几乎不相关。

这其实暗示一个有意思的事：**WAM 的 action latent dynamics 不像 video generation 那样 timestep-monotonic**。在 video generation 里，timestep 决定 noise level，noise level 决定 coarse-to-fine 进程。但在 WAM 里，conditioning context（observation、language）对 action latent 的影响可能比 noise level 还大。video domain 的 cache trick 不能无脑迁移到 WAM。

paper 的实用解法：经验性地选第 1 步和第 3 步做 full forward（第 3 步是 validation 上观察到变化最大的步骤），其余 8 步直接复用第 1 步或第 3 步的缓存结果。10 步降到 2 步，latency 1109ms → 619ms，**1.8× 加速，success rate 不掉**。

这是个很 pragmatic 的选择——选哪些步、为什么是第 3 步、能不能更精细，paper 没回答，留作 future work。

---

## 六、几个关键实验结果讲讲

### 6.1 主实验

| 方法 | 平均成功率 |
|---|---|
| $\pi_0$ | 30.8% |
| $\pi_{0.5}$ | 45.0% |
| ForceVLA | 50.8% |
| Cosmos Policy | 51.7% |
| **Dream-Tac** | **83.3%** |

31.6% 的绝对提升，在 robotics 实验里是很大的 gap。

### 6.2 Play Mahjong 这个任务最值得说

这个任务被故意设计成 **完全视觉遮挡**——视觉输入被 block 掉，机器人只能靠 tactile 判断摸到的是哪张牌，再决定推不推倒。

这个任务上：
- Dream-Tac: 100%
- ForceVLA: 55%

ForceVLA 是用 force-torque 信号（1D 时序标量）的 VLA。tactile image 是 2D 空间分布，信息密度高一个数量级。光知道"压了 5 牛顿"和知道"压出这个图案"完全不一样——后者能告诉你"摸到了哪张牌"，前者只能告诉你"碰到了点东西"。

更重要的是 attention bias 的作用：Dream-Tac 强制 policy 在 contact 强的时候 attend tactile，所以视觉被 block 之后 policy 不会"想念视觉"。ForceVLA 仍然保留对 visual prior 的依赖，视觉一断就慌。

这印证了 paper 想说的："tactile-grounded policy vs tactile-as-auxiliary" 是本质区别。

### 6.3 Ablation 里两个数字的比例很有意思

- 只加 tactile（不加 attention bias）：51.7% → 74.2%，**+22.5%**
- 再加 attention bias：74.2% → 83.3%，**+9.1%**

modality 引入的提升是 attention 改进提升的 2.5 倍。说明 **信息源比处理方式更重要**，但处理方式仍然显著，验证了 "sparse tactile 不能 symmetric 处理" 这个核心 hypothesis。

---

## 七、几个 Karpathy 会关心的设计直觉

### 7.1 用同一个 VAE 编码 tactile 是个 lucky discovery

paper 没训 tactile-specific encoder，直接用 Wan2.1 的 video VAE 把 tactile image 也编码了。Figure 6 的 t-SNE 显示：**不同 action 对应的 tactile image 在 latent space 天然分簇**。

为什么 web video VAE 能编码 tactile？因为 tactile image 本质上是 "illumination-free 的 surface deformation map"——它和视频里的 surface deformation 在 low-level statistics 上同构。视频 VAE 学到的 "局部形变先验" 直接迁移过来。

这是个 profound 的发现：**tactile community 可能不需要专门 design tactile encoder**，可以直接复用大规模预训练的 video encoder。

参考: Cosmos-Predict2 / Wan2.1 (https://arxiv.org/abs/2511.00062)

### 7.2 Asymmetric bias 的深层含义

$(1-M_i) M_j$ 隐含一个假设：**action 需要 tactile 信息，但 tactile 不需要额外从 action 获取信息**。

这在 contact-rich 任务里成立——action 应该由 tactile 状态决定，反过来不对。但有些任务这个假设可能反过来。比如 "tactile-driven visual search"——你摸到个东西，需要转头去看是什么——这里 tactile query 看 visual key 反而更重要。未来工作可以探索双向 asymmetric bias。

### 7.3 零参数 gate 的哲学

整个 gate $g_t$ 没有任何 learned 参数。这背后是一种笃信：**"contact 时 tactile 剧烈变化" 这个先验足够强，硬编码就行，别浪费样本去学**。

学出来的 gate 有几个风险：
- 训练分布 overfit
- 收敛慢
- 在 OOD sensor 上崩

硬编码的 gate 只要在 sensor noise profile 不变的时候永远 robust。paper 的 OOD 实验也印证了这一点。

代价：换个 tactile sensor（noise profile 不同）就要重新调 $(m, s, k)$。但这是几行代码的事，比 retrain 一个 gating network 便宜太多。

---

## 八、和兄弟工作的关系

- **Cosmos Policy (https://arxiv.org/abs/2601.16163)**：直接的 baseline，vision-only WAM。Dream-Tac 是在它基础上加 tactile + CASA + FlashBias，继承 Cosmos-Predict2 backbone。+31.6% 的提升就来自这些增量。
- **ForceVLA (https://arxiv.org/abs/2505.22159)**：1D force-torque 信号。Play Mahjong 100% vs 55% 是 spatial tactile image vs scalar force 的胜利。
- **RDP (Reactive Diffusion Policy, https://arxiv.org/abs/2503.02881)**：slow-fast 双系统。Dream-Tac 是 pure diffusion，但通过 joint tactile future prediction 实现 "predictive tactile"，不需要 fast reactive module。
- **Tactile-VLA (https://arxiv.org/abs/2507.09160)**：tactile 接入 VLA 但不是 world model formulation，没有 future tactile prediction。
- **Mask World Model (https://arxiv.org/abs/2604.19683)**：同组前作，关注 masking strategy，跟本文 contact-aware attention 是不同 axis 的 robustness 设计。

---

## 九、诚实说局限

paper 自己承认：
1. 只有 6 个任务，broader generalization 没验证
2. gate 仍基于 simple frame-to-frame difference，subtle / long-horizon pattern 可能 miss
3. 1.6Hz 还是慢，diffusion-based WAM 比 reactive policy 慢一个数量级

我再加几点：
4. 没显式 force closure / grasp wrench 约束
5. 假设单 gripper 2 fingertip，multi-finger dexterous 不直接适用
6. Gate 是 timestep 级别，token 级别 gating 可能更精细，但和 FlashBias trick 不兼容
7. 完全 real-world demo，100 个/task 的 data 在小任务上够用，scale 到 100+ task family 不确定
8. 没做 sim-to-real，但完全 real-world 的好处是结果 trustworthy

下一步明显方向：distillation 到 single-step、scale to more tasks、加 multi-finger、探索 token-level gating。

---

## 十、一句话总结

Dream-Tac 的核心 insight 就一句：**tactile 是稀疏事件流，不能当普通 modality 处理，得装个"事件触发的开关"动态调制 attention**。整个 paper 围绕这个 insight 做了三件事：用一个零参数的 frame-difference gate 实现这个开关、用 FlashBias 的 rank-one trick 把开关偷偷塞进 FlashAttention、用 step cache 把 10 步 diffusion 压到 2 步。最终在 6 个真实任务上把 contact-rich manipulation 的成功率从 51.7% 拉到 83.3%。

整篇 paper 的美学：**用先验硬编码 inductive bias、用 low-rank trick 保 fused kernel、用 cache 偷 diffusion 的懒**。这种 "工程上够用就行、理论上不追求 elegance、但每个设计点都有清晰 motivation" 的风格，是务实 robotics research 的范本。

GitHub: https://github.com/LYFCLOUDFAN/Dream-Tac

---

# Dream-Tac: 给 World Action Model 装上"触觉预知"的尝试

下面我用一份"Karpathy 风格的技术笔记"来拆这篇 paper。核心直觉、关键公式、实现 trick、实验数据都尽量讲透。

---

## 1. 问题动机：为什么 Vision-only WAM 在 contact-rich 任务上崩

World Action Model 的核心思想：把 action generation 嵌入到一个对未来 observation 做预测的 generative 过程里，让 policy 继承 video foundation model 在 web data 上学到的 dynamics priors。公式上，标准 WAM 联合建模：

$$p(a_{1:H}, v_{1:T} \mid o, l) = p(v_{1:T} \mid o, l)\, p(a_{1:H} \mid o, l, v_{1:T})$$

其中 $o$ 是当前 RGB observation，$l$ 是 language instruction，$v_{1:T}$ 是 future visual observations（horizon $T$），$a_{1:H}$ 是 action chunk（horizon $H$）。这个 factorization 的好处是 action generation 有了一个 "imagined future" 作为 predictive scaffold。

但 contact-rich 任务（USB 插入、削黄瓜、切香蕉）有几个 RGB 无法解决的问题：
- **Contact state 不可见**：gripper 是否真正碰到物体、压入多深、是否滑脱——这些在 RGB 里几乎完全 ambiguous。Figure 1(a) 直观展示了：contact 前后 RGB 帧看不出差异，但 tactile image 强烈变化。
- **Local geometry 模糊**：USB 端口对齐、刀刃切入角度，这些亚毫米级的信息超出 RGB 分辨率 + 第三视角的感知能力。
- **Force/closure 状态不可观测**：grasp 是否稳定、slip 是否发生，本质是 physical 信号。

paper 提出的关键观察是：**tactile signal 在时间上极其 sparse 且 transient**——大部分时间几乎不变，偶尔出现 contact onset / slip / release 的尖峰。如果像 vision token 一样均匀地把 tactile token 喂给 transformer 的 self-attention，那么 "静止期" 的弱相关 tactile 信号会持续被 attend 到，**稀释了关键 contact 事件的影响**。这是 Dream-Tac 设计 CASA 的根本动机。

---

## 2. Dream-Tac 的核心 formulation

Dream-Tac 把建模目标从 vision-only 扩展到 visuo-tactile：

$$p(a_{1:H}, v_{1:T}, x_{1:T} \mid o, x, l)$$

其中 $x$ 是当前 tactile observation（左右两个 fingertip sensor 的 RGB），$x_{1:T}$ 是 future tactile observations。这个 joint distribution 同时预测：
- 未来视觉 $v_{1:T}$（场景如何演化）
- 未来触觉 $x_{1:T}$（contact 如何演化）  
- 动作 chunk $a_{1:H}$（怎么执行）

关键点：tactile 既是 **conditioning modality**（输入侧 $x$），又是 **prediction target**（输出侧 $x_{1:T}$）。这让 action tokens 通过 bidirectional self-attention 不仅看未来 visual state，还能看到未来 tactile state，等于让 policy "想象自己执行动作后的触觉感受"，再反过来约束 action 生成。

这种 "joint denoising" 的好处是：当 future tactile 预测出 "持续强 contact" 时，action tokens 会被 push 到能产生这种 tactile future 的 action 分布上，形成一种 self-consistent 的 action-tactile coupling。

---

## 3. 架构总览

Backbone 选择很关键：paper 用的是 **Cosmos-Predict2-2B Video2World checkpoint**（基于 Wan2.1 的 video DiT），reuse 它的：
- **T5 text encoder**：把 language instruction $l$ 编码，通过 cross-attention 注入所有 tokens
- **Video VAE**：把 RGB frame 编码到 latent video tokens；**关键 trick 是 tactile image 也复用同一个 VAE**，不另训 tactile-specific encoder

在 backbone 之上：
- robot state + action chunk：following Cosmos-Policy，pad 成 latent-frame tokens 插入 video 序列
- tactile observation：通过同一个 VAE 编码到 latent space，作为 conditioning prefix
- 未来 visual/tactile/action 的 latents 都在同一序列里被 jointly denoise

**Figure 6 的 t-SNE 是一个非常重要的 sanity check**：用未微调的 Wan VAE 编码不同 action 对应的 tactile images，发现 latent space 已经天然分簇。这意味着视频 VAE 在 web data 上学到的 local texture/deformation prior 对 tactile image 也适用——tactile image 本质上是一种 "deformation texture map"，和视频中的 surface deformation 在统计上同构。这是个 lucky but profound 的 observation，避免了训练 tactile encoder 的开销。

---

## 4. Contact-Aware Self Attention (CASA) —— 最核心的创新

### 4.1 问题：为什么不能直接 concat tactile tokens

标准 self-attention 对所有 modality symmetric 处理：

$$\text{logit}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}$$

但 tactile 信号的统计特性和 vision 完全不同：
- 时间稀疏：episode 中可能 80% 时间 tactile image 不变
- 事件驱动：contact onset 是瞬态、关键的
- 信息密度高度集中：少数几帧决定 task success

如果用 symmetric attention，non-tactile queries（action, visual, proprioceptive tokens）会在静止期也 attend 到几乎不变的 tactile keys，把这些 "dead" tactile 信号累积进 representation，**削弱对真正 contact 事件的 sensitivity**。

### 4.2 CASA 的设计：gated asymmetric additive bias

核心公式 (Eq. 6)：

$$\text{logit}_{ij} = \frac{q_i^\top k_j}{\sqrt{d}} + \alpha\, g_t\, (1 - M_i)\, M_j$$

变量解释：
- $i, j$：flattened token index
- $q_i, k_j$：query / key vectors
- $d$：head dimension（用于 scaled dot-product 的归一化）
- $\alpha > 0$：bias magnitude 的全局 scale，paper 设 $\alpha = 2.0$
- $g_t \in [g_{\min}, g_{\max}] = [0.15, 1.0]$：**timestep-level contact gate**，标量
- $M_i \in \{0, 1\}$：query $i$ 是否属于 tactile token
- $M_j \in \{0, 1\}$：key $j$ 是否属于 tactile token

**关键设计点：asymmetric indicator $(1-M_i) M_j$**：
- 当 $M_i = 1, M_j = 1$（tactile-to-tactile）：bias = 0，不变
- 当 $M_i = 0, M_j = 0$（non-tactile-to-non-tactile）：bias = 0，不变
- 当 $M_i = 1, M_j = 0$（tactile query 看 non-tactile key）：bias = 0
- **当 $M_i = 0, M_j = 1$（non-tactile query 看 tactile key）：bias = $\alpha g_t$** ✓

这是个 directional 的 inductive bias：只允许 action/visual/proprioceptive tokens **额外**加强对 tactile tokens 的 attention，且加强程度由 $g_t$ 控制。tactile tokens 之间保持原始 attention，避免自我放大。

### 4.3 Gate $g_t$ 的计算：纯数据驱动，零学习参数

这是 paper 最优雅的地方——**gate 完全从 raw tactile RGB 计算，无 learned gating network**。

第一步，对每个 fingertip view 计算相邻帧的 normalized mean absolute difference (Eq. 7)：

$$\delta_t^L = \frac{1}{255} \mathbb{E}_{p, c}\left[|I_t^L(p, c) - I_{t-1}^L(p, c)|\right]$$

$$\delta_t^R = \frac{1}{255} \mathbb{E}_{p, c}\left[|I_t^R(p, c) - I_{t-1}^R(p, c)|\right]$$

变量：
- $I_t^L, I_t^R \in \{0, ..., 255\}^{H \times W \times 3}$：时刻 $t$ 的左/右指尖 tactile image
- $p$：spatial location (像素位置)
- $c$：RGB channel
- 除以 255 做归一化，让 $\delta \in [0, 1]$

第二步，per-timestep event strength (Eq. 8)：

$$\rho_t = \max(\delta_t^L, \delta_t^R)$$

用 max 而非 sum/mean：单边 contact 也应触发 gate，max 对 sparse bilateral contact 更敏感。$\rho_0 = 0$（首帧无前驱）。

第三步，robust normalization + bounded sigmoid (Eq. 9)：

$$z_t = k \frac{\rho_t - m}{s + \epsilon}$$
$$\tilde{g}_t = \sigma(z_t) = \frac{1}{1 + e^{-z_t}}$$
$$g_t = g_{\min} + (g_{\max} - g_{\min}) \tilde{g}_t$$

变量：
- $m, s$：**fixed** reference location/scale（不按 dataset 估计，median-MAD 风格的先验）
- $k$：sigmoid sharpness（控制 transition 陡峭程度）
- $\epsilon = 10^{-6}$：numerical stability
- 实现: $(m, s, k, \epsilon) = (0.002, 0.001, 4, 10^{-6})$
- $z_t$ clip 到 $[-30, 30]$ 防止 sigmoid 饱和
- $[g_{\min}, g_{\max}] = [0.15, 1.0]$

为什么 $g_{\min} = 0.15$ 而不是 0？保留一点 baseline tactile attention，避免完全静默 tactile 信号。为什么 $g_{\max} = 1.0$？避免过度放大造成 attention collapse。

**直觉**：当 tactile 几乎不变（静止期），$\rho_t \approx 0$，$z_t \approx k(0 - 0.002)/0.001 = -8$，$\sigma(-8) \approx 3.4 \times 10^{-4}$，$g_t \approx 0.15 + 0.85 \times 3.4 \times 10^{-4} \approx 0.15$。当 tactile 强烈变化（contact onset），$\rho_t = 0.01$，$z_t = 4 \times 0.008/0.001 = 32$，被 clip 到 30，$\sigma(30) \approx 1$，$g_t \approx 1.0$。从 0.15 到 1.0 的约 6.7× 动态范围，正是 paper 在 Section 4.4.2 实测的。

### 4.4 Gate 行为的实证分析

Section 4.4.2 的统计很有意思（基于 Peel Cucumber 的 5 个 episode，874 个 timesteps）：
- $\rho_t$ median: $1.73 \times 10^{-3}$
- $\rho_t$ 90th percentile: $6.08 \times 10^{-3}$
- $\rho_t$ mean: $2.73 \times 10^{-3}$, std: $2.32 \times 10^{-3}$
- coefficient of variation (std/mean) ≈ 0.85 → **高度 skewed 分布**，符合 "sparse event" 假设
- 每个 episode 内 $g_t$ 跨越约 0.85 的 $[g_{\min}, g_{\max}]$ 区间
- 时间平均 $\bar{g}_t \in [0.48, 0.61]$：大部分时间处于 low-to-mid regime，尖峰短促但显著

Figure 7 的轨迹分析揭示了一个非常符合直觉的 temporal structure：
- **Approach 阶段**（gripper 接近 cucumber 但未接触）：$g_t$ 周期性小波动，源于 sensor noise + minor non-contact disturbance，gate 保持在 low range
- **Contact 阶段**：$\rho_t$ 急升，$g_t$ 进入 sustained high regime
- **Post-contact 阶段**：gate 回落

Episode_2 是个有趣的 outlier——trajectory 起始就处于 high gate，因为该 episode 的初始 pose 已经接近物体。这印证了 gate 的 **数据驱动** 特性：它跟踪的是 "实际 contact 强度" 而非 phase label。

### 4.5 重要 caveat：gate 只做 coarse 调制

paper 在 Section 4.4.2 末尾明确强调："The fine-grained allocation of attention across tactile tokens remains governed by the content term $q^\top k$; $g_t$ only modulates, at a coarse level, how strongly non-tactile queries are biased toward tactile keys."

也就是说，gate 是 **per-timestep 的全局开关**，不区分 tactile token 之间的重要性。token-level 的细粒度 attention 仍由 content-based $q^\top k$ 决定。这是个工程上合理的简化——如果要做 token-level gating，需要额外的 learned gating network，会破坏 Section 4.6 描述的 FlashBias 加速 trick。

---

## 5. 训练目标

Dream-Tac 复用 pretrained video model 的 latent denoising objective。给定 clean prefix context $(o, x, l)$ 和 noisy target latents：

$$\tilde{y} = y + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad \sigma \sim p(\sigma)$$

其中 $y = \{z^v_{1:T}, z^x_{1:T}, z^a_{1:H}\}$：
- $z^v_{1:T}$：future visual latents
- $z^x_{1:T}$：future tactile latents
- $z^a_{1:H}$：action latents

Loss (Eq. 11)：

$$\mathcal{L}_{denoise} = \mathbb{E}_{y, \epsilon, \sigma}\left[\|f_\theta(\tilde{y}, \sigma, o, x, l) - \epsilon\|_2^2\right]$$

等价的 modality 分解 (Eq. 12)：

$$\mathcal{L} = \mathcal{L}_{act} + \lambda_v \mathcal{L}_{img} + \lambda_t \mathcal{L}_{tac}$$

$\lambda_v, \lambda_t$ 平衡三个 denoising sub-loss。这种 joint denoising 的妙处：**一个 forward pass 同时监督 action generation、future visual prediction、future tactile prediction**，三个任务在 shared representation 上互相 regularize。

Noise schedule 用 hybrid EDM（继承自 Cosmos-Predict2）：
- log-normal component: $(p_\mu, p_\sigma) = (\ln 4, 1.2)$, 范围 $[0.01, 200]$
- uniform component: 范围 $[1, 85]$
- observed context latents 保持 clean ($\sigma = 0$)
- action + future prediction tokens **同一 forward pass 内 jointly noised**

这是 rectified-flow 风格的 design，比纯 DDPM 的 schedule 更适合 video latent。

---

## 6. Dual-Level Acceleration —— 让 WAM 真的能跑实时

WAM 的部署瓶颈：DiT 的 quadratic self-attention + 多步 iterative denoising，加上 tactile tokens 增加的 temporal/modality 复杂度，原版 10 步 denoising 在 A800 上只有 5Hz，对 robot control 远远不够。

### 6.1 Training 端：FlashBias reformulation

问题：CASA 的 additive bias $\alpha g_t (1-M_i) M_j$ **不能直接塞进 FlashAttention**——FlashAttention 的 fused kernel 假设 logit = $\frac{q^\top k}{\sqrt{d}}$，任何 dense additive bias 都会破坏 fused path，导致需要 materialize 完整 $S \times S$ attention matrix，HBM 访问爆炸。Cosmos-Policy 的 naive 实现就是这个状态：full setting 训练 80.82s/iter。

paper 用了 **FlashBias-style 的 low-rank reformulation**。核心 observation：CASA 的 bias 是 **rank-one** 的：

$$\Delta_{ij} = \gamma_b\, a_i\, b_j$$

其中 $\gamma_b$ 吸收了 $\alpha g_t$ 和 fixed scale，$a_i, b_j \in \{0, 1\}$ 是 query/key 的 tactile mask。这可以分解为：

$$\Delta_{ij} = u_i v_j, \quad u_i = \sqrt{\gamma_b}\, a_i, \quad v_j = \sqrt{\gamma_b}\, b_j$$

然后 **augment query 和 key 各加一个 scalar channel** (Eq. 14)：

$$\tilde{q}_i = \left[\frac{1}{\sqrt{d}} q_i \,\|\, u_i\right], \quad \tilde{k}_j = \left[k_j \,\|\, v_j\right]$$

内积 (Eq. 15)：

$$\langle \tilde{q}_i, \tilde{k}_j \rangle = \frac{q_i^\top k_j}{\sqrt{d}} + u_i v_j = \frac{q_i^\top k_j}{\sqrt{d}} + \Delta_{ij}$$

完美等价于原始 biased attention logits，但形式上是标准 dot-product，可以喂给标准 FlashAttention kernel！

代价：每 token 每 head 多 $O(1)$ 个 channel，对齐 padding 用 0 填充。这避免了 $O(S^2)$ bias matrix 的 materialization，保留了 fused attention path。

效果（Table 5 左图）：
- 无 tactile 无 bias：19.02s → 10.49s (1.8× speedup)
- 有 tactile 无 bias：24.59s → 15.97s (1.5×)
- **有 tactile 有 bias：80.82s → 27.48s (2.9×)** ← 关键数据

关键 insight：tactile tokens 本身只增加 moderate overhead（24.59s vs 19.02s，约 29% 增加），**真正的 bottleneck 是 structured attention bias 的 naive 实现**。FlashBias reformulation 把这个 bottleneck 干掉了。

### 6.2 Inference 端：Diffusion-Step Cache

paper 在 Appendix B.2 做了一个重要的诊断实验（Figure 11）：相邻 denoising step 的 action latent **cosine similarity ≈ 0.997**——这意味着绝大多数 denoising step 的输出几乎不变，存在巨大的计算冗余。

但 paper 还测试了 timestep embedding 作为 cache 指示器的可行性——结论是 **不可靠**：timestep embedding 的 relative $L_1$ 距离和 cosine similarity 都和 action latent 的变化不相关。这说明 TeaCache [32] 那种 "用 timestep embedding 预测输出变化" 的策略在 WAM action latent 上不适用，因为 action latent 的 dynamics 不只由 timestep 决定，还受 context conditioning 强烈影响。

paper 的简化策略：**只在第 1 步和第 3 步做 full forward**（第 3 步是 validation set 上观察到变化最大的 step），其余步骤 reuse cached results。10 步 → 2 步，latency 1109ms → 619ms，**1.8× speedup，success rate 不降（85% maintained）**。

这里其实有个值得深究的设计选择：为什么是第 1 步和第 3 步？rectified-flow / flow matching 的 trajectory 通常在前几步变化最大（高 noise level 的 coarse-to-fine transition），后期是 refinement。第 1 步建立 coarse structure，第 3 步做关键 correction，后续 refinement step 之间高度相似，可以 cache。这是个经验性的 sweet spot，paper 没给出更系统的 step selection 方法，是个 potential future work。

---

## 7. 实验：六个真实 contact-rich 任务

### 7.1 任务设置

六个任务覆盖了不同 contact 类型：

| Task | Contact 类型 | 关键挑战 | Avg Episode Length | Max Steps |
|---|---|---|---|---|
| Pick Baguette | Deformable grasp | 控制变形 + 保持支撑 | 733 | 1200 |
| Insert USB | Precision insertion | 毫米级对齐 + force regulation | 618 | 1100 |
| Clean Whiteboard | Surface wiping | 持续 contact + friction | 833 | 1400 |
| Peel Cucumber | Blade contact | 曲面 contact 稳定性 | 220 | 450 |
| Play Mahjong | **Visual occlusion** | **纯 tactile 推理** | 200 | 400 |
| Cut Banana | Blade penetration | Force regulation + 防滑 | 277 | 500 |

硬件：Franka Emika Panda + 2× RealSense D435i（第三人称 + wrist）+ 2× Xense Photon tactile sensor（fingertip）。100 demos/task，30Hz 采集。

### 7.2 主结果（Table 1 + Figure 3）

| Method | Pick Baguette | Insert USB | Clean WB | Peel Cuc | Play Mahjong | Cut Banana | **Avg SR** |
|---|---|---|---|---|---|---|---|
| $\pi_0$ | 75 | 5 | 40 | 35 | 0 | 30 | 30.8% |
| $\pi_{0.5}$ | 90 | 10 | 70 | 50 | 30 | 20 | 45.0% |
| ForceVLA | 90 | 15 | 75 | **90** | 55 | 30 | 50.8% |
| Cosmos Policy | 95 | 15 | 65 | 50 | 35 | 50 | 51.7% |
| **Dream-Tac** | **100** | **35** | **90** | 85 | **100** | **90** | **83.3%** |

几个关键 observations：

1. **Pick Baguette 上所有方法都不错**——coarse manipulation 主要靠 vision + basic motion control，tactile 边际收益小。
2. **Dream-Tac 最大优势在 Insert USB 和 Cut Banana**——这两类任务需要精确 contact transition 推理 + spatial alignment + force interaction，是 vision-only 方法的死角。
3. **Play Mahjong 上 Dream-Tac 100%，ForceVLA 只有 55%**——这个任务被故意设计成 **完全视觉遮挡**，必须靠 tactile 判断牌面。Dream-Tac 的 attention bias 强制 policy attend tactile cue，而 ForceVLA 仍倾向于依赖 vision prior。这是 paper 想论证的 "tactile-grounded policy vs tactile-as-auxiliary" 的核心区别。
4. **Peel Cucumber 上 ForceVLA 略胜**（90% vs 85%）——这个任务 force-torque 信号（1D 时序）可能比 tactile image（2D spatial）更直接相关，说明 force 和 tactile 是互补 modality，不是替代关系。

### 7.3 Ablation（Table 1）

| Variant | Tactile | Attn Bias | Avg SR |
|---|---|---|---|
| Visual WAM | × | × | 51.7% |
| Visuo-tactile WAM | √ | × | 74.2% |
| Visuo-tactile WAM + Bias | √ | √ | 83.3% |

- **加 tactile 提升最大**：51.7% → 74.2%，**+22.5%**。说明 tactile modality 本身是核心 information source。
- **加 CASA 再提升**：74.2% → 83.3%，**+9.1%**。说明在已经有 tactile 的基础上，**selective emphasis** 比均匀处理更有效——验证了 "sparse tactile signal 不能 symmetric 处理" 的核心 hypothesis。

这两个数字的比例很有意思：modality 引入 : attention 改进 ≈ 2.5 : 1。说明 information 来源比处理方式更重要，但处理方式仍贡献显著。

### 7.4 Generalization（Figure 4）

四类 OOD variations：

| Variation | Task | Dream-Tac | Cosmos-Policy |
|---|---|---|---|
| Table height ±5cm | Peel Cuc | 85/90/75 | 65/30/0 |
| Spatial arrangement | Pick Bag | 100/80 | 100/80 |
| Object appearance | Mahjong | 100/85 | 35/15 |
| Background | Cut Ban | 90/70 | 40/25 |

Dream-Tac 在 tactile-relevant variations（table height、object appearance）上 generalization 优势巨大，在 visual-only relevant variations（spatial、background）上和 Cosmos-Policy 持平或略胜。这暗示 tactile signal 提供的 contact-aware representation 对 visual priors 是 **complementary** 而非 redundant 的。

---

## 8. 训练超参细节（Appendix A.5）

- Base checkpoint: Cosmos-Predict2-2B Video2World
- Optimizer: Fused Adam, lr = $10^{-4}$, $(\beta_1, \beta_2) = (0.9, 0.99)$, $\epsilon = 10^{-8}$, weight decay = 0.1
- Precision: mixed bfloat16
- LR schedule: warmup 2000 steps (linear) → decay 1.0 → 0.3 over 20000 steps → fixed 0.06
- Input: 224×224 RGB, 30Hz demos
- Action chunk: $H = 20$
- Batch per GPU: vision-only 25, visuo-tactile 16（tactile tokens 占显存）
- $\alpha = 2.0$（attention bias scale）
- Gate hyperparams: $(m, s, k, \epsilon) = (0.002, 0.001, 4, 10^{-6})$, $[g_{\min}, g_{\max}] = [0.15, 1.0]$
- Text dropout: 0, EMA disabled

注意 text dropout = 0 意味着 paper 没做 classifier-free guidance 的 conditional dropout——可能因为 instruction-conditioned manipulation 的 instruction 不可缺失，CFG 训练会破坏 conditioning 语义。

---

## 9. 我的几点直觉性思考

### 9.1 为什么不学一个 gating network

CASA 的 gate 完全从 raw tactile RGB 的 frame-to-frame difference 算出，**零额外参数**。这是非常 Karpathy-style 的设计：先验知识（"contact 时 tactile 会剧烈变化"）直接 hard-code 进 inductive bias，避免 learned gate 的训练不稳定 + 慢收敛。缺点是：如果 tactile sensor 的 noise profile 变了，固定 hyperparams $(m, s, k)$ 可能不 optimal——但 paper 的 generalization 实验说明这个简化在他们的 sensor 上 robust。

### 9.2 Asymmetric bias 的深层含义

$(1 - M_i) M_j$ 这个 asymmetric mask 隐含一个 strong assumption：**action/visual token 需要 tactile 信息，但 tactile token 不需要额外从 action/visual 获取信息**。这在 contact-rich 任务里成立（action 由 tactile 状态决定），但在某些任务（如 tactile-driven visual search）可能反向也重要。未来工作可以探索双向 asymmetric bias。

### 9.3 Pretrained VAE 通用性的好运

Figure 6 的 t-SNE 是个 lucky observation：Wan VAE 在 web video 上学到的 local deformation prior 直接 transfer 到 tactile image。这可能因为 tactile sensor image 本质上是 "illumination-free 的 surface deformation map"，和视频中的 surface deformation 在 low-level statistics 上同构。这个发现对 tactile community 很有价值——可能不需要专门 design tactile encoder。

### 9.4 Timestep embedding 失效的 implication

Appendix B.2 报告 timestep embedding 不能作为 cache indicator，这是个 negative result 但很重要。它暗示 WAM 的 action latent dynamics **不像 video generation 那样 timestep-monotonic**——action latent 同时受 conditioning context 和 noise level 影响，且 conditioning 主导。这意味着 video domain 的加速 trick 不能无脑迁移到 WAM。

### 9.5 5Hz → ~1.6Hz 的真实意义

10 步 denoising 5Hz（200ms）→ 2 步 cache 1.6Hz（619ms）。1.6Hz 对精细 manipulation 仍偏慢（典型 reactive policy 是 10-20Hz，比如 RDP [45] 的 fast tactile feedback）。这是 diffusion-based WAM 的根本限制，paper 在 Limitations 里也承认了。下一步可能需要 distillation 到 single-step 或 non-diffusion backbone。

---

## 10. 与相关工作的 positioning

- **Cosmos Policy [26]**：Dream-Tac 的 direct baseline，vision-only WAM。Dream-Tac 在它基础上加 tactile + CASA + FlashBias，继承其 Cosmos-Predict2 backbone。
- **ForceVLA [49]**：用 1D force-torque 信号，是 tactile image 的简化版。Play Mahjong 上 Dream-Tac 100% vs ForceVLA 55% 的对比很有说服力——spatial tactile pattern 比 scalar force 信息密度高得多。
- **RDP [45]**：slow-fast 双系统，diffusion policy + fast tactile reactive。Dream-Tac 是 pure diffusion，没用 hybrid 架构，但 joint tactile prediction 提供了一种 "predictive tactile" 替代 reactive tactile。
- **Tactile-VLA [24]**：把 tactile 接入 VLA，但不是 world model formulation，没有 future tactile prediction。
- **Mask World Model [33]**（同作者组的前作）：关注 robust policy learning 的 masking strategy，和本文的 contact-aware attention 是不同 axis 的 robustness 设计。

---

## 11. Limitations 的诚实评估

paper 在 Appendix D 自承三点：
1. 任务覆盖有限（六个任务），broader generalization 未验证
2. Gate 仍基于 simple frame-to-frame variation，可能 miss subtle / long-horizon interaction pattern
3. Diffusion-based WAM 仍比 reactive policy 慢

我再加几点：
4. **No force closure modeling**：tactile 只用了 image-level signal，没有显式 grasp wrench 约束
5. **Single-gripper assumption**：tactile signal 只有 2 fingertip，对 multi-finger dexterous 不直接适用
6. **Gate 是 timestep-level**：token-level gating 可能更精细，但和 FlashBias trick 不兼容
7. **No sim-to-real**：完全 real-world demo，100 个/task 的 data 量在小任务上够用，但 scale 到 100+ task family 不确定

---

## Reference

- Dream-Tac paper 本身（用户已提供）
- Cosmos-Policy: https://arxiv.org/abs/2601.16163
- Wan2.1 / Cosmos-Predict2: https://arxiv.org/abs/2511.00062
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- FlashBias: https://openreview.net/forum?id=7L4NvUtZY3
- TeaCache: https://arxiv.org/abs/2411.19108
- $\pi_0$: https://arxiv.org/abs/2410.24164
- $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
- ForceVLA: https://arxiv.org/abs/2505.22159
- RDP (Reactive Diffusion Policy): https://arxiv.org/abs/2503.02881
- Tactile-VLA: https://arxiv.org/abs/2507.09160
- Dream-Tac GitHub: https://github.com/LYFCLOUDFAN/Dream-Tac
- Mask World Model (同组前作): https://arxiv.org/abs/2604.19683
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- Token Expand-Merge (同组，WAM 加速): https://arxiv.org/abs/2512.09927

---

**一句话总结 intuition**：Dream-Tac 把 tactile 视为 "稀疏事件流" 而非 "持续 modality"，通过一个零参数的 frame-difference gate 在 attention logit 上做 asymmetric amplification，再用 FlashBias 的 low-rank reformulation 把这个 bias 偷偷塞进 fused attention kernel，最终让一个 video DiT 在保留 web-data pretrained prior 的同时学会 "想象未来的触觉" 来引导 action 生成。整个 design 充满 "用先验知识 hard-code inductive bias、用 low-rank trick 保 fused kernel" 的工程美学。
