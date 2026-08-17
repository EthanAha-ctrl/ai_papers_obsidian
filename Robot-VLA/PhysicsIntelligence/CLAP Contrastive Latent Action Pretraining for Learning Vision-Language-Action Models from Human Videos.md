---
source_pdf: CLAP Contrastive Latent Action Pretraining for Learning Vision-Language-Action
  Models from Human Videos.pdf
paper_sha256: 12b7f19703505c8b19919b2d0201b00f204896ec3c9ffc48f45c4ccafce1c8f1
processed_at: '2026-08-03T15:44:52-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个白板讲解的方式，把 CLAP 讲透。

## 一句话总结

机器人 teleop data 太贵太少，YouTube 上 human video 倒是海量. CLAP 干的事: 让 human video 里的"手部动作"自动对齐到机器人能执行的离散 action codebook 上, 这样海量无标注视频就能反过来教机器人学 skill.

## 问题到底出在哪

先看现有 LAM (Latent Action Model) 在干嘛.

你给模型两帧视频 $\mathbf{o}_t$ 和 $\mathbf{o}_{t+H}$, 让它推断中间发生了什么 action $z$, 再用 $z$ 重建下一帧. 听起来合理, 但重建 $\mathbf{o}_{t+H}$ 这件事里混进了太多非 action 的东西: 光照变了、相机抖了、有人从背后走过、背景里的窗帘飘了一下. 模型为了降低重建 loss, 会把这些 nuisance 全塞进 latent $z$ 里.

结果就是: 你拿这个 $z$ 去控制机器人, 机器人根本不知道该干啥, 因为 $z$ 里一半信息是"窗帘飘了". 所以 UniVLA 这种方法在 real-world 实验里全线崩溃 (Table I 只有 35% 平均成功率).

**根本矛盾**: video 的 visual dynamics 和 robot 的 physical action 是两个不同 distribution, 你不能靠一个 reconstruction loss 就指望它们对齐.

## CLAP 的核心 trick

CLAP 的做法分两步, 我觉得最精彩的是第一步.

### Step 1: 先造一个"机器人动作词典"

用 robot data 训一个 VQ-VAE (Act-VAE), 把连续的 14 维 action chunk 量化成离散 token. 训完之后你手上有 256 个 codebook 向量, 每个对应一种"原子动作".

这一步不稀奇, LAPA 和 UniVLA 也都做了. 关键在第二步.

### Step 2: 强制 video latent 落到这个词典上

给定 video 帧对 $(\mathbf{o}_t, \mathbf{o}_{t+H})$, encoder 输出两个 latent:

- $\mathbf{z}_{v,a}$: action-related, **强制量化到 Act-VAE 的 frozen codebook**
- $\mathbf{z}_{v,i}$: action-irrelevant, 量化到一个独立的 env codebook

然后做两件事:

1. **Contrastive loss**: 让 $\mathbf{z}_{v,a}$ 和 robot action latent $\mathbf{z}_a$ 在 batch 内做 SigLIP 对比学习. 有 robot label 的 sample 就是正常 positive pair, human video 就拿自己当 positive anchor, batch 内其他 sample 当 negative.

2. **L1 regularization on $\mathbf{z}_{v,i}$**: 强迫 action-irrelevant latent 稀疏, 逼 decoder 优先用 $\mathbf{z}_{v,a}$ 来重建, 防止 decoder 偷懒把所有信号都塞到 irrelevant stream.

这个设计的 intuition 非常 clean: **你告诉模型, video 里的 visual change 如果想被解释成 action, 必须落在机器人已知的 256 种原子动作里**. 背景变化、光照变化这些 nuisance 没法对应到任何 codebook cell, 就被挤到 $\mathbf{z}_{v,i}$ 那条路去了.

Fig. 1 的可视化很说明问题: 同一个 codebook cluster 里, Ego4D 的人类视频、AgiBot 机器人、Astribot 机器人的动作语义完全一致 (都是"向右移动"、"放下"、"抓取"). 更狠的是, 他们把 latent decode 回 3D 轨迹再投影到 2D 图像, 红色箭头和实际物体运动方向对得上 — 这说明 latent 真的是 physical action, 不只是 semantic label.

## Dual formulation: 为什么需要两个模型

讲完 pretraining, 下一个问题是怎么用这个 aligned latent 做控制.

### CLAP-NTP: 慢但聪明的 reasoning

直接把 action codebook 加到 Qwen3VL-4B 的词表里, 当成 language token 做 next-token prediction. 输入是 observation + instruction, 输出是 subtask 描述 + action tokens.

好处: 完全继承 VLM 的 reasoning 和 instruction following. OOD Pick & Place 上能到 85/80, 比 $\pi_{0.5}$ 还高.

坏处: autoregressive decoding 太慢, 788ms 一步, 做 fine-grained bimanual manipulation 根本来不及.

### CLAP-RF: 快但精细的 control

用 DiT (Diffusion Transformer) 当 action expert, 通过 Rectified Flow 训练连续 action chunk. DiT 通过 cross-attention 读 VLM 的 K/V cache 拿语义 context, 但 stop-gradient 防止 action 的高方差梯度污染 VLM.

为什么用 Rectified Flow 而不是 DDPM? Rectified Flow 在 noise 和 clean action 之间走直线, 推理时少步采样就能到位, latency 压到 183ms, 跟 $\pi_0$ 的 169ms 几乎持平.

为什么要 distill NTP 到 RF? 因为 NTP 的离散 token 学到了好的 high-level planning (subtask 分解, object 识别), RF 在这个语义基础上做 continuous refinement. 这就像先让 GPT 写出粗略代码框架, 再让专门的 formatter 做精细调整.

Table I 的数据印证: 精细任务 (Pack Doll Close, Fold T-shirt) 上 CLAP-RF 明显优于 CLAP-NTP; OOD 泛化上 CLAP-NTP 更强. 互补.

## Knowledge Matching: 为什么 fine-tune 会忘

LIBERO 实验里有个巨大的 domain gap: pre-training 是 dual-arm real-world ego-centric, fine-tuning 是 single-arm simulation third-person. 直接 fine-tune VLM 会让它忘掉 pre-training 学到的东西 — Long 任务从 82% 掉到 64%.

KM 的做法很直接: 保留一份 frozen 的 reference model, fine-tune 时加一个 KL penalty:

$$\mathcal{L}_{\text{KM}} = \alpha D_{\text{KL}}(P_{\text{ref}} \| P_{\text{policy}}) + \mathcal{L}_{\text{RF}}$$

这跟 RLHF 里 PPO 的 KL constraint 同源. 作用是: 你可以适应新 domain, 但不能偏离 reference 太远. 实验里 KM 拿到 91.0%, 比 full fine-tune 高 9 个点.

## 最有说服力的实验

**Fig. 8 (Make Bouquets 泛化)**:

teleop data 只有两个 flower 组合 (红心+黄向日葵), 所有方法在 unseen 组合上 ≤10%. 加入 human video 后:

- CLAP-NTP: unseen 组合到 35% (跟 seen 持平)
- UniVLA: 还是 10% 左右 (video 数据基本没帮上忙)

这个实验直接证明: **CLAP 的 contrastive alignment 确实把 human video 的 manipulability prior transfer 到了 robot policy**. UniVLA 虽然也用了 video, 但因为没做 alignment, video 学到的 latent 和 robot action 是脱节的.

## 我的几个疑问

**Codebook 256 够不够**. 双臂 14DoF, action chunk 32 步, 压到 16 个 token × 256 codebook = 128 bits. 折叠衣服这种精细任务, 128 bits 能表达多精细的轨迹? 我怀疑精细任务的 bottleneck 就在这里.

**Self-anchor contrastive 的有效性**. Human video 没 ground-truth action, positive pair 是 $(\mathbf{z}_{v,a}, \mathbf{z}_{v,a})$ 自己, 学习信号全靠 negative push. 这会不会让 codebook 坍缩到少数高频 cell? Paper 没分析 codebook usage distribution.

**$\lambda_{\text{con}} = 0.1$ 太小**. Reconstruction loss 的量级远大于 contrastive loss, 0.1 的权重可能让 alignment 信号被淹没. 他们没做 $\lambda_{\text{con}}$ 的 sweep, 我觉得这里有调优空间.

## 为什么这个工作 important

LAM 这个方向之前一直有个 conceptual blind spot: 大家默认假设 visual dynamics reconstruction 能学出 action representation. CLAP 明确指出这个假设是错的, latent action 必须有 physical anchor.

这个 insight 推广开去: 任何 cross-embodiment, cross-modal learning 都需要类似的 anchor 机制. 比如未来做 humanoid 全身控制, 可以先造一个 joint trajectory 的 codebook, 再把 human video 的 motion 对齐上去. dexterous hand manipulation 也一样, 先建 gripper action codebook, 再对齐 human hand motion.

Paper 的 limitation 也坦诚: task-level generalization 还做不到, hand-gripper gap 还在, multi-stage training 工程复杂. 但 conceptual contribution — "latent action needs physical grounding" — 是扎实的.

参考链接:
- Project page: https://lin-shan.com/CLAP/
- LAPA (原始 LAM): https://arxiv.org/abs/2410.11758
- UniVLA (baseline): https://univla.github.io/
- $\pi_0$ (SOTA baseline): https://arxiv.org/abs/2410.24164
- SigLIP: https://arxiv.org/abs/2303.15343
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Knowledge Insulating VLA (stop-gradient trick): https://arxiv.org/abs/2505.23705
- VQ-VAE: https://arxiv.org/abs/1711.00937
- AgiBot World dataset: https://arxiv.org/abs/2503.06669
- Ego4D: https://ego4d-data.org/
- LIBERO benchmark: https://libero-project.github.io/

---

# CLAP: Contrastive Latent Action Pretraining 深度解读

## 1. 核心问题的 intuition

VLA 模型面临一个数据规模的不对称性: robot teleoperation data 极其昂贵且场景单一, human egocentric video 却海量多样. 直接的思路是用 human video 来预训练, 但存在一个根本性的 modality gap: robot data 有 action label $\mathbf{a}_t$, human video 只有视觉观测 $\mathbf{o}_t \to \mathbf{o}_{t+H}$, 没有 action.

**Latent Action Models (LAMs)** 的标准做法是 inverse dynamics: 从相邻两帧推断一个 latent action $z$, 再用 forward dynamics 重建下一帧. 这里藏着一个严重的失败模式: 重建 $\mathbf{o}_{t+H}$ 这件事本身是 ill-posed 的, 因为帧间变化里既有 gripper 的物理运动, 也有相机抖动、光照变化、背景物体被人走过挡住等 nuisance factors. 如果直接用重建 loss, latent space 会**entangle** 这些非 action 信息, 学到的 token 既不能直接对应机器人可执行命令, 又把语义信号污染了.

CLAP 的核心 insight 是: **不要让 visual reconstruction 单独定义 latent action, 而是强制让视频推断出来的 latent 落在 robot action 量化的 codebook 上**. 通过 contrastive learning, video 的视觉动力学被"拉"到 robot 的物理可执行空间里, nuisance factors 被挤到一个独立的 latent stream.

参考: Latent Action Models 范式源自 LAPA (arXiv:2410.11758), UniVLA (RSS 2025) 是直接的 baseline, 它依赖重建 loss 而缺乏 alignment, CLAP 正是对它的修正.

---

## 2. 方法架构总览

整个 pipeline 分三个阶段:

### Stage 1: Act-VAE — 构造机器人 action 的"物理语言"

把连续的 dual-arm action chunk $\mathbf{a}_{t:t+H-1} \in \mathbb{R}^{H \times 14}$ (其中 14 = 7DoF×2 arms) 通过 VQ-VAE 编码成离散 token 序列 $\mathbf{z}_a$.

公式 (2) 的三项 loss:
$$\mathcal{L}_{\text{Act}} = \underbrace{\|\mathbf{a} - \mathcal{D}_\psi(\mathbf{z}_q)\|_2^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}(\mathcal{E}_\phi(\mathbf{a})) - \mathbf{z}_q\|_2^2}_{\text{codebook loss}} + \beta \underbrace{\|\mathcal{E}_\phi(\mathbf{a}) - \text{sg}(\mathbf{z}_q)\|_2^2}_{\text{commitment loss}}$$

变量含义:
- $\mathcal{E}_\phi, \mathcal{D}_\psi$: encoder 和 decoder, 都是 Transformer
- $\mathbf{z}_q$: 量化后的离散 latent, 来自 codebook $\mathcal{C} = \{\mathbf{e}_k\}_{k=1}^K$ 中最近邻查找
- $\text{sg}(\cdot)$: stop-gradient, 阻止梯度流过被操作的项
- $\beta$: commitment weight (论文里 = 1.0), 让 encoder 输出不偏离 codebook 太远

**Intuition**: codebook loss 拉近 codebook 向量到 encoder 输出, commitment loss 拉近 encoder 输出到 codebook 向量, 两者都用 stop-gradient 避免互相 collapse. 这种对称设计保证 codebook 既是离散的可索引的, 又能稳定收敛.

### Rate-distortion trade-off

公式 (9) 定义压缩率:
$$r = \frac{N_q \cdot \log(K)}{N_a \cdot D_a \cdot \log(R/\sqrt{\text{MSE}})}$$

变量:
- $N_q$: latent 序列长度 (每条轨迹的 token 数)
- $K$: codebook 大小 (词表大小)
- $N_a$: action chunk size
- $D_a$: action 维度
- $R$: 数据的动态范围
- MSE: 重建误差

Table IV 给出的 ablation 表明: $N_q=16, K=256$ 是 elbow point, PSNR=40.00 dB, $r=0.086$. 选这个点的 intuition 是: 再增大 codebook 或序列长度, PSNR 收益边际递减, 但 VLM 学起来更难 — 因为 attention 会稀释, sequence length 增加让自回归建模复杂度上升.

### Stage 2: VD-VAE — 从视频学习对齐的 latent action

这是 CLAP 最关键的创新. 给定视频帧对 $(\mathbf{o}_t, \mathbf{o}_{t+H})$:

1. 用 frozen DINOv3 提取 patch-level 特征 $\mathbf{f}_t, \mathbf{f}_{t+H}$
2. Inverse dynamics encoder 输出**两个解耦的 latent**:
   - $\mathbf{z}_{v,a}$: action-relevant, 量化到 **frozen Act-VAE codebook**
   - $\mathbf{z}_{v,i}$: action-irrelevant, 量化到一个独立的 env codebook
3. Forward dynamics decoder 用 $(\mathbf{f}_t, \mathbf{z}_{q,a}, \mathbf{z}_{q,i})$ 重建 $\hat{\mathbf{f}}_{t+H}$

**关键 insight**: 把 $\mathbf{z}_{v,a}$ 量化到 Act-VAE 的 codebook 上, 强制视频推断出的 latent action 必须落在机器人可执行的离散空间里. 这是"物理 grounding"的实现方式.

### Contrastive alignment with SigLIP

公式 (3):
$$\mathcal{L}_{\text{contrastive}} = -\log\sigma\left(\frac{s_p - b}{\tau}\right) - \sum_{j=1}^M \log\left(1 - \sigma\left(\frac{s_{n,j} - b}{\tau}\right)\right)$$

变量:
- $s_p = \cos(\mathbf{z}_{v,a}, \mathbf{z}_a)$: positive pair 的 cosine similarity, $\mathbf{z}_a$ 是同一 sample 的 robot action latent (robot data) 或 self-anchor (human video)
- $s_{n,j}$: 与 batch 内其他 negative sample 的相似度
- $\tau$: temperature
- $b$: learnable bias

**Intuition**: 用 SigLIP (arXiv:2303.15343) 而不是 InfoNCE, 因为它把每个 pair 当成独立 binary classification, 不需要全 batch softmax, 内存友好, 适合大规模 batch. Disco-CLIP (CVPR 2023) 用来做分布式 contrastive 实现.

对于 human video (没有 ground-truth action), $\mathbf{z}_{v,a}$ 自己做 positive anchor, batch 内其他 sample 做 negative. 学习信号完全来自 contrastive push — 强制不同 video transition 的 latent 必须区分开, 且量化后落到 codebook 不同 cell.

### L1 regularization 解耦

$$\mathcal{L}_{\text{reg}} = \|\mathbf{z}_{v,i}\|_1$$

这个 L1 penalty 的作用是 **sparse coding**: 强迫 action-irrelevant latent 只在必要时承载 nuisance 信息, 把 action-relevant 信号保留在 $\mathbf{z}_{v,a}$ 里. 没有这一项, decoder 会偷懒把所有重建信号都塞到 $\mathbf{z}_{v,i}$, 让 $\mathbf{z}_{v,a}$ 退化.

总 loss 公式 (4):
$$\mathcal{L}_{\text{VD}} = \mathcal{L}_{\text{rec}}(\hat{\mathbf{f}}_{t+H}) + \lambda_{\text{vq}}\mathcal{L}_{\text{VQ}} + \lambda_{\text{con}}\mathcal{L}_{\text{contrastive}} + \lambda_{\text{reg}}\|\mathbf{z}_{v,i}\|_1$$

权重 $\lambda_{\text{con}}=0.1$, $\lambda_{\text{reg}}=0.5$ (Table VIII).

---

## 3. Dual-formulation VLA: 离散 reasoning + 连续 control

### CLAP-NTP: Autoregressive Next-Token-Prediction

用 Qwen3VL-4B (arXiv:2511.21631) 作为 backbone, 把 action codebook 当成新增 token 加入词表. 输出序列 $Y = [\mathbf{y}_{\text{sub}}, \mathbf{z}_a]$, 包含 subtask 描述和离散 action tokens.

训练 loss 公式 (5):
$$\mathcal{L}_{\text{AR}} = -\sum_{t=1}^L \log P_\theta(y_t | y_{<t}, \mathcal{I}_t, \mathcal{T})$$

**Intuition**: 这个 formulation 直接复用 VLM 的 next-token prediction 训练范式, 保留了 VLM 的 instruction following 和 reasoning 能力. 对 robot data 用 ground-truth $\mathbf{z}_a$, 对 human video 用 VD-VAE 推断的 pseudo-label $\mathbf{z}_{q,a}$.

### CLAP-RF: Rectified Flow 连续控制

Autoregressive 的速度太慢 (788ms latency, Table VII), 不能做高频控制. CLAP-RF 用 Diffusion Transformer (DiT) 作为 action expert, 通过 Rectified Flow (ICLR 2022) 训练.

Rectified Flow 的核心 idea: 把 noise $\epsilon \sim \mathcal{N}(0, I)$ 和 clean action $\mathbf{a}_{1:H}$ 之间用**直线**插值:
$$\mathbf{a}_{1:H}^\tau = \tau \mathbf{a}_{1:H} + (1-\tau)\epsilon, \quad \tau \in [0, 1]$$

模型预测 vector field $\mathbf{v} = \mathbf{a}_{1:H} - \epsilon$, loss 公式 (7):
$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{\mathcal{D}, \tau, \epsilon}\left[\|(\mathbf{a}_{1:H} - \epsilon) - f^a(\mathbf{a}_{1:H}^\tau, \tau, \text{context})\|^2\right]$$

变量:
- $\mathbf{a}_{1:H}$: ground-truth action chunk (长度 H)
- $\tau$: flow time step, 训练时从 Beta 分布 $p(\tau) = \text{Beta}((s-\tau)/s; 1.5, 1.0)$ 采样, 这是从 $\pi_0$ (arXiv:2410.24164) 借来的, 让模型在低 noise 区域更精细
- $f^a$: DiT 网络
- context: VLM backbone 通过 cross-attention 提供的语义特征

### Stop-gradient cross-attention

公式 (6):
$$\text{Attn}(Q_{\text{DiT}}, K_b, V_b) = \text{softmax}\left(\frac{Q_{\text{DiT}} \cdot \text{sg}(K_b)^\top}{\sqrt{d_k}}\right)\text{sg}(V_b)$$

**Intuition**: 这是一个**单向信息桥**. DiT (action expert) 可以读取 VLM 的 K/V cache 获取语义信息, 但梯度不会回传到 VLM. 这避免了 action generation 的高方差梯度污染 VLM 的预训练权重 — 这个 trick 来自 Knowledge Insulating VLA (arXiv:2505.23705).

### Multi-scale feature aggregation

DiT 只有 16 层, 但 VLM 有 36 层. 论文实验发现: 用早期层 (layer 1-12) + 中间层 (14, 16, 18, 20, 22, 24) 的组合最好 (Table V). 单用 low-level feats: 86.5%, 加 high-level feats: 89.3%.

**Intuition**: 浅层特征保留空间细节 (对 precise manipulation 重要), 深层特征有语义抽象 (对 object recognition, task understanding 重要). 两者融合, 但又不全部用, 控制了 DiT 的规模.

---

## 4. Knowledge Matching: 防止 catastrophic forgetting

公式 (8):
$$\mathcal{L}_{\text{KM}} = \alpha D_{\text{KL}}\left(P(\cdot|\text{ctx}; \phi_{\text{ref}}) \|\| P(\cdot|\text{ctx}; \phi_{\text{policy}})\right) + \mathcal{L}_{\text{RF}}$$

变量:
- $\phi_{\text{ref}}$: frozen reference model (pre-trained 版本)
- $\phi_{\text{policy}}$: active policy (正在 fine-tune 的版本)
- $\alpha$: KL 权重
- $D_{\text{KL}}$: token 分布的 KL 散度

**Intuition**: 这是 RLHF 里 PPO 的 KL penalty 的对应物. 在 LIBERO 实验里, pre-training 是 dual-arm ego-centric real-world, fine-tuning 是 single-arm third-person simulation — 域差距巨大. Table V 显示: 
- 只用 KI (stop-gradient only, 不 fine-tune VLM): 56.8% — VLM 没适应新域
- 全 fine-tune VLM: 82.0% — VLM 适应了但丢失了 pre-trained 知识, Long 任务掉到 64%
- KM: 91.0% — KL anchor 让 VLM 既能适应新域, 又不丢失 reasoning 能力

---

## 5. 实验数据解读

### Real-world (Table I)

| Method | P&P | PnP OOD | Pack Doll | Fold | Bouquets | Mean |
|---|---|---|---|---|---|---|
| $\pi_0$ | 85/75 | 65/60 | 80/60 | 40/40 | 30 | 54.0 |
| $\pi_{0.5}$ | 90/80 | 80/75 | 80/60 | 50/30 | 40 | 60.0 |
| UniVLA | 75/60 | 65/50 | 70/30 | 10/30 | 20 | 35.0 |
| CLAP-NTP | 90/85 | 85/80 | 80/60 | 20/30 | 40 | 56.0 |
| **CLAP-RF** | **95/85** | 80/70 | **90/70** | 40/40 | 40 | **61.0** |

关键观察:
1. CLAP-NTP 在 OOD Pick & Place 上表现最好 (85/80), 远超 $\pi_0.5$ (80/75), 说明离散 token + aligned latent 的 object generalization 很强
2. CLAP-RF 在 precision 任务 (Pack Doll Close: 70% vs $\pi_0$ 60%, Fold: 40% vs $\pi_0$ 40%) 上更强, 验证了 Rectified Flow 的连续控制优势
3. UniVLA 全线崩溃, 说明 reconstruction-based LAM 没有 alignment 根本不能 ground 到 physical action

### Human video 带来的 generalization (Fig. 8)

Make Bouquets 任务上, 仅用 teleop 数据训练时, 所有方法在 unseen flower 组合上 ≤10%. 加入 human video pseudo-label 后:
- CLAP-NTP: 35% (matched seen performance)
- UniVLA: 10% (基本没改善)
- $\pi_{0.5}$: 0% (没有 human video 机制)

**这是 paper 最 impactful 的结果**: 直接证明 contrastive alignment 让 unlabeled human video 的 manipulability prior 真正 transfer 到了 robot policy.

### LIBERO (Table III)

Generalist models:
| Method | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| $\pi_0$ (PaliGemma) | 87 | 63 | 89 | 48 | 71.8 |
| $\pi_0$ | 90 | 86 | 95 | 73 | 86.0 |
| SmolVLA | 93 | 94 | 91 | 77 | 88.8 |
| **CLAP-RF** | **97** | 92 | 93 | **82** | **91.0** |

LIBERO-Long (long-horizon multi-stage) 上 CLAP-RF 达到 82%, 大幅领先. 这印证了 hierarchical design: VLM 提供 high-level planning, RF expert 提供 low-level control, 两者协同让长程任务不掉链子.

### Inference speed (Table VII)

- CLAP-RF: 183ms (comparable to $\pi_0$ 的 169ms)
- CLAP-NTP: 788ms (autoregressive 太慢)

这个 latency 对应 ~5Hz 控制频率, 够用但不极致. Paper 没说 chunk size 是多少, 但 action chunk 的设计意味着一次推理输出 H 步, 等效频率可以更高.

---

## 6. Ablation 关键发现

### Contrastive alignment 的影响 (Table VI)

去掉 contrastive loss:
- ID 性能基本不变 (Pick&Place 85%, Bouquets 35%)
- OOD 大幅下降 (Bouquets OOD: 35% → 20%)

去掉 human video data:
- 平均下降 11.3%
- Bouquets OOD: 35% → 5%

**Intuition**: contrastive loss 对 in-distribution 影响不大 (robot data 自己就能学好), 但对 OOD generalization 至关重要. human video 是语义泛化的来源.

### Act-VAE codebook 大小 (Table IV)

$N_q=4, K=256$: PSNR=15.93 (太压缩, 信息丢失)
$N_q=35, K=256$: PSNR=43.01 (太冗余, VLM 难学)
$N_q=16, K=256$: PSNR=40.00 (elbow, 最优)

---

## 7. Limitations 和未来方向

Paper 自己承认:
1. **Task-level generalization 不足**: 只能 generalize 到 seen task 的 new objects, 不能从 human video 学全新 task
2. **Hand-gripper morphological gap**: 人类灵巧手动作 vs 平行夹爪, latent space 的 alignment 有 inherent ambiguity
3. **Multi-stage pipeline**: Act-VAE, VD-VAE, NTP, RF 分开训练, 工程 complex

---

## 8. 我的 critical thoughts

**优点**:
- Contrastive alignment 的 idea elegant 且 effective: 把 visual latent "anchor" 到物理 codebook 上, 比单纯重建好得多
- Dual-formulation 抓住了 VLA 的核心矛盾: reasoning 需要慢的 autoregressive, control 需要快的 parallel generation. 用 distillation 思路结合两者很合理
- Knowledge Matching 对 cross-domain transfer (real → sim, dual-arm → single-arm) 有效

**疑问**:
- SigLIP 的 self-anchor 对 human video 的 contrastive learning 是否足够? 没 ground-truth action 时, positive pair 是 trivial 的, 学习信号全靠 negative push. 这会不会让 codebook collapse 到少数几个高频 cell?
- Codebook size 256 是否太小? 双臂 14DoF × H 步的 action chunk, 离散化到 256 个 token, 每条轨迹 16 个 token, 信息容量 $16 \log_2 256 = 128$ bits. 这个量级能否表达精细 manipulation?
- $\lambda_{\text{con}}=0.1$ 是不是太小? contrastive loss 的梯度可能被 reconstruction loss 主导, alignment 不够强

参考资源:
- Project page: https://lin-shan.com/CLAP/
- VQ-VAE: https://arxiv.org/abs/1711.00937
- DINOv3: https://arxiv.org/abs/2508.10104
- SigLIP: https://arxiv.org/abs/2303.15343
- Rectified Flow: https://arxiv.org/abs/2209.03003
- $\pi_0$: https://arxiv.org/abs/2410.24164
- LAPA: https://arxiv.org/abs/2410.11758
- UniVLA: RSS 2025
- AgiBot World: https://arxiv.org/abs/2503.06669
- Ego4D: https://arxiv.org/abs/2110.07058
- LIBERO: https://arxiv.org/abs/2306.03310
- Knowledge Insulating VLA: https://arxiv.org/abs/2505.23705
- Qwen2-VL: https://arxiv.org/abs/2407.10671

整体看, CLAP 在 LAM 这个方向上做出了一个关键的 conceptual contribution: **latent action space 不应该由 visual reconstruction 单独定义, 必须有 physical grounding 的 anchor**. 这个 insight 可以推广到更多 cross-embodiment, cross-modal learning 的场景, 例如 humanoid 全身控制、dexterous hand manipulation, 只要找到合适的 "physical codebook" 作为 anchor.
