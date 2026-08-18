---
source_pdf: Dual-Actor Fine-Tuning of VLA Models A Talk-and-Tweak.pdf
paper_sha256: 1d843c11971be343ac7c29527b9465889b8108dc0bbdf2a518a2e17e4ad43c35
processed_at: '2026-08-18T07:10:12-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍

Karpathy，我换个讲法，少堆公式，多讲故事，但该有的技术细节都保留。

---

## 一、这篇 paper 到底在解决什么痛点

你 fine-tune 过一个 VLA model 就知道，把 Octo 或者 OpenVLA 这种预训练模型搬到真机上，问题特别多：

1. **数据不够**：你不可能在真机上采几万条 trajectory，成本太高。
2. **Multi-task 互相打架**：你想让一个 policy 同时学会三个动作（把 bolt 立起来、抓起来、插进去），训着训着发现一个 task 把另一个 task 的性能带崩了——这是 RL fine-tune 里典型的 catastrophic interference。
3. **人的介入没法被 structured 利用**：你拿 SpaceMouse 介入纠错，这段 trajectory 存下来当 demonstration 用？那不是纯 BC 嘛，BC 上限很低。你不存？那人介入的信号就浪费了。
4. **Latent space RL 的 critic 不准**：之前 DSRL 那篇也在 latent space 做 RL，但 critic 只用 offline 数据训，OOD action 的 value 估得离谱，学不动。

这篇 paper 就是冲着这四个痛点来的。

---

## 二、他们最关键的一个 insight

Diffusion policy 大家都熟：输入一个高斯噪声 $w$，网络一步步 denoise 出 action。Consistency policy 把多步蒸馏成一步，推理快，适合 10Hz 控制。

这里有个被忽视的事实——**diffusion policy 的 output 由两样东西决定**：

1. **网络权重 $\theta$**：学到的是"技能本体"
2. **输入噪声 $w$**：决定"本次具体走哪个 mode"

之前 DSRL 已经嗅到这个味了，也在 $w$ 上做 RL。但这篇 paper 的 insight 更深：

> **我干脆不动 $\theta$，只在 $w$ 的 mean 上加一个 language-conditioned 的偏移。**

这是整个 paper 最漂亮的一点。为什么这样好？

你想，$\theta$ 学的是 multi-task 共享 representation。你直接拿 RL 去更新 $\theta$，Q 梯度会往一个 task 偏，其他 task 的性能立刻崩。这在 multi-task RL 里太常见了。

但 $w$ 的 mean shift 是一个 **low-rank perturbation**——它只影响 action 的一个小子空间，不碰 policy backbone 的共享表征。相当于你在 frozen 的 VLA 上，只允许一个"小旋钮"被 language 调。

这个 philosophy 你应该很有共鸣——ControlNet、LoRA 全是这个思路：**pretrain 大模型不动，在最小子空间做 adaptation**。

---

## 三、Dual-Actor 架构具体长什么样

两个 actor，一个 critic（其实是 per-task 多个 critic）。

### Primary Actor $\pi_\theta^w$

就是原版 consistency policy。输入 state $s$ 和 task embedding $h$，但生成 action 之前要 sample 一个 noise $w$。

**正常模式下**：

$$w \sim \mathcal{N}(0, K^2 I)$$

符号讲清楚：
- $w$ 是高斯噪声，维度跟 action chunk 一致
- $K$ 是控制 noise scale 的超参，paper 没给具体值
- $I$ 是单位协方差矩阵

意思是 $w$ 从零均值高斯采样，policy 完全按预训练行为走。

### Refinement Actor $\pi_\phi$

一个小网络：
- ResNet 吃 RGB
- T5 吃 language command（比如 "move right"）
- 两个 embedding 投影到同维度，concat 起来
- MLP 输出一个向量 $\mu$

**refinement 模式下**：

$$w \sim \mathcal{N}(\pi_\phi(\mu | s, l_{rf}), K^2 I)$$

- $l_{rf}$ 是 refinement language command
- $\pi_\phi(\mu | s, l_{rf})$ 是 refinement actor 输出的 mean
- 协方差还是 $K^2 I$，**只动 mean 不动 variance**

注意：mean 是被 $\pi_\phi$ 预测出来的，**conditioned on language**。没 language（输入 "[null]"）时，要回归到 0，这就是 regularization loss 干的事。

### Primary Actor 怎么训练

混合 BC + RL：

$$\mathcal{L}_{\pi_\theta} = \lambda_1 \mathcal{L}^{BC} + \lambda_2 \mathcal{L}^{Q}$$

- $\mathcal{L}^{BC}$：监督学习，模仿 expert demo
- $\mathcal{L}^{Q}$：让 Q 值最大化，RL 探索
- **warm-up 阶段**：$(\lambda_1, \lambda_2) = (1.0, 0.1)$，主要学 demo
- **online 阶段**：$(\lambda_1, \lambda_2) = (0.5, 0.5)$，开始信任 Q

Q 这边用 Cal-QL [Nakamoto et al. 2023] 而不是普通 TD。Cal-QL 把 OOD action 的 Q 值拉到 BC 附近，避免 offline RL 里 Q-value overestimation 的经典坑。

### Refinement Actor 怎么训练

三个 loss 加一起：

$$\mathcal{L}_{\pi_\phi} = \eta_1 \mathcal{L}^{BC} + \eta_2 \mathcal{L}^{Q} + \eta_3 \mathcal{L}^{Reg}$$

具体讲讲每一项：

**(1) BC loss**：

$$\mathcal{L}^{BC} = \mathbb{E}_{(s, a^*) \sim \mathcal{D}_{intv}}\left[\|\pi_\theta^w(h) - a^*\|^2\right]$$

- $a^*$ 是人介入时的 action
- $\pi_\theta^w(h)$ 是 policy 实际输出
- 注意 $h$ 依赖 $w$，$w$ 又依赖 $\pi_\phi$，所以梯度能从 BC loss 一路反传到 $\pi_\phi$ 的 mean 输出。这是 end-to-end 的设计。

**(2) Q-maximization**：

$$\mathcal{L}^{Q} = -\mathbb{E}_{w \sim \mathcal{N}(\pi_\phi(\mu|s, l_{rf}), K^2 I)}[Q_\psi(s, \pi_\theta^w(h))]$$

跟 primary actor 的 Q loss 形式一样，但这里 $w$ 是被 $\pi_\phi$ 调过的。意思是：让 refinement 后的 action 在 Q 值上更高。

**(3) Regularization**：

$$\mathcal{L}^{Reg} = \mathbb{E}_{s \sim \mathcal{D}_{intv}}\left[\|\pi_\theta^{w \sim \mathcal{N}(\pi_\phi(\mu|s, l_{rf}), K^2 I)}(h) - \pi_\theta^{w \sim \mathcal{N}(0, K^2 I)}(h)\|^2\right]$$

- 第一项：refinement 模式下 policy 输出
- 第二项：primary 模式下 policy 输出（应该是 stop-gradient）
- 强制 $\pi_\phi$ 在输入 "[null]" 时预测 mean ≈ 0

这个设计简直就是 ControlNet 里的 zero convolution。ControlNet 用零初始化保证 trainable branch 初始不影响 output；这里用 regularization loss 保证 refinement actor 在 null command 时不偏移。

$(\eta_1, \eta_2, \eta_3) = (1.0, 0.1, 0.1)$，主要靠 BC 学，RL 只是微调。

---

## 四、Talk-and-Tweak：自动 label 的关键 trick

这个 trick 是 paper 最 Karpathy-style 的工程思路。

人在 SpaceMouse 上介入的时候，action 是 7 维：$(\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw, gripper)$。

你介入了，存下来 $(s, a^{intv})$ pair。这个 pair 没有 language label，没法用来训 language-conditioned 的 refinement actor。

他们的解法：写个 rule-based mapper，**把物理位移自动翻译成 language command**。

具体公式：

$$\Delta_t = \sum_{j=0}^{J-1} a_{t+j}^{intv}[0:3]$$

- $J=5$：5 步时间窗口
- $a[0:3]$：只看 translational 部分
- $\Delta_t$：累积平移位移

然后阈值判断（per axis $d \in \{x,y,z\}$）：

$$l_{rf_t}^d = \begin{cases} 
\text{positive } d \text{ direction} & \Delta_t^d > \sigma \\
\text{negative } d \text{ direction} & \Delta_t^d < -\sigma \\
\text{no command} & |\Delta_t^d| \leq \sigma
\end{cases}$$

- $\sigma = 0.001$ m = 1 mm
- 三个 axis 拼起来，比如 "move right and forward"

最终三元组 $(s, a^{intv}, l_{rf})$ 进 dataset。

**为什么 1 mm**：bolt 任务要毫米级精度，1 mm 是 noise floor，能滤抖动。

**为什么只看 translation 不看 rotation**：rotation 的语言化太难——"rotate clockwise 15 degrees" 这种自然语言写起来就别扭。Paper 简化了，留了 extension 空间。

**这个 trick 真正的价值**：你介入的时候根本不用打字，手一动 label 自动来。数据效率极高。

它让我想到几件事：

1. **Yell at Your Robot (YAR) [Shi et al. 2024]**：让人直接说语言 correction；这篇反着来，让人手动介入，自动转语言。两边都没对方的优势——YAR 语言丰富但人要说话，这篇人省事但语言简单。两边可以融合。

   Reference: https://y-robot.github.io/

2. **VLM auto-captioning**：现在 rule-based mapper 只能生成 6 个 axis 命令。让 GPT-4V 看 intervention 前后两帧，生成 "move right to align with the slot" 这种 contextual command，refinement actor 能学到的语义就丰富多了。

3. **Visual programming / ViperGPT 那套**：用 LLM 写代码调用 visual API，类似地这里可以用 VLM 写代码调用 SpaceMouse API。一个 meta-level 的扩展。

---

## 五、Multi-task 怎么平衡的

这是 paper 另一个 contribution。Multi-task RL 直接套 SAC 之类的，肯定崩——一个 task 学得快 Q 值大，梯度就大，其他 task 被 starve。

他们的设计：
- **Shared actor**（一个 policy 学所有 task，参数效率高 + transfer）
- **Per-task critic** $Q_{\psi_i}$（每个 task 独立 value 估计，互不干扰）
- **Adaptive task weighting $\epsilon_i$**：Q 值大的 task 降权，Q 值小的 task 加权

加权公式：

$$\epsilon_i = \frac{\sum_{i=1}^{N}\overline{Q}_i}{N\overline{Q}_i + Nc}$$

- $\overline{Q}_i$：task $i$ 当前平均 Q 值
- $c = 0.1$：数值稳定常数，防止 $\overline{Q}_i$ 太小分母爆炸
- $\epsilon_{max} = 1.2$, $\epsilon_{min} = 0.8$：clip 范围

直觉：Q 大 = 学得好 = 减权；Q 小 = 学得差 = 加权。本质就是 self-paced learning / hard example mining。

最终 Q loss：

$$-\frac{1}{N}\sum_{i=1}^{N}\epsilon_i \cdot \mathbb{E}_{s \sim \mathcal{D}_i}[Q_{\psi_i}(s, \pi_\theta^w(h))]$$

每个 task 用自己的 critic 算自己的 Q，乘自己的 weight，平均起来。

Reference:
- Self-Paced Learning: https://arxiv.org/abs/2010.05217
- Class-balanced loss: https://arxiv.org/abs/1901.06683

---

## 六、Algorithm 1 流程

伪代码讲讲关键步骤：

**Stage I: Offline Warm-up**
```
for each offline step:
    for each task i:
        sample B/N demos from D_demos^i
        update Q_ψi with Cal-QL
    update π_θ^w (主要 BC，轻微 Q)
```

只用 demo 数据，policy + critic 都初始化好。

**Stage II: Online Interaction**

两个线程并行：
```
Learning Thread:
    for each online step:
        for each task i:
            sample half from D_demos^i, half from D_rollouts^i
            update Q_ψi with standard TD
        update π_θ^w (BC+Q, Q weight 加大)
        update π_φ with D_talk-tweak

Interaction Thread:
    for each task i:
        if no human intervention:
            a ~ π_θ^w(s, l_task)
            store (s, a, r, s') in D_rollouts^i
        else:
            take a^intv
            store (s, a^intv, r, s') in D_intv^i
    
    augment D_talk-tweak with D_intv^i
    augment D_demos^i with D_intv^i  ← 重点
```

**关键点（line 29）**：人介入数据同时进 demos buffer 和 talk-tweak buffer。意味着 primary actor 也学人介入数据（当 demo 用），refinement actor 学 language-conditioned 版本。一份 intervention data 复用两次。

---

## 七、实验结果，几个关键数

### 7.1 主实验（Table I）

| Method | Place | Pick | Assemble | Avg |
|---|---|---|---|---|
| HG-Dagger | 28 | 28 | 20 | 25.3 |
| HIL-ConRFT | 0 | 0 | 0 | 0 |
| DSRL | 28 | 28 | 16 | 24.0 |
| Ours (no dual) | 88 | 96 | 76 | 86.7 |
| **Ours (dual)** | **100** | **100** | **100** | **100** |

几个 takeaway：

1. **HIL-ConRFT 在 multi-task 直接 0%**：这是 paper 的核心 motivation。原版 ConRFT 是 single-task，flat optimization 一上来 multi-task 就崩。Q 值互相串味儿，policy 学不到东西。
2. **HG-Dagger 25%**：BC 天花板。人介入数据当 demo 用，再多也就这样。
3. **DSRL 24%**：latent RL 思路对，但 critic 全 offline 训，OOD action value 估不准，学不动。
4. **去掉 dual-actor 还有 86.7%**：multi-task balancing 和 per-task critic 已经很强了。Refinement actor 把 86.7% 推到 100%。
5. **Episode length 几乎一样**（30.7 vs 30.7）：dual-actor 提 success 不提 speed，符合设计——refinement 只在 error 时介入，正常路径不变。

### 7.2 训练曲线（Fig. 5）

- Dual-actor：20 分钟 → 60%，101 分钟 → 100%
- Single-actor：20 分钟 → 30%，128 分钟 → 86.7%

省 27 分钟（21% 提速）。更关键的是稳定性：single-actor 在第三个 task 上 fluctuating 厉害，dual-actor 一直稳。Latent space fine-tuning 本质上参数空间小，更新平滑。

### 7.3 VLA Backbone 泛化（Table II）

| Task | Octo (0.27B) | SmolVLA (0.6B) |
|---|---|---|
| All | 100 | 100 |

两个 backbone 都 100%。但 SmolVLA 没 CLS token，他们做了 prefix-mask pooling 从 attention KV cache 提取 task embedding，是一个非 trivial 的工程实现：

$$K_{pool} = \frac{\sum_{t=1}^{T} K \cdot m}{\sum_{t=1}^{T} m}, \quad V_{pool} = \frac{\sum_{t=1}^{T} V \cdot m}{\sum_{t=1}^{T} m}$$

- $m \in \{0,1\}^{B \times T}$：prefix mask，标记 task-critical token
- 最终 $h = E_\phi(s) \in \mathbb{R}^{B \times 2HD}$

意思是：找出 task instruction 对应的 token，把它们的 K 和 V 平均 pooling 出来当 task embedding。一个挺 hacky 但实用的"无 CLS token 怎么提 task embedding"的方法。

### 7.4 Long-Horizon（Fig. 6）

| # bolts | # steps | Success |
|---|---|---|
| 1 | 3 | 90% (9/10) |
| 2 | 6 | 60% (6/10) |
| 3 | 9 | 60% (6/10) |
| 4 | 12 | 50% (5/10) |

12 步连续操作保持 50%，相当于每步错误率约 5.7%（$(0.5)^{1/12}$），其实挺不错。衰减主要来自 occluded slot view，error 累积。

### 7.5 Multi-Robot（Fig. 7）

- Train A → Test A: 100%
- Train A → Test B: **0%**（硬件差异 + background）
- Train A+B → Test A: ~100%
- Train A+B → Test B: ~100%
- 47 分钟 vs 单 robot 100 分钟 → **2× 提速**

Centralized learner + decentralized actors，类似 A3C / IMPALA 架构。这说明这套 framework 可以 scale，不是只能跑一个 robot 的小把戏。

Reference:
- A3C: https://arxiv.org/abs/1602.01783
- IMPALA: https://arxiv.org/abs/1802.01561

---

## 八、几个有意思的联想

### 8.1 跟 RLHF 的对应

把 dual-actor 跟 RLHF [Ouyang et al. 2022] 摆一起看：

| RLHF | Dual-Actor |
|---|---|
| SFT model | Pretrained VLA |
| Reward model | Q-function |
| PPO update policy weight | Latent noise mean shift |
| KL constraint to SFT | $\mathcal{L}^{Reg}$ |
| Prompt → response | State → action |

特别 $\mathcal{L}^{Reg}$ 对应 RLHF 里 KL penalty。两个东西哲学完全一样：fine-tune 一个 frozen pretrain model，用一个 constraint 保证不偏太远。

Reference: InstructGPT: https://arxiv.org/abs/2203.02155

### 8.2 跟 Residual RL 的对应

ResiP [Ankile et al. 2024] 是 residual RL：在 frozen BC policy 上加一个 residual RL policy。Dual-actor 在 frozen diffusion policy 上加一个 residual **on the noise mean**。都是 residual learning。

Reference: https://arxiv.org/abs/2407.16677

### 8.3 跟 ControlNet 的对应（这个我觉得最深刻）

ControlNet 设计：
- Frozen pretrained UNet encoder
- 复制一份 trainable copy
- 用 zero convolution 连接，初始时 trainable branch 不影响 output
- 用 conditioning 训练 trainable branch

Dual-Actor 完全一样：
- Frozen primary actor
- Trainable refinement actor
- Regularization loss 保证初始时输出 ≈ 0（等价于 zero conv）
- 用 language command conditioning 训练

可以理解为：**Dual-Actor = ControlNet for Robot Action Diffusion**。

Reference: ControlNet: https://arxiv.org/abs/2302.05543

### 8.4 跟 LoRA 的对应

更激进类比：refinement actor 像 LoRA。LoRA 在 frozen weight 上加 low-rank update；refinement actor 在 frozen policy 的 noise 上加 mean offset。都是"低参数量 perturbation on top of frozen base"。

Reference: LoRA: https://arxiv.org/abs/2106.09685

### 8.5 Pretraining-Finetuning 范式对应

整个 paper 范式映射到 LLM 训练 pipeline：

| LLM | VLA (本 paper) |
|---|---|
| Pretraining | VLA pretraining (BC on demos) |
| SFT | Warm-up fine-tuning (Cal-QL + BC) |
| RLHF | Online RL fine-tuning |
| Constitutional AI | Human-in-the-loop talk-and-tweak |

Robotics 正在走的路径，paper 印证了 LLM 训练范式的可迁移性。

### 8.6 关于 $K$ 的隐含问题

Paper 没给 $K$ 的值，也没讨论 $K$ 的影响。但直觉上：
- $K$ 大 → action 分布 spread 大 → 探索强但精度差
- $K$ 小 → action 集中 → 精度高但探索弱
- Refinement actor 只调 mean 不调 variance，**这是简化假设**——可能 mean shift 就够用了

如果允许 refinement actor 也调 covariance，会更接近 conditional normalizing flow 或 conditional VAE。

---

## 九、Limitation 我补充几点

Paper 自己列了：
1. Long-horizon error accumulation
2. Dependence on human intervention quality

我额外想到：

3. **Language vocabulary 太小**：只有 6 个 axis 命令。复杂 motion（"rotate gently to align with the slot"）无法表达。可以接 VLM 做 captioning。

4. **Refinement actor 只在 intervention 数据上训**：如果某 state 从未被 human 介入过，refinement actor 没见过，泛化可能差。Paper 没讨论这个 OOD 问题。

5. **Single-step consistency policy 的精度限制**：consistency policy 把 multi-step denoising 蒸馏成 single-step，毫米级 precision 任务上精度可能损失。这是为什么需要 refinement actor 补偿。

6. **Multi-robot communication cost**：Fig. 7 显示 2× 提速，但没讨论 bandwidth / latency。4 robots、8 robots 呢？

7. **只测了 bolt manipulation 一种任务 family**：虽然 multi-task，但本质都是 bolt 操作。是不是能扩展到 pick-and-place + pouring + opening drawer 这种异构 task？未知。

---

## 十、一句话的 take-away

这篇 paper 的核心贡献：

**在 frozen 的预训练 VLA 上做 fine-tuning，更新点越少越好、越窄越好。**

Refinement actor 只动 mean shift，参数量小、作用域窄，恰好是 "minimal intervention" 原则。这与 ControlNet、LoRA 的设计哲学一脉相承——保留 pretrain 的通用性，只在最小子空间做 adaptation。

这个 philosophy 我想你应该有共鸣——你讲过 "the bitter lesson"，pretrain + small fine-tuning > end-to-end training。这篇 paper 在 robotics 上做了同样的事情。

Reference: The Bitter Lesson: http://incompleteideas.net/IncBitterLesson.html

---

## 十一、未来方向联想

1. **VLM auto-captioning**：让 GPT-4V 看 intervention 前后的 frame，生成 contextual command。Refinement actor 能学更丰富的语义。
2. **Refinement actor 学 covariance**：让 $\pi_\phi$ 输出 $\mu, \Sigma$，policy distribution 更灵活。要小心 covariance collapse。
3. **Hierarchical refinement**：当前 language 是 axis-level。引入 hierarchical command："first move right, then grasp"。
4. **Cross-embodiment generalization**：dual-robot 已经做了，跨 morphology（arm vs quadruped vs humanoid）还没做。Refinement actor 的 language 是 embodiment-agnostic 的话应该能 transfer。
5. **World model integration**：现在 Q-function 是 task-specific。用 world model 做 task-agnostic value estimation，可以省去 per-task critic。
6. **Diffusion policy 多步 vs 单步**：consistency policy 精度有损失。如果回到 multi-step diffusion policy，refinement actor 的设计可能要改。

---

## 十二、相关阅读

**VLA Models:**
- π0: https://www.physicalintelligence.company/blog/pi0
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://openvla.github.io/
- Octo: https://octo-models.github.io/
- SmolVLA: https://arxiv.org/abs/2504.05299

**Diffusion & Consistency:**
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Consistency Models: https://arxiv.org/abs/2303.01469
- Boosting Continuous Control with Consistency Policy: https://arxiv.org/abs/2310.06343

**Human-in-the-Loop:**
- HIL-SERL: https://arxiv.org/abs/2410.21845
- HIL-ConRFT: https://arxiv.org/abs/2502.05450
- RLIF: https://arxiv.org/abs/2311.12996
- HG-DAgger: https://arxiv.org/abs/1810.02890
- Yell at Your Robot: https://y-robot.github.io/

**Offline-to-Online RL:**
- Cal-QL: https://arxiv.org/abs/2303.05479
- CRR: https://arxiv.org/abs/2006.15134
- AWAC: https://arxiv.org/abs/2006.09359

**Policy Adaptation:**
- ResiP: https://arxiv.org/abs/2407.16677
- DSRL: https://steeringdiffusion.github.io/

**类比与 inspiration:**
- ControlNet: https://arxiv.org/abs/2302.05543
- LoRA: https://arxiv.org/abs/2106.09685
- InstructGPT: https://arxiv.org/abs/2203.02155
- The Bitter Lesson: http://incompleteideas.net/IncBitterLesson.html

---

希望这次更"人话"一些。整体上，这篇 paper 我觉得最 elegant 的是 dual-actor 的设计哲学：frozen base + minimal perturbation + language conditioning。Talk-and-tweak 的 auto-labeling 是最实用的工程 trick——一个简单 rule-based mapper 解决 RL 机器人 data scarcity 的核心痛点。这两个 idea 都有进一步扩展的空间，VLM auto-captioning 和 covariance learning 是最自然的下一步。

---

# Dual-Actor Fine-Tuning of VLA Models: Talk-and-Talk-and-Tweak 方法深度解析

Karpathy 老师好。这篇 paper 来自一个挺有意思的思路：把 diffusion-based VLA policy 的 **latent noise** 当作可以被语言 conditioning 的"接口"，从而实现一个 **dual-actor** 架构。下面我尽量把所有细节、直觉、和相关联想都铺开来谈，包括公式变量、架构图、实验数据。

---

## I. Paper 的一句话总结

在真实机器人上 fine-tune 一个预训练的 VLA model（Octo / SmolVLA），通过：
- 一个 **primary actor**（diffusion consistency policy）做"通用动作生成"
- 一个 **refinement actor**（也输出 latent noise 的 mean）做"基于语言指令的微调"
- 一个 **talk-and-tweak** scheme，把人在 SpaceMouse 上的物理修正自动转成 language command，去训练 refinement actor

最终在 bolt manipulation 任务上，101 分钟 online fine-tuning 达到 100% 单任务成功率，long-horizon 12 步连续操作保持 50% 成功率，并扩展到 dual-robot 训练实现 2× 效率提升。

Project page: https://sites.google.com/view/hil-daft/

---

## II. 为什么 latent noise 是个好"接口"：Intuition 的核心

这是整个 paper 最值得品味的设计。先讲直觉，再讲公式。

### 2.1 Diffusion Policy 复习

Diffusion policy [Chi et al. 2023] 把 action 生成建模成一个 **denoising process**：

$$a = \text{Denoise}(w; \theta)$$

其中 $w$ 是初始的高斯噪声。Consistency policy [Song et al. 2023, Chen et al. 2023] 把多步 denoising 蒸馏成 **single-step** 预测，推理速度快，适合 real robot 的 10Hz 控制频率。

Reference:
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Consistency Models: https://arxiv.org/abs/2303.01469
- Boosting Continuous Control with Consistency Policy: https://arxiv.org/abs/2310.06343

### 2.2 关键 Insight

Policy 的 output 由两部分决定：
1. **Network weight $\theta$**：学的 "技能本体"
2. **Input noise $w$**：决定 "本次采样哪个 mode"

之前 DSRL [Wagenmaker et al. 2025] 也是利用这个 idea：在 $w$ 的 latent space 做 RL。但他们用 RL 优化的还是"用一个 critic 去 steer noise"。

Paper 的 dual-actor 设计走得更巧：

> **不修改 primary policy 的参数 $\theta$，而是在 $w$ 的 mean 上加一个 language-conditioned shift。**

也就是 primary actor 负责"做对的事情"，refinement actor 负责"做稍微偏一点但更精确的事情"。

为什么这样比直接 fine-tune $\theta$ 更稳定？因为：
- $\theta$ 学的是 multi-task shared representation，直接 RL 更新会破坏其他 task 的性能（catastrophic interference）
- Latent noise $w$ 的 mean shift 是一个 **low-rank perturbation**，作用域窄，干扰小
- 没有语言指令时，refinement actor 通过 regularization term 回归到 $N(0, K^2 I)$，等价于"不动"

Reference: DSRL: https://steeringdiffusion.github.io/

---

## III. 形式化方法详解

### 3.1 Markov Decision Process 定义

$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \rho, \gamma)$

| 符号 | 含义 |
|---|---|
| $s \in \mathcal{S}$ | state（RGB image + proprioception） |
| $a \in \mathcal{A}$ | action（7-dim end-effector delta pose） |
| $P(s' \| s, a)$ | transition dynamics |
| $r(s, a)$ | reward（sparse: 1 for success, 0 otherwise） |
| $\rho(s)$ | initial state distribution |
| $\gamma \in (0, 1)$ | discount factor |

### 3.2 Action 生成的 Dual Mode

**Primary mode**（无 refinement command）：

$$w \sim \mathcal{N}(0, K^2 I) \tag{1}$$

- $w$：sampled Gaussian noise，进入 consistency policy head
- $K$：超参，控制初始 noise scale（隐含决定了 action 分布的 spread）
- $I$：identity covariance matrix

**Refinement mode**（有 language command $l_{rf}$）：

$$w \sim \mathcal{N}(\pi_\phi(\mu \| s, l_{rf}), K^2 I) \tag{2}$$

- $\pi_\phi$：refinement actor（要训练的）
- $\mu$：predicted mean shift
- $s$：state
- $l_{rf}$：refinement language command（例如 "move right"）

注意 **covariance 还是 $K^2 I$，只是 mean 被 shifted 了**。这个设计很重要：refinement actor 只控制 mean offset，不控制 variance，避免 policy collapse。

### 3.3 Task Embedding（从 VLA backbone 提取）

$$h = \mathcal{V}_{\tau_{pre}}(s, l_{task}) \tag{3}$$

- $\mathcal{V}_{\tau_{pre}}$：pretrained VLA model（Octo）
- $s$：state (RGB + proprio)
- $l_{task}$：task command
- $h$：task embedding

**对 Octo**：直接用 CLS token $h_{CLS} \in \mathbb{R}^{B \times D_{CLS}}$。

**对 SmolVLA**（没有 CLS token）：从 attention KV cache 做 prefix-guided pooling：

$$K_{pool} = \frac{\sum_{t=1}^{T} K_{prefix}}{\sum_{t=1}^{T} m}, \quad V_{pool} = \frac{\sum_{t=1}^{T} V_{prefix}}{\sum_{t=1}^{T} m} \tag{17}$$

- $K, V \in \mathbb{R}^{B \times T \times H \times D}$
- $B$：batch size
- $T$：sequence length
- $H$：number of attention heads
- $D$：head dimension
- $m \in \{0, 1\}^{B \times T}$：prefix mask，标记 task-critical tokens
- $K_{prefix} = K \cdot m$, $V_{prefix} = V \cdot m$
- 最终 $h = E_\phi(s) \in \mathbb{R}^{B \times 2HD}$

这是一个挺 hacky 但实用的"无 CLS token 的 VLA 怎么提 task embedding"的方法。

Reference:
- Octo: https://octo-models.github.io/
- SmolVLA: https://arxiv.org/abs/2504.05299
- OpenVLA: https://openvla.github.io/

### 3.4 Primary Actor 的 Loss

混合 BC + Q-maximization：

$$\mathcal{L}_{\pi_\theta^w}^{Q} = -\mathbb{E}_{s \sim \mathcal{D}}[Q_\psi(s, \pi_\theta^w(h))] \tag{4}$$

$$\mathcal{L}_{\pi_\theta^w}^{BC} = \mathbb{E}_{(s, a^*) \sim \mathcal{D}}\left[\|\pi_\theta^w(h) - a^*\|^2\right] \tag{5}$$

$$\mathcal{L}_{\pi_\theta} = \lambda_1 \mathcal{L}_{\pi_\theta}^{BC} + \lambda_2 \mathcal{L}_{\pi_\theta}^{Q} \tag{6}$$

- $\lambda_1, \lambda_2$：trade-off coefficients
- **Warm-up phase**：$(\lambda_1, \lambda_2) = (1.0, 0.1)$，重 BC 轻 RL（怕 early RL 不稳定）
- **Online phase**：$(\lambda_1, \lambda_2) = (0.5, 0.5)$，开始信任 Q 值

注意这种 schedule 想法类似 **CRR (Critic Regularized Regression)** 和 **AWAC**，可以参考。

Reference:
- CRR: https://arxiv.org/abs/2006.15134
- AWAC: https://arxiv.org/abs/2006.09359

### 3.5 Warm-up Phase: Cal-QL

Q-function 学习用 **Calibrated Q-Learning (Cal-QL)** [Nakamoto et al. 2023]：

$$\mathcal{L}(Q_\psi) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\left[\left(Q_\psi(s, a) - \mathcal{T}_{cal}\overline{Q}(s', a')\right)^2\right]$$

其中 $\mathcal{T}_{cal}$ 是 calibrated Bellman operator，把 OOD action 的 Q-value 拉到 BC 范围附近，避免 offline RL 里典型的 Q-value overestimation。这是 HIL-ConRFT [Chen et al. 2025] 的核心 trick，被本 paper 继承下来。

Reference:
- Cal-QL: https://arxiv.org/abs/2303.05479
- HIL-ConRFT: https://arxiv.org/abs/2502.05450

### 3.6 Refinement Actor 的 Loss（核心创新）

Refinement actor $\pi_\phi$ 是一个独立的小网络：
- ResNet encode RGB
- T5 encode language command $l_{rf}$
- 各自 MLP project 到同维度
- Concatenate → Final MLP → 输出 $\mu$

损失包含三项：

**(1) BC loss**：模仿 human intervention action

$$\mathcal{L}_{\pi_\phi}^{BC} = \mathbb{E}_{(s, a^*) \sim \mathcal{D}_{intv}}\left[\|\pi_\theta^w(h) - a^*\|^2\right] \tag{8}$$

注意这里 ground truth 是 human intervention action $a^*$（来自 $\mathcal{D}_{intv}$），但预测用的是 $\pi_\theta^w(h)$，而 $h$ 又依赖 $w$，$w$ 又依赖 $\pi_\phi$。所以这是一个 **end-to-end 通过 latent noise 反传梯度**的设计。

**(2) Q-maximization**：让生成的 action 在 Q 值上更好

$$\mathcal{L}_{\pi_\phi}^{Q} = -\mathbb{E}_{w \sim \mathcal{N}(\pi_\phi(\mu \| s, l_{rf}), K^2 I)}\left[Q_\psi(s, \pi_\theta^w(h))\right] \tag{9}$$

这是一个"latent space RL"loss，用 Q 值引导 mean shift 方向。

**(3) Regularization**：当 command 是 "[null]" 时，要退化为 primary mode

$$\mathcal{L}_{\pi_\phi}^{Reg} = \mathbb{E}_{s \sim \mathcal{D}_{intv}}\left[\|\pi_\theta^{w \sim \mathcal{N}(\pi_\phi(\mu \| s, l_{rf}), K^2 I)}(h) - \pi_\theta^{w \sim \mathcal{N}(0, K^2 I)}(h)\|^2\right] \tag{10}$$

- 第一项：refinement mode 下 policy output
- 第二项：primary mode 下 policy output（**no gradient through this term**，应该是 stop-gradient）
- 这个 loss 强制 refinement actor 在 "$l_{rf} = [null]$" 时预测 mean ≈ 0

合并：

$$\mathcal{L}_{\pi_\phi} = \eta_1 \mathcal{L}^{BC}_{\pi_\phi} + \eta_2 \mathcal{L}^{Q}_{\pi_\phi} + \eta_3 \mathcal{L}^{Reg}_{\pi_\phi} \tag{11}$$

$(\eta_1, \eta_2, \eta_3) = (1.0, 0.1, 0.1)$

---

## IV. Talk-and-Tweak：自动 Language Labeling

### 4.1 动机

人在 rollout 时用 SpaceMouse 物理介入（HG-DAgger 范式 [Kelly et al. 2019]），但只产生 $(s, a^{intv})$ pair，没有 language label。

Paper 提出一个 rule-based mapping 把 tweak 转成 talk：

Reference: HG-Dagger: https://arxiv.org/abs/1810.02890

### 4.2 公式

每个 action 是 7-dim：$(\Delta x, \Delta y, \Delta z, \Delta roll, \Delta pitch, \Delta yaw, gripper)$

只看 translational 部分 $a^{intv}[0:3]$。

**累积位移**（5 步窗口）：

$$\Delta_t = \sum_{j=0}^{J-1} a_{t+j}^{intv}[0:3] \tag{12}$$

- $J = 5$：time window
- $\Delta_t \in \mathbb{R}^3$：累积 translational 位移

**Threshold 判断**（per axis $d \in \{x, y, z\}$）：

$$l_{rf_t}^d = \begin{cases} 
\text{"positive } d \text{ direction"}, & \Delta_t^d > \sigma \\
\text{"negative } d \text{ direction"}, & \Delta_t^d < -\sigma \\
\text{"no command"}, & |\Delta_t^d| \leq \sigma
\end{cases} \tag{13}$$

- $\sigma = 0.001$m：threshold
- 三个 axis 拼起来：例如 "move right and forward"

最终生成 triplet：$(s_t, a_t^{intv}, l_{rf_t})$

### 4.3 Intuition

这个 trick 让我想到几件事：

1. **类似 "Yell at Your Robot" (YAR) [Shi et al. 2024]**：他们直接让人提供 language correction；本文反过来，**让人提供 physical correction，自动转成 language**。这是一个数据效率极高的设计：人不需要打字，只用手动微调，label 自动来。

   Reference: YAR: https://y-robot.github.io/

2. **类似 VLM 训练里的 "auto-captioning"**：用一个大 model 给 image 自动 caption，训练一个小 model；这里是 rule-based mapper 当 captioner，简单但 reliable。

3. **Threshold σ = 1mm 的物理意义**：毫米级 precision 任务里，1mm 是 noise floor，能过滤抖动。

4. **为什么只看 translation 不看 rotation**：rotation 的"语言化"不直观（"rotate clockwise 15 degrees" 太复杂）。Paper 简化了，但留下一个 extension 空间。

---

## V. Multi-Task Learning 设计

### 5.1 架构（Fig. 4）

- **Shared actor**（一个 policy $\pi_\theta^w$ 学所有 task）
- **Per-task critic** $Q_{\psi_i}(s, a)$，每个 task 独立 critic
- **Per-task buffer**：$\mathcal{D}_{demos}^i, \mathcal{D}_{rollouts}^i, \mathcal{D}_{intv}^i$

Critic 结构：
- ResNet encode RGB
- MLP encode proprioception
- MLP encode action
- Concatenate → MLP head → Q-value

### 5.2 为什么 Shared Actor + Per-task Critic

这是 multi-task RL 里的经典 trade-off：
- Shared actor：参数效率高，task 之间 knowledge transfer
- Per-task critic：每个 task 的 value 估计不会互相干扰
- 类似 "Multi-task SAC" 或 "Multi-task DDPG" 设计

类似的 idea 可以参考：
- Multi-task DDPG: https://arxiv.org/abs/1802.06011
- MT10 benchmark: https://github.com/rlworkgroup/gym-pomdps

### 5.3 Adaptive Task Weighting

如果直接 uniform sample $\frac{1}{N}\sum_{i=1}^{N}\mathbb{E}[Q_{\psi_i}(s, \pi_\theta^w(h))]$，会出现 dominant task 现象：一个 task 学得快，Q 值大，梯度就大，其他 task 被 starve。

**解决**：用 $\epsilon_i$ 反比例于 Q-value 来 reweight：

$$-\frac{1}{N}\sum_{i=1}^{N}\epsilon_i \cdot \mathbb{E}_{s \sim \mathcal{D}_i}\left[Q_{\psi_i}(s, \pi_\theta^w(h))\right] \tag{15}$$

$$\epsilon_i = \begin{cases}
\epsilon_{max}, & \epsilon_i > \epsilon_{max} \\
\frac{\sum_{i=1}^{N}\overline{Q}_i}{N\overline{Q}_i + Nc}, & \epsilon_{min} < \epsilon_i \leq \epsilon_{max} \\
\epsilon_{min}, & \epsilon_i \leq \epsilon_{min}
\end{cases} \tag{16}$$

- $\overline{Q}_i = \mathbb{E}_{s \sim \mathcal{D}_i}[Q_{\psi_i}(s, \pi_\theta^w(h))]$：task $i$ 的当前 Q-value
- $c = 0.1$：数值稳定常数
- $\epsilon_{max} = 1.2$, $\epsilon_{min} = 0.8$：clip 范围

Intuition：Q 大 = 学得好 = 降权；Q 小 = 学得差 = 加权。这本质上是 **multi-armed bandit 里的 UCB-style 反向 weight**。

这种 self-paced learning / hard example mining 思路在很多地方出现过：
- Self-Paced Learning: https://arxiv.org/abs/2010.05217
- Class-balanced loss: https://arxiv.org/abs/1901.06683

---

## VI. 实验结果深度分析

### 6.1 主实验（Table I）

| Method | Place upright | Pick up | Assemble | Average | Avg episode length |
|---|---|---|---|---|---|
| HG-Dagger [18] | 28 | 28 | 20 | 25.3 | 41.0 |
| HIL-ConRFT [12] | 0 | 0 | 0 | 0 | = |
| DSRL [20] | 28 | 28 | 16 | 24.0 | 27.3 |
| Ours (w/o dual-actor) | 88 | 96 | 76 | 86.7 | 30.7 |
| **Ours (w/ dual-actor)** | **100** | **100** | **100** | **100** | **30.7** |

**几个关键观察**：

1. **HIL-ConRFT 在 multi-task 上 0% 失败**：这是 paper 的核心 motivation。原版 HIL-ConRFT 是 single-task，直接迁移到 multi-task 完全不行。原因是 flat optimization（一个 critic 全包）在 multi-task 上 Q 值会 collapse，policy 被错误信号带偏。

2. **HG-Dagger 25%：BC 的天花板**：人的 demonstration 有 noise，BC 直接学不到精确 control。RL 的 Q-maximization 让 policy 能 explore 出更优 trajectory。

3. **DSRL 24%：latent RL 但 critic 不准**：DSRL 在 latent noise 做 RL，但 critic 只用 offline 数据训，OOD action 的 value 估不准，policy 学不到。

4. **Ours (w/o dual-actor) 86.7%：multi-task balancing 的功劳**：去掉 refinement actor，纯靠 multi-task balancing 和 per-task critic 也有 86.7%。Dual-actor 把它推到 100%。

5. **Episode length 几乎一样**：30 vs 30.7，说明 dual-actor 主要提升 success，不提升 speed。这与设计一致——refinement actor 只在 error 发生时介入。

### 6.2 训练曲线（Fig. 5）

- Dual-actor：20 分钟 → 60%，101 分钟 → 100%
- Single-actor：20 分钟 → 30%，128 分钟 → 86.7%

Dual-actor 节省 **约 27 分钟**（21% 提速），更关键是稳定性：single-actor 在第三个 task 上 fluctuating，dual-actor 一直稳定。

这与 latent space fine-tuning 的本质有关——参数空间小，更新 smooth。

### 6.3 VLA Backbone 泛化（Table II）

| Task | Octo (0.27B) | SmolVLA (0.6B) |
|---|---|---|
| Place upright | 100 | 100 |
| Pick up | 100 | 100 |
| Assemble | 100 | 100 |
| Average | 100 | 100 |

两个 backbone 都 100%。但 SmolVLA 用了 prefix-mask pooling 提取 task embedding，是一个非 trivial 的实现。

### 6.4 Long-Horizon（RQ4）

| # bolts | Success |
|---|---|
| 1 (3 steps) | 90% (9/10) |
| 2 (6 steps) | 60% (6/10) |
| 3 (9 steps) | 60% (6/10) |
| 4 (12 steps) | 50% (5/10) |

Note：每个 bolt 是 3 steps (place upright + pick up + assemble)，所以 4 bolts = 12 步。50% 在 12 步连续操作上其实挺不错的，相当于每步错误率约 5.7%（$(0.5)^{1/12}$）。

衰减主要来自 occluded slot view，error 累积。

### 6.5 Multi-Robot（Fig. 7）

- Train A → Test A: 100%
- Train A → Test B: 0%（硬件差异 + background）
- Train A+B → Test A: ~100%
- Train A+B → Test B: ~100%
- 47 分钟 vs 单 robot 100 分钟 → **2× 提速**

Centralized learner + decentralized actors，类似 A3C 或 IMPALA 架构。这个 scaling 表现令人鼓舞。

Reference:
- A3C: https://arxiv.org/abs/1602.01783
- IMPALA: https://arxiv.org/abs/1802.01561

---

## VII. 关联联想 & 直觉拓展

### 7.1 与 RLHF 的对应

把 dual-actor 跟 RLHF [Ouyang et al. 2022] 对照：

| RLHF | Dual-Actor |
|---|---|
| SFT model | Pretrained VLA |
| Reward model | Q-function |
| PPO update policy weight | Latent noise mean shift |
| KL constraint to SFT | $\mathcal{L}^{Reg}$ |
| Prompt → response | State → action |

非常 similar！特别是 $\mathcal{L}^{Reg}$ 对应 RLHF 里的 KL penalty。

Reference: InstructGPT: https://arxiv.org/abs/2203.02155

### 7.2 与 Residual RL 的对应

ResiP [Ankile et al. 2024] 是 residual RL：在 frozen BC policy 上加一个 residual RL policy。本 paper 在 frozen diffusion policy 上加一个 residual **on the noise mean**。本质都是 residual learning。

Reference: ResiP: https://arxiv.org/abs/2407.16677

### 7.3 与 Diffusion Steering via RL (DSRL) 的对比

DSRL：用 RL optimize latent noise 采样分布
Dual-Actor：用 BC + RL 训练一个 noise mean predictor，conditioned on language

DSRL 的 critic 全 offline 训，dual-actor 是 online RL，critic 持续更新。

### 7.4 ControlNet 的类比

让我联想一个非常有意思的 connection：**ControlNet [Zhang et al. 2023]**！

ControlNet 的设计：
- 把 pretrained diffusion model 的 UNet encoder **frozen**
- 复制一份 trainable copy
- 用 zero convolution 连接，保证初始时 trainable branch 不影响 output
- 用 conditioning (edge map, depth, pose) 训练 trainable branch

Dual-Actor 的设计完全类似：
- Primary actor frozen（不更新参数，仅作为 forward pass）
- Refinement actor 是 trainable
- Regularization loss 保证初始时 refinement actor 输出 ≈ 0（等价于 zero conv）
- 用 language command conditioning 训练

所以可以理解为：**Dual-Actor = ControlNet for Robot Action Diffusion**

Reference: ControlNet: https://arxiv.org/abs/2302.05543

### 7.5 LoRA 的类比

更激进的类比：refinement actor 像 LoRA。LoRA 在 frozen weight 上加 low-rank update；refinement actor 在 frozen policy 的 noise 上加 mean offset。两个都是"低参数量 perturbation on top of frozen base"。

### 7.6 Pretraining-Finetuning 范式对应

整个 paper 的范式可以映射到 LLM 的训练 pipeline：

| LLM | VLA (本 paper) |
|---|---|
| Pretraining (next token prediction) | VLA pretraining (BC on demos) |
| SFT (instruction tuning) | Warm-up fine-tuning (Cal-QL + BC) |
| RLHF (PPO) | Online RL fine-tuning |
| Constitutional AI | Human-in-the-loop talk-and-tweak |

这是 robotics 正在走的路径，paper 印证了 LLM 训练范式在 robot learning 上的可迁移性。

### 7.7 关于 $K$ 的隐含问题

Paper 没有明确给 $K$ 的值，也没有讨论 $K$ 的影响。但直觉上：
- $K$ 大 → action 分布 spread 大 → 探索强但精度差
- $K$ 小 → action 集中 → 精度高但探索弱
- Refinement actor 调 mean，不调 variance，**这是一个简化假设**——可能 mean shift 本身就够用了。

如果允许 refinement actor 也调 covariance，会更接近 **conditional normalizing flow** 或 **conditional VAE**。

### 7.8 Limitation 的诚实评估

Paper 自己列了：
1. Long-horizon error accumulation
2. Dependence on human intervention quality

我额外想到：
3. **Language vocabulary 太小**：只有 "positive/negative x/y/z direction"，最多 6 个 axis。复杂 motion（"rotate gently to align with the slot")无法表达。可以接一个 VLM 做 captioning。
4. **Refinement actor 只在 intervention 数据上训**：意味着如果某 state 从未被 human 介入过，refinement actor 没见过，可能泛化差。Paper 没有讨论这个 OOD 问题。
5. **Single-step consistency policy 的精度限制**：consistency policy 把 multi-step denoising 蒸馏成 single-step，可能在毫米级 precision 任务上损失精度。这是为什么需要 refinement actor 补偿。
6. **Multi-robot communication cost**：Fig. 7 显示 2× 提速，但没讨论 bandwidth / latency。如果是 4 robots、8 robots 呢？

---

## VIII. 训练流程的总览（Algorithm 1 解析）

Algorithm 1 把训练分两阶段：

**Stage I: Offline Warm-up**
```
for each offline step:
    for each task i:
        sample B/N_tasks demos from D_demos^i
        update critic Q_ψi with Cal-QL loss
    update shared policy π_θ^w with BC + Q loss (mostly BC)
```

**Stage II: Online Interaction**
```
Learning Thread:
    for each online step:
        for each task i:
            sample half from D_demos^i, half from D_rollouts^i
            update critic Q_ψi with standard TD loss
        update π_θ^w (now with more Q weight)
        update π_φ with D_talk-tweak (BC + Q + Reg)

Interaction Thread:
    for each task i:
        if no human intervention:
            take action a ~ π_θ^w(s, l_task)
            store (s, a, r, s') in D_rollouts^i
        else:
            take human-intervened action a^intv
            store (s, a^intv, r, s') in D_intv^i
    augment D_talk-tweak with D_intv^i
    augment D_demos^i with D_intv^i  ← 这个很关键！
```

**注意第 29 行**：human intervention data **同时进入 demos buffer 和 talk-tweak buffer**！这意味着 primary actor 也学 human intervention，refinement actor 学 language-conditioned version。是数据复用。

---

## IX. 总结：这个工作的真正贡献

Karpathy 老师，我把这个 paper 的核心 idea 总结成三个层次：

**Level 1 (Surface)：** 一个 dual-actor RL fine-tuning 框架，用 language command 做 latent space refinement，101 分钟达到 100% 成功率。

**Level 2 (Methodology)：** 借鉴 LLM 训练范式（pretrain → warm-up → online RL → human-in-the-loop），把 RLHF 的 KL constraint 思想搬到 robot policy 的 latent noise 上。

**Level 3 (Philosophical)：** 在 frozen 的预训练 VLA 上做 fine-tuning，**更新点越少越好、越窄越好**。Refinement actor 只动 mean shift，参数量小、作用域窄，恰好是 "minimal intervention" 原则。这与 ControlNet、LoRA 的设计哲学一脉相承——**保留 pretrain 的通用性，只在最小子空间做 adaptation**。

这个 philosophy 我想 you 应该很有共鸣——你在 "State of GPT" 演讲里提到过 "the bitter lesson"，pretrain + small fine-tuning > end-to-end training。这篇 paper 在 robotics 上做了同样的事情。

Reference: The Bitter Lesson: http://incompleteideas.net/IncBitterLesson.html

---

## X. 未来方向联想

1. **VLM auto-captioning**：现在 rule-based mapping 只生成 "move right" 这种 simple command。可以让 GPT-4V 看 intervention 前后的 frame，生成 "move right to align with the slot" 这种 contextual command，让 refinement actor 学更丰富的语义。
2. **Refinement actor 学 covariance**：让 $\pi_\phi$ 输出 $\mu, \Sigma$，policy distribution 更灵活。但要小心 covariance collapse。
3. **Hierarchical refinement**：当前 language 是 axis-level。可以引入 hierarchical command："first move right, then grasp"——类似 hierarchical RL。
4. **Cross-embodiment generalization**：dual-robot 已经做了，但跨 morphology（arm vs quadruped vs humanoid）还没做。如果 refinement actor 的 language 是 embodiment-agnostic 的，应该可以 transfer。
5. **World model integration**：现在 Q-function 是 task-specific，每个 task 一个。如果用 world model 做 task-agnostic value estimation，可以省去 per-task critic。

---

## XI. 相关工作与延伸阅读

**VLA Models:**
- π0: https://www.physicalintelligence.company/blog/pi0
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://openvla.github.io/
- Octo: https://octo-models.github.io/
- SmolVLA: https://arxiv.org/abs/2504.05299

**Diffusion Policy & Consistency:**
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Consistency Models: https://arxiv.org/abs/2303.01469
- Boosting Continuous Control with Consistency Policy: https://arxiv.org/abs/2310.06343

**Human-in-the-Loop RL:**
- HIL-SERL: https://arxiv.org/abs/2410.21845
- HIL-ConRFT: https://arxiv.org/abs/2502.05450
- RLIF: https://arxiv.org/abs/2311.12996
- HG-DAgger: https://arxiv.org/abs/1810.02890
- Yell at Your Robot: https://y-robot.github.io/

**Offline-to-Online RL:**
- Cal-QL: https://arxiv.org/abs/2303.05479
- CRR: https://arxiv.org/abs/2006.15134
- AWAC: https://arxiv.org/abs/2006.09359

**Policy Adaptation:**
- ResiP (Residual RL): https://arxiv.org/abs/2407.16677
- DSRL (Diffusion Steering): https://steeringdiffusion.github.io/

**类比与 inspiration:**
- ControlNet: https://arxiv.org/abs/2302.05543
- LoRA: https://arxiv.org/abs/2106.09685
- InstructGPT (RLHF): https://arxiv.org/abs/2203.02155
- The Bitter Lesson: http://incompleteideas.net/IncBitterLesson.html

---

希望这个讲解能帮你 build 直觉。整体来看，这篇 paper 是一个工程上挺 solid 的工作——把 LLM 范式（pretrain + RLHF-like fine-tuning + KL constraint）成功迁移到 robot VLA，dual-actor 的设计哲学（minimal intervention on frozen base）很 elegant。我最喜欢的是 talk-and-tweak 这个 auto-labeling trick，一个简单 rule-based mapper 解决了 RL 机器人学习里 data scarcity 的核心痛点，是非常 Karpathy-style 的工程 trick。
