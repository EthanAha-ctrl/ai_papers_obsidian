---
source_pdf: OmniVTA Visuo-Tactile World Modeling for Contact-Rich.pdf
paper_sha256: b1149831570b1db5d63e114abbdc9b60270feb489ef67f2fb09df1b8f95d1783
processed_at: '2026-08-05T23:34:14-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 OmniVTA

## 一、这篇 paper 在搞啥

你想想, 让机器人擦桌子、削苹果、插 USB — 这些任务有个共同点: **得用力, 得接触**。光靠摄像头看是搞不定的, 因为你看不见"力"。擦桌子时抹布贴没贴紧? 削皮时刀子压得够不够? 插 USB 时对准没对准? 这些信息全在"触觉"里。

但现在的触觉机器人有两个大坑:

**第一个坑 — 没数据**。之前最大的 visuo-tactile dataset 也就几千条 trajectory, 几个任务。这就好比你只看过 5 个人削苹果, 然后让你总结削苹果的规律 — 样本太少。

**第二个坑 — 没用对**。之前的方法把 tactile sensor 当"辅助眼睛"用 — 就是给 policy network 多塞一路输入, 模型自己看着办。这就像给你一只眼睛+一只手, 但不告诉你手摸到的是啥, 你自己猜吧。更糟的是, 这些方法生成一小段动作后**开环执行**, 中间出了啥意外(滑了、歪了)完全不管。

OmniVTA 说: 别这么搞。人做这些任务靠的是**两套系统**:

- **慢系统**(大脑): 提前想"接下来几秒会摸到啥", 做规划
- **快系统**(脊髓反射): 摸到不对立刻本能纠正, 60Hz 高频反应

paper 就是把这两套系统在 robot 上复刻了一遍。

## 二、数据集 OmniViTac: 先把粮草备齐

### 2.1 规模

他们搞了 **21,879 条 trajectory**, 覆盖 **86 个 task**, **126 个 object**, 用了 **4 种 tactile sensor**。对比之前最大的 AgiBot World 才 5,337 条 — 直接 4 倍量级碾压。

### 2.2 怎么采的

采集数据的工程挺巧。他们用了**两套硬件**:

- **xArm-7 机械臂**: 人在 gravity compensation 模式下手动引导, 或者用 [GELLO](https://arxiv.org/abs/2402.11236) 遥操作 — 数据直接是 robot-aligned 的
- **TacUMI**: 手持设备, 基于 [FastUMI](https://arxiv.org/abs/2402.10329), 配 RealSense T265 估计 6-DoF pose, 200Hz

两套硬件用**完全一样的 parallel-jaw gripper** + 模块化 fingertip tactile sensor。这是为了把"采集设备"和"执行设备"之间的 domain gap 压到最小 — 你用手持采集的数据, 机械臂拿来直接能用。

tracking drift 超过 8mm 的 trajectory 直接扔掉, 不凑数。

### 2.3 六类 interaction pattern — 这才是关键创新

他们没按"视觉动作"分类(像 pick-and-place, pour, stack 这种), 而是**按触觉物理机制**分了六类:

| Pattern | 通俗解释 | 触觉长啥样 |
|---|---|---|
| **Wiping** 擦拭 | 抹布贴着桌子推 | 大面积 + 切向摩擦力持续 |
| **Peeling** 削皮 | 刀贴着表面剥 | 切向+法向力耦合, 持续接触 |
| **Cutting** 切割 | 刀插进物体 | 法向力大, 切断瞬间力骤降 |
| **Assembly** 装配 | 插 USB, 拧盖子 | 局部精确接触, 多方向力, 容差小 |
| **Adjustment** 调整 | 手心转笔 | 扭转+剪切, 持续高频接触 |
| **Grasping** 抓取 | 捏蓝莓 | 法向力控制, 防碎防滑 |

每类的 **effective contact ratio**(轨迹中触觉有信号的比例)差异巨大:
- Adjustment 0.67 — 一直摸着
- Wiping 0.49
- Peeling 0.41
- Cutting 0.27 — 大部分时间在视觉对准, 真正切的时间短

**关键 insight**: 他们做 t-SNE 可视化(Fig. 4f), 发现这六类在 tactile latent space 里**天然分簇**, 而且分簇方式物理上 intuitive — Wiping 和 Peeling 靠近(都靠摩擦), Assembly 单独一块(局部精确)。这说明 dataset 真的抓到了 contact mechanics 的结构。

这给后面的方法设计提供了**两条 structural property**:
1. **Spatial locality**: 接触前触觉几乎全零, 接触后局部爆发
2. **Contact-driven dynamics**: 信号变化由物理接触驱动

整个 OmniVTA 的架构就是围绕这两条 property 设计的。

## 三、OmniVTA 整体架构: 慢快分层

```
                ┌──────────────────────────────────┐
                │  慢系统 (15Hz)                    │
                │  ┌────────┐   ┌────────────────┐ │
  Vision ──────►│  │  VTWM  │──►│  Adaptive      │ │
  Tactile ─────►│  │ (预测) │   │  Fusion Policy │─┼─► Action Chunk
  Proprio ─────►│  └────────┘   └────────────────┘ │
                └──────────────────────────────────┘
                                ↓ 加权融合
                ┌──────────────────────────────────┐
                │  快系统 (60Hz)                    │
                │  Reflexive Tactile Controller     │─► 微调动作
                │  (摸到不对立刻纠正)              │
                └──────────────────────────────────┘
```

慢系统负责"想接下来几步怎么走", 快系统负责"摸到不对赶紧调整"。最终动作是两者的加权和, 快系统的贡献由预定义系数缩放。

这套设计就是模仿人 — 你切菜时脑子在规划"下一刀往哪儿切", 手上同时在感受"刀有没有贴紧", 贴不紧瞬间微调。

## 四、模块一: TactileVAE — 把触觉压成紧凑表示

### 4.1 输入选择: 不用图像, 用 marker displacement

光学触觉传感器(Xense, GelSight)输出 RGB 图像 700×400, 直接编码太重。paper 改用 **3D marker displacement tensor**: $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$

- $H, W$: marker 网格的行数列数(Xense 是 $35 \times 20$)
- 最后一维 3: 每个标记点在 x, y, z 三个方向的位移

这相当于把"高分辨率图像"压成"稀疏 displacement field", 既保留了接触几何信息, 又大幅降维 — 这是后面能跑 60Hz 的前提。

### 4.2 Encoder: causal 3D 卷积

结构: **projection-in (causal 3D conv)** → $M$ 个 downsampling module → **projection-out (causal 3D conv)**

输出 latent: $\mathbf{z}_t \in \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times C}$, 其中 $s = 2^M$ 是 spatial downsampling factor, $C$ 是 latent channel 数。

**Causal** 是关键 — 时间 $t$ 的 latent 只依赖 $\le t$ 的观察, 不偷看未来。这跟 [WaveNet](https://arxiv.org/abs/1609.03499) 和 video diffusion 中的 causal mask 是同一个道理, 保证训练和部署时模型看到的信息一致。

### 4.3 Decoder: 隐式神经表示 (INR)

这是 TactileVAE 最有创意的部分。常规 VAE 重建像素网格, 但 marker displacement 本质上是 **elastomer 表面的连续 deformation field**, 用 INR 建模更自然:

$$\mathbf{d}(\mathbf{x}) = \mathcal{D}_\theta\big(\gamma(\mathbf{x}), \Phi(\mathbf{z}_t, \mathbf{x})\big) \tag{1}$$

逐个讲变量:
- $\mathbf{x} \in \mathbb{R}^2$: 空间坐标, 就是 marker 平面上的查询点
- $\mathbf{z}_t$: encoder 输出的 latent feature map
- $\gamma(\cdot)$: positional encoding, 类似 [NeRF](https://arxiv.org/abs/2003.08934) 的高频 encoding, 让 MLP 能学高频细节
- $\Phi(\mathbf{z}_t, \mathbf{x})$: 从 latent feature map 在 $\mathbf{x}$ 处做 spatial interpolation 取出 local feature
- $\mathcal{D}_\theta$: MLP decoder, 输出 $\mathbf{d}(\mathbf{x}) \in \mathbb{R}^3$ (该点的 3D displacement)

训练 loss:
$$\mathcal{L}_{\text{TacVAE}} = \|\mathbf{d}(\mathbf{x}) - \hat{\mathbf{d}}(\mathbf{x})\|_2^2 + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} \tag{2}$$

$\lambda_{\text{KL}} = 10^{-6}$ — 极小, 强调 reconstruction。这里不是 [β-VAE](https://arxiv.org/abs/1611.00770) 那种想学 disentangled 表示的场景, 而是要保真。

### 4.4 INR 的好处

Table V 的 ablation 数据:
- 不用 implicit decoder: 0.126 (GelSight-Mini L2)
- 只加 position embedding: 0.102
- 用 single token 不用 spatial feature map: 0.107
- **完整 INR decoder: 0.047**

保留 **spatial feature map**(而非 single global token)也很关键 — 这保留了 spatial locality, 对下游 contact localization 是基础。

## 五、模块二: Visuo-Tactile World Model — 预测接下来会摸到啥

这是 paper 的核心。World model 干的事就是: 给过去的 vision+tactile+action, 预测未来几帧的 vision 和 tactile latent。

### 5.1 两路 diffusion transformer

VTWM 是 two-stream 架构, vision 和 tactile 各一个 [spatial-temporal diffusion transformer](https://arxiv.org/abs/2212.09748) (DiT 的时空扩展版)。

输入: 过去 $c$ 帧 observation 作 condition, 迭代 denoise 生成 $K$ 帧 future latents。

Diffusion 训练目标:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{\mathbf{z}_o, \epsilon, t}\left[\sum_{i=1}^{K} (1 - m_i) \odot \|\epsilon_i - \epsilon_\theta(\mathbf{z}_o, t)_i\|_2^2\right] \tag{3}$$

变量解释:
- $\mathbf{z}_o = \{\mathbf{z}_o^1, \ldots, \mathbf{z}_o^K\}$: observation latents 序列, 包括 tactile latent $\mathbf{z}_t$ 和 visual latent $\mathbf{z}_v$
- $\epsilon$: 加的 Gaussian noise
- $t$: diffusion timestep (扩散步骤)
- $\epsilon_\theta(\mathbf{z}_o, t)$: 神经网络预测的 noise
- $m_i$: temporal mask, 第 $i$ 帧是已知过去 (mask=1) 还是要预测的未来 (mask=0)
- $(1 - m_i)$: 只在要预测的帧上算 loss

Modality encoding:
- Visual branch: [SD-VAE](https://arxiv.org/abs/2112.10752) 编图像到 latent
- Tactile branch: 用预训练的 TactileVAE

### 5.2 关键设计 1: action 用 2D 投影

Multi-modal Conditioner 把 vision/tactile/action 融合成 shared condition vector。这里 action 的表示方式特别有意思 — 用 **2D image-plane projection of 3D end-effector position**, 而不是 3D absolute/relative action。

Table VII 的对比:
- 3D absolute action: L2=0.075, cos=0.72
- 3D relative action: L2=0.056, cos=0.88
- **2D projected action: L2=0.042, cos=0.91**

**为啥 2D 反而更好?** Intuition 是: action condition 主要传达"motion intent"(图像上要往哪儿走), 2D 表示和 visual observation 自然在同一坐标系, 不会有 camera-to-world 的 frame mismatch。这跟 [RT-2](https://arxiv.org/abs/2307.15818) 把 action tokenize 进 VLM vocab 的思路异曲同工 — 让 action 和 observation 在同一表示空间。

### 5.3 关键设计 2: Dynamic-aware Weighted Loss

这是针对 tactile **contact-driven dynamics** 性质的专门设计。

Standard diffusion loss 对所有空间位置一视同仁, 但 tactile 信号在空间上极度 sparse — 没接触时几乎全零, 接触时局部爆发。等权训练模型会偏向"全零"这个 trivial 解。

paper 设计两个 weight map:

**Dynamic weight** (时间差分, 抓"变化快慢"):
$$w_{\text{dyn}}^i = \text{resize}\left(\text{clip}_{[0,1]}\left(\|X_{i+1} - X_i\|_2\right)\right) \tag{4}$$

$X_k$ 是第 $k$ 帧触觉, $\|X_{i+1} - X_i\|_2$ 衡量帧间变化幅度, clip 到 [0,1] 防止极端值, resize 到 latent 分辨率。

**Amplitude weight** (响应幅度, 抓"接触强度"):
$$w_{\text{amp}}^i = \text{resize}\left(\text{clip}_{[0,1]}\left(\|X_i\|_2\right)\right) \tag{6}$$

加权 loss:

$$\mathcal{L}_{\text{dyn}} = \mathbb{E}_{\mathbf{z}_o, \epsilon, t}\left[\sum_{i=2}^{K} w_{\text{dyn}}^i \odot (1-m_i) \odot \|\epsilon_i - \epsilon_\theta(\mathbf{z}_o, t)_i\|_2^2\right] \tag{5}$$

$$\mathcal{L}_{\text{amp}} = \mathbb{E}_{\mathbf{z}_o, \epsilon, t}\left[\sum_{i=2}^{K} w_{\text{amp}}^i \odot (1-m_i) \odot \|\epsilon_i - \epsilon_\theta(\mathbf{z}_o^i, t)\|_2^2\right] \tag{7}$$

注意 (5) 用整段序列 $\mathbf{z}_o$, (7) 用单帧 $\mathbf{z}_o^i$ — 因为 amp weight 只依赖当前帧 magnitude, 跟时间无关。

总 loss:
$$\mathcal{L}_{VTWM} = \mathcal{L}_{\text{diffusion}} + \lambda_1 \mathcal{L}_{\text{dyn}} + \lambda_2 \mathcal{L}_{\text{amp}} \tag{8}$$

$\lambda_1 = \lambda_2 = 1.0$。

**Intuition**: 这跟 [focal loss](https://arxiv.org/abs/1708.02002) 处理类别不平衡是一个思路 — "无接触"的 trivial 区域不用花太多 capacity 去拟合, 把模型注意力聚焦在"有接触 + 接触在变"的时空位置上。

### 5.4 实验对比

Table VI 跟四个 baseline 拼:
- [UVA](https://arxiv.org/abs/2503.00200): unified token sequence
- exUMI: latent diffusion conditioned on visual+action
- [KineDex](https://arxiv.org/abs/2505.01974): joint action+force diffusion
- [ForceMimic](https://arxiv.org/abs/2502.09909): 3D observation conditioned

OmniVTA 在所有 6 个 task 全面领先:
- Wipe: L2_avg=0.059 vs 次优 0.082
- Cut: L2_avg=0.050 vs 次优 0.090
- Assembly: L2_avg=0.025 (大幅领先)
- Grasp: L2_avg=0.010

Ablation 还显示:
- Joint visual-tactile generation 比单 stream tactile 预测更好 (L2: 0.035 vs 0.041) — visual branch 提供 complementary global cue
- Dynamic weighting 进一步降 L2 到 0.035, cos 升到 0.93

## 六、模块三: Adaptive Visuo-Tactile Fusion Policy — 视觉和触觉该信谁

### 6.1 LTD Encoder: 编码"未来-现在"的差

这是 paper 的核心创新。Motivation 还是触觉的 spatial locality — 接触前没用, 接触时才有信息。如果像之前工作一样把 historical tactile 直接和 visual feature concat, 模型抓不到"接下来要发生啥 contact 事件"。

LTD Encoder 构造:
- $\mathbf{f}_t^c$: current tactile feature, 2D conv + max pool 空间聚合得到 global vector
- $\mathbf{f}_t^p$: predicted multi-frame tactile feature, 先 per-frame spatial 聚合, 再 1D conv + max pool 时间聚合

最终 tactile 表示:
$$\mathbf{f}_t = \text{concat}(\mathbf{f}_t^c, \mathbf{f}_t^p, \mathbf{f}_t^p - \mathbf{f}_t^c) \tag{9}$$

**第三个分量 $\mathbf{f}_t^p - \mathbf{f}_t^c$ 是精髓** — predicted 减 current, 直接 highlight "接下来要发生什么 contact 变化"。

这跟 [Wolpert & Flanagan 2001](https://www.sciencedirect.com/science/article/pii/S0960982201004511) 提出的人脑 motor cortex 机制对应 — 大脑用 efference copy 生成 sensory prediction, 再和实际 feedback 比对, 差异用于 error correction。LTD 就是把这个机制显式化在 network 里。

### 6.2 Gating: 自适应权重

受 [FoAR](https://arxiv.org/abs/2501.02505) 启发, 用 predicted contact probability 调制 vision/touch 权重。

**Contact probability 预测**:
- 输入: $\mathbf{f}_t$ (LTD encoded tactile)
- 网络: MLP + sigmoid → 输出 $p_{\text{contact}} \in [0,1]$
- 监督: 自动生成 contact label (触觉 deformation magnitude 阈值化), BCE loss $\mathcal{L}_{\text{bce}}$

**Gating network**:
- 输入: concat(contact logit, $\mathbf{f}_t$)
- 网络: 2 个 FC layer
- 输出: $W_t, W_v$, 满足 $W_t + W_v = 1$ (per-channel)

注意 gating network **只看 tactile, 不看 visual** — 因为 tactile 通过 world model 已经编码了 future contact dynamics, 不需要视觉就能判断"接下来要不要靠 touch", 还能简化模型。

融合:
$$\mathbf{f}_{vt} = \text{concat}\left(W_v \odot \mathbf{f}_v, W_t \odot \tilde{\mathbf{f}}_t\right) \tag{10}$$

- $\mathbf{f}_v$: ResNet-18 提取的 visual feature
- $\tilde{\mathbf{f}}_t$: tactile feature 经 linear projection 维度对齐
- $\odot$: per-channel 乘法

**另一个关键决策**: 只用 current + historical visual, **不用 future visual prediction**。理由:
1. Current image 已提供足够 global context
2. Predicted tactile 已 capture potential contact dynamics
3. 加 visual generation branch 推理时间从 230ms 暴增到 480ms (Table IX), 性价比差

Table VIII 显示加 visual gen 只把 avg 从 0.53 提到 0.54, 几乎没用。

### 6.3 Gating 行为可视化

Fig. 14 是 paper 一个亮点。可视化显示:
- 任务开始(没接触): contact probability≈0, tactile weight≈0, **全靠视觉**
- 接触建立: contact probability 上升, tactile weight 同步上升, visual weight 下降
- 接触中: tactile weight 主导

这正符合直觉 — 没摸到时只能看, 摸到了就主要靠手。

Fig. 16 更关键: 当 tactile prediction accuracy 降到 60%, gating 网络无法正确估计 contact probability, modality weighting 失衡, policy 性能随之崩溃。**这说明 accurate tactile prediction 是整个 policy 的基石**, 不是可有可无的装饰。

### 6.4 Diffusion Policy

Action 生成用 [DDPM](https://arxiv.org/abs/2006.11239), 输出 $H$ 个 coarse action 组成的 chunk。

Reverse diffusion:
$$A_{c,t-1} = \alpha_t A_{c,t} - \gamma_k \epsilon_\theta(A_{c,t}, t, \mathbf{f}_c) + \sigma_t \mathcal{N}(0, I) \tag{11}$$

变量:
- $A_{c,t}$: timestep $t$ 的 noisy action chunk
- $\epsilon_\theta$: noise predictor (用 [FiLM](https://arxiv.org/abs/1709.07871) 注入 condition)
- $\mathbf{f}_c = \text{concat}(\mathbf{f}_{vt}, \mathbf{s})$: fused visuo-tactile feature + robot proprioception $\mathbf{s}$
- $\alpha_t, \gamma_t, \sigma_t$: scheduler 系数
- $\mathcal{N}(0, I)$: 标准高斯噪声

训练 loss:
$$\mathcal{L}_{\text{act}} = \mathbb{E}_{t, A_{c,0}, \epsilon_t}\left[\|\epsilon_t - \epsilon_\theta(\bar{\alpha}_t A_{c,0} + \bar{\beta}_t \epsilon_t, t, \mathbf{f}_c)\|_2^2\right] \tag{12}$$

- $\bar{\alpha}_t, \bar{\beta}_t$: 累积 noise schedule 系数
- $A_{c,0}$: clean action chunk (ground truth)

Policy 总 loss:
$$\mathcal{L}_{AFP} = \mathcal{L}_{\text{act}} + \lambda_{ct} \mathcal{L}_{\text{bce}} \tag{13}$$

$\lambda_{ct} = 0.2$, contact prediction 作为辅助任务帮 gating 学得更好。

## 七、模块四: Reflexive Latent Tactile Controller — 60Hz 反射弧

### 7.1 为啥需要

Slow policy 输出 action chunk 是**开环**执行, 在 contact-rich 任务中应对不了 rapid contact change(滑了、歪了、外界扰动)。RLTC 提供 60Hz 高频 closed-loop correction, 模仿人 spinal reflex arc([Augurelle 2003](https://journals.physiology.org/doi/10.1152/jn.00275.2002)) — 你碰到烫的东西瞬间缩手, 不等大脑反应。

### 7.2 工程细节

几个 temporal alignment 巧思:

1. **Single-frame → 短序列**: TactileVAE 沿时间压缩了 $M$ 倍, single-frame 不能直接喂 encoder。Paper 把单帧触觉 **temporally repeat $M$ 次**形成短序列兼容 encoder。

2. **Predicted latent 上采样**: World model 输出的 tactile prediction 是低时间分辨率(因 latent compression), 用 **nearest-neighbor interpolation** 上采样匹配 observed feedback 频率, 实现一对一时间对齐。

3. **LTD 编码**: 每个时间步, 用 LTD Encoder 联合编码 current tactile + 对应 predicted future tactile。

4. **Trajectory feature**: robot 过去 $h$ 步的 delta actions 转到 **TCP coordinate frame**(工具中心点坐标系, 跟末端执行器固连), concat 上 delta gripper states。

5. **3-layer MLP**: tactile representation + trajectory feature → single-step refined action $a_r$

### 7.3 训练数据: 教模型学"纠正"

这是 paper 的精彩工程。RLTC 要学"correction", 但正常 trajectory 里 correction 行为占比很小。他们的做法:

1. 对每个 task category, 从 human trajectories 估计 valid tactile distribution (mean + std)
2. 标记 tactile observation 落在 distribution 外 (force 过大或过小) 为 **abnormal state**
3. 提取 **recovery segments**: 系统从 abnormal state 过渡回 valid distribution 的片段 — 这就是 human 演示的 corrective behavior
4. 每个时间步构造 training pair: (current tactile, predicted tactile feature, corrective action $\hat{a}_r$)

Loss:
$$\mathcal{L}_{RLTC} = \|a_r - \hat{a}_r\|_2^2 \tag{14}$$

简单 MSE, 但训练数据是精心筛选的"recovery episodes"。

**Intuition**: RLTC 学的是 inverse dynamics — 给定"偏差状态 + 期望状态(预测)", 输出"纠正动作"。跟 [Learned Jacobian control](https://arxiv.org/abs/2006.08324) 思路类似, 但加了 predictive target 作 reference, 不只是 reactive。

### 7.4 性能

推理时间 (RTX 4090D):
- Slow policy: 230 ms (≈4Hz)
- Slow policy w/ visual gen: 480 ms (太慢)
- **Fast policy (RLTC): 3.5 ms (理论 285Hz, 实际跑 60Hz)**

Perturbation 实验(Fig. 15): 物体被突然向下移动打破接触, RLTC 在一个 generation chunk 内恢复接触。Table III 数据:
- Wipe perturbation: 0.60 (有 RLTC) vs 0.25 (无 RLTC) — 2.4x 提升
- Peel: 0.63 vs 0.20 — 3x
- Cut: 0.60 vs 0.20 — 3x
- Assembly: 0.40 vs 0.20 — 2x

paper 还给了一个力控的数据点: OmniVTA 平均 tangential deformation 0.35 (max 0.72), 而 RDP 的 reactive policy 平均 0.56 (max 1.1) — 后者经常用力过猛**损坏传感器**。RLTC 控制力更精准。

## 八、整体实验结果

### 8.1 三种 setting 对比

Table III 是核心:

| Method | Wipe(O/G/P) | Peel(O/G/P) | Cut(O/G/P) | Assembly(O/G/P) | Grasp(O) | Adj(O/G) |
|---|---|---|---|---|---|---|
| DP | 0.12/0.05/0 | 0.06/0/0 | 0.28/0.10/0 | 0.10/0/0.05 | 0.20 | 0/0 |
| DP+tactile | 0.36/0.28/0 | 0.32/0.20/0.08 | 0.33/0.15/0.13 | 0.30/0.10/0.10 | 0.48 | 0.25/0.15 |
| RDP | 0.50/0.38/0.42 | 0.48/0.36/0.45 | 0.65/0.50/0.43 | 0.60/0.50/0.35 | 0.88 | 0.50/0.50 |
| OmniVTA w/o RLTC | 0.66/0.40/0.25 | 0.40/0.30/0.20 | 0.50/0.50/0.20 | 0.40/0.35/0.20 | 0.70 | 0.40/0.30 |
| **OmniVTA** | **0.80/0.58/0.60** | **0.55/0.48/0.63** | **0.85/0.83/0.60** | **0.60/0.50/0.40** | **0.90** | **0.65/0.65** |

O=Object diversity, G=Generalization (unseen position/tool), P=Perturbation

**几个观察**:

1. **DP baseline 几乎全挂** — vision-only 在 contact-rich task 上彻底不行
2. **加 tactile 就有大提升** (DP→DP+tactile), 说明触觉本身有用
3. **RDP 在 perturbation 上比 OmniVTA w/o RLTC 强** — 因为 RDP 有 reactive controller, 而 OmniVTA w/o RLTC 是开环
4. **但 OmniVTA w/o RLTC 在 object diversity 上反超 RDP** — 说明 world model 的预测信息比单纯 reactive 更强
5. **OmniVTA (full) 全面碾压** — 证明 predictive world model + reflexive control 缺一不可

### 8.2 泛化能力

两种泛化:
- **Position generalization**: 物体放训练没见过的 height
- **Tool generalization**: cut 任务换没见过的 knife

OmniVTA 在 cut 上 tool generalization 几乎不掉点 (G=0.83 vs O=0.85)。**这非常重要** — 说明 policy 依赖 tactile feedback 而非 memorize trajectory, 学到了 transferable contact structure。

### 8.3 关键 ablation (Table VIII)

| Tactile Pred. Length | LTD | Gating | Visual Gen | Wipe | Peel | Avg |
|---|---|---|---|---|---|---|
| 0 | × | × | × | 0.12 | 0.06 | 0.09 |
| 2 | × | × | × | 0.40 | 0.26 | 0.33 |
| 4 | × | × | × | 0.45 | 0.30 | 0.38 |
| 6 | × | × | × | 0.50 | 0.30 | 0.40 |
| 6 | √ | × | × | 0.57 | 0.36 | 0.47 |
| 6 | √ | √ | × | 0.66 | 0.40 | 0.53 |
| 6 | √ | √ | √ | 0.70 | 0.38 | 0.54 |

**几个 takeaways**:

1. **预测长度越长越好** (0→2→4→6), 0 step 时 avg 只 0.09, 几乎失败
2. **LTD Encoder 比 concat 更好** (+7%), differential 编码确实抓到 dynamic relation
3. **Gating 比 concat 更好** (+6%), modality-adaptive fusion 比硬融合强
4. **Visual generation 几乎没用** (+1% 但推理时间翻倍), 砍掉是对的

## 九、跟其他工作的关系

### 9.1 触觉表示学习谱系

- [Sparsh](https://arxiv.org/abs/2410.24090): MAE-based, 用 masked autoencoder 学 tactile representation
- [AnyTouch](https://arxiv.org/abs/2502.12191): contrastive learning 跨 sensor
- [UniT](https://arxiv.org/abs/2502.12191): VQGAN-based compact latent
- [VTV-LLM](https://arxiv.org/abs/2505.22566): 把 tactile 塞进 MLLM

OmniVTA 的差异化: TactileVAE 用 INR 取代 pixel reconstruction, 而且把 tactile representation 直接嵌入 **predictive world model**, 不只是做 recognition。

### 9.2 World model 谱系

- [DreamerV3](https://arxiv.org/abs/2301.04112): latent world model + actor-critic, RL 训练
- [UVA](https://arxiv.org/abs/2503.00200): unified video-action token sequence
- [Stable Video Diffusion](https://arxiv.org/abs/2311.15127): 视频条件生成

OmniVTA 的 two-stream diffusion 设计专门为 tactile 的高频+sparse 特性做了 dynamic-aware loss, 不是简单照搬 video diffusion。

### 9.3 Slow-fast 架构谱系

- [ACT](https://arxiv.org/abs/2304.13705): 纯开环 chunk, 没 fast 层
- [RDP](https://arxiv.org/abs/2503.02881): slow diffusion + fast reactive, 但 fast 层看 visual
- OmniVTA: slow policy 用 world model 预测 tactile, fast 层用 tactile feedback

OmniVTA 把 fast 层从"视觉 reactive"升级为"触觉 reflexive", 更接近生物 spinal reflex 本质。

## 十、我读完之后的几个想法

### 10.1 这套设计的本质

读完最大的 intuition: **contact-rich manipulation 的本质是 contact dynamics 建模, 不是 contact observation**。

之前工作止步于"给 policy 喂 tactile feature", OmniVTA 前进到"先建模 contact dynamics (world model), 再用 dynamics prediction 指导 action planning + 高频 correction"。

四个模块形成闭环:

| 模块 | 解决的问题 | 关键设计 |
|---|---|---|
| TactileVAE | tactile 高维稀疏 → 低维结构化 latent | INR decoder + causal 3D conv |
| VTWM | 缺乏 contact dynamics 预测 | Two-stream diffusion + dynamic-aware loss |
| AFP | vision/touch 该信谁 | LTD encoder + contact-gated fusion |
| RLTC | 开环 chunk 应对不了 rapid change | Predicted-as-target + recovery-segment training |

本质上是 [Wolpert & Flanagan 2001](https://www.sciencedirect.com/science/article/pii/S0960982201004511) 人类 sensorimotor 控制模型在 robot 上的工程化: **feedforward prediction (world model) + sensory feedback comparison (LTD) + reflexive correction (RLTC)**。

### 10.2 几个开放问题

1. **Multi-finger 扩展**: 现在 parallel-jaw, dexterous hand(如 [Allegro](https://www.wonikrobotics.com/research) 或 [Shadow Hand](https://www.shadowrobot.com/))的 multi-contact 场景下, world model 要建模更复杂 contact topology。

2. **Tactile-vision 双向 conditioning**: 现在 visual branch 输出没用进 policy (因速度)。能否用 [consistency model](https://arxiv.org/abs/2303.01469) 加速 visual generation, 让 visual prediction 也进 policy? Table VIII 显示有 +1% 潜力, 但要速度够快才划算。

3. **Prediction uncertainty**: Fig. 16 显示 prediction accuracy < 60% 时 policy 崩溃。能否加 [Deep Ensembles](https://arxiv.org/abs/1612.01474) 或 [Bayesian Dropout](https://arxiv.org/abs/1506.02142) 估 uncertainty, 让 policy 在 prediction 不确定时自动 fallback 到 visual-only?

4. **RL fine-tuning**: paper 用 imitation learning 训完即止。能否用 RL (PPO/SAC) 在 world model 内做 planning, 进一步优化 policy? Dreamer 系列思路。

5. **Cross-sensor policy transfer**: dataset 有 4 种 sensor, 但实验主要在 Xense。TactileVAE 的 cross-sensor generalization (Fig. 11) 已有验证, 但 policy 端的 cross-sensor transfer 还没做实验。

6. **Deformable object**: 现在 object 多是 rigid + semi-rigid。布料、海绵这种高度 deformable 的物体, contact dynamics 会更复杂, INR 还能不能 hold 住是个问题。

7. **Real-time diffusion distillation**: slow policy 230ms 还是慢。用 [LCM](https://arxiv.org/abs/2310.05360) 或 [DMD](https://arxiv.org/abs/2311.18828) 把 diffusion 步数压到 1-4 步, 能否把 slow policy 也推到 30Hz+?

### 10.3 这篇 paper 给后续工作设的 bar

不止要求 dataset 大, 还要求 method:
- **显式建模 contact dynamics** (而非被动 observation)
- **支持 closed-loop tactile control** (而非开环 chunk)
- **Adaptive modality fusion** (而非硬 concat)

后续工作大概率沿两个方向展开:
- 更大规模 dataset (multi-finger, bi-manual, deformable, 更 diverse sensor)
- 更高效 world model (real-time diffusion distillation, latent-space planning, uncertainty-aware)

project page: [https://mrsecant.github.io/OmniVTA](https://mrsecant.github.io/OmniVTA)

总之这篇 paper 把"触觉机器人的天花板"抬高了一截, 至少在 contact-rich manipulation 这个垂直领域里, "world model + reflexive control" 已经成了新的 SOTA paradigm。

---

# OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation 深度解析

## 一、动机与核心直觉

这篇 paper 直击 contact-rich manipulation 的两个痛点。第一个痛点在 **data side**: 现有 visuo-tactile datasets (如 [VLA-Touch](https://arxiv.org/abs/2507.17294), [exUMI](https://arxiv.org/abs/2509.14688), [AgiBot World](https://arxiv.org/abs/2503.06669)) 规模普遍在几千条 trajectory, 任务覆盖窄, sensor 单一。第二个痛点在 **method side**: 主流方法 (如 [3D-ViTac](https://arxiv.org/abs/2410.24091), [RDP](https://arxiv.org/abs/2503.02881)) 把 tactile 当作"auxiliary observation"塞进 policy network, 既没显式建模 contact dynamics, 也没做真正的高频 closed-loop correction。

人类做 contact-rich 任务 (拧瓶盖、削苹果) 靠的是 [Wolpert & Flanagan, Curr Biol 2001](https://www.sciencedirect.com/science/article/pii/S0960982201004511) 提出的 **feedforward predictive model + rapid tactile feedback**。OmniVTA 想在 robot 上复刻这套机制: 用 world model 做 feedforward, 用 60Hz reflexive controller 做 feedback。

## 二、OmniViTac Dataset: 数据集设计

### 2.1 规模与多样性

| 维度 | OmniViTac | 之前最大 (AgiBot World*) |
|---|---|---|
| Trajectories | 21,879 | 5,337 |
| Tasks | 86 | 7 |
| Objects | 126 | – |
| Tactile sensors | 4 (Xense/GelSight Mini/DM-Tac/Tac3D) | 1 (Xense) |
| Frequency | 30~60 Hz | 30~60 Hz |

### 2.2 Dual-Embodiment 采集系统

这是 paper 一个巧思。他们用了两套硬件:

- **UFACTORY xArm-7** (7-DoF robot arm): kinesthetic teaching (gravity compensation 下手动引导) + [GELLO teleoperation](https://arxiv.org/abs/2402.11236) — 用于 robot-aligned 数据
- **TacUMI**: 基于 [FastUMI](https://arxiv.org/abs/2402.10329) 的手持设备, 配 RealSense T265 (200Hz 6-DoF pose 估计) — 用于大规模高效采集

两套硬件用 **identical parallel-jaw gripper** + 模块化 fingertip tactile sensor, 把 embodiment gap 压到最低。Tracking drift >8mm 的轨迹直接丢掉。

### 2.3 六类物理 grounded 的 interaction patterns

paper 摒弃了"按视觉 kinematics 分类"的做法, 改为按 **dominant tactile features + contact mechanics** 分类:

| Pattern | 主导 tactile 信号 | Active Ratio | Contact Area 分布 |
|---|---|---|---|
| Assembly | multi-directional force, tight tolerance | 中 | 0-10% (precision) |
| Cutting | normal force magnitude, sudden drop | 0.27 (低) | 70-90% (full patch) |
| Adjustment | torsional + shear (slip) | 0.67 (高) | 0-10% |
| Peeling | shear + normal coupling | 0.41 | 70-90% |
| Wiping | normal pressure + planar shear | 0.49 | 70-90% |
| Grasping | normal force, fragile handling | 中 | 0-10% |

t-SNE 可视化 (Fig. 4f) 显示这六类在 tactile latent space 中是 **physically separable** 的 — Wiping 和 Peeling 簇靠近 (共享 frictional mechanics), Assembly 是 distinct manifold。这说明 dataset 确实抓到了 contact mechanics 的结构, 而非噪声。

**关键 insight**: tactile 信号有两个 structural properties — **spatial locality** (接触前几乎全零, 接触后局部激活) 和 **contact-driven dynamics** (信号变化由物理接触驱动)。这两个 property 直接决定了后续模块的架构选择。

## 三、OmniVTA 架构总览

```
                ┌─────────────────────────────────────────┐
                │   Slow Policy (15Hz)                     │
                │   ┌─────────────┐    ┌────────────────┐  │
   Vision ─────►│   │   VTWM       │───►│  Adaptive      │  │
   Tactile ────►│   │  (predict    │    │  Visuo-Tactile │──┼──► Action Chunk
   Proprio ────►│   │   future)    │    │  Fusion Policy │  │
                │   └─────────────┘    └────────────────┘  │
                └─────────────────────────────────────────┘
                                          ↓ weighted sum
                ┌─────────────────────────────────────────┐
                │   Fast Policy (60Hz)                    │
                │   Reflexive Latent Tactile Controller   │──► Refined Action
                │   (predicted vs observed tactile diff)  │
                └─────────────────────────────────────────┘
```

这个 slow-fast 分层借鉴了 [Reflexive Evasion Robot](https://arxiv.org/abs/2502.1746) 这种 biologically inspired control 思路: slow 做 long-horizon planning, fast 做 high-frequency reflex。最终 action 是两者的 **weighted summation**, controller 贡献由预定义 coefficient 缩放。

## 四、TactileVAE: 隐式神经表示的 tactile encoder

### 4.1 输入表示选择

光学 tactile sensor (GelSight, DIGIT, Xense) 输出的 RGB image 分辨率很高 (700×400), 直接当 image 编码太重。paper 改用 **3D marker displacement** tensor: $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$

- $H$, $W$: marker 沿 y 轴和 x 轴的网格数 (例如 Xense 是 $35 \times 20$)
- channel 维度 3: marker 在 x, y, z 三个方向的位移

这把高分辨率图像压成稀疏 displacement field, 既保留接触几何, 又大幅降维, 是后续 60Hz 高频推理的关键前提。

### 4.2 Spatio-temporal Encoder

结构: **projection-in (causal 3D conv)** → $M$ 个 downsampling modules → **projection-out (causal 3D conv)**

输出 latent: $\mathbf{z}_t \in \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times C}$, 其中 $s = 2^M$ 是 spatial downsampling factor。

**Causal 3D convolution** 是关键: 让 time $t$ 的 latent 只依赖 $\le t$ 的观察, 保证训练-部署一致性, 避免"未来信息泄漏"。这与 [WaveNet](https://arxiv.org/abs/1609.03499) 和 video diffusion 中的 causal mask 同源。

### 4.3 Implicit Neural Representation Decoder

这是 TactileVAE 最有想法的部分。常规 VAE 重建 pixel grid, 但 marker displacement 本质是 **elastomer surface 的连续 deformation field**, 用 INR 建模更自然:

$$\mathbf{d}(\mathbf{x}) = \mathcal{D}_\theta\big(\gamma(\mathbf{x}), \Phi(\mathbf{z}_t, \mathbf{x})\big) \tag{1}$$

各变量含义:
- $\mathbf{x} \in \mathbb{R}^2$: spatial coordinate (marker 平面上的查询点)
- $\mathbf{z}_t$: encoder 输出的 latent feature map
- $\gamma(\cdot)$: positional encoding (类似 [NeRF](https://arxiv.org/abs/2003.08934) 的高频 encoding, 让 MLP 能学高频细节)
- $\Phi(\mathbf{z}_t, \mathbf{x})$: 从 latent feature map 在 $\mathbf{x}$ 处做 spatial interpolation 取 local feature
- $\mathcal{D}_\theta$: MLP decoder, 输出 $\mathbf{d}(\mathbf{x}) \in \mathbb{R}^3$ (3D displacement)

训练时 uniformly sample 查询点, ground-truth $\hat{\mathbf{d}}(\mathbf{x})$ 从原始 3D displacement 插值得到。Loss:

$$\mathcal{L}_{\text{TacVAE}} = \|\mathbf{d}(\mathbf{x}) - \hat{\mathbf{d}}(\mathbf{x})\|_2^2 + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} \tag{2}$$

$\lambda_{\text{KL}} = 10^{-6}$ — 极小的 KL weight, 强调 reconstruction, 这与 [β-VAE](https://arxiv.org/abs/1611.00770) 学 disentangled 时的策略相反, 因为这里目标是保真而非解耦。

### 4.4 INR 的优势

Table V 的 ablation 验证 INR decoder 必要性:
- w/o implicit decoder: 0.126 (GelSight-Mini)
- w/ position embedding: 0.102
- w/o spatial feature map (single token): 0.107
- **w/ implicit decoder (full): 0.047**

把 latent 表示为 **spatial feature map** 而非 single global token 也重要 — 这保留了 spatial locality, 对下游 contact localization 关键。

## 五、Visuo-Tactile World Model (VTWM)

### 5.1 Two-stream diffusion transformer

VTWM 是整个 paper 的核心, 采用 two-stream 架构同时预测 vision 和 tactile 未来。两条 branch 都是 [spatial-temporal diffusion transformer](https://arxiv.org/abs/2212.09748) (类似 DiT 但扩展到时空)。

输入: 过去 $c$ 帧 observation 作 conditioning, 迭代 denoise 生成 $K$ 帧 future latents。

Diffusion 训练目标:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{\mathbf{z}_o, \epsilon, t}\left[\sum_{i=1}^{K} (1 - m_i) \odot \|\epsilon_i - \epsilon_\theta(\mathbf{z}_o, t)_i\|_2^2\right] \tag{3}$$

各变量:
- $\mathbf{z}_o = \{\mathbf{z}_o^1, \ldots, \mathbf{z}_o^K\}$: 一段 observation latents 序列, 包含 tactile latent $\mathbf{z}_t$ 和 visual latent $\mathbf{z}_v$
- $\epsilon$: 加的 Gaussian noise
- $t$: diffusion timestep
- $\epsilon_\theta(\mathbf{z}_o, t)$: 神经网络预测的 noise
- $m_i$: temporal mask, 第 $i$ 帧是否作为 condition (已知过去) 还是要预测 (未来)
- $(1 - m_i)$: 只在需要预测的帧上计算 loss

Modality encoding:
- Visual branch: [SD-VAE](https://arxiv.org/abs/2112.10752) 编 image 到 latent
- Tactile branch: 用预训练的 TactileVAE 编码

### 5.2 Multi-modal Observation Conditioner

这是把 vision/tactile/action 融合成 shared condition 的模块。每个 modality 先独立做 feature extraction + temporal aggregation, 再在 shared linear projection space 融合, 得到 fixed-dim conditioning vector, 注入两条 branch。

**关键设计**: action 表示为 **2D image-plane projection of 3D end-effector position**, 而非 3D absolute/relative action。Table VII 显示 2D action 在 unseen position 上 L2=0.042, cos=0.91, 远优于 3D absolute (L2=0.075) 和 3D relative (L2=0.056)。

**Intuition**: action condition 主要传达"motion intent" (图像上要往哪儿走), 2D 表示与 visual observation 自然对齐, 避免 3D 坐标系 mismatch 问题。这跟 [RT-2](https://arxiv.org/abs/2307.15818) 把 action token 化进 VLM vocab 的思路同源 — 让 action 和 observation 在同一表示空间。

### 5.3 Dynamic-aware Weighted Loss

这是 paper 的另一个亮点, 直接针对 tactile 信号的 **contact-driven dynamics** 性质。

Standard diffusion loss 对所有 spatial location 一视同仁, 但 tactile 信号在空间上高度 sparse — 接触前几乎全零, 接触后局部爆发。如果等权训练, 模型会偏向"全零"的简单解。

Paper 设计了两个 weight map:

**Dynamic weight** (基于时间差分, 捕捉高频变化):
$$w_{\text{dyn}}^i = \text{resize}\left(\text{clip}_{[0,1]}\left(\|X_{i+1} - X_i\|_2\right)\right) \tag{4}$$

**Amplitude weight** (基于响应幅度, 捕捉接触强度):
$$w_{\text{amp}}^i = \text{resize}\left(\text{clip}_{[0,1]}\left(\|X_i\|_2\right)\right) \tag{6}$$

加权 diffusion loss:

$$\mathcal{L}_{\text{dyn}} = \mathbb{E}_{\mathbf{z}_o, \epsilon, t}\left[\sum_{i=2}^{K} w_{\text{dyn}}^i \odot (1-m_i) \odot \|\epsilon_i - \epsilon_\theta(\mathbf{z}_o, t)_i\|_2^2\right] \tag{5}$$

$$\mathcal{L}_{\text{amp}} = \mathbb{E}_{\mathbf{z}_o, \epsilon, t}\left[\sum_{i=2}^{K} w_{\text{amp}}^i \odot (1-m_i) \odot \|\epsilon_i - \epsilon_\theta(\mathbf{z}_o^i, t)\|_2^2\right] \tag{7}$$

注意 (7) 式用 $\mathbf{z}_o^i$ (单帧), 而 (5) 用 $\mathbf{z}_o$ (整段), 这是因为 amp weight 只依赖当前帧 magnitude。

总 loss:
$$\mathcal{L}_{VTWM} = \mathcal{L}_{\text{diffusion}} + \lambda_1 \mathcal{L}_{\text{dyn}} + \lambda_2 \mathcal{L}_{\text{amp}} \tag{8}$$

实验中 $\lambda_1 = \lambda_2 = 1.0$。

**Intuition**: 这等价于让模型在"有接触"和"接触变化快"的时空位置上更用力学。这与 [focal loss](https://arxiv.org/abs/1708.02002) 处理类别不平衡的思路类似 — 简单的 background 区域 (无接触) 不需要模型花太多 capacity 去拟合。

### 5.4 VTWM 实验结果

Table VI 对比 4 个 baseline:
- [UVA](https://arxiv.org/abs/2503.00200): unified token sequence, single generative model
- exUMI: latent diffusion conditioned on visual+action
- [KineDex](https://arxiv.org/abs/2505.01974): joint action+force diffusion
- [ForceMimic](https://arxiv.org/abs/2502.09909): 3D observation conditioned action+force diffusion

OmniVTA 在所有 6 个 task 上 L2 最低、cosine 最高:
- Wipe: L2_avg=0.059 vs 次优 KineDex 0.082
- Cut: L2_avg=0.050 vs 次优 ForceMimic 0.090
- Assembly: L2_avg=0.025, cos_avg=0.85 (大幅领先)
- Grasp: L2_avg=0.010, cos_avg=0.68

Ablation (Table VII) 还显示:
- Joint visual-tactile generation 比单 stream tactile 预测更好 (L2: 0.035 vs 0.041), 说明 visual branch 提供了 complementary global dynamic cue
- Dynamic weighting 进一步降到 0.035, cos 升到 0.93

## 六、Adaptive Visuo-Tactile Fusion Policy (AFP)

### 6.1 Latent Tactile Differential (LTD) Encoder

这是 paper 的核心创新之一, 用来编码 "current + predicted future" tactile 的关系。

**Motivation**: tactile 有 spatial locality — 接触前没用, 接触时才有信息。如果像之前工作一样把 historical tactile 直接和 visual feature concat, 模型无法显式捕捉"potential contact dynamics"。

LTD Encoder 构造:
- $\mathbf{f}_t^c$: current tactile feature, 通过 2D conv + max pool 空间聚合得到 global 表示
- $\mathbf{f}_t^p$: predicted multi-frame tactile feature, 先 per-frame spatial 聚合, 再用 1D conv + max pool 做时间聚合

最终 tactile 表示:
$$\mathbf{f}_t = \text{concat}(\mathbf{f}_t^c, \mathbf{f}_t^p, \mathbf{f}_t^p - \mathbf{f}_t^c) \tag{9}$$

**关键: 第三个分量 $\mathbf{f}_t^p - \mathbf{f}_t^c$** 是 "predicted - current" 的差分, 直接 highlight "未来要发生什么 contact 事件"。这跟人类 motor cortex 的 feedforward prediction 机制 ([Wolpert 2001](https://www.sciencedirect.com/science/article/pii/S0960982201004511)) 对应 — 大脑用 efference copy 生成预测, 再和实际 sensory feedback 比对。

### 6.2 Adaptive Visuo-Tactile Fusion (gating)

这部分受 [FoAR](https://arxiv.org/abs/2501.02505) 启发, 用 predicted contact probability 调制 modality 权重。

**Contact probability 预测**:
- 输入: $\mathbf{f}_t$ (LTD encoded tactile)
- 网络: MLP + sigmoid
- 监督: 自动生成的 contact label (threshold tactile deformation magnitude), BCE loss $\mathcal{L}_{\text{bce}}$

**Gating network**:
- 输入: concat(contact logit, $\mathbf{f}_t$)
- 网络: 2 个 FC layer
- 输出: $W_t$, $W_v$, 满足 $W_t + W_v = 1$ (per-channel normalization)

注意 gating network **只接收 tactile 信息, 不接收 visual** — 因为 tactile $\mathbf{f}_t$ 已经通过 world model 编码了 future tactile dynamics, 无需视觉输入也能判断"接下来要不要靠 touch", 还能减模型复杂度。

融合公式:
$$\mathbf{f}_{vt} = \text{concat}\left(W_v \odot \mathbf{f}_v, W_t \odot \tilde{\mathbf{f}}_t\right) \tag{10}$$

- $\mathbf{f}_v$: ResNet-18 提取的 visual feature
- $\tilde{\mathbf{f}}_t$: tactile feature 经 linear projection 后维度匹配 visual
- $\odot$: per-channel modulation

**关键设计**: 只用 current + historical visual, **不用 future visual prediction**。原因有二:
1. Current image 已提供足够 global context 给 action planning
2. Predicted tactile 已经 capture 了 potential contact dynamics
3. 加 visual generation branch 推理时间从 230ms 暴增到 480ms (Table IX), 性价比差

Table VIII 显示加 visual gen 只把 wipe/peel avg 从 0.53 提到 0.54, 几乎无收益。

### 6.3 Visuo-Tactile Diffusion Policy

Action 生成用 [DDPM](https://arxiv.org/abs/2006.11239), 输出 action chunk $A_c = (a_c^1, \ldots, a_c^H)$, $H$ coarse actions。

Reverse diffusion update:
$$A_{c,t-1} = \alpha_t A_{c,t} - \gamma_k \epsilon_\theta(A_{c,t}, t, \mathbf{f}_c) + \sigma_t \mathcal{N}(0, I) \tag{11}$$

各变量:
- $A_{c,t}$: timestep $t$ 的 noisy action chunk
- $\epsilon_\theta$: noise predictor (用 [FiLM](https://arxiv.org/abs/1709.07871) 注入 condition)
- $\mathbf{f}_c = \text{concat}(\mathbf{f}_{vt}, \mathbf{s})$: fused visuo-tactile feature + robot proprioception
- $\alpha_t, \gamma_t, \sigma_t$: scheduler 系数
- $\mathcal{N}(0, I)$: Gaussian noise

这个 update 公式类似 [Stochastic Langevin Dynamics](https://arxiv.org/abs/1104.2560), noise predictor 隐式参数化 score function。

训练 loss (DDPM objective):
$$\mathcal{L}_{\text{act}} = \mathbb{E}_{t, A_{c,0}, \epsilon_t}\left[\|\epsilon_t - \epsilon_\theta(\bar{\alpha}_t A_{c,0} + \bar{\beta}_t \epsilon_t, t, \mathbf{f}_c)\|_2^2\right] \tag{12}$$

- $\bar{\alpha}_t, \bar{\beta}_t$: 累积 noise schedule 系数
- $A_{c,0}$: clean action chunk (ground truth)

Policy 总 loss:
$$\mathcal{L}_{AFP} = \mathcal{L}_{\text{act}} + \lambda_{ct} \mathcal{L}_{\text{bce}} \tag{13}$$

$\lambda_{ct} = 0.2$ — contact prediction 作辅助任务, 帮 gating 学得更好。

### 6.4 Gating 行为可视化分析

Fig. 14 的可视化非常有意思: 随着任务执行, contact probability (purple) 和 tactile weight (red) 几乎同步上升, visual weight (blue) 同步下降。这说明 gating 学到了"接触前靠视觉, 接触时靠触觉"的 adaptive 策略。

Fig. 16 更关键: 当 tactile prediction accuracy 降到 60%, gating 网络无法正确估计 contact probability, modality weighting 失衡, action planning 性能随之崩溃。这证明 **accurate tactile prediction 是整个 policy 的基石**。

## 七、Reflexive Latent Tactile Controller (RLTC)

### 7.1 设计思路

Slow policy 输出 action chunk 是 open-loop 执行, 在 contact-rich 任务中无法应对 rapid contact change (slip, misalignment, disturbance)。RLTC 提供 60Hz 高频 closed-loop correction, 类似人类 spinal reflex arc ([Augurelle 2003](https://journals.physiology.org/doi/10.1152/jn.00275.2002))。

### 7.2 输入对齐的工程细节

这部分有几个 temporal alignment 巧思:

1. **Single-frame tactile → 短序列**: TactileVAE 沿时间维度压缩了 $M$ 倍, single-frame 输入无法直接喂入 encoder。Paper 把 single-frame tactile observation **temporally repeat $M$ 次**, 形成短序列兼容 encoder。

2. **Predicted latent 上采样**: World model 输出的 tactile prediction 在低 temporal resolution (因为 latent compression), 用 **nearest-neighbor interpolation** 上采样匹配 observed tactile feedback 频率, 实现一对一时间对齐。

3. **LTD 编码**: 对每个时间步, 用 LTD Encoder 联合编码 current tactile feature + 对应的 predicted future tactile feature, 产 tactile representation。

4. **Trajectory feature**: robot 过去 $h$ 步的 delta actions 转到 **TCP coordinate frame** (工具中心点坐标系), concat 上对应 delta gripper states。

5. **3-layer MLP**: tactile representation + trajectory feature → single-step refined action $a_r$

### 7.3 训练数据构造

这是 paper 的精彩部分。RLTC 需要学"correction", 但正常 trajectory 里 correction 行为占比小。

构造方法:
1. 对每个 task category, 从 human trajectories 估计 valid tactile distribution (mean + std)
2. 标记 tactile observation 落在 distribution 外 (force 过大或过小) 的为 **abnormal state**
3. 提取 **recovery segments**: 系统从 abnormal state 过渡回 valid distribution 的片段, 这就是 human 演示的 corrective behavior
4. 每个时间步构造 training pair: (current tactile, predicted tactile feature, corrective action $\hat{a}_r$)

Loss:
$$\mathcal{L}_{RLTC} = \|a_r - \hat{a}_r\|_2^2 \tag{14}$$

简单的 MSE, 但训练数据是精心筛选的"recovery episodes"。

**Intuition**: 这等价于让 RLTC 学一个 inverse dynamics: 给定"偏差状态 + 期望状态(预测)", 输出"纠正动作"。和 [Learned Jacobian control](https://arxiv.org/abs/2006.08324) 思路类似, 但加了 predictive target 作为 reference。

### 7.4 性能与推理速度

Table IX 推理时间 (RTX 4090D):
- Slow policy: 230 ms (≈ 4 Hz)
- Slow policy w/ visual gen: 480 ms (太慢)
- **Fast policy (RLTC): 3.5 ms (≈ 285 Hz, 实际跑 60Hz)**

整个 system 频率瓶颈在 slow policy, 但因为 action chunk 是开环执行 + RLTC 高频修正, 实际控制频率能维持 60Hz。

Perturbation 实验 (Fig. 15): 当物体被突然向下移动打破接触, RLTC 能在一个 generation chunk 内恢复接触。Table III 显示有 RLTC 的 OmniVTA 在 perturbation setting 下:
- Wipe: 0.60 vs 无 RLTC 的 0.25
- Peel: 0.63 vs 0.20
- Cut: 0.60 vs 0.20
- Assembly: 0.40 vs 0.20

RLTC 带来 2-3 倍的 robustness 提升。

paper 还给了一个数据点说明 RLTC 控制接触力的效果: 平均 tangential deformation 0.35 (max 0.72), 而 RDP 的 reactive policy 平均 0.56 (max 1.1) — 后者经常用力过猛损坏 sensor。

## 八、主要实验结果分析

### 8.1 整体性能 (Table III)

| Method | Wipe(O/G/P) | Peel(O/G/P) | Cut(O/G/P) | Assembly(O/G/P) | Grasp(O) | Adj(O/G) |
|---|---|---|---|---|---|---|
| DP | 0.12/0.05/0 | 0.06/0/0 | 0.28/0.10/0 | 0.10/0/0.05 | 0.20 | 0/0 |
| DP+tactile | 0.36/0.28/0 | 0.32/0.20/0.08 | 0.33/0.15/0.13 | 0.30/0.10/0.10 | 0.48 | 0.25/0.15 |
| RDP | 0.50/0.38/0.42 | 0.48/0.36/0.45 | 0.65/0.50/0.43 | 0.60/0.50/0.35 | 0.88 | 0.50/0.50 |
| OmniVTA w/o RLTC | 0.66/0.40/0.25 | 0.40/0.30/0.20 | 0.50/0.50/0.20 | 0.40/0.35/0.20 | 0.70 | 0.40/0.30 |
| **OmniVTA** | **0.80/0.58/0.60** | **0.55/0.48/0.63** | **0.85/0.83/0.60** | **0.60/0.50/0.40** | **0.90** | **0.65/0.65** |

**观察**:
- DP baseline 在所有 contact-rich task 上几乎全失败, 证明 vision-only 严重不足
- DP+tactile 比 DP 提升明显, 说明 tactile 信息本身有用
- RDP (slow-fast 已有工作) 在 perturbation 上比 OmniVTA w/o RLTC 强 (RDP 的 reactive controller 起作用), 但 OmniVTA w/o RLTC 在 object diversity 上反超 RDP, 说明 world model 提供的预测信息比单纯 reactive 更强
- OmniVTA (full) 全面碾压, 证明 **predictive world model + reflexive control** 缺一不可

### 8.2 Generalization 能力

两类泛化测试:
- **Position generalization**: 物体放在训练时未见过的 height
- **Tool generalization**: cut 任务换未见过的 knife

OmniVTA 在 cut task 上 tool generalization 几乎无掉点 (G=0.83 vs O=0.85), 说明 policy 依赖 tactile feedback 而非 memorize trajectory。这是 paper 一个强 claim — 学到 transferable contact structure。

### 8.3 关键 ablation 汇总

Table VIII 是 AFP 的 ablation, 揭示几个关键事实:

| Tactile Pred. Length | LTD | Gating | Visual Gen | Wipe | Peel | Avg |
|---|---|---|---|---|---|---|
| 0 | × | × | × | 0.12 | 0.06 | 0.09 |
| 2 | × | × | × | 0.40 | 0.26 | 0.33 |
| 4 | × | × | × | 0.45 | 0.30 | 0.38 |
| 6 | × | × | × | 0.50 | 0.30 | 0.40 |
| 6 | √ | × | × | 0.57 | 0.36 | 0.47 |
| 6 | √ | √ | × | 0.66 | 0.40 | 0.53 |
| 6 | √ | √ | √ | 0.70 | 0.38 | 0.54 |

**几个关键 takeaways**:

1. **Predicted tactile 信息有用**: 0 step (只用 current tactile) avg 只有 0.09, 几乎失败。预测长度 6 > 4 > 2, 长预测更好。

2. **LTD Encoder 比 concat 更好**: 同样 6-step prediction, 加 LTD 从 0.40 升到 0.47 (+7%), 证明 differential 编码确实捕捉到 dynamic relation。

3. **Gating 机制有用**: 加 gating 再升 6% (0.47 → 0.53), modality-adaptive fusion 比硬 concat 好。

4. **Visual generation 几乎无用**: 0.53 → 0.54 (+1%), 但推理时间翻倍 (230ms → 480ms), 性价比极低。这是 paper 设计 slow policy 时砍掉 visual generation branch 的依据。

## 九、关联工作与延伸思考

### 9.1 与其他 visuo-tactile 工作的关系

- [Sparsh](https://arxiv.org/abs/2410.24090): MAE-based tactile representation, OmniVTA 的 TactileVAE 用 INR 取代 pixel reconstruction
- [UniT](https://arxiv.org/abs/2502.12191): VQGAN-based compact tactile latent
- [VTV-LLM](https://arxiv.org/abs/2505.22566): visuo-tactile video understanding, 用 MLLM 对齐

OmniVTA 的差异化在于把 tactile representation 直接嵌入 **predictive world model** 而非仅做 recognition/classification。

### 9.2 与其他 world model 工作

- [UVA](https://arxiv.org/abs/2503.00200): unified video-action model, 把 video 和 action 都当 token 序列生成
- [DreamerV3](https://arxiv.org/abs/2301.04112): latent world model + actor-critic
- OmniVTA 的 two-stream 设计更像 [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) 的条件生成架构, 但专门为 tactile 的高频 + sparse 特性设计了 dynamic-aware loss

### 9.3 Slow-fast 架构的谱系

- [ACT](https://arxiv.org/abs/2304.13705) (Action Chunking Transformer): 纯开环 chunk, 无 fast 层
- [RDP](https://arxiv.org/abs/2503.02881): slow diffusion + fast reactive controller, 但 fast 层只看 visual
- OmniVTA: slow policy 用 world model 预测 tactile, fast 层用 tactile feedback 做高频 correction

OmniVTA 在这个谱系上把 fast 层从"视觉 reactive"升级为"触觉 reflexive", 更接近生物 spinal reflex 的本质。

### 9.4 一些可能的延伸方向

读 paper 时几个想到的开放问题:

1. **Multi-finger extension**: 当前只 parallel-jaw, dexterous hand (如 [Allegro](https://www.wonikrobotics.com/research) 或 [Shadow Hand](https://www.shadowrobot.com/)) 的 multi-contact 场景下, world model 需要建模更复杂的 contact topology。

2. **Tactile-vision bidirectional conditioning**: 当前 visual branch 输出没用于 policy (因速度), 但能否用 [consistency model](https://arxiv.org/abs/2303.01469) 加速 visual generation, 让 visual prediction 也进 policy?

3. **Tactile prediction 误差的 graceful degradation**: Fig. 16 显示 prediction accuracy < 60% 时 policy 崩溃。能否加 uncertainty estimation (如 [Deep Ensembles](https://arxiv.org/abs/1612.01474) 或 [Bayesian Dropout](https://arxiv.org/abs/1506.02142)) 让 policy 在 prediction 不确定时自动 fallback 到 visual-only?

4. **Closed-loop RL fine-tuning**: paper 用 imitation learning 训完即止。能否用 RL (PPO/SAC) 在 world model 内做 planning, 进一步优化 policy? Dreamer 系列思路。

5. **Cross-sensor transfer**: dataset 有 4 种 sensor, 但实验主要在 Xense 上。TactileVAE 的 cross-sensor generalization (Fig. 11) 已有初步验证, 但 policy 端的 cross-sensor transfer 还未实验。

## 十、总结与 intuition

读完这篇 paper, 我的核心 intuition 是: **contact-rich manipulation 的本质是 contact dynamics 建模, 而非 contact observation**。之前的 visuo-tactile 工作止步于"给 policy 喂 tactile feature", OmniVTA 前进到"先建模 contact dynamics (world model), 再用 dynamics prediction 指导 action planning + 高频 correction"。

四个模块各自的 design rationale 形成闭环:

| 模块 | 解决的问题 | 关键设计 |
|---|---|---|
| TactileVAE | tactile 高维稀疏 → 低维结构化 latent | INR decoder + causal 3D conv |
| VTWM | 缺乏 contact dynamics 预测 | Two-stream diffusion + dynamic-aware loss |
| AFP | vision/touch 该信谁 | LTD encoder + contact-gated fusion |
| RLTC | 开环 chunk 无法应对 rapid contact change | Predicted-as-target + recovery-segment training |

这套设计本质上是在 robot 上复刻 [Wolpert & Flanagan 2001](https://www.sciencedirect.com/science/article/pii/S0960982201004511) 提出的人类 sensorimotor 控制模型: **feedforward prediction (world model) + sensory feedback comparison (LTD) + reflexive correction (RLTC)**。

paper project page: [https://mrsecant.github.io/OmniVTA](https://mrsecant.github.io/OmniVTA)

这个工作给 contact-rich manipulation 设了一个新的 bar — 不仅要求 dataset 大, 还要求 method 显式建模 contact dynamics 并支持 closed-loop。后续工作大概率会沿着两个方向展开: 更大规模 dataset (覆盖 multi-finger, bi-manual, deformable) + 更高效的 world model (real-time diffusion distillation, latent-space planning)。
