---
source_pdf: RESWORLD TEMPORAL RESIDUAL WORLD MODEL FOR END-TO-END AUTONOMOUS DRIVING.pdf
paper_sha256: 5ee08c758f9ffd52872fb24568aed79ed96742dcd9cd4ac42a75ca861401a9fd
processed_at: '2026-08-11T23:04:40-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ResWorld

## 一句话版本

在 BEV 特征图上做"帧差法"——把相邻两帧相减, 静止的背景减没了, 动的车、人就以残差形式凸显出来。然后让 world model 只预测这些"动的东西"未来去哪, 不浪费算力去预测那些不动的路面、建筑。

---

## 用类比建立 intuition

你拍过监控视频吧? 早期安防里有个老技术叫 **background subtraction**: 把当前帧减去背景帧, 静止的墙、地减完是黑的, 但走动的人会留下一团亮亮的残影。这个残影就是"动的东西"。

ResWorld 在 BEV feature 上干的就是这个事, 但有三个关键升级:

### 1. 坐标对齐: 让"减法"有物理意义

不能直接拿 $t$ 和 $t-1$ 两帧 BEV 相减, 因为 ego car 自己也在动——$t-1$ 时车在 A 位置, $t$ 时车开到 B 位置了, 两帧的 BEV 坐标系不一样, 减出来全是噪声。

所以先做一件事: **把所有历史帧都 warp 到当前时刻 $t$ 的 ego 坐标系**。相当于"假装车没动, 看世界相对车怎么变"。

这样减法才合理: 同一 spatial location 上, $t-1$ 时刻有辆车, $t$ 时刻车开走了, feature 大变 → residual 亮; 路面、绿化带 $t-1$ 和 $t$ 都一样 → residual 接近零。

这跟 [BEVDet4D](https://arxiv.org/abs/2211.17054) 的 temporal fusion 是一脉相承的, 但 BEVDet4D 是相加融合, ResWorld 是相减提差异。

### 2. 用 sparse token 而非 dense pixel

不在整个 $H \times W$ BEV 图上做减法, 而是:
- 用一个 spatial attention map 从 BEV 里抽出 $N_s$ 个"关键位置"的 query (类似 [TokenLearner](https://arxiv.org/abs/2106.11210))
- 每个 timestamp 都用**同一个 attention map** 抽 query (保证空间对齐)
- 在这 $N_s$ 个 query 上做减法

好处: 计算轻, 信号集中, 而且 attention map 本身就 highlight 了 dynamic region——因为它是从融合多帧的 $\mathbf{B}_{fuse}$ 里学出来的, 融合帧对"哪里在动"最敏感。

### 3. World model 只管动的东西

传统 world model ([OccWorld](https://arxiv.org/abs/2311.16010), [Drive-WM](https://arxiv.org/abs/2401.16163)) 要预测整个未来场景。但场景里 80% 是静止的——路面、建筑、树、护栏。这些未来还在原地, 预测它们纯属浪费 capacity。

ResWorld 的 trick: **未来 BEV 的静态部分 = 当前 BEV**, 不用预测。World model 只需要预测"动的东西"未来去哪, 把这个 delta 叠加到当前 BEV 上, 就是 future BEV。

公式上就是: $\mathbf{B}_{future} = \mathrm{TokenFuser}(\hat{\mathbf{R}}, \mathbf{B}_{fuse}) + \mathbf{B}_{fuse}$。加号右边那个 $\mathbf{B}_{fuse}$ 就是"静态不变"的假设, 左边那项是"动态增量"。

---

## FGTR: 让 ego 在脑子里"预演"一遍

之前 world model 有个尴尬: 你费劲预测出 future BEV, 结果 planning 模块主要还是看 current BEV, future BEV 只是当个 proxy task 间接让 encoder 学得好点。future BEV 没直接参与规划。

ResWorld 不满足这个。它让 ego car 拿着 prior trajectory, 去 future BEV 上"踩点":
- 第 1 个 waypoint query 去 future BEV 上"看"一下, prior trajectory 说 1 秒后 ego 在 $(x_1, y_1)$, 这个位置周围有没有障碍物?
- 第 2 个 waypoint query 看 2 秒后位置 $(x_2, y_2)$ 周围有没有东西?
- ...

然后用 [Deformable Attention](https://arxiv.org/abs/2010.04159) 把这些信息收集回来, refine 出 final trajectory。

相当于 **ego 在 mental simulation 里开了一遍 prior trajectory, 检查会不会撞, 提前修正**。这跟人类老司机"过路口前先扫一眼"的直觉类似。

---

## 反直觉: 不监督 future BEV 反而更好

一般 world model 都会拿真实的未来帧去监督预测的未来 BEV, 否则怎么知道预测得对不对?

ResWorld 偏不。只用 trajectory L1 loss, 不监督 future BEV。

原因: 如果用 $t+1$ 时刻的真实数据监督, future BEV 就被"钉死"成 $t+1$ 那一帧的 snapshot。但 FGTR 需要的是 $t+1, t+2, \dots, t+N_t$ 整个时间段的动态信息(每个 waypoint query 要 query 不同时刻)。一监督, 多时刻信息就丢了。

Ablation 实验很硬: 无监督的 TR-World L2=0.59 CR=0.17, 加 future supervision 反而 L2=0.61 CR=0.21。

这跟 latent world model ([DreamerV3](https://arxiv.org/abs/2301.04104), [LAW](https://arxiv.org/abs/2406.08481)) 的哲学一致——在 latent/abstract space 预测未来, 不强求 pixel-level reconstruction。

---

## 防 collapse: 用 task demand 倒逼 representation

World model 有个臭毛病叫 **collapse**: 既然 planning 主要靠 current BEV, future BEV 没人查它, 它就偷懒输出个常数, 反正 loss 也不大。这跟 VAE 的 posterior collapse、GAN 的 mode collapse 是同源问题。

FGTR 的妙处: 它**强制 planning 必须用 future BEV** 做 refinement。future BEV 敢 collapse 成常数, refinement 就失效, collision rate 爆表, loss 爆增, gradient 会把它拉回来。

不是靠显式的 diversity regularizer, 而是靠下游 task 的"需求"倒逼 representation 必须有信息。这个思路挺 elegant 的, 我觉得可以推广到 VAE / diffusion training。

---

## 结果: 避障能力真的变强了

nuScenes 上 collision rate 比 baseline ([SSR](https://arxiv.org/abs/2501.13355)) 降了 40-50%, 这是最直接的 evidence——temporal residual 确实转化为避障能力。

NAVSIM (closed-loop) 上 TTC (Time-to-Collision) 指标 98.9, 遥遥领先 DiffusionDrive 的 94.7, Hydra-MDP 的 94.6。TTC 衡量"还能多久才撞", 越大越安全, ResWorld 这项指标特别强。

说明: temporal residual → 更准的 dynamic object 建模 → 更准的未来 BEV → 更早识别碰撞风险 → 更早避让。

---

## 局限: 抓不到"装死"的东西

帧差法的固有缺陷: **没动的东西抓不到**。

- 路边停的车, 一直没动, residual = 0, 但它随时可能启动
- 站人行道上的人, 没动, residual = 0, 但可能突然窜出来
- 红绿灯, 没动, 但状态会变

这些 "potential dynamic object" ResWorld 处理不了, 只能靠 prior trajectory 那条分支的 current BEV 兜底。作者在 limitation 里也承认了。

这是 frame differencing 的老毛病, 从 90 年代视频监控就有。要解决, 得引入语义先验——告诉模型"这个静止的车有潜在运动性", 纳入 world model 的预防性建模。

---

## 一张图总结整个 data flow

```
多帧图像 → GeoBEV → 各帧 BEV (warp 到 t 坐标系)
                         ↓
                  融合 → B_fuse
                    ┌────┴────┐
                    ↓         ↓
              Prior traj    同一 attention mask
              预测分支      提取各帧 sparse query
                    ↓         ↓
                T_prior    相减 → temporal residual
                    │         ↓
                    │    TR-World (只管动的东西)
                    │         ↓
                    │    Future dynamic delta
                    │         ↓
                    │    叠加到 B_fuse → B_future
                    │         ↓
                    └─→ FGTR: T_prior 去 B_future 踩点
                              ↓
                          T_final
```

左路看现在规划, 右路想象未来, 在 FGTR 处汇合修正。

---

## 我的整体看法

**真正聪明的点**:
1. **坐标系选择**这个 insight 太 elegant 了——一个简单的"把所有帧 align 到当前 ego 坐标系", 让"static 减完归零, dynamic 凸显"这个 classical idea 直接可用
2. **不监督 future BEV** 的反直觉设计, 有 ablation 撑腰, 是个 principled 的选择而非 hack
3. **FGTR 一石二鸟**: refine trajectory + 防 collapse, 一个模块解决两个问题

**我觉得可以吐槽的**:
1. Temporal window 只用 $k=2$, long-horizon dynamic modeling 弱——一辆车 3 秒前在动, 最近 1 秒停了, residual 就抓不到
2. Residual 是简单相减, 没 normalize, scale 大的 dynamic object 会 dominate, scale 小的 (远处的车) 信号弱
3. TR-World 内部用 simple sum 累积 residual, 没 fancy temporal modeling (RNN/Mamba/Transformer), 对复杂 motion pattern 建模能力有限
4. 只在 nuScenes/NAVSIM 验证, 没上 Waymo / 大规模真实数据

**我觉得 promising 的延伸方向**:
- 把 temporal residual 推广到 4D occupancy——dynamic occupancy mask 会更直观
- 结合 latent world model, 用 latent residual 替代显式 BEV residual, 可能更 compact
- FGTR 的 anti-collapse 思路推广到 VAE / diffusion, 用 task demand 而非显式 regularizer
- 加 interaction-aware modeling, 用 graph net 在 BEV query 间推理"谁可能影响谁"

代码在 https://github.com/mengtan00/ResWorld.git, 可以自己跑跑看。

---

## 关键 reference

- [ResWorld 代码](https://github.com/mengtan00/ResWorld.git)
- [SSR (直接 baseline)](https://arxiv.org/abs/2501.13355)
- [BEVDet4D (temporal alignment 源头)](https://arxiv.org/abs/2211.17054)
- [GeoBEV (BEV encoder)](https://arxiv.org/abs/2412.05976)
- [TokenLearner / Perceiver](https://arxiv.org/abs/2106.11210)
- [Deformable DETR](https://arxiv.org/abs/2010.04159)
- [OccWorld](https://arxiv.org/abs/2311.16010)
- [Drive-WM](https://arxiv.org/abs/2401.16163)
- [LAW (latent world model)](https://arxiv.org/abs/2406.08481)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [DiffusionDrive](https://arxiv.org/abs/2411.15149)
- [Hydra-MDP](https://arxiv.org/abs/2406.06978)
- [NAVSIM benchmark](https://arxiv.org/abs/2406.13349)
- [VAE Posterior Collapse](https://arxiv.org/abs/1807.04146)
- [ViBe (background subtraction 老祖宗)](https://ieeexplore.ieee.org/document/5654886)

要不要我再展开聊聊 temporal residual 推广到 occupancy 的可能性? 或者 FGTR 的 anti-collapse 思路能不能搬到 VAE/diffusion 上? 这俩方向我觉得都有料可挖。

---

# ResWorld: Temporal Residual World Model 深度解析

Karpathy 你好, 这篇 paper 我读完之后觉得它有几个真正聪明的 insight, 值得仔细拆解。我尽量从 first principles 出发 build intuition, 把公式、架构、实验都讲透。

---

## 1. 一句话本质: 在 BEV 上做 "frame differencing"

这篇 paper 的核心 idea, 如果类比到传统 computer vision, 就是 **background subtraction / frame differencing** 的 BEV 版本。早期 video surveillance 里, 用 $|I_t - I_{t-1}|$ 提取 moving object, 因为静态背景相减后归零。ResWorld 在 BEV feature 上做了同样的事, 但加了一层 sparse token 化和 world model prediction。

关键 trick: **把所有过去帧的 BEV 都 warp 到当前帧 $t$ 的 ego coordinate system**。这一步非常重要, 因为只有坐标对齐后, 减法才有物理意义——同一 spatial location 的 feature 差异, 才能解释为 "有什么东西移动了"。

如果用 future coordinate system (ego 已移动), 那么所有 static object 在新坐标系下也会"变化", residual 就被污染了。这点是这篇 paper 最 elegant 的地方。

参考 [BEVDet4D](https://arxiv.org/abs/2211.17054) 第一次提出 BEV 上做 temporal alignment, 是这个 idea 的源头。

---

## 2. 整体架构拆解

### 2.1 BEV Feature 提取与融合

模型用 [GeoBEV](https://arxiv.org/abs/2412.05976) 作为 BEV encoder, ResNet-50 backbone, 输入 256×704 的 multi-view image, 输出 $\mathbf{B}_t \in \mathbb{R}^{C \times H \times W}$。

选 GeoBEV 而非 BEVFormer 的原因是: **temporal residual 对 BEV 的几何质量非常敏感**。如果 BEV 本身有 geometric distortion, 相邻帧 warp 后做减法, 会产生大量 noise, dynamic object 信号被淹没。GeoBEV 通过 geometric embedding 提升了 BEV 的几何一致性, 这对 residual 计算至关重要。

公式 (1):
$$\mathbf{B}_{fuse} = \mathrm{Conv}(\mathrm{Concat}(\mathbf{B}_t, \mathbf{B}_{t-1}, \dots, \mathbf{B}_{t-k}))$$

- $\mathbf{B}_t \in \mathbb{R}^{C \times H \times W}$: 当前 timestamp BEV, $C$=channel, $H$=height, $W$=width
- $\mathbf{B}_{t-1}, \dots, \mathbf{B}_{t-k}$: 已 warp 到 $\mathbf{B}_t$ 坐标系下的历史 BEV
- $k=2$: 用过去 2 帧 + 当前帧
- Conv: 1×1 convolution 融合

注意所有历史帧都 align 到 $t$ 时刻的 ego 坐标, 这是后续 residual 计算的基础。

### 2.2 Prior Trajectory 预测

借用了 [SSR](https://arxiv.org/abs/2501.13355) 的 planning head。先把 dense BEV 压成 sparse query:

公式 (2)-(3):
$$\mathbf{S}_{fuse} = \mathrm{AvgPool}(\mathrm{SA}(\mathbf{B}_{fuse}) \odot \mathbf{B}_{fuse})$$
$$\mathbf{S}_{fuse} = \mathrm{SelfAttention}(\mathbf{S}_{fuse})$$

- $\mathbf{S}_{fuse} \in \mathbb{R}^{N_s \times C}$: $N_s$ 个 sparse scene query
- $\mathrm{SA}$: spatial attention map generator (类似 [TokenLearner](https://arxiv.org/abs/2106.11210))
- $\odot$: element-wise multiplication, 用 attention map 加权 BEV
- AvgPool: 把 H×W spatial 维度压成 1, 得到 $N_s$ 个 token

然后 waypoint queries $\mathbf{W} \in \mathbb{R}^{N_t \times C}$ 与 $\mathbf{S}_{fuse}$ 做 cross-attention, MLP decode 出 prior trajectory:

公式 (4):
$$\mathbf{T}_{prior} = \mathrm{MLP}(\mathrm{CrossAttention}(\mathbf{W}, \mathbf{S}_{fuse}, \mathbf{S}_{fuse}))$$

- $N_t$: 预测的 future timestamp 数 (nuScenes 上一般是 6 个 waypoints, 2 秒, 0.5s 间隔)
- $\mathbf{T}_{prior} \in \mathbb{R}^{N_t \times 2}$: 每行是 (x, y) ego 坐标

---

## 3. Temporal Residual Extraction: 核心 idea 详解

这一节是 paper 最关键的部分。

### 3.1 为什么用同一个 spatial attention mask

公式 (5):
$$\mathbf{S}_i = \mathrm{AvgPool}(\mathrm{SA}(\mathbf{B}_{fuse}) \odot \mathbf{B}_i)$$

这里有一个 subtle 但极重要的设计: **用 $\mathbf{B}_{fuse}$ (而非 $\mathbf{B}_i$) 来预测 spatial attention map**, 然后用这同一个 mask 去提取每个 timestamp 的 sparse query。

为什么? 因为 $\mathbf{B}_{fuse}$ 融合了多帧信息, 它的 spatial attention 更稳定, 更能 highlight 出 dynamic region。如果每个 timestamp 用自己的 BEV 生成 mask, 不同 timestamp 的 mask 会偏移, 导致 query 之间 spatial 不对齐, residual 计算就失去意义。

这是 attention map 的"锚点"作用——把 $k+1$ 个 timestamp 的信息都抽到同一组 spatial location 上。

### 3.2 Residual 计算

得到 $\{\mathbf{S}_t, \mathbf{S}_{t-1}, \dots, \mathbf{S}_{t-k}\}$ 后, 计算:
$$\mathbf{R}_i = \mathbf{S}_i - \mathbf{S}_{i-1}$$

- $\mathbf{R}_i \in \mathbb{R}^{N_s \times C}$: timestamp $i$ 相对 $i-1$ 的 residual
- 最终得到 $\{\mathbf{R}_t, \mathbf{R}_{t-1}, \dots, \mathbf{R}_{t-k+1}\}$, 共 $k$ 个 residual

**intuition**: 在 ego-centered coordinate system 下, static object (路面、建筑、绿化带) 在相邻帧的同一 location 上 feature 几乎不变, 减完接近零。Dynamic object (车、人) 因为移动了, 同一 location 上 $t-1$ 时刻有车, $t$ 时刻车开走了, feature 大幅变化, residual 就 capture 到了 dynamic signal。

这其实是个非常 classical 的思想, 类似 [ViBe](https://ieeexplore.ieee.org/document/5654886)、MOG background subtraction, 但作用在 learned BEV feature 上而非 raw pixel。

---

## 4. Temporal Residual World Model (TR-World)

### 4.1 为什么不做 static object 建模

这是 paper 最有 insight 的一句: **"如果预测 future BEV 时仍采用 $\mathbf{B}_t$ 的坐标系, 那么 static object 的空间分布可以视为不变"**。

这意味着 future BEV 的 static 部分 = $\mathbf{B}_{fuse}$, 不需要 world model 再去预测一遍。World model 只需要预测 dynamic object 在未来会移动到哪里, 把这个 delta 叠加到 $\mathbf{B}_{fuse}$ 上, 就是 future BEV。

这跟 [OccWorld](https://arxiv.org/abs/2311.16010)、[Drive-WM](https://arxiv.org/abs/2311.16010)、[Drive-OccWorld](https://arxiv.org/abs/2411.10865) 的做法形成对比——它们都在预测整个 future BEV / occupancy, 大量 capacity 浪费在 static 上, dynamic 的预测精度反而受限。

公式 (6):
$$\hat{\mathbf{R}} = \sum_{i=t-k+1}^{t} \mathrm{SelfAttention}(\mathbf{R}_i)$$

- $\hat{\mathbf{R}} \in \mathbb{R}^{N_s \times C}$: 累积后的 dynamic object future representation
- 每个 $\mathbf{R}_i$ 先 self-attention 增强, 然后跨 timestamp 累加

这里 accumulation 是 simple sum, 没用 RNN/Transformer 跨时序建模。这是个简化设计, 好处是 efficient, 坏处是 long-range temporal modeling 能力弱。如果未来扩展到更长 horizon, 可能需要换成 Mamba 或 temporal transformer。

### 4.2 TokenFuser: 从 sparse token 恢复 dense BEV

公式 (7):
$$\mathbf{B}_{future} = \mathrm{TokenFuser}(\hat{\mathbf{R}}, \mathbf{B}_{fuse}) + \mathbf{B}_{fuse} = \mathrm{MLP}(\mathbf{B}_{fuse}) \otimes \hat{\mathbf{R}} + \mathbf{B}_{fuse}$$

- $\mathrm{MLP}(\mathbf{B}_{fuse})$: 把 $\mathbf{B}_{fuse} \in \mathbb{R}^{C \times H \times W}$ 映射到 $\mathbb{R}^{N_s \times H \times W}$
- $\otimes$: matrix transpose + multiplication 组合操作, 把 sparse token $\hat{\mathbf{R}} \in \mathbb{R}^{N_s \times C}$ expand 成 dense $\mathbb{R}^{C \times H \times W}$
- 加 $\mathbf{B}_{fuse}$: residual connection, 保留 static 信息

这是 [Perceiver](https://arxiv.org/abs/2103.03206) 系列的 cross-attention 思想: sparse latent 与 dense input 交互, 用 dense 引导 sparse 的展开。

---

## 5. Future-Guided Trajectory Refinement (FGTR)

### 5.1 解决两个问题

FGTR 同时解决两个问题:
1. **Trajectory 与 future BEV 没有深度交互**: 传统方法 world model 只是把 future prediction 当 proxy task, 间接提升 encoder 能力, future BEV 没直接参与 planning
2. **World model collapse**: 没有 supervision 时, world model 倾向于把所有场景映射成相似的 future BEV (mode collapse / representation collapse)

### 5.2 Deformable Attention 的妙用

公式 (8)-(9):
$$\mathbf{W} = \mathrm{DeformAttention}(\mathbf{W}, \mathbf{B}_{future}, \mathbf{T}_{prior})$$
$$\mathbf{T}_{final} = \mathrm{MLP}(\mathbf{W})$$

- $\mathbf{W} \in \mathbb{R}^{N_t \times C}$: waypoint queries, 第 $i$ 个 query 代表 $t+i$ 时刻的 ego 状态
- $\mathbf{B}_{future} \in \mathbb{R}^{C \times H \times W}$: future BEV
- $\mathbf{T}_{prior} \in \mathbb{R}^{N_t \times 2}$: prior trajectory, 作为 deformable attention 的 reference points

[Deformable Attention](https://arxiv.org/abs/2010.04159) 的核心: 不做 dense attention, 只在 reference point 周围 sample 少量点。这里 reference points 是 prior trajectory 上每个 waypoint 对应的 BEV location。

intuition: 第 $i$ 个 waypoint query 去 $\mathbf{B}_{future}$ 上"看一眼" prior trajectory 在 future 第 $i$ 秒的位置周围有什么——是否有车? 是否出 road? 这个信息回流到 query, 然后 MLP decode 出 refined trajectory。

这相当于 ego car 在 "mental simulation" 里开了一遍 prior trajectory, 检查碰撞和越界, 修正。

### 5.3 双重作用: 监督 + refinement

FGTR 的副作用是给 $\mathbf{B}_{future}$ 提供 **sparse spatial-temporal supervision**:
- **Spatial**: reference points 是 prior trajectory 的具体 location, 强制 $\mathbf{B}_{future}$ 在这些 location 上有 meaningful 的 feature
- **Temporal**: 不同 waypoint query 对应不同 future timestamp, 强制 $\mathbf{B}_{future}$ 跨时序保留 dynamic object 分布

这个 supervision 不是直接的 label, 而是通过 gradient backprop 间接施加——trajectory L1 loss 会回传到 $\mathbf{B}_{future}$, 鼓励它在 trajectory 关心的 spatiotemporal 位置上有正确信息。

### 5.4 为什么能防止 collapse

World model collapse 的本质是: world model 发现 "把所有 future BEV 都映射成一个常数" 也能让 planning head 凑合工作, 因为 planning 主要靠 $\mathbf{B}_{fuse}$, future BEV 是辅助。

但 FGTR 让 planning head **必须依赖 $\mathbf{B}_{future}$** 去做 refinement, 这就强制 $\mathbf{B}_{future}$ 必须有信息。如果 $\mathbf{B}_{future}$ collapse 成常数, refinement 就失效, collision rate 上升, loss 增大, gradient 会 push world model 输出 diverse 的 future BEV。

这跟 GAN 中 mode collapse、VAE 中 posterior collapse 是同源问题, 都需要架构设计避免 trivial solution。这里用的是 **task-driven supervision** 而非 explicit diversity regularization, 比较 elegant。

参考 [World Model Collapse](https://arxiv.org/abs/2501.09867) 等讨论。

---

## 6. 反直觉设计: 不监督 future BEV

公式 (10):
$$\mathcal{L} = \mathrm{L1}(\mathbf{T}_{prior}, \mathbf{T}_{GT}) + \mathrm{L1}(\mathbf{T}_{final}, \mathbf{T}_{GT})$$

只监督 trajectory, 不监督 $\mathbf{B}_{future}$。这非常反直觉——传统 world model 都用 future frame 的 ground truth 监督 future prediction。

作者的解释: **如果用任何一个具体 timestamp (比如 $t+1$) 的 ground truth 监督 $\mathbf{B}_{future}$, 模型会把 $\mathbf{B}_{future}$ 拟合成 $t+1$ 的 specific snapshot, 丢失其他 timestamp ($t+2, t+3$) 的 dynamic object 位置信息**。

而 FGTR 需要的是跨 $N_t$ 个 timestamp 的 dynamic 信息(因为不同 waypoint query 要 query 不同时刻的 future BEV)。一旦用单一 timestamp 监督, multi-timestamp 的需求就无法满足。

Ablation (Table 4) 验证:
- TR-World + future supervision: L2 avg = 0.61, CR avg = 0.21
- TR-World 无 future supervision: L2 avg = 0.59, CR avg = 0.17

无监督反而好 6% L2, 19% CR。这是个很强的 evidence。

**intuition**: world model 应该输出一个 "future 的概要表示", 而不是 "某一帧的 snapshot"。这跟 latent world model ([LAW](https://arxiv.org/abs/2406.08481)、[DreamerV3](https://arxiv.org/abs/2301.04104)) 的哲学类似——latent space 上做 long-horizon planning, 不强求 pixel-level reconstruction。

---

## 7. 实验结果深度分析

### 7.1 nuScenes (Table 1)

ResWorld‡ (None, ego status): **L2 avg = 0.30 m, CR avg = 0.07**

对比 SOTA:
- [SSR](https://arxiv.org/abs/2501.13355)‡ (None): L2 = 0.39, CR = 0.15
- [BEV-Planner++](https://arxiv.org/abs/2311.09865)‡ (None): L2 = 0.35, CR = 0.34
- [SparseDrive](https://arxiv.org/abs/2405.19620)‡ (Det&Track&Map&Motion): L2 = 0.61, CR = 0.08
- [MomAD](https://arxiv.org/abs/2412.11904)‡ (Det&Track&Map&Motion): L2 = 0.60, CR = 0.09
- [DiffusionDrive](https://arxiv.org/abs/2411.15149)‡ (Det&Track&Map&Motion): L2 = 0.57, CR = N/A

ResWorld 在不用任何 auxiliary task 的情况下, L2 比 SSR 好 23%, CR 好 53%。这个 CR 的提升特别显著——collision rate 直接反映 world model 对 dynamic object 的建模质量。

注意 nuScenes open-loop 评估有 ego status 泄漏问题 ([BEV-Planner](https://arxiv.org/abs/2406.14039)), 所以 paper 同时报告两种配置。即使无 ego status, ResWorld 也 SOTA。

### 7.2 NAVSIM (Table 2)

NAVSIM 是 closed-loop 评估, 更接近真实驾驶:
- ResWorld* (with temporal residual, Det&Map): **PDMS = 89.0**
- [DiffusionDrive](https://arxiv.org/abs/2411.15149) (Det&Map): PDMS = 88.1
- [Hydra-MDP](https://arxiv.org/abs/2406.06978) (Det&Map): PDMS = 86.5
- ResWorld (None): PDMS = 87.3 (无 auxiliary task 也强)

特别关注 **TTC (Time-to-Collision)**:
- ResWorld* = 98.9, DiffusionDrive = 94.7, Hydra-MDP = 94.6

ResWorld* 的 TTC 遥遥领先, 说明 temporal residual 对 dynamic object 的建模确实转化为 collision avoidance 能力。

NAVSIM 实现里有趣: 因为 [TransFuser](https://arxiv.org/abs/2205.15997) baseline 不用历史帧, 作者用 **agent queries from detection** 替代 temporal residual 作为 TR-World 输入, 这说明 TR-World 框架灵活, residual 只是 dynamic info 的一种 source。

### 7.3 Ablation (Table 3)

Baseline (SSR + GeoBEV, ego status): L2 avg = 0.65, CR avg = 0.28
+ TR-World: L2 = 0.61, CR = 0.25
+ FGTR: L2 = 0.61, CR = 0.22
+ TR-World + FGTR: L2 = 0.59, CR = 0.17

两个模块互补:
- TR-World 提升 encoder 能力, 让 prior trajectory 也变好
- FGTR 直接 refine trajectory, 同时监督 TR-World

### 7.4 Prior vs Final Trajectory (Table 5)

Prior (用 baseline 结构, 但 BEV 被 TR-World+FGTR 优化过): L2 = 0.61, CR = 0.18
Final (加 FGTR refinement): L2 = 0.59, CR = 0.17
Baseline: L2 = 0.65, CR = 0.28

**Prior trajectory 已经比 baseline 好**, 因为 TR-World 让 BEV feature 更 informative。这暗示一个 deploy trick: 训练时用大 TR-World+FGTR, 推理时只跑 prior trajectory, 减少 latency。类似 [data distillation](https://arxiv.org/abs/2106.05285) 思想。

---

## 8. 架构图解析 (Figure 2)

我描述一下整体 data flow:

```
Multi-view images (t, t-1, t-2)
        ↓
    GeoBEV encoder
        ↓
B_t, B_{t-1}, B_{t-2} (warp to t's coord)
        ↓
    Concat + Conv → B_fuse
        ↓
    ┌─────────────────┬──────────────────┐
    ↓                 ↓                  ↓
TokenLearner       SA(B_fuse)         SA(B_fuse)
    ↓                 ↓                  ↓
S_fuse          SA(B_fuse)⊙B_t     SA(B_fuse)⊙B_{t-1}, ...
    ↓                 ↓                  ↓
Self-Attn       S_t              S_{t-1}, S_{t-2}
    ↓                 ↓                  ↓
Cross-Attn(W,S_fuse)  R_t = S_t - S_{t-1}  R_{t-1} = ...
    ↓                 ↓
T_prior ← MLP    Self-Attn(R_i) → Σ → R_hat
    ↓                                ↓
    │                      TokenFuser(R_hat, B_fuse) + B_fuse
    │                                ↓
    │                          B_future
    ↓                                ↓
    └───── DeformAttn(W, B_future, T_prior) ──→ W
                                    ↓
                              MLP → T_final
```

左路是 prior trajectory, 右路是 world model + future BEV, 在 FGTR 处交汇。

---

## 9. Limitations 与个人评论

作者自己提到: temporal residual 对 **potential dynamic object** (静止但可能突然动, 如停的车、路边人) 不敏感, 因为它们没产生 residual。这其实是个 fundamental limitation——frame differencing 的固有缺陷。

可能的解决思路 (paper 没提, 我来 speculation):
1. **Semantic prior**: 用 detection/map 分支给每个 static object 打 "potential motion" score, 高 score 的纳入 TR-World
2. **Longer temporal window**: 历史窗口拉到 5-10 秒, 捕获间歇性运动
3. **Future-conditioned residual**: 用 prior trajectory 生成 "假想未来场景", 在假想场景里 check 哪些 object 可能动
4. **Interaction modeling**: 用 graph neural network 在 BEV query 间建模交互, 推理 latent motion

另外一个我没看到讨论的点: **ego motion error 对 residual 的污染**。如果 ego pose 估计有误差, warp 不准, residual 会有大量 noise。GeoBEV 缓解了这个, 但没消除。在 GPS/IMU 退化场景 (隧道、地下车库) 可能成问题。

---

## 10. 与相关工作的联系

### 10.1 与 SSR 的关系
[SSR](https://arxiv.org/abs/2501.13355) 是 ResWorld 的直接 baseline。SSR 用 sparse scene query 做 world model, 输入是当前 BEV, 输出 future BEV, 但没区分 dynamic/static, 也没 FGTR。ResWorld 在 SSR 基础上加 temporal residual + FGTR, 是个 clean increment。

### 10.2 与 LAW 的关系
[LAW](https://arxiv.org/abs/2406.08481) 是 latent world model, 在 latent space 预测未来, 不显式 reconstruct BEV。LAW 更 abstract, ResWorld 更 spatial-explicit。两者哲学不同: latent world model 偏 representation learning, ResWorld 偏 explicit dynamic modeling。可能结合: 用 latent residual 替代显式 BEV residual。

### 10.3 与 OccWorld 的关系
[OccWorld](https://arxiv.org/abs/2311.16010) 在 occupancy 上做 world model, 4D occupancy prediction。Occupancy 比 BEV 更 dense, 信息量大, 但计算开销也大。ResWorld 选择 BEV + sparse query, 更 efficient。如果 ResWorld 的 temporal residual idea 推广到 occupancy, 会很有意思——4D occupancy residual = dynamic occupancy mask。

### 10.4 与 Drive-WM 的关系
[Drive-WM](https://arxiv.org/abs/2401.16163) 生成 driving video, pixel-level future prediction, 计算极重。ResWorld 只做 BEV future, 计算轻得多。Drive-WM 适合 sensor simulation, ResWorld 适合 planning。

### 10.5 与 Dreamer 系列的关系
[DreamerV3](https://arxiv.org/abs/2301.04104) 是 RL world model 的代表, 在 latent space rollout。ResWorld 借鉴了 "world model 辅助 planning" 的思想, 但没用 RL, 而是 imitation learning。未来如果 ResWorld + RL fine-tuning, 可能更 interesting。

### 10.6 与 Generative Model Collapse 的关系
World model collapse 与 [VAE posterior collapse](https://arxiv.org/abs/1807.04146)、[GAN mode collapse](https://arxiv.org/abs/1905.12452)、[LLM representation degeneration](https://arxiv.org/abs/2305.09773) 同源。FGTR 的 task-driven supervision 是一种新颖的 anti-collapse 策略, 值得在 generative model 领域推广。

---

## 11. 我的整体评价

**亮点**:
1. BEV coordinate system trick 是 simple but profound——一个坐标系选择解决 static/dynamic 分离
2. Temporal residual 不依赖 detection/tracking, 实现 perception-free world model
3. FGTR 同时解决 trajectory refinement 和 world model collapse, 一石二鸟
4. 不监督 future BEV 的反直觉设计有 ablation 支撑
5. 在 nuScenes 和 NAVSIM 两个 benchmark 都 SOTA

**不足**:
1. Temporal window 短 (k=2), long-horizon dynamic modeling 弱
2. Residual 简单相减, 没 normalize, 可能 scale sensitive
3. TR-World 内部用 simple sum accumulation, 没 fancy temporal modeling
4. 对 potential dynamic object 无能为力
5. 只在 nuScenes/NAVSIM, 没在大规模真实 driving 数据 (如 Waymo, nuPlan full) 验证
6. Loss 只 L1 on trajectory, 没探索 contrastive / auxiliary losses

**未来方向 speculation**:
- 把 temporal residual 推广到 4D occupancy
- 结合 latent world model, 用 latent residual
- 加入 interaction-aware dynamic modeling (graph net)
- 用 diffusion model 替代 simple sum, 生成 multi-modal future
- Self-supervised pretraining on large-scale driving video, transfer 到 planning

代码在 https://github.com/mengtan00/ResWorld.git, 可复现。

---

## 参考链接

- [ResWorld 代码](https://github.com/mengtan00/ResWorld.git)
- [SSR (baseline)](https://arxiv.org/abs/2501.13355)
- [LAW](https://arxiv.org/abs/2406.08481)
- [OccWorld](https://arxiv.org/abs/2311.16010)
- [Drive-WM](https://arxiv.org/abs/2401.16163)
- [Drive-OccWorld](https://arxiv.org/abs/2411.10865)
- [GeoBEV](https://arxiv.org/abs/2412.05976)
- [BEVDet4D](https://arxiv.org/abs/2211.17054)
- [BEVFormer](https://arxiv.org/abs/2203.17270)
- [TokenLearner / Perceiver](https://arxiv.org/abs/2106.11210)
- [Deformable DETR](https://arxiv.org/abs/2010.04159)
- [UniAD](https://arxiv.org/abs/2212.10156)
- [VAD](https://arxiv.org/abs/2303.12077)
- [VADv2](https://arxiv.org/abs/2402.13243)
- [GenAD](https://arxiv.org/abs/2312.01149)
- [SparseDrive](https://arxiv.org/abs/2405.19620)
- [DiffusionDrive](https://arxiv.org/abs/2411.15149)
- [MomAD](https://arxiv.org/abs/2412.11904)
- [Hydra-MDP](https://arxiv.org/abs/2406.06978)
- [TransFuser](https://arxiv.org/abs/2205.15997)
- [NAVSIM](https://arxiv.org/abs/2406.13349)
- [nuScenes](https://www.nuscenes.org/)
- [BEV-Planner (ego status critique)](https://arxiv.org/abs/2406.14039)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [World Model Collapse](https://arxiv.org/abs/2501.09867)
- [VAE Posterior Collapse](https://arxiv.org/abs/1807.04146)
- [GAN Mode Collapse](https://arxiv.org/abs/1905.12452)
- [ViBe (background subtraction)](https://ieeexplore.ieee.org/document/5654886)

如果你想进一步聊, 我对几个点特别感兴趣: (1) 把 temporal residual idea 推广到 occupancy 的可行性; (2) latent residual vs spatial residual 的 trade-off; (3) FGTR 的 anti-collapse 思路能不能用到 VAE / diffusion training。这些都没在 paper 里讨论, 但我觉得都是 promising 的 open direction。
