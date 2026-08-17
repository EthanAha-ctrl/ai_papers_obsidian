---
source_pdf: SIMSCALE Learning to Drive via Real-World Simulation at Scale.pdf
paper_sha256: 69877a91da6c362bec021cc5d153951a1d0b7f0290d607f48fee2b62fff15631
processed_at: '2026-08-12T06:41:19-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 SIMSCALE

## 一句话版本

**开车数据全是老司机在正常路上的录像，压根没有"差点撞车怎么救回来"的片段。SIMSCALE 用 3DGS 把真实录像变成可以乱动的"录像重拍机器"，故意让车开歪，再找个专家教它怎么救回来，把这些"救车视频"当训练数据喂给 planner。结果发现：数据越多效果越好，而且 diffusion / scoring 这种能输出多种开法的模型才吃得下这种数据。**

---

## 问题是什么

想象你训练一个小孩开车，只给他看 1000 小时爸爸开车的视频。爸爸开得稳，永远在车道中间，永远安全跟车。小孩学会了"在车道中间开"。

但有一天小孩稍微走神偏了 30cm——这个状态爸爸的视频里从来没出现过。小孩不知道怎么救，越偏越多，撞了。

这就是 imitation learning 的老问题，叫 **covariate shift**。训练数据永远在"好状态"上，测试时一旦偏到"坏状态"，模型懵了。

经典解法是 DAgger：让小孩开车，爸爸坐旁边随时纠正，收集"偏了之后怎么救"的数据。但在 real world 这么干太危险。

SIMSCALE 的思路：**用 3DGS 把真实道路变成"可重拍的电影"**。让 ego 在虚拟世界里故意开歪，再让一个 rule-based expert 示范怎么救，把整个过程渲染成新的训练数据。

---

## 怎么造数据

### 第一步：把真实场景变成可编辑的 3D 世界

用 3D Gaussian Splatting (3DGS) 重建真实场景。3DGS 你可以理解成：把一帧帧 RGB 图像 + LiDAR 点云喂进去，它学出一个"3D 场景表示"，之后你可以从任意视角重新渲染出图。

关键：场景里的静态背景（路面、建筑、树）是一个 asset，每辆车是独立 asset。runtime 时你可以把车放到任意 (x, y, θ) 位置，渲染出 ego 视角的 multi-view RGB。

这相当于把一段真实录像变成一个 "可重拍片场"——背景不变，但你可以重新安排演员位置和镜头角度。

技术细节：
- 用 [MTGS](https://arxiv.org/abs/2503.12552) 的 codebase
- 按 spatio-temporal block 切片重建，避免全量重建爆显存
- PSNR < 27 的 block 扔掉，保证渲染质量
- Multi-view camera 之间做 exposure alignment，用 projected LiDAR 点引导

### 第二步：故意把 ego 开歪 (Perturbation)

从 human trajectory vocabulary (16384 条聚类好的人类轨迹) 里采样，施加约束：
- 横向偏移 ±2m（差不多半个车道）
- 纵向偏移 ±20m
- 朝向偏 ±20°
- EPDMS ≥ 0.8（proxy 表示这条轨迹物理上还合法）

用 LQR 控制 ego 沿这条 perturbed trajectory 走 H 步（4 秒），把 ego 带到一个"有点偏但还没撞"的 OOD state。

同时其他 agent 用 IDM (Intelligent Driver Model) reactive 模拟——它们会根据 ego 的偏移做出反应，比如减速让行、变道躲避。

这一步本质是 **制造 covariate shift 的 training data**：强迫 ego 进入 training distribution 之外的状态。

### 第三步：让专家教怎么救

从这个 OOD state 出发，用两种 "pseudo-expert" 生成 demonstration：

**Recovery Expert（保守派）**：
从 10 万条 human trajectory bank 里检索一条"起末状态最匹配"的，直接复制过来当 expert。相当于说："这个偏了的状态，人类大概会这么开回去。"

公式：
$$
\tilde{a}_{t:t+H} = \arg\min_{a \in \mathcal{V}_h} \|\mathbf{m}(a) - \mathbf{m}_r\|_1
$$

其中 $\mathbf{m} = [v_t^x, v_t^y, \theta_t, x_{t+H-1}, y_{t+H-1}, \theta_{t+H-1}]$ 是 6 维起末状态向量，$\mathcal{V}_h$ 是完整 human vocab，$\mathbf{m}_r$ 是当前 perturbed state 的 target。

优点：保证 human-like。缺点：永远是"回到人类轨道上"，explore 不了新解法。

**Planner Expert（探索派）**：
用 [PDM-Closed](https://arxiv.org/abs/2308.05731)（nuPlan leaderboard 上的 SOTA rule-based planner）直接跑。它用 ground-truth BEV state，sweep 一堆 kinematically feasible 的 trajectory proposal，按 collision / progress / comfort / rules 算 cost，Kalman 平滑选最优。

公式：
$$
\tilde{a}_{t:t+H} = \mathbf{P}(\tilde{s}_{t:t+H})
$$

优点：能给 OOD state 提供"非人类但 optimal"的解法，比如猛打方向盘绕障。缺点：comfort 差，jerk 大，动作没真人那么丝滑。

### 第四步：渲染成视频喂给 planner

把两阶段 rollout (perturb H 步 + expert H 步) 的所有 agent pose 喂给 3DGS engine，渲染出 multi-view RGB。

每条真实 clip 可以 perturb 多次（5 轮），每轮采不同 perturbation，最终从 100K 真实场景造出 147K (recovery) 或 237K (planner) 仿真场景。

---

## 怎么训练

非常朴素：**real data 和 sim data 混在一起 random sample 做 joint training**。没有 domain adaptation，没有 adversarial loss，没有 KL reg。

对 regression / diffusion planner：
$$
\arg\min_\theta \mathbb{E}_{(a,o) \sim (\mathcal{D} \cup \mathcal{D}_{sim})} [\mathcal{L}_{im}(a, \pi_\theta(\hat{a}|o))]
$$

对 scoring planner (GTRS-Dense)，加 reward supervision：
$$
\arg\min_\theta \mathbb{E}_{(a,o,r) \sim (\mathcal{D} \cup \mathcal{D}_{sim})} [\lambda \mathcal{L}_{im} + \mathcal{L}_r(r, \pi_\theta(\hat{a}|o))]
$$

scoring planner 的 vocab 是 16384 条 clustered human trajectory，每条预测多个 sub-metric score (NC/DAC/DDC/TLC/TTC/EP/LK/HC/EC)，inference 时 argmax 选最高分轨迹。

### 最反直觉的发现：Reward-only 更好

对 scoring planner，在 sim data 上**只给 reward signal，不给 expert trajectory**：
$$
\arg\min_\theta \mathbb{E}_{(a,o,r) \sim \mathcal{D}} [\lambda \mathcal{L}_{im} + \mathcal{L}_r] + \mathbb{E}_{(o,r) \sim \mathcal{D}_{sim}} [\mathcal{L}_r]
$$

结果比"sim + expert trajectory"还好。Tab. 3：

| Backbone | 配置 | EPDMS |
|----------|------|-------|
| ResNet34 | real only baseline | 38.3 |
| ResNet34 | real + sim(recovery expert) | 43.0 |
| ResNet34 | real + sim(planner expert) | 46.1 |
| ResNet34 | real + sim(**reward only**) | **46.9** |
| V2-99 | real only baseline | 41.9 |
| V2-99 | real + sim(recovery expert) | 46.4 |
| V2-99 | real + sim(planner expert) | 47.7 |
| V2-99 | real + sim(**reward only**) | **48.0** |

但**只在 real data 上 reward-only 训练会崩**（38.8，比 baseline 41.9 还低）。

直觉：
- Real data 的 imitation 是 **anchor**，防止 distribution drift
- Sim data 的 reward 是 **exploration signal**，让 planner 在 vocab 中自己找高分轨迹
- Pseudo-expert trajectory 是 single sample from optimal manifold，强行 imitate 反而把 planner 锁死在这个 sample 上
- Reward 描述整个 optimal manifold，planner 自己 explore，等价于 infinite-sample supervision

这跟 RLHF 中 "SFT anchor + PPO explore" 的范式完全一致。reference: [InstructGPT](https://arxiv.org/abs/2203.02155)

---

## 结果有多好

### navhard（OOD + safety-critical benchmark）

| Model | w/o sim | w/ sim | 提升 |
|-------|---------|--------|------|
| LTF (regression, 56M) | 24.4 | 30.2 | **+24%** |
| DiffusionDrive (diffusion, 61M) | 27.5 | 32.6 | **+19%** |
| GTRS-Dense (scoring, ResNet34, 67M) | 38.3 | 46.9 | **+22%** |
| GTRS-Dense (scoring, V2-99, 83M) | 41.9 | 48.0 | **+15%** ← 新 SOTA |

弱 baseline 提升更大，强 baseline 绝对值更高。

### navtest（12146 真实 diverse scenarios）

所有模型 +0.8 ~ +2.9 EPDMS。提升比 navhard 小，因为 navtest 主要是 common scenario，real data 已经覆盖得不错，sim 的边际价值递减。

---

## Scaling Law 长什么样

这是 paper 最核心的 contribution。固定 real data 100K，逐渐加 sim data (0 → 237K)，用 log-quadratic 拟合：

$$
S(N) = a \log^2(N) + b \log(N) + c
$$

$N$ 是 total data size，$S(N)$ 是 performance。如果 $a \to 0$ 退化成 $S = b\log N + c$，就是 LLM 那种 linear log-scaling。

Fig. 4 四张图：
- **LTF (regression)**：sim:real 到 1:1 后开始 plateau 下降，$a < 0$，抛物线开口向下
- **DiffusionDrive (diffusion)**：近似 perfect log-linear，$a \approx 0$，没 saturate
- **GTRS-Dense (scoring)**：reward-only > planner > recovery，V2-99 比 ResNet34 斜率更陡

### 为什么 regression 会 saturate 但 diffusion 不会

同一个 perturbed state，planner-based expert 在不同 sim round 会给出**多个 valid recovery**（绕左、绕右、刹停）。这形成 multi-peak distribution：
- Regression loss（L1/L2 on waypoint）假设 unimodal target，multi-peak 下会 average 这些 mode，给出"绕左 + 绕右平均 = 撞上去"的 mode-confusion
- Diffusion 用 score matching / denoising，能正确 capture 多峰，每个 mode 都学

这跟你之前讲 diffusion policy 的思路完全一致。reference: [Diffusion Policy (Chi et al.)](https://diffusion-policy.cs.columbia.edu/)

### 大模型 scaling 更陡

GTRS-Dense V2-99 (83M) vs ResNet34 (67M)，同样 sim data 量下大模型 gain 更大。这跟 LLM 中 "大模型 + 多数据 = 更陡 scaling" 一致。reference: [Chinchilla](https://arxiv.org/abs/2203.15556)

---

## 其他关键 ablation

### Reactive 重要

| Type | #Round | #Sim | ResNet34 | V2-99 |
|------|--------|------|----------|-------|
| Non-reactive | 2 | 141K | 43.7 | 45.6 |
| Reactive | 2 | 120K | 44.4 | 46.7 |
| Reactive | 3 | 167K | 45.0 | 47.9 |

Non-reactive（其他 agent 按 log 走）样本多但效果差。如果其他车不响应 ego 偏移，ego 撞上去也是"按 log 演的"，没有交互因果，planner 学不到"我偏了别人会怎么反应"。

### Visual Fidelity 重要

PSNR ≥ 27 vs < 27，每 round 稳定 +0.5~1.5 EPDMS。3DGS 渲染质量直接传导到 planner 性能。

### Real Data 越少 Sim Gain 越大（但 100K 时 gain 不消失）

10K real data + sim：ResNet34 +22.4% EPDMS
100K real data + sim：gain **没有 narrowing**

这说明 sim 的 marginal value 不随 real data 增多而消失。这是一个 strong scaling claim——sim data 是 real data 的 scalable 互补，仅仅只是替换品。

---

## 跟 Online RL 比

**Online RL in 3DGS**（e.g. [RAD](https://arxiv.org/abs/2503.07152)）：
planner 在 3DGS env 里 closed-loop 跑，每次 action 触发 reward，policy gradient 更新。需要可微 / fast rollout，训练不稳定，sim-to-real gap 大。

**SimScale**：
把 exploration 的 hard part 用 rule-based expert 在 offline 阶段预先做完，剩下的训练退化成 supervised IL/reward regression。更稳定，可支持任意 planner paradigm。

本质是 AlphaGo 的 "policy pretrain + RL fine-tune" 思路，也是 RLAIF 的思路：用 offline synthetic data 替代 online exploration。reference: [AlphaGo](https://www.nature.com/articles/nature24270) | [GR00T N1](https://arxiv.org/abs/2503.14734)

---

## 几个 Karpathy 视角的延伸

**1. 这本质是 DAgger 的 offline 近似**

DAgger 需要 online expert 修正。SimScale 用 PDM-Closed 当 offline expert，在 sim 中预先造好 (OOD state, expert action) pair。次优解是 PDM-Closed 本身有限——后续用 learning-based BEV planner（GameFormer、Diffusion-ES）当 expert 应该能推高 scaling 曲线斜率。

**2. Multi-modality 是 scaling 的必要条件**

Regression planner 在 multi-peak 数据上 mode collapse，跟早期 LLM 用 MSE 训 next-token 的问题一模一样。Diffusion / scoring 这种 "implicit multi-modal" architecture 才能吃下 scaling。这跟 LLM 中 MoE / mixture-of-gaussian 的发现一致。

**3. Reward > Demonstration（在有 anchor 前提下）**

呼应你在 RLHF 讨论中的观点：preference signal > demonstration。但**没有 anchor 的 pure reward 训练会 collapse**，这跟 RLHF 中 SFT → PPO 两阶段范式呼应。Sim-only reward = 没有 SFT 的 RL，必崩。

**4. Scaling curve 的 log-quadratic fit 是聪明工具**

借鉴 [Kaplan 2020](https://arxiv.org/abs/2001.08361)。但 N 只到 ~250K，data point 有限（5 sim round × 3 expert = 15 points），拟合 a 是否显著非零需要 confidence interval，paper 给了 error band 没做 statistical test。后续应该跑 10+ sim round 验证 a 是否真趋于 0。

**5. 3DGS rendering 是 bottleneck**

5 round sampling 是 "due to computational limits"。3DGS rasterize 一个 multi-view frame 在 H20 上几百 ms，5 round × 100K scenario × (T+2H) frame × multi-view = 几亿 frame 渲染。比 LLM pretraining 的 token throughput 限制更严重，因为 3DGS rasterizer 还没像 CUDA matmul 那样高度优化。后续 feed-forward GS ([PixelSplat](https://arxiv.org/abs/2312.00137)、[OmniRe](https://arxiv.org/abs/2409.12910)) 可能解锁更激进 scaling。

**6. 未来方向**（paper Sec E）
- Self-evolving pseudo-expert：用 pretrained planner 自己 rollout 再 filter，iteratively improve
- Diffusion-based traffic simulation 替代 IDM，更 diverse
- Unified world model ([DriveWorldModel](https://arxiv.org/abs/2510.12796)、Cosmos) 替代 3DGS sensor + IDM behavior 的两段式 pipeline
- Self-play：ego + agent 共享 policy，共同 evolve，类似 AlphaGo self-play

---

## 一句话总结

**SimScale 把 closed-loop RL 的 exploration 问题用 rule-based expert 在 offline 阶段预先 solve，把 sim 数据当作 long-tail OOD augmentation 喂给任意 E2E planner，并用 log-quadratic scaling law 实证：multi-modal 架构 + exploratory expert + reactive env + reward signal = predictable scaling。**

三个 takeaway：
1. **Scale sim, 仅仅只是 scale real**——real 到 100K 后 gain 趋平，sim 可以继续 scale
2. **架构决定 scaling shape**——regression saturate，diffusion / scoring 不会
3. **Reward > expert trajectory**——给定 anchor，让 planner 自己 explore 比 imitate single expert 更好

---

# SIMSCALE: 用 Real-World Simulation at Scale 解锁 E2E Driving 的 Scaling Potential

Karpathy 你好，这篇 paper 触及了你长期关心的几个核心问题：data scaling、imitation learning 的 covariate shift、以及 multimodal modeling 的必要性。我会按照 motivation → architecture → formula → experiments → intuition 的脉络展开。

---

## 1. Motivation: 为什么 E2E Driving 需要 Simulation

E2E planning 把 raw sensor → action 的 mapping 全部塞进一个网络，靠 imitation learning 从 human demonstration 学。问题是 human demonstration **几乎全是 common scenario**，safety-critical / OOD cases 在数据分布长尾里 underrepresented。这导致两个 well-known failure modes：

1. **Covariate shift / Causal confusion** (de Haan et al., Ross et al. DAgger): deploy 时 planner 进入 training distribution 没见过的 state，next-action 预测崩溃。reference: [DAGGER paper](https://arxiv.org/abs/1011.0686)
2. **Open-loop imitation 的 distribution mismatch**: 训练时 expert trajectory 永远在 human manifold 上，inference 时 planner 一旦偏离就再也回不来 (compounding error)。

经典解法是 closed-loop RL 或 DAgger，但 real-world closed-loop 太危险，simulator (CARLA、MetaDrive) 又有 sim-to-real gap。**3DGS-based simulation** 是近 2 年崛起的折中：用真实采集数据 reconstruct 出可控制 ego/agent pose 重新渲染的高保真场景，既保留了 real-world visual fidelity，又能 simulate OOD state。reference: [3DGS](https://repo.openmmlab.com/research/3dgs/kerbl3DGAussians-dig/kopal.html) | [Street Gaussians](https://arxiv.org/abs/2404.01912) | [HugSim](https://arxiv.org/abs/2412.01718)

SIMSCALE 的核心 insight: **如果只 scale real-world data 是 inefficient 的 (Naumann et al. CVPR'25 已证明 diminishing returns), 那么 scale simulation data on top of fixed real data 是否能 predictable 地提升 robustness?** 答案是 yes，并且不同 planner paradigm 的 scaling 行为差异显著。

---

## 2. Pipeline 整体架构

整个 system 分三块，对应 paper Sec 2.2–2.4:

```
Real-world Logs (navtrain 100K scenarios)
         │
         ▼
[3DGS Reconstruction Engine Φ]  ← background asset + movable vehicle assets
         │
         ▼
[Pseudo-Expert Scene Simulation] 
   Stage 1 (t=T → T+H): ego perturbation + reactive rollout (LQR + IDM)
   Stage 2 (t=T+H → T+2H): pseudo-expert 生成 demonstration
         │
         ▼  喂回去渲染 multi-view RGB
[Sim-Real Co-training]  ← random mixture sampling
   • LTF (regression)        : imitation loss only
   • DiffusionDrive          : imitation loss only  
   • GTRS-Dense (scoring)    : imitation + reward loss (or reward only)
```

关键设计: **behavior simulation 和 sensor rendering 解耦**。先在 abstract state space 跑 LQR + IDM 的 reactive rollout 算出所有 agent 的 future pose，再把整个 3DGS scene 按 pose 渲染。这避免了"边 rollout 边 render"的 latency，也允许在 reactive 阶段快速 filter 掉 collision / off-road 的 invalid trajectory。

---

## 3. 3DGS Data Engine Φ(K_t, E_t, {x_{i,t}, y_{i,t}, θ_{i,t}})

输入:
- `K_t`: camera intrinsics at timestep t
- `E_t`: camera extrinsics at timestep t，从 ego pose (x_{0,t}, y_{0,t}, θ_{0,t}) + ego-to-camera transform 得到
- `(x_{i,t}, y_{i,t}, θ_{i,t})`: 第 i 个 non-ego vehicle 的 position 和 yaw

输出: 该时刻 multi-view RGB observation。

**Preprocessing**:
- Multi-view camera 之间做 **exposure alignment** (用 projected LiDAR 点做 guidance)，对齐色彩 / 亮度，减少 3DGS 在多相机边界处 artifact。
- Colored LiDAR 点按 3D bbox 分组：static background + 每个 vehicle 一组，作为 Gaussians 的 initialization。

**Block-wise Reconstruction**:
- 整条 clip 太长 (几十秒到几分钟)，全量一起 reconstruct 计算成本爆炸。按 spatio-temporal range 切 block，每个 block 独立 reconstruct。
- Background 一个 static asset，每个 vehicle 一个 movable asset，runtime 时按输入 pose 放置 / 旋转。
- **过滤 PSNR < 27 的 block** (paper Sec 3.1 curation)，保证 novel view synthesis 质量。这条 threshold 在 ablation Tab. 10 验证: PSNR≥27 vs <27 在 EPDMS 上稳定有 0.5–1.5 的差距。

reference: [MTGS (Li et al. 2025)](https://arxiv.org/abs/2503.12552) 是这个 engine 的 codebase 基础。

---

## 4. Pseudo-Expert Scene Simulation (Algorithm 1)

这是 paper 最核心的算法部分。给定 training clip `d = (o_t, s_t, a_t)_{t=0}^{T+2H}`，要生成 valid (perturbed state, expert recovery) pair。

### 4.1 两阶段 rollout 结构

- **History window** T: 给 planner 看的过去帧
- **Stage 1 (t=T → T+H)**: **扰动阶段**。从 human log 的 t=T 时刻出发，让 ego 走 perturbed trajectory `ã_{t:t+H} = a_per`，把 ego 带到 OOD terminal state。期间其他 agent 用 IDM reactive 模拟。
- **Stage 2 (t=T+H → T+2H)**: **expert 阶段**。从这个 OOD state 出发，用 pseudo-expert π_exp 生成 demonstration `ã_{t:t+H}`。
- 总长度 T+2H，前 T 是真实 history，中间 H 是 perturbed history，最后 H 是 expert future。

直觉上：你强迫 ego "走偏"到一个 human 不太会去的位置，然后让 expert 教它"从那里怎么恢复 / 怎么继续开"。这就把 OOD recovery behavior 注入了训练数据。

### 4.2 Trajectory Perturbation

从 vocabulary `V_c` (16,384 clustered human trajectory) 中采样 candidate，并施加 hard constraint:

- 纵向偏移范围 `r_lon = ±20m`
- 横向偏移范围 `r_lat = ±2m`
- 相对 heading `|Δθ| ≤ 20°`
- EPDMS ≥ 0.8 (proxy 表示这个 perturbation 还算 physically valid)
- 用 interleaved grid `(δ_lon=5m, δ_lat=0.5m)` 做 spatially sparse sampling，避免 endpoint 集中

这套 threshold 是为了在 "diversity" 和 "plausibility" 之间 trade-off。太宽 → 生成 physically invalid trajectory 浪费 render 算力；太窄 → OOD coverage 不够。

### 4.3 两种 Pseudo-Expert 对比

**(1) Recovery-based Expert** (Eq. 1, 2) —— 保守派

公式 (1):
$$
\mathbf{m} = [\tilde{v}_t^x, \tilde{v}_t^y, \tilde{\theta}_t, \tilde{x}_{t+H-1}, \tilde{y}_{t+H-1}, \tilde{\theta}_{t+H-1}]
$$

- $\tilde{v}_t^x, \tilde{v}_t^y$: ego 在时刻 t 的速度 x/y 分量
- $\tilde{\theta}_t$: 时刻 t 的 heading angle
- $\tilde{x}_{t+H-1}, \tilde{y}_{t+H-1}, \tilde{\theta}_{t+H-1}$: horizon 末端 (t+H-1) 时刻的 pose
- 这是一个 6 维 compact matching vector，把 H-step trajectory 压缩成起末状态

公式 (2):
$$
\tilde{a}_{t:t+H} = \arg\min_{a \in \mathcal{V}_h} \|\mathbf{m}(a) - \mathbf{m}_r\|_1
$$

- $\mathcal{V}_h$: 完整 human vocabulary (103,288 trajectories, 比聚类 vocabulary V_c 大)
- $\mathbf{m}(a)$: 候选 trajectory a 的 matching vector
- $\mathbf{m}_r$: 当前 perturbed state 的 target vector
- $\|\cdot\|_1$: L1 距离

直觉：recovery expert **从 human trajectory bank 里检索最匹配的轨迹**，所以它产出的 demonstration **一定落在 human distribution 内**。这意味着 safety 高、realism 高，但 exploration 弱——同一 perturbed state 反复采样得到的 expert 几乎一样。

**(2) Planner-based Expert** —— 探索派

$$
\tilde{a}_{t:t+H} = \mathbf{P}(\tilde{s}_{t:t+H})
$$

- $\mathbf{P}$: privileged planner，本文用 PDM-Closed (Dauner et al. CoRL'23, nuPlan leaderboard SOTA rule-based planner)
- $\tilde{s}_{t:t+H}$: 当前 simulated 全局 state (含所有 agent 的 GT pose)
- 它直接用 ground-truth BEV state 跑 rule-based 优化，生成 trajectory

PDM-Closed 内部用 **Iterative Kalman Filter + 5 phase title selection**：先 sweep a family of kinematically feasible trajectory proposals (横向偏移、纵向速度组合)，每条 rollout 出来计算 cost (collision, progress, comfort, rules)，再用 Kalman 平滑选 final trajectory。reference: [PDM-Closed](https://arxiv.org/abs/2308.05731)

这个 expert **不局限于 human manifold**，会在 OOD state 给出"非人但 optimal"的 recovery，比如猛打方向盘绕障、急刹车让行。代价是 HC (history comfort) 和 EC (extended comfort) metric 下降——因为 PDM 的 jerk / accel 控制没有真实人那么平顺。这在 Tab. 7 里直接可见: planner-based 的 HC/EC 都比 recovery-based 低。

**Fig. 3 数据**: planner-based expert 五轮 sampling 累积 237K scenes, recovery-based 只 147K。原因是 recovery 的硬约束把很多 OOD state 直接判定为 "无法匹配到合理 human trajectory" 给 filter 掉了，而 planner 永远能给出一个 (哪怕是 suboptimal 的) feasible 解。

---

## 5. Sim-Real Co-training Formulation

### 5.1 Regression / Diffusion Planner (Eq. 3)

$$
\arg\min_\theta \mathbb{E}_{(a,o) \sim (\mathcal{D} \cup \mathcal{D}_{sim})} \big[ \mathcal{L}_{im}(a, \pi_\theta(\hat{a}|o)) \big]
$$

- $\theta$: planner 参数
- $a$: expert trajectory (real data 上是 human, sim data 上是 pseudo-expert)
- $o$: observation
- $\mathcal{D}$: real-world data
- $\mathcal{D}_{sim}$: simulation data
- $\mathcal{L}_{im}$: imitation loss (一般用 L1 / L2 on waypoints, 或 ADE/FDE)
- $\pi_\theta(\hat{a}|o)$: planner 输出的 trajectory distribution

这本质就是 **joint training with random sampling**: 每个 batch 按 1:1 (或某比例) 从 D 和 D_sim 各采一部分，shuffle 后一起 backward。没有复杂的 domain adaptation，没有 adversarial loss，没有 KL regularization。**simplicity is the point**——证明这种最朴素的 co-training 就足够。

### 5.2 Vocabulary Scoring Planner (Eq. 4)

$$
\arg\min_\theta \mathbb{E}_{(a,o,r) \sim (\mathcal{D} \cup \mathcal{D}_{sim})} \big[ \lambda \mathcal{L}_{im} + \mathcal{L}_r(r, \pi_\theta(\hat{a}|o)) \big]
$$

- $r$: reward signal (这里是 EPDMS 子指标，NC/DAC/DDC/TLC/TTC/EP/LK/HC/EC 等 sub-scores)
- $\mathcal{L}_r$: reward loss (一般 per-trajectory head 用 BCE / L1 distill sub-metrics)
- $\lambda$: balancing weight (paper 没给具体值, 在 GTRS codebase 默认 ~1)

GTRS-Dense 这种 scoring planner 在 vocab 上有 16384 条候选 trajectory，每条 trajectory 用 multiple scoring heads 预测其 reward。inference 时 argmax 选 score 最高的。

### 5.3 Reward-Only (Eq. 5) —— 一个反直觉的关键发现

$$
\arg\min_\theta \mathbb{E}_{(a,o,r) \sim \mathcal{D}} [\lambda \mathcal{L}_{im} + \mathcal{L}_r] + \mathbb{E}_{(o,r) \sim \mathcal{D}_{sim}} [\mathcal{L}_r]
$$

- 在 real data 上: 同时监督 imitation 和 reward
- 在 sim data 上: **只监督 reward, 不给 expert trajectory**

这是 paper 最有意思的设计之一。直觉: scoring planner 的 reward signal 已经表达了 "什么是好 trajectory"，给定 observation 后 planner 可以**自己在 vocab 中探索** reward 高的方向，不一定要被某一条 pseudo-expert trajectory 锁死。pseudo-expert (尤其 recovery-based) 本身可能就是 suboptimal 的，强行 imitate 反而把 planner 带偏。

实验结果 (Tab. 3): 
- ResNet34, sim + reward only: 46.9 (胜过 planner-based 的 46.1)
- V2-99: 48.0 (胜过 planner-based 的 47.7)

但只在 real data 上做 reward-only 训练会 collapse (38.8 vs 41.9 baseline 退化)。说明 **real data 的 imitation 是稳定的 anchor, sim 的 reward 是 exploration signal**。这是一个 "anchor + explore" 的训练范式，类比 RL 中的 behavior cloning + RL fine-tuning。

---

## 6. EPDMS Metric (Eq. 6)

$$
\mathrm{EPDMS} = \underbrace{\left(\prod_{m \in \mathcal{M}_{pen}} S_m\right)}_{\text{penalties}} \cdot \underbrace{\left(\frac{\sum_{m \in \mathcal{M}_{avg}} w_m S_m}{\sum_{m \in \mathcal{M}_{avg}} w_m}\right)}_{\text{weighted average}}
$$

- $S_m$: 第 m 个子指标的 score (0~1)
- $\mathcal{M}_{pen} = \{\text{NC, DAC, DDC, TLC}\}$: **硬约束 penalty**，任一为 0 → EPDMS 为 0
  - NC: No-at-fault Collisions
  - DAC: Drivable Area Compliance
  - DDC: Driving Direction Compliance
  - TLC: Traffic Light Compliance
- $\mathcal{M}_{avg} = \{\text{TTC, EP, LK, HC, EC}\}$: **软质量指标**加权平均
  - TTC: Time-to-Collision (越大越安全)
  - EP: Ego Progress (行驶距离)
  - LK: Lane Keeping
  - HC: History Comfort (jerk / accel 平顺性)
  - EC: Extended Comfort
- $w_m$: 各项权重 (paper 没明给, NAVSIM 默认)

**关键设计**: penalty 部分用乘法，weighted average 用加法。这模仿 LLM RLHF 中的 "hard constraint × soft quality" 评分结构。如果 planner 撞车 (NC=0)，无论其他指标多好都是 0 分。这逼迫 planner 优先保证 safety。

navhard 在此基础上还有两阶段 aggregation + reactive traffic + 排除 human 也会失败的 case，更严格。

reference: [NAVSIM](https://github.com/autonomousvision/navsim) | [Hydra-MDP++](https://arxiv.org/abs/2503.12820)

---

## 7. 实验结果分析

### 7.1 主结果 (Tab. 1, Tab. 2)

**navhard (244 + 4164 synthetic scenarios)**:
- LTF (ResNet34): 24.4 → 30.2 (+24%, 用 planner-based)
- DiffusionDrive: 27.5 → 32.6 (+19%, 用 planner-based)
- GTRS-Dense (ResNet34): 38.3 → 46.9 (+22%, 用 reward-only)
- GTRS-Dense (V2-99): 41.9 → 48.0 (+15%, 用 reward-only) ← **新 SOTA**

**navtest (12146 real scenarios)**:
- 所有模型 +0.8 ~ +2.9 EPDMS

直觉解读: navhard 改进远大于 navtest，因为 navhard 本来就是 OOD / safety-critical 集合，simulation data 的 distribution 跟它更对齐。navtest 是普通 driving，real data 已经覆盖得很好，sim 的边际效益递减。

**weaker baseline 受益更多**: LTF / DiffusionDrive 这种 56M / 61M 的小模型 baseline 提升 19–24%, V2-99 这种 83M 大模型只提升 15%。直觉: 弱模型有更多 "unlocked potential"，sim data 帮它们 explore 出 latent 能力；强模型本身已经接近 data 上限，sim 主要填补 long-tail。

### 7.2 Data Scaling Curves (Fig. 4) + Log-Quadratic Fit

公式 (7):
$$
S(N) = a \log^2(N) + b \log(N) + c
$$

- $N$: total data size (sim + real)
- $S(N)$: performance at N
- $a, b, c$: 拟合参数

公式 (8): 用非线性最小二乘拟合
$$
(a^*, b^*, c^*) = \arg\min_{a,b,c} \sum_{i=1}^M (S_i - S(N_i; a,b,c))^2
$$

- $S_i$: 第 i 个 data point 的 observed performance
- $M$: data point 数量
- $N_i$: 第 i 个 data point 对应的 total data size

**Interpretation**:
- $a \to 0$: 退化为 $S(N) = b \log N + c$，即 linear log-scaling (LLM 经典 scaling law 形式)
- $a < 0$: 抛物线开口向下，存在 inflection point / saturation
- $a > 0$: 抛物线开口向上 (不会出现在合理 setting)

Fig. 4 四张子图:
- (a) LTF-ResNet34: planner-based 在 N 较小时 log-linear，达到 1:1 sim:real 后开始 plateau 下降 (a<0)。recovery-based 一直 plateau 低。
- (b) DiffusionDrive-ResNet34: planner-based **近似 perfect log-linear**, 没有明显 saturation。这是 paper 强调的 "multi-modality sparks scaling"。
- (c)(d) GTRS-Dense: reward-only > planner-based > recovery-based，scaling 趋势明显，V2-99 (83M) 比 ResNet34 (67M) 斜率更陡——**大模型 + 多模态 scoring = 更好的 scaling**。

这条公式直接借鉴 Kaplan et al. 2020 的 neural scaling laws 思路。reference: [Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361)

### 7.3 Key Findings 详解

#### Finding 1: Pseudo-Expert Should Be Exploratory

Recovery-based expert 在所有 planner 上都比 planner-based 早 saturate、性能低。原因:
- Recovery 总是回到同一个 human trajectory bank，sim data 增多但 trajectory 多样性不增多
- Planner-based 在不同 sim round 给出不同的 (exploratory) 行为，sim data 增多时 diversity 真的在涨

只有 small-data regime 下 recovery 占优——因为它 distribution 跟 real data 对齐最好，模型容易 fit。

直觉 (类比 RL): 这就是 "exploration vs exploitation" 的经典 trade-off。Recovery 是 greedy exploitation of human manifold，Planner 是 exploration beyond manifold。在 scaling regime 下 exploration wins，因为 reward signal 才是真正的 objective。

#### Finding 2: Multi-modality Modeling Sparks Scaling

LTF (single-mode regression) 在 sim:real = 1:1 后开始 degrade。DiffusionDrive (multi-mode diffusion) linear 改进。

为什么? 同一个 perturbed state 在不同 sim round 由 planner-based expert 会给出 **多个 valid 的 recovery trajectory** (绕左 / 绕右 / 刹停)。这形成 **multi-peak distribution**:
- Regression loss (一般 L1/L2 on waypoint) 假设 unimodal target，multi-peak 下 regression 会 average 这些 mode，给出 "绕左 + 绕右平均 = 撞上去" 的 mode-confusion 解
- Diffusion 用 score matching / denoising，能正确 capture 多峰分布，每个 mode 都学

类比: 这是 diffusion policy (Chi et al. RSS'23) 在 robotics manipulation 上的核心 motivation 的 driving 版本。reference: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

#### Finding 3: Reward Is All You Need

对于 scoring-based planner，sim 上 reward-only 比 reward + expert trajectory 还好。直觉:
- Pseudo-expert trajectory 是 **single sample from optimal distribution**，强行 imitate 会让 planner 把这个特定 sample 当作唯一答案
- Reward signal 描述 **entire optimal manifold**，让 planner 自己在 vocab 中探索，等价于 infinite-sample supervision
- 这跟 RL 中的 "preference learning" / "reward model" 思路一致

但要 **real data 上的 imitation 作 anchor** (Tab. 3 显示 real-only reward 训练会崩)。直觉: real anchor 防 distribution drift，sim reward 推动 exploration。

#### Finding 4: Reactive Simulation 重要 (Tab. 4)

Non-reactive (其他 agent 按 log 走) vs Reactive (其他 agent 用 IDM 对 ego 响应):
- Non-reactive 2 round: 141K samples → EPDMS 43.7 / 45.6
- Reactive 2 round: 120K samples → EPDMS 44.4 / 46.7
- Reactive 3 round: 167K samples → EPDMS 45.0 / 47.9

Non-reactive 样本多但效果差。直觉: 如果其他 agent 不响应 ego perturbation，ego 撞上去也是 "log 上的"，这种数据没有交互信息，planner 学不到"我偏离了，别人会怎么反应"的因果。Reactive 模拟虽然简单 (IDM)，但提供了 **闭环因果 feedback**。

这也是 closed-loop evaluation 比 open-loop 严格的根本原因——action 影响 state，state 反过来影响 action。reference: [NAVSIM paper](https://arxiv.org/abs/2411.17044)

### 7.4 其他 ablations

**Multi-Expert Ensemble (Tab. 9)**: 把 recovery + planner + reward-only 三个 model 的 sub-score 简单平均，ResNet34: 47.7 (+0.8 over best single), V2-99: 50.9 (+2.9 over best single)。三个 expert 互补，集成效果优于 backbone 升级 (V2-99 vs ResNet34 提升 +3.6)。

**Visual Fidelity (Tab. 10)**: PSNR≥27 vs <27，每 round +0.5~1.5 EPDMS。说明 sim visual 质量直接传导到 planner 性能。

**Scaling with Varying Real Data (Fig. 9)**: 在 10K / 20K / 50K / 100K real data 上固定 sim:real ratio。real data 越少，sim gain 越大 (10K real 时 ResNet34 +22.4%)。但 100K real 时 gain **没有 narrowing**，说明 sim 的 marginal value 不随 real data 增多而消失——这是一个非常 strong 的 scaling claim。

---

## 8. 与 Online RL 的对比 (Supplementary Fig. 6)

Paper Sec A.4 给了一个对比图，思路:
- **3DGS-based Online RL** (e.g. RAD, NeurIPS'25): planner 在 3DGS env 中 closed-loop 跑，每次 action 触发 reward，policy gradient 更新。需要可微 / fast rollout, 训练不稳定。
- **SimScale (Offline RL with Sim Data)**: 先离线 generate 一大批 (state, expert action, reward) tuple, 再用 IL / reward learning 离线训练。更稳定, 可支持任意 planner paradigm。

SimScale 本质上是把 **online RL 的 hard part (exploration) 用 rule-based expert 在 offline 阶段提前做完**，剩下的训练退化成 supervised IL/reward regression。这跟 AlphaGo 的 "policy network pretrain + RL fine-tune" 思路、Anthropic 的 RLAIF 思路都有共鸣。reference: [RAD](https://arxiv.org/abs/2503.07152) | [GR00T N1](https://arxiv.org/abs/2503.14734)

---

## 9. Intuition 总览 & 与 LLM Scaling 的类比

把整个 paper 压成一句话: **scaling simulation data on top of fixed real data, 是给 E2E driving 注入 "long-tail OOD experience" 的 scalable 途径, 且其 scaling law 的 shape 由 planner 架构决定**。

类比 LLM scaling:
- Real driving data = pretraining corpus (覆盖广, 长尾稀疏)
- Simulation data with OOD perturbation = RL fine-tuning / synthetic data (针对 long-tail 的 targeted amplification)
- Recovery expert = behavior cloning from human demonstrations (safe but limited exploration)
- Planner-based expert = RL with reward (exploratory but suboptimal in style)
- Reward-only sim training = RLAIF / preference learning (let model explore objective, not imitate trajectory)
- Multi-modal planner = MoE / mixture distribution (capture multi-peak target)

**Karpathy 视角的延伸观察**:

1. **Causal confusion 在 E2E driving 是 open problem**。SIMScale 用 "perturb + expert recovery" 的两阶段 rollout 制造 artificial covariate shift 训练数据，相当于 DAgger 的 offline 近似。但 DAgger 需要 online expert 修正，sim 中用 PDM-Closed 当 expert 是次优解——后续如果用更强的 learning-based BEV planner (e.g. GameFormer, Diffusion-ES) 当 expert，scaling 曲线应该更陡。

2. **Multi-modality 是 scaling 的必要条件, 这跟 LLM 中 MoE / mixture-of-gaussian 的发现一致**。Regression planner 在 multi-peak 数据上 "averaging to collision" 的失败模式, 跟早期 LLM 用 mean squared loss 训 next-token 的 mode collapse 一模一样。Diffusion / scoring 这类 "implicit multi-modal" architecture 才能吃下 scaling。

3. **Reward is all you need (在 anchor 存在前提下)**, 这呼应了你之前在 RLHF 讨论中的观点: preference signal > demonstration。但**没有 anchor (real imitation) 的 pure reward 训练会 collapse**, 这跟 RLHF 中 SFT → PPO 的两阶段范式呼应。Sim-only reward = 没有 SFT 的 RL, 必崩。

4. **Scaling curve 的 log-quadratic fit** 是个聪明工具——直接借鉴 Kaplan 2020。但 N 只到 ~250K, 数据点有限 (5 sim round × 3 expert = 15 points), 拟合 a 是否显著非零需要 confidence interval, paper 给了 error band 但没做 statistical test。后续应该跑 10+ sim round 验证 a 是否真的稳定趋于 0 (即真正 log-linear)。

5. **3DGS rendering 是 bottleneck**。Paper 提到 5 round sampling 是 "due to computational limits"。3DGS rendering 一个 multi-view frame 在 H20 上要几百 ms, 5 round × 100K scenario × (T+2H) frame × multi-view = 几亿 frame 的渲染。这比 LLM pretraining 的 token-throughput 限制更严重, 因为 3DGS rasterizer 还没像 CUDA kernel 一样高度优化。后续 feed-forward GS (PixelSplat, OmniRe) 可能解锁更激进 scaling。

6. **未来方向 (paper Sec E)**: 
   - Self-evolving pseudo-expert (用 pretrained planner 自己 rollout 再 filter)
   - Diffusion-based traffic simulation (替代 IDM, 更 diverse)
   - Unified world model (DriveWorldModel, Cosmos) 替代 3DGS sensor + IDM behavior 的两段式 pipeline
   - Self-play (ego + agent 共享 policy, 共同 evolve)

reference 补充:
- [DiffusionDrive](https://arxiv.org/abs/2411.15149)
- [GTRS-Dense](https://arxiv.org/abs/2506.06664)
- [VADv2 (vocabulary scoring)](https://arxiv.org/abs/2402.13293)
- [Vid2Sim](https://arxiv.org/abs/2501.06680)
- [NAVSIM v2 / navhard](https://github.com/autonomousvision/navsim)
- [nuPlan](https://github.com/motional/nuplan-devkit)
- [SimScale 项目页](https://opendrivelab.com/SimScale) | [GitHub](https://github.com/OpenDriveLab/SimScale)

---

## 10. 一句话提炼

**SimScale 把 closed-loop RL 的 exploration 问题用 rule-based expert 在 offline 阶段预先 solve, 把 sim 数据当作 long-tail OOD augmentation 喂给任意 E2E planner, 并用 log-quadratic scaling law 实证: multi-modal 架构 + exploratory expert + reactive env + reward signal = predictable scaling**。

核心 takeaway 三个:
1. **Scale sim, not real**——real data 到 100K 后 gain 趋平, sim data 可以继续 scale
2. **架构决定 scaling shape**——regression 会 saturate, diffusion / scoring 不会
3. **Reward > expert trajectory**——给定 anchor, 让 planner 自己探索比 imitate single expert 更好
