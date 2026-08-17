---
source_pdf: RAP 3D RASTERIZATION AUGMENTED END-TO-END PLANNING.pdf
paper_sha256: 79b5a3f272aa454c7d00435a7fa5ef44e9268c80fbcaf406d0fab0d5cadeb8d4
processed_at: '2026-08-11T20:51:49-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 RAP

Okay Andrej, 我换种讲法, 像跟你喝咖啡聊天那样讲。

---

## 一句话版本

这篇 paper 说: 训练自动驾驶 planner, 跟 real road 长得一模一样的 synthetic image 其实没那么重要, 重要的是**场景的结构和规模**。所以与其烧钱做 photorealistic rendering, 不如用最便宜的 rasterization (就是画线框图) 狂造数据, 然后在 feature space 里把 synthetic 和 real 对齐就行。

---

## 为什么这个问题值得做

先说 motivation。E2E driving 现在的 standard recipe 是 imitation learning: 拿一堆人类 expert 开车的 log, 学一个从 camera image 到 future trajectory 的 mapping。Open-loop 评估看着分很高, 一上 closed-loop 就崩。

为什么崩? 因为 IL 只见过 expert 的 "happy path"。Expert 从来没偏离过 lane, 从来没突然被 cut in, 从来没错过一个 turn。Planner 一旦 closed-loop 里犯个小错 (比如稍微偏了 30cm), 它从来没见过这种 "我偏了怎么 recover" 的 training example, 就懵了, 错误越滚越大, 最后撞了或者冲出 road。

这个是经典问题, [Ross et al. 2011](https://arxiv.org/abs/1011.0682) 的 DAgger paper 早就讲过, 叫 covariate shift。

业界的解法主要是两条路:

**第一条**: 用 photorealistic simulator (CARLA, LGSVL, MetaDrive) 生成 synthetic scene, 让 planner 在 simulator 里 closed-loop 训练, 见各种 recovery scenario。问题: simulator 贵, 造一个真实多样的 world 要大量手工 3D asset, traffic behavior 也很难模拟得真实。

**第二条**: 最近火的 NeRF / 3D Gaussian Splatting (3DGS), 把 real driving log 重建一个 photorealistic digital twin, 然后在 twin 里 render 各种 counterfactual viewpoint, 看 planner 偏离之后会看到什么。代表工作 [HUGSIM](https://arxiv.org/abs/2412.01718), [NeuroNCAP](https://arxiv.org/abs/2409.08047), [RealEngine](https://arxiv.org/abs/2505.16902)。问题: NeRF/3DGS 优化慢, 渲染也慢, scale 起来烧钱, 而且偏出 logged trajectory 太远会有 visual artifact (浮空、模糊), 所以现在主要用来做 evaluation, 训练用得少。

RAP 的 insight: 这两条路都 over-engineered 了。Driving decision 本质靠的是 geometry (车在哪、lane 在哪)、semantics (哪个是 pedestrian、哪个是 cone)、dynamics (谁要变道、谁在减速), 跟 texture、lighting、shading 没什么关系。你在 GTA 里学开车能迁移到 real road, 婴儿看简笔画也能认出 "车", 说明视觉里 texture 是冗余的, structure 才是核心。

所以 RAP 的策略: **画线框图就行, 但要画得快、画得 scalable, 然后在 feature space 对齐 sim 和 real**。

---

## RAP 具体怎么做

### 3D Rasterization — 就是画线框图

输入是 nuPlan 的 annotation (每个 frame 都标了 lane polyline、agent bounding box、traffic light)。RAP 把这些 annotation 直接 project 到 camera view, 画出来。

具体讲, 每个 frame 有两类东西:

**静态 map**: lane、crosswalk, 用 polyline 表示。每条 polyline $\mathbf{P}_k$ 是 $\mathbb{R}^{n_k \times 3}$ 的矩阵, $n_k$ 个 3D vertex, world coordinate 下 $(x, y, z)$。下标 $k$ 是第 $k$ 条 polyline。

**动态 agent**: vehicle、pedestrian、cone、barrier, 用 oriented cuboid 表示。第 $i$ 个 agent 是 $\mathcal{B}_i = (l_i, w_i, h_i, \mathbf{T}_i)$, $l_i$ length, $w_i$ width, $h_i$ height, $\mathbf{T}_i \in SE(3)$ 是 6-DoF pose in world frame。然后 $\mathbf{C}_i = \mathbf{T}_i [\pm l_i/2, \pm w_i/2, 0, h_i]^\top$ 算出 8 个 3D corner。

Project 用标准 pinhole camera model:
$$\mathbf{u}_{uv} = K \mathbf{T}_{w \to c} \tilde{\mathbf{p}}_w$$

- $K \in \mathbb{R}^{3\times3}$: camera intrinsics
- $\mathbf{T}_{w \to c} \in SE(3)$: world-to-camera extrinsics
- $\tilde{\mathbf{p}}_w \in \mathbb{R}^4$: 3D point 加 1 变 homogeneous
- $\mathbf{u}_{uv}$: 投影结果, 除以 $u_z$ 拿到 pixel $(u, v)$ 和 depth $u_z$

Rasterize 时用 depth-aware compositing, 每个 fragment 存 depth $d$, 用 fading weight $\alpha = \max(0, 1 - d/d_{\max})$ blend。越近越不透明, 越远越淡, 模拟大气透视。跨 view boundary 的 polygon 用 [Sutherland-Hodgman 1974](https://dl.acm.org/doi/10.1145/360767.360776) 的 clipping 算法切掉。

**结果**: 你看到的 "image" 就是黑色背景上画着彩色 lane、彩色 cuboid、彩色 traffic light 的线框图。完全没 texture, 没 lighting, 没 shadow。

**关键 evidence (Figure 4)**: 把 rasterized image 和 real image 都喂给 frozen DINOv3 encoder ([Simeoni et al. 2025](https://arxiv.org/abs/2508.10104)), 做 PCA 可视化 feature。发现两者的 feature structure 在 PCA 空间里长得非常像! 这说明对于 DINOv3 这种 trained on natural image 的 encoder, 简笔画输入激发的 feature 跟 real image 是 qualitatively consistent 的。这是整个 paper 的 intuition 支柱。

### Data Augmentation — 怎么用 rasterization 造数据

光画 ego view 没意思, 关键是能用 rasterization 造出训练数据里缺的 scenario。

**Augmentation 1: Recovery-oriented perturbation**

直接戳 IL 的痛点。把 expert trajectory $\tau^*(t)$ 加扰动:
$$\tilde{\tau}(t) = \tau^*(t) + \delta_{\text{lat}}(t) + \delta_{\text{long}}(t) + \epsilon_t$$

- $\delta_{\text{lat}}, \delta_{\text{long}}$: 从预定义 range 采样的 lateral/longitudinal offset (模拟 ego 偏了)
- $\epsilon_t$: Gaussian noise

把扰动后的 trajectory 拿去 rasterize, 生成 "ego 已经偏出 expert path 50cm" 的 counterfactual scene。Supervision 还是用原 expert trajectory, 所以 planner 学的是 "我现在位置偏了, 应该往 expert trajectory 回去"。这就直接教 planner 怎么 recovery。

这个 augmentation 在 NAVSIM v1 (open-loop) 上完全没提升 (92.5 → 92.5), 但在 NAVSIM v2 (pseudo-closed-loop, 用 3DGS simulate deviation 后的 view) 上从 32.5 → 36.9, 提升 4.4 分 (Table 6)。

这个 ablation 重要在哪? 它说明 **augmentation 的价值在 open-loop metric 上根本看不出来**, 很多 augmentation paper 报 open-loop 数字没用, 真正价值只在 closed-loop。这给 field 一个 warning。

**Augmentation 2: Cross-agent view synthesis**

nuPlan 每个 scenario 有 $n$ 个 agent 的 trajectory。本来只 render ego view, 现在把 ego trajectory 替换成 agent $j$ 的 trajectory, camera 参数不动, 就免费拿到 agent $j$ 视角的 driving scene。一个 scenario 本来只有 1 个 training sample, 现在变成 $n$ 个。

总规模: 85k paired real-raster + 8.5k perturbed + 272k ego raster + 200k cross-agent raster ≈ 500k+ synthetic samples。

**Scaling law 实验 (Figure 6)**: 从 85k real 开始, 加 1k / 10k / 100k / 500k / 1000k cross-agent raster。MinADE 拟合出 $y = -0.021 \ln(x) + 1.2173$, $R^2 = 0.9942$。完美 log-scaling, 跟 [Baniodeh et al. 2025](https://arxiv.org/abs/2412.02689) 在 real data 上发现的 scaling law 一致。这说明 raster augmentation 真的在贡献 planning-relevant signal, 跟 real data 一样有 scaling behavior, 哪怕是 other agent 的视角。

### Raster-to-Real (R2R) Alignment — 怎么让 sim feature 跟 real feature 对齐

虽然 Figure 4 说 raster 和 real feature 已经很像, 但要让 planner 在 real image 上 well-perform, 还得 explicitly align。RAP 用两个 level 的 alignment。

**Spatial-level alignment**

对每个有 paired (real $x^r$, raster $x^s$) 的 sample:
$$F^r = \phi(x^r), \quad F^s = \phi(x^s), \quad F^r, F^s \in \mathbb{R}^{N \times d'}$$

- $\phi(\cdot)$: visual encoder
- $N$: spatial location 数 (ViT patch tokens 或 CNN feature map positions)
- $d'$: projected feature dim

Loss 是 MSE, 但方向很重要 — **Real-to-Raster**, 也就是 freeze $F^s$, 只 update $F^r$:
$$\mathcal{L}_{\text{spatial}} = \frac{1}{N} \sum_{j=1}^{N} \| F^r_j - F^s_j \|_2^2$$

为什么不反过来? Section A.1 Table 7 ablation:
| Alignment | MinADE ↓ |
|---|---|
| Raster-to-Real | 1.12 |
| Symmetric | 1.14 |
| Real-to-Raster | **1.02** |

Real-to-Raster 最好。直觉: raster feature 来自 clean annotation, 没有 distracting detail (比如树、招牌、云), 它是个干净的 "structural scaffold"。让 real feature 往它靠, 等于给 real feature 加个 structural prior, 让 encoder 学到 "geometric structure 是 task-relevant, 别的 detail 是 noise"。反过来 Raster-to-Real 会把 raster feature 污染成 real, 丢失 scaffold 的 clean 优势。

同时 task loss 一直在 real feature 上起作用 (planning head 接 real feature), 保证 task-relevant detail 不被 alignment 抹掉。Figure 7 的 ablation 证明: 用同一个 trained model, condition on real image 时能识别 unannotated "Keep Left" sign 和 LED arrow; condition on raster image 时识别不了。说明 real feature 上的细节信息没被 alignment 抹掉, 只是 geometry 的 prior 被强化了。

**Global-level alignment**

Spatial alignment 需要 paired data, 但 cross-agent raster 和 perturbed raster 是 raster-only, 没 paired real。对这些 unpaired synthetic data, 用 [Ganin & Lempitsky 2015](https://arxiv.org/abs/1409.7495) 的 unsupervised domain adaptation。

Global representation $g \in \mathbb{R}^{d'}$ 由 feature map $F \in \mathbb{R}^{N \times d'}$ average pool 得到。Domain classifier $D$ 预测 $g$ 来自 real 还是 raster:
$$\mathcal{L}_{\text{global}} = -\mathbb{E}_{(g,y)} [y \log D(g) + (1-y) \log(1 - D(g))]$$

- $y \in \{0, 1\}$: domain label
- $D(g) \in [0, 1]$: classifier 输出

Gradient Reversal Layer (GRL) 插在 $D$ 之前。Forward 时 identity, backward 时 gradient 乘 $-\lambda$, 让 encoder 学 domain-invariant feature (maximize confusion), classifier 学区分 domain (minimize error)。Adversarial 博弈。

$\lambda$ 用 annealing schedule:
$$\lambda(p) = 0.1 \cdot \left(\frac{2}{1 + \exp(-\gamma p)} - 1\right), \quad p \in [0, 1], \gamma = 10$$

- $p$: training progress, 从 0 到 1
- $\gamma=10$: 控制 annealing 速度

直觉: 训练初期 $\lambda$ 很小, 让 task loss 先 dominate, 模型先学会做 task; 后期 $\lambda$ 逐渐增大, 开始 push domain confusion。

**Overall loss**:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_s \mathcal{L}_{\text{spatial}} + \lambda_g \mathcal{L}_{\text{global}}$$

$\lambda_s = 0.002, \lambda_g = 0.1$。$\lambda_s$ 很小, 因为 MSE 容易 dominate, 必须压低权重。

---

## 为什么这套设计 work

让我帮你想清楚每个 component 的 role:

**3D Rasterization** 提供的是 clean、scalable、controllable 的 "task-relevant scaffold"。它丢掉了 real image 里的 texture/lighting/detail, 但保留了 driving 决策真正需要的 geometry + semantics + dynamics。而且因为它就是 annotation project, 没有任何 neural network optimization, 速度快到能 generate 500k+ sample, 这是 NeRF/3DGS 完全做不到的。

**Recovery perturbation** 直接补 IL 的根本缺陷: 没见过 recovery scenario。它生成的 counterfactual scene 让 planner 在 training 里见过 "ego 偏了" 的 state, 学到怎么 recover 回 expert path。这个收益在 open-loop eval 上看不出来 (Table 6), 但在 pseudo-closed-loop eval (NAVSIM v2) 上巨大。

**Cross-agent view synthesis** 把数据规模乘以 $n$ (agent 数量), 而且每个新 viewpoint 提供的 interaction pattern 都不同 — 同一个 scenario 从 ego 角度看 vs 从被 cut-in 的车角度看, 学到的 interaction model 不一样。Figure 6 的 log-scaling 证明这些 secondary viewpoint 真的有 planning-relevant signal。

**R2R Alignment** 是 sim-to-real 的 bridge。Figure 4 的 DINOv3 PCA 已经证明 sim 和 real feature 天然相近, 所以 alignment 不是要强行对齐两个完全不同的 modality, 只是 refine 一下已有的相似性。Real-to-Raster 方向让 real feature 学到 raster 的 clean structure prior, 同时 task loss 保证 real feature 上的 task-relevant detail 不丢。Global alignment 用 GRL 处理 unpaired synthetic data, 把 raster-only sample 也利用上。

**整体 logic**: 用 rasterization 大规模造 task-relevant scenario, 用 feature alignment 让 sim-to-real 不掉点, 用 task loss 保证 real 上的 detail 保留。三个 component 各司其职, 没有 over-engineering。

---

## 实验数据的几个关键 take-away

**NAVSIM v1 (Table 1)**: RAP-DINO 拿 PDMS 93.8, 超过 Centaur (92.1) 和 iPad (91.7)。更有意思的是 RAP-iPad 比 iPad 提升 +0.7, RAP-DiffusionDrive 比 DiffusionDrive 提升 +3.2, 说明 RAP framework 是 model-agnostic 的, 套到现成 planner 上都 work。

**NAVSIM v2 (Table 2)**: RAP-DINO 拿 EPDMS 36.93, 比 LTF (23.12) 高 13.81 分。这个 gap 巨大, 主要来自 recovery perturbation (Table 6, +4.4 分) 和 cross-agent synthesis (Figure 6 log-scaling) 的组合收益。

**WOD-E2E Driving (Table 3)**: dataset 专为 long-tail 设计 (construction detour, pedestrian accident, freeway obstacle), 频率 < 0.003%。RAP-DINO (RFS Overall 8.04) 击败 Poutine (3B VLM, 7.99)。这点很 significant: lightweight rasterization + alignment 在 long-tail 上比 massive VLM 更高效。

**Bench2Drive (Table 4)**: 真正的 CARLA closed-loop eval, 220 routes, 每条有 safety-critical event。RAP-ResNet (只有 29M params) 拿 Success Rate 37.27%, Driving Score 66.42, 超过 UniAD-Base (16.36%, 45.81) 和 DriveTransformer (35.01%, 63.46)。

**Ablation — 3D Rasterization Design (Table 5)**:
| Face | Depth Decay | Background | MinADE ↓ |
|---|---|---|---|
| Colored | Yes | Black | **0.91** |
| Transparent | Yes | Black | 0.98 |
| Colored | No | Black | 1.05 |
| Colored | Yes | Natural | 1.33 |

三个 design choice 全 significant:
- Colored face > Transparent (+0.07): 实心 colored face 给 encoder 明确 object boundary, transparent 反而模糊 semantic
- Depth decay > No (+0.14): atmospheric fading 给 encoder 距离感, 没有 depth cue 时远近 agent 视觉无差异, encoder 难学 depth-aware planning
- Black bg > Natural bg (+0.42!): 巨大差异。Natural sky-ground split 引入 distracting pattern, encoder 浪费 capacity 学背景; black bg 让 encoder 聚焦 geometric primitive。这是 paper thesis 的最强 evidence — driving planner 不需要 photorealism

**Ablation — R2R Alignment (Figure 5)**: 固定 total size, real data 从 100% 逐步降到 1%, 用 raster 替。无论什么 ratio, Spatial+Global alignment 都优于 no alignment 和只 spatial。而且 50% real + alignment 比 100% real 还好! 高质量 raster feature 比 real feature 更 clean, 作为 augmentation 提供额外 signal。

---

## 我自己看完的几个联想

**联想 1: 跟 NeRF/3DGS 路线的对比**

最近一年 E2E driving field 的 hot direction 是 photorealistic digital twin。代表工作 [RAD](https://arxiv.org/abs/2502.13144) 用 3DGS 造数据 train RL policy, [HUGSIM](https://arxiv.org/abs/2412.01718) 做 closed-loop sim, [RealEngine](https://arxiv.org/abs/2505.16902) 用 NeRF 做 realistic context sim。这些都 fidelity 高但 scale 起来贵。RAP 的 stance 是 "fidelity 不是关键, scale 才是", 用最便宜的 rasterization 拿到 scale, 在 feature space 弥合 fidelity。这跟 LLM field 的 "data quality > model fancy" 的 scaling law 直觉一致。

**联想 2: 跟 VLM-based driving 的对比**

最近 [Poutine](https://arxiv.org/abs/2506.11234) (3B VLM), [DriveVLM](https://arxiv.org/abs/2402.12289), [AutoVLA](https://arxiv.org/abs/2506.13757) 这些 VLM-based driving 拿大 model 做 reasoning, 在 long-tail 上看着不错。RAP-DINO 在 WOD-E2E (专门为 long-tail 设计的 benchmark) 上击败 Poutine, 说明 lightweight geometric prior + scale data > massive VLM reasoning, 至少在 closed-loop planning 上。

**联想 3: 跟 representation alignment 的 connection**

R2R alignment 跟最近 [Yu et al. 2024](https://arxiv.org/abs/2410.06940) 在 diffusion transformer 上的 representation alignment 工作 (RA-CFG) 思路一致: alignment 应该在 well-structured feature space 做, 不在 raw pixel/latent space 做。RAP 把这个 idea 用到 sim-to-real 上, 取得了 Raster-to-Real → Real-to-Raster 的方向反转, 这是个 fresh 的 design choice。

**联想 4: Scaling law 在 synthetic data 上同样成立**

Figure 6 的 $R^2 = 0.9942$ log-scaling 拟合给我震撼。跟 [Baniodeh et al. 2025](https://arxiv.org/abs/2412.02689), [Zheng et al. 2024](https://arxiv.org/abs/2412.02689) 在 real data 上发现的 scaling law 一致, 说明 good synthetic data 在 representation learning 上贡献 predictable 且 predictable 的 gain。这给 future work 一个 quantitative framework: 如果你的 raster augmentation quality 够好, 你能 predict 加多少 sample 会拿到多少 gain。

**联想 5: Limitation**

作者自己承认 RAP 还在 IL paradigm 内, 继承 causal confusion 等 IL 固有问题。Future direction 是把 3D rasterization 扩展成 full simulator 支持 closed-loop RL — 这样可以从 reward signal 而不只是 expert demonstration 学习, 突破 IL 的根本局限。我同意, 这是个 promising direction, 而且 rasterization 速度快, 适合做 RL 的 fast rollout。

**联想 6: Black background > Natural background 这个 ablation**

这个看似反直觉, 其实是 paper thesis 的最强 evidence。Driving planner 不需要 photorealism, 需要 clean geometric + semantic signal。Natural background 引入 distracting pattern, 浪费 encoder capacity。这跟你在 [nanoGPT](https://github.com/karpathy/nanoGPT) 里强调的 "remove unnecessary complexity, keep what matters" 的工程品味一致。

---

## Reference

- Project page: https://alan-lanfeng.github.io/RAP/
- NAVSIM: https://github.com/autonomousvision/navsim
- nuPlan: https://www.nuscenes.org/nuplan
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- 3D Gaussian Splatting: https://repo.samoa.gwdg.de/cyrus/lastdoctoralenrichment/-/raw/main/3D_Gaussian/3DGAUSSIAN.pdf
- DINOv3: https://arxiv.org/abs/2508.10104
- iPad: https://arxiv.org/abs/2505.15111
- DiffusionDrive: https://arxiv.org/abs/2501.01879
- Poutine (3B VLM): https://arxiv.org/abs/2506.11234
- HUGSIM: https://arxiv.org/abs/2412.01718
- NeuroNCAP: https://arxiv.org/abs/2409.08047
- RealEngine: https://arxiv.org/abs/2505.16902
- RAD: https://arxiv.org/abs/2502.13144
- Gradient Reversal Layer: https://arxiv.org/abs/1409.7495
- Sutherland-Hodgman Clipping: https://dl.acm.org/doi/10.1145/360767.360776
- DAgger (Ross et al.): https://arxiv.org/abs/1011.0682
- WOD-E2E: https://arxiv.org/abs/2510.26125
- RA-CFG (representation alignment): https://arxiv.org/abs/2410.06940
- Scaling laws (Baniodeh): https://arxiv.org/abs/2412.02689

---

总结: RAP 的 elegance 在于它用最便宜的方式 (annotation-based rasterization) 生成最 task-relevant 的 augmentation (recovery + cross-agent), 然后用最自然的 alignment 方式 (feature space, Real-to-Raster direction) 弥合 sim-to-real gap。整个 pipeline 没有 fancy neural rendering, 但每个 design choice 都精准打击 E2E driving 的真实痛点。这种 "做减法" 的工程品味, 跟你一直强调的 "identify the bottleneck, remove unnecessary complexity" 一脉相承。

---

# RAP: 3D Rasterization Augmented End-to-End Planning 深度讲解

Hey Andrej, 这篇paper我读得很兴奋, 因为它戳中了一个 E2E driving 领域长期被忽视的痛点。让我从 intuition 层面一路拆到 technical detail。

---

## 1. 核心洞察 (Core Insight)

这篇 paper 的 central claim 可以一句话概括: **对于 E2E planner 的训练, photorealism 是 over-engineered 的, 真正重要的是 semantic fidelity + scalability**。

传统路线分成两个 camp:
- **IL from real logs** ([Pomerleau 1988](https://papers.nips.cc/paper/1988/hash/812b4ba287f5ee0bc9bb43d31b4195ad.html)): covariate shift 严重, 没有 recovery data, 一旦 closed-loop deployment 出现小错误就会 compound 成 failure
- **Photorealistic digital twins**: NeRF ([Mildenhall et al. 2020](https://arxiv.org/abs/2003.08934)), 3D Gaussian Splatting ([Kerbl et al. 2023](https://repo.samoa.gwdg.de/cyrus/lastdoctoralenrichment/-/raw/main/3D_Gaussian/3DGAUSSIAN.pdf)), CARLA ([Dosovitskiy et al. 2017](https://arxiv.org/abs/1711.03938))。视觉逼真, 但 optimization 慢、scale 起来贵, 主要用于 evaluation 而非 training

RAP 的 stance 类似于人类的 transfer: 你在 GTA 或 Mario Kart 里学的 driving intuition 能迁移到 real road, 因为 driving decisions 本质上依赖 geometry / semantics / dynamics, 跟 texture / lighting 几乎无关。所以与其在 pixel space 烧钱补 photorealism, 不如用 lightweight rasterization 在 feature space 对齐。

---

## 2. 3D Rasterization Pipeline 详解

### 2.1 Scene Representation

每个 log frame 被重建为两类 primitives:

**Static map elements** (lanes, crosswalks): polylines in world coordinates
$$\mathcal{M} = \{\mathbf{P}_k\}, \quad \mathbf{P}_k \in \mathbb{R}^{n_k \times 3}$$

- 下标 $k$: 第 $k$ 条 polyline (lane / crosswalk 等)
- $n_k$: 该 polyline 的 vertex 数量
- 上标维度 3: $(x, y, z)$ world coordinates

**Dynamic agents** (vehicles, pedestrians, cones, barriers): oriented cuboids
$$\mathcal{B}_i = (l_i, w_i, h_i, \mathbf{T}_i), \quad \mathbf{C}_i = \mathbf{T}_i [\pm l_i/2, \pm w_i/2, 0, h_i]^\top$$

- 下标 $i$: 第 $i$ 个 agent
- $l_i, w_i, h_i$: length / width / height
- $\mathbf{T}_i \in SE(3)$: rigid-body pose in world frame (6-DoF transformation)
- $\mathbf{C}_i \in \mathbb{R}^{8 \times 3}$: 8 个 3D corner points (cuboid 顶点)

**Traffic lights**: 固定尺寸的 upright cuboids, 颜色编码 state (red/yellow/green)

### 2.2 World-to-Image Projection

标准 pinhole camera model:
$$\mathbf{u}_{uv} = \pi(\mathbf{p}_w) = K \mathbf{T}_{w \to c} \tilde{\mathbf{p}}_w \tag{1}$$

- $\mathbf{p}_w \in \mathbb{R}^3$: 3D point in world frame
- $\tilde{\mathbf{p}}_w = [\mathbf{p}_w^\top, 1]^\top \in \mathbb{R}^4$: homogeneous coordinates
- $K \in \mathbb{R}^{3 \times 3}$: camera intrinsics (focal length, principal point)
- $\mathbf{T}_{w \to c} \in SE(3)$: world-to-camera extrinsics (4×4 transformation)
- $\mathbf{u}_{uv} \in \mathbb{R}^3$: 投影后的 raw homogeneous pixel + depth

Perspective division:
$$(u, v) = (u_x / u_z, u_y / u_z)$$

- $u_z$: depth value, 用于后续 depth-aware compositing
- 当 $u_z < z_{\text{near}}$ 时 point 被丢弃 (camera 后方的点)

### 2.3 Rasterization with Depth Compositing

所有 primitives rasterize 到 RGB canvas $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$, 每个 fragment 存 depth $d$ 并用 fading weight blend:

$$\alpha = \max(0, 1 - d / d_{\max})$$

- $d$: fragment 的 depth
- $d_{\max}$: 最大有效 depth (类似 far plane)
- $\alpha$: 越近的 fragment 越不透明, 远的 fade out — 模拟 atmospheric perspective

Occlusion resolution 用 single depth buffer, 跨 view boundary 的 primitives 用 **Sutherland-Hodgman polygon clipping** ([Sutherland & Hodgman 1974](https://dl.acm.org/doi/10.1145/360767.360776)) 切割。

**关键 insight (Figure 4)**: 把 rasterized image 喂给 frozen DINOv3 encoder ([Simeoni et al. 2025](https://arxiv.org/abs/2508.10104)) 后做 PCA 可视化, 发现 raster features 和 real image features 在结构上 qualitatively consistent。这就为后续 feature-space alignment 提供了 evidence — 不需要 pixel 对齐, feature 已经自然相近。

---

## 3. Data Augmentation 策略

### 3.1 Recovery-Oriented Perturbations

直接针对 IL brittleness: 给 ground-truth trajectory 加扰动来模拟"偏离 expert path"的 counterfactual, 让 planner 学会 recovery。

$$\tilde{\tau}(t) = \tau^*(t) + \delta_{\text{lat}}(t) + \delta_{\text{long}}(t) + \epsilon_t$$

- $\tau^*(t)$: expert ground-truth trajectory at time $t$
- $\delta_{\text{lat}}, \delta_{\text{long}}$: 从预定义 range 采样的 lateral / longitudinal offset (counterfactual drift)
- $\epsilon_t$: Gaussian noise (stochastic perturbation)

然后把 perturbed trajectory 重新 rasterize, 生成"ego 已经偏了"的 scene, supervision 让 planner 输出 recovery trajectory。

### 3.2 Cross-Agent View Synthesis

nuPlan 每个 scenario 有 $n$ 个 agents 的 trajectories。把 ego trajectory 替换为另一个 agent 的 trajectory, 保持 camera intrinsics/extrinsics 不变 — 这样就免费拿到一个新 viewpoint 的 training sample, 无需新 sensors。

**总规模**: 85k paired real–raster + 8.5k perturbed + 272k ego raster + 200k cross-agent raster ≈ 500k+ synthetic samples

---

## 4. Raster-to-Real (R2R) Alignment

这是 paper 的另一个核心 contribution。Rasterized image 视觉上跟 real 差很多 (黑底 + colored cuboids + polylines), 但 feature space 接近。所以与其在 pixel space 烧 fidelity, 在 feature space 做 alignment。

### 4.1 Spatial-Level Alignment

对每个 real sample $x^r$ + paired raster $x^s$:

$$F^r = \phi(x^r), \quad F^s = \phi(x^s), \quad F^r, F^s \in \mathbb{R}^{N \times d'}$$

- $\phi(\cdot)$: visual encoder (frozen DINOv3-H 或 learnable ResNet)
- $N$: spatial location 数 (ViT patch tokens 或 CNN feature map positions)
- $d'$: projected feature dim

**重要设计**: freeze raster features $F^s$, 只 update real features $F^r$ — 这是 Real-to-Raster 方向 (Section A.1 ablation Table 7 显示这个方向最优)。Loss:

$$\mathcal{L}_{\text{spatial}} = \frac{1}{N} \sum_{j=1}^{N} \| F^r_j - F^s_j \|_2^2 \tag{2}$$

- 下标 $j$: 第 $j$ 个 spatial location
- $\| \cdot \|_2^2$: squared L2 norm
- 直觉: raster features 来自 clean annotations, 没有干扰 details, 可以充当 dense + clean supervision proxy。让 real features 往这个 "干净 scaffold" 靠, 而不是反过来污染 raster features

### 4.2 Global-Level Alignment

Spatial alignment 需要 paired data (real + raster 同一 scene), 但 cross-agent / perturbed samples 是 raster-only, 无法 spatial align。对这些 unpaired synthetic data, 用 **unsupervised domain adaptation** 思路 ([Ganin & Lempitsky 2015](https://arxiv.org/abs/1409.7495))。

Global representation $g \in \mathbb{R}^{d'}$ 由 feature map $F \in \mathbb{R}^{N \times d'}$ average pool 得到。Domain classifier $D$ 预测 $g$ 来自 real 还是 raster:

$$\mathcal{L}_{\text{global}} = -\mathbb{E}_{(g,y)} [y \log D(g) + (1-y) \log(1 - D(g))] \tag{3}$$

- $y \in \{0, 1\}$: domain label (0=real, 1=raster)
- $D(g) \in [0, 1]$: classifier 输出 probability
- Gradient Reversal Layer (GRL) 插在 $D$ 之前: forward 时 identity, backward 时 gradient 乘 $-\lambda$, 让 encoder 学 domain-invariant features, classifier 学区分 domain — adversarial 博弈

GRL 的 annealing schedule:
$$\lambda(p) = 0.1 \cdot \left(\frac{2}{1 + \exp(-\gamma p)} - 1\right), \quad p \in [0, 1], \gamma = 10$$

- $p$: training progress (0 到 1)
- $\gamma$: 控制 annealing 速度
- 直觉: 训练初期 alignment 弱, 让 task loss 先 dominate, 后期逐渐加强 domain confusion

### 4.3 Overall Objective

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_s \mathcal{L}_{\text{spatial}} + \lambda_g \mathcal{L}_{\text{global}}$$

- $\mathcal{L}_{\text{task}}$: 来自 ([Guo et al. 2025](https://arxiv.org/abs/2505.15111)) iPad 的 total planning loss (multi-modal trajectory head + PDMS scoring head)
- $\lambda_s = 0.002, \lambda_g = 0.1$ (Table 8 hyperparameters): spatial alignment 权重很小, 因为 MSE 容易 dominate
- $\mathcal{L}_{\text{task}}$ 保证 real feature 上的 task-relevant info 不被 alignment 抹掉 (这是 Table 7 中 Real-to-Raster 优于 Symmetric 的关键)

---

## 5. 实验结果深度解读

### 5.1 NAVSIM v1 ([Dauner et al. 2024](https://github.com/autonomousvision/navsim))

Table 1 关键数字:

| Method | NC ↑ | DAC ↑ | TTC ↑ | EP ↑ | PDMS ↑ |
|---|---|---|---|---|---|
| Human | 100 | 100 | 100 | 87.5 | 94.8 |
| Centaur | 99.2 | 98.7 | 98.0 | 86.0 | 92.1 |
| iPad | 98.6 | 98.3 | 94.9 | 88.0 | 91.7 |
| RAP-iPad | 98.2 | 98.6 | 94.6 | 90.1 | **92.5** |
| RAP-DINO | 99.1 | 98.9 | 96.7 | 90.3 | **93.8** |

Metric 解读:
- **NC** (No at-fault Collision): 不发生 ego 有责任的碰撞
- **DAC** (Drivable Area Compliance): 不驶出可行驶区域
- **TTC** (Time-to-Collision): 与前车 TTC > 阈值的时间占比
- **EP** (Ego Progress): ego 在 route 上的进度
- **PDMS** (Planning-aware Driving Metric Score): 上述指标的加权聚合

RAP-iPad vs iPad 的 +0.7 PDMS 提升主要来自 EP (88.0 → 90.1) — augmentation 让 planner 更 aggressive 且 safer。

### 5.2 NAVSIM v2 (navhard split, [Cao et al. 2025](https://arxiv.org/abs/2502.01905))

Table 2 关键: **two-stage EPDMS**
- Stage 1: 标准 open-loop eval
- Stage 2: 用 3DGS ([Kerbl et al. 2023](https://repo.samoa.gwdg.de/cyrus/lastdoctoralenrichment/-/raw/main/3D_Gaussian/3DGAUSSIAN.pdf)) synthesize counterfactual camera views 模拟 closed-loop deviation — 这就是为什么 NAVSIM v2 更能反映 closed-loop robustness

RAP-DINO Stage 2 EPDMS = 36.93, 比 LTF (23.12) 高出 13.81 分, 差距巨大。这印证了 recovery perturbation 的价值: Table 6 显示 perturbation 让 v2 从 32.5 → 36.9 (提升 4.4), 但对 v1 无影响 (92.5 → 92.5)。**Augmentation 的收益集中在 closed-loop metric 上, open-loop 测不出来**。

### 5.3 WOD-E2E Driving ([Xu et al. 2025](https://arxiv.org/abs/2510.26125))

Table 3 关键: dataset 专为 long-tail events 设计 (construction detours / pedestrian accidents / unexpected freeway obstacles), 这些场景在 daily driving 中频率 < 0.003%。

| Method | ADE@5s ↓ | ADE@3s ↓ | RFS (Spotlight) ↑ | RFS (Overall) ↑ |
|---|---|---|---|---|
| Poutine (3B VLM) | 2.74 | 1.21 | 6.89 | 7.99 |
| RAP-DINO | **2.65** | **1.17** | **7.20** | **8.04** |

RAP-DINO 击败 Poutine ([Rowe et al. 2025](https://arxiv.org/abs/2506.11234)) 这个 3B scale 的 vision-language-trajectory model, 说明 lightweight rasterization + feature alignment 在 long-tail 上比 massive VLM 更 efficient。

### 5.4 Bench2Drive ([Jia et al. 2024](https://github.com/Thinklab-SJTU/Bench2Drive))

真正的 CARLA closed-loop evaluation: 220 routes, 每条 route 含 safety-critical event。

Table 4:
| Method | Success Rate ↑ | Driving Score ↑ |
|---|---|---|
| UniAD-Base | 16.36 | 45.81 |
| DriveTransformer | 35.01 | 63.46 |
| iPad | 35.91 | 65.02 |
| RAP-ResNet (29M) | **37.27** | **66.42** |

RAP-ResNet 只有 29M params, 优于 888M 的 RAP-DINO 在 closed-loop inference 速度上, 同时刷新 SOTA。

---

## 6. Ablation 深度分析

### 6.1 3D Rasterization Design (Table 5)

| ID | Face Rendering | Depth Decay | Background | MinADE ↓ |
|---|---|---|---|---|
| A | Colored | Yes | Black | **0.91** |
| B | Transparent | Yes | Black | 0.98 |
| C | Colored | No | Black | 1.05 |
| D | Colored | Yes | Natural | 1.33 |

Insights:
- **Colored faces > Transparent** (+0.07): 实心 colored face 给 encoder 提供 object boundary cue, transparent 反而模糊 semantic
- **Depth Decay > No Decay** (+0.14): atmospheric fading 给 encoder 距离感, 没有 depth cue 时远处 agent 和近处 agent 视觉上无差异, encoder 难学 depth-aware planning
- **Black Background > Natural** (+0.42!): 巨大差异。Natural sky-ground split 引入 distracting visual pattern, encoder 浪费 capacity 学习背景分布; 纯黑背景让 encoder 聚焦 geometric primitives — 这正是 paper 的 thesis 的强证据

### 6.2 R2R Alignment (Figure 5)

固定 total size, 把 real data 替换为不同比例 ({1%, 5%, 20%, 50%, 100%}) raster:
- **No alignment < Spatial < Spatial+Global** 一致 across 所有 ratio
- **50% synthetic + alignment > 100% real**! 这是反直觉但 strong 的结论: 高质量 raster features 反而比 real image features 更 clean, 作为 augmentation 提供额外 signal

### 6.3 Cross-Agent View Synthesis Scaling (Figure 6)

从 85k real 开始, 加 {1k, 10k, 100k, 500k, 1000k} synthetic cross-agent samples:

$$y = -0.021 \ln(x) + 1.2173, \quad R^2 = 0.9942$$

- $x$: synthetic sample count
- $y$: MinADE
- $R^2 = 0.9942$: 几乎完美 log-scaling fit

这跟 [Baniodeh et al. 2025](https://arxiv.org/abs/2412.02689) 和 [Zheng et al. 2024](https://arxiv.org/abs/2412.02689) 在 real data scaling 上发现的 log-law 一致 — **即使是 rasterized secondary viewpoint (其他 agent 的视角) 也遵循同样的 scaling law**, 说明 raster augmentation 真的在贡献 planning-relevant signal, 而不只是噪音。

---

## 7. 几个关键 Architecture 细节

### 7.1 RAP-DINO Architecture

- **Backbone**: frozen DINOv3-H+ (来自 [Simeoni et al. 2025](https://arxiv.org/abs/2508.10104))
- **Projector**: learnable MLP (把 DINOv3 features project 到 task space)
- **Decoder**: iterative deformable attention, adapted from [iPad](https://arxiv.org/abs/2505.15111)
- **Heads**: 
  - Multi-modal trajectory head (supervised by future trajectories)
  - Trajectory scoring head (supervised by PDMS scores)
- **总参数**: ~888M

### 7.2 RAP-ResNet (用于 Bench2Drive)

- **Backbone**: ResNet34 ([He et al. 2016](https://arxiv.org/abs/1512.03385)) — 为 closed-loop CARLA inference 设计, 速度优先
- **总参数**: ~29M, 比 RAP-DINO 小 30 倍

### 7.3 Training Setup (Table 8)

- 4 × H100 GPU, 80 hours
- AdamW ([Loshchilov & Hutter 2017](https://arxiv.org/abs/1711.05101))
- Initial LR 1e-4, cosine decay
- Batch size 128 (NAVSIM) / 64 (WOD & Bench2Drive)
- Dropout 0.1, weight decay 1e-4
- 20 epochs pretraining + 20 epochs finetuning

### 7.4 Bench2Drive Mixed Training 细节

由于 CARLA 和 nuPlan 数据格式不同, mixed training 前需要 alignment:
- Camera views reorder 到统一 sequence
- Image resize 到 576 × 1024
- Camera calibration matrix 加 rotation + scaling 匹配 nuPlan convention
- Ego kinematics + target trajectories normalize 到同一 coordinate frame

---

## 8. 关键 Discussion (Section A.1)

### 8.1 简化 Rasterization 会丢失 real-world cue 吗?

Figure 7 的 ablation 很有意思: 用同一个 fully-trained RAP-DINO, 分别 condition on (a) raw real image, (b) rasterized image:
- **Scenario A**: unannotated "Keep Left" sign (不在 raster ontology) → real condition 下 planner 正确反应, raster condition 下失败
- **Scenario B**: OOD LED arrow on truck → real condition 下正确 lane change, raster condition 下失败

这说明: raster 只提供 geometric scaffold, **real image 在 training 时和 inference 时一直存在, 补充 raster 编码不进去的细粒度信息**。两者是 complementary 的。

### 8.2 Real-to-Raster Alignment 会抹掉 real 信息吗?

Table 7 的三种 alignment 方向对比 (50% real data):
| Variant | MinADE ↓ |
|---|---|
| Raster-to-Real | 1.12 |
| Symmetric | 1.14 |
| Real-to-Raster (ours) | **1.02** |

Real-to-Raster 最好的原因: raster feature 是 clean scaffold, 让 real feature 往它靠等于提供 structural prior; 同时 task loss 在 real feature 上起作用, 保证 task-relevant detail 不被抹掉。这呼应 [Yu et al. 2024](https://arxiv.org/abs/2410.06940) 在 diffusion transformer 上发现的 "representation alignment 帮助 semantic abstraction 不丢 perceptual detail"。

---

## 9. Limitation & Future Work

作者诚实承认: RAP 仍在 IL paradigm 内, 继承 causal confusion 等 IL 固有问题。Future direction 是把 3D rasterization 扩展成 full simulator 支持 closed-loop RL — 这样可以从 reward signal 而不只是 expert demonstration 学习, 突破 IL 的根本局限。

---

## 10. 我的 Take-aways for Building Intuition

1. **Identify the bottleneck of the bottleneck**: E2E driving 的 bottleneck 是 covariate shift + 缺 recovery data; photorealistic rendering 解决了 pixel fidelity 但没解决 scalability, 所以是 over-engineered。RAP 找到了真正瓶颈 (scalable counterfactual generation) 并用最轻的方式解决。

2. **Feature space > Pixel space for transfer**: Figure 4 的 DINOv3 PCA 可视化是 paper 的 intuition 支柱 — 不同模态 (real / raster) 在好的 feature space 已经天然接近, 所以 alignment 应该在 feature 而非 pixel 上做。

3. **Augmentation 收益集中在 closed-loop**: Table 6 显示 recovery perturbation 对 NAVSIM v1 (open-loop) 完全无效, 对 v2 (pseudo-closed-loop) 提升 13.5%。这给 field 一个重要 warning: 只看 open-loop metric 会错过 augmentation 真正价值。

4. **Log-scaling 在 synthetic data 上同样成立**: Figure 6 的 $R^2 = 0.9942$ 拟合不是巧合, 跟 real data scaling law 一致。说明 good synthetic data 在 representation learning 上的 contribution 是 predictable 的。

5. **Black background > Natural background**: 这个看似反直觉的 ablation 是 paper thesis 的最强证据 — driving planner 不需要 photorealism, 需要 clean geometric + semantic signal。

---

## Reference Links

- Project page: https://alan-lanfeng.github.io/RAP/
- NAVSIM: https://github.com/autonomousvision/navsim
- nuPlan: https://www.nuscenes.org/nuplan
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- 3D Gaussian Splatting (原始 paper): https://repo.samoa.gwdg.de/cyrus/lastdoctoralenrichment/-/raw/main/3D_Gaussian/3DGAUSSIAN.pdf
- DINOv3: https://arxiv.org/abs/2508.10104
- iPad: https://arxiv.org/abs/2505.15111
- DiffusionDrive: https://arxiv.org/abs/2501.01879
- Gradient Reversal Layer 原始 paper: https://arxiv.org/abs/1409.7495
- Sutherland-Hodgman Clipping (1974): https://dl.acm.org/doi/10.1145/360767.360776
- WOD-E2E Driving: https://arxiv.org/abs/2510.26125
- Poutine (3B VLM baseline): https://arxiv.org/abs/2506.11234
- Centaur (test-time training baseline): https://arxiv.org/abs/2503.11650

---

总结一句: RAP 的 elegance 在于它**用最便宜的方式 (annotation-based rasterization) 生成最 task-relevant 的 augmentation (recovery + cross-agent), 然后用最自然的 alignment 方式 (feature space, not pixel space) 弥合 sim-to-real gap**。整个 pipeline 没有 fancy 的 neural rendering, 但每个 design choice 都精准打击 E2E driving 的真实痛点。这种 "做减法" 的工程品味非常值得学习。
