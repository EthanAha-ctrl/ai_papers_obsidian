---
source_pdf: DrivingSphere Building a High-fidelity 4D World for Closed-loop Simulation.pdf
paper_sha256: d2c5f85b09f7f4eede3c32fc4b0748339906c67c0a3d170bd15d39e626331892
processed_at: '2026-08-03T23:55:24-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 DrivingSphere

---

## 这篇 paper 在干啥？

你在搞自动驾驶，想验证你的 end-to-end driving model 行不行。光在 nuScenes 那点 log data 上跑 open-loop 不够 —— 那只是"看视频猜轨迹"，model 啥也不影响。你要的是 **closed-loop**：model 看到画面 → 输出方向盘刹车 → 车动了 → 画面变了 → model 再看新画面。跟真开车一样。

CARLA 能 closed-loop 但画面太假，model 训在真 data 上跑到假 data 里就瞎。MagicDrive 画面真但不能 closed-loop，没法测互动。

DrivingSphere 就是来填这个坑的：**画面像真的，又能 closed-loop**。

---

## 怎么填的？核心 trick 是什么？

核心就一句：**先在 3D voxel 世界里演戏，再把这出戏"拍"成视频给 model 看**。

为啥这么干？因为直接生成视频（像 MagicDrive 那样）有个老大难问题 —— **多视角之间对不上**。cam-front 看到一辆红车，cam-front-left 里可能就变蓝车了；第 10 帧的那辆车到第 30 帧可能就闪烁消失了。原因是没有一个明确的"这辆车到底长什么样、在哪儿"的物理 anchor。

DrivingSphere 的解法：**先把场景和所有车都在 3D occupancy grid 里摆好**（每辆车有唯一 ID、有 voxel 形状、有 6 路相机都能投影到的 3D 位置），然后再让 diffusion model "渲染"成视频。这样遮挡、depth、跨视角一致性全是 voxel geometry 自然决定的，不靠生成器去"猜"。

类比一下：拍电影是先有 3D 场景 + 走位 + 演员，再用摄影机拍。MagicDrive 那种是直接让 AI 画一段多视角视频，画着画着就崩了。

---

## 整个 pipeline 三层

### 第一层：搭一个城市级 3D 场景（OccDreamer）

你要一块儿一块儿地生成 3D voxel 场景。每块儿给两个 input：
- 一张 BEV 地图（路在哪儿）
- 一句 text（"郊区 + 多树" / "商业街 + 楼房"）

用 VQVAE 把 occupancy 压成 latent，然后 latent diffusion + ControlNet 生成。跟 Stable Diffusion + ControlNet 那套完全一样，只不过 latent 是 3D 的不是 2D 的。

**怎么扩成整座城市？** outpainting。相邻两块儿区域有 overlap，拿 overlap 当 anchor，往外画下一块儿，再 merge。一堵一堵墙接起来就是 city-scale。

### 第二层：往场景里塞车、塞人、给它们轨迹

- **Actor bank**：提前备好一堆车/人的 voxel asset，每个带个 caption（"红色轿车"）和一个 unique ID。用 CLIP 算 text 相似度检索合适的 actor。
- **轨迹**：ego agent（被测的 driving model）输出 trajectory；NPC 用 LimSim 这个 traffic simulator 算。
- **组合**：把每个 actor 按当前 pose 写进 occupancy grid 里对应的位置，得到时刻 t 的 4D world state $\mathcal{W}^t$。

这层是纯几何操作，不渲染像素，所以快。

### 第三层：把 4D world 渲染成 6 路相机视频

这是最 tricky 的地方。条件怎么给 diffusion model？

- **Global 路**：拿一个预训练的 4D occupancy VAE encoder 吃整个 $\mathcal{W}$（包括时间维度），编码出"全局 3D 结构 + temporal 关系"。
- **Local 路**：把 occupancy 按相机内外参 project 成 2D semantic map（每个 pixel 就是那条 ray 打中的第一个 voxel 的类别），用 image VAE 编码。这条路负责"每个相机看到的像素级精确语义"。
- 两条路用 cross-attention 融合，作为 ControlNet 的输入。

**ID-aware actor encoding 是关键 trick**：每个 actor 拼一个 embedding = Fourier(position) + Fourier(unique_ID) + T5(caption)。unique ID 用 Fourier 编码成高频正弦波，这样 cross-attention 的 query 能"查询"到具体某个 actor 的外观描述，跨视角跨帧都认得是它。这就是为什么 red sedan 在 6 个相机里都是同一辆 red sedan。

主干用 Open-Sora 的 ST-DiT transformer。VSSA 把 6 个视角的 token 拍平成一个长 sequence 做 self-attention，省参数又保证 multi-view coherence。

**autoregressive 生成**：第一帧 anchor，后面每帧 condition 在前一帧，保证 long video 时序稳定。

---

## Closed-loop 怎么转起来？

每个 timestep：
1. VideoDreamer 渲染出当前帧 6 路视频 $V^t$
2. Ego agent（UniAD 之类）吃 $V^t$，输出 ego 控制信号 $c_{\mathrm{ego}}^{t+1}$
3. LimSim 吃当前 world state $\mathcal{W}^t$，输出 NPC 控制信号 $c_n^{t+1}$
4. 所有 actor 的位置按公式 (6) 更新，得到 $\mathcal{W}^{t+1}$
5. 回到第 1 步

就这么 loop。

---

## 实验结果讲人话

- **场景生成**：FID 比 SemCity 好一倍多（274 vs 634）。BEV ControlNet 给精确道路结构功劳大。
- **视频生成**：FVD 103 vs DriveArena 185 vs MagicDrive 218。UniAD 在生成视频上跑 3D detection mAP 21.45，比 DriveArena 的 16.06 高 34%，说明生成视频里的车更"看得见"——occlusion 处理对了。
- **Open-loop**：PDMS 0.742 vs DriveArena* 0.698。视觉质量直接 translate 到 planner 评测分数。
- **Closed-loop**：UniAD 完成 11.7% 路线，DriveArena 只有 6.4%，差不多翻倍。但绝对数字还是低，说明 closed-loop 评测对 driving model 要求高很多 —— 这恰恰是 closed-loop sim 的价值。

---

## 我的几个直觉判断

1. **occupancy 作 bottleneck 的代价**：nuScenes 标准 voxel 0.5m，车轱辘后视镜这些细节根本没。VideoDreamer 渲染时得靠 diffusion "脑补"，可能幻觉出错误结构。actor bank 也只能覆盖有限的 vehicle type，长尾还是难。

2. **Scene Expansion 的 outpainting 在语义突变处可能崩**：从商场扩到郊区，overlap 区域 anchor 不够强，merge 边界可能出现"半楼半树"的鬼东西。paper 没讨论。

3. **autoregressive 生成慢**：一帧一帧来，推理 latency 高，做不到 realtime sim-in-the-loop。现在只能 offline batch 评测 100 条 trajectory。真正部署用还得 distill 或 cache。

4. **ID-aware encoding 本质是 retrieval**：把每个 actor 的外观当外部知识库，cross-attention 做 lookup。和 RAG 思路一致，很聪明。如果未来 actor bank 大到几万个 asset，可能要更结构化的 retrieval 机制（tree-based 或 hash）。

5. **和纯 video world model 路线的对比**：GAIA-1 / Vista 这种直接学 video→video 的 world model，跳过 explicit occupancy。优点是 pipeline 轻、端到端可微；缺点是 controllability 弱、physics 是隐式的。DrivingSphere 走 explicit representation 路线，controllable + interpretable，但 pipeline 重、asset 要 hand-curate。两条路最终比的是 long-tail generalization：explicit 路线受 actor bank 多样性限制，implicit 路线受训练数据多样性限制。

6. **真正的瓶颈不在 generator，在被测 model**：Tab.4 里 UniAD closed-loop 才 11.7% RC，说明 UniAD 在 OOD 视频上 generalization 不够。这反过来说明 closed-loop sim 才是能暴露这个问题的工具，open-loop 永远测不出来。

---

## 一句话总结

**DrivingSphere = "3D voxel 当世界状态 + diffusion model 当渲染器 + ID encoding 保证跨视角一致性 + traffic sim + driving model 双向耦合做 closed-loop"**。

工程拼装为主（VQVAE / ControlNet / ST-DiT / Fourier encoding 都是现成的），但组合方式解决了一个真问题：multi-view multi-frame actor consistency。对 AD 评测社区的意义是定了个新标准 —— 以后 closed-loop benchmark 大概率都得显式建模 4D world state，纯 2D 控制不够看了。

---

# DrivingSphere: 一篇把"4D 占用世界"和"扩散生成视频"焊在一起的 closed-loop 仿真平台

Hi Andrej, 这篇 paper 我读完了，下面我从 intuition 优先的角度给你拆解，公式变量都讲清楚，并把它放到整个 generative driving simulation 的 lineage 里去理解。

Project page: <https://yanty123.github.io/DrivingSphere/>

---

## 0. 一句话概括

DrivingSphere 把"基于 occupancy grid 的 4D 世界模型"当作中间表示，上游用 **OccDreamer** 生成可控的静态城市级 occupancy scene 并把动态 actor 摆进去，下游用 **VideoDreamer** 把 4D occupancy 渲染成 6 路相机的时序视频；最后让一个 end-to-end AD agent（UniAD/VAD 这类）吃这个视频输出 control signal，再去更新 occupancy world，形成 **closed-loop**。本质上是把 DriveArena [50] 的 "2D sketch → video" 那一跳替换为 "4D occupancy → video"，并在 actor 一致性、场景可扩展性上做了加固。

可以把它理解成一个 stack：

```
[BEV map + text] --OccDreamer--> [city-scale occupancy S_city]
[S_city + actor bank + trajectories] --Composition π--> [W^t : 4D world at t]
[W^0..T] --VideoDreamer (dual-path + ID-aware + ControlNet-DiT)--> [6-view video V^0..T]
[V^t] --F_driving--> [c_ego^{t+1}] --update W--> closed loop
```

---

## 1. 为什么是 4D occupancy 而不是 2D sketch？

DriveArena 用 2D traffic sketch 控生成，问题是 sketch 没有 depth/occlusion/geometry，生成器只能"猜"建筑物、植被的位置。DrivingSphere 选择 occupancy grid $S_k \in \mathbb{R}^{H \times W \times D}$ 作为静态背景表示，每个 voxel 带一个 semantic label。这样做的好处是：

1. **非交通元素显式建模**：building / vegetation / obstacle 直接占用 voxel 空间，渲染时会自然产生遮挡，对 perception 模型有真正的鲁棒性测试。
2. **物理一致性**：actor 放进 occupancy world 里，occlusion 由 voxel 几何决定，不靠生成器"学"。
3. **可扩展**：occupancy 可以做 outpainting 拼成 city-scale，而 2D sketch 拼接语义边界很难对齐。

代价是：occupancy 的 diffusion 比像素扩散贵得多，需要 VQVAE 先 tokenize。

---

## 2. Dynamic Environment Composition（Sec 3.1 + Fig.3）

### 2.1 4D world 表示

定义世界为三元组的集合：

$$
\mathcal{W} = \{S_{\mathrm{city}}, (A_1, P_1), \ldots, (A_N, P_N)\}
$$

- $S_{\mathrm{city}} = \{S_1, \ldots, S_K\}$: K 个区域静态 scene 的拼接；每个 $S_k \in \mathbb{R}^{H\times W\times D}$ 是 occupancy grid。
- $A_n \in \mathbb{R}^4$: 第 n 个 actor，由 3D 坐标 + semantic label 构成（注意 $A_n$ 在公式里是 4 维向量，但语义上是 actor 的 voxel + label）。
- $P_n = \{P_n^0, \ldots, P_n^T\}$: actor n 在 $t=0\ldots T$ 帧的位姿轨迹；每个 $P_n^t \in \mathbb{R}^4$ = (3D 相对 ego 的位移, yaw $\theta$)。

这里有个细节：$A_n$ 是 instance-level voxel（从 actor bank 取的实例化 voxel），$P_n$ 是它的时序位姿。$A_n$ 决定"长什么样"，$P_n$ 决定"怎么动"。

### 2.2 OccDreamer：静态城市生成

这是论文最 dense 的部分。三个组件：**Occupancy Tokenizer**、**Region Occupancy Generation**、**Scene Expansion**。

**(a) Occupancy Tokenizer**：VQVAE 把 $S_k$ 压成 latent $Z^{S_k} = \mathcal{F}_{\mathrm{VAE,Enc}}^{occ}(S_k)$。训练 loss 是公式 (1):

$$
\mathcal{L}_{occ} = \mathcal{L}_{CE}(S_k, S_k') + \alpha \mathcal{L}_{Lov}(S_k, S_k') + \beta \mathcal{L}_{emb}
$$

变量解释：
- $S_k$: GT occupancy（每个 voxel 有 semantic class）。
- $S_k' = \mathcal{F}_{\mathrm{VAE,Dec}}^{occ}(Z^{S_k})$: 重建。
- $\mathcal{L}_{CE}$: voxel-level cross-entropy，主要 supervision。
- $\mathcal{L}_{Lov}$: Lovász-Softmax loss [1]，是 IoU/Jaccard 的 tractable surrogate，对类别不均衡（vegetation 远多于 car）效果好。直觉上 Lovász 把离散的 IoU 优化问题凸松弛成连续 hinge loss，原 paper: <https://arxiv.org/abs/1705.08790>。
- $\mathcal{L}_{emb}$: VQVAE 的 codebook commitment loss（encoder output 和 codebook entry 的拉近），防止 codebook 漂移。
- $\alpha, \beta$: 标量权重。

**(b) Region Occupancy Generation**：text + BEV 双条件扩散。text prompt 通过 CLIP [31] 编码为 $F_{\mathrm{region}}$（e.g., "suburban area with rich vegetation"）；BEV map $M$ 通过预训练图像 VAE 提取 $F_M$（道路结构）。扩散 loss 是公式 (2):

$$
\mathcal{L}_{occ} = \mathbb{E}_{F_M, F_{\mathrm{region}}, Z^{S_k}, \epsilon^s, \tau} \Big[ \| \epsilon^s - \epsilon^s_\theta(Z^{S_k}_\tau, F_{\mathrm{region}}, \tau, \epsilon^s_\phi(F_M)) \|^2 \Big]
$$

- $\epsilon^s \sim \mathcal{N}(0, I)$: 加进去的 noise。
- $Z^{S_k}_\tau$: 对 $Z^{S_k}$ 在 timestep $\tau$ 加噪后的 latent。
- $\epsilon^s_\theta$: 主 denoiser（cross-attention 注入 $F_{\mathrm{region}}$）。
- $\epsilon^s_\phi$: ControlNet 分支，吃 $F_M$（BEV embedding），输出 residual feature 注入 $\epsilon^s_\theta$。
- $\tau \in \{1,\ldots,T_{\mathrm{diff}}\}$: diffusion timestep。

这个 design 完全平行于 Stable Diffusion + ControlNet [54] 的管线，只不过在 latent space 是 3D occupancy latent。这里 $F_{\mathrm{region}}$ 提供"这是什么类型的环境"，$F_M$ 提供"路在哪里"。

**(c) Scene Expansion Mechanism**：核心思想是 overlap-based outpainting。给定已生成区域 $S_k$，要扩到 $S_{k+1}$，先用 binary mask $O$ 取 $S_k$ 与 $S_{k+1}$ 的 overlap 区域（公式 3）:

$$
S_k^{\mathrm{partial}} = S_k \odot O
$$

$\odot$ 是 Hadamard product。然后用 outpainting 模式（公式 4）:

$$
Z^{S_{k+1}} = \mathcal{F}_{\mathrm{outpainting}}(\epsilon^s, M_{k+1}, S_k^{\mathrm{partial}})
$$

最后合并（公式 5）:

$$
S_{\mathrm{merged}} = \mathrm{Merge}\big(S_k \odot (1-O), S_{k+1}\big)
$$

注意 $S_k \odot (1-O)$ 把 overlap 区域从 $S_k$ 中抹掉，避免和新生成的 $S_{k+1}$ 冲突，merge 只在 boundary 做插值。这样能扩成 city-scale 而 boundary 不断裂。

### 2.3 Actor bank + trajectory

- **Actor bank $\mathcal{B}$**: 每个 actor $A_n$ 包含 instance-level voxel（从真实或合成 scene 分割出来的），caption $L_{A_n}$（CLIP text），unique ID $I_{A_n}$。Selection 用 CLIP similarity 检索或者 random sample。
- **Trajectory**: 控制信号 $c_n^t$ 驱动位置更新（公式 6）:

$$
P_n^{t+1} = P_n^t + \Delta P_n^t(c_n^t)
$$

$\Delta P_n^t$ 是控制信号导致的位置增量（kinematic bicycle model 或 IDM 之类，paper 没细说，应该用 LimSim [46] 内置的）。控制信号来源在 Sec 3.3：ego 用 end-to-end model 输出，其他 NPC 用 traffic flow simulator。

### 2.4 4D composition（公式 7）

$$
\mathcal{W}^t = \pi\big(S, (A_n, \mathbf{c}_n^t)_{n=1}^N\big)
$$

$\pi$ 是 composition operator：把每个 actor 按其 $P_n^t$ 摆到 static scene $S$ 上（替换/合并 voxel）。注意这里没有渲染，纯几何操作，所以快。$\mathcal{W}^t$ 就是时刻 t 的 4D occupancy world。

---

## 3. Visual Scene Synthesis（Sec 3.2 + Fig.4）

这是论文我最喜欢的部分。VideoDreamer 解决"怎么把 4D occupancy 渲染成真实多视角视频"，并且要 ID 一致。三个 trick：**Dual-path Condition Encoding**、**ID-aware Actor Encoding**、**ControlNet-DiT**。

### 3.1 Dual-path Condition Encoding：为什么需要两条路？

直觉：occupancy 既包含"全局几何 + temporal 关系"（一辆车在 BEV 上从前移到后，4D），又包含"每个视角看到的 2D 语义投影"（这辆车在 cam-front 占哪 100×80 个像素）。这两信息层次不同，用单一 encoder 容易丢。

**Global branch**:
$$
F_{\mathrm{global}} = \mathcal{F}_{\mathrm{VAE}}^{4\mathrm{Docc}}(\mathcal{W})
$$
用 OccSora [41] 预训练的 4D occupancy VAE encoder，直接吃整个 $\mathcal{W} \in \mathbb{R}^{T\times H\times W\times D}$（时间维度也编码进去），输出 capture 了时空几何。这一路负责"世界长什么样"。

**Local branch** (公式 8):
$$
\mathcal{M}_v^t = \mathcal{Q}(\mathcal{W}^t, \mathbf{K}_v, \mathbf{T}^t)
$$
- $\mathcal{Q}$: render + project function（论文用 Mayavi [33]，本质是 voxel ray casting + semantic label lookup）。
- $\mathbf{K}_v \in \mathbb{R}^{3\times 3}$: 第 $v$ 个视角的相机内参（fx, fy, cx, cy）。
- $\mathbf{T}^t \in \mathrm{SE}(3)$: ego 在 t 时刻的位姿，6 路 camera 各自的 extrinsic。
- $\mathcal{M}_v^t \in \mathbb{R}^{H\times W}$: 单视角单帧的 2D semantic map，每个 pixel = 那条 ray 命中的第一个 voxel 的 class。

得到 $\mathcal{M} \in \mathbb{R}^{T\times 6\times H\times W}$（6 个 camera，T 帧），用 image VAE 编码为 $F_{\mathrm{view}}$。这一路负责"每个 camera 应该看到哪些语义"，pixel-aligned，对 occlusion 边界特别准。

最后 $F_{\mathrm{global}}$ 和 $F_{\mathrm{view}}$ 通过 cross-attention 融合成 $F_{\mathrm{occ}}$。注意 $F_{\mathrm{occ}}$ 是 ControlNet 的输入，不是 cross-attention 的 K/V。

### 3.2 ID-aware Actor Encoding：为什么跨视角一致性是个问题？

直觉：naive diffusion 在生成时，actor "身份"是隐式的——同一辆车在 cam-front 和 cam-front-left 可能被生成成完全不同的车型/颜色。这就是 multi-view consistency 的核心痛点。MagicDrive [13] 用 trajectory box 来 anchor，但还是会闪烁；DriveDreamer [43] 几乎不处理。

DrivingSphere 的解法是给每个 actor 显式编码一个 ID。actor embedding（公式 10）:

$$
F_{A_n} = \mathrm{CONCAT}\big[\mathcal{F}_{\mathrm{Fourier}}(P_n), \mathcal{F}_{\mathrm{Fourier}}(I_n), \mathcal{F}_{T5}(L_{A_n})\big]
$$

- $P_n$: actor 的 3D 位置（公式里没说全，应该是 $P_n^t$ per frame，给一个 temporal sequence）。
- $I_n$: actor 的 unique ID（整数或 hash）。这是关键——Fourier encoding 把 ID 编码成一组高频正弦波，让模型能在 cross-attention 中区分"actor 1 vs actor 2"，即使它们都是"car"类别。
- $L_{A_n}$: actor 的 caption（"a red sedan", "a pedestrian wearing yellow shirt"），T5 [32] encoder 编码。
- $\mathcal{F}_{\mathrm{Fourier}}$: 经典 NeRF [29] 那种 $\gamma(x) = (\sin(2^0\pi x), \cos(2^0\pi x), \ldots, \sin(2^{L-1}\pi x), \cos(2^{L-1}\pi x))$ 位置编码。

公式 (9):
$$
F_{\mathrm{fuse}} = \mathrm{CONCAT}[F_W, F_{A_1}, \ldots, F_{A_N}]
$$
$F_W$ 是 scene caption（T5 编码，描述场景 context），整个序列作为 cross-attention 的 K/V 注入 ST-DiT。

intuition: 把 ID Fourier encode 后，cross-attention 的 query（latent feature 在某个空间位置）能精确地"查询"到对应 actor 的外观描述。这比单纯靠 spatial location anchor 强，因为 location 是连续的、易混淆，ID 是离散的、唯一的。

### 3.3 ControlNet-DiT：架构细节

主干用 Open-Sora [59] 的 **ST-DiT** (Spatial-Temporal DiT)。输入视频 $V \in \mathbb{R}^{v\times T\times H\times W}$（v=6 视角）经 image VAE 编码成 $Z^V \in \mathbb{R}^{v\times T\times H'\times W'\times C}$。

**View-aware Spatial Self-Attention (VSSA)**：把 $Z^V$ reshape 成 $T \times (vH'W') \times C$，即把 view/height/width 拍平成 sequence length，时间维度保留独立。这样每个 timestep 内所有视角的 token 互相 attend，跨 timestep 由 Temporal Self-Attention 处理。比直接 cross-view attention 参数少很多（不用每个 view pair 专门的 attention head）。

**Cross-attention**：注入 $F_{\mathrm{fuse}}$（scene + actor caption + ID + position）。
$$
\tilde{Z}^V = \mathrm{CrossAttention}(Z^V, F_{\mathrm{fuse}})
$$

**ControlNet 分支** $\epsilon^v_\phi$：复用 $\epsilon^v_\theta$ 的若干 block，输入 $F_{\mathrm{occ}}$（dual-path 融合的 occupancy condition），输出 residual 加到主 denoiser。这思路直接 from PixArt-$\delta$ [4] (<https://arxiv.org/abs/2401.05252>)。

最终 diffusion loss (公式 11):
$$
\mathcal{L}_{\mathrm{video}} = \mathbb{E}_{F_{\mathrm{occ}}, F_{\mathrm{fuse}}, Z^V, \epsilon^v, \tau}\Big[\|\epsilon^v - \epsilon^v_\theta(Z^V_\tau, F_{\mathrm{fuse}}, \tau, \epsilon^v_\phi(F_{\mathrm{occ}}))\|^2\Big]
$$

变量一一对应 scene 生成那份 loss。这里 $Z^V_\tau$ 是加噪后的 video latent，模型学的是预测 noise $\epsilon^v$。

### 3.4 Auto-regressive generation

公式 (12)（论文这里编号有点歧义，原文给的是 $V^t = \mathcal{F}_{\mathrm{video}}(V^{t-1}, Z^V, t)$）:

$$
V^t = g_\theta(V^{t-1}, Z^V, t)
$$

- $g_\theta$: 视频生成函数。
- $V^{t-1}$: 前一帧（first-f mask，把第 0 帧作为 anchor）。
- $Z^V$: 当前块的 latent condition。
- $t$: 时间索引。

这就是 Open-Sora 的 first-frame conditioning：把第一个 frame 作为已知 anchor，后面每帧 condition 在前一帧，保证 long-horizon temporal consistency。代价是生成慢（autoregressive），但比一次性生成 100 帧稳定性高。

---

## 4. Agent Interplay & Closed-Loop（Sec 3.3）

两种 agent：

**Ego agent** (公式 12):
$$
c_{\mathrm{ego}}^{t+1} = \mathcal{F}_{\mathrm{driving}}(V^t)
$$
- $\mathcal{F}_{\mathrm{driving}}$: 被测的 end-to-end driving model（UniAD/VAD/VADv2 等）。
- $V^t$: 6-view video frame。
- $c_{\mathrm{ego}}^{t+1}$: ego 的下一个控制信号（trajectory waypoint）。

**Environment agent** (公式 13):
$$
c_n^{t+1} = \mathcal{F}_{\mathrm{system}}(\mathcal{W}^t)
$$
- $\mathcal{F}_{\mathrm{system}}$: traffic flow simulator，这里用 LimSim [46]。
- $\mathcal{W}^t$: 当前 4D world state。
- $c_n^{t+1}$: 每个 NPC 的控制信号。

Ego 和 NPC 的控制信号都走公式 (6) 更新 $\mathcal{W}^{t+1}$，然后 VideoDreamer 渲染下一帧视频，loop 继续。这就是 closed-loop 的本质——ego 的动作改变环境，环境再影响 ego 的下一帧 observation。

---

## 5. Experiments

### 5.1 Scene generation（Tab.2）

| Method | FID↓ | MMD↓ |
|---|---|---|
| SemCity [22] | 634 | 0.251 |
| **OccDreamer** | **274** | **0.082** |

FID (Fréchet Inception Distance) 衡量生成分布与真实分布距离；MMD (Maximum Mean Discrepancy) 在 occupancy 渲染成 2D 图后算。DrivingSphere 比 SemCity 好一倍多，主要因为 BEV ControlNet 给了精确 road structure，SemCity 只用 tri-plane diffusion 没有 spatial conditioning。

### 5.2 Video generation（Tab.1）

| Source | FVD↓ | mAP↑ | NDS↑ | Lanes↑ | Drivable↑ | Divider↑ | L2@1s↓ | L2@2s↓ | L2@3s↓ |
|---|---|---|---|---|---|---|---|---|---|
| real nuScenes | — | 37.98 | 49.85 | 31.31 | 69.14 | 25.93 | 0.51 | 0.98 | 1.65 |
| MagicDrive | 218.12 | 12.92 | 28.36 | 21.95 | 51.46 | 17.10 | 0.57 | 1.14 | 1.95 |
| Panacea | 139.00 | 13.72 | 27.73 | 18.23 | 52.37 | 17.21 | 0.58 | 1.14 | 1.97 |
| DriveArena | 185.32 | 16.06 | 30.03 | 26.14 | 59.37 | 20.79 | 0.56 | 1.10 | 1.89 |
| **DrivingSphere** | **103.42** | **21.45** | **34.16** | **27.99** | **62.87** | **22.29** | **0.54** | **1.10** | **1.76** |

关键 intuition：mAP 21.45 vs DriveArena 16.06，提升约 34%，说明 occupancy conditioning 让生成视频里的 car 更 detectable（occlusion 更合理、actor ID 更一致）。L2 trajectory error 1.76 vs real 1.65，gap 很小，说明 generative video 已经接近 real 数据上的 planner 表现。

### 5.3 Open-loop（Tab.3）

NAVSIM [8] 的四个 metric：NC (No Collision), DAC (Drivable Area Compliance), TTC (Time to Collision), PDMS (PDM Score)。

| Data Source | NC↑ | DAC↑ | TTC↑ | PDMS↑ |
|---|---|---|---|---|
| nuScenes | 0.993 | 0.995 | 0.947 | 0.910 |
| DriveArena | 0.792 | 0.942 | 0.771 | 0.636 |
| DriveArena* | 0.829 | 0.964 | 0.812 | 0.698 |
| **DrivingSphere** | **0.852** | **0.968** | **0.853** | **0.742** |

DriveArena* 是他们自己 re-run 的 DriveArena（公平对比同样 100 条 trajectory）。DrivingSphere 在 NC/TTC/PDMS 上都更好，主要因为视频 fidelity 更高，perception 错误更少。

### 5.4 Closed-loop（Tab.4）

| Method | PDMS↑ | RC↑ | ADS↑ |
|---|---|---|---|
| DriveArena | 0.6901 | 0.0641 | 0.0508 |
| **DrivingSphere** | **0.7281** | **0.1170** | **0.0851** |

- RC (Route Completion): ego 完成路线的比例。
- ADS (Average Driving Score): 综合指标。

RC 从 6.41% 提到 11.70%（接近翻倍），说明 UniAD 在 DrivingSphere 里能多走 1 倍距离不犯错——视觉质量直接 translate 到 driving behavior。但 absolute 数字很低（11.7%），说明 UniAD 在 closed-loop 里仍然很脆弱，这也说明 closed-loop sim 的价值。

---

## 6. 把它放在更大的 lineage 里

为了 build intuition，我把相关 work 分三类梳理：

### 6.1 Open-loop generative driving

- **BEVGen** [38] / **BEVControl** [49]: 从 BEV layout 生成 street view，2D 控制，单帧。
- **MagicDrive** [13] <https://github.com/cjiabo/MagicDrive>: 多视角 + 3D box 控制，但 box 是稀疏的，scene geometry 隐式。
- **DriveDreamer / DriveDreamer-2** [43, 56] <https://github.com/PJLab-ADG/drivedreamer>: 用 LLM 增强 world model，生成多视角视频，condition 是 action + HDMap。
- **Panacea** [45]: panoramic + controllable video。
- **Vista** [14]: generalizable world model with high fidelity。

这些都没解决 closed-loop，只是生成更漂亮的数据。

### 6.2 Closed-loop simulation（非 generative）

- **CARLA** [10] <https://carla.org>: 物理引擎，但 visual gap 大，real→CARLA domain shift 严重。
- **SUMO** [27] <https://sumo.dlr.de>: 2D traffic flow，无 visual。
- **MetaDrive** [23]: procedural generation，物理好但 visual 弱。
- **LimSim / LimSim++** [12, 46] <https://github.com/PJLab-ADG/LimSim>: 长时多场景 traffic sim，DrivingSphere 用它做 NPC。
- **NeurOnCAP** [26], **UniSim** [51], **MARS** [47]: NeRF/3DGS 重建 sim，被数据范围限制，不能生成长尾。
- **DriveArena** [50] <https://github.com/PJLab-ADG/DriveArena>: 第一个 generative closed-loop，用 2D sketch 控制。DrivingSphere 是它的直接升级。

### 6.3 Occupancy world models

- **OccWorld** [57] <https://github.com/wzzheng/OccWorld>: 4D occupancy 预测，closed-loop occupancy forecasting。
- **OccSora** [41] <https://github.com/wzzheng/OccSora>: 4D occupancy diffusion，trajectory-conditioned 生成。DrivingSphere 借了它的 4D VAE encoder。
- **SemCity** [22] <https://github.com/zoomin-lee/SemCity>: tri-plane diffusion 生成 outdoor 3D occupancy。DrivingSphere 的 baseline。

### 6.4 Diffusion & video 基础设施

- **DDPM** [17] <https://arxiv.org/abs/2006.11239>: 扩散模型基础。
- **VQVAE** [40] <https://arxiv.org/abs/1711.00937>: 离散 latent tokenizer。
- **Stable Diffusion / LDM** [35] <https://arxiv.org/abs/2112.10752>: latent diffusion。
- **ControlNet** [54] <https://github.com/lllyasviel/ControlNet>: 条件注入的标准做法。
- **PixArt-δ** [4] <https://github.com/PixArt-alpha/PixArt-delta>: ControlNet + DiT。
- **Open-Sora** [59] <https://github.com/hpcaitech/Open-Sora>: ST-DiT 架构，DrivingSphere 直接用。
- **Stable Video Diffusion** [2] <https://arxiv.org/abs/2311.15127>: video diffusion 的 base。
- **NeRF** [29] <https://arxiv.org/abs/2003.08934>: Fourier encoding 的来源。

### 6.5 End-to-end AD agents

- **UniAD** [19] <https://github.com/OpenDriveLab/UniAD>: planning-oriented E2E，论文主要 evaluator。
- **VAD** [20] <https://github.com/hustvl/VAD>: vectorized scene rep。
- **VADv2** [6]: probabilistic planning。
- **LMDrive** [36]: LLM + closed-loop。
- **DME-Driver** [15]: human decision + 3D perception。

### 6.6 评测基准

- **NAVSIM** [8] <https://github.com/autonomousvision/navsim>: non-reactive sim，open-loop 指标来源（NC/DAC/TTC/PDMS）。
- **nuScenes** [3] <https://www.nuscenes.org>: 数据底座。

---

## 7. 几个 critical thoughts（build intuition 用）

1. **为什么 occupancy 优于 BEV sketch 但还不够？** Occupancy 已经携带 depth/occlusion，但 DrivingSphere 仍然要 dual-path encoding——global 4D VAE + local 2D semantic projection。这暗示 4D VAE 单独的 attention pattern 在 pixel-level alignment 上不够强，必须用 2D semantic map 做 hard supervision。如果未来用 3D attention 更强的 transformer（比如更长 context、3D rotary positional embedding），可能可以省掉 local path。

2. **ID-aware encoding 是否本质是 retrieval？** 给每个 actor 一个 Fourier-encoded ID + caption，相当于在 cross-attention 里 lookup 一张"actor 资料表"。这和 retrieval-augmented generation (RAG) 的思路一致——把 instance-specific 外观特征当成外部知识库。DriveArena 没做这步，所以 actor 跨视角会漂移。

3. **Closed-loop 的 fidelity 瓶颈在哪儿？** 看 Tab.4，UniAD 只完成 11.7% 路线。问题不完全在 generator，更在 UniAD 本身在 OOD 视频上 generalization 有限。这说明 closed-loop 评测对被测模型也提出更高要求——它得 robust 到 synthetic video。

4. **Scene Expansion 的 outpainting 是不是伪扩展？** 公式 (4) 的 outpainting 本质是 in-context 生成，依赖 overlap 区域 $S_k^{\mathrm{partial}}$ 提供 boundary。如果两个区域语义冲突（一边是商场一边是郊区），merge 可能出现 weird transition。论文没讨论这种 corner case，是个潜在 failure mode。

5. **Auto-regressive video gen 的延迟问题。** 公式 (12) 一帧一帧生成，推理 latency 高，做不到 realtime closed-loop。论文里 closed-loop 是 offline batch 评测（100 条 trajectory），实际部署到 sim-in-the-loop 还需要大幅 distillation 或 caching。这和 GAIA-1、Vista 这类 parallel decoding 路线相比有 trade-off。

6. **occupancy resolution 的取舍。** nuScenes occupancy 通常 0.5m voxel，actor 的几何细节（车轮、后视镜）就丢了。actor bank 用 instance-level voxel 也面临同问题——如果 voxel 太粗，video 渲染时细节靠 diffusion "想象"，可能幻觉出错误结构。

7. **和 world model 路线的对照。** GAIA-1 / Vista / DriveWorldM 这些直接学 video → video 的 world model，跳过 explicit occupancy。DrivingSphere 选择 explicit occupancy 作为 bottleneck——好处是 controllable、interpretable、可 inject physics；代价是 pipeline 重，actor bank 要 hand-curate。两种路线的最终对决可能要看 long-tail generalization：explicit representation 的长尾依赖 actor bank 多样性，implicit world model 依赖训练数据多样性。

---

## 8. Limitations / Open questions

paper 自己承认的：
- dynamic agent behavior 还比较简单（依赖 LimSim）。
- 长尾场景覆盖有限。

我加几个：
- **物理 fidelity**：occupancy grid 是离散的，连续 contact dynamics（碰撞、滑动）不能直接 sim，需要额外 physics engine。
- **Sensor realism beyond camera**：没生成 LiDAR / radar，对 multi-modal AD 系统评测不够。
- **Lighting / weather**：text caption 能控一些（"rainy night"），但和真实天气分布对齐需验证。
- **Evaluator 选择偏差**：UniAD 训练在 nuScenes，evaluator 在 nuScenes-style 生成数据上跑，可能有 evaluator-aware overfitting。

---

## 9. 关键参考链接汇总

| Resource | URL |
|---|---|
| DrivingSphere project | <https://yanty123.github.io/DrivingSphere/> |
| nuScenes | <https://www.nuscenes.org> |
| UniAD | <https://github.com/OpenDriveLab/UniAD> |
| VAD | <https://github.com/hustvl/VAD> |
| DriveArena | <https://github.com/PJLab-ADG/DriveArena> |
| MagicDrive | <https://github.com/cjiabo/MagicDrive> |
| DriveDreamer | <https://github.com/PJLab-ADG/drivedreamer> |
| OccSora | <https://github.com/wzzheng/OccSora> |
| OccWorld | <https://github.com/wzzheng/OccWorld> |
| SemCity | <https://github.com/zoomin-lee/SemCity> |
| Open-Sora | <https://github.com/hpcaitech/Open-Sora> |
| PixArt-δ | <https://github.com/PixArt-alpha/PixArt-delta> |
| ControlNet | <https://github.com/lllyasviel/ControlNet> |
| LimSim | <https://github.com/PJLab-ADG/LimSim> |
| CARLA | <https://carla.org> |
| NAVSIM | <https://github.com/autonomousvision/navsim> |
| VQVAE | <https://arxiv.org/abs/1711.00937> |
| Lovász Loss | <https://arxiv.org/abs/1705.08790> |
| NeRF (Fourier enc) | <https://arxiv.org/abs/2003.08934> |
| CLIP | <https://github.com/openai/CLIP> |
| T5 | <https://github.com/google-research/text-to-text-transfer-transformer> |
| End-to-end AD survey | <https://arxiv.org/abs/2306.16927> (Li Chen et al. E2E survey) |

---

## 10. 一图总结架构

```
┌────────────────────────────────────────────────────────────────────────┐
│ Dynamic Environment Composition                                         │
│                                                                        │
│  BEV map ──┐                          ┌─→ static scene S_k             │
│            ├─→ OccDreamer (diffusion) ─┤   (occupancy grid)             │
│  text  ────┘                          └─→ Scene Expansion (outpaint)   │
│                                              ↓                         │
│  Actor Bank (CLIP retrieve) ───────→ A_n (voxel + caption + ID)         │
│  Traffic sim (LimSim) ─────────────→ P_n^t (trajectory)                │
│                                              ↓                         │
│                            π (compose) → W^t (4D occupancy world)       │
└────────────────────────────────────────────────────────────────────────┘
                                              ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Visual Scene Synthesis (VideoDreamer)                                  │
│                                                                        │
│  W^{0..T} ──┬─→ 4D VAE encoder ──→ F_global ──┐                       │
│             └─→ render+project ──→ M_v^t ──→ Image VAE ──→ F_view     │
│                                              ↓ cross-attn → F_occ      │
│                                                                        │
│  (P_n, I_n, L_A_n) ──Fourier+T5──→ F_A_n ──┐                           │
│  L_scene ──T5──→ F_W                       └→ concat → F_fuse          │
│                                                                        │
│  F_occ → ControlNet-DiT ─┐                                              │
│  F_fuse → CrossAttn ────┴→ ST-DiT (VSSA + Temporal + FFN)              │
│                            ↓ denoise                                   │
│                      V^{0..T} (6-view video)                           │
│                                                                        │
│  Auto-regressive: V^t = g_θ(V^{t-1}, Z^V, t)                           │
└────────────────────────────────────────────────────────────────────────┘
                                              ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Closed-Loop                                                            │
│                                                                        │
│  V^t ─→ F_driving (UniAD/VAD) ─→ c_ego^{t+1}                          │
│  W^t ─→ F_system (LimSim)    ─→ c_n^{t+1}                              │
│                              ↓                                         │
│                 P_n updated via Eq.6 → W^{t+1}                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 最后的 takeaway

DrivingSphere 的核心贡献可以浓缩成一句：**用 4D occupancy 作 explicit world state，让 closed-loop 仿真既有 generative 的 visual fidelity，又有 physics-aware 的 geometry consistency**。它的工程拼装比理论创新多——VQVAE + ControlNet + ST-DiT + Fourier ID encoding 这些都是已有组件，但组合方式（特别是 ID-aware actor encoding + dual-path occupancy conditioning）解决了 multi-view multi-frame consistency 的痛点。

对 AD 评测社区的影响是：以后的 closed-loop benchmark 可能都得显式建模 4D world state，单纯 2D 控制（DriveArena）会被认为是 insufficient。但 occupancy 的 resolution 上限和 actor bank 的多样性，会成为下一个 bottleneck。

Hope this builds the intuition you wanted, Andrej. 如果你想深挖某个具体模块（比如 VSSA 实现、Scene Expansion 的 merge 算法、或 Lovász loss 在 occupancy 上的具体梯度行为），告诉我，可以再展开。
