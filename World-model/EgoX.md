---
source_pdf: EgoX.pdf
paper_sha256: 0fa9867dd1176514077f78c83ccd914be38f00ed351524bb9742175816cf128c
processed_at: '2026-08-04T02:57:43-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EgoX 用人话讲一遍

好，Karpathy，我抛开学术腔，用大白话把这篇 paper 拆开讲，重点放在 "为什么这么设计" 而不是 "公式是什么"。

## 一句话版本

把别人拍你的第三人称视频，变成你自己眼睛里看到的第一人称视频 — 只需要一段 exo input，不需要任何 ego ground truth frame。

## 问题为什么难

你可以把它想成：你在看一段别人做饭的视频，摄像机在厨房角落架着，能拍到厨师全身。现在问你 — 如果你是那个厨师，你眼睛看到的是什么？

这事儿为什么 hard：

**First，视角差太 extreme**。普通的 camera control 模型比如 TrajectoryCrafter，它们擅长的是 "摄像机稍微挪一挪、转一转" — 从左边拍变右边拍，从远变近。但 exo-to-ego 是 "从厨房角落看厨师" 变成 "厨师眼睛看砧板"，camera 位置和方向几乎 180 度翻转。这已经不是 camera control 了，这是 view teleportation。

**Second，一大半画面是 unseen**。厨师第三人称视角能看到整个厨房，但第一人称只看得到砧板和手 — 厨房后墙、厨师自己的脸都看不见。这些 unseen region 模型得 hallucinate 出来，但得 hallucinate 得合理 — 砧板上有什么菜、手在干嘛，这些都得从 exo view 推出来。

**Third，exo view 里有大量 irrelevant 信息**。比如 exo view 背景里有人聊天、有窗户、有冰箱 — 这些对 ego view 完全无用。模型得知道 ignore 这些，只 attend 到厨师手部、砧板、食材这些 relevant region。

## 核心思路 — 三层信息叠在一起

EgoX 的 insight 是：这个任务单靠一个 signal 搞不定，得用三个 complementary 的 signal：

### Signal 1: Ego Prior — 粗糙但 viewpoint-aligned

先把 exo video lift 成 3D point cloud，再从 ego camera pose 重新渲染一遍。

你可以这么想：exo view 是一张 2D 照片，我把它 "解包" 成 3D 场景（用 depth estimation），然后把虚拟摄像机放到厨师眼睛的位置，重新拍一遍。

得到的结果 $P$ — paper 里叫 "ego prior video" — 长什么样？它和最终 ego view 视角一致，pixel 都对得上，但是：
- 很 noisy（depth estimation 有误差，point cloud 稀疏）
- 很 incomplete（dynamic object 被 mask 掉了，前景可能有大空洞）
- 颜色可能不准（point splatting 的 artifact）

所以 ego prior 提供的是 "哪里的 pixel 该长什么样" 的粗略 hint，但远远不够好。

**深度对齐这里有个坑** — 单帧 depth estimator 每帧估的 depth scale 不一致（monocular depth 的老问题），video depth estimator 时间一致但是 affine-invariant（只有相对深度没有 absolute scale）。EgoX 的做法是两者做 affine alignment，公式是：

$$D^f = \frac{1}{\hat{\alpha}/D^v + \hat{\beta}}$$

变量解释：
- $D^v$：video depth（时间一致但无 scale）
- $\hat{\alpha}_f, \hat{\beta}_f$：per-frame 学的 affine 参数（把 video depth 对齐到 monocular depth 的 scale）
- $D^f$：final depth，既有 scale 又时间一致

Fig 9 给了直观对比 — 没做 alignment，哪怕 ego camera 固定不动，背景也会 frame-by-frame 乱跳，因为 depth scale 在 drift。这个 artifact 会直接 confuse 下游的 video diffusion model。

### Signal 2: Exo Clean Latent — 全局 context

Ego prior 只覆盖了 ego view 能看到的一小部分，大量 region 还是空白的。这些空白得从 exo view 里找信息填。

EgoX 把 exo video encode 成 clean latent $x_0$，直接拼到 noisy ego latent $z_t$ 旁边。

**这里有个关键 design choice** — exo 和 ego 视角不一致，pixel 对不上，所以不能按 channel 拼（channel concat 是给 pixel-aligned 用的），得按 width 拼（width concat 把 exo latent 放在 ego latent 左边，让模型自己学怎么从 exo region warp 信息到 ego region）。

### Signal 3: Pretrained Video Diffusion Prior — 自然视频的 manifold

未seen region 的 plausibility 完全靠 Wan 2.1 这个 14B pretrained model 的 spatio-temporal prior。它知道 "厨房长什么样"、"手拿刀该是什么姿势"、"砧板上蔬菜大概什么颜色"。这是大量未seen region 的 ultimate fallback。

EgoX 用 LoRA (rank 256) 做 minimal adaptation — 只学怎么 fuse 这三个 signal，不碰 base model 的 visual reasoning 能力。这就是为什么 8 GPU × 1 天就能 train 完，而且 generalization 到 in-the-wild (The Dark Knight Joker 场景) 依然 work。

## Geometry-Guided Self-Attention — 最有意思的部分

问题：ego latent 里的某个 token（比如 "砧板左边那个菜叶"）在做 self-attention 时，要 attend 到 exo latent 里的 key tokens。光靠 appearance similarity 的话，它会 attend 到 exo view 里所有 "菜叶" — 厨房角落可能还有别的菜，或者窗户外的绿色植物。

EgoX 的解法：加一个 3D geometric bias。

具体做法 — 对每个 query token 和 key token，算它们从 ego camera center 出发的 3D direction vector，看这两个 direction 夹角多大。夹角小（方向一致）→ 这两个 token 大概对应 3D 空间同一个区域 → attention boost。夹角大 → 它们是 3D 空间不同区域 → attention suppress。

公式拆解：

Direction vector：
$$\hat{q} = \frac{\tilde{q} - c_i}{\|\tilde{q} - c_i\|_2}$$

- $\tilde{q}$：query token 在 3D world space 的位置（从 point cloud 投影到 latent patch）
- $c_i$：第 $i$ 帧 ego camera center
- $\hat{q}$：unit direction vector，从 camera center 指向 query 的方向

Bias 加到 attention logits 上：
$$s'_{m,n} = s_{m,n} + \log\big(g(\hat{q}_m, \hat{k}_n) \cdot \lambda_g\big)$$

- $s_{m,n} = \frac{q_m^\top k_n}{\sqrt{c}}$：standard attention logit（appearance similarity）
- $g(\hat{q}, \hat{k}) = \cos\text{sim}(\hat{q}, \hat{k}) + 1$：geometry bias，加 1 保证非负
- $\lambda_g$：bias 强度超参

Intuition：log 加在 logits 上，softmax 之后等价于把 $g(\hat{q}, \hat{k})$ 直接乘到 attention weight 上。所以最终 attention = appearance similarity × geometry alignment。两者都高才高分，任一低就被压。

**为什么 video 里不能像 image 那样用 RoPE 旋转矩阵？**

Image generation 里你只有一个 camera，用旋转矩阵把 query/key 旋转一下就行。Video 里每帧 ego camera center 都变 — 同一个 exo key token，在 frame 0 和 frame 1 对应的 direction vector 完全不同。所以没法 precompute 一个旋转矩阵，必须 per query-key pair 现算。

EgoX 用 additive log bias 的形式就是为了复用优化过的 attention kernel — 你只需要 precompute 一个 bias matrix 加上去，不用改 attention 内部的 matmul。

Fig 4 的 illustration 很清楚：同一个 exo key position（红色），不同帧因为 camera center 不同，direction vector 完全不同（红色 vs 橙色）。蓝-红 pair 方向相近 → 高分；绿-橙 pair 方向相反 → 低分。

## 实验 — 为什么 EgoX 完胜

### 主结果 (Table 1)

Seen scenes 上：
- PSNR: 16.05 vs 次优 14.53 (Exo2Ego-V) — 高出 1.5 dB
- IoU (object-level): 0.363 vs 次优 0.128 — 几乎 3 倍
- Location Error: 61.81 vs 次优 100.74 — 减半
- FVD: 184.47 vs 次优 508.69 — 减到三分之一

Object-level metric 比 image-level metric 更说明问题，因为 PSNR 对 unseen region 不公平（GT 和 generated 在 unseen region 本就不同）。Object-level 只看能 identify 的 object，看它们的位置对不对、形状对不对，这个才真正反映 geometric consistency。

Unseen scenes 上趋势一致，证明 model 没 overfitting。

### Ablation — 每个组件单独的 contribution

| Variant | PSNR | IoU | FVD |
|---|---|---|---|
| Full EgoX | 16.05 | 0.363 | 184.47 |
| w/o GGA | 14.77 | 0.326 | 254.08 |
| w/o Ego prior | 13.67 | 0.417 | 211.50 |
| w/o Clean latent | 15.07 | 0.376 | 343.33 |

三个组件都关键，但作用不同：
- **GGA** 去掉 → FVD 暴涨（temporal consistency 崩了）— 因为 geometric misalignment 导致 frame 间 attention 乱跳
- **Ego prior** 去掉 → PSNR 掉最多 — 因为丢了 pixel-aligned 的 fine-grained guidance
- **Clean latent** 去掉 → FVD 掉最猛 — 因为 exo latent 加噪后 fine-grained detail 全 blurred

Fig 6 给了 qualitative 对比：w/o clean latent 时模型连 spoon 和 small circular ingredients 都 missing — 这些都是 fine-grained detail，加噪直接没了。

### Conditioning strategy ablation (Table 4) — 验证 width/channel 分配

| Variant | PSNR | IoU | FVD |
|---|---|---|---|
| Ours (Prior channel, Exo width) | 16.05 | 0.363 | 184.47 |
| Prior width, Exo Channel | 13.83 | 0.213 | 274.14 |
| Prior width, Exo width | 14.85 | 0.261 | 242.83 |

Reversed conditioning（prior 用 width、exo 用 channel）性能掉得最狠 — 因为 exo channel concat 丢了 spatial structure，模型没法 implicit warp；prior width concat 丢了 pixel alignment。这验证了 "pixel-aligned 用 channel、unaligned 用 width" 的设计原则。

## 一些更深的 intuition

### 为什么 clean latent 比 SDEdit-style noisy latent 好

SDEdit 的哲学是：condition 和 target 加同等噪声，让模型在 "同等模糊度" 下做 translation。这在 image editing 里 work，因为 condition 和 target 视角接近，加噪后都模糊，模型只要做 moderate edit。

但 exo-to-ego 不一样 — exo view 提供的信息本来就稀疏（只覆盖 ego view 的一小部分），每一个 pixel 都很珍贵。你给 exo latent 加噪，等于把宝贵的高频信息直接扔了，模型只能从模糊的 exo latent 里猜细节。

EgoX 让 $x_0$ 全程 clean，模型每个 denoising step 都能 reference 到精确的 exo information。代价是模型要学怎么从 clean space "cross-reference" 到 noisy space，但 LoRA rank 256 足够学这个。

### 为什么 Object-level evaluation 比 PSNR 更公平

PSNR 是 pixel-wise 的，但 exo-to-ego 的 GT 和 generated 在 unseen region 本质不同 — GT 是真实拍的，generated 是 hallucinate 的，pixel 值不可能完全 match。强算 PSNR 会惩罚合理的 hallucination。

Object-level metric 只看能 identify 的 object（用 SAM2 segment + DINOv3 match），看它们的 bounding box 位置对不对、shape 对不对。这更反映 "geometric consistency" — object 在不在该在的地方，shape 对不对。

EgoX 在 Object metric 上差距最大（IoU 3x、Location Error 减半），说明它的核心优势就是 geometric accuracy。

### Pretrained VDM 的角色

这个任务看起来是 geometry + rendering 问题，但 EgoX 的成功很大程度上靠 Wan 2.1 这个 14B video diffusion model 的 prior。为什么？

因为未seen region 的 synthesis 本质是 "hallucinate 一个合理的厨房/球场/街道场景"。这需要海量视频数据里学到的 visual common sense — "厨房通常有橱柜"、"球场通常有观众席"。这种 common sense 没法从 4000 clips 的 Ego-Exo4D 里学到，必须靠 pretrained model。

EgoX 的 design philosophy 是：geometry 负责硬约束（viewpoint alignment、object position），pretrained VDM 负责软填充（texture、lighting、plausible scene layout）。LoRA 只学怎么 fuse 这两者。

### 为什么 LoRA work — 不需要 full fine-tune

Base model 已经懂 "视频该长什么样"，它缺的是：
1. 怎么读 width-concat 的 exo latent（新 input format）
2. 怎么读 channel-concat 的 ego prior latent（新 input format）
3. 怎么 interpret GGA 的 geometry bias（新 attention pattern）

这三样都是 "interface" 层面的 adaptation，不涉及 deep visual reasoning。所以 rank 256 的 LoRA 够了。这也是为什么 in-the-wild（The Dark Knight）能 generalize — base model 的 visual prior 是通用的，LoRA 学的 conditioning interface 也是通用的。

### Limitation — ego camera pose 依赖

Paper 老实说了：需要 ego camera pose $\phi$ 作为输入。Ego-Exo4D 提供 ground truth，但 in-the-wild 没有。他们的 workaround 是用 Viser 手动选 camera pose — 这在生产环境里不可行。

未来如果能自动估 head pose（比如 [head-pose-estimation](https://github.com/yinguobing/head-pose-estimation) 或 SMPL-based 方法），就能 end-to-end。但这本身是个 open problem — 从 third-person video 估 first-person camera trajectory，需要理解人的 head movement 和 gaze。

### Failure case 揭示的 task ambiguity

Fig 12 的例子：exo view 里一个人 bending one arm，另一 arm 被 occluded。模型猜成 "both arms extended"。

这其实不是 model failure，是 task inherent ambiguity — exo view 里看不到另一 arm 在哪，信息不够。即使人看这段 exo video 也猜不准。

这告诉我 exo-to-ego 的 theoretical upper bound 不是 100% fidelity — 因为 exo view 在 information 上就是 underdetermined 的。EgoX 能做到的是 "plausible generation"，不一定 "correct generation"。

## 可能的延伸方向

### Multi-ego-camera generation

现在是单 ego camera pose，如果改成 ego camera trajectory，可以做 walk-through — viewer 在场景里 "走动"。这对 AR/VR、virtual tourism 很有价值。

### Robotics — demonstration transfer

Robot learning 里，human demo 通常 third-person 拍的。如果能转成 robot POV（robot 眼睛里看到什么），直接 align robot's own observation space，对 imitation learning 帮助巨大。参考 [npj Robotics 2025](https://www.nature.com/articles/s44182-025-00014-0)。

### Film immersion

Paper 开头用 The Dark Knight 举例 — viewer 可以 "be the Joker"。如果结合 real-time rendering，可以让 viewer 选择 "我想从哪个角色视角看这段戏"。这改变了 cinema 的叙事方式。

### Memory 和 compute 的 trade-off

Width concat 让 latent 加倍，GGA 的 bias matrix 要 precompute 所有 query-key pair 的 direction similarity。14B model + LoRA + 加宽 latent + GGA bias，memory 不小。Table 5 显示 full model 10.5 min per video on H200 — 这个速度对 research demo 够，对 production 可能要优化。

潜在优化：
- GGA bias 可以 sparse 化（只算 top-k relevant pairs）
- Width concat 可以用 cross-attention 替代（但会失去 pretrained weight 的好处）
- Latent 可以 compress（用更小的 VAE 或 downsample exo latent）

## 一句话总结 EgoX 的 design philosophy

**把一个看似极端的 view synthesis 任务，decompose 成三个互补的 conditioning signal（ego prior 给 geometry、exo clean 给 context、pretrained VDM 给 manifold prior），每个 signal 用最适合它的方式 inject（pixel-aligned 用 channel concat、unaligned 用 width concat、attention 用 additive log bias），最后用 lightweight LoRA 学怎么 fuse。**

这个 decomposition 思路其实可以推广到其他 "extreme conditional generation" 任务 — 只要你能把 condition 拆成 "pixel-aligned 部分" + "unaligned context 部分" + "pretrained prior 部分"，就能复用这套 design pattern。

---

Web links for reference:
- Wan 2.1 (base model): [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- Ego-Exo4D (dataset): [arXiv:2403.16182](https://arxiv.org/abs/2403.16182)
- MoGe-2 (monocular depth): [arXiv:2507.02546](https://arxiv.org/abs/2507.02546)
- Video Depth Anything: [arXiv:2503.11551](https://arxiv.org/abs/2503.11551)
- ViPE (depth alignment): [arXiv:2508.10934](https://arxiv.org/abs/2508.10934)
- PyTorch3D (renderer): [github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)
- SAM2 (segmentation): [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
- DINOv3 (object matching): [arXiv:2508.13157](https://arxiv.org/abs/2508.13157)
- Exo2Ego-V (baseline): [arXiv:2311.17055](https://arxiv.org/abs/2311.17055)
- EgoExo-Gen (baseline): [arXiv:2504.11732](https://arxiv.org/abs/2504.11732)
- TrajectoryCrafter (baseline): [arXiv:2503.05638](https://arxiv.org/abs/2503.05638)
- Wan VACE (baseline): [arXiv:2503.07598](https://arxiv.org/abs/2503.07598)
- 4Diff (related): [arXiv:2410.07557](https://arxiv.org/abs/2410.07557)
- ControlNet (design inspiration): [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
- RoPE (positional encoding): [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- EscherNet (camera ray encoding): [arXiv:2403.12018](https://arxiv.org/abs/2403.12018)
- SDEdit (noisy latent baseline): [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)
- In-context LoRA: [arXiv:2410.23775](https://arxiv.org/abs/2410.23775)
- Gen3C (point cloud condition): [arXiv:2412.06060](https://arxiv.org/abs/2412.06060)
- Viser (3D visualization): [github.com/nerfstudio-project/viser](https://github.com/nerfstudio-project/viser)
- head-pose-estimation: [github.com/yinguobing/head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)
- SMPL: [arXiv:1507.04660](https://arxiv.org/abs/1507.04660)
- VideoX-Fun (Wan Fun Control): [github.com/aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)

---

# EgoX Paper 深度解析

Karpathy 好，这篇 EgoX 我仔细看了，整体思路非常优雅，核心 insight 是把一个看似极端的 view synthesis 任务拆解成三个互补的 conditioning signal，让 pretrained video diffusion model 做 minimal adaptation 就能干活。让我把它拆开讲，重点 build intuition。

## 1. Task Framing：为什么这是个困难问题

任务形式化：
- **Input**: exocentric video $X = \{X_i\}_{i=0}^{F}$，外加目标 egocentric camera pose $\phi = \{\phi_i\}_{i=0}^{F}$
- **Output**: egocentric video $Y = \{Y_i\}_{i=0}^{F}$

这里 $F$ 是 frame 数量，下标 $i$ 是 frame index。$\phi_i$ 是第 $i$ 帧的 ego camera extrinsic matrix。

关键挑战其实有三层，paper 在 Sec 1 把它讲得很清楚：
1. **Extreme camera pose translation** — 不是 moderate viewpoint change，是从 third-person 到 first-person 的极端 jump，visible field of view 几乎完全翻转
2. **Large unseen regions** — exo view 里看不到的 region 必须 plausibly synthesize，这需要 scene understanding
3. **Selective attention** — exo view 里有大量无关背景信息需要 suppress，只有一小部分对应 ego view 的可见区域

这就是为什么 TrajectoryCrafter 这类 camera control model 会 fail — 它们设计假设是 modest viewpoint change。

## 2. Method 三大组件

### 2.1 Egocentric Point Cloud Rendering (Sec 3.1) — 提供 "ego prior"

核心思路：先用 3D 中间表示把 exo video "lift" 到 3D，再从 ego camera pose 重新渲染，得到一个粗糙但 viewpoint-aligned 的 ego prior video $P$。

**深度估计双路径融合** — 这一段我觉得是整个 pipeline 里最 underappreciated 的工程细节：

- $D^m \in \mathbb{R}^{F \times H \times W}$：单帧 monocular depth estimator (MoGe-2, [arXiv:2507.02546](https://arxiv.org/abs/2507.02546)) 估计的 depth。优点：per-frame 准确、有 metric scale。缺点：frame 间 scale 不一致
- $D^v \in \mathbb{R}^{F \times H \times W}$：video-based temporal depth estimator (Video Depth Anything, [arXiv:2503.11551](https://arxiv.org/abs/2503.11551)) 估计的 depth。优点：temporally smooth。缺点：affine-invariant（只有相对深度）

两者的对齐公式 (Eq. 1)：

$$D^f = \frac{1}{\hat{\alpha}/D^v + \hat{\beta}}$$

变量含义：
- $D^f$：final aligned depth map，时间上一致且 metric-accurate
- $\hat{\alpha} = \{\hat{\alpha}_f\}_{f=0}^{F}$、$\hat{\beta} = \{\hat{\beta}_f\}_{f=0}^{F}$：per-frame affine transformation parameters（下标 $f$ 是 frame index，hat 表示 momentum-based 优化得到的估计值）
- 这个公式形式上是 inverse depth space 做的 affine，等价于 depth 空间做 harmonic transform — 这是 ViPE ([arXiv:2508.10934](https://arxiv.org/abs/2508.10934)) 里的做法

**Intuition**: dynamic object 被单独 mask 掉，只用 static background 做对齐和渲染 — 这是因为 dynamic geometry 估计噪声大，会影响 alignment 优化。

最终用 PyTorch3D 的 point cloud renderer ([GitHub](https://github.com/facebookresearch/pytorch3d)) 渲染：

$$P = \text{render}(X, D^f, \phi)$$

这里 $P \in \mathbb{R}^{F \times 3 \times H \times W'}$ — 注意 width 是 $W'$ 不是 $W$，因为 ego camera 的 FOV 可能不同。$X$ 是 RGB video，$\phi$ 是 ego camera poses。

**为什么这个 prior 重要**: Fig 9 给了一个很直观的 visualization — 没做 depth alignment 的话，monocular depth 每帧 scale drift，结果 ego camera 即使 fixed，背景也会乱动，confuse 生成模型。Sec G.2 专门 ablate 这一点。

### 2.2 Unified Conditioning Strategy (Sec 3.2) — 最核心的设计

Base model：Wan 2.1 (14B) Image-to-Video 的 inpainting variant ([arXiv:2503.20314](https://arxiv.org/abs/2503.20314))。选 inpainting variant 是因为它天生支持 channel-wise concat noisy latent + clean condition latent。

**两个 latent 进入 diffusion 的方式不同，这是 paper 最 elegant 的设计**：

| Latent | 来源 | Concat 维度 | 为什么 |
|---|---|---|---|
| $p_0$ (ego prior) | Point cloud rendering | **Channel-wise** | 与 target $z_t$ pixel-aligned，channel concat 注入 fine-grained 几何 |
| $x_0$ (exo clean) | VAE encode exo video | **Width-wise** | 与 $z_t$ 不 pixel-aligned，width concat 让模型 implicit 学 spatial warping |

变量说明：
- $x_0 \in \mathbb{R}^{f \times c \times h \times w}$：exo latent，下标 0 表示 timestep 0（clean）
- $p_0 \in \mathbb{R}^{f \times c \times h \times w'}$：ego prior latent
- $z_t \in \mathbb{R}^{f \times c \times h \times w'}$：noisy ego latent，下标 $t$ 是 diffusion timestep
- $f$：latent frame 数；$c$：latent channel；$h, w$：latent spatial dims

整体 denoising step 公式 (Eq. 3)：

$$z_{t-1} = f_\theta(x_0, z_t \,|\, x_0, p_0 \,|\, m^1, m^0)$$

- $f_\theta$：single-step denoising function (参数 $\theta$)
- $m$：binary mask，$m^1$ 标记 conditioning region（来自 prior），$m^0$ 标记 synthesis region
- $z_t$ 是唯一被更新的 latent，$x_0$ 始终 clean、始终 fixed

**Clean latent 设计 vs SDEdit 的关键差异**：

In-context LoRA ([arXiv:2410.23775](https://arxiv.org/abs/2410.23775)) 和 SDEdit ([arXiv:2108.01073](https://arxiv.org/abs/2108.01073)) 的做法是给 condition latent 也加噪，让它和 noisy target 在同一噪声级别。EgoX **反其道而行**：$x_0$ 全程 clean，永远保持 fine-grained details，让模型随时可以 reference 到精确的 exo 信息。

这个设计的 ablation (Table 2 "w/o clean latent" 行) 显示 FVD 从 184.47 → 343.33，掉了 86%，非常 dramatic。Fig 6 最后一行可以看到，没有 clean latent 时模型连 spoon 和 small circular ingredients 都 missing — 这些都是 fine-grained detail，加噪声直接 blur 掉了。

**Intuition**: 这个任务里 exo view 提供的信息非常稀疏，每一个 pixel 都很珍贵，加噪声就是浪费信息。

### 2.3 Geometry-Guided Self-Attention (GGA) (Sec 3.3) — 最有意思的模块

这个模块解决的问题：ego latent 的 query token 在 attention 到 exo latent 的 key token 时，光看 appearance similarity 不够 — 会 attend 到无关的 background region。需要加一个 3D geometric prior。

**3D direction vector 计算**：

对 ego camera center $c_i \in \mathbb{R}^3$（frame $i$ 的 camera center in world space）：

$$\hat{q} = \frac{\tilde{q} - c_i}{\|\tilde{q} - c_i\|_2}, \quad \hat{k} = \frac{\tilde{k} - c_i}{\|\tilde{k} - c_i\|_2}$$

- $\tilde{q}, \tilde{k} \in \mathbb{R}^3$：query 和 key token 在 3D world space 中的位置（从 point cloud 投影到 latent patch）
- $\hat{q}, \hat{k}$：unit direction vectors
- 下标 $i$ 表示 frame index — 关键 insight：**camera center 每帧变化**，所以同一个 $\tilde{k}$ 位置在不同帧对应不同的 $\hat{k}$

Geometry bias function (Eq. 5)：

$$g(\hat{a}, \hat{b}) = \cos\text{sim}(\hat{a}, \hat{b}) + 1$$

加 1 是为了在 log 之前保证非负（cos sim 范围 [-1, 1]，加 1 后变 [0, 2]）。

Modified attention logits (Eq. 4)：

$$s'_{m,n} = s_{m,n} + \log\big(g(\hat{q}_m, \hat{k}_n) \cdot \lambda_g\big)$$

- $s_{m,n} = \frac{q_m^\top k_n}{\sqrt{c}}$：standard scaled dot-product attention logit，$c$ 是 channel dimension（scaling factor）
- $m$：query index；$n$：key index
- $\lambda_g$：geometry bias 强度超参

最终 attention weight (Eq. 6-7)：

$$a_{m,n} = \frac{\exp(s'_{m,n})}{\sum_{j=1}^{l} \exp(s'_{m,j})} = \frac{\exp(s_{m,n}) \cdot g(\hat{q}_m, \hat{k}_n) \cdot \lambda_g}{\sum_{j=1}^{l} \exp(s_{m,j}) \cdot g(\hat{q}_m, \hat{k}_j) \cdot \lambda_g}$$

- $l$：ego token sequence length
- 分母 $j$ 是 sum over all keys

**这个公式形式的妙处**：log 加在 logits 上，softmax 后等价于 multiplicative bias $g(\hat{q}_m, \hat{k}_n)$ 直接乘到 attention weight 上。几何上 misaligned 的 token（$g$ 接近 0）被直接抑制；aligned 的 token（$g$ 接近 2）被 boost。同时保留 appearance similarity 的作用，两者乘积决定最终 attention。

**为什么不用 RoPE-style 旋转矩阵？**

Paper 在 Sec 3.3 末尾解释了：image generation 里可以用旋转矩阵乘到 query/key 上（如 EscherNet [arXiv:2403.12018](https://arxiv.org/abs/2403.12018)、RoPE [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)）。但 video 里 camera center 每帧变化，必须 per query-key pair 重算 direction，旋转矩阵方案不可行。所以改成 **additive bias on logits** 形式 — 这样可以复用优化过的 attention kernel。

Fig 4 是这一段的最佳 illustration：橙色和红色是同一个 exo key token position，但对应不同的 ego camera center，所以 direction vector 完全不同。蓝-红 pair direction 相似 → 高分；绿-橙 pair direction 相反 → 低分。

**Implementation detail (Sec F.1)**：
- 3D direction vectors 在 pixel space 计算
- 按 $4 \times 16 \times 16$ patch average 下采样匹配 VAE 的 downsampling factor（temporal × height × width）
- 预计算一次，避免 runtime overhead
- ego-to-exo 和 exo-to-ego attention 用单独 kernel，避免 memory 爆炸

## 3. 架构图解析 (Fig 3)

整体 pipeline：

```
Exocentric Video X
        │
        ├──→ MoGe-2 (monocular depth) ──┐
        │                                ├──→ Affine Alignment ──→ D^f
        └──→ Video Depth Anything ──────┘                              │
                                                                         │
                                                                         ↓
                                Point Cloud Renderer ←── Ego Camera Pose φ
                                         │
                                         ↓
                                  Ego Prior P
                                         │
                ┌────────────────────────┴────────────────────────┐
                ↓                                                  ↓
         VAE Encode                                          VAE Encode
         (p_0: ego prior latent)                              (x_0: exo clean latent)
                │                                                  │
                │   z_t (noisy ego latent)                          │
                │   │                                              │
                ↓   ↓                                              ↓
        [Channel concat]                              [Width concat]
        (p_0 + z_t along channel)                     (x_0 + z_t along width)
                │                                                  │
                └─────────────────────┬────────────────────────────┘
                                      ↓
                  Video Diffusion Model + GGA + LoRA
                                      ↓
                            z_0 (denoised ego latent)
                                      ↓
                              VAE Decode (ego part only)
                                      ↓
                          Egocentric Video Y
```

关键 insight：width concat 之后模型输出也是 width-doubled，最后只 decode ego 的那半边。

## 4. Experiments 实验数据深度分析

### 4.1 主结果 (Table 1)

**Seen Scenes** (400 test clips)：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-I ↑ | Location Err ↓ | IoU ↑ | Contour Acc ↑ | FVD ↓ | Temporal Flicker ↑ | Motion Smooth ↑ | Dynamic Degree ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Exo2Ego-V | 14.53 | 0.384 | 0.569 | 0.774 | 156.66 | 0.074 | 0.364 | 622.47 | 0.960 | 0.966 | 0.985 |
| TrajectoryCrafter | 13.05 | 0.375 | 0.606 | 0.780 | 100.74 | 0.128 | 0.427 | 546.09 | 0.960 | 0.980 | 0.947 |
| Wan Fun Control | 12.25 | 0.463 | 0.617 | 0.810 | 112.57 | 0.076 | 0.417 | 595.07 | 0.968 | 0.980 | 0.901 |
| Wan VACE | 12.95 | 0.413 | 0.626 | 0.829 | 109.62 | 0.114 | 0.376 | 508.69 | 0.989 | 0.994 | 0.673 |
| **EgoX** | **16.05** | **0.556** | **0.498** | **0.896** | **61.81** | **0.363** | **0.546** | **184.47** | 0.977 | 0.990 | 0.974 |

观察：
- **Image criteria 全面领先**：PSNR +1.52 over 次优 Exo2Ego-V，SSIM 高出 17 个百分点
- **Object criteria 差距最大**：IoU 0.363 vs 次优 0.128 (TrajectoryCrafter) — 几乎 3 倍。Location Error 61.81 vs 次优 100.74 — 减半
- **FVD 184.47 vs 次优 508.69** — 几乎 3 倍降低
- Wan VACE 在 temporal metrics 上看起来高，但 Dynamic Degree 只有 0.673 — 它在生成静态视频，所以 flicker 自然低。这是 paper 在 Sec 4.3 特别指出的 trap

**Unseen Scenes** (100 clips) 趋势一致，EgoX 全场最优，证明 generalization 而非 overfitting。

### 4.2 Ablation (Table 2)

| Variant | PSNR | IoU | FVD | Dynamic Degree |
|---|---|---|---|---|
| **EgoX (full)** | 16.05 | 0.363 | 184.47 | 0.974 |
| w/o GGA | 14.77 | 0.326 | 254.08 | 0.877 |
| w/o Ego prior | 13.67 | 0.417 | 211.50 | 0.802 |
| w/o Clean latent | 15.07 | 0.376 | 343.33 | 0.864 |

观察：
- **w/o GGA**: IoU 略降但 FVD 爆炸式退化（184 → 254），Dynamic Degree 降到 0.877 — 说明几何 misalignment 导致 temporal 不稳定
- **w/o Ego prior**: PSNR 掉最多 (16.05 → 13.67)，说明 explicit pixel-wise guidance 是 fidelity 的关键
- **w/o Clean latent**: FVD 掉最多 (184 → 343)，Dynamic Degree 也掉 — fine-grained detail loss 影响整体动态质量

### 4.3 Conditioning Strategy Ablation (Table 4)

这个 ablation 验证了 width/channel 分配的设计：

| Variant | PSNR | IoU | FVD |
|---|---|---|---|
| **Ours (Prior channel, Exo width)** | 16.05 | 0.363 | 184.47 |
| Prior width, Exo Channel | 13.83 | 0.213 | 274.14 |
| Prior width, Exo width | 14.85 | 0.261 | 242.83 |

**Reversed conditioning (Prior width, Exo channel)** 性能掉得最狠 — 因为 exo channel concat 失去了 spatial structure，模型无法 implicit warp；同时 prior width concat 失去了 pixel-level alignment。Paper 在 Sec G.3 给了很清晰的论证。

### 4.4 GGA Training vs Inference-only (Table 4 最后一行)

| Variant | FVD |
|---|---|
| GGA in train + inference | 184.47 |
| GGA only in inference | 193.82 |

差距不大但有意义 — 模型需要训练时见过 geometry-biased attention distribution 才能正确 interpret 这个 bias。

### 4.5 Runtime (Table 5)

| Variant | Runtime |
|---|---|
| Full EgoX | ~10.5 min |
| w/o GGA | ~6.5 min |
| w/o Ego Prior | ~6.5 min |
| w/o Clean Latent | ~6.5 min |

GGA 增加 4 min runtime，但 paper Sec H.2 论证这是必要投资 — Fig 10 显示 w/o GGA 会把 exo view 里不可见区域的事件 leak 进 ego view（attention 到无关 region）。

## 5. 一些深入的 intuition 与延伸思考

### 5.1 为什么这套设计 work — 信息论视角

EgoX 的三路 conditioning 实际上覆盖了三个不同频段的信息：
- **$p_0$ (ego prior)**: 高频、pixel-aligned 的几何信息，但 noisy + incomplete
- **$x_0$ (exo clean)**: 中频、global context、未对齐，但 clean + 完整
- **Pretrained VDM**: 低频、自然图像/视频 manifold 的 prior，fill in 完全 unseen 的 region

三者 complementary，LoRA 只需要学怎么 fuse 它们，不用从头学 visual reasoning。

### 5.2 与 4Diff 的对比 — cross-attention 的失败

4Diff ([arXiv:2410.07557](https://arxiv.org/abs/2410.07557)) 用 cross-attention 把 exo feature 注入。Paper Sec 2.1 指出这个设计的问题：cross-attention 必须从头训 attention 层，无法复用 pretrained weights，generalization 差。

EgoX 的 channel-wise + width-wise concat 是 "non-invasive" conditioning — 不破坏 attention 结构，可以最大限度保留 VDM 的 spatio-temporal prior。这与 ControlNet ([arXiv:2302.05543](https://arxiv.org/abs/2302.05543)) 的设计哲学一脉相承。

### 5.3 GGA 与一些 related work 的关系

- **与 RoPE 的区别**：RoPE ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) 用旋转矩阵编码相对位置，是 multiplicative bias 在 attention logits 上。EgoX 的 GGA 也是 log-additive → 等价 multiplicative，但 bias 来自 3D 几何而非 learned positional encoding。差异：RoPE 是 translation-invariant 的相对位置，GGA 是 camera-center-dependent 的绝对方向。

- **与 EscherNet 的区别**：EscherNet 用 camera ray encoding 做新视角合成，但针对 image set。EgoX 把这个 idea 推广到 video，并且处理了 camera center per-frame 变化的问题。

- **与 Gen3C ([arXiv:2412.06060](https://arxiv.org/abs/2412.06060))** 的对比：Gen3C 也用 point cloud rendering 作为 condition，但它的应用场景是 moderate camera control，EgoX 是 extreme view translation。

### 5.4 局限与 failure case

Paper 在 Sec H.5 提到：当 exo frame 含 ambiguous action（如一个人 bending one arm，另一 arm 部分被 occluded），模型可能 misinterpret 成 "both arms extended"。Fig 12 给了 visualization。

这其实揭示了一个更深层的问题：**exo view 本身在信息上就是 underdetermined 的** — 即使人也猜不出来。这种 case 不能算 model failure，是 task ambiguity 的 inherent limit。

### 5.5 Ego camera pose 的依赖

Paper 在 Sec 5 明确写了 limitation — 需要 egocentric camera pose $\phi$ 作为输入。Ego-Exo4D 提供 ground truth，但 in-the-wild 没有。Paper 的 workaround (Sec F.2) 是用 Viser ([GitHub](https://github.com/nerfstudio-project/viser)) 手动选 camera pose。

未来方向：自动 head-pose estimation，可以参考 head-pose-estimation ([GitHub](https://github.com/yinguobing/head-pose-estimation)) 或 SMPL-based 方法 ([arXiv:1507.04660](https://arxiv.org/abs/1507.04660))。这块如果能做自动，整个 pipeline 就是 end-to-end。

### 5.6 Object Criteria 评估协议 (Sec F.3)

这个 evaluation design 我觉得很 smart，值得单独讲：

1. SAM2 ([arXiv:2408.00714](https://arxiv.org/abs/2408.00714)) 在 GT 和 generated video 上分别 segment + track 所有 object
2. DINOv3 ([arXiv:2508.13157](https://arxiv.org/abs/2508.13157)) 提取每个 object 的 appearance feature
3. 计算 cosine similarity (Eq. 8)：$s_{i,j} = \frac{\mathbf{f}_i^{GT} \cdot \mathbf{f}_j^{model}}{\|\mathbf{f}_i^{GT}\|_2 \|\mathbf{f}_j^{model}\|_2}$
4. 阈值 $\tau_{sim} = 0.9$ 过滤高 confidence 匹配
5. 对匹配上的 pair 算 3 个 metric：
   - **Location Error** (Eq. 9): $\mathcal{E}_{i,j}^{loc} = \|\mathbf{c}_i^{GT} - \mathbf{c}_j^{model}\|_2$，bounding box center 的欧氏距离
   - **Bounding Box IoU** (Eq. 10): standard IoU
   - **Contour Accuracy** (Eq. 11): $\text{IoU}_{i,j}^{contour} = \frac{|C_i^{GT} \cap C_j^{model}|}{|C_i^{GT} \cup C_j^{model}|}$，SAM2 mask 上的 IoU

这套 protocol 比 PSNR/SSIM 更能捕捉 "geometric consistency" — 因为传统 pixel metric 对 unseen region synthesis 不公平（GT 和 generated 在这些区域本就不同）。Object-level metric 只看 identifiable object，更公平。

### 5.7 LoRA Rank 256 的选择

Paper 用 rank=256，batch size=1，8 H200 一天训完。rank 256 算 LoRA 里偏高的（一般 8-64），但任务本身需要 significant adaptation — 既要学 width concat 的 spatial warping，又要学 channel concat 的 pixel alignment fusion，还要适应 GGA-biased attention。所以高 rank 合理。

### 5.8 与 EgoExo-Gen ([arXiv:2504.11732](https://arxiv.org/abs/2504.11732)) 的对比

EgoExo-Gen 需要 first ego frame 作为额外输入 — 这把任务退化成 conditional video prediction，大大简化。EgoX 不需要任何 ego frame，纯 exo input，更接近实际应用场景。这也意味着 EgoX 在 deployment 时只需要单 exo camera，user experience 好得多。

### 5.9 与 Exo2Ego-V ([NeurIPS 2024](https://arxiv.org/abs/2311.17055)) 的对比

Exo2Ego-V 需要 4 个 simultaneous exo camera views — 这极大缩小了 uninformed region，但 deployment cost 极高。EgoX 用单 exo view + point cloud prior 弥补，更 scalable。

### 5.10 Dataset 选择的考量

4000 clips from Ego-Exo4D ([CVPR 2024](https://arxiv.org/abs/2403.16182)) — 这个 dataset 同时有 exo 和 synchronized ego view，是天然的训练数据。但要注意 Ego-Exo4D 主要是 skilled activity (cooking, sports 等)，可能 domain-biased。in-the-wild 实验 (The Dark Knight Joker scene) 是真正的 OOD 测试，效果依然不错，证明 pretrained VDM 的 prior 足够强。

### 5.11 Text Prompt 的角色 (Sec F.4)

Paper 用 GPT-4o 自动生成 text prompt，描述 exo 和 ego view 的 scene 和 action。Table 6 给了 system prompt。这其实是个 silent contributor — Wan 2.1 是 text-conditioned 的，必须有 prompt。VLM 生成 prompt 是个合理的自动化方案。Fig 18 有几个例子，质量挺高。

### 5.12 联想 — 这个 framework 能扩展吗？

- **AR/VR 场景**: 把 exo video（比如别人拍的视频）转成 first-person，让 user 沉浸式体验
- **Robotics**: 参考 [npj Robotics 2025](https://www.nature.com/articles/s44182-025-00014-0)，ego view 是 robot learning 的重要视角。EgoX 可以从 human demo 的 third-person video 直接生成 robot POV
- **Film/Entertainment**: Paper 开头就用 The Dark Knight 举例 — viewer 可以 "be the Joker"
- **Sports**: MLB player perspective 从 broadcast 视频生成
- **多ego camera generation**: 当前是单 ego pose，如果改成 trajectory of ego poses，可以做 immersive walk-through

### 5.13 一个 potential issue — 动态物体

Point cloud rendering 时 dynamic object 被 mask 掉了 (Sec 3.1)，意味着 ego prior $p_0$ 里 dynamic object 区域是空白的。这些区域的生成完全依赖 pretrained VDM 的 prior + $x_0$ 的 appearance。这是为什么 GGA 这么重要 — 它确保 attention 不会乱找 dynamic object 信息。

### 5.14 Memory footprint

Width concat 让 latent 加倍 (从 $w'$ 到 $w + w'$)，加上 LoRA、GGA 的 attention bias precomputation，整体 memory 不小。Paper 用 140GB H200 训，这个配置意味着单 GPU 能放下 14B Wan + LoRA + 加宽 latent。如果换 80GB A100 可能需要 offload。

### 5.15 关于 evaluation 中的 fairness

Paper 在 Sec 4.1 强调 baselines 用相同训练数据 fine-tune。TrajectoryCrafter, Wan Fun Control, Wan VACE 都用同一份 4000 clips。这点很重要，否则就是比 base model 而不是比 method。

唯一没法 fair 对比的是 EgoExo-Gen（没开源）和 Exo2Ego-V（需要 4 个 exo view，输入都不一样，没法 apple-to-apple）。Paper 诚实标注了这一点。

## 6. 总结 — 这篇 paper 给我的几个 take-away

1. **Decompose 极端 task 到可 manage 的 conditioning 信号** — ego prior (geometric) + exo clean (context) + VDM prior (manifold)。每个 component 各司其职
2. **Clean vs Noisy latent 的选择有 deep implication** — 高价值 condition signal 应该保持 clean
3. **Additive log bias 形式让 attention kernel 可复用** — 当 bias 是 pairwise 而非 global 时这是关键
4. **Width vs Channel concat 不是 arbitrary** — 取决于是否 pixel-aligned，这个 design choice 可以推广到其他 conditional generation task
5. **LoRA 让 14B model minimal adaptation** — 8 GPU × 1 day 训 14B，效率惊人
6. **Object-level evaluation 比 pixel-level 更公平** — 对 view synthesis 这种有 unseen region 的 task，PSNR 本质不公平

Web links for reference:
- EgoX (本项目): 暂未找到 arXiv link, 见 paper 文件
- Wan 2.1: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- Ego-Exo4D: [arXiv:2403.16182](https://arxiv.org/abs/2403.16182)
- MoGe-2: [arXiv:2507.02546](https://arxiv.org/abs/2507.02546)
- Video Depth Anything: [arXiv:2503.11551](https://arxiv.org/abs/2503.11551)
- ViPE: [arXiv:2508.10934](https://arxiv.org/abs/2508.10934)
- PyTorch3D: [github.com/facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)
- SAM2: [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
- DINOv3: [arXiv:2508.13157](https://arxiv.org/abs/2508.13157)
- SDEdit: [arXiv:2108.01073](https://arxiv.org/abs/2108.01073)
- In-context LoRA: [arXiv:2410.23775](https://arxiv.org/abs/2410.23775)
- Exo2Ego-V (Exocentric-to-Egocentric Video Generation, NeurIPS 2024): [arXiv:2311.17055](https://arxiv.org/abs/2311.17055)
- EgoExo-Gen: [arXiv:2504.11732](https://arxiv.org/abs/2504.11732)
- TrajectoryCrafter: [arXiv:2503.05638](https://arxiv.org/abs/2503.05638)
- Wan VACE: [arXiv:2503.07598](https://arxiv.org/abs/2503.07598)
- 4Diff: [arXiv:2410.07557](https://arxiv.org/abs/2410.07557)
- ControlNet: [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
- RoPE: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- EscherNet: [arXiv:2403.12018](https://arxiv.org/abs/2403.12018)
- Gen3C: [arXiv:2412.06060](https://arxiv.org/abs/2412.06060)
- Viser: [github.com/nerfstudio-project/viser](https://github.com/nerfstudio-project/viser)
- SMPL: [arXiv:1507.04660](https://arxiv.org/abs/1507.04660)
- head-pose-estimation: [github.com/yinguobing/head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)
- VideoX-Fun (Wan Fun Control): [github.com/aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)

如果你感兴趣某个具体细节我深入再讲 — 比如 GGA 在 latent space 的 patch-level direction 下采样、或者 LoRA rank 256 的具体参数 placement、或者 in-the-wild scene 的 ego pose 交互式选取 process，都可以再展开。
