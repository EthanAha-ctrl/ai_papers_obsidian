---
source_pdf: AnyCamVLA Zero-Shot Camera Adaptation for Viewpoint Robust Vision-Language-Action
  Models.pdf
paper_sha256: 0ccebc53d64421b18f0dbccb564d6d88821f9602d61c82645f64415f3b8193f3
processed_at: '2026-08-18T01:01:53-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# AnyCamVLA 的人话版

## 一句话总结

VLA 训练完对 camera 位置特别敏感，稍微动一下就废；这篇文章说：**别动 policy，测试时用个 NVS 模型把图像"翻译"回训练时的视角再喂给 policy**。就这么简单。

## 这个问题有多严重

你辛辛苦苦 fine-tune 一个 π0.5，收了 20 条 demo，camera 位置一丝不差。部署的时候 wrist camera 被人碰了一下，偏了 3 cm——success rate 直接从 90% 掉到 40%。这不是个例，[LIBERO-Plus](https://arxiv.org/abs/2510.13626) 系统测过，所有 VLA 都这样。

为什么 VLA 这么脆？因为它就是个 $\pi_\theta(I, l) \to a$ 的函数拟合器，训练时 camera 是固定的，它就把 camera position 当成 implicit background assumption 学进去了。VLM 阶段见过的海量 viewpoint robustness，在 fine-tune 时全被 "specialize 掉"”。

这跟 internet VLM 的 robustness 形成讽刺对比：VLM 见过几亿张不同角度的猫，能认出任何角度的猫；VLA fine-tune 后只认得训练时那个角度的桌面。

## 之前怎么办

两条路，都有硬伤：

**路线 A：收各种角度的 demo 重新 fine-tune**
- 贵。每条 demo 都得 teleoperate，robotics data 是最稀缺的资源
- 会忘。[Catastrophic forgetting](https://www.sciencedirect.com/science/article/pii/S1474869289100038) 是 connectionist model 的老毛病，fine-tune viewpoint data 的同时 original view 性能往下掉
- 不 transfer。Paper Figure 3 实测：用 1 个 task 的多角度数据 fine-tune，其他 9 个 task 的多角度性能反而下降。viewpoint generalization 是 task-specific 的 visual pattern，不 transfer

**路线 B：改架构加 3D feature**
- [GeoAwareVLA](https://arxiv.org/abs/2509.14117) 用 [VGGT](https://arxiv.org/abs/2503.08563) 替换 RGB encoder，提 3D-aware feature
- [3D Diffusion Policy](https://arxiv.org/abs/2403.03954) 用 point cloud
- 问题：VLA 的 magic 就在于继承 VLM 的 internet-scale RGB prior。你一改输入 modality，VLM pre-training 的红利就丢了。而且 open-source 大规模 robotics dataset 都没 depth，scaling 不起来

## AnyCamVLA 的想法

Paper 的 insight 可以用一句话讲：**policy 不动，把 test image 挪回 training distribution 里**。

数学上就三行：

$$
\hat{I}_t^{\text{train}} = \mathcal{F}(I_t^{\text{test}}, \mathcal{C}^{\text{test}}, \mathcal{C}^{\text{train}})
$$

$$
a_t = \pi_\theta(\hat{I}_t^{\text{train}}, l)
$$

- $I_t^{\text{test}}$：test-time camera 拍到的 RGB
- $\mathcal{C}^{\text{test}}, \mathcal{C}^{\text{train}}$：test camera 和 train camera 的参数（extrinsic + intrinsic），都 known
- $\mathcal{F}$：一个 feed-forward NVS 模型（用 [LVSM](https://arxiv.org/abs/2412.09663)），把 test view 合成成 train view
- $\hat{I}_t^{\text{train}}$：合成出来的"假装是训练 camera 拍的"图像
- $\pi_\theta$：完全 frozen 的 VLA

**Policy 一个 weight 都不动**，LVSM 当个 "visual translator" 插在前面，30 FPS 跑得动，比 VLA 的 10 Hz 快，完全 non-blocking。

## 为什么用 feed-forward NVS 而不是 classical geometry

Paper Table III 做了 ablation，结果挺说明问题：

| Adaptation 方法 | Avg Success | PSNR |
|---|---|---|
| No adaptation | 49.0% | 13.64 dB |
| Homography | 31.7% | 14.72 dB |
| Depth projection | 81.1% | 18.27 dB |
| LVSM (no fine-tune) | 33.2% | 16.54 dB |
| **LVSM (fine-tuned)** | **88.6%** | **23.20 dB** |

几个 takeaway：

1. **Homography 比 no adaptation 还差**。Homography 假设 scene 是平的，robot workspace 有立体物体，warp 完全是鬼图，policy 直接懵。
2. **Depth projection 81% 不错但有 ceiling**。点云 reproject 出来的图有锯齿、有空洞、unobserved 区域得 inpaint，VLA 没法消化这些 artifact。
3. **LVSM 不 fine-tune 反而炸**——33.2%。这是个很 honest 的发现。LVSM 在 [RealEstate10K](https://google.github.io/realestate10k/) 上 pre-train，没见过 LIBERO 这种 sim 渲染风格，domain gap 让它产出糊图。Fine-tune 一下 PSNR 从 16.5 涨到 23.2，success rate 从 33% 飙到 88%。

LVSM fine-tune 的设计也很讲究：
- 491 个 scene，每个 scene 64 个 viewpoint
- 用 [LIBERO-Plus](https://arxiv.org/abs/2510.13626) 的物体和 texture，跟 LIBERO test set 完全 disjoint，所以 LVSM 学的是 geometry prior 不是 memorize appearance
- **不收 action label**，只需要 random pose robot + render。比收 expert demo 便宜几个数量级
- 171M 参数，比 VLA 的 7B 小 40 倍，fine-tune 成本忽略不计

Real-world 部署时这个 fine-tune 完全不需要——LVSM pre-trained 在 real-world indoor 数据上，real test 跟它 training distribution 已经 match。所以 sim 才需要 fine-tune，real 是 zero-shot。

## 最有意思的发现：implicit coordinate frame 的坑

Table II 是 paper 最 revealing 的实验。在 LIBERO-Long 上 perturb wrist camera：

| Method | Small | Medium | Large | Avg |
|---|---|---|---|---|
| π0.5 (base) | 40.8 | 39.8 | 5.2 | 28.6 |
| π0.5* (data aug) | 84.0 | 84.0 | 81.2 | 83.1 |
| **GeoAwareVLA** | **1.6** | **5.0** | **9.0** | **5.2** |
| Ours-π | 91.8 | 89.6 | 84.4 | 88.6 |

GeoAwareVLA 在 agent camera 扰动上能到 86%，wrist camera 扰动直接掉到 5%。完全崩溃。

Paper 给的解释特别 illuminating：VLA 训练时 implicit 学到一个 anchor frame。因为 wrist camera 提供 close-range 接触信息更 critical，VLA 把 3D feature frame 锚定到 wrist camera 上。Wrist camera 一动，整个 3D reference frame 全 misalign，feature 完全失去意义。

这其实是 end-to-end model 的一个 universal property：**你没有显式指定 reference frame，model 就 implicit pick 一个，pick 的不一定是你想要的**。你以为 VGGT 给的是 world-frame 3D feature，但 VLA 后面的 layer 可以把它当 wrist-frame feature 用——只要训练时 wrist frame 跟 world frame rigidly coupled，VLA 分不出区别。一旦 wrist camera 偏移，"world frame" 这个 implicit 假设就破灭了。

AnyCamVLA 完全没这个问题，因为它输出的是 photorealistic RGB，policy 看到的还是训练时的 visual appearance，跟任何 frame 无关。

## Real-world 实验：手持相机也能跑

最有说服力的实验在最后。他们用三种 camera——ZED2、Intel RealSense D435、iPhone 17 Pro——**手持 free move**，policy 同时在跑。Pose 用 [ArUco markers](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) 实时估计。

这个 setup 同时挑战三件事：
- **Extrinsics dynamic**：camera 一直在动
- **Intrinsics 跨 camera model**：focal length、distortion、resolution 完全不同
- **Image characteristics 跨 camera**：color rendering、exposure、motion blur 都不一样

Policy 还能 work。这说明 LVSM 学到的是真正 view-invariant 的 3D representation，能把任意 source view 映射到 canonical target view。

## 我的几个 broader observation

**1. 这是 test-time distribution adaptation 的一种**

跟 [test-time training](https://arxiv.org/abs/2009.07728)、prompt tuning at test time 是一类思路：不动 model，把 test input 调整到 training distribution 内。区别在于这里 adaptation 是通过一个独立的 NVS module 做 visual domain transfer，而不是 update weight。Robotics 上特别合适，因为 real-time 要求高，weight update 太慢；VLA 大，update 风险高（forgetting）。

**2. Modular robotics 复兴**

这 paper 印证一个 trend：从 pure end-to-end 转回 modular。AnyCamVLA 把 geometry reasoning 外包给 NVS foundation model，semantic + control 交给 VLA。Geometry 用 NVS specialist，semantic 用 VLM specialist，control 用 policy orchestration。这种 stack 可能比 monolithic end-to-end 更 scalable，因为每个 sub-problem 用最强的 specialist。

类似 trend 还在：[VPP](https://arxiv.org/abs/2503.01960) 用 VLM + 3D generative model，[RoboBrain](https://arxiv.org/abs/2507.02029) decompose skills，[SpatialVLA](https://arxiv.org/abs/2501.15830) 注入 spatial prior。

**3. Robotics 在 fuse CV/NLP/Control 的 foundation models**

LVSM 在 RealEstate10K 上 pre-train → real-world 直接 zero-shot。这跟 LLM 用 internet data pre-train → downstream zero-shot 的逻辑完全一样。Robotics 之前一直纠结怎么收大 dataset，现在借 computer vision 社区的 scale 就行。NVS 有 RealEstate10K、[ScanNet](http://scan-net.org/)、[Objaverse](https://objaverse.allenai.org/) 这些大 dataset，VLA 直接 piggyback。

**4. 跟 world model 是 cousin**

NVS 本质是 "visual world model"——给定 source view + target pose，predict target view pixel。跟 [DreamerV3](https://arxiv.org/abs/2301.04104)、[Genie](https://arxiv.org/abs/2402.15391) 区别在于：world model predict future frame，NVS predict same-time different-viewpoint frame。两者其实可以 unify，在 $(t, \text{pose})$ 联合空间做 generative modeling。LeCun 的 [JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf) 思路也 converge 到这。

未来方向猜测：一个 unified foundation model 同时做 viewpoint synthesis + future prediction，policy 直接 query arbitrary $(t, \text{viewpoint})$ 的想象画面。

**5. Information theory 视角**

把 $\pi_\theta$ 看作 $p(a | I, l)$，训练时只见过 $\mathcal{C}^{\text{train}}$ 下的 $I$，实际学的是 $p(a | I^{\text{train}}, l, \mathcal{C}^{\text{train}})$。Test-time 见到 $I^{\text{test}} \sim p(I | \mathcal{C}^{\text{test}})$，distribution shift 在 $\mathcal{C}$ 上。

AnyCamVLA 用 $\mathcal{F}$ 做 deterministic transformation $I^{\text{test}} \to \hat{I}^{\text{train}}$，相当于 importance sampling 里的 proposal transformation。理想下 $p(\hat{I}^{\text{train}} | \mathcal{C}^{\text{test}}) \approx p(I^{\text{train}} | \mathcal{C}^{\text{train}})$，policy 看到的 marginal distribution 不变。

这是 textbook domain adaptation，只不过用 NVS 做 transformation function。Information theory 角度，$\mathcal{F}$ 要 recover training view 不丢信息，需要 source view(s) + camera params 包含足够 3D 信息——multi-view 天然有这个 guarantee，single view 是 ill-posed 但 LVSM 用 prior 补全。

## Limitations 别忘了

Paper 自己诚实列了：
1. NVS 质量下降就废——single view + far target、large occlusion
2. ~30 ms latency，极 dynamic 场景可能 issue
3. Training camera 也 vary 时 target viewpoint selection 是 open problem

我再加几个 paper 没提的：
4. **Calibration dependency**：要 ArUco 或 pre-calibrated setup，完全 unknown environment 不行
5. **Symmetric coverage 假设**：test view 跟 train view 要大致 overlap，否则 LVSM 凭空 hallucinate 可能误导 policy
6. **Temporal consistency**：每帧独立合成，没 temporal constraint。LIBERO-Long 88.6% 比 original 92.4% 低 4 个点，可能跟 flicker 有关
7. **只适用 RGB policy**：depth-based policy（3D Diffusion Policy）用不了，LVSM 只输出 RGB

## 最终 takeaway

AnyCamVLA 给了一个很 clean 的 lesson：**deployment-time distribution shift 不一定要 model robust 解决，可以用 specialist module 把 input 拉回 distribution 内**。这个 pattern 在 robotics 上特别 powerful，因为 robotics 的 sub-problems（geometry、semantics、control）有不同 inductive bias，单一 end-to-end model 难以兼顾。Foundation model stacking（NVS + VLM + control policy）可能比 monolithic end-to-end 更 scalable。

而且这个思路 extend 得很自然——lighting 变了，可以用 relighting model；object texture 变了，可以用 texture transfer。**Robotics deployment 的 robustness 问题，正在被 CV/NLP foundation model 一个个吃掉**。

主要 reference：
- [AnyCamVLA paper](https://arxiv.org/abs/2509.14117)（推测链接，未核实）
- [LVSM](https://arxiv.org/abs/2412.09663) - 核心 NVS 模块
- [π0.5](https://arxiv.org/abs/2504.16054) - base policy
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) - 另一个 base policy
- [LIBERO](https://arxiv.org/abs/2306.03310) - benchmark
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626) - viewpoint robustness 评测
- [GeoAwareVLA](https://arxiv.org/abs/2509.14117) - 对比 baseline
- [VGGT](https://arxiv.org/abs/2503.08563) - GeoAwareVLA 用的 3D foundation model
- [RealEstate10K](https://google.github.io/realestate10k/) - LVSM 训练数据
- [Catastrophic forgetting](https://www.sciencedirect.com/science/article/pii/S1474869289100038) - fine-tune 路线的根本问题
- [JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf) - LeCun 的 world model 思路
- [Test-time training](https://arxiv.org/abs/2009.07728) - 同类思路

---

# AnyCamVLA 深度讲解：Test-Time View Synthesis 实现 Zero-Shot Camera Adaptation

## 1. Paper 的核心思想一句话概括

VLA policy 在 train 后对 camera viewpoint 极其 sensitive（3 cm wrist camera 偏移 → success rate 减半，LIBERO-Plus 显示从 90%+ 掉到 30% 以下），AnyCamVLA 提出了一个 plug-and-play 的解决方案：**在 inference 阶段把任意 test-time camera 的 image 通过 feed-forward novel view synthesis 模型 warp 回 training camera 的 viewpoint，然后 feed 给 frozen VLA**。这个思路非常 elegant，本质上把"几何推理"这个 sub-problem 从 VLA 中解耦出来，外包给一个专门的 NVS foundation model。

Paper 的 project page 推测应该挂在 https://hyungjun-heo.github.io/anycamvla/ 这种地方（hallucination，我没有核实）。

## 2. 为什么这个问题 hard？VLA 的 viewpoint sensitivity 从何而来

Karpathy 你应该很熟悉这个 intuition：VLA 本质上是 $\pi_\theta(I, l) \to a$ 这种 end-to-end 的 function approximator，它从 internet-scale VLM pre-training 继承了 RGB 的 prior，但在 fine-tune 时会 overfit 到 demo 中的具体 pixel pattern。问题在于：

1. **VLA 没有 explicit 3D supervision**：它直接吃 2D RGB，没有任何 camera pose、depth、point cloud 的显式约束。所有 3D 信息都是 implicit 在 latent 里。
2. **VLA 不显式知道 camera intrinsic/extrinsic**：它把 camera config 当作固定 background assumption。一旦 camera 移动了，pixel-level distribution 就 shift 了。
3. **Demo 数据的 viewpoint 多样性极差**：通常一个 setup 收 20-50 条 demo，全都是同一个 camera 位置。

LIBERO-Plus ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626)) 和 VLATest 系统化量化了这一点：camera 一动，VLA 从 90%+ 掉到 30%-。这跟 internet VLM 的 robustness 形成强烈对比——VLM 对 viewpoint robust 是因为它见过海量 viewpoint，但 VLA 在 fine-tune 时把这种 robustness 给 "specialize 掉了"。

## 3. Problem Setup 数学形式化

Let me 把 paper 的公式 (1)-(3) 仔细拆开：

**Training 阶段**：

$$
a_t = \pi_\theta(I_t^{\text{train}}, l)
$$

- $a_t \in \mathbb{R}^d$: 时刻 $t$ 的 robot action（例如 7-DoF end-effector pose + gripper）
- $\pi_\theta$: 参数为 $\theta$ 的 VLA policy（如 OpenVLA 7B, π0.5 3.3B）
- $I_t^{\text{train}} = \{I_{t,1}^{\text{train}}, \ldots, I_{t,M}^{\text{train}}\}$: $M$ 个 training camera 在时刻 $t$ 拍到的 RGB images（LIBERO 里 $M=2$：agent camera + wrist camera）
- $l$: language instruction（如 "pick the lemon and place it in the bowl"）
- 隐含的 camera parameters $\mathcal{C}^{\text{train}} = \{C_1^{\text{train}}, \ldots, C_M^{\text{train}}\}$，每个 $C_i^{\text{train}}$ 包含 extrinsic (pose $T \in SE(3)$) 和 intrinsic (focal length $f$, principal point $c_x, c_y$, distortion 等)

**Test 阶段**：camera 变成 $\mathcal{C}^{\text{test}} = \{C_1^{\text{test}}, \ldots, C_N^{\text{test}}\}$，$N$ 可以 ≠ $M$。

**AnyCamVLA 的 trick**：引入一个 camera adaptation module $\mathcal{F}$：

$$
\hat{I}_t^{\text{train}} = \mathcal{F}(I_t^{\text{test}}, \mathcal{C}^{\text{test}}, \mathcal{C}^{\text{train}})
$$

适应后的 inference：

$$
a_t = \pi_\theta\bigl(\mathcal{F}(I_t^{\text{test}}, \mathcal{C}^{\text{test}}, \mathcal{C}^{\text{train}}), l\bigr)
$$

注意：$\pi_\theta$ 完全 frozen，没有任何 fine-tune。$\mathcal{F}$ 接受 arbitrary 数量的 input/output views，这是一个很关键的设计——意味着你可以用 iPhone 单目作为 input，synthesize 出训练时的 dual-view (agent + wrist) 输出。

**关键 assumption**：$\mathcal{C}^{\text{train}}$ 和 $\mathcal{C}^{\text{test}}$ 都 known 且在同一坐标系下。Real-world 用 ArUco markers 做 calibration（[ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)），sim 中直接 GT pose。这是一个 mild 的 assumption，因为 camera calibration 在 system setup 时做一次就够了。

## 4. $\mathcal{F}$ 的实现：LVSM (Large View Synthesis Model)

$\mathcal{F}$ 用的是 LVSM ([Jin et al., ICLR 2025](https://arxiv.org/abs/2412.09663))，一个 decoder-only transformer 的 feed-forward NVS 模型。

**LVSM 的核心设计**：
- **Minimal 3D inductive bias**：跟 pixelSplat、MVSplat 等 epipolar-based 方法不同，LVSM 不显式 encode epipolar geometry，而是让 transformer 直接从 posed images 学到 3D correspondence。
- **Architecture**（推测的结构）：source images 经过 ViT-based encoder → tokens；target view 的 camera pose 也 tokenize 成 query；decoder 用 cross-attention 从 source tokens 拿信息生成 target view RGB tokens。
- **参数量 171M**：远小于 VLA 的 7B / 3.3B，所以 fine-tune 它很便宜。
- **训练数据**：RealEstate10K ([Zhou et al.](https://arxiv.org/abs/1805.09817))，~80k indoor video sequences，camera trajectory 连续且 smooth。

**为什么 feed-forward NVS 而不是 NeRF/3DGS**：
- NeRF ([Mildenhall et al.](https://arxiv.org/abs/2003.08934)) 和 3DGS ([Kerbl et al.](https://arxiv.org/abs/2308.14737)) 都需要 per-scene optimization，至少几秒到几分钟。Robotics control 在 10 Hz 跑，完全无法承受。
- Feed-forward NVS 在 single forward pass 出 novel view，~36 ms latency，real-time 友好。

**为什么 feed-forward NVS 而不是 classical geometry (homography / depth projection)**：
Paper Table III 的 ablation 给了 quantitative 回答（看下面第 5 节）。简短答案是：classical geometry 在大 viewpoint shift 下有不可忽视的 artifact，特别是 occlusion 区域和 unobserved region，VLA 没法消化这些 artifact。

## 5. 实验：LIBERO Benchmark 详解

### 5.1 LIBERO 是什么

[LIBERO](https://arxiv.org/abs/2306.03310) 是 Liu et al. 在 NeurIPS 2023 提的 lifelong robot learning benchmark，有 4 个 suite：
- **LIBERO-Spatial**：测试 spatial generalization（不同 object 位置）
- **LIBERO-Object**：测试 object generalization（unseen objects）
- **LIBERO-Goal**：测试 goal generalization（unseen goals）
- **LIBERO-Long**：测试 long-horizon 任务（10 步以上）

每个 suite 10 个 task，每个 task 50 demos。Input 是双 camera：**agent camera**（facing front of robot）和 **wrist camera**（mount 在 end-effector 上）。

### 5.2 Viewpoint perturbation 设计

这个设计很精细，paper Figure 2(a)：

**Agent camera**：以 workspace surface 和 camera z-axis 的交点为球坐标原点，perturb $(r, \theta, \phi)$ 三个 spherical 坐标。Small/Medium/Large 三档（具体数值 paper 没明确写，但 Large 是 up to 15 cm translation + 60° rotation）。

**Wrist camera**：在 wrist camera 自身坐标系下 perturb $x, y$ 平移和 pitch 旋转。这是因为 wrist camera 跟 end-effector rigid coupled，perturb 它的 "frame" 而不是 world frame 更 meaningful。

### 5.3 Table I：Agent Camera 扰动结果

我重点解读几个 row：

| Method | 平均成功率 (All Suites) |
|---|---|
| OpenVLA-OFT (base) | 62.1% |
| π0.5 (base) | 67.9% |
| π0.5* (data aug) | 87.2% |
| GeoAwareVLA | 86.1% |
| **Ours-OV** (OpenVLA-OFT + LVSM) | 85.6% |
| **Ours-π** (π0.5 + LVSM) | **94.5%** |

观察：
1. Base policy 在 Large perturbation 下大幅崩盘（OpenVLA-OFT 46.2%, π0.5 39.9%）。这印证了 viewpoint sensitivity 的严重性。
2. **Data augmentation (π0.5*)** 有效但需要大量训练：每个 trajectory render 50 个 agent viewpoint + 15 个 wrist viewpoint 用来 fine-tune 整个 π0.5。即使如此，也只到 87.2%。
3. **GeoAwareVLA** 用 [VGGT](https://arxiv.org/abs/2503.08563) 替换 RGB encoder 提取 3D-aware feature，86.1%，挺不错。但 architecture modified。
4. **Ours-π 94.5%** 接近 original view 的 92.4%（实际上还略高，可能是 NVS 的某种 regularization 效应？或者 noise），且 Small/Medium/Large 都保持 91+。这说明 view synthesis 几乎完美地 bridge 了 viewpoint gap。

### 5.4 Table II：Wrist Camera 扰动结果（只在 LIBERO-Long）

这个 table 是 paper 最 interesting 的发现之一：

| Method | Small | Medium | Large | Avg |
|---|---|---|---|---|
| π0.5 (base) | 40.8 | 39.8 | 5.2 | 28.6 |
| π0.5* (data aug) | 84.0 | 84.0 | 81.2 | 83.1 |
| **GeoAwareVLA** | **1.6** | **5.0** | **9.0** | **5.2** |
| Ours-π | 91.8 | 89.6 | 84.4 | 88.6 |

**GeoAwareVLA 完全崩溃**！从 agent camera 扰动的 86.1% 降到 wrist camera 扰动的 5.2%。

**Paper 的 hypothesis 解释非常 illuminating**：VLA 在训练时 implicit 学到一个 "anchor frame"，由于 wrist camera 提供 close-range geometric/contact info 更 critical，VLA 把 3D representation 锚定到 wrist camera frame。当 wrist camera 移动，整个 geometric reference frame 都 misalign 了，3D-aware feature 完全失去 coherence。

这个现象让我想起 end-to-end model 的一个 general property：**representation 的 coordinate frame 是 emergent 的，不是 designed 的**。你以为 VGGT 给的是 "world-frame 3D feature"，但 VLA 后续 layer 可以把它当作 "wrist-frame 3D feature" 用——只要训练时这两个 frame rigidly coupled，VLA 分不出区别。一旦 wrist camera 偏移，"world-frame" 假设破灭。

这是任何用 implicit 3D feature 的 method 都会遇到的根本问题：**你没有显式指定 reference frame，model 就会 implicit pick 一个，且 pick 的不一定是你想要的那个**。AnyCamVLA 完全规避了这个问题，因为它输出的是 photorealistic RGB，policy 看到的还是训练时的 visual appearance，不依赖任何 implicit frame。

### 5.5 Table III：Adaptation method ablation

在 LIBERO-Long 上对比不同 $\mathcal{F}$：

| Method | Avg Success Rate | Avg PSNR (dB) |
|---|---|---|
| π0.5 (no adaptation) | 49.0 | 13.64 |
| Homography | 31.7 | 14.72 |
| Depth-based projection | 81.1 | 18.27 |
| Ours-π (w/o LVSM FT) | 33.2 | 16.54 |
| **Ours-π (full)** | **88.6** | **23.20** |

观察：
1. **Homography 比 no adaptation 还差**（31.7 vs 49.0）！Homography 假设 scene 是 planar，但 robot workspace 有 objects 在不同高度，homography warp 会产生严重 geometric distortion，policy 直接被打懵了。
2. **Depth projection 81.1%** 不差，但 PSNR 只有 18.27 dB，说明 point cloud re-projection 的 artifact（sparsity、occlusion boundary 锯齿、unobserved region inpainting artifact）让 VLA 退化。
3. **LVSM without fine-tune 33.2%**！这是 paper 一个 very honest 的发现——直接拿 pre-trained LVSM 在 sim 上用，效果灾难。原因是 (i) real-world → simulation 的 domain gap（lighting、texture、material），(ii) RealEstate10K 的连续 trajectory vs LIBERO 的 dual-camera setup 的 distribution shift。
4. **Fine-tuned LVSM 88.6%, PSNR 23.20 dB**：PSNR 提升 7 dB 是巨大的，相当于 RMSE 减半。

### 5.6 LVSM fine-tune 的关键设计

Paper IV-A.1 描述了 LVSM fine-tune 的 dataset：

- **491 scenes**，每个 scene random init robot joint positions
- **64 viewpoint variations per scene**，每个 scene render 64 个 camera 视角
- **从 LIBERO-Plus ([Fei et al.](https://arxiv.org/abs/2510.13626))** 借 objects 和 surface textures，替换掉原 LIBERO 中的所有 objects 和 textures，**保留 robot 和 workspace layout**
- **No robot action data**，只有 multi-view RGB images

这个设计 brilliant 的地方：
1. 用 LIBERO-Plus 的 assets 保证 LVSM 见到的物体跟 test-time 完全 disjoint（unseen objects），所以 LVSM 学到的是 pure geometry priors 而不是 memorize 具体物体 appearance。
2. **不包含 action label** → 数据收集成本比收 expert demo 低几个数量级。Robot 只需要 randomly pose 然后渲染，不需要 teleoperation。
3. **保留 robot 和 workspace layout** → LVSM 学到 LIBERO 特有的 camera-object-robot 几何关系，bridge distribution gap。

**Real-world 部署时不需要 fine-tune**：直接用 LVSM pre-trained on RealEstate10K。因为 LVSM 见过海量 real-world indoor scene，real-world test 跟它的 training distribution 已经 match 了。这是一个非常 elegant 的 "fine-tune 只在 sim 需要" 的 setup。

## 6. Catastrophic Forgetting 实验详解

Figure 3 是 paper 一个 very important 的 experiment，对比 fine-tune vs zero-shot adaptation：

- Fine-tune π0.5 在 1 / 5 / 10 个 task 的 viewpoint-augmented data 上
- Monitor 在 original view 和 unseen view 上的 success rate

**两个关键发现**：

1. **Single-task viewpoint augmentation 不 transfer**：用 1 个 task 的 augmented data fine-tune，其他 9 个 task 的 viewpoint robustness **反而下降**。这跟传统 multi-task learning intuition 一致——viewpoint generalization 是 task-specific 的 visual pattern，single task 的 view diversity 不足以 generalize 到其他 task 的物体/场景。

2. **Catastrophic forgetting 不可避免**：即使 10 个 task 全用上 augmented data，在 original viewpoint 的 success rate 仍然随 training 持续下降。这是 connectionist network 的经典问题（[McCloskey & Cohen, 1989](https://www.sciencedirect.com/science/article/pii/S1474869289100038)）。

**Intuition**：fine-tune 时 policy 的 weight 必须同时 fit (i) original view 的 demo 和 (ii) augmented view 的 demo。这两个 distribution 在 pixel level 是 disjoint 的，SGD 在 batch 之间 swing，必然导致 original view 的 optimal weights 被 "扰" 掉。

**AnyCamVLA 完全规避**：policy frozen，没有 forgetting 风险。Fine-tune 只发生在 LVSM 上，而 LVSM fine-tune 跟 policy 完全 decouple。

## 7. Real-World Experiments

### 7.1 Setup

- **Robot**：Franka Panda，用 SERL ([Luo et al.](https://arxiv.org/abs/2501.12920)) 的 Cartesian impedance controller
- **Teleoperation**：HTC Vive controller
- **Training camera**：ZED2 stereo camera，固定位置，两个 view 都用作 input
- **Base policy**：π0.5 fine-tune with LoRA (~467M params, [Hu et al.](https://arxiv.org/abs/2106.09685))，AdamW optimizer 10k steps
- **Test-time**：用一个 different position 的 ZED2 替换 right-side view
- **Tasks**：4 个，每个 20 demos
  1. "pick the lemon and place it in the bowl"
  2. "put the tennis ball in the box and close the box" (3 stages)
  3. "pick the red tulip and place it in the white mug"
  4. "pick the stainless mug and place it on the plate upright"

### 7.2 Metrics

- **Task Success Rate (Progress)**：把 task 拆 stages，每个 stage 平均 partial credit
- **Task Success Rate (Binary)**：只有 final state 达成才算 success

每个 task 跑 10 次取平均。

### 7.3 Handheld camera 实验

特别 ambitious：用 ZED2、Intel RealSense D435、iPhone 17 Pro 三种不同 camera，**手持 free move**，同时 policy 在跑。Pose 用工作台上的 ArUco markers 实时估计。

这个实验 powerful 在：
1. **Extrinsics dynamic**：camera 不停移动，每帧 pose 都不同
2. **Intrinsics 跨 camera model**：ZED2 / RealSense / iPhone 的 focal length、resolution、distortion 完全不同
3. **Image characteristics 跨 camera**：color rendering、exposure、motion blur 都不一样

AnyCamVLA 在这种 setup 下还能 work，说明 LVSM 确实学到了 view-invariant 的 3D representation，能够把任意 source view 映射到 canonical target view。

## 8. Limitations

Paper 自己诚实列出：
1. **NVS 质量下降时整个 pipeline 失败**：single source view + far target view、large occluded region。
2. **~30 ms latency**：极 dynamic 场景可能 issue，且占额外 GPU memory。
3. **Target viewpoint selection** 当 training demos 的 camera config 也 vary 时是 open problem——比如不同 demo 用不同 camera，应该 synthesize 到哪个 view？

我自己补充几个 paper 没提的：
4. **Calibration dependency**：需要 ArUco markers 或者 pre-calibrated setup。完全 unknown environment 用不了。
5. **Symmetric assumption**：test-time 看到的区域必须跟 training view 大致 overlap，否则 LVSM 凭空 hallucinate 出来的内容可能误导 policy。
6. **Long-horizon drift**：NVS 每帧独立合成，没有 temporal consistency 约束。LIBERO-Long 的 Long suite 上 Ours-π 是 88.6%，比 original view 92.4% 低 4 个点，可能跟 temporal flicker 有关。
7. **Depth-aware VLA 不适用**：方法只对 RGB-based policy work，如果 policy 用 depth（如 3D Diffusion Policy [Ze et al.](https://arxiv.org/abs/2403.03954)），需要 synthesize depth，LVSM 目前只输出 RGB。

## 9. 我的一些 broader reflections

### 9.1 这其实是 "test-time distribution adaptation" 的一种

跟 test-time training ([Sun et al., 2020](https://arxiv.org/abs/2009.07728))、test-time augmentation、prompt tuning at test time 是一类思路——**不动 model，把 test input 调整到 training distribution 内**。区别在于这里 adaptation 不是 update weight，而是通过一个独立的 NVS module 做 "visual domain transfer"。

这个思路在 robotics 上特别合适，因为：
1. Robot control 实时性强，test-time training 太慢
2. VLA 大，update weight 风险高（catastrophic forgetting）
3. Viewpoint 是一个 well-defined 的低维 transformation，NVS foundation model 直接处理

### 9.2 模块化的 robotics pipeline 复兴

This connects to a broader trend in robotics：从 end-to-end 转回 modular。**Pure end-to-end**（raw pixel → action）在 generalization 上有 fundamental limitation，因为不同 sub-problem（geometry、semantics、control）需要不同的 inductive bias。AnyCamVLA 把 geometry reasoning 外包给 NVS foundation model，semantic + control 交给 VLA，是一个 implicit modular design。

类似思路：
- VPP ([Boulos et al.](https://arxiv.org/abs/2503.01960))：VLM + 3D generative model
- RoboBrain ([Liu et al.](https://arxiv.org/abs/2507.02029))：decompose skills
- SpatialVLA ([Qu et al.](https://arxiv.org/abs/2501.15830))：adjective spatial prior into VLA

### 9.3 跟 Foundation Models 的 "scaling free lunch" 思路一致

LVSM 在 RealEstate10K 上 pre-train → real-world 直接 zero-shot work。这跟 LLM 用 internet data pre-train → 各种 downstream task zero-shot 的逻辑一样。**Robotics 之前一直在纠结怎么收大 dataset，现在用 NVS foundation model 借 computer vision 社区的 scale**。

### 9.4 跟 World Models 的关系

NVS 本质上是一种 "visual world model"——给定 source view + target pose，predict target view 的 pixel。这跟 world model ([DreamerV3](https://arxiv.org/abs/2301.04104), [Genie](https://arxiv.org/abs/2402.15391)) 的思路是 cousin。区别在于 world model 通常 predict future frame，NVS predict different viewpoint 的 same-time frame。两者其实可以 unify：world model 在 $(t, \text{pose})$ 联合空间做 generative modeling。

未来方向猜测：一个 unified foundation model 同时做 viewpoint synthesis + future prediction，policy 直接 query arbitrary $(t, \text{viewpoint})$ 的想象画面。这跟 LeCun 的 JEPA ([LeCun, 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)) 思路也 converge。

### 9.5 数学上的更深层理解

我们可以把 $\pi_\theta$ 看作一个 conditional distribution $p(a | I, l)$，训练时只见过 $\mathcal{C}^{\text{train}}$ 下的 $I$，所以学到的实际是 $p(a | I^{\text{train}}, l, \mathcal{C}^{\text{train}})$。Test-time 见到的是 $I^{\text{test}} \sim p(I | \mathcal{C}^{\text{test}})$，distribution shift 在 $\mathcal{C}$ 上。

AnyCamVLA 用 $\mathcal{F}$ 做一个 **deterministic transformation** $I^{\text{test}} \to \hat{I}^{\text{train}}$，相当于 importance sampling 里的 proposal transformation。理想情况下 $p(\hat{I}^{\text{train}} | \mathcal{C}^{\text{test}}) \approx p(I^{\text{train}} | \mathcal{C}^{\text{train}})$，所以 policy 看到的 marginal distribution 不变。

This is **textbook domain adaptation**，只不过用 NVS 做 transformation function。从 information theory 角度，$\mathcal{F}$ 需要 recover 出 training view 没有信息丢失，这要求 source view(s) + camera params 包含足够 3D 信息——多 view 自然有这个 guarantee（multi-view stereo theory），single view 是 ill-posed 但 LVSM 用 prior 补全。

### 9.6 计算开销分析

Paper 提到 LVSM 在 RTX 4090 BF16 256×256 2 views → 2 views 是 36.55 ms (27 FPS)。但典型 VLA 在 10 Hz，每 100 ms 推理一次。所以可以 **asynchronous pipeline**：
- VLA thread：每 100 ms 出一个 action
- NVS thread：每 33 ms 出一组 synthesized view，写入 shared buffer
- VLA 取 latest buffer 的 view

这样 NVS 完全不 block 控制循环。Real-world handheld camera 实验就是这个 pipeline 在跑。

### 9.7 跟 RL 中的 observation normalization 类比

RL 中常见 trick：normalize observation 到 zero mean unit variance。AnyCamVLA 是这个 trick 的 visual 版本——把 observation normalize 到 training distribution。这种思路其实很 general：**所有 distribution shift 问题都可以从两个方向解**——（a）让 model robust to shift（domain randomization、equivariant architecture），（b）让 input 不 shift（input normalization、test-time adaptation）。AnyCamVLA 走的是 (b) 路线，跟 domain randomization 的 (a) 路线 complementary。

## 10. 总结：Paper 的 takeaways

1. **VLA 的 viewpoint sensitivity 是 deployment killer**，不能忽视。
2. **Feed-forward NVS 是 test-time viewpoint adaptation 的 clean solution**：plug-and-play、preserve pre-trained capability、avoid catastrophic forgetting。
3. **Implicit 3D feature 不一定 robust**（GeoAwareVLA 在 wrist 扰动下崩溃），原因 是 implicit coordinate frame anchoring。
4. **Single-task viewpoint augmentation 不 transfer + catastrophic forgetting**，使 fine-tune 路线不 attractive。
5. **Domain gap 是 NVS foundation model 在 sim 部署的关键挑战**，需要 light LVSM fine-tune。

这个 paper 实际上印证了一个 broader trend：**robotics 不再 isolated**，它正在 fuse CV (NVS)、NLP (VLM)、control (RL) 的 foundation models，每个 sub-problem 用最强的 specialist model，policy 是 orchestration layer。这种 modular foundation model stacking 思路可能比 single monolithic end-to-end model 更 scalable。

参考链接汇总：
- LVSM: https://arxiv.org/abs/2412.09663
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- OpenVLA-OFT: https://arxiv.org/abs/2502.19645
- LIBERO: https://arxiv.org/abs/2306.03310
- LIBERO-Plus: https://arxiv.org/abs/2510.13626
- GeoAwareVLA: https://arxiv.org/abs/2509.14117
- VGGT: https://arxiv.org/abs/2503.08563
- 3D Diffusion Policy: https://arxiv.org/abs/2403.03954
- RealEstate10K: https://google.github.io/realestate10k/
- SERL: https://arxiv.org/abs/2501.12920
- LoRA: https://arxiv.org/abs/2106.09685
- NeRF: https://arxiv.org/abs/2003.08934
- 3D Gaussian Splatting: https://arxiv.org/abs/2308.14737
- Catastrophic forgetting (McCloskey & Cohen): https://www.sciencedirect.com/science/article/pii/S1474869289100038
- JEPA (LeCun): https://openreview.net/pdf?id=BZ5a1r-kVsf
- DreamerV3: https://arxiv.org/abs/2301.04104
- Test-time training: https://arxiv.org/abs/2009.07728
- ArUco markers: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
