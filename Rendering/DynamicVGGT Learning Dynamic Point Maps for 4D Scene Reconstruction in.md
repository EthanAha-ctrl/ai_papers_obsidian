---
source_pdf: DynamicVGGT Learning Dynamic Point Maps for 4D Scene Reconstruction in.pdf
paper_sha256: 87b5886162281ee6711bd63e6ad471f55b5c80b0d11b44c0570fee2ac58f503a
processed_at: '2026-08-04T00:42:04-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，换个讲法，像在白板前聊。

---

## 这篇 paper 到底在干嘛

你有一个叫 VGGT 的 model，它吃几张图，吐一堆 3D geometry——point map、depth、camera pose、3DGS。它在静态场景上 work 得很好。

现在你想把它搬到自动驾驶上。问题是：**车在动，人也在动，世界不是静止的**。

最 naive 的办法——把 VGGT 在 video 上跑一遍，每帧独立重建，然后外面拼起来。这个不行，因为 VGGT 根本不知道"这辆车和上一帧那辆车是同一辆"，重建出来会抖、会漂、会不一致。

那怎么办？得让 model 在内部就理解 time。这就是这篇 paper 干的事。

---

## 之前的人怎么做的，为什么不行

StreamVGGT（同一作者团队的上一篇）试过一个办法：**把 temporal attention block 串在 VGGT 的 spatial attention block 后面**。听起来很自然对吧——先 spatial 再 temporal，串起来。

但这个 design 有个致命问题。VGGT 的 spatial block 是预训练好的，它学到了非常精细的 multi-view correspondence reasoning。你在它后面接一个新的 temporal block，这个新 block 一开始是随机初始化的，它会把 spatial block 的输出分布搞乱——**你等于在毁掉自己最值钱的东西**。

训练早期你会看到 loss 比 baseline 还高，要很久才能爬回来。而且你 fine-tune 真实数据的时候，这个不稳定会被放大。

StreamVGGT 在 Table 3 的 KITTI MVS 上 Abs Rel 是 **0.173**——比它的前身 VGGT (0.062) 还差 3 倍。这就是串联设计在 multi-view 大场景上崩了的证据。

---

## DynamicVGGT 的核心 idea：开一条旁路

最关键的 insight 就一句话：**别动 backbone，在旁边开一条 motion 通道**。

具体怎么做？引入一小组 learnable tokens，叫 motion tokens，paper 里没说具体多少个，但你可以想象成几十到上百个。这些 tokens 就像 "temporal scratchpad"，专门存跨帧的运动信息。

然后造一个新的 attention block，叫 MTA (Motion-aware Temporal Attention)。它的输入是什么？是 motion tokens + 从 backbone 那边拿过来的 patch features。它的 attention 是**只在 time 维度上**做的——同一个 pixel position，跨 τ 个 frame 做 self-attention。

这个 design 的妙处：

1. **Backbone 完全不变**，AA block 还是预训练的样子，几何 prior 一点没坏
2. **Motion tokens 是 learnable parameters**，它们会在训练中逐渐学会编码"运动"这个抽象
3. **梯度路径清晰**：temporal loss 直接回传到 MTA，再通过 skip connection 影响 backbone feature，不需要穿过整个串联栈
4. **职责分明**：AA 管 "这是什么形状"，MTA 管 "它怎么动"

这跟 LoRA 的精神很像——不侵入预训练好的 weight，加个旁路。也跟你之前讲过的 "residual stream" 那个直觉一致——transformer 内部其实是一个 residual highway，每个 block 是在往上面加东西。这里 MTA 就是新加的一条支流。

paper 在公式 5 里有个细节值得注意：第二层开始，输入是 $F_{v,t}^{p(l)} + F_{v,t}^{p(l-1)}$，把上一层 backbone feature 加回来。这是为了不让 MTA "跑偏"——它要持续从 backbone 拿 fresh feature，而不是自己一路演化成跟 backbone 无关的东西。

---

## 怎么让 motion tokens 真的学到运动

光有 architecture 还不够，得有 signal 告诉它"运动是什么"。paper 用了两个互补的 task。

### Task 1: Future Point Head (隐式)

让 model 从当前帧的 feature 直接预测**未来一帧**的 point map。输入是 $TA_{v,t}$，输出是 $\hat{P}_{v,t+\delta}$。

这个 task 的核心是：**要预测未来，你必须在当前 feature 里编码运动信息**。如果你不知道这辆车在往左开，你预测不出来下一帧它在哪。

loss 是 temporal consistency regularization（公式 11），它不是直接监督 point 位置，而是监督 displacement：

$$\mathcal{L}_{\text{temp}} = \frac{1}{|\mathcal{N}|} \sum_{i} \big\| \underbrace{(\mathbf{p}_{t+\delta} - \mathbf{p}_t)}_{\text{GT motion}} - \underbrace{(\hat{\mathbf{p}}_{t+\delta} - \hat{\mathbf{p}}_t)}_{\text{predicted motion}} \big\|_1$$

为什么用 displacement 而不是 absolute position？因为 absolute position 的 L1 loss 在大坐标尺度上，对小运动的梯度会非常小——一个 30m 远的车移动 0.1m，loss 几乎没变化。但 displacement 的尺度跟运动本身同尺度，梯度信号干净。

这个 task 是 **self-supervised** 的，因为它只需要 GT point map（重建的 supervision），不需要额外标注。

### Task 2: Dynamic 3DGS Head with Scene Flow (显式)

这个 task 在 stage 2 才启用。它做的是：把 point map 升级成一组 time-varying 3D Gaussians，每个 Gaussian 除了通常的 center/scale/rotation/color，还有一个 **velocity vector** $\nu_i \in \mathbb{R}^3$。

关键 design：**velocity 不是每个 Gaussian 独立预测的**，而是从 motion tokens decode 出一组 velocity bases $\nu_b$，然后每个 Gaussian 通过 attention 从 bases 加权组合出自己的 velocity。

这个 design 的 intuition 很有意思。想象一个十字路口：左边一辆车往北开，右边一辆车往东开，背景在往后退（因为 ego car 在前进）。这个场景里的 motion pattern 其实是 **low-rank 的**——所有点的运动可以由几个 basis motion（"车 A 的 rigid motion"、"车 B 的 rigid motion"、"ego motion"）线性组合表达。用 basis decomposition 比每个点独立预测 velocity 更高效，也更平滑。

Constant velocity 假设（公式 15）：
$$\mu_{i,t+\delta} = \mu_{i,t} + \delta \cdot \nu_{i,t}$$

这是说"在这个短 clip 里，每个 Gaussian 匀速运动"。在 δ ≤ 3 帧内对自动驾驶基本成立，但你也可以想象，长序列肯定 drift。

监督来自 scene flow（Waymo 有这个 GT）：
$$\mathcal{L}_{\text{flow}} = \text{MSE}(\mathbf{s}, \hat{\mathbf{s}})$$

这个 loss 直接告诉每个 Gaussian："你的 velocity 应该是多少"。这跟 $\mathcal{L}_{\text{temp}}$ 是互补的——后者在 point map level 约束，前者在 Gaussian primitive level 约束。

---

## 一个我特别欣赏的 trick：Depth Distillation

自动驾驶数据有个噩梦：**LiDAR 稀疏**。一辆车的 LiDAR 点可能就几十个，分布极不均匀。你直接拿这些稀疏点去监督 dense Gaussian depth，会让 Gaussian 优化崩掉——它会过拟合到有 LiDAR 点的地方，其他地方乱飘。paper 里 Fig.4 直观展示了这个问题，depth map 变得粗糙。

paper 的解法很巧：**用 stage-1 学好的 dense point map branch 当 teacher，去蒸馏 stage-2 的 Gaussian depth branch**。

$$\mathcal{L}_{\text{distill}} = \big\| D_{g} - \text{sg}(D^{\text{pm}}) \big\|_1$$

注意 `sg`——stop gradient。teacher 不被更新，只 student 学。这样 stage-1 的 dense geometric knowledge 就 transfer 到 stage-2 的 Gaussian 上，绕开了 LiDAR 稀疏的问题。

这个 trick 跟 DINO (https://arxiv.org/abs/2104.14294) 的 self-distillation 有精神上的相似——用一个稳定的 branch 教另一个 branch。也跟你之前讲过的 EMA teacher 在 BYOL/MoCo 里的角色类似。

但这里有个潜在的问题：**student 永远超不过 teacher 的上限**。如果 stage-1 在某些区域有 systematic bias，stage-2 会继承。paper 用了很小的 weight $\lambda_{\text{dist}} = 0.1$，加上 RGB + scene flow loss 一起作用，可能能拉回来一些，但长期看这是个 ceiling。

---

## 训练策略：从 synthetic 到 real 的 curriculum

- **Stage 1**：在 Virtual KITTI + MVS-Synth（合成数据）上训练。合成数据 dense、干净、scene flow 可靠。这里学 motion 的骨架——MTA + FPH。LR 非常小（$10^{-6}$），10 epochs
- **Stage 2**：在 Waymo（真实数据）上 fine-tune。启用 DGSHead。LR 大 50 倍（$5 \times 10^{-5}$），50 epochs

这个 schedule 的 intuition：stage 1 是在 "安全环境" 里学 motion 这个新能力，不用跟真实数据噪声搏斗。stage 2 才把 motion 能力适配到 real world，顺便学 appearance（Gaussian rendering）。

stage 2 LR 反而更高，是因为 stage 2 主要在 train 新增的 DGSHead 模块，那些参数是 fresh 的，需要更激进的更新。backbone 那边仍然 frozen 或缓慢更新。

---

## 实验数据怎么读

最值得看的是 Table 4 ablation：

| 配置 | KITTI Acc ↓ | Waymo Acc ↓ |
|---|---|---|
| Baseline (VGGT) | 1.489 | 4.635 |
| + TA & FPH | 0.927 | 4.330 |
| + DGSHead | 0.901 | 4.021 |

第一个跃迁（baseline → +TA&FPH）巨大——KITTI 从 1.489 到 0.927，砍了 38%。这说明**光有 motion token 旁路 + 隐式 future point 预测，就能拿到大部分收益**。DGSHead 是锦上添花，在 Waymo 上效果更明显（4.330 → 4.021）。

我自己的 interpretation：MTA + FPH 给 backbone 提供了 "时间一致性" 的 inductive bias，这本身就让单帧重建变好——因为现在每个 frame 的 feature 都被跨帧 context refine 过。而 DGSHead 主要在 explicit motion 监督（scene flow）上发力，对大尺度 dynamic object 多的 Waymo 帮助更大。

Table 3 里 StreamVGGT 在 KITTI MVS 上 Abs Rel = 0.173，比 VGGT (0.062) 差 3 倍——这是串联 temporal block 灾难性失败的直接证据。DynamicVGGT 在同样 setting 下是 0.051，比 VGGT 还好。这对比强烈支持了"旁路设计优于串联设计"这个核心 claim。

Table 2 的 4D 重建对比：DynamicVGGT 在 Waymo dynamic region PSNR 18.07，STORM 是 21.26。看起来输 3dB，但 **STORM 要 calibrated camera**，DynamicVGGT 只要 image。这是个重要的 trade-off——calibration 在真实自动驾驶里有时候不可靠，能去掉这个依赖是很有价值的。

---

## 一些我自己的联想

**1. Motion tokens 的数量是个超参，paper 没说**。我猜可能是几十到几百。这其实是个 low-rank 假设——M 个 basis 能表达多少种 motion pattern？在自动驾驶里 motion pattern 数量有限（几个 dynamic object + ego motion），所以 M 不需要太大。但如果你把这个 model 搬到室内场景（很多人、很多动作），M 可能要大很多，或者换成 MLP-based deformation field（像 DeformableGS 那样）。

**2. Constant velocity 假设的局限**。这跟 classic Kalman filter 的 constant velocity model 是一回事。在 short horizon 上 OK，long horizon 会 drift。一个自然的改进是引入 acceleration，或者让 velocity 本身 time-dependent $\nu_{i,t}$。但这就需要在更长 clip 上训练，数据量和优化难度都上来。

**3. MTA 的 RoPE temporal encoding**。RoPE 的好处是相对距离 encoding，理论上对 frame stride 泛化更好。paper 里训练和测试 stride 都是固定的，所以这个泛化能力没被验证。但如果未来要做 variable frame rate，RoPE 比 absolute positional encoding 更有优势。

**4. Velocity basis 跟 motion tokens 的关系**。MTA 里的 motion tokens 学的是 "temporal context"，DGSHead 里又用同样的 tokens decode velocity bases。这意味着 motion tokens 是个 **shared dynamic representation**——既参与 temporal attention，又参与 velocity decoding。这是个很 elegant 的 design，但也意味着 motion tokens 同时承担两个责任，可能有 capacity 瓶颈。

**5. 跟 NeRF 系 deformation model 的对比**。DeformableGS (https://arxiv.org/abs/2306.17838) 用一个 MLP 把每个 point 在 canonical space 的坐标 map 到 deformation offset。DynamicVGGT 用 basis motion + constant velocity。两种 parametrization 的 trade-off：MLP 灵活但慢，basis 快但表达力受限于 M。在自动驾驶这种 rigid motion 为主的场景，basis 更合理。

**6. 跟 diffusion / flow matching 的潜在结合**。现在 DynamicVGGT 是 deterministic 的——给定 clip，预测一个 motion。但真实驾驶里 motion 是 stochastic 的（行人可能左转也可能右转）。一个自然的扩展是用 flow matching (https://arxiv.org/abs/2512.06112) 或 diffusion 学 motion distribution，做 probabilistic 4D prediction。这对下游 motion planning 会更有用。

**7. 跟 VLA 的结合**。paper 提到 4D reconstruction 支持 closed-loop training。如果能把这个 4D representation 直接喂给一个 VLA policy (https://arxiv.org/abs/2512.11872)，让 policy 在 reconstructed 4D space 里做 planning，这会是个很自然的 end-to-end pipeline。当前 VLA 一般直接吃 image，但 4D representation 可能提供更稳定的 geometric grounding。

**8. MTA 可以换成 state-space model**。MTA 是 standard self-attention，复杂度 $O(\tau^2)$。如果 $\tau$ 变大（长序列），可以换成 Mamba (https://arxiv.org/abs/2312.00752) 这种 linear-complexity 的 SSM。Mamba 的 selective scan 对 temporal sequence 特别合适，可能在长 horizon 上比 attention 更好。

**9. Gaussian 的 semantic attribute**。现在 Gaussian 只有几何 + appearance + velocity。如果再加 semantic attribute（通过 SAM 或 open-vocabulary segmentation 监督），这个 4D representation 就能直接支持 query-based downstream task——"找到那辆红色的车并预测它未来 3 秒的轨迹"。这是把 4D reconstruction 跟 scene understanding 统一的一个方向。

**10.关于 frozen backbone 的长期代价**。DynamicVGGT frozen AA block 是为了保 prior，但这也意味着 backbone 不能适配 driving 数据的 distribution shift（比如 Waymo 的 camera 配置跟 VGGT 训练数据差很多）。Table 3 里 NYU-v2（室内）Abs Rel 从 VGGT 的 0.059 退化到 0.064，这就是 frozen 限制了 adaptation 的迹象。一个折中方案是 LoRA-style 微调——backbone 加 low-rank adapter，既保 prior 又允许小幅 adaptation。

---

## Reference Links

- VGGT (CVPR 2025): https://vgg-t.github.io/
- StreamVGGT: https://arxiv.org/abs/2507.11539
- Dynamic Point Maps: https://arxiv.org/abs/2503.16318
- DINO self-distillation: https://arxiv.org/abs/2104.14294
- LoRA: https://arxiv.org/abs/2106.09685
- DeformableGS: https://arxiv.org/abs/2306.17838
- 3DGS (SIGGRAPH 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- STORM: https://arxiv.org/abs/2501.00602
- DrivingForward (AAAI 2025): https://arxiv.org/abs/2408.01361
- MoVieS: https://arxiv.org/abs/2507.10065
- Anysplat: https://arxiv.org/abs/2505.23716
- DPT: https://arxiv.org/abs/2101.03979
- MoGe: https://arxiv.org/abs/2502.18403
- DINOv2: https://arxiv.org/abs/2304.07193
- Mamba: https://arxiv.org/abs/2312.00752
- Street Gaussians (ECCV 2024): https://arxiv.org/abs/2401.01339
- HUGS: https://arxiv.org/abs/2403.17816
- OmniRe: https://arxiv.org/abs/2408.16760
- WAM-Diff VLA: https://arxiv.org/abs/2512.11872
- WAM-Flow motion planning: https://arxiv.org/abs/2512.06112
- Waymo Open Dataset: https://waymo.com/open/
- KITTI: http://www.cvlibs.net/datasets/kitti/
- Virtual KITTI: https://virtual-kitti.woodward.win/
- DUSt3R: https://dust3r.europe.naverlabs.com/
- MASt3R (ECCV 2024): https://arxiv.org/abs/2406.09756
- MonST3R: https://arxiv.org/abs/2410.03825
- GS-LRM: https://arxiv.org/abs/2404.19138

---

# DynamicVGGT 深度解析

Andrej, 这篇 paper 我觉得非常对你的胃口——它在 VGGT 这种 feed-forward 3D foundation model 上做了一件非常微妙的事情：把静态的 point map 升级成 dynamic 的 4D 表示，但又不破坏原来学到的 geometric prior。下面我会一层一层拆开讲，重点放在 design intuition 上。

---

## 1. 问题动机：为什么 VGGT 不能直接搬到自动驾驶？

VGGT 是一个 feed-forward transformer，输入 multi-view images，输出 point map / depth / camera pose / 3DGS。它在静态场景上效果极好。但自动驾驶有三个根本性问题：

1. **Scene 是 dynamic 的**：moving vehicles, pedestrians, lighting changes
2. **数据是 sparse + noisy**：LiDAR point cloud 稀疏不均匀（见 paper 中 Fig.4，直接用 sparse LiDAR 监督会让 depth map 变粗糙）
3. **需要 temporal coherence**：单帧重建不够，必须 4D

StreamVGGT (ref [38], https://arxiv.org/abs/2507.11539) 已经尝试在 VGGT 上加 temporal attention，但它是把 AA block 和 temporal block 串联堆叠的，导致训练不稳定，早期性能退化。DynamicVGGT 的核心 insight 是：**temporal 信息应该走一条并行的 motion token 通道，而不是侵入 spatial attention**。

---

## 2. Dynamic Point Map (DPM) — 几何表示的统一

这是 paper 的核心 formulation。先看传统 DPM (来自 Dynamic Point Maps, https://arxiv.org/abs/2503.16318)：

$$P_{v,t}^{\text{ref}} = \mathcal{T}_{(v,t)\to\text{ref}}\big(\pi^{-1}(I_{v,t}; K_{v,t}, E_{v,t})\big)$$

变量含义：
- $v$: camera index $\in \{1, ..., N_v\}$
- $t$: frame index $\in \{1, ..., \tau\}$
- $\pi^{-1}$: inverse projection，从 2D image 反投影到 3D
- $I_{v,t}$: image at view $v$, time $t$
- $K_{v,t}, E_{v,t}$: intrinsics / extrinsics
- $\mathcal{T}_{(v,t)\to\text{ref}}$: 把当前帧坐标转到 shared reference frame

然后 motion 就是：
$$\Delta P_{v,t}^{\text{ref}} = P_{v,t+\delta}^{\text{ref}} - P_{v,t}^{\text{ref}}$$

这里 $\delta$ 是 temporal offset (paper 中 sampled from 1 to 3)。

**问题**：这需要外部指定 reference frame + 精确的 frame-to-reference transformation。在自动驾驶中 camera extrinsics 噪声大，这条路走不通。

DynamicVGGT 的 key idea 是——**让模型在 learned canonical frame 里直接预测**：

$$\hat{P}_{v,t}, \hat{P}_{v,t+\delta} = f_\theta(\{I_{v,t}\})\big|_{(v,t),(v,t+\delta)}$$

motion 隐式地通过 $\Delta\hat{P}_{v,t} = \hat{P}_{v,t+\delta} - \hat{P}_{v,t}$ 学习。

这个 formulation 看起来小，但意义很大：**它把"显式对齐"换成了"隐式一致性约束"**。模型在同一个 canonical space 里输出两帧的 point map，它们的差就是 motion。这就保留了 VGGT backbone 的几何 prior，同时给后续两个 task (FPH, DGSHead) 提供了统一的 coordinate space。

---

## 3. Motion-aware Temporal Attention (MTA) — 最核心的 architecture contribution

这部分是 paper 最有工程价值的地方。先看 paper 公式 5：

$$F_{m,v,t}^{(l)} = \begin{cases} \text{Concat}\big(M_{v,t}^{(l)}, F_{v,t}^{p(l)}\big), & l=1 \\ \text{Concat}\big(M_{v,t}^{(l)}, F_{v,t}^{p(l)} + F_{v,t}^{p(l-1)}\big), & l>1 \end{cases}$$

变量：
- $M_{v,t}^{(l)} \in \mathbb{R}^{M \times d}$: $M$ 个 learnable motion tokens，作为 learnable parameters 初始化
- $F_{v,t}^{p(l)}$: AA branch 第 $l$ 层输出的 patch tokens
- $F_{m,v,t}^{(l)}$: MTA 第 $l$ 层输入

直觉：MTA 是一个**并行的旁路**。AA block 处理 intra-frame spatial geometry (在 frozen 的 VGGT backbone 上)，MTA block 处理 inter-frame temporal dependencies。motion tokens 像是"temporal scratchpad"，专门存跨帧的运动信息。

第二层开始，输入加上 $F_{v,t}^{p(l-1)}$——这是个 skip connection 设计，让 MTA 能持续从 AA 那里拿 fresh spatial features。

公式 6 是 temporal self-attention：

$$A_{t,t'}^{(l)} = \text{Softmax}\left(\frac{Q_t^{\text{attn},(l)}(K_{t'}^{\text{attn},(l)})^\top}{\sqrt{d}} + B_{t,t'}^{\text{time}}\right)$$

变量：
- $t, t' \in \{1, ..., \tau\}$: 两个 frame index
- $Q_t, K_{t'}, V_{t'}$: query/key/value
- $B_{t,t'}^{\text{time}}$: **rotary position embedding (RoPE)** 作为 temporal positional bias

这里用 RoPE 而不是 absolute positional encoding 很聪明——它让 temporal attention 对 frame stride 有一定的 inductive bias 泛化能力（虽然 paper 里训练和测试 stride 都是固定的，但 RoPE 在相对距离上泛化更好）。

公式 7-8 是标准 attention + MLP + residual：

$$\tilde{F}_{m,v,t}^{(l)} = \sum_{t'=1}^{\tau} A_{t,t'}^{(l)} V_{t'}^{\text{attn},(l)}$$

$$F_{m,v,t}^{(l+1)} = \text{MLP}^{(l)}\big(\text{LayerNorm}(\tilde{F}_{m,v,t}^{(l)})\big) + F_{m,v,t}^{(l)}$$

最终输出记为 $TA_{v,t} = F_{m,v,t}^{(L)}$，$L=12$ 是 MTA 层数。整个 model 大概 1.4B 参数，其中约 800M 是可训练的（AA block frozen）。

**为什么这个设计比 StreamVGGT 好？**

直觉：VGGT 预训练的 AA block 学到了非常强的 geometric correspondence reasoning。如果像 StreamVGGT 那样把 temporal block 串联进去，会导致：
1. **训练不稳定**：新增的 temporal block 改变了 AA 的输入分布，破坏 prior
2. **梯度路径变长**：从 temporal loss 回到 spatial features 路径绕
3. **早期退化**：temporal block 没学好之前，整体性能比 baseline 还差

DynamicVGGT 用 motion tokens 作为"软通道"，AA block 还是按原样跑，MTA 只是旁路取 features。这样：
- AA 的 geometric prior 完全保留
- MTA 专注学 motion，职责清晰
- 训练可以 stage-wise，先 warm-up MTA + FPH，再加 DGSHead

---

## 4. Future Point Head (FPH) — 隐式 motion 学习

公式 10：

$$\hat{P}_{v,t+\delta}^{\text{fut}} = \text{DPT}_p(TA_{v,t})$$

DPT (Dense Prediction Transformer, https://arxiv.org/abs/2101.03979) head 从 $TA_{v,t}$ 预测**未来帧**的 point map。注意：这里输入是 $t$ 时刻的 feature，输出是 $t+\delta$ 的 point map——这逼迫模型在 $TA_{v,t}$ 里编码 motion 信息。

公式 11 是 temporal consistency regularization：

$$\mathcal{L}_{\text{temp}} = \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} \big\| \big(\mathbf{p}_{v,t+\delta}^{(i)} - \mathbf{p}_{v,t}^{(i)}\big) - \big(\hat{\mathbf{p}}_{v,t+\delta}^{(i)} - \hat{\mathbf{p}}_{v,t}^{(i)}\big) \big\|_1$$

变量：
- $\mathbf{p}_{v,t}^{(i)}$: GT 第 $i$ 个 valid point 在时刻 $t$ 的 3D 坐标
- $\hat{\mathbf{p}}_{v,t}^{(i)}$: 预测的第 $i$ 个点
- $\mathcal{N}$: valid points 集合

注意这个 loss 是在 **displacement field** $\Delta\mathbf{p}$ 上算的，不是直接在 point 上算的。这相当于让模型预测"运动量"而非"位置"——这有好处：
1. 减少对绝对坐标尺度的依赖
2. 在 shared DPM space 里自然学到 inter-frame correspondence
3. 对 small motion 区域梯度更友好（大坐标 L1 loss 在小运动上梯度太小）

paper 强调 $\mathcal{L}_{\text{temp}}$ 是 **implicit** motion supervision，与 DGSHead 的 explicit scene flow 互补。

---

## 5. Dynamic 3D Gaussian Splatting Head (DGSHead) — 显式 motion 学习

这个 head 是 stage-2 才启用的。先看 feature fusion (公式 12-14)：

$$F_{v,t}^{\text{app}} = \text{Conv}(I_{v,t}) \quad \text{（appearance feature）}$$
$$F_{g,v,t}, D_{g,v,t} = \text{DPT}_g(TA_{v,t}) \quad \text{（Gaussian feature + depth）}$$
$$G_{v,t} = F_{v,t}^{\text{app}} + F_{g,v,t}$$

paper 里有个 important observation：**AA block frozen 后会过度强调 geometry，弱化 appearance**。所以这里要把 RGB image 经过 Conv 的 appearance feature fuse 回来。这个 trick 在 Anysplat (https://arxiv.org/abs/2505.23716) 类工作里也常见。

Gaussian 参数化：
$$\{\mu_i, \sigma_i, r_i, c_i, \nu_i\}$$

- $\mu_i \in \mathbb{R}^3$: Gaussian center
- $\sigma_i$: scale
- $r_i$: rotation (quaternion)
- $c_i$: color (SH or RGB)
- $\nu_i \in \mathbb{R}^3$: **velocity vector** ← 这是 paper 的核心创新

**Key insight**：paper 没有让每个 Gaussian 单独预测自己的 velocity，而是用 MTA 的 motion tokens $M_{v,t}^{(l)}$ decode 出一组 **velocity bases** $\nu_b \in \mathbb{R}^3$，然后 Gaussian 通过 attention 从 bases 加权组合出自己的 velocity。这是个 low-rank decomposition 的思路——shared motion basis 大大减少 motion 表达的 over-parameterization，同时让相邻 Gaussian 的 velocity 自然平滑。

Constant velocity 假设 (公式 15)：

$$\mu_{i,t+\delta} = \mu_{i,t} + \delta \cdot \nu_{i,t}$$

这个假设在 short clip (δ ≤ 3) 内合理，对 autonomous driving 中大部分车辆运动近似成立。但在 long sequence 上会 drift——这是 paper 的一个 limitation。

---

## 6. Stage-wise Training + Depth Distillation — 工程上的关键

### 两阶段训练

**Stage 1**: synthetic data (Virtual KITTI, MVS-Synth)，loss 是：
$$\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{cam}} + \mathcal{L}_{\text{depth}} + \mathcal{L}_{\text{point}}^{(t)} + \mathcal{L}_{\text{point}}^{(t+\delta)} + \lambda_{\text{temp}}\mathcal{L}_{\text{temp}}$$

weight $\lambda_{\text{temp}} = 0.01$。LR peak $1\times10^{-6}$，10 epochs。这里只学 implicit motion via FPH，DGSHead 不启用。

**Stage 2**: real data (Waymo)，启用 DGSHead：
$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{stage1}} + \mathcal{L}_{\text{3DGS}}$$

$$\mathcal{L}_{\text{3DGS}} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{gs}}\mathcal{L}_{\text{gsdepth}} + \lambda_{\text{dist}}\mathcal{L}_{\text{distill}} + \lambda_{\text{flow}}\mathcal{L}_{\text{flow}}$$

weights: $\lambda_{\text{gs}} = \lambda_{\text{dist}} = 0.1$, $\lambda_{\text{flow}} = 0.01$。LR peak $5\times10^{-5}$（比 stage 1 高 50 倍！），50 epochs。

直觉：stage 1 学 motion 的"骨架"，stage 2 学 appearance + 精细 dynamic geometry。stage 2 LR 更高是因为只 fine-tune DGSHead + 部分 features，需要让 Gaussian 参数跟上。

### Depth Distillation — 关键 trick

paper 发现直接用 sparse LiDAR 监督 Gaussian depth 会让性能退化。原因：LiDAR 在 autonomous driving 里稀疏且不均匀，导致 Gaussian optimization 收敛到错误的 local minimum。

解决方案：**self-distillation**

$$\mathcal{L}_{\text{distill}} = \big\| D_{g,v,t} - \text{sg}(D_{v,t}^{\text{pm}}) \big\|_1$$

变量：
- $D_{g,v,t}$: stage-2 Gaussian depth branch 预测的 depth (student)
- $D_{v,t}^{\text{pm}}$: stage-1 point-map branch 预测的 dense depth (teacher)
- $\text{sg}(\cdot)$: stop-gradient operator，让 teacher 不被更新

这是个非常巧的设计：**stage-1 的 dense prediction 作为 stage-2 的 pseudo-dense-supervision**，绕开了 LiDAR 稀疏的问题。这和 self-distillation in DINO (https://arxiv.org/abs/2104.14294) 有精神上的相似——用一个 stable 的 branch 教另一个 branch。

$\mathcal{L}_{\text{gsdepth}}$ 是 $L_1$ loss，由 pretrained MoGe (ref [27], https://arxiv.org/abs/2502.18403) 提供 supervision。$\mathcal{L}_{\text{rgb}}$ 是 $\text{MSE}(I, \hat{I})$，标准 3DGS rendering loss。$\mathcal{L}_{\text{flow}} = \text{MSE}(\mathbf{s}, \hat{\mathbf{s}})$ 用 scene flow GT 监督 Gaussian velocity。

---

## 7. 实验结果解读

### Table 1: Point Map Reconstruction (KITTI + Waymo)

| Method | KITTI Acc ↓ | KITTI Comp ↓ | KITTI NC ↑ | Waymo Acc ↓ | Waymo Comp ↓ | Waymo NC ↑ |
|---|---|---|---|---|---|---|
| VGGT | 1.489 | 0.690 | 0.918 | 4.635 | 2.667 | 0.561 |
| StreamVGGT | 1.078 | 0.495 | 0.899 | 4.598 | 2.626 | 0.564 |
| DynamicVGGT | 0.901 | 0.584 | 0.939 | 4.021 | 2.390 | 0.603 |

关键观察：
- KITTI 上 Acc 从 1.489 → 0.901（提升 39%），Comp 略有下降（0.690 → 0.584），NC 提升（0.918 → 0.939）
- Waymo 上 Acc 从 4.635 → 4.021（提升 13%），Comp 从 2.667 → 2.390（10%）
- StreamVGGT 在 Waymo 上 Acc 提升很小（4.635 → 4.598），说明串联 temporal attention 在 multi-view 大场景上效果有限

### Table 2: 4D Scene Reconstruction on Waymo

| Method | Supervision | Dynamic PSNR ↑ | Full PSNR ↑ |
|---|---|---|---|
| 3DGS | Full | 17.13 | 25.13 |
| DeformableGS | Full | 17.10 | 25.29 |
| GS-LRM | Camera | 20.02 | 25.18 |
| STORM | Camera | 21.26 | 25.03 |
| DynamicVGGT | **Image-only** | 18.07 | 24.07 |

DynamicVGGT 在 dynamic region PSNR 18.07，比 STORM (21.26) 低 ~3dB，但 **STORM 需要 calibrated camera** 而 DynamicVGGT 只要 image。这是个重要的 trade-off。

### Table 3: Depth Estimation

| Method | KITTI Mono Abs Rel ↓ | NYU-v2 Abs Rel ↓ | KITTI MVS Abs Rel ↓ |
|---|---|---|---|
| DUSt3R | 0.109 | 0.081 | 0.143 |
| MASt3R | 0.077 | 0.110 | 0.115 |
| MonST3R | 0.098 | 0.094 | 0.107 |
| VGGT | 0.082 | 0.059 | 0.062 |
| StreamVGGT | 0.082 | 0.057 | 0.173 |
| DynamicVGGT | 0.070 | 0.064 | 0.051 |

MVS 上 DynamicVGGT (0.051) > VGGT (0.062)，说明 temporal modeling 在 multi-view 上真的有帮助。NYU-v2 上略有下降 (0.064 vs 0.059)——这是 indoor 场景，paper 没有专门训练，泛化性能下降可以理解。StreamVGGT 在 KITTI MVS 上崩盘 (0.173)，这印证了串联 temporal block 的训练不稳定问题。

### Table 4: Ablation

| Method | KITTI Acc ↓ | Waymo Acc ↓ |
|---|---|---|
| Baseline (VGGT) | 1.489 | 4.635 |
| + TA & FPH (stage1) | 0.927 | 4.330 |
| + DGSHead (stage2) | 0.901 | 4.021 |

观察：
- TA + FPH 单独就能把 KITTI Acc 从 1.489 砍到 0.927（巨大提升）
- DGSHead 在 KITTI 上提升很小 (0.927 → 0.901)，但在 Waymo 上效果显著 (4.330 → 4.021)
- 直觉：DGSHead 的 explicit motion 监督在 large-scale dynamic 场景里更有效

---

## 8. 一些更深的 intuition 和 critique

### 8.1 Motion tokens 的 low-rank 假设

paper 用 M 个 learnable motion tokens decode velocity bases。这隐含假设：**场景中的 motion 可以由 M 个 basis motion 线性组合表达**。这对自动驾驶场景合理（大部分是 ego-motion + 若干 dynamic object 的 rigid motion），但在高度 deformable 场景（人脸、布料）下可能不够。可以联想到 NeRF 里的 DeformableGS (https://arxiv.org/abs/2306.17838) 用 MLP decode offset，vs 这种 basis decomposition。

### 8.2 Constant Velocity 的局限

公式 15 的 $\mu_{i,t+\delta} = \mu_{i,t} + \delta \cdot \nu_{i,t}$ 假设 velocity 恒定。这在 δ ≤ 3 frame 内大致成立，但对 long-horizon 预测会 drift。一个可能的改进方向：把 velocity 也变成 time-dependent $\nu_{i,t}$，或者加 acceleration 项。

### 8.3 Depth distillation 的潜在问题

self-distillation 最大的风险：**student 永远超不过 teacher**。如果 stage-1 的 $D_{v,t}^{\text{pm}}$ 在某些区域系统性偏差，stage-2 的 Gaussian depth 会继承这个偏差。paper 用 $\lambda_{\text{dist}} = 0.1$，相对温和，加上 scene flow 和 RGB loss 一起作用，可能缓解这个问题。但长期看，可能需要 EMA teacher 或者 alternating optimization。

### 8.4 为什么 MTA 比 StreamVGGT 的串联设计更稳？

这是 paper 的核心 architecture claim。我个人的 intuition：VGGT 的 AA block 内部已经有跨 view 的 spatial attention，已经学到非常强的 multi-view geometric reasoning。在它后面加 temporal block，相当于 "在已经收敛的 representation 上再学一遍"——容易 overfit 到新数据。MTA 用 motion tokens 作旁路，相当于 "保留 main highway，开一条 side road"，risk 更可控。

这也呼应了 LoRA (https://arxiv.org/abs/2106.09685) 的精神：不在 backbone 上做改动，而是加 low-rank adapter。

### 8.5 Future Point Head 与 Scene Flow 监督的互补性

$\mathcal{L}_{\text{temp}}$ 在 point-map space 约束 motion，$\mathcal{L}_{\text{flow}}$ 在 Gaussian velocity space 约束 motion。这两者作用在不同 representation level：
- $\mathcal{L}_{\text{temp}}$ 监督 dense geometric structure
- $\mathcal{L}_{\text{flow}}$ 监督 sparse Gaussian primitive 的 velocity

如果只有 $\mathcal{L}_{\text{temp}}$，Gaussian velocity 学不到精确值（只能从 point displacement 间接推）。如果只有 $\mathcal{L}_{\text{flow}}$，Gaussian velocity 学到了，但全局 point map 的 motion 一致性约束缺失。两者一起形成 multi-level supervision。

---

## 9. 限制与未来方向

1. **Long sequence extrapolation**：constant velocity 假设 + 短 clip 训练，长序列会漂移
2. **Velocity basis 数量 $M$ 的选择**：paper 没说具体值，可能是 ablation 没放
3. **Dynamic region PSNR 比 STORM 低 3dB**：image-only supervision 的代价
4. **Scene flow 监督依赖 GT**：Waymo 有 scene flow GT，但很多数据集没有
5. **3DGS rendering 在 Waymo 上 SSIM 只有 0.376 (dynamic region)**：说明 dynamic Gaussian 的 photometric 质量还有很大空间
6. **未与 driving-specific downstream task 评估**：比如 motion planning, object detection on reconstructed 4D scenes

可能的扩展：
- 用 flow matching / diffusion 做 stochastic motion prediction (https://arxiv.org/abs/2512.06112)
- 结合 VLA (Vision-Language-Action) 把 4D reconstruction 直接喂给 planning (https://arxiv.org/abs/2512.11872)
- 用 Gaussian 的 semantic attribute 支持 4D scene understanding
- MTA 替换成 state-space model (Mamba, https://arxiv.org/abs/2312.00752) 处理更长的 temporal horizon

---

## 10. 总结

DynamicVGGT 的核心 design choices 用一句话概括：**保留 VGGT 的 spatial geometric prior，通过 motion token 旁路学 temporal，用 implicit (FPH) + explicit (DGSHead with scene flow) 双重 motion 监督，stage-wise 训练绕开 sparse LiDAR 的问题**。

这个工作在 autonomous driving 4D reconstruction 上迈了扎实的一步。它最大的工程价值在于 MTA 的并行设计——这个设计 pattern 可以推广到其他 frozen foundation model + temporal extension 的场景。

---

## References

- VGGT (CVPR 2025): https://vgg-t.github.io/ | https://arxiv.org/abs/2412.16700
- StreamVGGT: https://arxiv.org/abs/2507.11539
- Dynamic Point Maps: https://arxiv.org/abs/2503.16318
- DUSt3R: https://dust3r.europe.naverlabs.com/
- MASt3R (ECCV 2024): https://arxiv.org/abs/2406.09756
- MonST3R: https://arxiv.org/abs/2410.03825
- MoVieS: https://arxiv.org/abs/2507.10065
- STORM: https://arxiv.org/abs/2501.00602
- DrivingForward (AAAI 2025): https://arxiv.org/abs/2408.01361
- Anysplat: https://arxiv.org/abs/2505.23716
- 3DGS (SIGGRAPH 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- DeformableGS: https://arxiv.org/abs/2306.17838
- DPT: https://arxiv.org/abs/2101.03979
- MoGe: https://arxiv.org/abs/2502.18403
- Waymo Open Dataset: https://waymo.com/open/
- KITTI: http://www.cvlibs.net/datasets/kitti/
- Virtual KITTI: https://virtual-kitti.woodward.win/
- DINOv2: https://arxiv.org/abs/2304.07193
- DINO (self-distillation): https://arxiv.org/abs/2104.14294
- LoRA: https://arxiv.org/abs/2106.09685
- Mamba: https://arxiv.org/abs/2312.00752
- Street Gaussians (ECCV 2024): https://arxiv.org/abs/2401.01339
- OmniRe: https://arxiv.org/abs/2408.16760
- HUGS: https://arxiv.org/abs/2403.17816
