---
source_pdf: MoCapAnything V2 End-to-End Motion Capture for Arbitrary Skeletons.pdf
paper_sha256: 592ac76b024796f3c1639f0967fd2c53f42895254165686048a0e23d0520a9d0
processed_at: '2026-08-05T19:40:29-07:00'
target_folder: Automata
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MoCapAnything V2 的人话版

Karpathy 你好啊，这 paper 我跟你掰开了讲，不绕弯子。

---

## 一句话总结

这 paper 想干一件事：**拍一段 video，丢进去，任何 rigged skeleton 都能动起来**——不管它是人、狗、鸟、还是 Objaverse 里随便什么怪东西。只要给它一个 reference skeleton asset（带一帧 example animation），它就给你输出 animation-ready 的 joint rotation 序列，直接 drive 那个 asset 动起来。

Project page: https://animotionlab.github.io/MoCapAnythingV2/

---

## 问题为什么难

Human motion capture 早被 SMPL [Loper et al. 2015] 那帮人 solve 了——VIBE [Kocabas et al. 2020] 输入 video 输出 SMPL 参数，做得挺漂亮。但 SMPL 是 **fixed skeleton**：24 个 joint，topology 固定，joint 名字固定，最关键的是 **local coordinate frame convention 固定**。这意味着模型只要学会 "在这个固定的坐标系约定下，这种 pose 对应哪种 rotation" 就完事了。

一旦你把 skeleton 换成 arbitrary——狗有 70 个 joint，鸟有 50 个，马有 60 个，Objaverse 里某个人物有 120 个，每个 joint 的 local x 轴朝哪、y 轴朝哪，全看 rigging 艺术家心情——这事就崩了。

SMPL: https://smpl.is.tue.mpg.de/
VIBE: https://arxiv.org/abs/2004.06406

---

## V1 怎么搞的，为什么不行

V1 [Gong et al. 2026] 的思路很直觉：**分两步走，中间用 pose 当桥梁**。

$$\text{Video} \xrightarrow{\text{learned}} \text{Pose} \xrightarrow{\text{analytical IK}} \text{Rotation}$$

理由听起来挺合理：pose（joint 3D position）是跨 skeleton 共享的——不管人走还是狗走，关节运动轨迹在 position space 里看起来都挺像。所以先用网络从 video 预测 pose，再用 analytical IK（比如 FABRIK、CCD 这种经典 solver）把 pose 转成 rotation。

V1 paper: https://arxiv.org/abs/2512.10881

但这个 pipeline 有两个致命问题，我挨个拆。

### 问题一：P→R 是 ill-posed 的

什么叫 ill-posed？就是 **同一个 pose 对应好多个合法的 rotation 序列**，你没法从 pose 唯一确定 rotation。

为什么？看 forward kinematics 公式：

$$\mathbf{p}_j = \mathbf{p}_{\pi(j)} + \mathbf{R}_{\pi(j)} \cdot (\mathbf{R}_j \cdot \mathbf{o}_j)$$

变量解释：
- $\mathbf{p}_j$ = joint $j$ 的 3D position
- $\pi(j)$ = joint $j$ 的 parent joint index
- $\mathbf{R}_j$ = joint $j$ 的 local rotation（你要预测的东西）
- $\mathbf{o}_j$ = rest pose 下 joint $j$ 相对 parent 的 bone offset

问题出在哪？**bone-axis twist 是 unconstrained 的**。想象你有一根胳膊，从肩膀到肘。你绕着"肩膀→肘"这根 bone 轴转肩膀，肘的位置完全不变——因为 $\mathbf{R}_j \cdot \mathbf{o}_j$ 在绕 $\mathbf{o}_j$ 方向旋转时不变（这是 $\text{SO}(2)$ 对称性）。所以 analytical IK 算出来 rotation 后，twist 那个 degree of freedom 是 free 的，它只能随便选一个 default（比如 zero twist）。

结果就是 V1 论文里 Fig. 4 显示的鬼影：joint 在连续帧之间疯狂 spinning，因为每帧 IK 独立求解，twist 那个维度随机漂移。

更深一层：**同一个 pose 在不同 skeleton convention 下对应完全不同的 rotation**。两个 rigged asset，rest pose 骨架一模一样，但 artist A 把 local x 轴沿 bone 方向摆，artist B 把 local x 轴垂直于 bone 摆。同样一个抬手动作，输出的 rotation 序列天差地别。Analytical IK 不知道你的 convention 是啥，它按某种 fixed 规则算，unseen skeleton convention 一变就崩。

### 问题二：Non-differentiable IK 切断了 gradient

IK 是 iterative solver，per-frame 优化几百次迭代，不可微。后果是什么？

- V→P 网络只被 position loss 监督，它学到的 pose 表示 **只为了 position accuracy**
- P→R 阶段完全独立，gradient 没法 backprop 到 V→P
- V→P 学到的 pose 可能 position 精确但对 rotation recovery 极不友好——比如某些 joint 的 position 微小扰动会导致 rotation 解析解剧烈变化，IK 在这种 pose 附近不稳定

这就像你训练一个 detector 只为 mAP，但下游用 detector 输出做 tracking，tracking 的 objective 完全没影响 detector 的训练——detector 学到的 feature 对 tracking 不一定最优。

---

## V2 的核心 Trick：Reference Pose-Rotation Pair 当 Coordinate Anchor

V2 的 insight 极其 elegant。我直接讲 intuition：

**P→R 的 ill-posedness 不是数学上不可解，而是 conditioning 不够**。你只给 pose $\mathbf{P}$ 和 rest pose $\mathbf{o}$，模型不知道这个 skeleton 的 coordinate convention 是啥。那你就 **显式告诉它**——从 target asset 里拿一帧 example animation，也就是一个 (pose, rotation) pair，告诉模型 "在这个 skeleton 的 coordinate definition 下，这个 joint 配置对应这些 rotations"。

这就把问题从：
$$\mathbf{R} = f(\mathbf{P}, \mathbf{o}) \quad \text{(multi-valued, ill-posed)}$$

变成：
$$\mathbf{R} = f(\mathbf{P}, \mathbf{o}, \mathbf{p}^{ref}, \mathbf{r}^{ref}) \quad \text{(well-constrained, conditional prediction)}$$

变量解释：
- $\mathbf{p}^{ref} \in \mathbb{R}^{J \times 3}$ = reference frame 的 joint positions
- $\mathbf{r}^{ref} \in \mathbb{R}^{J \times 6}$ = reference frame 的 6D rotations [Zhou et al. 2019]

6D rotation: https://arxiv.org/abs/1812.07035

**为什么一帧就够？** 因为 coordinate convention 是 per-skeleton 的 static property，一帧 pose-rotation pair 就揭示了 "这个 joint 的 local x/y/z 轴在哪"。Ablation（Table 4）验证：在 Zoo-Seen / Zoo-Rare 上有没有 reference pair 差不多（因为 convention 在训练分布里被 memorize 了），但 Zoo-Unseen 上没 reference pair 崩到 24°，有 reference pair 直接降到 7.37°，加上 rest pose 再降到 6.54°。数量级差异。

类比：这就像学外语。如果只给你一本中文字典（rest pose），你不知道每个字怎么发音；但如果再给你一个 "这个字这样读"的 example（reference pair），你就 anchor 住了整个发音系统。

---

## V2 架构怎么搭的

整体 pipeline 还是两 stage，但**两个 stage 都 learnable，jointly trained end-to-end**：

$$\text{Video} \xrightarrow{\text{Stage 1 (learned)}} \text{Pose} \xrightarrow{\text{Stage 2 (learned)}} \text{Rotation}$$

没了 mesh intermediate，没了 analytical IK。整个 pipeline 可微，rotation loss 能 backprop 到 visual encoder。

### Stage 1: Video-to-Pose

输入：video frames $\{I_1, \ldots, I_T\}$（$T=48$）+ reference frame。

**Reference Query Encoder (Fig. 3A)**：
- Reference joint positions $\mathbf{p}^{ref}$：NeRF 风格的 frequency positional encoding [Mildenhall et al. 2021]，$\gamma(x) = (\sin(2^0 x), \cos(2^0 x), \ldots, \sin(2^{L-1}x), \cos(2^{L-1}x))$，再 project 到 dim $d$
- Per-joint semantic embedding：用 frozen T5 [Raffel et al. 2020] 编码 joint name（"left_shoulder" 等），提供 category-agnostic identity
- Reference image features：frozen DINOv2 [Oquab et al. 2023] 提取的 patch tokens

这些通过 RefFusionBlocks（GL-GMHA self-attention + vanilla self-attn + cross-attn to image）融合，输出 reference joint queries $\mathbf{Q}^{ref} \in \mathbb{R}^{J \times d}$。

NeRF: https://arxiv.org/abs/2003.08934
DINOv2: https://arxiv.org/abs/2304.07193
T5: https://arxiv.org/abs/1910.10683

**Temporal Pose Decoder (Fig. 3B)**：
- Query 是 $\mathbf{Q}^{ref}$，cross-attend 到 per-frame image features $\mathbf{z}_t$（同一个 DINOv2）
- GL-GMHA 做 spatial reasoning across joints
- Windowed per-joint temporal attention with RoPE [Su et al. 2024] across frames
- 输出 $\hat{\mathbf{P}} = \{\hat{\mathbf{p}}_t\} \in \mathbb{R}^{T \times J \times 3}$

RoPE: https://arxiv.org/abs/2104.09864

**为什么去掉 mesh？** V1 用 mesh 当 video→joint 的 bridge，因为 mesh 提供几何信息。但 predicted mesh 有噪声，噪声 propagate 进下游，而且 mesh reconstruction 慢得要死（V1 在 120 frames 上 mesh 阶段要 15 分钟）。V2 直接 video→pose，既快又 robust。Table 2 显示 V2 无 mesh 达到 2.20cm position error，V1 with GT mesh 1.06cm，V1 with predicted mesh 3.30cm——V2 不用 mesh 就能跟 V1-with-GT-mesh 差不多，但 GT mesh 推理时不可用。

### Stage 2: Pose-to-Rotation

这是 V2 最 novel 的部分，把 P→R 变成 learnable module。

**Rest Pose Encoder**：输入 bone offsets $\mathbf{o} \in \mathbb{R}^{J \times 3}$ + semantic embeddings，输出 $\mathbf{E}^{rest} \in \mathbb{R}^{J \times d}$，capture static geometry and topology。

**Reference Encoder (Anchor Encoder, Fig. 3D)**：输入 reference position $\mathbf{p}^{ref}$ + 6D rotation $\mathbf{r}^{ref}$，用 FiLM [Perez et al. 2018] 被 $\mathbf{E}^{rest}$ 调制，输出 coordinate-system anchor $\mathbf{C}^{ref} \in \mathbb{R}^{J \times d}$。

FiLM: https://arxiv.org/abs/1709.07871

FiLM 公式：$\mathbf{h}' = \gamma \odot \mathbf{h} + \beta$，其中 $\gamma, \beta$ 由 condition $\mathbf{E}^{rest}$ 通过 MLP 产生。直觉：rest pose feature "调制" reference pair feature，让 anchor 携带 skeleton-specific 结构信息。

**Pose Encoder (Fig. 3C)**：处理 predicted pose sequence $\mathbf{P}$，alternating GL-GMHA + windowed temporal attention with RoPE，输出 pose feature $\mathbf{Q} \in \mathbb{R}^{T \times J \times d}$。

**Rotation Decoder (Fig. 3E)**：8 个 block，每个 block 按 order：
1. FiLM modulation by $\mathbf{E}^{rest}$（skeleton-specific conditioning）
2. Per-joint temporal self-attention with RoPE（temporal coherence）
3. GL-GMHA spatial attention（cross-joint reasoning）
4. Cross-attention to $\mathbf{C}^{ref}$（只在 first $L_{cross}=6$ layers）
5. Feed-forward residual
6. Final layer → 2-layer MLP → 6D rotation $\mathbf{r}_t \in \mathbb{R}^{J \times 6}$

Table 9 显示 $L_{cross}=0$（不用 reference）时 Zoo-Unseen 崩到 23.49°，$L_{cross}=6$ 最佳 6.54°，$L_{cross}=8$ 略有退化——over-conditioning diminishing return。

---

## GL-GMHA：Local + Global 交替

这 paper 的另一个 contribution。建立在 AnyTop [Gat et al. 2025] 的 GMHA 之上。

AnyTop: https://dl.acm.org/doi/10.1145/3721238.373062

GMHA 用 graph-derived joint relations 当 attention bias：
$$\text{Attn}(i, j) = \text{softmax}_j \left( \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}} + b_{ij} \right)$$

其中 $b_{ij}$ 是 graph-based bias（kinematic connectivity + graph distance）。

V2 的 twist：**交替用 local mask 和 global mask**。
- Local layers：attention restricted 到 kinematic chain（ancestor mask），capture intra-limb dependencies
- Global layers：full connectivity，capture cross-limb coordination

Table 6 ablation：
- Full Attn (no graph bias)：Zoo-Unseen 11.92°
- GMHA all-global [Gat et al. 2025]：6.69°
- All-local（每层都 ancestor mask）：11.60°——失去 cross-branch coordination
- **GL-GMHA (Ours)：6.54°**

直觉：local 层抓 "我这胳膊该往哪弯"，global 层抓 "胳膊和腿要协调"。单 local 没全局视野，单 global 缺 kinematic chain 的 inductive bias，交替最好。

---

## Loss Function

Total loss（公式 2）：

$$\mathcal{L} = \lambda_{pos} \mathcal{L}_{pos} + \lambda_{rot} \mathcal{L}_{rot} + \lambda_{rot\_v} \mathcal{L}_{rot\_v} + \lambda_{root} \mathcal{L}_{root}$$

各项人话解释：
- $\mathcal{L}_{pos}$：predicted vs GT joint position 的 L2 距离，单位 cm
- $\mathcal{L}_{rot}$：geodesic angular error，就是两个 rotation matrix 之间的夹角 $\text{arccos}\left(\frac{\text{tr}(\mathbf{R}_1^{-1}\mathbf{R}_2) - 1}{2}\right)$，单位 degree
- $\mathcal{L}_{rot\_v}$：angular velocity difference，相邻帧 rotation 差分 $\omega_t = \log(\mathbf{R}_t \mathbf{R}_{t-1}^{-1})$，penalize $\|\hat{\omega}_t - \omega_t^{gt}\|$，promote temporal smoothness
- $\mathcal{L}_{root}$：root joint rotation 额外加权，加速 global orientation 收敛

权重 $\lambda_{pos} = \lambda_{rot} = \lambda_{rot\_v} = 1.0$，$\lambda_{root} = 0.1$。

Per-joint masking 处理 variable joint count。

---

## End-to-End Training 的 Mixed-Pose Schedule

train/test gap 问题：训练时 P→R 可以吃 GT pose，推理时只能吃 noisy predicted pose。直接 train 在 GT pose 上，推理时 distribution shift 崩掉。

解决：scheduled sampling。每个 batch sample 随机选 GT or predicted pose 喂给 rotation module，predicted 的概率按 schedule 退火（公式 3）：

$$p_{pred}(e) = p_{start} + (p_{end} - p_{start}) \cdot \min\left(1, \frac{e}{E_{warmup}}\right)$$

变量：
- $e$ = current epoch
- $E_{warmup}$ = transition rate，default 30
- $p_{start} = 0.1$：早期 90% GT pose，稳定收敛
- $p_{end} = 1.0$：末期全用 predicted pose，逼模型适应自己的 noise

Table 3 验证 end-to-end 的价值：
- Mixed (gradient detached)：Zoo-Unseen 7.82°
- Mixed (with joint opt)：Zoo-Unseen **6.54°**

Gradient flow 让 Unseen 提升 1.3°。这证明 benefit 来自 **co-adaptation**——V→P 学到的 pose 表示被 reshape 成对 rotation recovery 最有用的形式，而不只是 position 准。

GT pose only：Zoo-Unseen 13.28°（distribution gap 崩）
Pred pose only：早期噪声大不稳定
Mixed with joint opt：6.54°（最佳）

这是 teacher forcing 退火的经典思路，类似 seq2seq 训练里 scheduled sampling。

---

## 实验结果说什么

### Table 1：主结果

| Method | Zoo-Seen Ang | Zoo-Rare Ang | Zoo-Unseen Ang | Obj Ang |
|---|---|---|---|---|
| HRNet | 19.86 | 24.72 | 24.59 | 31.37 |
| GLoT | 20.24 | 26.13 | 25.95 | 29.07 |
| ViTPose | 20.90 | 25.48 | 24.46 | 29.30 |
| VIBE | 19.67 | 25.06 | 25.74 | 28.72 |
| **Ours** | **10.73** | **14.38** | **6.54** | **11.06** |

HRNet: https://arxiv.org/abs/1902.09212
ViT-Pose: https://arxiv.org/abs/2204.07343
GLoT: https://arxiv.org/abs/2303.10375

几个值得注意的点：

1. **所有 baseline 都 cap 在 ~20°**：即使给它们 end-to-end training + 相同 reference input，它们的 architecture 没法 leverage reference 来 resolve coordinate ambiguity。artifact 是 joint spinning。

2. **V2 砍半到 ~10°**：reference conditioning 的威力。

3. **Zoo-Unseen (6.54°) 比 Zoo-Seen (10.73°) 还低**！反直觉但解释合理：Unseen split 里 motion 多是 common locomotion（走、跑），rotation 本身简单；Seen/Rare 里有更 exotic 的 motion。一旦 coordinate axis 被 reference anchor 住，locomotion 的 rotation 就 trivial 了。这说明 **reference conditioning 在 convention 未知场景下收益最大**。

4. **AngV Err 也显著降**（0.17 vs 0.5+）：temporal consistency 大幅提升，这是 learned temporal decoder 相比 per-frame IK 的优势。

### Table 2：V1 vs V2

| Config | Seen Ang | Rare Ang | Unseen Ang |
|---|---|---|---|
| V1 (GT Mesh + IK) | 17.47 | 18.52 | 20.56 |
| V1 (Pred Mesh + IK) | 20.02 | 19.82 | 22.04 |
| **Ours (no mesh)** | **10.91** | **14.36** | **6.68** |

V1 即使 with GT mesh（理想情况，推理不可用）都 17-22°，因为 analytical IK resolve 不了 twist。V2 即使没 mesh 也比 V1-with-GT-mesh 好一倍。**end-to-end learnable P→R + reference conditioning 的收益远超 mesh intermediate 能提供的几何信息**。

---

## Efficiency

V1 在 120-frame sequence 上：
- Feature extraction ~40s
- Mesh reconstruction ~15 min（最慢）
- Pose estimation ~20s
- IK optimization ~5 min
- Total > 20 min

V2：
- Feature extraction ~40s（一样）
- Pose + rotation 单次 forward pass ~10s
- Total < 1 min
- **~20× speedup**

去掉 mesh reconstruction 和 iterative IK，换成 batched forward pass。而且没牺牲精度——average angle error 10.6° vs V1 (GT Mesh) 18.9° vs V1 (Pred Mesh) 20.63°。win-win。

---

## Ablation 的 Intuition

### Intermediate Pose Representation（Table 5）

| Architecture | Seen | Rare | Unseen |
|---|---|---|---|
| Direct (V→R) | 9.32 | 12.71 | 23.73 |
| Latent + Aux | 9.06 | 11.85 | 23.57 |
| **Full (explicit pose)** | 10.73 | 14.38 | **6.54** |

Direct V→R 在 Seen/Rare 上 competitive，但 Unseen 崩到 23.73°。Latent + Aux（用 auxiliary pose loss 监督 latent）也不行。

**Insight**：explicit pose 是 skeleton-shared canonical representation，强制 representation 只 encode transferable 信息。没有 explicit bottleneck，model 没法跨 topology generalize——latent 即使被 auxiliary supervise 也 lacks explicit skeleton-shared structure。

这是反直觉但 recurring 的 insight：**bottleneck = inductive bias**，explicit bottleneck 反而促进 generalization。

### Attention Mechanism（Table 6）

如前所述，GL-GMHA 最佳。local + global 互补，all-local 失去全局，all-global 缺 inductive bias。

### Model Depth（Table 8）

6 → 8 → 12：8 最佳（6.54°），12 退化（7.66°）。capacity 和 optimization 的 tradeoff，太深 overfits Seen/Rare 反而 hurt Unseen。

### Cross-Attention Depth $L_{cross}$（Table 9）

$L_{cross}=0$ 崩到 23.49°（reference conditioning essential），$L_{cross}=6$ 最佳，$L_{cross}=8$ 略退化（over-conditioning diminishing return）。

---

## Limitation

1. **Unnatural retargeting drift**：P→R decoder 隐式 learn per-skeleton motion prior，把 bird flapping retarget 到 dog 让它"飞"，predicted rotation 会 drift 到 dog 的 typical configuration。解决方法是 augment training set with 不自然配置。
2. **Occlusion 没处理**：未来工作。
3. **Animal 数据集只有 ~1000 sequences**：scale up 会更好。

---

## 我的 Takeaway

这 paper 给我几个 deep 的 intuition：

**第一，ill-posedness 通常不是 mathematical obstacle，而是 conditioning 不足**。你觉得一个 problem under-determined，往往是你没给模型足够的 context。这里 twist ambiguity 看起来是 mathematical，但只要 anchor 住 coordinate convention，ambiguity 就 resolve 了。

**第二，end-to-end 的 benefit 来自 intermediate representation 的 co-adaptation**。V1 的 pose 表示只优化 position accuracy；V2 的 pose 表示被 rotation loss reshape 成"对 rotation recovery 最有用"的 form。这种 co-adaptation 是 factorized pipeline 永远做不到的，无论 IK solver 多 smart。

**第三，explicit bottleneck 促进 generalization**。Direct V→R 看起来更 elegant（少一个 stage），但跨 skeleton generalize 失败；explicit pose intermediate 强制 representation 只 encode transferable 信息，反而 generalize 好。

**第四，reference conditioning 是 arbitrary-skeleton 问题的关键**。fixed-skeleton 方法（SMPL）的 coordinate convention 是 baked-in 的，所以不需要 reference；arbitrary-skeleton 方法必须 explicit 提供 convention，否则 model 不知道在哪个 frame 下输出 rotation。Reference pose-rotation pair 是 minimal sufficient conditioning，因为任何 rigged asset 都天然有它。

**潜在 future direction**：
- Multiple reference pairs 的 weighted ensemble（极端 skeleton 更鲁棒）
- Diffusion-based P→R 来 model multi-modal rotation distribution（truly ambiguous cases）
- Occlusion handling via temporal smoothing + uncertainty estimation
- Physics constraint（ground reaction force, momentum conservation）further constrain rotation
- 用 V2 output 作为 RL reward 训练 character controllers in simulation
- 与 text-to-motion 结合：text condition 直接 drive arbitrary skeleton 动画
- Real-time 推理部署（mobile / AR 场景）

Paper: https://animotionlab.github.io/MoCapAnythingV2/
V1 arXiv: https://arxiv.org/abs/2512.10881
AnyTop: https://anytop.github.io/
6D Rotation: https://arxiv.org/abs/1812.07035
DINOv2: https://arxiv.org/abs/2304.07193
NeRF: https://arxiv.org/abs/2003.08934
T5: https://arxiv.org/abs/1910.10683
RoPE: https://arxiv.org/abs/2104.09864
FiLM: https://arxiv.org/abs/1709.07871
VIBE: https://arxiv.org/abs/2004.06406
SMPL: https://smpl.is.tue.mpg.de/
HRNet: https://arxiv.org/abs/1902.09212
ViT-Pose: https://arxiv.org/abs/2204.07343
Objaverse: https://objaverse.allenai.org/
Truebones: https://truebones.gumroad.com/l/skZMC

---

# MoCapAnything V2: End-to-End Motion Capture for Arbitrary Skeletons 深度解析

## 1. Problem Setting 与核心 Motivation

这篇 paper 解决的是一个相当 fundamental 的问题：给定一段 monocular video $\mathbf{V} = \{I_1, \ldots, I_T\}$ 和一个 arbitrary rigged skeleton $\mathcal{S}$（由 parent indices $\pi$、bone offsets $\mathbf{o} \in \mathbb{R}^{J \times 3}$ 即 rest pose、per-joint semantic labels 定义），预测 per-frame 的 joint rotations $\mathbf{R} = \{\mathbf{r}_1, \ldots, \mathbf{r}_T\}$，其中 $\mathbf{r}_t \in \mathbb{R}^{J \times 6}$ 用 6D representation [Zhou et al. 2019] 表示，使得 $\mathcal{S}$ 能 perform 出 video 里的 motion。

关键变量：
- $J$ = joint 数量（任意，最大 150，因为训练集中最大的 skeleton 有 143 joints）
- $T$ = sequence length（训练时 48 frames）
- $\pi$ = parent index array，定义 tree topology
- $\mathbf{o}$ = bone offsets，即 rest pose 下每个 joint 相对 parent 的 offset vector
- $\mathbf{r}_t \in \mathbb{R}^{J \times 6}$ = 6D rotation representation，每 joint 6 个数（两个 unit vector 拼起来，比 quaternion / Euler 连续性好）

这个问题难在哪里？human motion capture（SMPL 系列）已经做得很好了 [Kocabas et al. 2020; Kanazawa et al. 2018a]，但它们绑死在 fixed skeleton topology 上。这里要 generalize 到 arbitrary skeleton——狗、鸟、马、甚至 Objaverse 里那些非动物的 rigged asset。topology 变了，joint 数量变了，最棘手的是 **local coordinate convention** 变了。

Project page: https://animotionlab.github.io/MoCapAnythingV2/
V1 paper: https://arxiv.org/abs/2512.10881

---

## 2. V1 的 Factorized Design 与其 Fundamental Limitation

V1 [Gong et al. 2026] 的 pipeline 是这样的：

$$\text{Video} \xrightarrow{\text{V→P (learned)}} \text{Mesh} \xrightarrow{\text{Pose}} \xrightarrow{\text{P→R (analytical IK)}} \text{Rotation}$$

V1 用了一个 4D mesh intermediate（mesh reconstruction），然后从 mesh 推 joint positions，再用 analytical IK（比如基于 Jacobian 的迭代优化，或者 closed-form IK like CCD / FABRIK）求 rotations。

这个设计有两个致命问题：

### 2.1 P→R 是 Ill-Posed 的

这是整篇 paper 最核心的 insight。Joint positions $\mathbf{P} = \{\mathbf{p}_t\}, \mathbf{p}_t \in \mathbb{R}^{J \times 3}$（root-relative 3D positions in camera frame）**不能 fully determine** rotations $\mathbf{R}$。

为什么？考虑 forward kinematics：
$$\mathbf{p}_j = \mathbf{p}_{\pi(j)} + \mathbf{R}_{\pi(j)} \cdot (\mathbf{R}_j \cdot \mathbf{o}_j)$$

其中 $\mathbf{R}_j$ 是 joint $j$ 的 local rotation，$\mathbf{o}_j$ 是 rest pose 下的 bone offset。给定所有 $\mathbf{p}_j$ 和 $\mathbf{o}_j$，要反求 $\mathbf{R}_j$。

问题在于 **bone-axis twist 是 unconstrained 的**。沿着 bone 方向的旋转（即 twist around $\mathbf{o}_j / \|\mathbf{o}_j\|$）完全不改变 child joint 的位置——因为 $\mathbf{R}_j \cdot \mathbf{o}_j$ 在绕 $\mathbf{o}_j$ 轴旋转时不变。这是一个 $\text{SO}(2)$ 的 ambiguity per joint，analytical IK 没法 resolve，只能选个 arbitrary default（比如 zero twist），结果就是 V1 出现的 joint spinning artifacts（Fig. 4 里能看到 ghosted frames 上关节乱转）。

更深层的问题：**同一个 pose 在不同 skeleton 的 local coordinate convention 下对应不同的 rotations**。Rest pose $\mathbf{o}$ 只定义了每个 joint local frame 的 **origin**（即 bone 的方向），但没定义 local frame 的 **axes**（即哪个方向是 x、y、z 轴）。两个不同的 rigged asset，rest pose 可能一样，但 local x 轴可能一个是沿 bone 方向，一个垂直于 bone 方向。这样同样的 pose 序列就要输出完全不同的 rotation 序列。

V1 因为 P→R 是 analytical 的，它没法 learn 这个 convention——它只能机械地按某种固定 convention 算 rotations，导致在 unseen skeleton 上 axis convention 一变就崩。

### 2.2 Non-Differentiable IK Block 端到端训练

IK 是 iterative solver（典型用法是 per-frame 优化几百次迭代），不可微，或者就算可微也 gradient 不稳定。这意味着：
- V→P 阶段的 gradient 来自 position loss $\mathcal{L}_{pos}$
- P→R 阶段完全独立优化
- V→P 学到的 pose 表示只为了 position accuracy，**不知道下游 rotation 需要什么**

这是 factorized pipeline 的根本缺陷：intermediate representation 没法 co-adapt 到 final objective。

---

## 3. V2 的核心思想：Reference Pose-Rotation Pair 作为 Coordinate Anchor

V2 的 key insight 是：**P→R 的 ill-posedness 来自 missing coordinate-system information**。要 resolve 它，需要显式告诉模型 "这个 skeleton 的 local frame convention 是什么样的"。

V2 的做法是：除了 rest pose $\mathbf{o}$（提供 origin），再加一个 **reference pose-rotation pair** $(\mathbf{p}^{ref}, \mathbf{r}^{ref})$，从同一个 rigged asset 采样得到。这个 pair 是任意 rigged asset 天然就有的——任何带动画的 rigged model 都能拿出一帧 pose 和对应的 rotation。

直觉上：
- Rest pose $\mathbf{o}$ → 提供 local frame 的 **origin**（bone 方向）
- Reference pair $(\mathbf{p}^{ref}, \mathbf{r}^{ref})$ → 提供 local frame 的 **axes**（坐标系朝向）

两者合起来 fully specify local frame convention。模型学到的是 "对于这个 skeleton 的 coordinate definition，这个 joint configuration 对应这些 rotations"——这把 multi-valued mapping $\mathbf{R} = f(\mathbf{P}, \mathbf{o})$ 变成 well-constrained conditional prediction $\mathbf{R} = f(\mathbf{P}, \mathbf{o}, \mathbf{p}^{ref}, \mathbf{r}^{ref})$。

这个 idea 非常漂亮，因为它把一个 mathematically ill-posed 的 inverse problem 转化成了 supervised learning 问题——只要 asset 里有任意一帧 ground truth animation（这对于任何 production-ready rigged asset 都是 trivially available），就能 anchor 整个 coordinate system。

**为什么 single reference pair 就够？** 因为 local coordinate convention 是一个 per-skeleton 的 static property，一帧就足以揭示它。Ablation study（Table 4）验证了这点：在 Zoo-Seen / Zoo-Rare 上，有没有 reference pair 差不多（因为 axis convention 在训练分布里被 memorize 了），但在 Zoo-Unseen 上，没 reference pair 角度误差从 $24.05°$ 飙到 $24.26°$，加 reference pair 暴跌到 $7.37°$，加 rest pose 再降到 $6.54°$。这是数量级的提升。

---

## 4. 整体 Architecture 拆解

V2 的 pipeline（Fig. 3）：

$$\text{Video} \xrightarrow{\text{Stage 1 (learned)}} \text{Pose} \xrightarrow{\text{Stage 2 (learned)}} \text{Rotation}$$

两个 stage 都是 learnable，jointly trained end-to-end。整个 framework 由 5 个 module 组成（对应 Fig. 3 的 A-E）：

### 4.1 Video-to-Pose Module（Stage 1）

#### A. Reference Query Encoder

输入：
- Reference joint positions $\mathbf{p}^{ref} \in \mathbb{R}^{J \times 3}$：通过 frequency-based positional embedding [Mildenhall et al. 2021]（即 NeRF 风格的 $\gamma(x) = (\sin(2^0 x), \cos(2^0 x), \ldots, \sin(2^{L-1} x), \cos(2^{L-1} x))$）编码，再 project 到 dimension $d$
- Per-joint semantic embeddings：用 frozen T5 [Raffel et al. 2020] text encoder 编码 joint name（比如 "left_shoulder", "neck"），提供 category-agnostic joint identity，能 generalize 到 arbitrary joint naming
- Reference image features $\mathbf{Z}^{ref} \in \mathbb{R}^{P \times d_{img}}$：从 frozen DINOv2 [Oquab et al. 2023] 提取的 patch tokens

这些通过 stack of **RefFusionBlocks** 融合，每个 block 包含：
1. GL-GMHA self-attention over joints（spatial reasoning）
2. Vanilla self-attention
3. Cross-attention to image features

输出：reference joint queries $\mathbf{Q}^{ref} \in \mathbb{R}^{J \times d}$，编码 skeletal structure + reference appearance。

DINOv2 paper: https://arxiv.org/abs/2304.07193
NeRF: https://arxiv.org/abs/2003.08934
T5: https://arxiv.org/abs/1910.10683

#### B. Temporal Pose Decoder

输入：
- $\mathbf{Q}^{ref}$ 作为 query
- Per-frame image features $\mathbf{z}_t$（同一个 frozen DINOv2 encoder）

架构是 temporal transformer：
- GL-GMHA 用于 spatial reasoning across joints
- Windowed per-joint temporal attention with RoPE [Su et al. 2024] across frames
- 输出 $\hat{\mathbf{P}} = \{\hat{\mathbf{p}}_t\} \in \mathbb{R}^{T \times J \times 3}$

RoPE (Rotary Position Embedding): https://arxiv.org/abs/2104.09864

**Position-static joint handling**：preprocessing 阶段 flag 一些 joint 是 position-static 的（比如某些 rigged asset 的 root 或者 fake joints），这些 joint 的 position 直接用 reference position 覆盖，保证 structural consistency。

#### 关键设计选择：No Mesh Intermediate

V1 用 mesh 作为 video 到 joint 的 bridge。V2 直接从 video 到 joint positions，跳过 mesh。原因：
- Predicted mesh 噪声会 propagate 进 downstream
- Mesh reconstruction 本身很慢（V1 mesh 阶段 ~15 min for 120 frames）
- Joint positions 是 skeleton-shared representation，跨 skeleton generalize 好

Table 2 显示：V1 with GT Mesh position error 最低（1.06cm on Seen），但 GT mesh 推理时不可用；V1 with Pred Mesh 退化到 3.30cm；V2 不用 mesh 也能达到 2.20cm，competitive with GT mesh，且 rotation 准确度高一个数量级。

### 4.2 Pose-to-Rotation Module（Stage 2）

这是 V2 最 novel 的部分。它把 P→R 变成 learnable module。

#### C. Rotation Prompt Encoder (Pose Encoder)

处理 predicted pose sequence $\mathbf{P} \in \mathbb{R}^{T \times J \times 3}$：
- Alternating GL-GMHA + per-joint windowed temporal attention with RoPE
- Optionally modulated by $\mathbf{E}^{rest}$
- 输出 pose feature sequence $\mathbf{Q} \in \mathbb{R}^{T \times J \times d}$

#### D. Anchor Encoder (Reference Encoder)

这是 reference conditioning 的核心：
- 输入 reference position $\mathbf{p}^{ref}$ 和 6D rotation $\mathbf{r}^{ref}$
- FiLM-modulated [Perez et al. 2018] by $\mathbf{E}^{rest}$
- 输出 coordinate-system anchor $\mathbf{C}^{ref} \in \mathbb{R}^{J \times d}$

FiLM (Feature-wise Linear Modulation): https://arxiv.org/abs/1709.07871

FiLM 的公式是 $\mathbf{h}' = \gamma \odot \mathbf{h} + \beta$，其中 $\gamma, \beta$ 由 condition $\mathbf{E}^{rest}$ 通过 MLP 产生。这里用 rest pose feature 调制 reference pair feature，让 anchor 携带 skeleton-specific 的结构信息。

还有 (i) **Rest Pose Encoder**：输入 bone offsets $\mathbf{o} \in \mathbb{R}^{J \times 3}$ + semantic embeddings，输出 $\mathbf{E}^{rest} \in \mathbb{R}^{J \times d}$，captures static geometry and topology。

#### E. Rotation Decoder

$L=8$ blocks，每个 block 按 order 应用：
1. **FiLM modulation by $\mathbf{E}^{rest}$**：skeleton-specific conditioning
2. **Per-joint temporal self-attention** (windowed with RoPE)：temporal coherence
3. **GL-GMHA spatial attention** (alternating local/global masking)：cross-joint reasoning
4. **Per-joint cross-attention to $\mathbf{C}^{ref}$** (只在 first $L_{cross} \leq L$ layers)：reference anchor 注入。$L_{cross}=6$ 是 default，Table 9 显示 $L_{cross}=0$ 时 Zoo-Unseen 崩到 23.49°（彻底没用 reference），$L_{cross}=6$ 达到 6.54° 最佳，$L_{cross}=8$ 略有退化说明 over-conditioning diminishing return。
5. **Feed-forward residual**
6. Final layer → 2-layer MLP → 6D rotation $\mathbf{r}_t \in \mathbb{R}^{J \times 6}$

**Rotation-static joint handling**：类似 position-static，但独立 flag——某些 joint 在 rest 下是 rotation-fixed 的，直接 overwrite 为 reference rotation。

---

## 5. GL-GMHA: Global-Local Graph-Guided Multi-Head Attention

这是 V2 的另一个贡献，建立在 GMHA [Gat et al. 2025] 之上（AnyTop 用的 attention）。

AnyTop: https://anytop.github.io/ 或 https://dl.acm.org/doi/10.1145/3721238.373062

### 5.1 GMHA Background

GMHA 用 graph-derived joint relations 作为 attention bias。给定 joints $i, j$：
- Kinematic connectivity（是否在同一 kinematic chain 上）
- Graph distance（skeleton tree 上的最短路径长度）

这些作为 additive bias 加到 attention logits 上：
$$\text{Attn}(i, j) = \text{softmax}_j \left( \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}} + b_{ij} \right)$$

其中 $b_{ij}$ 是 graph-based bias。

### 5.2 GL-GMHA: Local-Global Alternation

V2 引入 alternation：
- **Local layers**：attention restricted to kinematic chain（ancestor mask）——只 attend 到自己 ancestor path 上的 joints，capture intra-limb dependencies
- **Global layers**：full connectivity，capture cross-limb coordination

交替堆叠，无需额外参数，自然 generalize 到 diverse topology。

Table 6 的 ablation：
- Full Attn (no graph bias)：Zoo-Unseen 11.92°
- GMHA all-global [Gat et al. 2025]：6.69°
- All-local (ancestor mask every layer)：11.60°——失去 cross-branch coordination
- **GL-GMHA (Ours)：6.54°**——local + global 互补

直觉：local 层抓 "我的胳膊该往哪儿弯"，global 层抓 "我的胳膊和腿要协调"。单纯 local 不能整体协调，单纯 global 缺少 kinematic chain 的 inductive bias。

---

## 6. Loss Function 详解

Total loss（公式 2）：

$$\mathcal{L} = \lambda_{pos} \mathcal{L}_{pos} + \lambda_{rot} \mathcal{L}_{rot} + \lambda_{rot\_v} \mathcal{L}_{rot\_v} + \lambda_{root} \mathcal{L}_{root}$$

各项含义：
- $\mathcal{L}_{pos}$：per-joint position error between predicted 和 GT joint positions，$\|\hat{\mathbf{p}} - \mathbf{p}^{gt}\|_2$ averaged over joints
- $\mathcal{L}_{rot}$：geodesic angular error between predicted 和 GT rotations，$\text{arccos}\left(\frac{\text{tr}(\mathbf{R}_1^{-1}\mathbf{R}_2) - 1}{2}\right)$ averaged over joints。这是 rotation 之间最自然的 metric，单位是 degree
- $\mathcal{L}_{rot\_v}$：angular velocity difference，promote temporal consistency。Angular velocity 可以从相邻帧 rotation 差分计算：$\omega_t = \log(\mathbf{R}_t \mathbf{R}_{t-1}^{-1})$，然后 penalize $\|\hat{\omega}_t - \omega_t^{gt}\|$
- $\mathcal{L}_{root}$：root joint rotation error 的额外 weighting，加速 global orientation 收敛

权重：$\lambda_{pos} = \lambda_{rot} = \lambda_{rot\_v} = 1.0$，$\lambda_{root} = 0.1$

**Per-joint masking**：因为不同 skeleton joint 数量不同（padded 到 150），loss 只在 valid joints 上计算。

---

## 7. End-to-End Training 与 Mixed-Pose Schedule

### 7.1 Gradient Coupling 的核心 Benefit

让 P→R 可微后，rotation loss 的 gradient 能 backprop through predicted pose 进入 visual encoder。这意味着 V→P 学到的 pose 表示会 reshape 自己——不再只为 position accuracy 优化，而是为 downstream rotation objective 优化。

Table 3 直接验证：
- **Mixed (gradient detached)**： Zoo-Unseen 7.82°
- **Mixed (with joint opt, ours)**： 6.54°

Gradient flow 让 Unseen 提升 1.3°，证明 benefit 来自 end-to-end co-adaptation，不只是 P→R 本身 learnable。

### 7.2 Mixed-Pose Training Schedule

Train/test gap：训练时 P→R 可以接收 GT pose，推理时只能用 predicted pose（有噪声）。

解决方法：stochastically 选择是否 feed GT or predicted pose 给 rotation module。Probability of using predicted pose follows schedule（公式 3）：

$$p_{pred}(e) = p_{start} + (p_{end} - p_{start}) \cdot \min\left(1, \frac{e}{E_{warmup}}\right)$$

变量含义：
- $e$ = current epoch
- $E_{warmup}$ = transition rate（default 30）
- $p_{start} = 0.1$：训练初期 90% 用 GT pose，10% 用 predicted——保证 stable convergence
- $p_{end} = 1.0$：训练末期全部用 predicted pose——逼模型适应自己的 noise distribution

Table 7 显示 $E_w \in [20, 50]$ 都稳定，$E_w=30$ 最佳，方法对 schedule 不 sensitive。

Table 3 的对比：
- GT pose only：Zoo-Unseen 13.28°（distribution gap，崩）
- Pred pose only：早期噪声大，不稳定
- Mixed with joint opt：6.54°（最佳）

这是典型的 scheduled sampling / curriculum learning 思路，类似 Sequence-to-Sequence 训练里的 teacher forcing 退火。

---

## 8. 实验 Dataset 与 Baselines

### 8.1 Datasets

- **Truebones Zoo** [Truebones]: 1038 animal motion sequences, 104,715 frames。Test 60 sequences 分 Seen / Rare / Unseen
- **Objaverse** [Deitke et al. 2023a, b]: 1000 samples，humanoid + non-animal targets，全部 OOD
- **In-the-wild**：Internet videos，real-world robustness 测试

Truebones: https://truebones.gumroad.com/l/skZMC
Objaverse: https://objaverse.allenai.org/

### 8.2 Evaluation Metrics

- **MPJPE** (Mean Per Joint Position Error, cm)
- **MPJVE** (Mean Per Joint Velocity Error, cm)
- **Ang. Err** (geodesic angle error, °)
- **AngV Err** (angular velocity error, °)

所有 sample normalized to $[-1, 1]^3$ 训练，rescaled to $1\text{m}^3$ 评估，处理 inter-species scale variation。

### 8.3 Baselines

HRNet [Sun et al. 2019], ViT-Pose [Xu et al. 2022b], VIBE [Kocabas et al. 2020], GLoT [Shen et al. 2023]。每个 baseline 都 instantiated V→P 和 P→R 模块，end-to-end trained 在相同数据上。直接对比点是 V1。

VIBE: https://arxiv.org/abs/2004.06406
HRNet: https://arxiv.org/abs/1902.09212
ViT-Pose: https://arxiv.org/abs/2204.07343

---

## 9. Main Results（Table 1）深度分析

| Method | Zoo-Seen Ang | Zoo-Rare Ang | Zoo-Unseen Ang | Obj Ang |
|---|---|---|---|---|
| HRNet | 19.86 | 24.72 | 24.59 | 31.37 |
| GLoT | 20.24 | 26.13 | 25.95 | 29.07 |
| ViTPose | 20.90 | 25.48 | 24.46 | 29.30 |
| VIBE | 19.67 | 25.06 | 25.74 | 28.72 |
| **Ours** | **10.73** | **14.38** | **6.54** | **11.06** |

几个有意思的 observation：

1. **Baselines 全部 cap 在 ~20°**：即使给它们 end-to-end training + 相同 reference input，它们的 architecture 没法 leverage reference 和 topology 来 resolve coordinate-axis ambiguity。artifact 是 joint spinning（rotation 不稳定，绕 bone 轴乱转）。

2. **V2 把误差砍半到 ~10°**：reference-conditioned design 的威力。

3. **Zoo-Unseen (6.54°) 反而比 Zoo-Seen (10.73°) 和 Zoo-Rare (14.38°) 低**：这个反直觉的现象的解释很关键——Unseen split 里 motion 多是 common locomotion（walking, running），一旦 coordinate axis 被 reference pair anchor 住，rotation 就 trivial 了。Seen/Rare 里有更 exotic motion（varied behavior），所以 rotation 预测本身更难。这说明 reference conditioning 在 "axis convention 未知" 的场景下收益最大。

4. **Position error 全部都很小（2-4 cm）**：说明 pose 这块已经不是 bottleneck，rotation 才是。

5. **AngV Err 也显著降低**（0.17 vs 0.5+）：temporal consistency 大幅提升，这正是 learned temporal decoder 相比 per-frame IK 的优势。

---

## 10. V1 vs V2 对比（Table 2）

| Config | Seen Ang | Rare Ang | Unseen Ang |
|---|---|---|---|
| V1 (GT Mesh + IK) | 17.47 | 18.52 | 20.56 |
| V1 (Pred Mesh + IK) | 20.02 | 19.82 | 22.04 |
| **Ours (no mesh)** | **10.91** | **14.36** | **6.68** |

V1 even with GT mesh（理想化、推理不可用）rotation 误差 17-22°，因为 analytical IK resolve 不了 twist。V2 即使没 mesh 也比 V1-with-GT-mesh 好一倍。这证明 **end-to-end learnable P→R + reference conditioning 的收益远超 mesh intermediate 能提供的几何信息**。

---

## 11. Ablation Studies 的 Intuition Building

### 11.1 Intermediate Pose Representation（Table 5）

| Architecture | Seen | Rare | Unseen |
|---|---|---|---|
| Direct (V→R) | 9.32 | 12.71 | 23.73 |
| Latent + Aux | 9.06 | 11.85 | 23.57 |
| **Full (explicit pose)** | 10.73 | 14.38 | **6.54** |

Direct V→R 在 Seen/Rare 上 competitive，但 Unseen 崩到 23.73°。Latent + Aux（用 auxiliary pose loss 监督 latent）也不行。

**Insight**：explicit pose 是 skeleton-shared canonical representation，不同 skeleton 做 same motion 共享 similar pose pattern。Pose 把 motion content 和 skeleton-specific convention 解耦。没有 explicit pose intermediate，model 没法跨 topology generalize——latent 即使被 auxiliary supervise 也 lacks explicit skeleton-shared structure。

这是 information bottleneck 的好例子：explicit bottleneck 反而帮助 generalization，因为它强制 representation 只编码 transferable 信息。

### 11.2 Attention Mechanism（Table 6）

如前所述，GL-GMHA 最佳。**Insight**：kinematic chain locality 和 global skeleton coordination 是 complementary 而非 redundant。All-local 失去 cross-branch coordination，all-global 缺少 inductive bias，alternating 最好。

### 11.3 Model Depth（Table 8）

Depth 6 → 8 → 12：8 最佳（6.54°），12 退化（7.66°）。**Insight**：capacity 和 optimization 的 tradeoff，太深 overfits Seen/Rare，反而 hurt Unseen。

### 11.4 Cross-Attention Depth $L_{cross}$（Table 9）

$L_{cross} = 0$ 时 Unseen 崩到 23.49°——reference conditioning 是 essential 的。$L_{cross}=6$ 最佳。**Insight**：早期 layer 注入 reference anchor，后期 layer 利用已整合的 reference 信息做 reasoning，过度 conditioning 会 diminishing return。

---

## 12. Efficiency Analysis

V1 在 120-frame sequence 上：
- Feature extraction ~40s
- Mesh reconstruction ~15 min（最慢）
- Pose estimation ~20s
- IK optimization ~5 min
- Total > 20 min

V2：
- 同样的 feature extraction ~40s
- Pose + rotation 单次 forward pass ~10s
- Total < 1 min
- **~20× speedup**

来源：消除 mesh reconstruction（~15 min）和 iterative IK（~5 min）。Learned rotation decoder 是 batched computation，远快于 per-frame iterative optimization。

**关键 trade-off**：V2 的速度优势没牺牲精度——average angle error 10.6° vs V1 (GT Mesh) 18.9° vs V1 (Pred Mesh) 20.63°。这是 win-win，因为 V1 的 bottleneck 是 fundamental ill-posedness，不是计算量不够。

---

## 13. Qualitative Results 与 Cross-Skeleton Retargeting

Fig. 5: 跨 Objaverse / Zoo / in-the-wild 三个 domain 都稳定，多视角渲染显示 temporally consistent 3D motion。

Figs. 6, 7 最有意思：**single input video 同时驱动 mocap + retargeting 到多个 skeleton**，无需 skeleton-specific training。同一个 dance clip 可以 retarget 到 humanoid、狗、鸟、马等等。retargeted motion 保留 rhythm 和 semantics，同时 respect target topology。

这是 reference-conditioned design 的副产物——只要给 target skeleton 一个 reference pair，就能 drive 任意 skeleton。

---

## 14. Limitations 与未来方向

1. **Unnatural retargeting drift**：P→R decoder 隐式 learn per-skeleton motion prior，所以把 bird flapping retarget 到 dog 试图让它"飞"时，predicted rotation 会 drift 到 dog 的 typical configuration。Human skeleton 反而能 reproduce unusual motion（因为训练数据多）。解决方法是 augment training set with 不自然配置。
2. **Occlusion 没处理**：未来工作。
3. **Animal 数据集只有 ~1000 sequences**：scale up 会进一步提升。

---

## 15. 与相关 Work 的 Context

### 15.1 Parametric Human Models
SMPL [Loper et al. 2015], SMPL-X [Pavlakos et al. 2019] 绑定 fixed topology + fixed joint count。V2 处理 arbitrary topology。

SMPL: https://smpl.is.tue.mpg.de/

### 15.2 Model-Free Reconstruction
3D Fauna [Li et al. 2024], BANMo [Yang et al. 2022], MagicPony [Wu et al. 2023] recover deformable surface 但 lack explicit skeletal parameterization。V2 输出 animation-ready rotation。

BANMo: https://arxiv.org/abs/2201.01626
3D Fauna: https://embodied-ai.org/

### 15.3 Category-Specific Animal Models
SMAL [Zuffi et al. 2017] 是 animal 的 SMPL 类比，但绑 quadruped topology。V2 处理 arbitrary skeleton including non-animals。

SMAL: https://smal.is.tue.mpg.de/

### 15.4 AnyTop
AnyTop [Gat et al. 2025] 是 character animation diffusion with any topology，但它是 generation 任务，不是 motion capture from video。V2 的 GL-GMHA 借鉴了 AnyTop 的 GMHA。

### 15.5 6D Rotation Representation
[Zhou et al. 2019] 证明 6D representation 在神经网络里比 quaternion / Euler 连续性好，避免 topology discontinuity。V2 用 6D 是标准选择。
Paper: https://arxiv.org/abs/1812.07035

### 15.6 Inverse Kinematics 经典方法
CCD (Cyclic Coordinate Descent), FABRIK (Forward And Backward Reaching Inverse Kinematics) 是 analytical IK 的经典算法，per-frame 求解，没 temporal context，没 motion prior，所以 V1 用它们导致 joint spinning。

---

## 16. 我对这篇 Paper 的整体 Intuition 总结

这篇 paper 的核心 contribution 是把一个 mathematically ill-posed 的 inverse problem（P→R）通过一个 simple 但 elegant 的 conditioning signal（reference pose-rotation pair）转化为 well-posed supervised learning 问题。

更深层 intuition：**ill-posedness 通常不是一个 mathematically fundamental 的障碍，而是 conditioning 不足的表现**。当你觉得一个问题 under-determined，往往是你没给模型足够的 context 来 disambiguate。这里 bone-axis twist 看起来是 mathematical ambiguity，但只要告诉模型 "这个 skeleton 的 coordinate convention 是什么"，ambiguity 就 resolve 了。

另一个 takeaway：**end-to-end training 的 benefit 不是来自"更深的网络"，而是来自 intermediate representation 的 co-adaptation**。V1 的 pose 表示只优化 position accuracy，V2 的 pose 表示因为 gradient 来自 rotation loss，会被 reshape 成"对 rotation recovery 最有用的 pose 表示"。这种 co-adaptation 是 factorized pipeline 永远做不到的，无论 IK solver 多 smart。

第三个 takeaway：**explicit bottleneck 反而促进 generalization**。Direct V→R 看起来更 elegant（少一个 stage），但跨 skeleton generalize 失败；explicit pose intermediate 强制 representation 只 encode transferable 信息，反而 generalize 好。这是个反直觉但 recurring 的 insight——bottleneck = inductive bias。

最后一个 takeaway：**reference conditioning 是 arbitrary-skeleton 问题的关键**。fixed-skeleton 方法（SMPL 等）的 coordinate convention 是 baked-in 的，所以不需要 reference；arbitrary-skeleton 方法必须 explicit 提供 convention，否则 model 不知道在哪个 frame 下输出 rotation。Reference pose-rotation pair 是 minimal sufficient conditioning，因为任何 rigged asset 都天然有它。

潜在 future direction 我能想到的：
- 多个 reference pairs 的 weighted ensemble（paper 说 single pair 够了，但 multi-pair 可能在极端 skeleton 上更鲁棒）
- 引入 diffusion-based P→R 来 model multi-modal rotation distribution（处理 truly ambiguous cases）
- Occlusion handling via temporal smoothing + uncertainty estimation
- 加入 physics constraint（ground reaction force, momentum conservation）来 further constrain rotation
- 用 V2 的 output 作为 RL reward 来训练 character controllers in simulation
- 与 text-to-motion 的结合：用 text condition 直接 drive arbitrary skeleton 动画

Paper link: https://animotionlab.github.io/MoCapAnythingV2/
arXiv V1: https://arxiv.org/abs/2512.10881
