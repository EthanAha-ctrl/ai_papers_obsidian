---
source_pdf: Generated Reality Human-centric World Simulation using Interactive Video
  Generation with Hand and Camera Control.pdf
paper_sha256: 38427ce3488ab5ec3f24f65e29442e9643bf2c0e38309cdadb6b4e0bc6c83eea
processed_at: '2026-08-04T13:47:19-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

---

## 一句话版本

拿个VR头显戴上，你的头怎么转、手怎么动，AI就直接生成对应的虚拟世界视频，不需要任何3D建模，pure neural rendering，11帧每秒跑在远端H100上。

---

## 这帮人想干嘛

做VR/AR最烦的事情是什么？**搞3D资产**。你想让用户在虚拟世界里按个按钮，你得建模按钮、写物理引擎、搞碰撞检测、调shader……一个场景搞下来几个月就没了。

这帮Stanford的人就想：**Sora都能生成视频了，那我为啥不能拿video diffusion model直接当"3D引擎"用？** 用户头转一下、手动一下，model直接render出对应的画面，zero-shot，啥都不用建。

这个概念他们叫 **Generated Reality**——生成的现实。不是虚拟现实，是生成现实。

---

## 问题在哪

现有的video world model（Oasis、Genie 2那些）你给它什么控制信号？**键盘**。按W往前走，按A往左转。

但你想想，你想在VR里"拧开一个罐子"，你怎么用键盘告诉AI？你没法告诉它"左手握住罐子盖、右手逆时针旋转30度、同时手腕有个细微的pronation"。键盘这种coarse signal对**精细手部操作**完全没用。

所以核心问题是：**怎么把用户的head tracking和hand tracking数据，塞进video diffusion model里，让它能响应精细的身体动作？**

---

## 最核心的技术贡献：手怎么表示

这是整篇paper花最多篇幅做ablation的地方。问题是——你的手有20多个关节，每个关节有旋转角度，你怎么把这个信息喂给一个video生成模型？

他们试了一堆方案：

### 方案一：把手骨架render成2D图片
就像ControlNet那套，把手的skeleton画成一张2D图，跟目标视频对齐。
- 好处：天然跟image space对齐，模型好理解
- 坏处：**2D没深度信息**。你的手指头前后重叠的时候，2D分不清哪个在前哪个在后

### 方案二：直接用3D参数
用UmeTrack这个hand model，输出6-DoF的wrist pose + 20个joint angle。一串数字直接喂进去。
- 好处：有完整的3D信息，depth清清楚楚
- 坏处：**维度太高，小数据上学不稳**。HOT3D才5824个5秒clip，你让模型从一串高维数字学到image space的mapping，太难了

### 方案三（他们的答案）：两个一起用
2D骨架图走ControlNet-style的通道，3D参数走token addition。**2D告诉模型"手大概在这"，3D告诉模型"指节具体怎么弯"**。

这个思路其实很直觉：coarse-to-fine。2D是spatial prior，3D是articulation refinement。两个signal互补，单独用都不够好，合在一起就work了。

实验数据说话：pure 3D参数的MPJPE（手部关节误差）大概17-18mm，2D骨架能到12mm，hybrid能到12.2mm，而且qualitative上hybrid在手指被遮挡的场景明显更稳。

---

## 怎么把信号塞进DiT

他们试了四种conditioning injection方式：

1. **Token concatenation**（PlayerOne用的）：把hand embedding拼到input channel上。改了模型input shape，比较invasive
2. **Token addition**（ReCamMaster用的）：hand embedding直接加到patch token上。最干净，不改shape
3. **AdaLN**（PEVA用的）：hand参数生成scale和shift，调制DiT每个block的activation。问题是hand信息被压成全局modulation，丢了spatial信息
4. **Cross-attention**：hand embedding当key/value。理论上最flexible，但小数据上学不稳

结果：**token addition最好**。原因我觉得是——addition把signal broadcast到所有patch，每个patch都能access到，最spatial-agnostic，不需要模型学额外的alignment mapping。

最终公式长这样：

$$x = \text{patchify}([z_r, z_c]_{\text{channel}}) + \mathcal{E}_{\text{conv}}(H) + \mathcal{E}_{\text{cam}}(P)$$

视频latent和skeleton latent沿channel拼，hand参数和camera参数各经过自己的encoder，然后element-wise加到patch token上。三个signal在latent space汇合。

---

## Camera怎么处理

VR头显给你的是6-DoF camera pose（旋转矩阵+平移向量）。他们把这个转成**Plücker embedding**——每条pixel ray用6维向量表示（origin 3维 + direction 3维），变成一个 $b \times f \times 6 \times h \times w$ 的tensor，再经过encoder压成patch token shape，跟hand signal一样用addition注入。

Plücker embedding的好处是它天然encode了camera的geometry——每条ray从哪来、往哪去，一目了然。比直接塞rotation matrix好多了。

---

## 训练的坑：两个encoder一起训会炸

他们发现camera encoder和hand encoder从scratch一起训不稳定。原因很直觉：

**你看到一个pixel在动，到底是camera在动还是手在动？** 两个signal都通过token addition加到同一个地方，模型分不清谁是谁。

解决方案是**iterative training**：
1. 先单独训camera encoder（用FUN model预训练权重init）
2. 再单独训hand encoder
3. 最后joint fine-tune

DiT那边也用了continual training：hybrid model用skeleton-only的LoRA weights做init，joint model用hybrid的LoRA weights做init。相当于curriculum learning，从simple modality逐步加complexity。

---

## 怎么变成实时系统

bidirectional video diffusion model（比如Wan 2.2）是看全帧才能denoise的，你没法online interactive。所以他们用了**Self-Forcing**蒸馏——把14B的bidirectional teacher变成5B的causal student。

Self-Forcing的核心idea：传统teacher forcing训练时模型看GT context，推理时看自己生成的context，distribution mismatch导致drift。Self-Forcing让模型训练时就"自己rollout自己"，next chunk用自己上一个chunk的output当context，缩小train-test gap。

最终student model按12帧一个chunk自回归生成，用最近几帧当context + 用户最新的tracking data当conditioning。

跑在远端H100上：**11 FPS，1.4秒延迟**。conditioning本身只占0.002秒，bottleneck完全是chunk generation + VAE decode。

---

## User Study：真的好用吗

11个人，3个task："按绿色按钮"、"开罐子"、"转方向盘"，8秒内完成。

两组对比：
- **Ours**：hand + head conditioning
- **Baseline**：head only（用text prompt告诉模型该干嘛）

结果：
- Task accuracy：baseline **3%**，ours **71.2%**
- Perceived control（1-7分）：baseline **1.74**，ours **4.21**

3% vs 71%这个差距说明——**text prompt对精细手部操作完全没用**，你必须给model hand tracking signal才行。但4.21/7也说明1.4秒延迟和generation quality还是让用户觉得不够"掌控"。

---

## Limitations他们很诚实

- 1.4秒延迟，VR要求<20ms，差了两个数量级
- 不支持stereo rendering（VR必备）
- 自回归几秒后drift严重，画面质量掉
- DMD蒸馏的通病：mode-seeking + oversaturation
- 三体以上交互（hand + object + object）搞不定

但他们说这些都是**engineering问题不是fundamental问题**，video diffusion community正在解决。

---

## 我觉得最interesting的几个点

**1. Hybrid representation的intuition**
2D给spatial grounding，3D给articulation detail，两个互补。这个思路其实很多地方都适用——coarse signal建立大致对应关系，fine signal补充细节。就像LLM里先做retrieval再做reasoning一样。

**2. Token addition为什么work最好**
因为它最"傻瓜"——直接把signal加到每个patch上，不需要模型学额外的alignment。小数据 + 高维signal的场景下，越简单的injection方式越好。Cross-attention和AdaLN都太"聪明"了，需要更多data才能训稳。

**3. Iterative training解决multi-modal conflict**
两个modality都走同一个addition operation会打架。先各自学再joint fine-tune，这个curriculum思路在任何multi-modal conditioning场景都值得考虑。

**4. Generated Reality这个vision**
不需要3D assets，不需要物理引擎，不需要shader——video model就是引擎。用户的motion就是controller。这个跟LeCun的JEPA、Sora的"world simulator"叙事是同一个大方向，但在XR这个具体场景下给出了第一个end-to-end的proof of concept。

---

## 相关链接

- Project page: https://codeysun.github.io/generatedreality/
- Wan model: https://github.com/Wan-Video/Wan2.1
- HOT3D dataset: https://hot3d.github.io/
- UmeTrack: https://github.com/facebookresearch/UmeTrack
- Self-Forcing: https://selfforcing.github.io/
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Oasis: https://oasis-model.github.io/
- PlayerOne: https://playeroneofworld.github.io/
- PEVA: https://dannytran13.github.io/PEVA/
- CameraCtrl: https://hehao13.github.io/projects/CameraCtrl/
- ReCamMaster: https://jianhongbai.github.io/ReCamMaster/
- AC3D: https://sherwinbahmani.github.io/ac3d/
- GigaHands: https://github.com/RaoFu42/GigaHands

---

**Bottom line**: 这篇paper不是在搞什么颠覆性的新architecture，是在existing video diffusion model基础上，系统地解决"怎么把人的motion tracking信号塞进去"这个问题。第一个证明motion-conditioned video generation在XR场景下可行，虽然离production还远，但direction很clear。

---

# Generated Reality 深度解读

**一句话总结**: 这篇paper系统性地研究了如何把用户的 headpose 和 joint-level hand poses 作为conditioning signal注入 video diffusion transformer，然后把 bidirectional teacher 蒸馏成 causal autoregressive student model，最终在 Meta Quest 3 上跑出 11 FPS 的 interactive egocentric world simulator。

**Project page**: https://codeysun.github.io/generatedreality/
**Wan model**: https://github.com/Wan-Video/Wan2.1
**HOT3D dataset**: https://hot3d.github.io/
**UmeTrack**: https://github.com/facebookresearch/UmeTrack
**Self-forcing**: https://selfforcing.github.io/

---

## 1. Motivation: 为什么这件事现在才做得动

Current video world models (Oasis, Genie 2, GameFactory, Minecraft world model等) 都接受 coarse control signal，比如 keyboard input 或者 text prompt。对于 embodied interaction 来说太粗糙了——你没法用键盘"按按钮"、"开罐子"、"拧方向盘"。

作者认为 next-generation world model 应该 accept **rich stream of tracked user data**: head/gaze direction, body pose, foot placement, **hand and finger articulation**, full-body movement。这篇paper专注在 head + hand 这两个 modality 上。

**核心研究问题**:
1. 如何 represent hand pose 进 video diffusion model?
2. 如何 inject conditioning signal 进 DiT backbone?
3. 如何同时 control camera 和 hand?
4. 如何让 bidirectional model 变成 real-time interactive system?

---

## 2. Pipeline 架构解析

参考 Figure 3 的 pipeline diagram，整个系统是 closed-loop:

```
[Meta Quest 3 tracking] 
   ├── head pose (6-DoF: rotation R∈R^{3×3} + translation t∈R^3)
   └── hand poses (UmeTrack: 6-DoF wrist + 20 joint angles per hand)
            │
            ▼
   ┌─────────────────────────────────────────┐
   │ Conditioning pipeline                   │
   │ 1. Head → Plücker embeddings P∈R^{b×f×6×h×w}  │
   │ 2. Hand → 2D skeleton video (ControlNet-style) │
   │         + 3D HPP (hand pose parameters)        │
   └─────────────────────────────────────────┘
            │
            ▼
   [DiT 5B student (causal, distilled)]
            │
            ▼
   [3D VAE Decoder D] → generated video chunks (12 frames each)
            │
            ▼
   [Stream back to Quest 3] (1.4s latency, 11 FPS)
```

**Key insight**: 这个 closed-loop 的设计让 user 真的能用 motion 来 drive 生成内容，generated reality 就是把 video world model 当成一个 "implicit 3D engine" 用，省掉所有手工 3D asset 建模。

---

## 3. Preliminaries: Wan family + Rectified Flow

Model 基于 **Wan2.2** (latent video diffusion transformer)，包含一个 3D VAE $(\mathcal{E}, \mathcal{D})$ 和一个 transformer-based diffusion model $\mathcal{F}_\Theta$。

### Forward process (Eq. 1):

$$z_t = (1-t)z_0 + t\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $z_0 = \mathcal{E}(V_0)$: clean video latent (从 VAE encoder 出来的)
- $t \in [0,1]$: timestep，$t=0$ 是 clean data，$t=1$ 是 pure noise
- $\epsilon$: standard Gaussian noise
- 这是 **rectified flow** 的 formulation (跟 Stable Diffusion 3 一样)，相比 DDPM 的话 trajectory 是 straight line，采样效率高

### Training objective (Eq. 2):

$$\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t, z_0, \epsilon} \left[ \| v_\Theta(z_t, t) - u_t(z_0|\epsilon) \|_2^2 \right]$$

- $v_\Theta(z_t, t)$: 网络预测的 velocity field
- $u_t(z_0|\epsilon)$: 从 forward process 解析得到的 target velocity，对于 rectified flow 来说 $u_t = \epsilon - z_0$
- 这是 **conditional flow matching** (Lipman et al. 2023)

Inference 时候通过 ODE solver 积分 $v$ 把 noise 推回 data。

### Mixture-of-Experts (MoE) architecture:
Wan2.2 14B 有两个 DiT experts:
- **High-noise expert**: 处理 $t$ 较大的步（global structure）
- **Low-noise expert**: 处理 $t$ 较小的步（细节 refinement）

这个对后续 training strategy 有重要影响（见 §A.1 continual training）。

---

## 4. Hand Pose Representation: 2D vs 3D vs Hybrid

这是这篇paper最核心的 ablation 贡献。作者对比了几种 hand pose representation:

### Option A: 2D Skeleton video (ControlNet-style)
直接把 hand skeleton render 成 2D image sequence 当 conditioning video。
- **优点**: spatially aligned with image space，ControlNet 已经成熟
- **缺点**: depth ambiguity (2D 没法区分手指前后), self-occlusion (重叠的 joint 模糊)

### Option B: 3D Hand Pose Parameters (HPP)
使用 **UmeTrack** hand model: 6-DoF wrist pose + 20 joint angles per hand。
通过 forward kinematics 可以解析地得到所有 joint 的 3D 位置。
- **优点**: metric precision in depth, 完整 articulation 信息
- **缺点**: 高维度，跟 image space 没有 spatial alignment

### Option C (本文): Hybrid 2D-3D
两者结合，2D 提供 spatial grounding，3D 解决 depth/occlusion。

**Intuition**: 2D skeleton 像 coarse prior，告诉模型"手大概在这个位置"；3D HPP 像 refinement，告诉模型"具体每个指节怎么弯"。两个 signal 互补，缺一不可。

---

## 5. Conditioning Strategies 详细公式解析

作者对比了 4 种 HPP 注入方式:

### (1) Token Concatenation (PlayerOne 方式)

$$x = \mathrm{patchify}([z_r, \mathcal{E}_{\mathrm{conv}}(H)]_{\mathrm{channel-dim}})$$

- $z_r \in \mathbb{R}^{b \times f \times c \times h \times w}$: raw video latent
- $H \in \mathbb{R}^{b \times f \times d}$: HPP，$d$ 是 HPP 维度
- $\mathcal{E}_{\mathrm{conv}}$: lightweight 1D conv encoder
- $[\cdot]_{\mathrm{channel-dim}}$: 沿 channel 拼接
- **缺点**: 改变了输入 channel 数，需要修改 patchify layer

### (2) Token Addition (ReCamMaster 方式)

$$x = \mathrm{patchify}(z_r) + \mathcal{E}_{\mathrm{conv}}(H)$$

- HPP embedding 跟 patch tokens element-wise 相加
- **优点**: 不改 input shape，最干净
- **结果**: 在 HPP-only 实验里这个最好

### (3) AdaLN (PEVA 方式)

$$x = \alpha(H) \odot v_r + \beta(H)$$

- $\alpha(H), \beta(H)$: 从 HPP 学到的 scale/shift
- $\odot$: Hadamard product
- $v_r$: DiT block 内部的 activation
- **结果**: 在 HOT3D 这种小数据集上效果差，因为 high-dim HPP 学不稳

### (4) Cross-Attention

$$x^{(l+1)} = x^{(l)} + \mathrm{CrossAttn}(x^{(l)}, \mathcal{E}_{\mathrm{conv}}(H))$$

- HPP embedding 当 key/value，visual feature 当 query
- **结果**: 也比较差，需要更多 data 才能训稳

### Hybrid 2D-3D (本文最终方法, Eq. 7):

$$x = \mathrm{patchify}([z_r, z_c]_{\mathrm{channel-dim}}) + \mathcal{E}_{\mathrm{conv}}(H)$$

- $z_r$: raw video latent
- $z_c$: skeleton video latent (跟 $z_r$ 用同一个 VAE encoder $\mathcal{E}$)
- 沿 channel 拼接两个 latent (类似 ControlNet 的 input)
- HPP 通过 token addition 注入
- **关键设计**: 2D skeleton 走 ControlNet-style (spatial alignment)，3D HPP 走 token addition (depth/articulation)

### Joint Hand-Camera Control (Eq. 8):

$$x = \mathrm{patchify}([z_r, z_c]_{\mathrm{channel-dim}}) + \mathcal{E}_{\mathrm{conv}}(H) + \mathcal{E}_{\mathrm{cam}}(P)$$

- $P \in \mathbb{R}^{b \times f \times 6 \times h \times w}$: **Plücker embedding**，6 维 (origin + direction) per ray per pixel
- 三个 component 在 latent space element-wise 相加
- **关键 insight**: camera 和 hand 两个 modality 共用 token addition，可能会冲突——所以用了 **iterative encoder training** (先各自 train，再 joint fine-tune)

---

## 6. Ablation Study 解析 (Table 1)

| Method | PSNR↑ | LPIPS↓ | SSIM↑ | FVD↓ | MPJPE↓ | MPVPE↓ | L2Err↓ |
|---|---|---|---|---|---|---|---|
| No Cond. (Wan 2.2 baseline) | 14.59 | 0.4872 | 0.4855 | 601.55 | 17.86 | 12.29 | 67.50 |
| TokenConcat (PlayerOne) | 15.09 | 0.4633 | 0.4983 | 560.34 | 18.02 | 12.34 | 65.43 |
| AdaLN (PEVA) | 15.02 | 0.4591 | 0.4906 | 677.26 | 18.49 | 12.53 | 65.97 |
| CrossAttention | 14.71 | 0.4686 | 0.4840 | 662.22 | 17.56 | 12.04 | 63.23 |
| TokenAddition (ReCamMaster) | 15.19 | 0.4520 | 0.4975 | 601.15 | 17.84 | 12.14 | 56.66 |
| Binary Mask (InterDyn) | 16.58 | 0.3947 | 0.5533 | 356.11 | 12.83 | 9.56 | 35.64 |
| Skeleton Video (ControlNet*) | 16.89 | 0.3837 | 0.5601 | 389.26 | 12.38 | 9.25 | 11.72 |
| **Hybrid (Ours)** | **16.85** | **0.3874** | **0.5574** | **383.69** | **12.23** | **9.10** | **11.50** |

### 几个关键观察:

1. **Pure HPP conditioning 都没比 baseline 好多少**: TokenConcat/AdaLN/CrossAttn/TokenAdd 的 MPJPE 都在 17.5-18.5 之间，跟 No Cond. (17.86) 差不多。这说明 **3D 参数进 DiT 很难**，high-dim + 小数据集 = 学不出 mapping。

2. **Video conditioning (2D) 显著更好**: Binary mask 和 Skeleton video 把 MPJPE 从 17.86 降到 12.38，因为 2D 跟 image space 对齐得天然好。

3. **Hybrid 在所有 hand metrics 上都是 best**: MPJPE 12.23, MPVPE 9.10, L2Err 11.50。即使相比 Skeleton video 提升幅度小 (因为 HOT3D 手部动作简单)，但 qualitative 上 hybrid 在 self-occlusion 场景更稳。

4. **Lower bound**: 用 WiLoR 在 GT 上跑同样的 evaluation，得到 MPJPE=9.42, MPVPE=7.74, L2Err=9.08。Hybrid 方法 (12.23/9.10/11.50) 已经接近这个 lower bound。

**Intuition**: HPP 提供的是 "articulation prior"，2D skeleton 提供 "spatial prior"。HPP alone 在小数据上学不出 latent space 到 image 的 mapping；2D alone 在 depth/occlusion 上有歧义。Hybrid 让 2D 先建立 spatial grounding，3D 再 refine articulation，相当于 coarse-to-fine。

---

## 7. Joint Hand-Camera Control (Table 2)

| Method | PSNR | LPIPS | FVD | MPJPE | MPVPE | L2Err | TransErr | RotErr |
|---|---|---|---|---|---|---|---|---|
| CameraCtrl (camera only) | 18.58 | 0.2943 | 558.94 | 18.37 | 12.72 | 50.33 | 0.23m | 2.77° |
| HandCtrl (hand only) | 16.85 | 0.3874 | 383.69 | 12.23 | 9.10 | 11.50 | 2.27m | 13.40° |
| **JointCtrl (Ours)** | **18.60** | **0.2800** | 396.93 | 12.81 | 9.66 | 13.42 | 0.25m | 2.79° |

**关键 insight**:
- Camera-only: TransErr=0.23m, RotErr=2.77° 都很好，但是 hand MPJPE=18.37 完全不行
- Hand-only: hand 准了 (12.23)，但 camera RotErr=13.40° 烂掉
- Joint: **两边都接近各自 specialized model 的最佳** (MPJPE 12.81 vs 12.23, TransErr 0.25 vs 0.23)

这是一个很 elegant 的 result——joint training 几乎没有 modality 之间的 trade-off。作者归功于 **iterative encoder training** (先独立训再 joint fine-tune) 解决了 token addition 共享的 conflict。

---

## 8. Distillation: Bidirectional → Causal Autoregressive

这是系统层面的关键步骤。

**问题**: Bidirectional DiT 需要 access full sequence 才能 denoise (双向 attention)，没法 online interactive。

**方案**: 使用 **Self-Forcing** (Huang et al. 2025) 把 bidirectional teacher 蒸馏成 causal student。

参考: https://selfforcing.github.io/

**Self-Forcing 核心 idea** (build intuition):
- 传统 train/test gap: 训练时 model 看 GT context，inference 时看自己生成的 context，distribution mismatch 导致 drift
- Self-Forcing 让 model 训练时就"自己 rollout 自己"，next chunk 用自己上一个 chunk 的 output 当 context
- 用 teacher forcing warm-up，逐渐 transition 到 self-forcing

**架构变化**:
- Teacher: Wan2.2 14B (bidirectional MoE)
- Student: Wan2.2 5B (causal, single expert)
- Chunk size: 12 frames
- Autoregressive: 用 last few generated frames 当 context + 用户最新 tracked conditioning

**Performance**: 
- 11 FPS on H100
- 1.4s latency (bottleneck 是 12-frame chunk generation + decode)
- Conditioning overhead 只占 0.002s

---

## 9. VR System Integration

```
[Meta Quest 3] 
   │ (1) tracks head/hand poses
   │ (2) streams to remote server
   ▼
[H100 Server]
   │ (1) reads from circular frame buffer (latest tracked data)
   │ (2) runs distilled student model
   │ (3) generates 12-frame chunk
   │ (4) decodes via 3D VAE
   ▼
[Stream back to Quest 3] 
   │ (Unity 渲染到 headset)
```

**Circular frame buffer** 是一个重要设计: 用户动作持续发生，但 model 要按 12-frame chunk 处理。buffer 让 model 总是拿到最新的 tracking data，避免用户动作和生成内容 lag 太多。

**Latency breakdown**:
- ~1.4s end-to-end
- Conditioning injection: 0.002s (可忽略)
- 主要 bottleneck: chunk generation + decode
- 这是 remote server 上的数字，local 估计会好很多

---

## 10. User Study 设计与分析

**Setup**:
- 11 subjects (4F/7M, age 22-30)
- 3 tasks: "push the green button", "open the jar", "turn the steering wheel"
- 8s per task
- 2 conditions: 
  - **Ours**: hand + head conditioned
  - **Baseline**: head only (text prompt 告诉 model 要做什么)
- 每人每 task 跑 4 次 (2 ours + 2 baseline), randomized
- 2 practice runs before recording

**Critical design choice**: 让 user 先 align 手跟 input image 的 hand pose，对齐后 disable overlay，user 只看到 generated environment + generated hands。这隔离了"控制感"本身。

### Results (Figure 7):

**Task accuracy**:
- Baseline: **3.0%** (基本不可能用 text 控制 fine-grained hand-object interaction)
- Ours: **71.2%** (hand conditioning 让 task 真正可执行)

**Perceived control (7-point Likert)**:
- Baseline: 1.74
- Ours: 4.21

**Intuition**: Task accuracy 的差距 (3% vs 71%) 比 perceived control 差距 (1.74 vs 4.21) 大得多。说明:
1. Hand conditioning 是 task 完成的必要条件 (text 完全做不到)
2. 但 4.21/7 还不是"完全 control"，反映 1.4s latency 和 model quality 还是 barrier

---

## 11. Iterative Encoder Training: 解决 Joint Conditioning 不稳定

作者发现 joint train camera encoder + HPP encoder from scratch 不稳，归因:
1. 两者都用 token addition 注入 (同一个 operation)
2. Hand motion 和 camera motion 的 ambiguity (谁导致 pixel 变化?)

**Solution**:
1. **Stage 1**: Camera encoder 用 FUN model [35] 预训练权重 init, 独立 train
2. **Stage 1'**: HPP encoder 独立 train
3. **Stage 2**: Joint fine-tune 两个 encoder

**Continual training on DiT side**:
- Hybrid training: 用 skeleton-video LoRA weights init, 再加 HPP
- Joint training: 用 hybrid LoRA weights init, 再加 camera

这是一个 curriculum learning 思路: 从 simpler modality (2D skeleton) → richer modality (3D HPP) → cross-modal (camera)。

---

## 12. Limitations & Future Work

作者很诚实地列了几个 fundamental issues:

1. **Resolution & quality**: 远不如现代 VR system
2. **Latency**: 1.4s 远超 immersive XR 要求的 <20ms
3. **Stereo**: 不支持 stereo rendering (VR 必备)
4. **Long rollout drift**: 自回归 model 几秒后质量明显掉
5. **DMD distillation artifacts**: mode-seeking + oversaturation over long horizons
6. **Long-range hand-object-object dependencies**: 三体以上交互搞不定

**Future direction**:
- Retinal resolution stereo
- Sub-20ms latency
- Local wearable compute (不用 remote server)
- 这些都跟 video diffusion 整个 community 的研究方向对齐

---

## 13. 与 Related Work 的对比

| Work | Conditioning | Granularity | Causal? | Real-time? | XR? |
|---|---|---|---|---|---|
| Oasis [8] | Keyboard | Coarse | ✓ | ✓ | ✗ |
| Genie 2 [26] | Keyboard | Coarse | ✓ | ✓ | ✗ |
| Minecraft World [13] | Keyboard | Coarse | ✓ | ✓ | ✗ |
| PlayerOne [33] | Body pose + rotation-only camera | Coarse hand | ✓ | ✓ | ✗ |
| PEVA [4] | Body pose via AdaLN | Coarse hand | ✓ | ✓ | ✗ |
| InterDyn [1] | Binary mask | Mask-only | ✗ | ✗ | ✗ |
| CameraCtrl [16] | Plücker | N/A | ✗ | ✗ | ✗ |
| ReCamMaster [3] | Camera extrinsic | N/A | ✗ | ✗ | ✗ |
| AC3D [2] | ControlNet-style camera | N/A | ✗ | ✗ | ✗ |
| **Generated Reality (Ours)** | **Plücker + UmeTrack HPP + 2D skeleton** | **Joint-level** | ✓ | ✓ (11 FPS) | ✓ (Quest 3) |

这是 **first systematic study of joint-level hand conditioning in video diffusion**。

---

## 14. 我的 Intuition Takeaways

### A. 为什么 hybrid 2D-3D 比 pure 2D 或 pure 3D 都好?

视频生成本质是从 latent manifold mapping 到 image space。Hand pose 有两层信息:
- **Spatial**: 手在 image 哪里 (2D skeleton 天然对齐)
- **Articulation**: 指节怎么弯 (3D HPP 天然表达)

2D skeleton 把 spatial prior "硬连"到 image coordinates，DiT 不需要学习这个 mapping。3D HPP 把 articulation prior 编码成低维向量，通过 token addition 注入相当于给 DiT 一个 "articulation hint"。两者通过加法结合：spatial 部分由 ControlNet-style 通道接手，articulation 部分由 token addition 微调。

### B. 为什么 token addition > cross-attention > AdaLN?

我推测原因:
- **Token addition**: 信号 broadcast 到所有 patch，每个 patch 都能 access，最 spatial-agnostic
- **Cross-attention**: 需要 model 学 query-key mapping，高维 HPP 在小数据上学不稳
- **AdaLN**: 通过 scale-shift 调制每个 block 的 activation，但 HPP 信息被 "压缩" 成全局 modulation，丢了 spatial specificity

### C. 为什么 iterative training 必要?

Camera 和 hand 两个 modality 都通过 token addition 加到同一个 patch token 上。如果 from scratch joint train:
- Camera signal 抢占 hand signal 的 representation capacity (camera 影响所有 patch，hand 只影响 hand 区域)
- Ambiguity: pixel change 既可能来自 camera motion 也可能来自 hand motion

Iterative training 让 encoder 先各自学到 modality-specific representation，再 fine-tune 让它们 complement 而不 conflict。这是 multi-modal conditioning 的一个 general lesson。

### D. 跟 LLM world model 的关系

这篇 paper 让我想到 Yann LeCun 的 JEPA 和 Sora 的"simulator" narrative。Generated reality 是这个 vision 在 XR 上的具体实现:
- 不需要 explicit 3D assets
- 用 video model 当 implicit physics + geometry engine
- 用户 motion 直接 drive generation

跟 Genie 2 / Oasis / Minecraft world model 比的话，那些用 keyboard input 当 action space，这 paper 用 motion tracking 当 action space，更 embodied。

### E. Latency 是不是 fatal?

1.4s latency 在 modern XR 标准下是 unusable (motion-to-photon 要求 <20ms)。但作者也指出:
- Bottleneck 是 chunk generation + decode，不是 conditioning
- 用 local GPU + 更好的 distillation (consistency model, DMD2) 可以大幅改善
- 这个 paper 是 proof-of-concept，展示 motion-conditioned video generation 的 feasibility

### F. 关于 self-forcing 的选择

Self-Forcing 是 2025 年初的工作，专门解决 autoregressive video diffusion 的 train-test gap。相比传统 teacher forcing，self-forcing 让 model 在训练时就处理自己的 prediction error，减少 distribution shift。这个 paper 用 self-forcing 把 bidirectional Wan2.2 蒸馏成 causal student，是很合理的选择。

备选方案可以是:
- **Consistency distillation** (OpenAI 的 consistency model): 可能更快但 mode collapse 风险
- **DMD2** (Diffusion Distillation one-step): 已经在 SDXL 上 work，可能也适用 video
- **Flow matching distillation**: 跟 Wan 的 rectified flow formulation 天然对齐

---

## 15. 可延伸的相关方向

如果你想 deep dive, 推荐 follow-up 这些 work:

1. **Hand pose datasets**:
   - HOT3D: https://hot3d.github.io/
   - GigaHands: https://github.com/RaoFu42/GigaHands
   - UmeTrack: https://github.com/facebookresearch/UmeTrack

2. **Camera control in video diffusion**:
   - CameraCtrl: https://hehao13.github.io/projects/CameraCtrl/
   - CameraCtrl II: https://cameractrl2.github.io/
   - ReCamMaster: https://jianhongbai.github.io/ReCamMaster/
   - AC3D: https://sherwinbahmani.github.io/ac3d/
   - MotionCtrl: https://wangzhouxia.github.io/MotionCtrl/

3. **Autoregressive video models**:
   - Self-Forcing: https://selfforcing.github.io/
   - Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
   - Oasis: https://oasis-model.github.io/
   - Cosmos: https://www.nvidia.com/en-us/ai/cosmos/

4. **Egocentric video generation**:
   - PlayerOne: https://playeroneofworld.github.io/
   - PEVA: https://dannytran13.github.io/PEVA/
   - InterDyn: https://tivixon.github.io/interdyn/

5. **Body pose conditioning**:
   - PEVA (LeCun group): https://dannytran13.github.io/PEVA/
   - WorldMem: https://worldmem.github.io/

6. **VR/AR tracking**:
   - Meta Quest 3 hand tracking: https://developer.oculus.com/documentation/unity/quest-hand-tracking/
   - OpenXR hand tracking: https://www.khronos.org/openxr/

---

## 16. 总结

Generated Reality 这篇 paper 的核心 contribution 是在 **motion-conditioned video world model** 这个方向上迈出了第一步:
1. **Systematic ablation** 找出 hybrid 2D-3D 是 best hand conditioning strategy
2. **Joint conditioning** 解决 camera + hand 共训稳定性问题
3. **Distillation** 把 bidirectional teacher 变成 causal real-time student
4. **User study** 证明 hand conditioning 显著改善 task performance 和 perceived control

虽然 1.4s latency、low resolution、stereo 缺失让它暂时离 production XR 还远，但这个 paper 把 vision 清晰地铺出来了: **video world model + motion tracking = generative XR without 3D assets**。后续工作可以沿这个方向继续优化 distillation、resolution、stereo、latency，最终逼近 modern VR system 的指标。

核心 takeaway 给 build intuition 的人:
- **Coarse-to-fine conditioning** 优于单一 representation (2D 给 spatial, 3D 给 articulation)
- **Token addition** 是 conditioning injection 的稳健选择，特别在 small data + high-dim signal 场景
- **Iterative/curriculum training** 解决多模态 conditioning conflict 的通用范式
- **Self-Forcing distillation** 是 bidirectional-to-causal 的 effective recipe
- **Closed-loop user-in-the-loop evaluation** 比 offline metric 更能反映真实 system quality
