---
source_pdf: Causally Debiased Latent Action Model for.pdf
paper_sha256: 877c8bd96c03fd4e9ac64b58ccb1e709f51b68a2204a1802fd61fdec76a65b4e
processed_at: '2026-08-18T03:11:06-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CD-LAM 用人话讲

## 0. 一句话总结

这篇 paper 说了一件事：**你用 reconstruction 训练 latent action model，得到的 latent action 里面混了一堆跟 action 没关系的东西（背景、相机抖动、场景上下文），robot 拿这种 latent 去用，根本控制不了。用三个 loss 修一下，12 倍效率提升。**

就这么简单。剩下的全是细节和证据。

参考: https://arxiv.org/abs/2602.06949 (DreamDojo baseline)

---

## 1. 故事背景: 为什么会有这个问题

你想造一个 robot world model。给它当前画面 $o_t$ 和一个 action $u_t$（比如"机械臂往右移 5cm"），它预测未来几帧 $o_{t+1:t+H}$ 长啥样。这就是 action-conditioned world model (ACWM):

$$p_\theta(o_{t+1:t+H} \mid o_{\leq t}, u_{t:t+H-1})$$

变量解释:
- $o_t$: time $t$ 的 RGB frame
- $u_t$: recorded executable robot action（比如 end-effector 的相对位移）
- $o_{\leq t}$: 过去所有 observation（context）
- $u_{t:t+H-1}$: 从 $t$ 到 $t+H-1$ 的 action 序列
- $H$: 预测 horizon

这玩意儿有用吗？有用。你可以用它做 planning（试几个 candidate action，看哪个未来最好）、policy evaluation（不用真跑 robot，先在 world model 里 evaluate）、data augmentation（生成假 trajectory 训 policy）。

**问题**: 训 ACWM 需要海量 (video, action) 配对数据。真实 robot action data 贵到飞起。AgiBot World（https://arxiv.org/abs/2503.06669）、Open X-Embodiment（https://arxiv.org/abs/2310.08864）、DROID（https://arxiv.org/abs/2403.12945）这些 dataset 已经很大了，但跟 web 上的 video 比就是九牛一毛。

而 human egocentric video 海量、好采集、物理交互丰富，但**没有 action annotation**。Ego4D（https://arxiv.org/abs/2110.15270）3000 小时，EgoDex（https://arxiv.org/abs/2505.11709）更多，但你不知道人手当时是怎么动的精确数据。

**Latent Action Model (LAM) 来救场**。LAM 的想法: 既然没 action label，我就从相邻两帧 $o_t, o_{t+1}$ **反推**一个 latent action $z_t$ 出来:

$$z_t \sim q_\phi(z \mid o_t, o_{t+1}), \quad \hat{o}_{t+1} = D_\psi(o_t, z_t)$$

变量:
- $z_t \in \mathbb{R}^d$: latent action vector
- $q_\phi$: encoder（参数 $\phi$）
- $D_\psi$: decoder（参数 $\psi$）
- $\mu_\phi(o_t, o_{t+1})$: posterior mean，需要 deterministic latent 时用这个

训练 loss:
$$\mathcal{L}_{\mathrm{LAM}} = \mathbb{E}[\ell(D_\psi(o_t, z_t), o_{t+1})] + \mathrm{Reg}(z_t)$$

$\ell$ 是 reconstruction loss，$\mathrm{Reg}$ 是 bottleneck（比如 KL）。

这玩意儿火了一阵: Genie（https://arxiv.org/abs/2402.15391）、LAPO（https://arxiv.org/abs/2312.10812）、LAPA（https://arxiv.org/abs/2410.11758）、AdaWorld（https://arxiv.org/abs/2505.18834）、Moto（https://arxiv.org/abs/2410.23214）、IGOR（https://arxiv.org/abs/2411.00785），DreamDojo（https://arxiv.org/abs/2602.06949）是其中应用到 robot 的代表。

整体 pipeline 长这样:

```
大量 human video (无 action label)
        ↓ 用 LAM 提取 z_t
        ↓ 用 z_t 训 ACWM (Stage 2)
少量 robot video (有 action label)
        ↓ 学一个 bridge g_η(u_t) → z_t (Stage 3)
        ↓
能 follow robot action 的 ACWM
```

听起来很美。**但实际上 robot action following 一塌糊涂**。这就是这篇 paper 的起点。

---

## 2. Bug 是什么: reconstruction 是个懒学生

### 2.1 用人话说 confounding

你训 LAM 的时候，objective 就一句话: "给我个 $z_t$，让我能 reconstruct $o_{t+1}$"。

模型心想: OK，那我要预测下一帧，能用啥信息都用啥。action 相关的信息有用，但**背景的延续性也有用，相机抖动也有用，场景上下文也有用**——因为下一帧大部分 pixel 跟当前帧差不多（背景不动），我只要知道背景长啥样，就能瞎预测对一大半。

所以 $z_t$ 实际上学到的是:

$$z_t \approx f(A_t, C_t, V_t)$$

变量:
- $A_t$: embodied action effects（我们**想要**的）
- $C_t$: scene context（场景上下文，**不想要**）
- $V_t$: source-side visual factors（背景延续、相机变化、appearance continuity，**不想要**）
- $f(\cdot)$: 未知的 nonlinear mapping

**Reconstruction loss 没有任何机制阻止 $z_t$ 偷偷把 $C_t, V_t$ 编进来**。模型就偷懒了。

打个比方: 你让一个学生考试，考题是"预测下一帧长啥样"。学生发现 80% 的 pixel 都是背景不动，于是他把背景信息全抄进小抄。action 信息也抄了一点，但小抄位置大部分是背景。这就是 reconstruction objective 的 bug。

### 2.2 Causal diagram

画成 SCM:

```
C_t (scene) ──┐
              ↓
V_t (visual)──→ z_t ←── A_t (action)
              ↓
             o_{t+1}
```

$z_t$ 是个 **confounded treatment**。当你用它当 action condition 来 control $o_{t+1}$ 时，你以为是 control action，实际上 control 了一堆别的。

真正用 robot action $u_t$ 去 align 到 $z_t$ 空间时，$u_t$ **只能 specify $A_t$**，specify 不了 $C_t, V_t$。这就相当于: 你让 robot 告诉 world model"我要做 pick-up action"，但 world model 期望的是"pick-up action + 这套背景 + 这个相机角度"。robot 给不出后两个，就只能瞎对齐，结果 action following 一塌糊涂。

### 2.3 Smoking gun: 三个 diagnostic

Paper 设计了三个 test 直接看 latent space 干不干净。如果 $z_t$ 真的只 encode action，这三个 test 都应该 small response。

**Test 1: Zero-transition response**

给 encoder 喂**同一帧**两次，$z_t^0 = \mu_\phi(o_t, o_t)$。两帧完全一样，没 action，按理说 $z_t^0$ 应该接近零。

$$R_{\mathrm{zero}} = \frac{\|\mu_\phi(o_t, o_t)\|_2}{D}$$

$D = \mathrm{RMS}(\|\mu_\phi(o_t, o_{t+1})\|_2) + \epsilon$ 是 ordinary transition latents 的 RMS norm，作 normalization。

DreamDojo 结果: **0.527**。同一帧喂两次，latent 居然有普通 latent 一半那么大！这就是 bug 的铁证——encoder 看见两帧相同，还在编出一个"假 action"。

**Test 2: Camera-shift response**

把 $o_t$ 平移 3 个 pixel 得到 $o_t'$，喂 $\mu_\phi(o_t, o_t')$。相机平移不是 action，应该 small response。

$$R_{\mathrm{shift}} = \frac{\|\mu_\phi(o_t, T_3(o_t))\|_2}{D}$$

$T_3$ 是 horizontal 或 vertical 3-pixel translation。

DreamDojo: horizontal **0.555**, vertical **0.545**。相机晃一下，latent 跟真 action 一样大！这 model 把"相机移动"当 action 了。

**Test 3: Shortcut leakage**

两个 transition pair，看 latent 的 cosine similarity:
- 同一 episode、不同 primitive: 应该 cosine 小
- 不同 episode、同一 primitive: 应该 cosine 大

$$L_{\mathrm{shortcut}} = \mathbb{E}[\cos(z_i, z_j) | \text{same episode, diff primitive}] - \mathbb{E}[\cos(z_i, z_j) | \text{diff episode, same primitive}]$$

DreamDojo: **0.151**。意思是同 episode 不同 action 比 同 action 不同 episode 更接近——scene context 把 action similarity 给盖过了。

### 2.4 Rollout 层面: 两个直观 failure

**Failure 1: 你叫它别动，它还动**

实验: fix 初始帧，把 robot action 全设为 0，$\mathrm{do}(u_t = 0)$。按理说机械臂应该不动。

DreamDojo 结果: rollout 里的机械臂**继续动**。看 Fig. 3。

为什么？因为 $z_t$ 里混了 visual continuation 信息，即使 $u_t = 0$ 被 map 到某个非零 $z_t$，ACWM 看见非零 $z_t$ 就生成 motion。

**Failure 2: 你叫它模仿别的 video，它模仿不来**

实验: 用 video A 的初始帧，用 video B 的 action 序列。应该 reproduce video B 的 motion。

DreamDojo 结果: rollout 不 follow video B 的 action。看 Fig. 4。

为什么？$z_t$ 被 source context "锚定"，action condition 对 motion 的 effect 太弱。

---

## 3. CD-LAM 的修复: 三个 loss 各管一摊

总 loss:
$$\mathcal{L}_{\mathrm{CD}} = \mathcal{L}_{\mathrm{emb}} + \lambda_{\mathrm{ctr}}(k)\mathcal{L}_{\mathrm{ctr}} + \lambda_{\mathrm{cal}}\mathcal{L}_{\mathrm{cal}}$$

变量:
- $\mathcal{L}_{\mathrm{emb}}$: embodiment-centric reconstruction
- $\mathcal{L}_{\mathrm{ctr}}$: action-centric contrastive
- $\mathcal{L}_{\mathrm{cal}}$: latent space calibration
- $\lambda_{\mathrm{ctr}}(k)$: contrastive 权重，随 training step $k$ 变（paper 把 $t$ 留给 frame time，所以这里用 $k$）
- $\lambda_{\mathrm{cal}}$: calibration 权重，固定

### 3.1 $\mathcal{L}_{\mathrm{emb}}$: 别管背景，盯紧前景

最直接的 fix: reconstruction 时**多看 foreground，少看 background**。

用 SAM3（https://arxiv.org/abs/2511.16719）抠出 foreground mask $M_t \in [0,1]^{h \times w}$，定义每像素权重:

$$W_t = \alpha_{\mathrm{fg}} M_t + \alpha_{\mathrm{bg}}(1 - M_t), \quad \alpha_{\mathrm{fg}} > \alpha_{\mathrm{bg}}$$

变量:
- $M_t$: foreground mask（机械臂 + 被操作物体）
- $\alpha_{\mathrm{fg}}, \alpha_{\mathrm{bg}}$: foreground 和 background 权重，前者大于后者
- $h \times w$: frame resolution

Loss:
$$\mathcal{L}_{\mathrm{emb}} = \frac{1}{|\Omega|}\|W_t^{1/2} \odot (\hat{o}_{t+1} - o_{t+1})\|_2^2$$

变量:
- $\hat{o}_{t+1} = D_\psi(o_t, z_t^{\mathrm{CD}})$: 预测的下一帧
- $o_{t+1}$: 真实下一帧
- $\Omega$: pixel grid
- $\odot$: element-wise 乘
- $W_t^{1/2}$: 开根号让 loss 对 $W_t$ 的 gradient 是线性的

**人话**: reconstruction loss 在 foreground 区域放大，在 background 区域缩小。模型为了 minimize loss，$z_t$ 必须 encode 前景运动信息。背景信息 weakly encoded。

**为什么保留 $\alpha_{\mathrm{bg}} > 0$？** 完全忽略 background，模型会在 foreground 边界处乱生成 pixel（反正 background 不算 loss）。保留小权重确保 global visual consistency。

**Ablation 验证（Table V）**: 去掉 $\mathcal{L}_{\mathrm{emb}}$，FG-PSNR 降 0.31 dB，robot action FDCE 降 1.17 px。Camera-shift response 几乎不变。说明它管前景保真度，**不**管相机 confounder。

### 3.2 $\mathcal{L}_{\mathrm{ctr}}$: 别按视觉相似度聚类，按 action 聚类

Reconstruction 把每个 transition 当独立的。两个 transition 可能视觉上像（同 background）但 action 完全不同，reconstruction 不在乎。Contrastive loss 加 pairwise 结构。

公式:
$$\mathcal{L}_{\mathrm{ctr}} = \frac{1}{|\mathcal{P}|}\sum_{(i,j) \in \mathcal{P}} \mathrm{softplus}(-y_{ij}(\tau v_i^\top v_j + b))$$

变量:
- $v_i = \mathrm{norm}(r_\omega(z_i^{\mathrm{CD}}))$: 经过 projection head $r_\omega$ 并 L2 归一化的 latent
- $r_\omega$: auxiliary projection head，**只**这个 loss 用
- $y_{ij} = +1$ if same primitive, $-1$ otherwise
- $\tau$: learned temperature
- $b$: learned bias
- $\mathcal{P}$: pair set
- $\mathrm{softplus}(x) = \log(1 + e^x)$

**人话**: same-primitive 的 transition 在 latent space 拉近，different-primitive 的推开。这是 SigLIP-style sigmoid loss（https://arxiv.org/abs/2303.15343），对 batch size 不敏感。

**Primitive labels 哪来？** 12 类 coarse verb: pick–place, insert–remove, stack–unstack, scoop–dump, open, close, turn on, turn off, wash–rinse, cut, stir, pour。从 video caption 提取 verb 再聚类（Appendix B）。**关键**: 这些是 verb-level categories，**不**是 executable robot action。所以这不算 supervised robot action learning。

Coverage: 68,864 个 transition pair 里 25,192（36.6%）有 primitive label。Unlabeled pairs 不进 contrastive，只进 reconstruction 和 calibration。

**Ablation 验证**: 去掉 $\mathcal{L}_{\mathrm{ctr}}$，PSNR 和 camera-shift 不变，robot action FDCE 降 1.84 px。说明它的 role 是 organize latent space structure，**不**是 sharpen pixel 也**不**是 filter confounder。它是**结构性**的 term。

### 3.3 $\mathcal{L}_{\mathrm{cal}}$: 给 latent space 定个 origin

最有意思的 term。两个子 loss:

$$\mathcal{L}_{\mathrm{cal}} = \mathcal{L}_{\mathrm{KL-fb}} + \mathcal{L}_{\mathrm{zero}}$$

**子 loss 1: Free-bit KL** $\mathcal{L}_{\mathrm{KL-fb}}$

Standard VAE KL penalty 容易让 posterior 全塌缩到 prior（posterior collapse），latent 全变一样。Free-bits trick: 每个 latent dimension 的 KL 必须超过某个 floor 才被 penalize。

$$\mathcal{L}_{\mathrm{KL-fb}} = \sum_d \max(\mathrm{KL}_d, \lambda_{\mathrm{fb}})$$

$\lambda_{\mathrm{fb}}$ 是 free-bits floor。如果某 dimension 的 KL 小于 floor，不 penalize；大于 floor，才 penalize。这保证每个 dimension 至少有 $\lambda_{\mathrm{fb}}$ 的 information capacity。

**人话**: 给每个 latent dimension 留一点"自由空间"，别让它们都塌缩成零。但超过自由空间就罚，防止 latent 编码一切。

**子 loss 2: Zero-transition calibration** $\mathcal{L}_{\mathrm{zero}}$

$$\mathcal{L}_{\mathrm{zero}} = \mathbb{E}_{o_t}\left[\left(\left[\frac{\|z_t^0\|_2}{\mathrm{sg}(s_\Delta) + \epsilon} - m_{\mathrm{zero}}\right]_+\right)^2\right]$$

变量（重要，一个一个说）:
- $z_t^0 = \mu_\phi(o_t, o_t)$: zero-transition latent（同帧喂两次的输出）
- $s_\Delta$: running RMS norm of ordinary transition latents（普通 transition latent 的 RMS 范数）
- $\mathrm{sg}(\cdot)$: **stop-gradient**，让 $s_\Delta$ 作为常量，不参与 gradient
- $[x]_+ = \max(x, 0)$: hinge 函数，只 penalize 正的部分
- $m_{\mathrm{zero}}$: margin，zero-transition relative norm 的上限
- $\epsilon$: 小常数防除零

**人话**: 强制 zero-transition latent 的 norm **小于** $m_{\mathrm{zero}} \times s_\Delta$。把"no action"对应到 latent space 的原点。

为什么 stop-gradient $s_\Delta$？如果不 stop-gradient，optimizer 可能通过 inflate $s_\Delta$（让普通 latent 变更大）来 trivially satisfy constraint——零 latent 看起来就小了，但其实啥也没修。Stop-gradient 让 $s_\Delta$ 当固定 normalizer。

为什么 hinge？只罚"超过 margin"的部分，已经很小就不再罚，避免 over-shrink。

**Ablation 验证（Table V）**: 去掉 $\mathcal{L}_{\mathrm{zero}}$，camera-shift response 从 0.133/0.101 暴涨到 0.637/0.637（4.8× / 6.3× 大）。这是**主导 term** for suppressing camera-shift confounder。

**有意思的 footnote**: 去掉 $\mathcal{L}_{\mathrm{zero}}$ 后 FDCE 反而 lower（16.10 vs 17.23）。Paper 说这不是 better debiasing，因为它 fail 了 camera-shift diagnostic。这是个 **cautionary tale**: 不能只看 downstream metric，要看 causal diagnostic。模型可能 overfit 到 action condition 但还 absorb confounder，downstream 看着好但 generalization 会崩。

### 3.4 三个 loss 互补不冗余

汇总 ablation:

| Term | 控制什么 | 不控制什么 |
|------|---------|----------|
| $\mathcal{L}_{\mathrm{emb}}$ | 前景保真度，action following | camera-shift response |
| $\mathcal{L}_{\mathrm{ctr}}$ | latent space structure | pixel 保真，camera-shift |
| $\mathcal{L}_{\mathrm{cal}}$ | zero reference，camera-shift | pixel 保真 |

每个 term 对应一个 specific failure mode。好的 method 设计就这样: ablation 能 cleanly attribute 效果到具体 term。

---

## 4. Three-stage Pipeline: 为什么不一起训

```
Stage 1: LAM debiased fine-tuning (1k steps)
  Input: action-unlabeled human video
  Loss: L_CD
  Output: debiased LAM encoder μ_φ

Stage 2: ACWM debiased fine-tuning (2k steps)
  Input: 同样 unlabeled video，用 debiased LAM 提 z_t
  Loss: 标准 ACWM loss
  Output: ACWM 适应 debiased latent space

Stage 3: Robot action adaptation (3k for 2B, 6k for 14B)
  Input: paired (o_t, u_t, o_{t+1})
  Loss: g_η(u_t) ≈ sg(μ_φ(o_t, o_{t+1})) + cycle consistency
  Output: ACWM follow executable robot action
```

**为什么不 joint train？**

如果 joint train，ACWM 可能一边训一边把 LAM 的 debiasing "unlearn" 掉——因为 confounder 对 reconstruction 是 cheap signal，ACWM 会乐于利用。Separate stage 让 debiasing 在 upstream 先定型，downstream 只能 adapt 到 clean space。

**Stage 3 的 bridge 设计**:

$g_\eta(u_t) \approx \mathrm{sg}(\mu_\phi(o_t, o_{t+1}))$

- $\mathrm{sg}$: stop-gradient，让 loss 只 train bridge $g_\eta$，**不**改 LAM $\mu_\phi$
- 加 auxiliary action readout + cycle consistency: 鼓励 mapped latent 保留 $u_t$ 信息，防止 bridge collapse 到平凡 mapping

---

## 5. 结果: 数字怎么读

### 5.1 LAM audit (Table I)

| Diagnostic | DreamDojo | CD-LAM | 改善 |
|-----------|-----------|--------|------|
| Zero-transition median | 0.527 | 0.043 | 12.3× |
| Camera-shift horizontal | 0.555 | 0.156 | 3.6× |
| Camera-shift vertical | 0.545 | 0.110 | 4.9× |
| Shortcut leakage | 0.151 | 0.014 | 10.8× |

Action-neighbor preservation: same-primitive pairs from different episodes cosine **0.132 (DreamDojo) vs 0.131 (CD-LAM)**——几乎不变。这说明 CD-LAM **不**是 uniformly shrink latent space（那样会 destroy action 信息），是定向 debias。

### 5.2 ACWM rollouts (Table II, Stage 2)

**2B**:
- FDCE: 34.00 → 19.63（-42%）
- PSNR: 20.88 → 24.29（+3.41 dB）
- LPIPS: 0.413 → 0.308

**14B**:
- FDCE: 40.29 → 29.87（-26%）
- PSNR: 21.04 → 23.18（+2.14 dB）

**Crucial observation**: Baseline 从 2B 到 14B，FDCE 反而**worsened**（34 → 40）。Scale 没修 action following，反而放大 visual capability 但 confounder 一起放大了。CD-LAM 在两个 scale 都改善，14B CD-LAM 全面比 2B CD-LAM 好——scale 和 debiasing **complementary**。

### 5.3 Robot action rollouts (Table III, Stage 3) — 主结果

**2B**:
| Metric | DreamDojo | CD-LAM |
|--------|-----------|--------|
| FDCE mean | 12.63 | 8.24 |
| FDCE median | 8.15 | 6.75 |
| PSNR | 19.85 | 20.60 |
| $\mathrm{do}(u_t=0)$ FDCE | 10.71 | 5.03 |
| Target transfer FDCE | 24.36 | 22.55 |

**14B**:
| Metric | DreamDojo | CD-LAM |
|--------|-----------|--------|
| FDCE mean | 11.11 | 7.73 |
| FDCE median | 8.98 | 5.99 |
| PSNR | 20.01 | 21.01 |
| $\mathrm{do}(u_t=0)$ FDCE | 9.36 | **2.18** |
| Target transfer FDCE | 24.82 | 21.11 |

**最亮眼的数字**: 14B CD-LAM 的 zero-action FDCE 是 **2.18**，比 baseline 低 77%。你叫它不动，它真不动了。这就是 Fig. 3 baseline 缺失的行为，被修好了。

### 5.4 效率 (Fig. 8)

14B CD-LAM 在 3k-4k updates 内 reach DreamDojo 50k reference 水平，6k 时已经 surpass。**12× fewer updates**。

**Causal explanation**: Baseline 的 ACWM 学了 confounded $z_t$，Stage 3 要在扭曲的 mapping 上 navigate，optimization landscape ill-conditioned。CD-LAM 的 ACWM 从一开始就在 clean space 上训，Stage 3 mapping 直接，landscape well-conditioned。

这就像: 修好 bug vs 在 buggy code 上 workaround。前者省力，后者费力且 fragile。

### 5.5 Data scaling (Table IV)

| Model | PSNR | FDCE mean | FDCE median |
|-------|------|-----------|-------------|
| DreamDojo | 19.85 | 12.63 | 8.15 |
| CD-LAM-1h | 20.54 | 8.91 | 6.88 |
| CD-LAM-10h | 20.61 | 8.87 | 6.23 |
| CD-LAM-100h | 20.60 | 8.24 | 6.75 |
| CD-LAM-1000h | 20.64 | 7.97 | 6.12 |

**1 小时** debiasing 数据已经 capture 1000 小时 tier 的 80% FDCE improvement（3.72 of 4.66 points）。Benefit 主要来自 **debiasing mechanism itself**，**不**来自 data scale。

这是 **less is more** 的强证据。结构性 inductive bias 比单纯 scale data 更 efficient。

### 5.6 Per-action (Fig. 7, A.1)

CD-LAM 在 8 个 action category 上都有 improvement，**不**集中在 single primitive。说明 gain generalize across actions。

Fig. A.1 也老实承认: 少数 category 上 CD-LAM 持平甚至略输 baseline。诚实 reporting。

---

## 6. FDCE 这个 metric 为啥重要

Standard video metric 是 PSNR、FVD（https://arxiv.org/abs/1812.01717）。但这些都只看 pixel。

Fig. 6(b) 显示: PSNR 和 FDCE 的 correlation $r = -0.38$, $R^2 = 0.14$。**PSNR 只 explain 14% 的 FDCE variance**。

一个 rollout 可以 high PSNR 但 high FDCE——它 copy 上下文，sharp 看着像，但完全 ignore action condition。所以 pixel metric **不**足以 evaluate controllability。

FDCE 定义:
$$a_j^s = p_j^s - p_j^0, \quad \hat{a}_i^s = \hat{p}_i^s - \hat{p}_i^0$$

$p_j^s$ 是 reference track $j$ 在 step $s$ 的位置，$a_j^s$ 是相对初始帧的 displacement。**关键**: measure motion，**不**measure 绝对位置。

$$c_{ij} = \frac{1}{H}\sum_{s=1}^{H}\|\hat{a}_i^s - a_j^s\|_2$$

$$\mathrm{FDCE} = \frac{1}{2N_g}\sum_{i=1}^{N_g}\min_j c_{ij} + \frac{1}{2N_r}\sum_{j=1}^{N_r}\min_i c_{ij}$$

Symmetric Chamfer distance。$N_g$ 是 generated track 数，$N_r$ 是 reference track 数。双向最近邻平均。

Implementation:
- Foreground mask: SAM3（https://arxiv.org/abs/2511.16719）
- Point track: CoWTracker（https://arxiv.org/abs/2602.04877），warping-based tracker
- 每对 rollout 最多 16 个 anchor
- Anchor seeded in eroded foreground mask，避免边界 artifact
- Low visibility track 直接丢弃

相关 work:
- MotionPro（https://arxiv.org/abs/2502.10326）: 用 trajectory distance 测 motion control，提出 ObjMC metric
- TAP-Vid（https://arxiv.org/abs/2211.03726）: tracking any point benchmark
- TAPIR（https://arxiv.org/abs/2306.08637）: per-frame init + temporal refinement
- CoTracker（https://arxiv.org/abs/2407.07624）: better to track together

---

## 7. 几个 mental model，帮你 build intuition

### 7.1 Latent action 是 sufficient statistic 的副作用

Reconstruction 让 $z_t$ 成为 $p(o_{t+1}|o_t, z_t)$ 的 sufficient statistic。Sufficient statistic 包含**所有 predictive 信息**，包括 spurious correlate。

这就像 linear regression: 一个 feature 跟 target 相关，regression 就会用，哪怕它不是 causal。Reconstruction 是 statistical optimization，不是 causal optimization。

CD-LAM 加 structural constraint，让 $z_t$ 只 encode causally relevant 信息。

### 7.2 Latent space geometry

Reconstruction 只定义 "information content"，**不**定义 geometry。两个 transition 可以有相似 $z_t$ 但因为完全不同 reason（一个因为同 action，一个因为同 background）。

CD-LAM 三个 loss 塑造 geometry:
- $\mathcal{L}_{\mathrm{emb}}$: 对 foreground predictive 的 latent dimension 被 amplify
- $\mathcal{L}_{\mathrm{ctr}}$: same-primitive transitions cluster
- $\mathcal{L}_{\mathrm{cal}}$: zero-transition 在 origin 附近，ordinary transition 保持 distance

合在一起，把 latent space 从 "sufficient statistic embedding" 塑造成 "action semantic space with calibrated origin"。

### 7.3 Less is more 的 mechanism

Baseline: ACWM 学了 confounded $z_t$，robot action $u_t$ 要 navigate 扭曲 mapping，landscape ill-conditioned，需要很多 updates。

CD-LAM: LAM 先 debias，$z_t$ clean。ACWM 适应 clean space，robot action mapping 直接，landscape well-conditioned，少量 updates 够。

就像修 bug vs 在 buggy code 上 workaround。前者省力，后者费力。

### 7.4 Confounding 的 geometry 想法

两个 transition $(o_t^a, o_{t+1}^a)$ 和 $(o_t^b, o_{t+1}^b)$，同 action $A$ 但不同 background。

Reconstruction LAM:
$$z_t^a \approx f(A, C^a, V^a), \quad z_t^b \approx f(A, C^b, V^b)$$

$z_t^a \neq z_t^b$，即使 action 相同。Confounding。

CD-LAM:
$$z_t^a \approx f(A), \quad z_t^b \approx f(A)$$

$z_t^a \approx z_t^b$。Debiased。

$\mathcal{L}_{\mathrm{ctr}}$ 显式 enforce 这个: same-primitive 被 pull together。

---

## 8. 这篇 paper 在大图景里的位置

### 8.1 跟 Karpathy 之前讲的几个 idea 的连接

**Software 2.0/3.0**: Karpathy 说过用 learned function 替代 hand-coded rules。LAM 是这个 idea 的延伸: 用 learned latent action 替代 hand-coded action space。CD-LAM 加 inductive bias，是 "Software 2.0 with structure"。

**Inductive bias matters**: Karpathy 在 CS231n、build-nanogpt 反复强调，神经网络 architecture 的 inductive bias 重要，不能完全 scale 解决。CD-LAM 的三个 loss 就是 inductive bias 的实例化。

**Diagnostic visualization**: Karpathy 一直说不要只看 final metric，要看 intermediate representations。Paper 的 LAM audit (Table I) 就是这种 diagnostic。

**Causal thinking**: Karpathy 在多个 talk 里讲过 shortcut learning 是 deep learning 的 pervasive problem。CD-LAM 是 causal thinking 对 representation learning 的具体应用。

### 8.2 跟 LLM 的类比

LLM 也有类似 confounding: token representation 可能 confound syntax 和 semantics，sentence embedding 可能 confound topic 和 sentiment。Probing study 经常发现这种 confounding。

CD-LAM 的方法可能 inspire LLM debiasing:
- Contrastive loss 用 semantic label
- Calibration loss 定义 "neutral" reference
- "Foreground-style" weighting 强调 causally relevant token

参考:
- Probing classifier: https://arxiv.org/abs/1902.08966
- Disentangled representation: https://arxiv.org/abs/1812.02230
- Information Bottleneck: https://arxiv.org/abs/1703.00810

### 8.3 跟 VAE 和 masked autoencoder 的连接

VAE 用 reconstruction 训 latent，同样有 confounding 问题: latent 可能 encode 跟下游 task 无关的信息。MAE（https://arxiv.org/abs/2111.06377）用 mask reconstruction 训 representation，也是 statistical sufficiency 而非 causal sufficiency。

CD-LAM 的方法可能 transfer 到 VAE 和 MAE: 加 contrastive structure，加 calibration reference，加 task-relevant weighting。

---

## 9. 局限和 open question

### 9.1 SAM3 dependency

$\mathcal{L}_{\mathrm{emb}}$ 依赖 SAM3 foreground mask 质量。SAM3 fail 的 case（heavy occlusion、motion blur、transparent object），foreground weighting 会错。Possible fix: co-train SAM3 和 LAM，或 uncertainty-aware mask。

### 9.2 Primitive label noise

12-way primitive 来自 caption verb，对 caption 质量敏感。Caption mis-classify 或漏 verb，contrastive 推错方向。Possible fix: confidence-weighted contrastive。

### 9.3 Point tracker limitation

FDCE 依赖 CoWTracker，textureless gripper 上 drift。Paper 说这 bias 比较 toward null（所有 model 同样受影响），但绝对值 inflated。Possible fix: 多 tracker ensemble，或 learned tracker for robotic manipulation。

### 9.4 Cross-embodiment

Paper 只在 AgiBot 上 evaluate。其他 platform（Franka、UR5、Kuka）效果如何？Bridge $g_\eta$ 是否 embodiment-specific？多个 bridge share 一个 debiased LAM？

### 9.5 Long-horizon stability

Paper evaluate $H$ 步 rollout（具体 horizon 在 appendix），但 long-horizon（几百步）的 error accumulation 如何？Debiased latent space 在 long rollout 中应该更 stable（action 跟随更好），但要验证。

### 9.6 Compositional action

12-way primitive coarse。复杂 action（"screw in then tighten"）怎么 represent？Paper [27] 的 additively compositional latent action（https://arxiv.org/abs/2604.03340）是 relevant direction。

### 9.7 Supervised upper bound

如果直接用 supervised action prediction（有 action label）训 LAM，效果如何？这是 upper bound。Paper 没做这比较。

### 9.8 Theoretical analysis

三个 loss 的 interaction 能不能 theoretically 分析？Identifiability of debiased latent action under 某些 assumption？可能 connect 到 ICA（https://arxiv.org/abs/2010.02607）和 disentangled representation theory。

---

## 10. 我能想到的后续方向

### 10.1 Auto-discover primitive

12-way primitive 手工定义。能不能 auto-discover？Vector quantization 或 clustering，但要保证 action-consistent。

### 10.2 Hierarchical latent action

Coarse primitive + fine-grained action decomposition。比如 "scoop" primitive 下不同 scoop trajectory。Hierarchical VQ-VAE 的 application。

### 10.3 Closed-loop control

Paper evaluate open-loop。Closed-loop MPC with CD-LAM as forward model 效果如何？Debiased latent space 让 planning 更 stable？

### 10.4 Causal discovery of confounder

Paper 假设 $C_t, V_t$ 是 confounder。能不能 auto-discover confounder from data？Causal discovery + representation learning 的结合。

### 10.5 Multi-modal latent action

Video + audio + tactile 都有 action 信息。Multi-modal latent action 怎么 debias？

### 10.6 Adversarial debiasing

CD-LAM 用 explicit constraint。能不能用 adversarial training？Generator 学 $z_t$，discriminator 试图从 $z_t$ 预测 confounder（scene、visual），generator 试图 fool discriminator。参考 adversarial debiasing 文献。

---

## 11. 一句话再总结

**Reconstruction 是 sufficient statistic 的训练，会偷懒把所有 predictive 信息塞进 latent。Robot action 只能 specify 一部分，剩下对不上，action following 崩。用三个 loss 修一下——foreground 加权、action contrastive、zero calibration——12 倍效率提升。**

核心 insight 三个:

1. **Sufficient statistic $\neq$ causal representation**。Reconstruction 给你前者，不给后者。
2. **Confounding 在 conditioning variable 上比在 input 上更阴险**。它 corrupt controllability 而不是 accuracy，看起来 sharp 但 control 不动。
3. **Targeted debiasing 比 scale efficient**。1k 步修 LAM > 50k 步 robot action adaptation。修 bug 比 workaround 强。

这是 "小而精" 的 paper: 不引入大 architecture，不需要 massive compute，只 carefully 分析 bug 然后 fix。这种 clarity 在 scale-everything 潮流里很 refreshing。

参考全汇总:
- DreamDojo baseline: https://arxiv.org/abs/2602.06949
- Genie: https://arxiv.org/abs/2402.15391
- LAPO: https://arxiv.org/abs/2312.10812
- LAPA: https://arxiv.org/abs/2410.11758
- AdaWorld: https://arxiv.org/abs/2505.18834
- Moto: https://arxiv.org/abs/2410.23214
- IGOR: https://arxiv.org/abs/2411.00785
- AgiBot World: https://arxiv.org/abs/2503.06669
- Ego4D: https://arxiv.org/abs/2110.15270
- EgoDex: https://arxiv.org/abs/2505.11709
- SAM3: https://arxiv.org/abs/2511.16719
- CoWTracker: https://arxiv.org/abs/2602.04877
- SigLIP: https://arxiv.org/abs/2303.15343
- IRM: https://arxiv.org/abs/1907.02893
- Shortcut Learning: https://arxiv.org/abs/2004.07780
- Causal Confusion in IL: https://arxiv.org/abs/1905.11979

如果哪部分还想更 deep dive——比如 free-bits 的具体 gradient flow、$\lambda_{\mathrm{ctr}}(k)$ schedule、CoWTracker architecture、或者某个具体实验细节——跟我说，我展开。

---

# CD-LAM: Causally Debiased Latent Action Model for Embodied Action Conditioned World Models — 深度解析

## 1. 大图景: Why this paper matters

这篇paper的核心intellectual contribution在于: 它**精确诊断**了 Latent Action Model (LAM) 的一个**根本性representational bug** — reconstruction objective会surreptitiously把action-irrelevant confounders塞进latent action $z_t$ — 然后**针对性**地用三个causally-motivated objectives修复它, 最终在2B和14B两个backbone上证明了"less is more": 1k步LAM debiasing 比 50k步robot action adaptation更有效。

这背后是一个经典的causal inference问题在deep representation learning中的具体体现: **when you optimize a sufficient statistic for prediction, you get everything predictive, including spurious correlates**。这个insight不仅对LAM成立, 对所有"用reconstruction来定义latent code"的方法都有implications — VAE, masked autoencoders, contrastive predictive coding, JEPA等等。

参考链接:
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- DreamerV3: https://arxiv.org/abs/2301.04104
- iVideoGPT: https://arxiv.org/abs/2405.15223
- UniSim: https://arxiv.org/abs/2310.06114
- Cosmos: https://arxiv.org/abs/2501.03575

---

## 2. 问题setting: ACWM与LAM的两层架构

### 2.1 Action-Conditioned World Model (ACWM)

目标分布:
$$p_\theta(o_{t+1:t+H} \mid o_{\leq t}, u_{t:t+H-1})$$

变量含义:
- $o_t$: 在time $t$的visual observation (一般是RGB frame)
- $u_t$: recorded executable robot action (例如relative end-effector displacement)
- $H$: prediction horizon
- $o_{\leq t}$: observation history (context)
- $u_{t:t+H-1}$: action sequence (intervention handle)

训练方式一般是conditional next-video prediction with diffusion-style video losses。

### 2.2 Latent Action Model (LAM)

LAM解决"data bottleneck"问题: 真实robot data昂贵, 而human egocentric video海量但无action annotation。LAM从相邻frames推断latent action:

$$z_t \sim q_\phi(z \mid o_t, o_{t+1}), \quad \hat{o}_{t+1} = D_\psi(o_t, z_t)$$

变量:
- $z_t \in \mathbb{R}^d$: latent action vector (d维)
- $q_\phi$: encoder (posterior, 参数$\phi$)
- $D_\psi$: decoder (参数$\psi$)
- $\mu_\phi(o_t, o_{t+1})$: posterior mean, 用于需要deterministic latent的场合

标准LAM objective:
$$\mathcal{L}_{\mathrm{LAM}} = \mathbb{E}\big[\ell(D_\psi(o_t, z_t), o_{t+1})\big] + \mathrm{Reg}(z_t)$$

这里 $\ell$ 是reconstruction loss, $\mathrm{Reg}$ 是capacity control (如KL bottleneck)。**这个objective**只要求 $z_t$ 对 $p(o_{t+1}|o_t, z_t)$ 是predictively sufficient, 但**没有**任何constraint说 $z_t$ 必须**只**encode action信息。

### 2.3 LAM-based ACWM pipeline

先用LAM把video transitions encode为 $z_t$, 再用latent action训练ACWM:
$$p_\theta(o_{t+1:t+H} \mid o_{\leq t}, z_{t:t+H-1})$$

最后通过一个lightweight bridge $g_\eta(u_t) \approx z_t$ 把executable robot action映射到latent space。

代表系统: **DreamDojo** (https://arxiv.org/abs/2602.06949)。Genie (https://arxiv.org/abs/2402.15391), LAPO (https://arxiv.org/abs/2312.10812), LAPA (https://arxiv.org/abs/2410.11758), AdaWorld (https://arxiv.org/abs/2505.18834), Moto (https://arxiv.org/abs/2410.23214), IGOR (https://arxiv.org/abs/2411.00785) 都是这一family。

---

## 3. 核心诊断: 为什么reconstruction admits confounders

### 3.1 The decomposition that reveals the bug

Paper写出关键的decomposition (Eq. 5):
$$z_t = \mu_\phi(o_t, o_{t+1}) \approx f(A_t, C_t, V_t)$$

变量:
- $A_t$: **embodied action effects** (我们希望 $z_t$ 编码的)
- $C_t$: **scene context** (场景上下文, 我们不希望编码)
- $V_t$: **source-side visual factors** (background continuation, appearance continuity, camera-like variation)
- $f(\cdot)$: 一个未知的nonlinear mapping

这个decomposition的核心洞察: reconstruction objective**没有任何机制**阻止 $z_t$ 对 $C_t$ 和 $V_t$ 的non-trivial dependence。

### 3.2 Causal diagram intuition

可以画一个简化的SCM (Structural Causal Model):

```
   C_t (scene) ─────┐
                    ↓
   V_t (visual) ────→ z_t ←─── A_t (action)
                    ↓
                   o_{t+1}
```

这里 $C_t, V_t, A_t$ 三者都cause $z_t$, 而 $z_t$ 又cause $o_{t+1}$。当我们用 $z_t$ 作为action condition来control $o_{t+1}$ 时, $z_t$ 实际上是个**confounded treatment**: $C_t$ 和 $V_t$ 通过 $z_t$ 间接影响 $o_{t+1}$, 但它们**不**是robot能控制的。

当robot action $u_t$ 试图align到 $z_t$ space时, $u_t$ 只能specify $A_t$, **不能**specify $C_t$ 或 $V_t$。所以executable robot actions被forced进入一个部分non-actionable latent, 导致weak action following。

这个问题的formal name在causal inference里叫 **confounding by indication** 或者更广义的 **shortcut learning**。参考:
- Shortcut Learning in Deep Neural Networks: https://arxiv.org/abs/2004.07780
- Invariant Risk Minimization: https://arxiv.org/abs/1907.02893
- Causal Confusion in Imitation Learning: https://arxiv.org/abs/1905.11979

### 3.3 实证: DreamDojo的两个failure mode

Paper做了两个非常具体的intervention实验:

**Intervention 1: Zero action** $\mathrm{do}(u_t = 0)$
- Fix initial frame
- Replace conditioning action with zero
- Expected: embodiment motion suppressed (机械臂不动)
- Actual (DreamDojo): rollout仍然moves (Fig. 3)

这个failure的causal explanation: $z_t$ 仍然encoding了某些visual continuation信息, 即使 $u_t = 0$ 被map到一个非零 $z_t$, ACWM继续generate motion。

**Intervention 2: Target-action transfer** 
- Source video提供initial frame
- Target video提供action sequence
- Expected: rollout reproduce target embodiment dynamics
- Actual (DreamDojo): rollout doesn't follow target action (Fig. 4)

这个failure说明 $z_t$ 被 source context "anchor"住, action condition只对motion有weak effect。

### 3.4 LAM audit: 三个diagnostic

Paper设计了三个quantitative diagnostic检查latent action space的purity:

**Zero-transition response** (检查: duplicated frame应该output近零latent):
$$R_{\mathrm{zero}} = \frac{\|\mu_\phi(o_t, o_t)\|_2}{D}$$

其中 $D = \mathrm{RMS}(\|\mu_\phi(o_t, o_{t+1})\|_2) + \epsilon$ 是ordinary transition latents的RMS norm (作为normalization)。

DreamDojo: 0.527 (median)。这意味着即使input两帧完全相同, encoder仍然output一个norm约为ordinary latent一半的vector! 这就是bug的smoking gun。

**Camera-shift response** (检查: 纯相机shift应该只产生small latent response):
$$R_{\mathrm{shift}} = \frac{\|\mu_\phi(o_t, T_3(o_t))\|_2}{D}$$

其中 $T_3$ 是horizontal或vertical 3-pixel translation at 320×640 resolution。

DreamDojo: horizontal 0.555, vertical 0.545。Camera shift几乎和ordinary transition产生一样大的latent! 这意味着latent space把"相机移动"当成"action"了。

**Shortcut leakage** (检查: scene context是否confound action similarity):
$$L_{\mathrm{shortcut}} = \mathbb{E}[\cos(z_i, z_j) | \text{same episode, diff primitive}] - \mathbb{E}[\cos(z_i, z_j) | \text{diff episode, same primitive}]$$

如果latent space干净, 后一项(同action跨episode)应该比前一项(同episode跨action)更close。DreamDojo: 0.151, 说明scene context confound了primitive similarity。

---

## 4. CD-LAM方法: 三个causally debiased objectives

### 4.1 总objective

$$\mathcal{L}_{\mathrm{CD}} = \mathcal{L}_{\mathrm{emb}} + \lambda_{\mathrm{ctr}}(k)\mathcal{L}_{\mathrm{ctr}} + \lambda_{\mathrm{cal}}\mathcal{L}_{\mathrm{cal}}$$

变量:
- $\mathcal{L}_{\mathrm{emb}}$: embodiment-centric reconstruction
- $\mathcal{L}_{\mathrm{ctr}}$: action-centric contrastive learning
- $\mathcal{L}_{\mathrm{cal}}$: latent space calibration
- $\lambda_{\mathrm{ctr}}(k)$: contrastive weight, 随training step $k$ 变化 (paper reserve $t$ 给frame time)
- $\lambda_{\mathrm{cal}}$: calibration weight, 固定

设计原则:
1. **Embodiment-centric Reconstruction**: reconstruction signal应该emphasize embodiment dynamics相关的regions
2. **Action-centric Structure**: latent space应该group similar action videos, **不**是visually similar但action-different的
3. **Calibrated, Non-collapsed Latent Space**: duplicated-frame inputs应该map到zero-transition reference, ordinary transitions保留variation

### 4.2 Embodiment-centric Reconstruction Loss

这是最直接的intervention: 用SAM3 (https://arxiv.org/abs/2511.16719)得到foreground mask, 给foreground region更高weight。

具体公式:
$$W_t = \alpha_{\mathrm{fg}} M_t + \alpha_{\mathrm{bg}}(1 - M_t), \quad \alpha_{\mathrm{fg}} > \alpha_{\mathrm{bg}}$$

$$\mathcal{L}_{\mathrm{emb}} = \frac{1}{|\Omega|}\|W_t^{1/2} \odot (\hat{o}_{t+1} - o_{t+1})\|_2^2$$

变量:
- $M_t \in [0,1]^{h \times w}$: SAM3生成的embodiment-object foreground mask, $h \times w$是frame resolution
- $\alpha_{\mathrm{fg}}, \alpha_{\mathrm{bg}}$: foreground和background的weight, $\alpha_{\mathrm{fg}} > \alpha_{\mathrm{bg}}$ (paper具体值在appendix)
- $W_t$: 每像素spatial weight
- $\hat{o}_{t+1} = D_\psi(o_t, z_t^{\mathrm{CD}})$: 预测的下一帧
- $o_{t+1}$: ground truth下一帧
- $\Omega$: pixel grid
- $\odot$: element-wise multiplication
- $W_t^{1/2}$: square root是为了让loss的gradient相对 $W_t$ 是线性的

**Intuition**: 这个loss实际上是weighted MSE。Reconstruction在foreground region被amplified, 在background region被attenuated。因此 $z_t$ 必须encode foreground dynamics信息才能minimize loss, background信息只是weakly encoded。

为什么保留 $\alpha_{\mathrm{bg}} > 0$? 完全忽略background会让model学到foreground boundary附近的artifacts (比如把background region的pixel随意generate)。保留小weight确保global visual consistency。

Ablation (Table V)验证: 去掉这个term, FG-PSNR降0.31dB, robot action FDCE降1.17px。但camera-shift response几乎不变。说明这个term控制foreground fidelity, **不**控制source-side visual confounders。

### 4.3 Action-centric Contrastive Learning

Reconstruction alone treats每个transition独立, 缺少**inter-transition structure**。Contrastive loss引入action-consistent的pairwise constraint。

公式:
$$\mathcal{L}_{\mathrm{ctr}} = \frac{1}{|\mathcal{P}|}\sum_{(i,j) \in \mathcal{P}} \mathrm{softplus}(-y_{ij}(\tau v_i^\top v_j + b))$$

变量:
- $v_i = \mathrm{norm}(r_\omega(z_i^{\mathrm{CD}}))$: 经过projection head $r_\omega$并L2归一化的latent action
- $r_\omega$: auxiliary projection head, **只**用于这个loss, 不影响downstream
- $y_{ij} = +1$ for same-primitive pairs, $y_{ij} = -1$ otherwise
- $\tau$: learned temperature
- $b$: learned bias
- $\mathcal{P}$: pair set
- $\mathrm{softplus}(x) = \log(1 + e^x)$: smooth approximation of ReLU

这是 **SigLIP-style** sigmoid loss (https://arxiv.org/abs/2303.15343), 而**不是** InfoNCE-style softmax loss。好处是对batch size不敏感, 不需要大的negative pool。

**Intuition**: Same-primitive pairs被拉近 (positive samples), different-primitive pairs被推远 (negative samples)。这organize latent space按action semantics聚类, 而不是按visual context聚类。

Primitive labels来自12-way canonical verb set: pick–place, insert–remove, stack–unstack, scoop–dump, open, close, turn on, turn off, wash–rinse, cut, stir, pour。这些labels从video caption中extract并cluster (Appendix B)。**关键**: 这些是coarse verb-level categories, **不**是executable robot actions, 所以这个term只是shape representation, **不**让CD-LAM变成supervised robot action learning。

Label coverage: 68,864 transition pairs中25,192 (36.6%)有primitive label。Unlabeled pairs只contribute到reconstruction和calibration, 不进入contrastive pair。这样设计避免unlabeled pairs被错误地当positive或negative。

Ablation (Table V)发现: 去掉contrastive, PSNR和camera-shift几乎不变, 但robot action FDCE降1.84px。这说明它的role是 **organize transition neighborhoods**, **不**是sharpen frames或者过滤confounders。它是**structure-building**的term。

### 4.4 Latent Space Calibration

这是最causally-motivated的term, 也最有intuition:

$$\mathcal{L}_{\mathrm{cal}} = \mathcal{L}_{\mathrm{KL-fb}} + \mathcal{L}_{\mathrm{zero}}$$

#### Free-bit KL Term $\mathcal{L}_{\mathrm{KL-fb}}$

Standard VAE KL penalty容易导致posterior collapse (所有 $z$ 都塌缩到prior)。Free-bits trick: 每个latent dimension的KL必须超过某个floor才被penalize, 防止collapse同时提供capacity control。

形式上, per-dimension KL $\mathrm{KL}_d$:
$$\mathcal{L}_{\mathrm{KL-fb}} = \sum_d \max(\mathrm{KL}_d, \lambda_{\mathrm{fb}})$$

其中 $\lambda_{\mathrm{fb}}$ 是free-bits floor。如果某个dimension的KL小于floor, 不penalize; 大于floor, 才penalize。这保证每个dimension至少有 $\lambda_{\mathrm{fb}}$ 的information capacity, 防止collapse。

#### Zero-transition Calibration Term $\mathcal{L}_{\mathrm{zero}}$

$$\mathcal{L}_{\mathrm{zero}} = \mathbb{E}_{o_t}\left[\left(\left[\frac{\|z_t^0\|_2}{\mathrm{sg}(s_\Delta) + \epsilon} - m_{\mathrm{zero}}\right]_+\right)^2\right]$$

变量:
- $z_t^0 = \mu_\phi(o_t, o_t)$: zero-transition latent (duplicated-frame input)
- $s_\Delta$: running RMS norm of ordinary transition latents $z_t = \mu_\phi(o_t, o_{t+1})$
- $\mathrm{sg}(\cdot)$: stop-gradient (防止zero loss影响ordinary latents的scale)
- $[x]_+ = \max(x, 0)$: hinge function
- $m_{\mathrm{zero}}$: small margin (零action的relative norm上限)
- $\epsilon$: small constant防止除零

**Intuition**: 这个loss强制zero-transition latent的norm **below** $m_{\mathrm{zero}} \times s_\Delta$。它定义了latent space的"origin" — 即"no action"的reference point。

为什么需要stop-gradient on $s_\Delta$? 如果不加stop-gradient, optimizer可能通过inflate $s_\Delta$ (让ordinary latents更大)来trivially satisfy constraint, 这会破坏latent space scale。Stop-gradient让 $s_\Delta$ 作为固定的normalization constant。

为什么用hinge $[\cdot]_+$ 而不是直接square? Hinge只penalize当zero-transition norm **超过** margin, 不会over-shrink。如果zero-transition已经很小, loss为零, 不再施加force。这避免over-regularization。

Ablation (Table V): 去掉zero-trans calibration, camera-shift response从0.133/0.101暴涨到0.637/0.637 (4.8× / 6.3× larger)。这是**dominant term** for suppressing action-irrelevant camera-shift response。

有意思的footnote: 去掉zero-trans calibration后FDCE反而lower (16.10 vs 17.23)。Paper解释这不是better debiasing, 因为它fail了camera-shift diagnostic。这是一个**cautionary tale**: 不能只看downstream metric, 要看causal diagnostic。如果模型overfit到action condition但仍然absorb confounders, downstream可能看起来好但generalization会失败。

### 4.5 三个terms的互补性

综合ablation结果:

| Term | What it controls | What it doesn't control |
|------|------------------|------------------------|
| $\mathcal{L}_{\mathrm{emb}}$ | Foreground fidelity, action following | Camera-shift response |
| $\mathcal{L}_{\mathrm{ctr}}$ | Latent space structure, robot action FDCE | Pixel fidelity, camera-shift |
| $\mathcal{L}_{\mathrm{cal}}$ | Zero-transition reference, camera-shift | Pixel fidelity (基本中性) |

三个terms**互补不冗余**。这是好的method设计的关键: 每个component对应一个specific failure mode, ablation能cleanly attribute效果。

---

## 5. Three-stage Training Pipeline

```
Stage 1: LAM Debiased Fine-tuning (1k steps)
  Input: action-unlabeled video
  Loss: L_CD
  Output: debiased LAM encoder μ_φ
  
Stage 2: ACWM Debiased Fine-tuning (2k steps)
  Input: action-unlabeled video + debiased LAM
  Loss: standard ACWM loss with z_t^CD conditioning
  Output: ACWM adapted to debiased latent space
  
Stage 3: Robot Action Adaptation (3k steps for 2B, 6k for 14B)
  Input: paired (o_t, u_t, o_{t+1}) data
  Loss: bridge g_η(u_t) ≈ sg(μ_φ(o_t, o_{t+1})) + cycle consistency
  Output: ACWM that follows executable robot actions
```

为什么三stage而不是joint training?

1. **Stage 1 isolate LAM debiasing**: 只用video data, 不需要robot action。这让debiasing可以leverage海量unlabeled video。

2. **Stage 2 propagate debiasing到ACWM**: 在引入robot action之前, 让ACWM先适应debiased latent space。如果直接joint train, ACWM可能在训练中"unlearn" debiasing (因为它能从confounders中提取cheap signal)。

3. **Stage 3 align robot action到已debiased space**: 用lightweight MLP bridge $g_\eta$ mapping $u_t \to z_t^{\mathrm{CD}}$。关键设计:
   - $g_\eta(u_t) \approx \mathrm{sg}(\mu_\phi(o_t, o_{t+1}))$: regress到**stop-gradient**的latent。Stop-gradient防止这个loss改变LAM本身, 只train bridge。
   - Auxiliary action readout with cycle-consistency: 鼓励mapped latent retain $u_t$信息, 防止bridge collapse到平凡mapping。

这个pipeline的efficiency非常impressive:
- Stage 1: 1k updates (vs DreamDojo的50k robot action adaptation)
- Stage 2: 2k updates
- Stage 3: 3k (2B) / 6k (14B) updates
- Total: ~6k-9k vs 50k reference — **12× fewer updates**

---

## 6. Evaluation Metrics详解

### 6.1 Visual Fidelity Metrics

- **PSNR**: $\mathrm{PSNR}(x, \hat{x}) = 10\log_{10}\frac{1}{\mathrm{MSE}(x, \hat{x})}$
- **FG-PSNR**: 只在foreground mask内计算
- **SSIM**: structural similarity
- **LPIPS**: learned perceptual similarity (lower better)

### 6.2 FDCE (Foreground Displacement Chamfer Error) — the key metric

这是paper最重要的metric contribution。Standard metrics (PSNR, FVD)无法捕捉"action是否被followed"。

定义: 对于reference foreground point $p_j^s$ 和 generated foreground point $\hat{p}_i^s$ at rollout step $s$, 定义displacement vectors relative to initial frame:
$$a_j^s = p_j^s - p_j^0, \quad \hat{a}_i^s = \hat{p}_i^s - \hat{p}_i^0$$

变量:
- $p_j^0, p_j^s$: reference track $j$ 在初始帧和step $s$的位置
- $\hat{p}_i^0, \hat{p}_i^s$: generated track $i$ 在初始帧和step $s$的位置
- $a_j^s, \hat{a}_i^s$: 相对初始帧的displacement (这是关键: 我们measure motion, **不**measure absolute position)

Average distance between tracks:
$$c_{ij} = \frac{1}{H}\sum_{s=1}^{H}\|\hat{a}_i^s - a_j^s\|_2$$

变量:
- $H$: rollout horizon
- $c_{ij}$: track $i$ 和 track $j$ 的average displacement distance

Symmetric Chamfer distance:
$$\mathrm{FDCE}(\hat{o}, o) = \frac{1}{2N_g}\sum_{i=1}^{N_g}\min_j c_{ij} + \frac{1}{2N_r}\sum_{j=1}^{N_r}\min_i c_{ij}$$

变量:
- $N_g$: generated foreground tracks数量
- $N_r$: reference foreground tracks数量
- $\min_j c_{ij}$: 对每个generated track, 找最近的reference track
- $\min_i c_{ij}$: 对每个reference track, 找最近的generated track

**Intuition**: 这是bidirectional Chamfer distance applied to displacement tracks。它measure **motion geometry** rather than pixel appearance。两个track可以pixel-level不同但motion一致 (例如generated hand在不同位置但move方式相同), 仍然有low FDCE。

为什么用symmetric Chamfer而不是单向? Symmetric form对 $N_g$ 和 $N_r$ 数量差异robust (paper Appendix A)。如果一个model generate很少valid tracks, 单向distance可能看起来好, 但实际上model fail了。Symmetric form catch这种case。

为什么用displacement而不是absolute position? 因为initial frame已经fixed (intervention protocol), 我们只关心相对initial frame的motion, **不**关心绝对位置。这decouples visual fidelity (绝对pixel)和action following (相对motion)。

Implementation details:
- Foreground masks: SAM3 (https://arxiv.org/abs/2511.16719)
- Point tracks: CoWTracker (https://arxiv.org/abs/2602.04877), 一种warping-based tracker
- 最多16个valid foreground anchors per rollout pair
- Anchors seeded inside eroded foreground mask (避免boundary artifacts)
- Tracks with low visibility confidence discarded before scoring

### 6.3 关键insight: PSNR和FDCE weakly correlated

Fig. 6(b)显示: PSNR和FDCE的correlation $r = -0.38$, $R^2 = 0.14$。

这意味着PSNR只explain 14%的FDCE variance。一个rollout可以high PSNR但high FDCE (copy context, ignore action)。所以**pixel metric alone insufficient for controllability evaluation**。这个insight对整个video world model领域都有implication。

参考:
- FVD (Fréchet Video Distance): https://arxiv.org/abs/1812.01717
- MotionPro (用trajectory distance measure motion control): https://arxiv.org/abs/2502.10326
- TAP-Vid (Tracking Any Point benchmark): https://arxiv.org/abs/2211.03726
- TAPIR: https://arxiv.org/abs/2306.08637
- CoTracker: https://arxiv.org/abs/2407.07624

---

## 7. 实验结果深度分析

### 7.1 Latent Action Audit (Table I)

| Diagnostic | DreamDojo LAM | CD-LAM | Reduction |
|------------|---------------|--------|-----------|
| Zero-transition median response | 0.527 | 0.043 | 12.3× |
| Zero-transition absolute norm | 3.119 | 0.226 | 13.8× |
| Horizontal shift (mean/median) | 0.555/0.536 | 0.156/0.096 | 3.6-5.6× |
| Vertical shift (mean/median) | 0.545/0.529 | 0.110/0.064 | 4.9-8.3× |
| Shortcut leakage | 0.151 | 0.014 | 10.8× |

Action-neighbor preservation check: same-primitive pairs from different episodes的cosine similarity 0.132 (DreamDojo) vs 0.131 (CD-LAM) — **几乎不变**。这说明CD-LAM debiases latent space toward embodied transition semantics, 而不是uniformly shrink它 (那样会destroy action information)。

### 7.2 ACWM Rollouts after Stage 2 (Table II)

2B scale:
| Metric | DreamDojo | CD-LAM | Change |
|--------|-----------|--------|--------|
| FDCE (latent rollout) | 34.00 | 19.63 | -42% |
| PSNR | 20.88 | 24.29 | +3.41dB |
| SSIM | 0.780 | 0.827 | +0.047 |
| LPIPS | 0.413 | 0.308 | -0.105 |
| FDCE (target transfer) | 42.74 | 33.81 | -21% |

14B scale:
| Metric | DreamDojo | CD-LAM | Change |
|--------|-----------|--------|--------|
| FDCE (latent rollout) | 40.29 | 29.87 | -26% |
| PSNR | 21.04 | 23.18 | +2.14dB |
| FDCE (target transfer) | 50.27 | 33.22 | -34% |

**Crucial observation**: Baseline从2B到14B FDCE反而**worsened** (34.00 → 40.29, 42.74 → 50.27)。这证明scale alone**不能**解决action following问题。Scale amplifies visual capability但**不**amplify action controllability。

CD-LAM在两个scales都改善, 而且14B CD-LAM在所有metric上都比2B CD-LAM更好。这证明scale和debiasing是**complementary**的: scale在debiased space上发挥更好效果。

### 7.3 Robot Action Adaptation Rollouts (Table III) — main system result

2B:
| Metric | DreamDojo | CD-LAM |
|--------|-----------|--------|
| $\mathrm{FDCE}_{\mathrm{mean}}$ | 12.63 | 8.24 |
| $\mathrm{FDCE}_{\mathrm{med}}$ | 8.15 | 6.75 |
| PSNR | 19.85 | 20.60 |
| $\mathrm{do}(u_t=0)$ FDCE | 10.71 | 5.03 |
| Target transfer FDCE | 24.36 | 22.55 |

14B:
| Metric | DreamDojo | CD-LAM |
|--------|-----------|--------|
| $\mathrm{FDCE}_{\mathrm{mean}}$ | 11.11 | 7.73 |
| $\mathrm{FDCE}_{\mathrm{med}}$ | 8.98 | 5.99 |
| PSNR | 20.01 | 21.01 |
| $\mathrm{do}(u_t=0)$ FDCE | 9.36 | 2.18 |
| Target transfer FDCE | 24.82 | 21.11 |

**Zero-action intervention的巨大改善**: 14B CD-LAM的do($u_t=0$) FDCE是2.18, **比baseline低77%**。这意味着当action是zero时, rollout基本不动, 正好是Fig. 3 baseline lacking的行为。

### 7.4 Robot Action Adaptation Efficiency (Fig. 8)

14B CD-LAM在3k-4k updates内reach DreamDojo 50k reference水平, 在6k final checkpoint surpass it。这是**12× fewer updates**。

这个结果的causal explanation: 因为ACWM不再需要"unlearn" confounded action condition (它从来没学过), 只需要learn正确的action-to-latent mapping。Baseline需要从confounded state中fix, 这个"unlearning"过程expensive。

### 7.5 Debiasing Data Scaling (Table IV)

| Model | PSNR | $\mathrm{FDCE}_{\mathrm{mean}}$ | $\mathrm{FDCE}_{\mathrm{med}}$ |
|-------|------|-------------------------------|-------------------------------|
| DreamDojo | 19.85 | 12.63 | 8.15 |
| CD-LAM-1h | 20.54 | 8.91 | 6.88 |
| CD-LAM-10h | 20.61 | 8.87 | 6.23 |
| CD-LAM-100h | 20.60 | 8.24 | 6.75 |
| CD-LAM-1000h | 20.64 | 7.97 | 6.12 |

1h tier已经capture约80%的1000h tier的FDCE improvement (3.72 of 4.66 points)。这证明benefit主要来自**debiasing mechanism itself**, **不**来自debiasing-data scale。这是一个非常重要的**less-is-more** result: 结构性inductive bias比单纯scale data更efficient。

### 7.6 Per-action breakdown (Fig. 7, Fig. A.1)

CD-LAM在8个action categories上都有improvement (Fig. 7(a))。这说明gain**不**集中在single primitive, 而是generalize across actions。

不过paper也承认: 在少数categories上CD-LAM和baseline持平甚至slightly behind (Fig. A.1)。这是honest reporting, 也指向limitation: debiasing可能在某些action types上over-regularize。

---

## 8. Related Work Landscape

### 8.1 Latent Action Models evolution

- **iLPO** (Imitating Latent Policies from Observation, ICML 2019): https://arxiv.org/abs/1805.07914 — 早期idea, learn latent policies from observation
- **VPT** (Video PreTraining, NeurIPS 2022): https://arxiv.org/abs/2206.11795 — 半监督learn action from unlabeled video
- **Genie** (ICML 2024): https://arxiv.org/abs/2402.15391 — discrete latent action codebook at scale
- **LAPO** (ICLR 2024): https://arxiv.org/abs/2312.10812 — recover latent actions from observation alone
- **LAPA** (ICLR 2025): https://arxiv.org/abs/2410.11758 — discrete latent actions by quantizing inter-frame transitions
- **AdaWorld** (ICML 2025): https://arxiv.org/abs/2505.18834 — transferable latent action condition for fast adaptation
- **Moto** (ICCV 2025): https://arxiv.org/abs/2410.23214 — latent motion token as bridging language
- **IGOR** (2024): https://arxiv.org/abs/2411.00785 — image-goal representations as atomic control units
- **Additively compositional latent actions**: https://arxiv.org/abs/2604.03340
- **Co-evolving latent action world models**: https://arxiv.org/abs/2510.26433
- **MVP-LAM** (cross-viewpoint): https://arxiv.org/abs/2602.03668

这些方法的shared principle: latent action由"什么improves next-frame reconstruction"定义。CD-LAM的**contribution**: 第一个系统分析这种definition的confounding问题并提出debiasing方法。

### 8.2 Criticism of LAMs (并发工作)

- **"What do latent action models actually learn?"** (NeurIPS 2025): 分析LAM的representation问题
- **"Latent action learning requires supervision in the presence of distractors"** (ICML 2025): https://arxiv.org/abs/2505.16293 — 在distractor存在时LAM需要supervision
- **ConLA** (2026): https://arxiv.org/abs/2602.00557 — Contrastive latent action learning from human videos

这些并发工作都识别了similar问题, 但CD-LAM的approach更**系统化**: 提出三个互补objectives, 设计quantitative diagnostic, 并在large-scale ACWM (2B, 14B)上验证。

### 8.3 World Models

- **World Models (Ha & Schmidhuber)**: https://arxiv.org/abs/1803.10122 — 经典World Models paper
- **DreamerV3**: https://arxiv.org/abs/2301.04104 — Mastering diverse domains through world models
- **UniSim**: https://arxiv.org/abs/2310.06114 — Interactive real-world simulators
- **Cosmos** (NVIDIA): https://arxiv.org/abs/2501.03575 — World foundation model platform for physical AI
- **iVideoGPT**: https://arxiv.org/abs/2405.15223 — Interactive VideoGPTs as scalable world models

### 8.4 Embodied Datasets

- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864 — Robotic learning datasets and RT-X models
- **DROID**: https://arxiv.org/abs/2403.12945 — Large-scale in-the-wild robot manipulation dataset
- **Ego4D**: https://arxiv.org/abs/2110.15270 — 3,000 hours of egocentric video
- **Ego-Exo4D**: https://arxiv.org/abs/2404.01986 — First- and third-person skilled activity
- **EgoDex**: https://arxiv.org/abs/2505.11709 — Dexterous manipulation from egocentric video
- **AgiBot World**: https://arxiv.org/abs/2503.06669 — Large-scale manipulation platform

### 8.5 Tracking and Segmentation Tools

- **SAM** (Segment Anything): https://arxiv.org/abs/2304.02643
- **SAM 2**: https://arxiv.org/abs/2408.00714
- **SAM 3**: https://arxiv.org/abs/2511.16719
- **CoTracker**: https://arxiv.org/abs/2407.07624
- **CoWTracker**: https://arxiv.org/abs/2602.04877
- **TAP-Vid**: https://arxiv.org/abs/2211.03726
- **TAPIR**: https://arxiv.org/abs/2306.08637

### 8.6 Causal and Debiased Learning

- **IRM** (Invariant Risk Minimization): https://arxiv.org/abs/1907.02893
- **Shortcut Learning**: https://arxiv.org/abs/2004.07780
- **Causal Confusion in Imitation Learning**: https://arxiv.org/abs/1905.11979

CD-LAM的**novel positioning**: debias一个**conditioning variable** (latent action), 而不是policy的input。Confounding corrupts controllability, 而不是accuracy。这是一个新的causal debiasing setting。

---

## 9. Building Intuition — 几个关键mental models

### 9.1 Mental Model 1: Latent action as sufficient statistic

Reconstruction objective让 $z_t$ 成为 $p(o_{t+1}|o_t, z_t)$ 的sufficient statistic。但sufficient statistic包含**所有predictive信息**, 包括confounders。这就像linear regression中, 如果一个feature和target相关, 即使它不是causal, regression也会用上它。

修复方法: 加structural constraint让 $z_t$ 只encode **causally relevant**信息。CD-LAM的三个loss分别实现:
- $\mathcal{L}_{\mathrm{emb}}$: 只在foreground region要求sufficiency
- $\mathcal{L}_{\mathrm{ctr}}$: 加pairwise structure让similar action cluster
- $\mathcal{L}_{\mathrm{cal}}$: 定义origin让"no action"对应zero latent

### 9.2 Mental Model 2: Latent space geometry

想象latent space $\mathbb{R}^d$。Reconstruction objective只定义了"信息content"但**没**定义geometry。两个transition可以similar $z_t$ 但因为完全不同的reason (一个因为同action, 一个因为同background)。

CD-LAM通过三个loss塑造geometry:
- $\mathcal{L}_{\mathrm{emb}}$: 让 $z_t$ 的dimensions中, 对foreground predictive的维度被amplified
- $\mathcal{L}_{\mathrm{ctr}}$: 让same-primitive transitions在latent space中cluster
- $\mathcal{L}_{\mathrm{cal}}$: 让zero-transition在origin附近, ordinary transitions保持distance

这三个loss合在一起, 把latent space从一个"sufficient statistic的embedding"塑造成一个"action semantic space with calibrated origin"。

### 9.3 Mental Model 3: Causal vs Statistical

Standard LAM training是statistical optimization: minimize reconstruction loss。它找到的是**statistically optimal** $z_t$, 但这个 $z_t$ 可能依赖spurious correlates (背景continuation, scene context)。

CD-LAM引入causal constraints: $z_t$ 应该**只**depend on $A_t$, 而**不**depend on $C_t, V_t$。这通过:
- $\mathcal{L}_{\mathrm{emb}}$: 局部reconstruction, 让background $V_t$ 不进入 $z_t$
- $\mathcal{L}_{\mathrm{ctr}}$: cross-context对比, 让scene $C_t$ 不confound $z_t$ similarity
- $\mathcal{L}_{\mathrm{cal}}$: 定义causal reference point (zero action = zero latent)

### 9.4 Mental Model 4: Less is more的mechanism

为什么1k步LAM debiasing > 50k步robot action adaptation?

Think of it as optimization landscape:
- **Baseline**: ACWM learned confounded $z_t$ representation, robot action $u_t$ 需要navigate一个扭曲的mapping $u_t \to z_t$ where $z_t$ 包含non-actionable dimensions。Optimization landscape是ill-conditioned, 需要很多updates。
- **CD-LAM**: LAM先被debias, $z_t$ space clean。ACWM适应clean space, robot action mapping $u_t \to z_t$ direct。Optimization landscape well-conditioned, 少量updates足够。

这就像: 修好bug vs 在buggy code上workaround。前者省力, 后者费力且fragile。

### 9.5 Mental Model 5: Confounding的geometry

考虑两个transitions $(o_t^a, o_{t+1}^a)$ 和 $(o_t^b, o_{t+1}^b)$, 它们有相同action $A$ 但不同background。

Reconstruction LAM的 $z_t$:
$$z_t^a \approx f(A, C^a, V^a), \quad z_t^b \approx f(A, C^b, V^b)$$

由于 $C^a \neq C^b$ 和 $V^a \neq V^b$, $z_t^a \neq z_t^b$, 即使action相同。这是confounding。

CD-LAM的 $z_t$:
$$z_t^a \approx f(A), \quad z_t^b \approx f(A)$$

由于action相同, $z_t^a \approx z_t^b$。这是debiased。

$\mathcal{L}_{\mathrm{ctr}}$ 显式enforce这个property: same-primitive transitions被pulled together。

---

## 10. Limitations和Open Questions

Paper没有explicitly讨论的:

### 10.1 SAM3 dependency
$\mathcal{L}_{\mathrm{emb}}$ 依赖SAM3 foreground mask质量。在SAM3 fail的case (heavy occlusion, motion blur, transparent objects)上, foreground weighting会错误。可能的fix: co-train SAM3和LAM, 或者用uncertainty-aware mask。

### 10.2 Primitive labels的noise
Primitive labels来自caption的coarse verbs, 对caption质量敏感。如果caption mis-classify或miss verb, contrastive loss会push错方向。可能的fix: confidence-weighted contrastive loss。

### 10.3 Point tracker limitation
FDCE依赖CoWTracker, 在textureless gripper上会drift。Paper说这bias比较toward null (所有model同样受影响), 但绝对值会被inflated。可能的fix: 多个tracker ensemble, 或learned tracker specific for robotic manipulation。

### 10.4 Generalization
Paper只在AgiBot上evaluate robot action adaptation。其他robot platforms (Franka, UR5, Kuka)效果如何? Cross-embodiment transfer如何?

### 10.5 Long-horizon stability
Paper evaluate $H$步rollout (具体horizon在appendix), 但long-horizon (几百步)的error accumulation如何? CD-LAM的debiased latent space在long rollout中是否更stable? 直觉上应该更stable (action跟随更好), 但需要验证。

### 10.6 Compositional actions
12-way primitive是coarse的。复杂action (例如"screw in then tighten")如何被represent? Paper [27]的additively compositional latent actions是relevant direction: https://arxiv.org/abs/2604.03340

### 10.7 三个 $\lambda$ 的sensitivity
$\lambda_{\mathrm{ctr}}(k)$ 和 $\lambda_{\mathrm{cal}}$ 的具体schedule和值在appendix。这些hyperparameter的sensitivity如何? Robustness across datasets?

### 10.8 Comparison with supervised action prediction
如果直接用supervised action prediction (有action label的数据)训练LAM, 效果如何? 这是upper bound。Paper没做这个比较, 因为supervised data不够scale, 但作为ceiling reference有用。

---

## 11. Connections to Karpathy's Broader Vision

### 11.1 Software 2.0/3.0

Karpathy的"Software 2.0" idea: 用learned function替代hand-coded rules。LAM是这一idea的延伸: 用learned latent action替代hand-coded action space。CD-LAM加inductive bias, 是"Software 2.0 with structure"的实例。

"Software 3.0" (用natural language prompt program): 这篇paper的primitive labels来自caption, 是一个soft的instance — 用caption semantics structure latent space。

### 11.2 之前teaching中的相关intuitions

Karpathy的CS231n和build-nanogpt中强调:
- **Inductive bias matters**: 神经网络architectures的inductive bias重要, 不能完全scale解决。CD-LAM的三个loss就是inductive bias的实例化。
- **Optimization landscape conditioning**: 好的method让optimization easier。CD-LAM让robot action adaptation的landscape更well-conditioned。
- **Diagnostic visualization**: 不要只看final metric, 要看intermediate representations。Paper的LAM audit (Table I)就是这种diagnostic。
- **Causal thinking**: shortcut learning是deep learning的pervasive problem。CD-LAM是causal thinking对representation learning的应用。

### 11.3 和LLM representation的类比

LLM中也有similar confounding问题: token representation可能confound syntax和semantics, sentence embedding可能confound topic和sentiment。Probing studies经常发现这种confounding。

CD-LAM的方法可能inspire LLM debiasing: contrastive loss用semantic labels, calibration loss定义"neutral" reference, foreground-style weighting强调causally-relevant tokens。

参考:
- Probing classifiers: https://arxiv.org/abs/1902.08966
- Disentangled representations: https://arxiv.org/abs/1812.02230
- Information Bottleneck: https://arxiv.org/abs/1703.00810

---

## 12. 关键Takeaways汇总

1. **Reconstruction objective对latent action是necessary but not sufficient**: 它确保predictive sufficiency但**不**确保action purity。

2. **Confounding是LAM的核心问题**: action-irrelevant factors ($C_t, V_t$) leak into latent action space, confound downstream ACWM。

3. **三个debiasing objectives互补**:
   - $\mathcal{L}_{\mathrm{emb}}$: 控制foreground fidelity
   - $\mathcal{L}_{\mathrm{ctr}}$: 控制latent structure
   - $\mathcal{L}_{\mathrm{cal}}$: 控制origin和camera-shift response

4. **Less is more**: 1k步LAM debiasing > 50k步robot action adaptation。Targeted debiasing比scale robot action adaptation更efficient。

5. **Scale不能替代debiasing**: 14B baseline的FDCE反而比2B差。但scale和debiasing**complementary**: 14B CD-LAM全面提升。

6. **FDCE是proper controllability metric**: PSNR和FDCE weakly correlated ($R^2 = 0.14$)。Pixel metric insufficient for action following evaluation。

7. **Causal thinking对representation learning重要**: debiasing conditioning variable (latent action)比debiasing input更关键。

8. **三stage pipeline的设计哲学**: debiasing在upstream (LAM)完成, 避免downstream (ACWM)需要unlearn。

9. **Diagnostic > downstream metric**: ablation中zero-trans calibration去掉后FDCE反而lower, 但camera-shift diagnostic fail。看causal diagnostic而不是downstream metric。

10. **12-way primitive labels是weak supervision**: 不需要executable robot action, 只用verb-level categories就能structure latent space。

---

## 13. 可能的后续方向

### 13.1 Auto-discovering primitives
12-way primitive是手工defined的。能否auto-discover primitive set from data? 类似vector quantization或clustering, 但要保证action-consistent。

### 13.2 Hierarchical latent actions
Coarse primitive + fine-grained action decomposition。例如"scoop" primitive下的different scoop trajectories。可能是hierarchical VQ-VAE的application。

### 13.3 Cross-embodiment transfer
不同robot embodiments (Franka, UR5, human hand)如何share latent action space? Bridge $g_\eta$ 是否embody-specific? 多个bridge share一个debiased LAM?

### 13.4 Closed-loop control with CD-LAM
Paper evaluate open-loop action following。Closed-loop (model predictive control with CD-LAM as forward model)的效果如何? Debiased latent space是否让planning更stable?

### 13.5 Causal discovery of confounders
Paper assume $C_t$ (scene)和 $V_t$ (visual)是confounders。能否auto-discover confounders from data? Causal discovery + representation learning的结合。

### 13.6 Theoretical analysis
CD-LAM的三个loss的interaction是否能theoretically分析? Identifiability of debiased latent action under certain assumptions? 这可能connect到ICA (Independent Component Analysis)和disentangled representation theory。

参考:
- ICA and disentanglement: https://arxiv.org/abs/2010.02607
- Theory of contrastive learning: https://arxiv.org/abs/2105.13966
- Identifiability of VAE: https://arxiv.org/abs/1907.04809

---

## 14. Final Reflection

这篇paper是 **causal thinking meets deep representation learning** 的clean实例。它不引入新的大architecture, 不需要massive compute, 只是carefully分析bug然后针对性fix。这种"小而精"的method在当前scale-everything的潮流中很refreshing。

最inspiring的部分是**diagnostic-first**的方法论: 先quantify bug (Table I的audit), 再design fix (三个losses), 最后verify fix在upstream (audit)和downstream (rollouts)都work。这种rigor是deep learning research需要的。

对Karpathy可能特别resonant的是: 这正是"micrograd-style"的paper — 不复杂, 但insightful。三个losses每个都intuitive, 合在一起解决一个具体问题。这种clarity在arxiv上越来越rare。

希望这个dive能build your intuition: **reconstruction is necessary but not sufficient for controllable latent codes**, **causal confounding corrupts conditioning variables**, 和**targeted debiasing can be 12× more efficient than scale**。

参考汇总链接:
- Paper: https://arxiv.org/abs/2602.06949 (DreamDojo baseline)
- Project lead: franciskunzhou@gmail.com
- Aether AI lab
- UC San Diego
- Code release mentioned in paper (check author page)

如果对某个section想更deep dive (例如具体推导 $\mathcal{L}_{\mathrm{KL-fb}}$ 的free-bits formula, 或者更详细的CoWTracker architecture, 或者具体的 $\lambda_{\mathrm{ctr}}(k)$ schedule), 让我知道, 我可以expand更细节。
