---
source_pdf: OmniHumanoid Streaming Cross-Embodiment Video.pdf
paper_sha256: 9e4860b1b4d0508780804c6c4a37abd23653a92720c48c9fc8fd35308806a0e2
processed_at: '2026-08-05T23:20:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OmniHumanoid 用人话说

好，Andrej，我把学术腔都扔掉，咱们像白板前聊天那样过一遍这篇 paper。

---

## 一句话讲它在干嘛

你手头有一堆人干活儿的视频，你想把它"翻译"成某个机器人干同样活儿的视频。这篇 paper 就是干这个事儿的，而且换机器人不用重训大模型，只要拿几十段那个机器人的随便视频，训一个小小的 LoRA 就行。

就跟你 Tesla 想用 YouTube 上几千小时人类操作视频，给 Optimus 生成训练数据一个意思。每出一个新 robot，不用重新 teleop 采数据、不用重新跑整个 video model，贴个小模块就 adapt 了。

---

## 为什么这事儿难

你想啊，同一段"人拿杯子"的视频，retarget 到 Unitree G1 上，问题立刻冒出来：

- 人的手有 27 个 DOF，G1 的夹爪可能就 1 个 DOF——动作对不上
- 人腿和 robot 腿的关节限位完全不同，蹲下去的角度都不一样
- 人有皮肤有衣服，robot 有金属外壳有连杆——渲染上完全是两码事
- 人蹲下的时候腿挡住后面的桌子，robot 蹲下的时候因为腿短一截，遮挡关系全变了

所以 motion 和 appearance 在像素层面是死死缠在一起的。你没法简单地说"保留动作、换皮肤"，因为换皮肤本身就会改变动作怎么呈现。

更麻烦的是，通用 video editing 模型（Runway、Kling 这些）根本不认识 robot 的结构。让它画个 robot 拿杯子，它可能给你画个"长着机器人颜色的手"——关节对不上、连杆穿过桌子、夹爪莫名其妙开合。SSIM 直接掉到 0.6575 那种惨烈水平，就是 Table 1 里 Runway Gen4 的真实表现。

还有个效率问题。标准 DiT video model 要 50 步 denoising，渲染一个 720p 的视频，单 H200 上一秒钟 0.1 帧。你想批量生成几十万条 robot 操作视频，这速度根本没法用。

Reference: Phantom 那条线尝试用 rendered overlay 解决，但 overlay 碰到 self-occlusion 就崩——https://arxiv.org/abs/2503.00779

---

## 他们怎么拆这个问题

核心 insight 特别朴素：**动作是跨 robot 通用的，外观是每个 robot 独有的**。既然统计结构不一样，那就别让它们住一个屋子。

具体怎么做呢？他们搞了三条 token 流：

- $X^{\mathrm{text}}$：文本提示词
- $X^{\mathrm{den}}$：要生成的 robot 视频 latent（这条要 denoise）
- $X^{\mathrm{cond}}$：source 人类视频（这个当 condition）

LoRA 只挂在 denoising branch 上，conditioning branch 用共享的 base projection，**一个 embodiment 参数都不让它碰**。

到这儿还只是"分开住"。真正聪明的是后面那个单向门：

$$
\mathcal{M}(\mathrm{den} \to \mathrm{cond}) = 1, \quad \mathcal{M}(\mathrm{cond} \to \mathrm{den}) = 0
$$

翻译成人话：
- denoising token 可以读 conditioning token——"我要生成 robot，我需要知道人类在干啥动作"
- conditioning token 不能读 denoising token——"我只是个动作参考，我不想知道 robot 长啥样"

这就等于在 attention 层面强制了一个因果方向：**动作决定外观，外观不污染动作**。

为什么这个设计这么关键？看 ablation Table 2——把单向门拆掉，Embodiment score 从 8.43 直接崩到 2.53。崩得这么彻底是因为模型走捷径了：conditioning branch 偷偷从 den branch 学到了"哦原来人是长这样的"，于是 shared motion model 实际上变成了"see-through 当前 embodiment"的 motion model。你换个新 robot，它还在那儿渲染上个 robot 的样子，完蛋。

这跟我之前看过的 MotionDirector (https://arxiv.org/abs/2310.08465) 思路一脉相承，都是把 motion 和 appearance 物理隔离。但 OmniHumanoid 更激进，直接用 attention mask 砍掉反向信息流，不光分两个 module，而是强制单向因果关系。

---

## 训练怎么分两步走

既然架构上 factorize 了，训练自然也分两阶段。

**Stage I：先教每个 robot 长啥样**

对每个见过的 robot $e$，拿几十段它自己的随便视频 $\mathcal{U}_e$，冻结 DiT backbone，只训 LoRA $\Phi_e$。loss 就是标准 video diffusion denoising：

$$
\mathcal{L} = \mathbb{E}[\| \epsilon - \epsilon_\theta(x_t, t) \|^2]
$$

$x_0$ 是干净 video latent，$x_t$ 是加噪版，$t$ 是 timestep，$\epsilon$ 是 Gaussian noise。这个阶段不需要 paired data，不需要知道人类在干啥，就是让 LoRA 学会"这个 robot 长这样、动起来这样"。

LoRA 的数学形式：
$$
\Delta W_\bullet^{(e)} = B_\bullet^{(e)} A_\bullet^{(e)}
$$
$A$ 把 $d$ 维压到 $r$ 维（$r=64$），$B$ 再还原回 $d$ 维。低秩约束本身也是 inductive bias，逼着 LoRA 只学 embodiment-specific 的低维特征，别去抢 shared model 该学的活儿。

**Stage II：教共享模型怎么跨 robot 转动作**

冻结所有 LoRA，在 paired dataset $\mathcal{D}_{\mathrm{pair}}$ 上训 shared motion model。每个 batch 带一个 target embodiment $e$，激活对应的 $\Phi_e$，只更新共享参数。

有个小 trick 叫 **rolling LoRA loading**：因为每个 batch 的 target robot 不一样，所以每次迭代切换激活的 LoRA。这就像 multi-task learning 里的 task balancing——如果一直训同一个 robot，shared encoder 会偷偷把那个 robot 的 specific feature 当 signal 学进去，rolling 让这种捷径没法稳定形成。

**给新 robot 做 adaptation 的时候**

完全不动 shared motion model。新 robot $e^*$ 来了：
1. 实例化一个新 LoRA $\Phi_{e^*}$
2. 拍几十段这个 robot 在任意场景干任意活的视频
3. 训这个 LoRA
4. 推理时：人类视频走 condition branch，新 robot 的 LoRA 走 den branch

就这么多。不需要任何 paired human-robot 视频。这个 paired-free 特性是它能 scale 的根本——每加一个 robot 的边际成本接近零。

---

## 怎么把 50 步压到 4 步

这是第二个大贡献，跟你 Tesla 做 autonomous driving 时关心的 real-time inference 直接相关。

问题在哪儿：DiT v2v 默认要 50 步 denoising，而且 bidirectional attention 是 $O(N^2)$ 全序列的。长视频完全扛不住。

他们的方案是 **streaming distillation**，把昂贵的 bidirectional teacher 蒸馏成 causal streaming student。

### Token 重排

把 reference image、source condition、target 三个 token stream 拼成一条 interleaved 长序列：

$$
[\mathrm{ref} \mid \mathrm{cond}_0 \mid \mathrm{tgt}_0 \mid \cdots \mid \mathrm{cond}_M \mid \mathrm{tgt}_M]
$$

$\mathrm{ref}$ 是 reference image tokens（target robot 的 appearance anchor），$\mathrm{cond}_i$ 是第 $i$ 个 chunk 的 source video tokens，$\mathrm{tgt}_i$ 是要生成的第 $i$ 个 chunk 的 target latent。

关键约束：target chunk $i$ 只能 attend 到 ref、之前所有 chunk、和当前的 $\mathrm{cond}_i$，未来的全部 mask 掉。这是 block-wise causal attention，能做 KV cache 的 autoregressive rollout——每生成一个新 chunk，前面的 KV cache 都复用，不用重新算整个序列。

### 蒸馏的三个 loss

$$
\mathcal{L}_{\mathrm{stream}} = \mathcal{L}_{\mathrm{DSM}} + \lambda_{\mathrm{vsd}} \mathcal{L}_{\mathrm{VSD}} + \lambda_{\mathrm{gan}} \mathcal{L}_{\mathrm{GAN}}
$$

三个 loss 各干一件事：

**$\mathcal{L}_{\mathrm{DSM}}$ (Denoising Score Matching)**：student 在 causal mask 下做 teacher-forcing denoising，保证基础预测能力不掉。就是标准 diffusion 训练 loss，让 student 还能"听懂" noise schedule。

**$\mathcal{L}_{\mathrm{VSD}}$ (Video Score Distillation)**：把 frozen bidirectional teacher 当 score function guide，让 student 的 autoregressive rollout 分布对齐 teacher。这相当于 ProlificDreamer 那套 VSD 的 video 版本——https://arxiv.org/abs/2305.16291。作用是防止 student 在 few-step regime 走偏，因为 teacher 知道"完美视频长什么样"，student 通过匹配 teacher 的 score field 来纠正自己的轨迹。

**$\mathcal{L}_{\mathrm{GAN}}$**：few-step diffusion 的通病是 blurry，GAN loss 拉回 high-frequency detail。这跟 SDXL Turbo、DMD 的设计一致——https://arxiv.org/abs/2311.18828。

$\lambda_{\mathrm{vsd}}, \lambda_{\mathrm{gan}}$ 是权重，paper 没给具体值。

### 代价和收益

Ablation Table 2 给了明确 trade-off：

| 模型 | FPS | PSNR | Motion | Embod | BG |
|---|---|---|---|---|---|
| Teacher (50 步, bidirectional) | 0.10 | 25.47 | 9.06 | 8.43 | 9.94 |
| Causal student (4 步, 无 self-forcing) | 4.96 | 23.35 | 8.82 | 8.07 | 9.58 |
| Full streaming student | 4.96 | 23.34 | 8.90 | 8.09 | 9.64 |

PSNR 掉了约 2 dB，FPS 涨了 50 倍。5 FPS @ 720p on 单 H200，已经接近"能交互"的范畴了。对 data generation 这种离线批量场景，完全够用；对 real-time control 还差，但离那个目标近了一大步。

---

## 数据集是怎么"造"出来的

这 paper 的一个隐藏亮点是数据构造。他们用 Unity 自己渲染了一个 motion-aligned paired dataset。

具体操作：
- 从 Humoto motion library 拿 700+ 段人类动作（manipulation、locomotion、daily activity）
- 选 10 个 humanoid asset（5 个 robot + 5 个 digital human）
- 在 Blender 里把所有 skeleton 对齐到统一 topology
- 在 Unity 里做 retargeting，把同一段动作"刷"到所有 10 个 embodiment 上
- 在 100 个 scene 里渲染，每个 scene 里只换 humanoid asset，camera 和 scene 布局不变

结果是：每个动作有 10 个版本的视频，**motion 和 scene 完全一致，只有 embodiment 外观不同**。这就是 controlled supervision——你明确知道"视频之间的差异 100% 来自 embodiment"，所以模型必须学到 factorization。

总共 7,200 个 paired training samples。Unitree G1 整个 hold out 不参与训练，只在测试时用 50 个 motion-aligned videos 评估，所以能算 PSNR/SSIM 这种 pixel-level metric。

这种 synthetic evaluation 的好处是有 ground truth，real-world benchmark 就只能用 Gemini-3 Flash 当 VLM evaluator（Appendix A 给了完整 prompt）。Real-world benchmark 上 50 个 in-the-wild videos，覆盖厨房、车库、实验室、剧场这些场景。

---

## 效果到底咋样

### 量化对比

Synthetic Held-out Benchmark（有 ground truth）：

| Method | PSNR | SSIM | MSE | Motion | Embod | BG | Overall |
|---|---|---|---|---|---|---|---|
| Kling O1 | 22.70 | 0.8951 | 0.0067 | 8.06 | 6.94 | 9.52 | 7.08 |
| Kling O3 | 22.76 | 0.8959 | 0.0065 | 8.76 | 7.90 | 9.32 | 7.42 |
| Runway Gen4 | 18.83 | 0.6575 | 0.0181 | 7.26 | 7.50 | 8.14 | 7.22 |
| Wan2.1-VACE | 22.44 | 0.8599 | 0.0066 | 6.40 | 5.88 | 8.68 | 6.22 |
| X-Humanoid | 23.03 | 0.8891 | 0.0057 | 8.94 | 8.04 | 9.78 | 7.53 |
| **OmniHumanoid** | **25.47** | **0.9039** | **0.0033** | **9.06** | **8.43** | **9.94** | **7.92** |

几个直觉解读：
- PSNR 比 X-Humanoid 高 2.4 dB，MSE 是它的一半，structural error 大幅减少
- Embodiment score 8.43 比 Kling O3 的 7.90 高——即使 Kling 是商用大模型，zero-shot reference transfer 还是打不过 paired-free LoRA
- Runway Gen4 的 SSIM 0.6575 太惨了，说明通用 video editor 根本 hold 不住 robot 结构

Real-world Benchmark（无 ground truth，VLM 评分）：

| Method | Motion | Embod | BG | Overall |
|---|---|---|---|---|
| Kling O1 | 7.49 | 8.46 | 9.91 | 8.53 |
| Kling O3 | 7.47 | 8.34 | 9.82 | 8.21 |
| Runway Gen4 | 6.79 | 5.07 | 8.61 | 7.22 |
| Wan2.1-VACE | 5.60 | 5.65 | 7.85 | 6.45 |
| **OmniHumanoid** | **8.47** | **8.56** | **9.95** | **8.39** |

Real-world 上 OmniHumanoid 的 BG consistency 9.95 几乎满分——streaming causal attention 因为每个 chunk 都明确 attend 到 ref token，背景不容易 drift。

User study (Table 3) 更说明问题：72.7% 的人觉得 OmniHumanoid 的 motion fidelity 最好，65.7% 觉得 embodiment 最好，Kling O3 分别只有 12.1% 和 16.2%。人眼对 motion artifact 的 sensitivity 比 VLM 高得多，user study 的偏好差距比 VLM 评分更大。

---

## 这东西跟你 Tesla Optimus 有啥关系

直接说吧，这 paper 给你这种"每个 robot 都要重新采数据"的痛提供了一个干净的 path：

**未来想加一个新 robot，流程变成：**
1. 拍这个新 robot 几十段随便干活的视频（不用 teleop，不用配对人类视频）
2. 训一个小 LoRA（rank 64，几个 GPU hour）
3. 共享 motion model 完全不动
4. 把你手头几千小时 YouTube 人类操作视频"翻译"成这个新 robot 的视频
5. 用这些 synthetic video 喂下游 policy

边际成本接近零。每加一个 robot，不用重新 teleop、不用重新训整个 video model、不用采 paired data。

这跟你熟悉的 cross-embodiment policy learning（RT-X、Octo 那条线）哲学完全一致：shared trunk 学通用策略，embodiment-specific head 做 adapt。只不过 OmniHumanoid 是把这个 idea 用在了 video generation 这一层。

Reference: RT-X https://robotics-transformer-x.github.io/ ，Octo https://octo-models.github.io/

---

## 我会追问 paper 作者的几个问题

1. **Motion retargeting 的 information loss 怎么处理？** Humoto 是 human mocap，retarget 到 G1 时 wrist 从 3-DOF 压到 1-DOF，这个 loss 会不会让模型学到"假"的 motion prior？比如 G1 永远学不到 human wrist 的 rotation，shared motion model 是不是会被这个 systemic loss 污染？

2. **Rolling loading 真的能完全消除 embodiment bias 吗？** 如果某个 robot 的 unpaired video 数量明显多于其他，Stage I 之后那个 LoRA 训得更充分，Stage II 里它的 gradient signal 会不会主导 shared model？

3. **Long-horizon drift 数据呢？** Autoregressive rollout 一定有 error accumulation。paper 没给 >10s 视频的 fidelity 曲线。我猜长 horizon 上 motion drift 会明显，这跟 LLM autoregressive 生成的退化是同构的问题。你 Tesla 做 driving prediction 也遇到过这个吧——长 horizon planning 怎么 hold 住。

4. **5 FPS 够 interactive control 用吗？** 对 data generation 够，对 closed-loop control 远远不够。要到 30 FPS real-time，可能需要 consistency model 那种 1-step distillation，但 1-step 在 video 上 temporal coherence 几乎保不住。

5. **Real-world 上的 motion score 8.47 比 synthetic 9.06 低 0.6，这个 gap 是 sim-to-real 还是 motion complexity？** 如果是 sim-to-real，说明 synthetic 训练的 motion prior 没完全 transfer 到真实场景的 lighting/occlusion 分布。

6. **LoRA 之间会"串味儿"吗？** 10 个 embodiment LoRA 同时存在，inference 时只激活一个。但如果两个 robot 形态相近（比如两个双足 humanoid），LoRA 学到的 appearance prior 会不会有 overlap？这会不会让 shared model 在两个相近 embodiment 之间 confuse？

---

## 一句话总结

这篇 paper 干净地把"跨 robot 视频生成"这件事儿拆成了**通用动作学习 + 每个机器人小模块适应**，用单向 attention mask 强制因果方向，用 streaming distillation 把 50 步压到 4 步，用 Unity 渲染的 motion-aligned 数据做 controlled supervision。对你 Tesla 这种要不断加新 robot embodiment 的场景，这是个能直接 scale 的 path。

代码在 https://github.com/showlab/OmniHumanoid ，前作 X-Humanoid 在 https://arxiv.org/abs/2512.04537 ，建议两篇对比看演进路径。

---

补充几个相关工作的 link，方便你深挖：
- Wan2.2 base model: https://arxiv.org/abs/2503.20314
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- AnimateDiff (temporal module 解耦思路的源头): https://arxiv.org/abs/2307.04725
- MotionDirector (dual-path LoRA 解耦 motion/appearance): https://arxiv.org/abs/2310.08465
- DMD (Distribution Matching Distillation): https://arxiv.org/abs/2311.18828
- ProlificDreamer / VSD: https://arxiv.org/abs/2305.16291
- Phantom (rendered overlay baseline): https://arxiv.org/abs/2503.00779
- Masquerade (data-editing pipeline): https://arxiv.org/abs/2508.09976
- EgoVLA (egocentric human-robot joint training): https://arxiv.org/abs/2507.12440
- PH2D (Humanoid Policy ∼ Human Policy): https://arxiv.org/abs/2503.13441
- Cosmos Policy (NVIDIA video model → visuomotor): https://arxiv.org/abs/2601.16163
- Unified Video Action Model: https://arxiv.org/abs/2503.00200
- Mitty (另一个 human-to-robot video generation): https://arxiv.org/abs/2512.17253
- H2R-Grounder (paired-data-free physical grounding): https://arxiv.org/abs/2512.09406
- Sora review: https://arxiv.org/abs/2402.17177

---

# OmniHumanoid 深度解析

Andrej，这篇 paper 我觉得挺对你的胃口——它处理的恰好是你 Tesla Optimus 时代一直在琢磨的问题：**如何把人类视频这种廉价的 supervision signal，convert 成任意机器人 embodiment 能用的 observation**。下面我从 intuition、architecture、训练动力学、distillation、数据和实验几个角度过一遍，尽量把你可能会问的细节都铺出来。

---

## 1. 问题本质：为什么 cross-embodiment video generation 难

Embodied intelligence 社区一直在纠结一个根本性的数据 scaling 问题。我们手头有海量的人类视频（YouTube 上几十万小时），但每种 humanoid robot（Unitree G1、Atlas、Figure 02、Tesla Optimus）的 demonstration 都极其昂贵，每换一个新 robot 就要重新 teleop 采集。

**Cross-embodiment video generation** 想做的事情：给一段 human video $V^{\mathrm{src}}$，再指定一个 target embodiment $e$，输出一段 robot video $\hat{V}^e$，既保留 source 的 motion/scene dynamics，又渲染出 robot 的 morphology。

这里有一个数学上的 awkward 之处——**motion 和 morphology 在像素空间是 deeply entangled 的**。同一个"挥手"动作，人和 robot 的关节长度、自由度数、关节限位、自遮挡结构完全不同，retarget 之后像素层面几乎找不到一一对应。这也是为什么 Phantom 那种 rendered overlay 方法在复杂场景会崩，因为 overlay 解决不了 self-occlusion 和深度排序。

paper 列出四个 obstacles：
1. motion 和 embodiment-specific geometry/kinematics 纠缠
2. paired data 对每个新 robot 都要重新采集
3. 通用 video editing 模型 hold 不住 high-DOF robot 的 structural identity
4. v2v 生成太慢（50 步 denoising），做不了 interactive 或大规模 data production

Reference: 之前 X-Humanoid (arXiv:2512.04537) 是 showlab 自己 group 的工作，OmniHumanoid 是它的升级版。可以对比看：https://arxiv.org/abs/2512.04537

---

## 2. TAPE 原则：设计哲学

作者把方法的核心压缩成 **TAPE**：
- **T**ransferable motion
- **A**daptation paired-free
- **P**reservation of embodiment
- **E**fficiency (streaming)

这四条原则背后其实是同一个核心 hypothesis：**motion 的统计结构是 cross-embodiment invariant 的，而 appearance/morphology 是 embodiment-specific 的**。如果你承认这个 factorization，那合理的做法就是把两套参数物理隔离。

这个思路让我想起 MotionDirector (ECCV 2024) 的 dual-path LoRA——它把 "how to move" 和 "what to look like" 分到两条 LoRA 路径。OmniHumanoid 把这个 idea 从"个性化 motion"扩展到了"cross-embodiment motion transfer"，关键是加了 unidirectional 信息流来强化这个解耦。

Reference: MotionDirector https://arxiv.org/abs/2310.08465

---

## 3. Architecture 核心：Unidirectional Motion-Appearance Decoupling

这是这篇 paper 最聪明的地方，值得展开讲。

### 3.1 三条 token stream

DiT backbone 里有三组 token：
$$
X^{\mathrm{text}}, \quad X^{\mathrm{den}}, \quad X^{\mathrm{cond}}
$$
- $X^{\mathrm{text}}$：文本 prompt 的 tokens
- $X^{\mathrm{den}}$：要 denoise 的 target latent tokens（要生成的 robot video）
- $X^{\mathrm{cond}}$：source video 的 conditioning tokens（人类视频）

对应的 attention projections：
$$
Q^b = X^b W_Q, \quad K^b = X^b W_K, \quad V^b = X^b W_V \tag{3}
$$
其中 $b \in \{\mathrm{text}, \mathrm{den}, \mathrm{cond}\}$。这里 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ 都是 shared 的 base projections，$d$ 是 token 维度。

### 3.2 LoRA 只挂在 denoising branch

对每个 embodiment $e$，引入 LoRA $\Phi_e$：
$$
W_\bullet^{\mathrm{den}, e} = W_\bullet + \Delta W_\bullet^{(e)}, \quad \Delta W_\bullet^{(e)} = B_\bullet^{(e)} A_\bullet^{(e)}, \quad \bullet \in \{Q, K, V\} \tag{4}
$$
变量含义：
- $A_\bullet^{(e)} \in \mathbb{R}^{r \times d}$：down-projection，把 token 从 $d$ 维压到 $r$ 维
- $B_\bullet^{(e)} \in \mathbb{R}^{d \times r}$：up-projection，还原回 $d$ 维
- $r \ll d$：rank，paper 用 $r=64$
- 上标 $(e)$ 表示是 embodiment-specific 的
- 下标 $\bullet$ 表示对 $Q, K, V$ 三个 projection 都加 LoRA

**关键约束**：text 和 conditioning branch 用 shared base projection，**不加 LoRA**。这意味着 embodiment knowledge 完全没有通路渗透到 conditioning branch。

### 3.3 Asymmetric attention mask（最关键的设计）

$$
\mathcal{M}(\mathrm{den} \to \mathrm{cond}) = 1, \quad \mathcal{M}(\mathrm{cond} \to \mathrm{den}) = 0 \tag{5}
$$

读法：
- $\mathcal{M}(\mathrm{den} \to \mathrm{cond}) = 1$：denoising tokens 可以 **attend to** conditioning tokens（visible）
- $\mathcal{M}(\mathrm{cond} \to \mathrm{den}) = 0$：conditioning tokens **不能 attend to** denoising tokens（masked）

这是一个 **单向门**：denoising branch 读 source motion 信息，但 conditioning branch 看不到 embodiment rendering。这等于在 attention 层面强制了"motion 决定 appearance，appearance 不污染 motion"的因果方向。

为什么这很重要？看 ablation Table 2：
- **没有 decoupling**：Embodiment score 从 8.43 崩到 2.53，Motion 从 9.06 掉到 6.35
- **有 decoupling**：两者都高

这个崩塌的直觉解释是：如果不隔离，conditioning branch 会从 denoising branch 学到 embodiment-specific appearance prior，于是 shared motion model 实际上变成"see-through embodiment"的 motion model，换 robot 就废了。同时 denoising branch 也会被 conditioning branch 的 motion residual 干扰，导致 appearance 渲染时偷偷注入"人手"的 prior。

Figure 6 里的 qualitative 对比很说明问题：baseline 会出现 "Wrong Details"（比如把 robot 的关节渲染成人的手指）和 "Wrong Motion"（motion 漂移）——这两个 failure mode 同时出现，恰好就是 entanglement 的症状。

Reference: AnimateDiff 的 Temporal Module 也是类似思路——把 temporal 信息注入冻结的 spatial model，避免空间 identity 被时间学习干扰。https://arxiv.org/abs/2307.04725

---

## 4. Two-Stage Training + Paired-Free Adaptation

### Stage I: Embodiment Video LoRA Pretraining

对每个 seen embodiment $e$，只用 **unpaired videos** $\mathcal{U}_e$ 训练 $\Phi_e$。DiT backbone 冻结，只更新 LoRA 参数，目标是 standard video diffusion denoising：
$$
\mathcal{L}_{\mathrm{denoise}} = \mathbb{E}_{t, \epsilon, x_0} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$
这里 $x_0$ 是 clean video latent，$x_t$ 是加噪版本，$t$ 是 timestep，$\epsilon$ 是 Gaussian noise，$\epsilon_\theta$ 是模型预测的 noise。

**这个阶段不需要 paired motion data**，所以每个 robot 只要几十段视频就能训。LoRA 只挂在 den branch，所以它学的是"这个 robot 长什么样、怎么动"，不是"motion 怎么转"。

### Stage II: Shared Motion Transfer Training

冻结所有 $\Phi_e$，在 paired dataset $\mathcal{D}_{\mathrm{pair}}$ 上训练 shared motion model。对每个 batch sample $(V^{\mathrm{src}}, V^{\mathrm{tgt}}, e)$，激活对应的 $\Phi_e$，只更新 shared backbone 的 motion transfer 参数。

**Rolling LoRA loading strategy** 是这里的小 trick：因为不同 batch 的 target embodiment 不同，所以每个 iteration 切换 active LoRA。效果是 shared model 不会对某一个 embodiment 过拟合——它学到的是"motion 这件事本身"，而不是"motion 在 Unitree G1 上长什么样"。

直觉上，这就像在 multi-task learning 里做 task sampling 平衡。如果一直训同一个 embodiment，shared encoder 会偷偷把 embodiment-specific feature 当 signal 学进去；rolling 让这种"捷径"无法稳定形成。

### Adaptation to Unseen Embodiments

新 robot $e^*$ 来了之后：
1. 实例化新 LoRA $\Phi_{e^*}$
2. 用几十段 unpaired video $\mathcal{U}_{e^*}$ 训 $\Phi_{e^*}$
3. Shared motion model **完全冻结**
4. Inference 时：source video 走 conditioning branch，target robot 的 LoRA 走 den branch

这里 paired-free 是真的 paired-free——你不需要给新 robot 采集任何 paired human-robot video。这把 adaptation cost 从"采 teleop 数据 + 训整个模型"压到"采几十段随便的视频 + 训 LoRA"。

---

## 5. Streaming Distillation：从 50 步到 4 步

这是 paper 的第二个核心贡献，跟你做 autonomous driving 时关心的 real-time inference 直接相关。

### 5.1 问题

Standard DiT v2v 要 50 步 denoising，对 720p 长视频完全做不动。即使有 cache 优化，bidirectional attention 也要 O(N²) 全序列 attention，长 horizon 不可扩展。

### 5.2 Causal super-chunk layout

把 reference、source condition、target 三个 token stream 重排成 interleaved 序列：
$$
[\mathrm{ref} \mid \mathrm{cond}_0 \mid \mathrm{tgt}_0 \mid \cdots \mid \mathrm{cond}_M \mid \mathrm{tgt}_M] \tag{6}
$$

变量含义：
- $\mathrm{ref}$：reference image tokens（target robot 的 appearance anchor）
- $\mathrm{cond}_i$：第 $i$ 个 chunk 的 source video condition tokens
- $\mathrm{tgt}_i$：第 $i$ 个 chunk 的 target latent tokens
- $M$：总 chunk 数

target chunk $i$ 只能 attend to ref、之前所有 chunks、以及当前 $\mathrm{cond}_i$，**未来 chunks 被 mask 掉**。这是 block-wise causal attention，可以做 KV cache 的 autoregressive rollout。

### 5.3 Two-stage distillation objective

总 loss：
$$
\mathcal{L}_{\mathrm{stream}} = \mathcal{L}_{\mathrm{DSM}} + \lambda_{\mathrm{vsd}} \mathcal{L}_{\mathrm{VSD}} + \lambda_{\mathrm{gan}} \mathcal{L}_{\mathrm{GAN}} \tag{7}
$$

三个 loss 的角色：

1. **$\mathcal{L}_{\mathrm{DSM}}$ (Denoising Score Matching)**：student 在 causal mask 下做 teacher-forcing denoising。这是基础 token prediction fidelity——保证 student 还能"听懂" noise schedule。

2. **$\mathcal{L}_{\mathrm{VSD}}$ (Video Score Distillation)**：把 frozen bidirectional teacher 当作 score function guide，让 student 的 autoregressive rollout 分布对齐 teacher 的 score。这相当于 SDS / VSD (Variational Score Distillation) 的 video 版本，避免 student 在 few-step regime 陷入 mode collapse。
   - 直觉：teacher 知道"完美视频长什么样"，student 通过 KL-style 匹配 teacher 的 score field
   - Reference: VSD 来自 DreamFusion / ProlificDreamer 系列 https://arxiv.org/abs/2305.16291

3. **$\mathcal{L}_{\mathrm{GAN}}$**：局部 sharpness 损失。Few-step diffusion 普遍有 blurry 问题，GAN loss 可以恢复 high-frequency detail。这跟 SDXL Turbo、DMD (Distribution Matching Distillation) 的设计思路一致。
   - Reference: DMD https://arxiv.org/abs/2311.18828

$\lambda_{\mathrm{vsd}}, \lambda_{\mathrm{gan}}$ 是 weighting coefficients，paper 没给具体值（应该在 supplementary）。

### 5.4 效果

Ablation Table 2 显示：
- Teacher model（50 步，bidirectional）：PSNR 25.47，FPS 0.10
- Causal student（4 步，no self-forcing）：PSNR 23.35，FPS 4.96
- Full streaming student（self-forcing）：PSNR 23.34，FPS 4.96，但 motion/embodiment/BG score 都略涨

代价是 PSNR 掉 ~2 dB，收益是 FPS ×50。5 FPS @ 720p on 单 H200 已经接近"可交互"的范畴了。

这个 trade-off 让我想到 consistency models 在 image 上的成功——但 video 多了一个 temporal coherence 约束，所以单纯 consistency loss 不够，必须加 VSD 来对齐 teacher 的轨迹分布。

---

## 6. Synthetic Cross-Embodiment Dataset

数据集构造是这篇 paper 的 hidden gem。他们用 Unity 渲染了一个 motion-aligned paired dataset，这个 design 对 evaluation 至关重要。

### 6.1 Construction pipeline

- **Motion source**：Humoto motion library，700+ humanoid motion sequences（manipulation、locomotion、daily activities）
- **10 humanoid assets**：5 robots + 5 digital humans
- **Skeleton alignment**：在 Blender 里统一 topology，Unity 里做 retargeting
- **Scene variation**：100 个 scene（office、factory、outdoor）
- **Camera**：scene 内 camera 固定，只换 humanoid asset

### 6.2 Paired sample

对每个 motion sequence，retarget 到所有 10 个 embodiment，在同 scene 同 camera 下渲染。然后两两组合成 ordered pairs，总共 7,200 个 paired training samples。

这个数据集的关键属性是：**paired videos share identical motion and context, differ only in embodiment appearance and morphology**。这是 controlled supervision——你明确知道"差异来自 embodiment"，所以模型必须学到 factorization。

### 6.3 Held-out evaluation

Unitree G1 整个 embodiment 在训练时 hold out，只在测试时用。50 个 motion-aligned test videos，有 ground truth target，所以可以做 PSNR/SSIM/MSE。

另外 test set 还包括 unseen motion tasks 和 unseen scene configurations——确保不是 memorization。

这种 synthetic evaluation 的好处是能拿到 pixel-level ground truth，real-world benchmark 就只能 reference-free 用 Gemini-3 Flash 当 evaluator。Appendix A 里给了完整的 prompt template，覆盖 Motion Fidelity / Embodiment Similarity / Background Consistency / Overall Quality 四个维度。

---

## 7. 实验数据深度解读

### 7.1 Table 1 主结果

Synthetic Held-out Benchmark：
| Method | PSNR | SSIM | MSE | Motion | Embod | BG | Overall |
|---|---|---|---|---|---|---|---|
| Kling O1 | 22.70 | 0.8951 | 0.0067 | 8.06 | 6.94 | 9.52 | 7.08 |
| Kling O3 | 22.76 | 0.8959 | 0.0065 | 8.76 | 7.90 | 9.32 | 7.42 |
| Runway Gen4 | 18.83 | 0.6575 | 0.0181 | 7.26 | 7.50 | 8.14 | 7.22 |
| Wan2.1-VACE | 22.44 | 0.8599 | 0.0066 | 6.40 | 5.88 | 8.68 | 6.22 |
| X-Humanoid | 23.03 | 0.8891 | 0.0057 | 8.94 | 8.04 | 9.78 | 7.53 |
| **Ours** | **25.47** | **0.9039** | **0.0033** | **9.06** | **8.43** | **9.94** | **7.92** |

几个观察：
1. PSNR 25.47 vs 第二名 X-Humanoid 23.03，gap ~2.4 dB，相当显著。说明 motion-aligned 渲染质量明显更好。
2. MSE 0.0033 是 X-Humanoid 的一半，说明 structural error 大幅减少。
3. Embodiment score 8.43 vs Kling O3 的 7.90——OmniHumanoid 的 paired-free LoRA 仍然比 commercial API 的 zero-shot reference transfer 强。
4. Runway Gen4 的 SSIM 0.6575 异常低，说明 general video editor 在 robot structural preservation 上确实挣扎。

Real-world Benchmark 上：
| Method | Motion | Embod | BG | Overall |
|---|---|---|---|---|
| Kling O1 | 7.49 | 8.46 | 9.91 | 8.53 |
| Kling O3 | 7.47 | 8.34 | 9.82 | 8.21 |
| Runway Gen4 | 6.79 | 5.07 | 8.61 | 7.22 |
| Wan2.1-VACE | 5.60 | 5.65 | 7.85 | 6.45 |
| **Ours** | **8.47** | **8.56** | **9.95** | **8.39** |

注意 real-world 上 BG consistency OmniHumanoid 是 9.95，几乎满分——这说明 streaming causal attention 在背景稳定性上有优势，因为每个 chunk 都明确 attend 到 ref token，不容易 background drift。

### 7.2 User Study (Table 3)

72.7% Motion Fidelity、65.7% Embodiment、62.6% BG、63.6% Overall——大幅领先 Kling O3。User study 比 VLM evaluator 更可信，因为人眼对 motion artifact 的 sensitivity 远超 VLM。

### 7.3 Ablation 的隐藏信息

Table 2 里有个值得注意的细节：
- Teacher Model w/o Decoupling 的 Embodiment score 只有 2.53——几乎完全失败
- 同模型的 Motion score 是 6.35，BG 是 8.56

这个 pattern 说明：**没有 decoupling 时，模型会走捷径把 source video 几乎原样输出**（motion 还行，BG 也还行），但完全渲染不出 target robot 的 appearance（embodiment 崩了）。这恰好印证了 unidirectional flow 的作用——没有它，denoising branch 被 conditioning branch 的 human appearance 主导。

---

## 8. 与相关工作的脉络联系

### 8.1 Video Foundation Models

OmniHumanoid 基于 Wan2.2-TI2V-5B。Wan 系列是阿里的开源 DiT video model，TI2V = Text-Image to Video。在它之上做 LoRA finetune 是合理的——base model 提供强大的 visual prior，LoRA 做 embodiment-specific adaptation。
- Wan: https://arxiv.org/abs/2503.20314
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Sora review: https://arxiv.org/abs/2402.17177

### 8.2 Robotizing Human Videos 这条线

- **Phantom** (arXiv:2503.00779)：rendered arm overlay，简单但受限
- **Masquerade** (arXiv:2508.09976)：data-editing 把 egocentric human video 转 robot observation
- **EgoVLA / EgoMimic / PH2D**：egocentric 视角的 human-robot joint training
- **X-Humanoid** (arXiv:2512.04537)：showlab 前作，diffusion-based full-body translation
- **Mitty** (arXiv:2512.17253)：另一个 diffusion-based human-to-robot 工作
- **H2R-Grounder** (arXiv:2512.09406)：paired-data-free 物理 grounding

OmniHumanoid 相对 X-Humanoid 的进步是：(1) 明确 factorize 出 shared motion + embodiment LoRA；(2) streaming distillation 让它 practical；(3) 系统的 unseen embodiment evaluation。

### 8.3 Decoupled Learning

- **AnimateDiff** (arXiv:2307.04725)：temporal module 解耦空间与时间
- **MotionDirector** (arXiv:2310.08465)：dual-path LoRA 解耦 motion / appearance
- **OmniConsistency** (arXiv:2505.18445)：style-agnostic consistency，两阶段 progressive LoRA
- **InstantStyle / CSGO**：image stylization 的 content-style decoupling

OmniHumanoid 的 unidirectional attention mask 是这一脉思路在 cross-embodiment setting 上的最 aggressive 应用——不只是用 separate module，而是用 attention mask 强制信息流方向。

### 8.4 Video Models for Robotics

- **Cosmos Policy** (arXiv:2601.16163)：NVIDIA 的 video model fine-tune 到 visuomotor control
- **Unified Video Action Model** (arXiv:2503.00200)：video model 当 policy backbone
- **Fast-WAM** (arXiv:2603.16666)：test-time future imagination
- **GR-2 / π0 / OpenVLA**：这一系列 VLA 工作都在思考"video pretraining 怎么 transfer 到 control"

OmniHumanoid 给这条线提供了一个关键能力：**generate synthetic robot observation data at scale**。如果你能在 720p 5 FPS 上无限生成 Unitree G1 的 manipulation video，下游 policy 训练的数据瓶颈就被打通了。

### 8.5 Cross-Embodiment Policy Learning

虽然 OmniHumanoid 做的是 video generation，但它的 factorization 哲学跟 robot policy learning 里的 cross-embodiment 工作遥相呼应：
- **RT-X**：cross-embodiment robot transformer
- **Octo**：cross-embodiment generalist policy，用 embodiment token 区分 robot
- **Cross-embodiment 的一般做法**：把 robot 的 proprioception 和 action space 做 normalization，让 shared trunk 学通用 policy，head 做 embodiment-specific

OmniHumanoid 可以理解为 video generation 版本的同样思想：shared trunk 学 motion，LoRA head 学 embodiment appearance。

---

## 9. Limitations 和我想问的问题

paper Section C 提到的 limitation：
- 4-step distillation 在 fine detail、temporal smoothness、complex motion 上仍然比 50-step teacher 差
- distillation framework 本身有 speed-quality trade-off

我自己想追问几个问题：

1. **Motion retargeting 的准确性怎么验证？** Humoto 是 human motion capture 数据，retarget 到 G1 时关节自由度对不齐怎么处理？比如 G1 的 wrist 只有 1-DOF，human wrist 是 3-DOF，retarget 一定有 information loss。这个 loss 会不会让模型学到"假"的 motion prior？

2. **LoRA 之间的 interference**。10 个 embodiment LoRA 同时存在，rolling loading 真的能完全消除 bias 吗？如果某个 embodiment 数据量明显多，shared model 会不会依然偏？

3. **Real-world 到 synthetic 的 sim-to-real gap**。OmniHumanoid 在 synthetic 上 PSNR 25.47，但 real-world benchmark 没有 PSNR——这意味着 real-world 上没有 ground truth 比较。Real-world 的 motion score 8.47 比 synthetic 9.06 低，gap 0.6，是不是说明 real video 的 motion complexity 比 synthetic 高？

4. **5 FPS 够用吗？** 对 data generation 够，对 interactive control 远远不够。要做到 30 FPS real-time，可能需要更激进的 distillation 或者 consistency model。

5. **Long-horizon drift**。Autoregressive rollout 一定有 error accumulation。paper 没给 >10s 视频的 fidelity 曲线。我猜长 horizon 上 motion drift 会明显——这跟 language model autoregressive 生成的退化是同构的问题。

---

## 10. 我的 take

OmniHumanoid 的核心 contribution 是把 cross-embodiment video generation 这个看似纠缠的问题，用 **物理隔离 + 单向信息流 + 两阶段训练** 显式 factorize 出来。Unidirectional attention mask 是个特别干净的设计——它把"motion 决定 appearance"这个因果假设硬编码进 architecture，而不是寄希望于模型自己学出来。

streaming distillation 部分虽然不是 novelty 最高的（VSD + GAN 都是已有技术），但组合在一起把 50 步压到 4 步、做到 5 FPS @ 720p，是 engineering 上很有意义的 milestone。

对你 Tesla Optimus 这种场景，这个工作给了一个清晰的 path：**未来想加一个新 robot，只需要拍几十段 wild video 训个 LoRA，不用动核心 motion model，也不用采集任何 paired teleop 数据**。这个 scaling 特性是 cross-embodiment learning 真正能 scale 起来的关键。

如果想深入，建议看看 showlab 的 GitHub：https://github.com/showlab/OmniHumanoid ，还有 X-Humanoid 那篇前作对比一下演进路径。

---

Reference 集合：
- OmniHumanoid repo: https://github.com/showlab/OmniHumanoid
- X-Humanoid: https://arxiv.org/abs/2512.04537
- Wan: https://arxiv.org/abs/2503.20314
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- AnimateDiff: https://arxiv.org/abs/2307.04725
- MotionDirector: https://arxiv.org/abs/2310.08465
- OmniConsistency: https://arxiv.org/abs/2505.18445
- DMD: https://arxiv.org/abs/2311.18828
- VSD / ProlificDreamer: https://arxiv.org/abs/2305.16291
- Phantom: https://arxiv.org/abs/2503.00779
- Masquerade: https://arxiv.org/abs/2508.09976
- EgoVLA: https://arxiv.org/abs/2507.12440
- EgoMimic: https://arxiv.org/abs/2410.18856
- PH2D (Humanoid Policy ∼ Human Policy): https://arxiv.org/abs/2503.13441
- Cosmos Policy: https://arxiv.org/abs/2601.16163
- Unified Video Action Model: https://arxiv.org/abs/2503.00200
- Mitty: https://arxiv.org/abs/2512.17253
- H2R-Grounder: https://arxiv.org/abs/2512.09406
- Sora review: https://arxiv.org/abs/2402.17177
