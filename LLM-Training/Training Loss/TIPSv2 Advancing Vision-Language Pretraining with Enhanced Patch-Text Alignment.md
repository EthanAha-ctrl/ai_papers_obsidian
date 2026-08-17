---
source_pdf: TIPSv2 Advancing Vision-Language Pretraining with Enhanced Patch-Text
  Alignment.pdf
paper_sha256: 60e26116b053a5b14fd09c87766004857a4c2ebb63e0dbc67dc79275034a61a4
processed_at: '2026-08-12T16:24:41-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy 你好，没问题，我们抛开复杂的包装，用大白话把 TIPSv2 捋一遍。

现在的 vision model 一直有个老毛病。像 CLIP 这种模型，让它认出图里是猫没问题，但你要让它把猫对应的像素精确抠出来（zero-shot segmentation），它就歇菜了。因为 CLIP 的最后几层 transformer layers 把空间信息都揉碎了，全变成了一个 global feature 去跟 text 对齐。反过来看，像 DINOv2 这种 self-supervised model，对像素级的理解极强，但它不认识字。

之前大家试图把这两者捏合在一起（比如前作 TIPS 模型），结果发现了一个极其反直觉的 bug：**参数越多的大模型，它在 pixel-text 对齐上的表现，竟然还不如用这个大模型蒸馏出来的小模型。**

### 为什么会有这个 bug？

根本原因出在大家都在用的 Masked Image Modeling (MIM) 训练机制上。MIM 就是把图片挡住 75%，让模型猜被挡住的部分。原来的 iBOT loss 公式长这样：

$$ \mathcal{L}_{\mathrm{iBOT}} = - \sum_{i=1}^{N} m_{i} h_{t}(f_{t}(I)_{i})^{T} \log h_{s}(f_{s}(I_{\mathrm{mask}})_{i}) $$

公式里的变量解释一下：
*   $N$ 是图片切成的 patch 总数。
*   $m_i$ 是 mask 标记，$m_i=1$ 表示这个 patch 被挡住了，$m_i=0$ 表示没挡住。
*   $f_t$ 是 teacher model，$f_s$ 是 student model。
*   $h_t, h_s$ 是 projection heads。
*   $I$ 是原图，$I_{mask}$ 是被挡住后的图。

你看公式前面乘了个 $m_i$，这就意味着这个 loss **只惩罚被挡住的 patch**。
模型是很聪明的，既然没挡住的 patch 不参与 loss 计算，那模型就让这些“可见”的 patch 特征随便乱飘，只要能提供上下文帮它猜出被挡住的部分就行。这导致“可见”patch 的特征彻底丧失了局部语义，最后大模型的 patch feature 全变成了毫无语义的“中间件”。

### 绝妙的解法：iBOT++

作者的修改极其简单，把公式里的 $m_i$ 删掉就行了，这就是 iBOT++：

$$ \mathcal{L}_{\mathrm{iBOT++}} = - \sum_{i=1}^{N} h_{t}(f_{t}(I)_{i})^{T} \log h_{s}(f_{s}(I_{\mathrm{mask}})_{i}) $$

变量跟上面一样，只是去掉了 $m_i$。
这一删，loss 就会惩罚所有的 patch。模型不仅要猜被挡住的，还得保证那些没被挡住的 visible patch 特征，必须跟 teacher model 看完整图时输出的特征一模一样。这相当于强制把 visible patch 锚定在 teacher 的语义空间里。因为有 CLIP loss 的存在，这个空间天然是跟 text 对齐的，所以 patch-text alignment 就奇迹般地恢复了。

实验数据极其震撼，就这一个改动，ADE150 上的 zero-shot segmentation mIoU 从 3.5 直接飙升到 17.6。

### 省钱利器：Head-only EMA

另外，作者还搞了个 Head-only EMA。以前搞 self-distillation，必须弄一个跟 student 一模一样大的 teacher model 挂在内存里做 Exponential Moving Average (EMA)，参数直接翻倍。

作者发现，既然已经有 CLIP loss 这种强监督防止模型 collapse（所有特征变成常数）了，其实没必要对整个 vision encoder 做 EMA。只要让 student 和 teacher 共享同一个 backbone（$f_t = f_s$），只对最后那个小 projection head 做 EMA 更新就行了。这样在 ViT-B 上直接省了 42% 的内存，训练飞快。

### 喂不同长度的 Text

训练的时候，给图片配的文本太短了模型学不到细节，太长了模型又觉得任务太简单（一眼就能在 batch 里匹配出对应图片，导致 contrastive loss 毫无压力）。作者干脆混着喂：一半时间喂 PaliGemma 生成的中等长度 caption，一半时间喂 Gemini 生成的超长细节 caption。这样既保证了学习难度，又让模型学到了细粒度的空间语义。

### 总结

这篇 paper 的核心直觉就是：**之前的 MIM 训练太放任 visible tokens 了，只要把 visible tokens 也管起来，同时利用 CLIP loss 的防崩塌特性来砍掉多余的 EMA 参数，就能用极低的成本训出一个既懂文字又懂像素的超级 backbone。**

Web Links for the reference:
*   TIPSv2 Project Page: [https://gdm-tipsv2.github.io/](https://gdm-tipsv2.github.io/)
*   iBOT Paper: [https://arxiv.org/abs/2111.07832](https://arxiv.org/abs/2111.07832)
*   DINOv2 Paper: [https://arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)

---

Karpathy 你好！很高兴能和你一起探讨这篇来自 Google DeepMind 的最新 paper：**TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment**。这篇 paper 的核心动机非常符合你对 foundation model 训养机制的直觉：它从模型 scaling 过程中一个“反直觉”的 degenerate 现象出发，逆向工程找出了 pretraining recipe 中的缺陷，并提出了极其精简但极具成效的修正方案。

为了 build your intuition，我将从 paper 的核心发现、数学公式解析、架构设计、实验数据以及我个人的深度联想几个维度为你进行极其详细的拆解。

---

### 1. 核心直觉与反直觉发现

目前的 vision encoder 领域存在一个长期的 trade-off：
*   **Contrastive models (如 CLIP, SigLIP)**：拥有强大的 global image-text alignment 和 zero-shot 能力，但在 dense prediction (如 segmentation, depth) 上表现糟糕，因为最后几层 transformer layers 充当了 global contrastive "decoders"，丢失了 local spatial semantics。
*   **SSL models (如 DINOv2, iBOT)**：通过 Masked Image Modeling (MIM) 和 EMA teacher 机制，拥有极致的 spatial awareness 和 dense feature，但天生缺乏 text alignment。

之前的 TIPS 和 SigLIP2 试图将两者融合，但在评估 zero-shot semantic segmentation 时，作者发现了一个令人震惊的 **“反向 scaling”** 现象：**Flagship 大模型在 patch-text alignment 上的表现，竟然远逊于从它蒸馏出来的小模型。**

从 Table 1 中可以看到，TIPS ViT-g 在 ADE150 上的 mIoU 仅为 2.6，而作为 student 蒸馏出来的 TIPS ViT-L 却达到了 20.8。

**直觉解析：为什么会出现这种退化？**
作者通过消融实验 发现，问题出在 standard iBOT 的 MIM 机制上。在 iBOT 中，loss 只施加在被 mask 掉的 tokens 上，visible tokens 的 representation 是没有直接监督的。这就允许了 visible tokens 的 representation 可以随意漂移，只要它们能提供足够的 context 来重建 masked tokens 即可。这种机制导致模型在 pretraining 后期，patch features 退化为只服务于 global reconstruction 的“中间件”，丧失了与 text embedding 直接对齐的局部语义。

而在 distillation 阶段，由于去掉了 masking，student 的所有 tokens 都被迫去 mimic teacher 的所有 tokens。这种 dense 和全体的监督意外地“修复”了 patch-text alignment。

---

### 2. 核心方法：iBOT++ 的数学与机制

基于上述发现，作者提出了 **iBOT++**。这是一个对 iBOT 的极简修改，但效果极其震撼。

我们先看 standard iBOT 的 loss：
$$
\mathcal{L}_{\mathrm{iBOT}} = - \sum_{i=1}^{N} m_{i} h_{t}(f_{t}(I)_{i})^{T} \log h_{s}(f_{s}(I_{\mathrm{mask}})_{i}) \tag{2}
$$
*   $N$: Image 被切分成的 patch token 总数。
*   $m_i$: 二值 mask 变量。如果 $m_i = 1$，表示第 $i$ 个 patch 被 mask 掉了；如果为 $0$，则是 visible 的。
*   $f_t, f_s$: 分别代表 teacher 和 student 的 Vision Transformer backbone。Teacher 看到完整的 image $I$，Student 看到被 mask 过的 image $I_{\mathrm{mask}}$。
*   $f_t(I)_i$: Teacher 对第 $i$ 个 patch 输出的 feature embedding。
*   $h_t, h_s$: Projection heads，将 feature 投影到高维的 "prototype" 空间。
*   **关键点**：由于前面有 $m_i$ 作为系数，这个 loss **只对被 mask 的 patch 求和**。Visible tokens 在 student 侧的输出 $h_s(f_s(I_{\mathrm{mask}})_{i})$ 完全不参与这个 loss 的计算。

接下来是 **iBOT++** 的 loss：
$$
\mathcal{L}_{\mathrm{iBOT++}} = - \sum_{i=1}^{N} h_{t}(f_{t}(I)_{i})^{T} \log h_{s}(f_{s}(I_{\mathrm{mask}})_{i}) \tag{3}
$$
*   **修改极其简单**：去掉了前面的 $m_i$。
*   **数学意义**：Loss 现在施加在所有的 $N$ 个 tokens 上。Student 不仅要重建 masked tokens，还要确保它对 visible tokens 的 representation 也要与 teacher 看到 full image 时的 representation 保持一致。
*   **Intuition**：这相当于强制 visible tokens 也要“锚定”在 teacher 的语义空间里。结合 TIPS 里的 CLIP loss，这个空间天然是与 text 对齐的。因此，patch-text alignment 被完美保留了下来。

作者在 Table 12 中做了一个非常关键的 ablation：如果在 iBOT++ 中彻底不用 masking (ratio=0.0)，效果反而会崩盘 (ADE150 mIoU 降到 1.0)。这证明了 MIM 机制对于学习 general visual representation 依然是不可或缺的，iBOT++ 完美取了两者之长：既保留了 masking 带来的 global context 学习压力，又加上了 visible tokens 的 local semantic anchoring。

---

### 3. 训练效率革新：Head-only EMA

传统的 self-distillation (DINO/iBOT) 需要 teacher network 来提供稳定的 temporal ensembling targets。为了防止 representation collapse，teacher 必须是 student 的 Exponential Moving Average (EMA)，且整个 encoder 和 head 都要更新。这导致参数量翻倍。

TIPSv2 提出了 **Head-only EMA**，核心公式逻辑可以概括为：
$$
f_t := f_s
$$
$$
h_t \leftarrow \mathrm{EMA}(h_s)
$$
*   $f_t$ 不再是 EMA copy，而是直接共享 student 的参数 $f_s$。
*   只有 projection head $h_t$ 接收 $h_s$ 的 EMA 更新。

**为什么这能 work？**
在纯 SSL 中，如果 $f_t = f_s$ 且没有 stop gradient，模型会瞬间 collapse 到常数解。EMA teacher 提供了 stop gradient 和 temporal smoothing 来防止 collapse。
但在 TIPSv2 中，由于引入了 $\mathcal{L}_{\mathrm{CLIP}}$ (InfoNCE loss)，image encoder $f_s$ 必须不断去匹配不同的 text embeddings，这个 contrastive signal 本身就是极强的 anti-collapse 正则项。因此，我们可以安全地砍掉 encoder 的 EMA，只在容易 collapse 的 prototype head 上保留 EMA。

**收益**：在 ViT-B 上，trainable parameters 减少了 42%，极大提升了 TPU 的内存效率和吞吐量，使得在更大的 batch size 和更高分辨率下训练成为可能。

---

### 4. 数据侧的增强：Multi-Granularity Text Captions

TIPS 原本有两个 CLS tokens：一个用 Web alt-text 监督，一个用 PaliGemma synthetic captions 监督。TIPSv2 发现 PaliGemma 的 caption 还是太短，丢失了关键细节（如姿态、材质、季节）。

TIPSv2 引入了 **Gemini 1.5 Flash** 生成的超长 captions。但是，如果只用长 caption，对比学习任务会变得太简单（因为每个长文本包含的信息量太大，batch 内随便抓一个特征就能匹配），导致 contrastive loss 过于 trivial，学不到 robust 的 visual representation。

**解决方案**：
TIPSv2 采用了一种采样策略。第一个 CLS token 依然固定用 Web alt-text 监督；第二个 CLS token 在训练时，随机从 {PaliGemma captions, Gemini Flash captions} 中均匀采样一个作为 target。这样既保证了任务难度，又让模型能学到细粒度的 dense semantics。

---

### 5. 架构与训练细节

整体 Loss 架构如下：
$$
\mathcal{L} = \mathcal{L}_{\mathrm{CLIP}} + \alpha \mathcal{L}_{\mathrm{DINO}} + \beta \mathcal{L}_{\mathrm{iBOT++}}
$$
*   $\alpha = 1.0, \beta = 2.0$
*   $\mathcal{L}_{\mathrm{CLIP}}$ 是两个 CLS token 产生的 contrastive loss 的平均值。
*   $\mathcal{L}_{\mathrm{DINO}}$ 是 global level 的 self-distillation loss (见 paper 中 Eq 1)。
*   采用 Adafactor optimizer。EMA 机制中加入了 centering 和 sharpening 来进一步稳定 prototype space。

Training Recipe 分为两阶段：
1.  **Low-resolution stage**：90k steps，batch size 8192。1 个 global crop (224 res) + 6 个 local crops (98 res)。
2.  **High-resolution adaptation stage**：9k steps，batch size 4096。Global crop 提升到 448 res，local crops 提升到 140 res。这有助于模型适应下游高分辨率 dense prediction 任务。

在 ViT-g 规模 (1.1B image params) 上，使用 512 TPUv5 chips 训练 2 天。然后 ViT-g 作为 teacher，蒸馏出 ViT-B, ViT-L, SO 等小模型。蒸馏时依然保持 mask ratio = 0.0，确保 visible tokens 对齐。

---

### 6. 实验数据与直觉验证

Table 4 是核心 ablation study：
*   Baseline TIPS ViT-g 在 ADE150 zero-shot seg 只有 3.5 mIoU。
*   换成 iBOT++ 后，直接飙升到 17.6 (+14.1 mIoU)。这是整个 paper 最关键的 validation。
*   加上 Multi-granularity captions 提升到 18.1。
*   加上 Head-only EMA 提升到 19.1 (并且在 retrieval 任务上也全面提升)。Head-only EMA 甚至还略微提升了 zero-shot seg，这可能是因为共享 encoder 强制了更好的 feature consistency。

Table 7 和 Table 8 中，TIPSv2 g/14 在 dense prediction 上达到了 SOTA：
*   Semantic Segmentation (PASCAL) 达到 85.1 mIoU，超越了 DINOv2 (83.1) 和 FRANCA (81.3)。
*   Depth (NYUv2) RMSE 达到 0.334。
*   与刚发布的超级巨兽 DINOv3 (7B parameters, 2B images) 相比，TIPSv2 ViT-L 在 6 个核心指标中赢了 4 个。特别在 zero-shot seg (ADE150) 上，TIPSv2 L 达到 25.1，DINOv3 L 只有 24.7。

Paper 里的 Figure 5 PCA 可视化非常直观：TIPSv2 的 patch feature map 极其平滑且语义边界清晰，彻底摆脱了 CLIP/SigLIP 那种充满噪点、只关注 salient object 的特征图形态。

---

### 7. 深度联想与思考

1.  **关于 EMA 和 Stop Gradient 的本质**：Head-only EMA 的成功进一步印证了 self-distillation 中的一个深刻假设：EMA 和 stop gradient 的本质作用是提供一个稳定的、非退化的 target space，防止 representation collapse。当有外部强信号（如 CLIP loss）已经保证了 non-degeneracy 时，EMA 的作用就可以被极度缩减。这和 BYOL 早期争论 stop-gradient 是否必要时的结论一致。未来我们甚至可以思考：如果 contrastive loss 足够强，连 head 的 EMA 都可以用一个 stop-gradient 的 frozen random projection 来替代。
2.  **MIM 的真正作用**：长久以来我们认为 MIM 是在训练模型“预测缺失的部分”。但 iBOT++ 暗示，MIM 更大的价值在于迫使 visible tokens 构建 sufficient statistics 以支持重建。Standard iBOT 让 visible tokens 变成了纯粹的“推理上下文”。iBOT++ 则逼迫 visible tokens 自身也要落在有意义的 semantic manifold 上。这与 recent MAE 改进工作中关于 "feature mimicking vs pixel reconstruction" 的讨论高度共鸣。
3.  **VLM 的 Spatial Bottleneck**：目前 LLM 接入 vision tokens 通常通过 attention 机制。如果 visual tokens 本身就不具备清晰、独立的局部语义（如 CLIP 的 tokens），LLM 就很难做 fine-grained grounding。TIPSv2 提供的这种 native text-aligned 且 spatially coherent 的 encoder，极有可能是下一代 open-vocabulary segmentation 甚至 robotic manipulation 的完美 backbone。这比在 CLIP 后面接一个 awkward 的 spatial projector 要优雅得多。

---

### 8. Web Links & References

*   **TIPSv2 Project Page**: https://gdm-tipsv2.github.io/
*   **TIPSv2 Paper (ArXiv pending, based on provided text)**: 参考 Google DeepMind 最新发布。
*   **iBOT (Base method)**: https://arxiv.org/abs/2111.07832
*   **DINOv2 (Dense SSL baseline)**: https://arxiv.org/abs/2304.07193
*   **DINOv3 (Recent SOTA compared)**: https://arxiv.org/abs/2508.10104
*   **Perception Encoder (PE)**: https://arxiv.org/abs/2504.13181
*   **SimMIM (Previous visible patch attempt)**: https://arxiv.org/abs/2111.12733
*   **DMAE (Visible & Masked prediction)**: https://arxiv.org/abs/2306.16951

希望这个详细的拆解能进一步 build up 你的 intuition，Karpathy！这篇 paper 的核心 takeaway 在于：pretraining recipe 中很多约定俗成的设置（如只对 masked token 计算 loss，或必须有完整的 EMA），在 multimodal 场景下其实是可以被重新审视和极度精简的。
