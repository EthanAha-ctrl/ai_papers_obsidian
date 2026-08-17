---
source_pdf: Emu3.5 Native Multimodal Models are World Learners.pdf
paper_sha256: 68c26d4edc2591f12111c2daca0e2f8edc8c09afe769bae3acd4ea846e004365
processed_at: '2026-08-04T04:13:34-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Emu3.5 用人话说一遍

## 这篇论文到底在干嘛

BAAI 这帮人想证明一件事：**把所有模态（text、image、video frame）都 tokenize 成离散 token，然后丢给一个普通的 autoregressive transformer 做 next-token prediction，就能学成 world model**——不仅能生图、改图，还能讲故事、教你怎么做事、在虚拟世界里探索、控制机器人。

这跟现在主流路线不一样。主流是 image 用 diffusion、text 用 transformer、video 用另一个 diffusion，然后拼起来。Emu3.5 说：不用拼，全部当成一个 token 序列，让一个模型统一学。

这个想法 Emu3 就有了，Emu3.5 把它推到 34B 参数、13T tokens、跟 GPT-Image-1 / Nano Banana 平起平坐的级别，还顺便搞了个 20x 推理加速的 trick。

项目主页：https://emu.world  
GitHub：https://github.com/baaivision/Emu3.5

---

## 核心直觉：为什么用 video interleaved 数据

这是 Emu3.5 最关键的一个决定。看 Table 2 的数据配比，video-interleaved 占 55%，是大头。

为啥这么重视 video？因为 image-text pair 学不到时间连续性和因果关系。一张图配一段文字，模型学到的是"这个画面这个文字描述对应"，但它学不到"这个画面之后会发生什么"。

Emu3.5 的做法：从 63M 个视频（合计 790 年 footage）里，用 PySceneDetect 切场景，每个场景按固定间隔抽 keyframe，同时用 Whisper-large-v2 把音频转成带时间戳的文字，然后按时间顺序把 keyframe tokens 和 ASR text 交错排起来，变成一个长长的 document。

直觉上讲，这相当于把视频变成一本"图文书"——每页有图有字，按时间顺序展开。模型读这本书，自然就学到"看到这个画面，接下来文字会说什么、画面会变成什么样"。这就是 world modeling 的基础。

参考 Whisper：https://arxiv.org/abs/2212.04356  
PySceneDetect：https://github.com/Breakthrough/PySceneDetect

---

## Tokenizer 为什么这么花心思

Tokenizer 是整个 native multimodal 路线的"下限"。tokenize 得烂，后面 transformer 再大也只能 hallucinate 出模糊的图。

Emu3.5 用 IBQ（Index Backpropagation Quantization）框架，codebook 大小 131,072（$2^{17}$），下采样 16x，tokenizer 本身 455M 参数。

传统 VQ-VAE 的问题：用 stop-gradient + straight-through estimator，codebook entry 收不到真正的梯度，很多 entry 永远不被用，information capacity 浪费。IBQ 用 softmax 形式的软量化，训练时保持软概率，推理时取 argmax，但梯度能通过 softmax 流回 codebook，所以 131k 个 entry 都能被充分激活。

再加上 REPA 思路：让 tokenizer decoder 的中间 hidden state 去对齐 SigLIP（一个 contrastive vision-language encoder）的 patch embedding。等于强制 discrete tokens 不只 carry 像素信息，还 carry 语义信息。后续 autoregressive model 学起来更轻松。

还有个 dual decoder 设计：
- Vanilla decoder：1024 tokens 直接重建 512×512
- Diffusion decoder：1024 tokens 作为 condition，用 SD3.5 medium 初始化的 flow-matching 模型生成 1024×1024，再 LoRA distill 把 50 步压到 4 步

为啥要 diffusion decoder？因为 vanilla decoder 受 quantization loss 限制，文字、人脸这种高频细节会糊。Diffusion decoder 把 "token → pixel" 这个 ill-posed 映射变成 conditioned 生成过程，能 hallucinate 出 token 没保留的细节。

结果看 Table 15：text accuracy T-ACC_m 53.22，face similarity F-Sim_m 0.14，rFID 0.49，几乎是 O-MAGVIT2 的 1.3-3x。

参考 IBQ：https://arxiv.org/abs/2412.02692  
REPA：https://arxiv.org/abs/2410.06940  
SigLIP：https://arxiv.org/abs/2303.15343

---

## 模型架构：就是个大 LLM

34.1B 参数，64 层，hidden 5120，standard decoder-only transformer。GQA 64/8（KV cache 压 1/8），QK-Norm 防 attention 爆炸，SwiGLU activation，RoPE 位置编码，RMSNorm pre-norm。词表 282,926 = 151,854 text（直接复用 Qwen tokenizer）+ 131,072 vision。

没什么新架构，就是 Qwen3 那套 LLM 架构，把 vision token 加进词表而已。Context length 32,768。

初始化自 Qwen3，所以文本能力直接 carry over，不从零学。这是个很实际的决策——34B 模型从零学文本不划算，不如继承一个已经训好的 LLM。

参考 GQA：https://arxiv.org/abs/2305.13245  
Qwen3：https://arxiv.org/abs/2505.09388

---

## Pre-training 怎么训的

两阶段，合计 13T tokens：

Stage 1：10.3T tokens，512×512 分辨率，LR 5e-4，700k steps，batch 448，seq len 32768。数据在线 pack 到最大 context length。这个阶段主要靠 video-interleaved 学基础 multimodal alignment 和 NTP。

Stage 2：3.5T tokens，分辨率拉到 512-1024 dynamic，LR 降到 1e-5，240k steps。这个阶段加了更丰富的 annotation（semantic segmentation、visual caption、multimodal summary），提升数据质量和监督信号精度。

Loss 是标准 cross-entropy，但 visual token 权重 0.5，text token 权重 1.0。因为一张图 1024+ token，如果不降权重，梯度被视觉主导，文本能力会 degrade。

并行策略 TP=8, CP=2，用 FlagScale 框架。

关键验证：9 个 validation set 的 loss 全部单调下降（Figure 7），包括 3 个 OOD 下游任务（visual narrative, visual guidance, world exploration）。这证明 interleaved 训练范式有 robust generalization。

参考 FlagScale：https://github.com/FlagOpen/FlagScale

---

## Post-training：SFT + RL 两段

SFT 总共 150B tokens，覆盖 6 个任务（Table 3）：

1. **General Tasks**（29.7B）：T2I + Language + VL QA
2. **Any-to-Image**（56.2B）：任意输入（text + 0/1/多张图）→ 单张图
3. **Visual Narrative**（10.1B）：生成 image-text 交错的故事序列
4. **Visual Guidance**（22.5B）：step-by-step instruction + visual demonstration
5. **World Exploration**（17.5B）：两种模式，用户交互式 / 模型自驱探索
6. **Embodied Manipulation**（14.1B）：机器人 subtask-keyframe 交错预测

SFT 分两阶段：先标准分辨率 converge，再拉高分辨率精修。batch 1024，LR 6e-6，每 stage 3000 iterations。

RL 用 GRPO（Group Relative Policy Optimization），global batch 640，rollout G=8，LR 1e-6。

GRPO 相比 PPO 的好处：不需要单独 value network，用 group 内 baseline 算 advantage，省算力。对 34B 模型跑 RL 来说很关键。

Reward system 设计三要点：
- Generality：跨任务通用 reward（CLIP similarity、aesthetic score、VLM alignment）
- Task-specificity：OCR/text fidelity、face ID、narrative consistency 各有专门 reward
- Unified nature：所有 reward normalize 到 [1,10] 再合并，避免 hacking 单一 reward

Reward 从 ~4.5 涨到 >7.1（Figure 8），证明 unified multi-task RL 能同时优化多个异质目标。

参考 GRPO：https://arxiv.org/abs/2402.03300  
VeRL：https://arxiv.org/abs/2409.19256

---

## DiDA：20x 推理加速的核心 trick

这是 Emu3.5 在 inference engineering 上的亮点。

AR 模型生 1024×1024 图要 4096 token，逐个 decode，每步 full KV attention，latency 120s+。Diffusion 模型走并行 denoising，30 步左右，1-2 秒。AR 在速度上被 diffusion 吊打。

DiDA（Discrete Diffusion Adaptation）的思路：把 AR-trained 模型改造成 discrete diffusion predictor，但只改 attention mask，不改模型本质。

具体做法：
- 训练时，对每个 image 复制成一份 noisy copy
- Noisy image token attend（1）causally 到 preceding clean tokens，（2）bidirectionally 到同 image 的其他 noisy tokens
- Clean image/text token 保持 causal attention

推理时：
- Text 部分：仍然 causal AR
- Image 部分：4096 tokens 同时初始化成 noise mask token，迭代 denoise K 步，每步所有 token 同时预测，只 update 高熵位置（类似 MaskGIT 的 cosine masking schedule）

结果（Table 16）：DiDA 把 4096-token image 生成时间从 120s 压到 10s，20x 加速，benchmark 几乎不掉（GenEval 0.86→0.86，DPG 88.26→87.46，GEdit 7.59→7.56）。

第一次让 AR 模型在速度上比肩连续 diffusion。这是个非常聪明的工程决策——AR 训练成本 + diffusion 推理速度兼得。

参考 MaskGIT：https://arxiv.org/abs/2202.04200  
Discrete diffusion：https://arxiv.org/abs/2510.24717

Infrastructure 上还有两个创新点：
1. FlexAttention per-row block mask：用 per-row block mask 编码 causal + bidirectional + region-specific，不用 store full attention matrix，省内存
2. FSM-based scheduling：inference 时 text/image phase 动态切换，有限状态机调度，4-device 上 ≥50% speedup

---

## 实验结果挑重点说

### T2I（文生图）

TIIF Bench（Table 4）：Emu3.5 overall 89.48，跟 GPT-Image-1（89.15）同级别，超过 Nano Banana、Qwen-Image、Seedream 3.0。Text subtask 直接 100 满分。

OneIG-Bench EN（Table 5）：overall 0.564，超过 Nano Banana（0.550）。Text 维度 0.994，几乎满分。

LeX-Bench（Table 7）：Hard 难度 pNED 4.39，recall 0.87，SOTA。GPT-Image-1 pNED 5.52，recall 0.70。

CVTG-2K（Table 9）：Word Accuracy 0.9123，超过 GPT-Image-1（0.8569）、Qwen-Image（0.8288）。

Text rendering 强的原因：tokenizer 直接把像素切到 token，token-level AR 学 long-range 字符 pattern 比 diffusion 的连续 latent 更直接。而且 tokenizer 在 text-rich 数据上专门训过。

### Any-to-Image（图生图/编辑）

ImgEdit（Table 10）：overall 4.41，超过 Nano Banana（4.28）、Qwen-Image-Edit（4.35）、GPT-Image-1 High（4.20）。

GEdit-Bench（Table 11）：overall 7.59，最高。

OmniContext（Table 12）：subject-driven generation，average 8.82，甚至超过 GPT-4o（8.80）。Object 维度特别强（8.89-9.46）。

ICE-Bench（Table 13）：31 个 task 综合 0.637，超过 Nano Banana（0.631）。但 Task 2（face ref）、Task 3（style ref）、Task 31（face swap）偏弱，作者标注为未来工作。

### 4 个 interleaved 任务对比 Nano Banana（Table 14）

这是 Emu3.5 的核心卖点验证：

| Task | Win | Tie | Lose |
|---|---|---|---|
| Visual Narrative | 49.2 | 10.3 | 40.5 |
| Visual Guidance | 51.5 | 9.5 | 39.0 |
| World Exploration | 65.5 | 0.0 | 34.5 |
| Embodied Manipulation | 67.1 | 2.4 | 30.5 |

World Exploration 和 Embodied Manipulation 胜率大幅领先。这是 Emu3.5 的 thesis 验证：长程 multimodal world modeling 通过 native interleaved AR 训练获得，封闭源 SOTA 在这块没专门训。

### Tokenizer reconstruction（Table 15）

Emu3.5 vanilla：T-ACC_m 53.22，T-NED_m 91.78，rFID 0.49。Text accuracy 几乎是 O-MAGVIT2 的 1.3-3x。Tokenizer 好坏决定 native multimodal AR 的下限。

---

## 几个关键 takeaway

1. **Native multimodal world model 本质上是数据格式问题**。不是新 architecture，是把视频切成 keyframe+ASR 时间交错序列，丢给标准 transformer 做 NTP。这暗示 "world model" 不需要 latent dynamics 的特殊参数化，只要数据足够长足够多，AR transformer 自己学 dynamics。

2. **Tokenizer 是一等公民**。131k codebook + IBQ 可微量化 + SigLIP distillation + diffusion decoder。这块比 Emu3 改进巨大，直接决定了 text rendering 和人脸重建的 SOTA 表现。

3. **Video-interleaved 55% 数据配比是有意为之**。仅看 image-text pair 学不到 time/causal context；仅看短视频学不到长程 reasoning。这是 Emu3.5 跟 GPT-4o / Nano Banana 的差异化。

4. **DiDA 是非常聪明的 inference engineering**。不改 AR model 本质，只改 attention mask 让 image tokens 之间 bidirectional，NTP 变成 masked denoising。AR 训练成本 + diffusion 推理速度兼得。

5. **Unified multi-task RL 的 reward design**：所有 reward normalize 到 [1,10] 再合并，避免 hacking 单一 reward。这是 unified post-training 的工程经验。

6. **Limitations 明显**：
   - 32k context length，长视频受限（12 张 1024² 图就 50k token 装不下）
   - Embodied manipulation 只到 keyframe 层，没到 dense action control
   - Video decoder 是事后挂的（Wan2.2 based），不是 end-to-end NTP
   - World Exploration 评估以 ChatGPT auto-judge 为主，human eval 缺失

---

## 一句话总结

Emu3.5 是把 "next-token prediction is all you need" 这个 thesis 在 scale + post-training + inference efficiency 三个维度同时推到 SOTA 级别的工作。没发明新 architecture，但每一层（tokenizer / data / pretrain / SFT / RL / inference）都做对了工程，把 AR native multimodal 推到能跟 GPT-Image-1 / Nano Banana 平起平坐，并在 world modeling / 长程 reasoning 这种新维度上领先。

这是一篇工程论文，不是 thesis-type paper。核心贡献在工程决策的系统性正确，而非某个单点创新。

---

# Emu3.5: 一个 native multimodal world model 的深度解读

## 1. 这篇 paper 在讲什么 - philosophy 的层面

Emu3.5 是 BAAI 在 Emu3 之后的一步,核心 thesis 很清楚:**只要把所有模态都 tokenize 成离散 token,然后做 next-token prediction,一个 single autoregressive transformer 就能学成一个 world model**——不仅能 image generation / editing,还能长程 visual narrative、step-by-step visual guidance、open-world exploration、embodied manipulation。这个 framing 比 Emu3 更激进,Emu3 主要证明 "next-token prediction unifies perception and generation",Emu3.5 把它扩展到 "world modeling"——即预测 environment 在 time 和 action 下的演化。

参考 Emu3 paper: https://arxiv.org/abs/2409.18869  
项目主页: https://emu.world  
GitHub: https://github.com/baaivision/Emu3.5

这里有一个非常关键的设计直觉:他们没有走 "video diffusion + language head 拼接" 的路线,而是把 video 切成 keyframe → tokenize → 和 ASR transcript 按时间戳交错,变成一个长长的 document 序列。这是把视频 generation 重新 cast 成 sequence modeling,从而所有 video reasoning / image reasoning / text reasoning 共享同一个 loss surface 和同一个 transformer backbone。

---

## 2. Architecture - 34B decoder-only transformer

模型配置 (Table 1):

| 项 | 值 |
|---|---|
| Parameters | 34.1B |
| Layers | 64 |
| Hidden size | 5,120 |
| Intermediate size | 25,600 (SwiGLU 的 hidden,等于 $2/3 \times 4 \times 5120 \approx 13653$?这里实际是 $25{,}600 \approx 5120 \times 5$,说明 FFN expansion ratio 约 5) |
| Heads (Q / KV) | 64 / 8 (GQA) |
| Vocabulary | 282,926 = 151,854 text (Qwen tokenizer) + 131,072 vision |
| Context length | 32,768 |
| Dropout | 0.1 |
| Transformer params | 31.2B |
| Embedding params | 2.9B (词表大,这部分相当可观) |

关键技术点:

### 2.1 GQA (Grouped Query Attention)

每个 query head 共享一个 KV head,64 Q-heads / 8 KV-heads = 8:1 ratio。公式:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

其中 $Q \in \mathbb{R}^{N \times h_q d_k}$,$K, V \in \mathbb{R}^{N \times h_{kv} d_k}$,$h_q=64, h_{kv}=8$,$d_k = 5120/64 = 80$。每个 KV head 服务 $h_q/h_{kv} = 8$ 个 query heads。这把 KV cache 压到原来的 1/8。

参考 GQA: https://arxiv.org/abs/2305.13245

### 2.2 QK-Norm

对 $Q$ 和 $K$ 在送进 dot product 之前各自做 RMSNorm / LayerNorm:

$$\tilde Q = \frac{Q}{\text{RMS}(Q)}, \quad \tilde K = \frac{K}{\text{RMS}(K)}, \quad \text{logits} = \frac{\tilde Q \tilde K^\top}{\sqrt{d_k}}$$

直觉:随着模型变大,$Q K^\top$ 的 scale 容易爆炸,softmax 之后 attention 分布变得非常 sharp,梯度会死。QK-Norm 把 $Q, K$ 各自归一,logits 自动 bounded,训练稳定性大幅提升。这个 trick 在 Chameleon 22B 时期被验证为必需。

参考: https://arxiv.org/abs/2305.13245 (同 GQA 论文里也提到了 norm stabilization)

### 2.3 SwiGLU activation

$$\text{FFN}(x) = \big(\text{SwiGLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big) W_{\text{down}}$$

其中 $\text{SwiGLU}(u) = u \odot \text{SiLU}(u) = u \odot (u \cdot \sigma(u))$。$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$,$W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$,$d=5120, d_{ff}=25{,}600$。

直觉:SiLU gating 让 FFN 有一个 soft 路由,某条 path 被激活/抑制程度可学,比单纯 ReLU 表达力更强、梯度更平滑。

### 2.4 RoPE (Rotary Position Embedding)

每个 query/key vector 在每两个维度上做旋转:

$$\text{RoPE}(x, m)_{2k, 2k+1} = R(m, k) \cdot (x_{2k}, x_{2k+1})^\top, \quad R(m, k) = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix}$$

其中 $m$ 是 position index,$\theta_k = 10000^{-2k/d}$ 是 base frequency。RoPE 的性质: $\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \langle q, k \rangle$ rotated by $(n-m)$ —— 即 attention 是相对位置的函数。

参考: https://arxiv.org/abs/2104.09864

### 2.5 Pre-normalization with RMSNorm

$$x_{l+1} = x_l + f\big(\text{RMSNorm}(x_l)\big), \quad \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

$\gamma \in \mathbb{R}^d$ 是 learnable scale,$\epsilon$ 防 div0。比 LayerNorm 少了去均值那一步,更快。

---

## 3. Tokenizer - 视觉信号的离散化

这是整个 native multimodal 范式最 critical 的组件。Emu3.5 用 IBQ (Index Backpropagation Quantization) 框架:

### 3.1 IBQ vs VQ 的本质区别

传统 VQ-VAE / VQGAN 用 stop-gradient + straight-through estimator:

$$z_q = z + \text{sg}(e_k - z), \quad \text{where } k = \arg\min_i \|z - e_i\|$$

这意味着 codebook embedding $e_i$ 只能通过 commitment loss 和 codebook loss 间接更新,signal 弱、容易 collapse(很多 codebook entry 永远不被用)。

IBQ 用 softmax 形式的软量化:

$$p_i = \frac{\exp(-\|z - e_i\|^2 / \tau)}{\sum_j \exp(-\|z - e_j\|^2 / \tau)}, \quad z_q = \sum_i p_i \, e_i$$

(实际上 IBQ 用更精巧的可微 index selection,在训练时保持软概率,推理时取 argmax,但梯度通过 softmax-temperature schedule 流回 codebook)。

直觉:codebook entry 直接 receiving gradient,所以 131,072 个 entry 全能被充分激活,information capacity 远超传统 VQ。Emu3.5 把 codebook 从 Emu3 的 32768 扩到 131072,模型 455M params。

参考 IBQ: https://arxiv.org/abs/2412.02692

### 3.2 Codebook 配置

| 项 | 值 |
|---|---|
| Downsampling factor $f$ | 16 |
| Codebook size | 131,072 = $2^{17}$ |
| Token dim | $D = 256$ |
| Tokenizer model size | 455M |
| Input image | $H \times W$ |
| Output tokens | $(H/f) \times (W/f)$ |

所以 $512 \times 512$ 图像 → $32 \times 32 = 1024$ tokens;$1024 \times 1024$ → 4096 tokens。

### 3.3 REPA-style distillation

论文说 "inspired by REPA, integrate feature distillation from SigLIP into intermediate outputs of tokenizer decoder"。直觉:tokenizer 原本只学 "重建像素" 这个目标,语义信息弱。让 decoder 中间 hidden state 去对齐 SigLIP (一个 contrastive vision-language encoder) 的 patch embedding,等于强制 discrete tokens carry 语义信息,后续 autoregressive model 学起来更轻松。

参考 REPA: https://arxiv.org/abs/2410.06940  
SigLIP: https://arxiv.org/abs/2303.15343

### 3.4 Tokenizer 训练 objective

组合多 loss:

$$\mathcal{L} = \lambda_{\text{rec}} \mathcal{L}_{\text{rec}} + \lambda_{\text{quant}} \mathcal{L}_{\text{quant}} + \lambda_{\text{lpips}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}} + \lambda_{\text{sem}} \mathcal{L}_{\text{sem}}$$

- $\mathcal{L}_{\text{rec}}$: pixel reconstruction (MSE or L1)
- $\mathcal{L}_{\text{quant}}$: commitment loss $\|z - \text{sg}(z_q)\|^2 + \beta\|\text{sg}(z) - z_q\|^2$
- $\mathcal{L}_{\text{LPIPS}}$: perceptual loss,基于 AlexNet/VGG 中间 feature 计算
- $\mathcal{L}_{\text{adv}}$: PatchGAN discriminator 的 hinge loss
- $\mathcal{L}_{\text{ent}}$: codebook usage entropy,鼓励 codebook 各 entry 被均匀使用,防 collapse
- $\mathcal{L}_{\text{sem}}$: SigLIP feature distillation

Optimizer: Adam,$\beta_1 = 0.5, \beta_2 = 0.9$,LR 1e-4,15k warmup,500k iterations,batch 256。

### 3.5 双 decoder 设计

非常聪明的两点:

**Vanilla image decoder**: 直接从 quantized tokens 解码到 image,标准 CNN/transformer decoder。1024 tokens 即可重建 512×512。

**Diffusion-based image decoder**: 把 quantized tokens 作为 condition,用一个 flow-matching diffusion (SD3.5 medium 初始化) 生成 2x 分辨率图。即 1024 tokens → 1024×1024 image。再 LoRA distill 把 50 步 denoise 压到 4 步。

直觉:vanilla decoder 受限于 quantization loss,细节(尤其是文字、人脸)模糊。diffusion decoder 把 "token → pixel" 这个 ill-posed 映射变成一个 conditioned 生成过程,可以 hallucinate 出 token 没保留的高频细节。

**Video decoder**: 基于 Wan2.2 5B DiT,条件是 (1) quantized embeddings 提供细节,(2) 可选 text 给语义,(3) 4-channel mask 指定哪些 frames 已知。训练时随机把第一帧 latent 替换为 clean image tokens,bridge long-term dependency。

参考 Wan: https://github.com/Wan-Video/Wan2.1 (2.2 是延续)

---

## 4. Pre-training 数据 pipeline - 13T tokens 怎么来

### 4.1 数据组成 (Table 2)

| 数据类型 | Stage 1 ratio | Stage 2 ratio |
|---|---|---|
| Text-only | 0.20 | 0.18 |
| Image-text pair | 0.20 | 0.16 |
| Video-text pair | 0.05 | 0.08 |
| Any-to-Image | 0.00 | 0.03 |
| **Video-interleaved** | **0.55** | **0.55** |

**关键直觉**:video-interleaved 数据 55% 是大头。这是 Emu3.5 的 thesis:你要学 world model,得让模型看长 sequence,光看 (image, caption) 这种独立 pair 学不到 temporal continuity 和 causal dynamics。

### 4.2 Video interleaved 数据怎么造

源:63M videos,平均 6.5 min/视频,合计约 790 年 footage。

步骤:
1. **Scene segmentation**: PySceneDetect 把每个 video 切成 coherent scene
2. **Keyframe sampling**: 若 scene 时长 < t 秒,取中间 1 帧;否则每 t 秒取一帧(带 timestamp)。平均 0.27 keyframe/秒,即大约 4 秒一帧
3. **ASR**: Whisper-large-v2 (Faster-Whisper 实现) 生成 word-level timestamp
4. **Text segmentation**: spaCy 按 syntactic pause 切句,得到 grammatically coherent + temporally aligned transcripts
5. **Interleaving**: 按 timestamp 把 keyframe tokens 和 ASR text 交错排成 document

两阶段 filtering:

**Basic filtering** (Stage 1):
- 时长/分辨率过滤
- Talking-head 过滤(face detection + Qwen-VL 分类)
- 多语言 + 静音视频 balance

**Advanced filtering** (Stage 2):
- DeQA 评 frame quality
- DINO + FG-CLIP feature 计算跨帧相似度去重
- LLM 给 ASR 文本打分

Stage 2 还加 annotation:LLM 做 semantic segmentation + summarization,Qwen2.5-VL-7B 给每个 scene 生成 visual caption,LLM 整合出 multimodal summary。

### 4.3 Vision-text pairs

500M image-text + 30M video-text pair。Emu3 基础 + Qwen2.5-VL-7B 重 caption,加入 InfinityMM / LLaVA-OV 增强 understanding。video clip 按 1 FPS 采样,同源 clip 在训练时按时间顺序 pack 成 interleaved sequence。

### 4.4 Any-to-Image (X2I)

27.35M samples。开源:SEED-Data-Edit, WeatherStream, PromptFix, OmniGen-X2I, ShareGPT-4o-Image, ImgEdit, OmniGen2-X2I2, MultiRef, GPT-IMAGE-EDIT-1.5M。再加 in-house curated 数据,从真实视频/图像 + 合成数据组合。

### 4.5 Text-only

~3T tokens,基于 Emu3 + 高质量开源 corpora(英文中文都有)。

### 4.6 训练 recipe (Table 2)

| Hyperparam | Stage 1 | Stage 2 |
|---|---|---|
| LR | 5e-4 | 1e-5 |
| Scheduler | Cosine | Cosine |
| Weight decay | 0.1 | 0.1 |
| Gradient clip | 5.0 | 5.0 |
| Loss weight (vis:text) | 0.5 : 1.0 | 0.5 : 1.0 |
| Warmup | 700 steps | (同 S1) |
| Training steps | 700k | 240k |
| Seq length | 32768 | 32768 |
| Batch size | 448 | 448 |
| Resolution | 512×512 | 512-1024 dynamic |
| Tokens seen | 10.3T | 3.5T |

初始化自 Qwen3 (所以文本能力 carry over,不从头学)。Stage 1 在线 pack,Stage 2 离线 pack + dynamic token strategy (1024-4096 visual tokens,保持 aspect ratio)。

并行策略:TP=8, CP=2。FlagScale 框架。

Loss:

$$\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} w_t \log p(x_t | x_{<t}), \quad w_t = \begin{cases} 0.5 & x_t \in \mathcal{V}_{\text{vis}} \\ 1.0 & x_t \in \mathcal{V}_{\text{text}} \end{cases}$$

视觉 token 权重 0.5 的原因:视觉 token 数量大(一张图 1024+ token),如果 weight 等同 text,梯度被视觉主导,文本能力会 degrade。

### 4.7 验证 loss 曲线 (Figure 7)

9 个 validation set: ISG-Bench, OpenING, MMIE + 3 个 in-domain (T2I, I2T, video-interleaved) + 3 个 downstream SFT 任务(visual narrative, visual guidance, world exploration)。所有 9 个 val loss 单调下降——验证 "scaling + interleaved training" 给出 robust generalization,包括 OOD 任务。

---

## 5. Post-training - SFT + RL

### 5.1 Task formulation

SFT 覆盖 6 个任务(Table 3):

| Task | Tokens | Output |
|---|---|---|
| General (Language/VL/T2I) | 29.7B | Text/Image |
| Any-to-Image | 56.2B | Image |
| Visual Narrative | 10.1B | Interleaved |
| Visual Guidance | 22.5B | Interleaved |
| World Exploration | 17.5B | Interleaved |
| Embodied Manipulation | 14.1B | Interleaved |

总 SFT ~150B tokens。

#### Visual Narrative

要生成的不是 "单图+caption",而是 "image-text 交错的故事序列",character/style/temporal consistency 都要保持。覆盖 cartoon / photoreal / 历史 / 科幻 / 教育题材。构建:PySceneDetect + Whisper + Qwen2.5-VL dense caption + Qwen3 切分 narrative segment + 独立 image-level CoT 与 global CoT 标注。最终 430k 样本。

#### Visual Guidance

step-by-step instruction + visual demonstration,覆盖 cooking/DIY/crafting。要求 step 数 2-10。dual-level CoT:image-level (每步视觉推理) + global (跨步语义布局防 long-sequence forgetting)。960k 样本。

#### World Exploration

两种模式:
- **User-Interactive Mode**: 每条用户指令触发一步视觉更新
- **Free-Exploration Mode**: 模型自驱连续生成 trajectory

数据基于 Sekai (walking) + OpenDV (driving),DeQA 过滤 + camera trajectory re-annotation。每个 clip → 4 个 instance (2 input modalities × 2 modes)。200k 样本。

#### Embodied Manipulation

把长程 task 分解成 subtasks:

$$\text{Sub}_i = (l_i, O_{[t_{i-1}:t_i]})$$

其中 $l_i$ 是 subtask 语言指令,$O_{[t_{i-1}:t_i]}$ 是观测序列段,$o_{t_i}$ 是该 subtask 完成时的 keyframe。问题被 reframe 为:预测 (subtask instruction, keyframe state) 的 interleaved sequence。

数据:OXE (920k) + Agi-world Alpha (40k) + Songling Aloha (13k),合计 973k。训练时随机从中间步起,强制模型从任意状态都能 plan 后续。

#### Any-to-Image (X2I)

输入任意 (text + 0/1/multi images),输出单张图。X 是 "any interleaved image-text instruction"。被 paper 当成 single-step multimodal generation / world editing 的基础能力,是通向 X2X 的中间产物。修改对象类型:human / animal / object / text / scene / composite。

数据来源三档:fully real (从 video/image 检索) / semi-real (真实图 + 模型加工) / fully synthetic (T2I 模型生成)。质量过滤按 resolution / clarity / aesthetic,然后 image clustering 做 diversity compact。

### 5.2 SFT 训练细节

两阶段:
- **Stage SFT-1**: 各任务按各自标准分辨率训练。X2I 768px,Visual Guidance/Narrative/Embodied 512px,World Exploration 720px。Visual loss weight 1.0。seq len 16384。TP=8, CP=1。
- **Stage SFT-2**: 高分辨率继续训练。X2I 1024px,其他 720px。visual token 多了 → visual loss weight 0.5。seq len 32768。TP=8, CP=2。

Batch size 1024, LR 6e-6, AdamW($\beta_1=0.9, \beta_2=0.95$), cosine schedule。每 stage 3000 iterations。

直觉:Stage 1 让模型在 "好学" 的分辨率上先 converge,建立 task interface;Stage 2 把分辨率拉上去,这阶段 visual token 暴增(4096 tokens/image),需要重新平衡 modality loss weight。

### 5.3 Reinforcement Learning - 多任务联合 GRPO

Reward system 三特性:generality (跨任务的 aesthetic + alignment)/ task-specificity (OCR、face ID、consistency 各自有专门 reward)/ unified nature (所有 reward normalize 到 [1, 10] 再合并,统一优化)。

GRPO 公式 (DeepSeek 风格):

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q, \{o_i\}_{i=1}^{G}}\!\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\!\Big(\rho_{i,t} \hat A_i,\, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \hat A_i\Big) - \beta\, \mathbb{D}_{\text{KL}}\!\big(\pi_\theta\, \|\, \pi_{\text{ref}}\big)\right]$$

其中:
- $q$ = prompt
- $\{o_i\}_{i=1}^G$ = 对同一 prompt 模型 rollout 出的 $G$ 个 completion (这里 $G=8$)
- $\rho_{i,t} = \pi_\theta(o_{i,t} | q, o_{i,<t}) / \pi_{\text{old}}(o_{i,t} | q, o_{i,<t})$ 是 importance ratio
- $\hat A_i = (r_i - \bar r) / \sigma_r$ 是 group-relative advantage (group 内做 baseline,不训 critic)
- $\bar r, \sigma_r$ 是 group 内 reward 均值/标准差
- $\beta$ 是 KL penalty coefficient,防止 policy 跑离 reference policy
- $\epsilon$ 是 PPO clip ratio (通常 0.1-0.2)

直觉:相比 PPO 不需要单独 value network,group baseline 已经是 low-variance 的 advantage estimator。算力更友好。

参考 GRPO: https://arxiv.org/abs/2402.03300

训练细节:
- Global batch 640
- LR 1e-6
- Rollout $G=8$
- vLLM-based sampling engine,集成在 VeRL 框架
- 每个 batch 混合多任务,鼓励 cross-task synergy
- 单独 X2I + T2I 阶段 (58k X2I + 50k T2I)

每个任务 ~10k 高质量 prompt + 1k human feedback。Reward 从 ~4.5 涨到 >7.1 (Figure 8)。

### 5.4 Discrete Diffusion Adaptation (DiDA) - 20x 推理加速

这是 Emu3.5 的 inference engineering 亮点。

#### 问题

AR 模型生成 1024×1024 image = 4096 tokens,逐个 token decode 需要约 4k step forward。每步都涉及 full KV cache attention,latency 120s+。Diffusion 模型走并行 denoising,4096 个 token 一步同时 refine,只需 ~30 步,大概 1-2 秒。AR 模型在 inference speed 上被 diffusion 吊打。

#### DiDA 思路

把 AR-trained 模型改造成 **discrete diffusion predictor** on visual tokens:

**训练**:
- 自蒸馏数据集:用原 AR 模型生成 (text, image) pair 作为 teacher output
- 改 attention mask:对每个 image 复制成一份 noisy copy。Noisy image token attend (1) causally 到 preceding clean tokens,(2) bidirectionally 到同 image 的其他 noisy tokens。Clean image/text token 保持 causal attention 到 preceding clean tokens。

用 ASCII 画 attention mask (Figure 9):

```
           T1  T2  I1c I2c I3c I4c I1n I2n I3n I4n
T1          ✓
T2          ✓   ✓
I1_clean    ✓   ✓   ✓
I2_clean    ✓   ✓   ✓   ✓
I3_clean    ✓   ✓   ✓   ✓   ✓
I4_clean    ✓   ✓   ✓   ✓   ✓   ✓
I1_noisy    ✓   ✓   ✓   ✓   ✓   ✓   ✓       ✓   ✓
I2_noisy    ✓   ✓   ✓   ✓   ✓   ✓       ✓   ✓   ✓
I3_noisy    ✓   ✓   ✓   ✓   ✓   ✓       ✓   ✓   ✓   ✓
I4_noisy    ✓   ✓   ✓   ✓   ✓   ✓       ✓   ✓   ✓   ✓
```

注意 noisy image 区域是个 dense 9×9 块 (bidirectional),其余保持下三角 (causal)。

**推理**:
- Text 部分: 仍然 causal AR (一字一字出)
- Image 部分: 整个 4096 tokens 同时初始化成 noise mask token,迭代 denoise $K$ 步,每步所有 token 同时预测。然后取 final

用 flow matching / discrete diffusion 类似 MaskGIT 的迭代:

$$\text{Step } k: \quad x_{\text{noisy}}^{(k)} \to \hat x_{\text{clean}}^{(k)} = \arg\max_c p_\theta(c | x_{\text{ctx}}, x_{\text{noisy}}^{(k)})$$

只 update 一部分高熵位置 (类似 MaskGIT 的 cosine masking schedule)。

参考 discrete diffusion: https://arxiv.org/abs/2510.24717  
参考 MaskGIT 原理: https://arxiv.org/abs/2202.04200

#### Infrastructure 上的两个新点

1. **FlexAttention per-row block mask**: 4D attention mask 在长序列下 memory 巨大,改用 per-row block mask 编码 causal + bidirectional + region-specific,无需 store full attention matrix。

2. **FSM-based scheduling**: inference 时 text/image phase 动态切换,有限状态机调度,preallocate resource + async request + runtime state reuse + FP8 quant,4-device 上 ≥50% speedup。

#### 结果 (Table 16)

| Variant | Resolution | Gen Tokens | Method | Time(s) naive / FlagScale | GenEval | DPG | GEdit |
|---|---|---|---|---|---|---|---|
| Emu3 8B | 720² | 8100 | AR | 260 / 68 | 0.66 | 80.60 | - |
| Emu3.5 34B | 1024² | 4096 | AR | 512 / 120 | 0.86 | 88.26 | 7.59 |
| Emu3.5 34B | 1024² | 4096 | **DiDA** | **22 / 10** | 0.86 | 87.46 | 7.56 |

DiDA 把 4096-token image 生成时间从 120s 压到 10s,**20x 加速**,benchmark 几乎不掉。第一次让 AR 模型在速度上比肩连续 diffusion。

---

## 6. 实验 - 全面解读关键数字

### 6.1 T2I - TIIF Bench (Table 4)

满分 100,11 个 subtask。Top 对比:

| Model | Overall |
|---|---|
| GPT-Image-1 | 89.15 |
| Qwen-Image | 86.14 |
| Gemini 2.5 Flash Image (Nano Banana) | 86.02 (推算) |
| Seedream 3.0 | 86.02 |
| **Emu3.5** | **89.48** |

Text subtask:Emu3.5 拿 100.00 (满分),FLUX.1 dev 仅 66.67。AR-based 对比基线如 Janus-Pro 60.00,Infinity 80.00,Show-o 63.33。Emu3.5 是 AR 模型里第一个跟 GPT-Image-1 同级别的。

### 6.2 OneIG-Bench (Table 5, Table 6)

5 个维度: Alignment / Text / Reasoning / Style / Diversity。

EN:Emu3.5 overall 0.564,Nano Banana 0.550,Qwen-Image 0.539,GPT-Image-1 0.533。
ZH:Emu3.5 0.529,仅次于 Qwen-Image 0.548。

Text 维度 Emu3.5 EN 0.994(几乎是满分),ZH 0.941——这是 tokenizer 在 text-rich 数据训练的直接红利。

### 6.3 Text rendering - LeX-Bench (Table 7)

3 难度:Easy / Medium / Hard,指标 pNED↓ (normalized edit distance) + recall↑。

Hard:Emu3.5 pNED 4.39, recall 0.87,SOTA。Qwen-Image pNED 5.56, recall 0.74。GPT-Image-1 pNED 5.52, recall 0.70。

直觉:text rendering 的难点是长字符 + 复杂 layout + 字体风格多样,Emu3.5 tokenizer 直接把像素切到 token,token-level AR 学 long-range 字符 pattern 比 diffusion 的连续 latent 更直接。

### 6.4 CVTG-2K (Table 9)

每 prompt 2-5 个 text region。Word Accuracy Emu3.5 0.9123,GPT-Image-1 0.8569,Qwen-Image 0.8288,Nano Banana 0.7364。

### 6.5 Any-to-Image - ImgEdit (Table 10)

737 samples,9 个 edit subtask,GPT-4.1 judge。Top:

| Model | Overall |
|---|---|
| Emu3.5 | 4.41 |
| Qwen-Image-Edit | 4.35 |
| Gemini 2.5 Flash Image | 4.28 |
| GPT-Image-1 High | 4.20 |
| FLUX.1 Kontext Pro | 4.00 |

### 6.6 GEdit-Bench (Table 11)

606 samples,11 subtask,GPT-4o judge,三指标 G_SC / G_PQ / G_O。

Emu3.5 G_O 7.59,仅次于 Qwen-Image-Edit 7.56(其实 7.59 > 7.56,Emu3.5 最高)。G_SC (semantic consistency) 8.11,仅次于 Qwen-Image-Edit 8.15。

### 6.7 OmniContext - subject-driven generation (Table 12)

400 samples,7 sub-dim。Emu3.5 average 8.82,超过 GPT-4o 的 8.80,超过 Nano Banana 7.84。Object 维度强 (8.89-9.46),Character + Object 综合 8.78。

### 6.8 ICE-Bench (Table 13)

6538 samples,31 个 task(creation / ref / global edit / local edit / controllable / style transfer / face swap)。

| Model | Task 1-31 Overall |
|---|---|
| Emu3.5 | 0.637 |
| Gemini 2.5 Flash Image | 0.631 |
| Gemini 2.5 Flash Image Preview | 0.630 |
| Qwen-Image-Edit-2509 | 0.616 |

Task 2 (face reference) / Task 3 (style reference) / Task 31 (face swap) Emu3.5 偏弱,作者标注为未来工作。

### 6.9 4 个 interleaved 任务对比 Nano Banana (Table 14)

ChatGPT 自动 preference。

| Task | Win | Tie | Lose |
|---|---|---|---|
| Visual Narrative | 49.2 | 10.3 | 40.5 |
| Visual Guidance | 51.5 | 9.5 | 39.0 |
| World Exploration | **65.5** | 0.0 | 34.5 |
| Embodied Manipulation | **67.1** | 2.4 | 30.5 |

World Exploration 和 Embodied Manipulation 的胜率大幅领先——这是 Emu3.5 的核心 thesis 验证:**长程 multimodal world modeling 通过 native interleaved AR 训练获得,封闭源 SOTA 在这块没专门训**。

### 6.10 Tokenizer reconstruction (Table 15)

Tokbench 评测 + 60k 自建集,512×512 分辨率。

| Method | Type | T-ACC_me↑ | T-NED_m↑ | F-Sim_m↑ | rFID↓ |
|---|---|---|---|---|---|
| VQGAN | VQ | 0.76 | 8.99 | 0.08 | 1.19 |
| Chameleon | VQ | 2.67 | 17.82 | 0.13 | 1.03 |
| LlamaGen | VQ | 3.71 | 20.17 | 0.11 | 0.68 |
| TokenFlow | RQ | 20.59 | 49.56 | 0.11 | 0.55 |
| O-MAGVIT2 | LFQ | 16.25 | 80.48 | 0.13 | 0.45 |
| O-MAGVIT2 pretrain | LFQ | 39.59 | 87.38 | 0.14 | 0.42 |
| **Emu3.5 vanilla** | IBQ | **53.22** | **91.78** | 0.14 | **0.49** |
| Emu3.5 diffusion | IBQ | 51.11 | 75.96 | 0.14 | 0.42 |

T-ACC_m = medium-scale text accuracy,T-NED_m = medium-scale text normalized edit distance,F-Sim_m = medium-scale face similarity。Emu3.5 在 text 上几乎是 O-MAGVIT2 的 1.3-3x。这直接来自 131k codebook + SigLIP distillation + text-rich training data。

直觉:tokenizer 好坏是 native multimodal AR 的"下限"。Tokenizer 模糊了,后面再大模型也只能 hallucinate,认不出文字/人脸。Emu3.5 把 tokenizer 当成一等公民投入了大量 engineering。

---

## 7. 我对这篇 paper 的几条 takeaway / intuition

1. **Native multimodal world model 是个数据格式问题**。不是新 architecture,而是把视频切成 keyframe+ASR 时间交错序列,丢给标准 transformer 做 NTP。这暗示 "world model" 不需要 latent dynamics / world state 的特殊参数化,只要数据足够长足够多,AR transformer 自己学 dynamics。

2. **Tokenizer capacity 决定上限**。131k codebook + IBQ 可微量化 + SigLIP distillation + diffusion decoder,dual-decoder 设计让 1024 token 表达 1024² pixel 细节。这块比 Emu3 改进巨大(Emu3 用 32k codebook, 4096+ tokens per image)。

3. **Video-interleaved 55% 的数据配比**是有意识的选择。仅看 (image, caption) pair 学不到 time/causal context;仅看短视频学不到长程 reasoning。这一点是 Emu3.5 跟 GPT-4o / Nano Banana 的差异之处。

4. **DiDA 是非常聪明的 inference engineering**。它没改 AR model 的本质,只是改 attention mask 让 image tokens 之间 bidirectional,从而把 NTP 改成 masked denoising。这意味着 AR 训练成本 + diffusion 推理速度兼得。20x 加速不丢精度是巨大 win。

5. **Unified multi-task RL 的 reward design** 值得注意:把所有 reward normalize 到 [1,10] 再合并,避免 hacking 单一 reward。这是 unified post-training 的工程经验。GRPO + group baseline 省掉 critic,大模型 RL 才能跑得动。

6. **Limitations**:
   - 32k context length,长视频仍受限。1024×1024 image = 4096 token,12 张图就 50k,装不下
   - Embodied manipulation 只到 keyframe 层,没到 dense action control
   - Video decoder 是事后挂的 (Wan2.2 based),不是 end-to-end NTP,跟 paper 主张略张力
   - World Exploration 评估自动 judge (ChatGPT) 为主,human eval 缺失

参考:
- Emu3: https://arxiv.org/abs/2409.18869
- Qwen3: https://arxiv.org/abs/2505.09388
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Qwen-Image: https://arxiv.org/abs/2508.02324
- IBQ: https://arxiv.org/abs/2412.02692
- REPA: https://arxiv.org/abs/2410.06940
- SigLIP: https://arxiv.org/abs/2303.15343
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- VeRL: https://arxiv.org/abs/2409.19256
- MaskGIT: https://arxiv.org/abs/2202.04200
- Discrete diffusion: https://arxiv.org/abs/2510.24717
- Flow matching: https://arxiv.org/abs/2210.02747
- SD3.5: https://arxiv.org/abs/2403.03206
- Wan2.2: https://github.com/Wan-Video/Wan2.1
- Sekai: https://arxiv.org/abs/2506.15675
- OpenX Embodiment: https://arxiv.org/abs/2310.08864
- Agibot World: https://arxiv.org/abs/2503.06669
- FlagScale: https://github.com/FlagOpen/FlagScale
- FLUX Kontext: https://arxiv.org/abs/2506.15742
- PySceneDetect: https://github.com/Breakthrough/PySceneDetect
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
- Tokbench: https://arxiv.org/abs/2505.18142
- DeQA: https://arxiv.org/abs/2502.11187
- DINOv2: https://arxiv.org/abs/2304.07193
- ImgEdit: https://arxiv.org/abs/2505.20275
- GEdit-Bench: https://arxiv.org/abs/2504.17761
- OmniContext: https://arxiv.org/abs/2506.18871
- ICE-Bench: https://arxiv.org/abs/2503.14482
- LeX-Bench: https://arxiv.org/abs/2503.21749
- OneIG-Bench: https://arxiv.org/abs/2506.07977
- TIIF-Bench: https://arxiv.org/abs/2506.23301 (推测)
- LongText-Bench: https://arxiv.org/abs/2507.22058
- CVTG-2K: https://arxiv.org/abs/2503.23461

整体看,Emu3.5 是 BAAI 把 "next-token prediction is all you need" 这个 thesis 在 scale + post-training + inference efficiency 三个维度同时推进到 SOTA 级别的工作。它没发明新 architecture,但每一层 (tokenizer / data / pretrain / SFT / RL / inference) 都做对了工程,把 AR native multimodal 推到能跟 GPT-Image-1 / Nano Banana 平起平坐,并在 world modeling / 长程 reasoning 这种新维度上领先。这是一篇值得 read carefully 的工程论文,不是一篇 thesis-type paper。
