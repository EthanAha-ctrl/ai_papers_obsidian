---
source_pdf: DeepGen1.0.pdf
paper_sha256: 6d2ba01d09779d25e4f7d6e6a093ed2e1afd438d40548dfffca43c03cc4c05a1
processed_at: '2026-08-03T18:39:28-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeepGen 1.0 人话版

## 一句话讲清楚

这篇 paper 就在讲一件事:**别再卷参数量了,5B 的小模型配合聪明架构和训练策略,能把 80B 的大模型按在地上摩擦**。

## 背景:大家都怎么了

最近一年 unified multimodal model 这条赛道上,大家都在比谁参数大、谁数据多。HunyuanImage 搞到 80B 用了 5B 样本,LongCat 用 1.2B 样本,Qwen-Image 加上 Edit 版本合计 54B。社区基本形成一种共识:小模型做不了 unified multimodal,容量不够。

但 DeepGen 团队发现一个有意思的事 — 他们翻 recent benchmarks,看到 Lumina-DiMOO 只有 8B,在 DPG-Bench 上 86.04,居然超过 14B 的 BAGEL (85.10)。类似反例还不少。这说明在 unified multimodal 这个 paradigm 下,scaling 的规律和 LLM 不太一样,parameter count 不一定是主导因素。

于是他们决定赌一把:用 5B(3B VLM + 2B DiT)、~50M 样本,做到能和 80B 掰手腕。结果还真做成了。

## 核心问题在哪

你要让一个 model 同时做 image generation 和 editing,还要带 reasoning,本质上你是在组合两个东西:
- 一个负责 "看懂" 的 VLM(Qwen-2.5-VL 3B)
- 一个负责 "画" 的 DiT(SD3.5-Medium 2B)

这俩是各自预训练出来的,latent space 完全不一样。怎么让 VLM 理解到的东西,高效地传给 DiT 让它画出来?这就是 bottleneck。

之前的人怎么做的?大概三种路线:
1. **只拿 VLM 最后一个 layer 的输出** (Qwen-Image、OmniGen2、UniWorld-V1) — 问题:final layer 偏抽象语义,纹理、位置、颜色这些细节早丢了,DiT 拿到的是 "high-level summary",画 fine-grained 东西就费劲
2. **Deep fusion**(BAGEL、HunyuanImage) — VLM 和 DiT 每层 share attention,信息流充分但参数爆炸,优化也难
3. **Average pooling 多层**(Mammoth2) — 平均了就把细节平均掉了

DeepGen 选择第四条路:Stacked Channel Bridging (SCB)。

## SCB 的直觉

SCB 的核心思想很朴素:**别只看 VLM 最后一层,从底层、中层、高层各抽几个 layer,把它们的信息全部保留下来,再压缩给 DiT**。

具体三步:

**(1) 注入 Think Tokens**

在 VLM 的输入序列里塞 128 个 learnable tokens,这些 tokens 和 text/visual tokens 一起过所有 self-attention layer。它们干什么?当 "reasoning buffer" — VLM 知识被慢慢 distill 到这些 tokens 里。BAGEL 是让 model 先 explicit 生成一段 reasoning text 再画图,DeepGen 不生成文本,直接用 learnable tokens 做 implicit CoT,推理时更高效。

**(2) 选 6 个 layer 均匀采样**

从 VLM 的 low/mid/high level 各均匀采 6 个 layer 的 hidden states。低层抓纹理颜色,中层抓 object parts 和 attribute binding,高层抓 scene semantics。这一点参考了 Wang et al. 2025 (https://aclanthology.org/2025.acl-long.827/) 的发现 — VLM 里 visual information 是分布式 encoded 在多个 layer 的,不是全在最后一层。

**(3) Channel-wise concat 再融合**

这一步是 SCB 名字的由来。给定 6 个 layer 的 hidden states $[x_1, \dots, x_6] \in \mathbb{R}^{L \times d}$,其中 $L$ 是 sequence length(含 think tokens),$d$ 是 VLM hidden dim。

沿 channel 维度 concat(不是 token 维度!),得到 $\mathbb{R}^{L \times 6d}$,再用 2-layer MLP 投影到 DiT 的 width $d_{DiT}$,最后过 6 层 Transformer encoder 融合:

$$c = \text{Encoder}(\text{MLP}(\text{Concat}_{ch}(x_1, \dots, x_6))) \tag{1}$$

输出 $c \in \mathbb{R}^{L \times d_{DiT}}$,作为 DiT 的 multimodal condition。

为什么 channel 而不是 token concat?你想,如果沿 token concat,sequence length 变成 $6L$,DiT 的 self-attention 是 $O(L^2)$,直接 36 倍计算量。Channel concat 保持 $L$ 不变,只是 hidden dim 暂时变 6 倍,后续 MLP 立即压回 $d_{DiT}$,overhead 极小。这就是 "lightweight" 的关键。

**直觉**:这个设计相当于让 connector 自己学 "我应该从哪层抽多少信息",而不是 hardcode "只用最后一层" 或 "全部平均"。VLM 里 information 是 distributed 的,SCB 给 connector 一个 learnable 的方式去 aggregate。

## 三阶段训练

### Stage 1: Alignment Pre-Training

VLM 和 DiT 各自预训练过,latent space 不对齐。如果上来就 joint 训练,容易炸。所以 Stage 1 只训练 SCB connector 和 128 个 think tokens,其他全部 frozen。

- 200k iterations,batch size 512,lr 1e-4
- 35M generation pairs + 6.6M editing triplets ≈ 42M samples
- 64×H200,固定 512×512 resolution

这一步类似 LLaVA 的 stage-1 projection pre-training,让 connector 学会 "翻译" VLM features 到 DiT 的语言。

### Stage 2: Joint Supervised Fine-Tuning

Stage 2 unfreeze DiT 全参数,VLM 用 LoRA(rank 64, alpha 128) 微调,400k iterations。

为什么不 full fine-tune VLM?因为 VLM 里面 encode 了大量 world knowledge,full fine-tune 容易 catastrophic forgetting。WISE benchmark 测的就是 cultural/temporal/spatial/biology/physics/chemistry 这些知识,模型 reasoning 全靠 VLM 里这些 knowledge。LoRA 限制 update 在 low-rank subspace,既能让 VLM 适应下游任务,又保住预训练知识。

数据上,这个阶段引入了:
- 11M general generation
- 6.6M general editing
- 150k reasoning generation(来自 UniReason, https://arxiv.org/abs/2602.02437)
- 100k reasoning editing
- 560k text rendering

reasoning data 量虽小但关键 — WISE 和 RISE 上的 leading performance 直接来源于此。

### Stage 3: MR-GRPO 强化学习

这是最有创新性的部分,也是我觉得最值得细讲的地方。

#### 先说背景:为什么需要 RL

SFT 之后的 model 已经不错了,但 RL 能进一步对齐人类偏好。LLM 领域 RLHF 早就标配,但 diffusion model 做 RL 才刚开始。Diffusion 是连续 trajectory(50 步 denoising),不像 LLM 是离散 token,怎么定义 action、reward、policy 都要重新想。

#### GRPO 怎么用到 flow matching 上

GRPO(DeepSeek 提出,https://arxiv.org/abs/2402.03300)的思路:对一个 prompt $h$,sample 一组 $G=8$ 个 images,用 reward function 给每个 image 打分,组内 normalize 算 advantage,再 PPO-style 更新 policy。

DeepGen 用了 3 个 reward function:
1. **VLM-based pairwise preference reward**(来自 Unified-Reward-Think, https://arxiv.org/abs/2505.03318): 组内两两比较算 win rate
2. **OCR reward**(PaddleOCR 3.0, https://arxiv.org/abs/2507.05595): 测文字渲染准确性
3. **CLIP score**: 测整体图文一致性

每个 reward 独立 normalize:

$$A_k^i = \frac{R_k(x_0^i, h) - \text{mean}(\{R_k(x_0^j, h)\}_{j=1}^G)}{\text{std}(\{R_k(x_0^j, h)\}_{j=1}^G)} \tag{2}$$

变量解释:
- $R_k(x_0^i, h)$: 第 $k$ 个 reward 对第 $i$ 个生成 image 的评分
- $\text{mean}, \text{std}$: 在 group 内 $G=8$ 个 samples 上算
- $A_k^i$: sample $i$ 在 reward $k$ 上的标准化 advantage,大致服从 $\mathcal{N}(0,1)$

然后 weighted aggregation:

$$\hat{A}^i = \sum_{k=1}^3 w_k A_k^i$$

权重分 prompt 类别给:
- Text rendering prompts: $w_{pref}=0.2, w_{CLIP}=0.1, w_{OCR}=0.7$
- General T2I prompts: $w_{pref}=0.7, w_{CLIP}=0.3, w_{OCR}=0$

**为什么必须 decoupled normalization?** 三个 reward 的 scale 和 variance 完全不一样 — win rate 在 [0,1],OCR 在 [0,1],CLIP score 大概在 [0.2,0.4]。如果直接 raw reward 加权,高 variance 的 reward(比如 CLIP)会 dominate 梯度,其他 reward 就没用了。Decoupled norm(来自 GDPO, https://arxiv.org/abs/2601.05242)让每个 reward 在 policy update 里贡献相对均衡。

主目标:

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{h \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left( \min(r_t^i(\theta) \hat{A}^i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}^i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right] \tag{3}$$

变量解释:
- $h \sim \mathcal{D}$: 从训练分布采样 prompt
- $G=8$: group size
- $T=50$: denoising steps
- $r_t^i(\theta) = p_\theta(x_{t-\Delta t}^i | x_t^i, h) / p_{\theta_{\text{old}}}(x_{t-\Delta t}^i | x_t^i, h)$: per-step importance ratio,新旧 policy 在该 step 的概率比
- $\hat{A}^i$: aggregated advantage
- $\epsilon = 1 \times 10^{-4}$: clip range(很小,因为 flow matching 的 log-prob 是连续的)
- $\beta = 5 \times 10^{-7}$: KL coefficient
- $\pi_\theta, \pi_{\text{ref}}$: 当前 policy 和 reference policy(SFT 后 frozen 的 model)

KL 在 velocity space 算:

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \|\hat{v}_\theta(x_t, t) - \hat{v}_{\text{ref}}(x_t, t)\|^2 \tag{4}$$

- $\hat{v}_\theta(x_t, t)$: 当前 model 预测的 velocity(flow matching 里 $dx_t = v dt$)
- $\hat{v}_{\text{ref}}(x_t, t)$: reference model 预测的 velocity
- Euclidean distance squared,在 Gaussian 假设下等价于 KL 去掉常数

#### Noise-preserving stochastic sampling — 一个被忽视的细节

flow matching 默认是 deterministic ODE:$dx_t = \hat{v}_\theta(x_t, t) dt$。RL 需要 exploration,所以要转 SDE 加 noise。但 standard Flow-SDE 会注入超过 scheduler 预期 noise level 的随机性,sample quality 下降,reward signal 也不准。

DeepGen 采用 noise-preserving 策略(https://arxiv.org/abs/2509.05952):

$$x_{t-\Delta t} = (1-(t-\Delta t)) \hat{x}_0 + (t-\Delta t) \cos\left(\frac{\eta \pi}{2}\right) \hat{x}_1 + (t-\Delta t) \sin\left(\frac{\eta \pi}{2}\right) \epsilon \tag{6}$$

变量解释:
- $t \in [0,1]$: timestep,0 是 clean,1 是 pure noise
- $\Delta t$: step size($= 1/50$)
- $\hat{x}_0 = x_t - t \hat{v}_\theta$: predicted clean sample
- $\hat{x}_1 = x_t + (1-t) \hat{v}_\theta$: predicted noise
- $\epsilon \sim \mathcal{N}(0, I)$: fresh Gaussian noise
- $\eta = 1.0$: stochasticity strength
- $\cos(\eta \pi / 2), \sin(\eta \pi / 2)$: 振幅分解

**直觉**:flow matching 的 forward process 是 $x_t = (1-t) x_0 + t x_1$,所以从 $x_t$ 预测 $x_0$ 与 $x_1$ 后,用 cosine-sine 分配 deterministic 和 stochastic 分量,保证总 noise level 严格等于 $t - \Delta t$。这样 sample 的 noise level 始终和 scheduler 对齐,reward signal 才准。

Log-prob 简化为:

$$\log p_\theta(x_{t-\Delta t} | x_t) = -\|x_{t-\Delta t} - \mu_\theta(x_t, t)\|^2 \tag{7}$$

- $\mu_\theta(x_t, t) = (1-(t-\Delta t)) \hat{x}_0 + (t-\Delta t) \cos(\eta \pi / 2) \hat{x}_1$: 采样的 deterministic 部分

这个简化去掉了标准 log-prob 里的 variance normalization term,避免小 noise level 时的数值不稳定。

#### Auxiliary SFT Loss — 我觉得最关键的发现

DeepGen 团队发现一个反直觉现象:**只靠 KL regularization,RL 训练超过 ~1000 steps 后,model 在 complex instruction comprehension(比如 reasoning generation)上的 performance 会逐渐下降**。

为什么?KL 在 velocity space 约束 trajectory,每一步 velocity 不能偏离 reference 太远 — 这是 **process-level guidance**。但 KL 允许 final outcome 偏离 SFT distribution,只要每步偏离不大。长时间训练,微小 drift 累积,final image distribution 还是会偏离 SFT 学到的高质量区域。

所以他们引入 auxiliary SFT loss:

$$\mathcal{L}_{\text{total}} = (1-\lambda) \mathcal{L}_{\text{GRPO}} + \lambda \mathcal{L}_{\text{SFT}} \tag{5}$$

- $\lambda = 1 \times 10^{-4}$: 非常小的 mixing coefficient,确保 SFT loss 只做 anchor 不主导 optimization
- $\mathcal{L}_{\text{SFT}}$: 标准 flow matching loss $\mathbb{E}[\|\hat{v}_\theta - v_{\text{target}}\|^2]$,在高质量 SFT dataset 上算

**直觉**:KL 是 "don't go too far" 的负向约束,SFT loss 是 "stay close to good region" 的正向 anchor。两者互补,缺一不可。这就像你教小孩骑车,KL 是 "别骑太快别摔",SFT loss 是 "记得骑回车道中间"。

Table 7 的数据很 striking:
- w/o Auxiliary SFT Loss: UniGenBench 从 75.69 跌到 74.33,Text score 从 35.06 跌到 33.33
- Fig.6(a) 更直观:从 ~300 steps 开始,w/o SFT Loss 的版本 performance 持续下降,最终低于初始 checkpoint — RL 不仅没改善反而损害 model

这是 paper 最 actionable 的 finding,对未来 diffusion RL 工作有指引意义。

## 结果

直接上数字,自己感受:

| Benchmark | DeepGen 1.0 (5B) | 最强对手 | 差距 |
|-----------|------------------|----------|------|
| WISE (reasoning gen) | 0.73 | HunyuanImage 3.0 (80B): 0.57 | +28% |
| UniREditBench (reasoning edit) | 77.5 (SFT) | Qwen-Image-Edit (27B): 56.5 | +37% |
| DPG-Bench (general gen) | 87.90 | Qwen-Image (27B): 88.32 | -0.5% |
| GenEval | 0.87 | Qwen-Image: 0.87 | 持平 |
| CVTG-2K Word Acc | 0.7533 (RL) | GLM-Image: 0.9116 | -18% |

5B model 在 reasoning generation/editing 上超越 80B,在 general generation 上接近 SOTA,这就是核心 claim。

值得注意几个细节:
- **RL 对 text rendering 帮助巨大**:Word Accuracy 从 SFT 0.6605 飙到 RL 0.7533(+14%),这是 OCR reward 的直接效果
- **RL 对 reasoning editing 反而轻微下降**:RISE 从 SFT 13.3 跌到 RL 10.8,UniREditBench 从 77.5 跌到 75.7。说明当前 reward 设计(preference + OCR + CLIP)没有覆盖 reasoning editing 所需的 knowledge-grounded editing signal
- **T2I-CoREBench 的 R-RR(Reconstructive Reasoning)普遍低**:DeepGen 19.6,GPT-Image-1 也只有 47.5,说明 reconstructive reasoning 是 unified model 的普遍短板

## 我的几点直觉

### 1. Unified multimodal 的 scaling law 和 LLM 不一样

LLM 里 parameter count 主导 performance,因为 LLM 是单一 model,容量就是瓶颈。Unified multimodal 是两个 model 的组合,VLM 和 DiT 之间的 interface(alignment)是 bottleneck。HunyuanImage 80B 通过 brute-force scaling 增加 capacity,但 alignment 仍用 final-layer conditioning,信息 transfer 效率低。DeepGen 5B 通过 SCB 在 multiple levels 进行 alignment,information transfer 更 dense,反而用更少参数达到更高 performance。

这个观察如果被后续工作验证,可能重塑整个领域的架构设计哲学。

### 2. Implicit CoT vs Explicit CoT 的 trade-off

BAGEL 用 explicit textual CoT(先 generate 一段 reasoning text 再画图),DeepGen 用 implicit learnable tokens。两者在 WISE 上 DeepGen 0.73 vs BAGEL 0.70,DeepGen 略胜。

这暗示在 small model 上 implicit CoT 可能更 efficient — 因为 explicit CoT 需要 capacity 来 generate coherent reasoning text,而 small model 的 capacity 紧张。Think tokens 把 reasoning 压缩到 dense vector,效率更高,但代价是失去 interpretability。这个 trade-off 值得深挖。

### 3. RL for diffusion 的 stability 问题

Auxiliary SFT loss 的发现让我想到 AlphaGo 的 policy network anchor、RLHF 中的 KL penalty。本质上 RL optimization 在 high-capacity model 上容易 collapse,需要某种 "pull-back" mechanism。

KL 是 process-level pull-back(每步别偏离太远),SFT loss 是 outcome-level pull-back(最终结果要回到高质量区域)。两者互补,缺一不可。未来 diffusion RL 工作应该都采用类似 dual constraint 设计。

更进一步,RL 在 reasoning editing 上的退化表明,reward function 设计仍是 open problem。VLM-based reward 在 reasoning 场景下可能本身不可靠,需要 task-specific verifier(如 knowledge-grounded editing verifier),或 self-rewarding 机制。

### 4. Data efficiency 的启示

50M samples vs 5B samples(100× reduction)说明:
- 大规模 raw dataset(LAION、CC12M)在 unified model 训练中 marginal value 递减
- High-quality instruction data(ShareGPT-4o、BLIP3o)的 marginal value 远高于 raw pairs
- Reasoning data 即便只有 150k,对 reasoning capability 贡献巨大

这与 LLM 中的 Phi、Qwen 系列发现一致 — data quality > data quantity。未来 unified multimodal model 训练可能更多关注 data curation 而非 scaling。

### 5. 没解决的问题

paper 没充分讨论:
- **Resolution 限制**:固定 512×512,无法 native 高分辨率生成
- **RL 在 reasoning editing 上的退化**:需要 reward function 改进
- **Think tokens 可解释性**:没分析 think tokens 学到什么具体 reasoning pattern
- **Long-context editing**:多 reference image 或 multi-turn editing 没充分探索
- **Video generation**:当前仅 image

未来可能的方向:
- SCB 扩展到 video(VLM + Video DiT)
- Explicit + implicit CoT 混合
- Process reward model(PRM)替代 outcome reward
- Self-play RL 通过 VLM-as-judge 持续迭代
- 扩展到 long-context,支持 long-document grounded editing

## 总结

DeepGen 1.0 最大的贡献,是用 5B 击败 80B 这个 headline 背后揭示的一件事:**unified multimodal model 的 scaling behavior 与 LLM 不同,interface 是 bottleneck,不是 capacity**。

SCB 解决 VLM-DiT alignment bottleneck,MR-GRPO with auxiliary SFT loss 解决 RL stability,50M data strategy 解决 data efficiency。这三个 finding 对整个 unified multimodal 领域有普遍指导意义 — 未来工作不必盲目 scale,而应关注 interface design 与 training stability。

这对资源有限的研究者是个好消息:你不需要 80B、不需要 5B 样本、不需要万卡集群,也能做出有竞争力的 unified multimodal model。这才是 paper 真正的民主化意义。

## References

- DeepGen 1.0 GitHub: https://github.com/DeepGenTeam/DeepGen
- DeepGen 1.0 HuggingFace: https://huggingface.co/DeepGenTeam/DeepGen-1.0
- DeepGen 1.0 Datasets: https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0
- BAGEL: https://arxiv.org/abs/2505.14683
- HunyuanImage 3.0: https://arxiv.org/abs/2509.23951
- Qwen-Image: https://arxiv.org/abs/2508.02324
- LongCat-Image: https://arxiv.org/abs/2512.07584
- BLIP3-o: https://arxiv.org/abs/2505.09568
- Qwen-2.5-VL: https://arxiv.org/abs/2502.13923
- Skywork UniPic 2.0: https://arxiv.org/abs/2509.04548
- SigLIP: https://arxiv.org/abs/2303.15343
- OpenUni: https://arxiv.org/abs/2505.23661
- MetaQuery-XL: https://arxiv.org/abs/2504.06256
- UniWorld-V1: https://arxiv.org/abs/2506.03147
- Mammoth2: https://arxiv.org/abs/2511.18262
- GRPO: https://arxiv.org/abs/2402.03300
- Pref-GRPO: https://arxiv.org/abs/2508.20751
- GDPO: https://arxiv.org/abs/2601.05242
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- DanceGRPO: https://arxiv.org/abs/2505.07818
- LoRA: https://arxiv.org/abs/2106.09685
- WISE: https://arxiv.org/abs/2503.07265
- UniREditBench: https://arxiv.org/abs/2511.01295
- UniReason: https://arxiv.org/abs/2602.02437
- T2I-CoREBench: https://arxiv.org/abs/2509.03516
- RISE: https://arxiv.org/abs/2504.02826
- CVTG-2K (TextCrafter): https://arxiv.org/abs/2503.23461
- DPG-Bench (ELLA): https://arxiv.org/abs/2403.05135
- GenEval: https://arxiv.org/abs/2310.11513
- Unified-Reward-Think: https://arxiv.org/abs/2505.03318
- PaddleOCR 3.0: https://arxiv.org/abs/2507.05595
- CLIP: https://arxiv.org/abs/2103.00020
- ShareGPT-4o-Image: https://arxiv.org/abs/2506.18095
- Echo-4o-Image: https://arxiv.org/abs/2508.09987
- OpenGPT4o-Image: https://arxiv.org/abs/2509.24900
- NHR-Edit: https://arxiv.org/abs/2507.14119
- GPT-Image-Edit: https://arxiv.org/abs/2507.21033
- Pico-Banana: https://arxiv.org/abs/2510.19808
- RedCaps: https://arxiv.org/abs/2111.11431
- CC-12M: https://arxiv.org/abs/2102.08981
- Flux-Reason-6M: https://arxiv.org/abs/2509.09680
- LAION-5B: https://arxiv.org/abs/2210.08402
- LLaVA-OneVision-1.5: https://arxiv.org/abs/2509.23661
- Gemini 2.5 Pro: https://deepmind.google/models/gemini/pro/
- GLM-Image: https://z.ai/blog/glm-image
- Coefficients-preserving sampling: https://arxiv.org/abs/2509.05952
- Wang et al. 2025 ACL: https://aclanthology.org/2025.acl-long.827/

---

# DeepGen 1.0 深度解读 — 一个 5B unified multimodal model 如何击败 80B 巨兽

## 1. 背景: unified multimodal model 的 scaling 困境

近一年以来, unified multimodal model(同时做 understanding + generation + editing)的出现,本质上反映了 community 对 "纯 diffusion model 语义理解能力不足" 这一根本问题的回应。典型的代表作包括 BAGEL (14B, https://arxiv.org/abs/2505.14683)、HunyuanImage 3.0 (80B, https://arxiv.org/abs/2509.23951)、Qwen-Image + Qwen-Image-Edit (合计 54B, https://arxiv.org/abs/2508.02324)、LongCat-Image + Edit (26B, https://arxiv.org/abs/2512.07584)、Emu3.5 (34B) 等。这些 model 共同特征是参数量巨大、训练样本数动辄 billions(HunyuanImage 用了 5B samples,LongCat 用 1.2B samples),并且经常需要分开 generation 和 editing 两个 model 部署。

DeepGen 1.0 的核心 motivation 来自一个被忽视的观察:**model performance 不与 parameter count 单调正相关**。在 Fig.2 中可以看到 Lumina-DiMOO (8B) 在 DPG-Bench 拿到 86.04,反而超过 14B 的 BAGEL (85.10)。这暗示在 unified multimodal paradigm 下,architecture synergy 与 data-centric strategy 比 raw scaling 更重要。DeepGen 1.0 用 5B 参数(3B VLM + 2B DiT)、~50M 训练样本,在 WISE 上比 80B HunyuanImage 高 28%、在 UniREditBench 上比 27B Qwen-Image-Edit 高 37%。

## 2. Architecture: VLM-DiT dual-branch 与 Stacked Channel Bridging

### 2.1 整体结构

DeepGen 1.0 采用 VLM-DiT 范式,与 BAGEL、Qwen-Image、BLIP3-o (https://arxiv.org/abs/2505.09568) 一脉相承,而与 HunyuanImage 3.0、BAGEL 的 deep-fusion 路线不同。具体组件:

- **VLM**: Qwen-2.5-VL 3B (https://arxiv.org/abs/2502.13923) — 提供 multimodal understanding、world knowledge 与 reasoning 能力
- **DiT**: SD3.5-Medium 2B,从 Skywork UniPic 2.0 (https://arxiv.org/abs/2509.04548) 初始化,本身已具备 joint generation-editing 能力
- **Connector**: SigLIP visual encoder (https://arxiv.org/abs/2303.15343) + 6 层 Transformer encoder(来自 OpenUni,https://arxiv.org/abs/2505.23661),负责将 VLM features 投影到 DiT input width 并深度融合
- **Dual-branch visual encoding**: ViT encoder 为 VLM 提供高层语义;VAE encoder 为 DiT 提取 compressed latent,作为 reference image 条件

DiT 的 input sequence 是把 multimodal condition tokens、reference image VAE latents、target image 的 noise tokens 拼接成单一 sequence,通过 self-attention 让 condition 与 generation signal 互相交互。Positional encoding 显式区分 reference tokens 与 target tokens,这是 in-context editing 的关键。

### 2.2 Stacked Channel Bridging (SCB) — 核心创新

SCB 要解决的问题是:prior unified models 通常用 VLM 的 final-layer 或 penultimate-layer hidden states 作为 multimodal condition(如 Qwen-Image、OmniGen2、UniWorld-V1),这有两个 limitations:
1. Final layer 偏向高层语义抽象,丢失 fine-grained visual details,而 DiT 需要这些细节来精确建模纹理、文字、空间位置
2. 依赖单层容易受 layer-specific representation bias 影响,导致 VLM 与 DiT alignment 不稳定

另一种路线(deep fusion,如 BAGEL、LightFusion)在每层 share attention,但参数量与优化复杂度大幅增加,不利于 compact model 的 efficient training。Mammoth2 (https://arxiv.org/abs/2511.18262) 用 average pooling 聚合多层,会模糊细节。

SCB 通过三步解决:

**(1) Think Token Injection**: 在 VLM input sequence 注入 128 个 learnable "think tokens"。这些 tokens 通过 self-attention 在所有层与 text/visual inputs 交互,逐步 summarize hidden representations,充当 implicit Chain-of-Thought (CoT)。这与 MetaQuery-XL (https://arxiv.org/abs/2504.06256)、BLIP3-o 的 learnable queries 思路类似,但 DeepGen 把它们定位为 "reasoning buffer",专门用于 distill VLM 中的 world knowledge 给 DiT。BAGEL 用 explicit textual CoT,而 DeepGen 用 implicit learnable tokens,推理时不需要额外生成文本,效率更高。

**(2) Layer Selection**: 选择 6 个均匀分布在 low-、mid-、high-level 的 VLM layers。这个设计参考了 Wang et al. 2025 (https://aclanthology.org/2025.acl-long.827/) 的发现 — VLM 中稀疏且均匀分布的 layers 提供有效的 visual representation。Low layers 编码 texture、color、局部空间布局;mid layers 编码 object parts 与 attribute binding;high layers 编码 scene-level semantics。6 层均匀采样保证 hierarchical feature 都被捕获。

**(3) Feature Fusion**: 给定 selected hidden states $[x_1, \dots, x_n] \in \mathbb{R}^{L \times d}$,其中:
- $n = 6$ 是 selected layer 数量
- $L$ 是 sequence length(包含 think tokens)
- $d$ 是 VLM hidden dimension

首先沿 channel 维度拼接,得到 $\mathbb{R}^{L \times d'}$,其中 $d' = n \cdot d = 6d$。然后用 2-layer MLP 投影到 DiT input width $d_{DiT}$。最后送入 Transformer encoder 深度融合,产出最终 conditional input $c \in \mathbb{R}^{L \times d_{DiT}}$:

$$c = \text{Encoder}(\text{MLP}(\text{Concat}_{ch}(x_1, \dots, x_n))) \tag{1}$$

这里的 intuition 是:channel-wise concatenation 完整保留每一层的信息(不像 average pooling 会平均掉细节),再让 Transformer encoder 通过 self-attention 学习层间如何组合。这本质上是把 "layer selection" 与 "feature mixing" 都交给 connector 学习,而非 hardcode 平均或加权。

**为什么 channel 而不是 token concat?** 如果沿 token 维度 concat,sequence length 会变成 $6L$,对 DiT 的 attention 计算是 $O((6L)^2) = 36 L^2$,六倍增长。Channel concat 保持 sequence length $L$,只增加 hidden dimension,后续 MLP 再压缩回去,computation overhead 极小。这是 SCB "lightweight" 的关键。

### 2.3 Architecture ablation 直觉

Table 6 显示:
- **w/o SCB**: DPGBench 从 87.05 跌到 85.55(-1.5),GEdit 从 7.12 跌到 6.75(-0.37),RISE 从 13.3 跌到 12.6(-0.7)。这印证单层 conditioning 确实损失 fine-grained 信息,影响 instruction following 与 editing consistency。
- **w/o Think Tokens**: WISE 从 0.72 跌到 0.68(-0.04),RISE 从 13.3 跌到 11.7(-1.6,最大跌幅)。这非常符合直觉 — reasoning-intensive 任务最依赖 world knowledge distillation,think tokens 是 knowledge 的载体。
- **w/o Activate VLM**: 各项指标均下降,说明仅靠 connector alignment 不足以让 VLM 知识充分 transfer,LoRA fine-tuning 是必要的。

## 3. Three-Stage Training Strategy

### 3.1 Stage 1: Alignment Pre-Training

**目标**: 建立 VLM 与 DiT 之间的 representation alignment。VLM 与 DiT 是两个独立预训练的 model,latent space 完全不同,直接联合训练会导致 optimization 不稳定。

**训练配置**:
- 只训练 SCB connector 与 128 think tokens,其余全部 frozen
- 200k iterations,batch size 512,lr 1e-4,20k warmup steps
- 64×H200 GPUs
- Resolution 512×512 固定
- Data: 35M general generation pairs + 6.6M editing triplets ≈ 42M samples
- Optimizer: AdamW,weight decay 0.05,grad clip 1.0

这个阶段类似于 LLaVA 的 stage-1 projection pre-training,但 SCB 的 connector 包含 6 层 Transformer encoder(比 LLaVA 的 MLP projector 复杂得多),且 think tokens 需要从随机初始化学到如何 distill VLM 知识。冻结 VLM 与 DiT 的好处是避免 catastrophic forgetting,让 connector 专心学 alignment。

### 3.2 Stage 2: Joint Supervised Fine-Tuning

**目标**: 在 alignment 基础上 end-to-end 优化 instruction following、visual fidelity、reasoning ability。

**训练配置**:
- Unfreeze DiT(全参数),VLM 用 LoRA (https://arxiv.org/abs/2106.09685) 微调
  - LoRA rank = 64,alpha = 128,dropout = 0.05
  - LoRA 比例 alpha/rank = 2 是常见配置
- 400k iterations,batch size 768,lr 5e-5
- 仍 512 resolution,但允许 arbitrary aspect ratio(dynamic resizing 保持原比例)
- Data: 11M general gen + 6.6M general edit + 150k reasoning gen + 100k reasoning edit + 560k text rendering ≈ 18.4M samples

**为什么用 LoRA 而非 full fine-tuning VLM?** Joint optimization 风险是 VLM 的 multimodal comprehension 能力退化,VLM 已经 encode 大量 world knowledge,full fine-tuning 容易 catastrophic forgetting。LoRA 限制 update 在 low-rank subspace,既能让 VLM 适应下游 generation/editing 任务,又保留预训练知识。这一点对 reasoning-based generation 至关重要 — WISE 上的 0.73 分正是依赖 VLM 中 cultural/temporal/spatial/biology/physics/chemistry 等领域知识。

### 3.3 Stage 3: MR-GRPO Reinforcement Learning

这是 paper 最有创新性的部分。MR-GRPO(Multi-Reward GRPO)基于 Pref-GRPO (https://arxiv.org/abs/2508.20751),并整合了三个 concurrent 改进:noise-preserving stochastic sampling、decoupled advantage normalization、auxiliary SFT loss。

#### 3.3.1 GRPO 在 flow matching 上的扩展

GRPO 最初是 DeepSeek 提出 (https://arxiv.org/abs/2402.03300),用于 LLM 的 RL fine-tuning。Flow matching model 是 continuous trajectory,DeepGen 把每个 denoising step 视作一个 action, $x_{t-\Delta t}$ 是 action,$\hat{v}_\theta(x_t, t)$ 是 policy。

给定 text condition $h$,flow model 采样一组 $G=8$ 张 images $\{x_0^i\}_{i=1}^G$ 及对应 denoising trajectories $\{x_T^i, x_{T-1}^i, \dots, x_0^i\}_{i=1}^G$。对于 $K=3$ 个 reward functions $\{R_k\}_{k=1}^K$,每个 reward 独立 normalize:

$$A_k^i = \frac{R_k(x_0^i, h) - \text{mean}(\{R_k(x_0^j, h)\}_{j=1}^G)}{\text{std}(\{R_k(x_0^j, h)\}_{j=1}^G)} \tag{2}$$

- $R_k(x_0^i, h)$: 第 $k$ 个 reward function 对第 $i$ 个生成 image 的评分
- $\text{mean}, \text{std}$: 在 group 内 $G=8$ 个 samples 上计算
- $A_k^i$: sample $i$ 在 reward $k$ 上的 normalized advantage,大致服从 $\mathcal{N}(0, 1)$

最终 advantage 通过 weighted aggregation 得到:

$$\hat{A}^i = \sum_{k=1}^K w_k A_k^i$$

然后再做 batch-wise normalization。Table 11 显示权重:
- Text rendering prompts: $w_{pref}=0.2, w_{CLIP}=0.1, w_{OCR}=0.7$
- General T2I prompts: $w_{pref}=0.7, w_{CLIP}=0.3, w_{OCR}=0$

**为什么 decoupled normalization 关键?** 三个 reward 的 scale 与 variance 完全不同 — pairwise preference win rate 在 $[0,1]$ 之间,OCR accuracy 在 $[0,1]$,CLIP score 大约在 $[0.2, 0.4]$。如果直接用 raw reward 求和,高 variance reward(如 CLIP)会 dominate 梯度方向。Decoupled normalization(来自 GDPO, https://arxiv.org/abs/2601.05242)让每个 reward 在 policy update 中贡献相对均衡。Table 7 显示 w/o Reward-wise Norm 时 text generation score 从 35.06 跌到 32.18(-2.88),证明这点。

#### 3.3.2 GRPO 主目标

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{h \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left( \min(r_t^i(\theta) \hat{A}^i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}^i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right] \tag{3}$$

变量含义:
- $h \sim \mathcal{D}$: 从训练分布 $\mathcal{D}$ 采样 text condition
- $G = 8$: group size
- $T = 50$: total denoising steps
- $r_t^i(\theta) = p_\theta(x_{t-\Delta t}^i | x_t^i, h) / p_{\theta_{\text{old}}}(x_{t-\Delta t}^i | x_t^i, h)$: per-step importance ratio,新旧 policy 在该 step 采样概率比
- $\hat{A}^i$: aggregated advantage
- $\epsilon = 1 \times 10^{-4}$: PPO-style clip range(注意这里非常小,因为 flow matching 的 log-prob 是连续的,clip 范围太大会导致不稳定)
- $\beta = 5 \times 10^{-7}$: KL coefficient,非常小
- $\pi_\theta, \pi_{\text{ref}}$: 当前 policy 与 reference policy(SFT 后的 model frozen)

KL divergence 在 velocity space 计算:

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \|\hat{v}_\theta(x_t, t) - \hat{v}_{\text{ref}}(x_t, t)\|^2 \tag{4}$$

- $\hat{v}_\theta(x_t, t)$: 当前 model 预测的 velocity
- $\hat{v}_{\text{ref}}(x_t, t)$: reference model 预测的 velocity
- 这是 Euclidean distance squared,而非严格的 KL,但在 Gaussian 假设下等价于 KL(去掉常数项)

#### 3.3.3 Noise-preserving stochastic sampling

flow matching 的 deterministic ODE $dx_t = \hat{v}_\theta(x_t, t) dt$ 无法为 RL 提供 exploration。Prior works(Flow-GRPO, https://arxiv.org/abs/2505.05470;DanceGRPO, https://arxiv.org/abs/2505.07818)转换为 SDE,但标准 Flow-SDE 会注入超过 scheduler 预期 noise level 的随机性,导致 sample quality 下降、reward signal 不准。

DeepGen 采用 noise-preserving 策略 (https://arxiv.org/abs/2509.05952):

$$x_{t-\Delta t} = (1-(t-\Delta t)) \hat{x}_0 + (t-\Delta t) \cos\left(\frac{\eta \pi}{2}\right) \hat{x}_1 + (t-\Delta t) \sin\left(\frac{\eta \pi}{2}\right) \epsilon \tag{6}$$

变量含义:
- $t \in [0, 1]$: timestep,0 表示 clean data,1 表示 pure noise
- $\Delta t$: step size($= 1/T = 1/50$)
- $\hat{x}_0 = x_t - t \hat{v}_\theta$: predicted clean sample
- $\hat{x}_1 = x_t + (1-t) \hat{v}_\theta$: predicted noise
- $\epsilon \sim \mathcal{N}(0, I)$: fresh Gaussian noise
- $\eta = 1.0$: stochasticity strength($\eta=0$ 时 deterministic,$\eta=1$ 时 fully stochastic)
- $\cos(\eta \pi / 2), \sin(\eta \pi / 2)$: 振幅分解,确保 noise level 与 flow scheduler 在 timestep $t-\Delta t$ 处的预期值一致

intuition: 标准 flow matching 的 forward process 是 $x_t = (1-t) x_0 + t x_1$,所以从 $x_t$ 预测 $x_0$ 与 $x_1$ 后,用 cosine-sine 分配 deterministic 与 stochastic 分量,保证总 noise level 严格等于 $t-\Delta t$。Log-probability 简化为:

$$\log p_\theta(x_{t-\Delta t} | x_t) = -\|x_{t-\Delta t} - \mu_\theta(x_t, t)\|^2 \tag{7}$$

- $\mu_\theta(x_t, t) = (1-(t-\Delta t)) \hat{x}_0 + (t-\Delta t) \cos(\eta \pi / 2) \hat{x}_1$: 采样的 deterministic 部分

这个简化去掉了标准 log-prob 中的 variance normalization term,避免小 noise level 时的数值不稳定。

#### 3.3.4 Auxiliary SFT Loss — 关键创新

paper 最有意思的发现:KL regularization 单独不足以防止 capability degradation,RL training 超过 ~1000 steps 后,model 在 complex instruction comprehension(如 reasoning generation)上 performance 显著下降。

**为什么 KL 不够?** KL 在 velocity space 约束 trajectory(每一步 velocity 不能偏离 reference 太远),这是 **process-level guidance**。但 KL 允许最终 outcome 偏离 SFT distribution,只要每步偏离不大。长时间训练中,微小 drift 累积会导致 final image distribution 偏离 SFT 学到的高质量区域。

**SFT loss 是 outcome-level guidance**:直接在高质量 SFT dataset 上计算 flow matching loss,把 model 锚定在 SFT distribution:

$$\mathcal{L}_{\text{total}} = (1-\lambda) \mathcal{L}_{\text{GRPO}} + \lambda \mathcal{L}_{\text{SFT}} \tag{5}$$

- $\lambda = 1 \times 10^{-4}$: 非常小的 mixing coefficient,确保 SFT loss 仅作为 anchor,不主导 optimization
- $\mathcal{L}_{\text{SFT}}$: 标准 flow matching loss $\mathbb{E}[\|\hat{v}_\theta - v_{\text{target}}\|^2]$

这个设计哲学上类似 RLHF 中的 reference model constraint,但更直接 — 不只是惩罚偏离,而是主动拉回到 SFT distribution。Table 7 显示 w/o Auxiliary SFT Loss 时 UniGenBench 从 75.69 跌到 74.33(-1.36),Fig.6(a) 更直观地展示:从 ~300 steps 开始 performance 持续下降,最终低于初始 checkpoint。这是一个非常重要的 finding — 对未来 diffusion RL 工作有指引意义。

#### 3.3.5 Reward Functions

三个 reward 协同工作:
1. **VLM-based pairwise preference reward** (来自 Unified-Reward-Think, https://arxiv.org/abs/2505.03318): 比较组内所有生成 images,计算 per-sample win rate 作为 reward
2. **OCR reward** (PaddleOCR 3.0, https://arxiv.org/abs/2507.05595): 检测生成图像中的文字并与 prompt 中 target text 对比
3. **CLIP similarity score** (https://arxiv.org/abs/2103.00020): 生成图像与 text condition 的整体语义一致性

CLIP score 是 reward 而非 metric,这与 DPO-style 方法不同,直接在 prompt 级别提供 dense supervision。

## 4. Data Strategy

Fig.4 展示整体 data 组合。Total ~50M samples,远小于 HunyuanImage 的 5B、LongCat 的 1.2B。但 data 质量与多样性精心设计:

**General Generation (35M pretrain + 11M SFT)**:
- 公开 dataset: text-to-image-2M (https://huggingface.co/datasets/jackyhate/text-to-image-2M), LAION-Aesthetic-6M (https://arxiv.org/abs/2210.08402), Megalith-10M, RedCaps-5M (https://arxiv.org/abs/2111.11431), CC-12M (https://arxiv.org/abs/2102.08981)
- 高质量 SFT data: BLIP-3o (60k)、ShareGPT-4o-Image (45k, https://arxiv.org/abs/2506.18095)、Echo-4o-Image (100k, https://arxiv.org/abs/2508.09987)、OpenGPT4o-Image (40k, https://arxiv.org/abs/2509.24900)
- 10M in-house real samples,long:short prompt 比例 3:1
- 50k Nano Banana 合成 photorealistic images with fine-grained prompts(中英文),用 closed-source model 蒸馏数据增强 detail generation

**General Editing (6.6M × 2)**:
- NHR-Edit (720k, https://arxiv.org/abs/2507.14119)
- GPT-Image-Edit (1.5M, https://arxiv.org/abs/2507.21033)
- ShareGPT-4o-Image-Edit (50k)
- OpenGPT4o-Image-Edit (40k)
- Nano-banana-consist (150k)
- Pico-Banana (250k, https://arxiv.org/abs/2510.19808)
- X2I2 (1.6M)
- UniWorld-Edit set (1.2M, https://arxiv.org/abs/2506.03147)
- 1.1M in-house editing(中英文)

**Reasoning Generation/Editing (150k + 100k)**: 来自 UniReason (https://arxiv.org/abs/2602.02437),覆盖 cultural commonsense、natural science、spatial、temporal、logical reasoning 五大领域。这部分 data 量小但至关重要 — WISE 与 RISE 上的 leading performance 直接来源于此。

**Text Rendering (560k)**:
- Captions 来自 document/infographic-centric multimodal QA datasets(如 LLaVA-OneVision-1.5,https://arxiv.org/abs/2509.23661)
- Gemini 2.5 Pro (https://deepmind.google/models/gemini/pro/) 随机生成 font style、layout、color scheme 等 rendering attributes
- 与开源 text rendering prompt set (Flux-Reason-6M, https://arxiv.org/abs/2509.09680) 组合
- 用 Qwen-Image 生成 500k text-rendering images
- 加 60k application-oriented data(中文诗歌、海报设计)

## 5. Experimental Results 解读

### 5.1 General Generation & Editing (Table 1)

DeepGen 1.0 (RL) 在 5B 参数下:
- GenEval: 0.87(与 Qwen-Image 27B 持平,无 external LLM prompt rewriting)
- DPGBench: 87.90(第二名,仅次于 Qwen-Image 88.32)
- UniGenBench: 75.74(第二名)
- ImgEdit: 4.14,GEdit-EN: 7.17

关键观察:RL 阶段 universal 提升 SFT 版本(GenEval 0.868→0.87,DPGBench 87.05→87.90,UniGenBench 74.18→75.74),证明 MR-GRPO 对 general capability 也有正向作用,而非仅优化特定 reward。

### 5.2 Reasoning Generation (Table 2, 3)

**WISE benchmark** (https://arxiv.org/abs/2503.07265):
- DeepGen 1.0 (RL): 0.73 overall
- HunyuanImage 3.0 (80B): 0.57 → DeepGen 高 28%
- BAGEL (14B, with explicit CoT): 0.70
- Closed-source: GPT-Image-1 (0.80), Seedream 4.0 (0.78)

5B model 超越 80B 是 impressive 的。Domain breakdown 显示 DeepGen 在 Time reasoning 上从 SFT 0.71 提升到 RL 0.81(+0.10),Physics 从 0.79 到 0.82,体现 RL 帮助 consolidate world knowledge utilization。但 Cultural 从 0.70 跌到 0.72 几乎持平,RL 在某些 domain 收益有限。

**T2I-CoREBench** (https://arxiv.org/abs/2509.03516): 八个推理类别(Logical, Behavioral, Hypothetical, Procedural, Generalization, Analogical, Commonsense, Reconstructive)
- DeepGen 1.0 (RL): 46.5
- LongCat-Image: 52.2(top open-source)
- Qwen-Image: 46.3,HunyuanImage 3.0: 46.0
- Closed-source: Nano Banana 70.5,Seedream 4.0 69.4

注意 R-RR(Reconstructive Reasoning)普遍很低(DeepGen 19.6,GPT-Image-1 47.5),说明 reconstructive reasoning 是 unified model 的普遍短板。

### 5.3 Reasoning Editing (Table 4)

**RISE** (https://arxiv.org/abs/2504.02826):
- DeepGen 1.0 (SFT): 13.3 overall(top open-source)
- DeepGen 1.0 (RL): 10.8(下降!RL 反而损害 reasoning editing)
- BAGEL: 11.9,Qwen-Image-Edit: 8.9

**UniREditBench** (https://arxiv.org/abs/2511.01295):
- DeepGen 1.0 (SFT): 77.5(超 Qwen-Image-Edit 56.5 共 37%)
- DeepGen 1.0 (RL): 75.7
- Closed-source GPT-Image-1: 73.4(DeepGen SFT 版本甚至超越 closed-source!)

值得深思:**RL 阶段在 reasoning editing 上反而轻微下降**(RISE 13.3→10.8,UniREditBench 77.5→75.7)。这暗示当前 reward 设计(preference + OCR + CLIP)没有充分 cover reasoning editing 所需的 knowledge-grounded editing signal。未来工作可能需要专门的 reasoning-aware reward function。

### 5.4 Text Rendering (Table 5)

**CVTG-2K** (https://arxiv.org/abs/2503.23461):
- DeepGen 1.0 (SFT): Word Acc 0.6605, NED 0.8426, CLIPScore 0.8227
- DeepGen 1.0 (RL): Word Acc 0.7533(+0.0928), NED 0.8936, CLIPScore 0.8278

RL 让 Word Accuracy 提升 14%,这是 OCR reward 的直接结果。CLIPScore 保持最高(0.8278),证明 text fidelity 提升没有牺牲 semantic alignment。GLM-Image (https://z.ai/blog/glm-image) Word Acc 0.9116 更高,但 CLIPScore 仅 0.7877,显示 text 与 image context 整体一致性较低。

### 5.5 RL Ablation (Table 7, Fig. 6)

关键数据:
- w/o Auxiliary SFT Loss: UniGenBench -1.36,Text -1.73 — 验证 SFT loss 的 anchor 作用
- w/o Velocity KL: UniGenBench -0.62,Text -2.59 — KL 对 text rendering 影响最大
- w/o Reward-wise Norm: Text -2.88 — 多 reward 优化中 normalization 至关重要

Fig.6(a) 最 striking:从 ~300 steps 开始,w/o SFT Loss 版本 performance 持续下降,最终低于初始 checkpoint,即 RL 不仅没改善反而损害 model。这是 paper 最强的 finding 之一。

## 6. 个人 Intuition 与思考

### 6.1 为什么 5B 能击败 80B?

我的核心 hypothesis:**unified multimodal model 的 bottleneck 不是 capacity,而是 representation alignment**。VLM 已 encode 海量 world knowledge,问题在于如何把这部分 knowledge 高效 transfer 给 DiT。HunyuanImage 80B 通过 brute-force scaling 增加 capacity,但 alignment 仍是 final-layer conditioning,信息 transfer 效率低。DeepGen 5B 通过 SCB 在 multiple levels 进行 alignment,information transfer 更 dense,反而用更少参数达到更高 performance。

类似现象在 LLM 中也出现 — MiniCPM、Phi 系列证明 small model 配合高质量 data 与 careful training 能逼近大 model。Unified multimodal model 的 scaling law 可能与 LLM 不同,因为 VLM 与 DiT 之间的 interface 是 bottleneck,而非单一 model 的 capacity。

### 6.2 SCB vs Deep Fusion 的 trade-off

BAGEL、HunyuanImage 选择 deep fusion(每层 share attention),信息 flow 更充分但参数与计算成本高。SCB 选择 "多层 sampling + 单点融合",trade-off 偏向 efficiency。从 Table 6 看 w/o SCB 性能下降 1-2%,deep fusion 理论上可以进一步提升,但成本不成比例增长。对于 compact model,SCB 是 sweet spot。

### 6.3 Think Tokens 的 implicit CoT 与 explicit CoT

BAGEL 用 explicit textual CoT(reasoning with text before generation),DeepGen 用 implicit learnable tokens。Trade-off:
- Explicit CoT: 推理 transparent、可解释,但生成 text 增加 latency,且 VLM 必须支持 long context reasoning
- Implicit CoT: 推理 efficient,但不可解释,且 think tokens 学到什么完全 black-box

paper 没有直接对比两者,但 DeepGen 在 WISE 上 0.73 vs BAGEL 0.70,说明 implicit CoT 在 5B scale 至少不逊于 explicit CoT。这是有趣的方向 — 是否 implicit CoT 在 small model 上更 efficient?可能因为 explicit CoT 需要更多 capacity 来 generate coherent reasoning text。

### 6.4 RL for Diffusion 的未来

DeepGen 的 auxiliary SFT loss finding 令我想到 AlphaGo 的 policy network anchor、RLHF 中的 KL penalty。本质上 RL optimization 在 high-capacity model 上容易 collapse,需要某种 "pull-back" mechanism。SFT loss 是 outcome-level pull-back,KL 是 process-level pull-back,两者互补。未来 diffusion RL 工作应该都采用类似 dual constraint 设计。

更进一步,RL 在 reasoning editing 上的退化表明,reward function 设计仍是 open problem。VLM-based reward 可能本身在 reasoning 场景下不可靠。可能需要 task-specific verifier(如 knowledge-grounded editing verifier),或 self-rewarding 机制。

### 6.5 Data Efficiency 的启示

50M samples vs 5B samples(100× reduction)是惊人的。这说明:
1. 大部分大规模 dataset(LAION、CC12M)的 marginal value 在 unified model 训练中递减
2. High-quality instruction data(ShareGPT-4o、BLIP3o)的 marginal value 远高于 raw pairs
3. Reasoning data 即便只有 150k,对 reasoning capability 贡献巨大

这与 LLM 中的 Phi、Qwen 系列发现一致 — data quality > data quantity。未来 unified multimodal model 训练可能更多关注 data curation 而非 scaling。

### 6.6 局限与未来方向

paper 没充分讨论的几点:
- **Resolution 限制**: 512×512 fixed,无法 native 高分辨率生成(Qwen-Image 支持 higher resolution)
- **RL 在 reasoning editing 上的退化**: 需要 reward function 设计改进
- **Lack of video generation**: 当前仅 image generation/editing
- **Think tokens 可解释性**: 没分析 think tokens 学到什么具体 reasoning pattern
- **Long-context editing**: 多 reference image 或 multi-turn editing 没充分探索

未来可能的方向:
- 把 SCB 扩展到 video generation(VLM + Video DiT)
- 用 explicit + implicit CoT 混合(think tokens + textual reasoning interleaving)
- 引入 process reward model(PRM)替代 outcome reward
- Self-play RL 通过 VLM-as-judge 持续迭代
- 扩展到 4B+ token context,支持 long-document grounded editing

## 7. 总结

DeepGen 1.0 的核心贡献是证明 **lightweight unified multimodal model 通过 architecture synergy + data-centric training 可以达到甚至超越 massive model**。SCB 解决 VLM-DiT alignment bottleneck,MR-GRPO with auxiliary SFT loss 解决 RL stability,50M data strategy 解决 data efficiency。这些 finding 对整个 unified multimodal 领域有普遍指导意义 — 未来工作不必盲目 scale,而应关注 interface design 与 training stability。

paper 的最大启示可能不是 "5B 击败 80B" 这个 headline,而是揭示了 unified multimodal model 的 scaling behavior 与 LLM 不同,interface(alignment between VLM and DiT)是 bottleneck,而非单一 model capacity。这一观察如果被后续工作验证,可能重塑整个领域的 architecture 设计哲学。

## References

- DeepGen 1.0 GitHub: https://github.com/DeepGenTeam/DeepGen
- DeepGen 1.0 HuggingFace: https://huggingface.co/DeepGenTeam/DeepGen-1.0
- DeepGen 1.0 Datasets: https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0
- BAGEL: https://arxiv.org/abs/2505.14683
- HunyuanImage 3.0: https://arxiv.org/abs/2509.23951
- Qwen-Image: https://arxiv.org/abs/2508.02324
- LongCat-Image: https://arxiv.org/abs/2512.07584
- BLIP3-o: https://arxiv.org/abs/2505.09568
- Qwen-2.5-VL: https://arxiv.org/abs/2502.13923
- Skywork UniPic 2.0: https://arxiv.org/abs/2509.04548
- SigLIP: https://arxiv.org/abs/2303.15343
- OpenUni: https://arxiv.org/abs/2505.23661
- MetaQuery-XL: https://arxiv.org/abs/2504.06256
- UniWorld-V1: https://arxiv.org/abs/2506.03147
- Mammoth2: https://arxiv.org/abs/2511.18262
- GRPO: https://arxiv.org/abs/2402.03300
- Pref-GRPO: https://arxiv.org/abs/2508.20751
- GDPO: https://arxiv.org/abs/2601.05242
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- DanceGRPO: https://arxiv.org/abs/2505.07818
- LoRA: https://arxiv.org/abs/2106.09685
- WISE: https://arxiv.org/abs/2503.07265
- UniREditBench: https://arxiv.org/abs/2511.01295
- UniReason: https://arxiv.org/abs/2602.02437
- T2I-CoREBench: https://arxiv.org/abs/2509.03516
- RISE: https://arxiv.org/abs/2504.02826
- CVTG-2K (TextCrafter): https://arxiv.org/abs/2503.23461
- DPG-Bench (ELLA): https://arxiv.org/abs/2403.05135
- GenEval: https://arxiv.org/abs/2310.11513
- Unified-Reward-Think: https://arxiv.org/abs/2505.03318
- PaddleOCR 3.0: https://arxiv.org/abs/2507.05595
- CLIP: https://arxiv.org/abs/2103.00020
- ShareGPT-4o-Image: https://arxiv.org/abs/2506.18095
- Echo-4o-Image: https://arxiv.org/abs/2508.09987
- OpenGPT4o-Image: https://arxiv.org/abs/2509.24900
- NHR-Edit: https://arxiv.org/abs/2507.14119
- GPT-Image-Edit: https://arxiv.org/abs/2507.21033
- Pico-Banana: https://arxiv.org/abs/2510.19808
- RedCaps: https://arxiv.org/abs/2111.11431
- CC-12M: https://arxiv.org/abs/2102.08981
- Flux-Reason-6M: https://arxiv.org/abs/2509.09680
- LAION-5B: https://arxiv.org/abs/2210.08402
- LLaVA-OneVision-1.5: https://arxiv.org/abs/2509.23661
- Gemini 2.5 Pro: https://deepmind.google/models/gemini/pro/
- GLM-Image: https://z.ai/blog/glm-image
- Coefficients-preserving sampling: https://arxiv.org/abs/2509.05952
- Wang et al. 2025 ACL: https://aclanthology.org/2025.acl-long.827/
