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
HunyuanImage 搞到 80B 用了 5B 样本,LongCat 用 1.2B 样本,Qwen-Image 加上 Edit 版本合计 54B。Lumina-DiMOO 只有 8B,居然超过 14B 的 BAGEL。在 unified multimodal 这个 paradigm 下,scaling 的规律不一定是主导因素。于是用 5B(3B VLM + 2B DiT)、~50M 样本,做到能和 80B 掰手腕。

你要让一个 model 同时做 image generation 和 editing,还要带 reasoning,本质上你是在组合两个东西: VLM & DiT. 这俩是各自预训练出来的,latent space 完全不一样。怎么让 VLM 理解到的东西,高效地传给 DiT 让它画出来?这就是 bottleneck。之前三种路线: 1. 只拿 VLM 最后一个 layer 的输出 (Qwen-Image、OmniGen2、UniWorld-V1); 2. Deep fusion (BAGEL、HunyuanImage) 每层 share attention; 3. Average pooling 多层(Mammoth2) — 平均了就把细节平均掉了; 4. DeepGen 用第四条路:Stacked Channel Bridging (SCB)。

别只看 VLM 最后一层,从底层、中层、高层各抽几个 layer,把它们的信息全部保留下来,再压缩给 DiT。具体三步:
1. 注入 Think Tokens: 在 VLM 的输入序列里塞 128 个 learnable tokens,这些 tokens 和 text/visual tokens 一起过所有 self-attention layer。它们干什么?当 "reasoning buffer" — VLM 知识被慢慢 distill 到这些 tokens 里。BAGEL 是让 model 先 explicit 生成一段 reasoning text 再画图,DeepGen 不生成文本,直接用 learnable tokens 做 implicit CoT,推理时更高效。
2. 选 6 个 layer 均匀采样: 从 VLM 的 low/mid/high level 各均匀采 6 个 layer 的 hidden states。低层抓纹理颜色,中层抓 object parts 和 attribute binding,高层抓 scene semantics。VLM 里 visual information 是分布式 encoded 在多个 layer 的,不是全在最后一层。
3. Channel-wise concat 再融合: 这一步是 SCB 名字的由来。给定 6 个 layer 的 hidden states $[x_1, \dots, x_6] \in \mathbb{R}^{L \times d}$,其中 $L$ 是 sequence length(含 think tokens),$d$ 是 VLM hidden dim。沿 channel 维度 concat(不是 token 维度!),得到 $\mathbb{R}^{L \times 6d}$,再用 2-layer MLP 投影到 DiT 的 width $d_{DiT}$,最后过 6 层 Transformer encoder 融合: $$c = \text{Encoder}(\text{MLP}(\text{Concat}_{ch}(x_1, \dots, x_6))) \tag{1}$$输出 $c \in \mathbb{R}^{L \times d_{DiT}}$,作为 DiT 的 multimodal condition。为什么 channel 而不是 token concat?你想,如果沿 token concat,sequence length 变成 $6L$,DiT 的 self-attention 是 $O(L^2)$,直接 36 倍计算量。Channel concat 保持 $L$ 不变,只是 hidden dim 暂时变 6 倍,后续 MLP 立即压回 $d_{DiT}$,overhead 极小。这就是 "lightweight" 的关键。

相当于让 connector 自己学 "我应该从哪层抽多少信息",而不是 hardcode "只用最后一层" 或 "全部平均"。VLM 里 information 是 distributed 的,SCB 给 connector 一个 learnable 的方式去 aggregate。

unified multimodal model 的 scaling behavior 与 LLM 不同,interface 是 bottleneck,不是 capacity. SCB 解决 VLM-DiT alignment bottleneck,MR-GRPO with auxiliary SFT loss 解决 RL stability,50M data strategy 解决 data efficiency

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

