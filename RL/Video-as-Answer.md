---
source_pdf: Video-as-Answer.pdf
paper_sha256: 2328c283a20c8a8e5914843cca13237ed7bd500249d06f742408bdfcef3dae18
processed_at: '2026-08-13T00:38:38-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VANS: 用人话说一遍

Andrej，前面那版太 formal 了，我换个口吻重新讲一遍，目标是让你 gut-level 理解这篇 paper 在干嘛、为什么这么干。

---

## 这篇 paper 一句话讲完是什么

现在的 video generation model 都在搞娱乐——生成个搞笑视频、做个特效。Kuaishou Kling Team 觉得这太浪费了，video 其实可以当 **answer** 用。你问"打领带下一步怎么弄"，与其用文字描述"把宽端从内侧穿到外侧形成环"，不如直接生成一段 video 演给你看。

所以他们提了个新任务叫 **VNEP (Video-Next-Event Prediction)**：给一段 input video + 一个 question，模型要预测下一事件，并且用 **video** 回答你，而不是用 text。

这听起来简单，做起来一堆麻烦事。

---

## 为什么这事儿难

你要让模型生成"下一个事件"的 video，直觉上就是 VLM 先 reasoning 出下一事件是什么（输出 text caption），然后 VDM 拿这个 caption 去 generate video。Cascaded pipeline 嘛。

但问题是——**两个 model 各干各的，不知道对方的脾气**。

VLM 训练目标是"写出语言正确的 caption"，它根本不知道自己写的 caption 到了 VDM 手里能不能 visualize。它可能写"The man flies to the moon"，语言上完美，VDM 直接懵了生成一团糊。

反过来，VDM 训练目标是"生成漂亮的 video"，它拿到的 caption 可能含一些它根本不认识的 concept，或者 caption 描述的动作它执行不了，于是它就自由发挥，生成的 video 跟 caption 语义对不上。

这就是 **semantic-to-visual gap**。两个 model 各自很强，但拼一起就拉胯。

那能不能把 understanding 和 generation 塞进一个 unified model？Omni-Video 之类的试过，结果两个能力互相打架，trade-off 严重，一个强了另一个就弱 ([arXiv:2507.06119](https://arxiv.org/abs/2507.06119))。

所以 VANS 的核心 idea 是：**保持两个 specialized model 的分离，但用 RL 把它们紧紧绑在一起，让它们学会配合**。

---

## VANS 架构长什么样

很简单，两个 model 串联，但 conditioning 设计有讲究：

**VLM 这条路** (Qwen2.5-VL-3B, https://arxiv.org/abs/2309.16609):
- Input video 走 ViT 提取 high-level semantic features
- 这些 features + question 一起喂给 VLM
- VLM 输出 caption 描述下一事件，格式是 `[Think]...[/Think][Ans]...[/Ans]`，先 reasoning 再 answer

**VDM 这条路** (Wan-2.1-1.3B, https://arxiv.org/abs/2503.20314):
- Input video 取 6 帧过 VAE 得到 low-level visual tokens
- 这些 tokens + VLM 的 caption 一起作为 VDM 的 conditioning
- VDM 生成 output video

**为什么 ViT 给 VLM、VAE 给 VDM？** 因为 VLM 要理解 high-level semantics（什么动作、什么物体），VDM 要保持 low-level visual consistency（颜色、纹理、appearance）。两条路各取所需，不互相干扰。这个设计本身就比单纯 cascaded 强很多——你后面看 ablation 会发现光这个 architectural choice 就让 FVD 从 140 降到 85。

但 SFT 到这儿就到头了。两个 model 还是各自为政，RL 才是真正的 magic。

---

## Joint-GRPO：这篇 paper 的灵魂

### 先回忆 GRPO 是什么

GRPO 是 DeepSeekMath 提出来的 RL 算法 ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))，简单说就是：

对每个 prompt，让 model 生成 $G$ 个 candidate（一组），给每个 candidate 打分 $r_i$，然后算 group 内的 normalized advantage：

$$\tilde{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$$

- $r_i$ = 第 i 个 candidate 的 reward
- $\bar{r}$ = 这组的平均 reward
- $\sigma_r$ = 这组 reward 的标准差

然后像 PPO 一样用 clipped objective 更新 policy，加个 KL penalty 防止跑偏太远。

**Intuition**: 不用 critic network 估 baseline，直接用 group 平均当 baseline。比平均好的就 boost，比平均差的就 penalize。简单粗暴有效。

### 为什么 standard GRPO 在这儿不够

如果分别对 VLM 和 VDM 跑 GRPO——VLM 只优化自己的 caption reward，VDM 只优化自己的 video reward——两个 model 不会学会配合。VLM 还是不知道自己 caption 能不能被 VDM 执行。

那能不能同时优化两个 model？问题来了：生成的 video 质量差，你不知道怪谁——是 VLM 的 caption 写得烂，还是 VDM 的 generation 能力差？**Attribution problem**。Gradient signal 冲突，训练不稳定，容易 reward hacking。

### Joint-GRPO 的两阶段解法

核心 insight：**先把 attribution 搞清楚，再让两个 model 互相 adapt**。

#### Stage 1: 调 VLM，冻结 VDM

这个阶段 VDM 是 frozen 的。VLM 采样 $G$ 个 caption，每个 caption 喂给 frozen VDM 生成 video，然后打 joint reward：

$$r_1 = \lambda_f r_f + \lambda_{t1} r_{t1} + \lambda_{v1} r_{v1}$$

三个 component：
- $r_f$: format reward，caption 是否遵循 `[Think][Ans]` 模板，binary 0/1
- $r_{t1}$: text fidelity，生成的 caption 和 ground-truth caption 的 ROUGE-L 相似度
- $r_{v1}$: video fidelity，生成的 video 和 ground-truth video 的 CLIP similarity

**为什么三个都要？** 这是关键 intuition：

只用 $r_{t1}$：VLM 学会写语言正确的 caption，但可能写出 VDM 画不出来的东西（"飞去月球"语言通顺但 VDM 画不出）。

只用 $r_{v1}$：reward 信号太远太模糊。Video 质量差，VLM 不知道是 reasoning 错了、还是描述太抽象、还是用了 VDM 不认识的 concept。它收到的 gradient signal 没法指导它怎么改 caption。

三个合一起：$r_{t1}$ 保证语义方向对，$r_{v1}$ 提供"你的 caption 能不能被 visualize"的直接 feedback。VLM 被迫学会写既正确又 **visualization-friendly** 的 caption——它 internalize 了 VDM 的能力边界。

这个阶段训练完，VLM 不再是孤立的 reasoning machine，它变成了"懂 VDM 脾气的 reasoning machine"。

#### Stage 2: 调 VDM，冻结 VLM

Stage 1 的 VLM 现在当 frozen anchor。它生成一个高质量 anchor caption $s_{anchor}$（ROUGE-L < 0.6 的丢掉重采，保证 quality）。然后 VDM 采样 $G$ 个 video，打 joint reward：

$$r_2 = \lambda_{v2} r_{v2} + \lambda_{c2} r_{c2}$$

- $r_{v2}$: video fidelity，跟 ground-truth video 的 CLIP similarity，保持视觉质量和与 input 的 coherence
- $r_{c2}$: semantic alignment，生成的 video 跟 anchor caption 的 CLIPScore

**为什么两个都要？**

只用 $r_{v2}$：VDM 会 reward hack——直接复制 input video 帧就能拿高分，完全 ignore caption 语义。

只用 $r_{c2}$：VDM 会生成 static frame——单帧匹配 caption 的 CLIP embedding 就能拿高分，没有 motion。这也是 reward hacking。

两个合一起：VDM 必须 **preserve input video 的 visual elements**（颜色、appearance、背景），同时 **faithfully render** anchor caption 描述的事件。它被逼着做 cross-modal alignment。

---

## 为什么两阶段顺序不能反

**Intuition**: Stage 1 先让 VLM 适应 VDM，这样 VLM 输出的 caption 是 VDM "能听懂"的。Stage 2 再让 VDM 适应这个已经 visualization-friendly 的 caption，VDM 的任务就变成"忠实执行"，attribution 清晰。

如果反过来——先调 VDM 再调 VLM——VDM 在适应一个还没优化的烂 caption，调完 VLM 变了，VDM 又得重新适应，互相 chasing。

如果 all-in-one 同时调——回到 attribution problem，poor video 不知道怪谁，gradient 冲突。Ablation 数据证实了这点：all-in-one 版本 ROUGE-L 0.3577、FVD 81.01，都比 staged 版本差。

---

## 数据集 VANS-Data-100K

他们从零造了个数据集，因为现有的 NEP 数据集 video 质量差、question 不够 diverse。

**组成**：
- Procedural 30K：COIN (21K, https://arxiv.org/abs/1904.01258) + YouCook2 (9K, https://arxiv.org/abs/1812.02519)，都是 step-by-step 教学视频
- Predictive 70K：Video-Holmes (10K) + ActivityNet (20K) + V1-33K (10K) + YouTube (30K)，有 narrative 和 causal dynamics

**Pipeline**:
1. 收集 raw video
2. Shot split：procedural 用 ground-truth timestamp，predictive 用 shot boundary detection。过滤 < 3s 的短 segment
3. Clip selection：Gemini-2.5-Flash (https://arxiv.org/abs/2403.05530) 选 3-5s 最优 clip
4. QA generation：Gemini 模拟 diverse question + chain-of-thought reasoning + ground-truth answer，加 self-check 防 information leakage

最终 100K 用于 SFT，手动挑 1K 高质量样本做 RL post-training。这种"大 SFT + 小 RL"是现在标配。

Input video 平均 9.43s，target video 平均 3.76s——target 比 input 短，因为只展示"下一个事件"。

---

## 实验结果讲讲

### 主表的关键数字

Procedural benchmark 上：

| Model | ROUGE-L | FVD↓ | CLIP-V | CLIP-T |
|-------|---------|------|--------|--------|
| Omni-Video (unified) | 0.1075 | 105.32 | 0.6293 | 0.2323 |
| Gemini-FilmWeaver (cascaded strong baseline) | 0.2802 | 110.54 | 0.7102 | 0.2773 |
| VANS (SFT only) | 0.2812 | 85.34 | 0.7655 | 0.3202 |
| VANS (Joint-GRPO) | **0.3631** | **78.32** | **0.8021** | **0.3824** |

**怎么读**：

1. Unified model (Omni-Video) 全面拉胯——capability trade-off 验证了，一个 model 搞不定 understanding + generation。
2. VANS (SFT) vs Gemini-FilmWeaver：ROUGE-L 差不多（0.2812 vs 0.2802），但 FVD 从 110 降到 85——光靠架构设计（VAE reference tokens）就把 visual coherence 提升一大截。
3. Joint-GRPO 在 SFT 基础上把 ROUGE-L 从 0.2812 拉到 0.3631（+29.1%），CLIP-T 从 0.3202 到 0.3824（+19.4%）。**这是 RL 的决定性贡献**。

Video-GPT CLIP-T 只有 0.1997 最低——它做的是 spatiotemporal continuation 没有 event reasoning，侧面证明 VNEP 不是 simple extension。

### Ablation 怎么读

**Joint vs Isolated**：
- GRPO (VLM only): ROUGE-L 0.3190, CLIP-V 0.7798
- GRPO (VDM only): ROUGE-L 0.2812 (无变化), CLIP-V 0.7671
- GRPO (VLM+VDM cascaded): ROUGE-L 0.2894（比 VLM only 退步！）
- Joint-GRPO: ROUGE-L 0.3631, CLIP-V 0.8021

**Intuition**: VDM only 的 GRPO 对 text metrics 零贡献（VDM 不影响 caption 生成）。Cascaded 比 VLM only 退步——独立优化再拼接会产生 misalignment，两个 model 各自被优化到自己的 reward 最优点，但拼一起不在 joint optimum。

**Staged vs All-in-one**:
- All-in-one: ROUGE-L 0.3577, FVD 81.01, CLIP-V 0.7800
- Staged: ROUGE-L 0.3631, FVD 78.32, CLIP-V 0.8021

All-in-one 退化所有 visual metrics，证实 attribution problem 的存在。

**Reward component**:
- 去掉 $r_{t1}$：ROUGE-L 从 0.3631 降到 0.3498，caption 开始出现"没预测到 removing the mask"这种错误
- 去掉 $r_{v1}$：CLIP-V 从 0.7803 降到 0.7668，visual consistency 退化
- 去掉 $r_{c2}$：CLIP-V 降到 0.7921，且出现 static frames reward hacking
- 去掉 $r_{v2}$：CLIP-V 降到 0.7887，output coherence 退化

每个 reward 都不可缺。$r_{c2}$ 尤其关键——没有它 VDM 直接生成 static frame 骗 CLIPScore。

### Training curves 的 emergent behavior

Stage 1 训练过程中，**thinking length 持续增加**——VLM 自发学会写更长的 reasoning chain 来获得更高 reward。这不是 explicitly rewarded 的，是 emergent test-time scaling。跟 DeepSeek-R1、Video-R1 (https://arxiv.org/abs/2503.21776) 的发现一致。RL 让模型自发学会"想得更久"。

### Human evaluation

30 个 evaluator 评分 1-5：

| Model | Semantic | Visual | Overall |
|-------|----------|--------|---------|
| Gemini-FilmWeaver | 3.9 | 3.1 | 3.5 |
| VANS (SFT) | 3.8 | 3.9 | 3.7 |
| VANS (Joint-GRPO) | **4.7** | **4.6** | **4.8** |

VANS (SFT) visual 已经超过 Gemini-FilmWeaver，但 semantic 略低。Joint-GRPO 后两个维度都到 4.6+。Human satisfaction 飞跃。

### Generalization

**Multi-future prediction**: 同一 input video 用不同 question 能生成不同但 plausible 的 video。比如女人碰热东西，"现实情境"生成"咳嗽"反应，"电影夸张情境"生成"嘴里冒烟"反应。这是 deterministic NEP 做不到的。

**Reasoning I2V**: 扩展到 image-to-video，"把香蕉放一周"能正确生成香蕉皮 darkening 的因果物理变化。因为 mixed training data 包含 Koala-36M (https://arxiv.org/abs/2410.13791)。

---

## 实现细节几个值得注意的点

- **VDM 训练的 ODE→SDE 转换**: Wan 是 flow matching model，原本是 deterministic ODE。为了 enable GRPO，作者用 Flow-GRPO (https://arxiv.org/abs/2505.05470) 的方法把 ODE 转成等价 SDE，获得 stochastic policy 来算 importance sampling ratio。这是把 RL 应用到 diffusion/flow model 的关键 trick。

- **KL coefficient $\beta = 0.004$**: 比 DeepSeek 原版小很多（原版 0.04），因为这里是 post-training 不想太约束 policy 偏移。

- **Clip range $\epsilon = 1\times10^{-3}$**: 比 PPO 标准的 0.1-0.2 小三个数量级。这里 ratio 是 token-level 的，需要更严格的 clip 防止爆炸。

- **Group size $G = 8$**: 每个 prompt 采样 8 个 candidate。不大但够算 group statistics。

- **Stage 1 只训 800 steps，Stage 2 只训 1K steps**: RL post-training 非常快，因为有 SFT 好的初始化。

- **Inference time**: ~4s caption + ~35s video = ~39s 总耗时。比 unified model 快（Omni-Video 50s，VideoGPT 60s）。

---

## 我的直觉解读：这篇 paper 真正在干嘛

Andrej，我自己的 gut feeling 是这篇 paper 的核心贡献不是 VNEP 这个任务本身——VNEP 是个自然的 extension，迟早有人提。

真正的 contribution 是 **Joint-GRPO 这个 multi-model RL 范式**。

现在 AI 圈越来越多 multi-model system（VLM + tool use + executor、LLM + code interpreter + verifier、reasoner + actor 等）。这些 system 的 post-training 一直很 tricky——怎么让多个 model 协同优化？分别 RL 不够，joint RL 又有 attribution problem。

VANS 的解法很 elegant：**通过 training schedule 来 force attribution**。Stage 1 冻结 VDM 让 VLM 全责，Stage 2 冻结 VLM 让 VDM 全责。每个阶段 gradient signal 清晰，模型知道该改什么。两阶段顺序还能让 Stage 1 的成果成为 Stage 2 的 stable anchor。

这个思路是可以 generalize 的。比如 LLM agent + tool executor：先冻结 executor 调 LLM 学会发出 executor 能理解的指令，再冻结 LLM 调 executor 学会 robust 执行这些指令。或者 reasoning model + verifier：先冻结 verifier 调 reasoner，再冻结 reasoner 调 verifier。

所以这篇 paper 的价值超出 VNEP 本身，它给 multi-model RL post-training 提供了一个 actionable recipe。

---

## 我的一些 concerns

1. **Anchor caption filtering 的 train-test mismatch**: Stage 2 训练时用 ROUGE-L < 0.6 过滤 anchor caption，但 test time 没有 ground truth 可以 filter。Distribution mismatch 可能导致 test-time degradation。

2. **CLIP-based reward 是 noisy signal**: $r_{v1}$ 和 $r_{c2}$ 都用 CLIP similarity，CLIP 跟人类 visual judgment 相关性不完美。可能引入 CLIP-Score hack——生成 CLIP 高分但人类觉得奇怪的视频。Paper 里 ablation 也观察到 static frame hack，说明 CLIP reward 本身有漏洞。

3. **1K RL 数据的泛化**: 只用 1K 样本做 RL，泛化能力存疑。Multi-future prediction 的 capability 可能是 training data 已有 hypothetical question pattern 的产物，而非真正的 compositional generalization。

4. **Stage 2 只训 1K steps 可能 underfit**: Wan-2.1-1.3B 参数量不小，1K steps 调整个 DiT blocks 可能不够。

5. **没有探索 3+ model 场景**: Joint-GRPO 目前只验证了 2 model。扩展到 3+ model 的 attribution 会更复杂，可能需要更精细的 schedule 设计。

---

## 总结：你应该 take away 什么

1. **VNEP 是个自然但 overdue 的 task**: 把 video generation 从 entertainment 拉到 answer modality，逻辑上很顺。

2. **Joint-GRPO 的 two-stage 设计解决 multi-model attribution problem**: 这才是 paper 的灵魂。分别优化不够，all-in-one 不行，staged optimization 是 sweet spot。

3. **Joint reward 比 single reward 重要**: 每个 reward component 单独都不够，组合起来才能 force model 学到正确的 behavior。Reward design 是 multi-model RL 的核心难点。

4. **Emergent test-time scaling**: RL 让 VLM 自发学会写更长 reasoning chain，这跟 R1 系列发现一致。RL 是 eliciting latent capability 的有效工具。

5. **Architecture design 也贡献了不少**: VAE reference tokens + ViT high-level features 的双路径设计，光靠这个 SFT 就已经超过强 baseline 的 visual metrics。

Paper code 在 https://github.com/KlingTeam/VANS，project page 在 https://video-as-answer.github.io/。

希望这个版本更 gut-level 一点。核心 intuition 就一句：**multi-model RL 的关键是 attribution 清晰，通过 training schedule 来 force attribution 是一个 actionable recipe**。

---

# Video-as-Answer (VANS): 深度技术讲解

Andrej，这篇 paper 来自 Kuaishou Kling Team，核心创新点是把 video generation 从 entertainment 工具扩展为 next-event prediction 的 answer modality。我下面会逐层剖析技术细节，重点 build intuition 关于 Joint-GRPO 为什么 work。

---

## 1. Task Formulation: VNEP 的本质

传统 NEP 形式化：给定 input video $v_{in}$ 和 question $Q$，输出 text answer $s_{gt}$ 描述下一事件。VNEP 把输出空间从 $\Sigma^*$ (text) 扩展到 $\mathcal{V}$ (video space)，输出 video $v_{gt}$ 直观展示下一事件。

**关键区分（intuition）**：与 video extension 任务（如 Video-GPT, [arXiv:2505.12489](https://arxiv.org/abs/2505.12489)）不同，VNEP 不是基于 spatiotemporal pattern 的帧外推（forecasting 球的轨迹），而是 **event-conditioned reasoning**。模型必须先理解视频和问题，推理下一事件的因果或程序逻辑（例如观察到脏盘子在用水冲洗后，推断需要加 soap），然后生成既视觉连贯又语义忠实于所推断事件的 video。

这是 "telling → showing" 的范式转换。某些物理世界信息用 text 难以传达（例如打 Windsor knot 的步骤），而 video 可以直观展示 spatial layout、motion 和 temporal ordering，并 adapt 到用户当前状态。

---

## 2. VANS 架构解析

VANS 是 cascaded architecture，但通过 RL post-training 紧密整合两个 specialized model：

**Components:**
- **VLM**: Qwen2.5-VL-3B ([arXiv:2309.16609](https://arxiv.org/abs/2309.16609))
- **VDM**: Wan-2.1-1.3B ([arXiv:2503.20314](https://arxiv.org/abs/2503.20314))

**Information Flow:**
1. Input video $v_{in}$ 经过 ViT 提取 **high-level visual features**
2. Question $Q$ 被 tokenize，与 ViT features 一起输入 VLM
3. VLM 执行 instruction-grounded reasoning，生成 textual caption $s$ 描述预测的下一事件（"reason-then-answer" template）
4. Input video 通过 VAE tokenize **n=6 reference frames** 得到 low-level visual tokens
5. 这些 visual tokens 被 concatenate 进 VDM 的 conditioning latent space
6. VDM 同时受 caption $s$ 和 visual context tokens 条件，生成 output video $v_{out}$

**Intuition 关键点**：VLM 看 high-level semantics（ViT features），VDM 看 low-level visual cues（VAE tokens）。这种双路径设计让 VLM 专注 reasoning，VDM 专注 visual fidelity，但 SFT alone 会让两个 model 各自为政——VLM 不知道自己写的 caption 能否被 VDM visualize，VDM 不知道哪些 visual element 该 preserve 哪些该变。这就是 Joint-GRPO 要解决的 gap。

---

## 3. GRPO Preliminary

GRPO 由 DeepSeekMath 提出 ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))，核心 idea 是用 group-relative advantage 优化 policy，省去 critic。

**Setup**: 对每个 input context $c$，policy model $\pi_\theta$ 采样一组 $G$ 个 trajectories $\{o_i\}_{i=1}^G$，每个 trajectory 获得 reward $r_i$。

**Normalized advantage**：
$$\tilde{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$$

其中：
- $r_i$ = i-th trajectory 的 reward
- $\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j$ = group average reward
- $\sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2}$ = group reward 标准差

**Intuition**: $\tilde{A}_i$ 衡量 trajectory $i$ 相对 group 平均的好坏程度。如果 $r_i > \bar{r}$，$\tilde{A}_i > 0$，policy 应该增加该 trajectory 的概率；反之减少。这样不需要 value function 估计 baseline，而是用 group 平均代替。

**GRPO Objective**:
$$J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \left(\frac{1}{T_i}\sum_{t=0}^{T_i-1} \min\left(r_t^i(\theta)\tilde{A}_i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon)\tilde{A}_i\right)\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

变量解释：
- $r_t^i(\theta) = \frac{\pi_\theta(o_t^i|c)}{\pi_{\theta_{old}}(o_t^i|c)}$ = importance sampling ratio，i-th trajectory 在 t-th token 上的新旧 policy 概率比
- $T_i$ = i-th trajectory 长度
- $\epsilon$ = clip range（论文中 $1\times10^{-3}$），防止 ratio 过大偏离
- $\beta$ = KL coefficient（论文中 0.004），控制 policy 与 reference policy $\pi_{ref}$ 的距离
- $D_{KL}(\pi_\theta \| \pi_{ref})$ = KL divergence，正则项防止 catastrophic forgetting

这个公式借鉴 PPO 的 clip 机制 + DeepSeek 的 group-relative advantage，去掉 critic network。

---

## 4. Joint-GRPO: 核心创新

### 4.1 为什么 Standard GRPO 不够

Standard GRPO 在 single-model alignment 表现好，但在 VNEP 这种 multi-model 场景有两个 failure mode：

1. **Isolated optimization**: 分别对 VLM 和 VDM 应用 GRPO，各 model 优化自己的 reward，但无法 bridge semantic-to-visual gap。VLM 的 caption 可能语言正确但 VDM 无法 visualize；VDM 的 video 可能漂亮但偏离 caption 含义。

2. **All-in-one joint training 问题**: 同时优化两个 model 容易 reward hacking 和 training instability。当生成 video 质量差时，无法判断是 VLM 的 caption 错还是 VDM 的 generation 错——这是 **attribution problem**，gradient signal conflicting。

### 4.2 两阶段设计的 rationale

Joint-GRPO 通过 **structured two-stage optimization** 解决 attribution problem：

**Stage 1: Visualization-Friendly VLM Tuning**
冻结 VDM，仅优化 VLM。这样 reward 信号明确——caption 巙时 VLM 全责，video 差时也是 VLM 的 caption 不够 visualization-friendly。VLM 被迫 internalize VDM 的 capabilities 和 constraints。

**Stage 2: Context-Faithful VDM Adaptation**
冻结 VLM 作为 anchor，仅优化 VDM。此时 anchor caption $s_{anchor}$ 已经过 Stage 1 调整为 visualization-friendly，VDM 的任务变为 faithfully render 这个 caption 同时 preserve input video 的 visual context。reward 信号明确——video 差时 VDM 全责。

### 4.3 Stage 1 Reward 公式

对 input video $v_{in}$ 和 question $Q$，从 $\pi_{VLM}$ 采样 $G$ 个 captions $\{s_i\}_{i=1}^G$，每个 caption 经 **frozen** VDM 生成 video $v_{out}^i$。

Joint reward：
$$r_1(s_i, v_{out}^i) = \underbrace{\lambda_f r_f(s_i)}_{\text{format}} + \underbrace{\lambda_{t1} r_{t1}(s_i, s_{gt})}_{\text{text fidelity}} + \underbrace{\lambda_{v1} r_{v1}(v_{out}^i, v_{gt})}_{\text{video fidelity}}$$

Reward components:
- $r_f(s_i)$: **format reward**。Response 遵循 "reason-then-answer" template 给 1，否则 0。这是 binary hard reward。
- $r_{t1}(s_i, s_{gt})$: **text fidelity**。用 ROUGE-L ([aclanthology.org/W04-1013](https://aclanthology.org/W04-1013/)) 测量生成 caption 与 ground-truth caption 的语义相似度。ROUGE-L 基于 longest common subsequence，捕捉 token-level 重叠。
- $r_{v1}(v_{out}^i, v_{gt})$: **video fidelity**。用 CLIP Similarity ([arXiv:2103.00020](https://arxiv.org/abs/2103.00020)) 评估 generated video 与 ground-truth video 的视觉一致性。

$\lambda_f, \lambda_{t1}, \lambda_{v1}$ = 1（equal weight）。

**Critical Intuition**: 为什么需要 joint reward 而不是单独用 $r_{t1}$ 或 $r_{v1}$？

- 单独 $r_{t1}$：VLM 只优化语言正确性，可能生成 "linguistically correct but visually unrealistic" 的 caption。例如描述"飞人去月球"语言通顺但 VDM 生成不出。
- 单独 $r_{v1}$：reward 太 distal 模糊。当 video 质量差时，VLM 不知道该改 caption 哪方面——是 reasoning 错了？是描述太抽象？是用了 VDM 不认识的 concept？

Joint reward 的妙处在于：$r_{t1}$ 保证语义方向对（grounding），$r_{v1}$ 提供 visualization plausibility 的 feedback 信号。两个 signal 联合，VLM 学到的 caption 既是"对的"又是"VDM 能 execute 的"。

### 4.4 Stage 2 Reward 公式

Stage 1 训练后的 VLM 生成 anchor caption $s_{anchor}$（filter: ROUGE-L < 0.6 丢弃重新生成，保证 quality）。从 $\pi_{VDM}$ 采样 $G$ 个 video $\{v_{out}^i\}_{i=1}^G$。

Joint reward：
$$r_2(v_{out}^i, s_{anchor}) = \underbrace{\lambda_{v2} r_{v2}(v_{out}^i, v_{gt})}_{\text{video fidelity}} + \underbrace{\lambda_{c2} r_{c2}(v_{out}^i, s_{anchor})}_{\text{semantic alignment}}$$

Reward components:
- $r_{v2}(v_{out}^i, v_{gt})$: 与 Stage 1 相同 metric（CLIP Similarity with ground truth），维持视觉质量和与 input video 的 coherence。
- $r_{c2}(v_{out}^i, s_{anchor})$: **semantic alignment**。用 CLIPScore 测量 generated video 与 anchor caption 的语义一致性。

$\lambda_{v2}, \lambda_{c2}$ = 1。

**Critical Intuition**: 为什么 Stage 2 需要两个 reward？

- 单独 $r_{v2}$: VDM 可能 reward hack——直接复制或微调 input video 帧即可获得高 visual fidelity，但完全 ignore caption 语义。
- 单独 $r_{c2}$: VDM 可能 generate 静态帧（static frames）匹配 caption 的 CLIP embedding 但缺乏 motion，这是 paper 中观察到的 reward hacking 模式。

两个 reward 联合，VDM 必须 **dynamically attend to and preserve relevant visual elements**（如 IDs, backgrounds）from input video 的 VAE tokens，同时根据 $s_{anchor}$ 的语义内容生成 novel scene。

### 4.5 VDM 训练的技术细节

VDM 是 flow matching model（Wan），原本是 deterministic ODE。为了 enable GRPO 训练，作者采用 [arXiv:2505.05470](https://arxiv.org/abs/2505.05470) 的方法把 ODE 转换为等价 SDE（Stochastic Differential Equation），从而获得 stochastic policy 用于 importance sampling 和 ratio computation。这是把 RL 应用到 diffusion/flow model 的关键技术 trick。

---

## 5. VANS-Data-100K 数据集

| Component | Size |
|-----------|------|
| Procedural (COIN + YouCook2) | 30K (21K + 9K) |
| Predictive (Video-Holmes + ActivityNet + V1-33K + YouTube) | 70K (10K + 20K + 10K + 30K) |
| Input video avg duration | 9.43s |
| Target video avg duration | 3.76s |

**Pipeline 四阶段**：
1. **Raw Data Collection**: procedural 来自 COIN ([CVPR 2019](https://arxiv.org/abs/1904.01258)) + YouCook2 ([AAAI 2018](https://arxiv.org/abs/1812.02519))，predictive 来自 ActivityNet ([CVPR 2015](http://activity-net.org/))、V1-33K ([arXiv:2505.22457](https://arxiv.org/abs/2505.22457))、Video-Holmes ([arXiv:2505.21374](https://arxiv.org/abs/2505.21374)) + 短片
2. **Shot Split**: procedural 用 ground-truth timestamps 分段，predictive 用 shot-boundary detection model。过滤 < 3s 的 segment 保证 action 完整性
3. **Clip Selection**: Gemini-2.5-Flash ([arXiv:2403.05530](https://arxiv.org/abs/2403.05530)) 作为 quality filter，从每个 segment 选 3-5s 最优 clip
4. **QA Pair Generation**: Gemini 模拟 diverse question（procedural focus on "next step"，predictive focus on "what-if"），生成 chain-of-thought reasoning + ground-truth answer，加 self-check 防 information leakage

从 100K 中手动选 1K 高质量样本用于 RL post-training。这是典型的 SFT 大数据 + RL 小数据模式。

---

## 6. 实验结果深度分析

### 6.1 Main Quantitative Results (Table 1)

**Procedural Benchmark 关键数字**:
| Model | ROUGE-L↑ | FVD↓ | CLIP-V↑ | CLIP-T↑ |
|-------|----------|------|---------|---------|
| Omni-Video (unified) | 0.1075 | 105.32 | 0.6293 | 0.2323 |
| Gemini-Wan (cascaded) | 0.2802 | 120.34 | 0.6898 | 0.2547 |
| Gemini-FilmWeaver | 0.2802 | 110.54 | 0.7102 | 0.2773 |
| VANS (SFT) | 0.2812 | 85.34 | 0.7655 | 0.3202 |
| **VANS (Joint-GRPO)** | **0.3631** | **78.32** | **0.8021** | **0.3824** |

**Intuition 解读**：
1. Unified model (Omni-Video) 全面落后——验证了 capability trade-off 问题，单 model 难以同时 excel understanding 和 generation。
2. Cascaded baselines 中 FilmWeaver 比 Wan 强（FVD 110.54 vs 120.34），因为 FilmWeaver 专为 multi-shot consistency 设计 ([AAAI 2026](https://arxiv.org/abs/2501.08325)).
3. VANS (SFT) vs Gemini-FilmWeaver：ROUGE-L 相近（0.2812 vs 0.2802），但 FVD 大幅下降（85.34 vs 110.54）——SFT 已让 VANS 在 visual coherence 上超过强 baseline，主要因为 architectural design (VAE reference tokens)。
4. **Joint-GRPO 提升**：ROUGE-L 0.2812 → 0.3631 (+29.1%)，CLIP-T 0.3202 → 0.3824 (+19.4%)。这是决定性的飞跃，证明 RL post-training 是关键。

Video-GPT CLIP-T 仅 0.1997 最低，因为它做的是 spatiotemporal continuation 没有 event reasoning——佐证了 VNEP 不是 simple extension。

### 6.2 Ablation Study (Table 2) 关键 insight

**A. Joint vs. Isolated**：
- GRPO (VLM only): ROUGE-L 0.3190，CLIP-V 0.7798
- GRPO (VDM only): ROUGE-L 0.2812 (无变化)，CLIP-V 0.7671
- GRPO (VLM+VDM cascaded): ROUGE-L 0.2894，比 VLM only 退步
- **Joint-GRPO**: ROUGE-L 0.3631，CLIP-V 0.8021

**Intuition**: VDM only GRPO 完全没改善 text metrics——因为 VDM 不影响 caption 生成。Cascaded（VLM only 然后 VDM only）比单独 VLM only 退步，说明简单 concatenate 两个独立优化会产生 misalignment。

**B. Staged vs. All-in-one**：
- Joint-GRPO (all-in-one): ROUGE-L 0.3577，FVD 81.01，CLIP-V 0.7800
- Joint-GRPO Staged (Stage 1+2): ROUGE-L 0.3631，FVD 78.32，CLIP-V 0.8021

**Intuition**: all-in-one 退化所有 visual metrics——验证了 attribution problem。当 reward 同时回传到 VLM 和 VDM 时，poor video 难以归因，gradient signal 冲突。

**C. Reward Components**:

Stage 1 ablation:
- 去掉 $r_{t1}$: ROUGE-L 0.3631 → 0.3498，caption 准确度下降（如预测"removing the mask"失败）
- 去掉 $r_{v1}$: CLIP-V 0.7803 → 0.7668，visual consistency 下降

Stage 2 ablation:
- 去掉 $r_{c2}$: CLIP-V 0.8021 → 0.7921，且出现 reward hacking (static frames)
- 去掉 $r_{v2}$: CLIP-V 0.8021 → 0.7887，output coherence 下降

**Intuition**: 每个 reward 都不可缺。$r_{c2}$ 尤其关键——没有它 VDM 会 reward hack（生成 static frame 匹配 caption embedding 但无 motion）。

### 6.3 Training Dynamics (Figure 9)

**Stage 1**:
- (a) Format reward $r_f$ 快速饱和（< 100 steps）——VLM 快速学会 template
- (b) Text fidelity $r_{t1}$ 渐进提升
- (c) Video fidelity $r_{v1}$ 渐进提升
- (d) Total reward ~600 steps 收敛
- (e) Thinking length 持续增加——VLM 学会生成更详细 reasoning chain（emergent behavior）

**Stage 2**:
- (f) Video fidelity $r_{v2}$ 渐进提升
- (g) Semantic alignment $r_{c2}$ 渐进提升
- (h) Total reward ~1000 steps 收敛

**Intuition**: Thinking length 增加是 emergent test-time scaling——RL 让模型自发学会"想得更久"以提高 reward。这与 DeepSeek-R1 / Video-R1 ([arXiv:2503.21776](https://arxiv.org/abs/2503.21776)) 的发现一致。

### 6.4 Human Evaluation (Table 5)

30 evaluators, 20 examples each, scale 1-5:
| Model | Semantic | Visual | Overall |
|-------|----------|--------|---------|
| Video-GPT | 1.5 | 3.6 | 1.5 |
| Gemini-FilmWeaver | 3.9 | 3.1 | 3.5 |
| VANS (SFT) | 3.8 | 3.9 | 3.7 |
| VANS (Joint-GRPO) | **4.7** | **4.6** | **4.8** |

VANS (SFT) 的 visual 已经超过 Gemini-FilmWeaver，但 semantic 略低（3.8 vs 3.9）。Joint-GRPO 后两个维度都达到 4.6+，human satisfaction 跃升。

### 6.5 Generalization 实验

**Multi-Future Prediction**: 同一 input video 用不同 hypothetical question（"现实情境" vs "电影夸张情境"）能生成不同但 plausible 的 video answer。这超出了 deterministic NEP 的能力——VANS 能从单一起点 explore 多个 potential futures。

**Reasoning I2V**: 扩展到 image-to-video，把 single image 当作 static video clip。"Leave the bananas for a week" 能正确生成香蕉皮 darkening 的因果物理变化。得益于 mixed training data 包含 Koala-36M ([arXiv:2410.13791](https://arxiv.org/abs/2410.13791))。

---

## 7. 实现细节

### 7.1 SFT 阶段
- VLM: LoRA ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)) rank=8, alpha=32, 10K steps, lr=5e-5
- VDM: fully fine-tune 所有 DiT blocks, 20K steps, lr=5e-5
- 6 reference frames for VAE tokenization

### 7.2 Joint-GRPO 阶段
- Stage 1: 800 steps, lr=5e-5, LoRA rank=8 alpha=32
- Anchor caption filter: ROUGE-L < 0.6 丢弃
- Stage 2: 1K steps, lr=5e-5
- VDM: ODE→SDE 转换 ([arXiv:2505.05470](https://arxiv.org/abs/2505.05470)) enable GRPO
- $\beta = 0.004$ (KL coefficient)
- $\epsilon = 1\times10^{-3}$ (clip range)
- $G = 8$ (group size per prompt)

### 7.3 Inference
- Caption generation: ~4s
- Video generation: ~35s
- Total ~39s, comparable to cascaded baselines, faster than unified models (Omni-Video 50s, VideoGPT 60s)

---

## 8. 关键 Contribution 总结

1. **VNEP task formulation**: 把 NEP 从 text 输出推进到 video 输出，这是 multimodal reasoning 的新方向。

2. **Joint-GRPO 算法**: 通过 two-stage 结构化优化 + joint reward 解决 multi-model RL 的 attribution problem。关键技术贡献：
   - Stage 1 让 VLM internalize VDM constraints
   - Stage 2 让 VDM faithfully render visualization-friendly captions
   - 避免 all-in-one 的 reward ambiguity

3. **VANS-Data-100K**: 100K procedural + predictive video-question-answer triplets，填补 VNEP 数据空白。

4. **Emergent behaviors**:
   - Thinking length 自发增加
   - Multi-future prediction capability
   - 跨任务泛化（I2V）

---

## 9. 我的 Critique 与 Open Questions

**Strengths**:
- Joint-GRPO 的 two-stage 设计优雅地解决了 multi-model attribution problem，是 RL post-training 在 multimodal generation 上的重要进展。
- VAE reference tokens 与 VLM caption 的双路径 conditioning 让 visual consistency 和 semantic control 解耦。
- Reward 设计紧凑（每个 component 都有明确 ablation 验证）。

**Potential concerns**:
- Anchor caption filtering (ROUGE-L < 0.6 丢弃) 引入 train-test distribution mismatch——test time 没有 ground truth 用于 filter。
- CLIP-based reward $r_{v1}, r_{c2}$ 是已知 noisy signal，CLIP score 与人类 visual judgment 相关性不完美。可能引入 CLIP-Score hack。
- 1K RL 数据规模较小，泛化能力需要更多 evaluation。Multi-future prediction 的 capability 可能是 training data 中已有 hypothetical question pattern 的产物。
- Stage 2 仅 1K steps，可能 underfit VDM 的大规模参数空间。

**Open research directions**:
- 把 Joint-GRPO 扩展到 3+ models（如 VLM + VDM + Audio model）
- 探索 online RL 而非 GRPO 的 offline group-relative 版本
- 用 reward model 替代 CLIP-based reward
- 扩展到 long-horizon planning（多步 next-event 预测）

---

## Reference Links

**Paper & Code**:
- Project page: https://video-as-answer.github.io/
- GitHub: https://github.com/KlingTeam/VANS

**Core methods**:
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Flow-GRPO (VDM RL): https://arxiv.org/abs/2505.05470
- Qwen2.5-VL: https://arxiv.org/abs/2309.16609
- Wan video model: https://arxiv.org/abs/2503.20314
- LoRA: https://arxiv.org/abs/2106.09685
- CLIP: https://arxiv.org/abs/2103.00020

**Datasets**:
- COIN: https://arxiv.org/abs/1904.01258
- YouCook2: https://arxiv.org/abs/1812.02519
- ActivityNet: http://activity-net.org/
- V1-33K: https://arxiv.org/abs/2505.22457
- Video-Holmes: https://arxiv.org/abs/2505.21374
- Koala-36M: https://arxiv.org/abs/2410.13791
- UI2V-Bench: https://arxiv.org/abs/2509.24427

**Baselines**:
- TEMPURA: https://arxiv.org/abs/2505.01583
- Omni-Video: https://arxiv.org/abs/2507.06119
- Video-GPT: https://arxiv.org/abs/2505.12489
- Gemini 1.5: https://arxiv.org/abs/2403.05530

**Related RL for video**:
- Video-R1: https://arxiv.org/abs/2503.21776
- VideoChat-R1: https://arxiv.org/abs/2504.06958
- DanceGRPO: https://arxiv.org/abs/2505.07818
- GRPO-Care: https://arxiv.org/abs/2506.16141

**Metrics**:
- ROUGE: https://aclanthology.org/W04-1013/
- BLEU: https://aclanthology.org/P02-1040/
- FVD: https://arxiv.org/abs/1809.07961

希望这个深度讲解 help 你 build intuition 关于 Joint-GRPO 的设计哲学。核心 takeaway 是：multi-model RL 的关键不在 reward 本身，而在 **通过结构化训练 schedule 明确 attribution**，让 gradient signal 干净地流向 responsible model。
