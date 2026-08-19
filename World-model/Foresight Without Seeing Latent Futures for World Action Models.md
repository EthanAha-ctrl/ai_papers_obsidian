---
source_pdf: Foresight Without Seeing Latent Futures for World Action Models.pdf
paper_sha256: 8324ff119b20e43d0c2282aa6e52d17ab989546a8386badadaa39d4d3bb6e083
processed_at: '2026-08-18T22:43:09-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 ForeWAM

好，Andrej，我把刚才那堆术语扔掉，用最直白的方式重新讲一遍。

---

## 这篇 paper 在解决什么问题

想象你教一个 robot 抓桌子上的杯子。你给它看当前画面，让它输出动作。

**笨办法**：robot 只看当前画面，直接输出动作。问题是——它不知道"我抓下去之后杯子会怎么样"，所以动作很僵硬，遇到新情况就傻眼。

**聪明办法**：让 robot 先在脑子里"播放"一段未来视频——"如果我这么动，杯子会从这个位置滑到那个位置"——然后根据这段想象出来的未来视频来决定动作。这就是 WAM (World Action Model) 的思路。

但聪明办法有个麻烦：**生成未来视频太慢了**。Video diffusion 要迭代去噪十几步才能生成一段像样的视频，每次做 action 都要先这么"想象"一遍，robot 动作延迟就飙到几百毫秒甚至秒级，根本没法 real-time control。

所以现在 WAM 领域有两个阵营：

- **慢但聪明**：真的去 generate 未来视频，然后拿未来视频当 condition 来 predict action。问题是慢。
- **快但笨**：训练时让 model 学会想象未来，但推理时直接跳过想象步骤，从当前 observation 直接 predict action。这就是 Fast-WAM 那一派。问题是——你训练时虽然学了 future modeling，但推理时把这个能力砍掉了，action predictor 就"看不见"未来了，相当于你训练一个会预判的运动员但比赛时让他蒙着眼睛只凭直觉打。

**ForeWAM 想干的事**：又快又聪明。推理时不真的 generate 未来视频，但要能让 action predictor "感受到"未来的存在。

---

## ForeWAM 的核心 trick：想，但不说出来

这个 trick 说出来其实特别 simple。

你想啊，video diffusion model 在 denoise 未来视频时，它的中间层 hidden states 里其实已经"想"了未来是什么样——只不过这些 hidden states 最后被 decode 成了 pixels。如果你能偷听它"想"的过程，就不用等它"说"出来了。

**ForeWAM 就是偷听 video model 的"想法"，不等它把想法变成 pixels**。

具体怎么做：

1. **构造一个"假的未来"**：当前帧是真实的（clean latent），后面的 future slots 全填噪声。这看起来就像一段还没 denoise 的视频 latent。

2. **让 Video DiT 跑一次 forward pass**：把这段"当前帧 + 噪声未来"塞进 video model，让它走一遍。注意只走一次，不迭代 denoise。model 在这一步中会"扫一眼"这个序列，中间层产生一堆 K/V (key-value) states。

3. **把 K/V 缓存起来**：这就是偷到的"想法"。Video model 在这步 forward 中，它的 attention K/V 编码了"给定当前帧和 instruction，未来大概会往哪个方向演变"的统计信息——虽然未来 slots 是噪声，但 model 的 prior 让它对噪声做了 task-conditioned 的 interpretation。

4. **Action DiT 做动作时反复读这个 K/V cache**：Action model 在 denoise action chunk 的每一步，都去 attend 这份 cached K/V。相当于 action predictor 在"参考"video model 对未来的隐式判断。

**就这么简单**。一次 video forward pass 的开销，换来了 action pathway 对"未来"的感知能力。没有真的生成视频，没有 pixel decode，就一个 prefill pass。

这个思路特别像 LLM 里的 **prefix caching**——你把 prompt 喂进去一次，缓存住 K/V，后面生成 token 时反复读。ForeWAM 把"未来"当作一个 prefix，prefill 一次，然后 action denoising 时反复读。

也像人类运动员投篮前的 **mental practice**——你在脑子里 simulate 一下整个动作轨迹，但你不需要真的在眼前 hallucinate 出一幅高清画面，你只需要一个 "feel of the future" 来指导肌肉发力。ForeWAM 就是给 robot policy 装了一个 "mental practice" 模块，只不过这个 mental practice 是个 noise-initialized latent + 一次 forward pass。

---

## 但是有个问题：video model "想"的东西太杂了

你让 video model 自由地想未来，它会把什么都想进去——背景纹理、光照变化、桌子的纹路、跟任务无关的所有视觉细节。这些对 action prediction 来说大部分是噪音。

你需要一种方式告诉 video model："hey，我想让你关注的是**动作引起的变化**——东西怎么动、接触怎么变、任务进行到哪一步了——别给我扯什么背景纹理"。

这就是 **Dynamics Registers** 登场的地方。

---

## Dynamics Registers：专门记录"变化"的小本本

ForeWAM 在 Video DiT 里塞了 16 个特殊的 token，叫 dynamics registers。这些 register 的任务很简单：**总结"这个 transition 是怎么发生的"**。

怎么训练它们去关注 transition？这里用了一个很巧妙的设计——**找一个 frozen teacher 来当监督**。

teacher 是 LaWM（Latent World Action Model），一个已经训练好的 latent action encoder。你给它看"动作前"和"动作后"两帧画面，它吐出一个 32 维的 vector，描述这个 transition（注意是"transition 描述"，不是"motor command"，它是抽象层面的"发生了什么"，不能直接执行）。

训练时：
- ForeWAM 的 16 个 dynamics registers 做一次 mean pooling，通过一个 projection head，去逼近 teacher 的输出。
- teacher 的输出加 stop-gradient（不让 gradient 回传去改 teacher）。
- 这样 registers 被迫去 encode "transition-relevant" 的信息，而不是所有视觉信息。

推理时 teacher 不在场——registers 已经学会"从当前帧 + 噪声 future 中提取 transition summary"，这个能力 transfer 到了纯噪声的 inference 场景。

**这个设计的 intuition**：你给 model 一个 information bottleneck（16 个 register mean pool 到 32 维），再用一个 transition-specialist teacher 监督它，逼着它只保留"跟动作引起的变化有关"的信息。多余的全丢掉。

有点像你让一个学生写读书摘要——他会把什么都写进去；但你给他一个"只关注人物动机"的 rubric，他就只摘那个。LaWM teacher 就是这个 rubric。

---

## 两条路：rich context + compact summary

ForeWAM 最优雅的地方在于它给 action predictor 提供了**两条互补的 future 信息通道**：

**Channel 1: Future-KV** — Video DiT 所有层的 K/V cache。这是 rich、distributed、spatiotemporal 的视觉 context。相当于"完整但嘈杂的未来想象"。

**Channel 2: Dynamics Registers** — 16 个 register 经过 LaWM 监督学到的 compact transition summary。相当于"未来想象的一句话摘要"。

Action DiT 在 denoise 动作时，两条路都读——既看完整 K/V，又看 register 摘要。

为什么要两条？因为它们捕捉不同东西：
- Future-KV 保留 spatial detail（"杯子在哪、怎么朝向"）
- Registers 保留 transition essence（"杯子要被抓起来"）

Ablation 数据证明了这点：
- 只用 Future-KV：58.5%
- 只用 LA supervision（registers）：58.0%
- 两个都用：61.6%

1+1 > 2 的效果，说明它们是 complementary 的，不是 redundant 的。

我个人觉得这个设计有点像大脑的两条视觉通路：
- **Ventral stream**（"what" pathway）— 关注物体 identity、rich visual detail → 像 Future-KV
- **Dorsal stream**（"where/how" pathway）— 关注 motion、spatial relations、action-relevant info → 像 Dynamics Registers

大脑用两条路并行处理视觉信息来指导 action，ForeWAM 也用了类似的 dual-pathway 设计。

---

## 效果到底怎么样

光看数字：

**LIBERO（标准 benchmark）**：
- ForeWAM: 96.7% overall
- Fast-WAM: 97.6%（高 0.9pt，但 Fast-WAM 是 6B params，ForeWAM 是 2B，**1/3 的参数量**）
- ForeWAM-Flash: 96.9%（distill 到 2 步 denoising，几乎不掉点）

**LIBERO-Plus（robustness test，各种 perturbation）**：
- ForeWAM: 61.6% overall
- Fast-WAM: 51.5%（**ForeWAM 高 10.1pt**）
- 最夸张的是 camera viewpoint shift：**62.5% vs 16.4%**，+46pt！
- Sensor noise：58.8% vs 37.7%，+21pt

**推理延迟**：
- Fast-WAM: 667ms
- ForeWAM: 568ms
- ForeWAM-Flash: **220ms**（10Hz control 完全够用）

**这些数字告诉我们什么**：

1. **Camera shift 上 +46pt 是真的惊人**。传统 direct-policy 遇到 viewpoint 变化就崩，因为它过度依赖当前 observation 的 pixel-level feature。ForeWAM 的 dynamics registers 学到的是 transition-level abstraction（"这个东西被抓起来"），这种 abstraction 对 viewpoint 比较 invariant。LaWM teacher 的 supervision 起了关键作用——teacher 本身就是 viewpoint-robust 的 transition descriptor。

2. **2B 打 6B 只差 0.9pt**——Future-KV 的 K/V 复用让 Action DiT 能"借力"Video DiT 的容量，不需要 Action DiT 自己重新学 future representation。这是 parameter efficiency 的大胜利。

3. **Flash 版本几乎不掉点**——action manifold 在 2 步内就 converge 了，说明 action space 的 generative distribution 比 video space 简单很多。这也符合直觉——action 是低维的（7 维），video 是高维的，前者本来就该 easy to denoise。

4. **不需要 embodied pretraining**——OpenVLA、π0、π0.5 都在 large-scale robot data 上 pretrain 过，ForeWAM 直接在 LIBERO 上从头训就能 96.7%。这是 future-KV 这个 inductive bias 带来的 sample efficiency。

---

## 跟其他方法的区别在哪

跟几种 related idea 的对比，帮你 build intuition：

### vs. Dreamer (latent world model + actor-critic)

Dreamer 在 latent space 里 roll out 多步未来，从 rollout trajectory 里预测 reward 和 action。ForeWAM 只做 **single-step "prefill"**——不 roll out，不预测 reward，就一个 forward pass 把 K/V 抽出来。

可以理解为 **"one-step Dreamer"**：你 dream 一下未来，但只取这一个 step 的 hidden state 作为 planning signal，不做 multi-step rollout。简单很多，也快很多。

参考：https://arxiv.org/abs/2301.04104

### vs. JEPA (LeCun 的 predictive architecture)

JEPA 哲学是"在 latent space 预测，不在 pixel space 预测"。ForeWAM 的 future slots 是 noise（相当于 mask），Video DiT 在 latent space "fill in" 这些 slots 的 hidden representation——这就是 latent-level prediction。但 ForeWAM 不要求 reconstruct masked regions，只要求 hidden states 对 action 有用。

这跟 LeCun 的哲学高度一致——predict 在 abstract space，不要在 pixel space。ForeWAM 可以看作 "JEPA applied to action conditioning"。

参考：https://arxiv.org/abs/2301.08243

### vs. Latent Chain-of-Thought

最近有些工作做 "latent CoT"——在 latent space 做多步 thinking 然后输出 answer。ForeWAM 的 future slots 可以看作 **"latent planning tokens"**——它们承载了 network 对未来的 implicit reasoning。

区别是 ForeWAM 用 single forward pass（不像 latent CoT 通常需要 iterative refinement），更接近 "amortized planning"——你训练时学到的是"给定 current state + noise future，直接产出有用的 planning representation"，不需要 inference-time iteration。

参考：https://arxiv.org/abs/2412.06769

### vs. MAE (Masked Autoencoder)

MAE mask 掉 patches 让 encoder 预测。ForeWAM 的 future slots 用 noise 而不是 mask token，但精神类似——都是 "fill in the missing parts"。差别是 ForeWAM 不要求 reconstruct masked regions，只要 hidden states useful for downstream (action prediction)。

---

## 一些更深的联想和猜测

### 为什么 noise 而不是 mask token？

我猜是因为 video DiT 本来就是 diffusion model，它的 input distribution 里就经常出现"noisy future latents"。你给它 noise-filled future slots，它在 native distribution 里就能 handle。如果用 [MASK] token，反而 out-of-distribution 了。

这也意味着 future slots 的 noise 在某种意义上是"diffusion-native 的 mask token"——diffusion model 已经知道怎么从 noisy input 中 extract useful representation。

### 为什么 single forward pass 就够？

直觉上你可能会问：未来是 noise，一个 forward pass 怎么可能学到"有用的未来信息"？

答案是：训练时，Video DiT 学到的是 "given (noisy future, current clean, instruction) → predict denoising velocity"。在这个 conditional prediction 中，network 必须学会从 instruction + current frame 中 extract "future 会变成什么样" 的 prior，因为 future slots 本身没信息（是 noise）。

所以 future slots 在 inference 时其实是 **"placeholder for future"**——它们的 hidden representation 是 instruction + current frame 的 function，编码了 "task expects the world to evolve this way" 的 prior。这个 prior 对 action prediction 是有用的。

某种程度上，future slots 的 noise 是 "diagnostic probe"——你把不确定性塞进去，让 network 用 prior 去 fill in，fill in 的结果（hidden states）就反映了 network 的 mental model of the future。

### 为什么 mean pooling registers 有效？

16 个 registers mean pool 到一个 vector，再 project 到 32 维去匹配 LaWM teacher。这个 bottleneck 是关键——它逼着 registers 去 compress 信息。

如果直接用 16 个 registers 的 concatenated features（16 × 1536 = 24576 维），supervision 就太弱了——这么大的空间，project 到 32 维随便都能 fit。

mean pool 是更 aggressive 的 bottleneck，强制 registers 之间分工合作，每个 register 关注 transition 的不同 aspect，最后 pool 起来形成 complete summary。

### Future slots 的 stochasticity 没被利用

我注意到一个机会：Eq. 3 里 $\epsilon_F$ 是 sample 出来的，每次推理 K/V cache 都不同。论文没利用这个 stochasticity——没做 multi-sample averaging，没研究 variance 的影响。

理论上你可以 sample N 次，得到 N 个 K/V cache，做 ensemble。可能进一步 reduce noise、提升 robustness。这就像 diffusion model 里的 "ensemble of samples" 思路，只不过这里是 "ensemble of imaginations"。

### Long suite 上为什么差 2.4pt？

ForeWAM 在 Long horizon 任务上 92.8%，Fast-WAM 是 95.2%。差距最大。

Long suite 任务需要 10 步以上的 long-horizon planning。ForeWAM 的 single prefill 可能 insufficient——一个 forward pass 不足以 capture 长 horizon 的全部 dynamics。

可能的改进：用 iterative refinement——Action DiT 的 intermediate output 反过来 refine future slots 的 noise，再 prefill 一次，形成 "alternating imagination-action" 循环。这就更接近真正的 mental simulation。

参考 iterative latent CoT: https://arxiv.org/abs/2402.06319

---

## 最关键的 intuition

如果只记一句话：

> **Policy 不需要看见未来，但 policy 的 attention 需要能 attend 到一个"想象的未来"的 hidden representation。**

ForeWAM 的精髓是：把"未来"作为一个 latent substrate 喂给 video model 一次，让它"扫一眼"产生 K/V cache，然后让 action model 在 denoise 动作时反复读这份 cache。

未来不需要被 decode 成 pixels，只需要在 hidden state 里"存在过"一次，就足以指导 action。

这跟人类认知的 "mental imagery" 高度类似——你投篮前不需要在眼前 hallucinate 出高清视频，你只需要一个粗略的 "feel" 来校准动作。ForeWAM 给 robot policy 装了一个 mental imagery 模块，只不过这个 imagery 是 noise + 一次 forward pass + K/V cache 的组合。

---

## 一些 personal thoughts

Andrej，我读完这篇 paper 的第一反应是：**这个 idea 早该有人做了**。

把 LLM 的 prefill-cache 思想迁移到 video model 上，让 action pathway 通过 K/V cache "蹭" video model 的 future reasoning——这是一个非常 natural 的 idea，但需要同时理解 LLM inference 优化、video diffusion、robot learning 三个领域的人才能想到。

更难得的是 implementation 干净——Future-KV 和 Dynamics Registers 两个 component 互相 complement，ablation 证明了它们各自贡献、合在一起更强。不是 "kitchen sink" 式的堆 component，而是 principled design。

2B params、220ms latency、LIBERO 96.9%、不需要 embodied pretraining——这些数字加起来，让 ForeWAM 在 real-world robot deployment 上非常有 competitive 力。

如果说 Fast-WAM 是 "efficient but blind to future"，explicit-future WAM 是 "sees future but slow"，那 ForeWAM 是 **"feels the future without seeing it"**——这是我个人觉得最 promising 的 WAM paradigm。

希望这个人话版本讲清楚了，Andrej。如果还想深挖某个具体点（比如 attention mask 的实现、OneDP distillation 的细节、或者 future substrate 的 alternative design），随时说。

---

# ForeWAM: 在不"看见"未来的情况下获得 Foresight

Andrej, 这篇论文触及了一个非常 deep 的问题——**如何让一个 policy 拥有"预测未来"的能力，但不需要在推理时真的去 generate 那段未来**。这让我联想到 LLM 推理中的 prefill-decode 范式被迁移到 video diffusion 上的奇思妙想。让我把这篇 paper 的 intuition、技术细节、关键设计选择拆解给你看。

---

## 1. WAM 范式图谱：ForeWAM 站在哪儿

paper 的 Figure 1 把 World Action Model (WAM) 的设计空间分成四类，这是理解全文的坐标系：

| Paradigm | Future 处理 | Action 路径 | 推理开销 |
|---|---|---|---|
| (a) Cascaded WAM | 先 generate future obs → 再 condition action | $p(a|u_{1:T})$ | 高（iterative denoising） |
| (b) Joint WAM | 联合 denoise future + action | shared backbone | 中（耦合但仍是多步） |
| (c) Direct-policy WAM (Fast-WAM) | 训练时 modeling future，推理时 skip | $p(a|o,l,p)$ | 低，但 action expert 失去 future 接口 |
| **(d) ForeWAM (本文)** | latent future slots 作为内部 conditioning substrate | $p(a|o,l,p,D_\theta,\mathcal{H}_{KV})$ | 低 + predictive context |

**核心矛盾**：(c) 用效率换取了"信息透明度"——你把 future-video 生成砍掉后，Action DiT 一下子失去了"看到世界将如何演变"的途径。ForeWAM 的核心 insight 是：**你不需要真的把 future 像素解码出来，只要让 Video DiT 的 hidden states 里"想一下"未来，然后把这些 hidden states 通过 K/V cache 喂给 Action DiT 就够了**。

这本质上是一种 **"latent imagination"**：future 不出现在 pixel space，也不出现在 token space，而是出现在 attention 的 K/V space 里。这个思想非常接近 Friston 的 active inference 中 implicit planning 的概念——大脑在动作前确实有 "motor imagery"，但这种 imagery 不需要 fully render 出来，它只需要在 higher-level representation 中存在就足以指导行为。

参考链接：
- Fast-WAM: https://arxiv.org/abs/2603.16666
- Active inference 综述: https://www.nature.com/articles/s41583-024-00877-z

---

## 2. Future-KV：把 LLM prefill 思想搬到 Video DiT 上

这是论文最精彩的部分。让我把数学讲清楚。

### 2.1 Stochastic future substrate 的构造（Eq. 3）

$$\tilde{\boldsymbol{z}}_{1:T}^{\mathrm{Fsub}} = \mathrm{concat}(\boldsymbol{z}_{\mathrm{cur}}(o), \boldsymbol{\epsilon}_F), \quad \boldsymbol{\epsilon}_F \sim \mathcal{N}(0, I)$$

变量含义：
- $\tilde{\boldsymbol{z}}_{1:T}^{\mathrm{Fsub}}$：stochastic future substrate，是一个长度为 $T$ 的 latent sequence
- $\boldsymbol{z}_{\mathrm{cur}}(o)$：当前帧经过 VAE encode 后的 clean latent（占据 sequence 的第一个位置）
- $\boldsymbol{\epsilon}_F$：填充到 future slots 的高斯噪声
- $T$：video latent 的时序长度（论文中是 9，因为 32 action steps / temporal ratio 4 ≈ 9 video frames）

**Intuition**：这个 substrate 长得就像一个"待 denoise"的 video latent sequence——首帧是已知的 clean frame（条件），后面全是噪声。但这里的关键 trick 是：**我们不打算把它 denoise 出来**。我们只是借 Video DiT 的 prefill pass 来"读"这个序列，把读到的 K/V cache 抽出来给 Action DiT 用。

### 2.2 Single-pass prefill（Eq. 4）

$$\left(D_\theta, \mathcal{H}_{\mathrm{KV}}\right) = \mathrm{KVPrefill}_\phi\left(\tilde{z}_{1:T}^{\mathrm{Fsub}}, l, p\right)$$

变量含义：
- $D_\theta$：dynamics-register slice，是从 Video DiT 内部某些 register token 位置抽出来的 hidden states
- $\mathcal{H}_{\mathrm{KV}}$：per-layer 的 K/V cache，每一层都有一份
- $\phi$：Video DiT 的参数（用 Wan2.1-T2V-1.3B 初始化）

**这个操作只在推理时执行一次**，时间步固定在 $\sigma = 1.0$（即 fully noisy 状态）。注意这里其实是个有点 subtle 的设计——通常 flow matching 的 prefill 应该是 conditional context 处理，但这里把 noisy future 一起塞进去 prefill，让 network 在最大噪声水平上"扫一眼"未来 latent slots，输出 K/V。

为什么这能 work？我的猜测是：训练时 Video DiT 学到了 "given 当前帧 + 噪声 future slots + instruction → predict velocity field" 的能力。在这个过程中，network 的中间层 activations 一定编码了 "instruction 和当前帧约束下，未来可能演变成什么样" 的统计信息。即使你推理时只做一次 forward（不迭代 denoise），这个 forward 的 K/V 仍然是 task-conditioned 的 predictive features。

这非常像 LLM 中的 **prefix caching**：你不需要每次生成 token 都重新 encode prompt，prompt 的 K/V 已经包含了所有你需要的信息。ForeWAM 把同样的思想用在 Video DiT 上——把"想象的未来"作为一个 prefix，prefill 一次，然后在 Action DiT 的 denoising 过程中反复读取它。

### 2.3 为什么是 K/V cache 而不是直接复用 hidden states？

这涉及到 attention 的本质。如果 Action DiT 直接 concat Video DiT 的 hidden states 作为输入 token，那么 Action DiT 必须自己做 cross-attention 或者把它们当成额外的 input tokens 处理——这两种方式要么 params 多，要么信息融合受限。

K/V cache 复用是一个更优雅的设计：
- Action DiT 的每个 query 都能 attend 到 Video DiT 的 keys/values
- Action DiT 自己的 K/V 是按 denoising step 重新计算的（因为 action tokens 在变）
- Video K/V 是 frozen 的（因为 future slots 在 action denoising 期间不变）

公式上是：在每个 action denoising step，每个 action query 计算 attention 时：
$$\text{Attention}(Q_a, [K_a; K_v^{\text{cached}}], [V_a; V_v^{\text{cached}}])$$

其中 $K_a, V_a$ 是当前 step 重新算的 action K/V，$K_v^{\text{cached}}, V_v^{\text{cached}}$ 是 prefill 时算好并冻结的 video K/V。

参考：
- Prefix caching in LLM: https://arxiv.org/abs/2312.03234
- Wan2.1: https://arxiv.org/abs/2503.20314

---

## 3. Dynamics Registers + LaWM 监督：让 implicit future "知道"要关注什么

Future-KV 本身有一个问题：**video flow matching 的监督信号是 generic 的**——它只要求 network 能 reconstruct 任意 future 帧，包括背景、光照、与 task 无关的细节。这些信息对 action 来说大部分是冗余的。

为了让 implicit future 编码 **interaction-relevant transitions**（object motion、contact changes、task progress），论文引入了 dynamics registers + latent-action supervision。

### 3.1 LaWM teacher 提供什么监督

LaWM (Latent World Action Model, Chen et al. 2026a) 是一个 frozen 的 latent-action encoder，它被训练成 inverse-dynamics model：给定 (before_frame, after_frame)，输出一个 quantized latent action $z_{LA}$，描述"这个 transition 是怎么发生的"。

注意 $z_{LA}$ 是 **non-executable** 的——它不是 motor command，而是 transition 的 abstract descriptor。这一点很重要，因为 ForeWAM 不想引入第二个 action decoder，只是想用 $z_{LA}$ 作为 "transition summary" 的 supervisory signal。

### 3.2 Dynamics registers 的训练目标（Eq. 9）

$$\mathcal{L}_{\mathrm{LA}} = \left\| g_\psi\left(\frac{1}{N_D}\sum_{i=1}^{N_D} D_i\right) - \mathrm{sg}(z_{\mathrm{LA}}) \right\|_2^2$$

变量含义：
- $g_\psi$：trainable projection head，把 dynamics register 的特征空间映射到 teacher 的 latent-action space
- $N_D$：dynamics registers 数量（论文中 $N_D = 16$）
- $D_i$：第 $i$ 个 dynamics register 的 hidden state
- $\frac{1}{N_D}\sum D_i$：mean-pooled dynamics register 表示
- $z_{LA}$：teacher 输出的 32-dim latent action target
- $\mathrm{sg}(\cdot)$：stop-gradient，只对 teacher target 应用（student 端的 gradient 仍然流过）

**这个设计的妙处**：
1. **Information bottleneck**：16 个 register mean-pool 后通过一个小 projection head 去匹配 32-dim target。这强制 register 只能保留 transition-relevant 信息，丢弃无关视觉细节。
2. **Routing 设计**（Figure 3）：dynamics registers 既能 attend 到 current frame（去理解当前状态），也能被 future-slot tokens attend 到（让 future 表征 contact 到 transition 信息）。Action tokens 则能 attend 到所有这三类。
3. **训练-推理一致性**：teacher 只在训练时出现，$z_{LA}$ 是从 demonstrated (before, after) frame pair 算出来的。推理时 future slots 是噪声，registers 是从噪声 substrate prefill 出来的——但因为训练时 registers 已经学会"从 noisy future + clean current 中提取 transition-relevant summary"，这个能力 transfer 到了纯噪声 future 的推理场景。

### 3.3 Attention mask 的 routing 细节

Figure 3 中的 mask 仔细规定了信息流：
- **Current tokens $C$**：可以被 $D$, $F$, $A$ attend
- **Dynamics registers $D$**：可以 attend $C$，可以被 $F$, $A$ attend
- **Future-slot tokens $F$**：可以 attend $C$, $D$，可以被 $A$ attend
- **Action tokens $A$**：可以 attend $C$, $D$, $F$

这个 routing 让 future-slot tokens 成为"汇聚点"——它们吸收 current frame 信息 + register 提供的 transition summary，然后再被 action tokens 读取。这是一个三段式的 information flow：current → registers + future → action。

这个设计有点类似 Vision Transformers with Registers (Darcet et al.) 中的 register tokens，但用途不同：ViT registers 用来吸收 global information 应付 attention sink，这里用来塑造 transition representation。

参考：
- ViT with Registers: https://arxiv.org/abs/2309.16588
- LaWM: https://arxiv.org/abs/2606.15768
- Latent action concepts (LAPA): https://arxiv.org/abs/2410.11758

---

## 4. 训练目标函数详解

### 4.1 Flow matching 基本形式（Eq. 6-7）

$$y_t = (1-t)y + t\epsilon$$

$$\mathcal{L}_{\mathrm{FM}}(y) = \mathbb{E}_{y, \epsilon, t}\left[\| f_\theta(y_t, t, o, l, p) - (\epsilon - y) \|_2^2\right]$$

变量含义：
- $y$：target，可以是 video latent $z_{1:T}$ 或 action chunk $a_{1:H}$
- $\epsilon \sim \mathcal{N}(0, I)$：高斯噪声
- $t \in [0, 1]$：flow matching 时间变量
- $y_t$：interpolated state，$t=0$ 时是 clean target，$t=1$ 时是 pure noise
- $f_\theta$：denoising network
- Target velocity：$\epsilon - y$，这是从 $y$ 到 $\epsilon$ 的直线方向

注意 flow matching 与 DDPM 的区别：flow matching 用 linear interpolation 和 constant velocity field，优化更稳定，且对 step 数更友好（这就是为什么后面 OneDP distillation 能把 10 步压到 2 步）。

训练 schedule：1000 timesteps，shift 5.0（这个 shift 是 flow matching 中的 common trick，让更多采样点集中在 difficult region）。

### 4.2 三个 loss 项（Eq. 8-10）

$$\mathcal{L} = \mathcal{L}_{\mathrm{video}} + \mathcal{L}_{\mathrm{action}} + \lambda_{\mathrm{LA}}\mathcal{L}_{\mathrm{LA}}$$

- $\mathcal{L}_{\mathrm{video}} = \mathcal{L}_{\mathrm{FM}}(z_{1:T})$：训练 Video DiT 学会 predict future latent
- $\mathcal{L}_{\mathrm{action}} = \mathcal{L}_{\mathrm{FM}}(a_{1:H})$：训练 Action DiT 学会 predict action chunk
- $\lambda_{\mathrm{LA}}\mathcal{L}_{\mathrm{LA}}$：dynamics register 的 transition supervision

**关键点**：论文特别强调在 end-to-end 配置下，action loss 的 gradient 可以通过 video-to-action K/V interface 流回 Video DiT。这意味着 Video DiT 在训练时不仅受 video loss 驱动，还受到 action loss 的间接影响——它学到的 K/V 表示被优化成 "对 action prediction 有用" 的形式。这是一个重要的 inductive bias，让 Future-KV 不是纯粹的 video prediction features，而是 action-relevant predictive features。

参考：
- Flow matching: https://arxiv.org/abs/2210.02747
- OneDP (distillation): https://arxiv.org/abs/2410.21257

---

## 5. 实验数据深度解读

### 5.1 LIBERO 主表（Table 1）

| Method | Params | PT | Spatial | Object | Goal | Long | Overall |
|---|---|---|---|---|---|---|---|
| OpenVLA | 7B | Yes | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0 | 3.3B | Yes | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π0.5 | 3.3B | Yes | 98.8 | 98.2 | 98.0 | 92.4 | **96.9** |
| Fast-WAM | 6B | No | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| **ForeWAM** | **2B** | **No** | 97.0 | 99.6 | 97.2 | 92.8 | 96.7 |
| **ForeWAM-Flash** | **2B** | **No** | 97.8 | 99.2 | 97.4 | 93.0 | 96.9 |

观察：
1. **2B params 打 6B 的 Fast-WAM**：差 0.9 pt overall，但 params 只有 1/3。这是巨大的 efficiency gain。
2. **不需要 embodied pretraining**：所有用 PT 的方法（OpenVLA、π0、π0.5）都在大规模 robot data 上 pretrain 过，ForeWAM 直接在 LIBERO 上训就能达到 96.7%。
3. **ForeWAM-Flash ≈ ForeWAM**：distillation 把 10 步压到 2 步，性能几乎不掉（96.9% vs 96.7%）。这说明 action manifold 在 2 步内就能 converge。
4. **Long suite 上差距最大**（92.8 vs 95.2）：长 horizon 任务对 future modeling 的需求最高，Fast-WAM 通过 train-time future modeling 学到的 representation 还是有一些优势。

### 5.2 LIBERO-Plus 鲁棒性（Table 2）—— 这是论文真正的"亮点"

| Method | Camera | Robot | Lang | Light | Bg | Noise | Layout | Overall |
|---|---|---|---|---|---|---|---|---|
| Fast-WAM | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| **ForeWAM** | **62.5** | 37.4 | 73.0 | 74.1 | 55.1 | **58.8** | 70.4 | **61.6** |
| ForeWAM-Flash | 57.9 | 40.4 | 67.2 | 71.0 | 53.0 | 53.7 | 65.3 | 58.2 |

**+46.1 pt on camera viewpoint shift** 和 **+21.1 pt on sensor noise** 这两个数字令人震惊。我的解读：

- **Camera shift 上的巨大提升**：传统 direct-policy（Fast-WAM）依赖当前 observation 的视觉特征，camera 一变就崩。ForeWAM 的 dynamics register 学到的是 transition-level abstraction（contact、object motion），这种 abstraction 对 camera viewpoint 更 invariant。Latent action supervision 起了关键作用——teacher 的 $z_{LA}$ 本身就是 viewpoint-robust 的 transition descriptor。
- **Sensor noise 上的提升**：Future-KV 提供的 spatiotemporal averaging 效应——attention 在 noisy observation 上的 K/V 已经被 Video DiT "purified" 了一次，noise 被 implicit denoise 掉了。
- **Robot initial state 反而下降**（37.4 vs 44.5）：这是 ForeWAM 的弱点。Robot state 变化时，proprioceptive condition 变了，但 future substrate 没有相应调整（它只 condition on instruction + proprio + current visual）。可能 future slot 的 noise initialization 没有捕捉到 robot state 的 uncertainty。

### 5.3 推理延迟（Table 3）

| Method | Latency (ms) |
|---|---|
| Fast-WAM | 667 |
| ForeWAM | 568 (-14.8%) |
| ForeWAM-Flash | **220 (-67.0%)** |

220ms 的 action generation latency 对 real-time control 来说是 game-changer。10Hz 的 control frequency 完全可以达到。注意这还不包括 perception 和 robot communication 的开销，但已经是 real-world deployment 的 realistic 数字。

### 5.4 Ablation（Table 4）

| Config | Overall |
|---|---|
| Base policy | 53.6% |
| Future-KV only | 58.5% |
| LA supervision only | 58.0% |
| **Both** | **61.6%** |

Future-KV 贡献 +4.9，LA supervision 贡献 +4.4，both 贡献 +8.0。这说明两者有 **complementary** 的效应——Future-KV 提供 distributed visual context，LA 提供 compact transition summary，两者不可替代。这种 complementarity 让我想到大脑中的 "dorsal stream"（where/how pathway, 关注 motion 和 action）和 "ventral stream"（what pathway, 关注 identity）的分工——Future-KV 像是 ventral（rich visual），registers 像是 dorsal（action-relevant summary）。

---

## 6. 几个值得深挖的设计细节

### 6.1 Architecture specifics

- **Video DiT**：Wan2.1-T2V-1.3B，30 layers，hidden dim $d_v = 1536$
- **Action DiT**：30 layers，hidden dim $d_a = 1024$，从 Wan2.1 ActionDiT checkpoint 线性插值初始化
- **Action horizon**：$H = 32$
- **Video frames**：9（每 4 个 action step 对应 1 video frame，temporal ratio 4）
- **Observation**：33 frames（覆盖 action chunk + 一些 context）
- **Cameras**：2 个 synchronized views，concat 沿 width 方向成 $224 \times 448$
- **Action dim**：7（6-DoF end-effector pose + 1 gripper）
- **Proprio**：8-dim
- **Registers**：$N_D = 16$ dynamics registers，readability registers 关掉

### 6.2 为什么用 Wan2.1 而不是更大的 video model？

Wan2.1-T2V-1.3B 是 compact 的 choice。作者明确说 "using only a compact Wan2.1-T2V-1.3B Video DiT and eliminating the need for embodied robot-data pretraining"。这说明 Future-KV 的设计让小 model 也能达到 SOTA——因为 K/V cache 复用让 Action DiT 能"借力"Video DiT 的全部容量，而不需要 Action DiT 自己重新学 future representation。

### 6.3 OneDP distillation 的角色

ForeWAM-Flash 用 OneDP (Wang et al. 2024) 把 action denoising 从 10 步 distill 到 2 步。注意 OneDP 是 consistency-style distillation，它在 action space 学一个 consistency model。这里的关键是 **Future-KV 和 dynamics register 的接口在 distillation 后仍然保留**——distillation 只压缩 action denoising schedule，不动 Video DiT prefill 和 K/V cache 复用。这是为什么 Flash 版本几乎不掉点（96.9% vs 96.7%）。

参考：
- OneDP: https://arxiv.org/abs/2410.21257
- ACT (action chunking): https://arxiv.org/abs/2304.13705

---

## 7. 与相关工作的更深联想

### 7.1 与 LaWM 的关系

LaWM 本身是一个 latent world action model，用 VQ-VAE 把 (before, after) frame pair 编码成 discrete latent action。ForeWAM 借用 LaWM 作为 frozen teacher——这是一个非常 cheap 但 effective 的选择，因为 LaWM 已经在大规模 data 上预训练过，它的 latent action representation 是 transferable 的。

这种 "teacher 提供 auxiliary supervision" 的范式让我想到 BERT 的 distillation——student 学一个 rich representation，但用一个简化的 target 监督。ForeWAM 的 student（dynamics registers）比 teacher 的 representation 更 embedded 在 action pathway 中。

### 7.2 与 Dreamer / World Models 的对比

Dreamer (Hafner et al.) 是经典的 latent world model + actor-critic paradigm。ForeWAM 与 Dreamer 的核心区别：
- Dreamer：在 latent space 中 roll out 多步，从 roll-out trajectory 中预测 reward 和 action
- ForeWAM：只做 single-step "prefill"，不 roll out，不预测 reward，直接给 action pathway 提供 K/V context

ForeWAM 可以看作是 **"one-step dreamer"**——你 dream 一下未来（用 noise init 的 future slots 走一遍 Video DiT），但只取这一个 step 的 hidden state 作为 planning signal。这比 multi-step roll-out 简单得多，也不需要 reward model。

参考：
- DreamerV3: https://arxiv.org/abs/2301.04104

### 7.3 与 Masked Autoencoders / JEPA 的联系

ForeWAM 的 future-slot 用 noise 而非 mask token 填充，这其实和 MAE / JEPA 的 masked prediction 有精神上的相似：
- MAE：mask 掉 patches，让 encoder 预测
- JEPA：在 latent space 预测 masked regions
- ForeWAM：future slots 用 noise 填充，Video DiT 通过 attention "填充"这些 slots 的 hidden representation

差别是 ForeWAM 不要求 reconstruct masked regions，只要求 hidden states 对 action 有用。这是一种 "predictive but not reconstructive" 的 latent modeling，与 LeCun 的 JEPA 哲学一致——不要在 pixel space 上预测，要在 abstract latent space 上预测。

参考：
- JEPA: https://arxiv.org/abs/2301.08243
- MAE: https://arxiv.org/abs/2111.06377

### 7.4 与 Chain-of-Thought / Latent Planning 的关系

最近有不少工作做 "latent CoT" —— 在 latent space 做多步 "thinking" 然后输出 answer。ForeWAM 的 future slots 可以看作是 **"latent planning tokens"**——它们承载了 network 对未来的 implicit reasoning。区别是 ForeWAM 只用 single forward pass（不像 latent CoT 通常需要 iterative refinement），更接近 "amortized planning"。

参考：
- Latent CoT (Pause tokens): https://arxiv.org/abs/2210.04469
- Coconut: https://arxiv.org/abs/2412.06769

---

## 8. Limitations 和可能的改进方向

论文自己提到的 limitation：只在 LIBERO 上 evaluate，没测 real robot 和 cross-embodiment。我觉得还有几个更深的 limitation 值得讨论：

1. **Future slots 的 stochasticity 没被利用**：Eq. 3 中 $\epsilon_F \sim \mathcal{N}(0, I)$ 是 sample 出来的，意味着每次推理的 K/V cache 都不同。但论文没有 sample multiple times 并 ensemble，也没有 study variance 的影响。理论上可以做 multi-sample averaging 来 reduce noise、提升 robustness。

2. **Dynamics registers 的数量是手调的**（$N_D = 16$）。这个数量对不同 task 的最优值可能不同。Long-horizon task 可能需要更多 registers 来编码 multiple sub-goals。

3. **Future slots 完全是 noise，没有任何 prior**。如果用 instruction-conditioned prior（比如先粗粒度预测 future 的 semantic content）来初始化 future slots，可能会更好。这有点像 "imagined rollout with weak prior"。

4. **Single forward pass 是否真的够？** 对于需要 multi-step planning 的任务（比如 Long suite 中的 10-step task），single prefill 可能不足以 capture 全部 dynamics。可以想象一个 iterative version：用 Action DiT 的 intermediate output refine future slots 的 noise，再 prefill 一次，形成 "alternating imagination-action" 循环。这就更接近真正的 mental simulation。

5. **LaWM teacher 的 bias**：teacher 是 inverse-dynamics model，它的 $z_{LA}$ 偏向于 "what action caused this transition"。但对于 passive observation（external agent 造成的 transition），teacher 的 representation 可能不是最优的。一个改进方向是同时用 forward-dynamics 和 inverse-dynamics 监督。

---

## 9. 直觉总结：ForeWAM 在做什么？

如果让我用一句话总结 ForeWAM 的 essence：

> **Policy 不需要看见未来，但 policy 的 attention 需要能 attend 到一个"想象的未来"的 hidden representation。**

更具体地：
1. Video DiT 是一个 "imagination engine"——给它当前帧 + 噪声 future + instruction，它在一次 forward 中产生 hidden states，这些 hidden states 编码了"未来可能如何演变"的 task-conditioned statistics。
2. K/V cache 是 "imagination 的可读副本"——把这些 hidden states 固化成 Action DiT 可以反复读取的 context。
3. Dynamics registers 是 "imagination 的摘要"——用 LaWM 监督让一部分 registers 专门编码 transition-relevant 信息。
4. Action DiT 是 "imagination-conditioned actor"——它在 denoising action chunk 时，每一步都读取 cached imagination，相当于在 "mental simulation" 的指导下选 action。

这非常像人类运动员的 "mental practice"——投篮前在脑中 simulate 一下整个动作序列，但不需要 fully visualize 每一个细节，只需要一个粗略的 "feel of the future" 就能指导实际动作。ForeWAM 把这个 cognitive mechanism 实现在了一个 end-to-end transformer 中。

---

## 10. 为什么这是一个 important paper

ForeWAM 在我看来 important 有几个原因：

1. **它提出了一个真正的 interface problem**：direct-policy WAM 砍掉 future generation 后，Action DiT 失去 future 接口这个问题之前没被清晰 articulate。Future-KV 是一个 clean、低成本、effective 的解决方案。

2. **它把 LLM 的 prefill-cache 思想迁移到 video-action model**：这种 cross-domain idea transfer 永远是 ML 进步的重要 driver。

3. **它证明 latent imagination 可以替代 pixel-space imagination**：这是朝着 "world model without rendering" 迈出的一步，对 real-time robot control 有 immediate practical value。

4. **2B params + no embodied pretraining + 220ms latency**：这些数字让这个方法在 real-world deployment 上 realistic，不只是 academic benchmark 数字。

5. **Latent action supervision 的设计**：用 frozen inverse-dynamics model 作为 transition supervisor，这是一个非常 general 的 technique，可以应用到任何想引入 transition-aware supervision 的场景。

希望这个分析能 build 你的 intuition, Andrej。如果你想深挖某个具体方面（比如 attention mask 的实现细节、OneDP distillation 的数学、或者 future substrate 的 alternative design），我可以再展开。
