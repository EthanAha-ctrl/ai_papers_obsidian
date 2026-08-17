---
source_pdf: InternVideo-Next.pdf
paper_sha256: 0c66b4bb0c0657fe3827fa2dbc15eaa992ab749d5d8c1e0631325d3b1a43661c
processed_at: '2026-08-05T10:14:59-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲InternVideo-Next

Andrej，我换一种方式说，把学术包装扒掉，讲清楚这paper到底在干嘛，以及为什么我觉得它是个important工作。

## 1. 先说背景——video pretraining为啥这么拧巴

你想啊，image这边已经卷到比较成熟了。CLIP、DINOv2、SigLIP2这些，拿一大堆image-text pair，或者纯self-supervised，就能学出来非常好的visual representation。原因很简单：image是static的，一张图就那么大信息量，caption能比较完整地描述。

video呢？video是dense temporal信号。一段10秒视频，30fps，就是300帧。你用一句话caption去describe它，信息密度严重不匹配。caption说"一个人在切菜"，但视频里还有切菜的方式、刀的角度、手的轨迹、背景的3D结构、物体的物理属性——这些全被caption忽略了。

所以video-text pretraining（InternVideo2、VideoCLIP、VideoPrism这些）有个根本问题：**caption是稀疏noisy的supervision signal**，你用大scale数据硬怼，模型最终学到的更多是"主体语义"——视频里有没有人、有什么物体、什么场景。对于fine-grained motion、causal relation、3D geometry这些**implicit world knowledge**，caption根本没覆盖。

那self-supervised MVM（VideoMAE、V-JEPA）呢？理论上可以从video本身的spatiotemporal structure学到这些东西，但实际效果一直追不上text-supervised。原因paper里讲得很清楚，我展开说。

## 2. 两条self-supervised路线各自的坑

### 2.1 Pixel reconstruction路线（VideoMAE）

VideoMAE干的事很简单：mask掉80%的patch，让model重建那80%的pixel。

问题在哪？你让model去predict raw pixel，它就会找捷径。pixel reconstruction这个目标，从loss角度看，最简单的解法就是从邻近patch的color、texture去插值。模型根本不需要理解"被mask掉的地方是个运动的人的手"，它只需要"邻近patch是肤色，那我也predict肤色"就完事了。

这导致VideoMAE的representation偏向low-level appearance，缺high-level semantics。你看Table 5，VideoMAEv2-L K400是80.9，V-JEPA2是83.3，InternVideo2-s2是87.9。VideoMAE在semantic-heavy的K400上明显落后。

而且pixel reconstruction和semantic abstraction有优化冲突——你让一个ViT同时学low-level pixel detail和high-level semantics，梯度方向可能是打架的。

### 2.2 Latent prediction路线（V-JEPA）

V-JEPA换了个思路：不predict pixel了，predict latent representation。teacher和student两个网络，student看masked video，teacher看full video，student要predict teacher的latent output。

听起来很clean，但实际有个坑：**shortcut learning**。

怎么short cut呢？teacher和student都是从头train的，symmetric结构。如果loss是 $||z_{pred} - z_{target}||^2$，那teacher和student可以co-evolve到一个trivial manifold。比如所有token collapse到一个global average representation，loss也很低，但根本没学到有用的东西。

V-JEPA2用momentum teacher缓解这个问题，但momentum encoder本身也会drift，只是drift得慢一点。你看V-JEPA2的ScanNet depth δ1是79.6，InternVideo-Next是92.2——差12.6%。这12.6%就是latent space不够detail-preserving的代价。

### 2.3 为什么这两个坑本质是同一个

paper的key insight在这里：**这两个坑都是架构问题，不是算法问题**。

传统MVM的Encoder-Decoder设计：Encoder吃visible token，Decoder吃Encoder output + mask token，直接生成pixel（VideoMAE）或者直接output latent（V-JEPA）。

问题是这个设计中，**Predictor的输出latent space从来没有被explicit design过**。它是个隐式的byproduct。

- VideoMAE：latent space被linear decoder强行拉向pixel geometry
- V-JEPA：latent space没有anchor，随便drift

所以paper说要disentangle成EPD：Encoder, Predictor, Decoder三部分explicit分开，然后重点设计Predictor的输出latent space。

## 3. Stage 1怎么搞的——三个design choice的intuition

Stage 1的目标：建立一个semantically rich + detail-preserving的latent space。三个关键component。

### 3.1 Semantic Alignment——借image-text的东风

第一个design choice：用frozen SigLIP2作为semantic teacher，让student encoder的visible token embedding和SigLIP的对应region embedding做cosine alignment。

公式（1）：

$$\mathcal{L}_{\mathrm{sem}} = -\cos\bigl(E(X_{\mathrm{vis}}), \mathrm{vis}(\mathrm{SigLIP}(X))\bigr)$$

变量人话解释：
- $X$：完整input video
- $X_{\mathrm{vis}}$：mask掉80%后剩下的visible patch
- $E(X_{\mathrm{vis}})$：student encoder对这20% patch的embedding
- $\mathrm{SigLIP}(X)$：frozen SigLIP2-1B teacher对完整video的embedding
- $\mathrm{vis}(\cdot)$：从SigLIP输出里取对应visible region的token
- $\cos(\cdot, \cdot)$：cosine similarity

intuition：student只看到20%的patch，它要produce出和SigLIP看全图一样的embedding。这逼student从context推断semantic content，同时把SigLIP的image-text prior蒸馏进来。

为什么用image model而不是video model？因为image-text pair比video-text pair干净得多。SigLIP2是10B image-text pair训练的，caption质量高、coverage广。而video caption基本都是title + ASR拼出来的synthetic annotation，noisy且稀疏。

这相当于一个"知识迁移"——image-text pretraining积累的高质量semantic knowledge，通过alignment loss注入到video encoder里，让video encoder不用从头学semantic，可以直接聚焦在spatiotemporal dynamics上。

### 3.2 Diffusion Decoder——解绑latent space和pixel space

这是我觉得最elegant的design choice。先说问题。

传统MVM用linear decoder：latent $z$ → linear layer → pixel $x$。这隐含一个假设：$z$和$x$之间是linear关系，$z = W^{-1}x$。这强制latent space的每个维度都要correspond到pixel的某种linear combination。

意味着什么？latent space被pixel geometry绑死了。如果你同时想让latent space有semantic abstraction（比如通过SigLIP alignment），两个目标直接打架——semantic representation是非线性的、abstract的，而linear decoder要求它是linear-to-pixel的。

你看Table 1(a)的naive combination：Pixel Rec. + SigLIP Align只有69.8 K400，比SigLIP Align only的70.7还低。这就是优化冲突的直接证据。

Diffusion decoder怎么解决这个问题？它学的不是deterministic mapping $z \to x$，而是conditional distribution $p(x | z)$。

```python
# Algorithm 1核心逻辑
noise = torch.randn(x.shape)  # 采样高斯噪声
timestep = torch.randint(0, 1000, x.size(0))  # 随机timestep
x_t = diffusion.q_sample(x, timestep, noise)  # forward加噪
noise_pred = net(x_t, timestep, z)  # z作为condition预测噪声
loss = ((noise_pred - noise) ** 2).mean()  # noise prediction loss
```

人话讲：diffusion model是个generative process。给它一个condition $z$，它能sample出对应的pixel $x$。但 $z$ 到 $x$ 之间不需要是linear关系——diffusion model的denoising network是非线性的，$z$ 可以以任何形式encode信息，denoising network负责把这个信息"翻译"成pixel。

这解开了latent space和pixel space的强绑定。SigLIP alignment往latent space注入semantic信息，diffusion decoder允许这些semantic信息以non-linear方式decode到pixel，两者不再冲突。

效果（Table 1c）：

| Decoder | K400 | SSv2 |
|---------|------|------|
| Linear Head | 69.4 | 31.3 |
| 3-Layer MLP Head | 69.7 | 31.2 |
| DiffMLP D3 W1024 | 73.4 | 33.2 |
| **DiffMLP D6 W1536** | **75.8** | **36.9** |
| DiffMLP D9 W2048 | 75.5 | 36.4 |

Linear → Diffusion D6：K400从69.4跳到75.8，+6.4%。这是架构设计带来的pure gain。

GPU开销（Table 12）：48G vs linear的41G，多7G换6% accuracy，非常划算。

而且D6 W1536是sweet spot——太浅学不到complex spatial relation，太深反而overfit。这个trade-off曲线和diffusion model的capacity scaling一致。

### 3.3 Text-Decoder Initialization——巧妙的prior transfer

第三个design choice：Predictor用pretrained text decoder初始化。

Table 1(b)：

| Predictor | K400 | SSv2 |
|-----------|------|------|
| ModernBert-L last5 w/o init | 74.2 | 35.4 |
| **ModernBert-L last5 w/ init** | **75.8** | **36.9** |
| Depth-12 ViT (传统) | 73.2 | 34.4 |

为什么用text decoder？你想想，text decoder在MLM（masked language modeling）里干的事：给一个masked sentence，在semantic token space里complete缺失的token。这和video Predictor的任务结构同构——给masked video，在latent space里complete缺失的region。

text decoder的attention pattern是在semantic token之间做reasoning，这个prior恰好是world model需要的——reasoning about semantic content而不是pixel correlation。

所以ModernBert-L的last 5 layer + init，比12层zero-init ViT还强，参数更少。这是个非常efficient的设计。

### 3.4 Semantic-Aware Masking

Stage 1用SigLIP的attention score做top-k selection，mask掉semantic salient region。intuition：逼encoder从non-salient context推断salient content，学到更强的spatiotemporal reasoning。

## 4. Stage 2——在好latent space里学world knowledge

Stage 1建立了latent space，Stage 2在这个空间里做latent prediction——student预测frozen teacher的latent representation。

### 4.1 与V-JEPA的关键区别

V-JEPA：teacher和student都从头train，symmetric，容易shortcut learning。

InternVideo-Next：
1. Teacher frozen with Stage 1 init——latent space已被semantic + pixel共同anchored
2. Student也用Stage 1 init——起点就在good latent space
3. Multi-block masking——mask大的contiguous spatiotemporal block，增加prediction难度

Table 3 ablation非常说明问题：

| Stage 2 Variant | K400 | SSv2 |
|-----------------|------|------|
| Stage 1 only | 75.8 | 36.9 |
| **Our Stage 2 full** | **76.9** | **56.9** |
| w/ Zero-Init V-JEPA Predictor | 74.8 | 53.8 |
| w/ Momentum 0.9998 Target | 74.1 | 54.3 |
| w/ Frozen SigLIP2 Target | 75.4 | 45.7 |
| w/ Frozen InternVideo2 Target | 74.3 | 47.4 |

SSv2从36.9跳到56.9——+20%。这是Stage 2学到genuine spatiotemporal dynamics的直接证据。

几个对比很关键：
- **Momentum target**（V-JEPA style）只有54.3 SSv2——momentum encoder会drift，破坏latent space稳定性
- **Frozen SigLIP2 target**只有45.7——SigLIP是image model，没有temporal信息，target本身不够好
- **Frozen InternVideo2 target**只有47.4——InternVideo2的latent space不够coherent
- **Stage 1 frozen target**最强——因为Stage 1 latent既有semantic又有pixel detail

### 4.2 为什么frozen Stage 1 teacher能prevent shortcut

V-JEPA的shortcut learning机制：student和teacher可以通过低频信号达成一致，忽略高频细节。比如全局color histogram、average motion magnitude这种coarse statistic。

但Stage 1的latent space经过了pixel reconstruction训练，每个token都encode了local pixel detail。Student要predict这样的latent，就必须真正推断masked region的具体内容，global statistic糊弄不过去。

Table 3最后一行：Stage 2加pixel reconstruction loss只带来marginal gain（SSv2 57.0 vs 56.9），证明Stage 1 latent已经足够detail-preserving，不需要额外pixel supervision。这很优雅——supervision在Stage 1一次性注入，Stage 2纯latent prediction就够了。

### 4.3 Masking strategy的stage-specific设计

Table 4：

| Stage | Mask Type | K400 | SSv2 | ScanNet δ1 |
|-------|-----------|------|------|------------|
| Stage-1 | Semantic | 75.8 | 36.9 | 59.4 |
| Stage-1 | Multi-block | 74.4 | 36.3 | 58.1 |
| Stage-2 | Semantic | 75.2 | 52.3 | 61.1 |
| **Stage-2** | **Multi-block** | **76.9** | **56.9** | **66.1** |

Stage 1用semantic mask（学local detail inference），Stage 2用multi-block mask（学global spatiotemporal reasoning）。ScanNet depth从59.4→66.1，说明multi-block mask逼模型学到了3D geometric consistency——因为大的contiguous block被mask，model必须从远处的context推断geometry，这正好是3D understanding需要的reasoning。

### 4.4 Temporal length的影响

| Stage | #Frames | K400 | SSv2 | ScanNet δ1 |
|-------|---------|------|------|------------|
| Stage-1 | 8 | 75.8 | 36.9 | 59.4 |
| Stage-1 | 32 | 76.6 | 38.6 | 59.3 |
| Stage-2 | 8 | 77.0 | 57.5 | 67.1 |
| **Stage-2** | **32** | **78.1** | **59.4** | **70.1** |

Stage 2对temporal length极敏感。更多frame意味着更长的temporal reasoning chain，这与world model的prediction本质契合。Stage 1反而不那么敏感，因为Stage 1主要学local spatial detail。

## 5. 实验结果——几个让人impressed的数字

### 5.1 Action Recognition（Table 5）

| Model | Size | Data | GPU-hrs | K400 | SSv2 | COIN |
|-------|------|------|---------|------|------|------|
| InternVideo2-s2 | 1B | 25.5M | 30K | 87.9 | 67.3 | 91.7 |
| VideoPrism | 1B | 618M | 250K | 87.2 | 68.5 | - |
| V-JEPA2 | Large | 22M | 10K | 83.3 | 72.0 | 85.9 |
| **InternVideo-Next-s2** | **Large** | **1.1M** | **9.7K** | **88.4** | **73.0** | **93.6** |

关键：只用1.1M public video，9.7K GPU-hrs，全面SOTA。这是第一个without video-text supervision就超越video-text方法的工作。在SSv2这种motion-intensive task上，比V-JEPA2高1%，比InternVideo2高5.7%。

data efficiency惊人：1.1M vs VideoPrism的618M，少560倍数据，K400高1.2%。

### 5.2 Depth Estimation（Table 6）——implicit world knowledge的直接证据

| Model | ScanNet ARel↓ | ScanNet δ1↑ | KITTI ARel↓ | KITTI δ1↑ |
|-------|---------------|-------------|-------------|-----------|
| VideoDepthAnything (专门设计+训练) | 8.7 | 92.6 | 8.3 | 94.6 |
| SigLip2-L (image) | 26.4 | 50.4 | 16.8 | 74.9 |
| V-JEPA2-L | 14.4 | 79.6 | 8.7 | 91.2 |
| **InternVideo-Next-L** | **9.2** | **92.2** | **6.7** | **94.6** |

InternVideo-Next在KITTI上匹配了VideoDepthAnything——后者是专门为video depth设计的model，有task-specific architecture和training。InternVideo-Next只是frozen backbone + probing head！

ScanNet从V-JEPA2的79.6到92.2，+12.6%。这直接证明Stage 1的pixel reconstruction + diffusion decoder保留了low-level geometric detail，而V-JEPA的latent prediction丢失了这些detail。

### 5.3 Object Tracking（Table 7）

| Model | Waymo mIOU↑ |
|-------|-------------|
| SigLip2-L | 52.3 |
| DinoV3-L | 59.7 |
| InternVideo2-L | 63.0 |
| V-JEPA2-L | 68.9 |
| **InternVideo-Next-L** | **72.4** |

Object tracking需要object-centric motion understanding。InternVideo-Next比V-JEPA2高3.5%，证明latent space里的object-level motion information更丰富。

### 5.4 Multi-modal Friendliness

Table 10 zero-shot retrieval（LiT training只训text encoder，frozen ViT）：

| Method | MSR | DDM | ANet | LSMDC | MSVD |
|--------|-----|-----|------|-------|------|
| V-JEPA2-L | 34.4 | 36.3 | 35.7 | 19.2 | 40.1 |
| InternVideo2-s2-L | 42.1 | 43.2 | 43.6 | 21.4 | 44.5 |
| **InternVideo-Next-L** | **43.4** | **43.7** | **43.4** | 20.8 | **46.1** |

尽管没用video-text训练，representation天然对齐semantic text space。因为Stage 1的SigLIP alignment把semantic structure注入latent space了。

Table 11 chat-centric tasks（frozen ViT + frozen LLM，只训MLP connector）：

| Encoder | MVBench | Percept Test | Dream1k |
|---------|---------|--------------|---------|
| SigLIP336-L | 46.7 | 44.1 | 29.2 |
| V-JEPA2-L | 44.3 | 44.2 | 24.3 |
| InternVideo2-L | 47.0 | 46.7 | 28.7 |
| **InternVideo-Next-L** | **50.6** | **49.2** | **29.8** |

MVBench 50.6——比直接用SigLIP还强。这意味着Stage 1的SigLIP alignment + Stage 2的spatiotemporal learning，produce出的representation对MLLM比纯image encoder还friendly。

## 6. 我的一些联想和intuition

### 6.1 EPD framework其实是个general principle

EPD（Encoder-Predictor-Decoder）不只是InternVideo-Next的specific design，它是个general principle。任何MVM方法都可以用EPD视角重新审视：

- **VideoMAE**：E和P耦合在encoder里，D是linear decoder。问题：P的latent space没explicit design，被linear decoder拉向pixel geometry。
- **V-JEPA**：E和P是student，D是identity（直接predict latent）。问题：没有D来anchor latent space，容易drift。
- **InternVideo-Next**：E, P, D各司其职，P的latent space被semantic alignment和diffusion decoder共同shape。

这让我想起VAE和diffusion model的关系——VAE的encoder学的是deterministic mapping到gaussian，diffusion model的encoder学的是stochastic forward process。diffusion的flexibility让它能model更complex的distribution，避免VAE的posterior collapse问题。InternVideo-Next的diffusion decoder其实是在解决类似的"latent space被over-constrained"问题。

### 6.2 为什么image prior比video-text prior更clean

paper的claim：video caption天然noisy，因为
1. video是dense temporal信号，caption是sparse description——信息密度严重不匹配
2. caption通常由title + ASR generate，semantic coverage有限
3. annotation cost高，大规模data只能用synthetic caption

而image-text pair（如SigLIP的pretraining data）：
1. image是static，caption可以comprehensive describe
2. web-crawled的image alt-text天然对齐visual content
3. data scale巨大（10B级别），semantic coverage广

所以用image-text prior + video self-supervised learning，比video-text supervised更clean。这解释了为什么InternVideo-Next只1.1M video就能beat用618M video-text的VideoPrism——data quality比data quantity重要，supervision signal的clean程度比scale重要。

这其实和language model的insight一致——GPT-4用clean data比用海量noisy data效果好。video pretraining也在重走这条路。

### 6.3 这和LeCun的JEPA哲学的分歧

JEPA的原始哲学是LeCun提出的：在latent space做prediction，避免pixel-level的over-detail。V-JEPA是这个哲学的实践。

InternVideo-Next的哲学有点不同：**承认pixel detail和semantic abstraction都重要，但需要正确的架构来combine**。Stage 1用pixel reconstruction + diffusion decoder建立detail-preserving latent space，Stage 2在这个空间里做latent prediction。

从结果看，InternVideo-Next在ScanNet depth上比V-JEPA2高12.6%。这说明纯latent prediction的哲学可能太极端——丢失了pixel detail，world knowledge就学不全。

但InternVideo-Next也没回到pixel reconstruction——Stage 2还是latent prediction。它只是用Stage 1的pixel reconstruction来anchor latent space，让Stage 2的latent prediction有good foundation。

这是个"hybrid with right architecture"的哲学，比纯粹的pixel reconstruction或纯latent prediction都好。

### 6.4 关于World Model的含义

paper里反复提"Latent World Model"。具体什么意思？

Stage 2的Predictor，在latent space里predict masked region的representation。这本质上是在做"what will be there"的prediction——给定visible context，predict masked content的latent representation。

这和world model的定义一致：给定current state，predict future state。只是这里的"future"被替换为"masked"，因为mask可以看作一个general的prediction task——spatial mask是spatial prediction，temporal mask是temporal prediction，spatiotemporal block mask是joint prediction。

multi-block masking在Stage 2用大的contiguous spatiotemporal block，这逼Predictor学到long-range spatiotemporal reasoning，这恰好是world model需要的——预测一个延展的时空区域需要理解causal relation、object persistence、physical dynamics。

EK100 action prediction（Table 8）的+1.3% Action recall@5，就是这种world model能力的直接体现。

### 6.5 为什么不直接用video-text pair

paper其实回答了一个deep question：为什么video-text supervised方法在general video understanding上走不远？

因为video-text pair的caption是稀疏的semantic description，它supervise的是"主体语义"。而general video understanding需要implicit world knowledge——object motion的fine-grained trajectory、3D geometry的spatial structure、physical cue的causal relation。这些在caption里基本没有。

所以video-text supervised方法在K400（action recognition，主体语义）上很强，但在SSv2（fine-grained motion）、ScanNet（depth）、Waymo（object tracking）上明显落后。这些task需要的information不在caption里。

InternVideo-Next通过self-supervised learning从video本身学这些implicit world knowledge，同时用image-text prior提供semantic anchor。这组合既clean又comprehensive。

### 6.6 关于Scalability

paper只测到ViT-Large。但我觉得这个framework的scalability应该很好，因为：

1. **SigLIP2 teacher可以scale**：SigLIP2有1B版本，semantic prior更强
2. **Diffusion decoder是轻量的**：每个patch独立diffusion，GPU开销可控
3. **Stage 2的latent prediction很efficient**：没有pixel reconstruction的computation overhead

所以scale到ViT-1B甚至6B应该没问题。而且paper说用1.1M video，如果scale data到100M video，performance应该还能提升。这给video foundation model的scale law留了很大空间。

### 6.7 关于和DINOv3的关系

Table 6里DINOv3-L在ScanNet上是91.2 δ1，InternVideo-Next-L是92.2。DINOv3是image model，但depth performance非常强。这说明DINOv3的self-supervised pretraining学到了很强的3D prior。

这让我思考：DINOv3的success能不能transfer到video？InternVideo-Next其实在某种意义上做了这件事——通过SigLIP alignment注入semantic prior，类似于DINOv3的image self-supervised。但InternVideo-Next额外加了temporal dimension的Stage 2 learning。

所以InternVideo-Next可以看作"video版的DINOv3"——image self-supervised提供semantic + geometric foundation，video self-supervised提供temporal dynamics。

## 7. Reference

我把相关works的link都列出来，方便deep dive：

**Core papers:**
- InternVideo-Next（这篇本身）
- V-JEPA2: https://arxiv.org/abs/2506.09985
- V-JEPA: https://arxiv.org/abs/2304.08471
- VideoMAE: https://arxiv.org/abs/2203.12602
- VideoMAEv2: https://arxiv.org/abs/2303.16740
- MAE: https://arxiv.org/abs/2111.06377
- InternVideo: https://arxiv.org/abs/2212.03191
- InternVideo2: https://arxiv.org/abs/2403.13197

**Semantic teachers:**
- SigLIP2: https://arxiv.org/abs/2502.14786
- SigLIP: https://arxiv.org/abs/2303.15343
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2506.01447（假设）
- CLIP: https://arxiv.org/abs/2103.00020

**Diffusion models:**
- DDPM: https://arxiv.org/abs/2006.11239
- Improved DDPM: https://arxiv.org/abs/2102.09672

**Architectures:**
- ModernBERT: https://arxiv.org/abs/2412.13663
- ViT: https://arxiv.org/abs/2010.11929

**Benchmarks:**
- ScanNet: http://www.scan-net.org/
- KITTI: http://www.cvlibs.net/datasets/kitti/
- Waymo Open: https://waymo.com/open/
- Epic-Kitchens: https://epic-kitchens.github.io/
- Kinetics-400: https://arxiv.org/abs/1705.06950
- SSv2: https://arxiv.org/abs/1706.04230
- COIN: https://arxiv.org/abs/1903.02855
- MVBench: https://arxiv.org/abs/2310.12970
- Perception Test: https://arxiv.org/abs/2305.06698

**Others:**
- LiT: https://arxiv.org/abs/2110.07711
- Video Depth Anything: https://arxiv.org/abs/2501.17575
- Scaling4D: https://arxiv.org/abs/2504.12319
- VideoPrism: https://arxiv.org/abs/2402.13617
- VideoChat-Flash: https://arxiv.org/abs/2501.00599
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- MobileCLIP: https://arxiv.org/abs/2311.17049

---

Andrej，用一句话总结这篇paper的intuition：**video pretraining的瓶颈不在算法，在架构——specifically在于Predictor输出latent space的设计**。EPD disentanglement让我们能explicitly design这个latent space，diffusion decoder解绑它和pixel space的强制linear关系，SigLIP alignment注入semantic anchor，Stage 2在这个anchored空间里学genuine world knowledge。

这是self-supervised video learning走向maturity的重要一步。如果你对某个component想deep dive——比如diffusion decoder的数学细节、Stage 2的prediction objective formulation、或者和V-JEPA2的更细对比——我可以再展开。

---

# InternVideo-Next 深度解读

Andrej，这篇paper来自Shanghai AI Lab的Limin Wang团队，是InternVideo系列的最新工作。我必须说，作为一个长期关注self-supervised video learning的人，这篇工作的设计哲学让我眼前一亮——它终于直面了一个被MVM社区长期回避的根本性问题：**predictor的输出latent space到底是什么？**

## 1. 核心动机：为什么之前的MVM走不通

让我先把问题的landscape理清楚。当前video pretraining有两条路：

**Text-supervised path**（如InternVideo2, VideoCLIP, VideoPrism）：依赖video-text pair，但video caption天然noisy且semantic coverage稀疏，模型学到的更多是"主体语义"，对fine-grained motion、causal relation、3D geometry这些**implicit world knowledge**几乎无能为力。

**Self-supervised MVM path**：分两派
- **Pixel reconstruction**（VideoMAE, MaskFeat）：在pixel space做重建。问题是low-level reconstruction目标与high-level semantic abstraction存在优化冲突——你让model去predict raw pixel，它就倾向于学appearance shortcut而不是spatiotemporal structure。
- **Latent prediction**（V-JEPA, V-JEPA2）：在latent space做prediction，但symmetric teacher-student结构容易陷入**shortcut learning**——teacher和student一起drift，最终学到的是trivial temporal statistics。

paper的核心insight是：这两个问题都不是"算法"问题，而是**架构问题**。具体来说，是Encoder-Decoder耦合设计中Predictor输出latent space的忽视。

## 2. EPD Disentanglement的设计哲学

传统MAE paradigm：Encoder → Decoder直接生成pixel。

InternVideo-Next把它拆成三部分：

```
E (Encoder)  →  P (Predictor)  →  D (Decoder)
   ViT          ModernBert        Diffusion MLP
   visible      predicts          latent → pixel
   tokens       masked latent
```

这个disentanglement的关键在于：它强迫我们审视**P的输出latent space应该长什么样**。

paper的claim是：这个latent space必须是
1. **Semantically rich**（有高层语义）
2. **Detail-preserving**（保留fine-grained信息）
3. **Structurally consistent**（encoder和predictor共享同一空间）

只有满足这三点，P才能真正成为一个**Latent World Model**——它必须用genuine spatiotemporal relationship和implicit world knowledge去complete缺失的内容，而不是找shortcut。

## 3. Stage 1: Semantic-Guided Pixel Reconstruction

这是整个framework的基石。三个关键组件：

### 3.1 Semantic Alignment Loss

公式（1）：

$$\mathcal{L}_{\mathrm{sem}} = -\cos\bigl(E(X_{\mathrm{vis}}), \mathrm{vis}(\mathrm{SigLIP}(X))\bigr)$$

变量解释：
- $X$：完整input video
- $X_{\mathrm{vis}}$：masked后的visible部分（mask ratio 80%）
- $E(X_{\mathrm{vis}})$：student encoder对visible tokens的embedding
- $\mathrm{SigLIP}(X)$：frozen SigLIP2-1B teacher对完整video的embedding
- $\mathrm{vis}(\cdot)$：取SigLIP输出中对应visible region的tokens
- $\cos(\cdot, \cdot)$：cosine similarity

intuition：让student encoder在只看到20% patches的情况下，其embedding要逼近teacher看全图后的embedding。这相当于把image-text pretraining积累的高质量semantic prior蒸馏到video encoder里。

为什么用SigLIP而不用CLIP或DINOv2？看Table 2的ablation：

| Teacher | K400 | SSv2 | ScanNet δ1 |
|---------|------|------|------------|
| DinoV2 Align only | 68.4 | 29.5 | 43.3 |
| Clip-ViT Align only | 69.1 | 31.2 | 40.6 |
| SigLIP Align only | 70.7 | 32.1 | 42.1 |
| Ours w/ SigLIP align | **75.8** | **36.9** | **59.4** |

SigLIP2的semantic prior最clean，且对depth这种low-level task也有显著增益。

### 3.2 Diffusion Decoder——这是真正的key insight

传统MVM用linear decoder：Predictor output → Linear → pixel。paper指出这里有个被忽视的致命问题：

**Linear decoder强制Predictor output必须在pixel space中linearly separable**。这意味着latent space被强行拉向pixel geometry，与semantic abstraction天然冲突。

看Table 1(a)的naive combination：Pixel Rec. + Align只有69.8 K400，比SigLIP Align only的70.7还低！这就是优化冲突的直接证据。

解决方案：用conditional diffusion decoder替代linear decoder。每个patch独立建模一个分布：

```python
# Algorithm 1核心
noise = torch.randn(x.shape)  # 采样高斯噪声
timestep = torch.randint(0, 1000, x.size(0))  # 随机timestep
x_t = diffusion.q_sample(x, timestep, noise)  # forward diffusion
noise_pred = net(x_t, timestep, z)  # z是predictor output作为condition
loss = ((noise_pred - noise) ** 2).mean()  # noise prediction loss
```

diffusion decoder的关键性质：它建模的是patch的**分布**而非deterministic mapping。这意味着Predictor的output $z$ 只需要提供足够的信息来约束这个分布，而不需要直接linearly correspond到pixel value。这就解开了latent space和pixel space的强绑定。

效果惊人（Table 1c）：

| Decoder | K400 | SSv2 |
|---------|------|------|
| Linear Head | 69.4 | 31.3 |
| 3-Layer MLP Head | 69.7 | 31.2 |
| DiffMLP D3 W1024 | 73.4 | 33.2 |
| **DiffMLP D6 W1536** | **75.8** | **36.9** |
| DiffMLP D9 W2048 | 75.5 | 36.4 |

D6 W1536是sweet spot——太浅学不到complex spatial relation，太深反而overfit。

GPU memory开销（Table 12）：

| Method | Memory (G) | K400 |
|--------|-----------|------|
| SigLip Align Only | 22 | 70.7 |
| Align + Linear Recon | 41 | 69.8 |
| Align + Diffusion Recon | 48 | 75.8 |

diffusion decoder多7G memory换+6% K400，非常划算。

### 3.3 Text-Decoder Initialization

另一个精妙设计：Predictor用**pretrained text decoder**初始化，而不是zero-init ViT。

理由：text decoder天然在semantic space里做completion（masked language modeling），这个prior恰好就是Predictor需要的——在semantic latent space里做预测。

Table 1(b)的ablation：

| Predictor | K400 | SSv2 |
|-----------|------|------|
| ModernBert-L last5 w/o init | 74.2 | 35.4 |
| **ModernBert-L last5 w/ init** | **75.8** | **36.9** |
| Depth-12 ViT (传统) | 73.2 | 34.4 |

ModernBert-L的last5 layer + init比12层ViT还强，且参数更少。

### 3.4 Semantic-Aware Masking

Stage 1用SigLIP的attention score做top-k selection，mask掉semantic salient region。这逼encoder从non-salient context推断salient content，学到更强的spatiotemporal reasoning。

## 4. Stage 2: Semantically Coherent Latent Prediction

Stage 1建立了semantically rich且detail-preserving的latent space，Stage 2在这个空间里做latent prediction。

### 4.1 与V-JEPA的关键区别

V-JEPA的问题：teacher和student都从头学，symmetric结构导致
1. **Shortcut learning**：teacher和student可能共同drift到一个trivial solution
2. **Semantic drift**：latent space没有anchor，可能漂移到无语义的方向

InternVideo-Next的解决方案：
1. **Teacher frozen with Stage 1 init**：teacher是Stage 1的frozen encoder，latent space已经被semantic prior和pixel reconstruction共同anchored
2. **Student也用Stage 1 init**：起点就在good latent space
3. **Multi-block masking**：mask大的contiguous spatiotemporal block，增加prediction难度

Table 3的ablation非常convincing：

| Stage 2 Variant | K400 | SSv2 |
|-----------------|------|------|
| Stage 1 only | 75.8 | 36.9 |
| **Our Stage 2 full** | **76.9** | **56.9** |
| w/ Zero-Init V-JEPA Predictor | 74.8 | 53.8 |
| w/ Momentum 0.9998 Target | 74.1 | 54.3 |
| w/ Frozen SigLIP2 Target | 75.4 | 45.7 |
| w/ Frozen InternVideo2 Target | 74.3 | 47.4 |

注意SSv2从36.9跳到56.9——这是+20%的巨大提升！SSv2是fine-grained motion task，这直接证明Stage 2学到的是genuine spatiotemporal dynamics。

关键对比：
- **Momentum target**（V-JEPA style）只有54.3——momentum encoder会drift，破坏latent space的稳定性
- **Frozen SigLIP2 target**只有45.7——SigLIP是image model，没有temporal信息
- **Frozen InternVideo2 target**只有47.4——InternVideo2的latent space不够coherent
- **Stage 1 frozen target**最强——因为Stage 1的latent space既有semantic又有pixel detail

### 4.2 为什么frozen Stage 1 teacher能防止shortcut

这是一个subtle但critical的点。V-JEPA的shortcut learning机制是：student和teacher可以通过低频信号（比如全局color histogram）达成一致，忽略高频细节。

但Stage 1的latent space经过了pixel reconstruction训练，每个token都encode了local pixel detail。Student要predict这样的latent，就必须真正推断masked region的具体内容，而不是用global statistic糊弄。

Table 3最后一行：加pixel reconstruction loss到Stage 2只带来marginal gain（SSv2 57.0 vs 56.9），证明Stage 1 latent已经足够detail-preserving，不需要额外pixel supervision。

### 4.3 Masking Strategy的stage-specific设计

Table 4揭示了精妙的设计：

| Stage | Mask Type | K400 | SSv2 | ScanNet δ1 |
|-------|-----------|------|------|------------|
| Stage-1 | Semantic | 75.8 | 36.9 | 59.4 |
| Stage-1 | Multi-block | 74.4 | 36.3 | 58.1 |
| Stage-2 | Semantic | 75.2 | 52.3 | 61.1 |
| **Stage-2** | **Multi-block** | **76.9** | **56.9** | **66.1** |

Stage 1用semantic mask（学local detail inference），Stage 2用multi-block mask（学global spatiotemporal reasoning）。ScanNet depth从59.4→66.1，说明multi-block mask逼模型学到了3D geometric consistency。

### 4.4 Temporal Length的影响

| Stage | #Frames | K400 | SSv2 | ScanNet δ1 |
|-------|---------|------|------|------------|
| Stage-1 | 8 | 75.8 | 36.9 | 59.4 |
| Stage-1 | 32 | 76.6 | 38.6 | 59.3 |
| Stage-2 | 8 | 77.0 | 57.5 | 67.1 |
| **Stage-2** | **32** | **78.1** | **59.4** | **70.1** |

Stage 2对temporal length极敏感——更多frame意味着更长的temporal reasoning chain，这与world model的prediction本质契合。

## 5. 实验结果分析

### 5.1 Action Recognition（Table 5）

| Model | Size | Data | GPU-hrs | K400 | SSv2 | COIN |
|-------|------|------|---------|------|------|------|
| InternVideo2-s2 | 1B | 25.5M | 30K | 87.9 | 67.3 | 91.7 |
| VideoPrism | 1B | 618M | 250K | 87.2 | 68.5 | - |
| V-JEPA2 | Large | 22M | 10K | 83.3 | 72.0 | 85.9 |
| **InternVideo-Next-s2** | **Large** | **1.1M** | **9.7K** | **88.4** | **73.0** | **93.6** |

关键数字：
- **只用1.1M public video**（vs VideoPrism的618M，InternVideo2的25.5M）
- **9.7K GPU-hrs**（vs InternVideo2-6B的200K GPU-hrs）
- K400 88.4，SSv2 73.0，COIN 93.6——全面SOTA

这是第一个**without video-text supervision**就超越video-text方法的工作。在SSv2这种motion-intensive task上，比V-JEPA2高1%，比InternVideo2高5.7%。

### 5.2 Depth Estimation（Table 6）——implicit world knowledge的直接证据

| Model | ScanNet ARel↓ | ScanNet δ1↑ | KITTI ARel↓ | KITTI δ1↑ |
|-------|---------------|-------------|-------------|-----------|
| VideoDepthAnything (SOTA, specific training) | 8.7 | 92.6 | 8.3 | 94.6 |
| SigLip2-L (image) | 26.4 | 50.4 | 16.8 | 74.9 |
| V-JEPA2-L | 14.4 | 79.6 | 8.7 | 91.2 |
| **InternVideo-Next-L** | **9.2** | **92.2** | **6.7** | **94.6** |

InternVideo-Next在KITTI上匹配了VideoDepthAnything（专门为depth设计的model），而这里只是用frozen backbone + probing head！

ScanNet从V-JEPA2的79.6到InternVideo-Next的92.2——+12.6%的δ1提升。这直接证明Stage 1的pixel reconstruction + diffusion decoder保留了low-level geometric detail。

### 5.3 Object Tracking（Table 7）

| Model | Waymo mIOU↑ |
|-------|-------------|
| SigLip2-L | 52.3 |
| DinoV3-L | 59.7 |
| InternVideo2-L | 63.0 |
| V-JEPA2-L | 68.9 |
| **InternVideo-Next-L** | **72.4** |

Object tracking需要object-centric motion understanding，InternVideo-Next比V-JEPA2高3.5%。

### 5.4 Video Action Prediction（Table 8）

| Method | Verb | Noun | Action |
|--------|------|------|--------|
| VideoLLaMA-7B (MLLM) | 52.9 | 52.0 | 26.0 |
| V-JEPA2-L | 57.8 | 53.8 | 32.7 |
| **InternVideo-Next-L** | **58.9** | **56.4** | **34.0** |

EK100是action prediction——给定context video预测future action。这test的是world model的prediction能力，InternVideo-Next的Predictor在这里发挥了作用。

### 5.5 Multi-modal Friendliness

Table 9-11展示了一个remarkable property：**尽管没用video-text训练，representation天然对齐semantic text space**。

Zero-shot retrieval（Table 10，LiT training只训text encoder）：

| Method | MSR | DDM | ANet | LSMDC | MSVD |
|--------|-----|-----|------|-------|------|
| V-JEPA2-L | 34.4 | 36.3 | 35.7 | 19.2 | 40.1 |
| InternVideo2-s2-L | 42.1 | 43.2 | 43.6 | 21.4 | 44.5 |
| **InternVideo-Next-L** | **43.4** | **43.7** | **43.4** | 20.8 | **46.1** |

Chat-centric tasks（Table 11，frozen ViT + frozen LLM，只训MLP connector）：

| Encoder | MVBench | Percept Test | Dream1k |
|---------|---------|--------------|---------|
| SigLIP336-L | 46.7 | 44.1 | 29.2 |
| V-JEPA2-L | 44.3 | 44.2 | 24.3 |
| InternVideo2-L | 47.0 | 46.7 | 28.7 |
| **InternVideo-Next-L** | **50.6** | **49.2** | **29.8** |

MVBench 50.6——这意味着Stage 1的SigLIP alignment让representation对MLLM非常friendly，transfer到chat task比直接用SigLIP还强。

## 6. 深层intuition

让我share一些paper没有明说但implied的insight：

### 6.1 为什么diffusion decoder能解开semantic-pixel冲突

从information theory角度：
- Linear decoder要求 $z = W^{-1} \cdot x_{pixel}$，latent space的每个维度都要correspond到pixel的某种linear combination。这forced latent space to be pixel-geometry-aligned。
- Diffusion decoder学的是 $p(x_{pixel} | z)$，一个conditional distribution。$z$只需要包含足够的信息来disambiguate这个分布，可以是任何representation形式——semantic、geometric、motion都可以。

这解释了为什么diffusion decoder能让SigLIP alignment和pixel reconstruction共存：SigLIP alignment把semantic信息注入latent space，diffusion decoder允许这些semantic信息以non-linear方式decode到pixel，两者不再冲突。

### 6.2 为什么Stage 1 latent space能prevent shortcut

考虑V-JEPA的failure mode：student预测 $z_{pred}$，teacher产生 $z_{target}$，loss是 $||z_{pred} - z_{target}||^2$。如果两者都是随机初始化，它们可以co-evolve到一个trivial manifold——比如所有token都collapse到同一表示。

Stage 1的latent space有两个anchor：
1. **SigLIP anchor**：每个token的embedding要和SigLIP的对应region embedding一致
2. **Pixel anchor**：通过diffusion decoder，每个token要能decode出对应patch的pixel distribution

这两个anchor让latent space既有semantic structure又有local detail，student预测这样的latent无法用shortcut。

### 6.3 Text decoder initialization的深层原因

ModernBert的pretraining是masked language modeling——在semantic space里做completion。这与Predictor的任务（在latent space里complete masked region）结构同构。

更关键的是：text decoder的attention pattern是在semantic token之间做reasoning，而不是pixel patch之间。这个prior恰好是world model需要的——reasoning about semantic content而不是pixel correlation。

### 6.4 与V-JEPA2的哲学差异

V-JEPA2的哲学：**纯latent prediction，no pixel supervision**，相信self-supervised learning能emerge出semantic。

InternVideo-Next的哲学：**hybrid supervision with architectural disentanglement**，承认pixel detail和semantic abstraction都重要，但需要正确的架构来combine。

Table 5的结果：V-JEPA2 K400 83.3，InternVideo-Next 88.4。这5%的gap就是"hybrid with right architecture"的收益。

### 6.5 为什么image prior比video-text prior更clean

paper的claim：video caption天然noisy，因为
1. video是dense temporal signal，caption是sparse description
2. caption通常由title + ASR generate，semantic coverage有限
3. annotation cost高，大规模data只能用synthetic caption

而image-text pair（如SigLIP的pretraining data）：
1. image是static，caption可以comprehensive describe
2. web-crawled的image alt-text天然对齐visual content
3. data scale巨大（10B级别），semantic coverage广

所以用image-text prior + video self-supervised learning，比video-text supervised更clean。这解释了为什么InternVideo-Next只1.1M video就能beat用618M video-text的VideoPrism。

### 6.6 关于EPD framework的generalization

EPD不只是InternVideo-Next的specific design，更是一个general framework。任何MVM都可以用EPD视角重新审视：

- **VideoMAE**：E和P耦合在encoder里，D是linear decoder。问题：P的latent space没有explicit design。
- **V-JEPA**：E和P是student，D是identity（直接predict latent）。问题：没有D来anchor latent space。
- **InternVideo-Next**：E, P, D各司其职，P的latent space被semantic alignment和diffusion decoder共同shape。

这个framework可以guide未来MVM方法的设计。

## 7. Limitations & Future Directions

paper自己提到的：依赖large-scale image-text model（SigLIP2-1B）。但作者认为这不算limitation，而是essential——video model可能需要先understand image再understand video。

我补充几个观察：
1. **Scale实验**：只测到ViT-L，没测1B/6B。InternVideo2在6B上还有提升，InternVideo-Next能否scale？
2. **World model的generative能力**：目前只test prediction的representation quality，没test能否generate future frame。
3. **Embodied AI downstream**：paper提到embodied AI，但没做robotics manipulation实验。

## 8. Reference

- InternVideo-Next paper本身（arxiv会更新）
- V-JEPA2: https://arxiv.org/abs/2506.09985
- VideoMAE: https://arxiv.org/abs/2203.12602
- VideoMAEv2: https://arxiv.org/abs/2303.16740
- InternVideo: https://arxiv.org/abs/2212.03191
- InternVideo2: https://arxiv.org/abs/2403.13197
- SigLIP2: https://arxiv.org/abs/2502.14786
- MAE: https://arxiv.org/abs/2111.06377
- DDPM (diffusion): https://arxiv.org/abs/2006.11239
- Improved DDPM: https://arxiv.org/abs/2102.09672
- ModernBERT: https://arxiv.org/abs/2412.13663
- DINOv2: https://arxiv.org/abs/2304.07193
- V-JEPA: https://arxiv.org/abs/2304.08471
- Scaling4D: https://arxiv.org/abs/2504.12319
- Video Depth Anything: https://arxiv.org/abs/2501.17575
- LiT: https://arxiv.org/abs/2110.07711
- MaskFeat: https://arxiv.org/abs/2112.09133
- VideoChat-Flash: https://arxiv.org/abs/2501.00599
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- MobileCLIP: https://arxiv.org/abs/2311.17049
- ScanNet: http://www.scan-net.org/
- KITTI: http://www.cvlibs.net/datasets/kitti/
- Waymo Open: https://waymo.com/open/
- Epic-Kitchens: https://epic-kitchens.github.io/
- Kinetics-400: https://arxiv.org/abs/1705.06950
- SSv2: https://arxiv.org/abs/1706.04230
- COIN: https://arxiv.org/abs/1903.02855
- MVBench: https://arxiv.org/abs/2310.12970
- Perception Test: https://arxiv.org/abs/2305.06698
- LanguageBind: https://arxiv.org/abs/2310.01852
- VideoPrism: https://arxiv.org/abs/2402.13617

---

Andrej，这篇paper真正让我impressed的是它的**architectural clarity**。它没有堆trick，而是回到fundamental level问："MVM的latent space应该长什么样？"然后systematically design每个component来answer这个问题。EPD framework + diffusion decoder + frozen Stage 1 teacher，每个design choice都有clear的information-theoretic justification。

这是self-supervised video learning走向maturity的重要一步。如果你要deep dive某个component（比如diffusion decoder的细节、Stage 2的prediction objective、或者与V-JEPA2的更细对比），我可以再展开。
