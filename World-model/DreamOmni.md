---
source_pdf: DreamOmni.pdf
paper_sha256: ecbf689cc91e29f660e914ea598b043725b119839d970e34721a425cce3ad6c0
processed_at: '2026-08-03T23:29:57-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DreamOmni 人话版

Andrej，我把 paper 里那些包装过的学术语言全部拆掉，用工程师能听懂的方式重新讲一遍。

## 1. 这帮人到底想干什么

现在的 T2I 模型（SDXL, SD3, FLUX）都有一个共同毛病：**训练的时候根本没想过以后要被拿去 editing**。

结果就是你想让它做 editing，只能各种"打补丁"：
- 想做 inpainting，给 Unet 加几个 input channel（SD-inpainting）
- 想做 instruction editing "把猫换成狗"，又扩几个 channel（InstructP2P）
- 想做 ControlNet 那种 canny/depth 条件，加个外挂插件
- 想做 subject-driven "用这张脸生成新图"，又加 IP-Adapter

每个任务一套接法，互相不兼容。你想 joint train 一个能干所有事的模型，框架本身就不支持。

DreamOmni 想做的就是：**重新设计一个框架，从 day one 就把所有这些任务当作同一个东西来处理**。这是 native unified model 的核心思路，对标 LLM 那种"一个模型啥都能干"的哲学。

## 2. 难点在哪：数据，不是架构

架构其实不太难想——把所有条件都 token 化，喂进 transformer 就完了。真正卡脖子的是 **editing 数据**。

举例：你想训 instruction editing "把图里的红球换成蓝球"。你需要一张 source 图（有红球）和一张 target 图（同位置变蓝球），两张图除球颜色外完全一样。这种 pair 数据怎么做？

- **GPT-3 + Prompt-to-Prompt**（InstructP2P 的做法）：让模型自己生成 pair。成功率 <15%，剩下 85% 要么对不上，要么 target 整张图都变了。还得人工 filter。
- **人工标注**（MagicBrush 的做法）：让人来画。质量好但 1 张图要几分钟，scale 不起来。
- **视频抽帧**（InstaDrag 的做法）：从视频里抽相邻帧做 drag。但视频里大部分帧根本不动，filter 也很麻烦。

所有现有方法的瓶颈都在 **"如何高效生成 pixel-level 精确对应的 source/target pair"**。

## 3. 关键 Insight：editing 失败不是因为模型不认识 cat

作者有一个特别 sharp 的观察，这是整篇 paper 的灵魂。

你以为 instruction editing 做不好，是因为模型不知道 "cat" 长什么样？**完全不是**。T2I 训练已经让模型能生成各种 cat 了。

真正失败的原因是 **模型不知道 "remove the cat" 这个 instruction 意味着什么操作**。它可能把整个背景都改了，或者把 cat 换成另一只 cat，或者留下 cat 的影子。

换句话说：**模型缺的是"操作语义"，不是"概念知识"**。

这个观察一成立，事情就简单了。既然要教的是操作语义，那么 source/target pair **只要在操作层面精确对应就行，外观完全可以是合成的、卡通的、collage 风格的**。

## 4. Collage Pipeline：用一个 stickers 包解决所有 editing 任务

这是 paper 的另一半灵魂。想象你有一堆 PNG stickers（带 alpha 通道的小图，比如一只猫、一只狗、一朵花、一个苹果），还有一个空白 canvas。你可以程序化地做：

**Instruction editing 的 removal**：
- Source = canvas 上贴猫 sticker
- Target = 干净 canvas
- Instruction = "remove the cat"
- 像素级精确对应，零 artifact，因为就是同一个 canvas 砍掉 sticker

**Instruction editing 的 replacement**：
- Source = canvas + 猫 sticker
- Target = canvas + 同位置换狗 sticker
- Instruction = "replace the cat with a dog"

**Instruction editing 的 addition**：
- Source = 干净 canvas
- Target = canvas + 苹果 sticker
- Instruction = "add an apple"

**Drag editing 的 translation**：
- Source = canvas + sticker 在位置 $(x, y)$
- Target = canvas + 同一 sticker 在 $(x+dx, y+dy)$
- Prompt = "$(x, y, dx, dy)$"
- 纯 affine 变换，像素级精确

**Reference image generation**：
- Source = canvas 上贴了 subject sticker
- Target = 让模型基于这个 subject 生成新场景
- 模型必须从 sticker 里抽 subject 特征，不能 memorize 整张图

**Subject-driven**：
- 一个 canvas 上贴多个 sticker，target 只用一个 subject 重新生成
- 强迫模型学会 "从 reference 里挑我需要的那个"

**Seg & Detection（反向任务）**：
- Source = canvas + sticker
- Target = sticker 区域画 bounding box 或上色
- 把 generator 当 perception 用

所有这些数据都是 **程序化生成**，1 秒能产几百对，scale 到 60M+ 没问题。而且每对 pair 的 source/target 在像素上完全对齐，filter 几乎不需要——因为生成逻辑就是对的。

额外好处：**T2I 任务也能用 collage 增强**。在 canvas 上放 "3 个红圆 + 2 个蓝方块"，根据精确坐标生成 prompt "three red circles and two blue squares, ..."。这直接修好了 T2I 模型在 quantity / position / color / text rendering 上的老大难问题。GenEval 上 Counting 和 Position 指标超过 SD3 就是这么来的。

## 5. 框架的真正贡献：控制变量下的 Unet vs DIT 实验

这部分 paper 里写得最干练，但我觉得是最被低估的。作者做了一件很少 paper 会做的事：**在公平条件下做 framework ablation**。

他们把 SDXL (Unet), PixArt (DIT), SD3 (DIT) 全部 **压到 0.85B 参数、相同 VAE、相同 CLIP、相同 LAION 数据、相同 runtime** 训练。结论非常有意思：

**结论一**：DIT 优于 Unet **不是因为 transformer 本质更好**，而是因为 **DIT 把计算堆在 2× downsampled latent**（分辨率 /2），Unet 把计算堆在 4× latent（分辨率 /4）。

直觉：2× latent 还保留了 high-frequency 信息，attention 在这里做能学到 "哪个 prompt token 对应哪个像素区域"；4× latent 已经太 coarse，attention 只能做 global concept reasoning，对 T2I 这种需要 pixel-level 对齐的任务不划算。

**结论二**：Unet 的 **long skip connection** 让训练收敛快 4 倍，但纯 DIT 没这个机制。

为什么 skip 加速这么明显？纯 transformer stack 深层梯度要穿过几十层 attention/FFN 才回到浅层，梯度衰减严重；Unet 的 U 形 long skip 给了一条高速公路，浅层直接拿到深层监督信号。

**结论三**：所以最优解 = **DIT 的算力分配 + Unet 的 long skip**。这就是 DreamOmni-V3 的设计：
- 所有 DIT block 集中在 2× latent 上做 attention
- 加 Unet 风格 long skip：early feature 和 late feature concat 后过 linear
- 1× latent 用 residual conv 处理细节（attention 在 1× 上太贵）

最终 V3 比 SD3-Medium 收敛快 4×，FID 还更好。

## 6. 条件怎么统一进同一个 transformer

具体做法：
1. **不用 CLIP 或 T5，用 Qwen2-VL 7B 当 encoder**。理由是 Qwen2-VL 同时训练了 image token 和 text token，每个 image token 的 embedding 已经是 "知道自己在看什么" 的，对 reference image generation 这种要把 subject 像素特征传到 output 的任务至关重要。
2. **取 Qwen2-VL penultimate layer feature** 作为 prompt embedding。
3. **把 VLM feature tokens 和 noisy latent tokens 沿 sequence 维度拼接**，一起进 DIT block 做 **multi-head self-attention**（不是 cross-attention）。
4. **FeedForward 分两路**：VLM feature 走自己的 FFN，noisy latent 走另一个 FFN，结构一样但权重独立。理由是语义 token 和像素 token 分布差异大，共享 FFN 互相干扰。
5. **高一致性任务额外通道**：instruction editing、drag editing 这种要求非编辑区域像素级保留的，把 source image 的 VAE 编码直接接到 noisy latent stream（绕过 VLM），保证 source 的 raw 像素信息能直接被 attention 用上。

整个设计一个 cross-attention 都没有，全部是 self-attention。好处是 image condition 内部也能互相 attend（比如 reference image 不同 patch 之间的关系能被建模），坏处是计算量变成 $O((N_{text} + N_{latent})^2)$，但 2× latent 上 token 数可控。

## 7. 训练 setup 一些细节

- **2.5B 参数 DIT** + 7B Qwen2-VL（VLM 提前 precompute feature 省训练 forward 成本）
- **FLUX-schnell 的 VAE**，比 SD 的 VAE 保留更多 latent channel（16 channels），细节容量更高
- **Rectified Flow loss**：前向 $\mathbf{z}_t = t\mathbf{z} + (1-t)\boldsymbol{\epsilon}$，loss 是预测 velocity $(\mathbf{z} - \boldsymbol{\epsilon})$ 的 MSE。比 DDPM 的弯曲轨迹拉直成直线，inference step 可以很少
- **三阶段**：256 (377K iters) → 512 (189K iters) → 1024 (140K iters)。第三阶段只训 12M 高质量 T2I + 1M/class 高质量合成，避免低质数据破坏细节
- **31 个 aspect ratio bucket**（4:1 到 1:4），支持多分辨率生成
- **64×A100**，这是真烧钱

## 8. 结果到底有多强

**T2I (GenEval)**：Overall 0.70 和 SD3-Medium 持平，Position 0.34 vs SD3 的 0.28，Counting 0.65 vs 0.63——合成数据强化 quantity/position 见效了。

**Inpainting FID**：0.8371，SD-inpainting 1.3522，ControlNet 1.8393。比 ControlNet 低 54%。

**Outpainting FID**：1.6926，ControlNet 4.2337。差距 2.5×。Outpainting 这种需要理解整图结构的任务对 unified 框架特别友好。

**Instruction/Drag/Reference**：主要是 qualitative 比较，DreamOmni 在非编辑区域一致性、编辑区域生成质量、subject 保留上都打过 MGIE / InstructP2P / IP-Adapter / BLIP-Diffusion。

**Drag 大角度 rotation 仍然失败**——作者承认 limitation，因为 collage 只做了 2D affine，没做 3D 透视变换。

## 9. 我觉得这篇 paper 真正值得带走的东西

1. **"操作语义 vs 概念外观"的解耦**——这个 hypothesis 一旦接受，整个 editing 数据问题就破解了。同样的思路可以用在 robotics simulation data、code agent 的 synthetic function 生成、tool-use 的 sandbox 数据上。

2. **控制变量 framework ablation**——DIT 优势的真正来源是算力分配位置，不是 transformer 本身。这种 fair comparison 在 T2I paper 里几乎没人做，因为大家都各自调参。这帮人敢做，因为他们的目标就是 unification，不是为了证明自己 framework 最强。

3. **VLM 当 unified encoder 取代 cross-attention**——这可能是下一代 T2I 模型的标配。FLUX、SD3 还在用 CLIP/T5 双 encoder + cross-attention，DreamOmni 已经把这条路绕开了。

4. **Collage 数据哲学**：当你想教模型一个操作（而不是一个 concept），合成数据永远比真实数据强——因为合成数据保证 ground truth 精确，真实数据永远有噪声。

## 10. 我觉得 paper 没说清楚的

- **VLM 是 frozen 还是 finetuned**？如果 frozen，下游 editing 表达能力有 ceiling。
- **inference 时 VLM forward 7B 成本** paper 没算进去，27ms runtime 应该只是 DIT 部分，实际端到端慢得多。
- **collage 和真实照片的 stylization gap**——合成数据训出来的 model 在真实分布上 generalization 如何没单独 benchmark，stage 3 用高质真实数据部分缓解但没量化。
- **multi-subject reference** 的情况没评估，collage 里能放多 sticker 但 paper 只 demo 了单 subject。

更多细节参考：
- 项目主页: https://zj-binxia.github.io/DreamOmni-ProjectPage/
- OmniGen (同期类似工作): https://arxiv.org/abs/2409.11340
- InstructPix2Pix (对比 baseline): https://arxiv.org/abs/2211.09886
- InstaDrag (对比 baseline): https://arxiv.org/abs/2405.13722
- Qwen2-VL (VLM encoder): https://arxiv.org/abs/2409.12191
- Rectified Flow: https://arxiv.org/abs/2209.03003

一句话总结：**架构上把 DIT 算力分配 + Unet skip 缝合，条件上用 VLM 替代 cross-attention，数据上用 collage 程序化生成解决 editing pair 稀缺问题**。三个事情互相 reinforce，最后 2.5B 参数同时打过 SD3-Medium 的 T2I、SD-inpainting 的 inpainting、ControlNet 的条件生成、IP-Adapter 的 subject-driven。工程上是一次比较扎实的 unification 尝试。

---

# DreamOmni: 统一图像生成与编辑模型深度解析

Andrej，这篇 paper 由 CUHK 的 Jiaya Jia 组和 ByteDance 合作完成，核心目标是把 T2I 生成和各种 editing 任务（instruction editing, inpainting/outpainting, drag editing, reference image generation, segmentation & detection）统一到一个 native 框架里，同时用一套合成 collage 数据 pipeline 解决 editing 数据稀缺的问题。我会从框架设计直觉、数据合成哲学、训练动力学三个层面拆解。

## 1. 核心动机：为什么 T2I 模型无法直接被编辑化

当前 T2I foundation model（SDXL [39], SD3-Medium [15], PixArt [10]）设计之初只为 text-to-image，下游 editing 都靠"打补丁"实现：
- ControlNet [62] 用额外插件注入条件
- IP-Adapter [60] 用 cross-attention 注入 subject 信息
- InstructPix2Pix [8] 通过扩展 input channel 把 source image 喂进去
- BLIP-Diffusion [28] 用 cross-attention 维持 subject

这些碎片化设计阻碍多任务联合训练，部署复杂。DreamOmni 想做的是 **native unification**——一开始就把所有任务的条件输入当作 sequence 一起 encode 进去。

Reference: 
- 项目页: https://zj-binxia.github.io/DreamOmni-ProjectPage/
- InstructPix2Pix: https://arxiv.org/abs/2211.09886
- ControlNet: https://arxiv.org/abs/2302.05543
- IP-Adapter: https://arxiv.org/abs/2308.06721

## 2. 框架分析：Unet vs DIT 的关键诊断

这部分是 paper 里最有启发的一段实验。作者把 SDXL [39], PixArt [10], SD3-Medium [15] 三套框架在 **相同 VAE、相同 CLIP text encoder、相同参数量(0.85B)、相同 runtime、相同 LAION 数据** 下公平对比（Fig. 3），这是非常罕见的对照实验，因为各 paper 通常各自调参。

### 2.1 关键观察

**观察一**：DIT 之所以优于 Unet，**并非** transformer 本质优势，**更准确地说**是因为 **算力分配位置不同**。
- Unet（SDXL）：把大部分 transformer block 放在 4× downsampled latent（即 /4 分辨率），低分辨率上做 attention
- DIT（SD3/PixArt）：把大部分计算堆在 2× downsampled latent（即 /2 分辨率）

直觉上：2× latent 还保留了较多 high-frequency 结构信息，attention 在这里做能学到更精细的语义-像素对应；4× latent 已经丢失太多细节，attention 在那里做更像是 global concept reasoning。所以 DIT 在 T2I 这种需要 prompt-pixel 精确对齐的任务上更划算。

**观察二**：Unet 的 **long skip connection（residual across U 形）** 能显著加速训练收敛，**而** DIT 的纯 transformer stack 缺这个机制收敛慢。

### 2.2 DreamOmni-V1/V2/V3 消融

作者设计三个变体逐步验证（都控制在 0.85B）：

| Version | 结构 | 关键区别 |
|---------|------|----------|
| V1 | 2 个下采样层 (2× 和 4×) | 无 Unet long connection |
| V2 | V1 + Unet connection | 加上 long skip |
| V3 | V2 但所有 DIT 操作集中在 2× latent + residual Conv 处理 1× | 算力集中 + skip 保留 |

结果：V3 比 SD3-Medium **收敛快 4 倍**，FID 更优。这个 4× 加速主要来自 long connection 改善梯度流——纯 transformer 深层梯度传到浅层会衰减，long skip 给了一条高速公路。

**直觉**：纯 DIT 在低分辨率 latent 上做 attention 是局部最优，但全局梯度路径上输给了 Unet；DreamOmni 把两边优势缝合——DIT 的算力分配 + Unet 的 skip。

### 2.3 输入条件如何统一

这是统一框架的核心。传统做法不同任务用不同输入通道，DreamOmni 把所有条件都通过 **VLM 编码为 token sequence**，然后和 noisy latent **沿 token 维度拼接**进入 DIT block 做 joint self-attention：

```
[VLM tokens (text + image prompts)] ⊕ [noisy latent tokens] 
  → Multi-Head Self-Attention 
  → 分两路 FeedForward (一路处理 VLM features, 一路处理 noisy latent)
```

注意几个细节：
1. **不使用 cross-attention**——cross-attention 在 SD 里是 text→latent 单向注入，DreamOmni 改成 bidirectional self-attention，让 image condition tokens 也能互相 attend（reference image 内部像素之间的关系也能被建模）
2. **FeedForward 分两路**：VLM features 和 noisy latent 用两个结构相同但权重独立的 FFN。直觉是语义 token 和像素 token 的统计分布差异大，共享 FFN 容易互相干扰
3. **长连接拼接**：early features 和 late features 沿 channel 维度 concat，再用一个 linear layer 融合，VLM 和 latent 各自用不同 linear——保留 Unet 的 skip 思想
4. **高一致性任务额外路径**：instruction editing 和 drag editing 这种需要非编辑区域像素级保留的任务，把 **source image 的 VAE 编码直接喂入 DIT**（绕过 VLM，直接进 latent stream），保证像素级 consistency

VLM 选 **Qwen2-VL 7B [56]** 的三个理由：(1) 任意分辨率输入，(2) 性能强，(3) license 友好。取 penultimate layer feature 作为 prompt embedding。

VAE 选 **FLUX-schnell 的 VAE**，因为它保留了更多 latent channels（FLUX VAE 通常 16 channels），细节容量更高。

References:
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- FLUX: https://blackforestlabs.ai/
- SDXL: https://arxiv.org/abs/2307.01952
- PixArt-α: https://arxiv.org/abs/2310.00426
- SD3: https://arxiv.org/abs/2403.03206
- DiT: https://arxiv.org/abs/2212.09748

## 3. 合成数据 Pipeline：把"理解操作"和"学习概念"解耦

这是 paper 的另一半核心 insight，值得仔细体会。

### 3.1 核心 hypothesis

作者发现 editing 失败的根源**不是模型不知道某个 concept 长什么样**（T2I 训练已经教了），**而是**模型不知道 editing **指令**意味着对像素做什么操作。例如 "remove the cat" 模型可能不知道要保留背景里其他物体。

所以数据构造的目标是 **教模型"操作语义"**，而非"概念外观"。这就解放了数据生成——只要保证 source/target pair 在操作层面是精确对应的，外观可以是合成的、stylized 的、collage 风格的。

### 3.2 Collage Pipeline 六大任务

**Task 1: T2I 增强**
在 canvas 上随机摆放 stickers、文字、几何形状，根据精确坐标生成 prompt（"3 red circles on top-left, 2 blue squares on bottom-right"），再用 LLM (InternVL2 [11]) 润色为自然描述。这强化了 quantity/position/color/text rendering 的 ground truth signal，避免了真实数据描述不准的问题。

**Task 2: Inpainting / Outpainting**
随机生成 smear mask、block mask、edge mask。训练时 50% 概率额外把 image caption 喂给 VLM——这让模型既能纯 mask 推理也能用 text hint。

**Task 3: Instruction-based editing**
分三类：
- **Removal**：source = background + object，target = background（直接砍掉 object）
- **Replacement**：source = background + object A，target = background + object B（同位置换）
- **Addition**：source = blank background，target = background + object（注意 paper 里特意用 blank background，因为加 object 需要 contextually 合适位置，blank 简化问题）

这里 collage 的妙处：传统 InstructPix2Pix 用 GPT-3 + Prompt-to-Prompt 生成 pair，成功率 <15%，且 artifact 多；MagicBrush 人工标注，规模受限。Collage 方法**保证 source/target 像素级精确对应**，artifact 完全可控。

**Task 4: Drag editing**
分 translation / scaling / rotation 三类。drag point 用 $(x, y, dx, dy)$ 格式作为 prompt 输入：
- $x, y$：source image 中 drag point 坐标
- $dx, dy$：位移向量
- 全部 normalize 除以图像宽/高

和 InstaDrag [50] 不同，InstaDrag 把每对 drag point 当一张图（sparse 且固定点数），DreamOmni 把多个 drag point 当 sequence 编码进 prompt，更灵活。

**Task 5: Reference image generation**
分两类：
- **Image-conditioned**（类 ControlNet）：从高质量 image 生成 canny / depth / segmentation map 当 source
- **Subject-driven**（类 IP-Adapter / BLIP-Diffusion）：canvas 上摆 sticker，让模型参考某个 sticker 生成新场景。这强迫模型从 reference 图里抽 subject 特征，**而非**记住整个 reference 图

**Task 6: Segmentation & detection**
source = background + object 合成，target 用 alpha channel 画 bounding box 或 color manipulation。这是反向任务——把 generation 模型当 perception 模型用，统一框架反过来也支持 understanding。

### 3.3 数据规模

| 类型 | 数量 |
|------|------|
| T2I 真实数据 | 125M (LAION 103M + 收集 22M) |
| T2I 合成 | 12M |
| Instruction editing | 12M |
| Inpainting/Outpainting | 12M |
| Drag editing | 12M |
| Reference gen | 12M |
| Seg/Det | 8M |
| **合成总计** | **~68M** |

合成数据占比接近 35%，但都精确标注，质量高。

References:
- InstaDrag: https://arxiv.org/abs/2405.13722
- MagicBrush: https://arxiv.org/abs/2312.06639
- BLIP-Diffusion: https://arxiv.org/abs/2305.14793
- InternVL2: https://arxiv.org/abs/2404.16821
- LAION-5B: https://arxiv.org/abs/2210.08402

## 4. 训练目标与三阶段 schedule

### 4.1 Rectified Flow Loss

DreamOmni 用 **Rectified Flow [33]** 而非 standard DDPM/DDIM。前向过程是噪声到数据的线性插值：

$$\mathbf{z}_t = t \mathbf{z} + (1 - t) \boldsymbol{\epsilon}$$

变量含义：
- $\mathbf{z}$：VAE 编码后的 clean latent（ground truth）
- $\boldsymbol{\epsilon} \in \mathcal{N}(0, \mathbf{I})$：标准高斯噪声
- $t \in [0, 1]$：时间步，$t=1$ 时是 clean data，$t=0$ 时是纯噪声
- $\mathbf{z}_t$：在时间 $t$ 下的 noisy latent

训练目标：

$$\mathcal{L} = \mathbb{E}\left[ \| (\mathbf{z} - \boldsymbol{\epsilon}) - v_\theta(\mathbf{z}_t, \mathbf{c}, t) \|_2^2 \right]$$

变量含义：
- $v_\theta$：DIT 模型（参数 $\theta$），预测 velocity field $(\mathbf{z} - \boldsymbol{\epsilon})$
- $\mathbf{c}$：条件信息（VLM encoded prompt features）
- $(\mathbf{z} - \boldsymbol{\epsilon})$：从噪声到数据的"方向"，也就是 velocity ground truth
- 整个 loss 是 velocity prediction 的 MSE

**直觉**：Rectified Flow 把 DDPM 的弯曲轨迹拉直成一条直线，模型只需学一个恒定方向的 vector field。这等价于一个 ODE $\frac{dz_t}{dt} = v_\theta(\mathbf{z}_t, \mathbf{c}, t)$，积分时步可以很少（FLUX-schnell 4 步即可）。这也是为什么 SD3 也用 Rectified Flow。

### 4.2 三阶段训练 schedule

| Stage | Resolution | Batch | LR | Iters | 数据 |
|-------|-----------|-------|-----|-------|------|
| 1 | 256×256 | 2048 | 1e-4 | 377K | 全量 |
| 2 | 512×512 | 1024 | 5e-5 | 189K | 全量 |
| 3 | 1024×1024 | 256 | 2e-5 | 140K | 12M 高质 T2I + 1M/类高质合成 |

**直觉**：从低分辨率 coarse-to-fine 是扩散模型常规 schedule，但 stage 3 用 **高质量子集** 而非全量数据——粗阶段学分布整体形状，精阶段只调细节，避免低质数据破坏高频。Batch size 随分辨率降是因为显存。

总训练规模：64×A100，2.5B DIT + 7B VLM（VLM 提前 precompute features 节省 forward）。

**多分辨率支持**：参考 SDXL，分 31 个 buckets，aspect ratio 从 4:1 到 1:4，训练时按 bucket 采样同 batch 同尺寸。

References:
- Rectified Flow 原文: https://arxiv.org/abs/2209.03003
- Stable Diffusion (LDM): https://arxiv.org/abs/2112.10752

## 5. 实验结果

### 5.1 Framework 对比（Fig. 3）

0.85B 参数下 FID 排序（越低越好）：DreamOmni-V3 < DreamOmni-V2 < SD3 < PixArt < SDXL。同时 V3 runtime 27ms/step，比 SDXL 28.59ms 更快，比 PixArt 34.61ms 快很多。

### 5.2 T2I GenEval（Tab. 1）

| Model | Overall | Single | Two | Counting | Colors | Position | Color Attr |
|-------|---------|--------|-----|----------|--------|----------|-----------|
| SD3-Medium | 0.70 | 0.99 | 0.84 | 0.63 | 0.88 | 0.28 | 0.55 |
| **DreamOmni** | **0.70** | 0.99 | 0.81 | **0.65** | 0.88 | **0.34** | 0.54 |

Overall 与 SD3-Medium 持平，但 Position 和 Counting 更好——这正是合成 T2I 数据强化 quantity/position 的效果。

### 5.3 Inpainting / Outpainting（Tab. 2）

| Model | Inpaint FID↓ | Inpaint LPIPS↓ | Outpaint FID↓ | Outpaint LPIPS↓ |
|-------|-------------|----------------|---------------|-----------------|
| SD-inpainting | 1.3522 | 0.1560 | 2.9179 | 0.2475 |
| ControlNet-inpaint | 1.8393 | 0.1594 | 4.2337 | 0.2521 |
| **DreamOmni** | **0.8371** | **0.1203** | **1.6926** | **0.1995** |

DreamOmni 的 inpainting FID 比 SD 低 38%，比 ControlNet 低 54%。Outpainting 更明显——ControlNet outpainting FID 4.23，DreamOmni 1.69，差距 2.5×。说明 unified 框架对 outpainting 这种需要理解整图结构的任务特别友好。

### 5.4 Instruction / Drag / Reference

主要是 qualitative（Fig. 6, 7, 8, 9），对比 MGIE [17], InstructP2P, ControlNet, BLIP-Diffusion, IP-Adapter。结论：
- Instruction editing：非编辑区域更一致，编辑区域生成质量更高
- Drag editing：translation/scaling 完整性好，rotation 大角度有变形（作者承认这是 limitation）
- Reference gen：subject 保留 + prompt following 比 IP-Adapter 强

References:
- GenEval: https://arxiv.org/abs/2310.11525
- MGIE: https://arxiv.org/abs/2401.01399
- OmniGen (类似工作): https://arxiv.org/abs/2409.11340

## 6. 几点对架构 intuition 的延伸思考

### 6.1 为什么 VLM encoder 比 CLIP/T5 更适合 unified model

CLIP 是 contrastive 训练，text 和 image 在一个对齐的球面空间，但每个 modality 内部 token 级别的 representation 比较弱；T5 是纯 text encoder，无法吃 image。

VLM (Qwen2-VL) 同时训练了 image patch tokens 和 text tokens 的 next-token prediction，意味着每个 image token 的 embedding 已经"知道"自己对应的视觉内容。DreamOmni 把 VLM penultimate layer 输出直接拼到 noisy latent stream，**条件信号是 pixel-aware 的**，对 reference image generation 这种需要把 subject 像素特征传到 output 的任务至关重要。

### 6.2 为什么 bidirectional self-attention 比 cross-attention 更强

SD 的 cross-attention 是 $Q$ from latent, $K,V$ from text，单向信息流。DreamOmni 的 joint self-attention 让 image condition tokens 之间也能互相 attend，比如 subject-driven 时 reference image 内部不同 patch 的关系能被显式建模，相当于在生成阶段做了一层 image self-modeling。

代价是计算量 $O((N_{text} + N_{latent})^2)$ 而非 $O(N_{text} \cdot N_{latent})$，但 2× latent 上 token 数可控。

### 6.3 合成数据哲学的普适性

这篇 paper 最值得借鉴的不是 collage 技术本身，**而是** "操作语义 vs 概念外观"的解耦思想。任何 editing/agent 任务如果能把 instruction 的 ground truth 用合成方式精确表达，就能绕过人工标注瓶颈。类似思想在 robotics 的 simulation data、code generation 的 synthetic function 生成里都见过。

### 6.4 Limitations 猜测

Paper 没明说的：
1. Rotation drag 大角度失败——collage 只做了 2D affine，没做 3D 透视变换
2. Subject-driven 在多 subject 干扰下表现未评估
3. 合成数据 stylization gap——collage 风格和真实照片有差距，stage 3 用高质真实数据部分缓解，但 unified 任务的 real-world generalization 没单独 benchmark
4. VLM 7B frozen 还是 finetuned 未明说，如果 frozen，下游 editing 任务对 VLM 表达能力有 ceiling
5. 推理时 VLM forward 成本（7B 模型）会拖慢 inference，paper 给的 27ms runtime 应该是 DIT 部分，VLM 编码要单算

## 7. 总结

DreamOmni 在工程上做了三件有积累价值的事：

1. **Framework fairness analysis**：在控制变量下证明 DIT 优势来自算力分配位置 + Unet skip 来自梯度流加速，由此设计 V3 = DIT 算力分配 + Unet skip + 1× residual conv
2. **VLM as unified encoder**：用 Qwen2-VL 把所有条件（text / image / drag points / mask）统一为 token sequence，joint self-attention 替代 cross-attention
3. **Collage synthetic pipeline**：把"理解 editing 操作"和"学习 concept 外观"解耦，让 editing 数据生成从 hours/image 降到 seconds/image

最终 2.5B 参数的 native unified model 在 GenEval 与 SD3-Medium 持平，inpainting/outpainting 显著超过 SD-inpainting 和 ControlNet，instruction/drag/reference editing 在 qualitative 上超过 MGIE / InstructP2P / IP-Adapter / BLIP-Diffusion。

更多参考：
- 项目主页: https://zj-binxia.github.io/DreamOmni-ProjectPage/
- 论文作者 Bin Xia 主页: https://zj-binxia.github.io/
- OmniGen (类似 unified 思路): https://arxiv.org/abs/2409.11340
- WeMix (LLMGA 系列): https://arxiv.org/abs/2311.16500

整体看，这篇 paper 是 LLM unification 思想在视觉生成领域的延续，关键 contribution 在于 **framework 诊断实验** 和 **collage 数据哲学**，二者都值得在其他多模态统一任务上复用。
