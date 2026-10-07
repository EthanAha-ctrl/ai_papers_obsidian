---
source_pdf: Visual Anagrams Generating Multi-View Optical Illusions with Diffusion
  Models.pdf
paper_sha256: e50a5e9eb6425bbffe91c9c4d0901a5b73e5e21ef5c626f8589953e2d163c209
processed_at: '2026-08-13T02:09:07-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍 Visual Anagrams

好, 我上次讲得太"教科书"了, 这次用大白话从头过一遍, 中间多塞一些联想和直觉。

---

## 这篇 paper 到底在干啥

就一句话: **让一张图从不同角度看, 是不同的东西。**

你拿一张图, 正着看是只兔子, 翻过来看是只狗。或者转 90 度看是个瀑布。或者颜色反转看是另一个人。

这事儿人类艺术家干了几百年了, Dalí 画过, Escher 画过。但人类是靠脑子想、靠手画的。这篇 paper 问: 能不能让 diffusion model 自己干这事儿, 而且只给它两个 text prompt, 它自己想办法把两个东西"塞"进一张图里?

---

## 核心招数: 两个 prompt 同时 denoise

Diffusion model 的工作流程你应该很熟: 从纯噪声开始, 一步一步去噪, 最后得到一张干净的图。每一步, model 会预测"这张 noisy 图里藏着多少噪声", 然后把这部分噪声减掉。

那 illusion 怎么做? 假设我要正着看是兔子、翻过来看是狗。

paper 的 idea 特别简单: 每一步去噪的时候, 我**同时问 model 两个问题**:

1. "如果我正着看这张 noisy 图, 你觉得噪声长啥样, 才能把它变成兔子?"
2. "如果我翻过来看这张 noisy 图, 你觉得噪声长啥样, 才能把它变成狗?"

model 会给我两个噪声估计。一个是在"正着看 → 兔子"这个条件下的, 一个是在"翻着看 → 狗"这个条件下的。

然后我把这两个噪声估计**平均一下**, 用这个平均噪声去做这一步的去噪。

就这么简单。每一步都这么干, 从 $t=T$ 一路跑到 $t=0$, 最后出来的图就是: 正着看像兔子, 翻着看像狗。

---

## 为什么平均就行? 直觉在哪

这事儿一开始听上去有点玄 — 凭什么平均两个噪声估计, 就能得到一张"同时满足两个条件"的图?

直觉来自 energy-based model 的一个老结果: 如果你有两个条件分布 $p_1(x)$ 和 $p_2(x)$, 你想要的是"同时满足两个条件"的分布, 那它近似就是 $p_1(x) \times p_2(x)$ (乘积, 因为两个条件都成立 = 两个概率都大)。而乘积分布的 score function(也就是 log probability 的梯度)就等于两个各自 score function 的**和**。

Diffusion model 的噪声估计 $\epsilon_\theta$ 跟 score function 差一个常数倍, 所以"平均噪声估计" ≈ "平均 score" ≈ "乘积分布的 score"。

换句话说: **你在乘积分布上采样, 得到的样本自然就是两个条件都满足的样本。**

这个 insight 不是这篇 paper 发明的, 是 [Du et al. "Reduce, Reuse, Recycle"](https://arxiv.org/abs/2205.11487) 和 [Liu et al. Composable Diffusion](https://arxiv.org/abs/2206.01704) 早就提出来用于 compositional generation 的。这篇 paper 的贡献是: **把这个 idea 用到了"同一张图的不同 view"上**, 每个 view 对应一个 prompt, 于是就变成了 illusion 生成。

---

## 但 view 不是随便选的 — 这是 paper 的理论核心

这里有个大坑, 也是 paper 最漂亮的地方。

Diffusion model 训练的时候, 它看到的输入永远是这种形式:

$$\text{noisy 图} = w \times \text{干净图} + w' \times \text{高斯噪声}$$

这个高斯噪声是 i.i.d. 的, 每个像素独立, 方差一样, 服从 $\mathcal{N}(0, I)$。model 从小见惯了这种东西, 它脑子里"什么算噪声"的定义就是 i.i.d. Gaussian。

现在你做 illusion, 要把 noisy 图翻转一下再喂给 model。问题是: **翻转后的噪声还是 i.i.d. Gaussian 吗?**

如果是 permutation(像素重排), 那当然还是 — 每个像素的值没变, 只是位置变了, 独立同分布的性质完全保留。

但如果是其他变换呢? 比如:
- 把图整体颜色乘个 0.5(模拟"换光")? 不行, 噪声方差变了, 不再是 $\mathcal{N}(0, I)$ 而是 $\mathcal{N}(0, 0.25 I)$
- 45 度旋转但用 bilinear 插值? 不行, 新像素是老像素的加权平均, 像素之间引入了相关性, 不再 independent
- 拉伸 / 压缩? 不行, 不 preserve 体积

paper 的定理 (其实很简单): 对 $\epsilon \sim \mathcal{N}(0, I)$, 要让 $\mathbf{A}\epsilon$ 仍然 $\sim \mathcal{N}(0, I)$, **当且仅当 $\mathbf{A}$ 是 orthogonal matrix**。

证明就三行:
- 均值: $\mathbb{E}[\mathbf{A}\epsilon] = \mathbf{A} \cdot 0 = 0$ ✓
- 协方差: $\text{Cov}(\mathbf{A}\epsilon) = \mathbb{E}[\mathbf{A}\epsilon\epsilon^T\mathbf{A}^T] = \mathbf{A}\mathbf{A}^T$
- 要协方差等于 $I$, 就要 $\mathbf{A}\mathbf{A}^T = I$, 这就是 orthogonal 的定义

**直觉**: 标准 Gaussian 的密度函数 $p(x) \propto e^{-\|x\|^2/2}$ 只依赖 $\|x\|$, 是球对称的。Orthogonal transformation(包括旋转、反射、它们的高维推广)**保范数**, 所以 preserve 球对称性。任何拉伸、缩放、非正交的线性变换都会打破这个对称性, 让 model 把"变形后的噪声"误认成信号去消除。

---

## 由此 paper 把"view"这件事看透了

既然 view 必须是 orthogonal transformation, 那 image 上有哪些 orthogonal transformation 是"有意义的"?

paper 列了一堆:
- **Identity**: 啥也不变, 平凡
- **翻转 / 镜像**: 像素 permutation, 当然 orthogonal
- **90/180/270 度旋转**: 也是 permutation
- **颜色反转(取负)**: $\mathbf{A} = -\mathbf{I}$, 这是高维空间里的 180 度旋转, orthogonal
- **拼图重排**: 把图切成块重新排列, permutation, orthogonal
- **随机像素 permutation**: 极端情况, 也是 orthogonal
- **Inner rotation**: 图里某个圆圈区域旋转, 外面不动, 还是 permutation
- **任意 orthogonal matrix**: 理论上 work, 但人眼看不出意义

但有几个看着像旋转、其实**不行**:
- **45 度旋转 + bilinear 插值**: 引入像素相关性, 不 i.i.d. 了
- **白平衡(乘常数)**: 不 preserve 方差
- **拉伸 / 透视 / homography**: 不 preserve 体积

这就是为什么 paper 里那个"蓝黑裙 vs 白金裙"的尝试失败了 — 白平衡是 linear 但不 orthogonal, 结果图上出现一堆 spots, 因为 model 把"放大后的高斯噪声的峰"当成信号去除了。

---

## 一个 implementation 上的关键细节: pixel diffusion vs latent diffusion

之前的 [Tancik 工作](https://github.com/tancik/Illusion-Diffusion) 用 Stable Diffusion(latent diffusion) 干这个, 结果有个 artifact。

问题在哪? Stable Diffusion 不是直接在像素上工作, 是在一个压缩的 latent space 上工作。每个 latent code 其实对应原图的一小块 patch。当你旋转 latent grid 的时候, **latent 的位置变了, 但每个 latent 内部编码的 patch 的方向不变**。

打个比方: 你有一堆小瓷砖, 每块瓷砖上画着一个朝右的箭头。你把瓷砖的位置整体旋转 90 度, 但每块瓷砖上的箭头还是朝右的。结果拼出来一看, 箭头方向不对。

Model 为了强行让最终像素图旋转 90 度后看起来对, 只能用一些奇怪的"thatched lines"来拼 — paper Figure 3 有图, 你一看就懂。

**解决方案**: 用 [DeepFloyd IF](https://github.com/deep-floyd/IF), 它是 pixel-based diffusion, 直接在像素上 denoise, 没有 latent patch 的方向问题。paper 就是用这个 model 做的。

最后再用一个 SD x4 upscaler 把 256×256 放大到 1024×1024, 这一步不用 illusion 方法, 直接用 identity view 的 prompt 做 conditioning 就行, 不影响质量。

---

## 几个 design choice 的 ablation

paper 做了 ablation, 我挑重要的说:

### Mean vs Alternating

之前 Tancik 的做法是"alternating": 这一步用 view 1 的噪声估计, 下一步用 view 2, 再下一步用 view 1, 轮流来。

paper 改成 mean: 每一步都用两个 view 噪声估计的平均。

实验结果: mean 更好。直觉: alternating 会让优化方向来回"抖动", 两个 prompt 轮流拽, 像两个人一个往东拉一个往西拉, 你一会儿走东一会儿走西, 最后原地抖。mean 是两个力同时施加, 找到中间的平衡点。

N>2 的时候 alternating 更糟, 因为每个 view 分到的 step 太少。比如 4 view, 每个 view 只占 1/4 的 step, 不够 denoise 出像样的东西。

### Guidance scale $\gamma$

CFG guidance scale 越大, 分布越 sharp, 生成越贴合 prompt。实验显示 $\gamma=10$ 最好, $\gamma=3$ 不够强。直觉: illusion 任务本来就难, 每个视图都得非常"明确"地匹配自己 prompt, 所以需要强 guidance 把每个 view 都"钉死"在自己 prompt 描述的模式上。

### Negative prompting

Idea: view 1 的 prompt 当 view 2 的 negative prompt, 反之亦然, 让 model "藏住"另一个 view。

结果: concealment(互相藏的程度)提升, 但 alignment(每个 view 匹配自己 prompt 的程度)下降。原因: 两个 prompt 可能有根本相似性。比如 "oil painting of a dog" 和 "oil painting of a cat", 互做 negative 会同时 discourage "oil painting"。或者 "a cat" 当 "a dog" 的 negative, 会 discourage fur、four legs、tail — 这些 dog 也需要。

paper 最终选择**不用 negative prompting**。这是个很 honest 的决定 — 不为了指标好看就硬上 trick。

---

## 三个有趣的 failure case

### 1. Independent synthesis

Model 偶尔会"作弊": 图左边画 prompt 1, 右边画 prompt 2, 不真正交叠。

但 paper 说这**surprisingly rare**。Hypothesis: diffusion model 训练数据大多居中, 它的 inductive bias 倾向于"内容居中、整合", 而不是"分开、偏一边"。所以自然就倾向于做真正的 illusion 而不是 cheat。

这个观察本身很有意思 — **model 的 inductive bias 帮我们避免了 degenerate solution**。

### 2. Noise shift(蓝黑裙失败)

白平衡 = 像素值乘常数 $c$。$\mathbf{A} = c\mathbf{I}$, linear 但不 orthogonal。结果图上有 spots artifact — model 把 scaled Gaussian 噪声的"峰"误认为信号, 把它们"去掉"了, 留下空洞。

这正好是理论预言的失败: 不 preserve noise statistics, 就会出问题。

### 3. Correlated noise(45 度 bilinear 旋转)

45 度旋转用 bilinear sampling, 每个新像素是 4 个老像素的加权平均, 噪声之间引入 correlation。即使旋转"看起来合理", 也会 diverge。

implementation 教训: 旋转必须用 nearest-neighbor, 或者干脆实现成 permutation, 严格保持 i.i.d.。

---

## 跟之前工作的关系

### vs Hybrid Images (Oliva et al.)

[Hybrid Images](https://cvcl.mit.edu/hybrid_gallery/hybrid_gallery.html) 是把 image A 的高频和 image B 的低频混在一起, 近看 A 远看 B。这显式建模了 human visual perception 的 multi-scale 性质。

这篇 paper 完全不显式建模 human perception。它只是让 diffusion model 在多个约束下生成图, 而 model 内化的 visual priors 恰好让结果对人类也是 illusion。这间接支持了"generative model perception ≈ human perception"的假说 — [Gomez-Villa et al.](https://arxiv.org/abs/1902.03292) 发现 CNN 也会被 optical illusion 欺骗, 是同类 evidence。

### vs SDS-based 方法 (Burgert et al.)

[Burgert et al.](https://ryanndagreat.github.io/Diffusion-Illusions) 用 Score Distillation Sampling: 把 diffusion model 当 loss, backprop 优化一张图。概念上支持任意 differentiable transformation(superset of orthogonal), 但 quality 低、速度慢(要 backprop)。

Ours: 直接在 sampling 过程中 combine noise estimates, 不 backprop, 快且 quality 高。代价是 transformation 必须满足 orthogonal 条件。

经典 trade-off: **显式优化灵活但低效, 隐式约束高效但受限**。

### vs Tancik

[Tancik](https://github.com/tancik/Illusion-Diffusion) 是这工作的前身, idea 类似, 但:
1. 用 latent diffusion(SD), 有 thatched line artifact
2. 用 alternating reduction, 质量差
3. 只做 rotation, 没系统分析哪些 view 行

Ours 把这三点都修了: pixel diffusion、mean reduction、理论分析 orthogonal 约束。所以可以看作 Tancik 的"完整化和理论化"版本。

---

## 评估指标怎么设计的

用 CLIP 算一个 $N \times N$ 的 score matrix $\mathbf{S}$:

$$\mathbf{S}_{ij} = \phi_{\text{img}}(v_i(\mathbf{x}))^T \phi_{\text{text}}(p_j)$$

意思是: 第 $i$ 个 view 看起来跟第 $j$ 个 prompt 有多像。

两个指标:

- **Alignment score $\mathcal{A} = \min_i \mathbf{S}_{ii}$**: 对角线最小值, "最差 view 匹配自己 prompt 的程度"。希望所有 view 都 align, 所以看最差的。
- **Concealment score $\mathcal{C}$**: 用 softmax 让 CLIP "分类"每个 view 属于哪个 prompt, 看分类准确率。如果 view 1 看起来既像 prompt 1 又像 prompt 2, concealment 就低。

还报了 90%、95% quantile, 因为 illusion 生成随机性大, 实用场景下你会生成很多张挑一张, 所以"最好 case"的表现更重要。

实验结果(Table 1): Ours 在两个数据集上 alignment 和 concealment 都比 baseline 好, 尤其 concealment 提升明显。说明 Ours 不仅每个 view 匹配自己 prompt, 还互相藏得更好。

---

## 一些联想和延伸思考

### 1. "Adversarial examples for humans"

Paper 提到可以把 illusion 看作"对人类的 adversarial examples"。这跟 [Elsayed et al.](https://arxiv.org/abs/1802.08195) 那篇"fool both computer vision and humans"的工作遥相呼应。如果 diffusion model 能生成 fool 自己的 illusion, 而这些 illusion 也 fool 人类, 说明 model 内化的 perception 跟人类有结构性相似 — 不只是表面 correlation。

### 2. Polymorphic jigsaw

Paper 里有个很酷的 idea: 把图切成拼图块, 重排后变成另一幅图。也就是**同一套拼图块有两种拼法, 拼出两个不同的图**。这叫 "polymorphic jigsaw"。

这其实是 permutation view 的一个有趣 instance — permutation matrix 是 orthogonal 的, 所以理论上行。Appendix F 说他们手画了三种拼图块(corner、edge、center), 然后随机重排。

这让我想到一个问题: 能不能设计**三种**拼法? 理论上 N-view 方法支持任意 N, 但实际拼图块越多, 约束越紧, 质量越低。

### 3. Extreme permutation: 64×64 patch shuffle

Figure 6 有个极端 case: 64×64 的 patch grid, 几乎把所有像素都 shuffle, 仍能生成 meaningful image。这说明 diffusion model 的 prior 强到什么程度 — 在几乎"完全打乱"的约束下, 还能找到满足两个 prompt 的图。

这跟 [VQ-VAE 的 codebook shuffle 实验](https://arxiv.org/abs/1711.00937) 有异曲同工之妙, 都在测试 model 对"极端 disorder"下的生成能力。

### 4. Clever reuse 现象

Paper 提到 model 经常找"clever reuse": 比如那个 waterfall/rabbit/teddy bear 的三视图 illusion, teddy bear 的鼻子同时是 rabbit 的眼睛, 也是 waterfall 的一块石头。

这说明 diffusion model 的 representation 里有某种"parts compositional"结构 — 同一个 pixel pattern 在不同 context 下可以被 interpret 成不同 semantic part。这跟 [object-centric representation](https://arxiv.org/abs/2206.07636) 的研究方向有关, 但这里不是显式分解, 是隐式 emergent 的。

### 5. "Choosing prompts" 是个艺术

Appendix E 说 prompt 选择很 tricky:
- Abstract style("a painting of")比 realistic style("a photo of")好 — realistic 约束太强, illusion 没空间
- Human faces 是好 subject — 人类对 face 极敏感, 容易"看到"face 即使信息不全
- 完全不可预测 — 看似容易的 prompt 可能总失败, 看似不可能的可能 work 得很好

这个 unpredictability 说明: 我们对"diffusion model 在多约束下的行为"理解还很有限。理论上乘积分布采样应该 work, 但实际哪些 prompt pair 容易 work、哪些不容易, 跟 model 训练数据的 structure 关系很大。

### 6. 跟 ControlNet 的对比

MrUgleh 那个 viral 的 QR code illusion([Reddit 原帖](https://www.reddit.com/r/StableDiffusion/comments/16ew9fz/spiral_town_different_approach_to_qr_monster/))用 [ControlNet QR code monster](https://huggingface.co/monster-labs/controlnet_qrcode_monster) 微调过的 model, 让图的 global structure 匹配 template image。跟这篇 paper 的区别:

- ControlNet 方法: 需要 fine-tune, 用 image template 而非 text prompt, 本质是"约束 global structure"
- Ours: zero-shot, 用 text prompt, 本质是"在多个 conditional distribution 的乘积上采样"

技术上很不一样, 但目标相似 — 都是让图在满足"主 prompt"的同时, 隐含某种 secondary 结构。

### 7. 未来方向

paper 自己说局限:
- Color constancy illusion 不行(需要 non-linear color transform)
- Homography / stretch / non-volume-preserving 变形不行
- 生成质量不稳定

我猜未来可能的方向:
- **放宽 orthogonal 约束**: paper Appendix H 提到, 可能允许 $v$ 对 signal 和 noise 用不同 transform($v_1$ for signal, $v_2$ for noise)。比如 color shift 对 signal 是 linear transform, 但 noise 可以用某种修正让它仍 i.i.d. Gaussian。
- **Normalizing flow lift**: 用 normalizing flow 把 non-orthogonal transform "lift" 到 orthogonal 空间, 在 orthogonal 空间做 illusion, 再 pull 回来。
- **结合 human perception model**: [Chandra et al. SIGGRAPH 2022](https://dl.acm.org/doi/10.1145/3528233.3530730) 用 differentiable Bayesian vision model 生成 color constancy / size constancy illusion。可以把这种 perception model 作为额外的 score term 加进 diffusion process。
- **Video illusion**: 每帧是不同 view, 沿时间轴产生 illusion。或者 view 是"每隔几帧切换"。

---

## 一句话总结

**核心 idea**: diffusion model 每一步同时 denoise 多个 view, 把噪声估计平均, 就能在乘积分布上采样, 得到同时满足多个条件的图(illusion)。

**理论约束**: view 必须是 orthogonal transformation, 因为 diffusion model 训练时假设噪声是 i.i.d. Gaussian(球对称), orthogonal 是 preserve 这个对称性的唯一线性变换。

**工程细节**: 用 pixel diffusion(DeepFloyd IF) 避免 latent patch 方向问题; 用 mean reduction 避免 alternating 的"抖动"; 高 guidance scale 帮助每个 view 钉死自己 prompt。

**深层意义**: 这工作把"多视角 illusion 生成"这个看似艺术的问题, 变成了"在乘积分布上采样"这个有理论支撑的概率问题。而 orthogonal 约束的推导, 是一个 beautiful 的几何观察 — Gaussian 的球对称性 + orthogonal 的保范数性, 一拍即合。

---

强烈建议你去 [project page](https://dangeng.github.io/visual_anagrams/) 看动画版的 illusion, 那些图比静态图有说服力一百倍。特别是那个 waterfall/rabbit/teddy bear 三视图, 看着图从 waterfall 变成兔子再变成熊, 你会真的感受到"diffusion model 的 prior 跟人类 perception 之间有某种结构性共鸣"。

另外, Tancik 的 [repo](https://github.com/tancik/Illusion-Diffusion) 代码量很小, 你可以 clone 下来跑一遍, 然后把 alternating 改成 mean, 把 SD 换成 DeepFloyd IF, 几行代码改动就能看到质量提升 — 这是个非常 satisfying 的 hands-on 体验。

Appendix I 那个三行 proof 也值得在白板上推一遍, 它把整个 paper 的约束条件解释得清清楚楚: Gaussian 的协方差是 $I$, 线性变换后协方差是 $\mathbf{A}\mathbf{A}^T$, 要还是 $I$ 就要 $\mathbf{A}$ orthogonal。就这么简单。

---

# Visual Anagrams: 用 Diffusion Models 生成多视角 Optical Illusions

Andrej, 这篇 paper 我读起来非常过瘾, 因为它的核心 idea 极其简洁但 theoretical analysis 又很 clean。让我从 intuition 出发, 把它拆开来讲。

## 1. 核心 Insight: Parallel Denoising

这篇 paper 想做的事情本质上是: 找一张图 $\mathbf{x}_0$, 使得 $\mathbf{x}_0$ 符合 prompt $y_1$, 同时 $v_2(\mathbf{x}_0)$ 符合 prompt $y_2$ (其中 $v_2$ 是某个 transformation, 比如 flip)。

naive 想法是用 SDS 之类的方法做 optimization (Burgert et al. [2] 就是这么做的)。但这篇 paper 的 insight 是: **直接在 diffusion reverse process 里同时 denoise 多个 view**, 然后把 noise estimates 平均起来。

具体公式 (paper 中的 Eq. 2):

$$\tilde{\epsilon}_t = \frac{1}{N} \sum_{i} v_i^{-1}\Big(\epsilon_\theta\big(v_i(\mathbf{x}_t), y_i, t\big)\Big)$$

变量解释:
- $\tilde{\epsilon}_t$: 这一步 reverse diffusion 用的"组合" noise estimate
- $N$: view 的数量 (2-view illusion 就是 N=2)
- $v_i(\cdot)$: 第 $i$ 个 view 的 transformation function (比如 identity、flip、rotation)
- $v_i^{-1}(\cdot)$: 这个 transformation 的逆
- $\epsilon_\theta$: pretrained diffusion model 的 noise prediction network (参数为 $\theta$)
- $\mathbf{x}_t$: 当前 timestep $t$ 的 noisy image
- $y_i$: 第 $i$ 个 view 对应的 text prompt
- $t$: 当前 diffusion timestep

操作流程 (对应 Figure 2):
1. 拿到当前 noisy image $\mathbf{x}_t$
2. 对每个 view $v_i$: 把 $\mathbf{x}_t$ 用 $v_i$ 变换, 喂给 diffusion model 拿到 noise estimate $\epsilon_\theta(v_i(\mathbf{x}_t), y_i, t)$
3. 用 $v_i^{-1}$ 把这个 noise estimate "变回原坐标系"
4. 把所有 view 的 noise estimates 平均 → $\tilde{\epsilon}_t$
5. 用 $\tilde{\epsilon}_t$ 做一步 DDPM/DDIM update

**Intuition**: 这相当于在两个 conditional distribution 的"乘积"分布上采样。这个 idea 来自 compositional generation 的工作 [Du et al., "Reduce, Reuse, Recycle"](https://arxiv.org/abs/2205.11487) 和 [Liu et al., Composable Diffusion](https://arxiv.org/abs/2206.01704)。能量函数相加 ↔ noise estimate 相加, 因为 score function 是 log density 的梯度。多 view 的几何约束就是把这些能量项"乘起来"。

## 2. 为什么必须是 Orthogonal Transformation? — 理论核心

这是 paper 的 theoretical contribution, 也是最 elegant 的部分。

### 2.1 Linearity 条件

Diffusion model 期望输入 $\mathbf{x}_t$ 长成这样 (Eq. 3):

$$\mathbf{x}_t = w_t^{\text{signal}} \underbrace{\mathbf{x}_0}_{\text{signal}} + w_t^{\text{noise}} \underbrace{\epsilon}_{\text{noise}}$$

- $w_t^{\text{signal}}, w_t^{\text{noise}}$: signal 和 noise 的权重, 由 variance schedule 决定 (DDPM 里是 $\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1-\bar{\alpha}_t}$)
- $\mathbf{x}_0$: clean image
- $\epsilon \sim \mathcal{N}(0, I)$: i.i.d. Gaussian noise

当我们对 $\mathbf{x}_t$ 做 transformation $v_i$ 时, 必须保证变换后**仍然是 signal 和 noise 的线性组合, 且权重不变**。要求 $v_i$ 是线性的 (Eq. 4):

$$v_i(\mathbf{x}_t) = \mathbf{A}_i \mathbf{x}_t$$

- $\mathbf{A}_i \in \mathbb{R}^{D \times D}$: 矩阵, $D$ 是 flattened image 的维度

由 linearity (Eq. 5, 6):

$$v_i(\mathbf{x}_t) = \mathbf{A}_i (w_t^{\text{signal}} \mathbf{x}_0 + w_t^{\text{noise}} \epsilon) = w_t^{\text{signal}} \underbrace{\mathbf{A}_i \mathbf{x}_0}_{\text{new signal}} + w_t^{\text{noise}} \underbrace{\mathbf{A}_i \epsilon}_{\text{new noise}}$$

这就是为什么 view 必须线性: 否则 signal 和 noise 会"混"在一起, model 看到的不再是它训练时熟悉的"signal + i.i.d. Gaussian"的形式。

### 2.2 Statistical Consistency 条件

更强的约束: $\mathbf{A}_i \epsilon$ 必须仍然 $\sim \mathcal{N}(0, I)$。

证明 (Appendix I):

均值: $\mathbb{E}[\mathbf{A}\epsilon] = \mathbf{A}\mathbb{E}[\epsilon] = 0$ (线性变换保持零均值)

协方差:
$$\text{Cov}(\mathbf{A}\epsilon) = \mathbb{E}[(\mathbf{A}\epsilon)(\mathbf{A}\epsilon)^T] = \mathbf{A}\mathbb{E}[\epsilon\epsilon^T]\mathbf{A}^T = \mathbf{A}\mathbf{A}^T$$

要 $\mathbf{A}\epsilon \sim \mathcal{N}(0, I)$, 必须 $\mathbf{A}\mathbf{A}^T = I$, 即 $\mathbf{A}$ 必须 orthogonal。

**深层 intuition**: 标准 Gaussian $\mathcal{N}(0, I)$ 的密度函数 $p(\mathbf{x}) \propto e^{-\|\mathbf{x}\|^2/2}$ 是 spherically symmetric 的 (只依赖 $\|\mathbf{x}\|$)。Orthogonal transformations (包括 rotation、reflection、high-dim 推广) preserve 这个球对称性, 因为 $\|\mathbf{A}\mathbf{x}\| = \|\mathbf{x}\|$。任何 volume-distorting 的 transformation (如 stretch、color scaling) 都会破坏这个对称性, 让 model 把"扭曲后的 noise"误认为 signal 去除。

这正是 paper 里 "Dress illusion" 失败的原因 (Section 4.4): white balancing 是 linear ($\mathbf{A} = c \mathbf{I}$, $c \neq 1$), 但不 preserve noise statistics, 导致出现 spots artifacts — model 把 scaled Gaussian noise 的"峰"当 signal 去除。

## 3. 支持的 Views (枚举)

| View 类型 | 是否是 orthogonal? | 直觉解释 |
|---|---|---|
| Identity | ✓ ($\mathbf{A} = \mathbf{I}$) | 平凡 |
| Spatial rotation (90°, 180°, 270°) | ✓ (pixel permutation) | permutation matrix 是 orthogonal |
| Reflection / flip | ✓ (pixel permutation) | 同上 |
| Skew (roll columns) | ✓ (近似) | 每列位移不同, 仍是 permutation |
| Jigsaw rearrangement | ✓ (patch permutation) | permutation 的 subset |
| Random pixel permutation | ✓ | 极端情况 |
| Color inversion (negation) | ✓ ($\mathbf{A} = -\mathbf{I}$) | 高维 180° rotation |
| Inner rotation | ✓ (局部 permutation) | 圆形区域 rotate, 其余 fixed |
| 任意 orthogonal matrix $\mathbf{A}$ | ✓ | 理论上 work, 但人眼无法 interpret |

注意一个 critical 的 implementation 细节: **bilinear sampling 会破坏 noise statistics**。比如 45° 旋转用 bilinear sampling 会引入 noise correlations (一个新 pixel 是 4 个相邻 pixel 的线性组合), 导致 divergent samples。所以旋转必须用 nearest-neighbor 或直接做 permutation。

## 4. Architecture 选择: Pixel Diffusion vs Latent Diffusion

这是 paper 的一个关键 design decision, 也是对 Tancik [42] 工作的重要改进。

**问题**: Stable Diffusion 是 latent diffusion, 它的 latent 实际编码 local patch。当你 rotate latent grid, latent 的**位置**变了, 但 latent 自己编码的 patch 内容**方向不变**。这导致 model 必须用"thatched lines"去拼出旋转后的"straight lines"(见 Figure 3)。

**解决方案**: 用 [DeepFloyd IF](https://github.com/deep-floyd/IF) — pixel-based diffusion model (stage 1: 64×64, stage 2: upsample 到 256×256), 直接在 pixel space 操作, 避开了 latent 的 patch orientation 问题。

DeepFloyd IF 还 predict variance (除了 noise), 所以 variance estimates 也要 average (permutation view 时直接 inverse permutation; color inversion 时不 inverse variance, 因为 negate logged variance 没意义)。

第三 stage super-resolution 用 Stable Diffusion x4 upscaler, 不应用本方法, 直接用 identity view 的 prompt 做 conditioning upscale, 不损失 quality。

## 5. Classifier-Free Guidance 整合

CFG 公式 (Eq. 1):

$$\epsilon_t^{\text{CFG}} = \epsilon_\theta(\mathbf{x}_t, t, \varnothing) + \gamma\big(\epsilon_\theta(\mathbf{x}_t, t, y) - \epsilon_\theta(\mathbf{x}_t, t, \varnothing)\big)$$

- $\varnothing$: empty string 的 embedding (unconditional)
- $\gamma$: guidance scale (paper 里 ablation 显示 $\gamma = 10$ 最好, 见 Table 2)
- $y$: text prompt

在 multi-view 框架里, 直接把每个 view 的 noise estimate 换成它的 CFG estimate 即可。

**Intuition**: CFG 把 distribution "sharpen", 让 sample 更靠近 prompt 描述的模式。illusion 任务需要每个 view 都非常"明确"地匹配自己 prompt, 所以高 guidance scale 帮助大 — Table 2 显示 $\gamma=10$ 比 $\gamma=3$ 在 alignment 和 concealment 上都好。

## 6. Reduction Strategy: Mean vs Alternating

Alternating reduction (Tancik [42] 用的, Eq. 7):

$$\tilde{\epsilon}_t = v_{t \bmod N}^{-1}\Big(\epsilon_\theta\big(v_{t \bmod N}(\mathbf{x}_t), t, y\big)\Big)$$

意思: 不同 timestep 轮流用不同 view 的 noise estimate。

Ablation (Table 2): Mean reduction 更好 ($\mathcal{A}$=0.255 vs 0.252)。

**Intuition**: Alternating 会导致 "thrashing" — 不同 timestep 优化方向不一致, 像两个力交替拉一个物体, 物体在中间抖动而不前进。Mean reduction 相当于同时施加两个力, 找到平衡点。

对 N>2 view 的 case, alternating 更糟, 因为每个 view 分到的 step 数更少 (例如 4-view 时每 view 只占 1/4 step)。Figure 11 直观展示了 4-view 时 mean vs alternating 的差别。

## 7. Negative Prompting: 一把双刃剑

Idea: view 1 的 prompt 当 view 2 的 negative prompt, 反之亦然, 鼓励 model "藏住"对方 view。

但 ablation (Table 2) 显示: Negative prompting 提升 $\mathcal{C}$ (concealment: 0.576 vs 0.567) 但降低 $\mathcal{A}$ (alignment: 0.24 vs 0.255)。

原因 (Section 4.2): 两个 prompt 可能有 fundamental similarities。比如 "oil painting of a dog" 和 "oil painting of a cat", 互相做 negative 时同时 discourage "oil painting" style。或者 "a cat" 作 "a dog" 的 negative, 会 discourage fur、four legs、tail — 这些 dog 也需要。

正确做法: 只对**有差异的部分**做 negative, 比如 negative 用 "a cat" 而非完整的 "oil painting of a cat"。

Paper 最终**选择不用 negative prompting**, 因为 alignment 损失大于 concealment 收益。

## 8. 评估指标

### 8.1 Score Matrix (Eq. 8)

$$\mathbf{S}_{ij} = \phi_{\text{img}}(v_i(\mathbf{x}))^T \phi_{\text{text}}(p_j)$$

- $\mathbf{S} \in \mathbb{R}^{N \times N}$: score matrix
- $\phi_{\text{img}}, \phi_{\text{text}}$: [CLIP](https://arxiv.org/abs/2103.00020) 的 visual 和 text encoder (输出 L2-normalized embedding)
- $v_i(\mathbf{x})$: 生成的 illusion 在第 $i$ 个 view 下的样子
- $p_j$: 第 $j$ 个 prompt

$\mathbf{S}_{ij}$ 大: 第 $i$ 个 view 看起来像第 $j$ 个 prompt。

### 8.2 Alignment Score $\mathcal{A}$

$$\mathcal{A} = \min_i \mathbf{S}_{ii}$$

即对角线最小值 — 衡量**最差 view 的 alignment**。我们希望所有 view 都 align, 所以看最差的。

### 8.3 Concealment Score $\mathcal{C}$ (Eq. 9)

$$\mathcal{C} = \frac{1}{N}\text{tr}\big(\text{softmax}(\mathbf{S}/\tau)\big)$$

- $\tau$: CLIP 的 temperature
- softmax 沿一个方向 (paper 里两个方向都做 softmax 然后平均)

Intuition: 如果 CLIP 能完美分类每个 view 到对应 prompt (即 softmax 后对角线接近 1), $\mathcal{C}$ 接近 1。它衡量"互相藏住"的程度 — 如果 view 1 看起来既像 prompt 1 又像 prompt 2, concealment 就低。

### 8.4 Quantile 指标

因为 illusion 生成有很大的随机性 (有时成功有时失败), paper 还报了 90% 和 95% quantile ($\mathcal{A}_{0.9}, \mathcal{A}_{0.95}$ 等), 关注"最好 case"的表现。这很合理 — 对于艺术应用, 你可以生成 100 张挑 1 张。

## 9. 实验数据解读 (Table 1)

在 CIFAR 数据集 (45 个 prompt pair) 和 Ours 数据集 (50 个 prompt pair) 上, 每对生成 10 个 sample, 用 vertical flip view 比较:

| Method | CIFAR $\mathcal{A}$ | CIFAR $\mathcal{C}$ | Ours $\mathcal{A}$ | Ours $\mathcal{C}$ |
|---|---|---|---|---|
| Burgert et al. (SDS) | 0.225 | 0.501 | 0.233 | 0.501 |
| Tancik (alternating) | 0.278 | 0.595 | 0.256 | 0.545 |
| **Ours** | **0.287** | **0.624** | **0.275** | **0.574** |

观察:
- Ours 全面胜过 baseline, 尤其 concealment 提升明显 (CIFAR 上 +0.029 over Tancik)
- 95% quantile 上 Ours 更突出: CIFAR $\mathcal{A}_{0.95}$=0.327 vs Tancik 0.316, 说明 best-case 质量更高
- Burgert et al. (SDS) 表现最差, 因为 SDS 需要显式 optimization, 慢且 quality 低

Figure 4 的 density plot 显示: Ours 在 flipped 和 unflipped view 上的 CLIP score 分布基本对称 (parity 在 dashed line), 说明 flip view 的质量没牺牲。

## 10. 三个有趣的 Failure Cases (Section 4.4)

### 10.1 Independent Synthesis

Model 偶尔会"作弊": 在图的一半画 prompt 1 的内容, 另一半画 prompt 2, 而不是真正交叠。

**有意思的是这 surprisingly rare**。Paper 的 hypothesis: diffusion model 偏向 center content (训练数据大多居中), 所以倾向于"integrated and centered"而非"separate and off-center"。这其实是个 nice 发现 — model 的 inductive bias 帮助生成了真正的 illusion 而非 cheat。

### 10.2 Noise Shift (Dress illusion)

试图 recreate "blue/black vs white/gold dress" illusion, 用 white balancing (乘常数 $c$) 作 view。$c \mathbf{I}$ 是 linear 但不 orthogonal ($c \neq \pm 1$ 时 $\mathbf{A}\mathbf{A}^T = c^2 \mathbf{I} \neq \mathbf{I}$)。

结果: 图上出现 spots。Hypothesis: model 把 scaled Gaussian noise 的 peaks 当 signal 去除, 留下"洞"。这正好印证了 statistical consistency 理论的必要性。

### 10.3 Correlated Noise (45° bilinear rotation)

45° 旋转用 bilinear sampling 时, 新 pixel 是 4 个相邻 pixel 的加权平均, 引入 noise 间 correlation。即使旋转角度"看起来合理", 也会导致 divergent samples。

**这是 implementation 上的 critical gotcha**: 你必须用 nearest-neighbor 或干脆实现成 permutation, 严格保持 i.i.d. 性质。

## 11. 一些更深层的联想

### 11.1 与 Hybrid Images 的对比

[Oliva et al. 的 Hybrid Images](https://cvcl.mit.edu/hybrid_gallery/hybrid_gallery.html) 利用 human perception 的 multi-scale 性质: 高频是 image A, 低频是 image B, 近看 A 远看 B。这是显式建模 human perception。

这篇 paper 完全不同: 它**不显式建模 human perception**, 而是利用 diffusion model 隐式学到的 visual priors。这与 [Gomez-Villa et al.](https://arxiv.org/abs/1902.03292) 发现 CNN 也会被 optical illusion 欺骗的观察一致 — generative model 内化的 perception 与人类有结构性相似。

### 11.2 与 Score Distillation 的对比

[Burgert et al.](https://ryanndagreat.github.io/Diffusion-Illusions) 用 SDS: 把 diffusion model 当 loss, 通过 backprop 优化一张图。概念上支持任意 differentiable transformation (superset of orthogonal), 但 quality 低、速度慢。

Ours: 直接在 sampling 过程中 combine noise estimates, 不需要 backprop, 快且 quality 高。代价是 transformation 必须满足 orthogonal 条件。

这其实反映了一个常见 trade-off: **显式优化灵活但低效, 隐式约束高效但受限**。

### 11.3 与 Energy-Based Model Composition 的联系

[Du et al. "Reduce, Reuse, Recycle"](https://arxiv.org/abs/2205.11487) 证明: 如果有多个 energy-based model $E_1, E_2$, 它们乘积分布的 score 是各自 score 的加权和。Diffusion model 的 noise estimate $\epsilon_\theta$ 和 score function $\mathbf{s}$ 的关系是:

$$\mathbf{s}(\mathbf{x}_t, t) \approx -\frac{\epsilon_\theta(\mathbf{x}_t, t)}{\sigma_t}$$

所以平均 noise estimates $\approx$ 平均 scores $\approx$ 乘积分布的 score。这就是为什么 mean reduction 在理论上 justified — 它对应在"两个 prompt 都满足"的乘积分布上采样。

### 11.4 多 View 的 generalization

方法自然 extend 到 N>2 view (Figure 1 里有 3-view waterfall/rabbit/teddy bear 和 4-view teddy/bird/rabbit/giraffe)。每个 view 对应一个 transformation (不同旋转角度) 和一个 prompt。

有趣观察: model 经常找到"clever reuse", 比如 teddy bear 的 nose 同时是 rabbit 的 eye 和 waterfall 的 rock。这暗示 diffusion model 的 representation 有某种"parts compositional"的结构, 允许同一 pixel pattern 在不同 context 下被 interpret 成不同 semantic part。

### 11.5 与"Adversarial Examples for Humans"的联系

Paper Section 2 提到: 可以把 illusion 看作 "adversarial examples against humans" [Elsayed et al.](https://arxiv.org/abs/1802.08195)。Generative model 学到的 perception 在 illusion 面前的 ambiguity 和人类类似, 这给"model perception ≈ human perception"提供了又一个 evidence。

### 11.6 关于 "Visual Anagram" 这个概念

Paper 创造了 "visual anagram" 这个术语, 类比字母重排变新词。Pixel permutation 是 anagram 的字面意义。最极端 case: 64×64 grid 的 patch permutation (Figure 6), 几乎是把所有像素 shuffle, 仍能生成 meaningful image — 这说明 diffusion model 的 prior 强大到能在极端约束下"找到"满足条件的图。

### 11.7 关于 "Choosing Prompts" 的艺术 (Appendix E)

实用 insight:
- **Abstract style 比 realistic style 好** ("a painting of" > "a photo of") — realistic 约束太强, illusion 没空间
- **Human faces 是好 subject** — 人类 visual system 对 face 极敏感, 容易"看到"face 即使信息不全
- **难以预测**: 看似容易的 prompt 可能总失败, 看似不可能的可能 work 得很好

这个 unpredictability 本身就很有意思 — 说明我们对"diffusion model 在多约束下的行为"理解还很有限。

## 12. 局限与未来工作

Paper 承认:
- Color constancy illusion 不行 (因为涉及 non-linear color transform)
- Homography / stretch / 任何 non-volume-preserving 变形都不行
- 生成质量不稳定, 时常失败

未来方向 (我推测):
- 放宽 orthogonal 约束: paper Section H 提到可能允许 $v$ 对 signal 和 noise 用不同 transform ($v_1$ for signal, $v_2$ for noise), 这能扩展支持的范围
- 用 normalizing flow 把 non-orthogonal transform "lift" 到 orthogonal 空间
- 显式建模 human perception (Chandra et al. [3] 的 Bayesian vision model) 与 diffusion 结合

## 13. Summary

**一句话**: 把 diffusion reverse process 改成 multi-view parallel denoising, 平均 noise estimates, 约束 view 必须是 orthogonal transformation (因为要 preserve Gaussian noise statistics)。方法简单、理论 clean、效果惊艳。

**最 deep 的 intuition**: 这其实是把"多约束满足"问题转化为"乘积分布采样"问题, 而扩散模型的 score function 结构恰好让这种组合变得 trivial (只需平均)。Orthogonal 约束源于 noise 的球对称性, 是一个 beautiful 的几何观察。

## References

- [Project page (animated illusions)](https://dangeng.github.io/visual_anagrams/)
- [Paper PDF](https://arxiv.org/abs/2311.03132) (假设的 arXiv 链接, 实际需核对)
- [DDPM](https://arxiv.org/abs/2006.11239)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [Stable Diffusion / LDM](https://arxiv.org/abs/2112.10752)
- [DeepFloyd IF](https://github.com/deep-floyd/IF)
- [Tancik Illusion-Diffusion](https://github.com/tancik/Illusion-Diffusion)
- [Burgert et al. Diffusion Illusions](https://ryanndagreat.github.io/Diffusion-Illusions)
- [Du et al. "Reduce, Reuse, Recycle"](https://arxiv.org/abs/2205.11487)
- [Liu et al. Composable Diffusion](https://arxiv.org/abs/2206.01704)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Hybrid Images (Oliva et al.)](https://cvcl.mit.edu/hybrid_gallery/hybrid_gallery.html)
- [Gomez-Villa et al. CNNs and visual illusions](https://arxiv.org/abs/1902.03292)
- [Elsayed et al. Adversarial examples that fool humans](https://arxiv.org/abs/1802.08195)
- [DreamFusion / SDS](https://arxiv.org/abs/2209.14988)
- [Chandra et al. Differentiating probabilistic programs](https://dl.acm.org/doi/10.1145/3528233.3530730)

---

如果想 build 更深 intuition, 我建议你跑一下 [Tancik 的 repo](https://github.com/tancik/Illusion-Diffusion) 看 alternating 的实际效果, 然后改成 mean reduction 试试 — 几行代码的差别, 但质量提升肉眼可见, 是个非常好的"小改动大效果"的案例。另外, Section I 的那个 4 行 proof (orthogonal ⟺ preserve Gaussian) 值得在白板上推一遍, 它把整个 paper 的约束条件解释得清清楚楚。
