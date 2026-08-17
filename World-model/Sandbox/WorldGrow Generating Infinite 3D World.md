---
source_pdf: WorldGrow Generating Infinite 3D World.pdf
paper_sha256: f67dce0e6cd390708717e543ad0a84a7bc5969e10285bd3aebdff5cfcad32709
processed_at: '2026-08-13T05:45:33-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# WorldGrow 人话版

好, 我换一种讲法, 像在白板上给你画图那样讲。

---

## 一句话总结

这帮人想让你按一个按钮, 就能生成一个**无限大的、可以走进去乱逛的3D房子**, 而且每走一步看起来都还像那么回事, 不会走着走着墙就裂了、地板就穿了、沙发飘在空中。

---

## 他们到底要解决什么问题?

想象你在做一个小游戏, 需要一个3D场景。你有两条路:

**第一条路 — 让AI画2D图片, 再"举"到3D**
就像你让Midjourney画一个房间, 然后用depth estimation工具估算每堵墙离你多远, 再把2D图片"贴"到3D空间里。听起来work, 但有个致命问题: 你只能看到AI画图片那个角度。换个角度看, 墙的背面是空的, 沙发的侧面是扭曲的。就像你在一张照片后面糊个纸板, 正面看挺好, 侧面看就露馅了。

**第二条路 — 让AI直接学3D, 从3D数据生成**
这听起来更对, 但3D数据太少了。Objaverse有上千万个object (椅子、杯子、汽车), 但scene-level的3D数据, 全世界加起来也就几千个房子。你想train一个会生成房间的模型, 但训练数据还不如一张ImageNet的百分之一。学不出来strong prior。

**第三条路 — 用object-level的3D大模型, 硬transfer到scene**
TRELLIS这种模型, 在上千万个3D object上train过, 已经会生成非常漂亮的单个椅子、单个沙发。但它是"object-centric"的 — 它脑子里的concept是"一个孤立的物品", 没有wall, 没有floor, 没有"沙发旁边应该有茶几"这种spatial relationship。

WorldGrow走的就是这第三条路, 但要解决三个麻烦事:
1. 怎么把"生成单个object"的能力迁移到"生成房间"?
2. 怎么让相邻的房间无缝接起来, 不出seam?
3. 怎么保证全局布局合理, 不会生成10个卧室没厨房?

---

## 他们的做法, 用一个比喻先讲

想象你要盖一座无限长的大楼, 但你只有一个能盖"一个房间大小模块"的3D打印机。怎么办?

**WorldGrow的思路**:
1. 先用粗分辨率规划整栋楼的floor plan — 哪里是卧室, 哪里是走廊, 哪里是厨房。这一步用大block (覆盖2×2个fine block的范围), 只关心layout, 不关心细节。
2. 再用细分辨率, 在每个粗block的范围内, 把furniture、装饰、墙的纹理这些detail填进去。
3. 每生成一个新block时, 从已经生成好的左边、上边邻居各"借"一段过来作为context, 让model知道"我接下来生成的要和这部分接上"。
4. 借的这部分怎么用? 直接把它当成"已知区域", 让model做inpainting — 只生成中间空缺的那块。

就这么简单。但每个步骤都有魔鬼细节, 我下面拆开讲。

---

## 关键步骤1: 把TRELLIS改造成"懂scene"的版本

TRELLIS原本是这样工作的: 给一个3D object, 它会从150个角度渲染这个object, 每个角度用DINOv2提feature, 然后把这些feature"投影"回3D voxel grid上, 每个voxel获得的feature是"所有能看到它的views的feature的平均"。

听起来没毛病, 但在scene里出大问题了。

**举例**: 你有一个voxel在墙的后面, 比如沙发靠墙, 墙后面有个voxel。从正面看, 这个voxel是被墙挡住的。但TRELLIS原版的aggregation不care这个 — 它把所有view的ray都cast过来, 墙的texture feature也会被错误地分配到墙后面的voxel上。结果: 沙发靠近墙的那一面, 颜色被"墙的颜色"污染了, 出现color bleeding。

**WorldGrow的fix**: 对每个voxel, 每个view, 先用depth testing判断"这个voxel在这个view下到底可不可见"。只把可见views的feature平均进去。

光做这个fix还不够, 还有第二个问题: TRELLIS的decoder是在object数据上pretrain的, 它没见过"被切掉一半的墙"这种partial structure。当输入是scene block时, block边界处会有"半个墙", decoder会以为"哦这是个object, 应该补全", 然后在边界生成一堆floating geometry。

**WorldGrow的第二个fix**: 在scene block数据上重新train decoder, 让它学会"边界处partial是正常的, 别补全"。

这两个fix单独加都没用, 必须一起加。paper里有个ablation: 只加occlusion-aware反而LPIPS变差 (0.0741→0.0850), 因为encoder变了但decoder没跟上。两个一起加, LPIPS从0.0741降到0.0311, PSNR从23.17涨到31.32。

这个synergy insight我觉得很Karpathy: VAE这种encoder-decoder结构, 你改一边不改另一边, 它们mismatch, 反而更糟。必须jointly adapt。

---

## 关键步骤2: Block-by-block expansion的具体机制

这部分paper里Figure 4那个1D illustration我帮你画清楚:

假设block width = $w$。你现在有一个block A占据 $[0, w]$。你想生成block B接在A右边。

**关键设计**: B不是从 $[w, 2w]$ 开始, 而是从 $[w/2, 12w/8]$ 开始 — 它"往左挪了$w/2$"。具体来说:
- B的逻辑范围: $[4w/8, 12w/8]$, 总宽 $8w/8 = w$... 等等, 我重看下paper

OK paper里说: "reuse a 3/8w-wide margin from each neighboring block along X and Y axes. This overlapping region corresponds to $[1/2w, 7/8w)$ on each axis. Based on this context, we inpaint the central $5/8w \times 5/8w$ region to complete a new $12/8w \times 12/8w$ block."

所以:
- new block的总宽度 = $12w/8 = 1.5w$
- 其中 $3w/8$ 是从邻居借的context (左边邻居的 $[1/2w, 7/8w)$ 部分, 即邻居右半边的右3/8)
- 中间 $5w/8 \times 5w/8$ 是要inpaint的新区域
- 2D情况下, 从left + top + top-left三个邻居各借 $3w/8$, L-shape context

**重点**: 邻居原来在 $[7w/8, w]$ 这段的内容会被new block的生成覆盖掉 — 这部分被discard重新生成。这样做的目的是让new block和old block在边界处"融为一体", 而不是"硬拼"。

为什么是3/8而不是1/2或1/4? paper没明说, 但我猜: 太少了context不够, 太多了浪费computation (大部分block都是已知区域, 还生成个啥)。3/8可能是实验tune出来的sweet spot。

这个overlap+discard机制让我想到texture synthesis里的"seamless cloning"或者Poisson blending — 本质上都是用overlap region来smooth boundary。但WorldGrow是用generative model在overlap region"重新生成"而非简单blend, 这更高级。

---

## 关键步骤3: Coarse-to-fine的两阶段生成

这是我觉得最elegant的部分。我先把pipeline画出来:

```
Seed block (1个)
    ↓
[Stage 1: Coarse Structure Generation]
    用 G_s^c, block-by-block扩展, 得到coarse structure p_w^c
    (每个coarse block覆盖 2×2 fine blocks的范围, 关注layout)
    ↓
[Stage 2: Fine Structure Refinement]
    把 p_w^c 上采样到fine resolution → p_w^{c↑f}
    分割成fine blocks
    对每个fine block: 用SDEdit-style加噪 → G_s^f 去噪 → p_w^f
    ↓
[Stage 3: Latent/SLAT Generation]
    用 G_l^f, block-by-block生成latent z_w
    (输入: 之前生成的SLAT作为context + 当前fine structure)
    ↓
[Stage 4: Decode]
    用retrained decoder D 把 z_w decode成3D world W
```

**Stage 1为什么必要?** 你可以直接用fine model从seed开始生成, 但paper的ablation (Fig 9右)显示: 直接fine生成, furniture布局会非常混乱 — 一个block里床摆这边, 下一个block里衣柜摆那边, 没有global coherence。因为fine model的视野只有 $w \times w$, 它不知道"全局应该怎样"。

Coarse model的视野是 $2w \times 2w$, 能看到更大的spatial context, 所以它能学到"房间应该这样布局, 走廊应该这样延伸"。它生成的low-res structure相当于一个"全局规划图"。

**Stage 2的SDEdit trick很关键**:

公式: $\ell_{\mathrm{fblock}}^{(t')} = (1-t')\ell_{\mathrm{fblock}}^{(0)} + t'\epsilon$, 其中 $0 < t' < t$

- $\ell_{\mathrm{fblock}}^{(0)}$: 把coarse structure上采样到fine resolution, encode到latent, 这是"参考结构"
- $t'$: 一个比full noise $t=1$ 小的值, 比如0.3或0.5
- 我们只加部分噪声, 然后让fine model去噪

**为什么这样做?** 如果从full noise生成, fine model会用"typical fine structure prior"生成, 完全忽略coarse stage给的layout。这就白做Stage 1了。

SDEdit-style加部分噪声, 让latent保留coarse stage的信息, 但留有"细节优化空间"。fine model在这个"半噪声"基础上denoise, 既保持coarse layout, 又能添加fine detail。

类比: 你有一张模糊的低分辨率照片, 你想把它变成高分辨率。一种方法是直接用SR model生成, 但可能丢失原来的构图。另一种方法是先用低分辨率照片作为初始化, 加少量噪声, 再用diffusion model去噪 — 这样保住了原始构图, 同时获得细节。WorldGrow在3D上做的就是这件事。

**Stage 3为什么要单独做?** Structure (geometry)和SLAT (appearance)为什么分两阶段? 因为SLAT的representational capacity有限 — 一个voxel的latent vector $\mathbf{z}_i \in \mathbb{R}^C$ 要同时编码geometry和appearance信息, 容易冲突。分开做让每个stage专注一件事。

---

## 关键步骤4: Inpainting的输入设计

这个细节看起来小, 但很关键。

原TRELLIS的input: 就是noisy latent $\ell^{(t)}$

WorldGrow的input: 沿channel维concat三部分
1. noisy latent $\ell^{(t)}$
2. binary mask $m$ (1表示要inpaint, 0表示已知)
3. masked known region $\ell_m^{(0)} = \ell^{(0)} \otimes (1-m)$

为什么这样? 因为如果只把"已知区域"换成noise, model得从noise pattern里"猜"哪里是已知哪里是未知, 浪费capacity。显式给mask, model直接"知道"哪里要生成, 哪里要保留。

这个思想在2D inpainting里是标配 (RePaint, ControlNet), 但在3D latent space里做concat, WorldGrow算是第一次系统应用。

Loss:
$$\min_\theta \mathbb{E}_{(\ell^{(0)}, m, x), t, \epsilon} \| \mathcal{G}(\ell^{(t)}, m, \ell_m^{(0)}, x, t) - (\epsilon - \ell^{(0)}) \|_2^2$$

变量解释:
- $\ell^{(0)}$: clean latent
- $\ell^{(t)} = (1-t)\ell^{(0)} + t\epsilon$: noised
- $\ell_m^{(0)}$: 把已知区域extract出来, 未知区域置0
- $m$: mask
- $x$: text conditioning
- $\mathcal{G}$: 可以是structure generator $\mathcal{G}_s$ 或 latent generator $\mathcal{G}_l$
- target是 $(\epsilon - \ell^{(0)})$, flow matching的velocity

注意target里没有mask — 整个block都算loss, 包括已知区域。这样model既要"重建"已知区域 (相当于让它学会"copy context"), 也要"生成"未知区域。这比只在未知区域算loss效果更好, 因为让model明确知道"已知区域应该被原样保留"。

---

## 实验告诉我们什么?

### 1. Geometry质量 (Table 1)

WorldGrow vs 强baseline (TRELLIS†, 即TRELLIS fine-tune on 3D-FRONT):
- FID: 7.52 vs 24.61 (**3x better**)
- 1-NNA-CD: 66.30 vs 81.59 (越接近50%越好, 说明分布更接近GT)

这说明: 光fine-tune TRELLIS是不够的, WorldGrow的scene-friendly SLAT + inpainting + coarse-to-fine是真的有用。

### 2. Appearance质量 (Table 2)

- FID_CLIP: 3.95 vs TRELLIS†的13.17 (**3.3x better**)
- CLIP score: 0.843 vs 0.813

Appearance的大幅提升我归结于:
- Occlusion-aware feature避免了texture污染
- Retrained decoder学会了render partial structure
- Coarse-to-fine让texture有global consistency

### 3. 长程稳定性 (Table 4) — 这是我最喜欢的实验

设计: 生成7×7 scene, 只从外围 (beyond 3×3) sample block评估。

- WorldGrow外围FID = 5.43 (甚至比整体FID 7.52还好, 说明没有drift)
- SynCity外围FID = 51.97 (从整体的34.69退化)
- **关键note**: SynCity在70%的expansion attempt中直接fail, 表格只报告successful cases

这个实验直接测试"autoregressive generation的稳定性"。类比LM在long context下是否保持coherent — 如果你的LM生成长文会drift, 那它就没法用来写novel。WorldGrow的"不drift"性质让它真正具备infinite generation能力。

### 4. Human preference (Table 3)

91个evaluator, 1-5分:
- Structure Plausibility: 4.48 (TRELLIS 2.82)
- Geometry Detail: 4.44
- Appearance Fidelity: 4.33
- Continuity (unbounded scenes): 4.69

Continuity最高, 说明block-by-block inpainting真的做到了"看不出接缝"。

### 5. Outdoor generalization (Table 6)

在UrbanScene3D上train (只有10K fine blocks), FID从SynCity的93.45降到23.49 (**4x better**)。

这说明WorldGrow的方法论不局限于indoor, generalize到outdoor也work — 这是个好信号, 说明approach是general的, 不是overfit到某个domain。

---

## 一些更深的intuition

### 1. 为什么WorldGrow比SynCity快6x?

SynCity每个block 2分钟, WorldGrow 20秒。原因:
- SynCity是training-free的, 用2D diffusion + LLM caption + TRELLIS重建, pipeline冗长
- WorldGrow是end-to-end trained的3D model, 一次forward pass搞定一个block
- 但WorldGrow需要train (200K iterations), SynCity不需要

trade-off: 用训练时间换推理时间。如果要做大规模generation, 这个trade是值得的。

### 2. 为什么用Flow Matching而非DDPM?

TRELLIS选了flow matching (Lipman et al., 2023), WorldGrow继承了这个选择。原因:
- Flow matching的trajectory是OT (optimal transport) 直线, DDPM是弯曲的
- 直线trajectory意味着更少的采样步数就能达到同样质量
- 对sparse latent这种"非Euclidean structure"更友好

类比: DDPM像SDE (随机微分方程), flow matching像ODE (常微分方程), ODE更deterministic, 更高效。

### 3. Coarse-to-fine和cascaded diffusion的关系

WorldGrow的coarse-to-fine让我想到 cascaded generation:
- DeepFloyd IF: 64→256→1024 三阶段, 每阶段refine上一阶段
- Imagen: 类似
- WorldGrow: coarse (低res) → fine (高res), 用SDEdit-style加噪

差异:
- 2D cascaded通常是同semantic level不同resolution
- WorldGrow的coarse和fine在**不同semantic level** — coarse关注layout (房间布局), fine关注detail (家具、纹理)
- 这种"semantic hierarchy"比"resolution hierarchy"更powerful, 因为它explicitly model了"layout约束detail"的关系

### 4. Block size的选择哲学

Fine block: $w = h \approx 3m$ (一个房间大小)
Coarse block: $w = 2h$ (覆盖2×2 fine blocks, 即4个房间大小)

这个trade-off很关键:
- Block太大: SLAT capacity不够, detail损失
- Block太小: context不够, layout学不出来
- Coarse-to-fine本质上是"用大block学layout, 用小block学detail"的解耦

类比Transformer: coarse block像全局self-attention (理解long-range dependency), fine block像local attention (capture detail)。两者complementary。

### 5. 为什么overlap是3/8w?

这是个工程magic number, paper没解释。我猜:
- 1/4 (太少): context不够model infer出boundary continuity
- 1/2 (太多): new block大部分是已知区域, 生成效率低
- 3/8是个trade-off, 可能让"已知区域面积 / 未知区域面积 ≈ 1" (已知 : 未知 = 3×3/8 × 4 : 5/8 × 5/8 ≈ 多一点点)

实际上在2D下:
- L-shape context面积 = $(12/8w)^2 - (5/8w)^2 = 144/64 w^2 - 25/64 w^2 = 119/64 w^2$
- inpaint区域面积 = $(5/8w)^2 = 25/64 w^2$
- 比例 ≈ 4.76 : 1 (已知远大于未知)

所以context其实很充足, model有足够information做inpainting。这也解释了为什么生成质量稳定 — 每个new block都有大量context可参考。

### 6. 与World Models的connection

paper在Introduction里就提到"foundational for developing World Models and embodied AI"。我展开思考:

- WorldGrow生成的是static 3D world, 但这个world是navigable的 (Figure 1底部有embodied agent演示)
- 下一步logical的发展: 在WorldGrow生成的scene里训RL agent
- 关键挑战: WorldGrow生成的scene没有physics (没有collision, gravity), 不能直接用于embodied training
- 但作为"visual environment"用于navigation, planning tasks, 已经够用
- Future: 把WorldGrow + physics engine (e.g., PhysX, MuJoCo)结合, 这就是一个完整的world simulator

类比: Minecraft procedural generation + physics = 可玩的游戏世界。WorldGrow + physics = 可训练AI的synthetic world。

### 7. Limitations的真实意义

paper说limitation是: Z-axis不能扩展, data scale小, block-wise trade-off, 无semantic conditioning。

这些limitation其实指向了未来研究的方向:
- **Z-axis扩展**: 需要重新设计block topology — block从"horizontal slab"变成"3D cube"。技术上可行, 但inpainting mask设计会复杂很多 (要从6个邻居借context, 而非3个)
- **LLM conditioning**: 用GPT-4生成layout描述 (e.g., "3 bedrooms with a kitchen connected by hallway"), 让WorldGrow condition on这个描述 — paper说这是future work
- **UniLat3D integration**: UniLat3D (同一拨人的arXiv 2509.25079) 把geometry和appearance统一在single latent, 如果WorldGrow用它, 就不用分Stage 1和Stage 2, pipeline更简洁

---

## 我对这篇paper的最终看法

**它解决了一个真正的问题**: 之前3D generation要么object-level quality好但scene-level做不了, 要么scene-level能做但quality差。WorldGrow把两者的优势结合: object-level的strong prior + scene-level的spatial coherence。

**方法论elegance**: 几个fix都很targeted, 没有over-engineering:
- Occlusion-aware aggregation: 用visibility mask, 简单直接
- Retrain decoder: 用scene data fine-tune
- Inpainting formulation: 把generation转为inpainting, 直接利用pre-trained capability
- Coarse-to-fine: 用SDEdit-style bridge两个scale

**实验convincing**: 不仅在standard metrics上SOTA, 还有expansion stability实验直接验证"不drift"这个infinite generation的核心诉求。Human study 4.3-4.7分也说明qualitatively好。

**Potential**: 这套方法论 (block-wise + coarse-to-fine + inpainting + scene-friendly adaptation) 可以transfer到:
- 4D scene generation (时间维度扩展)
- City-scale generation (用不同scale的block hierarchy)
- Interactive world building (用户指定一些block, 让model填中间)

**不足**: data scale是个fundamental limit。3D-FRONT只有3425 houses, 这远不够train出真正diverse的世界。Objaverse-XL有10M objects, 但scene-level没有equivalent。如果有人能scale up scene-level 3D data (e.g., procedural generation + filtering, 或者auto-extract from games), 这条路线的天花板会高很多。

---

## Reference Links

- **WorldGrow project page**: https://World-Grow.github.io
- **TRELLIS** (Microsoft, CVPR 2025): https://microsoft.github.io/TRELLIS/ | https://github.com/microsoft/TRELLIS
- **SLAT paper**: https://arxiv.org/abs/2412.01506
- **BlockFusion** (SIGGRAPH 2024): https://wyliu.com/BlockFusion/
- **SynCity** (FAIR, 2025): https://github.com/facebookresearch/syncity
- **DINOv2** (Meta): https://github.com/facebookresearch/dinov2
- **3D Gaussian Splatting** (Inria, SIGGRAPH 2023): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Flow Matching** (Lipman et al., ICLR 2023): https://arxiv.org/abs/2210.02747
- **SDEdit** (Meng et al., ICLR 2022): https://arxiv.org/abs/2108.01073
- **3D-FRONT dataset**: https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future
- **Objaverse-XL**: https://objaverse.xl/
- **UrbanScene3D** (ECCV 2022): https://github.com/LordLiang/UrbanScene3D
- **UniLat3D** (arXiv 2509.25079): https://arxiv.org/abs/2509.25079
- **DiffInDScene** (CVPR 2024): https://arxiv.org/abs/2404.13293
- **Text2Room** (ICCV 2023): https://github.com/lukasHoel/text2room
- **LT3SD**: https://arxiv.org/abs/2404.09910
- **Hunyuan3D 2.0** (Tencent): https://github.com/Tencent/Hunyuan3D-2
- **Direct3D-S2**: https://github.com/pywu0316/Direct3D-S2
- **Infinigen** (Princeton, CVPR 2023): https://infinigen.org/
- **PointNet++** (Qi et al., NeurIPS 2017): https://github.com/charlesq34/pointnet2

---

希望这个版本更intuitive了! 核心take-away: WorldGrow把"生成无限3D世界"这个看似hard的问题, 分解成三个well-defined sub-problem (scene-friendly representation + inpainting-based extension + coarse-to-fine refinement), 每个sub-problem都有targeted solution, 组合起来就成了。这种"把大问题拆成可解的小问题"的taste, 我觉得就是好研究的标志。

---

# WorldGrow: Generating Infinite 3D World — 详细技术解读

你好 Andrej!这篇paper由SJTU MoE Key Lab的Sikuang Li等人(Huawei Inc.联合)撰写, 核心命题非常**Karpathy-friendly**: 把"3D原生生成模型"从object级别扩展到**可无限外推的scene级别**, 同时保住geometry和appearance的fidelity。让我把里面每个有意思的细节展开。

---

## 1. 问题本质: 为什么"无限3D世界"难?

paper把landscape分成三类approach, 我帮你梳理一下每种approach的"cost surface":

**Approach A — 2D lifting (Text2Room / LucidDreamer / WonderJourney / SynCity)**
- pipeline: text/image → 2D diffusion → depth estimation (MiDaS / DPT) → fuse到3D
- bottleneck是"local viewpoint optimization", 没有holistic 3D understanding
- 视角稍微偏离训练diffusion时见过的pose, 就会出现severe degradation (paper中明确说SynCity在expansion时FID从34.69涨到51.97)
- 类比: 像在用autoregressive LM生成图片但只看5x5 patch — 局部consistent但全局无结构

**Approach B — 3D native但scene-level data-starved (BlockFusion / LT3SD / Diffusion-SDF / NuiScene)**
- 直接学triplane / UDF / vector-set latent
- 问题: scene-level 3D数据集 (3D-FRONT只有~6800 houses, 真正能用的更少) 太小, 学不出strong generative prior
- 类比: 想在TinyStories上训练GPT-4 — capacity够, 但priors不够general

**Approach C — 3D foundation models但object-centric (TRELLIS / Hunyuan3D / Direct3D / Craftsman3D)**
- 在Objaverse-XL (10M+ objects) 上pretrain出极强的geometry+texture priors
- 但他们sample的是isolated assets, 一个沙发、一个椅子, 没有spatial continuity的概念
- 类比: 像训练了一个super-res model但只能处理256x256 patch, 不能处理4K image

WorldGrow的核心insight: **能不能把Approach C的strong priors"transfer"到Approach B的scene-level setting, 同时解决Approach A的不可扩展问题?** 这是这篇paper的真正贡献。

---

## 2. 基础: TRELLIS / SLAT 速览

你大概率已经知道TRELLIS (Microsoft, CVPR 2025), 但为了build intuition我快速过一下:

### 2.1 Structured Latents (SLAT)

representation: $\mathbf{z} = \{(\mathbf{z}_i, \mathbf{p}_i)\}_{i=1}^L$

- $\mathbf{z}_i \in \mathbb{R}^C$: 第 $i$ 个active voxel的latent feature, $C$是channel数
- $\mathbf{p}_i \in \{0, ..., N-1\}^3$: 第 $i$ 个active voxel的grid坐标, $N$是grid分辨率
- $L \ll N^3$: active voxels数量 (sparse!), 因为只在surface附近采样

注意这里 $N$ 是grid分辨率 (e.g., 384), $L$ 是真正活跃的voxel数量 (e.g., ~4000-8000), $C$ 是latent维度 (e.g., 1024 in XL model)。这种sparse representation比triplane的dense $N \times N \times N$ 体积节省了巨多memory, 也比pure point cloud更适合diffusion (有结构化topology)。

VAE encoder $\mathcal{E}$ + decoder $\mathcal{D}$ 的作用:
- $\mathcal{E}$: 把sparse voxel features $\mathbf{f} = \{(\mathbf{f}_i, \mathbf{p}_i)\}_{i=1}^L$ 压成latent $\mathbf{z}$
- $\mathcal{D}$: 把 $\mathbf{z}$ decode成3D Gaussians / radiance field / mesh
- $\mathbf{f}_i$ 的构造: 从~150个views渲染object, 用DINOv2提取feature map, 然后把每个view的pixel feature project到voxel $\mathbf{p}_i$ 上, 沿ray平均

### 2.2 SLAT生成: 两阶段Flow Matching

**Stage 1 (Structure Generation $\mathcal{G}_s$)**: 预测active voxel centers $\{\mathbf{p}_i\}_{i=1}^L$
- 输入latent: $\ell \in \mathbb{R}^{L' \times C'}$, 来自compressed occupancy volume的$L'$ tokens
- 这stage决定了"哪些voxel是active的" — 即coarse geometry

**Stage 2 (Latent Generation $\mathcal{G}_l$)**: 给定 $\{\mathbf{p}_i\}$, 预测 $\{\mathbf{z}_i\}_{i=1}^L$
- 输入latent: $\ell = \{\mathbf{z}_i\}_{i=1}^L \in \mathbb{R}^{L \times C}$
- 这stage决定了"每个active voxel长什么样"

两阶段都用同一个flow matching objective:

$$\min_\theta \mathbb{E}_{(\ell^{(0)}, x), t, \epsilon} \| v_\theta(\ell^{(t)}, x, t) - (\epsilon - \ell^{(0)}) \|_2^2$$

变量解释:
- $\ell^{(0)}$: clean latent (target)
- $\ell^{(t)} = (1-t)\ell^{(0)} + t\epsilon$: noised latent, linear interpolation
- $t \in [0, 1]$: time, $t=0$表示clean, $t=1$表示full noise
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: standard Gaussian
- $v_\theta$: flow transformer (DiT-like architecture with AdaLN)
- $x$: conditioning (image or text prompt, via CLIP/DINOv2 embedding)

注意这个loss形式和DDPM不一样, 是flow matching (Lipman et al., ICLR 2023) 的形式。$v_\theta$学的是velocity field, 即从noise到data的OT轨迹的切向量。这比DDPM的 $\epsilon$-prediction更general, 也更适合sparse token-based latents。

类比Karpathy的nanoGPT: 这里transformer架构相似, 但objective从next-token prediction换成flow matching, 因为我们学的不是sequential distribution而是continuous latent distribution。

---

## 3. WorldGrow的核心设计

### 3.1 Data Curation: 从house到scene block

3D-FRONT原始有6811个houses, 但作者filter后只剩3072 + 353 manually corrected = 3425个clean houses。filtering标准: mesh穿透(1971), furniture错位(1232), 太小(324), sparse(456), 其他(585)。

**Scene Slicing策略**:
1. 把一个house mesh导入Blender
2. 在bounding box内随机放置cuboid
3. 用Boolean Intersection切出block
4. Render top-down view, 计算occupancy — 如果 < 95% surface有visible content, 重新positioning
5. 迭代采样得到多个valid placements

这一步本质上是**rejection sampling**, 拒绝掉那些"大部分是空气"的slices。这避免了训练时model学到"sparse block = 随便生成"这种degenerate solution。

最终得到:
- 120K fine blocks ($w^f = h \approx 3m$, 立方体)
- 38K coarse blocks ($w^c = 2h$, 在XY平面上覆盖 $2 \times 2$ 个fine blocks, 高度不变)

这个 $2 \times 2$ 关系很关键, 是coarse-to-fine的物理基础。

### 3.2 Scene-friendly SLAT: 两个关键修改

这是paper里最"工程美学"的部分, 我详细展开:

**问题1: Direct feature aggregation失败**

vanilla TRELLIS给每个voxel $\mathbf{p}_i$ 提取DINOv2 feature $\mathbf{f}_i$ 时:
- 从~150个views渲染object
- 对每个view的每个pixel, cast ray
- 把ray上所有intersect的voxels都加上这个pixel的DINOv2 feature, 然后average

对object来说OK, 因为object很少有"wall在前面挡住chair"这种深层occlusion。但对scene来说:
- 一个voxel在墙的后面(被墙挡住), 但ray会穿过墙继续往后走
- 于是墙的texture feature错误地被分配到后面的voxel上
- 结果: scene里"墙的两面feature被混在一起", 出现color bleeding

**修改1: Occlusion-aware feature aggregation**

对每个voxel $\mathbf{p}_i$, 对每个view $v$:
- 计算binary visibility mask $M_v(\mathbf{p}_i)$: 用depth testing判断 $\mathbf{p}_i$ 在view $v$ 下是否可见
- 只在 $M_v(\mathbf{p}_i) = 1$ 的views上平均DINOv2 feature
- 公式上: $\mathbf{f}_i = \frac{\sum_v M_v(\mathbf{p}_i) \cdot \text{DINOv2}_v(\mathbf{p}_i)}{\sum_v M_v(\mathbf{p}_i)}$

这样, 墙后面的voxel只从"能看到它的views"获取feature, 不会被前面墙的feature污染。

类似的思想在NeRF的visibility-aware rendering中也有, 但这里是用在feature aggregation上, 而非rendering。

**问题2: Decoder在block边缘产生floaters**

vanilla TRELLIS的 $\mathcal{D}$ 是在Objaverse上预训练的, object没有"block boundary"概念, 所以当输入是scene block时, 在block的物理边缘 (e.g., 墙被切掉一半的位置) 会产生floating geometry — 因为decoder不知道这是被切掉的真实结构, 而是把它当成"object本来就这样"。

**修改2: Retrain decoder $\mathcal{D}$ on scene blocks**

用scene block数据(经过occlusion-aware feature提取)重新训练decoder, 让它学会:
- block边缘的partial structure是合法的
- 不要"补全"被切掉的部分
- 在block boundary处保持clean geometry

**Ablation验证**:
- 用LPIPS↓ / PSNR↑ / SSIM↑ 测量reconstruction quality
- vanilla SLAT: LPIPS=0.0741, PSNR=23.17
- +Occlusion-aware only: LPIPS=0.0850 (变差!)
- +Retrain D only: LPIPS=0.0491, PSNR=25.84
- +Both: LPIPS=0.0311, PSNR=31.32

注意: **单独加occlusion-aware反而变差**, 因为encoder给decoder的feature分布shift了, decoder没适应。只有两个一起加才能synergize。这是个很有意思的insight — 对VAE做修改时, encoder和decoder必须jointly adapt, 不然就mismatch。

### 3.3 3D Block Inpainting

这是让WorldGrow能"region growing"的关键mechanism。

**任务定义**: 给定一个block, 切掉一部分, 让model预测被切掉的部分。

**训练时的masking strategy**:
- 随机选两个splitting positions (沿X和Y轴), 把block切成4个quadrants
- 保留1个作为context, mask掉另外3个

这个ratio (3:1 masked:context) 我觉得是关键design choice — 比token-wise random masking更难, 因为context更少, 但更像实际inference时的设置 (inference时新生成的block大部分是missing的, 只能从邻居那借点context)。

**Input设计** (这是关键的modification):
- 原TRELLIS: 输入是noisy latent $\ell^{(t)}$
- WorldGrow: 沿channel维concat三部分:
  - noisy latent $\ell^{(t)}$
  - binary mask $m$ (指示哪些位置需要inpaint)
  - masked known region $\ell_m^{(0)} = \ell^{(0)} \otimes (1-m)$

这就像Repaint / ControlNet的思想 — 让model显式知道哪里是"已知", 哪里是"未知", 不用靠latent中的noise pattern隐式推断。

**Structure inpainting mask**: $m_s \in \{0,1\}^{N \times N \times N}$ (dense voxel mask)
**Latent inpainting mask**: $m_l = \{(m_i, \mathbf{p}_i)\}_{i=1}^L$ (sparse mask, 因为SLAT本身是sparse的)

**Loss**:
$$\min_\theta \mathbb{E}_{(\ell^{(0)}, m, x), t, \epsilon} \| \mathcal{G}(\ell^{(t)}, m, \ell_m^{(0)}, x, t) - (\epsilon - \ell^{(0)}) \|_2^2$$

变量解释:
- $\ell_m^{(0)} = \ell^{(0)} \otimes (1-m)$: 用Hadamard product把"已知区域"的clean latent保留下来
- $\mathcal{G}$: 可以是 $\mathcal{G}_s$ (structure) 或 $\mathcal{G}_l$ (latent)
- 其他符号同上

训练了4个独立model:
- $\mathcal{G}_s^c$: coarse structure inpainting (38K coarse blocks)
- $\mathcal{G}_s^f$: fine structure inpainting (120K fine blocks)
- $\mathcal{G}_l^f$: fine latent inpainting (120K fine blocks)
- 没有coarse latent inpainting (coarse只生成structure, appearance由fine stage负责)

### 3.4 Block-by-Block Expansion: 1D illustration

paper里Figure 4给了一个1D illustration, 我用文字描述:

假设block width = $w$。当前已有block A占据 $[0, w]$。要生成下一个block B占据 $[w, 2w]$。
- 实际上, B的"逻辑范围"是 $[w/2, 2w - w/8]$, 即 $[1/2 w, 15/8 w]$, 总宽 $12/8 w = 3/2 w$
- 但其实block B"生成时"的范围是 $[1/2 w, 7/8 w] \cup [7/8 w, 12/8 w]$ — 前半部分$[1/2w, 7/8w]$是从block A"借"过来的context (width $3/8 w$), 后半部分$[7/8w, 12/8w]$是要inpaint的新区域 (width $5/8 w$)
- 关键: **block A中原本 $[7/8w, w]$ 这部分被discard**, 因为它会被block B重新生成覆盖

为什么这样做?
- 重叠$3/8w$确保continuity (boundary处有足够的overlap来"blend")
- 但$3/8w$而不是$1/2w$或$1/4w$ — 这个比例是trade-off:
  - 重叠太少 → boundary处有seam
  - 重叠太多 → 每个block大部分是"已知"的, generation效率低
- $3/8$这个具体数字paper没解释为什么, 但我感觉是基于实验tuning的

2D情况下, 一个new block从三个邻居(left, top, top-left)各借$3/8w$的margin, 总共的"已知区域"是一个L-shape, 中间$5/8w \times 5/8w$的方形是要inpaint的。

### 3.5 Coarse-to-Fine Generation Pipeline

完整流程:

**Step 0**: 初始化seed block
- 可以用full mask做inpainting (相当于"从无到有")
- 或者从vanilla TRELLIS sample一个block作为起点

**Step 1 (Coarse Structure Generation)**: 用 $\mathcal{G}_s^c$ 逐block扩展, 得到coarse structure $\mathbf{p}_w^c$
- 每个coarse block覆盖 $2w \times 2w$ (在XY平面上是 $2 \times 2$ fine blocks的范围)
- 这个stage关心的是"global layout plausibility" — 哪里是房间, 哪里是走廊, 哪里是家具cluster
- 类比: 像先画floor plan, 再造房子

**Step 2 (Fine Structure Refinement)**: 用 $\mathcal{G}_s^f$ 精化 $\mathbf{p}_w^c$

这一步用SDEdit-inspired的strategy, 关键公式:

$$\ell_{\mathrm{fblock}}^{(t')} = (1-t')\ell_{\mathrm{fblock}}^{(0)} + t'\epsilon, \quad 0 < t' < t$$

- $\ell_{\mathrm{fblock}}^{(0)}$: 把upsampled coarse structure $\mathbf{p}_w^{c \uparrow f}$ encode到latent得到的clean reference
- $\ell_{\mathrm{fblock}}^{(t')}$: 加了"小"噪声 ($t' < t$, 即不完全加满噪声)的版本
- $\mathcal{G}_s^f$ 从 $\ell_{\mathrm{fblock}}^{(t')}$ denoise到 $\mathbf{p}_{\mathrm{fblock}}^f$

**为什么不直接从full noise生成?**
- 从full noise生成, model只能靠data prior生成"typical fine structure"
- 但这样会"丢弃" coarse stage学到的global layout信息
- 用SDEdit-style perturbation ($t' < 1$) 保留一部分coarse layout信息, 在此基础上加detail
- 类比: 类似于先用CLIP guide做粗grasp, 再用img2img refine detail

upsample方法: trilinear interpolation (简单但有效, 因为只是给一个起点)

**Step 3 (SLAT-based Appearance Generation)**: 用 $\mathcal{G}_l^f$ 生成latent
- 这stage和structure inpainting并行, 但用sparse latent mask而非dense voxel mask
- 输入: 之前生成的SLAT (作为context) + 当前block的fine structure (作为已知structure)
- 输出: 当前block的SLAT $\mathbf{z}$

**Step 4 (Decode)**: 用retrained $\mathcal{D}$ 把所有 $\mathbf{z}_w$ decode成renderable 3D world $\mathbf{W}$

---

## 4. 实验结果深度解析

### 4.1 Geometry评估 (Table 1)

指标 (在3D-FRONT上3×3 scene生成, 随机sample 1×1 block评估):

| Method | MMD-CD↓ | MMD-EMD↓ | COV-CD↑ | COV-EMD↑ | 1-NNA-CD↓ | 1-NNA-EMD↓ | FID↓ |
|---|---|---|---|---|---|---|---|
| DiffInDScene | 6.57 | 27.70 | 2.83 | 5.26 | 99.30 | 97.69 | 84.41 |
| BlockFusion | 2.90 | 28.79 | 16.60 | 13.16 | 97.89 | 98.19 | 25.09 |
| SynCity | 1.37 | 19.54 | 19.03 | 11.94 | 90.04 | 93.56 | 34.69 |
| TRELLIS | 3.15 | 23.75 | 13.97 | 11.74 | 99.20 | 98.79 | 53.49 |
| TRELLIS† (finetuned) | 1.47 | 15.03 | 46.56 | 45.95 | 81.59 | 74.55 | 24.61 |
| **WorldGrow** | **0.97** | **13.33** | **51.82** | **46.56** | **66.30** | **69.01** | **7.52** |
| Ours w/o DC | 1.00 | 13.84 | 46.76 | 40.49 | 69.01 | 74.65 | 9.09 |
| Ours w/o CSG | 1.08 | 13.62 | 43.93 | 40.28 | 73.24 | 72.33 | 17.04 |

变量解释:
- **MMD (Minimum Match Distance)**: 生成集到GT集的最小匹配距离的平均, 衡量fidelity
- **COV (Coverage)**: GT集被生成集"覆盖"的比例, 衡量diversity
- **1-NNA (1-Nearest Neighbor Accuracy)**: 二分类器区分real/fake的准确率, 50%最好, 越接近50%说明分布越像
- **CD (Chamfer Distance)**: 两个点集的双向最近邻距离
- **EMD (Earth Mover's Distance)**: 最优传输距离, 比CD更严格
- **FID**: 用PointNet++提取3D feature再算Fréchet distance

关键观察:
- WorldGrow在**所有**指标上SOTA
- 特别是FID: 7.52 vs TRELLIS†的24.61 — **3x better**
- 1-NNA: WorldGrow 66.30, 已经很接近"理想"的50% (相比其他method都90+)
- ablation里w/o CSG (coarse scene generation) FID从7.52涨到17.04, 证明coarse-to-fine critical

### 4.2 Visual Fidelity评估 (Table 2)

| Method | CLIP↑ | FID_Incep↓ | FID_DINOv2↓ | FID_CLIP↓ |
|---|---|---|---|---|
| DiffInDScene | 0.768 | 156.80 | 2066.13 | 42.43 |
| BlockFusion | 0.758 | 138.34 | 1776.79 | 42.04 |
| SynCity | 0.804 | 101.83 | 655.60 | 16.22 |
| TRELLIS† | 0.813 | 101.94 | 674.65 | 13.17 |
| **WorldGrow** | **0.843** | **29.87** | **313.54** | **3.95** |

CLIP score从TRELLIS†的0.813涨到0.843, FID_CLIP从13.17降到3.95 — 这说明WorldGrow不只是geometry好, **texture的真实感也显著更强**。

为什么? 我推断:
1. Occlusion-aware feature aggregation避免了color bleeding, texture边界更清晰
2. Retrained decoder在scene data上学会了"如何render好partial walls和furniture", 没有object-centric decoder的"补全bias"
3. Coarse-to-fine让appearance先有global consistency, 再有local detail, 避免了"local texture好但全局看不出是什么"的失败模式

### 4.3 Expansion Stability (Table 4)

这是我最喜欢的实验设计 — 验证"长时间生成不漂移":

设置: 生成7×7 scene, 只从**外围** (beyond initial 3×3) sample 1×1 block评估

| Method | MMD-CD↓ | MMD-EMD↓ | COV-CD↑ | COV-EMD↑ | 1-NNA-CD↓ | 1-NNA-EMD↓ | FID↓ |
|---|---|---|---|---|---|---|---|
| SynCity | 1.68 | 19.39 | 15.38 | 13.97 | 94.27 | 93.76 | 51.97 |
| **WorldGrow** | **0.96** | **12.83** | **48.99** | **48.18** | **59.66** | **64.79** | **5.43** |

- WorldGrow在外围区域FID=5.43, **甚至比Table 1的7.52还好** (说明没有error accumulation, 反而因为sample到不同区域有点diversity benefit)
- SynCity从34.69涨到51.97, degradation明显
- **重要note**: SynCity在70%的expansion attempt中**直接fail**了, Table 4只报告了successful cases! 所以真实情况更糟糕

这个实验为什么重要? 因为它直接测试了"autoregressive式生成"的稳定性 — 类似测试LM在long context下是否保持coherence。WorldGrow靠3个机制保证稳定:
1. **3/8w overlap context** 给inpainting model足够information
2. **Coarse-to-fine** 让global layout始终是"anchor"
3. **Occlusion-aware SLAT** 避免feature drift

### 4.4 Human Preference (Table 3)

91个participants, 1-5评分, 4个维度: SP (Structure Plausibility), GD (Geometry Detail), AF (Appearance Fidelity), CO (Continuity, 仅unbounded scenes)

| Method | SP | GD | AF | SP | GD | AF | CO |
|---|---|---|---|---|---|---|---|
| Text2Room | 2.07 | 1.56 | 2.07 | - | - | - | - |
| BlockFusion | - | - | - | 3.48 | 3.30 | 1.20 | 3.36 |
| TRELLIS | 2.82 | 2.26 | 2.89 | 2.15 | 2.96 | 3.33 | 2.38 |
| SynCity | 2.48 | 3.11 | 3.59 | 2.48 | 3.07 | 4.08 | 2.74 |
| **WorldGrow** | **4.48** | **4.44** | **4.33** | **4.46** | **4.37** | **4.33** | **4.69** |

WorldGrow在所有维度都4.3+, 说明human raters觉得:
- 结构合理 (4.48 / 4.46)
- 几何细节丰富 (4.44 / 4.37)
- 外观真实 (4.33 / 4.33)
- 无缝连续 (4.69)

特别CO=4.69, 是所有指标最高的 — 说明block-by-block inpainting确实做到了seamless。

### 4.5 Outdoor Generalization (Table 6)

UrbanScene3D/Shanghai subset, 10K fine + 3K coarse blocks, 每个fine block覆盖100m (室外尺度更大):

| Method | MMD-CD↓ | MMD-EMD↓ | COV-CD↑ | COV-EMD↑ | 1-NNA-CD↓ | 1-NNA-EMD↓ | FID↓ |
|---|---|---|---|---|---|---|---|
| SynCity | 0.42 | 0.41 | 6.78 | 6.35 | 95.30 | 90.00 | 93.45 |
| **WorldGrow** | 0.41 | 0.34 | 29.00 | 34.80 | 81.30 | 84.40 | **23.49** |

FID从93.45降到23.49 — **4x better**。这说明WorldGrow的方法论 (occlusion-aware SLAT + coarse-to-fine + inpainting) 不局限于indoor, generalize到outdoor也能work。

---

## 5. Limitations与Karpathy式思考

paper的Limitations section很坦诚:

1. **只在XY平面扩展, 不能沿Z轴**: 多层建筑是future work。技术上应该可行, 但需要重新设计block topology (block需要变成3D的, 不能只在水平面切片)。
2. **Data scale限制**: 3D-FRONT只有3425 houses, 远不如Objaverse的10M objects。WorldGrow的ceiling很大程度上由data决定。
3. **Block-wise trade-off**: 为了computational feasibility牺牲了fine detail。一个block内的细节分辨率有上限。
4. **无条件控制**: 目前只用一个fixed generic prompt, 没有semantic conditioning。这意味着用户不能说"生成一个有3个卧室的公寓"。

我补充几个更深的思考:

**A. 为什么不直接用autoregressive next-block?**
WorldGrow用flow matching而非autoregressive, 因为:
- autoregressive在3D scene上没有natural ordering (不像文本有left-to-right)
- flow matching可以parallel denoise整个block的latent
- 但缺点是: 不能像autoregressive那样"left-context attends to right-future"做planning
- 这就是为什么需要coarse-to-fine — coarse stage相当于"全局规划"

**B. BlockFusion的triplane vs SLAT的sparse voxel**
- BlockFusion用triplane (三个2D feature plane组合表示3D), 优点是dense可外推, 缺点是resolution固定
- SLAT用sparse voxel grid, 优点是adaptive resolution (只在surface附近dense), 缺点是extension需要inpainting而非简单concat
- WorldGrow证明了SLAT + inpainting > triplane + extrapolation

**C. 与UniLat3D的潜在融合**
paper在Discussion中提到: "promising to integrate WorldGrow into geometry-appearance unified generation models [61] (UniLat3D) for more efficient pipelines"
UniLat3D把geometry和appearance统一在single latent space, 如果WorldGrow用这种representation, 就不用分Stage 1 (structure)和Stage 2 (latent), 可以one-stage生成。

**D. 与World Models的联系**
paper在Introduction里就说: "infinite 3D world generation is foundational for developing World Models and embodied AI systems"
这个连接很有意思:
- World Models (e.g., Genie 3, DreamerV3)需要可交互的环境
- WorldGrow生成的scene是static的, 但是navigable (Figure 1底部有embodied agent演示)
- 下一步: 在WorldGrow生成的scene上训练RL agent, 这就是unbounded open-ended learning

**E. Scale considerations**
- 19×39 blocks ≈ 1800m², 一个A100 GPU, 30分钟, 13GB peak memory
- 对比: SynCity每个block 2分钟, WorldGrow每个block 20秒 (6x faster)
- 对比: BlockFusion也是block-wise, 但需要external texture synthesis, WorldGrow end-to-end
- 如果要扩展到city scale (~10km²), 大约需要5000×5000 blocks, 按当前速度需要~10天单卡 — 实际需要分布式

---

## 6. 相关reference链接

- **TRELLIS** (Microsoft, CVPR 2025): https://microsoft.github.io/TRELLIS/ | https://github.com/microsoft/TRELLIS
- **SLAT paper**: https://arxiv.org/abs/2412.01506
- **BlockFusion** (SIGGRAPH 2024): https://wyliu.com/BlockFusion/
- **SynCity** (Meta/FAIR, 2025): https://github.com/facebookresearch/syncity | https://arxiv.org/abs/2503.01169
- **DINOv2** (Meta): https://github.com/facebookresearch/dinov2
- **3D Gaussian Splatting** (Inria): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Flow Matching** (Lipman et al.): https://arxiv.org/abs/2210.02747
- **SDEdit** (Meng et al., ICLR 2022): https://arxiv.org/abs/2108.01073
- **3D-FRONT dataset**: https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future
- **Objaverse-XL**: https://objaverse.xl/
- **UrbanScene3D** (ECCV 2022): https://github.com/LordLiang/UrbanScene3D
- **UniLat3D** (Jiemin Fang et al., 2025): https://arxiv.org/abs/2509.25079
- **DiffInDScene** (CVPR 2024): https://arxiv.org/abs/2404.13293
- **Text2Room** (ICCV 2023): https://github.com/lukasHoel/text2room
- **LT3SD** (latent tree 3D scene diffusion): https://arxiv.org/abs/2404.09910
- **Hunyuan3D 2.0** (Tencent): https://github.com/Tencent/Hunyuan3D-2
- **Direct3D-S2**: https://github.com/pywu0316/Direct3D-S2
- **Infinigen** (Princeton, CVPR 2023): https://infinigen.org/
- **Project page (WorldGrow)**: https://World-Grow.github.io (从paper的header看)

---

## 7. 我的总体评价

从你的视角, 我觉得这篇paper有几个点值得深入思考:

1. **Methodological elegance**: 把object-level prior transfer到scene-level是个非trivial的domain transfer问题。WorldGrow没有简单粗暴地"直接fine-tune", 而是identification了两个具体问题 (occlusion, boundary), 给出针对性的solution。这种"identify problem → propose targeted solution → ablation verify"的pattern很教科书。

2. **Coarse-to-fine as multi-scale planning**: 类比Transformer里的"global attention vs local attention", WorldGrow用两个不同resolution的block做"global layout planning + local detail generation"。这让我想到Diffusion的cascaded generation (e.g., DeepFloyd IF,Imagen), 但在3D上做。

3. **Inpainting vs Generation**: WorldGrow把unbounded generation重新formulate为inpainting task, 这是个很巧妙的trick — 把"outpainting"问题转化为"已知大部分context, inpaint中心区域"的问题。这种reformulation让pre-trained model能用得更好, 因为inpainting是model已经擅长的task (TRELLIS-image inpainting等)。

4. **Scaling laws in 3D**: paper没有讨论, 但我觉得值得探索: WorldGrow的quality如何随block数量scale? 随training data scale? 随model size scale? 这些scaling law对于预测future capability重要。

5. **Limitations值得继续做**: Z-axis expansion + LLM-based semantic conditioning + integration with UniLat3D — 这三个方向都很有潜力, 每一个都可以是一篇follow-up paper。

如果你想build deeper intuition, 我建议从代码层面看: TRELLIS的official repo (microsoft/TRELLIS)有完整实现, WorldGrow的修改主要集中在:
- `vae/`: occlusion-aware feature aggregation (encoder端)
- `vae/decoder/`: retrained decoder
- `inpainting/`: 输入concat mask + known region
- `pipeline/`: block-by-block expansion logic

期待你的进一步想法! 这篇paper虽然不一定是final form (limitation还很多), 但它给"how to scale 3D generation to infinite worlds"提供了一个solid recipe — 而且这个recipe (block-wise + coarse-to-fine + inpainting + scene-friendly adaptation) 我相信可以transfer到video, 4D, 甚至更high-dimensional generation。
