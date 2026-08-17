---
source_pdf: PerpetualWonder.pdf
paper_sha256: 9562d4a1f635e645d32b2c499ef5eb4fb049a42f81b3bd06cc54a0cf25800a98
processed_at: '2026-08-06T02:47:14-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 PerpetualWonder

## 0. 一句话版本

你给一张照片 + 一串动作（比如"先 push 这个杯子，然后 poke 一下布，最后往沙堆吹风"），它给你生成一段 4D 视频——能从任意角度看，能跨多个动作连续交互，物理上还说得过去。之前的工作做不到连续多步，做一步就开始漂了。

项目主页：https://johnzhan2023.github.io/PerpetualWonder/

## 1. 先讲个具体场景，建立直觉

想象你给系统一张图：桌上有个铲子、旁边一座沙堡、背景是厨房。

你想做三步操作：
1. 把铲子抓起来在空中翻一圈
2. 用铲子铲进沙堡
3. 对沙堡吹一阵风

每一步动作完成之后，物体状态都变了。铲子位置变了，沙堡形状变了。下一步动作要建立在"上一步真的发生之后"的状态上。

听起来理所当然对吧？但之前的方法（WonderPlay，https://arxiv.org/abs/2506.04225，Stanford 同组前作）就做不到。它做完第一步"翻铲子"，到第二步"铲沙堡"的时候，铲子的形状已经漂得不成样了——因为它的物理状态和视觉表征是脱钩的。

## 2. 为什么 naive 的 hybrid 方法会崩

WonderPlay 的设计是：物理仿真器算出粗略的动态 → 渲染成视频 → 用 video diffusion model 把视频"美颜"成逼真的 → 用美颜后的视频优化 3D Gaussians 的外观。

问题出在：**video model 把铲子在空中的形状修得更真实了，但这个修正没法回写更新物理仿真器里的铲子状态**。物理仿真器还在用它自己算出来的、不准的铲子位置和形状。

下一轮动作来的时候，物理仿真器拿着"旧的、不准的"铲子状态继续算。铲子在物理仿真里可能已经歪了、变形了，video model 这次基于这么离谱的输入做美颜，再怎么修也救不回来。三轮之后，铲子就成了一坨乱七八糟的 Gaussians。

这就像你写代码，main loop 里有个 bug 把 state 改坏了，但是 UI 上看不出来（因为 UI 有自己的修正），你以为 state 是对的，下一轮继续用坏 state 算，越算越歪。

核心症结：**信息流是单向的**。物理驱动视觉，视觉没法回头改物理。

## 3. PerpetualWonder 的核心招：VPP

VPP = Visual-Physical Aligned Particle。

直觉是这样：之前的方法里，physics particles 和 3D Gaussians 是两套独立的东西，physics particles 驱动 Gaussians，但 Gaussians 的优化结果没法影响 physics particles。

VPP 把它们"绑"起来。具体怎么绑：每个 physics particle 锚定 K 个 Gaussians，Gaussians 的位置 = physics particle 位置 + 一个小的 offset，这个 offset 用 tanh 压缩在 particle 半径 δ 之内。

用大白话讲：**每个物理粒子挂着一串视觉小铃铛，铃铛可以晃，但晃不远，被橡皮筋拴在粒子上**。

这么绑之后发生两件事：

**正向（physics → visual）**：物理仿真器移动粒子，所有挂在它身上的铃铛跟着移动。这就是传统的"物理驱动视觉"。

**反向（visual → physics）**：video model 优化铃铛位置（因为视觉 loss 需要），但铃铛被橡皮筋拴着，所以优化完之后，铃铛们的平均位置就是粒子"应该"在的位置。把这个平均位置写回 physics particle，下一轮物理仿真就用这个修正后的位置开始算。

这就是闭环。每一轮 backward pass 都用 video model 的 prior 修正上一轮物理仿真的误差，error 不会单调累积。

## 4. 为什么橡皮筋是 tanh，不是 hard constraint

这里有个微妙的设计选择。如果硬约束 Gaussians 位置等于 particle 位置（K=1，offset=0），那就退化成 PhysGaussian（https://arxiv.org/abs/2311.12983），表达力不够——比如烟、水这种体积效果，单个粒子挂一个 gaussian 表达不出来。

如果完全不约束（offset 自由学习），那就是标准 3DGS，视觉优化会乱跑，物理约束失效。

tanh 是中间道路：允许铃铛在粒子周围 δ 范围内自由微调，表达体积细节、烟雾扩散这种 physics 仿真精度不够的东西；但又被限制不能跑远，保证物理 anchor 还在起作用。

这是一个 representation design 的经典 trade-off：**约束太强 = 表达力不够，约束太弱 = 优化空间太大会 degenerate**。

## 5. K 的自适应：solid 用 1，体积物体用 20

这个细节在 supplementary 里，但其实是 VPP 能 work 的关键工程点。

- Rigid body、cloth 这种表面/固体材料：K=1，一个粒子挂一个 gaussian，scale = δ。一对一严格绑定，防止大变形时出现 ghosting（视觉残影跟物理不同步）。
- Gas、liquid、sand、snow、elastic 这种体积/发射型材料：K=20，一个粒子挂 20 个小 gaussian，scale = 0.5δ。把一个物理粒子"雾化"成 20 个视觉小点，表达体积扩展和半透明。

为什么这么分？因为物理粒子的分辨率和视觉需要的分辨率 mismatch。固体表面，物理粒子和视觉点一一对应就够了。但烟雾这种，物理上你算的是几个离散粒子的运动，视觉上你需要一团连续的雾——一个物理粒子必须"撑开"成一片 gaussian 云。

这跟 SPH（smoothed particle hydrodynamics）里 kernel function 的精神类似，只不过这里 kernel 是可学习的 Gaussians。SPH 参考：https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics

## 6. Multi-view 的问题：video model 跨视角不一致

有了 VPP，理论上可以闭环了。但实际操作有个大坑：**video diffusion model 从不同视角生成的视频彼此不一致**。

比如你让它从正面看苹果生成一段视频，它可能 hallucinate 苹果上有个红斑。从侧面看生成视频，它没 hallucinate 那个斑。两个视频都是"合理的"，但彼此冲突。

如果你直接拿这两个视频同时优化 3D scene，优化器不知道信谁，结果就是两边都模糊化——苹果纹理糊掉，时间上闪烁。

WonderPlay 之所以只用 single view，就是因为这个坑。但 single view 又有 multi-view ambiguity——一个视角的 2D 监督无法唯一确定 3D 表征，从侧面看就漂了。

## 7. PerpetualWonder 的招：progressive multi-view

三步走：

**Stage 1**：只用 input image 那个原始视角。渲染、refine、优化。这一步建立最 reliable 的 anchor，因为 input view 是 ground truth 视角。

**Stage 2**：渲染其他视角（左、右），用更小的 control weight 做 refinement。这里的"control weight"是指 video model 的 bimodal control（RGB + optical flow）里，optical flow 的引导强度调低，让 video model 自由度大一点，但也意味着它生成的视频更"软"，不会跟 Stage 1 的 anchor 硬碰硬。

**Stage 3**：所有视角的 refined videos 一起优化。这时候 VPP 的物理 anchor 已经稳定了，能吸收跨视角的小冲突。

直觉：**先用最可靠的单视角把基础锁定，再渐进引入 noisy 的多视角监督，让物理约束逐步消化冲突**。

这跟 curriculum learning、progressive GAN training 的精神相通——别一上来就给优化器最难的问题，先从简单的开始。

## 8. 3D 场景初始化：用 GEN3C 造 dense views

这部分相对独立，但是 multi-view 能力的基础。

WonderPlay 用 single-view depth unprojection（https://arxiv.org/abs/2406.09394 WonderJourney），就是估个深度图把像素反投影到 3D，再加 inpainting 补全。结果只能从窄 baseline 看，稍微转一下角度就崩。

PerpetualWonder 用 GEN3C（https://gen3c.github.io/，NVIDIA 的工作）从单张图生成 242 个 dense surrounding views。具体策略是分两次：arc left 90° + arc right 90°，避免一次性 180° 大转动让 GEN3C 失败。

这 242 views 喂给 COLMAP（https://github.com/colmap/colmap）做 SfM 重建 point cloud，再做 3DGS 优化，得到完整的 3D scene。

接着用 SAM2（https://github.com/facebookresearch/sam2）+ Gaussian Grouping（https://github.com/lkeab/gaussian-grouping）分割前景物体。前景物体的 Gaussians 转 mesh（TSDFusion），rigid body 还额外用 Hunyuan3D（https://github.com/Tencent/Hunyuan3D-2）重新生成更干净的 mesh。Mesh 上采样 physics particles，再绑 VPP。

这一套下来，你拿到一个完整的、可以从任意视角渲染的、物理粒子已经摆好的初始 3D scene。

## 9. 完整的闭环 loop

```
每一轮 time window（T 步，392 物理步，49 视频帧）：

1. Forward pass（<1 min）：
   物理仿真器从 S_0 开始，按 action A_0, A_1, ..., A_{T-1}
   一步步算，得到粗糙的 4D 序列 {Ŝ_t}
   
2. Backward optimization（~7 min）：
   a. 把 {Ŝ_t} 渲染成 RGB + optical flow 视频
   b. Video diffusion model (Go-With-The-Flow) refine 视频
   c. Progressive multi-view 优化 VPP 的 Gaussians
      loss = photometric loss + λ * simulation consistency loss
   d. 得到 refined {S_t}
   
3. Loop closure：
   p_j_next = mean(挂在这个 particle 上的 K 个 Gaussians 位置)
   v_j_next = v_j_T (直接继承)
   这就是下一轮的 S_0
```

整个 loop 跑下来 16 分钟一轮，3 个 windows 大概半小时。

## 10. 为什么 velocity 可以直接继承

这是个工程近似，paper 里轻描淡写带过，但其实有意思。

严格说，loop closure 时你应该用 finite difference 重新估计 velocity：
$$v_T^{\text{next}} = \frac{p_T^{\text{next}} - p_{T-1}^{\text{next}}}{\Delta t}$$

但 paper 直接继承原 velocity。为什么能 work？

因为 tanh 软约束 + simulation consistency loss 把位置修正限制在小范围内，position 变化不大，所以原 velocity 还算合理。如果位置修正幅度大（比如 video model 大幅改了 dynamics），这个近似就会崩。但 paper 的实验场景里，物理仿真本身已经给出了大致正确的 dynamics，video model 只是 refine 细节，所以位置修正幅度小，velocity 继承 OK。

这是一个"信任物理仿真给出的大方向，video model 只做局部修正"的隐含假设。如果未来要处理物理仿真完全 wrong 的情况（比如物理仿真器不知道物体应该碎掉），这个近似就需要改。

## 11. 实验结果讲了什么

### 定量（World-Score，https://worldscore.github.io/）

- **Camera controllability 93.26**：断层第一。因为表征是 3DGS，可以从任意视角渲染，camera ctrl 天然好。所有纯 video diffusion 方法（Wan、Veo、Tora）都在 50-65，因为 video model 把 camera 编码在 attention 里，误差积累。
- **3D consistency 80.41**：明显领先。WonderPlay 只有 63.93，因为单视角优化必然 novel view 不一致。
- **Imaging 66.98**：跟 Veo3.1（67.82）、Wan2.2（67.03）差不多。因为 per-frame 质量主要由 video model backbone 决定，PerpetualWonder 用的也是同一个 backbone。这个 metric 上不会断层领先。
- **WonderPlay Imaging 36.80**：异常低，说明它的 single-view 优化产生了严重伪影，基本宣告方法不可用。

### User study（350 人，2AFC）

- vs Veo3.1：62% / 70.8%。Veo3.1 是 Google 最强 video model（https://arxiv.org/abs/2509.20328），能从文本直接生成 plausible dynamics，所以这是 PerpetualWonder 优势最小的地方。但 62% 仍说明 hybrid 物理仿真在 physics plausibility 上有优势。
- vs WonderPlay：80.8% / 86.3%。VPP + multi-view 的贡献用户感知层面也显著。
- vs GEN3C：93.5% / 83.5%。GEN3C 不能响应 action，物体静止，所以 physics plausibility 接近完全胜出。

## 12. Ablation 说明了什么

### VPP vs 标准 3DGS

去掉 VPP 约束，用标准 3DGS 做 multi-view 优化：结果 degenerate，gaussians 乱跑，dynamics 混乱，visual artifacts 满天飞。

这证明：**单纯的视觉 loss 在 multi-view 4D 优化下不足以约束 dynamics**。必须有 physics anchor 提供归纳偏置。这跟为什么 BERT 加 MLM 比纯 LM 难训好——inductive bias 很重要。

### Progressive vs Direct multi-view

直接用所有视角同时优化：frontal view 说苹果有红斑，side view 说没有，优化器两边都糊掉，时间上闪烁。

Progressive：先 single view 锁基础，再渐进引入多视角，矛盾被物理 anchor 吸收，结果干净。

### Isotropic vs Anisotropic Gaussians

Anisotropic 容易 overfit input view，novel view 模糊。Isotropic 更干净。这跟原版 3DGS 用 anisotropic 不一样，因为 3DGS 是 reconstruct 单一静态场景，anisotropic 提供更多表达力；4D generation 视角有限且 video model 有 hallucination，anisotropic 容易过拟合。

## 13. 跟 Sora 路线的对比

Sora（https://openai.com/research/video-generation-models-as-world-simulators）把 video model 直接当 world simulator，纯 neural 路线。OpenAI 自己也承认 Sora 物理不一致。

PerpetualWonder 走 hybrid 路线：物理仿真保证 consistency + controllability，video model 保证 realism。

两条哲学的根本分歧：
- **Sora 路线**：相信 scale，相信 video model 学到足够多数据后能 implicit 建模物理。优点是 end-to-end，缺点是 long-horizon 误差累积、controllability 差。
- **Hybrid 路线**：相信结构化先验（物理仿真）+ neural prior（video model）分工。优点是 controllable、long-horizon 稳定，缺点是表征设计复杂、运行慢。

我个人觉得 hybrid 路线在 embodied AI 应用上更靠谱，因为 robot learning 这种下游任务需要 controllability。纯 neural world model 在 generation 任务里 OK，在 control 任务里 hallucination 会致命。

但 hybrid 路线的代价是慢（16 min/loop）。如果未来 video model 的 long-horizon 一致性问题被解决（RL training、更长 context、更好的 architecture），纯 neural 路线可能反超。

## 14. 失败模式与未来方向

### 失败模式

Supplementary G 提到：hockey stick 从画面外进入，中间 frame 中 stick 看起来不完整（应该更长）。因为 input image 没看到完整几何，GEN3C 也没补上。

这是 single-image 4D generation 的根本限制：**unseen geometry completion 是 ill-posed**。你只看到物体一面，背面什么样靠 hallucinate，可能错。

可能解法：
- 用大型 3D generative model（TRELLIS、Hunyuan3D）做 object completion
- 用 video LLM 的 in-context reasoning 推断 unseen geometry
- 多模态输入（text + image）给完整 object description

### Future directions 我能想到的

1. **Differentiable physics**：把 forward pass 也变成可微的，backward pass 可以同时优化物理参数（friction、Young's modulus）。目前物理参数是 GPT-4o 估计 + 手动调，如果可微就能 end-to-end 学。

2. **Active view selection**：不是固定 frontal/left/right 三个视角，而是基于当前 4D scene uncertainty 选 next view。比如哪个区域 Gaussians 方差大，就多渲染那个区域的视角。这跟 active learning、Bayesian optimal experimental design 同源。

3. **Object-level VPP bundle**：每个 object 独立 VPP，支持 object insertion/removal in long horizon。目前场景是 fixed object set，不能"中途加个新物体"。

4. **Real-time**：用 3DGS fast rasterizer + 蒸馏 video model 到 lightweight refiner。目前 16 min/loop 离 real-time 差三个数量级。

5. **Robot learning integration**：用 PerpetualWonder 生成 synthetic training data for manipulation policies。类似 SplaSim（https://arxiv.org/abs/2505.04489）但带 dynamics。这对 embodied AI 是 big deal——现在的 sim2real gap 很大程度是因为 simulator 不真实。

6. **Multimodal actions**：除了 force，加 language-conditioned actions（"pour the water"、"cut the cloth"）。需要把 language 解析成 force sequences，可能用 VLM 做 planning。

7. **VPP 的 fiber 推广**：目前 fiber 是 K 个 Gaussians，可以推广成每个 particle 挂一个 small MLP，表达更复杂的局部变形。这跟 NeRF 的精神结合，但保持 bundle structure。

## 15. 一些更深的联想

### 15.1 VPP 跟 gauge theory 的结构同构

VPP 实际上定义了一个 bundle structure：
- Base space：physics particle manifold $\{p_j\}$
- Fiber：K 个 Gaussians 的属性空间
- Connection：tanh 软约束，限制 fiber 相对 base 的偏移

这跟物理里的 gauge theory（https://en.wikipedia.org/wiki/Gauge_theory）结构同构。Base 是时空，fiber 是 internal degrees of freedom，connection 是 gauge field。

虽然 paper 没用这种语言，但这种结构暗示了 VPP 可以推广到更复杂的 fiber——比如每个 particle 挂一个 small NeRF，或者挂一个 deformation field。Bundle structure 保证了"局部修改不影响全局一致性"，这是 long-horizon 稳定的数学基础。

### 15.2 Closed loop 跟 MPC 的同构

PerpetualWonder 的闭环结构跟 Model Predictive Control（MPC，https://en.wikipedia.org/wiki/Model_predictive_control）很像：
- Forward pass = MPC 的 forward model prediction
- Backward optimization = MPC 的 measurement-based state correction
- Loop closure = MPC 的 state estimation update

差别在于 MPC 的 measurement 是真实传感器数据，PerpetualWonder 的"measurement"是 video model 的 refinement。Video model 在这里充当了一个"软传感器"——它告诉你"基于先验，这个场景应该长这样"。

这个视角暗示可以借用 MPC 的理论来分析 PerpetualWonder 的稳定性、收敛性。比如能不能证明在某种意义上 closed-loop error 有界？

### 15.3 Progressive multi-view 跟 Bayesian update 的类比

Progressive 策略可以看作一个简化的 Bayesian update：
- Stage 1：prior 是 single view 的 refined video，confidence 高
- Stage 2：加入小 weight side views，相当于 noisy measurements，Bayesian update 缓慢
- Stage 3：置信度稳定后，所有 views 一起 average

这个类比可以推广——可以想象一个 **active view selection** 策略，根据当前 uncertainty 选 next view。哪些区域的 Gaussians 位置方差大？哪些视角能最大减少这个方差？这就是 Bayesian optimal experimental design（https://en.wikipedia.org/wiki/Bayesian_experimental_design）。

### 15.4 跟 differentiable simulation 的关系

目前 forward pass 用 Genesis（https://github.com/Genesis-Embodied-AI/Genesis），不可微。backward pass 只优化视觉表征，不优化物理参数。

如果换成 differentiable physics simulator（比如 Brax、DiffTaichi），就能让 backward pass 同时优化：
- 视觉表征（Gaussians）
- 物理参数（friction、Young's modulus、mass）
- 甚至 action 本身（optimal control）

这会让系统变成一个完整的 differentiable hybrid simulator，能做 system identification、model-based RL。这是我个人最期待的未来方向。

### 15.5 跟 world model 的边界

PerpetualWonder 是 world model 吗？取决于定义。

如果 world model = "能预测 action 后果的 generative model"，那它是。但它不是 end-to-end neural world model，是 hybrid（symbolic physics + neural refiner）。

Yann LeCun 的 JEPA（https://openreview.net/forum?id=BZ5a1r-kVsf）走的是另一个路线：在 latent space 做预测，不生成像素。优点是计算便宜、抽象层次高，缺点是没法直接 render 给人看、下游用起来不方便。

我个人觉得 world model 这个词被滥用了。Sora 是 world model 吗？某种意义上是，但它没有 explicit action interface，也不保证物理一致。PerpetualWonder 更接近"generative physics simulator"——它有 explicit physics、explicit action、explicit state，只是 appearance 部分 generative。

如果未来要给 robot 用，PerpetualWonder 这种结构化 world model 可能比纯 neural world model 更靠谱，因为：
1. 物理一致性保证（关键 for safety）
2. Action interface 明确（force/torque）
3. State 可解释（particle position/velocity）
4. 但 appearance 部分够 real（不像传统 simulator 有 realism gap）

### 15.6 跟 RL 的 synthetic data 生成

这是个很有想象力的应用方向。

现在 RL 训练 robot policy 的痛点是 data expensive。Sim2real 又有 gap。PerpetualWonder 这种 hybrid 系统可能是个 sweet spot：
- 用物理仿真器保证 action-response 的因果正确（重要 for policy learning）
- 用 video model 保证视觉真实（重要 for perception module）
- 生成的数据可以直接训 visuo-motor policy

类似 SplaSim（https://arxiv.org/abs/2505.04489）的思路，但带 dynamics。SplaSim 用 3DGS 做 sim2real，但它是 static scene manipulation，没 dynamics。PerpetualWonder 加上 dynamics，能训更复杂的 policy（push、pour、cut 这种）。

我觉得这是这篇 paper 最有 impact 的潜在方向，虽然 paper 自己没强调。

### 15.7 表征设计的哲学

VPP 反映了一个 deep 的 representation design 哲学：**好表征应该在"结构化先验"和"neural 灵活性"之间找平衡**。

完全结构化（传统 simulator）：可控、可解释，但不真实。
完全 neural（video model）：真实、灵活，但不可控、long-horizon 不稳定。
Hybrid：结构化提供骨架，neural 提供 flesh。

VPP 的 tanh 软约束就是这种哲学的具体体现——既不是 hard binding（太僵），也不是 free（太野）。橡皮筋的张力（δ 大小）调节了结构化与灵活性的比例。

这个哲学在很多地方都出现：
- ResNet 的 skip connection：结构化（identity）+ neural（residual）
- Neural ODE：结构化（ODE solver）+ neural（learned dynamics）
- Diffusion model：结构化（Gaussian noise schedule）+ neural（denoiser）

PerpetualWonder 的 VPP 是这个哲学在 4D generation 领域的实例。

### 15.8 跟 NeuroSymboic AI 的关系

更宏观地看，PerpetualWonder 是 NeuroSymbolic AI（https://en.wikipedia.org/wiki/Neuro-symbolic_AI）的一个 instance：
- Symbolic 部分：physics simulator（MPM、PBD 这些经典 solver）
- Neural 部分：video diffusion model
- 接口：VPP（视觉-物理对齐粒子）

NeuroSymbolic AI 的核心思想是结合 symbolic reasoning（可控、可解释）和 neural learning（灵活、数据驱动）。PerpetualWonder 用 VPP 做 interface，让 symbolic physics 和 neural video model 双向通信。

这种思路在 robotics、reasoning、tool use 里都有出现。比如 Voyager（https://arxiv.org/abs/2305.16291）用 LLM 写 code 控制 Minecraft agent，code 是 symbolic，LLM 是 neural。PerpetualWonder 用 VPP 让 physics simulator 和 video model 通信，physics 是 symbolic，video model 是 neural。

### 15.9 Potential 跟 LLM agent 的结合

想象一个 LLM agent 操作 3D world：
- LLM 决定 action（"把杯子推到桌子边缘"）
- PerpetualWonder 执行 action，生成 4D scene
- LLM 看生成的 scene，决定下一步 action

这就实现了 embodied LLM agent。目前的 embodied agent 大多在简化环境（ALFWorld、Habitat）里跑，视觉不真实。PerpetualWonder 提供了 realistic 4D environment，能让 LLM agent 在更真实的世界里学习。

这跟 Sora 当 world model for agent 的思路类似，但 PerpetualWonder 的物理一致性更适合需要精确 control 的任务。

### 15.10 Computational efficiency 的未来

目前 16 min/loop，离 real-time 差三个数量级。瓶颈在：
1. Video diffusion model inference（每帧几百 ms）
2. 3DGS optimization（几千次迭代）

可能的加速方向：
- **Video model distillation**：把 video diffusion model 蒸馏成 lightweight refiner（类似 LCM、SDXL Turbo 的思路）
- **3DGS optimization 加速**：用 procedural initialization、warm-start、low-rank adaptation
- **Adaptive refinement**：不是每帧都 refine，只在 dynamics 变化大的时刻 refine
- **Differentiable rendering + 物理联合优化**：避免 forward/backward 两次 pass

如果这些优化做到 real-time，PerpetualWonder 就能用于 interactive applications（VR/AR、gaming、robot teleoperation）。

## 16. 总结：intuition 的最简版本

**问题**：之前的 hybrid simulator 信息流单向，物理→视觉，但视觉→物理不通，long-horizon 误差累积。

**核心解法**：VPP 用 tanh 软约束把 physics particles 和 visual Gaussians 绑成 bundle，让视觉优化结果可以回写更新物理状态，实现闭环。

**工程难点**：video model 跨视角不一致，用 progressive multi-view optimization 逐步消化冲突。

**为什么 work**：tanh 软约束 + simulation consistency loss 让视觉只能在物理 anchor 周围微调，既保留修正能力又防止 degenerate。每一轮 backward pass 用 video prior 修正物理误差，error 不再单调累积。

**更大意义**：这是 hybrid generative simulator 从"单向"到"闭环"的关键一步，给 long-horizon action-conditioned 4D generation 开了条路。潜在 impact 在 embodied AI、robot learning、world model 这些方向。

希望这个人话版本帮你 build intuition，Karpathy！如果某个部分还想 deep dive（比如 VPP 的优化几何性质、video model 的 bimodal control 具体怎么实现、或者跟 differentiable simulation 的结合点），告诉我继续展开。

---

# PerpetualWonder: Long-Horizon Action-Conditioned 4D Scene Generation 深度解析

## 1. 核心直觉与定位

这篇 paper 来自 Stanford 的 Jiahao Zhan, Zizhang Li, Hong-Xing Yu, Jiajun Wu,定位是 **从单张图片生成长 horizon、action-conditioned 4D 场景** 的 hybrid generative simulator。任务定义本身很清晰:给定单张 input image $I$ 与一系列动作 $\{\mathcal{A}_t\}_{t=0}^{T-1}$ (local forces 如 push/poke,global forces 如 wind/gravity),输出 4D 场景序列 $\{S_t\}_{t=0}^{T}$。

关键 insight 是:之前 WonderPlay (ICLR'25, 同组工作, https://johnzhan2023.github.io/PerpetualWonder/) 这类 hybrid generative simulator 的物理状态 (physics particles) 与视觉表征 (gaussian splatting primitives) 是 **decoupled** 的,导致一个单向信息流——物理仿真驱动视觉,而视频生成模型的 refinement 无法回写更新物理状态。这就只能做单窗口短时交互,在 multi-step long-horizon 场景中误差会累积,表现为 object shape fracture、geometry drift。

PerpetualWonder 想做的是 **第一个真正的 closed-loop hybrid generative simulator**:让 generative refinement 既修正 appearance 又修正 dynamics,并能将修正后的状态作为下一轮的 initial condition。

这里"long-horizon"在他们实验里实际是 3 个 time windows,每个 window 392 个物理仿真步,49 视频帧 (每 8 物理步采 1 帧)。所以总 horizon 大约是 3 × 49 = 147 帧的视觉序列 + 1176 个物理步。这是相对"长"的范畴,但和真正的 embodied AI 持续交互还差几个量级。

参考链接:
- 项目主页:https://johnzhan2023.github.io/PerpetualWonder/
- WonderPlay (前作):https://news.stanford.edu/ (隐含 reference [21])

## 2. 核心问题的精确数学表述

把 hybrid generative simulator 形式化为两个算子:
- **Forward physics pass** $\Phi_p$:传统物理求解器,$\hat{S}_{t+1} = \Phi_p(\hat{S}_t, \mathcal{A}_t)$,产生 coarse dynamics。
- **Backward neural optimization** $\Psi_n$:用预训练 video diffusion model 作 refiner,对 render 出来的 RGB + optical flow 视频做 refinement,再通过 photometric loss 回写更新表征。

WonderPlay 的信息流:
$$\Phi_p \rightarrow \text{render} \rightarrow \text{video diffusion refiner} \rightarrow \text{photometric loss on } \mathcal{G}_t \rightarrow \text{update only Gaussians}$$

物理状态 $\mathcal{P}_t, \mathcal{V}_t$ (particle position/velocity) 在 backward pass 中没有任何梯度路径接收。这意味着下一轮 forward pass 用的是 **未修正** 的物理状态。误差源主要有两类:
1. 物理仿真本身的 approximation error (simplified material models, fixed friction, 不准确 Young's modulus 等)。
2. 视频模型对 dynamics 的"创造性修正"(比如生成的水花、烟雾细节超出物理仿真精度)。

这两类 error 在 long-horizon 下叠加,所以 WonderPlay 的铲子在空中翻一圈之后,插回沙堡时 shape 已经漂得不成样了。

PerpetualWonder 想造一个 **bidirectional bridge**:让 $\mathcal{G}_t \to \mathcal{P}_t$ 的回写路径存在。难点是 Gaussians 与 physics particles 数量、坐标、属性都不同,需要一个统一表征。

## 3. Visual-Physical Aligned Particle (VPP) 详解

VPP 是这篇 paper 最关键的贡献。核心设计:**每个 physics particle 锚定 K 个 gaussian primitives**,形成一对多绑定。

### 3.1 表征定义

对每个 object $o$ (省略 $o,t$ 下标):
- Physics dynamics: $\mathcal{P} = \{p_j\}_{j=1}^J$,$\mathcal{V} = \{v_j\}_{j=1}^J$。其中 $p_j \in \mathbb{R}^3$ 是 particle position,$v_j \in \mathbb{R}^3$ 是 velocity,$J$ 是 particle 数量。
- Visual appearance: $\mathcal{G} = \bigcup_{j=1}^J \{g_{j,k}\}_{k=1}^K$,$K$ 是 anchor 数 (每 particle 挂几个 gaussian)。

### 3.2 Gaussian 参数化

**位置偏移** (公式 1):
$$\mu_{j,k} = p_j + \tanh(\tilde{p}_{j,k}) \cdot \delta$$

变量解释:
- $\mu_{j,k} \in \mathbb{R}^3$:第 $j$ 个 physics particle 锚定的第 $k$ 个 gaussian 的最终 3D 位置。
- $p_j \in \mathbb{R}^3$:physics particle 的位置,由物理仿真器更新。
- $\tilde{p}_{j,k} \in \mathbb{R}^3$:可学习的 position offset (优化目标)。
- $\delta$:physics particle 的采样 size,由 simulator 采样过程定义,作为软约束的"半径"。
- $\tanh(\cdot)$:把 offset 限制在 $(-\delta, \delta)$,确保 gaussian 始终在 particle 的"影响球"内,不能漂走。

这是一个非常聪明的设计。比起直接学习绝对位置 $\mu_{j,k}$,这种 parameterization 让 backward optimization 只能在 particle 周围 $\delta$ 半径内微调,既保留修正能力,又防止 visual primitives 与 physics anchor "解绑"——这就是 VPP 中 "aligned" 的数学含义。

**Scale**:isotropic,$\le \delta$。这里 paper 用 isotropic 而不是 anisotropic,在 supplementary E 里有 ablation 说明——anisotropic 在 novel view 容易 overfit 输入视角,产生模糊伪影。

**Spatio-temporal opacity** (公式 2):
$$o_t(t) = \exp\left(-\frac{1}{2} \cdot \left(\frac{t - \mu_t}{s_d}\right)^2\right)$$

变量解释:
- $o_t(t) \in [0,1]$:第 $t$ 时刻的 temporal opacity。
- $\mu_t$:gaussian 的"中心时间",可学习。
- $s_d$:gaussian 在时间维度的"duration"(标准差),可学习。
- 这本质是一个时间维上的高斯窗,意思是每个 gaussian 只在某个时间窗内"激活"。

最终 opacity $o(t) = o_s \cdot o_t(t)$,$o_s$ 是标准 3DGS 的 learnable spatial opacity。

这个设计参考 FreeTimeGS (CVPR 2025, reference [42]),目的是让 4D 中的每个时空"事件"用不同的 gaussian 表示,而不是让一个 gaussian 永久存在——这对 dynamics refinement 至关重要,因为视频模型会生成新出现/消失的元素(如水花、烟雾),没有 temporal opacity 就只能用 spatial opacity 表达时间信息,表达力严重不足。

### 3.3 K 的自适应配置 (Supplementary D)

这个细节很关键,paper 在 supplementary 给了根据 material 自适应的 K 值:
- **Solid/Surface** (rigid, cloth): $K=1$,gaussian scale $= \delta$。一对一绑定,严格对齐,防止 ghosting/detachment。
- **Volumetric/Emitter** (gas, liquid, sand, snow, elastic): $K=20$,gaussian scale $= 0.5\delta$。单 particle 覆盖更大视觉体积,表达半透明体积效果与细粒度细节。

这反映了"physics-to-vision resolution mismatch"问题:物理 particle 是宏观离散点,而视觉上像烟这种体积效果需要细密表达。$K=20$ 实际上是把一个粒子"雾化"成 20 个小 gaussian,允许在物理约束内表达体积扩展。这跟 SPH (smoothed particle hydrodynamics) 里 kernel 函数 的精神很像,只不过这里是用可学习的 gaussian primitives 充当 visual kernel。

### 3.4 为什么 VPP 能闭环

backward optimization 对 $\tilde{p}_{j,k}, o_s, \mu_t, s_d, q_{j,k}, c_{j,k}$ 等做梯度更新,得到 refined $\{\mu_{j,k,T}\}$。Loop closure 时:
$$p_j^{\text{next}} = \frac{1}{K} \sum_{k=1}^K \mu_{j,k,T}$$

即对每个 particle 锚定的 K 个 gaussian 取位置均值作为下一轮 particle 的初始位置。由于 $\tanh$ 软约束 + $\mathcal{L}_{sim}$ (下节) 的强 regularizer,这个均值不会偏离 $p_j$ 太远,所以可以直接继承 velocity $\mathcal{V}_T$。这就是 **bidirectional bridge 的具体含义**——视觉优化可以回写更新物理状态。

## 4. Multi-View Optimization Mechanism 详解

### 4.1 为什么单视角不行

WonderPlay 用单视角的视频做 refinement,优化时只看见 frontal view 的 photometric loss。结果是从其他 novel view 渲染时出现严重 geometry inconsistency——3D Gaussian 的位置在 frontal view 看着对,但从侧面看就"飘"了,因为没有跨视角监督。

这是一个经典的 **multi-view ambiguity** 问题:单视角 2D 监督无法唯一确定 3D 表征,优化空间里有很多 false minimum。

### 4.2 3D 场景初始化

这部分是相对独立的工程模块,但决定了后续 multi-view 能力。

流程:
1. 用 GEN3C (reference [34], NVIDIA + Toronto 的工作) 从单张图生成 dense surrounding views (具体在 supplementary A:分成 "arc left" + "arc right" 两个 90° 轨迹,避免直接 180° 大转动导致 GEN3C 失败)。
2. 242 views 喂给 COLMAP (reference [35]) 做 SfM,得 point cloud。
3. 3DGS 优化得 $\{G_i\}_{i=1}^N$,每个 $G_i = (p_i, q_i, s_i, o_i, c_i)$。
4. SAM2 (reference [31]) 在 dense views 上分割 objects → Gaussian Grouping (reference [52]) 用 learnable feature $g_i$ 监督。
5. Foreground Gaussians → TSDFusion (reference [55]) → mesh。
6. Rigid 物体用 Hunyuan3D (reference [61]) 重新生成 mesh,6-DoF pose + scale 通过投影误差最小化自动对齐到 scene。
7. Mesh 上采样 physics particles $\mathcal{P}_0$。
8. 第二轮 3DGS 优化对 $\{\mathcal{G}_0^o\}_{o=1}^O$ 做 VPP 绑定。

对比 WonderPlay:WonderPlay 用单视角 depth unprojection (WonderJourney/WonderWorld, reference [53,54]) + 手动物体摆放。两者在 3D scene 表征质量上有本质差距——PerpetualWonder 的初始化支持任意视角渲染,WonderPlay 的初始化只在窄 baseline 内可见。

### 4.3 Loss Function (公式 3)

$$\mathcal{L} = \mathcal{L}_p(\text{Render}(\mathcal{B}_t) \odot (1-\mathbf{M}), \mathbf{V}_t \odot (1-\mathbf{M})) + \mathcal{L}_p(\text{Render}(\mathcal{G}_t), \mathbf{V}_t \odot \mathbf{M}) + \lambda_{sim} \mathcal{L}_{sim}$$

变量解释:
- $\mathcal{B}_t$:背景 Gaussians (也用 spatio-temporal opacity,但 position 不学习——这处理 secondary effects like shadows on background)。
- $\mathcal{G}_t$:前景 VPP primitives。
- $\mathbf{M}$:前景 binary mask。
- $\mathbf{V}_t$:refined video 第 $t$ 帧。
- $\text{Render}(\cdot)$:3DGS 渲染函数。
- $\mathcal{L}_p$:photometric loss = L1 + SSIM,标准 3DGS loss。
- $\odot$:element-wise 乘 (mask 应用)。
- $\lambda_{sim}$:simulation consistency loss 权重。

第一项是背景重建 loss,第二项是前景 VPP 渲染 loss,第三项是物理一致性正则。

### 4.4 Simulation Consistency Loss (公式 4)

$$\mathcal{L}_{sim} = \frac{1}{T \cdot J} \sum_{t=1}^T \sum_{j=1}^J \left\| p_{j,t} - \frac{1}{K} \sum_{k=1}^K \mu_{j,k,t} \right\|_2^2$$

变量解释:
- $T$:时间窗长度。
- $J$:particle 数量。
- $p_{j,t}$:第 $t$ 时刻第 $j$ 个 physics particle 位置 (由 $\Phi_p$ 给定,这次 backward pass 不更新)。
- $\mu_{j,k,t}$:第 $j$ particle 锚定的第 $k$ 个 gaussian 在时刻 $t$ 的位置。
- $\frac{1}{K}\sum_{k=1}^K \mu_{j,k,t}$:第 $j$ particle 锚定的所有 gaussian 的位置均值。
- $\|\cdot\|_2^2$:L2 范数平方。

直觉:**这个 loss 惩罚 gaussians 的"质心"偏离它的 physics anchor**。这是 VPP 设计的强 regularizer——视觉优化不能把 gaussian 拖离物理 particle 太远。

这跟 STAS (Dynamic 3D Gaussians, CVPR'24) 和很多 deformation-based 4D reconstruction 工作里的 "as-rigid-as-possible" 类约束同源,但这里 anchor 是物理仿真出来的 particle,而不是某个 reference frame 的位置。这是物理约束直接进入优化目标的形式。

注意这里 paper 在公式描述上有个微妙点:$p_{j,t}$ 在 backward pass 中是 **固定** 的 (stop gradient,因为它是 physics 给出的 ground truth 状态),但 backward pass 优化的结果是 refined $\mu_{j,k,t}$,然后通过 loop closure 把均值作为下一轮的 $p_j$。所以这个 loss 是 "**当前窗口内 visual 不能偏离 physics**" 的软约束,而不是说 visual 必须等于 physics——visual 可以在 particle 周围 $\delta$ 内微调,表达 physics 未能捕捉的细节。

### 4.5 Progressive Multi-View Optimization Strategy

这是解决"多视角视频本身不一致"的工程方法。Video diffusion model 不会从不同视角生成 perfectly consistent videos——frontal view 可能 hallucinate 某个颜色,而 side view 没有,直接用所有视角同时优化会导致冲突。

三步 progressive 策略:
1. **Stage 1**:只从 input image 视角渲染 + refine,用单一视频优化。
2. **Stage 2**:从其他视角渲染,用更小 control weight 做 refinement (bimodal control: RGB + optical flow)。
3. **Stage 3**:用所有视角的 refined videos 一起优化。

直觉:先用最 reliable 的 single view 建立 anchor,再逐渐引入 noisy 的多视角监督,让 $\mathcal{L}_{sim}$ 的强约束逐步"消化"跨视角冲突。这个策略类似 coarse-to-fine 的优化,但维度是 "view count" 而非 "resolution"。

实验中具体用 3 个 key views: frontal, left-side, right-side。

## 5. Closed-Loop Simulation Loop 完整流程

```
Loop over time windows:
  Forward pass: 
    for t = 0 to T-1:
      Ŝ_{t+1} = Φ_p(Ŝ_t, A_t)
    → coarse sequence {Ŝ_t}_{t=0}^T
  
  Backward optimization:
    Ψ_n applied to {Ŝ_t}
    → multi-view video refinement
    → progressive optimization with L = L_p + λ_sim * L_sim
    → refined {S_t}_{t=0}^T
  
  Loop closure:
    p_j^{next} = mean(μ_{j,k,T}) over k
    v_j^{next} = v_{j,T}  (inherited)
    S_0^{next} = (P_T^{next}, V_T^{next}, G_T)
```

这里的关键 insight 是 **loop closure 不需要重新 mesh 化或重新采样 particle**——因为 VPP 在整个 long-horizon 过程中保持同一个 particle set $\{p_j\}_{j=1}^J$,只是位置和速度更新。这避免了像 PhysGaussian (reference [48]) 那种每隔几步重新初始化的麻烦。

**为什么 velocity 可以直接继承**?因为 $\mathcal{L}_{sim}$ 和 $\tanh$ 软约束限制了位置更新的幅度 (paper 说"in a small range"),所以 position 的修正不会让 velocity 失配。这是一个工程上的近似,严格说应该用 finite difference 重新估计 velocity,但 paper 实验显示这个近似在他们的 time window 长度 (392 物理步) 下足够好。

## 6. 实验数据深度解读

### 6.1 World-Score 定量结果 (Table 1)

| Method | Camera Ctrl | 3D Consist | Imaging |
|---|---|---|---|
| Wan2.2 | 59.73 | 65.35 | 67.03 |
| GEN3C | 80.29 | 61.69 | 66.25 |
| WonderPlay | 75.95 | 63.93 | 36.80 |
| Tora | 51.80 | 60.77 | 54.37 |
| Wan2.6 | 64.75 | 70.49 | 66.09 |
| DaS | 78.96 | 62.18 | 60.23 |
| Veo3.1 | 60.61 | 73.93 | 67.82 |
| **PerpetualWonder** | **93.26** | **80.41** | 66.98 |

关键观察:
- **Camera Ctrl 93.26**:这是断层领先,因为 PerpetualWonder 的 4D 表征本身是 3DGS,可以从任意视角渲染,所以 camera ctrl 误差天然极低。所有纯 video diffusion 方法 (Wan/Veo/Tora) 都在 50-65 之间——video model 把 camera trajectory 编码到 attention 里,误差积累。
- **3D Consist 80.41**:也明显领先,WonderPlay 只有 63.93——单视角优化的 multi-view ambiguity 必然导致 novel view 不一致。GEN3C 只有 61.69,因为它支持 camera 但不支持 action,物体不动其实更难"保持一致" (因为没有 dynamics 信号给优化器参考)。
- **Imaging 66.98**:与 Veo3.1 (67.82)、Wan2.2 (67.03) 相当,不显著领先。这是因为 per-frame visual quality 主要由 video diffusion model (CogVideoX 或 Go-With-The-Flow, reference [5,50]) 决定,PerpetualWonder 用的是同一个 backbone,所以 imaging 上限相近。但 Veo3.1 是 Google 最新的超强 video model,这里能跟上已经很不错。
- **WonderPlay Imaging 36.80** 异常低,说明它的 single-view 优化产生了严重伪影——这个数字基本宣告了 WonderPlay 的方法在 long-horizon 下不可用。

World-Score 是 reference [8] (Stanford 同组的工作, ICCV 2025),专门为 world generation 设计的统一 benchmark。规则化指标 (rule-based) 测 camera ctrl 和 3D consistency,用 COLMAP 重建 + camera pose 误差等。Imaging 是 per-frame quality。

### 6.2 User Study (Table 2)

| 对比 baseline | Physics Plausibility favor % | Motion Fidelity favor % |
|---|---|---|
| vs Wan2.2 | 74.1% | 71.8% |
| vs GEN3C | 93.5% | 83.5% |
| vs WonderPlay | 80.8% | 86.3% |
| vs Veo3.1 | 62.0% | 70.8% |
| vs Wan2.6 | 68.5% | 77.3% |
| vs Tora | 83.5% | 85.3% |
| vs DaS | 80.9% | 81.9% |

350 participants, 2AFC protocol, 10 scenes。

最值得注意的:
- vs **Veo3.1**:62% / 70.8%。Veo3.1 是 Google 最强的 video model (https://arxiv.org/abs/2509.20328, reference [44]),它能从文本直接生成相当 plausible 的 dynamics,所以这是 PerpetualWonder 优势最小的地方。但即便如此,62% 仍表明 PerpetualWonder 在 **physics plausibility** 上更好——这正是 hybrid 物理仿真带来的优势。
- vs **WonderPlay**:80.8% / 86.3%。这说明 VPP + multi-view 的核心贡献在用户感知层面也是显著的,不只是 metric 游戏。
- vs **GEN3C**:93.5% / 83.5%。GEN3C 不能响应 action,物体静止,所以 physics plausibility 接近完全胜出。Motion fidelity 83.5% 说明 PerpetualWonder 的视觉动态质量比 GEN3C 的静态高质量还要好。

### 6.3 物理仿真参数 (Supplementary Table S1)

| Parameter | Default Value |
|---|---|
| Step time | $1e^{-3}$ |
| Sub-steps number | 10 |
| Sampled particle size | $1e^{-2}$ |
| Gravity | $(0, 0, -9.8)$ |
| friction coefficient (rigid) | 0.1 |
| Grid density (MPM) | 64 |
| Young's modulus (elastic) | $3e^{5}$ |
| Poisson's ratio (elastic) | 0.2 |
| Young's modulus (liquid) | $1e^{7}$ |
| Poisson's ratio (liquid) | 0.2 |
| Young's modulus (granular) | $1e^{6}$ |
| Friction angle (granular) | 0.2 |

这里值得注意:
- **Step time = 1ms, 10 sub-steps**:每个"物理步"实际是 10 ms 内 10 次积分,典型 MPM/PBD 设置。
- **Young's modulus 弹性 $3e^5$ vs 液体 $1e^7$**:液体的 "Young's modulus" 用得很大,这是 MPM 中 fluid 的 bulk modulus 近似,让流体几乎不可压缩。
- **Grid density 64**:MPM 的 grid 分辨率,典型值。
- 这些参数用 VLM (GPT-4o, reference [16]) 估计 + 可选手动调整。这意味着整个 pipeline 仍然有少量人工调参,不是完全 autonomous。

### 6.4 Runtime (Supplementary Table S2)

| Stage | Initialization | Forward Pass | Backward Opt. | Total (1st Loop) |
|---|---|---|---|---|
| Time | ~8 min | <1 min | ~7 min | ~16 min |

每个 loop 16 分钟,3 个 windows 大约 30-40 分钟。Forward pass <1 min 说明 Genesis simulator 很快,瓶颈在 backward optimization 的 video diffusion model inference + 3DGS optimization。

实时性是 future work 的方向。

## 7. Ablation 深度分析

### 7.1 VPP vs 标准 3DGS (Figure 6)

- **VPP**: visual primitives 被 $\tanh$ + $\mathcal{L}_{sim}$ 约束在 physics particle 周围,dynamics 由 physics 驱动,visual 只能在小范围内 refine。结果:coherent dynamics,清晰 appearance。
- **标准 3DGS** (无 VPP 约束):gaussians 自由优化,只看 photometric loss,结果是 degenerate——每个 view 让 gaussians 跑向不同方向,产生 chaotic dynamics 与 visual artifacts。这是因为标准 3DGS 没有 physics anchor,优化空间太大且 ill-posed。

这印证了 VPP 设计的必要性:**单纯的视觉 loss 在 multi-view 4D 优化下不足以约束 dynamics**,必须有 physics anchor 提供归纳偏置。

### 7.2 Progressive vs Direct Multi-View (Figure 7)

- **Progressive**:stage 1 single view → stage 2 small weight multi-view → stage 3 full multi-view,逐渐消化冲突。
- **Direct**:一开始就用所有视角的 refined videos 同时优化。

Direct 的失败模式:frontal view 视频模型 hallucinate 苹果上有红色斑点,side view 没有,优化器不知道信谁,结果两边都模糊化 (blurry textures + appearance flickering)。

Progressive 的成功:先让 single view 把基础 appearance 锁定,再用小 control weight 的多视角补充细节,矛盾被 $\mathcal{L}_{sim}$ 和已有 anchor 吸收。

这个策略类似于 GAN 中的 progressive training 或 curriculum learning,但用在 view 数量上。

### 7.3 Isotropic vs Anisotropic (Supplementary E)

Isotropic primitives 在 novel view 更干净。Anisotropic 容易 overfit input view,因为它们可以拉长去匹配某个特定 view 的形状,但 novel view 就出现模糊。

这跟 3DGS 原文 [18] 用 anisotropic 不一样——3DGS 的目标是 reconstruct 单一静态场景,anisotropic 提供更多表达力。但 4D generation 中 view 数有限且 video 模型本身有 hallucination,anisotropic 容易过拟合。

### 7.4 Particle Radius (Supplementary F)

$[0.25\delta, 4\delta]$ 范围内 robust。太小 ($\le 0.01\delta$):表征能力不足。太大 ($\ge 100\delta$):优化不稳定。

这反映了 VPP 设计的一个 trade-off:$\delta$ 决定了 visual 修正空间与物理对齐强度的平衡。$\delta$ 大 → visual 自由度高但物理约束弱;$\delta$ 小 → 物理约束强但视觉表达受限。

## 8. 关键设计选择与延伸思考

### 8.1 VPP 与其他统一表征的对比

类似思路的工作:
- **PhysGaussian** (reference [48]): 也用 physics-integrated 3D Gaussians,但每个 gaussian 直接对应一个 particle,没有 $K$ 绑定,所以体积表达受限。
- **4D Gaussian Splatting** (reference [49]): 直接在 4D 时空用 gaussian,没有 physics anchor。
- **DreamPhysics** (reference [14]): 用 video diffusion prior 优化 physics-based 3D dynamics,但表征是 mesh-based。

VPP 的独特性在于 **bidirectional + scalable**:
- 双向:physics → visual (forward) + visual → physics (backward via averaging)。
- 可扩展:$K$ 自适应,允许体积物体用多个 gaussian 表达。

### 8.2 信息流的数学结构

更形式化地,PerpetualWonder 的 closed-loop 可以写成:

$$
\begin{aligned}
\hat{S}_{t+1} &= \Phi_p(\hat{S}_t, \mathcal{A}_t) \\
S_{t+1} &= \Psi_n(\hat{S}_{t+1}, V_{t+1}^{(1:V)}) \\
\mathcal{P}_{T}^{\text{next}} &= \text{AverageGaussianPositions}(S_T)
\end{aligned}
$$

其中 $\Psi_n$ 内部:
$$
\Psi_n = \arg\min_{\{\mu_{j,k,t}, o, \dots\}} \sum_v \sum_t \mathcal{L}_p(\text{Render}(S_t, \text{view}_v), V_t^{(v)}) + \lambda_{sim} \mathcal{L}_{sim}
$$

关键设计:$\Phi_p$ 输出 $\mathcal{P}_t$,$\Psi_n$ 输入 $\hat{S}_t$ (含 $\mathcal{P}_t$),$\Psi_n$ 优化 $\mathcal{G}_t$ 但通过 $\mathcal{L}_{sim}$ 与 $\mathcal{P}_t$ 软对齐,$\Psi_n$ 输出 $S_T$ 中的 $\mathcal{G}_T$ 通过 averaging 反推 $\mathcal{P}_T^{\text{next}}$。

这跟 model-predictive control (MPC) 结构有点像:
- $\Phi_p$ = forward model (simulator)。
- $\Psi_n$ = observation-based state correction。
- Loop closure = state estimation update。

### 8.3 Failure modes

Supplementary G 提到一个 failure case:hockey stick 从画面外进入,中间 frame 中 stick 看起来不完整(应该更长)。这是因为 input image 没看到 stick 的完整几何,dense view generation (GEN3C) 也没补上。这是 single-image 4D generation 的根本限制——**unseen geometry 完成是 ill-posed**。

可能改进方向:
- 用 large 3D generative model (Hunyuan3D, TRELLIS) 做 object completion。
- 用 video LLM 的 in-context reasoning 推断 unseen geometry。
- 多模态输入 (text + image) 给完整 object description。

### 8.4 与 world model 路线的关系

Sora (reference [4], OpenAI 2024) 把 video model 当 world simulator,但 OpenAI 自己也承认 Sora 不能严格物理一致。PerpetualWonder 走的是 **hybrid** 路线:物理仿真保证 consistency + controllability,video model 保证 realism。这跟 Sora 的 pure neural 路线是两种哲学。

我个人觉得 hybrid 路线在 embodied AI 应用上更靠谱,因为:
1. 纯 neural world model 难以做长 horizon (compounding error 是 well-known)。
2. 物理仿真的 controllability 对 robot learning 等下游任务至关重要。
3. Video model 的 hallucination 在 generation 任务里 OK,但在 control 任务里致命。

但 hybrid 路线的代价是 **表征设计复杂** + **运行慢** (16 min/loop)。如果未来 video model 的 long-horizon 一致性问题被解决 (例如通过 RL training 或更长 context),纯 neural 路线可能反超。

### 8.5 VPP 的理论性质

VPP 实际上定义了一个 **bundle structure**:base space 是 physics particle manifold $\{p_j\}$,fiber 是 K 个 gaussians 的属性空间。这跟 gauge theory 或 associated bundle 的数学结构有点像——physics particle 是 "base",gaussians 是 "fiber",tanh 约束是 "connection" (限制 fiber 相对 base 的偏移)。

虽然 paper 没用这种语言,但这种结构暗示了 VPP 可以推广到更复杂的 fiber (例如每个 particle 挂一个 small MLP 而非 K 个 gaussians)。

### 8.6 Progressive Multi-View 与 Bayesian Model Averaging 的类比

Progressive 策略可以看作一个简单的 Bayesian update:
- Stage 1: prior 是 single view 的 refined video,confidence 高。
- Stage 2: 加入小 weight 的 side views,相当于 noisy measurements,bayesian update 缓慢。
- Stage 3: 等置信度稳定后,所有 views 一起 average。

这个类比可以推广到更多 views 时——可以想象一个 active view selection 策略,根据当前 uncertainty 选 next view。

### 8.7 跟相关工作的 link 表

- **WonderPlay** [21]: https://arxiv.org/abs/2506.04225 (前作, hybrid generative simulator)
- **GEN3C** [34]: https://gen3c.github.io/ (NVIDIA, dense view generation)
- **3DGS** [18]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ (基础表征)
- **Genesis** [19]: physics simulator
- **SAM2** [31]: https://github.com/facebookresearch/sam2 (segmentation)
- **Hunyuan3D** [61]: https://github.com/Tencent/Hunyuan3D-2 (rigid mesh)
- **WorldScore** [8]: https://worldscore.github.io/ (benchmark)
- **Go-With-The-Flow** [5]: https://ondrejtex.github.io/Go-With-The-Flow/ (video model)
- **GPT-4o** [16]: https://arxiv.org/abs/2410.21276 (VLM for physics params)
- **PhysGaussian** [48]: https://arxiv.org/abs/2311.12983 (前作, physics-integrated gaussians)
- **PhysDreamer** [58]: https://physdreamer.github.io/ (前作, video generation + physics)
- **FreeTimeGS** [42]: https://openreview.net/forum?id=FreeTimeGS (temporal opacity 灵感)
- **Veo3.1** [44]: https://arxiv.org/abs/2509.20328 (Google video model baseline)
- **Sora** [4]: https://openai.com/research/video-generation-models-as-world-simulators (OpenAI world model 路线)

## 9. 总结:Intuition Building

如果让我用一句话概括 PerpetualWonder 的核心贡献:

**通过把 physics particles 与 visual gaussians 用 tanh 软约束 + averaging loop closure 绑成 bundle structure,把单向信息流的 hybrid simulator 变成闭环,使 video diffusion model 的 refinement 可以回写更新物理状态,从而支持 long-horizon action-conditioned 4D 生成。**

公式 (1) $\mu_{j,k} = p_j + \tanh(\tilde{p}_{j,k}) \cdot \delta$ 是整个系统的灵魂——这个 $\tanh(\cdot) \cdot \delta$ 软约束既允许视觉在物理周围微调,又限制了视觉漂离物理。配合 $\mathcal{L}_{sim}$ 的均方 anchor loss,优化器只能在物理约束的小邻域内寻找视觉最优。这就是 "bidirectional bridge" 的具体数学形式。

而 progressive multi-view optimization 则解决了 video diffusion model 本身跨视角不一致的工程问题——单视角不能 unambiguously 确定 3D,直接多视角冲突,渐进式才能让 $\mathcal{L}_{sim}$ 的物理 anchor 逐步吸收冲突。

这两个设计合在一起,使 long-horizon 成为可能:**error 不再单调累积**,因为每一轮 backward pass 都用 video model 的 prior 修正了上一轮的物理误差。

潜在 future direction 我能想到的:
1. **Active view selection**:不是固定 3 个视角,而是基于当前 4D scene uncertainty 选 next view。
2. **Differentiable physics**:把 $\Phi_p$ 也变成可微的,让 backward pass 可以同时优化物理参数 (friction, Young's modulus 等)。
3. **Object-level VPP**:每个 object 独立 VPP bundle,支持 object insertion/removal in long horizon。
4. **Real-time optimization**:用 3DGS 的 fast rasterizer + 蒸馏 video model 到 lightweight refiner。
5. **Multimodal actions**:除了 force,加入 language-conditioned actions ("pour the water", "cut the cloth")。
6. **Robot learning integration**:用 PerpetualWonder 生成 synthetic training data for manipulation policies,类似 reference [30] SplaSim 但带 dynamics。

希望这些分析对你 build intuition 有帮助,Karpathy!如果想要更深入某个部分 (比如 VPP 的优化几何性质,或者 video model 的 bimodal control 具体如何实现),我可以继续展开。
