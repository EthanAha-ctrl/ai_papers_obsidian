---
source_pdf: From Seeing to Knowing the World A Survey of Vision World Models.pdf
paper_sha256: 69570e954e1bfbfdf99d6da3b3cd575b576c5a298d8bb3a162f0c2cc0a21d000
processed_at: '2026-08-04T10:59:35-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 其实是在给目前 AI 领域一个极其火爆但又乱七八糟的方向做“大扫除”和“立规矩”。这个方向就是 **Vision World Model (VWM)**。

我把这篇 paper 的核心逻辑用人话拆解一下，帮你快速 build intuition。

### 1. 这篇文章到底想解决什么痛点？

现在 AI 圈子里，搞视频生成的（Sora 团队）、搞机器人的、搞自动驾驶的，大家都在做同一件事：让 AI 看视频，然后预测未来会怎样。但是大家各说各话，用的名词不一样，评测标准也不一样。

最要命的问题是：**大家把“视频做得逼真”等同于“AI 懂物理规律了”**。

这篇 paper 的核心主张是：Vision 绝对不是随便给的输入模态，Vision 本身就是塑造世界模型的核心力量。单纯靠大力出奇迹去预测下一帧像素，是出不了真正的 AGI 的。我们需要从“看到”走向“知道”。

### 2. Vision World Model 到底是个啥？

通俗地说，VWM 就是给 AI 装一个“脑补引擎”。

公式 $p(S_{t+1:T} | v_{0:t}, c_t) = f_\theta\big(\mathcal{E}(v_{0:t}), c_t\big)$ 看着吓人，人话就是：给定过去的视频 $v_{0:t}$，加上你现在的动作 $c_t$（比如踩了油门），模型 $f_\theta$ 要能算出接下来会发生什么 $S_{t+1:T}$。

关键在于，过去的 World Model 喜欢用低维数字（比如坐标 x, y）来推演，现在的 VWM 直接拿高维的视觉信号当底层地基，这是本质的区别。

### 3. Paper 提出的统一框架：三步走

作者把所有的 VWM 拆解成三个零件，就像一条流水线：

*   **Vision Encoding（看）**：原始视频里包含光影、噪点、背景杂物。Vision Encoding 就是把无用的噪音滤掉，留下真正有用的信息。输出可以是连续的向量，也可以是离散的 token，或者是按物体切开的“插槽”。
*   **Knowledge Learning（懂）**：这是最核心的。AI 看视频不能只学个皮毛，得学三样东西：
    1.  **Spatio-temporal Coherence**：东西不能凭空消失，也不能瞬间移动。
    2.  **Physical Dynamics**：杯子砸地上会碎，车撞了会变形，不能穿模。
    3.  **Causal Mechanisms**：这最深。因为推了一下，所以球滚了。懂因果才能做 counterfactual reasoning（反事实推理，比如“如果我刚才没踩刹车会怎样”）。
*   **Controllable Simulation（演）**：给定现在的条件和动作，在脑子里 rollout 一把，看看未来几秒会咋样。输出可以是 latent state，可以是直接生成的视频画面，也可以是结构化的轨迹。

### 4. 市面上的四大门派

这篇 survey 把现在乱七八糟的方法分成了四大家族，这就跟武侠里的门派一样，各有各的绝招和软肋：

*   **Sequential Generation（拼图派）**
    *   **玩法**：把视频变成一串离散的 token，像大语言模型写文章一样，一个词一个词往后预测。Genie 就是这么干的。
    *   **软肋**：预测长了容易积攒误差，最后画面糊掉。而且只靠猜下一个 token，很难真正学到底层的物理因果，多是在模仿表面规律。
*   **Diffusion-based Generation（降噪派）**
    *   **玩法**：跟 Sora 一样，从一团马赛克里一点点洗出清晰的未来画面。GameNGen 玩 DOOM 就是这么搞的。
    *   **软肋**：画面极美，但推理太慢。而且如果一段一段往外生成，前面生成的有微小瑕疵，后面就会越来越崩。
*   **Embedding Prediction（抽象派）**
    *   **玩法**：Yann LeCun 的最爱（V-JEPA 2）。不生成画面了，太费劲且容易跑偏。直接在抽象的特征空间里预测未来的 representation。
    *   **软肋**：计算贼快，适合做 planning，但缺点是你看不见它预测了啥，缺乏解释性。而且能力被死死限制在 frozen encoder 的上限里。
*   **State Transition（状态机派）**
    *   **玩法**：Dreamer 系列的老路子。把画面压缩成一个极其紧凑的 state，然后用 $P(s_{t+1} | s_t, a_t)$ 一点点推演这个 state。
    *   **软肋**：长链推演很快，但丢掉了太多细节，你得不到直观的画面。

### 5. 为什么现在的评测标准很荒谬？

这篇 paper 最犀利的洞察在评测这里。

现在大家搞了个 World Model，怎么证明自己牛呢？拿去跟 Sora 比一下 FID、FVD（画面像不像、清不清）。这就好比考一个司机的驾驶技术，你不去看他能不能安全把车开到目的地，反而去量他打方向盘的姿势美不美观。

作者提出，真正的 VWM 评测必须看三层：
1.  **Visual Quality**：画面好看吗？（这只是基础）
2.  **Physical Plausibility**：物理上合理吗？轨迹对不对？几何有没有穿模？
3.  **Task Performance**：关键来了，用你这个 World Model 去 rollout 做规划，任务能不能成？能不能避开障碍？这就是用“执行结果”来反推模型到底懂不懂事。像 IntPhys 2 提出的 Surprise Score（惊讶分数）就很聪明：给 AI 看违反物理常识的视频，看它会不会觉得“卧槽这不对劲”。

### 6. 未来的出路：三个 Re-

作者最后开了个药方，指出 VWM 要继续往下走，得搞三个方向：

*   **Re-grounding（找回物理根基）**：现在光靠海量数据去炼丹是不行的，模型经常瞎猜。需要把 3D 几何结构硬塞进架构里，甚至搞 Neuro-symbolic（神经网络 + 符号逻辑），用真正的物理引擎去约束它的想象力。
*   **Re-evaluation（重造评测标准）**：别再光看画质了。要训练专门的 Judge Model 来审查 VWM 的预测，并且必须把模型丢到闭环里去跑任务，用任务成功率说话。
*   **Re-scaling（重新定义 Scaling Law）**：现在光把模型变大、数据变多，只会让画面更漂亮，并不会让模型更懂物理。未来要在数据中增加更多的因果互动，并且探索 Inference-time scaling（让模型在下结论前，多在脑子里 deliberation 几步，反复推演修改）。

### 总结一句

这篇 paper 就是在大喊一声：**别再沉迷于做漂亮的视频生成器了！Vision World Model 的本质是让 AI 在脑子里建立一个符合物理和因果规律的小世界，评测它好不好用得把它扔进真实任务里去溜溜，光看画质是自欺欺人。**

---

# From Seeing to Knowing the World: A Survey of Vision World Models 深度解读

## 1. 论文动机与核心立场

这篇survey来自Beijing Jiaotong University、ByteDance、Tencent等机构，由Xiaojie Jin和Yunchao Wei主导，发布于2026年4月。文章的核心主张非常清晰：**Vision 不应仅仅被视为world model的输入modality，而应被视为塑造world model representation、learning objective和evaluation方式的主导因素**。

paper观察到当前VWM研究存在三个痛点：
- **Inconsistent problem formulations**：不同社区（generative modeling, representation learning, embodied intelligence）用不同语言描述相似问题
- **Disconnected taxonomies**：autoregressive, diffusion, embedding prediction, state transition各自为战
- **Divergent evaluation protocols**：visual quality, physical plausibility, task performance各有各的衡量标准

paper的核心定义（Section 2.1）：

> A Vision World Model (VWM) is an AI model that **learns world knowledge from visual data and generates future world states conditioned on interaction**.

形式化定义（公式1）：
$$p(S_{t+1:T} | v_{0:t}, c_t) = f_\theta\big(\mathcal{E}(v_{0:t}), c_t\big)$$

变量解释：
- $v_{0:t}$：时间从0到$t$的visual data序列（可以是RGB images/videos, depth, BEV, point cloud等）
- $c_t$：interaction conditions，包括agent actions, language instructions, control commands
- $\mathcal{E}(\cdot)$：visual encoder，将raw visual data转为representation
- $S_{t+1:T}$：future world states，可能是future frames, latent states, occupancy grid, 3D primitives或trajectory
- $f_\theta$：参数化的probabilistic model，$\theta$是其参数
- 下标 $t+1:T$ 表示从下一时刻到horizon T的整段未来序列

这个定义的关键insight：**与传统world model（如Ha & Schmidhuber 2018）不同，VWM以高维visual data作为foundation，而不是predefined low-dimensional state space**。这从根本上重塑了world knowledge如何被表示和学习。

参考链接：
- 项目主页: https://AIWorldLab.github.io/survey
- arXiv: https://arxiv.org/abs/2504.2072 (preprint)

## 2. Unified Framework：三大核心组件

paper提出的unified framework将VWM拆解为三个核心组件，构成一个完整的pipeline。这个框架是整个survey的组织骨架。

### 2.1 Vision Encoding（Section 2.2）

**目的**：将raw visual data转换为disentangled representations，抑制无关variation（camera jitter, background clutter, sensor noise），保留与world change相关的因素。

#### Visual Inputs的种类

| Input Type | 优势 | 代表工作 |
|---|---|---|
| RGB images/videos | 数据易得，覆盖广 | Gaia-1, Genie, LWM |
| Depth maps/point clouds | 显式3D几何 | Geometry-aware 4D Video, LiDARCrafter |
| Optical flow | 显式motion | Taming Generative Video, LPS |
| BEV (Bird's-Eye-View) | 统一坐标frame | OccWorld, DriveWorld, OccLLaMA |
| Multi-view/Egocentric | 3D consistency + action alignment | MV-MWM, Ego4D |

#### Representation Forms

paper将representations分为四类，对应不同的"信息保留—抽象程度"权衡：

**Continuous Latent Representations**：使用CNN或ViT编码成连续latent space。Dreamer系列就是典型。优势在于state space平滑演化，适合motion和long-horizon dynamics建模。

**Discrete Tokenized Representations**：通过VQ-VAE或VQ-GAN将visual input映射到固定大小vocabulary。这种方式有两个重要后果：
1. 计算效率更高
2. 桥接visual modeling与sequence-based generative framework，使VWM可以借用LLM的scalable Transformer架构

**Object-/Entity-centric Representations**：以persistent identity的entity为基本单位，便于spatio-temporal coherence和causal interaction建模，支持compositional generalization。

**Hybrid/Hierarchical Representations**：组合多种形式，例如continuous representations over discrete tokens，或object-centric states embedded in latent spaces。

### 2.2 Knowledge Learning（Section 2.3）

paper将world knowledge分为三个complementary aspects，这是理解VWM"学了什么"的关键：

#### Spatio-temporal Coherence（时空一致性）

这是"脚手架"层。两个维度：
- **Spatial level**：multi-view consistency（同一object在不同视角被识别）+ geometric stability（shape不任意collapse或deform）
- **Temporal level**：object permanence（occlusion后仍存在）+ smooth state progression（变化遵循plausible trajectory，没有abrupt discontinuities）

#### Physical Dynamics（物理动力学）

要求遵守fundamental physical constraints：gravity, contact, material resistance。这避免常见artifact，例如object无端穿透solid surface。

paper区分了两个复杂度层次：
- **Macroscopic level**：classical mechanics主导，rigid-body movement和object interaction
- **Continuum mechanics level**：deformable materials和fluid behavior，由material properties决定

unifying constraint是conservation principles（energy, momentum）。

#### Causal Mechanisms（因果机制）

这是paper最强调、也是当前VWM最薄弱的部分。区别于statistical correlation（"看到红灯后通常会停"），causal mechanism要求理解action→outcome的fundamental关系。

关键能力：**counterfactual reasoning**——评估"如果采取替代action会怎样"。例如，理解高速撞击会导致structural deformation，使得模型即使在unseen environment也能预测crash consequence。

paper还指出causal机制超出纯物理范畴：human-centered environments中social norms、conventions、shared intentions也shape world behavior（traffic light在临时人工指挥下含义会改变）。

### 2.3 Controllable Simulation（Section 2.4）

Simulation产生future world states。三种形式：

**Latent States**：在compressed latent space中rollout，计算高效，适合planning和reasoning。

**Visual States**：直接在visual space中生成image/video/几何形式，对human interpretability和closed-loop evaluation重要。

**Structured Outputs**：object attributes, spatial configurations, action trajectories，直接作为control或planning输入。

Interaction的形式：
- **Action Signals**：连续/离散control（robot motor commands, keyboard/mouse）
- **Multimodal Interaction**：language指令，特别是Vision-Language-Action frameworks

---

## 3. 四大Architectural Families深度解析

paper将VWM design分为四个家族、七个sub-design。这是paper的技术核心。我用一个表概括整个taxonomy：

| Family | Sub-design | 核心机制 | 代表工作 |
|---|---|---|---|
| **Sequential Generation** | Visual Autoregressive | next-token prediction over visual tokens | Gaia-1, Genie, VideoWorld, MineWorld |
| | MLLM-guided Multimodal AR | LLM-compatible tokens + interleaved multimodal rollout | 3D-VLA, GR00T N1, F1, DreamVLA |
| **Diffusion-based Generation** | Latent Diffusion | block-wise denoising in continuous latent | DriveDreamer, GAIA-2, GWM |
| | Autoregressive Diffusion | sequentially conditioned denoising | GameNGen, Oasis, Matrix-Game, GameFactory |
| **Embedding Prediction** | JEPA-style | predict future embeddings, not pixels | V-JEPA 2, DINO-WM, FLARE |
| **State Transition** | State Space Modeling | compact recurrent state update | Dreamer V1/V2/V3, TD-MPC2, Think2Drive |
| | Object-Centric Modeling | factored entity slots | SlotFormer, COSMOS, SlotPi, Dreamweaver |

### 3.1 Sequential Generation

#### 3.1.1 Visual Autoregressive Model

**Pipeline**：video → VQ-VAE/VQ-GAN → discrete token sequence → next-token prediction → autoregressive rollout → decode

**Tokenization的关键variation**：
- 早期方法（GAIA-1, WorldDreamer）使用**spatial tokenization**：每帧独立编码成token序列
- 近期方法（Genie, iVideoGPT）采用**spatio-temporal tokenization**：多帧共享token，更好保留时序信息

paper提到了3D voxel grid的tokenization（OccWorld, RenderWorld, OccTENS）——将3D voxel grid discretize成vocabulary indices，比纯appearance token更explicit地表达spatial structure（geometry, free space）。

**Knowledge Learning机制**：通过next-token prediction捕获temporal dependencies以维持spatio-temporal coherence。具体包含：
1. **Entity consistency**：object在motion或occlusion后保持identifiable
2. **Temporal continuity**：state变化smooth而非abrupt

**Latent Action Learning**是关键创新：从video中infer action变量（无需action label），用这些变量condition prediction。Genie开创了这一思路，使model将control input（steering, pushing）与predicted consequence关联起来。

**Strengths/Limitations**：
- 优势：scalable，rollout长度灵活
- 局限：long-horizon error accumulation；discrete token限制fine-grained geometric detail；physics和causality主要靠data归纳，没有explicit constraint，distribution shift下robustness差

#### 3.1.2 MLLM-guided Multimodal Autoregressive Model

**动机**：纯visual AR model对language指令不友好。引入MLLM backbone（Vicuna, LLaMA, PaliGemma, Eagle-2等），将visual observation映射成language-compatible tokens。

**Architecture设计**：
- Visual encoder（CLIP, DINO, SigLIP）+ Connector/Adapter → 与text token拼接 → MLLM backbone处理
- 输出interleaved multimodal sequence：visual tokens + text explanations + action tokens

**Knowledge Learning机制**：除了visual token statistics，还利用pretrained LLM的knowledge作为semantic prior。当visual evidence incomplete或ambiguous时，language knowledge（object properties, likely effects）可以guide预测。

**典型例子**：
- ADriver-I：visual tokens + textual descriptions + discrete action tokens混合序列
- OccLLaMA：将semantic voxel grids discretize成scene-level token vocabulary
- GR00T N1：predicted futures附带textual reasoning，解释"为何如此演化"
- WALL-E 2.0：引入neuro-symbolic components加强long-horizon consistency

**Strengths/Limitations**：
- 优势：语言交互灵活、可解释、cross-domain knowledge
- 局限：projecting visual到language-compatible tokens可能丢失fine-grained信息；language-level knowledge在visual evidence偏离common pattern时会bias预测

### 3.2 Diffusion-based Generation

#### 3.2.1 Latent Diffusion

**Pipeline**：visual observations → VAE encoder → continuous latent features → iterative denoising → block-wise decode → future frames/4D representation

**Representation选择**：
- 早期：2D spatio-temporal latents（DrivingDiffusion, Panacea）
- 近期：3D-aware latents（GAIA-2的spatially grounded latent tokens；GWM的Gaussian splats；WoVoGen的voxel grids）

**Knowledge Learning机制**：通过denoising objective学习——从progressively corrupted inputs中恢复future latent states。因为是整块denoise，diffusion model自然鼓励在spatio-temporal block内model dependencies。

**Conditioning策略**：物理和action-conditioned dynamics通过conditioning signals注入：
- DriveDreamer, DOME：condition on 3D box layouts或voxel grids
- GeoDrive：manipulate geometric control conditions产生counterfactual futures

**Interaction形式**：conditional diffusion，control signals作为conditioning input引导denoising走向不同plausible futures。autonomous driving中广泛用于closed-loop evaluation。

**Strengths/Limitations**：
- 优势：visual quality高（continuous latents + iterative refinement）；intra-clip consistency好
- 局限：diffusion inference计算昂贵，限制real-time interaction；temporal scalability受fixed window length限制

#### 3.2.2 Autoregressive Diffusion

**核心问题**：latent diffusion只能在fixed window内generate，如何extend到long horizon？

**解决方案**：将diffusion sequential化——每一步denoising condition在previously generated outputs上。但引入sequential dependency会产生**training-inference mismatch**：
- Training时：denoising condition在ground-truth history上
- Inference时：denoising condition在self-generated history上（分布偏离真实）

这种偏离会随时间accumulate导致drift。近期方法用三种策略缓解：
1. **Noise augmentation**（PlayGen, GameFactory）：训练时给history加噪声
2. **Autoregressive rollout simulation during training**：训练时模拟inference过程
3. **Memory mechanisms**（WORLDMEM, VMem）：保留超出context window的额外历史信息

**架构选择**：
- Continuous VAE latents：保留fine visual detail
- Discrete token sequences：利用transformer scalability

**Sequential Denoised Rollouts**：每一步new future states在prior outputs和control inputs上generate。支持high-level language goals和low-level control。

**Inference效率突破**：GameNGen和Yan demonstrate了20+ FPS的playable neural environments，是real-time interactive generation的关键milestone。

**Strengths/Limitations**：
- 优势：high visual fidelity + long-horizon extension能力
- 局限：error accumulation敏感；diffusion sampling仍然计算intensive

### 3.3 Embedding Prediction

**核心思想**：完全跳过pixel-level generation，在representation space中直接predict future embeddings。以JEPA系列为典型。

**架构设计**：
- Visual encoder（DINOv2, CLIP, SigLIP）→ contextual embeddings
- Predictor module：从context预测future target embeddings
- 通常freeze encoder，只train predictor

**Mask-and-Predict训练**：mask掉input部分，预测其embeddings。鼓励model在representation space中维持spatio-temporal coherence——即使部分input被occluded，object identity和motion continuity仍能保留。

**Action-Conditioned扩展**：DINO-WM, FLARE通过引入control input预测不同的future embeddings，捕获causal relations。

**代表性工作分析**：

V-JEPA 2（LeCun团队）：核心是mask-and-predict，学习spatio-temporal coherence和motion understanding。latent prediction让其可用于planning而无需视觉解码。

DINO-WM：复用DINOv2 encoder，在embedding space evaluate多个candidate action sequences选择promising behaviors。

FLARE：引入robot state和text作为condition，学习long-term consequences，支持robot learning with implicit world modeling。

AD-L-JEPA：将JEPA应用到LiDAR point clouds，证明contextual embeddings可跨modality共享。

EchoWorld：将方法适配到ultrasound data，学习heart anatomy和motion dynamics。

**Embedding-Space Rollouts**：simulation产出target embeddings而非rendered images。因为不需要visual decoding，planning和action evaluation可以完全在representation space中完成，效率高，特别适合long-horizon planning。

**Strengths/Limitations**：
- 优势：computational efficiency（轻量feature space operation）；适合long-horizon planning；modality-agnostic
- 局限：无explicit visual decoding降低interpretability；reliance on frozen foundation models限制representational capacity

参考链接：
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- DINO-WM: https://arxiv.org/abs/2411.04983
- FLARE: https://arxiv.org/abs/2505.15659

### 3.4 State Transition

#### 3.4.1 State Space Modeling

**核心范式**：Dreamer系列的legacy。视觉observation → compact recurrent state → recurrent transition update → latent space rollout。

**架构演化**：
- 早期：CNN/ResNet encoder + RSSM（Recurrent State-Space Model）
- 近期：ViT-based encoder（MWM先用ViT autoencoder，再学transition）；BEV encoder（Think2Drive, DriveWorld聚合多相机输入到unified BEV）；domain-specific encoder（GASv2处理stereoscopic surgical images）

**Transition Model**：核心公式 $P(s_{t+1} | s_t, a_t)$
- $s_t$：latent state at time $t$
- $a_t$：action at time $t$
- $P$：transition distribution，可由RSSM, ConvLSTM, Transformer, SSM variant等参数化

**Long-horizon modeling挑战**：随着sequence增长，transition model需要有效retain和utilize历史信息。三种架构创新：
- R2I：用structured state-space mechanism替代GRU-style updates，学习long-range dependencies
- SSVWM：用mamba-style transitions在occlusion下维持coherence
- LS-Imagine：hierarchical transitions在多temporal scale上运作

**Latent State Rollouts**：simulation在latent state space中rollout，与embedding prediction类似但explicitly maintain recurrent state并step-by-step update。Dreamer家族showcase了fast long-horizon rollout用于evaluate candidate action sequences。

**Interaction层次**：
- Low-level motor commands（DayDreamer）
- Higher-level action abstractions（CADDY）
- Hierarchical control：Puppeteer（high-level world model guide low-level skills）；FOUNDER, RoboHorizon（用language model分解instructions成sub-goals）

**Strengths/Limitations**：
- 优势：efficient long-horizon rollout；保留past observations和actions信息
- 局限：human interpretability低（latent state非image）；compact state难保留fine-grained spatial/geometric detail

#### 3.4.2 Object-Centric Modeling

**核心思想**：将world表示为entity slots集合，而非单一monolithic vector。每个slot对应一个entity（identity/attributes + spatial extent）。

**Slot获取机制**：
- Slot Attention, SAVi：unsupervised binding，将pixels group成entity-centric components
- 近期enhancement：用ViT tokenize patches（SlotFormer, SlotSSMs）；分离foreground/background（G-SWM）；引入symbolic attributes（COSMOS将slot vectors与symbolic attributes对齐，并与CLIP representation对齐）

**Slot Interaction建模**：用attention或graph-based interactions传播slot间信息。
- SlotFormer, LSlotFormer：attention-based slot dynamics预测long-horizon trajectories
- SlotPi：引入Hamiltonian structure regularize energy exchange，物理约束
- G-SWM：用structured networks建模occlusion/collision-related interactions

**Compositional Generalization**：object-centric factorization能更好处理known entities的新组合。slot interaction patterns可transfer到novel object combinations。

**Slot-State Rollouts**：在slot space中rollout，更新entity states。支持controllable manipulation of entity attributes——Dreamweaver通过swap或alter properties实现object-level editing；Dyn-O利用static和dynamic disentanglement在simulating复杂motion时维持identity。

**Interaction**：multimodal——LSlotFormer融合slots与T5 embeddings用于text-conditioned manipulation；MEAD评估targeted object perturbations的outcomes。

**Strengths/Limitations**：
- 优势：compositional generalization好；explicit slot structure提升interpretability
- 局限：**Binding Problem**——在clutter, heavy occlusion或fine-grained texture下slot assignment ambiguous；许多方法依赖fixed number of slots；scaling robust slot discovery和interaction modeling到unconstrained natural videos仍是open challenge

参考链接：
- SlotFormer: https://arxiv.org/abs/2210.05861
- COSMOS: https://arxiv.org/abs/2310.12690
- SlotPi: https://arxiv.org/abs/2501.04983 (paper ref [292])

### 3.5 Other Emerging Directions

paper简略提到lightweight CNN-based设计、attention-based updates、graph neural networks、flow-matching formulations等alternative transition mechanisms，但它们尚未形成large unified research branches。

---

## 4. Evaluation体系深度解读

这是paper最有价值的部分之一，因为它揭示了VWM评估的核心瓶颈：当前protocol主要borrow video generation的metrics，过度强调appearance quality，忽视了VWM捕获fundamental physical和causal principle的能力。

### 4.1 三类Evaluation Metrics

#### Visual Quality

**Objective Fidelity**：
- PSNR [299]：pixel-wise差异，公式 $PSNR = 10 \cdot \log_{10}\big(\frac{MAX_I^2}{MSE}\big)$，$MAX_I$是pixel max value
- SSIM [300]：local structural similarity，考虑luminance, contrast, structure三维度
- FID [301]：image distributional metric，比较real和generated sample在pretrained Inception feature space的Fréchet distance
- FVD [302]：video distributional metric，FID的video扩展

**Perceptual Alignment**：
- LPIPS [303]：在pretrained feature space计算distance，更好align人类judgment
- DreamSim [304]：用synthetic data训练的perceptual metric
- DOVER Score [305]：no-reference metric，estimate artifact level, naturalness, perceptual quality

#### Physical Plausibility

**Kinematic Accuracy**（运动学正确性）：
- ADE, FDE [183]：trajectory deviation，ADE = Average Displacement Error（轨迹各点平均偏差），FDE = Final Displacement Error（终点偏差）
- RPE [306]：Relative Pose Error，ego-motion estimation
- Camera Pose Loss [234]：camera pose的estimation error
- MPJPE, PA-MPJPE, MPJVE [307]：articulated pose accuracy（Mean Per Joint Position Error等）
- Optical Flow Error [201]：short-term motion precision

**Geometric Validity**（几何有效性）：
- Chamfer Distance [308]：point cloud structural similarity，$CD(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} \|x-y\|^2 + \sum_{y \in S_2} \min_{x \in S_1} \|y-x\|^2$
- AbsRel [143]：depth prediction accuracy，Absolute Relative Error
- 2D Reprojection Error [309]：multi-view geometric alignment

**Spatio-temporal Consistency**：
- Scene Revisit Consistency (SRC) [310]：当camera回到同一spatial location时scene是否structurally consistent
- Revisit Error (RVE) [311]：revisit时的error量化

#### Task Performance

**Process-level Evaluation**（过程评估）：
- Raw Return [10]：累积reward
- Human Normalized Score [116]：归一化到human performance
- Driving Score [312]：自动驾驶综合评分
- PDMS Score [313]：Predictive Driving Model Score

**Goal Completion**：
- Success Rate [314]
- MTLC Success Rate [315]：Multi-Task Long-horizon Completion
- Contact Rate [285]
- Grasping Score [258]
- Collision Rate [316]

**Perception/Control Accuracy**：
- Top-K Accuracy [28]
- Precision, Recall, F1-Score [34, 141]
- Translation Error [260], Rotation Error [248]

### 4.2 Datasets and Benchmarks

paper将datasets/benchmarks分为两大组：Foundational World Modeling和Domain-specific World Modeling。

#### Foundational World Modeling

**General World Prediction and Simulation**：
- SSV2 [317] (2017)：大规模video-text预训练
- Ego4D [74] (2021)：3000小时egocentric video
- WorldModelBench [318] (2025)：Instruction/Physics/Commonsense Scores
- WorldScore [309] (2025)：Controllability/Quality/Dynamics scores
- WorldPrediction [321] (2025)：World Modeling Score + Procedural Planning Score
- Sekai [323] (2025)：用VBench [358]评估4D spatio-temporal consistency
- OmniWorld [326] (2025)：camera-parameter-based metrics

**Physics and Causality Benchmarks**（物理因果基准）——这是VWM最关键的evaluation：

- CoPhy [338] (2019)：counterfactual scenes，paired factual-counterfactual
- Physion++ [340] (2023)：要求infer latent physical properties (mass, friction)再forecast
- Physics-IQ [341] (2025)：spatio-temporal IoU
- VideoPhy-2 [92] (2025)：Semantic Adherence, Physical Commonsense, Physical Rule Violation
- VBench-2.0 [343] (2025)：VQA-based Physics和Commonsense Score
- IntPhys 2 [9] (2025)：violation-of-expectation protocol，**Surprise Score**——衡量model是否能distinguish plausible和physically impossible events
- PAI-Bench [346] (2025)：Quality/Domain/Control Fidelity

#### Domain-specific World Modeling

**Embodied AI and Robotics**：
- RLBench [359] (2019)：controlled manipulation，Success Rate
- CALVIN [315] (2021)：long-horizon + multi-step，MTLC Success Rate
- LIBERO [361] (2023)：FWT (Forward Transfer), AUC, Success Rate
- DROID [362] (2024)：diverse real-world interaction trajectories
- AgiBot World [363] (2025)：closed-loop task completion
- WoWBench [233] (2025)：planning ability + physical constraint adherence + instruction following

**Autonomous Driving**：
- KITTI [376] (2012), Waymo [378] (2019)：大规模real-world video
- nuScenes [377] (2019)：L2 Error, Collision Rate
- OpenDV-2K [379] (2024)：complex traffic scenes
- NAVSIM [313] (2024)：PDMS
- Act-bench [383] (2024)：Instruction-Execution Consistency + ADE/FDE
- DrivingDojo [381] (2024)：FID/FVD + instruction-following errors

**Interactive Environments and Gaming**：
- ALE [401] (2013)：Game Score
- DMC [402] (2018)：Raw Return
- Crafter [403] (2021)：procedural generation，systematic generalization
- Source [221] (2025)：high-fidelity visual data
- LOOPNAV [404] (2024)：spatial memory + revisit consistency
- Matrix-Game-MC [224] (2025)：composite GameWorld Score (Visual Quality + Physical Plausibility + Task Performance)

参考链接：
- WorldScore: https://arxiv.org/abs/2504.00983
- IntPhys 2: https://arxiv.org/abs/2506.09849
- Physics-IQ: https://arxiv.org/abs/2501.09038
- VBench-2.0: https://arxiv.org/abs/2503.21755

---

## 5. Future Directions深度解读

paper提出三个"Re-"方向，构成完整的roadmap：**Re-grounding**（强化knowledge foundation）, **Re-evaluation**（改进评估）, **Re-scaling**（scaling laws）。

### 5.1 Re-grounding：强化知识基础

#### 5.1.1 拓展world knowledge范围

**Richer Physical Interactions**：当前benchmark和dataset过度强调clean dynamics（rigid objects, simple motion），underrepresent那些依赖subtle interaction effects的场景：
- Contact-rich manipulation
- Deformable materials
- Surface-dependent motion

例如，robotic manipulation的成功取决于precise contact和friction；navigation reliability随surface conditions变化。

**Human-Centered Rules and Conventions**：人类环境中的behavior由social norms, conventions, shared intentions塑造。当前model常常无法capture这些principle如何modify action-effect relations。例如，"red light → stop"的correlation在temporary traffic control或human-directed override下不再成立。

#### 5.1.2 强化grounding的架构支持

**Geometry-aware Modeling**：当前VWM普遍缺乏explicit 3D structure representation，难以维持stable object identity, occlusion relations, spatial consistency。

paper提出两个互补方向：
1. **Explicitly modeling世界为time-varying 3D structure**：spatial layout在geometric primitives上直接evolve（例如4D Gaussian representations [208, 412]）
2. **Injecting geometry-aware constraints into现有架构**：multi-view consistency或camera-aware conditioning，无需full 3D reconstruction

**Neuro-symbolic Hybrid Modeling**：纯neural architecture在OOD（distribution shift, unseen intervention patterns）下generalize差；纯symbolic system（physics engines, rule-based planners）精确但缺乏flexibility。

Hybrid的优势：
1. 神经组件model perception和variability
2. Symbolic模块引入explicit physical或causal constraint
3. 可提取explicit physical或causal structure from visual data

例如，differentiable physical solver [90, 416] model dynamics，neural generator capture visual detail；rule-based causal planner guide action-effect reasoning，区分genuine intervention effects和spurious correlations。

参考链接：
- NewtonGen (differentiable physics): https://arxiv.org/abs/2509.21309
- WALL-E 2.0 (neuro-symbolic): https://arxiv.org/abs/2504.15785

### 5.2 Re-evaluation：面向versatile和reliable的评估

#### Judge Models和Execution-based Evaluation

**VWM Judge Models**：当前evaluation缺乏直接assess world modeling capability的holistic机制。paper提出训练dedicated judge models for VWMs，能evaluate predicted futures是否satisfy physical constraints并correctly respond to interaction conditions。这些judge models可进一步通过preference learning或reinforcement-based alignment refinement。

**Execution-based Evaluation**：将world model放入execution loop，agent用simulated rollouts规划act，task performance成为world modeling quality的直接indicator。当performance degrade或planning breakdown时，failures提供concrete evidence of model的physical或causal understanding哪里incomplete。这比static scoring提供更diagnostic的signal。

#### Complex Dynamics和Causal Interventions的Benchmarks

benchmark设计应包含：
- Contact-rich manipulation
- Deformable materials
- Friction-dependent motion

更重要的：**Causal Interventions under Controlled Conditions**——从同一initial context出发，vary一个action或environment condition，examine predicted futures是否在correct direction变化。

**Counterfactual Settings**特别有用：given a planted seed, model应在drought和adequate watering下generate distinct growth patterns。这直接assess model是否capture stable causal relationships，而非merely copy observed patterns。

### 5.3 Re-scaling：generalization和reasoning的scaling laws

**Empirical Observation**：增加model size主要improve visual fidelity [117, 233]，physical和causal knowledge的improvement在diverse settings下remain limited。

#### Pretraining Scaling：Toward Generalist VWMs

**目标**：在unified modeling interface下scale VWMs，让single model支持diverse world tasks和interaction settings，并potentially exhibit emergent capabilities：
- Cross-domain generalization
- Longer-horizon reasoning
- Improved robustness under novel interactions

**Scaling维度**：
1. Model capacity
2. Training data的breadth和structure
3. Objectives（应鼓励learning fundamental physical和causal relations，而非overfit superficial correlations）

**计算效率挑战**：visual data在space和time上highly redundant，naive scaling计算inefficient。设计更efficient spatio-temporal tokenizers和scalable conditioning methods至关重要 [417, 418]。

#### Inference-time Scaling：Reasoning Before Generation

**核心思想**：分配额外test-time compute用于better planning和causal reasoning，而非one-shot generate futures。

**Deliberation机制**：
- Proposing candidate outcomes
- Checking physical/causal constraints
- Iteratively refining rollout under intervention

这parallel to multimodal models [419, 420]中extra inference compute提升reliability的趋势。对VWMs，inference-time scaling对rare physical events, complex contact dynamics, counterfactual reasoning特别valuable，在这些场景single forward pass容易unstable。

参考链接：
- s1 (test-time scaling): https://arxiv.org/abs/2501.10499
- Cosmos (Foundation Model): https://arxiv.org/abs/2501.03575

---

## 6. 关键洞察与思考

### 6.1 VWM与Video Generation的本质区别

paper隐含的一个核心insight：**video generation models（如Sora, Veo, Kling）和VWM看似similar但目标different**。前者optimizing visual quality和short-term coherence；后者要求capture fundamental physical和causal principle以support reliable long-horizon prediction under interaction。

VWM的"world knowledge"必须包含：
1. **Spatio-temporal coherence**（视频生成也有）
2. **Physical dynamics**（视频生成部分有）
3. **Causal mechanisms**（视频生成普遍缺失）

当前的evaluation gap就源于此——大多数benchmark用video generation的metrics（FID, FVD, LPIPS）评估VWM，但这些metrics无法detect model是否学到了causal relation。

### 6.2 四大Architecture家族的Trade-offs

| Family | Visual Quality | Long-horizon | Physical Grounding | Causal Reasoning | Computational Cost |
|---|---|---|---|---|---|
| Sequential Generation | 中 | 好（但error accumulate） | 弱（data-driven） | 弱 | 中 |
| Diffusion-based | 高 | 中（block-wise局限） | 中（可inject geometry） | 中 | 高 |
| Embedding Prediction | N/A（无visual decode） | 好 | 中（encoder bound） | 中 | 低 |
| State Transition | 低（latent state） | 好 | 中 | 中 | 低 |

**核心tension**：visual fidelity ↔ physical/causal grounding ↔ computational efficiency。当前没有任何家族同时dominate所有维度。

### 6.3 Embedding Prediction的哲学意义

V-JEPA 2代表Yann LeCun的JEPA哲学：**真正的intelligence不需要generate pixels，只需要在abstract representation space中predict**。这与生成式AI主流方向形成对比。embedding prediction的优势是efficiency和planning适用性，limitation是缺乏interpretability和foundation model capacity bound。

### 6.4 Object-Centric Modeling的深层挑战

paper指出Object-Centric Modeling面临**Binding Problem**：在clutter, heavy occlusion, fine-grained texture下slot assignment ambiguous。这其实是认知科学中的经典问题——人类如何将连续sensory input分解为discrete objects？

当前方法依赖fixed number of slots，这在unconstrained natural videos中不切实际。未来需要dynamic slot allocation和robust binding mechanisms。SlotPi引入Hamiltonian structure是physics-aware的尝试，但仍局限在简单场景。

### 6.5 评估的Paradox

paper揭示一个paradox：**当前最好的VWM（按visual quality metrics）不一定是最reliable的world model**。一个能生成漂亮视频但无法理解causal consequence的model在embodied AI中毫无价值。

Judge Models和Execution-based Evaluation是paper提出的关键解决方案，但其实施难度大——需要训练专门的judge model或构建closed-loop evaluation infrastructure。IntPhys 2的Surprise Score（violation-of-expectation）是promising的early signal。

### 6.6 Scaling的Limitation

paper最practical的observation：**naive scaling只提升visual fidelity，不提升physical和causal knowledge**。这意味着简单扩大model size不会自动产生AGI-grade world model。

Re-scaling要求：
1. **Data structure scaling**：不只是更多video，而是涵盖diverse interaction patterns和long-horizon processes
2. **Objective scaling**：objective应鼓励learning fundamental relations，而非overfitting superficial correlations
3. **Inference-time scaling**：test-time reasoning比one-shot generation更适合rare events和counterfactuals

---

## 7. 论文的Limitations和我的Critique

paper本身有几个可以质疑的地方：

**1. Definition的边界**：定义中将VWM限定为"learns world knowledge from visual data"，但许多工作（如VLWM, F1）大量依赖language作为conditioning。这个边界在实践中blurry。

**2. Causal Mechanism的implementation gap**：paper反复强调causal mechanism重要性，但Section 3的方法分析显示，几乎没有architecture能explicitly model causal mechanism。Latent action learning是correlation的近似，不是causal。SlotPi的Hamiltonian structure是physics prior，不是causal discovery。

**3. Future Directions偏概念**：Re-grounding, Re-evaluation, Re-scaling的描述偏high-level，缺少具体technical roadmap。例如"neuro-symbolic hybrid modeling"如何与deep learning framework integrate？differentiable physics engine的computational cost？

**4. 评估体系仍不完整**：paper虽然指出visual quality metrics的局限，但提出的judge models和execution-based evaluation本身缺少具体protocol。Surprise Score作为violation-of-expectation的proxy很有趣，但如何scale到complex real-world scenarios？

**5. 漏掉的perspective**：paper未深入讨论active inference和predictive coding的neuroscience inspiration，这其实是world model概念的origin（Friston的工作）。FEP（Free Energy Principle）作为unifying framework值得讨论。

**6. Computational sustainability**：scale VWMs到general capability的compute需求未被讨论。当前largest VWMs已经消耗巨额算力，Re-scaling方向需要考虑energy efficiency和democratization。

---

## 8. 总结：这篇Survey的价值

这篇survey的最大价值在于：
1. **Vision-centric的unified framework**：将vision从input modality提升为shaping factor，这个conceptual shift为后续研究提供清晰lens
2. **四家族taxonomy**：将disconnected的sub-communities组织成coherent landscape，facilitate cross-paradigm comparison
3. **三层evaluation体系**：Visual Quality / Physical Plausibility / Task Performance的categorization揭示当前evaluation gap
4. **三大未来方向**：Re-grounding / Re-evaluation / Re-scaling为field提供roadmap

对researcher的practical guidance：
- 选architecture时根据task requirement权衡（visual fidelity? long-horizon? efficiency?）
- 评估VWM时必须超越FID/FVD，引入physical plausibility和task performance metrics
- 设计新method时考虑causal mechanism的explicit modeling，不要仅靠data归纳
- 关注inference-time scaling作为test-time reasoning的enabler

最终，paper揭示了一个deep question：**VWM的本质是generative modeling problem，还是world understanding problem？** 当前主流approach用generative modeling作为proxy for world understanding，但paper的analysis显示这个proxy可能misleading——能generate plausible futures不等于understand how world works。如何bridge这个gap是next-generation VWM的核心research question。

参考资源汇总：
- Survey主页: https://AIWorldLab.github.io/survey
- 关键recent papers:
  - V-JEPA 2: https://arxiv.org/abs/2506.09985
  - Cosmos: https://arxiv.org/abs/2501.03575
  - Genie 3: https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models
  - DINO-WM: https://arxiv.org/abs/2411.04983
  - WorldScore: https://arxiv.org/abs/2504.00983
  - IntPhys 2: https://arxiv.org/abs/2506.09849
